"""Small raw TCP forwarder for the LinQ Jupyter HTTP/WebSocket endpoint."""

import argparse
import os
from pathlib import Path
import select
import socket
import socketserver


TARGET = ("10.0.114.30", 80)
LISTEN = ("0.0.0.0", 18184)
STATE = Path("/run/secrets")


def target_reachable():
    try:
        with socket.create_connection(TARGET, timeout=2):
            return True
    except OSError:
        return False


def _process_alive(name):
    try:
        pid = int((STATE / name).read_text().strip())
        if pid <= 1:
            return False
        os.kill(pid, 0)
        # PID 1 is the image's idle process, so exited children may remain zombies.
        # Linux still accepts kill(pid, 0) for those; they must not signal readiness.
        process_stat = Path(f"/proc/{pid}/stat").read_text()
        state = process_stat.rsplit(")", 1)[1].split()[0]
        return state not in {"Z", "X", "x"}
    except (OSError, ValueError, IndexError):
        return False


def healthy():
    return ((STATE / "ready").is_file() and _process_alive("vpn.pid")
            and _process_alive("proxy.pid") and target_reachable())


class ForwardHandler(socketserver.BaseRequestHandler):
    def handle(self):
        try:
            upstream = socket.create_connection(TARGET, timeout=5)
        except OSError:
            return
        with upstream:
            upstream.settimeout(None)
            client = self.request
            client.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            upstream.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            peers = {client: upstream, upstream: client}
            active = set(peers)
            while active:
                try:
                    readable, _, _ = select.select(list(active), [], [], 60)
                except (OSError, ValueError):
                    return
                for source in readable:
                    destination = peers[source]
                    try:
                        data = source.recv(65536)
                        if data:
                            destination.sendall(data)
                        else:
                            active.discard(source)
                            try:
                                destination.shutdown(socket.SHUT_WR)
                            except OSError:
                                pass
                    except OSError:
                        return


class ForwardServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def handle_error(self, request, client_address):
        # Never write request payloads or connection details to the container log.
        return


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--health", action="store_true")
    args = parser.parse_args(argv)
    if args.check:
        return 0 if target_reachable() else 1
    if args.health:
        return 0 if healthy() else 1
    try:
        with ForwardServer(LISTEN, ForwardHandler) as server:
            server.serve_forever(poll_interval=0.5)
    except OSError:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
