"""Run an isolated Docker OpenVPN client and loopback-only TCP forward for LinQ."""

import argparse
import getpass
import json
from pathlib import Path
import re
import socket
import stat
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
DOCKER_DIR = ROOT / "deploy" / "docker-linq"
IMAGE = "mosprom-hackathon-linq-vpn:local"
CONTAINER = "linq-safety-vpn"
LABEL = "org.opencontainers.image.title"
LABEL_VALUE = "mosprom-hackathon-linq-vpn"
HOST_PORT = 18184
CONNECT_TIMEOUT = 100


class ConnectError(Exception):
    """An error safe to print without exposing Docker output or credentials."""


def _docker(arguments, *, input_bytes=None, timeout=30, allow_failure=False):
    try:
        result = subprocess.run(
            ["docker", *arguments], input=input_bytes, capture_output=True,
            timeout=timeout, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ConnectError("Docker command could not be completed") from error
    if result.returncode and not allow_failure:
        raise ConnectError("Docker operation failed")
    return result


def read_credentials(auth_file=None, *, prompt=None):
    """Return username/password bytes without placing either in process arguments."""
    if auth_file is not None:
        path = Path(auth_file)
        try:
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                raise ConnectError("Authentication file must be an ordinary private file")
            if metadata.st_mode & 0o077:
                raise ConnectError("Authentication file permissions must be 600 or stricter")
            lines = path.read_text(encoding="utf-8").splitlines()
        except ConnectError:
            raise
        except (OSError, UnicodeError) as error:
            raise ConnectError("Authentication file could not be read") from error
        if len(lines) != 2:
            raise ConnectError("Authentication file must contain exactly two lines")
        username, password = lines
    else:
        prompt = prompt or getpass.getpass
        try:
            username = input("LinQ username: ")
            password = prompt("LinQ password: ")
        except (EOFError, KeyboardInterrupt) as error:
            raise ConnectError("Authentication input was cancelled") from error
    if not username or not password or any("\n" in item or "\r" in item or "\x00" in item for item in (username, password)):
        raise ConnectError("Authentication values are invalid")
    return username, password


def _inspect_owned():
    result = _docker(["container", "inspect", CONTAINER], allow_failure=True)
    if result.returncode:
        message = result.stderr.decode("utf-8", "replace").lower()
        if "no such object" in message or "no such container" in message:
            return None
        raise ConnectError("Docker container status is unavailable")
    try:
        info = json.loads(result.stdout)[0]
    except (ValueError, IndexError, TypeError) as error:
        raise ConnectError("Docker container status is invalid") from error
    labels = (info.get("Config") or {}).get("Labels") or {}
    if labels.get(LABEL) != LABEL_VALUE:
        raise ConnectError("A container with the reserved helper name already exists")
    return info


def _health(info):
    state = info.get("State") or {}
    if state.get("Status") in {"exited", "dead", "removing"}:
        return "stopped"
    health = (state.get("Health") or {}).get("Status")
    if health == "healthy":
        return "healthy"
    if health == "unhealthy":
        return "unhealthy"
    return "starting"


def _container_health(info):
    status = _health(info)
    if status != "starting":
        return status
    # Images supplied with --skip-build may not define a Docker HEALTHCHECK.
    # The forwarding helper checks the readiness marker, process IDs, and target TCP route.
    result = _docker([
        "exec", CONTAINER, "python3", "/tmp/tcp_forward.py", "--health",
    ], allow_failure=True)
    return "healthy" if result.returncode == 0 else "starting"


def _docker_run_arguments(profile, image=IMAGE):
    return [
        "run", "--detach", "--rm", "--name", CONTAINER,
        "--label", f"{LABEL}={LABEL_VALUE}",
        "--cap-add", "NET_ADMIN",
        "--device", "/dev/net/tun:/dev/net/tun:rwm",
        "--memory", "256m", "--cpus", "0.5", "--dns", "10.0.113.1",
        "--security-opt", "no-new-privileges:true",
        "--health-cmd", "python3 /tmp/tcp_forward.py --health",
        "--health-interval", "2s", "--health-timeout", "3s",
        "--health-start-period", "90s", "--health-retries", "2",
        "--tmpfs", "/run/secrets:rw,noexec,nosuid,nodev,size=1m,mode=0700",
        "--mount", f"type=bind,src={profile},dst=/tmp/client.ovpn,readonly",
        "--mount", f"type=bind,src={DOCKER_DIR / 'connect-linq.sh'},dst=/tmp/connect-linq.sh,readonly",
        "--mount", f"type=bind,src={DOCKER_DIR / 'tcp_forward.py'},dst=/tmp/tcp_forward.py,readonly",
        "--publish", f"127.0.0.1:{HOST_PORT}:18184/tcp",
        "--entrypoint", "sleep", image, "infinity",
    ]


def _deliver_credentials(username, password):
    payload = (username + "\n" + password + "\n").encode("utf-8")
    _docker(
        ["exec", "--interactive", CONTAINER, "/bin/sh", "-c", "umask 077; cat > /run/secrets/vpn-auth"],
        input_bytes=payload,
        timeout=20,
    )


def _start_tunnel_processes():
    _docker(["exec", CONTAINER, "/bin/sh", "/tmp/connect-linq.sh"], timeout=CONNECT_TIMEOUT)


def _build_image():
    _docker(["build", "--tag", IMAGE, "--file", str(DOCKER_DIR / "Dockerfile"), str(DOCKER_DIR)], timeout=600)


def _profile_path(value):
    path = Path(value)
    try:
        if path.is_symlink() or not path.is_file():
            raise ConnectError("VPN profile must be a regular file")
        resolved = path.resolve(strict=True)
        if "," in str(resolved):
            raise ConnectError("VPN profile path contains an unsupported character")
        return resolved
    except OSError as error:
        raise ConnectError("VPN profile could not be read") from error


def start(profile, auth_file=None, *, image=IMAGE, skip_build=False):
    current = _inspect_owned()
    if current is not None:
        raise ConnectError("The helper container already exists; check status or stop it first")
    profile = _profile_path(profile)
    username, password = read_credentials(auth_file)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,254}", image):
        raise ConnectError("Docker image name is invalid")
    if skip_build:
        _docker(["image", "inspect", image])
    else:
        if image != IMAGE:
            raise ConnectError("A custom Docker image requires --skip-build")
        _build_image()
    _docker(_docker_run_arguments(profile, image))
    try:
        _deliver_credentials(username, password)
        _start_tunnel_processes()
        deadline = time.monotonic() + CONNECT_TIMEOUT
        while time.monotonic() < deadline:
            info = _inspect_owned()
            if info is None:
                raise ConnectError("The VPN container stopped before connecting")
            status = _container_health(info)
            if status == "healthy":
                if not _local_forward_ready():
                    raise ConnectError("The local LinQ forward did not open")
                return {"status": "connected", "container": CONTAINER,
                        "forward": f"http://127.0.0.1:{HOST_PORT}", "vpn_authenticated": True}
            if status in {"unhealthy", "stopped"}:
                raise ConnectError("The VPN connection did not become ready")
            time.sleep(1)
        raise ConnectError("The VPN connection did not become ready before the timeout")
    except Exception:
        _remove_own_container()
        raise


def _local_forward_ready():
    try:
        with socket.create_connection(("127.0.0.1", HOST_PORT), timeout=1):
            return True
    except OSError:
        return False


def status():
    info = _inspect_owned()
    if info is None:
        return {"status": "stopped", "container": CONTAINER}
    state = _container_health(info)
    return {"status": state, "container": CONTAINER,
            "forward": f"http://127.0.0.1:{HOST_PORT}" if state == "healthy" else None}


def _remove_own_container():
    info = _inspect_owned()
    if info is not None:
        _docker(["container", "rm", "--force", CONTAINER], timeout=20)


def stop():
    info = _inspect_owned()
    if info is None:
        return {"status": "already_stopped", "container": CONTAINER}
    _docker(["container", "stop", "--time", "5", CONTAINER], timeout=15)
    # Containers are launched with --rm; tolerate a short cleanup race only.
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if _inspect_owned() is None:
            return {"status": "stopped", "container": CONTAINER}
        time.sleep(0.2)
    return {"status": "stop_requested", "container": CONTAINER}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the isolated LinQ VPN forwarder in Docker.")
    actions = parser.add_subparsers(dest="action", required=True)
    start_parser = actions.add_parser("start", help="Build and start the isolated VPN forwarder")
    start_parser.add_argument("--profile", required=True, help="Path to the authorized OpenVPN profile")
    start_parser.add_argument("--auth-file", help="Private file with username and password on separate lines")
    start_parser.add_argument("--image", default=IMAGE, help="Use a prebuilt image (with --skip-build)")
    start_parser.add_argument("--skip-build", action="store_true", help="Use --image without building the bundled image")
    actions.add_parser("status", help="Show only the owned helper container's health")
    actions.add_parser("stop", help="Stop and remove only this helper's container")
    args = parser.parse_args(argv)
    try:
        if args.action == "start":
            result = start(args.profile, args.auth_file, image=args.image, skip_build=args.skip_build)
        elif args.action == "status":
            result = status()
        else:
            result = stop()
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except ConnectError as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
