import argparse
import base64
import datetime
import json
import os
from pathlib import Path
import socket
import time
import urllib.parse
import uuid

import requests


def _websocket_origin(value, base_url):
    """Validate and normalize a same-host origin; port may differ for local forwarding."""
    parsed = urllib.parse.urlsplit(value)
    base = urllib.parse.urlsplit(base_url)
    if (parsed.scheme not in {"http", "https"} or parsed.username or parsed.password
            or parsed.path not in {"", "/"} or parsed.query or parsed.fragment):
        raise ValueError("Configured WebSocket origin must be an HTTP(S) origin without a path")
    try:
        parsed_port = parsed.port or (443 if parsed.scheme == "https" else 80)
        _ = base.port  # Validate a malformed port in the configured base URL as well.
    except ValueError as error:
        raise ValueError("Configured WebSocket origin has an invalid port") from error
    if (parsed.scheme.lower() != base.scheme.lower()
            or not parsed.hostname or parsed.hostname.rstrip(".").lower() != (base.hostname or "").rstrip(".").lower()):
        raise ValueError("Configured WebSocket origin must use the LinQ hostname and scheme")
    host = parsed.hostname.lower()
    if ":" in host:
        host = "[" + host + "]"
    default = 443 if parsed.scheme.lower() == "https" else 80
    netloc = host if parsed_port == default else host + ":" + str(parsed_port)
    return urllib.parse.urlunsplit((parsed.scheme.lower(), netloc, "", "", ""))


def connect(session_file=".secrets/linq_jupyter_session.json", capabilities_file=".secrets/linq_jupyter_capabilities.json"):
    state = json.loads(Path(session_file).read_text())
    capabilities = json.loads(Path(capabilities_file).read_text())
    resolver = socket.getaddrinfo
    hostname = urllib.parse.urlsplit(state["base_url"]).hostname
    address = state["resolved_address"]
    socket.getaddrinfo = lambda host, *args, **kwargs: resolver(address if host == hostname else host, *args, **kwargs)
    session = requests.Session()
    session.trust_env = False
    for cookie in state["cookies"]:
        session.cookies.set(cookie["name"], cookie["value"], domain=cookie["domain"], path=cookie["path"], secure=cookie["secure"], expires=cookie["expires"], rest=cookie.get("rest", {}))
    websocket_origin = state.get("websocket_origin")
    origin = _websocket_origin(websocket_origin, state["base_url"]) if websocket_origin else _websocket_origin(state["base_url"], state["base_url"])
    return {"session": session, "base_url": capabilities["server_base_url"], "origin": origin, "hostname": hostname, "session_file": str(session_file), "state": state}


def save_session(client):
    state = {**client["state"], "cookies": [{"name": cookie.name, "value": cookie.value, "domain": cookie.domain, "path": cookie.path, "secure": cookie.secure, "expires": cookie.expires, "rest": cookie._rest} for cookie in client["session"].cookies]}
    path = Path(client["session_file"])
    temporary = path.with_name(path.name + "." + str(uuid.uuid4()) + ".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(state, stream, indent=2)
    os.replace(temporary, path)
    path.chmod(0o600)


def request(client, method, relative_path, timeout=30, **kwargs):
    url = urllib.parse.urljoin(client["base_url"], relative_path)
    if not url.startswith(client["base_url"]):
        raise ValueError("Requests must stay within the authenticated user's server")
    cookies = [cookie for cookie in client["session"].cookies if cookie.name == "_xsrf" and urllib.parse.urlsplit(url).path.startswith(cookie.path)]
    headers = {"Referer": client["base_url"]}
    if cookies:
        headers["X-XSRFToken"] = max(cookies, key=lambda cookie: len(cookie.path)).value
    response = client["session"].request(method, url, headers=headers, timeout=timeout, **kwargs)
    save_session(client)
    if response.status_code >= 400:
        raise RuntimeError("Jupyter API returned HTTP " + str(response.status_code))
    return response


def contents(client, path="", include_content=True):
    path = safe_path(path, allow_empty=True)
    return request(client, "GET", "api/contents/" + urllib.parse.quote(path, safe="/"), params={"content": int(include_content)}).json()


def safe_path(path, allow_empty=False):
    path = str(path)
    if path.startswith("/") or ".." in Path(path).parts or not path and not allow_empty:
        raise ValueError("A relative path inside the user's Jupyter home is required")
    return path


def ensure_directory(client, remote_path):
    parts = Path(safe_path(remote_path)).parts
    for index in range(1, len(parts) + 1):
        path = "/".join(parts[:index])
        try:
            entry = contents(client, path, include_content=False)
            if entry["type"] != "directory":
                raise ValueError("Remote path is not a directory")
        except RuntimeError as error:
            if "HTTP 404" not in str(error):
                raise
            request(client, "PUT", "api/contents/" + urllib.parse.quote(path, safe="/"), json={"type": "directory"})


def upload(client, local_path, remote_path):
    local_path = Path(local_path)
    remote_path = safe_path(remote_path)
    if str(Path(remote_path).parent) != ".":
        ensure_directory(client, str(Path(remote_path).parent))
    if local_path.suffix == ".ipynb":
        payload = {"type": "notebook", "format": "json", "content": json.loads(local_path.read_text())}
    else:
        payload = {"type": "file", "format": "base64", "content": base64.b64encode(local_path.read_bytes()).decode("ascii")}
    return request(client, "PUT", "api/contents/" + urllib.parse.quote(remote_path, safe="/"), json=payload, timeout=180).json()


def execute(client, code, timeout=120, kernel_name="python3"):
    import websocket
    created = request(client, "POST", "api/kernels", json={"name": kernel_name}).json()
    kernel_id = created["id"]
    result = {"kernel_id": kernel_id, "stdout": "", "stderr": "", "errors": [], "results": []}
    ws = None
    try:
        channel_url = urllib.parse.urljoin(client["base_url"], "api/kernels/" + kernel_id + "/channels")
        prepared = client["session"].prepare_request(requests.Request("GET", channel_url))
        ws_url = channel_url.replace("http://", "ws://", 1).replace("https://", "wss://", 1) + "?session_id=" + str(uuid.uuid4())
        ws = websocket.create_connection(ws_url, cookie=prepared.headers.get("Cookie", ""), origin=client["origin"], timeout=20, http_no_proxy=[client["hostname"]])
        msg_id = str(uuid.uuid4())
        message = {"header": {"msg_id": msg_id, "username": "linq_client", "session": str(uuid.uuid4()), "date": datetime.datetime.now(datetime.timezone.utc).isoformat(), "msg_type": "execute_request", "version": "5.3"}, "parent_header": {}, "metadata": {}, "content": {"code": code, "silent": False, "store_history": False, "user_expressions": {}, "allow_stdin": False, "stop_on_error": True}, "channel": "shell", "buffers": []}
        ws.send(json.dumps(message))
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            ws.settimeout(min(5, max(.1, deadline - time.monotonic())))
            try:
                incoming = json.loads(ws.recv())
            except websocket.WebSocketTimeoutException:
                continue
            if incoming.get("parent_header", {}).get("msg_id") != msg_id:
                continue
            kind = incoming.get("msg_type", incoming.get("header", {}).get("msg_type"))
            content = incoming.get("content", {})
            if kind == "stream":
                key = "stderr" if content.get("name") == "stderr" else "stdout"
                result[key] += content.get("text", "")
            elif kind == "error":
                result["errors"].append(content)
            elif kind in ("execute_result", "display_data"):
                result["results"].append(content)
            elif kind == "status" and content.get("execution_state") == "idle":
                result["status"] = "error" if result["errors"] else "completed"
                break
        else:
            result["status"] = "timeout"
    finally:
        if ws is not None:
            ws.close()
        result["own_kernel_delete_http_status"] = request(client, "DELETE", "api/kernels/" + kernel_id).status_code
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-file", default=".secrets/linq_jupyter_session.json")
    parser.add_argument("--capabilities-file", default=".secrets/linq_jupyter_capabilities.json")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status")
    listing = commands.add_parser("list")
    listing.add_argument("--remote", default="")
    uploading = commands.add_parser("upload")
    uploading.add_argument("--local", required=True)
    uploading.add_argument("--remote", required=True)
    executing = commands.add_parser("execute")
    executing.add_argument("--code-file", required=True)
    executing.add_argument("--timeout", type=int, default=120)
    executing.add_argument("--output", required=True)
    args = parser.parse_args()
    client = connect(args.session_file, args.capabilities_file)
    if args.command == "status":
        response = request(client, "GET", "api/status")
        print(json.dumps({"http_status": response.status_code, "authenticated": True}))
    elif args.command == "list":
        result = contents(client, args.remote)
        print(json.dumps({"type": result["type"], "items": [{"name": entry["name"], "type": entry["type"]} for entry in result.get("content", [])]}))
    elif args.command == "upload":
        result = upload(client, args.local, args.remote)
        print(json.dumps({"status": "uploaded", "type": result["type"]}))
    else:
        result = execute(client, Path(args.code_file).read_text(), args.timeout)
        path = Path(args.output)
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, "w") as stream:
            json.dump(result, stream, indent=2)
        path.chmod(0o600)
        print(json.dumps({"status": result["status"], "error_count": len(result["errors"]), "own_kernel_delete_http_status": result["own_kernel_delete_http_status"]}))


if __name__ == "__main__":
    main()
