"""Authenticate to the operator's LinQ JupyterHub and save a private session."""

import argparse
from contextlib import contextmanager
from html.parser import HTMLParser
import ipaddress
import json
import os
from pathlib import Path
import re
import socket
import sys
import threading
import urllib.parse
import uuid

import requests


_DNS_LOCK = threading.RLock()
_USERNAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_SESSION_NAME = "linq_jupyter_session.json"
_CAPABILITIES_NAME = "linq_jupyter_capabilities.json"


class AccessConfigurationError(Exception):
    """A safe-to-report access configuration failure without sensitive values."""


class LoginPageParser(HTMLParser):
    """Extract the login form's CSRF value without retaining page content."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.xsrf = None

    def handle_starttag(self, tag, attrs):
        if tag.lower() != "input":
            return
        fields = dict(attrs)
        if fields.get("name") == "_xsrf" and isinstance(fields.get("value"), str):
            self.xsrf = fields["value"]


def _origin(url):
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise AccessConfigurationError("Invalid HTTP origin")
    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError as error:
        raise AccessConfigurationError("Invalid HTTP origin") from error
    return parsed.scheme.lower(), parsed.hostname.rstrip(".").lower(), port


def _validate_base_url(value):
    if not isinstance(value, str) or not value.strip():
        raise AccessConfigurationError("A LinQ base URL is required")
    parsed = urllib.parse.urlsplit(value.strip())
    _origin(value.strip())
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise AccessConfigurationError("The LinQ base URL must be an origin root")
    return urllib.parse.urlunsplit((parsed.scheme.lower(), parsed.netloc, "/", "", ""))


def _validate_websocket_origin(value, base_url):
    """Allow a canonical same-host Origin when a local proxy changes the port."""
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise AccessConfigurationError("Invalid WebSocket origin")
    value = value.strip()
    parsed = urllib.parse.urlsplit(value)
    try:
        scheme, hostname, _ = _origin(value)
        base_scheme, base_hostname, _ = _origin(base_url)
    except AccessConfigurationError as error:
        raise AccessConfigurationError("Invalid WebSocket origin") from error
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise AccessConfigurationError("The WebSocket origin must not include a path or query")
    if scheme != base_scheme or hostname != base_hostname:
        raise AccessConfigurationError("The WebSocket origin must use the configured LinQ scheme and hostname")
    return urllib.parse.urlunsplit((scheme, parsed.netloc, "", "", ""))


def _resolve_address(hostname, port, address=None):
    if address is not None:
        try:
            return str(ipaddress.ip_address(address))
        except ValueError as error:
            raise AccessConfigurationError("The optional address must be an IP literal") from error
    try:
        results = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
    except OSError as error:
        raise AccessConfigurationError("LinQ host could not be resolved") from error
    if not results:
        raise AccessConfigurationError("LinQ host could not be resolved")
    return str(results[0][4][0])


@contextmanager
def _pinned_resolution(hostname, address):
    """Pin only this hostname while requests run, then restore the resolver."""
    original = socket.getaddrinfo

    def resolve(host, port, *args, **kwargs):
        if isinstance(host, str) and host.rstrip(".").lower() == hostname.rstrip(".").lower():
            return original(address, port, *args, **kwargs)
        return original(host, port, *args, **kwargs)

    with _DNS_LOCK:
        socket.getaddrinfo = resolve
        try:
            yield
        finally:
            socket.getaddrinfo = original


def _same_origin_url(target, base):
    try:
        return _origin(target) == _origin(base)
    except AccessConfigurationError:
        return False


def request_same_origin(session, method, url, base_url, *, data=None, headers=None, timeout=(10, 30), max_redirects=8):
    """Issue a request and manually follow redirects only within the same origin."""
    current = url
    current_method = method.upper()
    current_data = data
    for redirect_count in range(max_redirects + 1):
        if not _same_origin_url(current, base_url):
            raise AccessConfigurationError("Request left the configured LinQ origin")
        try:
            response = session.request(current_method, current, data=current_data, headers=headers,
                                       timeout=timeout, allow_redirects=False)
        except requests.RequestException as error:
            raise AccessConfigurationError("LinQ request failed") from error
        if response.status_code not in {301, 302, 303, 307, 308}:
            return response, current
        location = response.headers.get("Location")
        response.close()
        if not location or redirect_count == max_redirects:
            raise AccessConfigurationError("LinQ redirect could not be followed")
        target = urllib.parse.urljoin(current, location)
        if not _same_origin_url(target, base_url):
            raise AccessConfigurationError("LinQ returned a cross-origin redirect")
        if response.status_code in {301, 302, 303} and current_method not in {"GET", "HEAD"}:
            current_method, current_data = "GET", None
        current = target
    raise AccessConfigurationError("LinQ returned too many redirects")


def _response_ok(response, expected, failure):
    if response.status_code not in expected:
        response.close()
        raise AccessConfigurationError(failure)


def _cookie_records(session):
    result = []
    for cookie in session.cookies:
        result.append({
            "name": cookie.name,
            "value": cookie.value,
            "domain": cookie.domain,
            "path": cookie.path,
            "secure": bool(cookie.secure),
            "expires": cookie.expires,
            "rest": cookie._rest,
        })
    if not result:
        raise AccessConfigurationError("LinQ login did not establish a session")
    return result


def _write_private_json(path, value):
    path = Path(path)
    if path.is_symlink():
        raise AccessConfigurationError("Refusing a symlink at the credential destination")
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    descriptor = None
    try:
        descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            descriptor = None
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        path.chmod(0o600)
        try:
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except OSError:
            pass
    except OSError as error:
        raise AccessConfigurationError("Could not save private LinQ session files") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _username_from_user_url(final_url, base_url):
    if not _same_origin_url(final_url, base_url):
        raise AccessConfigurationError("LinQ user redirect left the configured origin")
    path = urllib.parse.urlsplit(final_url).path
    match = re.fullmatch(r"/user/([A-Za-z0-9][A-Za-z0-9_.-]{0,127})(?:/.*)?/?", path)
    if not match:
        raise AccessConfigurationError("LinQ did not redirect to a user server")
    return match.group(1)


def configure(base_url, username, password, address=None, destination=Path(".secrets"), websocket_origin=None):
    """Authenticate, verify the user's API, and atomically save private client state."""
    base_url = _validate_base_url(base_url)
    websocket_origin = _validate_websocket_origin(websocket_origin, base_url)
    if not isinstance(username, str) or not _USERNAME.fullmatch(username):
        raise AccessConfigurationError("Invalid LinQ username")
    if not isinstance(password, str) or not password:
        raise AccessConfigurationError("A LinQ password is required")
    origin = urllib.parse.urlsplit(base_url)
    hostname = origin.hostname
    port = origin.port or (443 if origin.scheme == "https" else 80)
    resolved_address = _resolve_address(hostname, port, address)
    session = requests.Session()
    session.trust_env = False
    try:
        with _pinned_resolution(hostname, resolved_address):
            login_url = urllib.parse.urljoin(base_url, "hub/login")
            response, _ = request_same_origin(session, "GET", login_url, base_url,
                                              headers={"Referer": base_url})
            _response_ok(response, {200}, "LinQ login page was unavailable")
            parser = LoginPageParser()
            parser.feed(response.text)
            response.close()
            if not parser.xsrf:
                raise AccessConfigurationError("LinQ login page had no CSRF token")

            response, _ = request_same_origin(
                session, "POST", login_url, base_url,
                data={"username": username, "password": password, "_xsrf": parser.xsrf},
                headers={"Referer": login_url},
            )
            _response_ok(response, {200}, "LinQ login was not accepted")
            response.close()

            redirect_url = urllib.parse.urljoin(base_url, "hub/user-redirect/")
            response, final_url = request_same_origin(session, "GET", redirect_url, base_url,
                                                       headers={"Referer": base_url})
            _response_ok(response, {200}, "LinQ user server was unavailable")
            actual_username = _username_from_user_url(final_url, base_url)
            response.close()
            if actual_username != username:
                raise AccessConfigurationError("LinQ redirected to a different user server")

            server_base_url = urllib.parse.urljoin(base_url, "user/" + urllib.parse.quote(actual_username, safe="") + "/")
            status_url = urllib.parse.urljoin(server_base_url, "api/status")
            response, _ = request_same_origin(session, "GET", status_url, base_url,
                                              headers={"Referer": server_base_url})
            _response_ok(response, {200}, "LinQ user API status check failed")
            try:
                status = response.json()
            except (ValueError, requests.JSONDecodeError) as error:
                response.close()
                raise AccessConfigurationError("LinQ user API returned invalid status data") from error
            response.close()
            if not isinstance(status, dict):
                raise AccessConfigurationError("LinQ user API returned invalid status data")
            cookies = _cookie_records(session)
    finally:
        session.close()

    destination = Path(destination)
    if destination.is_symlink():
        raise AccessConfigurationError("Refusing a symlink at the private destination directory")
    destination.mkdir(parents=True, exist_ok=True, mode=0o700)
    session_state = {"base_url": base_url, "resolved_address": resolved_address, "cookies": cookies}
    if websocket_origin is not None:
        session_state["websocket_origin"] = websocket_origin
    capabilities = {"server_base_url": server_base_url}
    _write_private_json(destination / _SESSION_NAME, session_state)
    _write_private_json(destination / _CAPABILITIES_NAME, capabilities)
    return {"status": "configured", "authenticated": True, "api_status": 200}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Configure a private LinQ Jupyter session for this user.")
    parser.add_argument("--base-url", required=True, help="LinQ JupyterHub origin, for example http://hackathon.lin-q.ru/")
    parser.add_argument("--websocket-origin", help="Optional canonical same-host WebSocket Origin for a local port-forwarding proxy, e.g. http://hackathon.lin-q.ru/")
    parser.add_argument("--address", help="Optional VPN-resolved IP literal for the LinQ host")
    parser.add_argument("--username", help="LinQ username; prompted if omitted")
    args = parser.parse_args(argv)
    username = args.username if args.username is not None else input("LinQ username: ")
    from getpass import getpass
    password = getpass("LinQ password: ")
    try:
        configure(args.base_url, username, password, args.address, websocket_origin=args.websocket_origin)
        print("LinQ access configured. Session files were saved locally with private permissions.")
        return 0
    except Exception as error:
        print("LinQ access setup failed (" + type(error).__name__ + ").", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
