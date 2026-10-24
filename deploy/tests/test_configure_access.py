import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from deploy import configure_access


class Cookie:
    name = "_xsrf"
    value = "cookie-xsrf"
    domain = "hackathon.lin-q.ru"
    path = "/"
    secure = False
    expires = None
    _rest = {}


class Response:
    def __init__(self, status=200, text="", url="", headers=None, payload=None):
        self.status_code = status
        self.text = text
        self.url = url
        self.headers = headers or {}
        self.payload = payload
        self.closed = False

    def close(self):
        self.closed = True

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, responses):
        self.trust_env = True
        self.cookies = [Cookie()]
        self.responses = list(responses)
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        response = self.responses.pop(0)
        response.url = url if not response.url else response.url
        return response

    def close(self):
        pass


class ConfigureAccessTests(unittest.TestCase):
    def test_parser_extracts_only_xsrf_value(self):
        parser = configure_access.LoginPageParser()
        parser.feed('<html><input type="hidden" name="_xsrf" value="token-123"></html>')
        self.assertEqual(parser.xsrf, "token-123")
        parser = configure_access.LoginPageParser()
        parser.feed('<input name="username" value="not-xsrf">')
        self.assertIsNone(parser.xsrf)

    def test_same_origin_redirect_is_followed_and_cross_origin_is_refused(self):
        session = FakeSession([
            Response(status=302, headers={"Location": "/hub/next"}),
            Response(status=200),
        ])
        response, final_url = configure_access.request_same_origin(
            session, "GET", "https://hub.example/hub/login", "https://hub.example/"
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(final_url, "https://hub.example/hub/next")
        self.assertTrue(session.responses == [])

        session = FakeSession([Response(status=302, headers={"Location": "https://evil.example/collect"})])
        with self.assertRaises(configure_access.AccessConfigurationError):
            configure_access.request_same_origin(
                session, "POST", "https://hub.example/hub/login", "https://hub.example/",
                data={"password": "private"},
            )
        self.assertEqual(len(session.calls), 1)
        self.assertEqual(len(session.responses), 0)

    def test_configure_saves_private_client_state_after_api_check(self):
        session = FakeSession([
            Response(text='<input name="_xsrf" value="page-xsrf">'),
            Response(status=200),
            Response(status=302, headers={"Location": "/user/alex/"}),
            Response(status=200),
            Response(status=200, payload={"version": "test"}),
        ])
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / ".secrets"
            with patch.object(configure_access.requests, "Session", return_value=session), \
                 patch.object(configure_access, "_resolve_address", return_value="192.0.2.4"), \
                 patch.object(configure_access, "_pinned_resolution"):
                result = configure_access.configure(
                    "https://hub.example/", "alex", "secret-password", destination=target
                )

            self.assertEqual(result["status"], "configured")
            state_path = target / "linq_jupyter_session.json"
            caps_path = target / "linq_jupyter_capabilities.json"
            self.assertEqual(os.stat(state_path).st_mode & 0o777, 0o600)
            self.assertEqual(os.stat(caps_path).st_mode & 0o777, 0o600)
            state = json.loads(state_path.read_text())
            capabilities = json.loads(caps_path.read_text())
            self.assertEqual(state["resolved_address"], "192.0.2.4")
            self.assertEqual(state["cookies"][0]["value"], "cookie-xsrf")
            self.assertEqual(capabilities["server_base_url"], "https://hub.example/user/alex/")
            self.assertNotIn("secret-password", state_path.read_text())
            self.assertFalse(session.trust_env)
            post = next(call for call in session.calls if call[0] == "POST")
            self.assertEqual(post[2]["data"], {
                "username": "alex", "password": "secret-password", "_xsrf": "page-xsrf"
            })

    def test_optional_websocket_origin_is_same_host_and_survives_private_state_roundtrip(self):
        session = FakeSession([
            Response(text='<input name="_xsrf" value="page-xsrf">'),
            Response(status=200),
            Response(status=302, headers={"Location": "/user/alex/"}),
            Response(status=200),
            Response(status=200, payload={"version": "test"}),
        ])
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / ".secrets"
            with patch.object(configure_access.requests, "Session", return_value=session), \
                 patch.object(configure_access, "_resolve_address", return_value="127.0.0.1"), \
                 patch.object(configure_access, "_pinned_resolution"):
                configure_access.configure(
                    "http://hackathon.lin-q.ru:18184/", "alex", "private-password",
                    destination=target, websocket_origin="http://hackathon.lin-q.ru",
                )
            state = json.loads((target / "linq_jupyter_session.json").read_text())
            self.assertEqual(state["websocket_origin"], "http://hackathon.lin-q.ru")
            self.assertNotIn("private-password", json.dumps(state))

    def test_websocket_origin_rejects_different_host_scheme_or_path(self):
        base = "http://hackathon.lin-q.ru:18184/"
        for value in ("https://hackathon.lin-q.ru", "http://evil.example", "http://alex@hackathon.lin-q.ru",
                      "http://hackathon.lin-q.ru/user/alex", "http://hackathon.lin-q.ru/?x=1"):
            with self.subTest(origin=value), self.assertRaises(configure_access.AccessConfigurationError):
                configure_access._validate_websocket_origin(value, base)

    def test_failure_messages_do_not_include_auth_values(self):
        session = FakeSession([
            Response(text='<input name="_xsrf" value="csrf">'),
            Response(status=302, headers={"Location": "https://other.example/"}),
        ])
        with patch.object(configure_access.requests, "Session", return_value=session), \
             patch.object(configure_access, "_resolve_address", return_value="192.0.2.4"), \
             patch.object(configure_access, "_pinned_resolution"):
            with self.assertRaises(configure_access.AccessConfigurationError) as raised:
                configure_access.configure("https://hub.example/", "alex", "sensitive")
        self.assertNotIn("sensitive", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
