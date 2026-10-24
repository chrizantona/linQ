import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from deploy import linq_client


class WebSocket:
    def __init__(self):
        self.message_id = None
        self.closed = False

    def send(self, value):
        self.message_id = json.loads(value)["header"]["msg_id"]

    def recv(self):
        return json.dumps({"parent_header": {"msg_id": self.message_id},
                           "msg_type": "status", "content": {"execution_state": "idle"}})

    def settimeout(self, _value):
        pass

    def close(self):
        self.closed = True


class LinQClientOriginTests(unittest.TestCase):
    def test_origin_override_keeps_forwarded_endpoint_and_authenticated_cookie(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            session_file = root / "session.json"
            capabilities_file = root / "capabilities.json"
            session_file.write_text(json.dumps({
                "base_url": "http://hackathon.lin-q.ru:18184/",
                "resolved_address": "127.0.0.1",
                "websocket_origin": "http://hackathon.lin-q.ru",
                "cookies": [{"name": "_xsrf", "value": "opaque-session", "domain": "hackathon.lin-q.ru",
                             "path": "/", "secure": False, "expires": None, "rest": {}}],
            }))
            capabilities_file.write_text(json.dumps({
                "server_base_url": "http://hackathon.lin-q.ru:18184/user/alex/",
            }))
            original_resolver = linq_client.socket.getaddrinfo
            try:
                client = linq_client.connect(session_file, capabilities_file)
            finally:
                linq_client.socket.getaddrinfo = original_resolver

            self.assertEqual(client["origin"], "http://hackathon.lin-q.ru")
            self.assertEqual(client["base_url"], "http://hackathon.lin-q.ru:18184/user/alex/")
            self.assertEqual(client["hostname"], "hackathon.lin-q.ru")
            ws = WebSocket()
            captured = {}

            def create_connection(url, **kwargs):
                captured["url"] = url
                captured.update(kwargs)
                return ws

            delete_response = SimpleNamespace(status_code=204)

            def request(_client, method, path, **_kwargs):
                if method == "POST":
                    return SimpleNamespace(json=lambda: {"id": "kernel-test"})
                self.assertEqual((method, path), ("DELETE", "api/kernels/kernel-test"))
                return delete_response

            with patch("websocket.create_connection", side_effect=create_connection), \
                 patch.object(linq_client, "request", side_effect=request):
                result = linq_client.execute(client, "pass", timeout=1)

            self.assertEqual(result["status"], "completed")
            self.assertTrue(ws.closed)
            self.assertIn("/user/alex/api/kernels/kernel-test/channels?", captured["url"])
            self.assertEqual(captured["origin"], "http://hackathon.lin-q.ru")
            self.assertIn("_xsrf=opaque-session", captured["cookie"])
            self.assertEqual(captured["http_no_proxy"], ["hackathon.lin-q.ru"])

    def test_client_rejects_unsafe_override(self):
        for origin in ("https://hackathon.lin-q.ru", "http://attacker.example", "http://hackathon.lin-q.ru/path"):
            with self.subTest(origin=origin), self.assertRaises(ValueError):
                linq_client._websocket_origin(origin, "http://hackathon.lin-q.ru:18184/")


if __name__ == "__main__":
    unittest.main()
