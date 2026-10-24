import json
import importlib.util
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch

from scripts import connect_linq_docker as helper

_spec = importlib.util.spec_from_file_location(
    "linq_tcp_forward", Path(__file__).resolve().parents[1] / "deploy/docker-linq/tcp_forward.py"
)
tcp_forward = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tcp_forward)


class DockerConnectTests(unittest.TestCase):
    def test_auth_file_is_private_and_parsed_as_two_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            auth = Path(directory) / "auth.txt"
            auth.write_text("demo-user\ndemo-password\n")
            auth.chmod(0o600)
            self.assertEqual(helper.read_credentials(auth), ("demo-user", "demo-password"))
            self.assertEqual(stat.S_IMODE(auth.stat().st_mode), 0o600)
            auth.chmod(0o644)
            with self.assertRaisesRegex(helper.ConnectError, "permissions"):
                helper.read_credentials(auth)

    def test_secret_is_sent_only_in_docker_exec_stdin(self):
        captured = {}

        def fake_docker(arguments, **kwargs):
            captured["arguments"] = arguments
            captured["input"] = kwargs["input_bytes"]

        with patch.object(helper, "_docker", side_effect=fake_docker):
            helper._deliver_credentials("unit-user", "unit-password")
        self.assertNotIn("unit-user", " ".join(captured["arguments"]))
        self.assertNotIn("unit-password", " ".join(captured["arguments"]))
        self.assertEqual(captured["input"], b"unit-user\nunit-password\n")
        self.assertIn("/run/secrets/vpn-auth", " ".join(captured["arguments"]))

    def test_container_args_are_isolated_readonly_and_loopback_only(self):
        args = helper._docker_run_arguments(Path("/tmp/client.ovpn"), "test/vpn:local")
        joined = " ".join(args)
        self.assertIn("--cap-add NET_ADMIN", joined)
        self.assertIn("/dev/net/tun:/dev/net/tun:rwm", joined)
        self.assertIn("--memory 256m", joined)
        self.assertIn("--cpus 0.5", joined)
        self.assertIn("--dns 10.0.113.1", joined)
        self.assertIn("--route", Path(helper.DOCKER_DIR / "connect-linq.sh").read_text())
        self.assertIn("127.0.0.1:18184:18184/tcp", joined)
        self.assertIn("dst=/tmp/client.ovpn,readonly", joined)
        self.assertIn("dst=/tmp/connect-linq.sh,readonly", joined)
        self.assertIn("dst=/tmp/tcp_forward.py,readonly", joined)
        self.assertNotIn("--privileged", args)
        self.assertNotIn("--network", args)
        self.assertNotIn("mosprom-linq-access", joined)
        self.assertIn("--entrypoint", args)
        self.assertEqual(args[-3:], ["sleep", "test/vpn:local", "infinity"])
        self.assertNotIn("--cap-drop", args)
        tunnel_script = Path(helper.DOCKER_DIR / "connect-linq.sh").read_text()
        self.assertIn("--daemon --log", tunnel_script)
        self.assertIn("--writepid", tunnel_script)
        self.assertNotIn("--verb", tunnel_script)

    def test_health_detection_distinguishes_startup_failure_and_ready(self):
        def info(status, health=None):
            state = {"Status": status}
            if health is not None:
                state["Health"] = {"Status": health}
            return {"State": state}

        self.assertEqual(helper._health(info("running", "starting")), "starting")
        self.assertEqual(helper._health(info("running", "healthy")), "healthy")
        self.assertEqual(helper._health(info("running", "unhealthy")), "unhealthy")
        self.assertEqual(helper._health(info("exited")), "stopped")

    def test_health_falls_back_to_private_readiness_check_for_custom_images(self):
        info = {"State": {"Status": "running"}}
        healthy = type("Result", (), {"returncode": 0})()
        with patch.object(helper, "_docker", return_value=healthy) as docker_call:
            self.assertEqual(helper._container_health(info), "healthy")
        self.assertEqual(docker_call.call_args.args[0][:3], ["exec", helper.CONTAINER, "python3"])

    def test_forward_health_requires_live_processes_and_target_reachability(self):
        with patch.object(tcp_forward.Path, "is_file", return_value=True), \
             patch.object(tcp_forward, "_process_alive", return_value=True), \
             patch.object(tcp_forward, "target_reachable", return_value=True):
            self.assertTrue(tcp_forward.healthy())
        with patch.object(tcp_forward.Path, "is_file", return_value=True), \
             patch.object(tcp_forward, "_process_alive", return_value=True), \
             patch.object(tcp_forward, "target_reachable", return_value=False):
            self.assertFalse(tcp_forward.healthy())

    def test_status_refuses_a_differently_owned_container(self):
        wrong = {"Config": {"Labels": {helper.LABEL: "someone-else"}}, "State": {"Status": "running"}}
        with patch.object(helper, "_docker", return_value=type("Result", (), {
            "returncode": 0, "stdout": json.dumps([wrong]).encode(), "stderr": b""
        })()):
            with self.assertRaisesRegex(helper.ConnectError, "reserved helper name"):
                helper.status()

    def test_exited_zombie_process_does_not_report_ready(self):
        with patch.object(tcp_forward.Path, "read_text", side_effect=["42", "42 (openvpn) Z 1 0"]), \
             patch.object(tcp_forward.os, "kill"):
            self.assertFalse(tcp_forward._process_alive("vpn.pid"))
        with patch.object(tcp_forward.Path, "read_text", side_effect=["42", "42 (openvpn) S 1 0"]), \
             patch.object(tcp_forward.os, "kill"):
            self.assertTrue(tcp_forward._process_alive("vpn.pid"))


if __name__ == "__main__":
    unittest.main()
