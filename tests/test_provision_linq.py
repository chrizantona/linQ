import ast
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout

from scripts.provision_linq import archive_files, collect_files, main, remote_apply_code, remote_setup_code, safe_relative, upload_archive


def make_tree(root):
    for name in ("config_single.json", "config_two.json"):
        (root / name).write_text(json.dumps({"cameras": [{
            "detector": {"options": {"models": [{"checkpoint": "models/phone.onnx"},
                                                   {"checkpoint": "models/cigarette.onnx"}],
                "linq_person": {"model": "rfdetr", "models_root": "models/linq"}}},
            "classifier": {"bundle": "models/convnext"},
        }]}))
    for name in (
        "scripts/bootstrap_linq_prototype.py", "scripts/linq_prototype.py",
        "scripts/prefetch_qwen_gguf.py", "scripts/render_causal_demo.py", "scripts/run_user_video.py",
        "deploy/requirements-runtime.txt", "deploy/requirements-web.txt", "deploy/vlm_host.example.json",
        "provenance/source.json", "deploy/demo_web.py", "demo/core.py", "service/index.html",
        "service/assets/a.svg", "service/assets/industrial/clip.mp4",
        "service/assets/samples/frame.jpg", "models/phone.onnx", "models/convnext/manifest.json",
        "models/convnext/encoder.onnx", "models/linq/decoder.py",
        "models/linq/rfdetr/report.json", "models/linq/rfdetr/program.tpu",
    ):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if name.endswith("report.json"):
            path.write_text(json.dumps({"models": [{"name": "rfdetr", "status": "compiled",
                                                    "tpu_program": {"path": "program.tpu"}}]}))
        elif name.endswith("manifest.json"):
            path.write_text(json.dumps({"files": [{"path": "encoder.onnx", "sha256": "a" * 64}]}))
        else:
            path.write_bytes(b"fixture:" + name.encode())
    (root / "deploy/tests").mkdir(parents=True)
    (root / "deploy/tests/test_private.py").write_text("private fixture")
    (root / "service/.secrets").mkdir(parents=True)
    (root / "service/.secrets/token").write_text("never upload")


class ProvisionLinQTests(unittest.TestCase):
    def test_upload_works_when_jupyter_rejects_hidden_contents_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "package.tar"
            archive.write_bytes(b"x" * 2048)
            digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            uploaded = []

            def jupyter_upload(client, local_path, remote_path):
                if any(part.startswith(".") for part in Path(remote_path).parts):
                    raise RuntimeError("Jupyter cannot create hidden contents paths")
                uploaded.append(remote_path)

            receipt = {"status": "completed", "stdout": json.dumps({
                "status": "uploaded", "archive_sha256": digest}), "errors": []}
            with patch("scripts.provision_linq.linq_client.upload", side_effect=jupyter_upload), \
                 patch("scripts.provision_linq.checked_execute", return_value=receipt):
                result = upload_archive({}, archive, [], digest, chunk_size=1024)
            self.assertEqual(result["status"], "uploaded")
            self.assertEqual(len(uploaded), 2)

    def remote_payload(self, home, files, suffix):
        archive_path = home / "upload.tar"
        entries, archive_sha, archive_bytes = archive_files(files, archive_path)
        data = archive_path.read_bytes()
        upload_name = "provision_" + suffix * 32
        parts_root = home / "work/mosprom_hackathon_linq" / upload_name / "parts"
        parts_root.mkdir(parents=True)
        part = parts_root / "part-000000"
        part.write_bytes(data)
        payload = {"upload_name": upload_name,
                   "parts": [{"name": part.name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}],
                   "entries": entries, "archive_bytes": archive_bytes, "archive_sha256": archive_sha}
        return payload

    def test_safe_relative_rejects_traversal_and_absolute_paths(self):
        for value in ("../outside", "/absolute/path", "models\\escape", ""):
            with self.subTest(value=value), self.assertRaises(ValueError):
                safe_relative(value)
        self.assertEqual(safe_relative("models/phone.onnx"), "models/phone.onnx")

    def test_allowlist_resolves_active_models_and_excludes_private_and_test_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_tree(root)
            (root / 'service/assets/demo-local').mkdir()
            (root / 'service/assets/demo-local/local.mp4').write_bytes(b'local UI video')
            (root / 'service/samples.local.json').write_text('[]')
            selected, missing = collect_files(root)
            self.assertEqual(missing, ["models/cigarette.onnx"])
            self.assertIn("models/phone.onnx", selected)
            self.assertIn("models/convnext/encoder.onnx", selected)
            self.assertIn("models/linq/rfdetr/program.tpu", selected)
            self.assertNotIn("deploy/tests/test_private.py", selected)
            self.assertNotIn("service/.secrets/token", selected)
            self.assertNotIn('service/samples.local.json', selected)
            self.assertNotIn('service/assets/demo-local/local.mp4', selected)
            self.assertFalse(any(path.startswith("service/assets/industrial/") or
                                 path.startswith("service/assets/samples/") for path in selected))
            (root / "models/cigarette.onnx").write_bytes(b"cigarette fixture")
            selected, missing = collect_files(root)
            self.assertEqual(missing, [])

    def test_allowlist_rejects_symlinked_runtime_files(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside:
            root = Path(directory)
            make_tree(root)
            link = root / "models/external.onnx"
            link.symlink_to(Path(outside) / "private")
            (Path(outside) / "private").write_text("private")
            (root / "config_single.json").write_text(json.dumps({"cameras": [{
                "detector": {"options": {"models": [{"checkpoint": "models/external.onnx"}]}},
                "classifier": {"bundle": "models/convnext"},
            }]}))
            with self.assertRaisesRegex(ValueError, "regular, non-symlink"):
                collect_files(root)

    def test_dry_run_does_not_connect_and_reports_missing_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_tree(root)
            output = io.StringIO()
            with patch("scripts.provision_linq.linq_client.connect", side_effect=AssertionError("network")):
                with patch("sys.stdout", output):
                    status = main(["--dry-run", "--root", str(root)])
            report = json.loads(output.getvalue())
            self.assertEqual(status, 2)
            self.assertIn("models/cigarette.onnx", report["missing"])
            self.assertFalse(any(item["path"].startswith("service/assets/industrial/") for item in report["files"]))

    def test_archive_manifest_matches_bytes_and_remote_code_parses(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "code.py"
            source.write_bytes(b"public source")
            archive = root / "package.tar"
            entries, archive_sha, archive_bytes = archive_files({"deploy/code.py": source}, archive)
            archive_value = archive.read_bytes()
            self.assertEqual(archive_sha, hashlib.sha256(archive_value).hexdigest())
            self.assertEqual(archive_bytes, len(archive_value))
            self.assertEqual(entries[0]["sha256"], hashlib.sha256(b"public source").hexdigest())
            with tarfile.open(fileobj=io.BytesIO(archive_value), mode="r:") as package:
                self.assertEqual(package.getnames(), ["deploy/code.py"])
            payload = {"upload_name": "provision_" + "1" * 32, "parts": [], "entries": entries,
                       "archive_bytes": archive_bytes, "archive_sha256": archive_sha}
            ast.parse(remote_apply_code(payload))

    def test_remote_apply_replaces_code_and_installs_model_without_touching_other_files(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            source = home / "source.py"
            source.write_bytes(b"new code")
            model = home / "phone.onnx"
            model.write_bytes(b"verified model")
            payload = self.remote_payload(home, {"deploy/app.py": source, "models/phone.onnx": model}, "2")
            remote_root = home / "work/mosprom_hackathon_linq"
            (remote_root / "deploy").mkdir(parents=True)
            (remote_root / "deploy/app.py").write_bytes(b"old code")
            (remote_root / ".runtime").mkdir()
            (remote_root / ".runtime/token").write_text("keep me")
            with __import__("unittest.mock").mock.patch.dict(os.environ, {"HOME": str(home)}):
                exec(remote_apply_code(payload), {})
            self.assertEqual((remote_root / "deploy/app.py").read_bytes(), b"new code")
            self.assertEqual((remote_root / "models/phone.onnx").read_bytes(), b"verified model")
            self.assertEqual((remote_root / ".runtime/token").read_text(), "keep me")
            self.assertFalse((remote_root / payload["upload_name"]).exists())

    def test_remote_model_conflict_fails_before_replacing_any_code(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            source = home / "source.py"
            source.write_bytes(b"new code")
            model = home / "phone.onnx"
            model.write_bytes(b"expected model")
            payload = self.remote_payload(home, {"deploy/app.py": source, "models/phone.onnx": model}, "3")
            remote_root = home / "work/mosprom_hackathon_linq"
            (remote_root / "deploy").mkdir(parents=True)
            (remote_root / "deploy/app.py").write_bytes(b"old code")
            (remote_root / "models").mkdir()
            (remote_root / "models/phone.onnx").write_bytes(b"different existing model")
            with __import__("unittest.mock").mock.patch.dict(os.environ, {"HOME": str(home)}):
                with self.assertRaisesRegex(ValueError, "Existing model differs"):
                    exec(remote_apply_code(payload), {})
            self.assertEqual((remote_root / "deploy/app.py").read_bytes(), b"old code")
            self.assertEqual((remote_root / "models/phone.onnx").read_bytes(), b"different existing model")

    def test_setup_exit_status_propagates_incomplete_remote_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_tree(root)
            (root / "models/cigarette.onnx").write_bytes(b"required model")
            output = io.StringIO()
            with patch("scripts.provision_linq.linq_client.connect", return_value={}), \
                 patch("scripts.provision_linq.upload_archive", return_value={"status": "uploaded"}), \
                 patch("scripts.provision_linq.remote_setup", return_value={
                     "status": "setup_incomplete",
                     "stages": [{"stage": "bootstrap", "status": "failed", "detail": "command_failed"}],
                 }), \
                 redirect_stdout(output):
                status = main(["--setup", "--root", str(root)])
            self.assertEqual(status, 1)
            self.assertEqual(json.loads(output.getvalue())["setup"]["status"], "setup_incomplete")

    def test_setup_discovers_python312_instead_of_reusing_kernel_interpreter(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            root = home / "work/mosprom_hackathon_linq"
            root.mkdir(parents=True)
            log = home / "invocations.txt"
            interpreter = home / "python3.12"
            interpreter.write_text(
                "#!/bin/sh\n"
                "if [ \"$1\" = \"-c\" ]; then echo 3.12; else echo \"$1\" >> \"" + str(log) + "\"; fi\n"
            )
            interpreter.chmod(0o755)
            output = io.StringIO()
            with patch("pathlib.Path.home", return_value=home), \
                 patch("shutil.which", return_value=str(interpreter)), \
                 redirect_stdout(output):
                exec(remote_setup_code(), {})
            receipt = json.loads(output.getvalue())
            self.assertEqual(receipt["status"], "setup_complete")
            self.assertEqual(receipt["python_version"], "3.12")
            calls = log.read_text().splitlines()
            self.assertEqual(calls, ["scripts/bootstrap_linq_prototype.py", "scripts/prefetch_qwen_gguf.py"])

    def test_setup_reports_bounded_generic_stage_failure_detail(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            root = home / "work/mosprom_hackathon_linq"
            root.mkdir(parents=True)
            log = home / "invocations.txt"
            interpreter = home / "python3.12"
            interpreter.write_text(
                "#!/bin/sh\n"
                "if [ \"$1\" = \"-c\" ]; then echo 3.12; else echo sensitive-output >&2; exit 17; fi\n"
            )
            interpreter.chmod(0o755)
            output = io.StringIO()
            with patch("pathlib.Path.home", return_value=home), \
                 patch("shutil.which", return_value=str(interpreter)), \
                 redirect_stdout(output):
                exec(remote_setup_code(), {})
            receipt = json.loads(output.getvalue())
            self.assertEqual(receipt["status"], "setup_incomplete")
            self.assertEqual(receipt["stages"], [{
                "stage": "bootstrap", "status": "failed", "returncode": 17, "detail": "command_failed"
            }])
            self.assertNotIn("sensitive-output", output.getvalue())


if __name__ == "__main__":
    unittest.main()
