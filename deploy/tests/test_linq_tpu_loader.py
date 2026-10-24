import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from deploy.linq_tpu import load_person_detector


class LinQDetectorLoaderTests(unittest.TestCase):
    def _write_assets(self, root):
        model = root / "rfdetr"
        model.mkdir(parents=True)
        program = model / "program.tpu"
        program.write_bytes(b"compiled LinQ program fixture")
        decoder = root / "decoder.py"
        decoder.write_text(
            "def preprocess(rgb, model, height, width): return None, None\n"
            "def decode_rfdetr(*args): return [], []\n"
            "def decode_yolo(*args): return [], []\n"
        )
        sha = hashlib.sha256(program.read_bytes()).hexdigest()
        report = {
            "models": [{
                "name": "rfdetr",
                "status": "compiled",
                "tpu_program": {"path": "program.tpu", "sha256": sha},
                "input_names": ["images"],
                "input_shapes": {"images": [1, 3, 384, 384]},
                "host_decode": {"person_class_index": 1, "max_detections": 300},
            }]
        }
        (model / "report.json").write_text(json.dumps(report))
        return program, decoder, sha

    def test_loads_report_program_and_shared_decoder_from_repo_assets(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "models" / "linq"
            program, decoder, expected_sha = self._write_assets(root)

            loaded = load_person_detector("rfdetr", root, .4)

            self.assertEqual(loaded["program"], program.resolve())
            self.assertEqual(loaded["sha256"], expected_sha)
            self.assertEqual(loaded["shape"], [1, 3, 384, 384])
            self.assertEqual(loaded["input_name"], "images")
            self.assertEqual(loaded["confidence"], .4)
            self.assertEqual(loaded["decoder_sha256"], hashlib.sha256(decoder.read_bytes()).hexdigest())
            self.assertTrue(callable(loaded["decoder"].decode_rfdetr))

    def test_rejects_program_that_does_not_match_report_sha(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "models" / "linq"
            program, _, _ = self._write_assets(root)
            program.write_bytes(b"changed program")

            with self.assertRaisesRegex(ValueError, "recorded SHA256"):
                load_person_detector("rfdetr", root)

    def test_rejects_unknown_model_without_legacy_path_fallback(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "Supported LinQ"):
                load_person_detector("legacy_model", Path(folder))


if __name__ == "__main__":
    unittest.main()
