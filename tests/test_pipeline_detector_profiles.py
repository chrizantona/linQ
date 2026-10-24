from contextlib import ExitStack
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

# The pipeline test mocks frame decoding, so it does not need Pillow installed.
try:
    import PIL  # noqa: F401
except ModuleNotFoundError:
    pil_stub = types.ModuleType('PIL')
    pil_stub.Image = object()
    sys.modules['PIL'] = pil_stub

from deploy import pipeline


class FixtureImage:
    width = 32
    height = 32

    def save(self, path, format=None, quality=None):
        Path(path).write_bytes(b'fixture image bytes')


class DetectorProfilePipelineTests(unittest.TestCase):
    def test_detector_profile_skips_encoder_but_keeps_detector_rules_and_buffers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / 'videos' / 'phone.mp4'
            video.parent.mkdir()
            video.write_bytes(b'fixture')
            output = root / 'run'
            output.mkdir()
            config = {
                'id': 'phone_camera', 'video': 'videos/phone.mp4', 'task_profile': 'phone_objects',
                'T': 3, 'V': 2, 'time_offset_s': 0,
                'detector': {'mode': 'fresh', 'callable': 'fixture:detect', 'options': {}},
                'classifier': None,
                'rules': [{'id': 'phone_present', 'kind': 'model_signal', 'source': 'detector',
                           'label': 'phone', 'on_threshold': 0.45, 'off_threshold': 0.27,
                           'activate_after_s': 0.6, 'clear_after_s': 0.6, 'max_evidence_gap_s': 0.75,
                           'semantics': 'visible object presence only'}],
                'quality': {'enabled': False},
            }
            detector = lambda image, context, options: {
                'actual_execution': True, 'mode': 'fresh_inference', 'detections': [],
                'signals': {'phone_present': 0.8}, 'provenance': {'fixture': True},
            }
            with ExitStack() as stack:
                stack.enter_context(patch.object(pipeline, 'open_encoder', side_effect=AssertionError('encoder must not be opened')))
                stack.enter_context(patch.object(pipeline, 'callback', return_value=(detector, {'fixture': True})))
                stack.enter_context(patch.object(pipeline, 'video_info', return_value={'fps': 2, 'frame_count': 2}))
                state = pipeline.camera_open(config, {'buffer': {'window_s': 4}}, root, output, stack)
                self.assertIsNone(state['runtime'])
                self.assertIsNone(state['head'])
                self.assertIsNone(state['classifier_provenance'])
                self.assertEqual(state['max_frames'], 3)
                self.assertEqual(state['classify'](state, {'frame_index': 0})['status'], 'disabled')
                events = {}
                first = pipeline.process_sample(state, FixtureImage(), 0, 0.0,
                                                {'tracker': {}}, None, events, lambda *_: None)
                second = pipeline.process_sample(state, FixtureImage(), 1, 0.6,
                                                 {'tracker': {}}, None, events, lambda *_: None)
                self.assertEqual(first['stages']['classifier']['status'], 'disabled')
                self.assertEqual(second['stages']['classifier']['status'], 'disabled')
                self.assertEqual(second['indicators'][0]['state'], 'active')
                self.assertTrue(events)
                event = next(iter(events.values()))
                self.assertEqual(event['rule_id'], 'phone_present')
                self.assertEqual(event['classifier_at_trigger']['backend'], None)

    def test_run_manifest_records_no_classifier_backend(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'videos').mkdir()
            (root / 'videos/phone.mp4').write_bytes(b'fixture')
            config_path = root / 'config.json'
            config_path.write_text(json.dumps({
                'root': '.', 'buffer': {'window_s': 4}, 'tracker': {},
                'cameras': [{'id': 'phone_camera', 'video': 'videos/phone.mp4', 'task_profile': 'phone_objects',
                             'T': 3, 'V': 1, 'detector': {'mode': 'fresh', 'callable': 'fixture:detect', 'options': {}},
                             'classifier': None, 'rules': [], 'quality': {'enabled': False}}],
            }))
            output = root / 'run'
            output.mkdir()
            detector = lambda image, context, options: {
                'actual_execution': True, 'mode': 'fresh_inference', 'detections': [],
                'signals': {'phone_present': 0.0}, 'provenance': {'fixture': True},
            }
            with ExitStack() as stack:
                stack.enter_context(patch.object(pipeline, 'open_encoder', side_effect=AssertionError('encoder must not be opened')))
                stack.enter_context(patch.object(pipeline, 'callback', return_value=(detector, {'fixture': True})))
                stack.enter_context(patch.object(pipeline, 'video_info', return_value={'fps': 1, 'frame_count': 1}))
                stack.enter_context(patch.object(pipeline, 'decode', return_value=iter([(0, FixtureImage())])))
                manifest = pipeline.run(config_path, output)
            self.assertIsNone(manifest['cameras'][0]['classifier_backend'])
            self.assertIsNone(manifest['cameras'][0]['classifier_provenance'])
            self.assertFalse(manifest['cameras'][0]['detector_TPU_execution_verified'])
            self.assertEqual(manifest['cameras'][0]['detector_TPU_frames'], 1)
            self.assertEqual(manifest['cameras'][0]['verified_detector_TPU_frames'], 0)
            self.assertFalse(manifest['cameras'][0]['classifier_TPU_execution_verified'])
            self.assertNotIn('TPU_execution_verified', manifest['cameras'][0])

    def test_tpu_manifest_claim_requires_validated_detector_receipts(self):
        self.assertFalse(pipeline.verified_tpu_receipt({'TPU_execution_verified': True}))
        self.assertFalse(pipeline.verified_tpu_receipt({'TPU_execution_verified': True, 'receipt': {'status': 'pending'}}))
        valid = {'TPU_execution_verified': True, 'receipt': {'status': 'received_and_validated'}}
        self.assertTrue(pipeline.verified_tpu_receipt(valid))
        self.assertFalse(pipeline.verified_tpu_receipt({**valid, 'tiled_receipts': [{'receipt': {'status': 'unverified'}}]}))
        self.assertTrue(pipeline.verified_tpu_receipt({**valid, 'tiled_receipts': [{'receipt': {'status': 'received_and_validated'}}]}))


if __name__ == '__main__':
    unittest.main()
