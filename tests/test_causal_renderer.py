import json
from pathlib import Path
import tempfile
import unittest

from scripts.render_causal_demo import create, result_rows, table_html, vlm_events


class CausalRendererTests(unittest.TestCase):
    def test_unknown_and_errors_remain_unknown_without_fabricated_explanation(self):
        rows = result_rows([{'event_id': 'error', 'camera_id': 'one', 'route_scope': 'open', 'trigger_timestamp_s': 0.7, 'request_reasons': ['periodic'], 'vlm': {'status': 'error', 'error': 'remote unavailable', 'actual_execution': False}}, {'event_id': 'unknown', 'camera_id': 'two', 'route_scope': '05', 'trigger_timestamp_s': 1.0, 'vlm': {'status': 'unknown', 'response': {'unsafe': None, 'reason': '', 'evidence': []}}}])
        self.assertIsNone(rows[0]['unsafe'])
        self.assertIsNone(rows[0]['reason'])
        self.assertFalse(rows[0]['http_response_received'])
        self.assertEqual(rows[0]['error'], 'remote unavailable')
        self.assertEqual(rows[1]['reason'], '')
        self.assertIsNone(rows[1]['unsafe'])
        self.assertIn('unknown', table_html(rows))

    def test_durable_event_record_keeps_actual_final_vlm_response(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / 'events' / 'stable-id'
            directory.mkdir(parents=True)
            event = {'event_id': 'stable-id', 'camera_id': 'one', 'trigger_timestamp_s': 0.7, 'route_scope': 'open', 'raw_frames': [{'timestamp_s': 0}, {'timestamp_s': 0.3}, {'timestamp_s': 0.7}], 'vlm': {'status': 'rejected', 'http_response_received': True, 'model_execution_verified': False, 'response': {'unsafe': False, 'reason': '<visible evidence>', 'evidence': ['ordered raw frames']}}}
            (directory / 'event.json').write_text(json.dumps(event))
            (root / 'events.jsonl').write_text(json.dumps({'event_id': 'stable-id', 'vlm': {'status': 'queued'}})+'\n')
            events = vlm_events(root)
            row = result_rows(events)[0]
            self.assertEqual(row['status'], 'rejected')
            self.assertFalse(row['unsafe'])
            self.assertFalse(row['model_execution_verified'])
            self.assertTrue(row['http_response_received'])
            self.assertEqual(row['raw_frame_timestamps_s'], [0, 0.3, 0.7])
            self.assertIn('&lt;visible evidence&gt;', table_html([row]))

    def test_recorded_replay_cannot_be_rendered_as_fresh_inference(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'manifest.json').write_text(json.dumps({'mode': 'recorded_inference_replay'}))
            config = root / 'config.json'
            config.write_text(json.dumps({'root': '.', 'cameras': []}))
            with self.assertRaisesRegex(ValueError, 'causal fresh-inference'):
                create(root, config, root / 'view')

if __name__ == '__main__':
    unittest.main()
