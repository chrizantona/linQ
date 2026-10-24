from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from PIL import Image
from fastapi.testclient import TestClient

from deploy import pipeline
import deploy.demo_web as demo_web
from deploy.tests.test_pipeline import make_state


class PipelineProgressTests(unittest.TestCase):
    def test_blocked_vlm_still_publishes_verified_candidate_through_demo_service(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            output, config = root / 'run', root / 'config.json'
            config.write_text(json.dumps({'root': str(root), 'cameras': [{'id': 'one'}], 'vlm': {
                'config': 'unused.json', 'scope': '03', 'prompt': 'phone action', 'max_frames': 3,
                'max_inflight': 1, 'max_queue': 0, 'periodic_seconds': 30,
                'scan_on_source_end': False, 'confirm_active_incidents': True}}))
            entered, release = threading.Event(), threading.Event()

            def open_camera(camera, settings, directory, destination, stack):
                state = make_state(destination)
                state['config'].update(V=100, rules=[{'id': 'phone_use_review', 'kind': 'model_signal',
                    'source': 'classifier', 'label': 'unsafe', 'activate_after_s': 0.3, 'clear_after_s': 0.6}])
                state.update(detector_provenance={'callable': 'test:detector', 'source_sha256': 'fixture'},
                             runtime=None, classifier_provenance=None, incident_model_provenance=None)
                state['source_sha256'] = 'd' * 64
                state['risk'] = 1.0
                state.pop('next_periodic_s', None)
                return state

            def remote(*arguments):
                entered.set()
                if not release.wait(5):
                    raise TimeoutError('test fixture was not released')
                return {'response': {'unsafe': True, 'reason': 'held until test release', 'evidence': []}}

            with patch.object(pipeline, 'camera_open', side_effect=open_camera), patch.object(
                    pipeline, 'sampled', side_effect=lambda state: (item for item in [
                        (0, 0.0, Image.new('RGB', (20, 20))), (1, 0.4, Image.new('RGB', (20, 20)))])), patch.object(
                    pipeline, 'call_vlm', side_effect=remote), ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(pipeline.run, config, output)
                try:
                    self.assertTrue(entered.wait(2))
                    deadline = time.monotonic() + 2
                    progress = {}
                    while time.monotonic() < deadline:
                        progress = json.loads((output / 'progress.json').read_text())
                        if progress.get('candidate_events'):
                            break
                        time.sleep(0.01)
                    self.assertEqual(len(progress['candidate_events']), 1)
                    summary = progress['candidate_events'][0]
                    self.assertEqual(summary['rule_id'], 'phone_use_review')
                    self.assertEqual(summary['event_trigger'], 'active_event')
                    self.assertEqual(summary['frame_index'], 1)
                    self.assertTrue(progress['vlm_pending'])
                    verification = progress['_candidate_publication_verification']
                    self.assertIsNotNone(verification['firstCandidatePublishedAt'])
                    self.assertGreaterEqual(verification['elapsedMs'], 0)
                    self.assertNotIn('raw_frames', summary)
                    rows = [json.loads(line) for line in (output / 'frames.jsonl').read_text().splitlines() if line.strip()]
                    job = {'id': 'a' * 32, 'sampleId': 'upload_' + 'a' * 32, 'task': 'phone_use',
                           'created_at': '2026-10-25T10:00:00+00:00', 'status': 'running',
                           'inputs': [{'camera': 'one', 'sha256': 'd' * 64}], 'events': []}
                    intermediate = demo_web.intermediate_result(rows, job, progress)
                    self.assertEqual(len(intermediate['events']), 1)
                    self.assertTrue(intermediate['events'][0]['provisional'])
                    self.assertEqual(intermediate['events'][0]['trigger_timestamp_s'], 0.4)
                    self.assertEqual(intermediate['result']['episode']['frames'], [])
                    self.assertEqual(intermediate['result']['episode']['eventId'], summary['event_id'])
                    forged = dict(progress)
                    forged['candidate_events'] = [{**summary, 'trigger_timestamp_s': summary['trigger_timestamp_s'] + 1}]
                    self.assertEqual(demo_web.intermediate_result(rows, job, forged)['events'], [])

                    metadata = root / 'artifacts/private/demo_web/jobs' / job['id'] / 'job.json'
                    demo_web.atomic_json(metadata, {**job, **intermediate})
                    with TestClient(demo_web.create_app(root)) as client:
                        surfaced = client.get('/api/events').json()['events']
                        self.assertEqual(len(surfaced), 1)
                        self.assertEqual(surfaced[0]['event_id'], summary['event_id'])
                        facade = client.get('/api/pipeline').json()['jobs'][0]
                        self.assertEqual(facade['candidateEventCount'], 1)
                        self.assertTrue(facade['candidateEvents'][0]['provisional'])
                finally:
                    release.set()
                self.assertEqual(future.result(timeout=3)['status'], 'completed')

    def test_actual_cv_records_are_published_while_vlm_is_still_pending(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            output = root / 'run'
            config = root / 'config.json'
            config.write_text(json.dumps({'root': str(root), 'cameras': [{'id': 'one'}], 'vlm': {
                'config': 'unused.json', 'scope': '05', 'prompt': 'test fixture',
                'max_frames': 3, 'max_inflight': 1, 'max_queue': 1,
                'periodic_seconds': 30, 'scan_on_source_end': True, 'coalesce_latest': True}}))
            release, entered = threading.Event(), threading.Event()

            def open_camera(camera, settings, directory, destination, stack):
                state = make_state(destination)
                state['config'].update(T=3, V=1, rules=[])
                state.pop('next_periodic_s')
                state['buffer_window_s'] = 4
                state.update(runtime={'backend': 'onnxruntime_cpu'}, detector_provenance={},
                             classifier_provenance={}, incident_model_provenance=None)
                return state

            def remote(*arguments):
                entered.set()
                if not release.wait(5):
                    raise TimeoutError('test fixture was not released')
                return {'response': {'unsafe': None}}

            with patch.object(pipeline, 'camera_open', side_effect=open_camera), patch.object(
                    pipeline, 'sampled', side_effect=lambda state: iter(
                        (index, float(index), Image.new('RGB', (20, 20))) for index in range(4))), patch.object(
                    pipeline, 'call_vlm', side_effect=remote), ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(pipeline.run, config, output)
                try:
                    self.assertTrue(entered.wait(2))
                    deadline = time.monotonic() + 2
                    progress = {}
                    while time.monotonic() < deadline:
                        progress = json.loads((output / 'progress.json').read_text())
                        if progress['cv_ready']:
                            break
                        time.sleep(0.01)
                    self.assertTrue(progress['cv_ready'])
                    self.assertTrue(progress['vlm_pending'])
                    self.assertEqual(progress['accepted_frame_count'], 4)
                    self.assertEqual(progress['vlm_pending_count'], 1)
                    self.assertEqual(progress['vlm_queued_count'], 1)
                    self.assertIsNotNone(progress['cv_completed_at'])
                    rows = [json.loads(line) for line in (output / 'frames.jsonl').read_text().splitlines()]
                    self.assertEqual(len(rows), 4)
                    self.assertTrue(all(row['stages']['detector']['actual_execution'] for row in rows))
                    self.assertFalse((output / 'manifest.json').exists())
                    self.assertFalse(future.done())
                finally:
                    release.set()
                self.assertEqual(future.result(timeout=2)['status'], 'completed')
            progress = json.loads((output / 'progress.json').read_text())
            self.assertEqual(progress['status'], 'completed')
            self.assertFalse(progress['vlm_pending'])
            self.assertFalse((output / 'progress.json.tmp').exists())

    def test_open_failure_publishes_error_without_claiming_cv_completion(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = root / 'config.json'
            config.write_text(json.dumps({'root': str(root), 'cameras': [{'id': 'one'}]}))
            with patch.object(pipeline, 'camera_open', side_effect=RuntimeError('fixture failure')):
                with self.assertRaisesRegex(RuntimeError, 'fixture failure'):
                    pipeline.run(config, root / 'run')
            progress = json.loads((root / 'run/progress.json').read_text())
            self.assertEqual(progress['status'], 'error')
            self.assertFalse(progress['cv_ready'])
            self.assertFalse(progress['vlm_pending'])


if __name__ == '__main__':
    unittest.main()
