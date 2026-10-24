from collections import deque
from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageFilter

from deploy import pipeline
from deploy.runtime import open_encoder


def detector(image, context, options):
    return {'actual_execution': True, 'detections': [{'xyxy': [1, 1, 8, 14], 'label': 'person', 'score': 0.9}], 'provenance': {'purpose': 'synthetic unit test fixture'}}


def make_state(root, camera='one', rule=None):
    buffer = root / 'buffer' / camera
    buffer.mkdir(parents=True)
    state = {'config': {'id': camera, 'quality': {'enabled': False}, 'V': 100, 'detector': {'options': {}}, 'classifier': {'task': 'test'}, 'rules': [rule or {'id': 'risk', 'kind': 'model_signal', 'source': 'classifier', 'label': 'unsafe', 'activate_after_s': 0.3, 'clear_after_s': 0.6}]}, 'output': root, 'source_sha256': 'frozen-source-hash', 'frames': deque(), 'features': deque(maxlen=3), 'spatial_features': deque(maxlen=3), 'max_frames': 3, 'buffer_window_s': 1.0, 'buffer_directory': buffer, 'tracker': {}, 'world': {}, 'incident_states': {}, 'episodes': {}, 'detector': detector, 'incident_model': None, 'risk': 1.0, 'next_periodic_s': 1000000000}
    def classify(current, frame):
        return {'status': 'ok', 'actual_execution': False, 'purpose': 'synthetic unit test fixture', 'classes': ['safe', 'unsafe'], 'probabilities': [1-current['risk'], current['risk']], 'prediction': 'unsafe' if current['risk'] else 'safe', 'input_timestamps_s': [frame['timestamp_s']]}
    state['classify'] = classify
    return state


def dispatcher(root, confirm_active=False):
    return pipeline.dispatch_create({'max_inflight': 1, 'max_queue': 0, 'max_frames': 3, 'confirm_active_incidents': confirm_active, 'config': 'not-read-by-test', 'scope': '05', 'prompt': 'test fixture'}, root)


class CausalPipelineTests(unittest.TestCase):
    def test_periodic_and_source_end_inspections_keep_configured_task_prompt_and_scope(self):
        with tempfile.TemporaryDirectory() as temporary:
            state = make_state(Path(temporary))
            config = {'scope': '03', 'prompt': 'phone active-use prompt', 'periodic_scope': '03',
                      'periodic_prompt': 'phone-only periodic prompt', 'max_frames': 3}
            jobs, events = {'config': config}, {}
            with patch.object(pipeline, 'snapshot', return_value=[]), patch.object(
                    pipeline, 'dispatch_submit', return_value={'status': 'queued'}):
                periodic = pipeline.inspection(state, 1.0, ['periodic'], jobs, events, lambda *args: None)
                source_end = pipeline.inspection(state, 2.0, ['source_end'], jobs, events, lambda *args: None)
            for event in [periodic, source_end]:
                self.assertEqual(event['route_scope'], '03')
                self.assertEqual(event['route_prompt'], 'phone-only periodic prompt')

    def test_vlm_receives_only_trigger_prefix_and_snapshot_survives_eviction(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = make_state(root)
            records, events, requests = [], {}, []
            def remote(config, paths, scope, prompt, context):
                requests.append({'context': context, 'paths': list(paths)})
                return {'response': {'unsafe': True, 'reason': 'test fixture', 'evidence': [], 'scope': scope}, 'transport': 'http', 'execution_device': 'unverified_remote_host', 'TPU_execution_verified': False}
            with patch.object(pipeline, 'call_vlm', side_effect=remote):
                jobs = dispatcher(root, confirm_active=True)
                for index in range(8):
                    state['risk'] = 1.0 if index <= 1 else 0.0
                    pipeline.process_sample(state, Image.new('RGB', (20, 20)), index, index*0.31, {}, jobs, events, lambda stream, row: records.append((stream, row)))
                pipeline.dispatch_drain(jobs, events, lambda stream, row: records.append((stream, row)), wait=True)
            created = [row for stream, row in records if row.get('type') == 'event_created']
            self.assertEqual(len(created), 1)
            self.assertEqual(len(requests), 1)
            request = requests[0]
            self.assertTrue(all(frame['timestamp_s'] <= 0.31 for frame in request['context']['frames']))
            self.assertTrue(all(Path(path).exists() for path in request['paths']))
            self.assertLessEqual(len(state['frames']), 3)
            self.assertEqual(next(iter(events.values()))['vlm']['status'], 'confirmed')
            self.assertFalse(next(iter(events.values()))['vlm']['model_execution_verified'])

    def test_cooldown_reactivation_keeps_event_id_and_does_not_reinvoke_vlm(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state, events, emitted = make_state(root), {}, []
            response = {'response': {'unsafe': None, 'reason': '', 'evidence': [], 'scope': '05'}}
            with patch.object(pipeline, 'call_vlm', return_value=response) as remote:
                jobs = dispatcher(root, confirm_active=True)
                for index, (timestamp, risk) in enumerate([(0, 1), (0.31, 1), (0.62, 0), (0.93, 1), (1.24, 1)]):
                    state['risk'] = risk
                    pipeline.process_sample(state, Image.new('RGB', (20, 20)), index, timestamp, {}, jobs, events, lambda stream, row: emitted.append(row))
                    pipeline.dispatch_drain(jobs, events, lambda stream, row: emitted.append(row))
                pipeline.dispatch_drain(jobs, events, lambda stream, row: emitted.append(row), wait=True)
            self.assertEqual(len(events), 1)
            self.assertEqual(remote.call_count, 1)
            self.assertEqual(sum(row.get('type') == 'event_created' for row in emitted), 1)
            self.assertEqual(next(iter(events.values()))['vlm']['status'], 'unknown')

    def test_detector_error_is_unknown_geometry_and_does_not_skip_classifier(self):
        with tempfile.TemporaryDirectory() as temporary:
            state = make_state(Path(temporary), rule={'id': 'roi', 'kind': 'restricted_zone', 'polygon': [[0, 0], [1, 0], [1, 1], [0, 1]], 'activate_after_s': 0})
            def failed(image, context, options):
                raise RuntimeError('actual detector unavailable')
            state['detector'] = failed
            row = pipeline.process_sample(state, Image.new('RGB', (20, 20)), 0, 0, {}, None, {}, lambda *args: None)
            self.assertEqual(row['stages']['detector']['status'], 'error')
            self.assertEqual(row['stages']['classifier']['status'], 'ok')
            self.assertEqual(row['indicators'][0]['state'], 'unknown')
            self.assertIsNone(row['indicators'][0]['score'])

    def test_geometric_zone_candidate_does_not_queue_action_vlm(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rule = {'id': 'zone_phone-area', 'kind': 'restricted_zone',
                    'polygon': [[0, 0.5], [0.5, 0.5], [0.5, 0.9], [0, 0.9]],
                    'coordinates': 'normalized', 'activate_after_s': 0.3, 'vlm_enabled': False}
            state = make_state(root, rule=rule)
            events, records = {}, []
            dispatcher = {'config': {'confirm_active_incidents': True, 'periodic_seconds': 30, 'scope': '03'}}
            for index, timestamp in enumerate([0.0, 0.31]):
                pipeline.process_sample(state, Image.new('RGB', (20, 20)), index, timestamp, {}, dispatcher,
                                       events, lambda stream, row: records.append((stream, row)))
            zone_events = [event for event in events.values() if event['rule_id'] == 'zone_phone-area']
            self.assertEqual(len(zone_events), 1)
            self.assertEqual(zone_events[0]['vlm']['status'], 'not_requested_for_geometric_rule')
            self.assertFalse(zone_events[0]['vlm']['actual_execution'])
            self.assertEqual(zone_events[0]['request_reasons'], [])

    def test_existing_clip_heads_warm_up_exactly_and_bound_feature_buffer(self):
        for required in [1, 2, 3]:
            with self.subTest(required=required), tempfile.TemporaryDirectory() as temporary:
                state = make_state(Path(temporary))
                state['features'] = deque(maxlen=required)
                state['buffer_window_s'] = 5
                state['head'] = {'feature_frames': required, 'classes': ['safe', 'unsafe'], 'mean': [0.0]*(768*required), 'scale': [1.0]*(768*required), 'coef': [[0.0]*(768*required)], 'intercept': [4.0]}
                state['runtime'] = {'manifest': {'preprocessing': {}}, 'run': lambda pixels: np.ones((1, 768), dtype=np.float32), 'backend': 'onnxruntime_cpu', 'device': 'cpu'}
                state['classify'] = pipeline.classifier_step
                rows = []
                with patch.object(pipeline, 'preprocess', return_value=np.zeros((1, 3, 224, 224), dtype=np.float32)):
                    for index in range(required+2):
                        rows.append(pipeline.process_sample(state, Image.new('RGB', (20, 20)), index, index*0.31, {}, None, {}, lambda *args: None))
                for row in rows[:required-1]:
                    self.assertEqual(row['stages']['classifier']['status'], 'warming_up')
                    self.assertIsNone(row['indicators'][0]['score'])
                last = rows[-1]['stages']['classifier']
                self.assertEqual(last['input_frame_indices'], list(range(2, required+2)))
                self.assertEqual(len(state['features']), required)
                self.assertEqual(last['status'], 'ok')
                self.assertFalse(last['TPU_execution_verified'])
                self.assertTrue(all(timestamp <= rows[-1]['timestamp_s'] for timestamp in last['input_timestamps_s']))

    def test_vlm_backpressure_is_explicit_and_does_not_block_detection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            states = [make_state(root, name, {'id': 'risk', 'kind': 'model_signal', 'source': 'classifier', 'label': 'unsafe', 'activate_after_s': 0}) for name in ['one', 'two']]
            events, release = {}, threading.Event()
            def blocked(*args):
                if not release.wait(3):
                    raise RuntimeError('unit test release timed out')
                raise RuntimeError('remote model unavailable')
            with patch.object(pipeline, 'call_vlm', side_effect=blocked):
                jobs = dispatcher(root, confirm_active=True)
                try:
                    rows = [pipeline.process_sample(state, Image.new('RGB', (20, 20)), 0, 0, {}, jobs, events, lambda *args: None) for state in states]
                    statuses = [event['vlm']['status'] for event in events.values()]
                    self.assertEqual(statuses, ['queued', 'queue_full'])
                    self.assertEqual(len(jobs['pending']), 1)
                    self.assertTrue(all(row['stages']['detector']['status'] == 'ok' for row in rows))
                    self.assertTrue(all(state['detector_TPU_frames'] == 1 for state in states))
                    self.assertTrue(all(state['verified_detector_TPU_frames'] == 0 for state in states))
                finally:
                    release.set()
                    pipeline.dispatch_drain(jobs, events, lambda *args: None, wait=True)
            failed = next(event for event in events.values() if event['camera_id'] == 'one')
            self.assertEqual(failed['vlm']['status'], 'error')
            self.assertIsNone(failed['vlm']['fallback'])

    def test_classifier_failure_never_reuses_a_previous_positive_score(self):
        with tempfile.TemporaryDirectory() as temporary:
            state = make_state(Path(temporary))
            events = {}
            first = pipeline.process_sample(state, Image.new('RGB', (20, 20)), 0, 0, {}, None, events, lambda *args: None)
            original = state['classify']
            def unavailable(current, frame):
                raise RuntimeError('encoder execution failed')
            state['classify'] = unavailable
            failed = pipeline.process_sample(state, Image.new('RGB', (20, 20)), 1, 0.31, {}, None, events, lambda *args: None)
            self.assertEqual(first['indicators'][0]['state'], 'pending')
            self.assertEqual(failed['indicators'][0]['state'], 'unknown')
            self.assertEqual(events, {})
            state['classify'] = original
            recovered = pipeline.process_sample(state, Image.new('RGB', (20, 20)), 2, 0.62, {}, None, events, lambda *args: None)
            self.assertEqual(recovered['indicators'][0]['state'], 'pending')
            self.assertEqual(events, {})

    def test_quality_gate_rejects_blur_and_exposure_without_test_tuning(self):
        pixels = (np.indices((32, 32)).sum(axis=0)%2*120+60).astype(np.uint8)
        sharp = Image.fromarray(pixels).convert('RGB')
        blurred = sharp.filter(ImageFilter.GaussianBlur(4))
        self.assertTrue(pipeline.quality_gate(sharp, {})['accepted'])
        self.assertIn('blur_or_no_texture', pipeline.quality_gate(blurred, {})['reasons'])
        self.assertIn('exposure_mean', pipeline.quality_gate(Image.new('RGB', (32, 32)), {})['reasons'])
        self.assertIn('exposure_clipping', pipeline.quality_gate(Image.new('RGB', (32, 32), 'white'), {})['reasons'])

    def test_uncertain_only_routes_allowlisted_scope_and_confident_does_not(self):
        for risk, scopes, expected in [(0.5, ['05'], 1), (0.99, ['05'], 0), (0.5, ['01'], 0)]:
            with self.subTest(risk=risk, scopes=scopes), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                state = make_state(root)
                state['config']['rules'] = []
                state['risk'] = risk
                settings = {'uncertainty': {'scopes': scopes, 'min_max_probability': 0.75, 'min_margin': 0.15}}
                events = {}
                with patch.object(pipeline, 'call_vlm', return_value={'response': {'unsafe': None}}) as remote:
                    jobs = dispatcher(root)
                    row = pipeline.process_sample(state, Image.new('RGB', (20, 20)), 0, 0, settings, jobs, events, lambda *args: None)
                    pipeline.dispatch_drain(jobs, events, lambda *args: None, wait=True)
                self.assertEqual(remote.call_count, expected)
                self.assertEqual(row['uncertainty']['uncertain'], risk == 0.5)
                if expected:
                    event = next(iter(events.values()))
                    self.assertEqual(event['request_reasons'], ['classifier_uncertainty'])
                    self.assertEqual(event['route_scope'], '05')

    def test_periodic_open_scan_runs_for_confident_camera_after_T_and_again(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = make_state(root)
            state.pop('next_periodic_s')
            state['risk'] = 0.99
            state['config']['rules'] = []
            events, contexts = {}, []
            def remote(config, paths, scope, prompt, context):
                contexts.append((scope, context))
                return {'response': {'unsafe': None}}
            with patch.object(pipeline, 'call_vlm', side_effect=remote):
                jobs = dispatcher(root)
                jobs['config']['periodic_seconds'] = 1
                jobs['config']['periodic_scope'] = 'open'
                jobs['config']['periodic_prompt'] = 'Inspect every visible kind of hazard'
                for index, timestamp in enumerate([0, 0.31, 0.62, 0.93, 1.24, 1.63]):
                    pipeline.process_sample(state, Image.new('RGB', (20, 20)), index, timestamp, {}, jobs, events, lambda *args: None)
                    pipeline.dispatch_drain(jobs, events, lambda *args: None)
                pipeline.dispatch_drain(jobs, events, lambda *args: None, wait=True)
            self.assertEqual(len(contexts), 2)
            self.assertEqual([value[0] for value in contexts], ['open', 'open'])
            self.assertEqual(contexts[0][1]['trigger_timestamp_s'], 0.62)
            self.assertTrue(all(context['request_reasons'] == ['periodic'] for scope, context in contexts))

    def test_quality_buffers_and_initial_periodic_schedules_are_camera_independent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            states = [make_state(root, name) for name in ['good', 'bad']]
            for state in states:
                state.pop('next_periodic_s')
                state['config']['quality'] = {'enabled': True}
                state['config']['rules'] = []
            def unavailable(*args):
                raise RuntimeError('fresh inference unavailable')
            states[0]['classify'] = unavailable
            states[0]['detector'] = unavailable
            sharp = Image.fromarray((np.indices((32, 32)).sum(axis=0)%2*120+60).astype(np.uint8)).convert('RGB')
            events = {}
            with patch.object(pipeline, 'call_vlm', return_value={'response': {'unsafe': None}}) as remote:
                jobs = dispatcher(root)
                jobs['capacity'] = 2
                for index in range(3):
                    pipeline.process_sample(states[0], sharp, index, index*0.31, {}, jobs, events, lambda *args: None)
                    rejected = pipeline.process_sample(states[1], Image.new('RGB', (32, 32)), index, index*0.31, {}, jobs, events, lambda *args: None)
                pipeline.dispatch_drain(jobs, events, lambda *args: None, wait=True)
            self.assertEqual(len(states[0]['frames']), 3)
            self.assertEqual(len(states[1]['frames']), 0)
            self.assertEqual(remote.call_count, 1)
            self.assertFalse(rejected['accepted'])
            self.assertNotIn('next_periodic_s', states[1])

    def test_round_robin_queue_preserves_slot_for_other_camera(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            release = threading.Event()
            calls, events = [], {}
            def remote(config, paths, scope, prompt, context):
                calls.append(context['camera_id'])
                release.wait(3)
                return {'response': {'unsafe': None}}
            with patch.object(pipeline, 'call_vlm', side_effect=remote):
                jobs = pipeline.dispatch_create({'max_inflight': 1, 'max_queue': 1, 'camera_ids': ['fast', 'slow'], 'config': 'unused', 'scope': '05', 'prompt': 'test'}, root)
                try:
                    for event_id, camera in [('a', 'fast'), ('b', 'slow')]:
                        event = {'event_id': event_id, 'camera_id': camera, 'rule_id': 'test', 'trigger_timestamp_s': 0}
                        events[event_id] = event
                        event['vlm'] = pipeline.dispatch_submit(jobs, event, [])
                    duplicate = pipeline.dispatch_submit(jobs, {**events['a'], 'event_id': 'duplicate'}, [])
                    self.assertEqual(duplicate['status'], 'camera_busy')
                    self.assertEqual(events['b']['vlm']['status'], 'queued')
                    self.assertEqual(list(jobs['order']), ['slow'])
                finally:
                    release.set()
                    pipeline.dispatch_drain(jobs, events, lambda *args: None, wait=True)
            self.assertEqual(calls, ['fast', 'slow'])

    def test_confident_positive_only_requests_vlm_when_active_confirmation_enabled(self):
        for confirmation, expected_calls in [(False, 0), (True, 1)]:
            with self.subTest(confirmation=confirmation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                state = make_state(root)
                events = {}
                with patch.object(pipeline, 'call_vlm', return_value={'response': {'unsafe': True}}) as remote:
                    jobs = dispatcher(root, confirm_active=confirmation)
                    for index, timestamp in enumerate([0, 0.31, 0.62]):
                        pipeline.process_sample(state, Image.new('RGB', (20, 20)), index, timestamp, {}, jobs, events, lambda *args: None)
                    pipeline.dispatch_drain(jobs, events, lambda *args: None, wait=True)
                self.assertEqual(len(events), 1)
                self.assertEqual(remote.call_count, expected_calls)
                event = next(iter(events.values()))
                self.assertEqual(event['event_trigger'], 'active_event')
                if not confirmation:
                    self.assertEqual(event['vlm']['status'], 'not_requested')
                    self.assertEqual(event['request_reasons'], [])

    def test_arbitrary_T_is_rejected_for_existing_trained_head(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'head.json').write_text(json.dumps({'feature_frames': 3}))
            runtime = {'bundle': root, 'manifest': {'heads': {'synthsite': {'path': 'head.json'}}}}
            @contextmanager
            def encoder(*args, **kwargs):
                yield runtime
            config = {'id': 'camera', 'video': 'unused.mp4', 'T': 5, 'detector': {'callable': 'unused:detect'}, 'classifier': {'bundle': 'unused', 'backend': 'cpu', 'task': 'synthsite'}}
            with pipeline.ExitStack() as stack, patch.object(pipeline, 'open_encoder', encoder):
                with self.assertRaisesRegex(ValueError, 'T must equal'):
                    pipeline.camera_open(config, {}, root, root, stack)

    def test_source_end_scan_covers_late_accepted_frame_while_first_scan_is_pending(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = make_state(root)
            state.pop('next_periodic_s')
            state['config'].update(V=1, rules=[])
            state['buffer_window_s'] = 4
            events, requests = {}, []
            release = threading.Event()
            def remote(config, paths, scope, prompt, context):
                release.wait(3)
                requests.append(context)
                late = Image.open(paths[-1]).getpixel((0, 0))[0] > 200
                return {'response': {'unsafe': late}}
            with patch.object(pipeline, 'call_vlm', side_effect=remote):
                jobs = pipeline.dispatch_create({'max_inflight': 1, 'max_queue': 1, 'max_frames': 3, 'scan_on_source_end': True, 'coalesce_latest': True, 'config': 'unused', 'scope': '05', 'prompt': 'narrow task', 'periodic_scope': 'open'}, root)
                try:
                    for index in range(4):
                        pipeline.process_sample(state, Image.new('RGB', (20, 20), 'white' if index == 3 else 'black'), index, index, {}, jobs, events, lambda *args: None)
                    final = pipeline.source_end_inspection(state, jobs, events, lambda *args: None)
                    self.assertEqual(final['request_reasons'], ['source_end'])
                    self.assertEqual(final['route_scope'], 'open')
                    self.assertEqual(final['trigger_timestamp_s'], 3)
                    self.assertEqual([frame['timestamp_s'] for frame in final['raw_frames']], [1, 2, 3])
                    self.assertIsNone(pipeline.source_end_inspection(state, jobs, events, lambda *args: None))
                finally:
                    release.set()
                    pipeline.dispatch_drain(jobs, events, lambda *args: None, wait=True)
            self.assertEqual([item['trigger_timestamp_s'] for item in requests], [2, 3])
            self.assertTrue(all(frame['timestamp_s'] <= item['trigger_timestamp_s'] for item in requests for frame in item['frames']))
            self.assertEqual(final['vlm']['status'], 'confirmed')

    def test_latest_queued_snapshot_replaces_old_without_starving_other_camera(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            events, calls, log = {}, [], []
            release = threading.Event()
            def remote(config, paths, scope, prompt, context):
                calls.append(context['event_id'])
                release.wait(3)
                return {'response': {'unsafe': None}}
            with patch.object(pipeline, 'call_vlm', side_effect=remote):
                jobs = pipeline.dispatch_create({'max_inflight': 1, 'max_queue': 2, 'coalesce_latest': True, 'config': 'unused', 'scope': '05', 'prompt': 'test'}, root)
                try:
                    for index, (identity, camera) in enumerate([('inflight', 'fast'), ('slow', 'slow'), ('old', 'fast'), ('latest', 'fast')]):
                        event = {'event_id': identity, 'camera_id': camera, 'rule_id': 'test', 'trigger_timestamp_s': index}
                        events[identity] = event
                        event['vlm'] = pipeline.dispatch_submit(jobs, event, [], lambda name, row: log.append(row))
                    self.assertEqual(list(jobs['order']), ['slow', 'fast'])
                    self.assertEqual(events['old']['vlm']['status'], 'superseded')
                    self.assertFalse(events['old']['vlm']['actual_execution'])
                    self.assertEqual(events['old']['vlm']['superseded_by_event_id'], 'latest')
                    self.assertEqual(len(jobs['queues']), 2)
                finally:
                    release.set()
                    pipeline.dispatch_drain(jobs, events, lambda *args: None, wait=True)
            self.assertEqual(calls, ['inflight', 'slow', 'latest'])
            self.assertEqual(json.loads((root / 'events/old/event.json').read_text())['vlm']['status'], 'superseded')
            self.assertEqual(log[0]['type'], 'vlm_superseded')

    def test_local_tpu_encoder_backend_is_rejected_before_asset_loading(self):
        with self.assertRaisesRegex(ValueError, 'Only the CPU encoder backend'):
            with open_encoder('missing-bundle', 'tpu'):
                self.fail('unsupported encoder backend must not open assets')


if __name__ == '__main__':
    unittest.main()
