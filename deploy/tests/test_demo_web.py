import contextlib
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

import deploy.demo_web as web


class DemoWebTests(unittest.TestCase):
    def test_local_examples_and_detector_profiles_are_usable_without_baked_boxes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            web.atomic_json(root / 'models/convnext/manifest.json', {'heads': {
                'omnifall_v2': {'frames': 3, 'classes': [False, True]}}})
            source = root / 'service/assets/demo-local/phone-group.mp4'
            source.parent.mkdir(parents=True)
            source.write_bytes(b'video upload fixture')
            web.atomic_json(root / 'service/samples.local.json', [{
                'id': 'phone-group', 'case': 'phone', 'video': 'assets/demo-local/phone-group.mp4',
                'title': 'Phones in a group', 'boxes': [{'label': 'phone'}]}])
            with patch.object(web.ThreadPoolExecutor, 'submit') as submit, TestClient(web.create_app(root)) as client:
                samples = client.get('/api/samples').json()['samples']
                self.assertEqual([sample['id'] for sample in samples], ['phone-group'])
                self.assertNotIn('boxes', samples[0])
                tasks = {task['id']: task for task in client.get('/api/tasks').json()['tasks']}
                self.assertEqual(tasks['phone_objects']['title_ru'], 'Телефоны')
                self.assertEqual(tasks['phone_objects']['semantics'], 'component')
                self.assertEqual(tasks['phone_use']['semantics'], 'action')
                response = client.post('/api/pipeline', json={'sampleId': 'phone-group', 'use_vlm': False})
                self.assertEqual(response.status_code, 202)
                job_id = response.json()['job']['id']
                saved = json.loads((root / 'artifacts/private/demo_web/jobs' / job_id / 'job.json').read_text())
                self.assertEqual(saved['task'], 'phone_use')
                submit.assert_called_once()

    def test_saved_zone_is_snapshotted_only_on_future_jobs_and_action_profiles_are_default(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            web.atomic_json(root / 'models/convnext/manifest.json', {'heads': {'omnifall_v2': {'frames': 3, 'classes': [False, True]}}})
            source = root / 'service/assets/demo-local/phone-group.mp4'
            source.parent.mkdir(parents=True)
            source.write_bytes(b'video upload fixture')
            web.atomic_json(root / 'service/samples.local.json', [{'id': 'phone-group', 'case': 'phone', 'video': 'assets/demo-local/phone-group.mp4', 'title': 'Phones'}])
            polygon = [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]
            with patch.object(web.ThreadPoolExecutor, 'submit') as submit, TestClient(web.create_app(root)) as client:
                saved = client.post('/api/zones', json={'sampleId': 'phone-group', 'polygon': polygon})
                self.assertEqual(saved.status_code, 200)
                self.assertEqual(saved.json()['zoneId'], 'phone-group')
                self.assertTrue(saved.json()['inferenceEnabledForFutureJobs'])
                self.assertFalse(saved.json()['appliedToExistingResults'])
                response = client.post('/api/pipeline', json={'sampleId': 'phone-group'})
                self.assertEqual(response.status_code, 202)
                job_id = response.json()['job']['id']
                job = json.loads((root / 'artifacts/private/demo_web/jobs' / job_id / 'job.json').read_text())
                self.assertEqual(job['task'], 'phone_use')
                self.assertEqual(job['zoneSnapshot'], {'zone_id': 'phone-group', 'camera_id': 'user_camera_01', 'polygon': polygon})
                submit.assert_called_once()

    def test_events_filter_by_camera_type_zone_and_finite_video_time_bounds(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            job_id = 'a' * 32
            job_path = root / 'artifacts/private/demo_web/jobs' / job_id / 'job.json'
            web.atomic_json(job_path, {
                'id': job_id, 'status': 'completed', 'sampleId': 'sample-one',
                'input_kind': 'image_set', 'sourceImages': [{'sha256': 'a'}, {'sha256': 'b'}],
                'zoneSnapshot': {'zone_id': 'sample-one', 'camera_id': 'user_camera_01', 'polygon': [[0, 0], [1, 0], [1, 1]]},
                'events': [
                    {'event_id': 'evt-a', 'camera_id': 'user_camera_01', 'rule_id': 'phone_use_review', 'trigger_timestamp_s': 2.5,
                     'raw_frames': [{'frame_index': 0, 'timestamp_s': 1.5}]},
                    {'event_id': 'evt-b', 'camera_id': 'user_camera_01', 'rule_id': 'zone_sample-one', 'trigger_timestamp_s': 8.0,
                     'raw_frames': [{'frame_index': 1, 'timestamp_s': 8.0}]},
                    {'event_id': 'evt-c', 'camera_id': 'user_camera_02', 'rule_id': 'phone_use_review', 'trigger_timestamp_s': 2.5,
                     'raw_frames': [{'frame_index': 1, 'timestamp_s': 2.5}]},
                ],
            })
            with TestClient(web.create_app(root)) as client:
                response = client.get('/api/events?camera=user_camera_01&type=phone_use_review&zone=sample-one&start_s=2&end_s=3')
                self.assertEqual(response.status_code, 200)
                events = response.json()['events']
                self.assertEqual([event['event_id'] for event in events], ['evt-a'])
                self.assertEqual(events[0]['sample_id'], 'sample-one')
                self.assertEqual(events[0]['zone_id'], 'sample-one')
                self.assertEqual(events[0]['zone_polygon'], [[0, 0], [1, 0], [1, 1]])
                self.assertEqual(events[0]['source_camera_id'], 'user_camera_01')
                self.assertEqual(events[0]['type'], 'phone_use_review')
                self.assertEqual(events[0]['image_index'], 0)
                other_camera = next(event for event in client.get('/api/events').json()['events'] if event['event_id'] == 'evt-c')
                self.assertEqual(other_camera['type'], 'phone_use_review')
                self.assertEqual(other_camera['image_index'], 1)
                self.assertIsNone(other_camera['zone_id'])
                self.assertIsNone(other_camera['zone_polygon'])
                self.assertIsNone(other_camera['source_camera_id'])
                for query in ['start_s=nan', 'end_s=inf', 'start_s=-1', 'end_s=604801', 'start_s=5&end_s=4']:
                    self.assertEqual(client.get('/api/events?' + query).status_code, 422, query)

    def test_event_journal_orders_by_job_creation_then_video_relative_time(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            jobs_dir = root / 'artifacts/private/demo_web/jobs'
            for job_id, created_at, events in [
                ('a' * 32, '2026-10-24T10:00:00+00:00', [
                    {'event_id': 'older-job-later-clip', 'camera_id': 'user_camera_01', 'rule_id': 'fall', 'trigger_timestamp_s': 90},
                ]),
                ('b' * 32, '2026-10-25T10:00:00+00:00', [
                    {'event_id': 'new-job-early-clip', 'camera_id': 'user_camera_01', 'rule_id': 'phone_use_review', 'trigger_timestamp_s': 1},
                    {'event_id': 'new-job-later-clip', 'camera_id': 'user_camera_01', 'rule_id': 'phone_use_review', 'trigger_timestamp_s': 3},
                ]),
            ]:
                web.atomic_json(jobs_dir / job_id / 'job.json', {'id': job_id, 'created_at': created_at, 'events': events})
            with TestClient(web.create_app(root)) as client:
                events = client.get('/api/events').json()['events']
            self.assertEqual([item['event_id'] for item in events],
                             ['new-job-later-clip', 'new-job-early-clip', 'older-job-later-clip'])
            self.assertEqual(events[0]['job_created_at'], '2026-10-25T10:00:00+00:00')
            self.assertEqual(events[0]['trigger_timestamp_s'], 3)

    def test_action_review_response_never_becomes_automatic_violation_verdict(self):
        selected = {'event_id': 'event-1', 'camera_id': 'user_camera_01', 'trigger_timestamp_s': 3.0,
                    'vlm': {'status': 'confirmed', 'request_ms': 100,
                            'response': {'unsafe': True, 'reason': 'Visible person holds phone and looks at screen.', 'evidence': ['frames']}}}
        for task in ['phone_use', 'smoking']:
            with self.subTest(task=task):
                analysis, error = web.analysis_from_event(selected, 'latest_positive_model_candidate', task)
                self.assertIsNone(error)
                self.assertEqual(analysis['assessment'], 'unknown')
                self.assertEqual(analysis['actionCandidate'], 'unknown')
                self.assertEqual(analysis['actionCandidateStatus'], 'insufficient_object_evidence')
                self.assertEqual(analysis['modelObservation']['reason'], selected['vlm']['response']['reason'])
                self.assertEqual(analysis['summary'], 'Недостаточно детекций целевого предмета в кадрах эпизода, чтобы оценить действие.')
                self.assertEqual(analysis['facts'], ['Свежий детектор не подтвердил целевой предмет в кадрах этого эпизода.'])
                self.assertNotIn('rawResponse', analysis)
                self.assertIn('not an automatic safety violation verdict', analysis['semantics'])
                with_object = web.analysis_from_event(selected, 'latest_positive_model_candidate', task, target_detected=True)[0]
                self.assertEqual(with_object['actionCandidate'], 'visible_action_candidate')
                self.assertEqual(with_object['actionCandidateStatus'], 'review_candidate')
        ordinary, error = web.analysis_from_event(selected, 'latest_positive_model_candidate', 'synthsite')
        self.assertIsNone(error)
        self.assertEqual(ordinary['assessment'], 'confirmed')

    def test_action_evidence_requires_actual_target_detection_in_causal_episode_frames(self):
        event = {'camera_id': 'user_camera_01', 'raw_frames': [{'frame_index': 2}]}
        book_rows = [{'camera_id': 'user_camera_01', 'frame_index': 2, 'accepted': True,
                      'stages': {'detector': {'actual_execution': True, 'mode': 'fresh_inference',
                                              'detections': [{'label': 'person'}, {'label': 'book'}]}}}]
        phone_rows = [{'camera_id': 'user_camera_01', 'frame_index': 2, 'accepted': True,
                       'stages': {'detector': {'actual_execution': True, 'mode': 'fresh_inference',
                                               'detections': [{'label': 'phone'}]}}}]
        self.assertFalse(web.event_has_target_detection(event, book_rows, 'phone_use'))
        self.assertTrue(web.event_has_target_detection(event, phone_rows, 'phone_use'))
        self.assertIsNone(web.event_has_target_detection(event, phone_rows, 'synthsite'))
        self.assertEqual(web.action_review_fields(
            {'vlm': {'status': 'confirmed', 'response': {'unsafe': True, 'reason': 'Book in hand', 'evidence': []}}},
            book_rows, 'phone_use')['actionCandidateStatus'], 'insufficient_object_evidence')

    def test_image_index_uses_bounded_latest_causal_evidence_frame(self):
        event = {'trigger_timestamp_s': 2.0, 'raw_frames': [
            {'frame_index': 0, 'timestamp_s': 1.0}, {'frame_index': 1, 'timestamp_s': 2.0},
            {'frame_index': 2, 'timestamp_s': 2.0},
        ]}
        self.assertEqual(web.event_image_index(event, 2), 1)
        self.assertEqual(web.event_image_index(event, 1), 0)
        self.assertIsNone(web.event_image_index({'raw_frames': [{'frame_index': 2, 'timestamp_s': 1}]}, 1))
        self.assertEqual(web.event_type({'kind': 'incident', 'rule_id': 'smoking_review'}), 'smoking_review')
        self.assertEqual(web.event_type({'kind': 'inspection', 'rule_id': 'inspection'}), 'inspection')

    def test_cv_only_action_event_stays_visible_without_claiming_a_vlm_result(self):
        from unittest.mock import patch

        event = {'event_id': 'candidate-1', 'camera_id': 'user_camera_01', 'rule_id': 'phone_use_review',
                 'trigger_timestamp_s': 2.0, 'vlm': {'status': 'unconfigured', 'actual_execution': False}}
        rows = [{'camera_id': 'user_camera_01', 'frame_index': 2, 'timestamp_s': 2.0, 'accepted': True,
                 'stages': {'detector': {'actual_execution': True, 'mode': 'fresh_inference',
                                         'detections': [{'label': 'phone'}]}}}]
        self.assertTrue(web.event_has_target_detection(event, rows, 'phone_use'))
        fields = web.action_review_fields(event, rows, 'phone_use')
        self.assertEqual(fields['actionCandidate'], 'unknown')
        self.assertEqual(fields['actionCandidateStatus'], 'model_response_unavailable')
        with tempfile.TemporaryDirectory() as folder, patch.object(web, 'download_file') as download:
            episode, files = web.collect_event_episode(None, 'remote', Path(folder), 'a' * 32, event)
        download.assert_not_called()
        self.assertEqual(files, [])
        self.assertEqual(episode['eventId'], 'candidate-1')
        self.assertEqual(episode['start'], 2.0)
        self.assertEqual(episode['frames'], [])

    def test_image_set_upload_preserves_originals_hashes_and_synthetic_time_without_running_inference(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            web.atomic_json(root / 'models/convnext/manifest.json', {'heads': {'omnifall_v2': {'frames': 3, 'classes': [False, True]}}})
            encoded = []
            for color in ['red', 'blue']:
                stream = io.BytesIO()
                Image.new('RGB', (16, 16), color).save(stream, format='PNG')
                encoded.append(stream.getvalue())

            def prepare(paths, output_dir, *, camera_id, ffmpeg='ffmpeg'):
                output_dir.mkdir()
                video = output_dir / 'video.mp4'
                video.write_bytes(b'prepared synthetic frames')
                sources = []
                for path in paths:
                    with Image.open(path) as image:
                        sources.append({'format': 'PNG', 'bytes': path.stat().st_size, 'sha256': web.digest(path),
                                        'width': image.width, 'height': image.height})
                receipt = output_dir / 'receipt.json'
                receipt.write_text('{}')
                return {'video_path': str(video), 'receipt_path': str(receipt), 'input_kind': 'image_set',
                        'synthetic_time': True, 'original_source_hashes': [item['sha256'] for item in sources],
                        'source_images': sources, 'video_sha256': web.digest(video), 'duration_seconds': len(sources)}

            with patch.object(web, 'prepare_image_set', side_effect=prepare), \
                 patch.object(web.ThreadPoolExecutor, 'submit') as submit, TestClient(web.create_app(root)) as client:
                response = client.post('/api/image-jobs', files=[('files', ('one.png', encoded[0], 'image/png')),
                                                                   ('files', ('two.png', encoded[1], 'image/png'))],
                                       data={'task': 'phone_objects', 'use_vlm': 'false'})
                self.assertEqual(response.status_code, 202, response.text)
                job = response.json()
                self.assertEqual(job['input_kind'], 'image_set')
                self.assertTrue(job['synthetic_time'])
                self.assertEqual(job['Vfreq'], 1)
                self.assertEqual(job['source_sha256s'], [hashlib.sha256(data).hexdigest() for data in encoded])
                self.assertEqual([item['name'] for item in job['sourceImages']], ['one.png', 'two.png'])
                self.assertEqual(client.get(job['sourceImages'][0]['url']).content, encoded[0])
                sample = next(item for item in client.get('/api/samples').json()['samples'] if item['id'] == job['sampleId'])
                self.assertEqual(sample['inputKind'], 'image_set')
                self.assertTrue(sample['syntheticTime'])
                self.assertEqual([item['sha256'] for item in sample['sourceImages']], job['source_sha256s'])
                facade = next(item for item in client.get('/api/pipeline').json()['jobs'] if item['id'] == job['id'])
                self.assertEqual(facade['task'], 'phone_objects')
                self.assertEqual(facade['taskTitle'], 'Телефоны')
                self.assertEqual(facade['case'], 'phone')
                self.assertEqual(facade['inputKind'], 'image_set')
                self.assertTrue(facade['syntheticTime'])
                self.assertEqual(facade['sourceSHA256s'], job['source_sha256s'])
                self.assertEqual(facade['sourceImages'][0]['name'], 'one.png')
                submit.assert_called_once()

    @unittest.skipUnless(shutil.which('ffmpeg'), 'ffmpeg is required for the real image-set route integration test')
    def test_image_set_route_uses_real_helper_and_returns_source_geometry_and_hashes(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            web.atomic_json(root / 'models/convnext/manifest.json', {'heads': {'omnifall_v2': {'frames': 3, 'classes': [False, True]}}})
            encoded = []
            for color in ['red', 'blue']:
                stream = io.BytesIO()
                Image.new('RGB', (32, 24), color).save(stream, format='PNG')
                encoded.append(stream.getvalue())
            with patch.object(web.ThreadPoolExecutor, 'submit') as submit, TestClient(web.create_app(root)) as client:
                response = client.post('/api/image-jobs', files=[('files', ('left.png', encoded[0], 'image/png')),
                                                                   ('files', ('right.png', encoded[1], 'image/png'))],
                                       data={'task': 'phone_objects', 'use_vlm': 'false'})
                self.assertEqual(response.status_code, 202, response.text)
                job = response.json()
                self.assertTrue(job['synthetic_time'])
                self.assertEqual(job['input_kind'], 'image_set')
                images = job['sourceImages']
                self.assertEqual([image['name'] for image in images], ['left.png', 'right.png'])
                self.assertEqual([(image['width'], image['height']) for image in images], [(32, 24), (32, 24)])
                self.assertEqual([image['sha256'] for image in images], [hashlib.sha256(data).hexdigest() for data in encoded])
                self.assertEqual([client.get(image['url']).content for image in images], encoded)
                stored_path = root / 'artifacts/private/demo_web/jobs' / job['id'] / 'job.json'
                stored_job = json.loads(stored_path.read_text())
                receipt = json.loads((stored_path.parent / 'input/image-set-receipt.json').read_text())
                self.assertEqual(receipt['frame_count'], 2)
                self.assertEqual((receipt['width'], receipt['height']), (32, 24))
                self.assertTrue(Path(stored_job['inputs'][0]['path']).is_file())
                submit.assert_called_once()

    def test_image_set_api_rejects_action_tasks_and_unsafe_filenames(self):
        with tempfile.TemporaryDirectory() as folder, TestClient(web.create_app(Path(folder))) as client:
            file = ('one.png', b'not decoded because task is rejected', 'image/png')
            rejected = client.post('/api/image-jobs', files=[('files', file), ('files', ('two.png', b'x', 'image/png'))],
                                   data={'task': 'phone_use'})
            self.assertEqual(rejected.status_code, 422)
            unsafe = client.post('/api/image-jobs', files=[('files', ('../one.png', b'x', 'image/png')),
                                                           ('files', ('two.png', b'x', 'image/png'))])
            self.assertEqual(unsafe.status_code, 400)

    def test_status_reads_nested_vlm_execution_metadata_without_exposing_extra_fields(self):
        with tempfile.TemporaryDirectory() as folder, TestClient(web.create_app(Path(folder))) as client:
            with patch.object(web.linq_client, 'connect', return_value=object()), \
                 patch.object(web.linq_client, 'request', return_value=Mock(status_code=200)), \
                 patch.object(web, 'execute_checked', return_value={
                     'ready': True,
                     'execution': {'model': 'Qwen3.5-0.8B', 'device': 'cpu', 'runtime_path': '/private/model/location'},
                 }):
                response = client.get('/api/status')
            self.assertEqual(response.status_code, 200)
            value = response.json()
            self.assertTrue(value['vlm_ready'])
            self.assertEqual(value['vlm_model'], 'Qwen3.5-0.8B')
            self.assertEqual(value['vlm_device'], 'cpu')
            self.assertNotIn('runtime_path', response.text)
            self.assertNotIn('/private/model/location', response.text)

    def test_primary_overlay_uses_first_input_even_if_other_camera_accepts_first(self):
        job = {'id': 'a' * 32, 'inputs': [{'camera': 'camera_01', 'sha256': 'a' * 64}, {'camera': 'camera_02', 'sha256': 'b' * 64}]}
        def row(camera, accepted, timestamp, source_sha):
            return {'camera_id': camera, 'accepted': accepted, 'timestamp_s': timestamp, 'frame_index': int(timestamp * 30),
                    'source_sha256': source_sha, 'width': 100, 'height': 100, 'tracks': [],
                    'stages': {'detector': {'status': 'ok', 'actual_execution': True, 'mode': 'fresh_inference',
                                           'detections': [{'xyxy': [1, 2, 3, 4], 'score': 0.9, 'label': 'person'}]}}}
        rejected = row('camera_01', False, 0, 'a' * 64)
        other = row('camera_02', True, 0, 'b' * 64)
        primary = row('camera_01', True, 1, 'a' * 64)
        episode = {'camera': 'camera_01', 'eventId': 'primary-event'}
        result = web.actual_result([rejected, other, primary], job, episode)
        self.assertEqual(result['primaryCamera'], 'camera_01')
        self.assertEqual([frame['camera'] for frame in result['frames']], ['camera_01'])
        self.assertEqual(result['episode']['camera'], result['primaryCamera'])
        empty = web.actual_result([rejected, other], job, episode)
        self.assertEqual(empty['primaryCamera'], 'camera_01')
        self.assertEqual(empty['frames'], [])
        intermediate = web.intermediate_result([rejected, other], job)
        self.assertEqual(intermediate['result']['primaryCamera'], 'camera_01')
        self.assertEqual(intermediate['result']['frames'], [])
        self.assertEqual(web.primary_camera_for_job(job, [other, primary]), 'camera_01')
        self.assertEqual(web.primary_camera_for_job({}, [other], episode), 'camera_01')
        swapped = {**other, 'camera_id': 'camera_01'}
        with self.assertRaisesRegex(ValueError, 'camera input'):
            web.intermediate_result([swapped], job)
        with self.assertRaisesRegex(ValueError, 'camera input'):
            web.actual_result([swapped], job)
        with self.assertRaisesRegex(ValueError, 'camera/source mapping'):
            web.validate_manifest_camera_sources({'cameras': [
                {'id': 'camera_01', 'source_sha256': 'b' * 64},
                {'id': 'camera_02', 'source_sha256': 'a' * 64},
            ]}, job, [rejected, other, primary])

    def test_incident_summary_retains_earlier_positive_and_each_causal_evidence(self):
        image_bytes = b'evidence fixture'
        image_sha = hashlib.sha256(image_bytes).hexdigest()
        def event(identity, timestamp, unsafe, status='confirmed'):
            return {'event_id': identity, 'camera_id': 'camera_01', 'trigger_timestamp_s': timestamp,
                    'kind': 'inspection', 'request_reasons': ['source_end'] if timestamp == 3 else ['periodic'],
                    'raw_frames': [{'camera_id': 'camera_01', 'relative_path': 'raw/' + identity + '.jpg', 'timestamp_s': timestamp, 'sha256': image_sha}],
                    'vlm': {'status': status, 'response': {'unsafe': unsafe, 'reason': identity, 'evidence': [identity]}}}
        positive = event('early-risk', 2, True)
        negative = event('later-safe', 3, False, 'rejected')
        selected, policy = web.select_model_event([positive, negative])
        self.assertIs(selected, positive)
        self.assertEqual(policy, 'latest_positive_model_candidate')
        with tempfile.TemporaryDirectory() as folder:
            results = Path(folder)
            def download(client, remote, output):
                output.write_bytes(image_bytes)
            with patch.object(web, 'download_file', side_effect=download):
                episodes = [web.collect_event_episode(None, 'runs/actual', results, 'a' * 32, item)[0] for item in [positive, negative]]
                self.assertEqual([item['eventId'] for item in episodes], ['early-risk', 'later-safe'])
                self.assertEqual([item['unsafe'] for item in episodes], [True, False])
                self.assertEqual(episodes[0]['triggerTime'], 2)
                self.assertNotEqual(episodes[0]['frames'], episodes[1]['frames'])
                self.assertEqual(episodes[0]['frameSHA256'], [image_sha])
                self.assertIsNone(episodes[0]['operatorConfirmed'])
                self.assertEqual(next(item for item in episodes if item['eventId'] == selected['event_id'])['eventId'], positive['event_id'])
                for frame_camera in ['camera_02', None]:
                    wrong_camera = event('wrong-camera', 2, True)
                    if frame_camera is None:
                        wrong_camera['raw_frames'][0].pop('camera_id')
                    else:
                        wrong_camera['raw_frames'][0]['camera_id'] = frame_camera
                    with self.assertRaises(ValueError):
                        web.collect_event_episode(None, 'runs/actual', results, 'a' * 32, wrong_camera)
                invalid = event('future', 2, True)
                invalid['raw_frames'][0]['timestamp_s'] = 3
                with self.assertRaises(ValueError):
                    web.collect_event_episode(None, 'runs/actual', results, 'a' * 32, invalid)
                mismatch = event('mismatch', 2, True)
                mismatch['raw_frames'][0]['sha256'] = '0' * 64
                with self.assertRaises(ValueError):
                    web.collect_event_episode(None, 'runs/actual', results, 'a' * 32, mismatch)

    def test_incident_summary_never_converts_unknown_error_or_string_to_safe(self):
        def event(identity, timestamp, unsafe, status='unknown'):
            return {'event_id': identity, 'trigger_timestamp_s': timestamp,
                    'vlm': {'status': status, 'response': {'unsafe': unsafe, 'reason': identity, 'evidence': []}}}
        negative = event('negative', 4, False, 'rejected')
        unknown = event('unknown', 2, None)
        error = event('error', 3, False, 'error')
        malformed = event('malformed', 3.5, 'false', 'rejected')
        self.assertIs(web.select_model_event([negative, unknown])[0], unknown)
        self.assertIs(web.select_model_event([negative, error])[0], error)
        self.assertIs(web.select_model_event([negative, malformed])[0], malformed)
        self.assertIsNone(web.model_event_response(error))
        self.assertIsNone(web.model_event_response(malformed))
        prior_negative = event('prior-negative', 1, False, 'rejected')
        selected, policy = web.select_model_event([prior_negative, negative])
        self.assertIs(selected, negative)
        self.assertEqual(policy, 'latest_negative_all_events_negative')
        self.assertEqual(web.select_model_event([]), (None, None))

    def test_console_modules_and_public_contract_exclude_private_job_fields(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            service = root / 'service'
            service.mkdir()
            names = ['console-data.js', 'console-analytics.js', 'console-analytics.css',
                     'console-cameras.js', 'console-cameras.css', 'viewer-annotations.js']
            for name in names:
                (service / name).write_text('exported module fixture')
            (service / 'server.mjs').write_text('private server fixture')
            (root / '.secrets').mkdir()
            (root / '.secrets/session.json').write_text('credential-sentinel-never-public')
            job_id = 'a' * 32
            job = {'id': job_id, 'sampleId': 'fall-01', 'created_at': '2026-10-24T10:00:00Z',
                   'completed_at': '2026-10-24T10:00:01Z', 'status': 'completed', 'stage': 'completed',
                   'inputs': [{'path': '/private/input.mp4', 'sha256': 'b' * 64}],
                   'launch': {'token': 'credential-sentinel-never-public'},
                   'upload_receipts': [{'token': 'credential-sentinel-never-public'}],
                   'result': {'frames': [], 'metrics': {'frames': 0, 'totalMs': None}},
                   'analysis': {'assessment': 'unknown', 'summary': 'Недостаточно данных', 'latencyMs': None},
                   'events': [{'event_id': 'evt', 'camera_id': 'camera_01', 'kind': 'inspection',
                               'vlm': {'response': {'unsafe': None, 'reason': 'Недостаточно данных', 'evidence': []}}}]}
            path = root / 'artifacts/private/demo_web/jobs' / job_id / 'job.json'
            web.atomic_json(path, job)
            with TestClient(web.create_app(root)) as client:
                for name in names:
                    response = client.get('/' + name)
                    self.assertEqual(response.status_code, 200, name)
                    self.assertEqual(response.text, 'exported module fixture')
                for url in ['/server.mjs', '/samples.json', '/.secrets/session.json', '/%2e%2e/.secrets/session.json']:
                    self.assertEqual(client.get(url).status_code, 404, url)
                snapshot = client.get('/api/pipeline').json()
                facade = snapshot['jobs'][0]
                self.assertEqual(facade['status'], 'ready')
                self.assertEqual(facade['createdAt'], job['created_at'])
                self.assertEqual(facade['finishedAt'], job['completed_at'])
                self.assertEqual(facade['analysis']['assessment'], 'unknown')
                self.assertIsNone(facade['analysis']['latencyMs'])
                self.assertEqual(snapshot['reviews'], [])
                self.assertEqual(snapshot['zones'], {})
                event = client.get('/api/events').json()['events'][0]
                self.assertIsNone(event['operator_confirmed'])
                self.assertIsNone(event['vlm']['response']['unsafe'])
                for url in ['/api/pipeline', '/api/jobs', '/api/jobs/' + job_id, '/api/events']:
                    text = client.get(url).text
                    self.assertNotIn('credential-sentinel-never-public', text)
                    self.assertNotIn('/private/input.mp4', text)

    def test_partial_remote_frame_log_exposes_only_complete_actual_records(self):
        row = {'accepted': True, 'frame_index': 0}
        content = json.dumps(row) + '\n' + '{"accepted":true,"frame_index":'
        with patch.object(web.linq_client, 'contents', return_value={'format': 'text', 'content': content}):
            self.assertEqual(web.remote_frame_rows(None, 'run/frames.jsonl'), [row])
        with patch.object(web.linq_client, 'contents', return_value={'format': 'text', 'content': '{invalid}\n' + json.dumps(row) + '\n'}):
            with self.assertRaises(json.JSONDecodeError):
                web.remote_frame_rows(None, 'run/frames.jsonl')

    def test_intermediate_cv_is_visible_while_vlm_pending_without_final_analysis(self):
        source_sha = 'a' * 64
        job = {'id': 'f' * 32, 'sampleId': 'fall-01', 'inputs': [{'sha256': source_sha}]}
        detection = {'label': 'person', 'xyxy': [10, 20, 50, 80], 'score': 0.7}
        row = {'accepted': True, 'source_sha256': source_sha, 'frame_index': 1, 'camera_id': 'camera_01', 'timestamp_s': 1,
               'width': 100, 'height': 100, 'tracks': [{'label': 'person', 'xyxy': [11, 20, 51, 80], 'observed': True, 'track_id': 0}],
               'stages': {'detector': {'status': 'ok', 'actual_execution': True, 'mode': 'fresh_inference', 'detections': [detection], 'elapsed_ms': 12}}}
        proof = {'mode': 'causal_fresh_video_inference', 'replay_fallback': False, 'cv_ready': False, 'vlm_pending': True}
        running = web.intermediate_result([row], job, proof)
        self.assertEqual(running['stage'], 'detecting')
        self.assertFalse(running['cv_ready'])
        self.assertEqual(running['result']['frames'][0]['detections'][0]['trackId'], 0)
        pending = web.intermediate_result([row], job, {**proof, 'cv_ready': True, 'cv_completed_at': 'actual timestamp'})
        self.assertEqual(pending['stage'], 'analyzing')
        self.assertTrue(pending['vlm_pending'])
        self.assertIsNone(pending['analysis'])
        self.assertIsNone(pending['result']['episode'])
        self.assertEqual(pending['result']['metrics']['frames'], 1)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / 'artifacts/private/demo_web/jobs' / job['id'] / 'job.json'
            web.atomic_json(path, {**job, **pending, 'created_at': 'actual timestamp'})
            with TestClient(web.create_app(root)) as client:
                facade = client.get('/api/pipeline').json()['jobs'][0]
                self.assertEqual(facade['status'], 'analyzing')
                self.assertTrue(facade['cv_ready'])
                self.assertTrue(facade['vlm_pending'])
                self.assertEqual(facade['result']['frames'], pending['result']['frames'])
        completed = web.intermediate_result([row], job, proof, {
            'status': 'completed', 'mode': 'causal_fresh_video_inference', 'replay_fallback': False,
            'cameras': [{'id': 'camera_01', 'source_sha256': source_sha}],
        })
        self.assertTrue(completed['cv_ready'])
        self.assertFalse(completed['vlm_pending'])
        self.assertEqual(completed['stage'], 'rendering')
        with self.assertRaises(ValueError):
            web.intermediate_result([{**row, 'source_sha256': 'b' * 64}], job, proof)
        with self.assertRaises(ValueError):
            web.intermediate_result([row], job, {**proof, 'replay_fallback': True})
        with self.assertRaises(ValueError):
            web.intermediate_result([row, row], job, proof)

    def test_reviewed_dataset_last_human_label_wins_and_csv_preserves_unicode_quotes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            job_id = 'e' * 32
            directory = root / 'artifacts/private/demo_web/jobs' / job_id
            evidence = '/media/' + job_id + '/evidence_0.jpg'
            source_sha = 'a' * 64
            job = {'id': job_id, 'status': 'completed', 'sampleId': 'fall-01', 'task': 'omnifall_v2',
                   'inputs': [{'path': '/private/never-export.mp4', 'sha256': source_sha, 'camera': 'camera_01'}], 'media_files': ['evidence_0.jpg'],
                   'result': {'episode': {'eventId': 'evt', 'frames': [evidence], 'frameTimes': [2]}},
                   'events': [{'event_id': 'evt', 'vlm': {'response': {'unsafe': True, 'reason': 'Человек, "лестница"', 'evidence': ['видимый кадр']}, 'request_ms': 100}}],
                   'analysis': {'assessment': 'confirmed', 'modelprovider': 'LinQ CPU Qwen3.5-0.8B'}}
            web.atomic_json(directory / 'job.json', job)
            web.atomic_json(directory / 'results/manifest.json', {'status': 'completed', 'mode': 'causal_fresh_video_inference', 'replay_fallback': False, 'cameras': [{'id': 'camera_01', 'source_sha256': source_sha}]})
            (directory / 'results/evidence_0.jpg').write_bytes(b'evidence fixture')
            note = 'Проверка, "цитата"\nвторая строка'
            stored = {'zones': {}, 'reviews': [{'jobId': job_id, 'verdict': 'confirmed', 'reviewedAt': '2026-01-01', 'note': 'earlier'},
                                               {'jobId': job_id, 'verdict': 'uncertain', 'reviewedAt': '2026-01-02', 'note': note}]}
            web.atomic_json(directory.parent.parent / 'workspace.json', stored)
            with TestClient(web.create_app(root)) as client:
                response = client.get('/api/dataset.jsonl')
                self.assertEqual(response.status_code, 200)
                records = [json.loads(line) for line in response.text.splitlines()]
                self.assertEqual(len(records), 1)
                self.assertEqual(records[0]['humanLabel'], 'uncertain')
                self.assertIs(records[0]['modelUnsafe'], True)
                self.assertEqual(records[0]['sourceSHA256'], [source_sha])
                self.assertEqual(records[0]['evidenceUrls'], [evidence])
                self.assertNotIn('/private/', response.text)
                csv_response = client.get('/api/dataset.csv')
                rows = list(csv.DictReader(io.StringIO(csv_response.text)))
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]['note'], note)
                self.assertEqual(rows[0]['modelReason'], 'Человек, "лестница"')
                self.assertEqual(rows[0]['humanLabel'], 'uncertain')
                self.assertIn('attachment;', csv_response.headers['content-disposition'])
                job['synthetic'] = True
                web.atomic_json(directory / 'job.json', job)
                self.assertEqual(client.get('/api/dataset.jsonl').text, '')

    def test_reviewed_dataset_empty_exports_have_headers_and_no_example_rows(self):
        with tempfile.TemporaryDirectory() as folder, TestClient(web.create_app(Path(folder))) as client:
            jsonl = client.get('/api/dataset.jsonl')
            self.assertEqual(jsonl.status_code, 200)
            self.assertEqual(jsonl.text, '')
            self.assertIn('linq-reviewed.jsonl', jsonl.headers['content-disposition'])
            response = client.get('/api/dataset.csv')
            self.assertEqual(response.status_code, 200)
            reader = csv.DictReader(io.StringIO(response.text))
            self.assertIn('humanLabel', reader.fieldnames)
            self.assertEqual(list(reader), [])

    def test_event_review_exports_nonprimary_episode_with_matching_camera_and_model_fields(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            job_id = 'f' * 32
            directory = root / 'artifacts/private/demo_web/jobs' / job_id
            first_sha, second_sha = 'a' * 64, 'b' * 64
            first_url, second_url = f'/media/{job_id}/camera_one.jpg', f'/media/{job_id}/camera_two.jpg'
            job = {'id': job_id, 'status': 'completed', 'sampleId': 'two-camera-upload', 'task': 'phone_use',
                   'completed_at': '2026-10-25T12:00:00+00:00',
                   'inputs': [{'path': '/private/one.mp4', 'sha256': first_sha, 'camera': 'camera_01'},
                              {'path': '/private/two.mp4', 'sha256': second_sha, 'camera': 'camera_02'}],
                   'media_files': ['camera_one.jpg', 'camera_two.jpg'],
                   'events': [
                       {'event_id': 'evt-primary', 'camera_id': 'camera_01', 'rule_id': 'phone_use_review',
                        'vlm': {'response': {'unsafe': True, 'reason': 'primary model claim', 'evidence': ['primary']}}},
                       {'event_id': 'evt-reviewed', 'camera_id': 'camera_02', 'rule_id': 'phone_use_review',
                        'vlm': {'response': {'unsafe': False, 'reason': 'matching event response', 'evidence': ['camera two']},
                                'request_ms': 42}},
                   ],
                   'result': {'episode': {'eventId': 'evt-primary', 'camera': 'camera_01', 'frames': [first_url], 'frameTimes': [1.0]},
                              'episodes': [
                                  {'eventId': 'evt-primary', 'camera': 'camera_01', 'frames': [first_url], 'frameTimes': [1.0]},
                                  {'eventId': 'evt-reviewed', 'camera': 'camera_02', 'frames': [second_url], 'frameTimes': [3.0]},
                              ]},
                   'analysis': {'eventId': 'evt-primary', 'assessment': 'confirmed', 'rawResponse': {'unsafe': True},
                                'modelprovider': 'selected primary model'}}
            web.atomic_json(directory / 'job.json', job)
            web.atomic_json(directory / 'results/manifest.json', {
                'status': 'completed', 'mode': 'causal_fresh_video_inference', 'replay_fallback': False,
                'cameras': [{'id': 'camera_01', 'source_sha256': first_sha},
                            {'id': 'camera_02', 'source_sha256': second_sha}],
            })
            (directory / 'results/camera_one.jpg').write_bytes(b'first evidence')
            (directory / 'results/camera_two.jpg').write_bytes(b'second evidence')
            with TestClient(web.create_app(root)) as client:
                saved = client.post(f'/api/events/{job_id}/evt-reviewed/review',
                                    json={'confirmed': True, 'note': 'Reviewed camera two'})
                self.assertEqual(saved.status_code, 200)
                response = client.get('/api/dataset.jsonl')
                records = [json.loads(line) for line in response.text.splitlines()]
                self.assertEqual(len(records), 1)
                row = records[0]
                self.assertEqual(row['eventId'], 'evt-reviewed')
                self.assertEqual(row['cameraId'], 'camera_02')
                self.assertEqual(row['evidenceUrls'], [second_url])
                self.assertEqual(row['evidenceTimes'], [3.0])
                self.assertIs(row['modelUnsafe'], False)
                self.assertIsNone(row['modelAssessment'])
                self.assertEqual(row['modelReason'], 'matching event response')
                self.assertIs(row['operatorConfirmed'], True)
                self.assertEqual(row['humanLabel'], 'confirmed')
                self.assertEqual(row['reviewSource'], 'event')

    def test_cv_only_completed_facade_reports_model_review_as_disabled_not_error(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            job_id = '9' * 32
            job = {'id': job_id, 'sampleId': 'upload_' + job_id, 'created_at': '2026-10-25T10:00:00+00:00',
                   'status': 'completed', 'task': 'phone_use', 'use_vlm': False,
                   'inputs': [{'sha256': 'a' * 64}],
                   'events': [{'event_id': 'event', 'event_trigger': 'active_event', 'camera_id': 'cam',
                               'rule_id': 'phone_use_review', 'vlm': {'status': 'disabled_by_request', 'actual_execution': False}}],
                   'analysis': None, 'analysisError': None, 'sceneReviewStatus': 'disabled_by_operator',
                   'result': {'episodes': [{'eventId': 'event', 'camera': 'cam', 'frames': []}],
                              'episode': {'eventId': 'event', 'camera': 'cam', 'frames': []}}}
            web.atomic_json(root / 'artifacts/private/demo_web/jobs' / job_id / 'job.json', job)
            with TestClient(web.create_app(root)) as client:
                facade = client.get('/api/pipeline').json()['jobs'][0]
                self.assertEqual(facade['sceneReviewStatus'], 'disabled_by_operator')
                self.assertIsNone(facade['analysis'])
                self.assertIsNone(facade['analysisError'])

    def test_reviewed_image_set_exports_original_hashes_and_marks_synthetic_evidence_time(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            job_id = '8' * 32
            directory = root / 'artifacts/private/demo_web/jobs' / job_id
            prepared_sha = 'a' * 64
            original_hashes = ['b' * 64, 'c' * 64]
            source_images = [
                {'name': 'arrival.png', 'sha256': original_hashes[0], 'url': f'/input/{job_id}/frame_01.png'},
                {'name': 'phone.png', 'sha256': original_hashes[1], 'url': f'/input/{job_id}/frame_02.png'},
            ]
            job = {'id': job_id, 'status': 'completed', 'sampleId': 'image_set_' + job_id,
                   'task': 'phone_objects', 'input_kind': 'image_set', 'synthetic_time': True,
                   'inputs': [{'path': '/private/prepared.mp4', 'sha256': prepared_sha, 'camera': 'user_camera_01'}],
                   'source_sha256s': original_hashes, 'sourceImages': source_images,
                   'imageSourceFiles': ['frame_01.png', 'frame_02.png'], 'events': [
                       {'event_id': 'still-event', 'camera_id': 'user_camera_01', 'rule_id': 'phone_present',
                        'vlm': {'response': {'unsafe': None, 'reason': 'visible phone', 'evidence': ['frame 2']}}}],
                   'result': {'episode': {'eventId': 'still-event', 'camera': 'user_camera_01',
                                          'image_index': 1, 'imageIndices': [0, 1], 'frames': [], 'frameTimes': [1.0]},
                              'episodes': [{'eventId': 'still-event', 'camera': 'user_camera_01',
                                            'image_index': 1, 'imageIndices': [0, 1], 'frames': [], 'frameTimes': [1.0]}]}}
            web.atomic_json(directory / 'job.json', job)
            web.atomic_json(directory / 'results/manifest.json', {
                'status': 'completed', 'mode': 'causal_fresh_video_inference', 'replay_fallback': False,
                'cameras': [{'id': 'user_camera_01', 'source_sha256': prepared_sha}],
            })
            with TestClient(web.create_app(root)) as client:
                reviewed = client.post(f'/api/events/{job_id}/still-event/review',
                                       json={'confirmed': True, 'note': 'Still frame verified'})
                self.assertEqual(reviewed.status_code, 200)
                rows = [json.loads(line) for line in client.get('/api/dataset.jsonl').text.splitlines()]
                self.assertEqual(len(rows), 1)
                row = rows[0]
                self.assertEqual(row['inputKind'], 'image_set')
                self.assertTrue(row['syntheticTime'])
                self.assertTrue(row['evidenceTimesSynthetic'])
                self.assertEqual(row['sourceSHA256'], original_hashes)
                self.assertEqual(row['originalSourceSHA256'], original_hashes)
                self.assertEqual(row['executionVideoSHA256'], [prepared_sha])
                self.assertEqual(row['imageIndex'], 1)
                self.assertEqual(row['evidenceUrls'], [source_images[1]['url']])
                self.assertEqual(row['evidenceTimes'], [1.0])
                csv_rows = list(csv.DictReader(io.StringIO(client.get('/api/dataset.csv').text)))
                self.assertEqual(csv_rows[0]['inputKind'], 'image_set')
                self.assertEqual(json.loads(csv_rows[0]['originalSourceSHA256']), original_hashes)
                self.assertEqual(json.loads(csv_rows[0]['executionVideoSHA256']), [prepared_sha])
                self.assertEqual(csv_rows[0]['evidenceTimesSynthetic'], 'true')

    def test_observed_track_matches_shifted_same_class_boxes_without_reusing_ids(self):
        detections = [
            {'label': 'person', 'xyxy': [10, 10, 50, 90]},
            {'label': 'person', 'xyxy': [60, 10, 100, 90]},
            {'label': 'person', 'xyxy': [200, 10, 240, 90]},
            {'label': 'phone', 'xyxy': [10, 10, 50, 90]},
            {'label': 'person', 'xyxy': [300, 10, 340, 90]},
        ]
        tracks = [
            {'label': 'person', 'xyxy': [61, 11, 101, 91], 'track_id': 22, 'observed': True},
            {'label': 'person', 'xyxy': [11, 11, 51, 91], 'track_id': 0, 'observed': True},
            {'label': 'person', 'xyxy': [200, 10, 240, 90], 'track_id': 33, 'observed': False},
        ]
        matches = web.observed_track_matches(detections, tracks)
        self.assertEqual(matches, {0: 0, 1: 22})
        self.assertEqual(len(set(matches.values())), len(matches))
        self.assertEqual(web.observed_track_matches([detections[0], detections[0]], [tracks[1]]), {0: 0})
        row = {'accepted': True, 'camera_id': 'camera_01', 'width': 400, 'height': 100, 'timestamp_s': 1,
               'tracks': tracks, 'stages': {'detector': {'detections': [{**item, 'score': 0.8} for item in detections]}}}
        result = web.actual_result([row], {'id': 'regression'})
        self.assertEqual([item['trackId'] for item in result['frames'][0]['detections']], [0, 22, None, None, None])
        self.assertEqual(result['metrics']['tracks'], 2)

    def test_assembly_verifies_sha_and_deletes_only_owned_parts(self):
        with tempfile.TemporaryDirectory() as folder:
            home = Path(folder)
            directory = home / 'work/mosprom_hackathon_linq/videos' / ('web_' + 'a' * 32)
            parts_dir = directory / 'upload_parts'
            parts_dir.mkdir(parents=True)
            data = b'ordered upload payload'
            part_name = 'camera_01.mp4.000000.part'
            (parts_dir / part_name).write_bytes(data)
            unrelated = parts_dir / 'unrelated.part'
            unrelated.write_bytes(b'preserve')
            expected = hashlib.sha256(data).hexdigest()
            parts = [{'name': part_name, 'sha256': expected, 'bytes': len(data)}]
            code = web.assembly_code('videos/web_' + 'a' * 32, 'camera_01.mp4', parts, expected, len(data))
            with patch.object(Path, 'home', return_value=home), contextlib.redirect_stdout(io.StringIO()):
                exec(code, {})
            self.assertEqual((directory / 'camera_01.mp4').read_bytes(), data)
            self.assertTrue(unrelated.exists())
            self.assertFalse((parts_dir / part_name).exists())

    def test_bad_assembly_sha_preserves_parts_and_never_publishes_video(self):
        with tempfile.TemporaryDirectory() as folder:
            home = Path(folder)
            directory = home / 'work/mosprom_hackathon_linq/videos' / ('web_' + 'b' * 32)
            parts_dir = directory / 'upload_parts'
            parts_dir.mkdir(parents=True)
            part = parts_dir / 'camera_01.mp4.000000.part'
            part.write_bytes(b'damaged')
            parts = [{'name': part.name, 'sha256': '0' * 64, 'bytes': 7}]
            code = web.assembly_code('videos/web_' + 'b' * 32, 'camera_01.mp4', parts, '0' * 64, 7)
            with patch.object(Path, 'home', return_value=home), self.assertRaises(ValueError):
                exec(code, {})
            self.assertTrue(part.exists())
            self.assertFalse((directory / 'camera_01.mp4').exists())
            self.assertFalse((directory / 'camera_01.mp4.joining').exists())

    def test_upload_retries_413_using_smaller_parts_and_verifies_receipt(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'input.mp4'
            source.write_bytes(b'x' * (9 * 1024 * 1024))
            sizes = []
            def upload(client, path, destination):
                size = Path(path).stat().st_size
                self.assertTrue(all(not component.startswith('.') for component in Path(destination).parts))
                self.assertIn('/upload_parts/', destination)
                sizes.append(size)
                if size > 4 * 1024 * 1024:
                    raise RuntimeError('Jupyter API returned HTTP 413')
            receipt = {'verified': True, 'sha256': web.digest(source), 'bytes': source.stat().st_size, 'video': 'videos/web_' + 'c' * 32 + '/camera_01.mp4'}
            with patch.object(web.linq_client, 'upload', side_effect=upload), patch.object(web, 'execute_checked', return_value=receipt):
                result = web.upload_video(None, source, 'c' * 32, 'camera_01', Mock())
            self.assertEqual(sizes, [8 * 1024 * 1024, 4 * 1024 * 1024, 4 * 1024 * 1024, 1024 * 1024])
            self.assertEqual(result, receipt)

    def test_jobs_limits_and_durable_review_are_separate_from_model_candidate(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = root / 'models/convnext/manifest.json'
            web.atomic_json(manifest, {'heads': {'synthsite': {'frames': 3, 'classes': ['safe', 'unsafe']}}})
            with patch.object(web.ThreadPoolExecutor, 'submit'), TestClient(web.create_app(root)) as client:
                self.assertEqual(client.post('/api/jobs', files=[('files', ('bad.exe', b'abcd'))]).status_code, 400)
                with patch.object(web, 'MAX_FILE', 3):
                    self.assertEqual(client.post('/api/jobs', files=[('files', ('test.mp4', b'abcd'))]).status_code, 413)
                response = client.post('/api/jobs', files=[('files', ('../../test.mp4', b'abcd'))], data={'task': 'synthsite', 'use_vlm': 'false', 'Vfreq': '1'})
                self.assertEqual(response.status_code, 202)
                job = response.json()
                self.assertNotIn('inputs', job)
                self.assertFalse(job['use_vlm'])
                self.assertEqual(client.post('/api/jobs', files=[('files', ('next.mp4', b'abcd'))]).status_code, 409)
                job_path = root / 'artifacts/private/demo_web/jobs' / job['id'] / 'job.json'
                stored = json.loads(job_path.read_text())
                stored['events'] = [{'event_id': 'evt', 'camera_id': 'user_camera_01', 'vlm': {'status': 'confirmed', 'response': {'unsafe': True}}}]
                web.atomic_json(job_path, stored)
                event = client.get('/api/events').json()['events'][0]
                self.assertIsNone(event['operator_confirmed'])
                route = '/api/events/' + job['id'] + '/evt/review'
                self.assertEqual(client.post(route, json={'confirmed': 'false'}).status_code, 422)
                self.assertEqual(client.post(route, json={'confirmed': False, 'note': 'Rejected by operator'}).status_code, 200)
                self.assertFalse(client.get('/api/events').json()['events'][0]['operator_confirmed'])
                self.assertEqual(client.get('/media/' + job['id'] + '/job.json').status_code, 404)

    def test_remote_chunk_paths_never_use_hidden_components(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'input.mp4'
            source.write_bytes(b'upload path fixture')
            receipt = {'verified': True, 'sha256': web.digest(source), 'bytes': source.stat().st_size}
            with patch.object(web.linq_client, 'upload') as upload, patch.object(web, 'execute_checked', return_value=receipt):
                web.upload_video(None, source, 'd' * 32, 'camera_01', Mock())
            remote_path = upload.call_args.args[2]
            self.assertIn('/upload_parts/', remote_path)
            self.assertTrue(all(not part.startswith('.') for part in Path(remote_path).parts))

    def test_unsafe_artifact_paths_are_rejected(self):
        for value in ['../video.mp4', '/tmp/video.mp4', 'a\\video.mp4', '..']:
            with self.assertRaises(ValueError):
                web.safe_filename(value)

    def test_existing_ui_catalog_pipeline_review_zone_and_static_facade(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            web.atomic_json(root / 'models/convnext/manifest.json', {'heads': {'omnifall_v2': {'frames': 3, 'classes': ['fall', 'normal']}}})
            source = root / 'service/assets/industrial/fall.mp4'
            source.parent.mkdir(parents=True)
            source.write_bytes(b'only uploaded by mocked worker')
            web.atomic_json(root / 'service/samples.json', [{'id': 'fall-01', 'case': 'fall', 'video': 'assets/industrial/fall.mp4', 'boxes': [[1, 2, 3, 4]], 'duration': 1}])
            web.atomic_json(root / 'service/object-references.json', [{'id': 'phone-01', 'case': 'phone', 'image': 'phone.jpg', 'boxes': [[1, 2, 3, 4]]}])
            (root / 'service/index.html').write_text('<html>Existing UI</html>')
            (root / 'service/workspace.js').write_text('const existing = true;')
            (root / 'service/server.mjs').write_text('never execute Node')
            with patch.object(web.ThreadPoolExecutor, 'submit') as submit, TestClient(web.create_app(root)) as client:
                self.assertEqual(client.get('/').status_code, 200)
                self.assertEqual(client.get('/workspace.js').status_code, 200)
                self.assertEqual(client.get('/server.mjs').status_code, 404)
                self.assertTrue(all('boxes' not in sample for sample in client.get('/api/samples').json()['samples']))
                self.assertEqual(client.post('/api/pipeline', json={'sampleId': 'phone-01'}).status_code, 422)
                zone = [[0.1, 0.1], [0.8, 0.1], [0.8, 0.8]]
                response = client.post('/api/zones', json={'sampleId': 'fall-01', 'polygon': zone})
                self.assertEqual(response.status_code, 200)
                self.assertTrue(response.json()['inferenceEnabledForFutureJobs'])
                self.assertFalse(response.json()['appliedToExistingResults'])
                job = client.post('/api/pipeline', json={'sampleId': 'fall-01'}).json()['job']
                self.assertEqual(job['status'], 'detecting')
                submit.assert_called_once()
                path = root / 'artifacts/private/demo_web/jobs' / job['id'] / 'job.json'
                stored = json.loads(path.read_text())
                self.assertEqual(stored['task'], 'omnifall_v2')
                self.assertTrue(stored['use_vlm'])
                self.assertEqual(stored['inputs'][0]['sha256'], web.digest(source))
                stored.update(status='completed', result={'episode': {'eventId': 'selected-event', 'frames': []}}, analysis={'assessment': 'confirmed'})
                web.atomic_json(path, stored)
                snapshot = client.get('/api/pipeline').json()
                self.assertEqual(snapshot['jobs'][0]['status'], 'ready')
                self.assertEqual(snapshot['jobs'][0]['task'], 'omnifall_v2')
                self.assertEqual(snapshot['jobs'][0]['taskTitle'], 'Падения')
                self.assertEqual(snapshot['jobs'][0]['case'], 'fall')
                self.assertEqual(snapshot['reviews'], [])
                response = client.post('/api/reviews', json={'jobId': job['id'], 'verdict': 'uncertain', 'note': 'Need context'})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()['review']['eventId'], 'selected-event')
                self.assertEqual(client.get('/api/pipeline').json()['reviews'][0]['verdict'], 'uncertain')

    def test_catalog_pipeline_honors_task_and_vlm_options_and_rejects_unknown_task(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            web.atomic_json(root / 'models/convnext/manifest.json', {'heads': {
                'omnifall_v2': {'frames': 3, 'classes': ['fall', 'normal']},
                'safelift': {'frames': 3, 'classes': ['lift', 'lower']},
            }})
            source = root / 'service/assets/industrial/fall.mp4'
            source.parent.mkdir(parents=True)
            source.write_bytes(b'catalog fixture')
            web.atomic_json(root / 'service/samples.json', [{'id': 'fall-01', 'case': 'fall', 'video': 'assets/industrial/fall.mp4'}])
            with patch.object(web.ThreadPoolExecutor, 'submit') as submit, TestClient(web.create_app(root)) as client:
                invalid = client.post('/api/pipeline', json={'sampleId': 'fall-01', 'task': 'unknown', 'use_vlm': False})
                self.assertEqual(invalid.status_code, 422)
                response = client.post('/api/pipeline', json={'sampleId': 'fall-01', 'task': 'safelift', 'use_vlm': False})
                self.assertEqual(response.status_code, 202)
                job = response.json()['job']
                path = root / 'artifacts/private/demo_web/jobs' / job['id'] / 'job.json'
                stored = json.loads(path.read_text())
                self.assertEqual(stored['task'], 'safelift')
                self.assertFalse(stored['use_vlm'])
                submit.assert_called_once()

    def test_actual_frame_mapping_preserves_coordinates_scores_and_camera_identity(self):
        detection = {'label': 'person', 'xyxy': [10, 20, 50, 80], 'score': 0.56789}
        row = {'accepted': True, 'camera_id': 'camera_01', 'width': 100, 'height': 100, 'timestamp_s': 1,
               'tracks': [{'observed': True, 'label': 'person', 'xyxy': [10, 20, 50, 80], 'track_id': 7}],
               'stages': {'detector': {'detections': [detection], 'elapsed_ms': 42}}}
        other = {**row, 'camera_id': 'camera_02'}
        result = web.actual_result([row, other], {'id': 'a', 'videos': [{'camera': 'camera_02', 'url': '/actual.mp4'}]})
        self.assertEqual(len(result['frames']), 1)
        self.assertEqual(result['frames'][0]['detections'][0], {'label': 'person', 'box': [0.1, 0.2, 0.5, 0.8], 'score': 0.56789, 'trackId': 7})
        self.assertEqual(result['metrics']['detectorP50Ms'], 42)
        self.assertIsNone(result['metrics']['frameP50Ms'])
        self.assertEqual(result['videos'][0]['camera'], 'camera_02')


if __name__ == '__main__':
    unittest.main()
