import copy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.run_user_video import build_config
from demo.core import incident_step, track_step


class UserVideoTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'videos').mkdir()
        for name in ['one.mp4', 'two.mp4']:
            (self.root / 'videos' / name).write_bytes(b'technical file selection fixture')
        bundle = self.root / 'models/convnext'
        bundle.mkdir(parents=True)
        source = Path(__file__).resolve().parents[1] / 'tests/fixtures/convnext-manifest.json'
        (bundle / 'manifest.json').write_text(source.read_text())
        source_config = json.loads((Path(__file__).resolve().parents[1] / 'config_single.json').read_text())
        source_config['cameras'][0]['detector']['options']['linq_person']['confidence'] = 0.25
        source_config['cameras'][0]['calibration'] = {'fixture': 'old camera calibration'}
        source_config['multicamera'] = {'fixture': 'old association'}
        self.original = copy.deepcopy(source_config)
        (self.root / 'config_single.json').write_text(json.dumps(source_config))
        source_config['cameras'].append(copy.deepcopy(source_config['cameras'][0]))
        (self.root / 'config_two.json').write_text(json.dumps(source_config))

    def test_new_video_keeps_models_and_exact_head_frames_without_old_calibration(self):
        config, details = build_config(self.root, ['one.mp4', 'videos/two.mp4'], 'safelift', 1, True)
        self.assertEqual(details['T'], 2)
        self.assertNotIn('vlm', config)
        self.assertNotIn('multicamera', config)
        for camera in config['cameras']:
            self.assertEqual(camera['classifier']['task'], 'safelift')
            self.assertEqual(camera['T'], 2)
            self.assertEqual(camera['rules'], [])
            self.assertNotIn('calibration', camera)
            self.assertEqual(camera['detector'], self.original['cameras'][0]['detector'])
        self.assertNotEqual(config['cameras'][0]['id'], config['cameras'][1]['id'])

    def test_changed_event_head_cannot_expand_uncertainty_allowlist(self):
        config, details = build_config(self.root, ['one.mp4'], 'physicalai_v2', 1, False)
        self.assertEqual(config['cameras'][0]['rules'][0]['label'], 'near_miss')
        self.assertEqual(config['vlm']['scope'], '02')
        self.assertEqual(config['uncertainty']['scopes'], [])
        self.assertEqual(config['vlm']['periodic_scope'], 'open')
        self.assertEqual(details['T'], 3)

    def test_detector_profiles_are_catalogued_without_fabricating_clip_heads(self):
        from scripts.run_user_video import available_tasks
        config = json.loads((self.root / 'config_single.json').read_text())
        tasks = available_tasks(self.root, config)
        self.assertEqual(len(tasks), 10)
        self.assertEqual(tasks['phone_objects']['classes'], None)
        self.assertEqual(tasks['cigarette_objects']['classes'], None)
        self.assertNotIn('path', tasks['phone_objects'])
        self.assertEqual(tasks['phone_objects']['title_ru'], 'Телефоны')
        self.assertEqual(tasks['cigarette_objects']['title_ru'], 'Сигареты')
        self.assertEqual(tasks['phone_objects']['task_spec']['scope'], '03')
        self.assertEqual(tasks['cigarette_objects']['task_spec']['scope'], '04')
        self.assertEqual(tasks['phone_use']['classes'], None)
        self.assertEqual(tasks['smoking']['classes'], None)
        self.assertEqual(tasks['phone_use']['task_spec']['label_semantics'], 'action')
        self.assertEqual(tasks['smoking']['task_spec']['label_semantics'], 'action')

    def test_phone_and_cigarette_profiles_use_detector_signals_and_presence_only_prompts(self):
        configured_thresholds = {}
        options = self.original['cameras'][0]['detector']['options']
        for label in ['phone', 'cigarette']:
            configured_thresholds[label] = next(model.get('confidence', options.get('confidence', 0.25))
                                                for model in options['models'] if label in model.get('labels', []))
        expected = [('phone_objects', 'Телефоны', 'phone', 'phone_present', '03', configured_thresholds['phone']),
                    ('cigarette_objects', 'Сигареты', 'cigarette', 'cigarette_present', '04', configured_thresholds['cigarette'])]
        for task, title, label, rule_id, scope, threshold in expected:
            with self.subTest(task=task):
                config, details = build_config(self.root, ['one.mp4'], task, 1, False)
                camera = config['cameras'][0]
                rule = camera['rules'][0]
                self.assertEqual(details['title_ru'], title)
                self.assertEqual(details['backend'], 'detector')
                self.assertIsNone(camera['classifier'])
                self.assertEqual(camera['task_profile'], task)
                self.assertEqual(camera['scope'], scope)
                self.assertEqual(rule['id'], rule_id)
                self.assertEqual(rule['source'], 'detector')
                self.assertEqual(rule['label'], label)
                self.assertEqual(rule['signal'], label + '_present')
                self.assertEqual(rule['on_threshold'], threshold)
                self.assertEqual(rule['activate_after_s'], 0.6)
                self.assertEqual(rule['clear_after_s'], 0.6)
                self.assertEqual([model['labels'] for model in camera['detector']['options']['models']], [[label]])
                self.assertEqual(camera['detector']['options']['linq_person'], self.original['cameras'][0]['detector']['options']['linq_person'])
                self.assertIn('visibly present', config['vlm']['prompt'])
                self.assertIn('visible person', config['vlm']['prompt'])
                self.assertIn('do not infer', config['vlm']['prompt'].lower())
                self.assertIn('safety violation', config['vlm']['prompt'].lower())
                self.assertEqual(config['vlm']['periodic_prompt'], config['vlm']['prompt'])
                self.assertEqual(config['vlm']['periodic_scope'], scope)
                self.assertTrue(config['vlm']['confirm_active_incidents'])
                self.assertEqual(config['uncertainty']['scopes'], [])

    def test_action_profiles_use_presence_trigger_and_review_only_prompts(self):
        for task, label, rule_id, action_a, action_b in [
            ('phone_use', 'phone', 'phone_use_review', 'actively looking at', 'holding or carrying'),
            ('smoking', 'cigarette', 'smoking_review', 'brings it to their mouth', 'carrying or holding'),
        ]:
            with self.subTest(task=task):
                config, details = build_config(self.root, ['one.mp4'], task, 1, False)
                camera = config['cameras'][0]
                self.assertIsNone(camera['classifier'])
                self.assertEqual(camera['task_profile'], task)
                self.assertEqual([model['labels'] for model in camera['detector']['options']['models']], [[label]])
                self.assertEqual(camera['rules'][0]['id'], rule_id)
                self.assertTrue(camera['rules'][0]['action_review'])
                self.assertIn(action_a, config['vlm']['prompt'])
                self.assertIn(action_b, config['vlm']['prompt'])
                self.assertIn('do not infer a safety violation', config['vlm']['prompt'].lower())
                self.assertEqual(config['vlm']['periodic_prompt'], config['vlm']['prompt'])
                self.assertEqual(config['vlm']['periodic_scope'], '03' if task == 'phone_use' else '04')
                self.assertEqual(details['backend'], 'detector')

    def test_phone_use_signal_from_built_profile_triggers_one_episode_after_three_fresh_samples(self):
        from PIL import Image
        from deploy import pipeline
        from deploy.tests.test_pipeline import make_state

        config, _ = build_config(self.root, ['one.mp4'], 'phone_use', 3, True)
        camera = config['cameras'][0]
        state = make_state(self.root / 'runtime')
        state['config'] = camera
        state['config']['quality'] = {'enabled': False}
        state['detector'] = lambda image, context, options: {
            'actual_execution': True,
            'detections': [
                {'xyxy': [2, 2, 18, 19], 'label': 'person', 'score': 0.95},
                {'xyxy': [8, 8, 11, 12], 'label': 'phone', 'score': 0.95},
            ],
            'signals': {'phone_present': 0.95},
        }
        state['classify'] = pipeline.classifier_step
        events, records = {}, []
        for index, timestamp in enumerate((0.0, 0.34, 0.68)):
            row = pipeline.process_sample(state, Image.new('RGB', (20, 20)), index, timestamp, {}, None, events,
                                          lambda stream, value: records.append((stream, value)))
            self.assertTrue(row['stages']['detector']['actual_execution'])
            self.assertEqual(row['stages']['detector']['signals']['phone_present'], 0.95)
        incidents = [event for event in events.values() if event.get('rule_id') == 'phone_use_review']
        self.assertEqual(len(incidents), 1)
        self.assertEqual(incidents[0]['state'], 'active')
        self.assertEqual(state['episodes'], {'phone_use_review': incidents[0]['event_id']})
        self.assertEqual([row[1]['rule_id'] for row in records if row[0] == 'events' and row[1].get('rule_id') == 'phone_use_review'],
                         ['phone_use_review'])

    def test_zone_snapshot_adds_normalized_restricted_zone_rule_without_affecting_heads(self):
        zone = {'zone_id': 'phone-area', 'camera_id': 'user_camera_01',
                'polygon': [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]}
        config, details = build_config(self.root, ['one.mp4'], 'phone_use', 1, False, zone=zone)
        rule = next(rule for rule in config['cameras'][0]['rules'] if rule['kind'] == 'restricted_zone')
        self.assertEqual(rule['id'], 'zone_phone-area')
        self.assertEqual(rule['polygon'], zone['polygon'])
        self.assertEqual(rule['coordinates'], 'normalized')
        self.assertEqual(details['zone'], zone)
        self.assertEqual(config['cameras'][0]['classifier'], None)
        trained, _ = build_config(self.root, ['one.mp4'], 'synthsite', 1, True, zone=zone)
        self.assertEqual(trained['cameras'][0]['classifier']['task'], 'synthsite')
        self.assertIn('restricted_zone', [rule['kind'] for rule in trained['cameras'][0]['rules']])

    def test_zone_snapshot_rejects_bad_identity_or_polygon(self):
        base = {'zone_id': 'zone', 'camera_id': 'user_camera_01', 'polygon': [[0, 0], [1, 0], [0, 1]]}
        for invalid in [
            {**base, 'camera_id': 'camera_99'},
            {**base, 'polygon': [[0, 0], [0.5, 0.5], [1, 1]]},
            {**base, 'polygon': [[0, 0], [1, 0], [0, float('nan')]]},
        ]:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                build_config(self.root, ['one.mp4'], 'phone_use', 1, True, zone=invalid)

    def test_image_set_config_records_original_hashes_and_forbids_motion_profiles(self):
        hashes = ['a' * 64, 'b' * 64]
        config, details = build_config(self.root, ['one.mp4'], 'phone_objects', 1, False,
                                       input_kind='image_set', source_sha256s=hashes)
        self.assertEqual(config['input_provenance'], {'kind': 'image_set', 'synthetic_time': True,
                                                      'original_source_sha256s': hashes})
        self.assertTrue(details['synthetic_time'])
        self.assertEqual(details['input_kind'], 'image_set')
        self.assertIn('synthetic one-second timestamps', config['vlm']['prompt'])
        self.assertIn('Do not infer motion', config['vlm']['prompt'])
        for task, frequency, source_hashes in [('phone_use', 1, hashes), ('phone_objects', 2, hashes),
                                                ('phone_objects', 1, ['not-sha'])]:
            with self.subTest(task=task, frequency=frequency, source_hashes=source_hashes), self.assertRaises(ValueError):
                build_config(self.root, ['one.mp4'], task, frequency, True,
                             input_kind='image_set', source_sha256s=source_hashes)

    def test_original_trained_tasks_keep_their_head_and_classifier_rule(self):
        for task in ['synthsite', 'safelift', 'hmdb51_v2', 'physicalai_v2', 'omnifall_v2', 'harness_author']:
            with self.subTest(task=task):
                config, details = build_config(self.root, ['one.mp4'], task, 1, True)
                camera = config['cameras'][0]
                self.assertEqual(camera['classifier']['task'], task)
                self.assertEqual(details['backend'], None)
                self.assertEqual(len(camera['rules']), int(bool(config['cameras'][0]['classifier']['task'] == 'synthsite' or task == 'physicalai_v2')))
                if camera['rules']:
                    self.assertEqual(camera['rules'][0]['source'], 'classifier')

    def test_one_fps_positive_evidence_activates_then_negative_evidence_clears(self):
        config, _ = build_config(self.root, ['one.mp4'], 'synthsite', 1, True)
        rule = config['cameras'][0]['rules'][0]
        state, observed = {}, []
        for timestamp, evidence in [(0, 0.95), (1, 0.95), (2, 0.95), (3, 0.05), (4, 0.05)]:
            item, _ = incident_step(state, 'candidate', timestamp, evidence, rule)
            observed.append(item['state'])
        self.assertEqual(observed, ['pending', 'active', 'active', 'cooldown', 'clear'])
        self.assertGreaterEqual(config['tracker']['max_age_s'], 2.5)
        tracks = track_step({}, [{'label': 'person', 'score': 0.3, 'xyxy': [1, 1, 20, 50]}], 0, config['tracker'])
        self.assertEqual(len(tracks), 1)

    def test_three_fps_preserves_existing_sufficient_timeouts(self):
        config, _ = build_config(self.root, ['one.mp4'], 'synthsite', 3, True)
        self.assertEqual(config['cameras'][0]['rules'][0]['max_evidence_gap_s'], 0.8)
        self.assertEqual(config['tracker']['max_age_s'], 1.0)
        self.assertEqual(config['tracker']['confidence'], 0.25)
        on_disk = json.loads((self.root / 'config_single.json').read_text())
        self.assertEqual(on_disk, self.original)

    def test_no_untrained_task_or_video_outside_upload_folder(self):
        with self.assertRaises(ValueError):
            build_config(self.root, ['one.mp4'], 'untrained', 1, True)
        (self.root / 'private.mp4').write_bytes(b'not an upload')
        with self.assertRaises(ValueError):
            build_config(self.root, ['../private.mp4'], 'synthsite', 1, True)
        for bad in [0, -1, float('inf'), float('nan')]:
            with self.assertRaises(ValueError):
                build_config(self.root, ['one.mp4'], 'synthsite', bad, True)


if __name__ == '__main__':
    unittest.main()
