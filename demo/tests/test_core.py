import unittest

import numpy as np

from demo.core import associate_world, calibration, footpoint, incident_step, project, signals, track_step


def detection(box, label='person'):
    return {'xyxy': box, 'label': label, 'score': 0.9}


class CalibrationTests(unittest.TestCase):
    def test_measured_correspondences_map_ground_footpoint(self):
        spec = {'plane': 'ground', 'units': 'metres', 'map_id': 'yard', 'image_points': [[100, 100], [500, 100], [500, 300], [100, 300]], 'world_points': [[0, 0], [8, 0], [8, 4], [0, 4]]}
        matrix = calibration(spec)
        np.testing.assert_allclose(project(footpoint([250, 100, 350, 200]), matrix), [4, 2], atol=1e-6)

    def test_degenerate_calibration_fails(self):
        with self.assertRaisesRegex(ValueError, 'Degenerate'):
            calibration({'plane': 'ground', 'units': 'metres', 'map_id': 'yard', 'image_points': [[0, 0], [1, 1], [2, 2], [3, 3]], 'world_points': [[0, 0], [1, 1], [2, 2], [3, 3]]})

    def test_raised_objects_are_excluded_and_separate_maps_never_merge(self):
        state = {}
        spec = {'plane': 'ground', 'units': 'metres', 'map_id': 'a', 'homography': [[0.01, 0, 0], [0, 0.01, 0], [0, 0, 1]]}
        load = {'track_id': 1, 'label': 'suspended_load', 'observed': True, 'xyxy': [0, 0, 20, 20]}
        self.assertEqual(associate_world(state, 'one', 0, [load], spec, {}), [])
        person = {**load, 'label': 'person'}
        first = associate_world(state, 'one', 0, [dict(person)], spec, {})[0]
        second = associate_world(state, 'two', 0, [dict(person)], {**spec, 'map_id': 'b'}, {})[0]
        self.assertNotEqual(first['global_id'], second['global_id'])

    def test_only_synchronized_different_camera_observations_merge(self):
        state = {}
        spec = {'plane': 'ground', 'units': 'metres', 'map_id': 'yard', 'homography': [[0.01, 0, 0], [0, 0.01, 0], [0, 0, 1]]}
        person = {'track_id': 1, 'label': 'person', 'observed': True, 'xyxy': [0, 0, 20, 20]}
        first = associate_world(state, 'one', 0, [dict(person)], spec, {})[0]
        second = associate_world(state, 'two', 0.05, [dict(person)], spec, {})[0]
        self.assertEqual(first['global_id'], second['global_id'])
        third = associate_world(state, 'three', 0.5, [dict(person)], spec, {})[0]
        self.assertNotEqual(first['global_id'], third['global_id'])
        local = associate_world(state, 'one', 0.1, [{**person, 'track_id': 2}], spec, {})[0]
        self.assertNotEqual(first['global_id'], local['global_id'])


class TrackerTests(unittest.TestCase):
    def test_motion_survives_brief_occlusion_but_expires(self):
        state = {}
        first = track_step(state, [detection([0, 0, 40, 100])], 0, {})[0]
        second = track_step(state, [detection([5, 0, 45, 100])], 0.25, {})[0]
        missing = track_step(state, [], 0.5, {})[0]
        self.assertFalse(missing['observed'])
        reappeared = track_step(state, [detection([15, 0, 55, 100])], 0.75, {})[0]
        self.assertEqual(first['track_id'], reappeared['track_id'])
        self.assertGreater(reappeared['xyxy'][0], second['xyxy'][0])
        expired = track_step(state, [detection([15, 0, 55, 100])], 3, {})[0]
        self.assertNotEqual(expired['track_id'], first['track_id'])

    def test_classes_cannot_exchange_identity(self):
        state = {}
        first = track_step(state, [detection([0, 0, 40, 100])], 0, {})[0]
        result = track_step(state, [detection([0, 0, 40, 100], 'truck')], 0.3, {})
        truck = next(row for row in result if row['label'] == 'truck')
        self.assertNotEqual(first['track_id'], truck['track_id'])

    def test_missing_observation_never_triggers_incident(self):
        tracks = [{'track_id': 1, 'label': 'person', 'xyxy': [0, 0, 90, 20], 'observed': False}]
        result = signals(tracks, [{'id': 'fall', 'kind': 'wide_person_proxy'}], 100, 100)
        self.assertEqual(result[0]['score'], 0)


class IncidentTests(unittest.TestCase):
    def test_short_pulse_cannot_activate_and_clear_requires_dwell(self):
        state, rule = {}, {'activate_after_s': 0.6, 'clear_after_s': 0.6}
        observed = []
        for timestamp, value in [(0, 1), (0.3, 0), (0.6, 1), (0.9, 1), (1.21, 1), (1.5, 0), (1.8, 0), (2.11, 0)]:
            result, _ = incident_step(state, 'test', timestamp, value, rule)
            observed.append(result['state'])
        self.assertEqual(observed, ['pending', 'clear', 'pending', 'pending', 'active', 'cooldown', 'cooldown', 'clear'])

    def test_missing_evidence_cannot_fill_dwell_time(self):
        state, rule = {}, {'activate_after_s': 0.5, 'max_evidence_gap_s': 0.6}
        incident_step(state, 'test', 0, 1, rule)
        result, _ = incident_step(state, 'test', 3, 1, rule)
        self.assertEqual(result['state'], 'pending')
        result, _ = incident_step(state, 'test', 3.1, None, rule)
        self.assertEqual(result['state'], 'unknown')

    def test_unknown_ground_distance_has_no_score(self):
        person = {'track_id': 1, 'label': 'person', 'observed': True, 'xyxy': [0, 0, 20, 100]}
        value = signals([person], [{'id': 'near', 'kind': 'ground_proximity'}], 100, 100)[0]
        self.assertIsNone(value['score'])


if __name__ == '__main__':
    unittest.main()
