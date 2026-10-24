import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from deploy import pipeline
from deploy.common import write_json
from deploy.tests.test_pipeline import make_state


def samples():
    for index, good in [(0, True), (15, False), (30, False), (31, True), (45, True), (61, True)]:
        image = Image.new('RGB', (20, 20))
        image.info.update(frame_index=index, quality_good=good)
        yield index, index / 30, image


def quality(image, config):
    good = image.info['quality_good']
    return {'accepted': good, 'enabled': True, 'reasons': [] if good else ['blur_or_no_texture']}


class PipelineCadenceTests(unittest.TestCase):
    def test_cadence_skips_quality_but_retries_after_bad_deadline_frame(self):
        with tempfile.TemporaryDirectory() as folder:
            state = make_state(Path(folder))
            state['config'].update(V=1, rules=[])
            rows, expected, last = [], [], float('-inf')
            for index, timestamp, image in samples():
                if image.info['quality_good'] and timestamp - last + 1e-9 >= 1:
                    expected.append(index)
                    last = timestamp
            with patch.object(pipeline, 'quality_gate', side_effect=quality) as gate:
                for index, timestamp, image in samples():
                    rows.append(pipeline.process_sample(state, image, index, timestamp, {}, None, {}, lambda *args: None))
            self.assertEqual(expected, [0, 31, 61])
            self.assertEqual([row['frame_index'] for row in rows if row['accepted']], expected)
            self.assertEqual(gate.call_count, 4)
            self.assertEqual([call.args[0].info['frame_index'] for call in gate.call_args_list], [0, 30, 31, 61])
            self.assertEqual(rows[2]['quality']['reasons'], ['blur_or_no_texture'])
            for index in [1, 4]:
                self.assertEqual(rows[index]['quality']['reasons'], ['sampling_cadence', 'quality_not_evaluated'])
                self.assertFalse(rows[index]['quality']['evaluated'])
                self.assertNotIn('statistics', rows[index]['quality'])

    def test_periodic_inspection_still_runs_on_cadence_skipped_frame(self):
        with tempfile.TemporaryDirectory() as folder:
            state = make_state(Path(folder))
            state['config'].update(V=1, rules=[])
            with patch.object(pipeline, 'quality_gate', side_effect=quality) as gate, patch.object(pipeline, 'periodic_due', side_effect=lambda current, timestamp, dispatcher: timestamp == 0.5), patch.object(pipeline, 'inspection') as inspection:
                for index, timestamp, image in list(samples())[:2]:
                    pipeline.process_sample(state, image, index, timestamp, {}, None, {}, lambda *args: None)
            self.assertEqual(gate.call_count, 1)
            inspection.assert_called_once()
            self.assertEqual(inspection.call_args.args[1:3], (0.5, ['periodic']))

    def test_manifest_distinguishes_quality_failure_from_unevaluated_cadence_skips(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            state = make_state(root / 'fixture')
            state['config'].update(V=1, rules=[])
            state.update(runtime={'backend': 'unit_test_fixture'}, detector_provenance={}, classifier_provenance={}, incident_model_provenance={})
            config = root / 'config.json'
            write_json(config, {'root': '.', 'cameras': [state['config']]})
            def decoded(current):
                yield from samples()
            with patch.object(pipeline, 'camera_open', return_value=state), patch.object(pipeline, 'sampled', side_effect=decoded), patch.object(pipeline, 'quality_gate', side_effect=quality) as gate:
                manifest = pipeline.run(config, root / 'run')
            self.assertEqual(gate.call_count, 4)
            self.assertEqual(manifest['accepted_frame_count'], 3)
            self.assertEqual(manifest['quality_rejected_frame_count'], 1)
            self.assertEqual(manifest['cadence_skipped_frame_count'], 2)
            self.assertIn('cadence-eligible', manifest['quality_rejection_count_policy'])
            progress = json.loads((root / 'run/progress.json').read_text())
            self.assertEqual(progress['quality_rejected_frame_count'], 1)
            self.assertEqual(progress['quality_rejection_count_policy'], manifest['quality_rejection_count_policy'])


if __name__ == '__main__':
    unittest.main()
