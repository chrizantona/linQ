import sys
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from deploy.common import digest
from deploy.detector_adapter import (associate_attributes, detect, merge_region_detections,
                                     open_onnx_detector, person_regions,
                                     predict_onnx_regions)


def box(label, xyxy, score=0.8):
    return {'label': label, 'xyxy': xyxy, 'score': score}


class DetectorAssociationTests(unittest.TestCase):
    def test_onnx_session_uses_explicit_cpu_thread_limits_and_cached_graph(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'trained.onnx'
            path.write_bytes(b'ONNX loader contract fixture')
            settings = SimpleNamespace()
            session = SimpleNamespace(get_inputs=lambda: [SimpleNamespace(type='tensor(float)', shape=['batch', 3, 'height', 'width'], name='images')],
                                      get_modelmeta=lambda: SimpleNamespace(custom_metadata_map={'names': "{0: 'phone'}", 'stride': '32'}))
            constructor = Mock(return_value=session)
            ort = SimpleNamespace(SessionOptions=lambda: settings, ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL='sequential'),
                                  InferenceSession=constructor, __version__='mock')
            open_onnx_detector.cache_clear()
            try:
                with patch.dict(sys.modules, {'onnxruntime': ort}):
                    first = open_onnx_detector(str(path), digest(path), 1)
                    second = open_onnx_detector(str(path), digest(path), 1)
                    self.assertIs(first[0], second[0])
                    constructor.assert_called_once()
                    self.assertEqual(settings.intra_op_num_threads, 1)
                    self.assertEqual(settings.inter_op_num_threads, 1)
                    self.assertEqual(constructor.call_args.kwargs['providers'], ['CPUExecutionProvider'])
                    self.assertEqual(first[2], {0: 'phone'})
                    session.get_inputs = lambda: [SimpleNamespace(type='tensor(float)', shape=[1, 3, 640, 640], name='images')]
                    with self.assertRaises(ValueError):
                        open_onnx_detector(str(path), digest(path), 2)
            finally:
                open_onnx_detector.cache_clear()

    def test_tpu_only_profile_never_imports_torch_or_yolo(self):
        person = box('person', [10, 10, 100, 200])
        tpu = SimpleNamespace(detect=Mock(return_value={'actual_execution': True, 'detections': [person], 'provenance': {'adapter': 'mock TPU protocol'}}))
        with patch.dict(sys.modules, {'torch': None, 'ultralytics': None, 'deploy.linq_tpu': tpu}):
            result = detect(None, {}, {'linq_person': {'model': 'rfdetr'}, 'models': []})
        self.assertEqual(result['detections'], [person])
        self.assertIsNone(result['provenance']['ultralytics_version'])
        self.assertIsNone(result['signals']['phone_present'])
        self.assertIsNone(result['signals']['cigarette_present'])
        self.assertEqual(result['attribute_status']['phone'], 'not_evaluated_no_model')
        tpu.detect.assert_called_once()

    def test_phone_inside_person_kept_and_truck_logo_rejected_exactly(self):
        person = box('person', [10, 10, 110, 210])
        phone = box('phone', [20, 50, 35, 75])
        logo = box('cigarette', [300, 100, 320, 110], 0.91)
        original = [person, phone, logo]
        result, receipt = associate_attributes(original, {})
        self.assertEqual(result, [person, phone])
        self.assertEqual(receipt['rejected_candidates'][0]['candidate'], logo)
        self.assertEqual(receipt['rejected_candidates'][0]['reason'], 'outside_expanded_person')
        self.assertEqual(len(original), 3)

    def test_expanded_person_and_area_ratio_are_both_required(self):
        person = box('person', [100, 100, 200, 300])
        edge = box('phone', [88, 130, 102, 150])
        too_big = box('cigarette', [100, 100, 200, 280])
        result, receipt = associate_attributes([person, edge, too_big], {})
        self.assertEqual(result, [person, edge])
        self.assertEqual(receipt['rejected_candidates'][0]['reason'], 'attribute_too_large')

    def test_no_person_visibility_is_unknown_and_candidates_are_retained_in_provenance(self):
        candidate = box('cigarette', [10, 10, 20, 20])
        result, receipt = associate_attributes([candidate], {})
        self.assertEqual(result, [])
        self.assertEqual(receipt['attribute_status']['cigarette'], 'unknown_when_no_person')
        self.assertEqual(receipt['rejected_candidates'][0]['candidate'], candidate)
        self.assertEqual(receipt['rejected_candidates'][0]['reason'], 'no_person_detection')

    def test_disabled_association_keeps_original_objects(self):
        original = [box('phone', [10, 10, 20, 20])]
        result, receipt = associate_attributes(original, {'attribute_association': {'enabled': False}})
        self.assertEqual(result, original)
        self.assertEqual(receipt['attribute_status']['phone'], 'association_disabled')

    def test_per_model_confidence_and_nullable_absence_signals(self):
        torch = SimpleNamespace(set_num_threads=Mock())
        ultralytics = SimpleNamespace(__version__='mock')
        modules = {'torch': torch, 'ultralytics': ultralytics, 'ultralytics.utils': SimpleNamespace(torch_utils=SimpleNamespace())}
        specs = [{'checkpoint': 'phone.onnx', 'sha256': 'phone', 'labels': ['phone'],
                  'confidence': 0.45, 'backend': 'onnxruntime_cpu'},
                 {'checkpoint': 'cigarette.onnx', 'sha256': 'cigarette', 'labels': ['cigarette'],
                  'backend': 'rfdetr_onnx_cpu'}]
        options = {'confidence': 0.3, 'models': specs, 'device': 'cpu'}
        with patch.dict(sys.modules, modules), \
             patch('deploy.detector_adapter.predict_onnx', return_value=([], {'backend': 'onnxruntime_cpu'})) as phone, \
             patch('deploy.rfdetr_onnx.predict_rfdetr_onnx', return_value=([], {'backend': 'onnxruntime_cpu'})) as cigarette:
            actual = detect(None, {}, options)
        self.assertEqual(phone.call_args.args[3], 0.45)
        self.assertEqual(cigarette.call_args.args[3], 0.3)
        self.assertIsNone(actual['signals']['phone_present'])
        self.assertIsNone(actual['signals']['cigarette_present'])
        self.assertEqual(actual['signals']['person_present'], 0)

    def test_legacy_backend_is_rejected_before_remote_person_inference(self):
        tpu = SimpleNamespace(detect=Mock())
        options = {'linq_person': {'model': 'rfdetr'}, 'models': [
            {'checkpoint': 'legacy.pt', 'sha256': 'unused', 'labels': ['phone'], 'backend': 'ultralytics'}]}
        with patch.dict(sys.modules, {'deploy.linq_tpu': tpu}):
            with self.assertRaisesRegex(ValueError, 'backend must be explicitly configured'):
                detect(None, {}, options)
        tpu.detect.assert_not_called()

    def test_non_cpu_device_is_rejected_before_remote_person_inference(self):
        tpu = SimpleNamespace(detect=Mock())
        options = {'device': 'cuda:0', 'linq_person': {'model': 'rfdetr'}, 'models': [
            {'checkpoint': 'phone.onnx', 'sha256': 'unused', 'labels': ['phone'], 'backend': 'onnxruntime_cpu'}]}
        with patch.dict(sys.modules, {'deploy.linq_tpu': tpu}):
            with self.assertRaisesRegex(ValueError, 'require device=cpu'):
                detect(None, {}, options)
        tpu.detect.assert_not_called()


class AttributeRegionTests(unittest.TestCase):
    class Image:
        def __init__(self, size, origin=(0, 0)):
            self.size = size
            self.origin = origin
            self.crops = []

        def crop(self, bounds):
            self.crops.append(bounds)
            left, top, right, bottom = bounds
            return type(self)((right - left, bottom - top),
                              (self.origin[0] + left, self.origin[1] + top))

    def test_person_regions_expand_clamp_deduplicate_and_cap(self):
        people = [box('person', [-10, 10, 50, 90], .95),
                  box('person', [-8, 12, 49, 89], .85),
                  box('person', [100, 20, 150, 80], .75),
                  box('person', [170, 20, 171, 30], .99)]
        regions = person_regions(people, (200, 120), {'person_expand': .15, 'max_people': 2,
                                                       'min_side': 32, 'merge_iou': .5})
        self.assertEqual([region['crop_xyxy'] for region in regions], [[0, 0, 59, 102], [92, 11, 158, 89]])
        self.assertEqual([region['person_score'] for region in regions], [.95, .75])
        self.assertTrue(all(0 <= x1 < x2 <= 200 and 0 <= y1 < y2 <= 120
                            for region in regions
                            for x1, y1, x2, y2 in [region['crop_xyxy']]))

    def test_merge_is_class_aware_clips_bounds_and_keeps_top_score(self):
        merged = merge_region_detections(
            [box('phone', [10, 10, 30, 30], .5), box('phone', [11, 11, 31, 31], .9),
             box('cigarette', [11, 11, 31, 31], .7), box('phone', [-20, 0, 10, 20], .4)],
            (40, 40), .5, 10)
        self.assertEqual([(item['label'], item['score']) for item in merged],
                         [('phone', .9), ('cigarette', .7), ('phone', .4)])
        self.assertEqual(merged[-1]['xyxy'], [0.0, 0.0, 10.0, 20.0])

    def test_crop_boxes_are_projected_into_full_frame_and_receipted(self):
        image = self.Image((100, 80))
        person = box('person', [40, 20, 90, 70], .9)
        spec = {'name': 'phone', 'checkpoint': 'phone.onnx', 'sha256': 'a' * 64,
                'source_checkpoint_sha256': 'b' * 64, 'labels': ['phone']}
        options = {'attribute_regions': {'enabled': True, 'person_expand': 0, 'max_people': 6,
                                         'min_side': 32, 'merge_iou': .5}, 'max_det': 10}

        def predict(view, current_spec, current_options, confidence):
            if view.size == (100, 80):
                return [box('phone', [5, 5, 15, 15], .4)], {'elapsed_ms': 2, 'backend': 'onnxruntime_cpu'}
            return [box('phone', [1, 2, 6, 8], .8)], {'elapsed_ms': 3, 'backend': 'onnxruntime_cpu'}

        with patch('deploy.detector_adapter.predict_onnx', side_effect=predict) as call:
            detections, receipt = predict_onnx_regions(image, spec, options, .45, [person])
        self.assertEqual(call.call_count, 2)
        self.assertEqual(image.crops, [(40, 20, 90, 70)])
        self.assertEqual(detections, [box('phone', [41.0, 22.0, 46.0, 28.0], .8),
                                      box('phone', [5.0, 5.0, 15.0, 15.0], .4)])
        self.assertEqual([region['kind'] for region in receipt['regions']], ['full_frame', 'person_crop'])
        self.assertEqual(receipt['regions'][1]['crop_xyxy'], [40, 20, 90, 70])
        self.assertEqual(receipt['regions'][1]['child_model_receipt']['backend'], 'onnxruntime_cpu')
        self.assertGreaterEqual(receipt['region_elapsed_ms'], receipt['full_frame_elapsed_ms'])
        self.assertGreaterEqual(receipt['regions'][1]['elapsed_ms'], 0)

    def test_no_real_person_means_no_crop_inference(self):
        image = self.Image((100, 80))
        spec = {'name': 'phone', 'checkpoint': 'phone.onnx', 'sha256': 'a' * 64,
                'source_checkpoint_sha256': 'b' * 64, 'labels': ['phone']}
        with patch('deploy.detector_adapter.predict_onnx', return_value=([], {'elapsed_ms': 1})) as call:
            detections, receipt = predict_onnx_regions(image, spec, {'attribute_regions': {'enabled': True}}, .45, [])
        call.assert_called_once()
        self.assertEqual(detections, [])
        self.assertEqual([region['kind'] for region in receipt['regions']], ['full_frame'])

    def test_region_mode_disabled_preserves_single_full_frame_call(self):
        person = box('person', [10, 10, 90, 70], .9)
        tpu = SimpleNamespace(detect=Mock(return_value={'actual_execution': True, 'detections': [person],
                                                         'provenance': {'adapter': 'mock TPU'}}))
        torch_utils = SimpleNamespace(NUM_THREADS=1)
        modules = {'torch': SimpleNamespace(set_num_threads=Mock()),
                   'ultralytics': SimpleNamespace(__version__='mock'),
                   'ultralytics.utils': SimpleNamespace(torch_utils=torch_utils),
                   'deploy.linq_tpu': tpu}
        spec = {'name': 'phone', 'checkpoint': 'phone.onnx', 'sha256': 'a' * 64,
                'source_checkpoint_sha256': 'b' * 64, 'labels': ['phone'], 'backend': 'onnxruntime_cpu'}
        image = self.Image((100, 80))
        expected = [box('phone', [20, 30, 30, 40], .7)]
        with patch.dict(sys.modules, modules), \
             patch('deploy.detector_adapter.predict_onnx', return_value=(expected, {'elapsed_ms': 1})) as call:
            result = detect(image, {}, {'linq_person': {'model': 'rfdetr'}, 'models': [spec]})
        call.assert_called_once()
        self.assertEqual(result['detections'], [person, *expected])
        self.assertEqual(result['provenance']['models'][1]['elapsed_ms'], 1)

    def test_model_region_override_disables_rfdetr_roi_only(self):
        person = box('person', [10, 10, 90, 70], .9)
        tpu = SimpleNamespace(detect=Mock(return_value={'actual_execution': True, 'detections': [person],
                                                         'provenance': {'adapter': 'mock TPU'}}))
        torch_utils = SimpleNamespace(NUM_THREADS=1)
        modules = {'torch': SimpleNamespace(set_num_threads=Mock()),
                   'ultralytics': SimpleNamespace(__version__='mock'),
                   'ultralytics.utils': SimpleNamespace(torch_utils=torch_utils),
                   'deploy.linq_tpu': tpu}
        phone_spec = {'name': 'phone', 'checkpoint': 'phone.onnx', 'sha256': 'a' * 64,
                      'source_checkpoint_sha256': 'b' * 64, 'labels': ['phone'], 'backend': 'onnxruntime_cpu'}
        cigarette_spec = {'name': 'cigarette', 'checkpoint': 'cigarette.onnx', 'sha256': 'c' * 64,
                          'source_checkpoint_sha256': 'd' * 64, 'labels': ['cigarette'],
                          'backend': 'rfdetr_onnx_cpu', 'attribute_regions': {'enabled': False}}
        image = self.Image((100, 80))
        calls = []

        def phone_predictor(image_arg, spec, options, confidence):
            calls.append((spec['name'], image_arg.size,
                          options.get('attribute_regions', {}).get('enabled', False)))
            return [], {'elapsed_ms': 1}

        with patch.dict(sys.modules, modules), \
             patch('deploy.detector_adapter.predict_onnx', side_effect=phone_predictor), \
             patch('deploy.rfdetr_onnx.predict_rfdetr_onnx', side_effect=phone_predictor):
            detect(image, {}, {'linq_person': {'model': 'rfdetr'}, 'models': [phone_spec, cigarette_spec],
                               'attribute_regions': {'enabled': True, 'person_expand': 0}})

        self.assertEqual(calls, [('phone', (100, 80), True), ('phone', (80, 60), True),
                                 ('cigarette', (100, 80), False)])

    def test_rfdetr_backend_routes_through_local_runner(self):
        torch_utils = SimpleNamespace(NUM_THREADS=1)
        modules = {'torch': SimpleNamespace(set_num_threads=Mock()),
                   'ultralytics': SimpleNamespace(__version__='mock'),
                   'ultralytics.utils': SimpleNamespace(torch_utils=torch_utils)}
        spec = {'name': 'rfdetr-cigdet', 'checkpoint': 'cigarette-rfdetr.onnx',
                'sha256': 'a' * 64, 'source_checkpoint_sha256': 'b' * 64,
                'labels': ['cigarette'], 'backend': 'rfdetr_onnx_cpu'}
        output = [box('cigarette', [10, 20, 30, 40], .92)]
        receipt = {'backend': 'onnxruntime_cpu', 'architecture': 'RF-DETR-Large2026'}
        with patch.dict(sys.modules, modules), \
             patch('deploy.rfdetr_onnx.predict_rfdetr_onnx', return_value=(output, receipt)) as run:
            result = detect(self.Image((100, 80)), {},
                            {'models': [spec], 'attribute_association': {'enabled': False}})
        run.assert_called_once()
        self.assertEqual(result['detections'], output)
        self.assertEqual(result['provenance']['models'][0]['architecture'], 'RF-DETR-Large2026')


if __name__ == '__main__':
    unittest.main()
