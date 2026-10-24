import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from deploy.common import digest
from deploy.rfdetr_onnx import decode_rfdetr_outputs, open_rfdetr_onnx, predict_rfdetr_onnx


class RfDetrOnnxTests(unittest.TestCase):
    def test_decoder_uses_foreground_sigmoid_and_projects_normalized_boxes(self):
        boxes = np.array([[[.5, .5, .4, .2], [.1, .1, .4, .2], [1.1, .5, .4, .2]]], dtype=np.float32)
        logits = np.array([[[4.0, 12.0], [3.0, -10.0], [8.0, 2.0]]], dtype=np.float32)
        result = decode_rfdetr_outputs(boxes, logits, (100, 50), .7, max_det=3)
        self.assertEqual([item['label'] for item in result], ['cigarette', 'cigarette', 'cigarette'])
        self.assertAlmostEqual(result[0]['score'], 1 / (1 + np.exp(-8.0)), places=6)
        for actual, expected in zip(result[0]['xyxy'], [90.0, 20.0, 100.0, 30.0]):
            self.assertAlmostEqual(actual, expected, places=5)
        for actual, expected in zip(result[1]['xyxy'], [30.0, 20.0, 70.0, 30.0]):
            self.assertAlmostEqual(actual, expected, places=5)
        self.assertTrue(all(np.isfinite(item['xyxy']).all() for item in result))

    def test_decoder_rejects_nonfinite_or_wrong_graph_contract(self):
        boxes = np.zeros((1, 3, 4), dtype=np.float32)
        logits = np.zeros((1, 3, 2), dtype=np.float32)
        boxes[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            decode_rfdetr_outputs(boxes, logits, (64, 48), .3)
        with self.assertRaises(ValueError):
            decode_rfdetr_outputs(np.zeros((1, 3, 5), dtype=np.float32), logits, (64, 48), .3)

    def test_session_loader_validates_and_maps_named_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'model.onnx'
            path.write_bytes(b'RF-DETR model contract fixture')
            session = SimpleNamespace(
                get_inputs=lambda: [SimpleNamespace(name='image', type='tensor(float)', shape=[1, 3, 704, 704])],
                get_outputs=lambda: [SimpleNamespace(name='labels', type='tensor(float)', shape=[1, 300, 2]),
                                     SimpleNamespace(name='dets', type='tensor(float)', shape=[1, 300, 4])],
            )
            options = SimpleNamespace(intra_op_num_threads=None, inter_op_num_threads=None, execution_mode=None)
            constructor = Mock(return_value=session)
            fake_ort = SimpleNamespace(SessionOptions=lambda: options,
                                       ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL='sequential'),
                                       InferenceSession=constructor, __version__='mock')
            open_rfdetr_onnx.cache_clear()
            try:
                with patch.dict(sys.modules, {'onnxruntime': fake_ort}):
                    loaded = open_rfdetr_onnx(str(path), digest(path), 1)
                self.assertIs(loaded[0], session)
                self.assertEqual(loaded[2], {'labels': 0, 'dets': 1})
                self.assertEqual(loaded[3:5], ((704, 704), 300))
                self.assertEqual(options.intra_op_num_threads, 1)
                self.assertEqual(options.inter_op_num_threads, 1)
                self.assertEqual(constructor.call_args.kwargs['providers'], ['CPUExecutionProvider'])
            finally:
                open_rfdetr_onnx.cache_clear()

    def test_runner_maps_named_outputs_and_reports_sha_provider_and_timing(self):
        labels = np.zeros((1, 300, 2), dtype=np.float32)
        labels[0, 0] = [0.0, 12.0]
        labels[0, 1] = [5.0, -2.0]
        dets = np.zeros((1, 300, 4), dtype=np.float32)
        dets[0, 1] = [.2, .5, .2, .4]
        session = SimpleNamespace(
            get_providers=lambda: ['CPUExecutionProvider'],
            run=lambda *_args, **_kwargs: [labels, dets],
        )
        spec = {'name': 'rfdetr-cigdet', 'checkpoint': '/model.onnx', 'sha256': 'a' * 64,
                'source_checkpoint_sha256': 'b' * 64, 'labels': ['cigarette']}
        settings = {'threads': 1}
        with patch('deploy.rfdetr_onnx.open_rfdetr_onnx', return_value=(session, 'image',
                {'labels': 0, 'dets': 1}, (704, 704), 300, 'mock')), \
             patch('deploy.rfdetr_onnx.preprocess_rfdetr', return_value=np.zeros((1, 3, 704, 704), dtype=np.float32)):
            detections, receipt = predict_rfdetr_onnx(SimpleNamespace(size=(100, 50)), spec, settings, .7)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]['label'], 'cigarette')
        for actual, expected in zip(detections[0]['xyxy'], [10.0, 15.0, 30.0, 35.0]):
            self.assertAlmostEqual(actual, expected, places=5)
        self.assertAlmostEqual(detections[0]['score'], float(1 / (1 + np.exp(-5.0))), places=6)
        self.assertEqual(receipt['architecture'], 'RF-DETR-Large2026')
        self.assertEqual(receipt['backend'], 'onnxruntime_cpu')
        self.assertEqual(receipt['providers'], ['CPUExecutionProvider'])
        self.assertEqual(receipt['checkpoint_sha256'], 'a' * 64)
        self.assertEqual(receipt['origin_PT_sha256'], 'b' * 64)
        self.assertEqual(receipt['input_shape'], [1, 3, 704, 704])
        self.assertGreaterEqual(receipt['elapsed_ms'], 0)


if __name__ == '__main__':
    unittest.main()
