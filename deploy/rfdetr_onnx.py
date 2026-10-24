"""Small CPU-only runner for the verified RF-DETR cigarette ONNX export.

The export uses Roboflow RF-DETR's official RGB resize and ImageNet normalization
contract, followed by independent sigmoid scores and normalized cxcywh decoding.
This module only supports the released one-foreground-class CigDet graph.
"""

from functools import lru_cache
import math
from pathlib import Path
import re
import time

import numpy as np

from deploy.common import digest


@lru_cache(maxsize=4)
def open_rfdetr_onnx(checkpoint, expected_sha256, threads):
    import onnxruntime as ort

    path = Path(checkpoint).resolve()
    if not path.is_file() or path.suffix != '.onnx' or digest(path) != expected_sha256:
        raise ValueError('A SHA-verified RF-DETR ONNX model is required')
    if not isinstance(expected_sha256, str) or not re.fullmatch(r'[a-f0-9]{64}', expected_sha256):
        raise ValueError('RF-DETR ONNX SHA256 must be explicit')
    if type(threads) is not int or not 1 <= threads <= 2:
        raise ValueError('Explicit one or two ORT CPU threads required')
    settings = ort.SessionOptions()
    settings.intra_op_num_threads = threads
    settings.inter_op_num_threads = 1
    settings.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    with np.errstate(all='ignore'):
        session = ort.InferenceSession(str(path), sess_options=settings, providers=['CPUExecutionProvider'])
    inputs, outputs = session.get_inputs(), session.get_outputs()
    if len(inputs) != 1 or inputs[0].type != 'tensor(float)':
        raise ValueError('RF-DETR ONNX requires one FP32 input')
    shape = inputs[0].shape
    if (len(shape) != 4 or shape[0] != 1 or shape[1] != 3
            or any(type(value) is not int or value <= 0 for value in shape[2:])):
        raise ValueError('RF-DETR graph must have static FP32 [1,3,H,W] input')
    output_by_name = {item.name: item for item in outputs}
    if set(output_by_name) != {'dets', 'labels'}:
        raise ValueError('RF-DETR graph must expose named dets and labels outputs')
    dets_shape, labels_shape = output_by_name['dets'].shape, output_by_name['labels'].shape
    if (len(dets_shape) != 3 or dets_shape[0] != 1 or dets_shape[2] != 4
            or len(labels_shape) != 3 or labels_shape[0] != 1 or labels_shape[2] != 2
            or dets_shape[1] != labels_shape[1] or any(type(value) is not int or value <= 0
                                                      for value in (*dets_shape, *labels_shape))):
        raise ValueError('RF-DETR outputs must be dets [1,Q,4] and labels [1,Q,2]')
    if any(item.type != 'tensor(float)' for item in outputs):
        raise ValueError('RF-DETR dets and labels outputs must be FP32')
    return session, inputs[0].name, {name: index for index, name in enumerate([item.name for item in outputs])}, \
        (shape[2], shape[3]), dets_shape[1], ort.__version__


def preprocess_rfdetr(image, height, width):
    """Match RF-DETR predict preprocessing: tensorize, resize without AA, normalize."""
    import torch
    import torchvision.transforms.functional as functional

    with torch.no_grad():
        image_tensor = functional.to_tensor(image.convert('RGB'))
        image_tensor = functional.resize(image_tensor, [height, width], antialias=False)
        image_tensor = functional.normalize(image_tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        tensor = image_tensor.unsqueeze(0).cpu().numpy()
    tensor = np.asarray(tensor, dtype=np.float32)
    if tensor.shape != (1, 3, height, width) or not np.isfinite(tensor).all():
        raise ValueError('RF-DETR preprocessing returned invalid FP32 NCHW input')
    return tensor


def decode_rfdetr_outputs(dets, labels, image_size, threshold, max_det=300):
    """Decode one-class RF-DETR outputs using official sigmoid/background/top-k semantics."""
    dets, labels = np.asarray(dets), np.asarray(labels)
    if dets.ndim == 3 and dets.shape[0] == 1:
        dets = dets[0]
    if labels.ndim == 3 and labels.shape[0] == 1:
        labels = labels[0]
    if (dets.ndim != 2 or dets.shape[1] != 4 or labels.ndim != 2 or labels.shape[1] != 2
            or dets.shape[0] != labels.shape[0] or dets.shape[0] == 0
            or not np.isfinite(dets).all() or not np.isfinite(labels).all()):
        raise ValueError('Expected finite RF-DETR dets [Q,4] and labels [Q,2] tensors')
    if not math.isfinite(float(threshold)) or not 0 <= float(threshold) <= 1:
        raise ValueError('RF-DETR confidence must be between zero and one')
    if type(max_det) is not int or not 1 <= max_det <= dets.shape[0]:
        raise ValueError('RF-DETR query cap must be in [1,Q]')
    width, height = image_size
    if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
        raise ValueError('Invalid source image dimensions')

    # The trained graph has class slot 0 = cigarette and slot 1 = background.
    # RF-DETR's official postprocess uses independent sigmoid, excludes background,
    # globally ranks query/class pairs, takes the query cap, then applies strict >.
    foreground_logits = np.clip(labels[:, 0], -88, 88)
    scores = 1.0 / (1.0 + np.exp(-foreground_logits))
    selected = np.lexsort((np.arange(len(scores)), -scores))[:max_det]
    selected = selected[scores[selected] > float(threshold)]
    boxes = dets[selected].astype(np.float32, copy=False)
    cx, cy, bw, bh = boxes.T
    xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
    scale = np.asarray([width, height, width, height], dtype=np.float32)
    xyxy = np.clip(xyxy * scale, 0.0, scale)
    decoded = []
    for coords, score in zip(xyxy, scores[selected]):
        if coords[2] > coords[0] and coords[3] > coords[1]:
            decoded.append({'xyxy': coords.tolist(), 'score': float(score), 'label': 'cigarette'})
    return decoded


def predict_rfdetr_onnx(image, spec, options, confidence):
    """Run the local RF-DETR ONNX asset with strict graph and source-hash checks."""
    threads = int(options.get('threads', 1))
    before = open_rfdetr_onnx.cache_info()
    started = time.perf_counter()
    session, input_name, output_indices, (height, width), query_count, version = open_rfdetr_onnx(
        spec['checkpoint'], spec['sha256'], threads)
    load_ms = (time.perf_counter() - started) * 1000
    after = open_rfdetr_onnx.cache_info()
    cache_hit = after.hits > before.hits if type(before.hits) is int and type(after.hits) is int else None
    if 'cigarette' not in set(spec.get('labels', [])):
        raise ValueError('This RF-DETR export only supports the cigarette label')
    original_sha = spec.get('source_checkpoint_sha256')
    if not isinstance(original_sha, str) or not re.fullmatch(r'[a-f0-9]{64}', original_sha):
        raise ValueError('Explicit trained RF-DETR checkpoint SHA256 required')

    began = time.perf_counter()
    tensor = preprocess_rfdetr(image, height, width)
    preprocess_ms = (time.perf_counter() - began) * 1000
    began = time.perf_counter()
    raw = session.run(None, {input_name: tensor})
    inference_ms = (time.perf_counter() - began) * 1000
    if len(raw) != 2:
        raise ValueError('RF-DETR graph must return exactly dets and labels')
    dets, labels = raw[output_indices['dets']], raw[output_indices['labels']]
    if dets.shape != (1, query_count, 4) or labels.shape != (1, query_count, 2):
        raise ValueError('RF-DETR runtime output shapes differ from the verified graph contract')
    if not np.isfinite(dets).all() or not np.isfinite(labels).all():
        raise ValueError('RF-DETR graph returned non-finite outputs')
    began = time.perf_counter()
    detections = decode_rfdetr_outputs(dets, labels, image.size, confidence, query_count)
    postprocess_ms = (time.perf_counter() - began) * 1000
    return detections, {'name': spec.get('name', Path(spec['checkpoint']).stem),
                        'checkpoint_sha256': spec['sha256'], 'origin_PT_sha256': original_sha,
                        'origin_identity_source': 'Original checkpoint SHA256 is configured metadata; this runtime does not verify export or parity receipts',
                        'architecture': 'RF-DETR-Large2026', 'backend': 'onnxruntime_cpu',
                        'onnxruntime_version': version, 'providers': session.get_providers(),
                        'device_requested': 'cpu', 'device_actual': 'cpu', 'threads': threads,
                        'input_shape': list(tensor.shape), 'precision': 'FP32', 'query_count': query_count,
                        'class_contract': {'0': 'cigarette', '1': 'background'},
                        'score_semantics': 'Independent sigmoid; background excluded; top queries then strict confidence threshold',
                        'confidence': confidence, 'model_cache_hit': cache_hit,
                        'model_load_or_cache_lookup_ms': load_ms,
                        'stage_ms': {'preprocess': preprocess_ms, 'inference': inference_ms, 'postprocess': postprocess_ms},
                        'elapsed_ms': preprocess_ms + inference_ms + postprocess_ms,
                        'preprocessing': 'Official RF-DETR torchvision tensorize, fixed resize antialias=False, ImageNet normalization'}
