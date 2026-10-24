from functools import lru_cache
import ast
import math
import os
from pathlib import Path
import re
import time

from deploy.common import digest


def associate_attributes(detections, options):
    settings = {'enabled': True, 'labels': ['phone', 'cigarette'], 'person_expand': 0.15,
                'min_inside_fraction': 0.6, 'max_area_ratio': 0.25, **options.get('attribute_association', {})}
    labels = set(settings['labels'])
    expand, minimum, maximum = (float(settings[key]) for key in ['person_expand', 'min_inside_fraction', 'max_area_ratio'])
    if not all(math.isfinite(value) for value in [expand, minimum, maximum]) or expand < 0 or not 0 <= minimum <= 1 or not 0 < maximum <= 1:
        raise ValueError('Invalid attribute association limits')
    people = [(index, box) for index, box in enumerate(detections) if box['label'] == 'person']
    receipt = {'settings': settings, 'input_detection_count': len(detections), 'person_detection_count': len(people),
               'rejected_candidates': [], 'accepted_associations': [], 'threshold_source': 'Configured engineering filter; no accuracy measured'}
    statuses = {label: 'association_disabled' if not settings['enabled'] else 'person_association_applied' if people else 'unknown_when_no_person' for label in labels}
    receipt['attribute_status'] = statuses
    if not settings['enabled']:
        receipt['output_detection_count'] = len(detections)
        return detections, receipt
    kept = []
    for index, box in enumerate(detections):
        if box['label'] not in labels:
            kept.append(box)
            continue
        x1, y1, x2, y2 = box['xyxy']
        area = max(0, x2 - x1) * max(0, y2 - y1)
        candidates = []
        for person_index, person in people:
            px1, py1, px2, py2 = person['xyxy']
            width, height = max(0, px2 - px1), max(0, py2 - py1)
            person_area = width * height
            if area <= 0 or person_area <= 0:
                continue
            expanded = [px1 - expand * width, py1 - expand * height, px2 + expand * width, py2 + expand * height]
            overlap = max(0, min(x2, expanded[2]) - max(x1, expanded[0])) * max(0, min(y2, expanded[3]) - max(y1, expanded[1]))
            candidates.append({'person_detection_index': person_index, 'expanded_person_xyxy': expanded,
                               'inside_fraction': overlap / area, 'attribute_to_person_area_ratio': area / person_area})
        matches = [candidate for candidate in candidates if candidate['inside_fraction'] >= minimum and candidate['attribute_to_person_area_ratio'] <= maximum]
        if matches:
            kept.append(box)
            receipt['accepted_associations'].append({'attribute_detection_index': index, 'candidate': dict(box), 'matching_people': matches})
        else:
            reason = 'no_person_detection' if not people else 'invalid_attribute_box' if area <= 0 else 'attribute_too_large' if any(candidate['inside_fraction'] >= minimum for candidate in candidates) else 'outside_expanded_person'
            receipt['rejected_candidates'].append({'attribute_detection_index': index, 'candidate': dict(box), 'reason': reason, 'person_comparisons': candidates})
    receipt['output_detection_count'] = len(kept)
    return kept, receipt


@lru_cache(maxsize=8)
def open_onnx_detector(checkpoint, expected_sha256, threads):
    import onnxruntime as ort
    path = Path(checkpoint).resolve()
    if not path.is_file() or path.suffix != '.onnx' or digest(path) != expected_sha256:
        raise ValueError('A SHA-verified existing trained ONNX model is required')
    if type(threads) is not int or not 1 <= threads <= 2:
        raise ValueError('Explicit one or two ORT CPU threads required')
    settings = ort.SessionOptions()
    settings.intra_op_num_threads = threads
    settings.inter_op_num_threads = 1
    settings.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(str(path), sess_options=settings, providers=['CPUExecutionProvider'])
    inputs = session.get_inputs()
    if len(inputs) != 1 or inputs[0].type != 'tensor(float)' or len(inputs[0].shape) != 4:
        raise ValueError('ONNX detector requires one FP32 BCHW input')
    if any(type(value) is int for value in inputs[0].shape[2:]):
        raise ValueError('Dynamic spatial ONNX required to preserve native rectangular letterboxing')
    metadata = session.get_modelmeta().custom_metadata_map
    names = ast.literal_eval(metadata['names'])
    if isinstance(names, list):
        names = dict(enumerate(names))
    if not isinstance(names, dict) or not all(isinstance(value, str) for value in names.values()):
        raise ValueError('ONNX class-name metadata is missing')
    return session, inputs[0].name, {int(key): value for key, value in names.items()}, int(metadata.get('stride', 32)), ort.__version__


def predict_onnx(image, spec, options, confidence):
    import numpy as np
    import torch
    from ultralytics.data.augment import LetterBox
    from ultralytics.utils import nms, ops
    if options.get('device', 'cpu') != 'cpu':
        raise ValueError('ONNX adapter is explicitly CPU-only')
    origin_sha = spec.get('source_checkpoint_sha256')
    if not isinstance(origin_sha, str) or not re.fullmatch(r'[a-f0-9]{64}', origin_sha):
        raise ValueError('Explicit original trained PT SHA256 required for the ONNX detector')
    threads = int(options.get('threads', 1))
    before = open_onnx_detector.cache_info()
    started = time.perf_counter()
    session, input_name, names, stride, version = open_onnx_detector(spec['checkpoint'], spec['sha256'], threads)
    load_ms = (time.perf_counter() - started) * 1000
    cache_hit = open_onnx_detector.cache_info().hits > before.hits
    started = time.perf_counter()
    original = np.asarray(image.convert('RGB'))[:, :, ::-1]
    size = options.get('imgsz', 640)
    if type(size) is not int or size <= 0:
        raise ValueError('ONNX adapter requires an integer image size')
    prepared = LetterBox(new_shape=(size, size), auto=True, stride=stride)(image=original)
    tensor = np.ascontiguousarray(prepared[:, :, ::-1].transpose(2, 0, 1)[None]).astype(np.float32) / 255.0
    preprocess_ms = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    outputs = session.run(None, {input_name: tensor})
    inference_ms = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    if len(outputs) != 1 or outputs[0].ndim != 3 or outputs[0].shape[0] != 1 or outputs[0].shape[1] != 4 + len(names) or not np.isfinite(outputs[0]).all():
        raise ValueError('Expected finite YOLO one-to-many output [1,4+classes,N] matching native default branch')
    predictions = nms.non_max_suppression(torch.from_numpy(outputs[0]), confidence, options.get('iou', 0.5),
                                          max_det=int(options.get('max_det', 100)), end2end=False)[0].numpy().copy()
    if len(predictions):
        boxes = torch.from_numpy(predictions[:, :4].copy())
        predictions[:, :4] = ops.scale_boxes(tensor.shape[2:], boxes, original.shape[:2]).numpy()
    aliases, allowed = spec.get('aliases', {}), set(spec['labels'])
    detections = []
    for row in predictions:
        class_id = int(row[5])
        if class_id != row[5] or class_id not in names:
            raise ValueError('ONNX returned an invalid class index')
        label = aliases.get(names[class_id], names[class_id])
        if label in allowed:
            detections.append({'xyxy': row[:4].tolist(), 'score': float(row[4]), 'label': label})
    postprocess_ms = (time.perf_counter() - started) * 1000
    return detections, {'name': spec.get('name', Path(spec['checkpoint']).stem), 'checkpoint_sha256': spec['sha256'],
                        'origin_PT_sha256': origin_sha, 'origin_identity_source': 'Explicit model spec linked to saved export and parity receipts',
                        'native_head': 'one-to-many', 'export_nms': None, 'external_nms': True, 'imgsz_requested': size,
                        'labels': sorted(allowed), 'confidence': confidence, 'device_requested': 'cpu', 'device_actual': 'cpu',
                        'backend': 'onnxruntime_cpu', 'onnxruntime_version': version, 'providers': session.get_providers(),
                        'threads': threads, 'inter_op_threads': 1, 'input_shape': list(tensor.shape), 'precision': 'FP32',
                        'model_cache_hit': cache_hit, 'model_load_or_cache_lookup_ms': load_ms,
                        'stage_ms': {'preprocess': preprocess_ms, 'inference': inference_ms, 'postprocess': postprocess_ms},
                        'elapsed_ms': preprocess_ms + inference_ms + postprocess_ms,
                        'preprocessing': 'Official Ultralytics dynamic rectangular LetterBox; same resolution; BGR-to-RGB /255',
                        'postprocessing': 'Native default one-to-many head; official Ultralytics non_max_suppression and scale_boxes with unchanged confidence/IoU/max_det'}


def person_regions(person_detections, image_size, settings):
    """Build bounded, expanded crop windows from actual detected person boxes."""
    width, height = image_size
    expand = float(settings.get('person_expand', 0.15))
    max_people = settings.get('max_people', 6)
    min_side = settings.get('min_side', 32)
    merge_iou = float(settings.get('merge_iou', 0.5))
    if (not math.isfinite(expand) or expand < 0 or type(max_people) is not int or not 1 <= max_people <= 32
            or type(min_side) is not int or not 1 <= min_side <= 4096
            or not math.isfinite(merge_iou) or not 0 <= merge_iou <= 1):
        raise ValueError('Invalid attribute region settings')
    if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
        raise ValueError('Invalid image dimensions for attribute regions')

    candidates = []
    for person in person_detections:
        if not isinstance(person, dict) or person.get('label') != 'person':
            continue
        coords = person.get('xyxy')
        score = person.get('score')
        if (not isinstance(coords, (list, tuple)) or len(coords) != 4
                or not all(type(value) in {int, float} and math.isfinite(value) for value in coords)
                or type(score) not in {int, float} or not math.isfinite(score)):
            continue
        x1, y1, x2, y2 = map(float, coords)
        box_width, box_height = x2 - x1, y2 - y1
        if box_width <= 0 or box_height <= 0:
            continue
        crop = [max(0, math.floor(x1 - expand * box_width)),
                max(0, math.floor(y1 - expand * box_height)),
                min(width, math.ceil(x2 + expand * box_width)),
                min(height, math.ceil(y2 + expand * box_height))]
        if crop[2] - crop[0] < min_side or crop[3] - crop[1] < min_side:
            continue
        candidates.append({'crop_xyxy': crop, 'person_xyxy': [x1, y1, x2, y2], 'person_score': float(score)})

    candidates.sort(key=lambda item: item['person_score'], reverse=True)
    selected = []
    for candidate in candidates:
        if any(_box_iou(candidate['crop_xyxy'], other['crop_xyxy']) > merge_iou for other in selected):
            continue
        selected.append(candidate)
        if len(selected) >= max_people:
            break
    return selected


def _box_iou(first, second):
    ix1, iy1 = max(first[0], second[0]), max(first[1], second[1])
    ix2, iy2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_first = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    area_second = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    union = area_first + area_second - intersection
    return intersection / union if union > 0 else 0.0


def merge_region_detections(detections, image_size, iou_threshold, max_det):
    """Merge full-frame and projected crop boxes with score-ordered class-aware NMS."""
    width, height = image_size
    if not math.isfinite(float(iou_threshold)) or not 0 <= float(iou_threshold) <= 1:
        raise ValueError('Region merge IoU must be between zero and one')
    if type(max_det) is not int or max_det <= 0:
        raise ValueError('Region max_det must be a positive integer')
    valid = []
    for detection in detections:
        coords = detection.get('xyxy') if isinstance(detection, dict) else None
        score = detection.get('score') if isinstance(detection, dict) else None
        if (not isinstance(coords, (list, tuple)) or len(coords) != 4
                or not all(type(value) in {int, float} and math.isfinite(value) for value in coords)
                or type(score) not in {int, float} or not math.isfinite(score)
                or not isinstance(detection.get('label'), str)):
            continue
        clipped = [min(width, max(0.0, float(coords[0]))), min(height, max(0.0, float(coords[1]))),
                   min(width, max(0.0, float(coords[2]))), min(height, max(0.0, float(coords[3])))]
        if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
            continue
        valid.append({**detection, 'xyxy': clipped})
    valid.sort(key=lambda item: float(item['score']), reverse=True)
    kept = []
    for candidate in valid:
        if any(candidate['label'] == previous['label']
               and _box_iou(candidate['xyxy'], previous['xyxy']) > float(iou_threshold)
               for previous in kept):
            continue
        kept.append(candidate)
        if len(kept) >= max_det:
            break
    return kept


def predict_onnx_regions(image, spec, options, confidence, person_detections, predictor=None):
    """Run the normal full-frame model, then optional ONNX crops of real person detections."""
    total_started = time.perf_counter()
    predictor = predictor or predict_onnx
    detections, receipt = predictor(image, spec, options, confidence)
    settings = options.get('attribute_regions') or {}
    candidates = person_regions(person_detections, image.size, settings)
    regions = [{'kind': 'full_frame', 'crop_xyxy': [0, 0, image.size[0], image.size[1]],
                'person_xyxy': None, 'person_score': None, 'candidate_count': len(detections),
                'elapsed_ms': receipt['elapsed_ms'], 'child_model_receipt': dict(receipt)}]
    combined = list(detections)
    for region in candidates:
        x1, y1, x2, y2 = region['crop_xyxy']
        child_started = time.perf_counter()
        crop_detections, child_receipt = predictor(image.crop((x1, y1, x2, y2)), spec, options, confidence)
        for candidate in crop_detections:
            projected = dict(candidate)
            bx1, by1, bx2, by2 = projected['xyxy']
            projected['xyxy'] = [bx1 + x1, by1 + y1, bx2 + x1, by2 + y1]
            combined.append(projected)
        regions.append({'kind': 'person_crop', 'crop_xyxy': list(region['crop_xyxy']),
                        'person_xyxy': list(region['person_xyxy']), 'person_score': region['person_score'],
                        'candidate_count': len(crop_detections),
                        'elapsed_ms': (time.perf_counter() - child_started) * 1000,
                        'child_model_receipt': child_receipt})
    receipt['regions'] = regions
    receipt['region_settings'] = dict(settings)
    receipt['full_frame_detection_count'] = len(detections)
    receipt['merged_detection_count_before_association'] = len(combined)
    merged = merge_region_detections(combined, image.size, settings.get('merge_iou', 0.5),
                                     int(options.get('max_det', 100)))
    receipt['merged_detection_count'] = len(merged)
    receipt['region_elapsed_ms'] = sum(float(region['elapsed_ms']) for region in regions)
    receipt['full_frame_elapsed_ms'] = receipt['elapsed_ms']
    receipt['elapsed_ms'] = (time.perf_counter() - total_started) * 1000
    return merged, receipt


def detect(image, context, options):
    specifications = options.get('models', [])
    if not specifications and not options.get('linq_person'):
        raise ValueError('Configure at least one existing trained detector')
    supported_backends = {'onnxruntime_cpu', 'rfdetr_onnx_cpu'}
    for spec in specifications:
        if spec.get('backend') not in supported_backends:
            raise ValueError('Detector backend must be explicitly configured as onnxruntime_cpu or rfdetr_onnx_cpu')
    device = options.get('device', 'cpu')
    if specifications and device != 'cpu':
        raise ValueError('Configured ONNX detector backends require device=cpu')
    ultralytics_version = None
    if specifications:
        import ultralytics
        import torch
        from ultralytics.utils import torch_utils
        torch_utils.NUM_THREADS = int(options.get('threads', 1))
        torch.set_num_threads(torch_utils.NUM_THREADS)
        ultralytics_version = ultralytics.__version__
    detections, receipts = [], []
    region_people = []
    started = time.perf_counter()
    if options.get('linq_person'):
        from deploy.linq_tpu import detect as tpu_detect
        response = tpu_detect(image, context, options['linq_person'])
        if response.get('actual_execution') is not True:
            raise RuntimeError('LinQ detector did not execute')
        tpu_detections = response['detections']
        detections.extend(tpu_detections)
        region_people = [item for item in tpu_detections if isinstance(item, dict) and item.get('label') == 'person']
        receipts.append(response['provenance'])
    for spec in specifications:
        confidence = float(spec.get('confidence', options.get('confidence', 0.25)))
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError('Detector confidence must be between zero and one')
        backend = spec['backend']
        if backend in {'onnxruntime_cpu', 'rfdetr_onnx_cpu'}:
            predictor = predict_onnx
            if backend == 'rfdetr_onnx_cpu':
                from deploy.rfdetr_onnx import predict_rfdetr_onnx
                predictor = predict_rfdetr_onnx
            global_region_settings = options.get('attribute_regions') or {}
            model_region_override = spec.get('attribute_regions', {})
            if not isinstance(global_region_settings, dict) or not isinstance(model_region_override, dict):
                raise ValueError('attribute_regions must be a configuration object')
            region_settings = {**global_region_settings, **model_region_override}
            region_enabled = region_settings.get('enabled', False)
            if type(region_enabled) is not bool:
                raise ValueError('attribute_regions.enabled must be an explicit boolean')
            model_options = {**options, 'attribute_regions': region_settings}
            if region_enabled and set(spec['labels']) & {'phone', 'cigarette'}:
                boxes, receipt = predict_onnx_regions(image, spec, model_options, confidence, region_people, predictor)
            else:
                boxes, receipt = predictor(image, spec, model_options, confidence)
            detections.extend(boxes)
            receipts.append(receipt)
            continue
    detections, association = associate_attributes(detections, options)
    evaluated_labels = {label for spec in specifications for label in spec['labels']}
    for label in association['attribute_status']:
        if label not in evaluated_labels:
            association['attribute_status'][label] = 'not_evaluated_no_model'
    signals = {label + '_present': max((box['score'] for box in detections if box['label'] == label), default=0.0)
               for label in {'phone', 'cigarette', 'person'}}
    for label, status in association['attribute_status'].items():
        if status in {'unknown_when_no_person', 'not_evaluated_no_model'}:
            signals[label + '_present'] = None
    return {'actual_execution': True, 'mode': 'fresh_inference', 'detections': detections,
            'signals': signals, 'attribute_status': association['attribute_status'], 'provenance': {'adapter': 'ready_trained_yolo_detectors',
            'attribute_association': association,
            'ultralytics_version': ultralytics_version, 'models': receipts,
            'process_id': os.getpid(),
            'elapsed_ms': (time.perf_counter() - started) * 1000,
            'signal_semantics': 'Object presence only; no smoking or unsafe phone-use verdict',
            'TPU_execution_verified': bool(options.get('linq_person')),
            'TPU_execution_scope': 'person detector only' if options.get('linq_person') else None}}
