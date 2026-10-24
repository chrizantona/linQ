"""Small runtime extraction of the original LinQ detector decoder helpers.

Source: detection-benchmark/scripts/benchmark_models.py
Original file SHA256: 0ae8e28fd569484a9a4ab0abbab75952bcf2c51b4d4802e964df87bd284db502
The extraction omits benchmark, dataset, reporting, and CLI code. Keep the
preprocessing and decode operations aligned with that source for model parity.
"""

import numpy as np


def bilinear_half_pixel(rgb: np.ndarray, height: int, width: int) -> np.ndarray:
    """Float32 bilinear resize without antialiasing, using half-pixel coordinates."""
    source = rgb.astype(np.float32) / np.float32(255.0)
    ih, iw = source.shape[:2]
    y = np.maximum((np.arange(height, dtype=np.float32) + .5) * (ih / height) - .5, 0)
    x = np.maximum((np.arange(width, dtype=np.float32) + .5) * (iw / width) - .5, 0)
    y0, x0 = np.floor(y).astype(np.int64), np.floor(x).astype(np.int64)
    y1, x1 = np.minimum(y0 + 1, ih - 1), np.minimum(x0 + 1, iw - 1)
    wy, wx = (y - y0).astype(np.float32)[:, None, None], (x - x0).astype(np.float32)[None, :, None]
    top = source[y0[:, None], x0[None, :]] * (1 - wx) + source[y0[:, None], x1[None, :]] * wx
    bottom = source[y1[:, None], x0[None, :]] * (1 - wx) + source[y1[:, None], x1[None, :]] * wx
    return top * (1 - wy) + bottom * wy


def preprocess(rgb: np.ndarray, model_type: str, height: int, width: int):
    original_h, original_w = rgb.shape[:2]
    if model_type == "rfdetr":
        array = bilinear_half_pixel(rgb, height, width)
        array = (array - np.array([.485, .456, .406], dtype=np.float32)) / np.array([.229, .224, .225], dtype=np.float32)
        transform = {"original_hw": (original_h, original_w)}
    else:
        import cv2
        ratio = min(height / original_h, width / original_w)
        resized_w, resized_h = max(round(original_w * ratio), 1), max(round(original_h * ratio), 1)
        left, top = round((width - resized_w) / 2 - .1), round((height - resized_h) / 2 - .1)
        array = np.full((height, width, 3), 114, dtype=np.uint8)
        resized = cv2.resize(rgb, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
        array[top:top + resized_h, left:left + resized_w] = resized
        array = array.astype(np.float32) / np.float32(255.0)
        transform = {"original_hw": (original_h, original_w), "pad_xy": (left, top),
                     "gain_xy": (resized_w / original_w, resized_h / original_h)}
    return np.ascontiguousarray(array.transpose(2, 0, 1)[None], dtype=np.float32), transform


def xywh_to_xyxy(boxes):
    result = np.empty_like(boxes, dtype=np.float32)
    result[:, :2] = boxes[:, :2] - boxes[:, 2:] / 2
    result[:, 2:] = boxes[:, :2] + boxes[:, 2:] / 2
    return result


def nms(boxes, scores, iou_threshold, maximum):
    order = np.argsort(-scores, kind="stable")
    areas = np.maximum(boxes[:, 2] - boxes[:, 0], 0) * np.maximum(boxes[:, 3] - boxes[:, 1], 0)
    keep = []
    while len(order) and len(keep) < maximum:
        chosen = order[0]
        keep.append(chosen)
        rest = order[1:]
        intersection = np.maximum(np.minimum(boxes[chosen, 2:], boxes[rest, 2:]) - np.maximum(boxes[chosen, :2], boxes[rest, :2]), 0)
        intersection = intersection[:, 0] * intersection[:, 1]
        union = areas[chosen] + areas[rest] - intersection
        overlap = intersection / np.maximum(union, np.finfo(np.float32).eps)
        order = rest[overlap <= iou_threshold]
    return np.asarray(keep, dtype=np.int64)


def decode_yolo(outputs: dict, transform: dict, person_index: int, confidence: float, iou: float, maximum: int):
    if len(outputs) != 1:
        raise ValueError(f"Expected one raw YOLOv8 output, got {list(outputs)}")
    raw = np.asarray(next(iter(outputs.values())))
    if raw.ndim != 3 or raw.shape[0] != 1 or raw.shape[1] != 84:
        raise ValueError(f"Expected official 80-class YOLOv8 raw [1,84,N], got {raw.shape}; embedded-NMS/custom-head exports are unsupported")
    predictions = raw[0].T
    if not 0 <= person_index < 80:
        raise ValueError("YOLO person class index must be in0..79")
    scores = predictions[:, 4 + person_index]
    valid = np.isfinite(predictions).all(axis=1) & (scores > confidence)
    boxes, scores = xywh_to_xyxy(predictions[valid, :4]), scores[valid]
    chosen = nms(boxes, scores, iou, maximum)
    boxes, scores = boxes[chosen], scores[chosen]
    px, py = transform["pad_xy"]
    gx, gy = transform["gain_xy"]
    boxes = (boxes - np.array([px, py, px, py], dtype=np.float32)) / np.array([gx, gy, gx, gy], dtype=np.float32)
    return clip_boxes(boxes, transform["original_hw"]), scores


def decode_rfdetr(outputs: dict, transform: dict, person_index: int, confidence: float, maximum: int,
                  background_index: int | None = None):
    box_keys = [name for name in outputs if "dets" in name or "pred_boxes" in name]
    logit_keys = [name for name in outputs if "labels" in name or "pred_logits" in name]
    if len(box_keys) != 1 or len(logit_keys) != 1:
        raise ValueError(f"Expected named RF-DETR dets/labels outputs, got {list(outputs)}")
    boxes, logits = np.asarray(outputs[box_keys[0]]), np.asarray(outputs[logit_keys[0]])
    if boxes.ndim != 3 or logits.ndim != 3 or boxes.shape[0] != 1 or logits.shape[0] != 1 or boxes.shape[2] != 4 or boxes.shape[1] != logits.shape[1]:
        raise ValueError(f"Unexpected RF-DETR outputs: {boxes.shape}, {logits.shape}")
    boxes, logits = boxes[0], logits[0].astype(np.float32)
    if not np.isfinite(boxes).all() or not np.isfinite(logits).all():
        raise ValueError("RF-DETR produced NaN/Inf")
    if not 0 <= person_index < logits.shape[1]:
        raise ValueError(f"person class index {person_index} outside {logits.shape[1]} exported slots")
    scores = 1.0 / (1.0 + np.exp(-np.clip(logits, -80, 80)))
    if background_index is not None:
        if not -logits.shape[1] <= background_index < logits.shape[1]:
            raise ValueError("background class index outside exported slots")
        background_index %= logits.shape[1]
        if background_index == person_index:
            raise ValueError("Person class index cannot be the background slot")
        scores[:, background_index] = -np.inf
    flat = scores.ravel()
    cap = min(logits.shape[0], flat.size)
    order = np.argsort(-flat, kind="stable")[:cap]
    query_ids, classes = np.divmod(order, logits.shape[1])
    selected = (classes == person_index) & (flat[order] > confidence)
    query_ids, confidences = query_ids[selected][:maximum], flat[order][selected][:maximum]
    pixel_boxes = xywh_to_xyxy(boxes[query_ids])
    oh, ow = transform["original_hw"]
    return clip_boxes(pixel_boxes * np.array([ow, oh, ow, oh], dtype=np.float32), (oh, ow)), confidences


def clip_boxes(boxes, original_hw):
    height, width = original_hw
    return np.clip(boxes, 0, np.array([width, height, width, height], dtype=np.float32))
