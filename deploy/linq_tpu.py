import ast
from functools import lru_cache
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import re
import shutil
import time
import uuid
import zipfile

import numpy as np


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def connection_constants(notebook):
    document = json.loads(Path(notebook).read_text())
    if "cells" not in document:
        document = document["content"]
    source = document["cells"][1]["source"]
    source = source if isinstance(source, str) else "".join(source)
    required = {"RABBITMQ_HOST", "AMQP_PORT", "CREDENTIALS", "EXCHANGE", "INPUT_QUEUE"}
    values = {}
    for statement in ast.parse(source).body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name):
            name = statement.targets[0].id
            if name in required:
                values[name] = ast.literal_eval(statement.value)
    if set(values) != required:
        raise ValueError("Provided connection notebook lacks the actual service constants")
    return values


def program_contract(path):
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo("metadata.json")
        if info.file_size > 32 * 1024 * 1024:
            raise ValueError("Unexpected TPU metadata size")
        metadata = json.loads(archive.read(info))
    def flatten(value):
        result = {}
        for name, item in value.items():
            if "user_shape" in item:
                result[name] = {"shape": item["user_shape"], "dtype": item["user_dtype"], "anchor": item.get("anchor", name)}
            else:
                result.update(flatten(item))
        return result
    return {"inputs": flatten(metadata["inputs"]), "outputs": flatten(metadata["outputs"])}


def stage_program(path, expected_sha256, login):
    path = Path(path)
    if digest(path) != expected_sha256:
        raise ValueError("Compiled program differs from its recorded SHA256")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", login) or login != Path.home().name:
        raise ValueError("Only the current user's service cache is allowed")
    root = Path("/auto/cache").resolve(strict=True)
    cache = root / login
    if cache.is_symlink() or not cache.is_dir() or cache.resolve().parent != root:
        raise ValueError("Own official LinQ cache is unavailable")
    target = cache / ("linq_safety_proto_" + expected_sha256 + ".tpu")
    if target.is_symlink():
        raise ValueError("Refusing a symlink at the cache destination")
    if target.exists():
        if digest(target) != expected_sha256:
            raise ValueError("Existing cache artifact differs")
        return target
    temporary = cache / (".linq_safety_proto_" + str(uuid.uuid4()) + ".tmp")
    try:
        with path.open("rb") as source, temporary.open("xb") as destination:
            shutil.copyfileobj(source, destination)
        if digest(temporary) != expected_sha256:
            raise ValueError("Program staging failed checksum validation")
        try:
            os.link(temporary, target)
        except FileExistsError:
            if digest(target) != expected_sha256:
                raise ValueError("Concurrent program cache artifact differs")
    finally:
        temporary.unlink(missing_ok=True)
    return target


def infer(program, expected_sha256, inputs, notebook=None, timeout=60):
    import pika
    import tpu_pycloud_format as protocol
    logging.getLogger("pika").setLevel(logging.CRITICAL)
    login = Path.home().name
    notebook = notebook or Path.home() / "work/examples/example.ipynb"
    constants = connection_constants(notebook)
    contract = program_contract(program)
    target = stage_program(program, expected_sha256, login)
    wire_inputs = {}
    for name, value in inputs.items():
        matches = [key for key, item in contract["inputs"].items() if key == name or item["anchor"] == name]
        if len(matches) != 1:
            raise ValueError("Input tensor name does not uniquely match the real TPU contract")
        key = matches[0]
        array = np.ascontiguousarray(value, dtype=np.dtype(contract["inputs"][key]["dtype"]))
        if list(array.shape) != contract["inputs"][key]["shape"] or not np.isfinite(array).all():
            raise ValueError("Input tensor violates the actual TPU shape/dtype contract")
        wire_inputs[key] = array
    if set(wire_inputs) != set(contract["inputs"]):
        raise ValueError("TPU input tensors are incomplete")
    request_id = uuid.uuid4().hex
    reply_queue = "linq_safety_proto_" + login + "_" + request_id
    started = time.perf_counter()
    connection = pika.BlockingConnection(pika.ConnectionParameters(constants["RABBITMQ_HOST"], constants["AMQP_PORT"], credentials=pika.PlainCredentials(*constants["CREDENTIALS"]), socket_timeout=10, blocked_connection_timeout=15, stack_timeout=20, connection_attempts=1))
    try:
        channel = connection.channel()
        channel.queue_declare(queue=reply_queue, exclusive=True, auto_delete=True, arguments={"x-expires": 120000})
        channel.queue_bind(exchange=constants["EXCHANGE"], queue=reply_queue, routing_key=reply_queue)
        channel.basic_publish(exchange=constants["EXCHANGE"], routing_key=constants["INPUT_QUEUE"], body=protocol.serialize(protocol.PyTask(wire_inputs)), properties=pika.BasicProperties(headers={protocol.TARGET_URI_ATTR_NAME: "file://" + str(target) + "?raw=False", protocol.RESPONSE_ROUTING_KEYS_ATTR_NAME: json.dumps([reply_queue]), protocol.MESSAGE_ID_ATTR_NAME: request_id}))
        for method, properties, body in channel.consume(reply_queue, auto_ack=True, inactivity_timeout=1):
            if time.perf_counter() - started > timeout:
                raise TimeoutError("No matching LinQ TPU service response before deadline")
            if method is None or (properties.headers or {}).get(protocol.MESSAGE_ID_ATTR_NAME) != request_id:
                continue
            data, metadata, worker = protocol.deserialize(body)
            if not isinstance(worker, dict) or worker.get("status_code") != 0:
                raise RuntimeError("LinQ worker returned a nonzero status_code")
            if not isinstance(data, dict) or set(data) != set(contract["outputs"]):
                raise ValueError("LinQ output tensor names differ from the TPU program")
            outputs = {}
            for name, item in contract["outputs"].items():
                array = np.asarray(data[name])
                if list(array.shape) != item["shape"] or str(array.dtype) != item["dtype"] or not np.isfinite(array).all():
                    raise ValueError("LinQ output violates the real TPU shape/dtype/finite contract")
                anchor = item["anchor"]
                outputs[anchor if anchor not in outputs else name] = array
            receipt = {"status": "received_and_validated", "linq_service_inference_executed": True, "local_TPU_device_used": False, "transport": "AMQP_pika_tpu_pycloud_format", "request_id": request_id, "program_sha256": expected_sha256, "request_seconds": time.perf_counter() - started, "worker_status_code": worker["status_code"], "worker_inference_time_raw": worker.get("inference_time"), "outputs": {name: {"shape": list(array.shape), "dtype": str(array.dtype), "finite": True} for name, array in outputs.items()}}
            return outputs, receipt
        raise RuntimeError("LinQ response consumer ended before a matching result")
    finally:
        connection.close()


def load_person_detector(model="rfdetr", models_root=None, confidence=.25):
    """Load a SHA-bound compiled LinQ detector and its repo-local decoder.

    ``models_root`` is the folder containing ``<model>/report.json`` and the
    shared ``decoder.py``. Relative values are resolved from the repository
    root, so inference does not depend on the caller's working directory.
    """
    if not isinstance(model, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", model):
        raise ValueError("A simple LinQ model name is required")
    if model not in {"rfdetr", "yolov8"}:
        raise ValueError("Supported LinQ person detectors are rfdetr and yolov8")
    if not np.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("Detector confidence must be between zero and one")
    repository = Path(__file__).resolve().parents[1]
    root = Path(models_root) if models_root is not None else Path("models/linq")
    if not root.is_absolute():
        root = repository / root
    root = root.resolve()
    model_root = (root / model).resolve()
    if not model_root.is_relative_to(root):
        raise ValueError("LinQ model directory must stay under models_root")
    report_path = model_root / "report.json"
    decoder_path = root / "decoder.py"
    if not report_path.is_file() or not decoder_path.is_file():
        raise FileNotFoundError("Expected models/linq/<model>/report.json and models/linq/decoder.py")
    report = json.loads(report_path.read_text())
    models = report.get("models")
    if not isinstance(models, list):
        raise ValueError("LinQ compile report must contain a models list")
    matches = [item for item in models if isinstance(item, dict) and item.get("name") == model and item.get("status") == "compiled"]
    if len(matches) != 1:
        raise ValueError("Compile report must contain exactly one compiled entry for the requested model")
    selected = matches[0]
    program_spec = selected.get("tpu_program")
    if not isinstance(program_spec, dict) or not isinstance(program_spec.get("path"), str):
        raise ValueError("Compiled model entry lacks a TPU program path")
    program = (model_root / program_spec["path"]).resolve()
    if not program.is_relative_to(model_root) or not program.is_file():
        raise ValueError("TPU program must be an existing file inside its model directory")
    expected_sha = program_spec.get("sha256")
    if not isinstance(expected_sha, str) or not re.fullmatch(r"[a-f0-9]{64}", expected_sha) or digest(program) != expected_sha:
        raise ValueError("Compiled TPU program does not match its recorded SHA256")
    input_names, input_shapes = selected.get("input_names"), selected.get("input_shapes")
    if not isinstance(input_names, list) or len(input_names) != 1 or not isinstance(input_names[0], str):
        raise ValueError("Detector must have exactly one named input")
    input_name = input_names[0]
    shape = input_shapes.get(input_name) if isinstance(input_shapes, dict) else None
    if not isinstance(shape, list) or len(shape) != 4 or any(type(value) is not int or value <= 0 for value in shape):
        raise ValueError("Detector input shape must be a static positive BCHW shape")
    decode = selected.get("host_decode")
    if not isinstance(decode, dict):
        raise ValueError("Compiled model entry lacks host decode settings")
    specification = importlib.util.spec_from_file_location("linq_repo_decoder", decoder_path)
    if specification is None or specification.loader is None:
        raise ImportError("Could not load the repo-local LinQ decoder")
    decoder = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(decoder)
    required = {"preprocess", "decode_rfdetr" if model == "rfdetr" else "decode_yolo"}
    if not all(callable(getattr(decoder, name, None)) for name in required):
        raise ValueError("Repo-local decoder does not implement the requested model contract")
    return {"program": program, "sha256": expected_sha, "input_name": input_name,
            "shape": shape, "decode": decode, "decoder": decoder, "model": model,
            "confidence": float(confidence), "notebook": Path.home() / "work/examples/example.ipynb",
            "decoder_sha256": digest(decoder_path), "compile_report_sha256": digest(report_path)}


def detect_person(detector, rgb, timeout=60):
    decoder = detector["decoder"]
    tensor, transform = decoder.preprocess(np.asarray(rgb), detector["model"], detector["shape"][-2], detector["shape"][-1])
    arrays, receipt = infer(detector["program"], detector["sha256"], {detector["input_name"]: tensor}, detector["notebook"], timeout)
    settings = detector["decode"]
    if detector["model"] == "yolov8":
        boxes, scores = decoder.decode_yolo(arrays, transform, settings["person_class_index"], detector["confidence"], settings["nms_iou_yolov8"], settings["max_detections"])
    else:
        boxes, scores = decoder.decode_rfdetr(arrays, transform, settings["person_class_index"], detector["confidence"], settings["max_detections"], settings["rfdetr_background_index"])
    return {"detections": [{"class_name": "person", "bbox_xyxy": [float(value) for value in box], "confidence": float(score)} for box, score in zip(boxes, scores)], "model": detector["model"], "backend": "LinQ_TPU_service", "receipt": receipt}


@lru_cache(maxsize=4)
def cached_person_detector(model, models_root, confidence):
    return load_person_detector(model, models_root, confidence)


def tile_regions(width, height, grid=2, overlap=.2):
    if grid != 2:
        raise ValueError("Only a 2x2 tile grid is supported")
    overlap = float(overlap)
    if width < 2 or height < 2 or not np.isfinite(overlap) or not 0 <= overlap < 1:
        raise ValueError("Tile dimensions and overlap are invalid")
    tile_width = int(np.ceil(width * (1 + overlap) / 2))
    tile_height = int(np.ceil(height * (1 + overlap) / 2))
    if tile_width >= width or tile_height >= height:
        raise ValueError("Tile grid would contain duplicate regions")
    return [[left, top, left + tile_width, top + tile_height]
            for top in [0, height - tile_height] for left in [0, width - tile_width]]


def map_tile_detections(detections, region):
    left, top, right, bottom = region
    mapped = []
    for item in detections:
        x1, y1, x2, y2 = item["xyxy"]
        if not np.isfinite([x1, y1, x2, y2, item["score"]]).all() or not 0 <= x1 < x2 <= right - left or not 0 <= y1 < y2 <= bottom - top:
            raise ValueError("A tile detection is outside its source crop")
        mapped.append({**item, "xyxy": [x1 + left, y1 + top, x2 + left, y2 + top]})
    return mapped


def global_class_nms(detections, iou_threshold=.5):
    if not np.isfinite(iou_threshold) or not 0 <= iou_threshold <= 1:
        raise ValueError("Invalid NMS IoU threshold")
    remaining = sorted(range(len(detections)), key=lambda index: (-detections[index]["score"], index))
    kept = []
    while remaining:
        selected = remaining.pop(0)
        item = detections[selected]
        kept.append(item)
        x1, y1, x2, y2 = item["xyxy"]
        area = max(0., x2 - x1) * max(0., y2 - y1)
        candidates = []
        for index in remaining:
            other = detections[index]
            if other["label"] != item["label"]:
                candidates.append(index)
                continue
            a1, b1, a2, b2 = other["xyxy"]
            intersection = max(0., min(x2, a2) - max(x1, a1)) * max(0., min(y2, b2) - max(y1, b1))
            union = area + max(0., a2 - a1) * max(0., b2 - b1) - intersection
            if not union > 0 or intersection / union <= iou_threshold:
                candidates.append(index)
        remaining = candidates
    return kept


def box_iou(first, second):
    x1, y1, x2, y2 = first
    a1, b1, a2, b2 = second
    intersection = max(0., min(x2, a2) - max(x1, a1)) * max(0., min(y2, b2) - max(y1, b1))
    union = max(0., x2 - x1) * max(0., y2 - y1) + max(0., a2 - a1) * max(0., b2 - b1) - intersection
    return intersection / union if union > 0 else 0.


def merge_tiled_detections(full_detections, tiled_receipts, min_tile_support=1):
    if type(min_tile_support) is not int or not 1 <= min_tile_support <= 4:
        raise ValueError("min_tile_support must be an integer from 1 to 4")
    candidates = [(tile_index, item) for tile_index, tile in enumerate(tiled_receipts) for item in tile["detections_source_xyxy"]]
    candidates.sort(key=lambda pair: -pair[1]["score"])
    merged = list(full_detections)
    clusters = []
    while candidates:
        tile_index, selected = candidates.pop(0)
        supporting_tiles = {tile_index}
        remaining = []
        for other_tile, other in candidates:
            if other["label"] == selected["label"] and box_iou(selected["xyxy"], other["xyxy"]) >= .4:
                supporting_tiles.add(other_tile)
            else:
                remaining.append((other_tile, other))
        candidates = remaining
        supported = len(supporting_tiles) >= min_tile_support
        already_full = any(other["label"] == selected["label"] and box_iou(selected["xyxy"], other["xyxy"]) > .5 for other in full_detections)
        accepted = supported and not already_full
        if accepted:
            merged.append(selected)
        clusters.append({"detection": selected, "supporting_tile_indices": sorted(supporting_tiles), "tile_support": len(supporting_tiles), "accepted": accepted, "reason": "insufficient_distinct_tiles" if not supported else "already_present_full_frame" if already_full else "supported_extra"})
    return merged, clusters


def detect(image, context, options):
    started = time.perf_counter()
    detector = cached_person_detector(options.get("model", "rfdetr"), options.get("models_root"), float(options.get("confidence", .25)))
    rgb = np.asarray(image.convert("RGB") if hasattr(image, "convert") else image)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("A uint8 RGB frame is required")
    tiling = options.get("tiling")
    regions = []
    min_tile_support = 1
    if tiling:
        if not isinstance(tiling, dict):
            raise ValueError("Tiling must be an options object")
        regions = tile_regions(rgb.shape[1], rgb.shape[0], tiling.get("grid", 2), tiling.get("overlap", .2))
        min_tile_support = tiling.get("min_tile_support", 1)
        merge_tiled_detections([], [], min_tile_support)
    result = detect_person(detector, rgb, float(options.get("timeout_seconds", 60)))
    detections = [{"label": "person", "xyxy": item["bbox_xyxy"], "score": item["confidence"]} for item in result["detections"]]
    provenance = {"adapter": "LinQ_AMQP_person_detector", "model": result["model"], "device_actual": "LinQ_remote_TPU_service", "TPU_execution_verified": result["receipt"]["status"] == "received_and_validated", "local_TPU_device_used": False, "program_sha256": detector["sha256"], "input_shape": detector["shape"], "decoder_sha256": detector["decoder_sha256"], "compile_report_sha256": detector["compile_report_sha256"], "receipt": result["receipt"], "semantics": "Person bounding boxes only; incident interpretation requires temporal rules or VLM"}
    if regions:
        full_count = len(detections)
        provenance["full_frame_detections"] = list(detections)
        tiled_receipts = []
        for region in regions:
            left, top, right, bottom = region
            tile_result = detect_person(detector, np.ascontiguousarray(rgb[top:bottom, left:right]), float(options.get("timeout_seconds", 60)))
            tile_detections = [{"label": "person", "xyxy": item["bbox_xyxy"], "score": item["confidence"]} for item in tile_result["detections"]]
            mapped = map_tile_detections(tile_detections, region)
            tiled_receipts.append({"source_xyxy": region, "source_size": [right - left, bottom - top], "detections_source_xyxy": mapped, "receipt": tile_result["receipt"]})
        candidate_count = full_count + sum(len(tile["detections_source_xyxy"]) for tile in tiled_receipts)
        detections, clusters = merge_tiled_detections(provenance["full_frame_detections"], tiled_receipts, min_tile_support)
        provenance.update(tiled_receipts=tiled_receipts, tile_support_clusters=clusters, tiling={"grid": 2, "overlap": float(tiling.get("overlap", .2)), "overlap_semantics": "Fraction of the source axis shared by the adjacent two tiles", "global_nms_iou": .5, "tile_support_match_iou": .4, "min_tile_support": min_tile_support, "full_frame_detections_preserved": True, "merge_policy": "Preserve every full-frame box; cluster tile extras at IoU0.4; discard extras matching a full-frame box at IoU>0.5", "source_size": [rgb.shape[1], rgb.shape[0]], "full_frame_candidate_count": full_count, "candidate_count_before_nms": candidate_count, "candidate_count_after_nms": len(detections)})
        provenance["TPU_execution_verified"] = provenance["TPU_execution_verified"] and all(item["receipt"]["status"] == "received_and_validated" for item in tiled_receipts)
    provenance["elapsed_ms"] = (time.perf_counter() - started) * 1000
    return {"actual_execution": True, "mode": "fresh_inference", "detections": detections, "signals": {"person_present": max((item["score"] for item in detections), default=0.)}, "provenance": provenance}
