from contextlib import contextmanager

if __package__:
    from .common import head_probabilities, load_bundle, preprocess
else:
    from common import head_probabilities, load_bundle, preprocess


@contextmanager
def open_encoder(bundle_path, backend, threads=4):
    if backend != "cpu":
        raise ValueError("Only the CPU encoder backend is supported")
    import onnxruntime as ort

    bundle, manifest = load_bundle(bundle_path)
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(bundle / "encoder.onnx"), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise RuntimeError("Explicit CPU provider is required")

    def run(pixels):
        return session.run([manifest["output"]["name"]],
                           {manifest["input"]["name"]: pixels})[0]

    yield {"run": run, "bundle": bundle, "manifest": manifest,
           "backend": "onnxruntime_cpu", "device": "cpu"}


def encode_pixels(runtime, pixels):
    import numpy as np
    values = np.asarray(pixels, dtype=np.float32)
    if values.ndim != 4 or values.shape[1:] != (3, 224, 224) or not len(values) or not np.isfinite(values).all():
        raise ValueError("Finite [N,3,224,224] pixels are required")
    features = [np.asarray(runtime["run"](frame[None]), dtype=np.float32) for frame in values]
    if any(value.shape != (1, 768) or not np.isfinite(value).all() for value in features):
        raise ValueError("Invalid encoder output")
    return np.concatenate(features)


def predict_frames(runtime, task, ordered_paths):
    manifest = runtime["manifest"]
    if task not in manifest["heads"]:
        raise ValueError("Unknown trained task: " + task)
    head = manifest["heads"][task]
    if len(ordered_paths) != head["frames"]:
        raise ValueError("Task requires exactly " + str(head["frames"]) + " ordered frames")
    pixels = preprocess(ordered_paths, manifest["preprocessing"])
    features = encode_pixels(runtime, pixels)
    import json
    decision = json.loads((runtime["bundle"] / head["path"]).read_text())
    probabilities = head_probabilities(features, decision)
    return {"task": task, "prediction": decision["classes"][int(probabilities.argmax())], "classes": decision["classes"], "probabilities": probabilities.tolist(), "backend": runtime["backend"], "device": runtime["device"], "task_spec": head["task_spec"], "bbox": None}
