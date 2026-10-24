import hashlib
import json
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def load_bundle(directory):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("status") != "onnx_cpu_parity_passed":
        raise ValueError("A completed ONNX parity bundle is required")
    for entry in manifest["files"]:
        path = (directory / entry["path"]).resolve()
        if not path.is_relative_to(directory) or digest(path) != entry["sha256"]:
            raise ValueError("Bundle artifact identity mismatch: " + entry["path"])
    return directory, manifest


def head_probabilities(features, decision):
    import numpy as np
    vector = np.asarray(features, dtype=np.float32).reshape(1, -1).copy()
    expected = 768 * decision["feature_frames"]
    if vector.shape != (1, expected) or not np.isfinite(vector).all():
        raise ValueError("Head requires finite ordered frame features with shape " + str((1, expected)))
    vector -= np.asarray(decision["mean"], dtype=np.float64)
    vector /= np.asarray(decision["scale"], dtype=np.float64)
    logits = vector @ np.asarray(decision["coef"], dtype=np.float64).T + np.asarray(decision["intercept"], dtype=np.float64)
    if logits.shape == (1, 1):
        value = float(logits[0, 0])
        positive = 1 / (1 + np.exp(-value)) if value >= 0 else np.exp(value) / (1 + np.exp(value))
        result = np.asarray([1 - positive, positive], dtype=np.float64)
    else:
        values = np.exp(logits[0] - logits[0].max())
        result = values / values.sum()
    if result.shape != (len(decision["classes"]),) or not np.isfinite(result).all():
        raise ValueError("Invalid classification probabilities")
    return result


def preprocess(paths, settings):
    import numpy as np
    from PIL import Image, ImageOps
    from torchvision.models import ConvNeXt_Tiny_Weights
    transform = ConvNeXt_Tiny_Weights.IMAGENET1K_V1.transforms()
    actual = {name: getattr(transform, name) for name in ("crop_size", "resize_size", "mean", "std", "antialias")}
    actual["interpolation"] = str(transform.interpolation)
    if actual != settings:
        raise ValueError("Installed torchvision preprocessing differs from exported bundle")
    result = []
    for path in paths:
        with Image.open(path) as image:
            pixels = transform(ImageOps.exif_transpose(image).convert("RGB")).numpy()
        result.append(pixels)
    values = np.stack(result).astype(np.float32)
    if values.shape[1:] != (3, 224, 224) or not np.isfinite(values).all():
        raise ValueError("Invalid preprocessing output")
    return values
