import argparse
import base64
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.parse
import urllib.request

if __package__:
    from .common import write_json
else:
    from common import write_json



def local_endpoint(value):
    parsed = urllib.parse.urlsplit(value)
    if (parsed.scheme != 'http' or parsed.hostname not in {'127.0.0.1', 'localhost', '::1'}
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.path != '/infer'):
        raise ValueError('VLM endpoint must be the local /infer service')
    return value


def call_vlm(config_path, frame_paths, scope, prompt, context=None):
    config = json.loads(Path(config_path).read_text())
    if config["protocol"] != "linq_vlm_frames_v1":
        raise ValueError("Unsupported VLM host protocol")
    endpoint = os.environ.get(config["url_env"])
    if not endpoint or not endpoint.startswith(("https://", "http://")):
        raise RuntimeError("Configure the local VLM endpoint in " + config["url_env"])
    endpoint = local_endpoint(endpoint)
    if not 1 <= len(frame_paths) <= config["max_frames"]:
        raise ValueError("Invalid ordered frame count")
    frames = []
    for path in frame_paths:
        path = Path(path)
        suffix = path.suffix.lower()
        content_type = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}.get(suffix)
        if content_type is None:
            raise ValueError("VLM host adapter accepts JPEG or PNG frame files")
        data = path.read_bytes()
        if len(data) > config["max_frame_bytes"]:
            raise ValueError("Frame exceeds configured request size")
        frames.append({"mime_type": content_type, "base64": base64.b64encode(data).decode("ascii")})
    payload = {"protocol": "linq_vlm_frames_v1", "scope": scope, "prompt": prompt, "frames": frames, "context": context or {}}
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    token = os.environ.get(config.get("token_env", ""))
    if config.get("auth_required", False) and not token:
        raise RuntimeError("Configure host authorization in " + config["token_env"])
    if token:
        headers["Authorization"] = "Bearer " + token
    request = urllib.request.Request(endpoint, data=json.dumps(payload).encode(), headers=headers, method="POST")
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=config["timeout_seconds"]) as response:
            raw = response.read(config["max_response_bytes"] + 1)
            if len(raw) > config["max_response_bytes"]:
                raise ValueError("VLM response exceeds configured size")
            result = json.loads(raw)
    except urllib.error.HTTPError as error:
        raise RuntimeError("VLM host returned HTTP " + str(error.code)) from None
    except urllib.error.URLError:
        raise RuntimeError("VLM host is unreachable; check endpoint, network and host process") from None
    if not isinstance(result, dict) or result.get("scope") != scope:
        raise ValueError("VLM response scope mismatch")
    if "unsafe" not in result or not (result["unsafe"] is None or type(result["unsafe"]) is bool):
        raise ValueError("VLM unsafe must be boolean or null")
    if not isinstance(result.get("reason"), str) or not isinstance(result.get("evidence"), list) or not all(isinstance(item, str) for item in result["evidence"]):
        raise ValueError("VLM reason/evidence contract mismatch")
    return {"response": result, "transport": "http_loopback", "execution_device": "local_model_server", "TPU_execution_verified": False, "request_ms": (time.perf_counter() - started) * 1000, "ordered_frame_count": len(frames)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="deploy/vlm_host.example.json")
    parser.add_argument("--frames", nargs="+", required=True)
    parser.add_argument("--scope", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Refusing to overwrite a VLM receipt")
    try:
        result = call_vlm(args.config, args.frames, args.scope, args.prompt)
        write_json(output, {"status": "host_http_response_received", **result})
    except Exception as error:
        write_json(output, {"status": "blocked", "error_type": type(error).__name__, "error": str(error), "TPU_execution_verified": False})
        raise SystemExit(1)


if __name__ == "__main__":
    main()
