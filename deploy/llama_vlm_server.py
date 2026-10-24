import argparse
import base64
import io
import hashlib
import subprocess
import json
import time
import urllib.error
import urllib.parse
import urllib.request

from deploy.vlm_server import Settings, create_app

DEFAULT_MODEL = 'Qwen3.5-0.8B-Q4_K_M'
RESPONSE_FORMAT = {
    'type': 'json_schema',
    'json_schema': {
        'name': 'visible_scene_assessment',
        'strict': True,
        'schema': {
            'type': 'object',
            'properties': {
                'unsafe': {'type': ['boolean', 'null']},
                'reason': {'type': 'string', 'maxLength': 160},
                'evidence': {'type': 'array', 'items': {'type': 'string'}, 'maxItems': 2},
                'scope': {'type': 'string'},
            },
            'required': ['unsafe', 'reason', 'evidence', 'scope'],
            'additionalProperties': False,
        },
    },
}


def loopback_endpoint(url):
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != 'http' or parsed.hostname not in ['127.0.0.1', 'localhost', '::1'] or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError('llama backend must be an unauthenticated loopback HTTP endpoint')
    if parsed.path != '/v1/chat/completions':
        raise ValueError('Expected llama /v1/chat/completions endpoint')
    return url


def encode_content(images, prompt):
    from PIL import Image
    if not 1 <= len(images) <= 3:
        raise ValueError('One to three ordered camera frames required')
    content, sizes = [], []
    for index, image in enumerate(images):
        value = image.convert('RGB').copy()
        value.thumbnail((512, 512), resample=Image.Resampling.BICUBIC)
        buffer = io.BytesIO()
        value.save(buffer, format='JPEG', quality=95)
        content.extend([{'type': 'text', 'text': f'Ordered frame {index+1}:'}, {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,'+base64.b64encode(buffer.getvalue()).decode('ascii')}}])
        sizes.append(list(value.size))
    content.append({'type': 'text', 'text': prompt})
    return content, sizes


def native_video_content(image_content, sizes, prompt):
    from PIL import Image
    urls = [item['image_url']['url'] for item in image_content if item['type'] == 'image_url']
    if not 2 <= len(urls) <= 3 or len(sizes) != len(urls):
        raise ValueError('Native video requires two or three ordered frames')
    if len({tuple(size) for size in sizes}) != 1:
        raise ValueError('Native video requires identical resized frame dimensions; no implicit crop or fallback')
    width, height = sizes[0]
    pixels, hashes = [], []
    for url in urls:
        jpeg = base64.b64decode(url.split(',', 1)[1])
        with Image.open(io.BytesIO(jpeg)) as image:
            rgb = image.convert('RGB').tobytes()
        if len(rgb) != width * height * 3:
            raise ValueError('Native video RGB frame dimensions differ')
        pixels.append(rgb)
        hashes.append(hashlib.sha256(rgb).hexdigest())
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-f', 'rawvideo', '-pixel_format', 'rgb24', '-video_size', f'{width}x{height}', '-framerate', '1', '-i', 'pipe:0', '-frames:v', str(len(urls)), '-an', '-c:v', 'ffv1', '-level', '3', '-pix_fmt', 'bgr0', '-threads', '1', '-fflags', '+bitexact', '-flags:v', '+bitexact', '-map_metadata', '-1', '-f', 'matroska', 'pipe:1']
    started = time.perf_counter()
    try:
        encoded = subprocess.run(command, input=b''.join(pixels), capture_output=True, check=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError('Native video encoding failed: '+type(error).__name__+'; no fallback') from None
    encoding_ms = (time.perf_counter()-started)*1000
    if not encoded or len(encoded) > 8*1024*1024:
        raise ValueError('Native video is empty or exceeds8MiB')
    content = [{'type': 'text', 'text': 'Ordered accepted camera snapshots in the supplied video. Container timestamps are ordinal positions; do not infer source-camera speed or elapsed time.'}, {'type': 'input_video', 'input_video': {'data': base64.b64encode(encoded).decode('ascii')}}, {'type': 'text', 'text': prompt}]
    return content, {'input_representation': 'native_video_ffv1_mkv_post_jpeg_rgb', 'image_transport': 'lossless FFV1 Matroska base64 video', 'post_jpeg_rgb_sha256': hashes, 'native_video_sha256': hashlib.sha256(encoded).hexdigest(), 'native_video_bytes': len(encoded), 'native_video_fps': 1, 'native_video_frame_timestamps_s': list(range(len(urls))), 'native_video_timestamp_basis': 'ordinal container timeline, not original camera timestamps', 'native_video_codec': 'ffv1', 'native_video_pixel_format': 'bgr0', 'video_encoding_ms': encoding_ms}


class LlamaEngine:
    def __init__(self, settings, endpoint='http://127.0.0.1:18776/v1/chat/completions', timeout=180, input_mode='images', cache_prompt=True):
        if input_mode not in ['images', 'native_video']:
            raise ValueError('Unknown input_mode; choose images or native_video')
        self.input_mode = input_mode
        if type(cache_prompt) is not bool:
            raise ValueError('cache_prompt must be boolean')
        self.cache_prompt = cache_prompt
        self.settings = settings
        self.endpoint = loopback_endpoint(endpoint)
        self.timeout = timeout
        if settings.device != 'cpu' or not 1 <= settings.max_new_tokens <= 192:
            raise ValueError('Bridge requires CPU backend and max_new_tokens1..192')
        health = urllib.parse.urljoin(endpoint, '/health')
        try:
            with urllib.request.urlopen(health, timeout=5) as response:
                status = json.load(response)
        except (urllib.error.URLError, TimeoutError, ValueError) as error:
            raise RuntimeError('Local llama-server is not ready') from error
        if status.get('status') != 'ok':
            raise RuntimeError('Local llama-server health is not ok')
        self.metadata = {'model': settings.model, 'model_family': 'Qwen3.5-0.8B', 'format': 'GGUF Q4 with vision mmproj',
                         'device': 'cpu', 'backend': 'local_llama_cpp', 'transport': 'loopback_http',
                         'execution_mode': 'actual_local_model_generation', 'actual_execution': False,
                         'backend_health_checked': True, 'input_mode_requested': input_mode, 'TPU_execution_verified': False,
                         'GPU_execution_claimed': False, 'model_device_source': 'operator-configured CPU llama-server'}

    def generate(self, images, prompt):
        preparation_started = time.perf_counter()
        content, sizes = encode_content(images, prompt)
        representation = {'input_representation': 'single_jpeg_image' if len(images) == 1 else 'ordered_jpeg_images', 'image_transport': 'ordered JPEG data URLs'}
        if self.input_mode == 'native_video' and len(images) > 1:
            content, representation = native_video_content(content, sizes, prompt)
        payload = {'model': self.settings.model, 'messages': [{'role': 'user', 'content': content}],
                   'max_tokens': self.settings.max_new_tokens, 'temperature': 0,
                   'stream': False, 'cache_prompt': self.cache_prompt, 'chat_template_kwargs': {'enable_thinking': False},
                   'response_format': RESPONSE_FORMAT}
        request = urllib.request.Request(self.endpoint, data=json.dumps(payload).encode(),
                                         headers={'Content-Type': 'application/json', 'Accept': 'application/json'}, method='POST')
        preparation_ms = (time.perf_counter()-preparation_started)*1000
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read(2_000_001)
            if len(raw) > 2_000_000:
                raise ValueError('Local llama response exceeds configured size')
            result = json.loads(raw)
        except urllib.error.HTTPError as error:
            raise RuntimeError('Local llama-server returned HTTP '+str(error.code)) from None
        except (urllib.error.URLError, TimeoutError):
            raise RuntimeError('Local llama-server is unreachable or timed out; no fallback') from None
        choices = result.get('choices')
        if not isinstance(choices, list) or not choices or not isinstance(choices[0].get('message', {}).get('content'), str):
            raise ValueError('Local llama response has no actual text completion')
        text = choices[0]['message']['content']
        usage = result.get('usage', {})
        timings = result.get('timings')
        return text, {'actual_execution': True, 'backend_response_received': True, 'backend_model': result.get('model'),
                      'ordered_frame_count': len(images), 'resized_frames': sizes, **representation,
                      'input_mode_requested': self.input_mode, 'preparation_ms': preparation_ms, 'cache_prompt': self.cache_prompt,
                      'enable_thinking': False, 'response_constraints': RESPONSE_FORMAT,
                      'max_new_tokens': self.settings.max_new_tokens, 'generation_ms': (time.perf_counter()-started)*1000,
                      'usage': usage, 'generated_tokens': usage.get('completion_tokens'),
                      'llama_timings': timings if isinstance(timings, dict) else None,
                      'finish_reason': choices[0].get('finish_reason'),
                      'hit_token_limit': choices[0].get('finish_reason') == 'length',
                      'completion_id': result.get('id'), 'TPU_execution_verified': False, 'GPU_execution_claimed': False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--llama-url', default='http://127.0.0.1:18776/v1/chat/completions')
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=18775)
    parser.add_argument('--max-new-tokens', type=int, default=192)
    parser.add_argument('--timeout', type=float, default=180)
    parser.add_argument('--input-mode', choices=['images', 'native_video'], default='images')
    parser.add_argument('--no-prompt-cache', action='store_true')
    parser.add_argument('--audit-log', default='vlm_requests.jsonl')
    args = parser.parse_args()
    settings = Settings(model=args.model, device='cpu', cpu_threads=1, max_frames=3,
                        max_new_tokens=args.max_new_tokens, audit_log=args.audit_log)
    import uvicorn
    uvicorn.run(create_app(settings, engine_factory=lambda current: LlamaEngine(current, args.llama_url, args.timeout, args.input_mode, not args.no_prompt_cache)),
                host=args.host, port=args.port, workers=1)


if __name__ == '__main__':
    main()
