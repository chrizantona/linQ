import base64
import hashlib
import hmac
import io
import json
import math
import os
import re
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

PROTOCOL = 'linq_vlm_frames_v1'
CRITERIA = {
    '01': 'Running in an industrial work area. Walking alone is not running.',
    '02': 'A person dangerously close to or on a collision course with a moving forklift or industrial vehicle. Require visible proximity and motion evidence.',
    '03': 'A person actively using a handheld phone or wearing headphones during work. Object presence alone does not establish active use.',
    '04': 'Smoking or open flame outside a designated permitted area. Cigarette, smoke, and flame are distinct evidence; do not invent whether an area is permitted.',
    '05': 'Climbing over a barrier or passing beneath a conveyor or suspended load. Require visible spatial and, for crossing, temporal evidence.',
    '06': 'Working at height without a correctly secured fall protection harness. Harness presence alone does not prove anchorage, correct fastening, or height.',
    '07': 'A person falling or lying in a way indicating a possible incident. An image cannot establish loss of consciousness.',
    '08': 'Unsafe manual lifting or carrying of a load. Require visible posture/load evidence; do not infer unknown weight or safety limits.',
}


@dataclass
class Settings:
    model: str = 'Qwen3.5-0.8B-Q4_K_M'
    device: str = 'cpu'
    cpu_threads: int = 1
    max_frames: int = 16
    max_frame_bytes: int = 10_000_000
    max_total_image_bytes: int = 32_000_000
    max_body_bytes: int = 45_000_000
    max_image_pixels: int = 16_000_000
    max_visual_tokens: int = 1024
    max_input_tokens: int = 8192
    max_new_tokens: int = 160
    token_env: str = 'VLM_HTTP_TOKEN'
    audit_log: str | None = '.cache/vlm_server/requests.jsonl'


def prompt_for(scope, instruction):
    criterion = instruction or (CRITERIA[scope] if scope in CRITERIA else
        'Visible potentially unsafe person-object interactions or unexpected dangerous behavior. Identify only observable evidence; return null when risk cannot be established.')
    return (f'Inspect only the supplied image or ordered sampled video frames for scope {scope}.\n'
            f'Criterion: {criterion}\n'
            'Decide whether the complete criterion is visibly established. Return unsafe=true for established unsafe behavior, false for visibly safe behavior with sufficient evidence, or null if the required context is missing or ambiguous. Never infer unseen height, anchorage, permitted areas, load weight, movement, or intent. Sampled frames can miss an event.\n'
            'Return exactly one JSON object, without markdown or extra text, with exactly these keys: '
            f'{{"unsafe": true|false|null, "reason": "brief visible justification", "evidence": ["observable fact"], "scope": "{scope}"}}\n'
            'Use boolean or null, not strings. Evidence must describe what is visible. Do not use dataset names or ground-truth labels. '
            'Keep reason to at most 15 words and evidence to at most two facts of at most 8 words each.')


def decode_request(payload, settings):
    from PIL import Image, ImageOps, UnidentifiedImageError
    if not isinstance(payload, dict) or payload.get('protocol') != PROTOCOL:
        raise ValueError('Unsupported request protocol')
    scope = payload.get('scope')
    if not isinstance(scope, str) or not re.fullmatch(r'open|0[1-8]', scope):
        raise ValueError('Scope must be open or01..08')
    instruction = payload.get('prompt', '')
    if not isinstance(instruction, str) or len(instruction) > 4000:
        raise ValueError('Prompt must be a string of at most4000 characters')
    context = payload.get('context', {})
    if not isinstance(context, dict):
        raise ValueError('Context must be an object')
    frames = payload.get('frames')
    if not isinstance(frames, list) or not 1 <= len(frames) <= settings.max_frames:
        raise ValueError('Invalid ordered frame count')
    images, metadata, total = [], [], 0
    for frame in frames:
        if not isinstance(frame, dict) or frame.get('mime_type') not in ['image/jpeg', 'image/png']:
            raise ValueError('Each frame must be a JPEG or PNG object')
        encoded = frame.get('base64')
        if not isinstance(encoded, str) or len(encoded) > 4 * math.ceil(settings.max_frame_bytes / 3):
            raise ValueError('Encoded frame exceeds configured size')
        try:
            raw = base64.b64decode(encoded, validate=True)
        except (ValueError, base64.binascii.Error):
            raise ValueError('Invalid base64 image') from None
        total += len(raw)
        if not raw or len(raw) > settings.max_frame_bytes or total > settings.max_total_image_bytes:
            raise ValueError('Decoded frames exceed configured size')
        try:
            with Image.open(io.BytesIO(raw)) as image:
                if image.format != {'image/jpeg': 'JPEG', 'image/png': 'PNG'}[frame['mime_type']]:
                    raise ValueError('Image format differs from MIME type')
                if image.width * image.height > settings.max_image_pixels:
                    raise ValueError('Image exceeds configured pixel limit')
                image.verify()
            with Image.open(io.BytesIO(raw)) as image:
                decoded = ImageOps.exif_transpose(image).convert('RGB')
                decoded.load()
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
            raise ValueError('Invalid or oversized image') from None
        images.append(decoded)
        metadata.append(dict(sha256=hashlib.sha256(raw).hexdigest(), width=decoded.width, height=decoded.height,
                             mime_type=frame['mime_type']))
    return dict(scope=scope, prompt=prompt_for(scope, instruction), images=images, frames=metadata, context=context)


def encode_images(processor, images, prompt, settings):
    from PIL import Image
    image_processor = processor.image_processor
    patch = int(getattr(image_processor, 'patch_size', 16))
    merge = int(getattr(image_processor, 'merge_size', getattr(image_processor, 'spatial_merge_size', 2)))
    factor = patch * merge
    per_frame = settings.max_visual_tokens // len(images)
    if per_frame < 4:
        raise ValueError('Visual token budget is too small for frame count')
    max_pixels = per_frame * factor * factor
    content = []
    for index, image in enumerate(images):
        width, height = image.size
        scale = min(1., math.sqrt(max_pixels / (width * height)))
        new_width = max(factor, int(width * scale) // factor * factor)
        new_height = max(factor, int(height * scale) // factor * factor)
        if new_width * new_height > max_pixels:
            raise ValueError('Extreme aspect ratio exceeds visual token budget')
        content.append(dict(type='text', text=f'Ordered frame {index + 1}:'))
        content.append(dict(type='image', image=image.resize((new_width, new_height), resample=Image.Resampling.BICUBIC)))
    content.append(dict(type='text', text=prompt))
    template = processor.chat_template
    options = {'enable_thinking': False} if isinstance(template, str) and 'enable_thinking' in template else {}
    if settings.model == 'Qwen/Qwen3.5-4B' and not options:
        raise RuntimeError('Qwen3.5 template lacks explicit non-thinking switch')
    inputs = processor.apply_chat_template([dict(role='user', content=content)], tokenize=True,
              add_generation_prompt=True, return_dict=True, return_tensors='pt', **options,
              processor_kwargs={'images_kwargs': {'size': {'shortest_edge': factor * factor, 'longest_edge': max_pixels}}})
    visual_tokens = sum(int(grid.prod().item()) // (merge * merge) for grid in inputs.get('image_grid_thw', []))
    input_tokens = inputs['input_ids'].shape[-1]
    if visual_tokens > settings.max_visual_tokens or input_tokens > settings.max_input_tokens:
        raise ValueError('Input exceeds token budget; truncation is disabled')
    return inputs, dict(visual_tokens=visual_tokens, input_tokens=input_tokens)


def parse_answer(raw, expected_scope):
    """Validate the strict visible-evidence response contract locally."""
    try:
        value = json.loads(raw)
        keys = {'unsafe', 'reason', 'evidence', 'scope'}
        if not isinstance(value, dict) or set(value) != keys:
            raise ValueError(f'Expected exactly {sorted(keys)}')
        if value['unsafe'] is not None and type(value['unsafe']) is not bool:
            raise ValueError('unsafe must be a JSON boolean or null')
        if not isinstance(value['reason'], str):
            raise ValueError('reason must be a string')
        if len(value['reason'].split()) > 15:
            raise ValueError('reason must contain at most 15 words')
        evidence = value['evidence']
        if not isinstance(evidence, list) or len(evidence) > 2 or any(not isinstance(item, str) for item in evidence):
            raise ValueError('evidence must be an array of at most two strings')
        if any(len(item.split()) > 8 for item in evidence):
            raise ValueError('each evidence fact must contain at most 8 words')
        if value['scope'] != expected_scope:
            raise ValueError('scope does not match the request')
        return value, None
    except (ValueError, TypeError) as error:
        return None, str(error)


def create_app(settings=None, engine_factory=None):
    from fastapi import FastAPI, HTTPException, Request
    settings = settings or Settings()
    if engine_factory is None:
        raise ValueError('A local model engine factory is required')
    factory = engine_factory
    token = os.environ.get(settings.token_env)
    if not token:
        raise RuntimeError('Set ' + settings.token_env + ' before serving')
    lock, log_lock = threading.Lock(), threading.Lock()
    state = dict(engine=None, ready=False, requests=0, successful_generations=0, actual_completed_generations=0)

    def authorize(request):
        supplied = request.headers.get('authorization', '')
        if not hmac.compare_digest(supplied, 'Bearer ' + token):
            raise HTTPException(401, 'Invalid bearer token')

    def record(value):
        if settings.audit_log:
            path = Path(settings.audit_log)
            path.parent.mkdir(parents=True, exist_ok=True)
            with log_lock, path.open('a') as output:
                output.write(json.dumps(dict(recorded_at=time.time(), **value), ensure_ascii=False) + '\n')

    @asynccontextmanager
    async def lifespan(app):
        state['engine'] = factory(settings)
        state['ready'] = True
        try:
            yield
        finally:
            state['ready'] = False
            state['engine'] = None

    app = FastAPI(lifespan=lifespan)

    @app.get('/health')
    def health(request: Request):
        authorize(request)
        engine = state['engine']
        return dict(ready=state['ready'], requests=state['requests'], successful_generations=state['successful_generations'], actual_completed_generations=state['actual_completed_generations'],
                    execution=dict(engine.metadata, actual_execution=state['actual_completed_generations'] > 0) if engine else None)

    @app.post('/')
    @app.post('/infer')
    async def infer(request: Request):
        authorize(request)
        if not state['ready']:
            raise HTTPException(503, 'Model is not ready')
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > settings.max_body_bytes:
                raise HTTPException(413, 'Request exceeds configured size')
        try:
            payload = json.loads(body)
            decoded = decode_request(payload, settings)
        except (ValueError, TypeError) as error:
            raise HTTPException(422, str(error)) from None
        if not lock.acquire(blocking=False):
            raise HTTPException(429, 'Model is busy; retry after current request')
        started = time.perf_counter()
        state['requests'] += 1
        try:
            import asyncio
            raw, execution = await asyncio.to_thread(state['engine'].generate, decoded['images'], decoded['prompt'])
            state['actual_completed_generations'] += int(execution.get('actual_execution') is True)
            parsed, error = parse_answer(raw, decoded['scope'])
            metadata = dict(state['engine'].metadata, **execution, ordered_frames=decoded['frames'],
                            server_pipeline_ms=(time.perf_counter() - started) * 1000)
            if error:
                record(dict(status='invalid_model_output', scope=decoded['scope'], raw_answer=raw, error=error, execution=metadata))
                raise HTTPException(422, dict(error=error, raw_answer=raw, execution=metadata))
            state['successful_generations'] += 1
            result = dict(parsed, execution=metadata)
            record(dict(status='model_response', scope=decoded['scope'], raw_answer=raw, response=result))
            return result
        except HTTPException:
            raise
        except (ValueError, RuntimeError) as error:
            record(dict(status='generation_failed', scope=decoded['scope'], error_type=type(error).__name__, error=str(error)))
            raise HTTPException(503, dict(error_type=type(error).__name__, error=str(error))) from None
        finally:
            lock.release()
    return app
