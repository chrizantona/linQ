"""Validate and encode an ordered set of still images as a synthetic stop-frame video."""

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import warnings

from PIL import Image


MIN_IMAGES = 2
MAX_IMAGES = 24
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_TOTAL_BYTES = 100 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
MAX_DIMENSION = 8192
CAMERA_ID_PATTERN = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z')
ALLOWED_FORMATS = {'JPEG', 'PNG'}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_image(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('Each image must be an ordinary file, not a symlink')
    size = path.stat().st_size
    if size <= 0 or size > MAX_IMAGE_BYTES:
        raise ValueError('Each image must be nonempty and at most 20 MiB')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(path) as image:
                image_format = image.format
                width, height = image.size
                if image_format not in ALLOWED_FORMATS:
                    raise ValueError('Only JPEG and PNG image data is accepted')
                if getattr(image, 'n_frames', 1) != 1:
                    raise ValueError('Animated image files are not accepted as still frames')
                if (width <= 0 or height <= 0 or width > MAX_DIMENSION or height > MAX_DIMENSION
                        or width * height > MAX_IMAGE_PIXELS):
                    raise ValueError('Image dimensions exceed the safe image-set limit')
                image.verify()
            with Image.open(path) as image:
                image.load()
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        raise ValueError('Image exceeds the safe decompression limit') from error
    except (OSError, SyntaxError) as error:
        raise ValueError('Image file is corrupt or has an unsupported format') from error
    return {'path': path.resolve(), 'name': path.name, 'bytes': size, 'format': image_format,
            'width': width, 'height': height, 'sha256': _sha256(path)}


def _run_ffmpeg(command):
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180)
    except FileNotFoundError as error:
        raise RuntimeError('ffmpeg is required to prepare an image-set video') from error
    except subprocess.TimeoutExpired as error:
        raise RuntimeError('ffmpeg timed out while preparing the image-set video') from error
    except subprocess.CalledProcessError as error:
        detail = error.stderr.decode('utf-8', errors='replace')[-1200:]
        raise RuntimeError('ffmpeg could not prepare the image-set video: ' + detail) from error


def prepare_image_set(images, output_dir, *, camera_id, ffmpeg='ffmpeg'):
    """Encode 2–24 ordered JPEG/PNG inputs at one still frame per second.

    `output_dir` must not already exist. The installed directory contains video.mp4 and
    receipt.json; the receipt describes synthetic time and hashes of both the source stills
    and encoded video. No model predictions are generated here.
    """
    if not isinstance(camera_id, str) or not CAMERA_ID_PATTERN.fullmatch(camera_id):
        raise ValueError('A safe physical camera id is required')
    if not isinstance(images, (list, tuple)) or not MIN_IMAGES <= len(images) <= MAX_IMAGES:
        raise ValueError('An image set must contain between 2 and 24 ordered images')
    checked = [_validate_image(path) for path in images]
    total_bytes = sum(item['bytes'] for item in checked)
    if total_bytes > MAX_TOTAL_BYTES:
        raise ValueError('Image set exceeds the 100 MiB total size limit')
    dimensions = {(item['width'], item['height']) for item in checked}
    if len(dimensions) != 1:
        raise ValueError('Every image in an image set must have the same dimensions')
    width, height = next(iter(dimensions))
    if width % 2 or height % 2:
        raise ValueError('Image-set width and height must be even for yuv420p video encoding')

    destination = Path(output_dir).absolute()
    parent = destination.parent.resolve()
    if destination.name in {'', '.', '..'} or not parent.is_dir():
        raise ValueError('Image-set destination parent must already exist')
    if destination.is_symlink() or destination.exists():
        raise FileExistsError('Image-set destination already exists or is a symlink')

    with tempfile.TemporaryDirectory(prefix='.image-set-stage-', dir=parent) as temp_name:
        stage = Path(temp_name)
        inputs = stage / 'inputs'
        inputs.mkdir()
        concat_lines = ['ffconcat version 1.0']
        for index, item in enumerate(checked, start=1):
            safe_name = f'frame-{index:02d}{".jpg" if item["format"] == "JPEG" else ".png"}'
            staged_input = inputs / safe_name
            shutil.copyfile(item['path'], staged_input)
            if _sha256(staged_input) != item['sha256']:
                raise ValueError('Source image changed while the image set was being staged')
            # Only generated safe basenames go into ffconcat, preventing directive injection.
            concat_lines.extend((f"file 'inputs/{safe_name}'", 'duration 1'))
        concat_lines.append(f"file 'inputs/frame-{len(checked):02d}{'.jpg' if checked[-1]['format'] == 'JPEG' else '.png'}'")
        concat_path = stage / 'frames.ffconcat'
        concat_path.write_text('\n'.join(concat_lines) + '\n', encoding='ascii')
        staged_video = stage / 'video.mp4'
        _run_ffmpeg([ffmpeg, '-hide_banner', '-loglevel', 'error', '-y', '-f', 'concat', '-safe', '0',
                     '-i', str(concat_path), '-fps_mode', 'cfr', '-r', '1', '-frames:v', str(len(checked)),
                     '-an', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '24', '-pix_fmt', 'yuv420p',
                     '-movflags', '+faststart', str(staged_video)])
        if not staged_video.is_file() or staged_video.stat().st_size == 0:
            raise ValueError('ffmpeg did not create a nonempty image-set video')
        sources = [{'original_path': str(item['path']), 'original_filename': item['name'],
                    'format': item['format'], 'bytes': item['bytes'], 'width': item['width'],
                    'height': item['height'], 'sha256': item['sha256']}
                   for item in checked]
        receipt = {'input_kind': 'image_set', 'synthetic_time': True, 'camera_id': camera_id,
                   'frame_rate_fps': 1, 'frame_count': len(checked), 'duration_seconds': len(checked),
                   'width': width, 'height': height, 'original_source_hashes': [item['sha256'] for item in checked],
                   'source_images': sources, 'video_filename': 'video.mp4',
                   'video_sha256': _sha256(staged_video)}
        (stage / 'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        # The final name must be absent; same-filesystem directory rename installs atomically.
        if destination.is_symlink() or destination.exists():
            raise FileExistsError('Image-set destination already exists or is a symlink')
        os.rename(stage, destination)
    return {'video_path': str(destination / 'video.mp4'), 'receipt_path': str(destination / 'receipt.json'), **receipt}
