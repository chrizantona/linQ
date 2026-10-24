import base64
import importlib
import io
import json
import subprocess
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image


def video_info(path):
    result = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height,avg_frame_rate,nb_frames:format=duration', '-of', 'json', str(path)], check=True, capture_output=True, text=True)
    info = json.loads(result.stdout)
    stream = info['streams'][0]
    numerator, denominator = stream['avg_frame_rate'].split('/')
    return {'width': stream['width'], 'height': stream['height'], 'fps': float(numerator)/float(denominator), 'duration_s': float(info['format']['duration'])}


def decode(path, info):
    process = subprocess.Popen(['ffmpeg', '-v', 'error', '-i', str(path), '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-threads', '1', '-'], stdout=subprocess.PIPE)
    size = info['width'] * info['height'] * 3
    try:
        index = 0
        while True:
            data = process.stdout.read(size)
            if not data:
                break
            if len(data) != size:
                raise ValueError('Incomplete decoded video frame')
            yield index, Image.frombytes('RGB', (info['width'], info['height']), data)
            index += 1
        if process.wait() != 0:
            raise RuntimeError('ffmpeg decode failed')
    finally:
        if process.poll() is None:
            process.terminate()
        process.stdout.close()
        process.wait()


def validate_detection(item, width, height):
    box = np.asarray(item['xyxy'], dtype=float)
    score = float(item['score'])
    if box.shape != (4,) or not np.isfinite(box).all() or not 0 <= score <= 1 or box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError('Invalid detector box or score')
    if box[0] < -width or box[1] < -height or box[2] > 2*width or box[3] > 2*height:
        raise ValueError('Detector coordinates do not match source pixel dimensions')
    return {**item, 'xyxy': box.tolist(), 'score': score, 'label': str(item['label'])}


def infer_records(camera, root, info):
    spec = camera['detector']
    mode = spec['mode']
    if mode == 'recorded_replay':
        path = root / spec['path']
        rows = []
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                if spec.get('sequence_id') and row.get('sequence_id') != spec['sequence_id']:
                    continue
                if row.get('width') != info['width'] or row.get('height') != info['height']:
                    raise ValueError('Recorded inference resolution differs from source video')
                rows.append(row)
    elif mode in {'python', 'http'}:
        if mode == 'python':
            module, name = spec['callable'].split(':')
            function = getattr(importlib.import_module(module), name)
        rows = []
        sample_fps = spec.get('sample_fps', 3.0)
        next_time = 0.0
        for index, image in decode(root / camera['video'], info):
            timestamp = index / info['fps']
            if timestamp + 1e-9 < next_time:
                continue
            next_time += 1 / sample_fps
            context = {'camera_id': camera['id'], 'frame_index': index, 'timestamp_s': timestamp, 'width': info['width'], 'height': info['height']}
            if mode == 'python':
                response = function(image, context, spec.get('options', {}))
            else:
                buffer = io.BytesIO()
                image.save(buffer, format='JPEG', quality=95)
                request = urllib.request.Request(spec['url'], data=json.dumps({**context, 'image_jpeg_base64': base64.b64encode(buffer.getvalue()).decode()}).encode(), headers={'Content-Type': 'application/json'})
                with urllib.request.urlopen(request, timeout=spec.get('timeout_s', 60)) as result:
                    response = json.load(result)
            rows.append({**context, **response})
    else:
        raise ValueError('Detector mode must be recorded_replay, python, or http')
    if not rows:
        raise ValueError('No detector records for camera ' + camera['id'])
    rows.sort(key=lambda row: row['frame_index'])
    previous = -1
    for row in rows:
        index = int(row['frame_index'])
        if index <= previous or index < 0 or index / info['fps'] >= info['duration_s'] + 0.1:
            raise ValueError('Duplicate, unordered, or out-of-range inference frame')
        if abs(row['timestamp_s'] - index / info['fps']) > 1/info['fps'] + 1e-3:
            raise ValueError('Recorded inference timestamp does not match source frame')
        if spec.get('source_sha256') and row.get('media_sha256') != spec['source_sha256']:
            raise ValueError('Recorded inference source hash mismatch')
        row['detections'] = [validate_detection(item, info['width'], info['height']) for item in row['detections']]
        previous = index
    return rows
