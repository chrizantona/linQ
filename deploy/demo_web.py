import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import csv
import datetime
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import uuid

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, StrictBool

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deploy import linq_client
from deploy.common import digest
from deploy.image_sources import MAX_IMAGE_BYTES, MAX_TOTAL_BYTES, prepare_image_set
from scripts.run_user_video import TITLES, available_tasks

MAX_FILE = 200 * 1024 * 1024
CHUNK_SIZE = 8 * 1024 * 1024
REMOTE = 'mosprom_hackathon_linq'
EXTENSIONS = {'.mp4', '.mov', '.mkv', '.avi', '.webm'}
IDENTITY = re.compile(r'[a-f0-9]{32}')
ACTION_REVIEW_TASKS = {'phone_use', 'smoking'}
ACTION_TARGETS = {'phone_use': 'phone', 'smoking': 'cigarette'}
TASK_CASES = {
    'omnifall_v2': 'fall', 'physicalai_v2': 'proximity', 'hmdb51_v2': 'activity',
    'phone_objects': 'phone', 'phone_use': 'phone',
    'cigarette_objects': 'smoking', 'smoking': 'smoking',
}
TASK_TITLES = {
    'omnifall_v2': 'Падения', 'physicalai_v2': 'Опасное сближение', 'hmdb51_v2': 'Действия',
    'phone_objects': 'Телефоны', 'phone_use': 'Использование телефона',
    'cigarette_objects': 'Сигареты', 'smoking': 'Курение',
}


def task_case(task):
    return TASK_CASES.get(task, 'observation')


def task_title(task):
    return TASK_TITLES.get(task, TITLES.get(task, 'Наблюдение'))


def candidate_event_public_view(event):
    if event.get('event_trigger') != 'active_event' or not event.get('event_id'):
        return None
    vlm = event.get('vlm') if isinstance(event.get('vlm'), dict) else {}
    value = {
        'event_id': event.get('event_id'), 'camera_id': event.get('camera_id'),
        'rule_id': event.get('rule_id'), 'type': event_type(event),
        'frame_index': event.get('frame_index'), 'trigger_timestamp_s': event.get('trigger_timestamp_s'),
        'source_sha256': event.get('source_sha256'), 'state': event.get('state'),
        'score': event.get('score'), 'track_ids': event.get('track_ids', []),
        'vlm_status': vlm.get('status', 'unknown'), 'provisional': event.get('provisional') is True,
    }
    if isinstance(event.get('image_index'), int):
        value['image_index'] = event['image_index']
    if event.get('actionCandidate') is not None:
        value['actionCandidate'] = event.get('actionCandidate')
        value['actionCandidateStatus'] = event.get('actionCandidateStatus')
    return value


def event_type(event):
    rule_id = event.get('rule_id')
    if rule_id and rule_id != 'inspection':
        return rule_id
    return event.get('kind') or rule_id or 'unknown'


def event_image_index(event, source_count):
    if not isinstance(source_count, int) or source_count <= 0:
        return None
    trigger_time = event.get('trigger_timestamp_s')
    frames = [frame for frame in event.get('raw_frames', [])
              if isinstance(frame.get('frame_index'), int) and not isinstance(frame.get('frame_index'), bool)
              and 0 <= frame['frame_index'] < source_count
              and (not isinstance(trigger_time, (int, float)) or frame.get('timestamp_s', trigger_time) <= trigger_time)]
    if not frames:
        return None
    frame = max(frames, key=lambda item: (item.get('timestamp_s', 0), item['frame_index']))
    return frame['frame_index']


def action_candidate(unsafe, target_detected=None):
    if target_detected is not True:
        return 'unknown'
    if unsafe is True:
        return 'visible_action_candidate'
    if unsafe is False:
        return 'no_visible_action_candidate'
    return 'unknown'


def analysis_from_event(selected, selection_policy, task, target_detected=None):
    response = model_event_response(selected)
    if response is None:
        return None, selected.get('vlm', {}).get('error', 'Ответ VLM отсутствует; нужен просмотр оператора.')
    unsafe = response.get('unsafe')
    common = {'action': 'Оценка видимого эпизода', 'summary': response.get('reason'), 'facts': response.get('evidence', []),
              'latencyMs': selected['vlm'].get('request_ms'), 'provider': 'LinQ CPU', 'model': 'Qwen3.5-0.8B',
              'modelprovider': 'LinQ CPU Qwen3.5-0.8B', 'operatorConfirmed': None,
              'eventId': selected['event_id'], 'triggerTime': selected['trigger_timestamp_s'], 'camera': selected['camera_id'],
              'summarySelection': selection_policy}
    if task in ACTION_REVIEW_TASKS:
        candidate = action_candidate(unsafe, target_detected)
        if target_detected is not True:
            common['summary'] = 'Недостаточно детекций целевого предмета в кадрах эпизода, чтобы оценить действие.'
            common['facts'] = ['Свежий детектор не подтвердил целевой предмет в кадрах этого эпизода.']
        return {**common, 'assessment': 'unknown', 'actionCandidate': candidate,
                'actionCandidateStatus': 'insufficient_object_evidence' if target_detected is not True else 'review_candidate',
                'modelObservation': {'reason': response.get('reason'), 'evidence': response.get('evidence', [])},
                'semantics': 'Visible action candidate for operator review; not an automatic safety violation verdict.'}, None
    return {**common, 'assessment': 'unknown' if unsafe is None else 'confirmed' if unsafe is True else 'not_confirmed',
            'rawResponse': response, 'semantics': 'Selected model candidate; not an operator verdict for the whole video'}, None


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        with temporary.open('rb') as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def check_identity(value):
    if not IDENTITY.fullmatch(value):
        raise HTTPException(404, 'Unknown job')
    return value


def safe_filename(name):
    if not isinstance(name, str) or Path(name).name != name or '\\' in name or not re.fullmatch(r'[A-Za-z0-9_.-]+', name) or name in {'.', '..'}:
        raise ValueError('Unsafe artifact filename')
    return name


def remote_json(client, path):
    try:
        value = linq_client.contents(client, path)
    except RuntimeError as error:
        if 'HTTP 404' in str(error):
            return None
        raise
    content = value['content']
    if value.get('format') == 'base64':
        content = base64.b64decode(content).decode('utf-8')
    return json.loads(content) if isinstance(content, str) else content


def remote_frame_rows(client, path):
    try:
        value = linq_client.contents(client, path)
    except RuntimeError as error:
        if 'HTTP 404' in str(error):
            return []
        raise
    text = value['content']
    if value.get('format') == 'base64':
        text = base64.b64decode(text).decode('utf-8')
    if not isinstance(text, str):
        raise ValueError('Expected actual JSONL frame records')
    rows = []
    lines = text.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1 and not line.endswith('\n'):
                break
            raise
        if not isinstance(row, dict):
            raise ValueError('Invalid actual frame record')
        rows.append(row)
    return rows


def execute_checked(client, code, timeout=120):
    result = linq_client.execute(client, code, timeout=timeout)
    if result.get('status') != 'completed' or result.get('errors'):
        names = [item.get('ename', 'RemoteError') for item in result.get('errors', [])]
        raise RuntimeError('LinQ command failed: ' + ', '.join(names or [result.get('status', 'unknown')]))
    return json.loads(result['stdout'])


def assembly_code(remote_directory, filename, parts, expected_sha, expected_bytes):
    safe_filename(filename)
    if not re.fullmatch(r'videos/web_[a-f0-9]{32}', remote_directory):
        raise ValueError('Invalid owned upload directory')
    for item in parts:
        safe_filename(item['name'])
        if not item['name'].startswith(filename + '.') or not re.fullmatch(r'[a-f0-9]{64}', item['sha256']):
            raise ValueError('Invalid owned upload part')
    payload = {'directory': remote_directory, 'filename': filename, 'parts': parts, 'sha256': expected_sha, 'bytes': expected_bytes}
    return """from pathlib import Path
import hashlib,json,os
settings=json.loads(%r)
root=Path.home()/'work/mosprom_hackathon_linq'
directory=root/settings['directory']
destination=directory/settings['filename']
if destination.exists():raise FileExistsError('Uploaded video already exists')
partial=destination.with_name(destination.name+'.joining')
whole=hashlib.sha256();size=0
try:
 with partial.open('xb') as output:
  for item in settings['parts']:
   part=directory/'upload_parts'/item['name']
   value=hashlib.sha256();count=0
   with part.open('rb') as stream:
    for chunk in iter(lambda:stream.read(1024*1024),b''):
     output.write(chunk);whole.update(chunk);value.update(chunk);size+=len(chunk);count+=len(chunk)
   if value.hexdigest()!=item['sha256'] or count!=item['bytes']:raise ValueError('Upload part SHA or size mismatch')
  output.flush();os.fsync(output.fileno())
 if whole.hexdigest()!=settings['sha256'] or size!=settings['bytes']:raise ValueError('Assembled video SHA or size mismatch')
 os.replace(partial,destination)
 for item in settings['parts']:(directory/'upload_parts'/item['name']).unlink()
 try:(directory/'upload_parts').rmdir()
 except OSError:pass
 print(json.dumps({'verified':True,'sha256':whole.hexdigest(),'bytes':size,'video':str(destination.relative_to(root))}))
finally:
 partial.unlink(missing_ok=True)
""" % json.dumps(payload)


def upload_video(client, source, job_id, camera, progress):
    extension = source.suffix.lower()
    filename = safe_filename(camera + extension)
    directory = 'videos/web_' + check_identity(job_id)
    parts, offset, chunk_size = [], 0, CHUNK_SIZE
    total, expected = source.stat().st_size, digest(source)
    with source.open('rb') as stream:
        while offset < total:
            stream.seek(offset)
            chunk = stream.read(chunk_size)
            part_name = filename + '.' + str(len(parts)).zfill(6) + '.part'
            descriptor, temporary = tempfile.mkstemp(suffix='.part')
            try:
                with os.fdopen(descriptor, 'wb') as part:
                    part.write(chunk)
                try:
                    linq_client.upload(client, temporary, REMOTE + '/' + directory + '/upload_parts/' + part_name)
                except RuntimeError as error:
                    if 'HTTP 413' in str(error) and chunk_size > 256 * 1024:
                        chunk_size //= 2
                        continue
                    if 'HTTP 400' in str(error):
                        target = REMOTE + '/' + directory + '/upload_parts/' + part_name
                        hidden = any(component.startswith('.') for component in Path(target).parts)
                        raise RuntimeError('Jupyter rejected video chunk: HTTP 400; relative_path=' + target + '; raw_bytes=' + str(len(chunk)) + '; hidden_components=' + str(hidden).lower() + '. No response body or credentials included.') from None
                    raise
            finally:
                Path(temporary).unlink(missing_ok=True)
            parts.append({'name': part_name, 'bytes': len(chunk), 'sha256': hashlib.sha256(chunk).hexdigest()})
            offset += len(chunk)
            progress(offset / total)
    result = execute_checked(client, assembly_code(directory, filename, parts, expected, total), timeout=120)
    if result.get('verified') is not True or result.get('sha256') != expected or result.get('bytes') != total:
        raise ValueError('LinQ upload verification failed')
    return result


def download_file(client, remote_path, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + '.part')
    received = 0
    try:
        response = linq_client.request(client, 'GET', 'files/' + urllib.parse.quote(remote_path, safe='/'), stream=True, timeout=180)
        with response, temporary.open('wb') as stream:
            for chunk in response.iter_content(1024 * 1024):
                received += len(chunk)
                if received > 512 * 1024 * 1024:
                    raise ValueError('Rendered artifact exceeds download limit')
                stream.write(chunk)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {'bytes': received, 'sha256': digest(destination)}


class Review(BaseModel):
    confirmed: StrictBool
    note: str = ''


class PipelineRequest(BaseModel):
    sampleId: str
    task: str | None = None
    use_vlm: StrictBool = True


class JobReview(BaseModel):
    jobId: str
    eventId: str | None = None
    verdict: str
    note: str = ''


class ZoneRequest(BaseModel):
    sampleId: str
    polygon: list[list[float]]


def observed_track_matches(detections, tracks, minimum_iou=0.3):
    candidates = []
    for detection_index, detection in enumerate(detections):
        x1, y1, x2, y2 = detection['xyxy']
        area = max(0, x2 - x1) * max(0, y2 - y1)
        for track_index, track in enumerate(tracks):
            if track.get('observed') is not True or track.get('track_id') is None or track['label'] != detection['label']:
                continue
            tx1, ty1, tx2, ty2 = track['xyxy']
            overlap = max(0, min(x2, tx2) - max(x1, tx1)) * max(0, min(y2, ty2) - max(y1, ty1))
            union = area + max(0, tx2 - tx1) * max(0, ty2 - ty1) - overlap
            iou = overlap / union if union > 0 else 0
            if iou >= minimum_iou:
                candidates.append((-iou, detection_index, track_index))
    matches, used_tracks = {}, set()
    for negative_iou, detection_index, track_index in sorted(candidates):
        track_id = tracks[track_index]['track_id']
        if detection_index not in matches and track_id not in used_tracks:
            matches[detection_index] = track_id
            used_tracks.add(track_id)
    return matches


def model_event_response(event):
    vlm = event.get('vlm') or {}
    response = vlm.get('response')
    if vlm.get('status') not in {'confirmed', 'rejected', 'unknown'}:
        return None
    if isinstance(response, dict) and 'unsafe' in response and (response['unsafe'] is None or type(response['unsafe']) is bool) and isinstance(response.get('reason'), str) and isinstance(response.get('evidence'), list) and all(isinstance(item, str) for item in response['evidence']):
        return response
    return None


def select_model_event(events):
    if not events:
        return None, None
    positive, uncertain, negative = [], [], []
    for event in events:
        response = model_event_response(event)
        if response is not None and response['unsafe'] is True:
            positive.append(event)
        elif response is None or response['unsafe'] is None:
            uncertain.append(event)
        else:
            negative.append(event)
    candidates, policy = (positive, 'latest_positive_model_candidate') if positive else (uncertain, 'latest_unknown_or_error') if uncertain else (negative, 'latest_negative_all_events_negative')
    return max(candidates, key=lambda event: (event['trigger_timestamp_s'], 'source_end' in event.get('request_reasons', []), event['event_id'])), policy


def collect_event_episode(client, remote_run, results, job_id, event):
    evidence, times, hashes, image_indices, media_files = [], [], [], [], []
    for index, frame in enumerate(event.get('raw_frames', [])):
        if not event.get('camera_id') or frame.get('camera_id') != event['camera_id']:
            raise ValueError('Evidence frame camera does not match event camera')
        relative = Path(frame['relative_path'])
        timestamp = frame['timestamp_s']
        if relative.is_absolute() or '..' in relative.parts or not math.isfinite(timestamp) or timestamp > event['trigger_timestamp_s'] or times and timestamp < times[-1]:
            raise ValueError('Invalid causal evidence path or timestamp')
        name = safe_filename('evidence_' + hashlib.sha256(event['event_id'].encode()).hexdigest()[:16] + '_' + str(index) + relative.suffix.lower())
        download_file(client, remote_run + '/' + relative.as_posix(), results / name)
        if digest(results / name) != frame['sha256']:
            raise ValueError('Evidence frame SHA256 mismatch')
        media_files.append(name)
        evidence.append('/media/' + job_id + '/' + name)
        times.append(timestamp)
        hashes.append(frame['sha256'])
        image_indices.append(frame.get('frame_index'))
    response = model_event_response(event)
    trigger_time = event['trigger_timestamp_s']
    return {'start': min(times) if times else trigger_time, 'end': trigger_time, 'triggerTime': trigger_time,
            'frames': evidence, 'frameTimes': times, 'frameSHA256': hashes, 'reason': event.get('kind', 'actual_event'),
            'eventId': event['event_id'], 'trackIds': [], 'camera': event['camera_id'],
            'imageIndices': image_indices,
            'unsafe': response['unsafe'] if response is not None else None,
            'status': (event.get('vlm') or {}).get('status'), 'responseValid': response is not None,
            'operatorConfirmed': None}, media_files


def event_has_target_detection(event, rows, task):
    target = ACTION_TARGETS.get(task)
    if target is None:
        return None
    camera_id = event.get('camera_id')
    frame_indices = {frame.get('frame_index') for frame in event.get('raw_frames', [])
                     if isinstance(frame.get('frame_index'), int) and not isinstance(frame.get('frame_index'), bool)}
    trigger_time = event.get('trigger_timestamp_s')
    if not camera_id:
        return False
    return any(row.get('camera_id') == camera_id and row.get('frame_index') in frame_indices
               and row.get('accepted') is True
               and row.get('stages', {}).get('detector', {}).get('actual_execution') is True
               and row.get('stages', {}).get('detector', {}).get('mode') == 'fresh_inference'
               and any(detection.get('label') == target
                       for detection in row.get('stages', {}).get('detector', {}).get('detections', []))
               for row in rows) or any(
        not frame_indices and row.get('camera_id') == camera_id and row.get('accepted') is True
        and isinstance(trigger_time, (int, float)) and math.isclose(row.get('timestamp_s', float('nan')), trigger_time, rel_tol=0, abs_tol=1e-6)
        and row.get('stages', {}).get('detector', {}).get('actual_execution') is True
        and row.get('stages', {}).get('detector', {}).get('mode') == 'fresh_inference'
        and any(detection.get('label') == target
                for detection in row.get('stages', {}).get('detector', {}).get('detections', []))
        for row in rows)


def action_review_fields(event, rows, task):
    if str(event.get('rule_id', '')).startswith('zone_'):
        return {}
    target_detected = event_has_target_detection(event, rows, task)
    response = model_event_response(event)
    status = ('insufficient_object_evidence' if target_detected is not True else
              'model_response_unavailable' if response is None else 'review_candidate')
    return {
        'actionCandidate': action_candidate(response.get('unsafe') if response is not None else None, target_detected),
        'actionCandidateStatus': status,
        'modelObservation': {'reason': response.get('reason'), 'evidence': response.get('evidence', [])} if response is not None else None,
        'semantics': 'Visible action candidate for operator review; not an automatic safety violation verdict.',
    }


def primary_camera_for_job(job, rows=(), episode=None):
    inputs = job.get('inputs') or []
    return (inputs[0].get('camera') if inputs else None) or (episode or {}).get('camera') or next((row.get('camera_id') for row in rows if row.get('camera_id')), None)


def expected_camera_sources(job, rows=()):
    inputs = job.get('inputs') or []
    if not inputs:
        return {}
    if len(inputs) == 1 and not inputs[0].get('camera'):
        cameras = {row.get('camera_id') for row in rows if isinstance(row.get('camera_id'), str)}
        if len(cameras) > 1:
            raise ValueError('Single input cannot be associated with multiple camera ids')
        camera = next(iter(cameras), '*')
        return {camera: inputs[0].get('sha256')}
    expected = {}
    for item in inputs:
        camera, source_hash = item.get('camera'), item.get('sha256')
        if not isinstance(camera, str) or not camera or camera in expected:
            raise ValueError('Each uploaded input must have one unique camera id')
        expected[camera] = source_hash
    if any(not isinstance(value, str) or not re.fullmatch(r'[a-f0-9]{64}', value) for value in expected.values()):
        raise ValueError('Each uploaded input must have a valid SHA256')
    return expected


def validate_row_camera_sources(rows, job):
    expected = expected_camera_sources(job, rows)
    if not expected:
        return
    for row in rows:
        if row.get('accepted') is not True:
            continue
        camera = row.get('camera_id')
        expected_hash = expected.get(camera, expected.get('*'))
        if not isinstance(camera, str) or expected_hash is None or row.get('source_sha256') != expected_hash:
            raise ValueError('Frame source SHA256 does not match its camera input')


def validate_manifest_camera_sources(manifest, job, rows=()):
    expected = expected_camera_sources(job, rows)
    if not expected:
        return
    cameras = manifest.get('cameras') if isinstance(manifest, dict) else None
    if not isinstance(cameras, list) or not cameras:
        raise ValueError('Completed manifest is missing camera source receipts')
    actual = {}
    for camera in cameras:
        if not isinstance(camera, dict) or not isinstance(camera.get('id'), str) or camera['id'] in actual:
            raise ValueError('Manifest camera source receipts are invalid')
        actual[camera['id']] = camera.get('source_sha256')
    if '*' in expected:
        expected = {next(iter(actual), ''): expected['*']}
    if actual != expected:
        raise ValueError('Manifest camera/source mapping does not match uploaded inputs')


def actual_result(rows, job, episode=None, elapsed_ms=None):
    validate_row_camera_sources(rows, job)
    frames_by_camera, track_ids, detector_times = {}, set(), []
    primary_camera = primary_camera_for_job(job, rows, episode)
    for row in rows:
        if row.get('accepted') is not True:
            continue
        width, height = row['width'], row['height']
        detections = []
        raw_detections = row.get('stages', {}).get('detector', {}).get('detections', [])
        matches = observed_track_matches(raw_detections, row.get('tracks', []))
        for detection_index, detection in enumerate(raw_detections):
            track_id = matches.get(detection_index)
            if track_id is not None:
                track_ids.add((row['camera_id'], track_id))
            xyxy = detection['xyxy']
            detections.append({'box': [xyxy[0] / width, xyxy[1] / height, xyxy[2] / width, xyxy[3] / height],
                               'score': detection['score'], 'trackId': track_id, 'label': detection['label']})
        frame = {'time': row.get('local_timestamp_s', row['timestamp_s']), 'camera': row['camera_id'], 'detections': detections}
        if job.get('input_kind') == 'image_set':
            frame['image_index'] = row.get('frame_index')
        frames_by_camera.setdefault(row['camera_id'], []).append(frame)
        latency = row.get('stages', {}).get('detector', {}).get('elapsed_ms')
        if isinstance(latency, (float, int)) and math.isfinite(latency):
            detector_times.append(latency)
    zone = job.get('zoneSnapshot')
    return {'id': job['id'], 'sampleId': job.get('sampleId'), 'frames': frames_by_camera.get(primary_camera, []),
            'cameraFrames': frames_by_camera, 'episode': episode, 'videos': job.get('videos', []), 'primaryCamera': primary_camera, 'sampleFrequency': job.get('Vfreq', 1),
            'inputKind': job.get('input_kind', 'video'), 'syntheticTime': bool(job.get('synthetic_time')),
            'sourceSHA256s': job.get('source_sha256s', []), 'sourceImages': job.get('sourceImages', []),
            'model': 'LinQ: RF-DETR person TPU; task-specific object detectors and optional clip head CPU',
            'device': 'LinQ CPU / TPU', 'tracker': 'Actual causal tracker records',
            'zone': zone, 'zoneId': zone.get('zone_id') if zone else None,
            'trackAssociation': {'method': 'Greedy descending IoU within the same label and frame', 'minimumIoU': 0.3, 'observedOnly': True, 'oneToOne': True},
            'requestedZone': zone, 'zoneAppliedToInference': bool(zone),
            'modelSha256': None, 'objectModelSha256': None,
            'metrics': {'frames': sum(len(items) for items in frames_by_camera.values()), 'tracks': len(track_ids), 'frameP50Ms': None, 'frameP95Ms': None,
                        'detectorP50Ms': statistics.median(detector_times) if detector_times else None,
                        'inferenceWallMs': None, 'totalMs': elapsed_ms}}


def provisional_candidate_events(progress, rows, job):
    summaries = progress.get('candidate_events', [])
    if not isinstance(summaries, list) or len(summaries) > 20:
        return []
    source_by_camera = {item.get('camera'): item.get('sha256') for item in job.get('inputs', [])}
    verified = []
    for item in summaries:
        if not isinstance(item, dict) or item.get('event_trigger') != 'active_event':
            continue
        event_id, camera_id, rule_id = item.get('event_id'), item.get('camera_id'), item.get('rule_id')
        frame_index, timestamp, source_hash = item.get('frame_index'), item.get('trigger_timestamp_s'), item.get('source_sha256')
        if (not isinstance(event_id, str) or not re.fullmatch(r'[a-f0-9]{24}', event_id)
                or not isinstance(camera_id, str) or not isinstance(rule_id, str)
                or not re.fullmatch(r'[A-Za-z0-9_.:-]{1,128}', rule_id)
                or not isinstance(frame_index, int) or isinstance(frame_index, bool) or frame_index < 0
                or not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp) or timestamp < 0
                or source_by_camera.get(camera_id) != source_hash):
            continue
        row = next((row for row in rows if row.get('camera_id') == camera_id
                    and row.get('frame_index') == frame_index and row.get('source_sha256') == source_hash
                    and row.get('accepted') is True and isinstance(row.get('timestamp_s'), (int, float))
                    and math.isfinite(row['timestamp_s']) and math.isclose(row['timestamp_s'], timestamp, rel_tol=0, abs_tol=1e-6)
                    and row.get('stages', {}).get('detector', {}).get('status') == 'ok'
                    and row.get('stages', {}).get('detector', {}).get('actual_execution') is True
                    and row.get('stages', {}).get('detector', {}).get('mode') == 'fresh_inference'
                    and any(indicator.get('id') == rule_id and indicator.get('state') == 'active'
                            for indicator in row.get('indicators', []))), None)
        if row is None:
            continue
        score = item.get('score')
        if not isinstance(score, (int, float)) or not math.isfinite(score):
            score = None
        track_ids = item.get('track_ids') if isinstance(item.get('track_ids'), list) else []
        track_ids = [track_id for track_id in track_ids[:64]
                     if isinstance(track_id, (str, int)) and not isinstance(track_id, bool)]
        vlm_status = item.get('vlm_status') if isinstance(item.get('vlm_status'), str) else 'unknown'
        verified.append({
            'event_id': event_id, 'camera_id': camera_id, 'rule_id': rule_id, 'kind': 'incident_candidate',
            'event_trigger': 'active_event', 'frame_index': frame_index, 'trigger_timestamp_s': float(timestamp),
            'source_sha256': source_hash, 'state': item.get('state', 'active'), 'score': score,
            'track_ids': track_ids, 'vlm': {'status': vlm_status, 'actual_execution': False},
            'provisional': True, 'semantics': 'Pipeline-created incident candidate from fresh inference; final event record pending.',
        })
    by_id = {event['event_id']: event for event in verified}
    return sorted(by_id.values(), key=lambda event: (event['trigger_timestamp_s'], event['camera_id'], event['event_id']))


def provisional_episode(event, job):
    trigger_time = event['trigger_timestamp_s']
    episode = {
        'start': trigger_time, 'end': trigger_time, 'triggerTime': trigger_time,
        'frames': [], 'frameTimes': [], 'frameSHA256': [], 'reason': event['rule_id'],
        'eventId': event['event_id'], 'trackIds': event['track_ids'], 'camera': event['camera_id'],
        'unsafe': None, 'status': event['vlm']['status'], 'responseValid': False,
        'operatorConfirmed': None, 'provisional': True, 'semantics': event['semantics'],
    }
    if job.get('input_kind') == 'image_set':
        image_index = event['frame_index']
        if 0 <= image_index < len(job.get('sourceImages', [])):
            episode['image_index'] = image_index
            episode['imageIndices'] = [image_index]
    return episode


def merge_candidate_events(existing, incoming, limit=100):
    by_id = {item['event_id']: item for item in existing if isinstance(item, dict) and item.get('event_id')}
    by_id.update({item['event_id']: item for item in incoming})
    return sorted(by_id.values(), key=lambda event: (event.get('trigger_timestamp_s', 0), event.get('camera_id', '')))[-limit:]


def dedupe_event_records(events):
    by_id = {}
    for event in events:
        event_id = event.get('event_id') if isinstance(event, dict) else None
        if isinstance(event_id, str) and event_id:
            by_id[event_id] = event
    return sorted(by_id.values(), key=lambda event: (event.get('trigger_timestamp_s', 0), event.get('camera_id', ''), event['event_id']))


def intermediate_result(rows, job, progress=None, manifest=None, elapsed_ms=None):
    progress = progress or {}
    for proof in [progress, manifest or {}]:
        if proof and (proof.get('mode') != 'causal_fresh_video_inference' or proof.get('replay_fallback') is not False):
            raise ValueError('Intermediate results require fresh inference without replay')
    validate_row_camera_sources(rows, job)
    if manifest and manifest.get('status') == 'completed':
        validate_manifest_camera_sources(manifest, job, rows)
    accepted, identities = [], set()
    for row in rows:
        if row.get('accepted') is not True:
            continue
        detector = row.get('stages', {}).get('detector', {})
        if detector.get('mode') not in {None, 'fresh_inference'} or detector.get('status') == 'ok' and detector.get('actual_execution') is not True:
            raise ValueError('Intermediate detector record is not actual fresh execution')
        if detector.get('detections') and (detector.get('actual_execution') is not True or detector.get('mode') != 'fresh_inference'):
            raise ValueError('Intermediate boxes have no actual fresh execution proof')
        identity = row['camera_id'], row['frame_index']
        if identity in identities:
            raise ValueError('Duplicate intermediate frame record')
        identities.add(identity)
        accepted.append(row)
    completed = (manifest or {}).get('status') == 'completed'
    cv_ready = progress.get('cv_ready') is True or completed
    pending = progress.get('vlm_pending') is True and not completed
    stage = 'rendering' if completed else 'analyzing' if cv_ready else 'detecting'
    live_candidates = provisional_candidate_events(progress, accepted, job)
    merged_events = merge_candidate_events(job.get('events', []), live_candidates)
    episodes = [provisional_episode(event, job) for event in merged_events if event.get('provisional') is True]
    result = actual_result(accepted, job, elapsed_ms=elapsed_ms) if accepted or cv_ready else None
    if result is not None:
        result['episodes'] = episodes
        primary_camera = primary_camera_for_job(job, accepted)
        primary_episodes = [episode for episode in episodes if episode.get('camera') == primary_camera]
        result['episode'] = max(primary_episodes, key=lambda episode: episode['triggerTime']) if primary_episodes else None
        result['episodeSelection'] = {'policy': 'latest_live_pipeline_candidate', 'camera': primary_camera,
                                      'eventId': result['episode']['eventId'] if result['episode'] else None,
                                      'semantics': 'Provisional actual candidate; final event collection may update this record'}
    return {'status': 'running', 'stage': stage, 'cv_ready': cv_ready, 'vlm_pending': pending,
            'cv_completed_at': progress.get('cv_completed_at'), 'progress': 80 if completed else 75 if cv_ready else 40,
            'events': merged_events, 'candidateEventCount': len(episodes), 'candidateEvents': merged_events,
            'result': result,
            'analysis': None, 'analysisError': None}


def create_app(root=ROOT):
    root = Path(root).resolve()
    directory = root / 'artifacts/private/demo_web/jobs'
    directory.mkdir(parents=True, exist_ok=True)
    app = FastAPI(title='LinQ video demo')
    @app.exception_handler(HTTPException)
    async def http_error(request, error):
        return JSONResponse(status_code=error.status_code, content={'detail': error.detail, 'error': {'code': 'HTTP_' + str(error.status_code), 'message': str(error.detail)}})

    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='linq-web-job')
    lock = threading.RLock()
    state = {'active': None, 'health': {'linq_available': False, 'vlm_ready': False}, 'health_at': 0}
    workspace = directory.parent / 'workspace.json'

    def workspace_state():
        return json.loads(workspace.read_text()) if workspace.exists() else {'reviews': [], 'zones': {}}

    def catalog():
        values = []
        for name in ['samples.json', 'object-references.json', 'samples.local.json']:
            path = root / 'service' / name
            if path.exists():
                values.extend({key: value for key, value in sample.items() if key != 'boxes'} for sample in json.loads(path.read_text()))
        for path in directory.glob('*/job.json'):
            job = json.loads(path.read_text())
            if job.get('sampleId', '').startswith('upload_') and job.get('inputs'):
                first = job['inputs'][0]
                values.append({'id': job['sampleId'], 'eventId': job['id'][:8], 'case': task_case(job['task']),
                               'task': job['task'], 'taskTitle': task_title(job['task']),
                               'title': 'Загруженное видео · ' + task_title(job['task']),
                               'video': '/input/' + job['id'] + '/' + Path(first['path']).name,
                               'image': job.get('videos', [{}])[0].get('poster', '') if job.get('videos') else '',
                               'duration': job.get('duration', 0), 'source': 'Пользовательское видео', 'sourceUrl': '#', 'originalFile': first['sha256'],
                               'rule': 'Проверяется выбранной обученной задачей; решение оператора отдельно.'})
            elif job.get('input_kind') == 'image_set' and job.get('sourceImages'):
                images = [{key: value for key, value in item.items() if key in {'name', 'sha256', 'bytes', 'format', 'width', 'height', 'url'}}
                          for item in job['sourceImages']]
                values.append({'id': job['sampleId'], 'eventId': job['id'][:8], 'case': task_case(job['task']),
                               'task': job['task'], 'taskTitle': task_title(job['task']),
                               'title': 'Набор изображений · ' + task_title(job['task']), 'inputKind': 'image_set', 'syntheticTime': True,
                               'image': images[0]['url'], 'sourceImages': images, 'duration': job.get('duration', 0),
                               'source': 'Пользовательский набор стоп-кадров', 'sourceUrl': '#', 'originalFile': job.get('source_sha256s', []),
                               'rule': 'Статическое присутствие объекта; временные действия по синтетическому порядку не оцениваются.'})
        return values

    def facade(job):
        mapping = {'queued': 'detecting', 'uploading': 'detecting', 'running': 'detecting', 'collecting': 'analyzing', 'completed': 'ready', 'error': 'failed'}
        status = 'analyzing' if job['status'] == 'running' and job.get('cv_ready') is True else mapping.get(job['status'], 'detecting')
        candidates = [view for event in job.get('events', [])
                      if (view := candidate_event_public_view(event)) is not None]
        value = {'id': job['id'], 'sampleId': job.get('sampleId', 'upload_' + job['id']), 'status': status,
                 'task': job.get('task'), 'taskTitle': task_title(job.get('task')), 'case': task_case(job.get('task')),
                 'input_kind': job.get('input_kind', 'video'), 'inputKind': job.get('input_kind', 'video'),
                 'syntheticTime': bool(job.get('synthetic_time')),
                 'sourceSHA256s': job.get('source_sha256s', []), 'sourceImages': job.get('sourceImages', []),
                 'candidateEventCount': len(candidates), 'candidateEvents': candidates,
                 'createdAt': job['created_at'], 'finishedAt': job.get('completed_at'), 'progress': job.get('progress'), 'result': job.get('result'),
                 'cv_ready': job.get('cv_ready', False), 'vlm_pending': job.get('vlm_pending', False), 'stage': job.get('stage'), 'cv_completed_at': job.get('cv_completed_at'),
                 'analysis': job.get('analysis'), 'analysisError': job.get('analysisError'),
                 'sceneReviewStatus': job.get('sceneReviewStatus'), 'mediaSha256': job.get('inputs', [{}])[0].get('sha256')}
        if job.get('error'):
            value['error'] = job['error'].get('message', str(job['error'])) if isinstance(job['error'], dict) else str(job['error'])
        return value

    def metadata_path(job_id):
        return directory / check_identity(job_id) / 'job.json'

    def load_job(job_id):
        path = metadata_path(job_id)
        if not path.is_file():
            raise HTTPException(404, 'Unknown job')
        with lock:
            return json.loads(path.read_text())

    def update(job_id, **values):
        with lock:
            job = load_job(job_id)
            job.update(values, updated_at=now())
            atomic_json(metadata_path(job_id), job)
            return job

    def tasks():
        config_path = root / 'config_single.json'
        config = json.loads(config_path.read_text()) if config_path.is_file() else {'cameras': [{'classifier': {'bundle': 'models/convnext'}}]}
        return [{'id': name, 'title_ru': head.get('title_ru', TITLES.get(name, name)), 'frames': head['frames'], 'classes': head['classes'],
                 'scope': head.get('task_spec', {}).get('scope'), 'semantics': head.get('task_spec', {}).get('label_semantics'),
                 'backend': 'detector' if head.get('detector_profile') else 'trained_clip_head'}
                for name, head in available_tasks(root, config).items()]

    def all_events():
        events = {}
        for path in sorted(directory.glob('*/job.json')):
            job = json.loads(path.read_text())
            for original in job.get('events', []):
                event = dict(original, job_id=job['id'])
                event['job_created_at'] = job.get('created_at')
                zone = job.get('zoneSnapshot')
                zone_applies = bool(zone and original.get('camera_id') == zone.get('camera_id'))
                event.setdefault('sample_id', job.get('sampleId'))
                event['zone_id'] = zone.get('zone_id') if zone_applies else None
                event['zone_polygon'] = zone.get('polygon') if zone_applies else None
                event['source_camera_id'] = zone.get('camera_id') if zone_applies else None
                event.setdefault('input_kind', job.get('input_kind', 'video'))
                event.setdefault('synthetic_time', bool(job.get('synthetic_time')))
                event.setdefault('source_sha256s', job.get('source_sha256s', []))
                event['type'] = event_type(event)
                if job.get('input_kind') == 'image_set':
                    event['image_index'] = event_image_index(original, len(job.get('sourceImages', [])))
                event_id = str(event.get('event_id', ''))
                review_file = path.parent / 'reviews' / (hashlib.sha256(event_id.encode()).hexdigest() + '.json')
                event['review'] = json.loads(review_file.read_text()) if review_file.exists() else None
                event['operator_confirmed'] = event['review']['confirmed'] if event['review'] else None
                event['semantics'] = 'Model candidate or inspection; operator review is separate'
                events[(job['id'], event_id)] = event
        return sorted(events.values(), key=lambda item: (
            str(item.get('job_created_at') or ''),
            item.get('trigger_timestamp_s') if isinstance(item.get('trigger_timestamp_s'), (int, float))
            and math.isfinite(item['trigger_timestamp_s']) else -1,
        ), reverse=True)

    def worker(job_id):
        started = time.monotonic()
        try:
            job = load_job(job_id)
            client = linq_client.connect(root / '.secrets/linq_jupyter_session.json', root / '.secrets/linq_jupyter_capabilities.json')
            uploads = []
            for index, source in enumerate(job['inputs']):
                update(job_id, status='uploading', stage='uploading')
                def progress(fraction):
                    update(job_id, progress=5 + 30 * (index + fraction) / len(job['inputs']))
                uploads.append(upload_video(client, Path(source['path']), job_id, 'camera_' + str(index + 1).zfill(2), progress))
            run_name = 'user_web_' + job_id
            command = ['.venv/bin/python', 'scripts/run_user_video.py', '--videos', *[item['video'] for item in uploads], '--task', job['task'], '--V', str(job['Vfreq']), '--name', run_name]
            if job.get('zoneSnapshot'):
                command.extend(['--zone-json', json.dumps(job['zoneSnapshot'], separators=(',', ':'))])
            if job.get('input_kind') == 'image_set':
                command.extend(['--input-kind', 'image_set', '--source-sha256s-json', json.dumps(job['source_sha256s'], separators=(',', ':'))])
            if not job['use_vlm']:
                command.append('--cv-only')
            code = """from pathlib import Path
import json,subprocess,sys
root=Path.home()/'work/mosprom_hackathon_linq'
sys.path.insert(0,str(root))
from scripts.linq_prototype import environment
result=subprocess.run(%r,cwd=root,env=environment(root),capture_output=True,text=True,timeout=120)
if result.returncode:raise RuntimeError(result.stderr[-1500:])
print(json.dumps(json.loads(result.stdout)))
""" % command
            launch = execute_checked(client, code, timeout=150)
            update(job_id, status='running', stage='detecting', progress=40, cv_ready=False, vlm_pending=False, upload_receipts=uploads, remote_run=run_name, launch={'pid': launch.get('pid'), 'status': launch.get('status')})
            remote_run = REMOTE + '/runs/' + run_name
            deadline, failures = time.monotonic() + 21600, 0
            while True:
                if time.monotonic() > deadline:
                    raise TimeoutError('LinQ run exceeded six hours')
                try:
                    manifest = remote_json(client, remote_run + '/manifest.json')
                    summary = remote_json(client, remote_run + '/view/summary.json') if manifest and manifest.get('status') == 'completed' else None
                    progress = remote_json(client, remote_run + '/progress.json')
                    live_rows = remote_frame_rows(client, remote_run + '/frames.jsonl')
                    failures = 0
                except Exception:
                    failures += 1
                    if failures >= 5:
                        raise
                    time.sleep(3)
                    continue
                if manifest and manifest.get('status') == 'error':
                    raise RuntimeError('LinQ inference failed: ' + str(manifest.get('error_type', 'unknown')) + ': ' + str(manifest.get('error', ''))[:1000])
                current_job = load_job(job_id)
                update(job_id, **intermediate_result(live_rows, current_job, progress, manifest, (time.monotonic() - started) * 1000))
                if manifest and manifest.get('status') == 'completed':
                    if manifest.get('mode') != 'causal_fresh_video_inference' or manifest.get('replay_fallback') is not False:
                        raise ValueError('Expected fresh LinQ execution without replay')
                    update(job_id, stage='rendering', progress=80)
                    if summary and summary.get('pipeline_status') == 'completed':
                        break
                time.sleep(1)
            update(job_id, status='collecting', stage='collecting', progress=90)
            results = directory / job_id / 'results'
            results.mkdir(exist_ok=True)
            atomic_json(results / 'manifest.json', manifest)
            atomic_json(results / 'summary.json', summary)
            vlm = remote_json(client, remote_run + '/view/vlm_results.json')
            if vlm is None and job.get('use_vlm', True):
                raise ValueError('Actual event results are missing')
            if vlm is None:
                vlm = {'source_events': [], 'results': [], 'status': 'disabled_by_request'}
            atomic_json(results / 'vlm_results.json', vlm)
            download_file(client, remote_run + '/frames.jsonl', results / 'frames.jsonl')
            rows = [json.loads(line) for line in (results / 'frames.jsonl').read_text().splitlines() if line.strip()]
            validate_row_camera_sources(rows, job)
            validate_manifest_camera_sources(manifest, job, rows)
            videos, media_files = [], []
            for item in summary['videos']:
                video, poster = safe_filename(item['video']), safe_filename(item['poster'])
                for name in [video, poster]:
                    download_file(client, remote_run + '/view/' + name, results / name)
                    media_files.append(name)
                source = next((entry for entry in job.get('inputs', []) if entry.get('camera') == item['camera_id']), None)
                videos.append({'url': '/media/' + job_id + '/' + video, 'poster': '/media/' + job_id + '/' + poster,
                               'sourceUrl': '/input/' + job_id + '/' + Path(source['path']).name if source and job.get('input_kind') != 'image_set' else None,
                               'camera': item['camera_id']})
            episode, analysis, analysis_error, scene_review_status = None, None, None, None
            events = dedupe_event_records(vlm.get('source_events', []))
            vlm['source_events'] = events
            if job.get('task') in ACTION_REVIEW_TASKS:
                for event in events:
                    if not str(event.get('rule_id', '')).startswith('zone_'):
                        event.update(action_review_fields(event, rows, job.get('task')))
            if job.get('input_kind') == 'image_set':
                source_count = len(job.get('sourceImages', []))
                for event in events:
                    event['image_index'] = event_image_index(event, source_count)
            primary_camera = primary_camera_for_job(job, rows)
            evidence_events = [event for event in events if event.get('raw_frames') or
                               not job.get('use_vlm', True) and event.get('rule_id') != 'inspection']
            episodes = []
            for event in evidence_events:
                collected, files = collect_event_episode(client, remote_run, results, job_id, event)
                if job.get('input_kind') == 'image_set':
                    collected['image_index'] = event_image_index(event, len(job.get('sourceImages', [])))
                if job.get('task') in ACTION_REVIEW_TASKS:
                    if not str(event.get('rule_id', '')).startswith('zone_'):
                        collected.update(action_review_fields(event, rows, job.get('task')))
                        collected['unsafe'] = None
                episodes.append(collected)
                media_files.extend(files)
            selected, selection_policy = select_model_event([event for event in events if event.get('camera_id') == primary_camera])
            if selected is not None:
                episode = next((item for item in episodes if item['eventId'] == selected['event_id'] and item['camera'] == selected['camera_id']), None)
                if episode is not None:
                    episode['summarySelection'] = selection_policy
                if not job.get('use_vlm', True):
                    scene_review_status = 'disabled_by_operator'
                elif str(selected.get('rule_id', '')).startswith('zone_'):
                    analysis_error = 'Geometric zone candidate is available for operator review; no action assessment was requested.'
                else:
                    target_detected = event_has_target_detection(selected, rows, job.get('task'))
                    analysis, analysis_error = analysis_from_event(selected, selection_policy, job.get('task'), target_detected)
            elif not job.get('use_vlm', True):
                scene_review_status = 'disabled_by_operator'
            result = actual_result(rows, {**job, 'videos': videos}, episode, (time.monotonic() - started) * 1000)
            result['episodes'] = episodes
            result['episodeSelection'] = {'policy': selection_policy, 'camera': primary_camera,
                                          'eventId': selected['event_id'] if selected is not None else None,
                                          'semantics': 'Model candidates; later negative does not cancel an earlier positive'}
            update(job_id, status='completed', stage='completed', progress=100, videos=videos, media_files=media_files,
                   cv_ready=True, vlm_pending=False,
                   events=events, vlm_results=vlm.get('results', []), summary=summary, result=result, analysis=analysis,
                   analysisError=analysis_error, sceneReviewStatus=scene_review_status, completed_at=now(), error=None)
        except Exception as error:
            update(job_id, status='error', stage='error', error={'type': type(error).__name__, 'message': str(error)[:1500]})
        finally:
            with lock:
                state['active'] = None

    @app.get('/api/status')
    def status():
        with lock:
            if time.monotonic() - state['health_at'] > 30:
                state['health_at'] = time.monotonic()
                state['health'] = {'linq_available': False, 'vlm_ready': False}
                try:
                    client = linq_client.connect(root / '.secrets/linq_jupyter_session.json', root / '.secrets/linq_jupyter_capabilities.json')
                    linq_client.request(client, 'GET', 'api/status', timeout=4)
                    state['health'] = {'linq_available': True, 'vlm_ready': False}
                    code = "from pathlib import Path\nimport sys,json\nroot=Path.home()/'work/mosprom_hackathon_linq'\nsys.path.insert(0,str(root))\nfrom scripts.linq_prototype import health\nprint(json.dumps(health(root)))"
                    health = execute_checked(client, code, timeout=15)
                    health = health if isinstance(health, dict) else {}
                    execution = health.get('execution')
                    execution = execution if isinstance(execution, dict) else {}
                    model = execution.get('model') or health.get('model')
                    device = execution.get('device') or health.get('device')
                    state['health'] = {
                        'linq_available': True,
                        'vlm_ready': health.get('ready') is True,
                        'vlm_model': model.strip()[:160] if isinstance(model, str) and model.strip() else None,
                        'vlm_device': device.strip()[:64] if isinstance(device, str) and device.strip() else None,
                    }
                except Exception as error:
                    state['health'] = {'linq_available': state['health'].get('linq_available', False), 'vlm_ready': False, 'error_type': type(error).__name__}
            return {**state['health'], 'busy': state['active'] is not None, 'configured': state['health'].get('linq_available', False),
                    'provider': 'LinQ CPU', 'model': 'Qwen3.5-0.8B'}

    @app.get('/api/tasks')
    def list_tasks():
        return {'tasks': tasks()}

    @app.post('/api/jobs', status_code=202)
    async def create_job(files: list[UploadFile] = File(...), task: str = Form('synthsite'), use_vlm: bool = Form(True), Vfreq: float = Form(1)):
        if not 1 <= len(files) <= 2:
            raise HTTPException(400, 'Upload one or two videos')
        if task not in {item['id'] for item in tasks()} or not math.isfinite(Vfreq) or Vfreq <= 0:
            raise HTTPException(400, 'Invalid task or sampling frequency')
        if any(Path(item.filename or '').suffix.lower() not in EXTENSIONS for item in files):
            raise HTTPException(400, 'Supported video extensions: mp4, mov, mkv, avi, webm')
        job_id = uuid.uuid4().hex
        with lock:
            if state['active'] is not None:
                raise HTTPException(409, 'Another video job is active')
            state['active'] = job_id
        job_dir = directory / job_id
        try:
            inputs = []
            for index, upload in enumerate(files):
                path = job_dir / 'input' / ('camera_' + str(index + 1).zfill(2) + Path(upload.filename).suffix.lower())
                path.parent.mkdir(parents=True, exist_ok=True)
                size = 0
                with path.open('xb') as stream:
                    while chunk := await upload.read(1024 * 1024):
                        size += len(chunk)
                        if size > MAX_FILE:
                            raise HTTPException(413, 'Each video must be at most 200 MiB')
                        stream.write(chunk)
                if not size:
                    raise HTTPException(400, 'Empty video')
                inputs.append({'path': str(path), 'bytes': size, 'sha256': digest(path), 'camera': 'user_camera_' + str(index + 1).zfill(2)})
            job = {'id': job_id, 'created_at': now(), 'updated_at': now(), 'status': 'queued', 'stage': 'queued', 'progress': 0,
                   'cv_ready': False, 'vlm_pending': False,
                   'sampleId': 'upload_' + job_id, 'task': task, 'use_vlm': use_vlm, 'Vfreq': Vfreq, 'inputs': inputs, 'videos': [], 'events': [], 'media_files': [], 'error': None}
            try:
                probe = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'json', inputs[0]['path']], capture_output=True, text=True, timeout=5)
                job['duration'] = float(json.loads(probe.stdout)['format']['duration'])
            except Exception:
                job['duration'] = 0
            atomic_json(job_dir / 'job.json', job)
            executor.submit(worker, job_id)
            return public_job(job)
        except Exception:
            with lock:
                state['active'] = None
            raise
        finally:
            for upload in files:
                await upload.close()

    @app.post('/api/image-jobs', status_code=202)
    async def create_image_job(files: list[UploadFile] = File(...), task: str = Form('phone_objects'),
                               use_vlm: bool = Form(True)):
        if not 2 <= len(files) <= 24:
            raise HTTPException(400, 'Upload an ordered set of two to 24 still images')
        if task not in {'phone_objects', 'cigarette_objects'}:
            raise HTTPException(422, 'Still-image sets support only phone or cigarette presence profiles; action and motion tasks require video')
        job_id = uuid.uuid4().hex
        with lock:
            if state['active'] is not None:
                raise HTTPException(409, 'Another video job is active')
            state['active'] = job_id
        job_dir = directory / job_id
        input_dir = job_dir / 'input'
        image_dir = input_dir / 'images'
        sources, paths, total = [], [], 0
        try:
            image_dir.mkdir(parents=True, exist_ok=False)
            for index, upload in enumerate(files, start=1):
                client_name = upload.filename or ''
                if (not client_name or Path(client_name).name != client_name or '/' in client_name or '\\' in client_name
                        or len(client_name) > 255 or any(ord(char) < 32 for char in client_name)):
                    raise HTTPException(400, 'Each image must have a simple filename')
                extension = Path(client_name).suffix.lower()
                if extension not in {'.jpg', '.jpeg', '.png'}:
                    raise HTTPException(400, 'Still images must be JPEG or PNG files')
                stored_name = 'frame_' + str(index).zfill(2) + extension
                target = image_dir / stored_name
                size = 0
                with target.open('xb') as stream:
                    while chunk := await upload.read(1024 * 1024):
                        size += len(chunk)
                        total += len(chunk)
                        if size > MAX_IMAGE_BYTES:
                            raise HTTPException(413, 'Each image must be at most 20 MiB')
                        if total > MAX_TOTAL_BYTES:
                            raise HTTPException(413, 'An image set must be at most 100 MiB total')
                        stream.write(chunk)
                if not size:
                    raise HTTPException(400, 'Empty image')
                paths.append(target)
                sources.append({'stored_name': stored_name, 'uploaded_name': client_name, 'extension': extension})
            prepared = prepare_image_set(paths, input_dir / 'prepared', camera_id='user_camera_01')
            if prepared['video_sha256'] is None or Path(prepared['video_path']).stat().st_size > MAX_FILE:
                raise HTTPException(413, 'Prepared image-set video exceeds the 200 MiB job limit')
            descriptors = []
            for source, receipt in zip(sources, prepared['source_images']):
                expected_format = 'JPEG' if source['extension'] in {'.jpg', '.jpeg'} else 'PNG'
                if receipt['format'] != expected_format:
                    raise HTTPException(400, 'Image filename extension does not match its encoded content')
                descriptors.append({'name': source['uploaded_name'], 'sha256': receipt['sha256'],
                                    'bytes': receipt['bytes'], 'format': receipt['format'],
                                    'width': receipt['width'], 'height': receipt['height'],
                                    'url': '/input/' + job_id + '/' + source['stored_name']})
            source_sha256s = [item['sha256'] for item in descriptors]
            sample_id = 'image_set_' + job_id
            job = {'id': job_id, 'created_at': now(), 'updated_at': now(), 'status': 'queued', 'stage': 'queued', 'progress': 0,
                   'cv_ready': False, 'vlm_pending': False, 'sampleId': sample_id, 'task': task, 'use_vlm': use_vlm,
                   'Vfreq': 1, 'duration': prepared['duration_seconds'],
                   'inputs': [{'path': prepared['video_path'], 'bytes': Path(prepared['video_path']).stat().st_size,
                               'sha256': prepared['video_sha256'], 'camera': 'user_camera_01'}],
                   'input_kind': 'image_set', 'synthetic_time': True, 'source_sha256s': source_sha256s,
                   'sourceImages': descriptors, 'imageSourceFiles': [item['stored_name'] for item in sources],
                   'imageSetReceipt': str(prepared['receipt_path']), 'events': [], 'videos': [], 'media_files': [], 'error': None}
            atomic_json(input_dir / 'image-set-receipt.json', prepared)
            atomic_json(job_dir / 'job.json', job)
            executor.submit(worker, job_id)
            return public_job(job)
        except HTTPException:
            with lock:
                state['active'] = None
            shutil.rmtree(job_dir, ignore_errors=True)
            raise
        except ValueError as error:
            with lock:
                state['active'] = None
            shutil.rmtree(job_dir, ignore_errors=True)
            raise HTTPException(400, str(error)) from error
        except RuntimeError as error:
            with lock:
                state['active'] = None
            shutil.rmtree(job_dir, ignore_errors=True)
            raise HTTPException(503, str(error)) from error
        except Exception:
            with lock:
                state['active'] = None
            shutil.rmtree(job_dir, ignore_errors=True)
            raise
        finally:
            for upload in files:
                await upload.close()

    def public_job(job):
        value = {key: item for key, item in job.items() if key not in {'inputs', 'upload_receipts', 'launch', 'media_files', 'imageSourceFiles', 'imageSetReceipt'}}
        candidates = [view for event in job.get('events', [])
                      if (view := candidate_event_public_view(event)) is not None]
        value['candidateEventCount'] = len(candidates)
        value['candidateEvents'] = candidates
        return value

    @app.get('/api/samples')
    def samples():
        return {'samples': catalog()}

    @app.get('/api/pipeline')
    def pipeline():
        stored = workspace_state()
        return {'jobs': [facade(json.loads(path.read_text())) for path in sorted(directory.glob('*/job.json'), key=lambda path: path.stat().st_mtime, reverse=True)],
                'reviews': stored['reviews'], 'zones': stored['zones']}

    @app.post('/api/pipeline', status_code=202)
    def start_sample(request: PipelineRequest):
        sample = next((item for item in catalog() if item['id'] == request.sampleId), None)
        if sample is None:
            raise HTTPException(422, 'Unknown sample')
        if not sample.get('video'):
            raise HTTPException(422, 'Image-only inference is not supported by this video pipeline')
        task = request.task if request.task is not None else {'fall': 'omnifall_v2', 'proximity': 'physicalai_v2', 'phone': 'phone_use', 'smoking': 'smoking'}.get(sample['case'], sample['case'])
        if task not in {item['id'] for item in tasks()}:
            raise HTTPException(422, 'No trained video task for this sample')
        if sample['video'].startswith('/input/'):
            source_job = load_job(sample['video'].split('/')[2])
            paths = [Path(item['path']) for item in source_job['inputs']]
        else:
            path = (root / 'service' / sample['video']).resolve()
            if not path.is_relative_to((root / 'service/assets').resolve()) or not path.is_file():
                raise HTTPException(422, 'Local source video is unavailable')
            paths = [path]
        if any(path.stat().st_size > MAX_FILE for path in paths):
            raise HTTPException(413, 'Each source must be at most 200 MiB')
        job_id = uuid.uuid4().hex
        with lock:
            if state['active'] is not None:
                raise HTTPException(409, 'Another video job is active')
            state['active'] = job_id
            saved_polygon = workspace_state()['zones'].get(sample['id'])
            zone_snapshot = {'zone_id': sample['id'], 'camera_id': 'user_camera_01', 'polygon': saved_polygon} if saved_polygon else None
            job = {'id': job_id, 'sampleId': sample['id'], 'created_at': now(), 'updated_at': now(), 'status': 'queued', 'stage': 'queued',
                   'cv_ready': False, 'vlm_pending': False,
                   'progress': 0, 'task': task, 'use_vlm': request.use_vlm, 'Vfreq': 1, 'duration': sample.get('duration'),
                   'inputs': [{'path': str(path), 'sha256': digest(path), 'bytes': path.stat().st_size,
                               'camera': 'user_camera_' + str(index + 1).zfill(2)} for index, path in enumerate(paths)],
                   'zoneSnapshot': zone_snapshot, 'videos': [], 'events': [], 'media_files': [], 'error': None}
            atomic_json(metadata_path(job_id), job)
            executor.submit(worker, job_id)
        return {'job': facade(job)}

    @app.post('/api/reviews')
    def facade_review(request: JobReview):
        job = load_job(request.jobId)
        if job['status'] != 'completed' or not job.get('result', {}).get('episode') or request.verdict not in {'confirmed', 'rejected', 'uncertain'} or len(request.note) > 600:
            raise HTTPException(422, 'Invalid operator review')
        selected_event_id = request.eventId or job['result']['episode'].get('eventId')
        known_event_ids = {item.get('event_id') for item in job.get('events', [])}
        if request.eventId and request.eventId not in known_event_ids | {job['result']['episode'].get('eventId')}:
            raise HTTPException(422, 'Review event does not belong to this job')
        value = {'id': uuid.uuid4().hex, 'jobId': request.jobId, 'eventId': selected_event_id,
                 'verdict': request.verdict, 'note': request.note, 'reviewedAt': now()}
        with lock:
            stored = workspace_state()
            stored['reviews'].append(value)
            atomic_json(workspace, stored)
        return {'review': value}

    def reviewed_dataset():
        with lock:
            reviews = {}
            for review in workspace_state()['reviews']:
                if isinstance(review, dict) and isinstance(review.get('jobId'), str):
                    reviews[review['jobId']] = review
            records = []
            job_ids = set(reviews) | {path.parent.name for path in directory.glob('*/job.json')}
            for job_id in sorted(job_ids):
                review = reviews.get(job_id, {})
                if not IDENTITY.fullmatch(job_id) or review and review.get('verdict') not in {'confirmed', 'rejected', 'uncertain'}:
                    continue
                job_path = directory / job_id / 'job.json'
                manifest_path = job_path.parent / 'results/manifest.json'
                if not job_path.is_file() or not manifest_path.is_file():
                    continue
                job = json.loads(job_path.read_text())
                manifest = json.loads(manifest_path.read_text())
                if job.get('status') != 'completed' or job.get('synthetic') is True or manifest.get('synthetic') is True:
                    continue
                if manifest.get('status') != 'completed' or manifest.get('mode') != 'causal_fresh_video_inference' or manifest.get('replay_fallback') is not False:
                    continue
                hashes = [item.get('sha256') for item in job.get('inputs', [])]
                if not hashes:
                    continue
                try:
                    validate_manifest_camera_sources(manifest, job)
                except ValueError:
                    continue
                image_set = job.get('input_kind') == 'image_set'
                originals = job.get('source_sha256s', []) if image_set else hashes
                if (not isinstance(originals, list) or not originals
                        or any(not isinstance(value, str) or not re.fullmatch(r'[a-f0-9]{64}', value) for value in originals)):
                    continue
                if image_set and (originals != [item.get('sha256') for item in job.get('sourceImages', [])]
                                  or len(originals) != len(job.get('imageSourceFiles', []))):
                    continue
                result = job.get('result') or {}
                episodes = result.get('episodes') or []
                primary = result.get('episode')
                events = {item.get('event_id'): item for item in job.get('events', []) if isinstance(item, dict) and item.get('event_id')}

                def record_for(event_id, human_label, reviewed_at, note, operator_confirmed, event_review=False):
                    if not isinstance(event_id, str):
                        return None
                    event = events.get(event_id)
                    if event is None:
                        return None
                    camera_id = event.get('camera_id')
                    episode = next((item for item in episodes if item.get('eventId') == event_id
                                     and (camera_id is None or item.get('camera') == camera_id)), None)
                    if episode is None and isinstance(primary, dict) and primary.get('eventId') == event_id \
                            and (camera_id is None or primary.get('camera') == camera_id):
                        episode = primary
                    episode = episode or {}
                    evidence = []
                    for url in episode.get('frames', []):
                        prefix = '/media/' + job_id + '/'
                        name = url.removeprefix(prefix) if isinstance(url, str) and url.startswith(prefix) else ''
                        if name in job.get('media_files', []) and Path(name).name == name and (job_path.parent / 'results' / name).is_file():
                            evidence.append(url)
                    image_index = episode.get('image_index', event.get('image_index'))
                    if isinstance(image_index, int) and not isinstance(image_index, bool):
                        sources = job.get('sourceImages', [])
                        files = job.get('imageSourceFiles', [])
                        if 0 <= image_index < len(sources) and image_index < len(files):
                            name = files[image_index]
                            if Path(name).name == name and sources[image_index].get('url') == '/input/' + job_id + '/' + name:
                                evidence.append(sources[image_index]['url'])
                    analysis = job.get('analysis') or {}
                    analysis_matches = analysis.get('eventId') == event_id
                    vlm = event.get('vlm') or {}
                    response = vlm.get('response') or (analysis.get('rawResponse') if analysis_matches else None) or {}
                    camera_episode = episode.get('camera') or camera_id
                    evidence_times = episode.get('frameTimes', [])
                    return {'jobId': job_id, 'sampleId': job.get('sampleId'), 'task': job.get('task'),
                            'inputKind': job.get('input_kind', 'video'), 'syntheticTime': bool(job.get('synthetic_time')),
                            'sourceSHA256': originals, 'originalSourceSHA256': originals,
                            'executionVideoSHA256': hashes, 'eventId': event_id, 'cameraId': camera_episode,
                            'imageIndex': image_index, 'evidenceUrls': evidence,
                            'evidenceTimes': evidence_times, 'evidenceTimesSynthetic': image_set,
                            'modelUnsafe': response.get('unsafe'),
                            'modelAssessment': analysis.get('assessment') if analysis_matches else None,
                            'modelReason': response.get('reason'), 'modelEvidence': response.get('evidence', []),
                            'modelProvider': analysis.get('modelprovider') if analysis_matches else None,
                            'modelRequestMs': vlm.get('request_ms'), 'operatorConfirmed': operator_confirmed,
                            'humanLabel': human_label, 'reviewedAt': reviewed_at, 'note': note,
                            'reviewSource': 'event' if event_review else 'legacy_job',
                            'completedAt': job.get('completed_at')}

                event_review_ids = set()
                for event_id, event in events.items():
                    review_path = job_path.parent / 'reviews' / (hashlib.sha256(event_id.encode()).hexdigest() + '.json')
                    if not review_path.is_file():
                        continue
                    event_review = json.loads(review_path.read_text())
                    confirmed = event_review.get('confirmed')
                    if type(confirmed) is not bool:
                        continue
                    label = 'confirmed' if confirmed else 'rejected'
                    record = record_for(event_id, label, event_review.get('reviewed_at'), event_review.get('note', ''), confirmed, True)
                    if record is not None:
                        records.append(record)
                        event_review_ids.add(event_id)
                legacy_event_id = review.get('eventId') or (primary or {}).get('eventId')
                if review and legacy_event_id not in event_review_ids:
                    legacy = record_for(legacy_event_id, review['verdict'], review.get('reviewedAt'),
                                        review.get('note', ''),
                                        True if review['verdict'] == 'confirmed' else False if review['verdict'] == 'rejected' else None)
                    if legacy is not None:
                        records.append(legacy)
            return records

    @app.get('/api/dataset.jsonl')
    def dataset_jsonl():
        records = reviewed_dataset()
        body = ''.join(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n' for record in records)
        return Response(body, media_type='application/x-ndjson', headers={'Content-Disposition': 'attachment; filename="linq-reviewed.jsonl"'})

    @app.get('/api/dataset.csv')
    def dataset_csv():
        fields = ['jobId', 'sampleId', 'task', 'inputKind', 'syntheticTime', 'sourceSHA256',
                  'originalSourceSHA256', 'executionVideoSHA256', 'eventId', 'cameraId', 'imageIndex',
                  'evidenceUrls', 'evidenceTimes', 'evidenceTimesSynthetic', 'modelUnsafe', 'modelAssessment', 'modelReason', 'modelEvidence',
                  'modelProvider', 'modelRequestMs', 'operatorConfirmed', 'humanLabel', 'reviewSource', 'reviewedAt', 'note', 'completedAt']
        stream = io.StringIO(newline='')
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in reviewed_dataset():
            row = {}
            for key, value in record.items():
                if isinstance(value, (list, dict)) or type(value) is bool:
                    value = json.dumps(value, ensure_ascii=False, allow_nan=False)
                if isinstance(value, str) and value.startswith(('=', '+', '-', '@', '\t', '\r')):
                    value = "'" + value
                row[key] = value
            writer.writerow(row)
        return Response(stream.getvalue(), media_type='text/csv', headers={'Content-Disposition': 'attachment; filename="linq-reviewed.csv"'})

    @app.post('/api/zones')
    def facade_zone(request: ZoneRequest):
        if not any(sample['id'] == request.sampleId and sample.get('video') for sample in catalog()):
            raise HTTPException(422, 'Unknown video source')
        polygon = request.polygon
        if not 3 <= len(polygon) <= 12 or any(len(point) != 2 or any(not math.isfinite(value) or not 0 <= value <= 1 for value in point) for point in polygon):
            raise HTTPException(422, 'Invalid normalized polygon')
        area = abs(sum(point[0] * polygon[(index + 1) % len(polygon)][1] - polygon[(index + 1) % len(polygon)][0] * point[1] for index, point in enumerate(polygon))) / 2
        if area < 0.0001:
            raise HTTPException(422, 'Polygon has no usable area')
        with lock:
            stored = workspace_state()
            stored['zones'][request.sampleId] = polygon
            atomic_json(workspace, stored)
        return {'zoneId': request.sampleId, 'polygon': polygon, 'appliedToExistingResults': False,
                'inferenceEnabledForFutureJobs': True, 'purpose': 'Applied to future jobs for this sample only'}

    @app.get('/api/jobs')
    def list_jobs():
        return {'jobs': [public_job(json.loads(path.read_text())) for path in sorted(directory.glob('*/job.json'), reverse=True)]}

    @app.get('/api/jobs/{job_id}')
    def get_job(job_id: str):
        job = public_job(load_job(job_id))
        job['events'] = [event for event in all_events() if event['job_id'] == job_id]
        return job

    @app.get('/api/events')
    def events(camera: str | None = None, type: str | None = None, zone: str | None = None,
               start_s: float | None = None, end_s: float | None = None):
        for name, value in [('camera', camera), ('type', type), ('zone', zone)]:
            if value is not None and (not value.strip() or len(value) > 128):
                raise HTTPException(422, 'Invalid ' + name + ' filter')
        for name, value in [('start_s', start_s), ('end_s', end_s)]:
            if value is not None and (not math.isfinite(value) or value < 0 or value > 86400 * 7):
                raise HTTPException(422, name + ' must be a finite video-relative time from 0 to 604800 seconds')
        if start_s is not None and end_s is not None and end_s < start_s:
            raise HTTPException(422, 'end_s must be greater than or equal to start_s')
        values = all_events()
        return {'events': [event for event in values
                           if (camera is None or event.get('camera_id') == camera)
                           and (type is None or event.get('type', event_type(event)) == type)
                           and (zone is None or event.get('zone_id') == zone)
                           and (start_s is None or isinstance(event.get('trigger_timestamp_s'), (int, float)) and event['trigger_timestamp_s'] >= start_s)
                           and (end_s is None or isinstance(event.get('trigger_timestamp_s'), (int, float)) and event['trigger_timestamp_s'] <= end_s)]}

    @app.post('/api/events/{job_id}/{event_id}/review')
    def review_event(job_id: str, event_id: str, review: Review):
        job = load_job(job_id)
        if not any(item.get('event_id') == event_id for item in job.get('events', [])):
            raise HTTPException(404, 'Unknown event')
        if len(review.note) > 4000:
            raise HTTPException(400, 'Review note is too long')
        value = {'confirmed': review.confirmed, 'note': review.note, 'reviewed_at': now()}
        path = metadata_path(job_id).parent / 'reviews' / (hashlib.sha256(event_id.encode()).hexdigest() + '.json')
        with lock:
            atomic_json(path, value)
            with (path.parent / 'review_history.jsonl').open('a') as stream:
                stream.write(json.dumps({'job_id': job_id, 'event_id': event_id, **value}, ensure_ascii=False) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
        return {'job_id': job_id, 'event_id': event_id, 'review': value}

    @app.get('/media/{job_id}/{filename}')
    def media(job_id: str, filename: str):
        job = load_job(job_id)
        try:
            name = safe_filename(filename)
        except ValueError:
            raise HTTPException(404, 'Unknown media')
        if name not in job.get('media_files', []):
            raise HTTPException(404, 'Unknown media')
        return FileResponse(metadata_path(job_id).parent / 'results' / name)

    @app.get('/')
    def index():
        path = root / 'service/index.html'
        if not path.is_file():
            raise HTTPException(503, 'Demo interface is being prepared')
        return FileResponse(path)

    @app.get('/input/{job_id}/{filename}')
    def input_video(job_id: str, filename: str):
        job = load_job(job_id)
        if job.get('input_kind') == 'image_set':
            if filename not in job.get('imageSourceFiles', []):
                raise HTTPException(404, 'Unknown uploaded source image')
            return FileResponse(metadata_path(job_id).parent / 'input' / 'images' / safe_filename(filename))
        source = next((item for item in job['inputs'] if Path(item['path']).name == filename), None)
        if source is None:
            raise HTTPException(404, 'Unknown uploaded video')
        return FileResponse(source['path'])

    @app.get('/{filename:path}')
    def service_static(filename: str):
        path = (root / 'service' / filename).resolve()
        allowed = filename in {'index.html', 'workspace.js', 'icons.js', 'style.css', 'review.css',
                               'console-data.js', 'console-analytics.js', 'console-analytics.css',
                               'console-cameras.js', 'console-cameras.css', 'viewer-annotations.js'} or filename.startswith('assets/')
        if not allowed or not path.is_relative_to((root / 'service').resolve()) or not path.is_file():
            raise HTTPException(404, 'Unknown static asset')
        return FileResponse(path)

    return app


app = create_app()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', default=4183, type=int)
    args = parser.parse_args()
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()
