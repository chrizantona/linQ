import argparse
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait as wait_futures
from contextlib import ExitStack
import hashlib
import heapq
import importlib
import json
import math
from pathlib import Path
import shutil
import threading
import time

import numpy as np

from demo.adapters import decode, validate_detection, video_info
from demo.core import associate_world, calibration, incident_step, signals, track_step
from deploy.common import digest, head_probabilities, preprocess, write_json
from deploy.runtime import encode_pixels, open_encoder
from deploy.vlm_http import call_vlm


def callback(spec):
    module, name = spec['callable'].split(':')
    loaded = importlib.import_module(module)
    return getattr(loaded, name), {'callable': spec['callable'], 'source_sha256': digest(loaded.__file__) if loaded.__file__ else None}


def execution_error(error):
    return {'status': 'error', 'actual_execution': False, 'attempted': True, 'error_type': type(error).__name__, 'error': str(error)}


def verified_tpu_receipt(provenance):
    if not isinstance(provenance, dict) or provenance.get('TPU_execution_verified') is not True:
        return False
    receipt = provenance.get('receipt')
    if not isinstance(receipt, dict) or receipt.get('status') != 'received_and_validated':
        return False
    tiled = provenance.get('tiled_receipts', [])
    if not isinstance(tiled, list):
        return False
    return all(isinstance(item, dict) and isinstance(item.get('receipt'), dict)
               and item['receipt'].get('status') == 'received_and_validated' for item in tiled)


def detector_tpu_summary(state):
    detector_frames = int(state.get('detector_TPU_frames', 0) or 0)
    verified_frames = int(state.get('verified_detector_TPU_frames', 0) or 0)
    return {
        'detector_TPU_execution_verified': bool(detector_frames and verified_frames == detector_frames),
        'detector_TPU_frames': detector_frames,
        'verified_detector_TPU_frames': verified_frames,
    }


def classifier_step(state, frame):
    if state.get('runtime') is None:
        return {'status': 'disabled', 'actual_execution': False, 'encoder_executed': False,
                'head_executed': False, 'backend': None, 'reason': 'detector_profile_has_no_clip_classifier'}
    runtime = state['runtime']
    pixels = preprocess([frame['path']], runtime['manifest']['preprocessing'])
    feature = encode_pixels(runtime, pixels)[0]
    state['features'].append({'frame_index': frame['frame_index'], 'timestamp_s': frame['timestamp_s'], 'feature': feature})
    while state['features'] and frame['timestamp_s'] - state['features'][0]['timestamp_s'] > state['buffer_window_s']:
        state['features'].popleft()
    head = state['head']
    required = head['feature_frames']
    if len(state['features']) < required:
        return {'status': 'warming_up', 'actual_execution': True, 'encoder_executed': True, 'head_executed': False, 'required_frames': required, 'available_frames': len(state['features']), 'backend': runtime['backend'], 'device': runtime['device'], 'TPU_execution_verified': False, 'provenance': state.get('classifier_provenance', {})}
    selected = list(state['features'])[-required:]
    if [item['frame_index'] for item in selected] != [item['frame_index'] for item in list(state['frames'])[-required:]]:
        return {'status': 'unknown', 'actual_execution': True, 'encoder_executed': True, 'head_executed': False, 'reason': 'missing feature for latest accepted frame window'}
    values = np.stack([item['feature'] for item in selected])
    probabilities = head_probabilities(values, head)
    return {'status': 'ok', 'actual_execution': True, 'encoder_executed': True, 'head_executed': True, 'task': state['config']['classifier']['task'], 'prediction': head['classes'][int(probabilities.argmax())], 'classes': head['classes'], 'probabilities': probabilities.tolist(), 'backend': runtime['backend'], 'device': runtime['device'], 'TPU_execution_verified': False, 'representation': 'ordered pooled frame features with existing trained linear clip head; no spatial Concat', 'sampling': 'causal rolling latest frames; differs from fixed-fraction whole-clip training sampling', 'input_frame_indices': [item['frame_index'] for item in selected], 'input_timestamps_s': [item['timestamp_s'] for item in selected], 'bbox': None, 'provenance': state.get('classifier_provenance', {})}


def retain_frame(state, image, context):
    path = state['buffer_directory'] / f"{context['frame_index']:09d}.jpg"
    image.save(path, format='JPEG', quality=95)
    frame = {**context, 'path': str(path), 'sha256': digest(path)}
    state['frames'].append(frame)
    while len(state['frames']) > state['max_frames'] or state['frames'] and context['timestamp_s'] - state['frames'][0]['timestamp_s'] > state['buffer_window_s']:
        expired = state['frames'].popleft()
        Path(expired['path']).unlink(missing_ok=True)
    return frame


def snapshot(state, event_id, timestamp, maximum):
    selected = [item for item in state['frames'] if item['timestamp_s'] <= timestamp]
    if not selected:
        raise ValueError('No causal raw frames for VLM trigger')
    if len(selected) > maximum:
        indices = np.linspace(0, len(selected)-1, maximum, dtype=int)
        selected = [selected[index] for index in indices]
    directory = state['output'] / 'events' / event_id / 'frames'
    directory.mkdir(parents=True, exist_ok=True)
    frames = []
    for item in selected:
        destination = directory / Path(item['path']).name
        shutil.copy2(item['path'], destination)
        frames.append({**item, 'path': str(destination), 'relative_path': str(destination.relative_to(state['output']))})
    return frames


def dispatch_create(config, output):
    workers = int(config.get('max_inflight', 1))
    queue = int(config.get('max_queue', 2))
    if workers < 1 or queue < 0 or workers+queue < len(config.get('camera_ids', [])):
        raise ValueError('VLM capacity must be positive and cover one outstanding request per configured camera')
    return {'executor': ThreadPoolExecutor(max_workers=workers, thread_name_prefix='incident-vlm'), 'pending': {}, 'queues': {}, 'order': deque(), 'workers': workers, 'capacity': workers+queue, 'config': config, 'output': output}


def dispatch_pump(dispatcher):
    while len(dispatcher['pending']) < dispatcher['workers'] and dispatcher['order']:
        busy = {item['camera_id'] for item in dispatcher['pending'].values()}
        camera = next((name for name in dispatcher['order'] if name not in busy), None)
        if camera is None:
            break
        dispatcher['order'].remove(camera)
        event, frames = dispatcher['queues'].pop(camera)
        config = dispatcher['config']
        context = {'event_id': event['event_id'], 'camera_id': camera, 'rule_id': event['rule_id'], 'trigger_timestamp_s': event['trigger_timestamp_s'], 'request_reasons': event.get('request_reasons', ['active_event']), 'route_scope': event.get('route_scope', config['scope']), 'frames': [{'frame_index': item['frame_index'], 'timestamp_s': item['timestamp_s'], 'sha256': item['sha256']} for item in frames]}
        future = dispatcher['executor'].submit(call_vlm, config['config'], [item['path'] for item in frames], event.get('route_scope', config['scope']), event.get('route_prompt', config['prompt']), context)
        dispatcher['pending'][future] = {'event_id': event['event_id'], 'camera_id': camera}


def dispatch_submit(dispatcher, event, frames, emit=None):
    camera = event['camera_id']
    coalesce = dispatcher['config'].get('coalesce_latest', False)
    queued = dispatcher['queues'].get(camera)
    busy = any(item['camera_id'] == camera for item in dispatcher['pending'].values())
    if (queued or busy) and not coalesce:
        return {'status': 'camera_busy', 'actual_execution': False, 'fallback': None}
    if queued:
        previous = queued[0]
        previous['vlm'] = {'status': 'superseded', 'actual_execution': False, 'superseded_by_event_id': event['event_id'], 'fallback': None}
        write_json(dispatcher['output'] / 'events' / previous['event_id'] / 'event.json', previous)
        if emit:
            emit('events', {'type': 'vlm_superseded', 'event_id': previous['event_id'], 'camera_id': camera, 'vlm': previous['vlm']})
        dispatcher['queues'][camera] = (event, frames)
    else:
        if len(dispatcher['pending'])+len(dispatcher['queues']) >= dispatcher['capacity']:
            return {'status': 'queue_full', 'actual_execution': False, 'fallback': None}
        dispatcher['queues'][camera] = (event, frames)
        dispatcher['order'].append(camera)
    dispatch_pump(dispatcher)
    return {'status': 'queued', 'actual_execution': False, 'ordered_frame_count': len(frames), 'remote_execution_verified': False, 'scheduling': 'one pending and one latest queued per camera; waiting order preserved' if coalesce else 'one outstanding request per camera, round-robin waiting cameras'}


def dispatch_drain(dispatcher, events, emit, wait=False, progress=None):
    while True:
        if wait and dispatcher['pending']:
            wait_futures(list(dispatcher['pending']), return_when=FIRST_COMPLETED)
        completed = [future for future in dispatcher['pending'] if future.done()]
        for future in completed:
            event_id = dispatcher['pending'].pop(future)['event_id']
            event = events[event_id]
            try:
                receipt = future.result()
                response = receipt['response']
                status = 'unknown' if response['unsafe'] is None else 'confirmed' if response['unsafe'] else 'rejected'
                result = {'status': status, 'actual_execution': True, 'http_response_received': True, 'model_execution_verified': False, **receipt}
            except Exception as error:
                result = execution_error(error)
                result['fallback'] = None
            event['vlm'] = result
            write_json(dispatcher['output'] / 'events' / event_id / 'event.json', event)
            emit('events', {'type': 'vlm_result', 'event_id': event_id, 'camera_id': event['camera_id'], 'rule_id': event['rule_id'], 'trigger_timestamp_s': event['trigger_timestamp_s'], 'request_reasons': event.get('request_reasons', []), 'vlm': result})
        dispatch_pump(dispatcher)
        if progress:
            progress()
        if not wait or not dispatcher['pending'] and not dispatcher['queues']:
            break
    if wait:
        dispatcher['executor'].shutdown(wait=True)


def quality_gate(image, config):
    if config.get('enabled', True) is False:
        return {'accepted': True, 'enabled': False, 'reasons': [], 'threshold_source': 'explicit configuration, no TEST tuning'}
    pixels = np.asarray(image.convert('L'), dtype=np.float64)
    centre = pixels[1:-1, 1:-1]
    laplacian = pixels[:-2, 1:-1]+pixels[2:, 1:-1]+pixels[1:-1, :-2]+pixels[1:-1, 2:]-4*centre
    variance = float(laplacian.var()) if laplacian.size else 0.0
    mean = float(pixels.mean())
    clipped = float(((pixels <= config.get('dark_pixel', 3)) | (pixels >= config.get('bright_pixel', 252))).mean())
    limits = {'min_laplacian_variance': config.get('min_laplacian_variance', 1.0), 'min_mean': config.get('min_mean', 5), 'max_mean': config.get('max_mean', 250), 'max_clipped_fraction': config.get('max_clipped_fraction', 0.98)}
    reasons = []
    if variance < limits['min_laplacian_variance']:
        reasons.append('blur_or_no_texture')
    if mean < limits['min_mean'] or mean > limits['max_mean']:
        reasons.append('exposure_mean')
    if clipped > limits['max_clipped_fraction']:
        reasons.append('exposure_clipping')
    return {'accepted': not reasons, 'enabled': True, 'reasons': reasons, 'statistics': {'laplacian_variance': variance, 'mean_luma': mean, 'clipped_fraction': clipped}, 'thresholds': limits, 'threshold_source': 'explicit configuration/default engineering limits, no TEST tuning'}


def uncertainty_result(classification, spec, scope):
    if classification.get('status') != 'ok':
        return {'status': 'unknown', 'uncertain': None, 'eligible_scope': scope in spec.get('scopes', []), 'reasons': ['classifier_not_available']}
    values = sorted(classification['probabilities'], reverse=True)
    maximum = float(values[0])
    margin = maximum-float(values[1]) if len(values) > 1 else maximum
    reasons = []
    if maximum < spec.get('min_max_probability', 0.75):
        reasons.append('low_max_probability')
    if margin < spec.get('min_margin', 0.15):
        reasons.append('low_top_two_margin')
    return {'status': 'ok', 'uncertain': bool(reasons), 'eligible_scope': scope in spec.get('scopes', []), 'scope': scope, 'reasons': reasons, 'max_probability': maximum, 'top_two_margin': margin, 'thresholds': {'min_max_probability': spec.get('min_max_probability', 0.75), 'min_margin': spec.get('min_margin', 0.15)}, 'routing_source': 'explicit configured scope allowlist; no automatic TEST-based choice'}


def periodic_due(state, timestamp, dispatcher):
    if not dispatcher:
        return False
    interval = float(dispatcher['config'].get('periodic_seconds', 30))
    if interval <= 0:
        raise ValueError('periodic_seconds must be positive')
    if 'next_periodic_s' not in state:
        if len(state['frames']) < state['max_frames']:
            return False
        state['next_periodic_s'] = timestamp+interval
        return True
    if timestamp+1e-9 < state['next_periodic_s']:
        return False
    while state['next_periodic_s'] <= timestamp+1e-9:
        state['next_periodic_s'] += interval
    return True


def inspection(state, timestamp, reasons, dispatcher, events, emit):
    identity = f"{state['source_sha256']}|{state['config']['id']}|inspection|{timestamp:.9f}|{','.join(reasons)}"
    event_id = hashlib.sha256(identity.encode()).hexdigest()[:24]
    periodic_scope = dispatcher['config'].get('periodic_scope', 'open')
    event = {'event_id': event_id, 'kind': 'inspection', 'camera_id': state['config']['id'], 'rule_id': 'inspection', 'trigger_timestamp_s': timestamp, 'request_reasons': reasons, 'route_scope': periodic_scope if reasons in (['periodic'], ['source_end']) else dispatcher['config']['scope'], 'route_prompt': dispatcher['config'].get('periodic_prompt', 'Inspect the visible scene for any hazards, unsafe interactions, or unexpected events. Describe only visible evidence and return unknown when context is insufficient.') if reasons in (['periodic'], ['source_end']) else dispatcher['config']['prompt'], 'source_sha256': state['source_sha256'], 'state': 'inspection'}
    events[event_id] = event
    try:
        frames = snapshot(state, event_id, timestamp, dispatcher['config'].get('max_frames', 3))
        event['raw_frames'] = [{key: value for key, value in item.items() if key != 'path'} for item in frames]
        event['vlm'] = dispatch_submit(dispatcher, event, frames, emit)
        if event['vlm']['status'] == 'queued':
            state['last_vlm_trigger_s'] = timestamp
    except Exception as error:
        event['vlm'] = {'status': 'unavailable', **execution_error(error)}
        event['vlm']['status'] = 'unavailable'
    write_json(state['output'] / 'events' / event_id / 'event.json', event)
    emit('events', {'type': 'inspection_created', **event})
    return event


def source_end_inspection(state, dispatcher, events, emit):
    if not dispatcher or not dispatcher['config'].get('scan_on_source_end', False) or len(state['frames']) < state['max_frames']:
        return None
    timestamp = state['frames'][-1]['timestamp_s']
    if timestamp <= state.get('last_vlm_trigger_s', float('-inf'))+1e-9:
        return None
    return inspection(state, timestamp, ['source_end'], dispatcher, events, emit)


def incident_score(rule, indicator, detector, classifier, custom):
    if rule['kind'] != 'model_signal':
        return indicator['score'] if detector['status'] == 'ok' else None
    source = rule.get('source', 'detector')
    if source == 'classifier':
        if classifier['status'] != 'ok' or rule['label'] not in classifier['classes']:
            return None
        return classifier['probabilities'][classifier['classes'].index(rule['label'])]
    selected = detector if source == 'detector' else custom if source == 'incident_model' else None
    if selected is None:
        raise ValueError('Unknown signal source: ' + source)
    signal_name = rule.get('signal', rule['id'])
    value = selected.get('signals', {}).get(signal_name) if selected.get('status') == 'ok' else None
    if value is not None and (not np.isfinite(value) or not 0 <= float(value) <= 1):
        raise ValueError('Incident model signal must be a finite probability or null')
    return value


def process_sample(state, image, frame_index, timestamp, settings, dispatcher, events, emit):
    context = {'camera_id': state['config']['id'], 'frame_index': frame_index, 'timestamp_s': timestamp, 'width': image.width, 'height': image.height, 'local_timestamp_s': timestamp-state['config'].get('time_offset_s', 0)}
    periodic = periodic_due(state, timestamp, dispatcher)
    cadence = 1/float(state['config'].get('V', state['config'].get('sample_fps', 3)))
    if timestamp-state.get('last_accepted_s', float('-inf'))+1e-9 < cadence:
        quality = {'accepted': False, 'evaluated': False, 'reasons': ['sampling_cadence', 'quality_not_evaluated'],
                   'evaluation_policy': 'Cadence checked before image quality; no image statistics computed for this frame'}
    else:
        quality = quality_gate(image, state['config'].get('quality', settings.get('quality', {})))
    if not quality['accepted']:
        if periodic:
            inspection(state, timestamp, ['periodic'], dispatcher, events, emit)
        unknown = {'status': 'skipped', 'actual_execution': False, 'reason': 'frame_not_accepted'}
        record = {**context, 'accepted': False, 'quality': quality, 'uncertainty': {'status': 'unknown', 'uncertain': None}, 'stages': {'detector': unknown, 'classifier': unknown}, 'tracks': [], 'indicators': [], 'buffers': {'raw_frames': len(state['frames']), 'pooled_features': len(state['features']), 'spatial_features': len(state['spatial_features'])}}
        emit('frames', record)
        return record
    state['last_accepted_s'] = timestamp
    frame = retain_frame(state, image, context)
    if not periodic:
        periodic = periodic_due(state, timestamp, dispatcher)
    detector, classification, custom = {}, {}, {'status': 'unconfigured', 'actual_execution': False}
    try:
        started = time.perf_counter()
        result = state['detector'](image, context, state['config']['detector'].get('options', {}))
        if result.get('actual_execution') is not True or result.get('mode') in {'recorded_replay', 'replay'}:
            raise ValueError('Detector must attest actual fresh execution; replay is not supported')
        detections = [validate_detection(item, image.width, image.height) for item in result['detections']]
        detector_provenance = result.get('provenance', {})
        tpu_verified = verified_tpu_receipt(detector_provenance)
        state['detector_TPU_frames'] = int(state.get('detector_TPU_frames', 0) or 0) + 1
        state['verified_detector_TPU_frames'] = int(state.get('verified_detector_TPU_frames', 0) or 0) + int(tpu_verified)
        detector = {'status': 'ok', 'actual_execution': True, 'mode': 'fresh_inference', 'detections': detections, 'signals': result.get('signals', {}), 'provenance': detector_provenance, 'TPU_execution_verified': tpu_verified, 'elapsed_ms': (time.perf_counter()-started)*1000}
        if result.get('feature_map') is not None:
            spatial = np.asarray(result['feature_map'])
            if spatial.ndim != 3 or not np.isfinite(spatial).all():
                raise ValueError('Model-supplied feature_map must be finite [C,H,W]')
            state['spatial_features'].append({'timestamp_s': timestamp, 'frame_index': frame_index, 'feature_map': spatial.copy()})
            detector['feature_map_shape'] = list(spatial.shape)
            detector['spatial_feature_source'] = 'actual detector callback output'
    except Exception as error:
        detector = execution_error(error)
    try:
        classification = state['classify'](state, frame)
    except Exception as error:
        classification = execution_error(error)
    uncertainty_spec = state['config'].get('uncertainty', settings.get('uncertainty', {}))
    scope = dispatcher['config']['scope'] if dispatcher else state['config'].get('scope', '')
    uncertainty = uncertainty_result(classification, uncertainty_spec, scope)
    if periodic:
        inspection(state, timestamp, ['periodic'], dispatcher, events, emit)
    reasons = []
    if dispatcher and uncertainty['uncertain'] and uncertainty['eligible_scope'] and timestamp-state.get('last_uncertainty_s', float('-inf')) >= uncertainty_spec.get('cooldown_seconds', 5):
        reasons.append('classifier_uncertainty')
        state['last_uncertainty_s'] = timestamp
    request_consumed = False
    tracks = track_step(state['tracker'], detector.get('detections', []), timestamp, settings.get('tracker', {}))
    positions = associate_world(state['world'], state['config']['id'], timestamp, tracks, state['config'].get('calibration'), settings.get('multicamera', {}))
    while state['spatial_features'] and timestamp - state['spatial_features'][0]['timestamp_s'] > state['buffer_window_s']:
        state['spatial_features'].popleft()
    if state.get('incident_model'):
        try:
            value = state['incident_model'](list(state['spatial_features']), tracks, classification, context, state['config']['incident_model'].get('options', {}))
            if value.get('actual_execution') is not True:
                raise ValueError('Incident model must attest actual execution')
            custom = {'status': 'ok', **value}
        except Exception as error:
            custom = execution_error(error)
    indicators = signals(tracks, state['config']['rules'], image.width, image.height)
    for rule, indicator in zip(state['config']['rules'], indicators):
        try:
            evidence = incident_score(rule, indicator, detector, classification, custom)
            indicator['score'] = evidence
            status, changed = incident_step(state['incident_states'], rule['id'], timestamp, evidence, rule)
            indicator.update(status)
        except Exception as error:
            indicator.update(score=None, state='unknown', error=execution_error(error))
            changed = False
        existing = state['episodes'].get(rule['id'])
        if indicator['state'] == 'active' and existing is None:
            identity = f"{state['source_sha256']}|{state['config']['id']}|{rule['id']}|{indicator['started_at']:.9f}"
            event_id = hashlib.sha256(identity.encode()).hexdigest()[:24]
            rule_vlm_enabled = rule.get('vlm_enabled', True) is not False
            active_reasons = ((['active_event'] if dispatcher and dispatcher['config'].get('confirm_active_incidents', False) else []) + reasons) if rule_vlm_enabled else []
            event = {'event_id': event_id, 'camera_id': state['config']['id'], 'rule_id': rule['id'], 'frame_index': frame_index, 'trigger_timestamp_s': timestamp, 'event_trigger': 'active_event', 'request_reasons': active_reasons, 'route_scope': dispatcher['config']['scope'] if dispatcher and rule_vlm_enabled else None, 'route_prompt': dispatcher['config']['prompt'] if dispatcher and rule_vlm_enabled else None, 'started_at': indicator['started_at'], 'state': 'active', 'semantics': rule.get('semantics', 'candidate requiring VLM validation'), 'score': indicator['score'], 'track_ids': indicator['track_ids'], 'source_sha256': state['source_sha256'], 'classifier_at_trigger': classification, 'detector_at_trigger': {key: value for key, value in detector.items() if key != 'detections'}, 'vlm': {'status': 'unconfigured', 'actual_execution': False}}
            request_consumed = bool(active_reasons)
            state['episodes'][rule['id']] = event_id
            events[event_id] = event
            if dispatcher and active_reasons:
                try:
                    frames = snapshot(state, event_id, timestamp, dispatcher['config'].get('max_frames', 3))
                    event['raw_frames'] = [{key: value for key, value in item.items() if key != 'path'} for item in frames]
                    event['vlm'] = dispatch_submit(dispatcher, event, frames, emit)
                    if event['vlm']['status'] == 'queued':
                        state['last_vlm_trigger_s'] = timestamp
                except Exception as error:
                    event['vlm'] = execution_error(error)
            elif not rule_vlm_enabled:
                event['vlm'] = {'status': 'not_requested_for_geometric_rule', 'actual_execution': False,
                                'reason': 'Geometric zone candidates are presented for operator review without action-model assessment.'}
            elif dispatcher:
                event['vlm'] = {'status': 'not_requested', 'actual_execution': False, 'reason': 'confident active incident does not request VLM under configured policy'}
            write_json(state['output'] / 'events' / event_id / 'event.json', event)
            emit('events', {'type': 'event_created', **event})
        elif existing and changed:
            event = events[existing]
            event.update(state=indicator['state'], last_transition_timestamp_s=timestamp)
            write_json(state['output'] / 'events' / existing / 'event.json', event)
            emit('events', {'type': 'event_transition', 'event_id': existing, 'camera_id': state['config']['id'], 'rule_id': rule['id'], 'timestamp_s': timestamp, 'state': indicator['state']})
            if indicator['state'] == 'clear':
                state['episodes'].pop(rule['id'])
        if state['episodes'].get(rule['id']):
            indicator['event_id'] = state['episodes'][rule['id']]
    if reasons and not request_consumed:
        inspection(state, timestamp, reasons, dispatcher, events, emit)
    record = {**context, 'accepted': True, 'quality': quality, 'uncertainty': uncertainty, 'raw_frame_sha256': frame['sha256'], 'source_sha256': state['source_sha256'], 'stages': {'detector': detector, 'classifier': classification, 'tracker': {'status': 'ok', 'actual_execution': True, 'algorithm': 'Bayesian constant velocity Kalman / class-gated IoU'}, 'incident_model': custom, 'incident_rules': {'status': 'ok', 'actual_execution': True}}, 'tracks': tracks, 'world_positions': positions, 'indicators': indicators, 'buffers': {'raw_frames': len(state['frames']), 'pooled_features': len(state['features']), 'spatial_features': len(state['spatial_features'])}}
    emit('frames', record)
    return record


def camera_open(config, settings, root, output, stack):
    if config['detector'].get('mode', 'fresh') != 'fresh':
        raise ValueError('Causal pipeline accepts fresh detector callbacks only')
    source = (root / config['video']).resolve()
    classifier = config.get('classifier')
    if classifier is None:
        if config.get('task_profile') not in {'phone_objects', 'cigarette_objects', 'phone_use', 'smoking'}:
            raise ValueError('Classifier can be disabled only for a registered detector task profile')
        runtime, head, classifier_provenance = None, None, None
        classify = classifier_step
        required = int(config.get('T', settings.get('buffer', {}).get('T', 3)))
    else:
        runtime = stack.enter_context(open_encoder(root / classifier['bundle'], classifier['backend'], threads=classifier.get('threads', 4)))
        task = runtime['manifest']['heads'][classifier['task']]
        head = json.loads((runtime['bundle'] / task['path']).read_text())
        classifier_provenance = None
        classify = classifier_step
        required = int(head['feature_frames'])
    buffer = settings.get('buffer', {})
    if float(buffer.get('window_s', 4)) <= 0:
        raise ValueError('Buffer window_s must be positive')
    requested = int(config.get('T', buffer.get('T', required)))
    if classifier is not None and requested != required:
        raise ValueError('T must equal the existing trained task head frame count: '+str(required))
    if requested < 1:
        raise ValueError('T must be positive')
    maximum = requested
    features = requested
    if classifier is not None and (maximum < head['feature_frames'] or features < head['feature_frames']):
        raise ValueError('Buffers cannot be smaller than the trained clip head input count')
    if classifier is not None:
        classifier_provenance = {'bundle_manifest_sha256': digest(runtime['bundle'] / 'manifest.json'), 'head_sha256': digest(runtime['bundle'] / task['path'])}
    detector, detector_provenance = callback(config['detector'])
    incident, incident_provenance = callback(config['incident_model']) if config.get('incident_model') else (None, None)
    directory = output / '.buffer' / config['id']
    directory.mkdir(parents=True)
    calibration(config.get('calibration'))
    return {'config': config, 'output': output, 'source': source, 'source_sha256': digest(source), 'info': video_info(source), 'runtime': runtime, 'head': head, 'classifier_provenance': classifier_provenance, 'classify': classify, 'detector': detector, 'detector_provenance': detector_provenance, 'incident_model': incident, 'incident_model_provenance': incident_provenance, 'frames': deque(), 'features': deque(maxlen=features), 'spatial_features': deque(maxlen=features), 'max_frames': maximum, 'buffer_window_s': float(buffer.get('window_s', 4)), 'buffer_directory': directory, 'tracker': {}, 'world': {}, 'incident_states': {}, 'episodes': {}, 'detector_TPU_frames': 0, 'verified_detector_TPU_frames': 0}


def sampled(state):
    rate = float(state['config'].get('V', state['config'].get('sample_fps', 3)))
    if rate <= 0:
        raise ValueError('V must be positive')
    for index, image in decode(state['source'], state['info']):
        local = index / state['info']['fps']
        yield index, local + state['config'].get('time_offset_s', 0), image


def run(config_path, output):
    config_path = Path(config_path).resolve()
    settings = json.loads(config_path.read_text())
    root = (config_path.parent / settings.get('root', '..')).resolve()
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Causal pipeline output must be empty')
    output.mkdir(parents=True, exist_ok=True)
    config_hash = digest(config_path)
    events, dispatcher, frame_count = {}, None, 0
    accepted_count, quality_rejected_count = 0, 0
    states, cv_ready, cv_completed_at = [], False, None
    run_started = time.monotonic()
    first_candidate_published_at, first_candidate_elapsed_ms = None, None
    cameras = settings['cameras']
    if not cameras or len({camera['id'] for camera in cameras}) != len(cameras) or any(not camera['id'].replace('_', '').replace('-', '').isalnum() for camera in cameras):
        raise ValueError('Distinct safe camera ids required')
    write_json(output / 'config.json', settings)
    def progress(status='running', error=None):
        nonlocal first_candidate_published_at, first_candidate_elapsed_ms
        pending = len(dispatcher['pending']) if dispatcher else 0
        queued = len(dispatcher['queues']) if dispatcher else 0
        candidates = []
        for event in events.values():
            if event.get('event_trigger') != 'active_event' or event.get('kind') == 'inspection':
                continue
            score = event.get('score')
            if not isinstance(score, (int, float)) or not math.isfinite(score):
                score = None
            candidates.append({
                'event_id': event.get('event_id'), 'camera_id': event.get('camera_id'),
                'rule_id': event.get('rule_id'), 'event_trigger': event.get('event_trigger'),
                'frame_index': event.get('frame_index'), 'trigger_timestamp_s': event.get('trigger_timestamp_s'),
                'source_sha256': event.get('source_sha256'), 'state': event.get('state'), 'score': score,
                'track_ids': list(event.get('track_ids', []))[:64],
                'vlm_status': (event.get('vlm') or {}).get('status', 'unconfigured'),
            })
        candidates.sort(key=lambda item: (item.get('trigger_timestamp_s', 0), item.get('camera_id', ''), item.get('event_id', '')))
        if candidates and first_candidate_published_at is None:
            first_candidate_published_at = time.time()
            first_candidate_elapsed_ms = round((time.monotonic() - run_started) * 1000, 3)
        value = {'status': status, 'mode': 'causal_fresh_video_inference', 'replay_fallback': False,
                 'config_sha256': config_hash, 'updated_at': time.time(), 'cv_ready': cv_ready,
                 'cv_completed_at': cv_completed_at, 'vlm_pending': pending + queued > 0,
                 'vlm_pending_count': pending, 'vlm_queued_count': queued,
                 'frame_count': frame_count, 'accepted_frame_count': accepted_count,
                 'quality_rejected_frame_count': quality_rejected_count,
                 'candidate_event_count': len(candidates), 'candidate_events': candidates[-20:],
                 '_candidate_publication_verification': {
                     'firstCandidatePublishedAt': first_candidate_published_at,
                     'elapsedMs': first_candidate_elapsed_ms,
                 },
                 'quality_rejection_count_policy': 'Quality failures among cadence-eligible frames only; cadence skips are not quality-evaluated',
                 'cameras': [{'id': state['config']['id'], 'source_sha256': state['source_sha256']} for state in states]}
        if error is not None:
            value['error'] = execution_error(error)
        temporary = output / 'progress.json.tmp'
        write_json(temporary, value)
        temporary.replace(output / 'progress.json')
    progress()
    with ExitStack() as stack:
        streams = {name: stack.enter_context((output / (name+'.jsonl')).open('w')) for name in ['frames', 'events']}
        lock = threading.Lock()
        def emit(name, value):
            with lock:
                streams[name].write(json.dumps(value, ensure_ascii=False, allow_nan=False)+'\n')
                streams[name].flush()
        try:
            states = [camera_open(camera, settings, root, output, stack) for camera in cameras]
            shared_world = {}
            for state in states:
                state['world'] = shared_world
            if settings.get('vlm'):
                vlm = dict(settings['vlm'])
                vlm['config'] = str((root / vlm['config']).resolve())
                vlm['camera_ids'] = [camera['id'] for camera in cameras]
                if int(vlm.get('max_frames', 3)) < 1:
                    raise ValueError('VLM max_frames must be positive')
                dispatcher = dispatch_create(vlm, output)
            scheduled = []
            for order, state in enumerate(states):
                iterator = sampled(state)
                stack.callback(iterator.close)
                first = next(iterator, None)
                if first:
                    heapq.heappush(scheduled, (first[1], order, first, iterator))
            while scheduled:
                timestamp, order, item, iterator = heapq.heappop(scheduled)
                if dispatcher:
                    dispatch_drain(dispatcher, events, emit)
                record = process_sample(states[order], item[2], item[0], timestamp, settings, dispatcher, events, emit)
                frame_count += 1
                accepted_count += int(record['accepted'])
                quality_rejected_count += int(not record['accepted'] and 'sampling_cadence' not in record['quality']['reasons'])
                if record['accepted']:
                    progress()
                following = next(iterator, None)
                if following:
                    heapq.heappush(scheduled, (following[1], order, following, iterator))
                else:
                    source_end_inspection(states[order], dispatcher, events, emit)
            cv_ready, cv_completed_at = True, time.time()
            progress()
            if dispatcher:
                dispatch_drain(dispatcher, events, emit, wait=True, progress=progress)
            manifest = {'status': 'completed', 'mode': 'causal_fresh_video_inference', 'config_sha256': config_hash,
                        'input_provenance': settings.get('input_provenance', {'kind': 'video', 'synthetic_time': False, 'original_source_sha256s': []}),
                        'frame_count': frame_count, 'accepted_frame_count': accepted_count, 'quality_rejected_frame_count': quality_rejected_count, 'cadence_skipped_frame_count': frame_count-accepted_count-quality_rejected_count, 'event_count': sum(event.get('kind') != 'inspection' for event in events.values()), 'inspection_count': sum(event.get('kind') == 'inspection' for event in events.values()), 'replay_fallback': False, 'event_notifications': 'durable local event records; no external messaging', 'cameras': [{'id': state['config']['id'], 'source_sha256': state['source_sha256'], 'detector': state['detector_provenance'], 'classifier_backend': state['runtime']['backend'] if state['runtime'] else None, 'classifier_provenance': state['classifier_provenance'], 'incident_model': state['incident_model_provenance'], **detector_tpu_summary(state), 'classifier_TPU_execution_verified': False} for state in states]}
            manifest['quality_rejection_count_policy'] = 'Quality failures among cadence-eligible frames only; cadence skips are not quality-evaluated'
            write_json(output / 'manifest.json', manifest)
            progress(status='completed')
            return manifest
        except Exception as error:
            if dispatcher:
                dispatch_drain(dispatcher, events, emit, wait=True)
            write_json(output / 'manifest.json', {'status': 'error', 'config_sha256': config_hash, 'frame_count': frame_count, 'replay_fallback': False, **execution_error(error)})
            progress(status='error', error=error)
            raise
        finally:
            shutil.rmtree(output / '.buffer', ignore_errors=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output)))


if __name__ == '__main__':
    main()
