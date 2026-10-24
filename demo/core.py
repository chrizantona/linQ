import math
import numpy as np


def footpoint(box):
    return [(box[0] + box[2]) / 2, box[3]]


def project(point, homography):
    value = np.asarray(homography, dtype=float) @ [*point, 1.0]
    if not np.isfinite(value).all() or abs(value[2]) < 1e-9:
        raise ValueError('Point projects to infinity')
    return (value[:2] / value[2]).tolist()


def calibration(spec):
    if not spec:
        return None
    if spec.get('plane') != 'ground' or not spec.get('map_id') or spec.get('units') != 'metres':
        raise ValueError('Calibration requires ground plane, map_id, and metre units')
    if 'homography' in spec:
        matrix = np.asarray(spec['homography'], dtype=float)
    else:
        source = np.asarray(spec['image_points'], dtype=float)
        target = np.asarray(spec['world_points'], dtype=float)
        if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 2 or len(source) < 4:
            raise ValueError('At least four corresponding image/world points required')
        rows = []
        for (x, y), (u, v) in zip(source, target):
            rows.extend([[-x, -y, -1, 0, 0, 0, u*x, u*y, u], [0, 0, 0, -x, -y, -1, v*x, v*y, v]])
        if np.linalg.matrix_rank(np.asarray(rows)) < 8:
            raise ValueError('Degenerate calibration correspondences')
        _, _, vectors = np.linalg.svd(rows)
        matrix = vectors[-1].reshape(3, 3)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-14:
        raise ValueError('Invalid or singular homography')
    matrix = matrix / np.linalg.norm(matrix)
    if 'image_points' in spec:
        error = max(math.dist(project(point, matrix), target) for point, target in zip(spec['image_points'], spec['world_points']))
        if error > spec.get('max_reprojection_error_m', 0.2):
            raise ValueError('Calibration reprojection error exceeds tolerance')
    return matrix.tolist()


def inside(point, polygon):
    x, y = point
    result = False
    for index, (x1, y1) in enumerate(polygon):
        x2, y2 = polygon[index - 1]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            result = not result
    return result


def iou(a, b):
    low = np.maximum(a[:2], b[:2])
    high = np.minimum(a[2:], b[2:])
    intersection = np.prod(np.maximum(high - low, 0))
    return float(intersection / max(np.prod(np.subtract(a[2:], a[:2])) + np.prod(np.subtract(b[2:], b[:2])) - intersection, 1e-9))


def track_step(state, detections, timestamp, options):
    if 'timestamp' in state and timestamp <= state['timestamp']:
        raise ValueError('Tracker timestamps must strictly increase')
    tracks = [t for t in state.setdefault('tracks', []) if timestamp - t['last_seen'] <= options.get('max_age_s', 1.0)]
    dt = max(1e-3, timestamp - state.get('timestamp', timestamp - 1/3))
    state['timestamp'] = timestamp
    transition = np.eye(8)
    transition[:4, 4:] = np.eye(4) * dt
    observation = np.eye(4, 8)
    process = np.eye(8) * options.get('process_noise', 4.0) * dt
    noise = np.eye(4) * options.get('measurement_noise', 9.0)
    for track in tracks:
        track['mean'] = transition @ track['mean']
        track['covariance'] = transition @ track['covariance'] @ transition.T + process
    detections = [d for d in detections if d['score'] >= options.get('confidence', 0.25) and (not options.get('labels') or d['label'] in options['labels'])]
    pairs = sorted([(iou(track['mean'][:4], det['xyxy']), ti, di) for ti, track in enumerate(tracks) for di, det in enumerate(detections) if track['label'] == det['label']], reverse=True)
    used_tracks, used_detections = set(), set()
    for overlap, ti, di in pairs:
        if overlap < options.get('iou', 0.15) or ti in used_tracks or di in used_detections:
            continue
        track, detection = tracks[ti], detections[di]
        innovation = np.asarray(detection['xyxy']) - observation @ track['mean']
        covariance = observation @ track['covariance'] @ observation.T + noise
        gain = np.linalg.solve(covariance, observation @ track['covariance']).T
        track['mean'] += gain @ innovation
        residual = np.eye(8) - gain @ observation
        track['covariance'] = residual @ track['covariance'] @ residual.T + gain @ noise @ gain.T
        track.update(last_seen=timestamp, score=detection['score'], hits=track['hits']+1)
        used_tracks.add(ti)
        used_detections.add(di)
    for index, detection in enumerate(detections):
        if index in used_detections:
            continue
        state['next_id'] = state.get('next_id', 0) + 1
        tracks.append({'track_id': state['next_id'], 'label': detection['label'], 'score': detection['score'], 'mean': np.array([*detection['xyxy'], 0, 0, 0, 0], dtype=float), 'covariance': np.diag([10.0]*4 + [1000.0]*4), 'last_seen': timestamp, 'hits': 1})
    state['tracks'] = [t for t in tracks if timestamp - t['last_seen'] <= options.get('max_age_s', 1.0)]
    return [{'track_id': t['track_id'], 'xyxy': t['mean'][:4].tolist(), 'label': t['label'], 'score': t['score'], 'observed': t['last_seen'] == timestamp, 'age_since_observation_s': round(timestamp - t['last_seen'], 4), 'position_uncertainty_px': float(np.sqrt(np.trace(t['covariance'][:4, :4])))} for t in state['tracks'] if t['hits'] >= options.get('min_hits', 1)]


def incident_step(state, key, timestamp, evidence, rule):
    item = state.setdefault(key, {'state': 'clear', 'since': timestamp, 'started_at': None, 'last_timestamp': timestamp})
    previous = item['state']
    if timestamp - item['last_timestamp'] > rule.get('max_evidence_gap_s', 0.8):
        item.update(state='unknown', since=timestamp, started_at=None)
    item['last_timestamp'] = timestamp
    on, off = rule.get('on_threshold', 0.7), rule.get('off_threshold', 0.3)
    if evidence is None:
        item.update(state='unknown', since=timestamp, started_at=None)
    elif evidence >= on:
        if item['state'] in {'clear', 'unknown', 'cooldown'}:
            item.update(state='pending', since=timestamp)
        if item['state'] == 'pending' and timestamp - item['since'] >= rule.get('activate_after_s', 0.6):
            item.update(state='active', started_at=item['since'])
    elif evidence <= off:
        if item['state'] == 'active':
            item.update(state='cooldown', since=timestamp)
        elif item['state'] == 'cooldown' and timestamp - item['since'] >= rule.get('clear_after_s', 0.7):
            item.update(state='clear', since=timestamp, started_at=None)
        elif item['state'] in {'pending', 'unknown'}:
            item.update(state='clear', since=timestamp, started_at=None)
    return dict(item), previous != item['state']


def signals(tracks, rules, width, height):
    observed = [t for t in tracks if t['observed']]
    people = [t for t in observed if t['label'] == 'person']
    result = []
    for rule in rules:
        kind = rule['kind']
        score, subjects, explanation = None, [], ''
        if kind == 'restricted_zone':
            polygon = rule['polygon']
            points = [[p[0]*width, p[1]*height] for p in polygon] if rule.get('coordinates', 'normalized') == 'normalized' else polygon
            matches = [t for t in people if inside(footpoint(t['xyxy']), points)]
            score, subjects, explanation = float(bool(matches)), [t['track_id'] for t in matches], 'Footpoint inside user supplied image ROI'
        elif kind == 'wide_person_proxy':
            matches = [t for t in people if (t['xyxy'][2]-t['xyxy'][0]) / max(t['xyxy'][3]-t['xyxy'][1], 1) >= rule.get('aspect_ratio', 1.25)]
            score, subjects, explanation = float(bool(matches)), [t['track_id'] for t in matches], 'Wide person box proxy; does not establish a fall'
        elif kind == 'under_load_proxy':
            loads = [t for t in observed if t['label'] in rule.get('load_labels', ['suspended_load'])]
            matches = [p for p in people if any(l['xyxy'][0] <= footpoint(p['xyxy'])[0] <= l['xyxy'][2] and l['xyxy'][3] < footpoint(p['xyxy'])[1] for l in loads)]
            score, subjects, explanation = float(bool(matches)), [t['track_id'] for t in matches], '2D vertical overlap candidate; depth, rigging and fall zone unverified'
        elif kind == 'ground_proximity':
            vehicles = [t for t in observed if t['label'] in rule.get('vehicle_labels', ['car', 'truck', 'forklift', 'bus']) and t.get('world_xy')]
            pairs = [(p, v) for p in people if p.get('world_xy') for v in vehicles if math.dist(p['world_xy'], v['world_xy']) <= rule.get('distance_m', 2.0)]
            score = float(bool(pairs)) if any(p.get('world_xy') for p in people) else None
            subjects, explanation = sorted({t['track_id'] for pair in pairs for t in pair}), 'Ground-plane footpoint distance; requires calibrated ground contact'
        elif kind == 'model_signal':
            explanation = 'Requires detector response signals with configured rule id'
        else:
            raise ValueError('Unknown incident rule: ' + kind)
        result.append({'id': rule['id'], 'kind': kind, 'score': score, 'track_ids': subjects, 'semantics': rule.get('semantics', 'candidate'), 'explanation': explanation})
    return result


def associate_world(state, camera_id, timestamp, tracks, calibration_spec, options):
    matrix = calibration(calibration_spec)
    if matrix is None:
        return []
    allowed = calibration_spec.get('ground_contact_labels', ['person', 'car', 'truck', 'bus', 'forklift'])
    tolerance = options.get('sync_tolerance_s', 0.12)
    radius = options.get('association_distance_m', 1.0)
    output = []
    for track in tracks:
        if not track['observed'] or track['label'] not in allowed:
            continue
        point = project(footpoint(track['xyxy']), matrix)
        if calibration_spec.get('valid_image_polygon') and not inside(footpoint(track['xyxy']), calibration_spec['valid_image_polygon']):
            continue
        track['world_xy'] = point
        existing = state.setdefault('identities', [])
        key = f"{camera_id}:{track['track_id']}"
        match = next((entry for entry in existing if key in entry['members'] and timestamp - entry['members'][key] <= options.get('handoff_max_gap_s', 1.0)), None)
        if match is None:
            candidates = [entry for entry in existing if entry['map_id'] == calibration_spec['map_id'] and entry['label'] == track['label'] and abs(timestamp-entry['timestamp_s']) <= tolerance and math.dist(point, entry['world_xy']) <= radius and not any(member.startswith(camera_id+':') and abs(timestamp - seen) <= tolerance for member, seen in entry['members'].items())]
            match = min(candidates, key=lambda entry: math.dist(point, entry['world_xy']), default=None)
        if match is None:
            match = {'global_id': len(existing)+1, 'map_id': calibration_spec['map_id'], 'label': track['label'], 'members': {}}
            existing.append(match)
        match.update(timestamp_s=timestamp, world_xy=point)
        match['members'][key] = timestamp
        track['global_id'] = match['global_id']
        output.append({'camera_id': camera_id, 'track_id': track['track_id'], 'global_id': match['global_id'], 'map_id': match['map_id'], 'world_xy': point, 'timestamp_s': timestamp})
    return output
