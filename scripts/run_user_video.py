import argparse
import copy
from datetime import datetime
import json
import math
from pathlib import Path
import re
import sys
import subprocess

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.linq_prototype import health, launch


TITLES = {
    'synthsite': 'Работник под подвешенным грузом: кандидат события',
    'safelift': 'Подъём / опускание груза: направление движения',
    'hmdb51_v2': 'Бег / ходьба: действие человека',
    'physicalai_v2': 'Погрузчик и человек: кандидат near miss',
    'omnifall_v2': 'Падение / упавший человек: состояние, не диагноз',
    'harness_author': 'Наличие страховочной привязи: компонент, не соблюдение правил',
}

DETECTOR_PROFILES = {
    'phone_objects': {
        'title_ru': 'Телефоны',
        'frames': 3,
        'label': 'phone',
        'rule_id': 'phone_present',
        'task_spec': {
            'task_id': 'phone_objects', 'scope': '03', 'label_semantics': 'component',
            'labels': ['phone'],
            'instruction': 'Report only whether a phone is visibly present with a visible person in the supplied frames. Describe the visible phone-person context only. Do not infer phone use, unsafe behavior, or a safety violation. Return unknown if no person is visible or the phone-person relation is unclear.',
        },
        'semantics': 'Visible phone object presence candidate only; this is not a phone-use or safety verdict.',
    },
    'cigarette_objects': {
        'title_ru': 'Сигареты',
        'frames': 3,
        'label': 'cigarette',
        'rule_id': 'cigarette_present',
        'task_spec': {
            'task_id': 'cigarette_objects', 'scope': '04', 'label_semantics': 'component',
            'labels': ['cigarette'],
            'instruction': 'Report only whether a cigarette is visibly present with a visible person in the supplied frames. Describe the visible cigarette-person context only. Do not infer smoking behavior, an unsafe act, or a safety violation. Return unknown if no person is visible or the cigarette-person relation is unclear.',
        },
        'semantics': 'Visible cigarette object presence candidate only; this is not a smoking-behavior or safety verdict.',
    },
    'phone_use': {
        'title_ru': 'Использование телефона',
        'frames': 3,
        'label': 'phone',
        'rule_id': 'phone_use_review',
        'action_review': True,
        'task_spec': {
            'task_id': 'phone_use', 'scope': '03', 'label_semantics': 'action',
            'labels': ['phone'],
            'instruction': 'Review only the visible person-phone action in these ordered frames. A phone detector only establishes that a phone may be present; inspect whether a visible person is visibly holding and actively looking at or interacting with that phone. Distinguish merely holding or carrying a phone from visible active use. Return unknown if no person is visible, the hand-phone relation is unclear, or action cannot be observed. Do not infer a safety violation or unseen behavior; describe visible evidence only.',
        },
        'semantics': 'Visible phone-use action candidate for VLM review; detector presence alone is not evidence of use or a safety violation.',
    },
    'smoking': {
        'title_ru': 'Курение',
        'frames': 3,
        'label': 'cigarette',
        'rule_id': 'smoking_review',
        'action_review': True,
        'task_spec': {
            'task_id': 'smoking', 'scope': '04', 'label_semantics': 'action',
            'labels': ['cigarette'],
            'instruction': 'Review only the visible person-cigarette action in these ordered frames. A cigarette detector only establishes that a cigarette may be present; inspect whether a visible person visibly holds it at or brings it to their mouth in a smoking action. Distinguish carrying or holding a cigarette away from the mouth from visible smoking action. Return unknown if no person is visible, the hand-cigarette relation is unclear, or action cannot be observed. Do not infer a safety violation or unseen behavior; describe visible evidence only.',
        },
        'semantics': 'Visible smoking-action candidate for VLM review; cigarette presence alone is not evidence of smoking or a safety violation.',
    },
}


def available_tasks(root, config):
    bundle = Path(config['cameras'][0]['classifier']['bundle'])
    manifest = json.loads((bundle if bundle.is_absolute() else root / bundle).joinpath('manifest.json').read_text())
    tasks = dict(manifest['heads'])
    for task_id, profile in DETECTOR_PROFILES.items():
        tasks[task_id] = {
            'title_ru': profile['title_ru'],
            'frames': profile['frames'], 'classes': None,
            'task_spec': dict(profile['task_spec']),
            'detector_profile': task_id,
        }
    return tasks



def validate_cadence(frequency):
    if not isinstance(frequency, (int, float)) or isinstance(frequency, bool) or not math.isfinite(frequency) or frequency <= 0:
        raise ValueError('V должна быть конечным положительным числом')
    return float(frequency)


def apply_cadence(config, frequency):
    frequency = validate_cadence(frequency)
    person_thresholds = []
    for camera in config['cameras']:
        for rule in camera.get('rules', []):
            gap = float(rule.get('max_evidence_gap_s', 0.8))
            if not math.isfinite(gap) or gap <= 0:
                raise ValueError('max_evidence_gap_s должна быть конечной и положительной')
            rule['max_evidence_gap_s'] = max(gap, 1.5 / frequency)
        options = camera['detector'].get('options', {})
        threshold = float(options.get('linq_person', {}).get('confidence', options.get('confidence', 0.25)))
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError('Person detector confidence должна быть от 0 до 1')
        person_thresholds.append(threshold)
    tracker = config.setdefault('tracker', {})
    age, confidence = float(tracker.get('max_age_s', 1.0)), float(tracker.get('confidence', 0.25))
    if not math.isfinite(age) or age <= 0 or not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError('Некорректные max_age_s или confidence tracker')
    tracker['max_age_s'] = max(age, 2.5 / frequency)
    tracker['confidence'] = min([confidence, *person_thresholds])


def _detector_confidence(camera, label):
    options = camera['detector'].get('options', {})
    models = options.get('models', [])
    matching = [model for model in models if label in model.get('labels', [])]
    if len(matching) != 1:
        raise ValueError(f'Exactly one configured detector model must provide {label}')
    threshold = float(matching[0].get('confidence', options.get('confidence', 0.25)))
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError(f'{label} detector confidence must be from 0 to 1')
    return threshold


def _validate_zone(zone):
    if zone is None:
        return None
    if not isinstance(zone, dict) or set(zone) != {'zone_id', 'camera_id', 'polygon'}:
        raise ValueError('Zone must contain only zone_id, camera_id, and polygon')
    if not isinstance(zone['zone_id'], str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', zone['zone_id']):
        raise ValueError('Invalid zone id')
    if not isinstance(zone['camera_id'], str) or not re.fullmatch(r'user_camera_\d{2}', zone['camera_id']):
        raise ValueError('Invalid zone camera id')
    polygon = zone['polygon']
    if not isinstance(polygon, list) or not 3 <= len(polygon) <= 12:
        raise ValueError('Zone polygon must contain 3 to 12 points')
    for point in polygon:
        if not isinstance(point, list) or len(point) != 2 or any(type(value) not in {int, float} or not math.isfinite(value) or not 0 <= value <= 1 for value in point):
            raise ValueError('Zone polygon coordinates must be finite normalized pairs')
    area = abs(sum(polygon[index][0] * polygon[(index + 1) % len(polygon)][1] - polygon[(index + 1) % len(polygon)][0] * polygon[index][1] for index in range(len(polygon)))) / 2
    if area < 0.0001:
        raise ValueError('Zone polygon has no usable area')
    return {'zone_id': zone['zone_id'], 'camera_id': zone['camera_id'], 'polygon': copy.deepcopy(polygon)}


def build_config(root, videos, task, frequency, cv_only, zone=None, input_kind='video', source_sha256s=None):
    root = Path(root).resolve()
    if not 1 <= len(videos) <= 2:
        raise ValueError('Загрузите один или два видео')
    frequency = validate_cadence(frequency)
    config = json.loads((root / ('config_single.json' if len(videos) == 1 else 'config_two.json')).read_text())
    heads = available_tasks(root, config)
    if task not in heads:
        raise ValueError('Неизвестная задача: ' + task)
    head = heads[task]
    count = int(head['frames'])
    spec = head.get('task_spec', {})
    detector_profile = head.get('detector_profile')
    zone = _validate_zone(zone)
    if input_kind not in {'video', 'image_set'}:
        raise ValueError('Input kind must be video or image_set')
    source_sha256s = list(source_sha256s or [])
    if len(source_sha256s) > 24 or any(not isinstance(value, str) or not re.fullmatch(r'[a-f0-9]{64}', value) for value in source_sha256s):
        raise ValueError('Invalid original source SHA256 list')
    if input_kind == 'image_set' and (task not in {'phone_objects', 'cigarette_objects'} or not 2 <= len(source_sha256s) <= 24):
        raise ValueError('Still-image sets support only component-presence tasks with original source hashes')
    if input_kind == 'image_set' and frequency != 1:
        raise ValueError('Still-image sets use fixed synthetic one-second timestamps')
    if input_kind == 'video' and source_sha256s:
        raise ValueError('Original still-image hashes are valid only for image_set input')
    video_root = (root / 'videos').resolve()
    paths = []
    for video in videos:
        candidate = Path(video)
        if not candidate.is_absolute():
            candidate = root / candidate if candidate.parts[0] == 'videos' else video_root / candidate
        candidate = candidate.resolve()
        if not candidate.is_relative_to(video_root) or not candidate.is_file():
            raise ValueError('Видео должно быть существующим файлом внутри ROOT/videos: ' + str(video))
        paths.append(candidate.relative_to(root).as_posix())
    cameras = []
    for index, path in enumerate(paths):
        camera = copy.deepcopy(config['cameras'][min(index, len(config['cameras']) - 1)])
        camera.update(id=f'user_camera_{index + 1:02d}', video=path, time_offset_s=0, T=count, V=frequency, scope=spec.get('scope', ''))
        camera.pop('calibration', None)
        camera.pop('uncertainty', None)
        if detector_profile:
            profile = DETECTOR_PROFILES[detector_profile]
            label = profile['label']
            threshold = _detector_confidence(camera, label)
            camera['task_profile'] = detector_profile
            camera['classifier'] = None
            camera['detector']['options']['models'] = [model for model in camera['detector']['options'].get('models', []) if label in model.get('labels', [])]
            camera['rules'] = [{
                'id': profile['rule_id'], 'kind': 'model_signal', 'source': 'detector',
                'label': label, 'signal': profile.get('signal', label + '_present'),
                'on_threshold': threshold, 'off_threshold': threshold * 0.6,
                'activate_after_s': 0.6, 'clear_after_s': 0.6, 'max_evidence_gap_s': 0.8,
                'semantics': profile['semantics'],
            }]
            if profile.get('action_review'):
                camera['rules'][0]['action_review'] = True
        else:
            camera['classifier']['task'] = task
            unsafe_label = spec.get('unsafe_label')
            if unsafe_label is None:
                camera['rules'] = []
            else:
                template = next((rule for rule in camera.get('rules', []) if rule.get('kind') == 'model_signal' and rule.get('source') == 'classifier'), {})
                camera['rules'] = [{**template, 'id': task + '_clip_candidate', 'kind': 'model_signal', 'source': 'classifier', 'label': unsafe_label, 'semantics': 'trained clip-head candidate; not a confirmed safety verdict; temporal sampling differs from training'}]
        if zone is not None and zone['camera_id'] == camera['id']:
            camera['rules'].append({
                'id': 'zone_' + zone['zone_id'], 'kind': 'restricted_zone',
                'polygon': zone['polygon'], 'coordinates': 'normalized',
                'vlm_enabled': False,
                'semantics': 'Person footpoint inside the configured image-space polygon; geometric candidate only, not a safety violation.',
            })
        cameras.append(camera)
    config.update(root='.', cameras=cameras)
    config['input_provenance'] = {'kind': input_kind, 'synthetic_time': input_kind == 'image_set',
                                  'original_source_sha256s': source_sha256s}
    apply_cadence(config, frequency)
    config.pop('multicamera', None)
    config.setdefault('buffer', {}).update(T=count, window_s=max(float(config.get('buffer', {}).get('window_s', 0)), count / frequency + 1))
    if cv_only:
        config.pop('vlm', None)
    elif 'vlm' not in config:
        raise ValueError('В базовом config отсутствует VLM')
    else:
        config['vlm'].update(scope=spec.get('scope', ''), prompt=spec.get('instruction', ''), max_frames=count, coalesce_latest=True, scan_on_source_end=True)
        if detector_profile:
            config['vlm']['periodic_scope'] = spec.get('scope', '')
            config['vlm']['periodic_prompt'] = config['vlm']['prompt']
            config['vlm']['confirm_active_incidents'] = True
        allowed = config.get('uncertainty', {}).get('scopes', [])
        config.setdefault('uncertainty', {})['scopes'] = [spec['scope']] if spec.get('scope') in allowed else []
        if input_kind == 'image_set':
            static_only = ('The inputs are still images submitted in order and assigned synthetic one-second timestamps, not a recorded video. '
                           'Do not infer motion, a fall, an action, duration, or behavior from this order. Assess only objects and people visibly present in each still; '
                           'return unknown when a static frame cannot establish the requested observation.')
            config['vlm']['prompt'] = (config['vlm'].get('prompt', '') + ' ' + static_only).strip()
            config['vlm']['periodic_prompt'] = config['vlm']['prompt']
            config['vlm']['periodic_scope'] = spec.get('scope', '')
    return config, {'task': task, 'title_ru': DETECTOR_PROFILES.get(detector_profile, {}).get('title_ru', TITLES.get(task, task)), 'T': count, 'V': frequency, 'classes': head['classes'], 'scope': spec.get('scope'), 'cv_only': cv_only, 'backend': 'detector' if detector_profile else None, 'zone': zone, 'input_kind': input_kind, 'synthetic_time': input_kind == 'image_set', 'source_sha256s': source_sha256s}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', nargs=2, metavar=('CONFIG', 'OUTPUT'), help=argparse.SUPPRESS)
    parser.add_argument('--videos', nargs='+')
    parser.add_argument('--task', default='synthsite')
    parser.add_argument('--name')
    parser.add_argument('--cv-only', action='store_true')
    parser.add_argument('--V', type=float, default=1)
    parser.add_argument('--list-tasks', action='store_true')
    parser.add_argument('--zone-json', help=argparse.SUPPRESS)
    parser.add_argument('--input-kind', choices=['video', 'image_set'], default='video', help=argparse.SUPPRESS)
    parser.add_argument('--source-sha256s-json', help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.execute:
        config_path, output = args.execute
        subprocess.run([sys.executable, '-m', 'deploy.pipeline', '--config', config_path, '--output', output], cwd=root, check=True)
        subprocess.run([sys.executable, str(root / 'scripts/render_causal_demo.py'), '--run', output, '--config', config_path, '--output', str(Path(output) / 'view')], cwd=root, check=True)
        return
    if args.list_tasks:
        config = json.loads((root / 'config_single.json').read_text())
        result = {'tasks': [{'id': task, 'title_ru': head.get('title_ru', TITLES.get(task, task)), 'frames': head['frames'], 'classes': head['classes'], 'scope': head.get('task_spec', {}).get('scope'), 'semantics': head.get('task_spec', {}).get('label_semantics'), 'backend': 'detector' if head.get('detector_profile') else 'trained_clip_head'} for task, head in available_tasks(root, config).items()]}
        print(json.dumps(result, ensure_ascii=False))
        return
    if not args.videos:
        parser.error('--videos обязателен для запуска')
    name = args.name or 'user_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    if not re.fullmatch(r'user_[A-Za-z0-9_-]+', name):
        raise ValueError('Имя запуска должно начинаться с user_ и содержать буквы, цифры, _ или -')
    output = root / 'runs' / name
    config_path = root / ('config_' + name + '.json')
    if output.exists() or config_path.exists():
        raise FileExistsError('Выберите новое имя: существующий запуск не перезаписывается')
    zone = json.loads(args.zone_json) if args.zone_json else None
    source_sha256s = json.loads(args.source_sha256s_json) if args.source_sha256s_json else None
    config, details = build_config(root, args.videos, args.task, args.V, args.cv_only, zone=zone,
                                   input_kind=args.input_kind, source_sha256s=source_sha256s)
    if not args.cv_only:
        status = health(root)
        if not status.get('ready') or status.get('mode') != 'local':
            raise RuntimeError('Локальный CPU VLM не готов; выберите --cv-only явно')
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n')
    command = [str(root / '.venv/bin/python'), str(root / 'scripts/run_user_video.py'), '--execute', str(config_path), str(output)]
    receipt = launch(root, 'run_' + name, command)
    print(json.dumps({**receipt, **details, 'status': 'starting', 'output': str(output), 'config': str(config_path)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
