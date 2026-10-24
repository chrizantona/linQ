import argparse
from collections import Counter
import csv
import html
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demo.adapters import video_info
from demo.run import digest, render, write_json


def read_rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def vlm_events(run_directory):
    files = sorted((run_directory / 'events').glob('*/event.json'))
    if files:
        values = [json.loads(path.read_text()) for path in files]
    else:
        index = {}
        for row in read_rows(run_directory / 'events.jsonl'):
            if row.get('event_id'):
                index.setdefault(row['event_id'], {}).update(row)
        values = list(index.values())
    return sorted(values, key=lambda row: (row.get('trigger_timestamp_s', 0), row.get('camera_id', ''), row.get('event_id', '')))


def result_rows(events):
    result = []
    for event in events:
        vlm = event.get('vlm', {})
        response = vlm.get('response', {})
        result.append({'event_id': event.get('event_id'), 'camera_id': event.get('camera_id'), 'trigger_timestamp_s': event.get('trigger_timestamp_s'), 'route_scope': event.get('route_scope'), 'request_reasons': event.get('request_reasons', []), 'status': vlm.get('status', 'unavailable'), 'http_response_received': vlm.get('http_response_received', False), 'model_execution_verified': vlm.get('model_execution_verified', False), 'execution_device': vlm.get('execution_device', 'unverified'), 'unsafe': response.get('unsafe'), 'reason': response.get('reason'), 'evidence': response.get('evidence', []), 'error': vlm.get('error'), 'raw_frame_timestamps_s': [frame.get('timestamp_s') for frame in event.get('raw_frames', [])]})
    return result


def table_html(rows):
    columns = [('camera_id', 'Камера'), ('trigger_timestamp_s', 'Триггер, с'), ('route_scope', 'Scope'), ('request_reasons', 'Причина запроса'), ('status', 'Статус'), ('unsafe', 'Unsafe'), ('reason', 'Ответ VLM'), ('evidence', 'Видимые признаки'), ('error', 'Ошибка')]
    header = ''.join('<th>'+label+'</th>' for key, label in columns)
    body = []
    for row in rows:
        cells = []
        for key, label in columns:
            value = row.get(key)
            if value is None:
                value = 'unknown' if key == 'unsafe' else '—'
            elif isinstance(value, list):
                value = '; '.join(str(item) for item in value) or '—'
            cells.append('<td>'+html.escape(str(value))+'</td>')
        body.append('<tr>'+''.join(cells)+'</tr>')
    return '<table><thead><tr>'+header+'</tr></thead><tbody>'+''.join(body)+'</tbody></table>' if body else '<p>Запросов VLM в этом прогоне нет.</p>'


def create(run_directory, config_path, output, width=1280):
    run_directory = Path(run_directory).resolve()
    config_path = Path(config_path).resolve()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    settings = json.loads(config_path.read_text())
    root = (config_path.parent / settings.get('root', '..')).resolve()
    manifest = json.loads((run_directory / 'manifest.json').read_text())
    if manifest.get('mode') != 'causal_fresh_video_inference':
        raise ValueError('Renderer requires an actual causal fresh-inference pipeline manifest')
    all_frames = read_rows(run_directory / 'frames.jsonl')
    accepted = [row for row in all_frames if row.get('accepted', True)]
    sources = {camera['id']: camera for camera in manifest.get('cameras', [])}
    videos = []
    for original in settings['cameras']:
        camera_id = original['id']
        rows = sorted([row for row in accepted if row['camera_id'] == camera_id], key=lambda row: row['frame_index'])
        source = (root / original['video']).resolve()
        source_hash = digest(source)
        expected_hash = sources.get(camera_id, {}).get('source_sha256')
        if expected_hash and source_hash != expected_hash:
            raise ValueError('Raw source video SHA256 differs from actual pipeline manifest: '+camera_id)
        info = video_info(source)
        normalized = []
        for row in rows:
            local = row.get('local_timestamp_s', row['timestamp_s']-original.get('time_offset_s', 0))
            if abs(local-row['frame_index']/info['fps']) > 1/info['fps']+1e-3:
                raise ValueError('Accepted inference timestamp does not align with source frame')
            if row['width'] != info['width'] or row['height'] != info['height']:
                raise ValueError('Actual inference coordinates differ from source video dimensions')
            normalized.append({**row, 'local_timestamp_s': local, 'detections': row.get('stages', {}).get('detector', {}).get('detections', []), 'tracks': row.get('tracks', []), 'indicators': row.get('indicators', [])})
        camera = {**original, 'detector': {'mode': 'fresh'}}
        destination = output / (camera_id+'.mp4')
        render(camera, source, info, normalized, destination, {'width': width, 'overlay_max_age_s': max(0.6, 1.5/float(original.get('V', original.get('sample_fps', 3)))), 'encoder_preset': 'ultrafast', 'encoder_threads': 1})
        poster = output / (camera_id+'.jpg')
        subprocess.run(['ffmpeg', '-y', '-v', 'error', '-ss', str(min(info['duration_s']/2, max(0, info['duration_s']-1/info['fps']))), '-i', str(destination), '-frames:v', '1', str(poster)], check=True)
        detector_status = dict(Counter(row.get('stages', {}).get('detector', {}).get('status', 'unknown') for row in rows))
        classifier_status = dict(Counter(row.get('stages', {}).get('classifier', {}).get('status', 'unknown') for row in rows))
        videos.append({'camera_id': camera_id, 'video': destination.name, 'poster': poster.name, 'source_sha256': source_hash, 'accepted_inference_frames': len(rows), 'detector_status_counts': detector_status, 'classifier_status_counts': classifier_status, 'runtime_provenance': sources.get(camera_id, {})})
    events = vlm_events(run_directory)
    vlm = result_rows(events)
    write_json(output / 'vlm_results.json', {'display_scope': 'Offline results after complete clip processing; never overlaid onto earlier video frames', 'results': vlm, 'source_events': events})
    fields = ['event_id', 'camera_id', 'trigger_timestamp_s', 'route_scope', 'request_reasons', 'status', 'unsafe', 'reason', 'evidence', 'error', 'http_response_received', 'model_execution_verified', 'execution_device']
    with (output / 'vlm_results.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for row in vlm:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, list) else value for key, value in row.items() if key in fields})
    summary = {'pipeline_status': manifest.get('status'), 'render_source': 'Actual accepted fresh-inference records only', 'future_vlm_overlay': False, 'decoded_frames': len(all_frames), 'accepted_frames': len(accepted), 'quality_or_cadence_rejections': len(all_frames)-len(accepted), 'videos': videos, 'vlm_result_count': len(vlm), 'vlm_status_counts': dict(Counter(row['status'] for row in vlm)), 'provenance_manifest': manifest}
    write_json(output / 'summary.json', summary)
    sections = []
    for video in videos:
        sections.append('<section><h2>'+html.escape(video['camera_id'])+'</h2><video controls preload="metadata" poster="'+html.escape(video['poster'])+'" src="'+html.escape(video['video'])+'"></video><p>Принятых кадров inference: '+str(video['accepted_inference_frames'])+'. <a href="'+html.escape(video['video'])+'">Скачать видео</a></p></section>')
    page = '<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Причинный pipeline · LinQ</title><style>body{font:18px system-ui,sans-serif;background:#f5f5f7;color:#1d1d1f;margin:40px auto;padding:0 24px;max-width:1380px}h1{font-size:42px}h2{font-size:28px}p{line-height:1.5}section{margin:38px 0}video{width:100%;max-height:720px;background:#101c2b;border-radius:14px}table{border-collapse:collapse;width:100%;font-size:16px;background:white}th,td{padding:14px;text-align:left;vertical-align:top;border-bottom:1px solid #dce1e8;overflow-wrap:anywhere}th{background:#eaf3ff}a{color:#007aff}.scroll{overflow:auto}</style><h1>Причинный pipeline на LinQ</h1><p>Видео содержит фактические боксы, track ID и состояния incident rules из свежего прогона. Между принятыми кадрами overlay хранится ограниченное время. Ошибки и unknown не заменяются replay.</p>'+''.join(sections)+'<section><h2>Результаты VLM после обработки клипа</h2><p>Это отдельный итоговый журнал. Ответ VLM не показан на более ранних кадрах. Scope <code>open</code> — периодический обзор сцены; остальные scopes относятся к настроенным событиям. Remote execution device считается непроверенным, если это указано в receipt.</p><p><a href="vlm_results.json">Полный JSON с provenance</a> · <a href="vlm_results.csv">CSV</a> · <a href="summary.json">Manifest и stage status</a></p><div class="scroll">'+table_html(vlm)+'</div></section></html>'
    (output / 'index.html').write_text(page)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--width', type=int, default=1280)
    args = parser.parse_args()
    result = create(args.run, args.config, args.output, args.width)
    print(json.dumps({'output': str(Path(args.output).resolve()), 'videos': len(result['videos']), 'accepted_frames': result['accepted_frames'], 'vlm_results': result['vlm_result_count']}))


if __name__ == '__main__':
    main()
