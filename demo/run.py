import hashlib
import json
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from demo.adapters import decode


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024*1024), b''):
            result.update(chunk)
    return result.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def load_font(size):
    for path in ['/System/Library/Fonts/Supplemental/Arial.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def render(camera, source, info, records, destination, options):
    output_width = min(options.get('width', 1280), info['width'])
    output_width -= output_width % 2
    scene_height = int(info['height'] * output_width / info['width']) // 2 * 2
    header_height = 66
    indicator_count = max((len(row.get('indicators', [])) for row in records), default=0)
    footer_height = 28 * (indicator_count + 1) + 8
    output_height = scene_height + header_height + footer_height
    process = subprocess.Popen(['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{output_width}x{output_height}', '-r', str(info['fps']), '-i', '-', '-an', '-c:v', 'libx264', '-preset', options.get('encoder_preset', 'fast'), '-threads', str(options.get('encoder_threads', 1)), '-crf', '22', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(destination)], stdin=subprocess.PIPE)
    font, small = load_font(19), load_font(15)
    current, cursor = None, 0
    factor = output_width / info['width']
    try:
        for index, image in decode(source, info):
            while cursor < len(records) and records[cursor]['frame_index'] <= index:
                current = records[cursor]
                cursor += 1
            timestamp = index / info['fps']
            scene = image.resize((output_width, scene_height))
            image = Image.new('RGB', (output_width, output_height), '#101c2b')
            image.paste(scene, (0, header_height))
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 0, output_width, 66), fill='#101c2b')
            mode = 'RECORDED INFERENCE REPLAY' if camera['detector']['mode'] == 'recorded_replay' else 'FRESH VIDEO INFERENCE'
            draw.text((16, 10), f"{camera['id']}  |  {mode}  |  {timestamp:.2f}s", font=font, fill='white')
            draw.text((16, 38), 'Ground calibration supplied' if camera.get('calibration') else 'Independent source - no common ground map', font=small, fill='#aec4df')
            for rule in camera.get('rules', []):
                if rule['kind'] == 'restricted_zone':
                    points = [(p[0]*output_width, header_height+p[1]*scene_height) for p in rule['polygon']] if rule.get('coordinates', 'normalized') == 'normalized' else [(p[0]*factor, header_height+p[1]*factor) for p in rule['polygon']]
                    draw.line(points + [points[0]], fill='#ffb142', width=2)
            if current:
                age = timestamp - current['local_timestamp_s']
                if age <= options.get('overlay_max_age_s', 0.6):
                    for track in current['tracks']:
                        x1, y1, x2, y2 = track['xyxy']
                        box = [max(0, min(output_width-1, float(x1)*factor)),
                               header_height+max(0, min(scene_height-1, float(y1)*factor)),
                               max(0, min(output_width-1, float(x2)*factor)),
                               header_height+max(0, min(scene_height-1, float(y2)*factor))]
                        box[2] = max(box[0]+1, box[2])
                        box[3] = max(box[1]+1, box[3])
                        color = '#43e0b0' if track['observed'] else '#97a1af'
                        draw.rectangle(box, outline=color, width=2)
                        label = f"{track['label']} #{track['track_id']} {track['score']:.2f}"
                        if track.get('global_id'):
                            label += f" G{track['global_id']}"
                        label_top = max(header_height, box[1]-23)
                        draw.rectangle((box[0], label_top, min(output_width, box[0]+len(label)*9), label_top+23), fill='#102a28')
                        draw.text((box[0]+3, label_top+1), label, font=small, fill=color)
                draw.rectangle((0, 0, output_width, 66), fill='#101c2b')
                draw.text((16, 10), f"{camera['id']}  |  {mode}  |  {timestamp:.2f}s", font=font, fill='white')
                draw.text((16, 38), 'Ground calibration supplied' if camera.get('calibration') else 'Independent source - no common ground map', font=small, fill='#aec4df')
                bottom = header_height + scene_height + 8
                for indicator in current['indicators']:
                    state = indicator['state'] if age <= options.get('overlay_max_age_s', 0.6) else 'stale'
                    color = '#ff7478' if state == 'active' else '#ffca6a' if state in {'pending', 'cooldown'} else '#a9bed6'
                    draw.text((16, bottom), f"{indicator['id']}: {state.upper()} | {indicator['semantics']}", font=small, fill=color)
                    bottom += 27
                draw.text((16, bottom), "Model boxes held between sampled frames", font=small, fill='#a9bed6')
            process.stdin.write(image.tobytes())
    finally:
        process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError('ffmpeg encoding failed')
    return {'width': output_width, 'height': output_height, 'fps': info['fps']}
