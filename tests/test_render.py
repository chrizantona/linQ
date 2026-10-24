"""Rendered evidence must retain the whole camera image, including its edges."""
import shutil
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from demo.adapters import decode, video_info
from demo.run import render


@pytest.mark.skipif(not shutil.which('ffmpeg') or not shutil.which('ffprobe'), reason='ffmpeg is required')
def test_evidence_bands_do_not_cover_small_objects_at_image_edges(tmp_path):
    source_image = tmp_path/'source.png'
    Image.new('RGB', (320, 240), (220, 40, 20)).save(source_image)
    source, destination = tmp_path/'input.mp4', tmp_path/'evidence.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-loop', '1', '-i', str(source_image),
                    '-t', '1', '-r', '1', '-pix_fmt', 'yuv420p', str(source)], check=True)
    info = video_info(source)
    camera = {'id': 'camera_01', 'detector': {'mode': 'fresh'}, 'rules': []}
    rows = [{'frame_index': 0, 'local_timestamp_s': 0, 'tracks': [
        {'xyxy': [10, 210, 40, 238], 'label': 'phone', 'track_id': 1, 'score': .9, 'observed': True}],
        'indicators': [{'id': 'presence', 'state': 'pending', 'semantics': 'Object candidate'}]}]
    result = render(camera, source, info, rows, destination, {})
    assert result['height'] > info['height']
    output_info = video_info(destination)
    frames = list(decode(destination, output_info))
    assert frames
    output = frames[0][1]
    for point in [(300, 68), (300, 66+238)]:
        red, green, blue = output.getpixel(point)
        assert red > 150 and green < 90 and blue < 90, (point, (red, green, blue))
    # The bottom-edge object remains inside the scene, above the separate footer.
    assert output_info['height'] >= 66 + info['height'] + 36
