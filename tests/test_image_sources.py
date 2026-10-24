import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from PIL import Image

from deploy.image_sources import prepare_image_set


class ImageSetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.paths = []
        for index, color in enumerate(('red', 'green', 'blue')):
            path = self.root / f'frame {index + 1}.jpg'
            Image.new('RGB', (96, 64), color).save(path, quality=90)
            self.paths.append(path)

    def tearDown(self):
        self.temp.cleanup()

    def test_encodes_ordered_stills_and_records_source_and_video_hashes(self):
        if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
            self.skipTest('ffmpeg and ffprobe are required for media integration test')
        target = self.root / 'prepared'
        result = prepare_image_set(self.paths, target, camera_id='camera-front')

        receipt = json.loads(Path(result['receipt_path']).read_text())
        self.assertEqual(receipt['input_kind'], 'image_set')
        self.assertIs(receipt['synthetic_time'], True)
        self.assertEqual(receipt['camera_id'], 'camera-front')
        self.assertEqual(receipt['frame_rate_fps'], 1)
        self.assertEqual(receipt['frame_count'], 3)
        self.assertEqual(receipt['duration_seconds'], 3)
        self.assertEqual(receipt['original_source_hashes'], [
            hashlib.sha256(path.read_bytes()).hexdigest() for path in self.paths])
        self.assertEqual([item['original_filename'] for item in receipt['source_images']],
                         [path.name for path in self.paths])
        self.assertEqual([(item['width'], item['height']) for item in receipt['source_images']],
                         [(96, 64)] * len(self.paths))
        self.assertEqual(receipt['video_sha256'], hashlib.sha256(Path(result['video_path']).read_bytes()).hexdigest())
        probe = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                                'stream=width,height,r_frame_rate,nb_frames', '-of', 'json', result['video_path']],
                               check=True, capture_output=True, text=True)
        stream = json.loads(probe.stdout)['streams'][0]
        self.assertEqual((stream['width'], stream['height']), (96, 64))
        self.assertEqual(stream['r_frame_rate'], '1/1')
        self.assertEqual(int(stream['nb_frames']), 3)

    def test_rejects_invalid_count_and_mismatched_dimensions_without_output(self):
        target = self.root / 'prepared'
        with self.assertRaisesRegex(ValueError, 'between 2 and 24'):
            prepare_image_set([self.paths[0]], target, camera_id='camera-front')
        mismatch = self.root / 'different.png'
        Image.new('RGB', (100, 64), 'white').save(mismatch)
        with self.assertRaisesRegex(ValueError, 'same dimensions'):
            prepare_image_set([self.paths[0], mismatch], target, camera_id='camera-front')
        self.assertFalse(target.exists())

    def test_rejects_non_image_mislabeled_file_without_output(self):
        invalid = self.root / 'fake.png'
        invalid.write_bytes(b'not an image')
        target = self.root / 'prepared'
        with self.assertRaisesRegex(ValueError, 'corrupt or has an unsupported'):
            prepare_image_set([invalid, self.paths[0]], target, camera_id='camera-front')
        self.assertFalse(target.exists())

    def test_rejects_odd_dimensions_before_ffmpeg(self):
        odd = self.root / 'odd.jpg'
        Image.new('RGB', (95, 63), 'yellow').save(odd)
        target = self.root / 'prepared'
        with self.assertRaisesRegex(ValueError, 'must be even'):
            prepare_image_set([odd, odd], target, camera_id='camera-front')
        self.assertFalse(target.exists())

    def test_refuses_existing_destination_without_overwriting(self):
        target = self.root / 'prepared'
        target.mkdir()
        sentinel = target / 'keep.txt'
        sentinel.write_text('keep')
        with self.assertRaises(FileExistsError):
            prepare_image_set(self.paths, target, camera_id='camera-front')
        self.assertEqual(sentinel.read_text(), 'keep')


if __name__ == '__main__':
    unittest.main()
