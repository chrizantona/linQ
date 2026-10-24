import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import install_demo_examples as installer


SOURCE_BYTES = b'verified full source fixture'


def source_record(sample_id='pexels-test-phone'):
    return {
        'id': sample_id,
        'case': 'phone',
        'title': 'Phone group',
        'shortLabel': 'Phone group',
        'icon': 'phone',
        'rule': 'Find visible phone use.',
        'creator': 'Contributor',
        'sourceUrl': 'https://www.pexels.com/video/a-video-1234567/',
        'downloadUrl': 'https://videos.pexels.com/video-files/1234567/1234567-hd_1920_1080_25fps.mp4',
        'sourceFilename': '1234567-hd_1920_1080_25fps.mp4',
        'sourceSha256': hashlib.sha256(SOURCE_BYTES).hexdigest(),
        'license': 'Pexels License',
        'licenseUrl': 'https://www.pexels.com/license/',
        'attribution': 'Contributor via Pexels',
        'sourceStart': 1.0,
        'excerptDuration': 6.0,
    }


class FakeResponse(io.BytesIO):
    def __init__(self, content, url='https://videos.pexels.com/video-files/1234567/1234567-hd_1920_1080_25fps.mp4'):
        super().__init__(content)
        self.url = url

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def geturl(self):
        return self.url


def fake_prepare(source, raw, video, poster, ffmpeg='ffmpeg'):
    video.write_bytes(b'trimmed video bytes')
    poster.write_bytes(b'poster bytes')


class InstallDemoExamplesTests(unittest.TestCase):
    def test_installs_verified_excerpt_poster_and_catalog(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / 'sources.json'
            manifest.write_text(json.dumps([source_record()]))
            with patch.object(installer, '_prepare_example', side_effect=fake_prepare):
                result = installer.install_examples(root, manifest, opener=lambda request, timeout: FakeResponse(SOURCE_BYTES))

            video = root / 'service/assets/demo-local/pexels-test-phone.mp4'
            poster = root / 'service/assets/demo-local/pexels-test-phone.jpg'
            catalog_path = root / 'service/samples.local.json'
            catalog = json.loads(catalog_path.read_text())
            self.assertEqual(result['examples'], 1)
            self.assertEqual(video.read_bytes(), b'trimmed video bytes')
            self.assertEqual(poster.read_bytes(), b'poster bytes')
            self.assertEqual(catalog[0]['video'], 'assets/demo-local/pexels-test-phone.mp4')
            self.assertEqual(catalog[0]['image'], 'assets/demo-local/pexels-test-phone.jpg')
            self.assertEqual(catalog[0]['sha256'], hashlib.sha256(b'trimmed video bytes').hexdigest())
            self.assertNotIn('unsafe', catalog[0])
            self.assertNotIn('operatorConfirmed', catalog[0])

    def test_source_hash_mismatch_does_not_overwrite_existing_media_or_catalog(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = source_record()
            source['sourceSha256'] = '0' * 64
            manifest = root / 'sources.json'
            manifest.write_text(json.dumps([source]))
            video = root / 'service/assets/demo-local/pexels-test-phone.mp4'
            video.parent.mkdir(parents=True)
            video.write_bytes(b'previous media')
            catalog = root / 'service/samples.local.json'
            catalog.write_text('[{"id":"previous"}]\n')

            with self.assertRaisesRegex(ValueError, 'source SHA256'):
                installer.install_examples(root, manifest, opener=lambda request, timeout: FakeResponse(SOURCE_BYTES))

            self.assertEqual(video.read_bytes(), b'previous media')
            self.assertEqual(catalog.read_text(), '[{"id":"previous"}]\n')
            self.assertFalse((video.parent / '1234567-hd_1920_1080_25fps.mp4.part').exists())

    def test_rejects_path_traversal_before_downloading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / 'sources.json'
            manifest.write_text(json.dumps([source_record('../escape')]))
            opened = []

            with self.assertRaisesRegex(ValueError, 'path-safe'):
                installer.install_examples(root, manifest, opener=lambda *args, **kwargs: opened.append(True))

            self.assertEqual(opened, [])
            self.assertFalse((root.parent / 'escape.mp4').exists())
            self.assertFalse((root / 'service').exists())

    def test_allows_verified_2048_source_dimensions_and_rejects_other_variants(self):
        source = source_record()
        source['downloadUrl'] = 'https://videos.pexels.com/video-files/6965763/6965763-hd_2048_1080_25fps.mp4'
        source['sourceFilename'] = '6965763-hd_2048_1080_25fps.mp4'
        self.assertEqual(installer._validate_sources([source]), [source])

        source['downloadUrl'] = 'https://videos.pexels.com/video-files/6965763/6965763-hd_2560_1440_25fps.mp4'
        source['sourceFilename'] = '6965763-hd_2560_1440_25fps.mp4'
        with self.assertRaisesRegex(ValueError, 'allowlisted Pexels media URL'):
            installer._validate_sources([source])

    def test_rejects_redirect_outside_allowlisted_host_without_installing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / 'sources.json'
            manifest.write_text(json.dumps([source_record()]))

            with self.assertRaisesRegex(ValueError, 'redirected outside'):
                installer.install_examples(root, manifest,
                    opener=lambda request, timeout: FakeResponse(SOURCE_BYTES, 'https://example.org/hostile.mp4'))

            self.assertFalse((root / 'service').exists())

    def test_redirect_handler_rejects_private_target_before_second_request(self):
        handler = installer._AllowlistedRedirectHandler()
        initial = installer.urllib.request.Request(source_record()['downloadUrl'])
        contacted = []

        def redirecting_opener(request, timeout):
            # urllib invokes redirect_request before opening the redirect target.
            handler.redirect_request(request, object(), 302, 'Found', {}, 'http://127.0.0.1/admin')
            contacted.append('http://127.0.0.1/admin')

        with self.assertRaisesRegex(ValueError, 'redirected outside'):
            redirecting_opener(initial, timeout=60)
        self.assertEqual(contacted, [])


if __name__ == '__main__':
    unittest.main()
