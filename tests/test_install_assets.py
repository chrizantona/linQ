import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import urllib.request

from scripts.install_assets import (_AllowlistedBundleRedirectHandler, _safe_archive_path,
                                   install_bundle, read_manifest)


def archive(path, files, extra=None):
    with tarfile.open(path, 'w:gz') as output:
        for name, value in files.items():
            data = value.encode()
            item = tarfile.TarInfo(name)
            item.size = len(data)
            output.addfile(item, io.BytesIO(data))
        if extra:
            output.addfile(extra)
    return path


def manifest_for(path):
    return {'bundle': {'filename': path.name, 'url': 'https://assets.invalid/model.tar.gz',
                       'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}}


def runtime_files():
    return {
        'models/phone.onnx': 'phone model fixture',
        'models/cigarette.onnx': 'cigarette model fixture',
        'models/convnext/manifest.json': '{}',
        'models/linq/rfdetr/report.json': '{"models": []}',
        'models/linq/rfdetr/program.tpu': 'actual program fixture',
    }


class InstallAssetsTests(unittest.TestCase):
    def test_manifest_requires_approved_https_download_host(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            bundle = archive(base / 'bundle.tar.gz', runtime_files())
            manifest_path = base / 'assets.json'
            valid_manifest = manifest_for(bundle)
            valid_manifest['bundle']['url'] = 'https://github.com/example/model.tar.gz'
            manifest_path.write_text(json.dumps(valid_manifest))

            self.assertEqual(read_manifest(base)['bundle']['filename'], 'bundle.tar.gz')
            for url in ('http://github.com/example/model.tar.gz',
                        'https://example.org/model.tar.gz',
                        'https://user@github.com/example/model.tar.gz',
                        'https://github.com:444/example/model.tar.gz'):
                with self.subTest(url=url):
                    manifest_path.write_text(json.dumps({
                        **manifest_for(bundle), 'bundle': {**manifest_for(bundle)['bundle'], 'url': url}}))
                    with self.assertRaisesRegex(ValueError, 'approved GitHub release host'):
                        read_manifest(base)

    def test_rejects_unapproved_redirect_before_creating_request(self):
        handler = _AllowlistedBundleRedirectHandler()
        request = urllib.request.Request('https://github.com/example/model.tar.gz')
        with patch.object(urllib.request.HTTPRedirectHandler, 'redirect_request') as follow:
            with self.assertRaisesRegex(ValueError, 'approved GitHub release host'):
                handler.redirect_request(request, object(), 302, 'Found', {}, 'https://attacker.example/collect')
        follow.assert_not_called()

    def test_installs_verified_files_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            bundle = archive(base / 'bundle.tar.gz', runtime_files())
            root = base / 'project'
            manifest = manifest_for(bundle)

            first = install_bundle(root, bundle, manifest)
            second = install_bundle(root, bundle, manifest)

            self.assertEqual(first['status'], 'installed')
            self.assertEqual(first['files'], len(runtime_files()))
            self.assertEqual(second, first)
            self.assertEqual((root / 'models/linq/rfdetr/program.tpu').read_text(), 'actual program fixture')

    def test_bundle_hash_mismatch_fails_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            bundle = archive(base / 'bundle.tar.gz', runtime_files())
            manifest = manifest_for(bundle)
            manifest['bundle']['sha256'] = '0' * 64
            root = base / 'project'

            with self.assertRaisesRegex(ValueError, 'SHA256'):
                install_bundle(root, bundle, manifest)
            self.assertFalse((root / 'models').exists())

    def test_rejects_path_traversal_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            files = runtime_files()
            files['models/../outside.txt'] = 'escape'
            bundle = archive(base / 'bundle.tar.gz', files)
            root = base / 'project'

            with self.assertRaisesRegex(ValueError, 'unsafe path'):
                install_bundle(root, bundle, manifest_for(bundle))
            self.assertFalse((base / 'outside.txt').exists())
            self.assertFalse((root / 'models').exists())

    def test_accepts_only_the_exact_additional_cigarette_model_path(self):
        self.assertEqual(_safe_archive_path('models/cigarette_rfdetr.onnx').as_posix(),
                         'models/cigarette_rfdetr.onnx')
        for path in ('models/cigarette_rfdetr/extra.onnx', 'models/cigarette_rfdetr.onnx/extra',
                     'models/cigarette2_rfdetr.onnx'):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, 'whitelist'):
                _safe_archive_path(path)

    def test_rejects_symlink_members(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            link = tarfile.TarInfo('models/convnext/encoder.onnx')
            link.type = tarfile.SYMTYPE
            link.linkname = '/outside/encoder.onnx'
            bundle = archive(base / 'bundle.tar.gz', runtime_files(), extra=link)

            with self.assertRaisesRegex(ValueError, 'regular files and directories'):
                install_bundle(base / 'project', bundle, manifest_for(bundle))

    def test_refuses_to_replace_a_different_existing_model(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            bundle = archive(base / 'bundle.tar.gz', runtime_files())
            root = base / 'project'
            target = root / 'models/phone.onnx'
            target.parent.mkdir(parents=True)
            target.write_text('different existing model')

            with self.assertRaisesRegex(FileExistsError, 'Refusing to replace'):
                install_bundle(root, bundle, manifest_for(bundle))
            self.assertEqual(target.read_text(), 'different existing model')


if __name__ == '__main__':
    unittest.main()
