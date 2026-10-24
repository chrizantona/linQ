"""Download, verify, and safely install the packaged runtime model bundle."""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import tempfile
from urllib.parse import urlparse
import urllib.request


ALLOWED_BUNDLE_HOSTS = {
    'github.com',
    'release-assets.githubusercontent.com',
    'objects.githubusercontent.com',
}


def _validate_bundle_url(url):
    parsed = urlparse(url) if isinstance(url, str) else None
    if (parsed is None or parsed.scheme != 'https' or parsed.hostname not in ALLOWED_BUNDLE_HOSTS
            or parsed.port not in (None, 443) or parsed.username or parsed.password):
        raise ValueError('Model bundle URL must use HTTPS on an approved GitHub release host')
    return parsed


class _AllowlistedBundleRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Reject a redirect before urllib opens a connection to an unapproved host."""

    def redirect_request(self, request, response, code, message, headers, new_url):
        _validate_bundle_url(new_url)
        return super().redirect_request(request, response, code, message, headers, new_url)


def sha256_file(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def read_manifest(root):
    manifest = json.loads((Path(root) / 'assets.json').read_text())
    bundle = manifest.get('bundle')
    if not isinstance(bundle, dict):
        raise ValueError('assets.json must contain a bundle object')
    filename, digest, size, url = (bundle.get(key) for key in ('filename', 'sha256', 'bytes', 'url'))
    if not isinstance(filename, str) or Path(filename).name != filename or filename in {'', '.', '..'}:
        raise ValueError('Bundle filename must be a safe basename')
    if not isinstance(digest, str) or not re.fullmatch(r'[a-f0-9]{64}', digest):
        raise ValueError('Bundle SHA256 is required')
    if type(size) is not int or size <= 0:
        raise ValueError('Bundle byte size must be a positive integer')
    if url is not None:
        _validate_bundle_url(url)
    return manifest


def _safe_archive_path(name):
    if not isinstance(name, str) or '\\' in name or name.startswith('/'):
        raise ValueError('Archive contains an unsafe path')
    path = PurePosixPath(name)
    if not path.parts or any(part in {'', '.', '..'} for part in path.parts):
        raise ValueError('Archive contains an unsafe path')
    allowed = (path.parts == ('models',) or
               path.parts[:2] == ('models', 'convnext') or
               path.parts in {('models', 'phone.onnx'), ('models', 'cigarette.onnx'),
                              ('models', 'cigarette_rfdetr.onnx')} or
               path.parts == ('models', 'linq') or
               path.parts[:3] == ('models', 'linq', 'rfdetr'))
    if not allowed:
        raise ValueError('Archive path is outside the model asset whitelist')
    return path


def _manifest_file_hashes(manifest):
    values = manifest.get('files', {})
    if values is None:
        return {}
    if isinstance(values, dict):
        items = ((name, value if isinstance(value, dict) else {'sha256': value}) for name, value in values.items())
    elif isinstance(values, list):
        items = ((item.get('path'), item) for item in values if isinstance(item, dict))
    else:
        raise ValueError('Optional per-file hashes must be an object or list')
    result = {}
    for name, value in items:
        path = _safe_archive_path(name)
        digest = value.get('sha256')
        if not isinstance(digest, str) or not re.fullmatch(r'[a-f0-9]{64}', digest):
            raise ValueError('Invalid per-file SHA256 in assets.json')
        size = value.get('bytes')
        if size is not None and (type(size) is not int or size < 0):
            raise ValueError('Invalid per-file byte size in assets.json')
        if path.as_posix() in result:
            raise ValueError('Duplicate per-file entry in assets.json')
        result[path.as_posix()] = {'sha256': digest, 'bytes': size}
    return result


def _verify_bundle(path, manifest):
    bundle = manifest['bundle']
    if Path(path).stat().st_size != bundle['bytes']:
        raise ValueError('Model bundle byte size does not match assets.json')
    if sha256_file(path) != bundle['sha256']:
        raise ValueError('Model bundle SHA256 does not match assets.json')


def _extract_to_stage(bundle_path, stage, manifest):
    expected_hashes = _manifest_file_hashes(manifest)
    observed_hashes = {}
    seen = set()
    with tarfile.open(bundle_path, mode='r:gz') as archive:
        for member in archive.getmembers():
            path = _safe_archive_path(member.name)
            relative = path.as_posix()
            if relative in seen:
                raise ValueError('Model bundle contains a duplicate path')
            seen.add(relative)
            destination = Path(stage).joinpath(*path.parts)
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise ValueError('Model bundle may contain only regular files and directories')
            source = archive.extractfile(member)
            if source is None:
                raise ValueError('Could not read a model bundle member')
            destination.parent.mkdir(parents=True, exist_ok=True)
            with source, destination.open('xb') as output:
                shutil.copyfileobj(source, output)
            observed_hashes[relative] = sha256_file(destination)
    if not observed_hashes:
        raise ValueError('Model bundle contains no model files')
    for name, expected in expected_hashes.items():
        if name not in observed_hashes or observed_hashes[name] != expected['sha256']:
            raise ValueError('Per-file SHA256 mismatch for ' + name)
        if expected['bytes'] is not None and (Path(stage) / name).stat().st_size != expected['bytes']:
            raise ValueError('Per-file byte size mismatch for ' + name)
    if expected_hashes and set(observed_hashes) != set(expected_hashes):
        raise ValueError('Model bundle file list differs from assets.json')
    required = {'models/phone.onnx', 'models/cigarette.onnx', 'models/convnext/manifest.json',
                'models/linq/rfdetr/report.json'}
    if not required.issubset(observed_hashes):
        raise ValueError('Model bundle is missing a required runtime manifest or checkpoint')
    return observed_hashes


def _check_destination(root, relative_paths, stage):
    root = Path(root).resolve()
    for relative in relative_paths:
        target = root.joinpath(*PurePosixPath(relative).parts)
        cursor = root
        for part in PurePosixPath(relative).parts[:-1]:
            cursor = cursor / part
            if cursor.exists() and (cursor.is_symlink() or not cursor.is_dir()):
                raise ValueError('Refusing an unsafe existing model directory')
            if not cursor.resolve().is_relative_to(root):
                raise ValueError('Model destination escapes the project root')
        if target.is_symlink():
            raise ValueError('Refusing a symlink at a model destination')
        staged = Path(stage).joinpath(*PurePosixPath(relative).parts)
        if target.exists():
            if not target.is_file() or sha256_file(target) != sha256_file(staged):
                raise FileExistsError('Refusing to replace a different existing model asset: ' + relative)


def install_bundle(root, bundle_path, manifest):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    _verify_bundle(bundle_path, manifest)
    with tempfile.TemporaryDirectory(prefix='.asset-stage-', dir=root) as temporary:
        stage = Path(temporary)
        hashes = _extract_to_stage(bundle_path, stage, manifest)
        _check_destination(root, hashes, stage)
        for relative in hashes:
            target = root.joinpath(*PurePosixPath(relative).parts)
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(stage.joinpath(*PurePosixPath(relative).parts), target)
    return {'status': 'installed', 'files': len(hashes), 'sha256': manifest['bundle']['sha256']}


def download_bundle(manifest, destination):
    bundle = manifest['bundle']
    _validate_bundle_url(bundle['url'])
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + '.part')
    try:
        opener = urllib.request.build_opener(_AllowlistedBundleRedirectHandler())
        with opener.open(bundle['url'], timeout=60) as response, temporary.open('wb') as output:
            received = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                received += len(chunk)
                if received > bundle['bytes']:
                    raise ValueError('Downloaded model bundle exceeds manifest byte size')
                output.write(chunk)
        if received != bundle['bytes']:
            raise ValueError('Downloaded model bundle byte size differs from assets.json')
        _verify_bundle(temporary, manifest)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def main():
    parser = argparse.ArgumentParser(description='Install SHA-verified local runtime model assets')
    parser.add_argument('--assets-dir', type=Path, help='Directory containing the bundle named in assets.json')
    parser.add_argument('--bundle', type=Path, help='Explicit local model bundle')
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1], help='Project root receiving models/')
    args = parser.parse_args()
    if args.assets_dir and args.bundle:
        parser.error('Choose either --assets-dir or --bundle')
    root = args.root.resolve()
    # --root selects the install destination; the checked-in manifest remains
    # next to this script so assets can be installed into an arbitrary project
    # directory without requiring a second copy of assets.json there.
    manifest = read_manifest(Path(__file__).resolve().parents[1])
    if args.bundle:
        bundle_path = args.bundle
    elif args.assets_dir:
        bundle_path = args.assets_dir / manifest['bundle']['filename']
    else:
        if not manifest['bundle']['url']:
            raise ValueError('No bundle URL is published yet; pass --bundle or --assets-dir')
        cache = root / '.cache' / 'asset-downloads' / manifest['bundle']['filename']
        bundle_path = download_bundle(manifest, cache)
    print(json.dumps(install_bundle(root, bundle_path, manifest)))


if __name__ == '__main__':
    main()
