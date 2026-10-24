"""Install small, source-verified Pexels video examples for the local demo."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
from urllib.parse import urlparse
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / 'examples' / 'media_sources.json'
ALLOWED_MEDIA_HOST = 'videos.pexels.com'
MAX_SOURCE_BYTES = 100 * 1024 * 1024
ID_PATTERN = re.compile(r'[a-z0-9]+(?:-[a-z0-9]+)*')
SHA_PATTERN = re.compile(r'[a-f0-9]{64}')
MEDIA_PATH_PATTERN = re.compile(r'/video-files/[0-9]+/[0-9]+-hd_(?:1920|2048)_1080_(?:25|30)fps\.mp4')


class _AllowlistedRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Reject an unsafe redirect before urllib opens a connection to its target."""

    def redirect_request(self, request, response, code, message, headers, new_url):
        target = urlparse(new_url)
        if (target.scheme != 'https' or target.hostname != ALLOWED_MEDIA_HOST
                or target.port not in (None, 443) or target.username or target.password
                or not MEDIA_PATH_PATTERN.fullmatch(target.path)):
            raise ValueError('Pexels download redirected outside the allowlisted host')
        return super().redirect_request(request, response, code, message, headers, new_url)


def sha256_file(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def _validate_sources(sources):
    if not isinstance(sources, list) or not sources:
        raise ValueError('Media source manifest must be a nonempty list')
    seen = set()
    required = ('id', 'case', 'title', 'shortLabel', 'icon', 'rule', 'creator', 'sourceUrl',
                'downloadUrl', 'sourceFilename', 'sourceSha256', 'license', 'licenseUrl',
                'attribution', 'sourceStart', 'excerptDuration')
    for source in sources:
        if not isinstance(source, dict) or any(key not in source for key in required):
            raise ValueError('Each media source must contain the required fields')
        sample_id = source['id']
        if not isinstance(sample_id, str) or not ID_PATTERN.fullmatch(sample_id) or sample_id in seen:
            raise ValueError('Media source id must be unique and path-safe')
        seen.add(sample_id)
        if source['case'] not in {'phone', 'smoking'}:
            raise ValueError('Unsupported demo example case')
        filename = source['sourceFilename']
        if not isinstance(filename, str) or Path(filename).name != filename or not filename.endswith('.mp4'):
            raise ValueError('Source filename must be a safe MP4 basename')
        if not isinstance(source['sourceSha256'], str) or not SHA_PATTERN.fullmatch(source['sourceSha256']):
            raise ValueError('Source SHA256 must be a lowercase 64-character digest')
        url = urlparse(source['downloadUrl'])
        if url.scheme != 'https' or url.hostname != ALLOWED_MEDIA_HOST or url.port not in (None, 443) or url.username or url.password or not MEDIA_PATH_PATTERN.fullmatch(url.path):
            raise ValueError('Download URL must be an allowlisted Pexels media URL')
        if not url.path.endswith('/' + filename):
            raise ValueError('Download URL filename does not match sourceFilename')
        page = urlparse(source['sourceUrl'])
        if page.scheme != 'https' or page.hostname != 'www.pexels.com' or not page.path.startswith('/video/'):
            raise ValueError('Source page must be a Pexels video page')
        for key in ('title', 'shortLabel', 'icon', 'rule', 'creator', 'sourceUrl', 'downloadUrl', 'license', 'licenseUrl', 'attribution'):
            if not isinstance(source[key], str) or not source[key].strip():
                raise ValueError('Media source field must be nonempty text: ' + key)
        for key in ('sourceStart', 'excerptDuration'):
            number = source[key]
            if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number) or number < 0:
                raise ValueError('Media source timing must be finite and nonnegative')
        if source['excerptDuration'] <= 0:
            raise ValueError('Excerpt duration must be positive')
    return sources


def read_sources(path=SOURCES):
    return _validate_sources(json.loads(Path(path).read_text(encoding='utf-8')))


def download_source(source, destination, opener=None):
    """Download one fixed-host source to a new file and verify its full-file SHA."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(source['downloadUrl'], headers={'User-Agent': 'LinQDemoExampleInstaller/1.0'})
    opener = opener or urllib.request.build_opener(_AllowlistedRedirectHandler()).open
    temporary = destination.with_name(destination.name + '.part')
    temporary.unlink(missing_ok=True)
    total = 0
    try:
        with opener(request, timeout=60) as response, temporary.open('xb') as output:
            final_url = urlparse(response.geturl())
            if (final_url.scheme != 'https' or final_url.hostname != ALLOWED_MEDIA_HOST
                    or final_url.port not in (None, 443) or final_url.username or final_url.password
                    or not MEDIA_PATH_PATTERN.fullmatch(final_url.path)):
                raise ValueError('Pexels download redirected outside the allowlisted host')
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_SOURCE_BYTES:
                    raise ValueError('Pexels source exceeds the installer size limit')
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        if total == 0 or sha256_file(temporary) != source['sourceSha256']:
            raise ValueError('Downloaded source SHA256 does not match media_sources.json')
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _run_ffmpeg(command):
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError as error:
        raise RuntimeError('ffmpeg is required to install the demo examples') from error
    except subprocess.CalledProcessError as error:
        detail = error.stderr.decode('utf-8', errors='replace')[-1200:]
        raise RuntimeError('ffmpeg could not prepare a demo example: ' + detail) from error


def _prepare_example(source, raw, video, poster, ffmpeg='ffmpeg'):
    _run_ffmpeg([ffmpeg, '-hide_banner', '-loglevel', 'error', '-y', '-ss', str(source['sourceStart']),
                 '-i', str(raw), '-t', str(source['excerptDuration']), '-map', '0:v:0', '-an',
                 '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '25', '-movflags', '+faststart', str(video)])
    _run_ffmpeg([ffmpeg, '-hide_banner', '-loglevel', 'error', '-y', '-ss', '2', '-i', str(video),
                 '-frames:v', '1', '-q:v', '3', str(poster)])
    if not video.is_file() or video.stat().st_size == 0 or not poster.is_file() or poster.stat().st_size == 0:
        raise ValueError('ffmpeg did not create a nonempty example video and poster')


def _sample_record(source, video_sha):
    sample_id = source['id']
    stem = 'assets/demo-local/' + sample_id
    return {
        'id': sample_id,
        'case': source['case'],
        'title': source['title'],
        'shortLabel': source['shortLabel'],
        'icon': source['icon'],
        'rule': source['rule'],
        'source': source['creator'] + ' · Pexels',
        'sourceUrl': source['sourceUrl'],
        'license': source['license'],
        'licenseUrl': source['licenseUrl'],
        'attribution': source['attribution'],
        'originalFile': source['sourceFilename'],
        'sourceStart': source['sourceStart'],
        'duration': source['excerptDuration'],
        'image': stem + '.jpg',
        'video': stem + '.mp4',
        'sha256': video_sha,
    }


def _write_catalog(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix='.' + path.name + '.', suffix='.tmp', dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            stream.write(json.dumps(records, ensure_ascii=False, indent=2) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _check_output_paths(root, output_dir, catalog_path):
    for target, final_is_dir in ((output_dir, True), (catalog_path, False)):
        cursor = root
        parts = target.relative_to(root).parts
        for index, part in enumerate(parts):
            cursor = cursor / part
            if not cursor.exists():
                continue
            should_be_dir = index < len(parts) - 1 or final_is_dir
            if cursor.is_symlink() or (should_be_dir and not cursor.is_dir()) or (not should_be_dir and not cursor.is_file()):
                raise ValueError('Refusing an unsafe existing demo example path')
            if not cursor.resolve().is_relative_to(root):
                raise ValueError('Demo example destination escapes the repository root')


def install_examples(root=ROOT, sources_path=SOURCES, opener=None, ffmpeg='ffmpeg'):
    """Download, verify, trim, and install the fixed Pexels examples."""
    root = Path(root).resolve()
    sources = read_sources(sources_path)
    root.mkdir(parents=True, exist_ok=True)
    output_dir = root / 'service' / 'assets' / 'demo-local'
    catalog_path = root / 'service' / 'samples.local.json'
    _check_output_paths(root, output_dir, catalog_path)
    with tempfile.TemporaryDirectory(prefix='.demo-examples-stage-', dir=root) as temporary:
        stage = Path(temporary)
        staged_media = []
        records = []
        for source in sources:
            raw = stage / source['sourceFilename']
            download_source(source, raw, opener=opener)
            video = stage / (source['id'] + '.mp4')
            poster = stage / (source['id'] + '.jpg')
            _prepare_example(source, raw, video, poster, ffmpeg=ffmpeg)
            video_relative = Path('service/assets/demo-local') / video.name
            poster_relative = Path('service/assets/demo-local') / poster.name
            staged_media.extend(((video, root / video_relative), (poster, root / poster_relative)))
            records.append(_sample_record(source, sha256_file(video)))

        for staged, destination in staged_media:
            if destination.is_symlink():
                raise ValueError('Refusing a symlink at a demo example destination')
            if destination.exists() and (not destination.is_file() or sha256_file(destination) != sha256_file(staged)):
                raise FileExistsError('Refusing to replace a different local demo example: ' + str(destination.relative_to(root)))

        output_dir.mkdir(parents=True, exist_ok=True)
        for staged, destination in staged_media:
            if not destination.exists():
                os.replace(staged, destination)
        _write_catalog(catalog_path, records)
    return {'status': 'installed', 'examples': len(records), 'catalog': str(catalog_path), 'mediaDirectory': str(output_dir)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT, help='LinQ repository root (defaults to this checkout)')
    parser.add_argument('--sources', type=Path, default=SOURCES, help='Allowlisted media source manifest')
    parser.add_argument('--ffmpeg', default='ffmpeg', help='ffmpeg executable')
    args = parser.parse_args(argv)
    result = install_examples(args.root, args.sources, ffmpeg=args.ffmpeg)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
