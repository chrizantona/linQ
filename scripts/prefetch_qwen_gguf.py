import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
import tempfile
import threading
import time
import urllib.request


REPOSITORY = 'lmstudio-community/Qwen3.5-0.8B-GGUF'
REVISION = '26bab2c9369648924251c0ebb3dae012f5147707'
RELEASE = 'b11853'
FILES = [
    ('Qwen3.5-0.8B-Q4_K_M.gguf', 527502816, 'f5b14da98939b60bbe1019a964eba656407e1e0b64f1fe3003ff6d650e93bfec'),
    ('mmproj-Qwen3.5-0.8B-BF16.gguf', 207345952, '6fdd1b4bdc3d2ae8bd15d783e23260dd07dcf83f45604a21dabfd6efad8f8bc5'),
]
BINARY = ('llama-b11853-bin-ubuntu-x64.tar.gz', 22438758, 'b0bba21f5673f80f242c1a8f7c1380800f08b6eff6f32fd55eae9fe220690fdc')
DEFAULT_ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def extract_runtime(archive, destination):
    if destination.exists():
        raise FileExistsError('Runtime destination already exists')
    staging = Path(tempfile.mkdtemp(prefix='.llama-extract-', dir=destination.parent))
    try:
        with tarfile.open(archive, 'r:gz') as bundle:
            members = bundle.getmembers()
            if len(members) > 500 or sum(member.size for member in members) > 512 * 1024 * 1024:
                raise ValueError('Unexpected runtime archive size')
            for member in members:
                path = PurePosixPath(member.name)
                if path.is_absolute() or '..' in path.parts or not (member.isfile() or member.isdir() or member.issym() or member.islnk()):
                    raise ValueError('Unsafe runtime archive entry')
            bundle.extractall(staging, filter='data')
        servers = [path for path in staging.rglob('llama-server') if path.is_file() and not path.is_symlink()]
        if len(servers) != 1:
            raise ValueError('Expected exactly one llama-server')
        os.replace(servers[0].parent, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default=DEFAULT_ROOT)
    args = parser.parse_args()
    root = Path(args.root).expanduser().resolve()
    model_dir = root / 'models/qwen_small'
    binary_dir = root / 'bin'
    model_dir.mkdir(parents=True, exist_ok=True)
    binary_dir.mkdir(parents=True, exist_ok=True)
    lock_fd = os.open(model_dir / '.prefetch.lock', os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(lock_fd)
    started = time.monotonic()
    lock = threading.Lock()
    receipt = {'status': 'prefetch_in_progress', 'repository': REPOSITORY, 'revision': REVISION,
               'runtime_release': RELEASE, 'model_execution': False, 'files': []}

    def save():
        temporary = model_dir / 'prefetch_receipt.json.tmp'
        temporary.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
        os.replace(temporary, model_dir / 'prefetch_receipt.json')

    def report(value):
        with lock:
            print(json.dumps(value), flush=True)

    def download(item, directory, url):
        name, size, expected = item
        destination = directory / name
        if destination.exists() and destination.stat().st_size == size and sha256(destination) == expected:
            return {'path': str(destination.relative_to(root)), 'bytes': size, 'sha256': expected, 'verified': True, 'reused': True}
        partial = destination.with_name(name + '.part')
        for attempt in range(1, 4):
            try:
                digest = hashlib.sha256()
                received = 0
                last_report = time.monotonic()
                request = urllib.request.Request(url, headers={'User-Agent': 'LinQ-Qwen-GGUF-prefetch/1.0'})
                with urllib.request.urlopen(request, timeout=60) as response, partial.open('wb') as stream:
                    while chunk := response.read(4 * 1024 * 1024):
                        received += len(chunk)
                        if received > size:
                            raise ValueError('Download exceeds published size')
                        stream.write(chunk)
                        digest.update(chunk)
                        if time.monotonic() - last_report >= 10:
                            report({'stage': 'downloading', 'file': name, 'bytes': received, 'expected_bytes': size})
                            last_report = time.monotonic()
                    stream.flush()
                    os.fsync(stream.fileno())
                if received != size or digest.hexdigest() != expected:
                    raise ValueError('Published size or SHA256 mismatch')
                os.replace(partial, destination)
                return {'path': str(destination.relative_to(root)), 'bytes': received, 'sha256': expected, 'verified': True, 'reused': False}
            except Exception as error:
                report({'stage': 'retry', 'file': name, 'attempt': attempt, 'error_type': type(error).__name__, 'http_status': getattr(error, 'code', None)})
                if attempt == 3:
                    raise
                time.sleep(attempt)

    save()
    try:
        tasks = [(item, model_dir, 'https://huggingface.co/' + REPOSITORY + '/resolve/' + REVISION + '/' + item[0]) for item in FILES]
        tasks.append((BINARY, binary_dir, 'https://github.com/ggml-org/llama.cpp/releases/download/' + RELEASE + '/' + BINARY[0]))
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(download, *task) for task in tasks]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                receipt['files'].append(result)
                save()
                report({'stage': 'file_verified', **result})
        runtime = binary_dir / 'llama'
        install_receipt = runtime / 'install_receipt.json'
        if runtime.exists():
            if not install_receipt.is_file() or json.loads(install_receipt.read_text()).get('archive_sha256') != BINARY[2]:
                raise ValueError('Existing runtime has no matching installation receipt')
        else:
            extract_runtime(binary_dir / BINARY[0], runtime)
            install_receipt.write_text(json.dumps({'archive_sha256': BINARY[2], 'release': RELEASE}) + '\n')
        environment = dict(os.environ)
        environment['LD_LIBRARY_PATH'] = str(runtime) + (':' + environment['LD_LIBRARY_PATH'] if environment.get('LD_LIBRARY_PATH') else '')
        version = subprocess.run([str(runtime / 'llama-server'), '--version'], env=environment, capture_output=True, text=True, timeout=30)
        receipt['runtime_version_check'] = {'returncode': version.returncode, 'stdout': version.stdout[-3000:], 'stderr': version.stderr[-3000:]}
        receipt.update(status='all_required_files_verified_no_model_execution', total_bytes=sum(item['bytes'] for item in receipt['files']),
                       elapsed_s=time.monotonic() - started, runtime_path=str(runtime / 'llama-server'))
        save()
        report({'stage': 'complete', 'status': receipt['status'], 'bytes': receipt['total_bytes'], 'runtime_version_returncode': version.returncode})
    except Exception as error:
        receipt.update(status='prefetch_failed', error_type=type(error).__name__, http_status=getattr(error, 'code', None), elapsed_s=time.monotonic() - started)
        save()
        report({'stage': 'failed', 'status': receipt['status'], 'error_type': type(error).__name__, 'http_status': receipt['http_status']})
        raise SystemExit(1)
    finally:
        (model_dir / '.prefetch.lock').unlink(missing_ok=True)


if __name__ == '__main__':
    main()
