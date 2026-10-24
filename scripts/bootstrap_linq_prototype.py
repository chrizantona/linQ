import argparse
import json
import platform
from pathlib import Path
import subprocess
import sys
import time
import zipfile


def validate_protocol_wheel(path):
    path = Path(path)
    if not path.is_file() or path.suffix != '.whl':
        raise FileNotFoundError('Provide the LinQ-supplied tpu_pycloud_format wheel: ' + str(path))
    try:
        with zipfile.ZipFile(path) as archive:
            metadata_files = [name for name in archive.namelist() if name.endswith('.dist-info/METADATA')]
            if len(metadata_files) != 1:
                raise ValueError('Wheel must contain exactly one dist-info/METADATA file')
            metadata = archive.read(metadata_files[0]).decode('utf-8')
    except (zipfile.BadZipFile, UnicodeDecodeError) as error:
        raise ValueError('Invalid LinQ protocol wheel') from error
    headers = {}
    for line in metadata.splitlines():
        if ': ' in line:
            key, value = line.split(': ', 1)
            headers.setdefault(key.lower(), value)
    if headers.get('name', '').replace('-', '_').lower() != 'tpu_pycloud_format' or headers.get('version') != '0.0.1':
        raise ValueError('Wheel must be tpu_pycloud_format version 0.0.1')
    return path


def validate_host():
    if platform.system() != 'Linux' or platform.machine().lower() not in {'x86_64', 'amd64'}:
        raise RuntimeError('The LinQ prototype runtime requires Linux x86_64')
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError('The LinQ prototype runtime requires Python 3.12')


def main():
    parser = argparse.ArgumentParser(description='Create the LinQ CPU/TPU prototype runtime')
    parser.add_argument('--protocol-wheel', type=Path,
                        default=Path.home() / 'work/packages/tpu_pycloud_format-0.0.1-py3-none-any.whl',
                        help='LinQ-provided tpu_pycloud_format 0.0.1 wheel')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    environment = root / '.venv'
    status = root / 'setup_status.json'
    def record(stage, **extra):
        status.write_text(json.dumps({'stage': stage, 'updated_at': time.time(), **extra}, indent=2) + '\n')
    record('creating_environment')
    try:
        validate_host()
        protocol = validate_protocol_wheel(args.protocol_wheel)
        subprocess.run([sys.executable, '-m', 'venv', str(environment)], check=True)
        python = str(environment / 'bin/python')
        commands = [
            [python, '-m', 'pip', 'install', '--index-url', 'https://download.pytorch.org/whl/cpu', 'torch==2.13.0', 'torchvision==0.28.0'],
            [python, '-m', 'pip', 'install', 'numpy==2.5.1', 'Pillow==12.3.0', 'onnxruntime==1.31.0',
             'fastapi==0.136.3', 'uvicorn==0.48.0', 'python-multipart==0.0.32', 'requests==2.32.5',
             'websocket-client==1.9.0', 'pika==1.3.2', 'ultralytics==8.4.184'],
        ]
        for index, command in enumerate(commands):
            record('installing_dependencies', step=index + 1, steps=len(commands))
            subprocess.run(command, check=True)
        subprocess.run([python, '-m', 'pip', 'install', '--no-deps', str(protocol)], check=True)
        record('ready', python=python)
    except Exception as error:
        record('error', error_type=type(error).__name__, error=str(error))
        raise


if __name__ == '__main__':
    main()
