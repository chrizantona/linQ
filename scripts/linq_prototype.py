import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import time
import urllib.error
import urllib.request

PORT = 18775


def root_path():
    return Path(__file__).resolve().parents[1]


def secret(root):
    path = root / '.runtime/vlm_token'
    path.parent.mkdir(mode=0o700, exist_ok=True)
    if not path.exists():
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(descriptor, 'w') as stream:
            stream.write(secrets.token_urlsafe(32))
    return path.read_text().strip()



def vlm_target(root):
    return dict(mode='local', url=f'http://127.0.0.1:{PORT}/infer', token=secret(root))


def environment(root):
    target = vlm_target(root)
    result = {**os.environ, 'VLM_HTTP_TOKEN': target['token'],
              'VLM_HTTP_URL': target['url'],
              'YOLO_CONFIG_DIR': str(root / '.cache/ultralytics'),
              'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
              'OPENBLAS_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1',
              'LD_LIBRARY_PATH': str(root / 'bin/llama') + (':' + os.environ['LD_LIBRARY_PATH'] if os.environ.get('LD_LIBRARY_PATH') else ''),
              'TOKENIZERS_PARALLELISM': 'false', 'HF_HUB_OFFLINE': '1'}
    result.pop('VLM_CA_FILE', None)
    return result


def health(root):
    target = vlm_target(root)
    request = urllib.request.Request(f'http://127.0.0.1:{PORT}/health',
                                   headers={'Authorization': 'Bearer ' + target['token']})
    try:
        with urllib.request.urlopen(request, timeout=3) as response:
            return {**json.load(response), 'mode': target['mode']}
    except (urllib.error.URLError, TimeoutError):
        return {'ready': False, 'mode': target['mode']}


def launch(root, name, command):
    directory = root / '.runtime'
    directory.mkdir(mode=0o700, exist_ok=True)
    log_path = root / (name + '.log')
    with log_path.open('ab', buffering=0) as log:
        process = subprocess.Popen(command, cwd=root, env=environment(root),
                                   stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    receipt = {'pid': process.pid, 'started_at': time.time(), 'log': log_path.name,
               'working_directory': str(root), 'command': command}
    (directory / (name + '.json')).write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


def start_vlm(root):
    current = health(root)
    if current.get('ready'):
        return {'status': 'already_ready', 'health': current}
    model = root / 'models/qwen_small'
    files = {
        'Qwen3.5-0.8B-Q4_K_M.gguf': 'f5b14da98939b60bbe1019a964eba656407e1e0b64f1fe3003ff6d650e93bfec',
        'mmproj-Qwen3.5-0.8B-BF16.gguf': '6fdd1b4bdc3d2ae8bd15d783e23260dd07dcf83f45604a21dabfd6efad8f8bc5',
    }
    for name, expected in files.items():
        value = hashlib.sha256()
        with (model / name).open('rb') as stream:
            for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                value.update(chunk)
        if value.hexdigest() != expected:
            raise ValueError('Qwen model SHA256 mismatch: ' + name)
    command = [str(root / 'bin/llama/llama-server'),
               '-m', str(model / 'Qwen3.5-0.8B-Q4_K_M.gguf'),
               '--mmproj', str(model / 'mmproj-Qwen3.5-0.8B-BF16.gguf'),
               '--alias', 'Qwen3.5-0.8B-Q4_K_M',
               '-ngl', '0', '--no-mmproj-offload', '-t', '2', '-tb', '2',
               '-c', '2048', '-b', '512', '-ub', '256', '-np', '1',
               '--image-max-tokens', '256', '--chat-template-kwargs', '{"enable_thinking":false}',
               '--video-fps', '1', '--video-timestamp-interval', '0',
               '--host', '127.0.0.1', '--port', '18776']
    backend = None
    previous = root / '.runtime/llama_cpu.json'
    if previous.exists():
        candidate = json.loads(previous.read_text())
        proc = Path('/proc') / str(candidate['pid'])
        if proc.exists():
            actual = (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
            if 'bin/llama/llama-server' in actual and (proc / 'cwd').resolve() == root:
                backend = {**candidate, 'reused_own_process': True}
    if backend is None:
        backend = launch(root, 'llama_cpu', command)
    deadline = time.monotonic() + 45
    while True:
        try:
            with urllib.request.urlopen('http://127.0.0.1:18776/health', timeout=1) as response:
                if json.load(response).get('status') == 'ok':
                    break
        except (urllib.error.URLError, TimeoutError):
            pass
        if time.monotonic() > deadline:
            raise TimeoutError('Local Qwen backend did not become ready; inspect llama_cpu.log')
        time.sleep(1)
    bridge = [str(root / '.venv/bin/python'), '-m', 'deploy.llama_vlm_server',
              '--model', 'Qwen3.5-0.8B-Q4_K_M', '--port', str(PORT),
              '--input-mode', 'native_video',
              '--max-new-tokens', '192', '--audit-log', str(root / 'vlm_requests.jsonl')]
    return {'status': 'starting', 'backend': backend, **launch(root, 'vlm_server', bridge)}


def run_pipeline(root, cameras, name=None):
    current = health(root)
    if not current.get('ready'):
        raise RuntimeError('Start the VLM server and wait for model readiness')
    run_name = name or ('single' if cameras == 1 else 'two')
    if not run_name.replace('_', '').replace('-', '').isalnum():
        raise ValueError('Use a simple run name')
    output = root / 'runs' / run_name
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Run exists; choose --name for a new run')
    config = root / ('config_single.json' if cameras == 1 else 'config_two.json')
    command = [str(root / '.venv/bin/python'), '-m', 'deploy.pipeline',
               '--config', str(config), '--output', str(output)]
    return {'status': 'starting', 'output': str(output),
            **launch(root, 'run_' + run_name, command)}


def stop_vlm(root):
    stopped = []
    for name, allowed in [('vlm_server', ['deploy.vlm_server', 'deploy.llama_vlm_server']),
                          ('llama_cpu', ['bin/llama/llama-server'])]:
        path = root / ('.runtime/' + name + '.json')
        if not path.exists():
            continue
        receipt = json.loads(path.read_text())
        pid = receipt['pid']
        proc = Path('/proc') / str(pid)
        if not proc.exists():
            continue
        command = (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
        if not any(value in command for value in allowed) or (proc / 'cwd').resolve() != root:
            raise ValueError('Refusing to signal a process not owned by this prototype')
        os.kill(pid, signal.SIGTERM)
        stopped.append(pid)
    return {'status': 'stop_requested' if stopped else 'already_stopped', 'pids': stopped}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['status', 'start-vlm', 'run', 'stop-vlm'])
    parser.add_argument('--cameras', type=int, choices=[1, 2], default=1)
    parser.add_argument('--name')
    args = parser.parse_args()
    root = root_path()
    if args.action == 'status':
        result = health(root)
    elif args.action == 'start-vlm':
        result = start_vlm(root)
    elif args.action == 'stop-vlm':
        result = stop_vlm(root)
    else:
        result = run_pipeline(root, args.cameras, args.name)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
