"""Upload the reviewed LinQ demo source and verified model assets to one's workspace."""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tarfile
import tempfile
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from deploy import linq_client

CHUNK_SIZE = 4 * 1024 * 1024
MAX_ARCHIVE_BYTES = 4 * 1024 * 1024 * 1024
CODE_FILES = (
    "config_single.json",
    "config_two.json",
)
SCRIPT_FILES = (
    "scripts/bootstrap_linq_prototype.py",
    "scripts/linq_prototype.py",
    "scripts/prefetch_qwen_gguf.py",
    "scripts/render_causal_demo.py",
    "scripts/run_user_video.py",
)
ALLOWED_SOURCE_DIRS = ("demo", "deploy", "service")
EXCLUDED_PARTS = {"tests", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".git", ".secrets", ".runtime", "runs", "videos", "logs", "cache"}
EXCLUDED_MEDIA_DIRS = {"service/assets/industrial", "service/assets/samples", "service/assets/demo-local"}
IDENTIFIER = re.compile(r"[a-f0-9]{32}\Z")
SHA256 = re.compile(r"[a-f0-9]{64}\Z")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_relative(value):
    path = PurePosixPath(str(value))
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("Only normalized relative POSIX paths are allowed")
    if "\\" in str(value) or "\x00" in str(value):
        raise ValueError("Invalid path")
    return path.as_posix()


def add_file(root, selected, missing, relative, required=True):
    relative = safe_relative(relative)
    path = root / Path(*PurePosixPath(relative).parts)
    if not path.exists():
        if required:
            missing.append(relative)
        return
    if path.is_symlink() or not path.is_file():
        raise ValueError("Upload paths must be regular, non-symlink files: " + relative)
    selected[relative] = path


def add_tree(root, selected, missing, relative, required=True):
    relative = safe_relative(relative)
    directory = root / Path(*PurePosixPath(relative).parts)
    if not directory.exists():
        if required:
            missing.append(relative + "/")
        return
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Upload trees must be ordinary directories: " + relative)
    for path in sorted(directory.rglob("*")):
        rel = path.relative_to(root).as_posix()
        if any(part in EXCLUDED_PARTS for part in PurePosixPath(rel).parts):
            continue
        if path.is_symlink():
            raise ValueError("Symlinks are not included in uploads: " + rel)
        if path.is_file():
            selected[rel] = path


def collect_files(root):
    """Return selected path mapping and missing required model/source paths."""
    root = Path(root).resolve()
    selected, missing = {}, []
    for relative in CODE_FILES + SCRIPT_FILES:
        add_file(root, selected, missing, relative)

    for directory in ALLOWED_SOURCE_DIRS:
        add_tree(root, selected, missing, directory)
    add_file(root, selected, missing, "deploy/requirements-runtime.txt")
    add_file(root, selected, missing, "deploy/requirements-web.txt")
    add_file(root, selected, missing, "deploy/vlm_host.example.json")
    add_file(root, selected, missing, "provenance/source.json")

    configs = []
    for config_name in ("config_single.json", "config_two.json"):
        try:
            configs.append(json.loads((root / config_name).read_text()))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("Cannot read runtime config: " + config_name) from error
    for config in configs:
        for camera in config.get("cameras", []):
            detector = camera.get("detector", {})
            options = detector.get("options", {})
            for model in options.get("models", []):
                add_file(root, selected, missing, model.get("checkpoint", ""))
            classifier = camera.get("classifier", {})
            bundle = safe_relative(classifier.get("bundle", ""))
            add_file(root, selected, missing, bundle + "/manifest.json")
            bundle_path = root / Path(*PurePosixPath(bundle).parts)
            manifest_path = bundle_path / "manifest.json"
            if manifest_path.is_file():
                try:
                    manifest = json.loads(manifest_path.read_text())
                except json.JSONDecodeError as error:
                    raise ValueError("Invalid model bundle manifest: " + bundle) from error
                for entry in manifest.get("files", []):
                    member = safe_relative(entry.get("path", ""))
                    add_file(root, selected, missing, bundle + "/" + member)
            person = options.get("linq_person")
            if person:
                model_name = safe_relative(person.get("model", ""))
                models_root = safe_relative(person.get("models_root", "models/linq"))
                model_dir = models_root + "/" + model_name
                report_relative = model_dir + "/report.json"
                add_file(root, selected, missing, report_relative)
                add_file(root, selected, missing, models_root + "/decoder.py")
                report_path = root / Path(*PurePosixPath(report_relative).parts)
                if report_path.is_file():
                    try:
                        report = json.loads(report_path.read_text())
                    except json.JSONDecodeError as error:
                        raise ValueError("Invalid LinQ compile report: " + report_relative) from error
                    programs = [item.get("tpu_program", {}).get("path") for item in report.get("models", [])
                                if isinstance(item, dict) and item.get("name") == model_name and item.get("status") == "compiled"]
                    if len(programs) != 1:
                        raise ValueError("LinQ report must identify exactly one compiled program: " + report_relative)
                    program = safe_relative(programs[0])
                    add_file(root, selected, missing, model_dir + "/" + program)

    # Keep the manifest and source tree itself out of local caches, private data, and test fixtures.
    filtered = {}
    for relative, path in selected.items():
        if relative == 'service/samples.local.json':
            continue
        parts = PurePosixPath(relative).parts
        if any(part in EXCLUDED_PARTS for part in parts):
            continue
        if any(relative == prefix or relative.startswith(prefix + "/") for prefix in EXCLUDED_MEDIA_DIRS):
            continue
        if path.suffix.lower() in {".ipynb", ".log", ".jsonl", ".pyc", ".pyo"}:
            continue
        if relative.startswith("service/tests/") or relative.startswith("deploy/tests/") or relative.startswith("demo/tests/"):
            continue
        filtered[relative] = path
    return filtered, sorted(set(missing))


def archive_files(files, destination):
    """Stream a deterministic tar archive to disk and return its manifest and digest."""
    expected_payload = sum(path.stat().st_size for path in files.values())
    if expected_payload > MAX_ARCHIVE_BYTES:
        raise ValueError("Upload archive exceeds the configured safety limit")
    entries = []
    with tarfile.open(destination, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for relative, path in sorted(files.items()):
            relative = safe_relative(relative)
            before = sha256_file(path)
            size = path.stat().st_size
            info = tarfile.TarInfo(relative)
            info.size = size
            info.mode = 0o644
            info.mtime = 0
            with path.open("rb") as stream:
                archive.addfile(info, stream)
            if sha256_file(path) != before:
                raise ValueError("Source file changed while building archive: " + relative)
            entries.append({"path": relative, "bytes": size, "sha256": before, "model": relative.startswith("models/")})
    archive_size = Path(destination).stat().st_size
    if archive_size > MAX_ARCHIVE_BYTES:
        raise ValueError("Upload archive exceeds the configured safety limit")
    return entries, sha256_file(destination), archive_size


def remote_apply_code(payload):
    """Generate bounded, self-contained code for the LinQ notebook kernel."""
    serialized = json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
    return REMOTE_APPLY_TEMPLATE.replace("__PAYLOAD_JSON__", repr(serialized))


REMOTE_APPLY_TEMPLATE = r'''import hashlib,json,os,shutil,tarfile,tempfile,uuid
from pathlib import Path,PurePosixPath
payload=json.loads(__PAYLOAD_JSON__)
root=Path.home()/'work'/'mosprom_hackathon_linq'
root.mkdir(parents=True,exist_ok=True)
if root.is_symlink() or not root.resolve().is_relative_to(Path.home().resolve()):raise ValueError('Unsafe owned workspace root')
upload_name=payload['upload_name']
if not __import__('re').fullmatch(r'provision_[a-f0-9]{32}',upload_name):raise ValueError('Invalid owned upload directory')
upload=root/upload_name
if upload.is_symlink() or not upload.is_dir() or upload.resolve().parent!=root.resolve():raise ValueError('Owned upload directory missing or unsafe')
parts=payload['parts'];entries=payload['entries'];expected_total=payload['archive_bytes']
if expected_total>4294967296 or sum(item['bytes'] for item in parts)!=expected_total:raise ValueError('Archive exceeds configured limit or has invalid part sizes')
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for block in iter(lambda:stream.read(1048576),b''):h.update(block)
 return h.hexdigest()
stage=Path(tempfile.mkdtemp(prefix='.provision-stage-',dir=root))
archive_path=stage/'package.tar'
whole=hashlib.sha256();total=0
with archive_path.open('xb') as output:
 for item in parts:
  if not __import__('re').fullmatch(r'part-[0-9]{6}',item['name']):raise ValueError('Invalid upload part name')
  path=upload/'parts'/item['name']
  if path.is_symlink() or not path.is_file():raise ValueError('Upload part missing or unsafe')
  part_hash=hashlib.sha256();part_size=0
  with path.open('rb') as stream:
   for block in iter(lambda:stream.read(1048576),b''):
    part_hash.update(block);whole.update(block);output.write(block);part_size+=len(block);total+=len(block)
  if part_size!=item['bytes'] or part_hash.hexdigest()!=item['sha256']:raise ValueError('Upload part integrity check failed')
 output.flush();os.fsync(output.fileno())
if total!=expected_total or whole.hexdigest()!=payload['archive_sha256']:raise ValueError('Archive integrity check failed')
expected={item['path']:item for item in entries}
if len(expected)!=len(entries):raise ValueError('Duplicate paths in upload manifest')
seen=set()
with tarfile.open(archive_path,'r:') as bundle:
 members=bundle.getmembers()
 if len(members)!=len(entries):raise ValueError('Archive member count differs from manifest')
 for member in members:
  name=PurePosixPath(member.name)
  if name.is_absolute() or not name.parts or any(part in ('','.','..') for part in name.parts) or '\\' in member.name or not member.isfile():raise ValueError('Unsafe archive member')
  relative=name.as_posix()
  if relative not in expected or relative in seen:raise ValueError('Archive contains an unexpected or duplicate member')
  item=expected[relative]
  source=bundle.extractfile(member)
  if source is None:raise ValueError('Archive member unreadable')
  destination=stage.joinpath(*name.parts)
  destination.parent.mkdir(parents=True,exist_ok=True)
  h=hashlib.sha256();size=0
  with destination.open('xb') as output:
   for block in iter(lambda:source.read(1048576),b''):
    h.update(block);output.write(block);size+=len(block)
  if size!=item['bytes'] or h.hexdigest()!=item['sha256']:raise ValueError('File integrity check failed: '+relative)
  seen.add(relative)
if seen!=set(expected):raise ValueError('Archive is incomplete')
def destination_for(relative):
 name=PurePosixPath(relative)
 if name.is_absolute() or not name.parts or any(part in ('','.','..') for part in name.parts) or '\\' in relative:raise ValueError('Unsafe destination path')
 destination=root.joinpath(*name.parts)
 current=root
 for part in name.parts[:-1]:
  current=current/part
  if current.exists() and (current.is_symlink() or not current.is_dir()):raise ValueError('Refusing unsafe destination directory')
 resolved=destination.parent.resolve()
 if not resolved.is_relative_to(root.resolve()):raise ValueError('Destination escaped owned workspace')
 return destination
# Preflight every destination before replacing any owned code file.
actions=[]
for relative,item in expected.items():
 destination=destination_for(relative)
 source=stage.joinpath(*PurePosixPath(relative).parts)
 if destination.is_symlink():raise ValueError('Refusing symlink destination: '+relative)
 if destination.exists() and not destination.is_file():raise ValueError('Destination is not a regular file: '+relative)
 if item['model'] and destination.is_file() and digest(destination)!=item['sha256']:raise ValueError('Existing model differs; refusing replacement: '+relative)
 actions.append((relative,item,source,destination))
written=[];unchanged=[]
for relative,item,source,destination in actions:
 destination.parent.mkdir(parents=True,exist_ok=True)
 if destination.is_file() and digest(destination)==item['sha256']:
  unchanged.append(relative);continue
 temporary=destination.with_name('.'+destination.name+'.provision-'+uuid.uuid4().hex+'.tmp')
 try:
  with source.open('rb') as incoming,temporary.open('xb') as outgoing:
   shutil.copyfileobj(incoming,outgoing);outgoing.flush();os.fsync(outgoing.fileno())
  if digest(temporary)!=item['sha256']:raise ValueError('Staged copy digest mismatch: '+relative)
  if item['model']:
   try:os.link(temporary,destination)
   except FileExistsError:
    if not destination.is_file() or digest(destination)!=item['sha256']:raise ValueError('Concurrent model destination differs: '+relative)
   temporary.unlink(missing_ok=True)
  else:os.replace(temporary,destination)
  written.append(relative)
 finally:temporary.unlink(missing_ok=True)
print(json.dumps({'status':'uploaded','files_written':len(written),'files_unchanged':len(unchanged),'archive_sha256':payload['archive_sha256'],'model_execution':False,'hardware_tested':False},separators=(',',':')))
shutil.rmtree(stage,ignore_errors=True)
shutil.rmtree(upload,ignore_errors=True)
'''


def checked_execute(client, code, timeout=600):
    result = linq_client.execute(client, code, timeout=timeout)
    if result.get("status") != "completed" or result.get("errors"):
        names = [item.get("ename", "RemoteError") for item in result.get("errors", [])]
        raise RuntimeError("LinQ command failed: " + ", ".join(names or [result.get("status", "unknown")]))
    return result


def upload_archive(client, archive_path, entries, archive_sha, chunk_size=CHUNK_SIZE):
    if not isinstance(chunk_size, int) or chunk_size < 1024 or chunk_size > CHUNK_SIZE:
        raise ValueError("Chunk size must be between 1 KiB and 4 MiB")
    archive_path = Path(archive_path)
    provision_id = uuid.uuid4().hex
    upload_name = "provision_" + provision_id
    parts, offset = [], 0
    whole = hashlib.sha256()
    with archive_path.open("rb") as source:
        index = 0
        while chunk := source.read(chunk_size):
            whole.update(chunk)
            name = "part-" + str(index).zfill(6)
            descriptor, temporary = tempfile.mkstemp(prefix="linq-provision-", suffix=".part")
            try:
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(chunk)
                linq_client.upload(client, temporary, "mosprom_hackathon_linq/" + upload_name + "/parts/" + name)
            finally:
                Path(temporary).unlink(missing_ok=True)
            parts.append({"name": name, "bytes": len(chunk), "sha256": hashlib.sha256(chunk).hexdigest()})
            offset += len(chunk)
            index += 1
    if offset != archive_path.stat().st_size or whole.hexdigest() != archive_sha:
        raise ValueError("Archive upload size accounting failed")
    payload = {"upload_name": upload_name, "parts": parts, "entries": entries,
               "archive_bytes": offset, "archive_sha256": archive_sha}
    result = checked_execute(client, remote_apply_code(payload), timeout=1800)
    lines = [line for line in result.get("stdout", "").splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("Remote provision returned no receipt")
    receipt = json.loads(lines[-1])
    if receipt.get("status") != "uploaded" or receipt.get("archive_sha256") != archive_sha:
        raise ValueError("Remote provision receipt failed validation")
    return receipt


def remote_check(client):
    response = linq_client.request(client, "GET", "api/status", timeout=15)
    code = r'''import json,urllib.request,urllib.error
from pathlib import Path
root=Path.home()/'work'/'mosprom_hackathon_linq'
def ready(url):
 try:
  with urllib.request.urlopen(url,timeout=2) as response:return json.load(response).get('status')=='ok'
 except (urllib.error.URLError,TimeoutError,ValueError):return False
def bundle_valid():
 try:
  import hashlib
  directory=root/'models/convnext';manifest=json.loads((directory/'manifest.json').read_text())
  if manifest.get('status')!='onnx_cpu_parity_passed':return False
  for item in manifest['files']:
   path=(directory/item['path']).resolve()
   if not path.is_relative_to(directory.resolve()) or not path.is_file():return False
   digest=hashlib.sha256()
   with path.open('rb') as stream:
    for block in iter(lambda:stream.read(1048576),b''):digest.update(block)
   h=digest.hexdigest()
   if h!=item['sha256']:return False
  return True
 except (OSError,ValueError,KeyError,TypeError):return False
def linq_program_present():
 try:
  import hashlib
  report=json.loads((root/'models/linq/rfdetr/report.json').read_text())
  rows=[item for item in report.get('models',[]) if item.get('name')=='rfdetr' and item.get('status')=='compiled']
  if len(rows)!=1:return False
  spec=rows[0]['tpu_program'];path=(root/'models/linq/rfdetr'/spec['path']).resolve()
  if not path.is_relative_to((root/'models/linq/rfdetr').resolve()) or not path.is_file():return False
  digest=hashlib.sha256()
  with path.open('rb') as stream:
   for block in iter(lambda:stream.read(1048576),b''):digest.update(block)
  return digest.hexdigest()==spec['sha256']
 except (OSError,ValueError,KeyError,TypeError):return False
def configured_models_valid():
 try:
  import hashlib
  config=json.loads((root/'config_single.json').read_text())
  result={}
  for spec in config['cameras'][0]['detector']['options']['models']:
   path=(root/spec['checkpoint']).resolve();expected=spec['sha256']
   if not path.is_relative_to(root.resolve()) or not path.is_file():result[spec['checkpoint']]=False;continue
   digest=hashlib.sha256()
   with path.open('rb') as stream:
    for block in iter(lambda:stream.read(1048576),b''):digest.update(block)
   result[spec['checkpoint']]=digest.hexdigest()==expected
  return result
 except (OSError,ValueError,KeyError,TypeError):return {}
required=['models/linq/decoder.py','models/qwen_small/Qwen3.5-0.8B-Q4_K_M.gguf','models/qwen_small/mmproj-Qwen3.5-0.8B-BF16.gguf','bin/llama/llama-server']
print(json.dumps({'linq_workspace_present':root.is_dir(),'required_assets_present':{item:(root/item).is_file() for item in required},'configured_onnx_valid':configured_models_valid(),'convnext_bundle_valid':bundle_valid(),'linq_program_sha_valid':linq_program_present(),'qwen_backend_ready':ready('http://127.0.0.1:18776/health'),'model_execution':False,'hardware_tested':False},separators=(',',':')))
'''
    result = checked_execute(client, code, timeout=120)
    lines = [line for line in result.get("stdout", "").splitlines() if line.strip()]
    return {"linq_api_status": response.status_code, "remote": json.loads(lines[-1]) if lines else None}


def remote_setup_code(remote_python=None):
    requested_python = repr(remote_python)
    code = r'''import glob,json,os,shutil,subprocess,sys
from pathlib import Path
root=Path.home()/'work'/'mosprom_hackathon_linq'
if not root.is_dir() or root.is_symlink():raise ValueError('Provisioned workspace is absent or unsafe')
requested=__REQUESTED_PYTHON__
if requested:
 requested=str(Path(requested).expanduser())
candidates=[]
if requested:
 candidates.append(requested)
else:
 detected=shutil.which('python3.12')
 if detected:candidates.append(detected)
 candidates.append(str(root/'.venv'/'bin'/'python'))
 home=Path.home()
 candidates.extend([str(home/'.local'/'bin'/'python3.12'),str(home/'miniforge3'/'bin'/'python3.12'),str(home/'mambaforge'/'bin'/'python3.12')])
 candidates.extend(sorted(glob.glob(str(home/'.pyenv'/'versions'/'3.12*'/'bin'/'python'))))
 candidates=list(dict.fromkeys(candidates))
python=None
for candidate in candidates:
 try:
  probe=subprocess.run([candidate,'-c','import sys; print("%d.%d" % sys.version_info[:2])'],capture_output=True,text=True,timeout=15)
  if probe.returncode==0 and probe.stdout.strip()=='3.12':
   python=candidate;break
 except (OSError,subprocess.TimeoutExpired):
  pass
if python is None:
 print(json.dumps({'status':'setup_incomplete','stages':[{'stage':'python_selection','status':'failed','detail':'python312_unavailable'}],'model_execution':False,'hardware_tested':False},separators=(',',':')))
 raise SystemExit(0)
stages=[('bootstrap',[python,'scripts/bootstrap_linq_prototype.py']),('qwen_prefetch',[python,'scripts/prefetch_qwen_gguf.py','--root','.'])]
results=[]
for name,command in stages:
 try:
  result=subprocess.run(command,cwd=root,capture_output=True,text=True,timeout=7200)
  if result.returncode:
   results.append({'stage':name,'status':'failed','returncode':result.returncode,'detail':'command_failed'})
   break
  results.append({'stage':name,'status':'completed','returncode':0})
 except subprocess.TimeoutExpired:
  results.append({'stage':name,'status':'failed','detail':'command_timed_out'})
  break
 except OSError:
  results.append({'stage':name,'status':'failed','detail':'command_unavailable'})
  break
print(json.dumps({'status':'setup_complete' if len(results)==len(stages) and all(item['status']=='completed' for item in results) else 'setup_incomplete','python_version':'3.12','stages':results,'model_execution':False,'hardware_tested':False},separators=(',',':')))
'''
    return code.replace("__REQUESTED_PYTHON__", requested_python)


def remote_setup(client, remote_python=None):
    code = remote_setup_code(remote_python)
    result = checked_execute(client, code, timeout=16000)
    lines = [line for line in result.get("stdout", "").splitlines() if line.strip()]
    receipt = json.loads(lines[-1]) if lines else None
    if not isinstance(receipt, dict):
        raise RuntimeError("Remote setup returned no status receipt")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description="Safely provision this demo into the authenticated LinQ workspace.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="List the allowlisted paths; do not connect to LinQ.")
    mode.add_argument("--check", action="store_true", help="Check LinQ access and remote asset/service status; do not run inference.")
    parser.add_argument("--setup", action="store_true", help="After upload, run remote dependency bootstrap and Qwen prefetch.")
    parser.add_argument("--root", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--session-file", default=".secrets/linq_jupyter_session.json")
    parser.add_argument("--capabilities-file", default=".secrets/linq_jupyter_capabilities.json")
    parser.add_argument("--remote-python", help="Explicit Python 3.12 interpreter path on LinQ (default: discover one).")
    args = parser.parse_args(argv)
    if args.setup and (args.dry_run or args.check):
        parser.error("--setup can only be used with the upload operation")
    if args.remote_python and not args.setup:
        parser.error("--remote-python can only be used with --setup")
    root = Path(args.root).resolve() if args.root else ROOT
    if args.check:
        client = linq_client.connect(args.session_file, args.capabilities_file)
        print(json.dumps(remote_check(client), ensure_ascii=False, indent=2))
        return 0
    files, missing = collect_files(root)
    listing = [{"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
               for relative, path in sorted(files.items())]
    if args.dry_run:
        print(json.dumps({"status": "ready" if not missing else "missing_required_files",
                          "files": listing, "missing": missing,
                          "model_execution": False, "hardware_tested": False}, ensure_ascii=False, indent=2))
        return 0 if not missing else 2
    if missing:
        raise SystemExit("Missing required source/model files: " + ", ".join(missing))
    with tempfile.TemporaryDirectory(prefix="linq-provision-") as temporary:
        archive_path = Path(temporary) / "package.tar"
        entries, archive_sha, _archive_bytes = archive_files(files, archive_path)
        client = linq_client.connect(args.session_file, args.capabilities_file)
        receipt = upload_archive(client, archive_path, entries, archive_sha)
    result = {"upload": receipt}
    if args.setup:
        result["setup"] = remote_setup(client, args.remote_python)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if args.setup and result["setup"].get("status") != "setup_complete" else 0


if __name__ == "__main__":
    raise SystemExit(main())
