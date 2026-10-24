# Setup

The local host runs the web UI and orchestration. The authenticated LinQ workspace runs the analysis pipeline and its models. This repository does not include a LinQ account, TPU connection notebook, VPN profile, or API credentials. Keep secrets under the ignored `.secrets/` directory and never put them in command arguments, logs, or Git.

## Requirements

- Python 3.12 or 3.13 on the UI host, with `deploy/requirements-web.txt`. Local verification used Python 3.13; the provisioned LinQ runtime uses Python 3.12. The web requirements include Pillow for still-image validation/preparation.
- Docker Engine/Desktop with `/dev/net/tun` available if using the repository's isolated VPN helper, plus an authorized self-contained LinQ OpenVPN profile.
- An authorized LinQ/Jupyter account and the TPU connection notebook supplied by LinQ.
- Linux x86_64 with Python 3.12 on LinQ for the provisioned pipeline verified in the recorded runs. Other remote platforms are not verified.
- `ffmpeg` on the LinQ `PATH` for native-video pipeline steps. `ffmpeg` is also needed locally to prepare image-set uploads or install Pexels examples.
- An authorized model bundle matching the local [`assets.json`](../assets.json). The v2 bundle currently has no public download URL; obtain it through the authorized release channel.
- Authorized LinQ network access to fetch the configured Qwen model/runtime during bootstrap.

## Install model assets

Use the v2 release archive and the manifest supplied with this source tree. `assets.json` pins each file and the complete archive by byte count and SHA-256. The local installer verifies these values before installing and refuses to overwrite different existing files.

```sh
python3 scripts/install_assets.py --bundle /path/to/linq-models-v2.tar.gz
```

If the release channel provides the extracted files instead:

```sh
python3 scripts/install_assets.py --assets-dir /path/to/release-assets
```

Do not edit manifest hashes to accommodate a different artifact. The v2 manifest adds an RF-DETR cigarette detector trained on CigDet; [third-party notices](../THIRD_PARTY_NOTICES.md) explain that the software license does not itself grant model-weight or training-data redistribution rights.

## Optional local demo examples

The initial sample catalog is empty. To download and locally trim the allowlisted Pexels examples in [`examples/media_sources.json`](../examples/media_sources.json), install `ffmpeg` locally and run:

```sh
python3 scripts/install_demo_examples.py
```

The installer checks the full downloaded-source SHA-256, creates six-second excerpts and posters, and writes ignored files under `service/assets/demo-local/` plus `service/samples.local.json`. It validates redirect targets before following them. These generated files are local test/demo media; they are not baked into Git or into the model archive. Review the linked Pexels license and each clip's attribution before use. This installer does not assess whether a particular use is permitted.

The image-set API accepts 2–24 JPEG/PNG stills, compiles them at one frame per second, and records `synthetic_time: true` plus each original image's hash and index. It represents stop-frame simulation, not captured motion or measured camera timing. A LinQ run of an ordered three-image set verified source hashes and indices; motion-dependent analysis remains unsupported for this input type.

## Prepare the local UI host

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r deploy/requirements-web.txt
```

### Connect to LinQ

The Docker helper uses a separate container, a read-only profile mount, a private container tmpfs for the authentication file, `NET_ADMIN`, and `/dev/net/tun`. It publishes the forward only on `127.0.0.1:18184`; it does not use host networking, privileged mode, or alter the host VPN. Use a self-contained profile with certificate/key material inline. The helper reports generic status and does not print credentials or OpenVPN logs.

Start with an authorized profile; the password is requested interactively:

```sh
.venv/bin/python scripts/connect_linq_docker.py start --profile /path/to/authorized-linq.ovpn
.venv/bin/python scripts/connect_linq_docker.py status
```

Or provide a private two-line auth file with restrictive permissions:

```sh
chmod 600 /path/to/linq-auth.txt
.venv/bin/python scripts/connect_linq_docker.py start --profile /path/to/authorized-linq.ovpn --auth-file /path/to/linq-auth.txt
```

Stop only the helper's container with:

```sh
.venv/bin/python scripts/connect_linq_docker.py stop
```

A healthy tunnel does not establish that Jupyter authentication or inference works. Configure the notebook session over the loopback forward. The password is entered interactively:

```sh
.venv/bin/python -m deploy.configure_access --base-url http://hackathon.lin-q.ru:18184/ --address 127.0.0.1 --websocket-origin http://hackathon.lin-q.ru/
```

The helper verifies the Jupyter API before saving private session/capability files under `.secrets/` and refuses cross-origin redirects. Keep those files private.

## Provision LinQ

Inspect the exact allowlist before uploading:

```sh
.venv/bin/python scripts/provision_linq.py --dry-run
```

After confirming the intended source list and installing the authorized v2 model bundle locally, provision the workspace:

```sh
.venv/bin/python scripts/provision_linq.py --setup
```

Provisioning uploads allowlisted source files in bounded chunks and verifies archive/part hashes. It installs dependencies and prefetches Qwen assets, but does not run a detection job. Start the local LinQ-side Qwen service in a Jupyter terminal when needed:

```sh
cd ~/work/mosprom_hackathon_linq
.venv/bin/python scripts/linq_prototype.py start-vlm
.venv/bin/python scripts/linq_prototype.py status
```

Use `--check` to inspect configured remote prerequisites without running inference:

```sh
.venv/bin/python scripts/provision_linq.py --check
```

Provisioning does not upload local Jupyter session credentials. It may replace its explicitly allowlisted application files, but does not delete unlisted models, jobs, tokens, or working directories.

## Start the local interface

```sh
.venv/bin/python -m deploy.demo_web --host 127.0.0.1 --port 4184
```

Open `http://127.0.0.1:4184/`. The interface is not an authentication boundary. Keep it on loopback unless you add an authenticated gateway. Upload only media you are authorized to process. The recorded LinQ evidence covers a paired two-camera CV-only job, an ordered image set, zone-triggered candidates, and filters by type, camera, zone, and relative time. It does not establish live-stream operation, synchronized multi-camera inference, accuracy, or production throughput; see [Requirements](REQUIREMENTS.md) and [QA](../QA.md).

## Local checks

```sh
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q
npm test --prefix service
```

Local tests do not contact LinQ. Current remote execution evidence and its limits are recorded in [QA.md](../QA.md) and the linked provenance files. LinQ verification applies only to the recorded inputs and artifact/configuration hashes.
