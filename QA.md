# Verification record

Date: 25 October 2026. Source project: [`cutroy/mosprom_hack`](provenance/source.json), revision `09ecbd3`.

## Local checks

- Python test suite: 192 passed, plus 52 passing subtests.
- Frontend test suite: 29 passed.
- The v2 model archive and all 13 contained model files were checked against `assets.json`; clean installation was verified. Archive size and digest are recorded in that manifest.
- The local demo-example installer was exercised against the five pinned Pexels sources. Downloaded source hashes and generated local outputs were checked.
- Requirements traceability was checked against page 13 of the supplied brief; all six mandatory rows are represented in [the matrix](docs/REQUIREMENTS.md).

These checks validate repository behavior and artifact integrity. Local tests do not establish LinQ hardware execution, model quality, or production readiness.

## Publication cleanup verification

The unused native `.pt` detector loader, local PyTPU ConvNeXt branch, and obsolete prototype notebook were removed. The supported detector backends are explicitly configured CPU ONNX adapters; person detection still uses the LinQ TPU service. Classifier TPU-verification fields remain explicitly false for schema compatibility.

After cleanup, phone use, smoking and the fall-head regression were each executed again on LinQ from a fresh validation source copy. All three jobs completed; target-object boxes and the CPU classifier path were checked. Receipts are in [`provenance/publication-runtime-validation.json`](provenance/publication-runtime-validation.json). The current remote source and model hashes are in [`provenance/publication-remote-source-verification.json`](provenance/publication-remote-source-verification.json).

The earlier `clean-source-code.json` and `remote-source-verification.json` describe the pre-publication revision `ff002536`; they remain historical evidence. The publication-specific files above describe the cleaned runtime. Aggregate execution records contain both revisions and must be read with their recorded source/model/configuration metadata.

## LinQ runs

The current LinQ evidence is indexed in [`provenance/service-features.json`](provenance/service-features.json), with per-job input, model/configuration digests, TPU request receipts, CPU model executions, outputs, and timing fields in [`provenance/linq-verification.json`](provenance/linq-verification.json). Read those records together: the index names the accepted feature cases, while the run record supplies the job-level evidence. They are scoped to their listed inputs and artifacts; new runs may add evidence or update the aggregate record.

The accepted feature set includes phone boxes on an office clip, cigarette boxes on a smoking-group clip, negative controls with no phone or cigarette boxes, a two-camera CV-only job retaining camera-isolated frames, an ordered three-image set preserving the source hashes and indices, a saved-zone-triggered candidate, and filtering by event type, camera, zone, and relative time. All six configured classifier heads were executed in the recorded LinQ work. The evidence also records candidate emission after real detector evidence and before optional Qwen review or final rendering. Raw Qwen action text without supporting object evidence is treated as unknown; negative controls do not produce action candidates.

These are selected engineering examples, not a representative evaluation set. Phone and cigarette misses remain possible, especially for small or occluded objects. Object boxes do not prove a use or smoking action. The recorded runs created no human operator labels, and no precision/recall or incident-level accuracy benchmark was performed.

False object boxes are also possible: the fall regression includes a false phone detection on a person's lower body. That raw output is retained in the run evidence. The positive and negative object-profile examples do not establish performance on other scenes.

[`provenance/progressive-candidates.json`](provenance/progressive-candidates.json) records actual early-candidate observations. [`provenance/clean-source-code.json`](provenance/clean-source-code.json) binds the fresh source copy to the packaged Python code and remote allowlist; [`provenance/remote-source-verification.json`](provenance/remote-source-verification.json) records the remote source/model hash check. Measured stage timings are in [`provenance/performance.json`](provenance/performance.json); they are selected engineering measurements, not a throughput benchmark or response-time guarantee.

Browser checks covered the monitor, event journal, camera list, analytics, still-image navigation and switching between the paired cameras. The raw video player no longer uses a rendered, annotated poster behind live frame annotations; the before/after DOM check is in [`provenance/video-poster-check.json`](provenance/video-poster-check.json).

## Execution boundary

The active pipeline uses LinQ TPU service inference for RF-DETR person detection; phone detection, the v2 RF-DETR cigarette detector, ConvNeXt and its six heads, and Qwen3.5-0.8B Q4 run on LinQ CPU. No local neural-network inference or external model API is configured. Exact artifact hashes and per-run device receipts are available in the provenance records. Local tests or configuration inspection alone are not evidence of remote execution.

## Scope not established

The prototype processes uploaded video jobs or ordered still-image sets. A still-image set is encoded as stop-frame simulation with synthetic one-second ordering; it is not captured motion or measured camera timing. The results do not establish detection accuracy, a real-time response target, an SLA, live RTSP ingestion, or capacity for 100 cameras. The interface presents candidates for operator review and is not a safety verdict system.

The local verification results above are independent of GitHub Actions. The Checks workflow runs on push; its current status is linked from README. These engineering checks do not establish production readiness or a model-quality benchmark.
