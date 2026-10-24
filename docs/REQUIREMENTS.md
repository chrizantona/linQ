# Requirements matrix

This matrix follows the authoritative LinQ case brief, printed page 13. “Implemented” describes a supported path with recorded execution evidence; “partial” means a material acceptance condition remains unmet or unmeasured. Run-level evidence is indexed in [`provenance/service-features.json`](../provenance/service-features.json) and detailed in [`provenance/linq-verification.json`](../provenance/linq-verification.json).

## Mandatory requirements

| Requirement from the brief | Current implementation | Status and remaining limit |
| --- | --- | --- |
| Ingest camera video streams **or** themed image sets from a place (stop-frame simulation) | The API accepts uploaded recorded video. `POST /api/image-jobs` accepts 2–24 JPEG/PNG files, validates and compiles them at one synthetic frame per second, preserves source hashes/indices, and uses the generated MP4 as inference transport. A three-image set was run on LinQ. | **Implemented for recorded uploads and image sets.** Direct live camera/RTSP ingestion is not established. Still-image ordering is synthetic and does not represent motion or measured capture time; motion-dependent rules are rejected for this input. |
| Neural-network recognition of safety violations | RF-DETR person detection runs on the LinQ TPU service; YOLO26m phone and RF-DETR cigarette object detection run on LinQ CPU. ConvNeXt and six trained heads provide clip-level classification. | **Partial.** Object boxes identify candidate components, not phone use or smoking. Small or occluded objects can be missed, and clip labels do not localize a violation. There is no representative precision/recall or incident-accuracy benchmark. |
| Timely GUI alert containing frame/video, camera number, time, and violation type | Candidate events can include source video/evidence frame, camera, relative trigger time, and event/rule type. Candidates are emitted after detector evidence is available, before optional Qwen review or final rendering. | **Partial.** This is batch-job review, not a demonstrated push notification to a responsible person. No end-to-end response-time target or SLA has been established. |
| Event journal filterable by type, camera, zone, and time | The event API and UI journal support type, camera, zone, and relative start/end time filters. These filters and rejection of invalid filter inputs were verified. | **Implemented for the verified event records.** The time fields are relative to the processed job; there is no claim of synchronized wall-clock camera time. |
| Suppress duplicate alerts: one violation, one alert | Temporal hysteresis groups repeated triggers into one camera/rule episode; the positive feature runs recorded one episode per continuous positive clip. | **Partial.** Episode identity is camera plus rule, not a confirmed person or incident. Separate people can be merged, and duplicate-suppression quality has not been evaluated on a representative set. |
| Run inference on LinQ accelerators and measure speed | RF-DETR person detection uses LinQ's TPU service. Phone/cigarette models, ConvNeXt and its six heads, and Qwen3.5-0.8B Q4 run on LinQ CPU. Run receipts include per-stage timing. | **Implemented for execution; partial for the performance requirement.** Timing was recorded for selected jobs, but no benchmark, few-seconds target, real-time/FPS guarantee, or SLA has been established. |

## Desirable requirements

| Requirement from the brief | Current implementation | Status and remaining limit |
| --- | --- | --- |
| Per-camera rules and polygon zones | Camera-specific polygon zones are saved with job evidence; the person footpoint is checked against the relevant camera's polygon when generating a candidate. A zone-triggered LinQ job was verified. | **Partial.** Zone-trigger behavior is verified, but it does not establish a violation or person identity. Broader per-camera rule coverage has not been evaluated. |
| Dashboard counts by event type, camera, time/day/shift | The UI provides job, source, review, and time-window analytics. | **Partial.** It is not a validated incident dashboard grouped by violation, zone, or shift; counts can describe processed sources rather than confirmed incidents. |
| Operator confirm/reject for evaluation or training | Review decisions can be saved and exported with event/camera context. | **Implemented for review and export; partial for the broader goal.** No operator labels were created by the verification runs, and no training feedback loop or evaluation study is evidenced. |
| Several simultaneous camera streams | A paired prerecorded-video job was processed with camera-specific frame isolation; CV-only events from both cameras were retained. | **Partial.** This verifies a bounded two-camera batch job, not simultaneous live streams, cross-camera synchronization, concurrency, or throughput. |
| Add scenarios without major rewrite | Model heads, detector profiles, and rule stages are configurable. | **Partial.** New model semantics and their UI/catalog mapping can still require coordinated code and configuration changes. |
| Scale to the described multi-camera industrial area | The workflow is upload/job based. | **Not established.** There is no camera-stream worker pool or load/capacity evidence for the described area or camera count. |

## Evidence and interpretation limits

The provenance files bind results to recorded inputs and model/configuration hashes. A successful model call establishes execution only; it does not establish model quality. The listed feature cases include selected positive clips and negative controls and are not a representative test set.

- A person, phone, or cigarette box is a localized object candidate. A phone box does not prove phone use; a cigarette box does not prove smoking.
- Clip-head labels are clip-level classifications, not event bounding boxes or confirmed violations.
- Geometric rules and camera/rule episodes create candidate events. They are not person-level incident identities.
- Qwen output without supporting object evidence is gated as unknown. Model review is not ground truth.
- Still-image timestamps provide synthetic ordering for stop-frame simulation, not captured temporal evidence.
- Operator review is stored separately from model output. The current verification record contains no human labels.

There is no measured detection-quality benchmark, validated production latency, live-stream guarantee, or demonstrated hundred-camera capacity. Do not infer these from successful model calls or selected completed jobs.
