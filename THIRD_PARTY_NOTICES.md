# Third-party notices and model provenance

This file identifies upstream components known to be used by the current runtime. The accompanying license copies are under [`licenses/`](licenses/). Their presence preserves upstream notices; it is not legal advice and does not independently authorize redistribution of model weights, compiled programs, or data.

## Runtime and model components

| Component | Use in this project | Upstream terms / source |
| --- | --- | --- |
| Ultralytics YOLO-derived phone detector | ONNX CPU object detection; export derives from a trained YOLO checkpoint | Ultralytics AGPL-3.0; see [`licenses/AGPL-3.0.txt`](licenses/AGPL-3.0.txt). A commercial license may be required for uses incompatible with AGPL obligations. No such license is represented here. |
| RF-DETR | Person detector compiled for the LinQ TPU service | Apache-2.0 distribution license; see [`licenses/RFDETR-Apache-2.0.txt`](licenses/RFDETR-Apache-2.0.txt). Upstream project: [Roboflow RF-DETR](https://github.com/roboflow/rf-detr). |
| Qwen3.5 | Local CPU vision-language model, distributed as GGUF model and vision projector | Apache-2.0 model license; see [`licenses/QWEN-Apache-2.0.txt`](licenses/QWEN-Apache-2.0.txt). Model source: [Qwen on Hugging Face](https://huggingface.co/Qwen). Verify the exact model card and file provenance for the release artifact. |
| PyTorch / TorchVision ConvNeXt-Tiny ImageNet-1K weights | CPU feature encoder and preprocessing implementation | PyTorch and TorchVision code are BSD-3-Clause; see the upstream [PyTorch license](https://github.com/pytorch/pytorch/blob/main/LICENSE) and [TorchVision license](https://github.com/pytorch/vision/blob/main/LICENSE). The pretrained weights are sourced from TorchVision's ImageNet-1K weights; review the upstream weights metadata and applicable dataset terms before redistribution. |
| llama.cpp | Local CPU model serving process | MIT; see the upstream [llama.cpp license](https://github.com/ggml-org/llama.cpp/blob/master/LICENSE). The build used for a release must retain its upstream notices. |

The YOLO-derived phone model has provenance and a digest recorded by the release installer. The upstream FPI-Det project is [MIT licensed](https://github.com/KvCgRv/FPI-Det); the CigDet dataset source identifies its release under [CC BY 4.0](https://doi.org/10.17632/6hyrr8typ7.1). These project/dataset terms do not establish ownership or redistribution rights for any checkpoint or image. Retain the source-model and dataset attribution supplied with each artifact, and do not infer model-weight rights from source-code or dataset terms.

The additional `models/cigarette_rfdetr.onnx` artifact in the v2 model bundle was supplied through the source project's release assets. Its metadata identifies an RF-DETR export trained on CigDet and records source checkpoint SHA-256 `64270cd6b14eb2f3bdacf0c3d16490566c1728022a18afcb39fc2f84ed9d8a64`. The ONNX file digest was independently verified; the checkpoint-to-export lineage was not independently reproduced. RF-DETR's Apache-2.0 software license does not grant rights to model weights or training data. No separate model-weight or dataset redistribution grant is represented here; see the cited CigDet terms and obtain any additional permissions required before redistribution.

## Data and media

The optional demo catalog downloads examples from Pexels on demand according to [`examples/media_sources.json`](examples/media_sources.json). The manifest records each source page, creator, and source-file SHA-256. The official [Pexels license](https://www.pexels.com/license/) allows downloading and modifying its media subject to its restrictions; it prohibits redistributing or selling the media on other stock-photo/wallpaper platforms and imposes restrictions involving identifiable people and implied endorsement. Review the current license and the specific source before use. Attribution is not required by the linked standard license but is included in the app's source metadata. The installer trims examples locally and writes them to Git-ignored paths; this repository does not include the raw Pexels files or claim a broader right to redistribute them.

The public source package excludes original research clips, extracted dataset frames, and sample images. iSafetyBench is described by its publisher as CC BY-NC-SA 4.0 and composed of clips sourced from public video; see the [iSafetyBench dataset card](https://huggingface.co/datasets/raiyaanabdullah/isafety-bench) for provenance and terms. Do not redistribute those clips or derived frames without confirming the applicable permissions.

The release must preserve the exact notices accompanying the downloaded weights and compiled TPU program. Where rights or terms for a particular artifact are not established, distribute the installer and digest metadata only and require the operator to obtain that artifact through an authorized channel.

## Project license

The project license is provided in [LICENSE](LICENSE), retained from the destination repository. Third-party software, model weights, compiled programs, and datasets retain their own terms; the project license does not replace those terms.
