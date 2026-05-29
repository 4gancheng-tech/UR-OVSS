# UR-OVSS MVP

Minimal runnable implementation of Uncertainty-Routed Multi-Expert Open-Vocabulary Semantic Segmentation.

This repository keeps DINO purity as a deterministic fallback expert. Semantic and mask experts can run in selectable modes:

- `fallback`: deterministic dense proxy logits, no model download
- `clip`: optional real CLIP region-crop scoring through `open_clip`
- `--mask-backend fallback`: deterministic class-agnostic fallback masks, the default option
- `--mask-backend sam`: optional SAM/MobileSAM class-agnostic masks from a user-provided checkpoint
- `--feature-backend fallback`: deterministic patch-level proxy features for region purity
- `--feature-backend dinov2`: optional DINOv2 dense patch features for region purity
- text expert: fixed positive and negative prompt templates, no external LLM

DINOv2 is used only for region purity / spatial uncertainty. It is not used for class prediction.

The current `clip` backend is region crop-level CLIP scoring. It is not dense CLIP or ClearCLIP logits.

## Run

```bash
pip install -r requirements.txt
python infer_ur_ovss.py --image path/to/image.jpg --classes "cat,dog,person,car" --output outputs/demo.png
```

To use the optional CLIP semantic backend:

```bash
pip install -r requirements-clip.txt
python infer_ur_ovss.py --image path/to/image.jpg --classes "cat,dog,person,car" --output outputs/demo.png --semantic-backend clip
```

CLIP weights are loaded by `open_clip` into its normal user cache, not into this repository. If `open_clip`, `torch`, the model weights, or network access are unavailable, the CLI exits with a clear semantic backend error.

To use the optional SAM mask backend:

```bash
pip install -r requirements-sam.txt
python infer_ur_ovss.py --image path/to/image.jpg --classes "cat,dog,person,car" --output outputs/demo.png --mask-backend sam --sam-checkpoint path/to/sam_vit_b.pth --max-masks 100
```

SAM/MobileSAM weights must be provided through `--sam-checkpoint`; they are not downloaded by this project and should not be committed to the repository. If `segment-anything` / `mobile-sam`, the checkpoint path, or model initialization is unavailable, the CLI exits with a clear mask backend error.

To use the optional DINOv2 feature backend:

```bash
pip install -r requirements-dino.txt
python infer_ur_ovss.py --image path/to/image.jpg --classes "cat,dog,person,car" --output outputs/demo.png --feature-backend dinov2 --dinov2-model facebook/dinov2-small
```

DINOv2 weights are loaded by `transformers` into its normal user cache, not into this repository. If `transformers`, `torch`, model weights, or network/cache access are unavailable, the CLI exits with a clear feature backend error.

## Pascal VOC Evaluation

Pascal VOC 2012 semantic segmentation evaluation is available through `eval_pascal_voc.py`. It expects an existing VOC2012 directory and does not download the dataset:

```bash
python eval_pascal_voc.py --voc-root path/to/VOCdevkit/VOC2012 --split val --limit 10 --output-dir outputs/voc_eval
```

The evaluator reads image ids from `ImageSets/Segmentation/{split}.txt`, images from `JPEGImages`, and masks from `SegmentationClass`. It reports foreground-only mIoU over the 20 Pascal VOC classes and writes `metrics.json` to the output directory. Add `--save-vis` to keep per-image visualization PNGs; otherwise it avoids saving lots of demo images.

The default fallback backends are useful for validating the evaluation pipeline, but they are not representative of real segmentation performance.

## Real Backend VOC Smoke Test

Use `scripts/run_voc_real_smoke.ps1` to run a small Pascal VOC validation smoke test with the real optional backends enabled: CLIP semantic scoring, SAM masks, and DINOv2 purity features.

You need:

- Pascal VOC 2012 dataset at `VOCdevkit/VOC2012`
- A local SAM checkpoint such as `sam_vit_b.pth`
- Dependencies from `requirements-clip.txt`
- Dependencies from `requirements-sam.txt`
- Dependencies from `requirements-dino.txt`

Windows PowerShell example:

```powershell
$env:VOC_ROOT="C:\path\to\VOCdevkit\VOC2012"
$env:SAM_CHECKPOINT="C:\path\to\sam_vit_b.pth"
.\scripts\run_voc_real_smoke.ps1
```

Start with the default `LIMIT=1`. After that works, try `LIMIT=10`:

```powershell
$env:LIMIT="10"
$env:OUTPUT_DIR="outputs/voc_real_smoke_limit10"
.\scripts\run_voc_real_smoke.ps1
```

Full VOC val can be slow, especially with SAM and DINOv2, so a GPU is recommended. Do not commit `outputs`, Pascal VOC data, or SAM/model weights to the repository.

The command saves:

- `outputs/demo.png`: blended segmentation visualization
- `outputs/demo_labels.png`: colorized label map
- `outputs/demo_mask.npy`: integer label map, with `-1` for unassigned pixels
- `outputs/demo_confidence.npy`: winning region confidence per pixel
- `outputs/demo.json`: per-region debug records with routing decisions

Each `regions[]` entry in the JSON includes `region_id`, `predicted_label`, `semantic_margin`, `dino_variance`, `semantic_uncertain`, `spatial_uncertain`, `route_type`, and `confidence`.
It also includes `base_scores`, `positive_scores`, `negative_scores`, and `prompt_rescore_scores` for debugging the selected semantic backend.

No dataset is included in this repository.
