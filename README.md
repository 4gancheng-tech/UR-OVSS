# UR-OVSS MVP

Minimal runnable implementation of Uncertainty-Routed Multi-Expert Open-Vocabulary Semantic Segmentation.

This repository keeps DINO purity as a deterministic fallback expert. Semantic and mask experts can run in selectable modes:

- `fallback`: deterministic dense proxy logits, no model download
- `clip`: optional real CLIP region-crop scoring through `open_clip`
- `clearclip`: optional ClearCLIP-style dense patch-level scoring through `open_clip`
- `--mask-backend fallback`: deterministic class-agnostic fallback masks, the default option
- `--mask-backend sam`: optional SAM/MobileSAM class-agnostic masks from a user-provided checkpoint
- `--feature-backend fallback`: deterministic patch-level proxy features for region purity
- `--feature-backend dinov2`: optional DINOv2 dense patch features for region purity
- text expert: fixed positive and negative prompt templates, no external LLM

DINOv2 is used only for region purity / spatial uncertainty. It is not used for class prediction.

The current `clip` backend is region crop-level CLIP scoring. The `clearclip` backend is the dense semantic path: it computes image-sized patch-level CLIP score maps and pools those logits over SAM regions.

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

To use the optional ClearCLIP-style dense semantic backend:

```bash
pip install -r requirements-clip.txt
python infer_ur_ovss.py --image path/to/image.jpg --classes "cat,dog,person,car" --output outputs/demo.png --semantic-backend clearclip
```

This backend currently uses a minimal ViT-B/16 dense patch-logit adapter inspired by the official ClearCLIP implementation. It does not vendor official ClearCLIP code or weights; see `docs/clearclip_integration_plan.md` for integration scope and attribution.

To evaluate ClearCLIP-style dense logits directly on Pascal VOC without SAM, DINOv2, or routing:

```bash
python eval_clearclip_dense_voc.py --voc-root path/to/VOCdevkit/VOC2012 --split val --limit 100 --output-dir outputs/voc_clearclip_dense_l100 --voc-mode voc20 --voc20-ignore-background
```

Use this dense-only ablation to compare against the existing ClearCLIP + SAM and ClearCLIP + SAM + DINOv2 paths. See `docs/clearclip_dense_ablation.md` for the A/B/C/D comparison commands and the remaining gaps versus the official ClearCLIP sliding-window setup.

### Dense-Only CLIP Baselines

The existing `--semantic-backend clip` path in `infer_ur_ovss.py` and `eval_pascal_voc.py` is region crop-level CLIP scoring over SAM masks. It is not the vanilla CLIP dense segmentation baseline used in open-vocabulary segmentation papers.

Use `eval_dense_voc.py --semantic-backend clip` for the paper-style vanilla CLIP dense-only baseline:

```bash
python eval_dense_voc.py --semantic-backend clip --voc-root path/to/VOCdevkit/VOC2012 --split val --limit 100 --output-dir outputs/voc_dense_clip_l100 --voc-mode voc20 --voc20-ignore-background
```

Use the same script with `--semantic-backend clearclip` for a direct ClearCLIP dense-only comparison under the same evaluator and output format:

```bash
python eval_dense_voc.py --semantic-backend clearclip --voc-root path/to/VOCdevkit/VOC2012 --split val --limit 100 --output-dir outputs/voc_dense_clearclip_l100 --voc-mode voc20 --voc20-ignore-background
```

Both dense-only modes save `metrics.json` and per-image prediction arrays under `predictions/`. They do not use SAM, DINOv2, or routing.

To use the optional SAM mask backend:

```bash
pip install -r requirements-sam.txt
python infer_ur_ovss.py --image path/to/image.jpg --classes "cat,dog,person,car" --output outputs/demo.png --mask-backend sam --sam-checkpoint path/to/sam_vit_b.pth --max-masks 100
```

SAM/MobileSAM weights must be provided through `--sam-checkpoint`; the inference CLI does not download them automatically, and they should not be committed to the repository. If `segment-anything` / `mobile-sam`, the checkpoint path, or model initialization is unavailable, the CLI exits with a clear mask backend error.

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

### VOC20 vs VOC21

`eval_pascal_voc.py` exposes `--voc-mode voc20|voc21`.

- `voc20` is the default and preserves the existing reported numbers. It maps prediction `-1` to VOC background label `0`, maps prediction labels `0-19` to VOC foreground labels `1-20`, ignores GT label `255`, and reports mIoU over only the 20 foreground classes. By default, GT background label `0` remains valid, so foreground predictions on background pixels still count in each foreground class union as false positives. Background IoU is not included in the final mIoU.
- `voc20 --voc20-ignore-background` switches to a stricter without-background style: GT background pixels are ignored before the confusion matrix. This can be closer to some paper tables that report VOC foreground-only evaluation without counting background pixels.
- `voc21` evaluates background plus the 20 foreground classes and includes background IoU in mIoU. The current model has no separate learned background class; prediction `-1` / unassigned pixels become VOC background.

Paper results can differ depending on whether they use VOC20 foreground-only, VOC20 with GT background ignored, or VOC21 background-inclusive evaluation. Check the table protocol before comparing against ClearCLIP or ProxyCLIP numbers.

## VOC Evaluation Diagnostics

After running `eval_pascal_voc.py`, use `analyze_voc_outputs.py` to inspect failure modes from the saved prediction arrays, per-image debug JSON, and Pascal VOC GT masks:

```bash
python analyze_voc_outputs.py --eval-dir outputs/voc_real_fullval --voc-root path/to/VOCdevkit/VOC2012 --output-json outputs/voc_real_fullval/diagnostics.json
```

The diagnostics script does not read or save large image files. It reports per-image foreground IoU, foreground pixel ratios, GT background pixel ratio, predicted background/unassigned ratio, foreground false positives on GT background, class distributions, region counts, finite-pixel average confidence, mean foreground confidence, finite confidence pixel ratio, unassigned pixel ratio, uncertainty counts, route-type counts, and summary rankings such as worst/best images, foreground over- or under-prediction, most predicted classes, most missed GT classes, and global route-type distribution.

## Background Filtering

`infer_ur_ovss.py` and `eval_pascal_voc.py` support two lightweight foreground filters after region routing and before pixel fusion:

- `--background-threshold`: filters regions whose routed confidence is below the threshold.
- `--background-margin-threshold`: filters regions whose semantic margin is below the threshold.

Both default to `0.0`, which keeps the previous behavior. Filtered regions stay in the debug JSON with `filtered_as_background` and `background_filter_reason`, but they do not contribute foreground pixels to the final segmentation map.

For Pascal VOC threshold sweeps, start with small validation limits before full val:

```bash
python eval_pascal_voc.py --voc-root path/to/VOCdevkit/VOC2012 --split val --limit 50 --output-dir outputs/voc_bg_t020_m000 --semantic-backend clip --mask-backend sam --sam-checkpoint path/to/sam_vit_b.pth --sam-model-type vit_b --feature-backend dinov2 --dinov2-model facebook/dinov2-small --max-masks 50 --background-threshold 0.20 --background-margin-threshold 0.00
```

## Preparing Real Backend Resources

The real backend smoke test needs Pascal VOC 2012 segmentation data and a local SAM ViT-B checkpoint. Keep datasets, model weights, `outputs/`, and cache files outside GitHub commits.

By default, the helper scripts use `D:\datasets` for datasets and `D:\models` for model weights. Override them with `DATASET_DIR` and `MODEL_DIR` if you prefer another location outside this repository.

Download and validate Pascal VOC 2012:

```powershell
$env:DATASET_DIR="D:\datasets"
.\scripts\download_voc2012.ps1
```

The script downloads `VOCtrainval_11-May-2012.tar`, extracts it to `DATASET_DIR\VOCdevkit\VOC2012`, checks `JPEGImages`, `SegmentationClass`, and `ImageSets\Segmentation\val.txt`, then prints a ready-to-use value:

```powershell
$env:VOC_ROOT="D:\datasets\VOCdevkit\VOC2012"
```

Download and validate the SAM ViT-B checkpoint:

```powershell
$env:MODEL_DIR="D:\models"
.\scripts\download_sam_checkpoint.ps1
```

The script writes `sam_vit_b_01ec64.pth` to `MODEL_DIR`, never to the repository by default, then prints:

```powershell
$env:SAM_CHECKPOINT="D:\models\sam_vit_b_01ec64.pth"
```

Install the optional real-backend dependencies, set the printed environment variables in the same PowerShell session, and start with one VOC validation image:

```powershell
pip install -r requirements-clip.txt
pip install -r requirements-sam.txt
pip install -r requirements-dino.txt

$env:VOC_ROOT="D:\datasets\VOCdevkit\VOC2012"
$env:SAM_CHECKPOINT="D:\models\sam_vit_b_01ec64.pth"
$env:LIMIT="1"
.\scripts\run_voc_real_smoke.ps1
```

After `LIMIT=1` works, try `LIMIT=10`:

```powershell
$env:LIMIT="10"
$env:OUTPUT_DIR="outputs/voc_real_smoke_limit10"
.\scripts\run_voc_real_smoke.ps1
```

Full VOC val is much slower with CLIP, SAM, and DINOv2 enabled; a GPU is recommended before running the complete validation split.

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
