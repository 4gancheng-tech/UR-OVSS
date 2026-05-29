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

The command saves:

- `outputs/demo.png`: blended segmentation visualization
- `outputs/demo_labels.png`: colorized label map
- `outputs/demo_mask.npy`: integer label map, with `-1` for unassigned pixels
- `outputs/demo_confidence.npy`: winning region confidence per pixel
- `outputs/demo.json`: per-region debug records with routing decisions

Each `regions[]` entry in the JSON includes `region_id`, `predicted_label`, `semantic_margin`, `dino_variance`, `semantic_uncertain`, `spatial_uncertain`, `route_type`, and `confidence`.
It also includes `base_scores`, `positive_scores`, `negative_scores`, and `prompt_rescore_scores` for debugging the selected semantic backend.

No evaluation entry is included yet because this empty repository does not contain dataset or evaluation code.
