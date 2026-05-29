# UR-OVSS MVP

Minimal runnable implementation of Uncertainty-Routed Multi-Expert Open-Vocabulary Semantic Segmentation.

This repository keeps DINO purity as a deterministic fallback expert. Semantic and mask experts can run in selectable modes:

- `fallback`: deterministic dense proxy logits, no model download
- `clip`: optional real CLIP region-crop scoring through `open_clip`
- `--mask-backend fallback`: deterministic class-agnostic fallback masks, the default option
- `--mask-backend sam`: optional SAM/MobileSAM class-agnostic masks from a user-provided checkpoint
- region purity expert: patch-level proxy features
- text expert: fixed positive and negative prompt templates, no external LLM

The routing code is model-agnostic, so a real DINO adapter can replace the fallback purity expert later.

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

The command saves:

- `outputs/demo.png`: blended segmentation visualization
- `outputs/demo_labels.png`: colorized label map
- `outputs/demo_mask.npy`: integer label map, with `-1` for unassigned pixels
- `outputs/demo_confidence.npy`: winning region confidence per pixel
- `outputs/demo.json`: per-region debug records with routing decisions

Each `regions[]` entry in the JSON includes `region_id`, `predicted_label`, `semantic_margin`, `dino_variance`, `semantic_uncertain`, `spatial_uncertain`, `route_type`, and `confidence`.
It also includes `base_scores`, `positive_scores`, `negative_scores`, and `prompt_rescore_scores` for debugging the selected semantic backend.

No evaluation entry is included yet because this empty repository does not contain dataset or evaluation code.
