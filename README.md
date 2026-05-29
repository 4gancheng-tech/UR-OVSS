# UR-OVSS MVP

Minimal runnable implementation of Uncertainty-Routed Multi-Expert Open-Vocabulary Semantic Segmentation.

This repository currently has no built-in CLIP/ClearCLIP, SAM, or DINO implementation, so `infer_ur_ovss.py` uses deterministic lightweight fallback experts:

- semantic expert: dense proxy logits
- spatial expert: class-agnostic fallback masks
- region purity expert: patch-level proxy features
- text expert: fixed positive and negative prompt templates, no external LLM

The routing code is model-agnostic, so real CLIP/SAM/DINO adapters can replace the fallback expert functions later.

## Run

```bash
pip install -r requirements.txt
python infer_ur_ovss.py --image path/to/image.jpg --classes "cat,dog,person,car" --output outputs/demo.png
```

The command saves:

- `outputs/demo.png`: blended segmentation visualization
- `outputs/demo_labels.png`: colorized label map
- `outputs/demo_mask.npy`: integer label map, with `-1` for unassigned pixels
- `outputs/demo_confidence.npy`: winning region confidence per pixel
- `outputs/demo.json`: per-region debug records with routing decisions

Each `regions[]` entry in the JSON includes `region_id`, `predicted_label`, `semantic_margin`, `dino_variance`, `semantic_uncertain`, `spatial_uncertain`, `route_type`, and `confidence`.

No evaluation entry is included yet because this empty repository does not contain dataset or evaluation code.
