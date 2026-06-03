# ClearCLIP Dense-Only VOC Ablation

## Goal

The current best UR-OVSS result mixes three effects:

- ClearCLIP-style dense semantic logits
- SAM region pooling and overlapping-mask fusion
- DINOv2 purity / uncertainty routing

`eval_clearclip_dense_voc.py` isolates the first item. It evaluates dense
ClearCLIP-style logits directly on Pascal VOC, with no SAM masks, no DINOv2
features, and no uncertainty routing. This gives a quick answer to whether the
main bottleneck is the dense semantic expert itself or the downstream
region/routing stack.

## Required Comparisons

A. **ClearCLIP Dense-Only**

```bash
python eval_clearclip_dense_voc.py --voc-root path/to/VOCdevkit/VOC2012 --split val --limit 100 --output-dir outputs/voc_clearclip_dense_l100 --voc-mode voc20 --voc20-ignore-background
```

B. **ClearCLIP + SAM**

Run the existing Pascal VOC evaluator with ClearCLIP semantic scoring and SAM
masks, but fallback purity features if you want to remove DINOv2 routing
effects:

```bash
python eval_pascal_voc.py --voc-root path/to/VOCdevkit/VOC2012 --split val --limit 100 --output-dir outputs/voc_clearclip_sam_l100 --semantic-backend clearclip --mask-backend sam --sam-checkpoint path/to/sam_vit_b.pth --sam-model-type vit_b --feature-backend fallback --max-masks 50 --voc-mode voc20 --voc20-ignore-background
```

C. **ClearCLIP + SAM + DINOv2 Routing**

```bash
python eval_pascal_voc.py --voc-root path/to/VOCdevkit/VOC2012 --split val --limit 100 --output-dir outputs/voc_clearclip_sam_dino_l100 --semantic-backend clearclip --mask-backend sam --sam-checkpoint path/to/sam_vit_b.pth --sam-model-type vit_b --feature-backend dinov2 --dinov2-model facebook/dinov2-small --max-masks 50 --voc-mode voc20 --voc20-ignore-background
```

D. **Crop-Level CLIP + SAM + DINOv2**

```bash
python eval_pascal_voc.py --voc-root path/to/VOCdevkit/VOC2012 --split val --limit 100 --output-dir outputs/voc_clip_sam_dino_l100 --semantic-backend clip --mask-backend sam --sam-checkpoint path/to/sam_vit_b.pth --sam-model-type vit_b --feature-backend dinov2 --dinov2-model facebook/dinov2-small --max-masks 50 --voc-mode voc20 --voc20-ignore-background
```

## Interpretation

- If A is strong and B/C drop, SAM pooling or region fusion is likely hurting
  dense logits.
- If A is weak, the ClearCLIP-style backend is the first bottleneck.
- If B is stronger than C, the current uncertainty routing / DINO purity signal
  is probably dragging predictions down.
- If D is close to C, dense ClearCLIP is not yet giving enough advantage over
  crop-level CLIP in this repository.

## Official ClearCLIP Alignment Check

Official ClearCLIP references:

- Code: https://github.com/mc-lan/ClearCLIP
- Paper: https://arxiv.org/abs/2407.12442

The official `clearclip_segmentor.py` uses CLIP ViT-B/16 by default, computes
text features from `openai_imagenet_template`, supports sliding-window
inference with `slide_crop=448` and `slide_stride=224`, upsamples logits to the
original image size, and applies the final segmentation argmax after softmax.
The VOC20 config resizes validation images with `scale=(2048, 448)` before
inference.

Current repository status:

- Final layer no residual: implemented in `clearclip_backend.py` for the last
  ViT block.
- Self-self attention: implemented for the final block by calling attention
  with the same normalized token tensor as query, key, and value.
- No final FFN: implemented by skipping the last block MLP / FFN path.
- Patch-level dense logits: implemented and exposed through
  `dense_logits_for_prompts()`.
- Resize to original image: implemented by resizing the dense grid to the input
  image shape.
- ImageNet prompts: dense-only evaluation uses OpenAI ImageNet-style templates
  and averages prompt logits per class.
- Sliding-window inference: not implemented yet. The current backend still
  relies on the `open_clip` preprocessing path, so large images are not decoded
  with the official 448 crop / 224 stride sliding-window protocol.
- Official text prototype averaging: approximated. This script averages dense
  logits across prompts; the official implementation averages normalized text
  features per class and normalizes the class prototype before image-text
  logits.
- MMSeg preprocessing and dataset protocol: not fully reproduced. This
  repository uses a lightweight PIL/numpy evaluator rather than the official
  MMSeg pipeline.

Because of those remaining gaps, dense-only numbers from this script are an
internal bottleneck diagnostic, not an official ClearCLIP reproduction.
