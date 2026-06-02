# ClearCLIP Integration Plan

## Current CLIP Backend Vs ClearCLIP

The current `--semantic-backend clip` path is a region crop-level scorer. For
each SAM region, it crops the image around the mask, zeros pixels outside the
mask, encodes that crop with `open_clip`, and compares the crop embedding
against positive and negative text prompts. This is simple and stable, but it
throws away dense spatial evidence before routing because every region gets one
global image embedding.

The new `--semantic-backend clearclip` path is dense. It computes patch-level
CLIP visual features for the full image, converts text prompts into text
features, builds dense image/text score maps with shape `[H, W, T]`, and pools
those dense logits inside each SAM mask. This keeps the existing UR-OVSS region
routing and fusion code while replacing crop-level CLIP scores with dense
semantic evidence.

## ClearCLIP Core Idea

ClearCLIP, from `mc-lan/ClearCLIP`, is based on the observation that standard
CLIP ViT representations are strong globally but noisy for dense segmentation.
The method decomposes CLIP representations for dense vision-language inference
by producing patch-level CLIP features and modifying the final ViT layer.

The official ClearCLIP implementation describes three final-layer changes:

- Remove the final layer residual connection.
- Use self-self attention for the final attention operation.
- Discard the final feed-forward network.

Those changes aim to reduce noisy global residual information and improve local
patch discriminability. The official repository is licensed under NTU S-Lab
License 1.0 and states that it is based on OpenCLIP and SCLIP. This project
does not vendor official ClearCLIP source code or weights; it implements a
minimal adapter inspired by the public method and leaves attribution in
`clearclip_backend.py`.

Sources:

- Official code: https://github.com/mc-lan/ClearCLIP
- Paper: https://arxiv.org/abs/2407.12442

## Why The Current 12.45% mIoU Is Not Directly Comparable

The Pascal VOC full-val result currently reported for this repository uses a
hybrid MVP pipeline: region crop-level CLIP, SAM class-agnostic masks, DINOv2
purity features, simple uncertainty routing, and no training. It is not the
same experimental setup as ClearCLIP or ProxyCLIP papers.

The main differences are:

- The previous semantic expert was crop-level CLIP, not dense CLIP logits.
- VOC evaluation here uses SAM region proposal fusion, not the official
  ClearCLIP dense segmentation decode/evaluation stack.
- Prompt templates, background handling, mask generation, and post-processing
  differ from paper settings.
- This repository currently uses a minimal training-free MVP and does not
  reproduce official ClearCLIP configs, sliding-window inference, or benchmark
  protocol.

Therefore the 12.45% mIoU is useful as an internal baseline for this UR-OVSS
pipeline, but it should not be compared directly against ClearCLIP/ProxyCLIP
paper numbers.

## Minimal Integration Plan

The current MVP integration is intentionally narrow:

1. Keep `fallback` and crop-level `clip` semantic backends unchanged.
2. Add `clearclip_backend.py` with a dense semantic adapter backed by
   `open_clip`.
3. Default to `ViT-B-16` / `openai`, matching the ClearCLIP paper's CLIP
   ViT-B/16 direction more closely than the crop CLIP ViT-B/32 path.
4. Compute image-sized dense patch features once per image.
5. Compute dense positive/negative prompt score maps and cache them per image.
6. For each SAM mask, pool dense logits inside the region and return the same
   score contract as existing semantic adapters:
   `base_scores`, `positive_scores`, `negative_scores`,
   `prompt_rescore_scores`.
7. Let `eval_pascal_voc.py` reuse the initialized adapter once per dataset, as
   it already does for other backends.

This is not yet a full official ClearCLIP reproduction. The open items are:

- Validate the final-layer decomposition against the exact official OpenCLIP
  model internals on a GPU server.
- Add optional sliding-window dense inference for large images.
- Add configurable ClearCLIP model/pretrained arguments if needed.
- Compare dense logits visualizations before and after SAM region pooling.
- Tune VOC-specific background behavior using dense logits instead of region
  confidence thresholds.
