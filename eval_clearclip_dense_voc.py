"""Dense-only ClearCLIP-style Pascal VOC evaluation.

This script evaluates the dense semantic output from `clearclip_backend.py`
directly on Pascal VOC. It intentionally bypasses SAM masks, DINOv2 purity, and
uncertainty routing so the dense semantic expert can be measured in isolation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from PIL import Image

from clearclip_backend import ClearClipSemanticAdapter
from eval_pascal_voc import (
    VOC21_CLASSES,
    VOC_CLASSES,
    compute_confusion_matrix,
    compute_voc_confusion_matrix,
    read_split_ids,
    summarize_voc_metrics,
)
from infer_ur_ovss import SemanticBackendError


OPENAI_IMAGENET_TEMPLATES = [
    "a bad photo of a {class}.",
    "a photo of many {class}.",
    "a sculpture of a {class}.",
    "a photo of the hard to see {class}.",
    "a low resolution photo of the {class}.",
    "a rendering of a {class}.",
    "graffiti of a {class}.",
    "a bad photo of the {class}.",
    "a cropped photo of the {class}.",
    "a tattoo of a {class}.",
    "the embroidered {class}.",
    "a photo of a hard to see {class}.",
    "a bright photo of a {class}.",
    "a photo of a clean {class}.",
    "a photo of a dirty {class}.",
    "a dark photo of the {class}.",
    "a drawing of a {class}.",
    "a photo of my {class}.",
    "the plastic {class}.",
    "a photo of the cool {class}.",
    "a close-up photo of a {class}.",
    "a black and white photo of the {class}.",
    "a painting of the {class}.",
    "a painting of a {class}.",
    "a pixelated photo of the {class}.",
    "a sculpture of the {class}.",
    "a bright photo of the {class}.",
    "a cropped photo of a {class}.",
    "a plastic {class}.",
    "a photo of the dirty {class}.",
    "a jpeg corrupted photo of a {class}.",
    "a blurry photo of the {class}.",
    "a photo of the {class}.",
    "a good photo of the {class}.",
    "a rendering of the {class}.",
    "a {class} in a video game.",
    "a photo of one {class}.",
    "a doodle of a {class}.",
    "a close-up photo of the {class}.",
    "a photo of a {class}.",
    "the origami {class}.",
    "the {class} in a video game.",
    "a sketch of a {class}.",
    "a doodle of the {class}.",
    "a origami {class}.",
    "a low resolution photo of a {class}.",
    "the toy {class}.",
    "a rendition of the {class}.",
    "a photo of the clean {class}.",
    "a photo of a large {class}.",
    "a rendition of a {class}.",
    "a photo of a nice {class}.",
    "a photo of a weird {class}.",
    "a blurry photo of a {class}.",
    "a cartoon {class}.",
    "art of a {class}.",
    "a sketch of the {class}.",
    "a embroidered {class}.",
    "a pixelated photo of a {class}.",
    "itap of the {class}.",
    "a jpeg corrupted photo of the {class}.",
    "a good photo of a {class}.",
    "a plushie {class}.",
    "a photo of the nice {class}.",
    "a photo of the small {class}.",
    "a photo of the weird {class}.",
    "the cartoon {class}.",
    "art of the {class}.",
    "a drawing of the {class}.",
    "a photo of the large {class}.",
    "a black and white photo of a {class}.",
    "the plushie {class}.",
    "a dark photo of a {class}.",
    "itap of a {class}.",
    "graffiti of the {class}.",
    "a toy {class}.",
    "itap of my {class}.",
    "a photo of a cool {class}.",
    "a photo of a small {class}.",
    "a tattoo of the {class}.",
]


VOC_PROMPT_NAMES = {
    "diningtable": "dining table",
    "pottedplant": "potted plant",
    "tvmonitor": "tv monitor",
}


def _load_rgb_image(path: Path) -> tuple[Image.Image, np.ndarray]:
    """Load an RGB image and normalized numpy array.

    Args:
        path: JPEG image path.

    Returns:
        Tuple of RGB PIL image and float32 image array with shape [H, W, 3].
    """

    image = Image.open(path).convert("RGB")
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    return image, image_array


def _load_target_mask(path: Path) -> np.ndarray:
    """Load a Pascal VOC segmentation mask with shape [H, W]."""

    return np.asarray(Image.open(path), dtype=np.int64)


def _prompt_name(class_name: str) -> str:
    """Return the natural-language prompt name for a VOC class."""

    return VOC_PROMPT_NAMES.get(class_name, class_name)


def build_imagenet_prompts(class_names: Sequence[str]) -> List[str]:
    """Build OpenAI ImageNet template prompts for class names.

    Args:
        class_names: Class names with length C.

    Returns:
        Prompt list with length C * T, grouped by class.
    """

    prompts: List[str] = []
    for class_name in class_names:
        prompt_class = _prompt_name(class_name)
        prompts.extend(template.format(**{"class": prompt_class}) for template in OPENAI_IMAGENET_TEMPLATES)
    return prompts


def _resize_logits(dense_logits: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    """Resize dense logits to image shape [H, W, C] with bilinear sampling."""

    height, width = output_shape
    if dense_logits.shape[:2] == output_shape:
        return dense_logits.astype(np.float32)

    channels = dense_logits.shape[-1]
    resized = np.empty((height, width, channels), dtype=np.float32)
    resample = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR
    for channel in range(channels):
        channel_image = Image.fromarray(dense_logits[..., channel].astype(np.float32), mode="F")
        resized[..., channel] = np.asarray(channel_image.resize((width, height), resample=resample), dtype=np.float32)
    return resized


def compute_dense_logits_for_classes(
    adapter: ClearClipSemanticAdapter,
    class_names: Sequence[str],
    output_shape: tuple[int, int],
) -> np.ndarray:
    """Compute image-sized class logits from ClearCLIP dense prompt logits.

    Args:
        adapter: Prepared ClearCLIP dense adapter.
        class_names: Class names with length C.
        output_shape: Desired image shape [H, W].

    Returns:
        Dense class logits with shape [H, W, C]. Prompt logits are averaged
        over OpenAI ImageNet templates for each class.
    """

    prompt_count = len(OPENAI_IMAGENET_TEMPLATES)
    prompt_logits = adapter.dense_logits_for_prompts(build_imagenet_prompts(class_names))
    prompt_logits = _resize_logits(prompt_logits, output_shape)
    height, width = output_shape
    class_logits = prompt_logits.reshape(height, width, len(class_names), prompt_count).mean(axis=-1)
    return class_logits.astype(np.float32)


def _prediction_to_confusion(
    pred_indices: np.ndarray,
    target: np.ndarray,
    voc_mode: str,
    voc20_ignore_background: bool,
) -> np.ndarray:
    """Convert dense argmax indices into a VOC confusion matrix."""

    if voc_mode == "voc21":
        return compute_confusion_matrix(pred_indices, target, num_classes=len(VOC21_CLASSES), ignore_index=255)
    return compute_voc_confusion_matrix(
        pred_indices,
        target,
        voc_mode="voc20",
        voc20_ignore_background=voc20_ignore_background,
    )


def evaluate_dataset(
    voc_root: Path,
    split: str,
    output_dir: Path,
    limit: Optional[int] = None,
    voc_mode: str = "voc20",
    voc20_ignore_background: bool = False,
    clearclip_model_name: str = "ViT-B-16",
    clearclip_pretrained: str = "openai",
) -> Dict[str, Any]:
    """Evaluate ClearCLIP dense logits directly on Pascal VOC.

    Args:
        voc_root: Path to `VOCdevkit/VOC2012`.
        split: VOC segmentation split name.
        output_dir: Directory where `metrics.json` is saved.
        limit: Optional maximum number of image ids.
        voc_mode: "voc20" evaluates foreground classes only; "voc21"
            includes background.
        voc20_ignore_background: In VOC20 mode, ignore GT background pixels
            before the confusion matrix.
        clearclip_model_name: open_clip model name for the dense adapter.
        clearclip_pretrained: open_clip pretrained tag for the dense adapter.

    Returns:
        Metrics dictionary written to `metrics.json`.
    """

    if voc_mode not in {"voc20", "voc21"}:
        raise ValueError(f"voc_mode must be 'voc20' or 'voc21', got {voc_mode!r}.")

    voc_root = Path(voc_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_ids = read_split_ids(voc_root, split)
    if limit is not None:
        image_ids = image_ids[:limit]

    adapter = ClearClipSemanticAdapter(
        model_name=clearclip_model_name,
        pretrained=clearclip_pretrained,
        backend_error_cls=SemanticBackendError,
    )
    print("ClearCLIP dense backend is initialized once for this dense-only VOC evaluation run.")

    confusion = np.zeros((len(VOC21_CLASSES), len(VOC21_CLASSES)), dtype=np.int64)
    evaluated_images = 0
    skipped_images = 0
    skipped: List[Dict[str, str]] = []
    eval_class_names = VOC21_CLASSES if voc_mode == "voc21" else VOC_CLASSES

    for image_id in image_ids:
        image_path = voc_root / "JPEGImages" / f"{image_id}.jpg"
        target_path = voc_root / "SegmentationClass" / f"{image_id}.png"
        if not image_path.exists() or not target_path.exists():
            skipped_images += 1
            missing = []
            if not image_path.exists():
                missing.append(str(image_path))
            if not target_path.exists():
                missing.append(str(target_path))
            skipped.append({"image_id": image_id, "reason": f"missing file(s): {', '.join(missing)}"})
            continue

        image, image_array = _load_rgb_image(image_path)
        adapter.prepare_image(image, image_array)
        dense_logits = compute_dense_logits_for_classes(adapter, eval_class_names, image_array.shape[:2])
        pred_indices = np.argmax(dense_logits, axis=-1).astype(np.int64)
        target = _load_target_mask(target_path)
        confusion += _prediction_to_confusion(
            pred_indices,
            target,
            voc_mode=voc_mode,
            voc20_ignore_background=voc20_ignore_background,
        )
        evaluated_images += 1

    if evaluated_images == 0:
        first_skip = f" First skipped item: {skipped[0]['reason']}." if skipped else ""
        raise RuntimeError(
            "ClearCLIP dense VOC evaluation did not evaluate any images. "
            f"Skipped {skipped_images} image(s); check --voc-root, --split, and dataset files.{first_skip}"
        )

    summarized = summarize_voc_metrics(confusion, voc_mode=voc_mode)
    metrics_path = output_dir / "metrics.json"
    metrics: Dict[str, Any] = {
        "split": split,
        "voc_root": str(voc_root),
        "semantic_backend": "clearclip_dense_only",
        "clearclip_model_name": clearclip_model_name,
        "clearclip_pretrained": clearclip_pretrained,
        "prompt_templates": "openai_imagenet_template_logit_average",
        "voc_mode": voc_mode,
        "voc20_ignore_background": bool(voc20_ignore_background),
        "background_iou": summarized["background_iou"],
        "mIoU": summarized["mIoU"],
        "per_class_iou": summarized["per_class_iou"],
        "evaluated_images": evaluated_images,
        "skipped_images": skipped_images,
        "skipped": skipped,
        "classes": summarized["classes"],
        "metrics_path": str(metrics_path),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for dense-only VOC evaluation."""

    parser = argparse.ArgumentParser(description="Evaluate ClearCLIP dense logits directly on Pascal VOC 2012.")
    parser.add_argument("--voc-root", required=True, type=Path, help="Path to VOCdevkit/VOC2012.")
    parser.add_argument("--split", default="val", help="VOC segmentation split name.")
    parser.add_argument("--limit", default=None, type=int, help="Optional maximum number of images to evaluate.")
    parser.add_argument(
        "--output-dir",
        default=Path("outputs/clearclip_dense_voc"),
        type=Path,
        help="Directory for metrics.json.",
    )
    parser.add_argument(
        "--voc-mode",
        choices=("voc20", "voc21"),
        default="voc20",
        help="VOC evaluation mode. voc20 reports foreground-only mIoU; voc21 includes background in mIoU.",
    )
    parser.add_argument(
        "--voc20-ignore-background",
        action="store_true",
        help="In voc20 mode, ignore GT background pixels before the confusion matrix.",
    )
    parser.add_argument("--clearclip-model-name", default="ViT-B-16", help="open_clip model name.")
    parser.add_argument("--clearclip-pretrained", default="openai", help="open_clip pretrained tag.")
    return parser


def main() -> None:
    """CLI entry point for ClearCLIP dense-only Pascal VOC evaluation."""

    args = build_arg_parser().parse_args()
    try:
        metrics = evaluate_dataset(
            voc_root=args.voc_root,
            split=args.split,
            output_dir=args.output_dir,
            limit=args.limit,
            voc_mode=args.voc_mode,
            voc20_ignore_background=args.voc20_ignore_background,
            clearclip_model_name=args.clearclip_model_name,
            clearclip_pretrained=args.clearclip_pretrained,
        )
    except (FileNotFoundError, ValueError, RuntimeError, SemanticBackendError) as exc:
        raise SystemExit(f"ClearCLIP dense VOC evaluation error: {exc}") from exc

    print(f"Evaluated images: {metrics['evaluated_images']}")
    print(f"Skipped images: {metrics['skipped_images']}")
    print(f"mIoU: {metrics['mIoU']:.6f}")
    print(f"VOC mode: {metrics['voc_mode']}")
    if metrics["voc_mode"] == "voc20":
        print(f"VOC20 ignore background: {metrics['voc20_ignore_background']}")
    if metrics["background_iou"] is not None:
        print(f"Background IoU: {metrics['background_iou']:.6f}")
    print(f"Metrics JSON: {metrics['metrics_path']}")
    print("Per-class IoU:")
    for class_name, iou in metrics["per_class_iou"].items():
        value = "nan" if iou is None else f"{iou:.6f}"
        print(f"  {class_name}: {value}")


if __name__ == "__main__":
    main()
