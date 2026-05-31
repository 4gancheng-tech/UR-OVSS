"""Pascal VOC 2012 semantic segmentation evaluation for UR-OVSS MVP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from PIL import Image

from infer_ur_ovss import FeatureBackendError, MaskBackendError, SemanticBackendError, run_inference


VOC_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


def read_split_ids(voc_root: Path, split: str) -> List[str]:
    """Read Pascal VOC segmentation image ids for a split.

    Args:
        voc_root: Path to `VOCdevkit/VOC2012`.
        split: Split name such as `val`.

    Returns:
        Image id list from `ImageSets/Segmentation/{split}.txt`.
    """

    if not voc_root.exists():
        raise FileNotFoundError(f"VOC root does not exist: {voc_root}")
    split_path = voc_root / "ImageSets" / "Segmentation" / f"{split}.txt"
    if not split_path.exists():
        raise FileNotFoundError(f"VOC split file does not exist: {split_path}")

    image_ids = [line.strip() for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not image_ids:
        raise ValueError(f"VOC split file is empty: {split_path}")
    return image_ids


def map_prediction_to_voc_labels(pred: np.ndarray) -> np.ndarray:
    """Map UR-OVSS labels to Pascal VOC labels.

    Args:
        pred: UR-OVSS label map with shape [H, W], using -1 for unassigned
            pixels and 0-19 for foreground class indices.

    Returns:
        VOC-style label map with shape [H, W], where -1 becomes background 0
        and foreground predictions become 1-20.
    """

    pred_array = np.asarray(pred, dtype=np.int64)
    mapped = np.zeros(pred_array.shape, dtype=np.int64)
    foreground = pred_array >= 0
    mapped[foreground] = pred_array[foreground] + 1
    return mapped


def compute_confusion_matrix(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int,
    ignore_index: int = 255,
) -> np.ndarray:
    """Compute a semantic segmentation confusion matrix.

    Args:
        pred: Predicted labels with shape [H, W].
        target: Ground-truth labels with shape [H, W].
        num_classes: Number of labels included in the matrix.
        ignore_index: Target label ignored during evaluation.

    Returns:
        Confusion matrix with shape [num_classes, num_classes], where rows are
        ground truth labels and columns are predictions.
    """

    pred_array = np.asarray(pred, dtype=np.int64)
    target_array = np.asarray(target, dtype=np.int64)
    if pred_array.shape != target_array.shape:
        raise ValueError(f"Prediction shape {pred_array.shape} does not match target shape {target_array.shape}.")

    valid = target_array != ignore_index
    valid &= target_array >= 0
    valid &= target_array < num_classes
    valid &= pred_array >= 0
    valid &= pred_array < num_classes

    encoded = target_array[valid] * num_classes + pred_array[valid]
    confusion = np.bincount(encoded, minlength=num_classes * num_classes)
    return confusion.reshape(num_classes, num_classes).astype(np.int64)


def compute_iou_from_confusion(confusion: np.ndarray) -> np.ndarray:
    """Compute per-class IoU from a confusion matrix.

    Args:
        confusion: Confusion matrix with shape [C, C].

    Returns:
        Per-class IoU array with shape [C]. Classes absent from prediction and
        target receive NaN.
    """

    matrix = np.asarray(confusion, dtype=np.float64)
    intersection = np.diag(matrix)
    target_area = matrix.sum(axis=1)
    pred_area = matrix.sum(axis=0)
    union = target_area + pred_area - intersection
    iou = np.full(intersection.shape, np.nan, dtype=np.float64)
    valid = union > 0
    iou[valid] = intersection[valid] / union[valid]
    return iou


def _load_target_mask(path: Path) -> np.ndarray:
    """Load a Pascal VOC segmentation mask as an integer numpy array."""

    return np.asarray(Image.open(path), dtype=np.int64)


def _remove_visualizations_if_disabled(outputs: Dict[str, str], save_vis: bool) -> None:
    """Remove per-image visualization PNGs unless save_vis is enabled."""

    if save_vis:
        return
    for key in ("visualization", "label_png"):
        path = Path(outputs[key])
        if path.exists():
            path.unlink()


def evaluate_dataset(
    voc_root: Path,
    split: str,
    output_dir: Path,
    limit: Optional[int] = None,
    semantic_backend: str = "fallback",
    mask_backend: str = "fallback",
    feature_backend: str = "fallback",
    sam_checkpoint: Optional[Path] = None,
    sam_model_type: str = "vit_b",
    max_masks: int = 100,
    dinov2_model: str = "facebook/dinov2-small",
    save_vis: bool = False,
) -> Dict[str, Any]:
    """Evaluate UR-OVSS on Pascal VOC 2012 semantic segmentation.

    Args:
        voc_root: Path to `VOCdevkit/VOC2012`.
        split: Segmentation split name.
        output_dir: Directory where predictions and `metrics.json` are saved.
        limit: Optional maximum number of image ids to evaluate.
        semantic_backend: Semantic backend passed to `run_inference`.
        mask_backend: Mask backend passed to `run_inference`.
        feature_backend: Feature backend passed to `run_inference`.
        sam_checkpoint: Optional SAM checkpoint for the SAM mask backend.
        sam_model_type: SAM model type key.
        max_masks: Maximum number of masks to keep.
        dinov2_model: DINOv2 model id or local path.
        save_vis: Whether to keep per-image visualization PNGs.

    Returns:
        Metrics dictionary also written to `metrics.json`.
    """

    voc_root = Path(voc_root)
    output_dir = Path(output_dir)
    prediction_dir = output_dir / "predictions"
    visualization_dir = output_dir / "visualizations"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    visualization_dir.mkdir(parents=True, exist_ok=True)

    image_ids = read_split_ids(voc_root, split)
    if limit is not None:
        image_ids = image_ids[:limit]

    confusion = np.zeros((len(VOC_CLASSES) + 1, len(VOC_CLASSES) + 1), dtype=np.int64)
    evaluated_images = 0
    skipped_images = 0
    skipped: List[Dict[str, str]] = []

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

        image_output_dir = visualization_dir if save_vis else prediction_dir
        output_path = image_output_dir / f"{image_id}.png"
        result = run_inference(
            image_path=image_path,
            class_names=VOC_CLASSES,
            output_path=output_path,
            semantic_backend=semantic_backend,
            mask_backend=mask_backend,
            feature_backend=feature_backend,
            sam_checkpoint=sam_checkpoint,
            sam_model_type=sam_model_type,
            max_masks=max_masks,
            dinov2_model=dinov2_model,
        )
        _remove_visualizations_if_disabled(result["outputs"], save_vis=save_vis)

        pred_raw = np.load(result["outputs"]["mask_npy"])
        pred_voc = map_prediction_to_voc_labels(pred_raw)
        target = _load_target_mask(target_path)
        confusion += compute_confusion_matrix(pred_voc, target, num_classes=len(VOC_CLASSES) + 1, ignore_index=255)
        evaluated_images += 1

    if evaluated_images == 0:
        first_skip = f" First skipped item: {skipped[0]['reason']}." if skipped else ""
        raise RuntimeError(
            "Pascal VOC evaluation did not evaluate any images. "
            f"Skipped {skipped_images} image(s); check --voc-root, --split, and dataset files.{first_skip}"
        )

    per_label_iou = compute_iou_from_confusion(confusion)
    foreground_iou = per_label_iou[1:]
    miou = float(np.nanmean(foreground_iou)) if not np.all(np.isnan(foreground_iou)) else float("nan")
    per_class_iou = {
        class_name: (None if np.isnan(iou) else float(iou))
        for class_name, iou in zip(VOC_CLASSES, foreground_iou)
    }

    metrics_path = output_dir / "metrics.json"
    metrics: Dict[str, Any] = {
        "split": split,
        "voc_root": str(voc_root),
        "mIoU": miou,
        "per_class_iou": per_class_iou,
        "evaluated_images": evaluated_images,
        "skipped_images": skipped_images,
        "skipped": skipped,
        "classes": VOC_CLASSES,
        "metrics_path": str(metrics_path),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for Pascal VOC evaluation."""

    parser = argparse.ArgumentParser(description="Evaluate UR-OVSS on Pascal VOC 2012 segmentation.")
    parser.add_argument("--voc-root", required=True, type=Path, help="Path to VOCdevkit/VOC2012.")
    parser.add_argument("--split", default="val", help="VOC segmentation split name.")
    parser.add_argument("--output-dir", default=Path("outputs/voc_eval"), type=Path, help="Evaluation output dir.")
    parser.add_argument("--limit", default=None, type=int, help="Optional maximum number of images to evaluate.")
    parser.add_argument("--semantic-backend", choices=("fallback", "clip"), default="fallback")
    parser.add_argument("--mask-backend", choices=("fallback", "sam"), default="fallback")
    parser.add_argument("--feature-backend", choices=("fallback", "dinov2"), default="fallback")
    parser.add_argument("--sam-checkpoint", default=None, type=Path)
    parser.add_argument("--sam-model-type", default="vit_b")
    parser.add_argument("--max-masks", default=100, type=int)
    parser.add_argument("--dinov2-model", default="facebook/dinov2-small")
    parser.add_argument("--save-vis", action="store_true", help="Keep per-image visualization PNGs.")
    return parser


def main() -> None:
    """CLI entry point for Pascal VOC evaluation."""

    args = build_arg_parser().parse_args()
    try:
        metrics = evaluate_dataset(
            voc_root=args.voc_root,
            split=args.split,
            output_dir=args.output_dir,
            limit=args.limit,
            semantic_backend=args.semantic_backend,
            mask_backend=args.mask_backend,
            feature_backend=args.feature_backend,
            sam_checkpoint=args.sam_checkpoint,
            sam_model_type=args.sam_model_type,
            max_masks=args.max_masks,
            dinov2_model=args.dinov2_model,
            save_vis=args.save_vis,
        )
    except (FileNotFoundError, ValueError, RuntimeError, SemanticBackendError, MaskBackendError, FeatureBackendError) as exc:
        raise SystemExit(f"Pascal VOC evaluation error: {exc}") from exc

    print(f"Evaluated images: {metrics['evaluated_images']}")
    print(f"Skipped images: {metrics['skipped_images']}")
    print(f"mIoU: {metrics['mIoU']:.6f}")
    print(f"Metrics JSON: {metrics['metrics_path']}")
    print("Per-class IoU:")
    for class_name, iou in metrics["per_class_iou"].items():
        value = "nan" if iou is None else f"{iou:.6f}"
        print(f"  {class_name}: {value}")


if __name__ == "__main__":
    main()
