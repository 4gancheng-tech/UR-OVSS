"""Diagnostics for Pascal VOC evaluation outputs produced by UR-OVSS."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from eval_pascal_voc import VOC_CLASSES, map_prediction_to_voc_labels


def _load_json(path: Path) -> Dict[str, Any]:
    """Load a JSON object from disk."""

    return json.loads(path.read_text(encoding="utf-8"))


def _load_target_mask(path: Path) -> np.ndarray:
    """Load a Pascal VOC segmentation mask as an integer array."""

    return np.asarray(Image.open(path), dtype=np.int64)


def _resolve_existing_path(raw_path: str, eval_dir: Path, debug_path: Path) -> Path:
    """Resolve an output path recorded in a debug JSON."""

    path = Path(raw_path)
    candidates = [path]
    if not path.is_absolute():
        candidates.extend(
            [
                eval_dir / path,
                debug_path.parent / path.name,
                eval_dir / "predictions" / path.name,
                eval_dir / "visualizations" / path.name,
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find output artifact recorded in {debug_path}: {raw_path}")


def _find_debug_jsons(eval_dir: Path, output_json: Optional[Path]) -> List[Path]:
    """Find per-image debug JSON files under an evaluation directory."""

    output_resolved = output_json.resolve() if output_json is not None and output_json.exists() else None
    debug_paths: List[Path] = []
    for path in sorted(eval_dir.rglob("*.json")):
        if path.name in {"metrics.json", "diagnostics.json"}:
            continue
        if output_resolved is not None and path.resolve() == output_resolved:
            continue
        try:
            payload = _load_json(path)
        except json.JSONDecodeError:
            continue
        if isinstance(payload.get("outputs"), dict) and isinstance(payload.get("regions"), list):
            debug_paths.append(path)
    if not debug_paths:
        raise FileNotFoundError(f"No per-image debug JSON files found under evaluation directory: {eval_dir}")
    return debug_paths


def _foreground_iou(pred_voc: np.ndarray, target: np.ndarray, valid: np.ndarray) -> Optional[float]:
    """Compute binary foreground IoU for one image, ignoring invalid pixels."""

    pred_fg = valid & (pred_voc > 0)
    target_fg = valid & (target > 0) & (target < 255)
    union = int(np.logical_or(pred_fg, target_fg).sum())
    if union == 0:
        return None
    intersection = int(np.logical_and(pred_fg, target_fg).sum())
    return float(intersection / union)


def _class_distribution(labels: np.ndarray, valid: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Summarize non-background VOC foreground class pixels for one image."""

    denominator = int(valid.sum())
    if denominator == 0:
        return {}

    distribution: Dict[str, Dict[str, float]] = {}
    for label, class_name in enumerate(VOC_CLASSES, start=1):
        pixels = int((valid & (labels == label)).sum())
        if pixels > 0:
            distribution[class_name] = {
                "pixels": pixels,
                "ratio": float(pixels / denominator),
            }
    return distribution


def _finite_mean(values: np.ndarray, selector: np.ndarray) -> Optional[float]:
    """Compute a finite-only mean for selected pixels, or None if empty."""

    selected = np.asarray(values, dtype=np.float32)[selector]
    finite_selected = selected[np.isfinite(selected)]
    if finite_selected.size == 0:
        return None
    return float(finite_selected.mean())


def _compact_image_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Create a small image summary entry for ranked lists."""

    return {
        "image_id": record["image_id"],
        "foreground_iou": record["foreground_iou"],
        "predicted_foreground_pixel_ratio": record["predicted_foreground_pixel_ratio"],
        "gt_foreground_pixel_ratio": record["gt_foreground_pixel_ratio"],
        "num_regions": record["num_regions"],
        "average_confidence": record["average_confidence"],
    }


def _ranked_images(records: List[Dict[str, Any]], reverse: bool) -> List[Dict[str, Any]]:
    """Rank images by foreground IoU and keep the top 20."""

    finite_records = [record for record in records if record["foreground_iou"] is not None]
    ranked = sorted(finite_records, key=lambda record: record["foreground_iou"], reverse=reverse)
    return [_compact_image_record(record) for record in ranked[:20]]


def _foreground_bias_images(records: List[Dict[str, Any]], over: bool) -> List[Dict[str, Any]]:
    """Rank images by foreground over- or under-prediction."""

    def with_difference(record: Dict[str, Any]) -> Dict[str, Any]:
        difference = record["predicted_foreground_pixel_ratio"] - record["gt_foreground_pixel_ratio"]
        compact = _compact_image_record(record)
        compact["foreground_ratio_difference"] = float(difference)
        return compact

    biased = [with_difference(record) for record in records]
    if over:
        filtered = [record for record in biased if record["foreground_ratio_difference"] > 0.0]
        return sorted(filtered, key=lambda record: record["foreground_ratio_difference"], reverse=True)[:20]
    filtered = [record for record in biased if record["foreground_ratio_difference"] < 0.0]
    return sorted(filtered, key=lambda record: record["foreground_ratio_difference"])[:20]


def _most_predicted_classes(predicted_counts: np.ndarray, total_valid_pixels: int) -> List[Dict[str, Any]]:
    """Summarize globally frequent predicted foreground classes."""

    records = []
    denominator = max(total_valid_pixels, 1)
    for label, class_name in enumerate(VOC_CLASSES, start=1):
        pixels = int(predicted_counts[label])
        records.append(
            {
                "class_name": class_name,
                "pixels": pixels,
                "ratio": float(pixels / denominator),
            }
        )
    return sorted(records, key=lambda record: record["pixels"], reverse=True)[:20]


def _most_missed_gt_classes(gt_counts: np.ndarray, missed_counts: np.ndarray) -> List[Dict[str, Any]]:
    """Summarize GT foreground classes that were most often missed."""

    records = []
    for label, class_name in enumerate(VOC_CLASSES, start=1):
        gt_pixels = int(gt_counts[label])
        missed_pixels = int(missed_counts[label])
        missed_ratio = None if gt_pixels == 0 else float(missed_pixels / gt_pixels)
        records.append(
            {
                "class_name": class_name,
                "gt_pixels": gt_pixels,
                "missed_pixels": missed_pixels,
                "missed_ratio": missed_ratio,
            }
        )
    return sorted(records, key=lambda record: (record["missed_pixels"], record["gt_pixels"]), reverse=True)[:20]


def _image_id_from_debug(debug: Dict[str, Any], debug_path: Path) -> str:
    """Infer an image id from a debug payload or debug filename."""

    image_path = debug.get("image")
    if image_path:
        return Path(str(image_path)).stem
    return debug_path.stem


def _analyze_one_image(
    debug_path: Path,
    eval_dir: Path,
    voc_root: Path,
) -> Dict[str, Any]:
    """Analyze one image from its debug JSON, prediction arrays, and VOC mask."""

    debug = _load_json(debug_path)
    image_id = _image_id_from_debug(debug, debug_path)
    outputs = debug.get("outputs", {})
    if "mask_npy" not in outputs or "confidence_npy" not in outputs:
        raise KeyError(f"Debug JSON is missing mask_npy or confidence_npy outputs: {debug_path}")

    mask_path = _resolve_existing_path(str(outputs["mask_npy"]), eval_dir, debug_path)
    confidence_path = _resolve_existing_path(str(outputs["confidence_npy"]), eval_dir, debug_path)
    target_path = voc_root / "SegmentationClass" / f"{image_id}.png"
    if not target_path.exists():
        raise FileNotFoundError(f"Pascal VOC GT mask does not exist for {image_id}: {target_path}")

    pred_raw = np.load(mask_path)
    pred_voc = map_prediction_to_voc_labels(pred_raw)
    confidence = np.load(confidence_path)
    target = _load_target_mask(target_path)
    if pred_voc.shape != target.shape:
        raise ValueError(f"Prediction shape {pred_voc.shape} does not match GT shape {target.shape} for {image_id}.")
    if confidence.shape != target.shape:
        raise ValueError(f"Confidence shape {confidence.shape} does not match GT shape {target.shape} for {image_id}.")

    valid = target != 255
    valid_pixels = int(valid.sum())
    if valid_pixels == 0:
        raise ValueError(f"VOC GT mask contains no valid pixels after ignoring 255 for {image_id}.")

    pred_fg_pixels = int((valid & (pred_voc > 0)).sum())
    gt_fg_pixels = int((valid & (target > 0) & (target < 255)).sum())
    gt_background = valid & (target == 0)
    gt_background_pixels = int(gt_background.sum())
    background_false_positive_pixels = int((gt_background & (pred_voc > 0)).sum())
    confidence_array = np.asarray(confidence, dtype=np.float32)
    finite_confidence = valid & np.isfinite(confidence_array)
    predicted_foreground = valid & (pred_voc > 0)
    regions = debug.get("regions", [])
    route_counts = Counter(str(region.get("route_type", "unknown")) for region in regions)

    return {
        "image_id": image_id,
        "foreground_iou": _foreground_iou(pred_voc, target, valid),
        "predicted_foreground_pixel_ratio": float(pred_fg_pixels / valid_pixels),
        "gt_foreground_pixel_ratio": float(gt_fg_pixels / valid_pixels),
        "gt_background_pixel_ratio": float(gt_background_pixels / valid_pixels),
        "predicted_background_or_unassigned_ratio": float(int((valid & (pred_voc == 0)).sum()) / valid_pixels),
        "foreground_false_positive_on_gt_background": (
            None
            if gt_background_pixels == 0
            else float(background_false_positive_pixels / gt_background_pixels)
        ),
        "predicted_class_distribution": _class_distribution(pred_voc, valid),
        "gt_class_distribution": _class_distribution(target, valid),
        "num_regions": int(len(regions)),
        "average_confidence": _finite_mean(confidence_array, valid),
        "mean_foreground_confidence": _finite_mean(confidence_array, predicted_foreground),
        "finite_confidence_pixel_ratio": float(int(finite_confidence.sum()) / valid_pixels),
        "unassigned_pixel_ratio": float(int((valid & (pred_voc == 0)).sum()) / valid_pixels),
        "semantic_uncertain_regions": int(sum(bool(region.get("semantic_uncertain")) for region in regions)),
        "spatial_uncertain_regions": int(sum(bool(region.get("spatial_uncertain")) for region in regions)),
        "route_type_counts": dict(sorted(route_counts.items())),
    }


def _accumulate_global_stats(
    record: Dict[str, Any],
    pred_voc: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    predicted_counts: np.ndarray,
    gt_counts: np.ndarray,
    missed_counts: np.ndarray,
) -> None:
    """Accumulate global class counts for summary diagnostics."""

    del record
    for label in range(1, len(VOC_CLASSES) + 1):
        predicted_counts[label] += int((valid & (pred_voc == label)).sum())
        gt_mask = valid & (target == label)
        gt_counts[label] += int(gt_mask.sum())
        missed_counts[label] += int((gt_mask & (pred_voc != label)).sum())


def analyze_eval_outputs(eval_dir: Path, voc_root: Path, output_json: Path) -> Dict[str, Any]:
    """Analyze UR-OVSS Pascal VOC evaluation outputs and write diagnostics JSON.

    Args:
        eval_dir: Directory produced by `eval_pascal_voc.py`.
        voc_root: Path to `VOCdevkit/VOC2012`.
        output_json: Destination path for diagnostics JSON.

    Returns:
        JSON-serializable diagnostics dictionary.
    """

    eval_dir = Path(eval_dir)
    voc_root = Path(voc_root)
    output_json = Path(output_json)
    if not eval_dir.exists():
        raise FileNotFoundError(f"Evaluation directory does not exist: {eval_dir}")
    if not voc_root.exists():
        raise FileNotFoundError(f"VOC root does not exist: {voc_root}")

    metrics_path = eval_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.json does not exist in evaluation directory: {metrics_path}")

    metrics = _load_json(metrics_path)
    debug_paths = _find_debug_jsons(eval_dir, output_json)
    image_records: List[Dict[str, Any]] = []
    predicted_counts = np.zeros(len(VOC_CLASSES) + 1, dtype=np.int64)
    gt_counts = np.zeros(len(VOC_CLASSES) + 1, dtype=np.int64)
    missed_counts = np.zeros(len(VOC_CLASSES) + 1, dtype=np.int64)
    global_route_counts: Counter[str] = Counter()
    total_valid_pixels = 0

    for debug_path in debug_paths:
        record = _analyze_one_image(debug_path, eval_dir, voc_root)
        debug = _load_json(debug_path)
        image_id = record["image_id"]
        pred_path = _resolve_existing_path(str(debug["outputs"]["mask_npy"]), eval_dir, debug_path)
        pred_voc = map_prediction_to_voc_labels(np.load(pred_path))
        target = _load_target_mask(voc_root / "SegmentationClass" / f"{image_id}.png")
        valid = target != 255
        total_valid_pixels += int(valid.sum())
        _accumulate_global_stats(record, pred_voc, target, valid, predicted_counts, gt_counts, missed_counts)
        global_route_counts.update(record["route_type_counts"])
        image_records.append(record)

    image_records = sorted(image_records, key=lambda record: record["image_id"])
    diagnostics: Dict[str, Any] = {
        "eval_dir": str(eval_dir),
        "voc_root": str(voc_root),
        "metrics": metrics,
        "num_images": len(image_records),
        "images": image_records,
        "summary": {
            "worst_20_images": _ranked_images(image_records, reverse=False),
            "best_20_images": _ranked_images(image_records, reverse=True),
            "most_predicted_classes": _most_predicted_classes(predicted_counts, total_valid_pixels),
            "most_missed_gt_classes": _most_missed_gt_classes(gt_counts, missed_counts),
            "images_with_foreground_over_prediction": _foreground_bias_images(image_records, over=True),
            "images_with_foreground_under_prediction": _foreground_bias_images(image_records, over=False),
            "global_route_type_distribution": dict(sorted(global_route_counts.items())),
        },
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    return diagnostics


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for VOC output diagnostics."""

    parser = argparse.ArgumentParser(description="Analyze UR-OVSS Pascal VOC evaluation outputs.")
    parser.add_argument("--eval-dir", required=True, type=Path, help="Directory produced by eval_pascal_voc.py.")
    parser.add_argument("--voc-root", required=True, type=Path, help="Path to VOCdevkit/VOC2012.")
    parser.add_argument("--output-json", required=True, type=Path, help="Diagnostics JSON output path.")
    return parser


def main() -> None:
    """CLI entry point for VOC diagnostics."""

    args = build_arg_parser().parse_args()
    try:
        diagnostics = analyze_eval_outputs(
            eval_dir=args.eval_dir,
            voc_root=args.voc_root,
            output_json=args.output_json,
        )
    except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"VOC diagnostics error: {exc}") from exc

    summary = diagnostics["summary"]
    print(f"Analyzed images: {diagnostics['num_images']}")
    print(f"Diagnostics JSON: {args.output_json}")
    if summary["worst_20_images"]:
        worst = summary["worst_20_images"][0]
        print(f"Worst image: {worst['image_id']} foreground IoU={worst['foreground_iou']:.6f}")
    if summary["best_20_images"]:
        best = summary["best_20_images"][0]
        print(f"Best image: {best['image_id']} foreground IoU={best['foreground_iou']:.6f}")
    print(f"Route types: {summary['global_route_type_distribution']}")


if __name__ == "__main__":
    main()
