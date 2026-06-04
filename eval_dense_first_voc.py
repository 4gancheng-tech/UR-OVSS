"""Dense-first uncertainty refinement evaluation for Pascal VOC.

This evaluator keeps dense CLIP/ClearCLIP predictions as the base segmentation
and lets SAM/DINO refine only low-confidence dense pixels. It intentionally
does not reuse the older region-first routing path.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from eval_dense_voc import (
    _compute_dense_logits_for_image,
    _image_to_array,
    _load_rgb_image,
    _load_target_mask,
    _prediction_to_confusion,
    _resize_image_for_dense_eval,
    _resize_logits,
    build_dense_adapter,
)
from eval_pascal_voc import VOC21_CLASSES, VOC_CLASSES, read_split_ids, summarize_voc_metrics
from infer_ur_ovss import (
    FeatureBackendError,
    MaskBackendError,
    SemanticBackendError,
    build_feature_adapter,
    build_mask_adapter,
)
from uncertainty_routing import compute_dino_variance


@dataclass
class DenseFirstRefinementResult:
    """Container for one-image dense-first refinement outputs."""

    base_prediction: np.ndarray
    refined_prediction: np.ndarray
    margin: np.ndarray
    entropy: np.ndarray
    confidence: np.ndarray
    uncertain_pixels: np.ndarray
    refined_pixel_ratio: float
    number_of_refined_regions: int
    region_debug: List[Dict[str, Any]]


def _softmax_last_dim(logits: np.ndarray) -> np.ndarray:
    """Compute a stable softmax over the final dimension of `[H, W, C]` logits."""

    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_values = np.exp(shifted)
    total = exp_values.sum(axis=-1, keepdims=True)
    return (exp_values / np.maximum(total, 1e-12)).astype(np.float32)


def compute_dense_pixel_uncertainty(dense_logits: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute pixel margin, entropy, and confidence from dense logits.

    Args:
        dense_logits: Dense semantic logits with shape `[H, W, C]`.

    Returns:
        Tuple `(margin, entropy, confidence)`, each with shape `[H, W]`.
    """

    logits = np.asarray(dense_logits, dtype=np.float32)
    if logits.ndim != 3 or logits.shape[-1] < 2:
        raise ValueError(f"dense_logits must have shape [H, W, C>=2], got {logits.shape}.")

    sorted_logits = np.sort(logits, axis=-1)
    margin = (sorted_logits[..., -1] - sorted_logits[..., -2]).astype(np.float32)
    probabilities = _softmax_last_dim(logits)
    confidence = np.max(probabilities, axis=-1).astype(np.float32)
    entropy = -(probabilities * np.log(np.maximum(probabilities, 1e-12))).sum(axis=-1).astype(np.float32)
    return margin, entropy, confidence


def select_uncertain_pixels(
    margin: np.ndarray,
    entropy: np.ndarray,
    uncertainty_threshold: Optional[float],
    entropy_threshold: Optional[float],
    max_refine_ratio: float,
) -> np.ndarray:
    """Select dense pixels eligible for SAM refinement.

    Args:
        margin: Top1-top2 margin map with shape `[H, W]`.
        entropy: Softmax entropy map with shape `[H, W]`.
        uncertainty_threshold: Pixels with margin less than or equal to this
            value are uncertain. `None` disables margin selection.
        entropy_threshold: Pixels with entropy greater than or equal to this
            value are uncertain. `None` disables entropy selection.
        max_refine_ratio: Upper bound on the fraction of image pixels eligible
            for refinement.

    Returns:
        Boolean uncertainty mask with shape `[H, W]`.
    """

    if margin.shape != entropy.shape:
        raise ValueError(f"margin shape {margin.shape} does not match entropy shape {entropy.shape}.")
    if not 0.0 <= max_refine_ratio <= 1.0:
        raise ValueError(f"max_refine_ratio must be in [0, 1], got {max_refine_ratio}.")
    if max_refine_ratio == 0.0:
        return np.zeros(margin.shape, dtype=bool)

    uncertain = np.zeros(margin.shape, dtype=bool)
    if uncertainty_threshold is not None:
        uncertain |= margin <= float(uncertainty_threshold)
    if entropy_threshold is not None:
        uncertain |= entropy >= float(entropy_threshold)

    selected_count = int(uncertain.sum())
    max_pixels = int(np.floor(margin.size * max_refine_ratio))
    if selected_count == 0 or selected_count <= max_pixels:
        return uncertain
    if max_pixels <= 0:
        return np.zeros(margin.shape, dtype=bool)

    uncertainty_score = entropy - margin
    candidate_indices = np.flatnonzero(uncertain.ravel())
    candidate_scores = uncertainty_score.ravel()[candidate_indices]
    keep_order = np.argsort(candidate_scores, kind="stable")[-max_pixels:]
    capped = np.zeros(margin.size, dtype=bool)
    capped[candidate_indices[keep_order]] = True
    return capped.reshape(margin.shape)


def _validate_refinement_params(max_refine_ratio: float, sam_min_area: int, sam_max_area_ratio: float) -> None:
    """Validate dense-first refinement hyperparameters."""

    if not 0.0 <= max_refine_ratio <= 1.0:
        raise ValueError(f"max_refine_ratio must be in [0, 1], got {max_refine_ratio}.")
    if sam_min_area < 0:
        raise ValueError(f"sam_min_area must be non-negative, got {sam_min_area}.")
    if not 0.0 < sam_max_area_ratio <= 1.0:
        raise ValueError(f"sam_max_area_ratio must be in (0, 1], got {sam_max_area_ratio}.")


def refine_dense_prediction(
    dense_logits: np.ndarray,
    masks: Sequence[Dict[str, Any]],
    dino_features: np.ndarray,
    uncertainty_threshold: Optional[float],
    entropy_threshold: Optional[float],
    max_refine_ratio: float,
    sam_min_area: int,
    sam_max_area_ratio: float,
) -> DenseFirstRefinementResult:
    """Refine dense predictions only inside dense-uncertain SAM pixels.

    Args:
        dense_logits: Dense semantic logits with shape `[H, W, C]`.
        masks: SAM-compatible mask records with boolean `segmentation` arrays
            of shape `[H, W]`.
        dino_features: Dense DINO/DINOv2 feature map with shape `[H, W, D]`.
        uncertainty_threshold: Low-margin threshold. `None` disables it.
        entropy_threshold: High-entropy threshold. `None` disables it.
        max_refine_ratio: Maximum fraction of pixels SAM may refine.
        sam_min_area: Minimum full SAM mask area in pixels.
        sam_max_area_ratio: Maximum full SAM mask area divided by image area.

    Returns:
        DenseFirstRefinementResult with base/refined maps and region debug.
    """

    logits = np.asarray(dense_logits, dtype=np.float32)
    if logits.ndim != 3:
        raise ValueError(f"dense_logits must have shape [H, W, C], got {logits.shape}.")
    _validate_refinement_params(max_refine_ratio, sam_min_area, sam_max_area_ratio)

    height, width, _ = logits.shape
    output_shape = (height, width)
    base_prediction = np.argmax(logits, axis=-1).astype(np.int64)
    margin, entropy, confidence = compute_dense_pixel_uncertainty(logits)
    uncertain_pixels = select_uncertain_pixels(
        margin=margin,
        entropy=entropy,
        uncertainty_threshold=uncertainty_threshold,
        entropy_threshold=entropy_threshold,
        max_refine_ratio=max_refine_ratio,
    )

    refined_prediction = base_prediction.copy()
    refinement_confidence = np.full(output_shape, -np.inf, dtype=np.float32)
    region_debug: List[Dict[str, Any]] = []
    max_area = int(np.floor(height * width * sam_max_area_ratio))
    number_of_refined_regions = 0

    for region_id, mask_record in enumerate(masks):
        if "segmentation" not in mask_record:
            continue
        mask = np.asarray(mask_record["segmentation"], dtype=bool)
        if mask.shape != output_shape:
            raise MaskBackendError(f"Mask shape {mask.shape} does not match dense logits shape {output_shape}.")
        area = int(mask.sum())
        if area < sam_min_area or area > max_area:
            region_debug.append(
                {
                    "region_id": region_id,
                    "source": str(mask_record.get("source", "unknown")),
                    "area": area,
                    "used_for_refinement": False,
                    "skip_reason": "area_filter",
                }
            )
            continue

        eligible_mask = mask & uncertain_pixels
        eligible_area = int(eligible_mask.sum())
        if eligible_area == 0:
            region_debug.append(
                {
                    "region_id": region_id,
                    "source": str(mask_record.get("source", "unknown")),
                    "area": area,
                    "eligible_area": 0,
                    "used_for_refinement": False,
                    "skip_reason": "no_uncertain_pixels",
                }
            )
            continue

        pooled_logits = logits[mask].mean(axis=0)
        region_probabilities = _softmax_last_dim(pooled_logits.reshape(1, 1, -1)).reshape(-1)
        label_id = int(np.argmax(region_probabilities))
        region_confidence = float(region_probabilities[label_id])
        dino_variance = compute_dino_variance(dino_features, mask)
        spatial_consistency = float(1.0 / (1.0 + max(0.0, dino_variance)))
        score = float(region_confidence * spatial_consistency)

        update = eligible_mask & (score > refinement_confidence)
        updated_pixels = int(update.sum())
        if updated_pixels > 0:
            refined_prediction[update] = label_id
            refinement_confidence[update] = score
            number_of_refined_regions += 1

        region_debug.append(
            {
                "region_id": region_id,
                "source": str(mask_record.get("source", "unknown")),
                "area": area,
                "eligible_area": eligible_area,
                "updated_pixels": updated_pixels,
                "label_id": label_id,
                "confidence": region_confidence,
                "dino_variance": float(dino_variance),
                "spatial_consistency": spatial_consistency,
                "refinement_score": score,
                "used_for_refinement": updated_pixels > 0,
            }
        )

    changed = refined_prediction != base_prediction
    refined_pixel_ratio = float(changed.sum() / max(1, height * width))
    return DenseFirstRefinementResult(
        base_prediction=base_prediction,
        refined_prediction=refined_prediction,
        margin=margin,
        entropy=entropy,
        confidence=confidence,
        uncertain_pixels=uncertain_pixels,
        refined_pixel_ratio=refined_pixel_ratio,
        number_of_refined_regions=number_of_refined_regions,
        region_debug=region_debug,
    )


def evaluate_dataset(
    voc_root: Path,
    split: str,
    output_dir: Path,
    limit: Optional[int] = None,
    semantic_backend: str = "clearclip",
    mask_backend: str = "fallback",
    feature_backend: str = "fallback",
    sam_checkpoint: Optional[Path] = None,
    sam_model_type: str = "vit_b",
    max_masks: int = 100,
    dinov2_model: str = "facebook/dinov2-small",
    voc_mode: str = "voc20",
    voc20_ignore_background: bool = False,
    model_name: str = "ViT-B-16",
    pretrained: str = "openai",
    resize_short_side: Optional[int] = 448,
    max_long_side: Optional[int] = 2048,
    slide_crop: int = 448,
    slide_stride: int = 224,
    prompt_ensemble: str = "imagenet",
    text_prototype_average: bool = True,
    uncertainty_threshold: Optional[float] = 0.0,
    entropy_threshold: Optional[float] = None,
    max_refine_ratio: float = 0.20,
    sam_min_area: int = 1,
    sam_max_area_ratio: float = 1.0,
    save_debug: bool = False,
) -> Dict[str, Any]:
    """Evaluate dense-first uncertainty refinement on Pascal VOC."""

    if semantic_backend not in {"clip", "clearclip"}:
        raise ValueError(f"semantic_backend must be 'clip' or 'clearclip', got {semantic_backend!r}.")
    if voc_mode not in {"voc20", "voc21"}:
        raise ValueError(f"voc_mode must be 'voc20' or 'voc21', got {voc_mode!r}.")
    _validate_refinement_params(max_refine_ratio, sam_min_area, sam_max_area_ratio)

    voc_root = Path(voc_root)
    output_dir = Path(output_dir)
    prediction_dir = output_dir / "predictions"
    base_prediction_dir = output_dir / "base_predictions"
    debug_dir = output_dir / "debug"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    base_prediction_dir.mkdir(parents=True, exist_ok=True)
    if save_debug:
        debug_dir.mkdir(parents=True, exist_ok=True)

    image_ids = read_split_ids(voc_root, split)
    if limit is not None:
        image_ids = image_ids[:limit]

    dense_adapter = build_dense_adapter(semantic_backend, model_name=model_name, pretrained=pretrained)
    mask_adapter = build_mask_adapter(
        mask_backend,
        sam_checkpoint=sam_checkpoint,
        sam_model_type=sam_model_type,
        max_masks=max_masks,
    )
    feature_adapter = build_feature_adapter(feature_backend, dinov2_model=dinov2_model)
    print("Dense-first backends are initialized once for this VOC evaluation run.")

    base_confusion = np.zeros((len(VOC21_CLASSES), len(VOC21_CLASSES)), dtype=np.int64)
    refined_confusion = np.zeros_like(base_confusion)
    evaluated_images = 0
    skipped_images = 0
    skipped: List[Dict[str, str]] = []
    total_changed_pixels = 0
    total_pixels = 0
    total_refined_regions = 0
    prediction_files: List[str] = []
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
        eval_image = _resize_image_for_dense_eval(
            image,
            resize_short_side=resize_short_side,
            max_long_side=max_long_side,
        )
        eval_array = _image_to_array(eval_image)
        dense_logits_eval = _compute_dense_logits_for_image(
            adapter=dense_adapter,
            image=eval_image,
            image_array=eval_array,
            class_names=eval_class_names,
            prompt_ensemble=prompt_ensemble,
            text_prototype_average=text_prototype_average,
            slide_crop=slide_crop,
            slide_stride=slide_stride,
        )
        dense_logits = _resize_logits(dense_logits_eval, image_array.shape[:2])
        masks = mask_adapter.generate_masks(image, image_array)
        dino_features = feature_adapter.extract_features(image, image_array)
        refinement = refine_dense_prediction(
            dense_logits=dense_logits,
            masks=masks,
            dino_features=dino_features,
            uncertainty_threshold=uncertainty_threshold,
            entropy_threshold=entropy_threshold,
            max_refine_ratio=max_refine_ratio,
            sam_min_area=sam_min_area,
            sam_max_area_ratio=sam_max_area_ratio,
        )

        prediction_path = prediction_dir / f"{image_id}.npy"
        base_prediction_path = base_prediction_dir / f"{image_id}.npy"
        np.save(prediction_path, refinement.refined_prediction)
        np.save(base_prediction_path, refinement.base_prediction)
        prediction_files.append(str(prediction_path))

        target = _load_target_mask(target_path)
        base_confusion += _prediction_to_confusion(
            refinement.base_prediction,
            target,
            voc_mode=voc_mode,
            voc20_ignore_background=voc20_ignore_background,
        )
        refined_confusion += _prediction_to_confusion(
            refinement.refined_prediction,
            target,
            voc_mode=voc_mode,
            voc20_ignore_background=voc20_ignore_background,
        )

        changed_pixels = int((refinement.refined_prediction != refinement.base_prediction).sum())
        total_changed_pixels += changed_pixels
        total_pixels += int(refinement.base_prediction.size)
        total_refined_regions += refinement.number_of_refined_regions
        evaluated_images += 1

        if save_debug:
            debug_payload = {
                "image_id": image_id,
                "image": str(image_path),
                "target": str(target_path),
                "semantic_backend": semantic_backend,
                "mask_backend": mask_backend,
                "feature_backend": feature_backend,
                "eval_image_size": {"width": eval_image.width, "height": eval_image.height},
                "dense_logits_shape": list(dense_logits.shape),
                "uncertain_pixel_ratio": float(refinement.uncertain_pixels.mean()),
                "refined_pixel_ratio": refinement.refined_pixel_ratio,
                "number_of_refined_regions": refinement.number_of_refined_regions,
                "regions": refinement.region_debug,
                "base_prediction_npy": str(base_prediction_path),
                "prediction_npy": str(prediction_path),
            }
            (debug_dir / f"{image_id}.json").write_text(json.dumps(debug_payload, indent=2), encoding="utf-8")

    if evaluated_images == 0:
        first_skip = f" First skipped item: {skipped[0]['reason']}." if skipped else ""
        raise RuntimeError(
            "Dense-first VOC evaluation did not evaluate any images. "
            f"Skipped {skipped_images} image(s); check --voc-root, --split, and dataset files.{first_skip}"
        )

    base_summary = summarize_voc_metrics(base_confusion, voc_mode=voc_mode)
    refined_summary = summarize_voc_metrics(refined_confusion, voc_mode=voc_mode)
    metrics_path = output_dir / "metrics.json"
    metrics: Dict[str, Any] = {
        "split": split,
        "voc_root": str(voc_root),
        "fusion_mode": "dense_first",
        "semantic_backend": semantic_backend,
        "mask_backend": mask_backend,
        "feature_backend": feature_backend,
        "model_name": model_name,
        "pretrained": pretrained,
        "resize_short_side": resize_short_side,
        "max_long_side": max_long_side,
        "slide_crop": slide_crop,
        "slide_stride": slide_stride,
        "prompt_ensemble": prompt_ensemble,
        "text_prototype_average": bool(text_prototype_average),
        "uncertainty_threshold": uncertainty_threshold,
        "entropy_threshold": entropy_threshold,
        "max_refine_ratio": max_refine_ratio,
        "sam_min_area": sam_min_area,
        "sam_max_area_ratio": sam_max_area_ratio,
        "voc_mode": voc_mode,
        "voc20_ignore_background": bool(voc20_ignore_background),
        "dense_base_mIoU": base_summary["mIoU"],
        "refined_mIoU": refined_summary["mIoU"],
        "mIoU": refined_summary["mIoU"],
        "per_class_iou": refined_summary["per_class_iou"],
        "dense_base_per_class_iou": base_summary["per_class_iou"],
        "background_iou": refined_summary["background_iou"],
        "evaluated_images": evaluated_images,
        "skipped_images": skipped_images,
        "skipped": skipped,
        "refined_pixel_ratio": float(total_changed_pixels / max(1, total_pixels)),
        "number_of_refined_regions": int(total_refined_regions),
        "classes": refined_summary["classes"],
        "predictions_dir": str(prediction_dir),
        "base_predictions_dir": str(base_prediction_dir),
        "prediction_files": prediction_files,
        "debug_dir": str(debug_dir) if save_debug else None,
        "metrics_path": str(metrics_path),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for dense-first VOC evaluation."""

    parser = argparse.ArgumentParser(description="Evaluate dense-first ClearCLIP/SAM/DINO refinement on Pascal VOC.")
    parser.add_argument("--voc-root", required=True, type=Path, help="Path to VOCdevkit/VOC2012.")
    parser.add_argument("--split", default="val", help="VOC segmentation split name.")
    parser.add_argument("--limit", default=None, type=int, help="Optional maximum number of images to evaluate.")
    parser.add_argument(
        "--output-dir",
        default=Path("outputs/dense_first_voc"),
        type=Path,
        help="Directory for metrics.json and per-image prediction npy files.",
    )
    parser.add_argument("--semantic-backend", choices=("clip", "clearclip"), default="clearclip")
    parser.add_argument("--mask-backend", choices=("fallback", "sam"), default="fallback")
    parser.add_argument("--feature-backend", choices=("fallback", "dinov2"), default="fallback")
    parser.add_argument("--sam-checkpoint", default=None, type=Path, help="SAM checkpoint path for --mask-backend sam.")
    parser.add_argument("--sam-model-type", default="vit_b", help="SAM model type key.")
    parser.add_argument("--max-masks", default=100, type=int, help="Maximum number of SAM masks to keep.")
    parser.add_argument("--dinov2-model", default="facebook/dinov2-small", help="DINOv2 model id or local path.")
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
    parser.add_argument("--model-name", default="ViT-B-16", help="open_clip model name.")
    parser.add_argument("--pretrained", default="openai", help="open_clip pretrained tag.")
    parser.add_argument("--resize-short-side", default=448, type=int, help="Dense inference short-side resize.")
    parser.add_argument("--max-long-side", default=2048, type=int, help="Dense inference long-side cap.")
    parser.add_argument("--slide-crop", default=448, type=int, help="Sliding-window crop size. Use 0 to disable.")
    parser.add_argument("--slide-stride", default=224, type=int, help="Sliding-window stride.")
    parser.add_argument("--prompt-ensemble", choices=("imagenet",), default="imagenet")
    parser.add_argument(
        "--text-prototype-average",
        action="store_true",
        default=True,
        help="Average normalized prompt text features per class before dense logits.",
    )
    parser.add_argument(
        "--logit-average",
        dest="text_prototype_average",
        action="store_false",
        help="Use legacy prompt-logit averaging instead of text prototype averaging.",
    )
    parser.add_argument(
        "--uncertainty-threshold",
        default=0.0,
        type=float,
        help="Refine pixels whose dense top1-top2 margin is at or below this value.",
    )
    parser.add_argument(
        "--entropy-threshold",
        default=None,
        type=float,
        help="Optionally refine pixels whose dense softmax entropy is at or above this value.",
    )
    parser.add_argument(
        "--max-refine-ratio",
        default=0.20,
        type=float,
        help="Maximum fraction of image pixels eligible for refinement.",
    )
    parser.add_argument("--sam-min-area", default=1, type=int, help="Minimum SAM mask area in pixels.")
    parser.add_argument(
        "--sam-max-area-ratio",
        default=1.0,
        type=float,
        help="Maximum SAM mask area as a fraction of image pixels.",
    )
    parser.add_argument("--save-debug", action="store_true", help="Save one lightweight debug JSON per image.")
    return parser


def main() -> None:
    """CLI entry point for dense-first Pascal VOC evaluation."""

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
            voc_mode=args.voc_mode,
            voc20_ignore_background=args.voc20_ignore_background,
            model_name=args.model_name,
            pretrained=args.pretrained,
            resize_short_side=args.resize_short_side,
            max_long_side=args.max_long_side,
            slide_crop=args.slide_crop,
            slide_stride=args.slide_stride,
            prompt_ensemble=args.prompt_ensemble,
            text_prototype_average=args.text_prototype_average,
            uncertainty_threshold=args.uncertainty_threshold,
            entropy_threshold=args.entropy_threshold,
            max_refine_ratio=args.max_refine_ratio,
            sam_min_area=args.sam_min_area,
            sam_max_area_ratio=args.sam_max_area_ratio,
            save_debug=args.save_debug,
        )
    except (
        FileNotFoundError,
        ValueError,
        RuntimeError,
        SemanticBackendError,
        MaskBackendError,
        FeatureBackendError,
    ) as exc:
        raise SystemExit(f"Dense-first VOC evaluation error: {exc}") from exc

    print(f"Evaluated images: {metrics['evaluated_images']}")
    print(f"Skipped images: {metrics['skipped_images']}")
    print(f"Dense base mIoU: {metrics['dense_base_mIoU']:.6f}")
    print(f"Refined mIoU: {metrics['refined_mIoU']:.6f}")
    print(f"Refined pixel ratio: {metrics['refined_pixel_ratio']:.6f}")
    print(f"Number of refined regions: {metrics['number_of_refined_regions']}")
    print(f"Metrics JSON: {metrics['metrics_path']}")


if __name__ == "__main__":
    main()
