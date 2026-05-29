"""Uncertainty routing utilities for the UR-OVSS MVP.

This module contains only region-level math and mask fusion. It does not load
CLIP, SAM, DINO, or any heavyweight model, so real experts and lightweight demo
experts can share the same routing path.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np


def _to_numpy(values: Any, dtype: Optional[np.dtype] = np.float32) -> np.ndarray:
    """Convert numpy-like or torch-like values to a numpy array.

    Args:
        values: A numpy array, Python sequence, scalar, or torch tensor.
        dtype: Optional dtype for the returned array.

    Returns:
        Numpy array detached from torch tensors when needed.
    """

    if hasattr(values, "detach") and callable(values.detach):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=dtype)


def _softmax(scores: Sequence[float]) -> np.ndarray:
    """Compute a numerically stable softmax over class scores with shape [C]."""

    logits = _to_numpy(scores, np.float32)
    if logits.ndim != 1:
        raise ValueError(f"scores must have shape [C], got {logits.shape}.")
    shifted = logits - np.max(logits)
    exp_scores = np.exp(shifted)
    total = exp_scores.sum()
    if total <= 0:
        return np.full_like(exp_scores, 1.0 / max(1, exp_scores.size), dtype=np.float32)
    return (exp_scores / total).astype(np.float32)


def compute_semantic_margin(scores: Sequence[float]) -> np.ndarray:
    """Compute top-1 minus top-2 semantic margin.

    Args:
        scores: Class scores with shape [C] for one region or [R, C] for R
            regions.

    Returns:
        A scalar numpy array for one region, or an array with shape [R]. If only
        one class is available, the margin is that class score.
    """

    score_array = _to_numpy(scores, np.float32)
    if score_array.ndim not in (1, 2):
        raise ValueError(f"scores must have shape [C] or [R, C], got {score_array.shape}.")
    if score_array.shape[-1] == 0:
        raise ValueError("scores must include at least one class.")
    if score_array.shape[-1] == 1:
        return score_array[..., 0]

    sorted_scores = np.sort(score_array, axis=-1)
    return np.asarray(sorted_scores[..., -1] - sorted_scores[..., -2], dtype=np.float32)


def compute_dino_variance(dino_features: Sequence[float], mask: Sequence[bool]) -> float:
    """Measure masked DINO feature variance as a spatial uncertainty proxy.

    Args:
        dino_features: Dense or patch-upsampled features with shape [H, W, C]
            or [C, H, W].
        mask: Boolean mask with shape [H, W]. True pixels belong to the region.

    Returns:
        Mean per-channel variance across feature vectors inside the mask. Higher
        values indicate a less pure region. Empty or single-pixel masks return
        0.0.
    """

    features = _to_numpy(dino_features, np.float32)
    region_mask = _to_numpy(mask, bool)
    if region_mask.ndim != 2:
        raise ValueError(f"mask must have shape [H, W], got {region_mask.shape}.")
    if features.ndim != 3:
        raise ValueError(f"dino_features must have shape [H, W, C] or [C, H, W], got {features.shape}.")
    if features.shape[:2] != region_mask.shape:
        if features.shape[1:] == region_mask.shape:
            features = np.moveaxis(features, 0, -1)
        else:
            raise ValueError(
                "dino_features spatial shape must match mask; "
                f"got features {features.shape} and mask {region_mask.shape}."
            )

    masked_features = features[region_mask]
    if masked_features.shape[0] <= 1:
        return 0.0
    return float(masked_features.var(axis=0).mean())


def compute_prompt_uncertainty(prompt_scores: Sequence[float]) -> np.ndarray:
    """Compute prompt-score variance for each class.

    Args:
        prompt_scores: Scores with shape [C, P] for C classes and P positive
            prompts, or shape [P] for one class.

    Returns:
        Per-class variance with shape [C], or a scalar numpy array for one
        class. Higher values indicate stronger prompt sensitivity.
    """

    scores = _to_numpy(prompt_scores, np.float32)
    if scores.ndim == 1:
        return np.asarray(scores.var(), dtype=np.float32)
    if scores.ndim != 2:
        raise ValueError(f"prompt_scores must have shape [P] or [C, P], got {scores.shape}.")
    return scores.var(axis=1).astype(np.float32)


def get_uncertain_regions_by_quantile(values: Sequence[float], ratio: float, mode: str) -> np.ndarray:
    """Select regions in the lowest or highest ratio of uncertainty values.

    Args:
        values: Region values with shape [R].
        ratio: Fraction of regions to mark, e.g. 0.30 for the UR-OVSS default.
        mode: "low" marks the lowest values, used for semantic margins.
            "high" marks the highest values, used for DINO variance.

    Returns:
        Boolean array with shape [R]. The function marks exactly
        ceil(R * ratio) regions when ratio > 0.
    """

    value_array = _to_numpy(values, np.float32)
    if value_array.ndim != 1:
        raise ValueError(f"values must have shape [R], got {value_array.shape}.")
    if mode not in {"low", "high"}:
        raise ValueError('mode must be either "low" or "high".')
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(f"ratio must be in [0, 1], got {ratio}.")

    uncertain = np.zeros(value_array.shape[0], dtype=bool)
    if value_array.size == 0 or ratio == 0.0:
        return uncertain

    count = int(np.ceil(value_array.size * ratio))
    order = np.argsort(value_array, kind="stable")
    selected = order[:count] if mode == "low" else order[-count:]
    uncertain[selected] = True
    return uncertain


def _average_probability_scores(score_sets: Iterable[Sequence[float]]) -> np.ndarray:
    """Average one or more class-score vectors after softmax normalization."""

    probabilities = [_softmax(scores) for scores in score_sets if scores is not None]
    if not probabilities:
        raise ValueError("at least one score vector is required.")
    return np.mean(np.stack(probabilities, axis=0), axis=0).astype(np.float32)


def route_region(
    region_id: int,
    scores: Sequence[float],
    class_names: Sequence[str],
    semantic_uncertain: bool,
    spatial_uncertain: bool,
    dino_variance: float,
    prompt_scores: Optional[Sequence[Sequence[float]]] = None,
    prompt_rescore_scores: Optional[Sequence[float]] = None,
    expert_scores: Optional[Sequence[Sequence[float]]] = None,
) -> Dict[str, Any]:
    """Route one region through the UR-OVSS uncertainty policy.

    Args:
        region_id: Integer region identifier.
        scores: Base semantic scores with shape [C].
        class_names: Class names with length C.
        semantic_uncertain: Whether this region belongs to the low-margin set.
        spatial_uncertain: Whether this region belongs to the high-DINO-variance
            set.
        dino_variance: Region DINO variance scalar.
        prompt_scores: Optional positive prompt scores with shape [C, P], used
            only for debug prompt uncertainty.
        prompt_rescore_scores: Optional prompt-rescored class scores with shape
            [C], used when semantic uncertainty is true.
        expert_scores: Optional additional score vectors, each with shape [C],
            used by the simple multi-expert average route.

    Returns:
        JSON-serializable region debug record containing label, confidence,
        margin, uncertainty flags, and route type.
    """

    base_scores = _to_numpy(scores, np.float32)
    if base_scores.ndim != 1:
        raise ValueError(f"scores must have shape [C], got {base_scores.shape}.")
    if len(class_names) != base_scores.shape[0]:
        raise ValueError(f"class_names length {len(class_names)} does not match scores shape {base_scores.shape}.")

    margin = float(compute_semantic_margin(base_scores))
    notes = []
    score_source = base_scores
    score_source_is_probability = False
    route_type = "direct"

    if semantic_uncertain and not spatial_uncertain:
        score_source = _to_numpy(prompt_rescore_scores, np.float32) if prompt_rescore_scores is not None else base_scores
        route_type = "prompt_rescore"
    elif not semantic_uncertain and spatial_uncertain:
        route_type = "spatial_downweight"
        notes.append("TODO: connect SAM prompt refinement for spatial-uncertain masks.")
    elif semantic_uncertain and spatial_uncertain:
        score_sets = [base_scores]
        if prompt_rescore_scores is not None:
            score_sets.append(_to_numpy(prompt_rescore_scores, np.float32))
        if expert_scores is not None:
            score_sets.extend(expert_scores)
        score_source = _average_probability_scores(score_sets)
        score_source_is_probability = True
        route_type = "multi_expert_average"
        notes.append("TODO: connect SAM prompt refinement for spatial-uncertain masks.")

    probabilities = score_source if score_source_is_probability else _softmax(score_source)
    label_id = int(np.argmax(probabilities))
    confidence = float(probabilities[label_id])
    if spatial_uncertain:
        confidence *= confidence

    prompt_uncertainty = None
    if prompt_scores is not None:
        prompt_variance = compute_prompt_uncertainty(prompt_scores)
        if np.ndim(prompt_variance) == 0:
            prompt_uncertainty = float(prompt_variance)
        else:
            prompt_uncertainty = float(prompt_variance[label_id])

    return {
        "region_id": int(region_id),
        "predicted_label": class_names[label_id],
        "label_id": label_id,
        "semantic_margin": margin,
        "dino_variance": float(dino_variance),
        "prompt_uncertainty": prompt_uncertainty,
        "semantic_uncertain": bool(semantic_uncertain),
        "spatial_uncertain": bool(spatial_uncertain),
        "route_type": route_type,
        "confidence": confidence,
        "notes": notes,
    }


def fuse_region_predictions(
    region_predictions: Sequence[Dict[str, Any]],
    output_shape: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Fuse region labels into a pixel segmentation map by confidence.

    Args:
        region_predictions: Sequence of dictionaries. Each dictionary must have
            a boolean mask with shape [H, W], an integer label_id, and a scalar
            confidence.
        output_shape: Target spatial shape [H, W].

    Returns:
        Tuple `(segmentation, confidence)` where segmentation has shape [H, W]
        and stores label ids, using -1 for unassigned pixels. Confidence has
        shape [H, W] and stores the winning region confidence per pixel.
    """

    segmentation = np.full(output_shape, -1, dtype=np.int32)
    confidence_map = np.full(output_shape, -np.inf, dtype=np.float32)

    for region in region_predictions:
        mask = _to_numpy(region["mask"], bool)
        if mask.shape != output_shape:
            raise ValueError(f"region mask shape {mask.shape} does not match output shape {output_shape}.")
        confidence = float(region["confidence"])
        update = mask & (confidence > confidence_map)
        segmentation[update] = int(region["label_id"])
        confidence_map[update] = confidence

    return segmentation, confidence_map
