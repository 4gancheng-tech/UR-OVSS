"""Minimal runnable UR-OVSS inference demo.

This script keeps the full uncertainty-routing loop runnable in an empty
repository. Because no CLIP/ClearCLIP, SAM, or DINO implementation exists here
yet, it uses deterministic lightweight proxy experts by default and records that
choice in the debug JSON. The routing interfaces are model-agnostic so real
experts can replace the proxy functions later without changing the fusion logic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - exercised only in missing-dependency environments.
    raise SystemExit("UR-OVSS MVP requires numpy. Install it with: pip install numpy") from exc

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover - exercised only in missing-dependency environments.
    raise SystemExit("UR-OVSS MVP requires Pillow for image I/O. Install it with: pip install pillow") from exc

from prompts import build_negative_prompts, build_positive_prompts, compute_prompt_rescore
from uncertainty_routing import (
    compute_dino_variance,
    compute_semantic_margin,
    fuse_region_predictions,
    get_uncertain_regions_by_quantile,
    route_region,
)


RHO_SEM = 0.30
RHO_SPA = 0.30
NEGATIVE_PROMPT_SUPPRESSION_ALPHA = 0.30


def parse_class_names(classes: str) -> List[str]:
    """Parse a comma-separated open-vocabulary class list.

    Args:
        classes: Comma-separated string such as "cat,dog,person,car".

    Returns:
        Clean class-name list with length C.
    """

    class_names = [name.strip() for name in classes.split(",") if name.strip()]
    if not class_names:
        raise SystemExit("--classes must contain at least one non-empty class name.")
    return class_names


def load_rgb_image(path: Path) -> Tuple[Image.Image, np.ndarray]:
    """Load an image as RGB PIL image and normalized numpy array.

    Args:
        path: Image path.

    Returns:
        Tuple of `(pil_image, image_array)`. The array has shape [H, W, 3] and
        float32 values in [0, 1].
    """

    if not path.exists():
        raise SystemExit(f"Image not found: {path}")
    image = Image.open(path).convert("RGB")
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    return image, image_array


def normalize_last_dim(features: np.ndarray) -> np.ndarray:
    """L2-normalize features along the final dimension.

    Args:
        features: Feature tensor with shape [..., D].

    Returns:
        Normalized feature tensor with shape [..., D].
    """

    norm = np.linalg.norm(features, axis=-1, keepdims=True)
    return features / np.maximum(norm, 1e-6)


def build_dense_proxy_features(image_array: np.ndarray) -> np.ndarray:
    """Create lightweight dense image features for the fallback semantic expert.

    Args:
        image_array: RGB image with shape [H, W, 3] and values in [0, 1].

    Returns:
        Dense feature tensor with shape [H, W, 8]. The dimensions contain RGB,
        luminance, saturation, normalized x/y coordinates, and edge magnitude.
    """

    height, width, _ = image_array.shape
    red = image_array[..., 0]
    green = image_array[..., 1]
    blue = image_array[..., 2]
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    saturation = image_array.max(axis=2) - image_array.min(axis=2)

    y_coords = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    x_coords = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    x_map = np.broadcast_to(x_coords, (height, width))
    y_map = np.broadcast_to(y_coords, (height, width))

    if height > 1 and width > 1:
        grad_y, grad_x = np.gradient(luminance)
        edge = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    else:
        edge = np.zeros_like(luminance, dtype=np.float32)
    if float(edge.max()) > 0.0:
        edge = edge / edge.max()

    features = np.stack(
        [red, green, blue, luminance, saturation, x_map, y_map, edge.astype(np.float32)],
        axis=-1,
    ).astype(np.float32)
    return normalize_last_dim(features)


def stable_text_embedding(text: str, dim: int) -> np.ndarray:
    """Build a deterministic text embedding for fallback scoring.

    Args:
        text: Prompt or class-name text.
        dim: Embedding dimension D.

    Returns:
        L2-normalized vector with shape [D].
    """

    values: List[float] = []
    counter = 0
    while len(values) < dim:
        digest = hashlib.sha256(f"{text}::{counter}".encode("utf-8")).digest()
        values.extend((byte / 127.5) - 1.0 for byte in digest)
        counter += 1
    embedding = np.asarray(values[:dim], dtype=np.float32)
    return normalize_last_dim(embedding)


def score_texts_against_features(features: np.ndarray, texts: Sequence[str]) -> np.ndarray:
    """Score text embeddings against dense image features.

    Args:
        features: Dense feature tensor with shape [H, W, D].
        texts: Text prompts with length T.

    Returns:
        Dense scores with shape [H, W, T].
    """

    embeddings = np.stack([stable_text_embedding(text, features.shape[-1]) for text in texts], axis=0)
    return np.tensordot(features, embeddings.T, axes=([-1], [0])).astype(np.float32)


def score_texts_against_vector(feature: np.ndarray, texts: Sequence[str]) -> np.ndarray:
    """Score text embeddings against one region-level feature prototype.

    Args:
        feature: Region prototype with shape [D].
        texts: Text prompts with length T.

    Returns:
        Region-level text scores with shape [T].
    """

    embeddings = np.stack([stable_text_embedding(text, feature.shape[-1]) for text in texts], axis=0)
    return np.dot(embeddings, feature).astype(np.float32)


def generate_fallback_masks(image_array: np.ndarray) -> List[Dict[str, Any]]:
    """Generate class-agnostic candidate masks when SAM is unavailable.

    Args:
        image_array: RGB image with shape [H, W, 3] and values in [0, 1].

    Returns:
        List of mask dictionaries. Each `segmentation` entry has shape [H, W].
    """

    height, width, _ = image_array.shape
    masks: List[Dict[str, Any]] = []

    grid = 4 if min(height, width) >= 64 else 2
    for row in range(grid):
        y0 = int(round(row * height / grid))
        y1 = int(round((row + 1) * height / grid))
        for col in range(grid):
            x0 = int(round(col * width / grid))
            x1 = int(round((col + 1) * width / grid))
            mask = np.zeros((height, width), dtype=bool)
            mask[y0:y1, x0:x1] = True
            masks.append({"segmentation": mask, "source": f"fallback_grid_{row}_{col}"})

    luminance = 0.299 * image_array[..., 0] + 0.587 * image_array[..., 1] + 0.114 * image_array[..., 2]
    saturation = image_array.max(axis=2) - image_array.min(axis=2)
    candidate_masks = {
        "fallback_bright": luminance >= np.quantile(luminance, 0.70),
        "fallback_dark": luminance <= np.quantile(luminance, 0.30),
        "fallback_saturated": saturation >= np.quantile(saturation, 0.75),
    }
    center = np.zeros((height, width), dtype=bool)
    center[height // 4 : max(height // 4 + 1, 3 * height // 4), width // 4 : max(width // 4 + 1, 3 * width // 4)] = True
    candidate_masks["fallback_center"] = center

    min_area = max(1, int(0.01 * height * width))
    for source, mask in candidate_masks.items():
        if int(mask.sum()) >= min_area:
            masks.append({"segmentation": mask.astype(bool), "source": source})

    return masks


def build_patch_proxy_features(image_array: np.ndarray) -> np.ndarray:
    """Create patch-level proxy features for the fallback DINO expert.

    Args:
        image_array: RGB image with shape [H, W, 3] and values in [0, 1].

    Returns:
        Patch-averaged feature tensor upsampled to shape [H, W, 8].
    """

    dense_features = build_dense_proxy_features(image_array)
    height, width, channels = dense_features.shape
    patch = max(4, min(height, width) // 16)
    patch_features = np.zeros_like(dense_features)

    for y0 in range(0, height, patch):
        y1 = min(height, y0 + patch)
        for x0 in range(0, width, patch):
            x1 = min(width, x0 + patch)
            prototype = dense_features[y0:y1, x0:x1].mean(axis=(0, 1))
            patch_features[y0:y1, x0:x1] = prototype

    return normalize_last_dim(patch_features.reshape(-1, channels)).reshape(height, width, channels)


def pool_mask_scores(dense_scores: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Pool dense class or prompt scores inside a region mask.

    Args:
        dense_scores: Score tensor with shape [H, W, ...].
        mask: Boolean region mask with shape [H, W].

    Returns:
        Mean score tensor with shape [...]. Empty masks return zeros.
    """

    if dense_scores.shape[:2] != mask.shape:
        raise ValueError(f"dense score shape {dense_scores.shape[:2]} does not match mask shape {mask.shape}.")
    if not bool(mask.any()):
        return np.zeros(dense_scores.shape[2:], dtype=np.float32)
    return dense_scores[mask].mean(axis=0).astype(np.float32)


def pool_mask_features(features: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Pool dense features into a normalized region-level prototype.

    Args:
        features: Dense feature tensor with shape [H, W, D].
        mask: Boolean region mask with shape [H, W].

    Returns:
        Normalized feature prototype with shape [D]. Empty masks return zeros.
    """

    if features.shape[:2] != mask.shape:
        raise ValueError(f"feature shape {features.shape[:2]} does not match mask shape {mask.shape}.")
    if not bool(mask.any()):
        return np.zeros(features.shape[-1], dtype=np.float32)
    return normalize_last_dim(features[mask].mean(axis=0)).astype(np.float32)


def compute_dino_region_scores(dino_features: np.ndarray, mask: np.ndarray, class_names: Sequence[str]) -> np.ndarray:
    """Score a DINO region prototype against deterministic class embeddings.

    Args:
        dino_features: Patch-level feature tensor with shape [H, W, D].
        mask: Boolean region mask with shape [H, W].
        class_names: Class names with length C.

    Returns:
        Region-level fallback DINO scores with shape [C].
    """

    if not bool(mask.any()):
        return np.zeros(len(class_names), dtype=np.float32)
    prototype = normalize_last_dim(dino_features[mask].mean(axis=0))
    class_embeddings = np.stack(
        [stable_text_embedding(f"dino region containing {name}", dino_features.shape[-1]) for name in class_names],
        axis=0,
    )
    return np.dot(class_embeddings, prototype).astype(np.float32)


def make_palette(num_classes: int) -> np.ndarray:
    """Create a deterministic RGB palette for visualization.

    Args:
        num_classes: Number of classes C.

    Returns:
        Palette array with shape [C, 3] and uint8 RGB values.
    """

    colors = []
    for index in range(num_classes):
        digest = hashlib.sha256(f"class-color-{index}".encode("utf-8")).digest()
        colors.append([64 + digest[0] % 160, 64 + digest[1] % 160, 64 + digest[2] % 160])
    return np.asarray(colors, dtype=np.uint8)


def colorize_segmentation(image_array: np.ndarray, segmentation: np.ndarray, num_classes: int) -> Image.Image:
    """Blend a predicted segmentation map over the input image.

    Args:
        image_array: RGB image with shape [H, W, 3] and values in [0, 1].
        segmentation: Label map with shape [H, W], using -1 for unassigned.
        num_classes: Number of classes C.

    Returns:
        RGB visualization as a PIL image.
    """

    base = (image_array * 255.0).clip(0, 255).astype(np.uint8)
    overlay = np.zeros_like(base)
    palette = make_palette(num_classes)
    assigned = segmentation >= 0
    if assigned.any():
        overlay[assigned] = palette[segmentation[assigned]]
    blended = base.copy()
    blended[assigned] = (0.55 * base[assigned] + 0.45 * overlay[assigned]).astype(np.uint8)
    return Image.fromarray(blended, mode="RGB")


def save_label_png(segmentation: np.ndarray, output_path: Path, num_classes: int) -> Path:
    """Save an indexed-color label map as a PNG next to the visualization.

    Args:
        segmentation: Label map with shape [H, W], using -1 for unassigned.
        output_path: Visualization output path.
        num_classes: Number of classes C.

    Returns:
        Path to the saved label PNG.
    """

    label_path = output_path.with_name(f"{output_path.stem}_labels.png")
    palette = make_palette(num_classes)
    label_rgb = np.zeros((*segmentation.shape, 3), dtype=np.uint8)
    assigned = segmentation >= 0
    if assigned.any():
        label_rgb[assigned] = palette[segmentation[assigned]]
    Image.fromarray(label_rgb, mode="RGB").save(label_path)
    return label_path


def run_inference(image_path: Path, class_names: Sequence[str], output_path: Path) -> Dict[str, Any]:
    """Run the UR-OVSS MVP loop and save visualization, mask, and JSON.

    Args:
        image_path: Input image path.
        class_names: Open-vocabulary classes with length C.
        output_path: PNG visualization path.

    Returns:
        Dictionary containing output paths and region debug records.
    """

    pil_image, image_array = load_rgb_image(image_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    positive_prompts = build_positive_prompts(class_names)
    negative_prompts = build_negative_prompts(class_names)
    positive_flat = [prompt for class_name in class_names for prompt in positive_prompts[class_name]]
    negative_flat = [prompt for class_name in class_names for prompt in negative_prompts[class_name]]

    dense_features = build_dense_proxy_features(image_array)
    class_prompt_texts = [positive_prompts[class_name][0] for class_name in class_names]

    height, width = image_array.shape[:2]
    num_positive_prompts = len(next(iter(positive_prompts.values())))
    num_negative_prompts = len(next(iter(negative_prompts.values())))

    masks = generate_fallback_masks(image_array)
    dino_features = build_patch_proxy_features(image_array)

    region_work: List[Dict[str, Any]] = []
    for region_id, mask_record in enumerate(masks):
        mask = mask_record["segmentation"]
        clip_prototype = pool_mask_features(dense_features, mask)
        base_scores = score_texts_against_vector(clip_prototype, class_prompt_texts)
        positive_scores = score_texts_against_vector(clip_prototype, positive_flat).reshape(
            len(class_names),
            num_positive_prompts,
        )
        negative_scores = score_texts_against_vector(clip_prototype, negative_flat).reshape(
            len(class_names),
            num_negative_prompts,
        )
        prompt_rescore_scores = compute_prompt_rescore(
            base_scores,
            positive_scores,
            negative_scores,
            alpha=NEGATIVE_PROMPT_SUPPRESSION_ALPHA,
        )
        dino_scores = compute_dino_region_scores(dino_features, mask, class_names)
        region_work.append(
            {
                "region_id": region_id,
                "mask": mask,
                "area": int(mask.sum()),
                "source": mask_record["source"],
                "clip_prototype": clip_prototype,
                "base_scores": base_scores,
                "positive_scores": positive_scores,
                "prompt_rescore_scores": prompt_rescore_scores,
                "dino_scores": dino_scores,
                "dino_variance": compute_dino_variance(dino_features, mask),
            }
        )

    base_score_matrix = np.stack([region["base_scores"] for region in region_work], axis=0)
    semantic_margins = compute_semantic_margin(base_score_matrix)
    dino_variances = np.asarray([region["dino_variance"] for region in region_work], dtype=np.float32)
    semantic_uncertain = get_uncertain_regions_by_quantile(semantic_margins, RHO_SEM, mode="low")
    spatial_uncertain = get_uncertain_regions_by_quantile(dino_variances, RHO_SPA, mode="high")

    fused_regions: List[Dict[str, Any]] = []
    debug_regions: List[Dict[str, Any]] = []
    for index, region in enumerate(region_work):
        routed = route_region(
            region_id=region["region_id"],
            scores=region["base_scores"],
            class_names=class_names,
            semantic_uncertain=bool(semantic_uncertain[index]),
            spatial_uncertain=bool(spatial_uncertain[index]),
            dino_variance=region["dino_variance"],
            prompt_scores=region["positive_scores"],
            prompt_rescore_scores=region["prompt_rescore_scores"],
            expert_scores=[region["dino_scores"]],
        )
        routed["area"] = region["area"]
        routed["source"] = region["source"]
        fused_regions.append({**routed, "mask": region["mask"]})
        debug_regions.append(routed)

    segmentation, confidence_map = fuse_region_predictions(fused_regions, output_shape=(height, width))
    visualization = colorize_segmentation(image_array, segmentation, len(class_names))
    visualization.save(output_path)

    mask_path = output_path.with_name(f"{output_path.stem}_mask.npy")
    confidence_path = output_path.with_name(f"{output_path.stem}_confidence.npy")
    json_path = output_path.with_suffix(".json")
    label_png_path = save_label_png(segmentation, output_path, len(class_names))

    np.save(mask_path, segmentation)
    np.save(confidence_path, confidence_map)

    debug_payload: Dict[str, Any] = {
        "image": str(image_path),
        "image_size": {"width": pil_image.width, "height": pil_image.height},
        "class_names": list(class_names),
        "routing": {
            "rho_sem": RHO_SEM,
            "rho_spa": RHO_SPA,
            "negative_prompt_suppression_alpha": NEGATIVE_PROMPT_SUPPRESSION_ALPHA,
        },
        "experts": {
            "semantic": "fallback dense proxy logits; replace with CLIP/ClearCLIP dense logits when available",
            "spatial": "fallback class-agnostic masks; replace with SAM AutomaticMaskGenerator when available",
            "purity": "fallback patch proxy features; replace with DINO features when available",
            "text": "template positive/negative prompts, no external LLM",
        },
        "positive_prompts": positive_prompts,
        "negative_prompts": negative_prompts,
        "regions": debug_regions,
        "outputs": {
            "visualization": str(output_path),
            "label_png": str(label_png_path),
            "mask_npy": str(mask_path),
            "confidence_npy": str(confidence_path),
            "debug_json": str(json_path),
        },
    }
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(debug_payload, handle, indent=2)

    return debug_payload


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for the UR-OVSS demo."""

    parser = argparse.ArgumentParser(description="Run the UR-OVSS MVP inference demo.")
    parser.add_argument("--image", required=True, type=Path, help="Path to an input image.")
    parser.add_argument("--classes", required=True, type=str, help='Comma-separated classes, e.g. "cat,dog,person".')
    parser.add_argument("--output", required=True, type=Path, help="Path for the output visualization PNG.")
    return parser


def main() -> None:
    """CLI entry point for UR-OVSS MVP inference."""

    args = build_arg_parser().parse_args()
    class_names = parse_class_names(args.classes)
    result = run_inference(args.image, class_names, args.output)
    outputs = result["outputs"]
    print("UR-OVSS MVP inference complete.")
    print(f"Visualization: {outputs['visualization']}")
    print(f"Label PNG: {outputs['label_png']}")
    print(f"Mask NPY: {outputs['mask_npy']}")
    print(f"Confidence NPY: {outputs['confidence_npy']}")
    print(f"Debug JSON: {outputs['debug_json']}")
    print("Note: this empty repository used deterministic fallback experts, not real CLIP/SAM/DINO weights.")


if __name__ == "__main__":
    main()
