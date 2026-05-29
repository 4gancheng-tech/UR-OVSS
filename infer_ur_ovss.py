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
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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


class SemanticBackendError(RuntimeError):
    """Raised when a semantic expert backend cannot be initialized or used."""


class MaskBackendError(RuntimeError):
    """Raised when a mask generation backend cannot be initialized or used."""


@dataclass
class RegionSemanticScores:
    """Container for region-level semantic scores.

    Attributes:
        base_scores: Base class scores with shape [C].
        positive_scores: Positive prompt scores with shape [C, P].
        negative_scores: Negative prompt scores with shape [C, N].
        prompt_rescore_scores: Prompt-rescored class scores with shape [C].
    """

    base_scores: np.ndarray
    positive_scores: np.ndarray
    negative_scores: np.ndarray
    prompt_rescore_scores: np.ndarray


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
    if len(class_names) < 2:
        raise SystemExit("--classes must contain at least two class names for top1-top2 semantic margin.")
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


class FallbackMaskAdapter:
    """Deterministic class-agnostic mask adapter used by the default MVP path."""

    description = "fallback class-agnostic masks"

    def __init__(self, max_masks: Optional[int] = None) -> None:
        """Initialize fallback mask generation.

        Args:
            max_masks: Optional maximum number of masks to keep.
        """

        self.max_masks = max_masks

    def generate_masks(self, image: Image.Image, image_array: np.ndarray) -> List[Dict[str, Any]]:
        """Generate fallback mask records compatible with SAM output.

        Args:
            image: RGB PIL image. Unused by this backend.
            image_array: RGB image with shape [H, W, 3] and values in [0, 1].

        Returns:
            List of mask dictionaries with bool `segmentation` arrays of shape
            [H, W] and string `source` fields.
        """

        del image
        masks = generate_fallback_masks(image_array)
        if self.max_masks is not None:
            masks = masks[: self.max_masks]
        return masks


class SamMaskAdapter:
    """Optional SAM/MobileSAM class-agnostic mask adapter."""

    description = "SAM/MobileSAM AutomaticMaskGenerator masks"

    def __init__(
        self,
        checkpoint_path: Optional[Path],
        model_type: str = "vit_b",
        device: Optional[str] = None,
        max_masks: Optional[int] = 100,
        sam_module: Optional[Any] = None,
    ) -> None:
        """Load a SAM or MobileSAM automatic mask generator.

        Args:
            checkpoint_path: Path to SAM/MobileSAM checkpoint. Must be provided
                by the user and is never downloaded into this repository.
            model_type: SAM model type key, e.g. "vit_b".
            device: Device string such as "cpu" or "cuda".
            max_masks: Optional maximum number of masks to keep.
            sam_module: Optional module-like object for tests.
        """

        if checkpoint_path is None:
            raise MaskBackendError("--sam-checkpoint is required when --mask-backend sam is selected.")

        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise MaskBackendError(f"SAM checkpoint does not exist: {self.checkpoint_path}")

        self.model_type = model_type
        self.device = device or self._default_device()
        self.max_masks = max_masks
        self.sam_module = sam_module or self._import_sam_module()
        registry = getattr(self.sam_module, "sam_model_registry", None)
        generator_cls = getattr(self.sam_module, "SamAutomaticMaskGenerator", None)
        if registry is None or generator_cls is None:
            raise MaskBackendError(
                "SAM dependency does not expose sam_model_registry and SamAutomaticMaskGenerator."
            )
        if self.model_type not in registry:
            available = ", ".join(sorted(registry.keys()))
            raise MaskBackendError(f"Unknown SAM model type {self.model_type!r}. Available model types: {available}")

        try:
            model = registry[self.model_type](checkpoint=str(self.checkpoint_path))
            if hasattr(model, "to"):
                model = model.to(device=self.device)
            if hasattr(model, "eval"):
                model.eval()
            self.generator = generator_cls(model)
        except Exception as exc:
            raise MaskBackendError(
                "Failed to initialize SAM mask generator. Check that --sam-checkpoint matches "
                f"--sam-model-type {self.model_type!r} and that SAM/MobileSAM dependencies are installed."
            ) from exc

    def _default_device(self) -> str:
        """Choose CUDA when torch is installed and a GPU is visible."""

        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _import_sam_module(self) -> Any:
        """Import segment_anything or mobile_sam with a clear error."""

        try:
            import segment_anything

            return segment_anything
        except ImportError as segment_error:
            try:
                import mobile_sam

                return mobile_sam
            except ImportError as mobile_error:
                message = (
                    "The SAM mask backend requires segment-anything or mobile-sam. "
                    "Install optional dependencies with `pip install -r requirements-sam.txt`."
                )
                raise MaskBackendError(message) from mobile_error

    def generate_masks(self, image: Image.Image, image_array: np.ndarray) -> List[Dict[str, Any]]:
        """Generate SAM masks in the MVP-compatible mask record format.

        Args:
            image: RGB PIL image. Unused because SAM expects numpy RGB.
            image_array: RGB image with shape [H, W, 3] and values in [0, 1].

        Returns:
            List of mask dictionaries with bool `segmentation` arrays of shape
            [H, W] and string `source` fields.
        """

        del image
        rgb_uint8 = (image_array * 255.0).clip(0, 255).astype(np.uint8)
        try:
            raw_masks = self.generator.generate(rgb_uint8)
        except Exception as exc:
            raise MaskBackendError(f"SAM mask generation failed: {exc}") from exc

        masks: List[Dict[str, Any]] = []
        for raw_mask in raw_masks:
            if "segmentation" not in raw_mask:
                continue
            segmentation = np.asarray(raw_mask["segmentation"], dtype=bool)
            if segmentation.shape != image_array.shape[:2]:
                raise MaskBackendError(
                    f"SAM mask shape {segmentation.shape} does not match image shape {image_array.shape[:2]}."
                )
            mask_record = dict(raw_mask)
            mask_record["segmentation"] = segmentation
            mask_record["source"] = f"sam_{self.model_type}"
            masks.append(mask_record)
            if self.max_masks is not None and len(masks) >= self.max_masks:
                break
        return masks


def validate_max_masks(max_masks: Optional[int]) -> None:
    """Validate that max_masks is absent or a positive integer."""

    if max_masks is not None and max_masks <= 0:
        raise MaskBackendError(f"--max-masks must be a positive integer, got {max_masks}.")


def build_mask_adapter(
    backend: str,
    sam_checkpoint: Optional[Path] = None,
    sam_model_type: str = "vit_b",
    max_masks: Optional[int] = 100,
) -> Any:
    """Build the requested class-agnostic mask generation adapter.

    Args:
        backend: Mask backend name, either "fallback" or "sam".
        sam_checkpoint: Optional SAM checkpoint path for the "sam" backend.
        sam_model_type: SAM model type key.
        max_masks: Optional maximum number of masks to keep.

    Returns:
        Adapter object exposing generate_masks().
    """

    validate_max_masks(max_masks)
    if backend == "fallback":
        return FallbackMaskAdapter(max_masks=max_masks)
    if backend == "sam":
        return SamMaskAdapter(
            checkpoint_path=sam_checkpoint,
            model_type=sam_model_type,
            max_masks=max_masks,
        )
    raise ValueError(f"Unknown mask backend {backend!r}; expected 'fallback' or 'sam'.")


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


def tensor_to_numpy(values: Any) -> np.ndarray:
    """Convert tensor-like CLIP outputs to float32 numpy arrays.

    Args:
        values: Torch tensor, fake tensor from tests, or numpy-like array.

    Returns:
        Float32 numpy array with the same logical shape.
    """

    if hasattr(values, "detach") and callable(values.detach):
        values = values.detach()
    if hasattr(values, "cpu") and callable(values.cpu):
        values = values.cpu()
    if hasattr(values, "numpy") and callable(values.numpy):
        values = values.numpy()
    return np.asarray(values, dtype=np.float32)


def clip_inference_context() -> Any:
    """Return a no-gradient context for CLIP inference when torch is available."""

    try:
        import torch

        return torch.inference_mode()
    except Exception:
        return nullcontext()


def flatten_prompt_dict(prompt_dict: Dict[str, List[str]], class_names: Sequence[str]) -> List[str]:
    """Flatten a class-to-prompts mapping in class order.

    Args:
        prompt_dict: Mapping from class name to prompt list.
        class_names: Class names with length C.

    Returns:
        Prompt list with length C * P.
    """

    return [prompt for class_name in class_names for prompt in prompt_dict[class_name]]


class FallbackSemanticAdapter:
    """Deterministic CLIP-like semantic adapter used by the default MVP path."""

    description = "fallback dense proxy logits"

    def __init__(self) -> None:
        """Initialize an empty fallback semantic adapter."""

        self.dense_features: Optional[np.ndarray] = None

    def prepare_image(self, image: Image.Image, image_array: np.ndarray) -> None:
        """Precompute dense proxy features for one image.

        Args:
            image: RGB PIL image. Unused by the fallback backend.
            image_array: RGB image with shape [H, W, 3] and values in [0, 1].
        """

        del image
        self.dense_features = build_dense_proxy_features(image_array)

    def score_region(
        self,
        mask: np.ndarray,
        class_names: Sequence[str],
        positive_prompts: Dict[str, List[str]],
        negative_prompts: Dict[str, List[str]],
    ) -> RegionSemanticScores:
        """Score one region with deterministic CLIP-like image/text features.

        Args:
            mask: Boolean region mask with shape [H, W].
            class_names: Open-vocabulary classes with length C.
            positive_prompts: Positive prompts, C classes by P prompts.
            negative_prompts: Negative prompts, C classes by N prompts.

        Returns:
            Region semantic scores with shapes [C], [C, P], [C, N], and [C].
        """

        if self.dense_features is None:
            raise SemanticBackendError("Fallback semantic backend was used before prepare_image().")

        clip_prototype = pool_mask_features(self.dense_features, mask)
        class_prompt_texts = [positive_prompts[class_name][0] for class_name in class_names]
        positive_flat = flatten_prompt_dict(positive_prompts, class_names)
        negative_flat = flatten_prompt_dict(negative_prompts, class_names)
        num_positive_prompts = len(positive_prompts[class_names[0]])
        num_negative_prompts = len(negative_prompts[class_names[0]])

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
        return RegionSemanticScores(
            base_scores=base_scores,
            positive_scores=positive_scores,
            negative_scores=negative_scores,
            prompt_rescore_scores=prompt_rescore_scores,
        )


class OpenClipSemanticAdapter:
    """Optional real CLIP semantic adapter backed by open_clip."""

    description = "open_clip region crop image/text similarity"

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: Optional[str] = None,
        open_clip_module: Optional[Any] = None,
    ) -> None:
        """Load an open_clip model for region-level semantic scoring.

        Args:
            model_name: open_clip model name.
            pretrained: open_clip pretrained weights tag.
            device: Device string such as "cpu" or "cuda". Defaults to CUDA
                when torch reports CUDA availability, otherwise CPU.
            open_clip_module: Optional module-like object for tests.
        """

        self.device = device or self._default_device()
        self.image: Optional[Image.Image] = None
        self.image_array: Optional[np.ndarray] = None
        if open_clip_module is None:
            try:
                import open_clip as open_clip_module  # type: ignore[import-not-found]
            except ImportError as exc:
                raise SemanticBackendError(
                    "The CLIP semantic backend requires open_clip. Install optional dependencies with "
                    "`pip install open_clip_torch` or `pip install -r requirements-clip.txt`."
                ) from exc

        self.open_clip = open_clip_module
        try:
            self.model, _, self.preprocess = self.open_clip.create_model_and_transforms(
                model_name,
                pretrained=pretrained,
                device=self.device,
            )
            self.model.eval()
            self.tokenizer = self.open_clip.get_tokenizer(model_name)
        except Exception as exc:
            raise SemanticBackendError(
                "Failed to load open_clip model "
                f"{model_name!r} with pretrained={pretrained!r} on device {self.device!r}. "
                "open_clip may download weights to its normal user cache, not this repository; "
                "check network access and installed torch/open_clip versions."
            ) from exc

    def _default_device(self) -> str:
        """Choose CUDA when torch is installed and a GPU is visible."""

        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def prepare_image(self, image: Image.Image, image_array: np.ndarray) -> None:
        """Store the image used for region crop scoring.

        Args:
            image: RGB PIL image.
            image_array: RGB image with shape [H, W, 3] and values in [0, 1].
        """

        self.image = image
        self.image_array = image_array

    def score_region(
        self,
        mask: np.ndarray,
        class_names: Sequence[str],
        positive_prompts: Dict[str, List[str]],
        negative_prompts: Dict[str, List[str]],
    ) -> RegionSemanticScores:
        """Score one region crop against positive and negative CLIP prompts.

        Args:
            mask: Boolean region mask with shape [H, W].
            class_names: Open-vocabulary classes with length C.
            positive_prompts: Positive prompts, C classes by P prompts.
            negative_prompts: Negative prompts, C classes by N prompts.

        Returns:
            Region semantic scores with shapes [C], [C, P], [C, N], and [C].
        """

        if self.image is None or self.image_array is None:
            raise SemanticBackendError("CLIP semantic backend was used before prepare_image().")

        crop = self._masked_region_crop(mask)
        class_prompt_texts = [positive_prompts[class_name][0] for class_name in class_names]
        positive_flat = flatten_prompt_dict(positive_prompts, class_names)
        negative_flat = flatten_prompt_dict(negative_prompts, class_names)
        num_positive_prompts = len(positive_prompts[class_names[0]])
        num_negative_prompts = len(negative_prompts[class_names[0]])

        try:
            base_scores = self._score_prompts(crop, class_prompt_texts)
            positive_scores = self._score_prompts(crop, positive_flat).reshape(
                len(class_names),
                num_positive_prompts,
            )
            negative_scores = self._score_prompts(crop, negative_flat).reshape(
                len(class_names),
                num_negative_prompts,
            )
        except Exception as exc:
            raise SemanticBackendError(f"Failed to score a region with open_clip: {exc}") from exc

        prompt_rescore_scores = compute_prompt_rescore(
            base_scores,
            positive_scores,
            negative_scores,
            alpha=NEGATIVE_PROMPT_SUPPRESSION_ALPHA,
        )
        return RegionSemanticScores(
            base_scores=base_scores,
            positive_scores=positive_scores,
            negative_scores=negative_scores,
            prompt_rescore_scores=prompt_rescore_scores,
        )

    def _masked_region_crop(self, mask: np.ndarray) -> Image.Image:
        """Crop the image around a mask and zero pixels outside the mask.

        Args:
            mask: Boolean region mask with shape [H, W].

        Returns:
            RGB PIL crop for CLIP image encoding.
        """

        if self.image is None or self.image_array is None:
            raise SemanticBackendError("CLIP semantic backend has no prepared image.")

        region_mask = np.asarray(mask, dtype=bool)
        if region_mask.shape != self.image_array.shape[:2]:
            raise SemanticBackendError(
                f"CLIP region mask shape {region_mask.shape} does not match image shape {self.image_array.shape[:2]}."
            )
        if not bool(region_mask.any()):
            return self.image.copy()

        ys, xs = np.where(region_mask)
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        crop_array = np.asarray(self.image.crop((x0, y0, x1, y1)), dtype=np.uint8).copy()
        crop_mask = region_mask[y0:y1, x0:x1]
        crop_array[~crop_mask] = 0
        return Image.fromarray(crop_array, mode="RGB")

    def _score_prompts(self, crop: Image.Image, prompts: Sequence[str]) -> np.ndarray:
        """Score one image crop against a list of prompts using cosine similarity.

        Args:
            crop: RGB PIL image crop.
            prompts: Prompt strings with length T.

        Returns:
            Similarity vector with shape [T].
        """

        with clip_inference_context():
            image_tensor = self.preprocess(crop).unsqueeze(0).to(self.device)
            token_tensor = self.tokenizer(list(prompts)).to(self.device)
            image_features = self.model.encode_image(image_tensor)
            text_features = self.model.encode_text(token_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            similarities = image_features @ text_features.T
        return tensor_to_numpy(similarities).reshape(-1).astype(np.float32)


def build_semantic_adapter(backend: str) -> Any:
    """Build the requested semantic expert adapter.

    Args:
        backend: Semantic backend name, either "fallback" or "clip".

    Returns:
        Adapter object exposing prepare_image() and score_region().
    """

    if backend == "fallback":
        return FallbackSemanticAdapter()
    if backend == "clip":
        return OpenClipSemanticAdapter()
    raise ValueError(f"Unknown semantic backend {backend!r}; expected 'fallback' or 'clip'.")


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


def run_inference(
    image_path: Path,
    class_names: Sequence[str],
    output_path: Path,
    semantic_backend: str = "fallback",
    mask_backend: str = "fallback",
    sam_checkpoint: Optional[Path] = None,
    sam_model_type: str = "vit_b",
    max_masks: int = 100,
) -> Dict[str, Any]:
    """Run the UR-OVSS MVP loop and save visualization, mask, and JSON.

    Args:
        image_path: Input image path.
        class_names: Open-vocabulary classes with length C.
        output_path: PNG visualization path.
        semantic_backend: Semantic expert backend, either "fallback" or "clip".
        mask_backend: Mask backend, either "fallback" or "sam".
        sam_checkpoint: Optional SAM checkpoint path for the "sam" backend.
        sam_model_type: SAM model type key.
        max_masks: Maximum number of masks to keep.

    Returns:
        Dictionary containing output paths and region debug records.
    """

    pil_image, image_array = load_rgb_image(image_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    positive_prompts = build_positive_prompts(class_names)
    negative_prompts = build_negative_prompts(class_names)
    semantic_adapter = build_semantic_adapter(semantic_backend)
    semantic_adapter.prepare_image(pil_image, image_array)
    mask_adapter = build_mask_adapter(
        mask_backend,
        sam_checkpoint=sam_checkpoint,
        sam_model_type=sam_model_type,
        max_masks=max_masks,
    )

    height, width = image_array.shape[:2]
    masks = mask_adapter.generate_masks(pil_image, image_array)
    if not masks:
        raise MaskBackendError(f"Mask backend {mask_backend!r} did not generate any mask.")
    dino_features = build_patch_proxy_features(image_array)

    region_work: List[Dict[str, Any]] = []
    for region_id, mask_record in enumerate(masks):
        mask = mask_record["segmentation"]
        semantic_scores = semantic_adapter.score_region(
            mask,
            class_names,
            positive_prompts,
            negative_prompts,
        )
        dino_scores = compute_dino_region_scores(dino_features, mask, class_names)
        region_work.append(
            {
                "region_id": region_id,
                "mask": mask,
                "area": int(mask.sum()),
                "source": mask_record["source"],
                "base_scores": semantic_scores.base_scores,
                "positive_scores": semantic_scores.positive_scores,
                "negative_scores": semantic_scores.negative_scores,
                "prompt_rescore_scores": semantic_scores.prompt_rescore_scores,
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
        routed["base_scores"] = region["base_scores"].tolist()
        routed["positive_scores"] = region["positive_scores"].tolist()
        routed["negative_scores"] = region["negative_scores"].tolist()
        routed["prompt_rescore_scores"] = region["prompt_rescore_scores"].tolist()
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
            "semantic": semantic_adapter.description,
            "spatial": mask_adapter.description,
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
    parser.add_argument(
        "--semantic-backend",
        choices=("fallback", "clip"),
        default="fallback",
        help="Semantic expert backend. 'clip' requires optional open_clip dependencies and model access.",
    )
    parser.add_argument(
        "--mask-backend",
        choices=("fallback", "sam"),
        default="fallback",
        help="Class-agnostic mask backend. 'sam' requires optional SAM dependencies and --sam-checkpoint.",
    )
    parser.add_argument(
        "--sam-checkpoint",
        type=Path,
        default=None,
        help="Path to a SAM/MobileSAM checkpoint. Required when --mask-backend sam is selected.",
    )
    parser.add_argument(
        "--sam-model-type",
        default="vit_b",
        help="SAM model type key used by sam_model_registry, e.g. vit_b.",
    )
    parser.add_argument(
        "--max-masks",
        type=int,
        default=100,
        help="Maximum number of candidate masks to keep from the selected mask backend.",
    )
    return parser


def main() -> None:
    """CLI entry point for UR-OVSS MVP inference."""

    args = build_arg_parser().parse_args()
    class_names = parse_class_names(args.classes)
    try:
        result = run_inference(
            args.image,
            class_names,
            args.output,
            semantic_backend=args.semantic_backend,
            mask_backend=args.mask_backend,
            sam_checkpoint=args.sam_checkpoint,
            sam_model_type=args.sam_model_type,
            max_masks=args.max_masks,
        )
    except SemanticBackendError as exc:
        raise SystemExit(f"Semantic backend error: {exc}") from exc
    except MaskBackendError as exc:
        raise SystemExit(f"Mask backend error: {exc}") from exc
    outputs = result["outputs"]
    print("UR-OVSS MVP inference complete.")
    print(f"Visualization: {outputs['visualization']}")
    print(f"Label PNG: {outputs['label_png']}")
    print(f"Mask NPY: {outputs['mask_npy']}")
    print(f"Confidence NPY: {outputs['confidence_npy']}")
    print(f"Debug JSON: {outputs['debug_json']}")
    if args.semantic_backend == "fallback":
        print("Note: semantic backend used deterministic fallback features, not real CLIP weights.")
    else:
        print("Note: DINO purity still uses deterministic fallback features.")
    if args.mask_backend == "fallback":
        print("Note: mask backend used deterministic fallback masks, not real SAM masks.")


if __name__ == "__main__":
    main()
