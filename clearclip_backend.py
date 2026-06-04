"""Minimal ClearCLIP-style dense semantic adapter for UR-OVSS.

This module is a small, dependency-optional adapter inspired by the official
ClearCLIP project: https://github.com/mc-lan/ClearCLIP

It does not vendor ClearCLIP code or weights. It reuses open_clip weights from
the normal user cache and exposes dense image/text logits that the existing
UR-OVSS region routing code can pool over SAM masks.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type

import numpy as np
from PIL import Image

from prompts import compute_prompt_rescore


DEFAULT_NEGATIVE_PROMPT_SUPPRESSION_ALPHA = 0.30


@dataclass
class ClearClipRegionScores:
    """Container matching the semantic adapter score contract.

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


def _normalize_last_dim(features: np.ndarray) -> np.ndarray:
    """L2-normalize a numpy feature array along the final dimension."""

    norm = np.linalg.norm(features, axis=-1, keepdims=True)
    return features / np.maximum(norm, 1e-6)


def _tensor_to_numpy(values: Any) -> np.ndarray:
    """Convert numpy-like or torch-like values to float32 numpy arrays."""

    if hasattr(values, "detach") and callable(values.detach):
        values = values.detach()
    if hasattr(values, "cpu") and callable(values.cpu):
        values = values.cpu()
    if hasattr(values, "numpy") and callable(values.numpy):
        values = values.numpy()
    return np.asarray(values, dtype=np.float32)


def _inference_context() -> Any:
    """Return torch inference_mode when available, otherwise a null context."""

    try:
        import torch

        return torch.inference_mode()
    except Exception:
        return nullcontext()


def _resize_feature_grid(feature_grid: np.ndarray, output_shape: Tuple[int, int]) -> np.ndarray:
    """Resize a dense feature/logit grid to image shape [H, W]."""

    height, width = output_shape
    channels = feature_grid.shape[-1]
    resized = np.empty((height, width, channels), dtype=np.float32)
    resample = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR
    for channel in range(channels):
        channel_image = Image.fromarray(feature_grid[..., channel].astype(np.float32), mode="F")
        resized[..., channel] = np.asarray(channel_image.resize((width, height), resample=resample), dtype=np.float32)
    return resized


def _pool_mask_scores(dense_scores: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Average dense prompt scores inside a region mask."""

    region_mask = np.asarray(mask, dtype=bool)
    if dense_scores.shape[:2] != region_mask.shape:
        raise ValueError(f"dense score shape {dense_scores.shape[:2]} does not match mask shape {region_mask.shape}.")
    if not bool(region_mask.any()):
        return np.zeros(dense_scores.shape[-1], dtype=np.float32)
    return dense_scores[region_mask].mean(axis=0).astype(np.float32)


def _flatten_prompt_dict(prompt_dict: Dict[str, List[str]], class_names: Sequence[str]) -> List[str]:
    """Flatten a class-to-prompts mapping in class order."""

    return [prompt for class_name in class_names for prompt in prompt_dict[class_name]]


class ClearClipSemanticAdapter:
    """Optional ClearCLIP-style dense semantic adapter backed by open_clip.

    The adapter computes image-sized dense visual features once per image, then
    scores all positive/negative text prompts as dense maps. SAM regions are
    scored by pooling dense logits inside each mask instead of re-encoding
    region crops.
    """

    description = "clearclip dense patch logits via open_clip"

    def __init__(
        self,
        model_name: str = "ViT-B-16",
        pretrained: str = "openai",
        device: Optional[str] = None,
        open_clip_module: Optional[Any] = None,
        backend_error_cls: Type[RuntimeError] = RuntimeError,
        score_cls: Callable[..., Any] = ClearClipRegionScores,
        alpha: float = DEFAULT_NEGATIVE_PROMPT_SUPPRESSION_ALPHA,
    ) -> None:
        """Load an open_clip model for ClearCLIP-style dense scoring.

        Args:
            model_name: open_clip model name. ViT-B-16 is closest to the
                official ClearCLIP default used here.
            pretrained: open_clip pretrained tag.
            device: Device string such as "cpu" or "cuda".
            open_clip_module: Optional module-like object for tests.
            backend_error_cls: Error class used for actionable backend errors.
            score_cls: Score container factory matching RegionSemanticScores.
            alpha: Negative prompt suppression factor.
        """

        self.error_cls = backend_error_cls
        self.score_cls = score_cls
        self.alpha = alpha
        self.device = device or self._default_device()
        self.model_name = model_name
        self.pretrained = pretrained
        self._uses_injected_open_clip = open_clip_module is not None
        self.dense_features: Optional[np.ndarray] = None
        self.image_shape: Optional[Tuple[int, int]] = None
        self._text_feature_cache: Dict[Tuple[str, ...], np.ndarray] = {}
        self._dense_score_cache: Dict[Tuple[str, ...], np.ndarray] = {}

        if open_clip_module is None:
            try:
                import open_clip as open_clip_module  # type: ignore[import-not-found]
            except Exception as exc:
                raise self.error_cls(
                    "The ClearCLIP semantic backend requires open_clip and torch. Install optional dependencies with "
                    "`pip install open_clip_torch` or `pip install -r requirements-clip.txt`."
                ) from exc

        self.open_clip = open_clip_module
        try:
            self.model, _, self.preprocess = self.open_clip.create_model_and_transforms(
                model_name,
                pretrained=pretrained,
                device=self.device,
            )
            if hasattr(self.model, "eval"):
                self.model.eval()
            self.tokenizer = self.open_clip.get_tokenizer(model_name)
        except Exception as exc:
            raise self.error_cls(
                "Failed to load ClearCLIP open_clip model "
                f"{model_name!r} with pretrained={pretrained!r} on device {self.device!r}. "
                "Weights are expected in the normal user cache, not this repository."
            ) from exc

    def _default_device(self) -> str:
        """Choose CUDA when torch is installed and a GPU is visible."""

        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def prepare_image(self, image: Image.Image, image_array: np.ndarray) -> None:
        """Precompute image-sized dense ClearCLIP visual features.

        Args:
            image: RGB PIL image.
            image_array: RGB image with shape [H, W, 3] and values in [0, 1].
        """

        self.image_shape = image_array.shape[:2]
        self._dense_score_cache.clear()
        try:
            with self._inference_context():
                image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
                dense_features = self._encode_dense_image_features(image_tensor, self.image_shape)
        except Exception as exc:
            raise self.error_cls(f"ClearCLIP dense image feature extraction failed: {exc}") from exc

        self.dense_features = _normalize_last_dim(dense_features).astype(np.float32)

    def dense_logits_for_prompts(self, prompts: Sequence[str]) -> np.ndarray:
        """Compute image-sized dense logits for prompt strings.

        Args:
            prompts: Text prompts with length T.

        Returns:
            Dense score map with shape [H, W, T].
        """

        if self.dense_features is None:
            raise self.error_cls("ClearCLIP semantic backend was used before prepare_image().")
        prompt_key = tuple(prompts)
        if prompt_key not in self._dense_score_cache:
            text_features = self._text_features(prompt_key)
            dense_scores = np.tensordot(self.dense_features, text_features.T, axes=([-1], [0]))
            self._dense_score_cache[prompt_key] = dense_scores.astype(np.float32)
        return self._dense_score_cache[prompt_key]

    def dense_logits_for_text_prototypes(self, prompt_groups: Sequence[Sequence[str]]) -> np.ndarray:
        """Compute dense logits from per-class averaged text prototypes.

        Args:
            prompt_groups: Prompt strings grouped by class. The outer length is
                C and each inner sequence contains prompt templates for one
                class.

        Returns:
            Dense class logits with shape [H, W, C]. For each class, prompt
            text features are normalized, averaged, normalized again, then
            compared against dense visual features.
        """

        if self.dense_features is None:
            raise self.error_cls("ClearCLIP semantic backend was used before prepare_image().")

        prototype_key = tuple(tuple(group) for group in prompt_groups)
        cache_key = ("__prototype_average__", *prototype_key)
        if cache_key not in self._dense_score_cache:
            prototypes = []
            for prompts in prototype_key:
                if not prompts:
                    raise self.error_cls("Text prototype averaging received an empty prompt group.")
                text_features = self._text_features(prompts)
                prototype = text_features.mean(axis=0)
                prototype = _normalize_last_dim(prototype)
                prototypes.append(prototype)
            text_prototypes = np.stack(prototypes, axis=0).astype(np.float32)
            dense_scores = np.tensordot(self.dense_features, text_prototypes.T, axes=([-1], [0]))
            self._dense_score_cache[cache_key] = dense_scores.astype(np.float32)
        return self._dense_score_cache[cache_key]

    def score_region(
        self,
        mask: np.ndarray,
        class_names: Sequence[str],
        positive_prompts: Dict[str, List[str]],
        negative_prompts: Dict[str, List[str]],
    ) -> Any:
        """Score one region by pooling ClearCLIP dense logits over its mask.

        Args:
            mask: Boolean region mask with shape [H, W].
            class_names: Open-vocabulary classes with length C.
            positive_prompts: Positive prompts, C classes by P prompts.
            negative_prompts: Negative prompts, C classes by N prompts.

        Returns:
            Region semantic scores with shapes [C], [C, P], [C, N], and [C].
        """

        class_prompt_texts = [positive_prompts[class_name][0] for class_name in class_names]
        positive_flat = _flatten_prompt_dict(positive_prompts, class_names)
        negative_flat = _flatten_prompt_dict(negative_prompts, class_names)
        num_positive_prompts = len(positive_prompts[class_names[0]])
        num_negative_prompts = len(negative_prompts[class_names[0]])

        base_scores = _pool_mask_scores(self.dense_logits_for_prompts(class_prompt_texts), mask)
        positive_scores = _pool_mask_scores(self.dense_logits_for_prompts(positive_flat), mask).reshape(
            len(class_names),
            num_positive_prompts,
        )
        negative_scores = _pool_mask_scores(self.dense_logits_for_prompts(negative_flat), mask).reshape(
            len(class_names),
            num_negative_prompts,
        )
        prompt_rescore_scores = compute_prompt_rescore(
            base_scores,
            positive_scores,
            negative_scores,
            alpha=self.alpha,
        )
        return self.score_cls(
            base_scores=base_scores,
            positive_scores=positive_scores,
            negative_scores=negative_scores,
            prompt_rescore_scores=prompt_rescore_scores,
        )

    def _text_features(self, prompts: Tuple[str, ...]) -> np.ndarray:
        """Encode and cache normalized text features for prompts."""

        if prompts not in self._text_feature_cache:
            with self._inference_context():
                token_tensor = self.tokenizer(list(prompts))
                if hasattr(token_tensor, "to"):
                    token_tensor = token_tensor.to(self.device)
                text_features = self.model.encode_text(token_tensor)
            self._text_feature_cache[prompts] = _normalize_last_dim(_tensor_to_numpy(text_features))
        return self._text_feature_cache[prompts]

    def _inference_context(self) -> Any:
        """Use no-grad inference for real open_clip and null context for injected fakes."""

        if self._uses_injected_open_clip:
            return nullcontext()
        return _inference_context()

    def _encode_dense_image_features(self, image_tensor: Any, output_shape: Tuple[int, int]) -> np.ndarray:
        """Extract dense image features from fake or real open_clip ViT models."""

        if hasattr(self.model, "encode_dense_image"):
            dense = _tensor_to_numpy(self.model.encode_dense_image(image_tensor))
            return self._coerce_dense_features(dense, output_shape)
        if hasattr(self.model, "visual"):
            dense = self._encode_open_clip_vit_dense(image_tensor)
            return self._coerce_dense_features(_tensor_to_numpy(dense), output_shape)
        raise self.error_cls(
            "ClearCLIP backend needs an open_clip ViT model exposing `visual` patch tokens "
            "or a test model exposing `encode_dense_image()`."
        )

    def _coerce_dense_features(self, dense: np.ndarray, output_shape: Tuple[int, int]) -> np.ndarray:
        """Normalize dense feature layout and resize to image shape [H, W, D]."""

        if dense.ndim == 4:
            dense = dense[0]
        if dense.ndim == 2:
            grid_size = int(np.sqrt(dense.shape[0]))
            if grid_size * grid_size != dense.shape[0]:
                raise self.error_cls(f"Cannot infer a square patch grid from dense features with shape {dense.shape}.")
            dense = dense.reshape(grid_size, grid_size, dense.shape[-1])
        if dense.ndim != 3:
            raise self.error_cls(f"ClearCLIP dense features must have shape [H, W, D], got {dense.shape}.")
        return _resize_feature_grid(dense.astype(np.float32), output_shape)

    def _encode_open_clip_vit_dense(self, image_tensor: Any) -> Any:
        """Best-effort ClearCLIP final-layer decomposition for open_clip ViTs."""

        try:
            import torch
            import torch.nn.functional as torch_f
        except Exception as exc:
            raise self.error_cls("ClearCLIP dense ViT extraction requires torch.") from exc

        visual = self.model.visual
        if not hasattr(visual, "conv1") or not hasattr(visual, "transformer"):
            raise self.error_cls("ClearCLIP dense extraction currently supports open_clip ViT visual towers only.")

        x = visual.conv1(image_tensor)
        batch_size, width, grid_h, grid_w = x.shape
        x = x.reshape(batch_size, width, grid_h * grid_w).permute(0, 2, 1)

        class_embedding = visual.class_embedding.to(dtype=x.dtype, device=x.device)
        class_token = class_embedding + torch.zeros(batch_size, 1, width, dtype=x.dtype, device=x.device)
        x = torch.cat([class_token, x], dim=1)
        if hasattr(visual, "positional_embedding"):
            x = x + self._position_embedding_for_grid(visual, x, grid_h, grid_w, torch, torch_f)
        if hasattr(visual, "patch_dropout"):
            x = visual.patch_dropout(x)
        if hasattr(visual, "ln_pre"):
            x = visual.ln_pre(x)

        blocks = list(getattr(visual.transformer, "resblocks", []))
        if not blocks:
            x = visual.transformer(x)
        else:
            for block in blocks[:-1]:
                x = self._run_transformer_block(block, x)
            x = self._run_clearclip_final_block(blocks[-1], x)

        patch_tokens = x[:, 1:, :]
        if hasattr(visual, "ln_post"):
            patch_tokens = visual.ln_post(patch_tokens)
        projection = getattr(visual, "proj", None)
        if projection is not None:
            patch_tokens = patch_tokens @ projection
        return patch_tokens.reshape(batch_size, grid_h, grid_w, patch_tokens.shape[-1])

    def _position_embedding_for_grid(
        self,
        visual: Any,
        tokens: Any,
        grid_h: int,
        grid_w: int,
        torch_module: Any,
        torch_f: Any,
    ) -> Any:
        """Return positional embeddings, interpolated if the patch grid changed."""

        pos = visual.positional_embedding.to(dtype=tokens.dtype, device=tokens.device)
        if pos.shape[0] == tokens.shape[1]:
            return pos

        patch_pos = pos[1:]
        old_grid = int(np.sqrt(patch_pos.shape[0]))
        if old_grid * old_grid != patch_pos.shape[0]:
            raise self.error_cls(f"Cannot interpolate positional embedding with shape {tuple(pos.shape)}.")
        cls_pos = pos[:1]
        patch_pos = patch_pos.reshape(1, old_grid, old_grid, -1).permute(0, 3, 1, 2)
        patch_pos = torch_f.interpolate(patch_pos, size=(grid_h, grid_w), mode="bicubic", align_corners=False)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(grid_h * grid_w, -1)
        return torch_module.cat([cls_pos, patch_pos], dim=0)

    def _run_transformer_block(self, block: Any, tokens: Any) -> Any:
        """Run a transformer block, handling batch-first or sequence-first variants."""

        try:
            return block(tokens)
        except Exception:
            return block(tokens.transpose(0, 1)).transpose(0, 1)

    def _run_clearclip_final_block(self, block: Any, tokens: Any) -> Any:
        """Apply final self-attention without residual or FFN, ClearCLIP-style."""

        if not hasattr(block, "ln_1") or not hasattr(block, "attn"):
            return self._run_transformer_block(block, tokens)

        query = block.ln_1(tokens)
        attn_mask = getattr(block, "attn_mask", None)
        try:
            output = block.attn(query, query, query, need_weights=False, attn_mask=attn_mask)[0]
        except Exception:
            query_t = query.transpose(0, 1)
            output = block.attn(query_t, query_t, query_t, need_weights=False, attn_mask=attn_mask)[0].transpose(0, 1)
        return output
