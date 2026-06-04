"""Dense-only CLIP/ClearCLIP Pascal VOC evaluation.

This script evaluates dense semantic logits directly on Pascal VOC. It is
separate from `eval_pascal_voc.py` because it intentionally bypasses SAM masks,
DINOv2 features, and uncertainty routing.
"""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from PIL import Image

from clearclip_backend import ClearClipSemanticAdapter
from eval_clearclip_dense_voc import OPENAI_IMAGENET_TEMPLATES, build_imagenet_prompts, _resize_logits
from eval_pascal_voc import (
    VOC21_CLASSES,
    VOC_CLASSES,
    compute_confusion_matrix,
    compute_voc_confusion_matrix,
    read_split_ids,
    summarize_voc_metrics,
)
from infer_ur_ovss import SemanticBackendError


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
    """Return torch inference_mode when torch is available."""

    try:
        import torch

        return torch.inference_mode()
    except Exception:
        return nullcontext()


def _load_rgb_image(path: Path) -> tuple[Image.Image, np.ndarray]:
    """Load an RGB image and normalized numpy array."""

    image = Image.open(path).convert("RGB")
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    return image, image_array


def _load_target_mask(path: Path) -> np.ndarray:
    """Load a Pascal VOC segmentation mask with shape [H, W]."""

    return np.asarray(Image.open(path), dtype=np.int64)


def _coerce_dense_grid(
    values: Any,
    patch_grid: tuple[int, int],
    value_name: str,
    error_cls: type[RuntimeError] = SemanticBackendError,
) -> np.ndarray:
    """Coerce dense patch values to [grid_h, grid_w, C].

    Args:
        values: Dense values with shape [H, W, C], [1, H, W, C], or [H*W, C].
        patch_grid: Expected patch grid as (grid_h, grid_w).
        value_name: Human-readable name for error messages.
        error_cls: Error class used for clear backend failures.

    Returns:
        Float32 dense grid with shape [grid_h, grid_w, C].
    """

    dense = np.asarray(values, dtype=np.float32)
    grid_h, grid_w = patch_grid
    if dense.ndim == 4 and dense.shape[0] == 1:
        dense = dense[0]
    if dense.ndim == 3:
        return dense.astype(np.float32)
    if dense.ndim == 2:
        expected_tokens = grid_h * grid_w
        if dense.shape[0] != expected_tokens:
            raise error_cls(
                f"{value_name} with shape {dense.shape} cannot be reshaped to patch grid {patch_grid}; "
                f"expected first dimension {expected_tokens}."
            )
        return dense.reshape(grid_h, grid_w, dense.shape[-1]).astype(np.float32)
    raise error_cls(
        f"{value_name} must have shape [H, W, C], [1, H, W, C], or [H*W, C]; got {dense.shape}."
    )


class VanillaClipDenseAdapter:
    """Vanilla CLIP ViT dense patch-logit adapter backed by open_clip.

    This adapter extracts patch-level visual tokens from a standard CLIP ViT
    without ClearCLIP final-layer changes. It then compares normalized dense
    patch features against normalized text prototypes built from OpenAI
    ImageNet prompt templates.
    """

    description = "vanilla open_clip dense patch logits"

    def __init__(
        self,
        model_name: str = "ViT-B-16",
        pretrained: str = "openai",
        device: Optional[str] = None,
        open_clip_module: Optional[Any] = None,
        backend_error_cls: type[RuntimeError] = SemanticBackendError,
    ) -> None:
        """Load a vanilla open_clip ViT model.

        Args:
            model_name: open_clip model name.
            pretrained: open_clip pretrained tag.
            device: Device string. Defaults to CUDA if visible.
            open_clip_module: Optional module-like object for tests.
            backend_error_cls: Error class used for clear backend failures.
        """

        self.error_cls = backend_error_cls
        self.device = device or self._default_device()
        self.model_name = model_name
        self.pretrained = pretrained
        self.dense_features: Optional[np.ndarray] = None
        self.image_shape: Optional[tuple[int, int]] = None
        self._class_feature_cache: Dict[tuple[str, ...], np.ndarray] = {}

        if open_clip_module is None:
            try:
                import open_clip as open_clip_module  # type: ignore[import-not-found]
            except Exception as exc:
                raise self.error_cls(
                    "The vanilla CLIP dense backend requires open_clip and torch. Install optional dependencies with "
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
                "Failed to load vanilla CLIP open_clip model "
                f"{model_name!r} with pretrained={pretrained!r} on device {self.device!r}. "
                "Weights are expected in the normal user cache, not this repository."
            ) from exc

    def _default_device(self) -> str:
        """Choose CUDA when torch reports a visible GPU."""

        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def prepare_image(self, image: Image.Image, image_array: np.ndarray) -> None:
        """Precompute image-sized vanilla CLIP dense patch features.

        Args:
            image: RGB PIL image.
            image_array: RGB image with shape [H, W, 3] and values in [0, 1].
        """

        self.image_shape = image_array.shape[:2]
        try:
            with _inference_context():
                image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
                dense = self._encode_dense_image_features(image_tensor)
                patch_grid = self._patch_grid_from_tensor(self.model.visual, image_tensor)
        except Exception as exc:
            raise self.error_cls(f"Vanilla CLIP dense image feature extraction failed: {exc}") from exc

        dense = _coerce_dense_grid(
            dense,
            patch_grid=patch_grid,
            value_name="Vanilla CLIP dense patch features",
            error_cls=self.error_cls,
        )
        dense = _resize_logits(dense.astype(np.float32), self.image_shape)
        self.dense_features = _normalize_last_dim(dense).astype(np.float32)

    def dense_logits_for_classes(self, class_names: Sequence[str], output_shape: tuple[int, int]) -> np.ndarray:
        """Compute image-sized dense class logits.

        Args:
            class_names: Class names with length C.
            output_shape: Desired output shape [H, W].

        Returns:
            Dense class logits with shape [H, W, C].
        """

        if self.dense_features is None:
            raise self.error_cls("Vanilla CLIP dense backend was used before prepare_image().")
        class_features = self._class_text_features(tuple(class_names))
        dense_logits = np.tensordot(self.dense_features, class_features.T, axes=([-1], [0]))
        return _resize_logits(dense_logits.astype(np.float32), output_shape)

    def _class_text_features(self, class_names: tuple[str, ...]) -> np.ndarray:
        """Build normalized class text prototypes from ImageNet templates."""

        if class_names not in self._class_feature_cache:
            prompts = build_imagenet_prompts(class_names)
            prompt_count = len(OPENAI_IMAGENET_TEMPLATES)
            try:
                with _inference_context():
                    token_tensor = self.tokenizer(prompts)
                    if hasattr(token_tensor, "to"):
                        token_tensor = token_tensor.to(self.device)
                    prompt_features = self.model.encode_text(token_tensor)
            except Exception as exc:
                raise self.error_cls(f"Vanilla CLIP text feature extraction failed: {exc}") from exc

            prompt_features = _normalize_last_dim(_tensor_to_numpy(prompt_features))
            class_features = prompt_features.reshape(len(class_names), prompt_count, -1).mean(axis=1)
            self._class_feature_cache[class_names] = _normalize_last_dim(class_features).astype(np.float32)
        return self._class_feature_cache[class_names]

    def _encode_dense_image_features(self, image_tensor: Any) -> np.ndarray:
        """Extract standard final-layer ViT patch tokens from open_clip."""

        visual = getattr(self.model, "visual", None)
        if visual is None:
            raise self.error_cls("Vanilla CLIP dense extraction currently supports open_clip ViT visual towers only.")
        if hasattr(visual, "forward_intermediates"):
            tokens = self._encode_with_forward_intermediates(visual, image_tensor)
        else:
            tokens = self._encode_with_manual_vit_forward(visual, image_tensor)
        return _tensor_to_numpy(tokens)

    def _encode_with_forward_intermediates(self, visual: Any, image_tensor: Any) -> Any:
        """Use open_clip's native intermediate-token helper when available."""

        outputs = visual.forward_intermediates(
            image_tensor,
            indices=1,
            normalize_intermediates=True,
            intermediates_only=True,
            output_fmt="NLC",
        )
        dense_tokens = outputs["image_intermediates"][-1]
        projection = getattr(visual, "proj", None)
        if projection is not None:
            dense_tokens = dense_tokens @ projection
        grid_h, grid_w = self._patch_grid_from_tensor(visual, image_tensor)
        return dense_tokens.reshape(dense_tokens.shape[0], grid_h, grid_w, dense_tokens.shape[-1])

    def _encode_with_manual_vit_forward(self, visual: Any, image_tensor: Any) -> Any:
        """Best-effort standard ViT forward path for older open_clip versions."""

        try:
            import torch
            import torch.nn.functional as torch_f
        except Exception as exc:
            raise self.error_cls("Vanilla CLIP dense ViT extraction requires torch.") from exc

        if not hasattr(visual, "conv1") or not hasattr(visual, "transformer"):
            raise self.error_cls("Vanilla CLIP dense extraction currently supports open_clip ViT visual towers only.")

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

        x = visual.transformer(x)
        if hasattr(visual, "ln_post"):
            x = visual.ln_post(x)
        patch_tokens = x[:, 1:, :]
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

    def _patch_grid_from_tensor(self, visual: Any, image_tensor: Any) -> tuple[int, int]:
        """Infer the ViT patch grid for a preprocessed image tensor."""

        patch_size = getattr(visual, "patch_size", None)
        if patch_size is None:
            raise self.error_cls("Cannot infer patch grid because the open_clip visual tower has no patch_size.")
        if isinstance(patch_size, int):
            patch_h = patch_w = patch_size
        else:
            patch_h, patch_w = int(patch_size[0]), int(patch_size[1])
        return int(image_tensor.shape[-2] // patch_h), int(image_tensor.shape[-1] // patch_w)


def build_dense_adapter(backend: str, model_name: str = "ViT-B-16", pretrained: str = "openai") -> Any:
    """Build a dense-only semantic adapter.

    Args:
        backend: "clip" for vanilla CLIP dense tokens or "clearclip" for the
            existing ClearCLIP-style dense adapter.
        model_name: open_clip model name.
        pretrained: open_clip pretrained tag.

    Returns:
        Adapter exposing prepare_image() and dense logits methods.
    """

    if backend == "clip":
        return VanillaClipDenseAdapter(model_name=model_name, pretrained=pretrained)
    if backend == "clearclip":
        return ClearClipSemanticAdapter(
            model_name=model_name,
            pretrained=pretrained,
            backend_error_cls=SemanticBackendError,
        )
    raise ValueError(f"Unknown dense semantic backend {backend!r}; expected 'clip' or 'clearclip'.")


def compute_dense_logits_for_classes(adapter: Any, class_names: Sequence[str], output_shape: tuple[int, int]) -> np.ndarray:
    """Compute image-sized class logits from a dense semantic adapter.

    Args:
        adapter: Dense adapter prepared for the current image.
        class_names: Class names with length C.
        output_shape: Desired output shape [H, W].

    Returns:
        Dense class logits with shape [H, W, C].
    """

    if hasattr(adapter, "dense_logits_for_classes"):
        return adapter.dense_logits_for_classes(class_names, output_shape)

    prompt_count = len(OPENAI_IMAGENET_TEMPLATES)
    prompt_logits = adapter.dense_logits_for_prompts(build_imagenet_prompts(class_names))
    prompt_logits = _resize_logits(prompt_logits, output_shape)
    height, width = output_shape
    return prompt_logits.reshape(height, width, len(class_names), prompt_count).mean(axis=-1).astype(np.float32)


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


def _label_space(voc_mode: str) -> str:
    """Describe the saved prediction label space."""

    if voc_mode == "voc21":
        return "VOC labels 0-20: background plus 20 foreground classes"
    return "foreground indices 0-19; evaluation maps them to VOC labels 1-20"


def evaluate_dataset(
    voc_root: Path,
    split: str,
    output_dir: Path,
    semantic_backend: str = "clip",
    limit: Optional[int] = None,
    voc_mode: str = "voc20",
    voc20_ignore_background: bool = False,
    model_name: str = "ViT-B-16",
    pretrained: str = "openai",
    save_debug: bool = False,
) -> Dict[str, Any]:
    """Evaluate dense-only CLIP/ClearCLIP logits on Pascal VOC.

    Args:
        voc_root: Path to `VOCdevkit/VOC2012`.
        split: VOC segmentation split name.
        output_dir: Directory where metrics and predictions are saved.
        semantic_backend: "clip" or "clearclip".
        limit: Optional maximum number of image ids.
        voc_mode: "voc20" evaluates foreground classes only; "voc21"
            includes background.
        voc20_ignore_background: In VOC20 mode, ignore GT background pixels
            before the confusion matrix.
        model_name: open_clip model name.
        pretrained: open_clip pretrained tag.
        save_debug: Save one lightweight JSON record per evaluated image.

    Returns:
        Metrics dictionary also written to `metrics.json`.
    """

    if semantic_backend not in {"clip", "clearclip"}:
        raise ValueError(f"semantic_backend must be 'clip' or 'clearclip', got {semantic_backend!r}.")
    if voc_mode not in {"voc20", "voc21"}:
        raise ValueError(f"voc_mode must be 'voc20' or 'voc21', got {voc_mode!r}.")

    voc_root = Path(voc_root)
    output_dir = Path(output_dir)
    prediction_dir = output_dir / "predictions"
    debug_dir = output_dir / "debug"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    if save_debug:
        debug_dir.mkdir(parents=True, exist_ok=True)

    image_ids = read_split_ids(voc_root, split)
    if limit is not None:
        image_ids = image_ids[:limit]

    adapter = build_dense_adapter(semantic_backend, model_name=model_name, pretrained=pretrained)
    print(f"{semantic_backend} dense backend is initialized once for this dense-only VOC evaluation run.")

    confusion = np.zeros((len(VOC21_CLASSES), len(VOC21_CLASSES)), dtype=np.int64)
    evaluated_images = 0
    skipped_images = 0
    skipped: List[Dict[str, str]] = []
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
        adapter.prepare_image(image, image_array)
        dense_logits = compute_dense_logits_for_classes(adapter, eval_class_names, image_array.shape[:2])
        pred_indices = np.argmax(dense_logits, axis=-1).astype(np.int64)

        prediction_path = prediction_dir / f"{image_id}.npy"
        np.save(prediction_path, pred_indices)
        prediction_files.append(str(prediction_path))

        target = _load_target_mask(target_path)
        confusion += _prediction_to_confusion(
            pred_indices,
            target,
            voc_mode=voc_mode,
            voc20_ignore_background=voc20_ignore_background,
        )

        if save_debug:
            debug_payload = {
                "image_id": image_id,
                "image": str(image_path),
                "target": str(target_path),
                "semantic_backend": semantic_backend,
                "adapter": getattr(adapter, "description", semantic_backend),
                "class_names": list(eval_class_names),
                "dense_logits_shape": list(dense_logits.shape),
                "prediction_npy": str(prediction_path),
                "prediction_label_space": _label_space(voc_mode),
            }
            (debug_dir / f"{image_id}.json").write_text(json.dumps(debug_payload, indent=2), encoding="utf-8")

        evaluated_images += 1

    if evaluated_images == 0:
        first_skip = f" First skipped item: {skipped[0]['reason']}." if skipped else ""
        raise RuntimeError(
            "Dense VOC evaluation did not evaluate any images. "
            f"Skipped {skipped_images} image(s); check --voc-root, --split, and dataset files.{first_skip}"
        )

    summarized = summarize_voc_metrics(confusion, voc_mode=voc_mode)
    metrics_path = output_dir / "metrics.json"
    metrics: Dict[str, Any] = {
        "split": split,
        "voc_root": str(voc_root),
        "semantic_backend": semantic_backend,
        "model_name": model_name,
        "pretrained": pretrained,
        "prompt_templates": (
            "openai_imagenet_template_text_prototype_average"
            if semantic_backend == "clip"
            else "openai_imagenet_template_logit_average"
        ),
        "uses_sam": False,
        "uses_dinov2": False,
        "uses_routing": False,
        "voc_mode": voc_mode,
        "voc20_ignore_background": bool(voc20_ignore_background),
        "background_iou": summarized["background_iou"],
        "mIoU": summarized["mIoU"],
        "per_class_iou": summarized["per_class_iou"],
        "evaluated_images": evaluated_images,
        "skipped_images": skipped_images,
        "skipped": skipped,
        "classes": summarized["classes"],
        "predictions_dir": str(prediction_dir),
        "prediction_files": prediction_files,
        "debug_dir": str(debug_dir) if save_debug else None,
        "metrics_path": str(metrics_path),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for dense-only VOC evaluation."""

    parser = argparse.ArgumentParser(description="Evaluate dense CLIP/ClearCLIP logits directly on Pascal VOC 2012.")
    parser.add_argument("--semantic-backend", choices=("clip", "clearclip"), default="clip")
    parser.add_argument("--voc-root", required=True, type=Path, help="Path to VOCdevkit/VOC2012.")
    parser.add_argument("--split", default="val", help="VOC segmentation split name.")
    parser.add_argument("--limit", default=None, type=int, help="Optional maximum number of images to evaluate.")
    parser.add_argument(
        "--output-dir",
        default=Path("outputs/dense_voc"),
        type=Path,
        help="Directory for metrics.json and per-image prediction npy files.",
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
    parser.add_argument("--model-name", default="ViT-B-16", help="open_clip model name.")
    parser.add_argument("--pretrained", default="openai", help="open_clip pretrained tag.")
    parser.add_argument("--save-debug", action="store_true", help="Save one lightweight debug JSON per image.")
    return parser


def main() -> None:
    """CLI entry point for dense-only Pascal VOC evaluation."""

    args = build_arg_parser().parse_args()
    try:
        metrics = evaluate_dataset(
            voc_root=args.voc_root,
            split=args.split,
            output_dir=args.output_dir,
            semantic_backend=args.semantic_backend,
            limit=args.limit,
            voc_mode=args.voc_mode,
            voc20_ignore_background=args.voc20_ignore_background,
            model_name=args.model_name,
            pretrained=args.pretrained,
            save_debug=args.save_debug,
        )
    except (FileNotFoundError, ValueError, RuntimeError, SemanticBackendError) as exc:
        raise SystemExit(f"Dense VOC evaluation error: {exc}") from exc

    print(f"Evaluated images: {metrics['evaluated_images']}")
    print(f"Skipped images: {metrics['skipped_images']}")
    print(f"mIoU: {metrics['mIoU']:.6f}")
    print(f"Semantic backend: {metrics['semantic_backend']}")
    print(f"VOC mode: {metrics['voc_mode']}")
    if metrics["voc_mode"] == "voc20":
        print(f"VOC20 ignore background: {metrics['voc20_ignore_background']}")
    if metrics["background_iou"] is not None:
        print(f"Background IoU: {metrics['background_iou']:.6f}")
    print(f"Predictions dir: {metrics['predictions_dir']}")
    print(f"Metrics JSON: {metrics['metrics_path']}")
    print("Per-class IoU:")
    for class_name, iou in metrics["per_class_iou"].items():
        value = "nan" if iou is None else f"{iou:.6f}"
        print(f"  {class_name}: {value}")


if __name__ == "__main__":
    main()
