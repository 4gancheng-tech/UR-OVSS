import builtins
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import infer_ur_ovss
from infer_ur_ovss import (
    FallbackMaskAdapter,
    SamMaskAdapter,
    MaskBackendError,
    build_mask_adapter,
    run_inference,
)


def _small_rgb_image():
    """Create a tiny deterministic RGB image and numpy array."""

    image_array = np.zeros((8, 10, 3), dtype=np.float32)
    image_array[..., 0] = 0.2
    image_array[2:6, 3:8, 1] = 0.8
    image = Image.fromarray((image_array * 255).astype(np.uint8), mode="RGB")
    return image, image_array


def test_build_mask_adapter_returns_fallback_backend():
    """Mask backend selection should keep fallback masks available."""

    adapter = build_mask_adapter("fallback")

    assert isinstance(adapter, FallbackMaskAdapter)


def test_build_mask_adapter_rejects_unknown_backend():
    """Mask backend selection should reject unknown backend names."""

    with pytest.raises(ValueError, match="Unknown mask backend"):
        build_mask_adapter("unknown")


def test_build_mask_adapter_rejects_non_positive_max_masks():
    """Mask backend selection should require max_masks to be positive."""

    with pytest.raises(MaskBackendError, match="positive integer"):
        build_mask_adapter("fallback", max_masks=0)
    with pytest.raises(MaskBackendError, match="positive integer"):
        build_mask_adapter("fallback", max_masks=-3)


def test_sam_mask_adapter_requires_checkpoint():
    """SAM backend should require an explicit checkpoint path."""

    with pytest.raises(MaskBackendError, match="sam-checkpoint"):
        SamMaskAdapter(checkpoint_path=None)


def test_sam_mask_adapter_rejects_missing_checkpoint(tmp_path):
    """SAM backend should reject nonexistent checkpoint paths."""

    missing_checkpoint = tmp_path / "missing_sam.pth"

    with pytest.raises(MaskBackendError, match="does not exist"):
        SamMaskAdapter(checkpoint_path=missing_checkpoint)


def test_sam_mask_adapter_missing_dependency_raises_clear_error(monkeypatch, tmp_path):
    """Missing SAM dependency should produce an actionable backend error."""

    checkpoint = tmp_path / "sam.pth"
    checkpoint.write_bytes(b"fake")
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        """Raise ImportError only for segment_anything imports."""

        if name in {"segment_anything", "mobile_sam"}:
            raise ImportError("segment_anything intentionally unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(MaskBackendError, match="requirements-sam.txt|segment-anything|mobile-sam"):
        SamMaskAdapter(checkpoint_path=checkpoint)


class FakeSamModule:
    """Small segment_anything stand-in for adapter format tests."""

    sam_model_registry = {"vit_b": lambda checkpoint: FakeSamModel(checkpoint)}

    class SamAutomaticMaskGenerator:
        """Fake automatic mask generator returning mixed mask records."""

        def __init__(self, model):
            """Store the fake SAM model."""

            self.model = model

        def generate(self, image_array):
            """Return fake SAM masks in segment_anything format."""

            first = np.zeros(image_array.shape[:2], dtype=bool)
            first[:4, :5] = True
            second = np.zeros(image_array.shape[:2], dtype=bool)
            second[4:, 5:] = True
            return [
                {"segmentation": first, "area": int(first.sum())},
                {"segmentation": second.astype(np.uint8), "area": int(second.sum())},
            ]


class FakeSamModel:
    """Fake SAM model that records checkpoint and device usage."""

    def __init__(self, checkpoint):
        """Store checkpoint path."""

        self.checkpoint = checkpoint
        self.device = None

    def to(self, device):
        """Mirror torch Module.to."""

        self.device = device
        return self

    def eval(self):
        """Mirror torch Module.eval."""

        return self


def test_sam_mask_adapter_outputs_compatible_mask_records_with_fake_module(tmp_path):
    """SAM adapter should return bool segmentation records compatible with fusion."""

    checkpoint = tmp_path / "sam.pth"
    checkpoint.write_bytes(b"fake")
    image, image_array = _small_rgb_image()
    adapter = SamMaskAdapter(
        checkpoint_path=checkpoint,
        model_type="vit_b",
        device="cpu",
        max_masks=1,
        sam_module=FakeSamModule,
    )

    masks = adapter.generate_masks(image, image_array)

    assert len(masks) == 1
    assert masks[0]["segmentation"].shape == image_array.shape[:2]
    assert masks[0]["segmentation"].dtype == bool
    assert masks[0]["source"] == "sam_vit_b"


class EmptyMaskAdapter:
    """Fake mask backend that returns no masks."""

    description = "empty fake masks"

    def generate_masks(self, image, image_array):
        """Return no masks."""

        return []


def test_run_inference_raises_when_mask_backend_returns_empty(monkeypatch, tmp_path):
    """run_inference should reject mask backends that produce no masks."""

    image, _ = _small_rgb_image()
    image_path = tmp_path / "input.png"
    output_path = tmp_path / "demo.png"
    image.save(image_path)

    monkeypatch.setattr(infer_ur_ovss, "build_mask_adapter", lambda *args, **kwargs: EmptyMaskAdapter())

    with pytest.raises(MaskBackendError, match="did not generate any mask"):
        run_inference(image_path, ["cat", "dog"], output_path, mask_backend="fallback")
