import builtins
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from infer_ur_ovss import (
    Dinov2FeatureAdapter,
    FallbackFeatureAdapter,
    FeatureBackendError,
    build_feature_adapter,
    run_inference,
)


def _small_rgb_image():
    """Create a tiny deterministic RGB image and numpy array."""

    image_array = np.zeros((8, 10, 3), dtype=np.float32)
    image_array[..., 0] = 0.2
    image_array[2:6, 3:8, 1] = 0.8
    image = Image.fromarray((image_array * 255).astype(np.uint8), mode="RGB")
    return image, image_array


def test_build_feature_adapter_returns_fallback_backend():
    """Feature backend selection should keep fallback DINO proxy features available."""

    adapter = build_feature_adapter("fallback")

    assert isinstance(adapter, FallbackFeatureAdapter)


def test_fallback_feature_adapter_outputs_dense_feature_map():
    """Fallback feature adapter should return normalized dense features."""

    image, image_array = _small_rgb_image()
    adapter = FallbackFeatureAdapter()

    features = adapter.extract_features(image, image_array)

    assert features.shape == (8, 10, 8)
    assert features.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(features, axis=-1), 1.0, rtol=1e-5, atol=1e-5)


def test_build_feature_adapter_rejects_unknown_backend():
    """Feature backend selection should reject unknown backend names."""

    with pytest.raises(ValueError, match="Unknown feature backend"):
        build_feature_adapter("unknown")


def test_dinov2_feature_adapter_missing_dependency_raises_clear_error(monkeypatch):
    """Missing transformers dependency should produce an actionable backend error."""

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        """Raise ImportError only for transformers imports."""

        if name == "transformers":
            raise ImportError("transformers intentionally unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(FeatureBackendError, match="requirements-dino.txt|transformers"):
        Dinov2FeatureAdapter()


class FailingTransformersModule:
    """Fake transformers module that fails while loading."""

    class AutoImageProcessor:
        """Fake processor loader."""

        @staticmethod
        def from_pretrained(model_name):
            """Raise a model-loading error."""

            raise RuntimeError(f"cannot load {model_name}")

    class Dinov2Model:
        """Fake model loader."""

        @staticmethod
        def from_pretrained(model_name):
            """Raise a model-loading error."""

            raise RuntimeError(f"cannot load {model_name}")


def test_dinov2_feature_adapter_load_failure_raises_clear_error():
    """DINOv2 model load failures should be wrapped in FeatureBackendError."""

    with pytest.raises(FeatureBackendError, match="Failed to load DINOv2 model"):
        Dinov2FeatureAdapter(transformers_module=FailingTransformersModule)


class FakeTensor:
    """Minimal tensor-like object for numpy-backed adapter tests."""

    def __init__(self, array):
        """Store the wrapped numpy array."""

        self.array = np.asarray(array, dtype=np.float32)

    @property
    def shape(self):
        """Return the wrapped array shape."""

        return self.array.shape

    def to(self, device):
        """Mirror torch Tensor.to."""

        return self

    def detach(self):
        """Mirror torch detach."""

        return self

    def cpu(self):
        """Mirror torch cpu."""

        return self

    def numpy(self):
        """Return the wrapped numpy array."""

        return self.array


class FakeBatch(dict):
    """Fake processor output that behaves like a mapping and supports .to()."""

    def to(self, device):
        """Mirror transformers BatchFeature.to."""

        return self


class FakeDinoOutput:
    """Fake DINOv2 output with patch tokens."""

    def __init__(self):
        """Build CLS plus a 2x3 patch grid with D=4 features."""

        patch_tokens = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        cls_token = np.zeros((1, 4), dtype=np.float32)
        self.last_hidden_state = FakeTensor(np.concatenate([cls_token, patch_tokens], axis=0)[None, ...])


class FakeDinoModel:
    """Fake DINOv2 model with a patch_size config."""

    config = type("Config", (), {"patch_size": 2})()

    def to(self, device):
        """Mirror torch Module.to."""

        return self

    def eval(self):
        """Mirror torch Module.eval."""

        return self

    def __call__(self, **inputs):
        """Return fake DINOv2 hidden states."""

        return FakeDinoOutput()


class FakeTransformersModule:
    """Fake transformers module for DINOv2 feature adapter tests."""

    class AutoImageProcessor:
        """Fake image processor class."""

        @staticmethod
        def from_pretrained(model_name):
            """Return a callable processor."""

            return FakeProcessor()

    class Dinov2Model:
        """Fake DINOv2 model class."""

        @staticmethod
        def from_pretrained(model_name):
            """Return a fake model."""

            return FakeDinoModel()


class FakeProcessor:
    """Fake processor that creates a 2x3 patch grid for patch_size=2."""

    def __call__(self, images, return_tensors):
        """Return fake pixel values with shape [1, 3, 4, 6]."""

        return FakeBatch({"pixel_values": FakeTensor(np.zeros((1, 3, 4, 6), dtype=np.float32))})


def test_dinov2_feature_adapter_outputs_image_sized_dense_features_with_fake_module():
    """DINOv2 adapter should output normalized dense features with shape [H, W, D]."""

    image, image_array = _small_rgb_image()
    adapter = Dinov2FeatureAdapter(transformers_module=FakeTransformersModule, device="cpu")

    features = adapter.extract_features(image, image_array)

    assert features.shape == (8, 10, 4)
    assert features.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(features, axis=-1), 1.0, rtol=1e-5, atol=1e-5)


def test_run_inference_fallback_feature_backend_writes_outputs(tmp_path):
    """run_inference should still work with fallback feature backend."""

    image, _ = _small_rgb_image()
    image_path = tmp_path / "input.png"
    output_path = tmp_path / "demo.png"
    image.save(image_path)

    result = run_inference(image_path, ["cat", "dog"], output_path, feature_backend="fallback")

    assert Path(result["outputs"]["visualization"]).exists()
    assert result["regions"]
    assert result["experts"]["purity"] == "fallback patch proxy features"
