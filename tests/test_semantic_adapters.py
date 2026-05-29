import builtins
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from infer_ur_ovss import (
    FallbackSemanticAdapter,
    OpenClipSemanticAdapter,
    SemanticBackendError,
    build_semantic_adapter,
    clip_inference_context,
    parse_class_names,
    run_inference,
)
from prompts import build_negative_prompts, build_positive_prompts


def _small_rgb_image():
    """Create a tiny deterministic RGB image and its numpy array."""

    image_array = np.zeros((8, 10, 3), dtype=np.float32)
    image_array[..., 0] = 0.25
    image_array[2:6, 3:8, 1] = 0.75
    image = Image.fromarray((image_array * 255).astype(np.uint8), mode="RGB")
    return image, image_array


def _prompt_sets(class_names):
    """Build positive and negative prompts for adapter tests."""

    return build_positive_prompts(class_names), build_negative_prompts(class_names)


def test_build_semantic_adapter_returns_fallback_backend():
    """Semantic backend selection should keep the fallback path available."""

    adapter = build_semantic_adapter("fallback")

    assert isinstance(adapter, FallbackSemanticAdapter)


def test_build_semantic_adapter_rejects_unknown_backend():
    """Semantic backend selection should reject unknown backend names."""

    with pytest.raises(ValueError, match="Unknown semantic backend"):
        build_semantic_adapter("unknown")


def test_parse_class_names_requires_at_least_two_classes():
    """Class parsing should require two classes for top1-top2 margin."""

    with pytest.raises(SystemExit, match="at least two"):
        parse_class_names("cat")
    assert parse_class_names("cat, dog") == ["cat", "dog"]


def test_fallback_semantic_adapter_scores_region_shapes():
    """Fallback semantic adapter should return CLIP-like score shapes."""

    image, image_array = _small_rgb_image()
    class_names = ["cat", "dog"]
    positive_prompts, negative_prompts = _prompt_sets(class_names)
    mask = np.zeros(image_array.shape[:2], dtype=bool)
    mask[1:7, 2:9] = True
    adapter = FallbackSemanticAdapter()
    adapter.prepare_image(image, image_array)

    scores = adapter.score_region(mask, class_names, positive_prompts, negative_prompts)

    assert scores.base_scores.shape == (2,)
    assert scores.positive_scores.shape == (2, 5)
    assert scores.negative_scores.shape == (2, 2)
    assert scores.prompt_rescore_scores.shape == (2,)


def test_run_inference_fallback_writes_outputs_and_regions(tmp_path):
    """Fallback run_inference should save outputs and region debug records."""

    image, _ = _small_rgb_image()
    image_path = tmp_path / "input.png"
    output_path = tmp_path / "demo.png"
    image.save(image_path)

    result = run_inference(image_path, ["cat", "dog"], output_path, semantic_backend="fallback")

    assert result["regions"]
    for output_file in result["outputs"].values():
        assert Path(output_file).exists()
    first_region = result["regions"][0]
    assert {"base_scores", "positive_scores", "negative_scores", "prompt_rescore_scores"} <= set(first_region)
    assert len(first_region["base_scores"]) == 2
    assert len(first_region["positive_scores"]) == 2
    assert len(first_region["positive_scores"][0]) == 5
    assert len(first_region["negative_scores"]) == 2
    assert len(first_region["negative_scores"][0]) == 2


class FakeOpenClipModule:
    """Small open_clip stand-in used to test adapter shape logic."""

    @staticmethod
    def create_model_and_transforms(model_name, pretrained, device):
        """Return a fake model and preprocess function."""

        return FakeClipModel(), None, FakePreprocess()

    @staticmethod
    def get_tokenizer(model_name):
        """Return a tokenizer that keeps track of prompt count."""

        return lambda prompts: FakeTokenBatch(len(prompts))


class FakeTensor:
    """Minimal tensor-like object for numpy-backed adapter tests."""

    def __init__(self, array):
        """Store the wrapped numpy array."""

        self.array = np.asarray(array, dtype=np.float32)

    def unsqueeze(self, axis):
        """Add a tensor dimension."""

        return FakeTensor(np.expand_dims(self.array, axis))

    def to(self, device):
        """Mirror torch Tensor.to for adapter compatibility."""

        return self

    @property
    def T(self):
        """Return a transposed fake tensor."""

        return FakeTensor(self.array.T)

    def norm(self, dim=-1, keepdim=True):
        """Compute vector norms."""

        return FakeTensor(np.linalg.norm(self.array, axis=dim, keepdims=keepdim))

    def __truediv__(self, other):
        """Divide fake tensors or arrays."""

        other_array = other.array if isinstance(other, FakeTensor) else other
        return FakeTensor(self.array / other_array)

    def __matmul__(self, other):
        """Matrix multiply fake tensors."""

        other_array = other.array if isinstance(other, FakeTensor) else other
        return FakeTensor(self.array @ other_array)

    def detach(self):
        """Mirror torch detach."""

        return self

    def cpu(self):
        """Mirror torch cpu."""

        return self

    def numpy(self):
        """Return the wrapped numpy array."""

        return self.array


class FakeTokenBatch:
    """Tokenizer result carrying only prompt count."""

    def __init__(self, count):
        """Store prompt count."""

        self.count = count

    def to(self, device):
        """Mirror torch Tensor.to for token batches."""

        return self


class FakePreprocess:
    """Preprocess PIL images into fake image tensors."""

    def __call__(self, image):
        """Return a fixed feature seed tensor."""

        return FakeTensor(np.array([1.0, 0.5, 0.25], dtype=np.float32))


class FakeClipModel:
    """Fake CLIP model with deterministic image/text features."""

    def eval(self):
        """Mirror torch eval."""

        return self

    def encode_image(self, image_tensor):
        """Return one image feature with shape [1, D]."""

        return FakeTensor(np.array([[1.0, 0.0, 0.0]], dtype=np.float32))

    def encode_text(self, tokens):
        """Return text features with shape [T, D]."""

        rows = []
        for index in range(tokens.count):
            rows.append([1.0, float(index + 1), 0.5])
        return FakeTensor(np.asarray(rows, dtype=np.float32))


def test_open_clip_semantic_adapter_scores_region_shapes_with_fake_module():
    """CLIP semantic adapter should expose the same score shapes as fallback."""

    image, image_array = _small_rgb_image()
    class_names = ["cat", "dog", "car"]
    positive_prompts, negative_prompts = _prompt_sets(class_names)
    mask = np.zeros(image_array.shape[:2], dtype=bool)
    mask[2:6, 3:8] = True
    adapter = OpenClipSemanticAdapter(open_clip_module=FakeOpenClipModule())
    adapter.prepare_image(image, image_array)

    scores = adapter.score_region(mask, class_names, positive_prompts, negative_prompts)

    assert scores.base_scores.shape == (3,)
    assert scores.positive_scores.shape == (3, 5)
    assert scores.negative_scores.shape == (3, 2)
    assert scores.prompt_rescore_scores.shape == (3,)


def test_open_clip_semantic_adapter_missing_dependency_raises_clear_error(monkeypatch):
    """Missing open_clip dependency should produce an actionable backend error."""

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        """Raise ImportError only for open_clip imports."""

        if name == "open_clip":
            raise ImportError("open_clip intentionally unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(SemanticBackendError, match="requirements-clip.txt|open_clip_torch"):
        OpenClipSemanticAdapter()


def test_clip_inference_context_uses_torch_inference_mode(monkeypatch):
    """CLIP scoring should request torch inference mode when torch exists."""

    calls = []
    real_import = builtins.__import__

    class FakeInferenceMode:
        """Tiny context manager returned by fake torch.inference_mode."""

        def __enter__(self):
            """Enter fake inference context."""

            calls.append("enter")

        def __exit__(self, exc_type, exc, traceback):
            """Exit fake inference context."""

            calls.append("exit")
            return False

    class FakeTorch:
        """Fake torch module exposing inference_mode."""

        @staticmethod
        def inference_mode():
            """Return a fake inference mode context."""

            calls.append("factory")
            return FakeInferenceMode()

    def fake_import(name, *args, **kwargs):
        """Return fake torch while preserving all other imports."""

        if name == "torch":
            return FakeTorch
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with clip_inference_context():
        calls.append("body")

    assert calls == ["factory", "enter", "body", "exit"]
