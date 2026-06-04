import json
from pathlib import Path

import numpy as np
from PIL import Image

import eval_dense_voc
import eval_pascal_voc
from infer_ur_ovss import SemanticBackendError


def _create_fake_voc_dataset(tmp_path: Path, image_ids):
    """Create a tiny VOC-like dataset for unified dense evaluation tests."""

    voc_root = tmp_path / "VOC2012"
    (voc_root / "ImageSets" / "Segmentation").mkdir(parents=True)
    (voc_root / "JPEGImages").mkdir()
    (voc_root / "SegmentationClass").mkdir()
    (voc_root / "ImageSets" / "Segmentation" / "val.txt").write_text(
        "\n".join(image_ids) + "\n",
        encoding="utf-8",
    )

    for image_id in image_ids:
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        image[..., 0] = 96
        image[:4, :4, 1] = 220
        Image.fromarray(image, mode="RGB").save(voc_root / "JPEGImages" / f"{image_id}.jpg")

        target = np.zeros((8, 8), dtype=np.uint8)
        target[:4, :4] = 1
        target[4:, 4:] = 2
        target[0, 0] = 255
        Image.fromarray(target).save(voc_root / "SegmentationClass" / f"{image_id}.png")

    return voc_root


class FakeDenseAdapter:
    """Fake dense adapter with deterministic class logits."""

    description = "fake dense logits"

    def __init__(self, backend: str):
        """Store the requested backend name."""

        self.backend = backend
        self.image_shape = None

    def prepare_image(self, image, image_array):
        """Record image shape for dense logits."""

        del image
        self.image_shape = image_array.shape[:2]

    def dense_logits_for_classes(self, class_names, output_shape):
        """Return class logits that match the fake VOC target foreground."""

        height, width = output_shape
        logits = np.full((height, width, len(class_names)), -3.0, dtype=np.float32)
        if class_names[0] == "background":
            logits[..., 0] = 2.0
            logits[:4, :4, 1] = 5.0
            logits[4:, 4:, 2] = 5.0
        else:
            logits[..., 0] = 4.0
            logits[4:, 4:, 1] = 5.0
        return logits


def test_dense_eval_parser_accepts_clip_and_clearclip_backends():
    """Unified dense evaluator should accept clip and clearclip modes."""

    parser = eval_dense_voc.build_arg_parser()

    clip_args = parser.parse_args(["--voc-root", "VOC2012", "--semantic-backend", "clip"])
    clearclip_args = parser.parse_args(["--voc-root", "VOC2012", "--semantic-backend", "clearclip"])

    assert clip_args.semantic_backend == "clip"
    assert clearclip_args.semantic_backend == "clearclip"


def test_dense_eval_runs_clip_backend_with_fake_logits(monkeypatch, tmp_path):
    """Vanilla CLIP dense-only path should evaluate fake dense logits."""

    voc_root = _create_fake_voc_dataset(tmp_path, ["fake_0001"])
    requested_backends = []

    def fake_build_dense_adapter(backend, model_name="ViT-B-16", pretrained="openai"):
        """Build fake dense adapters while recording requested backend."""

        del model_name, pretrained
        requested_backends.append(backend)
        return FakeDenseAdapter(backend)

    monkeypatch.setattr(eval_dense_voc, "build_dense_adapter", fake_build_dense_adapter)

    metrics = eval_dense_voc.evaluate_dataset(
        voc_root=voc_root,
        split="val",
        output_dir=tmp_path / "dense_clip_eval",
        limit=1,
        semantic_backend="clip",
        voc_mode="voc20",
        voc20_ignore_background=True,
    )

    assert requested_backends == ["clip"]
    assert metrics["semantic_backend"] == "clip"
    assert metrics["mIoU"] == 1.0
    assert metrics["per_class_iou"]["aeroplane"] == 1.0
    assert metrics["per_class_iou"]["bicycle"] == 1.0
    assert Path(metrics["metrics_path"]).exists()
    assert len(list((tmp_path / "dense_clip_eval" / "predictions").glob("*.npy"))) == 1


def test_dense_eval_runs_clearclip_backend_with_fake_logits(monkeypatch, tmp_path):
    """ClearCLIP dense-only path should use the same unified evaluator."""

    voc_root = _create_fake_voc_dataset(tmp_path, ["fake_0001"])
    monkeypatch.setattr(eval_dense_voc, "build_dense_adapter", lambda backend, **kwargs: FakeDenseAdapter(backend))

    metrics = eval_dense_voc.evaluate_dataset(
        voc_root=voc_root,
        split="val",
        output_dir=tmp_path / "dense_clearclip_eval",
        limit=1,
        semantic_backend="clearclip",
        voc_mode="voc20",
        voc20_ignore_background=True,
        save_debug=True,
    )

    assert metrics["semantic_backend"] == "clearclip"
    assert metrics["mIoU"] == 1.0
    assert len(list((tmp_path / "dense_clearclip_eval" / "debug").glob("*.json"))) == 1


def test_dense_eval_voc21_metrics_are_correct_with_fake_logits(monkeypatch, tmp_path):
    """VOC21 mode should evaluate background plus foreground classes."""

    voc_root = _create_fake_voc_dataset(tmp_path, ["fake_0001"])
    monkeypatch.setattr(eval_dense_voc, "build_dense_adapter", lambda backend, **kwargs: FakeDenseAdapter(backend))

    metrics = eval_dense_voc.evaluate_dataset(
        voc_root=voc_root,
        split="val",
        output_dir=tmp_path / "dense_voc21_eval",
        limit=1,
        semantic_backend="clip",
        voc_mode="voc21",
    )

    assert metrics["classes"][0] == "background"
    assert metrics["background_iou"] == 1.0
    assert metrics["mIoU"] == 1.0
    saved = json.loads(Path(metrics["metrics_path"]).read_text(encoding="utf-8"))
    assert saved["per_class_iou"]["background"] == 1.0


def test_vanilla_clip_dense_grid_squeezes_batch_dimension():
    """Vanilla CLIP dense grids should drop a singleton batch dimension."""

    dense = np.arange(1 * 2 * 3 * 4, dtype=np.float32).reshape(1, 2, 3, 4)

    coerced = eval_dense_voc._coerce_dense_grid(dense, patch_grid=(2, 3), value_name="test dense grid")

    assert coerced.shape == (2, 3, 4)
    np.testing.assert_array_equal(coerced, dense[0])


def test_vanilla_clip_dense_grid_reshapes_flat_patch_tokens():
    """Flat [N, C] patch logits should reshape using the patch grid."""

    dense = np.arange(6 * 4, dtype=np.float32).reshape(6, 4)

    coerced = eval_dense_voc._coerce_dense_grid(dense, patch_grid=(2, 3), value_name="test dense grid")

    assert coerced.shape == (2, 3, 4)
    np.testing.assert_array_equal(coerced, dense.reshape(2, 3, 4))


def test_vanilla_clip_dense_grid_rejects_illegal_shape():
    """Illegal dense grid shapes should produce a clear error."""

    dense = np.zeros((1, 1, 2, 3, 4), dtype=np.float32)

    try:
        eval_dense_voc._coerce_dense_grid(dense, patch_grid=(2, 3), value_name="test dense grid")
    except SemanticBackendError as exc:
        assert "test dense grid" in str(exc)
        assert "(1, 1, 2, 3, 4)" in str(exc)
    else:
        raise AssertionError("Expected SemanticBackendError for illegal dense grid shape.")


def test_clearclip_dense_prompt_logits_path_still_works():
    """ClearCLIP adapters without dense_logits_for_classes should keep working."""

    class FakeClearClipPromptAdapter:
        """Fake ClearCLIP-style prompt-logit adapter."""

        def dense_logits_for_prompts(self, prompts):
            """Return prompt logits grouped by class templates."""

            logits = np.zeros((2, 2, len(prompts)), dtype=np.float32)
            for index, prompt in enumerate(prompts):
                if "aeroplane" in prompt:
                    logits[..., index] = 3.0
                elif "bicycle" in prompt:
                    logits[..., index] = 1.0
            return logits

    dense_logits = eval_dense_voc.compute_dense_logits_for_classes(
        FakeClearClipPromptAdapter(),
        ["aeroplane", "bicycle"],
        output_shape=(2, 2),
    )

    assert dense_logits.shape == (2, 2, 2)
    assert np.all(dense_logits[..., 0] > dense_logits[..., 1])


def test_original_pascal_eval_cli_does_not_expose_dense_backend():
    """The existing SAM/routing evaluator CLI should remain unchanged."""

    parser = eval_pascal_voc.build_arg_parser()

    args = parser.parse_args(["--voc-root", "VOC2012", "--semantic-backend", "clearclip"])

    assert args.semantic_backend == "clearclip"
    assert "dense" not in parser.format_help()
