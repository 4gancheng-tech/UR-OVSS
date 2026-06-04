import json
from pathlib import Path

import numpy as np
from PIL import Image

import eval_dense_voc
import eval_pascal_voc


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


def test_original_pascal_eval_cli_does_not_expose_dense_backend():
    """The existing SAM/routing evaluator CLI should remain unchanged."""

    parser = eval_pascal_voc.build_arg_parser()

    args = parser.parse_args(["--voc-root", "VOC2012", "--semantic-backend", "clearclip"])

    assert args.semantic_backend == "clearclip"
    assert "dense" not in parser.format_help()
