import json
from pathlib import Path

import numpy as np
from PIL import Image

import eval_pascal_voc
import eval_clearclip_dense_voc


def _create_fake_voc_dataset(tmp_path: Path, image_ids):
    """Create a tiny VOC-like segmentation dataset for dense-only tests."""

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
        image[..., 0] = 120
        image[:4, :4, 1] = 220
        Image.fromarray(image, mode="RGB").save(voc_root / "JPEGImages" / f"{image_id}.jpg")

        target = np.zeros((8, 8), dtype=np.uint8)
        target[:4, :4] = 1
        target[0, 0] = 255
        Image.fromarray(target).save(voc_root / "SegmentationClass" / f"{image_id}.png")

    return voc_root


class FakeClearClipDenseAdapter:
    """Fake ClearCLIP adapter returning deterministic dense prompt logits."""

    description = "fake dense clearclip logits"

    def __init__(self, *args, **kwargs):
        """Accept the real adapter constructor signature."""

        del args, kwargs
        self.image_shape = None
        self.prompt_calls = []

    def prepare_image(self, image, image_array):
        """Record the image shape used by dense logits."""

        del image
        self.image_shape = image_array.shape[:2]

    def dense_logits_for_prompts(self, prompts):
        """Return high logits for aeroplane prompts and low logits otherwise."""

        self.prompt_calls.append(list(prompts))
        height, width = self.image_shape
        logits = np.zeros((height, width, len(prompts)), dtype=np.float32)
        for index, prompt in enumerate(prompts):
            if "aeroplane" in prompt:
                logits[..., index] = 3.0
            elif "background" in prompt:
                logits[..., index] = -1.0
            else:
                logits[..., index] = 0.0
        return logits


def test_dense_eval_parser_exposes_voc_background_mode():
    """Dense-only evaluator should expose explicit VOC20/VOC21 settings."""

    parser = eval_clearclip_dense_voc.build_arg_parser()

    args = parser.parse_args(
        [
            "--voc-root",
            "VOC2012",
            "--voc-mode",
            "voc21",
            "--voc20-ignore-background",
        ]
    )

    assert args.voc_mode == "voc21"
    assert args.voc20_ignore_background is True


def test_clearclip_dense_eval_runs_with_fake_dense_logits(monkeypatch, tmp_path):
    """Dense-only evaluation should consume image-sized ClearCLIP logits."""

    voc_root = _create_fake_voc_dataset(tmp_path, ["fake_0001"])
    monkeypatch.setattr(eval_clearclip_dense_voc, "ClearClipSemanticAdapter", FakeClearClipDenseAdapter)

    metrics = eval_clearclip_dense_voc.evaluate_dataset(
        voc_root=voc_root,
        split="val",
        output_dir=tmp_path / "dense_eval",
        limit=1,
        voc_mode="voc20",
        voc20_ignore_background=True,
    )

    assert metrics["evaluated_images"] == 1
    assert metrics["skipped_images"] == 0
    assert metrics["voc_mode"] == "voc20"
    assert metrics["voc20_ignore_background"] is True
    assert metrics["mIoU"] == 1.0
    assert metrics["per_class_iou"]["aeroplane"] == 1.0
    assert Path(metrics["metrics_path"]).exists()
    saved = json.loads(Path(metrics["metrics_path"]).read_text(encoding="utf-8"))
    assert saved["mIoU"] == 1.0


def test_clearclip_dense_eval_voc21_includes_background(monkeypatch, tmp_path):
    """VOC21 dense-only evaluation should include background in the metric JSON."""

    voc_root = _create_fake_voc_dataset(tmp_path, ["fake_0001"])
    monkeypatch.setattr(eval_clearclip_dense_voc, "ClearClipSemanticAdapter", FakeClearClipDenseAdapter)

    metrics = eval_clearclip_dense_voc.evaluate_dataset(
        voc_root=voc_root,
        split="val",
        output_dir=tmp_path / "dense_eval_voc21",
        limit=1,
        voc_mode="voc21",
    )

    assert metrics["classes"][0] == "background"
    assert "background" in metrics["per_class_iou"]
    assert metrics["background_iou"] == metrics["per_class_iou"]["background"]


def test_original_pascal_eval_still_defaults_to_fallback_voc20(tmp_path):
    """The existing SAM/routing evaluator should keep its default mode."""

    voc_root = _create_fake_voc_dataset(tmp_path, ["fake_0001"])

    metrics = eval_pascal_voc.evaluate_dataset(
        voc_root=voc_root,
        split="val",
        output_dir=tmp_path / "regular_eval",
        limit=1,
        semantic_backend="fallback",
        mask_backend="fallback",
        feature_backend="fallback",
    )

    assert metrics["voc_mode"] == "voc20"
    assert metrics["voc20_ignore_background"] is False
    assert "background" not in metrics["per_class_iou"]
