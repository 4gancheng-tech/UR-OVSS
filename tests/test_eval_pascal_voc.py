import json
from pathlib import Path

import numpy as np
from PIL import Image

import eval_pascal_voc
from eval_pascal_voc import (
    VOC_CLASSES,
    build_arg_parser,
    compute_confusion_matrix,
    compute_iou_from_confusion,
    compute_voc_confusion_matrix,
    summarize_voc_metrics,
    evaluate_dataset,
    map_prediction_to_voc_labels,
)
from infer_ur_ovss import RegionSemanticScores


def _create_fake_voc_dataset(tmp_path, image_ids):
    """Create a tiny Pascal VOC-like dataset for evaluation tests."""

    voc_root = tmp_path / "VOC2012"
    (voc_root / "ImageSets" / "Segmentation").mkdir(parents=True)
    (voc_root / "JPEGImages").mkdir()
    (voc_root / "SegmentationClass").mkdir()
    (voc_root / "ImageSets" / "Segmentation" / "val.txt").write_text(
        "\n".join(image_ids) + "\n",
        encoding="utf-8",
    )

    for index, image_id in enumerate(image_ids):
        image_array = np.zeros((16, 20, 3), dtype=np.uint8)
        image_array[..., 0] = 64 + index
        image_array[4:12, 6:16, 1] = 180
        Image.fromarray(image_array, mode="RGB").save(voc_root / "JPEGImages" / f"{image_id}.jpg")

        target = np.zeros((16, 20), dtype=np.uint8)
        target[4:12, 6:16] = 1
        target[0, 0] = 255
        Image.fromarray(target).save(voc_root / "SegmentationClass" / f"{image_id}.png")

    return voc_root


def test_voc_class_list_contains_20_foreground_classes():
    """Pascal VOC semantic segmentation uses 20 foreground classes."""

    assert len(VOC_CLASSES) == 20
    assert VOC_CLASSES[0] == "aeroplane"
    assert VOC_CLASSES[-1] == "tvmonitor"


def test_eval_parser_accepts_clearclip_semantic_backend():
    """VOC evaluation CLI should accept the ClearCLIP semantic backend."""

    parser = build_arg_parser()

    args = parser.parse_args(
        [
            "--voc-root",
            "VOC2012",
            "--semantic-backend",
            "clearclip",
        ]
    )

    assert args.semantic_backend == "clearclip"


def test_eval_parser_accepts_voc_mode():
    """VOC evaluation CLI should expose explicit VOC20/VOC21 modes."""

    parser = build_arg_parser()

    args = parser.parse_args(["--voc-root", "VOC2012", "--voc-mode", "voc21"])

    assert args.voc_mode == "voc21"


def test_confusion_matrix_ignores_255_labels():
    """Confusion matrix should ignore target pixels labeled 255."""

    pred = np.array([[1, 2], [2, 0]], dtype=np.int64)
    target = np.array([[1, 255], [1, 0]], dtype=np.int64)

    confusion = compute_confusion_matrix(pred, target, num_classes=3, ignore_index=255)

    expected = np.array(
        [
            [1, 0, 0],
            [0, 1, 1],
            [0, 0, 0],
        ],
        dtype=np.int64,
    )
    np.testing.assert_array_equal(confusion, expected)


def test_prediction_mapping_converts_unassigned_to_background_and_foreground_plus_one():
    """Model labels should map -1 to background and 0-19 to VOC labels 1-20."""

    pred = np.array([[-1, 0, 1, 19]], dtype=np.int32)

    mapped = map_prediction_to_voc_labels(pred)

    np.testing.assert_array_equal(mapped, np.array([[0, 1, 2, 20]], dtype=np.int64))


def test_iou_computation_returns_nan_for_absent_classes():
    """IoU should be intersection over union with NaN for absent classes."""

    confusion = np.array(
        [
            [2, 0, 0],
            [0, 1, 1],
            [0, 1, 3],
        ],
        dtype=np.int64,
    )

    iou = compute_iou_from_confusion(confusion)

    np.testing.assert_allclose(iou, np.array([1.0, 1.0 / 3.0, 3.0 / 5.0]), rtol=1e-6)


def test_voc20_and_voc21_background_handling_compute_expected_iou():
    """VOC20/VOC21 should differ in background handling while ignoring 255."""

    pred_raw = np.array(
        [
            [-1, 0, 1, 0],
            [0, 1, -1, -1],
        ],
        dtype=np.int32,
    )
    target = np.array(
        [
            [0, 1, 1, 255],
            [0, 2, 2, 0],
        ],
        dtype=np.uint8,
    )

    confusion_default = compute_voc_confusion_matrix(pred_raw, target, voc_mode="voc20")
    metrics_default = summarize_voc_metrics(confusion_default, voc_mode="voc20")
    metrics_voc21 = summarize_voc_metrics(confusion_default, voc_mode="voc21")
    confusion_ignore_bg = compute_voc_confusion_matrix(
        pred_raw,
        target,
        voc_mode="voc20",
        voc20_ignore_background=True,
    )
    metrics_ignore_bg = summarize_voc_metrics(confusion_ignore_bg, voc_mode="voc20")

    np.testing.assert_allclose(metrics_default["per_class_iou"]["aeroplane"], 1.0 / 3.0, rtol=1e-6)
    np.testing.assert_allclose(metrics_default["per_class_iou"]["bicycle"], 1.0 / 3.0, rtol=1e-6)
    assert "background" not in metrics_default["per_class_iou"]
    np.testing.assert_allclose(metrics_default["mIoU"], 1.0 / 3.0, rtol=1e-6)
    np.testing.assert_allclose(metrics_voc21["per_class_iou"]["background"], 0.5, rtol=1e-6)
    np.testing.assert_allclose(metrics_voc21["mIoU"], (0.5 + 1.0 / 3.0 + 1.0 / 3.0) / 3.0, rtol=1e-6)
    np.testing.assert_allclose(metrics_ignore_bg["per_class_iou"]["aeroplane"], 0.5, rtol=1e-6)
    np.testing.assert_allclose(metrics_ignore_bg["mIoU"], (0.5 + 1.0 / 3.0) / 2.0, rtol=1e-6)


def test_evaluate_dataset_runs_on_fake_voc_with_fallback_backend(tmp_path):
    """VOC evaluation should run end-to-end on a tiny fake dataset."""

    voc_root = _create_fake_voc_dataset(tmp_path, ["fake_0001"])

    output_dir = tmp_path / "voc_eval"
    metrics = evaluate_dataset(
        voc_root=voc_root,
        split="val",
        output_dir=output_dir,
        limit=1,
        semantic_backend="fallback",
        mask_backend="fallback",
        feature_backend="fallback",
        save_vis=False,
    )

    assert metrics["evaluated_images"] == 1
    assert metrics["skipped_images"] == 0
    assert len(metrics["per_class_iou"]) == 20
    assert metrics["voc_mode"] == "voc20"
    assert metrics["voc20_ignore_background"] is False
    assert metrics["background_iou"] is None
    assert "background" not in metrics["per_class_iou"]
    assert Path(metrics["metrics_path"]).exists()
    saved_metrics = json.loads(Path(metrics["metrics_path"]).read_text(encoding="utf-8"))
    assert {
        "split",
        "voc_root",
        "voc_mode",
        "voc20_ignore_background",
        "background_iou",
        "mIoU",
        "per_class_iou",
        "evaluated_images",
        "skipped_images",
        "skipped",
        "classes",
        "metrics_path",
    } <= set(saved_metrics)
    assert saved_metrics["evaluated_images"] == 1
    assert not list((output_dir / "visualizations").glob("*.png"))


def test_evaluate_dataset_voc21_reports_background_class(tmp_path):
    """VOC21 mode should include background in classes and mIoU."""

    voc_root = _create_fake_voc_dataset(tmp_path, ["fake_0001"])

    metrics = evaluate_dataset(
        voc_root=voc_root,
        split="val",
        output_dir=tmp_path / "voc_eval_21",
        limit=1,
        semantic_backend="fallback",
        mask_backend="fallback",
        feature_backend="fallback",
        voc_mode="voc21",
        save_vis=False,
    )

    assert metrics["voc_mode"] == "voc21"
    assert metrics["classes"][0] == "background"
    assert "background" in metrics["per_class_iou"]
    assert metrics["background_iou"] == metrics["per_class_iou"]["background"]


def test_evaluate_dataset_initializes_backends_once_with_reusable_adapters(monkeypatch, tmp_path):
    """VOC evaluation should reuse initialized adapters across images."""

    voc_root = _create_fake_voc_dataset(tmp_path, ["fake_0001", "fake_0002"])
    init_counts = {"semantic": 0, "mask": 0, "feature": 0}

    class CountingSemanticAdapter:
        """Fake semantic adapter that records construction count."""

        description = "counting semantic"

        def __init__(self):
            init_counts["semantic"] += 1

        def prepare_image(self, image, image_array):
            """Accept per-image preparation without loading a model."""

            del image, image_array

        def score_region(self, mask, class_names, positive_prompts, negative_prompts):
            """Return deterministic region scores with the expected shapes."""

            del mask
            base_scores = np.linspace(1.0, 0.0, len(class_names), dtype=np.float32)
            positive_scores = np.repeat(
                base_scores[:, None],
                len(positive_prompts[class_names[0]]),
                axis=1,
            )
            negative_scores = np.zeros(
                (len(class_names), len(negative_prompts[class_names[0]])),
                dtype=np.float32,
            )
            return RegionSemanticScores(
                base_scores=base_scores,
                positive_scores=positive_scores,
                negative_scores=negative_scores,
                prompt_rescore_scores=base_scores,
            )

    class CountingMaskAdapter:
        """Fake mask adapter that records construction count."""

        description = "counting masks"

        def __init__(self):
            init_counts["mask"] += 1

        def generate_masks(self, image, image_array):
            """Return one full-image mask per input image."""

            del image
            return [
                {
                    "segmentation": np.ones(image_array.shape[:2], dtype=bool),
                    "source": "counting_mask",
                }
            ]

    class CountingFeatureAdapter:
        """Fake feature adapter that records construction count."""

        description = "counting features"

        def __init__(self):
            init_counts["feature"] += 1

        def extract_features(self, image, image_array):
            """Return a normalized dense feature map with shape [H, W, D]."""

            del image
            features = np.ones((*image_array.shape[:2], 4), dtype=np.float32)
            return features / np.linalg.norm(features, axis=-1, keepdims=True)

    monkeypatch.setattr(eval_pascal_voc, "build_semantic_adapter", lambda backend: CountingSemanticAdapter())
    monkeypatch.setattr(
        eval_pascal_voc,
        "build_mask_adapter",
        lambda backend, sam_checkpoint=None, sam_model_type="vit_b", max_masks=100: CountingMaskAdapter(),
    )
    monkeypatch.setattr(
        eval_pascal_voc,
        "build_feature_adapter",
        lambda feature_backend, dinov2_model="facebook/dinov2-small": CountingFeatureAdapter(),
    )

    output_dir = tmp_path / "voc_eval_reuse"
    metrics = evaluate_dataset(
        voc_root=voc_root,
        split="val",
        output_dir=output_dir,
        limit=2,
        semantic_backend="fallback",
        mask_backend="fallback",
        feature_backend="fallback",
        save_vis=False,
    )

    assert metrics["evaluated_images"] == 2
    assert init_counts == {"semantic": 1, "mask": 1, "feature": 1}


def test_evaluate_dataset_runs_with_fake_clearclip_adapter(monkeypatch, tmp_path):
    """VOC evaluation should accept the clearclip semantic backend."""

    voc_root = _create_fake_voc_dataset(tmp_path, ["fake_0001"])
    requested_backends = []

    class FakeClearClipAdapter:
        """Fake dense semantic adapter for eval plumbing."""

        description = "fake clearclip dense logits"

        def prepare_image(self, image, image_array):
            """Accept per-image preparation."""

            del image, image_array

        def score_region(self, mask, class_names, positive_prompts, negative_prompts):
            """Return valid semantic scores."""

            del mask
            base_scores = np.linspace(1.0, 0.0, len(class_names), dtype=np.float32)
            positive_scores = np.repeat(
                base_scores[:, None],
                len(positive_prompts[class_names[0]]),
                axis=1,
            )
            negative_scores = np.zeros(
                (len(class_names), len(negative_prompts[class_names[0]])),
                dtype=np.float32,
            )
            return RegionSemanticScores(
                base_scores=base_scores,
                positive_scores=positive_scores,
                negative_scores=negative_scores,
                prompt_rescore_scores=base_scores,
            )

    def fake_build_semantic_adapter(backend):
        """Build only the fake clearclip backend for this test."""

        requested_backends.append(backend)
        return FakeClearClipAdapter()

    monkeypatch.setattr(eval_pascal_voc, "build_semantic_adapter", fake_build_semantic_adapter)

    metrics = evaluate_dataset(
        voc_root=voc_root,
        split="val",
        output_dir=tmp_path / "voc_clearclip_eval",
        limit=1,
        semantic_backend="clearclip",
        mask_backend="fallback",
        feature_backend="fallback",
    )

    assert requested_backends == ["clearclip"]
    assert metrics["evaluated_images"] == 1
