import json
from pathlib import Path

import numpy as np
from PIL import Image

from eval_pascal_voc import (
    VOC_CLASSES,
    compute_confusion_matrix,
    compute_iou_from_confusion,
    evaluate_dataset,
    map_prediction_to_voc_labels,
)


def test_voc_class_list_contains_20_foreground_classes():
    """Pascal VOC semantic segmentation uses 20 foreground classes."""

    assert len(VOC_CLASSES) == 20
    assert VOC_CLASSES[0] == "aeroplane"
    assert VOC_CLASSES[-1] == "tvmonitor"


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


def test_evaluate_dataset_runs_on_fake_voc_with_fallback_backend(tmp_path):
    """VOC evaluation should run end-to-end on a tiny fake dataset."""

    voc_root = tmp_path / "VOC2012"
    (voc_root / "ImageSets" / "Segmentation").mkdir(parents=True)
    (voc_root / "JPEGImages").mkdir()
    (voc_root / "SegmentationClass").mkdir()
    (voc_root / "ImageSets" / "Segmentation" / "val.txt").write_text("fake_0001\n", encoding="utf-8")

    image_array = np.zeros((16, 20, 3), dtype=np.uint8)
    image_array[..., 0] = 64
    image_array[4:12, 6:16, 1] = 180
    Image.fromarray(image_array, mode="RGB").save(voc_root / "JPEGImages" / "fake_0001.jpg")

    target = np.zeros((16, 20), dtype=np.uint8)
    target[4:12, 6:16] = 1
    target[0, 0] = 255
    Image.fromarray(target).save(voc_root / "SegmentationClass" / "fake_0001.png")

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
    assert Path(metrics["metrics_path"]).exists()
    saved_metrics = json.loads(Path(metrics["metrics_path"]).read_text(encoding="utf-8"))
    assert saved_metrics["evaluated_images"] == 1
    assert not list((output_dir / "visualizations").glob("*.png"))
