import json
from pathlib import Path

import numpy as np
from PIL import Image

from analyze_voc_outputs import analyze_eval_outputs


def _write_fake_eval_record(eval_dir, voc_root, image_id, pred_raw, target, confidence, regions):
    """Write one fake VOC evaluation record with prediction/debug outputs."""

    prediction_dir = eval_dir / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    mask_path = prediction_dir / f"{image_id}_mask.npy"
    confidence_path = prediction_dir / f"{image_id}_confidence.npy"
    debug_path = prediction_dir / f"{image_id}.json"

    np.save(mask_path, pred_raw)
    np.save(confidence_path, confidence)
    debug_path.write_text(
        json.dumps(
            {
                "image": str(voc_root / "JPEGImages" / f"{image_id}.jpg"),
                "regions": regions,
                "outputs": {
                    "mask_npy": str(mask_path),
                    "confidence_npy": str(confidence_path),
                    "debug_json": str(debug_path),
                },
            }
        ),
        encoding="utf-8",
    )
    Image.fromarray(target.astype(np.uint8)).save(voc_root / "SegmentationClass" / f"{image_id}.png")


def test_analyze_eval_outputs_writes_diagnostics_json_for_fake_eval_output(tmp_path):
    """Diagnostics should summarize per-image failures and global routing stats."""

    voc_root = tmp_path / "VOC2012"
    (voc_root / "JPEGImages").mkdir(parents=True)
    (voc_root / "SegmentationClass").mkdir()
    eval_dir = tmp_path / "voc_eval"
    eval_dir.mkdir()
    (eval_dir / "metrics.json").write_text(
        json.dumps({"mIoU": 0.25, "evaluated_images": 2, "skipped_images": 0}),
        encoding="utf-8",
    )

    _write_fake_eval_record(
        eval_dir=eval_dir,
        voc_root=voc_root,
        image_id="img_a",
        pred_raw=np.array([[0, 0, -1], [-1, -1, -1], [-1, 1, -1]], dtype=np.int32),
        target=np.array([[1, 1, 0], [0, 255, 0], [2, 2, 0]], dtype=np.uint8),
        confidence=np.full((3, 3), 0.5, dtype=np.float32),
        regions=[
            {"semantic_uncertain": True, "spatial_uncertain": False, "route_type": "semantic_rescore"},
            {"semantic_uncertain": False, "spatial_uncertain": True, "route_type": "spatial_downweight"},
        ],
    )
    _write_fake_eval_record(
        eval_dir=eval_dir,
        voc_root=voc_root,
        image_id="img_b",
        pred_raw=np.array([[0, 1, 1], [0, 1, -1], [-1, -1, -1]], dtype=np.int32),
        target=np.array([[0, 0, 0], [1, 1, 0], [0, 0, 0]], dtype=np.uint8),
        confidence=np.full((3, 3), 0.25, dtype=np.float32),
        regions=[
            {"semantic_uncertain": True, "spatial_uncertain": True, "route_type": "multi_expert_average"},
        ],
    )

    output_json = eval_dir / "diagnostics.json"
    diagnostics = analyze_eval_outputs(eval_dir=eval_dir, voc_root=voc_root, output_json=output_json)

    assert output_json.exists()
    saved = json.loads(output_json.read_text(encoding="utf-8"))
    assert diagnostics == saved
    assert set(saved) == {"eval_dir", "voc_root", "metrics", "num_images", "images", "summary"}
    assert saved["metrics"]["mIoU"] == 0.25
    assert saved["num_images"] == 2

    image_a = {record["image_id"]: record for record in saved["images"]}["img_a"]
    assert set(image_a) == {
        "image_id",
        "foreground_iou",
        "predicted_foreground_pixel_ratio",
        "gt_foreground_pixel_ratio",
        "predicted_class_distribution",
        "gt_class_distribution",
        "num_regions",
        "average_confidence",
        "semantic_uncertain_regions",
        "spatial_uncertain_regions",
        "route_type_counts",
    }
    assert image_a["num_regions"] == 2
    assert image_a["semantic_uncertain_regions"] == 1
    assert image_a["spatial_uncertain_regions"] == 1
    assert image_a["route_type_counts"] == {"semantic_rescore": 1, "spatial_downweight": 1}
    assert image_a["predicted_class_distribution"]["aeroplane"]["pixels"] == 2
    assert image_a["gt_class_distribution"]["bicycle"]["pixels"] == 2

    summary = saved["summary"]
    assert set(summary) == {
        "worst_20_images",
        "best_20_images",
        "most_predicted_classes",
        "most_missed_gt_classes",
        "images_with_foreground_over_prediction",
        "images_with_foreground_under_prediction",
        "global_route_type_distribution",
    }
    assert summary["worst_20_images"][0]["image_id"] == "img_b"
    assert summary["best_20_images"][0]["image_id"] == "img_a"
    assert summary["global_route_type_distribution"] == {
        "semantic_rescore": 1,
        "spatial_downweight": 1,
        "multi_expert_average": 1,
    }


def test_readme_documents_voc_diagnostics():
    """README should include the VOC diagnostics command."""

    readme = Path("README.md").read_text(encoding="utf-8")

    assert "VOC Evaluation Diagnostics" in readme
    assert "analyze_voc_outputs.py" in readme
    assert "--eval-dir outputs/voc_real_fullval" in readme
