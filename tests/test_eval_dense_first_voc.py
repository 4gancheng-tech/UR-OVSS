import json
from pathlib import Path

import numpy as np
from PIL import Image

import eval_dense_first_voc


def _create_fake_voc_dataset(tmp_path: Path) -> Path:
    """Create a tiny VOC-like dataset for dense-first tests."""

    voc_root = tmp_path / "VOC2012"
    (voc_root / "ImageSets" / "Segmentation").mkdir(parents=True)
    (voc_root / "JPEGImages").mkdir()
    (voc_root / "SegmentationClass").mkdir()
    (voc_root / "ImageSets" / "Segmentation" / "val.txt").write_text("fake_0001\n", encoding="utf-8")

    image = np.zeros((4, 4, 3), dtype=np.uint8)
    image[..., 0] = 128
    Image.fromarray(image, mode="RGB").save(voc_root / "JPEGImages" / "fake_0001.jpg")

    target = np.zeros((4, 4), dtype=np.uint8)
    target[:, :] = 1
    Image.fromarray(target).save(voc_root / "SegmentationClass" / "fake_0001.png")
    return voc_root


class FakeDenseAdapter:
    """Fake dense adapter that emits deterministic logits."""

    description = "fake dense adapter"

    def prepare_image(self, image, image_array):
        """Record nothing; logits are shape-driven."""

        del image, image_array

    def dense_logits_for_classes(self, class_names, output_shape):
        """Return logits favoring the first foreground class."""

        height, width = output_shape
        logits = np.full((height, width, len(class_names)), -4.0, dtype=np.float32)
        logits[..., 0] = 4.0
        return logits


class FakeMaskAdapter:
    """Fake mask backend with one full-image mask."""

    description = "fake masks"

    def generate_masks(self, image, image_array):
        """Return one mask covering the full image."""

        del image
        height, width = image_array.shape[:2]
        return [{"segmentation": np.ones((height, width), dtype=bool), "source": "fake_sam"}]


class FakeFeatureAdapter:
    """Fake feature backend with constant purity features."""

    description = "fake dino features"

    def extract_features(self, image, image_array):
        """Return image-sized constant features."""

        del image
        height, width = image_array.shape[:2]
        features = np.zeros((height, width, 2), dtype=np.float32)
        features[..., 0] = 1.0
        return features


def test_threshold_too_low_matches_dense_only_prediction():
    """When no pixels are uncertain, dense-first refinement should be a no-op."""

    dense_logits = np.zeros((2, 2, 2), dtype=np.float32)
    dense_logits[..., 0] = 5.0
    dense_logits[..., 1] = 1.0
    masks = [{"segmentation": np.ones((2, 2), dtype=bool), "source": "fake_sam"}]
    features = np.zeros((2, 2, 2), dtype=np.float32)

    result = eval_dense_first_voc.refine_dense_prediction(
        dense_logits=dense_logits,
        masks=masks,
        dino_features=features,
        uncertainty_threshold=-10.0,
        entropy_threshold=None,
        max_refine_ratio=1.0,
        sam_min_area=1,
        sam_max_area_ratio=1.0,
    )

    np.testing.assert_array_equal(result.refined_prediction, result.base_prediction)
    assert result.refined_pixel_ratio == 0.0
    assert result.number_of_refined_regions == 0


def test_confident_pixels_are_not_overwritten_by_sam_mask():
    """SAM masks may cover confident pixels but may not modify them."""

    dense_logits = np.zeros((2, 2, 2), dtype=np.float32)
    dense_logits[..., 0] = 6.0
    dense_logits[..., 1] = 0.0
    dense_logits[0, 0, 0] = 1.0
    dense_logits[0, 0, 1] = 1.2
    masks = [{"segmentation": np.ones((2, 2), dtype=bool), "source": "fake_sam"}]
    features = np.zeros((2, 2, 2), dtype=np.float32)

    result = eval_dense_first_voc.refine_dense_prediction(
        dense_logits=dense_logits,
        masks=masks,
        dino_features=features,
        uncertainty_threshold=0.3,
        entropy_threshold=None,
        max_refine_ratio=1.0,
        sam_min_area=1,
        sam_max_area_ratio=1.0,
    )

    assert result.refined_prediction[1, 1] == result.base_prediction[1, 1]
    assert result.uncertain_pixels[1, 1] is np.False_


def test_uncertain_pixels_can_be_refined_by_sam_region_pooling():
    """Low-margin pixels can take the pooled SAM-region class."""

    dense_logits = np.zeros((2, 2, 2), dtype=np.float32)
    dense_logits[..., 0] = 3.0
    dense_logits[..., 1] = 1.0
    dense_logits[0, 0, 0] = 1.0
    dense_logits[0, 0, 1] = 1.1
    dense_logits[0, 1, 1] = 4.0
    masks = [{"segmentation": np.ones((2, 2), dtype=bool), "source": "fake_sam"}]
    features = np.zeros((2, 2, 2), dtype=np.float32)

    result = eval_dense_first_voc.refine_dense_prediction(
        dense_logits=dense_logits,
        masks=masks,
        dino_features=features,
        uncertainty_threshold=0.2,
        entropy_threshold=None,
        max_refine_ratio=1.0,
        sam_min_area=1,
        sam_max_area_ratio=1.0,
    )

    assert result.base_prediction[0, 0] == 1
    assert result.refined_prediction[0, 0] == 0
    assert result.refined_pixel_ratio == 0.25
    assert result.number_of_refined_regions == 1


def test_dense_first_voc20_ignore_background_runs_with_fake_backends(monkeypatch, tmp_path):
    """Dense-first VOC evaluator should run on a fake VOC dataset."""

    voc_root = _create_fake_voc_dataset(tmp_path)
    monkeypatch.setattr(eval_dense_first_voc, "build_dense_adapter", lambda *args, **kwargs: FakeDenseAdapter())
    monkeypatch.setattr(eval_dense_first_voc, "build_mask_adapter", lambda *args, **kwargs: FakeMaskAdapter())
    monkeypatch.setattr(eval_dense_first_voc, "build_feature_adapter", lambda *args, **kwargs: FakeFeatureAdapter())

    metrics = eval_dense_first_voc.evaluate_dataset(
        voc_root=voc_root,
        split="val",
        output_dir=tmp_path / "dense_first_eval",
        limit=1,
        voc_mode="voc20",
        voc20_ignore_background=True,
        uncertainty_threshold=-10.0,
        entropy_threshold=None,
    )

    assert metrics["dense_base_mIoU"] == 1.0
    assert metrics["refined_mIoU"] == 1.0
    assert metrics["mIoU"] == 1.0
    assert metrics["refined_pixel_ratio"] == 0.0
    assert metrics["number_of_refined_regions"] == 0
    assert Path(metrics["metrics_path"]).exists()
    saved = json.loads(Path(metrics["metrics_path"]).read_text(encoding="utf-8"))
    assert saved["fusion_mode"] == "dense_first"
