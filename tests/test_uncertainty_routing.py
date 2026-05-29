import numpy as np

from uncertainty_routing import (
    compute_dino_variance,
    compute_prompt_uncertainty,
    compute_semantic_margin,
    fuse_region_predictions,
    get_uncertain_regions_by_quantile,
    route_region,
)


def test_compute_semantic_margin_supports_single_region_and_batches():
    """Semantic margin should work for one region and batched regions."""

    single_margin = compute_semantic_margin(np.array([0.70, 0.20, 0.10], dtype=np.float32))
    batch_margins = compute_semantic_margin(
        np.array(
            [
                [0.70, 0.20, 0.10],
                [0.30, 0.25, 0.05],
            ],
            dtype=np.float32,
        )
    )

    np.testing.assert_allclose(single_margin, 0.50, rtol=1e-6)
    np.testing.assert_allclose(batch_margins, np.array([0.50, 0.05], dtype=np.float32), rtol=1e-6)


def test_compute_dino_variance_uses_masked_feature_vectors():
    """DINO variance should be computed from only masked feature vectors."""

    # Shape: H=2, W=2, C=2.
    dino_features = np.array(
        [
            [[1.0, 0.0], [1.0, 0.0]],
            [[3.0, 0.0], [3.0, 0.0]],
        ],
        dtype=np.float32,
    )
    mask = np.array(
        [
            [True, False],
            [True, False],
        ]
    )

    assert compute_dino_variance(dino_features, mask) == 0.5


def test_compute_prompt_uncertainty_returns_per_class_variance():
    """Prompt uncertainty should return one variance value per class."""

    prompt_scores = np.array(
        [
            [0.2, 0.4, 0.6],
            [0.5, 0.5, 0.5],
        ],
        dtype=np.float32,
    )

    uncertainty = compute_prompt_uncertainty(prompt_scores)

    np.testing.assert_allclose(uncertainty, np.array([0.02666667, 0.0], dtype=np.float32), rtol=1e-6)


def test_get_uncertain_regions_by_quantile_marks_lowest_or_highest_exact_fraction():
    """Quantile routing should mark the requested low or high fraction."""

    values = np.array([0.10, 0.30, 0.20, 0.90], dtype=np.float32)

    low_uncertain = get_uncertain_regions_by_quantile(values, ratio=0.50, mode="low")
    high_uncertain = get_uncertain_regions_by_quantile(values, ratio=0.50, mode="high")

    np.testing.assert_array_equal(low_uncertain, np.array([True, False, True, False]))
    np.testing.assert_array_equal(high_uncertain, np.array([False, True, False, True]))


def test_route_region_uses_prompt_rescore_for_semantic_uncertainty():
    """Semantic-uncertain regions should use prompt rescoring when available."""

    routed = route_region(
        region_id=3,
        scores=np.array([0.60, 0.55], dtype=np.float32),
        class_names=["cat", "dog"],
        semantic_uncertain=True,
        spatial_uncertain=False,
        dino_variance=0.1,
        prompt_rescore_scores=np.array([0.20, 0.80], dtype=np.float32),
    )

    assert routed["route_type"] == "prompt_rescore"
    assert routed["predicted_label"] == "dog"
    assert routed["label_id"] == 1
    assert routed["semantic_uncertain"] is True
    assert routed["spatial_uncertain"] is False


def test_route_region_downweights_confidence_for_spatial_uncertainty():
    """Spatial-uncertain regions should preserve labels and lower confidence."""

    routed = route_region(
        region_id=1,
        scores=np.array([0.90, 0.10], dtype=np.float32),
        class_names=["road", "sky"],
        semantic_uncertain=False,
        spatial_uncertain=True,
        dino_variance=0.7,
    )

    assert routed["route_type"] == "spatial_downweight"
    assert routed["predicted_label"] == "road"
    assert 0.0 < routed["confidence"] < 0.90
    assert "SAM prompt refinement" in routed["notes"][0]


def test_fuse_region_predictions_keeps_highest_confidence_in_overlaps():
    """Pixel fusion should keep the label from the highest-confidence region."""

    low_conf_mask = np.array([[True, True], [False, False]])
    high_conf_mask = np.array([[False, True], [True, False]])
    regions = [
        {"mask": low_conf_mask, "label_id": 0, "confidence": 0.30},
        {"mask": high_conf_mask, "label_id": 1, "confidence": 0.80},
    ]

    segmentation, confidence = fuse_region_predictions(regions, output_shape=(2, 2))

    np.testing.assert_array_equal(segmentation, np.array([[0, 1], [1, -1]], dtype=np.int32))
    np.testing.assert_allclose(confidence, np.array([[0.30, 0.80], [0.80, -np.inf]], dtype=np.float32))
