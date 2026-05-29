import numpy as np
import pytest

from prompts import (
    NEGATIVE_PROMPT_TEMPLATES,
    POSITIVE_PROMPT_TEMPLATES,
    build_negative_prompts,
    build_positive_prompts,
    compute_prompt_rescore,
)


def test_build_positive_prompts_uses_all_templates_per_class():
    """Positive prompt builder should expand every template for every class."""

    prompts = build_positive_prompts(["cat", "fire truck"])

    assert prompts["cat"] == [template.format(**{"class": "cat"}) for template in POSITIVE_PROMPT_TEMPLATES]
    assert prompts["fire truck"] == [
        template.format(**{"class": "fire truck"}) for template in POSITIVE_PROMPT_TEMPLATES
    ]


def test_build_negative_prompts_uses_all_templates_per_class():
    """Negative prompt builder should expand every template for every class."""

    prompts = build_negative_prompts(["cat"])

    assert prompts["cat"] == [template.format(**{"class": "cat"}) for template in NEGATIVE_PROMPT_TEMPLATES]


def test_compute_prompt_rescore_adds_positive_and_suppresses_negative_scores():
    """Prompt rescoring should reward positives and suppress negatives."""

    base_scores = np.array([0.40, 0.40], dtype=np.float32)
    positive_scores = np.array(
        [
            [0.20, 0.40],
            [0.10, 0.10],
        ],
        dtype=np.float32,
    )
    negative_scores = np.array(
        [
            [1.00, 1.00],
            [0.00, 0.00],
        ],
        dtype=np.float32,
    )

    rescored = compute_prompt_rescore(base_scores, positive_scores, negative_scores, alpha=0.30)

    np.testing.assert_allclose(rescored, np.array([0.40, 0.50], dtype=np.float32), rtol=1e-6)


def test_compute_prompt_rescore_rejects_class_mismatches():
    """Prompt rescoring should reject prompt matrices with mismatched classes."""

    with pytest.raises(ValueError, match="same number of classes"):
        compute_prompt_rescore(
            np.array([0.1, 0.2], dtype=np.float32),
            np.array([[0.1], [0.2], [0.3]], dtype=np.float32),
            None,
        )
