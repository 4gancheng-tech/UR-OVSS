"""Prompt utilities for the UR-OVSS MVP.

The functions in this module are intentionally small and model-agnostic. They
build deterministic positive/negative prompt strings and combine prompt scores
with base class scores for region-level rescoring.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np


POSITIVE_PROMPT_TEMPLATES = (
    "a photo of a {class}",
    "a clear photo of a {class}",
    "a close-up photo of a {class}",
    "a photo of the {class}",
    "a {class} in the scene",
)

NEGATIVE_PROMPT_TEMPLATES = (
    "not a {class}",
    "background without {class}",
)


def build_positive_prompts(class_names: Sequence[str]) -> Dict[str, list[str]]:
    """Build positive prompt variants for each open-vocabulary class.

    Args:
        class_names: Sequence of class names, length C.

    Returns:
        Mapping from class name to P positive prompt strings, where P is the
        number of positive templates.
    """

    return {
        class_name: [template.format(**{"class": class_name}) for template in POSITIVE_PROMPT_TEMPLATES]
        for class_name in class_names
    }


def build_negative_prompts(class_names: Sequence[str]) -> Dict[str, list[str]]:
    """Build negative prompt variants for each open-vocabulary class.

    Args:
        class_names: Sequence of class names, length C.

    Returns:
        Mapping from class name to N negative prompt strings, where N is the
        number of negative templates.
    """

    return {
        class_name: [template.format(**{"class": class_name}) for template in NEGATIVE_PROMPT_TEMPLATES]
        for class_name in class_names
    }


def _as_2d_prompt_scores(
    prompt_scores: Optional[Sequence[Sequence[float]]],
    num_classes: int,
    score_name: str,
) -> Optional[np.ndarray]:
    """Convert optional class-by-prompt scores to a validated float array.

    Args:
        prompt_scores: Optional score matrix with shape [C, P].
        num_classes: Expected number of classes C.
        score_name: Human-readable name for error messages.

    Returns:
        A float32 numpy array with shape [C, P], or None.
    """

    if prompt_scores is None:
        return None

    scores = np.asarray(prompt_scores, dtype=np.float32)
    if scores.ndim == 1:
        scores = scores[:, None]
    if scores.ndim != 2 or scores.shape[0] != num_classes:
        raise ValueError(
            f"{score_name} must have the same number of classes as base scores; "
            f"got shape {scores.shape} for {num_classes} classes."
        )
    return scores


def compute_prompt_rescore(
    base_scores: Sequence[float],
    positive_prompt_scores: Optional[Sequence[Sequence[float]]] = None,
    negative_prompt_scores: Optional[Sequence[Sequence[float]]] = None,
    alpha: float = 0.30,
) -> np.ndarray:
    """Combine base class scores with positive and negative prompt scores.

    Args:
        base_scores: Region-level class scores with shape [C].
        positive_prompt_scores: Optional positive prompt scores with shape
            [C, P], where P is the number of positive prompts per class.
        negative_prompt_scores: Optional negative prompt scores with shape
            [C, N], where N is the number of negative prompts per class.
        alpha: Negative prompt suppression weight. The UR-OVSS MVP default is
            0.30.

    Returns:
        Rescored class logits with shape [C], computed as
        base + mean(positive) - alpha * mean(negative).
    """

    base = np.asarray(base_scores, dtype=np.float32)
    if base.ndim != 1:
        raise ValueError(f"base_scores must have shape [C], got {base.shape}.")

    positive = _as_2d_prompt_scores(positive_prompt_scores, base.shape[0], "positive_prompt_scores")
    negative = _as_2d_prompt_scores(negative_prompt_scores, base.shape[0], "negative_prompt_scores")

    rescored = base.copy()
    if positive is not None:
        rescored = rescored + positive.mean(axis=1)
    if negative is not None:
        rescored = rescored - float(alpha) * negative.mean(axis=1)
    return rescored.astype(np.float32)
