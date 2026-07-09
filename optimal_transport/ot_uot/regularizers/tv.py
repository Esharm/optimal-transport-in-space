"""Isotropic total variation utilities."""

from __future__ import annotations

import numpy as np

from ot_uot.core.finite_differences import gradient


def tv_value(image: np.ndarray) -> float:
    """Return isotropic finite-difference TV."""

    g = gradient(np.asarray(image, dtype=np.float64))
    return float(np.sum(np.sqrt(np.sum(g * g, axis=0))))


def sequence_tv_value(sequence: np.ndarray) -> float:
    """Return summed isotropic TV over a video sequence."""

    sequence = np.asarray(sequence, dtype=np.float64)
    return float(sum(tv_value(frame) for frame in sequence))


def project_tv_dual(dual: np.ndarray, radius: float) -> np.ndarray:
    """Project TV dual variable pointwise onto ``||q(x)||_2 <= radius``."""

    dual = np.asarray(dual, dtype=np.float64)
    norm = np.sqrt(np.sum(dual * dual, axis=0))
    return dual / np.maximum(1.0, norm / max(float(radius), 1e-30))[None, :, :]

