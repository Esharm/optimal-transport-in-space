"""Projection and proximal helpers for constrained image variables."""

from __future__ import annotations

import numpy as np


def project_nonnegative(x: np.ndarray) -> np.ndarray:
    """Project an array onto the nonnegative orthant."""

    return np.maximum(np.asarray(x, dtype=np.float64), 0.0)


def project_nonnegative_simplex(x: np.ndarray, total_mass: float) -> np.ndarray:
    """Euclidean projection onto ``x >= 0`` and ``sum(x) = total_mass``."""

    x = np.asarray(x, dtype=np.float64)
    total_mass = float(total_mass)
    if total_mass <= 0.0:
        return np.zeros_like(x)

    flat = x.ravel()
    ordered = np.sort(flat)[::-1]
    cumulative = np.cumsum(ordered) - total_mass
    indices = np.arange(1, flat.size + 1, dtype=np.float64)
    active = ordered - cumulative / indices > 0.0
    if not np.any(active):
        return np.full_like(x, total_mass / flat.size)
    rho = np.flatnonzero(active)[-1]
    threshold = cumulative[rho] / (rho + 1.0)
    return np.maximum(x - threshold, 0.0)


def prox_l1_nonnegative(x: np.ndarray, weight: float) -> np.ndarray:
    """Proximal map of ``weight * sum(x) + indicator{x >= 0}``."""

    return np.maximum(np.asarray(x, dtype=np.float64) - float(weight), 0.0)

