"""Finite-difference operators with explicit adjoint conventions."""

from __future__ import annotations

import numpy as np


Array = np.ndarray


def gradient(image: Array) -> Array:
    """Forward-difference gradient.

    Returns an array with shape ``(2, height, width)``. The first component is
    the horizontal derivative and the second component is the vertical
    derivative. Boundary forward differences are zero.
    """

    image = np.asarray(image, dtype=np.float64)
    gx = np.zeros_like(image)
    gy = np.zeros_like(image)
    gx[:, :-1] = image[:, 1:] - image[:, :-1]
    gy[:-1, :] = image[1:, :] - image[:-1, :]
    return np.stack((gx, gy), axis=0)


def divergence(field: Array) -> Array:
    """Negative adjoint of :func:`gradient` for the boundary convention above."""

    field = np.asarray(field, dtype=np.float64)
    if field.shape[0] != 2:
        raise ValueError(f"Expected field shape (2,H,W), got {field.shape}")

    px, py = field
    dx = np.zeros_like(px)
    dy = np.zeros_like(py)

    dx[:, 0] = px[:, 0]
    if px.shape[1] > 2:
        dx[:, 1:-1] = px[:, 1:-1] - px[:, :-2]
    if px.shape[1] > 1:
        dx[:, -1] = -px[:, -2]

    dy[0, :] = py[0, :]
    if py.shape[0] > 2:
        dy[1:-1, :] = py[1:-1, :] - py[:-2, :]
    if py.shape[0] > 1:
        dy[-1, :] = -py[-2, :]

    return dx + dy


def check_adjoint(image_shape: tuple[int, int], seed: int = 0) -> float:
    """Return ``<grad u,p> + <u,div p>`` for a random adjoint check."""

    rng = np.random.default_rng(seed)
    image = rng.normal(size=image_shape)
    field = rng.normal(size=(2, *image_shape))
    return float(np.sum(gradient(image) * field) + np.sum(image * divergence(field)))

