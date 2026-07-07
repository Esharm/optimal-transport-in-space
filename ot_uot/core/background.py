"""Background construction utilities for signed-residual models."""

from __future__ import annotations

from enum import Enum

import numpy as np


class BackgroundMode(str, Enum):
    """Background construction modes from a static reconstruction sequence."""

    MEAN = "mean"
    MEDIAN = "median"
    FIRST = "first"


def as_background_sequence(background: np.ndarray, frames: int) -> np.ndarray:
    """Return a background sequence with shape ``(frames,H,W)``."""

    background = np.asarray(background, dtype=np.float64)
    if background.ndim == 2:
        return np.repeat(background[None, :, :], int(frames), axis=0)
    if background.ndim == 3 and background.shape[0] == int(frames):
        return background
    raise ValueError("background must have shape (H,W) or (K,H,W)")


def make_background(static_sequence: np.ndarray, mode: str | BackgroundMode = BackgroundMode.MEAN) -> np.ndarray:
    """Construct a fixed background image from static reconstructions."""

    static_sequence = np.asarray(static_sequence, dtype=np.float64)
    if static_sequence.ndim != 3:
        raise ValueError("static_sequence must have shape (K,H,W)")
    mode = BackgroundMode(mode)
    if mode == BackgroundMode.MEAN:
        return np.mean(static_sequence, axis=0)
    if mode == BackgroundMode.MEDIAN:
        return np.median(static_sequence, axis=0)
    if mode == BackgroundMode.FIRST:
        return static_sequence[0].copy()
    raise ValueError(f"Unsupported background mode: {mode}")


def decompose_signed_residual(image: np.ndarray, background: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return nonnegative ``positive, negative`` channels for ``image-background``."""

    image = np.asarray(image, dtype=np.float64)
    background_sequence = as_background_sequence(background, image.shape[0])
    residual = image - background_sequence
    return np.maximum(residual, 0.0), np.maximum(-residual, 0.0)

