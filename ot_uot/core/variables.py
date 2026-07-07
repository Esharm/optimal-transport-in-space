"""Typed containers for exact signed-residual UOT variables."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ImageResidualState:
    """Image and residual variables satisfying ``u = background + pos - neg``."""

    image: np.ndarray
    positive: np.ndarray
    negative: np.ndarray
    background: np.ndarray

    def __post_init__(self) -> None:
        self.image = np.asarray(self.image, dtype=np.float64)
        self.positive = np.asarray(self.positive, dtype=np.float64)
        self.negative = np.asarray(self.negative, dtype=np.float64)
        self.background = np.asarray(self.background, dtype=np.float64)
        if self.image.shape != self.positive.shape or self.image.shape != self.negative.shape:
            raise ValueError("image, positive, and negative must have matching shapes")
        if self.background.ndim == 2:
            expected = self.image.shape[1:]
            if self.background.shape != expected:
                raise ValueError("2D background must match frame shape")
        elif self.background.shape != self.image.shape:
            raise ValueError("background must be either one image or a full sequence")
        if np.any(self.image < -1e-12):
            raise ValueError("image contains negative values")
        if np.any(self.positive < -1e-12) or np.any(self.negative < -1e-12):
            raise ValueError("residual channels must be nonnegative")

    @property
    def background_sequence(self) -> np.ndarray:
        if self.background.ndim == 2:
            return np.repeat(self.background[None, :, :], self.image.shape[0], axis=0)
        return self.background

    def decomposition_residual(self) -> np.ndarray:
        return self.image - self.background_sequence - self.positive + self.negative

    def max_decomposition_error(self) -> float:
        return float(np.max(np.abs(self.decomposition_residual())))


@dataclass
class TransportState:
    """Transport path variables for one adjacent pair and one residual channel."""

    density: np.ndarray
    momentum: np.ndarray
    source: np.ndarray

    def __post_init__(self) -> None:
        self.density = np.asarray(self.density, dtype=np.float64)
        self.momentum = np.asarray(self.momentum, dtype=np.float64)
        self.source = np.asarray(self.source, dtype=np.float64)
        if self.density.ndim != 3:
            raise ValueError("density must have shape (T,H,W)")
        if self.momentum.shape != (self.density.shape[0] - 1, 2, *self.density.shape[1:]):
            raise ValueError("momentum must have shape (T-1,2,H,W)")
        if self.source.shape != (self.density.shape[0] - 1, *self.density.shape[1:]):
            raise ValueError("source must have shape (T-1,H,W)")
        if np.any(self.density < -1e-12):
            raise ValueError("density must be nonnegative")

