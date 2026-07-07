"""Complex visibility data terms for controlled synthetic EHT observations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DirectVisibilityOperator:
    """Scaled direct nonuniform Fourier operator for real-valued images."""

    u: np.ndarray
    v: np.ndarray
    weight: np.ndarray
    shape: tuple[int, int]
    fov_rad: float
    data_scale: float = 1.0
    use_cache: bool = False
    chunk_size: int = 128

    def __post_init__(self) -> None:
        self.u = np.asarray(self.u, dtype=np.float64).ravel()
        self.v = np.asarray(self.v, dtype=np.float64).ravel()
        self.weight = np.asarray(self.weight, dtype=np.float64).ravel()
        if not (self.u.size == self.v.size == self.weight.size):
            raise ValueError("u, v, and weight must have the same length")
        if np.any(self.weight < 0.0):
            raise ValueError("visibility weights must be nonnegative")

        self.height, self.width = self.shape
        self.data_scale = float(max(self.data_scale, 1e-30))
        l = (np.arange(self.width) - self.width / 2.0) * self.fov_rad / self.width
        m = (np.arange(self.height) - self.height / 2.0) * self.fov_rad / self.height
        grid_l, grid_m = np.meshgrid(l, m)
        self._l_flat = grid_l.ravel()
        self._m_flat = grid_m.ravel()
        self._scale = (
            np.sqrt(np.sum(self.weight) + 1e-30)
            * np.sqrt(self.height * self.width)
            * self.data_scale
        )
        self._matrix = None
        if self.use_cache and self.u.size:
            phase = -2j * np.pi * (
                self.u[:, None] * self._l_flat + self.v[:, None] * self._m_flat
            )
            self._matrix = np.exp(phase).astype(np.complex128)

    @property
    def scale(self) -> float:
        """Scalar used to normalize the weighted Fourier operator."""

        return float(self._scale)

    def forward(self, image: np.ndarray) -> np.ndarray:
        flat = np.asarray(image, dtype=np.float64).reshape(-1)
        if flat.size != self.height * self.width:
            raise ValueError(f"Expected image shape {self.shape}")
        if self.u.size == 0:
            return np.zeros(0, dtype=np.complex128)
        if self._matrix is not None:
            values = self._matrix @ flat
        else:
            values = np.zeros(self.u.size, dtype=np.complex128)
            for start in range(0, self.u.size, self.chunk_size):
                section = slice(start, min(start + self.chunk_size, self.u.size))
                phase = -2j * np.pi * (
                    self.u[section, None] * self._l_flat
                    + self.v[section, None] * self._m_flat
                )
                values[section] = np.exp(phase) @ flat
        return np.sqrt(self.weight) * values / self._scale

    def adjoint(self, visibility: np.ndarray) -> np.ndarray:
        visibility = np.asarray(visibility, dtype=np.complex128).ravel()
        if visibility.size != self.u.size:
            raise ValueError("visibility vector length does not match operator")
        if self.u.size == 0:
            return np.zeros(self.shape, dtype=np.float64)
        weighted = np.sqrt(self.weight) * visibility / self._scale
        if self._matrix is not None:
            values = self._matrix.conj().T @ weighted
        else:
            values = np.zeros(self.height * self.width, dtype=np.complex128)
            for start in range(0, self.u.size, self.chunk_size):
                section = slice(start, min(start + self.chunk_size, self.u.size))
                phase = 2j * np.pi * (
                    self.u[section, None] * self._l_flat
                    + self.v[section, None] * self._m_flat
                )
                values += np.exp(phase).T @ weighted[section]
        return np.real(values.reshape(self.shape))


@dataclass
class ComplexVisibilityDataTerm:
    """Quadratic complex-visibility fidelity ``0.5 ||A u - y||_2^2``."""

    operator: DirectVisibilityOperator
    observed: np.ndarray

    def __post_init__(self) -> None:
        self.observed = np.asarray(self.observed, dtype=np.complex128).ravel()

    def residual(self, image: np.ndarray) -> np.ndarray:
        return self.operator.forward(image) - self.observed

    def value(self, image: np.ndarray) -> float:
        residual = self.residual(image)
        return float(0.5 * np.sum(np.abs(residual) ** 2))

    def gradient(self, image: np.ndarray) -> np.ndarray:
        return self.operator.adjoint(self.residual(image))

    def check_adjoint(self, seed: int = 0) -> float:
        rng = np.random.default_rng(seed)
        image = rng.normal(size=self.operator.shape)
        vis = rng.normal(size=self.observed.shape) + 1j * rng.normal(size=self.observed.shape)
        left = np.real(np.vdot(self.operator.forward(image), vis))
        right = np.sum(image * self.operator.adjoint(vis))
        return float(left - right)
