"""Proximal maps for dynamic optimal transport action terms."""

from __future__ import annotations

import numpy as np


def prox_kinetic_perspective(
    density_trial: np.ndarray,
    momentum_trial: np.ndarray,
    gamma: float,
    newton_iters: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """Prox of ``gamma * |m|^2/(2 rho) + indicator{rho >= 0}``.

    The positive-density solution is obtained from the scalar monotone equation

    ``rho - rho0 - gamma |m0|^2 / (2 (rho + gamma)^2) = 0``.
    """

    density_trial = np.asarray(density_trial, dtype=np.float64)
    momentum_trial = np.asarray(momentum_trial, dtype=np.float64)
    gamma = float(gamma)
    if momentum_trial.shape != (2, *density_trial.shape):
        raise ValueError("momentum_trial must have shape (2,H,W)")
    if gamma <= 0.0:
        return np.maximum(density_trial, 0.0), momentum_trial.copy()

    momentum_sq = np.sum(momentum_trial * momentum_trial, axis=0)
    active = density_trial + momentum_sq / (2.0 * gamma) > 0.0
    density = np.zeros_like(density_trial)
    for _ in range(max(5, int(newton_iters))):
        denom = density + gamma
        value = density - density_trial - gamma * momentum_sq / (2.0 * denom * denom)
        derivative = 1.0 + gamma * momentum_sq / (denom ** 3)
        candidate = density - value / derivative
        density = np.where(active, np.maximum(candidate, 0.0), 0.0)

    momentum = momentum_trial * (density / (density + gamma + 1e-30))[None, :, :]
    momentum[:, ~active] = 0.0
    return density, momentum


def kinetic_energy(density: np.ndarray, momentum: np.ndarray, density_floor: float = 1e-12) -> float:
    """Return ``sum |m|^2/(2 rho)`` over transport intervals."""

    density = np.asarray(density, dtype=np.float64)
    momentum = np.asarray(momentum, dtype=np.float64)
    rho = np.maximum(density[:-1], float(density_floor))
    return float(np.sum(np.sum(momentum * momentum, axis=1) / (2.0 * rho)))

