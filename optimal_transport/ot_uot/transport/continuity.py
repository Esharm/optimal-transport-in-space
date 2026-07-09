"""Continuity operators for Eulerian UOT paths."""

from __future__ import annotations

import numpy as np

from ot_uot.core.finite_differences import divergence, gradient


def uot_continuity(density: np.ndarray, momentum: np.ndarray, source: np.ndarray, dt: float) -> np.ndarray:
    """Compute ``(rho[t+1]-rho[t])/dt + div(m[t]) - source[t]``."""

    density = np.asarray(density, dtype=np.float64)
    momentum = np.asarray(momentum, dtype=np.float64)
    source = np.asarray(source, dtype=np.float64)
    if momentum.shape != (density.shape[0] - 1, 2, *density.shape[1:]):
        raise ValueError("momentum shape is incompatible with density")
    if source.shape != (density.shape[0] - 1, *density.shape[1:]):
        raise ValueError("source shape is incompatible with density")
    return (density[1:] - density[:-1]) / float(dt) + np.stack(
        [divergence(momentum[t]) for t in range(momentum.shape[0])],
        axis=0,
    ) - source


def uot_continuity_adjoint(phi: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Euclidean adjoint of :func:`uot_continuity`."""

    phi = np.asarray(phi, dtype=np.float64)
    intervals, height, width = phi.shape
    density_adj = np.zeros((intervals + 1, height, width), dtype=np.float64)
    density_adj[0] = -phi[0] / float(dt)
    density_adj[-1] = phi[-1] / float(dt)
    if intervals > 1:
        density_adj[1:-1] = (phi[:-1] - phi[1:]) / float(dt)

    momentum_adj = np.stack([-gradient(phi[t]) for t in range(intervals)], axis=0)
    source_adj = -phi
    return density_adj, momentum_adj, source_adj


def check_continuity_adjoint(shape: tuple[int, int], nodes: int = 5, seed: int = 0) -> float:
    """Return an adjoint-test scalar error for the UOT continuity operator."""

    rng = np.random.default_rng(seed)
    density = rng.normal(size=(nodes, *shape))
    momentum = rng.normal(size=(nodes - 1, 2, *shape))
    source = rng.normal(size=(nodes - 1, *shape))
    phi = rng.normal(size=(nodes - 1, *shape))
    dt = 1.0 / (nodes - 1)
    left = float(np.sum(uot_continuity(density, momentum, source, dt) * phi))
    da, ma, sa = uot_continuity_adjoint(phi, dt)
    right = float(np.sum(density * da) + np.sum(momentum * ma) + np.sum(source * sa))
    return left - right

