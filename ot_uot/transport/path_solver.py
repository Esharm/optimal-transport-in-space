"""Convex primal-dual solver for one UOT density path."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ot_uot.core.variables import TransportState
from ot_uot.transport.continuity import uot_continuity, uot_continuity_adjoint
from ot_uot.transport.kinetic_prox import kinetic_energy, prox_kinetic_perspective


@dataclass
class UOTPathInfo:
    iterations: int
    relative_change: float
    continuity_residual: float
    action: float
    source_l2: float
    source_mass_abs: float
    source_mass_signed: float


def initialize_path(target: np.ndarray, nodes: int) -> TransportState:
    """Initialize a path from endpoint/frame target data."""

    target = np.asarray(target, dtype=np.float64)
    if target.ndim == 2:
        density = np.repeat(np.maximum(target, 0.0)[None, :, :], int(nodes), axis=0)
    elif target.ndim == 3:
        if target.shape[0] != int(nodes):
            raise ValueError("3D target must have one image per transport node")
        density = np.maximum(target, 0.0)
    else:
        raise ValueError("target must have shape (H,W) or (T,H,W)")
    momentum = np.zeros((int(nodes) - 1, 2, *density.shape[1:]), dtype=np.float64)
    source = np.zeros((int(nodes) - 1, *density.shape[1:]), dtype=np.float64)
    return TransportState(density=density, momentum=momentum, source=source)


def uot_action(
    state: TransportState,
    transport_weight: float,
    source_weight: float,
    dt: float,
) -> float:
    """Weighted UOT action for a path state."""

    kinetic = kinetic_energy(state.density, state.momentum)
    source_cost = 0.5 * float(source_weight) * np.sum(state.source * state.source)
    return float(float(transport_weight) * float(dt) * (kinetic + source_cost))


def solve_uot_path(
    target: np.ndarray,
    target_weight: np.ndarray,
    transport_weight: float,
    source_weight: float,
    max_iter: int,
    tol: float,
    state: TransportState | None = None,
    check_every: int = 10,
) -> tuple[TransportState, UOTPathInfo]:
    """Solve one convex UOT path with quadratic density targets.

    Parameters
    ----------
    target:
        Array with shape ``(T,H,W)``.
    target_weight:
        Nonnegative weights with shape ``(T,H,W)`` or ``(T,1,1)``.
    """

    target = np.asarray(target, dtype=np.float64)
    target_weight = np.asarray(target_weight, dtype=np.float64)
    if target.ndim != 3:
        raise ValueError("target must have shape (T,H,W)")
    if target_weight.shape != target.shape:
        target_weight = np.broadcast_to(target_weight, target.shape)
    if np.any(target_weight < 0.0):
        raise ValueError("target weights must be nonnegative")
    nodes = target.shape[0]
    if nodes < 2:
        raise ValueError("a UOT path needs at least two density nodes")

    if state is None or state.density.shape != target.shape:
        state = initialize_path(np.maximum(target, 0.0), nodes)

    density = np.maximum(state.density.copy(), 0.0)
    momentum = state.momentum.copy()
    source = state.source.copy()
    phi = np.zeros((nodes - 1, *target.shape[1:]), dtype=np.float64)
    dt = 1.0 / (nodes - 1)

    operator_bound_sq = 4.0 / (dt * dt) + 9.0
    max_target_weight = float(np.max(target_weight)) if target_weight.size else 0.0
    step = 0.99 / np.sqrt(operator_bound_sq)
    tau = min(step, 1.9 / max(max_target_weight, 1e-12))
    sigma = 0.99 / (tau * operator_bound_sq)

    density_bar = density.copy()
    momentum_bar = momentum.copy()
    source_bar = source.copy()
    last_change = np.inf
    continuity_norm = np.inf
    iterations = int(max_iter)

    for iteration in range(1, int(max_iter) + 1):
        phi += sigma * uot_continuity(density_bar, momentum_bar, source_bar, dt)
        density_adj, momentum_adj, source_adj = uot_continuity_adjoint(phi, dt)

        old_density = density.copy()
        old_momentum = momentum.copy()
        old_source = source.copy()

        density_trial = density - tau * density_adj
        momentum_trial = momentum - tau * momentum_adj
        source_trial = source - tau * source_adj

        density_trial -= tau * target_weight * (density - target)

        kinetic_gamma = tau * float(transport_weight) * dt
        for t in range(nodes - 1):
            density[t], momentum[t] = prox_kinetic_perspective(
                density_trial[t], momentum_trial[t], kinetic_gamma
            )
        density[-1] = np.maximum(density_trial[-1], 0.0)

        source_gamma = tau * float(transport_weight) * dt * float(source_weight)
        source = source_trial / (1.0 + source_gamma)

        density_bar = 2.0 * density - old_density
        momentum_bar = 2.0 * momentum - old_momentum
        source_bar = 2.0 * source - old_source

        if iteration % int(check_every) == 0 or iteration == int(max_iter):
            delta = (
                np.sum((density - old_density) ** 2)
                + np.sum((momentum - old_momentum) ** 2)
                + np.sum((source - old_source) ** 2)
            )
            scale = (
                np.sum(old_density ** 2)
                + np.sum(old_momentum ** 2)
                + np.sum(old_source ** 2)
            )
            last_change = float(np.sqrt(delta / (scale + 1e-30)))
            residual = uot_continuity(density, momentum, source, dt)
            continuity_norm = float(
                np.linalg.norm(residual)
                / (np.linalg.norm(density) + np.linalg.norm(momentum) + np.linalg.norm(source) + 1e-30)
            )
            if max(last_change, continuity_norm) < float(tol):
                iterations = iteration
                break

    final_state = TransportState(density=density, momentum=momentum, source=source)
    info = UOTPathInfo(
        iterations=iterations,
        relative_change=float(last_change),
        continuity_residual=float(continuity_norm),
        action=uot_action(final_state, transport_weight, source_weight, dt),
        source_l2=float(np.linalg.norm(source)),
        source_mass_abs=float(np.sum(np.abs(source)) * dt),
        source_mass_signed=float(np.sum(source) * dt),
    )
    return final_state, info

