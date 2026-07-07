"""Convergence bookkeeping for exact signed-residual ADMM."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ot_uot.core.config import UOTParameters
from ot_uot.core.variables import ImageResidualState
from ot_uot.optimization.objective import ConstraintResiduals


@dataclass
class ConvergenceStatus:
    """ADMM stopping decision and tolerances."""

    converged: bool
    eps_primal: float
    eps_dual: float
    state_relative_change: float


def state_vector_norm(state: ImageResidualState) -> float:
    """Euclidean norm of the coupled image/residual primal variables."""

    return float(
        np.sqrt(
            np.sum(state.image * state.image)
            + np.sum(state.positive * state.positive)
            + np.sum(state.negative * state.negative)
        )
    )


def state_difference_norm(new: ImageResidualState, old: ImageResidualState) -> float:
    """Euclidean distance between two coupled image/residual states."""

    return float(
        np.sqrt(
            np.sum((new.image - old.image) ** 2)
            + np.sum((new.positive - old.positive) ** 2)
            + np.sum((new.negative - old.negative) ** 2)
        )
    )


def convergence_status(
    state: ImageResidualState,
    previous_state: ImageResidualState,
    residuals: ConstraintResiduals,
    params: UOTParameters,
    iteration: int,
) -> ConvergenceStatus:
    """Return ADMM convergence status using primal and state-change criteria."""

    diff = state_difference_norm(state, previous_state)
    scale = state_vector_norm(state) + 1e-30
    relative_change = float(diff / scale)
    dof = float(state.image.size + state.positive.size + state.negative.size)
    eps_primal = float(np.sqrt(dof) * params.abs_tol + params.rel_tol * scale)
    eps_dual = float(np.sqrt(dof) * params.abs_tol + params.rel_tol * scale)
    converged = bool(
        iteration >= params.min_admm_iters
        and residuals.primal_l2 <= eps_primal
        and diff <= eps_dual
    )
    return ConvergenceStatus(
        converged=converged,
        eps_primal=eps_primal,
        eps_dual=eps_dual,
        state_relative_change=relative_change,
    )

