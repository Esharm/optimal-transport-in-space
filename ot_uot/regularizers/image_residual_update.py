"""Exact coupled image/residual ADMM subproblem solver."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ot_uot.core.background import as_background_sequence
from ot_uot.core.finite_differences import divergence, gradient
from ot_uot.core.projections import prox_l1_nonnegative, project_nonnegative
from ot_uot.core.variables import ImageResidualState
from ot_uot.regularizers.tv import project_tv_dual


@dataclass
class ChannelTargets:
    """Quadratic target data for one residual channel."""

    target_sum: np.ndarray
    weight_sum: np.ndarray


@dataclass
class ImageResidualUpdateInfo:
    iterations: int
    relative_change: float
    decomposition_residual: float
    smooth_step: float


class ImageResidualUpdater:
    """Numerical solver for the convex ADMM subproblem in ``(u,p,n)``.

    The update minimizes the exact augmented objective described in the
    mathematical specification. The subproblem is solved by a primal-dual
    forward-backward scheme: TV is handled through its dual projection, while
    nonnegativity and the residual mass penalty are handled by proximal maps.
    """

    def __init__(
        self,
        tv_weight: float,
        background_weight: float,
        residual_mass_weight: float,
        decomposition_penalty: float,
        iterations: int,
        tau_cap: float = 10.0,
        dual_sigma: float = 0.25,
        data_power_iters: int = 8,
    ):
        self.tv_weight = float(tv_weight)
        self.background_weight = float(background_weight)
        self.residual_mass_weight = float(residual_mass_weight)
        self.decomposition_penalty = float(decomposition_penalty)
        self.iterations = int(iterations)
        self.tau_cap = float(tau_cap)
        self.dual_sigma = float(dual_sigma)
        self.data_power_iters = int(data_power_iters)
        self._lipschitz_cache: dict[int, float] = {}

    def _data_lipschitz(self, data_term, shape: tuple[int, int]) -> float:
        key = id(data_term)
        if key in self._lipschitz_cache:
            return self._lipschitz_cache[key]
        rng = np.random.default_rng(1729)
        x = rng.normal(size=shape)
        x /= np.linalg.norm(x) + 1e-30
        eigenvalue = 0.0
        for _ in range(self.data_power_iters):
            applied = data_term.operator.adjoint(data_term.operator.forward(x))
            norm = np.linalg.norm(applied)
            if norm <= 1e-30:
                eigenvalue = 0.0
                break
            x = applied / norm
            eigenvalue = float(np.sum(x * applied))
        estimate = max(1.1 * eigenvalue, 1e-12)
        self._lipschitz_cache[key] = estimate
        return estimate

    def _step_size(self, data_terms, image_shape, pos_targets: ChannelTargets, neg_targets: ChannelTargets) -> float:
        data_lip = max(self._data_lipschitz(term, image_shape) for term in data_terms)
        endpoint_lip = float(
            max(np.max(pos_targets.weight_sum), np.max(neg_targets.weight_sum), 0.0)
        )
        # Conservative coupled-variable smooth bound for (u,p,n).
        smooth_lip = data_lip + self.background_weight + 6.0 * self.decomposition_penalty + endpoint_lip
        if self.tv_weight > 0.0:
            safe = 0.99 / (0.5 * smooth_lip + 8.0 * self.dual_sigma)
        else:
            safe = 0.99 / smooth_lip
        return float(min(self.tau_cap, safe))

    def update(
        self,
        state: ImageResidualState,
        data_terms,
        decomposition_dual: np.ndarray,
        positive_targets: ChannelTargets,
        negative_targets: ChannelTargets,
    ) -> tuple[ImageResidualState, ImageResidualUpdateInfo]:
        """Solve the image/residual ADMM subproblem."""

        u = state.image.copy()
        p = state.positive.copy()
        n = state.negative.copy()
        background = as_background_sequence(state.background, u.shape[0])
        decomposition_dual = np.asarray(decomposition_dual, dtype=np.float64)
        if decomposition_dual.shape != u.shape:
            raise ValueError("decomposition_dual shape must match image sequence")

        pos_target_sum = np.asarray(positive_targets.target_sum, dtype=np.float64)
        pos_weight_sum = np.asarray(positive_targets.weight_sum, dtype=np.float64)
        neg_target_sum = np.asarray(negative_targets.target_sum, dtype=np.float64)
        neg_weight_sum = np.asarray(negative_targets.weight_sum, dtype=np.float64)
        for arr in (pos_target_sum, pos_weight_sum, neg_target_sum, neg_weight_sum):
            if arr.shape != u.shape:
                raise ValueError("channel target arrays must match image sequence shape")

        tau = self._step_size(data_terms, u.shape[1:], positive_targets, negative_targets)
        q = np.zeros((u.shape[0], 2, *u.shape[1:]), dtype=np.float64)
        u_bar = u.copy()
        old_all = np.concatenate([u.ravel(), p.ravel(), n.ravel()])

        for _ in range(self.iterations):
            if self.tv_weight > 0.0:
                for k in range(u.shape[0]):
                    q[k] = project_tv_dual(q[k] + self.dual_sigma * gradient(u_bar[k]), self.tv_weight)

            old_u = u.copy()
            constraint = u - background - p + n + decomposition_dual

            grad_u = np.empty_like(u)
            for k, term in enumerate(data_terms):
                grad_u[k] = term.gradient(u[k])
            grad_u += self.background_weight * (u - background)
            grad_u += self.decomposition_penalty * constraint
            if self.tv_weight > 0.0:
                for k in range(u.shape[0]):
                    grad_u[k] -= divergence(q[k])

            grad_p = -self.decomposition_penalty * constraint
            grad_p += pos_weight_sum * p - pos_target_sum

            grad_n = self.decomposition_penalty * constraint
            grad_n += neg_weight_sum * n - neg_target_sum

            u = project_nonnegative(u - tau * grad_u)
            p = prox_l1_nonnegative(p - tau * grad_p, tau * self.residual_mass_weight)
            n = prox_l1_nonnegative(n - tau * grad_n, tau * self.residual_mass_weight)
            u_bar = 2.0 * u - old_u

        new_state = ImageResidualState(image=u, positive=p, negative=n, background=state.background)
        new_all = np.concatenate([u.ravel(), p.ravel(), n.ravel()])
        rel = float(np.linalg.norm(new_all - old_all) / (np.linalg.norm(old_all) + 1e-30))
        info = ImageResidualUpdateInfo(
            iterations=self.iterations,
            relative_change=rel,
            decomposition_residual=float(np.linalg.norm(new_state.decomposition_residual())),
            smooth_step=tau,
        )
        return new_state, info

