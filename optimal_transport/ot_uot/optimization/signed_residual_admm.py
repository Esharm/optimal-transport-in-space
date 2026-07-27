"""Exact variational ADMM for signed-residual UOT dynamic reconstruction."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from ot_uot.core.background import decompose_signed_residual
from ot_uot.core.config import TransportMethod, UOTParameters
from ot_uot.core.variables import ImageResidualState
from ot_uot.optimization.admm_state import ADMMHistoryEntry, ADMMState
from ot_uot.optimization.convergence import convergence_status, state_difference_norm
from ot_uot.optimization.objective import constraint_residuals, objective_breakdown
from ot_uot.regularizers.image_residual_update import ChannelTargets, ImageResidualUpdater
from ot_uot.transport.global_velocity import GlobalVelocityUOTTransport, initialize_global_state
from ot_uot.transport.pairwise import PairwiseUOTTransport, initialize_pairwise_state


class SignedResidualUOTADMM:
    """Standalone optimizer for the exact signed-residual UOT formulation."""

    def __init__(self, params: UOTParameters):
        self.params = params
        self.image_updater = ImageResidualUpdater(
            tv_weight=params.tv_weight,
            hessian_weight=params.hessian_weight,
            background_weight=params.background_weight,
            residual_mass_weight=params.residual_mass_weight,
            image_l1_weight=params.image_l1_weight,
            decomposition_penalty=params.decomposition_penalty,
            data_weight=params.data_weight,
            reference_weight=params.reference_weight,
            iterations=params.image_inner_iters,
        )
        if params.transport_method == TransportMethod.PAIRWISE_UOT:
            self.transport_block = PairwiseUOTTransport(
                transport_weight=params.transport_weight,
                source_weight=params.source_weight,
                endpoint_penalty=params.endpoint_penalty,
                nodes=params.transport_nodes,
                inner_iters=params.transport_inner_iters,
                tol=params.abs_tol,
            )
        elif params.transport_method == TransportMethod.GLOBAL_VELOCITY:
            self.transport_block = GlobalVelocityUOTTransport(
                transport_weight=params.transport_weight,
                source_weight=params.source_weight,
                endpoint_penalty=params.endpoint_penalty,
                inner_iters=params.transport_inner_iters,
                tol=params.abs_tol,
            )
        else:
            raise ValueError(f"unsupported transport method: {params.transport_method}")

    def initialize(self, static_sequence: np.ndarray, background: np.ndarray) -> ADMMState:
        """Initialize image/residual and transport variables from static frames."""

        image = np.maximum(np.asarray(static_sequence, dtype=np.float64), 0.0)
        positive, negative = decompose_signed_residual(image, background)
        image_state = ImageResidualState(
            image=image,
            positive=positive,
            negative=negative,
            background=np.asarray(background, dtype=np.float64),
        )
        frames = image.shape[0]
        shape = image.shape[1:]
        if self.params.transport_method == TransportMethod.PAIRWISE_UOT:
            transport_state = initialize_pairwise_state(frames, shape)
        else:
            transport_state = initialize_global_state(frames, shape)
        return ADMMState(
            image_state=image_state,
            decomposition_dual=np.zeros_like(image, dtype=np.float64),
            transport_state=transport_state,
        )

    def _transport_update(self, state: ADMMState) -> tuple[Any, Any]:
        return self.transport_block.update(
            state.image_state.positive,
            state.image_state.negative,
            state.transport_state,
        )

    def _channel_targets(self, transport_state: Any, frames: int, shape: tuple[int, int]) -> tuple[ChannelTargets, ChannelTargets]:
        if self.params.transport_method == TransportMethod.PAIRWISE_UOT:
            pos_sum, pos_weight = self.transport_block.channel_quadratic_targets(
                transport_state.positive, frames, shape
            )
            neg_sum, neg_weight = self.transport_block.channel_quadratic_targets(
                transport_state.negative, frames, shape
            )
        else:
            pos_sum, pos_weight = self.transport_block.channel_quadratic_targets(transport_state.positive)
            neg_sum, neg_weight = self.transport_block.channel_quadratic_targets(transport_state.negative)
        return ChannelTargets(pos_sum, pos_weight), ChannelTargets(neg_sum, neg_weight)

    def _dual_update(self, state: ADMMState) -> Any:
        if self.params.transport_method == TransportMethod.PAIRWISE_UOT:
            positive = self.transport_block.dual_update(
                state.image_state.positive,
                state.transport_state.positive,
                self.params.dual_relaxation,
            )
            negative = self.transport_block.dual_update(
                state.image_state.negative,
                state.transport_state.negative,
                self.params.dual_relaxation,
            )
            return type(state.transport_state)(positive=positive, negative=negative)

        positive = self.transport_block.dual_update(
            state.image_state.positive,
            state.transport_state.positive,
            self.params.dual_relaxation,
        )
        negative = self.transport_block.dual_update(
            state.image_state.negative,
            state.transport_state.negative,
            self.params.dual_relaxation,
        )
        return type(state.transport_state)(positive=positive, negative=negative)

    def step(self, state: ADMMState, data_terms, reference_sequence: np.ndarray | None = None) -> ADMMState:
        """Run one exact outer ADMM iteration."""

        previous_image_state = state.image_state
        transport_state, transport_info = self._transport_update(state)
        frames, height, width = previous_image_state.image.shape
        pos_targets, neg_targets = self._channel_targets(transport_state, frames, (height, width))
        image_state, image_info = self.image_updater.update(
            previous_image_state,
            data_terms,
            state.decomposition_dual,
            pos_targets,
            neg_targets,
            reference_sequence=reference_sequence,
        )
        decomposition_dual = state.decomposition_dual + self.params.dual_relaxation * image_state.decomposition_residual()
        next_state = ADMMState(
            image_state=image_state,
            decomposition_dual=decomposition_dual,
            transport_state=transport_state,
            iteration=state.iteration + 1,
            history=list(state.history),
        )
        next_state.transport_state = self._dual_update(next_state)

        residuals = constraint_residuals(
            next_state.image_state,
            next_state.transport_state,
            self.params.transport_method,
        )
        obj = objective_breakdown(
            next_state.image_state,
            next_state.transport_state,
            data_terms,
            self.params,
            reference_sequence=reference_sequence,
        )
        status = convergence_status(
            next_state.image_state,
            previous_image_state,
            residuals,
            self.params,
            next_state.iteration,
        )
        history = ADMMHistoryEntry(
            iteration=next_state.iteration,
            objective=obj.total,
            data=obj.data,
            tv=obj.tv,
            hessian=obj.hessian,
            image_l1=obj.image_l1,
            background=obj.background,
            reference=obj.reference,
            residual_mass=obj.residual_mass,
            transport=obj.transport,
            decomposition_residual=residuals.decomposition_l2,
            endpoint_residual=residuals.endpoint_l2,
            continuity_residual=residuals.continuity_l2,
            image_relative_change=image_info.relative_change,
            state_relative_change=status.state_relative_change,
            primal_residual=residuals.primal_l2,
            dual_residual=state_difference_norm(next_state.image_state, previous_image_state),
            eps_primal=status.eps_primal,
            eps_dual=status.eps_dual,
        )
        next_state.history.append(history)
        return next_state

    def run(
        self,
        static_sequence: np.ndarray,
        background: np.ndarray,
        data_terms,
        callback=None,
        reference_sequence: np.ndarray | None = None,
    ) -> ADMMState:
        """Run ADMM until convergence or the configured iteration limit."""

        if reference_sequence is not None:
            reference_sequence = np.asarray(reference_sequence, dtype=np.float64)
            if reference_sequence.shape != np.asarray(static_sequence).shape:
                raise ValueError("reference_sequence shape must match static_sequence")
        elif self.params.reference_weight > 0.0:
            reference_sequence = np.asarray(static_sequence, dtype=np.float64)
        state = self.initialize(static_sequence, background)
        stable_iterations = 0
        for _ in range(self.params.max_admm_iters):
            state = self.step(state, data_terms, reference_sequence=reference_sequence)
            latest = state.history[-1]
            if callback is not None:
                callback(state)
            converged_now = (
                state.iteration >= self.params.min_admm_iters
                and latest.primal_residual <= latest.eps_primal
                and latest.dual_residual <= latest.eps_dual
            )
            stable_iterations = stable_iterations + 1 if converged_now else 0
            if stable_iterations >= self.params.patience:
                break
        return state


def history_as_dicts(state: ADMMState) -> list[dict[str, float | int]]:
    """Serialize ADMM history entries to plain dictionaries."""

    return [asdict(entry) for entry in state.history]
