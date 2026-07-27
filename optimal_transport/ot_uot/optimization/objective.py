"""Objective and residual evaluation for signed-residual UOT ADMM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ot_uot.core.background import as_background_sequence
from ot_uot.core.config import TransportMethod, UOTParameters
from ot_uot.core.variables import ImageResidualState, TransportState
from ot_uot.regularizers.hessian import sequence_hessian_value
from ot_uot.regularizers.tv import sequence_tv_value
from ot_uot.transport.global_velocity import GlobalTransportState
from ot_uot.transport.pairwise import PairwiseTransportState
from ot_uot.transport.path_solver import uot_action


@dataclass
class ObjectiveBreakdown:
    """Named components of the unaugmented variational objective."""

    total: float
    data: float
    tv: float
    hessian: float
    image_l1: float
    background: float
    reference: float
    residual_mass: float
    transport: float


@dataclass
class ConstraintResiduals:
    """Constraint residual norms used by ADMM diagnostics."""

    decomposition_l2: float
    endpoint_l2: float
    continuity_l2: float
    primal_l2: float


def _path_action(path: TransportState, params: UOTParameters) -> float:
    dt = 1.0 / float(path.density.shape[0] - 1)
    return uot_action(path, params.transport_weight, params.source_weight, dt)


def _path_continuity_l2(path: TransportState) -> float:
    from ot_uot.transport.continuity import uot_continuity

    dt = 1.0 / float(path.density.shape[0] - 1)
    return float(np.linalg.norm(uot_continuity(path.density, path.momentum, path.source, dt)))


def transport_objective(state: Any, method: TransportMethod, params: UOTParameters) -> float:
    """Evaluate the transport action for the selected transport block."""

    method = TransportMethod(method)
    total = 0.0
    if method == TransportMethod.PAIRWISE_UOT:
        if not isinstance(state, PairwiseTransportState):
            raise TypeError("pairwise objective requires PairwiseTransportState")
        for channel in (state.positive, state.negative):
            for path in channel.paths:
                if path is not None:
                    total += _path_action(path, params)
    elif method == TransportMethod.GLOBAL_VELOCITY:
        if not isinstance(state, GlobalTransportState):
            raise TypeError("global objective requires GlobalTransportState")
        for channel in (state.positive, state.negative):
            if channel.path is not None:
                total += _path_action(channel.path, params)
    else:
        raise ValueError(f"unsupported transport method: {method}")
    return float(total)


def objective_breakdown(
    image_state: ImageResidualState,
    transport_state: Any,
    data_terms,
    params: UOTParameters,
    reference_sequence: np.ndarray | None = None,
) -> ObjectiveBreakdown:
    """Evaluate the unaugmented reconstruction objective."""

    raw_data = float(sum(term.value(frame) for term, frame in zip(data_terms, image_state.image)))
    data = float(params.data_weight * raw_data)
    tv = float(params.tv_weight * sequence_tv_value(image_state.image))
    hessian_term = float(params.hessian_weight * sequence_hessian_value(image_state.image))
    image_l1 = float(params.image_l1_weight * np.sum(image_state.image))
    background = as_background_sequence(image_state.background, image_state.image.shape[0])
    background_value = float(0.5 * params.background_weight * np.sum((image_state.image - background) ** 2))
    if reference_sequence is not None:
        reference_sequence = np.asarray(reference_sequence, dtype=np.float64)
        if reference_sequence.shape != image_state.image.shape:
            raise ValueError("reference_sequence shape must match image sequence")
        reference_value = float(0.5 * params.reference_weight * np.sum((image_state.image - reference_sequence) ** 2))
    else:
        reference_value = 0.0
    residual_mass = float(params.residual_mass_weight * np.sum(image_state.positive + image_state.negative))
    transport = transport_objective(transport_state, params.transport_method, params)
    total = data + tv + hessian_term + image_l1 + background_value + reference_value + residual_mass + transport
    return ObjectiveBreakdown(
        total=float(total),
        data=data,
        tv=tv,
        hessian=hessian_term,
        image_l1=image_l1,
        background=background_value,
        reference=reference_value,
        residual_mass=residual_mass,
        transport=transport,
    )


def constraint_residuals(
    image_state: ImageResidualState,
    transport_state: Any,
    method: TransportMethod,
) -> ConstraintResiduals:
    """Compute decomposition, endpoint/frame, and continuity residuals."""

    decomposition = float(np.linalg.norm(image_state.decomposition_residual()))
    endpoint_sq = 0.0
    continuity_sq = 0.0
    method = TransportMethod(method)

    if method == TransportMethod.PAIRWISE_UOT:
        if not isinstance(transport_state, PairwiseTransportState):
            raise TypeError("pairwise residuals require PairwiseTransportState")
        for channel_values, channel_state in (
            (image_state.positive, transport_state.positive),
            (image_state.negative, transport_state.negative),
        ):
            for k, path in enumerate(channel_state.paths):
                if path is None:
                    continue
                endpoint_sq += float(np.sum((path.density[0] - channel_values[k]) ** 2))
                endpoint_sq += float(np.sum((path.density[-1] - channel_values[k + 1]) ** 2))
                continuity_sq += _path_continuity_l2(path) ** 2
    elif method == TransportMethod.GLOBAL_VELOCITY:
        if not isinstance(transport_state, GlobalTransportState):
            raise TypeError("global residuals require GlobalTransportState")
        for channel_values, channel_state in (
            (image_state.positive, transport_state.positive),
            (image_state.negative, transport_state.negative),
        ):
            if channel_state.path is None:
                continue
            endpoint_sq += float(np.sum((channel_state.path.density - channel_values) ** 2))
            continuity_sq += _path_continuity_l2(channel_state.path) ** 2
    else:
        raise ValueError(f"unsupported transport method: {method}")

    endpoint = float(np.sqrt(endpoint_sq))
    continuity = float(np.sqrt(continuity_sq))
    primal = float(np.sqrt(decomposition * decomposition + endpoint * endpoint))
    return ConstraintResiduals(
        decomposition_l2=decomposition,
        endpoint_l2=endpoint,
        continuity_l2=continuity,
        primal_l2=primal,
    )

