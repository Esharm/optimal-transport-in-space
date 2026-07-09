"""ADMM state containers for signed-residual UOT reconstruction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ot_uot.core.variables import ImageResidualState


@dataclass
class ADMMHistoryEntry:
    """Scalar diagnostics from one outer ADMM iteration."""

    iteration: int
    objective: float
    data: float
    tv: float
    background: float
    reference: float
    residual_mass: float
    transport: float
    decomposition_residual: float
    endpoint_residual: float
    continuity_residual: float
    image_relative_change: float
    state_relative_change: float
    primal_residual: float
    dual_residual: float
    eps_primal: float
    eps_dual: float


@dataclass
class ADMMState:
    """Complete mutable optimization state."""

    image_state: ImageResidualState
    decomposition_dual: np.ndarray
    transport_state: Any
    iteration: int = 0
    history: list[ADMMHistoryEntry] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.decomposition_dual = np.asarray(self.decomposition_dual, dtype=np.float64)
        if self.decomposition_dual.shape != self.image_state.image.shape:
            raise ValueError("decomposition_dual must match image sequence shape")

