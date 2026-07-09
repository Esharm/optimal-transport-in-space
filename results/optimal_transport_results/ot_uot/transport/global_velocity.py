"""Global Eulerian velocity-field signed-residual UOT transport update."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ot_uot.core.variables import TransportState
from ot_uot.transport.path_solver import UOTPathInfo, solve_uot_path


@dataclass
class GlobalChannelState:
    """One global UOT path and scaled frame duals for one residual channel."""

    path: TransportState | None
    dual: np.ndarray


@dataclass
class GlobalTransportState:
    """Global transport state for positive and negative residual channels."""

    positive: GlobalChannelState
    negative: GlobalChannelState


@dataclass
class GlobalTransportInfo:
    positive: UOTPathInfo
    negative: UOTPathInfo

    @property
    def max_continuity_residual(self) -> float:
        return float(max(self.positive.continuity_residual, self.negative.continuity_residual))

    @property
    def total_action(self) -> float:
        return float(self.positive.action + self.negative.action)

    @property
    def total_source_abs(self) -> float:
        return float(self.positive.source_mass_abs + self.negative.source_mass_abs)


def initialize_global_state(frames: int, shape: tuple[int, int]) -> GlobalTransportState:
    """Create zero-dual, empty-path global state."""

    dual = np.zeros((frames, *shape), dtype=np.float64)
    return GlobalTransportState(
        positive=GlobalChannelState(path=None, dual=dual.copy()),
        negative=GlobalChannelState(path=None, dual=dual.copy()),
    )


class GlobalVelocityUOTTransport:
    """ADMM transport block for one global Eulerian UOT path per channel."""

    def __init__(
        self,
        transport_weight: float,
        source_weight: float,
        endpoint_penalty: float,
        inner_iters: int,
        tol: float,
    ):
        self.transport_weight = float(transport_weight)
        self.source_weight = float(source_weight)
        self.endpoint_penalty = float(endpoint_penalty)
        self.inner_iters = int(inner_iters)
        self.tol = float(tol)

    def _update_channel(
        self,
        channel: np.ndarray,
        state: GlobalChannelState,
    ) -> tuple[GlobalChannelState, UOTPathInfo]:
        target = channel - state.dual
        target_weight = np.full_like(channel, self.endpoint_penalty, dtype=np.float64)
        path, info = solve_uot_path(
            target=target,
            target_weight=target_weight,
            transport_weight=self.transport_weight,
            source_weight=self.source_weight,
            max_iter=self.inner_iters,
            tol=self.tol,
            state=state.path,
        )
        return GlobalChannelState(path=path, dual=state.dual), info

    def update(
        self,
        positive: np.ndarray,
        negative: np.ndarray,
        state: GlobalTransportState,
    ) -> tuple[GlobalTransportState, GlobalTransportInfo]:
        positive_state, positive_info = self._update_channel(positive, state.positive)
        negative_state, negative_info = self._update_channel(negative, state.negative)
        return (
            GlobalTransportState(positive=positive_state, negative=negative_state),
            GlobalTransportInfo(positive=positive_info, negative=negative_info),
        )

    def channel_quadratic_targets(self, state: GlobalChannelState) -> tuple[np.ndarray, np.ndarray]:
        """Return framewise quadratic targets for the image/residual update."""

        if state.path is None:
            raise ValueError("global transport path has not been initialized")
        return (
            self.endpoint_penalty * (state.path.density + state.dual),
            np.full_like(state.path.density, self.endpoint_penalty, dtype=np.float64),
        )

    def dual_update(
        self,
        channel: np.ndarray,
        state: GlobalChannelState,
        relaxation: float = 1.0,
    ) -> GlobalChannelState:
        """Update scaled frame dual variables for one residual channel."""

        if state.path is None:
            raise ValueError("global transport path has not been initialized")
        dual = state.dual + float(relaxation) * (state.path.density - channel)
        return GlobalChannelState(path=state.path, dual=dual)

