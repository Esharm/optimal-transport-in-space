"""Pairwise adjacent-frame signed-residual UOT transport update."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ot_uot.core.variables import TransportState
from ot_uot.transport.path_solver import UOTPathInfo, solve_uot_path


@dataclass
class PairwiseChannelState:
    """Transport paths and scaled endpoint duals for one residual channel."""

    paths: list[TransportState | None]
    dual_left: np.ndarray
    dual_right: np.ndarray


@dataclass
class PairwiseTransportState:
    """Pairwise transport state for positive and negative channels."""

    positive: PairwiseChannelState
    negative: PairwiseChannelState


@dataclass
class PairwiseTransportInfo:
    positive: list[UOTPathInfo]
    negative: list[UOTPathInfo]

    @property
    def max_continuity_residual(self) -> float:
        infos = self.positive + self.negative
        return float(max((info.continuity_residual for info in infos), default=0.0))

    @property
    def total_action(self) -> float:
        return float(sum(info.action for info in self.positive + self.negative))

    @property
    def total_source_abs(self) -> float:
        return float(sum(info.source_mass_abs for info in self.positive + self.negative))


def initialize_pairwise_state(frames: int, shape: tuple[int, int]) -> PairwiseTransportState:
    """Create zero-dual, empty-path pairwise state."""

    dual_shape = (frames - 1, *shape)
    return PairwiseTransportState(
        positive=PairwiseChannelState(
            paths=[None] * (frames - 1),
            dual_left=np.zeros(dual_shape, dtype=np.float64),
            dual_right=np.zeros(dual_shape, dtype=np.float64),
        ),
        negative=PairwiseChannelState(
            paths=[None] * (frames - 1),
            dual_left=np.zeros(dual_shape, dtype=np.float64),
            dual_right=np.zeros(dual_shape, dtype=np.float64),
        ),
    )


class PairwiseUOTTransport:
    """ADMM transport block for independent adjacent-frame UOT paths."""

    def __init__(
        self,
        transport_weight: float,
        source_weight: float,
        endpoint_penalty: float,
        nodes: int,
        inner_iters: int,
        tol: float,
    ):
        self.transport_weight = float(transport_weight)
        self.source_weight = float(source_weight)
        self.endpoint_penalty = float(endpoint_penalty)
        self.nodes = int(nodes)
        self.inner_iters = int(inner_iters)
        self.tol = float(tol)

    def _update_channel(
        self,
        channel: np.ndarray,
        state: PairwiseChannelState,
    ) -> tuple[PairwiseChannelState, list[UOTPathInfo]]:
        frames, height, width = channel.shape
        new_paths: list[TransportState] = []
        infos: list[UOTPathInfo] = []
        target_weight = np.zeros((self.nodes, height, width), dtype=np.float64)
        target_weight[0] = self.endpoint_penalty
        target_weight[-1] = self.endpoint_penalty

        for k in range(frames - 1):
            target = np.zeros((self.nodes, height, width), dtype=np.float64)
            target[0] = channel[k] - state.dual_left[k]
            target[-1] = channel[k + 1] - state.dual_right[k]
            if self.nodes > 2:
                for t in range(1, self.nodes - 1):
                    frac = t / (self.nodes - 1)
                    target[t] = (1.0 - frac) * target[0] + frac * target[-1]
            path, info = solve_uot_path(
                target=target,
                target_weight=target_weight,
                transport_weight=self.transport_weight,
                source_weight=self.source_weight,
                max_iter=self.inner_iters,
                tol=self.tol,
                state=state.paths[k],
            )
            new_paths.append(path)
            infos.append(info)

        return (
            PairwiseChannelState(
                paths=new_paths,
                dual_left=state.dual_left,
                dual_right=state.dual_right,
            ),
            infos,
        )

    def update(
        self,
        positive: np.ndarray,
        negative: np.ndarray,
        state: PairwiseTransportState,
    ) -> tuple[PairwiseTransportState, PairwiseTransportInfo]:
        """Run the pairwise transport update for both residual channels."""

        positive_state, positive_info = self._update_channel(positive, state.positive)
        negative_state, negative_info = self._update_channel(negative, state.negative)
        return (
            PairwiseTransportState(positive=positive_state, negative=negative_state),
            PairwiseTransportInfo(positive=positive_info, negative=negative_info),
        )

    @staticmethod
    def endpoint_targets(channel_state: PairwiseChannelState) -> tuple[np.ndarray, np.ndarray]:
        """Return left and right endpoint densities from current paths."""

        left = np.stack([path.density[0] for path in channel_state.paths], axis=0)
        right = np.stack([path.density[-1] for path in channel_state.paths], axis=0)
        return left, right

    def channel_quadratic_targets(
        self,
        state: PairwiseChannelState,
        frames: int,
        shape: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Aggregate endpoint quadratic targets for the image/residual update."""

        target_sum = np.zeros((frames, *shape), dtype=np.float64)
        weight_sum = np.zeros((frames, *shape), dtype=np.float64)
        left, right = self.endpoint_targets(state)
        for k in range(frames - 1):
            target_sum[k] += self.endpoint_penalty * (left[k] + state.dual_left[k])
            weight_sum[k] += self.endpoint_penalty
            target_sum[k + 1] += self.endpoint_penalty * (right[k] + state.dual_right[k])
            weight_sum[k + 1] += self.endpoint_penalty
        return target_sum, weight_sum

    def dual_update(
        self,
        channel: np.ndarray,
        state: PairwiseChannelState,
        relaxation: float = 1.0,
    ) -> PairwiseChannelState:
        """Update scaled endpoint dual variables for one residual channel."""

        left, right = self.endpoint_targets(state)
        dual_left = state.dual_left.copy()
        dual_right = state.dual_right.copy()
        dual_left += float(relaxation) * (left - channel[:-1])
        dual_right += float(relaxation) * (right - channel[1:])
        return PairwiseChannelState(paths=state.paths, dual_left=dual_left, dual_right=dual_right)

