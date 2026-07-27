"""Per-frame static L1+TV+Hessian warm-start reconstruction."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable

import numpy as np

from ot_uot.core.finite_differences import divergence, gradient
from ot_uot.core.projections import prox_l1_nonnegative
from ot_uot.regularizers.hessian import hessian, hessian_adjoint, hessian_value, project_hessian_dual
from ot_uot.regularizers.tv import project_tv_dual, tv_value


@dataclass(frozen=True)
class StaticL1TVParameters:
    """Parameters for short per-frame static spatial warm-start solves.

    The objective for frame ``k`` is

    ``data_weight * D_k(u) + l1_weight * sum(u) + tv_weight * TV(u)``
    ``+ hessian_weight * HTV2(u)``

    subject to ``u >= 0``.  The solve is intentionally iteration-limited so it
    can be used as a warm start rather than as a fully converged static image.
    """

    l1_weight: float = 1e-8
    tv_weight: float = 1e-7
    hessian_weight: float = 0.0
    data_weight: float = 1.0
    iterations: int = 40
    tau_cap: float = 10.0
    dual_sigma: float = 0.25
    data_power_iters: int = 8
    seed: int = 1729

    def __post_init__(self) -> None:
        for name in ("l1_weight", "tv_weight", "hessian_weight", "data_weight"):
            value = float(getattr(self, name))
            if value < 0.0:
                raise ValueError(f"{name} must be nonnegative")
        if self.iterations < 0:
            raise ValueError("iterations must be nonnegative")
        if self.tau_cap <= 0.0:
            raise ValueError("tau_cap must be positive")
        if self.dual_sigma <= 0.0:
            raise ValueError("dual_sigma must be positive")
        if self.data_power_iters < 1:
            raise ValueError("data_power_iters must be positive")


@dataclass
class StaticL1TVFrameHistory:
    frame: int
    iteration: int
    objective: float
    data: float
    l1: float
    tv: float
    hessian: float
    relative_change: float
    step_size: float


def parameters_to_dict(params: StaticL1TVParameters) -> dict[str, float | int]:
    """Return a JSON-friendly parameter dictionary."""

    return asdict(params)


def _data_lipschitz(data_term, shape: tuple[int, int], *, power_iters: int, seed: int) -> float:
    """Estimate ``||A^* A||`` for one visibility data term."""

    rng = np.random.default_rng(seed)
    x = rng.normal(size=shape)
    x /= np.linalg.norm(x) + 1e-30
    eigenvalue = 0.0
    for _ in range(power_iters):
        applied = data_term.operator.adjoint(data_term.operator.forward(x))
        norm = np.linalg.norm(applied)
        if norm <= 1e-30:
            eigenvalue = 0.0
            break
        x = applied / norm
        eigenvalue = float(np.sum(x * applied))
    return max(1.1 * eigenvalue, 1e-12)


def static_l1_tv_objective(image: np.ndarray, data_term, params: StaticL1TVParameters) -> dict[str, float]:
    """Evaluate the static L1+TV warm-start objective for one frame."""

    image = np.asarray(image, dtype=np.float64)
    data = float(params.data_weight * data_term.value(image))
    l1 = float(params.l1_weight * np.sum(image))
    tv = float(params.tv_weight * tv_value(image))
    hessian_term = float(params.hessian_weight * hessian_value(image))
    return {
        "objective": data + l1 + tv + hessian_term,
        "data": data,
        "l1": l1,
        "tv": tv,
        "hessian": hessian_term,
    }


def solve_static_l1_tv_frame(
    data_term,
    shape: tuple[int, int],
    params: StaticL1TVParameters,
    *,
    initial: np.ndarray | None = None,
    frame_index: int = 0,
    callback: Callable[[StaticL1TVFrameHistory], None] | None = None,
) -> tuple[np.ndarray, list[StaticL1TVFrameHistory]]:
    """Run a short nonnegative L1+TV+Hessian reconstruction for one frame."""

    if initial is None:
        u = np.zeros(shape, dtype=np.float64)
    else:
        u = np.maximum(np.asarray(initial, dtype=np.float64), 0.0).copy()
        if u.shape != shape:
            raise ValueError("initial image shape does not match requested shape")

    data_lip = params.data_weight * _data_lipschitz(
        data_term,
        shape,
        power_iters=params.data_power_iters,
        seed=params.seed + int(frame_index),
    )
    dual_operator_bound = 0.0
    if params.tv_weight > 0.0:
        dual_operator_bound += 8.0
    if params.hessian_weight > 0.0:
        # ||H||^2 <= ||gradient||^4 <= 64 for H = gradient(gradient(.))
        dual_operator_bound += 64.0
    if dual_operator_bound > 0.0:
        tau = min(
            params.tau_cap,
            0.99 / (0.5 * data_lip + dual_operator_bound * params.dual_sigma),
        )
    else:
        tau = min(params.tau_cap, 0.99 / max(data_lip, 1e-12))

    q = np.zeros((2, *shape), dtype=np.float64)
    r = np.zeros((2, 2, *shape), dtype=np.float64)
    u_bar = u.copy()
    history: list[StaticL1TVFrameHistory] = []

    for iteration in range(1, int(params.iterations) + 1):
        if params.tv_weight > 0.0:
            q = project_tv_dual(q + params.dual_sigma * gradient(u_bar), params.tv_weight)
        if params.hessian_weight > 0.0:
            r = project_hessian_dual(
                r + params.dual_sigma * hessian(u_bar),
                params.hessian_weight,
            )

        old = u.copy()
        grad_u = np.zeros_like(u)
        if params.data_weight > 0.0:
            grad_u += params.data_weight * data_term.gradient(u)
        if params.tv_weight > 0.0:
            grad_u -= divergence(q)
        if params.hessian_weight > 0.0:
            grad_u += hessian_adjoint(r)

        u = prox_l1_nonnegative(u - tau * grad_u, tau * params.l1_weight)
        u_bar = 2.0 * u - old
        rel = float(np.linalg.norm(u - old) / (np.linalg.norm(old) + 1e-30))
        parts = static_l1_tv_objective(u, data_term, params)
        entry = StaticL1TVFrameHistory(
            frame=int(frame_index),
            iteration=int(iteration),
            objective=float(parts["objective"]),
            data=float(parts["data"]),
            l1=float(parts["l1"]),
            tv=float(parts["tv"]),
            hessian=float(parts["hessian"]),
            relative_change=rel,
            step_size=float(tau),
        )
        history.append(entry)
        if callback is not None:
            callback(entry)

    return u, history


def run_static_l1_tv_warm_start(
    data_terms,
    shape: tuple[int, int],
    params: StaticL1TVParameters,
    *,
    initial_sequence: np.ndarray | None = None,
    callback: Callable[[StaticL1TVFrameHistory], None] | None = None,
) -> tuple[np.ndarray, list[StaticL1TVFrameHistory]]:
    """Run independent short L1+TV+Hessian warm-start solves for all frames.

    Parameters
    ----------
    initial_sequence:
        Optional warm-start image(s).  If provided as ``(H,W)``, the same image
        is used for every frame.  If provided as ``(K,H,W)``, frame ``k`` uses
        ``initial_sequence[k]``.  ``None`` preserves the original black-image
        initialization.
    """

    initial_frames = None
    if initial_sequence is not None:
        initial_frames = np.asarray(initial_sequence, dtype=np.float64)
        if initial_frames.ndim == 2:
            if initial_frames.shape != tuple(shape):
                raise ValueError("2D initial image shape does not match requested shape")
            initial_frames = np.repeat(initial_frames[None, :, :], len(data_terms), axis=0)
        elif initial_frames.ndim == 3:
            if initial_frames.shape[0] != len(data_terms) or initial_frames.shape[1:] != tuple(shape):
                raise ValueError("3D initial sequence shape must be (frames,H,W)")
        else:
            raise ValueError("initial_sequence must have shape (H,W), (K,H,W), or be None")

    frames: list[np.ndarray] = []
    history: list[StaticL1TVFrameHistory] = []
    for k, data_term in enumerate(data_terms):
        initial = None if initial_frames is None else initial_frames[k]
        frame, frame_history = solve_static_l1_tv_frame(
            data_term,
            shape,
            params,
            initial=initial,
            frame_index=k,
            callback=callback,
        )
        frames.append(frame)
        history.extend(frame_history)
    return np.stack(frames, axis=0), history
