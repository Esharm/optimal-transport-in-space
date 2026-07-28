#!/usr/bin/env python3
"""Multiscale UOT path-consistency metric with lag diagnostics.

This evaluator is deliberately separate from any reconstruction/post-processing
OT solver.  For each ordered image pair a -> b, it solves the static entropic,
KL-relaxed unbalanced OT problem

    min_{Gamma >= 0}
        <C, Gamma>
        + eps * KL(Gamma || a b^T)
        + tau * KL(Gamma 1 || a)
        + tau * KL(Gamma^T 1 || b).

The relaxed marginal penalties permit brightness destruction/amplification.
The coupling is converted to a source-conditioned transfer kernel

    K_ij = Gamma_ij / a_i.

For lag ell, the evaluator compares the direct endpoint kernel

    D_{k,ell} = K_{k -> k+ell}

with the adjacent-path composition

    P_{k,ell} = K_{k -> k+1} ... K_{k+ell-1 -> k+ell}.

The source-weighted relative L1 composition defect is

                  sum_i I_k[i] ||D[i,:] - P[i,:]||_1
    e_{k,ell} = --------------------------------------------------- .
                  sum_i I_k[i](||D[i,:]||_1+||P[i,:]||_1) + tiny

For a candidate sequence S and reference GT,

    G_ell(S) = mean_k |e_{k,ell}(S) - e_{k,ell}(GT)|.

This version adds diagnostics intended to identify when large lags stop being
informative because of too few windows, accumulated unbalanced mass collapse,
or diffusion of repeatedly composed kernels.  It reports, for each lag:

* number of windows;
* moving-block bootstrap confidence intervals for E_ell and G_ell;
* direct/path output-mass ratios;
* path-to-direct mass ratio;
* normalized conditional row entropy;
* effective destination support;
* conditional spatial spread about each row barycenter;
* RMS source-to-destination displacement;
* an optional heuristic useful/unreliable flag with explicit reasons.



python uot_path_consistency_diagnostics_v2.py \
    --reference ../../blackhole_sim/data/aart_frames \
    --sequence starwarps=../../results/starwarps_results/34_telescopes/final_frames_34_png \
    --sequence static=../../results/static_reconstruction_results/reconstructed_frames_gray \
    --sequence uot_SW=../../results/optimal_transport_results/ot_uot_SW_34/frames \
    --sequence uot_static=../../results/optimal_transport_results/ot_uot_static_34/frames \
    --n 32 \
    --lags 2 3 4 5 6 7 8 9 10 11 12 13 14 \
    --kernel-topk 64 \
    --bootstrap-samples 2000 \
    --json uot_path_diagnostics.json \
    --csv uot_path_diagnostics.csv \
    --window-csv uot_path_windows.csv \
    --plot uot_G_l_with_uncertainty.png \
    --diagnostic-plot uot_lag_diagnostics.png

Dependencies
------------
Required: numpy, scipy, Pillow
Optional: POT (``pip install POT``), matplotlib

Input formats
-------------
* Directory containing image files (natural filename order)
* .npy array with shape (K,H,W) or (K,H,W,C)
* .npz containing one array, or ``file.npz::array_key``

All sequences must contain the same number of frames.  Frames are resized to
N x N with box/area resampling.  N defaults to 32 and may not be below 32.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

__version__ = "2.1.0-diagnostics"

import numpy as np
from PIL import Image
from scipy import sparse

try:  # Optional dependency.
    import ot  # type: ignore
except Exception:  # pragma: no cover
    ot = None


IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
    ".webp",
}


@dataclass(frozen=True)
class EvaluatorConfig:
    n: int = 32
    lags: tuple[int, ...] = (2, 3, 4, 5)
    entropic_reg: float = 0.03
    marginal_reg: float = 10.0
    max_iter: int = 1000
    stop_thr: float = 1e-8
    solver: str = "auto"
    pot_method: str = "sinkhorn_stabilized"
    kernel_topk: int = 64
    mass_floor_rel: float = 1e-12
    numerical_floor: float = 1e-300
    dtype: str = "float64"

    # Uncertainty diagnostics.
    bootstrap_samples: int = 1000
    bootstrap_confidence: float = 0.95
    bootstrap_seed: int = 0
    bootstrap_block_length: int = 0  # 0 => use min(lag, number of windows)

    # Heuristic lag-usability criteria. These do not alter the metric.
    min_windows: int = 5
    min_path_mass_ratio: float = 0.05
    min_path_to_direct_mass_ratio: float = 0.10
    max_path_to_direct_mass_ratio: float = 10.0
    max_path_entropy_excess: float = 0.15
    max_g_ci_width: float = 0.25

    def validate(self) -> None:
        if self.n < 32:
            raise ValueError("Evaluation resolution N must be at least 32.")
        if not self.lags or any(lag < 2 for lag in self.lags):
            raise ValueError("All lags must be integers >= 2.")
        if self.entropic_reg <= 0:
            raise ValueError("entropic_reg must be > 0.")
        if self.marginal_reg <= 0:
            raise ValueError("marginal_reg must be > 0.")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive.")
        if self.stop_thr <= 0:
            raise ValueError("stop_thr must be positive.")
        if self.kernel_topk <= 0:
            raise ValueError("kernel_topk must be positive.")
        if self.solver not in {"auto", "pot", "numpy"}:
            raise ValueError("solver must be one of: auto, pot, numpy.")
        if self.bootstrap_samples < 0:
            raise ValueError("bootstrap_samples must be >= 0.")
        if not 0 < self.bootstrap_confidence < 1:
            raise ValueError("bootstrap_confidence must lie in (0,1).")
        if self.bootstrap_block_length < 0:
            raise ValueError("bootstrap_block_length must be >= 0.")
        if self.min_windows < 1:
            raise ValueError("min_windows must be >= 1.")
        if self.min_path_mass_ratio < 0:
            raise ValueError("min_path_mass_ratio must be >= 0.")
        if self.min_path_to_direct_mass_ratio <= 0:
            raise ValueError("min_path_to_direct_mass_ratio must be > 0.")
        if self.max_path_to_direct_mass_ratio <= self.min_path_to_direct_mass_ratio:
            raise ValueError("max_path_to_direct_mass_ratio must exceed the minimum.")
        if self.max_path_entropy_excess < 0:
            raise ValueError("max_path_entropy_excess must be >= 0.")
        if self.max_g_ci_width < 0:
            raise ValueError("max_g_ci_width must be >= 0.")


@dataclass
class BootstrapSummary:
    mean: float
    std: float
    ci_low: float | None
    ci_high: float | None
    ci_width: float | None
    ci_available: bool
    n: int
    block_length: int


@dataclass
class KernelDiagnostics:
    """Diagnostics for one transfer kernel relative to its starting image."""

    output_mass_ratio: float
    entropy_fraction: float
    effective_support: float
    support_saturation: float
    conditional_spread_rms: float
    displacement_rms: float


@dataclass
class WindowDiagnostics:
    start: int
    lag: int
    error: float
    direct: KernelDiagnostics
    path: KernelDiagnostics
    path_to_direct_mass_ratio: float
    entropy_excess: float
    support_excess: float
    spread_excess: float


@dataclass
class LagDiagnosticsSummary:
    lag: int
    n_windows: int
    raw_error: BootstrapSummary
    direct_mass_ratio_mean: float
    direct_mass_ratio_std: float
    path_mass_ratio_mean: float
    path_mass_ratio_std: float
    path_to_direct_mass_ratio_mean: float
    path_to_direct_mass_ratio_std: float
    direct_entropy_fraction_mean: float
    path_entropy_fraction_mean: float
    direct_effective_support_mean: float
    path_effective_support_mean: float
    direct_support_saturation_mean: float
    path_support_saturation_mean: float
    direct_spread_rms_mean: float
    path_spread_rms_mean: float
    direct_displacement_rms_mean: float
    path_displacement_rms_mean: float


@dataclass
class SequenceConsistency:
    label: str
    mean_by_lag: dict[int, float]
    windows_by_lag: dict[int, list[float]]
    diagnostics_by_lag: dict[int, list[WindowDiagnostics]]
    summaries_by_lag: dict[int, LagDiagnosticsSummary]
    solve_count: int
    elapsed_seconds: float


@dataclass
class CandidateComparison:
    label: str
    g_by_lag: dict[int, float]
    g_windows_by_lag: dict[int, list[float]]
    g_uncertainty_by_lag: dict[int, BootstrapSummary]
    raw_mean_by_lag: dict[int, float]
    raw_windows_by_lag: dict[int, list[float]]
    delta_mean_by_lag: dict[int, float]
    diagnostics_by_lag: dict[int, LagDiagnosticsSummary]
    usable_by_lag: dict[int, bool]
    usability_reasons_by_lag: dict[int, list[str]]
    solve_count: int
    elapsed_seconds: float


def natural_sort_key(path: Path) -> list[Any]:
    parts = re.split(r"(\d+)", path.name.lower())
    return [int(part) if part.isdigit() else part for part in parts]


def parse_array_spec(spec: str) -> tuple[Path, str | None]:
    if "::" in spec:
        path_text, key = spec.rsplit("::", 1)
        if not key:
            raise ValueError(f"Missing NPZ key after '::' in {spec!r}.")
        return Path(path_text), key
    return Path(spec), None


def _to_grayscale_float(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim == 2:
        return array.astype(np.float64, copy=False)
    if array.ndim != 3:
        raise ValueError(f"Expected a 2-D image or HxWxC array; got shape {array.shape}.")

    channels = array.shape[-1]
    work = array.astype(np.float64, copy=False)
    if channels == 1:
        return work[..., 0]
    if channels >= 3:
        return 0.2126 * work[..., 0] + 0.7152 * work[..., 1] + 0.0722 * work[..., 2]
    return work.mean(axis=-1)


def resize_frame(frame: np.ndarray, n: int) -> np.ndarray:
    """Resize one frame to n x n using box/area resampling."""
    gray = _to_grayscale_float(frame)
    if gray.shape == (n, n):
        return np.array(gray, dtype=np.float64, copy=True)
    image = Image.fromarray(gray.astype(np.float32), mode="F")
    resized = image.resize((n, n), resample=Image.Resampling.BOX)
    return np.asarray(resized, dtype=np.float64)


def load_image_file(path: Path) -> np.ndarray:
    try:
        with Image.open(path) as image:
            array = np.asarray(image)
    except Exception as exc:
        raise RuntimeError(f"Failed to read image {path}: {exc}") from exc
    return _to_grayscale_float(array)


def _frames_from_array(array: np.ndarray, source: str) -> list[np.ndarray]:
    array = np.asarray(array)
    if array.ndim == 3:
        return [array[k] for k in range(array.shape[0])]
    if array.ndim == 4:
        return [array[k] for k in range(array.shape[0])]
    raise ValueError(
        f"Array {source} must have shape (K,H,W) or (K,H,W,C); got {array.shape}."
    )


def load_sequence(spec: str, n: int) -> np.ndarray:
    """Load and resize a sequence to shape (K,n,n)."""
    path, npz_key = parse_array_spec(spec)
    if not path.exists():
        raise FileNotFoundError(f"Input does not exist: {path}")

    frames: list[np.ndarray]
    if path.is_dir():
        files = sorted(
            [p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS],
            key=natural_sort_key,
        )
        if not files:
            raise ValueError(f"No supported image files found in directory: {path}")
        frames = [load_image_file(file) for file in files]
    elif path.suffix.lower() == ".npy":
        frames = _frames_from_array(np.load(path, allow_pickle=False), source=str(path))
    elif path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            keys = list(archive.keys())
            if npz_key is None:
                if len(keys) != 1:
                    raise ValueError(
                        f"{path} contains arrays {keys}; specify one as '{path}::KEY'."
                    )
                npz_key = keys[0]
            if npz_key not in archive:
                raise KeyError(f"Key {npz_key!r} not found in {path}; available: {keys}")
            frames = _frames_from_array(archive[npz_key], source=f"{path}::{npz_key}")
    elif path.suffix.lower() in IMAGE_EXTENSIONS:
        raise ValueError(
            f"{path} is one image, not a sequence. Put sequence images in a directory."
        )
    else:
        raise ValueError(
            f"Unsupported input {path}. Use a directory of images, .npy, or .npz."
        )

    resized = np.stack([resize_frame(frame, n) for frame in frames], axis=0)
    if not np.all(np.isfinite(resized)):
        raise ValueError(f"Sequence {spec!r} contains NaN or infinite values.")

    min_value = float(resized.min())
    max_abs = float(np.max(np.abs(resized)))
    tolerance = 1e-10 * max(1.0, max_abs)
    if min_value < -tolerance:
        raise ValueError(
            f"Sequence {spec!r} contains significant negative intensities "
            f"(minimum {min_value:.6g}). UOT requires nonnegative images."
        )
    return np.maximum(resized, 0.0)


def build_grid_coordinates(n: int) -> np.ndarray:
    axis = np.linspace(0.0, 1.0, n, dtype=np.float64)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    return np.column_stack([yy.ravel(), xx.ravel()])


def build_cost_matrix(n: int, dtype: np.dtype = np.float64) -> np.ndarray:
    """Squared Euclidean cost on [0,1]^2, normalized so max cost is 1."""
    coords = build_grid_coordinates(n)
    sq_norm = np.sum(coords * coords, axis=1)
    cost = sq_norm[:, None] + sq_norm[None, :] - 2.0 * coords @ coords.T
    np.maximum(cost, 0.0, out=cost)
    cost /= 2.0
    return cost.astype(dtype, copy=False)


def generalized_sinkhorn_numpy(
    a: np.ndarray,
    b: np.ndarray,
    cost: np.ndarray,
    entropic_reg: float,
    marginal_reg: float,
    max_iter: int,
    stop_thr: float,
    numerical_floor: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Solve KL-relaxed entropic UOT by generalized Sinkhorn scaling."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape:
        raise ValueError("a and b must be one-dimensional arrays with equal shape.")
    if np.any(a < 0) or np.any(b < 0):
        raise ValueError("UOT histograms must be nonnegative.")
    if float(a.sum()) <= 0 or float(b.sum()) <= 0:
        raise ValueError("Each frame must contain positive total brightness.")

    exponent = marginal_reg / (marginal_reg + entropic_reg)
    gibbs = np.exp(-cost / entropic_reg)
    reference = a[:, None] * b[None, :]
    q = reference * gibbs

    u = np.ones_like(a)
    v = np.ones_like(b)
    converged = False
    error = math.inf

    for iteration in range(1, max_iter + 1):
        previous_u = u.copy() if iteration % 10 == 0 else None
        previous_v = v.copy() if iteration % 10 == 0 else None

        qv = q @ v
        ratio_u = np.divide(a, qv, out=np.zeros_like(a), where=qv > numerical_floor)
        u = np.power(ratio_u, exponent, where=ratio_u > 0, out=np.zeros_like(ratio_u))

        qtu = q.T @ u
        ratio_v = np.divide(b, qtu, out=np.zeros_like(b), where=qtu > numerical_floor)
        v = np.power(ratio_v, exponent, where=ratio_v > 0, out=np.zeros_like(ratio_v))

        if not np.all(np.isfinite(u)) or not np.all(np.isfinite(v)):
            raise FloatingPointError(
                "Generalized Sinkhorn produced non-finite scalings. "
                "Increase --entropic-reg or use --solver pot."
            )

        if iteration % 10 == 0 and previous_u is not None and previous_v is not None:
            du = np.max(np.abs(u - previous_u) / np.maximum(1.0, np.abs(previous_u)))
            dv = np.max(np.abs(v - previous_v) / np.maximum(1.0, np.abs(previous_v)))
            error = float(max(du, dv))
            if error < stop_thr:
                converged = True
                break

    gamma = (u[:, None] * q) * v[None, :]
    if not np.all(np.isfinite(gamma)):
        raise FloatingPointError("Computed UOT coupling contains non-finite values.")
    return gamma, {
        "iterations": iteration,
        "converged": converged,
        "error": error,
        "backend": "numpy",
    }


def _last_log_error(log: Mapping[str, Any]) -> float | None:
    errors = log.get("err")
    if errors is None:
        return None
    try:
        if len(errors) == 0:
            return None
        return float(errors[-1])
    except Exception:
        return None


def solve_uot_coupling(
    a: np.ndarray,
    b: np.ndarray,
    cost: np.ndarray,
    config: EvaluatorConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    solver = config.solver
    if solver == "auto":
        solver = "pot" if ot is not None else "numpy"

    if solver == "pot":
        if ot is None:
            raise ImportError(
                "POT is not installed. Install it with 'pip install POT' or use --solver numpy."
            )
        gamma, log = ot.unbalanced.sinkhorn_unbalanced(
            a,
            b,
            cost,
            reg=config.entropic_reg,
            reg_m=config.marginal_reg,
            method=config.pot_method,
            reg_type="kl",
            numItermax=config.max_iter,
            stopThr=config.stop_thr,
            log=True,
            verbose=False,
        )
        gamma = np.asarray(gamma, dtype=np.float64)
        if gamma.shape != cost.shape or not np.all(np.isfinite(gamma)):
            raise FloatingPointError("POT returned an invalid UOT coupling.")
        return gamma, {
            "backend": "pot",
            "method": config.pot_method,
            "iterations": log.get("niter", log.get("it", None)),
            "error": _last_log_error(log),
        }

    return generalized_sinkhorn_numpy(
        a=a,
        b=b,
        cost=cost,
        entropic_reg=config.entropic_reg,
        marginal_reg=config.marginal_reg,
        max_iter=config.max_iter,
        stop_thr=config.stop_thr,
        numerical_floor=config.numerical_floor,
    )


def prune_csr_rows_preserve_mass(matrix: sparse.spmatrix, topk: int) -> sparse.csr_matrix:
    """Keep the largest top-k values per row and preserve each row's L1 mass."""
    csr = matrix.tocsr().astype(np.float64, copy=False)
    n_rows, n_cols = csr.shape
    if topk >= n_cols:
        csr.eliminate_zeros()
        return csr

    data_parts: list[np.ndarray] = []
    index_parts: list[np.ndarray] = []
    indptr = np.zeros(n_rows + 1, dtype=np.int64)

    for row in range(n_rows):
        start, end = csr.indptr[row], csr.indptr[row + 1]
        values = csr.data[start:end]
        columns = csr.indices[start:end]
        if values.size == 0:
            indptr[row + 1] = indptr[row]
            continue

        positive = values > 0
        values = values[positive]
        columns = columns[positive]
        if values.size == 0:
            indptr[row + 1] = indptr[row]
            continue

        full_mass = float(values.sum())
        if values.size > topk:
            selected = np.argpartition(values, -topk)[-topk:]
            values = values[selected]
            columns = columns[selected]

        order = np.argsort(columns)
        values = values[order]
        columns = columns[order]
        kept_mass = float(values.sum())
        if kept_mass > 0 and full_mass > 0:
            values = values * (full_mass / kept_mass)

        data_parts.append(values)
        index_parts.append(columns)
        indptr[row + 1] = indptr[row] + values.size

    data = np.concatenate(data_parts) if data_parts else np.empty(0, dtype=np.float64)
    indices = np.concatenate(index_parts) if index_parts else np.empty(0, dtype=np.int32)
    result = sparse.csr_matrix((data, indices, indptr), shape=csr.shape)
    result.eliminate_zeros()
    return result


def coupling_to_transfer_kernel(
    gamma: np.ndarray,
    source: np.ndarray,
    topk: int,
    mass_floor_rel: float,
) -> sparse.csr_matrix:
    source = np.asarray(source, dtype=np.float64)
    max_source = float(source.max(initial=0.0))
    floor = mass_floor_rel * max(1.0, max_source)
    valid = source > floor

    kernel = np.zeros_like(gamma, dtype=np.float64)
    np.divide(gamma, source[:, None], out=kernel, where=valid[:, None])
    kernel[~valid, :] = 0.0
    return prune_csr_rows_preserve_mass(sparse.csr_matrix(kernel), topk=topk)


def kernel_diagnostics(
    kernel: sparse.spmatrix,
    source: np.ndarray,
    coords: np.ndarray,
    topk: int,
) -> KernelDiagnostics:
    """Measure mass retention and conditional diffusion of one sparse kernel.

    All row-level quantities are weighted by actual transported brightness
    source[i] * row_sum[i]. Entropy is normalized by log(min(topk, n)), because
    every row has been truncated to at most topk entries.
    """
    csr = kernel.tocsr()
    source = np.asarray(source, dtype=np.float64)
    source_mass = float(source.sum())
    if source_mass <= 0:
        raise ValueError("Source image has nonpositive mass.")

    max_support = max(1, min(topk, csr.shape[1]))
    entropy_denominator = math.log(max_support) if max_support > 1 else 1.0

    transported_mass = 0.0
    entropy_weighted = 0.0
    effective_support_weighted = 0.0
    spread_sq_weighted = 0.0
    displacement_sq_weighted = 0.0

    for row in range(csr.shape[0]):
        if source[row] <= 0:
            continue
        start, end = csr.indptr[row], csr.indptr[row + 1]
        values = csr.data[start:end]
        columns = csr.indices[start:end]
        if values.size == 0:
            continue
        row_sum = float(values.sum())
        if row_sum <= 0:
            continue

        transported = float(source[row] * row_sum)
        probabilities = values / row_sum
        positive = probabilities > 0
        probabilities = probabilities[positive]
        destinations = columns[positive]

        entropy = float(-np.sum(probabilities * np.log(probabilities)))
        effective_support = float(math.exp(entropy))

        destination_coords = coords[destinations]
        barycenter = np.sum(probabilities[:, None] * destination_coords, axis=0)
        centered = destination_coords - barycenter
        row_spread_sq = float(np.sum(probabilities * np.sum(centered * centered, axis=1)))

        source_coord = coords[row]
        displacement = destination_coords - source_coord
        row_displacement_sq = float(
            np.sum(probabilities * np.sum(displacement * displacement, axis=1))
        )

        transported_mass += transported
        entropy_weighted += transported * entropy
        effective_support_weighted += transported * effective_support
        spread_sq_weighted += transported * row_spread_sq
        displacement_sq_weighted += transported * row_displacement_sq

    if transported_mass <= np.finfo(np.float64).tiny:
        return KernelDiagnostics(
            output_mass_ratio=0.0,
            entropy_fraction=0.0,
            effective_support=0.0,
            support_saturation=0.0,
            conditional_spread_rms=0.0,
            displacement_rms=0.0,
        )

    mean_entropy = entropy_weighted / transported_mass
    mean_effective_support = effective_support_weighted / transported_mass
    return KernelDiagnostics(
        output_mass_ratio=transported_mass / source_mass,
        entropy_fraction=float(mean_entropy / entropy_denominator),
        effective_support=float(mean_effective_support),
        support_saturation=float(mean_effective_support / max_support),
        conditional_spread_rms=float(math.sqrt(max(0.0, spread_sq_weighted / transported_mass))),
        displacement_rms=float(
            math.sqrt(max(0.0, displacement_sq_weighted / transported_mass))
        ),
    )


def moving_block_bootstrap_summary(
    values: Sequence[float],
    *,
    lag: int,
    samples: int,
    confidence: float,
    seed: int,
    block_length_override: int,
) -> BootstrapSummary:
    """Mean/std and a circular moving-block bootstrap interval.

    Adjacent lag windows overlap, so iid bootstrap intervals are too optimistic.
    By default, block length is min(lag, n_windows), reflecting the overlap scale.
    The interval remains heuristic for very short sequences, which is why the
    number of windows is always reported separately.
    """
    array = np.asarray(values, dtype=np.float64)
    n = int(array.size)
    if n == 0:
        raise ValueError("Cannot summarize an empty set of windows.")
    mean = float(np.mean(array))
    std = float(np.std(array, ddof=1)) if n > 1 else 0.0
    max_block_with_replication = max(1, n // 2) if n > 1 else 1
    requested_block = block_length_override if block_length_override > 0 else lag
    block_length = min(n, max_block_with_replication, max(1, requested_block))

    if samples <= 0 or n == 1:
        return BootstrapSummary(
            mean=mean,
            std=std,
            ci_low=None,
            ci_high=None,
            ci_width=None,
            ci_available=False,
            n=n,
            block_length=block_length,
        )

    rng = np.random.default_rng(seed)
    n_blocks = int(math.ceil(n / block_length))
    bootstrap_means = np.empty(samples, dtype=np.float64)
    offsets = np.arange(block_length, dtype=np.int64)

    for sample in range(samples):
        starts = rng.integers(0, n, size=n_blocks)
        indices = ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]
        bootstrap_means[sample] = float(np.mean(array[indices]))

    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(bootstrap_means, [alpha, 1.0 - alpha])
    return BootstrapSummary(
        mean=mean,
        std=std,
        ci_low=float(low),
        ci_high=float(high),
        ci_width=float(high - low),
        ci_available=True,
        n=n,
        block_length=block_length,
    )


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return math.nan, math.nan
    return float(np.mean(array)), float(np.std(array, ddof=1)) if array.size > 1 else 0.0


def summarize_lag_diagnostics(
    lag: int,
    windows: Sequence[WindowDiagnostics],
    config: EvaluatorConfig,
    seed_offset: int = 0,
) -> LagDiagnosticsSummary:
    raw_error = moving_block_bootstrap_summary(
        [window.error for window in windows],
        lag=lag,
        samples=config.bootstrap_samples,
        confidence=config.bootstrap_confidence,
        seed=config.bootstrap_seed + seed_offset,
        block_length_override=config.bootstrap_block_length,
    )

    direct_mass = [window.direct.output_mass_ratio for window in windows]
    path_mass = [window.path.output_mass_ratio for window in windows]
    path_direct = [window.path_to_direct_mass_ratio for window in windows]
    direct_mass_mean, direct_mass_std = _mean_std(direct_mass)
    path_mass_mean, path_mass_std = _mean_std(path_mass)
    path_direct_mean, path_direct_std = _mean_std(path_direct)

    return LagDiagnosticsSummary(
        lag=lag,
        n_windows=len(windows),
        raw_error=raw_error,
        direct_mass_ratio_mean=direct_mass_mean,
        direct_mass_ratio_std=direct_mass_std,
        path_mass_ratio_mean=path_mass_mean,
        path_mass_ratio_std=path_mass_std,
        path_to_direct_mass_ratio_mean=path_direct_mean,
        path_to_direct_mass_ratio_std=path_direct_std,
        direct_entropy_fraction_mean=float(
            np.mean([window.direct.entropy_fraction for window in windows])
        ),
        path_entropy_fraction_mean=float(
            np.mean([window.path.entropy_fraction for window in windows])
        ),
        direct_effective_support_mean=float(
            np.mean([window.direct.effective_support for window in windows])
        ),
        path_effective_support_mean=float(
            np.mean([window.path.effective_support for window in windows])
        ),
        direct_support_saturation_mean=float(
            np.mean([window.direct.support_saturation for window in windows])
        ),
        path_support_saturation_mean=float(
            np.mean([window.path.support_saturation for window in windows])
        ),
        direct_spread_rms_mean=float(
            np.mean([window.direct.conditional_spread_rms for window in windows])
        ),
        path_spread_rms_mean=float(
            np.mean([window.path.conditional_spread_rms for window in windows])
        ),
        direct_displacement_rms_mean=float(
            np.mean([window.direct.displacement_rms for window in windows])
        ),
        path_displacement_rms_mean=float(
            np.mean([window.path.displacement_rms for window in windows])
        ),
    )


class PairwiseKernelEvaluator:
    def __init__(self, frames: np.ndarray, config: EvaluatorConfig, label: str):
        self.frames = np.asarray(frames, dtype=np.float64)
        self.config = config
        self.label = label
        self.n_frames = self.frames.shape[0]
        self.flat = self.frames.reshape(self.n_frames, -1)
        self.coords = build_grid_coordinates(config.n)
        self.cost = build_cost_matrix(config.n)
        self.cache: dict[tuple[int, int], sparse.csr_matrix] = {}
        self.solve_logs: dict[str, dict[str, Any]] = {}

    def kernel(self, start: int, end: int) -> sparse.csr_matrix:
        if not (0 <= start < end < self.n_frames):
            raise IndexError(f"Invalid frame pair ({start}, {end}) for {self.n_frames} frames.")
        key = (start, end)
        if key in self.cache:
            return self.cache[key]

        a = self.flat[start]
        b = self.flat[end]
        gamma, info = solve_uot_coupling(a, b, self.cost, self.config)
        kernel = coupling_to_transfer_kernel(
            gamma=gamma,
            source=a,
            topk=self.config.kernel_topk,
            mass_floor_rel=self.config.mass_floor_rel,
        )
        self.cache[key] = kernel
        self.solve_logs[f"{start}->{end}"] = info
        return kernel

    def path_kernel(self, start: int, lag: int) -> sparse.csr_matrix:
        product = self.kernel(start, start + 1)
        for frame in range(start + 1, start + lag):
            product = product @ self.kernel(frame, frame + 1)
            product = prune_csr_rows_preserve_mass(product, self.config.kernel_topk)
        return product

    def window_diagnostics(self, start: int, lag: int) -> WindowDiagnostics:
        direct = self.kernel(start, start + lag)
        path = self.path_kernel(start, lag)
        source = self.flat[start]

        difference = direct - path
        abs_difference_rows = np.asarray(abs(difference).sum(axis=1)).ravel()
        direct_rows = np.asarray(direct.sum(axis=1)).ravel()
        path_rows = np.asarray(path.sum(axis=1)).ravel()
        numerator = float(np.dot(source, abs_difference_rows))
        denominator = float(np.dot(source, direct_rows + path_rows))
        error = numerator / max(denominator, np.finfo(np.float64).tiny)

        direct_diag = kernel_diagnostics(
            direct, source, self.coords, topk=self.config.kernel_topk
        )
        path_diag = kernel_diagnostics(
            path, source, self.coords, topk=self.config.kernel_topk
        )
        ratio = path_diag.output_mass_ratio / max(
            direct_diag.output_mass_ratio, np.finfo(np.float64).tiny
        )

        return WindowDiagnostics(
            start=start,
            lag=lag,
            error=error,
            direct=direct_diag,
            path=path_diag,
            path_to_direct_mass_ratio=float(ratio),
            entropy_excess=float(path_diag.entropy_fraction - direct_diag.entropy_fraction),
            support_excess=float(path_diag.effective_support - direct_diag.effective_support),
            spread_excess=float(
                path_diag.conditional_spread_rms - direct_diag.conditional_spread_rms
            ),
        )


def evaluate_sequence(
    frames: np.ndarray,
    label: str,
    config: EvaluatorConfig,
    verbose: bool = True,
) -> SequenceConsistency:
    start_time = time.perf_counter()
    evaluator = PairwiseKernelEvaluator(frames, config=config, label=label)
    windows_by_lag: dict[int, list[float]] = {}
    diagnostics_by_lag: dict[int, list[WindowDiagnostics]] = {}
    mean_by_lag: dict[int, float] = {}
    summaries_by_lag: dict[int, LagDiagnosticsSummary] = {}

    for lag_index, lag in enumerate(config.lags):
        diagnostics: list[WindowDiagnostics] = []
        n_windows = frames.shape[0] - lag
        for start in range(n_windows):
            if verbose:
                print(
                    f"[{label}] lag={lag}, window={start}->{start + lag}",
                    file=sys.stderr,
                    flush=True,
                )
            diagnostics.append(evaluator.window_diagnostics(start, lag))

        values = [window.error for window in diagnostics]
        diagnostics_by_lag[lag] = diagnostics
        windows_by_lag[lag] = values
        mean_by_lag[lag] = float(np.mean(values))
        summaries_by_lag[lag] = summarize_lag_diagnostics(
            lag, diagnostics, config, seed_offset=1009 * lag_index
        )

    return SequenceConsistency(
        label=label,
        mean_by_lag=mean_by_lag,
        windows_by_lag=windows_by_lag,
        diagnostics_by_lag=diagnostics_by_lag,
        summaries_by_lag=summaries_by_lag,
        solve_count=len(evaluator.cache),
        elapsed_seconds=time.perf_counter() - start_time,
    )


def lag_usability(
    lag: int,
    reference_summary: LagDiagnosticsSummary,
    candidate_summary: LagDiagnosticsSummary,
    g_uncertainty: BootstrapSummary,
    config: EvaluatorConfig,
) -> tuple[bool, list[str]]:
    """Heuristic flag only; all underlying diagnostics remain available."""
    reasons: list[str] = []
    n_windows = min(reference_summary.n_windows, candidate_summary.n_windows)
    if n_windows < config.min_windows:
        reasons.append(f"few_windows({n_windows}<{config.min_windows})")

    for name, summary in (("reference", reference_summary), ("candidate", candidate_summary)):
        if summary.path_mass_ratio_mean < config.min_path_mass_ratio:
            reasons.append(
                f"{name}_path_mass_collapse({summary.path_mass_ratio_mean:.3g}<"
                f"{config.min_path_mass_ratio:.3g})"
            )
        ratio = summary.path_to_direct_mass_ratio_mean
        if ratio < config.min_path_to_direct_mass_ratio:
            reasons.append(
                f"{name}_path_vs_direct_mass_low({ratio:.3g}<"
                f"{config.min_path_to_direct_mass_ratio:.3g})"
            )
        if ratio > config.max_path_to_direct_mass_ratio:
            reasons.append(
                f"{name}_path_vs_direct_mass_high({ratio:.3g}>"
                f"{config.max_path_to_direct_mass_ratio:.3g})"
            )
        entropy_excess = (
            summary.path_entropy_fraction_mean - summary.direct_entropy_fraction_mean
        )
        if entropy_excess > config.max_path_entropy_excess:
            reasons.append(
                f"{name}_composition_diffusion(dH={entropy_excess:.3g}>"
                f"{config.max_path_entropy_excess:.3g})"
            )

    if not g_uncertainty.ci_available:
        reasons.append("G_CI_unavailable")
    elif g_uncertainty.ci_width is not None and g_uncertainty.ci_width > config.max_g_ci_width:
        reasons.append(
            f"wide_G_CI({g_uncertainty.ci_width:.3g}>{config.max_g_ci_width:.3g})"
        )
    return len(reasons) == 0, reasons


def compare_to_reference(
    reference: SequenceConsistency,
    candidate: SequenceConsistency,
    config: EvaluatorConfig,
) -> CandidateComparison:
    g_by_lag: dict[int, float] = {}
    g_windows_by_lag: dict[int, list[float]] = {}
    g_uncertainty_by_lag: dict[int, BootstrapSummary] = {}
    delta_mean_by_lag: dict[int, float] = {}
    usable_by_lag: dict[int, bool] = {}
    usability_reasons_by_lag: dict[int, list[str]] = {}

    for lag_index, (lag, reference_windows) in enumerate(reference.windows_by_lag.items()):
        candidate_windows = candidate.windows_by_lag[lag]
        if len(reference_windows) != len(candidate_windows):
            raise ValueError(f"Window count differs for lag {lag}.")

        gaps = np.abs(np.asarray(candidate_windows) - np.asarray(reference_windows))
        gap_values = [float(value) for value in gaps]
        uncertainty = moving_block_bootstrap_summary(
            gap_values,
            lag=lag,
            samples=config.bootstrap_samples,
            confidence=config.bootstrap_confidence,
            seed=config.bootstrap_seed + 100_003 + 1013 * lag_index,
            block_length_override=config.bootstrap_block_length,
        )
        g_windows_by_lag[lag] = gap_values
        g_by_lag[lag] = uncertainty.mean
        g_uncertainty_by_lag[lag] = uncertainty
        delta_mean_by_lag[lag] = candidate.mean_by_lag[lag] - reference.mean_by_lag[lag]
        usable, reasons = lag_usability(
            lag,
            reference.summaries_by_lag[lag],
            candidate.summaries_by_lag[lag],
            uncertainty,
            config,
        )
        usable_by_lag[lag] = usable
        usability_reasons_by_lag[lag] = reasons

    return CandidateComparison(
        label=candidate.label,
        g_by_lag=g_by_lag,
        g_windows_by_lag=g_windows_by_lag,
        g_uncertainty_by_lag=g_uncertainty_by_lag,
        raw_mean_by_lag=candidate.mean_by_lag,
        raw_windows_by_lag=candidate.windows_by_lag,
        delta_mean_by_lag=delta_mean_by_lag,
        diagnostics_by_lag=candidate.summaries_by_lag,
        usable_by_lag=usable_by_lag,
        usability_reasons_by_lag=usability_reasons_by_lag,
        solve_count=candidate.solve_count,
        elapsed_seconds=candidate.elapsed_seconds,
    )


def parse_labeled_sequence(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "Sequence arguments must have the form LABEL=PATH (or LABEL=file.npz::key)."
        )
    label, spec = value.split("=", 1)
    label = label.strip()
    spec = spec.strip()
    if not label or not spec:
        raise argparse.ArgumentTypeError("Both LABEL and PATH must be nonempty.")
    return label, spec


def normalize_sequences(
    reference: np.ndarray,
    candidates: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, np.ndarray], float]:
    reference_masses = reference.reshape(reference.shape[0], -1).sum(axis=1)
    scale = float(np.mean(reference_masses))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Reference sequence has nonpositive mean total brightness.")
    return reference / scale, {label: frames / scale for label, frames in candidates.items()}, scale


def write_json(
    path: Path,
    config: EvaluatorConfig,
    reference: SequenceConsistency,
    comparisons: Sequence[CandidateComparison],
    reference_scale: float,
) -> None:
    payload = {
        "metric": "multiscale_uot_path_consistency_gap_with_diagnostics",
        "version": __version__,
        "definition": {
            "window_error": (
                "source-weighted relative L1 difference between direct and composed "
                "UOT transfer kernels"
            ),
            "G_l": "mean_k abs(e_reconstruction[k,l] - e_reference[k,l])",
            "entropy_fraction": (
                "transported-mass-weighted conditional row entropy divided by "
                "log(min(kernel_topk, n_pixels))"
            ),
            "conditional_spread_rms": (
                "transported-mass-weighted RMS destination spread about each row barycenter "
                "on the [0,1]^2 grid"
            ),
            "output_mass_ratio": "kernel output mass divided by source image mass",
            "useful_flag": "heuristic diagnostic flag; it does not alter G_l",
        },
        "config": asdict(config),
        "reference_intensity_scale": reference_scale,
        "reference": asdict(reference),
        "candidates": [asdict(comparison) for comparison in comparisons],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(
    path: Path,
    reference: SequenceConsistency,
    comparisons: Sequence[CandidateComparison],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "sequence",
        "lag",
        "n_windows",
        "G_l",
        "G_ci_low",
        "G_ci_high",
        "G_ci_width",
        "raw_E_l",
        "raw_E_ci_low",
        "raw_E_ci_high",
        "reference_E_l",
        "raw_minus_reference",
        "direct_mass_ratio",
        "path_mass_ratio",
        "path_to_direct_mass_ratio",
        "direct_entropy_fraction",
        "path_entropy_fraction",
        "direct_effective_support",
        "path_effective_support",
        "direct_support_saturation",
        "path_support_saturation",
        "direct_spread_rms",
        "path_spread_rms",
        "direct_displacement_rms",
        "path_displacement_rms",
        "usable",
        "usability_reasons",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for comparison in comparisons:
            for lag in sorted(comparison.g_by_lag):
                summary = comparison.diagnostics_by_lag[lag]
                g_unc = comparison.g_uncertainty_by_lag[lag]
                writer.writerow(
                    {
                        "sequence": comparison.label,
                        "lag": lag,
                        "n_windows": summary.n_windows,
                        "G_l": comparison.g_by_lag[lag],
                        "G_ci_low": g_unc.ci_low,
                        "G_ci_high": g_unc.ci_high,
                        "G_ci_width": g_unc.ci_width,
                        "raw_E_l": comparison.raw_mean_by_lag[lag],
                        "raw_E_ci_low": summary.raw_error.ci_low,
                        "raw_E_ci_high": summary.raw_error.ci_high,
                        "reference_E_l": reference.mean_by_lag[lag],
                        "raw_minus_reference": comparison.delta_mean_by_lag[lag],
                        "direct_mass_ratio": summary.direct_mass_ratio_mean,
                        "path_mass_ratio": summary.path_mass_ratio_mean,
                        "path_to_direct_mass_ratio": summary.path_to_direct_mass_ratio_mean,
                        "direct_entropy_fraction": summary.direct_entropy_fraction_mean,
                        "path_entropy_fraction": summary.path_entropy_fraction_mean,
                        "direct_effective_support": summary.direct_effective_support_mean,
                        "path_effective_support": summary.path_effective_support_mean,
                        "direct_support_saturation": summary.direct_support_saturation_mean,
                        "path_support_saturation": summary.path_support_saturation_mean,
                        "direct_spread_rms": summary.direct_spread_rms_mean,
                        "path_spread_rms": summary.path_spread_rms_mean,
                        "direct_displacement_rms": summary.direct_displacement_rms_mean,
                        "path_displacement_rms": summary.path_displacement_rms_mean,
                        "usable": comparison.usable_by_lag[lag],
                        "usability_reasons": ";".join(
                            comparison.usability_reasons_by_lag[lag]
                        ),
                    }
                )


def write_window_csv_from_sequences(
    path: Path,
    reference: SequenceConsistency,
    candidates: Sequence[SequenceConsistency],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "sequence",
        "lag",
        "start",
        "end",
        "e_window",
        "reference_e_window",
        "absolute_gap_window",
        "direct_mass_ratio",
        "path_mass_ratio",
        "path_to_direct_mass_ratio",
        "direct_entropy_fraction",
        "path_entropy_fraction",
        "direct_effective_support",
        "path_effective_support",
        "direct_spread_rms",
        "path_spread_rms",
        "direct_displacement_rms",
        "path_displacement_rms",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for candidate in candidates:
            for lag, windows in candidate.diagnostics_by_lag.items():
                reference_windows = reference.diagnostics_by_lag[lag]
                for window, reference_window in zip(windows, reference_windows):
                    writer.writerow(
                        {
                            "sequence": candidate.label,
                            "lag": lag,
                            "start": window.start,
                            "end": window.start + lag,
                            "e_window": window.error,
                            "reference_e_window": reference_window.error,
                            "absolute_gap_window": abs(window.error - reference_window.error),
                            "direct_mass_ratio": window.direct.output_mass_ratio,
                            "path_mass_ratio": window.path.output_mass_ratio,
                            "path_to_direct_mass_ratio": window.path_to_direct_mass_ratio,
                            "direct_entropy_fraction": window.direct.entropy_fraction,
                            "path_entropy_fraction": window.path.entropy_fraction,
                            "direct_effective_support": window.direct.effective_support,
                            "path_effective_support": window.path.effective_support,
                            "direct_spread_rms": window.direct.conditional_spread_rms,
                            "path_spread_rms": window.path.conditional_spread_rms,
                            "direct_displacement_rms": window.direct.displacement_rms,
                            "path_displacement_rms": window.path.displacement_rms,
                        }
                    )


def make_plot(
    path: Path,
    comparisons: Sequence[CandidateComparison],
    lags: Sequence[int],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required for --plot.") from exc

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    for comparison in comparisons:
        values = np.asarray([comparison.g_by_lag[lag] for lag in lags])
        lower = np.asarray([
            comparison.g_uncertainty_by_lag[lag].ci_low
            if comparison.g_uncertainty_by_lag[lag].ci_low is not None
            else comparison.g_by_lag[lag]
            for lag in lags
        ])
        upper = np.asarray([
            comparison.g_uncertainty_by_lag[lag].ci_high
            if comparison.g_uncertainty_by_lag[lag].ci_high is not None
            else comparison.g_by_lag[lag]
            for lag in lags
        ])
        yerr = np.vstack([values - lower, upper - values])
        ax.errorbar(lags, values, yerr=yerr, marker="o", capsize=3, label=comparison.label)

        unusable = [lag for lag in lags if not comparison.usable_by_lag[lag]]
        if unusable:
            ax.scatter(
                unusable,
                [comparison.g_by_lag[lag] for lag in unusable],
                marker="x",
                s=55,
            )

    ax.set_xlabel("Lag $\\ell$ (frames)")
    ax.set_ylabel("Ground-truth consistency gap $G_\\ell$")
    ax.set_title("Multiscale UOT path-consistency fidelity")
    ax.set_xticks(list(lags))
    ax.grid(True, alpha=0.3)
    if comparisons:
        ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_diagnostic_plot(
    path: Path,
    reference: SequenceConsistency,
    comparisons: Sequence[CandidateComparison],
    lags: Sequence[int],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required for --diagnostic-plot.") from exc

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), sharex=True)
    ax_g, ax_mass, ax_entropy, ax_windows = axes.ravel()

    for comparison in comparisons:
        values = np.asarray([comparison.g_by_lag[lag] for lag in lags])
        lower = np.asarray([
            comparison.g_uncertainty_by_lag[lag].ci_low
            if comparison.g_uncertainty_by_lag[lag].ci_low is not None
            else comparison.g_by_lag[lag]
            for lag in lags
        ])
        upper = np.asarray([
            comparison.g_uncertainty_by_lag[lag].ci_high
            if comparison.g_uncertainty_by_lag[lag].ci_high is not None
            else comparison.g_by_lag[lag]
            for lag in lags
        ])
        ax_g.errorbar(
            lags,
            values,
            yerr=np.vstack([values - lower, upper - values]),
            marker="o",
            capsize=3,
            label=comparison.label,
        )
        ax_mass.plot(
            lags,
            [comparison.diagnostics_by_lag[lag].path_mass_ratio_mean for lag in lags],
            marker="o",
            label=comparison.label,
        )
        ax_entropy.plot(
            lags,
            [
                comparison.diagnostics_by_lag[lag].path_entropy_fraction_mean
                for lag in lags
            ],
            marker="o",
            label=comparison.label,
        )

    ax_mass.plot(
        lags,
        [reference.summaries_by_lag[lag].path_mass_ratio_mean for lag in lags],
        marker="o",
        linestyle="--",
        label=reference.label,
    )
    ax_entropy.plot(
        lags,
        [reference.summaries_by_lag[lag].path_entropy_fraction_mean for lag in lags],
        marker="o",
        linestyle="--",
        label=reference.label,
    )
    ax_windows.plot(
        lags,
        [reference.summaries_by_lag[lag].n_windows for lag in lags],
        marker="o",
    )

    ax_g.set_title("$G_\\ell$ with moving-block bootstrap CI")
    ax_g.set_ylabel("$G_\\ell$")
    ax_mass.set_title("Composed-path output mass")
    ax_mass.set_ylabel("Output/source mass ratio")
    ax_entropy.set_title("Composed-kernel conditional diffusion")
    ax_entropy.set_ylabel("Entropy fraction of top-k maximum")
    ax_windows.set_title("Available windows")
    ax_windows.set_ylabel("$K-\\ell$")

    for ax in axes.ravel():
        ax.set_xlabel("Lag $\\ell$")
        ax.set_xticks(list(lags))
        ax.grid(True, alpha=0.3)
    if comparisons:
        ax_g.legend()
        ax_mass.legend()
        ax_entropy.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _format_bool(value: bool) -> str:
    return "yes" if value else "NO"


def print_report(
    reference: SequenceConsistency,
    comparisons: Sequence[CandidateComparison],
    lags: Sequence[int],
) -> None:
    print("\nReference raw path-consistency and diagnostics:")
    for lag in lags:
        summary = reference.summaries_by_lag[lag]
        raw_ci = (
            f"[{summary.raw_error.ci_low:.6g},{summary.raw_error.ci_high:.6g}]"
            if summary.raw_error.ci_available
            else "n/a"
        )
        entropy_excess = (
            summary.path_entropy_fraction_mean - summary.direct_entropy_fraction_mean
        )
        print(
            f"  l={lag:2d}  windows={summary.n_windows:2d}  "
            f"E={summary.raw_error.mean:.6g} CI={raw_ci}  "
            f"path_mass={summary.path_mass_ratio_mean:.4g}  "
            f"path/direct={summary.path_to_direct_mass_ratio_mean:.4g}  "
            f"H_path={summary.path_entropy_fraction_mean:.4g} dH={entropy_excess:+.3g}  "
            f"support={summary.path_effective_support_mean:.3g}  "
            f"spread={summary.path_spread_rms_mean:.4g}"
        )

    smallest_lag = min(lags)
    smallest = reference.summaries_by_lag[smallest_lag]
    if (
        smallest.direct_support_saturation_mean > 0.90
        or smallest.path_support_saturation_mean > 0.90
    ):
        print(
            "\nWARNING: kernel effective support already saturates the retained top-k "
            f"destinations at lag {smallest_lag}. Diffusion diagnostics may be "
            "top-k limited; consider increasing --kernel-topk or decreasing "
            "--entropic-reg before interpreting entropy trends."
        )

    print("\nCandidate G_l and lag-usability diagnostics:")
    for comparison in comparisons:
        print(f"\n[{comparison.label}]")
        for lag in lags:
            g = comparison.g_uncertainty_by_lag[lag]
            summary = comparison.diagnostics_by_lag[lag]
            reasons = comparison.usability_reasons_by_lag[lag]
            reason_text = "ok" if not reasons else ", ".join(reasons)
            g_ci = (
                f"[{g.ci_low:.6g},{g.ci_high:.6g}]" if g.ci_available else "n/a"
            )
            entropy_excess = (
                summary.path_entropy_fraction_mean - summary.direct_entropy_fraction_mean
            )
            print(
                f"  l={lag:2d}  G={g.mean:.6g} CI={g_ci}  "
                f"windows={g.n:2d}  "
                f"path_mass={summary.path_mass_ratio_mean:.4g}  "
                f"path/direct={summary.path_to_direct_mass_ratio_mean:.4g}  "
                f"H_path={summary.path_entropy_fraction_mean:.4g} dH={entropy_excess:+.3g}  "
                f"spread={summary.path_spread_rms_mean:.4g}  "
                f"usable={_format_bool(comparison.usable_by_lag[lag])} ({reason_text})"
            )

    globally_usable = [
        lag
        for lag in lags
        if all(comparison.usable_by_lag[lag] for comparison in comparisons)
    ]
    if globally_usable:
        contiguous: list[int] = []
        for lag in sorted(lags):
            if lag in globally_usable and (not contiguous or lag == contiguous[-1] + 1):
                contiguous.append(lag)
            elif contiguous:
                break
        if contiguous:
            if len(contiguous) == 1:
                text = str(contiguous[0])
            else:
                text = f"{contiguous[0]}-{contiguous[-1]}"
            print(f"\nHeuristic common useful lag range: {text}")
    else:
        print("\nNo lag passes the heuristic criteria for every candidate.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute multiscale UOT path-consistency gaps G_l and lag diagnostics "
            "for one reference and any number of candidate image sequences."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--reference", required=True, help="Reference sequence path/spec.")
    parser.add_argument("--reference-label", default="ground_truth")
    parser.add_argument(
        "--sequence",
        action="append",
        type=parse_labeled_sequence,
        required=True,
        metavar="LABEL=PATH",
        help="Candidate sequence; repeat for multiple sequences.",
    )
    parser.add_argument("--n", type=int, default=32, help="Evaluation side length (>=32).")
    parser.add_argument("--lags", type=int, nargs="+", default=[2, 3, 4, 5])
    parser.add_argument("--entropic-reg", type=float, default=0.03)
    parser.add_argument("--marginal-reg", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--stop-thr", type=float, default=1e-8)
    parser.add_argument("--solver", choices=["auto", "pot", "numpy"], default="auto")
    parser.add_argument(
        "--pot-method",
        choices=[
            "sinkhorn",
            "sinkhorn_stabilized",
            "sinkhorn_translation_invariant",
            "sinkhorn_reg_scaling",
        ],
        default="sinkhorn_stabilized",
    )
    parser.add_argument(
        "--kernel-topk",
        type=int,
        default=64,
        help="Largest destinations retained per row; row mass is preserved.",
    )

    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-confidence", type=float, default=0.95)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument(
        "--bootstrap-block-length",
        type=int,
        default=0,
        help="0 chooses min(lag, number of windows).",
    )

    parser.add_argument("--min-windows", type=int, default=5)
    parser.add_argument("--min-path-mass-ratio", type=float, default=0.05)
    parser.add_argument("--min-path-to-direct-mass-ratio", type=float, default=0.10)
    parser.add_argument("--max-path-to-direct-mass-ratio", type=float, default=10.0)
    parser.add_argument(
        "--max-path-entropy-excess",
        type=float,
        default=0.15,
        help="Maximum allowed path-minus-direct normalized entropy for useful-lag flag.",
    )
    parser.add_argument("--max-g-ci-width", type=float, default=0.25)

    parser.add_argument("--json", type=Path, help="Optional detailed JSON output.")
    parser.add_argument("--csv", type=Path, help="Optional per-lag summary CSV.")
    parser.add_argument("--window-csv", type=Path, help="Optional per-window diagnostic CSV.")
    parser.add_argument("--plot", type=Path, help="Optional G_l plot with bootstrap CIs.")
    parser.add_argument(
        "--diagnostic-plot",
        type=Path,
        help="Optional four-panel plot for G_l, mass, entropy, and window count.",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = EvaluatorConfig(
        n=args.n,
        lags=tuple(sorted(set(args.lags))),
        entropic_reg=args.entropic_reg,
        marginal_reg=args.marginal_reg,
        max_iter=args.max_iter,
        stop_thr=args.stop_thr,
        solver=args.solver,
        pot_method=args.pot_method,
        kernel_topk=args.kernel_topk,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_confidence=args.bootstrap_confidence,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_block_length=args.bootstrap_block_length,
        min_windows=args.min_windows,
        min_path_mass_ratio=args.min_path_mass_ratio,
        min_path_to_direct_mass_ratio=args.min_path_to_direct_mass_ratio,
        max_path_to_direct_mass_ratio=args.max_path_to_direct_mass_ratio,
        max_path_entropy_excess=args.max_path_entropy_excess,
        max_g_ci_width=args.max_g_ci_width,
    )
    config.validate()

    labels = [label for label, _ in args.sequence]
    if len(set(labels)) != len(labels):
        raise ValueError("Candidate sequence labels must be unique.")

    print(f"Loading reference: {args.reference}", file=sys.stderr)
    reference_frames = load_sequence(args.reference, n=config.n)
    candidate_frames: dict[str, np.ndarray] = {}
    for label, spec in args.sequence:
        print(f"Loading {label}: {spec}", file=sys.stderr)
        candidate_frames[label] = load_sequence(spec, n=config.n)

    n_frames = reference_frames.shape[0]
    max_lag = max(config.lags)
    if n_frames <= max_lag:
        raise ValueError(
            f"Need at least {max_lag + 1} frames for lag {max_lag}; reference has {n_frames}."
        )
    for label, frames in candidate_frames.items():
        if frames.shape[0] != n_frames:
            raise ValueError(
                f"Sequence {label!r} has {frames.shape[0]} frames; reference has {n_frames}."
            )

    reference_frames, candidate_frames, reference_scale = normalize_sequences(
        reference_frames, candidate_frames
    )

    selected_solver = config.solver
    if selected_solver == "auto":
        selected_solver = "pot" if ot is not None else "numpy"
    print(
        f"Evaluator: N={config.n}, solver={selected_solver}, eps={config.entropic_reg}, "
        f"tau={config.marginal_reg}, topk={config.kernel_topk}, lags={config.lags}",
        file=sys.stderr,
    )

    reference_result = evaluate_sequence(
        reference_frames,
        label=args.reference_label,
        config=config,
        verbose=not args.quiet,
    )

    candidate_results: list[SequenceConsistency] = []
    comparisons: list[CandidateComparison] = []
    for label, frames in candidate_frames.items():
        result = evaluate_sequence(frames, label=label, config=config, verbose=not args.quiet)
        candidate_results.append(result)
        comparisons.append(compare_to_reference(reference_result, result, config))

    print_report(reference_result, comparisons, config.lags)

    if args.json:
        write_json(args.json, config, reference_result, comparisons, reference_scale)
        print(f"Wrote JSON: {args.json}", file=sys.stderr)
    if args.csv:
        write_csv(args.csv, reference_result, comparisons)
        print(f"Wrote CSV: {args.csv}", file=sys.stderr)
    if args.window_csv:
        write_window_csv_from_sequences(args.window_csv, reference_result, candidate_results)
        print(f"Wrote window CSV: {args.window_csv}", file=sys.stderr)
    if args.plot:
        make_plot(args.plot, comparisons, config.lags)
        print(f"Wrote plot: {args.plot}", file=sys.stderr)
    if args.diagnostic_plot:
        make_diagnostic_plot(
            args.diagnostic_plot, reference_result, comparisons, config.lags
        )
        print(f"Wrote diagnostic plot: {args.diagnostic_plot}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        ValueError,
        RuntimeError,
        FileNotFoundError,
        KeyError,
        ImportError,
        FloatingPointError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
