#!/usr/bin/env python3
"""Multiscale UOT path-consistency metric for dynamic image sequences.

This file is intentionally independent of any reconstruction/post-processing OT
solver.  It evaluates one reference sequence and any number of candidate image
sequences with a fixed, pairwise, entropically regularized unbalanced optimal
transport (UOT) evaluator.

For nonnegative flattened images a, b in R_+^n, the evaluator solves

    min_{Gamma >= 0}
        <C, Gamma>
        + eps * KL(Gamma || a b^T)
        + tau * KL(Gamma 1 || a)
        + tau * KL(Gamma^T 1 || b),

where C is the squared spatial ground-cost matrix.  The two marginal KL terms
allow transported source and target masses to differ from the input masses.
The entropic/KL term makes the problem solvable by generalized Sinkhorn scaling.

The coupling is converted to a source-conditioned transfer kernel

    K_ij = Gamma_ij / a_i.

A row K[i, :] therefore describes the destination brightness per unit source
brightness originating at pixel i.  Its row sum is not constrained to one, so
attenuation/amplification is retained during kernel composition.

For lag ell, compare

    D_{k,ell} = K_{k -> k+ell}

against the chronological adjacent-path composition

    P_{k,ell} = K_{k -> k+1} ... K_{k+ell-1 -> k+ell}.

The window defect is the source-weighted, relative L1 difference

                  sum_i I_k[i] ||D[i,:] - P[i,:]||_1
    e_{k,ell} = --------------------------------------------------- .
                  sum_i I_k[i](||D[i,:]||_1+||P[i,:]||_1) + tiny

For a candidate reconstruction S and ground truth GT, the reported score is

    G_ell(S) = mean_k |e_{k,ell}(S) - e_{k,ell}(GT)|.

Lower G_ell means that the candidate reproduces the reference sequence's degree
and timing of multistep UOT path inconsistency at temporal scale ell.  It does
not claim that the inferred UOT plans are physical plasma flow fields.

Dependencies
------------
Required: numpy, scipy, Pillow
Optional: POT (Python Optimal Transport, imported as ``ot``), matplotlib

POT is used when available (``--solver auto``).  A mathematically equivalent
NumPy generalized-Sinkhorn implementation is included as a fallback, so this
file remains runnable without POT.

Examples
--------
One reconstruction:

    python uot_path_consistency.py \
        --reference /path/to/ground_truth \
        --sequence post_uot=/path/to/post_uot \
        --json results.json --csv results.csv --plot G_l.png

Several reconstructions:

python uot_path_consistency.py \
    --reference ../../blackhole_sim/data/aart_frames \
    --sequence starwarps=../../results/starwarps_results/34_telescopes/final_frames_34_png \
    --sequence static=../../results/static_reconstruction_results/reconstructed_frames_gray \
    --sequence uot_SW=../../results/optimal_transport_results/ot_uot_SW_34/frames \
    --sequence uot_static=../../results/optimal_transport_results/ot_uot_static_34/frames \
    --n 32 \
    --lags 2 3 4 5 \
    --json uot_consistency_results.json \
    --csv uot_consistency_results.csv \
    --plot uot_consistency_gap.png

Input formats
-------------
* Directory containing image files (natural filename order)
* .npy array with shape (K,H,W) or (K,H,W,C)
* .npz containing exactly one array, or ``file.npz::array_key``

All sequences must have the same number of frames.  Frames are resized to N x N
with area/box resampling.  N defaults to 32 and may not be set below 32 through
the CLI.
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
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image
from scipy import sparse

try:  # Optional dependency.
    import ot  # type: ignore
except Exception:  # pragma: no cover - depends on the user's environment.
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
    marginal_reg: float = 1.0
    max_iter: int = 1000
    stop_thr: float = 1e-8
    solver: str = "auto"
    pot_method: str = "sinkhorn_stabilized"
    kernel_topk: int = 256
    mass_floor_rel: float = 1e-12
    numerical_floor: float = 1e-300
    dtype: str = "float64"

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


@dataclass
class SequenceConsistency:
    label: str
    mean_by_lag: dict[int, float]
    windows_by_lag: dict[int, list[float]]
    solve_count: int
    elapsed_seconds: float


@dataclass
class CandidateComparison:
    label: str
    g_by_lag: dict[int, float]
    raw_mean_by_lag: dict[int, float]
    raw_windows_by_lag: dict[int, list[float]]
    delta_mean_by_lag: dict[int, float]
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
        # Standard luminance weights. Ignore alpha and any channels after RGB.
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
            # Keep the native numeric depth where Pillow supports it.
            array = np.asarray(image)
    except Exception as exc:
        raise RuntimeError(f"Failed to read image {path}: {exc}") from exc
    return _to_grayscale_float(array)


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
        array = np.load(path, allow_pickle=False)
        frames = _frames_from_array(array, source=str(path))
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

    # Numerical negative values are clipped; substantial negatives are rejected.
    min_value = float(resized.min())
    max_abs = float(np.max(np.abs(resized)))
    tolerance = 1e-10 * max(1.0, max_abs)
    if min_value < -tolerance:
        raise ValueError(
            f"Sequence {spec!r} contains significant negative intensities "
            f"(minimum {min_value:.6g}). UOT requires nonnegative images."
        )
    return np.maximum(resized, 0.0)


def _frames_from_array(array: np.ndarray, source: str) -> list[np.ndarray]:
    array = np.asarray(array)
    if array.ndim == 3:  # (K,H,W)
        return [array[k] for k in range(array.shape[0])]
    if array.ndim == 4:  # (K,H,W,C)
        return [array[k] for k in range(array.shape[0])]
    raise ValueError(
        f"Array {source} must have shape (K,H,W) or (K,H,W,C); got {array.shape}."
    )


def build_cost_matrix(n: int, dtype: np.dtype = np.float64) -> np.ndarray:
    """Squared Euclidean cost on [0,1]^2, normalized so max cost is 1."""
    axis = np.linspace(0.0, 1.0, n, dtype=np.float64)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    coords = np.column_stack([yy.ravel(), xx.ravel()])
    sq_norm = np.sum(coords * coords, axis=1)
    cost = sq_norm[:, None] + sq_norm[None, :] - 2.0 * coords @ coords.T
    np.maximum(cost, 0.0, out=cost)
    cost /= 2.0  # Max squared corner distance on [0,1]^2 is 2.
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
    """Solve KL-relaxed entropic UOT by generalized Sinkhorn scaling.

    The objective is

        <C,Gamma> + eps KL(Gamma || a b^T)
        + tau KL(Gamma 1 || a) + tau KL(Gamma^T 1 || b).

    Let Q = (a b^T) * exp(-C/eps).  The solution has the scaling form

        Gamma = diag(u) Q diag(v),

    with fixed-point updates

        u = (a / (Q v))^(tau/(tau+eps)),
        v = (b / (Q^T u))^(tau/(tau+eps)).
    """
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
        info = {
            "backend": "pot",
            "method": config.pot_method,
            "iterations": log.get("niter", log.get("it", None)),
            "error": _last_log_error(log),
        }
        return gamma, info

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
    np.divide(
        gamma,
        source[:, None],
        out=kernel,
        where=valid[:, None],
    )
    kernel[~valid, :] = 0.0
    return prune_csr_rows_preserve_mass(sparse.csr_matrix(kernel), topk=topk)


class PairwiseKernelEvaluator:
    def __init__(self, frames: np.ndarray, config: EvaluatorConfig, label: str):
        self.frames = np.asarray(frames, dtype=np.float64)
        self.config = config
        self.label = label
        self.n_frames = self.frames.shape[0]
        self.flat = self.frames.reshape(self.n_frames, -1)
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

    def window_error(self, start: int, lag: int) -> float:
        direct = self.kernel(start, start + lag)
        path = self.path_kernel(start, lag)
        source = self.flat[start]

        difference = direct - path
        abs_difference_rows = np.asarray(abs(difference).sum(axis=1)).ravel()
        direct_rows = np.asarray(direct.sum(axis=1)).ravel()
        path_rows = np.asarray(path.sum(axis=1)).ravel()

        numerator = float(np.dot(source, abs_difference_rows))
        denominator = float(np.dot(source, direct_rows + path_rows))
        return numerator / max(denominator, np.finfo(np.float64).tiny)


def evaluate_sequence(
    frames: np.ndarray,
    label: str,
    config: EvaluatorConfig,
    verbose: bool = True,
) -> SequenceConsistency:
    start_time = time.perf_counter()
    evaluator = PairwiseKernelEvaluator(frames, config=config, label=label)
    windows_by_lag: dict[int, list[float]] = {}
    mean_by_lag: dict[int, float] = {}

    for lag in config.lags:
        values: list[float] = []
        n_windows = frames.shape[0] - lag
        for start in range(n_windows):
            if verbose:
                print(
                    f"[{label}] lag={lag}, window={start}->{start + lag}",
                    file=sys.stderr,
                    flush=True,
                )
            values.append(evaluator.window_error(start, lag))
        windows_by_lag[lag] = values
        mean_by_lag[lag] = float(np.mean(values))

    return SequenceConsistency(
        label=label,
        mean_by_lag=mean_by_lag,
        windows_by_lag=windows_by_lag,
        solve_count=len(evaluator.cache),
        elapsed_seconds=time.perf_counter() - start_time,
    )


def compare_to_reference(
    reference: SequenceConsistency,
    candidate: SequenceConsistency,
) -> CandidateComparison:
    g_by_lag: dict[int, float] = {}
    delta_mean_by_lag: dict[int, float] = {}
    for lag, reference_windows in reference.windows_by_lag.items():
        candidate_windows = candidate.windows_by_lag[lag]
        if len(reference_windows) != len(candidate_windows):
            raise ValueError(f"Window count differs for lag {lag}.")
        g_by_lag[lag] = float(
            np.mean(np.abs(np.asarray(candidate_windows) - np.asarray(reference_windows)))
        )
        delta_mean_by_lag[lag] = (
            candidate.mean_by_lag[lag] - reference.mean_by_lag[lag]
        )

    return CandidateComparison(
        label=candidate.label,
        g_by_lag=g_by_lag,
        raw_mean_by_lag=candidate.mean_by_lag,
        raw_windows_by_lag=candidate.windows_by_lag,
        delta_mean_by_lag=delta_mean_by_lag,
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
    normalized_reference = reference / scale
    normalized_candidates = {label: frames / scale for label, frames in candidates.items()}
    return normalized_reference, normalized_candidates, scale


def write_json(
    path: Path,
    config: EvaluatorConfig,
    reference: SequenceConsistency,
    comparisons: Sequence[CandidateComparison],
    reference_scale: float,
) -> None:
    payload = {
        "metric": "multiscale_uot_path_consistency_gap",
        "definition": {
            "window_error": "source-weighted relative L1 difference between direct and composed UOT transfer kernels",
            "G_l": "mean_k abs(e_reconstruction[k,l] - e_reference[k,l])",
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
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "sequence",
                "lag",
                "G_l",
                "raw_E_l",
                "reference_E_l",
                "raw_minus_reference",
            ]
        )
        for comparison in comparisons:
            for lag in sorted(comparison.g_by_lag):
                writer.writerow(
                    [
                        comparison.label,
                        lag,
                        comparison.g_by_lag[lag],
                        comparison.raw_mean_by_lag[lag],
                        reference.mean_by_lag[lag],
                        comparison.delta_mean_by_lag[lag],
                    ]
                )


def make_plot(
    path: Path,
    comparisons: Sequence[CandidateComparison],
    lags: Sequence[int],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency.
        raise RuntimeError("matplotlib is required for --plot.") from exc

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for comparison in comparisons:
        values = [comparison.g_by_lag[lag] for lag in lags]
        ax.plot(lags, values, marker="o", label=comparison.label)
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


def print_report(
    reference: SequenceConsistency,
    comparisons: Sequence[CandidateComparison],
    lags: Sequence[int],
) -> None:
    headers = ["sequence", *[f"G_{lag}" for lag in lags], *[f"E_{lag}" for lag in lags]]
    rows: list[list[str]] = []
    for comparison in comparisons:
        rows.append(
            [
                comparison.label,
                *[f"{comparison.g_by_lag[lag]:.8g}" for lag in lags],
                *[f"{comparison.raw_mean_by_lag[lag]:.8g}" for lag in lags],
            ]
        )

    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row)]

    def format_row(row: Sequence[str]) -> str:
        return "  ".join(cell.ljust(width) for cell, width in zip(row, widths))

    print("\nReference raw path-consistency E_l:")
    print("  ".join(f"E_{lag}={reference.mean_by_lag[lag]:.8g}" for lag in lags))
    print("\nCandidate results (lower G_l is closer to reference):")
    print(format_row(headers))
    print(format_row(["-" * width for width in widths]))
    for row in rows:
        print(format_row(row))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute multiscale UOT path-consistency gaps G_l for one reference "
            "and any number of candidate image sequences."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--reference", required=True, help="Reference sequence path/spec.")
    parser.add_argument(
        "--reference-label", default="ground_truth", help="Label used in reports."
    )
    parser.add_argument(
        "--sequence",
        action="append",
        type=parse_labeled_sequence,
        required=True,
        metavar="LABEL=PATH",
        help="Candidate sequence; repeat this option for multiple sequences.",
    )
    parser.add_argument("--n", type=int, default=32, help="Evaluation image side length (>=32).")
    parser.add_argument(
        "--lags", type=int, nargs="+", default=[2, 3, 4, 5], help="Temporal lags."
    )
    parser.add_argument(
        "--entropic-reg", type=float, default=0.03, help="Entropic/KL coupling regularization epsilon."
    )
    parser.add_argument(
        "--marginal-reg", type=float, default=1.0, help="KL marginal-relaxation weight tau."
    )
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--stop-thr", type=float, default=1e-8)
    parser.add_argument(
        "--solver", choices=["auto", "pot", "numpy"], default="auto"
    )
    parser.add_argument(
        "--pot-method",
        choices=[
            "sinkhorn",
            "sinkhorn_stabilized",
            "sinkhorn_translation_invariant",
            "sinkhorn_reg_scaling",
        ],
        default="sinkhorn_stabilized",
        help="POT method when POT is used.",
    )
    parser.add_argument(
        "--kernel-topk",
        type=int,
        default=64,
        help=(
            "Largest destinations retained per source row. Row mass is preserved; "
            "this makes 32x32 kernel composition tractable."
        ),
    )
    parser.add_argument("--json", type=Path, help="Optional JSON output path.")
    parser.add_argument("--csv", type=Path, help="Optional CSV summary path.")
    parser.add_argument("--plot", type=Path, help="Optional G_l line-plot output path.")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-window progress.")
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

    comparisons: list[CandidateComparison] = []
    for label, frames in candidate_frames.items():
        candidate_result = evaluate_sequence(
            frames, label=label, config=config, verbose=not args.quiet
        )
        comparisons.append(compare_to_reference(reference_result, candidate_result))

    print_report(reference_result, comparisons, config.lags)

    if args.json:
        write_json(args.json, config, reference_result, comparisons, reference_scale)
        print(f"Wrote JSON: {args.json}", file=sys.stderr)
    if args.csv:
        write_csv(args.csv, reference_result, comparisons)
        print(f"Wrote CSV: {args.csv}", file=sys.stderr)
    if args.plot:
        make_plot(args.plot, comparisons, config.lags)
        print(f"Wrote plot: {args.plot}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, FileNotFoundError, KeyError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
