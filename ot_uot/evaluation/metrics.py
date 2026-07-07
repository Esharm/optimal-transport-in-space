"""Evaluation metrics for dynamic image reconstruction."""

from __future__ import annotations

import numpy as np


def nrmse(estimate: np.ndarray, reference: np.ndarray) -> float:
    """Normalized root mean squared error."""

    estimate = np.asarray(estimate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    return float(np.linalg.norm(estimate - reference) / (np.linalg.norm(reference) + 1e-30))


def framewise_nrmse(estimate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Framewise NRMSE for sequences with shape ``(K,H,W)``."""

    return np.asarray([nrmse(a, b) for a, b in zip(estimate, reference)], dtype=np.float64)


def normalized_correlation(estimate: np.ndarray, reference: np.ndarray) -> float:
    """Cosine similarity between two arrays after mean removal."""

    a = np.asarray(estimate, dtype=np.float64).ravel()
    b = np.asarray(reference, dtype=np.float64).ravel()
    a = a - np.mean(a)
    b = b - np.mean(b)
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-30))


def ssim_global(estimate: np.ndarray, reference: np.ndarray, data_range: float | None = None) -> float:
    """Global SSIM approximation without optional image-processing dependencies."""

    x = np.asarray(estimate, dtype=np.float64)
    y = np.asarray(reference, dtype=np.float64)
    if data_range is None:
        data_range = float(max(np.max(y) - np.min(y), np.max(x) - np.min(x), 1e-30))
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mux = float(np.mean(x))
    muy = float(np.mean(y))
    varx = float(np.mean((x - mux) ** 2))
    vary = float(np.mean((y - muy) ** 2))
    cov = float(np.mean((x - mux) * (y - muy)))
    return float(((2 * mux * muy + c1) * (2 * cov + c2)) / ((mux * mux + muy * muy + c1) * (varx + vary + c2)))


def framewise_ssim(estimate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Framewise global SSIM scores."""

    data_range = float(max(np.max(reference) - np.min(reference), np.max(estimate) - np.min(estimate), 1e-30))
    return np.asarray([ssim_global(a, b, data_range) for a, b in zip(estimate, reference)], dtype=np.float64)


def temporal_difference_error(estimate: np.ndarray, reference: np.ndarray) -> float:
    """NRMSE between adjacent-frame temporal differences."""

    return nrmse(np.diff(estimate, axis=0), np.diff(reference, axis=0))


def mass_curve(sequence: np.ndarray) -> np.ndarray:
    """Total flux/mass per frame."""

    return np.sum(np.asarray(sequence, dtype=np.float64), axis=(-2, -1))


def metric_summary(estimate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """Common scalar summary metrics for reconstruction papers."""

    return {
        "nrmse": nrmse(estimate, reference),
        "mean_frame_nrmse": float(np.mean(framewise_nrmse(estimate, reference))),
        "mean_ssim_global": float(np.mean(framewise_ssim(estimate, reference))),
        "temporal_difference_nrmse": temporal_difference_error(estimate, reference),
        "normalized_correlation": normalized_correlation(estimate, reference),
        "mass_curve_nrmse": nrmse(mass_curve(estimate), mass_curve(reference)),
    }

