"""Evaluation metrics for dynamic image reconstruction."""

from __future__ import annotations

import numpy as np

try:  # Optional dependency; fallback below keeps the package lightweight.
    from skimage.metrics import structural_similarity as _skimage_ssim
except Exception:  # pragma: no cover - exercised only when skimage is absent.
    _skimage_ssim = None


_EPS = 1e-30


def normalize_sequence(sequence: np.ndarray, mode: str = "flux", total_flux: float = 1.0) -> np.ndarray:
    """Normalize an image sequence for image-domain metric evaluation.

    Parameters
    ----------
    sequence:
        Array with shape ``(K,H,W)`` or a single frame ``(H,W)``.
    mode:
        ``"flux"`` rescales each frame to ``total_flux``. ``"minmax"`` rescales
        each frame to ``[0,1]``. ``"none"`` returns the input values.
    total_flux:
        Per-frame target flux when ``mode="flux"``.

    Notes
    -----
    The reconstruction arrays are usually in visibility-calibrated flux units,
    while PNG ground-truth frames are often loaded on a display scale. For the
    automatic run summary, per-frame flux normalization is the least surprising
    default and matches the legacy evaluation workflow more closely than raw
    array comparison.
    """

    x = np.asarray(sequence, dtype=np.float64)
    single_frame = x.ndim == 2
    if single_frame:
        x = x[None, :, :]
    if x.ndim != 3:
        raise ValueError("sequence must have shape (K,H,W) or (H,W)")

    mode = str(mode).lower()
    y = x.copy()
    if mode == "none":
        pass
    elif mode == "flux":
        masses = np.sum(y, axis=(1, 2), keepdims=True)
        y = y * (float(total_flux) / (masses + _EPS))
    elif mode == "minmax":
        mins = np.min(y, axis=(1, 2), keepdims=True)
        maxs = np.max(y, axis=(1, 2), keepdims=True)
        y = (y - mins) / (maxs - mins + _EPS)
    elif mode == "zscore":
        means = np.mean(y, axis=(1, 2), keepdims=True)
        stds = np.std(y, axis=(1, 2), keepdims=True)
        y = (y - means) / (stds + _EPS)
    else:
        raise ValueError("normalization mode must be one of: flux, minmax, zscore, none")
    return y[0] if single_frame else y


def nrmse(estimate: np.ndarray, reference: np.ndarray) -> float:
    """Normalized root mean squared error."""

    estimate = np.asarray(estimate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    return float(np.linalg.norm(estimate - reference) / (np.linalg.norm(reference) + _EPS))


def framewise_nrmse(estimate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Framewise NRMSE for sequences with shape ``(K,H,W)``."""

    return np.asarray([nrmse(a, b) for a, b in zip(estimate, reference)], dtype=np.float64)


def normalized_correlation(estimate: np.ndarray, reference: np.ndarray) -> float:
    """Cosine similarity between two arrays after mean removal."""

    a = np.asarray(estimate, dtype=np.float64).ravel()
    b = np.asarray(reference, dtype=np.float64).ravel()
    a = a - np.mean(a)
    b = b - np.mean(b)
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + _EPS))


def ssim_global(estimate: np.ndarray, reference: np.ndarray, data_range: float | None = None) -> float:
    """Global SSIM approximation without optional image-processing dependencies."""

    x = np.asarray(estimate, dtype=np.float64)
    y = np.asarray(reference, dtype=np.float64)
    if data_range is None:
        data_range = float(max(np.max(y) - np.min(y), np.max(x) - np.min(x), _EPS))
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mux = float(np.mean(x))
    muy = float(np.mean(y))
    varx = float(np.mean((x - mux) ** 2))
    vary = float(np.mean((y - muy) ** 2))
    cov = float(np.mean((x - mux) * (y - muy)))
    return float(((2 * mux * muy + c1) * (2 * cov + c2)) / ((mux * mux + muy * muy + c1) * (varx + vary + c2)))


def ssim_frame(estimate: np.ndarray, reference: np.ndarray, data_range: float | None = None) -> float:
    """SSIM for one frame, using scikit-image when available.

    Falls back to the older global SSIM approximation if ``scikit-image`` is not
    installed. The fallback is useful for tests and minimal installations, but
    paper numbers should preferably come from the scikit-image path.
    """

    x = np.asarray(estimate, dtype=np.float64)
    y = np.asarray(reference, dtype=np.float64)
    if data_range is None:
        data_range = float(max(np.max(y) - np.min(y), np.max(x) - np.min(x), _EPS))
    if _skimage_ssim is not None:
        # win_size must be odd and <= min(image dimensions). For tiny unit-test
        # images, use the largest valid odd window at least 3 when possible.
        min_dim = min(x.shape[-2:])
        kwargs = {"data_range": data_range}
        if min_dim < 7:
            win_size = min_dim if min_dim % 2 == 1 else min_dim - 1
            if win_size >= 3:
                kwargs["win_size"] = win_size
        return float(_skimage_ssim(y, x, **kwargs))
    return ssim_global(x, y, data_range=data_range)


def framewise_ssim(estimate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Framewise SSIM scores."""

    data_range = float(max(np.max(reference) - np.min(reference), np.max(estimate) - np.min(estimate), _EPS))
    return np.asarray([ssim_frame(a, b, data_range) for a, b in zip(estimate, reference)], dtype=np.float64)


def temporal_difference_error(estimate: np.ndarray, reference: np.ndarray) -> float:
    """NRMSE between adjacent-frame temporal differences."""

    return nrmse(np.diff(estimate, axis=0), np.diff(reference, axis=0))


def mass_curve(sequence: np.ndarray) -> np.ndarray:
    """Total flux/mass per frame."""

    return np.sum(np.asarray(sequence, dtype=np.float64), axis=(-2, -1))


def spatial_gradients(sequence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Forward spatial differences for a video sequence.

    Returns
    -------
    dx, dy:
        Arrays with shape ``(K,H,W-1)`` and ``(K,H-1,W)`` respectively.
    """

    x = np.asarray(sequence, dtype=np.float64)
    if x.ndim != 3:
        raise ValueError("sequence must have shape (K,H,W)")
    return np.diff(x, axis=2), np.diff(x, axis=1)


def temporal_gradients(sequence: np.ndarray) -> np.ndarray:
    """Forward temporal differences with shape ``(K-1,H,W)``."""

    x = np.asarray(sequence, dtype=np.float64)
    if x.ndim != 3:
        raise ValueError("sequence must have shape (K,H,W)")
    return np.diff(x, axis=0)


def _squared_norm(*arrays: np.ndarray) -> float:
    return float(sum(np.sum(np.asarray(array, dtype=np.float64) ** 2) for array in arrays))




def parse_stge_lambda(value: float | str) -> float | str:
    """Parse an STGE lambda option. Accepts ``'auto'`` or a positive number.

    Command-line arguments arrive as strings, so this helper treats strings like
    ``'0.5'`` as manual numeric lambda values.
    """

    if isinstance(value, str):
        if value.lower() == "auto":
            return "auto"
        try:
            parsed = float(value)
        except ValueError as exc:
            raise ValueError("STGE lambda must be 'auto' or a positive float") from exc
        if parsed <= 0.0:
            raise ValueError("STGE lambda must be positive")
        return parsed
    parsed = float(value)
    if parsed <= 0.0:
        raise ValueError("STGE lambda must be positive")
    return parsed


def auto_stge_lambda(reference: np.ndarray) -> float:
    """Choose STGE lambda from GT/reference gradient energy.

    The convention is

    ``lambda = ||spatial gradient of GT||_2 / ||temporal gradient of GT||_2``.

    With this choice, the weighted temporal-gradient reference energy is on the
    same order as the spatial-gradient reference energy. This is useful when the
    metric is intended to compare spatial and temporal gradient errors without
    one component dominating only because of units or sampling cadence.
    """

    dx, dy = spatial_gradients(reference)
    dt = temporal_gradients(reference)
    spatial_norm = np.sqrt(_squared_norm(dx, dy))
    temporal_norm = np.sqrt(_squared_norm(dt))
    if temporal_norm <= _EPS:
        return 1.0
    return float(spatial_norm / (temporal_norm + _EPS))


def spatiotemporal_gradient_error(
    estimate: np.ndarray,
    reference: np.ndarray,
    *,
    temporal_weight: float | str = "auto",
) -> dict[str, object]:
    """Spatiotemporal Gradient Error (STGE).

    STGE compares spatial and temporal finite-difference fields:

    ``sqrt(||d_x e - d_x r||^2 + ||d_y e - d_y r||^2 + lambda^2 ||d_t e - d_t r||^2)``
    divided by
    ``sqrt(||d_x r||^2 + ||d_y r||^2 + lambda^2 ||d_t r||^2)``.

    The same normalization is used for the spatial-only and temporal-only
    component errors. If ``temporal_weight='auto'``, lambda is computed from the
    reference sequence with :func:`auto_stge_lambda`.
    """

    estimate = np.asarray(estimate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if estimate.shape != reference.shape:
        raise ValueError(f"estimate and reference shapes differ: {estimate.shape} vs {reference.shape}")
    if estimate.ndim != 3:
        raise ValueError("estimate/reference must have shape (K,H,W)")

    parsed_weight = parse_stge_lambda(temporal_weight)
    if parsed_weight == "auto":
        lam = auto_stge_lambda(reference)
        lambda_mode = "auto"
    else:
        lam = float(parsed_weight)
        lambda_mode = "manual"

    ex, ey = spatial_gradients(estimate)
    rx, ry = spatial_gradients(reference)
    et = temporal_gradients(estimate)
    rt = temporal_gradients(reference)

    spatial_error_sq = _squared_norm(ex - rx, ey - ry)
    spatial_ref_sq = _squared_norm(rx, ry)
    temporal_error_sq = _squared_norm(et - rt)
    temporal_ref_sq = _squared_norm(rt)

    total_error_sq = spatial_error_sq + lam * lam * temporal_error_sq
    total_ref_sq = spatial_ref_sq + lam * lam * temporal_ref_sq

    frame_spatial = []
    for k in range(estimate.shape[0]):
        err = _squared_norm(ex[k] - rx[k], ey[k] - ry[k])
        ref = _squared_norm(rx[k], ry[k])
        frame_spatial.append(float(np.sqrt(err) / (np.sqrt(ref) + _EPS)))

    interval_temporal = []
    if estimate.shape[0] >= 2:
        for k in range(estimate.shape[0] - 1):
            err = float(np.sum((et[k] - rt[k]) ** 2))
            ref = float(np.sum(rt[k] ** 2))
            interval_temporal.append(float(np.sqrt(err) / (np.sqrt(ref) + _EPS)))

    return {
        "stge": float(np.sqrt(total_error_sq) / (np.sqrt(total_ref_sq) + _EPS)),
        "stge_spatial": float(np.sqrt(spatial_error_sq) / (np.sqrt(spatial_ref_sq) + _EPS)),
        "stge_temporal": float(np.sqrt(temporal_error_sq) / (np.sqrt(temporal_ref_sq) + _EPS)) if estimate.shape[0] >= 2 else float("nan"),
        "stge_lambda": float(lam),
        "stge_lambda_mode": lambda_mode,
        "stge_spatial_reference_norm": float(np.sqrt(spatial_ref_sq)),
        "stge_temporal_reference_norm": float(np.sqrt(temporal_ref_sq)),
        "frame_spatial_gradient_nrmse": frame_spatial,
        "interval_temporal_gradient_nrmse": interval_temporal,
    }



def fourier_chi2_report(sequence: np.ndarray, data_terms: list) -> dict[str, object]:
    """Fourier-plane chi-squared report against observed visibilities.

    The residuals are evaluated in the same normalized weighted visibility
    coordinates used by :class:`~ot_uot.core.visibility.ComplexVisibilityDataTerm`:

    ``r_k = A_k u_k - y_k``.

    Therefore ``fourier_chi2 = sum_k ||r_k||_2^2`` is exactly twice the
    unweighted data-term value used internally by the optimizer.  Because the
    observation loader rescales weights/operators for numerical conditioning,
    this is best interpreted as a comparable Fourier consistency metric within
    one experiment/sweep, not as an absolute calibrated EHT reduced chi-square.
    """

    x = np.asarray(sequence, dtype=np.float64)
    if x.ndim != 3:
        raise ValueError("sequence must have shape (K,H,W)")
    if len(data_terms) != x.shape[0]:
        raise ValueError(f"number of data terms ({len(data_terms)}) does not match sequence length ({x.shape[0]})")

    frame_chi2: list[float] = []
    frame_reduced_real: list[float] = []
    frame_reduced_complex: list[float] = []
    frame_counts: list[int] = []
    for image, data_term in zip(x, data_terms):
        residual = data_term.residual(image)
        count = int(residual.size)
        chi2 = float(np.sum(np.abs(residual) ** 2))
        frame_counts.append(count)
        frame_chi2.append(chi2)
        frame_reduced_real.append(float(chi2 / max(2 * count, 1)))
        frame_reduced_complex.append(float(chi2 / max(count, 1)))

    total_count = int(np.sum(frame_counts))
    total_chi2 = float(np.sum(frame_chi2))
    return {
        "fourier_chi2": total_chi2,
        "fourier_reduced_chi2": float(total_chi2 / max(2 * total_count, 1)),
        "fourier_complex_reduced_chi2": float(total_chi2 / max(total_count, 1)),
        "fourier_data_term_value": float(0.5 * total_chi2),
        "fourier_visibility_count": total_count,
        "fourier_degrees_of_freedom": int(2 * total_count),
        "frame_fourier_chi2": frame_chi2,
        "frame_fourier_reduced_chi2": frame_reduced_real,
        "frame_fourier_complex_reduced_chi2": frame_reduced_complex,
        "frame_visibility_count": frame_counts,
    }


def compare_initialization_and_final_fourier_reports(
    initialization: np.ndarray,
    final: np.ndarray,
    data_terms: list,
) -> dict[str, object]:
    """Return Fourier chi-squared metrics for initialization and final video."""

    init_report = fourier_chi2_report(initialization, data_terms)
    final_report = fourier_chi2_report(final, data_terms)

    init_frame = np.asarray(init_report["frame_fourier_reduced_chi2"], dtype=np.float64)
    final_frame = np.asarray(final_report["frame_fourier_reduced_chi2"], dtype=np.float64)
    init_frame_complex = np.asarray(init_report["frame_fourier_complex_reduced_chi2"], dtype=np.float64)
    final_frame_complex = np.asarray(final_report["frame_fourier_complex_reduced_chi2"], dtype=np.float64)
    return {
        "initialization_fourier_metrics": init_report,
        "post_uot_fourier_metrics": final_report,
        "delta_fourier_metrics": {
            "delta_fourier_chi2": float(final_report["fourier_chi2"] - init_report["fourier_chi2"]),
            "delta_fourier_reduced_chi2": float(final_report["fourier_reduced_chi2"] - init_report["fourier_reduced_chi2"]),
            "delta_fourier_complex_reduced_chi2": float(final_report["fourier_complex_reduced_chi2"] - init_report["fourier_complex_reduced_chi2"]),
            "delta_fourier_data_term_value": float(final_report["fourier_data_term_value"] - init_report["fourier_data_term_value"]),
            "delta_frame_fourier_reduced_chi2": (final_frame - init_frame).tolist(),
            "delta_frame_fourier_complex_reduced_chi2": (final_frame_complex - init_frame_complex).tolist(),
        },
    }

def metric_summary(estimate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """Common scalar summary metrics for reconstruction papers."""

    stge_report = spatiotemporal_gradient_error(estimate, reference, temporal_weight="auto")
    return {
        "nrmse": nrmse(estimate, reference),
        "mean_frame_nrmse": float(np.mean(framewise_nrmse(estimate, reference))),
        "mean_ssim": float(np.mean(framewise_ssim(estimate, reference))),
        "mean_ssim_global": float(np.mean([ssim_global(a, b) for a, b in zip(estimate, reference)])),
        "temporal_difference_nrmse": temporal_difference_error(estimate, reference),
        "normalized_correlation": normalized_correlation(estimate, reference),
        "mass_curve_nrmse": nrmse(mass_curve(estimate), mass_curve(reference)),
        "stge": float(stge_report["stge"]),
        "stge_spatial": float(stge_report["stge_spatial"]),
        "stge_temporal": float(stge_report["stge_temporal"]),
        "stge_lambda": float(stge_report["stge_lambda"]),
    }


def reconstruction_metric_report(
    estimate: np.ndarray,
    reference: np.ndarray,
    *,
    normalization: str = "flux",
    total_flux: float = 1.0,
    stge_lambda: float | str = "auto",
) -> dict[str, object]:
    """Return automatic run metrics comparing a video to ground truth.

    The headline values requested for smoke/formal runs are
    ``mean_frame_nrmse``, ``mean_ssim``, and ``stge``. Per-frame/per-interval
    arrays are included so the output can be inspected without rerunning
    evaluation scripts.
    """

    est = normalize_sequence(estimate, mode=normalization, total_flux=total_flux)
    ref = normalize_sequence(reference, mode=normalization, total_flux=total_flux)
    if est.shape != ref.shape:
        raise ValueError(f"estimate and reference shapes differ: {est.shape} vs {ref.shape}")
    nrmse_values = framewise_nrmse(est, ref)
    ssim_values = framewise_ssim(est, ref)
    stge_report = spatiotemporal_gradient_error(est, ref, temporal_weight=stge_lambda)
    return {
        "normalization": normalization,
        "total_flux": float(total_flux),
        "frames": int(est.shape[0]),
        "mean_frame_nrmse": float(np.mean(nrmse_values)),
        "mean_ssim": float(np.mean(ssim_values)),
        "frame_nrmse": nrmse_values.tolist(),
        "frame_ssim": ssim_values.tolist(),
        "sequence_nrmse": nrmse(est, ref),
        **stge_report,
    }


def compare_initialization_and_final_reports(
    initialization: np.ndarray,
    final: np.ndarray,
    reference: np.ndarray,
    *,
    normalization: str = "flux",
    total_flux: float = 1.0,
    stge_lambda: float | str = "auto",
) -> dict[str, object]:
    """Return GT metrics for both initialization and final reconstruction.

    This is the automatic report used by the command-line driver.  In
    StarWarps postprocessing experiments, ``initialization`` is the raw
    StarWarps video and ``final`` is the StarWarps+UOT output.  In ordinary
    reconstruction experiments, ``initialization`` is the loaded static or
    user-supplied initial video and ``final`` is the ADMM output.
    """

    # Resolve the STGE lambda once from normalized GT, so initialization and
    # post-UOT reports use exactly the same temporal weighting.
    normalized_reference = normalize_sequence(reference, mode=normalization, total_flux=total_flux)
    resolved_stge_lambda: float | str
    parsed_lambda = parse_stge_lambda(stge_lambda)
    if parsed_lambda == "auto":
        resolved_stge_lambda = auto_stge_lambda(normalized_reference)
        stge_lambda_mode = "auto"
    else:
        resolved_stge_lambda = float(parsed_lambda)
        stge_lambda_mode = "manual"

    init_report = reconstruction_metric_report(
        initialization,
        reference,
        normalization=normalization,
        total_flux=total_flux,
        stge_lambda=resolved_stge_lambda,
    )
    final_report = reconstruction_metric_report(
        final,
        reference,
        normalization=normalization,
        total_flux=total_flux,
        stge_lambda=resolved_stge_lambda,
    )
    # Keep the shared lambda mode explicit after forcing the resolved numeric
    # value into each report.
    init_report["stge_lambda_mode"] = stge_lambda_mode
    final_report["stge_lambda_mode"] = stge_lambda_mode

    init_nrmse = np.asarray(init_report["frame_nrmse"], dtype=np.float64)
    final_nrmse = np.asarray(final_report["frame_nrmse"], dtype=np.float64)
    init_ssim = np.asarray(init_report["frame_ssim"], dtype=np.float64)
    final_ssim = np.asarray(final_report["frame_ssim"], dtype=np.float64)
    init_spatial = np.asarray(init_report["frame_spatial_gradient_nrmse"], dtype=np.float64)
    final_spatial = np.asarray(final_report["frame_spatial_gradient_nrmse"], dtype=np.float64)
    init_temporal = np.asarray(init_report["interval_temporal_gradient_nrmse"], dtype=np.float64)
    final_temporal = np.asarray(final_report["interval_temporal_gradient_nrmse"], dtype=np.float64)
    return {
        "normalization": normalization,
        "total_flux": float(total_flux),
        "frames": int(final_report["frames"]),
        "stge_lambda": float(resolved_stge_lambda),
        "stge_lambda_mode": stge_lambda_mode,
        "initialization_metrics": init_report,
        "post_uot_metrics": final_report,
        "delta_metrics": {
            "delta_mean_frame_nrmse": float(final_report["mean_frame_nrmse"] - init_report["mean_frame_nrmse"]),
            "delta_mean_ssim": float(final_report["mean_ssim"] - init_report["mean_ssim"]),
            "delta_sequence_nrmse": float(final_report["sequence_nrmse"] - init_report["sequence_nrmse"]),
            "delta_stge": float(final_report["stge"] - init_report["stge"]),
            "delta_stge_spatial": float(final_report["stge_spatial"] - init_report["stge_spatial"]),
            "delta_stge_temporal": float(final_report["stge_temporal"] - init_report["stge_temporal"]),
            "delta_frame_nrmse": (final_nrmse - init_nrmse).tolist(),
            "delta_frame_ssim": (final_ssim - init_ssim).tolist(),
            "delta_frame_spatial_gradient_nrmse": (final_spatial - init_spatial).tolist(),
            "delta_interval_temporal_gradient_nrmse": (final_temporal - init_temporal).tolist(),
        },
    }
