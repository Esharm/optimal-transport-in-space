"""Result serialization for standalone OT/UOT experiments."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path

import numpy as np

from ot_uot.optimization.admm_state import ADMMState
from ot_uot.optimization.signed_residual_admm import history_as_dicts


def dataclass_to_jsonable(obj):
    """Convert dataclass configuration objects to JSON-compatible values."""

    if is_dataclass(obj):
        return dataclass_to_jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): dataclass_to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [dataclass_to_jsonable(v) for v in obj]
    if hasattr(obj, "value"):
        return obj.value
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_reconstruction_npz(
    output_path: Path | str,
    state: ADMMState,
    *,
    config=None,
    static_sequence: np.ndarray | None = None,
    reference_sequence: np.ndarray | None = None,
    ground_truth: np.ndarray | None = None,
    names: list[str] | None = None,
    extra: dict | None = None,
) -> Path:
    """Save reconstruction variables and history to a compressed NPZ."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "iteration": state.iteration,
        "history": history_as_dicts(state),
        "config": dataclass_to_jsonable(config) if config is not None else None,
        "extra": dataclass_to_jsonable(extra or {}),
    }
    image = state.image_state.image
    background = state.image_state.background
    if background.ndim == 2:
        background_video = np.repeat(background[None, :, :], image.shape[0], axis=0)
    else:
        background_video = background
    payload = {
        "image": image,
        "joint": image,
        "positive": state.image_state.positive,
        "negative": state.image_state.negative,
        "joint_positive_residual": state.image_state.positive,
        "joint_negative_residual": state.image_state.negative,
        "background": background,
        "background_video": background_video,
        "decomposition_dual": state.decomposition_dual,
        "metadata": json.dumps(metadata),
    }
    if reference_sequence is not None:
        reference = np.asarray(reference_sequence, dtype=np.float64)
        payload["reference"] = reference
        payload["reference_sequence"] = reference
        payload["starwarps_reference"] = reference
    if static_sequence is not None:
        static = np.asarray(static_sequence, dtype=np.float64)
        payload["static"] = static
        payload["joint_initialization"] = static
        payload["static_positive_residual"] = np.maximum(static - background_video, 0.0)
        payload["static_negative_residual"] = np.maximum(background_video - static, 0.0)
    if ground_truth is not None:
        payload["gt"] = np.asarray(ground_truth, dtype=np.float64)
    automatic_metrics = (extra or {}).get("automatic_metrics") if isinstance(extra, dict) else None
    if isinstance(automatic_metrics, dict):
        # New automatic metric layout: initialization_metrics, post_uot_metrics,
        # and delta_metrics.  For backward compatibility, keep the old generic
        # metric_* keys as aliases for post-UOT/final metrics.
        if "post_uot_metrics" in automatic_metrics:
            init_metrics = automatic_metrics.get("initialization_metrics") or {}
            final_metrics = automatic_metrics.get("post_uot_metrics") or {}
            delta_metrics = automatic_metrics.get("delta_metrics") or {}

            payload["metric_init_mean_frame_nrmse"] = np.asarray(float(init_metrics.get("mean_frame_nrmse", np.nan)))
            payload["metric_init_mean_ssim"] = np.asarray(float(init_metrics.get("mean_ssim", np.nan)))
            payload["metric_init_sequence_nrmse"] = np.asarray(float(init_metrics.get("sequence_nrmse", np.nan)))
            payload["metric_init_stge"] = np.asarray(float(init_metrics.get("stge", np.nan)))
            payload["metric_init_stge_spatial"] = np.asarray(float(init_metrics.get("stge_spatial", np.nan)))
            payload["metric_init_stge_temporal"] = np.asarray(float(init_metrics.get("stge_temporal", np.nan)))
            payload["metric_init_stge_lambda"] = np.asarray(float(init_metrics.get("stge_lambda", np.nan)))
            payload["metric_init_frame_nrmse"] = np.asarray(init_metrics.get("frame_nrmse", []), dtype=np.float64)
            payload["metric_init_frame_ssim"] = np.asarray(init_metrics.get("frame_ssim", []), dtype=np.float64)
            payload["metric_init_frame_spatial_gradient_nrmse"] = np.asarray(init_metrics.get("frame_spatial_gradient_nrmse", []), dtype=np.float64)
            payload["metric_init_interval_temporal_gradient_nrmse"] = np.asarray(init_metrics.get("interval_temporal_gradient_nrmse", []), dtype=np.float64)

            payload["metric_final_mean_frame_nrmse"] = np.asarray(float(final_metrics.get("mean_frame_nrmse", np.nan)))
            payload["metric_final_mean_ssim"] = np.asarray(float(final_metrics.get("mean_ssim", np.nan)))
            payload["metric_final_sequence_nrmse"] = np.asarray(float(final_metrics.get("sequence_nrmse", np.nan)))
            payload["metric_final_stge"] = np.asarray(float(final_metrics.get("stge", np.nan)))
            payload["metric_final_stge_spatial"] = np.asarray(float(final_metrics.get("stge_spatial", np.nan)))
            payload["metric_final_stge_temporal"] = np.asarray(float(final_metrics.get("stge_temporal", np.nan)))
            payload["metric_final_stge_lambda"] = np.asarray(float(final_metrics.get("stge_lambda", np.nan)))
            payload["metric_final_frame_nrmse"] = np.asarray(final_metrics.get("frame_nrmse", []), dtype=np.float64)
            payload["metric_final_frame_ssim"] = np.asarray(final_metrics.get("frame_ssim", []), dtype=np.float64)
            payload["metric_final_frame_spatial_gradient_nrmse"] = np.asarray(final_metrics.get("frame_spatial_gradient_nrmse", []), dtype=np.float64)
            payload["metric_final_interval_temporal_gradient_nrmse"] = np.asarray(final_metrics.get("interval_temporal_gradient_nrmse", []), dtype=np.float64)

            payload["metric_delta_mean_frame_nrmse"] = np.asarray(float(delta_metrics.get("delta_mean_frame_nrmse", np.nan)))
            payload["metric_delta_mean_ssim"] = np.asarray(float(delta_metrics.get("delta_mean_ssim", np.nan)))
            payload["metric_delta_sequence_nrmse"] = np.asarray(float(delta_metrics.get("delta_sequence_nrmse", np.nan)))
            payload["metric_delta_stge"] = np.asarray(float(delta_metrics.get("delta_stge", np.nan)))
            payload["metric_delta_stge_spatial"] = np.asarray(float(delta_metrics.get("delta_stge_spatial", np.nan)))
            payload["metric_delta_stge_temporal"] = np.asarray(float(delta_metrics.get("delta_stge_temporal", np.nan)))
            payload["metric_delta_frame_nrmse"] = np.asarray(delta_metrics.get("delta_frame_nrmse", []), dtype=np.float64)
            payload["metric_delta_frame_ssim"] = np.asarray(delta_metrics.get("delta_frame_ssim", []), dtype=np.float64)
            payload["metric_delta_frame_spatial_gradient_nrmse"] = np.asarray(delta_metrics.get("delta_frame_spatial_gradient_nrmse", []), dtype=np.float64)
            payload["metric_delta_interval_temporal_gradient_nrmse"] = np.asarray(delta_metrics.get("delta_interval_temporal_gradient_nrmse", []), dtype=np.float64)

            fourier_metrics = automatic_metrics.get("fourier_metrics") or {}
            if isinstance(fourier_metrics, dict):
                init_fourier = fourier_metrics.get("initialization_fourier_metrics") or {}
                final_fourier = fourier_metrics.get("post_uot_fourier_metrics") or {}
                delta_fourier = fourier_metrics.get("delta_fourier_metrics") or {}

                payload["metric_init_fourier_chi2"] = np.asarray(float(init_fourier.get("fourier_chi2", np.nan)))
                payload["metric_init_fourier_reduced_chi2"] = np.asarray(float(init_fourier.get("fourier_reduced_chi2", np.nan)))
                payload["metric_init_fourier_complex_reduced_chi2"] = np.asarray(float(init_fourier.get("fourier_complex_reduced_chi2", np.nan)))
                payload["metric_init_frame_fourier_reduced_chi2"] = np.asarray(init_fourier.get("frame_fourier_reduced_chi2", []), dtype=np.float64)

                payload["metric_final_fourier_chi2"] = np.asarray(float(final_fourier.get("fourier_chi2", np.nan)))
                payload["metric_final_fourier_reduced_chi2"] = np.asarray(float(final_fourier.get("fourier_reduced_chi2", np.nan)))
                payload["metric_final_fourier_complex_reduced_chi2"] = np.asarray(float(final_fourier.get("fourier_complex_reduced_chi2", np.nan)))
                payload["metric_final_frame_fourier_reduced_chi2"] = np.asarray(final_fourier.get("frame_fourier_reduced_chi2", []), dtype=np.float64)

                payload["metric_delta_fourier_chi2"] = np.asarray(float(delta_fourier.get("delta_fourier_chi2", np.nan)))
                payload["metric_delta_fourier_reduced_chi2"] = np.asarray(float(delta_fourier.get("delta_fourier_reduced_chi2", np.nan)))
                payload["metric_delta_fourier_complex_reduced_chi2"] = np.asarray(float(delta_fourier.get("delta_fourier_complex_reduced_chi2", np.nan)))
                payload["metric_delta_frame_fourier_reduced_chi2"] = np.asarray(delta_fourier.get("delta_frame_fourier_reduced_chi2", []), dtype=np.float64)

            # Backward-compatible aliases: final/post-UOT metrics.
            payload["metric_mean_frame_nrmse"] = payload["metric_final_mean_frame_nrmse"]
            payload["metric_mean_ssim"] = payload["metric_final_mean_ssim"]
            payload["metric_stge"] = payload["metric_final_stge"]
            payload["metric_stge_spatial"] = payload["metric_final_stge_spatial"]
            payload["metric_stge_temporal"] = payload["metric_final_stge_temporal"]
            payload["metric_frame_nrmse"] = payload["metric_final_frame_nrmse"]
            payload["metric_frame_ssim"] = payload["metric_final_frame_ssim"]
        else:
            payload["metric_mean_frame_nrmse"] = np.asarray(float(automatic_metrics.get("mean_frame_nrmse", np.nan)))
            payload["metric_mean_ssim"] = np.asarray(float(automatic_metrics.get("mean_ssim", np.nan)))
            if "frame_nrmse" in automatic_metrics:
                payload["metric_frame_nrmse"] = np.asarray(automatic_metrics["frame_nrmse"], dtype=np.float64)
            if "frame_ssim" in automatic_metrics:
                payload["metric_frame_ssim"] = np.asarray(automatic_metrics["frame_ssim"], dtype=np.float64)
    if names is not None:
        payload["names"] = np.asarray(names)
    np.savez_compressed(output_path, **payload)
    return output_path


def load_reconstruction_npz(path: Path | str) -> dict:
    """Load a saved reconstruction NPZ."""

    with np.load(Path(path), allow_pickle=False) as npz:
        result = {key: npz[key] for key in npz.files if key != "metadata"}
        result["metadata"] = json.loads(str(npz["metadata"]))
    return result
