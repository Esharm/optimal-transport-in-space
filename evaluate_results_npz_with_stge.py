"""Evaluate reconstruction arrays saved by main7/main8-style result folders.

This script is a thin adapter around the metric functions in PROJECT_ROOT /
evaluate_frames.py. It reads a result folder in the current directory, loads
its results.npz, compares saved video arrays against GT, and writes CSV/JSON
summaries.

Normal workflow:

    1. Edit CONFIG["results_folder"] below.
    2. Run:

        python evaluate_results_npz.py

Expected result file:

    <results_folder>/results.npz

The script looks for GT in the npz key "gt". It then evaluates available video
keys such as "static", "joint", "ot", "warmup", and "background_video".
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_frames_v2 import (
    azimuthal_profile,
    azimuthal_profile_error,
    compute_fourier_chi_squared_image_proxy,
    compute_frc_curve_image,
    compute_nrmse,
    compute_ssim,
    compute_spatiotemporal_gradient_error,
    compute_temporal_variance_map_error,
    curve_auc,
    default_center,
    estimate_annulus_from_truth,
    frc_cutoff_frequency,
    normalized_profile_l2_error,
    radial_profile,
)


SCRIPT_DIR = Path(__file__).resolve().parent


CONFIG = {
    # Folder in the current directory containing results.npz.
    #"results_folder": "OptimalTransportTest/main8_static_init_joint_tv_signed_uot_results",
    "results_folder": "ot_uot_pairwise_main7_21",
    "results_file": "reconstruction.npz",
    "output_subdir": "evaluation_from_npz",

    # If candidate_keys is None, this ordered list is filtered to available keys.
    # You can also set this to ["static", "joint"] etc.
    "candidate_keys": None,
    "default_candidate_keys": [
        "background_video",
        "static",
        "warmup",
        "ot",
        "joint",
    ],

    # Frame/image preprocessing passed into evaluate_frames.py functions.
    "normalization": "flux",  # minmax, flux, zscore, none
    "total_flux": 1.0,
    "max_frames": None,

    # Metric toggles.
    "metrics": {
        "nrmse": True,
        "ssim": True,
        "frc": True,
        "fourier_chi2": True,
        "radial_profile_error": True,
        "azimuthal_profile_error": True,
        "temporal_variance_map_error": True,

        # Sequence-level metric: compares spatial edge gradients and
        # frame-to-frame temporal gradients between GT and reconstruction.
        "spatiotemporal_gradient_error": True,
    },

    # Spatiotemporal gradient error settings.
    "spatiotemporal_gradient": {
        "norm": "l2",  # l2 or l1
        "spatial_weight": 1.0,
        "temporal_weight": 1.0,
        "use_absolute_gradients": False,
        "epsilon": 1e-12,
    },

    # Fourier/FRC settings.
    "frc_num_bins": None,
    "frc_threshold": 0.143,
    "frc_min_samples_per_ring": 1,
    "fourier_chi2_mode": "complex",
    "fourier_chi2_epsilon": 1e-8,
    "fourier_chi2_denominator": "global_power",
    "include_dc_in_fourier_chi2": False,

    # Radial/azimuthal profile settings.
    "radial_bins": 64,
    "azimuthal_bins": 72,
    "center_x": None,
    "center_y": None,
    "azimuthal_inner_radius": None,
    "azimuthal_outer_radius": None,
    "azimuthal_annulus_width_fraction": 0.25,
    "allow_azimuthal_roll": False,
}


@dataclass
class LoadedResults:
    path: Path
    gt: np.ndarray
    candidates: dict[str, np.ndarray]
    names: list[str]


def json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def as_video_array(value, key: str) -> np.ndarray | None:
    arr = np.asarray(value)
    if arr.ndim != 3:
        return None
    if arr.shape[0] <= 0 or arr.shape[1] <= 0 or arr.shape[2] <= 0:
        return None
    if not np.issubdtype(arr.dtype, np.number):
        return None
    return np.asarray(arr, dtype=np.float64)


def load_results_npz(cfg=CONFIG) -> LoadedResults:
    folder = SCRIPT_DIR / cfg["results_folder"]
    path = folder / cfg["results_file"]
    if not path.exists():
        raise FileNotFoundError(f"Could not find result file: {path}")

    loaded = np.load(path, allow_pickle=False)
    if "gt" not in loaded:
        raise KeyError(f"{path} does not contain key 'gt'")
    gt = as_video_array(loaded["gt"], "gt")
    if gt is None:
        raise ValueError(
            f"{path} contains an empty or invalid 'gt' array. "
            "Use result files that saved GT, or add GT loading here."
        )

    max_frames = cfg.get("max_frames")
    if max_frames is not None:
        gt = gt[: int(max_frames)]

    if cfg.get("candidate_keys") is None:
        candidate_keys = cfg["default_candidate_keys"]
    else:
        candidate_keys = cfg["candidate_keys"]

    candidates = {}
    for key in candidate_keys:
        if key not in loaded:
            continue
        arr = as_video_array(loaded[key], key)
        if arr is None:
            continue
        arr = arr[: len(gt)]
        if arr.shape != gt.shape:
            print(f"Skipping {key}: shape {arr.shape} does not match GT {gt.shape}")
            continue
        candidates[key] = arr

    if not candidates:
        available = ", ".join(sorted(loaded.files))
        raise ValueError(f"No candidate video arrays found. Available keys: {available}")

    names = []
    if "names" in loaded:
        names = [str(item) for item in loaded["names"][: len(gt)]]
    if not names:
        names = [f"frame_{i:03d}" for i in range(len(gt))]

    return LoadedResults(path=path, gt=gt, candidates=candidates, names=names)


def frame_metrics(gt_frame, recon_frame, cfg):
    rows = {}
    metrics = cfg["metrics"]
    normalization = cfg["normalization"]
    total_flux = cfg["total_flux"]

    if metrics.get("nrmse", True):
        rows["nrmse"] = compute_nrmse(
            gt_frame,
            recon_frame,
            normalization=normalization,
            total_flux=total_flux,
        )
    if metrics.get("ssim", True):
        rows["ssim"] = compute_ssim(gt_frame, recon_frame)
    if metrics.get("fourier_chi2", True):
        rows["fourier_chi2"] = compute_fourier_chi_squared_image_proxy(
            gt_frame,
            recon_frame,
            normalization=normalization,
            total_flux=total_flux,
            epsilon=cfg["fourier_chi2_epsilon"],
            mode=cfg["fourier_chi2_mode"],
            denominator_mode=cfg["fourier_chi2_denominator"],
            exclude_dc=not cfg["include_dc_in_fourier_chi2"],
        )
    if metrics.get("frc", True):
        freq, frc, counts = compute_frc_curve_image(
            gt_frame,
            recon_frame,
            normalization=normalization,
            total_flux=total_flux,
            num_bins=cfg["frc_num_bins"],
            min_samples=cfg["frc_min_samples_per_ring"],
        )
        rows["frc_auc"] = curve_auc(freq, frc)
        rows["frc_cutoff"] = frc_cutoff_frequency(
            freq,
            frc,
            threshold=cfg["frc_threshold"],
        )
        rows["frc_valid_bins"] = int(np.count_nonzero(np.isfinite(frc)))
        rows["frc_sample_count_total"] = int(np.sum(counts))

    center = default_center(
        gt_frame.shape,
        center_x=cfg["center_x"],
        center_y=cfg["center_y"],
    )
    if metrics.get("radial_profile_error", True):
        _, true_profile, _ = radial_profile(
            gt_frame,
            center=center,
            num_bins=cfg["radial_bins"],
        )
        _, recon_profile, _ = radial_profile(
            recon_frame,
            center=center,
            num_bins=cfg["radial_bins"],
        )
        rows["radial_profile_error"] = normalized_profile_l2_error(
            true_profile,
            recon_profile,
        )

    if metrics.get("azimuthal_profile_error", True):
        inner = cfg["azimuthal_inner_radius"]
        outer = cfg["azimuthal_outer_radius"]
        if inner is None or outer is None:
            inner, outer, peak = estimate_annulus_from_truth(
                gt_frame,
                center=center,
                radial_bins=cfg["radial_bins"],
                width_fraction=cfg["azimuthal_annulus_width_fraction"],
            )
            rows["azimuthal_annulus_peak_radius"] = peak
        rows["azimuthal_inner_radius"] = inner
        rows["azimuthal_outer_radius"] = outer

        _, true_az, _ = azimuthal_profile(
            gt_frame,
            center=center,
            num_bins=cfg["azimuthal_bins"],
            inner_radius=inner,
            outer_radius=outer,
        )
        _, recon_az, _ = azimuthal_profile(
            recon_frame,
            center=center,
            num_bins=cfg["azimuthal_bins"],
            inner_radius=inner,
            outer_radius=outer,
        )
        err, roll = azimuthal_profile_error(
            true_az,
            recon_az,
            allow_roll=cfg["allow_azimuthal_roll"],
        )
        rows["azimuthal_profile_error"] = err
        rows["azimuthal_best_roll"] = roll
    return rows


def stack_metrics(gt, recon, cfg):
    rows = {}
    metrics = cfg["metrics"]

    if metrics.get("temporal_variance_map_error", True):
        error, true_var, recon_var, diff = compute_temporal_variance_map_error(
            gt,
            recon,
            normalization=cfg["normalization"],
            total_flux=cfg["total_flux"],
        )
        rows["temporal_variance_map_error"] = error
        rows["temporal_variance_true_sum"] = float(np.sum(true_var))
        rows["temporal_variance_recon_sum"] = float(np.sum(recon_var))
        rows["temporal_variance_absdiff_sum"] = float(np.sum(np.abs(diff)))

    if metrics.get("spatiotemporal_gradient_error", True):
        st_cfg = cfg.get("spatiotemporal_gradient", {})
        st_result = compute_spatiotemporal_gradient_error(
            gt,
            recon,
            normalization=cfg["normalization"],
            total_flux=cfg["total_flux"],
            norm=st_cfg.get("norm", "l2"),
            spatial_weight=st_cfg.get("spatial_weight", 1.0),
            temporal_weight=st_cfg.get("temporal_weight", 1.0),
            use_absolute_gradients=st_cfg.get("use_absolute_gradients", False),
            epsilon=st_cfg.get("epsilon", 1e-12),
        )

        # evaluate_frames.py returns either:
        #   metrics_dict
        # or
        #   (metrics_dict, spatial_error_map, temporal_error_map).
        # The NPZ summary only stores scalar metrics; maps are intentionally ignored.
        if isinstance(st_result, tuple):
            st_metrics = st_result[0]
        else:
            st_metrics = st_result

        rows.update(st_metrics)

    return rows


def evaluate_results(cfg=CONFIG):
    loaded = load_results_npz(cfg)
    output_dir = loaded.path.parent / cfg["output_subdir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    per_frame_rows = []
    summary_rows = []
    for label, recon in loaded.candidates.items():
        print(f"Evaluating {label}: {recon.shape}")
        for k, (gt_frame, recon_frame) in enumerate(zip(loaded.gt, recon)):
            row = {
                "candidate": label,
                "frame_index": k,
                "frame_name": loaded.names[k] if k < len(loaded.names) else f"frame_{k:03d}",
            }
            row.update(frame_metrics(gt_frame, recon_frame, cfg))
            per_frame_rows.append(row)

        candidate_rows = [
            row for row in per_frame_rows
            if row["candidate"] == label
        ]
        summary = {"candidate": label, "frames": len(candidate_rows)}
        numeric_keys = sorted(
            key for key in candidate_rows[0]
            if key not in {"candidate", "frame_index", "frame_name"}
            and isinstance(candidate_rows[0][key], (int, float, np.integer, np.floating))
        )
        for key in numeric_keys:
            values = np.asarray([row.get(key, np.nan) for row in candidate_rows], dtype=float)
            summary[f"{key}_mean"] = float(np.nanmean(values))
            summary[f"{key}_min"] = float(np.nanmin(values))
            summary[f"{key}_max"] = float(np.nanmax(values))
        summary.update(stack_metrics(loaded.gt, recon, cfg))
        summary_rows.append(summary)

    per_frame_df = pd.DataFrame(per_frame_rows)
    summary_df = pd.DataFrame(summary_rows)
    per_frame_path = output_dir / "per_frame_metrics.csv"
    summary_path = output_dir / "summary_metrics.csv"
    config_path = output_dir / "evaluation_config.json"

    per_frame_df.to_csv(per_frame_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    config_path.write_text(json.dumps(cfg, indent=2, default=json_default))

    print("\nSummary:")
    print(summary_df.to_string(index=False))
    print(f"\nSaved per-frame metrics: {per_frame_path}")
    print(f"Saved summary metrics:   {summary_path}")
    print(f"Saved config:            {config_path}")


if __name__ == "__main__":
    evaluate_results(CONFIG)
