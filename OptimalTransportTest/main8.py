"""Joint data + TV + signed-residual UOT reconstruction.

Workflow:

    1. Load per-frame static reconstructions from
       PROJECT_ROOT/static_reconstruction/reconstructed_frames_gray.
    2. Calibrate them only to compute a fixed average background a and a
       spatial-only comparison baseline.
    3. Initialize the joint solve at the static per-frame reconstructions.
    4. Run one joint ADMM solve with data + TV + weak background prior and
       unbalanced BB transport on

           h_k^+ = max(u_k - a, 0),
           h_k^- = max(a - u_k, 0).

This tests whether the joint objective can improve the already data-consistent
static baseline when started in the correct dynamic basin.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from admm import SignedResidualUOTADMM
from run_three_frame_ablation import Config as LoaderConfig
from run_three_frame_ablation import load_data_terms, objective_terms, relative_change
from solvers import TotalVariationRegularizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Config(LoaderConfig):
    static_recon_folder: Path = (
        PROJECT_ROOT / "static_reconstruction" / "reconstructed_frames_gray"
    )
    output_root: Path = Path("main8_static_init_joint_tv_signed_uot_results")
    gt_folder: Path = PROJECT_ROOT / "blackhole_sim" / "data" / "aart_frames"
    fps: int = 5

    frames: int = 15
    frame_indices: tuple[int, ...] | None = None

    # Memory-safe visibility evaluation for ~5000 visibilities/frame.
    max_vis_per_frame: int | None = None
    use_visibility_cache: bool = False
    visibility_chunk_size: int = 128
    parallel_frames: bool = False

    # Static recon loading/scaling.
    static_recon_scale_mode: str = "per_frame"  # "per_frame" or "global"
    normalize_loaded_static: bool = True
    initialization_mode: str = "static"

    # Joint data + TV + signed residual UOT objective.
    tv_weight: float = 1e-5
    prior_weight: float = 1e-4
    beta: float = 1e-4
    eta: float = 1e-3
    uot_source_weight: float = 30.0

    ot_max_iter: int = 60
    ot_min_iter: int = 12
    ot_patience: int = 5
    ot_image_inner_iters: int = 8
    ot_power_iters: int = 5
    transport_slices: int = 7
    transport_inner_iters: int = 100
    transport_tol: float = 7e-4

    primal_tau: float = 10.0
    dual_sigma: float = 0.25

    # Motion diagnostics: compare annulus-restricted positive residual angles.
    motion_annulus_threshold_fraction: float = 0.20
    motion_min_residual_mass_fraction: float = 1e-4


def frame_number(value) -> int | None:
    stem = Path(str(value)).stem
    matches = re.findall(r"\d+", stem)
    if not matches:
        return None
    return int(matches[-1])


def load_gray_image(path: Path, resize: tuple[int, int], normalize: bool = True):
    image = Image.open(path).convert("L").resize(resize)
    arr = np.asarray(image, dtype=np.float64)
    if normalize:
        arr = arr / 255.0
    return arr


def find_static_reconstruction_paths(folder: Path, names):
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Static reconstruction folder not found: {folder}")
    image_files = sorted(
        p for p in folder.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    )
    if not image_files:
        raise FileNotFoundError(f"No static reconstruction images found in {folder}")

    by_frame = {}
    for path in image_files:
        number = frame_number(path)
        if number is not None:
            by_frame.setdefault(number, path)

    selected = []
    for name in names:
        number = frame_number(name)
        if number is not None and number in by_frame:
            selected.append(by_frame[number])
        else:
            raise FileNotFoundError(
                f"Could not match static reconstruction for observation {name}"
            )
    return selected


def calibrate_static_frames(frames, data_terms, mode: str):
    frames = np.asarray(frames, dtype=np.float64)
    numerators = []
    denominators = []
    for frame, term in zip(frames, data_terms):
        predicted = term.sampler.forward(frame)
        numerators.append(float(np.real(np.vdot(predicted, term.f))))
        denominators.append(float(np.vdot(predicted, predicted).real))

    numerators = np.asarray(numerators)
    denominators = np.asarray(denominators)
    if mode == "per_frame":
        scales = np.maximum(numerators / (denominators + 1e-30), 0.0)
    elif mode == "global":
        scale = max(float(numerators.sum() / (denominators.sum() + 1e-30)), 0.0)
        scales = np.full(len(frames), scale)
    else:
        raise ValueError("static_recon_scale_mode must be 'per_frame' or 'global'")

    scaled = frames * scales[:, None, None]
    return scaled, {
        "static_recon_scale_mode": mode,
        "static_recon_scale_min": float(scales.min()),
        "static_recon_scale_max": float(scales.max()),
        "static_recon_scale_mean": float(scales.mean()),
        "static_recon_scale_std": float(scales.std()),
        "static_recon_scale_numerator_sum": float(numerators.sum()),
        "static_recon_scale_denominator_sum": float(denominators.sum()),
    }


def load_static_initialization(cfg, names, data_terms):
    paths = find_static_reconstruction_paths(cfg.static_recon_folder, names)
    frames = [
        load_gray_image(
            path,
            resize=(cfg.image_width, cfg.image_height),
            normalize=cfg.normalize_loaded_static,
        )
        for path in paths
    ]
    scaled, scale_info = calibrate_static_frames(
        np.stack(frames),
        data_terms,
        cfg.static_recon_scale_mode,
    )
    background = scaled.mean(axis=0)
    return scaled, background, paths, scale_info


def run_signed_residual_uot(u_init, data_terms, background, cfg):
    regularizer = TotalVariationRegularizer(
        alpha=cfg.tv_weight,
        iters=cfg.ot_image_inner_iters,
        tau=cfg.primal_tau,
        sigma=cfg.dual_sigma,
        power_iters=cfg.ot_power_iters,
    )
    model = SignedResidualUOTADMM(
        data_terms=data_terms,
        regularizer=regularizer,
        background_image=background,
        beta=cfg.beta,
        eta=cfg.eta,
        source_weight=cfg.uot_source_weight,
        max_iter=cfg.ot_max_iter,
        abs_tol=1e-4,
        rel_tol=5e-3,
        min_iter=cfg.ot_min_iter,
        patience=cfg.ot_patience,
        transport_T=cfg.transport_slices,
        transport_inner_iters=cfg.transport_inner_iters,
        transport_tol=cfg.transport_tol,
        dual_relaxation=1.0,
        prior_image=background,
        prior_weight=cfg.prior_weight,
        stop_on_data_plateau=False,
    )
    print(
        "Starting signed-residual unbalanced BB/ADMM OT "
        f"(outer <= {cfg.ot_max_iter}, image iters = {cfg.ot_image_inner_iters}, "
        f"transport iters <= {cfg.transport_inner_iters})"
    )
    started = time.perf_counter()
    u_ot, history = model.run(u_init.copy())
    elapsed = time.perf_counter() - started
    converged = bool(
        history
        and history[-1]["r_norm"] <= history[-1]["eps_pri"]
        and history[-1]["s_norm"] <= history[-1]["eps_dual"]
        and history[-1]["bb_continuity_residual_max"] <= max(cfg.transport_tol, 1e-8)
    )
    return u_ot, history, elapsed, converged


def _normalize01(x):
    x = np.asarray(x, dtype=np.float64)
    return (x - x.min()) / (x.max() - x.min() + 1e-12)


def _to_uint8(frame):
    return np.clip(255.0 * _normalize01(frame), 0, 255).astype(np.uint8)


def make_labeled_panel(images, labels):
    panels = []
    for image, label in zip(images, labels):
        bgr = cv2.cvtColor(_to_uint8(image), cv2.COLOR_GRAY2BGR)
        cv2.putText(
            bgr,
            label,
            (6, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        panels.append(bgr)
    return np.concatenate(panels, axis=1)


def load_ground_truth_frames(folder: Path, names, resize):
    folder = Path(folder)
    if not folder.exists():
        print(f"Ground-truth folder not found, skipping GT comparison: {folder}")
        return None, []
    image_files = sorted(
        p for p in folder.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
    )
    by_frame = {
        frame_number(path): path for path in image_files if frame_number(path) is not None
    }
    selected = []
    for name in names:
        number = frame_number(name)
        selected.append(by_frame.get(number))
    if any(path is None for path in selected):
        print("Could not match all GT frames; skipping GT comparison.")
        return None, []
    frames = [
        load_gray_image(path, resize=resize, normalize=True)
        for path in selected
    ]
    return np.stack(frames), [path.name for path in selected]


def save_comparison_video(gt, static, ot, root: Path, fps: int):
    if gt is None:
        print("No ground truth available; skipping comparison video.")
        return None
    root.mkdir(parents=True, exist_ok=True)
    output_path = root / "comparison_gt_static_joint_uot.mp4"
    first = make_labeled_panel([gt[0], static[0], ot[0]], ["GT", "static", "joint TV+UOT"])
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (first.shape[1], first.shape[0]),
    )
    writer.write(first)
    for k in range(1, min(len(gt), len(static), len(ot))):
        writer.write(make_labeled_panel([gt[k], static[k], ot[k]], ["GT", "static", "joint TV+UOT"]))
    writer.release()
    return output_path


def sequence_diagnostics(label, frames):
    frames = np.asarray(frames, dtype=np.float64)
    rows = {
        f"{label}_global_min": float(frames.min()),
        f"{label}_global_max": float(frames.max()),
        f"{label}_total_brightness_min": float(frames.sum(axis=(1, 2)).min()),
        f"{label}_total_brightness_max": float(frames.sum(axis=(1, 2)).max()),
        f"{label}_total_brightness_mean": float(frames.sum(axis=(1, 2)).mean()),
    }
    if len(frames) >= 2:
        diffs = frames[1:] - frames[:-1]
        frame_norms = np.linalg.norm(frames.reshape(len(frames), -1), axis=1)
        diff_norms = np.linalg.norm(diffs.reshape(len(diffs), -1), axis=1)
        rows.update({
            f"{label}_adjacent_l2_min": float(diff_norms.min()),
            f"{label}_adjacent_l2_max": float(diff_norms.max()),
            f"{label}_adjacent_l2_mean": float(diff_norms.mean()),
            f"{label}_adjacent_relative_l2_mean": float(
                np.mean(diff_norms / (frame_norms[:-1] + 1e-12))
            ),
        })
    return rows


def residual_diagnostics(label, frames, background):
    positive = np.maximum(np.asarray(frames) - background[None, :, :], 0.0)
    negative = np.maximum(background[None, :, :] - np.asarray(frames), 0.0)
    positive_masses = positive.sum(axis=(1, 2))
    negative_masses = negative.sum(axis=(1, 2))
    rows = sequence_diagnostics(f"{label}_positive_residual", positive)
    rows.update(sequence_diagnostics(f"{label}_negative_residual", negative))
    rows.update({
        f"{label}_positive_residual_mass_min": float(positive_masses.min()),
        f"{label}_positive_residual_mass_max": float(positive_masses.max()),
        f"{label}_positive_residual_mass_mean": float(positive_masses.mean()),
        f"{label}_positive_residual_mass_relative_range": float(
            (positive_masses.max() - positive_masses.min()) / (positive_masses.mean() + 1e-12)
        ),
        f"{label}_negative_residual_mass_min": float(negative_masses.min()),
        f"{label}_negative_residual_mass_max": float(negative_masses.max()),
        f"{label}_negative_residual_mass_mean": float(negative_masses.mean()),
        f"{label}_negative_residual_mass_relative_range": float(
            (negative_masses.max() - negative_masses.min()) / (negative_masses.mean() + 1e-12)
        ),
    })
    return rows


def data_loss_stats(label, frames, data_terms):
    losses = np.asarray([term.loss(frames[k]) for k, term in enumerate(data_terms)])
    return {
        f"{label}_data_loss_min": float(losses.min()),
        f"{label}_data_loss_max": float(losses.max()),
        f"{label}_data_loss_mean": float(losses.mean()),
    }


def gt_comparison_diagnostics(label, frames, gt):
    if gt is None:
        return {}
    frames = np.asarray(frames, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    count = min(len(frames), len(gt))
    frames = frames[:count]
    gt = gt[:count]

    mse = np.mean((frames - gt) ** 2, axis=(1, 2))
    frame_flat = frames.reshape(count, -1)
    gt_flat = gt.reshape(count, -1)
    frame_centered = frame_flat - frame_flat.mean(axis=1, keepdims=True)
    gt_centered = gt_flat - gt_flat.mean(axis=1, keepdims=True)
    corr = np.sum(frame_centered * gt_centered, axis=1) / (
        np.linalg.norm(frame_centered, axis=1)
        * np.linalg.norm(gt_centered, axis=1)
        + 1e-30
    )

    scaled_mse = []
    for frame, truth in zip(frames, gt):
        scale = float(np.sum(frame * truth) / (np.sum(frame * frame) + 1e-30))
        scaled_mse.append(float(np.mean((scale * frame - truth) ** 2)))

    return {
        f"{label}_gt_mse_mean": float(mse.mean()),
        f"{label}_gt_mse_min": float(mse.min()),
        f"{label}_gt_mse_max": float(mse.max()),
        f"{label}_gt_corr_mean": float(corr.mean()),
        f"{label}_gt_corr_min": float(corr.min()),
        f"{label}_gt_corr_max": float(corr.max()),
        f"{label}_gt_scale_invariant_mse_mean": float(np.mean(scaled_mse)),
    }


def circular_angle_difference(a, b):
    return np.angle(np.exp(1j * (np.asarray(a) - np.asarray(b))))


def annulus_mask_from_background(background, cfg):
    background = np.asarray(background, dtype=np.float64)
    threshold = cfg.motion_annulus_threshold_fraction * float(background.max() + 1e-30)
    mask = background >= threshold
    if np.count_nonzero(mask) < 16:
        mask = background > 0.0
    return mask


def residual_hotspot_angles(frames, reference, mask, cfg):
    frames = np.asarray(frames, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    height, width = reference.shape
    y, x = np.indices(reference.shape)
    center_y = (height - 1) / 2.0
    center_x = (width - 1) / 2.0
    theta = np.arctan2(y - center_y, x - center_x)

    residual = np.maximum(frames - reference[None, :, :], 0.0)
    masked = residual * mask[None, :, :]
    masses = masked.sum(axis=(1, 2))
    min_mass = cfg.motion_min_residual_mass_fraction * float(reference.sum() + 1e-30)

    angles = np.full(len(frames), np.nan, dtype=np.float64)
    concentrations = np.full(len(frames), np.nan, dtype=np.float64)
    for k, mass in enumerate(masses):
        if mass <= min_mass:
            continue
        z = np.sum(masked[k] * np.exp(1j * theta))
        angles[k] = float(np.angle(z))
        concentrations[k] = float(np.abs(z) / (mass + 1e-30))
    return angles, concentrations, masses


def motion_diagnostics(label, frames, background, gt, cfg):
    if gt is None:
        return {}
    frames = np.asarray(frames, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    count = min(len(frames), len(gt))
    frames = frames[:count]
    gt = gt[:count]

    mask = annulus_mask_from_background(background, cfg)
    gt_reference = gt.mean(axis=0)
    recon_angles, recon_conc, recon_mass = residual_hotspot_angles(
        frames, background, mask, cfg
    )
    gt_angles, gt_conc, gt_mass = residual_hotspot_angles(
        gt, gt_reference, mask, cfg
    )
    valid = np.isfinite(recon_angles) & np.isfinite(gt_angles)

    def finite_mean(values):
        values = np.asarray(values, dtype=np.float64)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return np.nan
        return float(finite.mean())

    rows = {
        f"{label}_motion_valid_frames": int(np.count_nonzero(valid)),
        f"{label}_motion_recon_residual_mass_mean": float(np.mean(recon_mass)),
        f"{label}_motion_gt_residual_mass_mean": float(np.mean(gt_mass)),
        f"{label}_motion_recon_angle_concentration_mean": finite_mean(recon_conc),
        f"{label}_motion_gt_angle_concentration_mean": finite_mean(gt_conc),
    }
    if np.count_nonzero(valid) == 0:
        rows.update({
            f"{label}_motion_angle_error_rad_mean": np.nan,
            f"{label}_motion_angle_error_deg_mean": np.nan,
            f"{label}_motion_angle_circular_corr": np.nan,
            f"{label}_motion_delta_angle_error_rad_mean": np.nan,
            f"{label}_motion_delta_angle_error_deg_mean": np.nan,
        })
        return rows

    angle_error = np.abs(circular_angle_difference(recon_angles[valid], gt_angles[valid]))
    rows[f"{label}_motion_angle_error_rad_mean"] = float(np.mean(angle_error))
    rows[f"{label}_motion_angle_error_deg_mean"] = float(np.degrees(np.mean(angle_error)))

    # Circular correlation proxy: 1 is perfect angle match, 0 is unrelated on average.
    rows[f"{label}_motion_angle_circular_corr"] = float(
        np.mean(np.cos(circular_angle_difference(recon_angles[valid], gt_angles[valid])))
    )

    valid_pairs = valid[:-1] & valid[1:]
    if np.count_nonzero(valid_pairs) == 0:
        rows[f"{label}_motion_delta_angle_error_rad_mean"] = np.nan
        rows[f"{label}_motion_delta_angle_error_deg_mean"] = np.nan
    else:
        recon_delta = circular_angle_difference(recon_angles[1:], recon_angles[:-1])
        gt_delta = circular_angle_difference(gt_angles[1:], gt_angles[:-1])
        delta_error = np.abs(circular_angle_difference(recon_delta[valid_pairs], gt_delta[valid_pairs]))
        rows[f"{label}_motion_delta_angle_error_rad_mean"] = float(np.mean(delta_error))
        rows[f"{label}_motion_delta_angle_error_deg_mean"] = float(np.degrees(np.mean(delta_error)))
    return rows


def summary_row(label, u, data_terms, background, cfg, elapsed, converged, prior_weight):
    terms = objective_terms(u, data_terms, background, cfg.tv_weight, prior_weight)
    background_stack = np.repeat(background[None, :, :], len(u), axis=0)
    return {
        "stage": label,
        "converged": converged,
        "seconds": elapsed,
        "relative_change_from_background": relative_change(u, background_stack),
        **terms,
    }


def json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def serializable_config(cfg):
    values = asdict(cfg)
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in values.items()
    }


def save_results_file(
    cfg,
    names,
    static_paths,
    background,
    static,
    ot,
    gt,
    gt_names,
    ot_history,
    static_summary,
    init_summary,
    ot_summary,
    diagnostics,
):
    output_path = cfg.output_root / "results.npz"
    np.savez_compressed(
        output_path,
        names=np.asarray(names),
        static_recon_paths=np.asarray([str(path) for path in static_paths]),
        gt_names=np.asarray(gt_names),
        background=background,
        background_video=np.repeat(background[None, :, :], len(static), axis=0),
        static=static,
        joint_initialization=static if cfg.initialization_mode == "static" else np.repeat(background[None, :, :], len(static), axis=0),
        joint=ot,
        static_positive_residual=np.maximum(static - background[None, :, :], 0.0),
        static_negative_residual=np.maximum(background[None, :, :] - static, 0.0),
        joint_positive_residual=np.maximum(ot - background[None, :, :], 0.0),
        joint_negative_residual=np.maximum(background[None, :, :] - ot, 0.0),
        gt=np.asarray([]) if gt is None else gt,
        config_json=np.asarray(json.dumps(serializable_config(cfg), indent=2, default=json_default)),
        summary_json=np.asarray(json.dumps([static_summary, init_summary, ot_summary], indent=2, default=json_default)),
        diagnostics_json=np.asarray(json.dumps(diagnostics, indent=2, default=json_default)),
        ot_history_json=np.asarray(json.dumps(ot_history, indent=2, default=json_default)),
    )
    return output_path


def main():
    cfg = Config()
    cfg.output_root.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("MAIN8.PY: JOINT DATA + TV + SIGNED-RESIDUAL UOT ADMM")
    print("Outputs are intentionally minimal: results.npz and one comparison video.")
    print("=" * 100)
    print("Static frames define the background/baseline and initialize the joint solve.")
    print("Residual model: h+ = max(u-a,0), h- = max(a-u,0); UOT acts on both.")

    data_terms, names = load_data_terms(cfg)
    print(f"Loaded {len(data_terms)} observation frames:")
    print("  " + ", ".join(str(name) for name in names))

    print("\n" + "=" * 100)
    print("1. Load static L1+TV+data baseline and compute average background")
    print("=" * 100)
    started = time.perf_counter()
    u_static, background, static_paths, scale_info = load_static_initialization(
        cfg,
        names,
        data_terms,
    )
    static_elapsed = time.perf_counter() - started
    static_summary = summary_row(
        "external_static_l1_tv_data_baseline",
        u_static,
        data_terms,
        background,
        cfg,
        static_elapsed,
        True,
        cfg.prior_weight,
    )
    static_summary.update({
        "static_recon_folder": str(cfg.static_recon_folder),
        "static_recon_paths": [str(path) for path in static_paths],
        **scale_info,
    })
    print(
        f"Loaded/calibrated static baseline in {static_elapsed:.2f}s | "
        f"data={static_summary['data']:.3e} | "
        f"rel_from_background={static_summary['relative_change_from_background']:.3e}"
    )
    background_video = np.repeat(background[None, :, :], len(data_terms), axis=0)
    if cfg.initialization_mode == "static":
        u_init = u_static.copy()
        init_label = "static_initialization_for_joint_solve"
    elif cfg.initialization_mode == "background":
        u_init = background_video.copy()
        init_label = "repeated_background_initialization"
    else:
        raise ValueError("initialization_mode must be 'static' or 'background'")
    init_summary = summary_row(
        init_label,
        u_init,
        data_terms,
        background,
        cfg,
        0.0,
        True,
        cfg.prior_weight,
    )
    print(
        f"Joint solve initialization: {cfg.initialization_mode} | "
        f"data={init_summary['data']:.3e} | "
        f"rel_from_background={init_summary['relative_change_from_background']:.3e}"
    )

    print("\n" + "=" * 100)
    print("2. Joint data + TV + signed-residual unbalanced BB/ADMM solve")
    print("=" * 100)
    u_ot, ot_history, ot_elapsed, ot_converged = run_signed_residual_uot(
        u_init,
        data_terms,
        background,
        cfg,
    )
    ot_summary = summary_row(
        "joint_data_tv_signed_residual_uot",
        u_ot,
        data_terms,
        background,
        cfg,
        ot_elapsed,
        ot_converged,
        cfg.prior_weight,
    )

    gt, gt_names = load_ground_truth_frames(
        cfg.gt_folder,
        names,
        resize=(cfg.image_width, cfg.image_height),
    )
    video_path = save_comparison_video(gt, u_static, u_ot, cfg.output_root, cfg.fps)

    diagnostics = {
        **scale_info,
        **sequence_diagnostics("background_init", background_video),
        **sequence_diagnostics("joint_init", u_init),
        **sequence_diagnostics("static", u_static),
        **sequence_diagnostics("joint", u_ot),
        **residual_diagnostics("static", u_static, background),
        **residual_diagnostics("joint", u_ot, background),
        **data_loss_stats("background", background_video, data_terms),
        **data_loss_stats("joint_init", u_init, data_terms),
        **data_loss_stats("static", u_static, data_terms),
        **data_loss_stats("joint", u_ot, data_terms),
        **gt_comparison_diagnostics("background", background_video, gt),
        **gt_comparison_diagnostics("static", u_static, gt),
        **gt_comparison_diagnostics("joint", u_ot, gt),
        **motion_diagnostics("background", background_video, background, gt, cfg),
        **motion_diagnostics("static", u_static, background, gt, cfg),
        **motion_diagnostics("joint", u_ot, background, gt, cfg),
    }
    diagnostics["joint_vs_static_data_loss_ratio_mean"] = (
        diagnostics["joint_data_loss_mean"] / (diagnostics["static_data_loss_mean"] + 1e-30)
    )
    diagnostics["joint_vs_initialization_data_loss_ratio_mean"] = (
        diagnostics["joint_data_loss_mean"] / (diagnostics["joint_init_data_loss_mean"] + 1e-30)
    )
    diagnostics["joint_vs_background_data_loss_ratio_mean"] = (
        diagnostics["joint_data_loss_mean"] / (diagnostics["background_data_loss_mean"] + 1e-30)
    )

    results_path = save_results_file(
        cfg=cfg,
        names=names,
        static_paths=static_paths,
        background=background,
        static=u_static,
        ot=u_ot,
        gt=gt,
        gt_names=gt_names,
        ot_history=ot_history,
        static_summary=static_summary,
        init_summary=init_summary,
        ot_summary=ot_summary,
        diagnostics=diagnostics,
    )

    print("\nSummary:")
    print(json.dumps([static_summary, init_summary, ot_summary], indent=2, default=json_default))
    print("\nDiagnostics:")
    print(json.dumps(diagnostics, indent=2, default=json_default))
    print(f"\nOutputs saved to {cfg.output_root.resolve()}")
    print(f"  Results: {results_path}")
    if video_path is not None:
        print(f"  Comparison video: {video_path}")


if __name__ == "__main__":
    main()
