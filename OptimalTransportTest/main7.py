"""External static initialization followed by signed-residual UOT ADMM.

Workflow:

    1. Load per-frame static reconstructions from
       PROJECT_ROOT/static_reconstruction/reconstructed_frames_gray.
    2. Calibrate each static image to its frame's visibility scale.
    3. Use their average as a fixed background image a.
    4. Run ADMM with the data term on u_k, but unbalanced BB transport on

           h_k^+ = max(u_k - a, 0),
           h_k^- = max(a - u_k, 0).

This is the first version where residual mass can be created/destroyed through
a source term in the continuity equation, so it is appropriate when flare
brightness changes across frames.
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
    output_root: Path = Path("main7_signed_residual_uot_results")
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

    # Signed residual UOT image objective.
    tv_weight: float = 0.0
    prior_weight: float = 1e-4
    beta: float = 1e-4
    eta: float = 1e-3
    uot_source_weight: float = 10.0

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
    output_path = root / "comparison_gt_static_signed_uot.mp4"
    first = make_labeled_panel([gt[0], static[0], ot[0]], ["GT", "static", "signed UOT"])
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (first.shape[1], first.shape[0]),
    )
    writer.write(first)
    for k in range(1, min(len(gt), len(static), len(ot))):
        writer.write(make_labeled_panel([gt[k], static[k], ot[k]], ["GT", "static", "signed UOT"]))
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
        static=static,
        ot=ot,
        static_positive_residual=np.maximum(static - background[None, :, :], 0.0),
        static_negative_residual=np.maximum(background[None, :, :] - static, 0.0),
        ot_positive_residual=np.maximum(ot - background[None, :, :], 0.0),
        ot_negative_residual=np.maximum(background[None, :, :] - ot, 0.0),
        gt=np.asarray([]) if gt is None else gt,
        config_json=np.asarray(json.dumps(serializable_config(cfg), indent=2, default=json_default)),
        summary_json=np.asarray(json.dumps([static_summary, ot_summary], indent=2, default=json_default)),
        diagnostics_json=np.asarray(json.dumps(diagnostics, indent=2, default=json_default)),
        ot_history_json=np.asarray(json.dumps(ot_history, indent=2, default=json_default)),
    )
    return output_path


def main():
    cfg = Config()
    cfg.output_root.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("MAIN7.PY: EXTERNAL STATIC INIT -> SIGNED-RESIDUAL UOT ADMM")
    print("Outputs are intentionally minimal: results.npz and one comparison video.")
    print("=" * 100)
    print("Residual model: h+ = max(u-a,0), h- = max(a-u,0); UOT acts on both.")

    data_terms, names = load_data_terms(cfg)
    print(f"Loaded {len(data_terms)} observation frames:")
    print("  " + ", ".join(str(name) for name in names))

    print("\n" + "=" * 100)
    print("1. External static L1+TV+data initialization")
    print("=" * 100)
    started = time.perf_counter()
    u_static, background, static_paths, scale_info = load_static_initialization(
        cfg,
        names,
        data_terms,
    )
    static_elapsed = time.perf_counter() - started
    static_summary = summary_row(
        "external_static_l1_tv_data_initialization",
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
        f"Loaded/calibrated static frames in {static_elapsed:.2f}s | "
        f"data={static_summary['data']:.3e} | "
        f"rel_from_background={static_summary['relative_change_from_background']:.3e}"
    )

    print("\n" + "=" * 100)
    print("2. Signed-residual unbalanced BB/ADMM OT refinement")
    print("=" * 100)
    u_ot, ot_history, ot_elapsed, ot_converged = run_signed_residual_uot(
        u_static,
        data_terms,
        background,
        cfg,
    )
    ot_summary = summary_row(
        "signed_residual_unbalanced_bb_admm_ot",
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
        **sequence_diagnostics("background_init", np.repeat(background[None, :, :], len(u_static), axis=0)),
        **sequence_diagnostics("static", u_static),
        **sequence_diagnostics("ot", u_ot),
        **residual_diagnostics("static", u_static, background),
        **residual_diagnostics("ot", u_ot, background),
        **data_loss_stats("static", u_static, data_terms),
        **data_loss_stats("ot", u_ot, data_terms),
    }
    diagnostics["ot_vs_static_data_loss_ratio_mean"] = (
        diagnostics["ot_data_loss_mean"] / (diagnostics["static_data_loss_mean"] + 1e-30)
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
        ot_summary=ot_summary,
        diagnostics=diagnostics,
    )

    print("\nSummary:")
    print(json.dumps([static_summary, ot_summary], indent=2, default=json_default))
    print("\nDiagnostics:")
    print(json.dumps(diagnostics, indent=2, default=json_default))
    print(f"\nOutputs saved to {cfg.output_root.resolve()}")
    print(f"  Results: {results_path}")
    if video_path is not None:
        print(f"  Comparison video: {video_path}")


if __name__ == "__main__":
    main()
