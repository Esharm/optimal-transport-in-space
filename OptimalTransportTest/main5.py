"""Static L1+TV frame initialization followed by full-image balanced BB/ADMM OT.

This main5.py implements the current two-stage experimental plan:

    1. Load independently reconstructed static frames from

           PROJECT_ROOT / "static_reconstruction" / "reconstructed_frames_gray"

       with names such as

           recon_0000_frame_000.png, recon_0001_frame_001.png, ...

       These static reconstructions are assumed to have been produced outside
       this file from a per-frame objective like

           min_{u_k >= 0} D_k(u_k) + lambda_1 ||u_k||_1 + alpha TV(u_k).

       The loaded images are brightness-calibrated against their corresponding
       visibility data and used directly as the OT initialization u_k^(0).

    2. Define the static average image

           p = (1/K) sum_k u_k^(0)

       and run the existing full-image balanced Benamou-Brenier ADMM solver:

           sum_k [D_k(u_k) + alpha TV(u_k) + (mu/2)||u_k - p||_2^2]
           + beta sum_k BB(u_k, u_{k+1}),
           u_k >= 0.

This file intentionally does NOT use radial-prior warmup, residual-channel OT,
ring-basis fitting, hard visibility matching, or radial clipping. It is a clean
balanced-OT refinement of externally supplied static reconstructions.
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

from admm import ADMM
from run_three_frame_ablation import Config as LoaderConfig
from run_three_frame_ablation import load_data_terms, objective_terms, relative_change
from solvers import TotalVariationRegularizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Config(LoaderConfig):
    output_root: Path = Path("main5_static_ot_results")
    static_recon_folder: Path = (
        PROJECT_ROOT / "static_reconstruction" / "reconstructed_frames_gray"
    )
    gt_folder: Path = PROJECT_ROOT / "blackhole_sim" / "data" / "aart_frames"
    fps: int = 5

    # Use all first 15 frames by default. Set frame_indices=(0, 7, 14) for tests.
    frames: int = 15
    frame_indices: tuple[int, ...] | None = None

    # Memory-safe visibility evaluation for thousands of visibilities/frame.
    max_vis_per_frame: int | None = None
    use_visibility_cache: bool = False
    visibility_chunk_size: int = 128
    parallel_frames: bool = False

    # Static reconstruction loading/calibration.
    # If the static frames were saved as display-normalized PNGs, per_frame is
    # usually safest. If they were saved with a physically meaningful common
    # scale, set static_recon_scale_mode="none" or "global".
    static_recon_scale_mode: str = "per_frame"  # "none", "global", or "per_frame"
    static_recon_normalize_input: bool = True
    static_recon_enforce_nonnegative: bool = True

    # Balanced full-image BB/ADMM OT refinement.
    tv_weight: float = 0.0
    prior_weight: float = 1e-4
    beta: float = 1e-4
    eta: float = 1e-3

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


def _frame_number_from_name(name) -> int | None:
    """Extract the frame number from strings such as frame_007.npz."""
    stem = Path(str(name)).stem
    numbers = re.findall(r"\d+", stem)
    if not numbers:
        return None
    return int(numbers[-1])


def _candidate_frame_number(path: Path) -> int | None:
    """Extract the intended frame number from recon_0001_frame_001.png.

    The static recon filename has two numbers; the last one is the frame index.
    """
    numbers = re.findall(r"\d+", path.stem)
    if not numbers:
        return None
    return int(numbers[-1])


def _load_grayscale_image(path: Path, resize, normalize_input: bool) -> np.ndarray:
    image = Image.open(path).convert("L").resize(resize)
    array = np.asarray(image, dtype=np.float64)
    if normalize_input:
        array = array / 255.0
    if not np.all(np.isfinite(array)):
        raise ValueError(f"Static reconstruction contains nonfinite values: {path}")
    return array


def find_static_reconstruction_paths(folder: Path, names) -> list[Path]:
    """Match external static reconstruction images to observation frame names."""
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Static reconstruction folder does not exist: {folder}")

    image_files = sorted(
        p for p in folder.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    )
    if not image_files:
        raise FileNotFoundError(f"No static reconstruction images found in: {folder}")

    by_frame_number: dict[int, Path] = {}
    for path in image_files:
        number = _candidate_frame_number(path)
        if number is not None and number not in by_frame_number:
            by_frame_number[number] = path

    selected: list[Path | None] = []
    for name in names:
        frame_number = _frame_number_from_name(name)
        selected.append(by_frame_number.get(frame_number) if frame_number is not None else None)

    if any(path is None for path in selected):
        if len(image_files) < len(names):
            missing = [str(name) for name, path in zip(names, selected) if path is None]
            raise RuntimeError(
                "Could not match enough static reconstructions to observations. "
                f"Missing matches for: {missing}. Found only {len(image_files)} images."
            )
        print("Could not match all static reconstructions by frame number; using first K sorted images.")
        selected = image_files[: len(names)]

    return [Path(path) for path in selected]


def calibrate_static_reconstructions(frames, data_terms, mode: str):
    """Brightness-calibrate static reconstructions against visibility data.

    mode="none": return frames unchanged.
    mode="global": one least-squares scale shared by all frames.
    mode="per_frame": one least-squares scale per frame.
    """
    frames = np.asarray(frames, dtype=np.float64)
    mode = str(mode).lower()
    if mode not in {"none", "global", "per_frame"}:
        raise ValueError("static_recon_scale_mode must be 'none', 'global', or 'per_frame'")

    if mode == "none":
        scales = np.ones(len(frames), dtype=np.float64)
        return frames.copy(), {
            "static_recon_scale_mode": mode,
            "static_recon_scale_min": 1.0,
            "static_recon_scale_max": 1.0,
            "static_recon_scale_mean": 1.0,
        }

    numerators = []
    denominators = []
    for frame, term in zip(frames, data_terms):
        predicted = term.sampler.forward(frame)
        numerators.append(float(np.real(np.vdot(predicted, term.f))))
        denominators.append(float(np.vdot(predicted, predicted).real))
    numerators = np.asarray(numerators, dtype=np.float64)
    denominators = np.asarray(denominators, dtype=np.float64)

    if mode == "global":
        scale = max(float(numerators.sum() / (denominators.sum() + 1e-30)), 0.0)
        scales = np.full(len(frames), scale, dtype=np.float64)
    else:
        scales = np.maximum(numerators / (denominators + 1e-30), 0.0)

    calibrated = frames * scales[:, None, None]
    return calibrated, {
        "static_recon_scale_mode": mode,
        "static_recon_scale_min": float(scales.min()),
        "static_recon_scale_max": float(scales.max()),
        "static_recon_scale_mean": float(scales.mean()),
        "static_recon_scale_std": float(scales.std()),
        "static_recon_scale_numerator_sum": float(numerators.sum()),
        "static_recon_scale_denominator_sum": float(denominators.sum()),
    }


def load_static_initialization(data_terms, names, cfg):
    """Load, match, brightness-calibrate, and summarize static recon frames."""
    started = time.perf_counter()
    paths = find_static_reconstruction_paths(cfg.static_recon_folder, names)
    resize = (cfg.image_width, cfg.image_height)
    raw = np.stack([
        _load_grayscale_image(path, resize=resize, normalize_input=cfg.static_recon_normalize_input)
        for path in paths
    ])
    if cfg.static_recon_enforce_nonnegative:
        raw = np.maximum(raw, 0.0)

    static, scale_info = calibrate_static_reconstructions(
        raw,
        data_terms,
        mode=cfg.static_recon_scale_mode,
    )
    if cfg.static_recon_enforce_nonnegative:
        static = np.maximum(static, 0.0)

    average_prior = np.mean(static, axis=0)
    elapsed = time.perf_counter() - started
    losses = np.asarray([term.loss(static[k]) for k, term in enumerate(data_terms)])
    avg_losses = np.asarray([term.loss(average_prior) for term in data_terms])
    prior_stack = np.repeat(average_prior[None, :, :], len(static), axis=0)

    history = [{
        "stage": "external_static_l1_tv_data_initialization",
        "seconds": elapsed,
        "static_recon_folder": str(cfg.static_recon_folder),
        "static_recon_paths": [str(path) for path in paths],
        "data_loss_mean": float(losses.mean()),
        "data_loss_min": float(losses.min()),
        "data_loss_max": float(losses.max()),
        "average_prior_data_loss_mean": float(avg_losses.mean()),
        "relative_change_from_average_prior": relative_change(static, prior_stack),
        **scale_info,
    }]

    print(
        "static init | "
        f"loaded={len(static)} | seconds={elapsed:.2f} "
        f"| data_mean={losses.mean():.3e} "
        f"| avg_prior_data_mean={avg_losses.mean():.3e} "
        f"| rel_to_avg={history[-1]['relative_change_from_average_prior']:.3e}"
    )
    print("Static recon paths:")
    for path in paths:
        print(f"  {path.name}")

    return static, average_prior, history, elapsed, True, paths, scale_info


def run_full_image_ot(u_init, data_terms, average_prior, cfg):
    regularizer = TotalVariationRegularizer(
        alpha=cfg.tv_weight,
        iters=cfg.ot_image_inner_iters,
        tau=cfg.primal_tau,
        sigma=cfg.dual_sigma,
        power_iters=cfg.ot_power_iters,
    )
    model = ADMM(
        data_terms=data_terms,
        regularizer=regularizer,
        beta=cfg.beta,
        eta=cfg.eta,
        max_iter=cfg.ot_max_iter,
        abs_tol=1e-4,
        rel_tol=5e-3,
        min_iter=cfg.ot_min_iter,
        patience=cfg.ot_patience,
        transport_T=cfg.transport_slices,
        transport_inner_iters=cfg.transport_inner_iters,
        transport_tol=cfg.transport_tol,
        dual_relaxation=1.0,
        enforce_equal_mass=False,
        prior_image=average_prior,
        prior_weight=cfg.prior_weight,
        stop_on_data_plateau=False,
    )

    print(
        "Starting full-image balanced BB/ADMM OT "
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


def _normalize01_local(x):
    x = np.asarray(x, dtype=np.float64)
    return (x - x.min()) / (x.max() - x.min() + 1e-12)


def _to_uint8(frame, vmin=None, vmax=None):
    frame = np.asarray(frame, dtype=np.float64)
    if vmin is None:
        vmin = float(frame.min())
    if vmax is None:
        vmax = float(frame.max())
    scaled = (frame - vmin) / (vmax - vmin + 1e-12)
    return np.clip(255.0 * scaled, 0, 255).astype(np.uint8)


def make_labeled_panel(images, labels):
    panels = []
    for image, label in zip(images, labels):
        gray = _to_uint8(_normalize01_local(image))
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
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
    if not image_files:
        print(f"No ground-truth images found, skipping GT comparison: {folder}")
        return None, []

    by_stem = {p.stem: p for p in image_files}
    selected = []
    for name in names:
        stem = Path(str(name)).stem
        if stem in by_stem:
            selected.append(by_stem[stem])
            continue
        number = _frame_number_from_name(name)
        match = None
        if number is not None:
            for candidate in image_files:
                candidate_number = _candidate_frame_number(candidate)
                if candidate_number == number:
                    match = candidate
                    break
        selected.append(match)

    if any(item is None for item in selected):
        print("Could not match GT names exactly; using first K sorted GT images instead.")
        selected = image_files[: len(names)]

    frames = []
    for path in selected:
        img = Image.open(path).convert("L").resize(resize)
        frames.append(np.asarray(img, dtype=np.float64) / 255.0)
    return np.stack(frames), [path.name for path in selected]


def save_comparison_video(gt, static, ot, root: Path, fps: int):
    if gt is None:
        print("No ground truth available; skipping comparison video.")
        return None
    root.mkdir(parents=True, exist_ok=True)
    output_path = root / "comparison_gt_static_ot.mp4"
    frames = min(len(gt), len(static), len(ot))
    first = make_labeled_panel([gt[0], static[0], ot[0]], ["GT", "static L1+TV", "OT"])
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (first.shape[1], first.shape[0]),
    )
    writer.write(first)
    for k in range(1, frames):
        writer.write(make_labeled_panel([gt[k], static[k], ot[k]], ["GT", "static L1+TV", "OT"]))
    writer.release()
    return output_path


def sequence_pairwise_diagnostics(label, frames):
    frames = np.asarray(frames, dtype=np.float64)
    rows = {}
    if len(frames) < 2:
        return rows
    diffs = frames[1:] - frames[:-1]
    frame_norms = np.linalg.norm(frames.reshape(len(frames), -1), axis=1)
    diff_norms = np.linalg.norm(diffs.reshape(len(diffs), -1), axis=1)
    masses = frames.sum(axis=(1, 2))
    mean_mass = float(np.mean(masses))
    rows[f"{label}_adjacent_l2_min"] = float(diff_norms.min())
    rows[f"{label}_adjacent_l2_max"] = float(diff_norms.max())
    rows[f"{label}_adjacent_l2_mean"] = float(diff_norms.mean())
    rows[f"{label}_adjacent_relative_l2_mean"] = float(
        np.mean(diff_norms / (frame_norms[:-1] + 1e-12))
    )
    rows[f"{label}_global_min"] = float(frames.min())
    rows[f"{label}_global_max"] = float(frames.max())
    rows[f"{label}_total_brightness_min"] = float(masses.min())
    rows[f"{label}_total_brightness_max"] = float(masses.max())
    rows[f"{label}_total_brightness_mean"] = mean_mass
    rows[f"{label}_mass_relative_range"] = float(
        (masses.max() - masses.min()) / (mean_mass + 1e-12)
    )
    if len(masses) > 1:
        adjacent_mass_change = np.abs(masses[1:] - masses[:-1]) / (
            0.5 * (masses[1:] + masses[:-1]) + 1e-12
        )
        rows[f"{label}_adjacent_mass_change_min"] = float(adjacent_mass_change.min())
        rows[f"{label}_adjacent_mass_change_max"] = float(adjacent_mass_change.max())
        rows[f"{label}_adjacent_mass_change_mean"] = float(adjacent_mass_change.mean())
    return rows


def data_force_diagnostics(reference, data_terms, label="average_prior"):
    losses = np.asarray([term.loss(reference) for term in data_terms])
    gradients = np.stack([term.gradient(reference) for term in data_terms])
    gradient_norms = np.linalg.norm(gradients.reshape(len(data_terms), -1), axis=1)
    return {
        f"data_loss_at_{label}_min": float(losses.min()),
        f"data_loss_at_{label}_max": float(losses.max()),
        f"data_loss_at_{label}_mean": float(losses.mean()),
        f"data_gradient_at_{label}_l2_min": float(gradient_norms.min()),
        f"data_gradient_at_{label}_l2_max": float(gradient_norms.max()),
        f"data_gradient_at_{label}_l2_mean": float(gradient_norms.mean()),
    }


def static_data_loss_diagnostics(static, average_prior, ot, data_terms):
    static_losses = np.asarray([term.loss(static[k]) for k, term in enumerate(data_terms)])
    ot_losses = np.asarray([term.loss(ot[k]) for k, term in enumerate(data_terms)])
    avg_losses = np.asarray([term.loss(average_prior) for term in data_terms])
    return {
        "static_data_loss_min": float(static_losses.min()),
        "static_data_loss_max": float(static_losses.max()),
        "static_data_loss_mean": float(static_losses.mean()),
        "average_prior_data_loss_mean": float(avg_losses.mean()),
        "ot_data_loss_min": float(ot_losses.min()),
        "ot_data_loss_max": float(ot_losses.max()),
        "ot_data_loss_mean": float(ot_losses.mean()),
        "ot_vs_static_data_loss_ratio_mean": float(ot_losses.mean() / (static_losses.mean() + 1e-30)),
    }


def summary_row(label, u, data_terms, average_prior, cfg, elapsed, converged):
    terms = objective_terms(u, data_terms, average_prior, cfg.tv_weight, cfg.prior_weight)
    prior_stack = np.repeat(average_prior[None, :, :], len(u), axis=0)
    return {
        "stage": label,
        "converged": converged,
        "seconds": elapsed,
        "relative_change_from_average_prior": relative_change(u, prior_stack),
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
    average_prior,
    static,
    ot,
    gt,
    gt_names,
    static_history,
    ot_history,
    static_summary,
    ot_summary,
    diagnostics,
):
    output_path = cfg.output_root / "results.npz"
    np.savez_compressed(
        output_path,
        names=np.asarray(names),
        static_paths=np.asarray([str(path) for path in static_paths]),
        gt_names=np.asarray(gt_names),
        average_prior=average_prior,
        static=static,
        ot=ot,
        static_minus_average=static - average_prior[None, :, :],
        ot_minus_average=ot - average_prior[None, :, :],
        gt=np.asarray([]) if gt is None else gt,
        config_json=np.asarray(json.dumps(serializable_config(cfg), indent=2, default=json_default)),
        summary_json=np.asarray(json.dumps([static_summary, ot_summary], indent=2, default=json_default)),
        diagnostics_json=np.asarray(json.dumps(diagnostics, indent=2, default=json_default)),
        static_history_json=np.asarray(json.dumps(static_history, indent=2, default=json_default)),
        ot_history_json=np.asarray(json.dumps(ot_history, indent=2, default=json_default)),
    )
    return output_path


def main():
    cfg = Config()
    cfg.output_root.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("MAIN5.PY: STATIC L1+TV RECONSTRUCTIONS -> FULL-IMAGE BALANCED BB/ADMM OT")
    print("Outputs are intentionally minimal: results.npz and one comparison video.")
    print("=" * 100)
    print("Process:")
    print("  1. Load per-frame static reconstructions from:")
    print(f"     {cfg.static_recon_folder}")
    print("  2. Brightness-calibrate them against the corresponding visibility data.")
    print("  3. Use their image average as the OT-stage prior/average image term.")
    print("  4. Run full-image balanced BB/ADMM with data + TV + average-prior term.")
    print("  No radial prior, no warmup, no residual-channel OT, no unbalanced OT.")

    data_terms, names = load_data_terms(cfg)
    print(f"Loaded {len(data_terms)} observation frames:")
    print("  " + ", ".join(str(name) for name in names))

    print("\n" + "=" * 100)
    print("1. External static L1+TV+data initialization")
    print("=" * 100)
    (
        u_static,
        average_prior,
        static_history,
        static_elapsed,
        static_converged,
        static_paths,
        scale_info,
    ) = load_static_initialization(data_terms, names, cfg)

    static_summary = summary_row(
        "external_static_l1_tv_data_initialization",
        u_static,
        data_terms,
        average_prior,
        cfg,
        static_elapsed,
        static_converged,
    )
    static_summary.update(static_history[-1])

    print("\n" + "=" * 100)
    print("2. Full-image balanced BB/ADMM OT refinement")
    print("=" * 100)
    u_ot, ot_history, ot_elapsed, ot_converged = run_full_image_ot(
        u_static,
        data_terms,
        average_prior,
        cfg,
    )
    ot_summary = summary_row(
        "full_image_balanced_bb_admm_ot",
        u_ot,
        data_terms,
        average_prior,
        cfg,
        ot_elapsed,
        ot_converged,
    )

    gt, gt_names = load_ground_truth_frames(
        cfg.gt_folder,
        names,
        resize=(cfg.image_width, cfg.image_height),
    )
    video_path = save_comparison_video(gt, u_static, u_ot, cfg.output_root, cfg.fps)

    diagnostics = {
        **scale_info,
        **data_force_diagnostics(average_prior, data_terms, label="average_prior"),
        **sequence_pairwise_diagnostics("average_prior_init", np.repeat(average_prior[None, :, :], len(data_terms), axis=0)),
        **sequence_pairwise_diagnostics("static", u_static),
        **sequence_pairwise_diagnostics("ot", u_ot),
        **static_data_loss_diagnostics(u_static, average_prior, u_ot, data_terms),
    }

    mass_change = diagnostics.get("static_adjacent_mass_change_max", 0.0)
    diagnostics["balanced_ot_mass_variation_warning"] = bool(mass_change > 0.15)
    diagnostics["balanced_ot_mass_variation_note"] = (
        "Large adjacent mass changes make balanced OT less appropriate; consider UOT."
        if mass_change > 0.15
        else "Adjacent mass changes are modest enough that balanced OT is a reasonable baseline."
    )

    results_path = save_results_file(
        cfg=cfg,
        names=names,
        static_paths=static_paths,
        average_prior=average_prior,
        static=u_static,
        ot=u_ot,
        gt=gt,
        gt_names=gt_names,
        static_history=static_history,
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
