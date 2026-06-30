"""TV+prior initialization followed by residual signed OT/ADMM.

FULL_15_FRAME_VIDEO_COMPARISON_VERSION. This main2.py is intentionally configured
to run all first 15 observation frames and export ground-truth comparison videos.

This is the current experimental pipeline:

    1. Independently reconstruct each selected frame with

           D_k(u_k) + alpha TV(u_k) + (mu/2)||u_k - p||_2^2,

       where p is the averaged-image prior.

    2. Use those non-identical spatial reconstructions as the initialization
       for an ADMM solve over the full image u_k with residual

           delta_k = u_k - p,
           delta_k^+ = max(delta_k, 0),
           delta_k^- = max(-delta_k, 0).

       The temporal term is balanced BB on the nonnegative residual channels:

           beta sum_k [BB(delta_k^+, delta_{k+1}^+)
                       + BB(delta_k^-, delta_{k+1}^-)].

       The image step still contains data + TV + prior, plus nonlinear
       ADMM penalties tying (u-p)_+ and (p-u)_+ to transported endpoints.

Important: this is balanced residual-channel OT, not unbalanced OT. If
residual masses change, the balanced BB endpoint variables form a soft
equal-mass compromise through the ADMM endpoint penalties.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from admm import ResidualSignedOTADMM
from io_utils import load_prior_image, save_frames, images_to_video
from run_three_frame_ablation import Config as LoaderConfig
from run_three_frame_ablation import load_data_terms, objective_terms, relative_change
from run_tv_prior_delta_visualization import (
    pairwise_delta_differences,
    pairwise_distance_rows,
    save_montage,
    save_run_outputs,
    save_signed_frames,
)
from solvers import TotalVariationRegularizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
# FULL_15_FRAME_VIDEO_COMPARISON_VERSION
class Config(LoaderConfig):
    output_root: Path = Path("full_15_frame_residual_ot_comparison")
    gt_folder: Path = PROJECT_ROOT / "blackhole_sim" / "data" / "aart_frames"
    fps: int = 5

    # Full 15-frame run for video comparison. Set this back to e.g. (0, 7, 14)
    # for a faster diagnostic subset.
    frames: int = 15
    frame_indices: tuple[int, ...] | None = None

    # The 15-frame comparison uses the stronger prior because weaker priors
    # lost the ring morphology under sparse per-frame data. TV is off by default
    # because the averaged prior was already produced with L1+TV and extra TV
    # did not visibly change the result.
    prior_weight: float = 5e-2
    tv_weight: float = 0.0

    # Spatial TV+prior initialization.
    spatial_inner_iters: int = 25
    spatial_max_blocks: int = 50
    spatial_min_blocks: int = 3
    spatial_patience: int = 3
    objective_rel_tol: float = 1e-5
    iterate_rel_tol: float = 1e-4
    primal_tau: float = 10.0
    dual_sigma: float = 0.25

    # Balanced OT on positive/negative residual channels.
    # If it does almost nothing, try beta=3e-6. If it erases frame-specific
    # residuals or primal residual grows badly, reduce beta or eta.
    beta: float = 1e-6
    eta: float = 1e-2
    ot_max_iter: int = 40
    ot_min_iter: int = 8
    ot_patience: int = 3
    transport_slices: int = 7
    transport_inner_iters: int = 200
    transport_tol: float = 2e-4

    parallel_frames: bool = True
    delta_amplification: float = 6.0


def run_spatial_tv_prior(u_init, data_terms, prior, cfg):
    regularizer = TotalVariationRegularizer(
        alpha=cfg.tv_weight,
        iters=cfg.spatial_inner_iters,
        tau=cfg.primal_tau,
        sigma=cfg.dual_sigma,
    )

    u = np.asarray(u_init, dtype=np.float64).copy()
    history = []
    stable = 0
    previous_objective = None
    started = time.perf_counter()

    for block in range(1, cfg.spatial_max_blocks + 1):
        old = u.copy()

        def update_frame(k):
            return regularizer.solve(
                u_init=u[k],
                data_term=data_terms[k],
                admm_target=prior,
                admm_weight=cfg.prior_weight,
                target_mass=None,
            )

        if cfg.parallel_frames and len(data_terms) > 1:
            with ThreadPoolExecutor(max_workers=len(data_terms)) as executor:
                u = np.stack(list(executor.map(update_frame, range(len(data_terms)))))
        else:
            for k in range(len(data_terms)):
                u[k] = update_frame(k)

        terms = objective_terms(u, data_terms, prior, cfg.tv_weight, cfg.prior_weight)
        du = relative_change(u, old)
        if previous_objective is None:
            dobj = np.inf
        else:
            dobj = abs(terms["spatial_total"] - previous_objective) / (
                abs(previous_objective) + 1e-12
            )
        previous_objective = terms["spatial_total"]

        row = {
            "block": block,
            "updates": block * cfg.spatial_inner_iters,
            "relative_iterate_change": du,
            "relative_objective_change": dobj,
            **terms,
        }
        history.append(row)

        print(
            f"spatial block {block:03d} | objective={terms['spatial_total']:.6e} "
            f"| data={terms['data']:.3e} | prior={terms['prior']:.3e} "
            f"| d_obj={dobj:.3e} | d_u={du:.3e}"
        )

        is_stable = dobj < cfg.objective_rel_tol and du < cfg.iterate_rel_tol
        stable = stable + 1 if block >= cfg.spatial_min_blocks and is_stable else 0
        if stable >= cfg.spatial_patience:
            print(f"Spatial TV+prior converged after {block * cfg.spatial_inner_iters} updates")
            break

    elapsed = time.perf_counter() - started
    return u, history, elapsed, stable >= cfg.spatial_patience


def run_residual_signed_ot_from_initialization(u_init, data_terms, prior, cfg):
    regularizer = TotalVariationRegularizer(
        alpha=cfg.tv_weight,
        iters=cfg.spatial_inner_iters,
        tau=cfg.primal_tau,
        sigma=cfg.dual_sigma,
    )

    model = ResidualSignedOTADMM(
        data_terms=data_terms,
        regularizer=regularizer,
        prior_image=prior,
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
        prior_weight=cfg.prior_weight,
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


def save_ot_difference(spatial, ot, prior, names, cfg):
    stage_dir = cfg.output_root / "03_ot_minus_spatial"
    stage_dir.mkdir(parents=True, exist_ok=True)

    diff = ot - spatial
    spatial_delta = spatial - prior[None, :, :]
    ot_delta = ot - prior[None, :, :]
    delta_diff = ot_delta - spatial_delta
    pairwise_diff, pairwise_labels = pairwise_delta_differences(delta_diff, names)

    common_abs = float(np.max(np.abs(diff)) + 1e-12)
    save_signed_frames(diff, stage_dir / "ot_minus_spatial_signed", names=names, common_abs=common_abs)
    save_signed_frames(delta_diff, stage_dir / "delta_change_signed", names=names, common_abs=common_abs)
    save_montage(
        diff,
        stage_dir / "montage_ot_minus_spatial_signed.png",
        [Path(name).stem for name in names],
        signed=True,
        common_abs=common_abs,
    )

    pairwise_abs = float(np.max(np.abs(pairwise_diff)) + 1e-12)
    save_signed_frames(
        pairwise_diff,
        stage_dir / "pairwise_delta_change_difference_signed",
        names=pairwise_labels,
        common_abs=pairwise_abs,
    )
    save_montage(
        pairwise_diff,
        stage_dir / "montage_pairwise_delta_change_difference_signed.png",
        pairwise_labels,
        signed=True,
        common_abs=pairwise_abs,
    )

    rows = pairwise_distance_rows(spatial_delta, "spatial_delta")
    rows += pairwise_distance_rows(ot_delta, "ot_delta")
    rows += pairwise_distance_rows(delta_diff, "ot_minus_spatial_delta")
    pd.DataFrame(rows).to_csv(stage_dir / "difference_distances.csv", index=False)

    np.savez_compressed(
        stage_dir / "ot_minus_spatial.npz",
        spatial=spatial,
        ot=ot,
        ot_minus_spatial=diff,
        spatial_delta=spatial_delta,
        ot_delta=ot_delta,
        delta_change=delta_diff,
    )



def _normalize01_local(x):
    x = np.asarray(x, dtype=np.float64)
    return (x - x.min()) / (x.max() - x.min() + 1e-12)


def load_ground_truth_frames(folder: Path, names, resize):
    """Load GT frames matching observation names when possible, else first K images."""
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
        else:
            # Common case: observation is frame_07.npz and GT is frame_007.png or similar.
            digits = "".join(ch for ch in stem if ch.isdigit())
            match = None
            if digits:
                number = int(digits)
                for candidate in image_files:
                    cdigits = "".join(ch for ch in candidate.stem if ch.isdigit())
                    if cdigits and int(cdigits) == number:
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


def save_sequence_videos(frames, root: Path, names, fps: int, prefix: str):
    root.mkdir(parents=True, exist_ok=True)
    each_dir = root / f"{prefix}_frames_each_norm"
    global_dir = root / f"{prefix}_frames_global_norm"
    save_frames(frames, each_dir, names=names, normalize_each=True)
    save_frames(frames, global_dir, names=names, normalize_each=False)
    try:
        images_to_video(each_dir, root / f"{prefix}_each_norm.mp4", fps=fps)
        images_to_video(global_dir, root / f"{prefix}_global_norm.mp4", fps=fps)
    except Exception as exc:
        print(f"Video export failed for {prefix}: {exc}")


def _to_uint8(frame, vmin=None, vmax=None):
    frame = np.asarray(frame, dtype=np.float64)
    if vmin is None:
        vmin = float(frame.min())
    if vmax is None:
        vmax = float(frame.max())
    scaled = (frame - vmin) / (vmax - vmin + 1e-12)
    return np.clip(255.0 * scaled, 0, 255).astype(np.uint8)


def make_labeled_panel(images, labels, vmin=None, vmax=None):
    panels = []
    for image, label in zip(images, labels):
        gray = _to_uint8(image, vmin=vmin, vmax=vmax)
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


def save_comparison_video(gt, spatial, ot, root: Path, fps: int):
    """Save GT | spatial init | residual OT videos for visual comparison."""
    if gt is None:
        return
    root.mkdir(parents=True, exist_ok=True)
    K = min(len(gt), len(spatial), len(ot))

    # Per-frame normalization makes morphology/motion visible.
    first = make_labeled_panel(
        [_normalize01_local(gt[0]), _normalize01_local(spatial[0]), _normalize01_local(ot[0])],
        ["ground truth", "spatial", "residual OT"],
    )
    writer = cv2.VideoWriter(
        str(root / "gt_vs_spatial_vs_residual_ot_each_norm.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (first.shape[1], first.shape[0]),
    )
    writer.write(first)
    for k in range(1, K):
        frame = make_labeled_panel(
            [_normalize01_local(gt[k]), _normalize01_local(spatial[k]), _normalize01_local(ot[k])],
            ["ground truth", "spatial", "residual OT"],
        )
        writer.write(frame)
    writer.release()

    # Global normalization is more honest about brightness variation.
    global_min = float(min(gt[:K].min(), spatial[:K].min(), ot[:K].min()))
    global_max = float(max(gt[:K].max(), spatial[:K].max(), ot[:K].max()))
    first = make_labeled_panel(
        [gt[0], spatial[0], ot[0]],
        ["ground truth", "spatial", "residual OT"],
        vmin=global_min,
        vmax=global_max,
    )
    writer = cv2.VideoWriter(
        str(root / "gt_vs_spatial_vs_residual_ot_global_norm.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (first.shape[1], first.shape[0]),
    )
    writer.write(first)
    for k in range(1, K):
        frame = make_labeled_panel(
            [gt[k], spatial[k], ot[k]],
            ["ground truth", "spatial", "residual OT"],
            vmin=global_min,
            vmax=global_max,
        )
        writer.write(frame)
    writer.release()


def write_ground_truth_comparison_outputs(cfg, names, spatial, ot):
    video_root = cfg.output_root / "04_video_comparison"
    gt, gt_names = load_ground_truth_frames(
        cfg.gt_folder,
        names,
        resize=(cfg.image_width, cfg.image_height),
    )
    if gt is None:
        return

    save_sequence_videos(gt, video_root, gt_names, cfg.fps, "ground_truth")
    save_sequence_videos(spatial, video_root, names, cfg.fps, "spatial_init")
    save_sequence_videos(ot, video_root, names, cfg.fps, "residual_signed_ot_final")
    save_comparison_video(gt, spatial, ot, video_root, cfg.fps)
    np.savez_compressed(video_root / "gt_spatial_ot_sequences.npz", gt=gt, spatial=spatial, ot=ot)


def summary_row(label, u, data_terms, prior, cfg, elapsed, converged):
    terms = objective_terms(u, data_terms, prior, cfg.tv_weight, cfg.prior_weight)
    prior_stack = np.repeat(prior[None, :, :], len(u), axis=0)
    return {
        "stage": label,
        "converged": converged,
        "seconds": elapsed,
        "relative_change_from_prior": relative_change(u, prior_stack),
        **terms,
    }


def main():
    cfg = Config()
    if cfg.frames != 15 or cfg.frame_indices is not None:
        raise RuntimeError(
            f"This edited main2.py should run all 15 frames: got frames={cfg.frames}, "
            f"frame_indices={cfg.frame_indices}"
        )
    cfg.output_root.mkdir(parents=True, exist_ok=True)

    serializable = asdict(cfg)
    serializable.update({
        key: str(value)
        for key, value in serializable.items()
        if isinstance(value, Path)
    })
    (cfg.output_root / "config.json").write_text(json.dumps(serializable, indent=2))

    print("=" * 100)
    print("FULL 15-FRAME RESIDUAL-OT VIDEO COMPARISON MAIN2.PY")
    print("This is main2.py: frames=15, frame_indices=None, video comparison enabled.")
    print("=" * 100)
    print("Objective being tested")
    print("=" * 100)
    print("Spatial initialization:")
    print("  sum_k D_k(u_k) + alpha TV(u_k) + (mu/2)||u_k - prior||^2")
    print("Residual signed OT stage:")
    print("  same spatial objective + beta * sum_k [BB((u_k-p)_+,(u_{k+1}-p)_+) + BB((p-u_k)_+,(p-u_{k+1})_+)]")
    print("ADMM image step still contains data + TV + prior + residual endpoint penalties.")
    print("This is balanced residual-channel OT, not unbalanced OT.")

    data_terms, names = load_data_terms(cfg)
    print(f"Loaded {len(data_terms)} observation frames for the 15-frame video run:")
    print("  " + ", ".join(str(name) for name in names))
    prior = load_prior_image(
        cfg.prior_path,
        resize=(cfg.image_width, cfg.image_height),
        normalize=True,
    )
    u_prior = np.repeat(prior[None, :, :], len(data_terms), axis=0)
    save_frames(u_prior, cfg.output_root / "00_prior", names=names, normalize_each=True)

    print("\n" + "=" * 100)
    print("1. Spatial TV+prior initialization")
    print("=" * 100)
    u_spatial, spatial_history, spatial_elapsed, spatial_converged = run_spatial_tv_prior(
        u_prior,
        data_terms,
        prior,
        cfg,
    )
    pd.DataFrame(spatial_history).to_csv(
        cfg.output_root / "01_spatial_tv_prior_init_history.csv",
        index=False,
    )
    spatial_summary = save_run_outputs(
        label="01_spatial_tv_prior_init",
        u=u_spatial,
        history=spatial_history,
        names=names,
        prior=prior,
        data_terms=data_terms,
        prior_weight=cfg.prior_weight,
        cfg=cfg,
    )
    spatial_summary["seconds"] = spatial_elapsed
    spatial_summary["converged"] = spatial_converged

    print("\n" + "=" * 100)
    print("2. Residual signed OT initialized from spatial TV+prior")
    print("=" * 100)
    u_ot, ot_history, ot_elapsed, ot_converged = run_residual_signed_ot_from_initialization(
        u_spatial,
        data_terms,
        prior,
        cfg,
    )
    ot_stage_dir = cfg.output_root / "02_residual_signed_ot_from_spatial_init"
    ot_stage_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(ot_history).to_csv(ot_stage_dir / "history.csv", index=False)
    ot_summary = save_run_outputs(
        label="02_residual_signed_ot_from_spatial_init",
        u=u_ot,
        history=ot_history,
        names=names,
        prior=prior,
        data_terms=data_terms,
        prior_weight=cfg.prior_weight,
        cfg=cfg,
    )
    ot_summary["seconds"] = ot_elapsed
    ot_summary["converged"] = ot_converged

    save_ot_difference(u_spatial, u_ot, prior, names, cfg)

    print("\n" + "=" * 100)
    print("3. Writing 15-frame videos and GT comparison")
    print("=" * 100)
    write_ground_truth_comparison_outputs(cfg, names, u_spatial, u_ot)

    summary = pd.DataFrame([
        summary_row("01_spatial_tv_prior_init", u_spatial, data_terms, prior, cfg, spatial_elapsed, spatial_converged),
        summary_row("02_residual_signed_ot_from_spatial_init", u_ot, data_terms, prior, cfg, ot_elapsed, ot_converged),
    ])
    for extra in (spatial_summary, ot_summary):
        for key, value in extra.items():
            if key not in summary.columns:
                summary[key] = np.nan
        summary.loc[summary["stage"] == extra["stage"], list(extra.keys())] = list(extra.values())

    summary.to_csv(cfg.output_root / "summary.csv", index=False)

    print("\n" + summary.to_string(index=False))
    print(f"\nOutputs saved to {cfg.output_root.resolve()}")
    print("\nInspect:")
    print("  01_spatial_tv_prior_init/montage_delta_signed.png")
    print("  02_residual_signed_ot_from_spatial_init/montage_delta_signed.png")
    print("  03_ot_minus_spatial/montage_ot_minus_spatial_signed.png")
    print("  03_ot_minus_spatial/montage_pairwise_delta_change_difference_signed.png")
    print("  04_video_comparison/gt_vs_spatial_vs_residual_ot_each_norm.mp4")
    print("  04_video_comparison/gt_vs_spatial_vs_residual_ot_global_norm.mp4")


if __name__ == "__main__":
    main()
