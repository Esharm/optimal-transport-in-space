"""Radial-prior spatial reconstruction followed by residual signed OT/ADMM.

This main3.py is intentionally configured to run all first 15 observation
frames, initialize from a radially averaged static reconstruction, and write
only minimal outputs:

    main3_results/results.npz
    main3_results/comparison_gt_spatial_ot.mp4

This is the current experimental pipeline:

    1. Independently reconstruct each selected frame with

           D_k(u_k) + alpha TV(u_k) + (mu/2)||u_k - p||_2^2,

       where p is the radialized averaged-image prior.

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
from PIL import Image

from admm import ResidualSignedOTADMM
from data_terms import ComplexVisibilityDataTerm
from io_utils import load_prior_image
from operators import div, grad
from run_three_frame_ablation import Config as LoaderConfig
from run_three_frame_ablation import load_data_terms, objective_terms, relative_change
from solvers import TotalVariationRegularizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
# FULL_15_FRAME_VIDEO_COMPARISON_VERSION
class Config(LoaderConfig):
    prior_path: Path = PROJECT_ROOT / "radial_outputs" / "time_avg_static_recon_128pix_radial_round.png"
    output_root: Path = Path("main3_results")
    gt_folder: Path = PROJECT_ROOT / "blackhole_sim" / "data" / "aart_frames"
    fps: int = 5

    # Full 15-frame run for video comparison. Set this back to e.g. (0, 7, 14)
    # for a faster diagnostic subset.
    frames: int = 15
    frame_indices: tuple[int, ...] | None = None

    # The prior PNG is brightness-calibrated to the visibilities before use.
    # Spatial initialization reconstructs a signed residual delta_k around the
    # static prior instead of reconstructing each full image from scratch:
    #     u_k = prior + delta_k,   S_k delta_k ~= f_k - S_k prior.
    prior_weight: float = 4e-2
    tv_weight: float = 1e-5

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


def estimate_data_lipschitz(data_term, shape, cache):
    key = id(data_term)
    if key in cache:
        return cache[key]

    rng = np.random.default_rng(1729)
    vector = rng.normal(size=shape)
    vector /= np.linalg.norm(vector) + 1e-30
    eigenvalue = 0.0
    for _ in range(12):
        applied = data_term.sampler.adjoint(data_term.sampler.forward(vector))
        norm = np.linalg.norm(applied)
        if norm <= 1e-30:
            eigenvalue = 0.0
            break
        vector = applied / norm
        eigenvalue = float(np.sum(vector * applied))

    estimate = max(1.1 * eigenvalue, 1e-12)
    cache[key] = estimate
    return estimate


def tv_value(image):
    derivative = grad(image)
    return float(np.sum(np.sqrt(derivative[0] ** 2 + derivative[1] ** 2)))


def build_residual_data_terms(data_terms, prior):
    residual_terms = []
    for term in data_terms:
        residual_f = term.f - term.sampler.forward(prior)
        residual_terms.append(ComplexVisibilityDataTerm(term.sampler, residual_f))
    return residual_terms


def solve_signed_residual_tv(
    delta_init,
    residual_term,
    l2_weight,
    tv_weight,
    iters,
    tau_cap,
    sigma,
    lipschitz_cache,
):
    """Solve a signed residual problem without nonnegative projection."""
    delta = np.asarray(delta_init, dtype=np.float64).copy()
    smooth_lipschitz = (
        estimate_data_lipschitz(residual_term, delta.shape, lipschitz_cache)
        + l2_weight
    )

    if tv_weight <= 0:
        step = min(tau_cap, 0.99 / smooth_lipschitz)
        extrapolated = delta.copy()
        momentum_parameter = 1.0
        for _ in range(iters):
            old = delta.copy()
            gradient = residual_term.gradient(extrapolated) + l2_weight * extrapolated
            delta = extrapolated - step * gradient
            next_parameter = 0.5 * (
                1.0 + np.sqrt(1.0 + 4.0 * momentum_parameter ** 2)
            )
            extrapolated_next = delta + (
                (momentum_parameter - 1.0) / next_parameter
            ) * (delta - old)
            if np.sum((extrapolated - delta) * (delta - old)) > 0:
                momentum_parameter = 1.0
                extrapolated = delta.copy()
            else:
                momentum_parameter = next_parameter
                extrapolated = extrapolated_next
        return delta

    dual = np.zeros((2, *delta.shape), dtype=np.float64)
    safe_tau = 0.99 / (0.5 * smooth_lipschitz + 8.0 * sigma)
    tau = min(tau_cap, safe_tau)
    delta_bar = delta.copy()

    for _ in range(iters):
        dual += sigma * grad(delta_bar)
        norm_dual = np.sqrt(np.sum(dual * dual, axis=0))
        dual /= np.maximum(1.0, norm_dual / tv_weight)[None, :, :]

        old = delta.copy()
        gradient = residual_term.gradient(delta) + l2_weight * delta
        delta -= tau * (gradient - div(dual))
        delta_bar = 2.0 * delta - old

    return delta


def residual_objective_terms(delta, residual_terms, tv_weight, l2_weight):
    data = float(sum(term.loss(delta[k]) for k, term in enumerate(residual_terms)))
    tv = float(tv_weight * sum(tv_value(frame) for frame in delta))
    l2 = float(0.5 * l2_weight * np.sum(delta * delta))
    return {"residual_data": data, "residual_tv": tv, "residual_l2": l2}


def run_spatial_residual_tv_prior(data_terms, prior, cfg):
    residual_terms = build_residual_data_terms(data_terms, prior)
    delta = np.zeros((len(data_terms), *prior.shape), dtype=np.float64)
    history = []
    stable = 0
    previous_objective = None
    lipschitz_cache = {}
    started = time.perf_counter()

    for block in range(1, cfg.spatial_max_blocks + 1):
        old_u = np.maximum(prior[None, :, :] + delta, 0.0)

        def update_frame(k):
            return solve_signed_residual_tv(
                delta_init=delta[k],
                residual_term=residual_terms[k],
                l2_weight=cfg.prior_weight,
                tv_weight=cfg.tv_weight,
                iters=cfg.spatial_inner_iters,
                tau_cap=cfg.primal_tau,
                sigma=cfg.dual_sigma,
                lipschitz_cache=lipschitz_cache,
            )

        if cfg.parallel_frames and len(data_terms) > 1:
            with ThreadPoolExecutor(max_workers=len(data_terms)) as executor:
                delta = np.stack(list(executor.map(update_frame, range(len(data_terms)))))
        else:
            for k in range(len(data_terms)):
                delta[k] = update_frame(k)

        # Keep the physical image nonnegative while preserving signed residuals
        # everywhere positivity is not active.
        u = np.maximum(prior[None, :, :] + delta, 0.0)
        delta = u - prior[None, :, :]

        full_terms = objective_terms(u, data_terms, prior, cfg.tv_weight, cfg.prior_weight)
        residual_terms_row = residual_objective_terms(
            delta,
            residual_terms,
            cfg.tv_weight,
            cfg.prior_weight,
        )
        residual_total = sum(residual_terms_row.values())
        du = relative_change(u, old_u)
        if previous_objective is None:
            dobj = np.inf
        else:
            dobj = abs(residual_total - previous_objective) / (
                abs(previous_objective) + 1e-12
            )
        previous_objective = residual_total

        row = {
            "block": block,
            "updates": block * cfg.spatial_inner_iters,
            "relative_iterate_change": du,
            "relative_objective_change": dobj,
            "residual_total": residual_total,
            **residual_terms_row,
            **full_terms,
        }
        history.append(row)

        print(
            f"residual spatial block {block:03d} | residual_total={residual_total:.6e} "
            f"| residual_data={residual_terms_row['residual_data']:.3e} "
            f"| residual_tv={residual_terms_row['residual_tv']:.3e} "
            f"| residual_l2={residual_terms_row['residual_l2']:.3e} "
            f"| d_obj={dobj:.3e} | d_u={du:.3e}"
        )

        is_stable = dobj < cfg.objective_rel_tol and du < cfg.iterate_rel_tol
        stable = stable + 1 if block >= cfg.spatial_min_blocks and is_stable else 0
        if stable >= cfg.spatial_patience:
            print(f"Residual spatial initialization converged after {block * cfg.spatial_inner_iters} updates")
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
    """Save one GT | spatial init | residual OT comparison video."""
    if gt is None:
        print("No ground truth available; skipping comparison video.")
        return None

    root.mkdir(parents=True, exist_ok=True)
    K = min(len(gt), len(spatial), len(ot))

    output_path = root / "comparison_gt_spatial_ot.mp4"
    first = make_labeled_panel(
        [_normalize01_local(gt[0]), _normalize01_local(spatial[0]), _normalize01_local(ot[0])],
        ["GT", "spatial no OT", "OT"],
    )
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (first.shape[1], first.shape[0]),
    )
    writer.write(first)
    for k in range(1, K):
        frame = make_labeled_panel(
            [_normalize01_local(gt[k]), _normalize01_local(spatial[k]), _normalize01_local(ot[k])],
            ["GT", "spatial no OT", "OT"],
        )
        writer.write(frame)
    writer.release()

    return output_path


def load_ground_truth_for_comparison(cfg, names):
    gt, gt_names = load_ground_truth_frames(
        cfg.gt_folder,
        names,
        resize=(cfg.image_width, cfg.image_height),
    )
    return gt, gt_names


def calibrate_prior_brightness(prior_shape, data_terms):
    """Scale a prior shape to the least-squares brightness preferred by data."""
    numerator = 0.0
    denominator = 0.0
    for term in data_terms:
        predicted = term.sampler.forward(prior_shape)
        numerator += float(np.real(np.vdot(predicted, term.f)))
        denominator += float(np.vdot(predicted, predicted).real)

    scale = max(numerator / (denominator + 1e-30), 0.0)
    return scale * prior_shape, {
        "prior_brightness_scale": scale,
        "prior_scale_fit_numerator": numerator,
        "prior_scale_fit_denominator": denominator,
    }


def sequence_pairwise_diagnostics(label, frames):
    frames = np.asarray(frames, dtype=np.float64)
    rows = {}
    if len(frames) < 2:
        return rows
    diffs = frames[1:] - frames[:-1]
    frame_norms = np.linalg.norm(frames.reshape(len(frames), -1), axis=1)
    diff_norms = np.linalg.norm(diffs.reshape(len(diffs), -1), axis=1)
    rows[f"{label}_adjacent_l2_min"] = float(diff_norms.min())
    rows[f"{label}_adjacent_l2_max"] = float(diff_norms.max())
    rows[f"{label}_adjacent_l2_mean"] = float(diff_norms.mean())
    rows[f"{label}_adjacent_relative_l2_mean"] = float(
        np.mean(diff_norms / (frame_norms[:-1] + 1e-12))
    )
    rows[f"{label}_global_min"] = float(frames.min())
    rows[f"{label}_global_max"] = float(frames.max())
    rows[f"{label}_total_brightness_min"] = float(frames.sum(axis=(1, 2)).min())
    rows[f"{label}_total_brightness_max"] = float(frames.sum(axis=(1, 2)).max())
    return rows


def data_force_diagnostics(prior, data_terms):
    gradients = np.stack([term.gradient(prior) for term in data_terms])
    losses = np.asarray([term.loss(prior) for term in data_terms])
    dirty = np.stack([term.dirty_image() for term in data_terms])
    gradient_norms = np.linalg.norm(gradients.reshape(len(data_terms), -1), axis=1)
    dirty_norms = np.linalg.norm(dirty.reshape(len(data_terms), -1), axis=1)
    return {
        "data_loss_at_prior_min": float(losses.min()),
        "data_loss_at_prior_max": float(losses.max()),
        "data_loss_at_prior_mean": float(losses.mean()),
        "data_gradient_at_prior_l2_min": float(gradient_norms.min()),
        "data_gradient_at_prior_l2_max": float(gradient_norms.max()),
        "data_gradient_at_prior_l2_mean": float(gradient_norms.mean()),
        "dirty_image_l2_min": float(dirty_norms.min()),
        "dirty_image_l2_max": float(dirty_norms.max()),
        "dirty_image_global_max": float(dirty.max()),
    }


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


def serializable_config(cfg):
    values = asdict(cfg)
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in values.items()
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


def save_results_file(
    cfg,
    names,
    prior,
    spatial,
    ot,
    gt,
    gt_names,
    spatial_history,
    ot_history,
    spatial_summary,
    ot_summary,
    diagnostics,
):
    output_path = cfg.output_root / "results.npz"
    np.savez_compressed(
        output_path,
        names=np.asarray(names),
        gt_names=np.asarray(gt_names),
        prior=prior,
        spatial=spatial,
        ot=ot,
        spatial_delta=spatial - prior[None, :, :],
        ot_delta=ot - prior[None, :, :],
        gt=np.asarray([]) if gt is None else gt,
        config_json=np.asarray(json.dumps(serializable_config(cfg), indent=2, default=json_default)),
        summary_json=np.asarray(json.dumps([spatial_summary, ot_summary], indent=2, default=json_default)),
        diagnostics_json=np.asarray(json.dumps(diagnostics, indent=2, default=json_default)),
        spatial_history_json=np.asarray(json.dumps(spatial_history, indent=2, default=json_default)),
        ot_history_json=np.asarray(json.dumps(ot_history, indent=2, default=json_default)),
    )
    return output_path


def main():
    cfg = Config()
    if cfg.frames != 15 or cfg.frame_indices is not None:
        raise RuntimeError(
            f"This edited main3.py should run all 15 frames: got frames={cfg.frames}, "
            f"frame_indices={cfg.frame_indices}"
        )
    cfg.output_root.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("MAIN3.PY: RADIAL PRIOR -> SIGNED RESIDUAL SPATIAL INIT -> RESIDUAL OT")
    print("Outputs are intentionally minimal: results.npz and one comparison video.")
    print("=" * 100)
    print("Objective being tested")
    print("=" * 100)
    print("Spatial residual initialization:")
    print("  u_k = prior + delta_k")
    print("  sum_k 0.5||S_k delta_k - (f_k - S_k prior)||^2 + alpha TV(delta_k) + (mu/2)||delta_k||^2")
    print("Residual signed OT stage:")
    print("  same spatial objective + beta * sum_k [BB((u_k-p)_+,(u_{k+1}-p)_+) + BB((p-u_k)_+,(p-u_{k+1})_+)]")
    print("ADMM image step still contains data + TV + prior + residual endpoint penalties.")
    print("This is balanced residual-channel OT, not unbalanced OT.")

    data_terms, names = load_data_terms(cfg)
    print(f"Loaded {len(data_terms)} observation frames for the 15-frame video run:")
    print("  " + ", ".join(str(name) for name in names))
    prior_shape = load_prior_image(
        cfg.prior_path,
        resize=(cfg.image_width, cfg.image_height),
        normalize=True,
    )
    print(f"Using radial initialization/prior: {cfg.prior_path}")
    prior, prior_scale_diagnostics = calibrate_prior_brightness(prior_shape, data_terms)
    print(
        "Calibrated radial prior brightness by least-squares scale "
        f"{prior_scale_diagnostics['prior_brightness_scale']:.6e}"
    )
    u_prior = np.repeat(prior[None, :, :], len(data_terms), axis=0)

    print("\n" + "=" * 100)
    print("1. Signed residual spatial initialization")
    print("=" * 100)
    u_spatial, spatial_history, spatial_elapsed, spatial_converged = run_spatial_residual_tv_prior(
        data_terms,
        prior,
        cfg,
    )
    spatial_summary = summary_row(
        "signed_residual_spatial_init",
        u_spatial,
        data_terms,
        prior,
        cfg,
        spatial_elapsed,
        spatial_converged,
    )
    if spatial_history:
        for key in ("residual_total", "residual_data", "residual_tv", "residual_l2"):
            spatial_summary[key] = spatial_history[-1][key]

    print("\n" + "=" * 100)
    print("2. Residual signed OT initialized from spatial TV+prior")
    print("=" * 100)
    u_ot, ot_history, ot_elapsed, ot_converged = run_residual_signed_ot_from_initialization(
        u_spatial,
        data_terms,
        prior,
        cfg,
    )
    ot_summary = summary_row(
        "residual_signed_ot_from_spatial_init",
        u_ot,
        data_terms,
        prior,
        cfg,
        ot_elapsed,
        ot_converged,
    )

    print("\n" + "=" * 100)
    print("3. Writing minimal outputs")
    print("=" * 100)
    gt, gt_names = load_ground_truth_for_comparison(cfg, names)
    video_path = save_comparison_video(gt, u_spatial, u_ot, cfg.output_root, cfg.fps)
    diagnostics = {
        **prior_scale_diagnostics,
        **data_force_diagnostics(prior, data_terms),
        **sequence_pairwise_diagnostics("prior_init", u_prior),
        **sequence_pairwise_diagnostics("spatial", u_spatial),
        **sequence_pairwise_diagnostics("ot", u_ot),
    }
    results_path = save_results_file(
        cfg=cfg,
        names=names,
        prior=prior,
        spatial=u_spatial,
        ot=u_ot,
        gt=gt,
        gt_names=gt_names,
        spatial_history=spatial_history,
        ot_history=ot_history,
        spatial_summary=spatial_summary,
        ot_summary=ot_summary,
        diagnostics=diagnostics,
    )

    print("\nSummary:")
    print(json.dumps([spatial_summary, ot_summary], indent=2, default=json_default))
    print("\nDiagnostics:")
    print(json.dumps(diagnostics, indent=2, default=json_default))
    print(f"\nOutputs saved to {cfg.output_root.resolve()}")
    print(f"  Results: {results_path}")
    if video_path is not None:
        print(f"  Comparison video: {video_path}")


if __name__ == "__main__":
    main()
