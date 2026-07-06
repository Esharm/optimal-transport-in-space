"""Data-consistent ring augmentation followed by residual signed OT/ADMM.

This main4.py is intentionally configured to run all first 15 observation
frames, initialize from a radially averaged static reconstruction augmented by
fast bounded data-gradient moves, and write only minimal outputs:

    main4_results/results.npz
    main4_results/comparison_gt_spatial_ot.mp4

This is the current experimental pipeline:

    1. Start from the calibrated static ring prior p and take a few bounded,
       smooth, ring-supported residual-gradient moves for each frame:

           u_k = max(p + delta_k, 0).

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
class Config(LoaderConfig):
    prior_path: Path = PROJECT_ROOT / "radial_outputs" / "time_avg_static_recon_128pix_radial_round.png"
    output_root: Path = Path("main4_results")
    gt_folder: Path = PROJECT_ROOT / "blackhole_sim" / "data" / "aart_frames"
    fps: int = 5

    # Full 15-frame run for video comparison. Set this back to e.g. (0, 7, 14)
    # for a faster diagnostic subset.
    frames: int = 15
    frame_indices: tuple[int, ...] | None = None

    # The prior PNG is brightness-calibrated to the visibilities before use.
    # main4 does not use a spatial reconstruction/pre-prior solve. It directly
    # augments the calibrated ring with a few bounded data-gradient moves so the
    # initialization is frame-specific but quick and physically scaled.
    prior_weight: float = 4e-2
    tv_weight: float = 1e-5

    # Fast pre-OT augmentation: a few bounded residual-gradient moves, not a
    # converged reconstruction. The formal data term is still handled in ADMM.
    augmentation_steps: int = 2
    augmentation_smoothing_sigma_px: float = 1.25
    augmentation_support_threshold_fraction: float = 0.08
    augmentation_support_floor: float = 0.02
    augmentation_scale_fractions: tuple[float, ...] = (
        0.0, 0.05, 0.10, 0.20, 0.35, 0.55, 0.75, 1.0
    )
    max_initial_relative_change: float = 0.75
    max_initial_total_brightness_factor: float = 1.50
    min_initial_total_brightness_factor: float = 0.50
    max_initial_peak_factor: float = 3.0
    require_data_improvement: bool = True
    enforce_nonnegative_initialization: bool = True
    use_visibility_cache: bool = False
    visibility_chunk_size: int = 128

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
    ot_max_iter: int = 15
    ot_min_iter: int = 3
    ot_patience: int = 2
    # With streamed 5000-visibility operators, each image update is expensive.
    # Keep OT as a fast refinement step; the data term still appears every ADMM
    # iteration, but we do not solve the image subproblem tightly.
    ot_image_inner_iters: int = 3
    ot_power_iters: int = 3
    transport_slices: int = 7
    transport_inner_iters: int = 60
    transport_tol: float = 1e-3

    parallel_frames: bool = False
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


def _stack_complex(z):
    z = np.asarray(z, dtype=np.complex128).ravel()
    return np.concatenate((z.real, z.imag))


def gaussian_smooth_fft(image, sigma_px):
    if sigma_px <= 0:
        return np.asarray(image, dtype=np.float64)
    image = np.asarray(image, dtype=np.float64)
    height, width = image.shape
    fy = np.fft.fftfreq(height)
    fx = np.fft.fftfreq(width)
    freq_x, freq_y = np.meshgrid(fx, fy)
    gaussian = np.exp(-2.0 * (np.pi ** 2) * (sigma_px ** 2) * (freq_x ** 2 + freq_y ** 2))
    return np.fft.ifft2(np.fft.fft2(image) * gaussian).real


def make_ring_support(prior, cfg):
    prior = np.asarray(prior, dtype=np.float64)
    threshold = cfg.augmentation_support_threshold_fraction * float(prior.max() + 1e-30)
    support = (prior >= threshold).astype(np.float64)
    support = gaussian_smooth_fft(support, cfg.augmentation_smoothing_sigma_px)
    support = support / (support.max() + 1e-30)
    return cfg.augmentation_support_floor + (1.0 - cfg.augmentation_support_floor) * support


def filtered_gradient_direction(data_term, image, support, cfg):
    residual = data_term.sampler.forward(image) - data_term.f
    # Negative gradient of 0.5 ||S u - f||^2.
    direction = -data_term.sampler.adjoint(residual)
    direction = gaussian_smooth_fft(direction, cfg.augmentation_smoothing_sigma_px)
    return support * direction


def choose_bounded_update(data_term, prior, current, direction, base_loss, cfg):
    prior_norm = np.linalg.norm(prior) + 1e-30
    direction_norm = np.linalg.norm(direction)
    if direction_norm <= 1e-30:
        return current, 0.0, base_loss, "zero_direction"

    max_delta_norm = cfg.max_initial_relative_change * prior_norm
    current_delta = current - prior
    remaining_delta_norm = max_delta_norm - np.linalg.norm(current_delta)
    if remaining_delta_norm <= 0:
        return current, 0.0, base_loss, "relative_change_cap"

    scale_limit = remaining_delta_norm / (direction_norm + 1e-30)
    prior_sum = float(prior.sum())
    min_sum = cfg.min_initial_total_brightness_factor * prior_sum
    max_sum = cfg.max_initial_total_brightness_factor * prior_sum
    max_peak = cfg.max_initial_peak_factor * float(prior.max() + 1e-30)

    best = current
    best_scale = 0.0
    best_loss = base_loss
    best_reason = "no_improvement"

    for fraction in cfg.augmentation_scale_fractions:
        scale = scale_limit * float(fraction)
        candidate = current + scale * direction
        if cfg.enforce_nonnegative_initialization:
            candidate = np.maximum(candidate, 0.0)

        total = float(candidate.sum())
        peak = float(candidate.max())
        if total < min_sum or total > max_sum or peak > max_peak:
            continue

        loss = float(data_term.loss(candidate))
        allowed = (not cfg.require_data_improvement) or loss <= base_loss
        if allowed and loss <= best_loss:
            best = candidate
            best_scale = scale
            best_loss = loss
            best_reason = "accepted"

    return best, best_scale, best_loss, best_reason


def run_data_consistent_ring_augmentation(data_terms, prior, cfg):
    """Fast bounded augmentation: a few safe data-gradient moves from the ring."""
    started = time.perf_counter()
    support = make_ring_support(prior, cfg)

    def augment_frame(k):
        term = data_terms[k]
        u = prior.copy()
        prior_loss = float(term.loss(prior))
        loss = prior_loss
        accepted_steps = 0
        chosen_scales = []
        stop_reason = "max_steps"

        for _ in range(cfg.augmentation_steps):
            direction = filtered_gradient_direction(term, u, support, cfg)
            candidate, scale, candidate_loss, reason = choose_bounded_update(
                term, prior, u, direction, loss, cfg
            )
            if scale <= 0.0:
                stop_reason = reason
                break
            u = candidate
            loss = candidate_loss
            accepted_steps += 1
            chosen_scales.append(scale)

        final_residual = term.sampler.forward(u) - term.f
        data_norm = np.linalg.norm(_stack_complex(term.f)) + 1e-30
        delta = u - prior
        info = {
            "frame": k,
            "accepted_steps": accepted_steps,
            "stop_reason": stop_reason,
            "chosen_scale_sum": float(np.sum(chosen_scales)) if chosen_scales else 0.0,
            "prior_data_loss": prior_loss,
            "postclip_data_loss": loss,
            "data_loss_improvement": float(prior_loss - loss),
            "postclip_visibility_relative_residual": float(
                np.linalg.norm(_stack_complex(final_residual)) / data_norm
            ),
            "relative_change_from_prior": float(np.linalg.norm(delta) / (np.linalg.norm(prior) + 1e-30)),
            "positive_residual_mass": float(np.maximum(delta, 0.0).sum()),
            "negative_residual_mass": float(np.maximum(-delta, 0.0).sum()),
            "u_min": float(u.min()),
            "u_max": float(u.max()),
            "u_sum": float(u.sum()),
        }
        return u, info

    if cfg.parallel_frames and len(data_terms) > 1:
        with ThreadPoolExecutor(max_workers=len(data_terms)) as executor:
            results = list(executor.map(augment_frame, range(len(data_terms))))
    else:
        results = [augment_frame(k) for k in range(len(data_terms))]

    u = np.stack([item[0] for item in results])
    frame_infos = [item[1] for item in results]
    elapsed = time.perf_counter() - started

    losses = np.asarray([info["postclip_data_loss"] for info in frame_infos])
    prior_losses = np.asarray([info["prior_data_loss"] for info in frame_infos])
    postclip_residuals = np.asarray([
        info["postclip_visibility_relative_residual"] for info in frame_infos
    ])
    relative_changes = np.asarray([
        info["relative_change_from_prior"] for info in frame_infos
    ])
    accepted_steps = np.asarray([info["accepted_steps"] for info in frame_infos])

    aggregate = {
        "stage": "bounded_fast_ring_augmentation",
        "seconds": elapsed,
        "accepted_steps_min": int(accepted_steps.min()),
        "accepted_steps_max": int(accepted_steps.max()),
        "accepted_steps_mean": float(accepted_steps.mean()),
        "postclip_visibility_relative_residual_mean": float(postclip_residuals.mean()),
        "postclip_visibility_relative_residual_max": float(postclip_residuals.max()),
        "prior_data_loss_mean": float(prior_losses.mean()),
        "postclip_data_loss_mean": float(losses.mean()),
        "postclip_data_loss_max": float(losses.max()),
        "data_loss_improvement_mean": float((prior_losses - losses).mean()),
        "relative_change_from_prior_mean": float(relative_changes.mean()),
        "relative_change_from_prior_max": float(relative_changes.max()),
    }
    history = [{"aggregate": aggregate, "frames": frame_infos}]

    print(
        "bounded fast augmentation | "
        f"steps={aggregate['accepted_steps_min']}-{aggregate['accepted_steps_max']} "
        f"| postclip_vis_rel_mean={aggregate['postclip_visibility_relative_residual_mean']:.3e} "
        f"| data_loss {aggregate['prior_data_loss_mean']:.3e}->{aggregate['postclip_data_loss_mean']:.3e} "
        f"| rel_change_max={aggregate['relative_change_from_prior_max']:.3e}"
    )

    return u, history, elapsed, True


def run_residual_signed_ot_from_initialization(u_init, data_terms, prior, cfg):
    regularizer = TotalVariationRegularizer(
        alpha=cfg.tv_weight,
        iters=cfg.ot_image_inner_iters,
        tau=cfg.primal_tau,
        sigma=cfg.dual_sigma,
        power_iters=cfg.ot_power_iters,
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
    print(
        "Starting OT/ADMM refinement "
        f"(outer iters <= {cfg.ot_max_iter}, image iters = {cfg.ot_image_inner_iters}, "
        f"power iters = {cfg.ot_power_iters}, transport iters <= {cfg.transport_inner_iters})"
    )
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
            f"This edited main4.py should run all 15 frames: got frames={cfg.frames}, "
            f"frame_indices={cfg.frame_indices}"
        )
    cfg.output_root.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("MAIN4.PY: RADIAL PRIOR -> FAST BOUNDED AUGMENTATION -> RESIDUAL OT")
    print("Outputs are intentionally minimal: results.npz and one comparison video.")
    print("=" * 100)
    print("Objective being tested")
    print("=" * 100)
    print("Fast bounded ring augmentation:")
    print("  u_k = max(prior + delta_k, 0)")
    print("  delta_k comes from a few smooth/ring-supported negative-gradient data steps")
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
    print("1. Fast bounded ring augmentation")
    print("=" * 100)
    u_spatial, spatial_history, spatial_elapsed, spatial_converged = run_data_consistent_ring_augmentation(
        data_terms,
        prior,
        cfg,
    )
    spatial_summary = summary_row(
        "bounded_fast_ring_augmentation",
        u_spatial,
        data_terms,
        prior,
        cfg,
        spatial_elapsed,
        spatial_converged,
    )
    if spatial_history:
        spatial_summary.update(spatial_history[-1]["aggregate"])

    print("\n" + "=" * 100)
    print("2. Residual signed OT initialized from data-consistent ring augmentation")
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
