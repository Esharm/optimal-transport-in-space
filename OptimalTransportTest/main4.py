"""Data-consistent ring augmentation followed by residual signed OT/ADMM.

This main4.py is intentionally configured to run all first 15 observation
frames, initialize from a radially averaged static reconstruction augmented to
fit each frame's visibility data without leaving the physical ring scale, and write only minimal outputs:

    main4_results/results.npz
    main4_results/comparison_gt_spatial_ot.mp4

This is the current experimental pipeline:

    1. Start from the calibrated static ring prior p and compute a signed,
       bounded smooth ring-supported residual correction delta_k for each frame:

           delta_k = M G_sigma z_k,
           z_k fits S_k delta_k ~= f_k - S_k p,
           u_k = max(p + a_k delta_k, 0),

       where a_k is chosen by a trust-region line search so the frame remains
       near the brightness scale/support of the calibrated ring.

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
    # augments the calibrated ring with a signed minimum-norm correction that
    # fits each frame's residual visibilities.
    prior_weight: float = 4e-2
    tv_weight: float = 1e-5

    # Bounded ring-supported visibility augmentation.
    # The old main4 minimum-norm correction was too ill-conditioned: it could
    # match neither the data nor image scale after positivity clipping. These
    # settings restrict the correction to a smooth residual near the ring, then
    # line-search the largest physically safe scale.
    data_match_ridge: float = 1e-8
    data_match_max_iter: int = 120
    data_match_tol: float = 2e-3
    correction_smoothing_sigma_px: float = 1.5
    correction_support_threshold_fraction: float = 0.08
    correction_support_blur_sigma_px: float = 2.0
    correction_support_floor: float = 0.02
    max_initial_relative_change: float = 0.75
    max_initial_total_brightness_factor: float = 1.50
    min_initial_total_brightness_factor: float = 0.50
    max_initial_peak_factor: float = 3.0
    max_allowed_data_loss_factor: float = 1.05
    require_data_improvement: bool = True
    residual_mass_cap_factor: float = 2.5
    enforce_nonnegative_initialization: bool = True

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
    ot_min_iter: int = 5
    ot_patience: int = 3
    transport_slices: int = 7
    transport_inner_iters: int = 100
    transport_tol: float = 1e-3

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


def _stack_complex(z):
    z = np.asarray(z, dtype=np.complex128).ravel()
    return np.concatenate((z.real, z.imag))


def _unstack_complex(x):
    x = np.asarray(x, dtype=np.float64).ravel()
    half = x.size // 2
    return x[:half] + 1j * x[half:]


def _gaussian_blur(image, sigma):
    """Gaussian blur with a safe no-op path and reflect boundaries."""
    image = np.asarray(image, dtype=np.float64)
    sigma = float(sigma)
    if sigma <= 0:
        return image.copy()
    # OpenCV picks the kernel size from sigma when ksize=(0, 0).
    return cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT)


def build_ring_support_mask(prior, cfg):
    """Soft mask that keeps visibility corrections concentrated near the ring."""
    prior = np.asarray(prior, dtype=np.float64)
    peak = float(np.max(prior))
    if peak <= 1e-30:
        return np.ones_like(prior)

    hard = (prior >= cfg.correction_support_threshold_fraction * peak).astype(np.float64)
    soft = _gaussian_blur(hard, cfg.correction_support_blur_sigma_px)
    soft = soft / (float(np.max(soft)) + 1e-30)
    floor = float(np.clip(cfg.correction_support_floor, 0.0, 1.0))
    return floor + (1.0 - floor) * soft


def latent_to_correction(latent, support, cfg):
    """Map a free signed latent image to a smooth, ring-supported correction."""
    return support * _gaussian_blur(latent, cfg.correction_smoothing_sigma_px)


def correction_adjoint(image, support, cfg):
    """Adjoint of latent_to_correction for symmetric Gaussian blur."""
    return _gaussian_blur(support * image, cfg.correction_smoothing_sigma_px)


def constrained_visibility_correction(data_term, target_visibility, support, cfg):
    """Solve for a smooth, ring-supported signed correction.

    Let B z = M G_sigma z, where M is a soft ring-support mask and G_sigma is
    Gaussian smoothing. This solves the ridge-regularized minimum latent-norm
    problem

        min_z 0.5 || S B z - target_visibility ||^2 + 0.5 ridge ||z||^2

    by conjugate gradients in measurement space. The returned image correction
    is delta = B z. This keeps the correction from using arbitrary image-space
    null directions far away from the ring.
    """
    sampler = data_term.sampler
    rhs = _stack_complex(target_visibility)
    rhs_norm = float(np.linalg.norm(rhs))

    if rhs_norm <= 1e-30:
        delta = np.zeros((sampler.H, sampler.W), dtype=np.float64)
        return delta, {
            "cg_iterations": 0,
            "cg_relative_residual": 0.0,
            "preclip_visibility_relative_residual": 0.0,
            "delta_l2": 0.0,
            "delta_min": 0.0,
            "delta_max": 0.0,
        }

    def k_adjoint(coefficients):
        image = sampler.adjoint(coefficients)
        return correction_adjoint(image, support, cfg)

    def k_forward(latent):
        return sampler.forward(latent_to_correction(latent, support, cfg))

    def apply_gram(coefficients_real):
        coefficients = _unstack_complex(coefficients_real)
        latent = k_adjoint(coefficients)
        projected = k_forward(latent)
        result = _stack_complex(projected)
        if cfg.data_match_ridge > 0:
            result = result + cfg.data_match_ridge * coefficients_real
        return result

    coefficients = np.zeros_like(rhs)
    residual = rhs.copy()
    direction = residual.copy()
    residual_sq = float(np.dot(residual, residual))
    relative_residual = np.sqrt(residual_sq) / rhs_norm
    iterations = 0

    for iterations in range(1, cfg.data_match_max_iter + 1):
        gram_direction = apply_gram(direction)
        denominator = float(np.dot(direction, gram_direction))
        if abs(denominator) <= 1e-30:
            break

        step = residual_sq / denominator
        coefficients += step * direction
        residual -= step * gram_direction

        next_residual_sq = float(np.dot(residual, residual))
        relative_residual = np.sqrt(next_residual_sq) / rhs_norm
        if relative_residual <= cfg.data_match_tol:
            residual_sq = next_residual_sq
            break

        direction = residual + (next_residual_sq / (residual_sq + 1e-30)) * direction
        residual_sq = next_residual_sq

    latent = k_adjoint(_unstack_complex(coefficients))
    delta = latent_to_correction(latent, support, cfg)
    preclip_residual = sampler.forward(delta) - target_visibility
    preclip_relative = float(
        np.linalg.norm(_stack_complex(preclip_residual)) / (rhs_norm + 1e-30)
    )

    return delta, {
        "cg_iterations": iterations,
        "cg_relative_residual": float(relative_residual),
        "preclip_visibility_relative_residual": preclip_relative,
        "delta_l2": float(np.linalg.norm(delta)),
        "delta_min": float(delta.min()),
        "delta_max": float(delta.max()),
    }


def _candidate_scale_grid():
    """Dense near zero, with exact 1.0 included for successful clean fits."""
    return np.asarray([
        1.00, 0.85, 0.70, 0.55, 0.40, 0.30, 0.22, 0.16,
        0.12, 0.09, 0.06, 0.04, 0.025, 0.015, 0.008, 0.0,
    ], dtype=np.float64)


def choose_safe_correction_scale(prior, raw_delta, data_term, cfg):
    """Pick the most data-improving correction scale that stays physically sane."""
    prior = np.asarray(prior, dtype=np.float64)
    raw_delta = np.asarray(raw_delta, dtype=np.float64)

    prior_norm = float(np.linalg.norm(prior)) + 1e-30
    prior_sum = float(np.sum(prior)) + 1e-30
    prior_peak = float(np.max(prior)) + 1e-30
    prior_loss = float(data_term.loss(prior))

    best = None
    fallback = None
    for scale in _candidate_scale_grid():
        raw_u = prior + scale * raw_delta
        u = np.maximum(raw_u, 0.0) if cfg.enforce_nonnegative_initialization else raw_u

        rel_change = float(np.linalg.norm(u - prior) / prior_norm)
        total_factor = float(np.sum(u) / prior_sum)
        peak_factor = float(np.max(u) / prior_peak)
        data_loss = float(data_term.loss(u))
        improves = data_loss <= prior_loss * float(cfg.max_allowed_data_loss_factor)
        if cfg.require_data_improvement:
            improves = improves and data_loss <= prior_loss

        feasible_physical = (
            rel_change <= cfg.max_initial_relative_change
            and cfg.min_initial_total_brightness_factor <= total_factor <= cfg.max_initial_total_brightness_factor
            and peak_factor <= cfg.max_initial_peak_factor
        )

        row = {
            "chosen_scale": float(scale),
            "candidate_data_loss": data_loss,
            "candidate_prior_data_loss": prior_loss,
            "candidate_data_loss_ratio_to_prior": float(data_loss / (prior_loss + 1e-30)),
            "candidate_relative_change": rel_change,
            "candidate_total_brightness_factor": total_factor,
            "candidate_peak_factor": peak_factor,
            "candidate_improves_data": bool(improves),
            "candidate_feasible_physical": bool(feasible_physical),
        }

        if feasible_physical:
            # Fallback: best physically safe point even if it does not beat the prior.
            if fallback is None or data_loss < fallback[0]:
                fallback = (data_loss, scale, u, row)

            # Main selection: among safe and data-improving points, minimize data loss.
            if improves and (best is None or data_loss < best[0]):
                best = (data_loss, scale, u, row)

    if best is not None:
        _, scale, u, row = best
        row["accepted_data_improving_scale"] = True
        return u, row

    if fallback is not None:
        _, scale, u, row = fallback
        row["accepted_data_improving_scale"] = False
        return u, row

    # This should almost never happen because scale=0 is physically feasible.
    u = np.maximum(prior, 0.0)
    return u, {
        "chosen_scale": 0.0,
        "candidate_data_loss": prior_loss,
        "candidate_prior_data_loss": prior_loss,
        "candidate_data_loss_ratio_to_prior": 1.0,
        "candidate_relative_change": 0.0,
        "candidate_total_brightness_factor": 1.0,
        "candidate_peak_factor": 1.0,
        "candidate_improves_data": False,
        "candidate_feasible_physical": True,
        "accepted_data_improving_scale": False,
    }


def residual_mass_stats(u, prior):
    delta = np.asarray(u, dtype=np.float64) - np.asarray(prior, dtype=np.float64)[None, :, :]
    positive = np.maximum(delta, 0.0).sum(axis=(1, 2))
    negative = np.maximum(-delta, 0.0).sum(axis=(1, 2))
    return positive, negative


def apply_residual_mass_cap(u, prior, cfg):
    """Shrink extreme residual frames so balanced OT is not given absurd masses."""
    u = np.asarray(u, dtype=np.float64).copy()
    prior_stack = np.asarray(prior, dtype=np.float64)[None, :, :]
    positive, negative = residual_mass_stats(u, prior)

    pos_med = float(np.median(positive[positive > 1e-30])) if np.any(positive > 1e-30) else 0.0
    neg_med = float(np.median(negative[negative > 1e-30])) if np.any(negative > 1e-30) else 0.0
    cap = float(cfg.residual_mass_cap_factor)

    scales = []
    for k in range(len(u)):
        scale = 1.0
        if pos_med > 0 and positive[k] > cap * pos_med:
            scale = min(scale, cap * pos_med / (positive[k] + 1e-30))
        if neg_med > 0 and negative[k] > cap * neg_med:
            scale = min(scale, cap * neg_med / (negative[k] + 1e-30))
        if scale < 1.0:
            u[k] = np.maximum(prior + scale * (u[k] - prior), 0.0)
        scales.append(scale)

    return u, {
        "mass_cap_scale_min": float(np.min(scales)),
        "mass_cap_scale_mean": float(np.mean(scales)),
        "positive_residual_mass_median_before_cap": pos_med,
        "negative_residual_mass_median_before_cap": neg_med,
    }


def residual_mass_diagnostics(label, u, prior):
    positive, negative = residual_mass_stats(u, prior)
    return {
        f"{label}_positive_residual_mass_min": float(positive.min()),
        f"{label}_positive_residual_mass_max": float(positive.max()),
        f"{label}_positive_residual_mass_mean": float(positive.mean()),
        f"{label}_negative_residual_mass_min": float(negative.min()),
        f"{label}_negative_residual_mass_max": float(negative.max()),
        f"{label}_negative_residual_mass_mean": float(negative.mean()),
    }


def run_bounded_ring_supported_augmentation(data_terms, prior, cfg):
    """Bounded, smooth, ring-supported data augmentation of the static prior."""
    started = time.perf_counter()
    support = build_ring_support_mask(prior, cfg)

    def augment_frame(k):
        target = data_terms[k].f - data_terms[k].sampler.forward(prior)
        raw_delta, info = constrained_visibility_correction(data_terms[k], target, support, cfg)
        u, scale_info = choose_safe_correction_scale(prior, raw_delta, data_terms[k], cfg)

        postclip_residual = data_terms[k].sampler.forward(u) - data_terms[k].f
        target_norm = np.linalg.norm(_stack_complex(data_terms[k].f)) + 1e-30
        prior_loss = float(data_terms[k].loss(prior))
        postclip_loss = float(data_terms[k].loss(u))
        info = {
            "frame": k,
            **info,
            **scale_info,
            "postclip_data_loss": postclip_loss,
            "prior_data_loss": prior_loss,
            "data_loss_improvement_factor": float(prior_loss / (postclip_loss + 1e-30)),
            "postclip_visibility_relative_residual": float(
                np.linalg.norm(_stack_complex(postclip_residual)) / target_norm
            ),
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

    u, mass_cap_info = apply_residual_mass_cap(u, prior, cfg)

    # Refresh final per-frame losses after possible mass capping.
    for k, info in enumerate(frame_infos):
        info["final_data_loss"] = float(data_terms[k].loss(u[k]))
        info["final_relative_change_from_prior"] = float(
            np.linalg.norm(u[k] - prior) / (np.linalg.norm(prior) + 1e-30)
        )
        info["final_total_brightness_factor"] = float(
            np.sum(u[k]) / (np.sum(prior) + 1e-30)
        )
        info["final_peak_factor"] = float(
            np.max(u[k]) / (np.max(prior) + 1e-30)
        )

    elapsed = time.perf_counter() - started

    losses = np.asarray([info["final_data_loss"] for info in frame_infos])
    prior_losses = np.asarray([info["prior_data_loss"] for info in frame_infos])
    preclip_residuals = np.asarray([
        info["preclip_visibility_relative_residual"] for info in frame_infos
    ])
    postclip_residuals = np.asarray([
        info["postclip_visibility_relative_residual"] for info in frame_infos
    ])
    cg_residuals = np.asarray([info["cg_relative_residual"] for info in frame_infos])
    chosen_scales = np.asarray([info["chosen_scale"] for info in frame_infos])
    final_rel = np.asarray([info["final_relative_change_from_prior"] for info in frame_infos])
    final_total = np.asarray([info["final_total_brightness_factor"] for info in frame_infos])
    final_peak = np.asarray([info["final_peak_factor"] for info in frame_infos])

    aggregate = {
        "stage": "bounded_ring_supported_augmentation",
        "seconds": elapsed,
        "cg_iterations_max": int(max(info["cg_iterations"] for info in frame_infos)),
        "cg_relative_residual_max": float(cg_residuals.max()),
        "preclip_visibility_relative_residual_mean": float(preclip_residuals.mean()),
        "preclip_visibility_relative_residual_max": float(preclip_residuals.max()),
        "postclip_visibility_relative_residual_mean": float(postclip_residuals.mean()),
        "postclip_visibility_relative_residual_max": float(postclip_residuals.max()),
        "prior_data_loss_mean": float(prior_losses.mean()),
        "final_data_loss_mean": float(losses.mean()),
        "final_data_loss_max": float(losses.max()),
        "data_loss_improvement_factor_mean": float(np.mean(prior_losses / (losses + 1e-30))),
        "chosen_scale_min": float(chosen_scales.min()),
        "chosen_scale_mean": float(chosen_scales.mean()),
        "chosen_scale_max": float(chosen_scales.max()),
        "accepted_data_improving_frames": int(sum(bool(info["accepted_data_improving_scale"]) for info in frame_infos)),
        "final_relative_change_from_prior_min": float(final_rel.min()),
        "final_relative_change_from_prior_mean": float(final_rel.mean()),
        "final_relative_change_from_prior_max": float(final_rel.max()),
        "final_total_brightness_factor_min": float(final_total.min()),
        "final_total_brightness_factor_max": float(final_total.max()),
        "final_peak_factor_max": float(final_peak.max()),
        **mass_cap_info,
        **residual_mass_diagnostics("spatial", u, prior),
    }
    history = [{"aggregate": aggregate, "frames": frame_infos}]

    print(
        "bounded ring-supported augmentation | "
        f"scale_mean={aggregate['chosen_scale_mean']:.3e} "
        f"| accepted={aggregate['accepted_data_improving_frames']}/{len(data_terms)} "
        f"| rel_change_max={aggregate['final_relative_change_from_prior_max']:.3e} "
        f"| total_factor=[{aggregate['final_total_brightness_factor_min']:.3e}, "
        f"{aggregate['final_total_brightness_factor_max']:.3e}] "
        f"| data_improve_mean={aggregate['data_loss_improvement_factor_mean']:.3e}"
    )

    converged = bool(
        aggregate["final_relative_change_from_prior_max"] <= cfg.max_initial_relative_change + 1e-12
        and aggregate["final_total_brightness_factor_min"] >= cfg.min_initial_total_brightness_factor - 1e-12
        and aggregate["final_total_brightness_factor_max"] <= cfg.max_initial_total_brightness_factor + 1e-12
        and aggregate["final_peak_factor_max"] <= cfg.max_initial_peak_factor + 1e-12
    )
    return u, history, elapsed, converged


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
            f"This edited main4.py should run all 15 frames: got frames={cfg.frames}, "
            f"frame_indices={cfg.frame_indices}"
        )
    cfg.output_root.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("MAIN4.PY: RADIAL PRIOR -> BOUNDED RING-SUPPORTED AUGMENTATION -> RESIDUAL OT")
    print("Outputs are intentionally minimal: results.npz and one comparison video.")
    print("=" * 100)
    print("Objective being tested")
    print("=" * 100)
    print("Bounded ring-supported visibility augmentation:")
    print("  u_k = max(prior + delta_k, 0)")
    print("  delta_k = M G_sigma z_k is smooth/ring-supported and line-searched for safe data improvement")
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
    print("1. Bounded ring-supported visibility augmentation")
    print("=" * 100)
    u_spatial, spatial_history, spatial_elapsed, spatial_converged = run_bounded_ring_supported_augmentation(
        data_terms,
        prior,
        cfg,
    )
    spatial_summary = summary_row(
        "bounded_ring_supported_augmentation",
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
    print("2. Residual signed OT initialized from bounded ring-supported augmentation")
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
        **residual_mass_diagnostics("spatial", u_spatial, prior),
        **residual_mass_diagnostics("ot", u_ot, prior),
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
