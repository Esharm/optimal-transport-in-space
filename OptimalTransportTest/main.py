"""TV+prior initialization followed by temporal OT/ADMM.

This is the current experimental pipeline:

    1. Independently reconstruct each selected frame with

           D_k(u_k) + alpha TV(u_k) + (mu/2)||u_k - p||_2^2,

       where p is the averaged-image prior.

    2. Use those non-identical spatial reconstructions as the initialization
       for an ADMM solve of

           sum_k [D_k(u_k) + alpha TV(u_k) + (mu/2)||u_k - p||_2^2]
           + beta sum_k BB(u_k, u_{k+1}).

       The ADMM split introduces endpoint variables

           b0_k = rho_k(t=0),  b1_k = rho_k(t=1),

       and alternates:

           transport step:
               solve a time-discrete Benamou-Brenier problem for b0_k, b1_k
               near the current adjacent image frames;

           image step:
               update u_k using data + TV + prior + quadratic ADMM endpoint
               penalties;

           dual step:
               update multipliers enforcing b0_k ~= u_k and
               b1_k ~= u_{k+1}.

Important: this is still the existing balanced/full-image OT prototype, not
the future unbalanced or residual-OT model.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from admm import ADMM
from io_utils import load_prior_image, save_frames
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
class Config(LoaderConfig):
    output_root: Path = Path("tv_prior_init_then_ot")

    # Spaced frames have visible frame-specific residuals in the diagnostics.
    frame_indices: tuple[int, ...] | None = (0, 7, 14)

    # Chosen over 5e-2 because it keeps more frame-specific deviation while
    # still using the prior to prevent the ugly data-only solution.
    prior_weight: float = 1e-2
    tv_weight: float = 3e-5

    # Spatial TV+prior initialization.
    spatial_inner_iters: int = 25
    spatial_max_blocks: int = 50
    spatial_min_blocks: int = 3
    spatial_patience: int = 3
    objective_rel_tol: float = 1e-5
    iterate_rel_tol: float = 1e-4
    primal_tau: float = 10.0
    dual_sigma: float = 0.25

    # Existing full-image balanced OT prototype.
    # If OT does almost nothing, try beta=3e-6. If it erases differences or
    # primal residual grows badly, reduce beta or eta.
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


def run_ot_from_initialization(u_init, data_terms, prior, cfg):
    regularizer = TotalVariationRegularizer(
        alpha=cfg.tv_weight,
        iters=cfg.spatial_inner_iters,
        tau=cfg.primal_tau,
        sigma=cfg.dual_sigma,
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
        # Keep this false for now. The current data may not have equal flux per
        # frame; forcing equal mass is a modeling decision, not a numerical fix.
        enforce_equal_mass=False,
        prior_image=prior,
        prior_weight=cfg.prior_weight,
        stop_on_data_plateau=False,
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
    cfg.output_root.mkdir(parents=True, exist_ok=True)

    serializable = asdict(cfg)
    serializable.update({
        key: str(value)
        for key, value in serializable.items()
        if isinstance(value, Path)
    })
    (cfg.output_root / "config.json").write_text(json.dumps(serializable, indent=2))

    print("=" * 100)
    print("Objective being tested")
    print("=" * 100)
    print("Spatial initialization:")
    print("  sum_k D_k(u_k) + alpha TV(u_k) + (mu/2)||u_k - prior||^2")
    print("OT stage:")
    print("  same spatial objective + beta * sum_k BB(u_k, u_{k+1})")
    print("ADMM image step still contains data + TV + prior + endpoint quadratics.")
    print("This is full-image balanced OT with soft endpoint matching, not unbalanced OT.")

    data_terms, names = load_data_terms(cfg)
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
    print("2. OT initialized from spatial TV+prior")
    print("=" * 100)
    u_ot, ot_history, ot_elapsed, ot_converged = run_ot_from_initialization(
        u_spatial,
        data_terms,
        prior,
        cfg,
    )
    ot_stage_dir = cfg.output_root / "02_ot_from_spatial_init"
    ot_stage_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(ot_history).to_csv(ot_stage_dir / "history.csv", index=False)
    ot_summary = save_run_outputs(
        label="02_ot_from_spatial_init",
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

    summary = pd.DataFrame([
        summary_row("01_spatial_tv_prior_init", u_spatial, data_terms, prior, cfg, spatial_elapsed, spatial_converged),
        summary_row("02_ot_from_spatial_init", u_ot, data_terms, prior, cfg, ot_elapsed, ot_converged),
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
    print("  02_ot_from_spatial_init/montage_delta_signed.png")
    print("  03_ot_minus_spatial/montage_ot_minus_spatial_signed.png")
    print("  03_ot_minus_spatial/montage_pairwise_delta_change_difference_signed.png")


if __name__ == "__main__":
    main()
