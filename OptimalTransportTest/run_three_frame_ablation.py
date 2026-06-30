"""Fast three-frame regularization ablation.

Runs the same averaged-image initialization through:

    1. data only
    2. data + TV
    3. data + TV + averaged-image prior

The convex spatial problems are iterated until their full objective and image
iterates stabilize. The data-only stage uses a projected-gradient KKT residual
because that underdetermined problem can keep moving in visibility null space.

This file expects the project NPZ layout:

    one structured array named ``data`` with fields ``u``, ``v``, ``vis``,
    and ``sigma``.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from data_terms import ComplexVisibilityDataTerm
from io_utils import load_prior_image, save_frames
from operators import VisibilitySampler, grad
from solvers import TotalVariationRegularizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Config:
    obs_folder: Path = PROJECT_ROOT / "blackhole_sim_testing" / "observations_3min_npz"
    prior_path: Path = PROJECT_ROOT / "blackhole_sim" / "time_avg_static_recon_128pix.png"
    output_root: Path = Path("three_frame_ablation")
    frames: int = 3
    # Use spaced frames by default so a three-frame experiment has a better
    # chance of containing visible dynamics than frame_00/frame_01/frame_02.
    # Set to None to use the first `frames` files instead.
    frame_indices: tuple[int, ...] | None = (0, 7, 14)
    image_height: int = 128
    image_width: int = 128
    fov_rad: float = 160e-6 / 206265.0
    # At 128x128, caching dense direct Fourier matrices can be memory-heavy
    # when using many visibilities. Use None for the final full-data run.
    max_vis_per_frame: int | None = 750
    use_hermitian: bool = False
    seed: int = 0

    # Starting values, not conclusions. Tune these after inspecting term scales.
    tv_weight: float = 3e-5
    prior_weight: float = 5e-2

    # Spatial convergence: each outer block performs spatial_inner_iters updates.
    spatial_inner_iters: int = 25
    spatial_max_blocks: int = 40
    spatial_min_blocks: int = 3
    convergence_patience: int = 3
    objective_rel_tol: float = 1e-5
    iterate_rel_tol: float = 1e-4
    projected_gradient_tol: float = 2e-5
    parallel_frames: bool = True
    # This is a cap. The solver derives the actual safe step from ||S*S||.
    primal_tau: float = 10.0
    dual_sigma: float = 0.25


STAGES = (
    ("01_data", 0.0, 0.0),
    ("02_data_tv", "tv", 0.0),
    ("03_data_tv_prior", "tv", "prior"),
)


def _read_visibility_arrays(path: Path):
    """Read the structured project NPZ format."""
    with np.load(path, allow_pickle=False) as loaded:
        if "data" not in loaded:
            raise KeyError(f"{path.name}: expected structured array key 'data'")

        data = loaded["data"]
        names = data.dtype.names or ()
        missing = [key for key in ("u", "v", "vis", "sigma") if key not in names]
        if missing:
            raise KeyError(f"{path.name}: structured 'data' is missing {missing}")
        return tuple(np.asarray(data[key]) for key in ("u", "v", "vis", "sigma"))


def _hermitian_augment(u, v, vis, sigma):
    return (
        np.concatenate((u, -u)),
        np.concatenate((v, -v)),
        np.concatenate((vis, np.conj(vis))),
        np.concatenate((sigma, sigma)),
    )


def load_data_terms(cfg: Config):
    all_files = sorted(cfg.obs_folder.glob("*.npz"))
    if cfg.frame_indices is None:
        files = all_files[: cfg.frames]
    else:
        files = []
        for index in cfg.frame_indices:
            if index < 0 or index >= len(all_files):
                raise IndexError(
                    f"frame_indices contains {index}, but only "
                    f"{len(all_files)} NPZ files were found in {cfg.obs_folder}"
                )
            files.append(all_files[index])

    if not files:
        raise FileNotFoundError(f"No NPZ files found in {cfg.obs_folder}")

    rng = np.random.default_rng(cfg.seed)
    raw = []
    amplitudes = []
    for path in files:
        u, v, vis, sigma = _read_visibility_arrays(path)
        u = np.asarray(u, dtype=np.float64).ravel()
        v = np.asarray(v, dtype=np.float64).ravel()
        vis = np.asarray(vis, dtype=np.complex128).ravel()
        sigma = np.asarray(sigma, dtype=np.float64).ravel()
        if not (u.size == v.size == vis.size == sigma.size):
            raise ValueError(f"{path.name}: u, v, vis, sigma lengths differ")

        good = (
            np.isfinite(u) & np.isfinite(v)
            & np.isfinite(vis.real) & np.isfinite(vis.imag)
            & np.isfinite(sigma) & (sigma > 0)
        )
        u, v, vis, sigma = u[good], v[good], vis[good], sigma[good]

        if cfg.max_vis_per_frame is not None and u.size > cfg.max_vis_per_frame:
            keep = rng.choice(u.size, cfg.max_vis_per_frame, replace=False)
            u, v, vis, sigma = u[keep], v[keep], vis[keep], sigma[keep]
        if cfg.use_hermitian:
            u, v, vis, sigma = _hermitian_augment(u, v, vis, sigma)

        raw.append((path.name, u, v, vis, sigma))
        amplitudes.append(np.abs(vis))

    data_scale = float(np.percentile(np.concatenate(amplitudes), 95) + 1e-12)
    terms, names = [], []
    shape = (cfg.image_height, cfg.image_width)
    print(f"Global visibility scale: {data_scale:.3e}")
    for name, u, v, vis, sigma in raw:
        weight = sigma ** -2
        weight /= np.median(weight) + 1e-12
        sampler = VisibilitySampler(
            u=u,
            v=v,
            weight=weight,
            shape=shape,
            fov_rad=cfg.fov_rad,
            data_scale=data_scale,
        )
        observed = np.sqrt(weight) * vis / sampler.total_scale
        terms.append(ComplexVisibilityDataTerm(sampler=sampler, f=observed))
        names.append(name)
        print(f"Loaded {name}: {u.size} visibilities")
    return terms, names


def tv_value(image):
    derivative = grad(image)
    return float(np.sum(np.sqrt(derivative[0] ** 2 + derivative[1] ** 2)))


def objective_terms(u, data_terms, prior, tv_weight, prior_weight):
    data = float(sum(term.loss(u[k]) for k, term in enumerate(data_terms)))
    tv = float(tv_weight * sum(tv_value(frame) for frame in u))
    prior_loss = float(0.5 * prior_weight * np.sum((u - prior[None, :, :]) ** 2))
    return {"data": data, "tv": tv, "prior": prior_loss, "spatial_total": data + tv + prior_loss}


def relative_change(new, old):
    return float(np.linalg.norm(new - old) / (np.linalg.norm(old) + 1e-12))


def run_spatial_stage(label, u_init, data_terms, prior, tv_weight, prior_weight, cfg):
    """Solve a spatial convex stage in blocks and apply a two-part stop test."""
    regularizer = TotalVariationRegularizer(
        alpha=tv_weight,
        iters=cfg.spatial_inner_iters,
        tau=cfg.primal_tau,
        sigma=cfg.dual_sigma,
    )
    u = u_init.copy()
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
                admm_weight=prior_weight,
                target_mass=None,
            )

        if cfg.parallel_frames and len(data_terms) > 1:
            with ThreadPoolExecutor(max_workers=len(data_terms)) as executor:
                updated = list(executor.map(update_frame, range(len(data_terms))))
            u = np.stack(updated)
        else:
            for k in range(len(data_terms)):
                u[k] = update_frame(k)

        terms = objective_terms(u, data_terms, prior, tv_weight, prior_weight)
        du = relative_change(u, old)
        if previous_objective is None:
            dobj = np.inf
        else:
            dobj = abs(terms["spatial_total"] - previous_objective) / (
                abs(previous_objective) + 1e-12
            )
        previous_objective = terms["spatial_total"]
        if tv_weight == 0.0:
            kkt_residual = max(
                regularizer.projected_gradient_residual(
                    u[k], data_terms[k], prior, prior_weight
                )
                for k in range(len(data_terms))
            )
        else:
            kkt_residual = np.nan
        row = {"block": block, "updates": block * cfg.spatial_inner_iters,
               "relative_iterate_change": du, "relative_objective_change": dobj, **terms}
        row["projected_gradient_residual_max"] = kkt_residual
        history.append(row)
        print(
            f"{label} block {block:03d} | objective={terms['spatial_total']:.6e} "
            f"| d_obj={dobj:.3e} | d_u={du:.3e} | KKT={kkt_residual:.3e}"
        )

        if tv_weight == 0.0:
            # The data-only problem is highly underdetermined, so image-change
            # convergence is neither necessary nor generally fast. The
            # projected-gradient mapping is its correct first-order test.
            is_stable = kkt_residual < cfg.projected_gradient_tol
        else:
            is_stable = dobj < cfg.objective_rel_tol and du < cfg.iterate_rel_tol
        stable = stable + 1 if block >= cfg.spatial_min_blocks and is_stable else 0
        if stable >= cfg.convergence_patience:
            print(f"{label}: converged after {block * cfg.spatial_inner_iters} updates")
            break

    elapsed = time.perf_counter() - started
    converged = stable >= cfg.convergence_patience
    return u, history, elapsed, converged


def save_stage(label, u, history, names, output_root):
    stage_dir = output_root / label
    stage_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(stage_dir / "reconstruction.npz", frames=u)
    pd.DataFrame(history).to_csv(stage_dir / "history.csv", index=False)
    save_frames(u, stage_dir / "frames_each_norm", names=names, normalize_each=True)
    save_frames(u, stage_dir / "frames_global_norm", names=names, normalize_each=False)


def main():
    cfg = Config()
    cfg.output_root.mkdir(parents=True, exist_ok=True)
    serializable = asdict(cfg)
    serializable.update({key: str(value) for key, value in serializable.items() if isinstance(value, Path)})
    (cfg.output_root / "config.json").write_text(json.dumps(serializable, indent=2))

    data_terms, names = load_data_terms(cfg)
    prior = load_prior_image(
        cfg.prior_path,
        resize=(cfg.image_width, cfg.image_height),
        normalize=True,
    )
    # Every stage starts at precisely the same averaged reconstruction.
    u_init = np.repeat(prior[None, :, :], len(data_terms), axis=0)
    save_frames(u_init, cfg.output_root / "00_average_initialization", names=names)

    summaries = []
    for label, tv_flag, prior_flag in STAGES:
        print("\n" + "=" * 100)
        print(label)
        print("=" * 100)
        tv_weight = cfg.tv_weight if tv_flag else 0.0
        prior_weight = cfg.prior_weight if prior_flag else 0.0

        u, history, elapsed, converged = run_spatial_stage(
            label, u_init, data_terms, prior, tv_weight, prior_weight, cfg
        )

        save_stage(label, u, history, names, cfg.output_root)
        terms = objective_terms(u, data_terms, prior, tv_weight, prior_weight)
        summaries.append({
            "stage": label,
            "converged": converged,
            "seconds": elapsed,
            "relative_change_from_average": relative_change(u, u_init),
            **terms,
            "note": "",
        })

    summary = pd.DataFrame(summaries)
    summary.to_csv(cfg.output_root / "summary.csv", index=False)
    print("\n" + summary.to_string(index=False))
    print(f"\nOutputs saved to {cfg.output_root.resolve()}")


if __name__ == "__main__":
    main()
