import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage.metrics import structural_similarity as ssim

from admm import ADMM
from io_utils import (
    load_npz_visibility_data_terms,
    load_prior_image,
    save_frames,
    images_to_video,
)
from operators import normalize01, grad
from solvers import TotalVariationRegularizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def calibrate_weights(u_init, data_terms, prior_image, target_tv_frac=0.03, target_prior_frac=0.20):
    data = sum(dt.loss(u_init[k]) for k, dt in enumerate(data_terms))

    tv_raw = sum(tv_value(u_init[k]) for k in range(len(data_terms)))

    prior_raw = 0.0
    for k in range(len(data_terms)):
        prior_raw += 0.5 * np.sum((u_init[k] - prior_image) ** 2)

    tv_alpha = target_tv_frac * data / (tv_raw + 1e-12)
    prior_weight = target_prior_frac * data / (prior_raw + 1e-12)

    print("\nAUTO-SCALED WEIGHTS")
    print(f"data loss:      {data:.3e}")
    print(f"raw TV:         {tv_raw:.3e}")
    print(f"raw prior:      {prior_raw:.3e}")
    print(f"tv_alpha:       {tv_alpha:.3e}")
    print(f"prior_weight:   {prior_weight:.3e}")

    return tv_alpha, prior_weight


# ============================================================
# GROUND TRUTH / METRICS
# ============================================================

def load_ground_truth_frames(folder, max_frames=3, resize=(64, 64)):
    folder = Path(folder)

    if not folder.exists():
        raise FileNotFoundError(f"Ground-truth folder not found: {folder}")

    files = sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])[:max_frames]

    if len(files) == 0:
        raise ValueError(f"No ground-truth image files found in {folder}")

    frames = []

    for name in files:
        path = folder / name

        img = Image.open(path).convert("L")
        img = img.resize(resize)

        arr = np.asarray(img, dtype=np.float64) / 255.0
        arr = normalize01(arr)

        frames.append(arr)

    return np.stack(frames), files


def compute_nrmse(gt, test):
    gt = np.asarray(gt, dtype=np.float64)
    test = np.asarray(test, dtype=np.float64)

    rmse = np.sqrt(np.mean((gt - test) ** 2))
    denom = gt.max() - gt.min()

    return rmse / (denom + 1e-12)


def evaluate_reconstruction(u_true, u_dirty, u_rec, frame_names, model_name):
    K = min(len(u_true), len(u_dirty), len(u_rec))

    rows = []

    for k in range(K):
        gt = normalize01(u_true[k])
        dirty = normalize01(u_dirty[k])
        rec = normalize01(u_rec[k])

        rows.append({
            "model": model_name,
            "frame": frame_names[k] if k < len(frame_names) else f"frame_{k:02d}",
            "dirty_nrmse": compute_nrmse(gt, dirty),
            "rec_nrmse": compute_nrmse(gt, rec),
            "dirty_ssim": ssim(gt, dirty, data_range=1.0),
            "rec_ssim": ssim(gt, rec, data_range=1.0),
        })

    return pd.DataFrame(rows)


# ============================================================
# OBJECTIVE REPORTING
# ============================================================

def tv_value(u):
    g = grad(u)
    return np.sum(np.sqrt(g[0] ** 2 + g[1] ** 2 + 1e-12))


def objective_components(
    u,
    data_terms,
    prior_image=None,
    tv_alpha=0.0,
    prior_weight=0.0,
):
    data = 0.0
    tv = 0.0
    prior = 0.0

    for k in range(len(data_terms)):
        data += data_terms[k].loss(u[k])

        if tv_alpha > 0:
            tv += tv_alpha * tv_value(u[k])

        if prior_image is not None and prior_weight > 0:
            p = prior_image if prior_image.ndim == 2 else prior_image[k]
            prior += 0.5 * prior_weight * np.sum((u[k] - p) ** 2)

    total = data + tv + prior

    return {
        "data_obj": float(data),
        "tv_obj": float(tv),
        "prior_obj": float(prior),
        "total_obj": float(total),
        "data_pct": float(100 * data / (total + 1e-12)),
        "tv_pct": float(100 * tv / (total + 1e-12)),
        "prior_pct": float(100 * prior / (total + 1e-12)),
    }


def print_objective_report(label, comps):
    print("\n" + "=" * 80)
    print(f"OBJECTIVE REPORT: {label}")
    print("=" * 80)
    print(f"data   = {comps['data_obj']:.6e}  ({comps['data_pct']:5.1f}%)")
    print(f"TV     = {comps['tv_obj']:.6e}  ({comps['tv_pct']:5.1f}%)")
    print(f"prior  = {comps['prior_obj']:.6e}  ({comps['prior_pct']:5.1f}%)")
    print(f"total  = {comps['total_obj']:.6e}")
    print("=" * 80)


# ============================================================
# INIT
# ============================================================

def make_dirty_initialization(data_terms):
    dirty = []

    for dt in data_terms:
        img = dt.dirty_image()
        img = np.maximum(img, 0.0)
        img = normalize01(img)
        dirty.append(img)

    return np.stack(dirty)


# ============================================================
# SINGLE TOURNAMENT RUN
# ============================================================

def run_model(
    name,
    data_terms,
    u_init,
    prior_image,
    tv_alpha,
    prior_weight,
    out_root,
    obs_names,
    max_iter=12,
    regularizer_iters=25,
):
    print("\n" + "#" * 100)
    print(f"RUNNING MODEL: {name}")
    print("#" * 100)

    model_folder = out_root / name
    model_folder.mkdir(parents=True, exist_ok=True)

    regularizer = TotalVariationRegularizer(
        alpha=tv_alpha,
        iters=regularizer_iters,
        tau=3e-3,
        sigma=5e-3,
    )

    model = ADMM(
        data_terms=data_terms,
        regularizer=regularizer,

        # No temporal OT in tournament.
        beta=0.0,
        eta=0.0,

        max_iter=max_iter,
        abs_tol=1e-4,
        rel_tol=5e-3,
        min_iter=4,
        patience=3,

        transport_T=5,
        transport_inner_iters=5,

        dual_relaxation=1.0,
        enforce_equal_mass=False,

        prior_image=prior_image,
        prior_weight=prior_weight,

        stop_on_data_plateau=True,
        data_plateau_window=5,
        data_plateau_tol=1e-4,
    )

    init_comps = objective_components(
        u_init,
        data_terms,
        prior_image=prior_image,
        tv_alpha=tv_alpha,
        prior_weight=prior_weight,
    )
    print_objective_report(f"{name} initial", init_comps)

    u_rec, history = model.run(u_init.copy())

    final_comps = objective_components(
        u_rec,
        data_terms,
        prior_image=prior_image,
        tv_alpha=tv_alpha,
        prior_weight=prior_weight,
    )
    print_objective_report(f"{name} final", final_comps)

    save_frames(
        frames=u_rec,
        output_folder=model_folder / "frames_each_norm",
        names=obs_names,
        normalize_each=True,
    )

    save_frames(
        frames=u_rec,
        output_folder=model_folder / "frames_global_norm",
        names=obs_names,
        normalize_each=False,
    )

    pd.DataFrame(history).to_csv(model_folder / "history.csv", index=False)

    summary = {
        "model": name,
        "tv_alpha": tv_alpha,
        "prior_weight": prior_weight,
        "init_data_obj": init_comps["data_obj"],
        "init_tv_obj": init_comps["tv_obj"],
        "init_prior_obj": init_comps["prior_obj"],
        "init_total_obj": init_comps["total_obj"],
        "final_data_obj": final_comps["data_obj"],
        "final_tv_obj": final_comps["tv_obj"],
        "final_prior_obj": final_comps["prior_obj"],
        "final_total_obj": final_comps["total_obj"],
        "final_data_pct": final_comps["data_pct"],
        "final_tv_pct": final_comps["tv_pct"],
        "final_prior_pct": final_comps["prior_pct"],
        "u_mean": float(u_rec.mean()),
        "u_max": float(u_rec.max()),
        "u_min": float(u_rec.min()),
    }

    return u_rec, history, summary


# ============================================================
# MAIN TOURNAMENT
# ============================================================

def main():
    # ============================================================
    # PATHS
    # ============================================================

    obs_folder = PROJECT_ROOT / "blackhole_sim_testing" / "observations_npz"
    gt_folder = PROJECT_ROOT / "blackhole_sim" / "data" / "aart_frames"
    prior_path = PROJECT_ROOT / "blackhole_sim" / "time_avg_static_recon_128pix.png"

    out_root = Path("./quick_tournament_outputs")
    out_root.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # FAST DEV SETTINGS
    # ============================================================

    K = 3
    image_shape = (64, 64)
    fov_rad = 160e-6 / 206265.0

    max_vis_per_frame = 1000

    # You said data already includes Hermitian conjugates.
    use_hermitian = False

    # Keep this small.
    max_iter = 25
    regularizer_iters = 40

    # ============================================================
    # LOAD DATA
    # ============================================================

    print("=" * 100)
    print("LOADING QUICK TOURNAMENT DATA")
    print("=" * 100)

    data_terms, obs_names = load_npz_visibility_data_terms(
        folder=obs_folder,
        max_frames=K,
        image_shape=image_shape,
        fov_rad=fov_rad,
        max_vis_per_frame=max_vis_per_frame,
        use_hermitian=use_hermitian,
    )

    K = len(data_terms)

    # ============================================================
    # LOAD GT + PRIOR
    # ============================================================

    u_true, gt_names = load_ground_truth_frames(
        folder=gt_folder,
        max_frames=K,
        resize=image_shape[::-1],
    )

    prior_image = load_prior_image(
        prior_path,
        resize=image_shape[::-1],
        normalize=True,
    )

    prior_stack = np.repeat(prior_image[None, :, :], K, axis=0)

    # ============================================================
    # INITIALIZATION
    # ============================================================

    u_dirty = make_dirty_initialization(data_terms)

    save_frames(
        frames=u_dirty,
        output_folder=out_root / "dirty_init",
        names=obs_names,
        normalize_each=True,
    )

    save_frames(
        frames=prior_stack,
        output_folder=out_root / "prior_stack",
        names=obs_names,
        normalize_each=True,
    )

    # Use a mostly data-based init so we can see whether each objective moves
    # toward or away from the pooled prior.
    u_init = 0.75 * u_dirty + 0.25 * prior_stack
    u_init = np.maximum(u_init, 0.0)

    save_frames(
        frames=u_init,
        output_folder=out_root / "initialization",
        names=obs_names,
        normalize_each=True,
    )

    # ============================================================
    # TOURNAMENT DEFINITIONS
    # ============================================================
    tv_3pct, prior_20pct = calibrate_weights(
        u_init,
        data_terms,
        prior_image,
        target_tv_frac=0.03,
        target_prior_frac=0.20,
    )

    tv_1pct, prior_10pct = calibrate_weights(
        u_init,
        data_terms,
        prior_image,
        target_tv_frac=0.01,
        target_prior_frac=0.10,
    )

    tournament = [
        {
            "name": "A_data_only",
            "tv_alpha": 0.0,
            "prior_weight": 0.0,
        },
        {
            "name": "B_data_TV_scaled",
            "tv_alpha": tv_3pct,
            "prior_weight": 0.0,
        },
        {
            "name": "C_data_prior_scaled",
            "tv_alpha": 0.0,
            "prior_weight": prior_20pct,
        },
        {
            "name": "D_data_TV_prior_scaled",
            "tv_alpha": tv_1pct,
            "prior_weight": prior_10pct,
        },
    ]
    summaries = []
    metric_tables = []

    for cfg in tournament:
        u_rec, history, summary = run_model(
            name=cfg["name"],
            data_terms=data_terms,
            u_init=u_init,
            prior_image=prior_image,
            tv_alpha=cfg["tv_alpha"],
            prior_weight=cfg["prior_weight"],
            out_root=out_root,
            obs_names=obs_names,
            max_iter=max_iter,
            regularizer_iters=regularizer_iters,
        )

        summaries.append(summary)

        metrics = evaluate_reconstruction(
            u_true=u_true,
            u_dirty=u_dirty,
            u_rec=u_rec,
            frame_names=gt_names,
            model_name=cfg["name"],
        )

        metrics.to_csv(out_root / cfg["name"] / "metrics.csv", index=False)
        metric_tables.append(metrics)

    summary_df = pd.DataFrame(summaries)
    metrics_df = pd.concat(metric_tables, ignore_index=True)

    summary_df.to_csv(out_root / "tournament_summary.csv", index=False)
    metrics_df.to_csv(out_root / "tournament_metrics.csv", index=False)

    print("\n" + "=" * 100)
    print("TOURNAMENT SUMMARY")
    print("=" * 100)

    display_cols = [
        "model",
        "tv_alpha",
        "prior_weight",
        "final_data_obj",
        "final_tv_obj",
        "final_prior_obj",
        "final_total_obj",
        "final_data_pct",
        "final_tv_pct",
        "final_prior_pct",
        "u_mean",
        "u_max",
    ]

    print(summary_df[display_cols].to_string(index=False))

    print("\n" + "=" * 100)
    print("MEAN METRICS")
    print("=" * 100)

    mean_metrics = metrics_df.groupby("model").agg({
        "dirty_nrmse": "mean",
        "rec_nrmse": "mean",
        "dirty_ssim": "mean",
        "rec_ssim": "mean",
    }).reset_index()

    print(mean_metrics.to_string(index=False))

    mean_metrics.to_csv(out_root / "tournament_mean_metrics.csv", index=False)

    print("\nDone.")
    print(f"Outputs saved in: {out_root.resolve()}")


if __name__ == "__main__":
    main()