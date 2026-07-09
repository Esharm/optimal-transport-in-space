"""Run the current best spatial scheme and visualize frame-specific deltas.

Current best spatial scheme from the ablations:

    data fidelity + TV + averaged-image prior

Wavelet did not improve the objective or frame separation enough to be the
default here.  This runner focuses on the question raised by the diagnostics:

    Are the converged reconstructions actually different from the prior in
    frame-specific ways that are hidden by the common averaged morphology?

For each prior weight, this script runs TV+prior to convergence, then saves:

    - reconstructed frames
    - signed delta images: u_k - prior
    - amplified images: prior + amplification * (u_k - prior)
    - pairwise signed delta differences
    - compact montage PNGs
    - CSV diagnostics

The shared loader defaults to frame_indices=(0, 7, 14), so this tests spaced
frames unless you change Config.frame_indices.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from io_utils import load_prior_image, save_frames
from run_three_frame_ablation import Config as BaseConfig
from run_three_frame_ablation import load_data_terms, objective_terms, relative_change
from solvers import TotalVariationRegularizer


@dataclass(frozen=True)
class Config(BaseConfig):
    output_root: Path = Path("tv_prior_delta_visualization")

    # These are the useful convergent prior strengths from the previous runs.
    # 1e-2 reveals more frame-specific signal; 5e-2 is cleaner and more prior-like.
    prior_weights: tuple[float, ...] = (1e-2, 5e-2)
    tv_weight: float = 3e-5

    spatial_inner_iters: int = 25
    spatial_max_blocks: int = 50
    spatial_min_blocks: int = 3
    convergence_patience: int = 3
    objective_rel_tol: float = 1e-5
    iterate_rel_tol: float = 1e-4
    parallel_frames: bool = True

    primal_tau: float = 10.0
    dual_sigma: float = 0.25
    delta_amplification: float = 6.0


def _to_uint8_global(frames, vmin=None, vmax=None):
    frames = np.asarray(frames, dtype=np.float64)
    if vmin is None:
        vmin = float(np.min(frames))
    if vmax is None:
        vmax = float(np.max(frames))
    scaled = (frames - vmin) / (vmax - vmin + 1e-12)
    return np.round(255.0 * np.clip(scaled, 0.0, 1.0)).astype(np.uint8)


def save_signed_frames(frames, output_folder, names=None, common_abs=None):
    """Save signed arrays with zero mapped to mid-gray on a common scale."""
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    frames = np.asarray(frames, dtype=np.float64)
    if common_abs is None:
        common_abs = float(np.max(np.abs(frames)) + 1e-12)

    for k, frame in enumerate(frames):
        image = 0.5 + 0.5 * frame / common_abs
        image = np.clip(image, 0.0, 1.0)
        image_uint8 = np.round(255.0 * image).astype(np.uint8)
        filename = f"frame_{k:03d}.png" if names is None else f"{Path(str(names[k])).stem}.png"
        Image.fromarray(image_uint8).save(output_folder / filename)


def save_montage(frames, output_path, titles, signed=False, common_abs=None):
    """Save a simple horizontal montage with labels."""
    frames = np.asarray(frames, dtype=np.float64)
    if signed:
        if common_abs is None:
            common_abs = float(np.max(np.abs(frames)) + 1e-12)
        images = np.round(
            255.0 * np.clip(0.5 + 0.5 * frames / common_abs, 0.0, 1.0)
        ).astype(np.uint8)
    else:
        images = _to_uint8_global(frames)

    tile_h, tile_w = images.shape[1], images.shape[2]
    label_h = 22
    gap = 8
    width = len(images) * tile_w + (len(images) - 1) * gap
    height = tile_h + label_h
    canvas = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(canvas)

    for idx, image in enumerate(images):
        x0 = idx * (tile_w + gap)
        canvas.paste(Image.fromarray(image), (x0, label_h))
        draw.text((x0 + 2, 4), str(titles[idx]), fill=0)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def pairwise_delta_differences(deltas, names):
    arrays = []
    labels = []
    for i in range(len(deltas)):
        for j in range(i + 1, len(deltas)):
            arrays.append(deltas[i] - deltas[j])
            labels.append(f"{Path(names[i]).stem}-{Path(names[j]).stem}")
    return np.stack(arrays), labels


def data_loss_matrix(images, data_terms):
    matrix = np.zeros((len(images), len(data_terms)), dtype=np.float64)
    rows = []
    for image_index, image in enumerate(images):
        for data_index, data_term in enumerate(data_terms):
            loss = float(data_term.loss(image))
            matrix[image_index, data_index] = loss
            rows.append({
                "image_index": image_index,
                "data_index": data_index,
                "loss": loss,
            })
    return matrix, rows


def summarize_cross_losses(matrix):
    own = np.diag(matrix)
    cross = matrix[~np.eye(matrix.shape[0], dtype=bool)]
    return {
        "own_data_loss_mean": float(np.mean(own)),
        "cross_data_loss_mean": float(np.mean(cross)),
        "cross_minus_own_data_loss_mean": float(np.mean(cross) - np.mean(own)),
        "cross_over_own_data_loss_ratio": float(np.mean(cross) / (np.mean(own) + 1e-12)),
    }


def pairwise_distance_rows(frames, label):
    rows = []
    for i in range(len(frames)):
        for j in range(i + 1, len(frames)):
            dist = np.linalg.norm(frames[i] - frames[j]) / (
                np.linalg.norm(frames[i]) + np.linalg.norm(frames[j]) + 1e-12
            )
            rows.append({
                "label": label,
                "i": i,
                "j": j,
                "relative_distance": float(dist),
            })
    return rows


def run_tv_prior(prior_weight, u_init, data_terms, prior, cfg):
    regularizer = TotalVariationRegularizer(
        alpha=cfg.tv_weight,
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
                u = np.stack(list(executor.map(update_frame, range(len(data_terms)))))
        else:
            for k in range(len(data_terms)):
                u[k] = update_frame(k)

        terms = objective_terms(u, data_terms, prior, cfg.tv_weight, prior_weight)
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
            f"tv_prior_{prior_weight:.0e} block {block:03d} "
            f"| objective={terms['spatial_total']:.6e} "
            f"| data={terms['data']:.3e} | prior={terms['prior']:.3e} "
            f"| d_obj={dobj:.3e} | d_u={du:.3e}"
        )

        is_stable = dobj < cfg.objective_rel_tol and du < cfg.iterate_rel_tol
        stable = stable + 1 if block >= cfg.spatial_min_blocks and is_stable else 0
        if stable >= cfg.convergence_patience:
            print(
                f"tv_prior_{prior_weight:.0e}: converged after "
                f"{block * cfg.spatial_inner_iters} updates"
            )
            break

    elapsed = time.perf_counter() - started
    return u, history, elapsed, stable >= cfg.convergence_patience


def save_run_outputs(label, u, history, names, prior, data_terms, prior_weight, cfg):
    stage_dir = cfg.output_root / label
    stage_dir.mkdir(parents=True, exist_ok=True)

    prior_sequence = np.repeat(prior[None, :, :], len(u), axis=0)
    deltas = u - prior_sequence
    amplified = np.maximum(prior_sequence + cfg.delta_amplification * deltas, 0.0)
    delta_diffs, diff_labels = pairwise_delta_differences(deltas, names)

    delta_abs = float(np.max(np.abs(deltas)) + 1e-12)
    diff_abs = float(np.max(np.abs(delta_diffs)) + 1e-12)

    np.savez_compressed(
        stage_dir / "tv_prior_reconstruction.npz",
        frames=u,
        prior=prior,
        deltas=deltas,
        amplified=amplified,
        delta_differences=delta_diffs,
        names=np.array(names),
    )

    pd.DataFrame(history).to_csv(stage_dir / "history.csv", index=False)
    save_frames(u, stage_dir / "frames_each_norm", names=names, normalize_each=True)
    save_frames(u, stage_dir / "frames_global_norm", names=names, normalize_each=False)
    save_frames(amplified, stage_dir / "prior_plus_amplified_delta", names=names, normalize_each=True)
    save_signed_frames(deltas, stage_dir / "delta_signed_common_scale", names=names, common_abs=delta_abs)
    save_signed_frames(delta_diffs, stage_dir / "pairwise_delta_difference_signed", names=diff_labels, common_abs=diff_abs)

    save_montage(
        u,
        stage_dir / "montage_reconstruction_global.png",
        [Path(name).stem for name in names],
        signed=False,
    )
    save_montage(
        deltas,
        stage_dir / "montage_delta_signed.png",
        [f"{Path(name).stem}-prior" for name in names],
        signed=True,
        common_abs=delta_abs,
    )
    save_montage(
        amplified,
        stage_dir / "montage_prior_plus_amplified_delta.png",
        [f"{Path(name).stem} amplified" for name in names],
        signed=False,
    )
    save_montage(
        delta_diffs,
        stage_dir / "montage_pairwise_delta_difference_signed.png",
        diff_labels,
        signed=True,
        common_abs=diff_abs,
    )

    matrix, loss_rows = data_loss_matrix(u, data_terms)
    for row in loss_rows:
        row["stage"] = label
        row["prior_weight"] = prior_weight
    pd.DataFrame(loss_rows).to_csv(stage_dir / "cross_data_losses.csv", index=False)

    distance_rows = (
        pairwise_distance_rows(u, "reconstruction")
        + pairwise_distance_rows(deltas, "delta")
        + pairwise_distance_rows(delta_diffs, "pairwise_delta_difference")
    )
    for row in distance_rows:
        row["stage"] = label
        row["prior_weight"] = prior_weight
    pd.DataFrame(distance_rows).to_csv(stage_dir / "pairwise_distances.csv", index=False)

    loss_summary = summarize_cross_losses(matrix)
    mean_u_distance = float(np.mean([
        row["relative_distance"] for row in distance_rows
        if row["label"] == "reconstruction"
    ]))
    mean_delta_distance = float(np.mean([
        row["relative_distance"] for row in distance_rows
        if row["label"] == "delta"
    ]))
    terms = objective_terms(u, data_terms, prior, cfg.tv_weight, prior_weight)

    return {
        "stage": label,
        "prior_weight": prior_weight,
        "tv_weight": cfg.tv_weight,
        "delta_amplification": cfg.delta_amplification,
        "mean_pairwise_frame_distance": mean_u_distance,
        "mean_pairwise_delta_distance": mean_delta_distance,
        "max_abs_delta": delta_abs,
        "max_abs_pairwise_delta_difference": diff_abs,
        "relative_change_from_prior": relative_change(u, prior_sequence),
        **loss_summary,
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

    data_terms, names = load_data_terms(cfg)
    prior = load_prior_image(
        cfg.prior_path,
        resize=(cfg.image_width, cfg.image_height),
        normalize=True,
    )
    u_init = np.repeat(prior[None, :, :], len(data_terms), axis=0)
    save_frames(u_init, cfg.output_root / "00_prior", names=names)

    summaries = []
    for prior_weight in cfg.prior_weights:
        label = f"tv_prior_{prior_weight:.0e}".replace("-", "m")
        print("\n" + "=" * 100)
        print(label)
        print("=" * 100)

        u, history, elapsed, converged = run_tv_prior(
            prior_weight=prior_weight,
            u_init=u_init,
            data_terms=data_terms,
            prior=prior,
            cfg=cfg,
        )
        summary = save_run_outputs(
            label=label,
            u=u,
            history=history,
            names=names,
            prior=prior,
            data_terms=data_terms,
            prior_weight=prior_weight,
            cfg=cfg,
        )
        summary["seconds"] = elapsed
        summary["converged"] = converged
        summaries.append(summary)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(cfg.output_root / "summary.csv", index=False)

    print("\n" + summary_df.to_string(index=False))
    print(f"\nOutputs saved to {cfg.output_root.resolve()}")
    print("\nLook first at each run's:")
    print("  montage_delta_signed.png")
    print("  montage_prior_plus_amplified_delta.png")
    print("  montage_pairwise_delta_difference_signed.png")


if __name__ == "__main__":
    main()
