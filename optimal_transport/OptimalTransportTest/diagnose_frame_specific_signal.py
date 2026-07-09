"""Diagnose whether the first visibility frames contain separable image motion.

This is not another spatial-regularizer contest.  It asks a narrower question:

    Around the averaged-image prior, do the three data terms actually prefer
    different frame-specific corrections?

For each frame k, it solves the Tikhonov residual problem

    min_delta 0.5 ||S_k delta - (f_k - S_k prior)||_2^2
              + 0.5 * lambda_delta ||delta||_2^2

using conjugate gradients on

    (S_k^* S_k + lambda_delta I) delta = S_k^*(f_k - S_k prior).

Then it saves:

    prior_plus_delta frames,
    signed delta images,
    amplified prior_plus_delta images,
    CSV diagnostics and cross-data loss matrices.

Interpretation:

    If deltas are tiny/similar and each corrected frame barely fits its own
    data better than the other frames' data, spatial-only methods are unlikely
    to produce meaningful motion from these three frames.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from io_utils import load_prior_image, save_frames
from run_three_frame_ablation import Config as BaseConfig
from run_three_frame_ablation import load_data_terms, relative_change


@dataclass(frozen=True)
class Config(BaseConfig):
    output_root: Path = Path("frame_specific_signal_diagnostics")
    delta_weights: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 5e-2)
    cg_max_iter: int = 120
    cg_tol: float = 1e-7
    amplified_delta_scale: float = 8.0


def save_signed_frames(frames, output_folder, names=None):
    """Save signed frames with zero mapped to mid-gray."""
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    frames = np.asarray(frames, dtype=np.float64)
    scale = np.max(np.abs(frames)) + 1e-12

    for k, frame in enumerate(frames):
        image = 0.5 + 0.5 * frame / scale
        image = np.clip(image, 0.0, 1.0)
        image_uint8 = np.round(255.0 * image).astype(np.uint8)
        if names is None:
            filename = f"frame_{k:03d}.png"
        else:
            filename = f"{Path(str(names[k])).stem}.png"
        Image.fromarray(image_uint8).save(output_folder / filename)


def normal_apply(data_term, x, delta_weight):
    return data_term.sampler.adjoint(data_term.sampler.forward(x)) + delta_weight * x


def conjugate_gradient(apply_operator, rhs, max_iter=120, tol=1e-7):
    x = np.zeros_like(rhs)
    residual = rhs - apply_operator(x)
    direction = residual.copy()
    residual_sq = float(np.sum(residual * residual))
    initial_norm = np.sqrt(residual_sq) + 1e-30

    history = []
    for iteration in range(1, max_iter + 1):
        applied = apply_operator(direction)
        denom = float(np.sum(direction * applied)) + 1e-30
        step = residual_sq / denom
        x += step * direction
        residual -= step * applied
        next_residual_sq = float(np.sum(residual * residual))
        rel_residual = np.sqrt(next_residual_sq) / initial_norm
        history.append({
            "cg_iter": iteration,
            "relative_residual": float(rel_residual),
        })
        if rel_residual < tol:
            break
        beta = next_residual_sq / (residual_sq + 1e-30)
        direction = residual + beta * direction
        residual_sq = next_residual_sq

    return x, history


def solve_delta(data_term, prior, delta_weight, cfg):
    residual_visibility = data_term.f - data_term.sampler.forward(prior)
    rhs = data_term.sampler.adjoint(residual_visibility)

    started = time.perf_counter()
    delta, cg_history = conjugate_gradient(
        apply_operator=lambda x: normal_apply(data_term, x, delta_weight),
        rhs=rhs,
        max_iter=cfg.cg_max_iter,
        tol=cfg.cg_tol,
    )
    elapsed = time.perf_counter() - started

    unconstrained = prior + delta
    clipped = np.maximum(unconstrained, 0.0)
    return delta, clipped, cg_history, elapsed


def data_loss_matrix(images, data_terms):
    rows = []
    matrix = np.zeros((len(images), len(data_terms)), dtype=np.float64)
    for image_index, image in enumerate(images):
        for data_index, data_term in enumerate(data_terms):
            value = float(data_term.loss(image))
            matrix[image_index, data_index] = value
            rows.append({
                "image_index": image_index,
                "data_index": data_index,
                "loss": value,
            })
    return matrix, rows


def pairwise_relative_distances(frames, label):
    rows = []
    for i in range(len(frames)):
        for j in range(i + 1, len(frames)):
            value = np.linalg.norm(frames[i] - frames[j]) / (
                np.linalg.norm(frames[i]) + np.linalg.norm(frames[j]) + 1e-12
            )
            rows.append({
                "label": label,
                "i": i,
                "j": j,
                "relative_distance": float(value),
            })
    return rows


def cosine_rows(vectors, label):
    rows = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            denom = np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[j]) + 1e-12
            cosine = float(np.sum(vectors[i] * vectors[j]) / denom)
            rows.append({"label": label, "i": i, "j": j, "cosine": cosine})
    return rows


def summarize_loss_matrix(matrix):
    own = np.diag(matrix)
    cross = matrix[~np.eye(matrix.shape[0], dtype=bool)]
    return {
        "own_data_loss_mean": float(np.mean(own)),
        "cross_data_loss_mean": float(np.mean(cross)),
        "cross_minus_own_data_loss_mean": float(np.mean(cross) - np.mean(own)),
        "cross_over_own_data_loss_ratio": float(np.mean(cross) / (np.mean(own) + 1e-12)),
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
    prior_sequence = np.repeat(prior[None, :, :], len(data_terms), axis=0)
    save_frames(prior_sequence, cfg.output_root / "00_prior", names=names)

    prior_losses = [float(term.loss(prior)) for term in data_terms]
    descent_directions = [-term.gradient(prior) for term in data_terms]
    save_signed_frames(
        np.stack(descent_directions),
        cfg.output_root / "01_negative_gradient_at_prior_signed",
        names=names,
    )

    summaries = []
    distance_rows = []
    cosine_diagnostics = cosine_rows(descent_directions, "negative_gradient_at_prior")
    all_loss_rows = []
    all_cg_rows = []

    summaries.append({
        "stage": "prior_only",
        "delta_weight": np.nan,
        "seconds": 0.0,
        "mean_delta_norm_over_prior_norm": 0.0,
        "mean_pairwise_u_distance": 0.0,
        "mean_pairwise_delta_distance": 0.0,
        "prior_data_loss_mean": float(np.mean(prior_losses)),
        "own_data_loss_mean": float(np.mean(prior_losses)),
        "cross_data_loss_mean": float(np.mean(prior_losses)),
        "cross_minus_own_data_loss_mean": 0.0,
        "cross_over_own_data_loss_ratio": 1.0,
    })

    for delta_weight in cfg.delta_weights:
        print("\n" + "=" * 100)
        print(f"delta_weight={delta_weight:.3e}")
        print("=" * 100)

        deltas = []
        corrected = []
        elapsed_total = 0.0
        for k, data_term in enumerate(data_terms):
            delta, image, cg_history, elapsed = solve_delta(data_term, prior, delta_weight, cfg)
            deltas.append(delta)
            corrected.append(image)
            elapsed_total += elapsed
            for row in cg_history:
                row.update({
                    "delta_weight": delta_weight,
                    "frame_index": k,
                    "obs_name": names[k],
                })
                all_cg_rows.append(row)
            print(
                f"frame {k}: cg_iters={len(cg_history):03d} "
                f"| cg_rel={cg_history[-1]['relative_residual']:.3e} "
                f"| delta/prior={np.linalg.norm(delta)/(np.linalg.norm(prior)+1e-12):.3e}"
            )

        deltas = np.stack(deltas)
        corrected = np.stack(corrected)
        amplified = np.maximum(prior_sequence + cfg.amplified_delta_scale * deltas, 0.0)

        label = f"delta_weight_{delta_weight:.0e}".replace("-", "m")
        stage_dir = cfg.output_root / label
        stage_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(stage_dir / "residual_reconstruction.npz", deltas=deltas, frames=corrected)
        save_frames(corrected, stage_dir / "prior_plus_delta_each_norm", names=names, normalize_each=True)
        save_frames(corrected, stage_dir / "prior_plus_delta_global_norm", names=names, normalize_each=False)
        save_signed_frames(deltas, stage_dir / "delta_signed", names=names)
        save_frames(amplified, stage_dir / "prior_plus_amplified_delta", names=names, normalize_each=True)

        matrix, loss_rows = data_loss_matrix(corrected, data_terms)
        loss_summary = summarize_loss_matrix(matrix)
        for row in loss_rows:
            row.update({"delta_weight": delta_weight, "stage": label})
            all_loss_rows.append(row)

        u_distances = pairwise_relative_distances(corrected, "prior_plus_delta")
        delta_distances = pairwise_relative_distances(deltas, "delta")
        for row in u_distances + delta_distances:
            row.update({"delta_weight": delta_weight, "stage": label})
            distance_rows.append(row)

        mean_u_distance = float(np.mean([row["relative_distance"] for row in u_distances]))
        mean_delta_distance = float(np.mean([row["relative_distance"] for row in delta_distances]))
        mean_delta_norm = float(np.mean([
            np.linalg.norm(delta) / (np.linalg.norm(prior) + 1e-12)
            for delta in deltas
        ]))

        summary = {
            "stage": label,
            "delta_weight": delta_weight,
            "seconds": elapsed_total,
            "mean_delta_norm_over_prior_norm": mean_delta_norm,
            "mean_pairwise_u_distance": mean_u_distance,
            "mean_pairwise_delta_distance": mean_delta_distance,
            "prior_data_loss_mean": float(np.mean(prior_losses)),
            "relative_change_from_prior": relative_change(corrected, prior_sequence),
            **loss_summary,
        }
        summaries.append(summary)

        print(
            f"mean_u_dist={mean_u_distance:.3e} | mean_delta_dist={mean_delta_distance:.3e} "
            f"| own={loss_summary['own_data_loss_mean']:.3e} "
            f"| cross-own={loss_summary['cross_minus_own_data_loss_mean']:.3e}"
        )

    pd.DataFrame(summaries).to_csv(cfg.output_root / "summary.csv", index=False)
    pd.DataFrame(distance_rows).to_csv(cfg.output_root / "pairwise_distances.csv", index=False)
    pd.DataFrame(cosine_diagnostics).to_csv(cfg.output_root / "gradient_cosines.csv", index=False)
    pd.DataFrame(all_loss_rows).to_csv(cfg.output_root / "cross_data_losses.csv", index=False)
    pd.DataFrame(all_cg_rows).to_csv(cfg.output_root / "cg_history.csv", index=False)

    print("\n" + pd.DataFrame(summaries).to_string(index=False))
    print(f"\nOutputs saved to {cfg.output_root.resolve()}")
    print("\nDecision rule:")
    print("  If mean_pairwise_delta_distance is small and cross_minus_own is tiny,")
    print("  the three data terms do not strongly identify different images.")
    print("  If deltas differ but prior_plus_delta images do not, display amplified deltas")
    print("  and reduce the static-prior weight only after adding a better physical prior.")


if __name__ == "__main__":
    main()
