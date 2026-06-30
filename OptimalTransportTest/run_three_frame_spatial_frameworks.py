"""Three-frame spatial regularizer experiment.

This runner intentionally excludes temporal OT/BB.  It tests spatial
frameworks that all use the visibility data term and the averaged-image prior:

    1. data + prior + TV
    2. data + prior + Haar-wavelet L1
    3. data + prior + TV + Haar-wavelet L1

It also sweeps a small set of prior weights and reports frame-separation
diagnostics.  If the reconstructions stay almost identical across frames while
the per-frame data terms differ, the prior/spatial penalties are probably too
strong relative to the data.  If the cross-data losses are also almost
identical, then the three visibility frames may genuinely not distinguish the
image much at this resolution/noise level.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from io_utils import load_prior_image, save_frames
from operators import div, grad, project_nonnegative_mass
from run_three_frame_ablation import Config as BaseConfig
from run_three_frame_ablation import load_data_terms, relative_change, tv_value


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Config(BaseConfig):
    output_root: Path = Path("three_frame_spatial_frameworks")

    # The previous run with prior_weight=5e-2 made the prior dominate.  This
    # sweep keeps the prior present while testing whether smaller values let
    # the frame-specific data terms separate the outputs.
    prior_weights: tuple[float, ...] = (2e-3, 1e-2, 5e-2)

    tv_weight: float = 3e-5
    wavelet_weight: float = 2e-4
    wavelet_levels: int = 3

    spatial_inner_iters: int = 25
    spatial_max_blocks: int = 35
    spatial_min_blocks: int = 3
    convergence_patience: int = 3
    objective_rel_tol: float = 1e-5
    iterate_rel_tol: float = 1e-4
    parallel_frames: bool = True

    primal_tau: float = 10.0
    dual_sigma: float = 0.20
    power_iters: int = 12


FRAMEWORKS = (
    ("tv", True, False),
    ("wavelet", False, True),
    ("tv_wavelet", True, True),
)


def haar2_forward(image, levels):
    """Orthonormal 2D Haar transform for power-of-two image sizes."""
    coeffs = np.asarray(image, dtype=np.float64).copy()
    height, width = coeffs.shape
    if height != width or height & (height - 1):
        raise ValueError("haar2_forward expects a square power-of-two image")
    if 2 ** levels > height:
        raise ValueError("Too many Haar levels for image shape")

    h = height
    w = width
    scale = np.sqrt(2.0)
    for _ in range(levels):
        block = coeffs[:h, :w]
        rows = np.empty_like(block)
        rows[:, : w // 2] = (block[:, 0::2] + block[:, 1::2]) / scale
        rows[:, w // 2 : w] = (block[:, 0::2] - block[:, 1::2]) / scale

        cols = np.empty_like(block)
        cols[: h // 2, :] = (rows[0::2, :] + rows[1::2, :]) / scale
        cols[h // 2 : h, :] = (rows[0::2, :] - rows[1::2, :]) / scale

        coeffs[:h, :w] = cols
        h //= 2
        w //= 2
    return coeffs


def haar2_inverse(coeffs, levels):
    """Inverse of haar2_forward."""
    image = np.asarray(coeffs, dtype=np.float64).copy()
    height, width = image.shape
    h = height // (2 ** levels)
    w = width // (2 ** levels)
    scale = np.sqrt(2.0)

    for _ in range(levels):
        h *= 2
        w *= 2
        block = image[:h, :w]

        rows = np.empty_like(block)
        rows[0::2, :] = (block[: h // 2, :] + block[h // 2 : h, :]) / scale
        rows[1::2, :] = (block[: h // 2, :] - block[h // 2 : h, :]) / scale

        pixels = np.empty_like(block)
        pixels[:, 0::2] = (rows[:, : w // 2] + rows[:, w // 2 : w]) / scale
        pixels[:, 1::2] = (rows[:, : w // 2] - rows[:, w // 2 : w]) / scale

        image[:h, :w] = pixels
    return image


def haar_detail_mask(shape, levels):
    """True for detail coefficients; False for the final lowpass block."""
    height, width = shape
    mask = np.ones(shape, dtype=bool)
    low_h = height // (2 ** levels)
    low_w = width // (2 ** levels)
    mask[:low_h, :low_w] = False
    return mask


def wavelet_l1_value(image, levels):
    coeffs = haar2_forward(image, levels)
    mask = haar_detail_mask(image.shape, levels)
    return float(np.sum(np.abs(coeffs[mask])))


class SpatialFrameworkRegularizer:
    """Primal-dual update for data + prior + TV + Haar-wavelet L1.

    The objective for one frame is

        D_k(u) + (mu/2)||u-p||_2^2
        + alpha TV(u) + gamma ||W_detail u||_1 + indicator_{u >= 0}.

    The Haar transform is orthonormal, so the wavelet dual projection is the
    exact l_infinity projection of detail coefficients.
    """

    def __init__(
        self,
        tv_weight=0.0,
        wavelet_weight=0.0,
        wavelet_levels=3,
        iters=25,
        tau=10.0,
        sigma=0.20,
        power_iters=12,
    ):
        self.tv_weight = float(tv_weight)
        self.wavelet_weight = float(wavelet_weight)
        self.wavelet_levels = int(wavelet_levels)
        self.iters = int(iters)
        self.tau_cap = float(tau)
        self.sigma = float(sigma)
        self.power_iters = int(power_iters)
        self._lipschitz_cache = {}
        self._tv_dual_cache = {}
        self._wavelet_dual_cache = {}

    def _data_lipschitz(self, data_term, shape):
        key = id(data_term)
        if key in self._lipschitz_cache:
            return self._lipschitz_cache[key]

        rng = np.random.default_rng(1729)
        vector = rng.normal(size=shape)
        vector /= np.linalg.norm(vector) + 1e-30
        eigenvalue = 0.0
        for _ in range(self.power_iters):
            applied = data_term.sampler.adjoint(data_term.sampler.forward(vector))
            norm = np.linalg.norm(applied)
            if norm <= 1e-30:
                eigenvalue = 0.0
                break
            vector = applied / norm
            eigenvalue = float(np.sum(vector * applied))

        estimate = max(1.1 * eigenvalue, 1e-12)
        self._lipschitz_cache[key] = estimate
        return estimate

    def solve(self, u_init, data_term, prior, prior_weight):
        u = np.asarray(u_init, dtype=np.float64).copy()
        u_bar = u.copy()
        key = id(data_term)

        p_tv = self._tv_dual_cache.get(key)
        if p_tv is None or p_tv.shape != (2, *u.shape):
            p_tv = np.zeros((2, *u.shape), dtype=np.float64)

        q_wav = self._wavelet_dual_cache.get(key)
        if q_wav is None or q_wav.shape != u.shape:
            q_wav = np.zeros_like(u)

        detail_mask = haar_detail_mask(u.shape, self.wavelet_levels)
        smooth_lipschitz = self._data_lipschitz(data_term, u.shape) + prior_weight

        operator_norm_sq = 0.0
        if self.tv_weight > 0:
            operator_norm_sq += 8.0
        if self.wavelet_weight > 0:
            operator_norm_sq += 1.0
        tau = min(
            self.tau_cap,
            0.99 / (0.5 * smooth_lipschitz + self.sigma * operator_norm_sq + 1e-12),
        )

        for _ in range(self.iters):
            if self.tv_weight > 0:
                p_tv += self.sigma * grad(u_bar)
                norm = np.sqrt(np.sum(p_tv * p_tv, axis=0))
                p_tv /= np.maximum(1.0, norm / self.tv_weight)[None, :, :]
            else:
                p_tv.fill(0.0)

            if self.wavelet_weight > 0:
                q_wav += self.sigma * haar2_forward(u_bar, self.wavelet_levels)
                q_wav[~detail_mask] = 0.0
                np.clip(q_wav, -self.wavelet_weight, self.wavelet_weight, out=q_wav)
            else:
                q_wav.fill(0.0)

            old = u.copy()
            gradient = data_term.gradient(u) + prior_weight * (u - prior)
            if self.tv_weight > 0:
                gradient -= div(p_tv)
            if self.wavelet_weight > 0:
                gradient += haar2_inverse(q_wav, self.wavelet_levels)

            u = project_nonnegative_mass(u - tau * gradient)
            u_bar = 2.0 * u - old

        self._tv_dual_cache[key] = p_tv.copy()
        self._wavelet_dual_cache[key] = q_wav.copy()
        return u


def objective_terms(u, data_terms, prior, prior_weight, tv_weight, wavelet_weight, levels):
    data = float(sum(term.loss(u[k]) for k, term in enumerate(data_terms)))
    prior_loss = float(0.5 * prior_weight * np.sum((u - prior[None, :, :]) ** 2))
    tv = float(tv_weight * sum(tv_value(frame) for frame in u))
    wavelet = float(wavelet_weight * sum(wavelet_l1_value(frame, levels) for frame in u))
    total = data + prior_loss + tv + wavelet
    return {
        "data": data,
        "prior": prior_loss,
        "tv": tv,
        "wavelet": wavelet,
        "spatial_total": total,
    }


def frame_separation_metrics(u, data_terms):
    rows = []
    distances = []
    for i in range(len(u)):
        for j in range(i + 1, len(u)):
            rel = np.linalg.norm(u[i] - u[j]) / (np.linalg.norm(u[i]) + np.linalg.norm(u[j]) + 1e-12)
            distances.append(float(rel))
            rows.append({"kind": "frame_pair_distance", "i": i, "j": j, "value": float(rel)})

    for image_index in range(len(u)):
        for data_index, term in enumerate(data_terms):
            rows.append({
                "kind": "cross_data_loss",
                "i": image_index,
                "j": data_index,
                "value": float(term.loss(u[image_index])),
            })

    own_losses = np.array([data_terms[k].loss(u[k]) for k in range(len(u))], dtype=np.float64)
    cross_losses = np.array([
        data_terms[j].loss(u[i])
        for i in range(len(u))
        for j in range(len(u))
        if i != j
    ], dtype=np.float64)
    rows.append({
        "kind": "mean_pairwise_frame_distance",
        "i": -1,
        "j": -1,
        "value": float(np.mean(distances) if distances else 0.0),
    })
    rows.append({
        "kind": "own_data_loss_mean",
        "i": -1,
        "j": -1,
        "value": float(np.mean(own_losses)),
    })
    rows.append({
        "kind": "cross_data_loss_mean",
        "i": -1,
        "j": -1,
        "value": float(np.mean(cross_losses) if cross_losses.size else 0.0),
    })
    rows.append({
        "kind": "cross_minus_own_data_loss_mean",
        "i": -1,
        "j": -1,
        "value": float(np.mean(cross_losses) - np.mean(own_losses) if cross_losses.size else 0.0),
    })
    return rows


def run_stage(label, u_init, data_terms, prior, prior_weight, tv_weight, wavelet_weight, cfg):
    regularizer = SpatialFrameworkRegularizer(
        tv_weight=tv_weight,
        wavelet_weight=wavelet_weight,
        wavelet_levels=cfg.wavelet_levels,
        iters=cfg.spatial_inner_iters,
        tau=cfg.primal_tau,
        sigma=cfg.dual_sigma,
        power_iters=cfg.power_iters,
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
                prior=prior,
                prior_weight=prior_weight,
            )

        if cfg.parallel_frames and len(data_terms) > 1:
            with ThreadPoolExecutor(max_workers=len(data_terms)) as executor:
                u = np.stack(list(executor.map(update_frame, range(len(data_terms)))))
        else:
            for k in range(len(data_terms)):
                u[k] = update_frame(k)

        terms = objective_terms(
            u=u,
            data_terms=data_terms,
            prior=prior,
            prior_weight=prior_weight,
            tv_weight=tv_weight,
            wavelet_weight=wavelet_weight,
            levels=cfg.wavelet_levels,
        )
        du = relative_change(u, old)
        if previous_objective is None:
            dobj = np.inf
        else:
            dobj = abs(terms["spatial_total"] - previous_objective) / (
                abs(previous_objective) + 1e-12
            )
        previous_objective = terms["spatial_total"]

        separation = frame_separation_metrics(u, data_terms)
        mean_pair_distance = next(
            item["value"] for item in separation
            if item["kind"] == "mean_pairwise_frame_distance"
        )
        cross_minus_own = next(
            item["value"] for item in separation
            if item["kind"] == "cross_minus_own_data_loss_mean"
        )

        row = {
            "block": block,
            "updates": block * cfg.spatial_inner_iters,
            "relative_iterate_change": du,
            "relative_objective_change": dobj,
            "mean_pairwise_frame_distance": mean_pair_distance,
            "cross_minus_own_data_loss_mean": cross_minus_own,
            **terms,
        }
        history.append(row)

        print(
            f"{label} block {block:03d} | objective={terms['spatial_total']:.6e} "
            f"| data={terms['data']:.3e} | prior={terms['prior']:.3e} "
            f"| d_obj={dobj:.3e} | d_u={du:.3e} "
            f"| frame_dist={mean_pair_distance:.3e} | cross-own={cross_minus_own:.3e}"
        )

        is_stable = dobj < cfg.objective_rel_tol and du < cfg.iterate_rel_tol
        stable = stable + 1 if block >= cfg.spatial_min_blocks and is_stable else 0
        if stable >= cfg.convergence_patience:
            print(f"{label}: converged after {block * cfg.spatial_inner_iters} updates")
            break

    elapsed = time.perf_counter() - started
    return u, history, elapsed, stable >= cfg.convergence_patience


def save_stage(label, u, history, diagnostics, names, output_root):
    stage_dir = output_root / label
    stage_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(stage_dir / "reconstruction.npz", frames=u)
    pd.DataFrame(history).to_csv(stage_dir / "history.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(stage_dir / "frame_diagnostics.csv", index=False)
    save_frames(u, stage_dir / "frames_each_norm", names=names, normalize_each=True)
    save_frames(u, stage_dir / "frames_global_norm", names=names, normalize_each=False)


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
    save_frames(u_init, cfg.output_root / "00_average_initialization", names=names)

    summaries = []
    all_diagnostics = []
    for prior_weight in cfg.prior_weights:
        for framework, use_tv, use_wavelet in FRAMEWORKS:
            label = f"{framework}_prior_{prior_weight:.0e}".replace("-", "m")
            tv_weight = cfg.tv_weight if use_tv else 0.0
            wavelet_weight = cfg.wavelet_weight if use_wavelet else 0.0

            print("\n" + "=" * 100)
            print(label)
            print("=" * 100)

            u, history, elapsed, converged = run_stage(
                label=label,
                u_init=u_init,
                data_terms=data_terms,
                prior=prior,
                prior_weight=prior_weight,
                tv_weight=tv_weight,
                wavelet_weight=wavelet_weight,
                cfg=cfg,
            )
            diagnostics = frame_separation_metrics(u, data_terms)
            for item in diagnostics:
                item["stage"] = label
                item["framework"] = framework
                item["prior_weight"] = prior_weight
            all_diagnostics.extend(diagnostics)

            save_stage(label, u, history, diagnostics, names, cfg.output_root)
            terms = objective_terms(
                u=u,
                data_terms=data_terms,
                prior=prior,
                prior_weight=prior_weight,
                tv_weight=tv_weight,
                wavelet_weight=wavelet_weight,
                levels=cfg.wavelet_levels,
            )
            mean_pair_distance = next(
                item["value"] for item in diagnostics
                if item["kind"] == "mean_pairwise_frame_distance"
            )
            cross_minus_own = next(
                item["value"] for item in diagnostics
                if item["kind"] == "cross_minus_own_data_loss_mean"
            )
            summaries.append({
                "stage": label,
                "framework": framework,
                "prior_weight": prior_weight,
                "tv_weight": tv_weight,
                "wavelet_weight": wavelet_weight,
                "converged": converged,
                "seconds": elapsed,
                "relative_change_from_average": relative_change(u, u_init),
                "mean_pairwise_frame_distance": mean_pair_distance,
                "cross_minus_own_data_loss_mean": cross_minus_own,
                **terms,
            })

    summary = pd.DataFrame(summaries)
    diagnostics = pd.DataFrame(all_diagnostics)
    summary.to_csv(cfg.output_root / "summary.csv", index=False)
    diagnostics.to_csv(cfg.output_root / "all_frame_diagnostics.csv", index=False)

    print("\n" + summary.to_string(index=False))
    print(f"\nOutputs saved to {cfg.output_root.resolve()}")
    print("\nReading the frame diagnostics:")
    print("  mean_pairwise_frame_distance near 0 means frames are visually/numerically identical.")
    print("  cross_minus_own_data_loss_mean > 0 means each frame fits its own data better than other frames' data.")
    print("  If frame_distance is near 0 while cross_minus_own is meaningfully positive, reduce prior/spatial weights.")


if __name__ == "__main__":
    main()
