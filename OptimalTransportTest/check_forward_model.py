import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from io_utils import load_npz_visibility_data_terms
from operators import normalize01


PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# BASIC IMAGE LOADING
# ============================================================

def load_ground_truth_frames(folder, max_frames=30, resize=(64, 64)):
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
        img = Image.open(folder / name).convert("L")
        img = img.resize(resize)

        arr = np.asarray(img, dtype=np.float64) / 255.0
        arr = normalize01(arr)

        frames.append(arr)

    return np.stack(frames), files


# ============================================================
# RAW NPZ LOADING FOR SIGMA / NOISE CHECKS
# ============================================================

def load_raw_npz_frame(npz_path, max_vis_per_frame=None, seed=0):
    """
    Loads raw u, v, vis, sigma from one npz file.

    This is separate from load_npz_visibility_data_terms because we need
    sigma explicitly to estimate the expected noise scale.
    """
    rng = np.random.default_rng(seed)

    loaded = np.load(npz_path, allow_pickle=False)

    if "data" not in loaded:
        raise KeyError(f"{npz_path.name} does not contain key 'data'")

    data = loaded["data"]

    required = ["u", "v", "vis", "sigma"]
    for field in required:
        if field not in data.dtype.names:
            raise KeyError(
                f"{npz_path.name} missing field '{field}'. "
                f"Available fields: {data.dtype.names}"
            )

    u = np.asarray(data["u"], dtype=np.float64)
    v = np.asarray(data["v"], dtype=np.float64)
    vis = np.asarray(data["vis"], dtype=np.complex128)
    sigma = np.asarray(data["sigma"], dtype=np.float64)

    good = (
        np.isfinite(u)
        & np.isfinite(v)
        & np.isfinite(vis.real)
        & np.isfinite(vis.imag)
        & np.isfinite(sigma)
        & (sigma > 0)
    )

    u = u[good]
    v = v[good]
    vis = vis[good]
    sigma = sigma[good]

    if max_vis_per_frame is not None and len(u) > max_vis_per_frame:
        idx = rng.choice(len(u), size=max_vis_per_frame, replace=False)
        u = u[idx]
        v = v[idx]
        vis = vis[idx]
        sigma = sigma[idx]

    return {
        "u": u,
        "v": v,
        "vis": vis,
        "sigma": sigma,
    }


def load_raw_npz_sequence(folder, max_frames=3, max_vis_per_frame=None, seed=0):
    folder = Path(folder)

    files = sorted([
        p for p in folder.iterdir()
        if p.suffix.lower() == ".npz"
    ])[:max_frames]

    raw = []

    for k, path in enumerate(files):
        raw.append(load_raw_npz_frame(
            path,
            max_vis_per_frame=max_vis_per_frame,
            seed=seed + k,
        ))

    return raw, [p.name for p in files]


# ============================================================
# LINEAR ALGEBRA TESTS
# ============================================================

def adjoint_test(data_term, image_shape, seed=0, n_trials=10):
    """
    Checks the real-image adjoint relation.

    Since the image x is real and sampler.adjoint(y) returns a real gradient-like image,
    the correct identity is:

        Re <Sx, y> = <x, S*y>_R

    where <Sx,y> is the complex inner product and the right side is a real dot product.
    """
    rng = np.random.default_rng(seed)

    sampler = data_term.sampler
    errors = []

    for _ in range(n_trials):
        x = rng.normal(size=image_shape)
        y = rng.normal(size=data_term.f.shape) + 1j * rng.normal(size=data_term.f.shape)

        sx = sampler.forward(x)
        sty = sampler.adjoint(y)

        lhs = np.real(np.vdot(sx, y))
        rhs = np.sum(x * sty)

        err = abs(lhs - rhs) / (abs(lhs) + abs(rhs) + 1e-12)
        errors.append(err)

    return {
        "adjoint_relerr_mean": float(np.mean(errors)),
        "adjoint_relerr_max": float(np.max(errors)),
    }


def gradient_test(data_term, image_shape, seed=1, n_trials=8, eps=1e-6):
    """
    Checks gradient of

        phi(x) = 0.5 ||Sx - f||^2

    using centered finite differences.
    """
    rng = np.random.default_rng(seed)

    errors = []

    for _ in range(n_trials):
        x = rng.normal(size=image_shape)
        h = rng.normal(size=image_shape)
        h = h / (np.linalg.norm(h) + 1e-12)

        grad = data_term.gradient(x)

        phi_plus = data_term.loss(x + eps * h)
        phi_minus = data_term.loss(x - eps * h)

        finite_diff = (phi_plus - phi_minus) / (2 * eps)
        analytic = np.sum(grad * h)

        err = abs(finite_diff - analytic) / (
            abs(finite_diff) + abs(analytic) + 1e-12
        )

        errors.append(err)

    return {
        "gradient_relerr_mean": float(np.mean(errors)),
        "gradient_relerr_max": float(np.max(errors)),
    }


# ============================================================
# VISIBILITY MATCH TESTS
# ============================================================

def best_scaled_residual(pred, obs):
    """
    Finds real scalar a minimizing ||a pred - obs||_2.
    """
    denom = np.vdot(pred, pred).real + 1e-12
    scale = np.real(np.vdot(pred, obs)) / denom

    pred_scaled = scale * pred

    rel_res = np.linalg.norm(pred_scaled - obs) / (np.linalg.norm(obs) + 1e-12)

    corr = abs(np.vdot(pred, obs)) / (
        (np.linalg.norm(pred) * np.linalg.norm(obs)) + 1e-12
    )

    return float(scale), float(rel_res), float(corr)


def raw_residual(pred, obs):
    return float(np.linalg.norm(pred - obs) / (np.linalg.norm(obs) + 1e-12))


def expected_noise_relative_raw(raw_frame):
    """
    Estimates expected ||noise|| / ||vis|| from sigma.

    For complex visibilities, if real and imaginary parts each have std sigma,
    expected complex noise magnitude satisfies E|noise|^2 ≈ 2 sigma^2.

    Some simulators may define sigma for complex amplitude differently.
    So we report both one-component and two-component conventions.
    """
    vis = raw_frame["vis"]
    sigma = raw_frame["sigma"]

    vis_norm = np.linalg.norm(vis) + 1e-12

    noise_one_component = np.sqrt(np.sum(sigma ** 2)) / vis_norm
    noise_two_component = np.sqrt(np.sum(2.0 * sigma ** 2)) / vis_norm

    median_sigma_over_amp = np.median(sigma / (np.abs(vis) + 1e-12))
    mean_sigma_over_amp = np.mean(sigma / (np.abs(vis) + 1e-12))

    return {
        "expected_noise_rel_1comp": float(noise_one_component),
        "expected_noise_rel_2comp": float(noise_two_component),
        "median_sigma_over_amp": float(median_sigma_over_amp),
        "mean_sigma_over_amp": float(mean_sigma_over_amp),
    }


def visibility_match_report(data_term, gt_image):
    pred = data_term.sampler.forward(gt_image)
    obs = data_term.f

    unscaled = raw_residual(pred, obs)
    scale, scaled, corr = best_scaled_residual(pred, obs)

    conj_unscaled = raw_residual(np.conj(pred), obs)
    conj_scale, conj_scaled, conj_corr = best_scaled_residual(np.conj(pred), obs)

    return {
        "unscaled_rel_residual": unscaled,
        "best_flux_scale": scale,
        "scaled_rel_residual": scaled,
        "complex_corr": corr,
        "conj_unscaled_rel_residual": conj_unscaled,
        "conj_best_flux_scale": conj_scale,
        "conj_scaled_rel_residual": conj_scaled,
        "conj_complex_corr": conj_corr,
        "conjugate_better_scaled": bool(conj_scaled < scaled),
    }


# ============================================================
# ORIENTATION / FRAME PAIRING TESTS
# ============================================================

def image_variants(img):
    return {
        "original": img,
        "flip_lr": np.fliplr(img),
        "flip_ud": np.flipud(img),
        "transpose": img.T,
        "rot90": np.rot90(img, 1),
        "rot180": np.rot90(img, 2),
        "rot270": np.rot90(img, 3),
    }


def orientation_test_scaled(data_term, gt_image):
    """
    Tests orientation mistakes using best-fit flux scaling for each orientation.
    This is much more meaningful than unscaled orientation residuals.
    """
    obs = data_term.f
    rows = []

    for name, img in image_variants(gt_image).items():
        if img.shape != gt_image.shape:
            continue

        pred = data_term.sampler.forward(img)

        scale, scaled_res, corr = best_scaled_residual(pred, obs)
        conj_scale, conj_scaled_res, conj_corr = best_scaled_residual(np.conj(pred), obs)

        rows.append({
            "orientation": name,
            "best_flux_scale": scale,
            "scaled_rel_residual": scaled_res,
            "complex_corr": corr,
            "conj_best_flux_scale": conj_scale,
            "conj_scaled_rel_residual": conj_scaled_res,
            "conj_complex_corr": conj_corr,
            "conjugate_better_scaled": bool(conj_scaled_res < scaled_res),
        })

    rows = sorted(rows, key=lambda r: r["scaled_rel_residual"])
    return rows


def frame_pairing_test(data_term, gt_frames, gt_names):
    """
    Compares one observed visibility frame against every GT image and every
    simple orientation, with best-fit flux scaling.

    If the best match is not the same frame index and original-ish orientation,
    then evaluation pairing/orientation is suspect.
    """
    rows = []
    obs = data_term.f

    for j, gt in enumerate(gt_frames):
        for orient_name, img in image_variants(gt).items():
            if img.shape != gt.shape:
                continue

            pred = data_term.sampler.forward(img)
            scale, scaled_res, corr = best_scaled_residual(pred, obs)

            conj_scale, conj_scaled_res, conj_corr = best_scaled_residual(np.conj(pred), obs)

            rows.append({
                "gt_index": j,
                "gt_name": gt_names[j],
                "orientation": orient_name,
                "best_flux_scale": scale,
                "scaled_rel_residual": scaled_res,
                "complex_corr": corr,
                "conj_best_flux_scale": conj_scale,
                "conj_scaled_rel_residual": conj_scaled_res,
                "conj_complex_corr": conj_corr,
                "conjugate_better_scaled": bool(conj_scaled_res < scaled_res),
            })

    rows = sorted(rows, key=lambda r: r["scaled_rel_residual"])
    return rows


# ============================================================
# VERDICT LOGIC
# ============================================================

def classify_forward_model(row, best_pair_row=None):
    """
    Gives a practical verdict.

    This is not a theorem. It is a diagnostic classification.
    """
    adj_ok = row["adjoint_relerr_max"] < 1e-10
    grad_ok = row["gradient_relerr_max"] < 1e-5

    scaled_res = row["scaled_rel_residual"]
    noise_1 = row["expected_noise_rel_1comp"]
    noise_2 = row["expected_noise_rel_2comp"]

    # Compare residual to noise. Give slack because PNG normalization/resizing
    # and exact simulator convention may not perfectly match.
    noise_floor = max(noise_1, noise_2)
    consistent_with_noise = scaled_res <= max(0.15, 2.5 * noise_floor)

    same_frame_best = True
    same_orientation_best = True

    if best_pair_row is not None:
        same_frame_best = int(best_pair_row["gt_index"]) == int(row["frame_index"])
        same_orientation_best = best_pair_row["orientation"] == "original"

    if not adj_ok or not grad_ok:
        return "FAIL: coded forward/adjoint/gradient inconsistent"

    if consistent_with_noise and same_frame_best:
        if same_orientation_best:
            return "PASS: internally correct and GT match is consistent with noise"
        else:
            return "MOSTLY PASS: internally correct, noise-consistent, but orientation convention is suspicious"

    if not same_frame_best:
        return "WARNING: internally correct, but GT frame pairing appears mismatched"

    if scaled_res > max(0.15, 2.5 * noise_floor):
        return "WARNING: internally correct, but GT residual exceeds expected noise"

    return "AMBIGUOUS: internally correct, but physical GT/data consistency unclear"


# ============================================================
# MAIN
# ============================================================

def main():
    obs_folder = PROJECT_ROOT / "blackhole_sim_testing" / "observations_npz"
    gt_folder = PROJECT_ROOT / "blackhole_sim" / "data" / "aart_frames"

    out_dir = Path("./forward_model_check_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = out_dir / "forward_model_summary.csv"
    orientation_csv = out_dir / "orientation_scaled_checks.csv"
    pairing_csv = out_dir / "frame_pairing_checks.csv"

    # Use small mode for fast diagnostics.
    K_obs = 3
    K_gt = 15
    image_shape = (64, 64)
    fov_rad = 160e-6 / 206265.0
    max_vis_per_frame = 1000
    use_hermitian = False

    print("=" * 100)
    print("LOADING DATA TERMS")
    print("=" * 100)

    data_terms, obs_names = load_npz_visibility_data_terms(
        folder=obs_folder,
        max_frames=K_obs,
        image_shape=image_shape,
        fov_rad=fov_rad,
        max_vis_per_frame=max_vis_per_frame,
        use_hermitian=use_hermitian,
    )

    raw_frames, raw_names = load_raw_npz_sequence(
        folder=obs_folder,
        max_frames=K_obs,
        max_vis_per_frame=max_vis_per_frame,
        seed=0,
    )

    print("=" * 100)
    print("LOADING GROUND TRUTH FRAMES")
    print("=" * 100)

    gt_frames, gt_names = load_ground_truth_frames(
        folder=gt_folder,
        max_frames=K_gt,
        resize=image_shape[::-1],
    )

    summary_rows = []
    orientation_rows = []
    pairing_rows = []

    for k, data_term in enumerate(data_terms):
        print("\n" + "#" * 100)
        print(f"CHECKING OBS FRAME {k}: {obs_names[k]}")
        print("#" * 100)

        gt_same = gt_frames[k]

        adj = adjoint_test(
            data_term=data_term,
            image_shape=image_shape,
            seed=100 + k,
            n_trials=10,
        )

        grad = gradient_test(
            data_term=data_term,
            image_shape=image_shape,
            seed=200 + k,
            n_trials=8,
            eps=1e-6,
        )

        vis_match = visibility_match_report(
            data_term=data_term,
            gt_image=gt_same,
        )

        noise = expected_noise_relative_raw(raw_frames[k])

        orientations = orientation_test_scaled(
            data_term=data_term,
            gt_image=gt_same,
        )

        pairings = frame_pairing_test(
            data_term=data_term,
            gt_frames=gt_frames,
            gt_names=gt_names,
        )

        best_orientation = orientations[0]
        best_pair = pairings[0]

        row = {
            "frame_index": k,
            "obs_name": obs_names[k],
            "gt_same_index": k,
            "gt_same_name": gt_names[k],
            **adj,
            **grad,
            **vis_match,
            **noise,
            "best_same_frame_orientation": best_orientation["orientation"],
            "best_same_frame_scaled_residual": best_orientation["scaled_rel_residual"],
            "best_same_frame_corr": best_orientation["complex_corr"],
            "best_pair_gt_index": best_pair["gt_index"],
            "best_pair_gt_name": best_pair["gt_name"],
            "best_pair_orientation": best_pair["orientation"],
            "best_pair_scaled_residual": best_pair["scaled_rel_residual"],
            "best_pair_corr": best_pair["complex_corr"],
        }

        verdict = classify_forward_model(row, best_pair_row=best_pair)
        row["verdict"] = verdict

        summary_rows.append(row)

        for item in orientations:
            item = dict(item)
            item["frame_index"] = k
            item["obs_name"] = obs_names[k]
            orientation_rows.append(item)

        for rank, item in enumerate(pairings):
            item = dict(item)
            item["obs_frame_index"] = k
            item["obs_name"] = obs_names[k]
            item["rank"] = rank + 1
            pairing_rows.append(item)

        print(f"Adjoint max rel err:      {adj['adjoint_relerr_max']:.3e}")
        print(f"Gradient max rel err:     {grad['gradient_relerr_max']:.3e}")
        print(f"Same-frame scaled resid:  {vis_match['scaled_rel_residual']:.3e}")
        print(f"Same-frame corr:          {vis_match['complex_corr']:.3e}")
        print(f"Expected noise rel 1comp: {noise['expected_noise_rel_1comp']:.3e}")
        print(f"Expected noise rel 2comp: {noise['expected_noise_rel_2comp']:.3e}")
        print(f"Median sigma/amp:         {noise['median_sigma_over_amp']:.3e}")
        print()
        print("Best same-frame orientations:")
        for item in orientations[:5]:
            print(
                f"  {item['orientation']:>10s} | "
                f"scaled_res={item['scaled_rel_residual']:.3e} | "
                f"corr={item['complex_corr']:.3e} | "
                f"scale={item['best_flux_scale']:.3e}"
            )
        print()
        print("Best GT frame-pair candidates:")
        for item in pairings[:8]:
            print(
                f"  gt={item['gt_index']:02d} {item['orientation']:>10s} | "
                f"scaled_res={item['scaled_rel_residual']:.3e} | "
                f"corr={item['complex_corr']:.3e} | "
                f"scale={item['best_flux_scale']:.3e}"
            )
        print()
        print(f"VERDICT: {verdict}")

    summary_df = pd.DataFrame(summary_rows)
    orientation_df = pd.DataFrame(orientation_rows)
    pairing_df = pd.DataFrame(pairing_rows)

    summary_df.to_csv(summary_csv, index=False)
    orientation_df.to_csv(orientation_csv, index=False)
    pairing_df.to_csv(pairing_csv, index=False)

    print("\n" + "=" * 100)
    print("FINAL SUMMARY")
    print("=" * 100)

    display_cols = [
        "frame_index",
        "adjoint_relerr_max",
        "gradient_relerr_max",
        "scaled_rel_residual",
        "complex_corr",
        "expected_noise_rel_1comp",
        "expected_noise_rel_2comp",
        "best_same_frame_orientation",
        "best_same_frame_scaled_residual",
        "best_pair_gt_index",
        "best_pair_orientation",
        "best_pair_scaled_residual",
        "verdict",
    ]

    print(summary_df[display_cols].to_string(index=False))

    print("\nSaved:")
    print(f"  {summary_csv.resolve()}")
    print(f"  {orientation_csv.resolve()}")
    print(f"  {pairing_csv.resolve()}")

    print("\nInterpretation:")
    print("  1. If adjoint/gradient errors are tiny, the coded operator is internally correct.")
    print("  2. If scaled residual is comparable to expected noise, GT/data consistency is good.")
    print("  3. If best_pair_gt_index differs from frame_index, evaluation frame pairing may be wrong.")
    print("  4. If best orientation is not original, image coordinate convention may be wrong.")
    print("  5. If residual is much larger than noise even after scaling/pairing/orientation,")
    print("     the PNG GT images probably do not exactly match the images used to generate NPZ data.")


if __name__ == "__main__":
    main()