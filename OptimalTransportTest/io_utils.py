import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from data_terms import ComplexVisibilityDataTerm
from operators import VisibilitySampler, normalize01


def load_prior_image(path, resize=(128, 128), normalize=True):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Prior image not found: {path}")

    img = Image.open(path).convert("L")
    img = img.resize(resize)

    prior = np.asarray(img, dtype=np.float64) / 255.0

    if normalize:
        prior = normalize01(prior)

    return prior


def hermitian_augment(u, v, vis, sigma):
    """
    For real-valued Stokes I images:

        V(-u, -v) = conj(V(u, v))

    This is not a regularizer. It just enforces the Fourier symmetry expected
    from a real image.
    """

    u_aug = np.concatenate([u, -u])
    v_aug = np.concatenate([v, -v])
    vis_aug = np.concatenate([vis, np.conj(vis)])
    sigma_aug = np.concatenate([sigma, sigma])

    return u_aug, v_aug, vis_aug, sigma_aug


def load_npz_visibility_data_terms(
    folder,
    max_frames=15,
    image_shape=(128, 128),
    fov_rad=160e-6 / 206265.0,
    max_vis_per_frame=None,
    use_hermitian=True,
    seed=0,
):
    """
    Loads complex visibility data from .npz files.

    Expected format:
        each .npz has key "data"

    Required fields:
        u, v, vis, sigma

    Returns:
        data_terms : list[ComplexVisibilityDataTerm]
        names      : list[str]
    """

    folder = Path(folder)

    if not folder.exists():
        raise FileNotFoundError(f"Visibility folder not found: {folder}")

    files = sorted([
        p for p in folder.iterdir()
        if p.suffix.lower() == ".npz"
    ])[:max_frames]

    if len(files) == 0:
        raise ValueError(f"No .npz files found in {folder}")

    rng = np.random.default_rng(seed)

    raw_frames = []
    all_abs_vis = []

    for npz_path in files:
        loaded = np.load(npz_path, allow_pickle=False)

        if "data" not in loaded:
            raise KeyError(f"{npz_path.name} does not contain key 'data'")

        data = loaded["data"]

        required = ["u", "v", "vis", "sigma"]
        for field in required:
            if field not in data.dtype.names:
                raise KeyError(
                    f"{npz_path.name} missing field '{field}'. "
                    f"Available: {data.dtype.names}"
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

        if use_hermitian:
            u, v, vis, sigma = hermitian_augment(u, v, vis, sigma)

        raw_frames.append((npz_path.name, u, v, vis, sigma))
        all_abs_vis.append(np.abs(vis))

    all_abs_vis = np.concatenate(all_abs_vis)
    data_scale = np.percentile(all_abs_vis, 95) + 1e-12

    print(f"Global visibility scale: {data_scale:.3e}")

    data_terms = []
    names = []

    for name, u, v, vis, sigma in raw_frames:
        weight = 1.0 / np.maximum(sigma, 1e-12) ** 2
        weight = weight / (np.median(weight) + 1e-12)

        sampler = VisibilitySampler(
            u=u,
            v=v,
            weight=weight,
            shape=image_shape,
            fov_rad=fov_rad,
            data_scale=data_scale,
        )

        f = np.sqrt(weight) * vis / sampler.total_scale

        data_term = ComplexVisibilityDataTerm(
            sampler=sampler,
            f=f,
        )

        data_terms.append(data_term)
        names.append(name)

        print(
            f"Loaded {name}: "
            f"{len(vis)} visibilities | "
            f"|vis| median={np.median(np.abs(vis)):.3e}"
        )

    return data_terms, names


def save_frames(frames, output_folder, names=None, normalize_each=True):
    """
    Saves [K, H, W] sequence to PNG.

    If normalize_each=True:
        each frame is independently normalized for visualization.

    If normalize_each=False:
        global sequence normalization is used.
    """

    os.makedirs(output_folder, exist_ok=True)

    frames = np.asarray(frames, dtype=np.float64)
    K = frames.shape[0]

    if not normalize_each:
        global_min = np.min(frames)
        global_max = np.max(frames)

    for k in range(K):
        img = np.asarray(frames[k], dtype=np.float64)
        img = np.maximum(img, 0.0)

        if normalize_each:
            img = normalize01(img)
        else:
            img = (img - global_min) / (global_max - global_min + 1e-12)

        img_uint8 = (255.0 * img).astype(np.uint8)

        if names is not None:
            base = os.path.splitext(os.path.basename(str(names[k])))[0]
            filename = f"{base}.png"
        else:
            filename = f"frame_{k:03d}.png"

        Image.fromarray(img_uint8).save(os.path.join(output_folder, filename))


def images_to_video(image_folder, output_file, fps=5):
    images = sorted([
        f for f in os.listdir(image_folder)
        if f.lower().endswith(".png")
    ])

    if len(images) == 0:
        raise ValueError(f"No PNG images found in {image_folder}")

    first = cv2.imread(os.path.join(image_folder, images[0]))

    if first is None:
        raise ValueError(f"Could not read first image in {image_folder}")

    h, w, _ = first.shape

    video = cv2.VideoWriter(
        output_file,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

    for img_name in images:
        frame = cv2.imread(os.path.join(image_folder, img_name))
        if frame is not None:
            video.write(frame)

    video.release()