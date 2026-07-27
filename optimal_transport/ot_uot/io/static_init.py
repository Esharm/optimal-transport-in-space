"""Static reconstruction loading and background construction."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ot_uot.core.background import BackgroundMode, make_background
from ot_uot.io.observations import infer_frame_index


def _resize_image_to_shape(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resize a 2D image to ``shape`` using bilinear interpolation."""

    image = np.asarray(image, dtype=np.float64)
    if image.shape == tuple(shape):
        return image
    from PIL import Image

    height, width = int(shape[0]), int(shape[1])
    pil = Image.fromarray(image.astype(np.float32), mode="F")
    resized = pil.resize((width, height), resample=Image.BILINEAR)
    return np.asarray(resized, dtype=np.float64)


def _load_image(path: Path, *, normalize_images: bool = True, shape: tuple[int, int] | None = None) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path)
    elif suffix == ".npz":
        with np.load(path, allow_pickle=False) as npz:
            key = "image" if "image" in npz.files else npz.files[0]
            arr = npz[key]
    else:
        from PIL import Image

        arr = np.asarray(Image.open(path).convert("F"), dtype=np.float64)
        if normalize_images and arr.size and float(np.nanmax(arr)) > 1.5:
            arr = arr / 255.0
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"static reconstruction must be a 2D image: {path}")
    if shape is not None:
        arr = _resize_image_to_shape(arr, shape)
    return np.maximum(arr, 0.0)


def load_static_image(
    path: Path | str,
    *,
    normalize_images: bool = True,
    shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Load one nonnegative static/template image.

    This is used for template warm starts, e.g. a broad Gaussian/radial blob
    reused across all frames before the short per-frame L1+TV solves.
    """

    return _load_image(Path(path), normalize_images=normalize_images, shape=shape)


def load_static_sequence(
    directory: Path | str,
    *,
    frame_indices: list[int] | None = None,
    max_frames: int | None = None,
    normalize_images: bool = True,
) -> tuple[np.ndarray, list[Path]]:
    """Load a sequence of static reconstruction images."""

    directory = Path(directory)
    extensions = ("*.npy", "*.npz", "*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff")
    files: list[Path] = []
    for pattern in extensions:
        files.extend(directory.glob(pattern))
    files = sorted(files, key=infer_frame_index)
    if frame_indices is not None:
        wanted = set(int(i) for i in frame_indices)
        files = [path for path in files if infer_frame_index(path) in wanted]
    if max_frames is not None:
        files = files[: int(max_frames)]
    if not files:
        raise FileNotFoundError(f"no static reconstruction files found in {directory}")
    frames = np.stack([_load_image(path, normalize_images=normalize_images) for path in files], axis=0)
    return frames.astype(np.float64), files


def calibrate_static_sequence(
    frames: np.ndarray,
    data_terms,
    mode: str = "none",
) -> tuple[np.ndarray, dict[str, float | str]]:
    """Calibrate static frames to complex visibilities using legacy scaling.

    ``mode="per_frame"`` solves a nonnegative scalar least-squares scale for
    each frame. ``mode="global"`` solves one scalar for the whole sequence.
    ``mode="none"`` leaves the sequence unchanged.
    """

    frames = np.asarray(frames, dtype=np.float64)
    mode = str(mode)
    if mode == "none":
        return frames, {
            "static_recon_scale_mode": mode,
            "static_recon_scale_min": 1.0,
            "static_recon_scale_max": 1.0,
            "static_recon_scale_mean": 1.0,
            "static_recon_scale_std": 0.0,
        }
    if mode not in {"per_frame", "global"}:
        raise ValueError("static scale mode must be 'none', 'per_frame', or 'global'")
    if len(data_terms) != len(frames):
        raise ValueError("data_terms length must match static frame count")
    numerators = []
    denominators = []
    for frame, term in zip(frames, data_terms):
        predicted = term.operator.forward(frame)
        numerators.append(float(np.real(np.vdot(predicted, term.observed))))
        denominators.append(float(np.vdot(predicted, predicted).real))
    numerators = np.asarray(numerators, dtype=np.float64)
    denominators = np.asarray(denominators, dtype=np.float64)
    if mode == "per_frame":
        scales = np.maximum(numerators / (denominators + 1e-30), 0.0)
    else:
        scale = max(float(np.sum(numerators) / (np.sum(denominators) + 1e-30)), 0.0)
        scales = np.full(len(frames), scale, dtype=np.float64)
    scaled = frames * scales[:, None, None]
    return scaled, {
        "static_recon_scale_mode": mode,
        "static_recon_scale_min": float(scales.min()),
        "static_recon_scale_max": float(scales.max()),
        "static_recon_scale_mean": float(scales.mean()),
        "static_recon_scale_std": float(scales.std()),
        "static_recon_scale_numerator_sum": float(numerators.sum()),
        "static_recon_scale_denominator_sum": float(denominators.sum()),
    }


def load_static_with_background(
    directory: Path | str,
    *,
    mode: BackgroundMode | str = BackgroundMode.MEAN,
    frame_indices: list[int] | None = None,
    max_frames: int | None = None,
    normalize_images: bool = True,
    data_terms=None,
    scale_mode: str = "none",
) -> tuple[np.ndarray, np.ndarray, list[Path]]:
    """Load static frames and construct the background image/sequence."""

    frames, paths = load_static_sequence(
        directory,
        frame_indices=frame_indices,
        max_frames=max_frames,
        normalize_images=normalize_images,
    )
    if data_terms is not None:
        frames, _ = calibrate_static_sequence(frames, data_terms, scale_mode)
    background = make_background(frames, BackgroundMode(mode))
    return frames, background, paths
