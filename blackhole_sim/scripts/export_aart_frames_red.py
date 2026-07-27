#!/usr/bin/env python3
"""
Extract AART brightness frames and save EHT-style red/orange PNGs.

The normalized floating-point movie is still saved as `u_true.npy`.
Only the PNG visualization is colorized.
"""

import h5py
import numpy as np
from pathlib import Path
from PIL import Image
from skimage.transform import resize
from matplotlib import colormaps

# Change this after running: find . -name "*.h5"
# AART_OUTPUT = Path("../aart/Results/Images_a_0.5_i_30_inoisy.h5")
AART_OUTPUT = Path("../aart/Results/Images_a_0.5_i_30_inoisy2.h5")

OUT_DIR = Path("../data/aart_frames_test")
OUT_DIR.mkdir(parents=True, exist_ok=True)

N = 128
K = 30
CMAP_NAME = "afmhot"

with h5py.File(AART_OUTPUT, "r") as f:
    print("Datasets in output:")
    datasets = []

    def walk(name, obj):
        if hasattr(obj, "shape"):
            print(f"{name}: shape={obj.shape}, dtype={obj.dtype}")
            datasets.append(name)

    f.visititems(walk)

    brightness_keys = [k for k in datasets if "bght" in k.lower()]
    print("\nBrightness-like keys:", brightness_keys)

    if len(brightness_keys) == 0:
        raise RuntimeError("No brightness-like dataset found. Inspect printed keys.")

    # If AART separates lensing bands, sum brightness arrays with the same shape.
    arrays = [f[k][:] for k in brightness_keys]
    shapes = [a.shape for a in arrays]
    print("Brightness shapes:", shapes)

    base_shape = arrays[0].shape
    same_shape_arrays = [a for a in arrays if a.shape == base_shape]

    movie = np.zeros_like(same_shape_arrays[0], dtype=float)
    for array in same_shape_arrays:
        movie += np.asarray(array, dtype=float)

# Ensure movie has shape (time, height, width).
if movie.ndim == 2:
    movie = movie[None, :, :]
elif movie.ndim != 3:
    raise RuntimeError(
        f"Unexpected movie ndim: {movie.ndim}, shape={movie.shape}"
    )

movie = np.nan_to_num(movie, nan=0.0, posinf=0.0, neginf=0.0)
movie = np.maximum(movie, 0.0)

# Shared normalization over the full selected movie.
if movie.max() > 0:
    movie = movie / movie.max()

movie = movie[:K]

frames = []
for frame in movie:
    resized = resize(
        frame,
        (N, N),
        anti_aliasing=True,
        preserve_range=True,
    )
    frames.append(resized)

frames = np.stack(frames)
frames = np.maximum(frames, 0.0)

if frames.max() > 0:
    frames = frames / frames.max()

# Keep the numerical ground-truth array as normalized scalar intensity.
np.save(OUT_DIR / "u_true.npy", frames)

# Apply the EHT-style red/orange colormap only to the PNG outputs.
cmap = colormaps[CMAP_NAME]

for k, frame in enumerate(frames):
    rgba = cmap(np.clip(frame, 0.0, 1.0))
    rgb = np.rint(255.0 * rgba[..., :3]).astype(np.uint8)
    Image.fromarray(rgb, mode="RGB").save(
        OUT_DIR / f"frame_{k:03d}.png"
    )

print(f"\nColormap: {CMAP_NAME}")
print(f"Saved {len(frames)} frames to {OUT_DIR.resolve()}")
print(f"Saved NumPy array to {(OUT_DIR / 'u_true.npy').resolve()}")
