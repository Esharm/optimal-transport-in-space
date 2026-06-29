import h5py
import numpy as np
from pathlib import Path
from PIL import Image
from skimage.transform import resize

# Change this after running `find . -name "*.h5"`
#AART_OUTPUT = Path("../aart/Results/Images_a_0.5_i_30_inoisy.h5")
AART_OUTPUT = Path("../aart/Results/Images_a_0.5_i_30_my_sgra_inoisylongr.h5")

OUT_DIR = Path("../data/aart_frames_long")
OUT_DIR.mkdir(parents=True, exist_ok=True)

N = 128
K = 30

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

    # Simple first guess:
    # If AART separates lensing bands, sum all brightness arrays with same shape.
    arrays = [f[k][:] for k in brightness_keys]
    shapes = [a.shape for a in arrays]
    print("Brightness shapes:", shapes)

    base_shape = arrays[0].shape
    same_shape_arrays = [a for a in arrays if a.shape == base_shape]

    movie = np.zeros_like(same_shape_arrays[0], dtype=float)
    for a in same_shape_arrays:
        movie += np.asarray(a, dtype=float)

# Ensure movie has shape (time, height, width)
if movie.ndim == 2:
    movie = movie[None, :, :]
elif movie.ndim == 3:
    pass
else:
    raise RuntimeError(f"Unexpected movie ndim: {movie.ndim}, shape={movie.shape}")

movie = np.nan_to_num(movie, nan=0.0, posinf=0.0, neginf=0.0)
movie = np.maximum(movie, 0.0)

if movie.max() > 0:
    movie = movie / movie.max()

movie = movie[:K]

frames = []
for frame in movie:
    frame = resize(frame, (N, N), anti_aliasing=True, preserve_range=True)
    frames.append(frame)

frames = np.stack(frames)
frames = np.maximum(frames, 0.0)

if frames.max() > 0:
    frames = frames / frames.max()

np.save(OUT_DIR / "u_true.npy", frames)

for k, frame in enumerate(frames):
    img = (255 * frame).astype(np.uint8)
    Image.fromarray(img).save(OUT_DIR / f"frame_{k:03d}.png")

print(f"\nSaved {len(frames)} frames to {OUT_DIR.resolve()}")
print(f"Saved NumPy array to {(OUT_DIR / 'u_true.npy').resolve()}")