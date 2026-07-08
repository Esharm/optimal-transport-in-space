#!/usr/bin/env python3
"""
Convert a folder of grayscale PNGs to an ehtim-style redscale PNG version.

Edit the SETTINGS block below, then run:
    python png_to_ehtim_redscale_settings.py

Dependencies:
    pip install pillow numpy
"""

from pathlib import Path
import sys

import numpy as np
from PIL import Image


# ========================= SETTINGS =========================

# Folder containing your grayscale PNG files.
INPUT_FOLDER = "poster_images/ground_truth_gray"

# Folder where the redscale PNGs will be written.
OUTPUT_FOLDER = "poster_images/ground_truth_red"

# Process PNGs inside subfolders too. The output keeps the same subfolder layout.
RECURSIVE = False

# Add this before .png in each output filename, e.g. image.png -> image_redscale.png.
OUTPUT_SUFFIX = "_red"

# If False, existing output files are left untouched.
OVERWRITE = True

# If True, files already ending with OUTPUT_SUFFIX are ignored.
SKIP_ALREADY_REDSCALE_FILES = True

# How to map grayscale values to 0..1 before applying redscale:
#   "image"    = stretch each PNG independently from its own min to max
#   "global"   = use one shared min/max across all PNGs, good for frame sequences
#   "absolute" = preserve the PNG's native range, e.g. 0..255 or 0..65535
NORMALIZATION = "global"

# Optional manual display limits. Leave both as None to use NORMALIZATION.
# Example:
# MANUAL_MIN = 0
# MANUAL_MAX = 255
MANUAL_MIN = None
MANUAL_MAX = None

# Optional intensity scaling before coloring:
#   "linear" = normal display
#   "gamma"  = brighten dim structure when GAMMA < 1
#   "log"    = logarithmic display
SCALE = "linear"
GAMMA = 0.5
DYNAMIC_RANGE = 1000.0

# Keep alpha/transparency if present in the input PNG.
PRESERVE_ALPHA = True

# ============================================================


GRAYSCALE_MODES = {"1", "L", "I", "I;16", "I;16B", "I;16L", "F"}


def is_relative_to(path: Path, parent: Path) -> bool:
    """Compatibility helper for Python versions before Path.is_relative_to."""
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def dtype_limits(arr: np.ndarray) -> tuple[float, float]:
    """Return the natural numeric range for common image array dtypes."""
    if arr.dtype == np.bool_:
        return 0.0, 1.0
    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        return float(info.min), float(info.max)
    return 0.0, 1.0


def finite_min_max(arr: np.ndarray) -> tuple[float, float] | None:
    """Return min/max while ignoring NaN/Inf. Return None if no finite pixels exist."""
    arr_float = arr.astype(np.float32, copy=False)
    finite = arr_float[np.isfinite(arr_float)]
    if finite.size == 0:
        return None
    return float(finite.min()), float(finite.max())


def alpha_to_uint8(alpha: np.ndarray) -> np.ndarray:
    """Convert an alpha channel of almost any numeric dtype to uint8."""
    if alpha.dtype == np.uint8:
        return alpha

    alpha_float = alpha.astype(np.float32, copy=False)

    if np.issubdtype(alpha.dtype, np.integer) or alpha.dtype == np.bool_:
        amin, amax = dtype_limits(alpha)
        if amax <= amin:
            return np.zeros(alpha.shape, dtype=np.uint8)
        alpha01 = (alpha_float - amin) / (amax - amin)
    else:
        max_seen = np.nanmax(alpha_float) if np.isfinite(alpha_float).any() else 1.0
        alpha01 = alpha_float if max_seen <= 1.0 else alpha_float / 255.0

    return np.round(np.clip(alpha01, 0.0, 1.0) * 255.0).astype(np.uint8)


def read_png_as_gray(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Read a PNG as grayscale values plus optional alpha.

    Native grayscale PNG values are preserved when possible, including 16-bit PNGs.
    RGB/RGBA/palette PNGs are converted to luminance.
    """
    with Image.open(path) as im:
        alpha = None

        if im.mode == "LA":
            gray = np.array(im.getchannel("L"), copy=True)
            if PRESERVE_ALPHA:
                alpha = np.array(im.getchannel("A"), copy=True)
            return gray, alpha

        if im.mode in GRAYSCALE_MODES:
            gray = np.array(im, copy=True)
            return gray, alpha

        # Handles RGB, RGBA, palette PNGs, and palette PNGs with transparency.
        if PRESERVE_ALPHA and ("A" in im.getbands() or "transparency" in im.info):
            rgba = im.convert("RGBA")
            gray = np.array(rgba.convert("L"), copy=True)
            alpha = np.array(rgba.getchannel("A"), copy=True)
        else:
            gray = np.array(im.convert("L"), copy=True)

        return gray, alpha


def normalize_to_01(
    arr: np.ndarray,
    *,
    mode: str,
    global_min_max: tuple[float, float] | None = None,
) -> np.ndarray:
    """Normalize an image array to floating-point values in [0, 1]."""
    arr_float = arr.astype(np.float32, copy=False)

    if MANUAL_MIN is not None or MANUAL_MAX is not None:
        if MANUAL_MIN is None or MANUAL_MAX is None:
            raise ValueError("Set both MANUAL_MIN and MANUAL_MAX, or leave both as None.")
        vmin, vmax = float(MANUAL_MIN), float(MANUAL_MAX)
    else:
        mode = mode.lower().strip()

        if mode == "image":
            min_max = finite_min_max(arr)
            if min_max is None:
                return np.zeros(arr.shape, dtype=np.float32)
            vmin, vmax = min_max

        elif mode == "global":
            if global_min_max is None:
                raise ValueError("global_min_max is required when NORMALIZATION = 'global'.")
            vmin, vmax = global_min_max

        elif mode == "absolute":
            vmin, vmax = dtype_limits(arr)

        else:
            raise ValueError('NORMALIZATION must be "image", "global", or "absolute".')

    if vmax <= vmin:
        # For a constant positive image, show it as bright instead of all black.
        fill = 1.0 if vmax > 0 else 0.0
        return np.full(arr.shape, fill, dtype=np.float32)

    out = (arr_float - vmin) / (vmax - vmin)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def apply_intensity_scale(arr01: np.ndarray) -> np.ndarray:
    """Apply optional linear/gamma/log intensity scaling."""
    x = np.clip(arr01.astype(np.float32, copy=False), 0.0, 1.0)
    scale = SCALE.lower().strip()

    if scale in {"linear", "lin", "none"}:
        return x

    if DYNAMIC_RANGE <= 0:
        raise ValueError("DYNAMIC_RANGE must be positive.")

    if scale == "gamma":
        if GAMMA <= 0:
            raise ValueError("GAMMA must be positive.")
        y = np.power(x + 1.0 / DYNAMIC_RANGE, GAMMA)

    elif scale == "log":
        y = np.log10(x + 1.0 / DYNAMIC_RANGE)

    else:
        raise ValueError('SCALE must be "linear", "gamma", or "log".')

    min_max = finite_min_max(y)
    if min_max is None:
        return np.zeros(x.shape, dtype=np.float32)

    ymin, ymax = min_max
    if ymax <= ymin:
        return np.zeros(x.shape, dtype=np.float32)

    y = (y - ymin) / (ymax - ymin)
    return np.clip(y, 0.0, 1.0).astype(np.float32)


def ehtim_redscale_rgb(arr01: np.ndarray) -> np.ndarray:
    """
    Apply the ehtim-style afmhot/redscale mapping.

    The mapping is black -> red -> orange/yellow -> white.
    """
    x = np.clip(arr01.astype(np.float32, copy=False), 0.0, 1.0)

    red = np.clip(2.0 * x, 0.0, 1.0)
    green = np.clip(2.0 * x - 0.5, 0.0, 1.0)
    blue = np.clip(2.0 * x - 1.0, 0.0, 1.0)

    rgb = np.stack((red, green, blue), axis=-1)
    return np.round(rgb * 255.0).astype(np.uint8)


def list_input_pngs(input_dir: Path, output_dir: Path) -> list[Path]:
    pattern = "**/*.png" if RECURSIVE else "*.png"
    files: list[Path] = []

    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    skip_output_folder = output_dir != input_dir

    for path in sorted(input_dir.glob(pattern)):
        if not path.is_file():
            continue

        resolved = path.resolve()

        if skip_output_folder and is_relative_to(resolved, output_dir):
            continue

        if SKIP_ALREADY_REDSCALE_FILES and path.stem.endswith(OUTPUT_SUFFIX):
            continue

        files.append(resolved)

    return files


def output_path_for(input_path: Path, input_dir: Path, output_dir: Path) -> Path:
    rel = input_path.relative_to(input_dir)
    out = output_dir / rel
    return out.with_name(out.stem + OUTPUT_SUFFIX + ".png")


def compute_global_min_max(files: list[Path]) -> tuple[float, float]:
    mins: list[float] = []
    maxs: list[float] = []

    for path in files:
        gray, _ = read_png_as_gray(path)
        min_max = finite_min_max(gray)
        if min_max is not None:
            mins.append(min_max[0])
            maxs.append(min_max[1])

    if not mins:
        return 0.0, 1.0

    return min(mins), max(maxs)


def save_redscale_png(
    gray: np.ndarray,
    alpha: np.ndarray | None,
    output_path: Path,
    global_min_max: tuple[float, float] | None,
) -> None:
    arr01 = normalize_to_01(gray, mode=NORMALIZATION, global_min_max=global_min_max)
    arr01 = apply_intensity_scale(arr01)
    rgb = ehtim_redscale_rgb(arr01)

    if PRESERVE_ALPHA and alpha is not None and alpha.shape[:2] == rgb.shape[:2]:
        rgba = np.dstack((rgb, alpha_to_uint8(alpha)))
        out_img = Image.fromarray(rgba, mode="RGBA")
    else:
        out_img = Image.fromarray(rgb, mode="RGB")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_img.save(output_path)


def main() -> int:
    input_dir = Path(INPUT_FOLDER).expanduser().resolve()
    output_dir = Path(OUTPUT_FOLDER).expanduser().resolve()

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Input folder does not exist: {input_dir}", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    files = list_input_pngs(input_dir, output_dir)
    if not files:
        print(f"No input PNG files found in: {input_dir}", file=sys.stderr)
        return 1

    global_min_max = None
    if MANUAL_MIN is None and MANUAL_MAX is None and NORMALIZATION.lower().strip() == "global":
        global_min_max = compute_global_min_max(files)
        print(f"Using global min/max: {global_min_max[0]} / {global_min_max[1]}")

    converted = 0
    skipped = 0
    failed = 0

    for input_path in files:
        output_path = output_path_for(input_path, input_dir, output_dir)

        if input_path == output_path:
            print(f"Skipping because output would overwrite input: {input_path}", file=sys.stderr)
            skipped += 1
            continue

        if output_path.exists() and not OVERWRITE:
            print(f"Skipping existing file: {output_path}")
            skipped += 1
            continue

        try:
            gray, alpha = read_png_as_gray(input_path)
            save_redscale_png(gray, alpha, output_path, global_min_max)
            print(f"Wrote: {output_path}")
            converted += 1
        except Exception as exc:
            print(f"FAILED: {input_path}\n  {type(exc).__name__}: {exc}", file=sys.stderr)
            failed += 1

    print(f"\nDone. Converted {converted}; skipped {skipped}; failed {failed}.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())