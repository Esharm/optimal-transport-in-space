"""Radially average a 128x128 image while conserving total brightness.

The output image is circularized: every pixel in the same radial bin gets the
same brightness, equal to the mean brightness of the original pixels in that
bin. Since each bin receives its original bin average, total brightness is
conserved in the floating-point output array.

Example:
    python scripts/radial_brightness_average.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = Path("blackhole_sim")
DEFAULT_IMAGE = "time_avg_static_recon_128pix.png"


def load_grayscale_image(path: Path) -> tuple[np.ndarray, str]:
    """Load an image as floating-point grayscale brightness."""
    image = Image.open(path)
    mode = image.mode

    if mode in {"I;16", "I"}:
        array = np.asarray(image, dtype=np.float64)
    else:
        array = np.asarray(image.convert("L"), dtype=np.float64)

    if array.shape != (128, 128):
        raise ValueError(f"Expected a 128x128 image, got {array.shape} from {path}")

    return array, mode


def radial_bins(shape: tuple[int, int], mode: str) -> np.ndarray:
    """Return an integer bin index for each pixel's radial distance."""
    height, width = shape
    y, x = np.indices(shape)

    center_y = (height - 1) / 2.0
    center_x = (width - 1) / 2.0
    radius = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

    if mode == "round":
        return np.rint(radius).astype(np.int64)
    if mode == "floor":
        return np.floor(radius).astype(np.int64)
    if mode == "ceil":
        return np.ceil(radius).astype(np.int64)

    raise ValueError(f"Unknown bin mode: {mode}")


def circularize_by_radius(image: np.ndarray, bin_mode: str = "round") -> np.ndarray:
    """Make each radial bin constant while preserving each bin's total flux."""
    bins = radial_bins(image.shape, bin_mode)
    flat_bins = bins.ravel()
    flat_image = image.ravel()

    radial_sums = np.bincount(flat_bins, weights=flat_image)
    radial_counts = np.bincount(flat_bins)
    radial_means = radial_sums / np.maximum(radial_counts, 1)

    return radial_means[bins]


def save_preview_png(image: np.ndarray, path: Path, input_mode: str) -> None:
    """Save a viewable PNG. Exact floating-point values are saved separately."""
    if input_mode in {"I;16", "I"} or image.max(initial=0.0) > 255:
        clipped = np.clip(np.rint(image), 0, 65535).astype(np.uint16)
        Image.fromarray(clipped, mode="I;16").save(path)
    else:
        clipped = np.clip(np.rint(image), 0, 255).astype(np.uint8)
        Image.fromarray(clipped, mode="L").save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Radially average a 128x128 image while conserving brightness."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing the input image, relative to PROJECT_ROOT unless absolute.",
    )
    parser.add_argument(
        "--image",
        default=DEFAULT_IMAGE,
        help="Input image filename.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("radial_outputs"),
        help="Directory for outputs, relative to PROJECT_ROOT unless absolute.",
    )
    parser.add_argument(
        "--bin-mode",
        choices=("round", "floor", "ceil"),
        default="round",
        help="How to convert continuous radius to integer radial bins.",
    )
    return parser.parse_args()


def resolve_under_project(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    args = parse_args()

    input_dir = resolve_under_project(args.input_dir)
    output_dir = resolve_under_project(args.output_dir)
    input_path = input_dir / args.image
    output_dir.mkdir(parents=True, exist_ok=True)

    image, input_mode = load_grayscale_image(input_path)
    circularized = circularize_by_radius(image, bin_mode=args.bin_mode)

    stem = input_path.stem
    npy_path = output_dir / f"{stem}_radial_{args.bin_mode}.npy"
    png_path = output_dir / f"{stem}_radial_{args.bin_mode}.png"

    np.save(npy_path, circularized)
    save_preview_png(circularized, png_path, input_mode=input_mode)

    original_total = float(image.sum())
    output_total = float(circularized.sum())
    print(f"Input: {input_path}")
    print(f"Saved exact array: {npy_path}")
    print(f"Saved PNG preview: {png_path}")
    print(f"Original total brightness: {original_total:.12g}")
    print(f"Output total brightness:   {output_total:.12g}")
    print(f"Absolute difference:       {abs(output_total - original_total):.3e}")


if __name__ == "__main__":
    main()
