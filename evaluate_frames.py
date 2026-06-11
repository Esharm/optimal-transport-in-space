from __future__ import division, print_function

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from PIL import Image as PILImage
from skimage.metrics import structural_similarity as skimage_ssim

import ehtim as eh


# ============================================================
# Sorting / loading helpers
# ============================================================

VALID_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def natural_sort_key(path):
    """
    Sort filenames like frame_2.png before frame_10.png.
    """
    name = Path(path).name
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", name)
    ]


def list_images(folder):
    folder = Path(folder)

    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")

    files = [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_SUFFIXES
    ]

    files = sorted(files, key=natural_sort_key)

    if len(files) == 0:
        raise ValueError(f"No image files found in folder: {folder}")

    return files


def load_grayscale_array(path, target_size=None):
    """
    Load image as grayscale float array.

    target_size should be (width, height), PIL-style.
    """
    img = PILImage.open(path).convert("L")

    if target_size is not None:
        img = img.resize(target_size, PILImage.BILINEAR)

    arr = np.asarray(img, dtype=float)

    return arr


def minmax_normalize(arr):
    """
    Normalize array to [0, 1].
    """
    arr = np.asarray(arr, dtype=float)

    arr = arr - np.nanmin(arr)

    max_val = np.nanmax(arr)

    if max_val > 0:
        arr = arr / max_val

    return arr


def flux_normalize(arr, total_flux=1.0):
    """
    Normalize array so sum equals total_flux.
    This is useful for creating comparable ehtim images.
    """
    arr = np.asarray(arr, dtype=float)
    arr = arr - np.nanmin(arr)

    if np.nanmax(arr) > 0:
        arr = arr / np.nanmax(arr)

    s = np.nansum(arr)

    if s <= 0:
        return arr

    return arr / s * total_flux


# ============================================================
# Convert normal image arrays to ehtim Image objects
# ============================================================

def array_to_ehtim_image(arr, fov_uas=160.0, total_flux=1.0, source="frame"):
    """
    Convert a 2D grayscale array into an ehtim Image object.

    This is only for metric comparison. RA/Dec/RF are dummy metadata.
    """
    arr = flux_normalize(arr, total_flux=total_flux)

    if arr.ndim != 2:
        raise ValueError("Expected 2D grayscale image array.")

    ydim, xdim = arr.shape

    if xdim != ydim:
        raise ValueError(
            f"ehtim comparison is easiest with square images. Got {xdim} x {ydim}."
        )

    RADPERUAS = np.pi / (180.0 * 3600.0 * 1e6)
    fov_rad = fov_uas * RADPERUAS
    psize = fov_rad / xdim

    ra = 0.0
    dec = 0.0

    return eh.image.Image(
        arr,
        psize,
        ra,
        dec,
        rf=230e9,
        source=source
    )


# ============================================================
# Metrics
# ============================================================

def compute_ehtim_nrmse(original_im, recon_im, allow_shift=False):
    """
    Compute NRMSE using ehtim's built-in compare_images method.
    """

    try:
        result = original_im.compare_images(
            recon_im,
            metric=["nrmse"],
            shift=allow_shift,
            blur_frac=0.0
        )
    except TypeError:
        # Some ehtim versions may prefer metric as a string
        result = original_im.compare_images(
            recon_im,
            metric="nrmse",
            shift=allow_shift,
            blur_frac=0.0
        )

    # ehtim usually returns something like:
    # metric_value, image1_shifted, image2_shifted
    metric_value = result[0] if isinstance(result, tuple) else result

    if isinstance(metric_value, dict):
        metric_value = metric_value.get("nrmse")

    if isinstance(metric_value, (list, tuple, np.ndarray)):
        metric_value = np.asarray(metric_value).ravel()[0]

    return float(metric_value)


def compute_ssim(original_arr, recon_arr):
    """
    Compute SSIM using skimage.

    Images are min-max normalized first.
    """
    original_norm = minmax_normalize(original_arr)
    recon_norm = minmax_normalize(recon_arr)

    return float(
        skimage_ssim(
            original_norm,
            recon_norm,
            data_range=1.0
        )
    )


def compute_simple_nrmse(original_arr, recon_arr):
    """
    Optional plain array NRMSE, useful as a sanity check.
    This is not ehtim's built-in metric.
    """
    original = minmax_normalize(original_arr)
    recon = minmax_normalize(recon_arr)

    denom = np.sqrt(np.mean(original ** 2))

    if denom == 0:
        return np.nan

    return float(np.sqrt(np.mean((original - recon) ** 2)) / denom)


# ============================================================
# Main evaluation
# ============================================================

def evaluate_folders(
    original_dir,
    recon_dir,
    output_dir,
    fov_uas=160.0,
    total_flux=1.0,
    allow_shift=False,
    flip_original_vertical=False,
    flip_recon_vertical=False,
    max_frames=5
):
    original_dir = Path(original_dir)
    recon_dir = Path(recon_dir)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    original_files = list_images(original_dir)
    recon_files = list_images(recon_dir)

    if max_frames is not None:
        original_files = original_files[:max_frames]
        recon_files = recon_files[:max_frames]

    if len(original_files) != len(recon_files):
        print("WARNING: Folder image counts do not match.")
        print(f"Original count: {len(original_files)}")
        print(f"Recon count:    {len(recon_files)}")
        print("Comparing only the overlapping number of frames.")

    n = min(len(original_files), len(recon_files))

    records = []

    for i in range(n):
        original_path = original_files[i]
        recon_path = recon_files[i]

        print("=" * 60)
        print(f"Frame {i}")
        print(f"Original: {original_path.name}")
        print(f"Recon:    {recon_path.name}")

        # Load original first
        original_img_pil = PILImage.open(original_path).convert("L")
        target_size = original_img_pil.size  # (width, height)

        original_arr = np.asarray(original_img_pil, dtype=float)

        # Load reconstruction resized to original dimensions
        recon_arr = load_grayscale_array(recon_path, target_size=target_size)

        if flip_original_vertical:
            original_arr = np.flipud(original_arr)

        if flip_recon_vertical:
            recon_arr = np.flipud(recon_arr)

        if original_arr.shape != recon_arr.shape:
            raise ValueError(
                f"Shape mismatch after resizing: original {original_arr.shape}, "
                f"recon {recon_arr.shape}"
            )

        # ehtim requires square image for this simple constructor
        if original_arr.shape[0] != original_arr.shape[1]:
            side = min(original_arr.shape)

            original_arr = original_arr[:side, :side]
            recon_arr = recon_arr[:side, :side]

        original_eh = array_to_ehtim_image(
            original_arr,
            fov_uas=fov_uas,
            total_flux=total_flux,
            source=f"original_{i:04d}"
        )

        recon_eh = array_to_ehtim_image(
            recon_arr,
            fov_uas=fov_uas,
            total_flux=total_flux,
            source=f"recon_{i:04d}"
        )

        ehtim_nrmse = compute_ehtim_nrmse(
            original_eh,
            recon_eh,
            allow_shift=allow_shift
        )

        ssim_value = compute_ssim(original_arr, recon_arr)

        simple_nrmse = compute_simple_nrmse(original_arr, recon_arr)

        print(f"ehtim NRMSE: {ehtim_nrmse:.6f}")
        print(f"SSIM:        {ssim_value:.6f}")
        print(f"array NRMSE: {simple_nrmse:.6f}")

        records.append({
            "frame_index": i,
            "original_file": original_path.name,
            "recon_file": recon_path.name,
            "ehtim_nrmse": ehtim_nrmse,
            "ssim": ssim_value,
            "array_nrmse_sanity_check": simple_nrmse
        })

    df = pd.DataFrame(records)

    csv_path = output_dir / "frame_metrics.csv"
    df.to_csv(csv_path, index=False)

    print("=" * 60)
    print(f"Saved metrics CSV to: {csv_path}")

    # Summary
    summary_path = output_dir / "summary_metrics.csv"

    summary = df[["ehtim_nrmse", "ssim", "array_nrmse_sanity_check"]].describe()
    summary.to_csv(summary_path)

    print(f"Saved summary CSV to: {summary_path}")
    print(summary)

    # Plot NRMSE over frames
    plt.figure()
    plt.plot(df["frame_index"], df["ehtim_nrmse"], marker="o")
    plt.xlabel("Frame")
    plt.ylabel("ehtim NRMSE")
    plt.title("Frame-by-frame ehtim NRMSE")
    plt.savefig(output_dir / "nrmse_over_frames.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Plot SSIM over frames
    plt.figure()
    plt.plot(df["frame_index"], df["ssim"], marker="o")
    plt.xlabel("Frame")
    plt.ylabel("SSIM")
    plt.title("Frame-by-frame SSIM")
    plt.savefig(output_dir / "ssim_over_frames.png", dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved plots to: {output_dir}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Compare original and reconstructed image folders using ehtim NRMSE and SSIM."
    )

    parser.add_argument(
        "--original_dir",
        required=True,
        help="Folder containing original frames."
    )

    parser.add_argument(
        "--recon_dir",
        required=True,
        help="Folder containing reconstructed frames."
    )

    parser.add_argument(
        "--output_dir",
        default="evaluation_results",
        help="Folder where metrics CSV and plots will be saved."
    )

    parser.add_argument(
        "--fov_uas",
        type=float,
        default=160.0,
        help="Field of view in microarcseconds used when constructing ehtim images."
    )

    parser.add_argument(
        "--total_flux",
        type=float,
        default=1.0,
        help="Total flux normalization used for ehtim image comparison."
    )

    parser.add_argument(
        "--allow_shift",
        action="store_true",
        help="Allow ehtim to shift images for best alignment before NRMSE comparison."
    )

    parser.add_argument(
        "--flip_original_vertical",
        action="store_true",
        help="Flip original images vertically before comparison."
    )

    parser.add_argument(
        "--flip_recon_vertical",
        action="store_true",
        help="Flip reconstructed images vertically before comparison."
    )

    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Optional maximum number of frames to evaluate."
    )

    args = parser.parse_args()

    evaluate_folders(
        original_dir=args.original_dir,
        recon_dir=args.recon_dir,
        output_dir=args.output_dir,
        fov_uas=args.fov_uas,
        total_flux=args.total_flux,
        allow_shift=args.allow_shift,
        flip_original_vertical=args.flip_original_vertical,
        flip_recon_vertical=args.flip_recon_vertical,
        max_frames=args.max_frames
    )


if __name__ == "__main__":
    evaluate_folders(
        original_dir="optimal-transport-in-space/munchkin_testing/data",  #CHANGE input file 1
        recon_dir="Munchkin_Test_Results/reconstructed_frames_gray",      #CHANGE input file 2 
        output_dir="Munchkin_Test_Results/evaluation",                    #CHANGE output file
        fov_uas=160.0,
        total_flux=1.0,
        allow_shift=False,
        flip_original_vertical=False,
        flip_recon_vertical=False,
        max_frames=None
    )