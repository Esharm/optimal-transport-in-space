from __future__ import division, print_function

"""
Configurable evaluator for comparing a ground-truth image sequence against a
reconstructed image sequence.

Edit the CONFIG dictionary near the top of this file, then run:

    python evaluate_frames_configurable.py

No command-line arguments are required. Optional command-line overrides are kept
for convenience, but the intended workflow is to edit CONFIG.

Metrics supported:
  - NRMSE
  - SSIM
  - FRC, from image FFTs or optional sampled u-v/visibility data
  - Fourier chi-squared, from image FFTs or optional sampled u-v/visibility data
  - radial profile error
  - azimuthal profile error
  - temporal variance-map error
  - visibility-domain variability error
  - spatiotemporal gradient error

Optional u-v/visibility files:
  CSV/TXT/DAT expected columns:
      u, v, vis_real, vis_imag [, sigma]
    or
      u, v, real, imag [, sigma]
    or
      u, v, amp, phase [, sigma]
    or
      u, v, vis [, sigma]       # vis may contain complex strings such as 1+2j

  NPZ expected arrays:
      u, v, vis                 # vis may be complex
    or
      u, v, real, imag
    sigma is optional.

  NPY expected array shape:
      (N, 4) or (N, 5): columns are u, v, real, imag [, sigma]

If a single CSV/NPZ contains a frame column or frame-indexed arrays, the script
will try to select the current frame. If not, the same visibility set is reused
for every frame.
"""

import argparse
from copy import deepcopy
from dataclasses import dataclass
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from PIL import Image as PILImage
from skimage.metrics import structural_similarity as skimage_ssim

try:
    import ehtim as eh
except ImportError:  # Allows non-ehtim metrics to run on machines without ehtim.
    eh = None

try:
    from scipy.spatial import cKDTree
except ImportError:  # Nearest-neighbor u-v matching still has a slow fallback.
    cKDTree = None

try:
    from scipy.interpolate import RegularGridInterpolator
except ImportError:  # Image-to-visibility sampling falls back to nearest-neighbor FFT lookup.
    RegularGridInterpolator = None

try:
    from astropy.io import fits
except ImportError:  # UVFITS u-v sampling can still work through ehtim if installed.
    fits = None


# ============================================================
# SETTINGS AREA - edit this block for normal use
# ============================================================

CONFIG = {
    # --------------------------------------------------------
    # Input/output paths
    # --------------------------------------------------------
    "ground_truth_dir": "optimal-transport-in-space/blackhole_sim/data/aart_frames",
    "reconstruction_dir": "optimal-transport-in-space/blackhole_sim/data/aart_frames",
    "output_dir": "results/evaluation_test",
    "max_frames": None,  # Example: 10. Use None for all overlapping frames.

    # --------------------------------------------------------
    # Toggle metrics for each run
    # --------------------------------------------------------
    "metrics": {
        "nrmse": True,
        "ssim": True,
        "frc": True,
        "fourier_chi2": True,
        "radial_profile_error": True,
        "azimuthal_profile_error": True,
        "temporal_variance_map_error": False,

        # Sequence-level temporal metrics.
        # visibility_domain_variability_error needs u-v samples or matched visibility files.
        "visibility_domain_variability_error": True,
        "spatiotemporal_gradient_error": True,

        # Optional. Uses ehtim's compare_images when ehtim is installed.
        # The array-based NRMSE above is usually enough for folder comparisons.
        "ehtim_nrmse": False,
    },

    # --------------------------------------------------------
    # Image preprocessing and normalization
    # --------------------------------------------------------
    "normalization": "flux",  # minmax, flux, zscore, none
    "total_flux": 1.0,
    "fov_uas": 160.0,
    "flip_ground_truth_vertical": False,
    "flip_reconstruction_vertical": False,
    "crop_to_square": True,

    # Optional image alignment. Disabled by default because shifts may be a
    # scientifically meaningful error in EHT imaging.
    "align_reconstruction": True,
    "max_shift_pixels": 4,
    "allow_ehtim_shift": True,

    # --------------------------------------------------------
    # FRC settings
    # --------------------------------------------------------
    # image: FRC from image FFTs
    # uv:    FRC from optional sampled u-v/visibility data
    # auto:  use u-v data when available; otherwise use image FFTs
    "frc_source": "auto",  # auto, image, uv
    "frc_num_bins": None,
    "frc_threshold": 0.143,
    "frc_min_samples_per_ring": 1,

    # --------------------------------------------------------
    # Fourier chi-squared settings
    # --------------------------------------------------------
    # image: image-to-image Fourier proxy
    # uv:    visibility-domain chi-squared using optional u-v data
    # auto:  use u-v data when available; otherwise use image FFT proxy
    "fourier_chi2_source": "auto",  # auto, image, uv
    "fourier_chi2_mode": "complex",  # complex or amplitude for image FFT proxy
    "fourier_chi2_epsilon": 1e-8,
    "fourier_chi2_denominator": "global_power",  # global_power or per_frequency_power
    "include_dc_in_fourier_chi2": False,

    # --------------------------------------------------------
    # Visibility-domain variability settings
    # --------------------------------------------------------
    # This compares how the visibility amplitudes vary across time.
    # Sources:
    #   auto:           use matched ground_truth/reconstruction visibility files when
    #                   available; otherwise use observation u-v samples and compute
    #                   predicted visibilities from the image frames.
    #   matched_uv:     use ground_truth_uv_* and reconstruction_uv_* visibility files.
    #   observation_uv: use observation_uv_* files only for u-v coordinates, then sample
    #                   both image sequences at those u-v coordinates.
    "visibility_variability": {
        "source": "auto",  # auto, matched_uv, observation_uv
        "mode": "radial_bins",  # radial_bins or sample_index
        "num_bins": 24,
        "min_samples_per_bin": 3,
        "amplitude_mode": "amplitude",  # amplitude or log_amplitude
        "norm": "l2",  # l2 or l1
        "epsilon": 1e-12,
    },

    # --------------------------------------------------------
    # Spatiotemporal gradient error settings
    # --------------------------------------------------------
    # This compares spatial gradients (ring/shadow edges) and temporal gradients
    # (frame-to-frame changes) between the ground truth and reconstruction.
    "spatiotemporal_gradient": {
        "norm": "l2",  # l2 or l1
        "spatial_weight": 1.0,
        "temporal_weight": 1.0,
        "use_absolute_gradients": False,
        "save_error_maps": True,
        "epsilon": 1e-12,
    },

    # --------------------------------------------------------
    # Radial and azimuthal profile settings
    # --------------------------------------------------------
    "radial_bins": 64,
    "azimuthal_bins": 72,
    "center_x": None,  # None uses image center
    "center_y": None,
    "azimuthal_inner_radius": None,
    "azimuthal_outer_radius": None,
    "azimuthal_annulus_width_fraction": 0.25,
    "allow_azimuthal_roll": False,

    # --------------------------------------------------------
    # Optional u-v / visibility data settings
    # --------------------------------------------------------
    "uv_data": {
        "enabled": False,

        # Use either per-frame folders OR single files. Folders are easiest.
        # If folders are used, files are naturally sorted and matched by index.
        "ground_truth_uv_dir": None,
        "reconstruction_uv_dir": None,

        # Single-file mode. If a frame column exists, each frame is selected.
        # Otherwise the same visibility set is reused for every frame.
        "ground_truth_uv_file": None,
        "reconstruction_uv_file": None,

        # Observation u-v sampling mode. These files only need u and v coordinates.
        # This is useful when you have UVFITS observation files but not precomputed
        # visibility tables for the ground truth and reconstruction. The script will
        # compute predicted visibilities from the image frames at these sampled u-v points.
        "observation_uv_dir": None,
        "observation_uv_file": None,
        "observation_uv_frame_column": "frame_index",

        # Coordinate unit for observation u/v samples:
        #   auto    = seconds for UVFITS/FITS files, wavelengths for CSV/NPY/NPZ
        #   seconds = multiply u and v by observing_frequency_hz
        #   lambda  = already in wavelengths
        #   klambda, mlambda, glambda are also accepted
        "observation_uv_coordinate_unit": "auto",
        "observing_frequency_hz": 226e9,
        "image_sampling_interpolation": "linear",  # linear or nearest
        "image_sampling_out_of_bounds": "drop",  # drop or zero

        # CSV frame column names tried include this plus frame, frame_index, t, time.
        "frame_column": "frame_index",

        # row: assumes both visibility files use the same row order.
        # nearest: matches reconstruction samples to ground-truth samples by nearest (u,v).
        "match_mode": "row",  # row or nearest
        "nearest_tolerance": None,  # Example: 1e-6. None keeps all nearest matches.

        # auto assumes phases are degrees when abs(phase) is larger than 2*pi.
        "phase_unit": "auto",  # auto, radians, degrees

        # Which sigma values to use for visibility-domain chi-squared.
        # If sigma is missing, fixed_sigma is used.
        "sigma_mode": "ground_truth",  # ground_truth, reconstruction, average, fixed
        "fixed_sigma": 1.0,

        # FRC binning when using sampled u-v data.
        "uv_frc_num_bins": None,
        "uv_frc_min_samples_per_ring": 2,
    },

    # --------------------------------------------------------
    # Output behavior
    # --------------------------------------------------------
    "save_plots": True,
    "save_intermediate_csvs": True,
    "verbose": True,
}


# ============================================================
# Configuration / constants
# ============================================================

VALID_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
VALID_UV_SUFFIXES = {".csv", ".txt", ".dat", ".npy", ".npz", ".uvfits", ".fits"}
EPS = 1e-12
RAD_PER_UAS = np.pi / (180.0 * 3600.0 * 1e6)

try:
    RESAMPLE_BILINEAR = PILImage.Resampling.BILINEAR
except AttributeError:  # Pillow < 9
    RESAMPLE_BILINEAR = PILImage.BILINEAR


@dataclass
class VisibilityData:
    """Container for sampled Fourier/u-v visibility data."""

    u: np.ndarray
    v: np.ndarray
    vis: np.ndarray
    sigma: np.ndarray = None
    source_path: str = ""
    frame_index: int = None


@dataclass
class UVSamplingData:
    """Container for u-v coordinates used to sample predicted image visibilities."""

    u: np.ndarray
    v: np.ndarray
    source_path: str = ""
    frame_index: int = None
    coordinate_unit: str = "lambda"


@dataclass
class MatchedVisibilityData:
    """Matched visibility samples for comparing two visibility sets."""

    u: np.ndarray
    v: np.ndarray
    vis_true: np.ndarray
    vis_recon: np.ndarray
    sigma_true: np.ndarray = None
    sigma_recon: np.ndarray = None
    matched_count: int = 0
    match_mode: str = "row"


# ============================================================
# Sorting / loading helpers
# ============================================================

def natural_sort_key(path):
    """Sort filenames like frame_2.png before frame_10.png."""
    name = Path(path).name
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", name)
    ]


def has_valid_suffix(path, valid_suffixes):
    """Return True when a path has one of the accepted suffixes."""
    path = Path(path)
    lower_name = path.name.lower()

    if lower_name.endswith(".uvfits.gz") and ".uvfits" in valid_suffixes:
        return True
    if lower_name.endswith(".fits.gz") and ".fits" in valid_suffixes:
        return True

    return path.suffix.lower() in valid_suffixes


def list_files(folder, valid_suffixes, kind_name):
    folder = Path(folder)

    if not folder.exists():
        raise FileNotFoundError(f"{kind_name} folder does not exist: {folder}")

    files = [
        p for p in folder.iterdir()
        if p.is_file() and has_valid_suffix(p, valid_suffixes)
    ]

    files = sorted(files, key=natural_sort_key)

    if len(files) == 0:
        raise ValueError(f"No {kind_name} files found in folder: {folder}")

    return files


def list_images(folder):
    return list_files(folder, VALID_IMAGE_SUFFIXES, "image")


def list_uv_files(folder):
    return list_files(folder, VALID_UV_SUFFIXES, "u-v/visibility")


def load_grayscale_array(path, target_size=None):
    """
    Load image as grayscale float array.

    target_size should be (width, height), PIL-style.
    """
    img = PILImage.open(path).convert("L")

    if target_size is not None:
        img = img.resize(target_size, RESAMPLE_BILINEAR)

    return np.asarray(img, dtype=float)


def center_crop_square_pair(arr1, arr2):
    """Center-crop two same-shaped arrays to a common square."""
    if arr1.shape != arr2.shape:
        raise ValueError(f"Cannot crop arrays with different shapes: {arr1.shape}, {arr2.shape}")

    h, w = arr1.shape
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2

    return (
        arr1[y0:y0 + side, x0:x0 + side],
        arr2[y0:y0 + side, x0:x0 + side],
    )


# ============================================================
# Normalization helpers
# ============================================================

def minmax_normalize(arr):
    """Normalize array to [0, 1]."""
    arr = np.asarray(arr, dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    arr = arr - np.nanmin(arr)
    max_val = np.nanmax(arr)

    if max_val > 0:
        arr = arr / max_val

    return arr


def flux_normalize(arr, total_flux=1.0):
    """Normalize array so its sum equals total_flux."""
    arr = np.asarray(arr, dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    arr = arr - np.nanmin(arr)

    if np.nanmax(arr) > 0:
        arr = arr / np.nanmax(arr)

    s = np.nansum(arr)

    if s <= 0:
        return arr

    return arr / s * total_flux


def zscore_normalize(arr):
    """Zero-mean, unit-standard-deviation normalization."""
    arr = np.asarray(arr, dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    std = np.nanstd(arr)
    if std <= 0:
        return arr * 0.0

    return (arr - np.nanmean(arr)) / std


def normalize_array(arr, mode="minmax", total_flux=1.0):
    """
    Normalize an image before metric computation.

    mode options:
      - minmax: subtract minimum and divide by maximum.
      - flux: subtract minimum and scale total flux to total_flux.
      - zscore: zero-mean/unit-std.
      - none: use raw grayscale values.
    """
    mode = str(mode).lower()

    if mode == "minmax":
        return minmax_normalize(arr)
    if mode == "flux":
        return flux_normalize(arr, total_flux=total_flux)
    if mode == "zscore":
        return zscore_normalize(arr)
    if mode == "none":
        arr = np.asarray(arr, dtype=float)
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    raise ValueError(f"Unknown normalization mode: {mode}")


# ============================================================
# Optional image alignment helper
# ============================================================

def compute_ncc(a, b):
    """Normalized cross-correlation between two arrays."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    a = a - np.mean(a)
    b = b - np.mean(b)

    denom = np.sqrt(np.sum(a ** 2) * np.sum(b ** 2))
    if denom <= EPS:
        return np.nan

    return float(np.sum(a * b) / denom)


def shift_array_zero_fill(arr, dy, dx, fill_value=0.0):
    """
    Shift an array by integer pixels without wrap-around.

    Positive dy shifts downward. Positive dx shifts rightward.
    """
    arr = np.asarray(arr, dtype=float)
    h, w = arr.shape
    out = np.full_like(arr, fill_value, dtype=float)

    if abs(dy) >= h or abs(dx) >= w:
        return out

    if dy >= 0:
        src_y = slice(0, h - dy)
        dst_y = slice(dy, h)
    else:
        src_y = slice(-dy, h)
        dst_y = slice(0, h + dy)

    if dx >= 0:
        src_x = slice(0, w - dx)
        dst_x = slice(dx, w)
    else:
        src_x = slice(-dx, w)
        dst_x = slice(0, w + dx)

    out[dst_y, dst_x] = arr[src_y, src_x]
    return out


def align_reconstruction_by_ncc(original_arr, recon_arr, max_shift_pixels=0, normalization="minmax", total_flux=1.0):
    """Integer-pixel search that maximizes NCC; disabled when max_shift_pixels <= 0."""
    if max_shift_pixels is None or max_shift_pixels <= 0:
        return recon_arr, 0, 0, np.nan

    original_norm = normalize_array(original_arr, mode=normalization, total_flux=total_flux)
    recon_norm = normalize_array(recon_arr, mode=normalization, total_flux=total_flux)

    best_score = -np.inf
    best_dy = 0
    best_dx = 0
    best_shifted = recon_arr
    fill_value = float(np.nanmin(recon_arr))

    for dy in range(-int(max_shift_pixels), int(max_shift_pixels) + 1):
        for dx in range(-int(max_shift_pixels), int(max_shift_pixels) + 1):
            shifted_norm = shift_array_zero_fill(recon_norm, dy, dx, fill_value=0.0)
            score = compute_ncc(original_norm, shifted_norm)

            if np.isfinite(score) and score > best_score:
                best_score = score
                best_dy = dy
                best_dx = dx
                best_shifted = shift_array_zero_fill(recon_arr, dy, dx, fill_value=fill_value)

    if not np.isfinite(best_score):
        return recon_arr, 0, 0, np.nan

    return best_shifted, best_dy, best_dx, float(best_score)


# ============================================================
# Convert normal image arrays to ehtim Image objects
# ============================================================

def array_to_ehtim_image(arr, fov_uas=160.0, total_flux=1.0, source="frame"):
    """Convert a 2D grayscale array into an ehtim Image object."""
    if eh is None:
        raise ImportError("ehtim is not installed; cannot construct ehtim Image objects.")

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

    return eh.image.Image(
        arr,
        psize,
        0.0,
        0.0,
        rf=230e9,
        source=source,
    )


# ============================================================
# Frame-level image metrics
# ============================================================

def compute_ehtim_nrmse(original_im, recon_im, allow_shift=False):
    """Compute NRMSE using ehtim's built-in compare_images method."""
    try:
        result = original_im.compare_images(
            recon_im,
            metric=["nrmse"],
            shift=allow_shift,
            blur_frac=0.0,
        )
    except TypeError:
        result = original_im.compare_images(
            recon_im,
            metric="nrmse",
            shift=allow_shift,
            blur_frac=0.0,
        )

    metric_value = result[0] if isinstance(result, tuple) else result

    if isinstance(metric_value, dict):
        metric_value = metric_value.get("nrmse")

    if isinstance(metric_value, (list, tuple, np.ndarray)):
        metric_value = np.asarray(metric_value).ravel()[0]

    return float(metric_value)


def compute_nrmse(original_arr, recon_arr, normalization="minmax", total_flux=1.0):
    """Array-based normalized RMSE: RMS(original - recon) / RMS(original)."""
    original = normalize_array(original_arr, mode=normalization, total_flux=total_flux)
    recon = normalize_array(recon_arr, mode=normalization, total_flux=total_flux)

    denom = np.sqrt(np.mean(original ** 2))

    if denom <= EPS:
        return np.nan

    return float(np.sqrt(np.mean((original - recon) ** 2)) / denom)


def compute_ssim(original_arr, recon_arr):
    """Compute SSIM using skimage. Images are min-max normalized first."""
    original_norm = minmax_normalize(original_arr)
    recon_norm = minmax_normalize(recon_arr)

    min_dim = min(original_norm.shape)
    if min_dim < 3:
        return np.nan

    win_size = min(7, min_dim)
    if win_size % 2 == 0:
        win_size -= 1
    if win_size < 3:
        return np.nan

    return float(
        skimage_ssim(
            original_norm,
            recon_norm,
            data_range=1.0,
            win_size=win_size,
        )
    )


def compute_fourier_chi_squared_image_proxy(
    original_arr,
    recon_arr,
    normalization="minmax",
    total_flux=1.0,
    epsilon=1e-8,
    mode="complex",
    denominator_mode="global_power",
    exclude_dc=True,
):
    """
    Fourier-domain chi-squared-style image-to-image discrepancy.

    This is a proxy when measured visibilities and uncertainty values are not
    provided. If u-v/visibility data are provided, use compute_uv_chi_squared.
    """
    original = normalize_array(original_arr, mode=normalization, total_flux=total_flux)
    recon = normalize_array(recon_arr, mode=normalization, total_flux=total_flux)

    f_original = np.fft.fftshift(np.fft.fft2(original))
    f_recon = np.fft.fftshift(np.fft.fft2(recon))

    if str(mode).lower() == "amplitude":
        residual_sq = (np.abs(f_recon) - np.abs(f_original)) ** 2
    elif str(mode).lower() == "complex":
        residual_sq = np.abs(f_recon - f_original) ** 2
    else:
        raise ValueError(f"Unknown Fourier chi-squared mode: {mode}")

    power = np.abs(f_original) ** 2
    scale = np.nanmean(power)

    if scale <= EPS:
        return np.nan

    mask = np.ones(original.shape, dtype=bool)
    if exclude_dc:
        cy, cx = original.shape[0] // 2, original.shape[1] // 2
        mask[cy, cx] = False

    residual_sq = residual_sq[mask]
    power = power[mask]

    denominator_mode = str(denominator_mode).lower()

    if denominator_mode == "global_power":
        denom = np.nansum(power) + float(epsilon) * scale * power.size + EPS
        return float(np.nansum(residual_sq) / denom)

    if denominator_mode == "per_frequency_power":
        denom = power + float(epsilon) * scale + EPS
        chi_values = residual_sq / denom
        return float(np.nanmean(chi_values))

    raise ValueError(f"Unknown Fourier chi-squared denominator mode: {denominator_mode}")


# ============================================================
# Fourier Ring Correlation from image FFTs
# ============================================================

def fft_radius_grid(shape):
    """Radius grid in shifted Fourier-pixel coordinates."""
    ydim, xdim = shape
    y, x = np.indices((ydim, xdim))
    cy, cx = ydim // 2, xdim // 2
    return np.sqrt((x - cx) ** 2 + (y - cy) ** 2)


def compute_frc_curve_image(
    original_arr,
    recon_arr,
    normalization="minmax",
    total_flux=1.0,
    num_bins=None,
    min_samples=1,
):
    """Compute Fourier Ring Correlation from two image FFTs."""
    original = normalize_array(original_arr, mode=normalization, total_flux=total_flux)
    recon = normalize_array(recon_arr, mode=normalization, total_flux=total_flux)

    f_original = np.fft.fftshift(np.fft.fft2(original))
    f_recon = np.fft.fftshift(np.fft.fft2(recon))

    radius = fft_radius_grid(original.shape)
    max_radius = min(original.shape) // 2

    if max_radius <= 0:
        return np.array([]), np.array([]), np.array([])

    if num_bins is None:
        num_bins = max_radius

    num_bins = int(max(1, num_bins))
    edges = np.linspace(0.0, float(max_radius), num_bins + 1)

    freq_mid_norm = []
    frc_values = []
    sample_counts = []

    for i in range(num_bins):
        low = edges[i]
        high = edges[i + 1]

        if i == num_bins - 1:
            mask = (radius >= low) & (radius <= high)
        else:
            mask = (radius >= low) & (radius < high)

        count = int(np.count_nonzero(mask))
        sample_counts.append(count)

        if count < int(min_samples):
            frc = np.nan
        else:
            a = f_original[mask]
            b = f_recon[mask]

            numerator = np.sum(a * np.conj(b))
            denominator = np.sqrt(np.sum(np.abs(a) ** 2) * np.sum(np.abs(b) ** 2))

            if denominator <= EPS:
                frc = np.nan
            else:
                frc = np.real(numerator) / denominator
                frc = float(np.clip(frc, -1.0, 1.0))

        freq_mid = 0.5 * (low + high) / max_radius
        freq_mid_norm.append(freq_mid)
        frc_values.append(frc)

    return (
        np.asarray(freq_mid_norm, dtype=float),
        np.asarray(frc_values, dtype=float),
        np.asarray(sample_counts, dtype=int),
    )


def curve_auc(x, y):
    """Average area under a curve over the valid x-range."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)

    if np.count_nonzero(valid) == 0:
        return np.nan
    if np.count_nonzero(valid) == 1:
        return float(y[valid][0])

    xv = x[valid]
    yv = y[valid]
    denom = np.nanmax(xv) - np.nanmin(xv)

    if denom <= EPS:
        return np.nan

    area = np.trapezoid(yv, xv) if hasattr(np, "trapezoid") else np.trapz(yv, xv)
    return float(area / denom)


def frc_cutoff_frequency(freq, frc, threshold=0.5):
    """
    Return the largest normalized frequency where FRC >= threshold.

    This is a simple cutoff summary. It is not a formal resolution proof.
    """
    freq = np.asarray(freq, dtype=float)
    frc = np.asarray(frc, dtype=float)
    valid = np.isfinite(freq) & np.isfinite(frc)

    if np.count_nonzero(valid) == 0:
        return np.nan

    freq_v = freq[valid]
    frc_v = frc[valid]

    above = frc_v >= float(threshold)
    if np.count_nonzero(above) == 0:
        return np.nan

    return float(np.nanmax(freq_v[above]))


# ============================================================
# Radial / azimuthal profile metrics
# ============================================================

def default_center(shape, center_x=None, center_y=None):
    """Return center as (x, y)."""
    h, w = shape
    x = (w - 1) / 2.0 if center_x is None else float(center_x)
    y = (h - 1) / 2.0 if center_y is None else float(center_y)
    return x, y


def max_complete_radius(shape, center):
    """Largest radius fully inside the image bounds."""
    h, w = shape
    x0, y0 = center
    return float(min(x0, y0, w - 1 - x0, h - 1 - y0))


def radius_theta_grids(shape, center):
    """Return radius and theta grids for an image shape."""
    h, w = shape
    x0, y0 = center
    y, x = np.indices((h, w))
    dx = x - x0
    dy = y - y0
    radius = np.sqrt(dx ** 2 + dy ** 2)
    theta = np.mod(np.arctan2(dy, dx), 2.0 * np.pi)
    return radius, theta


def radial_profile(arr, center=None, num_bins=64, r_max=None):
    """Mean brightness as a function of radius."""
    arr = np.asarray(arr, dtype=float)

    if center is None:
        center = default_center(arr.shape)

    if r_max is None:
        r_max = max_complete_radius(arr.shape, center)

    radius, _ = radius_theta_grids(arr.shape, center)
    edges = np.linspace(0.0, float(r_max), int(num_bins) + 1)
    mids = 0.5 * (edges[:-1] + edges[1:])

    profile = []
    counts = []

    for i in range(int(num_bins)):
        low = edges[i]
        high = edges[i + 1]

        if i == int(num_bins) - 1:
            mask = (radius >= low) & (radius <= high)
        else:
            mask = (radius >= low) & (radius < high)

        count = int(np.count_nonzero(mask))
        counts.append(count)

        if count == 0:
            profile.append(np.nan)
        else:
            profile.append(float(np.nanmean(arr[mask])))

    return mids, np.asarray(profile, dtype=float), np.asarray(counts, dtype=int)


def normalized_profile_l2_error(true_profile, recon_profile):
    """Normalized L2 error between two 1D profiles."""
    true_profile = np.asarray(true_profile, dtype=float)
    recon_profile = np.asarray(recon_profile, dtype=float)

    valid = np.isfinite(true_profile) & np.isfinite(recon_profile)

    if np.count_nonzero(valid) == 0:
        return np.nan

    denom = np.linalg.norm(true_profile[valid])
    if denom <= EPS:
        return np.nan

    return float(np.linalg.norm(recon_profile[valid] - true_profile[valid]) / denom)


def estimate_annulus_from_truth(
    original_arr,
    center,
    radial_bins=64,
    width_fraction=0.25,
    r_max=None,
):
    """
    Estimate an annulus for azimuthal profiles from the ground-truth frame.

    The default annulus is centered on the brightest radial-profile peak, with
    half-width = width_fraction * peak_radius.
    """
    radii, profile, _ = radial_profile(
        original_arr,
        center=center,
        num_bins=radial_bins,
        r_max=r_max,
    )

    valid = np.isfinite(profile)

    if np.count_nonzero(valid) > 3:
        valid_indices = np.where(valid)[0]
        valid_indices = valid_indices[valid_indices > 0]
    else:
        valid_indices = np.where(valid)[0]

    if len(valid_indices) == 0:
        r_peak = 0.5 * max_complete_radius(original_arr.shape, center)
    else:
        best_local = valid_indices[np.nanargmax(profile[valid_indices])]
        r_peak = float(radii[best_local])

    r_max_use = float(r_max) if r_max is not None else max_complete_radius(original_arr.shape, center)
    half_width = max(2.0, float(width_fraction) * max(r_peak, 1.0))

    inner = max(0.0, r_peak - half_width)
    outer = min(r_max_use, r_peak + half_width)

    if outer <= inner:
        inner = max(0.0, r_peak - 2.0)
        outer = min(r_max_use, r_peak + 2.0)

    return inner, outer, r_peak


def azimuthal_profile(
    arr,
    center=None,
    num_bins=72,
    inner_radius=None,
    outer_radius=None,
):
    """Mean brightness as a function of angle within a chosen annulus."""
    arr = np.asarray(arr, dtype=float)

    if center is None:
        center = default_center(arr.shape)

    r_max = max_complete_radius(arr.shape, center)

    if inner_radius is None:
        inner_radius = 0.0
    if outer_radius is None:
        outer_radius = r_max

    radius, theta = radius_theta_grids(arr.shape, center)
    annulus_mask = (radius >= float(inner_radius)) & (radius <= float(outer_radius))

    edges = np.linspace(0.0, 2.0 * np.pi, int(num_bins) + 1)
    mids = 0.5 * (edges[:-1] + edges[1:])

    profile = []
    counts = []

    for i in range(int(num_bins)):
        low = edges[i]
        high = edges[i + 1]

        if i == int(num_bins) - 1:
            mask = annulus_mask & (theta >= low) & (theta <= high)
        else:
            mask = annulus_mask & (theta >= low) & (theta < high)

        count = int(np.count_nonzero(mask))
        counts.append(count)

        if count == 0:
            profile.append(np.nan)
        else:
            profile.append(float(np.nanmean(arr[mask])))

    return mids, np.asarray(profile, dtype=float), np.asarray(counts, dtype=int)


def azimuthal_profile_error(true_profile, recon_profile, allow_roll=False):
    """
    Normalized L2 error between angular profiles.

    If allow_roll=True, the reconstruction profile is circularly shifted to find
    the best angular alignment. Default is False because image orientation is
    usually scientifically meaningful.
    """
    if not allow_roll:
        return normalized_profile_l2_error(true_profile, recon_profile), 0

    true_profile = np.asarray(true_profile, dtype=float)
    recon_profile = np.asarray(recon_profile, dtype=float)

    best_error = np.inf
    best_roll = 0

    for roll in range(len(recon_profile)):
        rolled = np.roll(recon_profile, roll)
        err = normalized_profile_l2_error(true_profile, rolled)
        if np.isfinite(err) and err < best_error:
            best_error = err
            best_roll = roll

    if not np.isfinite(best_error):
        return np.nan, 0

    return float(best_error), int(best_roll)


# ============================================================
# Temporal variance-map metric
# ============================================================

def compute_temporal_variance_map_error(
    original_stack,
    recon_stack,
    normalization="minmax",
    total_flux=1.0,
):
    """
    Compare where variability occurs over time.

    For each sequence, compute per-pixel temporal variance, then compare the
    variance maps using normalized L2 error.
    """
    original_stack = np.asarray(original_stack, dtype=float)
    recon_stack = np.asarray(recon_stack, dtype=float)

    if original_stack.shape != recon_stack.shape:
        raise ValueError(
            f"Temporal stacks must have the same shape. Got {original_stack.shape} and {recon_stack.shape}."
        )

    if original_stack.ndim != 3:
        raise ValueError("Expected stacks with shape (frames, height, width).")

    if original_stack.shape[0] < 2:
        variance_true = np.zeros(original_stack.shape[1:], dtype=float)
        variance_recon = np.zeros(recon_stack.shape[1:], dtype=float)
        return np.nan, variance_true, variance_recon, np.zeros_like(variance_true)

    original_norm = np.stack([
        normalize_array(frame, mode=normalization, total_flux=total_flux)
        for frame in original_stack
    ], axis=0)

    recon_norm = np.stack([
        normalize_array(frame, mode=normalization, total_flux=total_flux)
        for frame in recon_stack
    ], axis=0)

    variance_true = np.var(original_norm, axis=0, ddof=1)
    variance_recon = np.var(recon_norm, axis=0, ddof=1)
    abs_error_map = np.abs(variance_recon - variance_true)

    denom = np.linalg.norm(variance_true.ravel())
    if denom <= EPS:
        variance_error = np.nan
    else:
        variance_error = float(np.linalg.norm((variance_recon - variance_true).ravel()) / denom)

    return variance_error, variance_true, variance_recon, abs_error_map


# ============================================================
# U-V / visibility data loading and metrics
# ============================================================

def canonical_column_name(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def find_column(columns, candidates):
    """Find a DataFrame column using flexible aliases."""
    canonical_map = {canonical_column_name(col): col for col in columns}
    for candidate in candidates:
        key = canonical_column_name(candidate)
        if key in canonical_map:
            return canonical_map[key]
    return None


def phase_to_radians(phase, phase_unit="auto"):
    phase = np.asarray(phase, dtype=float)
    unit = str(phase_unit).lower()

    if unit in {"rad", "radian", "radians"}:
        return phase
    if unit in {"deg", "degree", "degrees"}:
        return np.deg2rad(phase)
    if unit == "auto":
        finite = phase[np.isfinite(phase)]
        if finite.size > 0 and np.nanmax(np.abs(finite)) > (2.0 * np.pi + 0.5):
            return np.deg2rad(phase)
        return phase

    raise ValueError(f"Unknown phase unit: {phase_unit}")


def parse_complex_series(values):
    """Parse a pandas Series or array containing complex strings/numbers."""
    out = []
    for value in values:
        if isinstance(value, complex):
            out.append(value)
        else:
            text = str(value).strip().replace("i", "j")
            try:
                out.append(complex(text))
            except ValueError:
                out.append(np.nan + 1j * np.nan)
    return np.asarray(out, dtype=complex)


def select_frame_rows(df, frame_index=None, frame_column="frame_index"):
    """Return rows for a frame if a frame column exists; otherwise return df unchanged."""
    if frame_index is None:
        return df

    candidate_cols = [frame_column, "frame_index", "frame", "t", "time", "idx", "index"]
    col = find_column(df.columns, candidate_cols)
    if col is None:
        return df

    filtered = df[df[col].astype(str) == str(frame_index)]
    if len(filtered) == 0:
        # Also try numeric comparison for float/int frame labels.
        numeric = pd.to_numeric(df[col], errors="coerce")
        filtered = df[numeric == float(frame_index)]

    return filtered


def first_available_key(npz_data, candidates):
    keys = list(npz_data.keys())
    key_map = {canonical_column_name(key): key for key in keys}
    for candidate in candidates:
        ck = canonical_column_name(candidate)
        if ck in key_map:
            return key_map[ck]
    return None


def maybe_select_frame_array(arr, frame_index=None):
    """
    Select a frame from an array if it appears to be frame-indexed.

    If arr is 2D and frame_index is available, we treat the first dimension as
    frame only when selecting one row still leaves a 1D sample array. This works
    for arrays shaped (T, N). For normal N-by-columns arrays, loader-specific
    code handles the structure instead.
    """
    arr = np.asarray(arr)
    if frame_index is None:
        return arr
    if arr.ndim >= 2 and arr.shape[0] > int(frame_index):
        return arr[int(frame_index)]
    return arr


def load_uv_from_npz(path, frame_index=None, frame_column="frame_index", phase_unit="auto"):
    data = np.load(path, allow_pickle=True)

    u_key = first_available_key(data, ["u", "uu", "u_lambda", "u_lambdas", "ucoord"])
    v_key = first_available_key(data, ["v", "vv", "v_lambda", "v_lambdas", "vcoord"])

    if u_key is None or v_key is None:
        raise ValueError(f"NPZ file must contain u and v arrays: {path}")

    u = maybe_select_frame_array(data[u_key], frame_index).astype(float).ravel()
    v = maybe_select_frame_array(data[v_key], frame_index).astype(float).ravel()

    vis_key = first_available_key(data, ["vis", "visibility", "complex_vis", "complex_visibility"])
    real_key = first_available_key(data, ["vis_real", "real", "re", "vis_re", "r"])
    imag_key = first_available_key(data, ["vis_imag", "imag", "im", "vis_im", "i"])
    amp_key = first_available_key(data, ["amp", "amplitude", "vis_amp", "absvis", "visamp"])
    phase_key = first_available_key(data, ["phase", "phi", "vis_phase", "phase_rad", "phase_deg"])

    if vis_key is not None:
        vis = maybe_select_frame_array(data[vis_key], frame_index).ravel().astype(complex)
    elif real_key is not None and imag_key is not None:
        real = maybe_select_frame_array(data[real_key], frame_index).astype(float).ravel()
        imag = maybe_select_frame_array(data[imag_key], frame_index).astype(float).ravel()
        vis = real + 1j * imag
    elif amp_key is not None and phase_key is not None:
        amp = maybe_select_frame_array(data[amp_key], frame_index).astype(float).ravel()
        phase = phase_to_radians(maybe_select_frame_array(data[phase_key], frame_index), phase_unit).ravel()
        vis = amp * np.exp(1j * phase)
    else:
        raise ValueError(
            f"NPZ file must contain vis, real+imag, or amp+phase arrays: {path}"
        )

    sigma_key = first_available_key(data, ["sigma", "std", "uncertainty", "error", "noise", "sigma_vis"])
    sigma = None
    if sigma_key is not None:
        sigma = maybe_select_frame_array(data[sigma_key], frame_index).astype(float).ravel()

    return clean_visibility_data(VisibilityData(u=u, v=v, vis=vis, sigma=sigma, source_path=str(path), frame_index=frame_index))


def load_uv_from_npy(path, frame_index=None):
    arr = np.load(path, allow_pickle=True)
    arr = np.asarray(arr)

    if frame_index is not None and arr.ndim == 3 and arr.shape[0] > int(frame_index):
        arr = arr[int(frame_index)]

    if arr.ndim != 2 or arr.shape[1] < 4:
        raise ValueError(
            f"NPY visibility file should have shape (N, 4) or (N, 5): {path}"
        )

    u = arr[:, 0].astype(float)
    v = arr[:, 1].astype(float)
    vis = arr[:, 2].astype(float) + 1j * arr[:, 3].astype(float)
    sigma = arr[:, 4].astype(float) if arr.shape[1] >= 5 else None

    return clean_visibility_data(VisibilityData(u=u, v=v, vis=vis, sigma=sigma, source_path=str(path), frame_index=frame_index))


def read_table_auto(path):
    """Read CSV/TXT/DAT with automatic delimiter detection and whitespace fallback."""
    try:
        return pd.read_csv(path, sep=None, engine="python", comment="#")
    except Exception:
        return pd.read_csv(path, delim_whitespace=True, comment="#")


def load_uv_from_table(path, frame_index=None, frame_column="frame_index", phase_unit="auto"):
    df = read_table_auto(path)
    df = select_frame_rows(df, frame_index=frame_index, frame_column=frame_column)

    if len(df) == 0:
        raise ValueError(f"No rows found for frame {frame_index} in {path}")

    u_col = find_column(df.columns, ["u", "uu", "u_lambda", "u_lambdas", "ucoord", "u_coordinate"])
    v_col = find_column(df.columns, ["v", "vv", "v_lambda", "v_lambdas", "vcoord", "v_coordinate"])

    if u_col is None or v_col is None:
        raise ValueError(f"Visibility table must contain u and v columns: {path}")

    u = pd.to_numeric(df[u_col], errors="coerce").to_numpy(dtype=float)
    v = pd.to_numeric(df[v_col], errors="coerce").to_numpy(dtype=float)

    vis_col = find_column(df.columns, ["vis", "visibility", "complex_vis", "complex_visibility"])
    real_col = find_column(df.columns, ["vis_real", "real", "re", "vis_re", "r"])
    imag_col = find_column(df.columns, ["vis_imag", "imag", "im", "vis_im", "i"])
    amp_col = find_column(df.columns, ["amp", "amplitude", "vis_amp", "absvis", "visamp"])
    phase_col = find_column(df.columns, ["phase", "phi", "vis_phase", "phase_rad", "phase_deg"])

    if vis_col is not None:
        vis = parse_complex_series(df[vis_col].to_numpy())
    elif real_col is not None and imag_col is not None:
        real = pd.to_numeric(df[real_col], errors="coerce").to_numpy(dtype=float)
        imag = pd.to_numeric(df[imag_col], errors="coerce").to_numpy(dtype=float)
        vis = real + 1j * imag
    elif amp_col is not None and phase_col is not None:
        amp = pd.to_numeric(df[amp_col], errors="coerce").to_numpy(dtype=float)
        phase_raw = pd.to_numeric(df[phase_col], errors="coerce").to_numpy(dtype=float)
        phase = phase_to_radians(phase_raw, phase_unit=phase_unit)
        vis = amp * np.exp(1j * phase)
    else:
        raise ValueError(
            f"Visibility table must contain vis, real+imag, or amp+phase columns: {path}"
        )

    sigma_col = find_column(df.columns, ["sigma", "std", "uncertainty", "error", "noise", "sigma_vis", "vis_sigma"])
    sigma = None
    if sigma_col is not None:
        sigma = pd.to_numeric(df[sigma_col], errors="coerce").to_numpy(dtype=float)

    return clean_visibility_data(VisibilityData(u=u, v=v, vis=vis, sigma=sigma, source_path=str(path), frame_index=frame_index))


def load_uv_from_uvfits(path):
    """Best-effort UVFITS loader using ehtim, when available."""
    if eh is None:
        raise ImportError("ehtim is not installed; cannot load UVFITS files with this script.")

    if not hasattr(eh, "obsdata") or not hasattr(eh.obsdata, "load_uvfits"):
        raise ImportError("This ehtim installation does not expose eh.obsdata.load_uvfits.")

    obs = eh.obsdata.load_uvfits(str(path))
    data = obs.data

    def field(name_candidates):
        names = getattr(data, "dtype", None).names if hasattr(data, "dtype") and data.dtype.names else []
        return find_column(names, name_candidates)

    u_name = field(["u", "uu"])
    v_name = field(["v", "vv"])
    vis_name = field(["vis", "visibility"])
    sigma_name = field(["sigma", "sig", "sigma_vis"])

    if u_name is None or v_name is None or vis_name is None:
        raise ValueError(f"Could not find u, v, and vis fields in UVFITS data: {path}")

    u = np.asarray(data[u_name], dtype=float).ravel()
    v = np.asarray(data[v_name], dtype=float).ravel()
    vis = np.asarray(data[vis_name], dtype=complex).ravel()
    sigma = np.asarray(data[sigma_name], dtype=float).ravel() if sigma_name is not None else None

    return clean_visibility_data(VisibilityData(u=u, v=v, vis=vis, sigma=sigma, source_path=str(path), frame_index=None))


def clean_visibility_data(obs):
    """Remove rows with non-finite u/v/visibility/sigma values."""
    u = np.asarray(obs.u, dtype=float).ravel()
    v = np.asarray(obs.v, dtype=float).ravel()
    vis = np.asarray(obs.vis, dtype=complex).ravel()

    n = min(len(u), len(v), len(vis))
    u = u[:n]
    v = v[:n]
    vis = vis[:n]

    sigma = None
    if obs.sigma is not None:
        sigma = np.asarray(obs.sigma, dtype=float).ravel()[:n]

    valid = np.isfinite(u) & np.isfinite(v) & np.isfinite(vis.real) & np.isfinite(vis.imag)
    if sigma is not None:
        valid = valid & np.isfinite(sigma) & (sigma > 0)

    u = u[valid]
    v = v[valid]
    vis = vis[valid]
    if sigma is not None:
        sigma = sigma[valid]

    return VisibilityData(
        u=u,
        v=v,
        vis=vis,
        sigma=sigma,
        source_path=obs.source_path,
        frame_index=obs.frame_index,
    )


def load_visibility_data(path, frame_index=None, frame_column="frame_index", phase_unit="auto"):
    """Load visibility data from CSV/TXT/DAT, NPY, NPZ, or UVFITS."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".npz":
        return load_uv_from_npz(path, frame_index=frame_index, frame_column=frame_column, phase_unit=phase_unit)
    if suffix == ".npy":
        return load_uv_from_npy(path, frame_index=frame_index)
    if suffix == ".uvfits":
        return load_uv_from_uvfits(path)
    if suffix in {".csv", ".txt", ".dat"}:
        return load_uv_from_table(path, frame_index=frame_index, frame_column=frame_column, phase_unit=phase_unit)

    raise ValueError(f"Unsupported visibility file type: {path}")


def match_visibility_data(true_obs, recon_obs, match_mode="row", nearest_tolerance=None):
    """Match two visibility sets by row order or nearest u-v coordinate."""
    mode = str(match_mode).lower()

    if mode == "row":
        n = min(len(true_obs.u), len(recon_obs.u))
        if n == 0:
            raise ValueError("No visibility samples to match.")

        return MatchedVisibilityData(
            u=true_obs.u[:n],
            v=true_obs.v[:n],
            vis_true=true_obs.vis[:n],
            vis_recon=recon_obs.vis[:n],
            sigma_true=true_obs.sigma[:n] if true_obs.sigma is not None else None,
            sigma_recon=recon_obs.sigma[:n] if recon_obs.sigma is not None else None,
            matched_count=n,
            match_mode="row",
        )

    if mode == "nearest":
        if len(true_obs.u) == 0 or len(recon_obs.u) == 0:
            raise ValueError("No visibility samples to match.")

        true_points = np.column_stack([true_obs.u, true_obs.v])
        recon_points = np.column_stack([recon_obs.u, recon_obs.v])

        if cKDTree is not None:
            tree = cKDTree(recon_points)
            distances, indices = tree.query(true_points, k=1)
        else:
            distances = []
            indices = []
            for p in true_points:
                d2 = np.sum((recon_points - p) ** 2, axis=1)
                idx = int(np.argmin(d2))
                distances.append(float(np.sqrt(d2[idx])))
                indices.append(idx)
            distances = np.asarray(distances, dtype=float)
            indices = np.asarray(indices, dtype=int)

        keep = np.ones(len(indices), dtype=bool)
        if nearest_tolerance is not None:
            keep = distances <= float(nearest_tolerance)

        if np.count_nonzero(keep) == 0:
            raise ValueError("Nearest-neighbor u-v matching found no samples within tolerance.")

        idx_true = np.where(keep)[0]
        idx_recon = indices[keep]

        return MatchedVisibilityData(
            u=true_obs.u[idx_true],
            v=true_obs.v[idx_true],
            vis_true=true_obs.vis[idx_true],
            vis_recon=recon_obs.vis[idx_recon],
            sigma_true=true_obs.sigma[idx_true] if true_obs.sigma is not None else None,
            sigma_recon=recon_obs.sigma[idx_recon] if recon_obs.sigma is not None else None,
            matched_count=int(np.count_nonzero(keep)),
            match_mode="nearest",
        )

    raise ValueError(f"Unknown u-v match mode: {match_mode}")


def choose_sigma_for_chi2(matched, sigma_mode="ground_truth", fixed_sigma=1.0):
    """Select sigma array for visibility-domain chi-squared."""
    mode = str(sigma_mode).lower()
    n = matched.matched_count

    def valid_sigma(sigma):
        if sigma is None:
            return False
        sigma = np.asarray(sigma, dtype=float)
        return sigma.size == n and np.any(np.isfinite(sigma) & (sigma > 0))

    if mode == "ground_truth" and valid_sigma(matched.sigma_true):
        return np.asarray(matched.sigma_true, dtype=float), "ground_truth"

    if mode == "reconstruction" and valid_sigma(matched.sigma_recon):
        return np.asarray(matched.sigma_recon, dtype=float), "reconstruction"

    if mode == "average" and valid_sigma(matched.sigma_true) and valid_sigma(matched.sigma_recon):
        return 0.5 * (np.asarray(matched.sigma_true, dtype=float) + np.asarray(matched.sigma_recon, dtype=float)), "average"

    if mode == "fixed":
        return np.full(n, float(fixed_sigma), dtype=float), "fixed"

    # Fallback order if requested sigma is missing.
    if valid_sigma(matched.sigma_true):
        return np.asarray(matched.sigma_true, dtype=float), "ground_truth_fallback"
    if valid_sigma(matched.sigma_recon):
        return np.asarray(matched.sigma_recon, dtype=float), "reconstruction_fallback"

    return np.full(n, float(fixed_sigma), dtype=float), "fixed_fallback"


def compute_uv_chi_squared(matched, sigma_mode="ground_truth", fixed_sigma=1.0):
    """Visibility-domain chi-squared: mean(|V_recon - V_true|^2 / sigma^2)."""
    if matched.matched_count <= 0:
        return np.nan, "none"

    sigma, sigma_used = choose_sigma_for_chi2(
        matched,
        sigma_mode=sigma_mode,
        fixed_sigma=fixed_sigma,
    )

    sigma = np.asarray(sigma, dtype=float)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, float(fixed_sigma))

    residual_sq = np.abs(matched.vis_recon - matched.vis_true) ** 2
    chi_values = residual_sq / (sigma ** 2 + EPS)

    return float(np.nanmean(chi_values)), sigma_used


def compute_frc_curve_uv(matched, num_bins=None, min_samples=2):
    """Compute FRC-like correlation from sampled u-v visibilities in radial bins."""
    if matched.matched_count <= 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    radius = np.sqrt(matched.u ** 2 + matched.v ** 2)
    valid = (
        np.isfinite(radius)
        & np.isfinite(matched.vis_true.real)
        & np.isfinite(matched.vis_true.imag)
        & np.isfinite(matched.vis_recon.real)
        & np.isfinite(matched.vis_recon.imag)
    )

    radius = radius[valid]
    vis_true = matched.vis_true[valid]
    vis_recon = matched.vis_recon[valid]

    if radius.size == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    max_radius = float(np.nanmax(radius))
    if max_radius <= 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    if num_bins is None:
        num_bins = max(8, min(64, int(np.sqrt(radius.size))))

    num_bins = int(max(1, num_bins))
    edges = np.linspace(0.0, max_radius, num_bins + 1)

    freq_mid_norm = []
    radius_mid = []
    frc_values = []
    sample_counts = []

    for i in range(num_bins):
        low = edges[i]
        high = edges[i + 1]

        if i == num_bins - 1:
            mask = (radius >= low) & (radius <= high)
        else:
            mask = (radius >= low) & (radius < high)

        count = int(np.count_nonzero(mask))
        sample_counts.append(count)

        if count < int(min_samples):
            frc = np.nan
        else:
            a = vis_true[mask]
            b = vis_recon[mask]
            numerator = np.sum(a * np.conj(b))
            denominator = np.sqrt(np.sum(np.abs(a) ** 2) * np.sum(np.abs(b) ** 2))

            if denominator <= EPS:
                frc = np.nan
            else:
                frc = float(np.clip(np.real(numerator) / denominator, -1.0, 1.0))

        mid = 0.5 * (low + high)
        radius_mid.append(mid)
        freq_mid_norm.append(mid / max_radius)
        frc_values.append(frc)

    return (
        np.asarray(freq_mid_norm, dtype=float),
        np.asarray(frc_values, dtype=float),
        np.asarray(sample_counts, dtype=int),
        np.asarray(radius_mid, dtype=float),
    )


def make_visibility_pair_loader(uv_config, n_frames, verbose=True):
    """
    Return a loader function for matched visibility data.

    The returned function get_pair(i) returns:
        true_obs, recon_obs, matched
    or
        None, None, None
    if visibility data are disabled/unavailable.
    """
    uv_config = uv_config or {}
    if not uv_config.get("enabled", False):
        return lambda frame_index: (None, None, None), False

    gt_dir = uv_config.get("ground_truth_uv_dir")
    recon_dir = uv_config.get("reconstruction_uv_dir")
    gt_file = uv_config.get("ground_truth_uv_file")
    recon_file = uv_config.get("reconstruction_uv_file")

    use_dirs = bool(gt_dir) and bool(recon_dir)
    use_files = bool(gt_file) and bool(recon_file)

    if not use_dirs and not use_files:
        has_observation_uv = bool(uv_config.get("observation_uv_dir")) or bool(uv_config.get("observation_uv_file"))
        if verbose and not has_observation_uv:
            print("WARNING: uv_data.enabled=True, but no matched ground-truth/reconstruction u-v folders or files were provided. Using image-based Fourier metrics.")
        return lambda frame_index: (None, None, None), False

    gt_uv_files = None
    recon_uv_files = None
    if use_dirs:
        gt_uv_files = list_uv_files(gt_dir)
        recon_uv_files = list_uv_files(recon_dir)

        if len(gt_uv_files) != len(recon_uv_files) and verbose:
            print("WARNING: u-v folder file counts do not match.")
            print(f"Ground truth u-v count:   {len(gt_uv_files)}")
            print(f"Reconstruction u-v count: {len(recon_uv_files)}")
            print("Using the overlapping number of u-v files.")

    cache = {}

    def load_cached(path, frame_index_for_file):
        key = (str(path), frame_index_for_file)
        if key not in cache:
            cache[key] = load_visibility_data(
                path,
                frame_index=frame_index_for_file,
                frame_column=uv_config.get("frame_column", "frame_index"),
                phase_unit=uv_config.get("phase_unit", "auto"),
            )
        return cache[key]

    def get_pair(frame_index):
        try:
            if use_dirs:
                if frame_index >= len(gt_uv_files) or frame_index >= len(recon_uv_files):
                    return None, None, None
                gt_path = gt_uv_files[frame_index]
                recon_path = recon_uv_files[frame_index]
                true_obs = load_cached(gt_path, None)
                recon_obs = load_cached(recon_path, None)
            else:
                true_obs = load_cached(gt_file, frame_index)
                recon_obs = load_cached(recon_file, frame_index)

            matched = match_visibility_data(
                true_obs,
                recon_obs,
                match_mode=uv_config.get("match_mode", "row"),
                nearest_tolerance=uv_config.get("nearest_tolerance", None),
            )
            return true_obs, recon_obs, matched
        except Exception as exc:
            if verbose:
                print(f"WARNING: Could not load/match u-v data for frame {frame_index}: {exc}")
            return None, None, None

    return get_pair, True



# ============================================================
# Observation u-v sampling and image-to-visibility helpers
# ============================================================

def is_uvfits_like_path(path):
    """Return True for UVFITS/FITS-like filenames, including .gz variants."""
    lower = Path(path).name.lower()
    return lower.endswith((".uvfits", ".uvfits.gz", ".fits", ".fits.gz"))


def resolve_observation_uv_unit(path, requested_unit="auto"):
    """Resolve the requested observation u-v coordinate unit."""
    unit = str(requested_unit or "auto").lower()
    if unit == "auto":
        return "seconds" if is_uvfits_like_path(path) else "lambda"
    return unit


def convert_uv_coordinates(u, v, coordinate_unit="lambda", observing_frequency_hz=226e9):
    """Convert u/v coordinates to wavelengths."""
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    unit = str(coordinate_unit or "lambda").lower()

    if unit in {"lambda", "lambdas", "wavelength", "wavelengths", "wl"}:
        scale = 1.0
    elif unit in {"klambda", "kilolambda", "kilolambdas", "kλ"}:
        scale = 1e3
    elif unit in {"mlambda", "megalambda", "megalambdas", "mλ"}:
        scale = 1e6
    elif unit in {"glambda", "gigalambda", "gigalambdas", "gλ"}:
        scale = 1e9
    elif unit in {"seconds", "second", "sec", "s"}:
        scale = float(observing_frequency_hz)
    else:
        raise ValueError(
            f"Unknown observation u-v coordinate unit: {coordinate_unit}. "
            "Use auto, seconds, lambda, klambda, mlambda, or glambda."
        )

    return u * scale, v * scale


def clean_uv_sampling_data(sampling):
    """Remove rows with non-finite u/v values from a UVSamplingData object."""
    u = np.asarray(sampling.u, dtype=float).ravel()
    v = np.asarray(sampling.v, dtype=float).ravel()
    n = min(len(u), len(v))
    u = u[:n]
    v = v[:n]
    valid = np.isfinite(u) & np.isfinite(v)

    return UVSamplingData(
        u=u[valid],
        v=v[valid],
        source_path=sampling.source_path,
        frame_index=sampling.frame_index,
        coordinate_unit=sampling.coordinate_unit,
    )


def load_uv_sampling_from_table(path, frame_index=None, frame_column="frame_index"):
    """Load u/v coordinates from a CSV/TXT/DAT table."""
    df = read_table_auto(path)
    df = select_frame_rows(df, frame_index=frame_index, frame_column=frame_column)

    if len(df) == 0:
        raise ValueError(f"No rows found for frame {frame_index} in {path}")

    u_col = find_column(df.columns, ["u", "uu", "u_lambda", "u_lambdas", "ucoord", "u_coordinate"])
    v_col = find_column(df.columns, ["v", "vv", "v_lambda", "v_lambdas", "vcoord", "v_coordinate"])

    if u_col is None or v_col is None:
        raise ValueError(f"Observation u-v table must contain u and v columns: {path}")

    u = pd.to_numeric(df[u_col], errors="coerce").to_numpy(dtype=float)
    v = pd.to_numeric(df[v_col], errors="coerce").to_numpy(dtype=float)

    return clean_uv_sampling_data(
        UVSamplingData(u=u, v=v, source_path=str(path), frame_index=frame_index)
    )


def load_uv_sampling_from_npz(path, frame_index=None):
    """Load u/v coordinates from an NPZ file."""
    data = np.load(path, allow_pickle=True)
    u_key = first_available_key(data, ["u", "uu", "u_lambda", "u_lambdas", "ucoord"])
    v_key = first_available_key(data, ["v", "vv", "v_lambda", "v_lambdas", "vcoord"])

    if u_key is None or v_key is None:
        raise ValueError(f"NPZ observation u-v file must contain u and v arrays: {path}")

    u = maybe_select_frame_array(data[u_key], frame_index).astype(float).ravel()
    v = maybe_select_frame_array(data[v_key], frame_index).astype(float).ravel()

    return clean_uv_sampling_data(
        UVSamplingData(u=u, v=v, source_path=str(path), frame_index=frame_index)
    )


def load_uv_sampling_from_npy(path, frame_index=None):
    """Load u/v coordinates from an NPY array with columns u, v, ... ."""
    arr = np.load(path, allow_pickle=True)
    arr = np.asarray(arr)

    if frame_index is not None and arr.ndim == 3 and arr.shape[0] > int(frame_index):
        arr = arr[int(frame_index)]

    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"NPY observation u-v file should have shape (N, 2+) or (T, N, 2+): {path}")

    return clean_uv_sampling_data(
        UVSamplingData(
            u=arr[:, 0].astype(float),
            v=arr[:, 1].astype(float),
            source_path=str(path),
            frame_index=frame_index,
        )
    )


def _fits_random_group_parnames(data):
    if hasattr(data, "parnames"):
        return list(data.parnames)
    return []


def _find_uv_fits_name(names, kind):
    """Find likely u or v names in a FITS random-group/table data object."""
    names = list(names or [])
    normalized_to_original = {canonical_column_name(name): name for name in names}

    if kind == "u":
        exact = ["UU", "U", "UU---SIN", "UU---NCP", "UU---TAN"]
        prefixes = ["uu", "ucoord", "ucoordinate"]
    else:
        exact = ["VV", "V", "VV---SIN", "VV---NCP", "VV---TAN"]
        prefixes = ["vv", "vcoord", "vcoordinate"]

    for candidate in exact:
        key = canonical_column_name(candidate)
        if key in normalized_to_original:
            return normalized_to_original[key]

    for name in names:
        key = canonical_column_name(name)
        if any(key.startswith(prefix) for prefix in prefixes):
            return name

    return None


def load_uv_sampling_from_uvfits_astropy(path):
    """Best-effort extraction of u/v coordinates from UVFITS using astropy."""
    if fits is None:
        raise ImportError("astropy is not installed; cannot read UVFITS u-v coordinates without ehtim.")

    with fits.open(path, memmap=True) as hdul:
        for hdu in hdul:
            data = getattr(hdu, "data", None)
            if data is None:
                continue

            # UVFITS random groups: u/v are often random parameters named UU/VV.
            parnames = _fits_random_group_parnames(data)
            if parnames and hasattr(data, "par"):
                u_name = _find_uv_fits_name(parnames, "u")
                v_name = _find_uv_fits_name(parnames, "v")
                if u_name is not None and v_name is not None:
                    u = np.asarray(data.par(u_name), dtype=float).ravel()
                    v = np.asarray(data.par(v_name), dtype=float).ravel()
                    return clean_uv_sampling_data(UVSamplingData(u=u, v=v, source_path=str(path)))

            # Table fallback: u/v are columns.
            names = getattr(data, "names", None)
            if names is None and hasattr(data, "dtype") and data.dtype.names is not None:
                names = data.dtype.names
            if names is None:
                continue

            u_name = _find_uv_fits_name(names, "u")
            v_name = _find_uv_fits_name(names, "v")
            if u_name is not None and v_name is not None:
                if hasattr(data, "field"):
                    u = np.asarray(data.field(u_name), dtype=float).ravel()
                    v = np.asarray(data.field(v_name), dtype=float).ravel()
                else:
                    u = np.asarray(data[u_name], dtype=float).ravel()
                    v = np.asarray(data[v_name], dtype=float).ravel()
                return clean_uv_sampling_data(UVSamplingData(u=u, v=v, source_path=str(path)))

    raise ValueError(f"Could not find UU/VV or u/v coordinates in UVFITS file: {path}")


def load_uv_sampling_from_uvfits(path):
    """Load u/v coordinates from UVFITS using astropy first, then ehtim fallback."""
    errors = []

    if fits is not None:
        try:
            return load_uv_sampling_from_uvfits_astropy(path)
        except Exception as exc:
            errors.append(f"astropy: {exc}")

    if eh is not None and hasattr(eh, "obsdata") and hasattr(eh.obsdata, "load_uvfits"):
        try:
            obs = eh.obsdata.load_uvfits(str(path))
            data = obs.data
            names = getattr(data, "dtype", None).names if hasattr(data, "dtype") and data.dtype.names else []
            u_name = find_column(names, ["u", "uu"])
            v_name = find_column(names, ["v", "vv"])
            if u_name is None or v_name is None:
                raise ValueError("could not find u and v fields in ehtim obsdata")
            return clean_uv_sampling_data(
                UVSamplingData(
                    u=np.asarray(data[u_name], dtype=float).ravel(),
                    v=np.asarray(data[v_name], dtype=float).ravel(),
                    source_path=str(path),
                )
            )
        except Exception as exc:
            errors.append(f"ehtim: {exc}")

    detail = "; ".join(errors) if errors else "install astropy or ehtim to read UVFITS files"
    raise ImportError(f"Could not read UVFITS u-v coordinates from {path}: {detail}")


def load_uv_sampling_data(
    path,
    frame_index=None,
    frame_column="frame_index",
    coordinate_unit="auto",
    observing_frequency_hz=226e9,
):
    """Load u/v coordinates from CSV/TXT/DAT, NPY, NPZ, FITS, or UVFITS."""
    path = Path(path)
    lower_name = path.name.lower()
    suffix = path.suffix.lower()

    if suffix == ".npz":
        sampling = load_uv_sampling_from_npz(path, frame_index=frame_index)
    elif suffix == ".npy":
        sampling = load_uv_sampling_from_npy(path, frame_index=frame_index)
    elif is_uvfits_like_path(path):
        sampling = load_uv_sampling_from_uvfits(path)
    elif suffix in {".csv", ".txt", ".dat"}:
        sampling = load_uv_sampling_from_table(path, frame_index=frame_index, frame_column=frame_column)
    else:
        raise ValueError(f"Unsupported observation u-v file type: {path}")

    resolved_unit = resolve_observation_uv_unit(path, coordinate_unit)
    u_lambda, v_lambda = convert_uv_coordinates(
        sampling.u,
        sampling.v,
        coordinate_unit=resolved_unit,
        observing_frequency_hz=observing_frequency_hz,
    )

    return clean_uv_sampling_data(
        UVSamplingData(
            u=u_lambda,
            v=v_lambda,
            source_path=sampling.source_path,
            frame_index=frame_index,
            coordinate_unit=resolved_unit,
        )
    )


def make_uv_sampling_loader(uv_config, n_frames, verbose=True):
    """
    Return a loader for observation u-v samples.

    The returned function get_sampling(i) returns a UVSamplingData object or None.
    """
    uv_config = uv_config or {}
    if not uv_config.get("enabled", False):
        return lambda frame_index: None, False

    obs_dir = uv_config.get("observation_uv_dir")
    obs_file = uv_config.get("observation_uv_file")
    use_dir = bool(obs_dir)
    use_file = bool(obs_file)

    if not use_dir and not use_file:
        return lambda frame_index: None, False

    obs_files = None
    if use_dir:
        obs_files = list_uv_files(obs_dir)
        if len(obs_files) < n_frames and verbose:
            print("WARNING: observation u-v file count is smaller than image frame count.")
            print(f"Observation u-v count: {len(obs_files)}")
            print(f"Image frame count:     {n_frames}")

    cache = {}

    def load_cached(path, frame_index_for_file):
        key = (str(path), frame_index_for_file)
        if key not in cache:
            cache[key] = load_uv_sampling_data(
                path,
                frame_index=frame_index_for_file,
                frame_column=uv_config.get("observation_uv_frame_column", uv_config.get("frame_column", "frame_index")),
                coordinate_unit=uv_config.get("observation_uv_coordinate_unit", "auto"),
                observing_frequency_hz=float(uv_config.get("observing_frequency_hz", 226e9)),
            )
        return cache[key]

    def get_sampling(frame_index):
        try:
            if use_dir:
                if frame_index >= len(obs_files):
                    return None
                return load_cached(obs_files[frame_index], None)
            return load_cached(obs_file, frame_index)
        except Exception as exc:
            if verbose:
                print(f"WARNING: Could not load observation u-v samples for frame {frame_index}: {exc}")
            return None

    return get_sampling, True


def image_to_visibility_samples(
    arr,
    u,
    v,
    fov_uas=160.0,
    interpolation="linear",
    out_of_bounds="drop",
):
    """
    Sample the image Fourier transform at u/v coordinates in wavelengths.

    This is a deterministic image-domain forward model used for evaluation. It is
    not a replacement for a full calibrated interferometric measurement model,
    but it is appropriate for comparing two image sequences at the same sampled
    u-v coordinates.
    """
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 2:
        raise ValueError("Expected a 2D image array for visibility sampling.")

    h, w = arr.shape
    pixel_size_rad_x = float(fov_uas) * RAD_PER_UAS / float(w)
    pixel_size_rad_y = float(fov_uas) * RAD_PER_UAS / float(h)

    # Center the image before FFT so the image center corresponds to phase center.
    fft_image = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(arr)))
    u_grid = np.fft.fftshift(np.fft.fftfreq(w, d=pixel_size_rad_x))
    v_grid = np.fft.fftshift(np.fft.fftfreq(h, d=pixel_size_rad_y))

    u = np.asarray(u, dtype=float).ravel()
    v = np.asarray(v, dtype=float).ravel()
    points = np.column_stack([v, u])

    interpolation = str(interpolation or "linear").lower()
    out_of_bounds = str(out_of_bounds or "drop").lower()
    fill = np.nan if out_of_bounds == "drop" else 0.0

    if interpolation == "linear" and RegularGridInterpolator is not None:
        real_interp = RegularGridInterpolator(
            (v_grid, u_grid),
            np.real(fft_image),
            bounds_error=False,
            fill_value=fill,
        )
        imag_interp = RegularGridInterpolator(
            (v_grid, u_grid),
            np.imag(fft_image),
            bounds_error=False,
            fill_value=fill,
        )
        vis = real_interp(points) + 1j * imag_interp(points)
    else:
        # Nearest-neighbor fallback. Out-of-bounds points are dropped or set to zero.
        u_idx = np.searchsorted(u_grid, u)
        v_idx = np.searchsorted(v_grid, v)
        u_idx = np.clip(u_idx, 1, len(u_grid) - 1)
        v_idx = np.clip(v_idx, 1, len(v_grid) - 1)
        u_left = u_grid[u_idx - 1]
        u_right = u_grid[u_idx]
        v_left = v_grid[v_idx - 1]
        v_right = v_grid[v_idx]
        u_idx = np.where(np.abs(u - u_left) <= np.abs(u - u_right), u_idx - 1, u_idx)
        v_idx = np.where(np.abs(v - v_left) <= np.abs(v - v_right), v_idx - 1, v_idx)

        inside = (u >= np.nanmin(u_grid)) & (u <= np.nanmax(u_grid)) & (v >= np.nanmin(v_grid)) & (v <= np.nanmax(v_grid))
        vis = fft_image[v_idx, u_idx].astype(complex)
        if out_of_bounds == "drop":
            vis = np.where(inside, vis, np.nan + 1j * np.nan)
        else:
            vis = np.where(inside, vis, 0.0 + 0.0j)

    return np.asarray(vis, dtype=complex)


def make_image_sampled_visibility_pair(gt_arr, recon_arr, sampling, cfg, uv_config):
    """Compute predicted ground-truth and reconstruction visibilities at observation u-v samples."""
    if sampling is None or len(sampling.u) == 0:
        return None, None, None

    gt_vis = image_to_visibility_samples(
        gt_arr,
        sampling.u,
        sampling.v,
        fov_uas=float(cfg.get("fov_uas", 160.0)),
        interpolation=uv_config.get("image_sampling_interpolation", "linear"),
        out_of_bounds=uv_config.get("image_sampling_out_of_bounds", "drop"),
    )
    recon_vis = image_to_visibility_samples(
        recon_arr,
        sampling.u,
        sampling.v,
        fov_uas=float(cfg.get("fov_uas", 160.0)),
        interpolation=uv_config.get("image_sampling_interpolation", "linear"),
        out_of_bounds=uv_config.get("image_sampling_out_of_bounds", "drop"),
    )

    valid = (
        np.isfinite(sampling.u)
        & np.isfinite(sampling.v)
        & np.isfinite(gt_vis.real)
        & np.isfinite(gt_vis.imag)
        & np.isfinite(recon_vis.real)
        & np.isfinite(recon_vis.imag)
    )

    if np.count_nonzero(valid) == 0:
        return None, None, None

    true_obs = VisibilityData(
        u=sampling.u[valid],
        v=sampling.v[valid],
        vis=gt_vis[valid],
        sigma=None,
        source_path=sampling.source_path,
        frame_index=sampling.frame_index,
    )
    recon_obs = VisibilityData(
        u=sampling.u[valid],
        v=sampling.v[valid],
        vis=recon_vis[valid],
        sigma=None,
        source_path=sampling.source_path,
        frame_index=sampling.frame_index,
    )
    matched = match_visibility_data(true_obs, recon_obs, match_mode="row")
    matched.match_mode = "observation_uv_image_sampling"
    return true_obs, recon_obs, matched


# ============================================================
# Spatiotemporal metrics
# ============================================================

def _component_error(true_component, recon_component, norm="l2", epsilon=1e-12):
    """Normalized error between two gradient components."""
    true_component = np.asarray(true_component, dtype=float)
    recon_component = np.asarray(recon_component, dtype=float)
    valid = np.isfinite(true_component) & np.isfinite(recon_component)
    if np.count_nonzero(valid) == 0:
        return np.nan

    diff = recon_component[valid] - true_component[valid]
    truth = true_component[valid]
    norm = str(norm or "l2").lower()

    if norm == "l1":
        denom = np.sum(np.abs(truth))
        if denom <= epsilon:
            return np.nan
        return float(np.sum(np.abs(diff)) / (denom + epsilon))

    if norm == "l2":
        denom = np.linalg.norm(truth.ravel())
        if denom <= epsilon:
            return np.nan
        return float(np.linalg.norm(diff.ravel()) / (denom + epsilon))

    raise ValueError(f"Unknown gradient norm: {norm}")


def _combined_gradient_error(components, weights, norm="l2", epsilon=1e-12):
    """Weighted normalized error over multiple gradient components."""
    norm = str(norm or "l2").lower()

    if norm == "l1":
        num = 0.0
        den = 0.0
        for true_component, recon_component, weight in zip(components[0], components[1], weights):
            true_component = np.asarray(true_component, dtype=float)
            recon_component = np.asarray(recon_component, dtype=float)
            valid = np.isfinite(true_component) & np.isfinite(recon_component)
            if np.count_nonzero(valid) == 0:
                continue
            num += float(weight) * np.sum(np.abs(recon_component[valid] - true_component[valid]))
            den += float(weight) * np.sum(np.abs(true_component[valid]))
        return np.nan if den <= epsilon else float(num / (den + epsilon))

    if norm == "l2":
        num = 0.0
        den = 0.0
        for true_component, recon_component, weight in zip(components[0], components[1], weights):
            true_component = np.asarray(true_component, dtype=float)
            recon_component = np.asarray(recon_component, dtype=float)
            valid = np.isfinite(true_component) & np.isfinite(recon_component)
            if np.count_nonzero(valid) == 0:
                continue
            diff = recon_component[valid] - true_component[valid]
            truth = true_component[valid]
            num += float(weight) * np.sum(diff ** 2)
            den += float(weight) * np.sum(truth ** 2)
        return np.nan if den <= epsilon else float(np.sqrt(num) / (np.sqrt(den) + epsilon))

    raise ValueError(f"Unknown gradient norm: {norm}")


def compute_spatiotemporal_gradient_error(
    original_stack,
    recon_stack,
    normalization="minmax",
    total_flux=1.0,
    norm="l2",
    spatial_weight=1.0,
    temporal_weight=1.0,
    use_absolute_gradients=False,
    epsilon=1e-12,
):
    """
    Compare spatial and temporal derivatives of two image sequences.

    Spatial gradients detect whether ring/shadow edges are preserved. Temporal
    gradients detect whether frame-to-frame changes are preserved.
    """
    original_stack = np.asarray(original_stack, dtype=float)
    recon_stack = np.asarray(recon_stack, dtype=float)

    if original_stack.shape != recon_stack.shape:
        raise ValueError(
            f"Temporal stacks must have the same shape. Got {original_stack.shape} and {recon_stack.shape}."
        )
    if original_stack.ndim != 3:
        raise ValueError("Expected stacks with shape (frames, height, width).")

    original_norm = np.stack([
        normalize_array(frame, mode=normalization, total_flux=total_flux)
        for frame in original_stack
    ], axis=0)
    recon_norm = np.stack([
        normalize_array(frame, mode=normalization, total_flux=total_flux)
        for frame in recon_stack
    ], axis=0)

    dx_true = np.diff(original_norm, axis=2)
    dx_recon = np.diff(recon_norm, axis=2)
    dy_true = np.diff(original_norm, axis=1)
    dy_recon = np.diff(recon_norm, axis=1)

    if original_norm.shape[0] >= 2:
        dt_true = np.diff(original_norm, axis=0)
        dt_recon = np.diff(recon_norm, axis=0)
    else:
        dt_true = np.empty((0,) + original_norm.shape[1:], dtype=float)
        dt_recon = np.empty((0,) + recon_norm.shape[1:], dtype=float)

    if use_absolute_gradients:
        dx_true, dx_recon = np.abs(dx_true), np.abs(dx_recon)
        dy_true, dy_recon = np.abs(dy_true), np.abs(dy_recon)
        dt_true, dt_recon = np.abs(dt_true), np.abs(dt_recon)

    spatial_x_error = _component_error(dx_true, dx_recon, norm=norm, epsilon=epsilon)
    spatial_y_error = _component_error(dy_true, dy_recon, norm=norm, epsilon=epsilon)
    temporal_error = _component_error(dt_true, dt_recon, norm=norm, epsilon=epsilon) if dt_true.size else np.nan

    spatial_error = _combined_gradient_error(
        components=([dx_true, dy_true], [dx_recon, dy_recon]),
        weights=[spatial_weight, spatial_weight],
        norm=norm,
        epsilon=epsilon,
    )

    if dt_true.size:
        combined_error = _combined_gradient_error(
            components=([dx_true, dy_true, dt_true], [dx_recon, dy_recon, dt_recon]),
            weights=[spatial_weight, spatial_weight, temporal_weight],
            norm=norm,
            epsilon=epsilon,
        )
    else:
        combined_error = spatial_error

    # Diagnostic maps: average magnitude of gradient mismatch over time.
    h = original_norm.shape[1]
    w = original_norm.shape[2]
    spatial_error_map = np.zeros((h, w), dtype=float)
    dx_pad = np.zeros_like(original_norm)
    dy_pad = np.zeros_like(original_norm)
    dx_pad[:, :, :-1] = dx_recon - dx_true
    dy_pad[:, :-1, :] = dy_recon - dy_true
    spatial_error_map = np.nanmean(np.sqrt(dx_pad ** 2 + dy_pad ** 2), axis=0)

    if dt_true.size:
        temporal_error_map = np.nanmean(np.abs(dt_recon - dt_true), axis=0)
    else:
        temporal_error_map = np.zeros((h, w), dtype=float)

    return {
        "spatiotemporal_gradient_error": combined_error,
        "spatial_gradient_error": spatial_error,
        "spatial_gradient_x_error": spatial_x_error,
        "spatial_gradient_y_error": spatial_y_error,
        "temporal_gradient_error": temporal_error,
        "spatiotemporal_gradient_norm": norm,
        "spatiotemporal_gradient_spatial_weight": spatial_weight,
        "spatiotemporal_gradient_temporal_weight": temporal_weight,
        "spatiotemporal_gradient_abs_gradients": bool(use_absolute_gradients),
    }, spatial_error_map, temporal_error_map


def _visibility_series_values(vis, amplitude_mode="amplitude", epsilon=1e-12):
    """Convert complex visibilities into values used for variability analysis."""
    amp = np.abs(np.asarray(vis, dtype=complex))
    mode = str(amplitude_mode or "amplitude").lower()

    if mode == "amplitude":
        return amp
    if mode in {"log_amplitude", "logamp", "log_amp"}:
        return np.log(np.maximum(amp, float(epsilon)))

    raise ValueError(f"Unknown visibility amplitude_mode: {amplitude_mode}")


def _normalized_vector_error(true_values, recon_values, norm="l2", epsilon=1e-12):
    true_values = np.asarray(true_values, dtype=float)
    recon_values = np.asarray(recon_values, dtype=float)
    valid = np.isfinite(true_values) & np.isfinite(recon_values)
    if np.count_nonzero(valid) == 0:
        return np.nan

    diff = recon_values[valid] - true_values[valid]
    truth = true_values[valid]
    norm = str(norm or "l2").lower()

    if norm == "l1":
        denom = np.sum(np.abs(truth))
        if denom <= epsilon:
            return np.nan
        return float(np.sum(np.abs(diff)) / (denom + epsilon))
    if norm == "l2":
        denom = np.linalg.norm(truth)
        if denom <= epsilon:
            return np.nan
        return float(np.linalg.norm(diff) / (denom + epsilon))

    raise ValueError(f"Unknown visibility variability norm: {norm}")


def compute_visibility_domain_variability_error(matched_sequence, settings=None):
    """
    Compare temporal variability of visibility amplitudes between sequences.

    The default radial-bin mode is robust when the sampled u-v points change from
    frame to frame. Sample-index mode assumes the same sample order in every frame.
    """
    settings = settings or {}
    matched_sequence = [item for item in matched_sequence if item.get("matched") is not None and item["matched"].matched_count > 0]

    if len(matched_sequence) < 2:
        return {
            "visibility_domain_variability_error": np.nan,
            "visibility_variability_reason": "fewer_than_two_frames_with_uv_data",
        }, pd.DataFrame()

    mode = str(settings.get("mode", "radial_bins")).lower()
    amplitude_mode = settings.get("amplitude_mode", "amplitude")
    norm = settings.get("norm", "l2")
    epsilon = float(settings.get("epsilon", 1e-12))
    min_samples = int(settings.get("min_samples_per_bin", 3))

    records = []

    if mode == "sample_index":
        n_samples = min(item["matched"].matched_count for item in matched_sequence)
        if n_samples <= 0:
            return {
                "visibility_domain_variability_error": np.nan,
                "visibility_variability_reason": "no_common_sample_indices",
            }, pd.DataFrame()

        true_matrix = []
        recon_matrix = []
        radius_matrix = []
        for item in matched_sequence:
            matched = item["matched"]
            true_matrix.append(_visibility_series_values(matched.vis_true[:n_samples], amplitude_mode, epsilon))
            recon_matrix.append(_visibility_series_values(matched.vis_recon[:n_samples], amplitude_mode, epsilon))
            radius_matrix.append(np.sqrt(matched.u[:n_samples] ** 2 + matched.v[:n_samples] ** 2))

        true_matrix = np.asarray(true_matrix, dtype=float)
        recon_matrix = np.asarray(recon_matrix, dtype=float)
        radius_matrix = np.asarray(radius_matrix, dtype=float)

        true_variance = np.nanvar(true_matrix, axis=0, ddof=1)
        recon_variance = np.nanvar(recon_matrix, axis=0, ddof=1)
        error = _normalized_vector_error(true_variance, recon_variance, norm=norm, epsilon=epsilon)

        for sample_index in range(n_samples):
            records.append({
                "visibility_variability_index": sample_index,
                "visibility_variability_mode": mode,
                "uv_radius_mean": float(np.nanmean(radius_matrix[:, sample_index])),
                "ground_truth_visibility_variance": true_variance[sample_index],
                "reconstruction_visibility_variance": recon_variance[sample_index],
                "variance_abs_error": abs(recon_variance[sample_index] - true_variance[sample_index]),
                "frame_count": len(matched_sequence),
                "sample_count_mean": 1,
            })

        return {
            "visibility_domain_variability_error": error,
            "visibility_variability_mode": mode,
            "visibility_variability_source": ",".join(sorted(set(str(item.get("source", "unknown")) for item in matched_sequence))),
            "visibility_variability_amplitude_mode": amplitude_mode,
            "visibility_variability_norm": norm,
            "visibility_variability_num_frames_used": len(matched_sequence),
            "visibility_variability_num_comparison_points": n_samples,
            "visibility_variability_reason": "ok",
        }, pd.DataFrame(records)

    if mode != "radial_bins":
        raise ValueError(f"Unknown visibility variability mode: {mode}")

    all_radii = []
    for item in matched_sequence:
        matched = item["matched"]
        all_radii.append(np.sqrt(matched.u ** 2 + matched.v ** 2))
    all_radii = np.concatenate(all_radii)
    all_radii = all_radii[np.isfinite(all_radii)]

    if all_radii.size == 0:
        return {
            "visibility_domain_variability_error": np.nan,
            "visibility_variability_reason": "no_finite_uv_radii",
        }, pd.DataFrame()

    max_radius = float(np.nanmax(all_radii))
    if max_radius <= 0:
        return {
            "visibility_domain_variability_error": np.nan,
            "visibility_variability_reason": "zero_uv_radius_range",
        }, pd.DataFrame()

    num_bins = settings.get("num_bins", None)
    if num_bins is None:
        num_bins = max(8, min(64, int(np.sqrt(all_radii.size))))
    num_bins = int(max(1, num_bins))
    edges = np.linspace(0.0, max_radius, num_bins + 1)

    true_by_frame = np.full((len(matched_sequence), num_bins), np.nan, dtype=float)
    recon_by_frame = np.full((len(matched_sequence), num_bins), np.nan, dtype=float)
    count_by_frame = np.zeros((len(matched_sequence), num_bins), dtype=int)

    for frame_row, item in enumerate(matched_sequence):
        matched = item["matched"]
        radius = np.sqrt(matched.u ** 2 + matched.v ** 2)
        true_values = _visibility_series_values(matched.vis_true, amplitude_mode, epsilon)
        recon_values = _visibility_series_values(matched.vis_recon, amplitude_mode, epsilon)

        for bin_index in range(num_bins):
            low = edges[bin_index]
            high = edges[bin_index + 1]
            if bin_index == num_bins - 1:
                mask = (radius >= low) & (radius <= high)
            else:
                mask = (radius >= low) & (radius < high)
            mask = mask & np.isfinite(true_values) & np.isfinite(recon_values)
            count = int(np.count_nonzero(mask))
            count_by_frame[frame_row, bin_index] = count
            if count >= min_samples:
                true_by_frame[frame_row, bin_index] = float(np.nanmean(true_values[mask]))
                recon_by_frame[frame_row, bin_index] = float(np.nanmean(recon_values[mask]))

    true_variance = np.full(num_bins, np.nan, dtype=float)
    recon_variance = np.full(num_bins, np.nan, dtype=float)
    frame_counts = np.zeros(num_bins, dtype=int)

    for bin_index in range(num_bins):
        valid = np.isfinite(true_by_frame[:, bin_index]) & np.isfinite(recon_by_frame[:, bin_index])
        frame_counts[bin_index] = int(np.count_nonzero(valid))
        if frame_counts[bin_index] >= 2:
            true_variance[bin_index] = float(np.nanvar(true_by_frame[valid, bin_index], ddof=1))
            recon_variance[bin_index] = float(np.nanvar(recon_by_frame[valid, bin_index], ddof=1))

        records.append({
            "visibility_variability_index": bin_index,
            "visibility_variability_mode": mode,
            "uv_radius_low": edges[bin_index],
            "uv_radius_high": edges[bin_index + 1],
            "uv_radius_mid": 0.5 * (edges[bin_index] + edges[bin_index + 1]),
            "ground_truth_visibility_variance": true_variance[bin_index],
            "reconstruction_visibility_variance": recon_variance[bin_index],
            "variance_abs_error": abs(recon_variance[bin_index] - true_variance[bin_index]) if np.isfinite(true_variance[bin_index]) and np.isfinite(recon_variance[bin_index]) else np.nan,
            "frame_count": frame_counts[bin_index],
            "sample_count_mean": float(np.nanmean(count_by_frame[:, bin_index])),
        })

    error = _normalized_vector_error(true_variance, recon_variance, norm=norm, epsilon=epsilon)
    comparison_points = int(np.count_nonzero(np.isfinite(true_variance) & np.isfinite(recon_variance)))

    return {
        "visibility_domain_variability_error": error,
        "visibility_variability_mode": mode,
        "visibility_variability_source": ",".join(sorted(set(str(item.get("source", "unknown")) for item in matched_sequence))),
        "visibility_variability_amplitude_mode": amplitude_mode,
        "visibility_variability_norm": norm,
        "visibility_variability_num_frames_used": len(matched_sequence),
        "visibility_variability_num_comparison_points": comparison_points,
        "visibility_variability_min_samples_per_bin": min_samples,
        "visibility_variability_reason": "ok" if comparison_points > 0 else "no_bins_with_two_valid_frames",
    }, pd.DataFrame(records)

# ============================================================
# Plot/output helpers
# ============================================================

def save_image_array(arr, path, title=None, add_colorbar=True):
    """Save a 2D array visualization."""
    arr = np.asarray(arr, dtype=float)

    plt.figure()
    plt.imshow(minmax_normalize(arr))
    if title:
        plt.title(title)
    plt.axis("off")
    if add_colorbar:
        plt.colorbar(fraction=0.046, pad=0.04)
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_metric_over_frames(df, column, ylabel, title, output_path):
    """Save a frame-by-frame line plot for one metric column."""
    if column not in df.columns:
        return

    values = pd.to_numeric(df[column], errors="coerce")
    if values.notna().sum() == 0:
        return

    plt.figure()
    plt.plot(df["frame_index"], values, marker="o")
    plt.xlabel("Frame")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_mean_curve_plot(
    curve_df,
    x_col,
    y_cols,
    labels,
    title,
    xlabel,
    ylabel,
    output_path,
    group_col,
):
    """Save a plot of mean curves grouped by bin index."""
    if curve_df.empty:
        return

    plt.figure()

    grouped = curve_df.groupby(group_col)
    x_values = grouped[x_col].mean()

    any_plotted = False
    for y_col, label in zip(y_cols, labels):
        if y_col not in curve_df.columns:
            continue
        y_values = grouped[y_col].mean()
        if y_values.notna().sum() == 0:
            continue
        plt.plot(x_values, y_values, marker="o", markersize=2, label=label)
        any_plotted = True

    if not any_plotted:
        plt.close()
        return

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def metric_enabled(metrics, name):
    return bool(metrics.get(name, False))


def has_visibility_data(matched):
    return matched is not None and getattr(matched, "matched_count", 0) > 0


def resolve_fourier_source(requested_source, matched_available):
    requested_source = str(requested_source).lower()
    if requested_source == "auto":
        return "uv" if matched_available else "image"
    if requested_source in {"image", "uv"}:
        return requested_source
    raise ValueError(f"Unknown Fourier metric source: {requested_source}")


# ============================================================
# Main evaluation
# ============================================================

def evaluate_folders(config):
    """Run all selected metrics using the supplied CONFIG-style dictionary."""
    cfg = deepcopy(config)
    metrics = cfg.get("metrics", {})
    uv_config = cfg.get("uv_data", {})

    ground_truth_dir = Path(cfg["ground_truth_dir"])
    recon_dir = Path(cfg["reconstruction_dir"])
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    verbose = bool(cfg.get("verbose", True))
    save_plots = bool(cfg.get("save_plots", True))
    save_intermediate_csvs = bool(cfg.get("save_intermediate_csvs", True))

    gt_files = list_images(ground_truth_dir)
    recon_files = list_images(recon_dir)

    max_frames = cfg.get("max_frames", None)
    if max_frames is not None:
        gt_files = gt_files[:int(max_frames)]
        recon_files = recon_files[:int(max_frames)]

    if len(gt_files) != len(recon_files) and verbose:
        print("WARNING: Folder image counts do not match.")
        print(f"Ground truth count:   {len(gt_files)}")
        print(f"Reconstruction count: {len(recon_files)}")
        print("Comparing only the overlapping number of frames.")

    n = min(len(gt_files), len(recon_files))
    if n == 0:
        raise ValueError("No overlapping image frames to compare.")

    get_uv_pair, uv_loader_configured = make_visibility_pair_loader(
        uv_config,
        n_frames=n,
        verbose=verbose,
    )
    get_uv_sampling, uv_sampling_configured = make_uv_sampling_loader(
        uv_config,
        n_frames=n,
        verbose=verbose,
    )

    first_img = PILImage.open(gt_files[0]).convert("L")
    target_size = first_img.size  # PIL-style (width, height)

    records = []
    frc_records = []
    radial_records = []
    azimuthal_records = []
    uv_records = []
    visibility_variability_sequence = []
    visibility_variability_curve_df = pd.DataFrame()
    gt_stack = []
    recon_stack = []

    for i in range(n):
        gt_path = gt_files[i]
        recon_path = recon_files[i]

        if verbose:
            print("=" * 60)
            print(f"Frame {i}")
            print(f"Ground truth:   {gt_path.name}")
            print(f"Reconstruction: {recon_path.name}")

        gt_arr = load_grayscale_array(gt_path, target_size=target_size)
        recon_arr = load_grayscale_array(recon_path, target_size=target_size)

        if cfg.get("flip_ground_truth_vertical", False):
            gt_arr = np.flipud(gt_arr)

        if cfg.get("flip_reconstruction_vertical", False):
            recon_arr = np.flipud(recon_arr)

        if gt_arr.shape != recon_arr.shape:
            raise ValueError(
                f"Shape mismatch after resizing: ground truth {gt_arr.shape}, reconstruction {recon_arr.shape}"
            )

        if cfg.get("crop_to_square", True) and gt_arr.shape[0] != gt_arr.shape[1]:
            gt_arr, recon_arr = center_crop_square_pair(gt_arr, recon_arr)

        if cfg.get("align_reconstruction", False) and int(cfg.get("max_shift_pixels", 0)) > 0:
            recon_arr, best_dy, best_dx, align_ncc = align_reconstruction_by_ncc(
                gt_arr,
                recon_arr,
                max_shift_pixels=int(cfg.get("max_shift_pixels", 0)),
                normalization=cfg.get("normalization", "minmax"),
                total_flux=float(cfg.get("total_flux", 1.0)),
            )
        else:
            best_dy, best_dx, align_ncc = 0, 0, np.nan

        gt_metric = normalize_array(
            gt_arr,
            mode=cfg.get("normalization", "minmax"),
            total_flux=float(cfg.get("total_flux", 1.0)),
        )
        recon_metric = normalize_array(
            recon_arr,
            mode=cfg.get("normalization", "minmax"),
            total_flux=float(cfg.get("total_flux", 1.0)),
        )

        gt_stack.append(gt_arr)
        recon_stack.append(recon_arr)

        true_obs, recon_obs, matched_uv = get_uv_pair(i)
        matched_available = has_visibility_data(matched_uv)

        record = {
            "frame_index": i,
            "ground_truth_file": gt_path.name,
            "reconstruction_file": recon_path.name,
            "height_pixels": gt_arr.shape[0],
            "width_pixels": gt_arr.shape[1],
            "alignment_shift_dy_pixels": best_dy,
            "alignment_shift_dx_pixels": best_dx,
            "alignment_ncc": align_ncc,
        }

        if matched_available:
            uv_record = {
                "frame_index": i,
                "ground_truth_uv_file": Path(true_obs.source_path).name if true_obs and true_obs.source_path else "",
                "reconstruction_uv_file": Path(recon_obs.source_path).name if recon_obs and recon_obs.source_path else "",
                "ground_truth_uv_samples": len(true_obs.u),
                "reconstruction_uv_samples": len(recon_obs.u),
                "matched_uv_samples": matched_uv.matched_count,
                "uv_match_mode": matched_uv.match_mode,
            }
            uv_records.append(uv_record)
            record.update(uv_record)
        else:
            record.update({
                "ground_truth_uv_file": "",
                "reconstruction_uv_file": "",
                "ground_truth_uv_samples": 0,
                "reconstruction_uv_samples": 0,
                "matched_uv_samples": 0,
                "uv_match_mode": "none",
            })

        # ----------------------------------------------------
        # Optional observation u-v image sampling for visibility variability
        # ----------------------------------------------------
        observation_sampling = None
        obs_true = None
        obs_recon = None
        obs_matched_uv = None

        if metric_enabled(metrics, "visibility_domain_variability_error"):
            vv_settings = cfg.get("visibility_variability", {})
            vv_source_request = str(vv_settings.get("source", "auto")).lower()

            use_matched_for_vv = matched_available and vv_source_request in {"auto", "matched_uv"}
            use_observation_for_vv = vv_source_request in {"auto", "observation_uv"} and not use_matched_for_vv

            if use_observation_for_vv:
                observation_sampling = get_uv_sampling(i)
                if observation_sampling is not None:
                    obs_true, obs_recon, obs_matched_uv = make_image_sampled_visibility_pair(
                        gt_metric,
                        recon_metric,
                        observation_sampling,
                        cfg,
                        uv_config,
                    )

            if use_matched_for_vv:
                visibility_variability_sequence.append({
                    "frame_index": i,
                    "source": "matched_uv",
                    "matched": matched_uv,
                })
                record["visibility_variability_frame_source"] = "matched_uv"
                record["visibility_variability_frame_samples"] = matched_uv.matched_count
            elif obs_matched_uv is not None and has_visibility_data(obs_matched_uv):
                visibility_variability_sequence.append({
                    "frame_index": i,
                    "source": "observation_uv_image_sampling",
                    "matched": obs_matched_uv,
                })
                record["observation_uv_file"] = Path(observation_sampling.source_path).name if observation_sampling and observation_sampling.source_path else ""
                record["observation_uv_samples"] = len(observation_sampling.u) if observation_sampling is not None else 0
                record["observation_uv_unit_after_conversion"] = observation_sampling.coordinate_unit if observation_sampling is not None else ""
                record["visibility_variability_frame_source"] = "observation_uv_image_sampling"
                record["visibility_variability_frame_samples"] = obs_matched_uv.matched_count
            else:
                record["observation_uv_file"] = ""
                record["observation_uv_samples"] = 0
                record["observation_uv_unit_after_conversion"] = ""
                record["visibility_variability_frame_source"] = "none"
                record["visibility_variability_frame_samples"] = 0

        # ----------------------------------------------------
        # NRMSE
        # ----------------------------------------------------
        if metric_enabled(metrics, "nrmse"):
            nrmse_value = compute_nrmse(
                gt_arr,
                recon_arr,
                normalization=cfg.get("normalization", "minmax"),
                total_flux=float(cfg.get("total_flux", 1.0)),
            )
            record["nrmse"] = nrmse_value
            if verbose:
                print(f"NRMSE:                         {nrmse_value:.6f}")

        # ----------------------------------------------------
        # SSIM
        # ----------------------------------------------------
        if metric_enabled(metrics, "ssim"):
            ssim_value = compute_ssim(gt_arr, recon_arr)
            record["ssim"] = ssim_value
            if verbose:
                print(f"SSIM:                          {ssim_value:.6f}")

        # ----------------------------------------------------
        # Optional ehtim NRMSE
        # ----------------------------------------------------
        if metric_enabled(metrics, "ehtim_nrmse"):
            ehtim_nrmse = np.nan
            if eh is None:
                if verbose:
                    print("ehtim is not installed; skipping ehtim NRMSE.")
            else:
                try:
                    gt_eh = array_to_ehtim_image(
                        gt_arr,
                        fov_uas=float(cfg.get("fov_uas", 160.0)),
                        total_flux=float(cfg.get("total_flux", 1.0)),
                        source=f"ground_truth_{i:04d}",
                    )
                    recon_eh = array_to_ehtim_image(
                        recon_arr,
                        fov_uas=float(cfg.get("fov_uas", 160.0)),
                        total_flux=float(cfg.get("total_flux", 1.0)),
                        source=f"reconstruction_{i:04d}",
                    )
                    ehtim_nrmse = compute_ehtim_nrmse(
                        gt_eh,
                        recon_eh,
                        allow_shift=bool(cfg.get("allow_ehtim_shift", False)),
                    )
                except Exception as exc:
                    if verbose:
                        print(f"WARNING: ehtim NRMSE failed for frame {i}: {exc}")
                    ehtim_nrmse = np.nan
            record["ehtim_nrmse"] = ehtim_nrmse
            if verbose and np.isfinite(ehtim_nrmse):
                print(f"ehtim NRMSE:                   {ehtim_nrmse:.6f}")

        # ----------------------------------------------------
        # FRC: image FFT or u-v visibility data
        # ----------------------------------------------------
        if metric_enabled(metrics, "frc"):
            frc_source = resolve_fourier_source(cfg.get("frc_source", "auto"), matched_available)
            if frc_source == "uv" and not matched_available:
                if verbose:
                    print("WARNING: FRC source requested as 'uv' but no matched u-v data are available. FRC set to NaN.")
                frc_freq = np.array([])
                frc_values = np.array([])
                frc_counts = np.array([])
                frc_radius_units = np.array([])
            elif frc_source == "uv":
                frc_freq, frc_values, frc_counts, frc_radius_units = compute_frc_curve_uv(
                    matched_uv,
                    num_bins=uv_config.get("uv_frc_num_bins", cfg.get("frc_num_bins", None)),
                    min_samples=int(uv_config.get("uv_frc_min_samples_per_ring", 2)),
                )
            else:
                frc_freq, frc_values, frc_counts = compute_frc_curve_image(
                    gt_arr,
                    recon_arr,
                    normalization=cfg.get("normalization", "minmax"),
                    total_flux=float(cfg.get("total_flux", 1.0)),
                    num_bins=cfg.get("frc_num_bins", None),
                    min_samples=int(cfg.get("frc_min_samples_per_ring", 1)),
                )
                frc_radius_units = np.full_like(frc_freq, np.nan, dtype=float)

            frc_auc_value = curve_auc(frc_freq, frc_values)
            frc_cutoff_value = frc_cutoff_frequency(
                frc_freq,
                frc_values,
                threshold=float(cfg.get("frc_threshold", 0.5)),
            )

            record["frc_source"] = frc_source
            record["frc_auc"] = frc_auc_value
            record["frc_cutoff_frequency"] = frc_cutoff_value

            for bin_index, (freq, frc, count, radius_uv) in enumerate(
                zip(frc_freq, frc_values, frc_counts, frc_radius_units)
            ):
                frc_records.append({
                    "frame_index": i,
                    "frc_bin": bin_index,
                    "frc_source": frc_source,
                    "normalized_spatial_frequency": freq,
                    "uv_radius_midpoint": radius_uv,
                    "frc": frc,
                    "sample_count": count,
                })

            if verbose:
                print(f"FRC source:                    {frc_source}")
                print(f"FRC AUC:                       {frc_auc_value:.6f}")
                print(f"FRC cutoff @ {cfg.get('frc_threshold', 0.5):g}:          {frc_cutoff_value:.6f}")

        # ----------------------------------------------------
        # Fourier chi-squared: image FFT proxy or u-v visibility data
        # ----------------------------------------------------
        if metric_enabled(metrics, "fourier_chi2"):
            chi2_source = resolve_fourier_source(cfg.get("fourier_chi2_source", "auto"), matched_available)
            sigma_used = "none"

            if chi2_source == "uv" and not matched_available:
                if verbose:
                    print("WARNING: Fourier chi-squared source requested as 'uv' but no matched u-v data are available. Value set to NaN.")
                fourier_chi2 = np.nan
            elif chi2_source == "uv":
                fourier_chi2, sigma_used = compute_uv_chi_squared(
                    matched_uv,
                    sigma_mode=uv_config.get("sigma_mode", "ground_truth"),
                    fixed_sigma=float(uv_config.get("fixed_sigma", 1.0)),
                )
            else:
                fourier_chi2 = compute_fourier_chi_squared_image_proxy(
                    gt_arr,
                    recon_arr,
                    normalization=cfg.get("normalization", "minmax"),
                    total_flux=float(cfg.get("total_flux", 1.0)),
                    epsilon=float(cfg.get("fourier_chi2_epsilon", 1e-8)),
                    mode=cfg.get("fourier_chi2_mode", "complex"),
                    denominator_mode=cfg.get("fourier_chi2_denominator", "global_power"),
                    exclude_dc=not bool(cfg.get("include_dc_in_fourier_chi2", False)),
                )

            record["fourier_chi2_source"] = chi2_source
            record["fourier_chi2_sigma_source"] = sigma_used
            record["fourier_chi2"] = fourier_chi2

            if verbose:
                label = "Fourier chi-squared" if chi2_source == "uv" else "Fourier chi-squared proxy"
                print(f"{label}:     {fourier_chi2:.6f}")

        # ----------------------------------------------------
        # Radial profile error
        # ----------------------------------------------------
        center = default_center(
            gt_arr.shape,
            center_x=cfg.get("center_x", None),
            center_y=cfg.get("center_y", None),
        )
        r_max = max_complete_radius(gt_arr.shape, center)

        if metric_enabled(metrics, "radial_profile_error"):
            radial_radii, radial_true, radial_counts = radial_profile(
                gt_metric,
                center=center,
                num_bins=int(cfg.get("radial_bins", 64)),
                r_max=r_max,
            )
            _, radial_recon, _ = radial_profile(
                recon_metric,
                center=center,
                num_bins=int(cfg.get("radial_bins", 64)),
                r_max=r_max,
            )
            radial_error = normalized_profile_l2_error(radial_true, radial_recon)
            record["radial_profile_error"] = radial_error

            for bin_index, (radius_px, true_val, recon_val, count) in enumerate(
                zip(radial_radii, radial_true, radial_recon, radial_counts)
            ):
                radial_records.append({
                    "frame_index": i,
                    "radial_bin": bin_index,
                    "radius_pixels": radius_px,
                    "ground_truth_radial_profile": true_val,
                    "reconstruction_radial_profile": recon_val,
                    "pixel_count": count,
                })

            if verbose:
                print(f"Radial profile error:          {radial_error:.6f}")

        # ----------------------------------------------------
        # Azimuthal profile error
        # ----------------------------------------------------
        if metric_enabled(metrics, "azimuthal_profile_error"):
            if cfg.get("azimuthal_inner_radius", None) is not None and cfg.get("azimuthal_outer_radius", None) is not None:
                annulus_inner = float(cfg.get("azimuthal_inner_radius"))
                annulus_outer = float(cfg.get("azimuthal_outer_radius"))
                annulus_peak = np.nan
            else:
                annulus_inner, annulus_outer, annulus_peak = estimate_annulus_from_truth(
                    gt_metric,
                    center=center,
                    radial_bins=int(cfg.get("radial_bins", 64)),
                    width_fraction=float(cfg.get("azimuthal_annulus_width_fraction", 0.25)),
                    r_max=r_max,
                )

            theta_mid, az_true, az_counts = azimuthal_profile(
                gt_metric,
                center=center,
                num_bins=int(cfg.get("azimuthal_bins", 72)),
                inner_radius=annulus_inner,
                outer_radius=annulus_outer,
            )
            _, az_recon, _ = azimuthal_profile(
                recon_metric,
                center=center,
                num_bins=int(cfg.get("azimuthal_bins", 72)),
                inner_radius=annulus_inner,
                outer_radius=annulus_outer,
            )
            azimuthal_error, azimuthal_roll = azimuthal_profile_error(
                az_true,
                az_recon,
                allow_roll=bool(cfg.get("allow_azimuthal_roll", False)),
            )

            record["azimuthal_profile_error"] = azimuthal_error
            record["azimuthal_annulus_inner_pixels"] = annulus_inner
            record["azimuthal_annulus_outer_pixels"] = annulus_outer
            record["azimuthal_annulus_peak_radius_pixels"] = annulus_peak
            record["azimuthal_roll_bins"] = azimuthal_roll

            for bin_index, (theta_rad, true_val, recon_val, count) in enumerate(
                zip(theta_mid, az_true, az_recon, az_counts)
            ):
                azimuthal_records.append({
                    "frame_index": i,
                    "azimuthal_bin": bin_index,
                    "theta_radians": theta_rad,
                    "theta_degrees": np.degrees(theta_rad),
                    "ground_truth_azimuthal_profile": true_val,
                    "reconstruction_azimuthal_profile": recon_val,
                    "pixel_count": count,
                    "annulus_inner_pixels": annulus_inner,
                    "annulus_outer_pixels": annulus_outer,
                })

            if verbose:
                print(f"Azimuthal profile error:       {azimuthal_error:.6f}")

        records.append(record)

    # ========================================================
    # Sequence-level temporal metrics
    # ========================================================
    gt_stack = np.asarray(gt_stack, dtype=float)
    recon_stack = np.asarray(recon_stack, dtype=float)

    temporal_metric_record = {"num_frames": n}
    temporal_df = pd.DataFrame()
    variance_true = None
    variance_recon = None
    variance_abs_error = None
    st_spatial_error_map = None
    st_temporal_error_map = None

    if metric_enabled(metrics, "temporal_variance_map_error"):
        temporal_variance_error, variance_true, variance_recon, variance_abs_error = compute_temporal_variance_map_error(
            gt_stack,
            recon_stack,
            normalization=cfg.get("normalization", "minmax"),
            total_flux=float(cfg.get("total_flux", 1.0)),
        )

        temporal_metric_record.update({
            "temporal_variance_map_error": temporal_variance_error,
            "ground_truth_variance_mean": float(np.nanmean(variance_true)),
            "reconstruction_variance_mean": float(np.nanmean(variance_recon)),
            "ground_truth_variance_sum": float(np.nansum(variance_true)),
            "reconstruction_variance_sum": float(np.nansum(variance_recon)),
        })

    if metric_enabled(metrics, "spatiotemporal_gradient_error"):
        st_settings = cfg.get("spatiotemporal_gradient", {})
        st_metrics, st_spatial_error_map, st_temporal_error_map = compute_spatiotemporal_gradient_error(
            gt_stack,
            recon_stack,
            normalization=cfg.get("normalization", "minmax"),
            total_flux=float(cfg.get("total_flux", 1.0)),
            norm=st_settings.get("norm", "l2"),
            spatial_weight=float(st_settings.get("spatial_weight", 1.0)),
            temporal_weight=float(st_settings.get("temporal_weight", 1.0)),
            use_absolute_gradients=bool(st_settings.get("use_absolute_gradients", False)),
            epsilon=float(st_settings.get("epsilon", 1e-12)),
        )
        temporal_metric_record.update(st_metrics)

    if metric_enabled(metrics, "visibility_domain_variability_error"):
        vv_metrics, visibility_variability_curve_df = compute_visibility_domain_variability_error(
            visibility_variability_sequence,
            settings=cfg.get("visibility_variability", {}),
        )
        temporal_metric_record.update(vv_metrics)

    if len(temporal_metric_record) > 1:
        temporal_df = pd.DataFrame([temporal_metric_record])

    # ========================================================
    # Save CSVs
    # ========================================================
    df = pd.DataFrame(records)
    frc_df = pd.DataFrame(frc_records)
    radial_df = pd.DataFrame(radial_records)
    azimuthal_df = pd.DataFrame(azimuthal_records)
    uv_df = pd.DataFrame(uv_records)
    if not isinstance(visibility_variability_curve_df, pd.DataFrame):
        visibility_variability_curve_df = pd.DataFrame()

    frame_csv_path = output_dir / "frame_metrics.csv"
    summary_csv_path = output_dir / "summary_metrics.csv"
    df.to_csv(frame_csv_path, index=False)

    summary_columns = [
        col for col in [
            "nrmse",
            "ssim",
            "ehtim_nrmse",
            "frc_auc",
            "frc_cutoff_frequency",
            "fourier_chi2",
            "radial_profile_error",
            "azimuthal_profile_error",
        ]
        if col in df.columns
    ]

    if len(summary_columns) > 0:
        summary = df[summary_columns].describe()
    else:
        summary = pd.DataFrame()
    summary.to_csv(summary_csv_path)

    if save_intermediate_csvs:
        if not frc_df.empty:
            frc_df.to_csv(output_dir / "frc_curves.csv", index=False)
        if not radial_df.empty:
            radial_df.to_csv(output_dir / "radial_profiles.csv", index=False)
        if not azimuthal_df.empty:
            azimuthal_df.to_csv(output_dir / "azimuthal_profiles.csv", index=False)
        if not temporal_df.empty:
            temporal_df.to_csv(output_dir / "temporal_sequence_metrics.csv", index=False)
            # Backward-compatible filename for existing workflows that expect this name.
            temporal_df.to_csv(output_dir / "temporal_variance_metrics.csv", index=False)
        if not visibility_variability_curve_df.empty:
            visibility_variability_curve_df.to_csv(output_dir / "visibility_variability_curve.csv", index=False)
        if not uv_df.empty:
            uv_df.to_csv(output_dir / "uv_data_summary.csv", index=False)

    if verbose:
        print("=" * 60)
        print(f"Saved frame metrics CSV to:          {frame_csv_path}")
        print(f"Saved summary metrics CSV to:        {summary_csv_path}")
        if save_intermediate_csvs:
            if not frc_df.empty:
                print(f"Saved FRC curves CSV to:             {output_dir / 'frc_curves.csv'}")
            if not radial_df.empty:
                print(f"Saved radial profiles CSV to:        {output_dir / 'radial_profiles.csv'}")
            if not azimuthal_df.empty:
                print(f"Saved azimuthal profiles CSV to:     {output_dir / 'azimuthal_profiles.csv'}")
            if not temporal_df.empty:
                print(f"Saved temporal sequence CSV to:      {output_dir / 'temporal_sequence_metrics.csv'}")
            if not visibility_variability_curve_df.empty:
                print(f"Saved visibility variability CSV to: {output_dir / 'visibility_variability_curve.csv'}")
            if not uv_df.empty:
                print(f"Saved u-v data summary CSV to:       {output_dir / 'uv_data_summary.csv'}")
        print("=" * 60)
        if not summary.empty:
            print("Frame-level summary:")
            print(summary)
        if not temporal_df.empty:
            print("=" * 60)
            print("Temporal sequence metrics:")
            print(temporal_df)

    # ========================================================
    # Save plots
    # ========================================================
    if save_plots:
        plot_metric_over_frames(
            df,
            "nrmse",
            "NRMSE",
            "Frame-by-frame NRMSE",
            output_dir / "nrmse_over_frames.png",
        )
        plot_metric_over_frames(
            df,
            "ssim",
            "SSIM",
            "Frame-by-frame SSIM",
            output_dir / "ssim_over_frames.png",
        )
        plot_metric_over_frames(
            df,
            "frc_auc",
            "FRC AUC",
            "Frame-by-frame FRC AUC",
            output_dir / "frc_auc_over_frames.png",
        )
        plot_metric_over_frames(
            df,
            "frc_cutoff_frequency",
            "FRC cutoff frequency",
            "Frame-by-frame FRC cutoff frequency",
            output_dir / "frc_cutoff_frequency_over_frames.png",
        )
        plot_metric_over_frames(
            df,
            "fourier_chi2",
            "Fourier chi-squared",
            "Frame-by-frame Fourier chi-squared",
            output_dir / "fourier_chi2_over_frames.png",
        )
        plot_metric_over_frames(
            df,
            "radial_profile_error",
            "Radial profile error",
            "Frame-by-frame radial profile error",
            output_dir / "radial_profile_error_over_frames.png",
        )
        plot_metric_over_frames(
            df,
            "azimuthal_profile_error",
            "Azimuthal profile error",
            "Frame-by-frame azimuthal profile error",
            output_dir / "azimuthal_profile_error_over_frames.png",
        )

        if not frc_df.empty:
            save_mean_curve_plot(
                frc_df,
                x_col="normalized_spatial_frequency",
                y_cols=["frc"],
                labels=["Mean FRC"],
                title="Mean Fourier Ring Correlation",
                xlabel="Normalized spatial frequency",
                ylabel="FRC",
                output_path=output_dir / "mean_frc_curve.png",
                group_col="frc_bin",
            )

        if not radial_df.empty:
            save_mean_curve_plot(
                radial_df,
                x_col="radius_pixels",
                y_cols=["ground_truth_radial_profile", "reconstruction_radial_profile"],
                labels=["Ground truth", "Reconstruction"],
                title="Mean radial profiles",
                xlabel="Radius (pixels)",
                ylabel="Mean brightness",
                output_path=output_dir / "mean_radial_profiles.png",
                group_col="radial_bin",
            )

        if not azimuthal_df.empty:
            save_mean_curve_plot(
                azimuthal_df,
                x_col="theta_degrees",
                y_cols=["ground_truth_azimuthal_profile", "reconstruction_azimuthal_profile"],
                labels=["Ground truth", "Reconstruction"],
                title="Mean azimuthal profiles",
                xlabel="Angle (degrees)",
                ylabel="Mean brightness",
                output_path=output_dir / "mean_azimuthal_profiles.png",
                group_col="azimuthal_bin",
            )

        if variance_true is not None:
            save_image_array(
                variance_true,
                output_dir / "ground_truth_temporal_variance_map.png",
                title="Ground-truth temporal variance map",
            )
            save_image_array(
                variance_recon,
                output_dir / "reconstruction_temporal_variance_map.png",
                title="Reconstruction temporal variance map",
            )
            save_image_array(
                variance_abs_error,
                output_dir / "temporal_variance_abs_error_map.png",
                title="Absolute temporal variance-map error",
            )

        if st_spatial_error_map is not None and bool(cfg.get("spatiotemporal_gradient", {}).get("save_error_maps", True)):
            save_image_array(
                st_spatial_error_map,
                output_dir / "spatial_gradient_abs_error_map.png",
                title="Spatial gradient absolute error map",
            )
            save_image_array(
                st_temporal_error_map,
                output_dir / "temporal_gradient_abs_error_map.png",
                title="Temporal gradient absolute error map",
            )

        if not visibility_variability_curve_df.empty:
            x_col = "uv_radius_mid" if "uv_radius_mid" in visibility_variability_curve_df.columns else "uv_radius_mean"
            if x_col in visibility_variability_curve_df.columns:
                plt.figure()
                plt.plot(
                    visibility_variability_curve_df[x_col],
                    visibility_variability_curve_df["ground_truth_visibility_variance"],
                    marker="o",
                    label="Ground truth",
                )
                plt.plot(
                    visibility_variability_curve_df[x_col],
                    visibility_variability_curve_df["reconstruction_visibility_variance"],
                    marker="o",
                    label="Reconstruction",
                )
                plt.xlabel("u-v radius / sample index")
                plt.ylabel("Temporal variance of visibility amplitude")
                plt.title("Visibility-domain variability")
                plt.legend()
                plt.savefig(output_dir / "visibility_variability_true_vs_recon.png", dpi=200, bbox_inches="tight")
                plt.close()

        if verbose:
            print(f"Saved selected plots to: {output_dir}")

    return df, temporal_df


# ============================================================
# Optional command-line overrides
# ============================================================

def apply_cli_overrides(config):
    """
    Apply simple optional command-line overrides.

    No arguments are required. This keeps the script runnable from CONFIG while
    still allowing quick path changes when needed.
    """
    parser = argparse.ArgumentParser(
        description="Compare ground-truth and reconstructed image folders using configurable metrics."
    )
    parser.add_argument("--ground_truth_dir", "--original_dir", dest="ground_truth_dir", default=None)
    parser.add_argument("--reconstruction_dir", "--recon_dir", dest="reconstruction_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--normalization", choices=["minmax", "flux", "zscore", "none"], default=None)
    parser.add_argument(
        "--metrics",
        default=None,
        help="Comma-separated list of metrics to run, e.g. nrmse,ssim,frc,fourier_chi2",
    )
    parser.add_argument("--enable_uv", action="store_true")
    parser.add_argument("--ground_truth_uv_dir", default=None)
    parser.add_argument("--reconstruction_uv_dir", default=None)
    parser.add_argument("--ground_truth_uv_file", "--ground_truth_obs", dest="ground_truth_uv_file", default=None)
    parser.add_argument("--reconstruction_uv_file", "--reconstruction_obs", dest="reconstruction_uv_file", default=None)
    parser.add_argument("--observation_uv_dir", default=None)
    parser.add_argument("--observation_uv_file", default=None)
    parser.add_argument("--observation_uv_coordinate_unit", choices=["auto", "seconds", "lambda", "klambda", "mlambda", "glambda"], default=None)
    parser.add_argument("--observing_frequency_hz", type=float, default=None)
    parser.add_argument("--visibility_variability_source", choices=["auto", "matched_uv", "observation_uv"], default=None)
    parser.add_argument("--no_plots", action="store_true", help="Disable plot/image outputs for this run.")
    parser.add_argument(
        "--disable_metric",
        action="append",
        default=[],
        help="Metric name to disable. Can be repeated, e.g. --disable_metric frc",
    )
    parser.add_argument(
        "--enable_metric",
        action="append",
        default=[],
        help="Metric name to enable. Can be repeated, e.g. --enable_metric ehtim_nrmse",
    )

    args = parser.parse_args()
    cfg = deepcopy(config)

    for key in ["ground_truth_dir", "reconstruction_dir", "output_dir", "normalization"]:
        val = getattr(args, key)
        if val is not None:
            cfg[key] = val

    if args.max_frames is not None:
        cfg["max_frames"] = args.max_frames

    if args.metrics is not None:
        requested_metrics = [m.strip() for m in str(args.metrics).split(",") if m.strip()]
        valid_metrics = set(cfg.get("metrics", {}).keys())
        unknown_metrics = sorted(set(requested_metrics) - valid_metrics)
        if unknown_metrics:
            raise ValueError(f"Unknown metric(s): {unknown_metrics}. Valid metrics: {sorted(valid_metrics)}")
        for metric_name in valid_metrics:
            cfg.setdefault("metrics", {})[metric_name] = metric_name in requested_metrics

    if args.no_plots:
        cfg["save_plots"] = False

    if args.enable_uv:
        cfg.setdefault("uv_data", {})["enabled"] = True

    for key in [
        "ground_truth_uv_dir",
        "reconstruction_uv_dir",
        "ground_truth_uv_file",
        "reconstruction_uv_file",
        "observation_uv_dir",
        "observation_uv_file",
        "observation_uv_coordinate_unit",
        "observing_frequency_hz",
    ]:
        val = getattr(args, key)
        if val is not None:
            cfg.setdefault("uv_data", {})[key] = val
            cfg.setdefault("uv_data", {})["enabled"] = True

    if args.visibility_variability_source is not None:
        cfg.setdefault("visibility_variability", {})["source"] = args.visibility_variability_source

    for metric_name in args.disable_metric:
        cfg.setdefault("metrics", {})[metric_name] = False

    for metric_name in args.enable_metric:
        cfg.setdefault("metrics", {})[metric_name] = True

    return cfg


def main():
    cfg = apply_cli_overrides(CONFIG)
    evaluate_folders(cfg)


if __name__ == "__main__":
    main()
