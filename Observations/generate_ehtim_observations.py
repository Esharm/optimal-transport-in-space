#!/usr/bin/env python3
"""
Generate synthetic VLBI observations from PNG images using eht-imaging.

This script is intended for group data generation and benchmarking.
It reads PNG frames from either a folder or a .zip file, converts each image
into an ehtim Image, samples it with one or more telescope arrays using
Image.observe(...), and saves the resulting observation data as UVFITS, NPZ,
or both.

Typical use:
    1. Edit the settings in the USER SETTINGS section below.
    2. Run: python generate_ehtim_observations.py

Dependencies:
    pip install ehtim pillow numpy matplotlib

Notes:
    - The PNG image is treated as the ground-truth brightness distribution.
    - Intensities are clipped to nonnegative values and normalized to
      TOTAL_FLUX_JY before sampling.
    - NPZ files include visibility data, uv coordinates, sigmas, metadata,
      telescope table fields, and the normalized ground-truth image array.
"""

from __future__ import annotations

import csv
import json
import math
import re
import shutil
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image as PILImage

# Import ehtim only after standard scientific packages so import errors are clear.
import ehtim as eh


# =============================================================================
# USER SETTINGS
# =============================================================================

# -----------------------------------------------------------------------------
# Input and output paths
# -----------------------------------------------------------------------------
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

# INPUT_PATH may be:
#   - a folder containing .png files
#   - a .zip file containing .png files
#   - a single .png file
INPUT_PATH = BASE_DIR / "aart_test_results" / "ground_truth_average"

OUTPUT_DIR = BASE_DIR / "aart_test_results"

# Output choices: "uvfits", "npz", or "both"
OUTPUT_MODE = "uvfits"

# If True, delete OUTPUT_DIR before generating new data.
# Keep False if several people may add results to the same folder.
CLEAR_OUTPUT_DIR = False

# If True, do not regenerate outputs that already exist.
SKIP_EXISTING = False

# If True, copy the normalized/resized ground-truth PNG image used for sampling.
COPY_NORMALIZED_TRUTH_PNGS = True

# If True, create a .zip archive of the generated output folder at the end.
ZIP_OUTPUT_DIR = False


# -----------------------------------------------------------------------------
# PNG input handling
# -----------------------------------------------------------------------------
PNG_EXTENSIONS = (".png",)
RECURSIVE_INPUT_FOLDER_SEARCH = True

# Select a subset of frames after sorting.
# FRAME_STOP follows Python slicing convention: None means through the end.
FRAME_START = 0
FRAME_STOP = None
FRAME_STRIDE = 1
MAX_FRAMES = None

# Image preprocessing before creating the ehtim Image.
NPIX = 64                         # reconstructed/sampled image grid size; use even sizes for nfft
FOV_UAS = 160.0                   # field of view in microarcseconds
TOTAL_FLUX_JY = 1.0               # image total flux after normalization
INVERT_IMAGE = False              # set True if bright/dark are reversed in your PNGs
IMAGE_RESAMPLE = "lanczos"        # "nearest", "bilinear", "bicubic", or "lanczos"
BLANK_IMAGE_POLICY = "error"      # "error" or "skip"


# -----------------------------------------------------------------------------
# Telescope array settings
# -----------------------------------------------------------------------------
# Each entry may be:
#   - a full path to an ehtim array .txt file
#   - a relative path
#   - a filename such as "EHT2017.txt"
#   - a folder containing array .txt files
# You can list several arrays to generate observations for all of them.
ARRAY_INPUTS = [
    # Use a filename when the array is in one of ARRAY_SEARCH_DIRS, or use
    # a full path such as BASE_DIR / "arrays" / "EHTII.txt".
    "EHTII.txt",
]

# Additional places to look when ARRAY_INPUTS contains only a filename.
ARRAY_SEARCH_DIRS = [
    BASE_DIR,
    BASE_DIR / "arrays",
    BASE_DIR / "ehtim_pull" / "eht-imaging" / "arrays",
]


# -----------------------------------------------------------------------------
# Source and observing metadata
# -----------------------------------------------------------------------------
SOURCE_NAME_PREFIX = "synthetic_png"
RA_HOURS = 17.7611225             # fractional hours; example Sgr A* RA
DEC_DEG = -29.0078                # degrees; example Sgr A* Dec
RF_HZ = 230e9                     # observing frequency in Hz
MJD = 57849                       # integer MJD for the simulated observation
TIMETYPE = "UTC"                  # "UTC" or "GMST"
POLREP_OBS = None                 # None, "stokes", or "circ"


# -----------------------------------------------------------------------------
# ehtim Image.observe(...) sampling settings
# -----------------------------------------------------------------------------
TTYPE = "direct"                  # "direct", "nfft", or "fast"; use "direct" if nfft is unavailable
FFT_PAD_FACTOR = 2

TINT_SEC = 60.0                   # integration time in seconds
TADV_SEC = 540.0                   # cadence between scans in seconds
TSTART_HR = 12                  # observation start time in hours, interpreted by TIMETYPE
TSTOP_HR = 12.5             # observation stop time in hours, interpreted by TIMETYPE
BW_HZ = 2.0e9                     # bandwidth in Hz

ELEVMIN_DEG = 10.0
ELEVMAX_DEG = 85.0
NO_ELEVCUT_SPACE = False
FIX_THETA_GMST = False
SGRSCAT = False                   # True applies Sgr A* scattering kernel

# Noise and calibration behavior.
# In ehtim, ampcal=False and phasecal=False add time-dependent calibration errors.
ADD_THERMAL_NOISE = True
JONES = False
INV_JONES = False
OPACITYCAL = True
AMPCAL = True
PHASECAL = True
FRCAL = True
DCAL = True
RLGAINCAL = True
STABILIZE_SCAN_PHASE = False
STABILIZE_SCAN_AMP = False
NEGGAINS = False

# Error model parameters used when the corresponding calibration terms are enabled.
TAU = 0.1
TAUP = 0.1
GAIN_OFFSET = 0.1
GAINP = 0.1
PHASE_STD = -1
DTERM_OFFSET = 0.05
RLRATIO_STD = 0.0
RLPHASE_STD = 0.0
SIGMAT = None
PHASESIGMAT = None
RLG_SIGMAT = None
RLP_SIGMAT = None

# Reproducibility.
# ehtim warns not to use seed=0, so the script shifts 0 to 1 automatically.
BASE_SEED = 1
SEED_FRAME_STEP = 1000
SEED_ARRAY_STEP = 100000

VERBOSE_EHTIM = False


# -----------------------------------------------------------------------------
# UVFITS output settings
# -----------------------------------------------------------------------------
# "stokes" is convenient for Stokes I test data; "circ" is ehtim's default UVFITS style.
UVFITS_POLREP_OUT = "stokes"
UVFITS_FORCE_SINGLEPOL = None     # None, "R", or "L"


# -----------------------------------------------------------------------------
# NPZ output settings
# -----------------------------------------------------------------------------
NPZ_COMPRESSED = True
INCLUDE_RAW_OBSDATA_TABLE_IN_NPZ = True
INCLUDE_TELESCOPE_TABLE_IN_NPZ = True
INCLUDE_GROUND_TRUTH_IMAGE_IN_NPZ = True


# -----------------------------------------------------------------------------
# Optional diagnostic plots
# -----------------------------------------------------------------------------
SAVE_UV_COVERAGE_PLOTS = False
UV_PLOT_UNITS = "Glambda"         # "lambda", "Mlambda", or "Glambda"


# =============================================================================
# INTERNAL CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class RunConfig:
    input_path: str
    output_dir: str
    output_mode: str
    npix: int
    fov_uas: float
    total_flux_jy: float
    ra_hours: float
    dec_deg: float
    rf_hz: float
    mjd: int
    timetype: str
    tint_sec: float
    tadv_sec: float
    tstart_hr: float
    tstop_hr: float
    bw_hz: float
    ttype: str
    fft_pad_factor: float
    add_thermal_noise: bool
    ampcal: bool
    phasecal: bool
    base_seed: int | None


# =============================================================================
# PATH AND FILE UTILITIES
# =============================================================================

def safe_file_stem(text: str) -> str:
    """Return a filesystem-friendly stem."""
    text = str(text).strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("._") or "unnamed"


def natural_sort_key(path: Path) -> list[Any]:
    """Sort frame_2.png before frame_10.png."""
    parts = re.split(r"(\d+)", path.name)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def ensure_clean_output_dir(path: Path, clear: bool) -> None:
    if clear and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def zip_folder(folder_path: Path, zip_path: Path) -> Path:
    folder_path = Path(folder_path)
    zip_path = Path(zip_path)
    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(folder_path.rglob("*")):
            if file_path.is_file():
                zf.write(file_path, file_path.relative_to(folder_path))

    return zip_path


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# =============================================================================
# INPUT IMAGE LOADING
# =============================================================================

def extract_pngs_from_zip(zip_path: Path, extract_dir: Path) -> list[Path]:
    """Safely extract PNG files from a zip file and return sorted extracted paths."""
    zip_path = Path(zip_path)
    extract_dir = Path(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    frame_paths: list[Path] = []

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = sorted(zf.namelist())
        png_members = []
        for member in members:
            member_path = Path(member)
            if member_path.name.startswith("._") or "__MACOSX" in member_path.parts:
                continue
            if member_path.suffix.lower() in PNG_EXTENSIONS:
                png_members.append(member)

        for index, member in enumerate(png_members):
            original_name = Path(member).name
            safe_name = f"{index:06d}_{safe_file_stem(Path(original_name).stem)}.png"
            output_path = extract_dir / safe_name
            with zf.open(member) as src, open(output_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
            frame_paths.append(output_path)

    return sorted(frame_paths, key=natural_sort_key)


def collect_png_paths(input_path: Path, temp_dir: Path) -> list[Path]:
    """Collect PNG images from a folder, zip file, or single PNG file."""
    input_path = Path(input_path).expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"INPUT_PATH does not exist: {input_path}")

    if input_path.is_file() and input_path.suffix.lower() == ".zip":
        frame_paths = extract_pngs_from_zip(input_path, temp_dir / "extracted_pngs")

    elif input_path.is_file() and input_path.suffix.lower() in PNG_EXTENSIONS:
        frame_paths = [input_path]

    elif input_path.is_dir():
        iterator = input_path.rglob("*") if RECURSIVE_INPUT_FOLDER_SEARCH else input_path.glob("*")
        frame_paths = [p for p in iterator if p.is_file() and p.suffix.lower() in PNG_EXTENSIONS]
        frame_paths = sorted(frame_paths, key=natural_sort_key)

    else:
        raise ValueError(
            "INPUT_PATH must be a folder, a .zip file, or a single PNG file. "
            f"Got: {input_path}"
        )

    if not frame_paths:
        raise ValueError(f"No PNG images found in {input_path}")

    frame_paths = frame_paths[FRAME_START:FRAME_STOP:FRAME_STRIDE]
    if MAX_FRAMES is not None:
        frame_paths = frame_paths[: int(MAX_FRAMES)]

    if not frame_paths:
        raise ValueError("Frame selection settings produced an empty frame list.")

    return frame_paths


# =============================================================================
# ARRAY LOADING
# =============================================================================

def candidate_array_paths(spec: Path | str) -> Iterable[Path]:
    """Generate candidate paths for an array specification."""
    raw = Path(spec).expanduser()

    if raw.is_absolute():
        yield raw
        return

    yield (BASE_DIR / raw)
    yield raw.resolve()

    for directory in ARRAY_SEARCH_DIRS:
        yield Path(directory).expanduser() / raw

    if raw.suffix == "":
        with_txt = raw.with_suffix(".txt")
        yield BASE_DIR / with_txt
        for directory in ARRAY_SEARCH_DIRS:
            yield Path(directory).expanduser() / with_txt

    # Common package/install layouts. These are best-effort checks.
    try:
        eh_path = Path(eh.__file__).resolve()
        package_candidates = [
            eh_path.parent / "arrays" / raw.name,
            eh_path.parent.parent / "arrays" / raw.name,
            eh_path.parent.parent.parent / "arrays" / raw.name,
        ]
        if raw.suffix == "":
            package_candidates.extend([
                eh_path.parent / "arrays" / f"{raw.name}.txt",
                eh_path.parent.parent / "arrays" / f"{raw.name}.txt",
                eh_path.parent.parent.parent / "arrays" / f"{raw.name}.txt",
            ])
        for candidate in package_candidates:
            yield candidate
    except Exception:
        return


def resolve_array_file(spec: Path | str) -> Path:
    """Resolve an ehtim array .txt path from a flexible setting."""
    for candidate in candidate_array_paths(spec):
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()

    candidates_preview = "\n".join(str(p) for p in list(candidate_array_paths(spec))[:12])
    raise FileNotFoundError(
        f"Could not find array file for ARRAY_INPUTS entry: {spec}\n"
        f"Checked examples:\n{candidates_preview}"
    )


def expand_array_inputs(array_inputs: Iterable[Path | str]) -> list[Path]:
    """Expand array input settings into concrete .txt files."""
    array_files: list[Path] = []

    for item in array_inputs:
        item_path = Path(item).expanduser()
        if item_path.exists() and item_path.is_dir():
            array_files.extend(sorted(item_path.glob("*.txt"), key=natural_sort_key))
        else:
            array_files.append(resolve_array_file(item))

    # Deduplicate while preserving order.
    seen: set[Path] = set()
    unique_files: list[Path] = []
    for path in array_files:
        path = path.resolve()
        if path not in seen:
            seen.add(path)
            unique_files.append(path)

    if not unique_files:
        raise ValueError("ARRAY_INPUTS did not resolve to any ehtim array .txt files.")

    return unique_files


def load_array(array_file: Path):
    """Load an ehtim telescope array from a .txt file."""
    return eh.array.load_txt(str(array_file))


# =============================================================================
# IMAGE CONVERSION
# =============================================================================

def uas_to_rad(uas: float) -> float:
    return float(uas) * 1e-6 / 3600.0 * math.pi / 180.0


def pil_resample_filter(name: str) -> int:
    # Pillow >= 9 uses PILImage.Resampling; older versions expose these names
    # directly on PILImage. Supporting both keeps the script portable.
    resampling = getattr(PILImage, "Resampling", PILImage)
    filters = {
        "nearest": resampling.NEAREST,
        "bilinear": resampling.BILINEAR,
        "bicubic": resampling.BICUBIC,
        "lanczos": resampling.LANCZOS,
    }
    key = name.lower().strip()
    if key not in filters:
        raise ValueError(f"Unknown IMAGE_RESAMPLE value: {name}")
    return filters[key]


def load_png_as_flux_array(frame_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a PNG as a square grayscale image and normalize to total flux.

    Returns
    -------
    flux_array:
        2D float array in Jy/pixel, summing to TOTAL_FLUX_JY.
    unit_array:
        2D float array normalized to [0, 1] before total-flux normalization.
    """
    frame_path = Path(frame_path)
    image = PILImage.open(frame_path).convert("L")
    image = image.resize((int(NPIX), int(NPIX)), resample=pil_resample_filter(IMAGE_RESAMPLE))

    arr = np.asarray(image, dtype=np.float64)

    if INVERT_IMAGE:
        arr = 255.0 - arr

    arr = np.clip(arr, 0.0, None)
    arr -= np.nanmin(arr)

    max_value = np.nanmax(arr)
    if max_value > 0:
        unit_array = arr / max_value
    else:
        unit_array = arr

    total = float(np.nansum(unit_array))
    if total <= 0:
        message = f"Frame appears blank after preprocessing: {frame_path}"
        if BLANK_IMAGE_POLICY.lower() == "skip":
            raise BlankImageError(message)
        raise ValueError(message)

    flux_array = unit_array / total * float(TOTAL_FLUX_JY)
    return flux_array.astype(np.float64), unit_array.astype(np.float64)


class BlankImageError(ValueError):
    pass


def make_ehtim_image(flux_array: np.ndarray, source_name: str):
    """Create an ehtim Image from a Jy/pixel array."""
    flux_array = np.asarray(flux_array, dtype=np.float64)
    if flux_array.ndim != 2 or flux_array.shape[0] != flux_array.shape[1]:
        raise ValueError(f"Expected a square 2D image array, got shape {flux_array.shape}")

    fov_rad = uas_to_rad(FOV_UAS)
    psize_rad = fov_rad / flux_array.shape[1]

    return eh.image.Image(
        flux_array,
        psize_rad,
        RA_HOURS,
        DEC_DEG,
        rf=RF_HZ,
        source=source_name,
        mjd=MJD,
        polrep="stokes",
        pol_prim="I",
    )


def save_normalized_truth_png(unit_array: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arr_uint8 = np.clip(unit_array * 255.0, 0, 255).astype(np.uint8)
    PILImage.fromarray(arr_uint8, mode="L").save(output_path)


# =============================================================================
# OBSERVATION GENERATION AND SAVING
# =============================================================================

def seed_for(frame_index: int, array_index: int) -> int | bool:
    if BASE_SEED is None:
        return False

    seed = int(BASE_SEED) + int(frame_index) * int(SEED_FRAME_STEP) + int(array_index) * int(SEED_ARRAY_STEP)
    if seed == 0:
        seed = 1
    return seed


def observe_image(im, array, frame_index: int, array_index: int):
    """Sample one ehtim Image with one ehtim Array."""
    return im.observe(
        array,
        TINT_SEC,
        TADV_SEC,
        TSTART_HR,
        TSTOP_HR,
        BW_HZ,
        mjd=MJD,
        timetype=TIMETYPE,
        polrep_obs=POLREP_OBS,
        elevmin=ELEVMIN_DEG,
        elevmax=ELEVMAX_DEG,
        no_elevcut_space=NO_ELEVCUT_SPACE,
        ttype=TTYPE,
        fft_pad_factor=FFT_PAD_FACTOR,
        fix_theta_GMST=FIX_THETA_GMST,
        sgrscat=SGRSCAT,
        add_th_noise=ADD_THERMAL_NOISE,
        jones=JONES,
        inv_jones=INV_JONES,
        opacitycal=OPACITYCAL,
        ampcal=AMPCAL,
        phasecal=PHASECAL,
        frcal=FRCAL,
        dcal=DCAL,
        rlgaincal=RLGAINCAL,
        stabilize_scan_phase=STABILIZE_SCAN_PHASE,
        stabilize_scan_amp=STABILIZE_SCAN_AMP,
        neggains=NEGGAINS,
        tau=TAU,
        taup=TAUP,
        gain_offset=GAIN_OFFSET,
        gainp=GAINP,
        phase_std=PHASE_STD,
        dterm_offset=DTERM_OFFSET,
        rlratio_std=RLRATIO_STD,
        rlphase_std=RLPHASE_STD,
        sigmat=SIGMAT,
        phasesigmat=PHASESIGMAT,
        rlgsigmat=RLG_SIGMAT,
        rlpsigmat=RLP_SIGMAT,
        seed=seed_for(frame_index, array_index),
        verbose=VERBOSE_EHTIM,
    )


def npz_safe_array(values: Any) -> np.ndarray:
    """Convert values to an array that is safe and convenient in an NPZ file."""
    arr = np.asarray(values)
    if arr.dtype == object:
        arr = arr.astype(str)
    return arr


def add_recarray_fields(prefix: str, recarray: Any, payload: dict[str, Any]) -> None:
    names = getattr(getattr(recarray, "dtype", None), "names", None)
    if not names:
        return
    for name in names:
        payload[f"{prefix}_{name}"] = npz_safe_array(recarray[name])


def get_data_field(obs: Any, name: str, default: Any = None) -> Any:
    names = getattr(getattr(obs.data, "dtype", None), "names", None) or []
    if name in names:
        return obs.data[name]
    return default

def uv_coverage_stats(
    obs: Any,
    *,
    npix: int | None = None,
    fov_uas: float | None = None,
    include_conjugates: bool = True,
) -> dict[str, float]:
    """
    Estimate effective uv coverage using only two summary statistics:

      1. uv_effective_sampling_percent:
         Percent of the npix x npix Fourier grid cells occupied by at least
         one uv sample after binning continuous uv coordinates.

      2. uv_redundancy_ratio:
         Number of uv samples, optionally including conjugates, divided by
         the number of unique occupied Fourier-grid cells.

    A high redundancy ratio means many visibilities are clumped into the
    same effective Fourier-grid cells.
    """

    if npix is None:
        npix = int(NPIX)

    if fov_uas is None:
        fov_uas = float(FOV_UAS)

    fov_rad = uas_to_rad(float(fov_uas))

    u = np.asarray(get_data_field(obs, "u", np.array([], dtype=float)), dtype=float)
    v = np.asarray(get_data_field(obs, "v", np.array([], dtype=float)), dtype=float)

    if len(u) == 0:
        return {
            "uv_effective_sampling_percent": 0.0,
            "uv_redundancy_ratio": np.nan,
        }

    if include_conjugates:
        u_all = np.concatenate([u, -u])
        v_all = np.concatenate([v, -v])
    else:
        u_all = u
        v_all = v

    # Fourier grid spacing implied by the image field of view.
    du = 1.0 / fov_rad

    # Bin continuous uv coordinates to nearest Fourier-grid cell.
    ui = np.round(u_all / du).astype(int)
    vi = np.round(v_all / du).astype(int)

    half = int(npix) // 2

    # Approximate FFT grid indices: [-half, half - 1].
    inside = (
        (ui >= -half) & (ui < half) &
        (vi >= -half) & (vi < half)
    )

    ui_inside = ui[inside]
    vi_inside = vi[inside]

    occupied_cells = set(zip(ui_inside.tolist(), vi_inside.tolist()))
    unique_occupied = len(occupied_cells)

    total_grid_cells = int(npix) * int(npix)

    uv_effective_sampling_percent = 100.0 * unique_occupied / total_grid_cells

    if unique_occupied > 0:
        uv_redundancy_ratio = len(u_all) / unique_occupied
    else:
        uv_redundancy_ratio = np.nan

    return {
        "uv_effective_sampling_percent": float(uv_effective_sampling_percent),
        "uv_redundancy_ratio": float(uv_redundancy_ratio)
        if np.isfinite(uv_redundancy_ratio)
        else np.nan,
    }

def save_obs_as_npz(
    obs: Any,
    output_path: Path,
    *,
    frame_path: Path,
    frame_index: int,
    array_file: Path,
    array_name: str,
    array_index: int,
    flux_array: np.ndarray,
    unit_array: np.ndarray,
    source_name: str,
) -> None:
    """Save one ehtim Obsdata object as a method-friendly NPZ file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    u = get_data_field(obs, "u", np.array([], dtype=float))
    v = get_data_field(obs, "v", np.array([], dtype=float))
    vis = get_data_field(obs, "vis", np.array([], dtype=np.complex128))
    sigma = get_data_field(obs, "sigma", np.array([], dtype=float))

    payload: dict[str, Any] = {
        # Convenience fields used by many reconstruction methods.
        "u": npz_safe_array(u),
        "v": npz_safe_array(v),
        "uv_radius": np.sqrt(npz_safe_array(u).astype(float) ** 2 + npz_safe_array(v).astype(float) ** 2) if len(u) else np.array([], dtype=float),
        "vis": npz_safe_array(vis),
        "sigma": npz_safe_array(sigma),
        "amp": np.abs(npz_safe_array(vis)) if len(vis) else np.array([], dtype=float),
        "phase": np.angle(npz_safe_array(vis)) if len(vis) else np.array([], dtype=float),
        "time_hr": npz_safe_array(get_data_field(obs, "time", np.array([], dtype=float))),
        "tint_sec": npz_safe_array(get_data_field(obs, "tint", np.array([], dtype=float))),
        "station_1": npz_safe_array(get_data_field(obs, "t1", np.array([], dtype=str))),
        "station_2": npz_safe_array(get_data_field(obs, "t2", np.array([], dtype=str))),
        # Metadata.
        "frame_index": np.array(frame_index, dtype=np.int64),
        "frame_name": np.array(frame_path.name),
        "frame_stem": np.array(frame_path.stem),
        "array_name": np.array(array_name),
        "array_file": np.array(str(array_file)),
        "source_name": np.array(source_name),
        "seed": np.array(seed_for(frame_index, array_index) if BASE_SEED is not None else -1, dtype=np.int64),
        "npix": np.array(NPIX, dtype=np.int64),
        "fov_uas": np.array(FOV_UAS, dtype=np.float64),
        "fov_rad": np.array(uas_to_rad(FOV_UAS), dtype=np.float64),
        "pixel_size_rad": np.array(uas_to_rad(FOV_UAS) / int(NPIX), dtype=np.float64),
        "total_flux_jy": np.array(TOTAL_FLUX_JY, dtype=np.float64),
        "ra_hours": np.array(RA_HOURS, dtype=np.float64),
        "dec_deg": np.array(DEC_DEG, dtype=np.float64),
        "rf_hz": np.array(RF_HZ, dtype=np.float64),
        "bw_hz": np.array(BW_HZ, dtype=np.float64),
        "mjd": np.array(MJD, dtype=np.int64),
        "timetype": np.array(TIMETYPE),
        "tstart_hr": np.array(TSTART_HR, dtype=np.float64),
        "tstop_hr": np.array(TSTOP_HR, dtype=np.float64),
        "tadv_sec": np.array(TADV_SEC, dtype=np.float64),
        "observe_ttype": np.array(TTYPE),
        "polrep": np.array(getattr(obs, "polrep", "unknown")),
        "obsdata_dtype": np.array(str(getattr(obs.data, "dtype", "unknown"))),
    }

    if INCLUDE_GROUND_TRUTH_IMAGE_IN_NPZ:
        payload["truth_image_jy_per_pixel"] = npz_safe_array(flux_array)
        payload["truth_image_unit_interval"] = npz_safe_array(unit_array)

    if INCLUDE_RAW_OBSDATA_TABLE_IN_NPZ:
        add_recarray_fields("data", obs.data, payload)

    if INCLUDE_TELESCOPE_TABLE_IN_NPZ:
        add_recarray_fields("tarr", obs.tarr, payload)
        if hasattr(obs, "tkey"):
            payload["station_names"] = np.array(sorted(map(str, obs.tkey.keys())))

    if NPZ_COMPRESSED:
        np.savez_compressed(output_path, **payload)
    else:
        np.savez(output_path, **payload)


def save_obs_as_uvfits(obs: Any, output_path: Path) -> None:
    """Save one ehtim Obsdata object as UVFITS.

    The preferred path uses ehtim.io.save.save_obs_uvfits because that helper
    exposes polrep_out and force_singlepol. A method fallback is included for
    older/local ehtim builds that expose UVFITS saving directly on Obsdata.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        eh.io.save.save_obs_uvfits(
            obs,
            str(output_path),
            force_singlepol=UVFITS_FORCE_SINGLEPOL,
            polrep_out=UVFITS_POLREP_OUT,
        )
    except Exception as helper_error:
        try:
            obs.save_uvfits(str(output_path))
        except Exception as method_error:
            raise RuntimeError(
                "Could not save UVFITS using ehtim.io.save.save_obs_uvfits(...) "
                "or obs.save_uvfits(...)."
            ) from method_error


def save_uv_coverage_plot(obs: Any, output_path: Path) -> None:
    if not SAVE_UV_COVERAGE_PLOTS:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)

    u = np.asarray(get_data_field(obs, "u", np.array([], dtype=float)), dtype=float)
    v = np.asarray(get_data_field(obs, "v", np.array([], dtype=float)), dtype=float)

    units = UV_PLOT_UNITS.lower().strip()
    if units in {"glambda", "g", "giga"}:
        scale = 1e9
        label = r"$u,v$ (G$\lambda$)"
    elif units in {"mlambda", "m", "mega"}:
        scale = 1e6
        label = r"$u,v$ (M$\lambda$)"
    else:
        scale = 1.0
        label = r"$u,v$ ($\lambda$)"

    u_plot = u / scale
    v_plot = v / scale

    plt.figure(figsize=(6, 6))
    plt.scatter(u_plot, v_plot, s=7, alpha=0.75, label="observed")
    plt.scatter(-u_plot, -v_plot, s=7, alpha=0.45, label="conjugate")

    max_abs = 1.0
    if len(u_plot):
        max_abs = max(float(np.nanmax(np.abs(u_plot))), float(np.nanmax(np.abs(v_plot))), 1.0)

    plt.xlim(max_abs, -max_abs)
    plt.ylim(-max_abs, max_abs)
    plt.xlabel(label)
    plt.ylabel(label)
    plt.title("uv coverage")
    plt.grid(True, alpha=0.3)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


# =============================================================================
# MANIFEST AND SETTINGS SNAPSHOT
# =============================================================================

def build_config_snapshot() -> RunConfig:
    return RunConfig(
        input_path=str(INPUT_PATH),
        output_dir=str(OUTPUT_DIR),
        output_mode=OUTPUT_MODE,
        npix=int(NPIX),
        fov_uas=float(FOV_UAS),
        total_flux_jy=float(TOTAL_FLUX_JY),
        ra_hours=float(RA_HOURS),
        dec_deg=float(DEC_DEG),
        rf_hz=float(RF_HZ),
        mjd=int(MJD),
        timetype=str(TIMETYPE),
        tint_sec=float(TINT_SEC),
        tadv_sec=float(TADV_SEC),
        tstart_hr=float(TSTART_HR),
        tstop_hr=float(TSTOP_HR),
        bw_hz=float(BW_HZ),
        ttype=str(TTYPE),
        fft_pad_factor=float(FFT_PAD_FACTOR),
        add_thermal_noise=bool(ADD_THERMAL_NOISE),
        ampcal=bool(AMPCAL),
        phasecal=bool(PHASECAL),
        base_seed=None if BASE_SEED is None else int(BASE_SEED),
    )


def write_settings_snapshot(output_dir: Path, array_files: list[Path], frame_paths: list[Path]) -> None:
    settings = asdict(build_config_snapshot())
    settings["array_files"] = [str(p) for p in array_files]
    settings["num_frames"] = len(frame_paths)
    settings["png_extensions"] = list(PNG_EXTENSIONS)
    settings["uvfits_polrep_out"] = UVFITS_POLREP_OUT
    settings["copy_normalized_truth_pngs"] = COPY_NORMALIZED_TRUTH_PNGS
    settings["include_ground_truth_image_in_npz"] = INCLUDE_GROUND_TRUTH_IMAGE_IN_NPZ

    write_text(output_dir / "dataset_settings.json", json.dumps(settings, indent=2, sort_keys=True))


def write_manifest_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Write a CSV manifest with stable, readable column ordering."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    preferred_columns = [
        "status",
        "message",
        "frame_index",
        "frame_name",
        "frame_stem",
        "source_name",
        "array_index",
        "array_name",
        "array_file",
        "seed",
        "output_uvfits",
        "output_npz",
        "truth_png",
        "num_visibility_rows",
        "uv_effective_sampling_percent",
        "uv_redundancy_ratio",
        "u_min_lambda",
        "u_max_lambda",
        "v_min_lambda",
        "v_max_lambda",
        "polrep",
        "npix",
        "fov_uas",
        "total_flux_jy",
        "rf_hz",
        "bw_hz",
    ]

    all_columns: list[str] = []
    seen: set[str] = set()
    for column in preferred_columns:
        if any(column in row for row in rows):
            all_columns.append(column)
            seen.add(column)
    for row in rows:
        for column in row:
            if column not in seen:
                all_columns.append(column)
                seen.add(column)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_readme(output_dir: Path) -> None:
    readme = f"""Synthetic ehtim observation dataset
===================================

This folder was generated by generate_ehtim_observations.py.

Important files
---------------
- dataset_settings.json: parameter snapshot used for this run.
- dataset_manifest.csv: one row per generated observation.
- <array_name>/uvfits/*.uvfits: UVFITS observations, if OUTPUT_MODE includes uvfits.
- <array_name>/npz/*.npz: NumPy observations, if OUTPUT_MODE includes npz.
- <array_name>/truth_pngs/*.png: normalized/resized PNGs used as ground truth, if enabled.

NPZ field conventions
---------------------
Core convenience fields:
- u, v: uv coordinates in wavelengths.
- uv_radius: sqrt(u^2 + v^2), in wavelengths.
- vis: Stokes I complex visibility, when the observation uses Stokes data.
- sigma: Stokes I uncertainty, when available.
- amp, phase: amplitude and phase derived from vis.
- time_hr, tint_sec, station_1, station_2: observing metadata.

Raw ehtim fields are also stored with a data_ prefix when enabled, for example
 data_vis, data_sigma, data_t1, data_t2, data_qvis, data_uvis, data_vvis.
Telescope table fields are stored with a tarr_ prefix when enabled.

Ground-truth image fields:
- truth_image_jy_per_pixel: the exact Jy/pixel image sent to ehtim.Image.
- truth_image_unit_interval: the same image normalized to [0, 1].

Run summary
-----------
OUTPUT_MODE = {OUTPUT_MODE}
NPIX = {NPIX}
FOV_UAS = {FOV_UAS}
TOTAL_FLUX_JY = {TOTAL_FLUX_JY}
RF_HZ = {RF_HZ}
BW_HZ = {BW_HZ}
TTYPE = {TTYPE}
ADD_THERMAL_NOISE = {ADD_THERMAL_NOISE}
"""
    write_text(output_dir / "README_dataset.txt", readme)


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def validate_settings() -> None:
    mode = OUTPUT_MODE.lower().strip()
    if mode not in {"uvfits", "npz", "both"}:
        raise ValueError('OUTPUT_MODE must be "uvfits", "npz", or "both".')

    if int(NPIX) <= 0:
        raise ValueError("NPIX must be positive.")

    if float(FOV_UAS) <= 0:
        raise ValueError("FOV_UAS must be positive.")

    if float(TOTAL_FLUX_JY) <= 0:
        raise ValueError("TOTAL_FLUX_JY must be positive.")

    if float(TSTOP_HR) <= float(TSTART_HR):
        raise ValueError("TSTOP_HR must be greater than TSTART_HR.")


def should_write(path: Path) -> bool:
    return not (SKIP_EXISTING and path.exists())


def main() -> None:
    validate_settings()

    output_dir = Path(OUTPUT_DIR).expanduser().resolve()
    ensure_clean_output_dir(output_dir, CLEAR_OUTPUT_DIR)

    temp_dir = Path(tempfile.mkdtemp(prefix="ehtim_png_frames_"))
    manifest_rows: list[dict[str, Any]] = []

    try:
        frame_paths = collect_png_paths(Path(INPUT_PATH), temp_dir)
        array_files = expand_array_inputs(ARRAY_INPUTS)

        write_settings_snapshot(output_dir, array_files, frame_paths)
        write_readme(output_dir)

        print("=" * 72)
        print("Synthetic ehtim observation generation")
        print(f"Input: {Path(INPUT_PATH).expanduser().resolve()}")
        print(f"Output: {output_dir}")
        print(f"Frames: {len(frame_paths)}")
        print(f"Arrays: {len(array_files)}")
        print(f"Output mode: {OUTPUT_MODE}")
        print("=" * 72)

        for array_index, array_file in enumerate(array_files):
            array_name = safe_file_stem(array_file.stem)
            array = load_array(array_file)

            array_output_dir = output_dir / array_name
            uvfits_dir = array_output_dir / "uvfits"
            npz_dir = array_output_dir / "npz"
            truth_dir = array_output_dir / "truth_pngs"
            plot_dir = array_output_dir / "uv_coverage_plots"

            print(f"\nArray {array_index + 1}/{len(array_files)}: {array_name}")
            print(f"Array file: {array_file}")

            for frame_index, frame_path in enumerate(frame_paths):
                frame_stem = safe_file_stem(frame_path.stem)
                source_name = f"{SOURCE_NAME_PREFIX}_{frame_index:04d}_{frame_stem}"
                output_stem = f"{frame_index:04d}_{frame_stem}_{array_name}"

                uvfits_path = uvfits_dir / f"{output_stem}.uvfits"
                npz_path = npz_dir / f"{output_stem}.npz"
                truth_path = truth_dir / f"{frame_index:04d}_{frame_stem}.png"
                uv_plot_path = plot_dir / f"{output_stem}_uv.png"

                print(f"  [{frame_index + 1:>4}/{len(frame_paths)}] {frame_path.name}")

                try:
                    flux_array, unit_array = load_png_as_flux_array(frame_path)
                except BlankImageError as exc:
                    print(f"    Skipped blank frame: {exc}")
                    manifest_rows.append({
                        "frame_index": frame_index,
                        "frame_name": frame_path.name,
                        "array_name": array_name,
                        "status": "skipped_blank",
                        "message": str(exc),
                    })
                    continue

                im = make_ehtim_image(flux_array, source_name=source_name)

                needs_obs = True
                mode = OUTPUT_MODE.lower().strip()
                if SKIP_EXISTING:
                    expected_paths = []
                    if mode in {"uvfits", "both"}:
                        expected_paths.append(uvfits_path)
                    if mode in {"npz", "both"}:
                        expected_paths.append(npz_path)
                    needs_obs = not all(path.exists() for path in expected_paths)

                if needs_obs:
                    obs = observe_image(im, array, frame_index=frame_index, array_index=array_index)
                else:
                    obs = None

                if COPY_NORMALIZED_TRUTH_PNGS and should_write(truth_path):
                    save_normalized_truth_png(unit_array, truth_path)

                if mode in {"uvfits", "both"} and should_write(uvfits_path):
                    if obs is None:
                        obs = observe_image(im, array, frame_index=frame_index, array_index=array_index)
                    save_obs_as_uvfits(obs, uvfits_path)

                if mode in {"npz", "both"} and should_write(npz_path):
                    if obs is None:
                        obs = observe_image(im, array, frame_index=frame_index, array_index=array_index)
                    save_obs_as_npz(
                        obs,
                        npz_path,
                        frame_path=frame_path,
                        frame_index=frame_index,
                        array_file=array_file,
                        array_name=array_name,
                        array_index=array_index,
                        flux_array=flux_array,
                        unit_array=unit_array,
                        source_name=source_name,
                    )

                if SAVE_UV_COVERAGE_PLOTS and should_write(uv_plot_path):
                    if obs is None:
                        obs = observe_image(im, array, frame_index=frame_index, array_index=array_index)
                    save_uv_coverage_plot(obs, uv_plot_path)

                if obs is not None:
                    uv_stats = uv_coverage_stats(
                        obs,
                        npix=int(NPIX),
                        fov_uas=float(FOV_UAS),
                        include_conjugates=True,
                    )

                    print(
                        "    uv coverage: "
                        f"{uv_stats['uv_effective_sampling_percent']:.2f}% effective, "
                        f"redundancy ratio = {uv_stats['uv_redundancy_ratio']:.2f}"
                    )
                else:
                    uv_stats = {
                        "uv_effective_sampling_percent": np.nan,
                        "uv_redundancy_ratio": np.nan,
                    }

                if obs is not None:
                    u = np.asarray(get_data_field(obs, "u", np.array([], dtype=float)), dtype=float)
                    v = np.asarray(get_data_field(obs, "v", np.array([], dtype=float)), dtype=float)
                    num_vis = len(obs.data)
                    u_min = float(np.nanmin(u)) if len(u) else np.nan
                    u_max = float(np.nanmax(u)) if len(u) else np.nan
                    v_min = float(np.nanmin(v)) if len(v) else np.nan
                    v_max = float(np.nanmax(v)) if len(v) else np.nan
                    polrep = getattr(obs, "polrep", "unknown")
                else:
                    num_vis = np.nan
                    u_min = u_max = v_min = v_max = np.nan
                    polrep = "not_loaded_skip_existing"

                manifest_rows.append({
                    "frame_index": frame_index,
                    "frame_name": frame_path.name,
                    "frame_stem": frame_stem,
                    "source_name": source_name,
                    "array_index": array_index,
                    "array_name": array_name,
                    "array_file": str(array_file),
                    "seed": seed_for(frame_index, array_index) if BASE_SEED is not None else "none",
                    "output_uvfits": str(uvfits_path.relative_to(output_dir)) if mode in {"uvfits", "both"} else "",
                    "output_npz": str(npz_path.relative_to(output_dir)) if mode in {"npz", "both"} else "",
                    "truth_png": str(truth_path.relative_to(output_dir)) if COPY_NORMALIZED_TRUTH_PNGS else "",
                    "num_visibility_rows": num_vis,
                    "uv_effective_sampling_percent": uv_stats["uv_effective_sampling_percent"],
                    "uv_redundancy_ratio": uv_stats["uv_redundancy_ratio"],
                    "u_min_lambda": u_min,
                    "u_max_lambda": u_max,
                    "v_min_lambda": v_min,
                    "v_max_lambda": v_max,
                    "polrep": polrep,
                    "npix": NPIX,
                    "fov_uas": FOV_UAS,
                    "total_flux_jy": TOTAL_FLUX_JY,
                    "rf_hz": RF_HZ,
                    "bw_hz": BW_HZ,
                    "status": "ok",
                    "message": "",
                })

        manifest_path = output_dir / "dataset_manifest.csv"
        write_manifest_csv(manifest_rows, manifest_path)

        if ZIP_OUTPUT_DIR:
            zip_path = output_dir.with_suffix(".zip")
            zip_folder(output_dir, zip_path)
            print(f"\nZipped output folder: {zip_path}")

        print("\n" + "=" * 72)
        print("Done.")
        print(f"Manifest: {manifest_path}")
        print(f"Settings: {output_dir / 'dataset_settings.json'}")
        print("=" * 72)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
