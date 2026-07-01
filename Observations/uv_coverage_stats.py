#!/usr/bin/env python3
# =============================================================================
# START HERE: USER SETTINGS
# =============================================================================
# Edit these values, save the file, then run:
#     python uv_coverage.py
#
# You can still override any of these from the command line, for example:
#     python uv_coverage.py observation.uvfits --npix 512 --pixel-scale-arcsec 1.5

from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent

# Observation file to read. Example: "observation.uvfits" or "observation.npz".
# Leave as None only if you will pass the file path on the command line.
INPUT_FILE = BASE_DIR / "aart_test_results" / "EHTII" / "uvfits" / "0000_average_EHTII_30min.uvfits"

# Image/grid size n for an n x n uv grid.
NPIX = 64

# Use exactly one of PIXEL_SCALE_ARCSEC or FOV_ARCSEC for a physically meaningful FFT grid.
# If both are None, the script auto-scales the grid to the observed uv extent.
PIXEL_SCALE_ARCSEC = None
FOV_ARCSEC = 160e-6

# Only used when PIXEL_SCALE_ARCSEC and FOV_ARCSEC are both None.
# If None, uvmax is set to max(abs(u), abs(v)).
UVMAX_LAMBDA = None

# Include (-u, -v) cells as well as (u, v). Useful when imaging a real sky
# and you want Hermitian-conjugate uv coverage counted.
INCLUDE_CONJUGATES = False

# Also report coverage using the largest centered uv disc, in addition to the square n x n grid.
CIRCULAR_DENOMINATOR = True

# Fallback/reference observing frequency in Hz. Needed when u/v are stored in meters
# or seconds and no frequency exists in the file. Example: 150e6 for 150 MHz.
FREQ_HZ = 226e9

# NPZ u/v unit: "auto", "wavelength", "meters", or "seconds".
NPZ_UV_UNIT = "auto"

# UVFITS UU/VV unit: classic UVFITS usually stores UU/VV in seconds.
# Other accepted values: "meters" or "wavelength".
UVFITS_UVW_UNIT = "seconds"

# For UVFITS, set True to ignore visibility weights when filtering flagged/bad samples.
# If your UVFITS data cube has an unusual layout and the script errors during
# weight extraction, set this to True. The script will still count uv samples.
IGNORE_UVFITS_WEIGHTS = False

# Best-effort extraction of complex visibility values from UVFITS. This is only
# needed if you use MERGE_OUT and want vis_avg in the merged .npz file.
# Coverage statistics only need u/v, so you can set this to False for maximum robustness.
EXTRACT_UVFITS_VIS = False

# Optional output file. Set to a string path like "merged_uv.npz" to write one
# weighted-average row per occupied uv cell. Leave as None to only print the report.
MERGE_OUT = None

# Optional uv-plane diagnostic plots. Set these to output paths to save PNG files.
# The scatter plot shows the actual sampled uv points. The grid-count plot shows
# how many samples landed in each n x n Fourier-grid cell used for coverage.
SCATTER_PLOT_OUT = "uv_scatter.png"       # Example: "uv_scatter.png"
GRID_COUNT_PLOT_OUT = "uv_grid_counts.png"    # Example: "uv_grid_counts.png"

# Plot controls. If there are many samples, the scatter plot is randomly
# downsampled for readability and speed, but the grid-count plot still uses all samples.
PLOT_MAX_POINTS = 200_000
PLOT_DPI = 200
PLOT_SHOW = False
# =============================================================================

"""
uv_coverage.py

Report gridded uv coverage statistics for an observation stored as .npz or .uvfits.

Metrics reported:
  * total actual uv samples after optional flag/weight filtering
  * number of occupied Fourier-grid cells for an n x n image
  * effective uv coverage percentage = occupied cells / n^2 * 100
  * redundancy ratio = samples used for coverage / occupied cells

Optionally writes a compressed .npz in which samples that land in the same uv cell
are averaged/merged, reducing gridded redundancy.

Optionally writes diagnostic PNG plots:
  * raw uv scatter plot
  * n x n grid-cell count/occupancy plot

Examples
--------
# Easiest mode: edit the USER SETTINGS block at the very top of this file,
# then run without command-line arguments:
python uv_coverage.py

# NPZ with u/v already in wavelengths, using an auto-scaled uv grid:
python uv_coverage.py obs.npz --npix 512

# UVFITS, using a physically meaningful image pixel scale:
python uv_coverage.py obs.uvfits --npix 1024 --pixel-scale-arcsec 1.5 --include-conjugates

# Merge/reduce redundant samples by uv grid cell:
python uv_coverage.py obs.npz --npix 512 --pixel-scale-arcsec 2.0 --merge-out obs_gridded_merged.npz

# Save diagnostic plots to verify the uv-plane tracks and gridded occupancy:
python uv_coverage.py obs.uvfits --npix 128 --fov-arcsec 160e-6 --scatter-out uv_scatter.png --grid-count-plot-out uv_grid_counts.png

NPZ input conventions
---------------------
The script accepts several common key names:
  u/v arrays:    u,v  or uu,vv  or u_lambda,v_lambda  or u_m,v_m  or u_sec,v_sec
  combined uvw:  uvw with last dimension length at least 2
  frequency:     freq_hz, frequency_hz, freqs_hz, freq, frequency
  weights:       weight, weights, wgt
  flags:         flag, flags, mask     (True means flagged/bad)
  visibilities:  vis, visibility, data (optional; only needed for --merge-out vis_avg)

UVFITS notes
------------
Classic AIPS-style UVFITS stores UU/VV/WW random parameters in seconds
(light-seconds). Some files name these parameters UU---SIN/VV---SIN/WW---SIN;
this script accepts both forms. The default is --uvfits-uvw-unit seconds. The script multiplies
UU/VV by channel frequency to convert to wavelengths. If your UVFITS uses meters
or wavelengths instead, set --uvfits-uvw-unit meters or wavelength.

Settings vs. command-line arguments
-----------------------------------
Values in USER SETTINGS at the very top of this file are used as defaults. Any command-line argument you pass
overrides the matching setting.
"""


import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

C_M_PER_S = 299_792_458.0
ARCSEC_TO_RAD = math.pi / (180.0 * 3600.0)

@dataclass
class Observation:
    u: np.ndarray                 # wavelengths, flattened, actual samples only
    v: np.ndarray                 # wavelengths, flattened, actual samples only
    weight: Optional[np.ndarray] = None
    vis: Optional[np.ndarray] = None
    source: str = ""
    note: str = ""


@dataclass
class GridResult:
    iu: np.ndarray
    iv: np.ndarray
    valid_inside: np.ndarray
    occupied_count: int
    inside_count: int
    outside_count: int
    coverage_percent_square: float
    coverage_percent_disc: Optional[float]
    redundancy_ratio: float
    duplicate_fraction: float
    du_lambda: float
    uvmax_lambda: float
    mode: str
    disc_cell_count: Optional[int]


def first_key(keys: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    key_set = set(keys)
    lower_to_actual = {k.lower(): k for k in key_set}
    for c in candidates:
        if c in key_set:
            return c
        if c.lower() in lower_to_actual:
            return lower_to_actual[c.lower()]
    return None


def as_float_array(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def frequency_from_npz(npz: np.lib.npyio.NpzFile, fallback_freq_hz: Optional[float]) -> Optional[np.ndarray]:
    k = first_key(npz.files, ["freq_hz", "frequency_hz", "freqs_hz", "frequencies_hz", "freq", "frequency", "frequencies"])
    if k is not None:
        f = as_float_array(npz[k]).reshape(-1)
        # Heuristic: arrays named freq/frequency are assumed Hz unless values look like GHz/MHz are not distinguishable.
        # Keep them as supplied; the CLI has --freq-hz if a scalar override is needed.
        return f
    if fallback_freq_hz is not None:
        return np.asarray([float(fallback_freq_hz)], dtype=np.float64)
    return None


def infer_npz_uv_unit(npz: np.lib.npyio.NpzFile, requested: str) -> str:
    if requested != "auto":
        return requested
    files_lower = {k.lower() for k in npz.files}
    if {"u_lambda", "v_lambda"} <= files_lower or {"uu_lambda", "vv_lambda"} <= files_lower:
        return "wavelength"
    if {"u_m", "v_m"} <= files_lower or {"uu_m", "vv_m"} <= files_lower:
        return "meters"
    if {"u_sec", "v_sec"} <= files_lower or {"uu_sec", "vv_sec"} <= files_lower:
        return "seconds"
    # Most lightweight radio .npz products store u/v in wavelengths.
    return "wavelength"


def broadcast_freq_scale(u: np.ndarray, v: np.ndarray, scale: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Apply a scalar or channel-dependent frequency scale to u/v."""
    scale = np.asarray(scale, dtype=np.float64).reshape(-1)
    if scale.size == 1:
        return u * scale[0], v * scale[0]

    # Common case: one u/v per integration-baseline and one scale per channel.
    if u.ndim == 1:
        return u[:, None] * scale[None, :], v[:, None] * scale[None, :]

    # If the last dimension already looks like channels, broadcast along it.
    if u.shape[-1] == scale.size:
        shape = (1,) * (u.ndim - 1) + (scale.size,)
        return u * scale.reshape(shape), v * scale.reshape(shape)

    # If the first dimension looks like channels, broadcast along it.
    if u.shape[0] == scale.size:
        shape = (scale.size,) + (1,) * (u.ndim - 1)
        return u * scale.reshape(shape), v * scale.reshape(shape)

    raise ValueError(
        f"Could not broadcast {scale.size} frequency channels onto u/v shape {u.shape}. "
        "Store u/v in wavelengths, or provide arrays with a channel axis matching the frequency array."
    )


def convert_uv_to_wavelengths(
    u: np.ndarray,
    v: np.ndarray,
    unit: str,
    freqs_hz: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    unit = unit.lower()
    u = as_float_array(u)
    v = as_float_array(v)
    if unit in {"wavelength", "wavelengths", "lambda"}:
        return u, v
    if freqs_hz is None:
        raise ValueError(f"u/v unit '{unit}' requires a frequency. Provide freq_hz in the file or --freq-hz.")
    freqs_hz = as_float_array(freqs_hz).reshape(-1)
    if unit in {"meter", "meters", "m"}:
        scale = freqs_hz / C_M_PER_S
        return broadcast_freq_scale(u, v, scale)
    if unit in {"second", "seconds", "sec", "s", "light-second", "light-seconds"}:
        scale = freqs_hz
        return broadcast_freq_scale(u, v, scale)
    raise ValueError(f"Unknown u/v unit: {unit}")


def try_broadcast_to_shape(arr: Optional[np.ndarray], shape: Tuple[int, ...], name: str) -> Optional[np.ndarray]:
    if arr is None:
        return None
    arr = np.asarray(arr)
    if arr.shape == shape:
        return arr
    if arr.size == int(np.prod(shape)):
        return arr.reshape(shape)

    # Common after frequency expansion: u/v became (nrow, nchan), but
    # weights/flags were stored per row as (nrow,).
    if arr.ndim == 1 and len(shape) >= 2 and arr.shape[0] == shape[0]:
        reshaped = arr.reshape((arr.shape[0],) + (1,) * (len(shape) - 1))
        try:
            return np.broadcast_to(reshaped, shape)
        except ValueError:
            pass

    try:
        return np.broadcast_to(arr, shape)
    except ValueError as exc:
        raise ValueError(f"Could not broadcast {name} shape {arr.shape} to u/v shape {shape}.") from exc


def load_npz(path: Path, npz_uv_unit: str, fallback_freq_hz: Optional[float]) -> Observation:
    z = np.load(path, allow_pickle=False)
    unit = infer_npz_uv_unit(z, npz_uv_unit)
    files = z.files

    uvw_key = first_key(files, ["uvw", "UVW"])
    if uvw_key is not None:
        uvw = as_float_array(z[uvw_key])
        if uvw.shape[-1] < 2:
            raise ValueError(f"{uvw_key} must have last dimension length >= 2.")
        u_raw = uvw[..., 0]
        v_raw = uvw[..., 1]
    else:
        u_key = first_key(files, ["u_lambda", "uu_lambda", "u_m", "uu_m", "u_sec", "uu_sec", "u", "uu"])
        v_key = first_key(files, ["v_lambda", "vv_lambda", "v_m", "vv_m", "v_sec", "vv_sec", "v", "vv"])
        if u_key is None or v_key is None:
            raise ValueError(
                "Could not find u/v arrays in NPZ. Expected u,v; uu,vv; u_lambda,v_lambda; "
                "u_m,v_m; u_sec,v_sec; or uvw."
            )
        u_raw = as_float_array(z[u_key])
        v_raw = as_float_array(z[v_key])

    freqs = frequency_from_npz(z, fallback_freq_hz)
    u_lam, v_lam = convert_uv_to_wavelengths(u_raw, v_raw, unit, freqs)
    if u_lam.shape != v_lam.shape:
        raise ValueError(f"u and v shapes differ after conversion: {u_lam.shape} vs {v_lam.shape}")

    weight = None
    w_key = first_key(files, ["weight", "weights", "wgt", "wgts"])
    if w_key is not None:
        weight = try_broadcast_to_shape(np.asarray(z[w_key]), u_lam.shape, w_key).astype(np.float64, copy=False)

    valid = np.isfinite(u_lam) & np.isfinite(v_lam)
    if weight is not None:
        valid &= np.isfinite(weight) & (weight > 0)

    flag_key = first_key(files, ["flag", "flags", "mask"])
    if flag_key is not None:
        # Convention used here: True means flagged/bad.
        flags = try_broadcast_to_shape(np.asarray(z[flag_key]).astype(bool), u_lam.shape, flag_key)
        valid &= ~flags

    vis = None
    vis_key = first_key(files, ["vis", "visibility", "visibilities", "data"])
    if vis_key is not None:
        raw_vis = np.asarray(z[vis_key])
        # Accept complex arrays or real/imag stored on the final axis.
        if not np.iscomplexobj(raw_vis) and raw_vis.shape[-1:] == (2,):
            raw_vis = raw_vis[..., 0] + 1j * raw_vis[..., 1]
        try:
            vis = try_broadcast_to_shape(raw_vis, u_lam.shape, vis_key).astype(np.complex128, copy=False)
        except ValueError:
            # Leave vis=None rather than failing the coverage calculation.
            vis = None

    obs = Observation(
        u=u_lam[valid].reshape(-1),
        v=v_lam[valid].reshape(-1),
        weight=weight[valid].reshape(-1) if weight is not None else None,
        vis=vis[valid].reshape(-1) if vis is not None else None,
        source=str(path),
        note=f"NPZ u/v unit interpreted as '{unit}'.",
    )
    return obs


def uvfits_freqs_from_header(header, fallback_freq_hz: Optional[float]) -> Tuple[np.ndarray, Optional[int]]:
    naxis = int(header.get("NAXIS", 0))
    freq_axis = None
    for ax in range(1, naxis + 1):
        ctype = str(header.get(f"CTYPE{ax}", "")).strip().upper()
        if "FREQ" in ctype:
            freq_axis = ax
            break

    if freq_axis is None:
        if fallback_freq_hz is None:
            raise ValueError("No FREQ axis found in UVFITS header. Provide --freq-hz.")
        return np.asarray([float(fallback_freq_hz)], dtype=np.float64), None

    n = int(header.get(f"NAXIS{freq_axis}", 1) or 1)
    crval = float(header.get(f"CRVAL{freq_axis}", fallback_freq_hz if fallback_freq_hz is not None else np.nan))
    cdelt = float(header.get(f"CDELT{freq_axis}", 1.0))
    crpix = float(header.get(f"CRPIX{freq_axis}", 1.0))
    if not np.isfinite(crval):
        raise ValueError("FREQ axis found, but CRVAL is missing/invalid. Provide --freq-hz.")
    freqs = crval + (np.arange(n, dtype=np.float64) + 1.0 - crpix) * cdelt
    return freqs, freq_axis


def fits_axis_to_raw_axis(header, raw: np.ndarray, fits_axis: Optional[int]) -> Optional[int]:
    """
    Map a FITS random-groups axis number to a numpy data axis.

    For common UVFITS random groups, raw.shape is:
        (group, NAXISn, NAXISn-1, ..., NAXIS2)
    because NAXIS1 is zero/unused. Return None if the mapping does not fit.
    """
    if fits_axis is None:
        return None
    naxis = int(header.get("NAXIS", 0))
    fits_axes = list(range(2, naxis + 1))
    if fits_axis not in fits_axes:
        return None
    if raw.ndim != 1 + len(fits_axes):
        return None
    pos = fits_axes.index(fits_axis)
    # Reverse FITS axis order after the group dimension.
    return 1 + (len(fits_axes) - 1 - pos)


def find_uvfits_complex_axis(header, raw: np.ndarray) -> Optional[int]:
    naxis = int(header.get("NAXIS", 0))
    for ax in range(1, naxis + 1):
        ctype = str(header.get(f"CTYPE{ax}", "")).strip().upper()
        if "COMPLEX" in ctype:
            mapped = fits_axis_to_raw_axis(header, raw, ax)
            if mapped is not None and 0 <= mapped < raw.ndim and raw.shape[mapped] >= 3:
                return mapped
    # Fallbacks for common Astropy UVFITS layouts.
    if raw.ndim >= 2 and raw.shape[-1] >= 3:
        return raw.ndim - 1
    for i in range(1, raw.ndim):
        if raw.shape[i] == 3:
            return i
    return None


def collapse_uvfits_weights_to_group_channel(header, raw: np.ndarray, freq_axis: Optional[int]) -> Optional[np.ndarray]:
    """Return a boolean good mask shaped (ngroup, nfreq), if possible."""
    if raw.size == 0 or raw.ndim < 2:
        return None
    complex_axis = find_uvfits_complex_axis(header, raw)
    if complex_axis is None or raw.shape[complex_axis] < 3:
        return None

    freq_raw_axis = fits_axis_to_raw_axis(header, raw, freq_axis) if freq_axis is not None else None
    nfreq = int(header.get(f"NAXIS{freq_axis}", 1) or 1) if freq_axis is not None else 1
    if freq_raw_axis is None and nfreq > 1:
        matches = [i for i, s in enumerate(raw.shape) if i not in (0, complex_axis) and s == nfreq]
        if matches:
            freq_raw_axis = matches[0]

    weight = np.take(raw, indices=2, axis=complex_axis)
    if freq_raw_axis is not None and freq_raw_axis > complex_axis:
        freq_weight_axis = freq_raw_axis - 1
    else:
        freq_weight_axis = freq_raw_axis

    good = np.isfinite(weight) & (weight > 0)
    if freq_weight_axis is None:
        # Reduce everything except group, then add singleton channel axis.
        axes = tuple(i for i in range(good.ndim) if i != 0)
        good_gc = np.any(good, axis=axes)[:, None]
    else:
        axes = tuple(i for i in range(good.ndim) if i not in (0, freq_weight_axis))
        good_gc = np.any(good, axis=axes) if axes else good
        if good_gc.ndim == 1:
            good_gc = good_gc[:, None]

    if good_gc.shape[1] != nfreq:
        try:
            good_gc = np.broadcast_to(good_gc, (good_gc.shape[0], nfreq))
        except ValueError:
            return None
    return good_gc.astype(bool, copy=False)


def collapse_uvfits_vis_weight_to_group_channel(header, raw: np.ndarray, freq_axis: Optional[int]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Best-effort extraction of complex visibilities and weights as (ngroup, nfreq).
    Extra axes such as IF/Stokes are collapsed by weighted averaging.
    """
    if raw.size == 0 or raw.ndim < 2:
        return None, None
    complex_axis = find_uvfits_complex_axis(header, raw)
    if complex_axis is None or raw.shape[complex_axis] < 3:
        return None, None

    freq_raw_axis = fits_axis_to_raw_axis(header, raw, freq_axis) if freq_axis is not None else None
    nfreq = int(header.get(f"NAXIS{freq_axis}", 1) or 1) if freq_axis is not None else 1
    if freq_raw_axis is None and nfreq > 1:
        matches = [i for i, s in enumerate(raw.shape) if i not in (0, complex_axis) and s == nfreq]
        if matches:
            freq_raw_axis = matches[0]

    # Move complex axis to the end and compute where the frequency axis moved.
    moved = np.moveaxis(raw, complex_axis, -1)
    if freq_raw_axis is not None:
        # Build mapping old axis -> new axis after moveaxis.
        old_axes = list(range(raw.ndim))
        old_axes.pop(complex_axis)
        old_axes.append(complex_axis)
        freq_axis_moved = old_axes.index(freq_raw_axis)
    else:
        freq_axis_moved = None

    real = moved[..., 0].astype(np.float64, copy=False)
    imag = moved[..., 1].astype(np.float64, copy=False)
    wt = moved[..., 2].astype(np.float64, copy=False)
    vis = real + 1j * imag
    good = np.isfinite(real) & np.isfinite(imag) & np.isfinite(wt) & (wt > 0)
    safe_wt = np.where(good, wt, 0.0)
    safe_vis_wt = np.where(good, vis * wt, 0.0 + 0.0j)

    if freq_axis_moved is None:
        axes = tuple(i for i in range(vis.ndim) if i != 0)
        sumw = np.sum(safe_wt, axis=axes)[:, None]
        sumvw = np.sum(safe_vis_wt, axis=axes)[:, None]
    else:
        axes = tuple(i for i in range(vis.ndim) if i not in (0, freq_axis_moved))
        sumw = np.sum(safe_wt, axis=axes) if axes else safe_wt
        sumvw = np.sum(safe_vis_wt, axis=axes) if axes else safe_vis_wt
        if sumw.ndim == 1:
            sumw = sumw[:, None]
            sumvw = sumvw[:, None]

    with np.errstate(invalid="ignore", divide="ignore"):
        vis_gc = sumvw / sumw
    vis_gc = np.where(sumw > 0, vis_gc, np.nan + 1j * np.nan)

    if vis_gc.shape[1] != nfreq:
        try:
            vis_gc = np.broadcast_to(vis_gc, (vis_gc.shape[0], nfreq)).copy()
            sumw = np.broadcast_to(sumw, (sumw.shape[0], nfreq)).copy()
        except ValueError:
            return None, None
    return vis_gc, sumw


def load_uvfits(
    path: Path,
    uvfits_uvw_unit: str,
    fallback_freq_hz: Optional[float],
    use_weights: bool,
    extract_vis: bool = False,
) -> Observation:
    try:
        from astropy.io import fits
    except ImportError as exc:
        raise ImportError("Reading UVFITS requires astropy. Install it with: pip install astropy") from exc

    with fits.open(path, memmap=True) as hdul:
        hdu = hdul[0]
        header = hdu.header
        data = hdu.data
        if data is None:
            raise ValueError("Primary HDU contains no random-groups data.")

        raw_parnames = [str(p).strip() for p in getattr(data, "parnames", [])]
        parnames = [p.upper() for p in raw_parnames]

        def resolve_parname(name: str) -> str:
            """
            Resolve UVFITS random parameter names robustly.

            Some UVFITS files store the baseline coordinates as plain UU/VV/WW,
            while AIPS-style files often use projection-suffixed names such as
            UU---SIN, VV---SIN, and WW---SIN. Astropy's data.par(...) usually
            needs the actual stored name, so we map the requested short name to
            the matching full random-parameter name.
            """
            target = name.strip().upper()
            if target in parnames:
                return raw_parnames[parnames.index(target)]

            # Match names whose projection suffix starts after the base name,
            # e.g. UU---SIN, VV---SIN, WW---SIN.
            matches = []
            for actual, upper in zip(raw_parnames, parnames):
                base = upper.split("---", 1)[0].strip()
                if base == target:
                    matches.append(actual)

            # Fallback for variants like UU-SIN or UU_SIN while avoiding broad
            # accidental matches. This keeps the accepted aliases limited to the
            # requested random-parameter prefix.
            if not matches:
                for actual, upper in zip(raw_parnames, parnames):
                    if upper.startswith(target) and (len(upper) == len(target) or upper[len(target)] in "-_ "):
                        matches.append(actual)

            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                raise ValueError(
                    f"Ambiguous UVFITS random parameter '{name}'. Matches {matches}. Found {raw_parnames}."
                )
            raise ValueError(f"Could not find UVFITS random parameter '{name}'. Found {raw_parnames}.")

        def get_par(name: str) -> np.ndarray:
            actual_name = resolve_parname(name)
            try:
                return as_float_array(data.par(actual_name))
            except Exception as exc:
                raise ValueError(
                    f"Found UVFITS random parameter '{actual_name}' for requested '{name}', "
                    "but Astropy could not read it."
                ) from exc

        uu = get_par("UU")
        vv = get_par("VV")
        freqs_hz, freq_axis = uvfits_freqs_from_header(header, fallback_freq_hz)
        u_lam, v_lam = convert_uv_to_wavelengths(uu, vv, uvfits_uvw_unit, freqs_hz)

        valid = np.isfinite(u_lam) & np.isfinite(v_lam)
        weight_gc = None
        vis_gc = None
        uvfits_warnings = []

        # Some UVFITS files have high-dimensional data cubes whose Astropy axis
        # layout is difficult to collapse generically. Coverage only requires the
        # random parameters UU/VV, so weight/visibility extraction is deliberately
        # fail-soft instead of aborting the whole script.
        raw = None
        if use_weights or extract_vis:
            try:
                raw = np.asarray(data.data)
            except Exception as exc:
                uvfits_warnings.append(
                    "Could not convert the UVFITS visibility cube to a NumPy array; "
                    f"continuing with u/v coordinates only. Details: {exc}"
                )

        if use_weights and raw is not None:
            try:
                good_gc = collapse_uvfits_weights_to_group_channel(header, raw, freq_axis)
                if good_gc is not None:
                    valid &= np.broadcast_to(good_gc, valid.shape)
                else:
                    uvfits_warnings.append("Could not read UVFITS weights; counting all finite u/v samples.")
            except Exception as exc:
                uvfits_warnings.append(
                    "Could not apply UVFITS weights; counting all finite u/v samples. "
                    f"Details: {exc}"
                )

        # Extract collapsed vis/weights only when requested. This is optional and
        # is mainly useful for --merge-out vis_avg. If it fails, coverage still works.
        if extract_vis and raw is not None:
            try:
                vis_gc, weight_gc = collapse_uvfits_vis_weight_to_group_channel(header, raw, freq_axis)
            except Exception as exc:
                uvfits_warnings.append(
                    "Could not extract UVFITS visibilities for merged vis_avg; "
                    f"coverage statistics are unaffected. Details: {exc}"
                )
                vis_gc, weight_gc = None, None

        if weight_gc is not None:
            try:
                weight_gc = np.broadcast_to(weight_gc, u_lam.shape)
            except Exception:
                uvfits_warnings.append("Could not broadcast extracted UVFITS weights; omitting weights from output.")
                weight_gc = None
        if vis_gc is not None:
            try:
                vis_gc = np.broadcast_to(vis_gc, u_lam.shape)
            except Exception:
                uvfits_warnings.append("Could not broadcast extracted UVFITS visibilities; omitting vis_avg from output.")
                vis_gc = None

    note = (
        f"UVFITS UU/VV interpreted as '{uvfits_uvw_unit}' and converted to wavelengths. "
        f"Frequency channels used: {freqs_hz.size}."
    )
    if uvfits_warnings:
        note += " Warnings: " + " | ".join(uvfits_warnings)
    return Observation(
        u=u_lam[valid].reshape(-1),
        v=v_lam[valid].reshape(-1),
        weight=weight_gc[valid].reshape(-1) if weight_gc is not None else None,
        vis=vis_gc[valid].reshape(-1) if vis_gc is not None else None,
        source=str(path),
        note=note,
    )


def grid_uv(
    u: np.ndarray,
    v: np.ndarray,
    npix: int,
    pixel_scale_arcsec: Optional[float],
    fov_arcsec: Optional[float],
    uvmax_lambda: Optional[float],
    circular_denominator: bool = True,
) -> GridResult:
    if npix <= 0:
        raise ValueError("--npix must be positive.")
    u = as_float_array(u).reshape(-1)
    v = as_float_array(v).reshape(-1)
    if u.size == 0:
        raise ValueError("No valid uv samples remain after filtering.")

    if pixel_scale_arcsec is not None and fov_arcsec is not None:
        raise ValueError("Use either --pixel-scale-arcsec or --fov-arcsec, not both.")

    if pixel_scale_arcsec is not None or fov_arcsec is not None:
        if fov_arcsec is None:
            fov_arcsec = float(pixel_scale_arcsec) * npix
        fov_rad = float(fov_arcsec) * ARCSEC_TO_RAD
        if fov_rad <= 0:
            raise ValueError("Field of view / pixel scale must be positive.")
        du = 1.0 / fov_rad
        # FFT frequency indices: for even n, valid shifted integer modes are [-n/2, n/2-1].
        neg = -(npix // 2)
        pos_exclusive = npix - (npix // 2)
        ku = np.rint(u / du).astype(np.int64)
        kv = np.rint(v / du).astype(np.int64)
        inside = (ku >= neg) & (ku < pos_exclusive) & (kv >= neg) & (kv < pos_exclusive)
        iu = ku + (npix // 2)
        iv = kv + (npix // 2)
        uvmax = 0.5 / (fov_rad / npix)  # 1 / (2 * pixel_size_rad)
        mode = f"image FFT grid: FOV={fov_arcsec:.6g} arcsec, du={du:.6g} wavelengths"
    else:
        # Auto mode: make a square uv grid that spans the observed uv extent.
        if uvmax_lambda is None:
            uvmax = float(np.nanmax(np.abs(np.concatenate([u, v]))))
        else:
            uvmax = float(uvmax_lambda)
        if not np.isfinite(uvmax) or uvmax <= 0:
            raise ValueError("Could not determine a positive uvmax. Provide --uvmax-lambda.")
        du = (2.0 * uvmax) / npix
        iu = np.floor((u + uvmax) / du).astype(np.int64)
        iv = np.floor((v + uvmax) / du).astype(np.int64)
        # Include samples exactly on the upper boundary.
        iu = np.where(iu == npix, npix - 1, iu)
        iv = np.where(iv == npix, npix - 1, iv)
        inside = (iu >= 0) & (iu < npix) & (iv >= 0) & (iv < npix)
        mode = f"auto uv extent: [-{uvmax:.6g}, +{uvmax:.6g}] wavelengths, du={du:.6g} wavelengths"

    inside_count = int(np.count_nonzero(inside))
    outside_count = int(u.size - inside_count)
    if inside_count == 0:
        occupied = 0
        coverage = 0.0
        redundancy = float("nan")
        dup_fraction = float("nan")
    else:
        lin = iu[inside].astype(np.int64) * npix + iv[inside].astype(np.int64)
        occupied = int(np.unique(lin).size)
        coverage = 100.0 * occupied / float(npix * npix)
        redundancy = inside_count / float(occupied) if occupied else float("nan")
        dup_fraction = 1.0 - occupied / float(inside_count) if inside_count else float("nan")

    coverage_disc = None
    disc_cells = None
    if circular_denominator:
        # Number of grid cells whose centers lie in the largest centered circle fitting inside the n x n grid.
        yy, xx = np.ogrid[:npix, :npix]
        center = (npix - 1) / 2.0
        radius = npix / 2.0
        disc = (xx - center) ** 2 + (yy - center) ** 2 <= radius ** 2
        disc_cells = int(np.count_nonzero(disc))
        if disc_cells > 0:
            # occupied cells inside the centered disc only
            occ_grid = np.zeros((npix, npix), dtype=bool)
            if inside_count > 0:
                occ_grid[iu[inside], iv[inside]] = True
            occ_disc = int(np.count_nonzero(occ_grid & disc))
            coverage_disc = 100.0 * occ_disc / float(disc_cells)

    return GridResult(
        iu=iu,
        iv=iv,
        valid_inside=inside,
        occupied_count=occupied,
        inside_count=inside_count,
        outside_count=outside_count,
        coverage_percent_square=coverage,
        coverage_percent_disc=coverage_disc,
        redundancy_ratio=redundancy,
        duplicate_fraction=dup_fraction,
        du_lambda=du,
        uvmax_lambda=uvmax,
        mode=mode,
        disc_cell_count=disc_cells,
    )


def merge_by_cell(
    obs: Observation,
    grid: GridResult,
    npix: int,
    output_path: Path,
    include_conjugates: bool,
) -> None:
    """
    Write one row per occupied uv grid cell.

    If complex visibilities are available, writes weighted cell averages. Otherwise
    writes cell centers/counts/weight sums only.
    """
    inside = grid.valid_inside
    if not np.any(inside):
        raise ValueError("No samples inside the uv grid; cannot merge.")

    u = obs.u
    v = obs.v
    weight = obs.weight
    vis = obs.vis

    if include_conjugates:
        # Use the same arrays that were used for the coverage calculation.
        # Conjugate visibilities are V(-u,-v) = conj(V(u,v)).
        u = np.concatenate([obs.u, -obs.u])
        v = np.concatenate([obs.v, -obs.v])
        weight = np.concatenate([obs.weight, obs.weight]) if obs.weight is not None else None
        vis = np.concatenate([obs.vis, np.conj(obs.vis)]) if obs.vis is not None else None
        inside = grid.valid_inside

    iu = grid.iu[inside]
    iv = grid.iv[inside]
    linear = iu.astype(np.int64) * npix + iv.astype(np.int64)
    unique_linear, inv = np.unique(linear, return_inverse=True)
    counts = np.bincount(inv).astype(np.int64)

    if weight is None:
        w = np.ones_like(linear, dtype=np.float64)
    else:
        w = np.asarray(weight, dtype=np.float64)[inside]
        w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    weight_sum = np.bincount(inv, weights=w)

    u_sum = np.bincount(inv, weights=np.asarray(u, dtype=np.float64)[inside] * w)
    v_sum = np.bincount(inv, weights=np.asarray(v, dtype=np.float64)[inside] * w)
    with np.errstate(invalid="ignore", divide="ignore"):
        u_avg = u_sum / weight_sum
        v_avg = v_sum / weight_sum
    zero_w = weight_sum <= 0
    if np.any(zero_w):
        # Fall back to unweighted centers for zero-weight cells.
        u_avg[zero_w] = np.bincount(inv, weights=np.asarray(u, dtype=np.float64)[inside])[zero_w] / counts[zero_w]
        v_avg[zero_w] = np.bincount(inv, weights=np.asarray(v, dtype=np.float64)[inside])[zero_w] / counts[zero_w]

    # Cell centers in wavelength coordinates.
    if "auto uv extent" in grid.mode:
        cell_u = (unique_linear // npix + 0.5) * grid.du_lambda - grid.uvmax_lambda
        cell_v = (unique_linear % npix + 0.5) * grid.du_lambda - grid.uvmax_lambda
    else:
        # FFT integer modes times du.
        cell_i = unique_linear // npix
        cell_j = unique_linear % npix
        ku = cell_i - (npix // 2)
        kv = cell_j - (npix // 2)
        cell_u = ku * grid.du_lambda
        cell_v = kv * grid.du_lambda

    out: Dict[str, np.ndarray] = {
        "u_cell_center_lambda": cell_u.astype(np.float64),
        "v_cell_center_lambda": cell_v.astype(np.float64),
        "u_avg_lambda": u_avg.astype(np.float64),
        "v_avg_lambda": v_avg.astype(np.float64),
        "count": counts,
        "weight_sum": weight_sum.astype(np.float64),
        "cell_i": (unique_linear // npix).astype(np.int64),
        "cell_j": (unique_linear % npix).astype(np.int64),
        "npix": np.asarray(npix, dtype=np.int64),
        "du_lambda": np.asarray(grid.du_lambda, dtype=np.float64),
        "uvmax_lambda": np.asarray(grid.uvmax_lambda, dtype=np.float64),
    }

    if vis is not None:
        vv = np.asarray(vis, dtype=np.complex128)[inside]
        real_sum = np.bincount(inv, weights=np.real(vv) * w)
        imag_sum = np.bincount(inv, weights=np.imag(vv) * w)
        with np.errstate(invalid="ignore", divide="ignore"):
            vis_avg = (real_sum + 1j * imag_sum) / weight_sum
        out["vis_avg"] = vis_avg.astype(np.complex128)

    np.savez_compressed(output_path, **out)



def choose_uv_display_unit(u: np.ndarray, v: np.ndarray) -> Tuple[float, str]:
    """Choose a readable uv-axis scale factor and label."""
    max_abs = float(np.nanmax(np.abs(np.concatenate([np.ravel(u), np.ravel(v)])))) if u.size or v.size else 0.0
    if max_abs >= 1e9:
        return 1e9, r"G$\lambda$"
    if max_abs >= 1e6:
        return 1e6, r"M$\lambda$"
    if max_abs >= 1e3:
        return 1e3, r"k$\lambda$"
    return 1.0, r"$\lambda$"


def square_uv_bounds(grid: GridResult, npix: int) -> Tuple[float, float, float, float]:
    """Return approximate uv-grid square boundaries in wavelengths."""
    if "auto uv extent" in grid.mode:
        return -grid.uvmax_lambda, grid.uvmax_lambda, -grid.uvmax_lambda, grid.uvmax_lambda
    half_width = 0.5 * npix * grid.du_lambda
    return -half_width, half_width, -half_width, half_width


def maybe_downsample_indices(n: int, max_points: int, seed: int = 0) -> np.ndarray:
    """Return indices for plotting at most max_points points."""
    if max_points is None or max_points <= 0 or n <= max_points:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=max_points, replace=False))


def save_uv_scatter_plot(
    u: np.ndarray,
    v: np.ndarray,
    grid: GridResult,
    npix: int,
    output_path: Path,
    max_points: int = 200_000,
    dpi: int = 200,
    show: bool = False,
) -> None:
    """Save a scatter plot of uv samples in wavelength coordinates."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("Scatter plotting requires matplotlib. Install it with: pip install matplotlib") from exc

    u = np.asarray(u, dtype=np.float64).reshape(-1)
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    inside = np.asarray(grid.valid_inside, dtype=bool).reshape(-1)
    if inside.shape[0] != u.shape[0]:
        inside = np.ones_like(u, dtype=bool)

    idx = maybe_downsample_indices(u.size, int(max_points), seed=0)
    u_plot = u[idx]
    v_plot = v[idx]
    inside_plot = inside[idx]

    scale, unit_label = choose_uv_display_unit(u_plot, v_plot)
    xmin, xmax, ymin, ymax = square_uv_bounds(grid, npix)

    fig, ax = plt.subplots(figsize=(7, 7))
    if np.any(inside_plot):
        ax.scatter(u_plot[inside_plot] / scale, v_plot[inside_plot] / scale, s=2, alpha=0.45, label="Inside grid")
    if np.any(~inside_plot):
        ax.scatter(u_plot[~inside_plot] / scale, v_plot[~inside_plot] / scale, s=2, alpha=0.45, marker="x", label="Outside grid")

    # Draw the uv-grid boundary used by the coverage calculation.
    rect_x = np.array([xmin, xmax, xmax, xmin, xmin]) / scale
    rect_y = np.array([ymin, ymin, ymax, ymax, ymin]) / scale
    ax.plot(rect_x, rect_y, linewidth=1.0, label=f"{npix} x {npix} grid boundary")

    ax.axhline(0, linewidth=0.8, alpha=0.5)
    ax.axvline(0, linewidth=0.8, alpha=0.5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"u ({unit_label})")
    ax.set_ylabel(f"v ({unit_label})")
    ax.set_title("uv-plane sample locations")
    subtitle = f"plotted {idx.size:,} of {u.size:,} samples"
    if idx.size < u.size:
        subtitle += f"; random downsample for display"
    ax.text(0.01, 0.99, subtitle, transform=ax.transAxes, va="top", ha="left", fontsize=9)
    ax.legend(loc="best", markerscale=4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    if show:
        plt.show()
    plt.close(fig)


def save_grid_count_plot(
    grid: GridResult,
    npix: int,
    output_path: Path,
    dpi: int = 200,
    show: bool = False,
) -> None:
    """Save an n x n image showing the number of uv samples per occupied grid cell."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("Grid-count plotting requires matplotlib. Install it with: pip install matplotlib") from exc

    counts = np.zeros((npix, npix), dtype=np.int64)
    inside = np.asarray(grid.valid_inside, dtype=bool).reshape(-1)
    if np.any(inside):
        np.add.at(counts, (grid.iv[inside], grid.iu[inside]), 1)

    # log10(count + 1) makes both sparse occupancy and high-redundancy cells visible.
    display = np.log10(counts + 1.0)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(display, origin="lower", interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("log10(samples per cell + 1)")
    ax.set_title(f"Gridded uv occupancy: {grid.occupied_count:,} occupied cells")
    ax.set_xlabel("u grid index")
    ax.set_ylabel("v grid index")

    # Mark the center/DC cell.
    center = npix // 2
    ax.plot(center, center, marker="+", markersize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    if show:
        plt.show()
    plt.close(fig)

def fmt_int(x: int) -> str:
    return f"{x:,}"


def print_report(obs: Observation, grid: GridResult, npix: int, effective_sample_count: int, include_conjugates: bool) -> None:
    print("\nUV coverage report")
    print("=" * 18)
    print(f"File:                         {obs.source}")
    if obs.note:
        print(f"Input note:                   {obs.note}")
    print(f"Grid:                         {npix} x {npix}")
    print(f"Grid mode:                    {grid.mode}")
    print(f"Actual valid uv samples:       {fmt_int(obs.u.size)}")
    if include_conjugates:
        print(f"Samples used for coverage:     {fmt_int(effective_sample_count)} (actual + Hermitian conjugates)")
    else:
        print(f"Samples used for coverage:     {fmt_int(effective_sample_count)}")
    print(f"Samples inside grid:           {fmt_int(grid.inside_count)}")
    print(f"Samples outside grid:          {fmt_int(grid.outside_count)}")
    print(f"Occupied uv cells:             {fmt_int(grid.occupied_count)} / {fmt_int(npix * npix)}")
    print(f"Effective uv coverage:         {grid.coverage_percent_square:.6g}% of square n x n grid")
    if grid.coverage_percent_disc is not None:
        print(f"Coverage inside uv disc:       {grid.coverage_percent_disc:.6g}% of {fmt_int(grid.disc_cell_count or 0)} centered-disc cells")
    print(f"Redundancy ratio:              {grid.redundancy_ratio:.6g} samples per occupied uv cell")
    print(f"Duplicate/redundant fraction:  {100.0 * grid.duplicate_fraction:.6g}% of inside-grid samples")
    print("")


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    def path_or_none(value: Optional[str]) -> Optional[Path]:
        return Path(value) if value not in (None, "") else None

    p = argparse.ArgumentParser(
        description=(
            "Compute effective gridded uv coverage and redundancy for .npz or .uvfits observations. "
            "You can either edit the USER SETTINGS block at the top of this file or pass options here."
        )
    )
    p.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=path_or_none(INPUT_FILE),
        help="Observation file: .npz, .uvfits, .uvfits.gz, .fits, or .fits.gz. Defaults to INPUT_FILE in USER SETTINGS.",
    )
    p.add_argument(
        "--npix",
        type=int,
        default=NPIX,
        help=f"Image/grid size n for an n x n Fourier grid. Default from USER SETTINGS: {NPIX}.",
    )
    p.add_argument(
        "--pixel-scale-arcsec",
        type=float,
        default=PIXEL_SCALE_ARCSEC,
        help="Image pixel scale. If supplied, grid cell du = 1 / (npix * pixel_scale_rad).",
    )
    p.add_argument(
        "--fov-arcsec",
        type=float,
        default=FOV_ARCSEC,
        help="Image field of view. Equivalent to npix * pixel scale.",
    )
    p.add_argument(
        "--uvmax-lambda",
        type=float,
        default=UVMAX_LAMBDA,
        help="Only used in auto mode. Half-width of uv grid in wavelengths. Default: max(|u|,|v|).",
    )

    conjugate_group = p.add_mutually_exclusive_group()
    conjugate_group.add_argument(
        "--include-conjugates",
        dest="include_conjugates",
        action="store_true",
        default=INCLUDE_CONJUGATES,
        help="Also grid (-u,-v), useful when the sky image is real and Hermitian symmetry is desired.",
    )
    conjugate_group.add_argument(
        "--no-include-conjugates",
        dest="include_conjugates",
        action="store_false",
        help="Override USER SETTINGS and do not include Hermitian conjugates.",
    )

    circular_group = p.add_mutually_exclusive_group()
    circular_group.add_argument(
        "--circular-denominator",
        dest="circular_denominator",
        action="store_true",
        default=CIRCULAR_DENOMINATOR,
        help="Also report coverage using the largest centered uv disc as an additional denominator.",
    )
    circular_group.add_argument(
        "--no-circular-denominator",
        dest="circular_denominator",
        action="store_false",
        help="Override USER SETTINGS and skip the centered-disc coverage statistic.",
    )

    p.add_argument(
        "--freq-hz",
        type=float,
        default=FREQ_HZ,
        help="Fallback/reference frequency in Hz, needed when u/v are meters or seconds and no frequency is in the file.",
    )
    p.add_argument(
        "--npz-uv-unit",
        choices=["auto", "wavelength", "meters", "seconds"],
        default=NPZ_UV_UNIT,
        help="Unit for NPZ u/v arrays. Auto uses key names when possible and otherwise assumes wavelengths.",
    )
    p.add_argument(
        "--uvfits-uvw-unit",
        choices=["seconds", "meters", "wavelength"],
        default=UVFITS_UVW_UNIT,
        help="Unit of UVFITS UU/VV random parameters. Classic UVFITS default is seconds.",
    )

    weights_group = p.add_mutually_exclusive_group()
    weights_group.add_argument(
        "--ignore-uvfits-weights",
        dest="ignore_uvfits_weights",
        action="store_true",
        default=IGNORE_UVFITS_WEIGHTS,
        help="Do not use UVFITS visibility weights to filter flagged samples.",
    )
    weights_group.add_argument(
        "--use-uvfits-weights",
        dest="ignore_uvfits_weights",
        action="store_false",
        help="Override USER SETTINGS and use UVFITS visibility weights for filtering.",
    )

    extract_group = p.add_mutually_exclusive_group()
    extract_group.add_argument(
        "--extract-uvfits-vis",
        dest="extract_uvfits_vis",
        action="store_true",
        default=EXTRACT_UVFITS_VIS,
        help="Try to extract complex UVFITS visibilities so --merge-out can include vis_avg. More fragile for unusual UVFITS layouts.",
    )
    extract_group.add_argument(
        "--no-extract-uvfits-vis",
        dest="extract_uvfits_vis",
        action="store_false",
        help="Override USER SETTINGS and do not try to extract complex UVFITS visibilities.",
    )

    p.add_argument(
        "--merge-out",
        type=Path,
        default=path_or_none(MERGE_OUT),
        help="Write compressed NPZ with one averaged row per occupied uv grid cell.",
    )
    p.add_argument(
        "--scatter-out",
        type=Path,
        default=path_or_none(SCATTER_PLOT_OUT),
        help="Save a PNG scatter plot of the sampled uv points. Example: --scatter-out uv_scatter.png",
    )
    p.add_argument(
        "--grid-count-plot-out",
        type=Path,
        default=path_or_none(GRID_COUNT_PLOT_OUT),
        help="Save a PNG image of n x n gridded uv sample counts. Example: --grid-count-plot-out uv_grid_counts.png",
    )
    p.add_argument(
        "--plot-max-points",
        type=int,
        default=PLOT_MAX_POINTS,
        help="Maximum number of points to draw in the scatter plot. The grid-count plot always uses all samples.",
    )
    p.add_argument(
        "--plot-dpi",
        type=int,
        default=PLOT_DPI,
        help="DPI for saved diagnostic plots.",
    )

    show_group = p.add_mutually_exclusive_group()
    show_group.add_argument(
        "--show-plots",
        dest="plot_show",
        action="store_true",
        default=PLOT_SHOW,
        help="Display plots interactively after saving them.",
    )
    show_group.add_argument(
        "--no-show-plots",
        dest="plot_show",
        action="store_false",
        help="Save plots without displaying an interactive window.",
    )

    args = p.parse_args(argv)
    if args.input is None:
        p.error("No input file specified. Set INPUT_FILE in the USER SETTINGS block or pass a file path on the command line.")
    if args.npix is None:
        p.error("No grid size specified. Set NPIX in USER SETTINGS or pass --npix.")
    return args

def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)
    path = args.input
    suffixes = "".join(path.suffixes).lower()

    try:
        if suffixes.endswith(".npz"):
            obs = load_npz(path, args.npz_uv_unit, args.freq_hz)
        elif any(suffixes.endswith(s) for s in [".uvfits", ".uvfits.gz", ".fits", ".fits.gz"]):
            obs = load_uvfits(
                path,
                uvfits_uvw_unit=args.uvfits_uvw_unit,
                fallback_freq_hz=args.freq_hz,
                use_weights=not args.ignore_uvfits_weights,
                extract_vis=args.extract_uvfits_vis,
            )
        else:
            raise ValueError(f"Unsupported file extension: {path.suffixes}. Use .npz or .uvfits/.fits.")

        u_cov = obs.u
        v_cov = obs.v
        if args.include_conjugates:
            u_cov = np.concatenate([u_cov, -u_cov])
            v_cov = np.concatenate([v_cov, -v_cov])

        grid = grid_uv(
            u=u_cov,
            v=v_cov,
            npix=args.npix,
            pixel_scale_arcsec=args.pixel_scale_arcsec,
            fov_arcsec=args.fov_arcsec,
            uvmax_lambda=args.uvmax_lambda,
            circular_denominator=args.circular_denominator,
        )
        print_report(obs, grid, args.npix, effective_sample_count=u_cov.size, include_conjugates=args.include_conjugates)

        if args.scatter_out is not None:
            save_uv_scatter_plot(
                u_cov,
                v_cov,
                grid,
                args.npix,
                args.scatter_out,
                max_points=args.plot_max_points,
                dpi=args.plot_dpi,
                show=args.plot_show,
            )
            print(f"uv scatter plot written:       {args.scatter_out}")

        if args.grid_count_plot_out is not None:
            save_grid_count_plot(
                grid,
                args.npix,
                args.grid_count_plot_out,
                dpi=args.plot_dpi,
                show=args.plot_show,
            )
            print(f"uv grid-count plot written:    {args.grid_count_plot_out}")

        if args.merge_out is not None:
            merge_by_cell(obs, grid, args.npix, args.merge_out, include_conjugates=args.include_conjugates)
            print(f"Merged uv-cell file written:   {args.merge_out}")
            print("Merged file fields include cell centers, weighted average u/v, counts, and weight sums;")
            print("vis_avg is included when complex visibilities could be read.")

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
