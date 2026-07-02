from __future__ import annotations

import argparse
import csv
import logging
import shutil
import tempfile
import zipfile
from copy import deepcopy
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
import numpy as np
from PIL import Image as PILImage

# Use a non-interactive backend because WSL/headless runs often cannot show windows.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ehtim as eh


# -----------------------------------------------------------------------------
# User settings
# -----------------------------------------------------------------------------
# Set these values once, then run this file directly:
#
#     python Static_Reconstruction_run_ready.py
#
# By default this ignores command-line inputs and uses only the settings below.
#
# FRAME_INPUT can be:
#   - a .zip file containing PNG/JPG frames
#   - a folder containing PNG/JPG frames
#   - a single PNG/JPG frame
#
# UVFITS_INPUT can be:
#   - one UVFITS/FITS file, reused for every frame
#   - a folder of UVFITS/FITS files, one file per frame or one total file
#   - a .zip file of UVFITS/FITS files, one file per frame or one total file

try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

# ---- Inputs -----------------------------------------------------------------
FRAME_INPUT = (
    BASE_DIR
    / "optimal-transport-in-space"
    / "blackhole_sim"
    / "data"
    / "aart_frames"
)
UVFITS_INPUT = BASE_DIR / "optimal-transport-in-space" / "blackhole_sim_testing" / "observations_fixed"

# ---- Outputs ----------------------------------------------------------------
OUTPUT_DIR = BASE_DIR / "results"
OUTPUT_ZIP_NAME = "reconstructed_frames_output.zip"
CLEAR_OUTPUT_DIRS = False

# ---- Image/reconstruction grid ----------------------------------------------
NPIX = 128
FOV_UAS = 160.0
TOTAL_FLUX_JY = 2.64

# Metadata used when a PNG/JPG is converted into an ehtim Image template.
# If the UVFITS object contains these fields, the observation metadata is used.
RA_DEG = 17.7611225
DEC_DEG = -29.0078
RF_HZ = 230e9
MJD = 57849

# ---- UVFITS loading ----------------------------------------------------------
POLREP = "stokes"
FLIPBL = False
REMOVE_NAN = True
FORCE_SINGLEPOL = None
ALLOW_SINGLEPOL = True
IGNORE_PZERO_DATE = True

# Optional coherent averaging in seconds. Use None to leave the data unchanged.
AVG_COHERENT_SEC = None

# ---- Imager/reconstruction settings -----------------------------------------
TTYPE = "direct"  # change to "nfft" if nfft is installed and desired
USE_PREVIOUS_FRAME_AS_PRIOR = False
DATA_TERM = {"vis": 1.0}
LAMBDA_L1 = 0.00001
LAMBDA_TV = 0.1
MAXIT_FIRST_FRAME = 10000
MAXIT_LATER_FRAMES = 10000

# ---- Diagnostics -------------------------------------------------------------
SAVE_UV_COVERAGE = True
SAVE_OBSERVATION_CSV = True
UV_COVERAGE_UNITS = "Glambda"
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, or ERROR

# Leave this False to run from the USER SETTINGS block only.
# Set to True if you want terminal flags like --frame-input to override settings.
USE_COMMAND_LINE_OVERRIDES = False


@dataclass(frozen=True)
class ReconstructionConfig:
    """All reconstruction settings used by the pipeline."""

    # Inputs
    frame_input: Path = field(default_factory=lambda: Path(FRAME_INPUT))
    uvfits_input: Path = field(default_factory=lambda: Path(UVFITS_INPUT))

    # Outputs
    output_dir: Path = field(default_factory=lambda: Path(OUTPUT_DIR))
    output_zip_name: str = OUTPUT_ZIP_NAME
    clear_output_dirs: bool = CLEAR_OUTPUT_DIRS

    # Image/reconstruction grid
    npix: int = NPIX
    fov_uas: float = FOV_UAS
    total_flux_jy: float = TOTAL_FLUX_JY

    # Metadata used when a PNG/JPG is converted into an ehtim Image template.
    ra: float = RA_DEG
    dec: float = DEC_DEG
    rf_hz: float = RF_HZ
    mjd: int = MJD

    # UVFITS loading settings
    polrep: str = POLREP
    flipbl: bool = FLIPBL
    remove_nan: bool = REMOVE_NAN
    force_singlepol: Optional[str] = FORCE_SINGLEPOL
    allow_singlepol: bool = ALLOW_SINGLEPOL
    ignore_pzero_date: bool = IGNORE_PZERO_DATE

    # Optional coherent averaging, in seconds. Use None to leave data unchanged.
    avg_coherent_sec: Optional[float] = AVG_COHERENT_SEC

    # Imager settings
    ttype: str = TTYPE
    use_previous_frame_as_prior: bool = USE_PREVIOUS_FRAME_AS_PRIOR
    data_term: Dict[str, float] = field(default_factory=lambda: dict(DATA_TERM))
    lambda_l1: float = LAMBDA_L1
    lambda_tv: float = LAMBDA_TV
    maxit_first_frame: int = MAXIT_FIRST_FRAME
    maxit_later_frames: int = MAXIT_LATER_FRAMES

    # Diagnostics
    save_uv_coverage: bool = SAVE_UV_COVERAGE
    save_observation_csv: bool = SAVE_OBSERVATION_CSV
    uv_coverage_units: str = UV_COVERAGE_UNITS

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame_input", Path(self.frame_input).expanduser())
        object.__setattr__(self, "uvfits_input", Path(self.uvfits_input).expanduser())
        object.__setattr__(self, "output_dir", Path(self.output_dir).expanduser())

    @property
    def recon_display_dir(self) -> Path:
        return self.output_dir / "reconstructed_frames_display"

    @property
    def recon_gray_dir(self) -> Path:
        return self.output_dir / "reconstructed_frames_gray"

    @property
    def fourier_csv_dir(self) -> Path:
        return self.output_dir / "fourier_csv"

    @property
    def output_zip(self) -> Path:
        return self.output_dir / self.output_zip_name


DEFAULT_CONFIG = ReconstructionConfig()

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
UVFITS_SUFFIXES = {".uvfits", ".uvf", ".fits", ".fit"}

LOGGER = logging.getLogger("static_reconstruction")


# -----------------------------------------------------------------------------
# File discovery and extraction
# -----------------------------------------------------------------------------


def safe_clear_dir(path: Path) -> None:
    """Delete and recreate a directory."""
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _safe_zip_member_path(member_name: str) -> Optional[Path]:
    """Return a safe relative path for a zip member, or None for unsafe names."""
    member_path = PurePosixPath(member_name)

    if member_path.is_absolute() or ".." in member_path.parts:
        return None

    parts = [part for part in member_path.parts if part not in {"", "."}]

    if not parts:
        return None

    return Path(*parts)


def _extract_matching_files_from_zip(
    zip_path: Path,
    extract_dir: Path,
    suffixes: Iterable[str],
    description: str,
) -> List[Path]:
    """Extract files with matching suffixes from a zip archive."""
    suffixes = {suffix.lower() for suffix in suffixes}
    zip_path = Path(zip_path)
    extract_dir = Path(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    extracted_paths: List[Path] = []

    with zipfile.ZipFile(zip_path, "r") as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue

            member_path = _safe_zip_member_path(member.filename)
            if member_path is None or member_path.suffix.lower() not in suffixes:
                continue

            output_path = extract_dir / member_path
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with archive.open(member, "r") as src, open(output_path, "wb") as dst:
                shutil.copyfileobj(src, dst)

            extracted_paths.append(output_path)

    extracted_paths = sorted(extracted_paths, key=lambda path: str(path).lower())

    if not extracted_paths:
        suffix_list = ", ".join(sorted(suffixes))
        raise ValueError(f"No {description} files ({suffix_list}) found in {zip_path}")

    return extracted_paths


def collect_input_files(
    input_path: Path,
    suffixes: Iterable[str],
    extract_dir: Path,
    description: str,
) -> List[Path]:
    """
    Collect files from a single file, a directory, or a zip archive.

    Directory inputs are searched recursively. Zip inputs are extracted into
    extract_dir and returned as temporary paths.
    """
    input_path = Path(input_path).expanduser().resolve()
    suffixes = {suffix.lower() for suffix in suffixes}

    if not input_path.exists():
        raise FileNotFoundError(f"Could not find {description} input: {input_path}")

    if input_path.is_dir():
        paths = sorted(
            (
                path
                for path in input_path.rglob("*")
                if path.is_file() and path.suffix.lower() in suffixes
            ),
            key=lambda path: str(path).lower(),
        )
    elif input_path.is_file() and input_path.suffix.lower() == ".zip":
        paths = _extract_matching_files_from_zip(
            input_path,
            extract_dir=extract_dir,
            suffixes=suffixes,
            description=description,
        )
    elif input_path.is_file() and input_path.suffix.lower() in suffixes:
        paths = [input_path]
    else:
        suffix_list = ", ".join(sorted(suffixes))
        raise ValueError(
            f"Unsupported {description} input: {input_path}. "
            f"Expected a directory, .zip archive, or one of: {suffix_list}"
        )

    if not paths:
        suffix_list = ", ".join(sorted(suffixes))
        raise ValueError(f"No {description} files ({suffix_list}) found in {input_path}")

    return paths


def validate_observation_count(frame_paths: Sequence[Path], uvfits_paths: Sequence[Path]) -> None:
    """Require either one UVFITS file for all frames or one UVFITS per frame."""
    if len(uvfits_paths) in {1, len(frame_paths)}:
        return

    raise ValueError(
        "UVFITS/frame count mismatch. Provide either exactly one UVFITS file "
        f"for all frames or exactly one UVFITS per frame. Found "
        f"{len(uvfits_paths)} UVFITS file(s) and {len(frame_paths)} frame(s)."
    )


# -----------------------------------------------------------------------------
# Image and observation helpers
# -----------------------------------------------------------------------------


def uas_to_rad(uas: float) -> float:
    """Convert microarcseconds to radians."""
    return float(uas) * 1e-6 / 3600.0 * np.pi / 180.0


def _pil_resize_filter() -> int:
    """Return a high-quality Pillow resize filter across Pillow versions."""
    if hasattr(PILImage, "Resampling"):
        return PILImage.Resampling.LANCZOS
    return PILImage.LANCZOS


def observation_metadata(obs, config: ReconstructionConfig) -> Tuple[float, float, float, int, str]:
    """Pull image metadata from an Obsdata object when available."""
    ra = getattr(obs, "ra", config.ra)
    dec = getattr(obs, "dec", config.dec)
    rf = getattr(obs, "rf", config.rf_hz)
    mjd = getattr(obs, "mjd", config.mjd)
    source = getattr(obs, "source", "uvfits_observation")
    return ra, dec, rf, mjd, source


def image_file_to_ehtim_image(
    frame_path: Path,
    config: ReconstructionConfig,
    obs=None,
):
    """
    Convert a PNG/JPG frame into an ehtim Image template.

    The UVFITS data provide the actual observed visibilities. The image template
    controls the reconstruction grid, field of view, flux constraint, and output
    naming. Pixel values are normalized to config.total_flux_jy.
    """
    frame_path = Path(frame_path)

    with PILImage.open(frame_path) as pil_img:
        pil_img = pil_img.convert("L")
        pil_img = pil_img.resize((config.npix, config.npix), _pil_resize_filter())
        arr = np.asarray(pil_img, dtype=np.float64)

    arr -= np.nanmin(arr)

    max_val = np.nanmax(arr)
    if max_val > 0:
        arr /= max_val

    if np.nansum(arr) <= 0:
        raise ValueError(f"Frame appears blank after normalization: {frame_path}")

    arr = arr / np.nansum(arr) * float(config.total_flux_jy)

    fov_rad = uas_to_rad(config.fov_uas)
    psize = fov_rad / config.npix

    if obs is None:
        ra, dec, rf, mjd, source = (
            config.ra,
            config.dec,
            config.rf_hz,
            config.mjd,
            frame_path.stem,
        )
    else:
        ra, dec, rf, mjd, source = observation_metadata(obs, config)

    return eh.image.Image(
        arr,
        psize,
        ra,
        dec,
        rf=rf,
        source=f"{source}_{frame_path.stem}",
        mjd=mjd,
        polrep="stokes",
        pol_prim="I",
    )


def make_ehtim_image_from_array(
    x_true: np.ndarray,
    config: ReconstructionConfig = DEFAULT_CONFIG,
    source: str = "synthetic_template",
):
    """Convert a 2D numpy array into an ehtim Image template."""
    arr = np.asarray(x_true, dtype=np.float64).squeeze()

    if arr.ndim != 2:
        raise ValueError(f"x_true must be 2D, got shape {arr.shape}")

    arr = np.clip(arr, 0.0, None)
    total = arr.sum()
    if total <= 0:
        raise ValueError("Input image has zero total intensity after clipping.")

    arr = arr / total * float(config.total_flux_jy)
    ydim, xdim = arr.shape

    if xdim != ydim:
        raise ValueError(f"Expected a square image, got shape {arr.shape}")

    fov_rad = uas_to_rad(config.fov_uas)
    psize = fov_rad / xdim

    return eh.image.Image(
        arr,
        psize,
        config.ra,
        config.dec,
        rf=config.rf_hz,
        source=source,
        mjd=config.mjd,
        polrep="stokes",
        pol_prim="I",
    )


def load_uvfits_observation(path: Path, config: ReconstructionConfig):
    """Load an observed-data UVFITS file as an ehtim Obsdata object."""
    LOGGER.info("Loading observed data from UVFITS: %s", path)

    obs = eh.obsdata.load_uvfits(
        str(path),
        flipbl=config.flipbl,
        remove_nan=config.remove_nan,
        force_singlepol=config.force_singlepol,
        polrep=config.polrep,
        allow_singlepol=config.allow_singlepol,
        ignore_pzero_date=config.ignore_pzero_date,
    )

    if config.avg_coherent_sec is not None:
        obs = obs.avg_coherent(config.avg_coherent_sec)

    return obs


def copy_observation(obs):
    """Return a safe copy of an ehtim Obsdata object."""
    try:
        return obs.copy()
    except Exception:
        return deepcopy(obs)


def get_observation_for_frame(
    frame_index: int,
    uvfits_paths: Sequence[Path],
    config: ReconstructionConfig,
    cache: Dict[Path, object],
):
    """Load/copy the UVFITS observation assigned to the current frame."""
    uvfits_path = uvfits_paths[0] if len(uvfits_paths) == 1 else uvfits_paths[frame_index]
    uvfits_path = Path(uvfits_path)

    if uvfits_path not in cache:
        cache[uvfits_path] = load_uvfits_observation(uvfits_path, config)

    return copy_observation(cache[uvfits_path]), uvfits_path


# -----------------------------------------------------------------------------
# Plotting and saving
# -----------------------------------------------------------------------------


def save_uv_coverage(obs, output_path: Path, conj: bool = True, units: str = "Glambda") -> None:
    """Save a uv-coverage plot from an ehtim Obsdata object."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    u = np.asarray(obs.data["u"])
    v = np.asarray(obs.data["v"])

    units_lower = units.lower()
    if units_lower in {"glambda", "giga", "g"}:
        scale = 1e9
        axis_label = r"$u,v$ (G$\lambda$)"
    elif units_lower in {"mlambda", "mega", "m"}:
        scale = 1e6
        axis_label = r"$u,v$ (M$\lambda$)"
    else:
        scale = 1.0
        axis_label = r"$u,v$ ($\lambda$)"

    u_plot = u / scale
    v_plot = v / scale

    plt.figure(figsize=(6, 6))
    plt.scatter(u_plot, v_plot, s=8, alpha=0.7, label="Observed uv points")

    if conj:
        plt.scatter(-u_plot, -v_plot, s=8, alpha=0.7, label="Conjugate points")

    max_abs = max(
        float(np.max(np.abs(u_plot))) if len(u_plot) else 1.0,
        float(np.max(np.abs(v_plot))) if len(v_plot) else 1.0,
    )

    plt.xlim(max_abs, -max_abs)  # reversed x-axis, common for ehtim-style plots
    plt.ylim(-max_abs, max_abs)
    plt.xlabel(axis_label)
    plt.ylabel(axis_label)
    plt.title("uv Coverage")
    plt.grid(True, alpha=0.3)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    LOGGER.info("Saved uv coverage plot: %s", output_path)


def ehtim_image_to_array(ehtim_image) -> np.ndarray:
    """Convert an ehtim Image object to a 2D numpy array."""
    return ehtim_image.imvec.reshape(ehtim_image.ydim, ehtim_image.xdim)


def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Normalize a numeric image array to uint8 grayscale [0, 255]."""
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr - np.nanmin(arr)

    max_val = np.nanmax(arr)
    if max_val > 0:
        arr = arr / max_val

    return (255 * arr).clip(0, 255).astype(np.uint8)


def save_ehtim_grayscale_png(ehtim_image, output_path: Path, flip_vertical: bool = False) -> None:
    """Save raw ehtim image intensity as a grayscale PNG."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    arr = ehtim_image_to_array(ehtim_image)
    if flip_vertical:
        arr = np.flipud(arr)

    PILImage.fromarray(normalize_to_uint8(arr), mode="L").save(output_path)


def save_ehtim_image_display(ehtim_image, output_path: Path) -> None:
    """Save ehtim's display rendering of an image."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ehtim_image.display()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close("all")


def _first_existing_field(field_names: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    for candidate in candidates:
        if candidate in field_names:
            return candidate
    return None


def save_observation_csv(obs, output_path: Path) -> None:
    """Save observed visibility samples from an Obsdata object to a CSV file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    field_names = tuple(obs.data.dtype.names or ())
    vis_field = _first_existing_field(field_names, ("vis", "rrvis", "llvis"))
    sigma_field = _first_existing_field(field_names, ("sigma", "rrsigma", "llsigma"))

    if vis_field is None:
        LOGGER.warning("Skipping %s because no visibility field was found.", output_path)
        return

    columns = ["u", "v", "vis_real", "vis_imag", "amp", "phase_rad"]
    if sigma_field is not None:
        columns.append("sigma")
    for optional_field in ("time", "tint", "t1", "t2"):
        if optional_field in field_names:
            columns.append(optional_field)

    with open(output_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(columns)

        for row in obs.data:
            vis = row[vis_field]
            csv_row = [
                row["u"],
                row["v"],
                np.real(vis),
                np.imag(vis),
                np.abs(vis),
                np.angle(vis),
            ]

            if sigma_field is not None:
                csv_row.append(row[sigma_field])

            for optional_field in ("time", "tint", "t1", "t2"):
                if optional_field in field_names:
                    csv_row.append(row[optional_field])

            writer.writerow(csv_row)

    LOGGER.info("Saved observed Fourier CSV: %s", output_path)


def zip_folder(folder_path: Path, zip_path: Path) -> Path:
    """Zip every file under folder_path."""
    folder_path = Path(folder_path)
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in sorted(folder_path.rglob("*")):
            if file_path.is_file():
                archive.write(file_path, file_path.relative_to(folder_path))

    return zip_path


# -----------------------------------------------------------------------------
# Reconstruction
# -----------------------------------------------------------------------------


def reconstruct_frame(
    obs,
    template_image,
    config: ReconstructionConfig,
    previous_recon=None,
    frame_index: int = 0,
):
    """Reconstruct one frame using observed UVFITS data and L1 + TV regularization."""
    flux = float(template_image.total_flux())
    fov_recon = float(template_image.fovx())

    empty = eh.image.make_square(obs, config.npix, fov_recon)

    if previous_recon is None:
        init = empty.add_gauss(flux, (fov_recon / 2.0, fov_recon / 2.0, 0.0, 0.0, 0.0))
        maxit = config.maxit_first_frame
    else:
        try:
            init = previous_recon.blur_circ(fov_recon / 50.0)
        except Exception:
            init = deepcopy(previous_recon)
        maxit = config.maxit_later_frames

    imager = eh.imager.Imager(
        obs,
        init,
        prior_im=init,
        flux=flux,
        data_term=config.data_term,
        reg_term={"l1": config.lambda_l1, "tv": config.lambda_tv},
        maxit=maxit,
        ttype=config.ttype,
        norm_reg=True,
        epsilon_tv=1.0e-10,
    )

    return imager.make_image_I(show_updates=False)


def prepare_output_dirs(config: ReconstructionConfig) -> None:
    """Create output folders, clearing frame outputs when requested."""
    config.output_dir.mkdir(parents=True, exist_ok=True)

    if config.clear_output_dirs:
        safe_clear_dir(config.recon_display_dir)
        safe_clear_dir(config.recon_gray_dir)
        safe_clear_dir(config.fourier_csv_dir)
    else:
        config.recon_display_dir.mkdir(parents=True, exist_ok=True)
        config.recon_gray_dir.mkdir(parents=True, exist_ok=True)
        config.fourier_csv_dir.mkdir(parents=True, exist_ok=True)


def log_observation_summary(obs, uvfits_path: Path) -> None:
    """Print a compact summary of the loaded observation."""
    data = obs.data
    LOGGER.info("Observed data: %s", uvfits_path.name)
    LOGGER.info("  Number of data points: %d", len(data))

    if len(data) == 0:
        return

    LOGGER.info("  u range: %.6e to %.6e", np.nanmin(data["u"]), np.nanmax(data["u"]))
    LOGGER.info("  v range: %.6e to %.6e", np.nanmin(data["v"]), np.nanmax(data["v"]))

    if "time" in data.dtype.names:
        LOGGER.info("  Unique times: %d", len(set(data["time"])))


def run_reconstruction(config: ReconstructionConfig) -> Path:
    """Run the full reconstruction pipeline."""
    prepare_output_dirs(config)

    with tempfile.TemporaryDirectory(prefix="ehtim_reconstruction_") as temp_name:
        temp_dir = Path(temp_name)

        frame_paths = collect_input_files(
            config.frame_input,
            suffixes=IMAGE_SUFFIXES,
            extract_dir=temp_dir / "frames",
            description="image frame",
        )
        uvfits_paths = collect_input_files(
            config.uvfits_input,
            suffixes=UVFITS_SUFFIXES,
            extract_dir=temp_dir / "uvfits",
            description="UVFITS observation",
        )
        validate_observation_count(frame_paths, uvfits_paths)

        LOGGER.info("Found %d frame(s).", len(frame_paths))
        LOGGER.info("Found %d UVFITS observation file(s).", len(uvfits_paths))

        obs_cache: Dict[Path, object] = {}
        previous_recon = None

        for frame_index, frame_path in enumerate(frame_paths):
            LOGGER.info("%s", "=" * 72)
            LOGGER.info(
                "Processing frame %d/%d: %s",
                frame_index + 1,
                len(frame_paths),
                frame_path.name,
            )

            obs, uvfits_path = get_observation_for_frame(
                frame_index=frame_index,
                uvfits_paths=uvfits_paths,
                config=config,
                cache=obs_cache,
            )
            log_observation_summary(obs, uvfits_path)

            template_image = image_file_to_ehtim_image(frame_path, config=config, obs=obs)

            if config.save_uv_coverage and frame_index == 0:
                save_uv_coverage(
                    obs,
                    config.output_dir / "uv_coverage_first_frame.png",
                    units=config.uv_coverage_units,
                )

            if config.save_observation_csv:
                save_observation_csv(
                    obs,
                    config.fourier_csv_dir / f"observed_{frame_index:04d}_{frame_path.stem}.csv",
                )

            recon = reconstruct_frame(
                obs=obs,
                template_image=template_image,
                config=config,
                previous_recon=(
                    previous_recon if config.use_previous_frame_as_prior else None
                ),
                frame_index=frame_index,
            )

            output_name = f"recon_{frame_index:04d}_{frame_path.stem}.png"
            save_ehtim_image_display(recon, config.recon_display_dir / output_name)
            save_ehtim_grayscale_png(recon, config.recon_gray_dir / output_name)

            previous_recon = recon

    output_zip = zip_folder(config.recon_gray_dir, config.output_zip)

    LOGGER.info("%s", "=" * 72)
    LOGGER.info("Done.")
    LOGGER.info("Display frames: %s", config.recon_display_dir)
    LOGGER.info("Grayscale frames: %s", config.recon_gray_dir)
    LOGGER.info("Output zip: %s", output_zip)

    return output_zip


# -----------------------------------------------------------------------------
# Optional adapter for cv_regularization.py-style workflows
# -----------------------------------------------------------------------------

OBS_CACHE: Dict[Tuple[str, int], object] = {}


def reconstruct_image_for_cv(
    x_true: np.ndarray,
    lambda_l1: float = 0.0,
    lambda_tv: float = 0.0,
    lambda_w: float = 0.0,
    seed: Optional[int] = None,
    image_index: int = 0,
    fold_index: int = 0,
    uvfits_path: Optional[Path] = None,
    config: ReconstructionConfig = DEFAULT_CONFIG,
    obs_custom=None,
    return_ehtim_image: bool = False,
    **kwargs,
):
    """
    Adapter used by cv_regularization.py.

    Unlike the old version, this does not synthesize an observation with
    Image.observe(). It loads observed data from uvfits_path, or from
    config.uvfits_input when uvfits_path is not provided.
    """
    del lambda_w, seed, fold_index, kwargs  # accepted for backward compatibility

    cv_config = replace(config, lambda_l1=lambda_l1, lambda_tv=lambda_tv)
    template_image = make_ehtim_image_from_array(x_true, config=cv_config)

    if obs_custom is not None:
        obs = copy_observation(obs_custom)
    else:
        with tempfile.TemporaryDirectory(prefix="ehtim_cv_obs_") as temp_name:
            temp_dir = Path(temp_name)
            uvfits_paths = collect_input_files(
                Path(uvfits_path) if uvfits_path is not None else cv_config.uvfits_input,
                suffixes=UVFITS_SUFFIXES,
                extract_dir=temp_dir / "uvfits",
                description="UVFITS observation",
            )

            if len(uvfits_paths) not in {1}:
                if image_index >= len(uvfits_paths):
                    raise IndexError(
                        f"image_index={image_index} requested, but only "
                        f"{len(uvfits_paths)} UVFITS file(s) were found."
                    )
                selected_uvfits = uvfits_paths[image_index]
            else:
                selected_uvfits = uvfits_paths[0]

            cache_key = (str(selected_uvfits), int(image_index))
            if cache_key not in OBS_CACHE:
                OBS_CACHE[cache_key] = load_uvfits_observation(selected_uvfits, cv_config)

            obs = copy_observation(OBS_CACHE[cache_key])

    if image_index == 0 and cv_config.save_uv_coverage:
        save_uv_coverage(obs, cv_config.output_dir / "uv_coverage_frame_000.png")

    recon = reconstruct_frame(
        obs=obs,
        template_image=template_image,
        config=cv_config,
        previous_recon=None,
        frame_index=image_index,
    )

    if return_ehtim_image:
        return recon

    return ehtim_image_to_array(recon)


# -----------------------------------------------------------------------------
# Command-line interface
# -----------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct image frames using observed UVFITS data. The frame input "
            "may be a .zip file, a directory of PNG/JPG files, or a single image. "
            "The UVFITS input may be one file, a directory, or a .zip archive."
        )
    )
    parser.add_argument("--frame-input", type=Path, default=DEFAULT_CONFIG.frame_input)
    parser.add_argument("--uvfits-input", type=Path, default=DEFAULT_CONFIG.uvfits_input)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_CONFIG.output_dir)
    parser.add_argument("--npix", type=int, default=DEFAULT_CONFIG.npix)
    parser.add_argument("--fov-uas", type=float, default=DEFAULT_CONFIG.fov_uas)
    parser.add_argument("--total-flux-jy", type=float, default=DEFAULT_CONFIG.total_flux_jy)
    parser.add_argument("--ttype", choices=("direct", "nfft", "fast"), default=DEFAULT_CONFIG.ttype)
    parser.add_argument("--lambda-l1", type=float, default=DEFAULT_CONFIG.lambda_l1)
    parser.add_argument("--lambda-tv", type=float, default=DEFAULT_CONFIG.lambda_tv)
    parser.add_argument("--maxit-first-frame", type=int, default=DEFAULT_CONFIG.maxit_first_frame)
    parser.add_argument("--maxit-later-frames", type=int, default=DEFAULT_CONFIG.maxit_later_frames)
    parser.add_argument("--avg-coherent-sec", type=float, default=DEFAULT_CONFIG.avg_coherent_sec)
    parser.add_argument("--no-previous-prior", action="store_true")
    parser.add_argument("--no-uv-coverage", action="store_true")
    parser.add_argument("--no-observation-csv", action="store_true")
    parser.add_argument("--keep-output-dirs", action="store_true")
    parser.add_argument("--log-level", default=LOG_LEVEL, choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser


def config_from_args(args: argparse.Namespace) -> ReconstructionConfig:
    return replace(
        DEFAULT_CONFIG,
        frame_input=args.frame_input,
        uvfits_input=args.uvfits_input,
        output_dir=args.output_dir,
        npix=args.npix,
        fov_uas=args.fov_uas,
        total_flux_jy=args.total_flux_jy,
        ttype=args.ttype,
        lambda_l1=args.lambda_l1,
        lambda_tv=args.lambda_tv,
        maxit_first_frame=args.maxit_first_frame,
        maxit_later_frames=args.maxit_later_frames,
        avg_coherent_sec=args.avg_coherent_sec,
        use_previous_frame_as_prior=not args.no_previous_prior,
        save_uv_coverage=not args.no_uv_coverage,
        save_observation_csv=not args.no_observation_csv,
        clear_output_dirs=not args.keep_output_dirs,
    )


def main(argv: Optional[Sequence[str]] = None) -> Path:
    # With USE_COMMAND_LINE_OVERRIDES=False, running this file uses only the
    # USER SETTINGS block above. This also avoids issues from extra IDE/Jupyter
    # arguments that argparse does not recognize.
    if argv is None and not USE_COMMAND_LINE_OVERRIDES:
        argv = []

    parser = build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(message)s")
    config = config_from_args(args)

    return run_reconstruction(config)


if __name__ == "__main__":
    main()
