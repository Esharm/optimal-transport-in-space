from __future__ import division, print_function

import os
import zipfile
import shutil
import tempfile
from pathlib import Path
from copy import deepcopy

import numpy as np
import matplotlib
import pandas as pd
from skimage.metrics import structural_similarity as ssim

# Use non-interactive backend because WSL often cannot show windows
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from PIL import Image as PILImage

import ehtim as eh
plt.close('all')

# --------------------------------------------------------------
# User Settings
# --------------------------------------------------------------
# Fourier transform type - change this to 'direct' if 'nfft' is not installed!!!
ttype = 'direct'

#Base Directory
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

#Input
INPUT_ZIP = BASE_DIR / "Munchkin Frames (real).zip"

OUTPUT_DIR = BASE_DIR / "Munchkin_Test_Results"
RECON_DIR = OUTPUT_DIR / "reconstructed_frames_test"
FOURIER_DIR = OUTPUT_DIR / "fourier_test_csv"

OUTPUT_ZIP = OUTPUT_DIR / "munchkin_reconstructed_frames_output.zip"

# Image/reconstruction settings
NPIX = 64                 
FOV_UAS = 160.0
TOTAL_FLUX_JY = 1.0

# Sampling settings
NUM_SAMPLES = 250
UV_MAX = 5e9              # wavelengths
RANDOM_SEED = 0
INCLUDE_CONJUGATES = True

# Noise settings
ADD_NOISE = True
NOISE_FRAC = 0.1         # 5% of max visibility amplitude

# Reconstruction settings
USE_PREVIOUS_FRAME_AS_PRIOR = True

DATA_TERM = {"vis": 1}
REG_TERM = {"l1": 1, "tv": 0.1}

MAXIT_FIRST_FRAME = 200
MAXIT_LATER_FRAMES = 100

SAVE_FOURIER_CSV = True

# ============================================================
# 1. Utility functions
# ============================================================

def safe_clear_dir(path):
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def extract_frames_from_zip(zip_path, extract_dir):
    """
    Extract image frames from a zip file into extract_dir.
    Returns sorted list of extracted image paths.
    """
    zip_path = Path(zip_path)
    extract_dir = Path(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    valid_suffixes = {".png", ".jpg", ".jpeg"}

    frame_paths = []

    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            suffix = Path(member).suffix.lower()

            if suffix not in valid_suffixes:
                continue

            # Avoid nested path issues by using only the filename
            filename = Path(member).name
            output_path = extract_dir / filename

            with zf.open(member) as src, open(output_path, "wb") as dst:
                shutil.copyfileobj(src, dst)

            frame_paths.append(output_path)

    frame_paths = sorted(frame_paths)

    if len(frame_paths) == 0:
        raise ValueError("No PNG/JPG frames found in the zip file.")

    return frame_paths

def make_uv_samples(num_samples, uv_max, seed=0, include_conjugates=True):
    """
    Create custom sparse uv samples in a disk.
    u and v are in wavelengths.
    """
    rng = np.random.default_rng(seed)

    if include_conjugates:
        half = num_samples // 2
    else:
        half = num_samples

    theta = rng.uniform(0, 2 * np.pi, half)
    radius = uv_max * np.sqrt(rng.uniform(0, 1, half))

    u = radius * np.cos(theta)
    v = radius * np.sin(theta)

    if include_conjugates:
        u = np.concatenate([u, -u])
        v = np.concatenate([v, -v])

    return u, v

def png_to_ehtim_image(frame_path, npix=64, fov_uas=160.0, total_flux_jy=1.0):
    """
    Convert a PNG/JPG frame into an ehtim Image.
    """
    frame_path = Path(frame_path)

    pil_img = PILImage.open(frame_path).convert("L")
    pil_img = pil_img.resize((npix, npix))

    arr = np.asarray(pil_img, dtype=float)

    arr -= arr.min()

    if arr.max() > 0:
        arr /= arr.max()

    if arr.sum() <= 0:
        raise ValueError(f"Frame appears blank: {frame_path}")

    arr = arr / arr.sum() * total_flux_jy

    RADPERUAS = np.pi / (180.0 * 3600.0 * 1e6)
    fov_rad = fov_uas * RADPERUAS
    psize = fov_rad / npix

    ra = 0.0
    dec = 0.0

    im = eh.image.Image(
        arr,
        psize,
        ra,
        dec,
        rf=230e9,
        source=frame_path.stem
    )

    return im

def sample_image_fourier(im, u, v):
    """
    Forward transform ehtim Image at custom uv points.
    Returns complex Stokes I visibilities.
    """
    uv = np.column_stack([u, v])

    vis_out = im.sample_uv(
        uv,
        polrep_obs="stokes",
        ttype="direct",
        verbose=False
    )

    if isinstance(vis_out, (list, tuple)):
        vis = np.asarray(vis_out[0])
    else:
        vis = np.asarray(vis_out)

    return vis

def add_complex_gaussian_noise(vis, noise_frac=0.05, seed=0):
    """
    Add circular complex Gaussian noise to visibility data.
    """
    rng = np.random.default_rng(seed)

    sigma_value = noise_frac * np.max(np.abs(vis))
    sigma_value = max(sigma_value, 1e-12)

    noise_real = rng.normal(
        loc=0.0,
        scale=sigma_value / np.sqrt(2),
        size=vis.shape
    )

    noise_imag = rng.normal(
        loc=0.0,
        scale=sigma_value / np.sqrt(2),
        size=vis.shape
    )

    noise = noise_real + 1j * noise_imag

    vis_noisy = vis + noise
    sigma = np.full(len(vis), sigma_value)

    return vis_noisy, sigma, noise

def make_custom_obs_from_uv(
    im,
    u,
    v,
    vis,
    sigma=None,
    tint=10.0,
    bw=4e9,
    source="PNG_Test",
    mjd=51544
):
    """
    Create an ehtim Obsdata object from custom Fourier samples.

    Parameters
    ----------
    im : ehtim.image.Image
        Original ehtim image object. Used for RA, Dec, RF metadata.
    u, v : arrays
        uv coordinates in wavelengths.
    vis : complex array
        Complex visibility samples at those uv points.
    sigma : array or float
        Visibility noise standard deviation. If None, uses 5% of max amplitude.
    tint : float
        Integration time in seconds. Mostly metadata here.
    bw : float
        Bandwidth in Hz. Mostly metadata here.
    source : str
        Source name.
    mjd : int
        Modified Julian date.

    Returns
    -------
    obs : ehtim.obsdata.Obsdata
        Custom observation object usable by ehtim Imager.
    """

    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    vis = np.asarray(vis, dtype=complex)

    n = len(u)

    if len(v) != n or len(vis) != n:
        raise ValueError("u, v, and vis must have the same length.")

    if sigma is None:
        sigma_value = 0.05 * np.max(np.abs(vis))
        sigma = np.full(n, sigma_value, dtype=float)
    else:
        sigma = np.asarray(sigma, dtype=float)
        if sigma.ndim == 0:
            sigma = np.full(n, float(sigma), dtype=float)

    if len(sigma) != n:
        raise ValueError("sigma must be either scalar or same length as u/v/vis.")

    # Avoid zero sigma, because chi-squared weighting divides by sigma.
    sigma = np.maximum(sigma, 1e-12)

    # --------------------------------------------------------
    # Dummy telescope table.
    # --------------------------------------------------------
    # We are not using real telescope geometry here.
    # We only need two site names so ehtim has a valid baseline label.
    tarr = np.array([
        ("SITE1", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 + 0.0j, 0.0 + 0.0j, 0.0, 0.0, 0.0),
        ("SITE2", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 + 0.0j, 0.0 + 0.0j, 0.0, 0.0, 0.0),
    ], dtype=eh.const_def.DTARR)

    # --------------------------------------------------------
    # Custom Stokes visibility table.
    # --------------------------------------------------------
    # DTPOL_STOKES fields are:
    # time, tint, t1, t2, tau1, tau2, u, v,
    # vis, qvis, uvis, vvis, sigma, qsigma, usigma, vsigma
    # --------------------------------------------------------

    datatable = np.empty(n, dtype=eh.const_def.DTPOL_STOKES)

    # Use increasing fake times in hours.
    # The actual values do not matter much for direct custom vis imaging.
    datatable["time"] = np.linspace(0.0, 1.0, n)
    datatable["tint"] = tint
    datatable["t1"] = "SITE1"
    datatable["t2"] = "SITE2"
    datatable["tau1"] = 0.0
    datatable["tau2"] = 0.0
    datatable["u"] = u
    datatable["v"] = v

    # Stokes I visibility data
    datatable["vis"] = vis

    # No polarization for this PNG toy problem
    datatable["qvis"] = 0.0 + 0.0j
    datatable["uvis"] = 0.0 + 0.0j
    datatable["vvis"] = 0.0 + 0.0j

    # Noise standard deviations
    datatable["sigma"] = sigma
    datatable["qsigma"] = 1e9
    datatable["usigma"] = 1e9
    datatable["vsigma"] = 1e9

    obs = eh.obsdata.Obsdata(
        im.ra,
        im.dec,
        im.rf,
        bw,
        datatable,
        tarr,
        polrep="stokes",
        source=source,
        mjd=mjd,
        timetype="UTC",
        ampcal=True,
        phasecal=True
    )

    return obs

def reconstruct_im(obs_custom, current_im, previous_recon=None, frame_index=0):
    """
    Reconstruct one frame using L1 + TV.
    If previous_recon is supplied, use it as the initialization/prior.
    """
    flux = current_im.total_flux()
    fov_recon = current_im.fovx()
    npix_recon = current_im.xdim

    empty = eh.image.make_square(obs_custom, npix_recon, fov_recon)

    if previous_recon is None:
        init = empty.add_gauss(
            flux,
            (fov_recon / 2.0, fov_recon / 2.0, 0, 0, 0)
        )
        prior = init
        maxit = MAXIT_FIRST_FRAME

    else:
        # Use previous reconstruction as warm start.
        # A slight blur makes it less brittle if the object moved.
        try:
            init = previous_recon.blur_circ(fov_recon / 50.0)
        except Exception:
            init = deepcopy(previous_recon)

        prior = init
        maxit = MAXIT_LATER_FRAMES

    imgr = eh.imager.Imager(
        obs_custom,
        init,
        prior_im=prior,
        flux=flux,
        data_term=DATA_TERM,
        reg_term=REG_TERM,
        maxit=maxit,
        ttype=ttype,
        norm_reg=True,
        epsilon_tv = 1.e-10
    )

    recon = imgr.make_image_I(show_updates=False)

    return recon

def save_ehtim_image_display(im, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    im.display()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close("all")

def save_fourier_csv(output_path, u, v, vis_clean, vis_used, sigma):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    amp_clean = np.abs(vis_clean)
    phase_clean = np.angle(vis_clean)

    amp_used = np.abs(vis_used)
    phase_used = np.angle(vis_used)

    data = np.column_stack([
        u,
        v,
        vis_clean.real,
        vis_clean.imag,
        amp_clean,
        phase_clean,
        vis_used.real,
        vis_used.imag,
        amp_used,
        phase_used,
        sigma
    ])

    np.savetxt(
        output_path,
        data,
        delimiter=",",
        header=(
            "u,v,"
            "clean_real,clean_imag,clean_amp,clean_phase,"
            "used_real,used_imag,used_amp,used_phase,"
            "sigma"
        ),
        comments=""
    )

def zip_folder(folder_path, zip_path):
    folder_path = Path(folder_path)
    zip_path = Path(zip_path)

    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(folder_path.rglob("*")):
            if file_path.is_file():
                zf.write(file_path, file_path.relative_to(folder_path))

    return zip_path



# --------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------

def main():
    if not INPUT_ZIP.exists():
        raise FileNotFoundError(f"Could not find input zip: {INPUT_ZIP}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    safe_clear_dir(RECON_DIR)
    safe_clear_dir(FOURIER_DIR)

    temp_dir = Path(tempfile.mkdtemp(prefix="ehtim_frames_"))

    try:
        frame_dir = temp_dir / "frames"
        frame_paths = extract_frames_from_zip(INPUT_ZIP, frame_dir)

        print(f"Found {len(frame_paths)} frames.")

        u, v = make_uv_samples(
            NUM_SAMPLES,
            UV_MAX,
            seed=RANDOM_SEED,
            include_conjugates=INCLUDE_CONJUGATES
        )

        print(f"Using {len(u)} uv samples per frame.")

        previous_recon = None

        for i, frame_path in enumerate(frame_paths):
            print("=" * 60)
            print(f"Processing frame {i + 1}/{len(frame_paths)}: {frame_path.name}")

            # 1. Convert frame to ehtim image
            im = png_to_ehtim_image(
                frame_path,
                npix=NPIX,
                fov_uas=FOV_UAS,
                total_flux_jy=TOTAL_FLUX_JY
            )

            # 2. Forward transform at controlled uv samples
            vis_clean = sample_image_fourier(im, u, v)

            # 3. Add noise if desired
            if ADD_NOISE:
                vis_used, sigma, _ = add_complex_gaussian_noise(
                    vis_clean,
                    noise_frac=NOISE_FRAC,
                    seed=RANDOM_SEED + i
                )
            else:
                vis_used = vis_clean
                sigma_value = 0.05 * np.max(np.abs(vis_clean))
                sigma = np.full(len(vis_clean), max(sigma_value, 1e-12))

            # 4. Package custom Fourier data into Obsdata
            obs_custom = make_custom_obs_from_uv(
                im=im,
                u=u,
                v=v,
                vis=vis_used,
                sigma=sigma,
                source=frame_path.stem
            )

            # 5. Reconstruct
            if USE_PREVIOUS_FRAME_AS_PRIOR:
                recon = reconstruct_im(
                    obs_custom,
                    im,
                    previous_recon=previous_recon,
                    frame_index=i
                )
            else:
                recon = reconstruct_im(
                    obs_custom,
                    im,
                    previous_recon=None,
                    frame_index=i
                )

            # 6. Save reconstructed image
            output_name = f"recon_{i:04d}_{frame_path.stem}.png"
            save_ehtim_image_display(recon, RECON_DIR / output_name)

            # 7. Optional: save Fourier data
            if SAVE_FOURIER_CSV:
                csv_name = f"fourier_{i:04d}_{frame_path.stem}.csv"
                save_fourier_csv(
                    FOURIER_DIR / csv_name,
                    u,
                    v,
                    vis_clean,
                    vis_used,
                    sigma
                )

            # 8. Use current reconstruction as next prior
            previous_recon = recon

        # Zip reconstructed images
        zip_folder(RECON_DIR, OUTPUT_ZIP)

        print("=" * 60)
        print(f"Done.")
        print(f"Reconstructed frames saved in: {RECON_DIR}")
        print(f"Output zip saved to: {OUTPUT_ZIP}")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()

"""
# 7.5 Add complex Gaussian / thermal noise to Fourier data
# ============================================================

rng = np.random.default_rng(seed=1)

# Choose noise level.
# This is the standard deviation of the complex visibility noise amplitude.
# Example: 5% of the maximum visibility amplitude.
noise_frac = 0.05
sigma = noise_frac * np.max(np.abs(vis))

# For circular complex Gaussian noise:
# real noise ~ N(0, sigma^2 / 2)
# imag noise ~ N(0, sigma^2 / 2)
noise_real = rng.normal(loc=0.0, scale=sigma / np.sqrt(2), size=vis.shape)
noise_imag = rng.normal(loc=0.0, scale=sigma / np.sqrt(2), size=vis.shape)

complex_noise = noise_real + 1j * noise_imag

vis_noisy = vis + complex_noise

amp_noisy = np.abs(vis_noisy)
phase_noisy = np.angle(vis_noisy)

print("Added complex Gaussian noise.")
print("Noise sigma:", sigma)
print("First clean visibility:", vis[0])
print("First noisy visibility:", vis_noisy[0])



# If you created Gaussian noise earlier:
# vis_noisy = vis + complex_noise
# sigma = noise_sigma or an array of sigmas

obs_custom = make_custom_obs_from_uv(
    im=im,
    u=u,
    v=v,
    vis=vis_noisy,
    sigma=sigma,
    tint=10.0,
    bw=4e9,
    source="PNG_Test"
)

print(obs_custom)
print("Number of custom visibility samples:", len(obs_custom.data))

# 9. Plot uv coverage
# ============================================================

plt.figure()
plt.scatter(u / 1e9, v / 1e9, s=3)
plt.xlabel("u (Gλ)")
plt.ylabel("v (Gλ)")
plt.title("Sparse uv coverage")
plt.axis("equal")
plt.savefig(output_dir / "uv_coverage1.png", dpi=200, bbox_inches="tight")
plt.close()

print("Saved uv coverage plot.")


# ============================================================
# 10. Plot visibility amplitude vs uv radius
# ============================================================

uv_radius = np.sqrt(u**2 + v**2)

plt.figure()
plt.scatter(uv_radius / 1e9, amp_clean, s=3)
plt.xlabel("uv radius (Gλ)")
plt.ylabel("Visibility amplitude")
plt.title("Visibility amplitude vs uv radius")
plt.savefig(output_dir / "visibility_amplitude1.png", dpi=200, bbox_inches="tight")
plt.close()

print("Saved visibility amplitude plot.")

print("Done.")


# Plot clean vs noisy visibility amplitude
uv_radius = np.sqrt(u**2 + v**2)

plt.figure()
plt.scatter(uv_radius / 1e9, np.abs(vis), s=3, label="Clean")
plt.scatter(uv_radius / 1e9, np.abs(vis_noisy), s=3, label="Noisy", alpha=0.5)
plt.xlabel("uv radius (Gλ)")
plt.ylabel("Visibility amplitude")
plt.title("Clean vs Noisy Visibility Amplitude")
plt.legend()
plt.savefig(output_dir / "visibility_amplitude_clean_vs_noisy.png", dpi=200, bbox_inches="tight")
plt.close()

print("Saved clean vs noisy visibility amplitude plot.")
"""


#Next to-do list:
#Figure out adding noise to raw video frames all at once?
#Try observe function for sampling
#Implement NRMSE or SSIM
#Use visibility amplitude and closure phase
#Implement some way (CV maybe) to tune the optimization coefficients