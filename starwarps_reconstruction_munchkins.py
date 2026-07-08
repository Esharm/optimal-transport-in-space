# %%
from __future__ import division
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import ehtim as eh
import scipy.optimize as opt
import os, copy
from pathlib import Path
from PIL import Image as PILImage
import importlib
import ehtim.imaging.dynamical_imaging as di
import ehtim.image as image
from ehtim.imaging import starwarps as sw
importlib.reload(di)
plt.close('all')

ttype = 'direct'
outpath = './tutorial_results/blackhole'
if not os.path.exists(os.path.dirname(outpath)):
    os.makedirs(os.path.dirname(outpath))
# %%
# Load the image and the telescope array
folder = Path('/Users/emma.jia/Desktop/space_imaging/optimal-transport-in-space/blackhole_sim/data/aart_frames')

ra  = 17.7611225   # hours
dec = -29.0078   # degrees

target_flux = 0.7

im = []
for path in sorted(folder.glob('*.png')):
    pil_img = PILImage.open(path).convert('L')
    arr = np.array(pil_img, dtype=float)
    arr = arr / arr.sum() * target_flux
    fov = 100 * eh.RADPERUAS
    eht_img = eh.image.Image(arr, fov / arr.shape[0], ra, dec, rf=226e9)
    im.append(eht_img)

eht = eh.array.load_txt('/Users/emma.jia/Desktop/space_imaging/EHT2017.txt')
print(len(eht.tarr), 'stations:', eht.tarr['site'])
# %%
# simulation parameters
tint_sec = 10
tadv_sec = 10
bw_hz = 2e9
n_frames = 15

t_transit = ra  # 17.7611225 hrs

frame_duration_sec = 120
gap_sec = 0 # or your gap
slot_sec = frame_duration_sec + gap_sec

# Center the whole observation block around transit
t_start = t_transit - (n_frames * slot_sec / 3600) / 2

total_sec = n_frames * slot_sec
total_hrs = total_sec / 3600
# %%
obs_list = []
valid_images = []

for i in range(n_frames):
    t0 = t_start + (i * slot_sec) / 3600                          # start of this frame's observation window
    t1 = t_start + (i * slot_sec + frame_duration_sec) / 3600     # 20 seconds later

    frame_idx = int((i + 0.5) * len(im) / n_frames)
    image_frame = im[frame_idx]
    try:
        obs = image_frame.observe(
            eht, tint_sec, tadv_sec, t0, t1, bw_hz,
            sgrscat=False, ampcal=True, phasecal=True, ttype='direct'
        )
        obs_list.append(obs)
        valid_images.append(image_frame)
    except Exception as e:
        print(f"Frame {i} skipped: {e}")

# %%
obs_outpath = './tutorial_results/blackhole/observations'
os.makedirs(obs_outpath, exist_ok=True)

for i, obs in enumerate(obs_list):
    obs.save_uvfits(f'{obs_outpath}/frame_{i:02d}.uvfits')
# %%
obs_outpath = './tutorial_results/blackhole/observations2'
os.makedirs(obs_outpath, exist_ok=True)

for i, obs in enumerate(obs_list):
    np.savez(f'{obs_outpath}/frame_{i:02d}.npz', data=obs.data)
# %%
NPIX = 64
npixels = NPIX**2

def load_png_as_prior(png_path, npix, fov, ra, dec, rf, flux=None):
    # Load and convert to grayscale
    img = PILImage.open(png_path).convert('L')
    img = img.resize((npix, npix), PILImage.LANCZOS)
    arr = np.array(img, dtype=np.float64)

    # Flip vertically if needed to match ehtim's image convention (north up)
    arr = np.flipud(arr)

    # Normalize to sum to 1, then scale to target flux later (or now)
    arr = arr / arr.sum()

    psize = fov / npix  # radians per pixel

    prior = eh.image.Image(arr, psize, ra, dec, rf=rf)

    if flux is not None:
        prior.imvec = prior.imvec / prior.imvec.sum() * flux

    return prior

# Usage:
prior = load_png_as_prior(
    "/Users/emma.jia/Desktop/space_imaging/optimal-transport-in-space/radial_outputs/time_avg_static_recon_128pix_radial_round.png",
    npix=64,
    fov=fov,
    ra=obs_list[0].ra,
    dec=obs_list[0].dec,
    rf=226e9,
    flux=target_flux
)

static_results = []

for i, obs in enumerate(obs_list):
    print(f"Static imaging frame {i+1}/{n_frames}...")
    
    imager = eh.imager.Imager(
        obs, prior, prior,
        flux=target_flux,
        d1='vis',    alpha_d1=1,
        s1='tv2',    alpha_s1=1.0,
        s2='l1',     alpha_s2=0.1,   # l1 + tv2 combo works well for static
        maxit=200,
        ttype='direct'
    )
    imager.make_image_I(show_updates=False)
    static_results.append(imager.out_last())

avg_image = eh.image.Image(
    np.mean(np.array([im.imarr() for im in static_results]), axis=0),
    static_results[0].psize,
    ra, dec,
    rf=226e9
)

meanImg = [im.regrid_image(fov, NPIX) for im in static_results]
#meanImg = [custom_prior_img.copy()]
# %%
for m in meanImg:
    floor = 1e-6 * m.imvec.max()
    m.imvec = np.maximum(m.imvec, floor)

imCov = [sw.gaussImgCovariance_2(m, powerDropoff=2.0, frac=4) for m in meanImg]
eps_list = [1e-6 * np.max(C) for C in imCov]
imCov = [C + eps * np.eye(npixels) for C, eps in zip(imCov, eps_list)]

pixel_var_ref = (target_flux / npixels) ** 2
variance_img_diff = 1e-5       # tune this: try 1e-7 (smooth) to 1e-5 (variable)
noiseCov_img = np.eye(npixels) * variance_img_diff

# ---- Motion basis: affine warp without translation ----
# Returns the pixel grid coordinates and the parameterised flow basis
init_x, init_y, flowbasis_x, flowbasis_y, initTheta = \
    sw.affineMotionBasis_noTranslation(meanImg[0])
# %%
warp_method    = 'phase'
measurement    = {'vis': 1}      # can also try {'amp':1, 'cphase':1}
interiorPriors = True            # use interior (Kalman-smoother) priors
numLinIters = 3               # linearised iterations per E-step (>1 for non-linear data)
nIters    = 8             # number of EM iterations
reassign_apxImgs = False         # if True, recompute linearisation points each EM iter

# L-BFGS-B options for the M-step
NHIST  = 50
stop   = 1e-10
maxit  = 200
optdict = {'maxiter': maxit, 'ftol': stop, 'maxcor': NHIST, 'disp': True}

# Bounds on each warp-basis coefficient (keeps the warp physically reasonable)
nbasis = flowbasis_x.shape[2]
bnds = [(-1.5, 1.5) for _ in range(nbasis)]
# %%
print("Running StarWarps with no motion model...")
expVal_t_nm, expVal_t_t_nm, _, loglike_nm, apxImgs_nm = sw.computeSuffStatistics(
    meanImg, imCov, obs_list,
    noiseCov_img, initTheta,          # initTheta = "no motion"
    init_x, init_y, flowbasis_x, flowbasis_y, initTheta,
    method=warp_method,
    measurement=measurement,
    interiorPriors=interiorPriors,
    numLinIters=numLinIters,
    compute_expVal_tm1_t=False        # skip cross-covariance — not needed here
)
print(f"No-motion log-likelihood: {loglike_nm[2]:.3f}")

# Save no-motion movie
sw.movie(expVal_t_nm, out='starwarps_nomotion.mp4', fps=5)
# %%
# ============================================================
# General rotation-only StarWarps pipeline
# ============================================================

init_images = [im.regrid_image(fov, NPIX) for im in static_results]

def theta_of_phi(phi):
    a, b = np.cos(phi), np.sin(phi)
    return np.array([a, b, -b, a])

def neg_loglik_phi(phi, *args):
    return sw.expnegloglikelihood(theta_of_phi(phi[0]), *args)

def deriv_neg_loglik_phi(phi, *args):
    theta = theta_of_phi(phi[0])
    g_theta = sw.deriv_expnegloglikelihood(theta, *args)
    a, b = theta[0], theta[1]
    dtheta_dphi = np.array([-b, a, -a, -b])
    return np.array([np.dot(g_theta, dtheta_dphi)])

def estimate_warmstart_angle(meanImg, imCov, obs_list, noiseCov_img,
                              init_x, init_y, flowbasis_x, flowbasis_y, initTheta,
                              warp_method, measurement, interiorPriors, numLinIters,
                              init_images, angle_range_deg=(-40, 40), coarse_step_deg=5,
                              verbose=True):
    """
    Coarse scan over candidate rotation angles, one E-step each (no M-step),
    to find a non-identity starting point directly from the data.
    Avoids the identity-trap without requiring ground truth.
    """
    candidates = np.arange(angle_range_deg[0], angle_range_deg[1] + 1, coarse_step_deg)
    best_angle, best_ll = 0.0, -np.inf

    for ang in candidates:
        theta_try = theta_of_phi(np.deg2rad(ang))
        _, _, _, ll, _ = sw.computeSuffStatistics(
            meanImg, imCov, obs_list, noiseCov_img, theta_try,
            init_x, init_y, flowbasis_x, flowbasis_y, initTheta,
            method=warp_method, measurement=measurement,
            interiorPriors=interiorPriors, numLinIters=numLinIters,
            compute_expVal_tm1_t=False, init_images=init_images
        )
        if verbose:
            print(f"  scan angle={ang:+4.0f} deg  loglikelihood={ll[2]:.4f}")
        if ll[2] > best_ll:
            best_ll, best_angle = ll[2], ang

    if verbose:
        print(f"Best warm-start angle: {best_angle} deg (loglikelihood={best_ll:.4f})")
    return best_angle

def run_rotation_starwarps(meanImg, imCov, obs_list, noiseCov_img,
                            init_x, init_y, flowbasis_x, flowbasis_y, initTheta,
                            warp_method, measurement, interiorPriors, numLinIters,
                            init_images, optdict,
                            init_angle_deg=None, trust_region_deg=5, n_em_iters=15,
                            angle_range_deg=(-40, 40), coarse_step_deg=5,
                            movie_out=None, movie_fps=5):
    """
    Full rotation-only EM loop with trust-region M-step.
    If init_angle_deg is None, runs a coarse data-driven scan to pick a warm start.
    Returns: expVal_t (final reconstruction), negll, angles (per-iteration history)
    """
    if init_angle_deg is None:
        print("No warm-start angle given — running coarse scan over the data...")
        init_angle_deg = estimate_warmstart_angle(
            meanImg, imCov, obs_list, noiseCov_img,
            init_x, init_y, flowbasis_x, flowbasis_y, initTheta,
            warp_method, measurement, interiorPriors, numLinIters,
            init_images, angle_range_deg=angle_range_deg, coarse_step_deg=coarse_step_deg
        )

    newPhi = np.array([np.deg2rad(init_angle_deg)])
    newTheta = theta_of_phi(newPhi[0])

    negll, angles = [], []
    expVal_t = None

    for em_iter in range(n_em_iters + 1):
        print(f"\n===== EM iteration {em_iter}/{n_em_iters} =====")

        expVal_t, expVal_t_t, expVal_tm1_t, loglikelihood, apxImgs = sw.computeSuffStatistics(
            meanImg, imCov, obs_list, noiseCov_img, newTheta,
            init_x, init_y, flowbasis_x, flowbasis_y, initTheta,
            method=warp_method, measurement=measurement,
            interiorPriors=interiorPriors, numLinIters=numLinIters,
            compute_expVal_tm1_t=True, init_images=init_images
        )
        negll.append(-loglikelihood[2])
        angles.append(np.rad2deg(newPhi[0]))
        print(f"  loglikelihood = {loglikelihood[2]:.4f}, angle = {angles[-1]:.2f} deg")

        if em_iter < n_em_iters:
            args = (expVal_t, expVal_t_t, expVal_tm1_t, meanImg, imCov, obs_list,
                    noiseCov_img, init_x, init_y, flowbasis_x, flowbasis_y, initTheta, warp_method)

            local_bounds = [(newPhi[0] - np.deg2rad(trust_region_deg),
                              newPhi[0] + np.deg2rad(trust_region_deg))]

            result = opt.minimize(
                neg_loglik_phi, newPhi, args=args,
                method='L-BFGS-B', jac=deriv_neg_loglik_phi,
                bounds=local_bounds, options=optdict
            )
            newPhi = result.x
            newTheta = theta_of_phi(newPhi[0])
            print(f"  M-step converged: {result.success}  fun={result.fun:.4f}  new angle={np.rad2deg(newPhi[0]):.2f} deg")
    if movie_out is not None:
        try:
            sw.movie(expVal_t, out=movie_out, fps=movie_fps)
            print(f"Saved reconstruction movie to {movie_out}")
        except Exception as e:
            print(f"Movie save failed: {e}")

    return expVal_t, negll, angles
# %%
expVal_t, negll, angles = run_rotation_starwarps(
    meanImg, imCov, obs_list, noiseCov_img,
    init_x, init_y, flowbasis_x, flowbasis_y, initTheta,
    warp_method, measurement, interiorPriors, numLinIters,
    init_images, optdict, init_angle_deg=0, 
    trust_region_deg=5, n_em_iters=20, movie_out='starwarps_results/final_reconstruction_21.mp4', movie_fps=5
)

# ============================================================
# Convergence plot
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
ax1.plot(negll, 'o-')
ax1.set_xlabel('EM Iteration'); ax1.set_ylabel('Neg Log-Likelihood')
ax1.set_title('Convergence')
ax2.plot(angles, 'o-', color='tab:orange')
ax2.set_xlabel('EM Iteration'); ax2.set_ylabel('Angle (deg)')
ax2.set_title('Learned rotation angle')
plt.tight_layout()
plt.savefig('starwarps_convergence_rotation8_64.png', dpi=150)
plt.show()

# ============================================================
# Final comparison plot (reuses your existing plotting code)
# ============================================================
vmax = max(orig.regrid_image(fov, NPIX).imarr().max() for orig in valid_images)

fig, axes = plt.subplots(2, n_frames, figsize=(3 * n_frames, 6))
for i, (orig, recon) in enumerate(zip(valid_images, expVal_t)):
    orig_rg = orig.regrid_image(fov, NPIX)
    axes[0, i].imshow(orig_rg.imarr(), cmap='hot', origin='lower', vmin=0, vmax=vmax)
    axes[1, i].imshow(np.clip(recon.imarr(), 0, None), cmap='hot', origin='lower', vmin=0, vmax=vmax)
    axes[0, i].set_title(f'Original {i}')
    chisq = obs_list[i].chisq(recon, dtype='vis', ttype='direct')
    axes[1, i].set_title(f'χ²={chisq:.3f}')
    for ax in axes[:, i]:
        ax.axis('off')

axes[0, 0].set_ylabel('Original')
axes[1, 0].set_ylabel('StarWarps')
plt.tight_layout()
plt.savefig('starwarps_comparison_avgimage_rotation8_64.png', dpi=150)
plt.show()
# %%
import os
from PIL import Image as PILImage
import numpy as np

out_dir = './tutorial_results/blackhole/frames_8_64_png'
os.makedirs(out_dir, exist_ok=True)

for i, recon in enumerate(expVal_t):
    arr = recon.imarr()
    
    # Normalize to 0–255
    arr_norm = arr - arr.min()
    if arr_norm.max() > 0:
        arr_norm = arr_norm / arr_norm.max()
    arr_uint8 = (arr_norm * 255).astype(np.uint8)
    
    # Flip vertically (ehtim uses lower-left origin)
    arr_uint8 = np.flipud(arr_uint8)
    
    img = PILImage.fromarray(arr_uint8, mode='L')  # 'L' = grayscale
    img.save(f'{out_dir}/frame_8_64_{i:02d}.png')
    print(f"Saved frame {i}")

print(f"All frames saved to {out_dir}")
# %%
npy_outdir = 'starwarps_results/final_frames_8_64_npy'
os.makedirs(npy_outdir, exist_ok=True)

for i, recon in enumerate(expVal_t):
    arr2d = recon.imarr()  # 2D (NPIX x NPIX) array, same as used in your plots
    np.save(f'{npy_outdir}/frame_8_64_{i:02d}.npy', arr2d)

print(f"Saved {len(expVal_t)} frames to {npy_outdir}")
# %%
import os
from PIL import Image

def resize_folder(in_dir, out_dir, size=(32, 32), resample=Image.LANCZOS):
    os.makedirs(out_dir, exist_ok=True)
    for fname in sorted(os.listdir(in_dir)):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')):
            continue
        img = Image.open(os.path.join(in_dir, fname))
        img_resized = img.resize(size, resample)
        img_resized.save(os.path.join(out_dir, fname))
    print(f"Resized images from {in_dir} -> {out_dir}")

resize_folder('/Users/emma.jia/Desktop/space_imaging/optimal-transport-in-space/blackhole_sim/data/aart_frames', '/Users/emma.jia/Desktop/space_imaging/optimal-transport-in-space/blackhole_sim/data/aart_frames_32')
# %%
import os
from PIL import Image

def flip_folder(in_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for fname in sorted(os.listdir(in_dir)):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')):
            continue
        img = Image.open(os.path.join(in_dir, fname))
        img_flipped = img.transpose(Image.FLIP_TOP_BOTTOM)
        img_flipped.save(os.path.join(out_dir, fname))
    print(f"Flipped images from {in_dir} -> {out_dir}")

flip_folder('/Users/emma.jia/Desktop/space_imaging/optimal-transport-in-space/tutorial_results/blackhole/frames_8_64_png', '/Users/emma.jia/Desktop/space_imaging/optimal-transport-in-space/tutorial_results/blackhole/frames_8_64_flipped')
# %%
import os
import numpy as np

def flip_npy_folder(in_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for fname in sorted(os.listdir(in_dir)):
        if not fname.lower().endswith('.npy'):
            continue
        arr = np.load(os.path.join(in_dir, fname))
        arr_flipped = np.flipud(arr)   # flips along axis 0 (rows) = vertical/x-axis flip
        np.save(os.path.join(out_dir, fname), arr_flipped)
    print(f"Flipped arrays from {in_dir} -> {out_dir}")

flip_npy_folder('/Users/emma.jia/Desktop/space_imaging/optimal-transport-in-space/starwarps_results/final_frames_8_64_npy', '/Users/emma.jia/Desktop/space_imaging/optimal-transport-in-space/starwarps_results/final_frames_8_64_npy_flipped')
# %%
import os
import numpy as np
from PIL import Image as PILImage

out_dir = 'static_results_png'
os.makedirs(out_dir, exist_ok=True)

for i, im in enumerate(static_results):
    arr = im.imarr()
    arr_norm = arr - arr.min()
    if arr_norm.max() > 0:
        arr_norm = arr_norm / arr_norm.max()
    arr_uint8 = (arr_norm * 255).astype(np.uint8)
    PILImage.fromarray(arr_uint8, mode='L').save(f'{out_dir}/static_{i:02d}.png')

print(f"Saved {len(static_results)} frames to {out_dir}")
# %%
