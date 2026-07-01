# %%
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import ehtim as eh
import scipy
import scipy.optimize as opt
import time, os, copy
from pathlib import Path
from PIL import Image as PILImage
import importlib
import ehtim.imaging.dynamical_imaging as di
import ehtim.image as image
from ehtim.imaging import starwarps as sw   # <-- NEW IMPORT
from ehtim.imaging import patch_prior as pp
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

target_flux = 0.6  # Jy — typical M87* flux at 230 GHz (~2.4 Jy for Sgr A*)

im = []
for path in sorted(folder.glob('*.png')):
    pil_img = PILImage.open(path).convert('L')
    arr = np.array(pil_img, dtype=float)
    arr = arr / arr.sum() * target_flux
    fov = 100 * eh.RADPERUAS
    eht_img = eh.image.Image(arr, fov / arr.shape[0], ra, dec, rf=226e9)
    im.append(eht_img)

eht = eh.array.load_txt('/Users/emma.jia/Desktop/space_imaging/optimal-transport-in-space/EHTII.txt')
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
print(f"Array has {len(eht.tarr)} stations")
print(obs_list[0].tarr)

# After generating obs_list
for i, obs in enumerate(obs_list):
    n_vis = len(obs.data)
    baselines = set(zip(obs.data['t1'], obs.data['t2']))
    print(f"Frame {i}: {n_vis} visibilities, {len(baselines)} unique baselines")

# Plot uv-coverage for first frame
fig, ax = plt.subplots()
ax.scatter(obs_list[0].data['u']/1e9, obs_list[0].data['v']/1e9, s=1)
ax.scatter(-obs_list[0].data['u']/1e9, -obs_list[0].data['v']/1e9, s=1)  # conjugate
ax.set_xlabel('u (Gλ)'); ax.set_ylabel('v (Gλ)')
ax.set_title('uv-coverage frame 0')
plt.show()
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

fwhm_prior = 50 * eh.RADPERUAS
emptyprior = eh.image.make_square(obs_list[0], NPIX, fov)
gaussprior  = emptyprior.add_gauss(target_flux, (fwhm_prior, fwhm_prior, 0, 0, 0))

static_results = []
prior = eh.image.make_square(obs_list[0], npix=128, fov=fov)
prior.imvec += target_flux / prior.imvec.size  # uniform flux distribution
for i, obs in enumerate(obs_list):
    print(f"Static imaging frame {i+1}/{n_frames}...")
    
    imager = eh.imager.Imager(
        obs, prior, prior,
        flux=target_flux,
        d1='vis',    alpha_d1=100,
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
avg_rg = avg_image.regrid_image(fov, NPIX)
meanImg = [avg_rg.copy()]
# %%
# ---- Build the image covariance (spatial correlation of the prior) ----
pixel_var = 1e-6  # tune this
imCov   = [np.diag(np.full(npixels, pixel_var))]

# ---- Noise covariance: how much intensity change is allowed between frames ----
# Larger variance_img_diff -> more temporal variation allowed
variance_img_diff = 1e-3         # tune this: try 1e-7 (smooth) to 1e-5 (variable)
noiseCov_img = np.eye(npixels) * variance_img_diff

# ---- Motion basis: affine warp without translation ----
# Returns the pixel grid coordinates and the parameterised flow basis
init_x, init_y, flowbasis_x, flowbasis_y, initTheta = \
    sw.affineMotionBasis_noTranslation(meanImg[0])
# %%
warp_method    = 'phase'
measurement    = {'vis': 1}      # can also try {'amp':1, 'cphase':1}
interiorPriors = True            # use interior (Kalman-smoother) priors
numLinIters    = 3               # linearised iterations per E-step (>1 for non-linear data)
nIters         = 10              # number of EM iterations
reassign_apxImgs = False         # if True, recompute linearisation points each EM iter

# L-BFGS-B options for the M-step
NHIST  = 5000
stop   = 1e-10
maxit  = 4000
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
newTheta  = copy.deepcopy(initTheta)
apxImgs   = False    # first E-step uses the default init from the StarWarps paper
negll     = []
thetas    = []

init_images = [im.regrid_image(fov, NPIX) for im in static_results]

for em_iter in range(nIters + 1):
    print(f"\n===== EM iteration {em_iter}/{nIters} =====")
    # ---------- E-step ----------
    # Compute sufficient statistics (expected image at each frame, and cross-frame covariances)
    # apxImgs carries the linearisation point from the previous iteration
    if em_iter == 0 or reassign_apxImgs:
        apxImgs = None   # use default StarWarps initialisation

    expVal_t, expVal_t_t, expVal_tm1_t, loglikelihood, apxImgs = sw.computeSuffStatistics(
        meanImg, imCov, obs_list,
        noiseCov_img, newTheta,
        init_x, init_y, flowbasis_x, flowbasis_y, initTheta,
        method=warp_method,
        measurement=measurement,
        interiorPriors=interiorPriors,
        numLinIters=numLinIters,
        compute_expVal_tm1_t=True,    # needed for M-step
        init_images=init_images
    )
    negll.append(-loglikelihood[2])
    thetas.append(copy.deepcopy(newTheta))
    print(f"  Neg log-likelihood: {negll[-1]:.4f}")

    # ---------- Visualise this iteration ----------
    os.makedirs(f'starwarps_results/iter_{em_iter:02d}', exist_ok=True)

    stdevImg = meanImg[0].copy()
    for i, (mean_frame, cov_frame) in enumerate(zip(expVal_t, expVal_t_t)):
        stdevImg.imvec = np.sqrt(np.diag(cov_frame))
        mean_frame.save_fits(f'starwarps_results/iter_{em_iter:02d}/mean_{i}.fits')
        stdevImg.save_fits(f'starwarps_results/iter_{em_iter:02d}/stdev_{i}.fits')
    sw.movie(expVal_t, out=f'starwarps_results/iter_{em_iter:02d}/movie.mp4', fps=5)

    # ---------- M-step (skip on final iteration) ----------
    if em_iter < nIters:
        result = opt.minimize(
            sw.expnegloglikelihood,
            newTheta,
            args=(
                expVal_t, expVal_t_t, expVal_tm1_t,
                meanImg, imCov, obs_list,
                noiseCov_img,
                init_x, init_y, flowbasis_x, flowbasis_y, initTheta,
                warp_method
            ),
            method='L-BFGS-B',
            jac=sw.deriv_expnegloglikelihood,
            bounds=bnds,
            options=optdict
        )
        newTheta = result.x
        print(f"  M-step converged: {result.success}  fun={result.fun:.4f}")
# %%
# ============================================================
# Final comparison plot (original vs StarWarps reconstruction)
# ============================================================
fig, axes = plt.subplots(2, n_frames, figsize=(3 * n_frames, 6))
for i, (orig, recon) in enumerate(zip(valid_images, expVal_t)):
    # Regrid original to match NPIX for fair comparison
    orig_rg = orig.regrid_image(fov, NPIX)
    axes[0, i].imshow(orig_rg.imarr(),  cmap='hot', origin='lower')
    axes[1, i].imshow(recon.imarr(), cmap='hot', origin='lower')
    axes[0, i].set_title(f'Original {i}')
    chisq = obs_list[i].chisq(recon, dtype='vis', ttype='direct')
    axes[1, i].set_title(f'χ²={chisq:.3f}')
    for ax in axes[:, i]:
        ax.axis('off')

axes[0, 0].set_ylabel('Original')
axes[1, 0].set_ylabel('StarWarps')
plt.tight_layout()
plt.savefig('starwarps_comparison.png', dpi=150)
plt.show()
# %%
# Neg log-likelihood convergence curve
plt.figure(figsize=(6, 3))
plt.plot(negll, 'o-')
plt.xlabel('EM Iteration')
plt.ylabel('Negative Log-Likelihood')
plt.title('StarWarps EM Convergence')
plt.tight_layout()
plt.savefig('starwarps_convergence.png', dpi=150)
plt.show()

for i, (obs, recon) in enumerate(zip(obs_list, expVal_t)):
    chisq = obs.chisq(recon, dtype='vis', ttype='direct')
    print(f"Frame {i}: chi-sq = {chisq:.3f}")
# %%
import os
from PIL import Image as PILImage
import numpy as np

out_dir = './tutorial_results/blackhole/frames_bw2'
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
    img.save(f'{out_dir}/frame_{i:02d}.png')
    print(f"Saved frame {i}")

print(f"All frames saved to {out_dir}")
# %%
