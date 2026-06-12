# %%
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import ehtim as eh
import time
import os
from pathlib import Path
from PIL import Image as PILImage
import importlib
import ehtim.imaging.dynamical_imaging as di
importlib.reload(di)
plt.close('all')

ttype = 'direct'
outpath = './tutorial_results/munchkin'
if not os.path.exists(os.path.dirname(outpath)):
    os.makedirs(os.path.dirname(outpath))
# %%
# Load the image and the telescope array
folder = Path('munchkin_frames')

ra  = 12.513728717168174   # hours
dec = 12.391123306919757   # degrees

target_flux = 0.6  # Jy — typical M87* flux at 230 GHz (~2.4 Jy for Sgr A*)

im = []
for path in sorted(folder.glob('*.png')):
    pil_img = PILImage.open(path).convert('L')
    arr = np.array(pil_img, dtype=float)
    
    print(f"Before norm: sum = {arr.sum():.1f}")   # ← add this temporarily
    
    arr = arr / arr.sum() * target_flux             # ← must happen BEFORE Image()
    
    print(f"After norm:  sum = {arr.sum():.4f}")   # should print exactly 0.6
    
    fov = 100 * eh.RADPERUAS
    eht_img = eh.image.Image(arr, fov / arr.shape[0], ra, dec, rf=226e9)
    im.append(eht_img)
eht = eh.array.load_txt('EHTII.txt')
# %%
# Regrid the image for display
for image in im:
    imdisp = image.regrid_image(120*eh.RADPERUAS, 512)

# %%
# simulation parameters
tint_sec = 60  # Integration time in seconds, 
tadv_sec = 600 # Advance time between scans
tstart_hr = 0  # GMST time of the start of the observation
tstop_hr = 24  # GMST time of the end of the observation
total_hrs = 24
bw_hz = 4.e9   # Bandwidth in Hz
n_frames  = 8
slot_hrs = total_hrs / n_frames
# %%
obs_list = []
valid_images = []

for i in range(n_frames):
    t0 = i * slot_hrs
    t1 = (i + 1) * slot_hrs
    frame_idx = int((i + 0.5) * len(im) / n_frames)
    image = im[frame_idx]
    try:
        obs = image.observe(
            eht, tint_sec, tadv_sec, t0, t1, bw_hz,
            sgrscat=False, ampcal=True, phasecal=True, ttype='direct'
        )
        obs_list.append(obs)
        valid_images.append(image)
    except Exception as e:
        print(f"Frame {i} skipped: {e}")

# %%
prior = eh.image.make_square(obs_list[0], npix=128, fov=fov)
prior.imvec += target_flux / prior.imvec.size  # uniform flux distribution
#STATIC
static_results = []
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
# %%
flux_list = [target_flux] * n_frames
init_list = [prior.copy() for _ in range(30)]
# %%
static_regrided = [im.regrid_image(fov, 128) for im in static_results]
result = di.dynamical_imaging(
    obs_input     = obs_list,     # your 30 observed obs objects
    init_ims      = static_regrided,    # your initial images

    Prior         = prior,

    d1='vis',
    alpha_d1=100,
    flux_List     = flux_list,
    entropy1='tv2',
    alpha_s1=50.0,

    R_dt={'alpha': 5, 'metric': 'SymKL', 'p': 2.0, 'sigma_dt': 0.0},

    maxit=500,
    ttype='direct',
)
# %%
di.export_movie(result, out='movie.mp4', fps=5)

# %%
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, n_frames, figsize=(3*n_frames, 6))
for i, (orig, recon) in enumerate(zip(valid_images, result)):
    axes[0, i].imshow(orig.imarr(),  cmap='hot', origin='lower')
    axes[1, i].imshow(recon.imarr(), cmap='hot', origin='lower')
    axes[0, i].set_title(f'Original {i}')
    axes[1, i].set_title(f'χ²={obs_list[i].chisq(recon, dtype="vis", ttype ="direct"):.3f}')

plt.tight_layout()
plt.savefig('comparison.png', dpi=150)
plt.show()
# %%
from IPython.display import display, Image as IPImage

obs = obs_list[0]

u = obs.data['u']
v = obs.data['v']
uvdist = np.sqrt(u**2 + v**2)

print(f"Min baseline: {uvdist.min()/1e6:.1f} Mλ")
print(f"Max baseline: {uvdist.max()/1e6:.1f} Mλ")
print(f"Number of visibilities: {len(uvdist)}")

# Plot uv-coverage to see the sampling visually
fig, ax = plt.subplots(figsize=(6,6))
ax.scatter(u/1e6,  v/1e6,  s=1, alpha=0.5, label='data')
ax.scatter(-u/1e6, -v/1e6, s=1, alpha=0.5)  # conjugate points
ax.set_xlabel('u (Mλ)')
ax.set_ylabel('v (Mλ)')
ax.set_title('uv-coverage')
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig('uv_coverage.png', dpi=150)
plt.close()
display(IPImage('uv_coverage.png'))
# %%
for i, obs in enumerate(obs_list):
    n_vis = len(obs.data)
    print(f"Frame {i}: {n_vis} visibilities")
# %%
