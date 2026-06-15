import numpy as np
import pandas as pd
from io_utils import load_frames, save_frames
from operators import FourierSampler, random_uv_mask
from solvers import L1HessianRegularizer
from admm import ADMM
from skimage.metrics import structural_similarity as ssim

def compute_nrmse(gt, test):
    rmse = np.sqrt(np.mean((gt - test) ** 2))
    denom = gt.max() - gt.min()
    return rmse / (denom + 1e-8)

def normalize_range(img):
    return (img - img.min()) / (img.max() - img.min() + 1e-8)

def main():
    u_true, names = load_frames("data", max_frames=30)
    K, H, W = u_true.shape
    print(f"Loaded {K} frames. Launching Combined L1 + Hessian Structural Pipeline...")
    
    # Evaluate at a baseline mask density
    mask = random_uv_mask((H, W), density=0.10, seed=42)
    
    samplers = []
    f = []
    for k in range(K):
        S = FourierSampler(mask)
        samplers.append(S)
        noise = 0.01 * (np.random.randn(H, W) + 1j * np.random.randn(H, W))
        f.append(S.forward(u_true[k]) + (mask * noise))
        
    u_dirty = np.stack([samplers[k].adjoint(f[k]) for k in range(K)])
    u_dirty = np.maximum(u_dirty, 0.0)
    save_frames(u_dirty, "./dirty_reconstruction")

    # ============================================================
    # EVALUATION RUN 1: Standalone L1 + Hessian (No Wasserstein Transport)
    # ============================================================
    print("\n--- Computing Standalone L1 + Hessian Reconstructions ---")
    u_l1_hess = np.zeros_like(u_dirty)
    # Tuning: Alpha_l1 dampens background noise, Alpha_hess keeps features connected
    static_joint_prior = L1HessianRegularizer(alpha_l1=0.02, alpha_hess=0.01, iters=80)
    
    for k in range(K):
        u_l1_hess[k] = static_joint_prior.solve(u_dirty[k], f[k], samplers[k], admm_target=0, admm_weight=0)
    save_frames(np.clip(u_l1_hess, 0.0, 1.0), "./reconstruction_l1_hessian")

    # ============================================================
    # EVALUATION RUN 2: Spatiotemporal L1 + Hessian + Wasserstein (Full ADMM)
    # ============================================================
    print("\n--- Computing Full Spatiotemporal ADMM (L1 + Hessian + Wasserstein) ---")
    admm_joint_prior = L1HessianRegularizer(alpha_l1=0.001, alpha_hess=0.005, iters=25)
    
    # Controlled fluid weight parameters to curb noise-artifact tracking loops
    model = ADMM(samplers, regularizer=admm_joint_prior, beta=0.001, eta=0.02)
    u_admm = model.run(u_dirty.copy(), f, n_iter=10)
    save_frames(np.clip(u_admm, 0.0, 1.0), "./reconstruction_l1_hessian_wass")

    # ============================================================
    # METRICS ANALYTICS AGGREGATOR
    # ============================================================
    print("\nProcessing Performance Comparison Tables...")
    log = []
    for k in range(K):
        gt = normalize_range(u_true[k])
        d_img = normalize_range(u_dirty[k])
        lh_img = normalize_range(u_l1_hess[k])
        w_img = normalize_range(np.clip(u_admm[k], 0.0, 1.0))
        
        log.append({
            "Frame": names[k],
            "Dirty_SSIM": ssim(gt, d_img, data_range=1.0),
            "L1Hess_SSIM": ssim(gt, lh_img, data_range=1.0),
            "Wass_SSIM": ssim(gt, w_img, data_range=1.0),
            "Dirty_NRMSE": compute_nrmse(gt, d_img),
            "L1Hess_NRMSE": compute_nrmse(gt, lh_img),
            "Wass_NRMSE": compute_nrmse(gt, w_img)
        })
        
    df = pd.DataFrame(log)
    
    print("\n" + "="*95)
    print(f"{'JOINT PRIOR BENCHMARK REPORT (L1 + HESSIAN COMPOSITIONS)':^95}")
    print("="*95)
    print(df.to_string(index=False, max_rows=35, formatters={
        'Dirty_SSIM': '{:.4f}'.format, 'L1Hess_SSIM': '{:.4f}'.format, 'Wass_SSIM': '{:.4f}'.format,
        'Dirty_NRMSE': '{:.4f}'.format, 'L1Hess_NRMSE': '{:.4f}'.format, 'Wass_NRMSE': '{:.4f}'.format
    }))
    print("="*95)
    print(f"{'MEAN SEQUENCE STATISTICS':^95}")
    print("-"*95)
    print(f"Mean SSIM:  [Dirty: {df['Dirty_SSIM'].mean():.4f}]  [L1+Hess: {df['L1Hess_SSIM'].mean():.4f}]  [L1+Hess+Wass: {df['Wass_SSIM'].mean():.4f}]")
    print(f"Mean NRMSE: [Dirty: {df['Dirty_NRMSE'].mean():.4f}]  [L1+Hess: {df['L1Hess_NRMSE'].mean():.4f}]  [L1+Hess+Wass: {df['Wass_NRMSE'].mean():.4f}]")
    print("="*95)

if __name__ == "__main__":
    main()