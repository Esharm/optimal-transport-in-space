import numpy as np
import pandas as pd
from operators import FourierSampler, random_uv_mask
from solvers import HessianRegularizer, image_step, L1HessianRegularizer
from admm import ADMM
from io_utils import load_frames
from skimage.metrics import structural_similarity as ssim

def compute_nrmse(gt, test):
    rmse = np.sqrt(np.mean((gt - test) ** 2))
    denom = gt.max() - gt.min()
    return rmse / (denom + 1e-8)

def normalize_range(img):
    """Aligns structural intensities to a strict 0-1 range for fair assessment"""
    return (img - img.min()) / (img.max() - img.min() + 1e-8)

def evaluate_at_sparsity(density, u_true):
    K, H, W = u_true.shape
    mask = random_uv_mask((H, W), density=density, seed=42)
    
    samplers = []
    f = []
    
    # 1. Simulate data visibilities for this specific density stage
    for k in range(K):
        S = FourierSampler(mask)
        samplers.append(S)
        fk_clean = S.forward(u_true[k])
        noise = 0.01 * (np.random.randn(H, W) + 1j * np.random.randn(H, W))
        f.append(fk_clean + (mask * noise))
        
    # 2. BASELINE 1: Compute Dirty Images
    u_dirty = np.stack([samplers[k].adjoint(f[k]) for k in range(K)])
    u_dirty = np.maximum(u_dirty, 0.0)
    
    # 3. BASELINE 2: Compute Hessian Only (Static spatial inversion, bypassing ADMM)
    u_hessian_only = np.zeros_like(u_dirty)
    static_prior = L1HessianRegularizer(alpha_l1=0.02, alpha_hess=0.01, iters=80)
    for k in range(K):
        # Passing 0 weight completely drops the ADMM optimal transport constraints
        u_hessian_only[k] = static_prior.solve(u_dirty[k], f[k], samplers[k], admm_target=0, admm_weight=0)
        
    # 4. FRAMEWORK: Compute Full Spatiotemporal Hessian + Wasserstein (ADMM Engine)
    admm_prior = L1HessianRegularizer(alpha_l1=0.001, alpha_hess=0.005, iters=25)
    model = ADMM(samplers, regularizer=admm_prior, beta=0.001, eta=0.02)
    u_admm = model.run(u_dirty.copy(), f, n_iter=25)
    
    # 5. Collate evaluation profiles across the entire frame sequence
    metrics = {
        "dirty_ssim": [], "hess_ssim": [], "wass_ssim": [],
        "dirty_nrmse": [], "hess_nrmse": [], "wass_nrmse": []
    }
    
    for k in range(K):
        gt = normalize_range(u_true[k])
        d_img = normalize_range(u_dirty[k])
        h_img = normalize_range(u_hessian_only[k])
        w_img = normalize_range(np.clip(u_admm[k], 0.0, 1.0))
        
        metrics["dirty_ssim"].append(ssim(gt, d_img, data_range=1.0))
        metrics["hess_ssim"].append(ssim(gt, h_img, data_range=1.0))
        metrics["wass_ssim"].append(ssim(gt, w_img, data_range=1.0))
        
        metrics["dirty_nrmse"].append(compute_nrmse(gt, d_img))
        metrics["hess_nrmse"].append(compute_nrmse(gt, h_img))
        metrics["wass_nrmse"].append(compute_nrmse(gt, w_img))
        
    return {
        "Sparsity": f"{density*100:.0f}%",
        "Dirty_SSIM": np.mean(metrics["dirty_ssim"]),
        "Hess_SSIM": np.mean(metrics["hess_ssim"]),
        "Wass_SSIM": np.mean(metrics["wass_ssim"]),
        "Dirty_NRMSE": np.mean(metrics["dirty_nrmse"]),
        "Hess_NRMSE": np.mean(metrics["hess_nrmse"]),
        "Wass_NRMSE": np.mean(metrics["wass_nrmse"]),
    }

def main():
    u_true, _ = load_frames("data")
    
    sparsity_levels = [0.01, 0.02, 0.05,.1]
    sweep_data = []
    
    print("======================================================================")
    print("      LAUNCHING MULTI-METRIC MODULAR SPARSITY BENCHMARK RUN           ")
    print("======================================================================")
    
    for level in sparsity_levels:
        print(f"\n>>> Running Profile Suite at Sparsity Scale: {level*100:.1f}%")
        results = evaluate_at_sparsity(level, u_true)
        sweep_data.append(results)
        
    df = pd.DataFrame(sweep_data)
    
    print("\n" + "="*85)
    print(f"{'SPARSITY METRIC COMPARISON BENCHMARK REPORT':^85}")
    print("="*85)
    print(df.to_string(index=False, formatters={
        'Dirty_SSIM': '{:.4f}'.format, 'Hess_SSIM': '{:.4f}'.format, 'Wass_SSIM': '{:.4f}'.format,
        'Dirty_NRMSE': '{:.4f}'.format, 'Hess_NRMSE': '{:.4f}'.format, 'Wass_NRMSE': '{:.4f}'.format
    }))
    print("="*85)
    
    df.to_csv("comprehensive_sparsity_sweep.csv", index=False)
    print("\nBenchmark saved securely to 'comprehensive_sparsity_sweep.csv'")

if __name__ == "__main__":
    main()