import numpy as np
import pandas as pd
from operators import FourierSampler, random_uv_mask
from admm import ADMM
from io_utils import load_frames
from skimage.metrics import structural_similarity as ssim

def compute_nrmse(gt, test):
    rmse = np.sqrt(np.mean((gt - test) ** 2))
    denom = gt.max() - gt.min()
    return rmse / (denom + 1e-8)

def evaluate_at_density(density, u_true, samplers_template):
    K, H, W = u_true.shape
    mask = random_uv_mask((H, W), density=density, seed=42)
    
    samplers = []
    f = []
    
    # Generate data specific to this density stage
    for k in range(K):
        S = FourierSampler(mask)
        samplers.append(S)
        fk_clean = S.forward(u_true[k])
        noise = 0.01 * (np.random.randn(H, W) + 1j * np.random.randn(H, W))
        f.append(fk_clean + (mask * noise))
        
    # Baseline Dirty Reconstructions
    u0 = np.stack([samplers[k].adjoint(f[k]) for k in range(K)])
    u0 = np.maximum(u0, 0.0)
    
    # Optimization Engine Configuration
    model = ADMM(samplers, alpha=0.001, beta=0.02, eta=0.05)
    u_rec = model.run(u0, f, n_iter=10)
    
    # Metrics aggregations across the sequence
    dirty_ssims, rec_ssims = [], []
    dirty_nrmses, rec_nrmses = [], []
    
    for k in range(K):
        gt = u_true[k]
        d_img = u0[k]
        r_img = np.clip(u_rec[k], 0.0, 1.0)
        
        # Scale alignment safe checks
        d_img = (d_img - d_img.min()) / (d_img.max() - d_img.min() + 1e-8)
        r_img = (r_img - r_img.min()) / (r_img.max() - r_img.min() + 1e-8)
        gt_norm = (gt - gt.min()) / (gt.max() - gt.min() + 1e-8)
        
        dirty_ssims.append(ssim(gt_norm, d_img, data_range=1.0))
        rec_ssims.append(ssim(gt_norm, r_img, data_range=1.0))
        dirty_nrmses.append(compute_nrmse(gt_norm, d_img))
        rec_nrmses.append(compute_nrmse(gt_norm, r_img))
        
    return {
        "Density": density,
        "Dirty_SSIM": np.mean(dirty_ssims),
        "Rec_SSIM": np.mean(rec_ssims),
        "Dirty_NRMSE": np.mean(dirty_nrmses),
        "Rec_NRMSE": np.mean(rec_nrmses)
    }

def main():
    u_true, _ = load_frames("data", max_frames=10) # Using 10 frames to sweep faster
    
    # Testing 5%, 10%, 20%, and 40% Fourier spacing densities
    densities = [0.05, 0.10, 0.20, 0.40]
    results = []
    
    print("Starting Comprehensive Sparsity Sweep Profile...")
    for d in densities:
        print(f"\n--- Testing Density: {d*100:.1f}% ---")
        metrics = evaluate_at_density(d, u_true, None)
        results.append(metrics)
        
    df = pd.DataFrame(results)
    print("\n" + "="*70)
    print(f"{'FINAL SPARSITY SWEEP BENCHMARK REPORT':^70}")
    print("="*70)
    print(df.to_string(index=False, formatters={
        'Density': '{:.2f}'.format, 'Dirty_SSIM': '{:.4f}'.format,
        'Rec_SSIM': '{:.4f}'.format, 'Dirty_NRMSE': '{:.4f}'.format,
        'Rec_NRMSE': '{:.4f}'.format
    }))
    print("="*70)

if __name__ == "__main__":
    main()