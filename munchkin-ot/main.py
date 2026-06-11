import numpy as np
from io_utils import load_frames, save_frames
from operators import FourierSampler, random_uv_mask
from admm import ADMM

def make_mask(shape):
    # 15% density is optimal for structural recovery tests
    return random_uv_mask(shape, density=0.15, seed=42)

def main():
    folder = "data"
    u_true, names = load_frames(folder, max_frames=30)

    K, H, W = u_true.shape
    mask = make_mask((H, W))

    samplers = []
    f = []

    print("\n--- Generating Clean Complex Measurements ---")
    for k in range(K):
        S = FourierSampler(mask)
        samplers.append(S)
        
        # Compute clean forward coefficients
        fk_clean = S.forward(u_true[k])
        
        # CRITICAL FIX: Generate complex noise and mask it so the algorithm 
        # doesn't try to fit noise on unmeasured frequencies!
        noise = 0.01 * (np.random.randn(H, W) + 1j * np.random.randn(H, W))
        fk = fk_clean + (mask * noise)
        f.append(fk)

    # Instantiate initial dirty guess
    u0 = np.stack([samplers[k].adjoint(f[k]) for k in range(K)])
    # Ensure u0 starts in a valid positive range
    u0 = np.maximum(u0, 0.0)
    save_frames(u0, "./dirty_reconstruction")

    # Balanced hyperparameters
    # alpha: Spatial TV regularity force
    # beta: Temporal Optimal Transport fluid force
    # eta: ADMM convergence coupling force
    model = ADMM(samplers, alpha=0.0005, beta=0.01, eta=0.05)

    print("\n--- Beginning Structured ADMM Engine ---")
    u_rec = model.run(u0, f, n_iter=15)

    u_final = np.clip(u_rec, 0.0, 1.0)
    save_frames(u_final, "./reconstruction")
    print("\nReconstruction complete. Files exported safely.")

if __name__ == "__main__":
    main()