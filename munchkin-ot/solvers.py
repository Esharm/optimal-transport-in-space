import numpy as np
from operators import grad, div

def transport_step(u, lam0, lam1, beta=1.0, eta=1.0, T=5, inner_iters=10):
    """
    Step A: Solves Benamou-Brenier Optimal Transport.
    """
    K, H, W = u.shape
    b0 = np.zeros_like(u)
    b1 = np.zeros_like(u)
    
    frame_masses = np.sum(u, axis=(1, 2))
    
    for k in range(K - 1):
        mass_0 = max(frame_masses[k], 1e-5)
        mass_1 = max(frame_masses[k+1], 1e-5)
        
        rho = np.zeros((T, H, W))
        for t in range(T):
            alpha_t = t / (T - 1)
            rho[t] = (1 - alpha_t) * u[k] + alpha_t * u[k+1]
            
        m = np.zeros((T - 1, 2, H, W))
        phi = np.zeros((T - 1, H, W))
        
        dt = 1.0 / (T - 1)
        tau = 0.01
        sigma = 0.01

        for _ in range(inner_iters):
            for t in range(T - 1):
                grad_phi = grad(phi[t])
                m_proj = m[t] - tau * beta * grad_phi
                rho_avg = np.maximum(0.5 * (rho[t] + rho[t+1]), 1e-4)
                m[t] = m_proj / (1.0 + (2.0 * tau * beta) / rho_avg)

            for t in range(T):
                if t == 0:
                    target = u[k] - lam0[k] / eta
                    rho[t] = (rho[t] + tau * eta * target) / (1.0 + tau * eta)
                    current_mass = np.sum(rho[t])
                    if current_mass > 0:
                        rho[t] *= (mass_0 / current_mass)
                elif t == T - 1:
                    target = u[k+1] - lam1[k] / eta
                    rho[t] = (rho[t] + tau * eta * target) / (1.0 + tau * eta)
                    current_mass = np.sum(rho[t])
                    if current_mass > 0:
                        rho[t] *= (mass_1 / current_mass)
                else:
                    div_m = div(m[t]) - div(m[t-1]) if t < T-1 else -div(m[t-1])
                    rho[t] = np.maximum(rho[t] - tau * div_m, 0.0)
            
            for t in range(T - 1):
                d_rho = (rho[t+1] - rho[t]) / dt
                phi[t] = phi[t] + sigma * (d_rho + div(m[t]))
                
        b0[k] = rho[0]
        b1[k] = rho[-1]
        
    return b0, b1


def image_step(u, f, samplers, b0, b1, lam0, lam1, alpha=0.01, eta=1.0, iters=20):
    """
    Step B: Chambolle-Pock Primal-Dual framework with relaxed stabilization bounds.
    """
    K, H, W = u.shape
    u_new = u.copy()
    
    for k in range(K):
        S = samplers[k]
        uk = u_new[k].copy()
        uk_bar = uk.copy()
        p = np.zeros((2, H, W))
        
        admm_target = np.zeros((H, W))
        admm_weight = 0.0
        
        if k < K - 1:
            admm_target += b0[k] + lam0[k] / eta
            admm_weight += eta
        if k > 0:
            admm_target += b1[k-1] + lam1[k-1] / eta
            admm_weight += eta

        # Conservative step-lengths to stabilize optimization tracking
        tau = 0.1 / (1.0 + admm_weight)
        sigma = 0.1

        for _ in range(iters):
            # Dual Step: TV gradient clipping
            g = grad(uk_bar)
            p = p + sigma * alpha * g
            norm_p = np.sqrt(p[0]**2 + p[1]**2)
            denom = np.maximum(1.0, norm_p / alpha)
            p[0] /= denom
            p[1] /= denom
            
            # Primal Step
            uk_old = uk.copy()
            grad_data = S.adjoint(S.forward(uk) - f[k])
            
            if admm_weight > 0:
                grad_admm = admm_weight * uk - admm_weight * admm_target
            else:
                grad_admm = 0
                
            uk = uk - tau * (grad_data + grad_admm - div(p))
            uk = np.maximum(uk, 0.0)  # Enforce non-negativity softly
            
            uk_bar = uk + 1.0 * (uk - uk_old)
            
        u_new[k] = uk

    return u_new


def dual_step(u, b0, b1, lam0, lam1, eta=1.0):
    """ Step C: Scaled Multiplier updates """
    K = len(u)
    relaxation = 0.2  # Soften the update velocity to eliminate cell value snapping
    
    for k in range(K - 1):
        lam0[k] += relaxation * eta * (b0[k] - u[k])
        lam1[k] += relaxation * eta * (b1[k] - u[k + 1])
        
    return lam0, lam1