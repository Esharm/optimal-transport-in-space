import numpy as np
from operators import grad, div, hessian, div2

# ============================================================
# MODULAR SPATIAL REGULARIZER CLASSES
# ============================================================
class L1Regularizer:
    def __init__(self, alpha=0.01, iters=20):
        self.alpha = alpha
        self.iters = iters

    def solve(self, u_init, f_k, sampler, admm_target, admm_weight):
        """ Chambolle-Pock implementation for pure L1 regularization """
        uk = u_init.copy()
        uk_bar = uk.copy()
        p = np.zeros_like(uk)  # Dual variable matches image dimension
        
        tau = 0.5 / (1.0 + admm_weight)
        sigma = 0.5

        for _ in range(self.iters):
            # Dual step: L1 soft-thresholding conjugate projection (clipping to alpha)
            p = np.clip(p + sigma * self.alpha * uk_bar, -self.alpha, self.alpha)
            
            # Primal step
            uk_old = uk.copy()
            grad_data = sampler.adjoint(sampler.forward(uk) - f_k)
            grad_admm = admm_weight * (uk - admm_target) if admm_weight > 0 else 0
            
            uk = uk - tau * (grad_data + grad_admm + p)
            uk = np.maximum(uk, 0.0)
            uk_bar = uk + 1.0 * (uk - uk_old)
            
        return uk

class L1HessianRegularizer:
    def __init__(self, alpha_l1=0.005, alpha_hess=0.01, iters=30):
        self.alpha_l1 = alpha_l1
        self.alpha_hess = alpha_hess
        self.iters = iters

    def solve(self, u_init, f_k, sampler, admm_target, admm_weight):
        """ Combined Chambolle-Pock minimizing Data + ADMM + L1(u) + Hessian(u) """
        uk = u_init.copy()
        uk_bar = uk.copy()
        
        p_l1 = np.zeros_like(uk)            # Dual for L1
        Q_hess = np.zeros((4, *uk.shape))   # Dual for Hessian
        
        tau = 0.04 / (1.0 + admm_weight)
        sigma = 0.1

        for _ in range(self.iters):
            # 1. Dual Update: L1 component
            p_l1 = np.clip(p_l1 + sigma * self.alpha_l1 * uk_bar, -self.alpha_l1, self.alpha_l1)
            
            # 2. Dual Update: Hessian component
            Q_hess = Q_hess + sigma * self.alpha_hess * hessian(uk_bar)
            norm_Q = np.sqrt(Q_hess[0]**2 + Q_hess[1]**2 + Q_hess[2]**2 + Q_hess[3]**2)
            denom = np.maximum(1.0, norm_Q / self.alpha_hess)
            for i in range(4):
                Q_hess[i] /= denom
            
            # 3. Primal Step combining gradients and both dual pullbacks
            uk_old = uk.copy()
            grad_data = sampler.adjoint(sampler.forward(uk) - f_k)
            grad_admm = admm_weight * (uk - admm_target) if admm_weight > 0 else 0
            
            # Descent execution
            uk = uk - tau * (grad_data + grad_admm + p_l1 + div2(Q_hess))
            uk = np.maximum(uk, 0.0)
            uk_bar = uk + 1.0 * (uk - uk_old)
            
        return uk

        
class TotalVariationRegularizer:
    def __init__(self, alpha=0.01, iters=20):
        self.alpha = alpha
        self.iters = iters

    def solve(self, u_init, f_k, sampler, admm_target, admm_weight):
        """ Chambolle-Pock implementation for Isotropic TV """
        uk = u_init.copy()
        uk_bar = uk.copy()
        p = np.zeros((2, *uk.shape))
        
        tau = 0.1 / (1.0 + admm_weight)
        sigma = 0.1

        for _ in range(self.iters):
            # Dual step: Project onto L-infinity ball
            p = p + sigma * self.alpha * grad(uk_bar)
            norm_p = np.sqrt(p[0]**2 + p[1]**2)
            denom = np.maximum(1.0, norm_p / self.alpha)
            p[0] /= denom
            p[1] /= denom
            
            # Primal step
            uk_old = uk.copy()
            grad_data = sampler.adjoint(sampler.forward(uk) - f_k)
            grad_admm = admm_weight * (uk - admm_target) if admm_weight > 0 else 0
            
            uk = uk - tau * (grad_data + grad_admm - div(p))
            uk = np.maximum(uk, 0.0)
            uk_bar = uk + 1.0 * (uk - uk_old)
            
        return uk


class HessianRegularizer:
    def __init__(self, alpha=0.01, iters=20):
        self.alpha = alpha
        self.iters = iters

    def solve(self, u_init, f_k, sampler, admm_target, admm_weight):
        """ Chambolle-Pock implementation for Hessian Curvature regularization """
        uk = u_init.copy()
        uk_bar = uk.copy()
        # Dual variable matching the 4 elements of the forward Hessian matrix tensor
        Q = np.zeros((4, *uk.shape))
        
        # Hessian operator norm squared is bounded by 64. 
        # Adjust step lengths dynamically for stability guarantee.
        tau = 0.05 / (1.0 + admm_weight)
        sigma = 0.1

        for _ in range(self.iters):
            # Dual step: Project onto L-infinity ball for second-order differentials
            Q = Q + sigma * self.alpha * hessian(uk_bar)
            norm_Q = np.sqrt(Q[0]**2 + Q[1]**2 + Q[2]**2 + Q[3]**2)
            denom = np.maximum(1.0, norm_Q / self.alpha)
            for i in range(4):
                Q[i] /= denom
                
            # Primal step
            uk_old = uk.copy()
            grad_data = sampler.adjoint(sampler.forward(uk) - f_k)
            grad_admm = admm_weight * (uk - admm_target) if admm_weight > 0 else 0
            
            # Notice the addition of div2 pullback mapping second derivatives
            uk = uk - tau * (grad_data + grad_admm + div2(Q))
            uk = np.maximum(uk, 0.0)
            uk_bar = uk + 1.0 * (uk - uk_old)
            
        return uk

# ============================================================
# TEMPORAL OPTIMAL TRANSPORT AND COUPLING SOLVERS
# ============================================================

def transport_step(u, lam0, lam1, beta=1.0, eta=1.0, T=5, inner_iters=10):
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
        tau, sigma = 0.01, 0.01

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
                    if current_mass > 0: rho[t] *= (mass_0 / current_mass)
                elif t == T - 1:
                    target = u[k+1] - lam1[k] / eta
                    rho[t] = (rho[t] + tau * eta * target) / (1.0 + tau * eta)
                    current_mass = np.sum(rho[t])
                    if current_mass > 0: rho[t] *= (mass_1 / current_mass)
                else:
                    div_m = div(m[t]) - div(m[t-1]) if t < T-1 else -div(m[t-1])
                    rho[t] = np.maximum(rho[t] - tau * div_m, 0.0)
            
            for t in range(T - 1):
                d_rho = (rho[t+1] - rho[t]) / dt
                phi[t] = phi[t] + sigma * (d_rho + div(m[t]))
                
        b0[k] = rho[0]
        b1[k] = rho[-1]
        
    return b0, b1


def image_step(u, f, samplers, b0, b1, lam0, lam1, regularizer, eta=1.0):
    """
    Step B: Coordinates frame targets and lets the hand-off regularizer 
    object compute spatial reconstructions modularly.
    """
    K, H, W = u.shape
    u_new = u.copy()
    
    for k in range(K):
        admm_target = np.zeros((H, W))
        admm_weight = 0.0
        
        if k < K - 1:
            admm_target += b0[k] + lam0[k] / eta
            admm_weight += eta
        if k > 0:
            admm_target += b1[k-1] + lam1[k-1] / eta
            admm_weight += eta
            
        u_new[k] = regularizer.solve(u[k], f[k], samplers[k], admm_target, admm_weight)
        
    return u_new


def dual_step(u, b0, b1, lam0, lam1, eta=1.0):
    K = len(u)
    relaxation = 0.2
    for k in range(K - 1):
        lam0[k] += relaxation * eta * (b0[k] - u[k])
        lam1[k] += relaxation * eta * (b1[k] - u[k + 1])
    return lam0, lam1