import numpy as np
from solvers import transport_step, image_step, dual_step

class ADMM:
    def __init__(self, samplers, regularizer, beta=1.0, eta=1.0):
        self.samplers = samplers
        self.regularizer = regularizer  # Accept an instance of TV or Hessian
        self.beta = beta
        self.eta = eta

    def run(self, u, f, n_iter=10):
        K = len(u)
        H, W = u[0].shape

        lam0 = np.zeros((K - 1, H, W))
        lam1 = np.zeros((K - 1, H, W))

        for it in range(n_iter):
            b0, b1 = transport_step(u, lam0, lam1, beta=self.beta, eta=self.eta)
            u = image_step(u, f, self.samplers, b0, b1, lam0, lam1, self.regularizer, eta=self.eta)
            lam0, lam1 = dual_step(u, b0, b1, lam0, lam1, eta=self.eta)

            print(f"ADMM Iter {it+1:02d} -> Mean: {u.mean():.4f} | Max: {u.max():.4f}")

        return u