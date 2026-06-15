import numpy as np
import numpy.fft as fft

class FourierSampler:
    def __init__(self, mask):
        self.mask = mask.astype(float)

    def forward(self, x):
        return self.mask * fft.fft2(x, norm="ortho")

    def adjoint(self, y):
        return np.real(fft.ifft2(y, norm="ortho"))

# ============================================================
# FIRST ORDER GRADIENT & DIVERGENCE (For TV)
# ============================================================
def grad(u):
    """ Forward differences for gradient """
    ux = np.zeros_like(u)
    uy = np.zeros_like(u)
    ux[:, :-1] = u[:, 1:] - u[:, :-1]
    uy[:-1, :] = u[1:, :] - u[:-1, :]
    return np.stack([ux, uy])

def div(p):
    """ Backward differences for divergence """
    px, py = p
    dx = np.zeros_like(px)
    dy = np.zeros_like(py)
    
    dx[:, 1:-1] = px[:, 1:-1] - px[:, :-2]
    dx[:, 0] = px[:, 0]
    dx[:, -1] = -px[:, -2]
    
    dy[1:-1, :] = py[1:-1, :] - py[:-2, :]
    dy[0, :] = py[0, :]
    dy[-1, :] = -py[-2, :]
    return dx + dy

# ============================================================
# SECOND ORDER HESSIAN & ADJOINT HESSIAN 
# ============================================================
def hessian(u):
    """
    Computes second order finite differences.
    Returns: [4, H, W] tensor corresponding to (Dxx, Dyy, Dxy, Dyx)
    """
    H, W = u.shape
    dx = np.zeros_like(u)
    dy = np.zeros_like(u)
    dx[:, :-1] = u[:, 1:] - u[:, :-1]
    dy[:-1, :] = u[1:, :] - u[:-1, :]
    
    dxx = np.zeros_like(u)
    dxy = np.zeros_like(u)
    dxx[:, :-1] = dx[:, 1:] - dx[:, :-1]
    dxy[:-1, :] = dx[1:, :] - dx[:-1, :]
    
    dyx = np.zeros_like(u)
    dyy = np.zeros_like(u)
    dyx[:, :-1] = dy[:, 1:] - dy[:, :-1]
    dyy[:-1, :] = dy[1:, :] - dy[:-1, :]
    
    return np.stack([dxx, dyy, dxy, dyx])

def div2(Q):
    """
    Adjoint operator of the Hessian (Second divergence).
    Q: [4, H, W] matrix fields corresponding to (Qxx, Qyy, Qxy, Qyx)
    """
    qxx, qyy, qxy, qyx = Q
    
    def div_x(p):
        d = np.zeros_like(p)
        d[:, 1:-1] = p[:, 1:-1] - p[:, :-2]
        d[:, 0] = p[:, 0]
        d[:, -1] = -p[:, -2]
        return d

    def div_y(p):
        d = np.zeros_like(p)
        d[1:-1, :] = p[1:-1, :] - p[:-2, :]
        d[0, :] = p[0, :]
        d[-1, :] = -p[-2, :]
        return d

    # Apply backward finite differences sequentially matching the forward step
    return div_x(div_x(qxx)) + div_y(div_y(qyy)) + div_y(div_x(qxy)) + div_x(div_y(qyx))

def random_uv_mask(shape, density=0.1, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.random(shape) < density).astype(float)