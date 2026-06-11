import numpy as np
import numpy.fft as fft

class FourierSampler:
    """
    Forward: image -> sparse Fourier measurements
    Adjoint: visibilities -> dirty image
    """
    def __init__(self, mask):
        # Ensure mask is a boolean or float array of 1s and 0s
        self.mask = mask.astype(float)

    def forward(self, x):
        # norm="ortho" keeps energy scaled 1:1 with pixel values
        return self.mask * fft.fft2(x, norm="ortho")

    def adjoint(self, y):
        # CRITICAL FIX: The visibility y is ALREADY masked from the forward step.
        # Multiplying by the mask again compounds noise errors.
        return np.real(fft.ifft2(y, norm="ortho"))

def grad(u):
    """ Forward finite differences with Neumann boundary conditions """
    ux = np.zeros_like(u)
    uy = np.zeros_like(u)
    ux[:, :-1] = u[:, 1:] - u[:, :-1]
    uy[:-1, :] = u[1:, :] - u[:-1, :]
    return np.stack([ux, uy])

def div(p):
    """ Adjoint of grad operator (backward differences) """
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

def random_uv_mask(shape, density=0.1, seed=0):
    rng = np.random.default_rng(seed)
    mask = rng.random(shape) < density
    return mask.astype(float)