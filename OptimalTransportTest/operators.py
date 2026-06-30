import numpy as np


class VisibilitySampler:
    """Scaled direct nonuniform Fourier operator and its real-image adjoint."""

    def __init__(self, u, v, weight=None, shape=(128, 128),
                 fov_rad=160e-6 / 206265.0, data_scale=1.0,
                 use_cache=True, dtype=np.complex64):
        self.u = np.asarray(u, dtype=np.float64)
        self.v = np.asarray(v, dtype=np.float64)
        self.weight = np.ones_like(self.u) if weight is None else np.asarray(weight, dtype=np.float64)
        if not (len(self.u) == len(self.v) == len(self.weight)):
            raise ValueError("u, v, and weight must have the same length")
        self.H, self.W = shape
        self.fov_rad = float(fov_rad)
        self.data_scale = float(max(data_scale, 1e-12))
        self.dtype = dtype
        l = (np.arange(self.W) - self.W / 2) * self.fov_rad / self.W
        m = (np.arange(self.H) - self.H / 2) * self.fov_rad / self.H
        grid_l, grid_m = np.meshgrid(l, m)
        self.l_flat, self.m_flat = grid_l.ravel(), grid_m.ravel()
        self.op_scale = np.sqrt(np.sum(self.weight) + 1e-12) * np.sqrt(self.H * self.W)
        self.total_scale = self.op_scale * self.data_scale
        if use_cache and len(self.u):
            phase = -2j * np.pi * (
                self.u[:, None] * self.l_flat + self.v[:, None] * self.m_flat
            )
            self.A = np.exp(phase).astype(dtype)
        else:
            self.A = None

    def forward(self, x):
        flat = np.asarray(x, dtype=np.float64).ravel()
        if not len(self.u):
            return np.zeros(0, dtype=np.complex128)
        if self.A is not None:
            result = self.A @ flat
        else:
            result = np.zeros(len(self.u), dtype=np.complex128)
            for start in range(0, len(self.u), 512):
                section = slice(start, min(start + 512, len(self.u)))
                phase = -2j * np.pi * (
                    self.u[section, None] * self.l_flat
                    + self.v[section, None] * self.m_flat
                )
                result[section] = np.exp(phase) @ flat
        return np.sqrt(self.weight) * result / self.total_scale

    def adjoint(self, y):
        y = np.asarray(y, dtype=np.complex128)
        if not len(self.u):
            return np.zeros((self.H, self.W), dtype=np.float64)
        weighted = np.sqrt(self.weight) * y / self.total_scale
        if self.A is not None:
            result = self.A.conj().T @ weighted
        else:
            result = np.zeros(self.H * self.W, dtype=np.complex128)
            for start in range(0, len(self.u), 512):
                section = slice(start, min(start + 512, len(self.u)))
                phase = 2j * np.pi * (
                    self.u[section, None] * self.l_flat
                    + self.v[section, None] * self.m_flat
                )
                result += np.exp(phase).T @ weighted[section]
        return np.real(result.reshape(self.H, self.W))


def grad(u):
    ux, uy = np.zeros_like(u), np.zeros_like(u)
    ux[:, :-1] = u[:, 1:] - u[:, :-1]
    uy[:-1] = u[1:] - u[:-1]
    return np.stack((ux, uy))


def div(p):
    px, py = p
    dx, dy = np.zeros_like(px), np.zeros_like(py)
    dx[:, 0] = px[:, 0]
    dx[:, 1:-1] = px[:, 1:-1] - px[:, :-2]
    dx[:, -1] = -px[:, -2]
    dy[0] = py[0]
    dy[1:-1] = py[1:-1] - py[:-2]
    dy[-1] = -py[-2]
    return dx + dy


def hessian(u):
    first = grad(u)
    dxx, dxy = grad(first[0])
    dyx, dyy = grad(first[1])
    return np.stack((dxx, dyy, dxy, dyx))


def div2(q):
    qxx, qyy, qxy, qyx = q

    def div_x(p):
        result = np.zeros_like(p)
        result[:, 0] = p[:, 0]
        result[:, 1:-1] = p[:, 1:-1] - p[:, :-2]
        result[:, -1] = -p[:, -2]
        return result

    def div_y(p):
        result = np.zeros_like(p)
        result[0] = p[0]
        result[1:-1] = p[1:-1] - p[:-2]
        result[-1] = -p[-2]
        return result

    return (
        div_x(div_x(qxx)) + div_y(div_y(qyy))
        + div_y(div_x(qxy)) + div_x(div_y(qyx))
    )


def normalize01(x):
    x = np.asarray(x, dtype=np.float64)
    return (x - x.min()) / (x.max() - x.min() + 1e-12)


def project_nonnegative_mass(x, target_mass=None):
    """Euclidean projection onto x >= 0, optionally with sum(x)=target_mass."""
    x = np.asarray(x, dtype=np.float64)
    if target_mass is None:
        return np.maximum(x, 0.0)
    target_mass = float(target_mass)
    if target_mass <= 0:
        return np.zeros_like(x)

    flat = x.ravel()
    ordered = np.sort(flat)[::-1]
    cumulative = np.cumsum(ordered) - target_mass
    indices = np.arange(1, flat.size + 1)
    active = ordered - cumulative / indices > 0
    rho = np.flatnonzero(active)[-1]
    threshold = cumulative[rho] / (rho + 1.0)
    return np.maximum(x - threshold, 0.0)
