import numpy as np


class ComplexVisibilityDataTerm:
    """
    Complex visibility data fidelity:

        0.5 * || S u - f ||_2^2

    where S.forward(u) and f have matching weighting/scaling.

    This class provides the generic interface:
        loss(u)
        gradient(u)
        dirty_image()
    """

    def __init__(self, sampler, f):
        self.sampler = sampler
        self.f = np.asarray(f, dtype=np.complex128)

    def residual(self, x):
        return self.sampler.forward(x) - self.f

    def loss(self, x):
        r = self.residual(x)
        return 0.5 * np.sum(np.abs(r) ** 2)

    def gradient(self, x):
        r = self.residual(x)
        return self.sampler.adjoint(r)

    def dirty_image(self):
        return np.maximum(self.sampler.adjoint(self.f), 0.0)