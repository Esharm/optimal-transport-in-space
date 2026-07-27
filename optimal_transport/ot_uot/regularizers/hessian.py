"""Second-order Hessian-Schatten/Frobenius regularization utilities.

The discrete Hessian is implemented as the Jacobian of the forward-difference
image gradient.  At each pixel it has four components ``(Dxx,Dxy,Dyx,Dyy)``.
The regularizer is the sum of pointwise Frobenius norms,

``sum_x sqrt(Dxx^2 + Dxy^2 + Dyx^2 + Dyy^2)``.

This is a convex second-order analogue of isotropic TV.  It favors images that
are locally close to affine and generally produces less staircasing than
first-order TV.
"""

from __future__ import annotations

import numpy as np

from ot_uot.core.finite_differences import divergence, gradient


Array = np.ndarray


def hessian(image: Array) -> Array:
    """Return the discrete Hessian with shape ``(2,2,H,W)``.

    The first Hessian index chooses the component of ``gradient(image)`` and
    the second chooses the derivative direction.  Thus the entries are
    ``Dxx, Dxy, Dyx, Dyy`` under the package's forward-difference convention.
    """

    image = np.asarray(image, dtype=np.float64)
    first = gradient(image)
    return np.stack((gradient(first[0]), gradient(first[1])), axis=0)


def hessian_adjoint(field: Array) -> Array:
    """Return the adjoint of :func:`hessian` for the finite-difference convention."""

    field = np.asarray(field, dtype=np.float64)
    if field.ndim != 4 or field.shape[:2] != (2, 2):
        raise ValueError(f"Expected Hessian field shape (2,2,H,W), got {field.shape}")

    # gradient^* = -divergence.  Apply this once to each row of the Hessian
    # field and then once more to the resulting two-component vector field.
    first_adjoint = np.stack((-divergence(field[0]), -divergence(field[1])), axis=0)
    return -divergence(first_adjoint)


def hessian_value(image: Array) -> float:
    """Return the isotropic Hessian-Frobenius seminorm of one image."""

    h = hessian(np.asarray(image, dtype=np.float64))
    return float(np.sum(np.sqrt(np.sum(h * h, axis=(0, 1)))))


def sequence_hessian_value(sequence: Array) -> float:
    """Return the summed Hessian seminorm over a video sequence."""

    sequence = np.asarray(sequence, dtype=np.float64)
    return float(sum(hessian_value(frame) for frame in sequence))


def project_hessian_dual(dual: Array, radius: float) -> Array:
    """Project a Hessian dual field onto pointwise Frobenius balls."""

    dual = np.asarray(dual, dtype=np.float64)
    if dual.ndim != 4 or dual.shape[:2] != (2, 2):
        raise ValueError(f"Expected Hessian dual shape (2,2,H,W), got {dual.shape}")
    norm = np.sqrt(np.sum(dual * dual, axis=(0, 1)))
    return dual / np.maximum(1.0, norm / max(float(radius), 1e-30))[None, None, :, :]


def check_hessian_adjoint(image_shape: tuple[int, int], seed: int = 0) -> float:
    """Return ``<H u,q> - <u,H^* q>`` for a random numerical adjoint check."""

    rng = np.random.default_rng(seed)
    image = rng.normal(size=image_shape)
    field = rng.normal(size=(2, 2, *image_shape))
    return float(np.sum(hessian(image) * field) - np.sum(image * hessian_adjoint(field)))
