"""Core consistency checks for the standalone OT/UOT package."""

from __future__ import annotations

import unittest

import numpy as np

from ot_uot.core.finite_differences import check_adjoint
from ot_uot.regularizers.hessian import check_hessian_adjoint, hessian_value
from ot_uot.core.variables import ImageResidualState, TransportState
from ot_uot.core.visibility import ComplexVisibilityDataTerm, DirectVisibilityOperator


class CoreConsistencyTests(unittest.TestCase):
    def test_gradient_divergence_adjoint(self) -> None:
        self.assertLess(abs(check_adjoint((8, 9), seed=5)), 1e-10)

    def test_hessian_adjoint(self) -> None:
        self.assertLess(abs(check_hessian_adjoint((8, 9), seed=7)), 1e-10)

    def test_constant_image_has_zero_hessian(self) -> None:
        self.assertAlmostEqual(hessian_value(np.ones((8, 9))), 0.0, places=12)

    def test_direct_visibility_adjoint(self) -> None:
        rng = np.random.default_rng(10)
        u = rng.normal(size=12) * 1e9
        v = rng.normal(size=12) * 1e9
        weight = np.exp(rng.normal(size=12))
        operator = DirectVisibilityOperator(
            u=u,
            v=v,
            weight=weight,
            shape=(6, 6),
            fov_rad=160e-6 / 206265.0,
            data_scale=1.0,
            use_cache=False,
        )
        data_term = ComplexVisibilityDataTerm(
            operator=operator,
            observed=np.zeros(12, dtype=np.complex128),
        )
        self.assertLess(abs(data_term.check_adjoint(seed=11)), 1e-10)

    def test_image_residual_state_decomposition(self) -> None:
        background = np.ones((4, 4))
        positive = np.zeros((3, 4, 4))
        negative = np.zeros((3, 4, 4))
        positive[:, 1, 1] = 0.25
        image = background[None, :, :] + positive - negative
        state = ImageResidualState(
            image=image,
            positive=positive,
            negative=negative,
            background=background,
        )
        self.assertEqual(state.max_decomposition_error(), 0.0)

    def test_transport_state_shapes(self) -> None:
        density = np.ones((5, 4, 4))
        momentum = np.zeros((4, 2, 4, 4))
        source = np.zeros((4, 4, 4))
        TransportState(density=density, momentum=momentum, source=source)


if __name__ == "__main__":
    unittest.main()

