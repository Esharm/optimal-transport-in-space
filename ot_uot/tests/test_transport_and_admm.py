"""Transport and ADMM verification tests."""

from __future__ import annotations

import unittest

import numpy as np

from ot_uot.core.config import ImageGrid, TransportMethod, UOTParameters
from ot_uot.core.visibility import ComplexVisibilityDataTerm, DirectVisibilityOperator
from ot_uot.optimization.objective import constraint_residuals, objective_breakdown
from ot_uot.optimization.signed_residual_admm import SignedResidualUOTADMM
from ot_uot.transport.continuity import check_continuity_adjoint, uot_continuity
from ot_uot.transport.global_velocity import GlobalVelocityUOTTransport, initialize_global_state
from ot_uot.transport.kinetic_prox import prox_kinetic_perspective
from ot_uot.transport.pairwise import PairwiseUOTTransport, initialize_pairwise_state
from ot_uot.transport.path_solver import solve_uot_path


def zero_data_terms(frames: int, shape: tuple[int, int]) -> list[ComplexVisibilityDataTerm]:
    grid = ImageGrid(height=shape[0], width=shape[1], fov_rad=1.0)
    terms = []
    for _ in range(frames):
        op = DirectVisibilityOperator(
            u=np.zeros(0),
            v=np.zeros(0),
            weight=np.zeros(0),
            shape=grid.shape,
            fov_rad=grid.fov_rad,
        )
        terms.append(ComplexVisibilityDataTerm(operator=op, observed=np.zeros(0, dtype=np.complex128)))
    return terms


class TransportVerificationTests(unittest.TestCase):
    def test_continuity_adjoint(self) -> None:
        self.assertLess(abs(check_continuity_adjoint(nodes=4, shape=(5, 6), seed=21)), 1e-10)

    def test_kinetic_prox_zero_momentum_stays_zero(self) -> None:
        density = np.ones((5, 5))
        momentum = np.zeros((2, 5, 5))
        out_density, out_momentum = prox_kinetic_perspective(density, momentum, gamma=0.4)
        self.assertTrue(np.all(out_density >= 0.0))
        self.assertLess(np.linalg.norm(out_momentum), 1e-14)

    def test_constant_path_has_small_action_and_continuity(self) -> None:
        target = np.ones((4, 5, 5))
        path, info = solve_uot_path(
            target=target,
            target_weight=np.ones_like(target),
            transport_weight=0.1,
            source_weight=10.0,
            max_iter=60,
            tol=1e-5,
        )
        residual = uot_continuity(path.density, path.momentum, path.source, dt=1.0 / 3.0)
        self.assertLess(np.linalg.norm(residual), 1e-6)
        self.assertLess(info.action, 1e-8)

    def test_source_penalty_controls_mass_creation(self) -> None:
        target = np.zeros((3, 5, 5))
        target[-1, 2, 2] = 1.0
        weights = np.zeros_like(target)
        weights[0] = 1.0
        weights[-1] = 1.0
        _, low_source_info = solve_uot_path(
            target=target,
            target_weight=weights,
            transport_weight=0.1,
            source_weight=0.1,
            max_iter=80,
            tol=1e-5,
        )
        _, high_source_info = solve_uot_path(
            target=target,
            target_weight=weights,
            transport_weight=0.1,
            source_weight=100.0,
            max_iter=80,
            tol=1e-5,
        )
        self.assertGreater(low_source_info.source_mass_abs, high_source_info.source_mass_abs)

    def test_identity_transport_sequence(self) -> None:
        target = np.zeros((5, 6, 6))
        target[:, 3, 3] = 1.0
        path, info = solve_uot_path(
            target=target,
            target_weight=np.ones_like(target),
            transport_weight=0.2,
            source_weight=50.0,
            max_iter=80,
            tol=1e-5,
        )
        self.assertLess(np.linalg.norm(path.momentum), 1e-5)
        self.assertLess(info.source_mass_abs, 1e-5)

    def test_pairwise_transport_shapes(self) -> None:
        channel = np.zeros((3, 5, 5))
        channel[0, 2, 2] = 1.0
        channel[1, 2, 3] = 1.0
        channel[2, 3, 3] = 1.0
        block = PairwiseUOTTransport(transport_weight=0.1, source_weight=5.0, endpoint_penalty=1.0, nodes=4, inner_iters=20, tol=1e-5)
        state = initialize_pairwise_state(frames=3, shape=(5, 5))
        next_state, info = block.update(channel, np.zeros_like(channel), state)
        self.assertEqual(len(next_state.positive.paths), 2)
        self.assertTrue(np.isfinite(info.total_action))
        target_sum, weight_sum = block.channel_quadratic_targets(next_state.positive, 3, (5, 5))
        self.assertEqual(target_sum.shape, channel.shape)
        self.assertEqual(weight_sum.shape, channel.shape)

    def test_global_transport_shapes(self) -> None:
        channel = np.zeros((3, 5, 5))
        channel[:, 2, 2] = 1.0
        block = GlobalVelocityUOTTransport(transport_weight=0.1, source_weight=5.0, endpoint_penalty=1.0, inner_iters=20, tol=1e-5)
        state = initialize_global_state(frames=3, shape=(5, 5))
        next_state, info = block.update(channel, np.zeros_like(channel), state)
        self.assertEqual(next_state.positive.path.density.shape, channel.shape)
        self.assertTrue(np.isfinite(info.total_action))
        target_sum, weight_sum = block.channel_quadratic_targets(next_state.positive)
        self.assertEqual(target_sum.shape, channel.shape)
        self.assertEqual(weight_sum.shape, channel.shape)


class ADMMIntegrationTests(unittest.TestCase):
    def _run_small_problem(self, method: TransportMethod):
        shape = (6, 6)
        frames = 3
        static = np.zeros((frames, *shape))
        static[:, 2, 2] = 1.0
        static[1, 2, 3] = 0.5
        static[2, 3, 3] = 0.5
        background = np.mean(static, axis=0)
        params = UOTParameters(
            transport_method=method,
            tv_weight=0.0,
            background_weight=1e-4,
            residual_mass_weight=1e-5,
            transport_weight=1e-3,
            source_weight=5.0,
            decomposition_penalty=0.5,
            endpoint_penalty=0.5,
            transport_nodes=4,
            transport_inner_iters=15,
            image_inner_iters=12,
            max_admm_iters=3,
            min_admm_iters=1,
            abs_tol=1e-5,
            rel_tol=1e-3,
            patience=10,
        )
        solver = SignedResidualUOTADMM(params)
        state = solver.run(static, background, zero_data_terms(frames, shape))
        breakdown = objective_breakdown(state.image_state, state.transport_state, zero_data_terms(frames, shape), params)
        residuals = constraint_residuals(state.image_state, state.transport_state, method)
        self.assertEqual(state.iteration, 3)
        self.assertEqual(len(state.history), 3)
        self.assertTrue(np.isfinite(breakdown.total))
        self.assertTrue(np.isfinite(residuals.primal_l2))
        self.assertTrue(np.all(state.image_state.image >= -1e-12))
        self.assertTrue(np.all(state.image_state.positive >= -1e-12))
        self.assertTrue(np.all(state.image_state.negative >= -1e-12))

    def test_pairwise_admm_smoke(self) -> None:
        self._run_small_problem(TransportMethod.PAIRWISE_UOT)

    def test_global_admm_smoke(self) -> None:
        self._run_small_problem(TransportMethod.GLOBAL_VELOCITY)


if __name__ == "__main__":
    unittest.main()
