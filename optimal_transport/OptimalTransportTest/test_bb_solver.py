import numpy as np

from solvers import (
    _continuity,
    _continuity_adjoint,
    _prox_kinetic_perspective,
    solve_bb_pair,
)
from operators import project_nonnegative_mass


def test_continuity_adjoint():
    rng = np.random.default_rng(4)
    slices, height, width = 5, 7, 6
    dt = 1.0 / (slices - 1)
    rho = rng.normal(size=(slices, height, width))
    momentum = rng.normal(size=(slices - 1, 2, height, width))
    phi = rng.normal(size=(slices - 1, height, width))
    rho_adj, momentum_adj = _continuity_adjoint(phi, dt)
    lhs = np.sum(_continuity(rho, momentum, dt) * phi)
    rhs = np.sum(rho * rho_adj) + np.sum(momentum * momentum_adj)
    assert np.isclose(lhs, rhs, rtol=1e-12, atol=1e-12)


def test_kinetic_prox_optimality():
    rng = np.random.default_rng(5)
    for _ in range(50):
        rho0 = rng.normal(size=(8, 7))
        momentum0 = rng.normal(size=(2, 8, 7))
        gamma = 10.0 ** rng.uniform(-3.0, 1.0)
        rho, momentum = _prox_kinetic_perspective(rho0, momentum0, gamma)
        assert np.all(rho >= 0.0)
        active = rho > 1e-10
        scalar_kkt = rho - rho0 - gamma * np.sum(momentum0 ** 2, axis=0) / (
            2.0 * (rho + gamma) ** 2
        )
        vector_kkt = momentum - rho[None] * momentum0 / (rho + gamma)[None]
        assert np.max(np.abs(scalar_kkt[active]), initial=0.0) < 1e-10
        assert np.max(np.abs(vector_kkt[:, active]), initial=0.0) < 1e-10


def test_identical_endpoints_have_zero_action():
    rng = np.random.default_rng(6)
    image = rng.random((12, 12))
    zero = np.zeros_like(image)
    left, right, _, info = solve_bb_pair(
        image,
        image,
        zero,
        zero,
        beta=0.1,
        eta=2.0,
        slices=5,
        max_iter=100,
        tol=1e-8,
    )
    assert np.allclose(left, image, atol=1e-10)
    assert np.allclose(right, image, atol=1e-10)
    assert info["transport_action"] < 1e-12
    assert info["continuity_residual"] < 1e-12


def test_mass_projection_is_feasible_and_has_kkt_threshold():
    rng = np.random.default_rng(7)
    image = rng.normal(size=(20, 18))
    projected = project_nonnegative_mass(image, target_mass=11.5)
    assert np.all(projected >= 0.0)
    assert np.isclose(projected.sum(), 11.5, rtol=1e-12, atol=1e-12)
    active = projected > 0.0
    thresholds = image[active] - projected[active]
    assert np.max(thresholds) - np.min(thresholds) < 1e-12
