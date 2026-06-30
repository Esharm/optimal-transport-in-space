import numpy as np

from operators import grad, div, hessian, div2, project_nonnegative_mass


class TotalVariationRegularizer:
    """Primal-dual image update for data + isotropic TV + quadratic target."""

    def __init__(self, alpha=3e-5, iters=50, tau=1.0, sigma=0.25,
                 auto_step=True, power_iters=12):
        self.alpha = float(alpha)
        self.iters = int(iters)
        self.tau = float(tau)
        self.sigma = float(sigma)
        self.auto_step = bool(auto_step)
        self.power_iters = int(power_iters)
        self._lipschitz_cache = {}

    def _data_lipschitz(self, data_term, shape):
        """Power estimate of ||S* S||, cached for each fixed data term."""
        key = id(data_term)
        if key in self._lipschitz_cache:
            return self._lipschitz_cache[key]
        rng = np.random.default_rng(1729)
        vector = rng.normal(size=shape)
        vector /= np.linalg.norm(vector) + 1e-30
        eigenvalue = 0.0
        for _ in range(self.power_iters):
            applied = data_term.sampler.adjoint(data_term.sampler.forward(vector))
            norm = np.linalg.norm(applied)
            if norm <= 1e-30:
                eigenvalue = 0.0
                break
            vector = applied / norm
            eigenvalue = float(np.sum(vector * applied))
        # Power iteration approaches from below; retain numerical safety slack.
        estimate = max(1.1 * eigenvalue, 1e-12)
        self._lipschitz_cache[key] = estimate
        return estimate

    def projected_gradient_residual(
        self, u, data_term, admm_target, admm_weight, target_mass=None
    ):
        """Relative KKT residual for the smooth constrained problem (alpha=0)."""
        lipschitz = self._data_lipschitz(data_term, u.shape) + admm_weight
        step = min(self.tau, 0.99 / lipschitz)
        gradient = data_term.gradient(u)
        if admm_weight > 0:
            gradient += admm_weight * (u - admm_target)
        projected = project_nonnegative_mass(
            u - step * gradient, target_mass=target_mass
        )
        mapping_norm = np.linalg.norm(u - projected) / step
        return float(mapping_norm / max(1.0, np.linalg.norm(u)))

    def solve(self, u_init, data_term, admm_target, admm_weight, target_mass=None):
        uk = np.asarray(u_init, dtype=np.float64).copy()
        smooth_lipschitz = self._data_lipschitz(data_term, uk.shape) + admm_weight

        # With no TV this is a smooth bound-constrained quadratic. FISTA uses
        # the full 1/L step and is much faster than carrying the irrelevant TV
        # dual step-size restriction into the data-only problem.
        if self.alpha <= 0:
            step = min(self.tau, 0.99 / smooth_lipschitz)
            extrapolated = uk.copy()
            momentum_parameter = 1.0
            for _ in range(self.iters):
                old = uk
                smooth_gradient = data_term.gradient(extrapolated)
                if admm_weight > 0:
                    smooth_gradient += admm_weight * (extrapolated - admm_target)
                uk = project_nonnegative_mass(
                    extrapolated - step * smooth_gradient,
                    target_mass=target_mass,
                )
                next_parameter = 0.5 * (
                    1.0 + np.sqrt(1.0 + 4.0 * momentum_parameter ** 2)
                )
                extrapolated_next = uk + (
                    (momentum_parameter - 1.0) / next_parameter
                ) * (uk - old)
                # Gradient-based restart prevents oscillatory FISTA behavior.
                if np.sum((extrapolated - uk) * (uk - old)) > 0:
                    momentum_parameter = 1.0
                    extrapolated = uk.copy()
                else:
                    momentum_parameter = next_parameter
                    extrapolated = extrapolated_next
            return uk

        uk_bar = uk.copy()
        cache_key = id(data_term)
        if not hasattr(self, "_dual_cache"):
            self._dual_cache = {}
        p = self._dual_cache.get(cache_key)
        if p is None or p.shape != (2, *uk.shape):
            p = np.zeros((2, *uk.shape), dtype=np.float64)
        if self.auto_step:
            # ||grad||^2 <= 8 for forward differences in two dimensions.
            safe_tau = 0.99 / (0.5 * smooth_lipschitz + 8.0 * self.sigma)
            tau = min(self.tau, safe_tau)
        else:
            tau = self.tau / (1.0 + admm_weight)

        for _ in range(self.iters):
            if self.alpha > 0:
                p += self.sigma * grad(uk_bar)
                norm_p = np.sqrt(np.sum(p * p, axis=0))
                p /= np.maximum(1.0, norm_p / self.alpha)[None, :, :]
            else:
                p.fill(0.0)

            old = uk.copy()
            smooth_gradient = data_term.gradient(uk)
            if admm_weight > 0:
                smooth_gradient += admm_weight * (uk - admm_target)
            uk -= tau * (smooth_gradient - div(p))
            uk = project_nonnegative_mass(uk, target_mass=target_mass)
            uk_bar = 2.0 * uk - old
        self._dual_cache[cache_key] = p.copy()
        return uk


class HessianRegularizer:
    """Optional second-order spatial baseline."""

    def __init__(self, alpha=1e-5, iters=50, tau=1e-3, sigma=5e-3):
        self.alpha = float(alpha)
        self.iters = int(iters)
        self.tau = float(tau)
        self.sigma = float(sigma)

    def solve(self, u_init, data_term, admm_target, admm_weight, target_mass=None):
        uk = np.asarray(u_init, dtype=np.float64).copy()
        uk_bar = uk.copy()
        q = np.zeros((4, *uk.shape), dtype=np.float64)
        tau = self.tau / (1.0 + admm_weight)

        for _ in range(self.iters):
            q += self.sigma * hessian(uk_bar)
            norm_q = np.sqrt(np.sum(q * q, axis=0))
            q /= np.maximum(1.0, norm_q / self.alpha)[None, :, :]
            old = uk.copy()
            smooth_gradient = data_term.gradient(uk)
            if admm_weight > 0:
                smooth_gradient += admm_weight * (uk - admm_target)
            uk -= tau * (smooth_gradient + div2(q))
            uk = project_nonnegative_mass(uk, target_mass=target_mass)
            uk_bar = 2.0 * uk - old
        return uk


def _continuity(rho, momentum, dt):
    """K(rho,m) = forward-time density difference + spatial divergence."""
    return (rho[1:] - rho[:-1]) / dt + np.stack(
        [div(momentum[t]) for t in range(momentum.shape[0])]
    )


def _continuity_adjoint(phi, dt):
    """Exact Euclidean adjoint K* for the project's grad/div convention."""
    intervals, height, width = phi.shape
    rho_adj = np.zeros((intervals + 1, height, width), dtype=np.float64)
    rho_adj[0] = -phi[0] / dt
    rho_adj[-1] = phi[-1] / dt
    if intervals > 1:
        rho_adj[1:-1] = (phi[:-1] - phi[1:]) / dt

    # Here div = -grad*, hence (div)* = -grad.
    momentum_adj = np.stack([-grad(phi[t]) for t in range(intervals)])
    return rho_adj, momentum_adj


def _prox_kinetic_perspective(rho0, momentum0, gamma, newton_iters=12):
    """Exact prox of gamma*|m|^2/(2*rho), with rho >= 0.

    At each pixel, the positive-density solution obeys

        rho-rho0-gamma*|m0|^2/(2*(rho+gamma)^2) = 0,
        m = rho*m0/(rho+gamma).

    The scalar equation is strictly increasing. Safeguarded Newton therefore
    gives its unique nonnegative root; pixels whose unconstrained root is
    nonpositive map to (rho,m)=(0,0).
    """
    if gamma <= 0:
        return np.maximum(rho0, 0.0), momentum0.copy()

    momentum_sq = np.sum(momentum0 * momentum0, axis=0)
    active = rho0 + momentum_sq / (2.0 * gamma) > 0.0
    # On the active set f(0)<0. Since f is increasing and concave, Newton
    # started at zero moves monotonically toward the positive root without a
    # potentially fragile finite upper bracket.
    rho = np.zeros_like(rho0)
    for _ in range(max(newton_iters, 25)):
        denom = rho + gamma
        value = rho - rho0 - gamma * momentum_sq / (2.0 * denom * denom)
        derivative = 1.0 + gamma * momentum_sq / (denom ** 3)
        candidate = rho - value / derivative
        rho = np.where(active, np.maximum(candidate, 0.0), 0.0)

    rho = np.where(active, np.maximum(rho, 0.0), 0.0)
    momentum = momentum0 * (rho / (rho + gamma + 1e-30))[None, :, :]
    momentum[:, ~active] = 0.0
    return rho, momentum


def _transport_action(rho, momentum, beta, dt, density_floor=1e-12):
    density = np.maximum(rho[:-1], density_floor)
    kinetic = np.sum(momentum * momentum, axis=1) / (2.0 * density)
    return float(beta * dt * np.sum(kinetic))


def _initial_transport_state(left, right, slices):
    rho = np.stack([
        (1.0 - t / (slices - 1)) * left + (t / (slices - 1)) * right
        for t in range(slices)
    ])
    return {
        "rho": np.maximum(rho, 1e-12),
        "momentum": np.zeros((slices - 1, 2, *left.shape), dtype=np.float64),
        "phi": np.zeros((slices - 1, *left.shape), dtype=np.float64),
    }


def solve_bb_pair(
    left,
    right,
    dual_left,
    dual_right,
    beta,
    eta,
    slices=7,
    max_iter=200,
    tol=2e-4,
    state=None,
    check_every=10,
):
    """Solve one convex, time-discrete Benamou-Brenier ADMM subproblem.

    The discretization places rho on T time nodes and m on the T-1 forward
    intervals. Its kinetic action uses the left endpoint density. This is a
    first-order, convex, consistent discretization of dynamic W2.
    """
    if slices < 2:
        raise ValueError("slices must be at least 2")
    if beta <= 0 or eta <= 0:
        raise ValueError("solve_bb_pair requires beta > 0 and eta > 0")

    target_left = np.maximum(left - dual_left / eta, 0.0)
    target_right = np.maximum(right - dual_right / eta, 0.0)
    if state is None or state["rho"].shape[0] != slices:
        state = _initial_transport_state(target_left, target_right, slices)

    rho = np.maximum(np.asarray(state["rho"], dtype=np.float64).copy(), 0.0)
    momentum = np.asarray(state["momentum"], dtype=np.float64).copy()
    phi = np.asarray(state["phi"], dtype=np.float64).copy()
    dt = 1.0 / (slices - 1)

    # ||D_t||^2 <= 4/dt^2 and ||div||^2 <= 8 in 2-D. This conservative
    # analytic bound avoids slow power iterations and guarantees tau*sigma*||K||^2 < 1.
    operator_bound_sq = 4.0 / (dt * dt) + 8.0
    step = 0.99 / np.sqrt(operator_bound_sq)
    # Endpoint quadratics are handled by forward gradients, requiring tau*eta < 2.
    tau = min(step, 1.9 / eta)
    sigma = 0.99 / (tau * operator_bound_sq)

    rho_bar = rho.copy()
    momentum_bar = momentum.copy()
    last_change = np.inf
    continuity_norm = np.inf
    iterations = max_iter

    for iteration in range(1, max_iter + 1):
        phi += sigma * _continuity(rho_bar, momentum_bar, dt)
        rho_adj, momentum_adj = _continuity_adjoint(phi, dt)

        old_rho = rho.copy()
        old_momentum = momentum.copy()
        rho_trial = rho - tau * rho_adj
        momentum_trial = momentum - tau * momentum_adj

        # Smooth endpoint terms from the scaled ADMM augmented Lagrangian.
        rho_trial[0] -= tau * eta * (rho[0] - target_left)
        rho_trial[-1] -= tau * eta * (rho[-1] - target_right)

        gamma = tau * beta * dt
        for t in range(slices - 1):
            rho[t], momentum[t] = _prox_kinetic_perspective(
                rho_trial[t], momentum_trial[t], gamma
            )
        rho[-1] = np.maximum(rho_trial[-1], 0.0)

        rho_bar = 2.0 * rho - old_rho
        momentum_bar = 2.0 * momentum - old_momentum

        if iteration % check_every == 0 or iteration == max_iter:
            delta_sq = np.sum((rho - old_rho) ** 2) + np.sum(
                (momentum - old_momentum) ** 2
            )
            scale_sq = np.sum(old_rho ** 2) + np.sum(old_momentum ** 2)
            last_change = np.sqrt(delta_sq / (scale_sq + 1e-30))
            residual = _continuity(rho, momentum, dt)
            continuity_norm = np.linalg.norm(residual) / (
                np.linalg.norm(rho) + np.linalg.norm(momentum) + 1e-30
            )
            if max(last_change, continuity_norm) < tol:
                iterations = iteration
                break

    new_state = {"rho": rho, "momentum": momentum, "phi": phi}
    info = {
        "iterations": iterations,
        "relative_change": float(last_change),
        "continuity_residual": float(continuity_norm),
        "transport_action": _transport_action(rho, momentum, beta, dt),
    }
    return rho[0].copy(), rho[-1].copy(), new_state, info


def transport_step(
    u,
    lam0,
    lam1,
    beta=0.0,
    eta=0.0,
    T=7,
    inner_iters=200,
    tol=2e-4,
    state=None,
    return_state=False,
):
    """Solve all adjacent BB subproblems, retaining optional warm-start state."""
    frames, height, width = u.shape
    b0 = np.zeros((frames - 1, height, width), dtype=np.float64)
    b1 = np.zeros_like(b0)

    if frames <= 1:
        result = (b0, b1, [], []) if return_state else (b0, b1)
        return result
    if beta <= 0 or eta <= 0:
        b0[:] = u[:-1]
        b1[:] = u[1:]
        result = (b0, b1, [], []) if return_state else (b0, b1)
        return result

    if state is None or len(state) != frames - 1:
        state = [None] * (frames - 1)
    new_state, infos = [], []
    for k in range(frames - 1):
        b0[k], b1[k], pair_state, info = solve_bb_pair(
            left=u[k],
            right=u[k + 1],
            dual_left=lam0[k],
            dual_right=lam1[k],
            beta=beta,
            eta=eta,
            slices=T,
            max_iter=inner_iters,
            tol=tol,
            state=state[k],
        )
        new_state.append(pair_state)
        infos.append(info)
    return (b0, b1, new_state, infos) if return_state else (b0, b1)


def image_step(
    u,
    data_terms,
    b0,
    b1,
    lam0,
    lam1,
    regularizer,
    eta=0.0,
    target_mass=None,
    prior_image=None,
    prior_weight=0.0,
):
    frames, height, width = u.shape
    result = u.copy()
    use_prior = prior_image is not None and prior_weight > 0.0
    if use_prior:
        prior_image = np.asarray(prior_image, dtype=np.float64)

    for k in range(frames):
        target_sum = np.zeros((height, width), dtype=np.float64)
        total_weight = 0.0
        if k < frames - 1 and eta > 0:
            target_sum += eta * (b0[k] + lam0[k] / eta)
            total_weight += eta
        if k > 0 and eta > 0:
            target_sum += eta * (b1[k - 1] + lam1[k - 1] / eta)
            total_weight += eta
        if use_prior:
            prior = prior_image if prior_image.ndim == 2 else prior_image[k]
            target_sum += prior_weight * prior
            total_weight += prior_weight

        target = target_sum / total_weight if total_weight > 0 else target_sum
        result[k] = regularizer.solve(
            u_init=u[k],
            data_term=data_terms[k],
            admm_target=target,
            admm_weight=total_weight,
            target_mass=target_mass,
        )
    return result


def dual_step(u, b0, b1, lam0, lam1, eta=0.0, relaxation=1.0):
    if eta <= 0:
        return lam0, lam1
    for k in range(len(u) - 1):
        lam0[k] += relaxation * eta * (b0[k] - u[k])
        lam1[k] += relaxation * eta * (b1[k] - u[k + 1])
    return lam0, lam1
