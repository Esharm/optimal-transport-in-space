import numpy as np

from operators import project_nonnegative_mass
from solvers import dual_step, image_step, transport_step


class ADMM:
    """ADMM for spatial reconstruction with optional full-image dynamic-OT splitting.

    Objective, when beta > 0 and eta > 0, is approximately

        sum_k [ D_k(u_k) + spatial_regularizer(u_k)
              + prior_weight/2 ||u_k - prior||^2 ]
        + beta * sum_k BB(u_k, u_{k+1})

    The BB term is handled by splitting endpoint variables b0^k, b1^k and
    enforcing b0^k ~= u_k, b1^k ~= u_{k+1} through an augmented Lagrangian.
    """

    def __init__(
        self,
        data_terms,
        regularizer,
        beta=0.0,
        eta=0.0,
        max_iter=30,
        abs_tol=1e-4,
        rel_tol=5e-3,
        min_iter=5,
        patience=3,
        transport_T=7,
        transport_inner_iters=200,
        transport_tol=2e-4,
        dual_relaxation=1.0,
        enforce_equal_mass=False,
        prior_image=None,
        prior_weight=0.0,
        stop_on_data_plateau=False,
        data_plateau_window=5,
        data_plateau_tol=1e-3,
        **kwargs,
    ):
        self.data_terms = data_terms
        self.regularizer = regularizer
        self.beta = float(beta)
        self.eta = float(eta)
        self.max_iter = int(max_iter)
        self.abs_tol = float(abs_tol)
        self.rel_tol = float(rel_tol)
        self.min_iter = int(min_iter)
        self.patience = int(patience)
        self.transport_T = int(transport_T)
        self.transport_inner_iters = int(transport_inner_iters)
        self.transport_tol = float(transport_tol)
        self.dual_relaxation = float(dual_relaxation)
        self.enforce_equal_mass = bool(enforce_equal_mass)
        self.prior_image = None if prior_image is None else np.asarray(prior_image, dtype=np.float64)
        self.prior_weight = float(prior_weight)
        self.stop_on_data_plateau = bool(stop_on_data_plateau)
        self.data_plateau_window = int(data_plateau_window)
        self.data_plateau_tol = float(data_plateau_tol)

    def _primal_residual(self, u, b0, b1):
        if len(u) <= 1:
            return 0.0
        return float(np.sqrt(np.sum((b0 - u[:-1]) ** 2) + np.sum((b1 - u[1:]) ** 2)))

    def _dual_residual(self, b0, b1, b0_prev, b1_prev):
        if b0_prev is None or b1_prev is None:
            return np.inf
        return float(self.eta * np.sqrt(
            np.sum((b0 - b0_prev) ** 2) + np.sum((b1 - b1_prev) ** 2)
        ))

    def _eps_primal(self, u, b0, b1):
        if b0.size == 0:
            return 0.0
        n = b0.size + b1.size
        duplicated_u_norm = np.sqrt(np.sum(u[:-1] ** 2) + np.sum(u[1:] ** 2))
        b_norm = np.sqrt(np.sum(b0 ** 2) + np.sum(b1 ** 2))
        return float(np.sqrt(n) * self.abs_tol + self.rel_tol * max(duplicated_u_norm, b_norm))

    def _eps_dual(self, lam0, lam1):
        if lam0.size == 0:
            return 0.0
        n = lam0.size + lam1.size
        dual_norm = np.sqrt(np.sum(lam0 ** 2) + np.sum(lam1 ** 2))
        return float(np.sqrt(n) * self.abs_tol + self.rel_tol * dual_norm)

    def _project_sequence_mass(self, u, target_mass):
        if target_mass is None:
            return u
        return np.stack([
            project_nonnegative_mass(frame, target_mass=target_mass) for frame in u
        ])

    def _data_loss(self, u):
        return float(sum(term.loss(u[k]) for k, term in enumerate(self.data_terms)))

    def run(self, u):
        u = np.asarray(u, dtype=np.float64).copy()
        frames, height, width = u.shape
        if len(self.data_terms) != frames:
            raise ValueError("Number of data terms must equal number of frames")

        if self.enforce_equal_mass:
            target_mass = max(float(np.median(np.sum(np.maximum(u, 0.0), axis=(1, 2)))), 1e-12)
            u = self._project_sequence_mass(u, target_mass)
            print(f"Using fixed OT mass per frame: {target_mass:.6e}")
        else:
            target_mass = None
            u = np.maximum(u, 0.0)

        lam0 = np.zeros((frames - 1, height, width), dtype=np.float64)
        lam1 = np.zeros_like(lam0)
        b0_prev = b1_prev = None
        transport_state = None
        converged_count = 0
        history = []

        for iteration in range(1, self.max_iter + 1):
            b0, b1, transport_state, transport_info = transport_step(
                u=u,
                lam0=lam0,
                lam1=lam1,
                beta=self.beta,
                eta=self.eta,
                T=self.transport_T,
                inner_iters=self.transport_inner_iters,
                tol=self.transport_tol,
                state=transport_state,
                return_state=True,
            )

            u = image_step(
                u=u,
                data_terms=self.data_terms,
                b0=b0,
                b1=b1,
                lam0=lam0,
                lam1=lam1,
                regularizer=self.regularizer,
                eta=self.eta,
                target_mass=target_mass,
                prior_image=self.prior_image,
                prior_weight=self.prior_weight,
            )

            if self.enforce_equal_mass:
                u = self._project_sequence_mass(u, target_mass)

            lam0, lam1 = dual_step(
                u, b0, b1, lam0, lam1, self.eta, self.dual_relaxation
            )

            primal = self._primal_residual(u, b0, b1)
            dual = self._dual_residual(b0, b1, b0_prev, b1_prev)
            eps_primal = self._eps_primal(u, b0, b1)
            eps_dual = self._eps_dual(lam0, lam1)
            data_loss = self._data_loss(u)
            action = float(sum(item.get("transport_action", 0.0) for item in transport_info))
            bb_continuity = float(max(
                (item.get("continuity_residual", 0.0) for item in transport_info), default=0.0
            ))
            bb_iterations = int(max(
                (item.get("iterations", 0) for item in transport_info), default=0
            ))

            row = {
                "iter": iteration,
                "r_norm": primal,
                "s_norm": dual,
                "eps_pri": eps_primal,
                "eps_dual": eps_dual,
                "data_loss": data_loss,
                "transport_action": action,
                "bb_continuity_residual_max": bb_continuity,
                "bb_iterations_max": bb_iterations,
                "u_mean": float(u.mean()),
                "u_max": float(u.max()),
                "u_min": float(u.min()),
            }
            history.append(row)

            print(
                f"ADMM {iteration:03d} | r={primal:.3e}/{eps_primal:.3e} "
                f"| s={dual:.3e}/{eps_dual:.3e} | data={data_loss:.3e} "
                f"| BB action={action:.3e} cont={bb_continuity:.3e} ({bb_iterations} iters)"
            )

            if self.eta > 0 and iteration >= self.min_iter:
                converged_count = converged_count + 1 if (
                    primal <= eps_primal
                    and dual <= eps_dual
                    and bb_continuity <= max(self.transport_tol, 1e-8)
                ) else 0
                if converged_count >= self.patience:
                    print(f"Converged after {iteration} ADMM iterations")
                    break

            if self.stop_on_data_plateau and self.eta <= 0 and len(history) >= self.data_plateau_window:
                recent = [item["data_loss"] for item in history[-self.data_plateau_window:]]
                improvement = abs(recent[0] - recent[-1]) / (abs(recent[0]) + 1e-12)
                if iteration >= self.min_iter and improvement < self.data_plateau_tol:
                    print(f"Stopped after {iteration} iterations: data loss plateaued")
                    break

            b0_prev, b1_prev = b0.copy(), b1.copy()

        return u, history


class ResidualSignedOTADMM(ADMM):
    """Compatibility class so `from admm import ResidualSignedOTADMM` works.

    IMPORTANT: with the solvers.py you pasted, there are no residual-channel
    transport/image-update routines. Therefore this class deliberately runs the
    same stable full-image balanced OT ADMM as ADMM above.

    This fixes the import error and gives you an OT refinement on the actual
    nonnegative images u_k. It is not the experimental delta+/delta- OT variant.
    """

    def __init__(
        self,
        data_terms,
        regularizer,
        prior_image=None,
        beta=0.0,
        eta=0.0,
        max_iter=30,
        abs_tol=1e-4,
        rel_tol=5e-3,
        min_iter=5,
        patience=3,
        transport_T=7,
        transport_inner_iters=200,
        transport_tol=2e-4,
        dual_relaxation=1.0,
        prior_weight=0.0,
        enforce_equal_mass=False,
        stop_on_data_plateau=False,
        data_plateau_window=5,
        data_plateau_tol=1e-3,
        **kwargs,
    ):
        super().__init__(
            data_terms=data_terms,
            regularizer=regularizer,
            beta=beta,
            eta=eta,
            max_iter=max_iter,
            abs_tol=abs_tol,
            rel_tol=rel_tol,
            min_iter=min_iter,
            patience=patience,
            transport_T=transport_T,
            transport_inner_iters=transport_inner_iters,
            transport_tol=transport_tol,
            dual_relaxation=dual_relaxation,
            enforce_equal_mass=enforce_equal_mass,
            prior_image=prior_image,
            prior_weight=prior_weight,
            stop_on_data_plateau=stop_on_data_plateau,
            data_plateau_window=data_plateau_window,
            data_plateau_tol=data_plateau_tol,
        )