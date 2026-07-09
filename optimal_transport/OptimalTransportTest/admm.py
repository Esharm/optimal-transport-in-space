import numpy as np

from operators import project_nonnegative_mass
from solvers import (
    dual_step,
    image_step,
    positive_residual_dual_step,
    positive_residual_image_step,
    positive_residual_transport_step,
    signed_residual_dual_step,
    signed_residual_image_step,
    transport_step,
    unbalanced_positive_residual_transport_step,
)


class ADMM:
    """ADMM for spatial reconstruction with optional dynamic-OT splitting."""

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
        self.prior_image = prior_image
        self.prior_weight = float(prior_weight)
        self.stop_on_data_plateau = bool(stop_on_data_plateau)
        self.data_plateau_window = int(data_plateau_window)
        self.data_plateau_tol = float(data_plateau_tol)

    def _primal_residual(self, u, b0, b1):
        return float(np.sqrt(np.sum((b0 - u[:-1]) ** 2) + np.sum((b1 - u[1:]) ** 2)))

    def _dual_residual(self, b0, b1, b0_prev, b1_prev):
        if b0_prev is None or b1_prev is None:
            return np.inf
        return float(self.eta * np.sqrt(
            np.sum((b0 - b0_prev) ** 2) + np.sum((b1 - b1_prev) ** 2)
        ))

    def _eps_primal(self, u, b0, b1):
        n = b0.size + b1.size
        duplicated_u_norm = np.sqrt(np.sum(u[:-1] ** 2) + np.sum(u[1:] ** 2))
        b_norm = np.sqrt(np.sum(b0 ** 2) + np.sum(b1 ** 2))
        return float(np.sqrt(n) * self.abs_tol + self.rel_tol * max(duplicated_u_norm, b_norm))

    def _eps_dual(self, lam0, lam1):
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
            action = float(sum(item["transport_action"] for item in transport_info))
            bb_continuity = float(max(
                (item["continuity_residual"] for item in transport_info), default=0.0
            ))
            bb_iterations = int(max(
                (item["iterations"] for item in transport_info), default=0
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
                    primal <= eps_primal and dual <= eps_dual
                    and bb_continuity <= max(self.transport_tol, 1e-8)
                ) else 0
                if converged_count >= self.patience:
                    print(f"Converged after {iteration} ADMM iterations")
                    break

            # Retained only for non-OT legacy experiments. Data plateau is not
            # a valid composite-objective stopping test when temporal OT is on.
            if self.stop_on_data_plateau and self.eta <= 0 and len(history) >= self.data_plateau_window:
                recent = [item["data_loss"] for item in history[-self.data_plateau_window:]]
                improvement = abs(recent[0] - recent[-1]) / (abs(recent[0]) + 1e-12)
                if iteration >= self.min_iter and improvement < self.data_plateau_tol:
                    print(f"Stopped after {iteration} iterations: data loss plateaued")
                    break

            b0_prev, b1_prev = b0.copy(), b1.copy()
        return u, history


class PositiveResidualOTADMM(ADMM):
    """ADMM with BB transport on positive residuals above a fixed background.

    The transported density is

        h_k = max(u_k - background, 0),

    while the image update still uses the original visibility data term on
    u_k. This makes the temporal OT penalty focus on the dynamic bright
    residual/hotspot component instead of the mostly static full image.

    This is still balanced BB transport on the residual channel. It is not UOT.
    """

    def __init__(
        self,
        data_terms,
        regularizer,
        background_image,
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
        prior_image=None,
        prior_weight=0.0,
        stop_on_data_plateau=False,
        data_plateau_window=5,
        data_plateau_tol=1e-3,
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
            enforce_equal_mass=False,
            prior_image=prior_image,
            prior_weight=prior_weight,
            stop_on_data_plateau=stop_on_data_plateau,
            data_plateau_window=data_plateau_window,
            data_plateau_tol=data_plateau_tol,
        )
        self.background_image = np.asarray(background_image, dtype=np.float64)

    def _positive_residual(self, u):
        return np.maximum(np.asarray(u, dtype=np.float64) - self.background_image[None, :, :], 0.0)

    def _primal_residual(self, u, b0, b1):
        residual = self._positive_residual(u)
        return float(np.sqrt(
            np.sum((b0 - residual[:-1]) ** 2)
            + np.sum((b1 - residual[1:]) ** 2)
        ))

    def _eps_primal(self, u, b0, b1):
        residual = self._positive_residual(u)
        n = b0.size + b1.size
        duplicated_residual_norm = np.sqrt(
            np.sum(residual[:-1] ** 2) + np.sum(residual[1:] ** 2)
        )
        b_norm = np.sqrt(np.sum(b0 ** 2) + np.sum(b1 ** 2))
        return float(np.sqrt(n) * self.abs_tol + self.rel_tol * max(duplicated_residual_norm, b_norm))

    def run(self, u):
        u = np.maximum(np.asarray(u, dtype=np.float64).copy(), 0.0)
        frames, height, width = u.shape
        if len(self.data_terms) != frames:
            raise ValueError("Number of data terms must equal number of frames")
        if self.background_image.shape != (height, width):
            raise ValueError("background_image shape must match frame shape")

        lam0 = np.zeros((frames - 1, height, width), dtype=np.float64)
        lam1 = np.zeros_like(lam0)
        b0_prev = b1_prev = None
        transport_state = None
        converged_count = 0
        history = []

        for iteration in range(1, self.max_iter + 1):
            b0, b1, transport_state, transport_info = positive_residual_transport_step(
                u=u,
                background=self.background_image,
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

            u = positive_residual_image_step(
                u=u,
                data_terms=self.data_terms,
                background=self.background_image,
                b0=b0,
                b1=b1,
                lam0=lam0,
                lam1=lam1,
                regularizer=self.regularizer,
                eta=self.eta,
                prior_image=self.prior_image,
                prior_weight=self.prior_weight,
            )

            lam0, lam1 = positive_residual_dual_step(
                u,
                self.background_image,
                b0,
                b1,
                lam0,
                lam1,
                self.eta,
                self.dual_relaxation,
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
            residual = self._positive_residual(u)
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
                "positive_residual_mass_mean": float(residual.sum(axis=(1, 2)).mean()),
                "positive_residual_mass_min": float(residual.sum(axis=(1, 2)).min()),
                "positive_residual_mass_max": float(residual.sum(axis=(1, 2)).max()),
            }
            history.append(row)
            print(
                f"Residual ADMM {iteration:03d} | r={primal:.3e}/{eps_primal:.3e} "
                f"| s={dual:.3e}/{eps_dual:.3e} | data={data_loss:.3e} "
                f"| BB action={action:.3e} cont={bb_continuity:.3e} ({bb_iterations} iters)"
            )

            if self.eta > 0 and iteration >= self.min_iter:
                converged_count = converged_count + 1 if (
                    primal <= eps_primal and dual <= eps_dual
                    and bb_continuity <= max(self.transport_tol, 1e-8)
                ) else 0
                if converged_count >= self.patience:
                    print(f"Converged after {iteration} residual ADMM iterations")
                    break

            if self.stop_on_data_plateau and self.eta <= 0 and len(history) >= self.data_plateau_window:
                recent = [item["data_loss"] for item in history[-self.data_plateau_window:]]
                improvement = abs(recent[0] - recent[-1]) / (abs(recent[0]) + 1e-12)
                if iteration >= self.min_iter and improvement < self.data_plateau_tol:
                    print(f"Stopped after {iteration} iterations: data loss plateaued")
                    break

            b0_prev, b1_prev = b0.copy(), b1.copy()
        return u, history


class SignedResidualUOTADMM(ADMM):
    """ADMM with unbalanced BB transport on signed residual channels.

    Around a fixed background a, define

        h_k^+ = max(u_k - a, 0),
        h_k^- = max(a - u_k, 0).

    Each nonnegative channel is transported with unbalanced BB:

        d_t rho + div(m) = s,

    with a quadratic source/sink penalty. The data term still acts on the full
    image u_k.
    """

    def __init__(
        self,
        data_terms,
        regularizer,
        background_image,
        beta=0.0,
        eta=0.0,
        source_weight=1.0,
        max_iter=30,
        abs_tol=1e-4,
        rel_tol=5e-3,
        min_iter=5,
        patience=3,
        transport_T=7,
        transport_inner_iters=200,
        transport_tol=2e-4,
        dual_relaxation=1.0,
        prior_image=None,
        prior_weight=0.0,
        stop_on_data_plateau=False,
        data_plateau_window=5,
        data_plateau_tol=1e-3,
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
            enforce_equal_mass=False,
            prior_image=prior_image,
            prior_weight=prior_weight,
            stop_on_data_plateau=stop_on_data_plateau,
            data_plateau_window=data_plateau_window,
            data_plateau_tol=data_plateau_tol,
        )
        self.background_image = np.asarray(background_image, dtype=np.float64)
        self.source_weight = float(source_weight)

    def _signed_residuals(self, u):
        u = np.asarray(u, dtype=np.float64)
        background = self.background_image[None, :, :]
        return np.maximum(u - background, 0.0), np.maximum(background - u, 0.0)

    def _primal_residual(self, u, pos_b0, pos_b1, neg_b0, neg_b1):
        pos, neg = self._signed_residuals(u)
        return float(np.sqrt(
            np.sum((pos_b0 - pos[:-1]) ** 2)
            + np.sum((pos_b1 - pos[1:]) ** 2)
            + np.sum((neg_b0 - neg[:-1]) ** 2)
            + np.sum((neg_b1 - neg[1:]) ** 2)
        ))

    def _dual_residual_signed(
        self,
        pos_b0,
        pos_b1,
        neg_b0,
        neg_b1,
        pos_b0_prev,
        pos_b1_prev,
        neg_b0_prev,
        neg_b1_prev,
    ):
        if (
            pos_b0_prev is None or pos_b1_prev is None
            or neg_b0_prev is None or neg_b1_prev is None
        ):
            return np.inf
        return float(self.eta * np.sqrt(
            np.sum((pos_b0 - pos_b0_prev) ** 2)
            + np.sum((pos_b1 - pos_b1_prev) ** 2)
            + np.sum((neg_b0 - neg_b0_prev) ** 2)
            + np.sum((neg_b1 - neg_b1_prev) ** 2)
        ))

    def _eps_primal_signed(self, u, pos_b0, pos_b1, neg_b0, neg_b1):
        pos, neg = self._signed_residuals(u)
        n = pos_b0.size + pos_b1.size + neg_b0.size + neg_b1.size
        residual_norm = np.sqrt(
            np.sum(pos[:-1] ** 2) + np.sum(pos[1:] ** 2)
            + np.sum(neg[:-1] ** 2) + np.sum(neg[1:] ** 2)
        )
        b_norm = np.sqrt(
            np.sum(pos_b0 ** 2) + np.sum(pos_b1 ** 2)
            + np.sum(neg_b0 ** 2) + np.sum(neg_b1 ** 2)
        )
        return float(np.sqrt(n) * self.abs_tol + self.rel_tol * max(residual_norm, b_norm))

    def _eps_dual_signed(self, pos_lam0, pos_lam1, neg_lam0, neg_lam1):
        n = pos_lam0.size + pos_lam1.size + neg_lam0.size + neg_lam1.size
        dual_norm = np.sqrt(
            np.sum(pos_lam0 ** 2) + np.sum(pos_lam1 ** 2)
            + np.sum(neg_lam0 ** 2) + np.sum(neg_lam1 ** 2)
        )
        return float(np.sqrt(n) * self.abs_tol + self.rel_tol * dual_norm)

    def run(self, u):
        u = np.maximum(np.asarray(u, dtype=np.float64).copy(), 0.0)
        frames, height, width = u.shape
        if len(self.data_terms) != frames:
            raise ValueError("Number of data terms must equal number of frames")
        if self.background_image.shape != (height, width):
            raise ValueError("background_image shape must match frame shape")

        pos_lam0 = np.zeros((frames - 1, height, width), dtype=np.float64)
        pos_lam1 = np.zeros_like(pos_lam0)
        neg_lam0 = np.zeros_like(pos_lam0)
        neg_lam1 = np.zeros_like(pos_lam0)

        pos_b0_prev = pos_b1_prev = None
        neg_b0_prev = neg_b1_prev = None
        pos_state = None
        neg_state = None
        converged_count = 0
        history = []

        for iteration in range(1, self.max_iter + 1):
            pos_b0, pos_b1, pos_state, pos_info = unbalanced_positive_residual_transport_step(
                u=u,
                background=self.background_image,
                lam0=pos_lam0,
                lam1=pos_lam1,
                beta=self.beta,
                eta=self.eta,
                source_weight=self.source_weight,
                T=self.transport_T,
                inner_iters=self.transport_inner_iters,
                tol=self.transport_tol,
                state=pos_state,
                return_state=True,
            )
            neg_b0, neg_b1, neg_state, neg_info = unbalanced_positive_residual_transport_step(
                u=2.0 * self.background_image[None, :, :] - u,
                background=self.background_image,
                lam0=neg_lam0,
                lam1=neg_lam1,
                beta=self.beta,
                eta=self.eta,
                source_weight=self.source_weight,
                T=self.transport_T,
                inner_iters=self.transport_inner_iters,
                tol=self.transport_tol,
                state=neg_state,
                return_state=True,
            )

            u = signed_residual_image_step(
                u=u,
                data_terms=self.data_terms,
                background=self.background_image,
                pos_b0=pos_b0,
                pos_b1=pos_b1,
                neg_b0=neg_b0,
                neg_b1=neg_b1,
                pos_lam0=pos_lam0,
                pos_lam1=pos_lam1,
                neg_lam0=neg_lam0,
                neg_lam1=neg_lam1,
                regularizer=self.regularizer,
                eta=self.eta,
                prior_image=self.prior_image,
                prior_weight=self.prior_weight,
            )

            pos_lam0, pos_lam1, neg_lam0, neg_lam1 = signed_residual_dual_step(
                u,
                self.background_image,
                pos_b0,
                pos_b1,
                neg_b0,
                neg_b1,
                pos_lam0,
                pos_lam1,
                neg_lam0,
                neg_lam1,
                self.eta,
                self.dual_relaxation,
            )

            primal = self._primal_residual(u, pos_b0, pos_b1, neg_b0, neg_b1)
            dual = self._dual_residual_signed(
                pos_b0,
                pos_b1,
                neg_b0,
                neg_b1,
                pos_b0_prev,
                pos_b1_prev,
                neg_b0_prev,
                neg_b1_prev,
            )
            eps_primal = self._eps_primal_signed(u, pos_b0, pos_b1, neg_b0, neg_b1)
            eps_dual = self._eps_dual_signed(pos_lam0, pos_lam1, neg_lam0, neg_lam1)
            data_loss = self._data_loss(u)
            all_info = list(pos_info) + list(neg_info)
            action = float(sum(item.get("transport_action", 0.0) for item in all_info))
            bb_continuity = float(max(
                (item.get("continuity_residual", 0.0) for item in all_info), default=0.0
            ))
            bb_iterations = int(max(
                (item.get("iterations", 0) for item in all_info), default=0
            ))
            source_abs = float(sum(item.get("source_mass_abs", 0.0) for item in all_info))
            source_signed = float(sum(item.get("source_mass_signed", 0.0) for item in all_info))
            pos, neg = self._signed_residuals(u)
            pos_mass = pos.sum(axis=(1, 2))
            neg_mass = neg.sum(axis=(1, 2))

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
                "source_mass_abs": source_abs,
                "source_mass_signed": source_signed,
                "u_mean": float(u.mean()),
                "u_max": float(u.max()),
                "u_min": float(u.min()),
                "positive_residual_mass_mean": float(pos_mass.mean()),
                "positive_residual_mass_min": float(pos_mass.min()),
                "positive_residual_mass_max": float(pos_mass.max()),
                "negative_residual_mass_mean": float(neg_mass.mean()),
                "negative_residual_mass_min": float(neg_mass.min()),
                "negative_residual_mass_max": float(neg_mass.max()),
            }
            history.append(row)
            print(
                f"Signed UOT ADMM {iteration:03d} | r={primal:.3e}/{eps_primal:.3e} "
                f"| s={dual:.3e}/{eps_dual:.3e} | data={data_loss:.3e} "
                f"| UOT action={action:.3e} src_abs={source_abs:.3e} "
                f"cont={bb_continuity:.3e} ({bb_iterations} iters)"
            )

            if self.eta > 0 and iteration >= self.min_iter:
                converged_count = converged_count + 1 if (
                    primal <= eps_primal and dual <= eps_dual
                    and bb_continuity <= max(self.transport_tol, 1e-8)
                ) else 0
                if converged_count >= self.patience:
                    print(f"Converged after {iteration} signed UOT ADMM iterations")
                    break

            pos_b0_prev, pos_b1_prev = pos_b0.copy(), pos_b1.copy()
            neg_b0_prev, neg_b1_prev = neg_b0.copy(), neg_b1.copy()
        return u, history
