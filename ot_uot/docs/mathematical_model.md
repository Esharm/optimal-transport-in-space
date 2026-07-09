# Mathematical Specification: Signed-Residual UOT Dynamic Reconstruction

This document is the authoritative specification for the standalone `ot_uot`
implementation.

## 1. Forward Model

For each observed frame \(k=0,\ldots,K-1\), the unknown sky brightness image is

\[
u_k \in \mathbb{R}_+^{H\times W}.
\]

The complex visibility data term is

\[
D_k(u_k)=\frac12\|A_k u_k-y_k\|_2^2,
\]

where \(A_k\) is a weighted, scaled nonuniform Fourier operator and \(y_k\) is
the weighted, scaled complex visibility vector.

This likelihood is appropriate for controlled synthetic observations with
calibrated complex visibilities. Real EHT data may require closure quantities or
calibration-marginalized likelihoods; those are intentionally outside this
first standalone UOT implementation.

## 2. Signed Residual Decomposition

Let \(a_k\in\mathbb{R}_+^{H\times W}\) be a fixed background image. In most
experiments \(a_k=a\), the mean of calibrated static reconstructions, but the
code treats \(a_k\) as explicit data.

Instead of imposing nonsmooth identities

\[
(u_k-a_k)_+,\qquad (a_k-u_k)_+,
\]

the clean variational formulation introduces explicit nonnegative residual
channels

\[
p_k\ge 0,\qquad n_k\ge 0,
\]

with the linear decomposition constraint

\[
u_k-a_k-p_k+n_k=0.
\]

The positive channel \(p_k\) represents brightness above the background; the
negative channel \(n_k\) represents brightness removed from the background.

A residual mass penalty

\[
\lambda_r\sum_k \langle 1,p_k+n_k\rangle
\]

is included to remove the degeneracy \(p_k,n_k\mapsto p_k+r,n_k+r\).

## 3. Spatial Objective

The per-frame image objective is

\[
\sum_k D_k(u_k)
+\alpha\sum_k \mathrm{TV}(u_k)
+\frac{\mu}{2}\sum_k\|u_k-a_k\|_2^2.
\]

The implementation uses isotropic finite-difference TV:

\[
\mathrm{TV}(u)=\sum_x \sqrt{(\nabla_x u)^2+(\nabla_y u)^2}.
\]

## 4. Unbalanced Transport Action

For a nonnegative density path \(\rho_t\), momentum \(m_t\), and source/sink
field \(s_t\), the discrete UOT action is

\[
\mathcal{A}_\gamma(\rho,m,s)
=
\Delta t\sum_t\sum_x
\left[
\frac{|m_t(x)|^2}{2\rho_t(x)}
+\frac{\gamma}{2}|s_t(x)|^2
\right],
\]

with the convention that \(|m|^2/(2\rho)=+\infty\) when \(\rho=0\) and \(m\ne0\).

The unbalanced continuity equation is

\[
\frac{\rho_{t+1}-\rho_t}{\Delta t}+\nabla\cdot m_t-s_t=0.
\]

The implementation uses the quadratic source penalty above. This is a valid
relaxed-continuity UOT model, but it is not the Fisher-Rao/WFR source model.

## 5. Transport Formulations

The package supports two transport methods through one ADMM interface.

### 5.1 Pairwise Adjacent-Frame UOT

For every adjacent pair \(k,k+1\) and channel \(c\in\{+,-\}\), introduce an
independent transport path

\[
\rho^c_{k,t},m^c_{k,t},s^c_{k,t},\qquad t=0,\ldots,T-1.
\]

Endpoint constraints are

\[
\rho^+_{k,0}=p_k,\quad \rho^+_{k,T-1}=p_{k+1},
\]

\[
\rho^-_{k,0}=n_k,\quad \rho^-_{k,T-1}=n_{k+1}.
\]

The transport regularizer is

\[
\beta\sum_{k=0}^{K-2}
\left[
\mathcal{A}_\gamma(\rho^+_k,m^+_k,s^+_k)
+\mathcal{A}_\gamma(\rho^-_k,m^-_k,s^-_k)
\right].
\]

This method is local in time and computationally parallel across adjacent
pairs. It does not enforce a single globally coherent flow.

### 5.2 Global Eulerian Velocity-Field UOT

For each channel \(c\in\{+,-\}\), introduce one global density path over the
observed video frames:

\[
\rho^c_k,m^c_k,s^c_k,\qquad k=0,\ldots,K-1,
\]

with endpoint/frame constraints

\[
\rho^+_k=p_k,\qquad \rho^-_k=n_k
\quad\text{for all }k.
\]

The global continuity equation is

\[
\frac{\rho^c_{k+1}-\rho^c_k}{\Delta t}+\nabla\cdot m^c_k-s^c_k=0.
\]

The transport regularizer is

\[
\beta\left[
\mathcal{A}_\gamma(\rho^+,m^+,s^+)
+\mathcal{A}_\gamma(\rho^-,m^-,s^-)
\right].
\]

This method couples all frames through one Eulerian path per channel. It is
more globally coherent but more expensive and less parallel than pairwise UOT.

## 6. Full Variational Problem

The shared variational problem is

\[
\begin{aligned}
\min_{u,p,n,\rho,m,s}\quad
&\sum_k D_k(u_k)
+\alpha\sum_k \mathrm{TV}(u_k)
+\frac{\mu}{2}\sum_k\|u_k-a_k\|_2^2\\
&+\lambda_r\sum_k\langle 1,p_k+n_k\rangle
+\beta\,\mathcal{T}(p,n;\rho,m,s)
\\
\text{s.t.}\quad
&u_k-a_k-p_k+n_k=0,\\
&u_k,p_k,n_k\ge0,\\
&\text{UOT continuity constraints},\\
&\text{transport endpoint/frame constraints}.
\end{aligned}
\]

Here \(\mathcal{T}\) is either the pairwise adjacent-frame transport sum or the
global Eulerian transport sum.

## 7. Scaled ADMM

The implementation uses scaled dual variables.

### 7.1 Decomposition Constraint

For

\[
c_k(u,p,n)=u_k-a_k-p_k+n_k=0,
\]

the scaled dual is \(d_k\), and the augmented term is

\[
\frac{\eta_d}{2}\sum_k\|u_k-a_k-p_k+n_k+d_k\|_2^2.
\]

Dual update:

\[
d_k\leftarrow d_k+u_k-a_k-p_k+n_k.
\]

### 7.2 Endpoint Constraints

For a generic channel density endpoint/frame variable \(r_j\) and residual
channel variable \(h_j\in\{p_j,n_j\}\), the constraint is

\[
r_j-h_j=0.
\]

The scaled dual is \(e_j\), and the augmented term is

\[
\frac{\eta_e}{2}\|r_j-h_j+e_j\|_2^2.
\]

Transport update target:

\[
h_j-e_j.
\]

Image/residual update target:

\[
r_j+e_j.
\]

Dual update:

\[
e_j\leftarrow e_j+r_j-h_j.
\]

No endpoint target is clipped. Nonnegativity is enforced by the transport
density constraint and by \(p,n\ge0\).

## 8. ADMM Iteration Pseudocode

Given initial \(u,p,n\), background \(a\), and zero scaled duals:

```text
repeat
    transport update:
        if pairwise:
            for each adjacent pair k and channel c:
                solve UOT path with endpoint quadratic targets h_left - e_left,
                h_right - e_right
        if global_velocity:
            for each channel c:
                solve one global UOT path with frame targets h - e

    image/residual update:
        solve the convex subproblem in (u,p,n):
            data(u) + TV(u) + background quadratic
            + residual mass penalty
            + decomposition augmented term
            + endpoint/frame augmented terms
            + nonnegativity constraints

    dual update:
        d <- d + u - a - p + n
        e <- e + transport_endpoint_or_frame_density - residual_channel

    evaluate objective and primal/dual residuals
until residual tolerances are satisfied for patience iterations
```

## 9. Software Mapping

| Mathematical Object | Module | Class / Function |
|---|---|---|
| Image grid metadata | `core.config` | `ImageGrid` |
| Regularization/ADMM parameters | `core.config` | `UOTParameters` |
| \(A_k\), \(A_k^*\) | `core.visibility` | `DirectVisibilityOperator` |
| \(D_k\), \(\nabla D_k\) | `core.visibility` | `ComplexVisibilityDataTerm` |
| \(\nabla\), \(\nabla\cdot\) | `core.finite_differences` | `gradient`, `divergence` |
| \(u,p,n,a\) state | `core.variables` | `ImageResidualState` |
| Transport path \(\rho,m,s\) | `core.variables` | `TransportState` |
| UOT continuity operator | `transport.continuity` | `uot_continuity`, `uot_continuity_adjoint` |
| Kinetic prox | `transport.kinetic_prox` | `prox_kinetic_perspective` |
| Pairwise transport update | `transport.pairwise` | `PairwiseUOTTransport` |
| Global transport update | `transport.global_velocity` | `GlobalVelocityUOTTransport` |
| TV value/projection | `regularizers.tv` | `tv_value`, `project_tv_dual` |
| Exact image/residual update | `regularizers.image_residual_update` | `ImageResidualUpdater` |
| Objective diagnostics | `optimization.objective` | `objective_breakdown` |
| ADMM state | `optimization.admm_state` | `ADMMState` |
| Convergence checks | `optimization.convergence` | `convergence_status` |
| Main solver | `optimization.signed_residual_admm` | `SignedResidualUOTADMM` |
| Observation loading | `io.observations` | `load_observation_directory`, `load_observation_npz` |
| Static initialization/background | `io.static_init` | `load_static_sequence`, `load_static_with_background` |
| Configuration persistence | `io.config_io` | `save_experiment_config`, `load_experiment_config` |
| Result persistence | `io.results` | `save_reconstruction_npz` |

## 10. Pairwise and Global Formulation Evaluation

The pairwise formulation is computationally attractive because every adjacent
path can be solved independently for each channel. It is also closest to the
legacy implementation and to a temporal regularizer that penalizes local
frame-to-frame changes. Its main limitation is scientific: a sequence of
locally optimal adjacent transports need not correspond to a single coherent
velocity field over the full video. Under sparse uv coverage this can allow
locally plausible but globally inconsistent motion.

The global Eulerian formulation represents the video by one density path,
momentum field, and source/sink field per signed residual channel. It is more
faithful to a continuous dynamic-OT view because all frames share one continuity
equation. It should be more sensitive to persistent coherent motion and less
able to explain each frame pair independently. Its cost is higher memory use,
weaker temporal parallelism, and greater sensitivity to a poor global transport
scale or an overly permissive source penalty.

Both formulations are scientifically reasonable for controlled synthetic EHT
experiments. Neither should be assumed superior without ablations over
uv-coverage, source penalty, endpoint penalty, static initialization quality,
and temporal baselines such as framewise static, temporal L2/H1, and TV-only
refinement.

## 11. Convergence Guarantees and Limitations

For fixed background and convex data terms, the pairwise and global transport
subproblems are convex, and the image/residual subproblem is convex. The overall
constrained formulation is convex in the explicit variables under the quadratic
source UOT action and linear residual decomposition.

The implementation solves subproblems numerically to finite tolerance using
first-order primal-dual/proximal iterations. Reported convergence is therefore
numerical ADMM convergence, not an exact symbolic solution.

The model assumes calibrated complex visibility data. Closure quantities and
calibration nuisance parameters are future likelihood extensions, not part of
this specification.

## 12. Reference-Fidelity Postprocessing Mode

For StarWarps or other high-quality dynamic initializations, the package also
supports a reference-fidelity refinement mode. Let

\[
r_k \in \mathbb{R}_+^{H\times W}
\]

be a framewise reference sequence, typically the StarWarps reconstruction. The
postprocessing objective is

\[
\min_{u,p,n,\rho,m,s}
\frac{\lambda_{ref}}{2}\sum_k\|u_k-r_k\|_2^2
+\beta\mathcal{T}(p,n;\rho,m,s)
\]

subject to the same signed-residual decomposition,

\[
u_k-a_k-p_k+n_k=0,
\qquad u_k,p_k,n_k\ge 0,
\]

and the same UOT continuity and endpoint/frame constraints. In this mode the
visibility data term, TV term, background quadratic prior, and residual-mass
penalty are disabled by default:

\[
\lambda_D=0,
\qquad \alpha=0,
\qquad \mu=0,
\qquad \lambda_r=0.
\]

This is not pure UOT. The framewise reference term keeps the output anchored to
StarWarps while UOT acts as a temporal signed-residual coherence regularizer.
The command-line toggle is `--reference-postprocess`. If no explicit
`--reference` directory is supplied, the initialization sequence is used as the
reference sequence.
