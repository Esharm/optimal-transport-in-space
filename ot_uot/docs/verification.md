# Verification Plan

The test suite is designed to check the mathematical operators and optimization
interfaces used by the standalone signed-residual UOT implementation.

## Unit and Operator Tests

- Finite-difference gradient/divergence adjoint identity.
- Direct complex-visibility forward/adjoint identity.
- UOT continuity operator/adjoint identity.
- Kinetic perspective proximal map on a zero-momentum special case.
- Image/residual state shape and decomposition invariants.
- Transport state shape and nonnegativity invariants.

## Transport Tests

- Constant-density UOT path has near-zero action and continuity residual.
- Pairwise UOT returns one path per adjacent frame pair and produces correctly
  shaped quadratic targets for the image/residual update.
- Global UOT returns one full-video path per channel and produces correctly
  shaped framewise quadratic targets.

## Integration Tests

- Tiny full ADMM runs for `pairwise_uot` and `global_velocity`.
- Tests assert finite objective/residual diagnostics, nonnegative primal
  variables, populated history, and correct iteration accounting.

## Scientific Validation Experiments

For paper-scale experiments, the implementation should be validated with:

- static initialization only,
- TV-only or temporal L2/H1 refinement,
- pairwise UOT,
- global velocity-field UOT,
- StarWarps baseline,
- uv-sparsity sweeps over telescope count,
- sensitivity to `source_weight`, `transport_weight`, `endpoint_penalty`,
  `decomposition_penalty`, and background construction,
- moving Gaussian or ring-like synthetic sequences with known motion,
- mass-creation/mass-destruction tests to separate transport behavior from
  source/sink behavior.

