# Standalone Signed-Residual OT/UOT Reconstruction

`ot_uot` is a self-contained research implementation for dynamic EHT image
reconstruction with exact signed-residual unbalanced optimal transport splitting.

The static frame reconstructions may be produced externally by an existing
`eht-imaging` pipeline. After loading observations and static frames, the
dynamic OT/UOT reconstruction, diagnostics, visualization, and evaluation live
inside this package.

## Transport Methods

- `pairwise_uot`: independent adjacent-frame UOT paths.
- `global_velocity`: one Eulerian UOT path over the whole video for each signed
  residual channel.

Both methods share the same image/residual update and ADMM convergence logic.

## Package Layout

`core/`
: Configuration, finite differences, visibility operators, projections, and
  variable containers.

`transport/`
: Continuity operators, kinetic proximal maps, pairwise UOT, and global
  velocity-field UOT solvers.

`regularizers/`
: TV utilities and the exact coupled image/residual ADMM subproblem solver.

`optimization/`
: ADMM state, objective evaluation, convergence monitoring, and the main solver.

`io/`
: Observation loading, static sequence/background loading, JSON config
  serialization, and result saving.

`evaluation/`
: Image and sequence metrics for reconstruction comparisons.

`visualization/`
: PNG/video output helpers.

`drivers/`
: Command-line entry point for running the standalone solver.

`tests/`
: Unit, numerical, and integration checks.

`docs/`
: Formal mathematical specification.

## Quick Smoke Test

```bash
python -m unittest discover -s ot_uot/tests
```

## Command-Line Driver

```bash
python -m ot_uot \
  --transport-method pairwise_uot
```

Use `--transport-method global_velocity` to run the global Eulerian method.

By default, the runner assumes the repository-root layout:

```text
blackhole_sim_testing/observations_npz
static_reconstruction/reconstructed_frames_gray
blackhole_sim/data/aart_frames
```

Override these with `--observations`, `--static`, `--ground-truth`, and
`--output` when needed.

The driver writes `reconstruction.npz`, `experiment_config.json`, and optional
PNG frames. The result file includes both standalone keys (`image`, `positive`,
`negative`) and legacy-compatible keys (`joint`, `static`, `background_video`)
so existing evaluation scripts can consume it.

## Mathematical Reference

The implementation is governed by `docs/mathematical_model.md`. The central
state variables are the nonnegative image sequence `u`, signed residual channels
`p,n`, transport density/momentum/source variables, and scaled ADMM duals for
the linear decomposition and endpoint/frame constraints.

The package intentionally uses a quadratic source/sink UOT action. This is a
convex relaxed-continuity model, not a Fisher-Rao/WFR implementation.

## Raw Synthetic Observation Workflow

1. Generate eht-imaging NPZ observations using the legacy
   `generate_ehtim_observations.py` workflow. The standalone loader accepts
   top-level `u`, `v`, `vis`, `sigma` arrays and the older structured `data`
   array format.
2. Produce static frame reconstructions using the existing static
   reconstruction pipeline.
3. Run this package with the NPZ observation directory and static-frame
   directory.
4. Evaluate `reconstruction.npz`. It includes `joint`, `static`, and
   `background_video` keys for compatibility with legacy result evaluators.
