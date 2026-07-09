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

## StarWarps / Reference Postprocessing

For postprocessing a strong dynamic reconstruction such as StarWarps, use
`--reference-postprocess`. This changes the default objective from
visibility-driven reconstruction to framewise-reference refinement:

```text
reference fidelity + signed-residual UOT
```

In this mode the defaults become:

```text
data_weight = 0
tv_weight = 0
background_weight = 0
residual_mass_weight = 0
reference_weight = 1
source_weight = 10
```

The initialization sequence is also used as the framewise reference unless
`--reference` is supplied. The background is still a modeling choice: if
`--background` is omitted, it is constructed from the initialization sequence.

Example with StarWarps initialization and StarWarps mean background:

```bash
python -m ot_uot \
  --initialization path/to/starwarps_frames \
  --reference-postprocess \
  --background-mode mean \
  --transport-method pairwise_uot \
  --output ot_uot_starwarps_reference_postprocess \
  --save-pngs
```

Example with StarWarps initialization but static-reconstruction background:

```bash
python -m ot_uot \
  --initialization path/to/starwarps_frames \
  --background static_reconstruction/reconstructed_frames_gray \
  --reference-postprocess \
  --background-mode mean \
  --transport-method pairwise_uot \
  --output ot_uot_starwarps_reference_static_background \
  --save-pngs
```

### Automatic GT metrics

When ground-truth frames can be matched to the loaded observation frame indices, the driver automatically reports both the loaded initialization quality and the final/post-UOT quality against GT. For StarWarps postprocessing, this means raw StarWarps versus StarWarps+UOT:

```text
Initialization vs GT | normalization=flux | mean_frame_nrmse=... | mean_ssim=... | stge=... | stge_temporal=...
Post-UOT vs GT      | normalization=flux | mean_frame_nrmse=... | mean_ssim=... | stge=... | stge_temporal=...
Metric deltas       | delta_mean_frame_nrmse=... | delta_mean_ssim=... | delta_stge=... | delta_stge_temporal=...
```

The same metrics are saved to:

```text
<output>/metrics_summary.json
<output>/frame_metrics.csv
```

`metrics_summary.json` contains `initialization_metrics`, `post_uot_metrics`, and `delta_metrics`. The CSV contains per-frame initialization NRMSE/SSIM, post-UOT NRMSE/SSIM, per-frame spatial-gradient error components, and per-frame deltas.

The automatic temporal metric is STGE, the normalized spatiotemporal gradient error

```text
||[dx u - dx gt, dy u - dy gt, lambda(dt u - dt gt)]|| / ||[dx gt, dy gt, lambda dt gt]||.
```

Use `--stge-lambda auto` to compute lambda from GT so spatial and temporal gradient energies are balanced. You can also pass a positive numeric value, e.g. `--stge-lambda 0.5`, for fixed-lambda ablations.

Metrics are also embedded in `reconstruction.npz` under explicit keys such as `metric_init_mean_frame_nrmse`, `metric_init_mean_ssim`, `metric_init_stge`, `metric_final_mean_frame_nrmse`, `metric_final_mean_ssim`, `metric_final_stge`, `metric_delta_mean_frame_nrmse`, `metric_delta_mean_ssim`, and `metric_delta_stge`. Fourier chi-squared keys are also saved, including `metric_init_fourier_reduced_chi2`, `metric_final_fourier_reduced_chi2`, and `metric_delta_fourier_reduced_chi2`. Backward-compatible aliases `metric_mean_frame_nrmse`, `metric_mean_ssim`, `metric_stge`, `metric_frame_nrmse`, and `metric_frame_ssim` still refer to the final/post-UOT metrics. The default metric normalization is per-frame flux normalization, matching the usual legacy image-domain evaluation convention for comparing flux-scaled reconstructions to PNG ground truth. Override it with `--metric-normalization {flux,minmax,zscore,none}`.

## Transport-weight sweep

The driver can run a repeated sweep over the signed-residual UOT transport
weight while reusing the same loaded observations, initialization/background,
reference, and ground-truth matching setup.  Supply
`--sweep-transport-weights` with explicit values, or supply the flag alone to
use the default six-value sweep from `1e-8` through `1e-3`:

```text
1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3
```

Each run reports image-domain NRMSE/SSIM/STGE when ground truth is available, plus Fourier-plane chi-squared metrics against the observed complex visibilities. The reduced Fourier chi-squared is computed in the same normalized weighted visibility coordinates used by the optimizer, so it is most useful for comparing runs within the same sweep.

Example first-five-frame StarWarps-reference postprocessing sweep:

```bash
python -m ot_uot \
  --reference-postprocess \
  --initialization path/to/starwarps_frames \
  --output ot_uot_starwarps_first5_beta_sweep \
  --max-frames 5 \
  --sweep-transport-weights \
  --transport-method pairwise_uot \
  --background-mode mean \
  --static-scale-mode per_frame \
  --transport-nodes 7 \
  --transport-inner-iters 150 \
  --image-inner-iters 50 \
  --max-admm-iters 100 \
  --min-admm-iters 20 \
  --patience 8 \
  --reference-weight 0.3 \
  --source-weight 10.0 \
  --metric-normalization minmax \
  --stge-lambda auto \
  --save-pngs
```

Example first-five-frame static-initialization sweep with data + UOT only:

```bash
python -m ot_uot \
  --output ot_uot_static_first5_beta_sweep \
  --max-frames 5 \
  --sweep-transport-weights \
  --transport-method pairwise_uot \
  --background-mode mean \
  --static-scale-mode per_frame \
  --tv-weight 0.0 \
  --background-weight 0.0 \
  --residual-mass-weight 0.0 \
  --transport-nodes 7 \
  --transport-inner-iters 100 \
  --image-inner-iters 8 \
  --max-admm-iters 60 \
  --min-admm-iters 12 \
  --patience 5 \
  --source-weight 10.0 \
  --metric-normalization minmax \
  --save-pngs
```

Each sweep run is saved under
`<output>/transport_weight_<value>/`, with its own `reconstruction.npz`,
`experiment_config.json`, optional PNG frames, and metric files. Aggregate
comparison files are written to:

```text
<output>/sweep_summary.json
<output>/sweep_summary.csv
```
