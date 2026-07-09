"""Command-line driver for standalone signed-residual OT/UOT reconstruction."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from ot_uot.core.background import BackgroundMode, make_background
from ot_uot.core.config import ImageGrid, ReconstructionPaths, TransportMethod, UOTParameters
from ot_uot.evaluation.metrics import compare_initialization_and_final_fourier_reports, compare_initialization_and_final_reports
from ot_uot.io.ground_truth import load_ground_truth_sequence
from ot_uot.io.observations import load_observation_directory
from ot_uot.io.config_io import save_experiment_config
from ot_uot.io.results import save_reconstruction_npz
from ot_uot.io.static_init import calibrate_static_sequence, load_static_sequence
from ot_uot.optimization.signed_residual_admm import SignedResidualUOTADMM
from ot_uot.visualization.outputs import save_frame_pngs


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OBSERVATIONS = PROJECT_ROOT / "blackhole_sim_testing" / "observations_fixed_npz"
DEFAULT_INITIALIZATION = PROJECT_ROOT / "static_reconstruction" / "static_128_real"
DEFAULT_GROUND_TRUTH = PROJECT_ROOT / "blackhole_sim" / "data" / "aart_frames"
DEFAULT_OUTPUT = PROJECT_ROOT / "ot_uot_results"
SCALE_CHOICES = ["none", "per_frame", "global"]
METRIC_NORMALIZATION_CHOICES = ["flux", "minmax", "zscore", "none"]
DEFAULT_TRANSPORT_WEIGHT_SWEEP = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]


def _resolve_weight(value: float | None, *, normal: float, postprocess: float, enabled: bool) -> float:
    if value is not None:
        return float(value)
    return float(postprocess if enabled else normal)


def _parse_transport_weight_sweep(raw_values: list[str] | None) -> list[float] | None:
    """Parse optional transport-weight sweep values from CLI tokens.

    ``None`` means ordinary single-run mode.  An empty list means the user
    supplied ``--sweep-transport-weights`` without explicit values, in which
    case we use a six-point decade sweep from 1e-8 through 1e-3.
    Tokens may be passed either as separate values or as comma-separated lists.
    """

    if raw_values is None:
        return None
    if len(raw_values) == 0:
        return list(DEFAULT_TRANSPORT_WEIGHT_SWEEP)
    tokens: list[str] = []
    for raw in raw_values:
        tokens.extend(part for part in str(raw).replace(",", " ").split() if part)
    if not tokens:
        return list(DEFAULT_TRANSPORT_WEIGHT_SWEEP)
    weights = []
    for token in tokens:
        try:
            weight = float(token)
        except ValueError as exc:
            raise ValueError(f"invalid transport weight in sweep: {token!r}") from exc
        if weight <= 0.0:
            raise ValueError("all sweep transport weights must be positive")
        weights.append(weight)
    return weights


def _transport_weight_label(weight: float) -> str:
    """Return a readable, filesystem-safe label for a transport weight."""

    label = f"{float(weight):.0e}".replace("+", "")
    return f"transport_weight_{label}"


def _get_nested_scalar(data: dict | None, *keys: str) -> float | None:
    """Safely fetch a nested numeric metric for sweep summaries."""

    current: object = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    if current is None:
        return None
    try:
        return float(current)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _write_sweep_outputs(output_dir: Path, rows: list[dict[str, object]]) -> tuple[Path, Path]:
    """Persist aggregate transport-weight sweep results."""

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "sweep_summary.json"
    csv_path = output_dir / "sweep_summary.csv"
    json_payload = {
        "sweep_parameter": "transport_weight",
        "runs": rows,
    }
    json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")

    fieldnames = [
        "sweep_index",
        "transport_weight",
        "output_dir",
        "iterations",
        "objective",
        "data",
        "reference",
        "tv",
        "transport",
        "initialization_mean_frame_nrmse",
        "post_uot_mean_frame_nrmse",
        "delta_mean_frame_nrmse",
        "initialization_mean_ssim",
        "post_uot_mean_ssim",
        "delta_mean_ssim",
        "initialization_stge",
        "post_uot_stge",
        "delta_stge",
        "initialization_stge_temporal",
        "post_uot_stge_temporal",
        "delta_stge_temporal",
        "initialization_fourier_reduced_chi2",
        "post_uot_fourier_reduced_chi2",
        "delta_fourier_reduced_chi2",
        "initialization_fourier_chi2",
        "post_uot_fourier_chi2",
        "delta_fourier_chi2",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})
    return json_path, csv_path


def _load_and_calibrate_sequence(
    directory: Path,
    *,
    frame_indices: list[int],
    max_frames: int | None,
    normalize_images: bool,
    data_terms,
    scale_mode: str,
) -> tuple[np.ndarray, list[Path], dict]:
    sequence, paths = load_static_sequence(
        directory,
        frame_indices=frame_indices,
        max_frames=max_frames,
        normalize_images=normalize_images,
    )
    sequence, scale_info = calibrate_static_sequence(sequence, data_terms, mode=scale_mode)
    return sequence, paths, scale_info


def _write_metric_outputs(output_dir: Path, metrics: dict[str, object], gt_names: list[str] | None = None) -> tuple[Path, Path]:
    """Persist initialization/final GT metrics as JSON and per-frame CSV."""

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "metrics_summary.json"
    csv_path = output_dir / "frame_metrics.csv"
    json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    init = metrics.get("initialization_metrics") or {}
    final = metrics.get("post_uot_metrics") or {}
    delta = metrics.get("delta_metrics") or {}
    init_nrmse = list(init.get("frame_nrmse", []))
    init_ssim = list(init.get("frame_ssim", []))
    final_nrmse = list(final.get("frame_nrmse", []))
    final_ssim = list(final.get("frame_ssim", []))
    delta_nrmse = list(delta.get("delta_frame_nrmse", []))
    delta_ssim = list(delta.get("delta_frame_ssim", []))

    fourier = metrics.get("fourier_metrics") or {}
    init_fourier = fourier.get("initialization_fourier_metrics") if isinstance(fourier, dict) else {}
    final_fourier = fourier.get("post_uot_fourier_metrics") if isinstance(fourier, dict) else {}
    delta_fourier = fourier.get("delta_fourier_metrics") if isinstance(fourier, dict) else {}
    init_frame_chi2 = list((init_fourier or {}).get("frame_fourier_reduced_chi2", []))
    final_frame_chi2 = list((final_fourier or {}).get("frame_fourier_reduced_chi2", []))
    delta_frame_chi2 = list((delta_fourier or {}).get("delta_frame_fourier_reduced_chi2", []))

    frame_count = min(len(init_nrmse), len(init_ssim), len(final_nrmse), len(final_ssim))
    if frame_count == 0 and init_frame_chi2 and final_frame_chi2:
        frame_count = min(len(init_frame_chi2), len(final_frame_chi2))
    names = gt_names if gt_names is not None and len(gt_names) == frame_count else [str(k) for k in range(frame_count)]

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "frame",
                "gt_name",
                "initialization_nrmse",
                "initialization_ssim",
                "post_uot_nrmse",
                "post_uot_ssim",
                "delta_nrmse",
                "delta_ssim",
                "initialization_stge_spatial",
                "post_uot_stge_spatial",
                "delta_stge_spatial_frame",
                "initialization_fourier_reduced_chi2",
                "post_uot_fourier_reduced_chi2",
                "delta_fourier_reduced_chi2",
            ],
        )
        writer.writeheader()
        def at(values, index, default=None):
            return values[index] if index < len(values) else default

        init_spatial = list(init.get("frame_spatial_gradient_nrmse", []))
        final_spatial = list(final.get("frame_spatial_gradient_nrmse", []))
        delta_spatial = list(delta.get("delta_frame_spatial_gradient_nrmse", []))
        for index in range(frame_count):
            writer.writerow({
                "frame": index,
                "gt_name": names[index],
                "initialization_nrmse": at(init_nrmse, index),
                "initialization_ssim": at(init_ssim, index),
                "post_uot_nrmse": at(final_nrmse, index),
                "post_uot_ssim": at(final_ssim, index),
                "delta_nrmse": at(delta_nrmse, index, (at(final_nrmse, index, 0.0) - at(init_nrmse, index, 0.0)) if index < len(final_nrmse) and index < len(init_nrmse) else None),
                "delta_ssim": at(delta_ssim, index, (at(final_ssim, index, 0.0) - at(init_ssim, index, 0.0)) if index < len(final_ssim) and index < len(init_ssim) else None),
                "initialization_stge_spatial": at(init_spatial, index),
                "post_uot_stge_spatial": at(final_spatial, index),
                "delta_stge_spatial_frame": at(delta_spatial, index),
                "initialization_fourier_reduced_chi2": at(init_frame_chi2, index),
                "post_uot_fourier_reduced_chi2": at(final_frame_chi2, index),
                "delta_fourier_reduced_chi2": at(delta_frame_chi2, index, (
                    final_frame_chi2[index] - init_frame_chi2[index]
                    if index < len(final_frame_chi2) and index < len(init_frame_chi2) else None
                )),
            })
    return json_path, csv_path

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, default=DEFAULT_OBSERVATIONS)
    parser.add_argument("--static", type=Path, default=None, help="Backward-compatible alias for --initialization.")
    parser.add_argument("--initialization", type=Path, default=None, help="Frame sequence used to initialize u_k. Defaults to the static reconstruction directory.")
    parser.add_argument("--background", type=Path, default=None, help="Frame sequence used to construct the fixed residual background. Defaults to the initialization directory.")
    parser.add_argument("--reference", type=Path, default=None, help="Framewise reference sequence for reference-fidelity postprocessing. Defaults to initialization when needed.")
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GROUND_TRUTH)
    parser.add_argument("--metric-normalization", choices=METRIC_NORMALIZATION_CHOICES, default="flux", help="Normalization used for automatic GT NRMSE/SSIM reporting.")
    parser.add_argument("--metric-total-flux", type=float, default=1.0, help="Per-frame total flux used when --metric-normalization=flux.")
    parser.add_argument("--stge-lambda", default="auto", help="Temporal-gradient weight for STGE. Use 'auto' to match GT spatial/temporal gradient energy, or pass a positive float.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--transport-method", choices=[m.value for m in TransportMethod], default=TransportMethod.PAIRWISE_UOT.value)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--fov-rad", type=float, default=160e-6 / 206265.0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--max-visibilities-per-frame", type=int, default=None)
    parser.add_argument("--background-mode", choices=[m.value for m in BackgroundMode], default=BackgroundMode.MEAN.value)
    parser.add_argument("--static-scale-mode", choices=SCALE_CHOICES, default=None, help="Backward-compatible scale mode applied to initialization/background unless overridden.")
    parser.add_argument("--initialization-scale-mode", choices=SCALE_CHOICES, default=None)
    parser.add_argument("--background-scale-mode", choices=SCALE_CHOICES, default=None)
    parser.add_argument("--reference-scale-mode", choices=SCALE_CHOICES, default=None)
    parser.add_argument("--no-normalize-static-images", action="store_true", help="Backward-compatible alias for --no-normalize-initialization-images.")
    parser.add_argument("--no-normalize-initialization-images", action="store_true")
    parser.add_argument("--no-normalize-background-images", action="store_true")
    parser.add_argument("--no-normalize-reference-images", action="store_true")

    parser.add_argument("--reference-postprocess", action="store_true", help="Use framewise reference fidelity plus signed-residual UOT, with data/TV/background-prior/residual-mass defaults disabled.")
    parser.add_argument("--data-weight", type=float, default=None)
    parser.add_argument("--reference-weight", type=float, default=None)
    parser.add_argument("--tv-weight", type=float, default=None)
    parser.add_argument("--background-weight", type=float, default=None)
    parser.add_argument("--residual-mass-weight", type=float, default=None)
    parser.add_argument("--transport-weight", type=float, default=None)
    parser.add_argument(
        "--sweep-transport-weights",
        nargs="*",
        default=None,
        metavar="WEIGHT",
        help=(
            "Run one reconstruction per transport/UOT weight. Pass values as "
            "space-separated or comma-separated floats. If the flag is supplied "
            "with no values, uses 1e-8 1e-7 1e-6 1e-5 1e-4 1e-3."
        ),
    )
    parser.add_argument("--source-weight", type=float, default=None)
    parser.add_argument("--decomposition-penalty", type=float, default=1e-3)
    parser.add_argument("--endpoint-penalty", type=float, default=1e-3)
    parser.add_argument("--transport-nodes", type=int, default=7)
    parser.add_argument("--transport-inner-iters", type=int, default=100)
    parser.add_argument("--image-inner-iters", type=int, default=50)
    parser.add_argument("--max-admm-iters", type=int, default=60)
    parser.add_argument("--min-admm-iters", type=int, default=12)
    parser.add_argument("--abs-tol", type=float, default=1e-4)
    parser.add_argument("--rel-tol", type=float, default=5e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument("--save-pngs", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    grid = ImageGrid(height=args.height, width=args.width, fov_rad=args.fov_rad)

    observations = load_observation_directory(
        args.observations,
        grid,
        max_frames=args.max_frames,
        max_visibilities_per_frame=args.max_visibilities_per_frame,
        use_cache=args.use_cache,
    )
    data_terms = [obs.data_term for obs in observations]
    frame_indices = [obs.frame_index for obs in observations]

    initialization_dir = args.initialization or args.static or DEFAULT_INITIALIZATION
    background_dir = args.background or initialization_dir
    reference_dir = args.reference

    fallback_scale_mode = args.static_scale_mode or "per_frame"
    initialization_scale_mode = args.initialization_scale_mode or fallback_scale_mode
    background_scale_mode = args.background_scale_mode or fallback_scale_mode
    reference_scale_mode = args.reference_scale_mode or initialization_scale_mode

    normalize_initialization = not (args.no_normalize_static_images or args.no_normalize_initialization_images)
    initialization_sequence, initialization_paths, initialization_scale_info = _load_and_calibrate_sequence(
        initialization_dir,
        frame_indices=frame_indices,
        max_frames=args.max_frames,
        normalize_images=normalize_initialization,
        data_terms=data_terms,
        scale_mode=initialization_scale_mode,
    )
    if initialization_sequence.shape[0] != len(observations):
        raise ValueError("initialization frames and observation frames must have the same count")

    if Path(background_dir).resolve() == Path(initialization_dir).resolve() and background_scale_mode == initialization_scale_mode:
        background_sequence = initialization_sequence
        background_paths = initialization_paths
        background_scale_info = initialization_scale_info
    else:
        background_sequence, background_paths, background_scale_info = _load_and_calibrate_sequence(
            background_dir,
            frame_indices=frame_indices,
            max_frames=args.max_frames,
            normalize_images=not args.no_normalize_background_images,
            data_terms=data_terms,
            scale_mode=background_scale_mode,
        )
    background = make_background(background_sequence, BackgroundMode(args.background_mode))

    data_weight = _resolve_weight(args.data_weight, normal=1.0, postprocess=0.0, enabled=args.reference_postprocess)
    tv_weight = _resolve_weight(args.tv_weight, normal=1e-5, postprocess=0.0, enabled=args.reference_postprocess)
    background_weight = _resolve_weight(args.background_weight, normal=1e-4, postprocess=0.0, enabled=args.reference_postprocess)
    residual_mass_weight = _resolve_weight(args.residual_mass_weight, normal=1e-6, postprocess=0.0, enabled=args.reference_postprocess)
    reference_weight = _resolve_weight(args.reference_weight, normal=0.0, postprocess=1.0, enabled=args.reference_postprocess)
    transport_weight = _resolve_weight(args.transport_weight, normal=1e-4, postprocess=1e-4, enabled=args.reference_postprocess)
    source_weight = _resolve_weight(args.source_weight, normal=30.0, postprocess=10.0, enabled=args.reference_postprocess)

    reference_sequence = None
    reference_paths: list[Path] = []
    reference_scale_info = None
    if reference_weight > 0.0:
        if reference_dir is None or Path(reference_dir).resolve() == Path(initialization_dir).resolve():
            reference_sequence = initialization_sequence.copy()
            reference_paths = initialization_paths
            reference_scale_info = initialization_scale_info
        else:
            reference_sequence, reference_paths, reference_scale_info = _load_and_calibrate_sequence(
                reference_dir,
                frame_indices=frame_indices,
                max_frames=args.max_frames,
                normalize_images=not args.no_normalize_reference_images,
                data_terms=data_terms,
                scale_mode=reference_scale_mode,
            )

    ground_truth, ground_truth_paths = load_ground_truth_sequence(
        args.ground_truth,
        frame_indices,
        shape=grid.shape,
    )

    sweep_weights = _parse_transport_weight_sweep(args.sweep_transport_weights)
    sweep_mode = sweep_weights is not None
    weights_to_run = sweep_weights if sweep_weights is not None else [transport_weight]

    def run_single_weight(
        *,
        current_transport_weight: float,
        run_output_dir: Path,
        sweep_index: int | None = None,
        sweep_total: int | None = None,
    ) -> dict[str, object]:
        params = UOTParameters(
            transport_method=TransportMethod(args.transport_method),
            data_weight=data_weight,
            tv_weight=tv_weight,
            background_weight=background_weight,
            reference_weight=reference_weight,
            residual_mass_weight=residual_mass_weight,
            transport_weight=current_transport_weight,
            source_weight=source_weight,
            decomposition_penalty=args.decomposition_penalty,
            endpoint_penalty=args.endpoint_penalty,
            transport_nodes=args.transport_nodes,
            transport_inner_iters=args.transport_inner_iters,
            image_inner_iters=args.image_inner_iters,
            max_admm_iters=args.max_admm_iters,
            min_admm_iters=args.min_admm_iters,
            abs_tol=args.abs_tol,
            rel_tol=args.rel_tol,
            patience=args.patience,
        )
        solver = SignedResidualUOTADMM(params)

        progress_prefix = ""
        if sweep_index is not None and sweep_total is not None:
            progress_prefix = f"[sweep {sweep_index + 1}/{sweep_total} beta={current_transport_weight:g}] "

        def print_progress(state):
            if args.quiet:
                return
            latest = state.history[-1]
            print(
                progress_prefix
                + "ADMM "
                f"{latest.iteration:03d}/{params.max_admm_iters:03d} | "
                f"obj={latest.objective:.6e} | "
                f"data={latest.data:.3e} | "
                f"ref={latest.reference:.3e} | "
                f"tv={latest.tv:.3e} | "
                f"transport={latest.transport:.3e} | "
                f"r={latest.primal_residual:.3e}/{latest.eps_primal:.3e} | "
                f"s={latest.dual_residual:.3e}/{latest.eps_dual:.3e} | "
                f"decomp={latest.decomposition_residual:.3e} | "
                f"endpoint={latest.endpoint_residual:.3e} | "
                f"cont={latest.continuity_residual:.3e} | "
                f"dstate={latest.state_relative_change:.3e}",
                flush=True,
            )

        if not args.quiet:
            mode = "reference-postprocess" if args.reference_postprocess else "joint-reconstruction"
            if sweep_mode:
                print(
                    f"\n=== Transport-weight sweep run {sweep_index + 1}/{sweep_total} | "
                    f"transport_weight={current_transport_weight:g} ===",
                    flush=True,
                )
            print(
                "Starting OT/UOT reconstruction | "
                f"mode={mode} | "
                f"frames={len(observations)} | "
                f"method={params.transport_method.value} | "
                f"transport_nodes={params.transport_nodes} | "
                f"max_admm_iters={params.max_admm_iters}",
                flush=True,
            )
            print(
                "Sources | "
                f"observations={args.observations} | "
                f"initialization={initialization_dir} | "
                f"background={background_dir} | "
                f"reference={reference_dir or ('initialization' if reference_sequence is not None else 'none')}",
                flush=True,
            )
            print(
                "Weights | "
                f"data={params.data_weight:g} | ref={params.reference_weight:g} | "
                f"tv={params.tv_weight:g} | background={params.background_weight:g} | "
                f"residual_mass={params.residual_mass_weight:g} | transport={params.transport_weight:g} | "
                f"source={params.source_weight:g}",
                flush=True,
            )

        state = solver.run(
            initialization_sequence,
            background,
            data_terms,
            callback=print_progress,
            reference_sequence=reference_sequence,
        )

        run_output_dir.mkdir(parents=True, exist_ok=True)
        fourier_metrics = compare_initialization_and_final_fourier_reports(
            initialization_sequence,
            state.image_state.image,
            data_terms,
        )
        automatic_metrics: dict[str, object] | None = None
        metric_files: dict[str, str] = {}
        if ground_truth is not None:
            automatic_metrics = compare_initialization_and_final_reports(
                initialization_sequence,
                state.image_state.image,
                ground_truth,
                normalization=args.metric_normalization,
                total_flux=args.metric_total_flux,
                stge_lambda=args.stge_lambda,
            )
            automatic_metrics["fourier_metrics"] = fourier_metrics
            gt_names = [path.name for path in ground_truth_paths]
        else:
            automatic_metrics = {
                "normalization": args.metric_normalization,
                "total_flux": float(args.metric_total_flux),
                "frames": len(observations),
                "fourier_metrics": fourier_metrics,
            }
            gt_names = None

        metrics_json, metrics_csv = _write_metric_outputs(run_output_dir, automatic_metrics, gt_names=gt_names)
        metric_files = {"metrics_summary_json": str(metrics_json), "frame_metrics_csv": str(metrics_csv)}

        init_fourier = fourier_metrics["initialization_fourier_metrics"]
        final_fourier = fourier_metrics["post_uot_fourier_metrics"]
        delta_fourier = fourier_metrics["delta_fourier_metrics"]
        if ground_truth is not None:
            init_metrics = automatic_metrics["initialization_metrics"]
            final_metrics = automatic_metrics["post_uot_metrics"]
            delta_metrics = automatic_metrics["delta_metrics"]
            if not args.quiet:
                print(
                    "Initialization vs GT | "
                    f"normalization={automatic_metrics['normalization']} | "
                    f"mean_frame_nrmse={init_metrics['mean_frame_nrmse']:.6e} | "
                    f"mean_ssim={init_metrics['mean_ssim']:.6e} | "
                    f"stge={init_metrics['stge']:.6e} | "
                    f"stge_temporal={init_metrics['stge_temporal']:.6e}",
                    flush=True,
                )
                print(
                    "Post-UOT vs GT      | "
                    f"normalization={automatic_metrics['normalization']} | "
                    f"mean_frame_nrmse={final_metrics['mean_frame_nrmse']:.6e} | "
                    f"mean_ssim={final_metrics['mean_ssim']:.6e} | "
                    f"stge={final_metrics['stge']:.6e} | "
                    f"stge_temporal={final_metrics['stge_temporal']:.6e}",
                    flush=True,
                )
                print(
                    "Metric deltas       | "
                    f"delta_mean_frame_nrmse={delta_metrics['delta_mean_frame_nrmse']:+.6e} | "
                    f"delta_mean_ssim={delta_metrics['delta_mean_ssim']:+.6e} | "
                    f"delta_stge={delta_metrics['delta_stge']:+.6e} | "
                    f"delta_stge_temporal={delta_metrics['delta_stge_temporal']:+.6e}",
                    flush=True,
                )
        elif not args.quiet:
            print("Evaluation vs GT skipped | ground truth frames could not be matched", flush=True)

        if not args.quiet:
            print(
                "Fourier chi2        | "
                f"init_reduced={init_fourier['fourier_reduced_chi2']:.6e} | "
                f"post_reduced={final_fourier['fourier_reduced_chi2']:.6e} | "
                f"delta_reduced={delta_fourier['delta_fourier_reduced_chi2']:+.6e} | "
                f"visibilities={init_fourier['fourier_visibility_count']}",
                flush=True,
            )

        sweep_extra: dict[str, object] = {}
        if sweep_mode:
            sweep_extra = {
                "transport_weight_sweep": True,
                "sweep_index": sweep_index,
                "sweep_total": sweep_total,
                "sweep_transport_weights": weights_to_run,
                "sweep_root_output": str(args.output),
            }

        result_path = save_reconstruction_npz(
            run_output_dir / "reconstruction.npz",
            state,
            config=params,
            static_sequence=initialization_sequence,
            reference_sequence=reference_sequence,
            ground_truth=ground_truth,
            names=[obs.path.name for obs in observations],
            extra={
                "mode": "reference_postprocess" if args.reference_postprocess else "joint_reconstruction",
                "observation_files": [str(obs.path) for obs in observations],
                "initialization_files": [str(path) for path in initialization_paths],
                "static_files": [str(path) for path in initialization_paths],
                "background_files": [str(path) for path in background_paths],
                "reference_files": [str(path) for path in reference_paths],
                "ground_truth_files": [str(path) for path in ground_truth_paths],
                "initialization_scale_info": initialization_scale_info,
                "background_scale_info": background_scale_info,
                "reference_scale_info": reference_scale_info,
                "automatic_metrics": automatic_metrics,
                **metric_files,
                **sweep_extra,
            },
        )
        if args.save_pngs:
            save_frame_pngs(state.image_state.image, run_output_dir / "frames", prefix="uot")
        config_extra = {
            "mode": "reference_postprocess" if args.reference_postprocess else "joint_reconstruction",
            "reference_postprocess": args.reference_postprocess,
            "background_mode": args.background_mode,
            "initialization": str(initialization_dir),
            "background": str(background_dir),
            "reference": None if reference_sequence is None else str(reference_dir or initialization_dir),
            "initialization_scale_mode": initialization_scale_mode,
            "background_scale_mode": background_scale_mode,
            "reference_scale_mode": reference_scale_mode,
            "ground_truth": str(args.ground_truth),
            "ground_truth_loaded": ground_truth is not None,
            "metric_normalization": args.metric_normalization,
            "metric_total_flux": args.metric_total_flux,
            "stge_lambda": args.stge_lambda,
            "automatic_metrics": automatic_metrics,
            **metric_files,
            "max_frames": args.max_frames,
            "max_visibilities_per_frame": args.max_visibilities_per_frame,
            **sweep_extra,
        }
        save_experiment_config(
            run_output_dir / "experiment_config.json",
            grid=grid,
            params=params,
            paths=ReconstructionPaths(
                observations_dir=args.observations,
                static_reconstruction_dir=initialization_dir,
                output_dir=run_output_dir,
                ground_truth_dir=args.ground_truth,
            ),
            extra=config_extra,
        )

        latest = state.history[-1]
        row: dict[str, object] = {
            "sweep_index": sweep_index if sweep_index is not None else 0,
            "transport_weight": float(current_transport_weight),
            "output_dir": str(run_output_dir),
            "result": str(result_path),
            "iterations": int(state.iteration),
            "final": latest.__dict__,
            "metrics": automatic_metrics,
            "objective": float(latest.objective),
            "data": float(latest.data),
            "reference": float(latest.reference),
            "tv": float(latest.tv),
            "transport": float(latest.transport),
            "initialization_mean_frame_nrmse": _get_nested_scalar(automatic_metrics, "initialization_metrics", "mean_frame_nrmse"),
            "post_uot_mean_frame_nrmse": _get_nested_scalar(automatic_metrics, "post_uot_metrics", "mean_frame_nrmse"),
            "delta_mean_frame_nrmse": _get_nested_scalar(automatic_metrics, "delta_metrics", "delta_mean_frame_nrmse"),
            "initialization_mean_ssim": _get_nested_scalar(automatic_metrics, "initialization_metrics", "mean_ssim"),
            "post_uot_mean_ssim": _get_nested_scalar(automatic_metrics, "post_uot_metrics", "mean_ssim"),
            "delta_mean_ssim": _get_nested_scalar(automatic_metrics, "delta_metrics", "delta_mean_ssim"),
            "initialization_stge": _get_nested_scalar(automatic_metrics, "initialization_metrics", "stge"),
            "post_uot_stge": _get_nested_scalar(automatic_metrics, "post_uot_metrics", "stge"),
            "delta_stge": _get_nested_scalar(automatic_metrics, "delta_metrics", "delta_stge"),
            "initialization_stge_temporal": _get_nested_scalar(automatic_metrics, "initialization_metrics", "stge_temporal"),
            "post_uot_stge_temporal": _get_nested_scalar(automatic_metrics, "post_uot_metrics", "stge_temporal"),
            "delta_stge_temporal": _get_nested_scalar(automatic_metrics, "delta_metrics", "delta_stge_temporal"),
            "initialization_fourier_reduced_chi2": _get_nested_scalar(automatic_metrics, "fourier_metrics", "initialization_fourier_metrics", "fourier_reduced_chi2"),
            "post_uot_fourier_reduced_chi2": _get_nested_scalar(automatic_metrics, "fourier_metrics", "post_uot_fourier_metrics", "fourier_reduced_chi2"),
            "delta_fourier_reduced_chi2": _get_nested_scalar(automatic_metrics, "fourier_metrics", "delta_fourier_metrics", "delta_fourier_reduced_chi2"),
            "initialization_fourier_chi2": _get_nested_scalar(automatic_metrics, "fourier_metrics", "initialization_fourier_metrics", "fourier_chi2"),
            "post_uot_fourier_chi2": _get_nested_scalar(automatic_metrics, "fourier_metrics", "post_uot_fourier_metrics", "fourier_chi2"),
            "delta_fourier_chi2": _get_nested_scalar(automatic_metrics, "fourier_metrics", "delta_fourier_metrics", "delta_fourier_chi2"),
        }
        return row

    if sweep_mode:
        args.output.mkdir(parents=True, exist_ok=True)
        if not args.quiet:
            print(
                "Starting transport-weight sweep | "
                f"weights={[float(w) for w in weights_to_run]} | "
                f"frames={len(observations)} | output={args.output}",
                flush=True,
            )
        sweep_rows = []
        for index, weight in enumerate(weights_to_run):
            run_dir = args.output / _transport_weight_label(weight)
            sweep_rows.append(
                run_single_weight(
                    current_transport_weight=float(weight),
                    run_output_dir=run_dir,
                    sweep_index=index,
                    sweep_total=len(weights_to_run),
                )
            )
        sweep_json, sweep_csv = _write_sweep_outputs(args.output, sweep_rows)
        print(
            json.dumps(
                {
                    "sweep_parameter": "transport_weight",
                    "sweep_weights": [float(w) for w in weights_to_run],
                    "sweep_summary_json": str(sweep_json),
                    "sweep_summary_csv": str(sweep_csv),
                    "runs": sweep_rows,
                },
                indent=2,
            )
        )
    else:
        row = run_single_weight(
            current_transport_weight=float(transport_weight),
            run_output_dir=args.output,
        )
        print(json.dumps({"result": row["result"], "iterations": row["iterations"], "final": row["final"], "metrics": row["metrics"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
