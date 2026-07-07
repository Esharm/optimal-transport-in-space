"""Command-line driver for standalone signed-residual OT/UOT reconstruction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ot_uot.core.background import BackgroundMode
from ot_uot.core.config import ImageGrid, TransportMethod, UOTParameters
from ot_uot.io.ground_truth import load_ground_truth_sequence
from ot_uot.io.observations import load_observation_directory
from ot_uot.io.config_io import save_experiment_config
from ot_uot.io.results import save_reconstruction_npz
from ot_uot.io.static_init import load_static_with_background
from ot_uot.optimization.signed_residual_admm import SignedResidualUOTADMM
from ot_uot.visualization.outputs import save_frame_pngs


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OBSERVATIONS = PROJECT_ROOT / "blackhole_sim_testing" / "observations_fixed_npz"
DEFAULT_STATIC = PROJECT_ROOT / "starwarps_results" / "resized_128"
DEFAULT_GROUND_TRUTH = PROJECT_ROOT / "blackhole_sim" / "data" / "aart_frames"
DEFAULT_OUTPUT = PROJECT_ROOT / "ot_uot_results"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, default=DEFAULT_OBSERVATIONS)
    parser.add_argument("--static", type=Path, default=DEFAULT_STATIC)
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GROUND_TRUTH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--transport-method", choices=[m.value for m in TransportMethod], default=TransportMethod.PAIRWISE_UOT.value)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--fov-rad", type=float, default=160e-6 / 206265.0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--max-visibilities-per-frame", type=int, default=None)
    parser.add_argument("--background-mode", choices=[m.value for m in BackgroundMode], default=BackgroundMode.MEAN.value)
    parser.add_argument("--static-scale-mode", choices=["none", "per_frame", "global"], default="per_frame")
    parser.add_argument("--no-normalize-static-images", action="store_true")
    parser.add_argument("--tv-weight", type=float, default=1e-5)
    parser.add_argument("--background-weight", type=float, default=1e-4)
    parser.add_argument("--residual-mass-weight", type=float, default=1e-6)
    parser.add_argument("--transport-weight", type=float, default=1e-4)
    parser.add_argument("--source-weight", type=float, default=30.0)
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
    frame_indices = [obs.frame_index for obs in observations]
    static_sequence, background, static_paths = load_static_with_background(
        args.static,
        mode=BackgroundMode(args.background_mode),
        frame_indices=frame_indices,
        max_frames=args.max_frames,
        normalize_images=not args.no_normalize_static_images,
        data_terms=[obs.data_term for obs in observations],
        scale_mode=args.static_scale_mode,
    )
    if static_sequence.shape[0] != len(observations):
        raise ValueError("static frames and observation frames must have the same count")
    ground_truth, ground_truth_paths = load_ground_truth_sequence(
        args.ground_truth,
        frame_indices,
        shape=grid.shape,
    )
    params = UOTParameters(
        transport_method=TransportMethod(args.transport_method),
        tv_weight=args.tv_weight,
        background_weight=args.background_weight,
        residual_mass_weight=args.residual_mass_weight,
        transport_weight=args.transport_weight,
        source_weight=args.source_weight,
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

    def print_progress(state):
        if args.quiet:
            return
        latest = state.history[-1]
        print(
            "ADMM "
            f"{latest.iteration:03d}/{params.max_admm_iters:03d} | "
            f"obj={latest.objective:.6e} | "
            f"data={latest.data:.3e} | "
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
        print(
            "Starting OT/UOT reconstruction | "
            f"frames={len(observations)} | "
            f"method={params.transport_method.value} | "
            f"transport_nodes={params.transport_nodes} | "
            f"max_admm_iters={params.max_admm_iters}",
            flush=True,
        )
    state = solver.run(
        static_sequence,
        background,
        [obs.data_term for obs in observations],
        callback=print_progress,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    result_path = save_reconstruction_npz(
        args.output / "reconstruction.npz",
        state,
        config=params,
        static_sequence=static_sequence,
        ground_truth=ground_truth,
        names=[obs.path.name for obs in observations],
        extra={
            "observation_files": [str(obs.path) for obs in observations],
            "static_files": [str(path) for path in static_paths],
            "ground_truth_files": [str(path) for path in ground_truth_paths],
        },
    )
    if args.save_pngs:
        save_frame_pngs(state.image_state.image, args.output / "frames", prefix="uot")
    save_experiment_config(
        args.output / "experiment_config.json",
        grid=grid,
        params=params,
        extra={
            "background_mode": args.background_mode,
            "static_scale_mode": args.static_scale_mode,
            "ground_truth": str(args.ground_truth),
            "ground_truth_loaded": ground_truth is not None,
            "max_frames": args.max_frames,
            "max_visibilities_per_frame": args.max_visibilities_per_frame,
        },
    )
    print(json.dumps({"result": str(result_path), "iterations": state.iteration, "final": state.history[-1].__dict__}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
