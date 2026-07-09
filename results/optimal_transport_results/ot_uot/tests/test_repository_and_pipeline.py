"""Repository integrity and pipeline-level verification tests."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from ot_uot.core.config import ImageGrid, ReconstructionPaths, TransportMethod, UOTParameters
from ot_uot.core.visibility import ComplexVisibilityDataTerm, DirectVisibilityOperator
from ot_uot.evaluation.metrics import metric_summary
from ot_uot.io.config_io import load_experiment_config, save_experiment_config
from ot_uot.io.ground_truth import load_ground_truth_sequence
from ot_uot.io.observations import load_observation_directory
from ot_uot.io.results import load_reconstruction_npz, save_reconstruction_npz
from ot_uot.io.static_init import calibrate_static_sequence, load_static_sequence
from ot_uot.optimization.signed_residual_admm import SignedResidualUOTADMM
from ot_uot.visualization.outputs import save_comparison_strip, save_frame_pngs


SOURCE_MODULES = [
    "ot_uot",
    "ot_uot.__main__",
    "ot_uot.core.background",
    "ot_uot.core.config",
    "ot_uot.core.finite_differences",
    "ot_uot.core.projections",
    "ot_uot.core.variables",
    "ot_uot.core.visibility",
    "ot_uot.transport.continuity",
    "ot_uot.transport.kinetic_prox",
    "ot_uot.transport.path_solver",
    "ot_uot.transport.pairwise",
    "ot_uot.transport.global_velocity",
    "ot_uot.regularizers.tv",
    "ot_uot.regularizers.image_residual_update",
    "ot_uot.optimization.admm_state",
    "ot_uot.optimization.convergence",
    "ot_uot.optimization.objective",
    "ot_uot.optimization.signed_residual_admm",
    "ot_uot.io.config_io",
    "ot_uot.io.ground_truth",
    "ot_uot.io.observations",
    "ot_uot.io.static_init",
    "ot_uot.io.results",
    "ot_uot.evaluation.metrics",
    "ot_uot.visualization.outputs",
    "ot_uot.drivers.run_reconstruction",
]


def make_data_term(image: np.ndarray) -> ComplexVisibilityDataTerm:
    shape = image.shape
    u = np.asarray([0.0, 0.4, -0.3])
    v = np.asarray([0.0, -0.2, 0.5])
    weight = np.ones(3)
    operator = DirectVisibilityOperator(
        u=u,
        v=v,
        weight=weight,
        shape=shape,
        fov_rad=1.0,
        data_scale=1.0,
        use_cache=True,
    )
    observed = operator.forward(image)
    return ComplexVisibilityDataTerm(operator=operator, observed=observed)


class RepositoryIntegrityTests(unittest.TestCase):
    def test_all_source_modules_import(self) -> None:
        for module in SOURCE_MODULES:
            with self.subTest(module=module):
                importlib.import_module(module)

    def test_config_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            grid = ImageGrid(height=16, width=12, fov_rad=2.0)
            params = UOTParameters(
                transport_method=TransportMethod.GLOBAL_VELOCITY,
                max_admm_iters=4,
                min_admm_iters=1,
            )
            paths = ReconstructionPaths(
                observations_dir=tmp_path / "obs",
                static_reconstruction_dir=tmp_path / "static",
                output_dir=tmp_path / "out",
            )
            config_path = save_experiment_config(tmp_path / "config.json", grid=grid, params=params, paths=paths, extra={"tag": "unit"})
            grid2, params2, paths2, extra = load_experiment_config(config_path)
            self.assertEqual(grid2.shape, (16, 12))
            self.assertEqual(params2.transport_method, TransportMethod.GLOBAL_VELOCITY)
            self.assertEqual(paths2.output_dir, tmp_path / "out")
            self.assertEqual(extra["tag"], "unit")


class PipelineIOTests(unittest.TestCase):
    def test_observation_loader_uses_global_scale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            for k, amp in enumerate((1.0, 3.0)):
                np.savez(
                    tmp_path / f"frame_{k:04d}.npz",
                    u=np.asarray([0.0, 1.0]),
                    v=np.asarray([0.0, -1.0]),
                    vis=amp * np.asarray([1.0 + 0.0j, 0.5 + 0.2j]),
                    sigma=np.asarray([1.0, 1.0]),
                )
            frames = load_observation_directory(tmp_path, ImageGrid(height=4, width=4, fov_rad=1.0))
            self.assertEqual(len(frames), 2)
            self.assertAlmostEqual(frames[0].data_scale, frames[1].data_scale)
            self.assertGreater(frames[0].data_term.observed.size, 0)

    def test_static_png_normalization_and_calibration(self) -> None:
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            arr = np.zeros((5, 5), dtype=np.uint8)
            arr[2, 2] = 255
            Image.fromarray(arr).save(tmp_path / "frame_0000.png")
            frames, _ = load_static_sequence(tmp_path)
            self.assertLessEqual(float(frames.max()), 1.0)

            truth = np.zeros((1, 5, 5))
            truth[0, 2, 2] = 2.0
            data_terms = [make_data_term(truth[0])]
            scaled, info = calibrate_static_sequence(frames, data_terms, mode="per_frame")
            self.assertAlmostEqual(float(scaled[0, 2, 2]), 2.0, places=5)
            self.assertEqual(info["static_recon_scale_mode"], "per_frame")

    def test_ground_truth_loader_matches_frame_indices(self) -> None:
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            for index in (0, 2):
                arr = np.zeros((5, 5), dtype=np.uint8)
                arr[index + 1, 2] = 255
                Image.fromarray(arr).save(tmp_path / f"frame_{index:03d}.png")
            frames, paths = load_ground_truth_sequence(tmp_path, [0, 2], shape=(4, 4))
            self.assertIsNotNone(frames)
            self.assertEqual(frames.shape, (2, 4, 4))
            self.assertEqual([path.name for path in paths], ["frame_000.png", "frame_002.png"])

    def test_result_save_load_legacy_keys_and_visualization(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            static = np.ones((2, 4, 4))
            background = np.ones((4, 4))
            data_terms = [make_data_term(static[0]), make_data_term(static[1])]
            params = UOTParameters(
                transport_method=TransportMethod.PAIRWISE_UOT,
                tv_weight=0.0,
                max_admm_iters=1,
                min_admm_iters=1,
                transport_inner_iters=5,
                image_inner_iters=5,
            )
            state = SignedResidualUOTADMM(params).run(static, background, data_terms)
            result_path = save_reconstruction_npz(
                tmp_path / "reconstruction.npz",
                state,
                config=params,
                static_sequence=static,
                ground_truth=static,
                names=["frame_0000.npz", "frame_0001.npz"],
            )
            loaded = load_reconstruction_npz(result_path)
            self.assertIn("joint", loaded)
            self.assertIn("static", loaded)
            self.assertIn("background_video", loaded)
            summary = metric_summary(loaded["joint"], loaded["gt"])
            self.assertTrue(np.isfinite(summary["nrmse"]))
            frames = save_frame_pngs(loaded["joint"], tmp_path / "pngs")
            self.assertEqual(len(frames), 2)
            strip = save_comparison_strip({"gt": loaded["gt"], "joint": loaded["joint"]}, tmp_path / "strip.png")
            self.assertTrue(strip.exists())


if __name__ == "__main__":
    unittest.main()

class ReferencePostprocessCLITests(unittest.TestCase):
    def test_reference_postprocess_cli_smoke(self) -> None:
        from PIL import Image
        from ot_uot.drivers.run_reconstruction import main as driver_main

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            obs_dir = tmp_path / "obs"
            init_dir = tmp_path / "init"
            gt_dir = tmp_path / "gt"
            out_dir = tmp_path / "out"
            obs_dir.mkdir()
            init_dir.mkdir()
            gt_dir.mkdir()
            for k in range(2):
                image = np.zeros((6, 6), dtype=np.float64)
                image[2, 2 + k] = 1.0
                image_uint8 = np.asarray(255 * image, dtype=np.uint8)
                Image.fromarray(image_uint8).save(init_dir / f"frame_{k:04d}.png")
                Image.fromarray(image_uint8).save(gt_dir / f"frame_{k:03d}.png")
                np.savez(
                    obs_dir / f"frame_{k:04d}.npz",
                    u=np.asarray([0.0, 0.25]),
                    v=np.asarray([0.0, -0.15]),
                    vis=np.asarray([1.0 + 0.0j, 0.2 + 0.1j]),
                    sigma=np.asarray([1.0, 1.0]),
                )
            rc = driver_main([
                "--observations", str(obs_dir),
                "--initialization", str(init_dir),
                "--ground-truth", str(gt_dir),
                "--output", str(out_dir),
                "--height", "6",
                "--width", "6",
                "--fov-rad", "1.0",
                "--reference-postprocess",
                "--max-admm-iters", "1",
                "--min-admm-iters", "1",
                "--transport-inner-iters", "5",
                "--image-inner-iters", "5",
                "--quiet",
            ])
            self.assertEqual(rc, 0)
            loaded = load_reconstruction_npz(out_dir / "reconstruction.npz")
            self.assertIn("reference_sequence", loaded)
            self.assertIn("starwarps_reference", loaded)
            # Backward-compatible final metric aliases.
            self.assertIn("metric_mean_frame_nrmse", loaded)
            self.assertIn("metric_mean_ssim", loaded)
            # Explicit initialization/final/delta metric keys.
            self.assertIn("metric_init_mean_frame_nrmse", loaded)
            self.assertIn("metric_init_mean_ssim", loaded)
            self.assertIn("metric_final_mean_frame_nrmse", loaded)
            self.assertIn("metric_final_mean_ssim", loaded)
            self.assertIn("metric_delta_mean_frame_nrmse", loaded)
            self.assertIn("metric_delta_mean_ssim", loaded)
            self.assertTrue((out_dir / "metrics_summary.json").exists())
            self.assertTrue((out_dir / "frame_metrics.csv").exists())
            metadata = loaded["metadata"]
            self.assertEqual(metadata["config"]["data_weight"], 0.0)
            self.assertEqual(metadata["config"]["tv_weight"], 0.0)
            self.assertEqual(metadata["config"]["background_weight"], 0.0)
            self.assertEqual(metadata["config"]["reference_weight"], 1.0)
            automatic_metrics = metadata["extra"]["automatic_metrics"]
            self.assertIn("initialization_metrics", automatic_metrics)
            self.assertIn("post_uot_metrics", automatic_metrics)
            self.assertIn("delta_metrics", automatic_metrics)
            init_metrics = automatic_metrics["initialization_metrics"]
            final_metrics = automatic_metrics["post_uot_metrics"]
            self.assertIn("mean_frame_nrmse", init_metrics)
            self.assertIn("mean_ssim", init_metrics)
            self.assertIn("mean_frame_nrmse", final_metrics)
            self.assertIn("mean_ssim", final_metrics)
            self.assertTrue(np.isfinite(init_metrics["mean_frame_nrmse"]))
            self.assertTrue(np.isfinite(init_metrics["mean_ssim"]))
            self.assertTrue(np.isfinite(final_metrics["mean_frame_nrmse"]))
            self.assertTrue(np.isfinite(final_metrics["mean_ssim"]))
            csv_text = (out_dir / "frame_metrics.csv").read_text()
            self.assertIn("initialization_nrmse", csv_text)
            self.assertIn("post_uot_ssim", csv_text)



class STGEMetricTests(unittest.TestCase):
    def test_stge_identity_and_manual_lambda(self) -> None:
        from ot_uot.evaluation.metrics import compare_initialization_and_final_fourier_reports, compare_initialization_and_final_reports, spatiotemporal_gradient_error

        gt = np.zeros((3, 5, 5), dtype=np.float64)
        for k in range(3):
            gt[k, 2, 1 + k] = 1.0
        report = spatiotemporal_gradient_error(gt, gt, temporal_weight="0.5")
        self.assertAlmostEqual(report["stge"], 0.0)
        self.assertAlmostEqual(report["stge_lambda"], 0.5)
        self.assertEqual(report["stge_lambda_mode"], "manual")

        combined = compare_initialization_and_final_reports(
            gt,
            gt,
            gt,
            normalization="minmax",
            stge_lambda="auto",
        )
        self.assertIn("stge", combined["initialization_metrics"])
        self.assertIn("stge_temporal", combined["post_uot_metrics"])
        self.assertIn("delta_stge", combined["delta_metrics"])
        self.assertAlmostEqual(combined["initialization_metrics"]["stge"], 0.0)
        self.assertAlmostEqual(combined["post_uot_metrics"]["stge"], 0.0)
        self.assertAlmostEqual(combined["delta_metrics"]["delta_stge"], 0.0)

    def test_fourier_chi2_identity(self) -> None:
        from ot_uot.evaluation.metrics import compare_initialization_and_final_fourier_reports

        image = np.zeros((5, 5), dtype=np.float64)
        image[2, 2] = 1.0
        data_term = make_data_term(image)
        sequence = np.stack([image, image])
        report = compare_initialization_and_final_fourier_reports(sequence, sequence, [data_term, data_term])
        init = report["initialization_fourier_metrics"]
        final = report["post_uot_fourier_metrics"]
        delta = report["delta_fourier_metrics"]
        self.assertAlmostEqual(init["fourier_chi2"], 0.0, places=12)
        self.assertAlmostEqual(final["fourier_reduced_chi2"], 0.0, places=12)
        self.assertAlmostEqual(delta["delta_fourier_reduced_chi2"], 0.0, places=12)


class TransportWeightSweepCLITests(unittest.TestCase):
    def test_transport_weight_sweep_cli_smoke(self) -> None:
        from PIL import Image
        from ot_uot.drivers.run_reconstruction import main as driver_main

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            obs_dir = tmp_path / "obs"
            init_dir = tmp_path / "init"
            gt_dir = tmp_path / "gt"
            out_dir = tmp_path / "sweep_out"
            obs_dir.mkdir()
            init_dir.mkdir()
            gt_dir.mkdir()
            for k in range(2):
                image = np.zeros((6, 6), dtype=np.float64)
                image[2, 2 + k] = 1.0
                image_uint8 = np.asarray(255 * image, dtype=np.uint8)
                Image.fromarray(image_uint8).save(init_dir / f"frame_{k:04d}.png")
                Image.fromarray(image_uint8).save(gt_dir / f"frame_{k:03d}.png")
                np.savez(
                    obs_dir / f"frame_{k:04d}.npz",
                    u=np.asarray([0.0, 0.25]),
                    v=np.asarray([0.0, -0.15]),
                    vis=np.asarray([1.0 + 0.0j, 0.2 + 0.1j]),
                    sigma=np.asarray([1.0, 1.0]),
                )
            rc = driver_main([
                "--observations", str(obs_dir),
                "--initialization", str(init_dir),
                "--ground-truth", str(gt_dir),
                "--output", str(out_dir),
                "--height", "6",
                "--width", "6",
                "--fov-rad", "1.0",
                "--max-frames", "2",
                "--sweep-transport-weights", "1e-8,1e-6",
                "--tv-weight", "0.0",
                "--background-weight", "0.0",
                "--residual-mass-weight", "0.0",
                "--transport-inner-iters", "3",
                "--image-inner-iters", "3",
                "--max-admm-iters", "1",
                "--min-admm-iters", "1",
                "--metric-normalization", "minmax",
                "--quiet",
            ])
            self.assertEqual(rc, 0)
            self.assertTrue((out_dir / "sweep_summary.json").exists())
            self.assertTrue((out_dir / "sweep_summary.csv").exists())
            summary = json.loads((out_dir / "sweep_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["sweep_parameter"], "transport_weight")
            self.assertEqual(len(summary["runs"]), 2)
            for label in ("transport_weight_1e-08", "transport_weight_1e-06"):
                run_dir = out_dir / label
                self.assertTrue((run_dir / "reconstruction.npz").exists())
                self.assertTrue((run_dir / "metrics_summary.json").exists())
                loaded = load_reconstruction_npz(run_dir / "reconstruction.npz")
                self.assertTrue(loaded["metadata"]["extra"]["transport_weight_sweep"])
                metrics = json.loads((run_dir / "metrics_summary.json").read_text(encoding="utf-8"))
                self.assertIn("fourier_metrics", metrics)
                self.assertIn("fourier_reduced_chi2", metrics["fourier_metrics"]["post_uot_fourier_metrics"])

    def test_transport_weight_sweep_parser(self) -> None:
        from ot_uot.drivers.run_reconstruction import _parse_transport_weight_sweep, _transport_weight_label

        self.assertIsNone(_parse_transport_weight_sweep(None))
        self.assertEqual(_parse_transport_weight_sweep([]), [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3])
        self.assertEqual(_parse_transport_weight_sweep(["1e-8,1e-6", "1e-4"]), [1e-8, 1e-6, 1e-4])
        self.assertEqual(_transport_weight_label(1e-8), "transport_weight_1e-08")
