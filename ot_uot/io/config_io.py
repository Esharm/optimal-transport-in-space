"""JSON configuration serialization for reproducible experiments."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path

from ot_uot.core.config import ImageGrid, ReconstructionPaths, TransportMethod, UOTParameters


def _jsonable(value):
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "value"):
        return value.value
    return value


def save_experiment_config(
    path: Path | str,
    *,
    grid: ImageGrid,
    params: UOTParameters,
    paths: ReconstructionPaths | None = None,
    extra: dict | None = None,
) -> Path:
    """Write a complete experiment configuration to JSON."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "grid": _jsonable(grid),
        "params": _jsonable(params),
        "paths": _jsonable(paths) if paths is not None else None,
        "extra": _jsonable(extra or {}),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


def load_experiment_config(path: Path | str) -> tuple[ImageGrid, UOTParameters, ReconstructionPaths | None, dict]:
    """Read an experiment configuration written by :func:`save_experiment_config`."""

    payload = json.loads(Path(path).read_text())
    grid = ImageGrid(**payload["grid"])
    params_payload = dict(payload["params"])
    params_payload["transport_method"] = TransportMethod(params_payload["transport_method"])
    params = UOTParameters(**params_payload)
    paths_payload = payload.get("paths")
    paths = None
    if paths_payload is not None:
        if paths_payload.get("ground_truth_dir") is not None:
            paths_payload["ground_truth_dir"] = Path(paths_payload["ground_truth_dir"])
        paths = ReconstructionPaths(
            observations_dir=Path(paths_payload["observations_dir"]),
            static_reconstruction_dir=Path(paths_payload["static_reconstruction_dir"]),
            output_dir=Path(paths_payload["output_dir"]),
            ground_truth_dir=paths_payload.get("ground_truth_dir"),
        )
    return grid, params, paths, dict(payload.get("extra") or {})

