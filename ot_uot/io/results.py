"""Result serialization for standalone OT/UOT experiments."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path

import numpy as np

from ot_uot.optimization.admm_state import ADMMState
from ot_uot.optimization.signed_residual_admm import history_as_dicts


def dataclass_to_jsonable(obj):
    """Convert dataclass configuration objects to JSON-compatible values."""

    if is_dataclass(obj):
        return dataclass_to_jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): dataclass_to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [dataclass_to_jsonable(v) for v in obj]
    if hasattr(obj, "value"):
        return obj.value
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_reconstruction_npz(
    output_path: Path | str,
    state: ADMMState,
    *,
    config=None,
    static_sequence: np.ndarray | None = None,
    ground_truth: np.ndarray | None = None,
    names: list[str] | None = None,
    extra: dict | None = None,
) -> Path:
    """Save reconstruction variables and history to a compressed NPZ."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "iteration": state.iteration,
        "history": history_as_dicts(state),
        "config": dataclass_to_jsonable(config) if config is not None else None,
        "extra": dataclass_to_jsonable(extra or {}),
    }
    image = state.image_state.image
    background = state.image_state.background
    if background.ndim == 2:
        background_video = np.repeat(background[None, :, :], image.shape[0], axis=0)
    else:
        background_video = background
    payload = {
        "image": image,
        "joint": image,
        "positive": state.image_state.positive,
        "negative": state.image_state.negative,
        "joint_positive_residual": state.image_state.positive,
        "joint_negative_residual": state.image_state.negative,
        "background": background,
        "background_video": background_video,
        "decomposition_dual": state.decomposition_dual,
        "metadata": json.dumps(metadata),
    }
    if static_sequence is not None:
        static = np.asarray(static_sequence, dtype=np.float64)
        payload["static"] = static
        payload["joint_initialization"] = static
        payload["static_positive_residual"] = np.maximum(static - background_video, 0.0)
        payload["static_negative_residual"] = np.maximum(background_video - static, 0.0)
    if ground_truth is not None:
        payload["gt"] = np.asarray(ground_truth, dtype=np.float64)
    if names is not None:
        payload["names"] = np.asarray(names)
    np.savez_compressed(output_path, **payload)
    return output_path


def load_reconstruction_npz(path: Path | str) -> dict:
    """Load a saved reconstruction NPZ."""

    with np.load(Path(path), allow_pickle=False) as npz:
        result = {key: npz[key] for key in npz.files if key != "metadata"}
        result["metadata"] = json.loads(str(npz["metadata"]))
    return result
