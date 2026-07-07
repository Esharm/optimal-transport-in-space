"""Ground-truth frame loading utilities for evaluation-ready result files."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ot_uot.io.observations import infer_frame_index


def _load_truth_image(path: Path, shape: tuple[int, int] | None = None) -> np.ndarray:
    from PIL import Image

    image = Image.open(path).convert("F")
    if shape is not None:
        height, width = shape
        image = image.resize((width, height))
    arr = np.asarray(image, dtype=np.float64)
    if arr.size and float(np.nanmax(arr)) > 1.5:
        arr = arr / 255.0
    return np.maximum(arr, 0.0)


def load_ground_truth_sequence(
    directory: Path | str,
    frame_indices: list[int],
    *,
    shape: tuple[int, int] | None = None,
) -> tuple[np.ndarray | None, list[Path]]:
    """Load ground-truth PNG/JPG/TIFF frames matching observation indices.

    Returns ``(None, [])`` when the directory does not exist or not every
    requested frame can be matched.
    """

    directory = Path(directory)
    if not directory.exists():
        return None, []
    files: list[Path] = []
    for pattern in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"):
        files.extend(directory.glob(pattern))
    by_index = {
        infer_frame_index(path): path
        for path in files
        if path.stem and any(char.isdigit() for char in path.stem)
    }
    selected = []
    for index in frame_indices:
        path = by_index.get(int(index))
        if path is None:
            return None, []
        selected.append(path)
    frames = np.stack([_load_truth_image(path, shape=shape) for path in selected], axis=0)
    return frames.astype(np.float64), selected

