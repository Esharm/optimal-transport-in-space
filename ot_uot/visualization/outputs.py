"""Visualization outputs for reconstructed image sequences."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def normalize_for_display(image: np.ndarray, vmin: float | None = None, vmax: float | None = None) -> np.ndarray:
    """Convert a floating image to uint8 display values."""

    image = np.asarray(image, dtype=np.float64)
    if vmin is None:
        vmin = float(np.min(image))
    if vmax is None:
        vmax = float(np.max(image))
    scaled = (image - vmin) / (vmax - vmin + 1e-30)
    return np.clip(np.round(255.0 * scaled), 0, 255).astype(np.uint8)


def save_frame_pngs(
    sequence: np.ndarray,
    output_dir: Path | str,
    *,
    prefix: str = "frame",
    shared_scale: bool = True,
) -> list[Path]:
    """Save a sequence of frames as PNG files."""

    from PIL import Image

    sequence = np.asarray(sequence, dtype=np.float64)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vmin = float(np.min(sequence)) if shared_scale else None
    vmax = float(np.max(sequence)) if shared_scale else None
    paths: list[Path] = []
    for k, frame in enumerate(sequence):
        arr = normalize_for_display(frame, vmin, vmax)
        path = output_dir / f"{prefix}_{k:04d}.png"
        Image.fromarray(arr, mode="L").save(path)
        paths.append(path)
    return paths


def save_comparison_strip(
    sequences: dict[str, np.ndarray],
    output_path: Path | str,
    *,
    frame_index: int = 0,
) -> Path:
    """Save a horizontal strip comparing named sequences at one frame."""

    from PIL import Image, ImageDraw

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    all_values = np.concatenate([np.asarray(seq, dtype=np.float64).ravel() for seq in sequences.values()])
    vmin = float(np.min(all_values))
    vmax = float(np.max(all_values))
    tiles = []
    labels = []
    for label, seq in sequences.items():
        arr = normalize_for_display(seq[frame_index], vmin, vmax)
        tiles.append(Image.fromarray(arr, mode="L").convert("RGB"))
        labels.append(label)
    width = sum(tile.width for tile in tiles)
    height = max(tile.height for tile in tiles) + 22
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    x = 0
    for label, tile in zip(labels, tiles):
        canvas.paste(tile, (x, 22))
        draw.text((x + 3, 4), label, fill=(0, 0, 0))
        x += tile.width
    canvas.save(output_path)
    return output_path

