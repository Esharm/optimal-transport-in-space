"""Observation loading utilities for standalone OT/UOT reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np

from ot_uot.core.config import ImageGrid
from ot_uot.core.visibility import ComplexVisibilityDataTerm, DirectVisibilityOperator


@dataclass(frozen=True)
class ObservationFrame:
    """One frame of complex-visibility observations."""

    path: Path
    frame_index: int
    data_term: ComplexVisibilityDataTerm
    data_scale: float


def infer_frame_index(path: Path) -> int:
    """Infer an integer frame index from a file name."""

    matches = re.findall(r"\d+", path.stem)
    if not matches:
        raise ValueError(f"could not infer frame index from {path.name}")
    return int(matches[-1])


def _extract_npz_fields(npz) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if "data" in npz.files:
        data = npz["data"]
        names = set(data.dtype.names or ())
        if {"u", "v", "vis", "sigma"}.issubset(names):
            sigma = np.maximum(np.asarray(data["sigma"], dtype=np.float64), 1e-30)
            weight = 1.0 / (sigma * sigma)
            return (
                np.asarray(data["u"], dtype=np.float64),
                np.asarray(data["v"], dtype=np.float64),
                np.asarray(data["vis"], dtype=np.complex128),
                weight,
            )
        if {"u", "v", "vis", "weight"}.issubset(names):
            return (
                np.asarray(data["u"], dtype=np.float64),
                np.asarray(data["v"], dtype=np.float64),
                np.asarray(data["vis"], dtype=np.complex128),
                np.asarray(data["weight"], dtype=np.float64),
            )

    aliases = {
        "u": ("u", "uu", "ucoord"),
        "v": ("v", "vv", "vcoord"),
        "vis": ("vis", "visibility", "visibilities", "y"),
        "weight": ("weight", "weights", "w"),
        "sigma": ("sigma", "sigmas", "noise"),
    }

    def find(name: str):
        for key in aliases[name]:
            if key in npz.files:
                return npz[key]
        return None

    u = find("u")
    v = find("v")
    vis = find("vis")
    weight = find("weight")
    sigma = find("sigma")
    if u is None or v is None or vis is None:
        raise ValueError(f"NPZ is missing required visibility fields; found {npz.files}")
    if weight is None:
        if sigma is None:
            weight = np.ones_like(np.asarray(u, dtype=np.float64))
        else:
            sigma = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-30)
            weight = 1.0 / (sigma * sigma)
    return (
        np.asarray(u, dtype=np.float64),
        np.asarray(v, dtype=np.float64),
        np.asarray(vis, dtype=np.complex128),
        np.asarray(weight, dtype=np.float64),
    )


def _make_observation_frame(
    path: Path,
    grid: ImageGrid,
    u: np.ndarray,
    v: np.ndarray,
    vis: np.ndarray,
    weight: np.ndarray,
    data_scale: float,
    *,
    use_cache: bool,
    chunk_size: int,
) -> ObservationFrame:
    weight = np.maximum(np.asarray(weight, dtype=np.float64), 0.0)
    positive = weight > 0.0
    u, v, vis, weight = u[positive], v[positive], vis[positive], weight[positive]
    if weight.size:
        weight = weight / (np.median(weight) + 1e-30)
    operator = DirectVisibilityOperator(
        u=u,
        v=v,
        weight=weight,
        shape=grid.shape,
        fov_rad=grid.fov_rad,
        data_scale=data_scale,
        use_cache=use_cache,
        chunk_size=chunk_size,
    )
    observed = np.sqrt(weight) * vis / operator.scale
    return ObservationFrame(
        path=path,
        frame_index=infer_frame_index(path),
        data_term=ComplexVisibilityDataTerm(operator=operator, observed=observed),
        data_scale=float(data_scale),
    )


def load_observation_npz(
    path: Path | str,
    grid: ImageGrid,
    *,
    max_visibilities: int | None = None,
    data_scale: float | None = None,
    use_cache: bool = False,
    chunk_size: int = 128,
) -> ObservationFrame:
    """Load one observation NPZ as a normalized complex-visibility data term."""

    path = Path(path)
    with np.load(path, allow_pickle=False) as npz:
        u, v, vis, weight = _extract_npz_fields(npz)
    if max_visibilities is not None and vis.size > int(max_visibilities):
        keep = np.linspace(0, vis.size - 1, int(max_visibilities)).astype(int)
        u, v, vis, weight = u[keep], v[keep], vis[keep], weight[keep]
    if data_scale is None:
        data_scale = float(np.percentile(np.abs(vis), 95)) if vis.size else 1.0
    return _make_observation_frame(
        path,
        grid,
        u,
        v,
        vis,
        weight,
        max(float(data_scale), 1e-30),
        use_cache=use_cache,
        chunk_size=chunk_size,
    )


def load_observation_directory(
    directory: Path | str,
    grid: ImageGrid,
    *,
    frame_indices: list[int] | None = None,
    max_frames: int | None = None,
    max_visibilities_per_frame: int | None = None,
    use_cache: bool = False,
    chunk_size: int = 128,
) -> list[ObservationFrame]:
    """Load all observation NPZ files from a directory."""

    directory = Path(directory)
    files = sorted(directory.glob("*.npz"), key=infer_frame_index)
    if frame_indices is not None:
        wanted = set(int(i) for i in frame_indices)
        files = [path for path in files if infer_frame_index(path) in wanted]
    if max_frames is not None:
        files = files[: int(max_frames)]
    if not files:
        raise FileNotFoundError(f"no observation NPZ files found in {directory}")
    raw = []
    amplitudes = []
    for path in files:
        with np.load(path, allow_pickle=False) as npz:
            u, v, vis, weight = _extract_npz_fields(npz)
        if max_visibilities_per_frame is not None and vis.size > int(max_visibilities_per_frame):
            keep = np.linspace(0, vis.size - 1, int(max_visibilities_per_frame)).astype(int)
            u, v, vis, weight = u[keep], v[keep], vis[keep], weight[keep]
        finite = (
            np.isfinite(u)
            & np.isfinite(v)
            & np.isfinite(vis.real)
            & np.isfinite(vis.imag)
            & np.isfinite(weight)
            & (weight > 0.0)
        )
        u, v, vis, weight = u[finite], v[finite], vis[finite], weight[finite]
        raw.append((path, u, v, vis, weight))
        if vis.size:
            amplitudes.append(np.abs(vis))
    if amplitudes:
        data_scale = max(float(np.percentile(np.concatenate(amplitudes), 95)), 1e-30)
    else:
        data_scale = 1.0
    return [
        _make_observation_frame(
            path,
            grid,
            u,
            v,
            vis,
            weight,
            data_scale,
            use_cache=use_cache,
            chunk_size=chunk_size,
        )
        for path, u, v, vis, weight in raw
    ]
