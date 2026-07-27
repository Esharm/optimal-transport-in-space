#!/usr/bin/env python3
"""
Export EHT-style red/orange PNGs from the `data_env` dataset of an INOISY HDF5 file.

Default behavior:
- maps the first five AART frames to the nearest INOISY time indices
- assumes INOISY spans 0..128 M with 256 stored time samples
- assumes AART spans 0..90 M with 15 snapshots
- saves images using Matplotlib's `afmhot` colormap

Under those defaults, AART frames 0..4 map to H5 indices:
    [0, 13, 26, 38, 51]

Examples:
    python export_data_env_frames_red.py source.h5

    python export_data_env_frames_red.py source.h5 \
        --output-dir figures/inoisy --global-scale

    python export_data_env_frames_red.py source.h5 \
        --aart-start 0 --aart-end 12 --aart-snapshots 12

    python export_data_env_frames_red.py source.h5 \
        --h5-indices 0 13 26 38 51
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export selected `data_env` frames as EHT-style red/orange PNGs."
    )
    parser.add_argument("h5_file", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data_env_frames_red"),
    )
    parser.add_argument(
        "--dataset",
        default="data_env",
        help="Dataset name or full HDF5 path.",
    )
    parser.add_argument("--num-frames", type=int, default=5)
    parser.add_argument("--aart-start", type=float, default=0.0)
    parser.add_argument("--aart-end", type=float, default=90.0)
    parser.add_argument("--aart-snapshots", type=int, default=15)
    parser.add_argument("--h5-start", type=float, default=0.0)
    parser.add_argument("--h5-end", type=float, default=128.0)
    parser.add_argument(
        "--h5-indices",
        type=int,
        nargs="+",
        default=None,
        help="Explicit zero-based H5 indices; overrides time mapping.",
    )
    parser.add_argument(
        "--global-scale",
        action="store_true",
        help="Use one shared intensity scale for all exported frames.",
    )
    parser.add_argument(
        "--origin",
        choices=("lower", "upper"),
        default="lower",
    )
    parser.add_argument(
        "--cmap",
        default="afmhot",
        help="Matplotlib colormap (default: afmhot).",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def find_dataset(h5: h5py.File, requested: str) -> h5py.Dataset:
    candidates = [requested, requested.lstrip("/"), "/" + requested.lstrip("/")]
    for candidate in candidates:
        if candidate in h5 and isinstance(h5[candidate], h5py.Dataset):
            return h5[candidate]

    matches = []

    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            if name.split("/")[-1] == requested.split("/")[-1]:
                matches.append(name)

    h5.visititems(visit)

    if not matches:
        available = []

        def collect(name, obj):
            if isinstance(obj, h5py.Dataset):
                available.append(name)

        h5.visititems(collect)
        raise KeyError(
            f"Could not find dataset {requested!r}. "
            f"Available datasets: {available}"
        )

    if len(matches) > 1:
        raise KeyError(
            f"Dataset name {requested!r} is ambiguous. "
            f"Matches: {matches}. Pass the full path with --dataset."
        )

    return h5[matches[0]]


def infer_time_axis(shape: tuple[int, ...]) -> int:
    if len(shape) != 3:
        raise ValueError(f"Expected a 3-D dataset; received {shape}.")

    if shape[1] == shape[2] and shape[0] != shape[1]:
        return 0

    if shape[0] == shape[1] and shape[2] != shape[0]:
        return 2

    return 0


def map_aart_to_h5(
    h5_frame_count: int,
    h5_start: float,
    h5_end: float,
    aart_start: float,
    aart_end: float,
    aart_snapshots: int,
    num_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    if h5_end <= h5_start:
        raise ValueError("--h5-end must be greater than --h5-start.")
    if aart_snapshots < 1:
        raise ValueError("--aart-snapshots must be at least 1.")
    if not 1 <= num_frames <= aart_snapshots:
        raise ValueError("--num-frames must be between 1 and --aart-snapshots.")

    if aart_snapshots == 1:
        times = np.array([aart_start], dtype=float)
    else:
        times = np.linspace(
            aart_start,
            aart_end,
            aart_snapshots,
            dtype=float,
        )[:num_frames]

    indices = np.rint(
        (times - h5_start)
        / (h5_end - h5_start)
        * (h5_frame_count - 1)
    ).astype(int)

    return indices, times


def extract_frame(
    dataset: h5py.Dataset,
    time_axis: int,
    index: int,
) -> np.ndarray:
    selector = [slice(None), slice(None), slice(None)]
    selector[time_axis] = index

    frame = np.asarray(dataset[tuple(selector)], dtype=np.float64)
    frame = np.squeeze(frame)

    if frame.ndim != 2:
        raise ValueError(
            f"Frame {index} did not reduce to 2-D; got shape {frame.shape}."
        )

    return np.nan_to_num(frame, nan=0.0, posinf=0.0, neginf=0.0)


def save_frame(
    frame: np.ndarray,
    path: Path,
    vmin: float,
    vmax: float,
    origin: str,
    cmap: str,
    dpi: int,
) -> None:
    if vmax <= vmin:
        vmax = vmin + 1.0

    plt.imsave(
        path,
        frame,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        origin=origin,
        dpi=dpi,
    )


def main() -> None:
    args = parse_args()

    if not args.h5_file.is_file():
        raise FileNotFoundError(args.h5_file)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.h5_file, "r") as h5:
        dataset = find_dataset(h5, args.dataset)
        dataset_name = dataset.name
        dataset_shape = dataset.shape

        time_axis = infer_time_axis(dataset.shape)
        h5_frame_count = dataset.shape[time_axis]

        if args.h5_indices is not None:
            indices = np.asarray(args.h5_indices, dtype=int)
            aart_times = None
        else:
            indices, aart_times = map_aart_to_h5(
                h5_frame_count=h5_frame_count,
                h5_start=args.h5_start,
                h5_end=args.h5_end,
                aart_start=args.aart_start,
                aart_end=args.aart_end,
                aart_snapshots=args.aart_snapshots,
                num_frames=args.num_frames,
            )

        if np.any(indices < 0) or np.any(indices >= h5_frame_count):
            raise IndexError(
                f"Requested indices {indices.tolist()} are outside "
                f"0..{h5_frame_count - 1}."
            )

        frames = [
            extract_frame(dataset, time_axis, int(index))
            for index in indices
        ]

    if args.global_scale:
        vmin = min(float(frame.min()) for frame in frames)
        vmax = max(float(frame.max()) for frame in frames)
        scales = [(vmin, vmax)] * len(frames)
    else:
        scales = [
            (float(frame.min()), float(frame.max()))
            for frame in frames
        ]

    print(f"Input: {args.h5_file}")
    print(f"Dataset: {dataset_name}")
    print(f"Shape: {dataset_shape}")
    print(f"Time axis: {time_axis}")
    print(f"Colormap: {args.cmap}")
    print(f"Selected H5 indices: {indices.tolist()}")

    for output_number, (index, frame, scale) in enumerate(
        zip(indices, frames, scales)
    ):
        output_path = (
            args.output_dir / f"inoisy_frame_{int(index):03d}.png"
        )

        save_frame(
            frame=frame,
            path=output_path,
            vmin=scale[0],
            vmax=scale[1],
            origin=args.origin,
            cmap=args.cmap,
            dpi=args.dpi,
        )

        if aart_times is None:
            print(f"H5 index {int(index):3d} -> {output_path}")
        else:
            print(
                f"AART frame {output_number:2d}, "
                f"t={aart_times[output_number]:.3f}M "
                f"-> H5 index {int(index):3d} "
                f"-> {output_path}"
            )


if __name__ == "__main__":
    main()