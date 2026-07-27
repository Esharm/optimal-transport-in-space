"""Configuration dataclasses for the standalone OT/UOT framework."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class TransportMethod(str, Enum):
    """Supported dynamic transport formulations."""

    PAIRWISE_UOT = "pairwise_uot"
    GLOBAL_VELOCITY = "global_velocity"


@dataclass(frozen=True)
class ImageGrid:
    """Image grid metadata used by image and Fourier operators."""

    height: int = 128
    width: int = 128
    fov_rad: float = 160e-6 / 206265.0

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    @property
    def pixel_size_rad(self) -> float:
        return self.fov_rad / float(self.width)


@dataclass(frozen=True)
class UOTParameters:
    """Regularization and optimization parameters for signed-residual UOT."""

    transport_method: TransportMethod = TransportMethod.PAIRWISE_UOT
    data_weight: float = 1.0
    tv_weight: float = 1e-5
    hessian_weight: float = 0.0
    image_l1_weight: float = 0.0
    background_weight: float = 1e-4
    reference_weight: float = 0.0
    residual_mass_weight: float = 1e-6
    transport_weight: float = 1e-4
    source_weight: float = 30.0
    decomposition_penalty: float = 1e-3
    endpoint_penalty: float = 1e-3
    transport_nodes: int = 7
    transport_inner_iters: int = 100
    image_inner_iters: int = 50
    max_admm_iters: int = 60
    min_admm_iters: int = 12
    abs_tol: float = 1e-4
    rel_tol: float = 5e-3
    patience: int = 5
    dual_relaxation: float = 1.0
    random_seed: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.transport_method, str):
            object.__setattr__(self, "transport_method", TransportMethod(self.transport_method))
        if self.transport_nodes < 2:
            raise ValueError("transport_nodes must be at least 2")
        nonnegative = {
            "data_weight": self.data_weight,
            "tv_weight": self.tv_weight,
            "hessian_weight": self.hessian_weight,
            "image_l1_weight": self.image_l1_weight,
            "background_weight": self.background_weight,
            "reference_weight": self.reference_weight,
            "residual_mass_weight": self.residual_mass_weight,
        }
        for name, value in nonnegative.items():
            if float(value) < 0.0:
                raise ValueError(f"{name} must be nonnegative")
        positive = {
            "transport_weight": self.transport_weight,
            "source_weight": self.source_weight,
            "decomposition_penalty": self.decomposition_penalty,
            "endpoint_penalty": self.endpoint_penalty,
        }
        for name, value in positive.items():
            if float(value) <= 0.0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class ReconstructionPaths:
    """Filesystem locations used by the standalone driver."""

    observations_dir: Path
    static_reconstruction_dir: Path
    output_dir: Path
    ground_truth_dir: Path | None = None
