"""Small data models used by CrossRMSD."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True, order=True)
class AtomKey:
    """Stable atom identity used across PDB frames and trajectories."""

    chain: str
    resid: int
    atomname: str

    def display(self) -> str:
        chain = self.chain if self.chain else "<blank>"
        return f"chain {chain}, residue {self.resid}, atom {self.atomname}"


@dataclass(frozen=True)
class AtomRecord:
    """Atom metadata read from the first retained PDB model."""

    key: AtomKey
    resname: str
    index_1based: int
    element: str


@dataclass(frozen=True)
class Trajectory:
    """One selected multi-model PDB trajectory loaded into memory."""

    label: str
    path: Path
    atoms: tuple[AtomRecord, ...]
    coords_angstrom: np.ndarray
    source_frame_indices: np.ndarray
    frame_times_ps: np.ndarray | None = None

    @property
    def n_frames(self) -> int:
        return int(self.coords_angstrom.shape[0])

    @property
    def n_atoms(self) -> int:
        return int(self.coords_angstrom.shape[1])


@dataclass(frozen=True)
class LoadedTrajectory:
    """Trajectory plus selection and reader diagnostics."""

    trajectory: Trajectory
    fit_indices: np.ndarray
    rmsd_indices: np.ndarray
    source_frame_count: int
    first_model_atom_count: int
    same_order_frames: int
    remapped_frames: int
    read_seconds: float


@dataclass(frozen=True)
class RmsdStats:
    """Default summary statistics for one Cross-RMSD distribution."""

    n_pairs: int
    mean: float
    sd: float
    median: float


@dataclass(frozen=True)
class ComparisonResult:
    """Files and statistics produced for one trajectory comparison."""

    name: str
    kind: str
    label_a: str
    label_b: str
    path_a: Path
    path_b: Path
    frames_a: int
    frames_b: int
    stats: RmsdStats
    kde_file: str
    pairs_file: str
    xpm_file: str
    bandwidth: float
