"""Iterative generalized-Procrustes centroid for molecular ensembles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from crossrmsd.geometry import kabsch_rotations_to_reference
from crossrmsd.models import AtomRecord


@dataclass(frozen=True)
class CentroidResult:
    fit_coords_angstrom: np.ndarray
    rmsd_coords_angstrom: np.ndarray
    iterations: int
    final_shift_angstrom: float


def _centroid(coords: np.ndarray, weights: np.ndarray | None) -> np.ndarray:
    if weights is None:
        return np.mean(coords, axis=0)
    return np.average(coords, axis=0, weights=weights)


def generalized_procrustes_centroid(
    fit_ensembles_angstrom: list[np.ndarray],
    rmsd_ensembles_angstrom: list[np.ndarray],
    fit_weights: np.ndarray | None = None,
    tolerance_angstrom: float = 1.0e-6,
    max_iterations: int = 100,
) -> CentroidResult:
    """Return a synthetic mean structure after iterative rigid-body alignment.

    Every sampled frame contributes equally.  Each frame is centered on its fit
    selection, optimally rotated onto the current mean fit coordinates, and the
    aligned fit/RMSD coordinates are averaged.  Iteration stops when the RMS
    change of the mean fit coordinates falls below ``tolerance_angstrom``.
    """

    if not fit_ensembles_angstrom or len(fit_ensembles_angstrom) != len(rmsd_ensembles_angstrom):
        raise ValueError("Centroid calculation requires matching fit and RMSD ensembles.")
    if max_iterations < 1 or tolerance_angstrom <= 0.0:
        raise ValueError("Invalid centroid convergence settings.")

    fit_arrays = [np.asarray(value, dtype=np.float64) for value in fit_ensembles_angstrom]
    rmsd_arrays = [np.asarray(value, dtype=np.float64) for value in rmsd_ensembles_angstrom]
    fit_shape = fit_arrays[0].shape[1:]
    rmsd_shape = rmsd_arrays[0].shape[1:]
    if len(fit_shape) != 2 or fit_shape[1] != 3 or fit_shape[0] < 3:
        raise ValueError("Centroid fitting requires at least three fit atoms.")
    if len(rmsd_shape) != 2 or rmsd_shape[1] != 3 or rmsd_shape[0] < 1:
        raise ValueError("Centroid output requires at least one RMSD atom.")
    for fit, rmsd in zip(fit_arrays, rmsd_arrays):
        if fit.shape[1:] != fit_shape or rmsd.shape[1:] != rmsd_shape:
            raise ValueError("Centroid ensembles have incompatible atom dimensions.")
        if fit.shape[0] != rmsd.shape[0]:
            raise ValueError("Fit and RMSD centroid arrays have different frame counts.")

    first_fit = fit_arrays[0][0]
    first_center = _centroid(first_fit, fit_weights)
    mean_fit = first_fit - first_center
    mean_rmsd = rmsd_arrays[0][0] - first_center

    final_shift = float("inf")
    total_frames = sum(array.shape[0] for array in fit_arrays)
    if total_frames < 1:
        raise ValueError("Centroid calculation has no frames.")

    for iteration in range(1, max_iterations + 1):
        fit_sum = np.zeros_like(mean_fit)
        rmsd_sum = np.zeros_like(mean_rmsd)
        for fit_ensemble, rmsd_ensemble in zip(fit_arrays, rmsd_arrays):
            for fit_frame, rmsd_frame in zip(fit_ensemble, rmsd_ensemble):
                frame_center = _centroid(fit_frame, fit_weights)
                fit_centered = fit_frame - frame_center
                rmsd_centered = rmsd_frame - frame_center
                rotation = kabsch_rotations_to_reference(
                    mean_fit,
                    fit_centered[None, :, :],
                    fit_weights,
                )[0]
                fit_sum += fit_centered @ rotation
                rmsd_sum += rmsd_centered @ rotation

        new_mean_fit = fit_sum / total_frames
        new_mean_rmsd = rmsd_sum / total_frames
        # Keep the mean fit selection exactly centered to avoid numerical drift.
        mean_center = _centroid(new_mean_fit, fit_weights)
        new_mean_fit -= mean_center
        new_mean_rmsd -= mean_center
        final_shift = float(np.sqrt(np.mean(np.sum((new_mean_fit - mean_fit) ** 2, axis=1))))
        mean_fit = new_mean_fit
        mean_rmsd = new_mean_rmsd
        if final_shift <= tolerance_angstrom:
            return CentroidResult(mean_fit, mean_rmsd, iteration, final_shift)

    raise ValueError(
        f"Centroid calculation did not converge after {max_iterations} iterations "
        f"(last fit-coordinate shift {final_shift:.6g} Å)."
    )


def _format_atom_name(atom: AtomRecord) -> str:
    name = atom.key.atomname[:4]
    if len(name) >= 4:
        return name
    if len(atom.element.strip()) == 1:
        return f" {name:<3}"
    return f"{name:<4}"


def write_centroid_pdb(
    path: Path,
    atoms: list[AtomRecord],
    coords_angstrom: np.ndarray,
) -> None:
    """Write the synthetic centroid for the RMSD atom selection only."""

    coords = np.asarray(coords_angstrom, dtype=np.float64)
    if coords.shape != (len(atoms), 3):
        raise ValueError("Centroid coordinates do not match RMSD atom metadata.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("TITLE     CrossRMSD generalized-Procrustes centroid; RMSD selection only\n")
        for serial, (atom, xyz) in enumerate(zip(atoms, coords), start=1):
            chain = atom.key.chain[:1] if atom.key.chain else " "
            handle.write(
                f"ATOM  {serial:5d} {_format_atom_name(atom)} {atom.resname[:3]:>3s} "
                f"{chain}{atom.key.resid:4d}    "
                f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}"
                f"  1.00  0.00          {atom.element[:2]:>2s}\n"
            )
        handle.write("END\n")
