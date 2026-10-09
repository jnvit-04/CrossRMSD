"""Vectorized Kabsch alignment and Cross-RMSD calculation."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from crossrmsd.models import RmsdStats

ProgressCallback = Callable[[int, int, int, int], None]


def _validate_weights(weights: np.ndarray | None, n_atoms: int, name: str) -> None:
    if weights is None:
        return
    if weights.shape != (n_atoms,):
        raise ValueError(f"{name} weights must contain one value per selected atom.")
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError(f"{name} weights must be finite and positive.")


def centered_on_fit_centroid(
    calc_coords: np.ndarray,
    fit_coords: np.ndarray,
    fit_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Subtract each frame's fit-group centroid from both coordinate sets."""

    if fit_weights is None:
        centroid = fit_coords.mean(axis=1, keepdims=True)
    else:
        centroid = np.average(fit_coords, axis=1, weights=fit_weights)[:, None, :]
    return calc_coords - centroid, fit_coords - centroid


def kabsch_rotations_to_reference(
    reference_fit: np.ndarray,
    target_fit: np.ndarray,
    fit_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Return proper rotations aligning target fit atoms to one reference."""

    if fit_weights is None:
        covariance = np.einsum(
            "bki,kj->bij", target_fit, reference_fit, optimize=True
        )
    else:
        covariance = np.einsum(
            "k,bki,kj->bij",
            fit_weights,
            target_fit,
            reference_fit,
            optimize=True,
        )

    left, _singular_values, right_t = np.linalg.svd(covariance)
    rotations = np.matmul(left, right_t)
    reflected = np.linalg.det(rotations) < 0.0
    if np.any(reflected):
        left = left.copy()
        left[reflected, :, -1] *= -1.0
        rotations = np.matmul(left, right_t)
    return rotations


def expected_pair_count(n_reference: int, n_target: int, same_object: bool) -> int:
    """Return the number of RMSD values produced for a comparison."""

    if same_object:
        return n_reference * (n_reference - 1) // 2
    return n_reference * n_target


def pairwise_rmsd_values(
    reference_fit: np.ndarray,
    target_fit: np.ndarray,
    reference_calc: np.ndarray,
    target_calc: np.ndarray,
    chunk_size: int,
    same_object: bool = False,
    progress: ProgressCallback | None = None,
    fit_weights: np.ndarray | None = None,
    rmsd_weights: np.ndarray | None = None,
    ncpu: int = 1,
) -> np.ndarray:
    """Compute all optimal-fit RMSDs between two frame sets.

    Output order is reference-major. For an inter-trajectory comparison, values
    are ordered ``A1-B1, A1-B2, ...``. For an intra-trajectory comparison, only
    pairs with target index greater than reference index are emitted.
    """

    if reference_fit.shape[1:] != target_fit.shape[1:]:
        raise ValueError("Fit coordinate arrays must have matching atom dimensions.")
    if reference_calc.shape[1:] != target_calc.shape[1:]:
        raise ValueError("RMSD coordinate arrays must have matching atom dimensions.")
    if reference_fit.shape[1] < 3:
        raise ValueError("At least three fit atoms are required for Kabsch alignment.")
    if reference_calc.shape[1] < 1:
        raise ValueError("At least one RMSD atom is required.")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive.")
    if ncpu < 1:
        raise ValueError("ncpu must be positive.")
    if same_object and reference_fit.shape[0] != target_fit.shape[0]:
        raise ValueError("An intra-trajectory comparison requires the same frame set.")

    _validate_weights(fit_weights, reference_fit.shape[1], "Fit")
    _validate_weights(rmsd_weights, reference_calc.shape[1], "RMSD")

    ref_calc_centered, ref_fit_centered = centered_on_fit_centroid(
        reference_calc, reference_fit, fit_weights
    )
    tgt_calc_centered, tgt_fit_centered = centered_on_fit_centroid(
        target_calc, target_fit, fit_weights
    )

    total = expected_pair_count(
        ref_fit_centered.shape[0], tgt_fit_centered.shape[0], same_object
    )
    processed = 0
    chunks: list[np.ndarray] = []
    n_reference = ref_fit_centered.shape[0]

    def calculate_row(ref_index: int) -> np.ndarray:
        start = ref_index + 1 if same_object else 0

        if start >= tgt_fit_centered.shape[0]:
            return np.empty(0, dtype=np.float64)

        one_ref_fit = ref_fit_centered[ref_index]
        one_ref_calc = ref_calc_centered[ref_index]
        row_chunks: list[np.ndarray] = []

        for chunk_start in range(start, tgt_fit_centered.shape[0], chunk_size):
            chunk_stop = min(chunk_start + chunk_size, tgt_fit_centered.shape[0])

            rotations = kabsch_rotations_to_reference(
                one_ref_fit,
                tgt_fit_centered[chunk_start:chunk_stop],
                fit_weights,
            )

            aligned = np.einsum(
                "bki,bij->bkj",
                tgt_calc_centered[chunk_start:chunk_stop],
                rotations,
                optimize=True,
            )

            delta_squared = np.sum((aligned - one_ref_calc) ** 2, axis=2)

            if rmsd_weights is None:
                mean_squared = np.mean(delta_squared, axis=1)
            else:
                mean_squared = np.average(
                    delta_squared, axis=1, weights=rmsd_weights
                )

            row_chunks.append(
                np.sqrt(np.maximum(mean_squared, 0.0))
            )

        return (
            np.concatenate(row_chunks)
            if len(row_chunks) > 1
            else row_chunks[0]
        )

    def collect_rows(rows) -> None:
        nonlocal processed

        for ref_index, row_values in enumerate(rows):
            if row_values.size:
                chunks.append(row_values)
                processed += row_values.size

                if progress is not None:
                    progress(
                        processed,
                        total,
                        ref_index + 1,
                        n_reference,
                    )

    if ncpu == 1 or n_reference < 2:
        collect_rows(map(calculate_row, range(n_reference)))
    else:
        with ThreadPoolExecutor(max_workers=ncpu) as pool:
            collect_rows(
                pool.map(calculate_row, range(n_reference))
            )

    if not chunks:
        return np.asarray([], dtype=np.float64)

    return np.concatenate(chunks)

def summarize(values: np.ndarray) -> RmsdStats:
    """Return the compact default statistics used by the MVP."""
    if values.size == 0:
        nan = float("nan")
        return RmsdStats(n_pairs=0, mean=nan, sd=nan, median=nan)
    sd = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    return RmsdStats(
        n_pairs=int(values.size),
        mean=float(np.mean(values)),
        sd=sd,
        median=float(np.median(values)),
    )

