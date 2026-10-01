"""Campos-Baptista central-frame scoring."""

from __future__ import annotations

import numpy as np


def add_pairwise_squared_rmsd_sums(
    values: np.ndarray,
    sums_a: np.ndarray,
    sums_b: np.ndarray,
    same_object: bool,
) -> None:
    """Accumulate per-frame sums of squared pairwise RMSDs.

    The Campos-Baptista central structure minimizes
    ``sum_j RMSD(i,j)^2``. Division by the common ``n - 1`` factor does not
    change which sampled conformation is selected.
    """

    squared = np.square(np.asarray(values, dtype=np.float64).reshape(-1))
    if same_object:
        if sums_a is not sums_b and not np.shares_memory(sums_a, sums_b):
            raise ValueError("Intra-trajectory central scores must use the same array.")
        n_frames = sums_a.size
        if squared.size != n_frames * (n_frames - 1) // 2:
            raise ValueError("Unexpected number of intra-trajectory RMSD values.")
        cursor = 0
        for i in range(n_frames - 1):
            count = n_frames - i - 1
            row = squared[cursor : cursor + count]
            sums_a[i] += float(np.sum(row))
            sums_a[i + 1 :] += row
            cursor += count
        return

    n_a = sums_a.size
    n_b = sums_b.size
    if squared.size != n_a * n_b:
        raise ValueError("Unexpected number of inter-trajectory RMSD values.")
    matrix = squared.reshape(n_a, n_b)
    sums_a += matrix.sum(axis=1)
    sums_b += matrix.sum(axis=0)


def central_frame_index(sum_squared_rmsd: np.ndarray) -> int:
    """Return the sampled frame minimizing squared-RMSD dispersion."""

    scores = np.asarray(sum_squared_rmsd, dtype=np.float64)
    if scores.ndim != 1 or scores.size == 0:
        raise ValueError("Central-frame scores must be a non-empty 1D array.")
    if not np.all(np.isfinite(scores)):
        raise ValueError("Central-frame scores contain non-finite values.")
    return int(np.argmin(scores))


def dispersion(sum_squared_rmsd: float, structures_compared: int) -> float:
    """Return the Campos-Baptista mean squared RMSD dispersion D_i^2."""

    if structures_compared < 1:
        raise ValueError("At least one other structure is required for dispersion.")
    return float(sum_squared_rmsd) / structures_compared
