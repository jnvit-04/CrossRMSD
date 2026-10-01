"""NumPy-only kernel density estimate for RMSD distributions."""

from __future__ import annotations

import math

import numpy as np


def silverman_bandwidth(values: np.ndarray) -> float:
    """Choose a robust Gaussian-kernel bandwidth using Silverman's rule."""

    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        raise ValueError("Cannot estimate a KDE from an empty distribution.")

    if values.size > 1:
        sd = float(np.std(values, ddof=1))
        q25, q75 = np.percentile(values, [25, 75])
        robust_sd = float((q75 - q25) / 1.34)
        positive = [value for value in (sd, robust_sd) if value > 0.0]
        scale = min(positive) if positive else 0.0
    else:
        scale = 0.0

    if scale <= 0.0:
        scale = max(float(np.max(np.abs(values))), 1.0) * 1.0e-3
    return max(0.9 * scale * values.size ** (-0.2), np.finfo(float).eps)


def gaussian_kde_curve(
    values: np.ndarray,
    points: int = 512,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return a non-negative reflected Gaussian KDE.

    RMSD cannot be negative. Reflection at zero avoids assigning density to
    impossible negative values while keeping the implementation small.
    """

    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("Cannot estimate a KDE from an empty distribution.")
    if points < 16:
        raise ValueError("KDE point count must be at least 16.")

    bandwidth = silverman_bandwidth(values)
    upper = max(float(np.max(values)) + 4.0 * bandwidth, 4.0 * bandwidth)
    grid = np.linspace(0.0, upper, points, dtype=np.float64)
    density = np.zeros_like(grid)

    # Bound the temporary (chunk x grid) array to roughly 16 MB.
    chunk_size = max(1, 2_000_000 // points)
    normalizer = values.size * bandwidth * math.sqrt(2.0 * math.pi)
    for start in range(0, values.size, chunk_size):
        chunk = values[start : start + chunk_size, None]
        direct = (grid[None, :] - chunk) / bandwidth
        reflected = (grid[None, :] + chunk) / bandwidth
        density += np.exp(-0.5 * direct * direct).sum(axis=0)
        density += np.exp(-0.5 * reflected * reflected).sum(axis=0)
    density /= normalizer

    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    area = float(integrate(density, grid))
    if area > 0.0:
        density /= area
    return grid, density, bandwidth
