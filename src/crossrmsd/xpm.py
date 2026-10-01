"""GROMACS-compatible RMSD XPM matrix output."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from crossrmsd.models import Trajectory

# GROMACS 2024.3 src/gromacs/fileio/matio.cpp mapper, in the same order.
_MAPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*()-_=+{}|;:',<.>/?"


def _c_g(value: float) -> str:
    """Approximate C printf("%g") formatting used by GROMACS."""

    return format(float(value), ".6g")


def _c_3g(value: float) -> str:
    """Approximate C printf("%.3g") formatting used by GROMACS."""

    return format(float(value), ".3g")


def _round_positive(value: np.ndarray) -> np.ndarray:
    """Match std::round/roundToInt for non-negative values."""

    return np.floor(value + 0.5).astype(np.int64)


def _axis_values(trajectory: Trajectory) -> tuple[np.ndarray, str]:
    if trajectory.frame_times_ps is not None:
        return trajectory.frame_times_ps, "Time (ps)"
    return trajectory.source_frame_indices.astype(np.float64), "Source frame"


def _write_axis(handle, axis_name: str, values: np.ndarray) -> None:
    for start in range(0, values.size, 80):
        chunk = values[start : start + 80]
        text = " ".join(_c_g(value) for value in chunk)
        handle.write(f"/* {axis_name}-axis:  {text} */\n")


def _codes(indices: np.ndarray, levels: int) -> str:
    if levels > len(_MAPPER):
        raise ValueError(
            f"CrossRMSD currently supports at most {len(_MAPPER)} XPM levels; "
            "GROMACS uses 80 by default."
        )
    return "".join(_MAPPER[int(index)] for index in indices)


def _quantize(values: np.ndarray, lo: float, hi: float, levels: int) -> np.ndarray:
    scaled = (np.asarray(values, dtype=np.float64) - lo) * ((levels - 1) / (hi - lo))
    indices = _round_positive(scaled)
    return np.clip(indices, 0, levels - 1)


def _intra_row(values: np.ndarray, n_frames: int, j: int, offsets: np.ndarray) -> np.ndarray:
    row = np.empty(n_frames, dtype=np.float64)
    row[j] = 0.0
    if j > 0:
        i = np.arange(j, dtype=np.int64)
        indices = offsets[:j] + (j - i - 1)
        row[:j] = values[indices]
    if j + 1 < n_frames:
        count = n_frames - j - 1
        start = int(offsets[j])
        row[j + 1 :] = values[start : start + count]
    return row


def write_gromacs_xpm(
    path: Path,
    values_nm: np.ndarray,
    trajectory_a: Trajectory,
    trajectory_b: Trajectory,
    same_object: bool,
    title: str,
    levels: int = 80,
    user_min_nm: float | None = None,
    user_max_nm: float | None = None,
) -> None:
    """Write an RMSD XPM using the GROMACS 2024.3 matrix encoding semantics.

    The colormap, quantization, matrix orientation, default 80 levels, white-to-
    black gradient, metadata layout and axis-row ordering follow ``gmx rms`` /
    ``write_xpm``. Unlike the GROMACS 2024.3 ``-skip`` axis bug, retained axis
    values are written directly rather than being strided a second time.
    """

    values_nm = np.asarray(values_nm, dtype=np.float64).reshape(-1)
    if levels < 2:
        raise ValueError("--xpm-levels must be at least 2.")
    if levels > len(_MAPPER):
        raise ValueError(
            f"--xpm-levels cannot exceed {len(_MAPPER)} in this implementation."
        )

    n_x = trajectory_a.n_frames
    n_y = trajectory_b.n_frames
    expected = n_x * (n_x - 1) // 2 if same_object else n_x * n_y
    if values_nm.size != expected:
        raise ValueError("Internal error: XPM pair count does not match matrix dimensions.")

    if same_object:
        lo = 0.0
    else:
        lo = float(np.min(values_nm))
    hi = float(np.max(values_nm))
    if user_min_nm is not None:
        lo = float(user_min_nm)
    if user_max_nm is not None:
        hi = float(user_max_nm)
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        raise ValueError(
            f"XPM requires a finite range with max > min; received min={lo}, max={hi}."
        )

    axis_x, label_x = _axis_values(trajectory_a)
    axis_y, label_y = _axis_values(trajectory_b)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as handle:
        handle.write("/* XPM */\n")
        handle.write(
            "/* This file can be converted to EPS by the GROMACS program xpm2ps */\n"
        )
        handle.write(f'/* title:   "{title}" */\n')
        handle.write('/* legend:  "RMSD (nm)" */\n')
        handle.write(f'/* x-label: "{label_x}" */\n')
        handle.write(f'/* y-label: "{label_y}" */\n')
        handle.write('/* type:    "Continuous" */\n')
        handle.write("static char *gromacs_xpm[] = {\n")
        handle.write(f'"{n_x} {n_y}   {levels} 1",\n')

        invlevel = 1.0 / (levels - 1)
        for i in range(levels):
            gray = int(math.floor(255.0 * (1.0 - i * invlevel) + 0.5))
            level_value = ((levels - 1 - i) * lo + i * hi) * invlevel
            handle.write(
                f'"{_MAPPER[i]}  c #{gray:02X}{gray:02X}{gray:02X} " '
                f'/* "{_c_3g(level_value)}" */,\n'
            )

        _write_axis(handle, "x", axis_x)
        _write_axis(handle, "y", axis_y)

        if same_object:
            offsets = np.asarray(
                [i * (2 * n_x - i - 1) // 2 for i in range(n_x)], dtype=np.int64
            )
            for j in range(n_y - 1, -1, -1):
                row = _intra_row(values_nm, n_x, j, offsets)
                encoded = _codes(_quantize(row, lo, hi, levels), levels)
                suffix = "," if j > 0 else ""
                handle.write(f'"{encoded}"{suffix}\n')
        else:
            matrix = values_nm.reshape(n_x, n_y)
            for j in range(n_y - 1, -1, -1):
                encoded = _codes(_quantize(matrix[:, j], lo, hi, levels), levels)
                suffix = "," if j > 0 else ""
                handle.write(f'"{encoded}"{suffix}\n')
