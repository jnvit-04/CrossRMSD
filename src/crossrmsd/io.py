"""Human-readable logging, terminal progress, and flat CSV outputs."""

from __future__ import annotations

import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from crossrmsd.models import AtomRecord, ComparisonResult, Trajectory


def format_duration(seconds: float) -> str:
    """Format a duration as HH:MM:SS."""

    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class RunLog:
    """Write readable messages to the terminal and ``crossrmsd.log``."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle = path.open("w", encoding="utf-8")
        self._progress_active = False
        self._progress_width = 0

    def _clear_terminal_progress(self) -> None:
        if self._progress_active and sys.stdout.isatty():
            print("\r" + " " * self._progress_width + "\r", end="", flush=True)
        self._progress_active = False
        self._progress_width = 0

    def write(self, text: str = "") -> None:
        self._clear_terminal_progress()
        print(text)
        self._handle.write(text + "\n")
        self._handle.flush()

    def progress(self, text: str) -> None:
        """Show a changing progress line without filling an interactive terminal."""

        if sys.stdout.isatty():
            width = max(self._progress_width, len(text))
            print(f"\r{text:<{width}}", end="", flush=True)
            self._progress_active = True
            self._progress_width = width
        else:
            print(text)
            self._handle.write(text + "\n")
            self._handle.flush()

    def finish_progress(self, text: str) -> None:
        self._clear_terminal_progress()
        print(text)
        self._handle.write(text + "\n")
        self._handle.flush()

    def section(self, title: str) -> None:
        self.write()
        self.write(title.upper())
        self.write("-" * len(title))

    def close(self) -> None:
        self._clear_terminal_progress()
        self._handle.close()

    def __enter__(self) -> "RunLog":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


class ReadProgressReporter:
    """Show source-frame reading progress in verbose mode."""

    def __init__(self, log: RunLog, label: str, enabled: bool) -> None:
        self.log = log
        self.label = label
        self.enabled = enabled
        self.started = time.perf_counter()
        self.last_update = 0.0
        self.last_source_index = -1
        self.last_retained = 0

    def __call__(self, source_index: int, retained: int) -> None:
        self.last_source_index = source_index
        self.last_retained = retained
        if not self.enabled:
            return
        now = time.perf_counter()
        if now - self.last_update < 0.5 and source_index > 0:
            return
        elapsed = now - self.started
        rate = (source_index + 1) / elapsed if elapsed > 0.0 else 0.0
        self.log.progress(
            f"  Reading {self.label}: source frame {source_index:,} | "
            f"retained {retained:,} | {rate:,.1f} frames/s"
        )
        self.last_update = now

    def finish(self) -> None:
        if not self.enabled:
            return
        elapsed = time.perf_counter() - self.started
        self.log.finish_progress(
            f"  Read {self.label}: {self.last_source_index + 1:,} source frames, "
            f"retained {self.last_retained:,} in {format_duration(elapsed)}"
        )


class ProgressReporter:
    """Report either compact ten-percent or detailed live calculation progress."""

    def __init__(self, log: RunLog, comparison: str, verbose: bool) -> None:
        self.log = log
        self.comparison = comparison
        self.verbose = verbose
        self.next_percent = 10
        self.started = time.perf_counter()
        self.last_update = 0.0

    def __call__(
        self,
        completed: int,
        total: int,
        reference_index: int,
        n_reference: int,
    ) -> None:
        if total <= 0:
            return

        if not self.verbose:
            percent = int(completed * 100 / total)
            if percent < self.next_percent:
                return
            displayed = min((percent // 10) * 10, 100)
            self.log.write(
                f"  {displayed:3d}%  "
                f"({min(completed, total):,} / {total:,} frame pairs)"
            )
            self.next_percent = displayed + 10
            return

        now = time.perf_counter()
        if completed < total and now - self.last_update < 0.5:
            return
        elapsed = now - self.started
        rate = completed / elapsed if elapsed > 0.0 else 0.0
        remaining = (total - completed) / rate if rate > 0.0 else 0.0
        text = (
            f"  {self.comparison}: reference frame {reference_index:,}/{n_reference:,} | "
            f"{completed:,}/{total:,} pairs ({completed * 100 / total:5.1f}%) | "
            f"{rate:,.0f} pairs/s | ETA {format_duration(remaining)}"
        )
        if completed >= total:
            self.log.finish_progress(text)
        else:
            self.log.progress(text)
        self.last_update = now


def write_comparisons_csv(
    path: Path,
    results: list[ComparisonResult],
    unit: str,
) -> None:
    """Write one compact machine-readable row per comparison."""

    suffix = "nm" if unit == "nm" else "angstrom"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "comparison",
                "comparison_type",
                "trajectory_a",
                "trajectory_a_file",
                "trajectory_b",
                "trajectory_b_file",
                "frames_a",
                "frames_b",
                "frame_pairs",
                f"mean_{suffix}",
                f"standard_deviation_{suffix}",
                f"median_{suffix}",
                f"kde_bandwidth_{suffix}",
                "kde_file",
                "pairs_file",
                "xpm_file",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result.name,
                    result.kind,
                    result.label_a,
                    str(result.path_a),
                    result.label_b,
                    str(result.path_b),
                    result.frames_a,
                    result.frames_b,
                    result.stats.n_pairs,
                    f"{result.stats.mean:.8f}",
                    f"{result.stats.sd:.8f}",
                    f"{result.stats.median:.8f}",
                    f"{result.bandwidth:.8f}",
                    result.kde_file,
                    result.pairs_file,
                    result.xpm_file,
                ]
            )


def write_kde_csv(
    path: Path,
    comparison: str,
    grid: np.ndarray,
    density: np.ndarray,
    unit: str,
) -> None:
    """Write a smooth Cross-RMSD distribution without creating an image."""

    value_name = "rmsd_nm" if unit == "nm" else "rmsd_angstrom"
    density_name = "density_per_nm" if unit == "nm" else "density_per_angstrom"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["comparison", value_name, density_name])
        for x, y in zip(grid, density):
            writer.writerow([comparison, f"{x:.8f}", f"{y:.12g}"])


def write_pairs_csv(
    path: Path,
    comparison: str,
    trajectory_a: Trajectory,
    trajectory_b: Trajectory,
    values: np.ndarray,
    same_object: bool,
    unit: str,
) -> None:
    """Write optional one-row-per-frame-pair RMSD values."""

    value_name = "rmsd_nm" if unit == "nm" else "rmsd_angstrom"
    cursor = 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "comparison",
                "trajectory_a",
                "retained_frame_a_1based",
                "source_frame_a_0based",
                "trajectory_b",
                "retained_frame_b_1based",
                "source_frame_b_0based",
                value_name,
            ]
        )
        for frame_a in range(trajectory_a.n_frames):
            start = frame_a + 1 if same_object else 0
            for frame_b in range(start, trajectory_b.n_frames):
                writer.writerow(
                    [
                        comparison,
                        trajectory_a.label,
                        frame_a + 1,
                        int(trajectory_a.source_frame_indices[frame_a]),
                        trajectory_b.label,
                        frame_b + 1,
                        int(trajectory_b.source_frame_indices[frame_b]),
                        f"{values[cursor]:.8f}",
                    ]
                )
                cursor += 1
    if cursor != values.size:
        raise ValueError("Internal error while writing frame-pair RMSD values.")


def write_selected_atoms_csv(
    path: Path,
    trajectories: list[Trajectory],
    fit_indices: list[np.ndarray],
    rmsd_indices: list[np.ndarray],
) -> None:
    """Write optional exact fit and RMSD selections for auditing."""

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "selection",
                "trajectory",
                "atom_index_1based",
                "chain",
                "resid",
                "resname_in_first_frame",
                "atom_name",
                "element",
            ]
        )
        for trajectory, fit, rmsd in zip(trajectories, fit_indices, rmsd_indices):
            for selection_name, indices in (("fit", fit), ("rmsd", rmsd)):
                for index in indices:
                    atom: AtomRecord = trajectory.atoms[int(index)]
                    writer.writerow(
                        [
                            selection_name,
                            trajectory.label,
                            atom.index_1based,
                            atom.key.chain,
                            atom.key.resid,
                            atom.resname,
                            atom.key.atomname,
                            atom.element,
                        ]
                    )


def timestamp() -> str:
    """Return a readable local timestamp for logs."""

    return datetime.now().astimezone().isoformat(timespec="seconds")


def write_central_structure_csv(
    path: Path,
    trajectory: Trajectory,
    retained_frame_index: int,
    sum_squared_rmsd: float,
    dispersion_squared: float,
    structures_compared: int,
    unit: str,
    pdb_file: str,
) -> None:
    """Write the exact Campos-Baptista central-frame result."""

    suffix = "nm" if unit == "nm" else "angstrom"
    time_ps = ""
    if trajectory.frame_times_ps is not None:
        time_ps = f"{trajectory.frame_times_ps[retained_frame_index]:.8g}"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "method",
                "trajectory",
                "trajectory_file",
                "retained_frame_1based",
                "source_frame_0based",
                "time_ps",
                f"sum_squared_rmsd_{suffix}2",
                f"dispersion_D2_{suffix}2",
                f"dispersion_D_{suffix}",
                "structures_compared",
                "pdb_file",
            ]
        )
        writer.writerow(
            [
                "Campos-Baptista minimum mean squared pairwise RMSD dispersion",
                trajectory.label,
                str(trajectory.path),
                retained_frame_index + 1,
                int(trajectory.source_frame_indices[retained_frame_index]),
                time_ps,
                f"{sum_squared_rmsd:.10g}",
                f"{dispersion_squared:.10g}",
                f"{dispersion_squared ** 0.5:.10g}",
                structures_compared,
                pdb_file,
            ]
        )
