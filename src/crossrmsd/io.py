"""Human-readable logging, terminal progress, and CSV outputs."""

from __future__ import annotations

import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from crossrmsd.models import AtomRecord, Trajectory


def format_duration(seconds: float) -> str:
    """Format a duration as HH:MM:SS."""

    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class RunLog:
    """Write readable messages to the terminal and, optionally, a log file."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self._handle = None
        if path is not None:
            path = path.expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            self.path = path
            self._handle = path.open("w", encoding="utf-8")
        self._progress_active = False
        self._progress_width = 0

    def _write_file(self, text: str) -> None:
        if self._handle is None:
            return
        self._handle.write(text + "\n")
        self._handle.flush()

    def _clear_terminal_progress(self) -> None:
        if self._progress_active and sys.stdout.isatty():
            print("\r" + " " * self._progress_width + "\r", end="", flush=True)
        self._progress_active = False
        self._progress_width = 0

    def write(self, text: str = "") -> None:
        self._clear_terminal_progress()
        print(text)
        self._write_file(text)

    def progress(self, text: str) -> None:
        """Show a changing progress line without filling an interactive terminal."""

        if sys.stdout.isatty():
            width = max(self._progress_width, len(text))
            print(f"\r{text:<{width}}", end="", flush=True)
            self._progress_active = True
            self._progress_width = width
        else:
            print(text)
            self._write_file(text)

    def finish_progress(self, text: str) -> None:
        self._clear_terminal_progress()
        print(text)
        self._write_file(text)

    def section(self, title: str) -> None:
        self.write()
        self.write(title.upper())
        self.write("-" * len(title))

    def close(self) -> None:
        self._clear_terminal_progress()
        if self._handle is not None:
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


def write_kde_csv(
    path: Path,
    comparison: str,
    grid: np.ndarray,
    density: np.ndarray,
    unit: str,
) -> None:
    """Write a smooth CrossRMSD distribution without creating an image."""

    value_name = "rmsd_nm" if unit == "nm" else "rmsd_angstrom"
    density_name = "density_per_nm" if unit == "nm" else "density_per_angstrom"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["comparison", value_name, density_name])
        for x, y in zip(grid, density):
            writer.writerow([comparison, f"{x:.8f}", f"{y:.12g}"])


def write_pairwise_csv(
    path: Path,
    trajectory_a: Trajectory,
    trajectory_b: Trajectory,
    values: np.ndarray,
    same_object: bool,
    unit: str,
) -> None:
    """Write the primary one-row-per-frame-pair RMSD output.

    Only source-frame indices are written. They are zero-based and refer directly
    to the original input trajectories before ``--stride`` subsampling.
    """

    value_name = "rmsd_nm" if unit == "nm" else "rmsd_angstrom"
    if same_object:
        header = ["source_frame_i", "source_frame_j", value_name]
    else:
        header = [
            f"source_frame_{trajectory_a.label}",
            f"source_frame_{trajectory_b.label}",
            value_name,
        ]

    path.parent.mkdir(parents=True, exist_ok=True)
    cursor = 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for frame_a in range(trajectory_a.n_frames):
            start = frame_a + 1 if same_object else 0
            for frame_b in range(start, trajectory_b.n_frames):
                writer.writerow(
                    [
                        int(trajectory_a.source_frame_indices[frame_a]),
                        int(trajectory_b.source_frame_indices[frame_b]),
                        f"{values[cursor]:.8f}",
                    ]
                )
                cursor += 1
    if cursor != values.size:
        raise ValueError("Internal error while writing pairwise RMSD values.")


def write_selected_atoms_csv(
    path: Path,
    trajectories: list[Trajectory],
    fit_indices: list[np.ndarray],
    rmsd_indices: list[np.ndarray],
) -> None:
    """Write the exact fit and RMSD selections for auditing."""

    path.parent.mkdir(parents=True, exist_ok=True)
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
