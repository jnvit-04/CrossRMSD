"""High-level orchestration for CrossRMSD."""

from __future__ import annotations

import itertools
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from crossrmsd.database import load_atomic_masses
from crossrmsd.centroid import generalized_procrustes_centroid, write_centroid_pdb
from crossrmsd.central import (
    add_pairwise_squared_rmsd_sums,
    central_frame_index,
    dispersion,
)
from crossrmsd.geometry import expected_pair_count, pairwise_rmsd_values, summarize
from crossrmsd.io import (
    ProgressReporter,
    ReadProgressReporter,
    RunLog,
    timestamp,
    write_kde_csv,
    write_pairwise_csv,
    write_selected_atoms_csv,
)
from crossrmsd.kde import gaussian_kde_curve
from crossrmsd.models import AtomRecord, ComparisonResult, LoadedTrajectory, Trajectory
from crossrmsd.ndx import read_ndx
from crossrmsd.pdb import extract_pdb_model, read_selected_pdb_trajectory
from crossrmsd.xpm import write_gromacs_xpm

_KDE_POINTS = 512


def trajectory_label(index: int) -> str:
    if index < 0:
        raise ValueError("Trajectory index cannot be negative.")
    label = ""
    value = index + 1
    while value:
        value, remainder = divmod(value - 1, 26)
        label = chr(ord("A") + remainder) + label
    return label


def _compress_ranges(values: list[int]) -> str:
    ordered = sorted(set(values))
    if not ordered:
        return "none"
    parts: list[str] = []
    start = previous = ordered[0]
    for value in ordered[1:]:
        if value == previous + 1:
            previous = value
            continue
        parts.append(f"{start}-{previous}" if start != previous else str(start))
        start = previous = value
    parts.append(f"{start}-{previous}" if start != previous else str(start))
    return ",".join(parts)


def _selection_description(atoms: list[AtomRecord]) -> list[str]:
    chains = sorted({atom.key.chain or "<blank>" for atom in atoms})
    atom_names = sorted({atom.key.atomname for atom in atoms})
    residues_by_chain: dict[str, list[int]] = {}
    for atom in atoms:
        chain = atom.key.chain or "<blank>"
        residues_by_chain.setdefault(chain, []).append(atom.key.resid)
    residue_text = "; ".join(
        f"{chain}: {_compress_ranges(resids)}"
        for chain, resids in sorted(residues_by_chain.items())
    )
    return [
        f"Atoms: {len(atoms):,}",
        f"Chains: {', '.join(chains)}",
        f"Atom names: {', '.join(atom_names)}",
        f"Residues: {residue_text}",
    ]


def _strict_canonical_indices(
    trajectories: list[Trajectory],
    raw_indices: list[np.ndarray],
    selection_name: str,
) -> tuple[list[np.ndarray], list[AtomRecord]]:
    reference_atoms = [trajectories[0].atoms[int(i)] for i in raw_indices[0]]
    reference_keys = [atom.key for atom in reference_atoms]
    reference_set = set(reference_keys)
    if not reference_keys:
        raise ValueError(f"The {selection_name} selection matched no atoms.")

    canonical_indices: list[np.ndarray] = []
    for trajectory, indices in zip(trajectories, raw_indices):
        selected_atoms = [trajectory.atoms[int(i)] for i in indices]
        selected_by_key = {atom.key: int(i) for atom, i in zip(selected_atoms, indices)}
        selected_set = set(selected_by_key)
        if selected_set != reference_set:
            missing = sorted(reference_set - selected_set)
            extra = sorted(selected_set - reference_set)
            details = [
                f"The {selection_name} atom set differs for trajectory {trajectory.label}."
            ]
            if missing:
                details.append("Missing: " + "; ".join(key.display() for key in missing[:20]))
            if extra:
                details.append("Extra: " + "; ".join(key.display() for key in extra[:20]))
            details.append("No calculation was performed.")
            raise ValueError("\n".join(details))

        ordered = np.asarray([selected_by_key[key] for key in reference_keys], dtype=int)
        canonical_indices.append(ordered)
        by_key = {atom.key: atom for atom in selected_atoms}
        for reference_atom in reference_atoms:
            other = by_key[reference_atom.key]
            if other.element != reference_atom.element:
                raise ValueError(
                    f"Element mismatch for {reference_atom.key.display()}: "
                    f"trajectory A says {reference_atom.element}, trajectory "
                    f"{trajectory.label} says {other.element}."
                )
    return canonical_indices, reference_atoms


def _atom_masses(
    atoms: list[AtomRecord], selection_name: str, mass_table: dict[str, float]
) -> np.ndarray:
    masses: list[float] = []
    unknown: list[str] = []
    for atom in atoms:
        element = atom.element.upper()
        mass = mass_table.get(element)
        if mass is None:
            unknown.append(f"{atom.key.display()} (element '{atom.element}')")
        else:
            masses.append(mass)
    if unknown:
        raise ValueError(
            f"Cannot mass-weight the {selection_name} selection because some elements "
            "have no mass in the configured database:\n" + "\n".join(unknown[:20])
        )
    return np.asarray(masses, dtype=np.float64)


def _read_trajectory_worker(
    path: Path,
    label: str,
    fit_expression: str,
    rmsd_expression: str,
    groups: dict[str, set[int]],
    stride: int,
    known_elements: tuple[str, ...],
) -> LoadedTrajectory:
    return read_selected_pdb_trajectory(
        path=path,
        label=label,
        fit_expression=fit_expression,
        rmsd_expression=rmsd_expression,
        groups=groups,
        stride=stride,
        known_elements=known_elements,
    )


def _load_and_resolve(
    trajectory_paths: list[Path],
    fit_expression: str,
    rmsd_expression: str,
    index_path: Path | None,
    stride: int,
    ncpu: int,
    verbose: bool,
    known_elements: tuple[str, ...],
    log: RunLog,
) -> tuple[
    list[Trajectory],
    list[np.ndarray],
    list[np.ndarray],
    list[AtomRecord],
    list[AtomRecord],
]:
    groups = read_ndx(index_path) if index_path is not None else {}
    if index_path is not None:
        log.section("Index file")
        log.write(f"File: {index_path.expanduser().resolve()}")
        log.write(f"Groups read: {len(groups):,}")

    labels = [trajectory_label(index) for index in range(len(trajectory_paths))]
    worker_count = min(ncpu, len(trajectory_paths))
    loaded: list[LoadedTrajectory | None] = [None] * len(trajectory_paths)

    log.section("Input trajectories")
    log.write(f"Maximum CPUs requested: {ncpu}")
    log.write(f"Trajectory reader workers: {worker_count}")
    log.write("Coordinate loading: fit/RMSD selected-atom union only")
    log.write("Unselected atom-set consistency: not required")

    reading_started = time.perf_counter()
    if worker_count == 1:
        for index, (path, label) in enumerate(zip(trajectory_paths, labels)):
            reporter = ReadProgressReporter(log, label, verbose)
            loaded[index] = read_selected_pdb_trajectory(
                path=path,
                label=label,
                fit_expression=fit_expression,
                rmsd_expression=rmsd_expression,
                groups=groups,
                stride=stride,
                progress=reporter,
                known_elements=known_elements,
            )
            reporter.finish()
    else:
        if verbose:
            log.write(
                "Reading trajectories concurrently; each trajectory reports when its worker finishes."
            )
        with ProcessPoolExecutor(
            max_workers=worker_count, mp_context=get_context("spawn")
        ) as executor:
            futures = {
                executor.submit(
                    _read_trajectory_worker,
                    path,
                    label,
                    fit_expression,
                    rmsd_expression,
                    groups,
                    stride,
                    known_elements,
                ): index
                for index, (path, label) in enumerate(zip(trajectory_paths, labels))
            }
            for future in as_completed(futures):
                index = futures[future]
                loaded[index] = future.result()
                if verbose:
                    item = loaded[index]
                    if item is None:
                        raise AssertionError("Trajectory reader returned no result.")
                    log.write(
                        f"  Read {item.trajectory.label}: {item.source_frame_count:,} "
                        f"source frames, retained {item.trajectory.n_frames:,} in "
                        f"{item.read_seconds:.2f} seconds"
                    )

    loaded_items = [item for item in loaded if item is not None]
    if len(loaded_items) != len(trajectory_paths):
        raise AssertionError("One or more trajectory readers returned no result.")
    trajectories = [item.trajectory for item in loaded_items]

    log.write(f"Parallel/serial reading wall time: {time.perf_counter() - reading_started:.2f} seconds")
    log.write()
    for item in loaded_items:
        trajectory = item.trajectory
        log.write(trajectory.label)
        log.write(f"  File: {trajectory.path}")
        log.write(f"  Source frames scanned: {item.source_frame_count:,}")
        log.write(f"  Retained frames: {trajectory.n_frames:,}")
        log.write(f"  Source-frame stride: {stride:,}")
        log.write(
            f"  Source frame range (0-based): {int(trajectory.source_frame_indices[0]):,} "
            f"to {int(trajectory.source_frame_indices[-1]):,}"
        )
        if trajectory.frame_times_ps is not None:
            log.write(
                f"  Retained time range: {trajectory.frame_times_ps[0]:g} to "
                f"{trajectory.frame_times_ps[-1]:g} ps"
            )
        log.write(f"  Atoms in first model: {item.first_model_atom_count:,}")
        log.write(f"  Selected union atoms stored: {trajectory.n_atoms:,}")
        log.write(
            f"  Selected coordinate memory: {trajectory.coords_angstrom.nbytes / (1024 ** 2):.2f} MiB"
        )
        log.write(f"  Same-selected-order frames after first: {item.same_order_frames:,}")
        log.write(f"  Identity-remapped frames: {item.remapped_frames:,}")
        log.write(f"  Reader time: {item.read_seconds:.2f} seconds")
        log.write()

    raw_fit = [item.fit_indices for item in loaded_items]
    raw_rmsd = [item.rmsd_indices for item in loaded_items]
    fit_indices, fit_atoms = _strict_canonical_indices(trajectories, raw_fit, "fit")
    rmsd_indices, rmsd_atoms = _strict_canonical_indices(trajectories, raw_rmsd, "RMSD")

    if len(fit_atoms) < 3:
        raise ValueError(
            f"The fit selection matched {len(fit_atoms)} atoms; at least 3 are required."
        )
    if not rmsd_atoms:
        raise ValueError("The RMSD selection matched no atoms.")

    log.section("Atom selections")
    log.write("Fit selection")
    log.write(f"  Expression: {fit_expression}")
    for line in _selection_description(fit_atoms):
        log.write(f"  {line}")
    log.write()
    log.write("RMSD selection")
    log.write(f"  Expression: {rmsd_expression}")
    for line in _selection_description(rmsd_atoms):
        log.write(f"  {line}")
    log.write()
    log.write("Selected atom mapping is valid across every retained trajectory frame.")
    return trajectories, fit_indices, rmsd_indices, fit_atoms, rmsd_atoms


def _comparison_plan(trajectories: list[Trajectory]) -> list[tuple[int, int, bool]]:
    if len(trajectories) == 1:
        if trajectories[0].n_frames < 2:
            raise ValueError(
                "An intra-trajectory Cross-RMSD calculation requires at least 2 retained frames."
            )
        return [(0, 0, True)]
    return [
        (left, right, False)
        for left, right in itertools.combinations(range(len(trajectories)), 2)
    ]


def _xpm_title(rmsd_expression: str) -> str:
    normalized = " ".join(rmsd_expression.strip().split()).upper()
    if normalized == "NAME CA":
        return "C-alpha RMSD matrix"
    return "CrossRMSD RMSD matrix"


def _comparison_output_path(base: Path, comparison: str, multiple: bool) -> Path:
    """Return the primary CSV path for one comparison."""

    base = base.expanduser()
    if not multiple:
        return base
    if base.suffix:
        return base.with_name(f"{base.stem}_{comparison}{base.suffix}")
    return base.with_name(f"{base.name}_{comparison}")


def _kde_output_path(primary: Path) -> Path:
    return primary.with_name(f"{primary.stem}_kde.csv")


def _xpm_output_path(primary: Path) -> Path:
    return primary.with_suffix(".xpm")


def _selection_output_path(base: Path) -> Path:
    base = base.expanduser()
    return base.with_name(f"{base.stem}_selection.csv")



def _validate_output_paths(items: list[tuple[str, Path]]) -> None:
    """Reject accidental collisions between requested and derived outputs."""

    seen: dict[str, str] = {}
    for description, path in items:
        key = str(path.expanduser().resolve(strict=False))
        previous = seen.get(key)
        if previous is not None:
            raise ValueError(
                f"Output path collision: {previous} and {description} both resolve to {path}."
            )
        seen[key] = description


def run_crossrmsd(
    trajectory_paths: list[Path],
    output_file: Path,
    fit_expression: str,
    rmsd_expression: str,
    index_path: Path | None,
    unit: str,
    chunk_size: int,
    stride: int,
    ncpu: int,
    mass_weighted: bool,
    mass_database: Path | None,
    verbose: bool,
    write_kde: bool,
    write_selection: bool,
    write_xpm: bool,
    xpm_levels: int,
    xpm_min_nm: float | None,
    xpm_max_nm: float | None,
    central_output: Path | None,
    centroid_output: Path | None,
    dry_run: bool,
    command: str,
    log: RunLog,
) -> list[ComparisonResult]:
    started = time.perf_counter()
    log.write("CrossRMSD")
    log.write(f"Started: {timestamp()}")
    log.write(f"Command: {command}")

    mass_table, mass_database_path = load_atomic_masses(mass_database)
    known_elements = tuple(sorted(mass_table))
    log.section("Databases")
    log.write(f"Atomic masses/elements: {mass_database_path}")
    log.write(f"Elements loaded: {len(mass_table):,}")

    trajectories, fit_indices, rmsd_indices, fit_atoms, rmsd_atoms = _load_and_resolve(
        trajectory_paths=trajectory_paths,
        fit_expression=fit_expression,
        rmsd_expression=rmsd_expression,
        index_path=index_path,
        stride=stride,
        ncpu=ncpu,
        verbose=verbose,
        known_elements=known_elements,
        log=log,
    )

    plan = _comparison_plan(trajectories)
    multiple_outputs = len(plan) > 1
    output_paths: dict[str, Path] = {}

    log.section("Comparison plan")
    log.write(
        "Mode: intra-trajectory Cross-RMSD"
        if len(trajectories) == 1
        else "Mode: inter-trajectory Cross-RMSD"
    )
    for left, right, same_object in plan:
        a = trajectories[left]
        b = trajectories[right]
        name = f"{a.label}_intra" if same_object else f"{a.label}_vs_{b.label}"
        primary_path = _comparison_output_path(output_file, name, multiple_outputs)
        output_paths[name] = primary_path
        log.write(
            f"{name}: {expected_pair_count(a.n_frames, b.n_frames, same_object):,} "
            f"frame pairs -> {primary_path}"
        )

    if central_output is not None and len(trajectories) > 1:
        log.write(
            "Central frame: pooled ensemble; auxiliary intra-trajectory RMSDs are included "
            "in the exact Campos-Baptista dispersion."
        )

    path_items: list[tuple[str, Path]] = [
        (f"primary output {name}", path) for name, path in output_paths.items()
    ]
    if write_kde:
        path_items.extend(
            (f"KDE output {name}", _kde_output_path(path))
            for name, path in output_paths.items()
        )
    if write_xpm:
        path_items.extend(
            (f"XPM output {name}", _xpm_output_path(path))
            for name, path in output_paths.items()
        )
    if write_selection:
        path_items.append(("selected-atom audit", _selection_output_path(output_file)))
    if central_output is not None:
        path_items.append(("central structure", central_output))
    if centroid_output is not None:
        path_items.append(("centroid structure", centroid_output))
    if log.path is not None:
        path_items.append(("run log", log.path))
    _validate_output_paths(path_items)

    if write_selection:
        selection_path = _selection_output_path(output_file)
        write_selected_atoms_csv(selection_path, trajectories, fit_indices, rmsd_indices)
        log.write(f"Selected-atom audit: {selection_path}")

    if dry_run:
        log.section("Dry run")
        log.write("Input parsing, selections, and atom mapping succeeded.")
        log.write("No RMSD calculations were performed.")
        log.write(f"Elapsed time: {time.perf_counter() - started:.2f} seconds")
        return []

    scale = 0.1 if unit == "nm" else 1.0
    unit_symbol = "nm" if unit == "nm" else "Å"
    fit_stacks = [
        trajectory.coords_angstrom[:, indices, :] * scale
        for trajectory, indices in zip(trajectories, fit_indices)
    ]
    rmsd_stacks = [
        trajectory.coords_angstrom[:, indices, :] * scale
        for trajectory, indices in zip(trajectories, rmsd_indices)
    ]

    fit_weights = _atom_masses(fit_atoms, "fit", mass_table) if mass_weighted else None
    rmsd_weights = _atom_masses(rmsd_atoms, "RMSD", mass_table) if mass_weighted else None

    log.section("Calculation settings")
    log.write(f"Output units: {unit_symbol}")
    if mass_weighted:
        log.write("RMSD definition: mass-weighted fit and RMSD")
        log.write(f"Mass source: {mass_database_path}")
    else:
        log.write("RMSD definition: unweighted geometric RMSD")
    log.write(f"Frame stride: {stride:,}")
    log.write("Primary CSV frame identifiers: zero-based source-frame indices")
    log.write(f"Maximum CPUs: {ncpu}")
    log.write(f"Numerical-library thread limit: {ncpu}")
    log.write(f"Chunk size: {chunk_size:,} target frames")
    log.write(f"Verbose progress: {'yes' if verbose else 'no'}")
    log.write(f"Write KDE CSV: {'yes' if write_kde else 'no'}")
    log.write(f"Write GROMACS-compatible XPM: {'yes' if write_xpm else 'no'}")
    if write_xpm:
        log.write(f"XPM levels: {xpm_levels}")
        log.write("XPM RMSD unit: nm (matching gmx rms)")
    log.write(f"Campos-Baptista central frame: {'yes' if central_output is not None else 'no'}")
    log.write(f"Generalized-Procrustes centroid: {'yes' if centroid_output is not None else 'no'}")
    if write_kde:
        log.write(f"KDE points: {_KDE_POINTS}")

    if centroid_output is not None:
        log.section("Generalized-Procrustes centroid")
        centroid_path = centroid_output.expanduser()
        centroid_result = generalized_procrustes_centroid(
            [
                trajectory.coords_angstrom[:, indices, :]
                for trajectory, indices in zip(trajectories, fit_indices)
            ],
            [
                trajectory.coords_angstrom[:, indices, :]
                for trajectory, indices in zip(trajectories, rmsd_indices)
            ],
            fit_weights=fit_weights,
        )
        write_centroid_pdb(
            centroid_path,
            rmsd_atoms,
            centroid_result.rmsd_coords_angstrom,
        )
        log.write("Definition: iterative generalized-Procrustes mean after fit-selection alignment.")
        log.write("Output PDB contains the RMSD selection atoms only; it is a synthetic structure.")
        log.write(f"Frames averaged: {sum(t.n_frames for t in trajectories):,}")
        log.write(f"Iterations: {centroid_result.iterations}")
        log.write(f"Final fit-coordinate shift: {centroid_result.final_shift_angstrom:.6g} Å")
        log.write(f"Structure: {centroid_path}")

    n_trajectories = len(trajectories)
    pooled_central_sums = (
        [np.zeros(t.n_frames, dtype=np.float64) for t in trajectories]
        if central_output is not None
        else None
    )

    results: list[ComparisonResult] = []
    for left, right, same_object in plan:
        a = trajectories[left]
        b = trajectories[right]
        comparison = f"{a.label}_intra" if same_object else f"{a.label}_vs_{b.label}"
        kind = "intra" if same_object else "inter"
        total_pairs = expected_pair_count(a.n_frames, b.n_frames, same_object)
        primary_path = output_paths[comparison]

        log.section(f"Calculation: {comparison}")
        log.write(f"Trajectory {a.label}: {a.path}")
        if not same_object:
            log.write(f"Trajectory {b.label}: {b.path}")
        log.write(f"Expected frame pairs: {total_pairs:,}")
        log.write("Building Cross-RMSD distribution...")

        calculation_started = time.perf_counter()
        values = pairwise_rmsd_values(
            reference_fit=fit_stacks[left],
            target_fit=fit_stacks[right],
            reference_calc=rmsd_stacks[left],
            target_calc=rmsd_stacks[right],
            chunk_size=chunk_size,
            same_object=same_object,
            progress=ProgressReporter(log, comparison, verbose),
            fit_weights=fit_weights,
            rmsd_weights=rmsd_weights,
        )
        rmsd_elapsed = time.perf_counter() - calculation_started
        log.write(f"RMSD calculations finished in {rmsd_elapsed:.2f} seconds.")

        if pooled_central_sums is not None:
            add_pairwise_squared_rmsd_sums(
                values,
                pooled_central_sums[left],
                pooled_central_sums[left] if same_object else pooled_central_sums[right],
                same_object,
            )

        post_started = time.perf_counter()
        stats = summarize(values)
        log.write(f"Writing {stats.n_pairs:,} pairwise RMSD rows to {primary_path}...")
        write_pairwise_csv(
            primary_path,
            trajectory_a=a,
            trajectory_b=b,
            values=values,
            same_object=same_object,
            unit=unit,
        )

        bandwidth: float | None = None
        kde_name = ""
        if write_kde:
            grid, density, bandwidth = gaussian_kde_curve(values, points=_KDE_POINTS)
            kde_path = _kde_output_path(primary_path)
            write_kde_csv(
                kde_path,
                comparison=comparison,
                grid=grid,
                density=density,
                unit=unit,
            )
            kde_name = str(kde_path)

        xpm_name = ""
        if write_xpm:
            xpm_path = _xpm_output_path(primary_path)
            values_nm = values if unit == "nm" else values * 0.1
            write_gromacs_xpm(
                xpm_path,
                values_nm=values_nm,
                trajectory_a=a,
                trajectory_b=b,
                same_object=same_object,
                title=_xpm_title(rmsd_expression),
                levels=xpm_levels,
                user_min_nm=xpm_min_nm,
                user_max_nm=xpm_max_nm,
            )
            xpm_name = str(xpm_path)

        result = ComparisonResult(
            name=comparison,
            kind=kind,
            label_a=a.label,
            label_b=b.label,
            path_a=a.path,
            path_b=b.path,
            frames_a=a.n_frames,
            frames_b=b.n_frames,
            stats=stats,
            output_file=str(primary_path),
            kde_file=kde_name,
            xpm_file=xpm_name,
            bandwidth=bandwidth,
        )
        results.append(result)

        log.write()
        log.write(f"Result: {comparison}")
        log.write(f"  Frame pairs: {stats.n_pairs:,}")
        log.write(f"  Mean RMSD: {stats.mean:.6f} {unit_symbol}")
        log.write(f"  Median RMSD: {stats.median:.6f} {unit_symbol}")
        log.write(f"  Standard deviation: {stats.sd:.6f} {unit_symbol}")
        log.write(f"  Pairwise data: {primary_path}")
        if bandwidth is not None:
            log.write(f"  KDE bandwidth: {bandwidth:.6f} {unit_symbol}")
            log.write(f"  KDE data: {kde_name}")
        if xpm_name:
            log.write(f"  GROMACS-compatible XPM: {xpm_name}")
        log.write(f"  RMSD calculation time: {rmsd_elapsed:.2f} seconds")
        log.write(f"  Statistics/output time: {time.perf_counter() - post_started:.2f} seconds")

    # Multi-trajectory normal mode does not calculate intra comparisons. Exact
    # pooled central scoring needs those within-trajectory contributions as well.
    need_auxiliary_intra = n_trajectories > 1 and central_output is not None
    if need_auxiliary_intra:
        log.section("Auxiliary intra-trajectory calculations")
        for index, trajectory in enumerate(trajectories):
            if trajectory.n_frames < 2:
                continue
            log.write(
                f"Calculating {trajectory.label}_intra for exact pooled central scoring..."
            )
            values = pairwise_rmsd_values(
                reference_fit=fit_stacks[index],
                target_fit=fit_stacks[index],
                reference_calc=rmsd_stacks[index],
                target_calc=rmsd_stacks[index],
                chunk_size=chunk_size,
                same_object=True,
                progress=ProgressReporter(log, f"{trajectory.label}_intra_aux", verbose),
                fit_weights=fit_weights,
                rmsd_weights=rmsd_weights,
            )
            if pooled_central_sums is not None:
                add_pairwise_squared_rmsd_sums(
                    values,
                    pooled_central_sums[index],
                    pooled_central_sums[index],
                    True,
                )

    if pooled_central_sums is not None:
        total_structures = sum(trajectory.n_frames for trajectory in trajectories)
        if total_structures < 2:
            raise ValueError("Central-frame calculation requires at least two structures.")
        best_trajectory_index = 0
        best_frame_index = 0
        best_sum_squared = float("inf")
        for trajectory_index, sums in enumerate(pooled_central_sums):
            frame_index = central_frame_index(sums)
            score = float(sums[frame_index])
            if score < best_sum_squared:
                best_sum_squared = score
                best_trajectory_index = trajectory_index
                best_frame_index = frame_index

        central_trajectory = trajectories[best_trajectory_index]
        d2 = dispersion(best_sum_squared, total_structures - 1)
        if central_output is None:
            raise AssertionError("Central output path was not configured.")
        central_path = central_output.expanduser()
        extract_pdb_model(
            central_trajectory.path,
            int(central_trajectory.source_frame_indices[best_frame_index]),
            central_path,
        )
        log.section("Campos-Baptista central frame")
        log.write(
            "Definition: sampled conformation minimizing D_i^2 = "
            "(1/(n-1)) * sum_j RMSD(i,j)^2."
        )
        log.write(f"Trajectory: {central_trajectory.label}")
        log.write(f"Retained frame (1-based): {best_frame_index + 1:,}")
        log.write(
            f"Source frame (0-based): "
            f"{int(central_trajectory.source_frame_indices[best_frame_index]):,}"
        )
        log.write(f"D^2 dispersion: {d2:.8g} {unit_symbol}^2")
        log.write(f"D dispersion: {d2 ** 0.5:.8g} {unit_symbol}")
        log.write(f"Structure: {central_path}")

    log.section("Finished")
    for result in results:
        log.write(f"Primary output ({result.name}): {result.output_file}")
    log.write("No raster images were generated.")
    log.write(f"Completed: {timestamp()}")
    log.write(f"Total elapsed time: {time.perf_counter() - started:.2f} seconds")
    return results
