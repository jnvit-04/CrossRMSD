"""Strict multi-model PDB readers used by CrossRMSD."""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Collection
from pathlib import Path

import numpy as np

from crossrmsd.models import AtomKey, AtomRecord, LoadedTrajectory, Trajectory
from crossrmsd.selection import resolve_selection

ReadProgressCallback = Callable[[int, int], None]
_KeyTuple = tuple[str, int, str]
_TIME_RE = re.compile(r"(?:^|\s)t\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)")


def guess_element(atomname_field: str, known_elements: Collection[str] | None = None) -> str:
    """Guess an element from the four-character PDB atom-name field.

    PDB alignment distinguishes protein `` CA `` (carbon) from an ion named
    ``CA  `` (calcium). The element column remains authoritative when present.
    """

    field = atomname_field[:4]
    name = field.strip()
    while name and name[0].isdigit():
        name = name[1:]
    if not name:
        return ""
    upper = name.upper()
    known = {value.upper() for value in known_elements} if known_elements else set()
    if field and not field[0].isspace() and len(upper) >= 2 and upper[:2] in known:
        return upper[:2]
    return upper[0]


def _parse_title_time(line: str) -> float | None:
    match = _TIME_RE.search(line)
    return float(match.group(1)) if match else None


def _validate_atom_record(line: str, path: Path, line_number: int) -> None:
    if len(line.rstrip("\n")) < 54:
        raise ValueError(f"Malformed PDB atom record at {path}:{line_number}")

    altloc = line[16:17].strip()
    if altloc:
        raise ValueError(
            f"Alternate-location code '{altloc}' is not supported at "
            f"{path}:{line_number}. Normalize the PDB before running CrossRMSD."
        )

    insertion_code = line[26:27].strip()
    if insertion_code:
        raise ValueError(
            f"Insertion code '{insertion_code}' is not supported at "
            f"{path}:{line_number}. Normalize residue numbering first."
        )


def _parse_identity(line: str, path: Path, line_number: int) -> _KeyTuple:
    _validate_atom_record(line, path, line_number)
    atomname = line[12:16].strip()
    chain = line[21:22].strip()
    try:
        resid = int(line[22:26])
    except ValueError as exc:
        raise ValueError(f"Could not parse PDB atom record at {path}:{line_number}") from exc
    if not atomname:
        raise ValueError(f"Missing atom name at {path}:{line_number}")
    return chain, resid, atomname


def _parse_coordinates(
    line: str, path: Path, line_number: int
) -> tuple[float, float, float]:
    try:
        return float(line[30:38]), float(line[38:46]), float(line[46:54])
    except ValueError as exc:
        raise ValueError(f"Could not parse PDB coordinates at {path}:{line_number}") from exc


def _parse_atom_line(
    line: str,
    path: Path,
    line_number: int,
    known_elements: Collection[str] | None,
) -> tuple[AtomRecord, tuple[float, float, float]]:
    chain, resid, atomname = _parse_identity(line, path, line_number)
    resname = line[17:20].strip()
    element = line[76:78].strip().upper() if len(line) >= 78 else ""
    if not element:
        element = guess_element(line[12:16], known_elements)

    record = AtomRecord(
        key=AtomKey(chain=chain, resid=resid, atomname=atomname),
        resname=resname,
        index_1based=0,
        element=element,
    )
    return record, _parse_coordinates(line, path, line_number)


def _normalize_first_model(
    path: Path,
    model_atoms: list[AtomRecord],
    model_coords: list[tuple[float, float, float]],
) -> tuple[tuple[AtomRecord, ...], np.ndarray]:
    if not model_atoms:
        raise ValueError(f"Empty retained MODEL found in {path}")

    seen: dict[AtomKey, int] = {}
    normalized: list[AtomRecord] = []
    for position, atom in enumerate(model_atoms, start=1):
        if atom.key in seen:
            raise ValueError(
                "Duplicate atom identity in PDB model: "
                f"{atom.key.display()} (positions {seen[atom.key]} and {position})"
            )
        seen[atom.key] = position
        normalized.append(
            AtomRecord(
                key=atom.key,
                resname=atom.resname,
                index_1based=position,
                element=atom.element,
            )
        )

    return tuple(normalized), np.asarray(model_coords, dtype=np.float64)


def _selected_model_error(
    expected: tuple[_KeyTuple, ...], current: list[_KeyTuple]
) -> ValueError:
    expected_set = set(expected)
    current_set = set(current)
    missing = sorted(expected_set - current_set)
    duplicates = sorted({key for key in current if current.count(key) > 1})
    details = ["Atom identities differ between retained PDB models for the selected set."]
    if missing:
        details.append(
            "Missing: " + "; ".join(AtomKey(*key).display() for key in missing[:20])
        )
    if duplicates:
        details.append(
            "Duplicate: "
            + "; ".join(AtomKey(*key).display() for key in duplicates[:20])
        )
    return ValueError("\n".join(details))


def read_selected_pdb_trajectory(
    path: Path,
    label: str,
    fit_expression: str,
    rmsd_expression: str,
    groups: dict[str, set[int]] | None = None,
    stride: int = 1,
    progress: ReadProgressCallback | None = None,
    strict_full_model: bool = False,
    known_elements: Collection[str] | None = None,
) -> LoadedTrajectory:
    """Read only atoms needed by the fit and RMSD selections."""

    if stride < 1:
        raise ValueError("stride must be positive.")

    started = time.perf_counter()
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Trajectory does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Trajectory is not a file: {path}")

    frames: list[np.ndarray] = []
    retained_indices: list[int] = []
    retained_times: list[float | None] = []

    first_full_atoms: tuple[AtomRecord, ...] | None = None
    selected_atoms: tuple[AtomRecord, ...] | None = None
    expected_keys: tuple[_KeyTuple, ...] | None = None
    expected_key_set: set[_KeyTuple] = set()
    fit_local = np.empty(0, dtype=int)
    rmsd_local = np.empty(0, dtype=int)

    first_model_atoms: list[AtomRecord] = []
    first_model_coords: list[tuple[float, float, float]] = []
    selected_model_keys: list[_KeyTuple] = []
    selected_model_coords: list[tuple[float, float, float]] = []

    saw_model = False
    inside_model = False
    keep_model = True
    current_source_index = 0
    source_frame_count = 0
    same_order_frames = 0
    remapped_frames = 0
    pending_time_ps: float | None = None
    current_time_ps: float | None = None

    def finalize_retained_model() -> None:
        nonlocal first_full_atoms
        nonlocal selected_atoms
        nonlocal expected_keys
        nonlocal expected_key_set
        nonlocal fit_local
        nonlocal rmsd_local
        nonlocal same_order_frames
        nonlocal remapped_frames

        if first_full_atoms is None:
            first_full_atoms, full_coords = _normalize_first_model(
                path, first_model_atoms, first_model_coords
            )
            raw_fit = resolve_selection(fit_expression, first_full_atoms, groups or {})
            raw_rmsd = resolve_selection(rmsd_expression, first_full_atoms, groups or {})
            if raw_fit.size == 0:
                raise ValueError("The fit selection matched no atoms.")
            if raw_rmsd.size == 0:
                raise ValueError("The RMSD selection matched no atoms.")

            union_full = np.asarray(
                sorted(set(raw_fit.tolist()) | set(raw_rmsd.tolist())), dtype=int
            )
            full_to_local = {
                int(full_index): local_index
                for local_index, full_index in enumerate(union_full)
            }
            fit_local = np.asarray(
                [full_to_local[int(index)] for index in raw_fit], dtype=int
            )
            rmsd_local = np.asarray(
                [full_to_local[int(index)] for index in raw_rmsd], dtype=int
            )
            selected_atoms = tuple(first_full_atoms[int(index)] for index in union_full)
            expected_keys = tuple(
                (atom.key.chain, atom.key.resid, atom.key.atomname)
                for atom in selected_atoms
            )
            expected_key_set = set(expected_keys)
            frames.append(full_coords[union_full])
            return

        if expected_keys is None:
            raise AssertionError("Selected atom identities were not initialized.")
        if len(selected_model_keys) != len(expected_keys):
            raise _selected_model_error(expected_keys, selected_model_keys)

        if tuple(selected_model_keys) == expected_keys:
            frames.append(np.asarray(selected_model_coords, dtype=np.float64))
            same_order_frames += 1
            return

        current_index: dict[_KeyTuple, int] = {}
        for index, key in enumerate(selected_model_keys):
            if key in current_index:
                raise _selected_model_error(expected_keys, selected_model_keys)
            current_index[key] = index
        if set(current_index) != expected_key_set:
            raise _selected_model_error(expected_keys, selected_model_keys)

        reorder = np.asarray([current_index[key] for key in expected_keys], dtype=int)
        coords = np.asarray(selected_model_coords, dtype=np.float64)
        frames.append(coords[reorder])
        remapped_frames += 1

    def clear_model_buffers() -> None:
        first_model_atoms.clear()
        first_model_coords.clear()
        selected_model_keys.clear()
        selected_model_coords.clear()

    with path.open(errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            record_name = line[0:6].strip().upper()

            if record_name == "TITLE":
                parsed_time = _parse_title_time(line)
                if parsed_time is not None:
                    if inside_model:
                        current_time_ps = parsed_time
                    else:
                        pending_time_ps = parsed_time
                continue

            if record_name == "MODEL":
                if inside_model:
                    raise ValueError(f"Nested MODEL record at {path}:{line_number}")
                if first_model_atoms or selected_model_keys:
                    raise ValueError(
                        f"Atom records occur before the first MODEL in {path}; "
                        "use a consistent multi-model PDB."
                    )
                saw_model = True
                inside_model = True
                current_source_index = source_frame_count
                source_frame_count += 1
                keep_model = current_source_index % stride == 0
                current_time_ps = pending_time_ps
                pending_time_ps = None
                continue

            if record_name == "ENDMDL":
                if not inside_model:
                    raise ValueError(f"ENDMDL without MODEL at {path}:{line_number}")
                if keep_model:
                    finalize_retained_model()
                    retained_indices.append(current_source_index)
                    retained_times.append(current_time_ps)
                clear_model_buffers()
                inside_model = False
                current_time_ps = None
                if progress is not None:
                    progress(current_source_index, len(frames))
                continue

            if record_name not in {"ATOM", "HETATM"}:
                continue
            if saw_model and not inside_model:
                raise ValueError(
                    f"Atom record outside MODEL/ENDMDL block at {path}:{line_number}"
                )
            if saw_model and not keep_model:
                continue

            if first_full_atoms is None:
                atom, coord = _parse_atom_line(
                    line, path, line_number, known_elements
                )
                first_model_atoms.append(atom)
                first_model_coords.append(coord)
                continue

            key = _parse_identity(line, path, line_number)
            if key in expected_key_set or strict_full_model:
                selected_model_keys.append(key)
                selected_model_coords.append(_parse_coordinates(line, path, line_number))

    if inside_model:
        if keep_model:
            finalize_retained_model()
            retained_indices.append(current_source_index)
            retained_times.append(current_time_ps)
        if progress is not None:
            progress(current_source_index, len(frames))
    elif not saw_model and first_model_atoms:
        source_frame_count = 1
        finalize_retained_model()
        retained_indices.append(0)
        retained_times.append(pending_time_ps)
        if progress is not None:
            progress(0, 1)
    elif first_model_atoms or selected_model_keys:
        raise ValueError(f"Unclosed or inconsistent PDB model structure in {path}")

    if first_full_atoms is None or selected_atoms is None or not frames:
        raise ValueError(f"No retained ATOM or HETATM coordinates found in {path}")

    frame_times_ps: np.ndarray | None = None
    if retained_times and all(value is not None for value in retained_times):
        frame_times_ps = np.asarray(retained_times, dtype=np.float64)

    trajectory = Trajectory(
        label=label,
        path=path,
        atoms=selected_atoms,
        coords_angstrom=np.stack(frames, axis=0),
        source_frame_indices=np.asarray(retained_indices, dtype=np.int64),
        frame_times_ps=frame_times_ps,
    )
    return LoadedTrajectory(
        trajectory=trajectory,
        fit_indices=fit_local,
        rmsd_indices=rmsd_local,
        source_frame_count=source_frame_count,
        first_model_atom_count=len(first_full_atoms),
        same_order_frames=same_order_frames,
        remapped_frames=remapped_frames,
        read_seconds=time.perf_counter() - started,
    )


def read_pdb_trajectory(
    path: Path,
    label: str,
    stride: int = 1,
    progress: ReadProgressCallback | None = None,
    known_elements: Collection[str] | None = None,
) -> Trajectory:
    loaded = read_selected_pdb_trajectory(
        path=path,
        label=label,
        fit_expression="all",
        rmsd_expression="all",
        stride=stride,
        progress=progress,
        strict_full_model=True,
        known_elements=known_elements,
    )
    return loaded.trajectory


def extract_pdb_model(path: Path, source_frame_index: int, output_path: Path) -> None:
    """Extract one complete source PDB model as a standalone structure."""

    if source_frame_index < 0:
        raise ValueError("source_frame_index cannot be negative.")

    path = path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    global_cryst1: str | None = None
    saw_model = False
    inside_target = False
    current_index = -1
    selected_lines: list[str] = []
    single_model_lines: list[str] = []

    with path.open(errors="replace") as handle:
        for line in handle:
            record_name = line[0:6].strip().upper()
            if record_name == "CRYST1" and not saw_model:
                global_cryst1 = line
            if record_name == "MODEL":
                saw_model = True
                current_index += 1
                inside_target = current_index == source_frame_index
                continue
            if record_name == "ENDMDL":
                if inside_target:
                    break
                inside_target = False
                continue
            if saw_model:
                if inside_target:
                    selected_lines.append(line)
            else:
                single_model_lines.append(line)

    if not saw_model:
        if source_frame_index != 0:
            raise ValueError(f"PDB contains only one structure: {path}")
        selected_lines = single_model_lines
    elif not selected_lines:
        raise ValueError(
            f"Source frame {source_frame_index} was not found in trajectory: {path}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(
            f"TITLE     CrossRMSD central structure; source frame {source_frame_index}\n"
        )
        if global_cryst1 is not None and not any(
            line.startswith("CRYST1") for line in selected_lines
        ):
            handle.write(global_cryst1)
        for line in selected_lines:
            record_name = line[0:6].strip().upper()
            if record_name in {"END", "ENDMDL", "MODEL"}:
                continue
            handle.write(line if line.endswith("\n") else line + "\n")
        handle.write("END\n")
