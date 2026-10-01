"""Minimal reader for optional GROMACS index groups."""

from __future__ import annotations

from pathlib import Path


def read_ndx(path: Path) -> dict[str, set[int]]:
    """Read a GROMACS ``.ndx`` file as one-based atom-index sets."""

    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Index file does not exist: {path}")

    groups: dict[str, set[int]] = {}
    current: str | None = None
    with path.open() as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.split(";", 1)[0].strip()
            if not line:
                continue
            if line.startswith("[") and line.endswith("]"):
                name = line[1:-1].strip()
                if not name:
                    raise ValueError(f"Empty index-group name at {path}:{line_number}")
                if name in groups:
                    raise ValueError(f"Duplicate index group '{name}' in {path}")
                groups[name] = set()
                current = name
                continue
            if current is None:
                raise ValueError(
                    f"Atom indices occur before any group header at {path}:{line_number}"
                )
            try:
                values = [int(token) for token in line.split()]
            except ValueError as exc:
                raise ValueError(f"Invalid atom index at {path}:{line_number}") from exc
            if any(value < 1 for value in values):
                raise ValueError(f"Index values must be positive at {path}:{line_number}")
            groups[current].update(values)

    if not groups:
        raise ValueError(f"No index groups found in {path}")
    return groups
