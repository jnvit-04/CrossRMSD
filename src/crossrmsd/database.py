"""Small editable data tables used by CrossRMSD."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path


def default_mass_database_path() -> Path:
    """Return the packaged atomic-mass table path."""

    return Path(str(files("crossrmsd").joinpath("data", "atomic_masses.tsv")))


def load_atomic_masses(path: Path | None = None) -> tuple[dict[str, float], Path]:
    """Load an element-to-mass table from a simple two-column text file."""

    database_path = (path.expanduser().resolve() if path is not None else default_mass_database_path())
    if not database_path.exists():
        raise FileNotFoundError(f"Atomic-mass database does not exist: {database_path}")

    masses: dict[str, float] = {}
    with database_path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(
                    f"Invalid atomic-mass database row at {database_path}:{line_number}; "
                    "expected: ELEMENT MASS"
                )
            element = parts[0].upper()
            try:
                mass = float(parts[1])
            except ValueError as exc:
                raise ValueError(
                    f"Invalid atomic mass at {database_path}:{line_number}: {parts[1]}"
                ) from exc
            if not element.isalpha() or len(element) > 2:
                raise ValueError(
                    f"Invalid element symbol at {database_path}:{line_number}: {element}"
                )
            if mass <= 0.0:
                raise ValueError(
                    f"Atomic mass must be positive at {database_path}:{line_number}"
                )
            if element in masses:
                raise ValueError(
                    f"Duplicate element '{element}' in atomic-mass database: {database_path}"
                )
            masses[element] = mass

    if not masses:
        raise ValueError(f"Atomic-mass database is empty: {database_path}")
    return masses, database_path
