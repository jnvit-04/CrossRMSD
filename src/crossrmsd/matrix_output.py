"""Export a labeled RMSD matrix from CrossRMSD's pairwise CSV."""

from __future__ import annotations

import csv
import math
import tempfile
from pathlib import Path

import numpy as np


def write_matrix_from_pairs(
    pair_csv: str | Path,
    matrix_csv: str | Path,
) -> Path:
    """Convert pairwise RMSDs into a full CSV matrix.

    Expected pairwise columns:
        reference_frame, target_frame, rmsd

    Supports:
        - Full rectangular cross-trajectory matrices.
        - Upper-triangular intra-trajectory comparisons.

    No RMSDs are recalculated.
    """
    pair_csv = Path(pair_csv)
    matrix_csv = Path(matrix_csv)

    if pair_csv.resolve() == matrix_csv.resolve():
        raise ValueError(
            "Matrix and pairwise outputs must have different paths."
        )

    ref_ids: dict[str, int] = {}
    tgt_ids: dict[str, int] = {}
    n_pairs = 0

    # First pass: discover matrix dimensions.
    with pair_csv.open("r", newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        next(reader, None)  # Skip CSV header.

        for line_no, cells in enumerate(reader, start=2):
            if not cells:
                continue

            if len(cells) < 3:
                raise ValueError(
                    f"Invalid pairwise CSV at line {line_no}"
                )

            ref_ids.setdefault(cells[0], len(ref_ids))
            tgt_ids.setdefault(cells[1], len(tgt_ids))
            n_pairs += 1

    if not n_pairs:
        raise ValueError("No RMSD pairs found.")

    n_ref = len(ref_ids)
    n_tgt = len(tgt_ids)

    labels = list(dict.fromkeys([*ref_ids, *tgt_ids]))
    n_unique = len(labels)

    full = n_pairs == n_ref * n_tgt
    triangular = n_pairs == n_unique * (n_unique - 1) // 2

    # Resolve the two-frame intra-trajectory edge case.
    if n_pairs == 1 and n_ref == n_tgt == 1 and ref_ids != tgt_ids:
        full = False
        triangular = True

    if not full and not triangular:
        raise ValueError(
            "Pairwise data do not represent a complete "
            "rectangular or upper-triangular matrix."
        )

    if full:
        row_labels = list(ref_ids)
        col_labels = list(tgt_ids)
        row_index = ref_ids
        col_index = tgt_ids
    else:
        row_labels = labels
        col_labels = labels
        row_index = {label: i for i, label in enumerate(labels)}
        col_index = row_index

    matrix_csv.parent.mkdir(parents=True, exist_ok=True)

    # Use a disk-backed array to avoid duplicating a huge matrix in RAM.
    with tempfile.TemporaryDirectory(
        prefix="crossrmsd-matrix-",
        dir=matrix_csv.parent,
    ) as tmp:
        matrix = np.memmap(
            Path(tmp) / "rmsd.f64",
            dtype=np.float64,
            mode="w+",
            shape=(len(row_labels), len(col_labels)),
        )
        matrix[:] = np.nan

        if not full:
            np.fill_diagonal(matrix, 0.0)

        # Second pass: populate matrix entries.
        with pair_csv.open("r", newline="", encoding="utf-8") as file:
            reader = csv.reader(file)
            next(reader, None)

            for line_no, cells in enumerate(reader, start=2):
                if not cells:
                    continue

                i = row_index[cells[0]]
                j = col_index[cells[1]]
                value = float(cells[2])

                if not math.isfinite(value):
                    raise ValueError(
                        f"Non-finite RMSD at line {line_no}"
                    )

                if not math.isnan(matrix[i, j]):
                    raise ValueError(
                        f"Duplicate RMSD pair at line {line_no}"
                    )

                matrix[i, j] = value

                if not full:
                    matrix[j, i] = value

        if np.isnan(matrix).any():
            raise ValueError("RMSD matrix contains missing values.")

        # Export the matrix, preserving frame identifiers.
        with matrix_csv.open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.writer(file)
            writer.writerow(["frame", *col_labels])

            for label, row in zip(row_labels, matrix):
                writer.writerow(
                    [label, *(format(v, ".17g") for v in row)]
                )

        # Release the final row view before closing the memory map.
        del row
        del matrix

    return matrix_csv
