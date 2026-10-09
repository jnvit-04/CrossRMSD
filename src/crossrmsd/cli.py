"""Command-line interface for CrossRMSD."""

from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path

from crossrmsd import __version__

_THREAD_ENVIRONMENT_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def configure_cpu_limit(ncpu: int) -> None:
    value = str(ncpu)
    for variable in _THREAD_ENVIRONMENT_VARIABLES:
        os.environ[variable] = value
    os.environ["OMP_DYNAMIC"] = "FALSE"
    os.environ["MKL_DYNAMIC"] = "FALSE"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="crossrmsd",
        description=(
            "Calculate pairwise intra- or inter-trajectory RMSDs from prepared "
            "multi-model PDB trajectories."
        ),
    )
    parser.add_argument("--version", action="version", version=f"crossrmsd {__version__}")
    parser.add_argument(
        "--trajectory", "-t", action="append", required=True, type=Path,
        help="Prepared multi-model PDB trajectory. Repeat for multiple trajectories.",
    )
    parser.add_argument(
        "--fit", default="name CA",
        help="Readable atom selection used for rigid-body fitting (default: name CA).",
    )
    parser.add_argument(
        "--rmsd",
        help="Selection used for RMSD measurement. Default: the --fit selection.",
    )
    parser.add_argument("--index", type=Path, help="Optional GROMACS .ndx file.")
    parser.add_argument(
        "--out", required=True, type=Path, metavar="FILE",
        help=(
            "Primary pairwise RMSD CSV. With more than one comparison, the "
            "comparison name is appended before the extension."
        ),
    )
    parser.add_argument(
        "--log", type=Path, metavar="FILE",
        help="Optional run log. Terminal reporting is always shown.",
    )
    parser.add_argument(
        "--kde", action="store_true",
        help="Also write a KDE CSV derived from each pairwise RMSD output.",
    )
    parser.add_argument(
        "--xpm", action="store_true",
        help="Also write a GROMACS gmx rms-compatible XPM matrix for each comparison.",
    )
    parser.add_argument(
        "--write-selection", action="store_true",
        help="Write a selected-atom audit CSV next to --out.",
    )
    parser.add_argument(
        "--units", choices=("nm", "angstrom"), default="nm",
        help="RMSD output unit (default: nm).",
    )
    parser.add_argument(
        "--stride", type=int, default=1,
        help="Retain source frames 0, N, 2N, ... (default: 1).",
    )
    parser.add_argument(
        "--ncpu", type=int, default=1,
        help="Maximum CPU cores for readers and numerical libraries (default: 1).",
    )
    parser.add_argument(
        "--mass-weighted", action="store_true",
        help="Use atomic masses for fitting and RMSD.",
    )
    parser.add_argument(
        "--mass-database", type=Path,
        help="Optional editable ELEMENT MASS table. Default: packaged atomic_masses.tsv.",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Show live progress.")
    parser.add_argument(
        "--chunk-size", type=int, default=2048,
        help="Target frames per vectorized Kabsch batch (default: 2048).",
    )
    parser.add_argument(
        "--xpm-levels", type=int, default=80,
        help="Number of XPM color levels, matching gmx rms default 80.",
    )
    parser.add_argument(
        "--xpm-min", type=float, dest="xpm_min_nm",
        help="Optional XPM minimum in nm, analogous to gmx rms -min.",
    )
    parser.add_argument(
        "--xpm-max", type=float, dest="xpm_max_nm",
        help="Optional XPM maximum in nm, analogous to gmx rms -max.",
    )
    parser.add_argument(
        "--central", dest="central_output", type=Path, metavar="FILE",
        help="Write the exact Campos-Baptista central sampled frame to FILE.",
    )
    parser.add_argument(
        "--centroid", dest="centroid_output", type=Path, metavar="FILE",
        help="Write an iterative generalized-Procrustes synthetic centroid to FILE.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate inputs and selections without calculating RMSDs.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.chunk_size < 1:
        parser.error("--chunk-size must be positive.")
    if args.stride < 1:
        parser.error("--stride must be positive.")
    if args.ncpu < 1:
        parser.error("--ncpu must be positive.")
    if args.xpm_levels < 2:
        parser.error("--xpm-levels must be at least 2.")
    if (
        args.xpm_min_nm is not None
        and args.xpm_max_nm is not None
        and args.xpm_max_nm <= args.xpm_min_nm
    ):
        parser.error("--xpm-max must be greater than --xpm-min.")
    if args.log is not None and args.log.expanduser().resolve() == args.out.expanduser().resolve():
        parser.error("--log and --out must name different files.")

    configure_cpu_limit(1)
    from crossrmsd.analysis import run_crossrmsd
    from crossrmsd.io import RunLog

    command_items = sys.argv if argv is None else ["crossrmsd", *argv]
    command = " ".join(shlex.quote(item) for item in command_items)
    rmsd_expression = args.rmsd if args.rmsd is not None else args.fit

    with RunLog(args.log) as log:
        try:
            run_crossrmsd(
                trajectory_paths=args.trajectory,
                output_file=args.out,
                fit_expression=args.fit,
                rmsd_expression=rmsd_expression,
                index_path=args.index,
                unit=args.units,
                chunk_size=args.chunk_size,
                stride=args.stride,
                ncpu=args.ncpu,
                mass_weighted=args.mass_weighted,
                mass_database=args.mass_database,
                verbose=args.verbose,
                write_kde=args.kde,
                write_selection=args.write_selection,
                write_xpm=args.xpm,
                xpm_levels=args.xpm_levels,
                xpm_min_nm=args.xpm_min_nm,
                xpm_max_nm=args.xpm_max_nm,
                central_output=args.central_output,
                centroid_output=args.centroid_output,
                dry_run=args.dry_run,
                command=command,
                log=log,
            )
        except (OSError, ValueError) as exc:
            log.section("Error")
            log.write(str(exc))
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
