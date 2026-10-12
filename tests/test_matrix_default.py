"""Regression tests for default matrix CSV and optional pairwise output."""
import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from crossrmsd.cli import main
from crossrmsd.io import write_matrix_csv
from crossrmsd.models import Trajectory


def trajectory(label, indices):
    return Trajectory(
        label=label,
        path=Path(f"{label}.pdb"),
        atoms=(),
        coords_angstrom=np.empty((len(indices), 0, 3)),
        source_frame_indices=np.asarray(indices),
    )


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.reader(handle))


def pdb_traj(path, count, displacement):
    with path.open("w") as fh:
        for frame in range(count):
            fh.write(f"MODEL     {frame+1:4d}\n")
            fh.write(f"TITLE     t= {frame*10:.3f}\n")
            for atom_i, (x, y, z) in enumerate(
                [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)], start=1
            ):
                change = displacement * frame * atom_i / 12.0
                fh.write(
                    f"ATOM  {atom_i:5d}  CA  ALA A{atom_i:4d}    "
                    f"{x + change:8.3f}{y - change:8.3f}{z + change:8.3f}"
                    f"  1.00  0.00           C\n"
                )
            fh.write("ENDMDL\n")


class MatrixOutputTests(unittest.TestCase):
    def test_rectangular_matrix_uses_source_frame_labels(self):
        a = trajectory("A", [0, 5])
        b = trajectory("B", [0, 10, 20])
        values = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "matrix.csv"
            write_matrix_csv(path, a, b, values, False, "nm")
            rows = read_csv(path)
            self.assertEqual(rows[0], ["frame", "0", "10", "20"])
            self.assertEqual([r[0] for r in rows[1:]], ["0", "5"])
            np.testing.assert_array_equal(
                [[float(v) for v in r[1:]] for r in rows[1:]],
                values.reshape(2, 3),
            )

    def test_symmetric_matrix_from_packed_triangle(self):
        a = trajectory("A", [0, 5, 10, 15])
        vals = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "matrix.csv"
            write_matrix_csv(path, a, a, vals, True, "nm")
            rows = read_csv(path)
            self.assertEqual(rows[0], ["frame", "0", "5", "10", "15"])
            result = np.asarray([[float(v) for v in r[1:]] for r in rows[1:]])
            np.testing.assert_array_equal(result, [
                [0, 0.1, 0.2, 0.3],
                [0.1, 0, 0.4, 0.5],
                [0.2, 0.4, 0, 0.6],
                [0.3, 0.5, 0.6, 0],
            ])

    def test_bad_shape_fails_before_writing(self):
        a = trajectory("A", [0, 5])
        b = trajectory("B", [0, 5, 10])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.csv"
            with self.assertRaisesRegex(ValueError, "Matrix expects"):
                write_matrix_csv(path, a, b, np.array([0.1]), False, "nm")
            self.assertFalse(path.exists())

    def test_cli_integration_matrix_pairs_xpm_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            a, b = tmp / "a.pdb", tmp / "b.pdb"
            pdb_traj(a, 4, 0.2)
            pdb_traj(b, 6, 0.3)
            out = tmp / "results.csv"
            log = tmp / "results.log"
            args = ["-t", str(a), "-t", str(b), "--out", str(out),
                    "--log", str(log), "--fit", "name CA", "--rmsd", "name CA",
                    "--stride", "2", "--ncpu", "1", "--pairs", "--xpm"]
            self.assertEqual(main(args), 0)
            self.assertTrue(out.exists())
            self.assertTrue((tmp / "results_pairs.csv").exists())
            self.assertTrue((tmp / "results.xpm").exists())
            rows = read_csv(out)
            self.assertEqual(rows[0], ["frame", "0", "2", "4"])
            self.assertEqual([row[0] for row in rows[1:]], ["0", "2"])
            pairs = read_csv(tmp / "results_pairs.csv")
            self.assertEqual(len(pairs), 7)
            for pair in pairs[1:]:
                i = rows[0].index(pair[1])
                j = [r[0] for r in rows[1:]].index(pair[0]) + 1
                self.assertEqual(rows[j][i], pair[2])
            text = log.read_text()
            self.assertIn("Matrix CSV export time", text)
            self.assertIn("Pairwise CSV export time", text)
            self.assertLess(text.index("Matrix CSV export finished"), text.index("FINISHED"))
            self.assertIn("Total elapsed time", text)

    def test_cli_intra_matrix_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            a = tmp / "a.pdb"
            pdb_traj(a, 4, 0.2)
            out = tmp / "intra.csv"
            self.assertEqual(main(["-t", str(a), "--out", str(out)]), 0)
            self.assertFalse((tmp / "intra_pairs.csv").exists())
            rows = read_csv(out)
            self.assertEqual(rows[0], ["frame", "0", "1", "2", "3"])
            result = np.array([[float(v) for v in r[1:]] for r in rows[1:]])
            np.testing.assert_array_equal(result, result.T)
            np.testing.assert_array_equal(np.diag(result), np.zeros(4))

    def test_three_trajectories_named_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            trajs = [tmp / f"traj_{name}.pdb" for name in "ABC"]
            for i, p in enumerate(trajs):
                pdb_traj(p, i+3, 0.2 + 0.1 * i)
            args = sum((["-t", str(p)] for p in trajs), [])
            args += ["--out", str(tmp / "result.csv"), "--pairs"]
            self.assertEqual(main(args), 0)
            for name in ("A_vs_B", "A_vs_C", "B_vs_C"):
                self.assertTrue((tmp / f"result_{name}.csv").exists())
                self.assertTrue((tmp / f"result_{name}_pairs.csv").exists())

    def test_dry_run_does_not_write_matrix(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            a = tmp / "a.pdb"
            pdb_traj(a, 3, 0.2)
            out = tmp / "dry.csv"
            self.assertEqual(main(["-t", str(a), "--out", str(out), "--dry-run", "--pairs"]), 0)
            self.assertFalse(out.exists())
            self.assertFalse((tmp / "dry_pairs.csv").exists())


if __name__ == "__main__":
    unittest.main()
