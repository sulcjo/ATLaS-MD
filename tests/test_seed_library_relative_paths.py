"""GENPEPT writes survivor_pdb_path relative to ITS cwd (the parent of the output dir),
e.g. ``chignolin_genpept_r7/final_implicit_survivor_seeds/x.pdb``.  The loader must resolve
that from any cwd by re-rooting at ``seed_conformers_dir.parent`` (S3 pilot attempt 3 on
aurum loaded 0/1970 seeds because the job ran from the code directory)."""
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

from gareus.seeding import load_genpept_conformer_library

_PDB = (
    "ATOM      1  N   GLY A   1       0.000   0.000   0.000  1.00  0.00           N\n"
    "ATOM      2  CA  GLY A   1       1.458   0.000   0.000  1.00  0.00           C\n"
    "ATOM      3  C   GLY A   1       2.009   1.420   0.000  1.00  0.00           C\n"
    "END\n"
)


def _make_library(parent: Path, rel_pdb_path: str) -> Path:
    seed_dir = parent / "genpept_r7"
    seed_dir.mkdir()
    pdb = parent / rel_pdb_path
    pdb.parent.mkdir(parents=True, exist_ok=True)
    pdb.write_text(_PDB)
    (seed_dir / "final_survivor_seeds.csv").write_text(f"survivor_pdb_path\n{rel_pdb_path}\n")
    return seed_dir


def _load_from_other_cwd(seed_dir: Path):
    other = tempfile.mkdtemp(prefix="elsewhere_")
    old = os.getcwd()
    os.chdir(other)
    try:
        return load_genpept_conformer_library(seed_dir, args=SimpleNamespace(), topology=None)
    finally:
        os.chdir(old)


def test_genpept_cwd_relative_survivor_path_loads_from_any_cwd():
    with tempfile.TemporaryDirectory() as tmp:
        parent = Path(tmp)
        seed_dir = _make_library(parent, "genpept_r7/final_implicit_survivor_seeds/survivor_000.pdb")
        library = _load_from_other_cwd(seed_dir)
    assert len(library) == 1, "path relative to the GENPEPT cwd must be re-rooted at seed_dir.parent"
    assert library[0]["pdb_path"].name == "survivor_000.pdb"


def test_seed_dir_relative_survivor_path_still_loads():
    with tempfile.TemporaryDirectory() as tmp:
        parent = Path(tmp)
        seed_dir = parent / "genpept_r7"
        seed_dir.mkdir()
        pdb = seed_dir / "final_implicit_survivor_seeds" / "survivor_001.pdb"
        pdb.parent.mkdir(parents=True)
        pdb.write_text(_PDB)
        (seed_dir / "final_survivor_seeds.csv").write_text(
            "survivor_pdb_path\nfinal_implicit_survivor_seeds/survivor_001.pdb\n"
        )
        library = _load_from_other_cwd(seed_dir)
    assert len(library) == 1
