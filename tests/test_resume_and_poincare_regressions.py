from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace


def _write_csv(path: Path, rows: int, field: str = "value") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[field])
        writer.writeheader()
        for i in range(rows):
            writer.writerow({field: i})


def test_resume_candidate_generation_requires_expected_rows_and_pdbs(tmp_path: Path) -> None:
    from GENPEPT import _resume_candidate_generation_done

    out = tmp_path / "run"
    cand_dir = out / "candidate_seeds"
    cand_dir.mkdir(parents=True)
    args = SimpleNamespace(out=str(out), n=20, n_candidate_seeds=5)
    csv_path = out / "candidate_seeds.csv"

    _write_csv(csv_path, 1, "candidate_pdb_path")
    (cand_dir / "candidate_000.pdb").write_text("END\n")
    assert not _resume_candidate_generation_done(args, csv_path, cand_dir)

    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["candidate_pdb_path"])
        writer.writeheader()
        for i in range(5):
            pdb = cand_dir / f"candidate_{i:03d}.pdb"
            pdb.write_text("END\n")
            writer.writerow({"candidate_pdb_path": str(pdb)})
    assert _resume_candidate_generation_done(args, csv_path, cand_dir)


def test_resume_minimization_requires_all_expected_inputs(tmp_path: Path) -> None:
    from GENPEPT import _resume_minimization_done

    score_csv = tmp_path / "implicit_minimization_scores.csv"
    _write_csv(score_csv, 2)
    assert not _resume_minimization_done(score_csv, 3)

    _write_csv(score_csv, 3)
    assert _resume_minimization_done(score_csv, 3)


def test_resume_basin_hop_requires_parent_times_step_rows(tmp_path: Path) -> None:
    from GENPEPT import _resume_basin_hop_done, _write_resume_completion_marker

    out = tmp_path / "run"
    seed_dir = out / "aa_implicit_minimized_pdbs"
    seed_dir.mkdir(parents=True)
    for i in range(2):
        (seed_dir / f"seed_{i}.pdb").write_text("END\n")
    args = SimpleNamespace(out=str(out), bh_steps=3, bh_parent_seeds=0)
    hop_csv = out / "basin_hop_minima.csv"

    _write_csv(hop_csv, 7)
    assert not _resume_basin_hop_done(args, hop_csv, seed_dir)

    _write_resume_completion_marker(hop_csv, "basin_hopping", rows=7, expected_rows=8)
    assert _resume_basin_hop_done(args, hop_csv, seed_dir)

    _write_csv(hop_csv, 8)
    assert _resume_basin_hop_done(args, hop_csv, seed_dir)


def test_poincare_saved_frame_index_matches_segment_sampling() -> None:
    from analyze_gareus_mbar import _saved_frame_index_from_step

    assert _saved_frame_index_from_step(step=1200, resume_start=1000, step_per_frame=200) == 0
    assert _saved_frame_index_from_step(step=2000, resume_start=1000, step_per_frame=200) == 4
    assert _saved_frame_index_from_step(step=1000, resume_start=1000, step_per_frame=200) == -1


def test_poincare_primary_cv_gate_accepts_only_contact_fraction() -> None:
    from analyze_gareus_mbar import _poincare_primary_cv_supported

    assert _poincare_primary_cv_supported({"primary_cv": "nonlocal-contacts"})
    assert _poincare_primary_cv_supported({
        "primary_cv_label": "nonlocal contact fraction",
        "primary_cv_units": "dimensionless",
    })
    assert not _poincare_primary_cv_supported({
        "primary_cv": "distance",
        "primary_cv_label": "terminal distance",
        "primary_cv_units": "A",
    })
