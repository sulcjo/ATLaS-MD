"""Unit tests for rescore_seed_bank_secondary_cv.

Regression coverage for the seed-selection-goes-stale-across-a-tica_switch_cv2
bug: write_epoch_seed_bank()/write_seed_bank_from_run_dirs() write each seed
row's secondary_cv_value as a copy of the source state's own secondary_center
at write time, never a measurement of the seed structure itself, and that
write happens before the post-tICA-update centers exist. select_state_aware_
seeds_for_targets() then "matches" every state to its own seed trivially.
Confirmed on a real crashed run (chignolin_6, epoch_001): 12/12 gate-failing
"seeded" windows matched their own source state this way, each still 0.6-1.6
tIC1 units off the real post-switch target -- collapsed to <0.1 by re-scoring
with real structures and re-running the nearest-seed assignment.

Uses a fake OpenMM app/unit (positions supplied directly, not parsed from a
real PDB file) so the test exercises the CSV read/rewrite, path-resolution,
and failure-accounting logic without an OpenMM dependency, matching this
suite's convention of testing pure logic (see test_seed_scoring_degradation.py)
rather than round-tripping through real trajectory files. backbone_dihedral_
features/project_tica1 themselves are exercised for real (pure numpy, no
OpenMM needed).
"""
import csv
import math

import numpy as np
import pytest

from gareus.adaptive_production import (
    rescore_seed_bank_secondary_cv,
    select_state_aware_seeds_for_targets,
    WindowStateRegistry,
)
from gareus.tica import TICAResult, backbone_dihedral_features, project_tica1

FIELDNAMES = [
    "seed_name", "survivor_pdb_path", "source_run_dir", "source_pdb_path",
    "source_label", "source_state_id", "source_epoch_window",
    "primary_cv_value", "secondary_cv_value",
]


class _FakePositions:
    def __init__(self, positions_nm):
        self._positions_nm = np.asarray(positions_nm, dtype=np.float64)

    def value_in_unit(self, unit_):
        assert unit_ == "nanometer"
        return self._positions_nm


class _FakePDBFile:
    """Maps a path to canned positions instead of parsing a real PDB file."""

    positions_by_path = {}

    def __init__(self, path):
        if path not in self.positions_by_path:
            raise FileNotFoundError(path)
        self.positions = _FakePositions(self.positions_by_path[path])


class _FakeApp:
    def __init__(self, positions_by_path):
        self.PDBFile = type("PDBFile", (_FakePDBFile,), {"positions_by_path": positions_by_path})


class _FakeUnit:
    nanometer = "nanometer"


def _make_tica_result(n_torsions=2):
    # 2 phi + 2 psi torsions over 8 atoms -> 8 features (sin/cos each).
    phi = [(0, 1, 2, 3), (4, 5, 6, 7)][:n_torsions]
    psi = [(1, 2, 3, 4), (5, 6, 7, 0)][:n_torsions]
    n_feat = 4 * n_torsions
    rng = np.random.default_rng(0)
    weights = rng.normal(size=n_feat)
    weights /= np.linalg.norm(weights)
    return TICAResult(
        weights=weights, eigenvalue=0.9, mean=np.zeros(n_feat), offset=0.0,
        lag=10, phi_torsion_indices=phi, psi_torsion_indices=psi, n_samples=100,
    )


def _write_seed_bank_csv(seed_bank_dir, rows):
    csv_path = seed_bank_dir / "final_survivor_seeds.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def _positions_for_atoms(n_atoms, seed):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n_atoms, 3))


def test_rescore_overwrites_placeholder_with_real_measurement(tmp_path):
    result = _make_tica_result()
    positions = _positions_for_atoms(8, seed=1)
    expected = float(
        project_tica1(backbone_dihedral_features(positions, result.phi_torsion_indices, result.psi_torsion_indices)[None, :], result)[0]
    )

    row = {
        "seed_name": "epoch_000_state_0000_seed_00",
        "survivor_pdb_path": "pdbs/state0.pdb",
        "source_run_dir": "x", "source_pdb_path": "x",
        "source_label": "epoch_000", "source_state_id": 0, "source_epoch_window": 0,
        "primary_cv_value": 0.0,
        "secondary_cv_value": 999.0,  # stale placeholder that must get overwritten
    }
    _write_seed_bank_csv(tmp_path, [row])
    (tmp_path / "pdbs").mkdir()
    pdb_path = str(tmp_path / "pdbs" / "state0.pdb")
    (tmp_path / "pdbs" / "state0.pdb").write_text("")  # existence-checked before being opened

    app = _FakeApp({pdb_path: positions})
    report = rescore_seed_bank_secondary_cv(tmp_path, result, app, _FakeUnit())

    assert report["status"] == "ok"
    assert report["n_rescored"] == 1
    assert report["n_failed"] == 0

    with (tmp_path / "final_survivor_seeds.csv").open() as fh:
        rewritten = list(csv.DictReader(fh))
    assert len(rewritten) == 1
    assert float(rewritten[0]["secondary_cv_value"]) == pytest.approx(expected)
    # Other columns must survive the round trip untouched.
    assert rewritten[0]["source_state_id"] == "0"


def test_rescore_resolves_relative_and_pdbs_fallback_paths(tmp_path):
    result = _make_tica_result()
    positions = _positions_for_atoms(8, seed=2)
    (tmp_path / "pdbs").mkdir()
    pdb_path = str(tmp_path / "pdbs" / "orphan.pdb")
    (tmp_path / "pdbs" / "orphan.pdb").write_text("")  # existence-checked before being opened

    # survivor_pdb_path points at a name that doesn't exist directly under
    # seed_bank_dir; the pdbs/<basename> fallback (mirroring
    # filter_seed_bank_for_state_ids' own resolution) must find it.
    row = {
        "seed_name": "epoch_000_state_0001_seed_00",
        "survivor_pdb_path": "some/other/place/orphan.pdb",
        "source_run_dir": "x", "source_pdb_path": "x",
        "source_label": "epoch_000", "source_state_id": 1, "source_epoch_window": 1,
        "primary_cv_value": 0.0, "secondary_cv_value": 0.0,
    }
    _write_seed_bank_csv(tmp_path, [row])

    app = _FakeApp({pdb_path: positions})
    report = rescore_seed_bank_secondary_cv(tmp_path, result, app, _FakeUnit())

    assert report["status"] == "ok"
    assert report["n_rescored"] == 1


def test_rescore_counts_missing_pdb_as_failed_without_crashing(tmp_path):
    result = _make_tica_result()
    row = {
        "seed_name": "epoch_000_state_0002_seed_00",
        "survivor_pdb_path": "pdbs/does_not_exist.pdb",
        "source_run_dir": "x", "source_pdb_path": "x",
        "source_label": "epoch_000", "source_state_id": 2, "source_epoch_window": 2,
        "primary_cv_value": 0.0, "secondary_cv_value": 0.0,
    }
    _write_seed_bank_csv(tmp_path, [row])
    (tmp_path / "pdbs").mkdir()

    app = _FakeApp({})  # nothing resolvable
    report = rescore_seed_bank_secondary_cv(tmp_path, result, app, _FakeUnit())

    assert report["status"] == "all_failed"
    assert report["n_failed"] == 1
    assert report["n_rescored"] == 0


def test_rescore_returns_empty_status_for_missing_or_empty_csv(tmp_path):
    result = _make_tica_result()
    report = rescore_seed_bank_secondary_cv(tmp_path, result, _FakeApp({}), _FakeUnit())
    assert report["status"] == "empty"


def test_rescore_declines_with_no_torsion_indices(tmp_path):
    # Regression guard for the paired bug: a TICAResult saved before the
    # _maybe_update_tica_cvaux torsion-index fallback existed (or one that
    # never found a bootstrap file) has empty index lists and must not be
    # silently treated as usable.
    empty_result = TICAResult(
        weights=np.array([]), eigenvalue=0.5, mean=np.array([]), offset=0.0,
        lag=10, phi_torsion_indices=[], psi_torsion_indices=[], n_samples=10,
    )
    row = {
        "seed_name": "epoch_000_state_0000_seed_00",
        "survivor_pdb_path": "pdbs/state0.pdb",
        "source_run_dir": "x", "source_pdb_path": "x",
        "source_label": "epoch_000", "source_state_id": 0, "source_epoch_window": 0,
        "primary_cv_value": 0.0, "secondary_cv_value": 0.0,
    }
    _write_seed_bank_csv(tmp_path, [row])
    report = rescore_seed_bank_secondary_cv(tmp_path, empty_result, _FakeApp({}), _FakeUnit())
    assert report["status"] == "no_torsion_indices"


def test_cross_state_reassignment_prefers_closer_measured_seed_over_own_state(tmp_path):
    """The actual bug-fix behavior: after rescoring, a target should be able to
    borrow a *different* state's seed when it measures closer than its own
    state's seed under the (now-current) coordinate -- exactly what fixed the
    real chignolin_6 crash (see module docstring)."""
    registry = WindowStateRegistry()
    registry.add_state(primary_center=0.0, primary_k=10.0, secondary_center=5.0)  # state_id=0
    registry.add_state(primary_center=0.0, primary_k=10.0, secondary_center=0.0)  # state_id=1

    rows_dir_csv = [
        # state 0's own seed measured far from state 0's real (post-update)
        # target of 5.0 ...
        {"target_state_id": "", "seed_name": "seed0", "survivor_pdb_path": "p0",
         "source_run_dir": "x", "source_pdb_path": "x", "source_label": "e",
         "source_state_id": 0, "source_epoch_window": 0,
         "primary_cv_value": 0.0, "secondary_cv_value": 0.1},
        # ... but state 1's seed happens to measure right at 5.0.
        {"target_state_id": "", "seed_name": "seed1", "survivor_pdb_path": "p1",
         "source_run_dir": "x", "source_pdb_path": "x", "source_label": "e",
         "source_state_id": 1, "source_epoch_window": 1,
         "primary_cv_value": 0.0, "secondary_cv_value": 5.05},
    ]

    def fake_read_csv_dicts(path):
        return rows_dir_csv

    import gareus.adaptive_production as ap_mod
    orig = ap_mod._read_csv_dicts
    ap_mod._read_csv_dicts = fake_read_csv_dicts
    try:
        report = select_state_aware_seeds_for_targets(tmp_path, registry)
    finally:
        ap_mod._read_csv_dicts = orig

    by_target = {a["target_state_id"]: a for a in report["assignments"]}
    assert by_target[0]["seed_name"] == "seed1", (
        "target state 0 (secondary_center=5.0) should borrow state 1's seed "
        "(measured 5.05) over its own state 0's seed (measured 0.1)"
    )
