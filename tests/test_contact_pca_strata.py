"""Contact-pattern PCA stratification (GENPEPT --contact-pca-strata, swarm 5-axis --swarm-bins).

Two axes from the PCA of each conformer's binary CA residue-pair contact vector, one definition
(gareus/contact_pca.py) shared by GENPEPT seed selection and the swarm member planner; the swarm's
member total can be capped (--swarm-max-members) when more axes multiply the cells.
"""
from __future__ import annotations

import csv
import json
import types
from pathlib import Path

import numpy as np
import pytest

from gareus import contact_pca as CP

N_RES = 10


def _traces(n=60, seed=0):
    """CA traces (Angstrom): extended zig-zags, U-turns (hairpin-like) and compact coils."""
    rng = np.random.default_rng(seed)
    out = []
    for k in range(n):
        kind = k % 3
        if kind == 0:
            ca = np.array([[3.8 * i, 0.8 * (i % 2), 0.0] for i in range(N_RES)])
        elif kind == 1:
            half = N_RES // 2
            ca = np.array([[3.8 * i, 0.0, 0.0] for i in range(half)] +
                          [[3.8 * (N_RES - 1 - i), 5.0, 0.0] for i in range(half, N_RES)])
        else:
            t = np.linspace(0, 3 * np.pi, N_RES)
            ca = np.stack([4.5 * np.cos(t), 4.5 * np.sin(t), 1.5 * t], axis=1)
        out.append(ca + rng.normal(0, 0.6, ca.shape))
    return out


def _write_pdb(path: Path, ca_A: np.ndarray, caps: bool = True) -> Path:
    lines, serial = [], 1
    if caps:
        lines.append(f"HETATM{serial:5d}  CH3 ACE A   0    {0.0:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00           C")
        serial += 1
    for i, (x, y, z) in enumerate(ca_A, start=1):
        lines.append(f"ATOM  {serial:5d}  N   GLY A{i:4d}    {x - 1:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           N")
        serial += 1
        lines.append(f"ATOM  {serial:5d}  CA  GLY A{i:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C")
        serial += 1
    path.write_text("\n".join(lines) + "\nEND\n")
    return path


# ---- shared definition --------------------------------------------------------------------

def test_contact_vector_matches_genpept():
    import GENPEPT
    for ca in _traces(9):
        mine = CP.contact_vector(ca, cutoff_A=8.0, min_sep=3)
        theirs, count = GENPEPT.contact_vector_from_coords(ca, 8.0, 3)
        assert np.array_equal(mine, theirs) and int(mine.sum()) == count


def test_basis_fit_project_save_load(tmp_path):
    vecs = np.vstack([CP.contact_vector(c) for c in _traces()])
    b = CP.fit_basis(vecs, cutoff_A=8.0, min_sep=3, n_residues=N_RES, source="test")
    again = CP.fit_basis(vecs[::-1], cutoff_A=8.0, min_sep=3, n_residues=N_RES, source="test")
    assert np.allclose(np.abs(b.components), np.abs(again.components))
    assert np.allclose(b.components, again.components)               # deterministic sign
    pcs = b.project(vecs)
    assert pcs.shape == (len(vecs), 2) and abs(pcs[:, 0].mean()) < 1e-9
    assert b.explained_variance_ratio[0] >= b.explained_variance_ratio[1] > 0
    path = b.save(tmp_path / CP.BASIS_FILENAME)
    loaded = CP.load_basis(path)
    assert np.allclose(loaded.project(vecs), pcs) and loaded.sha256 == b.sha256
    rec = json.loads(path.read_text())
    rec["mean"][0] += 0.5
    path.write_text(json.dumps(rec))
    with pytest.raises(ValueError, match="sha256"):
        CP.load_basis(path)
    with pytest.raises(ValueError):
        b.project(np.zeros((1, 3)))


def test_read_pdb_ca_skips_caps(tmp_path):
    ca = _traces(1)[0]
    got = CP.read_pdb_ca_A(_write_pdb(tmp_path / "s.pdb", ca))
    assert got.shape == (N_RES, 3) and np.allclose(got, np.round(ca, 3), atol=1e-3)


# ---- GENPEPT --------------------------------------------------------------------------------

def test_genpept_stratified_select_represents_every_cell():
    import GENPEPT
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 5))
    keys = [(int(i % 3), int((i // 3) % 3)) for i in range(200)]
    keys[7] = (9, 9)                                    # a one-member cell
    badness = rng.normal(size=200)
    sel, labels = GENPEPT.stratified_select(X, 20, "farthest", badness, keys)
    assert len(sel) == 20 and len(set(sel.tolist())) == 20 and len(labels) == 200
    assert {keys[i] for i in sel} == set(keys)          # every occupied cell represented
    assert 7 in sel.tolist()


def test_genpept_preselection_key_gains_contact_bins():
    import GENPEPT
    n = 90
    traces = _traces(n)
    store = types.SimpleNamespace(records=[None] * n, rg_A=np.full(n, 10.0), end_to_end_A=np.full(n, 20.0),
                                  contact_count=np.full(n, 4), bank_name=np.array([b"default"] * n),
                                  contacts=np.vstack([CP.contact_vector(c) for c in traces]), seq="GYDPETGTWG")
    args = types.SimpleNamespace(preselection_bin_quota=1, preselection_rg_bin_A=1.0, preselection_e2e_bin_A=1.0,
                                 preselection_contact_bin=2, preselection_include_bank=True,
                                 contact_pca_strata=False, contact_pca_bins=3, contact_cutoff=8.0, contact_min_sep=3)
    off, s_off = GENPEPT.preselection_quota_indices(args, store)
    assert s_off["unique_bins"] == 1 and s_off["contact_pca_bins"] is None
    args.contact_pca_strata = True
    store.contact_pcs = CP.fit_basis(store.contacts, cutoff_A=8.0, min_sep=3, n_residues=N_RES).project(store.contacts)
    on, s_on = GENPEPT.preselection_quota_indices(args, store)
    assert s_on["unique_bins"] > 1 and len(on) > len(off)


def test_genpept_survivor_basis_prefers_the_saved_pool_basis(tmp_path):
    import GENPEPT
    vecs = np.vstack([CP.contact_vector(c) for c in _traces()])
    pool = CP.fit_basis(vecs, cutoff_A=8.0, min_sep=3, n_residues=N_RES, source="pool")
    pool.save(tmp_path / CP.BASIS_FILENAME)
    args = types.SimpleNamespace(contact_cutoff=8.0, contact_min_sep=3)
    basis, pcs = GENPEPT.survivor_contact_pca(args, tmp_path, vecs[:20], N_RES)
    assert basis.source == "pool" and pcs.shape == (20, 2)
    (tmp_path / CP.BASIS_FILENAME).unlink()
    basis2, _ = GENPEPT.survivor_contact_pca(args, tmp_path, vecs[:20], N_RES)
    assert "viable implicit pool" in basis2.source and (tmp_path / CP.BASIS_FILENAME).exists()


# ---- swarm planner ----------------------------------------------------------------------------

def test_parse_bins_and_axes():
    from gareus.swarm.driver import _parse_bins
    from gareus.swarm.stratify import axes_for_bins
    assert _parse_bins("4,3,3") == (4, 3, 3) and _parse_bins("4,3,3,3,3") == (4, 3, 3, 3, 3)
    assert axes_for_bins((4, 3, 3, 3, 3)) == ("cv1", "rg", "e2e", "cpc1", "cpc2")
    with pytest.raises(ValueError):
        _parse_bins("4,3,3,3")


def _seeds(n=60):
    from gareus.swarm.stratify import SeedDescriptor
    rng = np.random.default_rng(2)
    return [SeedDescriptor(f"seed_{i:05d}", f"/x/{i}.pdb", float(rng.random()), float(0.6 + rng.random()),
                           float(1 + rng.random()), float(rng.normal()), float(rng.normal())) for i in range(n)]


def test_five_axis_cells_and_capped_plan():
    from gareus.swarm.stratify import BASE_AXES, CONTACT_AXES, plan_members, stratify_cells
    seeds = _seeds()
    cells = stratify_cells(seeds, (2, 2, 2, 3, 3))
    assert all(len(k) == 5 for k in cells)
    axes = BASE_AXES + CONTACT_AXES
    rows, meta = plan_members(cells, 6, seed_ns=5.0, budget_ns=None, base_seed=0, axes=axes,
                              max_members=len(cells) + 3)
    assert len(rows) == len(cells) + 3 == meta["n_members"]
    assert {r["cell_id"] for r in rows} == {"_".join(map(str, k)) for k in cells}   # every cell first
    assert meta["members_per_cell_max"] == 2 and meta["n_cells_with_members"] == len(cells)
    assert all("cell_cpc1" in r and "cell_cpc2" in r for r in rows)
    few, meta2 = plan_members(cells, 6, seed_ns=5.0, budget_ns=None, base_seed=0, axes=axes, max_members=3)
    biggest = sorted(cells, key=lambda k: (-len(cells[k]), k))[:3]
    assert len(few) == 3 and {r["cell_id"] for r in few} == {"_".join(map(str, k)) for k in biggest}


def test_three_axis_plan_unchanged_without_cap():
    from gareus.swarm.stratify import plan_members, stratify_cells
    seeds = _seeds()
    cells = stratify_cells(seeds, (4, 3, 3))
    rows, meta = plan_members(cells, 2, seed_ns=1.0, budget_ns=None, base_seed=7)
    assert set(rows[0]) == {"member_id", "cell_id", "cell_cv1", "cell_rg", "cell_e2e", "seed_id", "seed_pdb",
                            "replicate", "velocity_seed"}
    assert len(rows) == 2 * len(cells) and "max_members" not in meta


def _swarm_args(tmp_path, bins, lib_dir):
    return types.SimpleNamespace(swarm_bins=bins, swarm_seed_ns=5.0, swarm_replicates_per_cell=2, swarm_budget_ns=None,
                                 seed=0, seed_conformers_dir=str(lib_dir), swarm_max_members=0,
                                 swarm_contact_pca_cutoff_a=8.0, swarm_contact_pca_min_sep=3,
                                 swarm_seed_source="production-frames")


def _library(tmp_path, n=45):
    lib = []
    for i, ca in enumerate(_traces(n)):
        pdb = _write_pdb(tmp_path / f"seed_{i:03d}.pdb", ca)
        lib.append({"pdb_path": str(pdb), "positions_nm": ca / 10.0, "topology_to_conformer_atom_index": {},
                    "primary_cv_value": float(i), "validated_rg_nm": 0.8, "validated_e2e_nm": 1.5})
    return lib


def _patch_driver(monkeypatch, library):
    from gareus.swarm import driver
    import gareus.cv
    monkeypatch.setattr(driver, "prepare_primary_cv_definition", lambda topology, args: {"contact_pairs": [(0, 1)]})
    monkeypatch.setattr(driver, "_load_seed_library_for_round", lambda args, r, topology, primary_cv_def: library)
    monkeypatch.setattr(driver, "_ca_indices_in_seed", lambda lib, topology: None)
    monkeypatch.setattr(gareus.cv, "nonlocal_contact_cv_from_positions_nm",
                        lambda pos, pairs, args: float(np.linalg.norm(pos[-1] - pos[0])))
    return driver


def test_round0_five_axis_plan_end_to_end(tmp_path, monkeypatch):
    library = _library(tmp_path)
    driver = _patch_driver(monkeypatch, library)
    out = tmp_path / "run"
    args = _swarm_args(tmp_path, "2,2,2,3,3", tmp_path)
    rows, meta = driver.build_or_load_plan(args, out, 0, topology=None, contact_pairs=[(0, 1)])
    rd = driver.round_dir(out, 0)
    header = next(csv.reader((rd / "plan.csv").open()))
    assert header[2:7] == ["cell_cv1", "cell_rg", "cell_e2e", "cell_cpc1", "cell_cpc2"]
    assert set(meta["edges"]) == {"cv1", "rg", "e2e", "cpc1", "cpc2"}
    assert meta["contact_pca"]["source"] == "fitted on the seed library"
    assert (rd / CP.BASIS_FILENAME).exists()
    desc_header = next(csv.reader((rd / "seed_descriptors.csv").open()))
    assert desc_header[-2:] == ["cpc1", "cpc2"]
    again, _m = driver.build_or_load_plan(args, out, 0, topology=None, contact_pairs=[(0, 1)])  # resume
    assert again == rows and isinstance(again[0]["cell_cpc1"], int)
    # round 1 reuses round 0's frozen basis + edges
    r1, meta1 = driver.build_or_load_plan(args, out, 1, topology=None, contact_pairs=[(0, 1)])
    assert meta1["contact_pca"]["sha256"] == meta["contact_pca"]["sha256"]
    assert all(len(r["cell_id"].split("_")) == 5 for r in r1)


def test_round0_prefers_the_library_basis(tmp_path, monkeypatch):
    library = _library(tmp_path)
    vecs = np.vstack([CP.contact_vector(c) for c in _traces(30, seed=9)])
    CP.fit_basis(vecs, cutoff_A=8.0, min_sep=3, n_residues=N_RES, source="genpept pool").save(tmp_path / CP.BASIS_FILENAME)
    driver = _patch_driver(monkeypatch, library)
    _rows, meta = driver.build_or_load_plan(_swarm_args(tmp_path, "2,2,2,3,3", tmp_path), tmp_path / "run", 0,
                                            topology=None, contact_pairs=[(0, 1)])
    assert meta["contact_pca"]["source"].startswith("seed library")


def test_three_axis_round0_writes_todays_files(tmp_path, monkeypatch):
    library = _library(tmp_path)
    driver = _patch_driver(monkeypatch, library)
    out = tmp_path / "run"
    driver.build_or_load_plan(_swarm_args(tmp_path, "2,2,2", tmp_path), out, 0, topology=None, contact_pairs=[(0, 1)])
    rd = driver.round_dir(out, 0)
    assert next(csv.reader((rd / "plan.csv").open())) == driver.PLAN_COLUMNS
    assert next(csv.reader((rd / "seed_descriptors.csv").open())) == driver.SEED_DESCRIPTOR_COLUMNS
    assert not (rd / CP.BASIS_FILENAME).exists()
    assert "contact_pca" not in json.loads((rd / "plan_meta.json").read_text())


def test_cli_validates_bins_and_cap():
    from gareus.cli import parse_args
    base = ["--seq", "GYDPETGTWG", "--swarm-stage", "analyze", "--out", "/tmp/x"]
    a = parse_args(base + ["--swarm-bins", "4,3,3,3,3", "--swarm-max-members", "174"])
    assert a.swarm_bins == "4,3,3,3,3" and a.swarm_max_members == 174
    for bad in (["--swarm-bins", "4,3"], ["--swarm-bins", "4,3,3,0,3"], ["--swarm-max-members", "-1"]):
        with pytest.raises(SystemExit):
            parse_args(base + bad)


def test_dashboard_cells_ignore_contact_axes():
    from gareus.swarm.round_progress import SwarmRoundProgress
    sink = types.SimpleNamespace(tui_mode="none", mode="both", args=None)
    rp = SwarmRoundProgress(sink, n_members=2, n_prod_steps=10, n_already_done=0, timestep_fs=2.0, round_index=0,
                            edges={"cv1": [0, 1, 2, 3, 4], "rg": [0, 1, 2, 3], "e2e": [0, 1, 2, 3],
                                   "cpc1": [-1, 0, 1, 2], "cpc2": [-1, 0, 1, 2]})
    rp.add_frame(0, 0.5, 0.5, 0.5)
    s = rp.snapshot()
    assert s.cells_total == 36 and s.cells_visited == 1 and s.bins == (4, 3, 3)
