import csv, json, pathlib, tempfile, types


def test_parse_member_range_half_open_and_default_all():
    from gareus.swarm.driver import parse_member_range
    assert list(parse_member_range("3:6", 10)) == [3, 4, 5]
    assert list(parse_member_range(None, 4)) == [0, 1, 2, 3]
    assert list(parse_member_range("8:20", 10)) == [8, 9]


def test_round_dirs_and_latest_round_index():
    from gareus.swarm.driver import round_dir, latest_round_index
    out = pathlib.Path(tempfile.mkdtemp())
    assert latest_round_index(out) is None
    round_dir(out, 0).mkdir(parents=True); round_dir(out, 2).mkdir(parents=True)
    assert latest_round_index(out) == 2
    assert round_dir(out, 1).name == "round_001"


def test_later_round_reuses_round0_bin_edges():
    from gareus.swarm.driver import stratify_with_frozen_edges
    from gareus.swarm.stratify import SeedDescriptor
    edges = {"cv1": [0.0, 0.5, 1.0], "rg": [0.0, 1.0], "e2e": [0.0, 3.0]}
    seeds = [SeedDescriptor("a", "/a", 0.1, 0.5, 1.0), SeedDescriptor("b", "/b", 0.9, 0.5, 1.0)]
    cells = stratify_with_frozen_edges(seeds, edges)
    assert set(cells) == {(0, 0, 0), (1, 0, 0)}


def test_run_swarm_stage_refuses_without_seed_conformers_dir():
    from gareus.swarm.driver import run_swarm_stage
    out = pathlib.Path(tempfile.mkdtemp())
    args = types.SimpleNamespace(seed_conformers_dir=None)
    try:
        run_swarm_stage(args, out, None)
    except SystemExit as e:
        assert "seed-conformers-dir" in str(e)
    else:
        raise AssertionError("expected SystemExit")

    # Also refuses when the dir exists but final_survivor_seeds.csv does not.
    bad_dir = out / "not_a_library"
    bad_dir.mkdir()
    args2 = types.SimpleNamespace(seed_conformers_dir=str(bad_dir))
    try:
        run_swarm_stage(args2, out, None)
    except SystemExit as e:
        assert "seed-conformers-dir" in str(e)
    else:
        raise AssertionError("expected SystemExit")


def test_build_or_load_plan_reload_is_a_pure_bookkeeping_readback():
    """A plan.csv/plan_meta.json already on disk is loaded verbatim (no MD, no
    library reload needed on this path) -- exercises _load_plan without OpenMM."""
    from gareus.swarm.driver import round_dir, _load_plan, _write_plan, PLAN_COLUMNS
    out = pathlib.Path(tempfile.mkdtemp())
    rd = round_dir(out, 0); rd.mkdir(parents=True)
    rows_in = [
        {"member_id": 0, "cell_id": "0_0_0", "cell_cv1": 0, "cell_rg": 0, "cell_e2e": 0,
         "seed_id": "seed_00000", "seed_pdb": "/x/a.pdb", "replicate": 0, "velocity_seed": 7},
        {"member_id": 1, "cell_id": "1_0_0", "cell_cv1": 1, "cell_rg": 0, "cell_e2e": 0,
         "seed_id": "seed_00001", "seed_pdb": "/x/b.pdb", "replicate": 0, "velocity_seed": 107},
    ]
    meta_in = {"n_cells": 2, "replicates_per_cell": 1, "n_members": 2, "budget_ns": 2.0, "seed_ns": 1.0,
               "edges": {"cv1": [0, 0.5, 1], "rg": [0, 2], "e2e": [0, 4]}}
    _write_plan(rd, rows_in, meta_in)
    rows_out, meta_out = _load_plan(rd)
    assert rows_out == rows_in
    assert meta_out["edges"] == meta_in["edges"] and meta_out["n_members"] == 2
    assert list(rows_out[0].keys()) == PLAN_COLUMNS


def test_build_or_load_plan_short_circuits_on_resume_without_topology_or_library():
    """Resume must never re-derive the plan: build_or_load_plan should return the
    persisted plan.csv/plan_meta.json without touching topology/contact_pairs at all
    (both are garbage here -- if the function tried to use them it would raise)."""
    from gareus.swarm.driver import round_dir, build_or_load_plan, _write_plan
    out = pathlib.Path(tempfile.mkdtemp())
    rd = round_dir(out, 0); rd.mkdir(parents=True)
    rows_in = [{"member_id": 0, "cell_id": "0_0_0", "cell_cv1": 0, "cell_rg": 0, "cell_e2e": 0,
                "seed_id": "seed_00000", "seed_pdb": "/x/a.pdb", "replicate": 0, "velocity_seed": 3}]
    meta_in = {"n_cells": 1, "replicates_per_cell": 1, "n_members": 1, "budget_ns": 1.0, "seed_ns": 1.0,
               "edges": {"cv1": [0, 1], "rg": [0, 1], "e2e": [0, 1]}}
    _write_plan(rd, rows_in, meta_in)
    rows_out, meta_out = build_or_load_plan(
        types.SimpleNamespace(), out, 0, topology=object(), contact_pairs=None,
    )
    assert rows_out == rows_in and meta_out["n_members"] == 1


def test_load_frozen_edges_requires_round0_meta_and_returns_persisted_edges():
    from gareus.swarm.driver import round_dir, _load_frozen_edges
    out = pathlib.Path(tempfile.mkdtemp())
    try:
        _load_frozen_edges(out)
    except SystemExit as e:
        assert "round_000" in str(e) or "round 0" in str(e)
    else:
        raise AssertionError("expected SystemExit when round_000/plan_meta.json is absent")

    rd = round_dir(out, 0); rd.mkdir(parents=True)
    edges = {"cv1": [0.0, 0.5, 1.0], "rg": [0.0, 1.0], "e2e": [0.0, 3.0]}
    (rd / "plan_meta.json").write_text(json.dumps({"edges": edges}))
    assert _load_frozen_edges(out) == edges


def test_resolve_conformer_resolves_by_seed_pdb_even_when_list_order_is_swapped():
    """The reloaded library can be re-sorted/grown relative to when plan.csv was
    written (frozen envelope: later rounds only add seeds) -- a bare positional
    seed_id index would silently grab the wrong entry here; resolution must go
    through the recorded seed_pdb path instead."""
    from gareus.swarm.seed_library import _resolve_conformer
    # plan-time order was [a, b] (seed_00000 -> a, seed_00001 -> b); the reloaded
    # library has swapped/grown to [b, a] -- seed_00000's positional index 0 would
    # now wrongly resolve to "b" if identity weren't checked.
    library = [{"pdb_path": "/b.pdb"}, {"pdb_path": "/a.pdb"}]
    library_by_path = {e["pdb_path"]: e for e in library}
    row = {"member_id": 3, "seed_id": "seed_00000", "seed_pdb": "/a.pdb"}
    resolved = _resolve_conformer(row, library, library_by_path)
    assert resolved is library[1] and resolved["pdb_path"] == "/a.pdb"


def test_resolve_conformer_raises_naming_both_paths_on_mismatch():
    from gareus.swarm.seed_library import _resolve_conformer
    library = [{"pdb_path": "/x.pdb"}, {"pdb_path": "/y.pdb"}]
    library_by_path = {e["pdb_path"]: e for e in library}
    # seed_pdb matches nothing by path; the positional fallback (seed_id index 0)
    # points at a different path ("/x.pdb") -- both must be named in the error.
    row = {"member_id": 9, "seed_id": "seed_00000", "seed_pdb": "/gone.pdb"}
    try:
        _resolve_conformer(row, library, library_by_path)
    except ValueError as e:
        msg = str(e)
        assert "/gone.pdb" in msg and "/x.pdb" in msg
    else:
        raise AssertionError("expected ValueError")


def _write_production_frames_csv(csv_path, rows):
    with pathlib.Path(csv_path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pdb_path", "cv1", "rg_nm", "e2e_nm"])
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _fake_seed_pdb(d):
    pdb = pathlib.Path(d) / "seed.pdb"
    pdb.write_text("ATOM      1  CA  GLY A   1       0.000   0.000   0.000  1.00  0.00           C\nEND\n")
    return pdb


def test_load_production_frame_library_raises_on_empty_field_naming_file_row_column():
    from gareus.swarm.seed_library import _load_production_frame_library
    d = pathlib.Path(tempfile.mkdtemp())
    pdb = _fake_seed_pdb(d)
    csv_path = d / "frames.csv"
    _write_production_frames_csv(csv_path, [{"pdb_path": str(pdb), "cv1": "", "rg_nm": "0.5", "e2e_nm": "1.0"}])
    try:
        _load_production_frame_library(csv_path)
    except ValueError as e:
        msg = str(e)
        assert str(csv_path) in msg and "row 0" in msg and "'cv1'" in msg
    else:
        raise AssertionError("expected ValueError on an empty required field")


def test_load_production_frame_library_raises_on_garbage_field_naming_file_row_column():
    from gareus.swarm.seed_library import _load_production_frame_library
    d = pathlib.Path(tempfile.mkdtemp())
    pdb = _fake_seed_pdb(d)
    csv_path = d / "frames.csv"
    _write_production_frames_csv(
        csv_path, [{"pdb_path": str(pdb), "cv1": "not-a-number", "rg_nm": "0.5", "e2e_nm": "1.0"}],
    )
    try:
        _load_production_frame_library(csv_path)
    except ValueError as e:
        msg = str(e)
        assert str(csv_path) in msg and "row 0" in msg and "'cv1'" in msg and "not-a-number" in msg
    else:
        raise AssertionError("expected ValueError on a non-numeric required field")


def test_load_production_frame_library_accepts_valid_rows():
    from gareus.swarm.seed_library import _load_production_frame_library
    d = pathlib.Path(tempfile.mkdtemp())
    pdb = _fake_seed_pdb(d)
    csv_path = d / "frames.csv"
    _write_production_frames_csv(csv_path, [{"pdb_path": str(pdb), "cv1": "0.05", "rg_nm": "0.5", "e2e_nm": "1.0"}])
    library = _load_production_frame_library(csv_path)
    assert len(library) == 1
    assert library[0]["primary_cv_value"] == 0.05
    assert library[0]["validated_rg_nm"] == 0.5 and library[0]["validated_e2e_nm"] == 1.0
