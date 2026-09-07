import csv, pathlib, tempfile
import numpy as np


def _frames(tmp, n_members=4, n_frames=50):
    rng = np.random.default_rng(0); out = []
    for m in range(n_members):
        for f in range(n_frames):
            p = tmp / f"m{m}_f{f}.pdb"; p.write_text("ATOM      1  CA  GLY A   1       0.000   0.000   0.000  1.00  0.00           C\nEND\n")
            out.append({"member_id": m, "frame": f, "cv1": float(rng.uniform(0, 1)), "pdb_path": str(p)})
    return out


def test_selection_prefers_distinct_members_and_nearest_frames():
    from gareus.swarm.seeds import select_window_seed_frames
    tmp = pathlib.Path(tempfile.mkdtemp())
    frames = _frames(tmp)
    sel = select_window_seed_frames(frames, [0.2, 0.8], per_window=3, discard_frames=10)
    assert set(sel) == {0, 1}
    for w, c in zip((0, 1), (0.2, 0.8)):
        assert 1 <= len(sel[w]) <= 3
        assert len({fr["member_id"] for fr in sel[w]}) == len(sel[w])          # distinct members
        assert all(fr["frame"] >= 10 for fr in sel[w])
        nearest = min(abs(fr["cv1"] - c) for fr in frames if fr["frame"] >= 10)
        assert abs(sel[w][0]["cv1"] - c) <= nearest + 1e-12


def test_export_seed_bank_writes_rows_load_genpept_library_can_read():
    from gareus.swarm.seeds import select_window_seed_frames, export_seed_bank
    tmp = pathlib.Path(tempfile.mkdtemp())
    sel = select_window_seed_frames(_frames(tmp), [0.5], per_window=2)
    bank = export_seed_bank(tmp / "seed_bank", sel, [0.5], round_index=0)
    rows = list(csv.DictReader((bank / "final_survivor_seeds.csv").open()))
    assert len(rows) == len(sel[0]) and all(pathlib.Path(r["survivor_pdb_path"]).exists() for r in rows)
    assert all(r["source_label"] == "swarm_round_000" for r in rows)
    assert all(r["secondary_cv_value"] == "" for r in rows) and all(float(r["primary_cv_value"]) >= 0 for r in rows)


def test_source_run_dir_uses_member_dir_grandparent_when_frames_subdir():
    import csv as _csv
    from gareus.swarm.seeds import export_seed_bank
    tmp = pathlib.Path(tempfile.mkdtemp())
    member_dir = tmp / "member_0003" / "frames"
    member_dir.mkdir(parents=True)
    pdb = member_dir / "frame_00010.pdb"
    pdb.write_text("ATOM      1  CA  GLY A   1       0.000   0.000   0.000  1.00  0.00           C\nEND\n")
    sel = {0: [{"member_id": 3, "frame": 10, "cv1": 0.05, "pdb_path": str(pdb)}]}
    bank = export_seed_bank(tmp / "seed_bank", sel, [0.05], round_index=0)
    rows = list(_csv.DictReader((bank / "final_survivor_seeds.csv").open()))
    assert len(rows) == 1
    assert pathlib.Path(rows[0]["source_run_dir"]).name == "member_0003"


def test_source_run_dir_uses_explicit_member_dir_verbatim():
    import csv as _csv
    from gareus.swarm.seeds import export_seed_bank
    tmp = pathlib.Path(tempfile.mkdtemp())
    frames_dir = tmp / "somewhere" / "frames"
    frames_dir.mkdir(parents=True)
    pdb = frames_dir / "frame_00007.pdb"
    pdb.write_text("ATOM      1  CA  GLY A   1       0.000   0.000   0.000  1.00  0.00           C\nEND\n")
    explicit_dir = tmp / "explicit_member_dir"
    sel = {0: [{"member_id": 2, "frame": 7, "cv1": 0.05, "pdb_path": str(pdb), "member_dir": str(explicit_dir)}]}
    bank = export_seed_bank(tmp / "seed_bank", sel, [0.05], round_index=0)
    rows = list(_csv.DictReader((bank / "final_survivor_seeds.csv").open()))
    assert rows[0]["source_run_dir"] == str(explicit_dir)


def test_export_seed_bank_skips_missing_frame_pdb_and_records_it():
    import csv as _csv, json as _json
    from gareus.swarm.seeds import export_seed_bank
    tmp = pathlib.Path(tempfile.mkdtemp())
    ok_pdb = tmp / "ok.pdb"
    ok_pdb.write_text("ATOM      1  CA  GLY A   1       0.000   0.000   0.000  1.00  0.00           C\nEND\n")
    missing_pdb = tmp / "does_not_exist.pdb"
    sel = {
        0: [
            {"member_id": 0, "frame": 10, "cv1": 0.05, "pdb_path": str(ok_pdb)},
            {"member_id": 1, "frame": 10, "cv1": 0.06, "pdb_path": str(missing_pdb)},
        ]
    }
    bank = export_seed_bank(tmp / "seed_bank", sel, [0.05], round_index=0)
    rows = list(_csv.DictReader((bank / "final_survivor_seeds.csv").open()))
    assert len(rows) == 1
    summary = _json.loads((bank / "selection.json").read_text())
    assert len(summary["skipped"]) == 1
    assert summary["skipped"][0]["member_id"] == 1
    assert summary["skipped"][0]["frame"] == 10
    assert summary["skipped"][0]["pdb_path"] == str(missing_pdb)


def test_export_seed_bank_records_empty_frames_for_windows_with_no_candidates():
    import json as _json
    from gareus.swarm.seeds import export_seed_bank
    tmp = pathlib.Path(tempfile.mkdtemp())
    pdb = tmp / "only.pdb"
    pdb.write_text("ATOM      1  CA  GLY A   1       0.000   0.000   0.000  1.00  0.00           C\nEND\n")
    sel = {0: [{"member_id": 0, "frame": 10, "cv1": 0.05, "pdb_path": str(pdb)}]}
    bank = export_seed_bank(tmp / "seed_bank", sel, [0.05, 0.9], round_index=0)
    summary = _json.loads((bank / "selection.json").read_text())
    assert summary["1"] == {"centre": 0.9, "frames": []}
