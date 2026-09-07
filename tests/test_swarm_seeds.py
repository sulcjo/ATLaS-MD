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
