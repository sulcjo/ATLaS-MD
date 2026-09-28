from __future__ import annotations

import csv
import os

import gareus.adaptive_production as ap


def _bank(tmp_path):
    bank = tmp_path / "seed_bank_final"
    (bank / "pdbs").mkdir(parents=True)
    rows = []
    for sid in (0, 1):
        pdb = bank / "pdbs" / f"seed_{sid}.pdb"
        pdb.write_text(f"ATOM {sid}\nEND\n")
        rows.append({"seed_name": f"seed_{sid}", "survivor_pdb_path": str(pdb),
                     "seed_source_state_id": sid})
    with (bank / "final_survivor_seeds.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return bank


def test_filtered_bank_uses_hard_links(tmp_path):
    bank = _bank(tmp_path)
    out = tmp_path / "seg" / "filtered_seed_bank"
    report = ap.filter_seed_bank_for_state_ids(bank, [0, 1], out)
    assert report["linked"] == 2, f"Expected 2 links, got {report['linked']}"
    assert report["copied"] == 0, f"Expected 0 copies, got {report['copied']}"
    for sid in (0, 1):
        src, dst = bank / "pdbs" / f"seed_{sid}.pdb", out / "pdbs" / f"seed_{sid}.pdb"
        assert dst.read_text() == src.read_text()
        assert os.stat(dst).st_ino == os.stat(src).st_ino


def test_falls_back_to_copy_when_link_fails(tmp_path, monkeypatch):
    bank = _bank(tmp_path)

    def no_link(*a, **k):
        raise OSError("cross-device")
    monkeypatch.setattr(ap.os, "link", no_link)
    out = tmp_path / "seg" / "filtered_seed_bank"
    report = ap.filter_seed_bank_for_state_ids(bank, [0], out)
    assert report["linked"] == 0, f"Expected 0 links, got {report['linked']}"
    assert report["copied"] == 1, f"Expected 1 copy, got {report['copied']}"
    dst = out / "pdbs" / "seed_0.pdb"
    assert dst.read_text() == "ATOM 0\nEND\n"
    assert os.stat(dst).st_ino != os.stat(bank / "pdbs" / "seed_0.pdb").st_ino


def test_source_rewrite_preserves_filtered_content(tmp_path):
    """Regression test: rewriting source PDB doesn't affect filtered bank's hard links."""
    bank = _bank(tmp_path)
    out = tmp_path / "seg" / "filtered_seed_bank"
    report = ap.filter_seed_bank_for_state_ids(bank, [0], out)
    assert report["linked"] == 1, "Should have created a hard link"

    # Verify initial content matches
    src = bank / "pdbs" / "seed_0.pdb"
    dst = out / "pdbs" / "seed_0.pdb"
    original_content = "ATOM 0\nEND\n"
    assert dst.read_text() == original_content
    assert os.stat(dst).st_ino == os.stat(src).st_ino

    # Rewrite source PDB with different content using _copy_replace
    new_content = "ATOM 999\nNEW\n"
    tmp_new_pdb = bank / "pdbs" / "temp_seed.pdb"
    tmp_new_pdb.write_text(new_content)
    ap._copy_replace(tmp_new_pdb, src)
    tmp_new_pdb.unlink()

    # Verify: source now has new content, but filtered bank still has original
    assert src.read_text() == new_content, "Source should be rewritten"
    assert dst.read_text() == original_content, "Filtered bank should retain original content"
    assert os.stat(dst).st_ino != os.stat(src).st_ino, "After rewrite, inodes should differ"
