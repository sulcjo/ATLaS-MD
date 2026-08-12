"""Unit tests for _discover_latest_seed_bank.

Regression coverage for a real recurring crash: current_seed_bank is a plain
local variable in run_adaptive_production_auto_loop, only ever advanced
in-process right after write_epoch_seed_bank()/write_seed_bank_from_run_dirs()
succeeds for the epoch that just finished. It is never persisted, so a fresh
--resume process re-initializes it from args.seed_conformers_dir (the raw
GENPEPT library) regardless of how much real adaptive sampling already
happened. If the resumed run picks back up inside an epoch's baseline/topup
segment -- past the point in the loop where current_seed_bank would normally
advance this process -- the real seed bank is silently never used again:
filter_seed_bank_for_state_ids finds no source_state_id match in the untagged
GENPEPT library, and starting-structure seeding quietly degrades to
generic/no-seed fallback. Confirmed on a real run (chignolin_6):
filtered_seed_bank_report.json recorded source_seed_bank as the raw GENPEPT
dir with 0 matched rows on a resume that should have used seed_bank_epoch_000.
"""
import csv
import json

from gareus.adaptive_production import _discover_latest_seed_bank, _seed_bank_dir_is_usable

FIELDNAMES = [
    "seed_name", "survivor_pdb_path", "source_run_dir", "source_pdb_path",
    "source_label", "source_state_id", "source_epoch_window",
    "primary_cv_value", "secondary_cv_value",
]


def _write_seed_bank(dir_path, *, n_rows=1, status="ok"):
    dir_path.mkdir(parents=True, exist_ok=True)
    csv_path = dir_path / "final_survivor_seeds.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        for i in range(n_rows):
            writer.writerow({
                "seed_name": f"seed_{i}", "survivor_pdb_path": f"pdbs/seed_{i}.pdb",
                "source_run_dir": "x", "source_pdb_path": "x", "source_label": "x",
                "source_state_id": i, "source_epoch_window": i,
                "primary_cv_value": 0.0, "secondary_cv_value": 0.0,
            })
    (dir_path / "seed_bank_report.json").write_text(json.dumps({"status": status}))


def test_no_seed_banks_returns_none(tmp_path):
    assert _discover_latest_seed_bank(tmp_path) is None


def test_picks_highest_numbered_epoch_bank(tmp_path):
    _write_seed_bank(tmp_path / "seed_bank_epoch_000")
    _write_seed_bank(tmp_path / "seed_bank_epoch_001")
    _write_seed_bank(tmp_path / "seed_bank_epoch_002")
    result = _discover_latest_seed_bank(tmp_path)
    assert result == tmp_path / "seed_bank_epoch_002"


def test_prefers_seed_bank_final_over_any_epoch_bank(tmp_path):
    _write_seed_bank(tmp_path / "seed_bank_epoch_000")
    _write_seed_bank(tmp_path / "seed_bank_epoch_003")
    _write_seed_bank(tmp_path / "seed_bank_final")
    result = _discover_latest_seed_bank(tmp_path)
    assert result == tmp_path / "seed_bank_final"


def test_skips_empty_or_failed_bank_falls_back_to_older_usable_one(tmp_path):
    # This is exactly the real-world case: the latest epoch's seed bank write
    # failed/was empty, but an earlier epoch's is still perfectly usable --
    # falling back to the raw GENPEPT library would be strictly worse than
    # reusing real, state-tagged adaptive sampling from an earlier epoch.
    _write_seed_bank(tmp_path / "seed_bank_epoch_000", n_rows=5, status="ok")
    _write_seed_bank(tmp_path / "seed_bank_epoch_001", n_rows=0, status="empty")
    result = _discover_latest_seed_bank(tmp_path)
    assert result == tmp_path / "seed_bank_epoch_000"


def test_ignores_non_numeric_or_unrelated_directories(tmp_path):
    (tmp_path / "seed_bank_epoch_final_backup").mkdir()  # non-numeric suffix
    (tmp_path / "epoch_000").mkdir()  # doesn't match the seed_bank_epoch_ prefix
    assert _discover_latest_seed_bank(tmp_path) is None


def test_seed_bank_dir_is_usable_true_for_real_bank(tmp_path):
    _write_seed_bank(tmp_path / "bank")
    assert _seed_bank_dir_is_usable(tmp_path / "bank") is True


def test_seed_bank_dir_is_usable_false_without_report_but_with_rows(tmp_path):
    # write_seed_bank_from_run_dirs' report presence is not guaranteed at the
    # exact instant of a crash mid-write; a real non-empty CSV with no report
    # file at all should still count as usable rather than being rejected.
    bank = tmp_path / "bank"
    bank.mkdir()
    with (bank / "final_survivor_seeds.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerow({
            "seed_name": "s", "survivor_pdb_path": "p", "source_run_dir": "x",
            "source_pdb_path": "x", "source_label": "x", "source_state_id": 0,
            "source_epoch_window": 0, "primary_cv_value": 0.0, "secondary_cv_value": 0.0,
        })
    assert _seed_bank_dir_is_usable(bank) is True


def test_seed_bank_dir_is_usable_false_for_missing_csv(tmp_path):
    bank = tmp_path / "bank"
    bank.mkdir()
    assert _seed_bank_dir_is_usable(bank) is False
