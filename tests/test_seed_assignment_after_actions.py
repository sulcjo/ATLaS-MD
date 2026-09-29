"""States an epoch's actions create must get a nearest-CV seed, not the generic fallback pool."""
import csv

import gareus.adaptive_production as ap
from gareus.adaptive_production import WindowStateRegistry


def _seed_bank(tmp_path, cvs):
    bank = tmp_path / "seed_bank"
    (bank / "pdbs").mkdir(parents=True)
    rows = []
    for i, cv in enumerate(cvs):
        pdb = bank / "pdbs" / f"seed_{i}.pdb"
        pdb.write_text("END\n")
        rows.append({"seed_name": f"seed_{i}", "survivor_pdb_path": str(pdb),
                     "source_state_id": str(i), "primary_cv_value": str(cv)})
    with (bank / "final_survivor_seeds.csv").open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return bank


def _registry():
    reg = WindowStateRegistry()
    for c in (0.1, 0.2, 0.3):
        reg.add_state(c, 800.0, epoch=0, source="seed")
    return reg


def _filter(bank, sid, out):
    report = ap.filter_seed_bank_for_state_ids(bank, [sid], out)
    rows = list(csv.DictReader((out / "final_survivor_seeds.csv").open()))
    return report, rows


def test_a_new_rung_state_is_seeded_from_its_centres_nearest_seed(tmp_path):
    bank = _seed_bank(tmp_path, [0.1, 0.2, 0.3])
    reg = _registry()
    ap.select_state_aware_seeds_for_targets(bank, reg)          # written before the actions
    new = reg.add_state(0.3, 800.0, gamd_lambda=0.5, epoch=0, source="adaptive", reason="add_rung")
    new_id = int(getattr(new, "state_id", new))

    ap._reassign_seeds_after_actions(bank, reg)
    _, rows = _filter(bank, new_id, tmp_path / "filtered")
    assert [r["filter_source"] for r in rows] == ["state_aware_assignment"]
    assert rows[0]["seed_name"] == "seed_2"                     # the seed at CV 0.3


def test_without_the_reassignment_a_new_state_falls_to_the_generic_pool(tmp_path):
    bank = _seed_bank(tmp_path, [0.1, 0.2, 0.3])
    reg = _registry()
    ap.select_state_aware_seeds_for_targets(bank, reg)
    new = reg.add_state(0.3, 800.0, gamd_lambda=0.5, epoch=0, source="adaptive", reason="add_rung")
    _, rows = _filter(bank, int(getattr(new, "state_id", new)), tmp_path / "filtered")
    assert [r["filter_source"] for r in rows] == ["generic_fallback"]
    assert rows[0]["seed_name"] == "seed_0"                     # first row of the bank, CV 0.1


def test_a_missing_seed_bank_is_a_no_op(tmp_path):
    assert ap._reassign_seeds_after_actions(None, _registry()) is None
    assert ap._reassign_seeds_after_actions(tmp_path / "absent", _registry()) is None


def test_the_epoch_loop_reassigns_seeds_after_applying_actions():
    import inspect
    src = inspect.getsource(ap.run_adaptive_production_auto_loop)
    apply_at = src.index("_apply_registry_actions(registry, actions, epoch, policy=policy)")
    reassign_at = src.index("_reassign_seeds_after_actions(current_seed_bank, registry)")
    assert apply_at < reassign_at < src.index("_write_runtime_pool_reports(adaptive_dir, runtime_pool)", apply_at)
