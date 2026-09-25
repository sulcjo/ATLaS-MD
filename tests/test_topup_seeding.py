import csv
import json
import logging
import os
import time
import types

import pytest

from gareus.topup_seeding import (
    SeedMismatchError, SeedState, assert_seed_matches, assert_seed_restraint_matches,
    load_seed_index, load_seed_states, state_id_of_window_from_epoch_map,
    topup_parent_dirs_by_creation_order,
)


def _fake_export(d, sid, cv1, cv2, *, window=None, primary_center=0.0, primary_k=100.0,
                 secondary_center=None, secondary_k=None, export_seq=None, xml="<State/>"):
    (d / "final_window_states").mkdir(parents=True, exist_ok=True)
    (d / "final_window_states" / f"state_{sid}.xml").write_text(xml)
    idx = d / "final_window_states" / "index.json"
    data = json.loads(idx.read_text()) if idx.exists() else {}
    data[str(sid)] = {
        "window": sid if window is None else window,
        "cv1": cv1,
        "cv2": cv2,
        "primary_center": primary_center,
        "primary_k": primary_k,
        "secondary_center": secondary_center,
        "secondary_k": secondary_k,
        "export_seq": time.time_ns() if export_seq is None else export_seq,
    }
    idx.write_text(json.dumps(data))


def _write_epoch_window_map(out_dir, window_to_state):
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "epoch_window_map.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["epoch_window", "state_id"])
        writer.writeheader()
        for w, sid in window_to_state.items():
            writer.writerow({"epoch_window": w, "state_id": sid})


def test_the_index_names_the_latest_parent_per_state(tmp_path):
    base, top = tmp_path / "baseline", tmp_path / "topup_001_1000"
    _fake_export(base, 3, 0.1, 0.2); _fake_export(base, 4, 0.3, 0.4); _fake_export(top, 3, 0.5, 0.6)
    idx = load_seed_index([base, top])
    assert idx == {3: str(top), 4: str(base)}
    assert load_seed_index([tmp_path / "nothing"]) == {}


def test_loads_only_the_states_it_has(tmp_path, monkeypatch):
    import gareus.topup_seeding as ts
    monkeypatch.setattr(ts, "_deserialize_state", lambda text: ("pos", "vel", "box"))
    base = tmp_path / "baseline"
    _fake_export(base, 3, 0.1, 0.2)
    got = load_seed_states([base], [3, 9])
    assert set(got) == {3} and got[3].cv1 == 0.1


def test_assertion_accepts_a_matching_frame_and_rejects_a_swapped_one():
    seed = SeedState(positions=None, velocities=None, box=None, cv1=0.40, cv2=-0.25, source="x")
    assert_seed_matches(seed, 0.4004, -0.2496)
    with pytest.raises(SeedMismatchError, match="does not reproduce"):
        assert_seed_matches(seed, 0.47, -0.25)


def test_assertion_error_names_window_and_state_id():
    seed = SeedState(positions=None, velocities=None, box=None, cv1=0.40, cv2=-0.25, source="x", state_id=17)
    with pytest.raises(SeedMismatchError) as exc_info:
        assert_seed_matches(seed, 0.47, -0.25, window=5)
    assert exc_info.value.window == 5 and exc_info.value.state_id == 17
    assert "window 5" in str(exc_info.value) and "state 17" in str(exc_info.value)


def test_a_non_finite_cv1_always_raises_even_when_both_sides_are_nan():
    # A NaN CV1 means the seeded/loaded state is physically broken (e.g. blown-up
    # positions) -- this must never pass silently just because NaN != NaN comparisons
    # are False. Covers both a NaN-now-vs-finite-seed and a NaN-vs-NaN case.
    seed = SeedState(positions=None, velocities=None, box=None, cv1=0.40, cv2=0.1, source="x")
    with pytest.raises(SeedMismatchError, match="does not reproduce"):
        assert_seed_matches(seed, float("nan"), 0.1)

    nan_seed = SeedState(positions=None, velocities=None, box=None, cv1=float("nan"), cv2=0.1, source="x")
    with pytest.raises(SeedMismatchError, match="does not reproduce"):
        assert_seed_matches(nan_seed, float("nan"), 0.1)


def test_a_nan_cv2_on_either_side_is_tolerated_when_cv1_matches():
    # CV1-only runs legitimately evaluate cv2 as NaN on both the recorded seed and the
    # freshly-read context; that comparison must be skipped rather than raised.
    seed = SeedState(positions=None, velocities=None, box=None, cv1=0.40, cv2=float("nan"), source="x")
    assert_seed_matches(seed, 0.4004, float("nan"))
    assert_seed_matches(seed, 0.4004, 99.0)  # seed.cv2 NaN -> still skipped even if cv2_now is finite

    finite_seed = SeedState(positions=None, velocities=None, box=None, cv1=0.40, cv2=0.1, source="x")
    assert_seed_matches(finite_seed, 0.4004, float("nan"))  # cv2_now NaN -> skipped too


def test_export_and_reload_round_trip_on_the_reference_platform(tmp_path):
    pytest.importorskip("openmm")
    import numpy as np
    from openmm import unit as openmm_unit
    from pep_gamd_fixture import build_small_simulation
    from gareus.topup_seeding import export_final_window_states
    sim = build_small_simulation(platform="Reference")
    export_final_window_states(
        tmp_path, [sim], assignments=[0], state_id_of_window={0: 7},
        cv_of_replica=lambda r: (0.11, 0.22),
        centers_nm=[0.35], ks_kj_nm2=[500.0],
    )
    seed = load_seed_states([tmp_path], [7])[7]
    st = sim.context.getState(getPositions=True, getVelocities=True)
    assert seed.cv1 == 0.11 and len(seed.positions) == len(st.getPositions())
    assert seed.box is not None
    assert seed.primary_center == pytest.approx(0.35) and seed.primary_k == pytest.approx(500.0)
    assert seed.secondary_center is None and seed.secondary_k is None

    src_pos = np.asarray(st.getPositions().value_in_unit(openmm_unit.nanometer))
    seed_pos = np.asarray(seed.positions.value_in_unit(openmm_unit.nanometer))
    src_vel = np.asarray(st.getVelocities().value_in_unit(openmm_unit.nanometer / openmm_unit.picosecond))
    seed_vel = np.asarray(seed.velocities.value_in_unit(openmm_unit.nanometer / openmm_unit.picosecond))
    assert np.allclose(seed_pos, src_pos, atol=1e-6)
    assert np.allclose(seed_vel, src_vel, atol=1e-6)


# ── ruling 14: a corrupt export must degrade to "missing", never crash ──────────────


def test_a_truncated_state_xml_is_treated_as_missing_with_a_warning(tmp_path, caplog):
    pytest.importorskip("openmm")
    base = tmp_path / "baseline"
    _fake_export(base, 3, 0.1, 0.2, xml="<State position")  # truncated/invalid XML
    with caplog.at_level(logging.WARNING, logger="gareus.topup_seeding"):
        got = load_seed_states([base], [3])
    assert 3 not in got
    assert any("state 3" in rec.message for rec in caplog.records if rec.name == "gareus.topup_seeding")


def test_a_corrupt_index_json_in_any_parent_raises_seed_mismatch_error(tmp_path):
    # Ruling 19: unlike a per-state problem (this test's sibling above), a whole
    # corrupt/unreadable index.json must NOT be silently skipped -- skipping it could
    # let an older parent's copy of the same state win and restart its chain from an
    # earlier point. It must raise, naming the broken parent, instead.
    base = tmp_path / "baseline"
    (base / "final_window_states").mkdir(parents=True)
    (base / "final_window_states" / "state_3.xml").write_text("<State/>")
    (base / "final_window_states" / "index.json").write_text("{not valid json")
    with pytest.raises(SeedMismatchError) as exc_info:
        load_seed_index([base])
    assert str(base) in str(exc_info.value)


def test_a_corrupt_index_in_the_newest_parent_is_not_silently_skipped_in_favor_of_an_older_one(tmp_path):
    # The exact scenario ruling 19 exists to prevent: an older parent genuinely has a
    # copy of state 3, and a naive "skip a broken parent" policy would silently seed
    # from that stale copy instead of failing loudly.
    older = tmp_path / "baseline"
    newer = tmp_path / "topup_001_1000"
    _fake_export(older, 3, 0.1, 0.2, export_seq=1)
    (newer / "final_window_states").mkdir(parents=True)
    (newer / "final_window_states" / "index.json").write_text("{not valid json")

    with pytest.raises(SeedMismatchError) as exc_info:
        load_seed_index([older, newer])
    assert str(newer) in str(exc_info.value)

    with pytest.raises(SeedMismatchError) as exc_info2:
        load_seed_states([older, newer], [3])
    assert str(newer) in str(exc_info2.value)


def test_a_record_missing_required_fields_is_treated_as_missing_with_a_warning(tmp_path, caplog):
    base = tmp_path / "baseline"
    (base / "final_window_states").mkdir(parents=True)
    (base / "final_window_states" / "state_3.xml").write_text("<State/>")
    (base / "final_window_states" / "index.json").write_text(json.dumps({"3": {"window": 0}}))  # no cv1/cv2
    with caplog.at_level(logging.WARNING, logger="gareus.topup_seeding"):
        got = load_seed_states([base], [3])
    assert 3 not in got
    assert any("state 3" in rec.message for rec in caplog.records if rec.name == "gareus.topup_seeding")


# ── ruling 15: a seed must belong to this window's own restraint ────────────────────


def test_assert_seed_restraint_matches_accepts_equal_values_and_cv1_only_none_secondary():
    seed = SeedState(positions=None, velocities=None, box=None, cv1=0.1, cv2=0.2, source="x",
                     primary_center=1.0, primary_k=100.0, secondary_center=None, secondary_k=None)
    assert_seed_restraint_matches(seed, window=0, primary_center=1.0000001, primary_k=99.9999999,
                                  secondary_center=None, secondary_k=None)


def test_assert_seed_restraint_matches_rejects_primary_mismatch():
    seed = SeedState(positions=None, velocities=None, box=None, cv1=0.1, cv2=0.2, source="x",
                     primary_center=1.0, primary_k=100.0)
    with pytest.raises(SeedMismatchError, match="primary_center mismatch"):
        assert_seed_restraint_matches(seed, window=2, primary_center=1.5, primary_k=100.0,
                                      secondary_center=None, secondary_k=None)


def test_assert_seed_restraint_matches_rejects_secondary_presence_mismatch():
    seed = SeedState(positions=None, velocities=None, box=None, cv1=0.1, cv2=0.2, source="x",
                     primary_center=1.0, primary_k=100.0, secondary_center=None, secondary_k=None)
    with pytest.raises(SeedMismatchError, match="secondary_center presence mismatch"):
        assert_seed_restraint_matches(seed, window=0, primary_center=1.0, primary_k=100.0,
                                      secondary_center=0.5, secondary_k=10.0)


def test_assert_seed_restraint_matches_rejects_secondary_value_mismatch():
    seed = SeedState(positions=None, velocities=None, box=None, cv1=0.1, cv2=0.2, source="x",
                     primary_center=1.0, primary_k=100.0, secondary_center=0.5, secondary_k=10.0)
    with pytest.raises(SeedMismatchError, match="secondary_center mismatch"):
        assert_seed_restraint_matches(seed, window=0, primary_center=1.0, primary_k=100.0,
                                      secondary_center=0.9, secondary_k=10.0)


def test_assert_seed_restraint_matches_rejects_missing_primary_on_seed():
    seed = SeedState(positions=None, velocities=None, box=None, cv1=0.1, cv2=0.2, source="x")  # legacy export
    with pytest.raises(SeedMismatchError, match="no recorded primary restraint"):
        assert_seed_restraint_matches(seed, window=0, primary_center=1.0, primary_k=100.0,
                                      secondary_center=None, secondary_k=None)


# ── production.py wiring: _seed_topup_windows_from_parent_states,
# state_id_of_window_from_epoch_map, topup_parent_dirs_by_creation_order ────────────


def test_seed_topup_windows_uses_the_segments_own_window_map_and_excludes_out_dir(tmp_path, monkeypatch):
    import gareus.topup_seeding as ts
    from gareus.production import _seed_topup_windows_from_parent_states

    monkeypatch.setattr(ts, "_deserialize_state", lambda text: (f"pos:{text}", f"vel:{text}", f"box:{text}"))

    campaign = tmp_path / "epoch_001"
    baseline = campaign / "baseline"
    topup = campaign / "topup_001_1000"
    out_dir = campaign / "topup_002_500"

    # window 0 -> state 10, window 1 -> state 11 (restraint centers 1.0/2.0).
    _fake_export(baseline, 10, 0.10, 0.20, primary_center=1.0, primary_k=100.0, export_seq=1)
    _fake_export(baseline, 11, 0.30, 0.40, primary_center=2.0, primary_k=150.0, export_seq=1)  # stale
    _fake_export(topup, 11, 0.35, 0.45, primary_center=2.0, primary_k=150.0, export_seq=2)  # later, wins
    # out_dir already holds its own (irrelevant) export -- it must never be read as a
    # parent of itself.
    _fake_export(out_dir, 10, 9.99, 9.99, primary_center=1.0, primary_k=100.0)

    # sparse (non-identity) epoch_window_map.csv: window 0 -> state 10, window 1 -> state 11.
    _write_epoch_window_map(out_dir, {0: 10, 1: 11})

    args = types.SimpleNamespace()
    positions, velocities, boxes, seed_by_window = _seed_topup_windows_from_parent_states(
        args, out_dir, nrep=2, centers_nm=[1.0, 2.0], ks_kj_nm2=[100.0, 150.0],
    )

    assert seed_by_window[0].cv1 == 0.10 and seed_by_window[0].source == str(baseline)
    assert seed_by_window[1].cv1 == 0.35 and seed_by_window[1].source == str(topup)
    assert positions[0] == "pos:<State/>" and velocities[0] == "vel:<State/>" and boxes[0] == "box:<State/>"


def test_seed_topup_windows_fails_closed_when_the_window_map_is_absent(tmp_path, monkeypatch):
    # Ruling 15: a top-up with no epoch_window_map.csv must refuse to seed via an
    # identity window->state fallback, not silently assume window index == state id.
    import gareus.topup_seeding as ts
    from gareus.production import _seed_topup_windows_from_parent_states

    monkeypatch.setattr(ts, "_deserialize_state", lambda text: ("pos", "vel", "box"))

    campaign = tmp_path / "epoch_001"
    baseline = campaign / "baseline"
    out_dir = campaign / "topup_001_1000"
    out_dir.mkdir(parents=True)
    _fake_export(baseline, 0, 0.10, 0.20, primary_center=1.0, primary_k=100.0)
    # No epoch_window_map.csv written for out_dir.

    args = types.SimpleNamespace()
    with pytest.raises(ts.SeedMismatchError, match="epoch_window_map.csv"):
        _seed_topup_windows_from_parent_states(args, out_dir, nrep=1, centers_nm=[1.0], ks_kj_nm2=[100.0])


def test_seed_topup_windows_rejects_a_seed_whose_restraint_does_not_match_the_window(tmp_path, monkeypatch):
    import gareus.topup_seeding as ts
    from gareus.production import _seed_topup_windows_from_parent_states

    monkeypatch.setattr(ts, "_deserialize_state", lambda text: ("pos", "vel", "box"))

    campaign = tmp_path / "epoch_001"
    baseline = campaign / "baseline"
    out_dir = campaign / "topup_001_1000"
    _fake_export(baseline, 0, 0.10, 0.20, primary_center=1.0, primary_k=100.0)
    _write_epoch_window_map(out_dir, {0: 0})

    args = types.SimpleNamespace()
    # window 0's own current restraint (2.0) differs from the seed's recorded one (1.0):
    # an index-misalignment case a CV-value check alone cannot catch.
    with pytest.raises(ts.SeedMismatchError, match="primary_center mismatch"):
        _seed_topup_windows_from_parent_states(args, out_dir, nrep=1, centers_nm=[2.0], ks_kj_nm2=[100.0])


def test_seed_topup_windows_raises_naming_the_missing_window_and_state(tmp_path, monkeypatch):
    import gareus.topup_seeding as ts
    from gareus.production import _seed_topup_windows_from_parent_states

    monkeypatch.setattr(ts, "_deserialize_state", lambda text: ("pos", "vel", "box"))

    campaign = tmp_path / "epoch_001"
    out_dir = campaign / "topup_001_1000"
    _write_epoch_window_map(out_dir, {0: 0})
    # No baseline/parent export exists anywhere -- window 0 has no seed.

    args = types.SimpleNamespace()
    with pytest.raises(ts.SeedMismatchError, match=r"window 0 \(state 0\)") as exc_info:
        _seed_topup_windows_from_parent_states(args, out_dir, nrep=1, centers_nm=[0.0], ks_kj_nm2=[100.0])
    assert exc_info.value.window == 0 and exc_info.value.state_id == 0


def test_topup_parent_dirs_ordered_by_export_seq_not_mtime_or_name(tmp_path):
    # Reproduces the real chignolin_5 shape documented in CLAUDE.md's 2026-08-05
    # plot_adaptive_diagnostics.py fix: topup_NNN_MMMMM's numeric suffix is a
    # remaining-step duration that SHRINKS across successive extension rounds, so it
    # sorts exactly backwards from true creation order. mtimes are also deliberately
    # set backwards here, to prove export_seq alone drives the ordering.
    campaign = tmp_path / "epoch_001"
    baseline = campaign / "baseline"
    t1 = campaign / "topup_001_34794000"  # created first, chronologically
    t2 = campaign / "topup_001_25733000"  # created second
    t3 = campaign / "topup_001_18937000"  # created third (most recent) but sorts FIRST by name
    out_dir = campaign / "topup_002_1"

    seqs = {baseline: 10, t1: 20, t2: 30, t3: 40}
    mtimes_reversed = {baseline: 4_000_000_000.0, t1: 3_000_000_000.0, t2: 2_000_000_000.0, t3: 1_000_000_000.0}
    for d in (baseline, t1, t2, t3):
        (d / "final_window_states").mkdir(parents=True)
        idx = d / "final_window_states" / "index.json"
        idx.write_text(json.dumps({"1": {"cv1": 0.1, "export_seq": seqs[d]}}))
        t = mtimes_reversed[d]
        os.utime(idx, (t, t))
    out_dir.mkdir(parents=True)

    ordered = topup_parent_dirs_by_creation_order(out_dir)
    assert ordered == [baseline, t1, t2, t3]
    assert out_dir not in ordered


def test_topup_parent_dirs_falls_back_to_mtime_when_export_seq_is_absent(tmp_path):
    # A legacy export written before ruling 16 has no export_seq at all.
    campaign = tmp_path / "epoch_001"
    baseline = campaign / "baseline"
    topup = campaign / "topup_001_1000"
    out_dir = campaign / "topup_002_1"

    for i, d in enumerate((baseline, topup)):
        (d / "final_window_states").mkdir(parents=True)
        idx = d / "final_window_states" / "index.json"
        idx.write_text(json.dumps({"1": {"cv1": 0.1}}))  # no export_seq
        t = time.time() + i
        os.utime(idx, (t, t))
    out_dir.mkdir(parents=True)

    ordered = topup_parent_dirs_by_creation_order(out_dir)
    assert ordered == [baseline, topup]


def test_topup_parent_dirs_puts_an_unexported_parent_first(tmp_path):
    campaign = tmp_path / "epoch_001"
    baseline = campaign / "baseline"  # no final_window_states dir at all yet
    baseline.mkdir(parents=True)
    topup = campaign / "topup_001_1000"
    (topup / "final_window_states").mkdir(parents=True)
    (topup / "final_window_states" / "index.json").write_text(json.dumps({"1": {"export_seq": 5}}))
    out_dir = campaign / "topup_002_500"
    out_dir.mkdir(parents=True)

    ordered = topup_parent_dirs_by_creation_order(out_dir)
    assert ordered == [baseline, topup]


def test_state_id_of_window_from_epoch_map_reads_a_sparse_map_and_defaults_to_empty(tmp_path):
    assert state_id_of_window_from_epoch_map(tmp_path / "nothing") == {}

    d = tmp_path / "seg"
    _write_epoch_window_map(d, {0: 12, 2: 7})
    assert state_id_of_window_from_epoch_map(d) == {0: 12, 2: 7}
