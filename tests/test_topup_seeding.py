import csv
import json
import os
import time
import types

import pytest

from gareus.topup_seeding import (
    SeedMismatchError, SeedState, assert_seed_matches, load_seed_index, load_seed_states,
)


def _fake_export(d, sid, cv1, cv2, xml="<State/>"):
    (d / "final_window_states").mkdir(parents=True, exist_ok=True)
    (d / "final_window_states" / f"state_{sid}.xml").write_text(xml)
    idx = d / "final_window_states" / "index.json"
    data = json.loads(idx.read_text()) if idx.exists() else {}
    data[str(sid)] = {"window": sid, "cv1": cv1, "cv2": cv2}
    idx.write_text(json.dumps(data))


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
    from pep_gamd_fixture import build_small_simulation
    from gareus.topup_seeding import export_final_window_states
    sim = build_small_simulation(platform="Reference")
    export_final_window_states(tmp_path, [sim], assignments=[0], state_id_of_window={0: 7},
                               cv_of_replica=lambda r: (0.11, 0.22))
    seed = load_seed_states([tmp_path], [7])[7]
    st = sim.context.getState(getPositions=True, getVelocities=True)
    assert seed.cv1 == 0.11 and len(seed.positions) == len(st.getPositions())
    assert seed.box is not None


# ── production.py wiring: _seed_topup_windows_from_parent_states,
# _topup_parent_dirs_by_creation_order, _state_id_of_window_from_epoch_map ──────────


def _write_epoch_window_map(out_dir, window_to_state):
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "epoch_window_map.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["epoch_window", "state_id"])
        writer.writeheader()
        for w, sid in window_to_state.items():
            writer.writerow({"epoch_window": w, "state_id": sid})


def test_seed_topup_windows_uses_the_segments_own_window_map_and_excludes_out_dir(tmp_path, monkeypatch):
    import gareus.topup_seeding as ts
    from gareus.production import _seed_topup_windows_from_parent_states

    monkeypatch.setattr(ts, "_deserialize_state", lambda text: (f"pos:{text}", f"vel:{text}", f"box:{text}"))

    campaign = tmp_path / "epoch_001"
    baseline = campaign / "baseline"
    topup = campaign / "topup_001_1000"
    out_dir = campaign / "topup_002_500"

    _fake_export(baseline, 10, 0.10, 0.20)  # window 0 -> state 10
    _fake_export(baseline, 11, 0.30, 0.40)  # window 1 -> state 11 (stale; overridden below)
    _fake_export(topup, 11, 0.35, 0.45)  # a later topup re-exports state 11
    # out_dir already holds its own (irrelevant) export -- it must never be read as a
    # parent of itself.
    _fake_export(out_dir, 10, 9.99, 9.99)

    # sparse (non-identity) epoch_window_map.csv: window 0 -> state 10, window 1 -> state 11.
    _write_epoch_window_map(out_dir, {0: 10, 1: 11})

    args = types.SimpleNamespace()
    positions, velocities, boxes, seed_by_window = _seed_topup_windows_from_parent_states(args, out_dir, nrep=2)

    assert seed_by_window[0].cv1 == 0.10 and seed_by_window[0].source == str(baseline)
    assert seed_by_window[1].cv1 == 0.35 and seed_by_window[1].source == str(topup)
    assert positions[0] == "pos:<State/>" and velocities[0] == "vel:<State/>" and boxes[0] == "box:<State/>"


def test_seed_topup_windows_falls_back_to_identity_when_the_window_map_is_absent(tmp_path, monkeypatch):
    import gareus.topup_seeding as ts
    from gareus.production import _seed_topup_windows_from_parent_states

    monkeypatch.setattr(ts, "_deserialize_state", lambda text: ("pos", "vel", "box"))

    campaign = tmp_path / "epoch_001"
    baseline = campaign / "baseline"
    out_dir = campaign / "topup_001_1000"
    out_dir.mkdir(parents=True)
    _fake_export(baseline, 0, 0.10, 0.20)
    _fake_export(baseline, 1, 0.30, 0.40)
    # No epoch_window_map.csv written for out_dir.

    args = types.SimpleNamespace()
    _, _, _, seed_by_window = _seed_topup_windows_from_parent_states(args, out_dir, nrep=2)
    assert seed_by_window[0].cv1 == 0.10 and seed_by_window[1].cv1 == 0.30


def test_seed_topup_windows_raises_naming_the_missing_window_and_state(tmp_path, monkeypatch):
    import gareus.topup_seeding as ts
    from gareus.production import _seed_topup_windows_from_parent_states

    monkeypatch.setattr(ts, "_deserialize_state", lambda text: ("pos", "vel", "box"))

    campaign = tmp_path / "epoch_001"
    out_dir = campaign / "topup_001_1000"
    out_dir.mkdir(parents=True)
    # No baseline/parent export exists anywhere -- window 0 has no seed.

    args = types.SimpleNamespace()
    with pytest.raises(ts.SeedMismatchError, match=r"window 0 \(state 0\)"):
        _seed_topup_windows_from_parent_states(args, out_dir, nrep=1)


def test_topup_parent_dirs_ordered_by_export_mtime_not_directory_name(tmp_path):
    # Reproduces the real chignolin_5 shape documented in CLAUDE.md's 2026-08-05
    # plot_adaptive_diagnostics.py fix: topup_NNN_MMMMM's numeric suffix is a
    # remaining-step duration that SHRINKS across successive extension rounds, so it
    # sorts exactly backwards from true creation order.
    from gareus.production import _topup_parent_dirs_by_creation_order

    campaign = tmp_path / "epoch_001"
    baseline = campaign / "baseline"
    t1 = campaign / "topup_001_34794000"  # created first, chronologically
    t2 = campaign / "topup_001_25733000"  # created second
    t3 = campaign / "topup_001_18937000"  # created third (most recent) but sorts FIRST by name
    out_dir = campaign / "topup_002_1"

    for i, d in enumerate((baseline, t1, t2, t3)):
        (d / "final_window_states").mkdir(parents=True)
        idx = d / "final_window_states" / "index.json"
        idx.write_text("{}")
        t = time.time() + i  # strictly increasing mtimes, in true creation order
        os.utime(idx, (t, t))
    out_dir.mkdir(parents=True)

    ordered = _topup_parent_dirs_by_creation_order(out_dir)
    assert ordered == [baseline, t1, t2, t3]
    assert out_dir not in ordered


def test_topup_parent_dirs_puts_an_unexported_parent_first(tmp_path):
    from gareus.production import _topup_parent_dirs_by_creation_order

    campaign = tmp_path / "epoch_001"
    baseline = campaign / "baseline"  # no final_window_states dir at all yet
    baseline.mkdir(parents=True)
    topup = campaign / "topup_001_1000"
    (topup / "final_window_states").mkdir(parents=True)
    (topup / "final_window_states" / "index.json").write_text("{}")
    out_dir = campaign / "topup_002_500"
    out_dir.mkdir(parents=True)

    ordered = _topup_parent_dirs_by_creation_order(out_dir)
    assert ordered == [baseline, topup]


def test_state_id_of_window_from_epoch_map_reads_a_sparse_map_and_defaults_to_empty(tmp_path):
    from gareus.production import _state_id_of_window_from_epoch_map

    assert _state_id_of_window_from_epoch_map(tmp_path / "nothing") == {}

    d = tmp_path / "seg"
    _write_epoch_window_map(d, {0: 12, 2: 7})
    assert _state_id_of_window_from_epoch_map(d) == {0: 12, 2: 7}
