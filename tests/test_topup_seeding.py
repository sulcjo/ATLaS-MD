import json

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
