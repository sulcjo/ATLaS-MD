import json
import math

from gareus.adaptive.topup_allocator import TopupPlan
from gareus.adaptive.topup_state import load_plan, load_state, save_plan, save_state, update_after_topup


def _plan():
    return TopupPlan(state_ids=(3, 4), steps=5000, deficit_state_ids=(3,), partner_state_ids=(4,),
                     weak_edges_topped=((3, 7),), predicted_sigma={3: 0.09, 4: math.nan},
                     sigma_before={3: 0.20, 4: 0.05}, cost_hours=1.5, reason="planned")


def test_plan_round_trips_including_int_keys_and_nan(tmp_path):
    save_plan(tmp_path, _plan())
    got = load_plan(tmp_path)
    assert got.state_ids == (3, 4) and got.weak_edges_topped == ((3, 7),)
    assert got.predicted_sigma[3] == 0.09 and math.isnan(got.predicted_sigma[4])
    assert not list(tmp_path.glob("*.tmp"))                       # atomic write leaves no temp file
    json.loads((tmp_path / "topup_plan.json").read_text())       # strict JSON (no bare NaN)


def test_state_defaults_and_round_trip(tmp_path):
    st = load_state(tmp_path)
    assert st == {"correction": {}, "edge_attempts": {}, "f_kT": {}, "wall_time": []}
    st["edge_attempts"][(1, 2)] = 1; st["correction"][3] = 0.7
    save_state(tmp_path, st)
    assert load_state(tmp_path)["edge_attempts"] == {(1, 2): 1}


def test_a_non_finite_free_energy_survives_the_round_trip_as_nan(tmp_path):
    st = load_state(tmp_path)
    st["f_kT"] = {0: 0.0, 1: math.nan}
    save_state(tmp_path, st)
    got = load_state(tmp_path)["f_kT"]
    assert got[0] == 0.0 and math.isnan(got[1])


def test_underdelivery_halves_and_smooths_the_correction_and_counts_the_edge():
    st = {"correction": {}, "edge_attempts": {}, "f_kT": {}, "wall_time": []}
    new = update_after_topup(st, _plan(), realised_sigma={3: 0.18})
    # ratio = (0.04-0.0324)/(0.04-0.0081) = 0.238; c = 0.7*1 + 0.3*0.238 = 0.771; halved -> 0.386
    assert abs(new["correction"][3] - 0.386) < 0.01
    assert new["edge_attempts"][(3, 7)] == 1


def test_the_correction_never_leaves_its_bounds():
    st = {"correction": {3: 0.1}, "edge_attempts": {}, "f_kT": {}, "wall_time": []}
    new = update_after_topup(st, _plan(), realised_sigma={3: 0.20})          # no improvement at all
    assert new["correction"][3] >= 0.1
