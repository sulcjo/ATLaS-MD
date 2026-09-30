"""Respring (--ap-cv2-respring): re-derive a CV2 window's k2 from its own production samples.

Measurement (F''_prod, c_real and their block-bootstrap interval), the interval trigger, the
spring and its refusals, the per-epoch cap, the atomic applier action (new centre on every rung,
old centre retired, net 0, Hamiltonians unchanged), the k2-aware duplicate check, seeding, the
ladder_adapt grouping, the 3.7 summary/grade, the epoch-loop hook (flag only, never on a
recovered epoch) and the policy/CLI/YAML wiring.
"""
import json
import math
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gareus.adaptive_production as ap  # noqa: E402
import gareus.production as prod  # noqa: E402
from gareus.adaptive import cv2_respring as rs  # noqa: E402
from gareus.adaptive import cv2_respring_io as rio  # noqa: E402
from gareus.adaptive import cv2_resolution_grade as grade  # noqa: E402
from gareus.adaptive import cv2_resolution_summary as summ  # noqa: E402
from gareus.adaptive import ladder_adapt as la  # noqa: E402
from gareus.cli import parse_args  # noqa: E402
from gareus.swarm.ladder_design import R_KCAL_MOL_K  # noqa: E402

RT = R_KCAL_MOL_K * 300.0
LADDER = (0.0, 0.3, 1.0)


def _ar1(n, var, rho, seed):
    rng = np.random.default_rng(seed)
    x = np.empty(n)
    x[0] = rng.normal() * math.sqrt(var)
    e = rng.normal(size=n) * math.sqrt(var * (1 - rho * rho))
    for i in range(1, n):
        x[i] = rho * x[i - 1] + e[i]
    return x


# ---- measurement -------------------------------------------------------------------------------

@pytest.mark.parametrize("f2_true", [0.5, 3.0])
def test_f2_prod_and_c_real_recover_a_known_landscape_under_a_known_spring(f2_true):
    k2 = 1.18
    var_true = RT / (k2 + f2_true)
    z = np.random.default_rng(1).normal(0.3, math.sqrt(var_true), 20000)
    m = rs.measure_window(var=float(np.var(z)), n=z.size, k2=k2, rt=RT, z=z, g=1.0)
    assert m["f2_prod"] == pytest.approx(f2_true, rel=0.05)
    assert m["c_real"] == pytest.approx(k2 / (k2 + f2_true), rel=0.03)
    lo, hi = m["c_interval"]
    assert lo < k2 / (k2 + f2_true) < hi
    assert m["f2_interval"][0] < f2_true < m["f2_interval"][1]
    assert m["n_eff"] == pytest.approx(z.size)


def test_block_bootstrap_widens_with_the_autocorrelation_time_and_scales_the_full_series_var():
    k2 = 1.18
    z = _ar1(8000, RT / (k2 + 2.0), 0.95, seed=3)
    iid = rs.measure_window(var=float(np.var(z)), n=z.size, k2=k2, rt=RT, z=z, g=1.0)
    corr = rs.measure_window(var=float(np.var(z)), n=z.size, k2=k2, rt=RT, z=z, g=39.0)
    w = lambda m: m["c_interval"][1] - m["c_interval"][0]
    assert w(corr) > 1.8 * w(iid)         # 5-row blocks already catch part of rho = 0.95
    assert corr["n_eff"] == pytest.approx(8000 / 39.0)
    # the interval is a ratio applied to the FULL-series variance (here 2x the subsample's)
    scaled = rs.measure_window(var=2 * float(np.var(z)), n=z.size, k2=k2, rt=RT, z=z, g=39.0)
    assert scaled["c_interval"][0] == pytest.approx(2 * corr["c_interval"][0])


def test_subsample_g_is_used_in_raw_spacing_when_x5_is_missing():
    z = _ar1(4000, 0.1, 0.9, seed=5)
    m = rs.measure_window(var=float(np.var(z)), n=8000, k2=1.0, rt=RT, z=z, stride=2, g=None)
    assert m["g_source"] == "subsample" and m["g"] > 2 * 10     # ~19 subsample rows x stride 2
    assert m["bootstrap"]["block_len"] == math.ceil(rs.BOOT_BLOCK_G_MULTIPLE * m["g"] / 2)


def test_blocks_never_cross_a_source():
    ids = rs.block_ids(10, np.array([0, 0, 0, 1, 1, 1, 1, 2, 2, 2]), 2)
    src = np.array([0, 0, 0, 1, 1, 1, 1, 2, 2, 2])
    for b in np.unique(ids):
        assert len(set(src[ids == b])) == 1


# ---- decision ----------------------------------------------------------------------------------

def _view(sid=0, k2=1.18, **kw):
    v = {"state_id": sid, "c1": 0.2, "k1": 500.0, "c2": 0.5, "k2": k2, "lam": 0.0, "created_epoch": 0,
         "metadata": {}, "members": [sid], "mandatory": False}
    v.update(kw)
    return v


def _m(c_lo, c_hi, k2=1.18, n_eff=1000.0, c=None):
    c = (c_lo + c_hi) / 2 if c is None else c
    var = c * RT / k2
    f2 = RT / var - k2
    return {"n": 5000, "var": var, "sd": math.sqrt(var), "c_real": c, "f2_prod": f2, "n_eff": n_eff,
            "c_interval": [c_lo, c_hi], "f2_interval": [RT / (c_hi * RT / k2) - k2, RT / (c_lo * RT / k2) - k2]}


S = rs.RespringSettings()


def test_whole_interval_below_the_trigger_proposes_the_compression_floor_spring():
    cand = rs.evaluate(_view(), _m(0.30, 0.40), S, epoch=1, touched=[])
    assert cand["decision"] == "proposed" and cand["triggered"]
    p = cand["proposal"]
    f2 = cand["metrics"]["f2_prod"]
    assert p["k2_new"] == pytest.approx(f2)                  # sampled-sd target: k2' = F''_prod
    assert p["predicted_c"] == pytest.approx(0.5)
    assert p["sigma_target_source"] == "sampled_sd"


def test_interval_straddling_the_trigger_is_no_action():
    cand = rs.evaluate(_view(), _m(0.40, 0.47, c=0.43), S, epoch=1, touched=[])   # point below, q95 above 0.45
    assert cand["decision"] == "no_action" and cand["reason"] == "compression_reached"


def test_nonpositive_curvature_is_recorded_never_acted_on():
    m = _m(1.05, 1.2, c=1.1)
    assert m["f2_prod"] < 0
    cand = rs.evaluate(_view(), m, S, epoch=1, touched=[])
    assert cand["decision"] == "no_action" and cand["reason"] == "f2_nonpositive"


def test_skips_and_refusals():
    assert rs.evaluate(_view(), _m(0.2, 0.3, n_eff=50.0), S, epoch=1, touched=[])["reason"] == \
        "too_few_effective_samples"
    assert rs.evaluate(_view(k2=0.0), _m(0.2, 0.3), S, epoch=1, touched=[])["reason"] == "cv2_unrestrained"
    assert rs.evaluate(_view(mandatory=True), _m(0.2, 0.3), S, epoch=1, touched=[])["reason"] == "mandatory"
    assert rs.evaluate(_view(members=[0, 5]), _m(0.2, 0.3), S, epoch=1, touched=[5])["reason"] == \
        "touched_by_other_action"
    done = _view(metadata={rs.METADATA_KEY: {"rule": rs.RULE}}, created_epoch=0)
    assert rs.evaluate(done, _m(0.2, 0.3), S, epoch=9, touched=[])["reason"] == "already_respringed"
    young = _view(metadata={rs.METADATA_KEY: {"rule": "R1"}}, created_epoch=1)
    assert rs.evaluate(young, _m(0.2, 0.3), S, epoch=2, touched=[])["reason"] == "protected"
    assert rs.evaluate(young, _m(0.2, 0.3), S, epoch=3, touched=[])["decision"] == "proposed"
    capped = rs.RespringSettings(k2_max=1.5)
    cand = rs.evaluate(_view(), _m(0.2, 0.3), capped, epoch=1, touched=[])
    assert cand["decision"] == "refused" and cand["refusal"] == "k2_capped_below_compression"
    # a spring within respring_k2_rtol of the old one: no_change (tolerance 0 so c 0.476 triggers)
    tight = rs.RespringSettings(respring_tolerance=0.0)
    cand = rs.evaluate(_view(), _m(0.47, 0.49, c=1.18 / (1.18 + 1.08 * 1.18)), tight, epoch=1, touched=[])
    assert cand["refusal"] == "no_change"


def test_coupling_gate_lowering_below_the_floor_refuses():
    class Gate:
        def gate(self, c1, k1, c2, k2, context=""):
            return 0.5 * k2, "lowered"
    cand = rs.evaluate(_view(), _m(0.2, 0.3), S, epoch=1, touched=[], gate=Gate())
    assert cand["refusal"] == "k2_capped_below_compression" and cand["proposal"]["gate_note"] == "lowered"


def test_per_epoch_cap_keeps_the_most_under_compressed_and_defers_the_rest():
    cands = [rs.evaluate(_view(sid=i), _m(0.1 + 0.02 * i, 0.2 + 0.02 * i), S, epoch=1, touched=[]) for i in range(6)]
    out, cap = rs.apply_cap(cands, n_centres=8, settings=S)
    assert cap["limit"] == 2
    assert [c["state_id"] for c in out if c["decision"] == "proposed"] == [0, 1]
    deferred = [c for c in out if c["decision"] == "deferred"]
    assert len(deferred) == 4 and all(c["refusal"] == "per_epoch_cap" for c in deferred)
    assert rs.summarise(out)["n_unresolved"] == 4


# ---- applier ------------------------------------------------------------------------------------

def _grid(lambdas=LADDER, k2=1.18):
    reg = ap.WindowStateRegistry()
    for c1 in (0.1, 0.2):
        for c2 in (-1.0, 0.0, 1.0):
            for lam in lambdas:
                reg.add_state(c1, 500.0, secondary_center=c2, secondary_k=k2, gamd_lambda=lam, epoch=0, source="seed")
    return reg


def _centre(reg, c1, c2, active=None):
    return [s for s in reg.all_states() if s.secondary_center is not None and abs(s.primary_center - c1) < 1e-9
            and abs(s.secondary_center - c2) < 1e-9 and (active is None or s.active == active)]


def _respring_action(reg, sid, k2_new, epoch=0):
    st = reg.get_state(sid)
    cand = {"state_id": sid, "c1": st.primary_center, "k1": st.primary_k, "c2": st.secondary_center,
            "reason": "under-compressed", "metrics": {"f2_prod": k2_new, "c_real": 0.3},
            "proposal": {"k2_old": st.secondary_k, "k2_new": k2_new, "predicted_c": 0.5}}
    return rs.action_of(cand, epoch)


def test_respring_adds_the_new_centre_on_every_rung_and_retires_the_old_one():
    reg = _grid()
    old = [s for s in _centre(reg, 0.2, 0.0)]
    before = {s.state_id: (s.primary_center, s.primary_k, s.secondary_center, s.secondary_k, s.gamd_lambda)
              for s in reg.all_states()}
    n0 = len(reg.active_states())
    action = _respring_action(reg, old[0].state_id, 2.4)
    ledger_form = tuple(json.loads(json.dumps(list(action))))       # what the P3 ledger replays
    refused = ap._apply_registry_actions(reg, [ledger_form], 0, policy=ap.AdaptiveDecisionPolicy())
    assert refused == []
    assert len(reg.active_states()) == n0                              # budget-neutral
    assert all(not s.active and s.usable_for_mbar for s in old)       # old stays in the union MBAR
    new = [s for s in _centre(reg, 0.2, 0.0, active=True)]
    assert sorted(s.gamd_lambda for s in new) == list(LADDER)
    assert all(s.secondary_k == 2.4 and s.primary_k == 500.0 and s.parent_state_id == old[0].state_id
               and s.source == "adaptive_production_cv2_respring" for s in new)
    assert all(s.metadata[rs.METADATA_KEY]["seed_source_state_id"] == old[0].state_id for s in new)
    for sid, ham in before.items():                                    # no existing Hamiltonian changed
        s = reg.get_state(sid)
        assert (s.primary_center, s.primary_k, s.secondary_center, s.secondary_k, s.gamd_lambda) == ham


def test_respring_accepts_a_cv1_free_axis_window_and_refuses_cv2_unrestrained_and_mandatory():
    reg = _grid(lambdas=(0.0,))
    x7 = reg.add_state(0.5, 0.0, secondary_center=0.3, secondary_k=1.18, epoch=0, source="seed",
                       metadata={"state_role": "axis"})
    cv1_only = reg.add_state(0.3, 500.0, epoch=0, source="seed")
    mand = _centre(reg, 0.1, 1.0)[0]
    mand.metadata["mandatory"] = True
    acts = [_respring_action(reg, x7.state_id, 2.0),
            ("respring", cv1_only.state_id, (0.3, 500.0, 0.0, 2.0), "cv2_respring: x", {}),
            _respring_action(reg, mand.state_id, 2.0)]
    refused = ap._apply_registry_actions(reg, acts, 0, policy=ap.AdaptiveDecisionPolicy())
    assert [(r["index"], r["reason"]) for r in refused] == [(1, "anchor_or_axis"), (2, "mandatory")]
    assert not x7.active
    kid = [s for s in reg.active_states() if s.parent_state_id == x7.state_id]
    assert len(kid) == 1 and kid[0].primary_k == 0.0 and kid[0].secondary_k == 2.0


def test_respring_refusals_change_nothing():
    reg = _grid(lambdas=(0.0,))
    sid = _centre(reg, 0.2, 0.0)[0].state_id
    ap._apply_registry_actions(reg, [_respring_action(reg, sid, 2.4)], 0, policy=ap.AdaptiveDecisionPolicy())
    snap = json.dumps(reg.to_dict(), sort_keys=True)
    new_id = _centre(reg, 0.2, 0.0, active=True)[0].state_id
    acts = [_respring_action(reg, new_id, 1.2),                 # back to the retired k2: a duplicate
            _respring_action(reg, new_id, 2.5),                 # within 10 %: no_change
            ("respring", sid, (0.2, 500.0, 0.0, 3.0), "cv2_respring: x", {})]   # retired parent
    refused = ap._apply_registry_actions(reg, acts, 1, policy=ap.AdaptiveDecisionPolicy())
    assert [r["reason"] for r in refused] == ["duplicate", "no_change", "inactive"]
    assert json.dumps(reg.to_dict(), sort_keys=True) == snap


def test_duplicate_check_is_k2_aware_only_when_asked():
    reg = _grid(lambdas=(0.0,))
    pol = ap.AdaptiveDecisionPolicy()
    assert reg.has_near_duplicate(0.2, 0.0, pol, primary_k=500.0, secondary_k=3.0)            # today's rule
    assert not reg.has_near_duplicate(0.2, 0.0, pol, primary_k=500.0, secondary_k=3.0, k2_rtol=0.1)
    assert reg.has_near_duplicate(0.2, 0.0, pol, primary_k=500.0, secondary_k=1.25, k2_rtol=0.1)


def test_respring_child_seeds_from_the_old_state_frames(tmp_path):
    reg = _grid(lambdas=(0.0,))
    old = _centre(reg, 0.2, 0.0)[0]
    near = _centre(reg, 0.2, 1.0)[0]
    ap._apply_registry_actions(reg, [_respring_action(reg, old.state_id, 2.4)], 0, policy=ap.AdaptiveDecisionPolicy())
    child = _centre(reg, 0.2, 0.0, active=True)[0]
    bank = tmp_path / "bank"
    bank.mkdir()
    with (bank / "final_survivor_seeds.csv").open("w") as fh:
        fh.write("seed_name,survivor_pdb_path,source_state_id,primary_cv_value,secondary_cv_value\n")
        fh.write(f"o.pdb,{bank / 'o.pdb'},{old.state_id},0.2,0.6\n")
        fh.write(f"n.pdb,{bank / 'n.pdb'},{near.state_id},0.2,0.01\n")
    out = ap.select_state_aware_seeds_for_targets(bank, reg)
    assert {a["target_state_id"]: a["seed_name"] for a in out["assignments"]}[child.state_id] == "o.pdb"


def test_ladder_adapt_never_pools_a_resprung_centre_with_its_retired_one():
    a = {"primary_center": 0.2, "primary_k": 500.0, "secondary_center": 0.0, "secondary_k": 1.18}
    assert la._centre_spring_key(a) == la._centre_spring_key(dict(a))
    assert la._centre_spring_key(a) != la._centre_spring_key({**a, "secondary_k": 2.4})
    assert la._centre_spring_key(a)[:4] == la.centre_key(a)


# ---- orchestration -------------------------------------------------------------------------------

def _payload(reg, var_by_sid, n=4000, seed=0):
    states, subs = [], {}
    rng = np.random.default_rng(seed)
    for s in reg.active_states():
        if s.secondary_center is None or s.state_id not in var_by_sid:
            states.append({"state_id": s.state_id})
            continue
        z = rng.normal(s.secondary_center, math.sqrt(var_by_sid[s.state_id]), n)
        states.append({"state_id": s.state_id, "paired_cv": {
            "cv2": {"n": n, "mean": float(z.mean()), "var": float(np.var(z))},
            "cv1": {"n": n, "var": 1e-4}, "cov_cv1_cv2": 0.0, "subsample": {"stride": 1}}})
        subs[s.state_id] = {"cv2": z, "source_index": np.zeros(n, dtype=np.int64), "step": np.arange(n) * 250}
    return {"states": states, "edges": []}, subs


def test_propose_respring_end_to_end_on_a_ladder():
    reg = _grid()
    lam0 = [s for s in reg.active_states() if s.gamd_lambda == 0.0]
    stiff, fine = lam0[0], lam0[1]
    var = {stiff.state_id: RT / (1.18 + 4.0), fine.state_id: RT / (2 * 1.18)}      # c 0.23 vs 0.5
    diag, subs = _payload(reg, var)
    acts, report = rio.propose_respring(reg, diag, [("extend", 99, "x")], S, epoch=1, effective={}, subsamples=subs)
    by = {c["state_id"]: c for c in report["candidates"]}
    assert by[stiff.state_id]["decision"] == "proposed"
    assert by[fine.state_id]["decision"] == "no_action"
    assert len(by) == len(lam0)                          # representative rung only
    assert acts[0] == ("extend", 99, "x") and [a[0] for a in acts[1:]] == ["respring"]
    assert acts[1][2][3] == pytest.approx(4.0, rel=0.1)
    assert report["schema_version"] == rs.SCHEMA_VERSION and report["summary"]["n_proposed"] == 1
    json.dumps(report)                                   # JSON-safe
    refused = ap._apply_registry_actions(reg, acts[1:], 1, policy=ap.AdaptiveDecisionPolicy())
    assert refused == [] and all(not s.active for s in _centre(reg, stiff.primary_center, stiff.secondary_center)
                                 if s.secondary_k == 1.18)


def test_another_actions_centre_is_skipped_not_overridden():
    reg = _grid()
    s0 = next(s for s in reg.active_states() if s.gamd_lambda == 0.0)
    rung = next(s for s in _centre(reg, s0.primary_center, s0.secondary_center) if s.gamd_lambda > 0)
    diag, subs = _payload(reg, {s0.state_id: RT / (1.18 + 4.0)})
    insert = ("insert", rung.state_id, [(0.2, 500.0, 0.4, 3.0)], "cv2_resolution R3: x", {})
    acts, report = rio.propose_respring(reg, diag, [insert], S, epoch=1, effective={}, subsamples=subs)
    assert acts == [insert]
    assert report["candidates"][0]["reason"] == "touched_by_other_action"


def test_hook_writes_the_report_and_never_raises(tmp_path, monkeypatch):
    reg = _grid(lambdas=(0.0,))
    epoch_dir = tmp_path / "epoch_001"
    epoch_dir.mkdir()
    diag, _subs = _payload(reg, {reg.active_states()[0].state_id: RT / 5.0})
    pol = ap.AdaptiveDecisionPolicy(cv2_respring=True)
    args = Namespace(temperature_k=300.0)
    out = rio.run_epoch_cv2_respring(adaptive_dir=tmp_path, epoch_dir=epoch_dir, epoch=1, registry=reg,
                                     diagnostics=diag, actions=[], policy=pol, args=args, out_dir=tmp_path,
                                     phase_dirs=[epoch_dir])
    rep = json.loads((epoch_dir / rs.REPORT_NAME).read_text())
    assert out == [] and rep["status"] == "ok"
    assert rep["candidates"][0]["reason"] == "no_subsample"          # no P4 NPZ: never act without an interval
    monkeypatch.setattr(rio, "propose_respring", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    acts = [("extend", 0, "x")]
    assert rio.run_epoch_cv2_respring(adaptive_dir=tmp_path, epoch_dir=epoch_dir, epoch=1, registry=reg,
                                      diagnostics=diag, actions=acts, policy=pol, args=args, out_dir=tmp_path,
                                      phase_dirs=[epoch_dir]) == acts
    assert json.loads((epoch_dir / rs.REPORT_NAME).read_text())["status"] == "error"


def test_layout_sigma_target_is_used_when_recorded():
    pol = ap.AdaptiveDecisionPolicy()
    plan = {"cv2_shape": {"sigma_w_target": 0.2, "columns": [{"placement": {"min_mean_compression": 0.5}}]}}
    s = rio.settings_from(pol, Namespace(temperature_k=300.0), plan, "lp.json")
    assert s.sigma_target == 0.2 and "layout_plan" in s.sigma_target_source
    cand = rs.evaluate(_view(), _m(0.2, 0.3), s, epoch=1, touched=[])
    f2 = cand["metrics"]["f2_prod"]
    assert cand["proposal"]["k2_new"] == pytest.approx(max(RT / 0.04 - f2, f2))


def test_applier_refusals_are_annotated_and_counted_idempotently(tmp_path):
    (tmp_path / rs.REPORT_NAME).write_text(json.dumps({"summary": {"n_unresolved": 1},
                                                       "candidates": [{"triggered": True, "decision": "deferred"}]}))
    acts = [("extend", 1, "x"), ("respring", 2, (0.2, 500.0, 0.0, 2.0), "cv2_respring: y", {})]
    for _ in range(2):
        rio.annotate_report_with_refusals(tmp_path, acts, [{"index": 1, "reason": "duplicate", "action": "respring"}])
    rep = json.loads((tmp_path / rs.REPORT_NAME).read_text())
    assert rep["apply"]["refused"][0]["reason"] == "duplicate" and rep["summary"]["n_unresolved"] == 2


# ---- policy, CLI, YAML, help ----------------------------------------------------------------------

def test_policy_validation_fails_at_construction():
    for bad in ({"respring_tolerance": 0.6}, {"respring_max_fraction": 1.5}, {"respring_k2_rtol": 0.0},
                {"respring_min_neff": -1.0}):
        with pytest.raises(ValueError):
            ap.AdaptiveDecisionPolicy(**bad)
    for f in ("cv2_respring", "respring_min_neff", "respring_tolerance", "respring_max_fraction", "respring_k2_rtol"):
        assert f in ap.DECISION_SETTINGS_FIELDS


def test_cli_and_yaml_reach_the_policy(tmp_path):
    args = parse_args(["--seq", "GYDPETGTWG", "--out", str(tmp_path / "r"), "--ap-cv2-respring",
                       "--ap-respring-min-neff", "150", "--ap-respring-tolerance", "0.02",
                       "--ap-respring-max-fraction", "0.5", "--ap-respring-k2-rtol", "0.2"])
    p = ap.policy_from_args(args)
    assert (p.cv2_respring, p.respring_min_neff, p.respring_tolerance, p.respring_max_fraction,
            p.respring_k2_rtol) == (True, 150.0, 0.02, 0.5, 0.2)
    cfg = tmp_path / "c.yaml"
    cfg.write_text("adaptive_production:\n  ap_cv2_respring: true\n  ap_respring_min_neff: 300\n")
    p = ap.policy_from_args(parse_args(["--config", str(cfg), "--seq", "GYDPETGTWG", "--out", str(tmp_path / "r2")]))
    assert p.cv2_respring is True and p.respring_min_neff == 300.0
    p = ap.policy_from_args(parse_args(["--seq", "GYDPETGTWG", "--out", str(tmp_path / "r3")]))
    assert p.cv2_respring is False and p.respring_min_neff == 200.0


def test_help_documents_respring():
    from gareus.helptext import _METHOD_ENCYCLOPEDIA
    assert "--ap-cv2-respring" in _METHOD_ENCYCLOPEDIA and "cv2_respring_report.json" in _METHOD_ENCYCLOPEDIA


# ---- 3.7 summary and grade -------------------------------------------------------------------------

def test_summary_carries_respring_counts_and_the_row_grades_unresolved_caution():
    diag = {"states": [], "edges": []}
    plain = summ.build_summary(diag, None, label="epoch_001")
    assert "respring" not in plain and "n_respring_unresolved" not in plain["counts"]
    rep = {"status": "ok", "epoch": 1, "summary": {"n_proposed": 2, "n_unresolved": 3}}
    out = summ.build_summary(diag, {"status": "ok", "candidates": [], "summary": {}}, label="epoch_001", respring=rep)
    assert out["counts"]["n_respring_proposed"] == 2 and out["counts"]["n_respring_unresolved"] == 3
    row = grade.check_cv2_resolution({"cv2_resolution": out})
    assert row["status"] == grade.CAUTION and "un-resprung" in row["detail"]


# ---- the real epoch loop -------------------------------------------------------------------------

class _ReachedFinal(Exception):
    pass


class _Crash(Exception):
    pass


def _campaign(tmp_path, flag):
    out = tmp_path / "run"
    adaptive = out / "adaptive_production"
    adaptive.mkdir(parents=True)
    reg = ap.WindowStateRegistry()
    for c in (0.1, 0.2, 0.3):
        reg.add_state(c, 800.0, secondary_center=0.0, secondary_k=1.0, epoch=0, source="seed")
    reg.save(adaptive)
    reg.write_active_window_csv(adaptive / "windows_epoch_000.csv")
    args = parse_args(["--seq", "GYDPETGTWG", "--out", str(out)] + (["--ap-cv2-respring"] if flag else []))
    args.adaptive_production_resume = True
    args.adaptive_production_allocation_scheduler = False
    args.adaptive_production_epochs = 1
    args.adaptive_production_epoch_steps = 1000
    args.adaptive_production_global_shared_gamd = False
    return args, out, adaptive


def _drive(monkeypatch, args, out, calls):
    def fake_run_gareus(_args, run_dir, *rest, **kw):
        if "final" in str(run_dir):
            raise _ReachedFinal()
    real_hook = rio.run_epoch_cv2_respring

    def counting_hook(**kw):
        calls["hook"] += 1
        return real_hook(**kw)
    monkeypatch.setattr(prod, "run_gareus", fake_run_gareus)
    monkeypatch.setattr(ap, "propose_actions_from_diagnostics", lambda registry, diagnostics, **kw: [])
    monkeypatch.setattr(ap, "collect_segmented_epoch_diagnostics", lambda *a, **k: {"states": [], "edges": []})
    monkeypatch.setattr(ap, "_assert_epoch_has_samples", lambda *a, **k: None)
    monkeypatch.setattr(rio, "run_epoch_cv2_respring", counting_hook)
    with pytest.raises(_ReachedFinal):
        ap.run_adaptive_production_auto_loop(args, out, None, None, None, None, None, None)


@pytest.mark.parametrize("flag", [False, True])
def test_epoch_loop_calls_the_hook_only_with_the_flag(tmp_path, monkeypatch, flag):
    args, out, adaptive = _campaign(tmp_path, flag)
    calls = {"hook": 0}
    _drive(monkeypatch, args, out, calls)
    assert calls["hook"] == (1 if flag else 0)
    assert (adaptive / "epoch_000" / rs.REPORT_NAME).exists() == flag
    settings = json.loads((adaptive / ap.DECISION_SETTINGS_FILENAME).read_text())["settings"]
    assert settings["cv2_respring"] is flag


def test_recovered_epoch_never_runs_the_hook_again(tmp_path, monkeypatch):
    args, out, adaptive = _campaign(tmp_path, True)
    calls = {"hook": 0}
    armed = {"on": True}
    real_pool = ap._write_runtime_pool_reports

    def pool_reports(*a, **k):                 # killed right after registry.save + the P3 ledger
        if armed["on"] and calls["hook"] >= 1:
            armed["on"] = False
            raise _Crash()
        return real_pool(*a, **k)
    monkeypatch.setattr(ap, "_write_runtime_pool_reports", pool_reports)
    with pytest.raises(_Crash):
        _drive(monkeypatch, args, out, calls)
    assert (adaptive / "epoch_000" / ap.APPLIED_ACTIONS_FILENAME).exists()
    (adaptive / ".gareus_run.lock").unlink(missing_ok=True)
    _drive(monkeypatch, args, out, calls)
    assert calls["hook"] == 1


# ---- off path --------------------------------------------------------------------------------------

RESPRING_BASE = "983ba22"          # the branch before respring
RESPRING_FIELDS = {"cv2_respring": False, "respring_min_neff": 200.0, "respring_tolerance": 0.05,
                   "respring_max_fraction": 0.25, "respring_k2_rtol": 0.10}


def test_off_path_is_identical_to_the_pre_respring_commit_except_the_new_off_keys(tmp_path, monkeypatch):
    import test_cv2_resolution_wiring as w
    monkeypatch.setattr(w, "BASE_COMMIT", RESPRING_BASE)
    base = w._base_module(tmp_path)
    old = w._run_off_path(base, tmp_path / "old")
    new = w._run_off_path(ap, tmp_path / "new")
    for key in ("report", "gate"):
        stamp, base_stamp = new[key].pop("policy"), old[key].pop("policy")
        assert {k: stamp[k] for k in RESPRING_FIELDS} == RESPRING_FIELDS
        assert {k: stamp[k] for k in base_stamp} == base_stamp and set(RESPRING_FIELDS) <= set(stamp) - set(base_stamp)
        new[key].pop("diagnostics_json", None), old[key].pop("diagnostics_json", None)
    assert w._strip_volatile(json.loads(json.dumps(old, default=str))) == \
        w._strip_volatile(json.loads(json.dumps(new, default=str)))
    assert not list((tmp_path / "new").rglob(rs.REPORT_NAME))
    assert set(RESPRING_FIELDS) <= set(ap.DECISION_SETTINGS_FIELDS) - set(base.DECISION_SETTINGS_FIELDS)
