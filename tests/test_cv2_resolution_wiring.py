"""Spec 3.3 wiring: the off path is byte-identical, flags/YAML reach the policy, the epoch loop
calls R1-R3 only with the flag (never again on a recovered epoch), the insert applier, seeding.
"""
import importlib.util
import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gareus.adaptive_production as ap  # noqa: E402
import gareus.production as prod  # noqa: E402
from gareus.adaptive import cv2_resolution as cr  # noqa: E402
from gareus.adaptive import cv2_resolution_io as cio  # noqa: E402
from gareus.cli import parse_args  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
BASE_COMMIT = "1830f85"          # the branch before spec 3.3


def _base_module(tmp_path):
    """gareus.adaptive_production as it was at BASE_COMMIT (skip when git cannot show it)."""
    try:
        src = subprocess.run(["git", "show", f"{BASE_COMMIT}:gareus/adaptive_production.py"], cwd=REPO,
                             capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("base commit not available")
    path = tmp_path / "base_adaptive_production.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location("gareus._base_adaptive_production", path)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "gareus"
    sys.modules[spec.name] = mod          # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(mod)
    return mod


def _fixture(mod):
    reg = mod.WindowStateRegistry()
    for c1 in (0.2, 0.4):
        for c2 in (-1.0, 0.0, 1.0):
            reg.add_state(c1, 800.0, secondary_center=c2, secondary_k=5.0, epoch=0, source="seed")
    ids = [s.state_id for s in reg.active_states()]
    states = [{"state_id": s, "sample_count": 5000 if s != ids[-1] else 10, "warnings": []} for s in ids]
    edges = [{"state_i": ids[0], "state_j": ids[1], "edge_type": "secondary_chain", "overlap": 0.05,
              "exchange_acceptance": 0.3, "mbar_overlap": None, "warnings": []},
             {"state_i": ids[1], "state_j": ids[2], "edge_type": "secondary_chain", "overlap": 0.6,
              "exchange_acceptance": 0.3, "mbar_overlap": None, "warnings": []}]
    return reg, {"states": states, "edges": edges}


def _run_off_path(mod, tmp_path):
    reg, diag = _fixture(mod)
    policy = mod.AdaptiveDecisionPolicy()
    plan = []
    actions = mod.propose_actions_from_diagnostics(reg, diag, policy=policy, temperature_K=300.0, bridge_plan_out=plan)
    epoch_dir = tmp_path / "epoch_000"
    epoch_dir.mkdir(parents=True)
    report = mod.write_epoch_action_report(epoch_dir, 0, reg, diag, actions, policy, bridge_plan=plan)
    gate = mod.evaluate_adaptive_convergence_gate(epoch_dir, 0, reg, diag, actions, policy)
    refused = mod._apply_registry_actions(reg, actions, 0, policy=policy)
    for p in (report, gate):
        for key in ("json", "markdown", "md", "path"):
            p.pop(key, None)
    return {"actions": [list(a) for a in actions], "registry": reg.to_dict(), "refused": refused,
            "report": json.loads((epoch_dir / "adaptive_epoch_actions.json").read_text()),
            "gate": json.loads((epoch_dir / "adaptive_convergence_gate.json").read_text()),
            "converged": mod._adaptive_production_converged(actions, diag, policy),
            "action_dicts": [mod._action_to_dict(a) for a in actions]}


def _strip_volatile(obj):
    if isinstance(obj, dict):
        return {k: _strip_volatile(v) for k, v in obj.items()
                if k not in ("created_unix", "written_unix", "timestamp", "generated_unix", "epoch_dir")}
    if isinstance(obj, list):
        return [_strip_volatile(v) for v in obj]
    return obj


NEW_FIELDS = {"cv2_resolution": False, "coverage_min_windows": 2.0, "refine_min_transitions": 10,
              "refine_pmf_sigma_kT": 0.5, "refine_budget_fraction": 0.5, "refine_protect_epochs": 2,
              "refine_min_sigma": 0.1}


def test_off_path_proposer_applier_reports_are_identical_to_the_base_commit(tmp_path):
    """Actions, applied registry, refusals, convergence and both reports are what the base
    commit produces; the reports' full policy stamp gains exactly the new keys, all off."""
    base = _base_module(tmp_path)
    old = _run_off_path(base, tmp_path / "old")
    new = _run_off_path(ap, tmp_path / "new")
    for key in ("report", "gate"):
        stamp, base_stamp = new[key].pop("policy"), old[key].pop("policy")
        assert {k: stamp[k] for k in NEW_FIELDS} == NEW_FIELDS          # present, all off
        assert {k: stamp[k] for k in base_stamp} == base_stamp          # the base's own keys unchanged
        new[key].pop("diagnostics_json", None), old[key].pop("diagnostics_json", None)
    assert _strip_volatile(json.loads(json.dumps(old, default=str))) == \
        _strip_volatile(json.loads(json.dumps(new, default=str)))
    assert old["actions"] and not list((tmp_path / "new").rglob(cr.REPORT_NAME))


def test_off_path_policy_and_decision_settings_only_gain_the_new_off_keys(tmp_path):
    base = _base_module(tmp_path)
    args = parse_args(["--seq", "GYDPETGTWG", "--out", str(tmp_path / "run")])
    old_p, new_p = base.policy_from_args(args), ap.policy_from_args(args)
    new_fields = set(NEW_FIELDS)
    assert {f: getattr(old_p, f) for f in base.DECISION_SETTINGS_FIELDS} == \
        {f: getattr(new_p, f) for f in base.DECISION_SETTINGS_FIELDS}
    assert new_fields <= set(ap.DECISION_SETTINGS_FIELDS) - set(base.DECISION_SETTINGS_FIELDS)
    assert new_p.cv2_resolution is False
    # A live campaign's record (written by the base code) resumes untouched; the new keys are
    # recorded from the job that first sees them, with the flag off.
    adaptive = tmp_path / "adaptive"
    adaptive.mkdir()
    base._resolve_decision_settings(adaptive, base.replace(old_p, min_rung_overlap=0.2))
    recorded = json.loads((adaptive / ap.DECISION_SETTINGS_FILENAME).read_text())["settings"]
    policy, record = ap._resolve_decision_settings(adaptive, new_p)
    assert policy.min_rung_overlap == 0.2 and policy.cv2_resolution is False
    assert {k: record["settings"][k] for k in recorded} == recorded
    assert {k: record["settings"][k] for k in new_fields} == NEW_FIELDS


def test_cli_flags_and_yaml_reach_the_policy(tmp_path):
    args = parse_args(["--seq", "GYDPETGTWG", "--out", str(tmp_path / "r"), "--ap-cv2-resolution",
                       "--ap-coverage-min-windows", "3", "--ap-refine-min-transitions", "4",
                       "--ap-refine-pmf-sigma-kt", "0.3", "--ap-refine-budget-fraction", "0.25",
                       "--ap-refine-protect-epochs", "1", "--ap-refine-min-sigma", "0.05"])
    p = ap.policy_from_args(args)
    assert (p.cv2_resolution, p.coverage_min_windows, p.refine_min_transitions, p.refine_pmf_sigma_kT,
            p.refine_budget_fraction, p.refine_protect_epochs, p.refine_min_sigma) == (True, 3.0, 4, 0.3, 0.25, 1, 0.05)
    cfg = tmp_path / "c.yaml"
    cfg.write_text("adaptive_production:\n  ap_cv2_resolution: true\n  ap_refine_min_transitions: 7\n")
    args = parse_args(["--config", str(cfg), "--seq", "GYDPETGTWG", "--out", str(tmp_path / "r2")])
    p = ap.policy_from_args(args)
    assert p.cv2_resolution is True and p.refine_min_transitions == 7


def test_help_topic_documents_the_flags():
    from gareus.helptext import _METHOD_ENCYCLOPEDIA
    assert "--ap-cv2-resolution" in _METHOD_ENCYCLOPEDIA and "trapped_or_orthogonal" in _METHOD_ENCYCLOPEDIA


# ---- insert applier --------------------------------------------------------------------------

def _reg(lambdas=(0.0, 1.0)):
    reg = ap.WindowStateRegistry()
    for c2 in (-1.0, 0.0, 1.0):
        for lam in lambdas:
            reg.add_state(0.2, 800.0, secondary_center=c2, secondary_k=5.0, gamd_lambda=lam, epoch=0, source="s")
    reg.add_state(0.2, 0.0, secondary_center=0.0, secondary_k=5.0, epoch=0, source="axis")   # CV2-only axis state
    return reg


def test_insert_keeps_the_parent_centre_and_adds_children_on_every_rung():
    reg = _reg()
    parent = next(s for s in reg.active_states() if s.secondary_center == 0.0 and s.primary_k > 0)
    parent.metadata["mandatory"] = True                     # an insert retires nothing: allowed
    meta = cr.action_metadata("R3", None, 0, parent.state_id)
    action = ("insert", parent.state_id, [(0.2, 800.0, -0.4, 6.0), (0.2, 800.0, 0.4, 6.0)], "cv2_resolution R3: x", meta)
    ledger_form = tuple(json.loads(json.dumps(list(action))))  # what a recovered epoch replays
    n0 = len(reg.active_states())
    refused = ap._apply_registry_actions(reg, [ledger_form], 0, policy=ap.AdaptiveDecisionPolicy())
    assert refused == []
    assert len(reg.active_states()) == n0 + 4               # 2 children x 2 rungs
    assert all(s.active for s in reg.all_states() if s.secondary_center == 0.0)
    kids = [s for s in reg.active_states() if s.source == cr.SOURCE]
    assert sorted(s.gamd_lambda for s in kids) == [0.0, 0.0, 1.0, 1.0]
    assert all(s.metadata[cr.METADATA_KEY]["rule"] == "R3" for s in kids)
    assert ap._action_to_dict(action)["action"] == "insert"


def test_insert_on_an_axis_state_is_refused():
    reg = _reg()
    axis = next(s for s in reg.active_states() if s.primary_k == 0.0)
    refused = ap._apply_registry_actions(reg, [("insert", axis.state_id, [(0.2, 0.0, 0.4, 6.0)], "r")], 0,
                                         policy=ap.AdaptiveDecisionPolicy())
    assert refused[0]["reason"] == "anchor_or_axis"


def test_resolution_children_seed_from_the_parent_frame_even_if_a_neighbour_is_nearer(tmp_path):
    reg = _reg(lambdas=(0.0,))
    parent = next(s for s in reg.active_states() if s.secondary_center == 0.0 and s.primary_k > 0)
    neighbour = next(s for s in reg.active_states() if s.secondary_center == 1.0)
    ap._apply_registry_actions(reg, [("add", parent.state_id, (0.2, 800.0, 0.9, 5.0), "cv2_resolution R2: x",
                                      cr.action_metadata("R2", None, 0, parent.state_id))], 0,
                               policy=ap.AdaptiveDecisionPolicy())
    child = next(s for s in reg.active_states() if s.source == "adaptive_production")
    bank = tmp_path / "bank"
    bank.mkdir()
    rows = [("p.pdb", parent.state_id, 0.2, 0.05), ("n.pdb", neighbour.state_id, 0.2, 0.95)]
    with (bank / "final_survivor_seeds.csv").open("w") as fh:
        fh.write("seed_name,survivor_pdb_path,source_state_id,primary_cv_value,secondary_cv_value\n")
        for name, sid, p, s in rows:
            fh.write(f"{name},{bank / name},{sid},{p},{s}\n")
    out = ap.select_state_aware_seeds_for_targets(bank, reg)
    by_target = {a["target_state_id"]: a for a in out["assignments"]}
    assert by_target[child.state_id]["seed_name"] == "p.pdb"          # parent, not the nearer neighbour
    assert by_target[neighbour.state_id]["seed_name"] == "n.pdb"      # everyone else unchanged


# ---- driver hook -------------------------------------------------------------------------------

def test_hook_failure_returns_the_actions_unchanged_and_records_the_error(tmp_path, monkeypatch):
    monkeypatch.setattr(cio, "propose_cv2_resolution", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    reg = _reg(lambdas=(0.0,))
    epoch_dir = tmp_path / "epoch_000"
    epoch_dir.mkdir()
    acts = [("extend", 0, "x")]
    out = cio.run_epoch_cv2_resolution(adaptive_dir=tmp_path, epoch_dir=epoch_dir, epoch=0, registry=reg,
                                       diagnostics={"states": [], "edges": []}, actions=acts,
                                       policy=ap.AdaptiveDecisionPolicy(cv2_resolution=True),
                                       args=Namespace(temperature_k=300.0), out_dir=tmp_path, phase_dirs=[epoch_dir])
    assert out == acts
    rep = json.loads((epoch_dir / cr.REPORT_NAME).read_text())
    assert rep["status"] == "error" and "boom" in rep["error"] and rep["schema_version"] == cr.SCHEMA_VERSION


def test_applier_refusals_are_annotated_into_the_report(tmp_path):
    (tmp_path / cr.REPORT_NAME).write_text(json.dumps({"summary": {}}))
    acts = [("extend", 1, "x"), ("add", 2, (0.2, 800.0, 0.5, 5.0), "cv2_resolution R1/weak: y", {})]
    cio.annotate_report_with_refusals(tmp_path, acts, [{"index": 1, "reason": "duplicate", "action": "add"}])
    rep = json.loads((tmp_path / cr.REPORT_NAME).read_text())
    assert rep["apply"]["refused"][0]["reason"] == "duplicate"


# ---- the real epoch loop ------------------------------------------------------------------------

class _ReachedFinal(Exception):
    pass


def _campaign(tmp_path, flag):
    out = tmp_path / "run"
    adaptive = out / "adaptive_production"
    adaptive.mkdir(parents=True)
    reg = ap.WindowStateRegistry()
    for c in (0.1, 0.2, 0.3):
        reg.add_state(c, 800.0, epoch=0, source="seed")
    reg.save(adaptive)
    reg.write_active_window_csv(adaptive / "windows_epoch_000.csv")
    argv = ["--seq", "GYDPETGTWG", "--out", str(out)] + (["--ap-cv2-resolution"] if flag else [])
    args = parse_args(argv)
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
    real_hook = cio.run_epoch_cv2_resolution

    def counting_hook(**kw):
        calls["hook"] += 1
        return real_hook(**kw)
    monkeypatch.setattr(prod, "run_gareus", fake_run_gareus)
    monkeypatch.setattr(ap, "propose_actions_from_diagnostics", lambda registry, diagnostics, **kw: [])
    monkeypatch.setattr(ap, "collect_segmented_epoch_diagnostics", lambda *a, **k: {"states": [], "edges": []})
    monkeypatch.setattr(ap, "_assert_epoch_has_samples", lambda *a, **k: None)
    monkeypatch.setattr(cio, "run_epoch_cv2_resolution", counting_hook)
    with pytest.raises(_ReachedFinal):
        ap.run_adaptive_production_auto_loop(args, out, None, None, None, None, None, None)


@pytest.mark.parametrize("flag", [False, True])
def test_epoch_loop_calls_the_hook_only_with_the_flag(tmp_path, monkeypatch, flag):
    args, out, adaptive = _campaign(tmp_path, flag)
    calls = {"hook": 0}
    _drive(monkeypatch, args, out, calls)
    assert calls["hook"] == (1 if flag else 0)
    report = adaptive / "epoch_000" / cr.REPORT_NAME
    assert report.exists() == flag
    if flag:
        rep = json.loads(report.read_text())
        assert rep["status"] == "ok" and rep["rules"]["R1"]["status"] == "unavailable"
        assert rep["rules"]["R2"]["status"] == "unavailable" and (adaptive / cr.HISTORY_NAME).exists()
        settings = json.loads((adaptive / ap.DECISION_SETTINGS_FILENAME).read_text())["settings"]
        assert settings["cv2_resolution"] is True


class _Crash(Exception):
    pass


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
    (adaptive / ".gareus_run.lock").unlink(missing_ok=True)   # the stale-lock path of a real resume
    _drive(monkeypatch, args, out, calls)     # resume: re-enters epoch 0 from the ledger
    assert calls["hook"] == 1


def test_parquet_runs_provider_breaks_runs_where_the_replica_changes(tmp_path):
    # Two replicas trapped in opposite modes swap through window 0 every 2 samples: the
    # state-indexed series flips, but no replica ever crosses while at the window.
    steps = np.arange(0, 2000, 250)
    rep = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    data = {"step": steps, "window_id": np.zeros(8, int), "replica": rep, "segment_id": np.array(["s1"] * 8),
            "cv2": np.where(rep == 0, -0.7, 0.7)}
    provide = cio.parquet_runs_provider([("p", tmp_path)], lambda d: {0: 5}, load=lambda d: data)
    rec = provide([5])[5]
    assert rec["source"] == "parquet"
    assert cr.count_transitions(rec["state_runs"], -0.3, 0.3) == 3
    assert cr.count_transitions(rec["replica_runs"], -0.3, 0.3) == 0
