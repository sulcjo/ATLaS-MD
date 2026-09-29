"""Spec 3.5: the replica cap is checked at phase start, never bricks a resume, is read from the
live registry at every apply, and production truncates before any per-window artifact."""
import inspect
from types import SimpleNamespace

import pytest

import gareus.adaptive_production as ap
import gareus.production as prod
from gareus.adaptive_production import AdaptiveDecisionPolicy, WindowStateRegistry


def _registry(centres=(0.1, 0.2, 0.3), lambdas=(0.0,)):
    reg = WindowStateRegistry()
    for c in centres:
        for lam in lambdas:
            reg.add_state(c, 800.0, gamd_lambda=lam, epoch=0, source="seed")
    return reg


# ---- phase start --------------------------------------------------------------------------

def test_zero_means_unlimited(tmp_path):
    ap._require_phase_within_replica_cap(SimpleNamespace(max_replicas=0), 10_000, tmp_path, "epoch 000")


def test_at_the_cap_starts(tmp_path):
    ap._require_phase_within_replica_cap(SimpleNamespace(max_replicas=4), 4, tmp_path, "epoch 000")


def test_above_the_cap_a_fresh_phase_refuses_with_counts_and_remedy(tmp_path, monkeypatch):
    monkeypatch.setattr(ap, "production_checkpoint_available", lambda d: False)
    with pytest.raises(RuntimeError) as exc:
        ap._require_phase_within_replica_cap(SimpleNamespace(max_replicas=4), 6, tmp_path, "epoch 001")
    msg = str(exc.value)
    assert "6 active states" in msg and "--max-replicas 4" in msg and "at least 6" in msg


def test_above_the_cap_a_checkpointed_phase_resumes_with_a_warning(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(ap, "production_checkpoint_available", lambda d: True)
    ap._require_phase_within_replica_cap(SimpleNamespace(max_replicas=4), 6, tmp_path, "final phase")
    assert "resumes with its checkpointed window set" in capsys.readouterr().out


# ---- apply budget -------------------------------------------------------------------------

def _pol(budget):
    return AdaptiveDecisionPolicy(max_replicas_budget=budget)


def test_an_add_over_budget_is_dropped_and_recorded():
    reg = _registry()
    ctl = ap.AdaptiveProductionController(reg, policy=_pol(4))
    ctl.apply_actions(0, [("add", None, (0.15, 800.0), "a"), ("add", None, (0.25, 800.0), "b")])
    assert len(reg.active_states()) == 4
    assert [r["action"] for r in ctl.refused_actions] == ["add"]


def test_an_add_costs_one_state_per_rung():
    reg = _registry(lambdas=(0.0, 0.5))          # 6 states, 2 rungs
    ctl = ap.AdaptiveProductionController(reg, policy=_pol(7))
    ctl.apply_actions(0, [("add", None, (0.15, 800.0), "a")])
    assert len(reg.active_states()) == 6         # would be 8 > 7


def test_add_rung_is_budgeted_by_the_centres_it_fills():
    reg = _registry()
    ap.AdaptiveProductionController(reg, policy=_pol(5)).apply_actions(0, [("add_rung", 0.5, "r")])
    assert len(reg.active_states()) == 3         # needs 3 new states, only 2 fit
    ap.AdaptiveProductionController(reg, policy=_pol(6)).apply_actions(0, [("add_rung", 0.5, "r")])
    assert len(reg.active_states()) == 6


def test_the_second_apply_site_reads_the_live_registry():
    reg = _registry()
    ap._apply_registry_actions(reg, [("add", None, (0.15, 800.0), "main")], 0, policy=_pol(4))
    ap._apply_registry_actions(reg, [("tica_coverage_add", None, (0.25, 800.0, 0.1, 50.0), "cov", {})],
                               0, policy=_pol(4))
    assert len(reg.active_states()) == 4


def test_no_budget_keeps_todays_behaviour():
    reg = _registry()
    ap.AdaptiveProductionController(reg, policy=_pol(0)).apply_actions(
        0, [("add", None, (0.15, 800.0), "a"), ("add_rung", 0.5, "r")])
    assert len(reg.active_states()) == 8


# ---- production truncation order ----------------------------------------------------------

def test_plain_run_truncation_precedes_every_per_window_artifact():
    src = inspect.getsource(prod.run_gareus)
    trunc = src.index("--max-replicas {_max_replicas}: truncating")
    for later in ("write_window_assignment_csv(", "write_explicit_2d_neighbor_graph_files(",
                  "write_explicit_window_analysis_files(", "repair_epoch_window_map_from_surviving_windows("):
        assert trunc < src.index(later), later
    assert src.count("centers_a = centers_a[:_max_replicas]") == 1       # one truncation site


def test_every_phase_launch_is_preceded_by_the_cap_check():
    src = inspect.getsource(ap)
    lines = src.splitlines()
    launches = [i for i, l in enumerate(lines)
                if ("run_gareus(" in l or "run_gareus_callable(" in l) and not l.lstrip().startswith(("#", "\"", "`"))
                and "def " not in l and "``" not in l and "\"" not in l.split("(")[0]]
    assert len(launches) == 4, [lines[i] for i in launches]
    for i in launches:
        window = "\n".join(lines[max(0, i - 4):i])
        assert "_require_phase_within_replica_cap(" in window, lines[i]


def test_the_ledger_lists_refused_proposals_apart_from_applied_ones(tmp_path):
    import json

    reg = _registry()
    actions = [("add", None, (0.15, 800.0), "fits"), ("add", None, (0.25, 800.0), "over budget")]
    refused = ap._apply_registry_actions(reg, actions, 0, policy=_pol(4))
    reg_path = tmp_path / "state_registry.json"
    reg_path.write_text("{}")
    ap._record_applied_actions(tmp_path, 0, actions, reg_path, refused=refused)
    led = json.loads((tmp_path / ap.APPLIED_ACTIONS_FILENAME).read_text())
    assert [a[3] for a in led["actions"]] == ["fits"]
    assert [r["proposal"][3] for r in led["refused"]] == ["over budget"]
    assert led["refused"][0]["reason"] == "max_replicas_budget"
    assert ap._without_refused(actions, refused) == actions[:1]
