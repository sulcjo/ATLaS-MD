"""--ap-ladder-adapt wiring: flags, policy, frozen campaign settings, the epoch proposer."""
from dataclasses import replace

from gareus.adaptive_production import (AdaptiveDecisionPolicy, WindowState, WindowStateRegistry,
                                        _propose_ladder_respace, _resolve_ladder_settings,
                                        policy_from_args)
from gareus.cli import parse_args
from test_ladder_adapt_loader import _fixture  # noqa: E402  (tests dir on sys.path)


def test_flags_default_off_and_reach_the_policy():
    a = parse_args(["--seq", "DPETG"])
    p = policy_from_args(a)
    assert p.ladder_adapt == "off" and p.ladder_min_overlap == 0.25 and p.ladder_overlap_quantile == 0.10
    assert p.ladder_max_moves == 2 and p.ladder_max_rungs == 8 and p.ladder_min_ess == 200.0


def test_flags_override():
    a = parse_args(["--seq", "DPETG", "--ap-ladder-adapt", "respace", "--ap-ladder-min-overlap", "0.3",
                    "--ap-ladder-max-moves", "3", "--max-replicas", "236"])
    p = policy_from_args(a)
    assert p.ladder_adapt == "respace" and p.ladder_min_overlap == 0.3 and p.ladder_max_moves == 3
    assert p.max_replicas_budget == 236


def _registry_from(states):
    reg = WindowStateRegistry()
    for s in states:
        reg.add_state(primary_center=s["primary_center"], primary_k=s["primary_k"],
                      secondary_center=s["secondary_center"], secondary_k=s["secondary_k"],
                      gamd_lambda=s["gamd_lambda"], epoch=0, source="t", reason="t")
    return reg


def test_proposer_off_returns_nothing(tmp_path):
    ad, states = _fixture(tmp_path)
    action, report = _propose_ladder_respace(ad, _registry_from(states), 0, AdaptiveDecisionPolicy())
    assert action is None and report["status"] == "off"


def test_proposer_reads_the_epoch_and_returns_an_atomic_action_or_none(tmp_path):
    ad, states = _fixture(tmp_path)
    policy = replace(AdaptiveDecisionPolicy(), ladder_adapt="respace", ladder_min_overlap=0.30, ladder_max_moves=4)
    action, report = _propose_ladder_respace(ad, _registry_from(states), 0, policy)
    assert report["status"] == "ok" and report["n_centres"] == 2
    assert report["current"] == [0.0, 0.5, 1.0]
    if action is not None:
        kind, drop, add, reason = action
        assert kind == "respace_ladder" and 0.0 not in drop and 1.0 not in drop


def test_proposer_never_raises_and_reports_errors(tmp_path):
    policy = replace(AdaptiveDecisionPolicy(), ladder_adapt="respace")
    reg = WindowStateRegistry()
    reg.add_state(primary_center=0.1, primary_k=100.0, gamd_lambda=0.0, epoch=0, source="t", reason="t")
    reg.add_state(primary_center=0.1, primary_k=100.0, gamd_lambda=1.0, epoch=0, source="t", reason="t")
    action, report = _propose_ladder_respace(tmp_path / "missing", reg, 0, policy)
    assert action is None and report["status"] in ("error", "no_data")


def test_settings_are_frozen_at_first_use_and_honoured_on_resume(tmp_path):
    first = replace(AdaptiveDecisionPolicy(), ladder_adapt="respace", ladder_min_overlap=0.25)
    p1 = _resolve_ladder_settings(tmp_path, first, override=False)
    assert p1.ladder_min_overlap == 0.25 and (tmp_path / "ladder_adapt_settings.json").exists()
    later = replace(AdaptiveDecisionPolicy(), ladder_adapt="off", ladder_min_overlap=0.4)
    p2 = _resolve_ladder_settings(tmp_path, later, override=False)
    assert p2.ladder_adapt == "respace" and p2.ladder_min_overlap == 0.25
    p3 = _resolve_ladder_settings(tmp_path, later, override=True)
    assert p3.ladder_adapt == "off" and p3.ladder_min_overlap == 0.4


def test_off_campaign_without_a_settings_file_writes_nothing(tmp_path):
    p = _resolve_ladder_settings(tmp_path, AdaptiveDecisionPolicy(), override=False)
    assert p.ladder_adapt == "off" and not (tmp_path / "ladder_adapt_settings.json").exists()
