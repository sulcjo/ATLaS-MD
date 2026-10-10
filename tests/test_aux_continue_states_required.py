"""Final fix wave I1: aux discovery requires --ap-continue-states, frozen per campaign.

Without it every post-admission phase re-pulls every worker, so the per-carrier burn-in (Task 11) drops every
worker row and most carriers' ordinary rows in every phase.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gareus.adaptive.aux_discovery.settings import (AUX_CAMPAIGN_OPTIONS_FILENAME, aux_discovery_incompatibilities,
                                                    resolve_aux_campaign_options)
from gareus.adaptive_production import (DECISION_SETTINGS_FIELDS, AdaptiveDecisionPolicy,
                                        _resolve_decision_settings, run_adaptive_production_auto_loop)

OK = dict(window_mode="adaptive-production", traj_interval=300, distance_output_interval=300,
          exchange_interval=3000, run_mode="cmd", ap_aux_reserve_slots=4, ap_continue_states=True)


def test_incompatibilities_require_continue_states_and_name_the_cost():
    assert aux_discovery_incompatibilities(SimpleNamespace(**OK)) == []
    msgs = aux_discovery_incompatibilities(SimpleNamespace(**{**OK, "ap_continue_states": False}))
    assert len(msgs) == 1 and "--ap-continue-states" in msgs[0] and "burn-in" in msgs[0]
    assert aux_discovery_incompatibilities(SimpleNamespace(**{k: v for k, v in OK.items()
                                                              if k != "ap_continue_states"}))


def _base(tmp_path):
    return ["--window-mode", "adaptive-production", "--seq", "GYDPETGTWG", "--out", str(tmp_path),
            "--gamd-boost-type", "pep-gamd-lower-dual", "--exchange-mode", "gibbs-walk", "--traj-interval", "250",
            "--distance-output-interval", "250", "--exchange-interval", "500"]


def test_parse_refuses_aux_discovery_without_continue_states(tmp_path, capsys):
    from gareus.cli import parse_args
    with pytest.raises(SystemExit):
        parse_args(_base(tmp_path) + ["--ap-aux-discovery"])
    assert "--ap-continue-states" in capsys.readouterr().err
    assert parse_args(_base(tmp_path) + ["--ap-aux-discovery", "--ap-continue-states"]).ap_continue_states


def test_aux_resume_without_continue_states_is_refused_at_driver_start(tmp_path):
    ad = tmp_path / "adaptive_production"; ad.mkdir()
    _resolve_decision_settings(ad, AdaptiveDecisionPolicy(aux_discovery=True), override=False)   # frozen: aux on
    args = SimpleNamespace(**{**OK, "ap_continue_states": False})                # resume omits both aux flags
    with pytest.raises(RuntimeError, match="frozen decision settings enable aux-CV discovery.*--ap-continue-states"):
        run_adaptive_production_auto_loop(args, tmp_path, None, None, None, None, None, None)
    assert not (ad / AUX_CAMPAIGN_OPTIONS_FILENAME).exists()


def test_campaign_options_frozen_at_first_use_and_mismatch_refused(tmp_path):
    rec = resolve_aux_campaign_options(tmp_path, SimpleNamespace(ap_continue_states=True))
    assert rec["options"] == {"continue_states": True}
    on_disk = json.loads((tmp_path / AUX_CAMPAIGN_OPTIONS_FILENAME).read_text())
    assert on_disk["options"] == {"continue_states": True}
    before = (tmp_path / AUX_CAMPAIGN_OPTIONS_FILENAME).read_bytes()
    assert resolve_aux_campaign_options(tmp_path, SimpleNamespace(ap_continue_states=True))["options"] == \
        {"continue_states": True}
    assert (tmp_path / AUX_CAMPAIGN_OPTIONS_FILENAME).read_bytes() == before        # resume never rewrites
    with pytest.raises(RuntimeError, match="continue_states"):
        resolve_aux_campaign_options(tmp_path, SimpleNamespace(ap_continue_states=False))
    # a record that says False (hand-edited / foreign) is refused too, never adopted
    on_disk["options"]["continue_states"] = False
    (tmp_path / AUX_CAMPAIGN_OPTIONS_FILENAME).write_text(json.dumps(on_disk))
    with pytest.raises(RuntimeError, match="continue_states"):
        resolve_aux_campaign_options(tmp_path, SimpleNamespace(ap_continue_states=True))


def test_malformed_campaign_options_record_refuses(tmp_path):
    (tmp_path / AUX_CAMPAIGN_OPTIONS_FILENAME).write_text("{not json")
    with pytest.raises(RuntimeError, match="not a valid"):
        resolve_aux_campaign_options(tmp_path, SimpleNamespace(ap_continue_states=True))


def test_flag_off_decision_settings_keys_unchanged_and_no_options_file(tmp_path):
    ad = tmp_path / "adaptive_production"; ad.mkdir()
    _resolve_decision_settings(ad, AdaptiveDecisionPolicy(), override=False)
    rec = json.loads((ad / "decision_settings.json").read_text())
    assert sorted(rec["settings"]) == sorted(DECISION_SETTINGS_FIELDS)
    assert "continue_states" not in rec["settings"] and "ap_continue_states" not in rec["settings"]
    assert not (ad / AUX_CAMPAIGN_OPTIONS_FILENAME).exists()
