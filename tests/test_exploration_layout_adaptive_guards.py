"""F05/P09: mandatory exploration states are protected across the loader, the adaptive registry
lifecycle and the swarm analysis output (spec F05; finding I03)."""
from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from gareus.swarm.ladder_design import (ROLE_UNRESTRAINED_ANCHOR, design_exploration_layout, layout_plan_record,
                                        layout_rows, write_ladder_windows_2d_csv)


def _table_with_plan(tmp_path, n1=3, n2=2, n_rungs=2, cap=40):
    plan = design_exploration_layout(n1, n2, n_rungs=n_rungs, max_replicas=cap, region_centre_indices=[n1 - 1])
    rows = layout_rows(plan, np.linspace(0.1, 0.9, n1), np.full(n1, 200.0), np.linspace(-1, 1, n2), np.full(n2, 0.7))
    lambdas = list(np.linspace(0.0, 1.0, n_rungs))
    path = write_ladder_windows_2d_csv(tmp_path / "windows_lambda_ladder.csv", rows, lambdas)
    rec = layout_plan_record(plan, rows, lambdas)
    (tmp_path / "layout_plan.json").write_text(json.dumps(rec))
    return path, rec


def test_loader_carries_mandatory_window_indices_from_the_companion_plan(tmp_path):
    from gareus.windows import load_explicit_2d_window_csv
    path, rec = _table_with_plan(tmp_path)

    class Args:
        primary_cv = "nonlocal-contacts"; secondary_cv = "residual-torsion-pc"; contact_k_kcal = None; secondary_cv_k_kcal = 1.0
    _, ks, _, sec_k, _, meta = load_explicit_2d_window_csv(Args(), path)
    assert meta["mandatory_window_indices"] == rec["mandatory_state_ids"]
    assert meta["state_roles"][0] == ROLE_UNRESTRAINED_ANCHOR
    for i in rec["mandatory_state_ids"][:2]:              # the anchor stack: both rungs unrestrained
        assert ks[i] == 0.0 and sec_k[i] == 0.0


def test_registry_marks_mandatory_states_and_refuses_to_retire_or_split_them(tmp_path):
    from gareus.adaptive_production import registry_from_window_csv
    path, rec = _table_with_plan(tmp_path)
    reg = registry_from_window_csv(path, epoch=0, source="epoch0_windows")
    states = reg.all_states()
    assert len(states) == rec["n_states"]
    mandatory = [s for s in states if s.metadata.get("mandatory")]
    assert [s.state_id for s in mandatory] == rec["mandatory_state_ids"]
    assert mandatory[0].metadata["state_role"] == ROLE_UNRESTRAINED_ANCHOR
    with pytest.raises(ValueError, match="mandatory exploration state"):
        reg.retire_state(mandatory[0].state_id, epoch=1, reason="test")
    assert mandatory[0].active
    ordinary = [s for s in states if not s.metadata.get("mandatory")][0]
    reg.retire_state(ordinary.state_id, epoch=1, reason="test")      # ordinary states still retire
    assert not ordinary.active


def test_reachability_filter_keeps_unrestrained_windows_and_blocks_on_a_mandatory_drop(monkeypatch, tmp_path):
    import types
    import gareus.seeding as S
    # library with one seed far from every restrained centre: every restrained window would be dropped
    library = [{"primary_cv_value": 5.0, "pdb_path": "x.pdb"}]
    monkeypatch.setattr(S, "load_genpept_conformer_library", lambda *a, **k: library)
    monkeypatch.setattr(S, "_score_seed_conformer", lambda conf, **k: (10.0, {"primary_score": 10.0, "secondary_score": 0.0}))
    monkeypatch.setattr(S, "primary_cv_is_contacts", lambda args: True)
    args = types.SimpleNamespace(seed_conformers_dir=str(tmp_path), us_seed_preflight_max_score=1.2, seed_selection_mode="active-cv",
                                 secondary_cv="residual-torsion-pc", seed_secondary_weight=1.0)
    (tmp_path / "final_survivor_seeds.csv").write_text("seed_id\n")
    centers = np.array([0.5, 0.1, 0.9]); ks = [0.0, 200.0, 200.0]
    meta = {"enabled": True, "mode": "residual-torsion-pc"}
    wm = {"mandatory_window_indices": [0, 2]}
    with pytest.raises(RuntimeError, match="mandatory exploration windows \\[2\\]"):
        S.filter_explicit_2d_windows_by_seed_reachability(args, tmp_path, None, {"contact_pairs": []}, centers, ks,
                                                          [0.0, -1.0, 1.0], [0.0, 0.7, 0.7], meta, wm)
    # without the mandatory role the unrestrained window is kept and the unreachable ones dropped
    out = S.filter_explicit_2d_windows_by_seed_reachability(args, tmp_path, None, {"contact_pairs": []}, centers, ks,
                                                            [0.0, -1.0, 1.0], [0.0, 0.7, 0.7], meta, {})
    kept_centers = list(np.asarray(out[0]).tolist())
    assert kept_centers == [0.5]
