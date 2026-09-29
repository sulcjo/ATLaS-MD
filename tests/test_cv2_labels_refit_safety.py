"""Spec 3.6: CV2 labels, refit safety for frozen residual pairs, family-order contract."""
import json
from types import SimpleNamespace

import pytest

import gareus.adaptive_production as ap
import gareus.cv_selection.contracts as C
from gareus.cv_selection.labels import component_label

from test_cv_selection_conditional_tica import slow_selection  # noqa: F401  (module fixture)


def _renumbered(raw, keep):
    """Keep the components whose original index is in ``keep``, renumbered 1..n in that order."""
    by_index = {c["component_index"]: c for c in raw["components"]}
    out = []
    for new, old in enumerate(keep, start=1):
        comp = dict(by_index[old])
        comp["component_index"] = new
        out.append(comp)
    return {**{k: v for k, v in raw.items() if k != "sha256"}, "components": out}


# ---- contract: all PCA indices precede all tICA indices, no gaps ---------------------------

def test_tica_modes_right_after_fewer_than_six_pcs_are_accepted(slow_selection):  # noqa: F811
    _, sel = slow_selection
    raw = _renumbered(sel.candidate_set.to_mapping(), [1, 2, 3, 4, 7, 8, 9])
    cs = C.CandidateSet.from_mapping(raw)
    assert [c.family for c in cs.components] == ["pca"] * 4 + ["conditional_tica"] * 3
    assert cs.components[4].component_index == 5


def test_a_gap_in_the_indices_is_rejected(slow_selection):  # noqa: F811
    _, sel = slow_selection
    raw = sel.candidate_set.to_mapping()
    raw = {**{k: v for k, v in raw.items() if k != "sha256"},
           "components": [c for c in raw["components"] if c["component_index"] != 3]}
    with pytest.raises(Exception, match="no gaps"):
        C.CandidateSet.from_mapping(raw)


def test_a_pc_after_a_tica_mode_is_rejected(slow_selection):  # noqa: F811
    _, sel = slow_selection
    raw = _renumbered(sel.candidate_set.to_mapping(), [1, 2, 3, 7, 4])
    with pytest.raises(Exception, match="precede"):
        C.CandidateSet.from_mapping(raw)


def test_the_original_artifact_still_validates_to_the_same_digest(slow_selection):  # noqa: F811
    _, sel = slow_selection
    data = sel.candidate_set.to_json_bytes()
    assert C.CandidateSet.from_json_bytes(data).sha256 == sel.candidate_set.sha256


# ---- labels ---------------------------------------------------------------------------------

def test_label_carries_family_index_and_lag_in_ps(slow_selection):  # noqa: F811
    _, sel = slow_selection
    j = sel.pair_model.selected_component_index
    label = component_label(sel.candidate_set, j, frame_dt_ps=2.0)
    comp = next(c for c in sel.candidate_set.components if c.component_index == j)
    assert label["cv2_component_family"] == "conditional_tica"
    assert label["cv2_component_index"] == j
    assert label["cv2_tica_lag_frames"] == comp.tica_lag_frames
    assert label["cv2_tica_lag_ps"] == pytest.approx(2.0 * comp.tica_lag_frames)
    pc = component_label(sel.candidate_set, 1, frame_dt_ps=2.0)
    assert pc["cv2_component_family"] == "pca" and pc["cv2_tica_lag_ps"] is None


# ---- refit safety ---------------------------------------------------------------------------

def _frozen_args(**kw):
    base = dict(secondary_cv="residual-torsion-pc", secondary_cv_model="/x/cv_pair_model.json",
                tica_obs_interval=10, tica_update_after_epochs=[0], tica_epochs_per_cycle=0,
                tica_switch_cv2=True)
    base.update(kw)
    return SimpleNamespace(**base)


def test_tica_refit_is_refused_for_a_frozen_pair(tmp_path):
    report = ap._maybe_update_tica_cvaux(0, tmp_path, tmp_path, _frozen_args())
    assert report["status"] == "refused_frozen_residual_pair"


def test_tica_switch_is_refused_for_a_frozen_pair():
    args = _frozen_args()
    report = {"status": "updated", "state_file": __file__}
    assert ap._apply_tica_cv2_switch(args, report, next_epoch=1) is False
    assert args.secondary_cv == "residual-torsion-pc"
    assert report["cv2_switch_skipped"]["reason"] == "frozen_residual_pair"


def test_recentring_is_refused_for_a_frozen_pair(tmp_path):
    reg = ap.WindowStateRegistry()
    reg.add_state(0.2, 800.0, secondary_center=0.5, secondary_k=50.0, epoch=0, source="seed")
    with pytest.raises(RuntimeError, match="frozen residual pair"):
        ap._apply_tica_centers_to_registry(reg, {0: 1.5}, tmp_path, frozen_residual_pair=True)
    assert reg.active_states()[0].secondary_center == 0.5


def test_non_residual_campaigns_are_unchanged():
    assert ap._frozen_residual_pair(SimpleNamespace(secondary_cv="torsion-pca")) is False
    assert ap._frozen_residual_pair(SimpleNamespace(secondary_cv="tica-linear", secondary_cv_model=None)) is False
    assert ap._frozen_residual_pair(SimpleNamespace(secondary_cv="residual-torsion-pc")) is True


def test_resume_of_a_frozen_pair_with_a_recorded_switch_fails_closed(tmp_path, monkeypatch):
    from gareus.cli import parse_args

    out = tmp_path / "run"
    adaptive = out / "adaptive_production"
    adaptive.mkdir(parents=True)
    reg = ap.WindowStateRegistry()
    reg.add_state(0.2, 800.0, secondary_center=0.5, secondary_k=50.0, epoch=0, source="seed")
    reg.save(adaptive)
    reg.write_active_window_csv(adaptive / "windows_epoch_001.csv")
    (adaptive / "adaptive_production_driver_summary.json").write_text(json.dumps({
        "epochs_completed": 1,
        "epoch_summaries": [{"epoch": 0, "tica_update": {
            "cv2_switched": {"from": "torsion-pca", "to": "tica-linear", "effective_epoch": 1}}}],
    }))
    args = parse_args(["--seq", "GYDPETGTWG", "--out", str(out)])
    args.adaptive_production_resume = True
    args.adaptive_production_epochs = 2
    args.adaptive_production_global_shared_gamd = False
    args.secondary_cv = "residual-torsion-pc"
    with pytest.raises(RuntimeError, match="resume refused"):
        ap.run_adaptive_production_auto_loop(args, out, None, None, None, None, None, None)
