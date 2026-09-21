"""Epoch 0 hands production a table AND a sidecar; the driver must apply both.

chignolin_8 job 2553783: the swarm selected CV2, wrote a CV1 x CV2 x lambda table and a
sidecar with cv2: residual-torsion-pc + the frozen model paths, and the adaptive driver
took only the table -- `args.secondary_cv` stayed 'auto' into production and the loader
rejected the residual centres (-1.98) against the default [0, 1] range.
"""
import types

import pytest


def _write_sidecar(out, payload):
    from gareus.config import _write_yaml_or_json
    an = out / "swarm" / "analysis"
    an.mkdir(parents=True, exist_ok=True)
    _write_yaml_or_json(an / "ladder_run_args.yaml", payload)
    return an


def _pair_sidecar(an):
    return {
        "cvs": {"cv1": "contacts", "cv2": "residual-torsion-pc"},
        "secondary_cv_model": str(an / "cv_pair_model.json"),
        "secondary_cv_candidate_set": str(an / "cv_candidate_set.json"),
        "secondary_cv_feature_schema": str(an / "cv_feature_schema.json"),
        "tica_switch_cv2": False,
        "windows": {"window_mode": "manual", "windows_2d_csv": str(an / "windows_lambda_ladder.csv")},
    }


def test_sidecar_resolves_auto_to_the_selected_mode_and_carries_the_model_paths(tmp_path):
    from gareus.swarm.epoch0 import apply_epoch0_sidecar
    an = _write_sidecar(tmp_path, _pair_sidecar(tmp_path / "swarm" / "analysis"))
    args = types.SimpleNamespace(secondary_cv="auto", tica_switch_cv2=True, secondary_cv_model=None)
    applied = apply_epoch0_sidecar(args, tmp_path)
    assert args.secondary_cv == "residual-torsion-pc"
    assert args.secondary_cv_model == str(an / "cv_pair_model.json")
    assert args.secondary_cv_candidate_set == str(an / "cv_candidate_set.json")
    assert args.secondary_cv_feature_schema == str(an / "cv_feature_schema.json")
    assert args.tica_switch_cv2 is False
    assert set(applied) == {"secondary_cv", "secondary_cv_model", "secondary_cv_candidate_set",
                            "secondary_cv_feature_schema", "tica_switch_cv2"}
    # idempotent: a later job in the chain applies the same thing again
    assert apply_epoch0_sidecar(args, tmp_path) == applied


def test_cv1_only_fallback_sidecar_turns_auto_into_none(tmp_path):
    from gareus.swarm.epoch0 import apply_epoch0_sidecar
    _write_sidecar(tmp_path, {"cvs": {"cv1": "contacts", "cv2": "none"},
                              "windows": {"window_mode": "manual", "windows_2d_csv": "x.csv"}})
    args = types.SimpleNamespace(secondary_cv="auto")
    apply_epoch0_sidecar(args, tmp_path)
    assert args.secondary_cv == "none" and not hasattr(args, "secondary_cv_model")


def test_no_sidecar_applies_nothing_and_a_sidecar_that_leaves_auto_unresolved_is_an_error(tmp_path):
    from gareus.swarm.epoch0 import apply_epoch0_sidecar
    args = types.SimpleNamespace(secondary_cv="none")
    assert apply_epoch0_sidecar(args, tmp_path) == {}
    _write_sidecar(tmp_path, {"windows": {"window_mode": "manual"}})
    with pytest.raises(RuntimeError, match="does not resolve cv2 'auto'"):
        apply_epoch0_sidecar(types.SimpleNamespace(secondary_cv="auto"), tmp_path)


def test_driver_handover_applies_the_sidecar_after_the_ladder(tmp_path, monkeypatch):
    import gareus.swarm.epoch0 as E
    from gareus.adaptive_production import _epoch0_swarm_window_table
    an = _write_sidecar(tmp_path, _pair_sidecar(tmp_path / "swarm" / "analysis"))
    ladder = an / "windows_lambda_ladder.csv"
    ladder.write_text("window,primary_cv_mode,primary_cv_center,primary_cv_k_kcal,secondary_cv_center,"
                      "secondary_cv_k_kcal_mol,gamd_lambda\n0,contacts,0.1,200,-1.98,0.67,0.0\n")
    monkeypatch.setattr(E, "run_or_resume_epoch0", lambda args, out_dir, **kw: ladder)
    args = types.SimpleNamespace(secondary_cv="auto", tica_switch_cv2=True)
    got = _epoch0_swarm_window_table(args, tmp_path, tmp_path / "adaptive_production", None, None)
    assert got == ladder
    assert args.secondary_cv == "residual-torsion-pc" and args.tica_switch_cv2 is False


def test_the_two_d_loader_refuses_auto_by_name_instead_of_a_range_complaint(tmp_path):
    from gareus.windows import load_explicit_2d_window_csv
    csv_path = tmp_path / "w.csv"
    csv_path.write_text("window,primary_cv_mode,primary_cv_center,primary_cv_k_kcal,secondary_cv_center,"
                        "secondary_cv_k_kcal_mol,gamd_lambda\n0,contacts,0.1,200,-1.98,0.67,0.0\n")

    class Args:
        primary_cv = "nonlocal-contacts"
        secondary_cv = "auto"
        contact_k_kcal = None
        secondary_cv_k_kcal = 1.0
    with pytest.raises(ValueError, match="apply_epoch0_sidecar"):
        load_explicit_2d_window_csv(Args(), csv_path)

    class Resolved(Args):
        secondary_cv = "residual-torsion-pc"
    centers, ks, sec_c, sec_k, meta, _ = load_explicit_2d_window_csv(Resolved(), csv_path)
    assert sec_c[0] == pytest.approx(-1.98) and meta["enabled"] is True
