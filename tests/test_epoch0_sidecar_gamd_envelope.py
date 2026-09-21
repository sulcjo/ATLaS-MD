"""The epoch-0 sidecar hands production the swarm's FROZEN GaMD envelope.

chignolin_8 job 2558439: ``ladder_run_args.yaml`` named ``gamd.shared_gamd_setup_dir`` (the
swarm's ``shared_gamd_setup/``) but the driver applied only the CV block, and it had already
resolved ``shared_gamd_setup_dir`` to an empty ``adaptive_production/global_shared_gamd_setup``
before the epoch-0 hook ran. The first worker therefore ran its own 1,010,000-step shared setup
and was heading into a per-state recon (~6.75 ns x 248 states) -- re-deriving the envelope the
frozen-envelope rule says is measured once. These tests pin the handoff on both jobs of a chain.
"""
import json
import types

import pytest


SWARM_GLOBALS = {
    "Vavg_Dihedral": 434.34, "Vavg_Total": -3012.09, "Vmax_Dihedral": 527.46, "Vmax_Total": -2371.33,
    "Vmin_Dihedral": 360.97, "Vmin_Total": -3711.52, "k0_Dihedral": 1.0, "k0_Total": 0.3331,
    "k_Dihedral": 0.006, "k_Total": 0.000249, "sigma0_Dihedral": 25.104, "sigma0_Total": 25.104,
    "sigmaV_Dihedral": 19.36, "sigmaV_Total": 157.63,
    "threshold_energy_Dihedral": 527.46, "threshold_energy_Total": -2371.33,
}


def _write_envelope(an):
    d = an / "shared_gamd_setup"
    d.mkdir(parents=True, exist_ok=True)
    (d / "shared_gamd_setup_globals.json").write_text(json.dumps({
        "mode": "swarm_unbiased_envelope", "all_globals": SWARM_GLOBALS,
        "interesting_globals": SWARM_GLOBALS, "gamd_boost_type": "pep-gamd-lower-dual",
    }))
    return d


def _sidecar(tmp_path, *, with_gamd=True):
    from gareus.config import _write_yaml_or_json
    an = tmp_path / "swarm" / "analysis"
    an.mkdir(parents=True, exist_ok=True)
    payload = {
        "cvs": {"cv1": "contacts", "cv2": "residual-torsion-pc"},
        "secondary_cv_model": str(an / "cv_pair_model.json"),
        "secondary_cv_candidate_set": str(an / "cv_candidate_set.json"),
        "secondary_cv_feature_schema": str(an / "cv_feature_schema.json"),
        "tica_switch_cv2": False,
        "windows": {"window_mode": "manual", "windows_2d_csv": str(an / "windows_lambda_ladder.csv")},
    }
    if with_gamd:
        payload["gamd"] = {"gamd_boost_type": "pep-gamd-lower-dual",
                           "shared_gamd_setup_dir": str(an / "shared_gamd_setup")}
    _write_yaml_or_json(an / "ladder_run_args.yaml", payload)
    return an


def test_sidecar_points_production_at_the_frozen_envelope_when_nobody_chose_a_dir(tmp_path):
    from gareus.swarm.epoch0 import apply_epoch0_sidecar
    an = _sidecar(tmp_path)
    env = _write_envelope(an)
    args = types.SimpleNamespace(secondary_cv="auto", shared_gamd_setup_dir="")
    applied = apply_epoch0_sidecar(args, tmp_path)
    assert args.shared_gamd_setup_dir == str(env)
    assert applied["shared_gamd_setup_dir"] == str(env)
    # idempotent on the next job of the chain
    assert apply_epoch0_sidecar(args, tmp_path) == applied


def test_sidecar_overrides_the_drivers_auto_resolved_default_but_not_an_explicit_user_dir(tmp_path):
    from gareus.swarm.epoch0 import apply_epoch0_sidecar
    an = _sidecar(tmp_path)
    env = _write_envelope(an)
    auto = tmp_path / "adaptive_production" / "global_shared_gamd_setup"
    # the driver resolves shared_gamd_setup_dir to its campaign default BEFORE the epoch-0 hook runs
    args = types.SimpleNamespace(secondary_cv="auto", shared_gamd_setup_dir=str(auto),
                                 _global_shared_gamd_setup_dir=str(auto))
    applied = apply_epoch0_sidecar(args, tmp_path)
    assert args.shared_gamd_setup_dir == str(env) and applied["shared_gamd_setup_dir"] == str(env)
    # a directory the user picked on purpose is kept, and the decision is visible
    user_dir = tmp_path / "my_pilot_envelope"
    args = types.SimpleNamespace(secondary_cv="auto", shared_gamd_setup_dir=str(user_dir),
                                 _global_shared_gamd_setup_dir=str(auto))
    applied = apply_epoch0_sidecar(args, tmp_path)
    assert args.shared_gamd_setup_dir == str(user_dir)
    assert applied["shared_gamd_setup_dir_kept_explicit"] == str(user_dir)
    assert "shared_gamd_setup_dir" not in applied


def test_sidecar_without_a_gamd_block_leaves_the_shared_dir_alone(tmp_path):
    from gareus.swarm.epoch0 import apply_epoch0_sidecar
    _sidecar(tmp_path, with_gamd=False)
    args = types.SimpleNamespace(secondary_cv="auto", shared_gamd_setup_dir="")
    applied = apply_epoch0_sidecar(args, tmp_path)
    assert args.shared_gamd_setup_dir == "" and "shared_gamd_setup_dir" not in applied


def test_sidecar_naming_a_dir_without_globals_is_an_error_not_a_silent_recalibration(tmp_path):
    from gareus.swarm.epoch0 import apply_epoch0_sidecar
    _sidecar(tmp_path)  # gamd block present, shared_gamd_setup/ never written
    args = types.SimpleNamespace(secondary_cv="auto", shared_gamd_setup_dir="")
    with pytest.raises(RuntimeError, match="shared_gamd_setup_globals.json is missing"):
        apply_epoch0_sidecar(args, tmp_path)


def test_production_reuses_the_swarm_envelope_and_skips_its_own_calibration_path(tmp_path):
    """The swarm payload has 16 globals and no context checkpoint; the reuse loader must accept it."""
    from gareus.production import load_reusable_shared_gamd_setup
    an = tmp_path / "swarm" / "analysis"
    env = _write_envelope(an)
    worker = tmp_path / "adaptive_production" / "epoch_000"
    worker.mkdir(parents=True)
    args = types.SimpleNamespace(shared_gamd_setup_dir=str(env))
    got = load_reusable_shared_gamd_setup(args, worker)
    assert got is not None, "a 16-global swarm envelope without a checkpoint must be reusable"
    all_globals, interesting, calib_steps, checkpoint, note = got
    assert all_globals == pytest.approx(SWARM_GLOBALS) and checkpoint is None and calib_steps == 0
    assert note["reused_shared_gamd_setup"] is True and note["source_shared_gamd_setup_dir"] == str(env)
    local = json.loads((worker / "shared_gamd_setup_globals.json").read_text())
    assert local["all_globals"] == pytest.approx(SWARM_GLOBALS) and local["reused_shared_gamd_setup"] is True


def test_driver_hook_seeds_the_campaign_global_dir_so_the_second_job_reuses_too(tmp_path, monkeypatch):
    """Job 1 applies the sidecar; job 2 (registry exists) never runs the hook and resolves the
    campaign default. The hook must leave the envelope there, and must not clobber one that exists."""
    import gareus.swarm.epoch0 as E
    from gareus.adaptive_production import _epoch0_swarm_window_table
    an = _sidecar(tmp_path)
    env = _write_envelope(an)
    ladder = an / "windows_lambda_ladder.csv"
    ladder.write_text("window,primary_cv_mode,primary_cv_center,primary_cv_k_kcal,secondary_cv_center,"
                      "secondary_cv_k_kcal_mol,gamd_lambda\n0,contacts,0.1,200,-1.98,0.67,0.0\n")
    monkeypatch.setattr(E, "run_or_resume_epoch0", lambda args, out_dir, **kw: ladder)
    adaptive_dir = tmp_path / "adaptive_production"
    auto = adaptive_dir / "global_shared_gamd_setup"
    auto.mkdir(parents=True)
    args = types.SimpleNamespace(secondary_cv="auto", tica_switch_cv2=True,
                                 shared_gamd_setup_dir=str(auto), _global_shared_gamd_setup_dir=str(auto))
    got = _epoch0_swarm_window_table(args, tmp_path, adaptive_dir, None, None)
    assert got == ladder and args.shared_gamd_setup_dir == str(env)
    seeded = json.loads((auto / "shared_gamd_setup_globals.json").read_text())
    assert seeded["all_globals"] == pytest.approx(SWARM_GLOBALS)
    source = json.loads((auto / "shared_gamd_envelope_source.json").read_text())
    assert source["source_dir"] == str(env)
    # job 2: the driver resolves the campaign default and finds the frozen envelope there
    from gareus.production import load_reusable_shared_gamd_setup
    job2 = types.SimpleNamespace(shared_gamd_setup_dir=str(auto))
    (tmp_path / "w2").mkdir(exist_ok=True)
    got2 = load_reusable_shared_gamd_setup(job2, tmp_path / "w2")
    assert got2 is not None and got2[0] == pytest.approx(SWARM_GLOBALS)
    # an envelope already in the global dir is the campaign's and is never overwritten
    (auto / "shared_gamd_setup_globals.json").write_text(json.dumps({"all_globals": {"marker": 1.0}}))
    from gareus.adaptive_production import _seed_global_shared_gamd_from_envelope
    res = _seed_global_shared_gamd_from_envelope(args, str(env))
    assert res["status"] == "already_present"
    assert json.loads((auto / "shared_gamd_setup_globals.json").read_text())["all_globals"] == {"marker": 1.0}
