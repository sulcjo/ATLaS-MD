import pytest
from gareus.cli import parse_args


def test_auto_cv2_parses_with_documented_defaults_and_forces_the_tica_switch_off():
    # Adapted from the task template: --cv1 contacts is passed explicitly because
    # parse_args defaults to the distance primary CV when --cv1 is absent (the
    # template's assumption of a nonlocal-contacts default does not hold).
    a = parse_args(["--seq", "GYDPETGTWG", "--cv1", "contacts", "--cv2", "auto",
                    "--window-mode", "adaptive-production", "--tica-switch-cv2"])
    assert a.secondary_cv == "auto" and a.primary_cv == "nonlocal-contacts"
    assert a.cv_selection_residual_degree == 1 and a.cv_selection_max_coupling_fraction == 0.25
    assert a.cv_selection_k2_reference_kcal == 1.0 and a.cv_selection_min_gain_nats == 0.02
    assert a.swarm_n_windows_cv2 == 4 and a.cv_selection_fallback == "cv1_only"
    assert a.tica_switch_cv2 is False


def test_cv1_auto_is_not_accepted_in_this_release():
    with pytest.raises(SystemExit):
        parse_args(["--seq", "GYDPETGTWG", "--cv1", "auto"])


def test_auto_cv2_cannot_reach_a_manual_production_run():
    with pytest.raises(SystemExit):
        parse_args(["--seq", "GYDPETGTWG", "--cv2", "auto", "--window-mode", "manual"])


def test_residual_mode_in_production_requires_all_three_model_paths():
    with pytest.raises(SystemExit):
        parse_args(["--seq", "GYDPETGTWG", "--cv2", "residual-torsion-pc", "--window-mode", "manual",
                   "--secondary-cv-model", "m.json"])
    a = parse_args(["--seq", "GYDPETGTWG", "--cv2", "residual-torsion-pc", "--window-mode", "manual",
                    "--secondary-cv-model", "m.json", "--secondary-cv-candidate-set", "c.json",
                    "--secondary-cv-feature-schema", "f.json", "--windows-2d-csv", "w.csv"])
    assert a.secondary_cv == "residual-torsion-pc" and a.tica_switch_cv2 is False


def test_residual_pc_alias_is_canonicalised():
    from gareus.cv import secondary_cv_mode
    assert secondary_cv_mode("residual-pc") == "residual-torsion-pc"
    assert secondary_cv_mode("auto") == "auto"


def test_yaml_cv_selection_block_reaches_args(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("schema_version: '2.0'\nsequence: {seq: GYDPETGTWG}\n"
                   "cvs: {cv1: contacts, cv2: auto}\nwindows: {window_mode: adaptive-production}\n"
                   "cv_selection: {residual_degree: 2, max_nonlinear_r2: 0.1, fallback: refuse}\n")
    a = parse_args(["--config", str(cfg)])
    assert a.cv_selection_residual_degree == 2 and a.cv_selection_fallback == "refuse"
    assert a.cv_selection_max_nonlinear_r2 == 0.1 and a.secondary_cv == "auto"
