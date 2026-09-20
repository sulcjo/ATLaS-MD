"""Swarm analysis -> frozen pair artifacts -> sidecar -> production argument parsing -> runtime load.

No MD and no OpenMM: the synthetic swarm from tests/conftest.py stands in for round 0,
and the dry run stops where production would build the System. What it proves is that
the artifacts the swarm writes are exactly the ones a manual production run can consume.
"""
import csv

from conftest import PHI_TORSIONS, PSI_TORSIONS  # tests/ is prepended to sys.path by the runner


def _analyze(synthetic_swarm, swarm_args, **overrides):
    from gareus.swarm.analyze import analyze_swarm_stage
    out = synthetic_swarm(with_features=True, wide_anchor=True)
    report = analyze_swarm_stage(out, swarm_args(secondary_cv="auto", **overrides))
    return out, report


def test_sidecar_parses_into_a_residual_mode_production_run(synthetic_swarm, swarm_args):
    from gareus.cli import parse_args
    from gareus.config import _load_config_file
    out, report = _analyze(synthetic_swarm, swarm_args)
    assert report["cv_selection"]["status"] == "pair", report["cv_selection"]
    side = _load_config_file(out / "swarm" / "analysis" / "ladder_run_args.yaml")
    argv = ["--seq", "GYDPETGTWG", "--cv1", side["cvs"]["cv1"], "--cv2", side["cvs"]["cv2"],
            "--window-mode", side["windows"]["window_mode"],
            "--windows-2d-csv", side["windows"]["windows_2d_csv"],
            "--secondary-cv-model", side["secondary_cv_model"],
            "--secondary-cv-candidate-set", side["secondary_cv_candidate_set"],
            "--secondary-cv-feature-schema", side["secondary_cv_feature_schema"],
            "--tica-switch-cv2"]
    a = parse_args(argv)
    assert a.secondary_cv == "residual-torsion-pc"
    assert a.tica_switch_cv2 is False
    assert a.secondary_cv_model == side["secondary_cv_model"]


def test_sidecar_artifacts_load_as_one_runtime_bound_to_the_fixture_topology(synthetic_swarm, swarm_args):
    from gareus.config import _load_config_file
    from gareus.cv_selection.models import PairModelRuntime
    out, report = _analyze(synthetic_swarm, swarm_args)
    side = _load_config_file(out / "swarm" / "analysis" / "ladder_run_args.yaml")
    rt = PairModelRuntime.load(side["secondary_cv_model"], side["secondary_cv_candidate_set"],
                               side["secondary_cv_feature_schema"])
    assert rt.pair_sha256 == report["cv_selection"]["pair_model_sha256"]
    assert rt.j == report["cv_selection"]["selected_component_index"]
    rt.check_topology(PHI_TORSIONS, PSI_TORSIONS)
    # Same width, permuted torsions: a different coordinate, refused by name.
    import pytest
    with pytest.raises(RuntimeError, match="different coordinate"):
        rt.check_topology(PSI_TORSIONS, PHI_TORSIONS)
    # The frozen anchor carries the run's contact parameters.
    ca = rt.contact_args()
    assert ca.contact_r0_a == 12.0 and ca.contact_beta_a_inv == 3.0 and ca.contact_min_sequence_separation == 4


def test_two_dimensional_ladder_rows_are_complete_and_loadable(synthetic_swarm, swarm_args):
    from gareus.windows import load_explicit_2d_window_csv
    out, report = _analyze(synthetic_swarm, swarm_args)
    csv_path = out / "swarm" / "analysis" / "windows_lambda_ladder.csv"
    with csv_path.open() as fh:
        recs = list(csv.DictReader(fh))
    n_rungs = len(report["ladder"]["lambdas"]) if "ladder" in report else len({r["gamd_lambda"] for r in recs})
    assert len(recs) == report["n_states"] and len(recs) % n_rungs == 0

    class Args:
        primary_cv = "nonlocal-contacts"
        secondary_cv = "residual-torsion-pc"
        contact_k_kcal = None
        secondary_cv_k_kcal = 1.0
    centers, ks, sec_c, sec_k, meta, _ = load_explicit_2d_window_csv(Args(), csv_path)
    assert len(centers) == len(sec_c) == len(sec_k) == len(recs) and meta["enabled"] is True
    assert any(k > 0 for k in sec_k), "a selected pair must restrain CV2 somewhere"
