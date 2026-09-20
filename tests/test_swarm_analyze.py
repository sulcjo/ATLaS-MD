import csv, json, pathlib, tempfile, types
import numpy as np
from gareus.swarm.members import TRACE_COLUMNS


def _fake_round(out, n_members=6, n_frames=300, shift_odd=0.0):
    rng = np.random.default_rng(0)
    rd = out / "swarm" / "round_000"; rd.mkdir(parents=True)
    with (rd / "plan.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["member_id", "cell_id", "cell_cv1", "cell_rg", "cell_e2e", "seed_id", "seed_pdb", "replicate", "velocity_seed"]); w.writeheader()
        for m in range(n_members):
            w.writerow({"member_id": m, "cell_id": f"{m % 2}_0_0", "cell_cv1": m % 2, "cell_rg": 0, "cell_e2e": 0, "seed_id": f"s{m}", "seed_pdb": "/x", "replicate": 0, "velocity_seed": m})
    json.dump({"n_cells": 2, "replicates_per_cell": 3, "n_members": n_members, "budget_ns": 6.0, "seed_ns": 1.0,
               "edges": {"cv1": [0, 0.5, 1], "rg": [0, 2], "e2e": [0, 4]}}, (rd / "plan_meta.json").open("w"))
    for m in range(n_members):
        md = rd / f"member_{m:04d}"; (md / "frames").mkdir(parents=True)
        with (md / "trace.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=TRACE_COLUMNS); w.writeheader()
            for i in range(n_frames):
                w.writerow({"frame": i, "t_ps": 2.0 * (i + 1), "cv1": float(rng.beta(2, 4)), "rg_nm": 0.6, "e2e_nm": 1.0,
                            "v_pep_kj": float(rng.normal(-50 + (shift_odd if m % 2 else 0), 5)), "v_dih_kj": float(rng.normal(20, 2)), "potential_kj": -1e4})
                if (i + 1) % 10 == 0:
                    (md / "frames" / f"frame_{i:05d}.pdb").write_text("ATOM      1  CA  GLY A   1       0.000   0.000   0.000  1.00  0.00           C\nEND\n")
        json.dump({"member_id": m, "n_frames": n_frames, "graft_fallback": False, "wall_s": 10.0, "ns_per_day": 400.0}, (md / "done.json").open("w"))
    return rd


def _args():
    return types.SimpleNamespace(temperature_k=300.0, sigma0p_kcal_mol=6.0, sigma0d_kcal_mol=6.0, swarm_n_windows=8, swarm_overlap_sigma=1.5,
                                 swarm_target_beta_sigma=1.0, swarm_min_rungs=3, swarm_max_rungs=12, swarm_ess_floor=50,
                                 swarm_seeds_per_window=2, swarm_discard_block_frames=20, swarm_min_discard_ps=0.0,
                                 swarm_output_interval_ps=2.0, contact_adaptive_max_k_kcal=1200.0, contact_adaptive_min_k_kcal=5.0, swarm_round=0)


def test_analyze_writes_every_artifact_and_passes_on_clean_data():
    from gareus.swarm.analyze import analyze_swarm_stage
    from gareus.pep_gamd import PepGamdEnvelope
    out = pathlib.Path(tempfile.mkdtemp()); _fake_round(out)
    rep = analyze_swarm_stage(out, _args())
    an = out / "swarm" / "analysis"
    for name in ("envelope_discard.json", "shared_gamd_setup/shared_gamd_setup_globals.json", "ladder_design.json",
                 "windows_lambda_ladder.csv", "ladder_run_args.yaml", "seed_bank/final_survivor_seeds.csv", "swarm_gate.json", "swarm_report.json"):
        assert (an / name).exists(), name
    from gareus.config import _load_config_file
    side = _load_config_file(an / "ladder_run_args.yaml")
    assert side["starting_structures"]["seed_conformers_dir"] == str((an / "seed_bank").resolve())
    assert side["starting_structures"]["seed_selection_mode"] == "active-cv" and side["gamd"]["gamd_boost_type"] == "pep-gamd-lower-dual"
    assert rep["status"] == "pass"
    env = PepGamdEnvelope.from_json(an / "shared_gamd_setup/shared_gamd_setup_globals.json")
    assert env.vmax_total > env.vmin_total
    ld = json.load((an / "ladder_design.json").open())
    assert ld["n_states"] == 8 * len(ld["lambdas"]) and ld["lambdas"][0] == 0.0 and ld["lambdas"][-1] == 1.0
    assert len(list(csv.DictReader((an / "windows_lambda_ladder.csv").open()))) == ld["n_states"]
    # probe metadata (autotuned CV1 upper bound) is recorded in the report, keyed by
    # "ladder_design" -- library_q99 survives as a reported diagnostic only (no
    # seed_descriptors.csv here, so it falls back to plan_meta's edges).
    probe = rep["ladder_design"]
    assert set(probe) >= {"library_q99", "autotuned", "hi", "hi_initial", "tol", "max_nearest_seed_gap"}
    assert probe["hi"] <= probe["hi_initial"] + 1e-9
    assert probe["max_nearest_seed_gap"] <= probe["tol"] + 1e-9


def test_analyze_fails_gate_and_withholds_windows_csv_on_shifted_halves():
    from gareus.swarm.analyze import analyze_swarm_stage
    out = pathlib.Path(tempfile.mkdtemp()); _fake_round(out, shift_odd=40.0)
    rep = analyze_swarm_stage(out, _args())
    assert rep["status"] == "fail" and rep["extension"]["extra_replicates_per_cell"] >= 3
    assert not (out / "swarm" / "analysis" / "windows_lambda_ladder.csv").exists()


def test_analyze_ladder_ess_warnings_no_longer_block_the_round_and_windows_csv_is_written():
    """UPDATED 2026-09-08: ladder_ess_gate is now advisory (item 2 of the reintegration --
    see gareus/swarm/gates.py's module comment for the measured overlap/lambda=0
    cross-check justification). A round that would previously have failed solely on
    ladder ESS (every rung below an artificially-raised floor) now passes, and its
    per-rung ESS / extrapolated_from_rung diagnostics survive as warnings in both
    swarm_gate.json on disk and the in-memory report -- both the flat report["warnings"]
    (what cli.py's --swarm-stage analyze console printout actually surfaces) and the
    nested report["gate"]["warnings"] -- so nothing is silently lost."""
    from gareus.swarm.analyze import analyze_swarm_stage
    out = pathlib.Path(tempfile.mkdtemp()); _fake_round(out)
    args = _args()
    args.swarm_ess_floor = 100000  # forces every rung below the floor -> extrapolation
    rep = analyze_swarm_stage(out, args)
    an = out / "swarm" / "analysis"
    assert rep["status"] == "pass"
    assert (an / "windows_lambda_ladder.csv").exists()

    gate_json = json.load((an / "swarm_gate.json").open())
    assert gate_json["status"] == "pass"
    assert gate_json["gates"]["ladder_ess"]["ok"] is True
    assert len(gate_json["warnings"]) > 0
    assert any("ESS" in w for w in gate_json["warnings"])

    assert rep["gate"]["gates"]["ladder_ess"]["ok"] is True
    assert len(rep["gate"]["warnings"]) > 0
    assert any("ESS" in w for w in rep["warnings"])


def test_analyze_rerun_removes_stale_ladder_artifacts_when_gate_flips_to_fail():
    from gareus.swarm.analyze import analyze_swarm_stage
    out = pathlib.Path(tempfile.mkdtemp()); rd = _fake_round(out)
    an = out / "swarm" / "analysis"

    rep1 = analyze_swarm_stage(out, _args())
    assert rep1["status"] == "pass"
    assert (an / "windows_lambda_ladder.csv").exists()
    assert (an / "ladder_run_args.yaml").exists()

    # Rewrite the odd members' traces with a large mean shift so envelope_stability now
    # fails on re-analysis of the SAME round -- a previously-passing gate must not leave
    # stale ladder artifacts behind for S2.
    rng = np.random.default_rng(1)
    for m in range(1, 6, 2):
        md = rd / f"member_{m:04d}"
        with (md / "trace.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=TRACE_COLUMNS); w.writeheader()
            for i in range(300):
                w.writerow({"frame": i, "t_ps": 2.0 * (i + 1), "cv1": float(rng.beta(2, 4)), "rg_nm": 0.6, "e2e_nm": 1.0,
                            "v_pep_kj": float(rng.normal(-10, 5)), "v_dih_kj": float(rng.normal(20, 2)), "potential_kj": -1e4})

    rep2 = analyze_swarm_stage(out, _args())
    assert rep2["status"] == "fail"
    assert rep2.get("withheld_ladder_artifacts") is True
    assert not (an / "windows_lambda_ladder.csv").exists()
    assert not (an / "ladder_run_args.yaml").exists()


def _write_seed_descriptors_csv(rd, cv1_values):
    with (rd / "seed_descriptors.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed_id", "pdb_path", "cv1", "rg_nm", "e2e_nm"]); w.writeheader()
        for i, cv1 in enumerate(cv1_values):
            w.writerow({"seed_id": f"seed_{i:05d}", "pdb_path": "/x", "cv1": cv1, "rg_nm": 0.5, "e2e_nm": 1.0})


def test_analyze_succeeds_and_records_probe_metadata_when_swarm_exceeds_the_library_q99():
    """UPDATED 2026-09-08 (was test_analyze_uses_seed_descriptors_csv_cap_and_records_failure_when_centres_exceed_it,
    which asserted the OLD library-quantil veto's failure): the seed_descriptors.csv library
    is now only a reported diagnostic, never a veto. A narrow library (mimicking r7's real
    ~0-0.069 coverage) whose q99 sits far below the swarm's own observed CV1 samples (beta(2,4),
    reaching up toward ~0.9) must NOT fail the round any more -- the swarm's own pooled frames
    are the seed-bank candidate pool the autotuned probe checks against, and they are dense
    across their own quantile range by construction, so the ladder is built and the CSV is
    written. The library's q99 and the probe metadata are both recorded in the report."""
    from gareus.swarm.analyze import analyze_swarm_stage
    out = pathlib.Path(tempfile.mkdtemp()); rd = _fake_round(out)
    _write_seed_descriptors_csv(rd, [0.001 * i for i in range(50)])
    rep = analyze_swarm_stage(out, _args())
    an = out / "swarm" / "analysis"
    assert rep["status"] == "pass"
    assert (an / "ladder_design.json").exists()
    assert (an / "windows_lambda_ladder.csv").exists()
    assert (an / "envelope_discard.json").exists()
    assert (an / "shared_gamd_setup" / "shared_gamd_setup_globals.json").exists()
    assert (an / "swarm_report.json").exists()

    ld = rep["ladder_design"]
    assert ld["library_q99"] is not None and ld["library_q99"] < 0.05  # the narrow library, reported only
    assert isinstance(ld["autotuned"], bool)
    assert ld["hi"] <= ld["hi_initial"] + 1e-9
    assert ld["tol"] > 0.0
    assert ld["max_nearest_seed_gap"] <= ld["tol"] + 1e-9


def test_analyze_falls_back_to_edges_and_warns_when_seed_descriptors_csv_missing():
    from gareus.swarm.analyze import analyze_swarm_stage
    out = pathlib.Path(tempfile.mkdtemp()); _fake_round(out)  # no seed_descriptors.csv written
    rep = analyze_swarm_stage(out, _args())
    assert rep["status"] == "pass"
    assert any("seed_descriptors.csv" in w for w in rep["warnings"])


def _fake_round1(out, n_members=2, n_frames=100, seed=2):
    rng = np.random.default_rng(seed)
    rd1 = out / "swarm" / "round_001"; rd1.mkdir(parents=True)
    with (rd1 / "plan.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["member_id", "cell_id", "cell_cv1", "cell_rg", "cell_e2e", "seed_id", "seed_pdb", "replicate", "velocity_seed"]); w.writeheader()
        for m in range(n_members):
            w.writerow({"member_id": m, "cell_id": f"{m}_0_0", "cell_cv1": m, "cell_rg": 0, "cell_e2e": 0, "seed_id": f"r1s{m}", "seed_pdb": "/x", "replicate": 0, "velocity_seed": m})
    json.dump({"n_cells": n_members, "replicates_per_cell": 1, "n_members": n_members, "budget_ns": float(n_members), "seed_ns": 1.0},
               (rd1 / "plan_meta.json").open("w"))
    for m in range(n_members):
        md = rd1 / f"member_{m:04d}"; (md / "frames").mkdir(parents=True)
        with (md / "trace.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=TRACE_COLUMNS); w.writeheader()
            for i in range(n_frames):
                w.writerow({"frame": i, "t_ps": 2.0 * (i + 1), "cv1": float(rng.beta(2, 4)), "rg_nm": 0.6, "e2e_nm": 1.0,
                            "v_pep_kj": float(rng.normal(-50, 5)), "v_dih_kj": float(rng.normal(20, 2)), "potential_kj": -1e4})
                if (i + 1) % 10 == 0:
                    (md / "frames" / f"frame_{i:05d}.pdb").write_text("ATOM      1  CA  GLY A   1       0.000   0.000   0.000  1.00  0.00           C\nEND\n")
        json.dump({"member_id": m, "n_frames": n_frames, "graft_fallback": False, "wall_s": 5.0, "ns_per_day": 400.0}, (md / "done.json").open("w"))
    return rd1


def test_analyze_round1_reuses_frozen_envelope_and_ladder_and_reports_out_of_envelope_fraction():
    from gareus.swarm.analyze import analyze_swarm_stage
    out = pathlib.Path(tempfile.mkdtemp()); _fake_round(out)
    an = out / "swarm" / "analysis"

    rep0 = analyze_swarm_stage(out, _args())
    assert rep0["status"] == "pass"
    setup_path = an / "shared_gamd_setup" / "shared_gamd_setup_globals.json"
    ladder_path = an / "ladder_design.json"
    setup_before, setup_mtime_before = setup_path.read_text(), setup_path.stat().st_mtime_ns
    ladder_before = ladder_path.read_text()

    _fake_round1(out)
    args1 = _args(); args1.swarm_round = 1
    rep1 = analyze_swarm_stage(out, args1)

    assert setup_path.read_text() == setup_before
    assert setup_path.stat().st_mtime_ns == setup_mtime_before
    assert ladder_path.read_text() == ladder_before
    assert (an / "seed_bank_round_001" / "final_survivor_seeds.csv").exists()
    assert "out_of_envelope_fraction" in rep1
    assert rep1["out_of_envelope_fraction"]["Total"] is not None
    assert rep1["out_of_envelope_fraction"]["Dihedral"] is not None


def test_probe_pool_is_the_exported_frame_set_not_every_trace_row():
    """Only frames with a written PDB can seed a window (select_window_seed_frames draws from
    frame_candidates), so the upper-bound probe must be measured against that pool, not the
    10x denser trace."""
    from gareus.swarm.analyze import analyze_swarm_stage
    from gareus.pep_gamd import PepGamdEnvelope
    out = pathlib.Path(tempfile.mkdtemp())
    rd = _fake_round(out)
    n_exported = 0
    for md in sorted((rd).glob("member_*")):
        n_exported += len(list((md / "frames").glob("*.pdb")))
    args = _args()
    res = analyze_swarm_stage(out, args)
    probe = res.get("ladder_design", {})
    assert "n_seed_pool" in probe, "the probe must report the pool it measured"
    # Post-discard subset of the exported PDB frames, never the full trace.
    assert 0 < probe["n_seed_pool"] <= n_exported


# ---------------------------------------------------------------------------
# Task 8: automatic CV2 selection wired into swarm analysis (fixtures: tests/conftest.py)
# ---------------------------------------------------------------------------

def test_auto_cv2_writes_a_pair_model_and_a_two_dimensional_ladder(synthetic_swarm, swarm_args):
    from gareus.swarm.analyze import analyze_swarm_stage
    from gareus.config import _load_config_file
    out = synthetic_swarm(with_features=True, wide_anchor=True)
    report = analyze_swarm_stage(out, swarm_args(secondary_cv="auto"))
    an = out / "swarm" / "analysis"
    assert report["cv_selection"]["status"] == "pair", report["cv_selection"]
    for name in ("cv_pair_model.json", "cv_candidate_set.json", "cv_feature_schema.json",
                 "cv_selection_report.json", "windows_lambda_ladder.csv"):
        assert (an / name).exists(), name
    with (an / "windows_lambda_ladder.csv").open() as fh:
        header = fh.readline().strip().split(",")
    assert {"secondary_cv_center", "secondary_cv_k_kcal_mol"} <= set(header)
    side = _load_config_file(an / "ladder_run_args.yaml")
    assert side["cvs"] == {"cv1": "contacts", "cv2": "residual-torsion-pc"}
    assert side["tica_switch_cv2"] is False
    for key in ("secondary_cv_model", "secondary_cv_candidate_set", "secondary_cv_feature_schema"):
        assert pathlib.Path(side[key]).exists()
    assert report["status"] == "pass" and report["gate"]["gates"]["pair"]["ok"] is True
    assert report["cv_selection"]["layout"]["kind"] in ("joint", "sparse")


def test_narrow_anchor_fails_the_pair_gate_whatever_the_fallback(synthetic_swarm, swarm_args):
    from gareus.swarm.analyze import analyze_swarm_stage
    out = synthetic_swarm(with_features=True, wide_anchor=False)
    report = analyze_swarm_stage(out, swarm_args(secondary_cv="auto", cv_selection_fallback="cv1_only"))
    an = out / "swarm" / "analysis"
    assert report["cv_selection"]["status"] == "no_deployable_anchor"
    assert report["status"] == "fail" and report["gate"]["gates"]["pair"]["ok"] is False
    assert not (an / "windows_lambda_ladder.csv").exists()
    assert not (an / "ladder_run_args.yaml").exists()


def test_refuse_fallback_withholds_the_ladder_when_no_component_passes(synthetic_swarm, swarm_args):
    from gareus.swarm.analyze import analyze_swarm_stage
    out = synthetic_swarm(with_features=True, wide_anchor=True)
    report = analyze_swarm_stage(out, swarm_args(secondary_cv="auto", cv_selection_fallback="refuse",
                                                 cv_selection_min_gain_nats=1e9))
    assert report["cv_selection"]["status"] == "cv1_only"
    assert report["status"] == "fail" and report["gate"]["gates"]["pair"]["ok"] is False
    assert not (out / "swarm" / "analysis" / "windows_lambda_ladder.csv").exists()


def test_cv1_only_fallback_writes_a_one_dimensional_ladder_and_says_so(synthetic_swarm, swarm_args):
    from gareus.swarm.analyze import analyze_swarm_stage
    from gareus.config import _load_config_file
    out = synthetic_swarm(with_features=True, wide_anchor=True)
    report = analyze_swarm_stage(out, swarm_args(secondary_cv="auto", cv_selection_fallback="cv1_only",
                                                 cv_selection_min_gain_nats=1e9))
    an = out / "swarm" / "analysis"
    assert report["cv_selection"]["status"] == "cv1_only" and report["status"] == "pass"
    with (an / "windows_lambda_ladder.csv").open() as fh:
        assert "secondary_cv_center" not in fh.readline()
    side = _load_config_file(an / "ladder_run_args.yaml")
    assert side["cvs"] == {"cv1": "contacts", "cv2": "none"}
    assert any("no deployable component" in w for w in report["warnings"])


def test_a_crashed_member_without_features_does_not_abort_the_analysis(synthetic_swarm, swarm_args):
    from gareus.swarm.analyze import analyze_swarm_stage
    out = synthetic_swarm(with_features=True, wide_anchor=True, crash_one_member=True)
    report = analyze_swarm_stage(out, swarm_args(secondary_cv="auto"))
    assert report["cv_selection"]["status"] == "pair"


def test_an_ok_member_without_features_is_a_hard_error_in_auto_mode(synthetic_swarm, swarm_args):
    import pytest
    from gareus.swarm.analyze import analyze_swarm_stage
    out = synthetic_swarm(with_features=True, wide_anchor=True)
    (out / "swarm" / "round_000" / "member_0003" / "torsion_features.npy").unlink()
    with pytest.raises(RuntimeError, match="torsion_features.npy"):
        analyze_swarm_stage(out, swarm_args(secondary_cv="auto"))


def test_explicit_cv2_none_leaves_the_existing_path_untouched(synthetic_swarm, swarm_args):
    from gareus.swarm.analyze import analyze_swarm_stage
    out = synthetic_swarm(with_features=False, wide_anchor=True)
    report = analyze_swarm_stage(out, swarm_args(secondary_cv="none"))
    assert "cv_selection" not in report and report["status"] == "pass"
    assert "pair" not in report["gate"]["gates"]


def test_pair_gate_semantics():
    from gareus.swarm.gates import pair_gate
    assert pair_gate({"status": "pair"}, fallback="refuse")["ok"]
    assert pair_gate({"status": "cv1_only"}, fallback="cv1_only")["ok"]
    assert not pair_gate({"status": "cv1_only"}, fallback="refuse")["ok"]
    assert not pair_gate({"status": "no_deployable_anchor", "anchor_reasons": ["2 resolvable windows"]},
                         fallback="cv1_only")["ok"]
