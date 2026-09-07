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


def test_analyze_fails_gate_and_withholds_windows_csv_on_shifted_halves():
    from gareus.swarm.analyze import analyze_swarm_stage
    out = pathlib.Path(tempfile.mkdtemp()); _fake_round(out, shift_odd=40.0)
    rep = analyze_swarm_stage(out, _args())
    assert rep["status"] == "fail" and rep["extension"]["extra_replicates_per_cell"] >= 3
    assert not (out / "swarm" / "analysis" / "windows_lambda_ladder.csv").exists()


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


def test_analyze_uses_seed_descriptors_csv_cap_and_records_failure_when_centres_exceed_it():
    from gareus.swarm.analyze import analyze_swarm_stage
    out = pathlib.Path(tempfile.mkdtemp()); rd = _fake_round(out)
    # A narrow library (mimicking r7's real ~0-0.069 coverage) whose q99 sits far below
    # the swarm's own observed CV1 samples (beta(2,4), reaching up toward ~0.9) -- the
    # cap must reject the ladder design rather than silently writing an unreachable
    # window, and the failure must be RECORDED (report written, no exception escapes).
    _write_seed_descriptors_csv(rd, [0.001 * i for i in range(50)])
    rep = analyze_swarm_stage(out, _args())
    an = out / "swarm" / "analysis"
    assert rep["status"] == "fail"
    assert any("exceeds" in r and "q99" in r for r in rep["reasons"])
    assert not (an / "ladder_design.json").exists()
    assert not (an / "windows_lambda_ladder.csv").exists()
    assert (an / "envelope_discard.json").exists()
    assert (an / "shared_gamd_setup" / "shared_gamd_setup_globals.json").exists()
    assert (an / "swarm_report.json").exists()


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
