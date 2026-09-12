from __future__ import annotations

import json
import re
import subprocess
import sys

import pytest


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "gareus", *args],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def test_package_imports() -> None:
    import gareus
    import gareus_peptide

    assert "cli" in gareus.__all__
    assert hasattr(gareus_peptide, "main")
    assert hasattr(gareus_peptide, "run_gareus")


def test_python_m_gareus_short_help() -> None:
    result = _run_cli("--help")
    assert result.returncode == 0
    assert result.stderr == ""
    assert "Minimal command" in result.stdout
    assert "--seq" in result.stdout
    assert "--window-mode" in result.stdout
    assert "--run-mode" in result.stdout
    assert "--cv1" in result.stdout
    assert "--cv2" in result.stdout
    assert "--box-shape" in result.stdout
    assert "dodecahedron" in result.stdout
    assert "run_manifest.json/yaml" in result.stdout
    assert "Method encyclopedia" not in result.stdout


def test_python_m_gareus_heavy_help() -> None:
    result = _run_cli("-hh")
    assert result.returncode == 0
    assert result.stderr == ""
    assert "Method encyclopedia" in result.stdout
    assert "Complete option reference" in result.stdout
    assert "P_accept = min(1, exp(-beta Delta))" in result.stdout
    assert "boxShape = dodecahedron" in result.stdout
    assert "Post-hoc intra/inter peptide energy analysis" in result.stdout
    assert "Exact MD methodology, start to finish" in result.stdout
    assert "OpenMM explicit-solvent Amber14 PME peptide CMD/GaMD/REUS" in result.stdout
    # Encyclopedia topic numbers are assigned sequentially at render time by
    # `_parse_encyclopedia` (gareus/helptext.py) from the prose's own headings,
    # precisely so the TOC can never drift out of sync with the text.  Pinning
    # a literal number here ("0.14 Adaptive-production mode") therefore
    # asserted an implementation detail that any unrelated heading edit is
    # *designed* to change - it was stale and red.  Match "<some number>. Title"
    # instead: immune to renumbering, but still proves the auto-TOC actually
    # rendered that heading as a numbered, jumpable topic rather than merely
    # that the words appear somewhere in the output.
    for topic_title in (
        "Adaptive-production mode",
        "Potential-energy handling and later decomposition",
        "Adaptive production and global runtime pool",
    ):
        assert re.search(rf"\d+\.\s+{re.escape(topic_title)}", result.stdout), (
            f"no numbered encyclopedia topic for {topic_title!r} in -hh output"
        )
    assert "E_ij = k_e q_i q_j / r_ij" in result.stdout
    assert "gareus-energy-decompose -hh" in result.stdout
    assert "Provenance and reproducibility manifest" in result.stdout
    assert "package source hash" in result.stdout
    assert "run_manifest.json" in result.stdout
    assert "--contact-r0-a" in result.stdout
    assert "--run-mode cmd" in result.stdout
    assert "--run-mode hmr-cmd" in result.stdout
    assert "--cv2 rama-map" in result.stdout
    assert "left-alpha" in result.stdout


def test_gareus_peptide_shim_help() -> None:
    result = subprocess.run(
        [sys.executable, "gareus_peptide.py", "--help"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert result.returncode == 0
    assert result.stderr == ""
    assert "Minimal command" in result.stdout
    assert "--seq" in result.stdout


def test_cli_performance_flags_are_documented() -> None:
    result = _run_cli("-hh")
    assert result.returncode == 0
    assert "--no-sample-potential-energy" in result.stdout
    # --no-flush-every-log dropped in v2.0 (internal default; not user-facing)


def test_adaptive_hist_overlap_accepts_numpy_arrays() -> None:
    import numpy as np
    from gareus.adaptive_feedback import _adaptive_hist_overlap

    a = np.linspace(0.0, 1.0, 50)
    b = np.linspace(0.25, 1.25, 50)
    arr_value = _adaptive_hist_overlap(a, b, 0.0, 1.25, bins=20)
    list_value = _adaptive_hist_overlap(a.tolist(), b.tolist(), 0.0, 1.25, bins=20)
    assert np.isfinite(arr_value)
    assert abs(arr_value - list_value) < 1.0e-12


def test_secondary_cv_score_populates_numeric_cache() -> None:
    import numpy as np
    from gareus.cv import secondary_structure_score_from_positions_nm

    positions = np.asarray([
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        [0.2, 0.1, 0.0],
        [0.3, 0.1, 0.1],
        [0.4, 0.2, 0.1],
        [0.5, 0.2, 0.2],
    ], dtype=float)
    ss_info = {
        "enabled": True,
        "mode": "alpha-coil-beta",
        "sigma_deg": 35.0,
        "phi_torsions": [(0, 1, 2, 3)],
        "psi_torsions": [(1, 2, 3, 4)],
    }
    value1 = secondary_structure_score_from_positions_nm(positions, ss_info)
    value2 = secondary_structure_score_from_positions_nm(positions, ss_info)
    assert np.isfinite(value1)
    assert value1 == value2
    assert "_np_alpha_beta_phi_targets" in ss_info


def test_sample_state_reader_can_skip_potential_energy() -> None:
    import numpy as np
    from types import SimpleNamespace
    from gareus.production import primary_secondary_and_potential_from_state

    class Positions:
        def value_in_unit(self, _unit):
            return np.asarray([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], dtype=float)

    class State:
        def __init__(self, get_energy: bool):
            self.get_energy = get_energy
        def getPositions(self, asNumpy=True):
            return Positions()
        def getPotentialEnergy(self):
            raise AssertionError("potential energy should not be requested")

    class Context:
        # `groups=` is load-bearing: production.primary_secondary_and_potential_from_state
        # always passes physical_energy_groups_for_args(args) so a GaMD run reads the
        # PHYSICAL potential, not the boosted one. A fake without the parameter made
        # this test pass only on the base branch and fail against real production.
        def getState(self, getPositions=False, getEnergy=False, enforcePeriodicBox=False,
                     groups=None, **kwargs):
            assert getPositions is True
            assert getEnergy is False
            # Pin that the real call site still SUPPLIES it -- a default-only
            # parameter would let the regression back in silently.
            assert groups is not None, "production must pass an explicit energy-group mask"
            return State(getEnergy)

    unit = SimpleNamespace(nanometer=object(), kilojoule_per_mole=object())
    cv_def = {"mode": "distance", "cv_atom1": 0, "cv_atom2": 1}
    cv, ss, pe = primary_secondary_and_potential_from_state(
        Context(), cv_def, SimpleNamespace(), unit, read_potential_energy=False
    )
    assert cv == 1.0
    assert np.isnan(ss)
    assert np.isnan(pe)



def test_run_mode_and_cv_aliases_parse_to_canonical_values() -> None:
    from gareus.cli import parse_args

    cmd = parse_args(["--seq", "AA", "--run-mode", "cmd", "--cv1", "contacts"])
    assert cmd.run_mode == "cmd"
    assert cmd.primary_cv == "nonlocal-contacts"
    assert cmd.hmr is False

    hmr_cmd = parse_args(["--seq", "AA", "--run-mode", "hmr-cmd"])
    assert hmr_cmd.run_mode == "hmr-cmd"
    assert hmr_cmd.hmr is True
    assert abs(float(hmr_cmd.timestep_fs) - 4.0) < 1.0e-12
    assert abs(float(hmr_cmd.hydrogen_mass_amu) - 3.024) < 1.0e-12
    assert getattr(hmr_cmd, "_run_mode_auto_timestep_fs") is True

    hmr_cmd_explicit = parse_args(["--seq", "AA", "--run-mode", "hmr-cmd", "--timestep-fs", "2.5"])
    assert abs(float(hmr_cmd_explicit.timestep_fs) - 2.5) < 1.0e-12
    assert getattr(hmr_cmd_explicit, "_run_mode_auto_timestep_fs") is False

    hmr = parse_args(["--seq", "AA", "--run-mode", "hmr-gamd"])
    assert hmr.run_mode == "hmr-gamd"
    assert hmr.hmr is True
    assert abs(float(hmr.timestep_fs) - 4.0) < 1.0e-12
    assert abs(float(hmr.hydrogen_mass_amu) - 3.024) < 1.0e-12

    two_d = parse_args(["--seq", "AA", "--cv1", "distance", "--cv2", "rama-map"])
    assert two_d.primary_cv == "distance"
    assert two_d.secondary_cv == "rama-map"
    assert two_d.secondary_cv_centers == [-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0]

    assert getattr(two_d, "_cv2_auto_centers") is True

    acb = parse_args(["--seq", "AA", "--cv2", "alpha-coil-beta"])
    assert acb.secondary_cv_centers == [-0.8, 0.0, 0.8]


def test_genpept_prior_cli_and_selector_smoke() -> None:
    from pathlib import Path
    from types import SimpleNamespace
    from gareus.cli import parse_args
    from gareus.genpept_window_prior import ScoredGenpeptPoint, select_prior_windows_from_points

    parsed = parse_args([
        "--seq", "AA",
        "--cv1", "distance",
        "--cv2", "rama-map",
    ])
    # genpept_prior dropped from CLI; compat shim sets disabled defaults
    assert parsed.genpept_prior_enabled is False
    assert parsed.genpept_prior_dir is None
    assert parsed.secondary_cv == "rama-map"

    args = SimpleNamespace(
        primary_cv="distance",
        secondary_cv="rama-map",
        secondary_cv_centers=[-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0],
        genpept_prior_bins="8,4",
        genpept_prior_max_windows=8,
        genpept_prior_min_hits_per_bin=1,
        genpept_prior_absence_means_unknown=True,
        genpept_prior_snap_secondary_centers=True,
        genpept_prior_basin_windows=2,
        genpept_prior_bridge_windows=2,
        genpept_prior_frontier_windows=2,
        genpept_prior_probe_windows=2,
        default_window_k_kcal_a2=1.0,
        adaptive_k_mode="spacing",
        adaptive_overlap_sigma=1.25,
        adaptive_min_k_kcal_a2=0.05,
        adaptive_max_k_kcal_a2=20.0,
        secondary_cv_k_mode="spacing",
        secondary_cv_k_kcal=50.0,
        secondary_cv_adaptive_overlap_sigma=1.25,
        secondary_cv_adaptive_min_sigma=0.02,
        secondary_cv_adaptive_min_k_kcal=0.0,
        secondary_cv_adaptive_max_k_kcal=500.0,
        secondary_cv_adaptive_k_scale=1.0,
        temperature_k=300.0,
    )
    points = [
        ScoredGenpeptPoint(Path(f"p{i}.pdb"), "final_search_pool", i, primary, secondary, -1000.0 + i, source="GENPEPT")
        for i, (primary, secondary) in enumerate([
            (4.0, -0.95), (4.2, -0.90), (5.5, -0.25), (7.0, 0.25),
            (9.0, 0.95), (9.2, 0.90), (6.2, 0.30), (5.8, -0.30),
        ])
    ]
    windows, graph = select_prior_windows_from_points(points, args)
    assert windows
    assert len(windows) <= 8
    assert graph["n_components"] >= 1
    assert all("primary_cv_center" in row for row in windows)
    assert all("secondary_cv_center" in row for row in windows)
    assert all("secondary_cv_k_kcal_mol" in row for row in windows)
    assert any(str(row["window_type"]).startswith("genpept_") for row in windows)


def test_energy_decomposition_help_is_available() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "gareus.energy_decomposition", "--help"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert result.returncode == 0
    assert "intrapeptide/interpeptide" in result.stdout
    assert "--include-peptide-environment" in result.stdout
    assert "--help-heavy" in result.stdout


def test_energy_decomposition_heavy_help_is_available() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "gareus.energy_decomposition", "-hh"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert result.returncode == 0
    assert "Energy-decomposition method reference" in result.stdout
    assert "Direct nonbonded equation" in result.stdout
    assert "PME caveat" in result.stdout
    assert "intrapeptide_direct_nonbonded_kj_mol" in result.stdout
    assert "Complete option reference" in result.stdout


def test_direct_nonbonded_decomposition_simple_coulomb_pair() -> None:
    import numpy as np
    from gareus.energy_decomposition import direct_nonbonded_pair_energies, COULOMB_KJ_MOL_NM_E2

    positions = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)
    charges = np.asarray([1.0, -1.0], dtype=float)
    sigmas = np.asarray([0.0, 0.0], dtype=float)
    epsilons = np.asarray([0.0, 0.0], dtype=float)

    intra = direct_nonbonded_pair_energies(
        positions, [[0, 1]], [], charges, sigmas, epsilons, {}, include_peptide_environment=False
    )
    inter = direct_nonbonded_pair_energies(
        positions, [[0], [1]], [], charges, sigmas, epsilons, {}, include_peptide_environment=False
    )
    assert abs(intra["intrapeptide_direct_nonbonded_kj_mol"] + COULOMB_KJ_MOL_NM_E2) < 1.0e-12
    assert abs(inter["interpeptide_direct_nonbonded_kj_mol"] + COULOMB_KJ_MOL_NM_E2) < 1.0e-12

def test_provenance_manifest_minimal(tmp_path) -> None:
    from types import SimpleNamespace
    from gareus.provenance import initialize_run_manifest, finalize_run_manifest

    cfg = tmp_path / "input.yaml"
    cfg.write_text("sequence:\n  seq: AA\n")
    (tmp_path / "umbrella_windows.csv").write_text("window,center_A,k_kcal_mol_A2\n0,5.0,1.0\n")
    args = SimpleNamespace(
        seq="AA",
        out=str(tmp_path),
        seed=123,
        config=str(cfg),
        windows_2d_csv=None,
        seed_conformers_dir=None,
        resume=False,
        platform="Reference",
        precision="mixed",
        device_index="",
        replica_device_mode="auto",
        replica_device_map="",
        cpu_threads=1,
        setup_platform="",
        setup_precision="",
        setup_device_index="",
        setup_cpu_threads=0,
        water_model="tip3p",
        box_shape="dodecahedron",
        padding_nm=1.0,
        ionic_strength_molar=0.15,
        temperature_k=300.0,
        pressure_bar=1.0,
        barostat_frequency=25,
        production_ensemble="npt",
        production_barostat_frequency=0,
        timestep_fs=2.0,
        friction_per_ps=1.0,
        nonbonded_cutoff_nm=0.8,
        ewald_error_tolerance=1.0e-4,
        hmr=False,
        hydrogen_mass_amu=0.0,
        primary_cv="distance",
        cv_mode="terminal-ca",
        contact_scheme="atom-pairs",
        contact_atom_selection="heavy",
        window_mode="adaptive",
        secondary_cv="none",
        gamd_boost_type="lower-dual",
        sigma0p_kcal_mol=6.0,
        sigma0d_kcal_mol=6.0,
        gamd_production_steps=10,
        exchange_mode="neighbor",
        exchange_interval=5,
        traj_format="none",
        sample_potential_energy=True,
        flush_every_log=True,
        analysis_array_dtype="float64",
    )
    start = initialize_run_manifest(args, tmp_path, argv=["--seq", "AA"])
    assert start["status"] == "started"
    assert "combined_sha256" in start["package_source"]
    assert start["forcefield"]["forcefield_xml"] == ["amber14-all.xml", "amber14/tip3p.xml"]
    final = finalize_run_manifest(args, tmp_path, status="completed")
    assert final["status"] == "completed"
    assert (tmp_path / "run_manifest.json").exists()
    assert (tmp_path / "run_manifest.yaml").exists()
    assert "umbrella_windows_csv" in final["artifact_hashes"]["window_table_hashes"]
    assert final["input_files"]["config"]["sha256"]


def test_tiny_integration_command_dry_run(tmp_path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "gareus.integration_test", "--dry-run", "--out", str(tmp_path / "tiny")],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert result.returncode == 0
    assert result.stderr == ""
    assert "gareus --seq AA" in result.stdout
    assert "--window-mode manual" in result.stdout
    assert "--windows-a 3.5 4.5" in result.stdout
    assert "--run-mode gamd" in result.stdout
    assert "--production-steps 20" in result.stdout
    assert "--exchange-interval 2" in result.stdout
    assert "--traj-format none" in result.stdout


def test_tiny_integration_dependency_check_can_skip(tmp_path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "gareus.integration_test",
            "--check-deps",
            "--skip-if-missing",
            "--out",
            str(tmp_path / "tiny"),
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert "ok" in payload
    assert "imports" in payload


def test_tiny_integration_cmd_dependency_check_does_not_require_gamd(tmp_path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "gareus.integration_test",
            "--check-deps",
            "--skip-if-missing",
            "--run-mode",
            "hmr-cmd",
            "--out",
            str(tmp_path / "tiny_hmr_cmd"),
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["run_mode"] == "hmr-cmd"
    assert payload["requires_gamd_openmm"] is False
    assert payload["imports"]["gamd.integrator_factory"]["required"] is False


def test_tiny_integration_is_documented_in_help() -> None:
    result = _run_cli("-hh")
    assert result.returncode == 0
    assert "Tiny real workflow test" in result.stdout
    assert "gareus-test-run --dry-run" in result.stdout
    assert "tiny_integration_test_report.json" in result.stdout


@pytest.mark.slow
def test_tiny_lambda_ladder_run_completes_end_to_end_slow() -> None:
    """SLOW: the first real (non-dry-run) execution of the lambda-ladder chain
    in one process. A GA-dipeptide variant of gareus.integration_test's tiny
    real workflow (see build_tiny_run_argv), with one CV1 (distance) window
    replicated at 2 rungs (gamd_lambda in {0.0, 1.0}) instead of the plain
    2-window chain the dry-run tests above only print. Requires OpenMM,
    PeptideBuilder and gamd-openmm. Marked via the standard pytest ``slow``
    marker (registered in pyproject.toml's [tool.pytest.ini_options]
    markers list) so a real pytest run can deselect it with
    ``-m "not slow"``; the fixture-free fallback runner used elsewhere in
    this repo ignores markers entirely and just calls the function, same
    as every other GaMD test here (tests/pep_gamd_fixture.py,
    tests/test_pep_gamd_boost.py already run real OpenMM builds
    unconditionally with no marker). Expect low-single-digit minutes on CPU.
    """
    import shutil
    import tempfile
    from pathlib import Path

    import numpy as np
    import pyarrow.dataset as ds

    from gareus.core import main as gareus_main
    from gareus.mbar_analysis.ladder import load_pep_gamd_envelope
    from gareus.pep_gamd import PepGamdEnvelope

    tmp_path = Path(tempfile.mkdtemp())
    try:
        out_dir = tmp_path / "tiny_ladder"
        windows_csv = tmp_path / "windows.csv"
        windows_csv.write_text(
            "window,primary_cv_mode,primary_cv_center,primary_cv_k_kcal,gamd_lambda\n"
            "0,distance,4.0,1.0,0.0\n"
            "1,distance,4.0,1.0,1.0\n"
        )

        argv = [
            "--seq", "GA",
            "--out", str(out_dir),
            "--seed", "2026",
            "--platform", "CPU",
            "--setup-platform", "CPU",
            "--box-shape", "dodecahedron",
            "--padding-nm", "0.55",
            "--ionic-strength-molar", "0.0",
            "--temperature-k", "300.0",
            "--production-ensemble", "npt",
            "--run-mode", "gamd",
            "--timestep-fs", "0.5",
            "--friction-per-ps", "5.0",
            "--minimize-iterations", "5",
            "--nvt-warmup-steps", "2",
            "--nvt-warmup-timestep-fs", "0.25",
            "--npt-ramp-steps", "2",
            "--npt-ramp-timestep-fs", "0.25",
            "--npt-steps", "4",
            "--window-mode", "manual",
            "--windows-2d-csv", str(windows_csv),
            "--us-starting-structure-mode", "pull",
            "--us-pull-steps-per-window", "2",
            "--us-pull-timestep-fs", "0.25",
            "--us-pull-minimize-iterations", "1",
            "--gamd-boost-type", "pep-gamd-lower-dual",
            "--gamd-cmd-steps", "100",
            "--equil-steps", "100",
            "--gamd-averaging-window", "50",
            "--gamd-multiwindow-recon-prep-steps", "1",
            "--gamd-multiwindow-recon-cmd-steps", "2",
            "--gamd-multiwindow-recon-steps", "2",
            "--gamd-recon-boosted-iters", "1",
            "--gamd-recon-boosted-tol", "0.05",
            "--gamd-multiwindow-recon-report-interval", "1",
            "--exchange-mode", "gibbs-walk",
            "--exchange-interval", "200",
            "--report-interval", "100",
            "--distance-output-interval", "100",
            "--distance-output-mode", "csv",
            "--sample-potential-energy",
            "--production-steps", "2000",
            "--traj-format", "none",
            "--checkpoint-interval", "0",
            "--progress-mode", "none",
            "--tui-mode", "none",
        ]

        # NPT correction staged state: this run is boosted GaMD under the NPT
        # ensemble, which resolves to the biased-MC backend.  Package 1 (the
        # gareus/npt.py controller) has landed and resolves fine, and package
        # A's half of the seam -- pep_gamd.make_npt_target_adapter -- now
        # exists too, bridged lazily by production._resolve_npt_adapter, so
        # the run composes end to end.  Skip only if the adapter factory is
        # genuinely absent from this build (never silently keeping the
        # known-wrong native-barostat acceptance energy).
        import gareus.pep_gamd as _pep_gamd

        if not hasattr(_pep_gamd, "make_npt_target_adapter"):
            pytest.skip(
                "boosted-NPT end-to-end run needs the stage-aware "
                "effective-potential adapter (pep_gamd.make_npt_target_adapter), "
                "which is not implemented in this build"
            )

        gareus_main(argv)

        # -- run completed and left the expected core artifacts --
        assert (out_dir / "run_manifest.json").exists()
        assert (out_dir / "umbrella_windows.csv").exists()

        # -- sample Parquet: v_pep_kj_mol finite everywhere, gamd_lambda in {0,1} --
        samples_dir = out_dir / "samples"
        assert samples_dir.exists(), f"no samples/ directory under {out_dir}"
        sample_table = ds.dataset(str(samples_dir), format="parquet").to_table()
        n_rows = sample_table.num_rows
        assert n_rows > 0, "no sample rows were written"
        v_pep = sample_table.column("v_pep_kj_mol").to_pylist()
        assert all(v is not None and v == v and abs(v) != float("inf") for v in v_pep), (
            f"v_pep_kj_mol has non-finite/NaN entries: {v_pep}"
        )
        v_pep_arr = np.asarray(v_pep, dtype=float)
        assert np.nanstd(v_pep_arr) > 0.0, (
            f"v_pep_kj_mol is constant across all {len(v_pep)} samples (stuck/stale column): {v_pep}"
        )
        gamd_lambdas_seen = set(sample_table.column("gamd_lambda").to_pylist())
        assert gamd_lambdas_seen <= {0.0, 1.0}, f"unexpected gamd_lambda values: {gamd_lambdas_seen}"
        assert gamd_lambdas_seen == {0.0, 1.0}, f"expected both rungs sampled, got: {gamd_lambdas_seen}"

        # -- exchange Parquet: at least one attempted swap --
        exchanges_dir = out_dir / "exchanges"
        assert exchanges_dir.exists(), f"no exchanges/ directory under {out_dir}"
        exchange_table = ds.dataset(str(exchanges_dir), format="parquet").to_table()
        assert exchange_table.num_rows >= 1, "no exchange attempts were recorded"

        # -- run_manifest.json: state_gamd_lambdas == [0.0, 1.0] --
        manifest = json.loads((out_dir / "run_manifest.json").read_text())
        method_settings = manifest.get("method_settings", {})
        assert method_settings.get("state_gamd_lambdas") == [0.0, 1.0], (
            f"state_gamd_lambdas = {method_settings.get('state_gamd_lambdas')!r}"
        )

        # -- shared GaMD envelope: PepGamdEnvelope.from_json loads it, via the
        # canonical two-location loader (single-production-run convention writes
        # the bare filename directly under out_dir; see gareus/mbar_analysis/
        # ladder.py's load_pep_gamd_envelope and gareus/production.py's
        # write_json(out_dir / "shared_gamd_setup_globals.json", ...)). --
        envelope = load_pep_gamd_envelope(out_dir)
        assert envelope is not None, "no shared_gamd_setup_globals.json found (bare or nested)"
        for candidate in (
            out_dir / "shared_gamd_setup_globals.json",
            out_dir / "global_shared_gamd_setup" / "shared_gamd_setup_globals.json",
        ):
            if candidate.exists():
                envelope_path = candidate
                break
        else:
            raise AssertionError("shared_gamd_setup_globals.json missing from both known locations")
        envelope_direct = PepGamdEnvelope.from_json(envelope_path)
        assert envelope_direct.k0max_total is not None and envelope_direct.k0max_dih is not None

        # -- MBAR/PMF post-processing: ladder_crosscheck must not be "fail", and
        # both lambda states must actually be present (a "skipped: no lambda=0
        # states" would mean the ladder machinery never engaged) --
        import analyze_gareus_mbar as agm

        analyze_rc = agm.main([str(out_dir), "--no-adaptive-diag", "--no-rg"])
        assert analyze_rc == 0, f"analyze_gareus_mbar.main returned {analyze_rc}"
        summary_candidates = list(out_dir.rglob("pmf_summary.json"))
        assert summary_candidates, f"pmf_summary.json not written anywhere under {out_dir}"
        summary_path = summary_candidates[0]
        summary = json.loads(summary_path.read_text())
        lcc = summary.get("ladder_crosscheck")
        assert isinstance(lcc, dict) and "status" in lcc, f"no ladder_crosscheck block in pmf_summary.json: {summary.keys()}"
        assert lcc["status"] != "fail", f"ladder cross-check FAILED: {lcc}"
        if lcc["status"] == "skipped":
            # gareus/mbar_analysis/crosscheck.py's ladder_crosscheck() reports
            # n_lambda0_samples=0 for BOTH "no λ=0 states in state_lambdas" and "λ=0
            # state(s) declared but hold zero samples" -- the two skip reasons that
            # mean the ladder machinery never actually engaged (not acceptable
            # here). A positive n_lambda0_samples with "skipped" instead means the
            # bin-count gate fell back (too few bins for a verdict on this tiny
            # run's handful of samples) -- an acceptable, data-volume-driven skip,
            # not a wrong-reason one. Check the structured field, not the
            # human-readable "reason" text: that text legitimately contains the
            # substring "λ=0" in the too-few-bins message too (e.g. "full/λ=0
            # pair"), so string-matching on it cannot distinguish the two cases.
            assert int(lcc.get("n_lambda0_samples", 0)) > 0, (
                f"ladder cross-check skipped with zero λ=0 samples (ladder never engaged): {lcc}"
            )
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_combined_genpept_gareus_config_parses(tmp_path) -> None:
    cfg = tmp_path / "combined.yaml"
    cfg.write_text(
        "schema_version: '1.0'\n"
        "sequence:\n"
        "  seq: GYDPETGTWG\n"
        "conformer_generation:\n"
        "  out: seeds/\n"
        "  n: 50000\n"
        "  n_candidate_seeds: 100\n"
        "  n_final_seeds: 32\n"
        "  two_stage: true\n"
        "  basin_hop: true\n"
        "simulation:\n"
        "  run_mode: hmr-gamd\n"
        "cvs:\n"
        "  cv1: distance\n"
        "  cv2: rama-map\n"
        "starting_structures:\n"
        "  seed_conformers_dir: seeds/\n"
    )
    from gareus.cli import parse_args as parse_gareus_args
    gargs = parse_gareus_args(["--config", str(cfg)])
    assert gargs.seq == "GYDPETGTWG"
    assert gargs.run_mode == "hmr-gamd"
    assert abs(float(gargs.timestep_fs) - 4.0) < 1.0e-12
    assert gargs.primary_cv == "distance"
    assert gargs.secondary_cv == "rama-map"
    assert str(gargs.seed_conformers_dir) == "seeds"

    import GENPEPT
    sargs = GENPEPT.parse_args(["--config", str(cfg)])
    assert sargs.seq == "GYDPETGTWG"
    assert str(sargs.out) == str(tmp_path / "seeds")
    assert sargs.n == 50000
    assert sargs.n_candidate_seeds == 100
    assert sargs.n_final_seeds == 32
    assert sargs.two_stage is True
    assert sargs.basin_hop is True


def test_contacts_manual_window_mode_accepts_explicit_windows_2d_csv_without_contact_centers() -> None:
    """Regression for a validation gap found while wiring the lambda-ladder pilot
    config (examples/chignolin_lambda_ladder_pilot.yaml): --cv1 contacts +
    --window-mode manual used to unconditionally require --contact-centers, even
    though the --windows-2d-csv explicit-window path never reads contact_centers
    at all (production.py bypasses choose_windows()/adaptive_contact_centers()
    entirely whenever windows_2d_csv is set). The check predates the
    lambda-ladder work (introduced for --primary-cv nonlocal-contacts, then
    carried over unchanged when generic contact --windows-2d-csv support was
    added in f939a8d), but blocked the first real end-to-end exercise of the
    ladder chain, which needs exactly this combination."""
    import tempfile
    from pathlib import Path

    from gareus.cli import parse_args

    tmp_path = Path(tempfile.mkdtemp())
    windows_csv = tmp_path / "windows.csv"
    windows_csv.write_text(
        "window,primary_cv_mode,primary_cv_center,primary_cv_k_kcal,gamd_lambda\n"
        "0,contacts,0.25,800,0.0\n"
        "1,contacts,0.25,800,1.0\n"
    )
    args = parse_args([
        "--seq", "GA",
        "--cv1", "contacts",
        "--window-mode", "manual",
        "--windows-2d-csv", str(windows_csv),
    ])
    assert args.primary_cv == "nonlocal-contacts"
    assert str(args.windows_2d_csv) == str(windows_csv)

    # Without --windows-2d-csv (and without --contact-centers), the same
    # combination must still raise -- this only relaxes the explicit-table case.
    try:
        parse_args(["--seq", "GA", "--cv1", "contacts", "--window-mode", "manual"])
        raise AssertionError("expected ValueError for contacts+manual with no window source")
    except ValueError as exc:
        assert "requires --contact-centers" in str(exc)


def test_combined_genpept_gareus_config_documented_in_heavy_help() -> None:
    result = _run_cli("-hh")
    assert result.returncode == 0
    assert "Combined GENPEPT -> GAREUS YAML" in result.stdout
    assert "python GENPEPT.py --config" in result.stdout
    assert "not one linked TUI" in result.stdout


def test_seed_selection_options_parse() -> None:
    from gareus.cli import parse_args

    args = parse_args([
        "--seq", "AA",
        "--cv1", "contacts",
        "--cv2", "rama-map",
        "--seed-conformers-dir", "seeds",
        "--seed-selection-mode", "active-cv",
        "--seed-cv2-weight", "2.5",
        "--seed-max-reuse-per-conformer", "1",
    ])
    assert args.primary_cv == "nonlocal-contacts"
    assert args.secondary_cv == "rama-map"
    assert str(args.seed_conformers_dir) == "seeds"
    assert args.seed_selection_mode == "active-cv"
    assert abs(float(args.seed_cv2_weight) - 2.5) < 1.0e-12
    assert args.seed_max_reuse_per_conformer == 1


def test_cv_discovery_suggests_cv2_for_distance_only_run(tmp_path) -> None:
    import numpy as np
    from gareus.cv_discovery import suggest_cvs

    (tmp_path / "umbrella_windows.csv").write_text(
        "window,distance_center_A,distance_k_kcal_mol_A2,primary_cv\n"
        "0,3.5,1.0,distance\n"
        "1,4.5,1.0,distance\n"
    )
    np.savez(
        tmp_path / "analysis_arrays.npz",
        cv_A=np.asarray([3.4, 3.6, 4.4, 4.6], dtype=float),
        window=np.asarray([0, 0, 1, 1], dtype=np.int32),
        umbrella_reduced_bias_nk=np.zeros((4, 2), dtype=float),
    )
    payload = suggest_cvs(tmp_path)
    titles = [s["title"] for s in payload["suggestions"]]
    assert any("Only a primary CV" in t for t in titles)
    assert any("Distance CV" in t for t in titles)


def test_cv_discovery_command_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "gareus.cv_discovery", "--help"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert result.returncode == 0
    assert "CV/window suggestion" in result.stdout
    assert "--run-dir" in result.stdout


def test_cv_discovery_is_documented_in_help() -> None:
    result = _run_cli("-hh")
    assert result.returncode == 0
    assert "Analysis-driven CV suggestion report" in result.stdout
    assert "gareus-suggest-cvs --run-dir" in result.stdout
    assert "seed_selection_report.csv" in result.stdout


def test_genpept_library_scores_active_cv_space(tmp_path) -> None:
    from types import SimpleNamespace
    from gareus.seeding import load_genpept_conformer_library

    class Atom:
        def __init__(self, index, name):
            self.index = index
            self.name = name
    class Residue:
        def __init__(self, index, name, atoms):
            self.index = index
            self.name = name
            self._atoms = atoms
        def atoms(self):
            return iter(self._atoms)
    class Topology:
        def __init__(self):
            self._res = [
                Residue(0, "GLY", [Atom(0, "N"), Atom(1, "CA"), Atom(2, "C")]),
                Residue(1, "ALA", [Atom(3, "N"), Atom(4, "CA"), Atom(5, "C")]),
            ]
        def residues(self):
            return iter(self._res)

    seed_dir = tmp_path / "seeds"
    seed_dir.mkdir()
    pdb = seed_dir / "seed0.pdb"
    coords = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        (3.0, 0.0, 0.0),
        (4.0, 0.0, 0.0),
        (5.0, 0.0, 0.0),
    ]
    lines = []
    names = ["N", "CA", "C", "N", "CA", "C"]
    for i, ((x, y, z), name) in enumerate(zip(coords, names), start=1):
        lines.append(f"ATOM  {i:5d} {name:^4s} ALA A   1    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n")
    pdb.write_text("".join(lines))
    (seed_dir / "final_survivor_seeds.csv").write_text("survivor_pdb_path\nseed0.pdb\n")
    primary = {"mode": "distance", "cv_atom1": 1, "cv_atom2": 4, "units": "A", "contact_pairs": []}
    secondary = {"enabled": True, "mode": "alpha-coil-beta", "sigma_deg": 35.0, "phi_torsions": [(0,1,2,3)], "psi_torsions": [(1,2,3,4)]}
    lib = load_genpept_conformer_library(seed_dir, primary_cv_def=primary, args=SimpleNamespace(), topology=Topology(), secondary_cv_metadata=secondary)
    assert len(lib) == 1
    assert abs(float(lib[0]["primary_cv_value"]) - 3.0) < 1.0e-12
    assert "secondary_cv_value" in lib[0]


def test_genpept_library_reroots_stale_absolute_survivor_path(tmp_path) -> None:
    """final_survivor_seeds.csv bakes absolute paths at GENPEPT-generation time.

    If the run tree is later moved/renamed (e.g. RUNS/runs3 -> RUNS/runs_rdy),
    those baked paths point nowhere. The loader must re-root them under the
    current seed_conformers_dir by matching the genpept output directory's own
    name, instead of silently skipping every survivor.
    """
    from gareus.seeding import load_genpept_conformer_library

    seed_dir = tmp_path / "runs_rdy" / "PEP" / "PEP_genpept_r3"
    survivors_dir = seed_dir / "final_implicit_survivor_seeds"
    survivors_dir.mkdir(parents=True)
    pdb = survivors_dir / "survivor_000.pdb"
    lines = [
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n",
        "ATOM      2  CA  ALA A   2       3.000   0.000   0.000  1.00  0.00           C\n",
    ]
    pdb.write_text("".join(lines))

    stale_abs = "/old/runs3/PEP/PEP_genpept_r3/final_implicit_survivor_seeds/survivor_000.pdb"
    (seed_dir / "final_survivor_seeds.csv").write_text(f"survivor_pdb_path\n{stale_abs}\n")

    lib = load_genpept_conformer_library(seed_dir, cv_atom1=0, cv_atom2=1)
    assert len(lib) == 1
    assert abs(float(lib[0]["primary_cv_value"]) - 3.0) < 1.0e-12


def test_json_ready_serializes_path_objects():
    from pathlib import Path
    import json
    import yaml
    from gareus.io import _json_ready

    payload = {"seed_conformers_dir": Path("chignolin_genpept_seeds")}
    ready = _json_ready(payload)
    assert ready == {"seed_conformers_dir": "chignolin_genpept_seeds"}
    json.dumps(ready)
    yaml.safe_dump(ready)


def test_contact_cv_uses_tanh_stable_switch() -> None:
    import inspect
    import numpy as np
    from types import SimpleNamespace
    from gareus.cv import nonlocal_contact_cv_from_positions_nm, primary_cv_value_from_positions_nm
    import gareus.forces as forces

    # The OpenMM contact umbrella expression should not use exp(); the tanh form
    # is algebraically equivalent but avoids overflow on GPU/fast-math paths.
    src = inspect.getsource(forces.add_contact_umbrella_force)
    assert "contact_expr" in src
    expr_line = next(line for line in src.splitlines() if "contact_expr =" in line)
    assert "tanh" in expr_line
    assert "exp(" not in expr_line

    args = SimpleNamespace(contact_r0_a=4.5, contact_beta_a_inv=6.0, contact_normalize=True)
    positions = np.asarray([[0.0, 0.0, 0.0], [1000.0, 0.0, 0.0]], dtype=float)
    val = nonlocal_contact_cv_from_positions_nm(positions, [(0, 1, 1.0)], args)
    assert np.isfinite(val)
    assert val == 0.0

    primary = {
        "mode": "nonlocal-contacts",
        "_np_idx_i": np.asarray([0], dtype=np.int32),
        "_np_idx_j": np.asarray([1], dtype=np.int32),
        "_np_weights": np.asarray([1.0], dtype=float),
        "_np_r0_nm": 0.45,
        "_np_beta_nm_inv": 60.0,
        "_np_norm_denom": 1.0,
        "contact_normalize": True,
    }
    v2 = primary_cv_value_from_positions_nm(positions, primary, args)
    assert np.isfinite(v2)
    assert v2 == 0.0


def test_production_threadpool_import_is_py314_safe() -> None:
    """`gareus.production` must hold a real, working ThreadPoolExecutor class.

    The hazard this guards is the aliased-submodule import form,
    ``import concurrent.futures as concurrent`` followed by
    ``concurrent.futures.ThreadPoolExecutor(...)``: the alias binds the
    *submodule*, not the ``concurrent`` package, so the attribute path is
    dead - verified on this interpreter (3.14) it raises
    ``AttributeError: module 'concurrent.futures' has no attribute
    'futures'``, and it does so lazily, at first thread-pool construction,
    i.e. only once a real production run is already several minutes in.

    Asserted as the actual property (the name is bound eagerly at import
    time and is the genuine class, and a pool built from it really runs a
    task) rather than as a source literal.  The two negative greps below are
    kept only as a cheap tripwire for the specific broken spelling; they are
    not the property, and this test would still fail correctly if the import
    regressed in some new way they do not match.
    """
    import concurrent.futures
    import inspect

    import gareus.production as production

    # Eagerly bound at module level by `from concurrent.futures import
    # ThreadPoolExecutor`, and the genuine stdlib class - not a shim, and not
    # an attribute lookup deferred to run time.
    assert production.ThreadPoolExecutor is concurrent.futures.ThreadPoolExecutor

    # ... and a pool built the way production builds them actually executes.
    with production.ThreadPoolExecutor(max_workers=1) as pool:
        assert pool.submit(lambda x: x + 1, 41).result(timeout=30) == 42

    src = inspect.getsource(production)
    assert "import concurrent.futures as concurrent" not in src
    assert "concurrent.futures.ThreadPoolExecutor" not in src

    # The only production consumer of that name is _ReplicaAffinityExecutor,
    # which builds one single-worker pool per replica.  Assert the link here so
    # a future refactor cannot satisfy the import property above while the
    # replica pools are built from something else entirely.  The affinity
    # invariant itself (each replica pinned to one OS thread for the lifetime of
    # the executor) is covered by tests/test_replica_affinity_executor.py, which
    # asserts on observed threading.get_ident() values rather than on
    # construction arguments.
    affinity_pool = production._ReplicaAffinityExecutor(2)
    try:
        assert len(affinity_pool._pools) == 2
        for _p in affinity_pool._pools:
            assert isinstance(_p, concurrent.futures.ThreadPoolExecutor)
            assert _p._max_workers == 1
        assert affinity_pool.submit(1, lambda: 7).result(timeout=30) == 7
    finally:
        affinity_pool.shutdown(wait=True)


def test_documented_quiet_pytest_invocation_still_prints_its_count_line() -> None:
    """`python -m pytest -q tests/` must still report "N passed" at the end.

    ``[tool.pytest.ini_options] addopts`` is *prepended* to the command line,
    so any ``-q`` living there compounds with the ``-q`` in this repo's own
    documented invocation: ``-q`` is a counting flag, and ``-qq`` suppresses
    pytest's final count line entirely.  That produced runs which printed
    bare ``FAILED`` lines and then nothing at all - indistinguishable, to a
    human or to a verification agent reading the tail of the output, from a
    suite that aborted mid-run.  Hence: no quiet flag in ``addopts``.

    Checked end-to-end (a real nested pytest run against this repo's real
    config via ``-c``, since the effect only exists once argparse has merged
    addopts with the CLI) rather than by grepping pyproject.toml for a
    literal, so any future way of re-introducing compounding verbosity flags
    is caught too.
    """
    import re
    import tempfile
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as tmp:
        probe = Path(tmp) / "test_addopts_probe.py"
        probe.write_text("def test_trivially_passes():\n    assert True\n")
        result = subprocess.run(
            [
                sys.executable, "-m", "pytest",
                "-c", str(root / "pyproject.toml"),
                "-q",
                "-p", "no:cacheprovider",
                str(probe),
            ],
            check=False,
            text=True,
            cwd=tmp,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    assert result.returncode == 0, result.stdout
    assert re.search(r"\d+ passed", result.stdout), (
        "pytest printed no count line - addopts is probably re-introducing a "
        f"quiet flag that compounds with the documented -q:\n{result.stdout}"
    )
