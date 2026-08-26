from __future__ import annotations

import json
import re
import subprocess
import sys


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
        def getState(self, getPositions=False, getEnergy=False, enforcePeriodicBox=False):
            assert getPositions is True
            assert getEnergy is False
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
