"""Task 11: swarm-stage provenance -- every ``--swarm-*`` dest (plus the public
``--shared-gamd-setup-dir``) lands in the run manifest's ``method_settings``, the
stage's own gated artifacts are in ``_key_artifact_paths``, the seed library's own
``GENPEPT_turbo_summary.json`` is hashed when present (never ``chignolin.yaml``,
global-constraints.md's binding anchor), and the ``--swarm-stage`` CLI dispatch
actually finalizes the run manifest (it used to fall through to a bare ``return``
and leave the manifest stuck at ``"started"``).

Fixture-free: every test is a plain zero-argument function (see
.superpowers/sdd/2026-09-07-swarm-stage/global-constraints.md's runner note).
"""
import json
import pathlib
import tempfile
import types


def test_method_settings_record_swarm_keys():
    from gareus.provenance import _method_settings
    a = types.SimpleNamespace(
        swarm_stage="run", swarm_seed_ns=1.0, swarm_replicates_per_cell=3, swarm_bins="4,3,3",
        swarm_budget_ns=None, swarm_output_interval_ps=2.0, shared_gamd_setup_dir="x/y",
    )
    ms = _method_settings(a)
    for k in ("swarm_stage", "swarm_seed_ns", "swarm_replicates_per_cell", "swarm_bins",
              "swarm_output_interval_ps", "shared_gamd_setup_dir"):
        assert k in ms


def test_method_settings_record_every_swarm_argparse_dest():
    """Cross-checked against gareus.cli._add_swarm_args's real dests (23 --swarm-* flags),
    not a hand-typed subset that could silently drift out of sync."""
    from gareus.cli import build_gareus_parser
    from gareus.config import _build_known_config_dests
    from gareus.provenance import _method_settings
    known = _build_known_config_dests(build_gareus_parser())
    swarm_dests = {d for d in known if d.startswith("swarm_")} | {"shared_gamd_setup_dir"}
    assert len(swarm_dests) >= 24  # 23 swarm_* dests + the public shared_gamd_setup_dir flag
    a = types.SimpleNamespace(**{d: None for d in swarm_dests})
    ms = _method_settings(a)
    missing = swarm_dests - set(ms)
    assert not missing, f"missing from _method_settings: {sorted(missing)}"


def test_method_settings_record_the_four_new_gate_tolerance_dests():
    """2026-09-08 fix: swarm_stability_sigma_rel_tol, swarm_stability_extrema_sigma_tol,
    swarm_min_done_fraction, and swarm_max_graft_fallback_fraction gained argparse flags
    (they were previously unreachable getattr-only defaults). The cross-check test above
    already covers this generically once these are real argparse dests, but this test
    pins the four names explicitly against a hand-built namespace."""
    from gareus.provenance import _method_settings
    a = types.SimpleNamespace(
        swarm_stability_sigma_rel_tol=0.10, swarm_stability_extrema_sigma_tol=1.0,
        swarm_min_done_fraction=0.9, swarm_max_graft_fallback_fraction=0.10,
    )
    ms = _method_settings(a)
    for k in ("swarm_stability_sigma_rel_tol", "swarm_stability_extrema_sigma_tol",
              "swarm_min_done_fraction", "swarm_max_graft_fallback_fraction"):
        assert k in ms


def test_key_artifact_paths_include_swarm_analysis_artifacts():
    """The controller's ruling named 5 paths as a floor ("_key_artifact_paths must
    include" them), not a ceiling -- ladder_design.json and swarm_gate.json are both
    real files analyze.py writes (write_json(an / "ladder_design.json", ...) and
    write_json(an / "swarm_gate.json", ...)), so both are recorded too (the union of
    the controller's and the brief's lists, both real -- collect_artifact_hashes
    filters on path.exists() so listing an artifact that isn't produced by a given
    run is free)."""
    from gareus.provenance import _key_artifact_paths
    out_dir = pathlib.Path("/tmp/some_run")
    paths = {str(p) for p in _key_artifact_paths(out_dir).values()}
    expected = [
        out_dir / "swarm" / "analysis" / "shared_gamd_setup" / "shared_gamd_setup_globals.json",
        out_dir / "swarm" / "analysis" / "windows_lambda_ladder.csv",
        out_dir / "swarm" / "analysis" / "ladder_run_args.yaml",
        out_dir / "swarm" / "analysis" / "seed_bank" / "final_survivor_seeds.csv",
        out_dir / "swarm" / "analysis" / "swarm_report.json",
        out_dir / "swarm" / "analysis" / "ladder_design.json",
        out_dir / "swarm" / "analysis" / "swarm_gate.json",
    ]
    for p in expected:
        assert str(p) in paths, f"missing key artifact path: {p}"


def test_input_file_hashes_includes_genpept_turbo_summary_when_present():
    from gareus.provenance import _input_file_hashes
    seed_dir = pathlib.Path(tempfile.mkdtemp())
    (seed_dir / "final_survivor_seeds.csv").write_text("survivor_pdb_path\na.pdb\n")
    (seed_dir / "GENPEPT_turbo_summary.json").write_text(json.dumps({"library": "r7", "n_survivors": 1970}))
    a = types.SimpleNamespace(seed_conformers_dir=str(seed_dir))
    hashes = _input_file_hashes(a)
    key = "seed_conformers_GENPEPT_turbo_summary.json"
    assert key in hashes
    # sha256_file's real return shape (gareus.provenance.sha256_file): a computed hash
    # (the file is well under the default hash-size limit) is a non-empty hex string,
    # and hash_skipped is explicitly False -- not an or-chain that could pass on a
    # skipped/errored record.
    assert hashes[key]["hash_skipped"] is False
    assert isinstance(hashes[key]["sha256"], str) and len(hashes[key]["sha256"]) == 64


def test_input_file_hashes_omits_genpept_turbo_summary_when_absent():
    from gareus.provenance import _input_file_hashes
    seed_dir = pathlib.Path(tempfile.mkdtemp())
    (seed_dir / "final_survivor_seeds.csv").write_text("survivor_pdb_path\na.pdb\n")
    a = types.SimpleNamespace(seed_conformers_dir=str(seed_dir))
    hashes = _input_file_hashes(a)
    assert "seed_conformers_GENPEPT_turbo_summary.json" not in hashes


def test_swarm_stage_dispatch_finalizes_run_manifest():
    """cli.py's --swarm-stage dispatch used to print its result and `return` without ever
    calling finalize_run_manifest, so the manifest stayed "started" forever. Monkeypatch
    the driver entry point (imported fresh inside main()'s dispatch block on every call)
    to avoid any real MD and just check the manifest is finalized with the mapped status."""
    import gareus.swarm.driver as driver_mod
    from gareus.cli import main

    out = pathlib.Path(tempfile.mkdtemp())

    def fake_run_swarm_stage(args, out_dir, progress=None):
        return {
            "status": "ok", "round": 0, "n_members": 0, "n_run": 0, "failed_in_range": [],
            "n_skipped_resume": 0, "status_counts": {}, "missing_in_range": [], "plan_meta": {},
        }

    original = driver_mod.run_swarm_stage
    driver_mod.run_swarm_stage = fake_run_swarm_stage
    try:
        main([
            "--seq", "GYDPETGTWG", "--out", str(out), "--swarm-stage", "run",
            "--seed-conformers-dir", str(out), "--tui-mode", "none", "--progress-mode", "none",
        ])
    finally:
        driver_mod.run_swarm_stage = original

    manifest = json.loads((out / "run_manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest.get("end_time_utc")


def test_swarm_stage_dispatch_finalizes_run_manifest_as_failed_on_fail_status():
    import gareus.swarm.analyze as analyze_mod
    from gareus.cli import main

    out = pathlib.Path(tempfile.mkdtemp())

    def fake_analyze_swarm_stage(out_dir, args):
        return {"status": "fail", "reasons": ["no members ran"], "n_members": 0,
                "n_ok_members": 0, "missing_members": [], "failed_members": [],
                "discard_frames": 0, "warnings": []}

    original = analyze_mod.analyze_swarm_stage
    analyze_mod.analyze_swarm_stage = fake_analyze_swarm_stage
    try:
        main([
            "--seq", "GYDPETGTWG", "--out", str(out), "--swarm-stage", "analyze",
            "--seed-conformers-dir", str(out), "--tui-mode", "none", "--progress-mode", "none",
        ])
    finally:
        analyze_mod.analyze_swarm_stage = original

    manifest = json.loads((out / "run_manifest.json").read_text())
    assert manifest["status"] == "failed"
