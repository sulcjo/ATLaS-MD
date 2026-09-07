import json, pathlib, tempfile


def _write(path, vmax_t, vmin_t, sig_t, k0_t, vmax_d=50.0, vmin_d=0.0, sig_d=4.0, k0_d=0.5):
    g = {"Vmax_Total": vmax_t, "Vmin_Total": vmin_t, "Vavg_Total": 0.5 * (vmax_t + vmin_t), "sigmaV_Total": sig_t, "k0_Total": k0_t,
         "k_Total": k0_t / (vmax_t - vmin_t), "threshold_energy_Total": vmax_t, "sigma0_Total": 25.1,
         "Vmax_Dihedral": vmax_d, "Vmin_Dihedral": vmin_d, "Vavg_Dihedral": 25.0, "sigmaV_Dihedral": sig_d, "k0_Dihedral": k0_d,
         "k_Dihedral": k0_d / (vmax_d - vmin_d), "threshold_energy_Dihedral": vmax_d, "sigma0_Dihedral": 25.1}
    json.dump({"all_globals": g, "joint_envelope": {"Total": {"sigmaV_kj_mol": sig_t}, "Dihedral": {"sigmaV_kj_mol": sig_d}}}, path.open("w"))


def test_compare_passes_within_tolerance_and_fails_outside():
    from gareus.swarm.compare_pilot import compare_envelopes
    d = pathlib.Path(tempfile.mkdtemp())
    _write(d / "swarm.json", 100.0, -100.0, 20.0, 0.6); _write(d / "pilot.json", 110.0, -95.0, 22.0, 0.55)
    r = compare_envelopes(d / "swarm.json", d / "pilot.json")
    assert r["status"] == "pass" and r["freeze_allowed"] and r["groups"]["Total"]["ok"]
    _write(d / "pilot2.json", 100.0, -100.0, 40.0, 0.3)             # σ doubled, k0 halved
    r2 = compare_envelopes(d / "swarm.json", d / "pilot2.json")
    assert r2["status"] == "fail" and not r2["freeze_allowed"] and any("Total" in s for s in r2["reasons"])


def test_realistic_pilot_numbers_pass_when_swarm_matches():
    from gareus.swarm.compare_pilot import compare_envelopes
    d = pathlib.Path(tempfile.mkdtemp())
    # Total: Vavg -2961.4, Vmin -3455.5, Vmax -2441.9, sigma 179.5, k0 0.273
    # Dihedral: Vavg 449.0, Vmin 377.3, Vmax 529.8, sigma 21.0, k0 1.0
    _write(d / "pilot.json", -2441.9, -3455.5, 179.5, 0.273, vmax_d=529.8, vmin_d=377.3, sig_d=21.0, k0_d=1.0)
    _write(d / "swarm.json", -2441.9, -3455.5, 179.5, 0.273, vmax_d=529.8, vmin_d=377.3, sig_d=21.0, k0_d=1.0)
    r = compare_envelopes(d / "swarm.json", d / "pilot.json")
    assert r["status"] == "pass" and r["freeze_allowed"]
    assert r["groups"]["Total"]["ok"] and r["groups"]["Dihedral"]["ok"]
    assert r["groups"]["Total"]["sigma_rel_diff"] == 0.0
    assert r["groups"]["Total"]["k0_ratio"] == 1.0


def test_missing_sigmaV_raises_keyerror_naming_file():
    from gareus.swarm.compare_pilot import compare_envelopes
    d = pathlib.Path(tempfile.mkdtemp())
    _write(d / "swarm.json", 100.0, -100.0, 20.0, 0.6)
    g = json.loads((d / "swarm.json").read_text())
    del g["all_globals"]["sigmaV_Total"]
    del g["joint_envelope"]["Total"]["sigmaV_kj_mol"]
    (d / "broken.json").write_text(json.dumps(g))
    _write(d / "pilot.json", 110.0, -95.0, 22.0, 0.55)
    try:
        compare_envelopes(d / "broken.json", d / "pilot.json")
        assert False, "expected KeyError"
    except KeyError as exc:
        assert "broken.json" in str(exc)


def test_main_writes_pilot_comparison_json(tmp_path=None):
    import subprocess, sys
    d = pathlib.Path(tempfile.mkdtemp())
    _write(d / "swarm.json", 100.0, -100.0, 20.0, 0.6)
    _write(d / "pilot.json", 110.0, -95.0, 22.0, 0.55)
    out = d / "pilot_comparison.json"
    from gareus.swarm.compare_pilot import main
    rc = main([str(d / "swarm.json"), str(d / "pilot.json"), "--out", str(out)])
    assert rc == 0
    assert out.exists()
    doc = json.loads(out.read_text())
    assert doc["status"] == "pass"
