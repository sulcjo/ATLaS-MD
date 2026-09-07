import json, math, pathlib, tempfile
import numpy as np


class _FakeGamdIntegrator:
    """Minimal CustomIntegrator stand-in: a fresh-context worth of globals (physics keys
    at their zero/default value, plus the bookkeeping/derived globals a real gamd-openmm
    CustomIntegrator carries) to verify what set_integrator_globals_from_dict does with a
    physics-only payload. The bookkeeping names below (stage, stepCount, windowCount,
    ForceScalingFactor_Total/_Dihedral, BoostPotential_Total) are copied verbatim from the
    real all_globals dump at RUNS/aurum_pilots/ll_pilot_s3/out/shared_gamd_setup_globals.json
    (62 keys total there; only 16 of them -- the 8 physics keys per group -- are what
    write_envelope_setup_dir emits)."""

    def __init__(self):
        self._names = [
            "Vmax_Total", "Vmin_Total", "Vavg_Total", "sigmaV_Total", "k0_Total", "k_Total",
            "threshold_energy_Total", "sigma0_Total",
            "Vmax_Dihedral", "Vmin_Dihedral", "Vavg_Dihedral", "sigmaV_Dihedral", "k0_Dihedral",
            "k_Dihedral", "threshold_energy_Dihedral", "sigma0_Dihedral",
            "stage", "stepCount", "windowCount",
            "ForceScalingFactor_Total", "ForceScalingFactor_Dihedral", "BoostPotential_Total",
        ]
        self._values = {n: 0.0 for n in self._names}
        self._values["stage"] = 3.0
        self._values["stepCount"] = 42.0
        self._values["windowCount"] = 7.0
        self._values["ForceScalingFactor_Total"] = 1.0
        self._values["ForceScalingFactor_Dihedral"] = 1.0

    def getNumGlobalVariables(self):
        return len(self._names)

    def getGlobalVariableName(self, i):
        return self._names[i]

    def getGlobalVariable(self, i):
        return self._values[self._names[i]]

    def setGlobalVariable(self, i, value):
        self._values[self._names[i]] = float(value)

    def setGlobalVariableByName(self, name, value):
        self._values[name] = float(value)


def _trace(n=400, drift_frames=60, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.normal(0.0, 5.0, n)
    v[:drift_frames] += np.linspace(60.0, 0.0, drift_frames)   # relaxing transient
    return v


def test_discard_detects_transient_and_zero_for_stationary():
    from gareus.swarm.envelope import discard_frames_from_trace
    d = discard_frames_from_trace(_trace(), block=20)
    assert 40 <= d <= 120
    assert discard_frames_from_trace(np.random.default_rng(1).normal(0, 5, 400), block=20) <= 40


def test_pooled_discard_is_upper_quantile_with_floor():
    from gareus.swarm.envelope import pooled_discard
    assert pooled_discard([10, 20, 30, 200], quantile=0.95) >= 30
    assert pooled_discard([1, 2], floor_frames=50) == 50


def test_pool_member_envelopes_drops_discard_and_reports_both_groups():
    from gareus.swarm.envelope import pool_member_envelopes
    traces = {0: {"v_pep_kj": _trace(seed=0), "v_dih_kj": _trace(seed=1) + 100.0},
              1: {"v_pep_kj": _trace(seed=2), "v_dih_kj": _trace(seed=3) + 100.0}}
    env = pool_member_envelopes(traces, discard=100)
    assert set(env) == {"Total", "Dihedral"}
    assert env["Total"].n_windows == 2 and env["Total"].n_total == 2 * 300
    assert env["Total"].vmax < 40.0          # transient (up to +60) was discarded
    assert 90.0 < env["Dihedral"].vavg < 110.0


def test_setup_dir_json_roundtrips_through_pep_gamd_envelope_and_reuse_loader():
    from gareus.swarm.envelope import pool_member_envelopes, write_envelope_setup_dir
    from gareus.pep_gamd import PepGamdEnvelope
    traces = {0: {"v_pep_kj": _trace(seed=0), "v_dih_kj": _trace(seed=1) + 100.0}}
    env = pool_member_envelopes(traces, discard=100)
    d = pathlib.Path(tempfile.mkdtemp()) / "shared_gamd_setup"
    p = write_envelope_setup_dir(d, env, sigma0_kj={"Total": 6.0 * 4.184, "Dihedral": 6.0 * 4.184}, temperature_k=300.0, meta={"n_members": 1})
    doc = json.loads(p.read_text())
    g = doc["all_globals"]
    for grp in ("Total", "Dihedral"):
        for k in ("Vmax", "Vmin", "Vavg", "sigmaV", "k0", "k", "threshold_energy", "sigma0"):
            assert f"{k}_{grp}" in g
    assert math.isclose(g["threshold_energy_Total"], g["Vmax_Total"])          # lower bound: E = Vmax
    assert 0.0 < g["k0_Total"] <= 1.0
    e = PepGamdEnvelope.from_json(p)
    assert math.isclose(e.k0max_total, g["k0_Total"]) and math.isclose(e.vmin_dih, g["Vmin_Dihedral"])
    assert doc["mode"] == "swarm_unbiased_envelope" and doc["gamd_boost_type"] == "pep-gamd-lower-dual"


def test_production_consumer_applies_physics_only_payload_without_touching_bookkeeping():
    """Step 4 verification (task-2-brief.md): production's set_integrator_globals_from_dict
    (production.py:3555) only writes global names present in the payload it is given. A
    shared_gamd_setup_globals.json written by write_envelope_setup_dir carries ONLY the
    physics keys (Vmax_*/Vmin_*/Vavg_*/sigmaV_*/k0_*/k_*/threshold_energy_*/sigma0_* for
    Total and Dihedral) -- applying that dict to a fresh integrator must overwrite exactly
    those 16 names and must leave stage/stepCount/windowCount/ForceScalingFactor at
    whatever the integrator itself initialized them to. No guard/merge helper is needed in
    production.py for this to hold: the loop iterates over the *source* dict's keys, so a
    name absent from the source (every bookkeeping global) is never visited."""
    from gareus.production import set_integrator_globals_from_dict
    from gareus.swarm.envelope import pool_member_envelopes, write_envelope_setup_dir

    traces = {0: {"v_pep_kj": _trace(seed=0), "v_dih_kj": _trace(seed=1) + 100.0}}
    env = pool_member_envelopes(traces, discard=100)
    d = pathlib.Path(tempfile.mkdtemp()) / "shared_gamd_setup"
    p = write_envelope_setup_dir(d, env, sigma0_kj={"Total": 6.0 * 4.184, "Dihedral": 6.0 * 4.184},
                                 temperature_k=300.0, meta={"n_members": 1})
    payload = json.loads(p.read_text())["all_globals"]

    integrator = _FakeGamdIntegrator()
    copied, skipped = set_integrator_globals_from_dict(integrator, payload)

    assert not skipped
    assert set(copied) == set(payload)
    for name, value in payload.items():
        assert math.isclose(integrator._values[name], float(value))
    # bookkeeping globals: untouched by a physics-only payload
    assert integrator._values["stage"] == 3.0
    assert integrator._values["stepCount"] == 42.0
    assert integrator._values["windowCount"] == 7.0
    assert integrator._values["ForceScalingFactor_Total"] == 1.0
    assert integrator._values["ForceScalingFactor_Dihedral"] == 1.0
    assert integrator._values["BoostPotential_Total"] == 0.0
