"""Pep-GaMD FSF clamp (spec docs/superpowers/specs/2026-10-01-pep-gamd-fsf-clamp.md).

Per channel, with d = E - V, k = k0/(Vmax - Vmin), a = 1 - f, d_c = a/k:
    dV  = 1/2 k min(d, d_c)^2 + a max(0, d - d_c)
    FSF = max(f, 1 - k d)
so FSF = 1 + d(dV)/dV everywhere and never drops below the floor f.
"""
import json

import numpy as np
import pytest

from gareus.imports import import_openmm

from pep_gamd_fixture import solvated_dipeptide, _fresh_system  # noqa: E402


E, VMAX, VMIN = 100.0, 100.0, 0.0


def _legacy(v, k0):
    from gareus.pep_gamd import _channel_boost
    return _channel_boost(v, E, VMAX, VMIN, k0)


# --------------------------------------------------------------------------- closed form

def test_unset_floor_is_the_legacy_boost_exactly():
    from gareus.pep_gamd import _channel_boost
    v = np.linspace(-400.0, 150.0, 1001)
    for k0 in (0.0, 0.3, 1.0):
        assert np.array_equal(_channel_boost(v, E, VMAX, VMIN, k0, fsf_floor=None), _legacy(v, k0))


def test_clamped_boost_is_c1_and_its_slope_is_the_clamped_fsf():
    from gareus.pep_gamd import _channel_boost, channel_force_scaling_factor
    k0, f = 1.0, 0.5
    v = np.linspace(-300.0, 99.0, 4001)
    b = _channel_boost(v, E, VMAX, VMIN, k0, fsf_floor=f)
    fsf = channel_force_scaling_factor(v, E, VMAX, VMIN, k0, fsf_floor=f)
    # FSF = 1 + dB/dV (central differences on the interior)
    h = v[1] - v[0]
    slope = (b[2:] - b[:-2]) / (2 * h)
    # crossover at d_c = a/k = 0.5*100/1 = 50 -> V_c = 50; harmonic above, linear below.
    # dV is C1 but the FSF has a kink at V_c, so skip the stencil that straddles it.
    vc = E - (1 - f) * (VMAX - VMIN) / k0
    smooth = np.abs(v[1:-1] - vc) > 1.5 * h
    assert np.max(np.abs((1.0 + slope) - fsf[1:-1])[smooth]) < 1e-6
    assert fsf.min() >= f - 1e-12
    # C1 at the crossover: one-sided slopes agree to O(h)
    i = int(np.argmin(np.abs(v - vc)))
    assert abs((b[i + 1] - b[i]) / h - (b[i] - b[i - 1]) / h) < 2 * k0 / (VMAX - VMIN) * h + 1e-9
    above = v >= vc
    assert np.allclose(b[above], _legacy(v[above], k0))
    leg = _legacy(v, k0)
    far = (v < vc - h) & (leg > 0)                               # tangent line lies below the parabola
    assert far.any() and np.all(b[far] < leg[far])
    assert np.all(fsf[~above] == f)


def test_legacy_boost_switches_off_discontinuously_deep_below_vmin_but_the_clamp_does_not():
    """Upstream zeroes the boost once V + dV >= E. Far below Vmin the harmonic dV is so large that
    this trips: dV jumps from ~(E-V)^2 k/2 to 0. The clamped boost keeps V + dV = f V + const < E."""
    from gareus.pep_gamd import _channel_boost
    v = np.linspace(-400.0, 99.0, 5000)
    leg = _legacy(v, 1.0)
    assert np.max(np.abs(np.diff(leg))) > 100.0                     # the legacy discontinuity exists
    b = _channel_boost(v, E, VMAX, VMIN, 1.0, fsf_floor=0.2)
    assert np.all(b > 0) and np.max(np.abs(np.diff(b))) < 1.0       # clamped: continuous, never switches off
    assert np.all(v + b < E)


def test_clamp_only_binds_where_the_legacy_fsf_is_below_the_floor():
    from gareus.pep_gamd import _channel_boost
    k0, f = 0.36, 0.5               # 1 - k d >= 0.64 on [Vmin, Vmax]: floor never reached inside the envelope
    v = np.linspace(VMIN, VMAX, 501)
    assert np.array_equal(_channel_boost(v, E, VMAX, VMIN, k0, fsf_floor=f), _legacy(v, k0))


def test_zero_k0_rung_gets_no_boost_with_a_floor():
    from gareus.pep_gamd import _channel_boost, channel_force_scaling_factor
    v = np.linspace(-1e4, 1e2, 11)
    assert np.all(_channel_boost(v, E, VMAX, VMIN, 0.0, fsf_floor=0.5) == 0.0)
    assert np.all(channel_force_scaling_factor(v, E, VMAX, VMIN, 0.0, fsf_floor=0.5) == 1.0)


def test_above_threshold_is_unboosted_with_fsf_one():
    from gareus.pep_gamd import _channel_boost, channel_force_scaling_factor
    v = np.array([100.0, 120.0, 1e5])
    assert np.all(_channel_boost(v, E, VMAX, VMIN, 1.0, fsf_floor=0.3) == 0.0)
    assert np.all(channel_force_scaling_factor(v, E, VMAX, VMIN, 1.0, fsf_floor=0.3) == 1.0)


def test_legacy_fsf_goes_negative_below_vmin_and_the_clamp_prevents_it():
    from gareus.pep_gamd import channel_force_scaling_factor
    v = np.array([-50.0])
    assert channel_force_scaling_factor(v, E, VMAX, VMIN, 1.0, fsf_floor=None)[0] < 0.0
    assert channel_force_scaling_factor(v, E, VMAX, VMIN, 1.0, fsf_floor=0.0)[0] == 0.0


# --------------------------------------------------------------------------- envelope

def _globals(**over):
    g = {"Vmax_Total": 100.0, "Vmin_Total": 0.0, "threshold_energy_Total": 100.0, "k0_Total": 1.0,
         "Vmax_Dihedral": 50.0, "Vmin_Dihedral": 0.0, "threshold_energy_Dihedral": 50.0, "k0_Dihedral": 1.0}
    g.update(over)
    return g


def test_envelope_reads_floors_from_globals_and_defaults_to_legacy():
    from gareus.pep_gamd import PepGamdEnvelope
    legacy = PepGamdEnvelope.from_integrator_globals(_globals())
    assert legacy.fsf_floor_total is None and legacy.fsf_floor_dih is None
    env = PepGamdEnvelope.from_integrator_globals(_globals(fsf_floor_Total=0.5, fsf_floor_Dihedral=0.1))
    assert env.fsf_floor_total == 0.5 and env.fsf_floor_dih == 0.1


def test_dual_boost_uses_each_channels_floor_in_dependent_order():
    from gareus.pep_gamd import PepGamdEnvelope, pep_gamd_boost_kj, _channel_boost
    env = PepGamdEnvelope.from_integrator_globals(_globals(fsf_floor_Total=0.5, fsf_floor_Dihedral=0.2))
    v_pep = np.array([-80.0, 10.0, 60.0]); v_dih = np.array([-30.0, 5.0, 40.0])
    for lam in (0.0, 0.4, 1.0):
        b_d = _channel_boost(v_dih, 50.0, 50.0, 0.0, lam * 1.0, fsf_floor=0.2)
        b_t = _channel_boost(v_pep + b_d, 100.0, 100.0, 0.0, lam * 1.0, fsf_floor=0.5)
        assert np.allclose(pep_gamd_boost_kj(v_pep, v_dih, lam, env), b_d + b_t)


def test_envelope_writer_records_floors(tmp_path):
    from gareus.swarm.envelope import write_envelope_setup_dir
    from gareus.gamd_calibration import PooledEnvelope
    from gareus.pep_gamd import PepGamdEnvelope
    env = {g: PooledEnvelope(group=g, vmax=v + 50, vmin=v - 50, vavg=v, sigmav=10.0, n_windows=3, n_total=300)
           for g, v in (("Total", -3000.0), ("Dihedral", 400.0))}
    kw = dict(sigma0_kj={"Total": 25.104, "Dihedral": 25.104}, temperature_k=300.0, meta={})
    legacy = json.loads(write_envelope_setup_dir(tmp_path / "a", env, **kw).read_text())
    assert not any(k.startswith("fsf_floor") for k in legacy["all_globals"])
    path = write_envelope_setup_dir(tmp_path / "b", env, fsf_floor={"Total": 0.5, "Dihedral": 0.0}, **kw)
    doc = json.loads(path.read_text())
    assert doc["all_globals"]["fsf_floor_Total"] == 0.5 and doc["all_globals"]["fsf_floor_Dihedral"] == 0.0
    e = PepGamdEnvelope.from_json(path)
    assert e.fsf_floor_total == 0.5 and e.fsf_floor_dih == 0.0


def test_ladder_report_floor_is_the_clamped_floor():
    from gareus.pep_gamd import PepGamdEnvelope
    from gareus.swarm.ladder_design import fsf_floor_per_rung
    env = PepGamdEnvelope.from_integrator_globals(_globals(fsf_floor_Total=0.5, fsf_floor_Dihedral=0.0))
    rep = fsf_floor_per_rung([0.0, 0.5, 1.0], env)
    assert rep["Total"] == [1.0, 0.5, 0.5] and rep["Dihedral"] == [1.0, 0.5, 0.0]
    assert rep["top_rung_floor_total"] == 0.5


# --------------------------------------------------------------------------- reconcile guard

class _FakeInteg:
    def __init__(self, names):
        self._n = dict(names)

    def getNumGlobalVariables(self):
        return len(self._n)

    def getGlobalVariableName(self, i):
        return list(self._n)[i]

    def getGlobalVariableByName(self, name):
        return self._n[name]


def test_reconcile_refuses_envelope_floor_on_unclamped_integrator():
    from gareus.pep_gamd import reconcile_fsf_floors
    with pytest.raises(ValueError, match="fsf_floor_Total"):
        reconcile_fsf_floors(_FakeInteg({"k0_Total": 0.1}), _globals(fsf_floor_Total=0.5, fsf_floor_Dihedral=0.0))


def test_reconcile_refuses_clamped_integrator_with_unclamped_envelope():
    from gareus.pep_gamd import reconcile_fsf_floors
    integ = _FakeInteg({"k0_Total": 0.1, "fsf_floor_Total": 0.5, "fsf_floor_Dihedral": 0.0})
    with pytest.raises(ValueError, match="no FSF floor"):
        reconcile_fsf_floors(integ, _globals())


def test_reconcile_refuses_different_floors_and_accepts_equal_or_partial_dicts():
    from gareus.pep_gamd import reconcile_fsf_floors
    integ = _FakeInteg({"k0_Total": 0.1, "fsf_floor_Total": 0.5, "fsf_floor_Dihedral": 0.0})
    with pytest.raises(ValueError, match="differs"):
        reconcile_fsf_floors(integ, _globals(fsf_floor_Total=0.4, fsf_floor_Dihedral=0.0))
    reconcile_fsf_floors(integ, _globals(fsf_floor_Total=0.5, fsf_floor_Dihedral=0.0))
    reconcile_fsf_floors(integ, {"stepCount": 10.0})               # not an envelope: nothing to check
    reconcile_fsf_floors(_FakeInteg({"k0_Total": 0.1}), _globals())  # legacy on both sides


# --------------------------------------------------------------------------- CLI

def test_cli_floors_parse_validate_and_require_pep_gamd():
    from gareus.cli import parse_args
    base = ["--seq", "GYDPETGTWG", "--gamd-boost-type", "pep-gamd-lower-dual"]
    a = parse_args(base + ["--pep-gamd-fsf-floor-total", "0.5"])
    assert a.pep_gamd_fsf_floor_total == 0.5 and a.pep_gamd_fsf_floor_dihedral == 0.0
    a = parse_args(base)
    assert a.pep_gamd_fsf_floor_total is None and a.pep_gamd_fsf_floor_dihedral is None
    for bad in ("1.0", "-0.1"):
        with pytest.raises(SystemExit):
            parse_args(base + ["--pep-gamd-fsf-floor-total", bad])
    with pytest.raises(SystemExit):
        parse_args(["--seq", "GYDPETGTWG", "--gamd-boost-type", "lower-dihedral", "--pep-gamd-fsf-floor-total", "0.5"])


# --------------------------------------------------------------------------- integrator

def _kwargs(unit):
    return dict(dt=0.002 * unit.picoseconds, ntcmdprep=2, ntcmd=4, ntebprep=2, nteb=4, nstlim=100, ntave=2,
                sigma0p=6.0 * unit.kilocalories_per_mole, sigma0d=6.0 * unit.kilocalories_per_mole,
                collision_rate=1.0 / unit.picoseconds, temperature=300.0 * unit.kelvin)


def _system():
    from gareus import pep_gamd
    fx = solvated_dipeptide()
    system = _fresh_system()
    pep_gamd.ensure_pep_gamd_partition(system, fx["peptide"])
    return system, fx


def _integ(system, unit, **floors):
    from gareus import pep_gamd
    return pep_gamd.PepGaMDLowerDualIntegrator(
        pep_gamd.DIHEDRAL_GROUP, bias_force_groups=pep_gamd.pep_gamd_bias_force_groups(system),
        **floors, **_kwargs(unit))


def _program(integ):
    return [integ.getComputationStep(i) for i in range(integ.getNumComputations())]


def test_unclamped_integrator_program_is_unchanged():
    openmm, _app, unit = import_openmm()
    system, _fx = _system()
    a, b = _integ(system, unit), _integ(system, unit, fsf_floor_total=None, fsf_floor_dihedral=None)
    assert _program(a) == _program(b)
    names = {a.getGlobalVariableName(i) for i in range(a.getNumGlobalVariables())}
    assert not any(n.startswith("fsf_floor") for n in names)


def _pre_step_energies(ctx, unit):
    from gareus import pep_gamd
    e = {g: ctx.getState(getEnergy=True, groups={g}).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
         for g in (0, 1, 2)}
    return e[0] - e[1] + e[2], e[2]


def _clamped_stage5(v_pep, v_dih, *, above):
    """Stage-5 globals that put the current configuration deep in the clamped region: E far above V
    so k d > 1 - f on both channels (Vmax - Vmin chosen so the legacy FSF would be strongly negative)."""
    return {"stepCount": 50, "stage": 5, "k0_Total": 1.0, "k0_Dihedral": 1.0,
            "Vmax_Total": v_pep + above, "Vmin_Total": v_pep + above - 40.0, "threshold_energy_Total": v_pep + above,
            "Vmax_Dihedral": v_dih + above, "Vmin_Dihedral": v_dih + above - 40.0,
            "threshold_energy_Dihedral": v_dih + above}


def test_integrator_boost_and_fsf_match_the_closed_form_in_the_clamped_region():
    from gareus.pep_gamd import PepGamdEnvelope, pep_gamd_boost_kj, channel_force_scaling_factor, _channel_boost
    openmm, _app, unit = import_openmm()
    system, fx = _system()
    integ = _integ(system, unit, fsf_floor_total=0.5, fsf_floor_dihedral=0.25)
    ctx = openmm.Context(system, integ, openmm.Platform.getPlatformByName("Reference"))
    ctx.setPositions(fx["positions"]); ctx.setVelocitiesToTemperature(300 * unit.kelvin, 3)
    integ.step(1)
    ctx.setPositions(fx["positions"]); ctx.setVelocitiesToTemperature(300 * unit.kelvin, 3)
    v_pep, v_dih = _pre_step_energies(ctx, unit)
    g = _clamped_stage5(v_pep, v_dih, above=100.0)
    for k, v in g.items():
        integ.setGlobalVariableByName(k, v)
    integ.step(1)
    b_d = _channel_boost(v_dih, g["threshold_energy_Dihedral"], g["Vmax_Dihedral"], g["Vmin_Dihedral"], 1.0, fsf_floor=0.25)
    fsf_d = channel_force_scaling_factor(v_dih, g["threshold_energy_Dihedral"], g["Vmax_Dihedral"], g["Vmin_Dihedral"], 1.0, fsf_floor=0.25)
    b_t = _channel_boost(v_pep + b_d, g["threshold_energy_Total"], g["Vmax_Total"], g["Vmin_Total"], 1.0, fsf_floor=0.5)
    fsf_t = channel_force_scaling_factor(v_pep + b_d, g["threshold_energy_Total"], g["Vmax_Total"], g["Vmin_Total"], 1.0, fsf_floor=0.5)
    assert fsf_d == 0.25 and fsf_t == 0.5, "configuration must sit in the clamped region on both channels"
    got = {n: integ.getGlobalVariableByName(n) for n in
           ("BoostPotential_Dihedral", "BoostPotential_Total", "ForceScalingFactor_Dihedral", "ForceScalingFactor_Total")}
    assert abs(got["BoostPotential_Dihedral"] - b_d) < 1e-6 * max(1.0, b_d)
    assert abs(got["BoostPotential_Total"] - b_t) < 1e-6 * max(1.0, b_t)
    assert abs(got["ForceScalingFactor_Dihedral"] - 0.25) < 1e-12
    assert abs(got["ForceScalingFactor_Total"] - 0.5) < 1e-12
    env = PepGamdEnvelope(g["Vmax_Total"], g["Vmin_Total"], g["threshold_energy_Total"], 1.0,
                          g["Vmax_Dihedral"], g["Vmin_Dihedral"], g["threshold_energy_Dihedral"], 1.0,
                          fsf_floor_total=0.5, fsf_floor_dih=0.25)
    assert abs(pep_gamd_boost_kj(v_pep, v_dih, 1.0, env) - (got["BoostPotential_Dihedral"] + got["BoostPotential_Total"])) < 1e-5


def test_integrator_force_is_the_gradient_of_the_clamped_boosted_energy():
    """Finite-difference oracle on the real partitioned system: move one peptide atom by +-h and compare
    the energy change of U_phys + dV_clamped against the force the integrator would apply,
    (f0 - f1)*FSF_T + f_dih*FSF_T*FSF_D + f1 (no bias forces here)."""
    from gareus.pep_gamd import pep_gamd_boost_kj, PepGamdEnvelope
    openmm, _app, unit = import_openmm()
    system, fx = _system()
    ref = openmm.Context(system, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName("Reference"))
    ref.setPositions(fx["positions"])
    v_pep, v_dih = _pre_step_energies(ref, unit)
    g = _clamped_stage5(v_pep, v_dih, above=100.0)
    env = PepGamdEnvelope(g["Vmax_Total"], g["Vmin_Total"], g["threshold_energy_Total"], 1.0,
                          g["Vmax_Dihedral"], g["Vmin_Dihedral"], g["threshold_energy_Dihedral"], 1.0,
                          fsf_floor_total=0.5, fsf_floor_dih=0.25)

    def u_eff(pos):
        ref.setPositions(pos)
        vp, vd = _pre_step_energies(ref, unit)
        phys = ref.getState(getEnergy=True, groups={0, 2}).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        return phys + pep_gamd_boost_kj(vp, vd, 1.0, env)

    from gareus.pep_gamd import channel_force_scaling_factor, _channel_boost
    b_d = _channel_boost(v_dih, g["threshold_energy_Dihedral"], g["Vmax_Dihedral"], g["Vmin_Dihedral"], 1.0, fsf_floor=0.25)
    fsf_d = channel_force_scaling_factor(v_dih, g["threshold_energy_Dihedral"], g["Vmax_Dihedral"], g["Vmin_Dihedral"], 1.0, fsf_floor=0.25)
    fsf_t = channel_force_scaling_factor(v_pep + b_d, g["threshold_energy_Total"], g["Vmax_Total"], g["Vmin_Total"], 1.0, fsf_floor=0.5)
    ref.setPositions(fx["positions"])
    f = {gid: ref.getState(getForces=True, groups={gid}).getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
         for gid in (0, 1, 2)}
    applied = (f[0] - f[1]) * fsf_t + f[2] * fsf_t * fsf_d + f[1]
    x0 = np.array(fx["positions"].value_in_unit(unit.nanometer), dtype=float)
    atom = int(fx["peptide"][len(fx["peptide"]) // 2])
    h = 2e-5
    for axis in range(3):
        xp, xm = x0.copy(), x0.copy()
        xp[atom, axis] += h; xm[atom, axis] -= h
        fd = -(u_eff(xp) - u_eff(xm)) / (2 * h)
        assert abs(fd - applied[atom, axis]) < 2e-3 * max(1.0, abs(fd)), (axis, fd, applied[atom, axis])


# --------------------------------------------------------------------------- NPT adapter

def test_npt_channel_boost_matches_closed_form_with_a_floor():
    from gareus.pep_gamd import _npt_lower_bound_channel_boost, _channel_boost
    for v in (-300.0, -10.0, 30.0, 80.0, 120.0):
        for floor in (None, 0.0, 0.5):
            a = _npt_lower_bound_channel_boost(v, VMAX, VMIN, E, 1.0, fsf_floor=floor)
            b = float(_channel_boost(np.array([v]), E, VMAX, VMIN, 1.0, fsf_floor=floor)[0])
            assert abs(a - b) < 1e-9, (v, floor, a, b)


# --------------------------------------------------------------------------- review fixes (2026-10-01)

def test_uninitialised_envelope_gives_zero_boost_not_nan_in_closed_form():
    from gareus.pep_gamd import _channel_boost, _npt_lower_bound_channel_boost
    b = _channel_boost(np.array([-3000.0, 0.0]), -1e99, -1e99, 1e99, 0.0, fsf_floor=0.6)
    assert np.all(np.isfinite(b)) and np.all(b == 0.0)
    assert _npt_lower_bound_channel_boost(-3000.0, -1e99, 1e99, -1e99, 0.0, fsf_floor=0.6) == 0.0


def test_clamped_integrator_with_default_globals_stays_finite():
    """Adversarial finding: an integrator stepped in a boost stage with its default globals
    (Vmax = -1e99, Vmin = 1e99, k0 = 0) used to NaN through d_c = -inf. Legacy stays finite there."""
    openmm, _app, unit = import_openmm()
    system, fx = _system()
    integ = _integ(system, unit, fsf_floor_total=0.6, fsf_floor_dihedral=0.0)
    ctx = openmm.Context(system, integ, openmm.Platform.getPlatformByName("Reference"))
    ctx.setPositions(fx["positions"]); ctx.setVelocitiesToTemperature(300 * unit.kelvin, 5)
    integ.step(1)
    for k, v in {"stepCount": 50, "stage": 5, "k0_Total": 0.0, "k0_Dihedral": 0.0,
                 "Vmax_Total": -1e99, "Vmin_Total": 1e99, "Vmax_Dihedral": -1e99, "Vmin_Dihedral": 1e99}.items():
        integ.setGlobalVariableByName(k, v)
    integ.step(3)
    for n in ("BoostPotential_Total", "BoostPotential_Dihedral", "ForceScalingFactor_Total",
              "ForceScalingFactor_Dihedral", "StartingPotentialEnergy_Total"):
        assert np.isfinite(integ.getGlobalVariableByName(n)), n
    pos = ctx.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    assert np.all(np.isfinite(pos))


def test_sidecar_run_adopts_the_envelopes_floors(tmp_path):
    from types import SimpleNamespace
    from gareus.swarm.epoch0 import _apply_sidecar_gamd_envelope
    d = tmp_path / "shared_gamd_setup"; d.mkdir()
    (d / "shared_gamd_setup_globals.json").write_text(json.dumps({"all_globals": _globals(fsf_floor_Total=0.6, fsf_floor_Dihedral=0.0)}))
    side = {"gamd": {"shared_gamd_setup_dir": str(d)}}
    args = SimpleNamespace(shared_gamd_setup_dir="", pep_gamd_fsf_floor_total=None, pep_gamd_fsf_floor_dihedral=None)
    rec = _apply_sidecar_gamd_envelope(args, side, tmp_path / "ladder_run_args.yaml")
    assert args.pep_gamd_fsf_floor_total == 0.6 and args.pep_gamd_fsf_floor_dihedral == 0.0
    assert rec["pep_gamd_fsf_floor_total"] == 0.6
    explicit = SimpleNamespace(shared_gamd_setup_dir="", pep_gamd_fsf_floor_total=0.4, pep_gamd_fsf_floor_dihedral=0.0)
    _apply_sidecar_gamd_envelope(explicit, side, tmp_path / "ladder_run_args.yaml")
    assert explicit.pep_gamd_fsf_floor_total == 0.4          # explicit args kept; reconcile refuses later


def test_interesting_globals_keep_the_floors():
    from gareus.production import integrator_globals
    openmm, _app, unit = import_openmm()
    system, _fx = _system()
    g = integrator_globals(_integ(system, unit, fsf_floor_total=0.6, fsf_floor_dihedral=0.0))
    assert g.get("fsf_floor_Total") == 0.6 and g.get("fsf_floor_Dihedral") == 0.0
