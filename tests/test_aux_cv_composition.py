"""The auxiliary force must enter the integrator once, at full strength, outside the boost (spec 3.3).

Design note (verifier finding B4): adding or removing a force GROUP changes the Pep-GaMD integrator
program and therefore its random stream. One step at 300 K then shifts every atom by ~1e-3 nm,
which swamps the ~1e-6 nm the restraint itself causes. Every arm below has the SAME group
structure: the aux force sits in its group in every system (k = 0 where inactive), and every
constant external force has a zero-parameter twin in the same group. Differences between paired
arms are therefore exactly the effect of the force under test.
"""
import types

import numpy as np

from aux_cv_fixture import dipeptide, model_payload
from gareus.auxiliary_cv.evaluate import aux_energy_kj, aux_forces_kj_nm, z_from_positions
from gareus.auxiliary_cv.model import AuxModel
from gareus.auxiliary_cv.runtime import add_aux_cv_force
from gareus.auxiliary_cv.state_table import AuxStateTable
from gareus.imports import import_openmm

ARGS = types.SimpleNamespace(umbrella_force_group=31, secondary_cv_force_group=29)
K_KCAL, OFFSET_TO_CENTER = 40.0, 0.5


def _table():
    d = dipeptide()
    blocks = [lab.split("-")[0] for lab in d["labels"]]
    m = AuxModel.from_mapping(model_payload(d["quads"], [1.0] + [0.5] * (2 * len(d["quads"]) - 1),
                                            offset=0.0, blocks=blocks))
    z0 = float(z_from_positions(d["positions_nm"], m)[0])
    return AuxStateTable(m, (z0 - OFFSET_TO_CENTER,), (K_KCAL,), (None,)), d


def _with_aux(system, table, active):
    openmm, _app, _unit = import_openmm()
    rt = add_aux_cv_force(openmm, system, table, ARGS)
    force = system.getForce(rt.force_index)
    force.setGlobalParameterDefaultValue(0, K_KCAL * 4.184 if active else 0.0)   # aux_k (kJ/mol per z^2)
    force.setGlobalParameterDefaultValue(1, table.centers[0] if active else 0.0)  # aux_c
    return rt


def _constant_force(system, forces_kj_nm, positions_nm, group, scale):
    """F = scale * forces_kj_nm, energy -(F . (x - x0)): zero energy at the start positions, so the
    Total-channel energy (and therefore the FSF) is identical between a pair of arms."""
    openmm, _app, _unit = import_openmm()
    f = openmm.CustomExternalForce("-(fx*(x-x0)+fy*(y-y0)+fz*(z-z0))")
    for name in ("fx", "fy", "fz", "x0", "y0", "z0"):
        f.addPerParticleParameter(name)
    for i, vec in enumerate(forces_kj_nm):
        if np.any(vec):
            f.addParticle(i, [float(scale * v) for v in vec] + [float(c) for c in positions_nm[i]])
    f.setForceGroup(group)
    system.addForce(f)


def _one_step_positions_and_fsf(system, integ, positions, openmm, unit, seed=7, stage_globals=None):
    """test_pep_gamd_boost._one_step_positions (same warm-up rule: step once, re-seat the start state,
    set the stage globals, step once and measure), but it also reads ForceScalingFactor_Total while the
    Context is alive: after the Context is freed the integrator's globals are unreadable (std::bad_cast)."""
    integ.setRandomNumberSeed(seed)
    ctx = openmm.Context(system, integ, openmm.Platform.getPlatformByName("Reference"))
    ctx.setPositions(positions)
    ctx.setVelocitiesToTemperature(300.0 * unit.kelvin, seed)
    integ.step(1)
    ctx.setPositions(positions)
    ctx.setVelocitiesToTemperature(300.0 * unit.kelvin, seed)
    if stage_globals:
        for k, v in stage_globals.items():
            integ.setGlobalVariableByName(k, v)
    integ.step(1)
    pos = ctx.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    fsf = float(integ.getGlobalVariableByName("ForceScalingFactor_Total"))
    del ctx
    return np.array(pos, dtype=float), fsf


def dipeptide_peptide():
    from pep_gamd_fixture import solvated_dipeptide
    return solvated_dipeptide()["peptide"]


def test_group_energies_reconstruct_the_total_and_aux_is_a_bias_group():
    from pep_gamd_fixture import _fresh_system
    from gareus import pep_gamd
    openmm, _app, unit = import_openmm()
    table, d = _table()
    system = _fresh_system()
    pep_gamd.ensure_pep_gamd_partition(system, dipeptide_peptide())
    rt = _with_aux(system, table, active=True)
    assert rt.info.force_group in pep_gamd.pep_gamd_bias_force_groups(system)
    ctx = openmm.Context(system, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName("Reference"))
    ctx.setPositions(d["positions_nm"])
    groups = sorted({system.getForce(i).getForceGroup() for i in range(system.getNumForces())})
    e = {g: ctx.getState(getEnergy=True, groups={g}).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
         for g in groups}
    total = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    assert abs(sum(e.values()) - total) < 1e-6 * max(1.0, abs(total))
    z = z_from_positions(d["positions_nm"], table.model)
    assert abs(e[rt.info.force_group] - aux_energy_kj(z, table.centers[0], K_KCAL)[0]) < 1e-8


def test_pep_gamd_applies_the_aux_gradient_once_unscaled():
    """Paired, structure-matched arms (see module docstring), one Pep-GaMD step with an ACTIVE boost:

      aux:  (aux active)                  - (aux k = 0)                     == effect of A_s
      ext:  (const F_aux in aux group)    - (zero const in aux group)       == effect of an unboosted bias force
      bst:  (const F_aux in group 0)      - (zero const in group 0)         == effect of a Total-boosted force

    Required: aux == ext (applied once, unscaled), bst == FSF_Total * ext (the comparison can see the
    boost), and all effects non-negligible. Warm-up rule (CLAUDE.md): _one_step_positions_and_fsf steps once,
    re-seats the start state, then measures.
    """
    from pep_gamd_fixture import _fresh_system, solvated_dipeptide
    from gareus import pep_gamd
    from test_pep_gamd_boost import _gamd_kwargs, _assert_moved
    openmm, _app, unit = import_openmm()
    table, d = _table()
    fx = solvated_dipeptide()
    x0 = d["positions_nm"]
    f_aux = aux_forces_kj_nm(x0, table.model, table.centers[0], K_KCAL)
    assert np.abs(f_aux).max() > 10.0, "the auxiliary force must be large enough to move atoms measurably"

    def _system(kind):
        s = _fresh_system()
        pep_gamd.ensure_pep_gamd_partition(s, fx["peptide"])
        rt = _with_aux(s, table, active=(kind == "aux_on"))
        if kind in ("ext_on", "ext_off"):
            _constant_force(s, f_aux, x0, rt.info.force_group, 1.0 if kind == "ext_on" else 0.0)
        elif kind in ("bst_on", "bst_off"):
            _constant_force(s, f_aux, x0, 0, 1.0 if kind == "bst_on" else 0.0)
        return s

    probe_sys = _system("aux_off")
    probe = openmm.Context(probe_sys, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName("Reference"))
    probe.setPositions(fx["positions"])
    v_pep = pep_gamd.peptide_essential_energy_kj(probe, unit)
    v_dih = probe.getState(getEnergy=True, groups={pep_gamd.DIHEDRAL_GROUP}).getPotentialEnergy().value_in_unit(
        unit.kilojoule_per_mole)
    del probe
    stage = {"stepCount": 50, "stage": 5,
             "k0_Total": 0.5, "Vmax_Total": v_pep + 50.0, "Vmin_Total": v_pep - 50.0, "threshold_energy_Total": v_pep + 50.0,
             "k0_Dihedral": 0.5, "Vmax_Dihedral": v_dih + 50.0, "Vmin_Dihedral": v_dih - 50.0,
             "threshold_energy_Dihedral": v_dih + 50.0}
    kw = _gamd_kwargs(unit)
    out, fsf_total = {}, {}
    for kind in ("aux_on", "aux_off", "ext_on", "ext_off", "bst_on", "bst_off"):
        s = _system(kind)
        integ = pep_gamd.PepGaMDLowerDualIntegrator(pep_gamd.DIHEDRAL_GROUP,
                                                    bias_force_groups=pep_gamd.pep_gamd_bias_force_groups(s), **kw)
        out[kind], fsf_total[kind] = _one_step_positions_and_fsf(s, integ, fx["positions"], openmm, unit,
                                                                 stage_globals=stage)
    _assert_moved(out["aux_on"], fx["positions"], unit)
    d_aux = out["aux_on"] - out["aux_off"]
    d_ext = out["ext_on"] - out["ext_off"]
    d_bst = out["bst_on"] - out["bst_off"]
    scale = np.abs(d_ext).max()
    assert scale > 1e-8, f"the bias-group force moved nothing measurable ({scale:.3g} nm): test is vacuous"
    assert np.abs(d_aux - d_ext).max() < 1e-6 * scale, "aux gradient is not applied once at full strength"
    fsf = fsf_total["bst_on"]
    assert fsf_total["bst_off"] == fsf, "pair arms must see the same Total-channel FSF"
    assert abs(1.0 - fsf) > 0.05, f"FSF_Total {fsf:.3f} is ~1: the boosted copy could not be told apart"
    assert np.abs(d_bst - fsf * d_ext).max() < 1e-3 * scale, "boosted copy is not FSF_Total-scaled"


def test_npt_trial_energy_includes_aux_and_tracks_geometry():
    """Pep-GaMD NPT adapter: the aux group is a bias group evaluated on the live coordinates.

    A molecular-centroid volume trial translates whole molecules, so intramolecular torsions -- and
    therefore A_s -- are INVARIANT under it: an adapter that read old coordinates would still pass a
    before/after-scaling check. The geometry dependence is therefore checked by displacing one
    torsion atom (a configuration change the adapter must see). The controlled-distribution NPT test
    of spec 17 is a Stage D prerequisite (see the handoff list).
    """
    from pep_gamd_fixture import _fresh_system, solvated_dipeptide
    from gareus import npt, pep_gamd
    from gareus.pep_gamd import PepGamdLowerDualNptTargetAdapter
    from test_pep_gamd_boost import _gamd_kwargs
    openmm, _app, unit = import_openmm()
    table, d = _table()
    fx = solvated_dipeptide()
    system = _fresh_system()
    pep_gamd.ensure_pep_gamd_partition(system, fx["peptide"])
    rt = _with_aux(system, table, active=True)
    integ = pep_gamd.PepGaMDLowerDualIntegrator(pep_gamd.DIHEDRAL_GROUP,
                                                bias_force_groups=pep_gamd.pep_gamd_bias_force_groups(system),
                                                **_gamd_kwargs(unit))
    adapter = PepGamdLowerDualNptTargetAdapter(system, integ)
    assert rt.info.force_group in adapter._bias_groups
    ctx = openmm.Context(system, integ, openmm.Platform.getPlatformByName("Reference"))
    ctx.setPositions(d["positions_nm"])
    snap = adapter.snapshot(ctx, integ)
    other_bias = adapter.evaluate(ctx, snap).bias_kj_mol - aux_energy_kj(
        z_from_positions(d["positions_nm"], table.model), table.centers[0], K_KCAL)[0]
    # Molecular scaling leaves the aux energy unchanged (documented cancellation).
    mol_ids, mol_sizes = npt._molecule_index_arrays(npt._molecules_from_context(ctx), system.getNumParticles())
    trial = npt._scale_about_molecule_centroids(d["positions_nm"], mol_ids, mol_sizes, 0.02)
    assert abs(float(z_from_positions(trial, table.model)[0]) - float(z_from_positions(d["positions_nm"], table.model)[0])) < 1e-9
    # Displace one torsion atom: the adapter's bias must follow the new A_s exactly.
    moved = d["positions_nm"].copy()
    atom = table.model.feature_schema.features[0].atom_indices[0]
    moved[atom] += np.array([0.02, -0.015, 0.01])
    ctx.setPositions(moved)
    after = adapter.evaluate(ctx, snap)
    expected = aux_energy_kj(z_from_positions(moved, table.model), table.centers[0], K_KCAL)[0]
    other_after = after.bias_kj_mol - expected
    assert abs(expected - aux_energy_kj(z_from_positions(d["positions_nm"], table.model), table.centers[0], K_KCAL)[0]) > 1e-3
    # The non-aux bias groups (umbrellas) are absent here, so the remainder is unchanged (0 both times).
    assert abs(other_after - other_bias) < 1e-8
