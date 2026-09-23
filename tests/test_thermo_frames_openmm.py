"""OpenMM oracles for gareus.mbar_analysis.thermo_frames (peptide-only PME energies, pp/pe split).

Independent checks, none of which reuse the extraction code under test:
- a peptide-only System built from a water-deleted topology by the force field itself;
- charge / epsilon scaling of the peptide inside the FULL system, which separates pp, pe and ee
  exactly because Ewald/PME and LJ-with-geometric-epsilon are bilinear/quadratic in the scaled
  parameters at a fixed alpha and grid;
- the production kernel's own v_pep (E0 - E1 + E2) on a partitioned copy.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("openmm")

from pep_gamd_fixture import solvated_dipeptide, tiny_solvated_system  # noqa: E402
from gareus.mbar_analysis import thermo_frames as tf  # noqa: E402


def _ctx_energy(openmm, unit, system, positions, groups=None, box=None):
    ctx = openmm.Context(system, openmm.VerletIntegrator(0.001 * unit.picoseconds), openmm.Platform.getPlatformByName("Reference"))
    if box is not None:
        ctx.setPeriodicBoxVectors(*box)
    ctx.setPositions(positions)
    kw = {} if groups is None else {"groups": groups}
    e = ctx.getState(getEnergy=True, **kw).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    del ctx
    return float(e)


def _nb_only_copy(openmm, system, pme, *, pep=None, q_scale=1.0, eps_scale=1.0, charges=True, lj=True, lrc=None):
    """Copy of the system with ONLY its physical NonbondedForce, peptide params scaled."""
    from gareus.seeding import deserialize_system
    s = deserialize_system(openmm, system)
    for i in reversed(range(s.getNumForces())):
        if not isinstance(s.getForce(i), openmm.NonbondedForce):
            s.removeForce(i)
    nb = s.getForce(0)
    nb.setPMEParameters(*pme)
    if lrc is not None:
        nb.setUseDispersionCorrection(lrc)
    pep = set(pep or [])
    for i in range(nb.getNumParticles()):
        q, sg, e = nb.getParticleParameters(i)
        if i in pep:
            q, e = q * q_scale, e * eps_scale
        nb.setParticleParameters(i, q if charges else 0.0 * q, sg, e if lj else 0.0 * e)
    for k in range(nb.getNumExceptions()):
        a, b, qq, sg, e = nb.getExceptionParameters(k)
        f_q = (q_scale if a in pep else 1.0) * (q_scale if b in pep else 1.0)
        f_e = (eps_scale if a in pep else 1.0) * (eps_scale if b in pep else 1.0)
        f_e = f_e ** 0.5 if f_e > 0 else 0.0
        nb.setExceptionParameters(k, a, b, qq * f_q if charges else 0.0 * qq, sg, e * f_e if lj else 0.0 * e)
    return s


@pytest.fixture(scope="module")
def fx():
    openmm, app, unit, topology, system, positions = tiny_solvated_system()
    pep = solvated_dipeptide()["peptide"]
    pme = tf.resolved_pme_parameters(system, openmm, unit)
    box = system.getDefaultPeriodicBoxVectors()
    return dict(openmm=openmm, app=app, unit=unit, topology=topology, system=system, positions=positions,
                pep=pep, pme=pme, box=box)


def _pep_positions(fx):
    unit = fx["unit"]
    xyz = np.array(fx["positions"].value_in_unit(unit.nanometer))
    return xyz[fx["pep"]]


def test_builder_matches_forcefield_built_peptide_system(fx):
    openmm, app, unit = fx["openmm"], fx["app"], fx["unit"]
    ev = tf.PeptideOnlyEvaluator(fx["system"], fx["pep"], openmm, unit, platform="Reference", pme=fx["pme"])
    box = np.array([[v[i].value_in_unit(unit.nanometer) for i in range(3)] for v in fx["box"]])
    e_nb, e_bond, e_dih = ev.evaluate(_pep_positions(fx), box)
    # independent: delete everything but the peptide and let the force field build it
    m = app.Modeller(fx["topology"], fx["positions"])
    m.delete([r for r in m.topology.residues() if not any(a.index in set(fx["pep"]) for a in r.atoms())])
    ff = solvated_dipeptide()["ff"]
    ref = ff.createSystem(m.topology, nonbondedMethod=app.PME, nonbondedCutoff=0.9 * unit.nanometer,
                          constraints=app.HBonds, rigidWater=True, ewaldErrorTolerance=0.0005)
    ref.setDefaultPeriodicBoxVectors(*fx["box"])
    for f in ref.getForces():
        if isinstance(f, openmm.NonbondedForce):
            f.setPMEParameters(*fx["pme"]); f.setUseDispersionCorrection(False); f.setForceGroup(0)
        elif isinstance(f, (openmm.PeriodicTorsionForce, openmm.CMAPTorsionForce)):
            f.setForceGroup(2)
        else:
            f.setForceGroup(1)
    pos = m.positions
    assert e_nb == pytest.approx(_ctx_energy(openmm, unit, ref, pos, groups={0}), abs=1e-6)
    assert e_bond == pytest.approx(_ctx_energy(openmm, unit, ref, pos, groups={1}), abs=1e-6)
    assert e_dih == pytest.approx(_ctx_energy(openmm, unit, ref, pos, groups={2}), abs=1e-6)


def test_peptide_only_electrostatics_equal_charge_scaling_pp_term(fx):
    """E_el(s) = s^2 pp + s pe + ee in the full system: pp from s = +-1, 0 must equal the builder's."""
    openmm, unit = fx["openmm"], fx["unit"]
    E = {s: _ctx_energy(openmm, unit, _nb_only_copy(openmm, fx["system"], fx["pme"], pep=fx["pep"], q_scale=s, lj=False, lrc=False),
                        fx["positions"]) for s in (-1.0, 0.0, 1.0)}
    pp_el = 0.5 * (E[1.0] + E[-1.0]) - E[0.0]
    pep_sys, _, _ = tf.build_peptide_only_system(fx["system"], fx["pep"], openmm, unit, pme=fx["pme"])
    for f in pep_sys.getForces():
        if isinstance(f, openmm.NonbondedForce):
            for i in range(f.getNumParticles()):
                q, sg, e = f.getParticleParameters(i); f.setParticleParameters(i, q, sg, 0.0 * e)
            for k in range(f.getNumExceptions()):
                a, b, qq, sg, e = f.getExceptionParameters(k); f.setExceptionParameters(k, a, b, qq, sg, 0.0 * e)
    got = _ctx_energy(openmm, unit, pep_sys, _pep_positions(fx) * unit.nanometer, groups={tf.NB_GROUP})
    assert got == pytest.approx(pp_el, abs=1e-4)


def test_production_v_pep_minus_v_pp_is_the_pe_interaction(fx):
    openmm, unit = fx["openmm"], fx["unit"]
    from gareus.seeding import deserialize_system
    from gareus.pep_gamd import ensure_pep_gamd_partition, peptide_essential_energy_kj
    part = deserialize_system(openmm, fx["system"])
    ensure_pep_gamd_partition(part, fx["pep"])
    ctx = openmm.Context(part, openmm.VerletIntegrator(0.001 * unit.picoseconds), openmm.Platform.getPlatformByName("Reference"))
    ctx.setPositions(fx["positions"])
    v_pep = peptide_essential_energy_kj(ctx, unit)
    del ctx
    ev = tf.PeptideOnlyEvaluator(fx["system"], fx["pep"], openmm, unit, platform="Reference", pme=fx["pme"])
    e_nb, e_bond, e_dih = ev.evaluate(_pep_positions(fx), None)
    v_pe = v_pep - (e_nb + e_bond + e_dih)
    # independent pe: Coulomb from charge scaling, LJ from epsilon scaling (pair eps ~ sqrt(eps_i eps_j))
    Eq = {s: _ctx_energy(openmm, unit, _nb_only_copy(openmm, fx["system"], fx["pme"], pep=fx["pep"], q_scale=s, lj=False, lrc=False),
                         fx["positions"]) for s in (-1.0, 1.0)}
    pe_el = 0.5 * (Eq[1.0] - Eq[-1.0])
    El = {s: _ctx_energy(openmm, unit, _nb_only_copy(openmm, fx["system"], fx["pme"], pep=fx["pep"], eps_scale=s * s, charges=False, lrc=False),
                         fx["positions"]) for s in (0.0, 0.5, 1.0)}
    # E(s) = s^2 pp + s pe + ee  ->  solve with s = 0, 0.5, 1
    ee, a, b = El[0.0], El[0.5], El[1.0]
    pe_lj = 4 * a - b - 3 * ee
    lrc_full = _ctx_energy(openmm, unit, _nb_only_copy(openmm, fx["system"], fx["pme"], lrc=True), fx["positions"]) \
        - _ctx_energy(openmm, unit, _nb_only_copy(openmm, fx["system"], fx["pme"], lrc=False), fx["positions"])
    lrc_env = _ctx_energy(openmm, unit, _nb_only_copy(openmm, fx["system"], fx["pme"], pep=fx["pep"], q_scale=0.0, eps_scale=0.0, lrc=True), fx["positions"]) \
        - _ctx_energy(openmm, unit, _nb_only_copy(openmm, fx["system"], fx["pme"], pep=fx["pep"], q_scale=0.0, eps_scale=0.0, lrc=False), fx["positions"])
    assert v_pe == pytest.approx(pe_el + pe_lj + (lrc_full - lrc_env), abs=2e-3)
    assert abs(pe_el) > 1.0  # the check is not vacuous


def test_backbone_quads_and_dihedral_angles(fx):
    quads, labels = tf.backbone_torsion_quads(fx["topology"], fx["pep"])
    assert labels == [l for l in labels if l.startswith(("phi-", "psi-"))] and len(quads) >= 2
    # analytic geometry: a planar trans chain has dihedral pi, a 90-degree twist gives +-pi/2
    xyz = np.array([[[1, 0, 0], [0, 0, 0], [0, 1, 0], [-1, 1, 0]],
                    [[1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 1, 1]]], dtype=float)
    ang = tf.dihedral_angles(xyz, [(0, 1, 2, 3)])
    assert abs(abs(ang[0, 0]) - np.pi) < 1e-12 and abs(abs(ang[1, 0]) - np.pi / 2) < 1e-12


def test_coupling_term_is_refused(fx):
    openmm, unit = fx["openmm"], fx["unit"]
    from gareus.seeding import deserialize_system
    s = deserialize_system(openmm, fx["system"])
    bad = openmm.HarmonicBondForce(); bad.addBond(fx["pep"][0], max(fx["pep"]) + 1, 0.1, 1.0); s.addForce(bad)
    with pytest.raises(ValueError, match="couples peptide"):
        tf.build_peptide_only_system(s, fx["pep"], openmm, unit, pme=fx["pme"])


def test_evaluate_sample_frames_aligns_by_step_and_later_segment_wins(fx, tmp_path):
    """Real XTCs: base segment + a resume segment starting off-multiple (1900) that overlaps it."""
    import json
    md = pytest.importorskip("mdtraj")
    from pathlib import Path
    from gareus.mbar_analysis.data import Data
    openmm, app, unit = fx["openmm"], fx["app"], fx["unit"]
    root = tmp_path / "run"; (root / "replica_trajectories").mkdir(parents=True)
    app.PDBFile.writeFile(fx["topology"], fx["positions"], open(root / "01_solvated_start.pdb", "w"))
    (root / "run_args.json").write_text(json.dumps(dict(nonbonded_cutoff_nm=0.9, ewald_error_tolerance=0.0005, hmr=False,
                                                        hydrogen_mass_amu=0.0, forcefield="ff14SB", water_model="tip3p",
                                                        timestep_fs=3.5, traj_interval=250)))
    full = md.Topology.from_openmm(fx["topology"])
    sol = full.subset(fx["pep"])
    base_xyz = np.array(fx["positions"].value_in_unit(unit.nanometer))[fx["pep"]]
    md.Trajectory(base_xyz[None], sol).save_pdb(str(root / "solute_only.pdb"))
    box = np.array([[v[i].value_in_unit(unit.nanometer) for i in range(3)] for v in fx["box"]])
    rng = np.random.default_rng(0)

    def write(name, steps):
        xyz = base_xyz[None] + rng.normal(0, 0.01, (len(steps), len(base_xyz), 3))
        tr = md.Trajectory(xyz.astype(np.float32), sol, time=np.asarray(steps) * 0.0035,
                           unitcell_lengths=np.tile(np.diag(box), (len(steps), 1)), unitcell_angles=np.full((len(steps), 3), 90.0))
        tr.save_xtc(str(root / "replica_trajectories" / name))

    write("replica_000.xtc", list(range(250, 2750, 250)))                          # 250..2500
    write("replica_000_resume_from_000001900.xtc", list(range(2000, 3250, 250)))   # 2000..3000 overrides
    steps = np.array(list(range(250, 3250, 250)) + [3250])                         # last sample has no frame
    n = steps.size
    d = Data(prod_dir=root, out_dir=root / "pmf", cv=np.zeros(n), cv2=np.full(n, np.nan), rg_A=np.full(n, np.nan),
             window=np.zeros(n, int), replica=np.zeros(n, int), step=steps, u_nk=np.zeros((n, 1)), centers=np.zeros(1),
             k_kcal=np.ones(1), beta=0.4, temp=300.0, boost_kj=np.zeros(n), potential_kj=np.zeros(n), source="t", meta={})
    out = tf.evaluate_sample_frames(d, platform="Reference")
    assert np.isnan(out["e_dih"][-1]) and np.all(np.isfinite(out["e_dih"][:-1]))
    ev = tf.PeptideOnlyEvaluator(fx["system"], fx["pep"], openmm, unit, platform="Reference")
    res = md.load(str(root / "replica_trajectories" / "replica_000_resume_from_000001900.xtc"), top=str(root / "solute_only.pdb"))
    basef = md.load(str(root / "replica_trajectories" / "replica_000.xtc"), top=str(root / "solute_only.pdb"))
    i_2250 = int(np.flatnonzero(steps == 2250)[0]); i_500 = int(np.flatnonzero(steps == 500)[0])
    exp_2250 = ev.evaluate(res.xyz[1].astype(float), box)       # resume frame at 2250, not the base one
    exp_500 = ev.evaluate(basef.xyz[1].astype(float), box)
    assert out["e_dih"][i_2250] == pytest.approx(exp_2250[2], abs=1e-6)
    assert out["e_nb"][i_2250] == pytest.approx(exp_2250[0], abs=1e-6)
    assert out["e_dih"][i_500] == pytest.approx(exp_500[2], abs=1e-6)
    assert out["e_dih"][i_2250] != pytest.approx(ev.evaluate(basef.xyz[8].astype(float), box)[2], abs=1e-6)
    assert out["diag"]["n_samples_with_frames"] == n - 1
    assert np.all(np.isfinite(out["theta"][:-1])) and np.all(np.isnan(out["theta"][-1]))
    strided = tf.evaluate_sample_frames(d, platform="Reference", stride=2)
    kept = np.isfinite(strided["e_dih"])
    assert np.array_equal(kept[:-1], (steps[:-1] // 250) % 2 == 0)


def test_multiple_cmap_forces_are_all_kept(fx):
    openmm, unit = fx["openmm"], fx["unit"]
    s = openmm.System()
    for _ in range(8):
        s.addParticle(12.0)
    s.setDefaultPeriodicBoxVectors(openmm.Vec3(3, 0, 0), openmm.Vec3(0, 3, 0), openmm.Vec3(0, 0, 3))
    nb = openmm.NonbondedForce(); nb.setNonbondedMethod(openmm.NonbondedForce.PME); nb.setCutoffDistance(0.9)
    for _ in range(8):
        nb.addParticle(0.0, 0.3, 0.0)
    s.addForce(nb)
    rng = np.random.default_rng(3)
    for k in range(2):
        c = openmm.CMAPTorsionForce()
        c.addMap(4, list(rng.normal(0, 5, 16)))
        c.addTorsion(0, 0, 1, 2, 3, 1, 2, 3, 4) if k == 0 else c.addTorsion(0, 2, 3, 4, 5, 3, 4, 5, 6)
        c.setForceGroup(2)
        s.addForce(c)
    pos = [openmm.Vec3(*v) for v in rng.uniform(0.5, 2.5, (8, 3))]
    ref = _ctx_energy(openmm, unit, s, pos, groups={2})
    pep_sys, _, _ = tf.build_peptide_only_system(s, range(8), openmm, unit, pme=(3.0, 20, 20, 20))
    cm = [f for f in pep_sys.getForces() if isinstance(f, openmm.CMAPTorsionForce)]
    assert len(cm) == 1 and cm[0].getNumMaps() == 2 and cm[0].getNumTorsions() == 2
    assert _ctx_energy(openmm, unit, pep_sys, pos, groups={tf.DIHEDRAL_GROUP}) == pytest.approx(ref, abs=1e-9)
    assert abs(ref) > 1e-3


def test_epoch_source_dirs_mapping_is_explicit_or_refused(tmp_path):
    from gareus.mbar_analysis.data import Data
    n = 3
    base = dict(prod_dir=tmp_path, out_dir=tmp_path, cv=np.zeros(n), cv2=np.full(n, np.nan), rg_A=np.full(n, np.nan),
                window=np.zeros(n, int), replica=np.zeros(n, int), step=np.array([250, 500, 750]), u_nk=np.zeros((n, 1)),
                centers=np.zeros(1), k_kcal=np.ones(1), beta=0.4, temp=300.0, boost_kj=np.zeros(n), potential_kj=np.zeros(n),
                source="t")
    # three discovered phases, the middle one contributed nothing -> _epoch_source {0, 1} means phases A and C
    d = Data(**base, meta={"_epoch_source": [0, 0, 1], "adaptive_epoch_run_dirs": ["A", "B", "C"]})
    with pytest.raises(ValueError, match="_epoch_source"):
        tf.evaluate_sample_frames(d, platform="Reference")


def test_mie_entropy_rejects_nan_bins():
    from gareus.mbar_analysis.thermo_entropy import mie_entropy, torsion_bins
    b = torsion_bins(np.array([[0.1, np.nan], [0.2, 0.3]]), 12)
    with pytest.raises(ValueError, match="NaN torsions"):
        mie_entropy(b, np.ones(2), 12)


def test_frame_alignment_oracle():
    ok = tf.peptide_energy_oracle_ok([10.0, 11.0, 12.1], [10.05, 10.98, 12.0])
    assert ok["ok"] and ok["n"] == 3
    bad = tf.peptide_energy_oracle_ok([10.0, 40.0, 70.0], [50.0, 10.0, 12.0])
    assert not bad["ok"]
