"""The bias force may be read directly instead of recovered by subtraction.

The Pep-GaMD integrator used to obtain the bias force (umbrella + secondary CV)
as ``f - f0 - f1 - f_dih``, which costs a full all-groups evaluation -- a second
PME pass over the whole system -- every MD step. Reading the bias groups
directly is the same sum, and on a 21,384-atom box it removes ~30% of the step.

These tests pin the identity that makes the substitution legitimate, and the
structural property that the expensive evaluation is really gone.
"""
import re

import numpy as np

from gareus.imports import import_openmm
from gareus.pep_gamd import (
    AUX_NONBONDED_GROUP,
    DIHEDRAL_GROUP,
    PHYSICAL_NONBONDED_GROUP,
    ensure_pep_gamd_partition,
    pep_gamd_bias_force_groups,
)
from pep_gamd_fixture import solvated_dipeptide, tiny_solvated_system  # noqa: E402  (tests dir is on sys.path)

UMBRELLA_GROUP = 31
SECONDARY_CV_GROUP = 29


def _add_bias_forces(openmm, system, peptide):
    """An umbrella and a secondary CV of the shapes production actually uses.

    Umbrella: single-bond CustomBondForce (forces.py add_umbrella_force).
    Secondary CV: CustomCVForce over a torsion-score inner force
    (production.py add_secondary_structure_cv_force). A CustomCVForce is used
    deliberately -- it evaluates its own inner forces, so it is the most
    expensive thing the bias groups actually contain.
    """
    umb = openmm.CustomBondForce("0.5*k*(r-r0)^2")
    umb.addGlobalParameter("k", 500.0)
    umb.addGlobalParameter("r0", 0.4)
    umb.addBond(int(peptide[0]), int(peptide[-1]), [])
    umb.setForceGroup(UMBRELLA_GROUP)
    system.addForce(umb)

    inner = openmm.CustomTorsionForce("cos(theta-phi0)")
    inner.addGlobalParameter("phi0", -1.05)
    quads = [peptide[i:i + 4] for i in range(0, max(0, len(peptide) - 4), 4)]
    for q in quads[:6]:
        inner.addTorsion(int(q[0]), int(q[1]), int(q[2]), int(q[3]), [])
    cv = openmm.CustomCVForce("0.5*kcv*score^2")
    cv.addGlobalParameter("kcv", 10.0)
    cv.addCollectiveVariable("score", inner)
    cv.setForceGroup(SECONDARY_CV_GROUP)
    system.addForce(cv)
    return umb, cv


def _partitioned_system(with_bias=True):
    openmm, _app, unit, _topology, system, positions = tiny_solvated_system()
    peptide = solvated_dipeptide()["peptide"]
    ensure_pep_gamd_partition(system, peptide)
    if with_bias:
        _add_bias_forces(openmm, system, peptide)
    return openmm, unit, system, positions, peptide


def _perdof(ig, name):
    names = [ig.getPerDofVariableName(i) for i in range(ig.getNumPerDofVariables())]
    return np.array(ig.getPerDofVariable(names.index(name)))


def _probe_both_bias_forms(openmm, unit, system, positions, bias_groups):
    """Compute BOTH bias expressions inside ONE context, from ONE state.

    A single integrator guarantees the two expressions see identical
    coordinates, so any difference is floating-point association alone rather
    than a different configuration. Reference platform: double precision, so the
    residual measured here is the algebra's, not the GPU's.
    """
    ig = openmm.CustomIntegrator(0.001 * unit.picoseconds)
    for nm in ("PepF0", "PepF1", "PepF2", "PepFall", "OldBias", "NewBias"):
        ig.addPerDofVariable(nm, 0.0)
    for j in range(len(bias_groups)):
        ig.addPerDofVariable("Fb%d" % j, 0.0)
    ig.addComputePerDof("PepF0", "f%d" % PHYSICAL_NONBONDED_GROUP)
    ig.addComputePerDof("PepF1", "f%d" % AUX_NONBONDED_GROUP)
    ig.addComputePerDof("PepF2", "f%d" % DIHEDRAL_GROUP)
    ig.addComputePerDof("PepFall", "f")
    ig.addComputePerDof("OldBias", "PepFall - PepF0 - PepF1 - PepF2")
    for j, g in enumerate(bias_groups):
        ig.addComputePerDof("Fb%d" % j, "f%d" % g)
    expr = " + ".join("Fb%d" % j for j in range(len(bias_groups))) or "0"
    ig.addComputePerDof("NewBias", expr)

    ctx = openmm.Context(system, ig, openmm.Platform.getPlatformByName("Reference"))
    ctx.setPositions(positions)
    ig.step(1)
    old = _perdof(ig, "OldBias")
    new = _perdof(ig, "NewBias")
    allf = _perdof(ig, "PepFall")
    del ctx
    return old, new, allf


def test_bias_groups_are_the_complement_of_the_boosted_set():
    _openmm, _unit, system, _pos, _pep = _partitioned_system(with_bias=True)
    assert pep_gamd_bias_force_groups(system) == (SECONDARY_CV_GROUP, UMBRELLA_GROUP)


def test_a_system_with_no_bias_forces_reports_no_bias_groups():
    _openmm, _unit, system, _pos, _pep = _partitioned_system(with_bias=False)
    assert pep_gamd_bias_force_groups(system) == ()


def test_the_boosted_groups_are_never_reported_as_bias():
    _openmm, _unit, system, _pos, _pep = _partitioned_system(with_bias=True)
    groups = set(pep_gamd_bias_force_groups(system))
    assert groups.isdisjoint(
        {PHYSICAL_NONBONDED_GROUP, AUX_NONBONDED_GROUP, DIHEDRAL_GROUP})


def test_direct_bias_read_equals_the_subtraction_it_replaces():
    """The identity the whole change rests on.

    Not asserted bit-identical: the two forms sum the same contributions in a
    different association order. The tolerance is scaled to the BIAS force --
    the quantity actually being computed -- because agreeing to 1e-6 of a
    PME-magnitude total would be a vacuous test.
    """
    openmm, unit, system, positions, _pep = _partitioned_system(with_bias=True)
    groups = pep_gamd_bias_force_groups(system)
    old, new, allf = _probe_both_bias_forms(openmm, unit, system, positions, groups)

    bias_scale = float(np.abs(new).max())
    assert bias_scale > 0.0, "the fixture must exert a non-zero bias force"
    dev = float(np.abs(old - new).max())
    assert dev <= 1e-6 * bias_scale, (
        "bias force differs by %.3e kJ/mol/nm against a bias scale of %.3e; "
        "total force scale %.3e" % (dev, bias_scale, float(np.abs(allf).max())))


def test_the_subtraction_form_carries_cancellation_error():
    """Cancellation, quantified: the old form's error floor is the TOTAL scale.

    ``f_all - f0 - f1 - f2`` recovers a small restraint force by cancelling
    arrays orders of magnitude larger, so its accuracy is set by the total force
    magnitude, not the bias magnitude. Documents that the direct read is an
    improvement in conditioning, not merely a speedup.
    """
    openmm, unit, system, positions, _pep = _partitioned_system(with_bias=True)
    groups = pep_gamd_bias_force_groups(system)
    old, new, allf = _probe_both_bias_forms(openmm, unit, system, positions, groups)
    total_scale = float(np.abs(allf).max())
    dev = float(np.abs(old - new).max())
    assert dev <= 1e-9 * max(total_scale, 1.0)


def _build_integrator(bias_groups):
    from gareus.pep_gamd import _integrator_class
    _openmm, _app, unit = import_openmm()
    return _integrator_class()(
        DIHEDRAL_GROUP,
        bias_force_groups=bias_groups,
        dt=0.002 * unit.picoseconds,
        ntcmdprep=10, ntcmd=20, ntebprep=10, nteb=20, nstlim=100, ntave=10,
        sigma0p=6.0 * unit.kilocalories_per_mole,
        sigma0d=6.0 * unit.kilocalories_per_mole,
        temperature=300.0 * unit.kelvin,
    )


def _step_expressions(ig):
    return [ig.getComputationStep(i)[2] for i in range(ig.getNumComputations())]


def _reads_all_groups(expr):
    """True if `expr` reads the bare `f` (all groups) rather than `fN`."""
    return bool(re.search(r"(?<![A-Za-z0-9_])f(?![0-9A-Za-z_])", expr))


def test_reads_all_groups_helper_discriminates_bare_f_from_masked_f():
    """The structural tests are only as good as this predicate."""
    assert _reads_all_groups("f")
    assert _reads_all_groups("PepFall - f")
    assert not _reads_all_groups("f0")
    assert not _reads_all_groups("f29")
    assert not _reads_all_groups("PepF0 + PepF2")
    assert not _reads_all_groups("fscale*x")


def test_integrator_no_longer_evaluates_every_force_group():
    """The expensive all-groups read must be gone from every stage.

    A bare ``f`` is what triggered the second PME pass; ``f0``/``f29`` are
    group-masked and cheap. Scanning every computation step means a
    reintroduction anywhere in the program fails the test.
    """
    ig = _build_integrator((SECONDARY_CV_GROUP, UMBRELLA_GROUP))
    bare = [e for e in _step_expressions(ig) if _reads_all_groups(e)]
    assert not bare, "all-groups force read still present: %r" % (bare,)


def test_integrator_reads_each_bias_group_directly():
    ig = _build_integrator((SECONDARY_CV_GROUP, UMBRELLA_GROUP))
    exprs = [e.strip() for e in _step_expressions(ig)]
    for g in (SECONDARY_CV_GROUP, UMBRELLA_GROUP):
        assert ("f%d" % g) in exprs, "group %d never read directly" % g


def test_integrator_without_bias_groups_still_builds():
    """A Pep-GaMD system carrying no umbrella is valid; the term becomes 0."""
    ig = _build_integrator(())
    exprs = _step_expressions(ig)
    assert not [e for e in exprs if _reads_all_groups(e)]
    assert any("(0)" in e for e in exprs)


def test_conventional_md_stage_still_excludes_the_auxiliary_force():
    """cMD applies f0 + f_dih + bias, i.e. (f - f1); the aux force must not leak in.

    Regression guard for the rewrite: the old form subtracted PepF1 explicitly,
    the new one never evaluates it, so an error here would silently add the
    water-only force to the cMD stages.
    """
    openmm, unit, system, positions, _pep = _partitioned_system(with_bias=True)
    groups = pep_gamd_bias_force_groups(system)

    ig = openmm.CustomIntegrator(0.001 * unit.picoseconds)
    for nm in ("A", "B", "PepF0", "PepF1", "PepF2", "PepFall"):
        ig.addPerDofVariable(nm, 0.0)
    for j in range(len(groups)):
        ig.addPerDofVariable("Fb%d" % j, 0.0)
    ig.addComputePerDof("PepF0", "f%d" % PHYSICAL_NONBONDED_GROUP)
    ig.addComputePerDof("PepF1", "f%d" % AUX_NONBONDED_GROUP)
    ig.addComputePerDof("PepF2", "f%d" % DIHEDRAL_GROUP)
    ig.addComputePerDof("PepFall", "f")
    ig.addComputePerDof("A", "PepFall - PepF1")
    for j, g in enumerate(groups):
        ig.addComputePerDof("Fb%d" % j, "f%d" % g)
    expr = " + ".join("Fb%d" % j for j in range(len(groups))) or "0"
    ig.addComputePerDof("B", "PepF0 + PepF2 + (%s)" % expr)

    ctx = openmm.Context(system, ig, openmm.Platform.getPlatformByName("Reference"))
    ctx.setPositions(positions)
    ig.step(1)
    a = _perdof(ig, "A")
    b = _perdof(ig, "B")
    del ctx
    scale = max(float(np.abs(a).max()), 1.0)
    assert float(np.abs(a - b).max()) <= 1e-9 * scale


def test_full_gamd_applied_force_is_unchanged():
    """The complete velocity increment, not just the bias sub-expression.

    The sub-expression identity plus "bare f is gone" would already imply this,
    but only if the rewritten format string is right. This evaluates both whole
    applied-force expressions -- boost scaling factors included -- so a typo in
    the substitution cannot pass.
    """
    openmm, unit, system, positions, _pep = _partitioned_system(with_bias=True)
    groups = pep_gamd_bias_force_groups(system)
    t, d = 0.83, 0.61  # stand-ins for ForceScalingFactor_Total / _Dihedral

    ig = openmm.CustomIntegrator(0.001 * unit.picoseconds)
    for nm in ("PepF0", "PepF1", "PepF2", "PepFall", "OldApplied", "NewApplied"):
        ig.addPerDofVariable(nm, 0.0)
    for j in range(len(groups)):
        ig.addPerDofVariable("Fb%d" % j, 0.0)
    ig.addComputePerDof("PepF0", "f%d" % PHYSICAL_NONBONDED_GROUP)
    ig.addComputePerDof("PepF1", "f%d" % AUX_NONBONDED_GROUP)
    ig.addComputePerDof("PepF2", "f%d" % DIHEDRAL_GROUP)
    ig.addComputePerDof("PepFall", "f")
    for j, g in enumerate(groups):
        ig.addComputePerDof("Fb%d" % j, "f%d" % g)
    bias = " + ".join("Fb%d" % j for j in range(len(groups))) or "0"

    ig.addComputePerDof(
        "OldApplied",
        "(PepF0 - PepF1)*{t} + PepF2*{t}*{d} + PepF1"
        " + (PepFall - PepF0 - PepF1 - PepF2)".format(t=t, d=d))
    ig.addComputePerDof(
        "NewApplied",
        "(PepF0 - PepF1)*{t} + PepF2*{t}*{d} + PepF1 + ({b})".format(t=t, d=d, b=bias))

    ctx = openmm.Context(system, ig, openmm.Platform.getPlatformByName("Reference"))
    ctx.setPositions(positions)
    ig.step(1)
    old = _perdof(ig, "OldApplied")
    new = _perdof(ig, "NewApplied")
    del ctx

    scale = max(float(np.abs(old).max()), 1.0)
    dev = float(np.abs(old - new).max())
    assert dev <= 1e-9 * scale, (
        "applied force differs by %.3e against scale %.3e" % (dev, scale))


def test_omitting_bias_force_groups_raises_instead_of_dropping_forces():
    """The failure mode this contract exists to prevent.

    An earlier draft defaulted the argument to (). A caller that forgot it then
    kept running happily and integrated WITHOUT the umbrella and secondary-CV
    forces -- a different potential, no crash, no NaN. It was caught only
    because an existing force-algebra test measured the applied force and found
    it short by exactly the group-31 contribution. Omission must be loud.
    """
    from gareus.pep_gamd import _integrator_class
    _openmm, _app, unit = import_openmm()
    try:
        _integrator_class()(
            DIHEDRAL_GROUP,
            dt=0.002 * unit.picoseconds,
            ntcmdprep=10, ntcmd=20, ntebprep=10, nteb=20, nstlim=100, ntave=10,
            sigma0p=6.0 * unit.kilocalories_per_mole,
            sigma0d=6.0 * unit.kilocalories_per_mole,
            temperature=300.0 * unit.kelvin,
        )
    except ValueError as exc:
        assert "bias_force_groups" in str(exc)
    else:
        raise AssertionError(
            "omitting bias_force_groups must raise, not silently drop the bias forces")
