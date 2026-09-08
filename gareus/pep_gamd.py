"""Pep-GaMD: boost the peptide essential potential, leave water-water alone.

V_pep = V_bonded(pep) + V_nb(pep-pep) + V_nb(pep-water)

PME reciprocal space cannot be partitioned by atom subset, so the partition is by
subtraction: an auxiliary water-only NonbondedForce (peptide charges/epsilon and
peptide exceptions zeroed, identical PME parameters) lives in its own force group,

    V_pep = energy0 - energy1 + energy2

The auxiliary force is a measuring instrument, not physics. It must exist only in
Systems driven by the Pep-GaMD integrator (or an integrator that excludes its
group), because a plain Langevin integrator applies every group and would count
water-water twice.

A second variant, ``pep-gamd-internal-lower-dual``, boosts the peptide-INTERNAL
energy instead:

    V_int = V_nb(pep-pep) + V_dihedral(pep)     (no peptide-water term at all)

Its auxiliary force is the mirror image -- a peptide-only NonbondedForce, every
non-peptide charge/epsilon zeroed, in force group 3 -- and it is added on its own
(the internal variant never needs the water-only force). Because the boosted
channel contains no peptide-water term, scaling it down can never let the solvent
collapse onto the peptide, which is the failure mode that capped k0 for the
essential variant at 4 fs with HMR. Peptide bonds and angles stay in group 0 and
are NOT boosted either; desolvation barriers are not accelerated.
"""
from __future__ import annotations

from typing import Iterable

PEP_GAMD_PREFIX = "pep-gamd-"
# The infix that distinguishes the peptide-internal variant from the essential one.
# gamd_calibration.threshold_and_k0 strips PEP_GAMD_PREFIX and then this, so that
# both variants dispatch to the same lower-bound threshold formula.
PEP_GAMD_INTERNAL_INFIX = "internal-"
PEP_GAMD_BOOST_TYPE = PEP_GAMD_PREFIX + "lower-dual"
PEP_GAMD_INTERNAL_BOOST_TYPE = PEP_GAMD_PREFIX + PEP_GAMD_INTERNAL_INFIX + "lower-dual"
AUX_FORCE_NAME = "PepGaMDWaterOnlyNonbonded"
AUX_PEPTIDE_FORCE_NAME = "PepGaMDPeptideOnlyNonbonded"
AUX_FORCE_NAMES = frozenset({AUX_FORCE_NAME, AUX_PEPTIDE_FORCE_NAME})

PHYSICAL_NONBONDED_GROUP = 0
AUX_NONBONDED_GROUP = 1
DIHEDRAL_GROUP = 2
AUX_PEPTIDE_GROUP = 3
PHYSICAL_GROUPS = frozenset({PHYSICAL_NONBONDED_GROUP, DIHEDRAL_GROUP})
# Groups Pep-GaMD owns: 0/2 are physical, 1/3 are the two auxiliary measuring
# instruments. Nothing else may be parked in any of them.
RESERVED_GROUPS = (PHYSICAL_NONBONDED_GROUP, AUX_NONBONDED_GROUP, DIHEDRAL_GROUP, AUX_PEPTIDE_GROUP)

# variant name per boost type; the analysis needs it to know what v_pep_kj_mol means
PEP_GAMD_VARIANTS = {PEP_GAMD_BOOST_TYPE: "essential", PEP_GAMD_INTERNAL_BOOST_TYPE: "internal"}

_DIHEDRAL_CLASSES = ("PeriodicTorsionForce", "CMAPTorsionForce")
_GROUP0_CLASSES = (
    "NonbondedForce", "HarmonicBondForce", "HarmonicAngleForce",
    "CMMotionRemover", "MonteCarloBarostat", "MonteCarloAnisotropicBarostat",
    "MonteCarloMembraneBarostat",
)


def find_named_force(system, name):
    """(index, force) of the force called ``name``, or (None, None)."""
    for i in range(system.getNumForces()):
        f = system.getForce(i)
        if f.getName() == name:
            return i, f
    return None, None


def find_aux_force(system):
    """(index, force) of the first Pep-GaMD auxiliary force of EITHER variant."""
    for i in range(system.getNumForces()):
        f = system.getForce(i)
        if f.getName() in AUX_FORCE_NAMES:
            return i, f
    return None, None


def aux_force_groups_present(system) -> set:
    """Force groups occupied by Pep-GaMD auxiliary forces in `system`.

    Read off each force's own getForceGroup() rather than the AUX_*_GROUP
    constants, so the answer stays correct for a System whose groups have not
    been (re)assigned yet.
    """
    return {int(system.getForce(i).getForceGroup())
            for i in range(system.getNumForces())
            if system.getForce(i).getName() in AUX_FORCE_NAMES}


def _physical_nonbonded(system, openmm):
    found = [
        (i, system.getForce(i)) for i in range(system.getNumForces())
        if isinstance(system.getForce(i), openmm.NonbondedForce)
        and system.getForce(i).getName() not in AUX_FORCE_NAMES
    ]
    if len(found) != 1:
        raise ValueError(f"Pep-GaMD needs exactly one physical NonbondedForce; found {len(found)}")
    return found[0]


def assign_pep_gamd_force_groups(system) -> None:
    """Physical forces -> groups 0/2, auxiliaries -> 1/3; refuse anything else in 0..3.

    The auxiliary NonbondedForces are matched by NAME before the class table, because
    they are NonbondedForces too and would otherwise be swept into the physical group 0
    and silently applied as physics.
    """
    for i in range(system.getNumForces()):
        f = system.getForce(i)
        name = f.__class__.__name__
        if f.getName() == AUX_FORCE_NAME:
            f.setForceGroup(AUX_NONBONDED_GROUP)
        elif f.getName() == AUX_PEPTIDE_FORCE_NAME:
            f.setForceGroup(AUX_PEPTIDE_GROUP)
        elif name in _DIHEDRAL_CLASSES:
            f.setForceGroup(DIHEDRAL_GROUP)
        elif name in _GROUP0_CLASSES:
            f.setForceGroup(PHYSICAL_NONBONDED_GROUP)
        elif f.getForceGroup() in RESERVED_GROUPS:
            raise ValueError(
                f"{name} sits in force group {f.getForceGroup()}, which Pep-GaMD reserves; "
                f"non-physical forces (umbrella, secondary CV) must use a group outside 0..3"
            )


def _alpha_per_nm(alpha, unit) -> float:
    return float(alpha.value_in_unit(unit.nanometer ** -1)) if hasattr(alpha, "value_in_unit") else float(alpha)


def pme_parameters_from_tolerance(cutoff_nm: float, tolerance: float, box_nm) -> tuple[float, int, int, int]:
    """OpenMM's own PME auto-choice (NonbondedForceImpl::calcPMEParameters), reproduced.

    alpha = sqrt(-ln(2 tol)) / r_c ; grid_i = ceil(2 alpha L_i / (3 tol^(1/5))).
    Pinning both NonbondedForces to the same explicit values guarantees the water-water
    reciprocal terms cancel exactly in energy0 - energy1. Like an auto-chosen grid, the
    pinned grid is fixed for the life of the Context (NPT does not re-grid either way).
    """
    import math
    alpha = math.sqrt(-math.log(2.0 * tolerance)) / float(cutoff_nm)
    grid = [int(math.ceil(2.0 * alpha * float(box_nm[i][i]) / (3.0 * tolerance ** 0.2))) for i in range(3)]
    return alpha, grid[0], grid[1], grid[2]


def _pin_pme_parameters(system, physical, openmm, unit):
    alpha, nx, ny, nz = physical.getPMEParameters()
    if _alpha_per_nm(alpha, unit) > 0.0:
        return alpha, nx, ny, nz
    cutoff_nm = float(physical.getCutoffDistance().value_in_unit(unit.nanometer))
    box = [[float(v[i].value_in_unit(unit.nanometer)) for i in range(3)] for v in system.getDefaultPeriodicBoxVectors()]
    alpha, nx, ny, nz = pme_parameters_from_tolerance(cutoff_nm, float(physical.getEwaldErrorTolerance()), box)
    physical.setPMEParameters(alpha, nx, ny, nz)
    return alpha, nx, ny, nz


def _peptide_atom_set(peptide_atoms: Iterable[int]) -> set:
    pep = set(int(i) for i in peptide_atoms)
    if not pep:
        raise ValueError("Pep-GaMD partition needs a non-empty peptide atom set")
    return pep


def _ensure_masked_nonbonded_copy(system, force_name: str, keep: set) -> int:
    """Add a named copy of the physical NonbondedForce in which only `keep` particles
    keep their real charge/epsilon, and assign the Pep-GaMD force groups.

    Idempotent: a second call finds the existing force by name and only re-asserts the
    group layout. Every exception with an endpoint outside `keep` is zeroed too, so the
    copy's energy is exactly the nonbonded energy of the `keep` subset (its own PME
    reciprocal term included). Returns the added force's index.
    """
    from .imports import import_openmm
    openmm, _app, unit = import_openmm()

    idx, existing = find_named_force(system, force_name)
    if existing is not None:
        assign_pep_gamd_force_groups(system)
        return idx

    _phys_idx, physical = _physical_nonbonded(system, openmm)
    alpha, nx, ny, nz = _pin_pme_parameters(system, physical, openmm, unit)

    aux = openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(physical))
    aux.setName(force_name)
    aux.setPMEParameters(alpha, nx, ny, nz)
    for i in range(aux.getNumParticles()):
        if i not in keep:
            _q, sig, _eps = aux.getParticleParameters(i)
            aux.setParticleParameters(i, 0.0, sig, 0.0)
    for k in range(aux.getNumExceptions()):
        a, b, _qq, sig, _eps = aux.getExceptionParameters(k)
        if a not in keep or b not in keep:
            aux.setExceptionParameters(k, a, b, 0.0, sig, 0.0)
    idx = system.addForce(aux)
    assign_pep_gamd_force_groups(system)
    return idx


def ensure_pep_gamd_partition(system, peptide_atoms: Iterable[int]) -> int:
    """Add the water-only auxiliary NonbondedForce (group 1) and assign force groups.

    Idempotent: a second call finds the existing auxiliary force by name and only
    re-asserts the group layout. Returns the auxiliary force's index.
    """
    pep = _peptide_atom_set(peptide_atoms)
    keep = set(range(system.getNumParticles())) - pep
    return _ensure_masked_nonbonded_copy(system, AUX_FORCE_NAME, keep)


def ensure_pep_gamd_internal_partition(system, peptide_atoms: Iterable[int]) -> int:
    """Add the peptide-only auxiliary NonbondedForce (group 3) and assign force groups.

    The mirror image of ``ensure_pep_gamd_partition``: every NON-peptide particle is
    zeroed, so the force's energy E3 is the peptide-internal nonbonded energy. It does
    NOT add the water-only force -- the internal variant never needs it. E3 carries its
    own PME reciprocal self-term, a constant across configurations, which shifts
    Vmax/Vmin/Vavg by the same amount and therefore leaves k0 and the boost unchanged.
    Idempotent. Returns the auxiliary force's index.
    """
    return _ensure_masked_nonbonded_copy(system, AUX_PEPTIDE_FORCE_NAME, _peptide_atom_set(peptide_atoms))


def peptide_essential_energy_kj(context, unit) -> float:
    e_phys = context.getState(getEnergy=True, groups=set(PHYSICAL_GROUPS)).getPotentialEnergy()
    e_aux = context.getState(getEnergy=True, groups={AUX_NONBONDED_GROUP}).getPotentialEnergy()
    return float((e_phys - e_aux).value_in_unit(unit.kilojoule_per_mole))


def peptide_internal_energy_kj(context, unit) -> float:
    """V_int = E3 + E2 for the internal variant -- ONE getState read, not two."""
    e = context.getState(getEnergy=True, groups={AUX_PEPTIDE_GROUP, DIHEDRAL_GROUP}).getPotentialEnergy()
    return float(e.value_in_unit(unit.kilojoule_per_mole))


def total_energy_groups(integrator) -> tuple[frozenset, frozenset]:
    """(plus, minus) force groups whose energies define this integrator's Total channel.

    Stock gamd-openmm integrators read bare ``energy`` (all groups); Pep-GaMD reads
    ``energy0 - energy1 + energy2``. Measurement code must follow the integrator's own
    definition or the envelope it calibrates will not be the one the boost uses.
    """
    plus = getattr(integrator, "TOTAL_ENERGY_PLUS_GROUPS", None)
    minus = getattr(integrator, "TOTAL_ENERGY_MINUS_GROUPS", None)
    if plus is None:
        return frozenset(range(32)), frozenset()
    return frozenset(plus), frozenset(minus or ())


def _build_integrator_classes():
    """Build both Pep-GaMD integrator classes (lazily: importing gamd is not free)."""
    from gamd.langevin.dual_boost_integrators import LowerBoundIntegrator
    from gamd.stage_integrator import BoostType

    class _PepGaMDBase(LowerBoundIntegrator):
        """Shared plumbing: which force group carries the dihedrals."""

        def _dihedral_group_id(self) -> int:
            (gid,) = [g for g, name in self.get_group_dict().items() if name == "Dihedral"]
            return int(gid)

    class PepGaMDLowerDualIntegrator(_PepGaMDBase):
        """Dual lower-bound GaMD on the peptide essential potential and peptide dihedrals.

        Total channel:    V_pep = energy0 - energy1 + energy2 (physical minus water-only)
        Dihedral channel: energy<dihedral group>, unchanged
        Applied force:    (f0 - f1)*FSF_T + f_dih*FSF_T*FSF_D + f1 + (f - f0 - f1 - f_dih)
        so water-water (f1) and every non-physical group (umbrella, secondary CV) are
        applied unscaled, and the integrator's own cMD stages exclude the auxiliary force.
        One force group per computation step, as OpenMM's CustomIntegrator requires.
        """

        TOTAL_ENERGY_PLUS_GROUPS = frozenset({PHYSICAL_NONBONDED_GROUP, DIHEDRAL_GROUP})
        TOTAL_ENERGY_MINUS_GROUPS = frozenset({AUX_NONBONDED_GROUP})

        def _add_common_variables(self):
            super()._add_common_variables()
            for name in ("PepF0", "PepF1", "PepF2", "PepFall"):
                self.addPerDofVariable(name, 0.0)
            for name in ("PepE0", "PepE1", "PepE2"):
                self.addGlobalVariable(name, 0.0)

        def _setup_energy_values(self):
            self.add_global_variables_by_name("StartingPotentialEnergy", 0.0)
            dih = self._dihedral_group_id()
            self.addComputeGlobal("PepE0", f"energy{PHYSICAL_NONBONDED_GROUP}")
            self.addComputeGlobal("PepE1", f"energy{AUX_NONBONDED_GROUP}")
            self.addComputeGlobal("PepE2", f"energy{dih}")
            self.addComputeGlobal(self._append_group_name("StartingPotentialEnergy", "Dihedral"), "PepE2")
            self.addComputeGlobal(self._append_group_name("StartingPotentialEnergy", BoostType.TOTAL.value),
                                  "PepE0 - PepE1 + PepE2")

        def _add_conventional_md_update_step(self):
            self.addComputePerDof("newx", "x")
            self.addComputePerDof("PepFall", "f")
            self.addComputePerDof("PepF1", f"f{AUX_NONBONDED_GROUP}")
            self.addComputePerDof("v", "vscale*v + fscale*(PepFall - PepF1)/m + noisescale*gaussian/sqrt(m)")
            self.addComputePerDof("x", "x+dt*v")
            self.addConstrainPositions()
            self.addComputePerDof("v", "(x-newx)/dt")

        def _add_gamd_update_step(self):
            dih = self._dihedral_group_id()
            fsf_t = self._append_group_name("ForceScalingFactor", BoostType.TOTAL.value)
            fsf_d = self._append_group_name("ForceScalingFactor", "Dihedral")
            self.addComputePerDof("newx", "x")
            self.addComputePerDof("v", "vscale*v + noisescale*gaussian/sqrt(m)")
            self.addComputePerDof("PepF0", f"f{PHYSICAL_NONBONDED_GROUP}")
            self.addComputePerDof("PepF1", f"f{AUX_NONBONDED_GROUP}")
            self.addComputePerDof("PepF2", f"f{dih}")
            self.addComputePerDof("PepFall", "f")
            self.addComputePerDof(
                "v",
                "v + fscale*((PepF0 - PepF1)*{t} + PepF2*{t}*{d} + PepF1 + (PepFall - PepF0 - PepF1 - PepF2))/m"
                .format(t=fsf_t, d=fsf_d),
            )
            self.addComputePerDof("x", "x+dt*v")
            self.addConstrainPositions()
            self.addComputePerDof("v", "(x-newx)/dt")

    class PepGaMDInternalLowerDualIntegrator(_PepGaMDBase):
        """Dual lower-bound GaMD on the peptide-INTERNAL potential and peptide dihedrals.

        Total channel:    V_int = energy3 + energy2 (peptide-only nonbonded + dihedrals)
        Dihedral channel: energy<dihedral group>, unchanged
        Applied force:    f_applied = (f - f3) + f3*FSF_T + f2*(FSF_T*FSF_D - 1)

        where ``f`` is the force over all integration groups (0, 2, 29, 31 ...; never 1
        or 3). The peptide-only auxiliary force in group 3 is a measuring instrument,
        not physics: ``f - f3`` is the physical force with the peptide-internal
        nonbonded part removed, which the next term puts back scaled by FSF_T. In the
        code ``f`` over EVERY group is read into PepFall and the physical force is
        formed as ``PepFall - PepF3``, so no reliance on setIntegrationForceGroups.
        At FSF_T = FSF_D = 1 the update collapses onto the cMD update, ``PepFall - PepF3``.

        Nothing in the boosted channel involves water, so scaling it down can never
        release the solvent onto the peptide. Peptide bonds/angles live in group 0 and
        are not boosted; desolvation barriers are not accelerated.
        One force group per computation step, as OpenMM's CustomIntegrator requires.
        """

        TOTAL_ENERGY_PLUS_GROUPS = frozenset({AUX_PEPTIDE_GROUP, DIHEDRAL_GROUP})
        TOTAL_ENERGY_MINUS_GROUPS = frozenset()

        def _add_common_variables(self):
            super()._add_common_variables()
            for name in ("PepF2", "PepF3", "PepFall"):
                self.addPerDofVariable(name, 0.0)
            for name in ("PepE2", "PepE3"):
                self.addGlobalVariable(name, 0.0)

        def _setup_energy_values(self):
            self.add_global_variables_by_name("StartingPotentialEnergy", 0.0)
            dih = self._dihedral_group_id()
            self.addComputeGlobal("PepE3", f"energy{AUX_PEPTIDE_GROUP}")
            self.addComputeGlobal("PepE2", f"energy{dih}")
            self.addComputeGlobal(self._append_group_name("StartingPotentialEnergy", "Dihedral"), "PepE2")
            self.addComputeGlobal(self._append_group_name("StartingPotentialEnergy", BoostType.TOTAL.value),
                                  "PepE3 + PepE2")

        def _add_conventional_md_update_step(self):
            self.addComputePerDof("newx", "x")
            self.addComputePerDof("PepFall", "f")
            self.addComputePerDof("PepF3", f"f{AUX_PEPTIDE_GROUP}")
            self.addComputePerDof("v", "vscale*v + fscale*(PepFall - PepF3)/m + noisescale*gaussian/sqrt(m)")
            self.addComputePerDof("x", "x+dt*v")
            self.addConstrainPositions()
            self.addComputePerDof("v", "(x-newx)/dt")

        def _add_gamd_update_step(self):
            dih = self._dihedral_group_id()
            fsf_t = self._append_group_name("ForceScalingFactor", BoostType.TOTAL.value)
            fsf_d = self._append_group_name("ForceScalingFactor", "Dihedral")
            self.addComputePerDof("newx", "x")
            self.addComputePerDof("v", "vscale*v + noisescale*gaussian/sqrt(m)")
            self.addComputePerDof("PepF2", f"f{dih}")
            self.addComputePerDof("PepF3", f"f{AUX_PEPTIDE_GROUP}")
            self.addComputePerDof("PepFall", "f")
            # (PepFall - PepF3) is the physical force f; the two correction terms
            # rescale its peptide-internal nonbonded and dihedral parts in place.
            self.addComputePerDof(
                "v",
                "v + fscale*((PepFall - PepF3) + PepF3*({t} - 1) + PepF2*({t}*{d} - 1))/m"
                .format(t=fsf_t, d=fsf_d),
            )
            self.addComputePerDof("x", "x+dt*v")
            self.addConstrainPositions()
            self.addComputePerDof("v", "(x-newx)/dt")

    return PepGaMDLowerDualIntegrator, PepGaMDInternalLowerDualIntegrator


_INTEGRATOR_CLASS_BY_VARIANT = {"essential": "PepGaMDLowerDualIntegrator",
                                "internal": "PepGaMDInternalLowerDualIntegrator"}


def __getattr__(name):
    if name in ("PepGaMDLowerDualIntegrator", "PepGaMDInternalLowerDualIntegrator"):
        return _integrator_class("internal" if "Internal" in name else "essential")
    raise AttributeError(name)


def pep_gamd_variant(args):
    """``"essential"``, ``"internal"``, or None for every non-Pep-GaMD boost type."""
    return PEP_GAMD_VARIANTS.get(str(getattr(args, "gamd_boost_type", "") or ""))


def is_pep_gamd(args) -> bool:
    return pep_gamd_variant(args) is not None


# Boost types the λ-ladder can scale per rung: the dependent dual Pep-GaMD boost
# (k0_Total and k0_Dihedral) and the stock single dihedral boost (k0_Dihedral only,
# no auxiliary force, 1x cost). Both use the lower-bound threshold E = Vmax, which
# is what pep_gamd_boost_kj reproduces in closed form; upper-bound variants use a
# different threshold rule and are deliberately excluded.
LADDER_BOOST_TYPES = frozenset({PEP_GAMD_BOOST_TYPE, PEP_GAMD_INTERNAL_BOOST_TYPE, "lower-dihedral"})


def ladder_supports_boost_type(args) -> bool:
    return str(getattr(args, "gamd_boost_type", "") or "") in LADDER_BOOST_TYPES


def prepare_pep_gamd_args(args, topology) -> None:
    """Record the peptide atom set on args while a topology is in hand.

    make_gamd_integrator() only sees (system, args, unit), so the partition needs the
    atom set carried in from wherever the topology lives.
    """
    if not is_pep_gamd(args):
        return
    from .energy_decomposition import peptide_atom_groups_from_topology
    _groups, pep, _nonpep = peptide_atom_groups_from_topology(topology, "all-peptide")
    args.pep_gamd_peptide_atoms = [int(i) for i in pep]


def physical_energy_groups_for_args(args):
    """Force groups that constitute the physical potential: everything but this
    variant's auxiliary measuring instrument (group 1 essential, group 3 internal)."""
    variant = pep_gamd_variant(args)
    if variant == "essential":
        return set(range(32)) - {AUX_NONBONDED_GROUP}
    if variant == "internal":
        return set(range(32)) - {AUX_PEPTIDE_GROUP}
    return -1


def physical_potential_energy_kj(context, system, unit) -> float:
    groups = set(range(32)) - aux_force_groups_present(system)
    return float(context.getState(getEnergy=True, groups=groups).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))


def total_energy_groups_for_args(args) -> tuple[frozenset, frozenset]:
    """The Total channel's (plus, minus) groups as a property of the boost type.

    The cMD-kind recon steps a plain Langevin integrator on an already-partitioned
    system, so the stepping integrator cannot be the source of this definition.
    """
    variant = pep_gamd_variant(args)
    if variant == "essential":
        return frozenset(PHYSICAL_GROUPS), frozenset({AUX_NONBONDED_GROUP})
    if variant == "internal":
        return frozenset({AUX_PEPTIDE_GROUP, DIHEDRAL_GROUP}), frozenset()
    return frozenset(range(32)), frozenset()


def boost_target_energy_kj(context, gid, unit, *, total_groups) -> float:
    """Energy of one GaMD boost target; ``total_groups`` = (plus, minus) for the Total channel."""
    def _e(groups):
        return float(context.getState(getEnergy=True, groups=set(groups)).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
    if gid is not None:
        return _e({int(gid)})
    plus, minus = total_groups
    e = _e(plus)
    if minus:
        e -= _e(minus)
    return e


def _integrator_class(variant: str = "essential"):
    name = _INTEGRATOR_CLASS_BY_VARIANT[variant]
    cls = globals().get(name)
    if cls is None:
        essential, internal = _build_integrator_classes()
        globals()["PepGaMDLowerDualIntegrator"] = essential
        globals()["PepGaMDInternalLowerDualIntegrator"] = internal
        cls = globals()[name]
    return cls


def build_pep_gamd_integrator(system, args, unit) -> list:
    """Partition `system` and build the Pep-GaMD integrator.

    Returns ``[aux_group, dihedral_group, integrator]`` so ``result[2]`` is the
    integrator, matching gamd-openmm's factory convention. ``aux_group`` is 1 for the
    essential variant and 3 for the internal one.
    """
    variant = pep_gamd_variant(args)
    atoms = getattr(args, "pep_gamd_peptide_atoms", None)
    if not atoms:
        raise ValueError(
            f"--gamd-boost-type {getattr(args, 'gamd_boost_type', PEP_GAMD_BOOST_TYPE)} needs "
            "args.pep_gamd_peptide_atoms; call prepare_pep_gamd_args(args, topology) after the "
            "system is built"
        )
    if variant == "internal":
        aux_group = AUX_PEPTIDE_GROUP
        ensure_pep_gamd_internal_partition(system, atoms)
    else:
        aux_group = AUX_NONBONDED_GROUP
        ensure_pep_gamd_partition(system, atoms)
    total_steps = (
        int(args.gamd_cmd_prep_steps) + int(args.gamd_cmd_steps)
        + int(args.gamd_equil_prep_steps) + int(args.gamd_equil_steps)
        + int(args.gamd_production_steps)
    )
    integrator = _integrator_class(variant or "essential")(
        DIHEDRAL_GROUP,
        dt=float(args.timestep_fs) * unit.femtosecond,
        ntcmdprep=int(args.gamd_cmd_prep_steps),
        ntcmd=int(args.gamd_cmd_steps),
        ntebprep=int(args.gamd_equil_prep_steps),
        nteb=int(args.gamd_equil_steps),
        nstlim=total_steps,
        ntave=int(args.gamd_averaging_window),
        sigma0p=float(args.sigma0p_kcal_mol) * unit.kilocalories_per_mole,
        sigma0d=float(args.sigma0d_kcal_mol) * unit.kilocalories_per_mole,
        temperature=float(args.temperature_k) * unit.kelvin,
    )
    return [aux_group, DIHEDRAL_GROUP, integrator]


from dataclasses import dataclass
import json as _json
import numpy as _np


@dataclass(frozen=True)
class PepGamdEnvelope:
    """Frozen per-channel GaMD envelope; k0max_* is the top rung (λ = 1)."""
    vmax_total: float; vmin_total: float; threshold_total: float; k0max_total: float
    vmax_dih: float;   vmin_dih: float;   threshold_dih: float;   k0max_dih: float
    # False for a single dihedral boost (stock ``lower-dihedral``): the globals carry no
    # ``*_Total`` entries, the Total channel contributes nothing, and v_pep is not needed.
    has_total: bool = True

    @classmethod
    def from_integrator_globals(cls, g: dict) -> "PepGamdEnvelope":
        f = lambda k: float(g[k])
        if "k0_Total" in g:
            return cls(f("Vmax_Total"), f("Vmin_Total"), f("threshold_energy_Total"), f("k0_Total"),
                       f("Vmax_Dihedral"), f("Vmin_Dihedral"), f("threshold_energy_Dihedral"), f("k0_Dihedral"))
        return cls(0.0, 0.0, 0.0, 0.0,
                   f("Vmax_Dihedral"), f("Vmin_Dihedral"), f("threshold_energy_Dihedral"), f("k0_Dihedral"),
                   has_total=False)

    @classmethod
    def from_json(cls, path) -> "PepGamdEnvelope":
        """Load from ``shared_gamd_setup_globals.json``.

        Every writer in ``production.py`` (the joint-envelope calibration path,
        the disabled/plain-MD path, and the worker-reuse path) nests the real
        CustomIntegrator globals under the top-level key ``"all_globals"``
        (with ``"interesting_globals"`` as a smaller, non-authoritative
        subset written alongside it) -- never under ``"globals"``,
        ``"integrator_globals"``, or ``"shared_gamd_globals_all"``. Probe the
        real key first; the others are kept harmlessly in case some other
        caller ever nests it differently.
        """
        doc = _json.loads(open(path).read())
        for cand in (doc, doc.get("all_globals"), doc.get("interesting_globals"),
                     doc.get("globals"), doc.get("integrator_globals"), doc.get("shared_gamd_globals_all")):
            if isinstance(cand, dict) and "k0_Dihedral" in cand:
                return cls.from_integrator_globals(cand)
        raise KeyError(f"{path}: no dict with k0_Dihedral/Vmax_Dihedral/... found")


def _channel_boost(v, e, vmax, vmin, k0):
    v = _np.asarray(v, dtype=float)
    rng = vmax - vmin
    scale = _np.maximum(_np.maximum(abs(e), _np.abs(v)), 1.0)
    b = 0.5 * k0 * (e - v) ** 2 / rng
    b = _np.where(_np.abs(rng) <= 0.001 * scale, 0.0, b)
    return _np.where((b + v) < e, b, 0.0)


def pep_gamd_boost_kj(v_pep_kj, v_dih_kj, lam, env: PepGamdEnvelope):
    """gamd-openmm's dependent dual boost under rung λ: dihedral first, then Total with the
    dihedral boost added to the Total energy before the square (stage_integrator
    _add_dihedral_boost_to_total_energy)."""
    lam = float(lam)
    b_dih = _channel_boost(v_dih_kj, env.threshold_dih, env.vmax_dih, env.vmin_dih, lam * env.k0max_dih)
    if not env.has_total:
        # Single dihedral boost: no Total channel exists, v_pep is ignored (it is NaN
        # on such runs by construction, see production._fetch_v_pep_v_dih).
        out = _np.asarray(b_dih, dtype=float)
        return float(out) if out.ndim == 0 else out
    b_tot = _channel_boost(_np.asarray(v_pep_kj, dtype=float) + b_dih, env.threshold_total, env.vmax_total, env.vmin_total, lam * env.k0max_total)
    out = b_dih + b_tot
    return float(out) if out.ndim == 0 else out


def pep_gamd_boost_matrix_kj(v_pep_kj, v_dih_kj, lambdas, env: PepGamdEnvelope) -> _np.ndarray:
    """(n_states, n_samples): boost of each sample's configuration under each state's λ."""
    v_pep = _np.asarray(v_pep_kj, dtype=float); v_dih = _np.asarray(v_dih_kj, dtype=float)
    return _np.vstack([_np.asarray(pep_gamd_boost_kj(v_pep, v_dih, float(l), env), dtype=float) for l in lambdas])


def k0max_from_globals(shared_globals: dict) -> dict:
    """Top-rung k0 per channel; ``Total`` is 0.0 for a single dihedral boost (no such global)."""
    return {"Total": float(shared_globals.get("k0_Total", 0.0)), "Dihedral": float(shared_globals["k0_Dihedral"])}


def set_replica_lambda(integrator, lam: float, k0max: dict) -> None:
    """Put a replica on rung λ: k0_c = λ·k0max_c for every channel the integrator has. Nothing else differs between rungs."""
    lam = float(lam)
    if not (0.0 <= lam <= 1.0):
        raise ValueError(f"gamd_lambda={lam} must lie in [0, 1]")
    try:
        names = {str(integrator.getGlobalVariableName(i)) for i in range(int(integrator.getNumGlobalVariables()))}
    except AttributeError:
        names = None  # minimal test doubles expose only setGlobalVariableByName: keep the two-channel behaviour
    integrator.setGlobalVariableByName("k0_Dihedral", lam * float(k0max["Dihedral"]))
    if names is None or "k0_Total" in names:
        integrator.setGlobalVariableByName("k0_Total", lam * float(k0max.get("Total", 0.0)))
    elif float(k0max.get("Total", 0.0)) != 0.0:
        raise ValueError("k0max['Total'] is non-zero but the integrator has no k0_Total global (single dihedral boost?)")


def set_replica_lambda_for_window(integrator, window_index, state_lambdas, k0max_by_channel) -> None:
    """Apply the rung λ for `window_index` to `integrator`, or no-op if the ladder is inactive.

    k0max_by_channel is None on every run that does not have the λ-ladder active
    (plain GaMD, conventional MD, or GaMD without a ladder -- the common case);
    on those runs state_lambdas is still an unconditionally-populated array, and
    this must be a silent no-op, not an error. Only when the ladder IS active
    (k0max_by_channel is not None) is a per-window λ required; state_lambdas is
    None in that combination only as a real misconfiguration, so that's the one
    case this raises on. One helper, one invariant, used at every call site that
    (re)applies a replica's window assignment so the guard can't be forgotten or
    mismatched at any individual site.
    """
    if k0max_by_channel is None:
        return
    if state_lambdas is None:
        raise ValueError(
            "set_replica_lambda_for_window: the λ-ladder is active (k0max_by_channel is set) "
            "but state_lambdas is None -- no per-window λ was supplied."
        )
    set_replica_lambda(integrator, float(state_lambdas[int(window_index)]), k0max_by_channel)
