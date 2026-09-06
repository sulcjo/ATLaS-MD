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
"""
from __future__ import annotations

from typing import Iterable

PEP_GAMD_PREFIX = "pep-gamd-"
PEP_GAMD_BOOST_TYPE = PEP_GAMD_PREFIX + "lower-dual"
AUX_FORCE_NAME = "PepGaMDWaterOnlyNonbonded"

PHYSICAL_NONBONDED_GROUP = 0
AUX_NONBONDED_GROUP = 1
DIHEDRAL_GROUP = 2
PHYSICAL_GROUPS = frozenset({PHYSICAL_NONBONDED_GROUP, DIHEDRAL_GROUP})

_DIHEDRAL_CLASSES = ("PeriodicTorsionForce", "CMAPTorsionForce")
_GROUP0_CLASSES = (
    "NonbondedForce", "HarmonicBondForce", "HarmonicAngleForce",
    "CMMotionRemover", "MonteCarloBarostat", "MonteCarloAnisotropicBarostat",
    "MonteCarloMembraneBarostat",
)


def find_aux_force(system):
    for i in range(system.getNumForces()):
        f = system.getForce(i)
        if f.getName() == AUX_FORCE_NAME:
            return i, f
    return None, None


def _physical_nonbonded(system, openmm):
    found = [
        (i, system.getForce(i)) for i in range(system.getNumForces())
        if isinstance(system.getForce(i), openmm.NonbondedForce)
        and system.getForce(i).getName() != AUX_FORCE_NAME
    ]
    if len(found) != 1:
        raise ValueError(f"Pep-GaMD needs exactly one physical NonbondedForce; found {len(found)}")
    return found[0]


def assign_pep_gamd_force_groups(system) -> None:
    """Physical forces -> groups 0/2; refuse any other force parked in 0..2."""
    for i in range(system.getNumForces()):
        f = system.getForce(i)
        name = f.__class__.__name__
        if f.getName() == AUX_FORCE_NAME:
            f.setForceGroup(AUX_NONBONDED_GROUP)
        elif name in _DIHEDRAL_CLASSES:
            f.setForceGroup(DIHEDRAL_GROUP)
        elif name in _GROUP0_CLASSES:
            f.setForceGroup(PHYSICAL_NONBONDED_GROUP)
        elif f.getForceGroup() in (PHYSICAL_NONBONDED_GROUP, AUX_NONBONDED_GROUP, DIHEDRAL_GROUP):
            raise ValueError(
                f"{name} sits in force group {f.getForceGroup()}, which Pep-GaMD boosts; "
                f"non-physical forces (umbrella, secondary CV) must use a group outside 0..2"
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


def ensure_pep_gamd_partition(system, peptide_atoms: Iterable[int]) -> int:
    """Add the water-only auxiliary NonbondedForce and assign force groups.

    Idempotent: a second call finds the existing auxiliary force by name and only
    re-asserts the group layout. Returns the auxiliary force's index.
    """
    from .imports import import_openmm
    openmm, _app, unit = import_openmm()

    idx, aux = find_aux_force(system)
    if aux is not None:
        assign_pep_gamd_force_groups(system)
        return idx

    pep = set(int(i) for i in peptide_atoms)
    if not pep:
        raise ValueError("Pep-GaMD partition needs a non-empty peptide atom set")
    _phys_idx, physical = _physical_nonbonded(system, openmm)
    alpha, nx, ny, nz = _pin_pme_parameters(system, physical, openmm, unit)

    aux = openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(physical))
    aux.setName(AUX_FORCE_NAME)
    aux.setPMEParameters(alpha, nx, ny, nz)
    for i in range(aux.getNumParticles()):
        if i in pep:
            _q, sig, _eps = aux.getParticleParameters(i)
            aux.setParticleParameters(i, 0.0, sig, 0.0)
    for k in range(aux.getNumExceptions()):
        a, b, _qq, sig, _eps = aux.getExceptionParameters(k)
        if a in pep or b in pep:
            aux.setExceptionParameters(k, a, b, 0.0, sig, 0.0)
    idx = system.addForce(aux)
    assign_pep_gamd_force_groups(system)
    return idx


def peptide_essential_energy_kj(context, unit) -> float:
    e_phys = context.getState(getEnergy=True, groups=set(PHYSICAL_GROUPS)).getPotentialEnergy()
    e_aux = context.getState(getEnergy=True, groups={AUX_NONBONDED_GROUP}).getPotentialEnergy()
    return float((e_phys - e_aux).value_in_unit(unit.kilojoule_per_mole))


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


def _build_integrator_class():
    from gamd.langevin.dual_boost_integrators import LowerBoundIntegrator
    from gamd.stage_integrator import BoostType

    class PepGaMDLowerDualIntegrator(LowerBoundIntegrator):
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

        def _dihedral_group_id(self) -> int:
            (gid,) = [g for g, name in self.get_group_dict().items() if name == "Dihedral"]
            return int(gid)

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

    return PepGaMDLowerDualIntegrator


def __getattr__(name):
    if name == "PepGaMDLowerDualIntegrator":
        return _integrator_class()
    raise AttributeError(name)


def is_pep_gamd(args) -> bool:
    return str(getattr(args, "gamd_boost_type", "") or "") == PEP_GAMD_BOOST_TYPE


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
    """Force groups that constitute the physical potential: everything but the auxiliary."""
    if is_pep_gamd(args):
        return set(range(32)) - {AUX_NONBONDED_GROUP}
    return -1


def physical_potential_energy_kj(context, system, unit) -> float:
    groups = set(range(32))
    if find_aux_force(system)[1] is not None:
        groups -= {AUX_NONBONDED_GROUP}
    return float(context.getState(getEnergy=True, groups=groups).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))


def total_energy_groups_for_args(args) -> tuple[frozenset, frozenset]:
    """The Total channel's (plus, minus) groups as a property of the boost type.

    The cMD-kind recon steps a plain Langevin integrator on an already-partitioned
    system, so the stepping integrator cannot be the source of this definition.
    """
    if is_pep_gamd(args):
        return frozenset(PHYSICAL_GROUPS), frozenset({AUX_NONBONDED_GROUP})
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


def _integrator_class():
    cls = globals().get("PepGaMDLowerDualIntegrator")
    if cls is None:
        cls = _build_integrator_class()
        globals()["PepGaMDLowerDualIntegrator"] = cls
    return cls


def build_pep_gamd_integrator(system, args, unit) -> list:
    """Partition `system` and build the Pep-GaMD integrator.

    Returns ``[aux_group, dihedral_group, integrator]`` so ``result[2]`` is the
    integrator, matching gamd-openmm's factory convention.
    """
    atoms = getattr(args, "pep_gamd_peptide_atoms", None)
    if not atoms:
        raise ValueError(
            f"--gamd-boost-type {PEP_GAMD_BOOST_TYPE} needs args.pep_gamd_peptide_atoms; "
            "call prepare_pep_gamd_args(args, topology) after the system is built"
        )
    ensure_pep_gamd_partition(system, atoms)
    total_steps = (
        int(args.gamd_cmd_prep_steps) + int(args.gamd_cmd_steps)
        + int(args.gamd_equil_prep_steps) + int(args.gamd_equil_steps)
        + int(args.gamd_production_steps)
    )
    integrator = _integrator_class()(
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
    return [AUX_NONBONDED_GROUP, DIHEDRAL_GROUP, integrator]


from dataclasses import dataclass
import json as _json
import numpy as _np


@dataclass(frozen=True)
class PepGamdEnvelope:
    """Frozen per-channel GaMD envelope; k0max_* is the top rung (λ = 1)."""
    vmax_total: float; vmin_total: float; threshold_total: float; k0max_total: float
    vmax_dih: float;   vmin_dih: float;   threshold_dih: float;   k0max_dih: float

    @classmethod
    def from_integrator_globals(cls, g: dict) -> "PepGamdEnvelope":
        f = lambda k: float(g[k])
        return cls(f("Vmax_Total"), f("Vmin_Total"), f("threshold_energy_Total"), f("k0_Total"),
                   f("Vmax_Dihedral"), f("Vmin_Dihedral"), f("threshold_energy_Dihedral"), f("k0_Dihedral"))

    @classmethod
    def from_json(cls, path) -> "PepGamdEnvelope":
        doc = _json.loads(open(path).read())
        for cand in (doc, doc.get("globals"), doc.get("integrator_globals"), doc.get("shared_gamd_globals_all")):
            if isinstance(cand, dict) and "k0_Total" in cand:
                return cls.from_integrator_globals(cand)
        raise KeyError(f"{path}: no dict with k0_Total/Vmax_Total/... found")


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
    b_tot = _channel_boost(_np.asarray(v_pep_kj, dtype=float) + b_dih, env.threshold_total, env.vmax_total, env.vmin_total, lam * env.k0max_total)
    out = b_dih + b_tot
    return float(out) if out.ndim == 0 else out


def pep_gamd_boost_matrix_kj(v_pep_kj, v_dih_kj, lambdas, env: PepGamdEnvelope) -> _np.ndarray:
    """(n_states, n_samples): boost of each sample's configuration under each state's λ."""
    v_pep = _np.asarray(v_pep_kj, dtype=float); v_dih = _np.asarray(v_dih_kj, dtype=float)
    return _np.vstack([_np.asarray(pep_gamd_boost_kj(v_pep, v_dih, float(l), env), dtype=float) for l in lambdas])
