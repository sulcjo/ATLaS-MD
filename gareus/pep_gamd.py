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


# Boost types the λ-ladder can scale per rung: the dependent dual Pep-GaMD boost
# (k0_Total and k0_Dihedral) and the stock single dihedral boost (k0_Dihedral only,
# no auxiliary force, 1x cost). Both use the lower-bound threshold E = Vmax, which
# is what pep_gamd_boost_kj reproduces in closed form; upper-bound variants use a
# different threshold rule and are deliberately excluded.
LADDER_BOOST_TYPES = frozenset({PEP_GAMD_BOOST_TYPE, "lower-dihedral"})


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


# ---------------------------------------------------------------------------
# NPT target adapters (design spec 2026-09-12, work package 2)
#
# U*_a(x, B) = U_phys(x, B) + W_a(x, B) + Delta_a(x, B): the acceptance energy
# for the application-controlled barostat (gareus/npt.py). Delta must reproduce
# the boost the integrator ACTUALLY applies -- its threshold formula, small-range
# and branch guards, dependent ordering and global names -- read fresh from the
# integrator at every trial (snapshot), with the k0 globals already carrying the
# replica's ladder rung (evaluating an envelope built from them at lambda = 1;
# applying lambda again is the double-application bug).


def _npt_lower_bound_channel_boost(e_channel, vmax, vmin, threshold, k0):
    """One channel of gamd-openmm's lower-bound dual boost.

    Reimplemented from the upstream kernels (vendored
    ``_gamd_reference/langevin/base_integrator.py::_add_gamd_pre_calc_step``):

        energy_scale   = max(|threshold|, |E|, 1)
        boost_threshold= 0.001 * energy_scale
        b              = select(step(|Vmax-Vmin| - boost_threshold),
                                 0.5*k0*(threshold-E)^2/(Vmax-Vmin), 0)
        b             *= step(threshold - (b + E))

    Deliberately NOT reusing ``_channel_boost`` above: that closed form is the
    independent oracle the NPT boost-agreement tests compare against (spec:
    never use one helper as both implementation and oracle).
    """
    energy_scale = max(abs(threshold), abs(e_channel), 1.0)
    if abs(vmax - vmin) <= 0.001 * energy_scale:
        return 0.0
    b = 0.5 * k0 * (threshold - e_channel) ** 2 / (vmax - vmin)
    if not (e_channel + b < threshold):
        return 0.0
    return b


def _npt_group_energy_kj(context, groups):
    if not groups:
        return 0.0
    from .imports import import_openmm
    _openmm, _app, unit = import_openmm()
    e = context.getState(getEnergy=True, groups=set(int(g) for g in groups)).getPotentialEnergy()
    return float(e.value_in_unit(unit.kilojoule_per_mole))


def _npt_force_roles(system):
    """Explicit force inventory: (index, class, force group, role).

    role is 'auxiliary' for the water-only measuring force, 'physical' for the
    classes the boost channels read, 'bias' for everything else (umbrella,
    secondary CV, restraints). ``total_energy_groups`` describes the integrator's
    boost channel, NOT total physical energy -- this inventory is the authority.
    """
    roles = []
    for i in range(system.getNumForces()):
        f = system.getForce(i)
        cls = f.__class__.__name__
        if f.getName() == AUX_FORCE_NAME:
            role = "auxiliary"
        elif cls in _DIHEDRAL_CLASSES or cls in _GROUP0_CLASSES:
            role = "physical"
        else:
            role = "bias"
        roles.append((i, cls, int(f.getForceGroup()), role))
    return roles


def _npt_split_groups(system):
    """(physical_groups, bias_groups, aux_groups) from the force inventory.

    A group that mixes physical- and bias-class forces cannot be split by
    energy; its whole contribution is counted as physical so the effective
    total stays exact (the split is a reporting concern, the sum is what the
    acceptance uses). The auxiliary force's group is reported separately.
    """
    physical, bias, aux = set(), set(), set()
    group_roles = {}
    for _i, _cls, g, role in _npt_force_roles(system):
        group_roles.setdefault(g, set()).add(role)
    for g, roles in group_roles.items():
        if "auxiliary" in roles:
            aux.add(g)
        elif "bias" in roles and "physical" not in roles:
            bias.add(g)
        else:
            physical.add(g)
    return physical, bias, aux


class _NptChannelParams:
    __slots__ = ("threshold_kj_mol", "vmax_kj_mol", "vmin_kj_mol", "k0")

    def __init__(self, threshold_kj_mol, vmax_kj_mol, vmin_kj_mol, k0):
        self.threshold_kj_mol = float(threshold_kj_mol)
        self.vmax_kj_mol = float(vmax_kj_mol)
        self.vmin_kj_mol = float(vmin_kj_mol)
        self.k0 = float(k0)


class _NptBoostSnapshot:
    """Immutable per-trial target parameters: stage plus both channels.

    Window (umbrella) parameters need no capture here: they live in the System's
    CustomForces and are read fresh from the Context by ``evaluate``. Only the
    integrator-held GaMD state could move between the two endpoint evaluations
    (calibration between moves), so that is what the snapshot freezes.
    """

    __slots__ = ("stage", "dihedral", "total")

    def __init__(self, stage, dihedral, total):
        self.stage = int(stage)
        self.dihedral = dihedral
        self.total = total


_CMD_STAGES = frozenset({-1, 1, 2})  # -1 = fresh context, not yet stepped
_BOOST_STAGES = frozenset({3, 4, 5})


def _npt_read_stage(integrator) -> int:
    try:
        return int(integrator.getGlobalVariableByName("stage"))
    except Exception as exc:
        raise ValueError(
            f"integrator {type(integrator).__name__} exposes no 'stage' global; "
            "it is not a gamd-openmm stage integrator"
        ) from exc


def _npt_read_channel(integrator, channel, names):
    missing = [f"{n}_{channel}" for n in names
               if not _npt_has_global(integrator, f"{n}_{channel}")]
    if missing:
        raise ValueError(
            f"integrator lacks the GaMD channel globals {missing} for the "
            f"'{channel}' channel; unsupported boost implementation for NPT"
        )
    return _NptChannelParams(
        integrator.getGlobalVariableByName(f"threshold_energy_{channel}"),
        integrator.getGlobalVariableByName(f"Vmax_{channel}"),
        integrator.getGlobalVariableByName(f"Vmin_{channel}"),
        integrator.getGlobalVariableByName(f"k0_{channel}"),
    )


def _npt_has_global(integrator, name) -> bool:
    try:
        n = int(integrator.getNumGlobalVariables())
        return any(integrator.getGlobalVariableName(i) == name for i in range(n))
    except AttributeError:
        return False


def _npt_boost_for_stage(boost_when_active, stage):
    if stage in _CMD_STAGES:
        return 0.0
    if stage in _BOOST_STAGES:
        return boost_when_active()
    raise RuntimeError(
        f"unsupported GaMD stage {stage}; NPT volume moves are only defined for "
        "stages 1-5 (conventional MD stages carry zero boost)"
    )


class PepGamdLowerDualNptTargetAdapter:
    """U* for Pep-GaMD lower-dual (``--gamd-boost-type pep-gamd-lower-dual``).

    Boost inputs, exactly as the integrator computes them:
      V_pep = E0 - E1 + E2 (physical groups minus the water-only auxiliary)
      V_d   = E2            (peptide dihedral group)
      b_d   = b(V_d)                        (Dihedral channel, evaluated first)
      b_t   = b(V_pep + b_d)                (Total channel sees b_d inside)
      Delta = b_d + b_t                     (dependent dual ordering)
    E1 is bookkeeping only: never added to the effective energy.
    """

    adapter_id = "pep-gamd-lower-dual"

    def __init__(self, system, integrator):
        idx, aux = find_aux_force(system)
        if aux is None:
            raise ValueError(
                "Pep-GaMD NPT adapter: the system carries no "
                f"{AUX_FORCE_NAME} auxiliary force; run ensure_pep_gamd_partition first"
            )
        if int(aux.getForceGroup()) != AUX_NONBONDED_GROUP:
            raise ValueError(
                "Pep-GaMD NPT adapter: the auxiliary force sits in force group "
                f"{aux.getForceGroup()}, expected {AUX_NONBONDED_GROUP}"
            )
        plus = getattr(integrator, "TOTAL_ENERGY_PLUS_GROUPS", None)
        minus = getattr(integrator, "TOTAL_ENERGY_MINUS_GROUPS", None)
        if (frozenset(plus or ()) != PHYSICAL_GROUPS
                or frozenset(minus or ()) != frozenset({AUX_NONBONDED_GROUP})):
            raise ValueError(
                "Pep-GaMD NPT adapter: the integrator is not the Pep-GaMD lower-dual "
                "integrator (its Total channel does not read energy0 - energy1 + energy2)"
            )
        dih = [g for g, name in (integrator.get_group_dict() or {}).items()
               if name == "Dihedral"]
        if len(dih) != 1 or int(dih[0]) != DIHEDRAL_GROUP:
            raise ValueError(
                "Pep-GaMD NPT adapter: expected exactly one 'Dihedral' channel group "
                f"== {DIHEDRAL_GROUP}; got {dict(integrator.get_group_dict() or {})!r}"
            )
        for name in ("stage", "k0_Dihedral", "k0_Total"):
            if not _npt_has_global(integrator, name):
                raise ValueError(
                    f"Pep-GaMD NPT adapter: integrator lacks the global {name!r}"
                )
        physical, bias, aux_groups = _npt_split_groups(system)
        stray = sorted({
            g for _i, _cls, g, role in _npt_force_roles(system)
            if role == "bias" and g in (PHYSICAL_NONBONDED_GROUP, AUX_NONBONDED_GROUP, DIHEDRAL_GROUP)
        })
        if stray:
            raise ValueError(
                "Pep-GaMD NPT adapter: bias-class forces sit in "
                + ", ".join(f"force group {g}" for g in stray)
                + ", which the boost channels read; every non-physical force (umbrella, "
                "secondary CV) must use a group outside 0..2"
            )
        self._physical_groups = frozenset(physical)
        self._bias_groups = frozenset(bias)
        self._aux_groups = frozenset(aux_groups)

    def snapshot(self, context, integrator):
        return _NptBoostSnapshot(
            _npt_read_stage(integrator),
            _npt_read_channel(integrator, "Dihedral", ("threshold_energy", "Vmax", "Vmin", "k0")),
            _npt_read_channel(integrator, "Total", ("threshold_energy", "Vmax", "Vmin", "k0")),
        )

    def evaluate(self, context, snapshot):
        from .npt import EnergyBreakdown

        e0 = _npt_group_energy_kj(context, {PHYSICAL_NONBONDED_GROUP})
        e1 = _npt_group_energy_kj(context, {AUX_NONBONDED_GROUP})
        e2 = _npt_group_energy_kj(context, {DIHEDRAL_GROUP})
        extra_physical = self._physical_groups - {PHYSICAL_NONBONDED_GROUP, DIHEDRAL_GROUP}
        physical = e0 + e2 + _npt_group_energy_kj(context, extra_physical)
        bias = _npt_group_energy_kj(context, self._bias_groups)

        def _boost():
            b_d = _npt_lower_bound_channel_boost(
                e2, snapshot.dihedral.vmax_kj_mol, snapshot.dihedral.vmin_kj_mol,
                snapshot.dihedral.threshold_kj_mol, snapshot.dihedral.k0)
            b_t = _npt_lower_bound_channel_boost(
                (e0 - e1 + e2) + b_d, snapshot.total.vmax_kj_mol, snapshot.total.vmin_kj_mol,
                snapshot.total.threshold_kj_mol, snapshot.total.k0)
            return b_d + b_t

        boost = _npt_boost_for_stage(_boost, snapshot.stage)
        return EnergyBreakdown(
            physical_kj_mol=physical, bias_kj_mol=bias, boost_kj_mol=boost,
            auxiliary_kj_mol=e1, effective_kj_mol=physical + bias + boost)


class LowerDihedralNptTargetAdapter:
    """U* for the stock single dihedral lower boost (``lower-dihedral``).

    No auxiliary force exists on this path; the boost is the Dihedral channel
    alone (BoostMethod.GROUPS: the group boost then the unscaled f0 update).
    """

    adapter_id = "lower-dihedral"

    def __init__(self, system, integrator):
        if find_aux_force(system)[1] is not None:
            raise ValueError(
                "lower-dihedral NPT adapter: the system carries a "
                f"{AUX_FORCE_NAME} force, which only Pep-GaMD systems have"
            )
        if _npt_has_global(integrator, "k0_Total"):
            raise ValueError(
                "lower-dihedral NPT adapter: the integrator has a Total channel "
                "(dual boost?); only the single dihedral boost is supported here"
            )
        dih = [g for g, name in (integrator.get_group_dict() or {}).items()
               if name == "Dihedral"]
        if len(dih) != 1:
            raise ValueError(
                "lower-dihedral NPT adapter: integrator has no single 'Dihedral' "
                f"channel group; got {dict(integrator.get_group_dict() or {})!r}"
            )
        self._dihedral_group = int(dih[0])
        for name in ("stage", "k0_Dihedral"):
            if not _npt_has_global(integrator, name):
                raise ValueError(
                    f"lower-dihedral NPT adapter: integrator lacks the global {name!r}"
                )
        physical, bias, _aux = _npt_split_groups(system)
        self._physical_groups = frozenset(physical)
        self._bias_groups = frozenset(bias)

    def snapshot(self, context, integrator):
        return _NptBoostSnapshot(
            _npt_read_stage(integrator),
            _npt_read_channel(integrator, "Dihedral", ("threshold_energy", "Vmax", "Vmin", "k0")),
            None,
        )

    def evaluate(self, context, snapshot):
        from .npt import EnergyBreakdown

        e_dih = _npt_group_energy_kj(context, {self._dihedral_group})
        rest = self._physical_groups - {self._dihedral_group}
        physical = e_dih + _npt_group_energy_kj(context, rest)
        bias = _npt_group_energy_kj(context, self._bias_groups)

        def _boost():
            return _npt_lower_bound_channel_boost(
                e_dih, snapshot.dihedral.vmax_kj_mol, snapshot.dihedral.vmin_kj_mol,
                snapshot.dihedral.threshold_kj_mol, snapshot.dihedral.k0)

        boost = _npt_boost_for_stage(_boost, snapshot.stage)
        return EnergyBreakdown(
            physical_kj_mol=physical, bias_kj_mol=bias, boost_kj_mol=boost,
            auxiliary_kj_mol=0.0, effective_kj_mol=physical + bias + boost)


class ConventionalNptTargetAdapter:
    """U* for conventional MD: no boost, and the auxiliary force (if a partitioned
    system is driven by a plain integrator, e.g. the cMD reconnaissance path)
    is excluded, matching that integrator's integration force groups."""

    adapter_id = "conventional"

    def __init__(self, system):
        physical, bias, aux = _npt_split_groups(system)
        self._physical_groups = frozenset(physical)
        self._bias_groups = frozenset(bias)
        self._aux_groups = frozenset(aux)

    def snapshot(self, context, integrator):
        return None  # no integrator-held target state on this path

    def evaluate(self, context, snapshot):
        from .npt import EnergyBreakdown

        physical = _npt_group_energy_kj(context, self._physical_groups)
        bias = _npt_group_energy_kj(context, self._bias_groups)
        aux = _npt_group_energy_kj(context, self._aux_groups)
        return EnergyBreakdown(
            physical_kj_mol=physical, bias_kj_mol=bias, boost_kj_mol=0.0,
            auxiliary_kj_mol=aux, effective_kj_mol=physical + bias)


def make_npt_target_adapter(system, integrator, args):
    """Build the EffectivePotentialAdapter for this run's boost implementation.

    Stage-aware dispatch over the boost types that have a validated NPT target:
    ``pep-gamd-lower-dual``, ``lower-dihedral`` and (zero boost, for reference
    comparisons) conventional MD. Every other boosted mode fails loudly.
    """
    boost_type = str(getattr(args, "gamd_boost_type", "") or "")
    run_mode = str(getattr(args, "run_mode", "gamd") or "gamd").strip().lower().replace("_", "-")
    if run_mode in ("cmd", "hmr-cmd"):
        return ConventionalNptTargetAdapter(system)
    if boost_type == PEP_GAMD_BOOST_TYPE:
        return PepGamdLowerDualNptTargetAdapter(system, integrator)
    if boost_type == "lower-dihedral":
        return LowerDihedralNptTargetAdapter(system, integrator)
    raise ValueError(
        f"--gamd-boost-type {boost_type!r} has no validated NPT target adapter "
        f"(supported: {PEP_GAMD_BOOST_TYPE}, lower-dihedral); refusing to run NPT "
        "with an acceptance energy that does not match the propagated dynamics"
    )
