"""Application-controlled NPT volume moves for boosted simulations.

OpenMM's native ``MonteCarloBarostat`` evaluates acceptance from the Context's
ordinary potential energy, which is computed over the integrator's integration
force groups. Under Pep-GaMD that is the wrong energy twice over:

* the applied boost is constructed *inside* the integrator, which combines force
  groups itself, so the Context potential does **not** contain it; and
* the auxiliary water-only nonbonded force added by
  ``ensure_pep_gamd_partition`` **is** in the System, so it enters acceptance as
  if it were physical energy.

Boosted NPT therefore samples the wrong volume distribution. Note that lambda = 0
is affected too: the boost vanishes there, but the auxiliary term does not.

This module supplies the corrected move. Its acceptance energy is

    U*_a(x, B) = U_phys(x, B) + W_a(x, B) + Delta_a(x, B)

where ``W`` is every unscaled bias (umbrella, secondary CV, restraints) and
``Delta`` is the boost the integrator actually applies. Auxiliary bookkeeping
forces contribute only to constructing boost inputs, never as physical energy.

Design contract is frozen here so the controller internals and the production
scheduler can be built independently. See
``docs/superpowers/specs/2026-09-12-npt-correction-design.md``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Protocol

__all__ = [
    "BarostatBackend",
    "EnergyBreakdown",
    "VolumeMoveResult",
    "EffectivePotentialAdapter",
    "BiasedMCBarostatController",
    "resolve_npt_backend",
    "count_native_barostats",
    "BAR_NM3_TO_KJ_PER_MOL",
]

# 1 bar * nm^3 expressed in kJ/mol. Fixed by the spec; do not re-derive inline.
BAR_NM3_TO_KJ_PER_MOL = 0.0602214076

# Molar gas constant in kJ/(mol K) (CODATA 2018). beta = 1/(R T) with molar
# energies throughout this module.
_R_KJ_PER_MOL_K = 8.31446261815324e-3

BarostatBackend = Literal["none", "native", "biased_mc"]

# Boost types for which a validated U* target adapter exists (Package A):
# the dependent dual Pep-GaMD boost and the stock single dihedral lower boost.
# Kept in sync with gareus.pep_gamd.LADDER_BOOST_TYPES on purpose -- tests
# assert the equality so drift fails loudly.
SUPPORTED_BIASED_MC_BOOST_TYPES = frozenset({"pep-gamd-lower-dual", "lower-dihedral"})

_ENSEMBLES = frozenset({"npt", "nvt"})
_BACKEND_REQUESTS = frozenset({"auto", "native", "biased_mc"})
_RUN_MODES = frozenset({"cmd", "hmr-cmd", "gamd", "hmr-gamd"})
_BOOSTED_RUN_MODES = frozenset({"gamd", "hmr-gamd"})

# MonteCarloBarostat-family force class names. Matched by name (not isinstance)
# so a future OpenMM release adding another MonteCarlo*Barostat is still caught
# by the "MonteCarlo...Barostat" prefix rule below.
_NATIVE_BAROSTAT_CLASSES = frozenset({
    "MonteCarloBarostat",
    "MonteCarloAnisotropicBarostat",
    "MonteCarloMembraneBarostat",
    "MonteCarloFlexibleBarostat",
})


def _normalise(value: object) -> str:
    return str(value or "").strip().lower()


def _normalise_run_mode(value: object) -> str:
    return _normalise(value).replace("_", "-")


def resolve_npt_backend(
    *,
    ensemble: str,
    requested: str,
    run_mode: str,
    boost_type: str,
) -> BarostatBackend:
    """Choose the volume controller, or fail loudly.

    Explicit ``native`` with boosted dynamics must raise rather than preserve the
    known mismatch, and an unsupported boosted mode must raise rather than
    silently downgrade the requested ensemble to NVT.
    """
    ensemble_n = _normalise(ensemble)
    requested_n = _normalise(requested)
    run_mode_n = _normalise_run_mode(run_mode)
    boost_type_n = _normalise(boost_type)
    if ensemble_n not in _ENSEMBLES:
        raise ValueError(f"unknown ensemble {ensemble!r}; expected one of npt/nvt")
    if requested_n not in _BACKEND_REQUESTS:
        raise ValueError(f"unknown npt_barostat_backend {requested!r}; expected one of auto/native/biased_mc")
    if run_mode_n not in _RUN_MODES:
        raise ValueError(f"unknown run_mode {run_mode!r}; expected one of cmd/hmr-cmd/gamd/hmr-gamd")

    boosted = run_mode_n in _BOOSTED_RUN_MODES
    if ensemble_n == "nvt":
        if requested_n != "auto":
            raise ValueError(
                f"nvt ensemble cannot use npt_barostat_backend={requested_n!r}: "
                "a volume controller contradicts the requested NVT ensemble"
            )
        return "none"

    if requested_n == "native":
        if boosted:
            raise ValueError(
                f"npt_barostat_backend=native is invalid for boosted run_mode={run_mode_n!r} "
                f"(boost_type={boost_type_n!r}): the native MonteCarloBarostat accepts volume "
                "moves with physical energy plus the auxiliary force and without the GaMD "
                "boost, i.e. the wrong target distribution. Use biased_mc."
            )
        return "native"

    if requested_n == "biased_mc":
        # Explicit biased_mc is allowed with conventional MD too: it is the
        # zero-boost reference comparison path.
        return "biased_mc"

    # auto
    if boosted:
        if not boost_type_n:
            raise ValueError(
                f"boosted run_mode={run_mode_n!r} carries no gamd_boost_type; "
                "auto cannot pick a volume controller for it"
            )
        if boost_type_n not in SUPPORTED_BIASED_MC_BOOST_TYPES:
            raise ValueError(
                f"NPT is requested with boosted mode {boost_type_n!r}, which has no validated "
                "target adapter for the application-controlled barostat (supported boost types: "
                f"{sorted(SUPPORTED_BIASED_MC_BOOST_TYPES)}). Refusing to silently downgrade the "
                "ensemble to NVT or to keep the incorrect native-barostat acceptance energy."
            )
        return "biased_mc"
    return "native"


def count_native_barostats(system: Any) -> int:
    """Number of MonteCarloBarostat-family forces in ``system``.

    Used to assert exactly one volume controller exists; the native barostat must
    be removed from application-controlled Systems *before* Context creation.
    """
    n = 0
    for i in range(system.getNumForces()):
        f = system.getForce(i)
        name = f.__class__.__name__
        if name in _NATIVE_BAROSTAT_CLASSES or (
            name.startswith("MonteCarlo") and "Barostat" in name
        ):
            n += 1
    return n


@dataclass(frozen=True)
class EnergyBreakdown:
    """Components of U* for one configuration.

    Kept separate rather than summed so reporters can distinguish physical U,
    boost, umbrella and effective U* -- a generic StateDataReporter's raw Context
    potential must never be presented as either physical or effective energy
    while the auxiliary force exists.
    """

    physical_kj_mol: float
    bias_kj_mol: float
    boost_kj_mol: float
    auxiliary_kj_mol: float
    effective_kj_mol: float


@dataclass(frozen=True)
class VolumeMoveResult:
    step: int
    accepted: bool
    reason: str
    old_volume_nm3: float
    proposed_volume_nm3: float
    log_acceptance: float
    current_energy: EnergyBreakdown


class EffectivePotentialAdapter(Protocol):
    """Evaluates U* for a supported boost implementation and stage.

    ``snapshot`` captures stage, channel parameters and window parameters once
    per trial; the *same* snapshot must evaluate both endpoints, or the move is
    accepting against a moving target.
    """

    adapter_id: str

    def snapshot(self, context: Any, integrator: Any) -> object: ...

    def evaluate(self, context: Any, snapshot: object) -> EnergyBreakdown: ...


class BiasedMCBarostatController:
    """Per-replica isotropic Metropolis volume move against U*.

    Owned by the replica's context-owning worker. A trial never steps MD, never
    updates calibration statistics, never rethermalizes velocities and never
    changes labels.
    """

    @classmethod
    def initialize(
        cls,
        context: Any,
        adapter: EffectivePotentialAdapter,
        *,
        pressure_bar: float,
        temperature_k: float,
        frequency_steps: int,
        volume_step_fraction: float,
        seed: int,
    ) -> "BiasedMCBarostatController":
        """``volume_step_fraction`` is converted once, against the *starting*
        volume, into a fixed absolute half-width in nm^3. It must not silently
        become a fraction of the current volume later in the run."""
        raise NotImplementedError("package 1")

    @classmethod
    def restore(
        cls,
        context: Any,
        adapter: EffectivePotentialAdapter,
        *,
        state: Mapping[str, object],
        expected_pressure_bar: float,
        expected_temperature_k: float,
    ) -> "BiasedMCBarostatController":
        """Restore exact schedule and random stream. Must not attempt an extra
        move as a side effect of resuming."""
        raise NotImplementedError("package 1")

    @property
    def next_due_step(self) -> int:
        raise NotImplementedError("package 1")

    def steps_until_due(self, current_step: int) -> int:
        raise NotImplementedError("package 1")

    def attempt_due(self, current_step: int) -> VolumeMoveResult:
        """Run one trial transaction. Rejection consumes the random draws but
        leaves physical state, MD time and integration step count unchanged."""
        raise NotImplementedError("package 1")

    def state_dict(self) -> dict[str, object]:
        """Serializable controller state for the checkpoint manifest: backend,
        molecule-partition fingerprint, RNG algorithm and state, fixed width,
        counters, last/next due step and schema version."""
        raise NotImplementedError("package 1")
