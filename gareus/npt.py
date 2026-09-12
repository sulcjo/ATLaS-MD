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

from dataclasses import dataclass
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

BarostatBackend = Literal["none", "native", "biased_mc"]


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
    raise NotImplementedError("package 1")


def count_native_barostats(system: Any) -> int:
    """Number of MonteCarloBarostat-family forces in ``system``.

    Used to assert exactly one volume controller exists; the native barostat must
    be removed from application-controlled Systems *before* Context creation.
    """
    raise NotImplementedError("package 1")


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
