"""Pure sampling-policy guard; no coordinate mutation or RNG consumption."""
from __future__ import annotations
from dataclasses import dataclass
from ._io import IntegrityError


@dataclass(frozen=True)
class SamplingPolicy:
    phase_kind: str
    equilibrium_analysis_eligible: bool
    rescue_enabled: bool
    schema_version: int = 1


def _bool_or_none(value, name):
    if value is None or type(value) is bool:
        return value
    raise IntegrityError(f"{name} must be a parsed boolean or None, not {value!r}")


def resolve_sampling_policy(args) -> SamplingPolicy:
    phase = getattr(args, "sampling_phase", "production")
    if phase not in {"production", "exploration", "pilot", "equilibration"}:
        raise IntegrityError(f"Unknown explicit sampling_phase: {phase!r}")
    requested = _bool_or_none(getattr(args, "cv1_stuck_reseed", None), "cv1_stuck_reseed")
    override = _bool_or_none(getattr(args, "rescue_in_final_production", False), "rescue_in_final_production")
    eligible = phase == "production"
    rescue = requested is True
    if eligible and (rescue or override is True):
        raise IntegrityError(
            "Configuration rescue is forbidden in equilibrium production; use excluded "
            "exploration, then separate equilibration and a new production segment."
        )
    if override is True:
        raise IntegrityError("rescue_in_final_production is obsolete; it cannot authorize intervention")
    return SamplingPolicy(phase, eligible, rescue)


def require_rescue_disabled_in_current_driver(args) -> SamplingPolicy:
    """Conservative first-release guard until exploratory driver wiring exists.

    The integration patch deliberately disables rescue for *all* run_gareus calls,
    including pilots, rather than pretending the legacy rescue branch is safe.
    """
    policy = resolve_sampling_policy(args)
    if policy.rescue_enabled:
        raise IntegrityError("Exploration rescue needs explicit excluded-segment wiring; disabled in this core patch")
    return policy
