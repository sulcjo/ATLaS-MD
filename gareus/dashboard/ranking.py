"""Severity ranking for window-facing panels.

Positional truncation is only acceptable if position encodes severity, so every
list panel sorts through here first. Hysteresis keeps a flagged window in its
slot for a few frames after it recovers: a table that re-sorts on noise moves
the row out from under whoever is reading it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from ..units import KJ_PER_KCAL, K_B_KJ_PER_MOL_K

OK = "ok"
WARN = "WARN"
BAD = "BAD"

DEAD_ACCEPTANCE = 0.02

# How many attempts a pair needs before a zero acceptance is allowed to condemn
# its windows. Derived, not chosen: by the rule of three, observing 0 successes
# in n trials puts the 95% upper bound on the true rate at 3/n, so n = 3/
# DEAD_ACCEPTANCE is the point where "never accepted" first means "really below
# the dead threshold" rather than "we have barely looked".
#
# Without this, a sparse 2D exchange graph condemns most of its own windows: on
# a real 32-window run (chignolin_6, 2026-08-29) 16 of 140 pairs had
# `attempts: 1, accepted: 0` -- long-range edges the Gibbs walk proposed once
# and rejected once -- and because `acceptance_by_window` takes each window's
# WORST pair, that single 0.0 flagged 15 of 32 windows `BAD dead exchange
# 0.000` while global acceptance was 3858/4428 = 0.871. Same class as the
# fabricated-zero the `delta` reading below is already guarded against.
MIN_ATTEMPTS_FOR_DEAD = math.ceil(3.0 / DEAD_ACCEPTANCE)
LOW_ACCEPTANCE = 0.15
DEAD_OVERLAP = 0.10
PINNED_SIGMA_MULTIPLE = 2.0

_SEV_BAD_DEAD = 40
_SEV_BAD_OVERLAP = 30
_SEV_BAD_PINNED = 20
_SEV_WARN_ACCEPT = 10
_SEV_OK = 0


@dataclass(frozen=True)
class WindowStatus:
    window: int
    severity: int
    status: str
    reasons: tuple[str, ...] = ()


def _as_float(value: object, default: float) -> float:
    """Coerce a measurement to float, treating anything unusable as missing.

    Callers hand these maps in from live sampling, and "not measured yet" gets
    encoded as `None` at least as often as `nan`. Guarding only against `nan`
    means one `None` raises `TypeError` and takes down the ranking pass for
    *every* window, not just the one with the bad value.
    """
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float(default)
    return out


def restraint_sigma(k_kcal_per_a2: float, temperature_k: float) -> float:
    """Gaussian width of a harmonic umbrella, in the CV's own units.

    ``sigma = sqrt(k_B T / k)`` with ``k`` converted from kcal/mol/A^2 to
    kJ/mol/A^2 so it divides a kJ/mol thermal energy. Anything unusable -- a
    non-positive or non-finite ``k``, a non-positive temperature -- yields
    ``inf``, which makes the pinned-window comparison unsatisfiable rather than
    raising or flagging spuriously.
    """
    k = _as_float(k_kcal_per_a2, float("nan")) * KJ_PER_KCAL
    temperature = _as_float(temperature_k, float("nan"))
    if not math.isfinite(k) or k <= 0.0:
        return math.inf
    if not math.isfinite(temperature) or temperature <= 0.0:
        return math.inf
    return math.sqrt(K_B_KJ_PER_MOL_K * temperature / k)


def rank_windows(
    *,
    n_windows: int,
    centers_a: Sequence[float],   # accepted but unused today; a later phase's
                                  # reason strings will quote the centre

    k_list: Sequence[float],
    acceptance_by_window: Mapping[int, float],
    overlap_by_pair: Mapping[tuple[int, int], float],
    delta_by_window: Mapping[int, float],
    temperature_k: float,
    overlap_by_window: "Mapping[int, float] | None" = None,
) -> tuple[WindowStatus, ...]:
    """Rank windows worst-first. Ties break by window index for stability.

    ``overlap_by_window`` (2D layouts) gives each window's best neighbour
    overlap directly; when it is set, a window is flagged only if it is isolated.
    Without it (1D ladders), either side of a dead (w, w+1) pair is flagged,
    since in a chain every gap disconnects the ladder.
    """
    low_overlap: dict[int, float] = {}
    if overlap_by_window:
        for w, raw_ov in overlap_by_window.items():
            ov = _as_float(raw_ov, float("nan"))
            if math.isfinite(ov) and ov < DEAD_OVERLAP:
                low_overlap[int(w)] = float(ov)
    else:
        for (a, b), raw_ov in overlap_by_pair.items():
            ov = _as_float(raw_ov, float("nan"))
            if math.isfinite(ov) and ov < DEAD_OVERLAP:
                for w in (a, b):
                    low_overlap[w] = min(low_overlap.get(w, math.inf), float(ov))

    out: list[WindowStatus] = []
    for w in range(int(n_windows)):
        acc = _as_float(acceptance_by_window.get(w), float("nan"))
        # `nan`, not 0.0: a missing delta is unmeasured, and defaulting it to a
        # valid "exactly on target" reading is the same fabricated-zero-deviation
        # mistake this codebase already had to fix once in bias reconstruction.
        delta = abs(_as_float(delta_by_window.get(w), float("nan")))
        k = _as_float(k_list[w], float("nan")) if w < len(k_list) else float("nan")
        sigma = restraint_sigma(k, temperature_k)

        if math.isfinite(acc) and acc < DEAD_ACCEPTANCE:
            out.append(WindowStatus(w, _SEV_BAD_DEAD, BAD, (f"dead exchange {acc:.3f}",)))
        elif w in low_overlap:
            out.append(WindowStatus(w, _SEV_BAD_OVERLAP, BAD,
                                    (f"overlap {low_overlap[w]:.2f}",)))
        elif math.isfinite(sigma) and delta > PINNED_SIGMA_MULTIPLE * sigma:
            out.append(WindowStatus(w, _SEV_BAD_PINNED, BAD,
                                    (f"pinned |d|={delta:.2f} > 2s={2 * sigma:.2f}",)))
        elif math.isfinite(acc) and acc < LOW_ACCEPTANCE:
            out.append(WindowStatus(w, _SEV_WARN_ACCEPT, WARN, (f"low accept {acc:.2f}",)))
        else:
            out.append(WindowStatus(w, _SEV_OK, OK, ()))
    return tuple(sorted(out, key=lambda s: (-s.severity, s.window)))


def rank_windows_for(ctx) -> tuple[WindowStatus, ...]:
    """`rank_windows` over a DashboardContext -- the one call every panel shares."""
    return rank_windows(
        n_windows=ctx.n_windows, centers_a=ctx.centers_a, k_list=ctx.k_list,
        acceptance_by_window=ctx.acceptance_windows, overlap_by_pair=ctx.overlap_pairs,
        delta_by_window=ctx.deltas, temperature_k=ctx.temperature_k,
        overlap_by_window=getattr(ctx, "overlap_windows", None),
    )


class Hysteresis:
    """Hold a key in its ranked slot until it has been clear for `frames` frames."""

    def __init__(self, frames: int = 5) -> None:
        self.frames = max(1, int(frames))
        self._countdown: dict[str, int] = {}

    def update(self, bad_keys: Iterable[str]) -> frozenset[str]:
        live = {str(k) for k in bad_keys}
        for key in live:
            self._countdown[key] = self.frames
        for key in list(self._countdown):
            if key not in live:
                self._countdown[key] -= 1
                if self._countdown[key] <= 0:
                    del self._countdown[key]
        return frozenset(self._countdown)


__all__ = [
    "BAD", "DEAD_ACCEPTANCE", "DEAD_OVERLAP", "Hysteresis", "LOW_ACCEPTANCE", "OK",
    "PINNED_SIGMA_MULTIPLE", "WARN", "WindowStatus", "rank_windows", "rank_windows_for",
    "restraint_sigma",
]
