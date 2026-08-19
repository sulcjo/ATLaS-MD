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


def restraint_sigma(k_kcal_per_a2: float, temperature_k: float) -> float:
    """Gaussian width of a harmonic umbrella, in the CV's own units.

    ``sigma = sqrt(k_B T / k)`` with ``k`` converted from kcal/mol/A^2 to
    kJ/mol/A^2 so it divides a kJ/mol thermal energy.
    """
    k = float(k_kcal_per_a2) * KJ_PER_KCAL
    if not math.isfinite(k) or k <= 0.0:
        return math.inf
    return math.sqrt(K_B_KJ_PER_MOL_K * float(temperature_k) / k)


def rank_windows(
    *,
    n_windows: int,
    centers_a: Sequence[float],
    k_list: Sequence[float],
    acceptance_by_window: Mapping[int, float],
    overlap_by_pair: Mapping[tuple[int, int], float],
    delta_by_window: Mapping[int, float],
    temperature_k: float,
) -> tuple[WindowStatus, ...]:
    """Rank windows worst-first. Ties break by window index for stability."""
    low_overlap: dict[int, float] = {}
    for (a, b), ov in overlap_by_pair.items():
        if math.isfinite(ov) and ov < DEAD_OVERLAP:
            for w in (a, b):
                low_overlap[w] = min(low_overlap.get(w, math.inf), float(ov))

    out: list[WindowStatus] = []
    for w in range(int(n_windows)):
        acc = float(acceptance_by_window.get(w, float("nan")))
        delta = abs(float(delta_by_window.get(w, 0.0)))
        k = float(k_list[w]) if w < len(k_list) else float("nan")
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
    "PINNED_SIGMA_MULTIPLE", "WARN", "WindowStatus", "rank_windows", "restraint_sigma",
]
