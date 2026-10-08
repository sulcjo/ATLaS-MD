"""Ordered exchange-event ledger: schema, assignment checksums and replay (spec Section 7)."""
from __future__ import annotations

from typing import Sequence

from ..correctness._io import digest, json_bytes

EXCHANGE_EVENT_SCHEMA = "atlas-exchange-events-v1"
#: skip = a proposal the swap kernel declined to attempt (no holder / no outcome).
EVENT_KINDS = ("swap", "stay", "no_candidates", "skip")


def assignment_sha256(assignments: Sequence[int]) -> str:
    """Checksum of the replica -> window assignment list (index = replica)."""
    return digest(json_bytes([int(x) for x in assignments]))
