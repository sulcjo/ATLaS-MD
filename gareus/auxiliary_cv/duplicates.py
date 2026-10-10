"""Identical-Hamiltonian merge utility (library; Stage D pilot analysis, not wired)."""
from __future__ import annotations

import numpy as np

from gareus.correctness._io import IntegrityError


def merge_identical_hamiltonians(u_nk, origins, hamiltonian_ids):
    """Library utility (Stage D pilot analysis); MVP pools shams as separate origins instead."""
    u = np.asarray(u_nk, dtype=np.float64)
    origins = np.asarray(origins, dtype=np.int64)
    ids = [str(x) for x in hamiltonian_ids]
    if u.ndim != 2 or u.shape[1] != len(ids):
        raise IntegrityError(f"u_nk has {u.shape[1] if u.ndim == 2 else '?'} columns for {len(ids)} Hamiltonian ids")
    groups: list[list[int]] = []
    where: dict[str, int] = {}
    for col, hid in enumerate(ids):
        if hid not in where:
            where[hid] = len(groups)
            groups.append([col])
        else:
            first = groups[where[hid]][0]
            if not np.array_equal(u[:, first], u[:, col], equal_nan=True):
                raise IntegrityError(f"columns {first} and {col} share Hamiltonian {hid} but are not bitwise equal")
            groups[where[hid]].append(col)
    bad = origins[(origins < 0) | (origins >= len(ids))]
    if bad.size:
        raise IntegrityError(f"sample origin(s) {sorted(set(bad.tolist()))[:8]} are not columns 0..{len(ids) - 1}")
    merged = u[:, [g[0] for g in groups]].copy()
    col_to_group = {c: gi for gi, g in enumerate(groups) for c in g}
    merged_origins = np.asarray([col_to_group[int(o)] for o in origins], dtype=np.int64)
    return merged, merged_origins, groups
