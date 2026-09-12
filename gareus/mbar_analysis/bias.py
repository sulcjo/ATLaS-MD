"""Bias-reconstruction helpers relocated from analyze_gareus_mbar.py.

_compute_u_nk_analytical and _reconstruct_union_bias_block delegate to the
canonical gareus.query.reconstruct_bias_matrix (fixed as part of this same
modularization plan -- see Plan A3's Task 1) rather than duplicating its
formula a third and fourth time. Their own guard logic was already correct
(a prior audit this session fixed both -- see
tests/test_bias_reconstruction_nan_handling.py), so this relocation is a
pure deduplication with zero intended behavior change.

_parse_epoch_window_map_native_params and _epoch_bias_param_vectors are pure
supporting glue for load_parquet_adaptive_union's per-epoch-native bias
reconstruction (see CLAUDE.md's "Union-MBAR bias matrix used a stale global
registry snapshot" entry) and have no formula of their own to reconcile:
the first parses one epoch's own epoch_window_map.csv rows into a
{state_id: {...}} native-params dict, the second consumes exactly that dict.
They are always used together, which is why both live here.

_is_usable_for_mbar / _merge_missing_usable_states deliberately do NOT live
here -- Plan A2 relocates them into
gareus/mbar_analysis/loaders_union_parquet.py, the only cluster that calls
them (see Plan A3's Global Constraints CORRECTION).
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from gareus import query

# Module-level `from gareus import query` (not `from gareus.query import
# reconstruct_bias_matrix`) so `monkeypatch.setattr(gareus.query,
# "reconstruct_bias_matrix", fake)` in a test actually intercepts calls made
# from this module too -- a plain name-import would bind a local reference
# at import time that a later monkeypatch on gareus.query's own attribute
# would not affect.

__all__ = [
    "_compute_u_nk_analytical",
    "_parse_epoch_window_map_native_params",
    "_epoch_bias_param_vectors",
    "_reconstruct_union_bias_block",
    "_reconstruct_union_bias_block_per_regime",
]


def _strict_window(primary_center, primary_k, secondary_center, secondary_k):
    """Translate union-loader inactive sentinels at strict-core boundary."""
    try:
        secondary_center = float(secondary_center)
        secondary_k = float(secondary_k)
    except (TypeError, ValueError):
        secondary_center, secondary_k = 0.0, 0.0
    if (not math.isfinite(secondary_center)
            or not math.isfinite(secondary_k)
            or secondary_k <= 0.0):
        secondary_center, secondary_k = 0.0, 0.0
    return {
        "center1": primary_center, "k1": primary_k,
        "center2": secondary_center, "k2": secondary_k,
    }


def _compute_u_nk_analytical(cv1: np.ndarray, cv2: np.ndarray,
                              union_windows: list, beta: float) -> np.ndarray:
    """Compute N×K_union reduced bias matrix analytically from CV values."""
    windows = [_strict_window(w["primary_center"], w["primary_k_kcal"],
                              w["secondary_cv_center"], w["secondary_k_kcal"])
               for w in union_windows]
    return query.reconstruct_bias_matrix(cv1, cv2, windows, beta)


def _parse_epoch_window_map_native_params(rows: list) -> dict:
    """Return ``{state_id: {primary_center, primary_k, secondary_center, secondary_k}}``.

    These are the window parameters that were *actually in effect* for that
    specific epoch, as recorded in that epoch's own ``epoch_window_map.csv``
    snapshot at the time it ran — as opposed to whatever a state's row in the
    live/final registry says today. A state's secondary (or even primary)
    center/k can be recentered by later epochs (e.g. the tICA CV2 auto-switch
    in ``gareus/adaptive_production.py``, which overwrites
    ``state.secondary_center`` in place); this dict preserves the
    epoch-specific ground truth so bias energies for that epoch's own samples
    can be reconstructed against what was really applied, not what the state
    looks like now.
    """
    def _f(row: dict, key: str, default: float) -> float:
        v = row.get(key, '')
        if v in ('', 'None', 'nan', None):
            return default
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    out: dict = {}
    for r in rows:
        if 'state_id' not in r:
            continue
        sid = int(r['state_id'])
        out[sid] = {
            'primary_center': _f(r, 'primary_center', float('nan')),
            'primary_k': _f(r, 'primary_k', float('nan')),
            'secondary_center': _f(r, 'secondary_center', float('nan')),
            'secondary_k': _f(r, 'secondary_k', float('nan')),
        }
    return out


def _epoch_bias_param_vectors(native_params: dict, state_ids: list,
                               global_primary_centers: np.ndarray, global_primary_ks: np.ndarray,
                               global_sec_centers: np.ndarray, global_sec_ks: np.ndarray) -> tuple:
    """Per-epoch (primary_center, primary_k, secondary_center, secondary_k) vectors.

    Starts from the global (final-registry) arrays — the existing fallback
    behavior — then overrides entries for any state_id this epoch's own
    ``epoch_window_map.csv`` snapshot actually covers. States created in a
    later epoch (absent from this epoch's snapshot) keep the global fallback,
    which is correct: they didn't exist yet, so there is no "native" value to
    prefer, and no sample from this epoch can be assigned to them anyway.
    """
    pc = global_primary_centers.copy()
    pk = global_primary_ks.copy()
    sc = global_sec_centers.copy()
    sk = global_sec_ks.copy()
    for k, sid in enumerate(state_ids):
        row = native_params.get(sid)
        if row is None:
            continue
        if math.isfinite(row['primary_center']):
            pc[k] = row['primary_center']
        if math.isfinite(row['primary_k']):
            pk[k] = row['primary_k']
        if math.isfinite(row['secondary_center']):
            sc[k] = row['secondary_center']
        if math.isfinite(row['secondary_k']):
            sk[k] = row['secondary_k']
    return pc, pk, sc, sk


def _reconstruct_union_bias_block(cv: np.ndarray, cv2: np.ndarray, beta: float,
                                   primary_centers: np.ndarray, primary_ks: np.ndarray,
                                   sec_centers: np.ndarray, sec_ks: np.ndarray) -> np.ndarray:
    """Build one epoch-block's N x K reduced-bias-energy matrix.

    Thin delegation to reconstruct_bias_matrix: converts the array-based
    calling convention this function's callers already use into the
    windows:list[dict] shape reconstruct_bias_matrix expects. K is typically
    tens to low hundreds (real runs: K=91, K=364) and this is called once per
    epoch block, not per sample, so the K-dict-object construction cost here
    is negligible next to the O(N*K) numpy arithmetic reconstruct_bias_matrix
    itself performs.
    """
    windows = [_strict_window(primary_centers[k], primary_ks[k],
                              sec_centers[k], sec_ks[k])
               for k in range(len(primary_centers))]
    return query.reconstruct_bias_matrix(cv, cv2, windows, beta)


def _reconstruct_union_bias_block_per_regime(cv: np.ndarray, cv2_by_regime: dict, beta: float,
                                              primary_centers: np.ndarray, primary_ks: np.ndarray,
                                              sec_centers: np.ndarray, sec_ks: np.ndarray,
                                              state_regimes: Sequence[str]) -> np.ndarray:
    """Like ``_reconstruct_union_bias_block``, but each COLUMN gets the cv2 of
    the regime its own secondary params were written for.

    ``u_nk[n, k]`` must be the reduced bias state ``k`` would apply to frame
    ``n``. When a run changes its CV2 definition mid-campaign, state ``k``'s
    ``secondary_center``/``secondary_k`` are expressed in ONE definition, so
    evaluating them against a cv2 computed in the other definition does not
    describe any Hamiltonian -- and the two definitions' weight vectors are
    near-orthogonal in practice (cosine similarity 0.141, 81.9 degrees, on the
    motivating run), so it is not a small error either.

    Passing every regime's cv2 for every row (see ``cv2_reprojection``) lets
    each column be evaluated in its own definition, which is what makes the
    pooled solve a single well-defined problem rather than a splice.

    ``state_regimes[k]`` names the regime for column ``k``; every name must be
    a key of ``cv2_by_regime``. Rows where a needed cv2 is NaN (no frame at
    that step, so the foreign CV2 was never recoverable) propagate NaN, which
    the downstream clean() step drops -- silently biasing them to zero would be
    worse than losing them.
    """
    K = len(primary_centers)
    if len(state_regimes) != K:
        raise ValueError(f'state_regimes has {len(state_regimes)} entries but there '
                         f'are {K} states')
    missing = sorted({r for r in state_regimes} - set(cv2_by_regime))
    if missing:
        raise ValueError(f'no cv2 supplied for regime(s) {missing}; '
                         f'have {sorted(cv2_by_regime)}')
    cv = np.asarray(cv, dtype=np.float64)
    out = np.empty((cv.size, K), dtype=np.float64)
    regimes = np.asarray(list(state_regimes), dtype=object)
    for regime in dict.fromkeys(state_regimes):  # stable order, deduplicated
        cols = np.where(regimes == regime)[0]
        windows = [
            {"center1": primary_centers[k], "k1": primary_ks[k],
             "center2": sec_centers[k], "k2": sec_ks[k]}
            for k in cols
        ]
        block = query.reconstruct_bias_matrix(
            cv, np.asarray(cv2_by_regime[regime], dtype=np.float64), windows, beta)
        out[:, cols] = block
    return out
