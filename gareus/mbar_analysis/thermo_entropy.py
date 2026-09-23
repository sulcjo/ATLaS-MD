"""Weighted torsion configurational entropy per basin (mutual-information expansion).

Spec: docs/superpowers/specs/2026-09-23-thermo-energy-decomposition/spec.md §11 (phase 2).

For torsions theta_1..theta_M (radians) of the samples in a basin, with unbiased weights w:

    S1/k = sum_i H_i                     H_i = -sum_b p_b ln p_b          (first-order MIE)
    S2/k = S1/k - sum_{i<j} I_ij         I_ij = H_i + H_j - H_ij           (second-order MIE)

The constant bin-width term of a differential entropy is omitted: it cancels in a basin DIFFERENCE
when both basins use the same bins, and only differences are reported. Histogram entropies are
biased low at small effective sample size; ESS is reported per basin.

The caller supplies the block index and the bootstrap multinomial draws, so entropy replicates are
paired with the energy replicates of gareus.mbar_analysis.thermo (spec §6).
"""
from __future__ import annotations

import math

import numpy as np


def torsion_bins(theta_rad, n_bins: int) -> np.ndarray:
    """(n, M) bin indices in [0, n_bins) for angles in radians on any branch; NaN -> -1."""
    theta = np.asarray(theta_rad, dtype=np.float64)
    out = np.full(theta.shape, -1, dtype=np.int64)
    ok = np.isfinite(theta)
    frac = np.mod(theta[ok] + math.pi, 2.0 * math.pi) / (2.0 * math.pi)
    out[ok] = np.minimum((frac * n_bins).astype(np.int64), n_bins - 1)
    return out


def _h(p) -> float:
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def mie_entropy(bins, w, n_bins: int, *, second_order: bool = True) -> dict:
    """S1/k, pairwise-MI sum and S2/k of weighted torsion histograms (bins all >= 0)."""
    bins = np.asarray(bins, dtype=np.int64)
    w = np.asarray(w, dtype=np.float64)
    tot = float(w.sum())
    nan = {"S1": float("nan"), "MI": float("nan"), "S2": float("nan")}
    if bins.ndim != 2 or bins.shape[0] == 0 or not tot > 0:
        return nan
    if np.any(bins < 0) or np.any(bins >= n_bins):
        raise ValueError("torsion bin indices outside [0, n_bins): drop samples with NaN torsions first")
    w = w / tot
    M = bins.shape[1]
    marg = [np.bincount(bins[:, i], weights=w, minlength=n_bins) for i in range(M)]
    H = [_h(p) for p in marg]
    S1 = float(sum(H))
    if not second_order:
        return {"S1": S1, "MI": float("nan"), "S2": float("nan")}
    MI = 0.0
    for i in range(M):
        for j in range(i + 1, M):
            pij = np.bincount(bins[:, i] * n_bins + bins[:, j], weights=w, minlength=n_bins * n_bins)
            MI += H[i] + H[j] - _h(pij)
    return {"S1": S1, "MI": float(MI), "S2": S1 - float(MI)}


def basin_entropy(bins, w, blk, counts, mask, n_bins: int, *, second_order: bool = True) -> dict:
    """Point estimate and bootstrap replicates of one basin's torsion entropy.

    ``blk`` is the 0..B-1 block index of every sample, ``counts`` the (n_boot, B) multinomial draws;
    replicate r reweights each sample by ``counts[r, blk]``.
    """
    mask = np.asarray(mask, dtype=bool)
    bb = np.asarray(bins)[mask]
    wb = np.asarray(w, dtype=np.float64)[mask]
    kb = np.asarray(blk, dtype=np.int64)[mask]
    point = mie_entropy(bb, wb, n_bins, second_order=second_order)
    reps = {"S1": [], "S2": []}
    for c in np.asarray(counts):
        r = mie_entropy(bb, wb * c[kb], n_bins, second_order=second_order)
        reps["S1"].append(r["S1"]); reps["S2"].append(r["S2"])
    ess = float(wb.sum() ** 2 / np.sum(wb ** 2)) if wb.size else 0.0
    return {"point": point, "reps": {k: np.asarray(v, dtype=np.float64) for k, v in reps.items()},
            "ess": ess, "n_frames": int(mask.sum()), "n_blocks": int(np.unique(kb).size)}
