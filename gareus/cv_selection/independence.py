"""Held-out independence and structural-information diagnostics for CV candidates.

Design decisions fixed by the adversarial review of the plan:

* ``gain = L1 - L12``, the held-out cross-entropy the anchor leaves on the
  table and the candidate removes -- an estimate of ``I(cell; z2 | z1)``.
  The earlier ``min(L1, L2) - L12`` switched, the moment a candidate beat the
  anchor on its own, to ``I(cell; z1 | z2) ~ 0`` and scored a perfect CV2
  identically to noise (measured: -0.003 nats for both).
* The joint model uses ``ceil(sqrt(n_bins))`` bins per axis so its capacity
  matches the marginals'; a 100-cell joint table is smoothed harder than a
  10-cell marginal and biases ``L12`` upward regardless of the data.
* Folds are grouped by seed family and the families are shuffled under a fixed
  seed before striding: the swarm plan emits seeds in stratum order, so an
  unshuffled stride hands each fold a different stratum mix.
* Cells are a *frame-level* geometric partition (:func:`frame_partition`),
  never the seed-level swarm strata, which are constant within a member
  (effective n = number of seeds) and contain the anchor as an axis.

Everything here is a diagnostic about the discovery distribution. Nothing is
a thermodynamic claim.
"""
from __future__ import annotations

import math

import numpy as np

DEFAULT_FOLD_SEED = 20260920


def grouped_folds(groups, n_folds: int, *, seed: int = DEFAULT_FOLD_SEED) -> list[np.ndarray]:
    """Row indices held out per fold; a group is never split across folds."""
    groups = np.asarray(groups)
    if groups.ndim != 1 or groups.size == 0:
        raise ValueError("groups must be a nonempty 1-D array")
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    unique = np.unique(groups)
    if unique.size < n_folds:
        raise ValueError(f"{unique.size} distinct groups cannot fill {n_folds} folds")
    unique = unique[np.random.default_rng(seed).permutation(unique.size)]
    return [np.flatnonzero(np.isin(groups, unique[i::n_folds])) for i in range(n_folds)]


def _equal_mass_edges(x: np.ndarray, n_bins: int) -> np.ndarray:
    return np.quantile(x, np.linspace(0.0, 1.0, n_bins + 1)[1:-1])


def heldout_nonlinear_r2(z_target, z_predictor, groups, *, n_bins: int = 10,
                         n_folds: int = 4) -> tuple[float, np.ndarray]:
    """Binned conditional-mean predictor of ``z_target`` from ``z_predictor``, scored out of fold.

    Returns the pooled R² and the per-fold R² so the caller can gate on
    ``mean + 2·SE`` rather than on a point estimate whose swarm-to-swarm
    spread is comparable to the threshold.
    """
    y = np.asarray(z_target, dtype=np.float64)
    x = np.asarray(z_predictor, dtype=np.float64)
    if y.shape != x.shape or y.ndim != 1:
        raise ValueError("target and predictor must be 1-D of equal length")
    per_fold = []
    sse = sst = 0.0
    for hold in grouped_folds(groups, n_folds):
        train = np.setdiff1d(np.arange(y.size), hold)
        edges = _equal_mass_edges(x[train], n_bins)
        bins_train = np.digitize(x[train], edges)
        bins_hold = np.digitize(x[hold], edges)
        overall = float(y[train].mean())
        means = np.array([y[train][bins_train == b].mean() if np.any(bins_train == b) else overall
                          for b in range(n_bins)])
        fold_sse = float(np.sum((y[hold] - means[bins_hold]) ** 2))
        fold_sst = float(np.sum((y[hold] - overall) ** 2))
        per_fold.append(1.0 - fold_sse / fold_sst if fold_sst > 0 else 0.0)
        sse += fold_sse
        sst += fold_sst
    pooled = 1.0 - sse / sst if sst > 0 else 0.0
    return float(pooled), np.asarray(per_fold, dtype=np.float64)


def frame_partition(features, *, n_cells: int, seed: int = DEFAULT_FOLD_SEED) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic farthest-point partition of standardised per-frame features.

    The first centre is the first row (row order is immutable); each further
    centre is the frame farthest from every existing centre. ``seed`` is
    accepted for signature stability but the construction is deterministic.
    Returns ``(labels, centres)``. A geometric partition, not a basin claim.
    """
    F = np.asarray(features, dtype=np.float64)
    if F.ndim != 2 or F.shape[0] < n_cells:
        raise ValueError("features must be (n, d) with n >= n_cells")
    scale = F.std(axis=0)
    F = (F - F.mean(axis=0)) / np.where(scale > 0, scale, 1.0)
    centres = [F[0]]
    distance = np.linalg.norm(F - centres[0], axis=1)
    while len(centres) < n_cells:
        nxt = int(np.argmax(distance))
        centres.append(F[nxt])
        distance = np.minimum(distance, np.linalg.norm(F - centres[-1], axis=1))
    centres = np.asarray(centres)
    labels = np.argmin(((F[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2), axis=1)
    return labels.astype(np.int64), centres


def _heldout_cross_entropy(cells, coords, groups, n_bins, n_folds, alpha) -> float:
    """Mean held-out ``-log P(cell | bin(coords))`` with additive smoothing."""
    cells = np.asarray(cells, dtype=np.int64)
    coords = np.asarray(coords, dtype=np.float64)
    coords = coords[:, None] if coords.ndim == 1 else coords
    k_axes = coords.shape[1]
    # Match model capacity: ~n_bins cells whether one or several axes.
    bins_per_axis = n_bins if k_axes == 1 else max(2, math.ceil(n_bins ** (1.0 / k_axes)))
    n_cells = int(cells.max()) + 1
    total = 0.0
    count = 0
    for hold in grouped_folds(groups, n_folds):
        train = np.setdiff1d(np.arange(cells.size), hold)
        edges = [_equal_mass_edges(coords[train, k], bins_per_axis) for k in range(k_axes)]

        def key(rows):
            idx = np.zeros(rows.size, dtype=np.int64)
            for k, e in enumerate(edges):
                idx = idx * bins_per_axis + np.digitize(coords[rows, k], e)
            return idx

        keys_train, keys_hold = key(train), key(hold)
        table: dict[int, np.ndarray] = {}
        for kk, c in zip(keys_train, cells[train]):
            table.setdefault(int(kk), np.zeros(n_cells))[c] += 1
        prior = np.bincount(cells[train], minlength=n_cells) + alpha
        prior = prior / prior.sum()
        for kk, c in zip(keys_hold, cells[hold]):
            counts = table.get(int(kk), np.zeros(n_cells)) + alpha * prior
            total += -math.log(counts[c] / counts.sum())
            count += 1
    return total / max(count, 1)


def incremental_cell_information(cells, z1, z2, groups, *, n_bins: int = 10, n_folds: int = 4,
                                 alpha: float = 1.0) -> dict:
    """``I(cell; z2 | z1)`` estimated as ``L(z1) - L(z1, z2)`` in nats, plus the pieces."""
    l1 = _heldout_cross_entropy(cells, z1, groups, n_bins, n_folds, alpha)
    l2 = _heldout_cross_entropy(cells, z2, groups, n_bins, n_folds, alpha)
    l12 = _heldout_cross_entropy(cells, np.column_stack([z1, z2]), groups, n_bins, n_folds, alpha)
    return {"l1": float(l1), "l2": float(l2), "l12": float(l12), "gain": float(l1 - l12)}
