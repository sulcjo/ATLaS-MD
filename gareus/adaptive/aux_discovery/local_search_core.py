"""Local side-chain candidate kernel; opt-in integration is intentionally separate.

split=0 fits/tunes preprocessing, split=1 selects C, split=2 scores candidates.
Rows 0+1 refit the chosen classifier. Region boundaries and split assignments
must already be frozen by the caller. Scores are heuristics, not equilibrium
probabilities or calibrated conditional-randomisation significance tests.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from numbers import Integral
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from .z3_search import _sklearn_has_l1_ratio_api


def _ids(values, n, name):
    a = np.asarray(values)
    if a.shape != (n,) or a.dtype.kind not in 'iu' or (a < 0).any():
        raise ValueError(f'{name} must contain nonnegative integer IDs, shape ({n},)')
    return a


def _count(x, name, minimum=1):
    if isinstance(x, (bool, np.bool_)) or not isinstance(x, Integral) or x < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(x)


def _fit(X, y, c):
    # Non-converged optimisation is not a successful scientific candidate.
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        kwargs = {'l1_ratio': 1.0} if _sklearn_has_l1_ratio_api() else {'penalty': 'l1'}
        return LogisticRegression(solver='liblinear', C=c, max_iter=3000, random_state=0,
                                  **kwargs).fit(X, y)


def _ll(model, X, y):
    logit = model.decision_function(X)
    return float(np.mean(-np.logaddexp(0., np.where(y == 1, -logit, logit))))


@dataclass(frozen=True)
class LocalCandidate:
    partition_id: str
    region: int
    pair: tuple[int, int]
    family: str
    c: float
    mean: np.ndarray
    sd: np.ndarray
    weights: np.ndarray
    scale: float
    local_gain: float
    prevalence: float
    score: float

    def __post_init__(self):
        for name in ('mean', 'sd', 'weights'):
            a = np.array(getattr(self, name), dtype=float, copy=True)
            a.setflags(write=False)
            object.__setattr__(self, name, a)

    @property
    def coefficients(self):
        return self.weights / self.sd

    @property
    def offset(self):
        return -float(self.coefficients @ self.mean)

    def values(self, features):
        X = np.asarray(features, dtype=float)
        if X.ndim != 2 or X.shape[1] != len(self.weights) or not np.isfinite(X).all():
            raise ValueError('candidate feature matrix has invalid shape/values')
        return (X @ self.coefficients + self.offset) / self.scale

    @property
    def identity(self):
        return (self.partition_id, self.region, self.pair, self.family, self.c)


def fit_local_candidates(features, labels, regions, split, *, partition_id, families,
                         c_grid=(.003, .01, .03, .1, .3), min_train=100,
                         min_holdout=50, min_share=.1, variance_floor=1e-8):
    """Return deterministic regional L1 candidates in descending comparable score.

    families maps family name to feature-column indices in one shared dictionary.
    Candidate coefficients remain full dictionary width for direct Projection
    construction. The search re-enumerates pairs/support for each label array;
    invoke it afresh for every scrambled label array. No holdout-derived regions.
    """
    X = np.asarray(features, dtype=float)
    if X.ndim != 2 or not len(X) or not np.isfinite(X).all():
        raise ValueError('features must be a nonempty finite 2D matrix')
    n, width = X.shape
    y, r, s = (_ids(v, n, name) for v, name in [(labels, 'labels'), (regions, 'regions'), (split, 'split')])
    if set(np.unique(s)) != {0, 1, 2}:
        raise ValueError('split requires fit=0, tune=1 and heldout=2 rows')
    if not isinstance(partition_id, str) or not partition_id.strip():
        raise ValueError('partition_id is required')
    min_train = _count(min_train, 'min_train', 2)
    min_holdout = _count(min_holdout, 'min_holdout', 2)
    if not np.isfinite(min_share) or not 0 < min_share <= .5:
        raise ValueError('min_share must be in (0, .5]')
    if not np.isfinite(variance_floor) or variance_floor <= 0:
        raise ValueError('variance_floor must be positive')
    cs = tuple(sorted(set(c_grid)))
    if not cs or any(isinstance(c, (bool, np.bool_)) or not np.isfinite(c) or c <= 0 for c in cs):
        raise ValueError('C grid must be finite and positive')
    views = []
    for family, cols in sorted(families.items()):
        if not isinstance(family, str) or not family:
            raise ValueError('family name is required')
        cols = tuple(cols)
        if any(isinstance(c, (bool, np.bool_)) or not isinstance(c, Integral) or c < 0 or c >= width for c in cols):
            raise ValueError('invalid feature column')
        if len(set(cols)) != len(cols):
            raise ValueError('duplicate feature column')
        if cols:
            views.append((family, tuple(sorted(cols))))
    train = s != 2
    out = []
    for region in sorted(np.unique(r[train])):
        for pair in combinations(sorted(np.unique(y[train & (r == region)])), 2):
            member = (r == region) & np.isin(y, pair)
            masks = [member & (s == k) for k in range(3)]
            tr = masks[0] | masks[1]
            binary = (y == pair[1]).astype(int)
            if tr.sum() < min_train or masks[2].sum() < min_holdout:
                continue
            if any(not m.any() or len(np.unique(binary[m])) < 2 or
                   min(binary[m].mean(), 1-binary[m].mean()) < min_share for m in masks):
                continue
            for family, cols in views:
                mean = X[masks[0]].mean(axis=0)
                std = X[masks[0]].std(axis=0)
                keep = np.array([c for c in cols if std[c] > variance_floor], dtype=int)
                if not keep.size:
                    continue
                sd = np.where(std > variance_floor, std, 1.)
                A = (X[:, keep]-mean[keep])/sd[keep]
                choices = []
                for c in cs:
                    model = _fit(A[masks[0]], binary[masks[0]], c)
                    choices.append((_ll(model, A[masks[1]], binary[masks[1]]), -c))
                chosen = -max(choices)[1]  # ties prefer stronger regularisation
                model = _fit(A[tr], binary[tr], chosen)
                weights = np.zeros(width)
                weights[keep] = model.coef_[0]
                raw = ((X-mean)/sd) @ weights
                scale = float(raw[tr].std())
                if not np.isfinite(scale) or scale <= variance_floor or not np.any(weights):
                    continue
                prior = (binary[tr].sum()+1.)/(tr.sum()+2.)
                ll0 = np.mean(np.where(binary[masks[2]] == 1, np.log(prior), np.log1p(-prior)))
                gain = _ll(model, A[masks[2]], binary[masks[2]]) - ll0
                prevalence = float(tr.sum()/train.sum())
                if not np.isfinite(gain):
                    raise ValueError('nonfinite heldout score')
                out.append(LocalCandidate(partition_id, int(region), tuple(map(int,pair)), family,
                                          float(chosen), mean, sd, weights, scale, float(gain),
                                          prevalence, float(prevalence*gain)))
    return sorted(out, key=lambda c: (-c.score, np.count_nonzero(c.weights), c.identity))


def compare_null(labels, lineage, step, search, *, n_null=20, seed=0):
    """Repeat a COMPLETE caller-supplied search on phase-local shifted labels.

    labels may be (n_frames,) or (n_frames,n_partitions). search(labels) must
    enumerate ALL partitions/families/regions/support each time and return their
    combined candidates. Shared row shifts preserve relationships among the
    partitions. Never select a winner from separate family/partition null tests.
    """
    n_null = _count(n_null, 'n_null')
    seed = _count(seed, 'seed', 0)
    y = np.asarray(labels)
    if y.ndim not in (1, 2) or not len(y) or (y.ndim == 2 and not y.shape[1]) or y.dtype.kind not in 'iu' or (y < 0).any():
        raise ValueError('labels must be nonempty integer IDs, optionally one column per partition')
    steps = _ids(step, len(y), 'step')
    lin = np.asarray(lineage)
    if lin.shape != (len(y),) or lin.dtype.kind not in 'USiu':
        raise ValueError('lineage must be a same-length array of phase-local string/integer IDs')
    blocks = []
    for key in np.unique(lin):
        idx = np.flatnonzero(lin == key)
        idx = idx[np.argsort(steps[idx], kind='stable')]
        if len(np.unique(steps[idx])) != len(idx):
            raise ValueError('duplicate phase-local observation steps')
        blocks.append(idx)
    def best(candidates):
        scores = [float(c.score) for c in candidates]
        if not np.isfinite(scores).all():
            raise ValueError('nonfinite candidate score')
        return max([0.] + scores)

    real = search(y.copy())
    observed = best(real)
    if all(np.all(y[b] == y[b[0]]) for b in blocks):
        return dict(status='null_uninformative_trapped_lineages', passed=False,
                    real_best=observed, null_best=[], n_null_run=0)
    if not real:
        return dict(status='no_real_candidate', passed=False, real_best=0., null_best=[], n_null_run=0)
    rng = np.random.default_rng(seed)
    maxima = []
    for _ in range(n_null):
        shuffled = y.copy()
        for b in blocks:
            if len(b) > 1:
                shuffled[b] = np.roll(y[b], int(rng.integers(1, len(b))), axis=0)
        candidates = search(shuffled)
        maxima.append(best(candidates))
    if not np.isfinite([observed]+maxima).all():
        raise ValueError('nonfinite candidate score')
    return dict(status='heuristic', passed=bool(observed > max(maxima)), real_best=observed,
                null_best=maxima, n_null_run=n_null)
