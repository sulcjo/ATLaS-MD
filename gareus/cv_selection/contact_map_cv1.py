"""Contact-map tICA CV1: fit, gate, pick and freeze (spec ``2026-10-01-contact-map-cv1.md``
section 3 with the section 7 decisions).

Input per swarm frame: the residue contact map recorded by ``gareus.swarm.contact_map`` (soft-min
heavy-atom distance d_p in A per residue pair p) turned into contacts by the calibrated rational
switch s_p = (1 - x^6)/(1 - x^12) = 1/(1 + x^6), x = d_p/r0 (r0 4.5 A). tICA on s (lagged pairs
within one swarm member, design-weighted, pair-centred, symmetrised, ridge-regularised) gives the
slow linear combinations; every mode is a candidate CV1

    c = sum_p w_p s_p - offset            (standardised: design-measure mean 0, sd 1)

Gates per mode (all must pass): lag autocorrelation >= ``min_slowness_rho``; reproduced in both
seed-family halves (each half's from-scratch tICA holds a mode with |r| >= ``half_split_min_corr``
on all rows); >= ``min_windows`` resolvable umbrella windows at ``k_max_kcal`` (kcal/mol per unit^2
of the standardised coordinate); breadth not demonstrably below ``min_breadth_nats``
(fold-averaged mean + 2 sd). Breadth = held-out information (nats) about a conformational
partition that does not use the contact map: per-residue backbone basins (torsion features, every
row) plus C-alpha geometry clusters (the rows with a PDB frame), averaged over
``breadth_resamples`` seed-family-to-fold assignments. Pick: among passing modes the breadth
tie-set (mean >= broadest mean - ``breadth_tie_sd`` x combined sd), then the slowest of it. No
mode passes -> the fallback CV1 (contacts) with the reasons.

Contacts and end-to-end distance are scored with the same breadth and slowness as reference rows
(``baselines``); they are never picked here (``cv1: auto`` across kinds is a later step).

Everything is about the swarm's design measure, never a statement about the 300 K ensemble.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from ..correctness._io import digest, json_bytes
from ..swarm.ladder_design import n_resolvable_windows
from .independence import DEFAULT_FOLD_SEED, _heldout_cross_entropy
from .slowness import conditional_autocorrelation, lag_pairs

SCHEMA = "cv1_contact_map_v1"
REPORT_SCHEMA = "cv1_contact_map_selection_v1"
FAMILY = "contact-map-tica"
SWITCH_FORM = "rational_6_12"
DEFAULT_SWITCH_R0_ANGSTROM = 4.5
MODEL_NAME = "cv1_model.json"
REPORT_NAME = "cv1_selection_report.json"
FALLBACK_CV1 = "contacts"
STATUS_SELECTED = "selected"
STATUS_FALLBACK = "fallback"


def rational_switch(d_angstrom, r0_angstrom: float = DEFAULT_SWITCH_R0_ANGSTROM) -> np.ndarray:
    """(1 - x^6)/(1 - x^12) with x = d/r0, written as 1/(1 + x^6) (no 0/0 at d = r0)."""
    x = np.asarray(d_angstrom, dtype=np.float64) / float(r0_angstrom)
    return 1.0 / (1.0 + x ** 6)


@dataclass(frozen=True)
class CV1FitConfig:
    switch_r0_angstrom: float = DEFAULT_SWITCH_R0_ANGSTROM
    tica_lag_ps: float = 50.0
    slowness_lag_ps: float = 200.0
    n_modes: int = 3
    ridge: float = 1e-6                    # x trace(C0)/p added to C0
    min_slowness_rho: float = 0.72
    half_split_min_corr: float = 0.8
    # kcal/mol per unit^2 of the standardised coordinate (spec section 4 default). NOTE: for a
    # coordinate with design sd 1 the range is >= 2 sd, so at 200 / 300 K / overlap 1.5 the gate
    # always finds >= 24 windows: it binds only when k_max is set far lower. Reported per mode;
    # the physically meaningful check is the CV1 ladder's own once the anchor kind exists.
    k_max_kcal: float = 200.0
    min_windows: int = 4
    overlap_sigma: float = 1.5
    temperature_k: float = 300.0
    min_breadth_nats: float = 0.02
    breadth_resamples: int = 8
    breadth_tie_sd: float = 1.0
    use_breadth_tie: bool = True           # False: every passing mode is in the tie-set (slowest wins)
    n_folds: int = 4

    def __post_init__(self):
        if not (self.switch_r0_angstrom > 0 and math.isfinite(self.switch_r0_angstrom)):
            raise ValueError("switch_r0_angstrom must be positive and finite")
        if int(self.n_modes) < 1 or int(self.breadth_resamples) < 1:
            raise ValueError("n_modes and breadth_resamples must be >= 1")
        if not (self.tica_lag_ps > 0 and self.slowness_lag_ps > 0):
            raise ValueError("lags must be positive")


@dataclass(frozen=True)
class CV1Data:
    """Rows of the swarm the fit uses, all aligned (one row per frame)."""
    distances: np.ndarray                  # (n, p) soft-min distances, A
    member_ids: np.ndarray
    frame_index: np.ndarray                # trace row index within the member
    groups: np.ndarray                     # seed family per row (folds, halves)
    cells: np.ndarray                      # coarse design partition (balanced weights)
    basin_labels: np.ndarray               # (n, n_residues) backbone basin index per residue
    frame_dt_ps: float
    lag_stride: int = 1                    # rows exist every lag_stride frames (PDB-frame source)
    ca_rows: Optional[np.ndarray] = None   # rows with a C-alpha cluster label
    ca_labels: Optional[np.ndarray] = None
    baselines: Mapping[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self):
        n = np.asarray(self.distances).shape[0]
        for name in ("member_ids", "frame_index", "groups", "cells", "basin_labels"):
            if np.asarray(getattr(self, name)).shape[0] != n:
                raise ValueError(f"{name} must have one entry per row")
        if (self.ca_rows is None) != (self.ca_labels is None):
            raise ValueError("ca_rows and ca_labels go together")
        if self.ca_rows is not None and np.asarray(self.ca_rows).shape != np.asarray(self.ca_labels).shape:
            raise ValueError("ca_rows and ca_labels must have equal length")
        for name, v in self.baselines.items():
            if np.asarray(v).shape != (n,):
                raise ValueError(f"baseline {name!r} must have one value per row")


def balanced_cell_weights(cells) -> np.ndarray:
    """Equal total mass per cell, equal mass per row within a cell (sum 1)."""
    _, inverse = np.unique(np.asarray(cells), return_inverse=True)
    counts = np.bincount(inverse)
    w = 1.0 / counts[inverse]
    return w / w.sum()


def lag_frames(lag_ps: float, frame_dt_ps: float, stride: int) -> int:
    """Lag in frames, a positive multiple of the row stride (PDB-frame rows are every stride)."""
    raw = float(lag_ps) / float(frame_dt_ps)
    return int(max(1, round(raw / int(stride))) * int(stride))


def fit_tica(S, member_ids, frame_index, weights, *, lag: int, n_modes: int, ridge: float):
    """Slowest ``n_modes`` tICA directions of S: (vectors (m, p), eigenvalues, mean (m,), sd (m,)).

    Lagged pairs within one member only, weighted by the first frame's design weight,
    centred on the pair set, symmetrised; C0 gets ``ridge`` x its mean eigenvalue. Sign:
    the largest-|weight| entry positive. mean/sd = the scores' design-measure moments."""
    S = np.asarray(S, dtype=np.float64)
    n, p = S.shape
    w = np.asarray(weights, dtype=np.float64)
    i, j = lag_pairs(member_ids, frame_index, lag)
    if i.size < 10 * p:
        raise ValueError(f"only {i.size} lag-{lag} frame pairs for {p} contact features")
    pw = w[i] / w[i].sum()
    pair_mean = 0.5 * (pw @ S[i] + pw @ S[j])
    x, y = S[i] - pair_mean, S[j] - pair_mean
    C0 = 0.5 * ((x * pw[:, None]).T @ x + (y * pw[:, None]).T @ y)
    Ct = 0.5 * ((x * pw[:, None]).T @ y + (y * pw[:, None]).T @ x)
    C0 = C0 + float(ridge) * max(float(np.trace(C0)) / p, 1e-300) * np.eye(p)
    lam, V = np.linalg.eigh(C0)
    W = V / np.sqrt(lam)
    ev, U = np.linalg.eigh(W.T @ Ct @ W)
    order = np.argsort(ev)[::-1][: int(n_modes)]
    vecs = []
    for k in order:
        v = W @ U[:, k]
        v = v / np.linalg.norm(v)
        pivot = v[np.argmax(np.abs(v))]
        vecs.append(v * (1.0 if pivot >= 0 else -1.0))
    vecs = np.asarray(vecs)
    scores = S @ vecs.T
    wn = w / w.sum()
    mu = wn @ scores
    sd = np.sqrt(wn @ (scores - mu) ** 2)
    if np.any(sd <= 1e-12):
        raise ValueError("a contact-map tICA mode has zero variance on the training rows")
    return vecs, ev[order].astype(np.float64), mu, sd


def lag_autocorrelation(z, data: CV1Data, lag: int, weights) -> float:
    return conditional_autocorrelation(z, np.zeros(np.asarray(z).size, dtype=np.int64),
                                       data.member_ids, data.frame_index, lag, weights=weights)


class _Breadth:
    """Held-out information of a coordinate about basins + CA clusters, per fold assignment."""

    def __init__(self, data: CV1Data, cfg: CV1FitConfig):
        self.data, self.cfg = data, cfg
        self._base: Dict[tuple, float] = {}

    def _ce(self, labels, coords, groups, seed) -> float:
        return _heldout_cross_entropy(labels, coords, groups, 10, self.cfg.n_folds, 1.0, seed)

    def _baseline(self, key, labels, groups, seed) -> float:
        if (key, seed) not in self._base:
            self._base[(key, seed)] = self._ce(labels, np.zeros((labels.size, 1)), groups, seed)
        return self._base[(key, seed)]

    def draw(self, z, seed: int) -> tuple:
        d = self.data
        z = np.asarray(z, dtype=np.float64)[:, None]
        basin = sum(self._baseline(("res", r), d.basin_labels[:, r], d.groups, seed)
                    - self._ce(d.basin_labels[:, r], z, d.groups, seed)
                    for r in range(d.basin_labels.shape[1]))
        ca = 0.0
        if d.ca_rows is not None and np.asarray(d.ca_rows).size:
            rows = np.asarray(d.ca_rows)
            g = np.asarray(d.groups)[rows]
            ca = self._baseline(("ca",), np.asarray(d.ca_labels), g, seed) - self._ce(d.ca_labels, z[rows], g, seed)
        return float(basin), float(ca)

    def score(self, z) -> dict:
        draws = [self.draw(z, DEFAULT_FOLD_SEED + k) for k in range(int(self.cfg.breadth_resamples))]
        tot = np.asarray([b + c for b, c in draws])
        return {"breadth_nats_mean": float(tot.mean()), "breadth_nats_sd": float(tot.std()),
                "breadth_resamples": len(draws),
                "basin_nats_mean": float(np.mean([b for b, _ in draws])),
                "ca_cluster_nats_mean": float(np.mean([c for _, c in draws]))}


def seed_halves(groups) -> tuple:
    unique = np.unique(np.asarray(groups))
    unique = unique[np.random.default_rng(DEFAULT_FOLD_SEED).permutation(unique.size)]
    return unique[: unique.size // 2], unique[unique.size // 2:]


def _half_scores(S, data: CV1Data, cfg: CV1FitConfig, lag: int) -> List[Optional[np.ndarray]]:
    """Each seed-family half's own tICA modes, evaluated on every row (None = fit failed)."""
    out = []
    for fam in seed_halves(data.groups):
        rows = np.isin(data.groups, fam)
        try:
            vecs, _ev, _mu, _sd = fit_tica(S[rows], data.member_ids[rows], data.frame_index[rows],
                                           balanced_cell_weights(data.cells[rows]), lag=lag,
                                           n_modes=cfg.n_modes, ridge=cfg.ridge)
            out.append(S @ vecs.T)
        except ValueError:
            out.append(None)
    return out


def _gate(row: dict, cfg: CV1FitConfig, *, mode: bool) -> None:
    reasons = row["reasons"]
    if not (row["slowness_rho"] >= cfg.min_slowness_rho):
        reasons.append(f"slowness rho {row['slowness_rho']:.3f} < {cfg.min_slowness_rho}")
    if row["breadth_nats_mean"] + 2.0 * row["breadth_nats_sd"] < cfg.min_breadth_nats:
        reasons.append(f"breadth {row['breadth_nats_mean']:.4f} +- {row['breadth_nats_sd']:.4f}: "
                       f"mean + 2 sd < {cfg.min_breadth_nats}")
    if mode:
        if row["n_resolvable_windows"] < cfg.min_windows:
            reasons.append(f"{row['n_resolvable_windows']} resolvable windows < {cfg.min_windows} "
                           f"at k_max {cfg.k_max_kcal}")
        corrs = row["half_split_abs_corr"]
        if any(c is None or not c >= cfg.half_split_min_corr for c in corrs):
            reasons.append(f"not reproduced in both seed-family halves (|r| {corrs}, need >= "
                           f"{cfg.half_split_min_corr})")
    row["passes"] = not reasons


def breadth_tie_set(rows: Mapping[str, dict], k: float) -> set:
    """Rows whose breadth mean is within k combined sd of the broadest (exact ties: lowest mode)."""
    if not rows:
        return set()
    broad = max(rows, key=lambda c: (rows[c]["breadth_nats_mean"], -int(rows[c].get("mode", 0))))
    m0, s0 = rows[broad]["breadth_nats_mean"], rows[broad]["breadth_nats_sd"]
    return {c for c, r in rows.items()
            if r["breadth_nats_mean"] >= m0 - float(k) * float(np.hypot(s0, r["breadth_nats_sd"]))}


def _score_modes(S, z_modes, ev, data, cfg, lag_t, lag_s, weights, breadth) -> Dict[str, dict]:
    halves = _half_scores(S, data, cfg, lag_t)
    rows = {}
    for k in range(z_modes.shape[1]):
        z = z_modes[:, k]
        corrs = [None if h is None else float(np.max(np.abs([np.corrcoef(h[:, q], z)[0, 1]
                                                              for q in range(h.shape[1])])))
                 for h in halves]
        row = {"role": "mode", "mode": k + 1, "family": FAMILY, "tica_eigenvalue": float(ev[k]),
               "slowness_rho": float(lag_autocorrelation(z, data, lag_s, weights)),
               "half_split_abs_corr": corrs,
               "n_resolvable_windows": int(n_resolvable_windows(float(z.max() - z.min()), cfg.temperature_k,
                                                                k_max_kcal=cfg.k_max_kcal,
                                                                overlap_sigma=cfg.overlap_sigma)),
               "coverage_range": float(z.max() - z.min()), "reasons": []}
        row.update(breadth.score(z))
        _gate(row, cfg, mode=True)
        rows[f"mode_{k + 1}"] = row
    return rows


def _score_baselines(data, cfg, lag_s, weights, breadth) -> Dict[str, dict]:
    rows = {}
    for name, v in data.baselines.items():
        v = np.asarray(v, dtype=np.float64)
        sd = float(v.std())
        z = (v - v.mean()) / (sd if sd > 0 else 1.0)
        row = {"role": "baseline", "slowness_rho": float(lag_autocorrelation(z, data, lag_s, weights)),
               "reasons": []}
        row.update(breadth.score(z))
        _gate(row, cfg, mode=False)
        rows[name] = row
    return rows


def select_contact_map_cv1(data: CV1Data, cfg: CV1FitConfig, *, definition: Mapping[str, Any],
                           training: Mapping[str, Any]) -> tuple:
    """(model or None, report). The model is the frozen ``cv1_contact_map_v1`` artifact."""
    S = rational_switch(data.distances, cfg.switch_r0_angstrom)
    weights = balanced_cell_weights(data.cells)
    stride = int(data.lag_stride)
    lag_t = lag_frames(cfg.tica_lag_ps, data.frame_dt_ps, stride)
    lag_s = lag_frames(cfg.slowness_lag_ps, data.frame_dt_ps, stride)
    report: Dict[str, Any] = {
        "schema": REPORT_SCHEMA, "n_rows": int(S.shape[0]), "n_pairs": int(S.shape[1]),
        "n_seed_families": int(np.unique(data.groups).size),
        "n_ca_rows": int(np.asarray(data.ca_rows).size) if data.ca_rows is not None else 0,
        "tica_lag_frames": lag_t, "tica_lag_ps": lag_t * float(data.frame_dt_ps),
        "slowness_lag_frames": lag_s, "slowness_lag_ps": lag_s * float(data.frame_dt_ps),
        "config": dict(vars(cfg)), "training": dict(training), "fallback_cv1": FALLBACK_CV1,
    }
    breadth = _Breadth(data, cfg)
    report["baselines"] = _score_baselines(data, cfg, lag_s, weights, breadth)
    try:
        vecs, ev, mu, sd = fit_tica(S, data.member_ids, data.frame_index, weights, lag=lag_t,
                                    n_modes=cfg.n_modes, ridge=cfg.ridge)
    except ValueError as exc:
        report.update(status=STATUS_FALLBACK, candidates={}, selection_reason=f"tICA failed: {exc}")
        return None, report
    z_modes = (S @ vecs.T - mu) / sd
    cands = _score_modes(S, z_modes, ev, data, cfg, lag_t, lag_s, weights, breadth)
    report["candidates"] = cands
    passing = {c: r for c, r in cands.items() if r["passes"]}
    if not passing:
        report.update(status=STATUS_FALLBACK, selection_reason=(
            f"no contact-map tICA mode passed the slowness, reproducibility, resolvability and "
            f"breadth gates; CV1 falls back to {FALLBACK_CV1}"))
        return None, report
    tie = breadth_tie_set(passing, cfg.breadth_tie_sd) if cfg.use_breadth_tie else set(passing)
    best = max(tie, key=lambda c: (passing[c]["slowness_rho"], -passing[c]["mode"]))
    k = passing[best]["mode"] - 1
    report.update(status=STATUS_SELECTED, selected=best, breadth_tie_set=sorted(tie),
                  selection_reason=(f"slowest of the breadth tie-set {sorted(tie)} among passing "
                                    f"{sorted(passing)}"))
    model = build_model(definition, cfg, vecs[k] / sd[k], float(mu[k] / sd[k]),
                        {**passing[best], "tica_lag_frames": lag_t, "tica_lag_ps": lag_t * float(data.frame_dt_ps),
                         "slowness_lag_ps": lag_s * float(data.frame_dt_ps)}, training)
    report["model_sha256"] = model["sha256"]
    return model, report


def build_model(definition: Mapping[str, Any], cfg: CV1FitConfig, weights, offset: float,
                component: Mapping[str, Any], training: Mapping[str, Any]) -> Dict[str, Any]:
    keep = ("mode", "family", "tica_eigenvalue", "tica_lag_frames", "tica_lag_ps", "slowness_rho",
            "slowness_lag_ps", "half_split_abs_corr", "breadth_nats_mean", "breadth_nats_sd",
            "n_resolvable_windows")
    body = {
        "schema": SCHEMA,
        "contact_map": dict(definition),
        "switch": {"form": SWITCH_FORM, "r0_angstrom": float(cfg.switch_r0_angstrom)},
        "weights": [float(x) for x in np.asarray(weights, dtype=np.float64)],
        "offset": float(offset),
        "units": "standardised (design-measure mean 0, sd 1)",
        "component": {k: component[k] for k in keep if k in component},
        "training": dict(training),
    }
    if len(body["weights"]) != len(definition["pairs"]):
        raise ValueError("one weight per contact-map residue pair")
    return {**body, "sha256": digest(json_bytes(body))}


def check_model(model: Mapping[str, Any]) -> Dict[str, Any]:
    """The model, verified: schema, digest, switch form and one weight per pair."""
    if model.get("schema") != SCHEMA:
        raise ValueError(f"not a {SCHEMA} model")
    body = {k: v for k, v in model.items() if k != "sha256"}
    if digest(json_bytes(body)) != model.get("sha256"):
        raise ValueError("cv1 model digest mismatch (edited after it was frozen?)")
    if model["switch"].get("form") != SWITCH_FORM:
        raise ValueError(f"unsupported switch form {model['switch'].get('form')!r}")
    if len(model["weights"]) != len(model["contact_map"]["pairs"]):
        raise ValueError("cv1 model weights do not match its contact-map pairs")
    return dict(model)


def evaluate_model(model: Mapping[str, Any], distances_angstrom) -> np.ndarray:
    """CV1 values of soft-min distance rows (n, p) or one row (p,)."""
    s = rational_switch(distances_angstrom, float(model["switch"]["r0_angstrom"]))
    return s @ np.asarray(model["weights"], dtype=np.float64) - float(model["offset"])


def write_model(path, model: Mapping[str, Any]) -> None:
    from ..io import write_json
    write_json(path, dict(check_model(model)))


def read_model(path) -> Dict[str, Any]:
    import json
    with open(path) as fh:
        return check_model(json.load(fh))


__all__ = ["CV1Data", "CV1FitConfig", "DEFAULT_SWITCH_R0_ANGSTROM", "FALLBACK_CV1", "FAMILY", "MODEL_NAME",
           "REPORT_NAME", "SCHEMA", "STATUS_FALLBACK", "STATUS_SELECTED", "balanced_cell_weights",
           "breadth_tie_set", "build_model", "check_model", "evaluate_model", "fit_tica", "lag_frames",
           "rational_switch", "read_model", "select_contact_map_cv1", "write_model"]
