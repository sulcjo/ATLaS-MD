# Adaptive λ Ladder (X1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Between epochs, re-place the interior Pep-GaMD λ rungs so that every adjacent rung pair overlaps at least `ladder_min_overlap` (default 0.25), keeping the endpoints λ = 0 and λ = λ_max (1.0) fixed, dropping redundant rungs and adding rungs only where needed.

**Architecture:** At a fixed umbrella centre the rung states differ only in boost strength, so each centre's rung samples define a one-parameter family whose reduced energy at ANY λ is computable per sample from the stored `v_pep_kj_mol`/`v_dih_kj_mol` and the frozen envelope (`pep_gamd_boost_kj`). A small per-centre MBAR over that centre's rungs predicts the pairwise overlap between any two λ values, sampled or not. A designer places the interior rungs by equal-overlap (thermodynamic-length) spacing on a low quantile of the per-centre overlaps; a planner turns the design into drop/add rung actions with hysteresis; the registry applies them atomically at the epoch boundary. New λ values are new states (a state_id's Hamiltonian never changes); retired rung states keep their samples in the union MBAR.

**Tech Stack:** Python 3, numpy, existing gareus modules (`gareus.pep_gamd`, `gareus.adaptive_production`, `gareus.mbar_analysis.loaders_adaptive`, `gareus.query`), pytest (run via opencode, per repo hook).

**Spec:** `docs/superpowers/specs/2026-09-29-adaptive-cv2-resolution-design.md` (v0.4), Section 12 X1, redesigned per the user on 2026-09-29: cover the whole λ range, respace/drop/add adaptively from measured overlaps, minimum adjacent overlap 0.25.

## Global Constraints

- Endpoints fixed: λ = 0 and λ = λ_max (the largest λ in the current ladder, 1.0 in all current campaigns) are never moved or dropped.
- Overlap scale: symmetrised pairwise MBAR overlap for two equal-weight states, O(a,b) = ∫ p_a p_b / (p_a + p_b), range [0, 0.5] -- the same scale as `gareus.mbar_analysis.ladder.pairwise_state_overlap` and `min_rung_overlap` (0.15) / `target_rung_overlap` (0.25).
- Default `ladder_min_overlap` = 0.25, aggregated over centres at quantile 0.10 (`ladder_overlap_quantile`).
- A state_id's Hamiltonian never changes: moving a rung = retiring that rung's states and adding states at the new λ. Asserted.
- Changes only at epoch boundaries, never in the frozen final phase.
- Off by default (`--ap-ladder-adapt off`); enabling it is recorded in `run_manifest.method_settings` at campaign start and honoured on resume.
- No native information anywhere.
- Tests run through opencode: `timeout 900 opencode run "In /run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler run exactly this command and nothing else: timeout 800 python -m pytest -q -p no:cacheprovider <files> 2>&1 | tail -6   Read-only: do not edit anything. Report the summary line and each failing test id with its error."` Never run pytest directly (hook-blocked); never append a bare directory to the file list.

## Review Focus

1. A centre whose rung set is incomplete this epoch (new centre, retired rung, a rung with < `ladder_min_samples` samples) must be skipped by the aggregate, not treated as zero overlap -- Task 2 test `test_incomplete_centres_are_skipped`.
2. NaN `v_pep`/`v_dih` samples (a segment written before the channels existed) must be dropped per sample, and a centre left with too few samples skipped -- Task 1 test `test_nan_channel_samples_are_dropped`.
3. A predicted λ outside the sampled range of a centre, or with reweighting ESS < `ladder_min_ess`, must not be trusted: the designer never places a rung where no centre can predict it -- Task 2 test `test_unsupported_lambda_is_not_placed`.
4. The replica cap: a plan that needs more states than `max_replicas` allows must be reported and not applied -- Task 4 test `test_respace_refused_over_cap`.
5. Resume between the registry save and the summary write after a respace must not re-propose or double-apply -- Task 5 test `test_resume_after_applied_actions_does_not_repropose`.

---

## File Structure

- Create `gareus/adaptive/ladder_adapt.py` -- per-centre rung model, overlap prediction, ladder designer, change planner, epoch sample loader, replay CLI (`python -m gareus.adaptive.ladder_adapt replay ...`). One responsibility: decide the λ set from data. ~350 lines.
- Modify `gareus/adaptive_production.py` -- `AdaptiveDecisionPolicy` fields; `AdaptiveProductionController.apply_actions` gains `retire_rung` and atomic `respace_ladder`; the driver calls the ladder proposer instead of `_propose_rung_actions` when enabled; applied-actions ledger.
- Modify `gareus/cli.py`, `gareus/config.py` -- flags and YAML keys.
- Modify `gareus/provenance.py` -- record the ladder settings in `method_settings`.
- Tests: `tests/test_ladder_adapt_model.py`, `tests/test_ladder_adapt_design.py`, `tests/test_ladder_adapt_loader.py`, `tests/test_ladder_adapt_actions.py`, `tests/test_ladder_adapt_resume.py`, `tests/test_ladder_adapt_wiring.py`.

---

### Task 1: Per-centre rung model and overlap prediction

**Files:**
- Create: `gareus/adaptive/ladder_adapt.py`
- Test: `tests/test_ladder_adapt_model.py`

**Interfaces:**
- Consumes: `gareus.pep_gamd.pep_gamd_boost_kj(v_pep_kj, v_dih_kj, lam, env) -> ndarray`, `gareus.pep_gamd.PepGamdEnvelope`.
- Produces:
  - `CentreRungModel` (frozen dataclass): `lambdas: np.ndarray` (sampled rungs, sorted), `n_k: np.ndarray`, `f_k: np.ndarray`, `v_pep: np.ndarray`, `v_dih: np.ndarray`, `beta: float`, `env: PepGamdEnvelope`; methods `reduced(lam) -> np.ndarray` (β·boost per pooled sample), `log_weights(lam) -> np.ndarray` (normalised MBAR log-weights of every pooled sample under λ), `ess(lam) -> float`, `overlap(lam_a, lam_b) -> float`.
  - `fit_centre_model(samples: dict[float, tuple[np.ndarray, np.ndarray]], env, beta, *, max_per_rung=5000) -> Optional[CentreRungModel]` -- `samples` maps sampled λ to `(v_pep, v_dih)` arrays; returns None if fewer than 2 rungs survive.

- [ ] **Step 1: Write the failing test**

```python
"""Per-centre rung model: predicted overlap between any two lambda values."""
import numpy as np
import pytest

from gareus.pep_gamd import PepGamdEnvelope, pep_gamd_boost_kj
from gareus.adaptive.ladder_adapt import fit_centre_model

BETA = 1.0 / (0.0083144626 * 300.0)
ENV = PepGamdEnvelope(vmax_total=-2400.0, vmin_total=-3700.0, threshold_total=-2400.0, k0max_total=0.36,
                      vmax_dih=545.0, vmin_dih=371.0, threshold_dih=545.0, k0max_dih=1.0)


def _base(n, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(-3000.0, 150.0, n), rng.normal(440.0, 20.0, n)


def _sample_at(lam, n, seed):
    """Exact samples of the lambda ensemble by importance resampling a big lambda=0 base."""
    vp, vd = _base(400_000, seed)
    logw = -BETA * pep_gamd_boost_kj(vp, vd, lam, ENV)
    w = np.exp(logw - logw.max()); w /= w.sum()
    idx = np.random.default_rng(seed + 1).choice(vp.size, n, p=w)
    return vp[idx], vd[idx]


def _direct_overlap(lam_a, lam_b, n=40_000):
    """Two-state overlap from large direct samples of both ensembles (reference)."""
    a = _sample_at(lam_a, n, 11); b = _sample_at(lam_b, n, 12)
    m = fit_centre_model({lam_a: a, lam_b: b}, ENV, BETA, max_per_rung=n)
    return m.overlap(lam_a, lam_b)


def test_identical_states_overlap_one_half():
    s = _sample_at(0.3, 5000, 3)
    m = fit_centre_model({0.0: _sample_at(0.0, 5000, 1), 0.3: s}, ENV, BETA)
    assert m.overlap(0.3, 0.3) == pytest.approx(0.5, abs=1e-9)


def test_overlap_decreases_with_lambda_separation():
    m = fit_centre_model({0.0: _sample_at(0.0, 5000, 1), 0.5: _sample_at(0.5, 5000, 2),
                          1.0: _sample_at(1.0, 5000, 4)}, ENV, BETA)
    vals = [m.overlap(0.0, l) for l in (0.1, 0.3, 0.5, 0.8, 1.0)]
    assert all(x > y for x, y in zip(vals, vals[1:]))


def test_unsampled_lambda_is_predicted_from_neighbouring_rungs():
    m = fit_centre_model({0.0: _sample_at(0.0, 8000, 1), 0.5: _sample_at(0.5, 8000, 2),
                          1.0: _sample_at(1.0, 8000, 4)}, ENV, BETA)
    assert m.overlap(0.5, 0.75) == pytest.approx(_direct_overlap(0.5, 0.75), abs=0.03)
    assert m.overlap(0.0, 0.25) == pytest.approx(_direct_overlap(0.0, 0.25), abs=0.03)


def test_ess_is_high_inside_and_falls_outside_the_sampled_range():
    m = fit_centre_model({0.0: _sample_at(0.0, 5000, 1), 0.3: _sample_at(0.3, 5000, 2)}, ENV, BETA)
    assert m.ess(0.15) > 1000
    assert m.ess(1.0) < m.ess(0.15)


def test_nan_channel_samples_are_dropped():
    vp, vd = _sample_at(0.0, 3000, 1)
    vp = vp.copy(); vp[:1000] = np.nan
    m = fit_centre_model({0.0: (vp, vd), 0.5: _sample_at(0.5, 3000, 2)}, ENV, BETA)
    assert m.n_k[0] == 2000


def test_fewer_than_two_rungs_gives_none():
    assert fit_centre_model({0.0: _sample_at(0.0, 1000, 1)}, ENV, BETA) is None
```

- [ ] **Step 2: Run test to verify it fails** (opencode command from Global Constraints with `tests/test_ladder_adapt_model.py`). Expected: collection error, `No module named 'gareus.adaptive.ladder_adapt'`.

- [ ] **Step 3: Write minimal implementation**

```python
"""Adaptive lambda ladder: predict rung overlaps from data and re-place interior rungs.

At a fixed umbrella centre the rung states differ only in boost strength, so every sample's
reduced energy at any lambda is beta * pep_gamd_boost_kj(v_pep, v_dih, lambda, env) plus terms
common to all rungs of that centre (they cancel). A per-centre MBAR over the sampled rungs then
gives the ensemble at ANY lambda by reweighting, and the pairwise overlap between any two lambda
values. The boost is not linear in lambda (the dependent dual boost adds the dihedral boost
inside the Total channel's square), so energies are always recomputed, never interpolated.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from gareus.pep_gamd import PepGamdEnvelope, pep_gamd_boost_kj


def _logsumexp(a, axis=None):
    m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    out = np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True)) + m
    return np.squeeze(out, axis=axis) if axis is not None else float(out)


@dataclass(frozen=True)
class CentreRungModel:
    lambdas: np.ndarray
    n_k: np.ndarray
    f_k: np.ndarray
    v_pep: np.ndarray
    v_dih: np.ndarray
    beta: float
    env: PepGamdEnvelope
    _denom: np.ndarray = field(repr=False, compare=False)   # log sum_k N_k exp(f_k - u_k(n))

    def reduced(self, lam: float) -> np.ndarray:
        return self.beta * np.asarray(pep_gamd_boost_kj(self.v_pep, self.v_dih, float(lam), self.env), dtype=float)

    def log_weights(self, lam: float) -> np.ndarray:
        """Normalised log-weights of the pooled samples in the ensemble at ``lam``."""
        lw = -self.reduced(lam) - self._denom
        return lw - _logsumexp(lw)

    def ess(self, lam: float) -> float:
        lw = self.log_weights(lam)
        return float(np.exp(-_logsumexp(2.0 * lw)))

    def overlap(self, lam_a: float, lam_b: float) -> float:
        """O(a,b) = integral p_a p_b / (p_a + p_b): 0.5 for identical ensembles."""
        la, lb = self.log_weights(lam_a), self.log_weights(lam_b)
        # at each sample, p_b/p_a = exp(lb - la) (both are densities w.r.t. the same mixture)
        return float(np.sum(np.exp(la) / (1.0 + np.exp(la - lb))))


def _mbar_f(u_kn: np.ndarray, n_k: np.ndarray, tol: float = 1e-10, max_iter: int = 20000) -> np.ndarray:
    f = np.zeros(u_kn.shape[0])
    log_n = np.log(n_k)
    for _ in range(max_iter):
        denom = _logsumexp(log_n[:, None] + f[:, None] - u_kn, axis=0)
        f_new = -_logsumexp(-u_kn - denom[None, :], axis=1)
        f_new -= f_new[0]
        if np.max(np.abs(f_new - f)) < tol:
            return f_new
        f = f_new
    return f


def fit_centre_model(samples: Dict[float, Tuple[np.ndarray, np.ndarray]], env: PepGamdEnvelope, beta: float,
                     *, max_per_rung: int = 5000) -> Optional[CentreRungModel]:
    lams, vps, vds, ns = [], [], [], []
    for lam in sorted(samples):
        vp, vd = (np.asarray(x, dtype=float) for x in samples[lam])
        ok = np.isfinite(vd) & (np.isfinite(vp) if env.has_total else True)
        vp, vd = vp[ok], vd[ok]
        if vp.size == 0:
            continue
        if vp.size > max_per_rung:
            idx = np.linspace(0, vp.size - 1, max_per_rung).astype(int)
            vp, vd = vp[idx], vd[idx]
        lams.append(float(lam)); vps.append(vp); vds.append(vd); ns.append(vp.size)
    if len(lams) < 2:
        return None
    v_pep, v_dih = np.concatenate(vps), np.concatenate(vds)
    lambdas, n_k = np.asarray(lams), np.asarray(ns, dtype=float)
    u_kn = np.vstack([beta * np.asarray(pep_gamd_boost_kj(v_pep, v_dih, l, env), dtype=float) for l in lambdas])
    f_k = _mbar_f(u_kn, n_k)
    denom = _logsumexp(np.log(n_k)[:, None] + f_k[:, None] - u_kn, axis=0)
    return CentreRungModel(lambdas, n_k, f_k, v_pep, v_dih, float(beta), env, denom)
```

- [ ] **Step 4: Run test to verify it passes.** Expected: 6 passed.
- [ ] **Step 5: Commit** `git add gareus/adaptive/ladder_adapt.py tests/test_ladder_adapt_model.py && git commit -m "feat: per-centre rung model predicts lambda overlaps from v_pep/v_dih"`

### Task 2: Ladder designer and change planner

**Files:**
- Modify: `gareus/adaptive/ladder_adapt.py`
- Test: `tests/test_ladder_adapt_design.py`

**Interfaces:**
- Consumes: `CentreRungModel` (Task 1).
- Produces:
  - `aggregate_overlap(models, lam_a, lam_b, *, quantile=0.10, min_ess=200.0) -> Optional[float]` -- quantile over centres whose model supports both λ (ESS >= min_ess at both, both inside `[min(lambdas), max(lambdas)]`); None if no centre supports them.
  - `LadderDesign` (frozen dataclass): `lambdas: tuple[float, ...]`, `predicted: tuple[Optional[float], ...]` (adjacent-pair aggregate overlaps), `min_overlap: Optional[float]`, `feasible: bool`, `reason: str`.
  - `design_ladder(models, *, lam_max=1.0, target=0.25, quantile=0.10, min_ess=200.0, max_rungs=8) -> LadderDesign` -- endpoints 0 and lam_max; minimum rung count whose greedy placement keeps every adjacent pair >= target; interior rungs then equalised (bisection on the common overlap level); `feasible=False` with a reason if even max_rungs cannot reach the target.
  - `LadderChange` (frozen dataclass): `drop: tuple[float, ...]`, `add: tuple[float, ...]`, `current_min: Optional[float]`, `design: LadderDesign`, `reason: str`.
  - `plan_ladder_change(current: Sequence[float], models, design: LadderDesign, *, target=0.25, quantile=0.10, min_ess=200.0, hysteresis=0.03, max_changes=2) -> LadderChange` -- no change if the current ladder already has every adjacent predicted overlap >= target, the same rung count as the design, and the design's minimum overlap is not better by more than `hysteresis`; otherwise drop/add to move toward the design, at most `max_changes` rung changes per epoch (a move counts as one drop + one add = 2); endpoints never dropped.

- [ ] **Step 1: Write the failing test**

```python
"""Ladder designer: endpoints fixed, adjacent overlaps >= target, minimal rungs, hysteresis."""
import numpy as np
import pytest

from gareus.adaptive.ladder_adapt import (aggregate_overlap, design_ladder, fit_centre_model,
                                          plan_ladder_change)
from tests.test_ladder_adapt_model import BETA, ENV, _sample_at


@pytest.fixture(scope="module")
def models():
    out = []
    for c in range(4):
        out.append(fit_centre_model({l: _sample_at(l, 4000, 100 + 10 * c + i)
                                     for i, l in enumerate((0.0, 0.25, 0.5, 0.75, 1.0))}, ENV, BETA))
    return out


def test_design_keeps_endpoints_and_meets_target(models):
    d = design_ladder(models, target=0.25)
    assert d.feasible and d.lambdas[0] == 0.0 and d.lambdas[-1] == 1.0
    assert all(p is not None and p >= 0.25 - 1e-6 for p in d.predicted)


def test_design_uses_the_minimum_rung_count(models):
    d = design_ladder(models, target=0.25)
    fewer = list(np.linspace(0.0, 1.0, len(d.lambdas) - 1))
    worst = min(aggregate_overlap(models, a, b) for a, b in zip(fewer, fewer[1:]))
    assert worst < 0.25                     # one rung fewer, even evenly spaced, fails


def test_design_equalises_adjacent_overlaps(models):
    d = design_ladder(models, target=0.25)
    assert max(d.predicted) - min(d.predicted) < 0.03


def test_higher_target_needs_more_rungs(models):
    assert len(design_ladder(models, target=0.35).lambdas) > len(design_ladder(models, target=0.2).lambdas)


def test_infeasible_target_is_reported_not_forced(models):
    d = design_ladder(models, target=0.49, max_rungs=4)
    assert not d.feasible and "max_rungs" in d.reason


def test_no_change_when_current_ladder_is_already_good(models):
    d = design_ladder(models, target=0.25)
    ch = plan_ladder_change(list(d.lambdas), models, d, target=0.25)
    assert ch.drop == () and ch.add == ()


def test_redundant_rung_is_dropped(models):
    d = design_ladder(models, target=0.20)
    current = sorted(set(d.lambdas) | {0.9, 0.95})
    ch = plan_ladder_change(current, models, d, target=0.20, max_changes=4)
    assert 0.95 in ch.drop or 0.9 in ch.drop
    assert 0.0 not in ch.drop and 1.0 not in ch.drop


def test_changes_per_epoch_are_capped(models):
    d = design_ladder(models, target=0.3)
    ch = plan_ladder_change([0.0, 1.0], models, d, target=0.3, max_changes=2)
    assert len(ch.drop) + len(ch.add) <= 2


def test_incomplete_centres_are_skipped(models):
    two_rung = fit_centre_model({0.0: _sample_at(0.0, 3000, 7), 0.25: _sample_at(0.25, 3000, 8)}, ENV, BETA)
    with_partial = list(models) + [two_rung]
    # (0.5, 1.0) is outside the partial centre's sampled range: it must not drag the quantile down
    assert aggregate_overlap(with_partial, 0.5, 1.0) == pytest.approx(aggregate_overlap(models, 0.5, 1.0))


def test_unsupported_lambda_is_not_placed():
    only_low = [fit_centre_model({0.0: _sample_at(0.0, 3000, 1), 0.3: _sample_at(0.3, 3000, 2)}, ENV, BETA)]
    d = design_ladder(only_low, target=0.25, lam_max=1.0)
    assert not d.feasible and "support" in d.reason
```

- [ ] **Step 2: Run test to verify it fails.** Expected: ImportError for `aggregate_overlap`.

- [ ] **Step 3: Write minimal implementation** (append to `gareus/adaptive/ladder_adapt.py`)

```python
from typing import List, Sequence


def _supports(m: CentreRungModel, lam: float, min_ess: float) -> bool:
    return float(m.lambdas[0]) - 1e-12 <= lam <= float(m.lambdas[-1]) + 1e-12 and m.ess(lam) >= min_ess


def aggregate_overlap(models, lam_a: float, lam_b: float, *, quantile: float = 0.10,
                      min_ess: float = 200.0) -> Optional[float]:
    vals = [m.overlap(lam_a, lam_b) for m in models
            if m is not None and _supports(m, lam_a, min_ess) and _supports(m, lam_b, min_ess)]
    return float(np.quantile(vals, quantile)) if vals else None


@dataclass(frozen=True)
class LadderDesign:
    lambdas: Tuple[float, ...]
    predicted: Tuple[Optional[float], ...]
    min_overlap: Optional[float]
    feasible: bool
    reason: str


def _next_rung(models, lam: float, lam_max: float, level: float, quantile: float, min_ess: float) -> Optional[float]:
    """Largest lam' in (lam, lam_max] with aggregate_overlap(lam, lam') >= level (bisection)."""
    o = aggregate_overlap(models, lam, lam_max, quantile=quantile, min_ess=min_ess)
    if o is not None and o >= level:
        return lam_max
    lo, hi = lam, lam_max
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        o = aggregate_overlap(models, lam, mid, quantile=quantile, min_ess=min_ess)
        if o is not None and o >= level:
            lo = mid
        else:
            hi = mid
    return lo if lo > lam + 1e-6 else None


def _greedy(models, lam_max, level, quantile, min_ess, max_rungs):
    lams = [0.0]
    while lams[-1] < lam_max - 1e-9 and len(lams) < max_rungs:
        nxt = _next_rung(models, lams[-1], lam_max, level, quantile, min_ess)
        if nxt is None:
            return None
        lams.append(nxt)
    return lams if lams[-1] >= lam_max - 1e-9 else None


def _predicted(models, lams, quantile, min_ess):
    return tuple(aggregate_overlap(models, a, b, quantile=quantile, min_ess=min_ess) for a, b in zip(lams, lams[1:]))


def design_ladder(models, *, lam_max: float = 1.0, target: float = 0.25, quantile: float = 0.10,
                  min_ess: float = 200.0, max_rungs: int = 8) -> LadderDesign:
    models = [m for m in models if m is not None]
    if not models or aggregate_overlap(models, 0.0, 0.0, quantile=quantile, min_ess=min_ess) is None \
            or aggregate_overlap(models, lam_max, lam_max, quantile=quantile, min_ess=min_ess) is None:
        return LadderDesign((0.0, lam_max), (None,), None, False, "no centre has support at both endpoints")
    first = _greedy(models, lam_max, target, quantile, min_ess, max_rungs)
    if first is None:
        reason = (f"target {target} not reachable within max_rungs={max_rungs}"
                  if _next_rung(models, 0.0, lam_max, target, quantile, min_ess) is not None
                  else f"no support for a rung above 0 at overlap {target}")
        if _greedy(models, lam_max, 0.0, quantile, min_ess, max_rungs) is None:
            reason = "insufficient reweighting support between sampled rungs"
        return LadderDesign((0.0, lam_max), _predicted(models, [0.0, lam_max], quantile, min_ess), None, False,
                            reason if "max_rungs" in reason else reason + " (support)")
    n = len(first)
    lo, hi, best = target, 0.5, first
    for _ in range(30):                       # equalise: highest common level still reaching lam_max in n rungs
        mid = 0.5 * (lo + hi)
        cand = _greedy(models, lam_max, mid, quantile, min_ess, n)
        if cand is not None and len(cand) == n:
            lo, best = mid, cand
        else:
            hi = mid
    best = best[:-1] + [lam_max]
    pred = _predicted(models, best, quantile, min_ess)
    return LadderDesign(tuple(best), pred, min(p for p in pred if p is not None), True, "ok")


@dataclass(frozen=True)
class LadderChange:
    drop: Tuple[float, ...]
    add: Tuple[float, ...]
    current_min: Optional[float]
    design: LadderDesign
    reason: str


def plan_ladder_change(current: Sequence[float], models, design: LadderDesign, *, target: float = 0.25,
                       quantile: float = 0.10, min_ess: float = 200.0, hysteresis: float = 0.03,
                       max_changes: int = 2) -> LadderChange:
    cur = sorted(float(x) for x in current)
    pred = _predicted(models, cur, quantile, min_ess)
    cur_min = min((p for p in pred if p is not None), default=None)
    if not design.feasible:
        return LadderChange((), (), cur_min, design, f"design infeasible: {design.reason}")
    ok_now = cur_min is not None and all(p is not None and p >= target for p in pred)
    if ok_now and len(cur) == len(design.lambdas) and (design.min_overlap or 0.0) - cur_min <= hysteresis:
        return LadderChange((), (), cur_min, design, "current ladder meets the target")
    ends = {cur[0], cur[-1]}
    interior_cur = [x for x in cur if x not in ends]
    interior_new = [x for x in design.lambdas if x not in ends]
    # pair each current interior rung with its nearest design rung; unmatched ones are dropped/added
    drop, add = list(interior_cur), list(interior_new)
    for x in sorted(interior_cur, key=lambda v: min((abs(v - y) for y in interior_new), default=9.0)):
        near = [y for y in add if abs(y - x) <= 0.02]
        if near:
            add.remove(near[0]); drop.remove(x)
    # budget: prefer fixing the weakest gap first (adds), then drops
    changes: List[Tuple[str, float]] = [("add", y) for y in add] + [("drop", x) for x in drop]
    changes = changes[:max(0, int(max_changes))]
    return LadderChange(tuple(v for k, v in changes if k == "drop"), tuple(v for k, v in changes if k == "add"),
                        cur_min, design, "respace toward design")
```

- [ ] **Step 4: Run test to verify it passes.** Expected: 10 passed. If `test_design_uses_the_minimum_rung_count` fails because greedy overshoots by one, keep the test and fix `_greedy` (it is the minimum-count guarantee).
- [ ] **Step 5: Commit** `git commit -m "feat: equal-overlap lambda ladder designer and change planner"`

### Task 3: Epoch sample loader and dry-run replay CLI

**Files:**
- Modify: `gareus/adaptive/ladder_adapt.py`
- Test: `tests/test_ladder_adapt_loader.py`

**Interfaces:**
- Consumes: `gareus.mbar_analysis.loaders_adaptive._find_adaptive_epoch_dirs(adaptive_dir, epoch_ids)`, `_validate_and_repair_epoch_window_map`, `gareus.mbar_analysis.ladder.load_pep_gamd_envelope(run_dir)`, `gareus.io.resolve_run_temperature_k`.
- Produces:
  - `load_centre_rung_samples(adaptive_dir: Path, registry_states: Sequence[dict], *, epoch_ids: Optional[set]) -> dict[tuple, dict[float, tuple[np.ndarray, np.ndarray]]]` -- key = centre key `(round(primary_center, 6), primary_k > 0, round(secondary_center or nan, 6), (secondary_k or 0) > 0)`; value maps λ -> `(v_pep, v_dih)`; rows deduplicated on (replica, step) keeping the latest segment; window->state via the repaired phase window map.
  - `replay(adaptive_dir, *, epoch_ids=None, target=0.25, quantile=0.10) -> dict` (JSON-able): current rungs, predicted adjacent overlaps (quantile and median over centres), design, planned change.
  - CLI: `python -m gareus.adaptive.ladder_adapt replay <adaptive_dir> [--epochs 3] [--target 0.25] [--quantile 0.1] [--json out.json]` -- read-only.

- [ ] **Step 1: Write the failing test** -- build a tiny adaptive dir fixture in `tmp_path`: `epoch_000/samples/seg_001/data.parquet` (columns `replica, step, window_id, cv1, cv2, gamd_lambda, v_pep_kj_mol, v_dih_kj_mol`), `epoch_000/segments.json` (`{"segments": []}`), `epoch_000/epoch_window_map.csv` (`epoch_window,state_id,primary_center,secondary_center`), `state_registry.json`, `global_shared_gamd_setup/shared_gamd_setup_globals.json` (globals from `ENV`). Two centres x three rungs, samples from `_sample_at`. Assert: the loader returns 2 centres x 3 λ with the right counts; a duplicated (replica, step) row is counted once; `replay()` returns `current == [0.0, 0.5, 1.0]` and a `design` with endpoints 0 and 1.

```python
import json
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from gareus.adaptive.ladder_adapt import load_centre_rung_samples, replay
from tests.test_ladder_adapt_model import ENV, _sample_at


def _fixture(tmp_path):
    ad = tmp_path / "adaptive_production"; ep = ad / "epoch_000"; (ep / "samples" / "seg_001").mkdir(parents=True)
    rows, wmap, states, w = [], [], [], 0
    for c, cv1 in enumerate((0.3, 0.7)):
        for i, lam in enumerate((0.0, 0.5, 1.0)):
            vp, vd = _sample_at(lam, 800, 1000 + 10 * c + i)
            for t in range(vp.size):
                rows.append(dict(replica=w, step=250 * (t + 1), window_id=w, cv1=cv1, cv2=0.0, gamd_lambda=lam,
                                 v_pep_kj_mol=vp[t], v_dih_kj_mol=vd[t]))
            wmap.append(dict(epoch_window=w, state_id=w, primary_center=cv1, secondary_center=0.0))
            states.append(dict(state_id=w, active=True, primary_center=cv1, primary_k=300.0, secondary_center=0.0,
                               secondary_k=1.0, gamd_lambda=lam))
            w += 1
    rows.append(dict(rows[0]))                                   # duplicate (replica, step)
    pq.write_table(pa.Table.from_pandas(pd.DataFrame(rows)), ep / "samples" / "seg_001" / "data.parquet")
    (ep / "segments.json").write_text(json.dumps({"segments": []}))
    pd.DataFrame(wmap).to_csv(ep / "epoch_window_map.csv", index=False)
    (ad / "global_shared_gamd_setup").mkdir()
    (ad / "global_shared_gamd_setup" / "shared_gamd_setup_globals.json").write_text(json.dumps({"all_globals": {
        "Vmax_Total": ENV.vmax_total, "Vmin_Total": ENV.vmin_total, "threshold_energy_Total": ENV.threshold_total,
        "k0_Total": ENV.k0max_total, "Vmax_Dihedral": ENV.vmax_dih, "Vmin_Dihedral": ENV.vmin_dih,
        "threshold_energy_Dihedral": ENV.threshold_dih, "k0_Dihedral": ENV.k0max_dih}, "temperature_K": 300.0}))
    return ad, states


def test_loader_groups_by_centre_and_rung_and_dedupes(tmp_path):
    ad, states = _fixture(tmp_path)
    got = load_centre_rung_samples(ad, states, epoch_ids=None)
    assert len(got) == 2 and all(sorted(v) == [0.0, 0.5, 1.0] for v in got.values())
    assert sum(len(v[0.0][0]) for v in got.values()) == 1600


def test_replay_reports_current_ladder_and_a_design(tmp_path):
    ad, _ = _fixture(tmp_path)
    rep = replay(ad, target=0.25)
    assert rep["current"] == [0.0, 0.5, 1.0]
    assert rep["design"]["lambdas"][0] == 0.0 and rep["design"]["lambdas"][-1] == 1.0
```

  Note for the implementer: check how `load_pep_gamd_envelope` reads the globals file (keys may be top-level or under `all_globals`) and write the fixture in the format it reads; the registry is read from `<adaptive_dir>/state_registry.json` in `replay`, while the loader takes the states list directly.

- [ ] **Step 2: Run, expect ImportError.**
- [ ] **Step 3: Implement** `load_centre_rung_samples` (for each `(phase_dir, wmap)` from `_find_adaptive_epoch_dirs`: read parquet with duckdb `select replica, step, window_id, gamd_lambda, v_pep_kj_mol, v_dih_kj_mol, filename`, dedupe on (replica, step) keeping the max segment number, map `window_id` through the repaired window map to `state_id`, look the state up in `registry_states`, group by centre key and by the state's `gamd_lambda`), `replay` (load envelope and temperature, fit one model per centre with >= 2 rungs, run `design_ladder` and `plan_ladder_change`, return a dict with `current`, `predicted_q`, `predicted_median`, `design` (lambdas, predicted, min_overlap, feasible, reason), `change` (drop, add, reason), `n_centres`), and the `argparse` CLI under `if __name__ == "__main__":`.
- [ ] **Step 4: Run, expect 2 passed.**
- [ ] **Step 5: Validation on real data (no test file; record the output in the commit message body):**
  `python -m gareus.adaptive.ladder_adapt replay RUNS/chignolin_9/adaptive_production --epochs final --target 0.25`
  and the same for `RUNS/chignolin_7/adaptive_production`. Check: predicted median adjacent overlaps at the current rungs agree with the union pairwise values in `pmf_analysis/overlap_pairs_mbar.csv` (chignolin_9 medians 0.243 / 0.272 / 0.402) within 0.03; report the proposed design.
- [ ] **Step 6: Commit** `git commit -m "feat: epoch rung-sample loader and dry-run ladder replay"`

### Task 4: Registry actions `retire_rung` and atomic `respace_ladder`

**Files:**
- Modify: `gareus/adaptive_production.py` (`AdaptiveProductionController.apply_actions`, new `_retire_rung_at_every_centre`, `AdaptiveDecisionPolicy.max_replicas_budget: int = 0`)
- Test: `tests/test_ladder_adapt_actions.py`

**Interfaces:**
- Consumes: `LadderChange` (Task 2).
- Produces: action tuple `("respace_ladder", drop: tuple[float, ...], add: tuple[float, ...], reason: str)`; `AdaptiveProductionController._retire_rung_at_every_centre(epoch, lam, reason) -> list[int]` (retired ids). Apply order inside one action: validate (endpoints `0` and `max(rung_lambdas())` never in `drop`; projected active count `len(active) + n_centres*len(add) - n_retired <= max_replicas_budget` when budget > 0), then add every new rung via `_add_rung_at_every_centre`, then retire every dropped rung. A refused action prints the reason and changes nothing.

- [ ] **Step 1: Write the failing tests** -- registry with 3 centres x rungs {0, 0.5, 1.0}:
  - `test_respace_adds_then_retires_whole_rungs`: apply `("respace_ladder", (0.5,), (0.3, 0.7), "t")`; `rung_lambdas() == [0.0, 0.3, 0.7, 1.0]`; every centre has exactly those rungs; retired 0.5 states have `retired_epoch == epoch + 1` and are still in `registry.states` (samples stay in the union).
  - `test_endpoints_are_never_dropped`: `("respace_ladder", (0.0,), (), "t")` and `(1.0,)` leave the registry unchanged and print a refusal.
  - `test_respace_refused_over_cap`: policy budget = 9 (current 9 states); `("respace_ladder", (), (0.3,), "t")` changes nothing; `(0.5,) -> (0.3,)` (net 0) applies.
  - `test_new_rung_states_are_new_ids`: new states have fresh state_ids, `parent_state_id` = the centre's λ = 0 representative, and no existing state's `gamd_lambda` changed.
- [ ] **Step 2: Run, expect failures (unknown action kind).**
- [ ] **Step 3: Implement** the branch in `apply_actions` and `_retire_rung_at_every_centre` (iterate `registry.active_states()`, retire those with `abs(gamd_lambda - lam) <= 1e-9`, skipping any with `metadata.mandatory`, returning ids). The controller receives the budget from `AdaptiveProductionController(registry, policy)`; update `_apply_registry_actions(registry, actions, epoch, policy=None)` to pass it (default keeps today's behaviour).
- [ ] **Step 4: Run, expect 4 passed; rerun `tests/test_adaptive_production*.py` files that exercise `apply_actions` (find with `grep -ln "apply_actions\|_apply_registry_actions" tests/`) -- expect no regressions.**
- [ ] **Step 5: Commit** `git commit -m "feat: atomic respace_ladder registry action (whole rungs, endpoints fixed, budgeted)"`

### Task 5: Applied-actions ledger (resume idempotence)

**Files:**
- Modify: `gareus/adaptive_production.py` (driver around the `_apply_registry_actions(...)` / `registry.save(...)` pair, and the resume path that sets `start_epoch` and reloads the registry)
- Test: `tests/test_ladder_adapt_resume.py`

**Interfaces:**
- Produces: `epoch_NNN/actions_applied.json` = `{"epoch": N, "actions": [...], "registry_digest": sha256 of the saved registry JSON}` written atomically (temp + `os.replace`) immediately after `registry.save`; `state_registry_pre_epoch_NNN.json` written before applying. Helper `_load_applied_actions(epoch_dir) -> Optional[dict]`.
- Resume rule: when re-entering epoch N and `actions_applied.json` exists with a digest matching the current registry file, skip proposal and apply for epoch N, and compute any epoch-N diagnostics against `state_registry_pre_epoch_NNN.json`.

- [ ] **Step 1: Write the failing test** -- unit-level: call the new helper pair `_record_applied_actions(epoch_dir, epoch, actions, registry_path)` and `_should_skip_proposal(epoch_dir, registry_path) -> bool`; assert skip is True after recording with an unchanged registry, False when the registry file changed after recording, False when no ledger exists; the ledger is valid JSON even if the process dies between temp write and replace (write to temp only, assert the old ledger is intact).
- [ ] **Step 2: Run, expect ImportError.**
- [ ] **Step 3: Implement the helpers and call them in the driver**: record right after `registry.save(adaptive_dir)`; at the top of the per-epoch action block, if `_should_skip_proposal(...)`, load the recorded actions (for the report) and skip `propose_actions_from_diagnostics` and `_apply_registry_actions`.
- [ ] **Step 4: Run, expect passes; rerun the resume tests (`grep -ln "resume" tests/test_adaptive*.py`).**
- [ ] **Step 5: Commit** `git commit -m "fix: applied-actions ledger makes an epoch's registry changes resume-idempotent"`

### Task 6: Driver wiring, flags, frozen settings, report

**Files:**
- Modify: `gareus/adaptive_production.py` (policy fields; call site of `_propose_rung_actions` at `propose_actions_from_diagnostics`; the driver after diagnostics), `gareus/cli.py`, `gareus/config.py`, `gareus/provenance.py`
- Test: `tests/test_ladder_adapt_wiring.py`

**Interfaces:**
- Policy fields: `ladder_adapt: str = "off"`, `ladder_min_overlap: float = 0.25`, `ladder_overlap_quantile: float = 0.10`, `ladder_min_ess: float = 200.0`, `ladder_max_rungs: int = 8`, `ladder_hysteresis: float = 0.03`, `ladder_max_changes_per_epoch: int = 2`.
- Flags: `--ap-ladder-adapt {off,respace}`, `--ap-ladder-min-overlap`, `--ap-ladder-overlap-quantile`, `--ap-ladder-min-ess`, `--ap-ladder-max-rungs`, `--ap-ladder-hysteresis`, `--ap-ladder-max-changes`; YAML keys under `adaptive_production:` mapped in `gareus/config.py`.
- Driver: when `ladder_adapt == "respace"` and the ladder is active (`len(registry.rung_lambdas()) > 1`) and the epoch is a numbered epoch (never `final`): `models` from `load_centre_rung_samples(adaptive_dir, [s.to_dict() for s in registry.states], epoch_ids={epoch})`, design + plan, append `("respace_ladder", change.drop, change.add, change.reason)` if non-empty, and skip `_propose_rung_actions`; always write `epoch_NNN/ladder_adapt_report.json` (the `replay`-shaped dict). `method_settings["ladder_adapt"]` records the policy values at campaign start; on resume the recorded values win over args unless `--ap-ladder-adapt-override` is given.

- [ ] **Step 1: Write the failing tests** -- parser defaults and overrides; YAML key mapping; `method_settings` contains `ladder_adapt` block; with `ladder_adapt="off"` the proposer output is byte-identical to today's (compare the action list on a fixed diagnostics fixture); with `"respace"` on a fixture registry + samples (reuse Task 3 fixture) the actions contain one `respace_ladder` and no `add_rung`; the final phase never gets one.
- [ ] **Step 2-4:** implement, run, pass; rerun `tests/test_cv_selection_cli.py tests/test_ap_epoch0_step_fraction.py tests/test_epoch_loop_convergence_guard.py` for regressions.
- [ ] **Step 5: Commit** `git commit -m "feat: --ap-ladder-adapt respace wires the adaptive lambda ladder into the epoch loop"`

### Task 7: Documentation and real-data record

**Files:** `CLAUDE.md` (new section), `gareus/helptext.py` (short `-hh` topic "Adaptive λ ladder"), spec Section 12 X1 (status line).

- [ ] Write the CLAUDE.md section: files, the overlap scale, endpoints rule, state-id invariant, flags and defaults, the chignolin_7/9 replay results from Task 3 Step 5, and the known limits (design uses per-epoch samples only; boosted rungs' seeding follows the existing `add_rung` path; final phase frozen).
- [ ] Commit `git commit -m "docs: adaptive lambda ladder"`

---

## Execution log (2026-09-29)

- Task 1 done: 8 tests. Deviation: the reference ensembles use a dihedral-only envelope (exact
  Gaussian rung ensembles, numerically exact reference overlap) instead of importance-resampling
  a lambda=0 base with the real dual envelope, which collapses (beta*sd(dV) ~ 10). Shared helpers
  live in `tests/ladder_adapt_fixture.py`, imported as `from ladder_adapt_fixture import ...`
  (repo convention; `from tests....` collides with an installed `tests` package). The model caches
  log-weights per lambda.
- Task 2 done: 13 tests. Deviations: the designer searches a fixed lambda grid (`grid_step`
  0.005) with memoised aggregate overlaps (0.1 s per design instead of minutes); the planner budget
  is `max_moves` (one interior rung changed), and a change within budget is the whole design
  applied atomically at one epoch boundary (intermediate ladders never run), so a respace keeps
  the rung count and needs no replica headroom.
- Task 3 done: 3 tests; replay on real data (read-only):
  - chignolin_9 final/baseline + final_extension_001, 59 centres: current [0, 0.235, 0.636, 1]
    predicted median overlaps 0.244 / 0.272 / 0.404 (union pairwise MBAR measured
    0.243 / 0.272 / 0.402); 10th percentile 0.209 / 0.248 / 0.396. Design [0, 0.175, 0.47, 1]:
    0.281 / 0.281 / 0.286. Change: drop 0.235, 0.636; add 0.175, 0.47.
  - chignolin_7 final, 16 centres: current [0, 0.232, 0.643, 1] q10 0.234 / 0.236 / 0.406;
    design [0, 0.18, 0.46, 1]: 0.291 / 0.290 / 0.286.
- Task 4 done: 5 tests; `respace_ladder` in `AdaptiveProductionController.apply_actions`
  (`_apply_respace_ladder`, `_retire_rung_at_every_centre`), `AdaptiveDecisionPolicy.max_replicas_budget`
  (0 = unlimited), controller takes `policy=`; `_apply_registry_actions(..., policy=None)`.
  Regression: 73 passed across the ladder-adapt files and the four existing apply_actions test files.
- Task 5 done: 5 tests. Ledger files live in the epoch directory
  (`actions_applied.json`, `state_registry_pre_actions.json`). The driver re-enters a recovered
  epoch on the pre-action registry for the whole epoch body (window map, MD, diagnostics,
  reports, seed bank), replays the recorded actions instead of proposing, re-records the ledger
  after the post-tICA coverage save, and does not re-apply coverage actions. The recovery branch
  is covered by helper tests only (no end-to-end kill/resume test).
- Task 6 done: 7 tests. `_resolve_ladder_settings` freezes the settings in
  `adaptive_production/ladder_adapt_settings.json` (instead of `method_settings`, which is per
  phase); YAML needs no `config.py` change (leaf keys are argparse dests: `ap_ladder_adapt`, ...).
  `max_replicas_budget` is filled from `--max-replicas`.
- Task 7 done: `-hh` topic "Adaptive lambda ladder", CLAUDE.md section.
- Regression: 391 passed across the ladder, rung, economy, synth-driver, tICA-coverage, CLI,
  epoch-loop, retire, new-state-guard, top-up and window-map-rewrite test files; helptext tests
  pass; `tests/test_atlas_md_docs.py` keeps its 2 pre-existing failures.
- Not committed (the branch also holds the uncommitted conditional-tICA selection work).

## Self-review

- Spec coverage (X1 as redefined): endpoints fixed (Tasks 2, 4), respace/drop/add from measured overlaps (Tasks 1-2, 4), 0.25 minimum (Global Constraints, Task 6 default), state-id invariant (Task 4), resume safety (Task 5), off by default + frozen settings (Task 6), replay on chignolin_7/9 before enabling (Task 3). Remaining spec items (P2/P4-P8 beyond what X1 needs, 3.1-3.7, X2-X8) are separate plans.
- Placeholder scan: the Task 3 note about the envelope file format is an instruction to check an existing reader, not a missing design.
- Type consistency: `CentreRungModel`, `fit_centre_model`, `aggregate_overlap`, `design_ladder -> LadderDesign`, `plan_ladder_change -> LadderChange`, `load_centre_rung_samples`, `replay`, action `("respace_ladder", drop, add, reason)` are used with the same names and shapes in Tasks 1-6.
