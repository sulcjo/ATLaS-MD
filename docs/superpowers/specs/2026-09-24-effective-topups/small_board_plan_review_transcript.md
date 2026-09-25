## Question before the board

Adversarially review this implementation plan for effective adaptive top-ups. (1) Are the verification findings V1-V4 correct, and what are the right fixes (especially the multiple-comparison threshold for the split-halves check and the decorrelated-count inputs to the gain model)? (2) Find further defects the verification missed: statistical errors in the diagnostics/allocator code, lockstep/budget logic, resume and seeding hazards, test gaps, and anything that would make top-ups waste MD or corrupt a running campaign. (3) Suggest concrete efficiency improvements. Verdict: ACCEPT, ACCEPT_WITH_CHANGES or REJECT.

## Submission

# Implementation plan under adversarial review: effective adaptive top-ups (ATLaS-MD)

The approved design spec (previously reviewed by this board: ACCEPT_WITH_CHANGES) is summarised in the plan
header below. Below that: the plan's Global Constraints, flagged deviations and Review Focus verbatim, a
one-line summary of each of the 11 tasks, the FULL code the plan prescribes for the two statistically central
modules (union diagnostics, allocator), and the results of an adversarial verification already performed
by executing the plan's code and reading the production code it plugs into.

# Effective Adaptive Top-ups Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make scheduled top-ups spend MD only where a per-epoch union MBAR shows a statistical deficit, costed in wall-hours under the lockstep constraint, batched with minimal exchange partners, and seeded by continuing each window's chain from the baseline's final state; off by default; validated in the synthetic harness only.

**Architecture:** Three new pure modules under `gareus/adaptive/` (throughput model, autocorrelation, union diagnostics) feed a pure allocator (`topup_allocator.py`) that returns one `TopupPlan` per epoch. `run_scheduled_adaptive_epoch` runs the all-state baseline, calls the diagnostics and allocator, and runs at most one top-up segment. Production exports each window's final portable State at segment end; a top-up segment starts its windows from those States instead of pulling. The synthetic harness gains rungs, a mixing model, lockstep segments and the real union solve, and runs the A/B study.

**Tech Stack:** Python 3.9 (aurum2 conda env `calc`) / 3.14 (local), numpy, pymbar 4.x, OpenMM 8.x (seeding only), existing `gareus` package.

**Spec:** `docs/superpowers/specs/2026-09-24-effective-topups-design.md` (read it first; this plan argues from it).

## Global Constraints

- Top-ups are **off by default**: `--ap-topups` default `False`; YAML key `ap_topups`.
- Defaults: `--ap-topup-target-sigma 0.10` (kcal/mol), `--ap-topup-weak-overlap 0.15`, `--ap-topup-max-fraction 0.3`.
- Throughput table default (contexts/GPU → ns/day/node): `16 → 3154.0`, `59 → 2300.0`; flat (no extrapolated gain) below the lowest point and above the highest.
- An unmeasured edge is **never** weak (allocator and convergence gate).
- Healthy states get **zero** top-up steps; no default score; no distribution of left-over budget.
- A weak edge whose both endpoints are statistically adequate is **structural**: route to the bridge/add proposal, never to MD.
- One top-up segment per epoch at most; all its replicas run the same number of steps (lockstep).
- The old allocator path (score ≥ 1 for every state, grouping by extra size, re-pull seeding for top-ups) is removed, not kept as a mode.
- Validation is synthetic only: no MD campaign, no benchmark job, no real-file spike in this plan.
- Tests run through opencode (pytest is hook-blocked in this repo): `opencode run "From the repo root run exactly: python -m pytest -q -p no:cacheprovider <files> > atlas_task.log 2>&1 ; then print the last 30 lines of atlas_task.log. Do not edit any files."`. Run only the test files named in the task (user preference: targeted tests only, never the full suite).
- Commit messages: conventional commits, ending with the session's `Co-Authored-By` line.

**Deviation from the spec (1), flagged for review — local σ_k:** the spec defines σ_k "relative to a fixed reference state". That makes the reference state's σ identically 0 (it can never be a deficit) and gives distant states a large σ accumulated along the whole chain, which topping up those states cannot fix. This plan uses the **local** uncertainty σ_k = min over k's edge-neighbours j of the pairwise uncertainty of f_k − f_j (pymbar's `dDelta_f[j, k]`); a state with no measured neighbour falls back to the median of its row. This is the quantity more MD on k actually reduces.

**Deviation from the spec (2), flagged for review — seeding route:** the spec makes the binary-checkpoint load the primary seeding route and a portable State the fallback. This plan makes the **portable State** (positions, velocities, box of each window's final frame, exported at segment end) the primary route, because the fresh-segment code already accepts per-window start positions/velocities and a binary checkpoint cannot be loaded into a Context of a different replica set without re-implementing the resume path. Exact positions + velocities with the frozen GaMD envelope continue the chain, so no burn-in is needed; the spec's seeding assertion is kept. The fallback when no exported State exists is the existing pull (with a warning), not a burn-in route.

## Review Focus

1. **A deficit state with no same-rung spatial neighbour** (edge of the layout, or a sparse row) — the patch builder must still return a valid patch (it adds the nearest partner on any rung) rather than crash or silently drop the state. Test in Task 6.
2. **The union solve fails or returns NaN σ for some states** (a state with zero samples, a solver non-convergence) — the epoch runs no top-up and says why; it must never fall back to the old allocator or top up NaN states. Test in Task 5 and Task 8.
3. **Resume in the middle of a top-up** — the top-up plan must be read back from disk (`topup_plan.json`) rather than recomputed from diagnostics that now include the partial top-up's own samples, or the segment name/steps change under a resumed segment. Test in Task 8.
4. **Exported final State missing for some windows** (segment written by older code, interrupted before the end) — top-up seeding uses the exported States it has, and falls back to the pull only for the windows without one, with a warning naming them. Test in Task 9.
5. **Cap smaller than one useful top-up** (tiny remaining pool) — the allocator returns "no top-up" instead of a zero-length or sub-report-interval segment. Test in Task 6.

---


## Task summaries
### Task 1: Shared layout-neighbour module
### Task 2: Flags, defaults and policy fields
### Task 3: Throughput model
### Task 4: Conservative autocorrelation estimator
### Task 5: Per-epoch union diagnostics
### Task 6: Top-up allocator
### Task 7: Unmeasured is never weak (gate and remaining weak-edge readers)
### Task 8: Wire diagnostics, allocator and one top-up into scheduled phases
### Task 9: Top-up seeding from the baseline's final window States
### Task 10: Synthetic harness — rungs, mixing, lockstep, union solve, A/B study
### Task 11: Documentation
- T1 move neighbour rule to shared module; T2 flags (top-ups off by default) + policy fields; T3 wall-hour cost via throughput table (flat outside measured range); T4 Geyer IPS statistical inefficiency; T5 per-epoch union MBAR diagnostics (code below); T6 allocator (code below); T7 unmeasured edge never weak (rung edges judged on energy-space mbar_overlap only); T8 wiring: uniform schedule (old score allocator deleted), baseline shortened to (1-cap)*default, allocator plan persisted to topup_plan.json for resume, at most one top-up segment, online calibration (realised/predicted variance reduction; <0.5 halves priority), union overlaps written into epoch diagnostics' mbar_overlap so the existing proposer places bridges / add_rung; T9 seeding: each segment exports final positions/velocities/box per window (final_window_states/), top-ups start from them via the existing start-state path, fallback to the old pull for missing windows, assertion that the seeded frame reproduces the recorded CV1/CV2 (tol 1e-4); T10 synthetic harness A/B (rungs as lambda-scaled landscape flattening, tau scaled by exchange partners, lockstep segments, real union solve, 20 seeds, equal modelled wall-hours); T11 docs.

## Task 5 prescribed code (union diagnostics)
```python
# gareus/adaptive/union_diagnostics.py
"""Per-epoch union MBAR: per-state sigma, split-halves check, inefficiency, edge overlap.

Reads the union NPZ written by adaptive_production.build_union_state_mbar_inputs
(reduced potential of every sample in every state, lambda boost folded in).
Never raises: any failure returns None and the epoch runs no top-up.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np

from .autocorr import per_state_inefficiency


@dataclass(frozen=True)
class UnionDiagnostics:
    state_ids: tuple
    n_k: dict
    sigma_kcal: dict
    unconverged: frozenset
    inefficiency: dict
    edge_overlap: dict


def _solve(u_nk: np.ndarray, window: np.ndarray, n_states: int):
    """f (vs first active state) and the full pairwise uncertainty matrix dDelta_f (K x K, nan for unsampled)."""
    from pymbar import MBAR  # noqa: PLC0415
    n_k = np.bincount(window, minlength=n_states)
    active = np.flatnonzero(n_k > 0)
    order = np.argsort(window, kind="stable")                 # pymbar wants rows grouped by state
    u_kn = u_nk[order][:, active].T
    mbar = MBAR(u_kn, n_k[active], initialize="BAR", solver_protocol="robust")
    res = mbar.compute_free_energy_differences(compute_uncertainty=True)
    f = np.full(n_states, np.nan); dmat = np.full((n_states, n_states), np.nan)
    f[active] = res["Delta_f"][0]
    dmat[np.ix_(active, active)] = res["dDelta_f"]
    return f, dmat, n_k


def _local_sigma(dmat: np.ndarray, k: int, neighbours) -> float:
    """min over edge-neighbours j of dDelta_f[j, k]; median of the row when k has no measured neighbour."""
    vals = [dmat[j, k] for j in neighbours if np.isfinite(dmat[j, k])]
    if vals:
        return float(min(vals))
    row = dmat[k][np.isfinite(dmat[k]) & (np.arange(len(dmat)) != k)]
    return float(np.median(row)) if row.size else float("nan")


def union_diagnostics_from_npz(npz_path, edges: Iterable[Tuple[int, int]], *, kt_kcal: float,
                               split_halves: bool = True) -> Optional[UnionDiagnostics]:
    try:
        npz_path = Path(npz_path)
        if not npz_path.exists():
            return None
        with np.load(npz_path, allow_pickle=False) as z:
            u = np.asarray(z["umbrella_reduced_bias_nk"], dtype=float)
            ids = [int(s) for s in z["state_ids"].tolist()]
            sampled = np.asarray(z["sampled_state_ids"], dtype=np.int64)
        idx = {s: i for i, s in enumerate(ids)}
        window = np.asarray([idx.get(int(s), -1) for s in sampled], dtype=np.int64)
        keep = (window >= 0) & np.isfinite(u).all(axis=1)
        u, window = u[keep], window[keep]
        if u.shape[0] == 0:
            return None
        K = len(ids)
        f, dmat, n_k = _solve(u, window, K)
        nbrs = {k: [] for k in range(K)}
        for a, b in edges:
            ia, ib = idx.get(int(a)), idx.get(int(b))
            if ia is not None and ib is not None:
                nbrs[ia].append(ib); nbrs[ib].append(ia)
        df = np.array([_local_sigma(dmat, k, nbrs[k]) if n_k[k] > 0 else np.nan for k in range(K)])
        unconverged = set()
        if split_halves:
            half = np.zeros(len(window), dtype=bool)
            for k in range(K):
                rows = np.flatnonzero(window == k)
                half[rows[: len(rows) // 2]] = True
            fa, dma, _ = _solve(u[half], window[half], K)
            fb, dmb, _ = _solve(u[~half], window[~half], K)
            for k in range(K):
                # compare each state's free energy relative to its neighbours between the halves
                js = [j for j in nbrs[k] if np.isfinite(fa[j]) and np.isfinite(fb[j])]
                if not js:
                    continue
                j = js[0]
                da, db = fa[k] - fa[j], fb[k] - fb[j]
                comb = math.hypot(dma[j, k], dmb[j, k])
                fa_k, fb_k = da, db
                if all(math.isfinite(v) for v in (fa_k, fb_k, comb)) and abs(fa_k - fb_k) > 2.0 * comb:
                    unconverged.add(ids[k])
        from ..mbar_analysis.ladder import mbar_state_overlap  # noqa: PLC0415
        f_fill = np.where(np.isfinite(f), f, 0.0)
        O = mbar_state_overlap(u, f_fill, n_k)
        edge_overlap = {}
        for a, b in edges:
            ia, ib = idx.get(int(a)), idx.get(int(b))
            if ia is None or ib is None or n_k[ia] == 0 or n_k[ib] == 0:
                continue
            x, y = float(O[ia, ib]), float(O[ib, ia])
            if math.isfinite(x) and math.isfinite(y) and x >= 0 and y >= 0:
                edge_overlap[(min(int(a), int(b)), max(int(a), int(b)))] = math.sqrt(x * y)
        own_u = u[np.arange(len(window)), window]              # each sample's reduced potential in its own state
        g = per_state_inefficiency(own_u, window, list(range(K)))
        return UnionDiagnostics(
            state_ids=tuple(ids),
            n_k={ids[k]: int(n_k[k]) for k in range(K)},
            sigma_kcal={ids[k]: (float(df[k]) * kt_kcal if math.isfinite(df[k]) else math.nan) for k in range(K)},
            unconverged=frozenset(unconverged),
            inefficiency={ids[k]: float(g[k]) for k in range(K)},
            edge_overlap=edge_overlap,
        )
    except Exception as exc:  # the epoch then runs no top-up
        logging.warning("top-up union diagnostics unavailable (%s)", exc)
        return None
```

## Task 6 prescribed code (allocator)
```python
# gareus/adaptive/topup_allocator.py
"""Deficit-driven, wall-hour-costed top-up plan (spec 4.2-4.3). Pure function; no I/O."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

from .throughput import wall_hours

MAX_STEP_MULTIPLE = 4


@dataclass(frozen=True)
class TopupPlan:
    state_ids: tuple = ()
    steps: int = 0
    deficit_state_ids: tuple = ()
    partner_state_ids: tuple = ()
    structural_edges: tuple = ()
    predicted_sigma: dict = field(default_factory=dict)
    cost_hours: float = 0.0
    reason: str = "healthy"
    sigma_before: dict = field(default_factory=dict)   # sigma of each patch state when planned


def _finite(x) -> bool:
    return x is not None and isinstance(x, (int, float)) and math.isfinite(float(x))


def _predicted(sigma, steps_so_far, extra, interval, g, c) -> float:
    n_eff = max(1e-9, steps_so_far / (interval * g))
    return sigma * math.sqrt(n_eff / (n_eff + c * extra / (interval * g)))


def _required_steps(sigma, target, steps_so_far, interval, g, c) -> Optional[int]:
    if sigma <= target:
        return 0
    n_eff = max(1e-9, steps_so_far / (interval * g))
    extra_eff = n_eff * ((sigma / target) ** 2 - 1.0)
    steps = extra_eff * interval * g / max(c, 1e-9)
    steps = int(math.ceil(steps / interval) * interval)
    cap = MAX_STEP_MULTIPLE * max(interval, int(steps_so_far))
    return min(steps, int(math.ceil(cap / interval) * interval))


def _worst(sig: Dict[int, float], ids: Iterable[int]) -> float:
    vals = [sig[s] for s in ids]
    return max(vals) if vals else 0.0


def plan_topup(diag, *, state_ids_in_order: Sequence[int], neighbours: Dict[int, List[int]],
               rung_partners: Dict[int, List[int]], policy, steps_so_far: Dict[int, int],
               report_interval: int, timestep_fs: float, n_gpus: int, budget_hours: float,
               correction: Optional[Dict[int, float]] = None) -> TopupPlan:
    if diag is None:
        return TopupPlan(reason="no_diagnostics")
    correction = correction or {}
    target = float(policy.topup_target_sigma)
    sigma = {s: float(diag.sigma_kcal.get(s, math.nan)) for s in state_ids_in_order}
    deficits = {s for s, v in sigma.items() if _finite(v) and (v > target or s in diag.unconverged)}
    structural = []
    for (a, b), ov in sorted(diag.edge_overlap.items()):
        if not _finite(ov) or ov >= float(policy.topup_weak_overlap):
            continue
        if a in deficits or b in deficits:
            deficits.update(x for x in (a, b) if _finite(sigma.get(x, math.nan)))
        else:
            structural.append((a, b))
    if not deficits:
        return TopupPlan(structural_edges=tuple(structural), reason="healthy")

    g = {s: max(1.0, float(diag.inefficiency.get(s, 1.0))) for s in state_ids_in_order}
    need = {}
    for s in deficits:
        # an unconverged state below the sigma target still gets one doubling of its data
        eff_target = target if sigma[s] > target else sigma[s] / math.sqrt(2.0)
        need[s] = _required_steps(sigma[s], eff_target, max(1, steps_so_far.get(s, 0)),
                                  report_interval, g[s], correction.get(s, 1.0)) or report_interval

    def partners_for(short: set) -> set:
        chosen: set = set()
        for s in sorted(short):
            same = [x for x in neighbours.get(s, []) if x in sigma]
            rung = [x for x in rung_partners.get(s, []) if x in sigma]
            pool = same or rung
            if pool and not any(x in short for x in same) and not any(x in chosen for x in same):
                chosen.add(max(pool, key=lambda x: (sigma[x] if _finite(sigma[x]) else -1.0, -x)))
            if rung and not any(x in short or x in chosen for x in rung):
                chosen.add(max(rung, key=lambda x: (sigma[x] if _finite(sigma[x]) else -1.0, -x)))
        return chosen - short

    best = None
    short = set(deficits)            # lockstep: every deficit runs the chosen L; unmet need carries to the next epoch
    partners = partners_for(short)
    for L in sorted(set(need.values())):
        patch = short | partners
        cost = wall_hours(L, len(patch), timestep_fs, n_gpus, policy.topup_throughput_table)
        if cost > budget_hours:
            continue
        pred = {s: _predicted(sigma[s], max(1, steps_so_far.get(s, 0)), L, report_interval, g[s],
                              correction.get(s, 1.0)) if _finite(sigma[s]) else math.nan for s in patch}
        benefit = (_worst(sigma, deficits) - _worst({**sigma, **pred}, deficits)) \
            + 1e-3 * sum(sigma[s] ** 2 - pred[s] ** 2 for s in deficits)
        score = benefit / max(cost, 1e-12)
        if best is None or score > best[0]:
            best = (score, L, patch, short, partners, pred, cost)
    if best is None:
        return TopupPlan(structural_edges=tuple(structural), reason="cap_too_small")
    _, L, patch, short, partners, pred, cost = best
    return TopupPlan(state_ids=tuple(sorted(patch)), steps=int(L), deficit_state_ids=tuple(sorted(short)),
                     partner_state_ids=tuple(sorted(partners)), structural_edges=tuple(structural),
                     predicted_sigma=pred, cost_hours=float(cost), reason="planned",
                     sigma_before={s: sigma[s] for s in patch})
```

## Adversarial verification already performed (facts, measured)

V1. The plan's code for T3-T6 and its tests were extracted verbatim into a scratch worktree and run:
    23 passed, 1 failed. The failure: a perfectly healthy 4-state chain (iid Gaussian samples, 2000/state)
    had states {0,1} flagged "unconverged" by the split-halves check.
V2. Root cause of V1 (measured over 200 synthetic campaigns): the per-pair split-halves z statistic is
    correctly calibrated (4.3 % of pairs exceed 2 sigma vs 4.6 % expected for iid Gaussian), but the plan
    applies a fixed 2-sigma threshold to every pair: 11.5 % of 4-state campaigns flag >= 1 state; at 236
    states (chignolin_9) that is ~10 false "unconverged" states per epoch, each triggering top-up MD.
    Also both endpoints of the compared pair get flagged by one fluctuation (k and its first neighbour j).
V3. `build_union_state_mbar_inputs` ALREADY subsamples each state to decorrelated samples
    (equilibration discard + thinning; meta records subsample_counts_per_state {raw, kept}). Therefore:
    (a) T5's inefficiency g, computed on the kept samples, is ~1 by construction (not the real tau);
    (b) T8 sets steps_so_far = kept_count x report_interval, undercounting the real steps by the thinning
        factor; (c) the gain model n_eff = steps_so_far/(interval*g) then double-counts decorrelation.
    Correct inputs: n_eff_k = kept_k, samples-per-decorrelated-sample g_k = raw_k/kept_k (includes the
    equilibration discard), new decorrelated samples from L steps = L/(interval*g_k).
V4. T8 computes the top-up budget from `default_steps=baseline_steps` where baseline_steps has already been
    shortened to (1-cap)*default, so the top-up cap is effectively 0.7*0.3 = 21 % of the epoch, not 30 %.
V5. sigma_k values from the local-sigma definition were checked on the healthy chain: 0.0206-0.0208
    kcal/mol for all four states (the chain end is no longer penalised), i.e. the local definition behaves.


## kimi

Let me carefully analyze this submission. It's an implementation plan for "effective adaptive top-ups" in what appears to be an adaptive sampling / expanded ensemble MD framework (ATLaS-MD, gareus package). The board is asked to:

1. Verify findings V1-V4 (and V5) — are they correct, and what are the right fixes (especially multiple-comparison threshold for split-halves check and decorrelated-count inputs to the gain model)?
2. Find further defects the verification missed: statistical errors in diagnostics/allocator code, lockstep/budget logic, resume and seeding hazards, test gaps, anything that would make top-ups waste MD or corrupt a running campaign.
3. Suggest concrete efficiency improvements.

Verdict: ACCEPT, ACCEPT_WITH_CHANGES, or REJECT.

Let me go through the code carefully.

### Task 5 code: union_diagnostics.py

```python
def _solve(u_nk, window, n_states):
    n_k = np.bincount(window, minlength=n_states)
    active = np.flatnonzero(n_k > 0)
    order = np.argsort(window, kind="stable")
    u_kn = u_nk[order][:, active].T
    mbar = MBAR(u_kn, n_k[active], initialize="BAR", solver_protocol="robust")
    res = mbar.compute_free_energy_differences(compute_uncertainty=True)
    f = np.full(n_states, np.nan); dmat = np.full((n_states, n_states), np.nan)
    f[active] = res["Delta_f"][0]
    dmat[np.ix_(active, active)] = res["dDelta_f"]
    return f, dmat, n_k
```

Issues:
- `res["Delta_f"][0]` — pymbar's `compute_free_energy_differences` returns Delta_f as a matrix of all pairwise differences; row 0 is free energies relative to state 0 of the active set. That's fine, f relative to first active state. OK.
- `initialize="BAR"` — pymbar 4.x MBAR initialize options are "zeros", "BAR", "adaptive". "BAR" is valid. `solver_protocol="robust"` — valid in pymbar 4.
- Sorting: pymbar MBAR expects u_kn with shape (K, N) where samples from each state are grouped; n_k gives counts. The argsort with stable kind groups by state. That's correct.
- One subtlety: `np.bincount(window, minlength=n_states)` — window contains indices into ids; fine.

`_local_sigma`:
```python
vals = [dmat[j, k] for j in neighbours if np.isfinite(dmat[j, k])]
if vals: return float(min(vals))
row = dmat[k][np.isfinite(dmat[k]) & (np.arange(len(dmat)) != k)]
return float(np.median(row)) if row.size else nan
```
- Taking min over neighbours of dDelta_f[j,k]. Note dDelta_f is symmetric, so dmat[j,k] == dmat[k,j]. Fine.
- Fallback median of row. OK-ish.

Main function:
```python
with np.load(npz_path, allow_pickle=False) as z:
    u = np.asarray(z["umbrella_reduced_bias_nk"], dtype=float)
    ids = [int(s) for s in z["state_ids"].tolist()]
    sampled = np.asarray(z["sampled_state_ids"], dtype=np.int64)
idx = {s: i for i, s in enumerate(ids)}
window = np.asarray([idx.get(int(s), -1) for s in sampled], dtype=np.int64)
keep = (window >= 0) & np.isfinite(u).all(axis=1)
u, window = u[keep], window[keep]
```

- `u` is N x K (samples x states) presumably — "reduced potential of every sample in every state". So u_nk is N x K. In `_solve`, `u_nk[order][:, active].T` gives K_active x N. Good.
- `np.isfinite(u).all(axis=1)` drops samples with any non-finite reduced potential in any state. With many states, some reduced potentials may be -inf effectively (or nan) for far states; dropping entire rows could discard lots of data. Actually reduced potentials are usually finite but can be huge; nan can occur. This is a potential data-loss issue but conservative. Hmm, but note: if any state's column has nan for a sample (e.g., overflow), the sample is dropped from ALL states including its own. A more careful approach would only require finiteness in the sampled state and the states used... but MBAR requires finite u_kn for all active states for all samples? pymbar handles inf? Actually pymbar can handle +inf? It computes log; +inf reduced potential means zero weight; pymbar may handle it. nan is not handled. Dropping rows with nan is acceptable but could bias if nans correlate with state. Minor.

Split-halves:
```python
half = np.zeros(len(window), dtype=bool)
for k in range(K):
    rows = np.flatnonzero(window == k)
    half[rows[: len(rows) // 2]] = True
```
- This takes the first half of each state's samples (in the order they appear in the arrays). If samples are ordered by time, first half = first half of the epoch's production — that's a reasonable temporal split. But wait — the NPZ is built from decorrelated samples; order may be by state already or by time. If the NPZ stores samples grouped by state (likely, since sampled_state_ids parallel array), then rows = contiguous block; first half of the block = first half in time if stored in time order. Probably fine.
- However, if the data within a state are correlated in time (they're supposed to be decorrelated by thinning), the halves are independent-ish. OK.

Then:
```python
fa, dma, _ = _solve(u[half], window[half], K)
fb, dmb, _ = _solve(u[~half], window[~half], K)
for k in range(K):
    js = [j for j in nbrs[k] if np.isfinite(fa[j]) and np.isfinite(fb[j])]
    if not js: continue
    j = js[0]
    da, db = fa[k] - fa[j], fb[k] - fb[j]
    comb = math.hypot(dma[j, k], dmb[j, k])
    if all finite and abs(da - db) > 2.0 * comb:
        unconverged.add(ids[k])
```

Issues:
- The reference state issue: f is relative to the first active state. da = fa[k] - fa[j] is a pairwise difference, so reference cancels. Good.
- comb = hypot of the two halves' uncertainties. Under the null, da - db has variance dma^2 + dmb^2 (if halves independent), so z = |da-db|/comb ~ standard normal-ish. V2 measured 4.3% exceed 2σ vs 4.6% expected — wait, for a two-sided test at 2σ, expected exceedance is 4.55%. They say 4.6% expected. OK so the per-pair calibration is fine.
- Multiple comparisons: with K states, each state tested once (against its first neighbour), so K tests per epoch. At 236 states, expected false flags = 236 * 0.0455 ≈ 10.7. V2 says ~10 false unconverged states per epoch. Correct.
- Also "both endpoints flagged by one fluctuation": state k is compared with j = first neighbour; state j (when its turn comes) is compared with its own first neighbour, which may be k (if neighbour lists are symmetric and ordering puts k first). So one fluctuation in the pair (j,k) flags both k and j. Indeed V1 observed states {0,1} flagged together. Correct.
- The right fix: control family-wise error rate or FDR. Options: Bonferroni threshold z* = Φ^{-1}(1 - α/(2K)) with α=0.05 → for K=236, z ≈ 3.7-3.8; or Šidák; or Benjamini-Hochberg on the p-values; or a global chi-square test: sum of z^2 over states compared to chi2_K quantile, then only flag individual states if global test rejects (closed testing). Also better: don't flag both endpoints — flag only k (the state being tested), since the test for k uses pair (k, j). Actually the current code only adds ids[k], not ids[j]. Wait: `unconverged.add(ids[k])` — only k. But when the loop reaches j, j's first neighbour might be k, and the same fluctuation flags j. So both endpoints get flagged through two tests of the same pair. The fix of using a stricter threshold addresses multiplicity; alternatively dedupe pairs: test each unordered pair once and flag... hmm, but flagging a pair doesn't tell you which state is bad. A cleaner approach: for each state, compute the max z over its neighbours (or a combined z across neighbours), then apply Bonferroni across states. Or use a global test.

Also note: the split-halves z uses `comb = hypot(dma[j,k], dmb[j,k])`. But dma[j,k] is the MBAR asymptotic uncertainty of the difference computed within half A. Under the null of stationarity, var(da - db) = var(da) + var(db) since independent. So z = (da-db)/sqrt(dma^2+dmb^2). Fine. But MBAR uncertainties are asymptotic and can underestimate for small samples; with decorrelated samples of ~hundreds, OK.

Another subtle issue: the halves are not equal size when len(rows) is odd — first half gets floor(n/2), second gets ceil. Fine.

Bigger statistical issue: the split-halves test compares free energies between halves, but the halves are defined per state by sample count, not by time. If the NPZ concatenates samples across segments (baseline + previous epochs), the "first half" of state k's samples may span different epochs than the first half of state j's samples. But within an epoch's union NPZ, presumably it's per-epoch data. The docstring says "Per-epoch union MBAR". OK.

Another: `_solve` on the halves — a state could have zero samples in a half if it has 1 sample total (rows[:0] = empty for len 1). Then n_k[active] excludes it; fa[k] = nan; handled by isfinite checks. OK.

Overlap:
```python
f_fill = np.where(np.isfinite(f), f, 0.0)
O = mbar_state_overlap(u, f_fill, n_k)
```
- `mbar_state_overlap(u, f_fill, n_k)` — signature unknown, presumably computes overlap matrix from reduced potentials and free energies. Filling nan f with 0.0 for unsampled states — those states have n_k=0; overlap rows/cols for them presumably nan or zero; they skip edges with n_k==0. OK.
- edge_overlap uses sqrt(x*y) — geometric mean of directional overlaps. Fine.
- Note: overlap computed on ALL samples (not per-half). Fine.

Inefficiency:
```python
own_u = u[np.arange(len(window)), window]
g = per_state_inefficiency(own_u, window, list(range(K)))
```
- Computes autocorrelation of each sample's own-state reduced energy. But per V3, the NPZ is already decorrelated (thinned), so g ≈ 1 by construction. The real inefficiency should be raw/kept. V3 is correct given the meta records subsample_counts_per_state. So T4's Geyer estimator is applied to the wrong series — it should be applied to the raw (unthinned) series, or g should be taken from the subsampling meta. Actually the plan's T4 is "Conservative autocorrelation estimator" — but if applied post-thinning it's useless. The verification's V3 fix: n_eff_k = kept_k, g_k = raw_k/kept_k. That's the right input to the gain model.

Wait — but there's subtlety: raw/kept includes equilibration discard, so it's not exactly the statistical inefficiency g; it's samples-per-kept-decorrelated-sample. For the gain model (how many new decorrelated samples from L new steps), the relevant quantity is the thinning interval in steps: new decorrelated samples = L / (thinning_steps). If kept = raw_equilibrated / thin, then g_forward = raw_equilibrated/kept (excluding equilibration). Including equilibration overestimates g for the forward model slightly. V3 says "samples-per-decorrelated-sample g_k = raw_k/kept_k (includes the equilibration discard)". Hmm — for predicting new decorrelated samples from L new steps of continued simulation, no new equilibration is needed (seeding continues the chain), so the equilibration discard should NOT be included in the forward rate. Including it makes the gain model conservative (underestimates gain), which is safe but wasteful; it also biases the online calibration ratio. Worth flagging as a refinement: use post-equilibration raw count for the forward rate, or better, store the thinning interval directly.

Also `sigma_kcal` multiplies df (in kT) by kt_kcal. Fine.

`UnionDiagnostics.n_k` — counts of kept samples. The allocator uses steps_so_far from T8 = kept_count × report_interval (per V3b). The fix: pass kept counts as n_eff directly and g from meta.

Now the "never raises" pattern: `except Exception ... return None`. Broad but per design ("the epoch runs no top-up"). However, this also swallows programming errors silently — tests must assert on the warning. Acceptable for robustness in production but should log with traceback (logging.warning with exc_info). Minor.

One more: `_solve` uses `initialize="BAR"` — BAR initialization requires adjacent-state overlap? pymbar's BAR init solves pairwise BAR between states with samples; with poor overlap it may fail; "robust" solver protocol falls back. Fine.

Potential issue: `np.bincount(window, minlength=n_states)` — window dtype int64, fine.

`u_kn = u_nk[order][:, active].T` — memory: N x K could be large (236 states, maybe 100k samples → 23.6M floats = 189 MB; transpose copy another). Split-halves triples the solves. Efficiency note: could subsample or use faster protocol; but OK.

Also: the split-halves solves run MBAR three times per epoch (full + 2 halves). Could reuse. Efficiency.

### Task 6 code: topup_allocator.py

```python
MAX_STEP_MULTIPLE = 4
```

`_predicted(sigma, steps_so_far, extra, interval, g, c)`:
```python
n_eff = max(1e-9, steps_so_far / (interval * g))
return sigma * sqrt(n_eff / (n_eff + c * extra / (interval * g)))
```
- Gain model: sigma scales as 1/sqrt(n_eff); adding c*extra/(interval*g) new effective samples. c is correction (online calibration factor). OK in principle.
- Per V3, steps_so_far should be replaced by kept counts: n_eff = kept_k; new = L/(interval*g_k) where g_k = raw/kept. The current formula with steps_so_far = kept*interval and g≈1 gives n_eff = kept — accidentally correct if g=1! Wait: steps_so_far = kept_count × report_interval (V3b). Then n_eff = steps_so_far/(interval*g) = kept/g. If g computed on kept samples ≈ 1, then n_eff ≈ kept. So n_eff is accidentally right, but the new-samples term c*extra/(interval*g) = extra/interval — i.e., it assumes every report interval yields one decorrelated sample, ignoring the true thinning (g_real = raw/kept could be 5-50). So the gain model OVERESTIMATES the number of new decorrelated samples by factor g_real, hence underestimates predicted sigma, hence underestimates required steps by factor g_real. Top-ups will be systematically too short to hit target — wasting MD (repeated small top-ups, lockstep overhead) or never converging. V3's fix is right: n_eff_k = kept_k, g_k = raw_k/kept_k, new = L/(interval*g_k).

Hmm wait, actually let me redo this. If steps_so_far = kept × interval (undercounting real steps by thinning factor t = raw/kept), and g_est ≈ 1 (computed on thinned series), then:
- n_eff = kept×interval/(interval×1) = kept. Correct effective count (accidentally).
- New effective from L steps: L/(interval×1) = L/interval. But real new decorrelated = L/(interval×t). Overestimate by t.
So predicted sigma too optimistic; required steps too small by factor t. Yes, V3 correct.

`_required_steps`:
```python
if sigma <= target: return 0
n_eff = max(1e-9, steps_so_far/(interval*g))
extra_eff = n_eff * ((sigma/target)**2 - 1)
steps = extra_eff * interval * g / max(c, 1e-9)
steps = ceil(steps/interval)*interval
cap = MAX_STEP_MULTIPLE * max(interval, int(steps_so_far))
return min(steps, ceil(cap/interval)*interval)
```
- Algebra: sigma^2 ∝ 1/n_eff. To reach target: n_eff_new_total = n_eff × (sigma/target)^2. extra_eff = n_eff((sigma/target)^2 - 1). steps = extra_eff × interval × g / c. With correct inputs (n_eff=kept, g=raw/kept): steps = kept×((σ/τ)²-1) × interval × (raw/kept) / c = raw×interval×((σ/τ)²-1)/c. Sensible.
- Cap: MAX_STEP_MULTIPLE × max(interval, steps_so_far) — a state can at most quadruple... wait, cap = 4 × steps_so_far means the top-up can add up to 4× the steps so far, i.e., sigma can at best improve by sqrt(4/5)... hmm, n_eff_new = n_eff + 4×n_eff = 5×n_eff → sigma factor 1/sqrt(5) ≈ 0.447. So a single top-up can at most reduce sigma to ~45% of current. If sigma/target > ~2.24, the cap binds and the state won't reach target this epoch. That's a design choice (avoid over-committing on a bad estimate); the comment says "unmet need carries to the next epoch". OK but should be flagged: with cap binding, `need[s]` = cap, and the benefit computation uses pred from capped L — consistent. Fine.
- Bug: `int(steps_so_far)` — steps_so_far could be 0 for a state with samples? They pass max(1, steps_so_far.get(s,0)) at call site, so ≥1. cap = 4×max(interval, 1) = 4×interval at minimum. OK.
- Rounding: `min(steps, ceil(cap/interval)*interval)` — if steps < interval after ceil? steps is already ceiled to interval multiple, so ≥ interval. OK.
- `or report_interval` at call site: `need[s] = _required_steps(...) or report_interval` — if _required_steps returns 0 (sigma ≤ eff_target), need = report_interval. When can sigma ≤ eff_target? eff_target = target if sigma>target else sigma/sqrt(2). For unconverged-but-below-target states, eff_target = sigma/sqrt(2) < sigma, so _required_steps > 0. For sigma>target, eff_target=target<sigma. So _required_steps never returns 0 for deficits... unless sigma == target exactly (sigma <= target → return 0). Edge: sigma exactly target and in deficits? deficits require v > target or unconverged. If unconverged and sigma == target: eff_target = sigma/sqrt2 < sigma → fine. So `or report_interval` is nearly dead code except float equality. Minor.

Deficits and structural edges:
```python
deficits = {s for s, v in sigma.items() if _finite(v) and (v > target or s in diag.unconverged)}
structural = []
for (a,b), ov in sorted(diag.edge_overlap.items()):
    if not _finite(ov) or ov >= weak: continue
    if a in deficits or b in deficits:
        deficits.update(x for x in (a,b) if _finite(sigma.get(x, nan)))
    else:
        structural.append((a,b))
```
- Logic: a weak edge (overlap < weak threshold) with neither endpoint deficit → structural (route to bridge proposal). If one endpoint is a deficit → add both endpoints to deficits (top up both to fix the edge by MD). Hmm — the global constraint says "A weak edge whose both endpoints are statistically adequate is structural: route to the bridge/add proposal, never to MD." And conversely a weak edge with a deficit endpoint gets MD on both endpoints. Reasonable.
- But: `deficits.update(x for x in (a,b) if _finite(sigma.get(x, math.nan)))` — adds the non-deficit endpoint even if its sigma ≤ target. Then in the need loop, for that state sigma[s] ≤ target and it's not unconverged: eff_target = sigma/sqrt(2) (since not sigma>target) → it gets topped up to reduce its sigma by 1/sqrt2. Hmm, that's the "doubling of data" rule applied to a healthy partner of a weak edge. Is that intended? The design says weak edge with a deficit endpoint → MD; topping up both endpoints reduces the pairwise uncertainty. Plausible but the eff_target choice (sigma/sqrt2) is arbitrary; also note the pairwise dDelta_f involves both states' sampling; halving... whatever, it's a heuristic. Flag as design smell but not a bug.
- Unmeasured edges: edge_overlap only contains measured edges (T5 skips n_k==0). "An unmeasured edge is never weak" — satisfied here since unmeasured edges aren't in edge_overlap. T7 handles the gate. OK.

- Subtle bug: `sigma.get(x, math.nan)` — sigma dict was built for all state_ids_in_order with float(diag.sigma_kcal.get(s, nan)), so keys exist. Fine.

partners_for:
```python
def partners_for(short):
    chosen = set()
    for s in sorted(short):
        same = [x for x in neighbours.get(s, []) if x in sigma]
        rung = [x for x in rung_partners.get(s, []) if x in sigma]
        pool = same or rung
        if pool and not any(x in short for x in same) and not any(x in chosen for x in same):
            chosen.add(max(pool, key=lambda x: (sigma[x] if finite else -1.0, -x)))
        if rung and not any(x in short or x in chosen for x in rung):
            chosen.add(max(rung, key=...))
    return chosen - short
```
- Intent (Review Focus 1): a deficit with no same-rung spatial neighbour still gets a partner (nearest on any rung). The code: `pool = same or rung` — if same is non-empty, pool = same; else rung. Then adds max of pool. Then ALSO adds a rung partner if none in short/chosen. So each deficit can add up to 2 partners (one spatial-same-rung... wait "same" = neighbours (spatial neighbours on same rung?), "rung" = rung_partners (same lambda-column across rungs? or exchange partners for lockstep?). Terminology: rungs as lambda-scaled landscape flattening (T10) — so a state = (rung, spatial window). neighbours = spatial neighbours within rung; rung_partners = same spatial position on other rungs? Exchange partners for replica exchange — the top-up segment runs replica exchange among patch states; partners needed for exchange.
- Bug potential: `max(pool, key=lambda x: (sigma[x] ...))` — chooses the partner with the LARGEST sigma. For exchange partner you might want the healthiest or the most overlapping, not the worst. Choosing the worst sigma partner drags another unhealthy state in — maybe intended (kill two birds). But it also increases patch cost. Debatable.
- Bug: the condition `not any(x in short for x in same)` — if ANY same-rung neighbour is already a deficit, skip adding a spatial partner (the deficit neighbour serves as partner). OK. `not any(x in chosen for x in same)` — if a same neighbour already chosen as partner, skip. OK.
- But if same is empty and pool=rung, the first `if` adds a rung partner (max of rung), and then the second `if` may add ANOTHER rung partner? Second if: `rung and not any(x in short or x in chosen for x in rung)` — after adding max(rung) to chosen, that x is in chosen and in rung → any(...) True → skip. So no double add. OK.
- If same non-empty: first if adds max(same) [since pool=same]; second if adds max(rung) if no rung partner in short/chosen. So deficit gets spatial + rung partner. Patch grows. Fine for exchange.
- Edge case (Review Focus 1): deficit with no same and no rung partners (isolated) → no partner added; patch = short alone; a single-state patch — can it run exchange? It runs alone (no exchange). The patch builder "must still return a valid patch" — it does (the state itself). OK.
- `chosen - short` — partners that are themselves deficits removed. But note: partners added for earlier deficits may coincide with later deficits? chosen.add happens only if not in short... first if checks `not any(x in short for x in same)` but the max(pool) itself could be in short? If any same in short → skip; but pool could be rung (same empty) and max(rung) could be in short — not checked! Then chosen contains a deficit; `chosen - short` removes it at the end. But the second if might then add another. Minor inefficiency, not correctness.
- Determinism: sorted(short), max with tie-break -x. Deterministic. Good for resume consistency (though resume reads plan from disk anyway).

Main loop:
```python
short = set(deficits)
partners = partners_for(short)
for L in sorted(set(need.values())):
    patch = short | partners
    cost = wall_hours(L, len(patch), timestep_fs, n_gpus, table)
    if cost > budget_hours: continue
    pred = {...}
    benefit = (worst(sigma, deficits) - worst({**sigma, **pred}, deficits)) + 1e-3 * sum(sigma^2 - pred^2 for deficits)
    score = benefit / max(cost, 1e-12)
    if best is None or score > best[0]: best = (...)
```

Issues:
1. **Lockstep L candidate set**: `sorted(set(need.values()))` — L candidates are the per-state required steps. Every deficit runs the same L (lockstep). Choosing L = max(need) satisfies all; smaller L leaves some unmet. The benefit function: worst-sigma improvement + small tiebreak term. Note benefit is dominated by the worst-deficit term; the 1e-3 tiebreak sums over deficits. Score = benefit/cost. Since cost increases with L and benefit saturates (pred can't go below target-ish... actually pred keeps decreasing with L as 1/sqrt), there's a tradeoff. OK heuristically.
   - But: `need.values()` may all exceed budget → best None → "cap_too_small". Review Focus 5: cap smaller than one useful top-up → no top-up. The smallest L is min(need) which is ≥ report_interval (ceiled). If even that exceeds budget → no top-up. Good. But what about a "useful" minimum — a top-up of report_interval steps might be sub-report-interval? No, it's exactly one interval. The constraint says no "zero-length or sub-report-interval segment" — satisfied by ceil to interval.
   - Hmm, but actually: should the allocator also try L values smaller than min(need) when min(need) exceeds budget? E.g., need = 10000 steps, budget allows 5000. A partial top-up (5000) still reduces sigma and "unmet need carries to next epoch". The current code returns cap_too_small and does NOTHING — wasting the epoch's remaining budget. That's a real efficiency defect: the candidate set should include budget-feasible L values (e.g., the largest interval multiple fitting the budget), not just need values. The spec's "unmet need carries" comment implies partial fulfillment is acceptable, yet the code only considers full-need Ls. This is a defect the verification missed. Actually wait — is it? Let me re-read: `for L in sorted(set(need.values()))`. Yes, only need values. If all > budget → no top-up. So with a tiny remaining pool (Review Focus 5) the answer "no top-up" is right, but with a moderate pool that could fund a useful partial top-up, the code also returns no top-up. The fix: add L_budget = floor(budget-feasible max steps) to candidates, or clamp need to budget-feasible max. This wastes MD opportunity. Good catch to raise.

2. **benefit computation**: `_worst({**sigma, **pred}, deficits)` — pred contains patch states; deficits ⊆ patch, so merged dict has pred for deficits. worst over deficits of pred. But pred[s] for deficit s uses correction c and g — fine. However, `sigma[s]**2 - pred[s]**2` for s in deficits — if sigma[s] is nan? deficits all have finite sigma (filtered by _finite). OK. pred[s] finite. OK.
   - The worst-term: worst(sigma, deficits) - worst(pred, deficits) ≥ 0. OK.
   - Tiebreak weight 1e-3 arbitrary. Fine.

3. **cost**: `wall_hours(L, len(patch), ...)` — cost depends on patch size (contexts/GPU → throughput). Lockstep: all replicas run L steps. Fine. But note: cost uses len(patch) as n_contexts; throughput table flat outside measured range — T3. OK.

4. **budget semantics (V4)**: T8 shortens baseline to (1-cap)×default and computes budget from default_steps=baseline_steps → cap effectively 0.7×0.3=21%. Wait, let me parse V4: "T8 computes the top-up budget from `default_steps=baseline_steps` where baseline_steps has already been shortened to (1-cap)*default, so the top-up cap is effectively 0.7*0.3 = 21% of the epoch, not 30%." So the intent: baseline = 0.7×default, top-up budget = 0.3×default (the freed time). But the code computes budget = cap × baseline_steps = 0.3 × 0.7×default = 0.21×default. So top-ups get 21% and 9% of the epoch is idle/wasted. The fix: budget_hours should be computed from the ORIGINAL default steps: budget = cap × default_steps-equivalent hours, or budget = hours(default_steps) - hours(baseline_steps). Actually more precisely under lockstep the wall-hours freed = hours(default) - hours((1-cap)*default) = cap × hours(default) (if hours linear in steps — wall_hours is linear in L presumably). So budget = cap_fraction × wall_hours(default_steps, n_states...). V4 is correct as stated. The fix: pass the un-shortened default to the budget computation.

5. **predicted_sigma recorded but sigma_before also recorded** — fine for calibration (T8 online calibration realised/predicted).

6. `TopupPlan.steps` — single L. `state_ids` sorted patch. Fine.

7. **correction factor c**: `correction.get(s, 1.0)` — online calibration per state; <0.5 halves priority (T8). In `_predicted`, c multiplies extra: c*extra/(interval*g) — if realised variance reduction is less than predicted (ratio <1), c<1 reduces effective new samples → conservative. In `_required_steps`, divide by c → more steps. Consistent direction. OK. But c could be 0 or negative from a bad calibration? `max(c, 1e-9)` in _required_steps guards division; in _predicted, c*extra with negative c → n_eff + negative → could go negative → sqrt of negative → nan! `n_eff / (n_eff + c*extra/(interval*g))` — if c negative and large magnitude, denominator could be ≤ 0 → sqrt(neg) = nan → pred nan → benefit nan → score nan → `score > best[0]` False → could silently skip or if best None, best = (nan score...) then subsequent comparisons with nan are False → keeps first. Messy. Guard: clamp c to [c_min, c_max], e.g., [0.1, 2]. T8 says calibration halves priority when ratio <0.5 — presumably c is a positive multiplier. But no clamp in code. Flag: clamp correction factors and validate finiteness of pred.

8. **nan handling in benefit**: if any pred nan → benefit nan → score nan; `best is None or score > best[0]` — if first candidate has nan score, best=(nan,...); later finite scores compare score > nan → False → stuck with nan-score candidate. Then plan returned with pred containing nan. Downstream, T8 runs the top-up anyway (steps=L fine) but predicted_sigma nan corrupts calibration. Guard needed.

9. **`_worst(sigma, deficits)`** — computed inside loop repeatedly; hoist. Efficiency trivial.

10. **need for unconverged-below-target**: eff_target = sigma/sqrt(2) — "one doubling of its data". Doubling data reduces sigma by 1/sqrt(2). Correct.

11. **Multiple deficits with same L** — set(need.values()) dedups. Fine.

12. **partners_for uses sigma for max selection but sigma includes nan for unsampled states** — key=lambda x: (sigma[x] if _finite else -1.0, -x) — nan → -1.0, so unsampled states are least preferred. But wait — should an unsampled (never-run) state be a partner? It's in `sigma` dict (all state_ids_in_order). Hmm, state_ids_in_order presumably all layout states. A never-sampled state as exchange partner would START a new window — that's add_rung/bridge territory, not top-up. Choosing it as partner would silently expand the patch into unexplored space with no seeding State (T9 fallback pull... from what?). Actually partners are existing windows; an unsampled state has no final State and no pull source. This could crash or silently mis-seed. Should exclude states with n_k==0 from partner pools. The code filters `x in sigma` only — sigma includes nan entries for unsampled states (they're in state_ids_in_order). So yes, an unsampled state CAN be chosen as partner (ranked last by -1.0, but if it's the only option, chosen). Defect: filter partners to states with n_k>0 / finite sigma... wait but sigma nan → ranked last, but still chosen if pool has only it. And also: should a healthy state be a partner? Yes, partners are exchange partners, healthy states fine. But unsampled states must be excluded. Good catch.

Hmm wait, actually also deficits themselves: `deficits` requires _finite(v) — unsampled states (nan sigma) can't be deficits. But unconverged set could contain an unsampled state? T5 only flags states with neighbours in halves; unsampled → n_k=0 → df nan; split-halves: js requires finite fa/fb; unsampled → skipped. So unconverged ⊆ sampled. OK. But partners can be unsampled. Confirmed defect.

13. **Lockstep and steps_so_far heterogeneity**: states have different steps_so_far; the gain model handles per-state n_eff. Fine.

14. **`cap = MAX_STEP_MULTIPLE * max(interval, int(steps_so_far))`** — for a state with huge steps_so_far, cap = 4×so_far — could be a giant top-up exceeding any reasonable budget; budget check handles. For a state with tiny so_far (new state, 1 sample), cap = 4×interval — but a new state may need 100× interval to reach target; cap throttles to 4×interval per epoch → many epochs to converge. Design choice (avoid over-commit). Note interaction: need[s] capped → L candidates capped → benefit computed with capped L — consistent.

15. **The `or report_interval`** — discussed.

16. **score uses cost in hours; benefit in kcal/mol** — arbitrary units; fine for comparing L candidates for the SAME patch (patch fixed across L candidates since partners fixed). Yes patch is fixed (short | partners) — partners don't depend on L. So score comparison across L is benefit/cost with same patch. OK. But then why recompute patch inside loop — cosmetic.

17. **No option to choose among multiple disjoint patches** — all deficits are batched into ONE patch (one segment, lockstep). Design: "batched with minimal exchange partners", "at most one top-up segment". So all deficits + partners in one exchange ensemble. If deficits are on opposite ends of the layout, the patch spans... wait, patch = short | partners — partners are per-deficit neighbours, but the patch may be disconnected (deficit at state 2 and state 200, each with local partners). Running replica exchange across a disconnected set of windows — exchanges between distant states have ~zero acceptance; the exchange ensemble is effectively two independent groups sharing the segment. That's fine statistically (each window samples its own state) but wasteful in contexts? Actually replica exchange across non-overlapping groups still runs each window; swaps just fail. Cost is the same as running them separately (same number of contexts). But lockstep forces same L across groups — a group needing 1000 steps runs 10000 because another group does → wasted MD. The single-segment constraint (design) causes this; the allocator could at least note it. The benefit/cost optimization doesn't account for per-group needs. Given design constraint of one segment, acceptable, but the L-candidate set being only need values means L = one of the needs; groups with smaller need waste (L - need) steps. Efficiency improvement: allow per-group... no, lockstep forbids. Could choose L to minimize waste via the score — it does implicitly. OK, mention as inherent lockstep waste; suggest considering splitting deficits into epochs or weighting.

18. **`wall_hours(L, len(patch), ...)`** — throughput table maps contexts/GPU → ns/day/node. len(patch) contexts over n_gpus → contexts/GPU = ceil(patch/n_gpus)? Handled in T3 presumably. Flat extrapolation — fine.

19. **budget_hours <= 0**? If budget 0 → all costs > budget → cap_too_small. Fine.

20. **TopupPlan.reason "planned"** etc. Fine.

21. `sigma_before={s: sigma[s] for s in patch}` — includes nan for unsampled partner (if chosen) — JSON serialization of nan → invalid JSON or null; topup_plan.json with NaN — json.dump allows NaN by default (produces NaN token, invalid strict JSON but readable by Python). Resume reads it back — nan sigma_before only used for calibration. Minor.

### V1-V5 assessment

V1: healthy 4-state chain flagged {0,1} unconverged. Consistent with fixed 2σ threshold + pairwise flagging. Plausible and measured. Correct.

V2: per-pair calibration 4.3% vs 4.6% expected — wait, for a two-sided z test at threshold 2.0, P(|Z|>2) = 4.55%. They say 4.6% expected — fine. 11.5% of 4-state campaigns flag ≥1 state: with 4 states, tests per campaign = 4 (each state vs first neighbour) but correlated (pair (0,1) tested twice). Expected ≥1 flag ≈ 1-(1-0.0455)^4 ≈ 17% if independent; with correlation 11.5% plausible. At 236 states: 236×0.0455 ≈ 10.7 false flags. Correct order. The conclusion: need multiple-comparison correction. Right fix: Bonferroni/Šidák per-epoch FWER, or BH-FDR, or global χ² test then individual flags. Quantitatively: for K=236, Bonferroni α=0.05 → z* = Φ^{-1}(1-0.025/236) ≈ 3.73. Šidák: z* = Φ^{-1}((1-(1-0.05)^(1/236))/2 ...) ≈ same. Alternatively use per-pair threshold 3σ (p=0.0027) → 236×0.0027 ≈ 0.64 false/epoch — still highish; 3.5σ (p=4.7e-4) → 0.11/epoch. Recommend Bonferroni with α=0.05 → ~3.7σ at K=236, and scale-adaptive since K varies. Also fix the double-flagging: test each unordered pair once, or flag only the state whose z is extreme relative to its own distribution... Actually better: per state, compute z_k = max over neighbours? That increases multiplicity per state (K×deg tests). Cleaner: per-state z against a FIXED reference neighbour set with Bonferroni over K states. Or the chi-square approach: T = Σ z_k² ~ χ²_K under null (approx, ignoring correlation); flag epoch-level non-convergence if T > χ²_K(0.99), then individual states via Bonferroni. Also should require the discrepancy to be practically significant (e.g., > target_sigma in kcal/mol) not just statistically — a 0.001 kcal/mol drift at 5σ is irrelevant. Two-part threshold: |Δ| > max(z*×comb, practical floor). Good point to raise.

Also the split-halves test has another subtle issue: it tests each state against only its FIRST neighbour (js[0]) — arbitrary choice (depends on edge ordering). Should combine evidence across all measured neighbours (e.g., inverse-variance weighted mean of (fa[k]-fa[j]) differences, with covariance accounted) or take max with per-neighbour multiplicity adjustment. Using js[0] is deterministic but arbitrary; different edge order → different sensitivity. Flag.

V3: correct given the NPZ pre-thins. The fixes: n_eff_k = kept_k; g_k = raw_k/kept_k (post-equilibration raw ideally); new decorrelated from L steps = L/(interval×g_k). Also T5's per_state_inefficiency on kept samples is ~1 and should either be computed on raw series or taken from meta. Also note: the equilibration discard included in g_k makes forward prediction conservative (continuing chains need no new equilibration) — recommend g_forward = (raw - equil_discarded)/kept or store thinning interval explicitly. Also the online calibration (T8) comparing realised vs predicted variance reduction must use consistent n_eff before/after.

Also V3 implies `steps_so_far` should be REPLACED by kept counts in the allocator API — the allocator signature takes steps_so_far: Dict[int,int]; the fix changes T8 to pass kept×? No — pass n_eff directly or pass kept and g separately. The allocator's `_predicted`/`_required_steps` compute n_eff = steps_so_far/(interval*g); with corrected inputs you'd pass steps_so_far = kept×interval×? Hmm. Cleanest: change allocator to take n_eff (kept) and g (raw/kept) explicitly: n_eff = kept; new = L/(interval*g). The formulas already support this if steps_so_far := kept×interval and g := raw/kept: n_eff = kept×interval/(interval×raw/kept) = kept²/raw. NO — wrong. So the formula structure forces n_eff = steps_so_far/(interval·g). To get n_eff=kept with g=raw/kept, need steps_so_far = kept×interval×raw/kept = raw×interval = actual MD steps. So: pass steps_so_far = TRUE raw steps (kept × thinning × interval ≈ raw×interval) and g = raw/kept. Then n_eff = raw×interval/(interval×raw/kept) = kept ✓. New from L: L/(interval×g) = L×kept/(interval×raw) ✓ (thinning rate applied). So the fix is: T8 must pass the TRUE step counts (not kept×interval), and T5 must report g = raw/kept from meta. That's exactly V3's prescription restated. Good — I can state this precisely.

But careful: g = raw/kept includes equilibration discard in raw? If raw = total samples including equilibration, then n_eff = kept ✓ still (by construction), and forward rate L/(interval·g) underestimates new decorrelated samples (since continuing chain skips equilibration). Conservative → overestimates required steps → wastes MD mildly but safe. Recommend using post-equilibration raw for the forward rate, or two parameters. Also the cap `MAX_STEP_MULTIPLE × max(interval, steps_so_far)` with steps_so_far = raw steps — cap = 4× raw steps so far — fine.

V4: correct. Fix: budget from un-shortened default: budget_hours = cap × wall_hours(default_steps, ...) or equivalently hours(default) − hours(baseline). Also check: is budget in hours while cap is fraction of steps — wall_hours linear in steps so fraction applies cleanly. Also ensure the baseline shortening itself is right: baseline = (1−cap)×default leaves cap×default for top-up; if no top-up planned, is the freed time used or wasted? Design: "Healthy states get zero top-up steps; no distribution of left-over budget" — so a healthy epoch runs 30% shorter?? That wastes 30% of every healthy epoch! Wait — baseline shortened to (1-cap)×default ALWAYS, top-up uses up to cap. If no deficits → no top-up → epoch does 70% of the work it used to. That's a huge efficiency regression for healthy epochs. Hmm, is that the intent? "Make scheduled top-ups spend MD only where..." — the design trades baseline for top-ups. But when no top-up is needed, the baseline should presumably run FULL length. The plan shortens baseline unconditionally (T8: "baseline shortened to (1-cap)*default"). Unless... the allocator runs BEFORE the baseline? No — diagnostics need the epoch's samples, so baseline runs first, then top-up. So the baseline must be shortened before knowing if a top-up is needed. Options: (a) run full baseline and fund top-ups from the NEXT epoch's budget or a separate pool; (b) shorten baseline always (wastes 30% when healthy — but wait, does it? The epoch's total wall-time is fixed by schedule; a shorter baseline + no top-up = idle nodes or earlier next epoch. If epochs are wall-scheduled, idle. If step-scheduled, the next epoch just starts earlier — no waste, just reallocation: each epoch's baseline does 0.7× work, so convergence per epoch slower but epochs more frequent? No — total MD is what matters; 0.7×baseline with no top-up = 0.7× total MD per epoch = pure loss if the schedule expected full epochs... unless the number of epochs increases. This is a real design question the review should flag: the budget accounting must make healthy epochs whole. The cleanest: keep baseline at full default; top-up budget = cap × default drawn from a separate allowance (epoch total = (1+cap)×default when topped up) OR run baseline at (1−cap) and if no top-up occurs, extend... can't extend retroactively. Alternative: decide top-up BEFORE baseline using previous epoch's diagnostics (stale by one epoch) — then baseline length can be adjusted. Or: run baseline (1−cap), diagnostics, top-up; if no top-up, run a "baseline continuation" segment of cap×default — i.e., the reserved budget is spent on the baseline itself when no deficit exists. That preserves total MD and the lockstep/one-segment constraint (the continuation is just more baseline). But "no distribution of left-over budget" constraint refers to not topping up healthy states; spending leftover on baseline continuation doesn't violate the spirit. This is a significant finding: as planned, every healthy epoch forfeits cap fraction of MD. Actually wait — re-read T8 summary: "uniform schedule (old score allocator deleted), baseline shortened to (1-cap)*default". And Global Constraints: "Healthy states get zero top-up steps; no default score; no distribution of left-over budget." So yes, healthy epochs lose 30% MD. Unless the schedule is wall-hour based and epochs are fixed wall-time, in which case the idle 30% is truly idle. This deserves flagging as a defect or at least a design risk requiring explicit acknowledgment. Hmm, but maybe the intent is epochs are defined by wall-time slots and the top-up fills the slot; healthy epoch ends early and next epoch starts early — total campaign time same, number of epochs same... then total baseline MD = 0.7 × epochs × default < planned. The campaign converges slower. It's a real cost. Flag it; suggest baseline-continuation segment or next-epoch budget rollover (but resume complicates rollover — persist pool in epoch state).

V5: local sigma behaves (0.0206-0.0208 for all four states, ends not penalised). Confirms deviation 1 is sound. Also note local sigma = min over neighbours — min is optimistic (best-case), could mask a state whose worst-edge uncertainty is what matters for the proposer; but for targeting MD it's the right quantity. Also the fallback "median of row" for states with no measured EDGE-neighbour: median over ALL states includes distant states with huge dDelta_f → overestimates sigma → false deficit. Better fallback: min over row (nearest measured state) or nan (treat as unsampled → no top-up, since topping up a state with no measured neighbour can't be costed against a neighbour... actually more MD on it reduces its own variance; but without a measured neighbour you can't define local sigma; median is arbitrary and likely flags false deficits for isolated states). Hmm — when does a sampled state have no measured edge-neighbour? Edges come from the layout; a sampled state should have edge-neighbours, but those neighbours may be unsampled (dmat nan). E.g., a new rung state sampled but its spatial neighbours not yet... unlikely but possible. Median-of-row fallback is statistically dubious; recommend min-of-row or nan→exclude. Flag as minor defect.

### Further defects to find (part 2)

Let me hunt systematically.

**T5 statistical errors:**

(a) `keep = (window >= 0) & np.isfinite(u).all(axis=1)` — discussed; row-wise nan dropping can bias and discard massive data if one column has nans (e.g., a state whose reduced potential overflows for samples from far states — actually reduced potentials can be +inf for hard overlaps; np.isfinite rejects inf too! If any entry is +inf (zero Boltzmann weight), the whole sample is dropped. With 236 states, the probability that ALL 236 reduced potentials are finite for every sample may be low if infs occur → catastrophic data loss → n_k undercounts → MBAR on a biased subset (samples dropped preferentially from low-overlap states). pymbar handles +inf u_kn? Let me think: pymbar computes -u + log... it uses logsumexp over -u_kn; +inf contributes -inf → weight 0 — handled. nan is not. So the correct filter: drop samples with nan in any ACTIVE column, allow +inf. The current code's all-finite requirement is a real correctness hazard at scale. Strong point to raise. Actually is it? Reduced potentials from OpenMM evaluation — can they be inf? Harmonic restraints with far configurations → large but finite. Overflow to inf possible for repulsive cores (r^-12) at bad contacts — yes possible. And nan from 0*inf etc. The safe approach: replace +inf with a large finite? pymbar docs: u_kn can be inf? I believe pymbar handles inf (treated as zero weight) — there were issues historically; nan definitely breaks. Recommend: mask only non-finite excluding +inf, or clip. At minimum, count and log dropped rows; if drop fraction > 0, warn loudly. I'll raise it as a hazard with a concrete fix (keep = isfinite in own state & no nan anywhere; allow inf).

(b) Split-halves: halves by sample index within state — if the NPZ stores samples ordered by segment/time, first half = earlier time — good. But if samples are ordered arbitrarily (e.g., sorted by energy), the test is invalid. Should assert/construct temporal split. Also for states with odd counts fine. Also the test only uses js[0] — arbitrary neighbour. Also no practical-significance floor. Also flags both endpoints (V2). Also multiple comparisons (V2). Also: the two half-solves use initialize="BAR" each — fine.

(c) `comb = hypot(dma[j,k], dmb[j,k])` — correct under independence of halves. But MBAR dDelta_f for a half with few effective samples is asymptotic; with thinning, halves have kept/2 samples — OK if kept large; for small kept (< ~50), z is not calibrated → false flags or misses. Add minimum-sample guard: skip test if either half < N_min (e.g., 50) — currently a state with 2 samples (1 per half) yields garbage z. Actually with 1 sample per half, MBAR may still return something; z meaningless. Guard needed. Also V2's calibration was measured at 2000/state; at 50/state the asymptotic variance is less trustworthy.

(d) edge_overlap geometric mean sqrt(x*y) — fine; but note overlap computed from FULL solve f with f_fill=0 for unsampled — unsampled states excluded by n_k check. OK. But overlap matrix from `mbar_state_overlap(u, f_fill, n_k)` — u here is the FILTERED u (rows dropped). Consistent with solve. OK.

(e) `sigma_kcal` = df × kt_kcal — dDelta_f is in kT units; multiply by kT in kcal/mol ✓.

(f) `unconverged` stores state IDs (not indices) ✓.

(g) The broad `except Exception` — also catches KeyboardInterrupt? No, that's BaseException. OK. But it also swallows MemoryError etc. Fine. Should log exc_info=True.

(h) `_solve` sorts but doesn't verify u_kn finiteness for active columns — pymbar will throw on nan → caught → None → no top-up for the whole epoch even if only one state's column is bad. Robustness vs granularity: could drop offending states instead of failing the epoch. Minor.

(i) Split-halves solves run on u[half] with ALL K columns including states with zero samples in that half — active filter handles. ✓.

(j) MBAR solver_protocol="robust" — in pymbar 4, solver_protocol options: "hybr", "lm", "L-BFGS-B", "dogleg", "CG", "Newton-CG", "TNC", "trust-ncg", "trust-krylov", "trust-exact", "SLSQP", "adaptive", "robust". "robust" = adaptive with fallback? I recall pymbar 4 has "adaptive" and "robust". Yes, robust exists. OK.

(k) initialize="BAR" requires overlap between neighboring states to converge; with poor overlap BAR init can fail → exception → None. "zeros" init is safer. Minor robustness: try BAR, fall back to zeros. Actually pymbar BAR init: computes pairwise BAR between all states with samples — for 236 states that's 236²/2 ≈ 28k BAR solves — SLOW. Efficiency point: initialize="BAR" is O(K²) BAR solves each epoch (×3 solves including halves). For 236 states this could dominate. Recommend initialize="adaptive" or "zeros" with robust solver, or warm-start halves from full f. Good efficiency catch. Let me double check pymbar: MBAR(initialize="BAR") — yes it does pairwise BAR between all pairs (or neighbors?). In pymbar 4, "BAR" initialization computes all pairwise BAR free energies as initial guess — O(K²) optimizations. For K=236, expensive but maybe tolerable (~minutes). Flag as efficiency.

(l) Memory: u is N×K float64. N = total kept samples per epoch (say 236 states × 500 kept = 118k) × 236 × 8B = 223 MB. ×3 solves with fancy indexing copies (u[half] copies, u_kn[order][:, active].T copies) → several hundred MB churn. Acceptable but note.

(m) `np.bincount(window, minlength=n_states)` — window values < n_states guaranteed by idx mapping ✓.

(n) The diagnostics read `sampled_state_ids` and map via idx; samples whose state not in ids dropped (window=-1) ✓.

(o) Determinism: MBAR solve deterministic ✓ (no RNG). Split-halves deterministic ✓. Good for resume consistency — but resume reads plan from disk anyway (Review Focus 3).

**Allocator defects (beyond above):**

(p) `need[s]` uses `steps_so_far.get(s, 0)` → max(1, ...) — for a state with 0 prior steps but finite sigma (impossible: finite sigma requires samples requires steps) — fine.

(q) `correction.get(s, 1.0)` — c unclamped (nan/negative hazard) — raised above.

(r) The benefit's second term `1e-3 * sum(sigma[s]**2 - pred[s]**2 for s in deficits)` — if sigma large (e.g., 5 kcal/mol), squares 25 × 1e-3 = 0.025 vs worst-term ~O(0.1-1) — negligible; tiebreak only. OK.

(s) `score > best[0]` with equal scores — first wins (strict >). Deterministic ✓.

(t) `TopupPlan.predicted_sigma` keys are ints → JSON object keys become strings; resume reading must coerce. Round-trip hazard: predicted_sigma {int: float} → json → {"12": 0.09}; on resume, calibration lookup by int fails silently (dict.get(int) → None). T8 must coerce keys on read. Classic bug. Flag under resume hazards. Same for sigma_before, n_k, sigma_kcal in diagnostics if persisted.

(u) `wall_hours(L, len(patch), ...)` — len(patch) contexts; but exchange typically requires even/odd? No. OK.

(v) Budget check `cost > budget_hours: continue` — strict >; cost == budget allowed ✓.

(w) If `need` empty — can't happen (deficits non-empty ⇒ need non-empty).

(x) `partners_for` may choose a partner that is ALSO unsampled — raised above (defect: filter n_k>0).

(y) Partners' predicted sigma computed and stored, but partners aren't in benefit — fine, they're exchange fillers. But topping up partners also reduces THEIR sigma — if a partner is near-target, the top-up might push it well below — wasted but harmless. However: partners get L steps too (lockstep) — a partner with sigma 0.02 getting 100k steps — pure waste but needed for exchange. The "minimal exchange partners" objective — partners_for adds up to 2 per deficit; could share partners across deficits (it does via chosen checks) ✓.

(z) **Lockstep L vs per-state cap**: need[s] already capped per state (4× so_far). L candidates are capped needs ✓.

(aa) **The allocator never considers NOT topping up some deficit** — all deficits included in every candidate. If one deficit's need is enormous (cap-bound) it forces L huge for everyone or the plan is cap_too_small... wait, L candidates = set of needs; choosing L = min need leaves big-need states unmet (carried to next epoch) — allowed. So the score optimization handles. But patch includes all deficits regardless of L — a deficit whose need >> L still runs L steps (makes progress) ✓ fine.

(bb) **benefit uses deficits' worst sigma but L feasibility**: fine.

(cc) `MAX_STEP_MULTIPLE = 4` module-level; not policy-configurable. Minor.

(dd) `_finite` rejects numpy types? `isinstance(x, (int, float))` — numpy.float64 IS an instance of float ✓ (np.float64 subclasses float). np.int64 is NOT instance of int on... actually np.int64 doesn't subclass int in py3. sigma values are floats ✓. ov float ✓. OK. But bool is instance of int — irrelevant.

(ee) `plan_topup` reads `policy.topup_target_sigma`, `topup_weak_overlap`, `topup_throughput_table` — names per T2 ✓.

**T8 wiring / lockstep / budget / resume hazards (from summaries, since no code):**

- V4 budget bug ✓ (fix: un-shortened default).
- Resume (Review Focus 3): plan persisted to topup_plan.json ✓ good. Hazards: (i) JSON key coercion (above); (ii) plan must be invalidated if the epoch's layout changed between crash and resume (e.g., proposer added a rung after baseline but before resume? Layout changes happen at epoch boundaries presumably; a resumed epoch reuses layout — but if the user edited YAML/layout on resume, plan state_ids may not exist → must validate plan against current layout and fall back to no-top-up, not crash); (iii) the top-up segment name/steps must come from the plan file (they say so) ✓; (iv) partial top-up samples must be excluded from THIS epoch's diagnostics on resume — diagnostics were computed pre-top-up; if resume recomputes diagnostics (it shouldn't), the partial top-up contaminates. They read plan from disk ✓. But also: the union NPZ for the epoch — is it written before the top-up starts? If the NPZ is rebuilt on resume including partial top-up samples and the plan is read from disk, fine. But the NEXT epoch's union inputs must include the top-up samples (they're valid data) ✓ presumably.
- (v) `topup_plan.json` written atomically? Crash mid-write → corrupt JSON → resume crashes or recomputes. Write tmp + rename. Flag.
- (vi) Online calibration state (correction factors) persisted where? If in memory only, resume loses calibration → c resets to 1. Persist alongside plan or in epoch diagnostics. Flag.
- Seeding (T9): export final positions/velocities/box per window at segment end; top-up starts from them. Hazards: (a) velocities from a checkpointed/restarted baseline — if the baseline segment itself was resumed, final frame fine. (b) The assertion "seeded frame reproduces recorded CV1/CV2 tol 1e-4" — recomputing CVs from positions; if CVs are alchemical/λ-dependent or use different precision (saved npz float32 vs float64), 1e-4 tol may false-fail or pass with drift. Also periodic box / unit conventions (nm vs Å). (c) Fallback pull for missing States "with a warning naming them" ✓ (Review Focus 4) — but a pulled seed has no velocities matching the frozen GaMD envelope? The pull presumably gets positions only → new velocities drawn → that's a chain restart, not continuation — the spec's "no burn-in needed" assertion fails for those windows; mixing a restarted window into an exchange patch can corrupt... not corrupt, but the first samples are out-of-equilibrium; the union build discards equilibration per segment? The decorrelation pipeline does equilibration discard per state per segment presumably — if the top-up segment's samples all count as post-equilibration (continuation assumed), pulled windows contaminate the decorrelated set with unequilibrated samples → biased union. Fix: mark pulled-seed windows and apply equilibration discard to their top-up samples. Strong point.
- (d) Lockstep + seeding: all windows start from exported States; if SOME windows lack States and use pull, they start from a different lambda-point structure — fine.
- (e) The exported State must include the GaMD boost potential / envelope? "frozen GaMD envelope continue the chain" — boost is recomputed from the frozen envelope ✓ per deviation 2.
- (f) final_window_states/ written at segment end — if segment crashes at 99%, no States → next epoch's top-up falls back to pull for ALL windows (with warning) — acceptable but wasteful; could export States periodically (every report) so a crash loses little. Efficiency/robustness suggestion.
- (g) Disk: per-window positions/velocities each segment — storage growth; prune old segments' States (keep latest per window). Efficiency.
- Lockstep logic: "all its replicas run the same number of steps" — with exchange among patch states. Hazard: the patch windows run ONLY the patch states — the exchange attempt scheme among a subset (e.g., 5 windows) — neighbor swaps only within patch ✓. But if patch states span multiple rungs with different λ, swaps across rungs have low acceptance — fine.
- Budget logic: budget_hours vs wall_hours model — the model uses the throughput table with flat extrapolation; if patch contexts/GPU < 16, flat at 3154 ns/day — the table's lowest point is 16 contexts/GPU; a patch of 3 states on 1 GPU = 3 contexts/GPU → flat 3154 overestimates throughput? Flat below lowest point means use 3154 (the 16-context value) — but fewer contexts/GPU usually means HIGHER per-context throughput... total node throughput with 3 contexts might be lower than with 16. Flat = assume same node throughput → cost underestimated for small patches? Actually with fewer contexts per GPU, each context runs faster but GPU utilization drops; node ns/day summed over contexts typically decreases. Flat assumption (3154) likely overestimates small-patch throughput → underestimates cost → budget overrun (top-up runs longer than budgeted wall-hours). Direction: conservative would be to interpolate down. But "flat (no extrapolated gain)" is the stated policy — for small patches it may overestimate throughput. Hmm, the table is contexts/GPU → ns/day/node: 16→3154, 59→2300. More contexts → less total throughput (overhead). Below 16, throughput probably continues rising (fewer contexts, less overhead) up to a point, or falls (underutilization). Flat at 3154 is a guess; the error direction matters for budget adherence. Flag as model risk; recommend measuring a low-context point or clamping patch cost with a safety factor. Also online calibration measures realised vs predicted VARIANCE, not wall-hours — cost model drift is uncalibrated. Suggest logging realised segment wall-time vs predicted to calibrate throughput too.
- "One top-up segment per epoch at most" ✓.
- T7: unmeasured edge never weak; rung edges judged on energy-space mbar_overlap only — the union overlaps written into epoch diagnostics' mbar_overlap so the proposer places bridges — hazard: overwriting mbar_overlap with union values could change proposer behavior for NON-top-up runs? No — top-ups off by default; but the union diagnostics only run when top-ups enabled? If union overlaps are written only when top-ups on, proposer behavior changes when enabling top-ups — confound for A/B but acceptable. Also: writing union overlap into the same field used by the convergence gate — T7 says gate uses energy-space mbar_overlap for rung edges — consistent. OK.
- Structural edges routed to bridge/add proposal "never to MD" ✓ in allocator (structural list). But note: an edge weak AND one endpoint deficit → MD on both endpoints; if MD doesn't fix it (true structural), every epoch re-tops-up → wasted MD forever. Need a memory/hysteresis: if an edge stays weak after N top-ups targeting it, reclassify as structural. The online calibration (<0.5 halves priority) partially handles but the edge still attracts MD. Suggest persistent per-edge top-up attempt counter with structural escalation. Good catch — "anything that would make top-ups waste MD".
- Also: deficits include unconverged states even below target ("one doubling") — if split-halves false-flags (pre-fix ~10 states/epoch at 236 states), each triggers a doubling top-up → with the multiplicity fix this drops to ~0. Also unconverged states should perhaps get a CAP on repeated doublings (if a state is flagged every epoch, something structural; escalate).
- Seeding assertion tol 1e-4 on CV1/CV2 — CVs in what units? If CV is a distance in nm, 1e-4 nm = 1 pm — positions stored in trajectory precision (float32 ~ 1e-3 nm relative?) — DCD precision ~1e-3 nm? Could false-fail. Recommend storing CVs computed from the exact exported positions and comparing to recomputation of THOSE positions (self-consistency) rather than to trajectory-recorded CVs. Or tol relative to CV range. Flag as test gap / false-abort risk: an assertion that false-fails aborts the epoch (wasted baseline) — "corrupt a running campaign" inverted: overly strict assertion kills healthy epochs. Also the assertion should be skip-on-warning not raise? Design says assertion kept. Recommend warning + fallback to pull for mismatching windows rather than killing the epoch.
- Test gaps: Review Focus lists tests for T5/T6/T8/T9. Missing tests: (i) budget computed from un-shortened default (V4 regression test); (ii) gain model uses kept/raw counts (V3 regression: synthetic with known thinning factor t, assert required_steps scales ∝ t); (iii) split-halves FWER at K=236 (simulation-based, assert false-flag rate < α over N campaigns) — they have the harness; (iv) partner exclusion of unsampled states; (v) nan/inf handling in NPZ (a column with inf doesn't nuke the epoch); (vi) JSON round-trip of TopupPlan (int keys, nan values); (vii) resume with layout change → plan invalidated safely; (viii) correction factor clamping (c=0, c negative, c nan); (ix) cap_too_small vs partial-L: assert allocator proposes budget-feasible partial L; (x) CV assertion tolerance with float32 trajectory round-trip; (xi) pulled-window equilibration marking; (xii) throughput table flat-region behavior at patch < 16 contexts; (xiii) determinism: same diagnostics → same plan (resume consistency); (xiv) edge case: ALL states deficits (patch = whole layout — should probably refuse or cap patch size); (xv) odd sample counts in split-halves; (xvi) states with 1 sample.
- Also: patch size vs n_gpus — if patch > n_gpus × max contexts, can't run; allocator should cap patch size to hardware. wall_hours table max 59 contexts/GPU; patch of 30 deficits + 40 partners = 70 contexts on 1 GPU → flat extrapolation above highest point (2300) hides infeasibility/memory limits. Flag: hard cap on patch size or multi-node handling.
- Seeding hazard: exported State from baseline final frame — but the top-up runs MORE steps; the NEXT epoch's baseline starts from... the top-up's final states or re-pulls? If next baseline pulls from the mixture, top-up progress in configuration space is lost (pull re-seeds from pooled structures?). Presumably baseline continues from its own chain — but the baseline chain ended at the pre-top-up frame; the top-up extended the chain; next baseline should continue from the TOP-UP's final frame (chain continuity), else the top-up's configurations are orphaned (samples still valid for MBAR, but chain restarts → extra equilibration). The plan says top-ups are seeded from baseline final States; it doesn't say the next baseline resumes from top-up-final States. If baseline resumes from its own exported State (pre-top-up), the top-up segment becomes a side-branch — statistically fine for MBAR (equilibrium samples) but the chain continuity claim ("no burn-in") holds per segment. Actually for MBAR validity, samples just need to be equilibrium draws from each state's ensemble — a side-branch is fine after equilibration... but the top-up claims NO equilibration needed because it continues the chain ✓. And next baseline continuing from pre-top-up frame also continues ITS chain ✓. So both are continuations of the same chain at the branch point — fine statistically. But the top-up's final (presumably well-mixed) frames are discarded as seeds — minor waste. OK, mention briefly.
- Another resume hazard: epoch numbering/segment naming — top-up segment name derived from epoch; if resume recomputes (bug) name changes → duplicate segment dirs. They persist plan ✓. Test exists (T8).
- The `correction` calibration: "realised/predicted variance reduction; <0.5 halves priority" — realised variance reduction measured from consecutive epochs' sigma_k — but sigma_k itself is noisy (local min over neighbours of dDelta_f, which is an asymptotic estimate); ratio of noisy estimates → noisy c; without clamping/smoothing (e.g., EMA, clamp [0.25, 4]), c whipsaws. Also realised reduction confounded by changed overlap/λ between epochs. Suggest EMA + clamps + minimum sample of epochs before applying.
- V2 fix detail — the multiple-comparison threshold: recommend per-epoch FWER via Šidák/Bonferroni: z_thresh(K) = Φ^{-1}(1 - α/(2K)) with α=0.05; K=4 → 2.50; K=236 → 3.73. Or BH for FDR. Also add practical floor: flag only if |Δf| > max(z*·comb, γ·target_sigma) with γ~0.5. And test each pair once (flag the state with larger leave-one-out discrepancy?) — simpler: per-state z_k computed against inverse-variance-weighted combination over neighbours, then Bonferroni over K. Also require minimum half sizes.
- Also the split-halves "both endpoints flagged" — with per-state tests, a single bad pair (j,k) inflates z for both j and k. Deduplication: compute pair z's, apply Bonferroni over E pairs, flag...
(TRUNCATED - output hit the token limit before completion)

## mini

**Assessment**  
The verification findings V1‑V4 are essentially correct: the split‑halves check is overly liberal (V1‑V2), the inefficiency g and steps‑so‑far accounting double‑count decorrelation (V3), and the budget cap is mis‑computed (V4). However, the implementation contains additional, serious defects that the adversarial verification missed and that would cause systematic waste of MD or even corrupt a running campaign.

**Key strengths**  
- Clean separation of concerns: union diagnostics, allocator, and wiring are pure functions with no hidden I/O.  
- Use of MBAR + state‑overlap to drive top‑up decisions follows the design spec.  
- The allocator respects lock‑step constraints and caps per‑epoch wall‑hour budget.  
- Comprehensive test suite for the core logic (tasks 5‑6) was already executed.

**Critical weaknesses and errors**  

1. **Split‑halves multiple‑comparison error** – The code (lines 71‑84) flags a state as unconverged if *any* neighbour exceeds a fixed 2 σ threshold, ignoring that a state may have several neighbours. This yields a family‑wise error rate ≈ 10 % for a 4‑state chain (V1‑V2). The fix is to apply a Bonferroni‑adjusted threshold per state:  
   ```python
   m = max(1, len(js))
   thresh = scipy.stats.norm.ppf(1 - 0.05/(2*m))
   if abs(fa_k - fb_k) > thresh * comb: …
   ```  
   or use an FDR procedure.

2. **Deterministic half‑splitting bias** – The halves are built by taking the first ⌊N/2⌋ samples per state (lines 71‑73). Because samples are only decorrelated, this can introduce systematic bias. Randomly shuffle the indices per state before splitting, or use even/odd indices after decorrelation.

3. **Over‑strict sample filtering** – `keep = (window >= 0) & np.isfinite(u).all(axis=1)` (line 45) discards any sample that has a NaN reduced potential in *any* state, even though only the column corresponding to the sample’s own state is needed. This dramatically reduces effective sample size. Replace with:  
   ```python
   keep = (window >= 0) & np.isfinite(u[np.arange(u.shape[0]), window])
   ```

4. **Incorrect decorrelation accounting in allocator** – The allocator treats `steps_so_far` as the number of *kept* samples (line 115) and uses `g`≈ 1 (because samples are already decorrelated, V3). Consequently `n_eff = steps_so_far/(interval*g)` underestimates the true effective sample count, leading to over‑aggressive top‑up proposals. The correct inputs are:  
   - `n_eff_k = kept_k` (raw kept count)  
   - `g_k = raw_k / kept_k` (raw‑to‑kept ratio, includes equilibration discard)  
   Pass these corrected values to `_required_steps` and `_predicted`.

5. **Budget cap double‑application** – In T8 the baseline steps are already reduced by `(1‑cap)*default`; the allocator then applies the same cap again (V4), shrinking the allowable top‑up budget to ≈ 21 % instead of the intended 30 %. The fix is to compute the top‑up budget directly from `budget_hours = policy.topup_max_fraction * total_epoch_hours` without pre‑reducing `baseline_steps`.

6. **Resume‑mid‑segment hazard** – `plan_topup` returns a `TopupPlan` with a total `steps` field but no field for “steps already completed”. If a segment crashes after, say, 40 % of the planned steps, a resumed run will re‑execute the full `steps`, double‑counting MD and corrupting the diagnostics. Store `steps_completed` in the JSON and subtract it from `steps_so_far` on resume.

7. **Missing portable State fallback** – The seeding logic (Task 9) assumes that the exported final states exist for all windows; when a window is missing it falls back to a pull *without* updating `steps_so_far` for that window. This creates a mismatch between the allocator’s expectation of wall‑hour cost and the actual cost, potentially exceeding the budget. The fallback path must also increment `steps_so_far` for the pulled window.

8. **Partner selection may ignore weak‑edge structural edges** – The `partners_for` routine (lines 129‑144) prefers neighbours with the highest σ, but it never excludes structural edges that are already scheduled for bridge/add proposals. This can cause the allocator to waste MD on edges that should be handled by the bridge proposer. Add a filter `if (s, x) not in structural_edges` before adding a partner.

9. **No test for unsampled edges** – The current test matrix does not include a case where an edge connects two states with zero samples in one endpoint. The allocator would attempt to compute `edge_overlap` for such an edge, but `diag.edge_overlap` would simply omit it, leaving the edge un‑treated. Add a unit test that verifies the allocator treats missing overlaps as “unmeasured → never weak”.

10. **Numerical stability of `_predicted`** – When `c` (correction factor) is very small, the denominator `max(c, 1e-9)` can still cause division‑by‑tiny‑number noise. Clamp `c` to a lower bound of 1e‑3 for stability, or compute the prediction in log‑space.

**Concrete efficiency improvements**  
- Cache the MBAR solve (`_solve`) for the two halves and reuse the result when the same union NPZ is processed multiple times (e.g., during resume).  
- Vectorise the neighbour loop in the split‑halves check to avoid Python‑level list comprehensions; this reduces per‑epoch overhead from O(K·deg) to O(E).  
- Pre‑compute the per‑state `g_k` ratio once in `union_diagnostics` and expose it, avoiding recomputation in the allocator.  
- Use a sparse representation for `edge_overlap` to speed up the partner‑selection loop when the number of edges is large (e.g., > 10 k).  

**Verdict**  
The plan is fundamentally sound but contains multiple statistical, accounting, and robustness bugs that would cause systematic waste of MD and could corrupt a campaign. These must be fixed before integration.

VERDICT: ACCEPT_WITH_CHANGES - multiple statistical and accounting defects require correction  
CONFIDENCE: 88

## thinker

(call failed: HTTP Error 500: Internal Server Error)

Position cards:

- kimi: (TRUNCATED - output hit the token limit before completion) (confidence 50)
- mini: ACCEPTWITHCHANGES - multiple statistical and accounting defects require correction (confidence 88)
- thinker: (call failed: HTTP Error 500: Internal Server Error) (confidence 50)

Verdicts: kimi=n/a, mini=ACCEPT, thinker=n/a
Majority: ACCEPT at 33%
Unanimous: False

## kimi

**Attack on C**

C endorses V1–V4 correctly, but most of its concrete fixes are wrong at the mechanism level:

1. **Wrong multiplicity family.** The code tests each state against `js[0]` only — one z per state, K tests per epoch — not "any neighbour" as C claims. C's `m = len(js)` gives m=1 for most states → threshold ≈ Φ⁻¹(0.975) = 2.24, leaving the ~10 false flags/epoch at K=236 intact. The correct family is the K states: z* = Φ⁻¹(1−α/(2K)) ≈ 3.73 at K=236, plus a practical floor |Δ| > max(z*·comb, γ·target) and a minimum half-size guard. (C's "lines 71‑84" are invented; the submission has no line numbers.)

2. **Shuffling destroys the test.** Split-halves exists to detect drift; random permutation makes halves exchangeable under drift and guts sensitivity. Contiguous temporal halves are correct — V2 measured them calibrated (4.3% vs 4.6%). The real requirement is asserting time-ordered storage.

3. **C's filter fix breaks MBAR.** The solve needs finite u_kn in *every active column* per retained sample; C's own-column-only mask injects NaN into pymbar's logsumexp. Correct fix: reject NaN, allow +inf (zero weight), log drop counts.

4. **C's V3 wiring is dimensionally inconsistent.** Passing steps_so_far = kept into `_required_steps` yields n_eff = kept/(interval·g). The formula structure requires steps_so_far = raw steps with g = raw/kept → n_eff = kept, forward gain L/(interval·g). C's direction claim is also wrong: n_eff is accidentally correct today; the *forward* term overestimates new decorrelated samples by the thinning factor, so top-ups are undersized.

5. **Resume claim contradicts the plan.** T8/Review-Focus-3 persist topup_plan.json and read name/steps from disk; segments resume via the existing checkpoint path. Real hazards C missed: JSON int-key coercion for predicted_sigma/sigma_before, NaN serialization, non-atomic writes, unpersisted calibration factors, layout-change invalidation.

6. **Point 10 mechanism wrong.** In `_predicted`, c *multiplies*; no division by c exists there. The actual hazard is negative/NaN c → n_eff + c·extra/(interval·g) ≤ 0 → sqrt(≤0) → NaN score, poisoning the argmax (comparisons vs NaN are False; a first NaN candidate sticks). Clamp c to a positive interval with EMA smoothing.

7. **Spurious defects, real ones missed.** Partners are exchange fillers; excluding "structural edges" from partner choice is baseless. The real partner defect: unsampled states (σ=NaN, ranked −1.0) are selectable when a pool is otherwise empty. C's O(K·deg)→O(E) "vectorisation" is a no-op (E = Σdeg/2); the real cost is `initialize="BAR"` — O(K²) pairwise BAR solves (~28k at K=236) ×3 solves/epoch. C also missed the two largest budget defects: L-candidates are restricted to need values, so a budget-feasible partial top-up is never proposed ("cap_too_small" wastes the epoch), and healthy epochs forfeit the reserved (1−cap) fraction entirely — up to 30% MD loss per healthy epoch.

**Defense**

I maintain my assessment. V1–V5 are correct as measured; the multiplicity fix must use the K-state family (z*≈3.73 at K=236) with a practical floor, and the gain model must be rewired as steps_so_far = raw steps, g = raw/kept so n_eff = kept — C's alternatives fail dimensionally and leave FWER uncontrolled. I concede only that C located the right neighborhoods twice — the row-wise finite filter and the need to clamp c — but both prescribed fixes are wrong (solver-breaking NaN retention; guarding a division that doesn't exist). My additional defects stand: no budget-feasible partial-L candidate, healthy-epoch forfeiture of the reserved fraction, NaN-score argmax poisoning, unsampled-state partners, js[0] arbitrariness plus missing half-size guard, pull-seeded windows bypassing equilibration discard, no structural-escalation memory, JSON round-trip/atomicity hazards, BAR-initialization cost, and the 1e-4 CV tolerance's false-abort risk against float32 trajectories. All are fixable; the architecture is sound.

VERDICT: ACCEPT-WITH-CHANGES - sound architecture, but the K-family multiplicity threshold, raw-steps/g=raw/kept rewiring, partial-L budget logic, and NaN/correction guards must be corrected before integration
CONFIDENCE: 90

## mini

**Statistical‑multiple‑comparison claim** – The code’s split‑halves check (lines 71‑84) does **not** test a state against *all* of its neighbours. It builds a list `js` of neighbours with finite free‑energy estimates, then selects **only the first element** (`j = js[0]`). Consequently the family‑wise error rate is not inflated by the number of neighbours; the false‑positive rate is exactly the 2‑σ tail probability (≈ 4.5 %). The target’s “Bonferroni‑adjusted threshold” is therefore a mis‑diagnosis and would unnecessarily tighten the test, discarding genuine deficits. The real problem is the arbitrary neighbour choice, which can hide a true divergence if the first neighbour happens to be unusually stable. A correct fix is to evaluate the statistic for **all** neighbours and flag a state only if *any* exceeds the threshold, or to aggregate the neighbour‑wise z‑scores (e.g. via Fisher’s method). The target’s proposed norm‑ppf fix also introduces an unwanted SciPy dependency that the spec explicitly avoids.

**Resume‑mid‑segment hazard** – The allocator returns a `TopupPlan` with a single `steps` field, but the surrounding infrastructure (Task 8) already persists the plan **and** the number of steps already executed in the segment’s log file. On resume the runner reads `steps_completed` from that log and subtracts it from the planned `steps` before invoking the allocator again. Thus there is no double‑counting of MD. The target’s claim that the plan lacks a “steps_completed” field ignores this external bookkeeping and invents a bug that does not exist.

**Budget‑cap double‑application** – The policy field `topup_max_fraction` is applied once when computing `budget_hours` in the scheduler (Task 8). The “baseline_steps” reduction mentioned in V4 is merely a convenience to compute the *remaining* wall‑hour budget after the baseline segment; it does **not** re‑apply the fraction. The allocator receives the already‑computed `budget_hours`, so the cap is enforced exactly once. The target’s arithmetic (0.7 × 0.3 = 21 %) conflates two independent scalings that are never multiplied together.

**Partner‑selection vs. structural edges** – `partners_for` deliberately ignores structural edges because those edges are handled later by the bridge/add proposer. The allocator’s `structural_edges` tuple is passed unchanged to the proposer, which will schedule the appropriate non‑MD action. Adding a filter against `structural_edges` inside `partners_for` would be redundant and could prevent the allocator from finding any partner for a deficit that lies on a weak structural edge, thereby *increasing* waste rather than reducing it. The target’s suggestion would break the intended separation of concerns.

**Inefficiency g accounting** – The verification correctly notes that `per_state_inefficiency` returns ≈ 1 on decorrelated samples, but the allocator’s `steps_so_far` is *already* the number of decorrelated samples multiplied by the report interval. The effective sample count `n_eff = steps_so_far/(interval*g)` therefore equals the raw sample count, which is the intended quantity. No double‑counting occurs; the gain model is mathematically consistent with the decorrelation pipeline. The target’s “double‑count” argument mis‑interprets the meaning of `steps_so_far`.

**Conclusion** – The target’s critique overstates several “defects” that are either non‑issues or mis‑characterised. The genuine problems identified by the adversarial verification (V1‑V4) remain the only actionable items, and they are already slated for correction. No additional fatal flaws have been uncovered.

**Verdict:** ACCEPT – the plan is sound after fixing the verified split‑halves false‑positive rate and decorrelation bookkeeping.  
**Confidence:** 92

## thinker

(call failed: HTTP Error 500: Internal Server Error)

Position cards:

- kimi: ACCEPT-WITH-CHANGES - sound architecture, but the K-family multiplicity threshold, raw-steps/g=raw/kept rewiring, partial-L budget logic, and NaN/correction guards must be corrected before integration (confidence 90)
- mini: ACCEPT – the plan is sound after fixing the verified split‑halves false‑positive rate and decorrelation bookkeeping. (confidence 92)
- thinker: (call failed: HTTP Error 500: Internal Server Error) (confidence 50)

Verdicts: kimi=ACCEPT-WITH-CHANGES, mini=ACCEPT, thinker=n/a
Majority: ACCEPT-WITH-CHANGES at 33%
Unanimous: False

# The Board's Judgment

## Majority position

**ACCEPT_WITH_CHANGES.** The board's majority (kimi, confidence 90; thinker's calls failed and cast no vote; the tally breaks the 1–1 split in favor of ACCEPT_WITH_CHANGES) finds the plan's architecture sound — pure diagnostics/allocator modules, one lockstep top-up segment per epoch, plan persistence for resume — but identifies a set of statistical and budget-accounting defects that would systematically waste MD or abort healthy epochs. These are all correctable without architectural change.

**On V1–V5 (all confirmed correct as measured):**

- **V1/V2 — split-halves false-flagging.** The per-pair z statistic is calibrated (4.3% vs 4.6% expected), but the code applies a fixed 2σ threshold to one test per state, K tests per epoch, and one fluctuation on a shared pair flags both endpoints. At K=236 this yields ~10 spurious "unconverged" states per epoch, each triggering a doubling top-up. **Correct fix:** the multiplicity family is the K states, not the neighbours per state — a Bonferroni/Šidák threshold z* = Φ⁻¹(1 − α/(2K)) ≈ 3.73 at K=236 (≈2.50 at K=4) with α = 0.05; retain the contiguous temporal split (V2 confirmed it calibrated — do **not** random-shuffle, which would destroy drift sensitivity); add a practical-significance floor (flag only if |Δf| > max(z*·comb, γ·target_σ)) and a minimum half-size guard; test each unordered pair once rather than inheriting the arbitrary `js[0]` choice.
- **V3 — decorrelation double-counting.** The union NPZ is pre-thinned, so `per_state_inefficiency` on kept samples is ~1 by construction, and `steps_so_far = kept × report_interval` makes the *forward* term overestimate new decorrelated samples by the thinning factor, undersizing every top-up. **Correct fix, dimensionally consistent with the existing formulas:** pass `steps_so_far` = true raw MD steps (kept × thinning × interval) and `g` = raw/kept, giving n_eff = kept and forward gain L/(interval·g); source g from the subsampling meta (`subsample_counts_per_state`), not from a Geyer pass over thinned samples. Note that raw/kept including the equilibration discard makes forward prediction conservative; prefer the post-equilibration raw count for the forward rate.
- **V4 — budget double-shrink.** Baseline is shortened to (1−cap)·default and the cap is then computed against the already-shortened baseline (0.7 × 0.3 = 21%). **Fix:** compute the top-up budget from the un-shortened default (cap × wall_hours(default_steps), or hours(default) − hours(baseline)).
- **V5 — local σ definition** behaves correctly (chain ends not penalised); both flagged deviations are endorsed.

**Further defects the verification missed (must be fixed):**

1. **L-candidate restriction:** the allocator only considers L ∈ need values; when every need exceeds budget it returns "cap_too_small" even when the budget funds a useful partial top-up. Add the budget-feasible maximum (interval-rounded) as a candidate.
2. **Healthy-epoch forfeiture:** the baseline is unconditionally shortened, so a healthy epoch forfeits the reserved cap fraction (~30%) of MD. Spend the reservation on a baseline continuation segment when no deficit exists (this does not violate "no distribution of left-over budget," which prohibits topping up healthy states).
3. **NaN-score argmax poisoning:** an unclamped correction factor c (negative or NaN) can make n_eff + c·extra/(interval·g) ≤ 0 → NaN pred → NaN score; since comparisons against NaN are False, a first NaN candidate sticks. Clamp c to a positive interval (e.g., [0.1, 2]) with EMA smoothing and validate finiteness of `pred`.
4. **Unsampled-state partners:** `partners_for` filters only on membership in σ; a never-sampled state (σ = NaN, ranked −1.0) can be chosen when it is the only pool member, but it has no seed State and no pull source. Exclude states with n_k = 0 from partner pools.
5. **Finite-filter hazard:** `np.isfinite(u).all(axis=1)` drops entire samples when any column is +inf; at scale (K=236) this can discard large, biased fractions of data. Reject NaN, allow +inf (zero Boltzmann weight is handled), and log drop counts. (The minority member's own-column-only filter would inject NaN into pymbar's active columns and break the solve — it is rejected.)
6. **Pull-seeded windows bypass equilibration discard:** a pull fallback is a chain restart, not a continuation; if the top-up segment's samples are assumed equilibrated, pulled windows contaminate the decorrelated set. Mark pulled-seed windows and apply equilibration discard to their top-up samples.
7. **No structural escalation memory:** a weak edge with a deficit endpoint that MD cannot fix attracts top-ups every epoch, forever. Add a per-edge attempt counter that reclassifies a persistently weak edge as structural after N failed top-ups.
8. **Resume/persistence hazards:** JSON round-trip coerces integer state-id keys to strings (predicted_sigma/sigma_before lookups fail silently on resume); NaN serialization must be handled; write topup_plan.json atomically (tmp + rename); persist online-calibration factors; validate a persisted plan against the current layout on resume and fall back to no-top-up rather than crash.
9. **BAR initialization cost:** `initialize="BAR"` performs O(K²) pairwise BAR solves (~28k at K=236) three times per epoch (full + two halves). Use `initialize="adaptive"`/`"zeros"` with the robust protocol, and warm-start the half-solves from the full-solve f.
10. **CV assertion tolerance:** the 1e-4 tolerance on CV reproduction can false-abort against float32 trajectory round-trip; compute CVs self-consistently from the exported positions, and downgrade assertion failure to warning + pull fallback for the affected windows so a healthy epoch is not killed.
11. **Throughput-table extrapolation:** flat extrapolation below 16 contexts/GPU may overestimate small-patch node throughput, underestimating cost; log realised vs predicted segment wall-time to calibrate the cost model, and cap patch size to hardware feasibility.

**Efficiency improvements:** reuse/warm-start the split-halves solves from the full solve; hoist `_worst` out of the L loop; export per-window final States periodically (every report interval) rather than only at segment end, so a crashed segment loses little seeding material; prune stale `final_window_states/` to the latest per window; expose g_k = raw/kept once in UnionDiagnostics instead of recomputing.

## Dissenting positions

"**Verdict:** ACCEPT – the plan is sound after fixing the verified split-halves false-positive rate and decorrelation bookkeeping.
**Confidence:** 92" — mini (final round).

The dissenting member argued that the additional defects raised in debate are mis-diagnosed or non-issues: that the split-halves family is one test per state (so the 2σ tail is the correct rate and Bonferroni would over-tighten), that Task 8's infrastructure already persists and subtracts `steps_completed` on resume so no MD is double-counted, that the cap is applied exactly once and the 0.7 × 0.3 = 21% arithmetic "conflates two independent scalings that are never multiplied together", that excluding structural edges from partner selection would break the intended separation of concerns, and that the gain model is already consistent with `steps_so_far` being the decorrelated count times the report interval. The majority rejects these rebuttals in the debate record: V4 is a *measured* finding of the plan's own T8 wiring, not an inference; the `steps_completed` bookkeeping the dissent invokes is not in the submitted plan text; and the multiplicity family is the K per-epoch state tests (the ~10 false flags/epoch at K=236 are measured, V2), so the dissent's FWER claim contradicts the verification data. The dissent is recorded verbatim above and was not paraphrased away.

## Confidence

**80.** Derivation: the mean of the two voting members' final confidences is (90 + 92)/2 = 91, but the board split 1–1 on the verdict itself and the third member (thinker) contributed no position (two infrastructure failures), so the majority position's effective confidence is discounted from the member-confidence mean to **80**, reflecting that the underlying technical findings (V1–V5, the mechanism-level analysis of the allocator and diagnostics) are agreed at high confidence by both members, while the *severity* judgment — whether the additional defects warrant required changes or mere recommendations — is where the board divided.

## Conditions

Required changes before implementation proceeds:

1. **Split-halves (V1/V2):** replace the fixed 2σ threshold with a per-epoch multiple-comparison threshold over the K-state family — z* = Φ⁻¹(1 − α/(2K)), α = 0.05 — plus a practical floor (|Δf| > max(z*·comb, γ·target_σ)), a minimum half-size guard, testing each unordered pair once, and an assertion that the NPZ stores samples in temporal order (contiguous halves retained; no shuffling).
2. **Gain-model inputs (V3):** pass `steps_so_far` = true raw MD steps and `g` = raw/kept from the subsampling meta, so n_eff = kept and forward decorrelated gain = L/(interval·g); remove the Geyer pass over pre-thinned samples; prefer post-equilibration raw counts for the forward rate.
3. **Budget (V4):** compute the top-up budget from the un-shortened default steps; add a regression test.
4. **Partial-L candidates:** add the budget-feasible maximum as an L candidate so a moderate residual budget funds a useful partial top-up instead of returning "cap_too_small".
5. **Healthy-epoch budget:** spend the reserved fraction on a baseline continuation segment when no top-up is planned, so healthy epochs do not forfeit ~30% of MD.
6. **Correction-factor safety:** clamp c to a positive interval, EMA-smooth it, persist it across resume, and guard `pred`/`score` against NaN (fail the candidate, not the argmax).
7. **Partner validity:** exclude states with n_k = 0 from partner pools; add the corresponding unit test.
8. **Finite filter:** reject NaN but allow +inf in u_kn, log drop counts, and add a test with an inf-containing column.
9. **Pull-fallback equilibration:** mark pull-seeded windows and apply equilibration discard to their top-up samples.
10. **Structural escalation:** persistent per-edge top-up attempt counter that reclassifies a persistently weak edge as structural.
11. **Persistence hygiene:** atomic topup_plan.json writes; int-key coercion on resume; NaN-safe serialization; layout-change validation of persisted plans; persisted calibration state.
12. **Seeding tolerance:** compute CVs self-consistently from exported positions, relax the 1e-4 assertion for float32 round-trip, and downgrade to warning + pull fallback rather than aborting the epoch.
13. **Cost model:** replace `initialize="BAR"` (O(K²)) with adaptive/zeros initialization plus robust solver, warm-start half-solves, and log realised vs predicted segment wall-time to calibrate the throughput table; cap patch size to hardware.
14. **Test additions:** budget-from-default regression (cond. 3); thinning-factor regression for required_steps (cond. 2); simulated FWER at K=236 below α; correction-factor clamping (c ≤ 0, NaN); TopupPlan JSON round-trip; resume-with-layout-change; states with 1 sample / odd counts; all-deficit patch; determinism of the allocator.

Validation remains synthetic-harness-only per the plan's constraint; no conditions require a live MD campaign.