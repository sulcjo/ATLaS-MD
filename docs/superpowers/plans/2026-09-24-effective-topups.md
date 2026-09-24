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

### Task 1: Shared layout-neighbour module

The nearest same-rung, same-restraint-pattern neighbour rule lives in `gareus/dashboard/neighbours.py`. The allocator needs it; a core module must not import from the dashboard.

**Files:**
- Create: `gareus/layout_neighbours.py` (moved content of `gareus/dashboard/neighbours.py`)
- Modify: `gareus/dashboard/neighbours.py` (becomes a re-export shim)
- Test: `tests/test_layout_neighbours.py`

**Interfaces:**
- Produces: `gareus.layout_neighbours.spatial_neighbour_pairs(centers, secondary_centers, lambdas, k1, k2, temperature_k) -> list[tuple[int, int]]`, `overlap_by_pair_2d(...)`, `best_neighbour_overlaps(...)`, `NEIGHBOUR_SLACK` — identical signatures to the dashboard versions.
- Produces: `gareus.layout_neighbours.same_rung_neighbours(pairs, lambdas) -> dict[int, list[int]]` and `other_rung_same_centre(centers, secondary_centers, lambdas) -> dict[int, list[int]]` (new, used by Task 6).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_layout_neighbours.py
import math

from gareus.layout_neighbours import (
    other_rung_same_centre, same_rung_neighbours, spatial_neighbour_pairs,
)


def _layout():
    # 2 centres x 2 rungs; windows 0,1 at centre A (lambda 0, 1); 2,3 at centre B.
    centers = [0.1, 0.1, 0.2, 0.2]
    sec = [0.0, 0.0, 0.0, 0.0]
    lam = [0.0, 1.0, 0.0, 1.0]
    return centers, sec, lam


def test_the_dashboard_module_reexports_the_same_function():
    from gareus.dashboard import neighbours as dash

    assert dash.spatial_neighbour_pairs is spatial_neighbour_pairs


def test_same_rung_neighbours_maps_each_window_to_its_rung_partners():
    c, s, lam = _layout()
    pairs = spatial_neighbour_pairs(c, s, lam, [300.0] * 4, [1.0] * 4, 300.0)
    nb = same_rung_neighbours(pairs, lam)
    assert nb[0] == [2] and nb[2] == [0] and nb[1] == [3] and nb[3] == [1]


def test_other_rung_same_centre_groups_rungs_of_one_centre():
    c, s, lam = _layout()
    other = other_rung_same_centre(c, s, lam)
    assert other == {0: [1], 1: [0], 2: [3], 3: [2]}


def test_unknown_lambda_windows_have_no_rung_partners():
    other = other_rung_same_centre([0.1, 0.1], [0.0, 0.0], [math.nan, math.nan])
    assert other == {0: [], 1: []}
```

- [ ] **Step 2: Run test to verify it fails**

Run (via opencode, see Global Constraints): `python -m pytest -q -p no:cacheprovider tests/test_layout_neighbours.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'gareus.layout_neighbours'`.

- [ ] **Step 3: Move the module and add the two helpers**

```bash
git mv gareus/dashboard/neighbours.py gareus/layout_neighbours.py
```

In `gareus/layout_neighbours.py`: change `from ..math_helpers import _hist_overlap` to `from .math_helpers import _hist_overlap` and `from .ranking import restraint_sigma` to `from .dashboard.ranking import restraint_sigma`. Append:

```python
def same_rung_neighbours(pairs, lambdas) -> dict:
    """window -> sorted list of its spatial-neighbour windows (pairs are same-rung already)."""
    out: dict = {}
    for a, b in pairs:
        out.setdefault(int(a), []).append(int(b))
        out.setdefault(int(b), []).append(int(a))
    return {w: sorted(set(v)) for w, v in out.items()}


def other_rung_same_centre(centers, secondary_centers, lambdas) -> dict:
    """window -> other windows at the same (CV1, CV2) centre on a different, known rung."""
    n = min(len(centers), len(secondary_centers))
    key = [(round(float(centers[w]), _CENTRE_DECIMALS), round(float(secondary_centers[w]), _CENTRE_DECIMALS))
           for w in range(n)]
    lam = [(_finite(lambdas[w]) if w < len(lambdas) else None) for w in range(n)]
    out = {}
    for w in range(n):
        out[w] = [] if lam[w] is None else [
            j for j in range(n)
            if j != w and key[j] == key[w] and lam[j] is not None and lam[j] != lam[w]]
    return out
```

Add both names to `__all__`. Create the shim `gareus/dashboard/neighbours.py`:

```python
"""Re-export: the neighbour rule lives in gareus.layout_neighbours (shared with the allocator)."""
from ..layout_neighbours import *  # noqa: F401,F403
from ..layout_neighbours import __all__  # noqa: F401
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest -q -p no:cacheprovider tests/test_layout_neighbours.py tests/test_dashboard_neighbours.py tests/test_dashboard_grid2d.py`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add gareus/layout_neighbours.py gareus/dashboard/neighbours.py tests/test_layout_neighbours.py
git commit -m "refactor: move the layout-neighbour rule out of the dashboard"
```

---

### Task 2: Flags, defaults and policy fields

**Files:**
- Modify: `gareus/cli.py` (the `--ap-topups` argument added in PR #98; the `_apply_v2_compat_shims` block that sets `args.adaptive_production_topups`)
- Modify: `gareus/adaptive_production.py` (`AdaptiveDecisionPolicy` dataclass; `policy_from_args`)
- Modify: `tests/test_no_topups.py` (default flips to off)
- Test: `tests/test_topup_flags.py`

**Interfaces:**
- Produces: `AdaptiveDecisionPolicy.topups_enabled: bool = False`, `.topup_target_sigma: float = 0.10`, `.topup_weak_overlap: float = 0.15`, `.topup_max_fraction: float = 0.3`, `.topup_throughput_table: tuple = ((16.0, 3154.0), (59.0, 2300.0))`.
- Produces: `args.adaptive_production_topups` and `args.adaptive_production_topup_{target_sigma,weak_overlap,max_fraction,throughput_table}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_topup_flags.py
from gareus.adaptive_production import AdaptiveDecisionPolicy, policy_from_args
from gareus.cli import build_gareus_parser, parse_args


def test_topups_are_off_by_default_and_opt_in():
    p = build_gareus_parser()
    assert p.parse_args(["--seq", "AA"]).ap_topups is False
    assert p.parse_args(["--seq", "AA", "--ap-topups"]).ap_topups is True


def test_topup_knobs_reach_the_policy():
    args = parse_args(["--seq", "AA", "--ap-topups", "--ap-topup-target-sigma", "0.2",
                       "--ap-topup-weak-overlap", "0.1", "--ap-topup-max-fraction", "0.25"])
    pol = policy_from_args(args)
    assert pol.topups_enabled is True
    assert (pol.topup_target_sigma, pol.topup_weak_overlap, pol.topup_max_fraction) == (0.2, 0.1, 0.25)
    assert pol.topup_throughput_table == ((16.0, 3154.0), (59.0, 2300.0))


def test_policy_defaults_match_the_spec():
    pol = AdaptiveDecisionPolicy()
    assert pol.topups_enabled is False
    assert (pol.topup_target_sigma, pol.topup_weak_overlap, pol.topup_max_fraction) == (0.10, 0.15, 0.3)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest -q -p no:cacheprovider tests/test_topup_flags.py`
Expected: FAIL (`ap_topups` default is True; unknown `--ap-topup-target-sigma`).

- [ ] **Step 3: Implement**

In `gareus/cli.py`, change the `--ap-topups` argument's `default=True` to `default=False` and its help's first sentence to "Top-ups (off by default): ...". Add after it:

```python
    p.add_argument("--ap-topup-target-sigma", type=float, default=0.10,
                   help="Per-state free-energy uncertainty target (kcal/mol) for top-ups.")
    p.add_argument("--ap-topup-weak-overlap", type=float, default=0.15,
                   help="Symmetric energy-space overlap below which an edge is weak.")
    p.add_argument("--ap-topup-max-fraction", type=float, default=0.3,
                   help="Cap on the share of an epoch's wall-hour budget spent on top-ups.")
    p.add_argument("--ap-topup-throughput-table", default="16:3154,59:2300",
                   help="contexts_per_gpu:ns_per_day_node pairs, comma separated.")
```

In `_apply_v2_compat_shims`, after `args.adaptive_production_topups = args.ap_topups`:

```python
    args.adaptive_production_topup_target_sigma = args.ap_topup_target_sigma
    args.adaptive_production_topup_weak_overlap = args.ap_topup_weak_overlap
    args.adaptive_production_topup_max_fraction = args.ap_topup_max_fraction
    args.adaptive_production_topup_throughput_table = tuple(
        (float(a), float(b)) for a, b in (item.split(":") for item in str(args.ap_topup_throughput_table).split(",") if item))
```

In `AdaptiveDecisionPolicy` (after `min_exchange_acceptance`):

```python
    topups_enabled: bool = False
    topup_target_sigma: float = 0.10
    topup_weak_overlap: float = 0.15
    topup_max_fraction: float = 0.3
    topup_throughput_table: tuple = ((16.0, 3154.0), (59.0, 2300.0))
```

In `policy_from_args(...)` add:

```python
        topups_enabled=_arg_bool(args, "adaptive_production_topups", False),
        topup_target_sigma=_arg_float(args, "adaptive_production_topup_target_sigma", 0.10),
        topup_weak_overlap=_arg_float(args, "adaptive_production_topup_weak_overlap", 0.15),
        topup_max_fraction=_arg_float(args, "adaptive_production_topup_max_fraction", 0.3),
        topup_throughput_table=tuple(getattr(args, "adaptive_production_topup_throughput_table",
                                             ((16.0, 3154.0), (59.0, 2300.0)))),
```

In `tests/test_no_topups.py::test_the_flag_parses_and_defaults_on` rename to `test_the_flag_parses_and_defaults_off` and swap the asserts (`--seq AA` → False, `--seq AA --ap-topups` → True). In `run_scheduled_adaptive_epoch` change `_arg_bool(args, "adaptive_production_topups", True)` to `False`.

- [ ] **Step 4: Run tests**

Run: `python -m pytest -q -p no:cacheprovider tests/test_topup_flags.py tests/test_no_topups.py tests/test_dashboard_cli_flags.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add gareus/cli.py gareus/adaptive_production.py tests/test_topup_flags.py tests/test_no_topups.py
git commit -m "feat: top-ups off by default; add top-up target/overlap/cap/throughput knobs"
```

---

### Task 3: Throughput model

**Files:**
- Create: `gareus/adaptive/__init__.py` (empty docstring module), `gareus/adaptive/throughput.py`
- Test: `tests/test_topup_throughput.py`

**Interfaces:**
- Produces: `node_ns_per_day(n_states: int, n_gpus: int, table) -> float`; `wall_hours(steps: int, n_states: int, timestep_fs: float, n_gpus: int, table) -> float` (hours for `n_states` replicas to each advance `steps`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_topup_throughput.py
import pytest

from gareus.adaptive.throughput import node_ns_per_day, wall_hours

TABLE = ((16.0, 3154.0), (59.0, 2300.0))


def test_interpolates_between_measured_points():
    assert node_ns_per_day(64, 4, TABLE) == pytest.approx(3154.0)          # 16/GPU
    assert node_ns_per_day(236, 4, TABLE) == pytest.approx(2300.0)         # 59/GPU
    mid = node_ns_per_day(4 * 37, 4, TABLE)                                 # 37/GPU
    assert 2300.0 < mid < 3154.0


def test_is_flat_outside_the_measured_range():
    assert node_ns_per_day(8, 4, TABLE) == pytest.approx(3154.0)            # 2/GPU: no extrapolated gain
    assert node_ns_per_day(400, 4, TABLE) == pytest.approx(2300.0)


def test_wall_hours_scales_with_steps_and_states():
    # 236 states x 1e6 steps x 4 fs = 944 ns aggregate at 2300 ns/day -> 9.85 h
    assert wall_hours(1_000_000, 236, 4.0, 4, TABLE) == pytest.approx(944.0 / 2300.0 * 24.0)
    # the same per-state ns on a 64-state patch costs less wall time per state-ns
    per_state_small = wall_hours(1_000_000, 64, 4.0, 4, TABLE) / 64
    per_state_full = wall_hours(1_000_000, 236, 4.0, 4, TABLE) / 236
    assert per_state_small < per_state_full


def test_rejects_an_empty_table():
    with pytest.raises(ValueError):
        node_ns_per_day(10, 4, ())
```

- [ ] **Step 2: Run to verify it fails** — `python -m pytest -q -p no:cacheprovider tests/test_topup_throughput.py`; expected `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

```python
# gareus/adaptive/__init__.py
"""Adaptive-production helpers that are pure functions of registry/diagnostics state."""
```

```python
# gareus/adaptive/throughput.py
"""Wall-clock cost of a lockstep segment from a measured node throughput table.

The table maps contexts per GPU -> aggregate ns/day for the node. Between points
it interpolates linearly; outside them it stays flat (no extrapolated gain),
because the low-occupancy end is unmeasured (spec section 4.3.5).
"""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


def node_ns_per_day(n_states: int, n_gpus: int, table: Sequence[Tuple[float, float]]) -> float:
    if not table:
        raise ValueError("throughput table is empty")
    pts = sorted((float(c), float(v)) for c, v in table)
    x = max(1.0, float(n_states)) / max(1, int(n_gpus))
    xs = np.array([p[0] for p in pts]); ys = np.array([p[1] for p in pts])
    return float(np.interp(x, xs, ys, left=ys[0], right=ys[-1]))


def wall_hours(steps: int, n_states: int, timestep_fs: float, n_gpus: int,
               table: Sequence[Tuple[float, float]]) -> float:
    aggregate_ns = float(steps) * float(timestep_fs) * 1e-6 * float(n_states)
    return aggregate_ns / node_ns_per_day(n_states, n_gpus, table) * 24.0
```

- [ ] **Step 4: Run tests** — expected PASS.
- [ ] **Step 5: Commit** — `git add gareus/adaptive/ tests/test_topup_throughput.py && git commit -m "feat: wall-hour cost model for lockstep segments"`

---

### Task 4: Conservative autocorrelation estimator

**Files:**
- Create: `gareus/adaptive/autocorr.py`
- Test: `tests/test_topup_autocorr.py`

**Interfaces:**
- Produces: `statistical_inefficiency_ips(x: np.ndarray) -> float` returning g = 1 + 2τ (≥ 1) with Geyer's initial positive sequence; `per_state_inefficiency(values: np.ndarray, state_of_row: np.ndarray, state_ids: Sequence[int]) -> dict[int, float]` (rows must be in time order within each state).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_topup_autocorr.py
import numpy as np
import pytest

from gareus.adaptive.autocorr import per_state_inefficiency, statistical_inefficiency_ips


def _ar1(phi, n, seed):
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.normal()
    return x


def test_white_noise_has_inefficiency_near_one():
    x = np.random.default_rng(0).normal(size=20000)
    assert statistical_inefficiency_ips(x) == pytest.approx(1.0, abs=0.15)


def test_ar1_matches_the_closed_form():
    phi = 0.9                                  # g = (1 + phi) / (1 - phi) = 19
    g = statistical_inefficiency_ips(_ar1(phi, 200000, 1))
    assert g == pytest.approx(19.0, rel=0.15)


def test_constant_or_short_traces_return_one():
    assert statistical_inefficiency_ips(np.ones(100)) == 1.0
    assert statistical_inefficiency_ips(np.array([1.0, 2.0])) == 1.0


def test_per_state_splits_by_state_in_time_order():
    a = _ar1(0.9, 50000, 2); b = np.random.default_rng(3).normal(size=50000)
    values = np.concatenate([a, b]); state = np.array([7] * 50000 + [9] * 50000)
    g = per_state_inefficiency(values, state, [7, 9, 11])
    assert g[7] > 10 and g[9] < 1.5 and g[11] == 1.0
```

- [ ] **Step 2: Run to verify it fails** — expected `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

```python
# gareus/adaptive/autocorr.py
"""Statistical inefficiency with Geyer's initial positive sequence (conservative)."""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def statistical_inefficiency_ips(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 4:
        return 1.0
    x = x - x.mean()
    var = float(np.dot(x, x) / n)
    if var <= 0.0:
        return 1.0
    f = np.fft.rfft(x, n=2 * n)
    acf = np.fft.irfft(f * np.conjugate(f))[:n] / (var * np.arange(n, 0, -1))
    g = 1.0
    # Geyer: sum consecutive pairs Gamma_m = rho(2m) + rho(2m+1) while positive.
    for m in range(1, n // 2):
        pair = acf[2 * m - 1] + acf[2 * m]
        if pair <= 0.0:
            break
        g += 2.0 * pair
    return float(max(1.0, g))


def per_state_inefficiency(values, state_of_row, state_ids: Sequence[int]) -> Dict[int, float]:
    values = np.asarray(values, dtype=float)
    state_of_row = np.asarray(state_of_row)
    return {int(s): statistical_inefficiency_ips(values[state_of_row == s]) for s in state_ids}
```

- [ ] **Step 4: Run tests** — expected PASS.
- [ ] **Step 5: Commit** — `git add gareus/adaptive/autocorr.py tests/test_topup_autocorr.py && git commit -m "feat: conservative statistical-inefficiency estimator for top-up gain model"`

---

### Task 5: Per-epoch union diagnostics

Solve the union MBAR the driver already builds and return per-state σ, split-halves flags, inefficiency and a symmetric overlap for every geometry edge (spatial and rung).

**Files:**
- Create: `gareus/adaptive/union_diagnostics.py`
- Test: `tests/test_topup_union_diagnostics.py`

**Interfaces:**
- Consumes: the union NPZ written by `build_union_state_mbar_inputs` (keys `umbrella_reduced_bias_nk`, `state_ids`, `sampled_state_ids`); `gareus.mbar_analysis.ladder.mbar_state_overlap(u_nk, f_k, n_k)`; `gareus.adaptive.autocorr.per_state_inefficiency`.
- Produces:

```python
@dataclass(frozen=True)
class UnionDiagnostics:
    state_ids: tuple            # union order
    n_k: dict                   # state_id -> samples in the solve
    sigma_kcal: dict            # state_id -> LOCAL free-energy uncertainty (kcal/mol): min_j dDelta_f[j,k] over edge-neighbours; nan if unknown
    unconverged: frozenset      # state_ids failing the split-halves check
    inefficiency: dict          # state_id -> g = 1 + 2 tau (>= 1)
    edge_overlap: dict          # (min_id, max_id) -> symmetric sqrt(O_ij O_ji)

def union_diagnostics_from_npz(npz_path, edges, *, kt_kcal: float, split_halves: bool = True) -> Optional[UnionDiagnostics]
```

`edges` is a list of `(state_i, state_j)`. Returns `None` (never raises) when the file is missing, has no finite rows, or the solve fails.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_topup_union_diagnostics.py
import math

import numpy as np

from gareus.adaptive.union_diagnostics import union_diagnostics_from_npz

KT = 0.596  # kcal/mol at 300 K


def _write(tmp_path, centers, k, n_per, seed=0, drift_state=None):
    rng = np.random.default_rng(seed)
    x, sid = [], []
    for s, c in enumerate(centers):
        sig = 1.0 / math.sqrt(k)
        xs = rng.normal(c, sig, n_per)
        if drift_state == s:                       # second half sits somewhere else
            xs[n_per // 2:] += 4 * sig
        x.append(xs); sid += [s] * n_per
    x = np.concatenate(x)
    u = 0.5 * k * (x[:, None] - np.asarray(centers)[None, :]) ** 2   # reduced (kT units)
    p = tmp_path / "u.npz"
    np.savez(p, umbrella_reduced_bias_nk=u, state_ids=np.arange(len(centers)),
             sampled_state_ids=np.asarray(sid))
    return p


def test_healthy_chain_has_small_sigma_and_all_edges_measured(tmp_path):
    p = _write(tmp_path, [0.0, 1.0, 2.0, 3.0], k=4.0, n_per=2000)
    d = union_diagnostics_from_npz(p, [(0, 1), (1, 2), (2, 3)], kt_kcal=KT)
    assert d is not None
    assert all(math.isfinite(d.sigma_kcal[s]) and d.sigma_kcal[s] > 0 for s in (0, 1, 2, 3))
    # local sigma: the chain end is not penalised for its distance from state 0
    assert d.sigma_kcal[3] < 2.0 * d.sigma_kcal[1]
    assert set(d.edge_overlap) == {(0, 1), (1, 2), (2, 3)}
    assert min(d.edge_overlap.values()) > 0.15
    assert not d.unconverged


def test_fewer_samples_means_larger_sigma(tmp_path):
    big = union_diagnostics_from_npz(_write(tmp_path, [0.0, 1.0], 4.0, 4000), [(0, 1)], kt_kcal=KT)
    small = union_diagnostics_from_npz(_write(tmp_path, [0.0, 1.0], 4.0, 250), [(0, 1)], kt_kcal=KT)
    assert small.sigma_kcal[1] > big.sigma_kcal[1]


def test_a_state_whose_halves_disagree_is_flagged(tmp_path):
    p = _write(tmp_path, [0.0, 1.0, 2.0], k=4.0, n_per=3000, drift_state=1)
    d = union_diagnostics_from_npz(p, [(0, 1), (1, 2)], kt_kcal=KT)
    assert 1 in d.unconverged


def test_missing_or_empty_input_returns_none(tmp_path):
    assert union_diagnostics_from_npz(tmp_path / "nope.npz", [], kt_kcal=KT) is None
    p = tmp_path / "e.npz"
    np.savez(p, umbrella_reduced_bias_nk=np.full((3, 2), np.nan), state_ids=np.arange(2),
             sampled_state_ids=np.array([0, 1, 1]))
    assert union_diagnostics_from_npz(p, [(0, 1)], kt_kcal=KT) is None


def test_a_state_with_zero_samples_gets_nan_sigma_not_a_crash(tmp_path):
    p = _write(tmp_path, [0.0, 1.0, 2.0], k=4.0, n_per=1000)
    with np.load(p) as z:
        u, ids, sid = z["umbrella_reduced_bias_nk"], z["state_ids"], z["sampled_state_ids"]
    keep = sid != 2
    np.savez(p, umbrella_reduced_bias_nk=u[keep], state_ids=ids, sampled_state_ids=sid[keep])
    d = union_diagnostics_from_npz(p, [(0, 1), (1, 2)], kt_kcal=KT)
    assert d is not None and math.isnan(d.sigma_kcal[2]) and d.n_k[2] == 0
```

- [ ] **Step 2: Run to verify it fails** — expected `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

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

Check the rows passed to `per_state_inefficiency` are in time order within each state: the union builder writes `sample_rows` in source order and keeps `sorted(_kept_global)` indices, so within a state rows are chronological per source. State that assumption in a comment.

- [ ] **Step 4: Run tests** — expected PASS. If `test_a_state_whose_halves_disagree_is_flagged` is flaky, raise `n_per`, not the threshold.
- [ ] **Step 5: Commit** — `git add gareus/adaptive/union_diagnostics.py tests/test_topup_union_diagnostics.py && git commit -m "feat: per-epoch union MBAR diagnostics for top-up allocation"`

---

### Task 6: Top-up allocator

**Files:**
- Create: `gareus/adaptive/topup_allocator.py`
- Test: `tests/test_topup_allocator.py`

**Interfaces:**
- Consumes: `UnionDiagnostics` (Task 5), `wall_hours` (Task 3), `same_rung_neighbours` / `other_rung_same_centre` (Task 1).
- Produces:

```python
@dataclass(frozen=True)
class TopupPlan:
    state_ids: tuple            # patch, sorted
    steps: int                  # lockstep length, multiple of report_interval
    deficit_state_ids: tuple
    partner_state_ids: tuple
    structural_edges: tuple     # (i, j) routed to bridge proposal
    predicted_sigma: dict       # state_id -> sigma after the top-up
    cost_hours: float
    reason: str                 # "planned" | "healthy" | "no_diagnostics" | "cap_too_small"
    sigma_before: dict          # state_id -> sigma when planned (for calibration)

def plan_topup(diag, *, state_ids_in_order, neighbours, rung_partners, policy,
               steps_so_far: dict, report_interval: int, timestep_fs: float, n_gpus: int,
               budget_hours: float, correction: dict | None = None) -> TopupPlan
```

`neighbours`: state_id → same-rung spatial neighbour state_ids; `rung_partners`: state_id → other rungs of the same centre; `steps_so_far`: state_id → steps sampled so far; `correction`: state_id → realised/predicted gain factor from earlier top-ups (default 1.0).

Rules (from spec 4.2–4.3):
- deficit = `sigma > target` or in `unconverged`; NaN sigma is **not** a deficit (reason recorded) — it is an unsampled state, which is a coverage problem, not a top-up target;
- weak edge = measured overlap `< topup_weak_overlap`; noise if either endpoint is a deficit → both endpoints join the deficit set; structural otherwise → `structural_edges`;
- predicted σ after L extra steps: `sigma * sqrt(n_eff / (n_eff + c * L / (report_interval * g)))`, with `n_eff = steps_so_far / (report_interval * g)`, `c = correction.get(s, 1.0)`;
- required steps for s: smallest L (rounded up to `report_interval`) with predicted σ ≤ target, capped at 4× `steps_so_far[s]`;
- partners: for each deficit s, if no deficit in `neighbours[s]` add the neighbour with the largest σ (any rung if `neighbours[s]` is empty: fall back to the nearest listed in `rung_partners`); if no deficit among `rung_partners[s]` add the rung partner with the largest σ;
- length: candidates = sorted distinct required steps; for each L, patch = deficits still short at L ∪ their partners; benefit = reduction of max σ over deficits (primary) + 1e-3 × reduction of Σ σ² (tie-break); pick best benefit / `wall_hours(L, |patch|, ...)`; skip any L whose cost exceeds `budget_hours`;
- if no candidate fits the budget → `reason="cap_too_small"`, empty patch; if no deficits → `reason="healthy"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_topup_allocator.py
import math

from gareus.adaptive.topup_allocator import plan_topup
from gareus.adaptive.union_diagnostics import UnionDiagnostics
from gareus.adaptive_production import AdaptiveDecisionPolicy

POL = AdaptiveDecisionPolicy(topups_enabled=True)          # target 0.10, weak 0.15, cap 0.3
TABLE = POL.topup_throughput_table


def _diag(sigma, edges=None, unconverged=(), g=None):
    ids = tuple(sorted(sigma))
    return UnionDiagnostics(state_ids=ids, n_k={s: 1000 for s in ids}, sigma_kcal=dict(sigma),
                            unconverged=frozenset(unconverged),
                            inefficiency=g or {s: 2.0 for s in ids}, edge_overlap=edges or {})


def _chain(n):
    # 1D chain of centres on one rung; rung partner = the same index offset by n (second rung)
    nb = {s: [x for x in (s - 1, s + 1) if 0 <= x < n] for s in range(n)}
    nb.update({s + n: [x + n for x in (s - 1, s + 1) if 0 <= x < n] for s in range(n)})
    rp = {s: [s + n] for s in range(n)}; rp.update({s + n: [s] for s in range(n)})
    return nb, rp


def _plan(diag, n, budget=100.0, steps=None, correction=None):
    nb, rp = _chain(n)
    return plan_topup(diag, state_ids_in_order=list(diag.state_ids), neighbours=nb, rung_partners=rp,
                      policy=POL, steps_so_far=steps or {s: 100_000 for s in diag.state_ids},
                      report_interval=500, timestep_fs=4.0, n_gpus=4, budget_hours=budget,
                      correction=correction)


def test_no_diagnostics_means_no_topup():
    nb, rp = _chain(4)
    p = plan_topup(None, state_ids_in_order=list(range(8)), neighbours=nb, rung_partners=rp, policy=POL,
                   steps_so_far={}, report_interval=500, timestep_fs=4.0, n_gpus=4, budget_hours=100.0)
    assert p.reason == "no_diagnostics" and p.state_ids == ()


def test_a_healthy_campaign_gets_no_topup():
    p = _plan(_diag({s: 0.05 for s in range(8)}), 4)
    assert p.reason == "healthy" and p.state_ids == () and p.steps == 0


def test_unmeasured_edges_never_make_a_state_deficient():
    d = _diag({s: 0.05 for s in range(8)}, edges={})       # no overlap measured at all
    assert _plan(d, 4).reason == "healthy"


def test_one_deficit_state_gets_minimal_partners_not_the_whole_layout():
    sigma = {s: 0.05 for s in range(8)}; sigma[1] = 0.15
    p = _plan(_diag(sigma), 4)
    assert p.reason == "planned" and p.deficit_state_ids == (1,)
    assert set(p.state_ids) == {1} | set(p.partner_state_ids)
    assert len(p.partner_state_ids) == 2                    # one same-rung, one other-rung
    assert 5 in p.partner_state_ids                         # its rung partner
    assert p.predicted_sigma[1] <= POL.topup_target_sigma + 1e-9
    assert p.steps % 500 == 0 and p.steps > 0


def test_a_weak_edge_between_healthy_states_is_structural_not_md():
    sigma = {s: 0.05 for s in range(8)}
    p = _plan(_diag(sigma, edges={(1, 2): 0.02}), 4)
    assert p.structural_edges == ((1, 2),) and p.reason == "healthy"


def test_a_weak_edge_touching_a_deficit_tops_up_both_endpoints():
    sigma = {s: 0.05 for s in range(8)}; sigma[1] = 0.15
    p = _plan(_diag(sigma, edges={(1, 2): 0.02}), 4)
    assert {1, 2} <= set(p.deficit_state_ids) and p.structural_edges == ()


def test_nan_sigma_is_not_a_topup_target():
    sigma = {s: 0.05 for s in range(8)}; sigma[3] = math.nan
    assert _plan(_diag(sigma), 4).reason == "healthy"


def test_a_budget_smaller_than_one_useful_segment_plans_nothing():
    sigma = {s: 0.05 for s in range(8)}; sigma[1] = 0.15
    p = _plan(_diag(sigma), 4, budget=1e-6)
    assert p.reason == "cap_too_small" and p.state_ids == () and p.steps == 0


def test_a_deficit_at_the_layout_edge_still_gets_a_partner():
    sigma = {s: 0.05 for s in range(8)}; sigma[0] = 0.15
    nb = {0: [], 1: [0], 2: [], 3: [], 4: [], 5: [], 6: [], 7: []}   # window 0 has no same-rung neighbour
    rp = {0: [4], 4: [0], 1: [5], 5: [1], 2: [6], 6: [2], 3: [7], 7: [3]}
    p = plan_topup(_diag(sigma), state_ids_in_order=list(range(8)), neighbours=nb, rung_partners=rp,
                   policy=POL, steps_so_far={s: 100_000 for s in range(8)}, report_interval=500,
                   timestep_fs=4.0, n_gpus=4, budget_hours=100.0)
    assert p.reason == "planned" and 4 in p.partner_state_ids


def test_uniformly_deficient_states_degenerate_to_all_states():
    p = _plan(_diag({s: 0.30 for s in range(8)}), 4)
    assert set(p.state_ids) == set(range(8)) and p.partner_state_ids == ()


def test_a_state_that_underdelivered_needs_more_steps():
    sigma = {s: 0.05 for s in range(8)}; sigma[1] = 0.15
    normal = _plan(_diag(sigma), 4)
    penalised = _plan(_diag(sigma), 4, correction={1: 0.5})
    assert penalised.steps > normal.steps
```

- [ ] **Step 2: Run to verify it fails** — expected `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

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

Note: in a lockstep segment every deficit runs length L, so the patch is fixed and only L varies; states needing more than L carry to the next epoch (the diagnostics then still show them deficient).

- [ ] **Step 4: Run tests** — expected PASS. If `test_uniformly_deficient_states_degenerate_to_all_states` finds partners, the partner rule is adding non-deficit states when every state is a deficit; fix `partners_for`, not the test.
- [ ] **Step 5: Commit** — `git add gareus/adaptive/topup_allocator.py tests/test_topup_allocator.py && git commit -m "feat: deficit-driven wall-hour top-up allocator"`

---

### Task 7: Unmeasured is never weak (gate and remaining weak-edge readers)

**Files:**
- Modify: `gareus/adaptive_production.py` — `evaluate_adaptive_convergence_gate` (the loop `weak = overlap is None or ...`), `_weak_edge_touch_counts`
- Test: `tests/test_unmeasured_not_weak.py`

**Interfaces:**
- Consumes: edge dicts with keys `overlap`, `mbar_overlap`, `edge_type`, `exchange_acceptance`.
- Produces: gate `weak_edges` counts only edges with a measured value below threshold; rung edges are judged on `mbar_overlap` only.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_unmeasured_not_weak.py
from gareus.adaptive_production import AdaptiveDecisionPolicy, _edge_is_measured_weak


POL = AdaptiveDecisionPolicy()


def test_an_unmeasured_rung_edge_is_not_weak():
    assert _edge_is_measured_weak({"edge_type": "rung", "overlap": None, "mbar_overlap": None}, POL) is False


def test_a_measured_low_rung_edge_is_weak():
    assert _edge_is_measured_weak({"edge_type": "rung", "mbar_overlap": 0.05}, POL) is True


def test_a_spatial_edge_uses_its_cv_overlap():
    assert _edge_is_measured_weak({"edge_type": "nearest_2d", "overlap": 0.9}, POL) is False
    assert _edge_is_measured_weak({"edge_type": "nearest_2d", "overlap": 0.1}, POL) is True
    assert _edge_is_measured_weak({"edge_type": "nearest_2d", "overlap": None}, POL) is False


def test_low_measured_acceptance_still_counts_for_spatial_edges():
    assert _edge_is_measured_weak({"edge_type": "nearest_2d", "overlap": 0.9,
                                   "exchange_acceptance": 0.01}, POL) is True
```

- [ ] **Step 2: Run to verify it fails** — expected `ImportError: cannot import name '_edge_is_measured_weak'`.

- [ ] **Step 3: Implement** — add next to `_weak_edge_touch_counts`:

```python
def _edge_is_measured_weak(edge: Dict[str, Any], policy: AdaptiveDecisionPolicy) -> bool:
    """Weak only if MEASURED below threshold; an unmeasured edge is never weak.

    Rung edges are judged on the energy-space ``mbar_overlap`` alone (their CV
    overlap is ~1 by construction and gibbs-walk inflates their acceptance).
    """
    if str(edge.get("edge_type")) == "rung":
        value = edge.get("mbar_overlap")
        return value is not None and float(value) < float(policy.topup_weak_overlap)
    overlap = edge.get("overlap")
    weak = overlap is not None and float(overlap) < float(policy.target_overlap)
    acc = edge.get("exchange_acceptance")
    if acc is not None and float(acc) < float(policy.min_exchange_acceptance):
        weak = True
    return weak
```

Replace the body of the gate loop with `if _edge_is_measured_weak(edge, policy): weak_edges.append(edge)` and the `weak = ...` lines in `_weak_edge_touch_counts` with `weak = _edge_is_measured_weak(edge, policy)`.

- [ ] **Step 4: Run tests** — `tests/test_unmeasured_not_weak.py tests/test_epoch_loop_convergence_guard.py tests/test_adaptive_ladder_rungs.py tests/test_campaign_scoped_ladder_active.py`; expected PASS. If a ladder test asserted the old "missing reads weak" behaviour, update it to the new rule and say so in the commit body.
- [ ] **Step 5: Commit** — `git commit -am "fix: an unmeasured edge is never weak in the gate or allocator"`

---

### Task 8: Wire diagnostics, allocator and one top-up into scheduled phases

**Files:**
- Modify: `gareus/adaptive_production.py` — `build_adaptive_epoch_schedule` (remove score distribution), `run_scheduled_adaptive_epoch` (baseline sizing, allocator call, single top-up, plan persistence, recalibration)
- Create: `gareus/adaptive/topup_state.py` (plan and calibration persistence)
- Test: `tests/test_topup_epoch_wiring.py`

**Interfaces:**
- Consumes: `build_union_state_mbar_inputs(adaptive_dir, registry, include_epochs=True)` (existing), `union_diagnostics_from_npz` (Task 5), `plan_topup` (Task 6), `build_geometry_edges(registry, policy)` (existing, returns `(a, b, edge_type, normalized_distance)`), `spatial_neighbour_pairs` / `same_rung_neighbours` / `other_rung_same_centre` (Task 1).
- Produces: `topup_state.save_plan(epoch_dir, plan)`, `topup_state.load_plan(epoch_dir) -> TopupPlan | None`, `topup_state.load_correction(adaptive_dir) -> dict`, `topup_state.update_correction(adaptive_dir, plan, realised_sigma: dict, sigma_before: dict) -> dict`. Files: `<epoch_dir>/topup_plan.json`, `<adaptive_dir>/topup_calibration.json`.

Behaviour:
1. `build_adaptive_epoch_schedule` gives every active state `requested_steps = default_steps`, `baseline_steps = default_steps`, `reason = "baseline"`; the score/weak-count distribution is deleted (the old allocator). Keep its signature and the schedule file format so resume keeps working.
2. In `run_scheduled_adaptive_epoch`: if `policy.topups_enabled` is false, keep the current baseline-only path (PR #98). If true: baseline runs `default_steps * (1 - policy.topup_max_fraction)` steps (quantized to 1000); after it completes and before diagnostics, compute the plan:
   - if `<epoch_dir>/topup_plan.json` exists (resume) → load it, do not recompute;
   - else: `build_union_state_mbar_inputs(adaptive_dir, registry)` → `union_diagnostics_from_npz(meta["arrays_npz"], edges, kt_kcal=0.0019872041 * temperature_k)` → neighbours from `spatial_neighbour_pairs` on the registry's active states → `plan_topup(...)` with `budget_hours = wall_hours(default_steps, n_active, timestep_fs, n_gpus, table) * policy.topup_max_fraction`, `steps_so_far` from the union NPZ's per-state sample counts × `report_interval`, `correction = load_correction(adaptive_dir)`; save the plan.
   - print one line: `top-up plan: <reason>, <n> states (<n_deficit> deficit + <n_partner> partners), <steps> steps, <cost_hours:.2f} h`;
   - if `plan.reason == "planned"`: `run_segment(f"topup_001_{plan.steps}", plan.state_ids, plan.steps)`; then rebuild the union diagnostics and call `update_correction(adaptive_dir, plan, realised_sigma)`.
   - write the union overlaps into the epoch diagnostics: for every edge dict in `diagnostics["edges"]`, set `edge["mbar_overlap"] = diag.edge_overlap[(min, max)]` when measured. The existing proposer already turns a measured low `mbar_overlap` into `add_rung` and a low spatial overlap into a bridge, so structural edges reach it with no new key, and rung gaps are now caught every epoch instead of only at campaign end. Save `plan.structural_edges` in `topup_plan.json` for the record.
3. `n_gpus` = number of entries in `args.device_index` split on commas; `report_interval` = `args.report_interval`; `timestep_fs` = `args.timestep_fs`; `temperature_k` = `args.temperature_k`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_topup_epoch_wiring.py
import json
from pathlib import Path

import pytest

import gareus.adaptive_production as ap
import gareus.production as prod
from gareus.adaptive.topup_allocator import TopupPlan
from gareus.adaptive.topup_state import load_correction, load_plan, save_plan, update_correction
from gareus.lifecycle import _graceful_shutdown

from test_scheduled_final_interruption import _scheduled_final_campaign


@pytest.fixture(autouse=True)
def _clear():
    _graceful_shutdown.clear(); yield; _graceful_shutdown.clear()


def _drive(monkeypatch, args, out, plan):
    calls = []
    monkeypatch.setattr(prod, "run_gareus", lambda a, d, *r, **k: calls.append((Path(d).name, int(a.gamd_production_steps))))
    monkeypatch.setattr(ap, "_topup_plan_for_phase", lambda *a, **k: plan)
    ap.run_adaptive_production_auto_loop(args, out, None, None, None, None, None, None)
    return calls


def test_schedule_is_uniform_no_score_distribution(tmp_path):
    args, out = _scheduled_final_campaign(tmp_path)
    reg = ap.WindowStateRegistry.load(Path(out) / "adaptive_production")
    rows = ap.build_adaptive_epoch_schedule(reg, {"edges": [{"state_i": 0, "state_j": 1, "overlap": None}]},
                                            ap.AdaptiveDecisionPolicy(), epoch=1, default_steps=10_000)
    assert {r["requested_steps"] for r in rows} == {10_000}


def test_topups_on_runs_a_shortened_baseline_and_exactly_one_topup(tmp_path, monkeypatch):
    args, out = _scheduled_final_campaign(tmp_path)
    args.adaptive_production_topups = True
    plan = TopupPlan(state_ids=(0, 1), steps=2000, deficit_state_ids=(0,), partner_state_ids=(1,), reason="planned")
    calls = _drive(monkeypatch, args, out, plan)
    names = [n for n, _ in calls]
    assert names.count("baseline") == 1 and [n for n in names if n.startswith("topup_")] == ["topup_001_2000"]


def test_a_healthy_plan_runs_no_topup(tmp_path, monkeypatch):
    args, out = _scheduled_final_campaign(tmp_path)
    args.adaptive_production_topups = True
    calls = _drive(monkeypatch, args, out, TopupPlan(reason="healthy"))
    assert not [n for n, _ in calls if n.startswith("topup_")]


def test_a_saved_plan_survives_resume(tmp_path):
    plan = TopupPlan(state_ids=(3, 4), steps=5000, deficit_state_ids=(3,), partner_state_ids=(4,),
                     predicted_sigma={3: 0.09, 4: 0.05}, cost_hours=1.5, reason="planned")
    save_plan(tmp_path, plan)
    assert load_plan(tmp_path) == plan


def test_calibration_halves_priority_after_underdelivery(tmp_path):
    plan = TopupPlan(state_ids=(3,), steps=5000, deficit_state_ids=(3,), predicted_sigma={3: 0.09},
                     reason="planned")
    before = {3: 0.20}
    corr = update_correction(tmp_path, plan, realised_sigma={3: 0.18}, sigma_before=before)
    assert corr[3] <= 0.5 and load_correction(tmp_path)[3] == corr[3]
```

- [ ] **Step 2: Run to verify they fail** — expected import errors (`topup_state`, `_topup_plan_for_phase`).

- [ ] **Step 3: Implement `gareus/adaptive/topup_state.py`**

```python
"""Top-up plan persistence (resume-stable) and online gain calibration."""
from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional

from .topup_allocator import TopupPlan

PLAN_NAME = "topup_plan.json"
CALIBRATION_NAME = "topup_calibration.json"


def save_plan(epoch_dir, plan: TopupPlan) -> Path:
    path = Path(epoch_dir) / PLAN_NAME
    payload = asdict(plan)
    payload["predicted_sigma"] = {str(k): v for k, v in plan.predicted_sigma.items()}
    payload["sigma_before"] = {str(k): v for k, v in plan.sigma_before.items()}
    path.write_text(json.dumps(payload, indent=2))
    return path


def load_plan(epoch_dir) -> Optional[TopupPlan]:
    path = Path(epoch_dir) / PLAN_NAME
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    d["predicted_sigma"] = {int(k): float(v) for k, v in d.get("predicted_sigma", {}).items()}
    d["sigma_before"] = {int(k): float(v) for k, v in d.get("sigma_before", {}).items()}
    for key in ("state_ids", "deficit_state_ids", "partner_state_ids"):
        d[key] = tuple(int(x) for x in d.get(key, ()))
    d["structural_edges"] = tuple(tuple(int(x) for x in e) for e in d.get("structural_edges", ()))
    return TopupPlan(**d)


def load_correction(adaptive_dir) -> Dict[int, float]:
    path = Path(adaptive_dir) / CALIBRATION_NAME
    if not path.exists():
        return {}
    return {int(k): float(v) for k, v in json.loads(path.read_text()).items()}


def update_correction(adaptive_dir, plan: TopupPlan, realised_sigma: Dict[int, float],
                      sigma_before: Dict[int, float]) -> Dict[int, float]:
    """factor = realised variance reduction / predicted; < 0.5 of prediction halves priority."""
    corr = load_correction(adaptive_dir)
    for s in plan.deficit_state_ids:
        pred, real, before = plan.predicted_sigma.get(s), realised_sigma.get(s), sigma_before.get(s)
        if not all(isinstance(v, float) and math.isfinite(v) for v in (pred, real, before)) or before <= pred:
            continue
        ratio = (before ** 2 - real ** 2) / (before ** 2 - pred ** 2)
        factor = max(0.05, min(2.0, ratio)) * corr.get(s, 1.0)
        corr[s] = factor * 0.5 if ratio < 0.5 else factor
    (Path(adaptive_dir) / CALIBRATION_NAME).write_text(json.dumps({str(k): v for k, v in corr.items()}, indent=2))
    return corr
```

- [ ] **Step 4: Implement the wiring in `gareus/adaptive_production.py`**

(a) In `build_adaptive_epoch_schedule`, replace everything after `state_rows = _state_rows_by_id(diagnostics)` with:

```python
    rows = []
    for state in active:
        sid = int(state.state_id)
        rows.append({
            "state_id": sid, "requested_steps": int(default_steps), "baseline_steps": int(default_steps),
            "extra_steps": 0, "score": 0.0,
            "sample_count": int((state_rows.get(sid) or {}).get("sample_count", 0) or 0),
            "allocation_reason": "baseline",
        })
    return rows
```

Delete `_weak_edge_touch_counts` and the now-unused allocation-score code; keep the `policy` fields (`weak_edge_bonus` etc.) only if other code reads them (grep before deleting).

(b) Add a module-level helper (monkeypatched by the tests):

```python
def _topup_plan_for_phase(args, epoch_dir: Path, registry: "WindowStateRegistry",
                          policy: AdaptiveDecisionPolicy, default_steps: int):
    """Resume-stable top-up plan for this phase: read it back if saved, else compute and save."""
    from .adaptive.topup_state import load_correction, load_plan, save_plan
    from .adaptive.topup_allocator import plan_topup, TopupPlan
    from .adaptive.union_diagnostics import union_diagnostics_from_npz
    from .adaptive.throughput import wall_hours
    from .layout_neighbours import other_rung_same_centre, same_rung_neighbours, spatial_neighbour_pairs

    saved = load_plan(epoch_dir)
    if saved is not None:
        return saved
    adaptive_dir = Path(epoch_dir).parent
    active = registry.active_states()
    ids = [int(s.state_id) for s in active]
    c1 = [float(s.primary_center) for s in active]
    c2 = [float(s.secondary_center) if s.secondary_center is not None else 0.0 for s in active]
    lam = [float(s.gamd_lambda or 0.0) for s in active]
    k1 = [float(s.primary_k) for s in active]
    k2 = [float(s.secondary_k or 0.0) for s in active]
    temperature = float(getattr(args, "temperature_k", 300.0) or 300.0)
    pairs = spatial_neighbour_pairs(c1, c2, lam, k1, k2, temperature)
    nb_local = same_rung_neighbours(pairs, lam)
    rp_local = other_rung_same_centre(c1, c2, lam)
    neighbours = {ids[w]: [ids[x] for x in nb_local.get(w, [])] for w in range(len(ids))}
    rung_partners = {ids[w]: [ids[x] for x in rp_local.get(w, [])] for w in range(len(ids))}
    edges = [(a, b) for a, b, _t, _d in build_geometry_edges(registry, policy)]
    try:
        meta = build_union_state_mbar_inputs(adaptive_dir, registry)
        diag = union_diagnostics_from_npz(meta["arrays_npz"], edges, kt_kcal=0.0019872041 * temperature)
    except Exception as exc:
        print(f"      top-up diagnostics unavailable ({exc}); no top-up this phase")
        diag = None
    interval = int(getattr(args, "report_interval", 5000) or 5000)
    n_gpus = max(1, len(str(getattr(args, "device_index", "0")).split(",")))
    timestep = float(getattr(args, "timestep_fs", 4.0) or 4.0)
    table = policy.topup_throughput_table
    steps_so_far = {s: int(diag.n_k.get(s, 0)) * interval for s in ids} if diag else {}
    budget = wall_hours(default_steps, len(ids), timestep, n_gpus, table) * float(policy.topup_max_fraction)
    plan = plan_topup(diag, state_ids_in_order=ids, neighbours=neighbours, rung_partners=rung_partners,
                      policy=policy, steps_so_far=steps_so_far, report_interval=interval,
                      timestep_fs=timestep, n_gpus=n_gpus, budget_hours=budget,
                      correction=load_correction(adaptive_dir))
    save_plan(epoch_dir, plan)
    return plan
```

(c) In `run_scheduled_adaptive_epoch`: replace the `topups_enabled` block from PR #98 and the `groups` loop with:

```python
    topups_enabled = bool(getattr(policy, "topups_enabled", False)) or _arg_bool(args, "adaptive_production_topups", False)
    if topups_enabled:
        baseline_steps = max(1000, _quantized_extra_steps(int(baseline_steps * (1.0 - float(policy.topup_max_fraction)))))
    else:
        requested = [int(r.get("requested_steps", 0) or 0) for r in schedule if int(r.get("requested_steps", 0) or 0) > 0]
        baseline_steps = max(baseline_steps, _quantized_extra_steps(int(round(sum(requested) / len(requested)))))
```

and after `run_segment("baseline", ...)` plus its interruption check:

```python
    if topups_enabled:
        plan = _topup_plan_for_phase(args, epoch_dir, registry, policy, default_steps=baseline_steps)
        print(f"      top-up plan: {plan.reason}, {len(plan.state_ids)} states "
              f"({len(plan.deficit_state_ids)} deficit + {len(plan.partner_state_ids)} partners), "
              f"{plan.steps} steps, {plan.cost_hours:.2f} h")
        if plan.reason == "planned" and plan.state_ids and plan.steps > 0:
            run_segment(f"topup_001_{plan.steps}", list(plan.state_ids), int(plan.steps))
            if _graceful_shutdown.is_set():
                payload = {
                    "schema_version": "adaptive_scheduled_epoch_v1",
                    "status": "interrupted_after_checkpoint",
                    "epoch_dir": str(epoch_dir),
                    "baseline_steps": int(baseline_steps),
                    "segments": segment_summaries,
                    "diagnostics_json": "",
                    "schedule_csv": str(epoch_dir / "epoch_schedule.csv"),
                }
                write_json(epoch_dir / "scheduled_epoch_summary.json", payload)
                return {"summary": payload, "diagnostics": {}}
```

(Same payload as the existing baseline-interruption return a few lines above; keep both explicit in this task.) After the top-up segment, recompute diagnostics with the same helper minus the saved-plan short-circuit (call `union_diagnostics_from_npz` directly), then `update_correction(adaptive_dir, plan, realised_sigma=diag_after.sigma_kcal, sigma_before=plan.sigma_before)` (`TopupPlan.sigma_before` is filled by `plan_topup` in Task 6 and persisted by `save_plan`).

Remove the leftover old-path code (`groups`, `_quantized_extra_steps(... - baseline_steps)` grouping).

- [ ] **Step 5: Run tests**

Run: `python -m pytest -q -p no:cacheprovider tests/test_topup_epoch_wiring.py tests/test_no_topups.py tests/test_scheduled_final_interruption.py tests/test_epoch_window_map_rewrite_after_drop.py tests/test_adaptive_segmented_diagnostics.py tests/test_ap_epoch0_step_fraction.py`
Expected: PASS. Tests in `test_epoch_window_map_rewrite_after_drop.py` that construct top-up segments through the old grouping must be updated to set `adaptive_production_topups = True` and monkeypatch `_topup_plan_for_phase`; say so in the commit body.

- [ ] **Step 6: Commit**

```bash
git add gareus/adaptive_production.py gareus/adaptive/topup_state.py gareus/adaptive/topup_allocator.py tests/
git commit -m "feat: scheduled phases run at most one allocator-planned top-up; remove score allocator"
```

---

### Task 9: Top-up seeding from the baseline's final window States

**Files:**
- Modify: `gareus/production.py` — segment end (next to the `final_pdbs` writer, `# Save one final PDB per configuration replica.`), the fresh-path call `generate_us_starting_states_by_pulling(` in `run_gareus`, and the production replica construction that consumes `window_start_positions` / `window_start_velocities` (add per-window box).
- Create: `gareus/topup_seeding.py`
- Test: `tests/test_topup_seeding.py`

**Interfaces:**
- Produces: `export_final_window_states(out_dir, sims, assignments, state_id_of_window, cv_of_replica) -> Path` writing `<out_dir>/final_window_states/state_<sid>.xml` (OpenMM `XmlSerializer` State with positions, velocities, box) and `index.json` (`{sid: {"window": w, "cv1": x, "cv2": y}}`).
- Produces: `load_seed_states(parent_dirs, state_ids) -> dict[int, SeedState]` with `SeedState(positions, velocities, box, cv1, cv2, source)`; missing states are absent from the dict.
- Produces: `assert_seed_matches(seed: SeedState, cv1_now: float, cv2_now: float, tol: float = 1e-4) -> None` raising `SeedMismatchError`.

Behaviour in `run_gareus` fresh path, when `args._adaptive_phase_info["is_topup"]` is true: parent dirs = the same phase's `baseline/` (sibling of this segment dir) then earlier top-ups; for windows with a seed state, set `window_start_positions[w]`, `window_start_velocities[w]`, and a new `window_start_boxes[w]`; call `generate_us_starting_states_by_pulling` only for the windows without one (warning listing their state_ids); if every window has a seed state, skip the pull entirely. After replica construction and `set_window`, evaluate each replica's CV (use the CV evaluation already used for sample logging in `run_gareus`) and call `assert_seed_matches`; on mismatch raise (aborts the top-up; the driver's interruption handling keeps the campaign resumable).

- [ ] **Step 1: Write the failing tests** (pure part; the OpenMM part uses the Reference platform fixture)

```python
# tests/test_topup_seeding.py
import json

import pytest

from gareus.topup_seeding import SeedMismatchError, SeedState, assert_seed_matches, load_seed_states


def _fake_export(d, sid, cv1, cv2, xml="<State/>"):
    (d / "final_window_states").mkdir(parents=True, exist_ok=True)
    (d / "final_window_states" / f"state_{sid}.xml").write_text(xml)
    idx = d / "final_window_states" / "index.json"
    data = json.loads(idx.read_text()) if idx.exists() else {}
    data[str(sid)] = {"window": sid, "cv1": cv1, "cv2": cv2}
    idx.write_text(json.dumps(data))


def test_loads_only_the_states_it_has_and_prefers_the_latest_parent(tmp_path, monkeypatch):
    import gareus.topup_seeding as ts
    monkeypatch.setattr(ts, "_deserialize_state", lambda text: ("pos", "vel", "box"))
    base, top = tmp_path / "baseline", tmp_path / "topup_001_1000"
    _fake_export(base, 3, 0.1, 0.2); _fake_export(base, 4, 0.3, 0.4); _fake_export(top, 3, 0.5, 0.6)
    got = load_seed_states([base, top], [3, 4, 9])
    assert set(got) == {3, 4}
    assert got[3].cv1 == 0.5 and got[3].source.endswith("topup_001_1000")


def test_assertion_accepts_a_matching_frame_and_rejects_a_swapped_one():
    seed = SeedState(positions=None, velocities=None, box=None, cv1=0.40, cv2=-0.25, source="x")
    assert_seed_matches(seed, 0.40, -0.25)
    with pytest.raises(SeedMismatchError, match="state"):
        assert_seed_matches(seed, 0.47, -0.25)


def test_export_and_reload_round_trip_on_the_reference_platform(tmp_path):
    pytest.importorskip("openmm")
    from pep_gamd_fixture import build_small_system  # tests/pep_gamd_fixture.py
    from gareus.topup_seeding import export_final_window_states
    sim = build_small_system(platform="Reference")
    export_final_window_states(tmp_path, [sim], assignments=[0], state_id_of_window={0: 7},
                               cv_of_replica=lambda r: (0.11, 0.22))
    seed = load_seed_states([tmp_path], [7])[7]
    st = sim.context.getState(getPositions=True, getVelocities=True)
    assert seed.cv1 == 0.11 and len(seed.positions) == len(st.getPositions())
```

If `tests/pep_gamd_fixture.py` has no `build_small_system`, add a thin wrapper there that returns an `app.Simulation` on the GA dipeptide system it already builds (read the fixture first; reuse its builder, do not duplicate the system setup).

- [ ] **Step 2: Run to verify they fail** — expected `ModuleNotFoundError: gareus.topup_seeding`.

- [ ] **Step 3: Implement `gareus/topup_seeding.py`**

```python
"""Continue each top-up window's chain from its parent segment's final State (spec 4.4)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Sequence

DIR_NAME = "final_window_states"


class SeedMismatchError(RuntimeError):
    pass


@dataclass(frozen=True)
class SeedState:
    positions: Any
    velocities: Any
    box: Any
    cv1: float
    cv2: float
    source: str


def _deserialize_state(text: str):
    from openmm import XmlSerializer  # noqa: PLC0415
    st = XmlSerializer.deserialize(text)
    return st.getPositions(), st.getVelocities(), st.getPeriodicBoxVectors()


def export_final_window_states(out_dir, sims: Sequence, assignments: Sequence[int],
                               state_id_of_window: Dict[int, int],
                               cv_of_replica: Callable[[int], tuple]) -> Path:
    from openmm import XmlSerializer  # noqa: PLC0415
    d = Path(out_dir) / DIR_NAME
    d.mkdir(parents=True, exist_ok=True)
    index = {}
    for r, sim in enumerate(sims):
        w = int(assignments[r])
        sid = int(state_id_of_window.get(w, w))
        st = sim.context.getState(getPositions=True, getVelocities=True, enforcePeriodicBox=False)
        (d / f"state_{sid}.xml").write_text(XmlSerializer.serialize(st))
        cv1, cv2 = cv_of_replica(r)
        index[str(sid)] = {"window": w, "cv1": float(cv1), "cv2": float(cv2)}
    (d / "index.json").write_text(json.dumps(index, indent=2))
    return d


def load_seed_states(parent_dirs: Iterable, state_ids: Iterable[int]) -> Dict[int, SeedState]:
    wanted = {int(s) for s in state_ids}
    out: Dict[int, SeedState] = {}
    for parent in parent_dirs:                       # later parents override earlier ones
        d = Path(parent) / DIR_NAME
        idx_path = d / "index.json"
        if not idx_path.exists():
            continue
        index = json.loads(idx_path.read_text())
        for key, rec in index.items():
            sid = int(key)
            xml = d / f"state_{sid}.xml"
            if sid not in wanted or not xml.exists():
                continue
            pos, vel, box = _deserialize_state(xml.read_text())
            out[sid] = SeedState(pos, vel, box, float(rec["cv1"]), float(rec["cv2"]), str(parent))
    return out


def assert_seed_matches(seed: SeedState, cv1_now: float, cv2_now: float, tol: float = 1e-4) -> None:
    if abs(cv1_now - seed.cv1) > tol or abs(cv2_now - seed.cv2) > tol:
        raise SeedMismatchError(
            f"seeded state from {seed.source} does not reproduce its recorded CVs "
            f"(cv1 {cv1_now:.6f} vs {seed.cv1:.6f}, cv2 {cv2_now:.6f} vs {seed.cv2:.6f}): "
            "the parent manifest's window assignment is out of step with its final frames")
```

- [ ] **Step 4: Wire into `production.py`**
  - Export: immediately after the `final_pdbs` loop, call `export_final_window_states(out_dir, sims, assignments, state_id_of_window, cv_of_replica)`; `state_id_of_window` comes from this segment's `epoch_window_map.csv` (read with `_read_csv_dicts`; identity map when absent); `cv_of_replica` evaluates CV1/CV2 on `sims[r].context` with the same function the sample logger uses (find it: `grep -n "cv_A" gareus/production.py | head` near the per-step logging; reuse, do not reimplement).
  - Seeding: in the fresh path just before `generate_us_starting_states_by_pulling(`, when `(getattr(args, "_adaptive_phase_info", {}) or {}).get("is_topup")`: load seeds from `[out_dir.parent / "baseline", *sorted(out_dir.parent.glob("topup_*"))]` excluding `out_dir`; build the window→state_id map from this segment's `epoch_window_map.csv`; fill `window_start_positions/velocities/boxes`; pull only the missing windows (pass the reduced window list to the existing call, then merge); print `top-up seeding: <n> windows continued from <source>, <m> pulled (state_ids ...)`.
  - Box: in the production replica construction that uses `window_start_positions[i]`, add `box = window_start_boxes[i] if window_start_boxes and window_start_boxes[i] is not None else equil_box` and use it in `setPeriodicBoxVectors`. Default `window_start_boxes = [None] * nrep` wherever `window_start_positions` is initialised.
  - Assertion: after the production replicas are built and `set_window` applied, for each seeded window call `assert_seed_matches(seed, *cv_of_replica(r))`.

- [ ] **Step 5: Run tests** — `tests/test_topup_seeding.py tests/test_pep_gamd_wiring.py tests/test_resume_round_trip.py`; expected PASS (the Reference round-trip may be skipped where OpenMM is absent; it must run locally).
- [ ] **Step 6: Commit** — `git add gareus/topup_seeding.py gareus/production.py tests/ && git commit -m "feat: top-ups continue each window from the parent segment's final State"`

---

### Task 10: Synthetic harness — rungs, mixing, lockstep, union solve, A/B study

**Files:**
- Modify: `gareus/synth/sampler.py` (`Window` gets `lam: float = 0.0`; `BIAS` unchanged; new `boost_dv(landscape, window, cv1, cv2)`), `gareus/synth/ess.py` (`tau_int(..., n_partners: int = 2)` scaling)
- Create: `gareus/synth/topup_study.py`
- Test: `tests/test_synth_topup_study.py`

**Interfaces:**
- Consumes: `plan_topup` (Task 6), `union_diagnostics_from_npz` (Task 5), `wall_hours` (Task 3), existing `Landscape`, `sample_window_exact`, `reference_pmf`.
- Produces: `run_arm(landscape, *, arm: str, seed: int, wall_hours_budget: float, ...) -> ArmResult(max_sigma: float, pmf_rmse: float, hours_used: float, topup_md_fraction: float, structural_routed: tuple)`; `run_study(landscapes, *, n_seeds: int = 20, out: Path) -> dict` writing `topup_study.json`; CLI `python -m gareus.synth.topup_study --n-seeds 20 --out <dir>`.

Model (spec 6.1):
- windows on the landscape's default 2D ladder × rungs λ ∈ {0, 1/3, 2/3, 1}; the boost is a λ-scaled flattening `dV = λ * a * max(0, E_ref - F(x))` with `a = 0.5`, `E_ref` = 60th percentile of F on the grid, so rung states at one centre are distinct Hamiltonians; samples are drawn from the window's biased+boosted density; reduced potentials in every state = umbrella bias + boost term of that state (closed form);
- τ per window from `tau_int` (existing landscape steepness) × `2 / max(1, n_partners)` where `n_partners` = the window's exchange partners present in its segment;
- a segment samples all its windows for the same length; the number of decorrelated samples per window = length / (interval · (1 + 2τ)); wall time is charged with `wall_hours` and the same throughput table the allocator uses;
- the pooled samples are written in the union NPZ format and solved with `union_diagnostics_from_npz` (real production code path);
- arm A: uniform baseline only for the whole budget; arm B: per epoch, baseline at (1 − cap) then `plan_topup` with the same budget;
- 3 epochs per campaign; equal modelled wall-hours per arm (arm A absorbs arm B's unused top-up hours into its baseline).

- [ ] **Step 1: Write the failing tests** (small N so the suite stays fast; the 20-seed study is the CLI)

```python
# tests/test_synth_topup_study.py
import numpy as np

from gareus.synth import LANDSCAPES
from gareus.synth.topup_study import run_arm, run_study


def test_arms_spend_equal_modelled_wall_hours():
    ls = LANDSCAPES["rugged_2d"]()
    a = run_arm(ls, arm="uniform", seed=1, wall_hours_budget=5.0)
    b = run_arm(ls, arm="topup", seed=1, wall_hours_budget=5.0)
    assert abs(a.hours_used - b.hours_used) / a.hours_used < 0.02


def test_topups_beat_uniform_on_a_heterogeneous_landscape():
    ls = LANDSCAPES["gated_barrier"]()
    wins = [run_arm(ls, arm="topup", seed=s, wall_hours_budget=5.0).max_sigma
            < run_arm(ls, arm="uniform", seed=s, wall_hours_budget=5.0).max_sigma for s in range(5)]
    assert sum(wins) >= 4


def test_topups_match_uniform_on_a_homogeneous_landscape():
    ls = LANDSCAPES["mixture_wells"]()
    a = np.median([run_arm(ls, arm="uniform", seed=s, wall_hours_budget=5.0).max_sigma for s in range(5)])
    b = np.median([run_arm(ls, arm="topup", seed=s, wall_hours_budget=5.0).max_sigma for s in range(5)])
    assert abs(b - a) / a < 0.05


def test_a_missing_bridge_is_routed_not_topped_up():
    ls = LANDSCAPES["gated_barrier"]()
    r = run_arm(ls, arm="topup", seed=2, wall_hours_budget=5.0, remove_bridge=True)
    assert r.structural_routed


def test_study_writes_a_summary(tmp_path):
    summary = run_study(["rugged_2d"], n_seeds=2, out=tmp_path)
    assert (tmp_path / "topup_study.json").exists() and "rugged_2d" in summary
```

Use the landscape names that exist in `gareus/synth/landscapes.py` (`LANDSCAPES`): `rugged_2d`, `gated_barrier`, `slow_cv2_double_branch` for heterogeneous; for homogeneous use the smoothest available (`mixture_wells` if it is near-uniform in difficulty; otherwise add a `harmonic_bowl` landscape to `landscapes.py` in this task with a one-line docstring).

- [ ] **Step 2: Run to verify they fail** — expected `ModuleNotFoundError: gareus.synth.topup_study`.

- [ ] **Step 3: Implement** `gareus/synth/topup_study.py` per the model above (functions `_ladder(landscape)`, `_sample_segment(landscape, windows, length, rng, n_partners)`, `_write_union_npz(path, samples, windows)`, `run_arm`, `run_study`, `main`). Reuse `sample_window_exact` for draws and `reference_pmf` for the RMSE along CV1. Keep the file under 400 lines; put the boost closed form in `sampler.py` (`boost_dv`) so it is testable on its own.

- [ ] **Step 4: Run tests** — expected PASS. If `test_topups_beat_uniform_on_a_heterogeneous_landscape` fails, inspect `topup_study.json` per epoch (plan reasons, deficits, hours): a real allocator problem is fixed in Task 6's module with a new unit test there; do not loosen this test.

- [ ] **Step 5: Run the full synthetic study** (not a test; the spec's validation):

```bash
python -m gareus.synth.topup_study --n-seeds 20 --out docs/superpowers/specs/2026-09-24-effective-topups/synth_study
```

Record in the commit body: per landscape, median max σ and PMF RMSE for both arms and the paired one-sided p-value (Wilcoxon signed-rank on per-seed max σ). Pass criteria are spec section 6.2.

- [ ] **Step 6: Commit** — `git add gareus/synth/ tests/test_synth_topup_study.py docs/superpowers/specs/2026-09-24-effective-topups/synth_study && git commit -m "test: synthetic A/B study for effective top-ups"`

---

### Task 11: Documentation

**Files:**
- Modify: `gareus/helptext.py` (a `-hh` topic "Adaptive top-ups" with `Title\n-----` heading: off by default, what triggers them, lockstep and wall-hour cost, seeding, knobs, where the plan file lands)
- Modify: `CHANGELOG.md` (an `## [Unreleased]` entry: top-ups off by default, new allocator, knobs, seeding; results-changing note: campaigns with `ap_topups: true` now allocate differently)
- Modify: `CLAUDE.md` (a short section: files, invariants — unmeasured never weak, one top-up per phase, plan persisted for resume, seeding assertion)
- Modify: `docs/atlas-md/developer/topups-todo.md` (mark "synthetic validation" done with the study numbers)
- Test: `tests/test_helptext_topups.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_helptext_topups.py
from gareus.cli import build_gareus_parser
from gareus.helptext import render_encyclopedia_help, _METHOD_ENCYCLOPEDIA


def test_the_encyclopedia_has_a_topups_topic():
    text = render_encyclopedia_help(build_gareus_parser(), _METHOD_ENCYCLOPEDIA, topic="top-ups", color=False)
    assert "off by default" in text and "--ap-topup-max-fraction" in text
```

- [ ] **Step 2: Run to verify it fails** — expected no matching topic.
- [ ] **Step 3: Write the docs** (plain prose; the helptext heading underline must match the title length or the TOC drops the topic).
- [ ] **Step 4: Run** — `tests/test_helptext_topups.py`; also `python -m gareus -hh list | grep -i top-up` shows the topic.
- [ ] **Step 5: Commit** — `git commit -am "docs: adaptive top-ups help topic, changelog and handoff notes"`
