# Effective Adaptive Top-ups Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make scheduled top-ups spend MD only where a per-epoch union MBAR shows a statistical deficit, costed in wall-hours under the lockstep constraint, batched with minimal exchange partners, and seeded by continuing each window's chain from the baseline's final state; off by default; validated in the synthetic harness only.

**Architecture:** Pure modules under `gareus/adaptive/` (throughput model, union diagnostics, allocator, plan/calibration persistence) produce one `TopupPlan` per epoch. `run_scheduled_adaptive_epoch` runs the all-state baseline, calls the diagnostics and allocator, and runs at most one top-up segment. The union builder records per-state equilibration cut and statistical inefficiency, which the allocator uses. Production exports each window's final portable State at segment end; a top-up runs only windows that have one, continuing their chains. The synthetic harness gains rungs, a mixing model, lockstep segments and the real union solve, and runs the A/B study.

**Tech Stack:** Python 3.9 (aurum2 conda env `calc`) / 3.14 (local), numpy, scipy, pymbar 4.0.3, OpenMM 8.x (seeding only), existing `gareus` package.

**Spec:** `docs/superpowers/specs/2026-09-24-effective-topups-design.md` (read it first; this plan argues from it). Review records: `docs/superpowers/specs/2026-09-24-effective-topups/` (design board, plan board, `plan_code_check.log`).

**Revision 2 (2026-09-25).** Revision 1's pure code for Tasks 3-6 was executed verbatim in a scratch worktree (23/24 tests passed) and reviewed by the small board (ACCEPT_WITH_CHANGES, split 1-1, one judge failed). Folded in: (V2) split-halves threshold corrected for multiple comparisons with an effect-size floor, each neighbour pair tested once; (V3) the union inputs are already decorrelated, so the allocator uses the builder's kept counts and its per-state inefficiency instead of a second autocorrelation pass (old Task 4 replaced); (V4) top-up budget computed from the un-shortened default steps; plus from the board: budget-feasible maximum length as a candidate, clamped and smoothed calibration factor, never-sampled states excluded as partners, persistent weak-edge escalation to structural, atomic plan writes and layout validation on resume, MBAR initialised from zeros / previous epoch instead of BAR (states are not in overlap order), predicted vs realised wall-time logged. Rejected with reasons: "healthy epochs forfeit the reserved share" (the live pool already rolls it forward) and "allow +inf in u_kn" (harmonic biases are never infinite; NaN rows are dropped by design).

## Global Constraints

- Top-ups are **off by default**: `--ap-topups` default `False`; YAML key `ap_topups`.
- Defaults: `--ap-topup-target-sigma 0.10` (kcal/mol), `--ap-topup-weak-overlap 0.15`, `--ap-topup-max-fraction 0.3`, `--ap-topup-min-effect 0.05` (kcal/mol, split-halves practical floor), `--ap-topup-max-edge-attempts 2`.
- Throughput table default (contexts/GPU → ns/day/node): `16 → 3154.0`, `59 → 2300.0`; flat outside the measured range.
- An unmeasured edge is **never** weak (allocator and convergence gate).
- Healthy states get **zero** top-up steps; no default score; no distribution of left-over budget.
- A weak edge whose both endpoints are statistically adequate, or that stayed weak after `max_edge_attempts` top-ups, is **structural**: routed to the bridge/add proposal, never to MD.
- One top-up segment per epoch at most; all its replicas run the same number of steps (lockstep).
- **No pull inside a top-up.** A window without an exported final State (or whose State fails the seeding check) is dropped from that top-up, with a warning naming it; the next baseline exports it.
- The top-up budget is `topup_max_fraction × wall_hours(un-shortened default steps, all active states)`.
- The old allocator path (score ≥ 1 for every state, grouping by extra size, re-pull seeding for top-ups) is removed, not kept as a mode.
- Validation is synthetic only: no MD campaign, no benchmark job, no real-file spike in this plan.
- Tests run through opencode (pytest is hook-blocked in this repo): `opencode run "From the repo root run exactly: python -m pytest -q -p no:cacheprovider <files> > atlas_task.log 2>&1 ; then print the last 30 lines of atlas_task.log. Do not edit any files."`. Run only the test files named in the task (user preference: targeted tests only, never the full suite).
- Commit messages: conventional commits, ending with the session's `Co-Authored-By` line.

**Deviation from the spec (1), flagged — local σ_k:** the spec defines σ_k "relative to a fixed reference state", which makes the reference state's σ identically 0 and gives distant states an uncertainty accumulated along the whole chain that topping them up cannot fix. This plan uses σ_k = min over k's edge-neighbours j of pymbar's pairwise `dDelta_f[j, k]` (median of the row when k has no measured neighbour). Verified on a healthy chain: 0.021 kcal/mol for every state.

**Deviation from the spec (2), flagged — seeding route:** the spec makes the binary-checkpoint load the primary route. This plan exports each window's final positions, velocities and box at segment end (`final_window_states/`) and starts top-up windows from them through the existing start-state path; a binary checkpoint cannot be loaded into a Context of a different replica set without re-implementing the resume path. Exact positions + velocities with the frozen GaMD envelope continue the chain, so no burn-in is needed. Where the spec's fallback was "portable State + burn-in", this plan's fallback is stricter: the window is left out of that top-up.

## Review Focus

1. **A deficit state with no same-rung spatial neighbour** (layout edge, sparse row) — the patch builder still returns a valid patch (partner from another rung) instead of crashing or dropping the state. Test in Task 6.
2. **The union solve fails or gives NaN σ for some states** (zero samples, non-convergence) — the epoch runs no top-up and says why; never the old allocator, never a top-up of a NaN state. Tests in Tasks 5 and 6.
3. **Resume in the middle of a top-up, or after the layout changed** — the saved `topup_plan.json` is reused if its states are all still active, otherwise discarded with a warning and recomputed. Tests in Task 8.
4. **Exported final States missing or inconsistent for some windows** — those windows are dropped from the top-up with a warning; a `SeedMismatchError` inside the segment ends that top-up only, never the campaign. Tests in Tasks 8 and 9.
5. **Split-halves false alarms at production scale** — on an iid layout of 236 states the check flags at most ~α of epochs. Test in Task 5.

---

### Task 0: pymbar availability — hardening and a loud check (prerequisite)

Found while verifying revision 1 (2026-09-25): in aurum2's production environment (`/home/sulcjo/conda-envs/calc`: scipy 1.13.1, numpy 1.24.2, pymbar 4.0.3) `import pymbar` fails with `AttributeError: module 'scipy.linalg' has no attribute 'tril'` (pymbar's optional JAX backend references a scipy function removed in 1.13; pymbar only catches ImportError). Consequences today, silently:
- `gareus.mbar_subsample.equilibrated_subsample` returns `status="pymbar_missing"`, so the driver's union builder does **no** equilibration discard or thinning on aurum2;
- the campaign-end union MBAR analysis (`adaptive_production.py`, `from pymbar import MBAR` guarded by `except Exception`) writes coverage only;
- `_compute_mbar_weights_for_tica` guards with `except ImportError` only, so a tICA refit would crash there;
- this plan's per-epoch diagnostics (Task 5) would always return None → top-ups would never run on aurum2.

Fixing the environment (e.g. removing or pinning `jax` in `calc`) changes the live chignolin_9 campaign's behaviour at its next resubmit (subsampling starts working), so it is a **user decision, not part of this task**. This task makes the failure loud and non-crashing.

**Files:**
- Create: `gareus/pymbar_check.py`
- Modify: `gareus/adaptive_production.py` (`_compute_mbar_weights_for_tica`: `except ImportError:` → `except Exception:`), `gareus/mbar_subsample.py` (log the reason once when returning `pymbar_missing`), `gareus/cli.py` (call the check once at startup when adaptive production runs)
- Test: `tests/test_pymbar_check.py`

**Interfaces:**
- Produces: `pymbar_status() -> tuple[bool, str]` (cached; `(True, "pymbar 4.0.3")` or `(False, "<exception type>: <message>")`); `warn_if_pymbar_unusable(context: str) -> bool` printing one line `WARNING [pymbar]: unusable (<reason>); <context> will run degraded` and returning the availability.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pymbar_check.py
import builtins

import gareus.pymbar_check as pc


def test_reports_available_when_import_works():
    pc.pymbar_status.cache_clear()
    ok, msg = pc.pymbar_status()
    assert ok and msg.startswith("pymbar")


def test_reports_any_import_time_exception_not_just_importerror(monkeypatch, capsys):
    real_import = builtins.__import__

    def broken(name, *a, **k):
        if name == "pymbar":
            raise AttributeError("module 'scipy.linalg' has no attribute 'tril'")
        return real_import(name, *a, **k)

    pc.pymbar_status.cache_clear()
    monkeypatch.setattr(builtins, "__import__", broken)
    ok, msg = pc.pymbar_status()
    assert not ok and "AttributeError" in msg and "tril" in msg
    assert pc.warn_if_pymbar_unusable("top-up diagnostics") is False
    assert "WARNING [pymbar]" in capsys.readouterr().out
    pc.pymbar_status.cache_clear()
```

- [ ] **Step 2: Run to verify it fails** — expected `ModuleNotFoundError: gareus.pymbar_check`.

- [ ] **Step 3: Implement**

```python
# gareus/pymbar_check.py
"""Whether pymbar can actually be imported here, and why not (any exception, not only ImportError)."""
from __future__ import annotations

from functools import lru_cache
from typing import Tuple


@lru_cache(maxsize=1)
def pymbar_status() -> Tuple[bool, str]:
    try:
        import pymbar  # noqa: PLC0415
        from pymbar import MBAR  # noqa: F401,PLC0415
    except Exception as exc:  # e.g. AttributeError from an incompatible jax/scipy pair
        return False, f"{type(exc).__name__}: {exc}"
    return True, f"pymbar {getattr(pymbar, '__version__', '?')}"


def warn_if_pymbar_unusable(context: str) -> bool:
    ok, msg = pymbar_status()
    if not ok:
        print(f"WARNING [pymbar]: unusable ({msg}); {context} will run degraded", flush=True)
    return ok
```

In `cli.py`, call `warn_if_pymbar_unusable("equilibration subsampling, union MBAR analysis and top-up diagnostics")` once, in the function that holds the three `prod_summary = run_adaptive_production_auto_loop(` call sites (currently ~lines 1837, 1867, 1919), before the first of them runs. In `_compute_mbar_weights_for_tica` change the import guard to `except Exception:`. In `mbar_subsample.equilibrated_subsample`, before `return _fallback("pymbar_missing")`, call `warn_if_pymbar_unusable("equilibration subsampling")` once per process (guard with a module-level flag).

- [ ] **Step 4: Run tests** — `tests/test_pymbar_check.py tests/test_mbar_subsample*.py tests/test_adaptive_segmented_diagnostics.py`; expected PASS.
- [ ] **Step 5: Commit** — `git commit -am "fix: detect an unusable pymbar loudly instead of degrading silently"`

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
- Modify: `gareus/adaptive_production.py` (`AdaptiveDecisionPolicy` dataclass; `policy_from_args`; the `_arg_bool(args, "adaptive_production_topups", True)` default inside `run_scheduled_adaptive_epoch`)
- Modify: `tests/test_no_topups.py` (default flips to off)
- Test: `tests/test_topup_flags.py`

**Interfaces:**
- Produces: `AdaptiveDecisionPolicy.topups_enabled: bool = False`, `.topup_target_sigma: float = 0.10`, `.topup_weak_overlap: float = 0.15`, `.topup_max_fraction: float = 0.3`, `.topup_min_effect: float = 0.05`, `.topup_max_edge_attempts: int = 2`, `.topup_throughput_table: tuple = ((16.0, 3154.0), (59.0, 2300.0))`.
- Produces: `args.adaptive_production_topups` and `args.adaptive_production_topup_{target_sigma,weak_overlap,max_fraction,min_effect,max_edge_attempts,throughput_table}`.

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
                       "--ap-topup-weak-overlap", "0.1", "--ap-topup-max-fraction", "0.25",
                       "--ap-topup-min-effect", "0.03", "--ap-topup-max-edge-attempts", "3"])
    pol = policy_from_args(args)
    assert pol.topups_enabled is True
    assert (pol.topup_target_sigma, pol.topup_weak_overlap, pol.topup_max_fraction) == (0.2, 0.1, 0.25)
    assert (pol.topup_min_effect, pol.topup_max_edge_attempts) == (0.03, 3)
    assert pol.topup_throughput_table == ((16.0, 3154.0), (59.0, 2300.0))


def test_policy_defaults_match_the_spec():
    pol = AdaptiveDecisionPolicy()
    assert pol.topups_enabled is False
    assert (pol.topup_target_sigma, pol.topup_weak_overlap, pol.topup_max_fraction) == (0.10, 0.15, 0.3)
    assert (pol.topup_min_effect, pol.topup_max_edge_attempts) == (0.05, 2)


def test_a_malformed_throughput_table_is_a_clear_error():
    import pytest
    with pytest.raises(SystemExit):
        parse_args(["--seq", "AA", "--ap-topup-throughput-table", "16-3154"])
```

- [ ] **Step 2: Run to verify it fails**

Run (via opencode, see Global Constraints): `python -m pytest -q -p no:cacheprovider tests/test_topup_flags.py`
Expected: FAIL (`ap_topups` default is True; unknown `--ap-topup-target-sigma`).

- [ ] **Step 3: Implement**

In `gareus/cli.py`, change the `--ap-topups` argument's `default=True` to `default=False` and start its help with "Top-ups (off by default): ...". Add after it:

```python
    p.add_argument("--ap-topup-target-sigma", type=float, default=0.10,
                   help="Per-state local free-energy uncertainty target (kcal/mol) for top-ups.")
    p.add_argument("--ap-topup-weak-overlap", type=float, default=0.15,
                   help="Symmetric energy-space overlap below which an edge is weak.")
    p.add_argument("--ap-topup-max-fraction", type=float, default=0.3,
                   help="Cap on the share of an epoch's wall-hour budget spent on top-ups.")
    p.add_argument("--ap-topup-min-effect", type=float, default=0.05,
                   help="Smallest split-halves free-energy discrepancy (kcal/mol) that can flag a state.")
    p.add_argument("--ap-topup-max-edge-attempts", type=int, default=2,
                   help="Top-ups a weak edge may receive before it is treated as structural (bridge).")
    p.add_argument("--ap-topup-throughput-table", default="16:3154,59:2300",
                   help="contexts_per_gpu:ns_per_day_node pairs, comma separated.")
```

In `_apply_v2_compat_shims`, after `args.adaptive_production_topups = args.ap_topups`:

```python
    args.adaptive_production_topup_target_sigma = args.ap_topup_target_sigma
    args.adaptive_production_topup_weak_overlap = args.ap_topup_weak_overlap
    args.adaptive_production_topup_max_fraction = args.ap_topup_max_fraction
    args.adaptive_production_topup_min_effect = args.ap_topup_min_effect
    args.adaptive_production_topup_max_edge_attempts = args.ap_topup_max_edge_attempts
    try:
        args.adaptive_production_topup_throughput_table = tuple(
            (float(a), float(b)) for a, b in (item.split(":") for item in str(args.ap_topup_throughput_table).split(",") if item))
    except ValueError as exc:
        raise SystemExit(f"--ap-topup-throughput-table must be 'ctx:ns_per_day,...' ({exc})")
```

In `AdaptiveDecisionPolicy` (after `min_exchange_acceptance`):

```python
    topups_enabled: bool = False
    topup_target_sigma: float = 0.10
    topup_weak_overlap: float = 0.15
    topup_max_fraction: float = 0.3
    topup_min_effect: float = 0.05
    topup_max_edge_attempts: int = 2
    topup_throughput_table: tuple = ((16.0, 3154.0), (59.0, 2300.0))
```

In `policy_from_args(...)` add:

```python
        topups_enabled=_arg_bool(args, "adaptive_production_topups", False),
        topup_target_sigma=_arg_float(args, "adaptive_production_topup_target_sigma", 0.10),
        topup_weak_overlap=_arg_float(args, "adaptive_production_topup_weak_overlap", 0.15),
        topup_max_fraction=_arg_float(args, "adaptive_production_topup_max_fraction", 0.3),
        topup_min_effect=_arg_float(args, "adaptive_production_topup_min_effect", 0.05),
        topup_max_edge_attempts=_arg_int(args, "adaptive_production_topup_max_edge_attempts", 2),
        topup_throughput_table=tuple(getattr(args, "adaptive_production_topup_throughput_table",
                                             ((16.0, 3154.0), (59.0, 2300.0)))),
```

In `tests/test_no_topups.py::test_the_flag_parses_and_defaults_on` rename to `test_the_flag_parses_and_defaults_off` and swap the asserts (`--seq AA` → False, `--seq AA --ap-topups` → True). Its driver test sets `args.adaptive_production_topups = False` explicitly already, so it keeps passing. In `run_scheduled_adaptive_epoch` change `_arg_bool(args, "adaptive_production_topups", True)` to `False`.

- [ ] **Step 4: Run tests**

Run: `python -m pytest -q -p no:cacheprovider tests/test_topup_flags.py tests/test_no_topups.py tests/test_dashboard_cli_flags.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add gareus/cli.py gareus/adaptive_production.py tests/test_topup_flags.py tests/test_no_topups.py
git commit -m "feat: top-ups off by default; add top-up target/overlap/cap/effect/attempt/throughput knobs"
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

### Task 4: Per-state equilibration cut and inefficiency in the union meta

The union builder already discards equilibration and thins each state's samples to decorrelated ones (`equilibrated_subsample_indices`), recording only `{raw, kept}`. The allocator needs the statistical inefficiency and the post-equilibration count, which the full `equilibrated_subsample` result already carries (`SubsampleResult(indices, t0, g, status, ...)`). Record them; do not run a second autocorrelation pass on the already-thinned samples (revision-1 defect V3).

**Files:**
- Modify: `gareus/adaptive_production.py` — `build_union_state_mbar_inputs`, the per-state subsampling loop (`from .mbar_subsample import equilibrated_subsample_indices as _esi` … `_subsample_counts[str(_sid)] = {"raw": ..., "kept": ...}`)
- Test: `tests/test_union_subsample_counts.py`

**Interfaces:**
- Produces: `meta["subsample_counts_per_state"][str(state_id)] == {"raw": int, "t0": int, "kept": int, "g": float, "status": str}` where `g` is the inefficiency the subsampler used and, when it is NaN or < 1, `max(1.0, (raw - t0) / max(1, kept))`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_union_subsample_counts.py
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_adaptive_segmented_diagnostics import N_WINDOWS, _write_parquet_epoch_run, _write_window_csv

from gareus.adaptive_production import build_union_state_mbar_inputs, registry_from_window_csv


def test_union_meta_records_equilibration_cut_and_inefficiency(tmp_path):
    registry = registry_from_window_csv(_write_window_csv(tmp_path / "w.csv"), epoch=0, source="t")
    adaptive = tmp_path / "adaptive"
    _write_parquet_epoch_run(adaptive / "final", n_windows=N_WINDOWS, rows_per_window=200)
    meta = build_union_state_mbar_inputs(adaptive, registry)
    counts = meta["subsample_counts_per_state"]
    assert set(counts) == {str(i) for i in range(N_WINDOWS)}
    for rec in counts.values():
        assert set(rec) >= {"raw", "t0", "kept", "g", "status"}
        assert rec["raw"] == 200 and 0 <= rec["t0"] < rec["raw"] and 0 < rec["kept"] <= rec["raw"] - rec["t0"]
        assert math.isfinite(rec["g"]) and rec["g"] >= 1.0
```

- [ ] **Step 2: Run to verify it fails** — `python -m pytest -q -p no:cacheprovider tests/test_union_subsample_counts.py`; expected FAIL (`KeyError: 't0'` / missing keys).

- [ ] **Step 3: Implement** — in the subsampling loop replace the `_esi` call and the counts line:

```python
    from .mbar_subsample import equilibrated_subsample as _es  # noqa: PLC0415
    ...
    for _sid, _idx_list in _state_to_indices.items():
        _trace = np.asarray([float(sample_rows[i]["cv_A"]) for i in _idx_list], dtype=np.float64)
        _res = _es(_trace)
        _keep = np.asarray(_res.indices, dtype=np.int64)
        _kept_global.extend(_idx_list[k] for k in _keep.tolist())
        _raw, _t0, _kept = len(_idx_list), int(_res.t0), int(len(_keep))
        _g = float(_res.g)
        if not math.isfinite(_g) or _g < 1.0:
            _g = max(1.0, (_raw - _t0) / max(1, _kept))
        _subsample_counts[str(_sid)] = {"raw": _raw, "t0": _t0, "kept": _kept, "g": _g,
                                        "status": str(_res.status)}
```

`equilibrated_subsample_indices(series)` is `equilibrated_subsample(series).indices`, so the kept rows are unchanged; only the recorded provenance grows. Remove the now-unused `_esi` import.

- [ ] **Step 4: Run tests** — `tests/test_union_subsample_counts.py tests/test_adaptive_segmented_diagnostics.py tests/test_union_state_mbar_native_params.py tests/test_lambda_ladder_mbar.py`; expected PASS.
- [ ] **Step 5: Commit** — `git commit -am "feat: union meta records per-state equilibration cut and inefficiency"`

---

### Task 5: Per-epoch union diagnostics

Solve the union MBAR the driver builds and return per-state local σ, a multiplicity-corrected split-halves flag, the builder's decorrelated counts and inefficiency, a symmetric overlap for every geometry edge (spatial and rung), and f_k for warm-starting the next epoch.

**Files:**
- Create: `gareus/adaptive/union_diagnostics.py`
- Test: `tests/test_topup_union_diagnostics.py`

**Interfaces:**
- Consumes: the union NPZ written by `build_union_state_mbar_inputs` (keys `umbrella_reduced_bias_nk`, `state_ids`, `sampled_state_ids`; samples already decorrelated per state, in chronological order within each state and source); `meta["subsample_counts_per_state"]` (Task 4); `gareus.mbar_analysis.ladder.mbar_state_overlap(u_nk, f_k, n_k)`.
- Produces:

```python
@dataclass(frozen=True)
class UnionDiagnostics:
    state_ids: tuple            # union order
    n_k: dict                   # state_id -> decorrelated samples in the solve (= kept)
    sigma_kcal: dict            # state_id -> LOCAL uncertainty: min_j dDelta_f[j,k] over edge-neighbours (kcal/mol); nan if unsampled
    unconverged: frozenset      # state_ids flagged by the multiplicity-corrected split-halves check
    inefficiency: dict          # state_id -> g >= 1: raw report-interval samples per decorrelated sample
    edge_overlap: dict          # (min_id, max_id) -> symmetric sqrt(O_ij O_ji)
    f_kT: dict                  # state_id -> reduced free energy (warm start for the next epoch)

def union_diagnostics_from_npz(npz_path, edges, *, kt_kcal: float, subsample_counts: dict | None = None,
                               min_effect_kcal: float = 0.05, alpha: float = 0.05,
                               f_init: dict | None = None, split_halves: bool = True) -> UnionDiagnostics | None
```

Returns `None` (never raises) when the file is missing, has no finite rows, or the solve fails.

Split-halves rule (revision-1 defect V2): each unordered edge (j, k) with both endpoints holding ≥ `MIN_HALF` (20) samples per half is tested **once**; the family size m is the number of such tests; the pair disagrees if |Δ_A − Δ_B| > max(z*·√(σ_A² + σ_B²), min_effect/kT) with z* = Φ⁻¹(1 − α/(2m)) (Bonferroni, α = 0.05) and Δ = f_k − f_j within each half; both endpoints of a disagreeing pair are flagged. Halves are contiguous in time (first vs second half of each state's rows); never shuffled (a shuffle destroys the drift signal).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_topup_union_diagnostics.py
import math

import numpy as np

from gareus.adaptive.union_diagnostics import union_diagnostics_from_npz

KT = 0.596  # kcal/mol at 300 K


def _write(tmp_path, centers, k, n_per, seed=0, drift_state=None, name="u.npz"):
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
    p = tmp_path / name
    np.savez(p, umbrella_reduced_bias_nk=u, state_ids=np.arange(len(centers)),
             sampled_state_ids=np.asarray(sid))
    return p


def _chain_edges(n):
    return [(i, i + 1) for i in range(n - 1)]


def test_healthy_chain_has_small_local_sigma_and_all_edges_measured(tmp_path):
    p = _write(tmp_path, [0.0, 1.0, 2.0, 3.0], k=4.0, n_per=2000)
    d = union_diagnostics_from_npz(p, _chain_edges(4), kt_kcal=KT)
    assert d is not None
    assert all(math.isfinite(d.sigma_kcal[s]) and d.sigma_kcal[s] > 0 for s in range(4))
    assert d.sigma_kcal[3] < 2.0 * d.sigma_kcal[1]          # local sigma: no penalty for distance from state 0
    assert set(d.edge_overlap) == set(_chain_edges(4)) and min(d.edge_overlap.values()) > 0.15
    assert not d.unconverged


def test_fewer_samples_means_larger_sigma(tmp_path):
    big = union_diagnostics_from_npz(_write(tmp_path, [0.0, 1.0], 4.0, 4000, name="b.npz"), [(0, 1)], kt_kcal=KT)
    small = union_diagnostics_from_npz(_write(tmp_path, [0.0, 1.0], 4.0, 250, name="s.npz"), [(0, 1)], kt_kcal=KT)
    assert small.sigma_kcal[1] > big.sigma_kcal[1]


def test_a_state_whose_halves_disagree_is_flagged(tmp_path):
    p = _write(tmp_path, [0.0, 1.0, 2.0], k=4.0, n_per=3000, drift_state=1)
    d = union_diagnostics_from_npz(p, _chain_edges(3), kt_kcal=KT)
    assert 1 in d.unconverged


def test_false_alarms_are_controlled_at_production_scale(tmp_path):
    # 236 iid states: a fixed 2-sigma rule flags ~10 states; Bonferroni keeps the family error ~alpha.
    p = _write(tmp_path, [0.5 * i for i in range(236)], k=4.0, n_per=160, seed=7)
    d = union_diagnostics_from_npz(p, _chain_edges(236), kt_kcal=KT)
    assert d is not None and len(d.unconverged) <= 2


def test_inefficiency_comes_from_the_builder_meta(tmp_path):
    p = _write(tmp_path, [0.0, 1.0], k=4.0, n_per=500)
    counts = {"0": {"raw": 5000, "t0": 500, "kept": 500, "g": 9.0, "status": "subsampled"},
              "1": {"raw": 5000, "t0": 0, "kept": 500, "g": float("nan"), "status": "detect_failed"}}
    d = union_diagnostics_from_npz(p, [(0, 1)], kt_kcal=KT, subsample_counts=counts)
    assert d.inefficiency[0] == 9.0
    assert d.inefficiency[1] == 10.0                        # fallback: (raw - t0) / kept
    assert d.n_k == {0: 500, 1: 500}


def test_a_warm_start_reproduces_the_cold_solution(tmp_path):
    p = _write(tmp_path, [0.0, 1.0, 2.0], k=4.0, n_per=1500)
    cold = union_diagnostics_from_npz(p, _chain_edges(3), kt_kcal=KT)
    warm = union_diagnostics_from_npz(p, _chain_edges(3), kt_kcal=KT, f_init=cold.f_kT)
    assert all(abs(cold.f_kT[s] - warm.f_kT[s]) < 1e-6 for s in range(3))


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
    d = union_diagnostics_from_npz(p, _chain_edges(3), kt_kcal=KT)
    assert d is not None and math.isnan(d.sigma_kcal[2]) and d.n_k[2] == 0
```

- [ ] **Step 2: Run to verify it fails** — expected `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

```python
# gareus/adaptive/union_diagnostics.py
"""Per-epoch union MBAR: local sigma, multiplicity-corrected split halves, edge overlap.

Reads the union NPZ written by adaptive_production.build_union_state_mbar_inputs.
Its samples are ALREADY decorrelated per state (equilibration discard + thinning),
so no second autocorrelation pass runs here: the inefficiency comes from the
builder's meta (subsample_counts_per_state). Never raises: any failure returns
None and the epoch runs no top-up.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
from scipy.stats import norm

MIN_HALF = 20


@dataclass(frozen=True)
class UnionDiagnostics:
    state_ids: tuple
    n_k: dict
    sigma_kcal: dict
    unconverged: frozenset
    inefficiency: dict
    edge_overlap: dict
    f_kT: dict


def _solve(u_nk: np.ndarray, window: np.ndarray, n_states: int, f_init: Optional[np.ndarray] = None):
    """f (reduced, vs first active state), pairwise uncertainty matrix (K x K, nan where unsampled), n_k.

    Initialised from zeros or a warm start, not BAR: pymbar's BAR initialisation chains
    consecutive states, and the union's state order (centres x rungs) is not overlap order.
    """
    from pymbar import MBAR  # noqa: PLC0415
    n_k = np.bincount(window, minlength=n_states)
    active = np.flatnonzero(n_k > 0)
    order = np.argsort(window, kind="stable")                 # pymbar wants rows grouped by state
    u_kn = u_nk[order][:, active].T
    kwargs = {"initialize": "zeros", "solver_protocol": "robust"}
    if f_init is not None and np.all(np.isfinite(f_init[active])):
        kwargs["initial_f_k"] = f_init[active] - f_init[active][0]
    mbar = MBAR(u_kn, n_k[active], **kwargs)
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


def _inefficiency(ids, n_k, subsample_counts) -> Dict[int, float]:
    out = {}
    for k, sid in enumerate(ids):
        rec = (subsample_counts or {}).get(str(sid)) or {}
        g = rec.get("g")
        if not (isinstance(g, (int, float)) and math.isfinite(float(g)) and float(g) >= 1.0):
            raw, t0, kept = rec.get("raw"), rec.get("t0", 0), rec.get("kept")
            g = max(1.0, (raw - t0) / max(1, kept)) if raw is not None and kept else 1.0
        out[sid] = float(g)
    return out


def _split_halves(u, window, K, nbr_pairs, min_effect_kT, alpha, f_init):
    half = np.zeros(len(window), dtype=bool)
    per_state = [np.flatnonzero(window == k) for k in range(K)]
    for rows in per_state:
        half[rows[: len(rows) // 2]] = True                    # contiguous halves, never shuffled
    eligible = [(j, k) for j, k in nbr_pairs
                if min(len(per_state[j]), len(per_state[k])) >= 2 * MIN_HALF]
    if not eligible:
        return set()
    fa, da, _ = _solve(u[half], window[half], K, f_init)
    fb, db, _ = _solve(u[~half], window[~half], K, f_init)
    z_star = float(norm.ppf(1.0 - alpha / (2.0 * len(eligible))))
    flagged = set()
    for j, k in eligible:
        delta_a, delta_b = fa[k] - fa[j], fb[k] - fb[j]
        comb = math.hypot(da[j, k], db[j, k])
        if not all(math.isfinite(v) for v in (delta_a, delta_b, comb)):
            continue
        if abs(delta_a - delta_b) > max(z_star * comb, min_effect_kT):
            flagged.update((j, k))
    return flagged


def union_diagnostics_from_npz(npz_path, edges: Iterable[Tuple[int, int]], *, kt_kcal: float,
                               subsample_counts: Optional[dict] = None, min_effect_kcal: float = 0.05,
                               alpha: float = 0.05, f_init: Optional[dict] = None,
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
        keep = (window >= 0) & np.isfinite(u).all(axis=1)       # NaN rows = missing energies, excluded by design
        if int((~keep).sum()):
            logging.info("top-up diagnostics: %d of %d rows excluded (non-finite)", int((~keep).sum()), len(keep))
        u, window = u[keep], window[keep]
        if u.shape[0] == 0:
            return None
        K = len(ids)
        f0 = None
        if f_init:
            f0 = np.asarray([float(f_init.get(s, np.nan)) for s in ids])
        f, dmat, n_k = _solve(u, window, K, f0)
        nbrs = {k: [] for k in range(K)}
        pairs = set()
        for a, b in edges:
            ia, ib = idx.get(int(a)), idx.get(int(b))
            if ia is not None and ib is not None and ia != ib:
                nbrs[ia].append(ib); nbrs[ib].append(ia)
                pairs.add((min(ia, ib), max(ia, ib)))
        sigma = np.array([_local_sigma(dmat, k, nbrs[k]) if n_k[k] > 0 else np.nan for k in range(K)])
        flagged = (_split_halves(u, window, K, sorted(pairs), min_effect_kcal / kt_kcal, alpha, f)
                   if split_halves else set())
        from ..mbar_analysis.ladder import mbar_state_overlap  # noqa: PLC0415
        O = mbar_state_overlap(u, np.where(np.isfinite(f), f, 0.0), n_k)
        edge_overlap = {}
        for ia, ib in sorted(pairs):
            if n_k[ia] == 0 or n_k[ib] == 0:
                continue
            x, y = float(O[ia, ib]), float(O[ib, ia])
            if math.isfinite(x) and math.isfinite(y) and x >= 0 and y >= 0:
                edge_overlap[(min(ids[ia], ids[ib]), max(ids[ia], ids[ib]))] = math.sqrt(x * y)
        return UnionDiagnostics(
            state_ids=tuple(ids),
            n_k={ids[k]: int(n_k[k]) for k in range(K)},
            sigma_kcal={ids[k]: (float(sigma[k]) * kt_kcal if math.isfinite(sigma[k]) else math.nan)
                        for k in range(K)},
            unconverged=frozenset(ids[k] for k in flagged),
            inefficiency=_inefficiency(ids, n_k, subsample_counts),
            edge_overlap=edge_overlap,
            f_kT={ids[k]: float(f[k]) for k in range(K) if math.isfinite(f[k])},
        )
    except Exception as exc:  # the epoch then runs no top-up
        logging.warning("top-up union diagnostics unavailable (%s)", exc)
        return None
```

- [ ] **Step 4: Run tests** — expected PASS. If `test_a_warm_start_reproduces_the_cold_solution` fails on the `initial_f_k` keyword, check pymbar 4.0.3's `MBAR.__init__` signature (`python -c "import inspect, pymbar; print(inspect.signature(pymbar.MBAR.__init__))"`) and use its name for the initial free energies; do not drop the warm start.
- [ ] **Step 5: Commit** — `git add gareus/adaptive/union_diagnostics.py tests/test_topup_union_diagnostics.py && git commit -m "feat: per-epoch union MBAR diagnostics for top-up allocation"`

---

### Task 6: Top-up allocator

**Files:**
- Create: `gareus/adaptive/topup_allocator.py`
- Test: `tests/test_topup_allocator.py`

**Interfaces:**
- Consumes: `UnionDiagnostics` (Task 5: `n_k` = decorrelated count, `inefficiency` = raw samples per decorrelated sample, `sigma_kcal`, `unconverged`, `edge_overlap`), `wall_hours` (Task 3).
- Produces:

```python
@dataclass(frozen=True)
class TopupPlan:
    state_ids: tuple = ()             # patch, sorted
    steps: int = 0                    # lockstep length, multiple of report_interval
    deficit_state_ids: tuple = ()
    partner_state_ids: tuple = ()
    structural_edges: tuple = ()      # (i, j) routed to the bridge/add proposal
    weak_edges_topped: tuple = ()     # (i, j) noise-weak edges this top-up tries to fix (attempt counting)
    predicted_sigma: dict = {}        # state_id -> sigma after the top-up
    sigma_before: dict = {}           # state_id -> sigma when planned
    cost_hours: float = 0.0
    reason: str = "healthy"           # planned | healthy | no_diagnostics | cap_too_small

def plan_topup(diag, *, state_ids_in_order, neighbours, rung_partners, policy, report_interval: int,
               timestep_fs: float, n_gpus: int, budget_hours: float,
               correction: dict | None = None, edge_attempts: dict | None = None) -> TopupPlan
```

`neighbours`: state_id → same-rung spatial neighbours; `rung_partners`: state_id → other rungs of the same centre; `correction`: state_id → realised/predicted gain factor (clamped to [0.1, 2.0] here, whatever is passed); `edge_attempts`: (min_id, max_id) → top-ups already spent on that weak edge.

Rules (spec 4.2–4.3, revision 2):
- Decorrelated count n_k = `diag.n_k[s]`; one new decorrelated sample costs `report_interval × g_s` steps with g_s = `diag.inefficiency[s]`; current steps ≈ n_k·interval·g_s.
- Deficit = σ > target or s ∈ `unconverged`; NaN σ or n_k = 0 is **not** a deficit (unsampled = coverage, not top-up).
- Weak edge = measured overlap < `topup_weak_overlap`. Structural if both endpoints are adequate **or** `edge_attempts[edge] >= topup_max_edge_attempts`; otherwise noise → both sampled endpoints join the deficits and the edge goes into `weak_edges_topped`.
- Predicted σ after L steps: σ·√(n / (n + c·L / (interval·g))). Required L for s: smallest multiple of `interval` reaching the target (an unconverged state below target: target = σ/√2, i.e. double its data), capped at 4× its current steps.
- Partners (lockstep: the patch is fixed, only L varies): for each deficit s, if no deficit among its same-rung neighbours, add the same-rung neighbour with the largest σ (another rung's partner when it has none); if no deficit among its rung partners, add the rung partner with the largest σ. Candidates with n_k = 0 or NaN σ are never partners.
- Length candidates: every distinct required L, plus the largest budget-feasible L (`floor(budget / wall_hours(interval, |patch|)) × interval`). For each candidate within budget: benefit = drop in max σ over deficits + 1e-3 × drop in Σσ²; score = benefit / wall_hours. Candidates with a non-finite prediction or score are skipped. Best score wins; ties keep the shorter L.
- No deficits → `healthy`; no candidate within budget or best L < interval → `cap_too_small`; `diag is None` → `no_diagnostics`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_topup_allocator.py
import math

from gareus.adaptive.throughput import wall_hours
from gareus.adaptive.topup_allocator import plan_topup
from gareus.adaptive.union_diagnostics import UnionDiagnostics
from gareus.adaptive_production import AdaptiveDecisionPolicy

POL = AdaptiveDecisionPolicy(topups_enabled=True)          # target 0.10, weak 0.15, cap 0.3, 2 attempts
IV = 500                                                   # report interval


def _diag(sigma, edges=None, unconverged=(), n=None):
    ids = tuple(sorted(sigma))
    return UnionDiagnostics(state_ids=ids, n_k=n or {s: 100 for s in ids}, sigma_kcal=dict(sigma),
                            unconverged=frozenset(unconverged), inefficiency={s: 2.0 for s in ids},
                            edge_overlap=edges or {}, f_kT={})


def _chain(n):
    # n centres on one rung (0..n-1) and the same centres on a second rung (n..2n-1)
    nb = {s: [x for x in (s - 1, s + 1) if 0 <= x < n] for s in range(n)}
    nb.update({s + n: [x + n for x in (s - 1, s + 1) if 0 <= x < n] for s in range(n)})
    rp = {s: [s + n] for s in range(n)}; rp.update({s + n: [s] for s in range(n)})
    return nb, rp


def _plan(diag, n=4, budget=100.0, correction=None, attempts=None, nb=None, rp=None):
    nb0, rp0 = _chain(n)
    return plan_topup(diag, state_ids_in_order=list(diag.state_ids), neighbours=nb or nb0,
                      rung_partners=rp or rp0, policy=POL, report_interval=IV, timestep_fs=4.0,
                      n_gpus=4, budget_hours=budget, correction=correction, edge_attempts=attempts)


def _one_deficit(sigma1=0.15):
    s = {x: 0.05 for x in range(8)}; s[1] = sigma1
    return s


def test_no_diagnostics_means_no_topup():
    nb, rp = _chain(4)
    p = plan_topup(None, state_ids_in_order=list(range(8)), neighbours=nb, rung_partners=rp, policy=POL,
                   report_interval=IV, timestep_fs=4.0, n_gpus=4, budget_hours=100.0)
    assert p.reason == "no_diagnostics" and p.state_ids == ()


def test_a_healthy_campaign_gets_no_topup():
    p = _plan(_diag({s: 0.05 for s in range(8)}))
    assert p.reason == "healthy" and p.state_ids == () and p.steps == 0


def test_unmeasured_edges_never_make_a_state_deficient():
    assert _plan(_diag({s: 0.05 for s in range(8)}, edges={})).reason == "healthy"


def test_one_deficit_state_gets_minimal_partners_and_the_right_length():
    p = _plan(_diag(_one_deficit()))
    assert p.reason == "planned" and p.deficit_state_ids == (1,)
    assert len(p.partner_state_ids) == 2 and 5 in p.partner_state_ids   # one same-rung, one rung partner
    # n=100 decorrelated, g=2, sigma 0.15 -> 0.10 needs 125 more decorrelated = 125*500*2 steps
    assert p.steps == 125_000
    assert p.predicted_sigma[1] <= POL.topup_target_sigma + 1e-9


def test_a_weak_edge_between_healthy_states_is_structural_not_md():
    p = _plan(_diag({s: 0.05 for s in range(8)}, edges={(1, 2): 0.02}))
    assert p.structural_edges == ((1, 2),) and p.reason == "healthy"


def test_a_weak_edge_touching_a_deficit_tops_up_both_endpoints():
    p = _plan(_diag(_one_deficit(), edges={(1, 2): 0.02}))
    assert {1, 2} <= set(p.deficit_state_ids) and p.structural_edges == ()
    assert p.weak_edges_topped == ((1, 2),)


def test_an_edge_that_stayed_weak_after_max_attempts_becomes_structural():
    p = _plan(_diag(_one_deficit(), edges={(1, 2): 0.02}), attempts={(1, 2): 2})
    assert p.structural_edges == ((1, 2),) and 2 not in p.deficit_state_ids


def test_nan_sigma_or_zero_samples_is_not_a_topup_target():
    s = {x: 0.05 for x in range(8)}; s[3] = math.nan
    assert _plan(_diag(s)).reason == "healthy"


def test_a_never_sampled_state_is_never_a_partner():
    s = _one_deficit(); s[0] = math.nan; s[2] = math.nan          # both same-rung neighbours of 1 unsampled
    n = {x: 100 for x in range(8)}; n[0] = 0; n[2] = 0
    p = _plan(_diag(s, n=n))
    assert p.reason == "planned" and not ({0, 2} & set(p.partner_state_ids))


def test_a_budget_smaller_than_one_report_interval_plans_nothing():
    p = _plan(_diag(_one_deficit()), budget=1e-9)
    assert p.reason == "cap_too_small" and p.state_ids == () and p.steps == 0


def test_a_moderate_budget_funds_a_partial_topup():
    budget = wall_hours(60_000, 3, 4.0, 4, POL.topup_throughput_table)   # half of the 125k need, 3-state patch
    p = _plan(_diag(_one_deficit()), budget=budget)
    assert p.reason == "planned" and 0 < p.steps <= 60_000
    assert POL.topup_target_sigma < p.predicted_sigma[1] < 0.15


def test_a_deficit_at_the_layout_edge_still_gets_a_partner():
    s = {x: 0.05 for x in range(8)}; s[0] = 0.15
    nb = {0: [], 1: [0], 2: [], 3: [], 4: [], 5: [], 6: [], 7: []}
    rp = {0: [4], 4: [0], 1: [5], 5: [1], 2: [6], 6: [2], 3: [7], 7: [3]}
    p = _plan(_diag(s), nb=nb, rp=rp)
    assert p.reason == "planned" and 4 in p.partner_state_ids


def test_uniformly_deficient_states_degenerate_to_all_states():
    p = _plan(_diag({s: 0.15 for s in range(8)}))
    assert set(p.state_ids) == set(range(8)) and p.partner_state_ids == ()


def test_a_state_that_underdelivered_needs_more_steps():
    assert _plan(_diag(_one_deficit()), correction={1: 0.5}).steps > _plan(_diag(_one_deficit())).steps


def test_a_bad_correction_factor_is_clamped_not_propagated():
    p = _plan(_diag(_one_deficit()), correction={1: -3.0})
    assert p.reason == "planned" and math.isfinite(p.predicted_sigma[1]) and p.steps > 0


def test_the_plan_is_deterministic():
    assert _plan(_diag(_one_deficit())) == _plan(_diag(_one_deficit()))
```

- [ ] **Step 2: Run to verify it fails** — expected `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

```python
# gareus/adaptive/topup_allocator.py
"""Deficit-driven, wall-hour-costed top-up plan (spec 4.2-4.3, revision 2). Pure function; no I/O."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .throughput import wall_hours

MAX_STEP_MULTIPLE = 4
CORRECTION_BOUNDS = (0.1, 2.0)


@dataclass(frozen=True)
class TopupPlan:
    state_ids: tuple = ()
    steps: int = 0
    deficit_state_ids: tuple = ()
    partner_state_ids: tuple = ()
    structural_edges: tuple = ()
    weak_edges_topped: tuple = ()
    predicted_sigma: dict = field(default_factory=dict)
    sigma_before: dict = field(default_factory=dict)
    cost_hours: float = 0.0
    reason: str = "healthy"


def _ok(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def _clamp(c) -> float:
    lo, hi = CORRECTION_BOUNDS
    return min(hi, max(lo, float(c))) if _ok(c) else 1.0


def _round_up(steps: float, interval: int) -> int:
    return int(math.ceil(max(0.0, steps) / interval) * interval)


def plan_topup(diag, *, state_ids_in_order: Sequence[int], neighbours: Dict[int, List[int]],
               rung_partners: Dict[int, List[int]], policy, report_interval: int, timestep_fs: float,
               n_gpus: int, budget_hours: float, correction: Optional[Dict[int, float]] = None,
               edge_attempts: Optional[Dict[Tuple[int, int], int]] = None) -> TopupPlan:
    if diag is None:
        return TopupPlan(reason="no_diagnostics")
    interval = max(1, int(report_interval))
    target = float(policy.topup_target_sigma)
    corr = {s: _clamp((correction or {}).get(s, 1.0)) for s in state_ids_in_order}
    attempts = edge_attempts or {}
    sigma = {s: float(diag.sigma_kcal.get(s, math.nan)) for s in state_ids_in_order}
    n_eff = {s: int(diag.n_k.get(s, 0)) for s in state_ids_in_order}
    g = {s: max(1.0, float(diag.inefficiency.get(s, 1.0))) for s in state_ids_in_order}
    sampled = {s for s in state_ids_in_order if n_eff[s] > 0 and _ok(sigma[s])}

    deficits = {s for s in sampled if sigma[s] > target or s in diag.unconverged}
    structural, noise_edges = [], []
    for (a, b), ov in sorted(diag.edge_overlap.items()):
        if not _ok(ov) or ov >= float(policy.topup_weak_overlap):
            continue
        tried_out = attempts.get((a, b), 0) >= int(policy.topup_max_edge_attempts)
        if (a in deficits or b in deficits) and not tried_out:
            deficits.update(x for x in (a, b) if x in sampled)
            noise_edges.append((a, b))
        else:
            structural.append((a, b))
    if not deficits:
        return TopupPlan(structural_edges=tuple(structural), reason="healthy")

    def predicted(s: int, L: int) -> float:
        n = max(1e-9, float(n_eff[s]))
        return sigma[s] * math.sqrt(n / (n + corr[s] * L / (interval * g[s])))

    def required(s: int) -> int:
        eff_target = target if sigma[s] > target else sigma[s] / math.sqrt(2.0)
        extra_eff = n_eff[s] * ((sigma[s] / eff_target) ** 2 - 1.0)
        need = _round_up(extra_eff * interval * g[s] / corr[s], interval)
        cap = _round_up(MAX_STEP_MULTIPLE * n_eff[s] * interval * g[s], interval)
        return max(interval, min(need, cap))

    def pick(pool: List[int]) -> int:
        return max(pool, key=lambda x: (sigma[x], -x))

    chosen: set = set()
    for s in sorted(deficits):
        same = [x for x in neighbours.get(s, []) if x in sampled]
        rung = [x for x in rung_partners.get(s, []) if x in sampled]
        if not any(x in deficits or x in chosen for x in same):
            pool = same or [x for x in rung if x not in deficits]
            if pool:
                chosen.add(pick(pool))
        if rung and not any(x in deficits or x in chosen for x in rung):
            chosen.add(pick(rung))
    partners = chosen - deficits
    patch = deficits | partners

    table = policy.topup_throughput_table
    per_step_hours = wall_hours(interval, len(patch), timestep_fs, n_gpus, table) / interval
    candidates = {required(s) for s in deficits}
    budget_max = int(budget_hours / per_step_hours // interval) * interval if per_step_hours > 0 else 0
    if budget_max >= interval:
        candidates.add(budget_max)
    worst0 = max(sigma[s] for s in deficits)
    ssq0 = sum(sigma[s] ** 2 for s in deficits)
    best = None
    for L in sorted(candidates):
        if L < interval:
            continue
        cost = per_step_hours * L
        if cost > budget_hours:
            continue
        pred = {s: predicted(s, L) for s in patch}
        if not all(_ok(v) for v in pred.values()):
            continue
        benefit = (worst0 - max(pred[s] for s in deficits)) + 1e-3 * (ssq0 - sum(pred[s] ** 2 for s in deficits))
        score = benefit / max(cost, 1e-12)
        if not _ok(score):
            continue
        if best is None or score > best[0] + 1e-15:
            best = (score, L, pred, cost)
    if best is None:
        return TopupPlan(structural_edges=tuple(structural), reason="cap_too_small")
    _, L, pred, cost = best
    return TopupPlan(state_ids=tuple(sorted(patch)), steps=int(L), deficit_state_ids=tuple(sorted(deficits)),
                     partner_state_ids=tuple(sorted(partners)), structural_edges=tuple(structural),
                     weak_edges_topped=tuple(noise_edges), predicted_sigma=pred,
                     sigma_before={s: sigma[s] for s in patch}, cost_hours=float(cost), reason="planned")
```

Note: lockstep means the patch is fixed and only L varies; states needing more than the chosen L carry to the next epoch (the diagnostics then still show them deficient).

- [ ] **Step 4: Run tests** — expected PASS. If `test_uniformly_deficient_states_degenerate_to_all_states` finds partners, the partner rule is adding non-deficit states when every state is a deficit; fix the rule, not the test.
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
- Modify: `gareus/adaptive_production.py` — `build_adaptive_epoch_schedule` (remove score distribution), `run_scheduled_adaptive_epoch` (baseline sizing, allocator call, single top-up, persistence, calibration, wall-time log), new module-level `_topup_plan_for_phase` and `_seedable_patch`
- Create: `gareus/adaptive/topup_state.py` (atomic persistence of plan and campaign top-up state)
- Test: `tests/test_topup_epoch_wiring.py`, `tests/test_topup_state.py`

**Interfaces:**
- Consumes: `build_union_state_mbar_inputs(adaptive_dir, registry)` (existing; meta has `arrays_npz` and, after Task 4, `subsample_counts_per_state`), `union_diagnostics_from_npz` (Task 5), `plan_topup`/`TopupPlan` (Task 6), `wall_hours` (Task 3), `build_geometry_edges(registry, policy)` (existing; yields `(a, b, edge_type, normalized_distance)`), `spatial_neighbour_pairs`/`same_rung_neighbours`/`other_rung_same_centre` (Task 1), `load_seed_index(parent_dirs) -> dict[int, str]` and `SeedMismatchError` (Task 9).
- Produces (`gareus/adaptive/topup_state.py`):
  - `save_plan(epoch_dir, plan) -> Path`, `load_plan(epoch_dir) -> TopupPlan | None` (`<epoch_dir>/topup_plan.json`);
  - `load_state(adaptive_dir) -> dict` with keys `correction` (int → float), `edge_attempts` ((int, int) → int), `f_kT` (int → float), `wall_time` (list of dicts); `save_state(adaptive_dir, state)` (`<adaptive_dir>/topup_state.json`);
  - `update_after_topup(state, plan, realised_sigma: dict) -> dict` (returns the new state: calibration + edge attempts);
  - all writes atomic (`<name>.tmp` + `os.replace`); JSON keys stored as strings and restored to ints/tuples; non-finite floats stored as `null` and restored as `nan`.

Behaviour:
1. `build_adaptive_epoch_schedule` gives every active state `requested_steps = baseline_steps = default_steps`, `extra_steps = 0`, `score = 0.0`, `allocation_reason = "baseline"`; the score/weak-count distribution and `_weak_edge_touch_counts` are deleted (grep first; keep any policy field another module still reads). Signature and schedule file format unchanged, so resume keeps working.
2. In `run_scheduled_adaptive_epoch`, keep `full_steps = baseline_steps` (the un-shortened default) **before** any shortening. Top-ups off → the PR #98 path (baseline at the quantized mean requested steps). Top-ups on → the baseline runs `max(1000, quantize(full_steps × (1 − topup_max_fraction)))`.
3. After the baseline (and its interruption check), `plan = _topup_plan_for_phase(args, epoch_dir, registry, policy, full_steps=full_steps)`:
   - a saved plan is reused only if every `plan.state_ids` is still an active state id; otherwise print `top-up plan discarded: layout changed (<missing ids>)` and recompute;
   - recompute: union inputs → `union_diagnostics_from_npz(meta["arrays_npz"], edges, kt_kcal=0.0019872041 × T, subsample_counts=meta.get("subsample_counts_per_state"), min_effect_kcal=policy.topup_min_effect, f_init=state["f_kT"])` → `plan_topup(..., budget_hours = policy.topup_max_fraction × wall_hours(full_steps, n_active, timestep_fs, n_gpus, table), correction=state["correction"], edge_attempts=state["edge_attempts"])`; store the new `f_kT` into the state; save plan and state;
   - write the union overlaps into the epoch diagnostics: for each edge dict in `diagnostics["edges"]`, `edge["mbar_overlap"] = diag.edge_overlap[(min, max)]` when measured (the existing proposer turns a measured low `mbar_overlap` into `add_rung` and a low spatial overlap into a bridge, so structural edges reach it with no new key).
4. `_seedable_patch(plan, parent_dirs)`: drop from the plan every state without an exported final State in the parent segments (`load_seed_index`), printing `top-up: <n> window(s) without a final State left out (state_ids ...)`; if no deficit state remains, the top-up is skipped (`reason="no_seed_states"`).
5. If the (seedable) plan is `planned`: time it, `run_segment(f"topup_001_{plan.steps}", list(plan.state_ids), int(plan.steps))`, with the same interrupted-payload return as the baseline. A `SeedMismatchError` from inside the segment is caught here: print it, record `reason="seed_mismatch"` in the saved plan, and continue the epoch without the top-up (the campaign is never stopped by it).
6. After a completed top-up: recompute the union diagnostics (no saved-plan short-circuit), `state = update_after_topup(state, plan, diag_after.sigma_kcal)`, append `{"segment", "n_states", "steps", "predicted_h": plan.cost_hours, "realised_h"}` to `state["wall_time"]`, save state.
7. `n_gpus` = number of comma-separated entries in `args.device_index`; `report_interval` = `args.report_interval`; `timestep_fs` = `args.timestep_fs`; `T` = `args.temperature_k`.

Calibration rule (`update_after_topup`): for each deficit s with finite `sigma_before > predicted`, `ratio = (before² − realised²) / (before² − predicted²)`; `c = clamp(0.7·c_old + 0.3·ratio, 0.1, 2.0)`; if `ratio < 0.5`, `c = max(0.1, 0.5·c)`. Edge attempts: every edge in `plan.weak_edges_topped` gets `+1`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_topup_state.py
import json
import math

from gareus.adaptive.topup_allocator import TopupPlan
from gareus.adaptive.topup_state import load_plan, load_state, save_plan, save_state, update_after_topup


def _plan():
    return TopupPlan(state_ids=(3, 4), steps=5000, deficit_state_ids=(3,), partner_state_ids=(4,),
                     weak_edges_topped=((3, 7),), predicted_sigma={3: 0.09, 4: math.nan},
                     sigma_before={3: 0.20, 4: 0.05}, cost_hours=1.5, reason="planned")


def test_plan_round_trips_including_int_keys_and_nan(tmp_path):
    save_plan(tmp_path, _plan())
    got = load_plan(tmp_path)
    assert got.state_ids == (3, 4) and got.weak_edges_topped == ((3, 7),)
    assert got.predicted_sigma[3] == 0.09 and math.isnan(got.predicted_sigma[4])
    assert not list(tmp_path.glob("*.tmp"))                       # atomic write leaves no temp file
    json.loads((tmp_path / "topup_plan.json").read_text())       # strict JSON (no bare NaN)


def test_state_defaults_and_round_trip(tmp_path):
    st = load_state(tmp_path)
    assert st == {"correction": {}, "edge_attempts": {}, "f_kT": {}, "wall_time": []}
    st["edge_attempts"][(1, 2)] = 1; st["correction"][3] = 0.7
    save_state(tmp_path, st)
    assert load_state(tmp_path)["edge_attempts"] == {(1, 2): 1}


def test_underdelivery_halves_and_smooths_the_correction_and_counts_the_edge():
    st = {"correction": {}, "edge_attempts": {}, "f_kT": {}, "wall_time": []}
    new = update_after_topup(st, _plan(), realised_sigma={3: 0.18})
    # ratio = (0.04-0.0324)/(0.04-0.0081) = 0.238; c = 0.7*1 + 0.3*0.238 = 0.771; halved -> 0.386
    assert abs(new["correction"][3] - 0.386) < 0.01
    assert new["edge_attempts"][(3, 7)] == 1


def test_the_correction_never_leaves_its_bounds():
    st = {"correction": {3: 0.1}, "edge_attempts": {}, "f_kT": {}, "wall_time": []}
    new = update_after_topup(st, _plan(), realised_sigma={3: 0.20})          # no improvement at all
    assert new["correction"][3] >= 0.1
```

```python
# tests/test_topup_epoch_wiring.py
from pathlib import Path

import pytest

import gareus.adaptive_production as ap
import gareus.production as prod
from gareus.adaptive.topup_allocator import TopupPlan
from gareus.lifecycle import _graceful_shutdown
from gareus.topup_seeding import SeedMismatchError

from test_scheduled_final_interruption import _scheduled_final_campaign


@pytest.fixture(autouse=True)
def _clear():
    _graceful_shutdown.clear(); yield; _graceful_shutdown.clear()


def _drive(monkeypatch, args, out, plan, *, worker=None, seedable=None):
    calls = []

    def _fake(a, d, *r, **k):
        calls.append((Path(d).name, int(a.gamd_production_steps)))
        if worker:
            worker(Path(d).name)

    monkeypatch.setattr(prod, "run_gareus", _fake)
    monkeypatch.setattr(ap, "_topup_plan_for_phase", lambda *a, **k: plan)
    monkeypatch.setattr(ap, "_seedable_patch", seedable or (lambda plan, parents: plan))
    monkeypatch.setattr(ap, "_after_topup_update", lambda *a, **k: None)
    ap.run_adaptive_production_auto_loop(args, out, None, None, None, None, None, None)
    return calls


def _topups_on(tmp_path):
    args, out = _scheduled_final_campaign(tmp_path)
    args.adaptive_production_topups = True
    return args, out


PLAN = TopupPlan(state_ids=(0, 1), steps=2000, deficit_state_ids=(0,), partner_state_ids=(1,), reason="planned")


def test_schedule_is_uniform_no_score_distribution(tmp_path):
    args, out = _scheduled_final_campaign(tmp_path)
    reg = ap.WindowStateRegistry.load(Path(out) / "adaptive_production")
    rows = ap.build_adaptive_epoch_schedule(reg, {"edges": [{"state_i": 0, "state_j": 1, "overlap": None}]},
                                            ap.AdaptiveDecisionPolicy(), epoch=1, default_steps=10_000)
    assert {r["requested_steps"] for r in rows} == {10_000}


def test_topups_on_runs_a_shortened_baseline_and_exactly_one_topup(tmp_path, monkeypatch):
    args, out = _topups_on(tmp_path)
    calls = _drive(monkeypatch, args, out, PLAN)
    names = [n for n, _ in calls]
    assert names.count("baseline") == 1 and [n for n in names if n.startswith("topup_")] == ["topup_001_2000"]


def test_the_topup_budget_uses_the_unshortened_default(tmp_path, monkeypatch):
    args, out = _topups_on(tmp_path)
    seen = {}

    def _capture(a, epoch_dir, registry, policy, *, full_steps):
        seen["full_steps"] = full_steps
        return TopupPlan(reason="healthy")

    monkeypatch.setattr(prod, "run_gareus", lambda a, d, *r, **k: seen.setdefault("baseline", int(a.gamd_production_steps)))
    monkeypatch.setattr(ap, "_topup_plan_for_phase", _capture)
    ap.run_adaptive_production_auto_loop(args, out, None, None, None, None, None, None)
    assert seen["full_steps"] > seen["baseline"]                 # budget from full, baseline shortened


def test_a_healthy_plan_runs_no_topup(tmp_path, monkeypatch):
    args, out = _topups_on(tmp_path)
    calls = _drive(monkeypatch, args, out, TopupPlan(reason="healthy"))
    assert not [n for n, _ in calls if n.startswith("topup_")]


def test_a_seed_mismatch_ends_the_topup_not_the_campaign(tmp_path, monkeypatch):
    args, out = _topups_on(tmp_path)

    def worker(name):
        if name.startswith("topup_"):
            raise SeedMismatchError("state 1 seeded from the wrong window")

    calls = _drive(monkeypatch, args, out, PLAN, worker=worker)
    assert any(n.startswith("topup_") for n, _ in calls)          # it was attempted, the drive returned normally


def test_windows_without_a_final_state_are_left_out(tmp_path, monkeypatch):
    args, out = _topups_on(tmp_path)
    shrunk = TopupPlan(state_ids=(0,), steps=2000, deficit_state_ids=(0,), reason="planned")
    calls = _drive(monkeypatch, args, out, PLAN, seedable=lambda plan, parents: shrunk)
    assert [n for n, _ in calls if n.startswith("topup_")] == ["topup_001_2000"]


def test_a_saved_plan_is_discarded_when_the_layout_changed(tmp_path, monkeypatch):
    from gareus.adaptive.topup_state import save_plan
    args, out = _topups_on(tmp_path)
    reg = ap.WindowStateRegistry.load(Path(out) / "adaptive_production")
    epoch_dir = Path(out) / "adaptive_production" / "final"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    save_plan(epoch_dir, TopupPlan(state_ids=(999,), steps=1000, deficit_state_ids=(999,), reason="planned"))
    monkeypatch.setattr(ap, "build_union_state_mbar_inputs", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no data")))
    plan = ap._topup_plan_for_phase(args, epoch_dir, reg, ap.policy_from_args(args), full_steps=10_000)
    assert 999 not in plan.state_ids and plan.reason == "no_diagnostics"
```

- [ ] **Step 2: Run to verify they fail** — expected import errors (`topup_state`, `_topup_plan_for_phase`, `gareus.topup_seeding` if Task 9 is not in yet: implement Task 9's `SeedMismatchError`/`load_seed_index` first or temporarily stub them in `gareus/topup_seeding.py` with the Task 9 signatures).

- [ ] **Step 3: Implement `gareus/adaptive/topup_state.py`**

```python
"""Top-up plan persistence (resume-stable) and campaign-level top-up state, written atomically."""
from __future__ import annotations

import json
import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

from .topup_allocator import TopupPlan

PLAN_NAME = "topup_plan.json"
STATE_NAME = "topup_state.json"
CORRECTION_BOUNDS = (0.1, 2.0)


def _clean(v):
    return None if isinstance(v, float) and not math.isfinite(v) else v


def _atomic_write(path: Path, obj: Any) -> Path:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False))
    os.replace(tmp, path)
    return path


def _num(v) -> float:
    return math.nan if v is None else float(v)


def save_plan(epoch_dir, plan: TopupPlan) -> Path:
    d = asdict(plan)
    for key in ("predicted_sigma", "sigma_before"):
        d[key] = {str(k): _clean(float(v)) for k, v in getattr(plan, key).items()}
    d["cost_hours"] = _clean(float(plan.cost_hours))
    d["structural_edges"] = [list(e) for e in plan.structural_edges]
    d["weak_edges_topped"] = [list(e) for e in plan.weak_edges_topped]
    return _atomic_write(Path(epoch_dir) / PLAN_NAME, d)


def load_plan(epoch_dir) -> Optional[TopupPlan]:
    path = Path(epoch_dir) / PLAN_NAME
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    for key in ("state_ids", "deficit_state_ids", "partner_state_ids"):
        d[key] = tuple(int(x) for x in d.get(key, ()))
    for key in ("structural_edges", "weak_edges_topped"):
        d[key] = tuple(tuple(int(x) for x in e) for e in d.get(key, ()))
    for key in ("predicted_sigma", "sigma_before"):
        d[key] = {int(k): _num(v) for k, v in d.get(key, {}).items()}
    d["cost_hours"] = _num(d.get("cost_hours"))
    return TopupPlan(**d)


def load_state(adaptive_dir) -> Dict[str, Any]:
    path = Path(adaptive_dir) / STATE_NAME
    if not path.exists():
        return {"correction": {}, "edge_attempts": {}, "f_kT": {}, "wall_time": []}
    d = json.loads(path.read_text())
    return {
        "correction": {int(k): float(v) for k, v in d.get("correction", {}).items()},
        "edge_attempts": {tuple(int(x) for x in k.split("-")): int(v) for k, v in d.get("edge_attempts", {}).items()},
        "f_kT": {int(k): float(v) for k, v in d.get("f_kT", {}).items()},
        "wall_time": list(d.get("wall_time", [])),
    }


def save_state(adaptive_dir, state: Dict[str, Any]) -> Path:
    d = {
        "correction": {str(k): v for k, v in state["correction"].items()},
        "edge_attempts": {f"{a}-{b}": v for (a, b), v in state["edge_attempts"].items()},
        "f_kT": {str(k): _clean(float(v)) for k, v in state["f_kT"].items()},
        "wall_time": state["wall_time"],
    }
    return _atomic_write(Path(adaptive_dir) / STATE_NAME, d)


def update_after_topup(state: Dict[str, Any], plan: TopupPlan, realised_sigma: Dict[int, float]) -> Dict[str, Any]:
    lo, hi = CORRECTION_BOUNDS
    new = {"correction": dict(state["correction"]), "edge_attempts": dict(state["edge_attempts"]),
           "f_kT": dict(state["f_kT"]), "wall_time": list(state["wall_time"])}
    for s in plan.deficit_state_ids:
        before, pred, real = plan.sigma_before.get(s), plan.predicted_sigma.get(s), realised_sigma.get(s)
        if not all(isinstance(v, float) and math.isfinite(v) for v in (before, pred, real)) or before <= pred:
            continue
        ratio = (before ** 2 - real ** 2) / (before ** 2 - pred ** 2)
        c = min(hi, max(lo, 0.7 * new["correction"].get(s, 1.0) + 0.3 * ratio))
        if ratio < 0.5:
            c = max(lo, 0.5 * c)
        new["correction"][s] = c
    for edge in plan.weak_edges_topped:
        key = (int(edge[0]), int(edge[1]))
        new["edge_attempts"][key] = new["edge_attempts"].get(key, 0) + 1
    return new
```

- [ ] **Step 4: Implement the wiring in `gareus/adaptive_production.py`**

(a) `build_adaptive_epoch_schedule`: replace everything after `state_rows = _state_rows_by_id(diagnostics)` with:

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

(b) Module-level helpers (monkeypatched by the tests, so call them through the module globals):

```python
def _topup_plan_for_phase(args, epoch_dir: Path, registry: "WindowStateRegistry",
                          policy: AdaptiveDecisionPolicy, *, full_steps: int):
    """Resume-stable top-up plan for this phase (spec 4.1-4.3)."""
    from .adaptive.throughput import wall_hours
    from .adaptive.topup_allocator import TopupPlan, plan_topup
    from .adaptive.topup_state import load_plan, load_state, save_plan, save_state
    from .adaptive.union_diagnostics import union_diagnostics_from_npz
    from .layout_neighbours import other_rung_same_centre, same_rung_neighbours, spatial_neighbour_pairs

    epoch_dir = Path(epoch_dir)
    adaptive_dir = epoch_dir.parent
    active = registry.active_states()
    ids = [int(s.state_id) for s in active]
    saved = load_plan(epoch_dir)
    if saved is not None:
        missing = sorted(set(saved.state_ids) - set(ids))
        if not missing:
            return saved
        print(f"      top-up plan discarded: layout changed (states {missing} no longer active)")
    state = load_state(adaptive_dir)
    c1 = [float(s.primary_center) for s in active]
    c2 = [float(s.secondary_center) if s.secondary_center is not None else 0.0 for s in active]
    lam = [float(s.gamd_lambda or 0.0) for s in active]
    k1 = [float(s.primary_k) for s in active]
    k2 = [float(s.secondary_k or 0.0) for s in active]
    temperature = float(getattr(args, "temperature_k", 300.0) or 300.0)
    pairs = spatial_neighbour_pairs(c1, c2, lam, k1, k2, temperature)
    nb_local, rp_local = same_rung_neighbours(pairs, lam), other_rung_same_centre(c1, c2, lam)
    neighbours = {ids[w]: [ids[x] for x in nb_local.get(w, [])] for w in range(len(ids))}
    rung_partners = {ids[w]: [ids[x] for x in rp_local.get(w, [])] for w in range(len(ids))}
    edges = [(a, b) for a, b, _t, _d in build_geometry_edges(registry, policy)]
    try:
        meta = build_union_state_mbar_inputs(adaptive_dir, registry)
        diag = union_diagnostics_from_npz(
            meta["arrays_npz"], edges, kt_kcal=0.0019872041 * temperature,
            subsample_counts=meta.get("subsample_counts_per_state"),
            min_effect_kcal=float(policy.topup_min_effect), f_init=state["f_kT"])
    except Exception as exc:
        print(f"      top-up diagnostics unavailable ({exc}); no top-up this phase")
        diag = None
    interval = int(getattr(args, "report_interval", 5000) or 5000)
    n_gpus = max(1, len(str(getattr(args, "device_index", "0")).split(",")))
    timestep = float(getattr(args, "timestep_fs", 4.0) or 4.0)
    budget = float(policy.topup_max_fraction) * wall_hours(int(full_steps), len(ids), timestep, n_gpus,
                                                           policy.topup_throughput_table)
    plan = plan_topup(diag, state_ids_in_order=ids, neighbours=neighbours, rung_partners=rung_partners,
                      policy=policy, report_interval=interval, timestep_fs=timestep, n_gpus=n_gpus,
                      budget_hours=budget, correction=state["correction"], edge_attempts=state["edge_attempts"])
    if diag is not None:
        state["f_kT"] = dict(diag.f_kT)
        save_state(adaptive_dir, state)
    save_plan(epoch_dir, plan)
    return plan


def _seedable_patch(plan, parent_dirs):
    """Drop states without an exported final State; skip the top-up if no deficit remains."""
    from dataclasses import replace
    from .topup_seeding import load_seed_index
    have = set(load_seed_index(parent_dirs))
    missing = [s for s in plan.state_ids if s not in have]
    if not missing:
        return plan
    print(f"      top-up: {len(missing)} window(s) without a final State left out (state_ids {missing})")
    keep = tuple(s for s in plan.state_ids if s in have)
    deficits = tuple(s for s in plan.deficit_state_ids if s in have)
    if not deficits:
        return replace(plan, state_ids=(), steps=0, deficit_state_ids=(), partner_state_ids=(), reason="no_seed_states")
    return replace(plan, state_ids=keep, deficit_state_ids=deficits,
                   partner_state_ids=tuple(s for s in plan.partner_state_ids if s in have))


def _after_topup_update(args, epoch_dir: Path, registry, policy, plan, elapsed_s: float) -> None:
    """Calibration, edge attempts and wall-time log after a completed top-up."""
    from .adaptive.topup_state import load_state, save_state, update_after_topup
    from .adaptive.union_diagnostics import union_diagnostics_from_npz
    adaptive_dir = Path(epoch_dir).parent
    state = load_state(adaptive_dir)
    try:
        meta = build_union_state_mbar_inputs(adaptive_dir, registry)
        edges = [(a, b) for a, b, _t, _d in build_geometry_edges(registry, policy)]
        diag = union_diagnostics_from_npz(meta["arrays_npz"], edges,
                                          kt_kcal=0.0019872041 * float(getattr(args, "temperature_k", 300.0)),
                                          subsample_counts=meta.get("subsample_counts_per_state"),
                                          min_effect_kcal=float(policy.topup_min_effect), f_init=state["f_kT"])
    except Exception as exc:
        print(f"      top-up calibration skipped ({exc})")
        diag = None
    if diag is not None:
        state = update_after_topup(state, plan, diag.sigma_kcal)
        state["f_kT"] = dict(diag.f_kT)
    state["wall_time"].append({"segment": f"{Path(epoch_dir).name}/topup_001_{plan.steps}",
                               "n_states": len(plan.state_ids), "steps": int(plan.steps),
                               "predicted_h": float(plan.cost_hours), "realised_h": float(elapsed_s) / 3600.0})
    save_state(adaptive_dir, state)
```

(c) In `run_scheduled_adaptive_epoch`: set `full_steps = baseline_steps` right after it is first computed; replace the PR #98 `topups_enabled` block and the `groups` loop with:

```python
    topups_enabled = bool(getattr(policy, "topups_enabled", False)) or _arg_bool(args, "adaptive_production_topups", False)
    if topups_enabled:
        baseline_steps = max(1000, _quantized_extra_steps(int(full_steps * (1.0 - float(policy.topup_max_fraction)))))
    else:
        requested = [int(r.get("requested_steps", 0) or 0) for r in schedule if int(r.get("requested_steps", 0) or 0) > 0]
        baseline_steps = max(baseline_steps, _quantized_extra_steps(int(round(sum(requested) / len(requested)))))
```

and after `run_segment("baseline", ...)` plus its interruption check:

```python
    if topups_enabled:
        from dataclasses import replace
        from .adaptive.topup_state import save_plan
        from .topup_seeding import SeedMismatchError
        plan = _topup_plan_for_phase(args, epoch_dir, registry, policy, full_steps=full_steps)
        if plan.reason == "planned":
            plan = _seedable_patch(plan, [epoch_dir / "baseline"])
        print(f"      top-up plan: {plan.reason}, {len(plan.state_ids)} states "
              f"({len(plan.deficit_state_ids)} deficit + {len(plan.partner_state_ids)} partners), "
              f"{plan.steps} steps, {plan.cost_hours:.2f} h predicted")
        if plan.reason == "planned" and plan.state_ids and plan.steps > 0:
            t_start = time.monotonic()
            try:
                run_segment(f"topup_001_{plan.steps}", list(plan.state_ids), int(plan.steps))
            except SeedMismatchError as exc:
                print(f"      top-up aborted, campaign continues: {exc}")
                save_plan(epoch_dir, replace(plan, reason="seed_mismatch"))
            else:
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
                _after_topup_update(args, epoch_dir, registry, policy, plan, time.monotonic() - t_start)
```

Delete the leftover old-path code (`groups`, grouping by extra size). After `diagnostics = collect_segmented_epoch_diagnostics(...)`, set `edge["mbar_overlap"]` from the saved plan's diagnostics as described in behaviour 3 (load the diagnostics again with `union_diagnostics_from_npz` only if a plan was computed this call; otherwise leave the diagnostics as they are). Import `time` at module top if it is not imported.

- [ ] **Step 5: Run tests**

Run: `python -m pytest -q -p no:cacheprovider tests/test_topup_state.py tests/test_topup_epoch_wiring.py tests/test_no_topups.py tests/test_scheduled_final_interruption.py tests/test_epoch_window_map_rewrite_after_drop.py tests/test_adaptive_segmented_diagnostics.py tests/test_ap_epoch0_step_fraction.py`
Expected: PASS. Tests in `test_epoch_window_map_rewrite_after_drop.py` that build top-up segments through the old grouping must set `adaptive_production_topups = True` and monkeypatch `_topup_plan_for_phase`/`_seedable_patch`; say so in the commit body.

- [ ] **Step 6: Commit**

```bash
git add gareus/adaptive_production.py gareus/adaptive/topup_state.py tests/test_topup_state.py tests/test_topup_epoch_wiring.py tests/
git commit -m "feat: scheduled phases run at most one allocator-planned top-up; remove score allocator"
```

---

### Task 9: Top-up seeding from the parent segment's final window States

**Files:**
- Create: `gareus/topup_seeding.py`
- Modify: `gareus/production.py` — segment end (right after the `final_pdbs` loop, `# Save one final PDB per configuration replica.`); the fresh-path block around `generate_us_starting_states_by_pulling(` in `run_gareus`; the production replica construction that consumes `window_start_positions` / `window_start_velocities` (add a per-window box)
- Modify: `tests/pep_gamd_fixture.py` (add `build_small_simulation`, below)
- Test: `tests/test_topup_seeding.py`

**Interfaces:**
- Produces: `export_final_window_states(out_dir, sims, assignments, state_id_of_window: dict, cv_of_replica) -> Path` — writes `<out_dir>/final_window_states/state_<sid>.xml` (OpenMM `XmlSerializer` State: positions, velocities, box) and `index.json` (`{sid: {"window": w, "cv1": x, "cv2": y}}`).
- Produces: `load_seed_index(parent_dirs) -> dict[int, str]` (state_id → parent dir holding its State; later parents win; reads `index.json` only, no XML parsing — used by the driver's `_seedable_patch`, Task 8).
- Produces: `load_seed_states(parent_dirs, state_ids) -> dict[int, SeedState]` with `SeedState(positions, velocities, box, cv1, cv2, source)`.
- Produces: `assert_seed_matches(seed, cv1_now, cv2_now, tol=1e-3)` raising `SeedMismatchError(RuntimeError)`.

Behaviour in `run_gareus`, fresh path, when `(getattr(args, "_adaptive_phase_info", {}) or {}).get("is_topup")`:
- parent dirs = `[out_dir.parent / "baseline", *sorted(out_dir.parent.glob("topup_*"))]` minus `out_dir`; window → state_id from this segment's `epoch_window_map.csv` (read with `_read_csv_dicts`; identity when absent);
- every window must have a seed State (the driver removed the others, Task 8); if one is missing anyway, raise `SeedMismatchError` naming it (the driver ends the top-up, the campaign continues);
- set `window_start_positions[w]`, `window_start_velocities[w]`, `window_start_boxes[w]` from the seeds and **skip `generate_us_starting_states_by_pulling` entirely** (no pull inside a top-up);
- after the production replicas are built and `set_window` applied, for each window evaluate CV1/CV2 on its context (the same evaluation the sample logger uses) and call `assert_seed_matches`; a mismatch raises out of `run_gareus` (same handling).

Export at the end of every segment: `state_id_of_window` from the segment's `epoch_window_map.csv` (identity when absent); `cv_of_replica(r)` evaluates CV1/CV2 on `sims[r].context` with the sample logger's function (find it: `grep -n "cv_A" gareus/production.py | head`; reuse it, do not reimplement). The export uses the live in-memory `assignments` (authoritative for which window each replica holds), not the checkpoint manifest.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_topup_seeding.py
import json

import pytest

from gareus.topup_seeding import (
    SeedMismatchError, SeedState, assert_seed_matches, load_seed_index, load_seed_states,
)


def _fake_export(d, sid, cv1, cv2, xml="<State/>"):
    (d / "final_window_states").mkdir(parents=True, exist_ok=True)
    (d / "final_window_states" / f"state_{sid}.xml").write_text(xml)
    idx = d / "final_window_states" / "index.json"
    data = json.loads(idx.read_text()) if idx.exists() else {}
    data[str(sid)] = {"window": sid, "cv1": cv1, "cv2": cv2}
    idx.write_text(json.dumps(data))


def test_the_index_names_the_latest_parent_per_state(tmp_path):
    base, top = tmp_path / "baseline", tmp_path / "topup_001_1000"
    _fake_export(base, 3, 0.1, 0.2); _fake_export(base, 4, 0.3, 0.4); _fake_export(top, 3, 0.5, 0.6)
    idx = load_seed_index([base, top])
    assert idx == {3: str(top), 4: str(base)}
    assert load_seed_index([tmp_path / "nothing"]) == {}


def test_loads_only_the_states_it_has(tmp_path, monkeypatch):
    import gareus.topup_seeding as ts
    monkeypatch.setattr(ts, "_deserialize_state", lambda text: ("pos", "vel", "box"))
    base = tmp_path / "baseline"
    _fake_export(base, 3, 0.1, 0.2)
    got = load_seed_states([base], [3, 9])
    assert set(got) == {3} and got[3].cv1 == 0.1


def test_assertion_accepts_a_matching_frame_and_rejects_a_swapped_one():
    seed = SeedState(positions=None, velocities=None, box=None, cv1=0.40, cv2=-0.25, source="x")
    assert_seed_matches(seed, 0.4004, -0.2496)
    with pytest.raises(SeedMismatchError, match="does not reproduce"):
        assert_seed_matches(seed, 0.47, -0.25)


def test_export_and_reload_round_trip_on_the_reference_platform(tmp_path):
    pytest.importorskip("openmm")
    from pep_gamd_fixture import build_small_simulation
    from gareus.topup_seeding import export_final_window_states
    sim = build_small_simulation(platform="Reference")
    export_final_window_states(tmp_path, [sim], assignments=[0], state_id_of_window={0: 7},
                               cv_of_replica=lambda r: (0.11, 0.22))
    seed = load_seed_states([tmp_path], [7])[7]
    st = sim.context.getState(getPositions=True, getVelocities=True)
    assert seed.cv1 == 0.11 and len(seed.positions) == len(st.getPositions())
    assert seed.box is not None
```

Append to `tests/pep_gamd_fixture.py` (reuses the cached GA-dipeptide system):

```python
def build_small_simulation(platform="Reference"):
    """An app.Simulation on a fresh copy of the cached GA dipeptide in TIP3P, velocities set."""
    openmm, app, unit, topology, system, positions = tiny_solvated_system()
    integ = openmm.LangevinMiddleIntegrator(300 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picoseconds)
    sim = app.Simulation(topology, system, integ, openmm.Platform.getPlatformByName(platform))
    sim.context.setPositions(positions)
    sim.context.setVelocitiesToTemperature(300 * unit.kelvin, 1)
    return sim
```

- [ ] **Step 2: Run to verify they fail** — expected `ModuleNotFoundError: gareus.topup_seeding`.

- [ ] **Step 3: Implement `gareus/topup_seeding.py`**

```python
"""Continue each top-up window's chain from its parent segment's final State (spec 4.4, revision 2)."""
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
    tmp = d / "index.json.tmp"
    tmp.write_text(json.dumps(index, indent=2))
    tmp.replace(d / "index.json")
    return d


def load_seed_index(parent_dirs: Iterable) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for parent in parent_dirs:                       # later parents override earlier ones
        idx = Path(parent) / DIR_NAME / "index.json"
        if not idx.exists():
            continue
        for key in json.loads(idx.read_text()):
            if (Path(parent) / DIR_NAME / f"state_{int(key)}.xml").exists():
                out[int(key)] = str(parent)
    return out


def load_seed_states(parent_dirs: Iterable, state_ids: Iterable[int]) -> Dict[int, SeedState]:
    parents = list(parent_dirs)
    where = load_seed_index(parents)
    out: Dict[int, SeedState] = {}
    for sid in state_ids:
        parent = where.get(int(sid))
        if parent is None:
            continue
        d = Path(parent) / DIR_NAME
        rec = json.loads((d / "index.json").read_text())[str(int(sid))]
        pos, vel, box = _deserialize_state((d / f"state_{int(sid)}.xml").read_text())
        out[int(sid)] = SeedState(pos, vel, box, float(rec["cv1"]), float(rec["cv2"]), str(parent))
    return out


def assert_seed_matches(seed: SeedState, cv1_now: float, cv2_now: float, tol: float = 1e-3) -> None:
    if abs(cv1_now - seed.cv1) > tol or abs(cv2_now - seed.cv2) > tol:
        raise SeedMismatchError(
            f"seeded state from {seed.source} does not reproduce its recorded CVs "
            f"(cv1 {cv1_now:.6f} vs {seed.cv1:.6f}, cv2 {cv2_now:.6f} vs {seed.cv2:.6f})")
```

- [ ] **Step 4: Wire into `production.py`** as described under Behaviour: export after the `final_pdbs` loop; in the fresh path, when `is_topup`, fill `window_start_positions/velocities/boxes` from `load_seed_states(...)` and skip the pull call; default `window_start_boxes = [None] * nrep` wherever `window_start_positions` is initialised; in the production replica construction use `box = window_start_boxes[i] if window_start_boxes and window_start_boxes[i] is not None else equil_box` for `setPeriodicBoxVectors`; after replicas are built and `set_window` applied, `assert_seed_matches(seed, *cv_of_replica(r))` per seeded window.
- [ ] **Step 5: Run tests** — `tests/test_topup_seeding.py tests/test_pep_gamd_wiring.py tests/test_resume_round_trip.py`; expected PASS (the Reference round-trip must run locally).
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
- the pooled samples are thinned to decorrelated ones per window and written in the union NPZ format, then solved with `union_diagnostics_from_npz` (real production code path), passing `subsample_counts={str(state): {"raw": n_raw, "t0": 0, "kept": n_kept, "g": 1 + 2*tau, "status": "subsampled"}}` from the harness's own mixing model;
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
