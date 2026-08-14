# MBAR Analysis Modularization — Plan A6a Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Relocate `analyze_gareus_mbar.py`'s 49 plotting/reporting functions (41 from the original function list, plus `_write_csv_rows` and the `_OPT_IN_GAMD_METHODS` constant, plus the six CV-label helpers the cross-plan reconciliation pass found unowned — see Global Constraints) plus `gareus_plotstyle.py` into five new, size-bounded modules under `gareus/mbar_analysis/`, with zero behavior change and zero broken imports (either in `analyze_gareus_mbar.py`'s own internal callers or in the ~21 existing test files that do `from analyze_gareus_mbar import <name>`). `_window_moments` is **not** relocated here — see the correction in Global Constraints below.

**Architecture:** Five new files — `plotstyle.py` (whole-file relocation), `plotting.py` (matplotlib renderers + smoothing/label helpers), `writers.py` (CSV/NPZ serializers, imports from `plotting.py`), `convergence_reporting.py` (convergence-specific figure/report writers, imports from `plotting.py`), `summary.py` (markdown summary renderers, no cross-file imports). `analyze_gareus_mbar.py` keeps every relocated name resolvable via `from gareus.mbar_analysis.<module> import <name>` re-exports placed where the original definitions lived. One dead function (`write_convergence_report`) is deleted outright, not relocated. The chignolin-kJ plotting trio is explicitly left untouched (Plan A6c's job).

**Tech Stack:** Python 3.10+, pytest, matplotlib (Agg backend in tests), numpy.

**Spec:** `docs/superpowers/specs/2026-08-13-mbar-analysis-modularization-a6a-design.md`
**Split rationale:** `docs/superpowers/specs/2026-08-13-mbar-analysis-modularization-a6-split-proposal.md`

## Global Constraints

- **Byte-identical output before/after.** This is a pure code relocation. No function signature, default value, docstring content, or line of logic changes — only *which file it lives in*.
- **Re-verify every line range against the current `analyze_gareus_mbar.py` before editing.** All ranges in this plan are pinned to commit `024cea7` of this worktree. Run `grep -n "^def <name>" analyze_gareus_mbar.py` for the first and last function in each group immediately before that task's Step 3 — if a range has shifted (e.g. because A2-A5 landed a change first), stop and recompute the exact range rather than trusting this document blindly.
- **Land in this exact order: `plotstyle.py` → `plotting.py` → `writers.py`/`convergence_reporting.py` (either order) → `summary.py`.** `writers.py` and `convergence_reporting.py` both import `_smooth_pmf_1d` (and `writers.py` also imports `_plot_2d_fes_multirange`/`_want_gamd_method`) from `plotting.py`, which itself imports `plotstyle`. Landing out of order means writing code that imports a module that doesn't exist yet.
- **Preserve the lazy per-function `import matplotlib.pyplot as plt` pattern exactly.** Do not hoist matplotlib (or `matplotlib.cm`/`matplotlib.ticker`/`matplotlib.patches`) imports to module scope in any new file — the original file does this deliberately so a missing matplotlib install degrades per-call, not at import time.
- **`analyze_gareus_mbar.py` must keep every relocated name resolvable at its original qualified path.** After each task, `from analyze_gareus_mbar import <name>` and `analyze_gareus_mbar.<name>` must both still work, and must refer to the *same object* as the new module's copy (`is`, not `==`) — proving a real re-export, not a coincidentally-equal reimplementation.
- **`write_convergence_report` (L4576-4604) is deleted, not relocated.** Confirmed dead: `grep -rn write_convergence_report .` (repo-wide) finds only its own `def` line, before this plan's changes.
- **Deleting the top-level `gareus_plotstyle.py` requires fixing *every* remaining bare-name consumer in the same change**, not just the ones this plan's 41-function list owns. `analyze_secondary_cv_pmf` (not part of this plan — it stays in `analyze_gareus_mbar.py`, likely relocated by a later plan) has its own function-local `import gareus_plotstyle as ps` that must be repointed to `import gareus.mbar_analysis.plotstyle as ps` in Task 1, or it will raise `ModuleNotFoundError` the moment the old file is deleted.
- **No test file may monkeypatch a name through a stale module reference.** `tests/test_perf_plotting_redundancy.py`'s `test_plot_2d_fes_multirange_renders_each_range_exactly_once` patches `agm._plot_2d_fes_range` and calls `agm._plot_2d_fes_multirange(...)`; once both live in `gareus/mbar_analysis/plotting.py`, `_plot_2d_fes_multirange`'s internal call to `_plot_2d_fes_range` resolves in `plotting.py`'s own globals, not `analyze_gareus_mbar`'s re-exported binding — the patch would silently stop taking effect. Task 6 fixes this explicitly; do not skip it.
- **`_write_csv_rows` (L4440-4456) belongs to `writers.py` (Task 3), not `checkpoint_steps_from_data`'s A6b neighborhood, despite living physically closer to `run_epoch_pmf_convergence`.** Direct verification (`grep -n "_write_csv_rows(" analyze_gareus_mbar.py`) shows 15 call sites spanning Plan A6b (10, both convergence drivers), A6c (1, `analyze_rg`), and A6d (4, three functions) — all of which execute *after* this plan. Leaf-first ordering requires it land in the one plan that precedes all three consumers, which is A6a.
- **CORRECTION — `_window_moments` does NOT move with this plan; it is claimed by Plan A4, not A6a.** An earlier draft of this constraint listed `_window_moments` (L7104-7116, a private skew/kurtosis/anharmonicity helper called only by `_per_window_gamd_boost_stats`) as an A6a addition. That is wrong: `docs/superpowers/specs/2026-08-13-mbar-analysis-modularization-a4-design.md` explicitly lists `_window_moments` among its own "24 functions move, verbatim" (alongside `boost_stats`, `_window_cv_mean_std`) into `gareus/mbar_analysis/pmf.py`, and states outright that "A6, when it relocates [`_per_window_gamd_boost_stats`], will need to import ... `_window_moments` from `gareus.mbar_analysis.pmf`." So `_per_window_gamd_boost_stats` (Task 2, staying in A6a) must `from gareus.mbar_analysis.pmf import _window_moments` — or, until A4 actually lands, `from analyze_gareus_mbar import _window_moments` (script still owns it at this plan's likely execution time) — never define its own copy. The module-level `_OPT_IN_GAMD_METHODS` dict constant (L6386, defined immediately above `_want_gamd_method`, its only reader) is unaffected by this correction and does move with Task 2 as originally stated — it is not named in A4's function list and has no owner besides this plan.
- **One external symbol this plan's relocated code calls but does not own**: `_slug(text, max_len=80)` (L6165-6177, inside Plan A6d's own future "residue/topology observables" scope), called by `_regime_slug` (Task 2, `plotting.py`) and by `_write_rama_2d` (Task 3, `writers.py`). Both call sites are handled the same way — **a local, function-body `from analyze_gareus_mbar import _slug`** (the script still holds the real definition at this plan's execution time) — flagged explicitly in Tasks 2 and 3 below rather than silently added as an undocumented dependency. Note `_regime_slug` is a single-expression `return ... if regime else 'unknown'` with no statement to place a local import before, so Task 2 gives its replacement body verbatim (an early return, semantically identical, keeping the import off the `regime == ''` path) — the one relocated body in this plan whose structure, not just its imports, changes. `_window_moments` (Task 2, Plan A4's) is the other interim import, handled with a defensive `try/except ImportError`; see the `_window_moments` CORRECTION above.
- **CORRECTION — the six CV-label helpers ARE this plan's, and are relocated by Task 2 into `plotting.py`.** An earlier draft of this plan treated `_primary_cv_axis_label` (L603-608) as an unowned external symbol, bridged back to `analyze_gareus_mbar` from `plot_outputs`/`plot_rg_outputs` with a local import, on the grounds that no A2-A6 domain description claimed it. The cross-plan reconciliation pass confirmed that the non-ownership was real and, crucially, **not limited to that one function**: `_secondary_cv_label` (L323-340), `_secondary_cv_regions` (L341-347), `_regime_slug` (L510-512), `_primary_cv_label` (L589-595), `_primary_cv_units` (L596-602), and `_primary_cv_axis_label` (L603-608) were all unclaimed by any of the ten A1-A6a plans. `...-a2-design.md`'s Out Of Scope explicitly disclaims four of them ("they belong with whichever plan owns secondary-CV PMF/2D-FES analysis — referenced but not claimed by this plan"), and `...-a4-design.md` lists them in its own dependency table as `unowned` (`_secondary_cv_label`, `_secondary_cv_regions`) or resolves them through its call-time `_bridge()` helper. (Note that A4's table attributes `_regime_slug` to A2 — that attribution is stale, since A2's own spec explicitly disclaims it; A4 needs no edit either way, because `_bridge()` resolves through `analyze_gareus_mbar`'s namespace regardless of which module ultimately backs the name.) The coordinator assigned all six here: they are pure, stateless `meta: dict -> str/list` label formatters whose entire job is producing matplotlib axis/annotation text, which is exactly `plotting.py`'s stated domain ("matplotlib renderers **+ smoothing/label helpers**"), and `plot_outputs`/`plot_rg_outputs` — already Task 2's — are among their callers. **Consequences**: `_primary_cv_axis_label` is no longer an interim bridged import (it lives in `plotting.py` alongside its callers, so no import statement is needed at all); `_regime_slug` brings a *new* interim dependency on `_slug` into Task 2, handled exactly like Task 3's; and A4's `_bridge()`-based resolution of these names keeps working unchanged, since it resolves through `analyze_gareus_mbar`'s namespace, which this task's re-export block keeps populated.

---

### Task 1: Relocate `gareus_plotstyle.py` → `gareus/mbar_analysis/plotstyle.py`

**Files:**
- Create: `gareus/mbar_analysis/plotstyle.py` (128 lines, verbatim copy of `gareus_plotstyle.py`)
- Delete: `gareus_plotstyle.py`
- Modify: `tests/test_gareus_plotstyle.py` (1-line import path update)
- Modify: `analyze_gareus_mbar.py` (1-line import path update inside `analyze_secondary_cv_pmf`'s body — this function is otherwise untouched by this plan)
- Test: `tests/test_mbar_analysis_plotstyle.py` (new file)

**Interfaces:**
- Produces: `gareus.mbar_analysis.plotstyle.{OKABE_ITO, OKABE_ITO_LINES, METHOD_STYLE, _PRETTY, BAR_COLOR, REFERENCE_GREY, method_style, method_color, pretty_method, plot_method_curve, style_line_axes, annotate_minimum}` — consumed by Task 2 (`plotting.py`) and by `analyze_secondary_cv_pmf` (unchanged, outside this plan's scope).

- [ ] **Step 1: Write the failing test**

Create `tests/test_mbar_analysis_plotstyle.py`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_plotstyle_module_is_importable_from_package():
    import gareus.mbar_analysis.plotstyle as ps  # noqa: F401


def test_plotstyle_exports_match_original_top_level_module():
    import gareus.mbar_analysis.plotstyle as ps

    assert ps.OKABE_ITO == [
        "#000000", "#E69F00", "#56B4E9", "#009E73",
        "#F0E442", "#0072B2", "#D55E00", "#CC79A7",
    ]
    assert ps.METHOD_STYLE["gamd_cumulant2"] == ("#0072B2", "-")
    assert ps.pretty_method("gamd_cumulant2") == "GaMD cumulant-2"
    assert ps.pretty_method("unknown_method") == "unknown_method"
    assert ps.method_color("umbrella_only") == "#009E73"


def test_old_top_level_module_is_gone():
    assert not Path(__file__).resolve().parent.parent.joinpath("gareus_plotstyle.py").exists()


def test_analyze_secondary_cv_pmf_no_longer_references_bare_gareus_plotstyle():
    repo_root = Path(__file__).resolve().parent.parent
    text = (repo_root / "analyze_gareus_mbar.py").read_text()
    assert "import gareus_plotstyle" not in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_mbar_analysis_plotstyle.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gareus.mbar_analysis.plotstyle'` (first test), and the last test fails too since `analyze_gareus_mbar.py` still has `import gareus_plotstyle as ps` inside `analyze_secondary_cv_pmf`.

- [ ] **Step 3: Relocate the file**

Read `gareus_plotstyle.py` in full (128 lines) and write its exact, unmodified content to `gareus/mbar_analysis/plotstyle.py`. Do not change a single line — this is a whole-file move, not a rewrite.

Delete `gareus_plotstyle.py`.

In `tests/test_gareus_plotstyle.py`, change:
```python
import gareus_plotstyle as ps  # noqa: E402
```
to:
```python
import gareus.mbar_analysis.plotstyle as ps  # noqa: E402
```
(Re-confirm the exact current line via `grep -n "import gareus_plotstyle" tests/test_gareus_plotstyle.py` first — it is the only import of this module in that file per the file's own docstring, so no other line in it needs touching.)

In `analyze_gareus_mbar.py`, find `analyze_secondary_cv_pmf`'s own local import:

Run: `grep -n "import gareus_plotstyle" analyze_gareus_mbar.py`

This should show exactly one remaining match after Task 2-5 haven't run yet (there are 4 total in the file today — 3 inside functions this plan relocates in Task 2, which disappear naturally when those function bodies are deleted; this one, inside `analyze_secondary_cv_pmf`, is the only one that survives and must be fixed directly). Change it from:
```python
    import gareus_plotstyle as ps
```
to:
```python
    import gareus.mbar_analysis.plotstyle as ps
```
(same indentation, same local/lazy-import position — do not hoist it to module scope).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_mbar_analysis_plotstyle.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Confirm the two real consumers still work**

Run: `pytest tests/test_gareus_plotstyle.py -v`
Expected: PASS, unchanged pass count from before this task.

Run: `python -m py_compile gareus/mbar_analysis/plotstyle.py analyze_gareus_mbar.py`
Expected: clean compile.

Run: `grep -rn "import gareus_plotstyle" .` (repo-wide)
Expected: zero matches (confirms no other stray consumer exists beyond the two already fixed).

- [ ] **Step 6: Commit**

```bash
git add gareus/mbar_analysis/plotstyle.py tests/test_gareus_plotstyle.py tests/test_mbar_analysis_plotstyle.py analyze_gareus_mbar.py
git rm gareus_plotstyle.py
git commit -m "refactor: relocate gareus_plotstyle.py into gareus.mbar_analysis.plotstyle"
```

---

### Task 2: Create `gareus/mbar_analysis/plotting.py`

**Files:**
- Create: `gareus/mbar_analysis/plotting.py` (~475 lines)
- Modify: `analyze_gareus_mbar.py` (delete 25 function/constant definitions — 19 renderer/smoothing/FES-range names plus the 6 CV-label helpers; count them off the table below rather than trusting this number, an earlier draft said "18" and under-counted the renderer cluster by one — and add 2 re-export blocks)
- Test: `tests/test_mbar_analysis_plotting.py` (new file)

**Interfaces:**
- Consumes: `gareus.mbar_analysis.plotstyle` (Task 1); `_window_moments` from wherever Plan A4 lands `gareus/mbar_analysis/pmf.py` (or, until A4 lands, from `analyze_gareus_mbar` — see the CORRECTION in Global Constraints; `_per_window_gamd_boost_stats` needs it but this plan does not own or relocate it); `_slug` from `analyze_gareus_mbar` (Plan A6d's, needed by `_regime_slug` — same interim local-import treatment as Task 3's `_write_rama_2d`).
- Produces: `gareus.mbar_analysis.plotting.{_secondary_cv_label, _secondary_cv_regions, _regime_slug, _primary_cv_label, _primary_cv_units, _primary_cv_axis_label, _smooth_masked_grid, _smooth_pmf_1d, _eff_smooth, FES_PLOT_VMAX_VALUES, _fes_range_tag, _fes_range_label, _fes_variant_path, _plot_2d_fes_range, _plot_2d_fes_multirange, plot_2d_fes, plot_cv1_cv2_2d_fes, plot_pca_2d_fes, _per_window_gamd_boost_stats, plot_gamd_boost, plot_outputs, plot_rg_outputs, _OPT_IN_GAMD_METHODS, _want_gamd_method, _visible_pmfs}` — consumed by Task 3 (`writers.py`), Task 4 (`convergence_reporting.py`), and (once A4 lands, out of scope here) `run_pmf_and_gamd_boost_report`/`run_secondary_cv_analyses`/`analyze_secondary_cv_pmf`, which reach the six CV-label helpers through A4's own call-time `_bridge()` and so need no change from this task.

**Exact line ranges to relocate** (re-verify each with `grep -n "^def <name>"` before extracting — the renderer/smoothing/FES-range entries are pinned to commit `024cea7`; the six CV-label entries were added later and re-verified directly against the working tree, which agrees with the older pinning — `_primary_cv_axis_label` is L603-608 in both):

```
--- CV-label helper cluster (three regions near the top of the file) ---
_secondary_cv_label           L323-340  (18)
_secondary_cv_regions         L341-347  ( 7)
_regime_slug                  L510-512  ( 3)
_primary_cv_label             L589-595  ( 7)
_primary_cv_units             L596-602  ( 7)
_primary_cv_axis_label        L603-608  ( 6)
--- renderers / smoothing / FES-range machinery ---
_smooth_masked_grid           L3822-3841 (20)
_smooth_pmf_1d                L3842-3862 (21)
_eff_smooth                   L3863-3869 ( 7)
FES_PLOT_VMAX_VALUES (const)  L3870      ( 1)
_fes_range_tag                L3872-3879 ( 8)
_fes_range_label              L3880-3886 ( 7)
_fes_variant_path             L3887-3891 ( 5)
_plot_2d_fes_range            L3892-3961 (70)
_plot_2d_fes_multirange       L3962-3986 (25)
plot_2d_fes                   L3987-3994 ( 8)
plot_cv1_cv2_2d_fes           L3995-4002 ( 8)
plot_pca_2d_fes               L6078-6085 ( 8)
_OPT_IN_GAMD_METHODS (const)  L6386      ( 1)
_want_gamd_method             L6388-6394 ( 7)
_visible_pmfs                 L6395-6399 ( 5)
_per_window_gamd_boost_stats  L7119-7182 (64)
plot_gamd_boost               L7183-7307 (125)
plot_outputs                  L7308-7324 (17)
plot_rg_outputs               L5665-5683 (19)
```

**`_window_moments` (L7104-7116) does NOT move with this task** despite sitting immediately above `_per_window_gamd_boost_stats` in the source file — see the CORRECTION in Global Constraints: it is explicitly claimed by Plan A4's design spec for `gareus/mbar_analysis/pmf.py`, alongside `boost_stats` (L7094-7102, also A4's, two lines above it). Do not extract L7104-7116 into this region; the extraction for this cluster starts at L7119 (`_per_window_gamd_boost_stats`), not L7104.

Note `_eff_smooth` and `FES_PLOT_VMAX_VALUES` are adjacent (L3863-3870) — extract them as one contiguous block. **`_OPT_IN_GAMD_METHODS` (L6386) is a module-level dict constant immediately preceding `_want_gamd_method`, its only reader — verified directly by reading L6376-6399: `_finalize_acc_2d` (Plan A6d's function, stays behind) actually ends at L6384, then a blank line, then `_OPT_IN_GAMD_METHODS = {...}` at L6386, then a blank line, then `def _want_gamd_method` at L6388. The region for this cluster is `6386-6399`, not `6388-6399` — extracting only `6388-6399` would leave `_OPT_IN_GAMD_METHODS` behind in `analyze_gareus_mbar.py`, and `_want_gamd_method`'s body (`_OPT_IN_GAMD_METHODS.get(method)`) would raise `NameError` at call time in its new home. Re-verify this exact boundary with `sed -n '6376,6399p' analyze_gareus_mbar.py` (or current equivalent line numbers) before extracting — do not trust the round-number `6388` boundary from a naive `next-def-start` scan.**

**The CV-label cluster's three regions each abut a function that must stay behind** — the same boundary-precision hazard as `_window_moments`/`_OPT_IN_GAMD_METHODS` above, so pin each region by its neighbor, not by a naive `next-def` scan:

- Region `323-347` ends at `_secondary_cv_regions`'s last line plus one blank. **`_epoch_run_manifest_secondary_cv_type` (L349) stays behind** — it reads a phase's `run_manifest.json` from disk (Plan A2's `Data`/epoch-metadata domain, `...-a2-design.md` claims it explicitly), not a label. Do not let the region reach L349.
- Region `510-512` is `_regime_slug` alone. **`run_secondary_cv_analyses` (L514) stays behind** — it is the ~74-line secondary-CV PMF/2D-FES orchestrator claimed by Plan A4, and it is `_regime_slug`'s only caller in this file. A `next-def`-boundary scan starting at L510 would sweep the whole orchestrator into `plotting.py`.
- Region `589-608` ends at `_primary_cv_axis_label`'s last line plus one blank. **`_poincare_primary_cv_supported` (L609) stays behind** — despite reading the same three `meta` keys, it is a Poincaré-analysis capability predicate (Plan A6d's observables scope), not a label formatter, and nothing in `plotting.py` calls it.

Re-verify all three with `grep -n "^def _secondary_cv_label\|^def _secondary_cv_regions\|^def _epoch_run_manifest_secondary_cv_type\|^def _regime_slug\|^def run_secondary_cv_analyses\|^def _primary_cv_label\|^def _primary_cv_units\|^def _primary_cv_axis_label\|^def _poincare_primary_cv_supported" analyze_gareus_mbar.py` before extracting — the stay-behind names are in that grep deliberately, so you can see exactly where each region must stop.

`plot_rg_outputs` (L5665-5683), `plot_pca_2d_fes` (L6078-6085), and the `_OPT_IN_GAMD_METHODS`/`_want_gamd_method`/`_visible_pmfs` cluster (L6386-6399) are physically distant from the L3822-4002 and L7119-7324 clusters — eight separate contiguous source regions in total: `323-347`, `510-512`, `589-608`, `3822-4002`, `5665-5683`, `6078-6085`, `6386-6399`, `7119-7324`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_mbar_analysis_plotting.py`:

```python
import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import analyze_gareus_mbar as agm

_NAMES = [
    "_secondary_cv_label", "_secondary_cv_regions", "_regime_slug",
    "_primary_cv_label", "_primary_cv_units", "_primary_cv_axis_label",
    "_smooth_masked_grid", "_smooth_pmf_1d", "_eff_smooth", "FES_PLOT_VMAX_VALUES",
    "_fes_range_tag", "_fes_range_label", "_fes_variant_path", "_plot_2d_fes_range",
    "_plot_2d_fes_multirange", "plot_2d_fes", "plot_cv1_cv2_2d_fes", "plot_pca_2d_fes",
    "_per_window_gamd_boost_stats", "plot_gamd_boost", "plot_outputs",
    "plot_rg_outputs", "_OPT_IN_GAMD_METHODS", "_want_gamd_method", "_visible_pmfs",
]


def test_plotting_module_is_importable():
    import gareus.mbar_analysis.plotting  # noqa: F401


def test_every_relocated_name_is_the_same_object_via_reexport():
    import gareus.mbar_analysis.plotting as plotting

    for name in _NAMES:
        assert hasattr(plotting, name), f"plotting.py missing {name}"
        assert hasattr(agm, name), f"analyze_gareus_mbar re-export missing {name}"
        assert getattr(agm, name) is getattr(plotting, name), (
            f"{name}: analyze_gareus_mbar copy is not the same object as "
            "gareus.mbar_analysis.plotting's — re-export is broken"
        )


def test_signatures_unchanged():
    import gareus.mbar_analysis.plotting as plotting

    assert list(inspect.signature(plotting.plot_outputs).parameters) == [
        "d", "pmfs", "selected", "O", "out", "warnings", "smooth_sigma", "args",
    ]
    assert list(inspect.signature(plotting.plot_gamd_boost).parameters) == ["d", "out", "warnings"]
    assert list(inspect.signature(plotting._smooth_pmf_1d).parameters) == ["pmf", "sigma"]


def test_fes_range_helpers_unchanged_behavior():
    import gareus.mbar_analysis.plotting as plotting

    assert plotting._fes_range_tag(None) == "0_all"
    assert plotting._fes_range_tag(5.0) == "0_5"
    assert plotting.FES_PLOT_VMAX_VALUES == (2.0, 5.0, 10.0, 20.0, None)


def test_want_gamd_method_resolves_its_module_constant_after_move():
    # Regression guard for the L6386 boundary: _want_gamd_method reads
    # _OPT_IN_GAMD_METHODS from its own module's globals. If the constant
    # were left behind in analyze_gareus_mbar.py, this would raise
    # NameError instead of returning a bool.
    import gareus.mbar_analysis.plotting as plotting

    assert plotting._OPT_IN_GAMD_METHODS == {
        "gamd_exponential": "plot_gamd_exponential",
        "gamd_cumulant3": "plot_gamd_cumulant3",
    }

    class _Args:
        plot_gamd_exponential = False

    assert plotting._want_gamd_method("gamd_exponential", "gamd_exponential", _Args()) is True
    assert plotting._want_gamd_method("gamd_exponential", "umbrella_only", _Args()) is False
    assert plotting._want_gamd_method("umbrella_only", "umbrella_only", _Args()) is True


def test_cv_label_helpers_unchanged_behavior():
    # The six CV-label helpers are pure meta-dict -> str/list formatters with
    # no I/O; a spot-check of each branch-selecting input is enough to prove
    # the relocation copied bodies rather than paraphrasing them.
    # _regime_slug additionally exercises the interim `from
    # analyze_gareus_mbar import _slug` local import (Plan A6d's symbol) --
    # if that import were placed at module level instead, importing
    # gareus.mbar_analysis.plotting at all would already have failed above.
    import gareus.mbar_analysis.plotting as plotting

    assert plotting._secondary_cv_label({"secondary_cv": "torsion-pca"}) == "Bootstrap torsion PC1"
    assert plotting._secondary_cv_label({"secondary_cv": {"mode": "tica-linear"}}) == "tIC1 torsion CV"
    assert plotting._secondary_cv_label({}) == "Secondary CV"

    assert plotting._secondary_cv_regions(
        {"secondary_cv": {"regions": [{"value": 1.5, "label": "alpha"}, {"name": "no-value"}]}}
    ) == [{"value": 1.5, "label": "alpha"}]
    assert plotting._secondary_cv_regions({"secondary_cv": "tica-linear"}) == []

    assert plotting._regime_slug("") == "unknown"
    # A non-empty regime must route through the interim
    # `from analyze_gareus_mbar import _slug` local import. Assert only that
    # the import resolves and yields a usable slug -- do NOT pin _slug's own
    # output format here, that belongs to Plan A6d, which owns _slug.
    slug = plotting._regime_slug("torsion-pca")
    assert isinstance(slug, str) and slug and slug != "unknown"

    assert plotting._primary_cv_label({"primary_cv": "nonlocal-contacts"}) == "nonlocal contact fraction"
    assert plotting._primary_cv_units({"primary_cv": "nonlocal-contacts"}) == "dimensionless"
    assert plotting._primary_cv_axis_label({"primary_cv": "nonlocal-contacts"}) == "nonlocal contact fraction"
    assert plotting._primary_cv_axis_label({}) == "CV distance (A)"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_mbar_analysis_plotting.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gareus.mbar_analysis.plotting'`.

- [ ] **Step 3: Extract and relocate**

Re-confirm every line range with `grep -n "^def _secondary_cv_label\|^def _secondary_cv_regions\|^def _regime_slug\|^def _primary_cv_label\|^def _primary_cv_units\|^def _primary_cv_axis_label\|^def _smooth_masked_grid\|^def _smooth_pmf_1d\|^def _eff_smooth\|^def _fes_range_tag\|^def _fes_range_label\|^def _fes_variant_path\|^def _plot_2d_fes_range\|^def _plot_2d_fes_multirange\|^def plot_2d_fes\|^def plot_cv1_cv2_2d_fes\|^def plot_pca_2d_fes\|^def _want_gamd_method\|^def _visible_pmfs\|^def _per_window_gamd_boost_stats\|^def plot_gamd_boost\|^def plot_outputs\|^def plot_rg_outputs\|^_OPT_IN_GAMD_METHODS" analyze_gareus_mbar.py` before proceeding — the last pattern (no leading `^def `) locates the module-level constant, which does not match a `def` grep. Do **not** include `_window_moments` in this grep or in the regions below — it stays behind (Plan A4's, per the CORRECTION in Global Constraints). Run the separate stay-behind-neighbor grep from the boundary note above as well, so the three CV-label regions' end boundaries are pinned by `_epoch_run_manifest_secondary_cv_type`/`run_secondary_cv_analyses`/`_poincare_primary_cv_supported` rather than guessed.

Read `analyze_gareus_mbar.py` lines 323-347, 510-512, 589-608, 3822-4002, 5665-5683, 6078-6085, 6386-6399, and 7119-7324 (eight separate `Read` calls with the confirmed offsets/limits — the first three are the CV-label cluster, each stopping one blank line short of a function that stays behind; the seventh region starts 2 lines earlier than a naive `def`-only scan would suggest, to include `_OPT_IN_GAMD_METHODS`; see the boundary notes above. The eighth region starts at `_per_window_gamd_boost_stats` (L7119), skipping over `_window_moments`, L7104-7116, which stays behind). Concatenate their contents, in that order, beneath this header, and write the result to `gareus/mbar_analysis/plotting.py`:

```python
"""Matplotlib rendering + smoothing/label helpers for the MBAR/PMF analysis
report. Relocated verbatim from analyze_gareus_mbar.py (Plan A6a) -- no
logic changes, only module location.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np

from gareus.mbar_analysis import plotstyle as ps

```

Two exceptions to "nothing beyond this header":

1. `_regime_slug` calls `_slug(regime)` (defined at L6165-6177, inside Plan A6d's own future "residue/topology observables" scope — A6d executes *after* this plan; see Global Constraints).
2. `_per_window_gamd_boost_stats` calls `_window_moments(seg_comb)` — per the CORRECTION in Global Constraints, `_window_moments` is claimed by Plan A4 for `gareus/mbar_analysis/pmf.py`, not relocated here.

`_primary_cv_axis_label` is **not** on this list any more: per the six-CV-label-helper CORRECTION in Global Constraints, it (together with `_primary_cv_label` and `_primary_cv_units`, which it calls) is relocated by this very task, so `plot_outputs`/`plot_rg_outputs` resolve it in `plotting.py`'s own module globals with no import statement at all. Do not add the `from analyze_gareus_mbar import _primary_cv_axis_label` line an earlier draft of this step specified — it would shadow the local definition with a re-exported alias of the same object, which is harmless but misleading, and would break outright once A6e retires the script.

Add a **local, function-body import** for each of the two remaining exceptions, matching this file's own existing lazy-import convention (`plot_outputs`/`plot_rg_outputs` already do `import matplotlib.pyplot as plt` the same way) — do **not** add either as a module-level import at the top of `plotting.py`: `analyze_gareus_mbar.py` itself imports *from* `gareus.mbar_analysis.plotting` (the re-export blocks below), so a module-level `from analyze_gareus_mbar import _slug` would be a circular import evaluated at load time, not call time; `_window_moments` isn't guaranteed to exist at `gareus.mbar_analysis.pmf` yet either (depends on whether A4 has landed), so a plain module-level import could raise `ImportError` at `plotting.py`'s own import time regardless of whether `_per_window_gamd_boost_stats` is ever called.

```python
def _regime_slug(regime: str) -> str:
    if not regime:
        return 'unknown'
    from analyze_gareus_mbar import _slug
    return _slug(regime)
```
for `_regime_slug` — same interim import Task 3 adds inside `_write_rama_2d`, for the same reason (Plan A6d must update both to import from wherever it relocates `_slug` to). **This is the one relocated body in this plan whose *structure* changes, not just its imports**, and the exact replacement text is given above rather than left to the executor: the original is a single-expression `return _slug(regime) if regime else 'unknown'`, which has no line "immediately before its first use of `_slug`" to hang a local import on. Putting the import at the top of the one-line body instead would run it on the `regime == ''` path too, which never touches `_slug` — a new, unnecessary failure mode for that branch once A6e retires the script. The early-return form above is semantically identical to the original for every input (empty/falsy → `'unknown'`, otherwise `_slug(regime)`) and keeps the import strictly on the path that needs it. And:
```python
    try:
        from gareus.mbar_analysis.pmf import _window_moments
    except ImportError:
        from analyze_gareus_mbar import _window_moments
```
as the first lines inside `_per_window_gamd_boost_stats`'s body (falls back to the not-yet-relocated script copy if Plan A4 hasn't landed `pmf.py` yet at this plan's execution time; once A4 has landed, re-verify at Task 7 that the `try` branch is the one actually taken, and simplify to a plain `from gareus.mbar_analysis.pmf import _window_moments` if so — do not leave the fallback in permanently once A4 is confirmed present).

Both are temporary/defensive interim imports — relocating `_slug` is explicitly Plan A6d's job, and relocating `_window_moments` is explicitly Plan A4's job, already done in its own spec. Every other name in the eight extracted regions is already covered by `math`/`Path`/`Optional`/`np`/`ps`/builtins (the six CV-label helpers themselves need nothing beyond builtins, `isinstance`, and each other); if any other name looks unresolved, that is a signal the region boundary was mis-copied — re-check against the design spec's exact line numbers rather than adding an import to paper over it.

In `analyze_gareus_mbar.py`: delete the eight source regions (323-347, 510-512, 589-608, 3822-4002, 5665-5683, 6078-6085, 6386-6399, 7119-7324) entirely, working from the bottom of the file upward (7119-7324 first, then 6386-6399, then 6078-6085, then 5665-5683, then 3822-4002, then 589-608, then 510-512, then 323-347) so earlier deletions don't shift the line numbers of later ones you still need to delete. **Do not delete L7104-7116 (`_window_moments`)** — it sits immediately above the 7119-7324 region and stays behind for Plan A4. **Do not delete L349 (`_epoch_run_manifest_secondary_cv_type`), L514 (`run_secondary_cv_analyses`), or L609 (`_poincare_primary_cv_supported`)** — each sits immediately below one of the three CV-label regions and stays behind (Plans A2, A4, and A6d respectively; see the boundary note above).

Two re-export blocks, each placed where the earliest deleted region of its cluster used to start — matching this plan's own "re-exports placed where the original definitions lived" convention. In place of the region that started at **L323**, insert:

```python
from gareus.mbar_analysis.plotting import (
    _secondary_cv_label, _secondary_cv_regions, _regime_slug,
    _primary_cv_label, _primary_cv_units, _primary_cv_axis_label,
)
```

and in place of the region that started at **L3822**, insert:

```python
from gareus.mbar_analysis.plotting import (
    _smooth_masked_grid, _smooth_pmf_1d, _eff_smooth, FES_PLOT_VMAX_VALUES,
    _fes_range_tag, _fes_range_label, _fes_variant_path, _plot_2d_fes_range,
    _plot_2d_fes_multirange, plot_2d_fes, plot_cv1_cv2_2d_fes, plot_pca_2d_fes,
    _per_window_gamd_boost_stats, plot_gamd_boost, plot_outputs,
    plot_rg_outputs, _OPT_IN_GAMD_METHODS, _want_gamd_method, _visible_pmfs,
)
```

The other six deleted regions are simply removed with nothing left in their place (the two imports above already re-export everything). Two blocks rather than one merged block is deliberate: it keeps each re-export where its definition lived, which is this plan's stated convention and what makes the diff reviewable. (Either placement would *work* — Python resolves a function body's global names at call time, not at def time, so `run_secondary_cv_analyses`'s L514 call to `_regime_slug` would still find a name bound anywhere at module level. Correctness is not the reason for the split; locality is.) Both are plain module-level imports of the same already-imported module, so there is no extra import cost.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_mbar_analysis_plotting.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Confirm nothing else broke**

Run: `python -m py_compile gareus/mbar_analysis/plotting.py analyze_gareus_mbar.py`
Expected: clean compile.

Run: `pytest -q tests/test_perf_plotting_redundancy.py -v` (do **not** fix the monkeypatch hazard yet — that's Task 6; this run is expected to show `test_plot_2d_fes_multirange_renders_each_range_exactly_once` newly FAILING, and every other test in the file still passing — confirms the hazard is real and precisely scoped before you fix it deliberately in Task 6, rather than accidentally masking it here.)

- [ ] **Step 6: Commit**

```bash
git add gareus/mbar_analysis/plotting.py tests/test_mbar_analysis_plotting.py analyze_gareus_mbar.py
git commit -m "refactor: relocate plotting/rendering functions into gareus.mbar_analysis.plotting"
```

---

### Task 3: Create `gareus/mbar_analysis/writers.py`

**Files:**
- Create: `gareus/mbar_analysis/writers.py` (~265 lines)
- Modify: `analyze_gareus_mbar.py` (delete 15 function definitions, add 1 re-export block)
- Test: `tests/test_mbar_analysis_writers.py` (new file)

**Interfaces:**
- Consumes: `gareus.mbar_analysis.plotting.{_smooth_pmf_1d, _plot_2d_fes_multirange, _want_gamd_method}` (Task 2).
- Produces: `gareus.mbar_analysis.writers.{write_2d_fes_csv, write_2d_fes_npz, write_cv1_cv2_2d_fes_csv, write_cv1_cv2_2d_fes_npz, write_pca_2d_fes_csv, write_pca_2d_fes_npz, _write_scalar_pmfs, _write_generic_2d_fes, _write_rama_2d, write_rg_pmf, write_rg_all, write_pmf, write_all, write_cv2_pmf, _write_csv_rows}` — the last of these, `_write_csv_rows`, is consumed by Plans A6b/A6c/A6d (not yet written), each of which must import it from here rather than from `analyze_gareus_mbar`.

**Exact line ranges** (re-verify before extracting):

```
write_2d_fes_csv          L3757-3779 (23)
write_2d_fes_npz          L3780-3793 (14)
write_cv1_cv2_2d_fes_csv  L3794-3807 (14)
write_cv1_cv2_2d_fes_npz  L3808-3821 (14)
write_pca_2d_fes_csv      L6060-6073 (14)
write_pca_2d_fes_npz      L6074-6077 ( 4)
_write_scalar_pmfs        L6407-6440 (34)
_write_generic_2d_fes     L6441-6483 (43)
_write_rama_2d            L6484-6527 (44)
write_rg_pmf              L5648-5657 (10)
write_rg_all              L5658-5664 ( 7)
write_pmf                 L7067-7076 (10)
write_all                 L7077-7083 ( 7)
write_cv2_pmf             L7084-7093 (10)
_write_csv_rows           L4440-4456 (17)
```

Five contiguous source regions: `3757-3821`, `4440-4456`, `5648-5664`, `6060-6077`, `6407-6527`, `7067-7093` — six total: `3757-3821`, `4440-4456`, `5648-5664`, `6060-6077`, `6407-6527`, `7067-7093`.

**Why `_write_csv_rows` belongs here, not with convergence (Task 4) or a later plan**: it is a generic list-of-dicts-to-CSV writer with no FES/PMF-shape assumptions and no call-site dependency on anything else in this file. Direct verification (`grep -n "_write_csv_rows(" analyze_gareus_mbar.py`) shows **15 call sites spanning three domains that all execute after this plan** — 10 inside Plan A6b's two convergence drivers (`run_epoch_pmf_convergence`, `run_observable_pmf_convergence`), 1 inside Plan A6c's `analyze_rg`, and 4 across Plan A6d's `analyze_extra_observable_pmfs`/`analyze_poincare_map`/`analyze_poincare_residue_torsions`. Leaf-first ordering requires it live in whichever plan lands before *all* of its consumers — that is this plan (A6a), not A6b (which is itself one of the consumers).

**Important note on execution order relative to Task 2**: `6407-6527` (which contains `_write_scalar_pmfs`/`_write_generic_2d_fes`/`_write_rama_2d`) sits physically *after* the L6388-6399 region Task 2 already deleted (`_want_gamd_method`/`_visible_pmfs`) and *before* nothing else Task 2 touched in that neighborhood; `4440-4456` sits well before any Task 2 deletion. Task 2's deletions do not overlap this task's regions, but re-confirm all six ranges with a fresh `grep -n "^def "` pass regardless, since Task 2's deletions shifted every line number after L3822 downward by the total size of what it removed.

- [ ] **Step 1: Write the failing test**

Create `tests/test_mbar_analysis_writers.py`:

```python
import csv
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import analyze_gareus_mbar as agm

_NAMES = [
    "write_2d_fes_csv", "write_2d_fes_npz", "write_cv1_cv2_2d_fes_csv",
    "write_cv1_cv2_2d_fes_npz", "write_pca_2d_fes_csv", "write_pca_2d_fes_npz",
    "_write_scalar_pmfs", "_write_generic_2d_fes", "_write_rama_2d",
    "write_rg_pmf", "write_rg_all", "write_pmf", "write_all", "write_cv2_pmf",
    "_write_csv_rows",
]


def test_writers_module_is_importable():
    import gareus.mbar_analysis.writers  # noqa: F401


def test_every_relocated_name_is_the_same_object_via_reexport():
    import gareus.mbar_analysis.writers as writers

    for name in _NAMES:
        assert hasattr(writers, name), f"writers.py missing {name}"
        assert getattr(agm, name) is getattr(writers, name), (
            f"{name}: re-export is broken"
        )


def test_writers_imports_from_plotting_not_from_agm():
    import gareus.mbar_analysis.writers as writers
    import gareus.mbar_analysis.plotting as plotting

    assert writers._smooth_pmf_1d is plotting._smooth_pmf_1d
    assert writers._plot_2d_fes_multirange is plotting._plot_2d_fes_multirange
    assert writers._want_gamd_method is plotting._want_gamd_method


def test_write_pmf_produces_expected_csv_shape():
    import gareus.mbar_analysis.writers as writers
    import numpy as np

    pmf = {
        "cv_A": np.array([1.0, 2.0]),
        "prob": np.array([0.5, 0.5]),
        "pmf": np.array([0.0, 0.1]),
        "counts": np.array([10, 20]),
    }
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "out.csv"
        writers.write_pmf(p, pmf, "umbrella_only")
        rows = list(csv.DictReader(p.open()))
    assert len(rows) == 2
    assert rows[0]["method"] == "umbrella_only"
    assert rows[1]["cv_A"] == "2.0"


def test_write_csv_rows_handles_heterogeneous_and_empty_input():
    import gareus.mbar_analysis.writers as writers

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "rows.csv"
        writers._write_csv_rows(p, [{"a": 1, "b": 2}, {"a": 3, "c": 4}])
        rows = list(csv.DictReader(p.open()))
        assert [r["a"] for r in rows] == ["1", "3"]
        assert rows[0]["b"] == "2"

        p2 = Path(td) / "empty.csv"
        writers._write_csv_rows(p2, [])
        assert p2.read_text() == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_mbar_analysis_writers.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gareus.mbar_analysis.writers'`.

- [ ] **Step 3: Extract and relocate**

Re-confirm all six ranges with fresh greps (per the note above — `_write_csv_rows` is a `def`, so it's found the same way as the rest: `grep -n "^def _write_csv_rows"`). Read the six regions (`3757-3821`, `4440-4456`, `5648-5664`, `6060-6077`, `6407-6527`, `7067-7093`) in order and write to `gareus/mbar_analysis/writers.py` beneath:

```python
"""CSV/NPZ serialization of already-built PMF/FES dicts, plus the generic
_write_csv_rows list-of-dicts writer used by Plans A6b/A6c/A6d. Relocated
verbatim from analyze_gareus_mbar.py (Plan A6a) -- no logic changes, only
module location.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

import numpy as np

from gareus.mbar_analysis.plotting import (
    _smooth_pmf_1d, _plot_2d_fes_multirange, _want_gamd_method,
)

```

One exception to "nothing beyond this header": `_write_rama_2d` calls `_slug(residue_label)` (defined at L6165-6177, inside Plan A6d's own future "residue/topology observables" scope — A6d executes *after* this plan). Add a **local, function-body import** inside `_write_rama_2d`, immediately before its first use of `_slug` (this file has no existing lazy-import convention of its own since it does no matplotlib work directly, but the same local-import technique applies for the same circular-import reason as Task 2's own `_slug` note — `_regime_slug` in `plotting.py` needs the identical import, and Plan A6d must update both call sites together):

```python
    from analyze_gareus_mbar import _slug
```

This is a temporary interim import — Plan A6d must update it to import from wherever it relocates `_slug` to. Every other name in the six extracted regions is already covered by `csv`/`Path`/`Optional`/`np`/the three `plotting` imports/builtins.

In `analyze_gareus_mbar.py`, delete the six regions bottom-to-top (`7067-7093`, `6407-6527`, `6060-6077`, `5648-5664`, `4440-4456`, `3757-3821`), and insert, in place of the first deleted region (`3757-3821`):

```python
from gareus.mbar_analysis.writers import (
    write_2d_fes_csv, write_2d_fes_npz, write_cv1_cv2_2d_fes_csv, write_cv1_cv2_2d_fes_npz,
    write_pca_2d_fes_csv, write_pca_2d_fes_npz, _write_scalar_pmfs, _write_generic_2d_fes,
    _write_rama_2d, write_rg_pmf, write_rg_all, write_pmf, write_all, write_cv2_pmf,
    _write_csv_rows,
)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_mbar_analysis_writers.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Confirm nothing else broke**

Run: `python -m py_compile gareus/mbar_analysis/writers.py analyze_gareus_mbar.py`

- [ ] **Step 6: Commit**

```bash
git add gareus/mbar_analysis/writers.py tests/test_mbar_analysis_writers.py analyze_gareus_mbar.py
git commit -m "refactor: relocate CSV/NPZ writer functions into gareus.mbar_analysis.writers"
```

---

### Task 4: Create `gareus/mbar_analysis/convergence_reporting.py`

**Files:**
- Create: `gareus/mbar_analysis/convergence_reporting.py` (~390 lines)
- Modify: `analyze_gareus_mbar.py` (delete 7 function definitions + 1 dead function, add 1 re-export block)
- Test: `tests/test_mbar_analysis_convergence_reporting.py` (new file)

**Interfaces:**
- Consumes: `gareus.mbar_analysis.plotting._smooth_pmf_1d` (Task 2).
- Produces: `gareus.mbar_analysis.convergence_reporting.{_add_epoch_annotations_to_axes, _write_epoch_ess_plot, write_convergence_plots, _add_aggregate_ns_secondary_axis, write_observable_convergence_plots, _plot_basin_population_convergence, write_observable_convergence_report}` — consumed by Plan A6b (not yet written; A6b's own spec will add the necessary imports once it exists).

**Exact line ranges** (re-verify before extracting; these have shifted downward from their originally-observed positions by whatever Tasks 2-3 removed above them — recompute with `grep -n` before touching this task):

```
_add_epoch_annotations_to_axes       L4282-4297 (16, at commit 024cea7 -- BEFORE Tasks 2/3's deletions shift it)
_write_epoch_ess_plot                 L4382-4425 (44)
write_convergence_plots               L4457-4575 (119)
[write_convergence_report — DEAD, DELETE — L4576-4604]
_add_aggregate_ns_secondary_axis      L4353-4381 (29)
write_observable_convergence_plots    L4631-4745 (115)
_plot_basin_population_convergence    L4746-4783 (38)
write_observable_convergence_report   L4881-4909 (29)
```

At the pre-Task-2/3 line numbers above, this is one contiguous region, `4282-4909`, **except** it also contains `checkpoint_steps_from_data` (L4426-4456, Plan A6b's function, not this plan's), `_get_convergence_mbar_cache`/`_convergence_mask_digest`/`_convergence_mbar_cache_key` (L4784-4834, also A6b's), and `_get_checkpoint_steps_cache`/`_checkpoint_steps_cache_key` (L4835-4880, also A6b's) interleaved between this plan's own functions. **Do not delete or move those five A6b-owned functions** — they stay in `analyze_gareus_mbar.py` untouched, to be relocated by Plan A6b later. This means Task 4's extraction is *not* a single contiguous cut; it is eight separate single-function extractions from within the `4282-4909` span, each individually confirmed by its own `grep -n "^def <name>"` immediately beforehand.

- [ ] **Step 1: Write the failing test**

Create `tests/test_mbar_analysis_convergence_reporting.py`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import analyze_gareus_mbar as agm

_NAMES = [
    "_add_epoch_annotations_to_axes", "_write_epoch_ess_plot", "write_convergence_plots",
    "_add_aggregate_ns_secondary_axis", "write_observable_convergence_plots",
    "_plot_basin_population_convergence", "write_observable_convergence_report",
]


def test_convergence_reporting_module_is_importable():
    import gareus.mbar_analysis.convergence_reporting  # noqa: F401


def test_every_relocated_name_is_the_same_object_via_reexport():
    import gareus.mbar_analysis.convergence_reporting as cr

    for name in _NAMES:
        assert hasattr(cr, name), f"convergence_reporting.py missing {name}"
        assert getattr(agm, name) is getattr(cr, name), f"{name}: re-export is broken"


def test_write_convergence_report_is_fully_gone():
    assert not hasattr(agm, "write_convergence_report")
    import gareus.mbar_analysis.convergence_reporting as cr
    assert not hasattr(cr, "write_convergence_report")


def test_convergence_reporting_functions_a6b_still_owns_stayed_behind():
    # checkpoint_steps_from_data and its cache-key siblings are Plan A6b's,
    # not this plan's -- confirm they were NOT accidentally swept up here.
    assert hasattr(agm, "checkpoint_steps_from_data")
    assert hasattr(agm, "_get_convergence_mbar_cache")
    assert hasattr(agm, "_get_checkpoint_steps_cache")
    import gareus.mbar_analysis.convergence_reporting as cr
    assert not hasattr(cr, "checkpoint_steps_from_data")
    assert not hasattr(cr, "_get_convergence_mbar_cache")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_mbar_analysis_convergence_reporting.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gareus.mbar_analysis.convergence_reporting'`.

- [ ] **Step 3: Extract and relocate**

Run fresh greps to re-locate all eight functions (`_add_epoch_annotations_to_axes`, `_write_epoch_ess_plot`, `write_convergence_plots`, `write_convergence_report`, `_add_aggregate_ns_secondary_axis`, `write_observable_convergence_plots`, `_plot_basin_population_convergence`, `write_observable_convergence_report`) individually — do not assume the L4282-4909 numbers above still hold after Tasks 2-3's deletions.

Read each of the seven **non-dead** functions' bodies individually (in their current file order — `_add_epoch_annotations_to_axes`, `_add_aggregate_ns_secondary_axis`, `_write_epoch_ess_plot`, `write_convergence_plots`, `write_observable_convergence_plots`, `_plot_basin_population_convergence`, `write_observable_convergence_report`, reordered here to put the two Axes-drawing helpers first since the design spec's function-to-file map lists them before their consumers) and concatenate beneath:

```python
"""Convergence-sweep figure and report writers. Relocated verbatim from
analyze_gareus_mbar.py (Plan A6a) -- no logic changes, only module
location. write_convergence_report (the dead CV1-legacy sibling of
write_observable_convergence_report) was deleted, not relocated -- it had
zero callers anywhere in the repo.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from gareus.mbar_analysis.plotting import _smooth_pmf_1d

```

In `analyze_gareus_mbar.py`: delete each of the eight functions' bodies (the seven relocated ones, plus the dead `write_convergence_report`) working from the bottom of the file upward by line number. In place of whichever of the seven relocated functions appears **first** in the file (by original line number, before any deletions), insert:

```python
from gareus.mbar_analysis.convergence_reporting import (
    _add_epoch_annotations_to_axes, _write_epoch_ess_plot, write_convergence_plots,
    _add_aggregate_ns_secondary_axis, write_observable_convergence_plots,
    _plot_basin_population_convergence, write_observable_convergence_report,
)
```

`write_convergence_report`'s deletion site gets nothing in its place (it is gone, not re-exported — confirmed dead, per Global Constraints).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_mbar_analysis_convergence_reporting.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Confirm nothing else broke**

Run: `python -m py_compile gareus/mbar_analysis/convergence_reporting.py analyze_gareus_mbar.py`

Run: `grep -rn write_convergence_report .` (repo-wide, excluding this plan's own new test file's negative assertion)
Expected: zero matches outside `tests/test_mbar_analysis_convergence_reporting.py`.

- [ ] **Step 6: Commit**

```bash
git add gareus/mbar_analysis/convergence_reporting.py tests/test_mbar_analysis_convergence_reporting.py analyze_gareus_mbar.py
git commit -m "refactor: relocate convergence report/plot writers into gareus.mbar_analysis.convergence_reporting; delete dead write_convergence_report"
```

---

### Task 5: Create `gareus/mbar_analysis/summary.py`

**Files:**
- Create: `gareus/mbar_analysis/summary.py` (~85 lines)
- Modify: `analyze_gareus_mbar.py` (delete 3 function definitions, add 1 re-export block)
- Test: `tests/test_mbar_analysis_summary.py` (new file)

**Interfaces:**
- Consumes: nothing from the other four new files (pure dict-to-markdown renderer, per the design spec).
- Produces: `gareus.mbar_analysis.summary.{_render_health_section_md, _key_diagnostics_md, summary_md}` — consumed by `analyze()` (Plan A6e, not yet written).

**Exact line ranges** (re-verify before extracting; this is the last A6a task, so re-grep against whatever Tasks 2-4 have shifted):

```
_render_health_section_md     (originally L7325-7333,  9 lines)
_key_diagnostics_md           (originally L7334-7368, 35 lines)
summary_md                    (originally L7369-7409, 41 lines)
```

These three are contiguous in the original file (`7325-7409`) and physically adjacent to `plot_outputs`/`plot_gamd_boost` (already relocated in Task 2) — re-confirm the exact current range with `grep -n "^def _render_health_section_md\|^def _key_diagnostics_md\|^def summary_md"` before extracting; read exactly the body actually returned, not the ranges above.

- [ ] **Step 1: Write the failing test**

Create `tests/test_mbar_analysis_summary.py`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import analyze_gareus_mbar as agm

_NAMES = ["_render_health_section_md", "_key_diagnostics_md", "summary_md"]


def test_summary_module_is_importable():
    import gareus.mbar_analysis.summary  # noqa: F401


def test_every_relocated_name_is_the_same_object_via_reexport():
    import gareus.mbar_analysis.summary as summary

    for name in _NAMES:
        assert hasattr(summary, name), f"summary.py missing {name}"
        assert getattr(agm, name) is getattr(summary, name), f"{name}: re-export is broken"


def test_summary_module_has_no_cross_imports_from_sibling_a6a_modules():
    import ast

    src = Path("gareus/mbar_analysis/summary.py").read_text()
    tree = ast.parse(src)
    imported_modules = {
        n.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
        for n in [node]
    }
    assert "gareus.mbar_analysis.plotting" not in imported_modules
    assert "gareus.mbar_analysis.writers" not in imported_modules
    assert "gareus.mbar_analysis.convergence_reporting" not in imported_modules
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_mbar_analysis_summary.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gareus.mbar_analysis.summary'`.

- [ ] **Step 3: Extract and relocate**

Re-confirm the range with fresh greps, read the three functions' bodies (contiguous), and write to `gareus/mbar_analysis/summary.py` beneath a header built by inspecting exactly what the three bodies reference (per the design spec: "confirm exact set by reading the three function bodies at implementation time — they were not fully transcribed into this spec"). At minimum expect to need:

```python
"""Markdown rendering of the already-fully-assembled `s` summary dict
(pmf_summary.md). Relocated verbatim from analyze_gareus_mbar.py (Plan
A6a) -- no logic changes, only module location.
"""
from __future__ import annotations

```

then append whatever additional stdlib imports the three function bodies actually turn out to need (check for `math`/`json` usage by reading the extracted text itself before finalizing the header — do not guess).

In `analyze_gareus_mbar.py`: delete the three functions' bodies, and in their place insert:

```python
from gareus.mbar_analysis.summary import _render_health_section_md, _key_diagnostics_md, summary_md
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_mbar_analysis_summary.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Confirm nothing else broke**

Run: `python -m py_compile gareus/mbar_analysis/summary.py analyze_gareus_mbar.py`

- [ ] **Step 6: Commit**

```bash
git add gareus/mbar_analysis/summary.py tests/test_mbar_analysis_summary.py analyze_gareus_mbar.py
git commit -m "refactor: relocate pmf_summary.md renderers into gareus.mbar_analysis.summary"
```

---

### Task 6: Fix the monkeypatch-vs-shim hazard in `tests/test_perf_plotting_redundancy.py`

**Files:**
- Modify: `tests/test_perf_plotting_redundancy.py` (repoint one test's monkeypatch target and call site)

**Interfaces:**
- Consumes: `gareus.mbar_analysis.plotting.{_plot_2d_fes_range, _plot_2d_fes_multirange, FES_PLOT_VMAX_VALUES}` (Task 2).

**Why this task exists:** `test_plot_2d_fes_multirange_renders_each_range_exactly_once` does `monkeypatch.setattr(agm, "_plot_2d_fes_range", counting)` then calls `agm._plot_2d_fes_multirange(...)`. Before this plan, both names lived in `analyze_gareus_mbar`'s own namespace, so the patch worked. After Task 2, both live in `gareus.mbar_analysis.plotting`; `_plot_2d_fes_multirange`'s internal call to `_plot_2d_fes_range` resolves in `plotting.py`'s own `__globals__`, not `analyze_gareus_mbar`'s re-exported binding. Patching `agm._plot_2d_fes_range` no longer has any effect on what the real call actually invokes — the test would keep passing for the wrong reason (`calls` stays empty, but nothing asserts `calls` is non-empty before comparing it to `FES_PLOT_VMAX_VALUES`, so this fails loudly as a list-mismatch, not silently — confirmed by Task 2 Step 5's deliberate before-the-fix run).

`test_plot_chignolin_fes_kj_multirange_renders_each_range_exactly_once` (the other monkeypatch test in this file) is **not** affected — `_plot_chignolin_fes_kj_range`/`_plot_chignolin_fes_kj_multirange`/`CHIGNOLIN_FES_PLOT_VMAX_VALUES_KJ` are explicitly out of this plan's scope (deferred to Plan A6c) and still live together in `analyze_gareus_mbar.py`'s own namespace — leave that test completely untouched.

- [ ] **Step 1: Confirm the currently-failing test**

Run: `pytest tests/test_perf_plotting_redundancy.py::test_plot_2d_fes_multirange_renders_each_range_exactly_once -v`
Expected: FAIL (per Task 2 Step 5's prediction) — `assert [] == [2.0, 5.0, 10.0, 20.0, None]` or similar.

- [ ] **Step 2: Fix the test**

In `tests/test_perf_plotting_redundancy.py`, replace the function body of `test_plot_2d_fes_multirange_renders_each_range_exactly_once` (currently at approximately L286-308 — re-confirm with `grep -n "def test_plot_2d_fes_multirange_renders_each_range_exactly_once"`):

```python
def test_plot_2d_fes_multirange_renders_each_range_exactly_once(monkeypatch):
    """SHA256 equality alone can't fail if the second full render is
    reinstated (it would just re-produce the same bytes, slower). The
    regression this fix actually targets is the *render count*: assert
    `_plot_2d_fes_range` is called exactly once per FES_PLOT_VMAX_VALUES
    entry, with no extra call to (re-)produce the 'main' file.

    Patches and calls through `gareus.mbar_analysis.plotting` directly, not
    `analyze_gareus_mbar`'s re-export -- `_plot_2d_fes_multirange` resolves
    `_plot_2d_fes_range` in its own module's globals (Plan A6a relocated
    both), so patching the re-exported `analyze_gareus_mbar` name would
    silently patch a different binding than the one the real call site
    reads.
    """
    import gareus.mbar_analysis.plotting as plotting

    F, xedges, yedges, xc, yc = _synthetic_2d_grid()
    real = plotting._plot_2d_fes_range
    calls = []

    def counting(*a, **kw):
        calls.append(kw.get("range_vmax"))
        return real(*a, **kw)

    monkeypatch.setattr(plotting, "_plot_2d_fes_range", counting)
    warnings = []
    with tempfile.TemporaryDirectory() as td:
        out_png = Path(td) / "test_fes.png"
        files = plotting._plot_2d_fes_multirange(F, xedges, yedges, xc, yc, out_png, "Test FES", "x", "y", warnings)
        assert "main" in files
    assert calls == list(plotting.FES_PLOT_VMAX_VALUES)
```

Do not modify `test_plot_chignolin_fes_kj_multirange_renders_each_range_exactly_once`, the module's top-level `from analyze_gareus_mbar import (...)` block, or any other test in this file.

- [ ] **Step 3: Run test to verify it passes**

Run: `pytest tests/test_perf_plotting_redundancy.py -v`
Expected: PASS, all tests in the file (both monkeypatch tests, plus the SHA256-identity tests) — same pass count as before Task 2 ran.

- [ ] **Step 4: Commit**

```bash
git add tests/test_perf_plotting_redundancy.py
git commit -m "fix: repoint _plot_2d_fes_multirange monkeypatch test at its new gareus.mbar_analysis.plotting home"
```

---

### Task 7: Full regression verification

**Files:** none modified — verification only.

- [ ] **Step 1: Compile every touched file**

Run:
```bash
python -m py_compile gareus/mbar_analysis/plotstyle.py gareus/mbar_analysis/plotting.py \
    gareus/mbar_analysis/writers.py gareus/mbar_analysis/convergence_reporting.py \
    gareus/mbar_analysis/summary.py analyze_gareus_mbar.py
```
Expected: clean compile, no output.

- [ ] **Step 2: Run this plan's own new tests together**

Run:
```bash
pytest -q tests/test_mbar_analysis_plotstyle.py tests/test_mbar_analysis_plotting.py \
    tests/test_mbar_analysis_writers.py tests/test_mbar_analysis_convergence_reporting.py \
    tests/test_mbar_analysis_summary.py tests/test_gareus_plotstyle.py \
    tests/test_perf_plotting_redundancy.py -v
```
Expected: all PASS.

- [ ] **Step 3: Full existing suite, before/after diff**

Run: `pytest -q tests/ --ignore=tests/test_validation_common.py 2>&1 | tail -40`

Compare the pass/fail/error set against a run of the same command from before Task 1 (capture that baseline before starting this plan, if not already captured earlier this session). Expected: identical set of pre-existing failures (the same known-unrelated ones this repo's other A-series plans document — help-text encyclopedia numbering, threadpool-import check, missing validation launcher fixtures), zero new failures, zero newly-skipped tests. If anything outside that known set fails, stop and investigate before proceeding — per this plan's Global Constraints, no numeric/behavioral test should be able to detect this plan's changes at all.

- [ ] **Step 4: Final structural checks**

Run: `test ! -f gareus_plotstyle.py && echo "OK: old file gone"`
Run: `grep -rn "import gareus_plotstyle" . ; echo "exit code: $?"` — expect exit code 1 (no matches).
Run: `grep -rn write_convergence_report . ; echo "exit code: $?"` — expect exit code 1 (no matches).
Run: `wc -l gareus/mbar_analysis/plotstyle.py gareus/mbar_analysis/plotting.py gareus/mbar_analysis/writers.py gareus/mbar_analysis/convergence_reporting.py gareus/mbar_analysis/summary.py` — expect every file under 800 lines (design target: 128/474/248/390/85 — `plotting.py`'s target is 474, not the 426 an earlier draft used, because Task 2 also relocates the six CV-label helpers, +48 lines; see the CORRECTION in Global Constraints).

- [ ] **Step 5: Leave a note for Plan A4's own cross-check (informational only — not this plan's task to act on)**

Run: `grep -rn "write_pmf\|write_all\|plot_outputs\|_eff_smooth" gareus/mbar_analysis/*.py | grep -v "gareus/mbar_analysis/plotting.py\|gareus/mbar_analysis/writers.py"`

At the time this plan executes, this is expected to find nothing (Plan A4's `pmf.py`, which owns `run_pmf_and_gamd_boost_report` per its own design spec, does not exist as code yet — only as a design document). This command is recorded here so whoever implements Plan A4 can re-run it once `gareus/mbar_analysis/pmf.py` exists, to confirm `run_pmf_and_gamd_boost_report` reaches these four names' real final homes (`gareus.mbar_analysis.writers`/`gareus.mbar_analysis.plotting`) rather than resolving them through `analyze_gareus_mbar`'s re-export indefinitely.

**Note on A4's actual resolution mechanism**: A4's own design spec
(`2026-08-13-mbar-analysis-modularization-a4-design.md`) does not use a plain
`from gareus.mbar_analysis.writers import write_pmf`-style import for names
owned by plans that may not have executed yet — it resolves them at call
time through a `_bridge()` helper (`_bridge().write_pmf(...)`,
`_bridge().plot_outputs(...)`, etc.), specifically to avoid a real hazard: a
bare `import analyze_gareus_mbar` from inside a relocated module would create
a **second, independent copy** of the whole script when it's invoked via the
still-supported `python analyze_gareus_mbar.py <run_dir>` path (which runs it
as `__main__`, not as a module named `analyze_gareus_mbar` in `sys.modules`),
silently missing whatever `parse_args()` mutated on the real running copy's
globals. The grep above still works as a smoke test either way — literal
`_bridge().write_pmf(...)` calls still contain the substring `write_pmf`, so
they are not filtered out by the `grep -v` — but "confirm it imports X from Y"
in the sentence above is the wrong mental model for what a pass actually
looks like once A4 lands: a passing result may be `_bridge().write_pmf(...)`
calls inside `pmf.py`, not a static `from gareus.mbar_analysis.writers import
write_pmf` line, and that is equally acceptable. Whoever runs this check
should read what they find, not just its exit code, since `_bridge()`'s two
correct answers (a plain import once every dependency has landed, or a bridged
call while some haven't) look textually different from each other. This plan
does not need its own `_bridge()`-style resolver for the interim dependency
it has itself (`_slug`, imported locally in Task 2's `_regime_slug` and
Task 3's `_write_rama_2d`) because it does not depend on any
run-parameterized mutable module state the way `parse_args()`'s
MBAR-backend/SAMBAR globals do — a second, independently-imported copy would
return the same answer either way, so the plain `from analyze_gareus_mbar
import _slug` used there is safe as written and does not need the extra
machinery A4 needed for its own, state-sensitive bridge targets. (An earlier
draft listed `_primary_cv_axis_label` here as a second such dependency; it is
now relocated by Task 2 itself and needs no import — see the six-CV-label-helper
CORRECTION in Global Constraints.)

No commit for this task — it is verification-only, confirming Tasks 1-6 together satisfy this plan's Global Constraints.

---

## Plan Self-Review Notes

- **Spec coverage**: the original 41-function inventory, plus `_write_csv_rows` (correctly reassigned here from an earlier draft's A6b placement — see Global Constraints) and the `_OPT_IN_GAMD_METHODS` module constant (found while writing Task 2, since `_want_gamd_method`'s only reader for it would otherwise be stranded), plus the six CV-label helpers the cross-plan reconciliation pass found unowned and the coordinator assigned here (`_secondary_cv_label`, `_secondary_cv_regions`, `_regime_slug`, `_primary_cv_label`, `_primary_cv_units`, `_primary_cv_axis_label` — Task 2, see Global Constraints), plus `gareus_plotstyle.py`'s 12 exports, are covered across Tasks 1-5. `_window_moments` was caught mid-draft as a false addition — Plan A4's own design spec already explicitly claims it for `gareus/mbar_analysis/pmf.py` — and is deliberately *not* relocated here; Task 2 instead gives `_per_window_gamd_boost_stats` a defensive `try/except ImportError` local import so it works whether or not A4 has landed yet. The one deletion (`write_convergence_report`) and the one explicit exclusion (the chignolin-kJ trio, left to Plan A6c) are both called out, not silently handled.
- **Signature/interface consistency**: every task's "Interfaces" section states exactly what it consumes from an earlier task and produces for a later one; the cross-file import graph (`writers.py`/`convergence_reporting.py` → `plotting.py` → `plotstyle.py`) matches the design spec's stated dependency direction with no cycles, and the task order (1→2→3/4→5) respects it. Two cross-plan boundaries are handled by the same local-import pattern rather than silently assumed: `_window_moments` (Task 2, A4's) and `_slug` (Task 2's `_regime_slug` and Task 3's `_write_rama_2d`, both A6d's) — each documented with which plan is expected to eventually own the real definition. A third, `_primary_cv_axis_label`, was handled that way in an earlier draft but is now relocated into `plotting.py` by Task 2 itself, together with the five sibling CV-label helpers the reconciliation pass found equally unowned; see the CORRECTION in Global Constraints.
- **The one real behavioral risk in this plan** — the monkeypatch-vs-shim hazard — is not just noted but given its own task (6) with a concrete before/after test run proving the hazard is real (Task 2 Step 5) before proving the fix (Task 6 Step 3), rather than silently patched alongside the relocation.
- **No placeholders**: every task gives exact current-line-range data (with an explicit, repeated instruction to re-verify via `grep` before trusting it, since five sequential tasks each shift what came after them), exact header code, exact re-export blocks, and complete test code. The two places this plan cannot give a fully pre-written body (Task 5's summary.py header imports, and the exact byte contents of all 49 relocated function bodies) are handled by instructing extraction from the live source rather than hand-transcription — the correct way to relocate ~1,218 lines of already-audited, already-correct code without introducing transcription risk, and consistent with this plan's own "byte-identical" success bar. The single deliberate exception to byte-identity is `_regime_slug`, whose one-line body is restructured into an early return so its interim `_slug` import stays off the `regime == ''` path; Task 2 gives that replacement body verbatim rather than leaving it to the executor, and Task 2's own test pins both branches.
