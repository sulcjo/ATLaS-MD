# MBAR Analysis Modularization — Plan A4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Relocate `analyze_gareus_mbar.py`'s 24 PMF-construction/GaMD-cumulant/
bootstrap-uncertainty functions into `gareus/mbar_analysis/pmf.py`, byte-
faithfully, with `analyze_gareus_mbar.py` re-importing every relocated name
so its own ~40 other call sites (owned by Plans A2/A3/A5/A6, none of them
touched here) keep working unchanged.

**Architecture:** New file `gareus/mbar_analysis/pmf.py` gains a `_bridge()`
resolver (prefers an already-loaded `analyze_gareus_mbar` module under either
its real name or `__main__`, only falling back to a fresh `import` when
neither is loaded — a bare top-level `import analyze_gareus_mbar` is unsafe
here because the still-current `python analyze_gareus_mbar.py <run_dir>`
invocation loads the script as `__main__`, and a bare import would create a
second copy with its own never-`parse_args`-mutated MBAR/SAMBAR globals).
Every function that needs a name owned by a not-yet-executed plan
(`norm_logw`/`ess` → A3, `overlap_matrix` → A5, `_sample_block_ids` → A2,
`write_pmf`/`write_all`/`write_cv2_pmf`/`plot_outputs`/`analyze_cv1_cv2_2d_fes`/
`run_observable_pmf_convergence` → A6, `_masked_data`/`_subset_logw_from_global_fk`/
`_secondary_cv_epoch_regime_masks`/`_regime_slug`/`wjson` → A2, and the
currently-unowned `_eff_smooth`/`_smooth_pmf_1d`/`_secondary_cv_label`/
`_secondary_cv_regions`/`_visible_pmfs`) resolves it through `_bridge()`
rather than a bare global reference. The 24 functions move in four
dependency-ordered tasks (leaf histogram/cumulant math and the two bootstrap
helpers first — one contiguous, near-zero-bridge block — then boost/window
statistics, then the two orchestrators, heaviest-bridge last), each task
extracting via `sed -n` (never hand-retyped) and byte-verified against
`git show HEAD:...`.

**Tech Stack:** Python 3.10+, pytest, NumPy, `sed`/`grep`/`diff` for
mechanical extraction and verification.

**Spec:** `docs/superpowers/specs/2026-08-13-mbar-analysis-modularization-a4-design.md`

## Global Constraints

- All 24 relocated functions must be **byte-identical** to their current
  `analyze_gareus_mbar.py` bodies. The only permitted textual change inside
  any of them is replacing a bare reference to a name owned by a
  not-yet-executed plan with `_agm.<name>` after `_agm = _bridge()` — never a
  formula, guard, default value, or control-flow change. Every such
  substitution is listed explicitly in this plan; do not improvise others.
- Every task must re-confirm its stated line ranges via the given `grep -n`
  commands immediately before editing — line numbers in this plan reflect
  the repository state this plan was written against and WILL have shifted
  if Plan A1 (or any other plan) has already landed by the time this plan
  executes. This mirrors Plan A1's own stated convention.
- `analyze_gareus_mbar.py` must re-import every relocated name at module
  scope (growing one `from gareus.mbar_analysis.pmf import (...)` block
  across Tasks 1-4) so its own remaining call sites need zero changes. Do
  not touch any of those other call sites.
- No existing test file is modified. Every existing test that imports these
  functions from `analyze_gareus_mbar` must keep passing unchanged.
- `gareus/mbar_analysis/pmf.py` must be importable standalone
  (`python -c "import gareus.mbar_analysis.pmf"`) with zero eager
  dependency on `analyze_gareus_mbar.py` having been imported first — the
  `_bridge()` calls are all deferred to inside function bodies, never at
  module level.
- Placement of the growing re-export import in `analyze_gareus_mbar.py`:
  immediately after the `from gareus.units import KJ_PER_KCAL,
  K_B_KJ_PER_MOL_K` line that Plan A1 adds (re-confirm via
  `grep -n "from gareus.units import"`). If that line is not present
  (Plan A1 not yet executed in this checkout), add immediately after the
  `import numpy as np` line instead (`grep -n "^import numpy as np"`).

---

### Task 1: Relocate the histogram/cumulant math block (18 functions) + create `_bridge()`

**Files:**
- Create: `gareus/mbar_analysis/pmf.py`
- Modify: `analyze_gareus_mbar.py` (delete the block; add/start the re-export import)
- Test: `tests/test_mbar_analysis_pmf_module.py` (new file)

**Interfaces:**
- Produces: `gareus.mbar_analysis.pmf._bridge() -> Any`, and
  `gareus.mbar_analysis.pmf.{make_bins, _bin_indices, pmf_from_weights,
  _cumulant_shared_stats, _cumulant_from_shared, _cumulant_expansion,
  _cumulant_expansion_both, cumulant2, cumulant3, pmf2d_from_weights,
  _cumulant_shared_stats_2d, _cumulant_from_shared_2d, _cumulant_expansion_2d,
  _cumulant_expansion_2d_both, _bootstrap_pmf_uncertainty_1d,
  _bootstrap_pmf_uncertainty_2d, cumulant2_2d, cumulant3_2d}` — all with
  their current, unchanged signatures (see
  `docs/superpowers/specs/2026-08-13-mbar-analysis-modularization-a4-design.md`
  for the full architecture). Consumed by Tasks 2-4 (as local calls, no
  bridge needed) and by every other `analyze_*` call site in
  `analyze_gareus_mbar.py` (via the re-export import this task starts).

- [ ] **Step 1: Re-confirm current line boundaries**

Run:
```bash
grep -n "^def make_bins\|^def cumulant3_2d\|^def write_2d_fes_csv\|^def solve_mbar\b" analyze_gareus_mbar.py
```
Expected (as of this plan being written): `make_bins` at line 3290,
`cumulant3_2d` at line 3750, `write_2d_fes_csv` at line 3757, `solve_mbar` at
line 3161. If these differ, use the new `make_bins`-line and
`write_2d_fes_csv`-line to recompute the extraction range below (the block
runs from `make_bins`'s own line through the blank line immediately before
`write_2d_fes_csv`'s line — confirm no other `def` line falls inside that
range via `sed -n '<make_bins_line>,<write_2d_fes_csv_line>p' analyze_gareus_mbar.py | grep -n "^def "`, which must list exactly these 18 names in this order: `make_bins`, `_bin_indices`, `pmf_from_weights`, `_cumulant_shared_stats`, `_cumulant_from_shared`, `_cumulant_expansion`, `_cumulant_expansion_both`, `cumulant2`, `cumulant3`, `pmf2d_from_weights`, `_cumulant_shared_stats_2d`, `_cumulant_from_shared_2d`, `_cumulant_expansion_2d`, `_cumulant_expansion_2d_both`, `_bootstrap_pmf_uncertainty_1d`, `_bootstrap_pmf_uncertainty_2d`, `cumulant2_2d`, `cumulant3_2d`, plus `write_2d_fes_csv` itself as the 19th/final line of that grep output — that last one stays behind).

For the rest of this task, the confirmed range is **lines 3290-3755**
(`write_2d_fes_csv` starts at 3757, with one blank line at 3756).

- [ ] **Step 2: Write the failing test**

Create `tests/test_mbar_analysis_pmf_module.py`:

```python
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_pmf_module_is_importable_standalone():
    """Must import with ZERO dependency on analyze_gareus_mbar having been
    loaded first -- proves _bridge()'s deferred-import design (no eager
    cross-import at gareus/mbar_analysis/pmf.py's own module level)."""
    import gareus.mbar_analysis.pmf  # noqa: F401


def test_bridge_prefers_already_loaded_real_module(monkeypatch):
    import gareus.mbar_analysis.pmf as pmfmod
    sentinel = object()

    class _FakeModule:
        marker = sentinel

    monkeypatch.setitem(sys.modules, 'analyze_gareus_mbar', _FakeModule())
    assert pmfmod._bridge().marker is sentinel


def test_bridge_prefers_main_module_when_it_is_the_script(monkeypatch):
    import gareus.mbar_analysis.pmf as pmfmod
    monkeypatch.delitem(sys.modules, 'analyze_gareus_mbar', raising=False)

    class _FakeMain:
        __file__ = '/some/path/analyze_gareus_mbar.py'
        marker = 'main-copy'

    monkeypatch.setitem(sys.modules, '__main__', _FakeMain())
    assert pmfmod._bridge().marker == 'main-copy'


def test_bridge_falls_back_to_fresh_import_when_neither_present(monkeypatch):
    import gareus.mbar_analysis.pmf as pmfmod
    monkeypatch.delitem(sys.modules, 'analyze_gareus_mbar', raising=False)

    class _FakeMainUnrelated:
        __file__ = '/some/other/script.py'

    monkeypatch.setitem(sys.modules, '__main__', _FakeMainUnrelated())
    result = pmfmod._bridge()
    assert result.__name__ == 'analyze_gareus_mbar'


_TASK1_NAMES = [
    'make_bins', '_bin_indices', 'pmf_from_weights',
    '_cumulant_shared_stats', '_cumulant_from_shared', '_cumulant_expansion', '_cumulant_expansion_both',
    'cumulant2', 'cumulant3',
    'pmf2d_from_weights',
    '_cumulant_shared_stats_2d', '_cumulant_from_shared_2d', '_cumulant_expansion_2d', '_cumulant_expansion_2d_both',
    '_bootstrap_pmf_uncertainty_1d', '_bootstrap_pmf_uncertainty_2d',
    'cumulant2_2d', 'cumulant3_2d',
]


def test_task1_names_are_reexported_identically_by_analyze_gareus_mbar():
    import gareus.mbar_analysis.pmf as pmfmod
    import analyze_gareus_mbar as agm
    for name in _TASK1_NAMES:
        assert hasattr(pmfmod, name), f'{name} missing from gareus.mbar_analysis.pmf'
        assert getattr(agm, name) is getattr(pmfmod, name), (
            f'analyze_gareus_mbar.{name} is not the SAME object as '
            f'gareus.mbar_analysis.pmf.{name} -- re-export import did not replace the local def'
        )
```

Byte-identity of the extracted block is a one-time transcription-correctness
check, not an ongoing regression test (once this task's commit lands, `HEAD`
*is* the post-move file, so a test that keeps re-diffing against `HEAD`
would break permanently the moment this task's own commit exists — there is
no "old" location left to diff against). It belongs in Step 6 below as a
shell command run once during this task, not in the pytest file.

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_mbar_analysis_pmf_module.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gareus.mbar_analysis.pmf'`.

- [ ] **Step 4: Extract the block verbatim (mechanical, not hand-retyped)**

```bash
mkdir -p /tmp/a4_extract
git show HEAD:analyze_gareus_mbar.py | sed -n '3290,3755p' > /tmp/a4_extract/task1_block.txt
wc -l /tmp/a4_extract/task1_block.txt   # expect 466
```

- [ ] **Step 5: Write `gareus/mbar_analysis/pmf.py`**

Create the file with this exact header, then append the contents of
`/tmp/a4_extract/task1_block.txt` unchanged immediately after it (a plain
file-append, e.g. `cat /tmp/a4_extract/task1_block.txt >> gareus/mbar_analysis/pmf.py`
after writing the header below with a trailing newline):

```python
"""PMF construction: weighted histograms, GaMD cumulant expansion (2nd/3rd
order reweighting correction), boost-distribution statistics, and the
fixed-f_k block-bootstrap uncertainty machinery.

Relocated verbatim from analyze_gareus_mbar.py by Plan A4 of the
mbar-analysis-modularization sequence (see
docs/superpowers/specs/2026-08-13-mbar-analysis-modularization-a4-design.md).
This is a byte-faithful move of already-audited numerical code -- no
formula, guard, or default was changed. run_pmf_and_gamd_boost_report,
run_secondary_cv_analyses, and analyze_secondary_cv_pmf (added by this same
plan's later tasks) and their _bridge()-resolved dependencies on the
not-yet-migrated remainder of analyze_gareus_mbar.py are appended below this
header by those tasks; see _bridge()'s own docstring for why a plain
`import analyze_gareus_mbar` is not used anywhere in this module.
"""
from __future__ import annotations

import math
import sys
from typing import Any, Optional

import numpy as np


def _bridge() -> Any:
    """Resolve the not-yet-migrated remainder of analyze_gareus_mbar.py.

    Temporary, explicit strangler-fig bridge (see
    docs/superpowers/specs/2026-08-13-mbar-analysis-modularization-a1-design.md
    and this module's own -a4-design.md): functions below call back into
    analyze_gareus_mbar.py for names that belong to a different,
    not-yet-executed plan in this sequence (data loading/masking -> A2, MBAR
    solving -> A3, histogram-overlap diagnostics -> A5, writers/plotting/
    2D-FES-family/convergence -> A6) or that no plan in the A1-A6
    decomposition explicitly owns (_eff_smooth, _smooth_pmf_1d,
    _secondary_cv_label, _secondary_cv_regions, _visible_pmfs -- see the
    design doc's Out Of Scope).

    A plain `import analyze_gareus_mbar` is NOT safe here: running
    `python analyze_gareus_mbar.py <run_dir>` (the still-current, unchanged
    invocation) loads that file as `__main__`, not as a module named
    `analyze_gareus_mbar` -- `sys.modules` has no entry under that name in
    that case. A bare import would then load a SECOND, independent copy of
    the whole script, with its own never-`parse_args`-mutated globals
    (DEFAULT_MBAR_BACKEND, SAMBAR_*, MBAR_ANDERSON_HISTORY) -- silently
    diverging from whatever the user actually passed on the command line.
    This resolver prefers whichever copy is already loaded, `__main__`
    included, and only falls back to a fresh import (which Python then
    caches under the real module name for any subsequent call) when neither
    is present -- e.g. a test or notebook that imports
    `gareus.mbar_analysis.pmf` directly without ever loading the script.
    """
    mod = sys.modules.get('analyze_gareus_mbar')
    if mod is not None:
        return mod
    main_mod = sys.modules.get('__main__')
    if str(getattr(main_mod, '__file__', '')).endswith('analyze_gareus_mbar.py'):
        return main_mod
    import analyze_gareus_mbar as mod
    return mod


# --- Relocated verbatim from analyze_gareus_mbar.py (lines 3290-3755 at the
# commit this plan was written against -- see Task 1, Step 1 for how to
# re-derive the range if it has shifted). ---
```

- [ ] **Step 6: Verify byte-identity of the extracted block (one-time check, not a standing test)**

```bash
grep -n "^def make_bins(cv,bins,lo,hi):" gareus/mbar_analysis/pmf.py
# use the line number found, N, below:
sed -n "${N},$((N+465))p" gareus/mbar_analysis/pmf.py > /tmp/a4_extract/task1_block_in_new_file.txt
diff /tmp/a4_extract/task1_block.txt /tmp/a4_extract/task1_block_in_new_file.txt
```
Expected: no diff output (the 466 lines appended in Step 5 are byte-identical
to what Step 4 extracted from `HEAD`). Also run
`python -m py_compile gareus/mbar_analysis/pmf.py` — expected: clean compile.

- [ ] **Step 7: Run tests to verify the current state is as expected**

Run: `pytest tests/test_mbar_analysis_pmf_module.py -v`
Expected: `test_pmf_module_is_importable_standalone` and the three
`_bridge()` tests PASS (the new module is self-contained and importable).
`test_task1_names_are_reexported_identically_by_analyze_gareus_mbar` FAILS
— at this point `gareus.mbar_analysis.pmf.make_bins` and
`analyze_gareus_mbar.make_bins` are two genuinely distinct function objects
(the script still has its own local `def`, untouched so far), so the `is`
comparison correctly fails. This is expected and gets fixed by Step 8 below.

- [ ] **Step 8: Update `analyze_gareus_mbar.py` — delete the block, add the re-export import**

Locate the units import (or fallback) per Global Constraints:
```bash
grep -n "from gareus.units import\|^import numpy as np" analyze_gareus_mbar.py
```

Immediately after that line, add:
```python
from gareus.mbar_analysis.pmf import (
    make_bins, _bin_indices, pmf_from_weights,
    _cumulant_shared_stats, _cumulant_from_shared, _cumulant_expansion, _cumulant_expansion_both,
    cumulant2, cumulant3,
    pmf2d_from_weights,
    _cumulant_shared_stats_2d, _cumulant_from_shared_2d, _cumulant_expansion_2d, _cumulant_expansion_2d_both,
    _bootstrap_pmf_uncertainty_1d, _bootstrap_pmf_uncertainty_2d,
    cumulant2_2d, cumulant3_2d,
)
```

Then re-confirm and delete the now-duplicated block:
```bash
grep -n "^def make_bins\|^def write_2d_fes_csv" analyze_gareus_mbar.py
```
Delete every line from `make_bins`'s `def` line through (and including) the
blank line immediately before `write_2d_fes_csv`'s `def` line (the 466-line
range confirmed in Step 1, shifted by however many lines this step's own
import insertion added above it).

- [ ] **Step 9: Run tests to verify they pass for real**

Run: `pytest tests/test_mbar_analysis_pmf_module.py -v`
Expected: PASS (all 6 tests) — in particular
`test_task1_names_are_reexported_identically_by_analyze_gareus_mbar` now
proves the import replaced the local defs, not merely duplicated them.

- [ ] **Step 10: Confirm the rest of the file still compiles and the existing suite is unaffected**

Run:
```bash
python -m py_compile analyze_gareus_mbar.py gareus/mbar_analysis/pmf.py
pytest -q tests/test_cumulant_expansion.py tests/test_cumulant_shared_computation.py \
          tests/test_pmf_gamd_nan_bin_handling.py tests/test_pmf_bootstrap_uncertainty.py
```
Expected: clean compile; all four test files pass exactly as they did before
this task (these are the existing test files that import these 18 names
directly from `analyze_gareus_mbar` — see the design doc's Testing section).

- [ ] **Step 11: Commit**

```bash
git add gareus/mbar_analysis/pmf.py analyze_gareus_mbar.py tests/test_mbar_analysis_pmf_module.py
git commit -m "refactor: relocate histogram/cumulant PMF math into gareus.mbar_analysis.pmf"
```

---

### Task 2: Relocate boost/window statistics (`_window_cv_mean_std`, `boost_stats`, `_window_moments`)

**Files:**
- Modify: `gareus/mbar_analysis/pmf.py` (append 3 functions)
- Modify: `analyze_gareus_mbar.py` (delete the 2 source ranges; extend the re-export import)
- Test: `tests/test_mbar_analysis_pmf_module.py`

**Interfaces:**
- Produces: `gareus.mbar_analysis.pmf.{_window_cv_mean_std, boost_stats,
  _window_moments}`, same signatures as today. `_window_moments` is also
  consumed today by `_per_window_gamd_boost_stats`/`plot_gamd_boost` (A6's
  future `gareus/mbar_analysis/...` reporting module) via the re-export
  import this task extends — flagged for A6's own cross-check.
- Consumes (bridged): `norm_logw`, `ess` (both A3-owned; `boost_stats` only).

- [ ] **Step 1: Re-confirm current line boundaries**

```bash
grep -n "^def overlap_matrix\|^def _window_cv_mean_std\|^def pmf_probability" analyze_gareus_mbar.py
grep -n "^def write_cv2_pmf\|^def boost_stats\|^def _window_moments\|^def _per_window_gamd_boost_stats" analyze_gareus_mbar.py
```
Expected (as of this plan being written, i.e. before Task 1's own deletion
shifts everything below line 3755 up by roughly 466 lines minus whatever
Task 1's import block added): `_window_cv_mean_std` runs from its own `def`
line through the line immediately before `pmf_probability`'s `def` line
(46-42 lines: originally 4022-4063, i.e. `def _window_cv_mean_std` through
`return n_k,mean,std`). `boost_stats` + `_window_moments` run from
`boost_stats`'s `def` line through the line immediately before
`_per_window_gamd_boost_stats`'s `def` line (originally 7094-7116, i.e.
`def boost_stats` through `_window_moments`'s `return skew, kurt,
anharmonicity`, with one blank line between the two functions and two blank
lines after). Re-derive both ranges from the post-Task-1 file using these
grep results rather than trusting the numbers above.

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_mbar_analysis_pmf_module.py`:

```python
_TASK2_NAMES = ['_window_cv_mean_std', 'boost_stats', '_window_moments']


def test_task2_names_are_reexported_identically_by_analyze_gareus_mbar():
    import gareus.mbar_analysis.pmf as pmfmod
    import analyze_gareus_mbar as agm
    for name in _TASK2_NAMES:
        assert hasattr(pmfmod, name), f'{name} missing from gareus.mbar_analysis.pmf'
        assert getattr(agm, name) is getattr(pmfmod, name)


def test_boost_stats_still_resolves_norm_logw_and_ess_via_bridge():
    """boost_stats calls norm_logw/ess internally (via _bridge()); a real,
    non-degenerate boost array must still produce a finite ESS -- this is
    the one behavioral check that _bridge() is actually wired into this
    function's body, not just present in the module."""
    import gareus.mbar_analysis.pmf as pmfmod
    import numpy as np
    rng = np.random.default_rng(0)
    boost = rng.normal(50.0, 8.0, size=2000)
    beta = 1.0 / (0.00831446261815324 * 300.0)
    out = pmfmod.boost_stats(boost, beta)
    assert out['available'] is True
    assert np.isfinite(out['boost_reweight_ess'])
    assert 0.0 < out['boost_reweight_ess_fraction'] <= 1.0
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_mbar_analysis_pmf_module.py -v -k task2_or_boost_stats`
Expected: FAIL — `_window_cv_mean_std`/`boost_stats`/`_window_moments` don't
exist in `gareus.mbar_analysis.pmf` yet.

- [ ] **Step 4: Extract both blocks verbatim**

```bash
grep -n "^def _window_cv_mean_std\|^def pmf_probability" analyze_gareus_mbar.py
# use the two line numbers found, call them W_START and W_END_EXCL (pmf_probability's own line)
git show HEAD:analyze_gareus_mbar.py | sed -n '4022,4063p' > /tmp/a4_extract/task2_window_cv_mean_std.txt
wc -l /tmp/a4_extract/task2_window_cv_mean_std.txt   # expect 42

grep -n "^def boost_stats\|^def _per_window_gamd_boost_stats" analyze_gareus_mbar.py
git show HEAD:analyze_gareus_mbar.py | sed -n '7094,7116p' > /tmp/a4_extract/task2_boost_window_moments.txt
wc -l /tmp/a4_extract/task2_boost_window_moments.txt   # expect 23
```
(Re-run both `grep`s against the CURRENT file state per Step 1 and adjust
the `sed` ranges accordingly if they've shifted; the `git show HEAD:...`
source stays anchored to the ranges confirmed at plan-writing time only if
`HEAD` hasn't moved — if intervening commits from Task 1 have landed,
extract from the CURRENT working tree instead of `git show HEAD:...`, since
by this point in execution the "original" reference for byte-identity is
the pre-Task-2 working tree, not the original pre-Task-1 `HEAD`.)

- [ ] **Step 5: Append to `gareus/mbar_analysis/pmf.py`**

Append `task2_window_cv_mean_std.txt`'s contents, then a blank line, then
`task2_boost_window_moments.txt`'s contents, to the end of
`gareus/mbar_analysis/pmf.py` — no other changes. `boost_stats`'s body
already reads `norm_logw(beta*b)` and `ess(w)` as bare names; edit those two
specific lines (only inside `boost_stats`, not anywhere else) to resolve
through the bridge:

Before:
```python
def boost_stats(boost,beta):
    b=boost[np.isfinite(boost)]
    if b.size==0: return {'available':False}
    kc=b/KJ_PER_KCAL; out={'available':True,'n':int(b.size),'mean_kcal_mol':float(np.mean(kc)),'std_kcal_mol':float(np.std(kc)),'min_kcal_mol':float(np.min(kc)),'max_kcal_mol':float(np.max(kc))}
    if b.size>=3 and np.std(b)>0:
        z=(b-np.mean(b))/np.std(b); out['skew']=float(np.mean(z**3)); out['excess_kurtosis']=float(np.mean(z**4)-3.0); out['anharmonicity_score']=float(math.sqrt(out['skew']**2+0.25*out['excess_kurtosis']**2))
    else: out.update({'skew':None,'excess_kurtosis':None,'anharmonicity_score':None})
    w=norm_logw(beta*b); out['boost_reweight_ess']=float(ess(w)); out['boost_reweight_ess_fraction']=float(out['boost_reweight_ess']/b.size)
    return out
```

After:
```python
def boost_stats(boost,beta):
    _agm = _bridge()
    b=boost[np.isfinite(boost)]
    if b.size==0: return {'available':False}
    kc=b/_agm.KJ_PER_KCAL; out={'available':True,'n':int(b.size),'mean_kcal_mol':float(np.mean(kc)),'std_kcal_mol':float(np.std(kc)),'min_kcal_mol':float(np.min(kc)),'max_kcal_mol':float(np.max(kc))}
    if b.size>=3 and np.std(b)>0:
        z=(b-np.mean(b))/np.std(b); out['skew']=float(np.mean(z**3)); out['excess_kurtosis']=float(np.mean(z**4)-3.0); out['anharmonicity_score']=float(math.sqrt(out['skew']**2+0.25*out['excess_kurtosis']**2))
    else: out.update({'skew':None,'excess_kurtosis':None,'anharmonicity_score':None})
    w=_agm.norm_logw(beta*b); out['boost_reweight_ess']=float(_agm.ess(w)); out['boost_reweight_ess_fraction']=float(out['boost_reweight_ess']/b.size)
    return out
```

Two names in this function are bridged (`KJ_PER_KCAL` -> `_agm.KJ_PER_KCAL`,
since it is Plan A1's constant, re-exported from `gareus.units` by
`analyze_gareus_mbar.py`; `norm_logw`/`ess` -> `_agm.norm_logw`/`_agm.ess`,
both A3's). `math.sqrt` is NOT bridged — `math` is stdlib, not a
not-yet-migrated dependency, so Task 1's header (already amended above to
include `import math`) covers it and the call stays a bare `math.sqrt(...)`.

`_window_cv_mean_std` and `_window_moments` have no bridge dependencies —
append them unchanged.

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_mbar_analysis_pmf_module.py -v -k task2_or_boost_stats`
Expected: still FAIL (import line not added to `analyze_gareus_mbar.py` yet
— see next step) except `test_boost_stats_still_resolves_norm_logw_and_ess_via_bridge`,
which should already PASS (it calls `pmfmod.boost_stats` directly, which
works via `_bridge()` regardless of whether the reverse re-export has
happened yet, as long as `analyze_gareus_mbar` — with its own `norm_logw`/
`ess`/`math`/`KJ_PER_KCAL` — is importable, which it already is).

- [ ] **Step 7: Update `analyze_gareus_mbar.py`**

Extend the Task 1 import block (same `from gareus.mbar_analysis.pmf import
(...)` statement) to add `_window_cv_mean_std, boost_stats, _window_moments,`.
Re-confirm and delete both now-duplicated ranges (`_window_cv_mean_std`'s
and `boost_stats`+`_window_moments`'s) from `analyze_gareus_mbar.py`, per
Step 1's re-derived boundaries.

- [ ] **Step 8: Run tests to verify they pass for real**

Run: `pytest tests/test_mbar_analysis_pmf_module.py -v`
Expected: PASS (all tests so far).

- [ ] **Step 9: Regression check**

```bash
python -m py_compile analyze_gareus_mbar.py gareus/mbar_analysis/pmf.py
pytest -q tests/test_perf_report_redundancy.py -k window_cv_mean_std
pytest -q tests/test_perf_plotting_redundancy.py
```
Expected: clean compile; `test_perf_report_redundancy.py`'s
`_window_cv_mean_std`-specific tests pass (vectorized-vs-loop equivalence);
`test_perf_plotting_redundancy.py` (exercises `_window_moments` via
`_per_window_gamd_boost_stats`/`plot_gamd_boost`, still in
`analyze_gareus_mbar.py`, now calling the re-exported name) passes
unchanged.

- [ ] **Step 10: Commit**

```bash
git add gareus/mbar_analysis/pmf.py analyze_gareus_mbar.py tests/test_mbar_analysis_pmf_module.py
git commit -m "refactor: relocate boost/window statistics into gareus.mbar_analysis.pmf"
```

---

### Task 3: Relocate `run_pmf_and_gamd_boost_report`

**Files:**
- Modify: `gareus/mbar_analysis/pmf.py` (append 1 function)
- Modify: `analyze_gareus_mbar.py` (delete the source range; extend the re-export import)
- Test: `tests/test_mbar_analysis_pmf_module.py`

**Interfaces:**
- Produces: `gareus.mbar_analysis.pmf.run_pmf_and_gamd_boost_report(d, args,
  logw, bins, kbt_kcal, out, warnings, progress, warning_prefix='',
  extra_pmfs=None, precomputed_base_w=None) -> dict` — identical signature
  and return-dict shape (`pmfs`, `selected`, `boost_ok`, `boost`,
  `pmf_span_kcal_mol`, `pmf_minimum_cv_A`, `neighbor_overlap`, `n_samples`,
  `O`, `pmf_uncertainty_std`, `files`) to today.
- Consumes (bridged): `norm_logw` (A3), `_eff_smooth` (unowned),
  `overlap_matrix` (A5), `_sample_block_ids` (A2), `write_pmf`, `write_all`,
  `plot_outputs` (A6). Consumes (local, from Tasks 1-2): `pmf_from_weights`,
  `boost_stats`, `_cumulant_expansion_both`, `_bootstrap_pmf_uncertainty_1d`,
  `_window_cv_mean_std`.

- [ ] **Step 1: Re-confirm current line boundaries**

```bash
grep -n "^def _epoch_zero_split_masks\|^def run_pmf_and_gamd_boost_report\|^def analyze\b" analyze_gareus_mbar.py
```
Expected (as of this plan being written): `run_pmf_and_gamd_boost_report`
runs from its own `def` line through the closing `}` of its `return`
statement, immediately before two blank lines and `def analyze(...)`
(originally lines 9449-9596). `_epoch_zero_split_masks` (not moved by this
plan — it's a `Data`-masking helper, A2's territory) ends two blank lines
above `run_pmf_and_gamd_boost_report`'s `def` line.

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_mbar_analysis_pmf_module.py`:

```python
def test_run_pmf_and_gamd_boost_report_is_reexported_identically():
    import gareus.mbar_analysis.pmf as pmfmod
    import analyze_gareus_mbar as agm
    assert agm.run_pmf_and_gamd_boost_report is pmfmod.run_pmf_and_gamd_boost_report


def test_run_pmf_and_gamd_boost_report_end_to_end_still_works(tmp_path):
    """Real end-to-end call through the relocated function, proving every
    bridged name (norm_logw, _eff_smooth, overlap_matrix, _sample_block_ids,
    write_pmf, write_all, plot_outputs) resolves correctly via _bridge()."""
    import gareus.mbar_analysis.pmf as pmfmod
    import analyze_gareus_mbar as agm
    import numpy as np
    from pathlib import Path as _Path

    rng = np.random.default_rng(20)
    n_windows, samples_per_window = 3, 300
    centers = np.linspace(-1.0, 1.0, n_windows)
    k_kcal = np.full(n_windows, 5.0)
    beta = 1.0 / (agm.K_B_KJ_PER_MOL_K * 300.0)

    cv_parts, window_parts, replica_parts, epoch_src_parts = [], [], [], []
    for k in range(n_windows):
        cv_parts.append(centers[k] + rng.normal(0.0, 0.3, size=samples_per_window))
        window_parts.append(np.full(samples_per_window, k, dtype=np.int64))
        replica_parts.append(np.zeros(samples_per_window, dtype=np.int64) + k)
        epoch_src_parts.append(np.zeros(samples_per_window, dtype=np.int64))
    cv = np.concatenate(cv_parts)
    window = np.concatenate(window_parts)
    replica = np.concatenate(replica_parts)
    epoch_src = np.concatenate(epoch_src_parts)
    n = cv.size
    u_nk = np.zeros((n, n_windows), dtype=np.float64)
    for k in range(n_windows):
        u_nk[:, k] = beta * agm.KJ_PER_KCAL * 0.5 * k_kcal[k] * (cv - centers[k]) ** 2

    out_dir = _Path(tmp_path) / 'out'
    out_dir.mkdir(parents=True, exist_ok=True)
    d = agm.Data(
        prod_dir=_Path(tmp_path), out_dir=out_dir, cv=cv, cv2=np.full(n, np.nan),
        rg_A=np.full(n, np.nan), window=window, replica=replica, step=np.arange(n),
        u_nk=u_nk, centers=centers, k_kcal=k_kcal, beta=beta, temp=300.0,
        boost_kj=np.full(n, np.nan), potential_kj=None, source='test',
        meta={'_epoch_source': epoch_src.tolist()},
    )
    m = agm.solve_mbar(d.u_nk, d.window)
    logw = np.asarray(m['logw'], dtype=np.float64)
    bins = agm.make_bins(d.cv, 20, None, None)
    kbt_kcal = (1.0 / d.beta) / agm.KJ_PER_KCAL

    class _Args:
        bins = 20
        min_neighbor_overlap = 0.0
        selected_method = 'auto'
        gamd_smooth_sigma = 0.0
        pmf_smooth_sigma = 0.0
        smooth_sigma = 0.0
        pmf_uncertainty = False
        pmf_uncertainty_n_boot = 30
        pmf_uncertainty_seed = 0

    info = pmfmod.run_pmf_and_gamd_boost_report(d, _Args(), logw, bins, kbt_kcal, out_dir, [], None)
    assert info['selected'] == 'umbrella_only'  # no finite boost in this fixture
    assert (out_dir / 'pmf_unbiased.csv').exists()
    assert (out_dir / 'window_diagnostics.csv').exists()
    assert (out_dir / 'overlap_matrix.csv').exists()
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_mbar_analysis_pmf_module.py -v -k run_pmf_and_gamd_boost_report`
Expected: FAIL — `run_pmf_and_gamd_boost_report` doesn't exist in
`gareus.mbar_analysis.pmf` yet.

- [ ] **Step 4: Extract the block verbatim**

```bash
git show HEAD:analyze_gareus_mbar.py | sed -n '9449,9596p' > /tmp/a4_extract/task3_report.txt
wc -l /tmp/a4_extract/task3_report.txt   # expect 148
```
(Re-derive the range from the current working tree per Step 1 if Tasks 1-2
have already shifted it; use `git diff` against the pre-Task-1 `HEAD` to
extract from the current file instead of `git show HEAD:...` if so — the
byte-identity bar is "unchanged since before this plan started," not
specifically "matches original `HEAD`".)

- [ ] **Step 5: Append to `gareus/mbar_analysis/pmf.py`, with the bridge substitutions**

Append `task3_report.txt`'s contents, then apply exactly these edits to the
newly-appended copy (not the original in `analyze_gareus_mbar.py`, which
Step 7 deletes wholesale):

1. Add `_agm = _bridge()` as the function's first line, immediately after
   the docstring's closing `"""`.
2. `base_w = norm_logw(np.asarray(logw, dtype=np.float64))` →
   `base_w = _agm.norm_logw(np.asarray(logw, dtype=np.float64))`
3. `exp_w = norm_logw(logw + d.beta * d.boost_kj)` →
   `exp_w = _agm.norm_logw(logw + d.beta * d.boost_kj)`
4. `(cum_pmf, cdiag), (cum3_pmf, cdiag3) = _cumulant_expansion_both(d.cv, base_w, d.boost_kj, bins, d.beta, kbt_kcal, smooth_logfac_sigma=_eff_smooth(args, 'gamd_smooth_sigma'))`
   → `(cum_pmf, cdiag), (cum3_pmf, cdiag3) = _cumulant_expansion_both(d.cv, base_w, d.boost_kj, bins, d.beta, kbt_kcal, smooth_logfac_sigma=_agm._eff_smooth(args, 'gamd_smooth_sigma'))`
   (`_cumulant_expansion_both` itself stays a bare local call — relocated in Task 1.)
5. `O = overlap_matrix(d.cv, d.window, bins, K)` →
   `O = _agm.overlap_matrix(d.cv, d.window, bins, K)`
6. `block_ids = _sample_block_ids(d)` → `block_ids = _agm._sample_block_ids(d)`
7. Inside the same `if` block, `smooth_logfac_sigma=_eff_smooth(args, 'gamd_smooth_sigma'),`
   (the kwarg passed to `_bootstrap_pmf_uncertainty_1d`, itself a bare local
   call — relocated in Task 1) → `smooth_logfac_sigma=_agm._eff_smooth(args, 'gamd_smooth_sigma'),`
8. All five `write_pmf(...)` calls (`pmf_unbiased.csv`,
   `pmf_umbrella_only.csv`, `pmf_gamd_exponential.csv`,
   `pmf_gamd_cumulant2.csv`, `pmf_gamd_cumulant3.csv`) → `_agm.write_pmf(...)`
9. `write_all(out / 'pmf_all_methods.csv', pmfs)` → `_agm.write_all(out / 'pmf_all_methods.csv', pmfs)`
10. `plot_outputs(d, pmfs, selected, O, out, warnings, smooth_sigma=_eff_smooth(args, 'pmf_smooth_sigma'), args=args)`
    → `_agm.plot_outputs(d, pmfs, selected, O, out, warnings, smooth_sigma=_agm._eff_smooth(args, 'pmf_smooth_sigma'), args=args)`

Every other line (the four PMF-method branch, `pmfs` dict assembly,
`_force_method` override logic, neighbor-overlap warning, `_sel_diag`
selection, the `window_diagnostics.csv` writer loop, the returned dict) is
untouched — `pmf_from_weights`, `boost_stats`, `_window_cv_mean_std`, and
`_bootstrap_pmf_uncertainty_1d` are all now LOCAL names (Tasks 1-2), so
those calls need no edit.

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_mbar_analysis_pmf_module.py -v -k run_pmf_and_gamd_boost_report`
Expected: `test_run_pmf_and_gamd_boost_report_end_to_end_still_works` PASSES
already (calls `pmfmod.run_pmf_and_gamd_boost_report` directly — works via
`_bridge()` regardless of the reverse re-export). `test_..._is_reexported_identically`
still FAILS (re-export not added yet — see Step 7).

- [ ] **Step 7: Update `analyze_gareus_mbar.py`**

Extend the import block to add `run_pmf_and_gamd_boost_report,`. Re-confirm
and delete the now-duplicated 148-line range.

- [ ] **Step 8: Run tests to verify they pass for real**

Run: `pytest tests/test_mbar_analysis_pmf_module.py -v`
Expected: PASS (all tests so far).

- [ ] **Step 9: Regression check**

```bash
python -m py_compile analyze_gareus_mbar.py gareus/mbar_analysis/pmf.py
pytest -q tests/test_cumulant_shared_computation.py -k run_pmf_and_gamd_boost_report
pytest -q tests/test_epoch0_pmf_gamd_split.py tests/test_masked_logw_subset_pmf.py tests/test_perf_report_redundancy.py
pytest -q tests/test_pmf_bootstrap_uncertainty.py -k pmf_uncertainty_o
```
Expected: clean compile; all pass exactly as before this task — these are
the existing test files that exercise `run_pmf_and_gamd_boost_report`
end-to-end (see the design doc's Testing section for what each covers).

- [ ] **Step 10: Commit**

```bash
git add gareus/mbar_analysis/pmf.py analyze_gareus_mbar.py tests/test_mbar_analysis_pmf_module.py
git commit -m "refactor: relocate run_pmf_and_gamd_boost_report into gareus.mbar_analysis.pmf"
```

---

### Task 4: Relocate `run_secondary_cv_analyses` + `analyze_secondary_cv_pmf`

**Files:**
- Modify: `gareus/mbar_analysis/pmf.py` (append 2 functions)
- Modify: `analyze_gareus_mbar.py` (delete both source ranges; extend the re-export import)
- Test: `tests/test_mbar_analysis_pmf_module.py`

**Interfaces:**
- Produces: `gareus.mbar_analysis.pmf.run_secondary_cv_analyses(d, args,
  base_logw, selected, boost_ok, kbt_kcal, out, warnings, progress,
  f_k_global=None) -> tuple` and
  `gareus.mbar_analysis.pmf.analyze_secondary_cv_pmf(d, args, base_logw,
  selected, boost_ok, kbt_kcal, out, warnings, progress) -> dict` —
  identical signatures to today.
- Consumes (bridged): `norm_logw` (A3), `_eff_smooth`, `_smooth_pmf_1d`,
  `_secondary_cv_label`, `_secondary_cv_regions`, `_visible_pmfs` (all
  unowned), `_secondary_cv_epoch_regime_masks`, `_masked_data`,
  `_subset_logw_from_global_fk`, `_regime_slug`, `wjson` (all A2),
  `analyze_cv1_cv2_2d_fes`, `write_cv2_pmf`, `run_observable_pmf_convergence`
  (all A6). Consumes (local, from Task 1): `make_bins`, `pmf_from_weights`,
  `_cumulant_expansion_both`.
- Test coverage note: this task's own new end-to-end test (Step 2) exercises
  `analyze_secondary_cv_pmf` directly (the function with the larger, 9-name
  bridge surface) and is not duplicated for `run_secondary_cv_analyses`'s
  multi-regime branch. That branch's own three regime-specific bridged names
  (`_masked_data`, `_subset_logw_from_global_fk`, `_regime_slug`) are already
  exhaustively exercised against the *relocated* function by the pre-existing
  `tests/test_secondary_cv_regime_split.py`, re-run unchanged in Step 9 —
  they are covered by regression, not left untested.

- [ ] **Step 1: Re-confirm current line boundaries**

```bash
grep -n "^def _regime_slug\|^def run_secondary_cv_analyses\|^def _primary_cv_label" analyze_gareus_mbar.py
grep -n "^def analyze_secondary_cv_pmf\|^def analyze_cv1_cv2_2d_fes" analyze_gareus_mbar.py
```
Expected (as of this plan being written): `run_secondary_cv_analyses` runs
from its own `def` line through `return dominant_pmf_info,
dominant_fes_info`, immediately before one blank line and `def
_primary_cv_label(...)` (originally lines 514-587). `analyze_secondary_cv_pmf`
runs from its own `def` line through `return info`, immediately before two
blank lines and `def analyze_cv1_cv2_2d_fes(...)` (originally lines
7619-7684).

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_mbar_analysis_pmf_module.py`:

```python
def test_task4_names_are_reexported_identically():
    import gareus.mbar_analysis.pmf as pmfmod
    import analyze_gareus_mbar as agm
    assert agm.run_secondary_cv_analyses is pmfmod.run_secondary_cv_analyses
    assert agm.analyze_secondary_cv_pmf is pmfmod.analyze_secondary_cv_pmf


def test_analyze_secondary_cv_pmf_end_to_end_still_works(tmp_path):
    """Real end-to-end call proving every bridged name in this function
    (norm_logw, _eff_smooth, write_cv2_pmf, _secondary_cv_label,
    _secondary_cv_regions, _visible_pmfs, _smooth_pmf_1d,
    run_observable_pmf_convergence, wjson) resolves via _bridge()."""
    import gareus.mbar_analysis.pmf as pmfmod
    import analyze_gareus_mbar as agm
    import numpy as np
    from pathlib import Path as _Path

    rng = np.random.default_rng(30)
    n_windows, samples_per_window = 3, 300
    centers = np.linspace(-1.0, 1.0, n_windows)
    k_kcal = np.full(n_windows, 5.0)
    beta = 1.0 / (agm.K_B_KJ_PER_MOL_K * 300.0)
    cv_parts, cv2_parts, window_parts = [], [], []
    for k in range(n_windows):
        cv_parts.append(centers[k] + rng.normal(0.0, 0.3, size=samples_per_window))
        cv2_parts.append(rng.normal(0.0, 1.0, size=samples_per_window))
        window_parts.append(np.full(samples_per_window, k, dtype=np.int64))
    cv = np.concatenate(cv_parts)
    cv2 = np.concatenate(cv2_parts)
    window = np.concatenate(window_parts)
    n = cv.size
    u_nk = np.zeros((n, n_windows), dtype=np.float64)
    for k in range(n_windows):
        u_nk[:, k] = beta * agm.KJ_PER_KCAL * 0.5 * k_kcal[k] * (cv - centers[k]) ** 2

    out_dir = _Path(tmp_path) / 'out'
    out_dir.mkdir(parents=True, exist_ok=True)
    d = agm.Data(
        prod_dir=_Path(tmp_path), out_dir=out_dir, cv=cv, cv2=cv2,
        rg_A=np.full(n, np.nan), window=window, replica=window.copy(), step=np.arange(n),
        u_nk=u_nk, centers=centers, k_kcal=k_kcal, beta=beta, temp=300.0,
        boost_kj=np.full(n, np.nan), potential_kj=None, source='test', meta={},
    )
    m = agm.solve_mbar(d.u_nk, d.window)
    logw = np.asarray(m['logw'], dtype=np.float64)
    kbt_kcal = (1.0 / d.beta) / agm.KJ_PER_KCAL

    class _Args:
        bins = 20
        cv2_bins = None
        cv2_min = None
        cv2_max = None
        gamd_smooth_sigma = 0.0
        pmf_smooth_sigma = 0.0
        smooth_sigma = 0.0
        plot_gamd_exponential = False
        plot_gamd_cumulant3 = False
        no_convergence = True

    info = pmfmod.analyze_secondary_cv_pmf(d, _Args(), logw, 'umbrella_only', False, kbt_kcal, out_dir, [], None)
    assert info['available'] is True
    assert (out_dir / 'cv2_pmf_unbiased.csv').exists()
    assert (out_dir / 'cv2_pmf_summary.json').exists()
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_mbar_analysis_pmf_module.py -v -k task4_or_secondary_cv_pmf_end_to_end`
Expected: FAIL — neither function exists in `gareus.mbar_analysis.pmf` yet.

- [ ] **Step 4: Extract both blocks verbatim**

```bash
git show HEAD:analyze_gareus_mbar.py | sed -n '514,587p' > /tmp/a4_extract/task4_run_secondary_cv_analyses.txt
wc -l /tmp/a4_extract/task4_run_secondary_cv_analyses.txt   # expect 74
git show HEAD:analyze_gareus_mbar.py | sed -n '7619,7684p' > /tmp/a4_extract/task4_analyze_secondary_cv_pmf.txt
wc -l /tmp/a4_extract/task4_analyze_secondary_cv_pmf.txt   # expect 66
```
(Re-derive both ranges from the current working tree per Step 1 if Tasks
1-3 have already shifted them.)

- [ ] **Step 5: Append to `gareus/mbar_analysis/pmf.py`, with bridge substitutions**

Append `task4_run_secondary_cv_analyses.txt`'s contents first (it calls
`analyze_secondary_cv_pmf`, appended second, but Python resolves both names
from the module namespace at call time, not def time, so order between the
two doesn't matter), then `task4_analyze_secondary_cv_pmf.txt`'s contents.
Apply exactly these edits to the newly-appended copies:

**In `run_secondary_cv_analyses`:**

1. Add `_agm = _bridge()` as the function's first line, immediately after
   the docstring's closing `"""`.
2. `regimes = _secondary_cv_epoch_regime_masks(d, warnings=warnings)` →
   `regimes = _agm._secondary_cv_epoch_regime_masks(d, warnings=warnings)`
3. Both occurrences of `analyze_cv1_cv2_2d_fes(d, args, ...)` /
   `analyze_cv1_cv2_2d_fes(d_regime, args, ...)` → `_agm.analyze_cv1_cv2_2d_fes(...)`
   (`analyze_secondary_cv_pmf` itself, called at both of the same two call
   sites, stays a bare LOCAL call — it's defined in this same module.)
4. `d_regime = _masked_data(d, mask, meta_override=regime_meta)` →
   `d_regime = _agm._masked_data(d, mask, meta_override=regime_meta)`
5. `base_logw_regime = _subset_logw_from_global_fk(d_regime, f_k_global)` →
   `base_logw_regime = _agm._subset_logw_from_global_fk(d_regime, f_k_global)`
6. `regime_out = out if is_dominant else out / f'secondary_cv_regime_{_regime_slug(regime)}'`
   → `regime_out = out if is_dominant else out / f'secondary_cv_regime_{_agm._regime_slug(regime)}'`

**In `analyze_secondary_cv_pmf`:**

1. Add `_agm = _bridge()` as the function's first line, immediately after
   the docstring's closing `"""`.
2. `base_w=norm_logw(base_logw_sel)` → `base_w=_agm.norm_logw(base_logw_sel)`
3. `bins=make_bins(cv2_sel, bins_n, getattr(args,'cv2_min',None), getattr(args,'cv2_max',None))`
   stays a bare LOCAL call (Task 1).
4. `umbrella=pmf_from_weights(cv2_sel, base_w, bins, kbt_kcal)` stays LOCAL.
5. `exp_w=norm_logw(base_logw_sel + d.beta*boost_sel)` →
   `exp_w=_agm.norm_logw(base_logw_sel + d.beta*boost_sel)`
6. `exp_pmf=pmf_from_weights(cv2_sel, exp_w, bins, kbt_kcal)` stays LOCAL.
7. `(cum_pmf,cdiag),(cum3_pmf,cdiag3)=_cumulant_expansion_both(cv2_sel, base_w, boost_sel, bins, d.beta, kbt_kcal, smooth_logfac_sigma=_eff_smooth(args,'gamd_smooth_sigma'))`
   → keep `_cumulant_expansion_both` bare (LOCAL, Task 1), bridge only the
   kwarg: `smooth_logfac_sigma=_agm._eff_smooth(args,'gamd_smooth_sigma')`
8. All five `write_cv2_pmf(...)` calls → `_agm.write_cv2_pmf(...)`
9. `cv2_label=_secondary_cv_label(d.meta)` → `cv2_label=_agm._secondary_cv_label(d.meta)`
10. `regions=_secondary_cv_regions(d.meta)` → `regions=_agm._secondary_cv_regions(d.meta)`
11. `_cv2_smooth=_eff_smooth(args,'pmf_smooth_sigma')` → `_cv2_smooth=_agm._eff_smooth(args,'pmf_smooth_sigma')`
12. `for i,(name,p) in enumerate(_visible_pmfs(pmfs, chosen, args).items()):`
    → `for i,(name,p) in enumerate(_agm._visible_pmfs(pmfs, chosen, args).items()):`
13. `pmf_plot=_smooth_pmf_1d(p['pmf'],_cv2_smooth); m=np.isfinite(pmf_plot)` →
    `pmf_plot=_agm._smooth_pmf_1d(p['pmf'],_cv2_smooth); m=np.isfinite(pmf_plot)`
14. `cv2_conv=run_observable_pmf_convergence(...)` → `cv2_conv=_agm.run_observable_pmf_convergence(...)`
    (argument list unchanged)
15. `wjson(out/'cv2_pmf_summary.json', info)` → `_agm.wjson(out/'cv2_pmf_summary.json', info)`

Every other line (the mask/availability check, the four-PMF-method
branch, `chosen`/`chosen_pmf` selection, the `plot_method_curve`/
`gareus_plotstyle` import inside the plotting `try` block, the `axvline`
region-annotation loop, `span`/`info` assembly) is untouched.

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_mbar_analysis_pmf_module.py -v -k task4_or_secondary_cv_pmf_end_to_end`
Expected: `test_analyze_secondary_cv_pmf_end_to_end_still_works` PASSES
already (direct call into `pmfmod`, works via `_bridge()`).
`test_task4_names_are_reexported_identically` still FAILS (re-export not
added yet).

- [ ] **Step 7: Update `analyze_gareus_mbar.py`**

Extend the import block to add `run_secondary_cv_analyses,
analyze_secondary_cv_pmf,`. Re-confirm and delete both now-duplicated
ranges.

- [ ] **Step 8: Run tests to verify they pass for real**

Run: `pytest tests/test_mbar_analysis_pmf_module.py -v`
Expected: PASS (every test in the file).

- [ ] **Step 9: Regression check**

```bash
python -m py_compile analyze_gareus_mbar.py gareus/mbar_analysis/pmf.py
pytest -q tests/test_secondary_cv_regime_split.py
pytest -q tests/test_masked_logw_subset_pmf.py -k secondary_cv_analyses
```
Expected: clean compile; both pass exactly as before this task.

- [ ] **Step 10: Commit**

```bash
git add gareus/mbar_analysis/pmf.py analyze_gareus_mbar.py tests/test_mbar_analysis_pmf_module.py
git commit -m "refactor: relocate secondary-CV PMF analysis into gareus.mbar_analysis.pmf"
```

---

### Task 5: Full-suite verification

**Files:** none modified (verification only).

- [ ] **Step 1: Full regression suite**

```bash
pytest -q tests/ --ignore=tests/test_validation_common.py
```
Expected: the same pre-existing failures as `main` today (the 6
known-unrelated failures documented in Plan A1's own Testing section:
help-text encyclopedia numbering, threadpool-import check, missing
validation launcher fixtures) and no new ones.

- [ ] **Step 2: Compile and standalone-importability check**

```bash
python -m py_compile analyze_gareus_mbar.py gareus/mbar_analysis/pmf.py
python -c "import gareus.mbar_analysis.pmf"
python -c "import sys; assert 'analyze_gareus_mbar' not in sys.modules; import gareus.mbar_analysis.pmf; assert 'analyze_gareus_mbar' not in sys.modules; print('OK: importing pmf.py did not eagerly pull in analyze_gareus_mbar')"
```
Expected: clean compile; both import checks pass, the second one proving
`_bridge()`'s deferred design holds (no function in the new module was
*called* by this check, so `analyze_gareus_mbar` should never load).

- [ ] **Step 3: End-to-end CLI smoke test unaffected**

```bash
python analyze_gareus_mbar.py --help > /tmp/a4_help_after.txt
diff /tmp/a4_help_after.txt <(git show HEAD~10:analyze_gareus_mbar.py > /tmp/a4_old_script.py 2>/dev/null && python /tmp/a4_old_script.py --help 2>/dev/null || true)
```
(This is a soft sanity check, not a hard gate — argparse `--help` output
should be byte-identical since Global Constraints forbid touching
`parse_args()`; if the `HEAD~10` comparison script fails to run standalone
due to unrelated import-path assumptions, skip the diff and instead confirm
`python analyze_gareus_mbar.py --help` exits 0 and mentions
`--pmf-uncertainty` in its output.)

- [ ] **Step 4: Self-review**

- **Spec coverage**: all 24 functions listed in the spec's Goal are
  accounted for across Tasks 1-4 (18 + 3 + 1 + 2 = 24). The spec's
  `_bootstrap_pmf_uncertainty_2d`-stays-unwired non-goal, the
  `write_pmf`/`plot_outputs`/`overlap_matrix` temporary-bridge decision, and
  the "no formula/behavior change" constraint are all honored — every code
  change inside a relocated function body is a bridge substitution listed
  explicitly in Tasks 3-4, nothing else.
- **Placeholder scan**: every step gives complete, runnable shell commands
  or complete Python code; the `sed`/`grep`-based extraction steps are
  mechanical instructions with real, copy-pasteable commands, not
  "extract the relevant code" placeholders. Confirm by re-reading Tasks 1-4
  Step 4/5 pairs: each names an exact line range (with a re-confirmation
  grep) and an exact list of substitutions (with literal before/after code).
- **Signature consistency across tasks**: `_bridge()` is defined once
  (Task 1) and referenced identically (`_agm = _bridge()`, then
  `_agm.<name>`) in every function that needs it in Tasks 2-4. The growing
  `from gareus.mbar_analysis.pmf import (...)` block in
  `analyze_gareus_mbar.py` is extended, never replaced, across Tasks 1-4 —
  confirm the final block (after Task 4) lists all 24 names exactly once
  each, matching the design doc's function list.

- [ ] **Step 5: Report to the coordinating session**

No commit for this task (verification-only) — report the full-suite result,
any of the 6 known-unrelated pre-existing failures re-confirmed present,
and explicit confirmation that no other domain's plan (A2/A3/A5/A6) needs to
change anything as a *result* of this plan's relocation — only that A6, when
it eventually executes, should import `make_bins`/`pmf_from_weights`/
`pmf2d_from_weights`/`_cumulant_expansion_both`/`_cumulant_expansion_2d_both`/
`_window_moments` from `gareus.mbar_analysis.pmf` directly (see the design
doc's "Who else calls these 24 functions" subsection) rather than continuing
through `analyze_gareus_mbar.py`'s re-export once it does its own
relocation.

---

## Plan Self-Review Notes

- **Spec coverage**: every one of the design doc's 24 named functions is
  relocated across Tasks 1-4, in the dependency order the design doc's
  Architecture section specifies (leaf math and the two bootstrap helpers
  first, since they form one contiguous, near-zero-bridge block; boost/
  window statistics second; the two heaviest-bridge orchestrators last).
  The design doc's `_bridge()` design (preferring an already-loaded module,
  `__main__` included, over a bare `import`) is implemented verbatim in
  Task 1 and exercised by dedicated tests, not just asserted in prose. The
  `write_pmf`/`plot_outputs`/`overlap_matrix` temporary-bridge decision and
  the unowned-helper flags (`_eff_smooth`, `_smooth_pmf_1d`,
  `_secondary_cv_label`, `_secondary_cv_regions`, `_visible_pmfs`) are both
  implemented as specified and re-stated for the coordinating session in
  Task 5.
- **Type/signature consistency checked**: every relocated function keeps
  its exact current signature (verified against the actual source read
  during this plan's authoring, not assumed) — no task changes a parameter
  name, default, or return-dict key. The `_agm = _bridge()` pattern is
  applied identically everywhere it's needed.
- **No placeholders**: every task gives exact `grep`/`sed` commands with
  real line numbers (explicitly flagged as needing re-confirmation, per
  Plan A1's own convention, rather than blindly trusted), exact before/after
  code for every bridge substitution, and complete, runnable test code —
  no "extract and adapt as needed" steps.
