# MBAR Analysis Modularization — Plan A1 (Scaffolding + Constants Reconciliation)

## Goal

Fold `analyze_gareus_mbar.py` (a ~9,950-line top-level script, ~260 functions)
into the `gareus` package, incrementally. This is Plan A1 of a 6-plan sequence
(A1-A6, see Out Of Scope) — the lowest-risk, highest-leverage first slice:
prove the new package plumbing (subpackage layout, installable entry point)
and reconcile the one piece of genuinely trivial duplication (physical
constants), without touching any of the file's actual analysis logic.

## Problem

`analyze_gareus_mbar.py` grew as a standalone script alongside — not inside —
the `gareus` package, and duplicates physics/math the package already has:

- `analyze_gareus_mbar.py` defines its own `K_B_KJ_PER_MOL_K = 0.00831446261815324`
  and `KJ_PER_KCAL = 4.184` at module scope. `gareus/units.py` independently
  hardcodes the same `4.184` factor inline inside `kcal_a2_to_kj_nm2`/
  `kj_nm2_to_kcal_a2`/`kcal_to_kj`, with no named constant at all. Two sources
  of the same physical fact, in two files, with no shared definition.
- This is the first of three known duplications between the script and the
  package (the other two — MBAR bias reconstruction vs. `gareus/query.py`'s
  `reconstruct_bias_matrix`, and histogram-overlap statistics vs.
  `gareus/math_helpers.py`/`gareus/diagnostics.py` — are scientifically
  consequential and explicitly deferred to Plan A3/A5; see Out Of Scope).
- `analyze_gareus_mbar.py` has no package entry point. `pyproject.toml`
  already declares 5 console scripts (`gareus`, `gareus-energy-decompose`,
  `gareus-test-run`, `gareus-suggest-cvs`, `gareus-consolidate-traj`, all
  `gareus.<module>:main`) and an `analysis` extras group
  (`pymbar>=4, pyarrow>=12, duckdb>=0.9, scipy>=1.10` — exactly what this
  script needs) that nothing currently uses — the packaging metadata already
  anticipated this move.

## Recommended Approach

**New `gareus/mbar_analysis/` subpackage, populated with a thin delegating
entry point; reconcile constants into `gareus/units.py`.** Concretely:

1. `gareus/units.py` gains named constants `KJ_PER_KCAL = 4.184` and
   `K_B_KJ_PER_MOL_K = 0.00831446261815324`, added to `__all__`. Its existing
   three functions (`kcal_a2_to_kj_nm2`, `kj_nm2_to_kcal_a2`, `kcal_to_kj`)
   are refactored to reference `KJ_PER_KCAL` instead of the inline `4.184`
   literal — same values, one name.
2. `analyze_gareus_mbar.py`'s own `K_B_KJ_PER_MOL_K`/`KJ_PER_KCAL`
   definitions are replaced with
   `from gareus.units import KJ_PER_KCAL, K_B_KJ_PER_MOL_K` — the first real
   cross-import from the script into the package. Every one of the script's
   ~50+ existing use sites is unchanged (same names, same values, now a
   single source).
3. New subpackage `gareus/mbar_analysis/` (auto-discovered by
   `pyproject.toml`'s existing `packages.find include = ["gareus*"]` — no
   packaging-config change needed for the subpackage itself):
   - `__init__.py` — package marker and one-line docstring.
   - `cli.py` — `main(argv=None)` that imports `analyze_gareus_mbar` and
     calls its existing `main(argv)`. This is a **strangler-fig entry
     point**: it owns no real logic yet. Plans A2-A6 progressively move real
     logic into `gareus/mbar_analysis/` until this delegation becomes real
     ownership (by A6, `analyze_gareus_mbar.py` itself is retired).
4. `pyproject.toml` gains one new line in `[project.scripts]`:
   `gareus-analyze = "gareus.mbar_analysis.cli:main"` — matching the
   existing 5-script convention exactly. No new extras group: the existing
   `analysis` extra already lists everything this needs.

**The old invocation (`python analyze_gareus_mbar.py <run_dir> ...`) keeps
working, unchanged, for the entire A1-A5 transition.** It is only formally
retired in Plan A6, once all real logic has actually relocated — breaking it
before then would remove the only working copy of ~9,900 lines of logic for
no benefit. The new `gareus-analyze` entry point is additive during the
transition, not a replacement yet.

## Why This Approach

- Proves every piece of new plumbing (subpackage discovery, console-script
  registration, cross-import from script into package) on the
  lowest-possible-risk material — a hardcoded float and a delegating
  function call — before any later plan touches scientifically meaningful
  code.
- The constants reconciliation is the one duplication in this codebase with
  zero judgment calls: `4.184` is `4.184`, unlike the bias-reconstruction and
  histogram-overlap duplications (Plans A3/A5), which have measurable
  behavioral differences between their copies and need the kind of careful,
  audited reconciliation this session's earlier physics/perf audits already
  demonstrated is necessary for this file.
- Matches the packaging metadata's own already-declared intent (the
  `analysis` extras group, the 5-script convention) rather than inventing a
  new pattern.
- Fully reversible and low-blast-radius: if anything about the subpackage
  shape turns out wrong once A2 starts moving real code into it, nothing
  built in A1 constrains that decision — `cli.py`'s delegation doesn't
  encode any assumption about where A2's code will eventually live.

## Alternatives Considered

### Move everything in one plan

Rejected per the already-agreed incremental sequencing: the file is 9,950
lines with real scientific-correctness stakes in several domains (this
session's own physics and performance audits found and fixed multiple
subtle bugs in this exact file) — a single giant move has no natural
checkpoint to catch a mistake before it's buried under five more domains'
worth of changes.

### Skip the constants reconciliation, do pure scaffolding only

Considered making A1 *only* the subpackage + entry point, with constants
deferred to a later plan. Rejected: the constants duplication is genuinely
free to fix (no judgment calls, confirmed identical values, `gareus/units.py`
already exists as the obviously-correct home) and doing it now means A1
delivers one real, verifiable piece of deduplication rather than pure
infrastructure with no code-quality payoff yet.

### `python -m gareus.mbar_analysis` instead of a console script

Considered as the entry point instead of registering in `pyproject.toml`.
Rejected once the existing 5-script convention was found: a `gareus-analyze`
console script is more consistent with how every other CLI in this project
is exposed, and registering it costs one line.

## Architecture

### Files touched

```
gareus/units.py                      MODIFY  (add 2 named constants, refactor 3 functions)
gareus/mbar_analysis/__init__.py     CREATE  (package marker)
gareus/mbar_analysis/cli.py          CREATE  (delegating main())
analyze_gareus_mbar.py               MODIFY  (2-line constant-definition swap for an import)
pyproject.toml                       MODIFY  (1 new line in [project.scripts])
```

### `gareus/units.py` (after)

```python
KJ_PER_KCAL = 4.184
K_B_KJ_PER_MOL_K = 0.00831446261815324  # Boltzmann constant, kJ/(mol*K)

def kcal_a2_to_kj_nm2(k_kcal_a2: float) -> float:
    return float(k_kcal_a2) * KJ_PER_KCAL / 0.01

def kj_nm2_to_kcal_a2(k_kj_nm2: float) -> float:
    return float(k_kj_nm2) * 0.01 / KJ_PER_KCAL

def kcal_to_kj(k_kcal: float) -> float:
    return float(k_kcal) * KJ_PER_KCAL

__all__ = [
    "KJ_PER_KCAL", "K_B_KJ_PER_MOL_K",
    "kcal_a2_to_kj_nm2", "kj_nm2_to_kcal_a2", "kcal_to_kj", "bias_energy_kj",
]
```

`bias_energy_kj` (already calls `kcal_to_kj` internally) is untouched.

### `analyze_gareus_mbar.py` (relevant lines)

Before:
```python
K_B_KJ_PER_MOL_K=0.00831446261815324
KJ_PER_KCAL=4.184
```
After:
```python
from gareus.units import KJ_PER_KCAL, K_B_KJ_PER_MOL_K
```

### `gareus/mbar_analysis/cli.py`

```python
"""Delegating CLI entry point for `gareus-analyze`.

Strangler-fig placeholder: owns no analysis logic yet. Plans A2-A6
progressively move analyze_gareus_mbar.py's real logic into this
subpackage; until then this module's only job is to make `gareus-analyze`
behave identically to `python analyze_gareus_mbar.py`.
"""
from __future__ import annotations

from typing import Optional, Sequence


def main(argv: Optional[Sequence[str]] = None) -> None:
    import analyze_gareus_mbar
    analyze_gareus_mbar.main(argv)
```

### `pyproject.toml`

One line added to `[project.scripts]`:
```
gareus-analyze = "gareus.mbar_analysis.cli:main"
```

### Import direction and the repo-root constraint

`gareus/mbar_analysis/cli.py` imports `analyze_gareus_mbar` — a top-level
module, not part of the `gareus*` package glob. This only resolves when the
repo root is on `sys.path`, which is already the project's standing
convention (confirmed directly: `pyproject.toml`'s own
`[tool.pytest.ini_options] pythonpath = ["."]`, and `gareus` itself is only
importable today the same way — it isn't `pip install`-ed, per direct check
in this environment). This is not a new fragility A1 introduces; it's the
project's existing invocation convention, temporarily crossing in the
opposite direction (package importing the script) until A6 removes the need
entirely.

## Error Handling

Nothing new to guard: `cli.py`'s `main()` is a pure delegation with no logic
of its own, so any error handling stays exactly where it already is, inside
`analyze_gareus_mbar.main()`. No new failure modes are introduced by this
plan.

## Testing

- `gareus.units.KJ_PER_KCAL == 4.184` and `gareus.units.K_B_KJ_PER_MOL_K`
  matches the documented CODATA value (both already numerically verified
  earlier this session against a real T=300K round-trip check).
- `analyze_gareus_mbar.KJ_PER_KCAL is gareus.units.KJ_PER_KCAL` and
  `analyze_gareus_mbar.K_B_KJ_PER_MOL_K is gareus.units.K_B_KJ_PER_MOL_K`
  after the import change — same object, not just coincidentally-equal
  values, proving the import actually replaced the local definitions rather
  than merely shadowing them.
- `gareus/units.py`'s three conversion functions still produce identical
  output before/after the refactor (regression guard on the
  literal-to-named-constant swap).
- `gareus.mbar_analysis.cli.main(['--help'])` and
  `analyze_gareus_mbar.main(['--help'])` produce identical stdout — proves
  the delegation is real, not just "doesn't crash."
- The existing full test suite (`pytest tests/`) must show the same
  pre-existing failures as `main` today (6 known-unrelated failures,
  confirmed via `git diff --stat` against files this plan doesn't touch) and
  no new ones — this plan changes constant *definitions*, not their values,
  so no numeric test should be able to detect the change at all.

Verification commands (once implemented):

```bash
pytest -q tests/test_mbar_analysis_scaffolding.py
python -m py_compile analyze_gareus_mbar.py gareus/units.py gareus/mbar_analysis/cli.py
python -c "import gareus.mbar_analysis.cli"  # confirms subpackage discovery works pre-install
```

## Out Of Scope

- **Plan A2**: relocate data loading (`load_csv`/`load_npz`/`load_parquet*`/
  `clean()`) into `gareus/mbar_analysis/loading.py`.
- **Plan A3**: relocate MBAR solvers and bias reconstruction, reconciling
  with `gareus/query.py`'s `reconstruct_bias_matrix` — including fixing the
  already-documented NaN-guard gap between the two existing copies. Highest
  scientific stakes of the whole sequence; gets its own audit-driven
  verification pass, not a plain code move.
- **Plan A4**: relocate PMF/GaMD cumulant/uncertainty math (including the
  just-shipped bootstrap-uncertainty helpers) into `pmf.py`.
- **Plan A5**: relocate statistical diagnostics, reconciling with
  `gareus/diagnostics.py`'s existing `_hist_overlap_np` family.
- **Plan A6**: relocate trajectory observables (Rg/PCA/SASA/contacts/DSSP/
  Poincaré/chignolin-FES), convergence analysis, and plotting/reporting;
  retire `analyze_gareus_mbar.py`'s direct invocation once everything has
  actually moved.
- Any change to `analyze_gareus_mbar.py`'s own argument parsing, output
  format, or numeric behavior — A1 touches only where two constants are
  *defined*, never how they're used.
- Making `gareus-peptide` (the whole project) `pip install`-able in a way it
  isn't today — out of scope; A1 works within the project's existing
  run-from-repo-root convention.
