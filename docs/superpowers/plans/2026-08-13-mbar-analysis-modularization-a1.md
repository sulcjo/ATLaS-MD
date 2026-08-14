# MBAR Analysis Modularization — Plan A1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reconcile the duplicated `KJ_PER_KCAL`/`K_B_KJ_PER_MOL_K` physical constants into `gareus/units.py`, and create a new `gareus/mbar_analysis/` subpackage with a delegating CLI entry point (`gareus-analyze`), as the first of a 6-plan sequence folding `analyze_gareus_mbar.py` into the `gareus` package.

**Architecture:** `gareus/units.py` gains named constants that its own existing functions are refactored to use; `analyze_gareus_mbar.py` imports those constants instead of redefining them. A new `gareus/mbar_analysis/` subpackage gets a `cli.py` whose `main()` is a pure delegation to `analyze_gareus_mbar.main()` (a strangler-fig placeholder — later plans move real logic behind it). `pyproject.toml` registers `gareus-analyze` as a console script, matching the project's existing 5-script convention.

**Tech Stack:** Python 3.10+, pytest, setuptools (`pyproject.toml` `[project.scripts]`).

**Spec:** `docs/superpowers/specs/2026-08-13-mbar-analysis-modularization-a1-design.md`

## Global Constraints

- `KJ_PER_KCAL = 4.184` and `K_B_KJ_PER_MOL_K = 0.00831446261815324` are the exact values already in use — do not change them, only relocate their definition.
- The old invocation (`python analyze_gareus_mbar.py <run_dir> ...`) must keep working unchanged for the entire A1-A5 transition — this plan is strictly additive, never a replacement.
- No change to `analyze_gareus_mbar.py`'s argument parsing, output format, or numeric behavior — only where two constants are *defined*.
- New subpackage `gareus/mbar_analysis/` is auto-discovered by `pyproject.toml`'s existing `[tool.setuptools.packages.find] include = ["gareus*"]` — no packaging-config change needed for subpackage discovery itself.
- No new dependency-extras group: `pyproject.toml`'s existing `analysis` extra (`pymbar>=4, pyarrow>=12, duckdb>=0.9, scipy>=1.10`) already covers what this subpackage needs.

---

### Task 1: Reconcile physical constants in `gareus/units.py`

**Files:**
- Modify: `gareus/units.py` (all 46 lines — add 2 constants, refactor 3 functions to reference them)
- Test: `tests/test_gareus_units_constants.py` (new file)

**Interfaces:**
- Produces: `gareus.units.KJ_PER_KCAL` (float, `4.184`), `gareus.units.K_B_KJ_PER_MOL_K` (float, `0.00831446261815324`) — module-level constants consumed by Task 2.

- [ ] **Step 1: Write the failing test**

Create `tests/test_gareus_units_constants.py`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gareus.units import (
    KJ_PER_KCAL,
    K_B_KJ_PER_MOL_K,
    kcal_a2_to_kj_nm2,
    kj_nm2_to_kcal_a2,
    kcal_to_kj,
    bias_energy_kj,
)


def test_kj_per_kcal_is_the_exact_conversion_factor():
    assert KJ_PER_KCAL == 4.184


def test_k_b_matches_codata_value():
    assert K_B_KJ_PER_MOL_K == 0.00831446261815324


def test_k_b_and_kj_per_kcal_round_trip_kbt_at_300k():
    # Same check this codebase already validated numerically elsewhere this
    # session: at T=300K, kT should be ~0.596 kcal/mol.
    beta = 1.0 / (K_B_KJ_PER_MOL_K * 300.0)
    kbt_kj = 1.0 / beta
    kbt_kcal = kbt_kj / KJ_PER_KCAL
    assert abs(kbt_kcal - 0.5961612775922496) < 1e-9


def test_conversion_functions_unchanged_after_refactor():
    # Regression guard: refactoring these to reference the named constant
    # instead of an inline literal must not change any output.
    assert kcal_a2_to_kj_nm2(5.0) == 5.0 * 4.184 / 0.01
    assert kj_nm2_to_kcal_a2(2092.0) == 2092.0 * 0.01 / 4.184
    assert kcal_to_kj(1.0) == 4.184
    assert bias_energy_kj(2.5) == kcal_to_kj(2.5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_gareus_units_constants.py -v`
Expected: FAIL with `ImportError: cannot import name 'KJ_PER_KCAL' from 'gareus.units'`

- [ ] **Step 3: Write minimal implementation**

Replace the full contents of `gareus/units.py` with:

```python
"""
Unit conversion helpers.

These functions convert between the unconventional kcal/mol/Å² units used
by some umbrella sampling inputs and the canonical kJ/mol/nm² units.
They were originally defined in ``gareus_peptide.py``.
"""

KJ_PER_KCAL = 4.184
K_B_KJ_PER_MOL_K = 0.00831446261815324  # Boltzmann constant, kJ/(mol*K)


def kcal_a2_to_kj_nm2(k_kcal_a2: float) -> float:
    """Convert a force constant from kcal/mol/Å² to kJ/mol/nm².

    KJ_PER_KCAL converts kcal to kJ, and 0.01 converts Å² to nm².

    Args:
        k_kcal_a2: Force constant in kcal/mol/Å².

    Returns:
        The corresponding force constant in kJ/mol/nm².
    """
    return float(k_kcal_a2) * KJ_PER_KCAL / 0.01


def kj_nm2_to_kcal_a2(k_kj_nm2: float) -> float:
    """Convert a force constant from kJ/mol/nm² to kcal/mol/Å²."""
    return float(k_kj_nm2) * 0.01 / KJ_PER_KCAL


def kcal_to_kj(k_kcal: float) -> float:
    """Convert an energy from kcal/mol to kJ/mol."""
    return float(k_kcal) * KJ_PER_KCAL


def bias_energy_kj(bias_kcal: float) -> float:
    """Convert a bias energy from kcal/mol to kJ/mol.

    Bias energies are often specified in kcal/mol.  Converting to kJ/mol
    ensures consistency with OpenMM, which uses kJ by default.
    """
    return kcal_to_kj(bias_kcal)


__all__ = [
    "KJ_PER_KCAL",
    "K_B_KJ_PER_MOL_K",
    "kcal_a2_to_kj_nm2",
    "kj_nm2_to_kcal_a2",
    "kcal_to_kj",
    "bias_energy_kj",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_gareus_units_constants.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Confirm every real consumer of `gareus.units` still works**

`gareus/units.py` is load-bearing for the live simulation pipeline, not just
this analysis effort — confirmed via `grep -rn "from \.units import\|from gareus\.units import"`
across the repo, these files import from it: `gareus/forces.py` (`kcal_to_kj`),
`gareus/legacy.py` (`from .units import *` — a wildcard import, so it will
also newly expose `KJ_PER_KCAL`/`K_B_KJ_PER_MOL_K`; confirmed via
`grep -n "KJ_PER_KCAL\|K_B_KJ_PER_MOL_K" gareus/legacy.py` that neither name
already exists there, so this is a pure addition, not a collision),
`gareus/cv.py`, `gareus/system_setup.py` (`kcal_a2_to_kj_nm2`),
`gareus/seeding.py` (`kj_nm2_to_kcal_a2`), `gareus/production.py`
(`kcal_to_kj`, `kcal_a2_to_kj_nm2`), plus three real-OpenMM test files
(`tests/test_thermodynamic_validity_2d.py`, `tests/test_thermodynamic_validity_2d_rough.py`,
`tests/test_thermodynamic_validity_real_md.py`, all importing
`kcal_a2_to_kj_nm2`/`kcal_to_kj`). Every one of these imports an existing
function by its unchanged name — this refactor changes only what those
functions reference internally, not their names, signatures, or return
values — so a full re-run of the slow real-MD tests is not warranted here.
Instead, confirm import-time correctness cheaply:

Run: `python3 -c "import gareus.forces, gareus.legacy, gareus.cv, gareus.system_setup, gareus.seeding, gareus.production"`
Expected: no exception (proves every real consumer's import statement still
resolves against the refactored `gareus/units.py`).

- [ ] **Step 6: Commit**

```bash
git add gareus/units.py tests/test_gareus_units_constants.py
git commit -m "refactor: add named KJ_PER_KCAL/K_B_KJ_PER_MOL_K constants to gareus.units"
```

---

### Task 2: Point `analyze_gareus_mbar.py` at `gareus.units` for its constants

**Files:**
- Modify: `analyze_gareus_mbar.py:17,43-44` (exact current lines — re-confirm with `grep -n "^K_B_KJ_PER_MOL_K\|^KJ_PER_KCAL\|^import numpy"` before editing, since earlier lines could have shifted if another change landed on `main` first)
- Test: `tests/test_analyze_gareus_mbar_uses_shared_units.py` (new file)

**Interfaces:**
- Consumes: `gareus.units.KJ_PER_KCAL`, `gareus.units.K_B_KJ_PER_MOL_K` (Task 1).
- Produces: `analyze_gareus_mbar.KJ_PER_KCAL` and `analyze_gareus_mbar.K_B_KJ_PER_MOL_K` still exist as module attributes (unchanged name, now imported rather than locally defined) — every one of the file's ~50+ existing use sites needs zero changes.

- [ ] **Step 1: Write the failing test**

Create `tests/test_analyze_gareus_mbar_uses_shared_units.py`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gareus.units
import analyze_gareus_mbar as agm


def test_analyze_gareus_mbar_kj_per_kcal_is_the_shared_constant():
    # `is`, not `==` -- proves the import actually replaced the local
    # definition rather than merely producing a coincidentally-equal copy.
    assert agm.KJ_PER_KCAL is gareus.units.KJ_PER_KCAL


def test_analyze_gareus_mbar_k_b_is_the_shared_constant():
    assert agm.K_B_KJ_PER_MOL_K is gareus.units.K_B_KJ_PER_MOL_K


def test_values_are_unchanged():
    assert agm.KJ_PER_KCAL == 4.184
    assert agm.K_B_KJ_PER_MOL_K == 0.00831446261815324
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_analyze_gareus_mbar_uses_shared_units.py -v`
Expected: FAIL on the `is` assertions — `agm.KJ_PER_KCAL == gareus.units.KJ_PER_KCAL` is `True` (same float value) but `is` is `False` (two independently-created float objects from two separate literal definitions), since at this point `analyze_gareus_mbar.py` still defines its own copy.

- [ ] **Step 3: Write minimal implementation**

In `analyze_gareus_mbar.py`, first re-locate the exact current lines:

Run: `grep -n "^K_B_KJ_PER_MOL_K\|^KJ_PER_KCAL\|^import numpy as np" analyze_gareus_mbar.py`

This should show `import numpy as np` at one line, and the two constant definitions a short distance below it (there are several `try:`/`except:` blocks for optional dependencies — numba, scipy — in between; leave those untouched). Make two edits:

1. Immediately after the `import numpy as np` line, add:

```python
from gareus.units import KJ_PER_KCAL, K_B_KJ_PER_MOL_K
```

2. Delete the two lines:

```python
K_B_KJ_PER_MOL_K=0.00831446261815324
KJ_PER_KCAL=4.184
```

(Do not reformat surrounding code, do not touch the numba/scipy optional-import try/except blocks that sit between the new import and the deleted lines — leave their position exactly as-is relative to what remains.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_analyze_gareus_mbar_uses_shared_units.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Confirm the rest of the file still works**

Run: `python -m py_compile analyze_gareus_mbar.py`
Expected: clean compile, no output.

Run: `pytest -q tests/ --ignore=tests/test_validation_common.py -k "not physics_oracle and not thermodynamic_validity"`
Expected: same result as on `main` before this change — the 6 known pre-existing, unrelated failures (help-text encyclopedia numbering, threadpool-import check, missing validation launcher fixtures), no new failures. This plan only changes where two constants are *defined*, never their value, so no numeric test should be able to detect this change at all — if anything beyond those 6 fails, stop and investigate before continuing (most likely cause: the constant-definition lines moved from where `grep` found them, or something else in the file references `K_B_KJ_PER_MOL_K`/`KJ_PER_KCAL` as a local name assignment target rather than a read, which `grep -n "K_B_KJ_PER_MOL_K\s*="` across the whole file would reveal).

- [ ] **Step 6: Commit**

```bash
git add analyze_gareus_mbar.py tests/test_analyze_gareus_mbar_uses_shared_units.py
git commit -m "refactor: import KJ_PER_KCAL/K_B_KJ_PER_MOL_K from gareus.units instead of redefining"
```

---

### Task 3: Create the `gareus/mbar_analysis/` subpackage with a delegating CLI

**Files:**
- Create: `gareus/mbar_analysis/__init__.py`
- Create: `gareus/mbar_analysis/cli.py`
- Test: `tests/test_mbar_analysis_scaffolding.py` (new file)

**Interfaces:**
- Consumes: `analyze_gareus_mbar.main(argv)` (pre-existing, unchanged).
- Produces: `gareus.mbar_analysis.cli.main(argv=None)` — used by Task 4's console-script registration.

- [ ] **Step 1: Write the failing test**

Create `tests/test_mbar_analysis_scaffolding.py`:

```python
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_mbar_analysis_package_is_importable():
    import gareus.mbar_analysis  # noqa: F401


def test_cli_main_is_callable_with_no_args_signature():
    from gareus.mbar_analysis.cli import main
    import inspect
    sig = inspect.signature(main)
    assert list(sig.parameters) == ["argv"]
    assert sig.parameters["argv"].default is None


def test_cli_help_output_matches_analyze_gareus_mbar_help_output():
    # Run both as real subprocesses (not in-process calls) since --help
    # triggers SystemExit, which is awkward to capture reliably in-process
    # across two different entry modules in the same test process.
    new_entry = subprocess.run(
        [sys.executable, "-c", "from gareus.mbar_analysis.cli import main; main(['--help'])"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    old_entry = subprocess.run(
        [sys.executable, "analyze_gareus_mbar.py", "--help"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert new_entry.stdout == old_entry.stdout
    assert new_entry.returncode == old_entry.returncode
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_mbar_analysis_scaffolding.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gareus.mbar_analysis'`

- [ ] **Step 3: Write minimal implementation**

Create `gareus/mbar_analysis/__init__.py`:

```python
"""MBAR/PMF post-hoc analysis subpackage.

Currently a strangler-fig shell: `cli.py` delegates to the top-level
`analyze_gareus_mbar.py` script, which still holds all real analysis logic.
Later plans in the same modularization sequence (see
docs/superpowers/specs/2026-08-13-mbar-analysis-modularization-a1-design.md)
progressively move that logic into this subpackage.
"""
```

Create `gareus/mbar_analysis/cli.py`:

```python
"""Delegating CLI entry point for `gareus-analyze`.

Strangler-fig placeholder: owns no analysis logic yet. Later plans in this
modularization sequence progressively move analyze_gareus_mbar.py's real
logic into this subpackage; until then this module's only job is to make
`gareus-analyze` behave identically to `python analyze_gareus_mbar.py`.
"""
from __future__ import annotations

from typing import Optional, Sequence


def main(argv: Optional[Sequence[str]] = None) -> None:
    import analyze_gareus_mbar
    analyze_gareus_mbar.main(argv)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_mbar_analysis_scaffolding.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add gareus/mbar_analysis/__init__.py gareus/mbar_analysis/cli.py tests/test_mbar_analysis_scaffolding.py
git commit -m "feat: add gareus.mbar_analysis subpackage with delegating CLI entry point"
```

---

### Task 4: Register the `gareus-analyze` console script

**Files:**
- Modify: `pyproject.toml:47-53` (the `[project.scripts]` table — re-confirm the exact line range with `grep -n "project.scripts" pyproject.toml` before editing)
- Test: `tests/test_pyproject_console_scripts.py` (new file)

**Interfaces:**
- Consumes: `gareus.mbar_analysis.cli:main` (Task 3).
- Produces: a `gareus-analyze` console-script declaration other tooling (packaging, CI, docs) can rely on existing.

- [ ] **Step 1: Write the failing test**

Create `tests/test_pyproject_console_scripts.py`:

```python
import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # dev dependency already covers this via pytest's own toolchain; if unavailable, this test documents the requirement

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_gareus_analyze_console_script_is_registered():
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    scripts = data["project"]["scripts"]
    assert scripts["gareus-analyze"] == "gareus.mbar_analysis.cli:main"


def test_existing_console_scripts_are_unchanged():
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    scripts = data["project"]["scripts"]
    assert scripts["gareus"] == "gareus.core:main"
    assert scripts["gareus-peptide"] == "gareus.core:main"
    assert scripts["gareus-energy-decompose"] == "gareus.energy_decomposition:main"
    assert scripts["gareus-test-run"] == "gareus.integration_test:main"
    assert scripts["gareus-suggest-cvs"] == "gareus.cv_discovery:main"
    assert scripts["gareus-consolidate-traj"] == "gareus.traj_consolidate:main"
```

If `tomllib`/`tomli` genuinely isn't available in this environment (`python --version` below 3.11 and `tomli` not installed), fall back to a plain text-containment check instead of a real TOML parse:

```python
def test_gareus_analyze_console_script_is_registered_fallback():
    content = (REPO_ROOT / "pyproject.toml").read_text()
    assert 'gareus-analyze = "gareus.mbar_analysis.cli:main"' in content
```

Use whichever of the two approaches actually runs in this environment — check `python3 --version` and whether `import tomllib` or `import tomli` succeeds before writing the final test file, and delete the approach that doesn't apply rather than shipping both.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pyproject_console_scripts.py -v`
Expected: FAIL — `KeyError: 'gareus-analyze'` (TOML approach) or a plain assertion failure (fallback approach), since the line doesn't exist yet.

- [ ] **Step 3: Write minimal implementation**

In `pyproject.toml`, find the `[project.scripts]` table (`grep -n "project.scripts" pyproject.toml`) and add one line to it, alongside the existing five:

```toml
[project.scripts]
gareus-peptide = "gareus.core:main"
gareus = "gareus.core:main"
gareus-energy-decompose = "gareus.energy_decomposition:main"
gareus-test-run = "gareus.integration_test:main"
gareus-suggest-cvs = "gareus.cv_discovery:main"
gareus-consolidate-traj = "gareus.traj_consolidate:main"
gareus-analyze = "gareus.mbar_analysis.cli:main"
```

(Only the last line is new — the six lines above it already exist and must not be reordered or altered.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pyproject_console_scripts.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Confirm the TOML file is still well-formed**

Run: `python3 -c "import tomllib; tomllib.load(open('pyproject.toml','rb'))"` (or the `tomli` equivalent per whichever import path Step 1 used)
Expected: no exception.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml tests/test_pyproject_console_scripts.py
git commit -m "feat: register gareus-analyze console script entry point"
```

---

## Plan Self-Review Notes

- **Spec coverage**: constants reconciliation (Tasks 1-2), subpackage + delegating entry point (Task 3), console-script registration (Task 4) all covered. The spec's explicit non-goals (moving real analysis logic, changing argument parsing/output, retiring the old invocation) are correctly left untouched by every task — none of the four tasks edits anything inside `analyze_gareus_mbar.py` except the two-line constant-definition swap in Task 2.
- **Type/signature consistency checked**: `gareus.mbar_analysis.cli.main(argv=None)`'s signature (Task 3) matches what Task 4's console-script entry (`gareus.mbar_analysis.cli:main`) expects setuptools to find; the `is`-identity tests in Task 2 depend on Task 1 actually being a plain `from ... import ...` (not a re-assignment or a `copy.copy`), which the Task 1 implementation code satisfies.
- **No placeholders**: every step has complete, runnable code; the one conditional branch (Task 4's tomllib-vs-tomli fallback) gives complete code for both paths and explicit instructions to pick one, not a vague "handle appropriately."
