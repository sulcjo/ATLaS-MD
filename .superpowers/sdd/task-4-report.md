# Task 4 Report

## Result

Task 4 completed in the owned surface:
- `tests/test_tica_cv_mode.py`
- `analyze_gareus_mbar.py`
- `gareus/helptext.py`
- `examples/chignolin_runs3.yaml`

## What changed

- Added adaptive-handoff regression coverage for `torsion-pca -> tica-linear`.
- Added MBAR-guard regression coverage for the bootstrap-disabled epoch split after the handoff.
- Updated MBAR analysis labels so `torsion-pca` and `tica-linear` stay distinct in metadata-driven plots/reports:
  - `Bootstrap torsion PC1`
  - `tIC1 torsion CV`
- Extended CLI heavy/simple help text to list `torsion-pca`, explain the bootstrap PCA stage, and document the recommended two-stage YAML setup.
- Updated `examples/chignolin_runs3.yaml` to start from `cv2: torsion-pca`, add a `bootstrap_torsion_cv:` block, and add the `tica:` handoff block while preserving unrelated structure and values.

## RED evidence

Test-first note:
- The two new regression tests were added before production-surface edits, but they passed immediately.
- This matches the brief: Tasks 1-3 already implemented the core handoff/versioning behavior, so Task 4 needed to lock that behavior and document/expose it rather than change default logic.

Pre-change missing user-facing coverage from `HEAD`:
- `git show HEAD:analyze_gareus_mbar.py | rg -n 'Bootstrap torsion PC1|tIC1 torsion CV|torsion-pca'` -> no matches
- `git show HEAD:gareus/helptext.py | rg -n 'torsion-pca|Bootstrap torsion PCA'` -> no matches
- `git show HEAD:examples/chignolin_runs3.yaml | rg -n 'cv2: torsion-pca|bootstrap_torsion_cv:|tica_switch_cv2'` -> no matches

New regression tests added first:
- `tests/test_tica_cv_mode.py::TestCV2AutoSwitch::test_switch_logic_changes_torsion_pca_to_tica_linear`
- `tests/test_tica_cv_mode.py::TestMBARGuard::test_epoch_sample_sources_excludes_bootstrap_disabled_epoch_when_tica_active`

## GREEN evidence

- `pytest -q tests/test_tica_cv_mode.py::TestCV2AutoSwitch::test_switch_logic_changes_torsion_pca_to_tica_linear` -> pass
- `pytest -q tests/test_tica_cv_mode.py::TestMBARGuard::test_epoch_sample_sources_excludes_bootstrap_disabled_epoch_when_tica_active` -> pass
- `pytest -q tests/test_tica_cv_mode.py` -> `29 passed`
- `python -m py_compile analyze_gareus_mbar.py` -> pass
- `python -m py_compile gareus/helptext.py` -> pass
- `git diff --check` -> pass

## Constraints check

- No default behavior changed.
- `torsion-pca` and `tica-linear` remain distinct in metadata labels and analysis text.
- Existing `tica_switch_cv2` one-shot behavior preserved and explicitly regression-tested.
- Resume semantics unchanged; no edits outside the owned files.
- Dirty scratch file `.superpowers/sdd/task-1-report.md` left untouched for staging.
