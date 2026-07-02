# Task 1 Report

## Result

Task 1 completed in the owned surface:
- `gareus/cv.py`
- `gareus/tica.py`
- `tests/test_bootstrap_torsion_cv.py`

## What changed

- Added `torsion-pca` as a canonical secondary CV mode with aliases for `bootstrap-torsion`, `bootstrap_linear`, and `torsion-linear`.
- Set `torsion-pca` range to `(-6.0, 6.0)` and treated it as a transition-style secondary CV.
- Extended `TICAResult` metadata with `method` and `explained_variance_ratio`, and preserved them through save/load.
- Added `compute_bootstrap_torsion_pca(...)` as a PCA fitter that can residualize against `cv1`, returns `TICAResult`, and fail-closes on zero total variance or zero selected-component variance.
- Added focused regression tests for aliases, range, round-trip metadata, residualization, and fail-closed variance handling.

## Verification

- `pytest -q tests/test_bootstrap_torsion_cv.py` -> pass
- `pytest -q tests/test_tica_cv_mode.py` -> pass
- `git diff --check -- gareus/cv.py gareus/tica.py tests/test_bootstrap_torsion_cv.py` -> pass

## Concern

I did not capture the original red failure output directly in this session. The brief indicated the previous worker had partial edits and the controller had already observed the bootstrap test passing on those edits, so the red phase here is reconstructed from current repo state and the task brief rather than from a saved failing test log.
