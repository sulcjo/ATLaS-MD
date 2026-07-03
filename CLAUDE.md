# Claude Handoff

Updated 2026-07-03.

## Torsion-PCA CV2

- Added `cv2: torsion-pca` as first-run bootstrap secondary CV for peptide runs.
- Intended first-run pair: `cv1: contacts`, `cv2: torsion-pca`.
- `torsion-pca` fits backbone torsion PCA from GENPEPT seed conformers, residualized against seed CV1 by default.
- Runtime force uses same raw-feature linear torsion projection as `tica-linear`.
- `tica_switch_cv2: true` switches only after a successful tICA update writes a valid tICA state file.
- Fast resume restores linear torsion state paths from secondary-CV metadata and fails closed if missing.

## Main Files

- `gareus/tica.py`
- `gareus/cv.py`
- `gareus/cli.py`
- `gareus/production.py`
- `gareus/adaptive_production.py`
- `analyze_gareus_mbar.py`
- `gareus/helptext.py`
- `examples/chignolin_runs3.yaml`
- `tests/test_bootstrap_torsion_cv.py`
- `tests/test_tica_cv_mode.py`

## Verification

- `pytest -q tests/test_bootstrap_torsion_cv.py tests/test_tica_cv_mode.py`
- `python -m py_compile gareus/cv.py gareus/tica.py gareus/production.py gareus/adaptive_production.py gareus/cli.py gareus/helptext.py analyze_gareus_mbar.py`
- Parser smoke: `--cv2 torsion-pca`
- Help smoke: `python -m gareus -h` and `python -m gareus -hh`
- `git diff --check`
