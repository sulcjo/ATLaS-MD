# Claude Handoff

Updated 2026-07-12.

## GENPEPT contact-count bias

- Added `--contact-bias-strength` (default `0.0`, no behavior change) to `GENPEPT.py`: a Metropolis-style soft acceptance bias applied per-conformer in `generate_one`, on top of the existing energy/steric-only filtering.
- `accept_prob = min(1, exp(strength * (ccount - ref)))`. For `strength > 0`, `ref` is the combinatorial max `contact_count` for the sequence/`--contact-min-sep` (so the best-possible conformer is always accepted, lower-contact ones progressively suppressed); for `strength < 0`, `ref = 0` (mirrored, favors fewer contacts).
- Motivation: GENPEPT's per-residue Ramachandran sampling has no compactness/contact bias anywhere (generation, basin-hop, NMA) — confirmed by an A/B test on chignolin (`chignolin_genpept_r3` vs `_r4`, see project memory) where even switching to a hairpin-tuned `diversity_bank_preset` only modestly enriched near-native density without shifting the dominant (open) mode, because per-residue sampling stays uncorrelated across residues.
- `ref` is a loose combinatorial ceiling, often far above what's actually reachable — start with small `strength` (0.05-0.3). Values near/above 1.0 can crush acceptance to near-zero and exhaust `--n` before enough candidates are found (reproduced directly: `strength=1.0` on chignolin with `--n 4000` raised `RuntimeError: Only 0 all-atom-valid candidate seeds found`; `strength=0.15` with `--n 8000` worked and shifted mean `contact_count` from 3.92 to 6.01 on a 300-candidate pool).
- Applied consistently in both `generate_candidates` (initial pool) and `generate_adaptive_frontier_proposals` (explore-loop proposals) via the shared `GenConfig.contact_bias_strength` field, so explore-loop rounds don't silently bypass the bias.

## Torsion-PCA CV2

- Added `cv2: torsion-pca` as first-run bootstrap secondary CV for peptide runs.
- Intended first-run pair: `cv1: contacts`, `cv2: torsion-pca`.
- `torsion-pca` fits backbone torsion PCA from GENPEPT seed conformers. Residualization against seed CV1 is now OFF by default: for a folder the fold is correlated torsion+contact motion, so residualizing strips the fold signal and collapses PC1, cramming CV2 window centers into a thin band. Opt in only when an orthogonal-to-CV1 secondary motion is genuinely wanted. CV2 seed centers span the full seed PC1 distribution (2/98% quantiles).
- Runtime force uses same raw-feature linear torsion projection as `tica-linear`.
- `tica_switch_cv2: true` switches only after a successful tICA update writes a valid tICA state file.
- Fast resume restores linear torsion state paths from secondary-CV metadata and fails closed if missing.

## Main Files

- `GENPEPT.py`
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
- `tests/test_genpept_contact_bias.py`

## Verification

- `pytest -q tests/test_bootstrap_torsion_cv.py tests/test_tica_cv_mode.py tests/test_genpept_contact_bias.py`
- `python -m py_compile GENPEPT.py gareus/cv.py gareus/tica.py gareus/production.py gareus/adaptive_production.py gareus/cli.py gareus/helptext.py analyze_gareus_mbar.py`
- Parser smoke: `--cv2 torsion-pca`, `--contact-bias-strength 0.15`
- Help smoke: `python -m gareus -h` and `python -m gareus -hh`, `python GENPEPT.py -h`
- `git diff --check`
