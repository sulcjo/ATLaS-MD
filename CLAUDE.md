# Claude Handoff

Updated 2026-07-14.

## US pull-crash recovery (`gareus/seeding.py`)

- Root cause: contact-CV umbrella pull (`generate_us_starting_states_by_pulling` → `relax_to_window`) can NaN a window's positions on the very first, gentlest ramp sub-stage if that window's target contact combination is sterically close to unreachable given where the secondary CV sits. NaN corrupts CUDA addressing → `CUDA_ERROR_ILLEGAL_ADDRESS` → hard process abort, no auto-recovery, whole run dies (observed twice: `adaptive_feedback_round_01` and `adaptive_production/epoch_000`, both at primary-pull ramp stage 1/5 for a contact-CV window).
- Fix, all in `gareus/seeding.py`:
  - `_run_primary_pull_segment` takes `stage_override` (finer contact ramp than `--contact-us-pull-ramp-stages`, used only for crash retries).
  - `relax_to_window` takes `contact_ramp_stage_override`, threaded through to both its internal `_run_primary_pull_segment` calls (staged-2D distance phase and single-stage path).
  - New `_relax_to_window_recovering(sim, w, direction_label, device_idx)` wraps `relax_to_window`: on any exception, rebuilds a fresh `Simulation` (the raising Context may be CUDA-poisoned — never reused), resets to the equilibrated state, and retries once with a 4×-finer contact ramp (`max(16, contact_us_pull_ramp_stages*4)` stages). If the retry also fails, falls back to `_unpulled_window_row` — the plain unpulled equilibrated state for that window only, tagged `direction: pull_crash_fallback_unpulled (...)`, written to `window_..._CRASHFALLBACK_start.pdb`.
  - All 7 call sites of `relax_to_window` (conformer-seeded concurrent/sequential, independent concurrent, nearest/compact_branch/extended_branch/fallback in the sequential walk) now go through `_relax_to_window_recovering`; the 2D row-base pre-pull loop is wrapped in its own try/except that rebuilds the context and skips that row's warm-start state on failure (individual windows still get attempted below).
  - Quality-control loop (`us_starting_structure_quality.json`) flags any `pull_crash_fallback_unpulled` row as `status: bad` with an explicit warning — one bad window can no longer take down the run, but it's never silently marked "ok".
- One bad steric target now costs one degraded window (flagged `bad` in quality report), not the whole umbrella-window seeding run.

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

## Torsion-PCA scree diagnostic (analyze_gareus_mbar.py)

- Added `_analyze_torsion_pca_scree` to `analyze_gareus_mbar.py`: full-spectrum PCA (all components, not just PC1) over the same sin/cos backbone-torsion feature space used by `cv2: torsion-pca`, computed from one adaptive-production epoch's real `tica_obs/dihedral_obs_*.npz` samples (default epoch 0) rather than the GENPEPT seed bank — real sampled dynamics, not generator diversity.
- Outputs land in `<out>/torsion_pca_scree/`: `torsion_pca_scree.png` (per-component + cumulative variance), `torsion_pca_scree_table.csv` (per-PC eigenvalue/EVR/cumulative-EVR plus top-3 loading torsions), `torsion_pca_scree_data.npz` (mean, full eigenvector matrix, eigenvalues, labels — for reuse without recomputing).
- Per-torsion labels are residue-derived (`phi-D3`, `psi-G7`, ...) when the run's sequence length matches the torsion count; falls back to generic `phi_i`/`psi_i` otherwise. Loading magnitude per torsion = `sqrt(w_sin² + w_cos²)` from the eigenvector, not a full angular-span reconstruction (that heavier pseudo-trajectory/PDB-structure version — sweeping each PC from lowest to highest observed seed-bootstrap projection and rebuilding real 3D structures via `GENPEPT.build_structure` — was a one-off ad hoc analysis, not wired into this script; see chat history / `chignolin_2d_run7/torsion_pca_pseudotrajectories/` if regenerating it).
- `tica_obs/dihedral_obs_*.npz` is only flushed to disk at **epoch end** (`DihedralObsBuffer.save`, `gareus/tica.py`) — the analysis returns `available: False` gracefully if run mid-epoch, not a bug.
- Gated by `--no-torsion-pca-scree` (on by default) and `--torsion-pca-scree-epoch N` (default 0). Result merged into the JSON summary under `s['torsion_pca_scree']`, files merged into `s['files']` — same integration pattern as `_analyze_tica_epochs`/`_analyze_epoch_cv_exploration` (no bespoke markdown section).
- Shares the epoch-npz loader with `_analyze_tica_epochs` via the hoisted top-level `_load_epoch_dihedral_features` (previously a private nested closure).
- `--adaptive-diag-stride` default lowered 30→3 (denser adaptive diagnostic density maps by default).

## Main Files

- `GENPEPT.py`
- `gareus/tica.py`
- `gareus/cv.py`
- `gareus/cli.py`
- `gareus/production.py`
- `gareus/adaptive_production.py`
- `gareus/seeding.py`
- `analyze_gareus_mbar.py`
- `gareus/helptext.py`
- `examples/chignolin_runs3.yaml`
- `tests/test_bootstrap_torsion_cv.py`
- `tests/test_tica_cv_mode.py`
- `tests/test_genpept_contact_bias.py`

## Verification

- `pytest -q tests/test_bootstrap_torsion_cv.py tests/test_tica_cv_mode.py tests/test_genpept_contact_bias.py`
- `pytest -q -k seed` (32 seeding-related tests; no dedicated crash-recovery test — the retry/fallback logic lives in closures over a live OpenMM `Simulation`/`Context`, same untestable-without-full-MD-stack constraint as `relax_to_window` itself)
- `python -m py_compile GENPEPT.py gareus/cv.py gareus/tica.py gareus/production.py gareus/adaptive_production.py gareus/cli.py gareus/helptext.py gareus/seeding.py analyze_gareus_mbar.py`
- Parser smoke: `--cv2 torsion-pca`, `--contact-bias-strength 0.15`, `analyze_gareus_mbar.py --no-torsion-pca-scree`, `--torsion-pca-scree-epoch 0`
- Help smoke: `python -m gareus -h` and `python -m gareus -hh`, `python GENPEPT.py -h`
- `git diff --check`
