# Claude Handoff

Updated 2026-07-16.

## `--sigma0d` is a no-op for single-boost GaMD types (only `--sigma0p` matters)

- Root cause of a real finding on the chignolin quicktest run: its `gamd:` section set `sigma0d: 2.0` under `boost_type: lower-dihedral`, expecting that to control the boost target. It does nothing there. Traced into the upstream `gamd` package (`gamd/integrator_factory.py`): `get_integrator`'s `"lower-dihedral"` branch (and every other single-boost type) calls `create_lower_dihedral_boost_integrator(..., sigma0p)` — passing only `sigma0p`. That function's signature is `(..., sigma0=6.0*kcal)`; `sigma0d` isn't even a parameter. `sigma0d` only reaches anything for the *dual*-boost integrators (`lower-dual`, `upper-dual`, and their `*-nonbonded-dihedral` variants), which do take both.
- Confirmed in the quicktest run's own recorded state: `run_args.json` had `sigma0d_kcal_mol: 2.0` (set) and `sigma0p_kcal_mol: 6.0` (untouched default), and `global_shared_gamd_setup/shared_gamd_setup_globals.json` shows the calibration actually targeted `sigma0_Dihedral: 25.104 kJ/mol` — exactly 6.0 kcal/mol, i.e. sigma0p, not the intended 2.0.
- Consequence: 6.0 kcal/mol was too large a target for DPETG's dihedral-energy dynamic range (calibrated `Vmax-Vmin ≈ 129 kJ/mol` for a 5-residue/8-torsion system). `k0` saturated at its ceiling (1.0) and stayed there even after the epoch-0 real-sampling recalibration (`recalibration_history`: `k0_before == k0_after == 1.0`) — a structural ceiling, not calibration noise — achieving only `sigmaV ≈ 11.5-11.7 kJ/mol`, under half the target. A maxed-out, unmoderated boost applied to DPETG's fast multi-basin flickering (Poincaré map: median recurrence 0.1-0.2 ns) produced a strongly non-Gaussian boost distribution: `anharmonicity_score = 1.11` (skew 0.96, excess kurtosis 1.13), flagged HIGH by the pipeline's own PMF analysis (`analyze_gareus_mbar.py`, threshold >1.0).
- Fix: `gareus/cli.py`'s `_validate_gamd_args` (new, called from `parse_args` alongside `_validate_contact_args`) warns when `--sigma0d` is set to a non-default value but `--gamd-boost-type` is one of the single-boost types (`gamd-cmd-base`, `lower-total`, `upper-total`, `lower-dihedral`, `upper-dihedral`, `lower-nonbonded`, `upper-nonbonded`) — so this exact mistake surfaces immediately instead of silently mis-calibrating. Tests: `tests/test_gamd_boost_default.py` (warns for all 7 single-boost types with `sigma0d` set, doesn't warn for the 4 dual-boost types or when `sigma0d` is left at default).
- `RUNS/chignolin/chignolin_quicktest.yaml` fixed: `sigma0d: 2.0` → `sigma0p: 2.0` (the parameter that actually governs `boost_type: lower-dihedral`).
- **Not yet fixed**: `RUNS/chignolin/chignolin.yaml` (the real 10-residue production config) has the *identical* pattern — `gamd: sigma0d: 2.0` under `boost_type: lower-dihedral`, `sigma0p` unset (defaults to 6.0). Every past production run using this config (chignolin_2d_run5/7 etc.) was very likely calibrated against the same accidental 6.0 kcal/mol sigma0p target, not the intended 2.0. Left untouched pending a decision — chignolin's dihedral-energy dynamic range (more torsions, bigger system) may or may not actually saturate at 6.0 the way DPETG's did; that needs its own calibration data before picking a new number, and retroactively changing it affects comparability with prior runs.

## Minimization step-count defaults raised to >= 1000

Every "how many minimization steps by default" argparse default under 1000 raised to 1000 — these were all short defaults tuned for a bigger/older workflow, not a hard physical requirement:

- `GENPEPT.py`: `--implicit-max-iterations` 300→1000, `--explicit-max-iterations` 500→1000, `--bh-initial-min-iterations` 200→1000, `--bh-min-iterations` 150→1000 (basin-hop minimization, part of GENPEPT seed generation).
- `gareus/cli.py`: `--us-pull-minimize-iterations` 100→1000 — this one governs *both* the graft-into-solvated-context clash minimization (`graft_conformer_into_context` in `gareus/seeding.py`, used when seeding umbrella windows from GENPEPT survivors) and the umbrella-pull-ramp minimizations (`relax_to_window`'s `_run_primary_pull_segment` calls) — "for seeds" and "in pulling" both trace back to this one knob.
- `gareus/cli.py`'s `--minimize-iterations` (main NVT-stage system minimization, `gareus/system_setup.py`) was already 20000 — untouched.

**Deliberate exceptions, not touched:**
- `GENPEPT.py --tier-scout-iterations` (default 25) — the scout tier of `--tiered-implicit-min` (off by default) is *designed* to be a cheap fast pre-filter before refining a smaller subset; forcing it to 1000 would make scout as expensive as refine and defeat the tier split's purpose.
- `GENPEPT.py --adaptive-min-chunk-iterations` (default 25) — a convergence-check chunk *granularity* inside the adaptive-minimization loop, not a total step count; the actual total is still governed by `--implicit-max-iterations`/`--explicit-max-iterations` above.
- `GENPEPT.py --tier-refine-iterations` (default 0) — `0` is a sentinel meaning "fall back to `--implicit-max-iterations`," not literally zero steps.
- `gareus/seeding.py`'s first-ramp-stage clash minimization (`relax_to_window`'s `_run_primary_pull_segment`, the exact stage the pull-crash-recovery logic guards) was hardcoded `min(minimize_iters, 25)` regardless of `--us-pull-minimize-iterations`. Raised: now just `minimize_iters` (uncapped), matching every other minimization call site in that function — a longer clash-relax before the ramp's first dynamics steps is a stabilizing change, not a destabilizing one, so this doesn't fight the crash-recovery mechanism.
- These are argparse *default* changes only, not a runtime floor/clamp — a YAML config or CLI flag can still explicitly set a lower value (e.g. `RUNS/chignolin/chignolin_quicktest.yaml` intentionally uses smaller values for fast local smoke tests; those overrides are untouched and still apply).
- No existing test asserted the old default values, so no test changes were needed for this one.

## Torsion-PCA CV2 — `bootstrap_torsion_component` is a count, not an index

- `compute_bootstrap_torsion_pca` (`gareus/tica.py`) previously took `component` as a 1-based index selecting a *single* PCA eigenvector as the `cv2: torsion-pca` direction. Changed: `component` is now a **count** — the top N components (by variance, ranked 1..N) are combined into one unit-norm direction via a variance-weighted linear combination, `coefficient_i = sqrt(variance_i)` then renormalized. CV2 stays a single scalar restraint either way; this just lets that one restraint draw on more than PC1 alone.
- `component=1` is exactly backward compatible: a single positive scalar coefficient doesn't change a normalized direction's identity, so it reduces to the old "PC1 alone" behavior bit-for-bit.
- Default changed `1` → `5` (`--bootstrap-torsion-component`, `gareus/cli.py`; same default fallback in `gareus/production.py`'s `getattr`).
- Important asymmetry: PC1 is *by construction* the single direction of maximum variance, so combining in more components can only match or **reduce** that one direction's own explained variance (`TICAResult.eigenvalue`/`explained_variance_ratio`) relative to PC1 alone — the payoff isn't more variance, it's picking up influence from a mode PC1 alone doesn't see (e.g. a torsion that only loads heavily on PC4).
- Tests: `tests/test_bootstrap_torsion_cv.py::test_bootstrap_pca_component_count_combines_top_n_variance_weighted` verifies the coefficient ratio matches `sqrt(variance_i/variance_j)` exactly and that combined-direction variance is strictly ≤ PC1-alone variance; `test_bootstrap_pca_component_out_of_range_raises` covers the bounds check. `test_bootstrap_pca_fail_closed_on_zero_component_variance` updated for the new semantics (old test's X made requesting component=2 trivially succeed under the new scheme, since PC1 there still carried real variance — not a bug, just no longer a failure case).
- Any cached `<out>/tica/bootstrap_torsion_cv.json` from a run predating this change holds a single raw eigenvector under the old semantics; it's loaded as-is if present (`_ensure_bootstrap_torsion_cv_ready` in `gareus/production.py`), so changing `bootstrap_torsion_component` on a resumed run does nothing until that cache file is deleted and window/seed setup is redone.

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
- `tests/test_gamd_boost_default.py`

## Verification

- `pytest -q tests/test_bootstrap_torsion_cv.py tests/test_tica_cv_mode.py tests/test_genpept_contact_bias.py tests/test_gamd_boost_default.py`
- `pytest -q -k seed` (32+ seeding-related tests; no dedicated crash-recovery test — the retry/fallback logic lives in closures over a live OpenMM `Simulation`/`Context`, same untestable-without-full-MD-stack constraint as `relax_to_window` itself)
- `python -m py_compile GENPEPT.py gareus/cv.py gareus/tica.py gareus/production.py gareus/adaptive_production.py gareus/cli.py gareus/helptext.py gareus/seeding.py analyze_gareus_mbar.py`
- Parser smoke: `--cv2 torsion-pca`, `--contact-bias-strength 0.15`, `analyze_gareus_mbar.py --no-torsion-pca-scree`, `--torsion-pca-scree-epoch 0`
- Help smoke: `python -m gareus -h` and `python -m gareus -hh`, `python GENPEPT.py -h`
- `git diff --check`
