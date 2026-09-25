# Changelog

All notable changes to ATLaS-MD are documented here.

The project follows semantic-style release numbering where practical. Research-method changes that alter a sampled Hamiltonian, estimator, output contract, or thermodynamic assumption should be called out explicitly even when backwards compatibility is retained.

## [Unreleased]

### Added

- **Adaptive top-ups (`--ap-topups`, off by default).** A scheduled phase (numbered epoch or final) can now follow its all-state baseline with at most one lockstep top-up segment over the states a per-epoch union-MBAR solve finds deficient (local sigma above `--ap-topup-target-sigma`, or a per-state local split-halves drift test), plus their layout partners. The allocator picks one step length that brings the worst deficit to target, capped by `--ap-topup-max-fraction` of the phase's wall-hour budget; a per-state calibration correction learned from realised-vs-predicted sigma is clamped to [0.1, 2.0]. With top-ups on, each phase's baseline is shortened by that fraction to fund the top-up; in the final phase, when the top-up does not run (anything but completed or pool-skipped), the baseline is resumed to its full length so the withheld budget is not stranded. New knobs: `--ap-topup-target-sigma` (0.10 kcal/mol), `--ap-topup-weak-overlap` (0.15), `--ap-topup-max-fraction` (0.3), `--ap-topup-min-effect` (0.05 kcal/mol), `--ap-topup-max-edge-attempts` (2), `--ap-topup-throughput-table`, `--ap-topup-diagnostics-max-gb` (8.0).
- **Top-up seeding continues each window from its own parent segment.** With top-ups on, every adaptive-production segment (baseline or top-up) exports a `final_window_states/` directory (top-ups off, or outside adaptive production: nothing is written — ~3 MB per State at 19k atoms) (an OpenMM `State` per window plus `index.json` recording restraint centres/k, CVs, and a write-order `export_seq`); a top-up loads the newest export per window across the phase's full ancestor chain and asserts the seed's restraint/CVs match the top-up's own window table. No pull runs inside a top-up. A window with a *missing* seed is dropped from the top-up before it launches (the top-up still runs for the rest); a restraint/CV *mismatch* discovered at runtime, or a corrupt/unreadable parent `index.json`, ends the *whole* top-up (never the campaign). A phase with no `epoch_window_map.csv` fails closed before it starts.
- **Unmeasured overlap edges are never treated as weak**, in the per-epoch top-up gate, the campaign quality gate, action proposals, and reports alike — an edge with no measured sample overlap used to be read as "weak" in several of these paths, which could spend MD reacting to a number that was never actually measured.

### Changed — results-changing

- **With `--no-ap-topups` (the default), scheduled-phase allocation is now a uniform per-state share of the phase's budget.** The previous per-state score allocator (bonuses for low-sample, weak-edge, frontier and high-boost states, and a 2x step multiplier for newly added states) is removed outright, not just superseded. Any campaign run under `ap_topups: false` after this change allocates differently per-epoch than before, even though no top-up ever runs. The now-unused score-policy fields (`low_sample_bonus`, `weak_edge_bonus`, `frontier_bonus`, `high_boost_bonus`, `new_state_steps`, `articulation_degenerate_fraction`) stay readable from old configs but warn once if set to a non-default value.
- **`--ap-topups` is not a new flag and its default flipped.** It already existed pre-branch with `default=True`, driving a pull-based per-state restart plus the now-removed score allocator. That default is now `False`. Any campaign config that explicitly sets `ap_topups: true` (or otherwise still runs with top-ups on) keeps top-ups enabled, but the mechanism underneath it has changed completely: the deficit-driven allocator described above, at most one top-up per phase, and seed continuation from the parent segment's exported final States instead of a fresh pull. This is a full behaviour change for any such campaign, not just a default flip.
- **Campaign-end rung overlap (the quality gate's `add_rung` and `analyze_gareus_mbar.py`'s ladder-health check) now uses pairwise MBAR state overlap** (each edge evaluated on its own two-state sample set with the union `f_k` held fixed) instead of the full-union matrix. On a real run (chignolin_7, 64 lambda-ladder states) the full-union statistic read ~2.7x too low against the same edges (median 0.089 vs 0.258 pairwise; 38/48 vs 0/48 below the 0.15 threshold). Both ladder-health axes flip toward pass on that run, not just the rung axis: the `lambda_direction` verdict goes from 38/48 to 0/48 pairs below 0.15, and the `cv1_direction` verdict from 46/60 to 0/60. **The existing 0.15/0.25 rung-overlap thresholds were calibrated on a full-matrix computation and have not been re-measured on this pairwise scale** — they are carried over unchanged for now.
- With top-ups on, per-epoch union diagnostics give the campaign-end rung gate and `add_rung` real overlap numbers mid-campaign, not only at campaign end. With top-ups off this limitation is unchanged: a rung gap is still only caught when the campaign-end union is built.

### Validation

- A synthetic A/B study (four analytic 2D landscapes, a 112-state 28-centre x 4-rung ladder, 20 seeds; `docs/superpowers/specs/2026-09-24-effective-topups/synth_study/README.md`) found no budget on any landscape where the design's heterogeneous-deficit regime actually exists — at the rows floor the split-halves test needs, every uniform-baseline state is already at or under target. Reported for information only: top-ups lower the worst per-state sigma on two of four landscapes (gated-barrier -8.4%, 20/20 seeds; slow-cv2-double-branch -1.3%, 19/20) but improve PMF RMSE on none (gated-barrier 7% worse). Missing-bridge routing passes 20/20. The patch-vs-all-state exchange-partner penalty (kept deliberately) measured 1.19-1.26x. Top-ups are not shown to help in this harness; a real-MD comparison (chignolin_10) is the open test — see `docs/atlas-md/developer/topups-todo.md`.
- The per-epoch union-MBAR solve top-ups add costs real memory on the driver node: 1,000,000 rows / 236 states took 102.7 s and peaked at 14.6 GB RSS; 250,000 rows took 26.4 s and 4.06 GB. An OOM kill during this solve cannot be caught, so a guard now skips it: after subsampling and before any rows x states matrix is allocated, the peak is estimated as kept rows x states x 8 B x 7.7; above `--ap-topup-diagnostics-max-gb` (default 8.0, about 550,000 kept rows at 236 states) the phase logs a WARNING with the estimate and runs no top-up (`no_diagnostics`).

## [0.8.3] — 2026-09-24

### Added

- **Thermodynamic energy decomposition in `gareus-analyze`.** ΔH / −TΔS decomposition of the reweighted ensemble (`gareus/mbar_analysis/thermo*.py`).
- **Shared contact-sum CV force.** When CV2 is `residual-torsion-pc` over a contact CV1, the CV1 umbrella rides on CV2's existing contact sub-CV instead of a second private copy (+14.6 % node ns/day at 236 MPS contexts). Bias energy and forces are identical to the split layout; resumed runs rebuild the layout they recorded.
- **`--us-pull-device-index`** lets the umbrella pull use every GPU on the node.
- **2D window map** on the dashboard (WINDOWS view, and PROGRESS when there is room): the (CV1, CV2) layout as a grid, one status glyph per λ rung, unrestrained windows in their own row.
- **`-V` / `--version`** on `gareus`, `gareus-analyze`, `gareus-energy-decompose`, `gareus-suggest-cvs`, `gareus-consolidate-traj`, `gareus-test-run` and `gareus_monitor.py`.

### Changed

- **User-facing name is ATLaS-MD everywhere.** Help (`-h`, `-hh`), the live dashboard, the monitor, console messages, report headings and plot titles no longer say "GaREUS"/"GAREUS". The `-h`/`-hh` titles, the dashboard status line, the monitor title and the run/PMF/CV-suggestion/integration-test report headings carry the version (`ATLaS-MD v0.8.3`), built from one place (`gareus.branding`). Command names, the `gareus` Python package, on-disk artifact names (`gareus_metadata.json`, ...) and `GAREUS_*` environment variables are unchanged, so existing scripts, configs and runs keep working.
- **Dashboard colour on by default.** `--color auto` now enables colour for `--tui-mode dashboard` even when stdout is a log file (SLURM logs previously carried no colour at all); `--color never` or `NO_COLOR=1` turns it off. The startup wordmark is redrawn as solid 5x7 block letters.
- **Dashboard window ranking on 2D layouts** compares spatial neighbours (nearest windows on the same λ rung, in restraint-width units) and flags a window only when its best neighbour overlap is dead. The flat `(w, w+1)` pairing it replaces compared windows in different cells or rungs and flagged 78 of 236 windows on a chignolin_9-shaped layout. Dashboard warnings only; sampling and analysis are unaffected. 1D ladders are unchanged.

### Correctness

- Frozen-envelope GaMD production is seated in stage 5 and refuses to run otherwise (previously a swarm hand-off could leave the integrator in stage 2, i.e. never boosted).
- Resume from a skeleton run manifest instead of refusing an eligible segment.

## [0.8.2] — 2026-09-22

Official release of the adaptive CV-selection and thermodynamic-repair work landed after v0.8.1. This release supersedes the untagged internal `0.9.0` version bump that briefly existed on `main`; no v0.9.0 tag or GitHub release was published.

### Added

- **Ab initio automatic CV2 selection.** Swarm/GENPEPT evidence can select and freeze an orthogonal secondary coordinate without a native structure or supervised folded/unfolded labels. The pipeline includes residual-torsion-PC candidates, held-out independence/information diagnostics, coupling and stability checks, deployability certificates, frozen pair digests, resume invariants, and 2D ladder construction.
- **Region-aware exploration layouts.** Adaptive layouts preserve an exactly unrestrained spatial state and representatives of every supported CV1 region before allocating the remaining state budget, so sparse production cannot silently erase discovered territory.
- **Molecular-cartography project identity.** The README landing hero now reflects adaptive mapping of conformational/free-energy landscapes.

### Correctness

- **One residual-CV definition across the thermodynamic path.** Fitting, deployment, the OpenMM force, fast-path sampling, exchange cross-pricing, reprojection, and stored artifacts now use the same compiled residual coordinate, including declared transform and clamp semantics.
- **Versioned kernel identity and sample eligibility.** Residual-CV segments record evaluator/exchange-kernel identity; affected or unknown residual segments are excluded by default from equilibrium analysis rather than silently pooled.
- **Collision-aware solvent repair and transactional grafting.** Solvent molecules are moved rigidly with peptide-solvent and emergency solvent-solvent clearance checks. Unresolved/invalid grafts fail closed and restore the original Context; non-finite coordinates, forces, or energies are rejected explicitly.
- **Exploration-preserving adaptive guards.** Mandatory region and unrestrained states survive retirement, reachability filtering, and post-pull auto-drop.
- **Frozen GaMD envelope handoff.** The swarm epoch-0 sidecar now seeds the campaign-global shared GaMD setup, preventing production from silently recalibrating the already-frozen envelope.

### Operations

- The chignolin_8 248-state launcher raises the file-descriptor limit and runs without CUDA MPS because the measured MPS client ceiling is about 60 contexts per L40S. The repository records the measured throughput trade-off and exact launch settings.
- Release metadata, citation metadata, package version, README badge, and release notes are synchronized at v0.8.2.

### Compatibility and validation

- The residual-CV kernel and artifact schemas changed after v0.8.1. Older residual-CV segments that cannot prove the current evaluator/exchange identity are classified `affected` or `unknown` by the loader and are not treated as equilibrium data by default.
- At the thermodynamic-repair completion point, the recorded full-suite result was **3374 passed, 2 skipped, 7 pre-existing failures**; the 47-file touched/new targeted batch was **835 passed**. Follow-up envelope-handoff tests added six passing cases.
- The v0.8.2 release workflow re-runs the fast exchange-kernel gate, the official package-version assertion, and a strict MkDocs build before publishing the GitHub release.

## [0.8.1] — 2026-09-18

Analysis-only release. No simulation, integrator, estimator or Hamiltonian code
is touched, so no run needs redeploying and no previously produced PMF along a
sampled CV changes. Every fix below is a defect in how results were *read back*,
and three of the four shared one failure mode: a wrong-but-plausible fallback
standing in for an unresolved value, with nothing signalling that resolution had
failed.

### Correctness

- **`traj_interval` resolved from the run root, not from a silent default.**
  `_read_traj_interval` searched only `d.prod_dir`, which for an adaptive run is
  `<run>/adaptive_production`, while `traj_interval` is written one level up at
  the run root. Nothing was found, so it returned its default of 50 against a
  real value of 2500. The default is what made it silent: all five call sites
  guarded with `if spf <= 0 or spf == 500`, where 500 is
  `_read_traj_interval_from_epoch_dirs`'s fallback but 50 was
  `_read_traj_interval`'s, so an unresolved value was indistinguishable from a
  configured one and the epoch-dir fallback never ran. Downstream,
  `_sample_to_segment_frame` accepts samples in
  `[resume_start + spf, resume_start + n_frames*spf]`, so a 50x too-small `spf`
  shortened that window 50x and the rest were dropped by a bare `continue`.
  On chignolin_7: 103,664 of 5,124,128 samples assigned, 2.023% against 2.00%
  predicted; after the fix 5,118,016 (99.88%), from 611,536 frames rather than
  at most 103,664. Both functions now return 0 when unresolved, resolution
  happens once in `_resolve_traj_interval(d, warnings)` which also reports its
  source, and a test asserts no call site reimplements it.
- **`gareus_report` declared in `py-modules`.** It is a root-level module, and
  `packages.find` only matches `gareus*`, so it was never installed. The
  installed `gareus-analyze` raised `ModuleNotFoundError` for every run started
  from any working directory other than the project root. All three import sites
  wrap the import in `try`/`except` and degrade to a warning, so runs still
  exited 0 while silently producing no result-health verdict
  (`health.overall` "UNKNOWN", `checks` empty), no warning triage
  (`warnings_grouped` empty), no ladder-overlap axis report and no terminal
  `RESULT HEALTH` banner.
- **Row filters slice the lambda-ladder channel energies.**
  `_skip_first_n_frames` and `_apply_analysis_stride` filtered every per-sample
  array except `v_pep_kj` / `v_dih_kj`, which were added later and wired only
  into `clean()` and `_masked_data()`. With `--skip-first-n-frames` on a ladder
  run those two kept their pre-cut length and the epoch_000 split then raised
  `IndexError: boolean index did not match indexed array`. Nothing downstream of
  the loaders reads them per-sample, so the misalignment could only crash, never
  bias a PMF. The field list now lives in one registry driving all four
  functions, mis-length arrays are dropped rather than left unsliced, and
  alignment is asserted at the end of each filter.

### Output contract

- **The ladder axes are graded at the state-overlap calibration.**
  `ladder_overlap_by_axis` grades symmetrised MBAR state overlap
  `sqrt(O_ab * O_ba)` but defaulted to 0.30 and was handed
  `--min-neighbor-overlap`, a CV1-marginal histogram-intersection target — a
  different quantity on a different scale. On chignolin_7 no pair cleared it on
  either axis (0 of 48 lambda, 0 of 60 CV1), so both reported "split into 64
  components" while the CV1-marginal check on the same axis passed at worst
  0.491 / connected. New `--min-ladder-state-overlap`, defaulting to 0.15 =
  `AdaptiveDecisionPolicy.min_rung_overlap`, the value the adaptive driver
  already gates rung edges at, calibrated on the S3 pilot's adjacent-rung
  entries 0.240-0.298. Health rows for these axes will change on re-analysis of
  existing runs; the 0.15 figure is a rung calibration from a 1-D single-centre
  ladder and the CV1-direction rows reuse it.
- **New `Trajectory frame coverage` health row.** Warns and grades when
  substantially fewer samples are assigned a frame than there are samples, so a
  future mis-resolution surfaces in `RESULT HEALTH` rather than only in a
  warning list — which is how the `traj_interval` defect stayed invisible.
- **Trajectory-derived results produced before this release are superseded.**
  Rg, PCA, the 2D FES, SASA and phi/psi were all built from the subset that
  survived the `traj_interval` defect. Re-analysis is required for numbers
  quoted from them; PMFs along a sampled CV are unaffected, as that path never
  reads trajectories.

### Known issues

- The Parquet-source failure recorded under 0.8 is carried forward unchanged.
  Nothing in this release touches that path and it was not re-checked.

## [0.8] — 2026-09-18

### Correctness

- `ladder_active` is now a property of the **campaign**, not of one sub-run. It was
  derived from `args.state_gamd_lambdas`, which only ever describes the states handed
  to a single adaptive sub-run. A top-up given nothing but lambda=0 states therefore
  concluded "no ladder here", with two consequences: `k0max_by_channel` stayed `None`
  so `set_replica_lambda_for_window()` no-oped and every replica **kept the shared
  calibration's full `k0`** instead of `lambda*k0max = 0` — running boosted while
  recorded as lambda=0 — and `pep_env` stayed `None` so `v_pep`/`v_dih` were written
  as NaN. `resolve_ladder_active()` now also consults the campaign state registry,
  and the parent walk is bounded at `adaptive_production/` so one campaign cannot
  switch on another's boost recording. Plain umbrella/REUS and plain GaMD are
  deliberately unaffected: the ladder path costs two extra Context energy reads per
  logged sample.
- Observed in the chignolin_7 campaign, where three lambda=0-only top-ups carried a
  mean recorded boost of 9.44-10.60 kJ/mol against exactly 0.000 for lambda=0 samples
  from mixed sub-runs and 9.57-10.16 for the genuine lambda=1 rung. The MBAR loader's
  existing refusal to fabricate a 0.0 boost for samples lacking raw channel energies
  excluded those 992,896 samples, which was protective rather than lossy.

### Performance

- Removed a redundant all-groups force evaluation from the Pep-GaMD integrator by
  deriving the bias force groups as the complement of the boosted set. Measured at
  22.7% of MD step cost by controlled A/B on the real 19,008-atom system.
- The application-controlled barostat reads positions and box vectors from a single
  `State` instead of two.
- Together ~1.26x campaign throughput, confirmed by in-campaign A/B on identical
  hardware.

### Known issues

- `tests/test_package_smoke.py::test_tiny_lambda_ladder_run_completes_end_to_end_slow`
  fails with `pyarrow.lib.ArrowInvalid: Could not open Parquet input source`. Present
  in 0.7 as well; unrelated to the changes above.

## [0.7] — 2026-09-12

### Added

- Explicit thermodynamic target and detailed-balance contract for NVT, NPT, replica exchange, Gibbs-walk proposals, and GaMD lambda ladders.
- Application-controlled NPT Metropolis volume moves for supported boosted Hamiltonians.
- Stage-aware effective-potential adapters that separate physical, bias, auxiliary, and GaMD boost energies.
- Exact finite-state exchange-kernel validation that distinguishes elementary detailed balance from stationary invariance of composed sweeps.
- State-by-configuration GaMD boost cross-evaluation for lambda/Hamiltonian replica exchange.
- Adaptive-production and double-adaptive workflows with phase-local state provenance.
- Parquet-backed sample/exchange storage and query tooling.
- Modernized documentation and project landing page.

### Correctness

- Native OpenMM barostat use is rejected for supported boosted paths where its acceptance energy would not match the propagated effective potential.
- Gibbs-walk proposals include the reverse/forward proposal correction required by Metropolis-Hastings.
- NPT volume acceptance uses the molecular Jacobian convention consistent with rigid-molecule isotropic scaling.
- Adaptive phases are tracked as distinct Hamiltonian epochs rather than silently pooled under reused local window identifiers.

### Validation scope

ATLaS-MD does not claim mathematically exact finite-timestep propagation. Exchange and NPT transition-kernel correctness are tested separately from numerical integration accuracy and from post-hoc reweighting validity.

[0.7]: https://github.com/sulcjo/ATLaS-MD/releases/tag/v0.7
