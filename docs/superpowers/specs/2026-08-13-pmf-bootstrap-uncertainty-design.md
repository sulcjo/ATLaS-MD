# PMF Bootstrap Uncertainty Design

## Goal

Give every selected/headline PMF that `analyze_gareus_mbar.py` produces (main CV1,
Rg, PCA, secondary CV2, and every 2D-FES family) a per-bin uncertainty estimate,
opt-in via a CLI flag, off by default.

## Problem

The pipeline reports point-estimate free energies with no statistical uncertainty
anywhere. There is no way to tell whether a PMF difference (between two runs, two
regions of one PMF, or two estimator choices) is real or sampling noise. Confirmed
via a full-file grep: zero existing uncertainty/error-bar/covariance machinery
(the only `bootstrap` hits in the file are the unrelated torsion-PCA seed-bootstrap
CV concept).

The default headline estimator for a real GaMD run is `gamd_cumulant2` (or
`gamd_cumulant3`), not plain MBAR reweighting. This rules out the standard MBAR
asymptotic-covariance approach as a complete solution on its own — there is no
published closed-form uncertainty for "MBAR free energies plus a GaMD cumulant
correction," and deriving one from scratch carries real correctness risk in a file
that has already needed two rounds of subtle-math-bug fixes this session.

Samples come from replica trajectories with real time-autocorrelation. A naive
per-sample bootstrap (resampling individual rows with replacement) treats
correlated frames as independent draws and reports falsely tight error bars.

## Recommended Approach

**Fixed-`f_k` block bootstrap.** Solve MBAR once (already happening), then
resample only the final weighted-histogram / cumulant-expansion step, many times,
reusing the same `f_k`:

1. Solve MBAR once as today (`m = solve_mbar(...)`), producing `f_k` and per-sample
   `logw`/`base_w`.
2. For each window `k`, identify its **blocks** — see below — and, for each of
   `n_boot` replicates, resample `k`'s own block set with replacement (same block
   *count* as the original, different composition).
3. Concatenate every window's resampled blocks into one resampled sample set.
   Reuse each drawn sample's *already-computed* weight (`base_w`, from the
   original full-population solve) — no MBAR re-solve. This is what keeps a
   replicate cheap: it's a `pmf_from_weights`/`cumulant2`/`cumulant3` call, not a
   `solve_mbar` call.
4. Rebuild the selected PMF (whichever method — `umbrella_only`,
   `gamd_exponential`, `gamd_cumulant2`, `gamd_cumulant3` — is `selected` for this
   analysis) on the resampled set.
5. Shift the replicate PMF by the **same constant** the main PMF used (the value
   at the main PMF's own minimum bin) — never each replicate's own minimum. See
   Error Handling for why this matters.
6. Per-bin uncertainty = standard deviation across the `n_boot` replicate PMF
   curves, reported as `pmf_std_kcal_mol` (a percentile-interval alternative was
   considered and dropped for this design — std is simpler and consistent with
   how this file's other diagnostics, e.g. `boost_stats`, already report spread).

This is the standard per-window block bootstrap used in the umbrella-sampling
literature for correlated-sample error estimation (the same idea g_wham's error
analysis uses), applied here with a fixed-`f_k` shortcut to avoid `n_boot` MBAR
re-solves.

### Block definition

A block is one `(epoch_source, replica)` pair: one replica's samples within one
epoch/phase count as one autocorrelated unit, even if the same replica index is
reused in a later epoch — adaptive-production runs don't guarantee trajectory
continuity across epoch boundaries, so treating epoch-crossing reuse as a new
block is the conservative assumption. For single-source/non-adaptive-production
runs (no `meta['_epoch_source']`), block = replica alone. Both fields already
exist on `Data` — no new data collection required.

## Why This Approach

- Reuses `pmf_from_weights`/`cumulant2`/`cumulant3` as-is — the exact functions
  this session's cumulant-refactor fix already hardened and verified
  bit-identical. No new PMF-construction math to get wrong.
- No MBAR re-solve per replicate: `n_boot=100` costs roughly
  `100 × (histogram + cumulant-expansion time)`, measured elsewhere in this file
  at ~1-2s/call at full real-data scale (13.8M samples) — so ~1-3 minutes added,
  paid only when the flag is set.
- Respects real sample autocorrelation via block-level (not per-sample)
  resampling — the alternative (naive per-sample bootstrap) would silently
  under-report uncertainty.
- Stays inside established, already-verified statistical practice (the umbrella-
  sampling block bootstrap) rather than inventing new machinery.

## Alternatives Considered

### Full bootstrap, re-solve MBAR per replicate

Resample, resolve MBAR from scratch, rebuild the PMF, repeat. Gold standard —
captures `f_k`'s own sampling uncertainty in addition to per-bin histogram noise,
which the fixed-`f_k` approach ignores. Rejected as the default because of cost:
`n_boot` full MBAR solves at real scale (tens of millions of samples, up to ~364
windows) is minutes-to-tens-of-minutes even with the existing warm-starting and
numba threading. Left as a possible future `--pmf-uncertainty-full-refit` upgrade
path if the fixed-`f_k` approximation ever proves too optimistic in practice — not
built now (YAGNI: no evidence yet that `f_k` uncertainty is material at this
pipeline's typical sample sizes, where windows routinely carry hundreds of
thousands to millions of samples each).

### Analytic (delta-method) propagation of MBAR's asymptotic covariance through the cumulant correction

No resampling, single fast pass. Rejected: there is no published closed form for
this, so it would mean deriving new statistical machinery from scratch and
validating it ourselves. Highest implementation/correctness risk of the three
options — a confidently-wrong error bar is worse than no error bar, and this file
has already needed real time spent finding subtle math bugs.

### Naive per-sample bootstrap (no blocking)

Cheapest, simplest to implement. Rejected outright: ignores replica-trajectory
autocorrelation, which for MD data means treating thousands of correlated frames
as independent draws — this doesn't just add noise to the uncertainty estimate, it
produces a systematically, confidently wrong (too narrow) one.

## Architecture

### Shared helpers (new)

```text
_sample_block_ids(d: Data) -> np.ndarray
```
Computes the `(epoch_source, replica)` block-id array once per `Data` object.
Called once per `analyze()` invocation (via `analyze()` itself), not recomputed
per `analyze_*` function — follows this session's established
avoid-redundant-recompute convention.

```text
_bootstrap_pmf_uncertainty_1d(cv, base_w, boost, bins, beta, kbt_kcal,
                               window, block_ids, selected_method,
                               main_pmf, n_boot, rng) -> dict
```
```text
_bootstrap_pmf_uncertainty_2d(x, y, base_w, boost, xbins, ybins, beta, kbt_kcal,
                               window, block_ids, selected_method,
                               main_pmf2d, n_boot, rng) -> dict
```
Both do the resample-rebuild-shift loop described above and return
`{'pmf_std': ndarray, 'blocks_per_window': ndarray, 'low_block_windows': list[int]}`.
One shared implementation serves every `analyze_*` call site — CV1, Rg, PCA,
secondary CV2, and every 2D-FES family — no per-analysis reimplementation.

### CLI flags (new)

- `--pmf-uncertainty` (`store_true`, default off)
- `--pmf-uncertainty-n-boot` (default `100`)
- `--pmf-uncertainty-seed` (default a fixed integer, e.g. `0`) — seeded so
  re-running with the flag on reproduces identical error bars, matching this
  file's existing emphasis on reproducible MBAR results (bitwise determinism
  across processes/thread counts is already a proven, documented property
  elsewhere in this codebase).

### Call sites

One call to the appropriate shared helper added immediately after each
`analyze_*` function builds its own selected PMF, gated behind
`if getattr(args, 'pmf_uncertainty', False)`. No changes to the PMF-construction
functions themselves.

### Output

- Each PMF CSV writer (`write_pmf`, `write_all`, `write_cv2_pmf`,
  `write_2d_fes_csv`, `write_cv1_cv2_2d_fes_csv`, `write_pca_2d_fes_csv`, `write_rg_pmf`)
  gains an optional `pmf_std_kcal_mol` column, populated only when uncertainty was
  computed for that call.
- Plots gain a shaded `fill_between` band around the selected PMF curve when
  uncertainty is available.
- `pmf_summary.json` gains a scalar per analysis, e.g.
  `pmf_uncertainty_at_minimum_kcal_mol`, so the headline number is visible
  without opening a CSV.

## Error Handling

- **Shift-anchor consistency**: every bootstrap replicate is shifted by the value
  the *main* PMF had at the *main* PMF's own minimum bin — never each replicate's
  own minimum. Shifting each replicate to its own minimum artificially erases
  uncertainty exactly at that bin and distorts every other bin's uncertainty
  relative to it. This must be covered by a dedicated regression test (see
  Testing) since it's the easiest part of this design to silently get wrong.
- **Too-few-blocks windows**: a window with fewer than 3 blocks can't be
  meaningfully block-bootstrapped. `blocks_per_window` from the helper feeds a
  warning (existing `warnings.append(...)` convention) naming which windows/bins
  are affected; those bins are flagged in the CSV/summary rather than silently
  reported with a falsely-precise error bar.
- **All-NaN-boost / empty windows**: excluded from block-count denominators the
  same way they're already excluded from the main PMF (existing `clean()`/
  `isfinite`-masking convention — no new exclusion logic needed, reuse what's
  there).
- **`n_boot` too small**: no hard floor enforced, but the CLI help text should
  note that very small `n_boot` (e.g. <20) gives a noisy uncertainty-of-the-
  uncertainty; this is documentation, not a runtime check.

## Testing

- Synthetic multi-window, multi-replica, multi-epoch dataset with a KNOWN
  injected inter-block correlation structure (e.g. all samples within a block
  drawn from a shared per-block offset plus independent per-sample noise):
  - block bootstrap reports wider uncertainty than a naive per-sample bootstrap
    on the identical data (proves it actually respects the correlation — this is
    the core scientific-validity check for the whole feature).
  - the shift-to-main-minimum anchoring is followed exactly, not each
    replicate's own minimum (construct a case where the two would visibly
    differ).
  - the few-blocks warning fires for a window with 1-2 blocks and not for a
    window with many.
  - uncertainty is bit-identical across two runs with the same
    `--pmf-uncertainty-seed`, and changes with a different seed.
  - 2D variant produces a per-bin std grid consistent with the 1D marginal case
    (same style of cross-check already used elsewhere in this file for
    1D/2D consistency).
- Run the existing test files touching every `analyze_*` call site being
  modified, to confirm the opt-in flag being *off* leaves all current output
  byte-for-byte unchanged (the new code path must be strictly additive when
  `--pmf-uncertainty` is not passed).

Verification commands (once implemented):

```bash
pytest -q tests/test_pmf_bootstrap_uncertainty.py
python -m py_compile analyze_gareus_mbar.py
```

## Out Of Scope

- Full bootstrap with per-replicate MBAR re-solve (`f_k` re-estimation) — noted
  above as a possible future upgrade, not built now.
- Analytic/delta-method uncertainty propagation through the cumulant correction.
- Uncertainty for the non-selected estimator methods (e.g. reporting error bars
  for `umbrella_only` when `gamd_cumulant2` is the selected/headline method) —
  only the selected/default method per analysis gets uncertainty.
- Changing which estimator is auto-selected as the default (a separate,
  previously-discussed idea — auto-selecting based on measured ESS/anharmonicity
  — not part of this design).
- Retroactively adding uncertainty to past run outputs.
