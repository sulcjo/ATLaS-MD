# chignolin_6 — why MBAR base ESS is 0.7%

Run: `RUNS/chignolin_6/adaptive_production/pmf_analysis/` (12,103,761 samples, 27 states,
base ESS 89,542 = 0.74%, health verdict FAIL, `pmf_span` 18.94 kcal/mol,
`pmf_minimum_cv_A` 0.337).

Date: 2026-08-25. Analysis only — no code or run data changed.

## Answer in one line

Not the low-sample windows. `--us-auto-drop-bad-windows` pruned windows post-pull and renumbered the
survivors 0..N-1, but the drop was never propagated back to the state registry — so every
phase's `epoch_window_map.csv` still maps local index -> state as if all 27 states were live.
The MBAR loader therefore attributes ~1.8M samples (15% of the campaign) to the wrong umbrella
state. That destroyed states 21 and 22, the only bridges between state 23's CV2 region and the
main block; with the bridges gone `f_23` is unconstrained, and MBAR parked 94% of the population
on state 23. Contamination in 21/22 *plus* isolation of 23 — neither alone is the story.

## Evidence chain

### 1. ESS is entirely two states' internal weight spread

Per-state unbiased-weight totals (`pmf_analysis/rg_samples_with_weights.csv`, the 6.19M
trajectory-matched subset; its ESS 59,269 reproduces the headline 0.74%):

| state | weight share |
|---|---|
| 23 | 0.9435 |
| 22 | 0.0564 |
| 21 | 1.17e-4 |
| all other 24 states | <= 2.7e-9 |

ESS = 1/sum(w^2) = 1/(1.644e-5 + 4.285e-7) ~ 59.3k, i.e. state 23 alone accounts for 54,148 of it.
Note state 23's own data is mostly clean (965,780 correct samples vs 1,250 wrong) — it dominates
not because it is corrupt but because its neighbours 21/22 were corrupted away, leaving nothing
to pin its free-energy offset.
The base-ESS number is not measuring cross-window overlap at all. The 20 epoch-0 windows and
the three late bridge states contribute ~0 to the ensemble.

### 2. The user's suspects are not the problem

States 24/25/26 have the fewest samples (8,960 / 10,750 / 8,958) but overlap the main block at
0.43-0.90 and have textbook self-bias (~0.5 kT median). The health check's
"worst 0.013 (pair 23-24)" pairs states **by index, not by CV position**, which is why the
diagnostic points at 24.

### 3. Self-bias explodes in exactly three states

Reduced bias of each state's own samples in its own restraint (`adaptive_union_mbar.npz`,
should be ~1 kT for a 2-DOF harmonic):

| state | n | median | p90 | max | frac > 100 kT |
|---|---|---|---|---|---|
| 0-19, 24-26 | - | 0.5-1.4 | 1.7-4.9 | 6-14 | 0.000 |
| 20 | 6030 | 3.4 | 6.7 | 14 | 0.000 |
| 21 | 1250 | **63.6** | 80.7 | 116 | 0.006 |
| 22 | 9740 | 1.1 | **652.8** | 876 | 0.128 |
| 23 | 2440 | **132.9** | 221.8 | 310 | 0.512 |

### 4. Cause: stale `epoch_window_map.csv` after auto-drop

`final/baseline/gareus_metadata.json`:
`window_metadata.dropped_post_pull_bad_windows = [20, 22, 23]`, with
`explicit_window_table.n_windows = 27` (stale) while `windows_A` holds 24 renumbered entries.
`final/baseline/umbrella_explicit_windows.csv` (the table the sampler actually ran) has 24 rows
numbered 0..23; `final/baseline/epoch_window_map.csv` has 27 identity rows.
Same shape in `epoch_001/baseline`: 23 windows physically run, 24 identity map rows, state 20 dropped.

`registry.write_active_window_csv(..., map_path=.../epoch_window_map.csv)`
(`gareus/adaptive_production.py:5474`, and `write_epoch_window_map` at `:588`) enumerates
`active_states()` -> `epoch_window = i`. The post-pull drop is recorded only in the *phase's*
`gareus_metadata.json` and never propagates back to the registry: `final_active_windows.csv`
still carries state 20 as `patch_lifecycle: adaptive_production_active`, `usable_for_mbar: 1`.
So `active_states()` permanently believes 27 states are live and **every** phase's map inherits
that. The topup maps that came out right (`topup_001`, `002`, `005`-`008`) are correctly compacted
and non-identity (e.g. `final/topup_002` maps 11->12, 12->15) — they are right only because they
never touched a dropped state. `epoch_001/topup_004` and `final/topup_003`, which did, are shifted
like the baselines. This is a registry-state-lifecycle bug, not a baseline-writer bug.

Verified by CV2 fingerprint: every local window's sampled `cv2` mean matches a *different*
state's recorded `secondary_center` to 3-4 decimals.

| phase | local w | map says | truly is | consequence |
|---|---|---|---|---|
| epoch_001/baseline | 20, 21, 22 | 20, 21, 22 | 21, 22, 23 | shift by 1 (state 20 dropped) |
| epoch_001/topup_004 | 0, 1 | 20, 21 | 21, 22 | shift by 1 |
| final/baseline | 20, 21, 22, 23 | 20..23 | 21, 24, 25, 26 | states 20, 22, 23 dropped |
| final/topup_003 | 0 | 20 | 21 | shift by 1 |
| epoch_000, topups 001/002/005-008 | - | correct | correct | never touched a dropped state |

Reconstructing the pipeline's state assignment from the parquet `window_id` + each phase's own map
reproduces `pmf_summary.json`'s `n_k` **exactly** (state 20: 861,547 vs 861,536; 21: 856,767 vs
856,756; 22: 107,397 vs 107,386; 23: 967,030 vs 967,020; the uniform ~10-11 shortfall is a
systematic boundary/burn-in row, not a mismatch), confirming this is the mapping
`analyze_gareus_mbar.py` actually used.

State 15 discriminates *map-honouring* from *raw `window_id`* attribution, which are otherwise
indistinguishable for the identity-map baselines: reconstruction gives
156,250 + 97,657 + 199,990 (topup_001 localw=14) + 1,250 + 2,160 (topup_002 localw=12) = 457,307
against pipeline `n_k[15]` = 457,297. Under raw `window_id` those two topup blocks would land in
states 14 and 12 and `n_k[15]` would be ~255k. So the loader does read the maps — the maps are
what is wrong.

### 5. Damage

- **956,924 samples (7.91%)** carry >= 80 kT of fabricated self-bias: state 21 gets 856,767
  samples whose true center2 is +1.407 while it is scored against -1.4379 (k2 = 148.5 ->
  **2,012 kT**); state 22 gets 98,907 (82 kT and 1,370 kT); state 23 gets 1,250 (380 kT).
- **State 20 was never actually sampled.** All 861,547 samples attributed to it are really state
  21's (center2 -1.6282 vs -1.4379, ~5-6 kT, below the 20 kT cut so it looks benign) — so state 20
  is a phantom duplicate of 21 and its `f_k` is fiction.
- Total mis-attributed: ~1.82M samples, 15% of the campaign, all in the 4 states that hold
  99.99% of the posterior weight.
- State 22's slot pools **three different Hamiltonians**: real-23 data (97,657), real-25 data
  (1,250) and its own real-22 data (8,490). That is why its self-bias is bimodal (median 1.1,
  p90 653) rather than uniformly bad.
- `pmf_span` 18.94 kcal/mol and `pmf_minimum_cv_A` 0.337 are unusable: the reported minimum sits
  above every window center (max 0.176) and is sampled only by this corrupted island.

## Secondary findings (real, independent of the mapping bug)

1. **The overlap matrix is CV1-marginal only.** States 0-19 share just two CV1 centers (0.0 and
   0.0654) and differ only in CV2, where centre spacing / sigma is ~3-6 (sigma_obs 0.05-0.08,
   matching sqrt(kT/k2) for k2 up to 200). Their reported 0.6-0.99 "overlap" is illusory; the
   pipeline's own diagnostic cannot see the CV2-axis gap, which is why nothing flagged this.
2. **`cv2_k_max: 200` is not enforced on adaptively created states** — states 22 and 25 carry
   k2 = 402.5 and 275.5.
3. **CV1 ladder collapsed.** 27 states, 6 distinct CV1 centers, spanning 0 -> 0.176 of a
   normalized contact CV whose samples reach 0.416; `cv1_target_spacing: 0.1` was not realized,
   and every window sits at `cv1_k_max` = 200 (saturated).
4. **Bridge states were created late and starved.** State 24 was spawned to repair edge 14-20
   (overlap 0.127); its actual overlap to 20 is 0.055 — worse than what it was built to fix.
   State 26 was spawned for edge 11-13 at overlap 0.844, which is not weak. All three got
   ~9-11k samples because they were born in the last epoch.
5. Separately, epoch_000 ran CV2 = `torsion-pca` and epoch_001+ ran `tica-linear`
   (confirmed in per-phase `run_manifest.json`). Per-epoch bias uses each epoch's own centers so
   `u_nk` is internally consistent, but state k is still not one Hamiltonian across the switch.

## Prognosis — fixing the mapping will not restore a healthy ESS

Re-running the analysis with correct labels removes the fabricated bias, but under *correct*
labels the CV2 ladder is still genuinely under-resolved at the extremes, so overlap will stay
poor there:

| pair | CV2 centre spacing | sigma_obs | spacing / sigma |
|---|---|---|---|
| 20 -> 21 | 0.190 | ~0.07 | ~2.6 |
| 12 -> 22 | 0.286 | 0.122 / 0.038 | 2.3 / 7.5 |
| 22 -> 23 | 0.352 | 0.038 / 0.062 | 9.3 / 5.7 |

(sigma ~ sqrt(kT/k2); k2 = 402.5 for state 22 gives sigma 0.038.) Healthy umbrella overlap wants
spacing/sigma ~ 1-1.5. So expect the corrected PMF to be *defensible* over the main CV2 block and
still *unconstrained* out at states 22/23 — i.e. re-analysis fixes correctness, not resolution.
Restoring resolution needs softer k2 and/or intermediate CV2 windows out there, which is new
sampling, not new analysis.

## Suggested fixes (not applied)

1. Propagate the post-pull drop back into the registry (retire the dropped states, or mark them
   `usable_for_mbar: 0`) so `active_states()` stops handing out phantom states, **and** derive
   `epoch_window_map.csv` from the phase's own surviving window list
   (`umbrella_explicit_windows.csv` in the phase dir) rather than from `active_states()`, so
   `epoch_window` is the true local index. Rewriting only the baseline map is not enough —
   `epoch_001/topup_004` and `final/topup_003` are shifted too.
2. Add a loader guard: assert `len(epoch_window_map) == n_windows` from the phase's own
   `gareus_metadata.json` / `analysis_metadata_validation.json` (which already record the correct
   24 and 23), and fail closed instead of silently mis-attributing.
3. Add a cheap self-bias sanity check to `analyze_gareus_mbar.py`: median own-state reduced bias
   per state should be ~1 kT; anything above ~10 kT means the sample-to-state mapping or the
   window params are wrong. This one check would have caught the whole thing.
4. Report worst-overlap pairs by CV-space adjacency, not state index, and compute overlap in the
   full (CV1, CV2) space rather than the CV1 marginal.

---

## Re-analysis caveat: what changes in `pmf_summary.json` after the diagnostics fix (2026-08-25)

Written by the analysis/diagnostics fix round. Applies to **re-running
`analyze_gareus_mbar.py` on any existing run** — no run data changes, but some published
numbers and the top-level health verdict can move. Read this before comparing a fresh
`pmf_summary.json` against an older one.

### Unchanged, on purpose

The overlap numbers that already existed keep their exact meaning, so cross-run comparisons
against older analyses stay valid:

- `s['neighbor_overlap']`, `overlap_matrix.csv`, `overlap_matrix.png` and
  `window_diagnostics.csv`'s `overlap_left`/`overlap_right` are still the **CV1-marginal**
  histogram overlap, bit-identical to what the pre-fix code produced.
- The `Weak neighbor CV overlap below 0.30 …` warning is still computed on that marginal
  matrix with `--min-neighbor-overlap`. A 2D run re-analysed today does **not** gain new
  copies of it.
- `overlap_matrix.csv` now carries a leading `#` comment line naming its space (readers need
  `comment='#'`). That stamp is the only way to tell a marginal matrix from a joint one on
  disk, which is why it exists.

### New, published alongside

- `s['overlap_space']` = `'cv1_marginal'` — the space of every key listed above.
- `s['joint_overlap']` — the full **joint (CV1, CV2)** overlap: `available`/`reason`,
  `dim`, `threshold`, index-adjacent `neighbor_overlap`, CV-space-adjacent
  `cv_space_neighbor_overlap`, `worst_cv_space_pair`, binning `stats`, and
  `overlap_matrix_joint.csv` (also stamped). Computed by default for any run with a biased
  second axis and ≥50 % of samples carrying a finite secondary CV; `--no-joint-overlap`
  disables it, restoring pre-fix behaviour exactly.
- The joint number has **its own threshold**, `--min-joint-neighbor-overlap`, defaulting to
  `--min-neighbor-overlap ** 2` (0.09). Joint overlap is bounded above by the CV1 marginal
  and deflates further at small per-state N, so the marginal-calibrated 0.30 is not a valid
  threshold for it; reusing it would have failed well-resolved 2D runs.
- `s['cv_space_neighbor_overlap']` — the marginal matrix re-paired by true nearest neighbour
  in restraint-centre space (σ units), per suggested fix #4 above.
- `s['self_bias']` — per-state median/p90/max reduced bias of each state's own samples in its
  own restraint (suggested fix #3), plus `self_bias_median_kT`/`self_bias_p90_kT`,
  `secondary_center`, `secondary_k_kcal_mol`, `cv2_mean`, `cv2_std`,
  `cv_space_neighbor`, `cv_space_overlap_marginal` and `cv_space_overlap_joint` columns in
  `window_diagnostics.csv`.
- `s['epoch_000_report']` carries the same fields as the main block (one shared builder,
  `analyze_gareus_mbar._report_summary_fields`).

### Verdict changes to expect (nothing here means the data got worse)

`gareus_report.build_health_verdict` can move on unchanged data, in four ways:

1. **New `Sample-to-state mapping` check.** FAIL when a phase was loaded with
   `GAREUS_ALLOW_STALE_WINDOW_MAP=1` (samples knowingly attributed to the wrong umbrella
   states — this can no longer sit inside a PASS), or when any state's own-restraint
   self-bias median exceeds 10 kT / p90 exceeds 50 kT. CAUTION above 5 kT median, or when a
   stale `epoch_window_map.csv` was repaired in memory for this analysis. **chignolin_6 and
   any other auto-drop-affected run will FAIL this check**, correctly.
2. **`Window overlap` now grades both spaces** and takes the worse. A 2D run whose states
   are well separated on CV1 but essentially disjoint on CV2 — chignolin_6's states 0–19 —
   flips from pass to fail. That is the blindness this whole document is about, not a
   recalibration.
3. **`Window overlap` pairs windows by CV space, not index.** The reported worst pair
   changes even when the status does not (chignolin_6: "pair 23-24" → the real weak pair).
   **Expect a new HIGH warning on re-analysis of essentially any 2D run**:
   `Weak CV-space nearest-neighbour overlap below 0.30 (CV1 marginal) for pairs: …`. It is
   the same quantity and the same threshold as the pre-existing index-adjacent warning — only
   the *pairing* is corrected, so pairs the index ordering never compared are now compared and
   some of them are genuinely below 0.30 (chignolin_6's own marginal numbers: 14-20 at 0.210,
   22-23 at 0.326). It is triaged HIGH, so it appears in the terminal health block and can
   raise a run's warning tally without a single number having changed. That is a real
   unbridged gap being named for the first time, not a recalibration; `--min-neighbor-overlap`
   still governs it if you want to move the line.
4. **Warning severities.** `[stale window map]` + `GAREUS_ALLOW_STALE_WINDOW_MAP` → CRITICAL;
   a repaired-in-memory `[stale window map]` → HIGH; `[cv2 regime change]` → HIGH;
   `Weak joint (CV1, CV2) nearest-neighbour overlap` and
   `Joint (CV1, CV2) window overlap was NOT computed` → HIGH. All were MEDIUM-by-default
   before, i.e. hidden from the terminal health block.

No PMF, MBAR or free-energy number is recomputed by any of this: `pmf_span_kcal_mol`,
`pmf_minimum_cv_A`, `base_ess`, `f_k` and every PMF curve are untouched by the diagnostics
fix (they do change for auto-drop-affected runs, but from the loader-side window-map repair,
documented with those fixes — not from anything in this section).

### Measured on this run's own data (why the joint number is worth its verdict impact)

Recomputed directly from `adaptive_production/adaptive_union_mbar.npz` (126,470 merged
samples, all 27 states, 60 primary x 30 secondary bins, 100 % of samples carrying a finite
secondary CV), marginal vs joint overlap for the same state pairs:

| pair | CV1 marginal | joint (CV1, CV2) |
|---|---|---|
| 0-1 | 0.696 | 0.0000 |
| 5-6 | 0.781 | 0.0012 |
| 10-11 | 0.831 | 0.0664 |
| 14-20 | 0.210 | 0.0000 |
| 20-21 | 0.464 | 0.0000 |
| 22-23 | 0.326 | 0.0000 |

Mean off-diagonal overlap among states 0-19: **0.724 marginal vs 0.124 joint**. The marginal
diagnostic was not merely optimistic on this run, it was reporting a different quantity than
the one that had failed. (At K = 27 the byte-sized chunk guard leaves the joint reduction in a
single 10.5 MB block — 1800 of 1800 cells — so this costs nothing at real local window counts;
it only engages for the K = 364-scale oracle geometries.)

---

## Verified on real data: the fix works (2026-08-25)

The mapping repair was run against this run's real Parquet samples and real per-phase maps,
read-only, through the shipped `_validate_and_repair_epoch_window_map`.

**All four affected phases reproduce the ground truth exactly**, via the phase's own window
table, i.e. independently of the drop record:

| phase | map rows -> real windows | corrected mapping | phantom states |
|---|---|---|---|
| `final/baseline` | 27 -> 24 | 20->21, 21->24, 22->25, 23->26 | 20, 22, 23 |
| `epoch_001/baseline` | 24 -> 23 | 20->21, 21->22, 22->23 | 20 |
| `epoch_001/topup_004_75786000` | 3 -> 2 | 0->21, 1->22 | 20 |
| `final/topup_003_478000` | 2 -> 1 | 0->21 | 20 |

**1,818,471 of 12,104,001 samples change attributed state** — independently reproducing this
document's ~1.82M / 12.1M figure from a completely separate code path.

Per-state self-bias median, stale map -> repaired map (whole campaign, full Parquet, centres
from each phase's own `epoch_window_map.csv`, k from the global registry, T = 300 K):

| state | stale | repaired |
|---|---|---|
| 21 | 1007.57 kT | **1.11 kT** |
| 22 | 44.18 kT | **0.98 kT** |
| 20 | 4.95 kT over 861,547 samples | **zero samples** (the phantom duplicate of 21, which was never actually sampled) |
| 23 | 2.63 kT | 2.63 kT (unchanged — see below) |
| the other 23 states | 0.46-1.30 kT | bit-identical |

State 23's 2.63 kT is **not** a residual of the mapping bug: 964,590 of its 1,063,437 samples
(90.7 %) come from `epoch_001/topup_003_68359000`, a single-window phase (map: local 0 ->
state 23) with zero re-attributed samples. That value is a pre-existing property of that phase.

### Provenance note on the 63.6 / 132.9 kT figures quoted earlier in this document

Those reproduce **exactly** from `adaptive_union_mbar.npz`'s own diagonal (state 21 = 63.58,
state 23 = 132.88), so they are correctly traceable — but that artifact is a 126,470-row
non-uniformly thinned export (1.0 % of the campaign; per-state retention 0.25 %-9 %, e.g.
state 21 kept 1,250 rows and state 22 kept 9,740). Its per-state medians are therefore not
comparable to full-Parquet medians. The apples-to-apples before/after is the table above.

## Known residuals in the fix (read before relying on the loud-failure guarantee)

1. **Two attempts with different drop sets are now detected, and a permutation is repaired.**
   This residual is closed. The loader cross-checks each map row's `(primary, secondary)` centre
   against the phase's own post-drop `umbrella_explicit_windows.csv` — deterministic evidence,
   not a statistic over sampled positions — so an equal-size/different-membership map is caught
   rather than silently accepted. Where the disagreement is a **permutation** (same window set,
   wrong order) the correct mapping is derivable from that same comparison and is repaired in
   memory with a loud note; where a window that really ran has no row at all, it fails closed.
   Real-data validation: 90/90 real phases repaired correctly for a rotation-by-one and for an
   adjacent swap, 90/90 genuinely-different window sets refused, 0 falsely repaired, and 0 notes
   across the 118 phases where the check runs on unmodified data.
   Two scope limits remain, both with zero incidence on the 291 real window tables measured:
   a **state_id-only swap** (two rows exchange `state_id` while both centres stay put) is
   invisible to this check and to the write-side one — no current writer can produce it, since
   both columns are written from the same in-memory window object in one pass — and two windows
   sharing an identical `(primary, secondary)` centre cannot be told apart (0 such pairs exist).
2. **The no-clobber rule is driver-side only.** `production.py`'s
   `rewrite_epoch_window_map_after_drop` is deliberately not vetoed by the samples check: after
   a *no-drop* attempt 1 it will compact a map that attempt 1's samples were logged against.
   That is caught loudly by the loader's `observed > len(survivors)` check, not silently. Do
   **not** "fix" it by adding the veto there — that rewrite must run on a genuinely fresh attempt.
3. **A mixed-attempt map cannot be correct for both attempts.** If attempt 1 left loader-visible
   samples *and* attempt 2 drops a **larger** set, the loader can "repair" the map onto attempt 2's
   window set and mis-attribute attempt 1's pooled rows. Once two attempts' drop sets differ there
   is no single map file correct for both; the ledger prevents double-application, not this.
4. **The running-and-last residual closes only when a later attempt reaches `open_segment`.**
   A campaign abandoned inside that window leaves attempt 1's rows loader-visible under a
   superseded map permanently. The refresh's breadcrumb print is the only signal in that case.

## Other results-changing caveats from this branch

- **Scheduled final phase now sizes itself from the remaining MD pool.** `_scheduled_final_default_steps`
  replaces the pool-independent fallback (`gamd_production_steps // 20`) that the scheduled-final
  path was still using while the scheduled-*epoch* path already recomputed from the pool. On this
  run's numbers that is 27 states x 500,000 steps x 4 fs = **54 ns becoming up to ~7,530 ns** of
  remaining pool — roughly two orders of magnitude more MD in the final phase, so wall-clock and
  GPU-hours for every new scheduled-final run change substantially. It cannot overrun the budget
  (`_policy_with_pool_step_budget` still clips against the pool), `--ap-final-steps` still overrides
  outright, and an absent/disabled pool keeps the old fallback. This is the fix for the starved
  late-born bridge states: the final phase produced 1.0 % of the campaign's samples, which is the
  ceiling for any state born in the last epoch.
- **`GAREUS_ALLOW_STALE_WINDOW_MAP=1`** loads a run whose samples are knowingly attributed to the
  wrong umbrella states. It is now triaged so its warning cannot sit inside a PASS verdict, and it
  is documented in `gareus -hh`. Do not leave it exported.

## Corrections to this document's own earlier claims

- The "8 disconnected components" figure that circulated during review describes the **CV-space
  nearest-neighbour pairing** graph (19 edges over 27 nodes, which must split by construction),
  **not** the thresholded overlap graph. The real connectivity result: **CV1-marginal @ 0.30 =
  1 component (silent); joint (CV1,CV2) @ 0.09 = 3 components** — `[18]` 3.4k samples, `[20]`
  6.0k, everything else 117.0k -> FAIL. A marginal-only connectivity check would have stayed
  quiet on this run; only the joint-space check sees the split.
- Suggested fix #1 in the list above ("rewrite `epoch_window_map.csv` after the post-pull drop")
  was **incomplete as written**. A second, independent drop path — the *pre-pull* seed-reachability
  filter (`dropped_unreachable_windows`) — caused 2 of the 5 shifted phases here
  (`epoch_001/topup_004`, `final/topup_003`), and both paths can fire in the same phase, the
  second legitimately supplying indices in the already-compacted space. Correctness needs an
  idempotence ledger (`<phase>/epoch_window_map_rewrites.json`), not just a second rewrite call.

## On-disk artifact format changes (external consumers)

No in-repo reader breaks, but "no in-repo reader" is not "no reader" — a notebook parsing these
positionally will:

- `overlap_matrix.csv` gains a leading `#` comment line stamping which space the numbers are in.
- `window_diagnostics.csv` gains columns **inserted mid-list**, not appended: secondary centre and
  k, observed `cv2_mean`/`cv2_std`, `self_bias_median_kT`/`self_bias_p90_kT`, and the CV-space
  overlap columns. Parse by header name, not by position.
- `pmf_summary.json`'s health block: the `metric` key under the overlap check changed meaning
  (joint if computed, else CV-space marginal, else index-adjacent). Compare it against
  `--min-neighbor-overlap` only after reading its companion space field — a joint number and a
  marginal number are not on the same scale.

## Verdict changes with no data change

Beyond the auto-drop and final-steps caveats above: **a 2D run that previously reported PASS can
now report CAUTION or FAIL purely from the new joint-overlap and connectivity checks**, on
byte-identical data. That is the intended effect — the CV1-marginal diagnostic was reporting a
different quantity than the one that had failed — but it means a verdict regression after this
commit is not evidence that anything about the run got worse. Compare the preserved marginal
numbers, which keep their exact former meaning and threshold, when checking continuity.
