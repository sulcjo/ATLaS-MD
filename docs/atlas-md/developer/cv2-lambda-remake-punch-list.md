# CV2/λ-ladder remake — punch list (2026-09-30)

Status: ATLaS-MD v0.8.4 released (PR #120, tag v0.8.4). CV2/λ-ladder remake (P1, 3.2, 3.3 R1-R3, 3.7, respring) merged to main, all flags off by default, **zero MD run with any of it**. This list tracks what's left before/around the chignolin_10 launch.

## A. chignolin_10 configuration (decided with the user 2026-09-30)

Scope: full CV2/lambda remake stack with staged risk -- measuring/reporting features and bounded
actions on, R3 flag-only, respring on at a low cap. One campaign, attribution via the per-feature
reports.

| # | Decision | Setting |
|---|---|---|
| 1 | Scope | full stack, staged risk |
| 2 | CV2 selection | `--cv-selection-rank slowness` (default; c9 swarm picks tICA component 7, psi(P4) + psi(D3)) |
| 3 | Swarm | reuse c9 `swarm/round_000` (copy into c10), re-run `--swarm-stage analyze` on main only -- no new MD, frozen envelope identical to c9 |
| 4 | Layout / capacity | `--swarm-cv2-layout shape`, `max_replicas` 236, 4 rungs, `--swarm-adaptive-reserve-fraction 0.10` (23 reserved -> 53 base centres, ~5 centres of headroom); cap-bound shape layout accepted |
| 5 | Edge metric / neighbours | `--ap-edge-metric pairwise-mbar`, `--layout-neighbour-rule restraint-width`; pairwise-mbar is the CLI default since 2026-09-30 (branch `feat/pairwise-mbar-default`; c10's YAML sets it explicitly anyway) |
| 6 | 3.3 R1-R3 | `--ap-cv2-resolution` on, `--ap-refine-r3-mode flag`, `--ap-refine-transition-count replica-path`, 4 x parent k2 cap kept |
| 7 | Respring | `--ap-cv2-respring` on, `--ap-respring-max-fraction 0.10` (<= 5 centres per epoch) |
| 8 | lambda ladder | `--ap-ladder-adapt respace`, defaults (min overlap 0.25, q10, max 2 moves, hysteresis 0.03); starts from c9's [0, 0.235, 0.636, 1] |
| 9 | Top-ups / retirement / budget | `ap_topups: false` (R2 not requested), `--ap-retire-converged` off (default; F4), `md_budget_ns` 10000, `ap_epochs` 4 |
| 10 | Everything else | tICA update flags unset (C3); uncalibrated knobs at defaults (refine_min_transitions 10, refine_min_sigma 0.1, min_mean_compression 0.5, coupling gate off/0.25, rung 0.15/0.25, respring min n_eff 200 / tolerance 0.05); physics + platform as c9 (r7 seeds, sigma0 6, gibbs-walk, MPS 50 %, 8 active/GPU, blocking sync off, keep-4 checkpoints) |

Prepared 2026-09-30 on aurum2 (not submitted): `~/gareus/chignolin/chignolin_10.yaml` + `chignolin_10.sh` (copies in
`RUNS/`), `chignolin_10/swarm/{round_000,system}` copied from chignolin_9 (174/174 members done, byte-identical;
`analysis/` deliberately NOT copied -- its sidecar holds c9 absolute paths). First job: skips swarm MD, re-runs analyze
with the c10 flags, charges ~174 ns (174 members x 1 ns) to the 10 us pool, starts production. After it: check the CV2
pick, layout and reserve in `chignolin_10/swarm/analysis/layout_plan.json`, and that `ladder_run_args.yaml` points only
at chignolin_10.

## B. Deploy (closed 2026-09-30)

- [x] ~~c10 own CODE_DIR~~ -- moot: chignolin_9 killed for good (last job 2753503, ran bad9029, CANCELLED 18:05), so c10 uses the shared `~/2026_peptide_sampler`.
- [x] Commit: shared CODE_DIR = 2d92a7d (v0.8.4), deployed 18:37 by another session; both aurum2 trees md5-identical to main (188/188 `.py`).
- [x] Discrepancy: the MEMORY.md line was right (v0.8.4 deploy happened after c9 was killed); hub memory updated.

## C. Code left unwired from this spec (branch `fix/punch-list-c-items`, uncommitted)

- [x] C1 `s['cv2_resolution']`: no code needed. Union-Parquet `d.prod_dir` = adaptive root, where the final-combined summary is written; `find_summary` discovers it (covered by `test_discovery_reads_only_the_final_combined_summary`). The line as proposed would pass a Path where a dict is expected.
- [~] C2 3.1 edge metric:
  - [x] post-union refresh: `refresh_edge_metric_after_union` (end of `_apply_union_edge_overlap`) re-grades counts/tags/`components` from the union `mbar_overlap` -- R1 (fed the refreshed scheduled-epoch payload) no longer bridges a split the union joins. Only active with `--ap-topups` + `pairwise-mbar` (the union exists only when a top-up ran). Pre-union bytes identical to main. `tests/test_edge_metric_post_union_refresh.py`.
  - [x] YAML key: `ap_edge_metric` already works (generic dest mapping); pinned by `tests/test_edge_metric_yaml_key.py`.
  - [ ] **VERY IMPORTANT TODO (deferred 2026-09-30 by user): retirement/redundancy on the pairwise metric.** `retire_converged` + `_non_neighbor_redundant_pairs` grade redundancy (>= `redundant_overlap` 0.45) and the poor-edge veto (< `target_overlap` 0.30) on the CV1-MARGINAL overlap even under `pairwise-mbar`. The marginal cannot see CV2 separation (c9 epoch_002 chain medians: marginal 0.91 vs joint 2D 0.37 vs pairwise ~0.24), so on a 2D NON-ladder campaign it can retire CV2-distinct windows (retire/3.3-add churn). Work: switch both to the pairwise value (union, else BAR), recalibrate 0.45/0.30 on the pairwise scale (T2 synthetic + c9 replay), wire `refine_protect_epochs` + same-epoch split/retire exclusion (spec 3.1). Inert under an active lambda ladder (every representative is an articulation point), so no effect on a ladder c10. **Until built: set `retire_converged` off for any 2D non-ladder campaign.**
- [x] C3 3.5 priority order: main apply already runs bridges > add_rung/respace > 3.3 resolution > respring. Only coverage (`tica_coverage_add`, separate later apply) is out of order, and it needs an in-campaign tICA refit, which a frozen residual pair refuses and which c9 never enabled (`tica_obs_interval: 0`, `tica_update_after_epochs: null`). Unreachable on c10 **given the A1 constraint: leave tICA update flags unset**. No code.
- [~] C4 respring:
  - [x] `ladder_overlap_by_axis` spring-aware lambda key + tied-slot chain pairing; c9 registry pairs identical (177/196/108). `tests/test_ladder_overlap_respring.py`.
  - [ ] whether k2' realises 0.5 compression -- needs MD (T4).
  - [ ] lambda > 0 not read by respring -- by design (one k2' applied to every rung); revisit only if T4 shows per-rung F'' differs.
- [ ] C5 `resolve-f`: blocked on data. T2 9.10 ran the calibration; it failed the pre-registered rule; nothing to implement until c10/T4.

## D. Spec items not started at all

- [ ] X2, X4, X6 (see `docs/superpowers/specs/2026-09-29-adaptive-cv2-resolution-design.md` Section 13).
- [ ] T4 (follow-on validation after real chignolin_10 MD exists).
- [ ] Default flips (turning any of the above on by default) — blocked on T4.

## E. Pre-existing bugs, unrelated to this spec, still open

- [ ] `load_epoch_csv_adaptive`'s NaN-object dict-key default → real `KeyError` risk on CV1-only production runs through the legacy epoch-CSV loader path (`gareus/mbar_analysis/loaders_adaptive.py`). Fix recipe already written in CLAUDE.md, not applied.
- [ ] `gareus/mbar_analysis/pmf.py`'s `_bridge()` checks `sys.modules['analyze_gareus_mbar']` before `__main__`, can resolve to a pre-`parse_args()` script copy. Narrowed to 2 of 21 bridged names by Plan A5; fix deferred to (currently unscoped) Plan A6b.
- [ ] `--sambar-*`/`--mbar-anderson-history` argparse defaults are sticky across repeated `parse_args()` calls in one process. Narrow reachability (multi-call-per-process only), not fixed.

## F. Validation still owed

- [ ] Effective top-ups (PR #104): synthetic study showed no PMF gain; real-MD test on chignolin_10 still to run.
- [ ] Replica-admission cap + MPS 25% (+78% node ns/day benchmark, jobs 2664328/2665264): not yet applied to any production launcher config.
- [ ] X8 discovery census: per-epoch hook cost at c9 scale (~10 min between epochs) is extrapolated from a read-only replay, never measured live.

## G. Adversarial verification findings (2026-09-30, branch `fix/cv2-verification-f1-f3`)

| ID | Finding | Outcome |
|---|---|---|
| F1 | CV2 row PASSed with every rule `unavailable` | Fixed: rule-completeness metadata; incomplete/no report/legacy = CAUTION; R1 under the marginal metric = CAUTION (user choice b); respring-only (recorded policy.cv2_resolution false) = nothing requested; final-combined R2/R3 labelled as carried from the newest epoch, not re-evaluated |
| F2 | Respring n_eff/blocks from the CV2 series' g, not the variance's; guard merged sources; g = 1 on failure | Fixed: g_variance = max(g_cv2, g_q) (user choice), source-local stratified blocks, skip on failure. c9 replay: decisions nearly unchanged |
| F3 | Tied old/new states on a single-slot row read as a connectivity split | Fixed: slot-level components (single-slot groups still count as one expected component; no per-group not_applicable field) |
| F4 | Retirement on CV1 marginal | Deferred (C2, VERY IMPORTANT TODO) |

Respring still not validated by production MD; its knobs (min n_eff 200, tolerance 0.05, cap 0.25) stay uncalibrated.

## Stale branches (deferred by user, not on this list's critical path)

`fix/residual-cv-thermodynamic-consistency`, `feat/pep-gamd-internal-boost`, `docs/atlas-md-scientific-manual`, `fix/parquet-segment-manifests-v2` (PR #90 closed). **Never merge** `perf/centroid-run-plan` (marked DO NOT MERGE).
