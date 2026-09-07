# Adaptive production under a λ-ladder: rung-preserving insertion and energy-space rung diagnostics

Status: bounded change to `gareus/adaptive_production.py` (+ one helper in `gareus/mbar_analysis/ladder.py`).
Motivation (2026-09-07): the λ-ladder core made `gamd_lambda` a per-state property and the union MBAR reweights
rungs exactly, but adaptive production still adapts as if states were CV-only:

1. Every state an action adds (`add`, `tica_coverage_add`, `split`) is created with `gamd_lambda = 0.0`
   (`WindowStateRegistry.add_state` default), so an inserted CV1 window exists only at λ = 0: the windows × rungs
   cross product is broken and that window is never boosted. `has_near_duplicate` ignores λ, so replicating a
   centre across rungs would be rejected as a duplicate.
2. Edges come from `build_geometry_edges` over CV centres and are scored by CV-histogram overlap. Two rungs at
   one centre have CV overlap ≈ 1 by construction whatever the boost spacing, so the gate cannot see a rung gap.
   Exchange acceptance only annotates a warning, and under gibbs-walk (kept, by decision) the heat-bath choice
   inflates per-pair acceptance (91–95 % in the S3 pilot against a true pairwise overlap of 0.24–0.30), so
   acceptance is not a usable rung diagnostic either.

Measured calibration (S3 pilot attempt 8, σ0 = 6, rungs 0/.1/.25/.5/1, `RUNS/aurum_pilots/ll_pilot_s3`):
MBAR state-overlap matrix O_ij (definition below) has adjacent-rung entries 0.298, 0.250, 0.240, 0.273 and
diagonal 0.36–0.66. The reweighting estimate of pairwise Metropolis acceptance for the same ladder is 0.39–0.55.

## Change 1 — rung-preserving insertion

- `WindowStateRegistry.rung_lambdas() -> list[float]`: sorted distinct `gamd_lambda` over **active** states.
- `WindowStateRegistry.has_near_duplicate(primary, secondary, policy, gamd_lambda=0.0)`: a state is a duplicate
  only if the centre matches (existing tolerances) **and** `abs(state.gamd_lambda - gamd_lambda) <= 1e-9`.
  All existing callers pass λ (grep them; default keeps old behaviour for callers that have no rung).
- `apply_actions`: for kinds `add`, `tica_coverage_add`, `split`, when `rung_lambdas()` contains any λ > 0,
  create **one state per rung** with identical centre/k/parent/source and `gamd_lambda = λ`, reason suffixed
  `"; rung lambda=<λ>"`. When the ladder is inactive (all λ = 0) behaviour is byte-for-byte unchanged.
  Action tuples are unchanged (1-D/2-D params). `policy.max_new_windows_per_epoch` counts **centres**, not states
  (state the rule in its docstring).
- After `apply_actions`, whatever feeds `args.state_gamd_lambdas` / the production replica build must see the new
  rungs: grep how the adaptive loop derives per-state λ (`_derive_state_gamd_lambdas`, registry rows → windows
  CSV, `set_replica_lambda_for_window`) and add a test that a freshly added centre yields states at every rung in
  the registry rows and in the windows CSV the loop writes.

## Change 2 — rung edges scored in energy space

- `build_geometry_edges`: group active states by centre (round `primary_center` to `policy.duplicate_primary_tol`
  and `secondary_center` to `duplicate_secondary_tol`; pass `policy` or the tolerances in). Within each centre
  group, sort by λ and add edges `(state_i, state_j, "rung", Δλ)` between **adjacent** rungs. Build the existing
  `primary_chain` / `nearest_2d` edges over **one representative per centre group** (the λ = 0 member if present,
  else the lowest λ), so cross-centre edges are not multiplied by rungs and same-centre pairs never appear as
  geometry edges.
- New helper `mbar_state_overlap(u_nk, f_k, n_k) -> np.ndarray` in `gareus/mbar_analysis/ladder.py`:
  `W_nk = exp(f_k − u_nk) / Σ_l N_l exp(f_l − u_nl)` (log-sum-exp stable), `O_ij = Σ_n N_i W_ni W_nj`.
  Properties to test: rows sum to 1 (Σ_j O_ij = 1), symmetric up to the N scaling only when N_i = N_j
  (assert `O_ij·N_j == O_ji·N_i` numerically), identical states give O_ij = N_i/(N_i+N_j)… — write the test
  against a two-state synthetic u_nk with an analytic answer and against the pilot's adjacent values above
  (loadable fixture: copy the five-column `umbrella_reduced_bias_nk` and `window` arrays into a small `.npz`
  under `tests/fixtures/` — no runtime dependency on `RUNS/`).
- `EdgeDiagnostics` gains `mbar_overlap: Optional[float] = None`. At both edge-diagnostic sites (~L2131 and
  ~L3653; unify into one helper if the two loops are the same code) rung edges get `mbar_overlap = O_ij` and
  their `overlap` is **not** the CV histogram (leave `overlap=None` for rung edges); warning
  `"low_rung_overlap"` when `mbar_overlap < policy.min_rung_overlap`. Acceptance is still recorded for rung
  edges but never produces `low_exchange_acceptance` for them (gibbs-walk), with a one-line comment why.
  O_ij comes from the union MBAR solve (the block around L2660 already has `mbar.f_k`, `u_nk`, `N_k`); compute it
  once there and pass it to the diagnostics.
- New policy fields: `min_rung_overlap: float = 0.15`, `target_rung_overlap: float = 0.25`,
  `max_new_rungs_per_epoch: int = 1`.
- `propose_actions_from_diagnostics`: a rung edge with `mbar_overlap < min_rung_overlap` proposes
  `("add_rung", lambda_mid, reason)` with `lambda_mid = (λ_i + λ_j)/2`, at most `max_new_rungs_per_epoch` per
  epoch (weakest first). `apply_actions` creates the new rung at **every** active centre (cross product
  preserved; k0 scaling follows through `state_gamd_lambdas`). Rungs are never moved or removed. The frozen
  envelope is untouched (adding a rung does not recalibrate).
- Quality gate (`evaluate_adaptive_quality_gate` and the union-MBAR gate around L3752): a weak rung edge counts
  as a gate failure exactly like a weak CV edge, with the reason naming both λ values and the O_ij.

## Out of scope
Rung placement for round 0 (the swarm stage's `design_lambda_ladder`), the Pep-GaMD instability at k0 ≥ 0.4,
retirement of rungs, any change to gibbs-walk.

## Tests (fixture-free, zero-argument functions, no test-runner import; file `tests/test_adaptive_ladder_rungs.py`)
1. registry with rungs {0, .5, 1}: `apply_actions([("add", None, (0.05, 800.0), "r")])` creates 3 states at
   0.05 with λ 0/.5/1; with all-λ=0 registry it creates exactly 1 (unchanged behaviour).
2. `has_near_duplicate` distinguishes λ: same centre, λ 0.5 vs 1.0 → not a duplicate.
3. `build_geometry_edges` on 2 centres × 3 rungs: 4 rung edges (2 per centre), 1 `primary_chain` edge between
   the λ=0 representatives, no edge between same-centre states of type primary_chain/nearest_2d.
4. `mbar_state_overlap`: two-state analytic case; row sums 1; pilot fixture reproduces 0.298/0.250/0.240/0.273
   within 0.01.
5. diagnostics: a rung edge with O_ij 0.05 gets `low_rung_overlap`; `propose_actions_from_diagnostics` yields one
   `add_rung` at λ_mid; applying it creates the rung at every centre.
6. quality gate fails on a weak rung edge with a reason that names both λ.

## Global constraints
Ab initio (no native reference anywhere). Conventional commits, no Co-Authored-By or other trailers. Test runner:
a repo hook rejects any Bash command containing the literal name of the Python test runner; the sanctioned runner
is `opencode run "…"` — try it once with a 10-minute timeout; if it prints nothing and does not launch, use the
fixture-free fallback recipe from `.superpowers/sdd/2026-09-07-swarm-stage/global-constraints.md` at the repo root
and say so in the report. Keep new functions under ~50 lines; do not reformat unrelated code.

## Implementation notes (2026-09-07, commits 59cb8f8..)

- **O_ij source.** The spec's pointer to "the union MBAR solve … around L2660" names
  `_compute_mbar_weights_for_tica`, which is the tICA reweighting block: opt-in
  (`tica_obs_interval <= 0` returns immediately), run only on selected epochs, over a
  different frame subset (dihedral obs), and wrapped in `except: return uniform`. It
  cannot supply a rung diagnostic for the registry states. The implementation uses the
  union NPZ written by `build_union_state_mbar_inputs` — the only matrix carrying every
  state's reduced potential with the ladder boost already folded in
  (`apply_ladder_boost_to_u`) — via `rung_mbar_overlap_from_union`.
- **The per-edge metric is symmetric.** `mbar_state_overlap` returns `O = diag(N) @ S`, so
  `O_ij != O_ji` whenever `N_i != N_j`, and adaptive extension makes unequal per-state
  sample counts normal. A raw `O[i, j]` would make the gate value depend on which of the
  edge's two states holds the lower state id, so rung edges store
  `sqrt(O_ij * O_ji) = S_ij * sqrt(N_i * N_j)` (`_symmetric_state_overlap`). It equals
  `O_ij` when `N_i == N_j`, so the calibration above and the 0.15 / 0.25 thresholds are
  unchanged.
- **"Not scored" is not "weak".** `evaluate_adaptive_quality_gate` runs three times per
  campaign and only the post-union pass can have O_ij. An unscored rung edge is warned
  (`rung_overlap_unavailable`) and never counted as weak; failing on it would make
  `quality_gate_fixable_by_more_final_sampling` true on every ladder run and spend
  `final_quality_extension_rounds` of MD chasing a number more sampling cannot produce.
- **Accepted limitation 1 — a rung gap is caught at campaign end.** `add_rung` is proposed
  only from the post-union diagnostics: `collect_segmented_epoch_diagnostics` (the path
  feeding `propose_actions` in the real driver) passes no `rung_mbar_overlap`, so its rung
  edges arrive unscored and `_propose_rung_actions` declines to act on a missing
  measurement. A per-epoch union solve is what would move the repair in-loop.
- **Accepted limitation 2 — `retire_converged` is inert under an active ladder.** Every
  centre representative carries its own rungs and is therefore an articulation point of
  the rung-aware graph; non-representative rung states earn no CV-overlap credit. Both
  guards are pinned by `test_no_ladder_state_is_ever_proposed_for_retirement`.
