# Effective adaptive top-ups — design

Date: 2026-09-24. Status: draft for review. Reviewed by the small board (kimi, mini, thinker; chair glm):
ACCEPT_WITH_CHANGES, unanimous, confidence 84/100; its conditions are folded in below.

## 1. Goal and scope

Top-ups (extra MD on a subset of states inside a scheduled epoch or the final phase) should spend MD only
where the pooled data is measurably short, and buy more precision per wall-hour than extending every state
uniformly. They stay **optional and off by default** (`--ap-topups`, YAML `ap_topups`); chignolin_9 runs
without them.

Validation in this spec is **synthetic only** (the `gareus/synth` harness). No MD campaign, benchmark job or
real-file spike is part of it; those are listed as later todos (section 9).

Out of scope: walker duplication (section 8), the sequential-panel layout (section 9), any change to how the
union MBAR analysis itself pools or reweights data.

## 2. Why the current top-ups are ineffective (chignolin_9 epoch_001, measured)

| # | Defect | Where |
|---|---|---|
| D1 | An edge with no measurement counts as weak: `weak = overlap is None or overlap < target`. All 177 rung edges (no energy-space overlap before the campaign-end solve) counted weak; the 84 spatial edges were healthy (median CV overlap 0.887, none < 0.1). Gate: "184 weak edges". | `adaptive_production._weak_edge_touch_counts`, convergence gate |
| D2 | Every state starts at score >= 1 and the epoch's whole remaining budget is distributed by score, so all 236 states got extra steps (3.0x the baseline in total). | `build_adaptive_epoch_schedule` |
| D3 | States are grouped into segments by identical extra-step size; on the ladder that produced one rung per segment (topup_001 = 59 states at lambda 1.0, topup_002 = 48 at lambda 0.0), so no lambda exchange inside a top-up. | `run_scheduled_adaptive_epoch` |
| D4 | Each top-up re-seeds from the conformer bank and re-pulls its windows (~25+ min for 59 windows) instead of continuing from the baseline's final replica states, adding an equilibration transient. | top-up seeding in `production.py` |

## 3. Constraints the design must respect

- **Lockstep.** A segment advances all its replicas by the same number of steps. Per-state targets inside a
  segment are impossible; a top-up of n states and length L costs its wall-time, L / T(n).
- **Throughput depends on occupancy.** Measured on chignolin_8's node with the production integrator under
  MPS: 16 contexts/GPU = 3,154 ns/day/node; 59 contexts/GPU = ~2,300 ns/day/node (CPU co-limited). So per
  context ~5x faster at low occupancy and ~1.37x more aggregate. Points below 16/GPU are unmeasured.
- **MBAR vs exchange.** MBAR validity and precision depend on pooled CV/energy overlap only. Exchange affects
  mixing (autocorrelation), which is where it enters the allocator.
- **Correctness invariants already in the codebase stay:** per-segment native window parameters, window-map
  repair, frozen GaMD envelope, kernel identity.

## 4. Design

### 4.1 Per-epoch union MBAR (the measurement)

After each baseline segment the driver builds union inputs over everything pooled so far
(`build_union_state_mbar_inputs`, which already uses each segment's native centres and applies per-state
equilibration discard and autocorrelation subsampling) and solves it, warm-started from the previous epoch's
f_k. The existing campaign-end call (`rung_mbar_overlap_from_union`) becomes a per-epoch call. Outputs, per
state k and edge (i, j):

- σ_k: asymptotic free-energy uncertainty of state k (relative to a fixed reference state);
- symmetric energy-space overlap sqrt(O_ij O_ji) for every spatial AND rung edge (`mbar_state_overlap`);
- a split-halves check: f_k from the first and second half of each state's pooled trace; a discrepancy
  beyond 2 combined σ flags k as "unconverged" even when σ_k is small (catches unsampled basins that
  asymptotic σ and ESS cannot see);
- τ_k: integrated autocorrelation time of the state's reduced-potential trace on the pooled data, with a
  conservative estimator (initial positive / convex sequence).

The solve runs on the node while GPUs are idle between segments; warm start and decorrelated subsamples bound
its cost. If it fails, the epoch falls back to "no top-ups this epoch" (never to the old allocator).

**Rule (fixes D1):** an unmeasured edge is never weak. After 4.1 every edge is measured, so this is a fallback
only.

### 4.2 What counts as short, and what fixes it

- **Statistical deficit:** σ_k above the campaign target σ*, or k flagged unconverged by the split-halves
  check. Fixed by top-up MD on k.
- **Weak edge** (symmetric overlap < 0.15):
  - either endpoint statistically deficient → the low overlap may be noise → top up both endpoints;
  - both endpoints well sampled → the gap is structural (more MD cannot create overlap between two fixed
    Hamiltonians) → route to the existing add/bridge-state proposal, never to MD.
- **Healthy state:** gets zero top-up. No default score, no distribution of left-over budget (fixes D2).
- If every state is equally deficient, the allocator degenerates to uniform extension (patch = all states).

### 4.3 Allocation in wall-hours (fixes D2/D3, efficiency core)

1. **Gain model.** Top-up MD on state k for L steps yields about L / (s·(1 + 2τ_k)) new decorrelated samples
   (s = report interval). σ_k after the top-up is predicted by scaling σ_k² with the ratio of decorrelated
   sample counts. This is a surrogate (the true MBAR covariance couples states); it is used to rank and size,
   not claimed optimal.
2. **Online recalibration.** After each top-up, the realised change in σ_k and τ_k is compared with the
   prediction; the per-state correction factor is stored in the registry. A top-up that delivered less than
   half its predicted gain halves that state's priority next epoch.
3. **Patch.** Deficit states plus the minimum set of exchange partners: for each deficit state, at least one
   partner on the same rung (nearest spatial neighbour) and at least one on another rung of its centre,
   chosen greedily by the partner's own σ (a partner that is itself below target gains from the steps it is
   forced to run). Neighbours come from the nearest-neighbour rule now in `gareus/dashboard/neighbours.py`,
   moved to a non-dashboard module shared by both.
4. **Length.** Candidate lengths L are the steps each deficit state needs to reach σ*. For each L: the patch
   is the states still short at L plus their partners; cost = L / T(|patch|) wall-hours; benefit = reduction
   of max_k σ_k, then of Σ σ_k² over deficit states, with partial (not all-or-nothing) credit. Pick the L
   with the best benefit per wall-hour. States whose need exceeds L carry to the next epoch's top-up.
5. **Throughput curve T(n).** Provisional: piecewise-linear in contexts/GPU through the two measured points,
   flat below 16/GPU (no extrapolated gain), with a configurable floor. Stored as a small table in config so
   a later benchmark can replace it without code changes.
6. **Cap.** Top-ups may take at most `ap_topup_max_fraction` (default 0.3) of the epoch's wall-hour budget
   (the pool's ns converted with T). Unused budget rolls into the next phase through the live pool. A
   healthy epoch spends about zero.

### 4.4 Execution (fixes D3/D4)

- One top-up segment per epoch, containing the whole patch; gibbs-walk exchange inside it across its
  centres and rungs.
- **Seeding from the baseline's final states.** Each top-up window starts from the binary checkpoint of the
  replica that occupied it at the baseline's last checkpoint (manifest assignment), loaded into a Context
  built from the same System; per-window umbrella and lambda globals are re-applied; the frozen GaMD envelope
  stays in stage 5. No US pull, no re-equilibration, no second equilibration discard: the segment continues
  each window's chain.
- **Seeding assertion.** Before any top-up sample counts, the seeded frame's reduced potential under window
  k's parameters must match the last logged u_k for that window within tolerance. A mismatch (e.g. a
  manifest written one swap early) aborts the top-up for that epoch with a clear error.
- **Fallback** when a binary checkpoint cannot be loaded (System or platform mismatch): positions,
  velocities and box from a portable State, plus a burn-in of 5·τ_k steps that is recorded and discarded.

### 4.5 Configuration

| Key (CLI / YAML) | Default | Meaning |
|---|---|---|
| `--ap-topups` / `ap_topups` | **off** (changed from on) | enable top-ups |
| `--ap-topup-target-sigma` | 0.10 kcal/mol | σ* per state |
| `--ap-topup-weak-overlap` | 0.15 | symmetric energy-space overlap below which an edge is weak |
| `--ap-topup-max-fraction` | 0.3 | cap on the epoch's wall-hour budget |
| `ap_topup_throughput_table` | two measured points | contexts/GPU → ns/day/node |

The old allocator path (score ≥ 1 for every state, grouping by extra size, re-pull seeding) is removed, not
kept as a mode. Configs that set `ap_topups: true` get the new behaviour.

## 5. Components

| Unit | Responsibility | Depends on |
|---|---|---|
| `gareus/adaptive/union_diagnostics.py` (new) | per-epoch union solve → σ_k, overlaps, split-halves flags, τ_k | `build_union_state_mbar_inputs`, MBAR solvers |
| `gareus/adaptive/topup_allocator.py` (new) | deficit rules, edge routing, gain model, patch, length choice, cap | union diagnostics, neighbours, throughput table |
| `gareus/layout_neighbours.py` (moved from dashboard) | nearest same-rung, same-restraint-pattern neighbours | — |
| `run_scheduled_adaptive_epoch` (changed) | runs baseline, then at most one top-up segment from the allocator | allocator |
| top-up seeding (changed, `production.py`) | load baseline checkpoints per window, assertion, fallback | checkpoint manifest |
| convergence gate (changed) | weak edges = measured and below threshold only | union diagnostics |

The allocator is a pure function of (registry, diagnostics, policy, throughput table, pool state), so it is
testable without MD and drivable from the synthetic harness.

## 6. Validation (synthetic harness only)

### 6.1 Harness extensions

- **Rungs:** each synthetic window gets a lambda; the boost is modelled as a lambda-scaled flattening of the
  analytic landscape inside the window (a synthetic dV with a known closed form), so states at one centre have
  distinct Hamiltonians and energy-space rung overlap is well defined.
- **Mixing model:** τ per window from the existing landscape-steepness model, reduced by a factor depending
  on the number of exchange partners the window has in its segment (so a truncated patch mixes worse, and
  the recalibration in 4.3.2 has something real to learn).
- **Lockstep segments:** a segment samples all its windows for the same length; wall-time is charged with the
  same throughput table the allocator uses.
- **Union MBAR:** the synthetic samples go through the real `build_union_state_mbar_inputs`-shaped inputs and
  the real solver, so σ_k, overlaps and split-halves come from the production code path.
- **Ground truth:** the harness's reference PMF, and repeated independent campaigns (many seeds, cheap) for
  the true spread of f_k.

### 6.2 Scenarios and success criteria

Arms at equal modelled wall-hours: (A) uniform extension (top-ups off), (B) new top-ups. N = 20 seeds per arm
per landscape.

| Scenario | Landscape | Pass |
|---|---|---|
| Heterogeneous difficulty | `rugged_2d`, `gated_barrier`, `slow_cv2_double_branch` with a lambda ladder | B has lower median max_k σ_k and lower PMF RMSE vs reference than A; paired over seeds, one-sided p < 0.05 |
| Homogeneous | a smooth single-well landscape | B within 5 % of A (allocator degenerates to ~uniform, spends ≤ cap) |
| Structural gap | a layout with a deliberate missing bridge | allocator routes the edge to a bridge proposal and spends no top-up MD on it |
| Unmeasured edges | diagnostics with rung overlaps absent | no edge counted weak, no top-up triggered by them |
| Seeding | Reference-platform Pep-GaMD fixture | seeded frame's u_k matches last logged u_k; an injected off-by-one manifest is caught |

Unit tests alongside: unmeasured-is-not-weak; healthy states get 0; noise vs structural routing; minimal
partner selection; length choice on a toy cost/benefit table; cap and rollover; degeneration to uniform.

## 7. Risks

- σ_k is asymptotic and optimistic under poor mixing; split-halves and τ-based gain are the mitigations.
- The throughput table is provisional (two points, another peptide's node); a wrong curve mis-sizes patches.
  The table is config, so a later benchmark replaces it.
- Binary checkpoint loading across segments is unproven on real files; the fallback path exists and the
  real-file spike is a todo.

## 8. Deferred: walker duplication

Extra replicas of deficit states inside the next all-state segment (no partner waste, full exchange graph).
Needs spare context capacity (chignolin_9 is at the MPS client limit), variable replicas per state in the
driver and manifest, velocity re-randomisation and RNG separation on cloning, and a short discard before
duplicates count. The board judged it complementary to top-up segments, chosen per epoch by predicted σ
reduction per wall-hour. Separate spec when needed.

## 9. Todos created by this spec (not part of it)

- chignolin_10: real-MD test of the new top-ups against uniform extension (occupancy-controlled, paired
  seeds, replicates), after the synthetic validation passes.
- Throughput benchmark: T(n) at 4, 8, 16, 24, 32, 48, 59 contexts/GPU on the production node.
- Real-file seeding spike: load chignolin_9 baseline checkpoints into top-up contexts, check the assertion.
- Sequential-panel benchmark: whole campaign as sequential ~16 ctx/GPU panels vs one 236-context run
  (board's largest efficiency lever; mixing cost across panels unmeasured).
