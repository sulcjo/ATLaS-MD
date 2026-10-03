# CV2 bridge sets: allocate whole barrier bridges, not scattered fills

Status: approved direction (user, 2026-10-03), spec for implementation. Off by default.

## 1. Problem (measured)

chignolin_10 epoch_000 (first contact-map campaign, 2026-10-03) read pairwise MBAR overlap
≈ 0 on 15 of 21 same-column CV2 edges (`secondary_chain`).

- **The metric is right.** 13 of those 15 edges had no exchange attempts. On attempted edges, pairwise overlap and exchange acceptance agree.
- **The layout is cap-bound.** The CV2 shape layout designed 171 centres over 15 CV1 columns, but the 236-replica cap (4 rungs, 10 % reserve) left room for 53. As a result:
  - 143 of 194 layout requests were dropped;
  - every column kept only its two CV2 mode ("well") windows;
  - the sampled CV2 means are 6–17 sd apart.

The V3 redesign (3 rungs to λ = 0.45, 13 columns) raises the spatial budget to 71. A dry re-analysis of the same swarm then grants 69 requests, but **13 gaps above 3 σ_w remain**, because the extra slots are spread thinly.

**Per-column geometry** (V3 dry run, mixture profile F(z) = −RT ln p_mix(z), every column):
- two wells near c2 ≈ −0.8 and ≈ +1.0;
- a barrier of 2.6–5.7 kT;
- spinodal width (where F'' < 0) of 0.30–0.94 CV2 units;
- F''_min from −19 to −88 kcal/mol/CV2².

**Numerical check:** predicted window densities p_k(z) ∝ exp(−(F + ½k(z − c)²)/RT) on a 4001-point grid, overlap Σ p_a p_b/(p_a + p_b) dz.
- The fills the current placement designs between the wells (4–7 per gap, 70 in total) reach adjacent overlap **0.28–0.35**, which is on target.
- **One** fill at the target width, stabilised against the barrier curvature, reaches 0.00–0.09.

So the current placement and springs (narrow fills that inherit the well F'' under the 0.5 compression floor) are right. **The allocation is wrong:**
1. A fill is granted alone, but a partial bridge connects nothing.
2. Fills are ranked by owned swarm density (`owned_mass × weight_share`), which is lowest on the barrier, so barrier fills are dropped first.
3. R1 bridges a weak CV2 edge with **one** midpoint window, which by the numerical check cannot connect a 3–5 kT barrier.

## 2. Change

Three parts. Each is off by default and byte-identical when off.

### 2.1 Bridge sets in the shape layout (`--swarm-cv2-bridge-sets`, YAML `swarm: cv2_bridge_sets`)

**Definition.** In each column's `CV2Placement`, a *bridge set* is the maximal run of `fill` centres strictly between two consecutive placement anchors of kind `mode`. A *tail* is a fill or edge centre below the first anchor or above the last one.

Each bridge set records:
- column, lower and upper anchor positions, fill positions and centres;
- `n_fills`;
- `score = weight_share / n_fills` (connectivity bought per spatial state);
- `predicted_min_overlap`: the minimum adjacent predicted overlap along [lower anchor, fills…, upper anchor] on the column's mixture profile (§2.4).

**Sparse-branch allocation.** Tiers are granted in order until `fill_cap`:
1. mandatory stacks (unchanged);
2. mode tier: mode cells + X7 windows (unchanged);
3. **barrier chain** (§2.2), atomic: granted only if all its windows fit;
4. connectivity states (unchanged);
5. **bridge sets**, atomic:
   - Order: by `score` descending, then column, then lower anchor position.
   - A set is granted only if every fill fits the remaining budget; otherwise it is skipped (`reason: bridge_set_does_not_fit`) and the next one is tried.
   - Sets whose `predicted_min_overlap < BRIDGE_MIN_PREDICTED_OVERLAP` (0.15) are still eligible but carry a warning.
6. **tails**, by today's greedy rule (`_greedy_fill`, adjacency + owned mass), restricted to tail fills;
7. bridge fills of sets that were not granted are never granted alone (`reason: bridge_set_not_granted`).

**Joint branch** (everything fits): unchanged. All cells are granted, the barrier chain is added, and bridge-set records are still written.

**Record.** `cv2_shape.bridge_sets` (a list of the per-set records above, plus `granted` and `reason`), `cv2_shape.barrier_chain` (§2.2), and `cv2_shape.bridge_sets_enabled: true`. Every request gains `bridge_set` (index or null).

### 2.2 CV1-free barrier chain

**Pooled profile.** The union of every column's components, each weight × the column's `weight_share`. This is a single `CV2MixtureFit` built from records, with no new fit, so it can be replayed from `layout_plan.json`:
- `pooled_variance` = the `weight_share`-weighted mean of the columns' pooled variances;
- `n_members` = the maximum over columns.

**Placement.** `place_cv2_centres(pooled_fit, envelope, …)` with the same settings. Anchors within `mode_merge_sigma` merge as today, so near-identical per-column wells collapse to the pooled wells. The chain is the union of the pooled placement's bridge sets: fills strictly between consecutive pooled anchors.

**Windows.** Each chain window is a CV1-free (k1 = 0), CV2-restrained window (role `axis`, cell `(None, j)`), kind `barrier_axis`, with its k2 from the pooled model. The chain's ends are the X7 mode windows when present. The chain is atomic (tier 3).

**Why.** It connects the two CV2 wells for the whole campaign at the cost of one bridge (~4–6 spatial states instead of 13 × 4–7).

**Record.** `barrier_chain` {centres, k2, f2, predicted_overlaps, predicted_min_overlap, granted, reason}.

### 2.3 R1 bridges whole sets (`--ap-cv2-bridge-sets`, frozen in `DECISION_SETTINGS_FIELDS`, YAML `ap_cv2_bridge_sets`)

When R1 proposes a bridge for an edge between eligible states a and b:
1. Find the layout column whose `centre1` equals the endpoints' shared CV1 centre (within 1e-6) in the campaign's `layout_plan.json` (`cv2_respring_io.find_layout_plan`), and rebuild its `CV2MixtureFit` from the record (`fit_from_record`).
2. Compute the fills between a.c2 and b.c2 with the column's placement model (`bridge_fills(fit, z_a, z_b, …)`, the same `_walk` step rule and springs as the layout).
3. Children are all those fills, at nominal CV1 = the endpoints' c1/k1, with k2 from the model, each through `gate_child` (coupling gate, k bounds).
   - One child gives an `add` (as today).
   - Several give an `insert` action at parent a, whose applier path validates the whole action before executing it (atomic, every rung, duplicate check).
4. Cost = n_rungs × n_children, against the 3.3 resolution budget. A set that doesn't fit is refused with `resolution_budget` (never partially funded).
5. No layout plan, no matching column, or a fit that can't be rebuilt: today's single midpoint bridge, with `bridge_mode: midpoint_fallback` recorded.

The candidate records `bridge_mode` (`set` | `midpoint_fallback`), `n_children` and `predicted_min_overlap`.

### 2.4 Predicted overlap (shared helper)

`gareus/adaptive/cv2_shape.py`:
- **`predicted_overlaps(fit, centres, k2s, temperature_k, n_grid=4001)`:**
  - F(z) = −RT ln p_mix(z) over all components (`mixture_logpdf`), on a grid spanning the centres ± 6 sampled σ;
  - p_k ∝ exp(−(F + ½k(z − c)²)/RT), normalised;
  - returns the adjacent overlaps Σ p_a p_b/(p_a + p_b) dz (the two-state MBAR overlap scale, 0–0.5, for umbrella-only bias differences).
- **`bridge_fills(fit, z_lo, z_hi, *, sigma_w_target, temperature_k, k_min, k_max, spacing_sigma, min_mean_compression, n_grid)`:** fills strictly between z_lo and z_hi with springs, reusing `_ShapeModel` + `_walk`.
- **`fit_from_record(record)`:** rebuilds a `CV2MixtureFit` from `CV2MixtureFit.as_record()`.

The swarm mixture is a design measure, not an equilibrium profile. The predicted overlap is a design diagnostic, never a gate on MD.

## 3. Unchanged on purpose

- Fill placement, springs, compression floor and the X7 rule.
- The joint branch.
- The uniform layout.
- Respring: windows with F''_prod ≤ 0 (barrier) are already never re-sprung.
- R3.

## 4. Tests

**Layout:**
- Synthetic two-well columns under a cap:
  - every granted bridge set is complete;
  - no partial set is granted;
  - sets are granted in score order;
  - a too-large set is skipped and a smaller one granted;
  - tails are granted only after the sets.
- The barrier chain is granted atomically with role `axis`, k1 = 0.
- Flag off: requests, ranks and plan are byte-identical to today on the existing shape fixtures.

**Helpers:**
- `predicted_overlaps` on a single Gaussian with equal springs matches the analytic two-Gaussian overlap (to 1e-3).
- On a double well, today's fills give ≥ 0.25 and a single midpoint < 0.1.
- `fit_from_record(fit.as_record())` round-trips.
- `bridge_fills` equals the interior fills of `place_cv2_centres` between the same anchors.

**R1:**
- With a layout plan, the R1 bridge on a two-well column emits one `insert` with all fills and cost n_rungs × n.
- Without a plan, the midpoint fallback is byte-identical to today.
- The budget never funds part of a set.
- Flag off: byte-identical.

**Replay:** rebuild the V3 dry-run columns from its `layout_plan.json` (fits + placements + modes) and re-run `design_shape_layout` with the flag on. Report granted sets, chain and gaps. Acceptance:
- every granted set and the chain have `predicted_min_overlap` ≥ 0.15;
- at least one complete bridge exists (the chain counts);
- flag off reproduces the recorded requests.

## 5. c10 use

**Both flags stay OFF for the chignolin_10 relaunch** (decision after the adversarial reviews, section 7):
plain V3 (3 rungs to lambda 0.45, 13 CV1 columns). The code ships, off by default, for later use
once production evidence shows a column whose CV2-free window does not connect its wells.

## 6. Risks

- **The pooled profile mixes columns** whose well positions differ slightly (≈ ±0.2). The merged pooled anchors can sit between per-column wells, so the chain's ends may overlap their X7 windows less than designed. This is reported (`predicted_min_overlap` uses pooled anchors), not corrected.
- **Bridge sets cost 4–7 states each.** Under a tight cap, few columns get one. That is deliberate (a partial bridge connects nothing), but per-column CV2 coverage on the barrier stays sparse. The barrier chain plus CV1 exchange within each well carries the connectivity.
- **R1 sets can drain the reserve in one epoch.** At most one set per component pair per epoch, as today.

## 7. Adversarial review (2026-10-03) and what changed

Two independent reviews (code; statistical mechanics, with chignolin_10 epoch_000 production data).

**The premise is weaker than section 1 states.** Every CV1 column already has a CV2-unrestrained window.
In epoch_000 it overlaps both of its column's well windows (two-state BAR >= 0.12, both sides) in
11 of 15 columns, and in the low-CV1 columns it crosses 13-24 times (state series, so exchange swaps
are included). A zero well-to-well edge is therefore expected and is not a disconnected column. The
exceptions are c1 +0.48 / +1.57 / +1.81. Also, production CV2 sd is about 0.72x the swarm prediction
for stiff windows, and well weights are off by up to 0.5 in places, so the swarm profile is a weak
design measure.

**Changes made after review:**
- Predicted overlap is documented as an UPPER bound: it is an equilibrium overlap and cannot see
  trapping. `internal_barriers_kT` (the barrier inside each window's own predicted density) is now
  recorded per fill, with a warning above 1 kT.
- Scoring is by density, (weight_share / sigma_w1) / n_fills. A CV1-loose column (CV1 window wider
  than 2x the median, e.g. at the CV1 k floor) is ineligible: such windows barely restrain CV1 and
  duplicate the CV1-free chain. Before this change the ranking chose exactly those three columns.
- Near-duplicate fills (within 0.5 sampled sigma of an anchor) are dropped from sets and never granted.
- The chain is deduplicated against uniform rows and X7 windows. It is granted only if it raises the
  CV1-free bottleneck (rows + X7 + chain, on the pooled profile) by >= 0.05. On V3 it does not
  (0.278 -> 0.305), so it is not proposed.
- Refusal reasons (`bridge_set_does_not_fit`, `cv1_loose_column`, `near_duplicate`,
  `barrier_chain_does_not_fit`, chain `no_gain` / `all_duplicates` / `no_barrier`) are no longer
  overwritten with "cap".
- R1 sets:
  - each child seeds from its nearer endpoint (`seed_source_by_child`, applied per child by the applier);
  - near duplicates are dropped;
  - compression bookkeeping is the same as for the midpoint;
  - overlap is computed after the gate;
  - a placement failure falls back to the midpoint.
  - A set that does not fit the resolution budget falls back to today's midpoint
    (`midpoint_after_budget`); on V3, resolution slots (<= 11) are below the smallest set (12 states).

**V3 replay after the changes (flag on):**
- the chain is not proposed (no_gain);
- 4 whole bridges are granted: columns 9, 10, 8, plus one further set, 19 fills in total;
- 3 warnings for fills straddling a 1.0-2.9 kT barrier inside their own window;
- with the flag off, the recorded grants are reproduced exactly.

Open, not built: rank sets by need, i.e. columns whose CV2-free window does not cross in
production, which is evidence only epochs can give.
