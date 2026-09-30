# ATLaS-MD: adaptive machinery that sees and resolves CV2

Status: proposed specification, v0.4 -- revised after a three-way adversarial review
(code grounding, statistics/physics, failure modes; Section 9) and a small-board review
(ACCEPT-WITH-CHANGES, 2-1; conditions folded into the body, record in Section 10), a
self-verification pass (Section 11) and exploration optimizations (Section 12). Date: 2026-09-29.
Branch: `feat/cv2-conditional-tica` (builds on the conditional-tICA CV2 selection there).
Code references are to that branch's working tree; line numbers drift.

## 1. Problem

The CV2 selector can now choose a slow, bimodal CV2. The layout and adaptive layers were built
for a fast, uniformly gridded one:

| Stage | Current behaviour | Evidence |
|---|---|---|
| Swarm layout, CV2 centres | one global `linspace` between rung-reweighted 2/98 % quantiles | `ladder_design.py:556-586` |
| Swarm layout, k2 | per-gap rule `RT/(spacing/1.5)^2`, uniform in practice because the centres are a linspace (chignolin_9: 1.18 everywhere) | `ladder_design.py:589-604` |
| Sparse layout | joint cells chosen along a fixed index diagonal; at CV1 >= 0.82 chignolin_9 has CV2 rows +0.28 and +1.35 only | `ladder_design.py:465-481` |
| Layout headroom | the layout fills `max_replicas` exactly (chignolin_9: 236/236 active, 0 adaptive adds); no slot is ever free for an adaptive state | `ladder_design.py:462-481` |
| Adaptive edge overlap | CV1-sample histogram in all four places it is computed (`_build_edge_diagnostics`, the segmented pooling that scheduled epochs use, the final combined collector, `_non_neighbor_redundant_pairs`); CV2 gaps invisible unless exchange acceptance < 0.08 | `adaptive_production.py:2139, 6131-6142, 6222-6226, 2309-2369` |
| Local CV2 resolution | for a residual CV2 no action adds CV2 resolution or stiffens k2: bridges interpolate k between endpoints; `split` has no producer; the one k2-stiffening path (`_propose_tica_coverage_actions`) runs only after a tICA refit | `adaptive_production.py:4878-4909, 7278, 2687-2860` |
| Coupling CV2 -> CV1 | gated at selection (k2_ref 1.0 vs the global k1 ceiling); at swarm layout only a warning at the maximum deployed k2 (CV1 width shrink > 10 %); adaptive k2 never checked | `select_pair.py:161-167`, `analyze.py:682-687` |
| Replica cap | adaptive additions unchecked; production truncates the window list with a print, after the window-map repair, so truncated windows keep map rows | `production.py:6716, 6730-6750` |
| Labels / refits | sidecar always writes `residual-torsion-pc`; an opt-in tICA refit (or a resumed recorded switch) would overwrite residual CV2 centres | `analyze.py:749`, `adaptive_production.py:3018, 2558-2575, 7679-7690` |

Motivation (computed in-session, not stored as an artifact): re-running the new selector on
chignolin_9's swarm picks conditional tICA component 7 (psi(P4) 0.32 + psi(D3) 0.13). On
chignolin_9 production frames (sampled under the deployed PC1 restraint, so biased), native
D3N-T8O H-bond frames (analysis-only label) sit at CV1 > 0.75 in a narrow band of that
coordinate (0.44 +/- 0.15) against +/- 0.56 for all high-CV1 frames. The deployed layout would
give such a band two windows of width 0.71 and has no way to notice.

## 2. Objectives, non-goals, invariants

Objectives: (O1) every CV2 gap is measured, on one metric, and can trigger action; (O2) CV2
windows are designed from the local free-energy shape per CV1 region, with explicit budget;
(O3) where evidence says CV2 is under-resolved, the campaign can add resolution within a
reserved budget; (O4) no deployed or adaptive k2 breaks the CV1 overlap design; (O5) no state
is silently dropped and no campaign is bricked by a cap check; (O6) artifacts say what CV2 is.

Non-goals: selecting CV2; any native information; kinetics; CV1 definitions; the lambda
envelope; multiple CV2s (auxiliary-CV spec).

Invariants: no native information in any decision; a state_id's Hamiltonian never changes (any
centre/k change creates a new state_id -- asserted); changes only at epoch boundaries; every
state change goes through the registry and the phase window-map funnel
(`_write_phase_window_map`) with its no-clobber vetoes; retired states stay in the union MBAR
with their native per-epoch parameters; campaign method settings are frozen at campaign start
(Section 3.0 P8); fail closed on missing provenance, never on a running campaign's resume.

## 3. Design

### 3.0 Prerequisites (no behaviour change on their own)

- **P1 Layout headroom.** `--swarm-adaptive-reserve-fraction` (0 until O3 is enabled): the
  swarm layout leaves that fraction of `max_replicas` unfilled, recorded in `layout_plan.json`.
  Without headroom O3 is unreachable on ladder layouts (retirement is inert under a ladder).
  The default is set from T3's cap-ignoring dry-run (95th percentile of per-epoch need, plus
  the immediate-bridge case of 3.3 R1); 0.15 is a placeholder. On a 4-rung ladder the reserve
  buys few centres (0.15 x 236 = 35 states = 8 centres), so Section 12 X1 (rung reallocation)
  is the preferred source of headroom. Within the reserve, add_rung may take at most 1/3 of
  the free slots per epoch; resolution actions at most 1/2 of what remains.
- **P2 Split/insert applier rewrite.** Resolve the whole centre (all rungs) via the centre
  key; refuse if any member is an anchor or axis state (structural: k1 = 0 or k2 = 0) or
  mandatory; validate every child first (duplicate check, `_clamp_secondary_k`, coupling
  3.4, budget 3.5); only then add children on every rung and, where the action retires,
  retire every rung of the parent. One atomic unit for budgeting and for the ledger (P3).
  `_adaptive_production_converged` treats split/refine as blocking convergence.
- **P3 Applied-actions ledger and resume idempotence.** Write
  `epoch_NNN/actions_applied.json` (actions, post-apply registry digest) atomically with the
  registry save, and keep `state_registry_pre_epoch_NNN.json`. On resume, an epoch with an
  applied-actions file never re-proposes; its diagnostics use the pre-action snapshot. This
  also closes the existing hazard for adds (kill between `registry.save` and the summary
  write re-enters epoch N with states its window map does not have).
- **P4 CV2 data in every collector.** Both epoch collectors and the final combined collector
  keep paired (CV1, CV2) samples per state (or, for memory, paired per-state subsamples plus
  moments up to fourth order) and the per-state restraint parameters; the segmented pooling no
  longer overwrites edge overlap with a CV1-only value.
- **P5 Pair-model threading.** The adaptive driver loads the frozen candidate set and pair
  model (`args.secondary_cv_candidate_set`, restored on resume) and rebuilds the fit with
  `from_candidate_set`. Only for `residual-torsion-pc` with a bound, digest-verified model;
  otherwise coupling is recorded `NA` and nothing is blocked.
- **P6 Restraint-aware identity.** Every centre key (`_centre_group_key`,
  `has_near_duplicate`, `ladder_overlap` centre key) includes the restraint pattern
  (k1 > 0, k2 > 0); placeholder coordinates of unrestrained axes are never identity.
- **P7a One neighbour rule.** One per-pair restraint-width neighbour rule (normalise each
  axis by the pair's own sigma_w, not a global median; an unrestrained axis uses the pooled
  sampled sd) in a shared module, delivered before 3.1.
- **P7b Layout schema v2 and its consumers.** `layout_plan.json` v2 lists every state
  explicitly (c1, k1, c2, k2, region, role); readers of v1 keep working. The exchange graph
  (`windows.py`), top-up partners (`layout_neighbours.py`) and `ladder_overlap` (which gains a
  CV2-direction axis) switch to the P7a rule.
- **P8 Frozen method settings.** `run_manifest.method_settings` records edge metric, layout
  mode, refine on/off and thresholds at campaign start; resume honours them unless an
  explicit override flag is given; each epoch's action report stamps the metric used. A code
  deploy never changes a live campaign's decision rules.

### 3.1 One edge metric: two-state MBAR overlap

- Every spatial edge is graded on the symmetrised pairwise MBAR overlap sqrt(O_ij O_ji):
  - pre-union: a two-state solve (BAR, float64, log-sum-exp) on the two states' own samples.
    Spatial edges are graded between same-rung states (the lambda = 0 representative of each
    centre, as today; rung edges separately). Both states share lambda and the frozen
    envelope, so the Pep-GaMD boost (a function of V_pep and V_dih, not of the CVs) is the same
    function of configuration in both and cancels; the reduced-energy difference is the two
    umbrella terms on the paired (CV1, CV2) samples -- exact and dimension-free. Samples of a
    state pooled across epochs are valid because a state_id's Hamiltonian never changes;
  - post-union: `pairwise_state_overlap` with the union f_k (already written onto spatial
    edges by `_apply_union_edge_overlap`; the weak predicate and warnings start reading it).
- Thresholds: `min_rung_overlap` / `target_rung_overlap` (0.15 / 0.25), now for spatial and
  rung edges alike; carried over, calibrated in T2 before defaults flip. The CV1-marginal and
  joint-histogram overlaps stay as reported diagnostics only.
- Sample sufficiency and decisions: tau by blocking (Flyvbjerg-Petersen) with its
  uncertainty; an edge is graded only with an effective sample count >= `min_edge_neff`
  (default 200) per state, below which it is "unmeasured" (never weak -- same rule as today).
  The weak/ok decision uses the upper 90 % bound of the overlap from a block bootstrap
  (block >= 2 tau): an edge is weak only if confidently below threshold; the point estimate
  is recorded.
- Graph: all same-rung pairs within a restraint-width radius (P7a rule) are measured
  (K ~ 250, cheap); connectivity is decided on connected components (`overlap_components`),
  not on a chain. Spanning guarantee for connectivity checks: axis states (k1 = 0 or k2 = 0)
  and the anchor get explicit edges to their nearest restrained neighbours; a test asserts
  chignolin_9's registry stays connected.
- Retirement and redundancy (`retire_converged`, `_non_neighbor_redundant_pairs`) read the
  same metric; a state created by refinement is not retired within `refine_protect_epochs`
  (default 2); retire/extend actions on a state being split in the same epoch are dropped.

### 3.2 CV2 window design from the local free-energy shape (swarm layout)

`--swarm-cv2-layout {uniform,shape}`; `uniform` is today's behaviour.

- Regions: the existing CV1 region inventory (`build_region_inventory`); the conditional CV2
  distribution for a CV1 window uses a CV1 kernel of that window's width.
- The swarm distribution is a design measure (seeded short runs), not an equilibrium
  conditional. It is used only as an upper bound on resolution and to locate candidate
  structure: a Gaussian mixture per CV1 window; BIC selects the component count,
  member-blocked cross-validation selects the covariance regularisation; a component is
  accepted only with >= `min_mode_members` (default 8) independent swarm members. This is a
  heuristic; R2 (3.3) is the swarm-independent fallback when it misses structure.
- F''_est per accepted component = RT / (component variance), the variance shrunk toward the
  pooled region variance with weight n_members / (n_members + 8), floored so F''_est >= 0.
- Placement: a mandatory centre at every accepted component mean; then fill the support by
  stepping with the minimum predicted sampled sigma over [z, z + delta] (not the value at z),
  using the predicted sampled distribution under k2 + F''_est (the CV1 curvature design rule
  of `ladder_design.py:288-356`, applied to CV2): k2 = RT/sigma_w^2 - F''_est, floored at
  `cv2_k_min`, clamped at `cv2_k_max`, then the coupling gate (3.4).
- Budget: requests from all regions are ranked by predicted contribution to connectivity and
  PMF variance (not by narrowness); the sparse fill keeps mandatory cells, then the P1
  reserve, then ranked requests; `layout_plan.json` v2 records granted and dropped cells.
- Top-rung support: the per-rung reweighted support of today's `reweighted_cv2_centers` is
  kept as the outer envelope of the placement.

### 3.3 Resolution actions

Eligibility is structural: only states with k1 > 0 and k2 > 0; never anchors or axis states.

- R1 (primary) CV2-gap bridge, for edges whose endpoints differ mainly in CV2:
  - structural: if the edge's absence splits the overlap graph into components (a component
    boundary MBAR cannot cross), bridge immediately -- more sampling cannot connect it;
  - weak inside a connected graph (confidently below threshold, 3.1): bridge;
  - unmeasured inside a connected graph: extend sampling first; bridge only if it is still
    unmeasured or weak after 2 epochs.
  Bridges go through the existing `add` path, with the CV2 spring from 3.2's shape rule
  instead of endpoint interpolation.
- R2 coverage hole: post-union, a CV2 interval at fixed CV1 whose unbiased weight is
  concentrated in < `coverage_min_windows` windows, or whose block-bootstrap PMF sigma exceeds
  `refine_pmf_sigma_kT`, gets a window at the interval centre.
- R3 mode resolution (an insertion, not a retirement): a window whose CV2 samples show two
  modes by the 3.2 mixture test (depth >= 1 kT, both modes >= 10 %) AND observed
  within-window transitions between them (>= `refine_min_transitions` in its trajectories).
  Two child windows at the modes; the parent is kept (it bridges the barrier), so no
  resolution action retires anything and the `split` action's retirement path stays unused.
  No transitions: the window is flagged `trapped_or_orthogonal` and nothing is inserted (a
  hidden slow mode projecting onto CV2 cannot be resolved by more CV2 windows; Section 12 X3
  and X4 address it).
- Diagnostics only (never triggers): sampled sd / sigma_w, Sarle bimodality, PMF curvature
  inside a window. Under a harmonic restraint sd^2 = RT/(k2 + F''), so these measure
  landscape confinement, not missing resolution, and they fire at walls.
- New windows: sigma_child from the shape rule so neighbour spacing is 1.5 sigma (for a pair
  of children at +/- delta around the sampled mean: sigma_child = 2 delta / 1.5); centred on
  the sampled mean or the mixture modes, never the nominal centre; child k2 <= 4x parent k2
  per epoch; never below `refine_min_sigma`.
- Seeding: children get `seed_source_state_id = parent`; the state-aware seed assignment is
  re-run after apply; the start is the parent frame nearest the child centre, with a
  declared burn-in discarded.
- Budget: resolution actions draw only on the P1 reserve and never more than
  `refine_budget_fraction` (0.5) of the free slots per epoch left after add_rung. Cost per
  action on an n-rung ladder: R1/R2 one centre (n states), R3 two centres (2n states).

### 3.4 Coupling gate at deployed and adaptive k2

`cv2_coupling_fraction(fit, j, k2, window)`: the CV1 curvature the CV2 umbrella induces,
d2/dc2 [k2 (z - z0)^2 / 2] = k2 [(dz/dc)^2 + (z - z0) d2z/dc2], with dz/dc and d2z/dc2 from the
fit's regression rows (dz/dc = -(K1 + 2 K2 a)/(sigma_j sigma_c) varies with a for a degree-2
residual, so the maximum over a_w +/- 2 sigma_w1 and |z - z0| <= 2 sigma_w2 is well-posed),
divided by the designed CV1 curvature k1 + F''_1,est = RT/sigma_w1^2 of that window. Anchor
values outside the fit's training clamp are evaluated at the clamp and logged. For k1 = 0
states: bound the predicted CV1 mean shift (tilt k2 (z - z0) dz/dc) and width change instead
(axis states are not eligible for refinement, but deployed k2 still applies). Called for every
window at swarm layout (replacing the warning-only check), in `_clamp_secondary_k`, and for
every adaptive k2. Above `max_coupling_fraction` (0.25) k2 is lowered to the largest passing
value; below `cv2_k_min` the state is not created; a lowered k2 is re-graded by 3.1 the next
epoch and can trigger R1. O4 is enforced only for `residual-torsion-pc` with P5's bound,
digest-verified model; for other CV2 types (an explicit scope limitation) the gate is `NA` and
adaptive k2 is capped at the layout's k2 for that region.

### 3.5 Replica cap

- `max_replicas = 0` means unlimited (today's CLI meaning).
- Before starting a new phase the driver checks `len(active states) <= max_replicas`; above
  it the phase refuses to start with counts and the remedy. A phase with an existing
  checkpoint resumes with its checkpointed window set and a warning -- a cap check never
  bricks a resume.
- Every apply call (main and post-coverage) reads the budget from the live registry;
  priority weak/disconnected-edge bridges > coverage > add_rung > resolution actions.
- `production.py`: for plain (non-adaptive) runs the truncation moves before the window-map
  repair so no map row names an unrun window; adaptive phases never reach it.
- Default at merge: the check is on (it only refuses to start phases that would exceed the
  cap, which today lose states). Release note: while the reserve is 0 (until the O3 flip),
  adaptive additions on full layouts are refused loudly, where today they are truncated. The
  priority order is a no-op until 3.1/3.3 land.

### 3.6 Labels and refit safety

- Sidecar and manifests record `cv2_component_family`, `cv2_component_index`, tICA lag in ps.
- A campaign whose frozen pair is residual refuses tICA refits and centre overwrites as a
  campaign invariant, checked in `_maybe_update_tica_cvaux`, `_apply_tica_centers_to_registry`
  and at resume (a recorded switch on such a campaign fails closed with a message).
- Contract: "all PCA indices precede all tICA indices, no gaps" replaces "tICA iff index > 6";
  old artifacts validate byte-for-byte; consumers read the `family` field.

### 3.7 Reporting

Per state: sampled CV2 mean/sd, sigma_w, confinement ratio, mixture modes, transition count,
`trapped_or_orthogonal` flag. Per edge: pairwise MBAR overlap, marginal and joint-histogram
diagnostics, space stamp. `ladder_overlap` CV2-direction axis; `gareus_report` "CV2
resolution" row; `plot_adaptive_diagnostics` sample grid by explicit state coordinates.

## 4. Configuration

| Flag | Default at merge | After T2/T3 |
|---|---|---|
| `--ap-edge-metric {marginal,pairwise-mbar}` | marginal | pairwise-mbar |
| `--swarm-cv2-layout {uniform,shape}` | uniform | shape |
| `--swarm-adaptive-reserve-fraction` | 0 | 0.15 |
| `--ap-cv2-resolution` (R1-R3) | off | on |
| `--ap-enforce-replica-cap` | on | on |
| thresholds: `min_edge_neff`, `min_mode_members`, `refine_min_transitions`, `refine_pmf_sigma_kT`, `refine_budget_fraction`, `refine_protect_epochs`, `refine_min_sigma` | spec values | calibrated |

All recorded in `method_settings` (P8). YAML: `cv_selection:` has a short-name map in
`gareus/config.py`; `adaptive_production:` / `swarm:` keys are added the same way (verify
the flattener at implementation).

## 5. Validation

T1 unit (each with a failing test first):
- two-state MBAR edge equals the union pairwise value on a synthetic two-state set;
  unmeasured below `min_edge_neff`;
- chignolin_9's registry stays connected under the new graph; axis states and anchor edged;
- both collectors (flat and segmented) produce CV2 moments and paired samples;
- applier: atomic whole-centre insert/split, refusal on axis/anchor/mandatory, duplicate
  check with restraint-aware key, children replicated on every rung;
- resume: kill between registry save and summary write, resume, no re-proposal, no
  missing-state error; epoch-N diagnostics on the pre-action snapshot;
- no state_id changes Hamiltonian (assertion on every apply);
- retired parent stays in the union MBAR with native per-epoch params;
- shape layout: shoulder mixture (80 % N(0,0.71) + 20 % N(1.2..1.6, 0.15)) gets a centre at
  the narrow mode; single Gaussian reproduces a uniform-equivalent layout;
- R3: bimodal-with-transitions splits; bimodal-without-transitions flags, no split;
  plateau between hard walls: zero actions; stiff bowl: zero actions;
- coupling at a_w for degree-2 fits; `NA` without a bound model;
- cap: refusal at phase start, resume never refused, priority order, add_rung counted;
- label/refit invariants including the resume path; contract ordering rule.

T2 synthetic: the harness gets an adapter that runs the real collectors
(`collect_*_diagnostics`) and proposers on its samples, in `--mode langevin` (reports tau).
Landscapes: `slow-cv2-double-branch`, `gated-barrier`, `harmonic-bowl` (F'' < k2: zero
actions; plus a stiff-bowl variant), new `narrow-cv2-band-at-high-cv1` (shoulder geometry),
plateau with walls, and a 3D landscape with a hidden slow CV3 projecting bimodally onto CV2;
swarm input built as a design-measure swarm (short correlated runs from stratified seeds).
Metrics vs today's machinery at matched budget: MBAR PMF error against the analytic FES,
worst pairwise edge overlap, connected components, number of actions by kind, verdict flips
pre- vs post-union. Calibrates the thresholds before any default flips.

T3 retrospective: replay chignolin_9's recorded samples through the new edge metric and the
proposer in cap-ignoring dry-run. Stated limits: chignolin_9's CV2 is residual PC1, not the
conditional tICA component; R2 needs per-epoch union solves (top-ups were off).

T2 additionally reports false-weak / false-strong edge rates against N_eff at the 0.15
threshold, asserts end-to-end that `slow-cv2-double-branch` is resolved (PMF error in both
branches below a pre-set tolerance, not merely that actions fired), and measures union-MBAR
memory against `max_replicas` and `UNION_PEAK_BYTES_PER_CELL`.

T4 real MD: chignolin, campaigns per arm set by a power analysis from chignolin_7/9
between-block variance (at least 3), pre-registered spread estimator, at matched node-hours, (contacts, cond-tIC1,
uniform, today's adaptive) vs (same pair, shape layout, pairwise-MBAR edges, R1-R3).
Pre-registered reference-free basin definition on the CV grid (native used only in the
report). Primary: between-campaign spread of the basin free energy. Accuracy: lambda = 0 vs
full-ladder crosscheck and self-bias; also connected components and worst edge per arm.

## 6. Risks

- Headroom costs sampling up front (15 % of states idle until used); T2/T4 report the trade.
- Narrow stiff windows can trap walkers; lambda transport and `refine_min_sigma` mitigate;
  T2 langevin measures it.
- More states: more contexts per GPU under MPS and more union-MBAR memory (guard at
  `UNION_PEAK_BYTES_PER_CELL`); both checked against the P1 reserve size.
- The 0.15 / 0.25 thresholds are carried over; T2 calibrates.
- The block bootstrap does not account for adaptive design choices; stated in reports.
- R3's transition test depends on trajectory length per window; short windows default to
  "flag, no split".

## 7. Delivery order (dependencies explicit)

1. P8 frozen settings; P6 restraint-aware identity; 3.6 labels/refit/contract (small, safe).
2. P3 applied-actions ledger + resume idempotence (fixes an existing hazard for adds).
3. 3.5 cap check at phase start (needs P3 for the resume rule).
4. P4 collectors; P5 model threading; 3.4 coupling helper.
5. P7a shared per-pair restraint-width neighbour rule; then 3.1 edge metric and graph behind
   `--ap-edge-metric` (needs P4, P6, P7a).
6. P2 applier rewrite; P7b layout schema v2 and the remaining consumers of the neighbour rule.
7. 3.2 shape layout and P1 reserve behind flags (needs P7).
8. 3.3 R1-R3 behind `--ap-cv2-resolution` (needs P1-P5, 3.1, 3.4).
9. T2 calibration, T3 replay; then default flips (release note; live campaigns unaffected
   by P8); T4.

## 8. Open questions for review

- Is 15 % headroom the right order of magnitude, or should headroom come from retiring
  redundant windows under the new metric instead?
- Should R3 ever retire the parent, or is "keep parent" always right under a ladder?
- Is a two-state BAR on per-epoch samples robust enough at small effective N, or should the
  pre-union regime grade "unmeasured" more often and rely on top-ups for the union solve?

## 9. Adversarial review record (v0.1 -> v0.2)

| ID | Source | Severity | Finding | Disposition |
|---|---|---|---|---|
| A1 | code | blocking | layout fills the cap exactly; refine/add always rejected under an enforced cap | P1 reserve; dry-run T3 |
| A2/C4 | code, failure | high | edge overlap computed in 4 places; scheduled path overwrites with CV1 pooling; CV2 sd dropped; CV1/CV2 not row-paired | P4; 3.1 names every site |
| A3/C2 | code, failure | high | split applier retires parent first, one rung only, no duplicate check | P2 |
| A4 | code | high | refine proposer lacks CV2 sd / moments / curvature | P4; triggers redesigned |
| A5 | code | high | synth harness never calls the real collectors | T2 adapter |
| A6/C9 | code, failure | medium-high | cap budget spans two apply sites; add_rung missing; 0 = unlimited undefined; fail-closed could brick resume | 3.5 rewritten |
| A7 | code | medium | production truncation after window-map repair | 3.5 |
| A8/C7 | code, failure | medium/high | layout cells index one global CV2 list; consumers equate row = exact CV2 | P7 |
| A9/B-H2/C14 | all | medium | sigma/4 bins far finer than the analysis calibration; small-N bias | histogram demoted to diagnostic |
| A10/C10 | code, failure | medium | coupling helper signature; no model in the adaptive driver | P5, 3.4 |
| A11/C8 | code, failure | high | new edge set could isolate axis states and brick the final | spanning guarantee, test |
| A-B1..B9 | code | false/misleading claims in v0.1 table (coupling re-check, tica coverage stiffening, "silently", uniform k2, component-7 provenance, ledger, applier obligations, union plumbing) | table corrected |
| B-C1 | stats | critical | 1 kT minima miss a narrow shoulder band; walk jumps narrow modes | 3.2: GMM/BIC, mandatory component centres, min-sigma stepping |
| B-C2 | stats | critical | sd/sigma_w and curvature triggers measure confinement; fire at walls; runaway refinement | demoted to diagnostics; R1-R3 |
| B-H1 | stats | high | histogram 0.09 vs MBAR 0.15 disagree (3.4 sigma vs 2.5 sigma gaps) | one metric (3.1) |
| B-H3/C6 | stats, failure | high | children 3 sigma apart; (b) retires the bridging parent | sigma_child = 2 delta/1.5; keep parent |
| B-H4 | stats | high | swarm density is a design measure; spurious modes; k2 rule sqrt(2) off | upper-bound use, member minimum, k2 = RT/sigma^2 - F'' |
| B-H5 | stats | high | coupling vs bare k1 wrong; k1 = 0 reference meaningless; a = 0 only | 3.4 |
| B-H6 | stats | high | cap priority inverted | 3.5 order; refine budget fraction |
| B-H7 | stats | high | in-place k2 change = two Hamiltonians per id; child seeding; retired parents in union | invariants; seeding; T1 |
| B-M1 | stats | medium | bimodality can be a hidden orthogonal slow mode | R3 transition test, flag |
| B-M2..M4 | stats | medium | chain-only edges; budget favours easy regions; rung pooling | 3.1 graph; 3.2 budget; same-rung edges |
| C1 | failure | high | k2 = 0 axis states always trigger; mandatory metadata absent in chignolin_9 | structural eligibility |
| C3 | failure | high | resume between registry save and summary re-enters epoch / double split | P3 |
| C5 | failure | high | retire_converged undoes refinements on marginal overlap | 3.1 retirement on the same metric; protection |
| C11 | failure | medium | children born without seeds | 3.3 seeding |
| C12 | failure | medium | resume re-applies recorded tICA switch | 3.6 invariant |
| C13 | failure | medium | centre identity ignores restraint pattern | P6 |
| C16 | failure | medium | four neighbour-graph definitions | P7 shared rule |
| C17 | failure | medium | default flips reach live campaigns via deploy | P8 |
| B-low, C-low | both | low | determinism, plot grid, union memory, top-up waste, bootstrap caveat, torsion stiffness ceiling | 3.7, Section 6, T1 |

## 10. Small-board review (2026-09-29) and dispositions

Board: kimi, mini, thinker; chair glm. Final: ACCEPT-WITH-CHANGES, 2-1 (mini REJECT, 78).
Transcript: `/home/sulcjo/.claude/jobs/55927a12/tmp/small_board_cv2spec/` (job-temporary).

| # | Condition | Disposition |
|---|---|---|
| 1 | P7 used by 3.1 before delivery | P7 split: P7a (neighbour rule) before 3.1, P7b (schema v2, consumers) later; Section 7 updated |
| 2 | R1 conflates unmeasured and disconnected edges | R1 acts immediately only when the edge's absence splits the overlap graph into components (structural MBAR failure); an unmeasured edge inside a connected component first gets extended sampling and must persist for >= 2 epochs before a bridge |
| 3 | F''_est estimator unspecified; GMM/BIC vs blocked CV ambiguous | F''_est per accepted component = RT / (component variance), component variance shrunk toward the pooled region variance with weight n_members/(n_members + 8), floored so F''_est >= 0; BIC selects component count, member-blocked CV selects covariance regularisation. The board's suggested assumption "boost is a function of the CVs only" is NOT adopted: it is false for Pep-GaMD (boost depends on V_pep, V_dih). Cancellation in 3.1 holds because both states of a same-rung edge share lambda and the frozen envelope, so the boost is the same function of configuration in both |
| 4 | Coupling degree-2 ambiguity; support; NA scope | Induced CV1 curvature written explicitly as k2 [(dz/dc)^2 + z d2z/dc2] (dz/dc varies with a for a degree-2 residual, so the maximum over a_w +/- 2 sigma is well-posed); evaluation outside the pair model's training anchor range (its clamp bounds) clamps to the bound and logs it; O4 is enforced only for residual-torsion-pc with a bound, digest-verified model -- other CV2 types are an explicit scope limitation, with k2 capped at the layout's k2 there |
| 5 | add_rung vs reserve | add_rung draws on the reserve, at most 1/3 of the free slots per epoch; resolution actions keep their own 0.5 fraction of what remains |
| 6 | decision-boundary statistics | tau by blocking (Flyvbjerg-Petersen) with its uncertainty; bootstrap block length >= 2 tau; `min_edge_neff` default 200 per state; weak-edge decision on the lower 90 % bound of the overlap (bridge only if confidently weak), point estimate recorded; T2 reports false-weak / false-strong rates vs N_eff at 0.15, and asserts end-to-end resolution on `slow-cv2-double-branch` |
| 7 | 15 % headroom unproven | reserve default set from T3's cap-ignoring dry-run need (95th percentile over epochs, plus the R1 immediate case); 0.15 is a placeholder until then |
| 8 | T4 power | power analysis from chignolin_7/9 between-block variance before T4; pre-registered spread estimator; campaigns per arm set from it (>= 3) |
| 9 | union memory | T2 measures union memory vs max_replicas against `UNION_PEAK_BYTES_PER_CELL` |
| 10 | interim state | release note: with the cap check on and reserve 0, adaptive adds on full layouts are refused loudly until the O3 flip |
| 11 | minor | float64 / log-sum-exp in the two-state solve and PMF slice; Sarle kept as a diagnostic only (outlier-sensitive); 3.5 priority is a no-op until 3.1/3.3 land; the swarm "upper bound" is a heuristic, with R2 as the swarm-independent fallback |

Dissent (mini) points and dispositions: boost non-cancellation under k2 changes -- does not
arise (any k2 change creates a new state_id; same-rung states share lambda and envelope);
low-N BAR bias -- handled by `min_edge_neff`, lower-bound decisions and "unmeasured" below it;
clamp could create gaps -- a clamped k2 is re-graded by 3.1 next epoch and can trigger R1;
R3 flag-only for orthogonal modes -- deliberate: a CV2 split cannot resolve a slow mode
orthogonal to CV2 (the auxiliary-CV spec covers that); headroom magnitude -- condition 7.

## 11. Self-verification (v0.4)

Issues found re-reading v0.3 against the code and the chignolin_9 data, and their fixes:

| Issue | Fix |
|---|---|
| R3 was "the only retiring action" yet kept its parent (contradiction) | R3 is an insertion; no resolution action retires; the `split` retirement path stays unused (P2 still makes it atomic for future producers) |
| A 15 % reserve on a 4-rung ladder buys only ~8 centres; R3 costs 2 centres x 4 rungs | P1 notes it; X1 below is the preferred headroom source |
| Boost cancellation was argued from "same Hamiltonian"; the board then proposed "boost is a function of the CVs", which is false for Pep-GaMD | 3.1 now states the actual condition: same rung and frozen envelope |
| 3.1 needed the neighbour rule before P7 was delivered | P7 split into P7a (before 3.1) and P7b |
| Coupling "maximum over a +/- 2 sigma" was not written as a formula | 3.4 now gives d2/dc2 of the CV2 umbrella explicitly |
| R1's "unmeasured" policy cannot act in the frozen final phase (no later epoch), and R2 needs a union solve (mid-campaign only with top-ups on) | Stated here: in the final phase R1/R2 act only through a `final_extension_NNN` round; without top-ups R2 is campaign-end only |
| A lowered (clamped) k2 could open a gap | 3.4: re-graded next epoch, can trigger R1 |

Checked and holding: spatial edges are graded among lambda = 0 centre representatives
(`_split_rung_groups`); `pairwise_state_overlap(u_nk, window, f_k, n_k, i, j)` exists
(`mbar_analysis/ladder.py:256`); the union path already writes spatial `mbar_overlap`
(`_apply_union_edge_overlap`); chignolin_9's registry is 236/236 with zero adaptive adds.

## 12. Optimizations for exploring conformational space and minima

Ranked by evidence and cost. Each is a separate, flag-gated change with its own test; none
uses native information in a decision. Numbers are chignolin_9 unless stated.

**X1 Reallocate the lambda ladder (highest value, low risk).** The top rung pair
0.636-1.0 overlaps 0.40 (0-0.235: 0.24), the two rungs have near-equal transition rates (turn
H-bond 4.9x vs 5.8x lambda = 0) and near-equal delivery shares of lambda = 0 H-bonded frames
(23 % vs 22 %), and lambda = 1.0 carries 0.4 % of the MBAR weight. Dropping it (or respacing
to e.g. {0, 0.12, 0.3, 0.65}, balancing adjacent overlaps near 0.25-0.3) frees 25 % of the
states -- 59 on chignolin_9 -- which funds P1 headroom and the high-CV1 CV2 rows without
touching lambda = 0 sampling. Test: T4 arm with the respaced ladder; the ladder crosscheck and
rung overlaps must stay >= 0.20.

**X2 Rung-sparse insertion (medium value, medium risk).** Resolution only needs statistics
on the rungs that carry weight (lambda = 0 and ~0.23: 97 % of the MBAR weight). New centres
from R1-R3 are inserted on the low rungs plus the lowest boosted rung that keeps a rung edge,
not on every rung, halving their cost. Requires relaxing "every centre on every rung" in the
registry, `ladder_overlap` and the rung graph (a centre's rung set becomes explicit);
transport to the new centre then goes through spatial exchange on the low rungs.

**X3 Slow-mode-aware reseeding at epoch boundaries (high value, low cost).** The slowest
motion left after (CV1, component 7) is psi(D3) (loading 0.39, autocorrelation 0.990 at
200 ps) -- the same torsion that separates native-like frames (about 111 deg vs -10 deg). A
walker stuck on the wrong side of it stays stuck for the whole segment. At each epoch
boundary, reseed a fraction (e.g. 25 %) of windows from the pooled end states, choosing for
each window a configuration inside its restraint that balances occupancy of the hidden
mode's states (reference-free: the hidden mode is the leading conditional tICA mode of the
residual after the deployed pair). MBAR validity is unaffected (starting points only; the
declared burn-in is discarded); cost is the burn-in. Test: T2 3D landscape with a hidden
slow CV3 (Section 5) -- CV3 state balance per window and PMF error with and without X3.

**X4 Auxiliary bias on the hidden mode on one exploratory rung (high value, high cost).**
The frozen, bounded 1D auxiliary bias of the auxiliary-CV spec, applied to the X3 hidden
mode on the top rung only, so boosted walkers cross it and deliver both states down the
ladder. Needs that spec's force-group audit (A06) and trials; after X1/X3.

**X5 Allocate MD time by effective samples (medium value, low cost).** Top-ups and
per-state segment lengths use raw sample counts. Weighting by the tau-corrected deficit
(the 3.1 blocking estimate) moves steps to windows whose samples are correlated (the slow
turn states) and away from fast, already-decorrelated ones.

**X6 Longer swarm members for selection and layout (medium value, pre-production cost).**
Swarm members are 1 ns with 450 ps discarded; candidate autocorrelations at 200 ps are
0.95-0.99, so slowness is extrapolated and each member barely crosses a slow mode. Fewer,
longer members (e.g. 3-5 ns) for the selection and 3.2's mixture fit give real timescales,
real within-member transitions for the R3-style test at design time, and more independent
evidence per mode (`min_mode_members`).

**X7 CV1-free windows per CV2 mode (low cost).** The CV2-only (k1 = 0) windows held about one sixth
(16-18 % in two independent counts) of chignolin_9's H-bonded frames although they are 8 % of
states: letting contacts relax
at fixed CV2 helps close the hairpin. The shape layout keeps one CV1-free window per
accepted CV2 mode (not only per uniform row).

**X8 Discovery census as a campaign diagnostic (low cost).** Per epoch, count new
reference-free structural states (core backbone basin strings; 2 A C-alpha clusters) and
report the discovery curve. Both chignolin_7 and chignolin_9 were still discovering at their
end; a flat curve is evidence to stop exploring and spend on precision, a rising one to keep
X3/R2 active.

**Measured and not recommended as a lever:** transferring configurations seen only on
boosted rungs to lambda = 0 windows -- 85 of 345 core states are never seen at lambda = 0,
but they are 0.01 % of frames (transient); keep as a diagnostic only. NMA-derived CVs were
assessed separately and are not competitive.

Suggested order: X1 with the P-prerequisites (it creates the headroom), X3 and X8 next
(cheap, directly target the hidden psi(D3) mode and tell whether exploration is saturating),
then 3.1-3.3, X5, X6; X2 and X4 last.

## 13. Implementation status (2026-09-29)

X1, P3 and the CV2 selection are in PR #111 (`feat/cv2-conditional-tica`). P6, 3.5, 3.6, P8 and the two
X1 gap fixes are in PR #113 (`feat/adaptive-cv2-prereqs`, stacked on #111).

| Item | Status | Where |
|---|---|---|
| CV2 selection: conditional tICA candidates, slowness ranking (the premise of this spec) | done | `gareus/cv_selection/slowness.py`, `select_pair.py`; CLAUDE.md section |
| X1 adaptive lambda ladder | done, redesigned: keeps lambda = 0 and the top rung, respaces/adds/drops interior rungs to a 0.25 minimum overlap (not "drop lambda = 1") | `gareus/adaptive/ladder_adapt.py`, `--ap-ladder-adapt`; plan `docs/superpowers/plans/2026-09-29-adaptive-lambda-ladder.md` |
| P3 applied-actions ledger, resume idempotence | done (all campaigns); kill/resume verified end to end through the real epoch loop | `_record_applied_actions` / `_load_applied_actions`, epoch loop; `tests/test_ladder_adapt_resume_e2e.py` |
| Seeding of states created by actions (add, add_rung, respace_ladder, split) | done: the nearest-seed assignment is re-run after the actions are applied; before, a segment holding only new states got the bank's first rows (`generic_fallback`) | `_reassign_seeds_after_actions`; `tests/test_seed_assignment_after_actions.py` |
| P8 frozen method settings | done: ladder settings (`ladder_adapt_settings.json`) plus the adaptive decision rules (`DECISION_SETTINGS_FIELDS` -> `adaptive_production/decision_settings.json`, mirrored to `run_manifest.method_settings["adaptive_decision_settings"]`), frozen at the first job, honoured on resume, `--ap-decision-settings-override` replaces; budgets stay per-job; later decision knobs (edge metric, layout mode, refine) append their field to the tuple; each epoch action report already stamps the full resolved policy | `_resolve_decision_settings`; `tests/test_decision_settings_frozen.py` |
| 3.5 replica cap | done except the priority order (a no-op until 3.1/3.3): phase-start refusal at all four launch sites (checkpointed phase resumes with a warning); every add/add_rung/split/coverage apply reads the budget from the live registry; plain-run truncation moved before the window table, neighbour graph and map repair, and a fast resume keeps its checkpointed window set | `_require_phase_within_replica_cap`, `AdaptiveProductionController._within_replica_budget`, `production.run_gareus`; `tests/test_replica_cap.py` |
| P6 restraint-aware identity | done: `_centre_group_key`, `has_near_duplicate`, `ladder_overlap_by_axis` (new `secondary_k`) and `ladder_adapt.centre_key` key an unrestrained axis as None; k not recorded counts as restrained, so old registries key as before | `tests/test_restraint_aware_identity.py` |
| 3.6 labels and refit safety | done: `cv2_component` (family, index, tICA lag frames/ps) in `cv_selection_report.json` and `run_manifest.method_settings`; a frozen residual pair refuses tICA refit (skip + warning), tIC1 recentring (raises) and the tica-linear switch, and a resume that finds a recorded switch fails closed; contract rule "PCA indices precede tICA indices, no gaps" replaces "tICA iff index > 6" (c8/c9 artifacts validate to the same digests) | `gareus/cv_selection/labels.py`, `_frozen_residual_pair`, `contracts._require_family_order`; `tests/test_cv2_labels_refit_safety.py` |
| P7a one neighbour rule | done (module only, no consumer switched): per-axis scale sqrt(sigma_a^2 + sigma_b^2) of the pair's own restraint widths, axes in quadrature; an axis unrestrained on both is left out (P6), restrained on one uses the passed pooled sampled sd and the free state's sampled mean (else unmeasurable, inf); same rung, same restraint pattern (spanning mode `same_pattern_only=False` for 3.1), radius 2.5 (equal-width Gaussian MBAR-scale overlap 0.058; 235 lambda = 0 edges on chignolin_9 vs 108 for `layout_neighbours`, a strict superset; the exchange graph's 36 cross-pattern lambda = 0 edges are excluded; 4 components per rung by construction: anchor, CV1-only, CV2-only, 2D) | `gareus/adaptive/neighbour_rule.py`; `tests/test_neighbour_rule.py` |
| P4 CV2 data in every collector | done (additive keys only; every existing key unchanged, checked on real data): flat, segmented and final-combined collectors attach per-state `paired_cv` (registry restraint c1/k1/c2/k2/lambda, row-paired counts and drops, population moments to 4th order per axis, cov/corr) and a bounded constant-stride subsample (<= 2,000 pairs/state, each source step-sorted, `step` recorded) in a sidecar `<json stem>_paired_cv.npz`; every edge gets `overlap_joint_2d` (pair-local 30x30 joint histogram over all pooled pairs, with `overlap_joint_2d_reason`). Overwrite audit: inside the segmented collector the per-segment `overlap` was already the CV1 marginal (no joint value existed to clobber); at the callers, a scheduled epoch's union `mbar_overlap` (spatial + rung) is applied after collection and that dict is what the driver uses, the bare re-collects (`adaptive_production.py` flat-epoch and flat-final branches) never follow a scheduled phase, and the final-combined file is built fresh (pre-union CV1 only, then re-collected with rung overlap only), so per-epoch spatial union overlap never reaches it -- a gap for 3.1, not an overwrite. chignolin_9: 0 non-null `mbar_overlap` in all 9 stored diagnostics JSONs (top-ups off). `overlap` stays the CV1 marginal (chignolin_9 epoch_002 primary-chain edges: marginal median 0.91, joint 0.37) | `gareus/adaptive/paired_cv.py`; `tests/test_p4_paired_cv_collectors.py` |
| P2 split/insert applier | done: `apply_actions` validates each action whole against the live registry (after the batch's earlier actions), then applies it; a refused action changes nothing and is recorded in `refused_actions` (ledger `refused`) with `reason` + `index`: `unknown_state`, `inactive`, `mandatory`, `anchor_or_axis`, `duplicate`, `below_k_min` (3.4 gate), `max_replicas_budget`, and for `respace_ladder` `no_ladder`/`ladder_endpoint`/`no_change`. Split: parent resolved to its whole centre via `_centre_group_key`; refused if any member is mandatory or structurally anchor/axis (k1 = 0 or k2 = 0, or no CV2 centre in a CV2 campaign; also layout roles); children clamped to the live `cv2_k_max`, duplicate-checked (P6 key, every rung, registry incl. retired and earlier children), gated, budgeted as children x rungs - parent rungs; children added on every rung, then every parent rung retired (kept for the union MBAR). Adds/coverage adds get the same child validation. Every apply asserts no existing state_id's (c1, k1, c2, k2, lambda, sigma0) changed and none disappeared. `_adaptive_production_converged` blocks on split/refine. The ceiling is the live `_resolve_secondary_k_max(args)` (not frozen: the tICA switch rewrites it); no standalone `cv2_k_min` floor. Changes vs before: within-batch duplicate adds refused; refused mandatory retire / refused respace now recorded; split retires all rungs and validates. No split producer yet; `_apply_tica_centers_to_registry` still mutates centres in place outside the applier | `tests/test_applier_p2.py` |
| P7b schema v2 + consumers | done: `layout_plan.json` v2 (`layout_plan_v2`, `schema_version` 2) lists every state with c1, k1, c2, k2, `role`, `region` (the CV1 region its centre represents; None when CV1 is unrestrained) and `restrained` [CV1, CV2]; `gareus/layout_plan.py` `read_layout_plan` reads v1 and v2 (v1 region from `region_inventory.region_of_centre`), used by both loaders (c8/c9 v1 files read unchanged). Consumers behind `--layout-neighbour-rule {legacy,restraint-width}` (default legacy; policy `layout_neighbour_rule` in `DECISION_SETTINGS_FIELDS`, and the driver writes the frozen value into every phase's args -- the rule drives MD, so a deploy never changes a live campaign's exchange graph). restraint-width: exchange graph (`windows.restraint_width_neighbor_edges`; swaps under `--exchange-mode neighbor`, and the post-pull drop connectivity check under every mode) = per rung the P7a pairs within 2.5 + `neighbour_rule.chain_edges` (true-neighbour chains + pattern links, the anchor included) + `lambda_neighbor` between adjacent rungs of one centre (P7a pairs one rung only) + a connectivity bridge; top-up partners (`layout_neighbours.p7a_spatial_neighbour_pairs`). `ladder_overlap` gains `cv2_direction` (same rung, pattern and CV1 column, adjacent in CV2; not a radius) with two health rows, only when some state restrains CV2. P7a open point: k not recorded -> `fallback_axis_sigmas` (run default k, else median recorded width, else median centre spacing; source returned). `layout_neighbours._restrained_mask` treats k < 0 as unrestrained (P6). chignolin_9 (236 states): legacy exchange graph 687 edges of which 589 join DIFFERENT rungs (the legacy distance ignores lambda) and 35 same-rung edges cross patterns; restraint-width 1129 (940 P7a, 177 lambda_neighbor, 12 pattern_link); top-up pairs 432 -> 940. No MD has run with restraint-width | `gareus/layout_plan.py`, `gareus/windows.py`, `gareus/layout_neighbours.py`, `gareus/mbar_analysis/ladder_overlap.py`, `gareus/adaptive/neighbour_rule.py`; `tests/test_p7b_neighbour_consumers.py` |
| Geometry-edge fix (T3 Section 5, T2 item 2) | done, ungated (a bug): `build_geometry_edges` = `primary_chain` (CV1 within one CV2 row of one restraint pattern), `secondary_chain` (new: CV2 within one CV1 column), `nearest_2d` (nearest same-pattern state by P7a distance + closest pair joining pieces of a pattern), `pattern_link` (new: connectivity between patterns on shared restrained axes, anchors to the most central state of the largest pattern; never weak under either metric, skipped by retirement). CV1-only layouts are byte-identical (c7 real registry tested). Retirement now keeps >= 1 state (a proper 2x2 cycle retired all). chignolin_9 registry: 44 `primary_chain` + 36 `nearest_2d` (10 non-adjacent, 42 cross-pattern) -> 49 + 27 `secondary_chain` + 3 `pattern_link`, all adjacent, connected. Replay (read-only, `min_edge_neff` 200): epoch_001 weak marginal 2 -> 0, pairwise 5 -> 0 (unmeasured 7 -> 6), geometry-edge pairwise min 0.271 (CV1), 0.350 (CV2); final-combined weak marginal 6 -> 0, pairwise 2 -> 0 (unmeasured 6 -> 5; still 2 measured components, the n_eff-starved end window 60 as before), geometry-edge pairwise min 0.275 (CV1), 0.371 (CV2), pattern links 0.19-0.21. At the new default neff 100: epoch_001 and final-combined 0 weak, 0 unmeasured, 1 measured component. The artefact bridges T3 counted (3-11 centres per epoch under pairwise, 3-8 under marginal) therefore disappear. Live effect: the final quality gate counts weak spatial edges into `needs_more_sampling`, so on c9 (final-combined marginal 6 -> 0) deploying this can change whether the gate asks for more sampling. `min_active_states` floor 0 -> 1 in retirement (every campaign; tiny). `_non_neighbor_redundant_pairs` alert counts unchanged on c9 (per-segment 3881 -> 3881, pooled 0 -> 0): it compares CV1 marginals across patterns and is already saturated there, a pre-existing residual not fixed here. Changes every 2D/sparse campaign's weak edges and bridge adds, retirement articulation, the final connectivity gate, the top-up union edge list, the non-neighbour redundancy exclusion set and the convergence weak-edge count | `gareus/adaptive_production.py`, `gareus/adaptive/neighbour_rule.py` (`chain_edges`); `tests/test_p7b_neighbour_consumers.py` |
| T2 bug 1: pairwise-mbar disabled retirement | fixed: the retire loop skips the appended `neighbour`/`spanning` edges (overlap None by construction); an unmeasured GEOMETRY edge still protects its ends (deliberate reading of "unmeasured is never weak": retirement is on by default and retiring next to an edge with no data would change marginal campaigns). The strict-xfail test passes | `tests/test_synth_adapter_collectors.py`, `tests/test_p7b_neighbour_consumers.py` |
| Bridge reason text | fixed: under pairwise-mbar the reason quotes `pairwise_mbar_overlap` (two-state point + q90, or the union value) against `min_rung_overlap`, then the CV1 marginal labelled; the bridge plan record gains `graded_overlap`/`graded_overlap_source`; marginal text unchanged | |
| Thresholds (T2/T3) | `min_rung_overlap` 0.15 (weak) and `target_rung_overlap` 0.25 kept; `min_edge_neff` 200 -> 100 (T2: same false-weak/-strong rates, 35 % vs 47 % unmeasured; a frozen decision setting, so a campaign that recorded 200 keeps it). Union memory guard: 0.9 GB fixed + 61.6 B per (row x state) cell (T2: 0.8 GB + 55-58 B; the old cell-only model was 28-106 % low at 100k rows); the 8 GB guard now admits ~488k kept rows at 236 states (was ~550k), so it skips top-up diagnostics earlier | `gareus/adaptive/union_diagnostics.py` (`max_union_rows_under_guard`) |
| P1 headroom | done behind `--swarm-adaptive-reserve-fraction` (default 0 = today's bytes; swarm-only, so not in `DECISION_SETTINGS_FIELDS`; in `method_settings`): the 2-D layout fills `(max_replicas - floor(f max_replicas)) // n_rungs` spatial states, never fewer than the mandatory stacks (the reserve shrinks, `reserve_shortfall`), `cap_spatial` unchanged; `layout_plan.json` `adaptive_reserve` {fraction, max_replicas, n_rungs, reserved_replicas_requested, cap_spatial_full, fill_cap_spatial, granted_states, free_slots, reserve_shortfall, shares}, read with `layout_plan.adaptive_reserve`; a CV1-only ladder records it in the swarm report and warns when it cannot be honoured. Per-epoch split `reserve_budget.reserve_allowances(max_replicas, n_active, reserve, n_rungs=, add_rung_taken=)`: add_rung floor(free/3), resolution floor((free - add_rung_taken)/2), also in whole centres; no reserve -> add_rung ungoverned (today's rule), resolution 0; not wired into any proposer (3.3 will). Until then the existing adds spend the reserve like any free slot. At c9 scale a new rung costs 59 states against a 35-slot reserve (1/3 = 12), so the split keeps add_rung out of a small reserve | `gareus/adaptive/reserve_budget.py`, `ladder_design.design_exploration_layout`/`reserve_fill_cap`; `tests/test_layout_reserve.py` |
| 3.2 shape layout (+ X7) | done behind `--swarm-cv2-layout shape` (default uniform: all 31 swarm-analysis artifacts byte-identical to the pre-change code on the synthetic fixtures) and `--swarm-cv2-min-mode-members` (8). Math in `gareus/adaptive/cv2_shape.py` (public, pure, for 3.3 R1/R3): `fit_cv2_mixture` (weighted EM per CV1 window, CV1 kernel of the window's own width truncated at 3 sigma; BIC over K = 1..3 with n = sum of weights; regularisation reg x pooled variance, reg from {1e-4, 1e-3, 1e-2, 3e-2} by 4-fold member-blocked CV; a member supports a component with >= 3 weighted frames AND >= 20 % of its own frames -- without the fraction rule broad members' tails counted 5 narrow members as 10; stride subsample to 5000 frames), `mode_depth`/`mode_pair_resolvable` (R3: depth >= 1 kT between the means, both weights >= 10 %; 0 without an interior density minimum), `estimate_f2` (as specified, variance-space shrinkage), `shape_rule_k2`, `predicted_sampled_sigma`, `place_cv2_centres` (mandatory centre per accepted mode, modes within 0.5 sampled sigma of a heavier one merged; greedy steps of 1.5 x the minimum sigma_s over the step; edge centre if > 0.5 step remains). Choices: the base width sigma_w is the uniform layout's (spacing / overlap_sigma over the same reweighted envelope), so a single Gaussian with F'' < RT/sigma_w^2 reproduces the uniform grid exactly; F''(z) = the dominant accepted component's (pooled when none); per-column support = union over rungs of kernel- and boost-weighted 2/98 % quantiles, clipped to the global envelope. Budget (`gareus/swarm/cv2_shape_layout.py`): mandatory stacks vs the full cap, then the reserve, then mode cells + X7 CV1-free windows (one per accepted mode, deduplicated within 0.5 sigma_s of a uniform row or another mode window) by owned design mass, then today's axis states, then fill cells greedily (adjacent in their column to a granted cell, largest owned design mass = frames in the cell's CV2 Voronoi interval x column share, the stand-in for PMF-variance contribution); all fits -> kind joint. Roles stay `joint`/`axis`, `mandatory` stays anchor + representatives; `cv2_shape` records per-column fits, placements (k2, F'', sigma_s, mean compression k2/(k2 + F''), floor count) and every request (rank, tier, granted, reason; dropped cells are never states). Coupling gate applied per cell by the caller, unchanged. Real data (chignolin_9 swarm, read-only, 48,024 frames, 174 members, 66 s): 15 CV1 columns, 26 accepted modes (a recurrent narrow mode at CV2 ~ -1.4, sd 0.2-0.35, 10-16 %, in columns CV1 0.18-0.58; one at +0.8..+0.9, sd ~0.4, 11-12 %, at CV1 0.70-0.76); 63 centres (uniform: 4 per column); under the 59-state cap 25 mode cells + 5 X7 + 14 CV1-axis + 4 CV2 rows + 9 of 38 fills (uniform: 39 joint cells); with reserve 0.15, 50 spatial states, 0 fills. **The k2 floor dominates**: sigma_w target 0.711 equals the per-column swarm CV2 sd (0.6-0.75), so RT/sigma_w^2 - F'' <= 0 and 49 of 63 centres (and 4 of 5 X7 windows) floor at cv2_k_min (1e-3 there: c9's cv2_k_min 0); median k2 0.001, max 0.106 vs uniform 1.18; mean compression <= 0.09, i.e. the windows follow the mode, not their centres (analyze warns). An X7 window at the floor is an effectively unrestrained anchor that P6 still counts as CV2-restrained. Decided 2026-09-30 (user): k2 floor tied to F'' -- `shape_rule_k2(..., min_mean_compression=0.5)` (default) gives k2 >= F''_est c/(1 - c) = F''_est, so every window mean moves >= half-way to its centre; cv2_k_max still wins; 0 restores the plain rule; `n_at_compression_floor` recorded. c9 replay with it: 0/63 centres at the k_min floor, k2 1.07-2.73 (uniform 1.18), compression 0.50 at every centre (the floor always binds there, so the width target is inactive on c9), 5-7 centres per column, 57 of 66 fills dropped at the 59-state cap. Shrinkage switched to precision space 2026-09-30 (user): `estimate_f2` = RT (w/var_c + (1 - w)/var_pool), w = n/(n + 8), so a 12-member sd-0.15 mode keeps most of its curvature (variance space widened it to sampled ~0.4-0.56). c9 replay with both: narrow-mode k2 up to 13.1 (was 2.7), 5-9 centres per column, 6 X7 windows, 8 of 74 fills granted at the 59-state cap (with the (f) curvature-variance fix: k2 up to 13.45, 5-10 per column, 7 X7, 7 of 76). | `gareus/adaptive/cv2_shape.py`, `gareus/swarm/cv2_shape_layout.py`; `tests/test_cv2_shape_layout.py` |
| 3.3 R1-R3 | done behind `--ap-cv2-resolution` (default off: proposer, applier, action report and convergence gate identical to 1830f85 except the 7 new off policy keys; decision_settings.json gains them, a live record resumes untouched). Eligible: k1 > cv1_k_min and k2 > cv2_k_min ("k2 > 0" read as above the floor), representative rung, never anchor/axis. R1 (needs `--ap-edge-metric pairwise-mbar`, else `unavailable`): same-pattern geometry edges whose P7a per-axis distances have d2 >= d1; structural = confidently below threshold across two `edge_metric.components` (one bridge per component pair, closest edge) and weak = `edge_is_weak_pairwise` bridge now, at the midpoint of the sampled CV2 means, and replace the midpoint bridger on that edge (a refused R1 leaves it); unmeasured = extend both ends (a lifecycle record only: the schedule is uniform, so the edge waits through ordinary epochs), bridge once bad in 3 consecutive epochs (first sighting + 2), history `adaptive_production/cv2_resolution_history.json` stores sets of epochs per (min, max) pair, so a re-proposed epoch never double-counts and a ledger-recovered epoch never runs the hook. R2 only on this phase's top-up union (`topup_union_overlap.json` present; else `unavailable`): own MBAR over lambda = 0 rows and states (exact umbrella energies), per CV1 column intervals of one sigma_w2 widened to the slab's weighted 1-99 % range; hole = >= 2 % of the slab weight AND (< `coverage_min_windows` centres contributing >= 10 %, or bootstrap sigma > `refine_pmf_sigma_kT`); holes split at window centres; placed only >= 1.5 sigma_w2 from existing windows. R3: mixture on the P4 subsample with 32 time blocks per state as members (a mode must recur in >= 8), `mode_pair_resolvable`; transitions = core-to-core crossings within replica residences (runs broken at source, segment, step gap and replica change; cores at barrier -/+ half the narrower sd); the state-indexed count (exchange swaps included) is reported only; without a replica column the count is `transitions_lower_bound` and R3 never inserts; no transitions = `trapped_or_orthogonal`. New action `insert` (P2 applier: whole centre, anchor/axis refused, mandatory parent allowed, parent kept). Springs: F'' = `estimate_f2` - own k2 (biased samples measure k2 + F''); `shape_rule_k2` in [cv2_k_min, min(cv2_k_max, 4 x parent k2)], target sigma never below `refine_min_sigma`; refusals `k2_at_floor`, `k2_capped_below_compression` (cap or gate below the mean-compression floor), `below_k_min`; mean compression reported per child. Seeding: `seed_source_state_id` restricts the state-aware assignment to the parent's bank rows; burn-in as every window. Budget: P1 reserve only, priority R1 structural > weak > waited > R2 > R3, cost n / 2n states; add_rung capped at 1/3 of free slots with the flag on; no reserve (or unlimited cap) = `no_reserve`, recorded and blocking convergence. Report `epoch_NNN/cv2_resolution_report.json` (`cv2_resolution_report_v1`, schema in the `cv2_resolution_io` docstring). Knobs (P8-frozen): `coverage_min_windows` 2, `refine_min_transitions` 10, `refine_pmf_sigma_kT` 0.5, `refine_min_sigma` 0.1 are uncalibrated (T2); `refine_budget_fraction` 0.5 and `refine_protect_epochs` 2 are spec values. c9 dry run (read-only, cap ignored): epoch_002 0 actions (R1: 24 CV2-mainly edges, all ok; R2 unavailable, top-ups were off; R3: 36 no_action, 3 trapped_or_orthogonal with 370-469 state-series switches but 0-5 within-residence transitions; deciding instead on the state-series count all 3 are refused `k2_capped_below_compression`: upper-mode F'' 7-30 vs the 4 x 1.18 cap); final-combined 0 actions (R2 on 143,513 lambda = 0 union rows: 2-6 contributing centres per interval, bootstrap sigma <= 0.34 kT). | `gareus/adaptive/cv2_resolution.py`, `cv2_resolution_rules.py`, `cv2_coverage.py`, `cv2_resolution_io.py`; `tests/test_cv2_resolution.py`, `tests/test_cv2_resolution_wiring.py` |
| 3.7 reporting | done (additive; nothing new is written with `--ap-cv2-resolution` off unless the CLI is run). Collector `python -m gareus.adaptive.cv2_resolution_summary <adaptive_dir> [--phases ...] [--out DIR] [--diagnostics F --report F --label L]` reads each epoch/final dir's phase-level `adaptive_epoch_diagnostics.json` (never `baseline/`, so no double count), the 3.3 report when present, `adaptive_final_combined_diagnostics.json` and `state_registry.csv` (restraints for pre-P4 payloads); writes `cv2_resolution_summary.json` (`cv2_resolution_summary_v1`) + `_states.csv`/`_edges.csv` into the phase dir (final-combined: `*_final_combined.*` at the adaptive root); with the flag on the driver writes each numbered epoch's after the apply (`write_epoch_summary`, never raises). Per state: c1/k1/c2/k2/lambda + restraint pattern (P6: k <= 0 unrestrained), sampled CV2 mean/sd (P4, ddof 0), sigma_w2 = sqrt(kT/k2) (null when unrestrained), confinement ratio = sd/sigma_w2 (landscape diagnostic, never a trigger; equals 3.3's `sd_over_sigma_w`, pinned by test), accepted mixture modes, R3 mode pair, depth, transitions with estimator `replica` | `state_series_lower_bound` | `none`, trapped_or_orthogonal; states R3 did not evaluate say so. Per edge: pairwise MBAR point/q10/q90/status/reason/min n_eff, union value, CV1 marginal and joint-2D overlap, each with a space stamp (`cv1_marginal`, `cv1_cv2_joint`, `two_state_mbar`, `union_mbar`); weak = `edge_is_weak_pairwise` (not `below_threshold`); rung edges listed, never graded. `gareus_report` row "CV2 resolution", appended only when a summary exists (`s['cv2_resolution']`, else `<production_dir>/cv2_resolution_summary_final_combined.json` -- the table over the data the PMF pools; per-epoch tables are never auto-graded, c9 epoch_002 would read CAUTION against a PASS final-combined; no freshness check, the row prints the label), so existing verdicts are byte-identical (tested): FAIL for a weak pairwise edge or > 1 spatial component (lowered to CAUTION when Overlap connectivity already FAILs), CAUTION for unmeasured graded edges, trapped_or_orthogonal windows, budget refusals (no_reserve, resolution_budget, max_replicas_budget) or a metric/report error, NA for a CV1-only table. Not wired into `analyze_gareus_mbar.py` (it does not attach the block; the file discovery covers a campaign where the CLI or hook wrote one). `plot_adaptive_diagnostics` adds `adaptive_fig5_state_coordinates.png` (fig2 unchanged): every active state at its own (c1, c2), one panel per rung, marker per restraint pattern, an unrestrained axis at the sampled mean (open marker), colour = samples, trapped rings, weak/unmeasured edges. `ladder_overlap` `cv2_direction` is P7b's. chignolin_9 replay (read-only): epoch_002 276 graded edges, 0 weak, 0 unmeasured (91 below 0.15, all radius/spanning/cross), 1 component, 3 trapped (84, 200, 220; 0-5 within-residence vs 370-469 state-series transitions) -> CAUTION; final-combined 0 trapped -> PASS; confinement ratio at lambda = 0 0.56-0.84 (median 0.72) and 0.55-0.77 (0.66) | `gareus/adaptive/cv2_resolution_summary.py`, `cv2_resolution_grade.py`, `state_grid_plot.py`; `tests/test_cv2_resolution_summary.py`, `tests/test_cv2_resolution_report_row.py`, `tests/test_state_grid_plot.py` |
| 2026-09-30 follow-ups | (a) `shape_rule_k2` has a mean-compression floor, k2 >= F''_est c/(1 - c), c = `min_mean_compression` 0.5 (k2 >= F''; cv2_k_max still wins; 0 = plain width rule), so a single Gaussian reproduces the uniform grid only while F'' <= RT/(2 sigma_w^2) (not on c9, where the floor binds in every column); (b) `estimate_f2` shrinks in precision space, RT (w/var_c + (1 - w)/var_pool); (c) R3 refuses `k2_capped_below_compression` when the 4 x parent / cv2_k_max cap leaves the compression below c; (d) `--ap-refine-transition-count {replica,state-series}` (default replica, frozen decision setting, per-candidate `transitions_estimator`) selects R3's crossing count -- replica-resolved rarely fires under exchange (c9 replicas stay ~2 samples per window), state-series counts swaps too; (e) with `--ap-cv2-resolution` on, `collect_final_combined_diagnostics` writes `cv2_resolution_summary_final_combined.json` (`write_final_combined_summary`), the file the gareus_report row reads via `production_dir` (= adaptive root on the union-Parquet path). (f) review-board fixes (ACCEPT-WITH-CHANGES, 2026-09-30; supersede the 3.2/3.3/3.7 bullets where they differ): budget refusals (`no_reserve`, `resolution_budget`) never block convergence -- summary `n_blocking` counts `proposed` only, `is_blocking` recounts from `candidates` (v1 reports read the same), refusals stay in `n_refused_budget`, a gate recommendation and the 3.7 CAUTION -- and with no governing reserve the hook prints ONE WARNING per campaign job (`warn_no_reserve_once`, keyed by adaptive dir); report schema `cv2_resolution_report_v2`: `transitions_lower_bound` renamed `transitions_state_series_lower_bound` (a lower bound on the state-series count, an UPPER bound on the replica count; the summary reads both keys); state-series >= replica always (a replica run is a contiguous piece of a state run, pinned by a randomized test), so `--ap-refine-transition-count state-series` is the permissive choice (kept as an option); `k2_capped_below_compression` refusals are counted (`counts.n_refused_spring_cap`) and graded CAUTION, not blocking; every R3 candidate records `r3_gate` (`cv2_resolution.R3_GATES`: single_component, member_support, no_density_minimum, depth_below_1kT, mode_weight_below_10pct, too_few_samples, no_replica_series, transitions_below_min, passed) + `r3_gate_values`, carried per state by the summary; mixture components carry `reg_variance` and `variance_curvature` (the fit re-converged at the smallest regularisation 1e-4 x pooled, `curvature_variance`; falls back to variance - reg, never above the regularised variance) and every F'' consumer uses it (`_ShapeModel`, X7 `mode_axis_windows`, R3 children; density/BIC/`mode_depth` keep the regularised one): synthetic var-0.041 mode at reg 3e-2 read 0.089 (F'' ~54 % low), plain subtraction 0.057, re-converged 0.043 (test tolerance 10 %); `AdaptiveDecisionPolicy.__post_init__` validates `refine_transition_count`, and a bad recorded `decision_settings.json` value fails at load naming the file; `_step` docs: feasible, may undershoot the fixed point (conservative). c9 shape replay before -> after: k2 1.07-13.05 -> 1.10-13.45, centres 99 -> 101 (5-9 -> 5-10 per column), 0 -> 0 at the k_min floor, compression floor binds at every centre (0.50), fills granted 8/74 -> 7/76, 57 of 59 spatial states granted both, X7 6 -> 7 (median variance_curvature/variance 0.97, min 0.43). c9 dry runs (decisions unchanged, 0 actions): epoch_002 gates depth_below_1kT 15, no_density_minimum 11, member_support 9, single_component 1, transitions_below_min 3 (84/200/220: depth 1.55/1.10/1.65 kT, replica crossings 5/0/5 vs state-series 370/469/458); final-combined member_support 16, single_component 12, no_density_minimum 9, depth_below_1kT 2: the upper CV2 mode (84: 1.10 w 0.27 in 16 of 32 blocks; 220: 1.20 w 0.23, 19 blocks; 200: 1.27 w 0.36, 28 blocks) shrinks in the pooled data: 84 w 0.075 in 5 of 33 blocks, 220 0.83 w 0.13 in 1 block and 1.54 in 0 (member_support); 200 no density minimum between 0.49 and 0.79 (its 1.47 component: 1 block). With state-series counting the 3 become `k2_capped_below_compression` (children F'' 30.4/7.3/21.9 vs cap 4.72) -> CAUTION. Residual: the final-combined summary is built with no 3.3 report, so its trapped/budget/spring-cap/r3_gate counts are empty and the gareus_report row cannot see them in a live campaign. Still open: `analyze_gareus_mbar.py` does not attach `s['cv2_resolution']` (one line before `build_health_verdict`, left for the other writer's pending edits); at c9 scale the shape layout is cap-bound (7 of 76 fills fit 59 spatial states since (f); 8 of 74 before) and `--ap-ladder-adapt respace` frees no replicas. | `gareus/adaptive/cv2_shape.py`, `cv2_resolution_rules.py`, `cv2_resolution_summary.py`; `tests/test_cv2_shape_layout.py`, `tests/test_cv2_resolution.py`, `tests/test_cv2_resolution_summary.py` |
| 2026-09-30 follow-ups (h): report v3 | All behind `--ap-cv2-resolution`, frozen decision settings (`DECISION_SETTINGS_FIELDS` gains `refine_r3_mode`, `coverage_count`), CLI + YAML `ap_*`, validated at policy/settings construction. (1) `--ap-refine-r3-mode {flag,insert}` default flag: a passing R3 window with placeable children is `flagged`, reason `r3_flag_only`, children/springs kept in `proposal`, `metrics.would_be` {decision, refusal, r3_mode}; no action, no budget, never blocking; summary `n_r3_flag_only`, 3.7 CAUTION; a would-be spring-cap refusal stays `refused`. (2) `--ap-refine-transition-count replica-path` (new default; choices replica, replica-path, state-series): per (source, segment, replica) the replica's finite CV2 at the state in step order joined across its absences, never broken at a step gap; every candidate records `transitions_replica`, `transitions_replica_path`, `transitions_state_series`. replica <= replica-path and replica <= state-series by construction; replica-path and state-series not ordered (counterexamples pinned). c9 epoch_002 84/200/220: 22/12/21 (= t3c), all then refused at the 4 x cap. (3) `--ap-coverage-count {any,same-column}` default same-column: only centres restrained on CV1 (k1 > floor) at the interval's column count; CV1-unrestrained states in no column; centre keys drop unrestrained-axis placeholders (P6). T2 9.9: recall of moderate holes up (k20 0-7 -> 20-23 of 24) at the cost of geometric proposals (plateau-drop1, dbl-gap, long series: 43-58 proposals <= 0.2 kT at 8,000 vs 2-6); coverage_min_windows kept 2 (1 inert, 3 floods 2-row layouts); c9 final-combined 0 -> 3 proposals (sigma <= 0.14 kT). (4) R2 bootstrap blocks = ceil(5 g) rows per state (Geyer g, max over restrained axes, per sample source from the union `.samples.csv`, >= 5 blocks per state, recorded under `rules.R2.bootstrap`). T2 9.9: barely changes calibration (median err/sigma 2.34 -> 2.17 indep/2,000; re unchanged) because the old blocks were already >= 2 g; the measured root cause is the fixed f (re-solving it: 2.2 -> 1.1-1.3, calibrated 0.67), not built (open, T4). `refine_pmf_sigma_kT` default 0.5 -> 0.25. (5) Report `cv2_resolution_report_v3`, summary `cv2_resolution_summary_v2` (reads report v1-v3). | `gareus/adaptive/cv2_resolution.py`, `cv2_resolution_rules.py`, `cv2_coverage.py`, `cv2_resolution_io.py`, `cv2_resolution_summary.py`, `cv2_resolution_grade.py`; `tests/test_cv2_resolution_v3.py`; `t2_synthetic.md` 9.9, `t3_retrospective.md` 10.6 |
| 2026-09-30 follow-ups (i): respring | Done behind `--ap-cv2-respring`, which is off by default and independent of `--ap-cv2-resolution`. The knobs `--ap-respring-min-neff` 200, `-tolerance` 0.05, `-max-fraction` 0.25 and `-k2-rtol` 0.10 are uncalibrated, frozen in `DECISION_SETTINGS_FIELDS`, and set by CLI or YAML `ap_*`. **Hook:** after each numbered epoch, after the 3.3 hook, on non-recovered epochs only. **Measurement:** each CV2-restrained window on the lambda = 0 rung is re-measured from its own samples: F''_prod = RT/var - k2 (P4 full-series variance) and c_real = k2/(k2 + F''_prod), with a 5-95 % block bootstrap. Blocks are 5 x the X5 g, never across a source. **Trigger:** the whole interval must lie below 0.5 - tolerance, with n_eff >= 200. F''_prod <= 0 is recorded and never acted on; over-stiff windows are flagged only. **New spring:** k2' = `shape_rule_k2`(sigma_t, F''_prod). sigma_t is the layout's `cv2_shape.sigma_w_target`, else the sampled sd; with the sampled sd, k2' = F''_prod. Then the cv2_k_max clamp and the 3.4 gate. **Action:** one atomic `respring` per centre. The new centre goes on every rung, then every rung of the old centre is retired, for a net change of 0 states. The duplicate check is k2-aware. CV1-free X7 windows are allowed. Mandatory, CV2-unrestrained, protected, already re-sprung and other-action centres are skipped or refused. **Cap:** 25 % of centres per epoch; the rest are deferred. **Convergence:** never blocking. **Reporting:** the 3.7 table carries the counts and grades unresolved windows CAUTION. **Other change:** ladder_adapt groups rung samples by centre + springs. **c9 read-only dry run (uniform k2 1.18, `t3_retrospective.md` 10.7):** there are 43 candidates. epoch_002: c_real median 0.52, 6 triggered. final-combined: c_real 0.44, 20 triggered, 14 proposed at the cap, k2' 1.60-2.72. The conditional F''_prod equals t3c's F''_loc on all 39 2-D windows. **Not measured:** whether k2' realises 0.5 (needs MD), and the lambda > 0 rungs. | `gareus/adaptive/cv2_respring.py`, `cv2_respring_io.py`; `tests/test_cv2_respring.py`; `scripts/t3d_respring_dryrun.py`, `t2_data/t3d_respring_c9.json` |
| X2, X4, X6 | not started | |
| X5 allocate MD by effective samples | done for top-ups (`--ap-allocation-weight ess`, default raw, frozen in P8). Per-state g is the integrated autocorrelation time (Geyer initial monotone sequence, pooled autocovariance) of the state-indexed CV1/CV2 series, with runs broken at every sample dir, segment, step gap and NaN, and each run centred on its own mean; the slowest restrained axis wins. The top-up decides on sigma x sqrt(max(1, g/g_builder)), which is one-sided because g is a lower bound for slow states; recorded sigma, predictions and the MAX_STEP_MULTIPLE cap stay on the builder scale. A failed estimate records g = 1 and leaves that state uncorrected. Audit: `<phase>/effective_samples.json`. Deviation: per-state baseline lengths are NOT reweighted, because the baseline runs every state in lockstep under replica exchange. Also not done: the 3.1 blocking estimate with its uncertainty (this module is the estimator 3.1 should reuse). chignolin_9 final: g 3.0/5.0/22.2 samples (min/median/max; 1 sample = 250 steps); the slowest states are at CV1 = 0.07 and CV2 row -1.85/-0.78. g exceeds the builder's g for 43 of 236 states. At the shipped 0.10 target no state is above target: all 10 deficits come from split-halves and are identical under raw and ess, and 4 % of state-steps move through partner choice. At a target equal to the median sigma, 3 slow states (g 7-10) join and 2 % move | `gareus/adaptive/effective_samples.py`, `plan_topup(effective_g=)`; `tests/test_effective_samples_allocation.py` |
| P5 pair-model threading | done: the driver resolves the three artifact paths from args (config/sidecar) or, on a resume, the first root/phase `run_manifest.json` that records them (a swarm campaign's root manifest does not), loads them with `PairModelRuntime.load` (digests, semantics, real bindings; `from_candidate_set`) and checks `cv_pair_model_sha256`; anything but a verified `residual-torsion-pc` pair is `NA` with a reason, never an exception. Loaded per epoch only when the 3.4 gate is on | `gareus/adaptive/pair_runtime.py` (`load_driver_pair`); `tests/test_cv2_coupling_gate.py` |
| 3.4 coupling gate | done behind flags (off = today's bytes for windows, maps, registry, swarm report): `cv2_coupling_fraction` (closed-form box maximum; clamp evaluated at the clamp and recorded; k1 + F'' <= 0 falls back to k1; k1 = 0 bounds the mean shift in sigma_ref and the width change, gated at the same 0.25 -- the spec gives no separate shift threshold) and `largest_passing_k2` (quadratic in sqrt(k2)). Adaptive (`--ap-cv2-coupling-gate`, `--ap-max-coupling-fraction`; both in `DECISION_SETTINGS_FIELDS`): the weak-edge bridge's k2, after the `cv2_k_max` clamp, is lowered or the bridge refused below `cv2_k_min`; report `epoch_NNN/cv2_coupling_gate.json`. Swarm (`--swarm-cv2-coupling-gate`, `--swarm-cv2-max-coupling-fraction`): per-cell k2 lowered in the table; a cell below `cv2_k_min` fails gate `cv2_coupling` (no cell is removed). Not done: the non-residual cap at the layout's regional k2 (the registry carries no region), and `tica_coverage_add`/`split` are not gated (unreachable under a frozen residual pair / no producer). chignolin_9 (degree 1, k2 1.18): 39 restrained windows max 0.0026, median 0.0022, 0 > 0.25; 4 CV1-unrestrained windows 0.42 with the swarm's pooled CV1 sd 0.195 (mean-shift bound 0.42 sigma; curvature ratio 0.044, width change 2 %), so the gate would lower their k2 1.18 -> 0.42 -- the only change it makes to this design, and it rests on holding the shift to 0.25, which the spec does not set | `gareus/cv_selection/coupling.py`, `gareus/adaptive/pair_runtime.py`; `tests/test_cv2_coupling_gate.py`, one in `tests/test_swarm_analyze.py` |
| X8 discovery census | done (diagnostics only): per phase and per epoch group, new / cumulative reference-free states for two definitions (core backbone basin strings over residues with both phi and psi, letters A/B/P/L/O with explicit boundaries; greedy leader C-alpha clusters at 2 A, campaign order, recorded stride), raw and populated (>= 5 frames) counts, rates per 100 ns of aggregate replica time, saturation verdict on the last epoch group (saturated if its populated rate < 0.1 x campaign mean); CLI `python -m gareus.adaptive.discovery_census <adaptive_dir>`, opt-in `--ap-discovery-census` writes `epoch_NNN/discovery_census.json` after each numbered epoch's MD | `gareus/adaptive/discovery_census.py`; `tests/test_discovery_census.py` |
| X3 slow-mode-aware reseeding | done behind `--ap-slow-mode-reseed-fraction` (0 = off); hidden mode = leading conditional tICA mode of the torsion residual after CV1 and the deployed CV2 direction (from saved XTC frames joined to the Parquet samples, or `tica_obs`); one-sided = minority side <= 20 % of a window's frames; seeds = pooled `final_pdbs` end states within 2 sigma_w on the minority side that pass the seeding preflight, injected through a seed-bank `slow_mode_reseed/` override that `filter_seed_bank_for_state_ids` carries and `generate_us_starting_states_by_pulling` honours per window; report `epoch_NNN/slow_mode_reseed.json`. No MD has run with it; T2 (3D landscape with a hidden CV3) not done | `gareus/adaptive/slow_mode_reseed.py`, `slow_mode_reseed_io.py` (+ read-only replay CLI); `tests/test_slow_mode_reseed.py` |
| 3.1 edge metric | done behind `--ap-edge-metric pairwise-mbar` (default `marginal`: no key added, diagnostics identical except the two new policy fields `edge_metric`/`min_edge_neff`, which also join `DECISION_SETTINGS_FIELDS`): two-state MBAR/BAR (float64, log-sum-exp) on the P4 stride subsample, umbrella terms only (same rung, boost cancels), overlap via `pairwise_state_overlap`; tau by Flyvbjerg-Petersen blocking within each source, unmeasured below `--ap-min-edge-neff` (200 at the time; 100 since P7b) per state; moving-block bootstrap (block >= 2 tau), weak iff the q90 (upper) bound < `min_rung_overlap` (Section 10 item 6's "lower bound" read as a slip: only the upper bound means confidently weak); union `mbar_overlap` wins when present. Graph: P7a radius pairs (same pattern) + 2 spanning edges per axis/anchor state; components verdict keeps unmeasured edges (chignolin_9 lambda = 0: 1 component). Deviations: weak-eligible are only the collector's same-pattern geometry edges (radius edges reach overlap ~0.06 by design: 86 second/third-neighbour edges on chignolin_9 epoch_002 were below 0.15 without being gaps; acting on components is R1's); cross-pattern edges never weak; exchange acceptance no longer makes a spatial edge weak; retirement/redundancy and `refine_protect_epochs` still on the marginal (not done; the appended edges disabled retirement until the P7b fix); record stamped `stage: pre_union` (not refreshed after the union apply). Real data (read-only): chignolin_9 epoch_002 weak edges 2 (marginal, both cross-pattern) -> 1 (CV2-only chain 64-76, point 0.080, marginal 0.78); chignolin_7 epoch_002 0 -> 0 | `gareus/adaptive/edge_metric.py`; `tests/test_edge_metric_two_state_mbar.py` |
| T1 | per-feature unit tests with every item above | |
| T2 synthetic | done for everything short of 3.3: harness runs the real collectors, proposer and applier (MALA sampler); threshold calibration (0.15 kept, `min_edge_neff` 100), coupling (0.25 safe for width, blind to tilt), X3, union memory; end-to-end resolution test written and skipped until 3.3 | `docs/superpowers/specs/2026-09-29-adaptive-cv2-validation/t2_synthetic.md`; `gareus/synth/t2_*.py`; `tests/test_synth_adapter_collectors.py` |
| T3 retrospective | done (c9, c7 read-only): found the `build_geometry_edges` non-neighbour chain bug (fixed in P7b); no real gaps on c9, so the failure threshold is not locatable from it; coupling-gate shift model mispredicts c9's CV2-only windows | `docs/superpowers/specs/2026-09-29-adaptive-cv2-validation/t3_retrospective.md` |
| T4 real MD | not started | |

Notes from implementation: the X1 replay found that both chignolin_7 and chignolin_9 keep
4 rungs and move the interior down ([0, ~0.18, ~0.47, 1]), so the extra headroom X1 was
expected to free (Section 12) does not appear under a 0.25 minimum; P1 headroom still has to
come from somewhere else.

X3 replay on chignolin_9 (read-only, 200 ps lag, 3.5 ps frame stride): chignolin_9 deployed
residual PC1, not component 7, so its hidden mode is the one after (CV1, PC1): phi(E5) 0.81 +
psi(P4) 0.52 (epoch_002; epoch_001 0.79/0.48), tICA eigenvalue 0.988 (implied 16 ns),
autocorrelation at fixed CV1 0.95, bimodality 0.88, split at a density minimum. psi(D3) loads
only 0.07 there. Counterfactually, after (CV1, the slowest CV1-residual mode = phi(E5) 0.76 +
psi(D3) 0.53, a production analogue of component 7) the hidden mode is psi(P4) 0.73 + psi(D3)
0.56. Per window (236), 102 are one-sided in epoch_002 (54 in epoch_001); a 25 % reseed picks 59
(34: 12 windows' assigned seed is already on the minority side, 8 have no admissible seed)
windows, 58 of 59 seeds from another state's end state (53 at another lambda), and raises the
windows whose assigned-or-X3 seed sits on their own minority side from 71 to 130 (91 to 125).
"Assigned" is the bank's state-aware assignment; without X3 seeding re-ranks the pool by CV
distance, so the structure actually grafted may differ. A seed is admissible only if it also
passes the seeding preflight (<= 1.2 window spacings per axis); without that check 28 of the 59
(14 of 42) first choices would not have been pulled. The tICA timescales (11-16 ns) are
indicative: lagged pairs within one replica trajectory cross lambda swaps.
