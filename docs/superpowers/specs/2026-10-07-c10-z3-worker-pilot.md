# ATLaS-MD: chignolin_10 z3 worker pilot (λ = 0 only, local metric)

Status: draft spec **v2**, awaiting review. Date: 2026-10-07.
v1 → v2: revised after an independent re-measurement (fresh-context agent), a Codex/ChatGPT review (REJECT 97) and the-board (3 of 4 judges APPROVE-WITH-CHANGES; glm failed). Synthesis: `docs/_local_docs/c10_aux_diagnosis/review_synthesis_2026-10-07.md`; v1 kept as `pilot_spec_v1.md` there.
Depends on: `2026-10-07-z3-auxiliary-umbrella-design.md` (z3 machinery, stages S1 + S2; on branch `docs/z3-auxiliary-umbrella-spec`, not yet on main).
Background: `2026-09-28-adaptive-auxiliary-cv-ladder-spec.md` (Sections 3, 10, 15, 16) and the c10 shadow diagnosis in `docs/_local_docs/c10_aux_diagnosis/` (gitignored).

## 1. Purpose

This is the first real-MD use of z3 workers. It asks one **local** question:

> When a replica leaves a parent state, is pushed along z3 in a worker, and returns, does it come back with a different committed structure (judged by contacts and H-bonds, not by z3) more often than after an identical excursion into a sham state without the z3 bias?

**Auxiliary CVs are local by design.** The longer-term plan is different auxiliary CVs in different (CV1, CV2) regions, at most 4 model slots per phase (z3 spec D4). This pilot tests one model in one region. Campaign-wide mixing is reported, never used to decide. With 4 workers among 232 states, a pooled rate cannot detect even a perfect local effect: the verifier's upper bound on the pooled ratio is about 1.03.

Scope: this is a **screening** trial. A pass authorises a confirmatory run (Section 5.5), never direct adoption into the live c10 campaign. Nothing here changes chignolin_10's live campaign.

## 2. Evidence that motivates it (c10 diagnosis, 2026-10-07)

All findings below except the first are post-hoc exploration on the same 1.27 µs (epoch_000, epoch_001, partial epoch_002). They motivate a hypothesis; they are not confirmatory evidence.

- **Pre-registered shadow decision:** `insufficient_evidence`. The 8-group partition does not reproduce: validation-refit ARI 0.375; true lineage-bootstrap median 0.30 (the original code dropped repeat draws and reported 0.36; fixed in `bootstrap_fix.py`).
- **Hidden structure (post-hoc):**
  - Coarse 2- and 3-group partitions on all descriptors reproduce.
  - A coarse 6 × 6 binning of (CV1, CV2) explains only 5-7 % of their predictive log-loss.
  - Labels persist along replicas: p_same 0.56 at 525 ps, against 0.15 if independent.
  - Initial-condition (lineage) dependence: 0.35 nats.
- **Evaluation partition (post-hoc, frozen):**
  - A GMM on (CV1, CV2)-residualised contact + H-bond descriptors: 91 features, no torsions.
  - k = 5 is the largest k with validation-refit ARI ≥ 0.5 and true-bootstrap median ≥ 0.5 (0.571 / 0.56).
  - The choice is robust across seeds (the verifier's seeds 100/200/300 also pick k = 5), but its margins are thin.
  - Hidden fraction 0.77; persistence p_same 0.59 at 525 ps vs 0.23 if independent.
- **The z3 candidate (post-hoc):** an L1 discriminant between two of the original 8 groups that co-occur at matched (CV1, CV2), refitted on backbone torsions only so `z3_model_v1` can express it.
  - On held-out epoch_002 at λ = 0: stability 0.91; |corr| with CV1 and CV2 ≤ 0.01; matched-(CV1, CV2) AUC 0.75 for helical turn, against a permutation null of 0.53 and CV1 alone at 0.59 with a fixed sign.
  - **No native signal.** The < 2 Å AUC is 0.56 with a fixed sign, at its null (0.55). v1's 0.67 came from choosing max(AUC, 1−AUC) per bin. +z3 points to the helical turn, not to native.
  - **Weak coupling to the evaluation partition:** 0.043 nats held-out, 2.7 % of the label entropy. The basin-code gain (0.56 nats) is **not** independent evidence, because basin codes are torsion-derived.
- **Native readout** (rmsd, helix, H-bond distances) was looked at after the shadow decision. It never selects parents or workers in v2 (Section 4.2).

## 3. User decisions

| # | Decision |
|---|---|
| P-D1 | Workers sit **only on the λ = 0 rung** (2026-10-07). This overrides z3-spec D3 (every rung) for this pilot. |
| P-D2 | Workers are harmonic z3 umbrellas (z3-spec D1); no flattening bias. |
| P-D3 | The z3 model is frozen from the c10 diagnosis (`z3_model_draft.json`, sha 8b1f5d5a…), ported to `z3_model_v1` at S1. It is never refitted. |
| P-D4 | **Auxiliary CVs are local** (2026-10-07). The primary metric is local to the parents; pooled numbers are secondary. Future: different auxiliary CVs per region. |
| P-D5 | **Placement is automatic** (2026-10-07): forecast-benefit placement (Section 4.2), never hand-picked states. The procedure is the prototype for the regional discovery stage. |

**Amendment proposed to the z3 spec, not made here:** the adaptive route (`--ap-z3-worker-requests`, S3) needs a per-request `rungs` field (`"all"` default, or a list of λ values). The plain-run route needs nothing, because the explicit window table decides placement.

## 4. Design

### 4.1 z3 model

- **Form:** z3(x) = K0 + Σ_j w_j f_j(x), with f the 36 sin/cos backbone torsion features (phi of residues 2-10, psi of residues 1-9). 35 coefficients are non-zero; K1 = K2 = 0.
- **Unit:** dimensionless. The draft divides by 3.181, the pooled sd over all c10 λ = 0 frames (validation included). The training-only sd is 3.264. The unit is part of the model definition and is frozen with it; it does not claim sd = 1 on future data.
- **Sign convention:** the coefficients are in IUPAC sign (mdtraj). If the z3 torsion schema stores negated angles (swarm convention), every sin coefficient flips sign and cos is unchanged.
- **Parity gate (release blocker):** port, then compare draft and ported z3 on ≥ 1,000 c10 frames spanning all phases and both parents. Max |Δz3| ≤ 1e-5 is required. A single-frame test is not enough.
- **Binding:** the model binds the torsion schema sha and the contact-map CV1 anchor identity (z3-spec Section 3.2).

### 4.2 Parents, workers and shams

**Placement is automatic** (P-D5; `placement.py`, output `placement.json`). No state is chosen by hand, and no named structural group or restraint pattern enters. The same code applies to any system, region or z3 model.

1. **Candidates.** Every λ = 0 state with ≥ 100 training frames (epoch_000 + epoch_001): 74 of 76; the 2 states added in epoch 2 are skipped. Per state, 12 candidate workers:
   - centre c3 at the 5, 10, 90 or 95 % quantile of that state's own z3;
   - width σ_w = 0.35, 0.5 or 0.7 × that state's z3 sd, so k3 = RT/σ_w^2.

   Everything scales with the parent, so nothing is tuned to chignolin. This gives 888 candidates.
2. **Forecast.** Reweight the parent's training frames by exp(-0.5 k3 (z3 - c3)^2 / RT) and compute three quantities.
   - **Overlap O:** pairwise MBAR, ∫pq/(p+q); identical states give 0.5.
   - **Shift TV:** the total-variation shift of the evaluation-label distribution.
   - **Null:** the mean TV after circularly shifting z3 against the labels along the parent's lineage-ordered series. The offset is at least the longest lineage, which keeps each series' own time dependence and breaks their alignment.
   - **Net benefit** = TV - null; **utility** = O × net, with O standing in for how often a replica enters the worker.
3. **Gates**, from a lineage bootstrap of 100 draws with the null recomputed inside each draw:
   - O 5th percentile ≥ 0.20;
   - weighted ESS ≥ 50 frames;
   - effective lineages ≥ 20;
   - top-3 lineage weight share ≤ 0.5.

   412 of 888 candidates pass.
4. **Ranking** by the bootstrap 10th percentile of the utility.
5. **Selection with held-out confirmation, against the winner's curse.**
   - Walk down the ranking, taking at most one worker per (parent, side of the parent's median z3).
   - Each pick must confirm on the parent's held-out epoch_002 frames with its frozen (c3, k3): net > 0 and O ≥ 0.15.
   - Stop at 4 confirmed workers; fewer than 4 is allowed.

**Result** (2026-10-07). Arm W: 4 workers, all at λ = 0, each on a different parent:

| Parent | (c1, k1, c2, k2) | Worker c3 / k3 (σ_w) | O (train / held-out) | Net TV (train / held-out) | Rank |
|---|---|---|---|---|---|
| 120 | (0.47, 9.67, -0.06, 2.09) | +1.49 / 1.82 (0.57) | 0.36 / 0.35 | 0.17 / 0.21 | 1 |
| 117 | (-0.20, 20.02, -0.42, 1.48) | -1.92 / 1.20 (0.71) | 0.27 / 0.29 | 0.21 / 0.07 | 3 |
| 9 | (0.13, 0, 1.02, 17.81) | +1.59 / 2.40 (0.50) | 0.24 / 0.25 | 0.22 / 0.07 | 4 |
| 198 | (-0.53, 9.83, 1.12, 21.30) | +1.29 / 2.16 (0.53) | 0.26 / 0.30 | 0.20 / 0.12 | 6 |

- Ranks 2 and 5 were second candidates on an already-used (parent, side) and were skipped.
- All four confirmed on held-out data; held-out null percentiles 0.90-1.00.
- **The ranking is shallow.** Utility lower bounds of the chosen four are 0.044-0.050, and states 147, 3 and 174 follow within 0.003. The method picks among near-equal candidates; the pilot tests the procedure, not uniquely best states.
- Three of the four workers push to high z3. Nothing in the procedure balances sides.
- Held-out net benefit is roughly a third of the training value for 117 and 9. This is the winner's curse the confirmation step exists for; both still pass.

v1's hand-picked parents (174, 81) and v2-draft's rule (`parent_selection.py`, which picked 15 and 81) are withdrawn: that rule was written after the parents had been picked by eye and was shaped to agree with them.

**Shams** (arm B): one sham per worker. A sham is an exact copy of its parent's Hamiltonian (same c1, k1, c2, k2, λ = 0, no z3) with the same exchange edges as its worker: parent ↔ sham only.
- Both arms then have 232 replicas, the same graph and the same exposure structure. This answers the parent spec's matched-replica requirement (aux spec §10).
- A parent ↔ sham swap always accepts, while parent ↔ worker swaps generally do not. Entry rates, residence times and occupancy are therefore reported per arm (M3) and can differ.
- The contrast estimates the effect of the z3 bias **relative to adding duplicate states**.
- **Implementation check:** the explicit window loader and neighbour graph must accept duplicate restraints and give each sham exactly the edges its worker gets. If a loader refuses duplicates, the pilot needs that allowance as a fourth dependency.

### 4.3 Route: forked plain runs

The pilot does not run inside the live c10 campaign, for two reasons:
- the z3 adaptive stage (S3) does not exist yet;
- c10's recorded epochs have no torsion columns and can never share a union with z3 states (z3-spec Section 9.3).

The pilot's unions therefore contain only pilot phases, all of which record torsions.

| Arm | Window table | Replicas |
|---|---|---|
| **B** (sham) | c10's current 228-state table + 4 shams | 232 |
| **W** (workers) | c10's current 228-state table + 4 workers (Section 4.2) | 232 |

**Identical in both arms:**
- force field, integrator, frozen Pep-GaMD envelope, exchange mode and interval, neighbour rule;
- `--record-torsions`, sample and trajectory cadence (300 steps);
- per-replica lengths and node-hours.

**Starting structures**
- Each original state continues from its own end state in the newest c10 phase that left one.
- The resolver is the chain-end resolver of `--ap-continue-states` (`_chain_end_pdb_by_window`):
  - the replica → window assignment comes from the segment's last checkpoint manifest, never a glob or a file's mtime;
  - window → state goes through that phase's `epoch_window_map.csv`;
  - the restraint is checked for every state.
- State of the local copy, 2026-10-07:
  - epoch_002 has 228 final PDBs, from the interrupted segment at step 470,400 of 778,289;
  - epoch_001 has 440 files for 222 windows, of which 218 are stale forks.
- The source phase and step of every start are recorded.
- Each worker and sham starts from its parent's end state. Workers then get a US pull with the k3 ramp (z3-spec Section 7.2) and need |z3_delta| ≤ 2 σ_w3. Shams need no pull.

**Seeds and pairing**
- 2 seed pairs: (W1, B1) share the starting structures and the integrator seed values; so do (W2, B2) with a second seed set.
- Seed pairing is the unit of the primary contrast (Section 5.2).
- A 3rd pair (+50 % cost) is recommended; it is mandatory for the confirmatory run.

**Length and cost**
- 3 ns per replica after a 0.2 ns discard: 3.2 ns × 232 replicas = 0.74 µs per run, 2.97 µs over 4 runs.
- At c10's measured ~3,600 ns/day (epoch_002: 375 ns pooled over 2.48 h of MD) that is about 4.9 h of MD per run and about 20 h total, plus setup and pulls.
- The z3 force overhead is unmeasured and is benchmarked by z3 S1.

### 4.4 What runs where

- A scratch copy on aurum2 under `RUNS/chignolin_10_z3pilot/{W1,B1,W2,B2}`, from the code commit containing z3 S1 + S2.
- The commit is recorded in each `run_manifest.json`.
- The live c10 campaign directory is read-only.

## 5. Pre-registered metrics and decision

Freeze this section, together with `eval_partition_model.pkl` and the analysis script, as `pilot_prereg.json` + hashes beside the runs before launch. Later changes need a new version and do not apply to runs already started.

### 5.1 Correctness gates (any failure = `invalid`)

1. **z3-spec tests 1-9** on the pilot's commit, plus the Pep-GaMD gradient-once test (z3 force outside FSF_T / FSF_D).
2. **Model parity:** the parity gate of Section 4.1.
3. **Offline/online check:** `z3_offline_check.json` passes for every W segment.
4. **Union MBAR builds** for every run with `--z3-union-scope fail`, with z3 row exclusions ≤ 0.1 %.
5. **λ = 0-only placement** (P-D1) does not break rung-dependent consumers. The following must pass or explicitly skip z3 states with a recorded reason:
   - `ladder_overlap` and ladder health;
   - the λ = 0 vs full-ladder PMF crosscheck, a hard FAIL in `gareus_report` (z3 states excluded from the λ = 0-only side, or included identically on both sides; stated in the report);
   - the restraint-width neighbour graph (worker and sham edges exactly as Section 4.2);
   - `other_rung_same_centre` and the top-up partners;
   - the exchange energies of all four states in a swap, compared with direct `getState` evaluations after `set_window`.
6. **Off-path identity:** B runs carry no z3 columns beyond the torsions, and their `EXCHANGE_ENERGY_VERSION` is `state_bias_matrix_v2`.

### 5.2 Primary metric L1: structural change on return (local)

**Labels**
- Every trajectory frame gets its evaluation-partition label from the frozen `eval_partition_model.pkl`, at a 10.5 ps stride (one exchange interval).
- The committed label at time t is the majority label over the 5 frames ending at t (pre-window) or starting at t (post-window), regardless of which state the replica occupies.
- Ties resolve to the earlier frame's label.

**Excursion**
- A replica (= configuration carrier) moves from parent P to its attached worker (W) or sham (B).
- It stays in that one worker or sham, possibly several exchange periods, then returns directly to P.
- Excursions that leave the worker or sham for anywhere other than P are not eligible; they are counted and reported per arm.

**Outcome**
- `changed` = committed pre-label (window ending at the departure frame) ≠ committed post-label (window starting at the return frame).
- Windows are fixed in time and follow the replica wherever it goes. A second excursion inside the post-window is allowed and is counted as exposure; there is no censoring.

**Contrast**
- p_W − p_B over all eligible excursions, standardised to frozen weights over the strata (parent × committed pre-label).
- The weights are each stratum's share of the pooled B excursions of the same seed pair. If a W stratum has no excursions, its stratum difference is set to 0 and the stratum is reported.
- Excursion length is **not** a stratum, because the bias changes it. Length-stratified differences are secondary (S3).

**Uncertainty**
- Within a run: a synchronized time-block bootstrap. Blocks are 0.5 ns of the whole ensemble, resampled jointly across all replicas; ≥ 3 × the measured label correlation time is required, else the block is enlarged and the change reported.
- Between runs: the 2 seed-pair contrasts.

**Pass (screening)**
- The pooled seed-pair contrast is ≥ +0.05 **and** its block-bootstrap 90 % lower bound is > 0;
- **and** both seed pairs have a positive point estimate.

**Power**
- Historical c10 return events (no workers) give a change probability after ≤ 4-period excursions of 0.13 / 0.16 / 0.28 / 0.24 for parents 120 / 117 / 9 / 198 (n = 39 / 43 / 29 / 42), with 147-174 return events per pooled µs per parent (`local_power.py` in the diagnosis directory, stride 10.5 ps). These are any-neighbour excursions, not sham excursions; the pilot's B arms measure the real baseline.
- The forecast is ~1 worker visit per pooled ns, about 750 per run.
- Exchange coupling and the eligibility filter make the effective count lower. The analysis script reports the achieved block-bootstrap half-width.
- If the half-width exceeds 0.05, the outcome is `insufficient_evidence`, never `reject_coordinate`.

### 5.3 Required transport and plumbing metrics

| ID | Metric | Rule |
|---|---|---|
| M2 | Pairwise MBAR overlap parent ↔ each worker (W runs) | ≥ 0.15 for all 4 edges |
| M3 | Exchange exposure per worker and sham: parent ↔ worker/sham transitions (per-pair counts; `gibbs-walk` acceptance is not a measure of overlap), mean residence, occupancy, distinct replicas | each worker: ≥ 200 parent ↔ worker transitions per run and ≥ 50 % of the parent's own distinct visitors |
| M4 | Bidirectional return transport | in the W runs, eligible excursions with `changed` must include both changes into and out of each of at least 2 evaluation groups. One-way drift alone fails. |

### 5.4 Secondary, reported only

- **S1:** pooled label-change rate W/B over the 228 original states, with an elapsed-time denominator (worker and sham time included). Expected ≈ 1; stated as such.
- **S2:** raw z3 side switches (z3 < -0.75 ↔ > +0.75) per replica-ns. They are mechanical under the bias and are never a pass criterion.
- **S3:** L1 by excursion length; committed changes per ns of parent + worker/sham exposure (rate per exposure, not only per excursion).
- **S4:** per-worker effects; forces on the z3 force; z3 support excursions.

### 5.5 Harm gates

**H1, CV1 PMF.**
- Union MBAR over all λ and states, W vs B, per seed pair.
- PMFs are aligned by their weighted mean over the common support (bins with ≥ 1 % weight in the pooled B arms).
- Equivalence margin: 0.3 kT RMS, decided with 90 % block-bootstrap intervals.

**H2, evaluation-group populations and z3 marginal.**
- Same procedure as H1.
- Margins: 0.03 absolute per group population; 0.3 kT RMS for the z3 PMF.
- These are reference-free observables. The helical-turn fraction from v1 is withdrawn as a gate, because it is native-set.

An interval that is neither inside nor outside the margin is `insufficient_evidence`.

### 5.6 Decision (precedence top to bottom)

| Result | Condition | Next step |
|---|---|---|
| `invalid` | any 5.1 failure | fix the machinery; nothing is interpreted |
| `harm_detected` | H1 or H2 outside margin | stop; investigate the force composition and parents |
| `insufficient_evidence` | L1 half-width > 0.05, or M3 below its floors, or an H1/H2 interval inconclusive | extend the runs or add a seed pair, under the same prereg |
| `retune` | M2 or M3 fails while L1 cannot be evaluated | rerun `placement.py` with the failed workers' parents excluded, or with the overlap gate raised; the model is unchanged |
| `reject_coordinate` | L1 or M4 fails with adequate precision and exposure | this z3 does not carry structure back; return to discovery (a regional model, or a contact-capable z3 schema) |
| `screen_pass` | L1, M2, M3 and M4 pass; H1 and H2 within margins | **confirmatory run:** 3 fresh seed pairs, starting structures from a different c10 phase, analysis frozen unchanged. Only a confirmatory pass leads to z3 S3 with the `rungs` amendment and adoption in the live campaign. |

**Cost of adoption:** a campaign union with z3 states must use `torsion-phases`, which excludes c10's 1.27 µs of torsion-less history (z3-spec Section 9.3). This is stated as an explicit cost of adopting.

### 5.7 Native readout after the decision only

Report the following per arm; they never feed back into 5.6:
- < 2 Å and < 1.5 Å fractions, and fold events;
- the D3N-T8O, D3O-G7N, misregister and helical-turn fractions;
- representative structures from the workers.

## 6. Deliverables

- `RUNS/chignolin_10_z3pilot/pilot_prereg.json`: Section 5 frozen, plus the sha256 of `z3_model.json`, `eval_partition_model.pkl` and `analyze_pilot.py`.
- `z3_model.json` (`z3_model_v1`, ported from the draft, parity-checked).
- `eval_partition_model.pkl`, frozen completely: training means, sd, sd floor, keep mask, family scaling, kNN training set, PCA, GMM, feature order and the feature-schema sha. It must no longer depend on the gitignored `features.parquet`.
- Explicit window tables for W (4 worker rows: `z3_model_sha256`, `z3_center`, `z3_k_kcal_mol`, `gamd_lambda` = 0) and B (4 sham rows).
- `docs/_local_docs/c10_z3_pilot/analyze_pilot.py`: excursion extraction, committed labels, standardised contrast, block bootstrap, M2-M4, H1-H2, decision.
- `placement.py` and `placement.json` (all 888 candidates, gates, ranking, held-out checks, selection), frozen and hashed in the prereg. The chosen four are applied unchanged.
- `pilot_report.json` + a short report, including L1 per worker, so the forecast utility can be checked against the measured effect (a first calibration of the placement forecast).

## 7. Dependencies and risks

**Four dependencies:**
1. z3 S1;
2. z3 S2;
3. per-state continuation from end states in a **plain** run: either a `--continue-from <phase dirs>` reusing the chain-end resolver, or each arm run as a one-phase adaptive campaign with `--ap-continue-states`;
4. duplicate-restraint sham states accepted by the explicit window loader and neighbour graph.

**Risks**
- **Weak coupling** of z3 to the evaluation partition (0.043 nats). `reject_coordinate` is a likely outcome; if so, it is informative for the regional plan, not a failure of the machinery.
- **Short stints:** parent residence is ~16 ps (exchange every 10.5 ps), so a worker acts on a configuration for a few exchange periods. Whether that moves a slow torsional mode is the core uncertainty, and L1 measures it.
- **Thin evaluation-partition margins** (ARI 0.56-0.57 at k = 5). Label noise biases L1 toward 0 in both arms, which reduces power but does not create a false positive.
- **Seed lock:** both arms start from the same c10 end states, so their difference is causal for the z3 bias, but the absolute rates inherit c10's lineage dependence (0.35 nats). The confirmatory run uses a different start phase for this reason.
- **Post-hoc origin:** the parents, workers, model and evaluation partition were chosen on the same data the diagnosis explored. The frozen prereg makes the pilot prospective; it does not make the historical evidence confirmatory.
- **Placement uses the outcome's labels.** The forecast uses the same evaluation partition that scores L1. That is legitimate for a prospective test, because the forecast uses old data and the verdict new data, but it makes the forecast optimistic. The held-out confirmation already shows this: net benefit fell to about a third of its training value for 2 of the 4 workers. Comparing forecast with measured L1 per worker is the first calibration of the placement method.
