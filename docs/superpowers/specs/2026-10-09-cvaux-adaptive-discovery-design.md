# ATLaS-MD: adaptive auxiliary-CV discovery and admission (CVaux in adaptive-production)

Status: design spec v1, 2026-10-09, awaiting user review. Branch `feat/cvaux-adaptive` off `feat/cvaux` @6f10688 (PR #141, Stages A-C).
Depends on: `2026-10-07-auxiliary-cv-gibbs-production-spec.md` (state semantics, Hamiltonian, Gibbs exchange, storage, resume; Stages A-C built) and the c10 manual pipeline in `docs/_local_docs/c10_aux_diagnosis/` (gitignored; scripts `build_features.py`, `diagnose.py`, `eval_partition.py`, `torsion_only_candidate.py`, `placement.py`).
Target campaign: chignolin_11 (c10 settings, auxiliary states discovered and admitted by the driver).

## 1. Purpose

chignolin_10's auxiliary-CV ("z3") workers were found by hand: a shadow diagnosis of c10's production frames, a post-hoc torsion discriminant, and an automatic forecast-benefit placement. This spec moves that whole pipeline into the adaptive-production driver. At a numbered-epoch boundary the driver tests its own data for structure that CV1/CV2 miss, fits a frozen torsion-linear z3 model, places at most four worker states, and admits them into the running campaign. From then on the workers exchange with every state (unrestricted Gibbs walk) and their samples enter the union MBAR.

No model, placement or state id from c10 is reused. c10's data serve only as the read-only replay acceptance test (Section 9).

## 2. User decisions (2026-10-09)

| # | Decision |
|---|---|
| U1 | Build adaptive admission now (option C), instead of the plain-run W/B pilot of gibbs-spec Stage D. This deviates from the gibbs spec's staging (adaptive entry only after a screen pass and an independent confirmation, gibbs spec Sections 14 and 16); this spec records that ruling. |
| U2 | Do it adaptively, the same way c10's was done manually: descriptors, hidden-structure test, z3 fit, placement, admission, all inside the driver on the campaign's own data. |
| U3 | No shams. The worker readout is descriptive; an aux effect cannot be separated from adding replicas. |
| U4 | Replicas come from a dedicated aux slice of the P1 reserve (default 4 slots) that R1/R3 cannot consume. `max_replicas` stays 236. |
| U5 | Attempt at every numbered-epoch boundary from the end of epoch_001 until the first admission; afterwards frozen for the rest of the campaign. |
| U6 | z3 chosen by a pairwise discriminant search, as c10: best passing candidate on held-out data; one model. Discovery labels (torsions + contacts) and evaluation labels (residualised contacts + H-bonds) are separate partitions. |
| U7 | Reuse c10's swarm: copy `swarm/round_000` into chignolin_11 and re-run the analysis there. |
| U8 | Worker samples are full MBAR states after one equilibration segment, with a hard ordinary-only vs all-states PMF crosscheck gate. |
| U9 | Workers are frozen and invisible to every other adaptive action (respring, R1-R3, retirement, ladder respace, rung replication, edge grading). May change later. |
| U10 | Launch once the code is done and GPU parity passes. Finite-timestep, NPT and cost validation run in parallel; admission requires a validation record. |
| U11 | Approach 1: in-driver boundary hook plus a pure discovery package, with a replay CLI. |
| U12 | (2026-10-09, after the c10 replay admitted nothing) z3 gate simplified: search candidates from EVERY discovery k that passes the reproducibility gates (not only the largest), and replace the fixed info-gain >= 0.10, stability >= 0.8 and basin-gain >= 0.05 thresholds by one rule: the best candidate's held-out info gain must exceed the best-of-candidates gain of every one of N_NULL (20) searches re-run on group labels scrambled by an independent circular shift within each lineage (train and holdout alike). The |corr(z, CV1/CV2)| <= 0.7 redundancy guard stays as an internal default. Effect size is left to placement (forecast benefit beyond its own null, confirmed on holdout). User-facing knobs: on/off, max workers, reserve slots. |
| U13 | (2026-10-09) ONE partition replaces the discovery/evaluation pair (supersedes U6): groups are fitted on CV-residualised contacts + H-bonds only, never torsions, so the torsion-only z3 must predict structure defined independently of its inputs. Reason: with torsions in the discovery features the z3 gate passed a noise control (torsions decoupled within lineage) by construction; with the single contacts+H-bond partition the noise control passed 0/10 seeds while real c10 data passed (best 0.084 vs null max 0.066). z3 candidates come from every passing+triggered k of this partition; placement uses its largest passing k. Trade-off: z3 training labels = placement labels (forecast optimistic; placement held-out confirmation still guards). Large-protein descriptor scaling: issue #143. |

## 3. Architecture

```
gareus/adaptive/aux_discovery/
    features.py      descriptor table from phase XTCs (pure given frames)
    partitions.py    discovery + evaluation GMMs, reproducibility, persistence
    z3_search.py     pairwise L1 torsion discriminants -> atlas-aux-cv-model-v1
    placement.py     forecast-benefit placement (port of c10 placement.py)
    settings.py      AuxDiscoverySettings, frozen record, override
    report.py        aux_discovery_report_v1 schema helpers
    __main__.py      replay CLI
gareus/adaptive/aux_admission_io.py   boundary hook, backfill, admission action builder
```

Each module is pure (arrays in, dataclasses out) except `features.py`'s frame reader and `aux_admission_io.py`. The hook never raises: any exception becomes report status `error` and nothing is admitted.

Phases already run as `window_mode: adaptive` with an explicit `windows_2d_csv` (`adaptive_production.py` segment, epoch, final and extension launches), which is the path the Stage B/C aux machinery accepts. Integration therefore means: write aux columns into every post-admission phase table, and inject `aux_cv_model` into phase args from the frozen record.

## 4. Boundary sequence and data flow

**Trigger.** `--ap-aux-discovery` (default off; field in `DECISION_SETTINGS_FIELDS`). Runs at the end of numbered epoch N, N >= 1, after the existing hooks (ladder respace, 3.3, respring, X3), only in non-recovered epochs and only while `aux_admission.json` does not exist. Never in the final phase or extensions.

**Data.** lambda = 0 frames only. Training = epochs 0..N-1, holdout = epoch N (first attempt: train epoch_000, confirm epoch_001). Frames from each phase's solute XTCs, joined to Parquet samples on (phase, replica, step) with the PR #133 sample-to-frame plan; state from the phase's own `epoch_window_map.csv`; stride = one exchange interval; resume overlaps removed. A frame budget (`max_frames`, frozen) caps memory by uniform striding within each (phase, state).

**Pipeline** (each step writes status and numbers; the first failing step ends the attempt, the driver retries at the next boundary):

1. `features`: 36 backbone torsion sin/cos (phi residues 2..n, psi 1..n-1; the aux runtime's angle convention), heavy-atom side-chain contacts and backbone H-bond switches (c10's families: 36 torsion, 28 heavy-atom contacts, 63 backbone H-bond switches for chignolin; schema recorded with sha).
2. `discovery` (c10 `prereg_v2.json` method, verbatim values): preprocess (drop features with training sd < 1e-3, scale by training sd floored at 0.05, equal total variance per family); conditional mean by kNN regression on standardised (CV1, CV2), k = 100, fitted on training; PCA of residuals to 80 % variance, at most 12 components; full-covariance GMM, k = largest in 2..8 with validation-refit ARI >= 0.5 AND median lineage-bootstrap ARI (10 resamples of (phase, replica) lineages) >= 0.5, a group counting only with >= 1 % of holdout frames. Trigger (prereg decision rule): hidden_fraction >= 0.5 with co-occurrence (a pair of groups each >= 10 % inside a 6 x 6 training-quantile (CV1, CV2) bin holding >= 2 % of holdout frames), or lineage_info >= 0.10 nats. No reproducible partition: `insufficient_evidence`; reproducible but no trigger: `keep`. Along-replica label persistence (1/10/50 frames) is reported, never a gate (exchange interval = frame stride, as c10).
3. `evaluation`: GMM on (CV1, CV2)-residualised contacts + H-bonds only, same reproducibility rule. Fail: `no_evaluation_partition`. Frozen into `aux_eval_partition.json` (means, sd, keep mask, PCA, GMM, feature order, schema sha; no dependence on any feature table).
4. `z3_search` (amended by U12, which supersedes the fixed thresholds below): for every pair of discovery groups that co-occur (rule above), fit an L1 logistic discriminant (liblinear, C on a frozen grid) on torsion sin/cos only, training frames. Candidate gates (prereg_v2 values), all on holdout frames: held-out information gain about the discovery groups beyond the s-bin >= 0.10 nats (multinomial logistic, z cubic); stability |corr| between refits on even vs odd training replicas >= 0.8; |corr(z, CV1)| <= 0.7 and |corr(z, CV2)| <= 0.7; independent-family check, held-out info gain about per-residue backbone basin codes beyond the s-bin >= 0.05 nats. Rank passing candidates by held-out info gain; ties to fewer non-zero coefficients. No candidate passes: `broaden` (prereg term; no admission). Native readouts (RMSD, helix, H-bond distances) are never computed by the hook. Output: `atlas-aux-cv-model-v1` with offset, `scale` = training-pooled sd of the projection, feature schema bound to the campaign's canonical topology digest.
5. `placement`: c10 `placement.py` rule, ported unchanged in logic, with the frozen evaluation partition as labels: candidates per lambda = 0 ordinary state at z3 training quantiles {5, 10, 90, 95} %, width fractions {0.35, 0.5, 0.7}, k3 = RT / sigma_w^2; forecast O, TV, circular-shift null, net, utility; lineage-bootstrap gates O_q05 >= 0.20, ESS >= 50 frames, effective lineages >= 20, top-3 lineage share <= 0.5; rank by utility q10; greedy, one worker per (parent, side); each pick confirmed on holdout (net > 0, O >= 0.15); at most `max_workers` (4). Candidates with k3 > `k3_max_validated` are refused (`k3_above_validated`). Zero confirmed: `no_worker`.
6. `gates`: validation record (Section 7) present and passing, else `validation_missing`; workers fit the aux reserve slice, else `aux_budget` (admit the top-ranked that fit).
7. Emit action `("admit_aux", parent_state_id, worker_params, reason, meta)` per worker; the P2 applier validates and adds the worker. Freeze `adaptive_production/aux_model.json` (+ sha), `aux_eval_partition.json`, `aux_admission.json`, and run the backfill (Section 5).

**Worker state.** Registry row with role `auxiliary`, `gamd_lambda` 0, the parent's CV1/CV2 centres and springs, `aux_center` c3, `aux_k_kcal_mol` k3, `aux_model_sha256`, `state_instance_id`, `spawn_parent_state_id`, `burnin_steps` = the first post-admission segment's steps. Never replicated onto other rungs.

**Next phase.** Window table carries `AUX_CSV_COLUMNS` and `INSTANCE_CSV_COLUMNS` for every row (ordinary rows: k = 0, role ordinary); phase args get `aux_cv_model`, `aux_phase_kind production`, `aux_equilibrium_eligible`. Ordinary states continue from chain ends (`ap_continue_states`); workers start from the parent's chain end and pull along z3 (new aux pull ramp in admission; a worker whose pull fails is dropped from the admission and recorded, the parent untouched).

**Reports.** `epoch_NNN/aux_discovery_report.json` (`aux_discovery_report_v1`) per attempt, with every step's status, numbers and the full placement candidate table; `aux_admission.json` on admission.

## 5. Pooling

**Backfill.** MBAR needs every sample's reduced energy in every worker state. Post-admission phases record the full torsion basis (`tor_###`) and `aux_z_##`. For every pre-admission phase the hook writes `<phase>/aux_z_backfill.parquet` keyed by (replica, step), with the model sha and a digest. Requirements: one trajectory frame per sample (`traj_interval == distance_output_interval`, enforced at parse), PR #133 alignment, completeness (no sample may lack z3), and a spot check: the backfill evaluator recomputed on post-admission frames equals the recorded `aux_z` within 1e-5.

**Union build.** Today it refuses aux snapshots and mixed kernel versions. New rule: a campaign with a frozen `aux_admission.json` may pool pre-admission legacy phases with post-admission aux phases if every legacy phase holds a complete backfill for that model sha. Every other aux combination is refused as now. Each state's bias row = CV1 + CV2 + boost + aux, the aux term added before any lambda = 0 early return. Worker samples inside `burnin_steps` are excluded. A missing z3 refuses with counts; it never evaluates as 0.

## 6. Diagnostics

- **Hard gate (new):** PMF from ordinary states only vs PMF from all states, along CV1, CV2 and z3, modelled on the lambda = 0 vs full-ladder crosscheck; over tolerance = FAIL in `gareus_report`.
- **Excluded and named:** `ladder_overlap`, CV2 resolution, edge metric, coverage, retirement list workers as "auxiliary, not graded".
- **Worker table (new, descriptive):** best ordinary partner by pairwise MBAR overlap (0.15 floor, CAUTION), occupancy, positive-residence entries/exits, carrier diversity, z3 distribution vs the placement forecast, local excursion-return readout (label change on return to an ordinary state, frozen evaluation partition). States plainly that without shams nothing is attributed to the bias.
- **Banner:** CAUTION cap stays until the gibbs-spec Stage D diagnostics audit.

## 7. Frozen settings, validation record, resume, refusals

**Frozen settings.** `adaptive_production/aux_settings.json` at the campaign's first job, mirrored into the run manifest (P8 pattern): the prereg_v2 values of steps 2 and 4 (preprocessing, kNN k, PCA cap, k range 2..8, ARI 0.5, group floor 1 %, hidden-fraction 0.5, co-occurrence 10 %/2 %, lineage info 0.10 nats, info gain 0.10, stability 0.8, |corr| 0.7, basin-code gain 0.05, C grid), placement quantiles, width fractions, gates, holdout floors, `max_workers` 4, `aux_reserve_slots` 4, frame stride, `max_frames`, RNG seed base. `--ap-aux-settings-override` replaces it wholesale. After admission, model, evaluation partition, worker rows and backfill shas are immutable for every later phase, the final phase and extensions.

**Validation record.** `adaptive_production/aux_validation.json` from the validation scripts (Section 9): code commit, per-check status (finite timestep, NPT controlled distribution, 236-context cost), `k3_max_validated`, `timestep_fs`. Missing, failing, or timestep mismatch: `validation_missing`, campaign continues without workers. Format (F03, `atlas-aux-validation-v1`, typed): `timestep_fs` and `k3_max_validated` finite real numbers > 0 (booleans, NaN, inf refused), optional `temperature_k` > 0, and each of `finite_timestep`, `npt`, `cost` a mapping `{"status": "pass", "evidence": <non-empty string or list of non-empty strings>}`; a bare `"pass"` string is refused (`validation_evidence_missing:<name>`). The writer CLI takes `--<check> pass|fail --<check>-evidence ...`. The same validator runs at driver start (a malformed record raises before MD) and at admission. `aux_settings.json` values are type- and range-checked on load.

**Reserve.** P1 `reserve_allowances` gains an aux slice taken before add_rung and resolution shares; R1/R3 never consume it; unspent aux slots stay idle until admission.

**Resume.** `admit_aux` goes through the applied-actions ledger (a kill between registry save and summary re-enters on the pre-action registry; never discovers or admits twice). An attempt interrupted before its report re-runs deterministically (seeds from the epoch index). Post-admission phases use the Stage C per-phase aux checkpoint binding. Top-level args never carry `aux_cv_model`; the driver injects it per phase from `aux_model.json`, so `refuse_resume_of_aux_campaign` stays per-phase correct.

**Parse-time refusals with `--ap-aux-discovery`:** no `--ap-continue-states` (repair fix wave I1, 2026-10-10: without it every phase re-pulls every worker, so the per-carrier burn-in drops every worker row and most carriers' ordinary rows; the value is frozen per campaign in `adaptive_production/aux_campaign_options.json`, not in `decision_settings.json`, and re-checked at driver start); window mode other than adaptive-production; non-`pep-gamd` boost; `exchange_mode: neighbor`; `--us-auto-drop-bad-windows`; `--ap-topups`; `traj_interval != distance_output_interval`; either not dividing `exchange_interval`. At first phase: force field with CMAP or virtual sites (existing physical-hash refusal), calibration-step alignment. Flag off: every path byte-identical (pinned).

**Topology.** The model binds the campaign's own canonical topology digest; it is fitted on the campaign's frames.

## 8. Worker invisibility (U9)

Every consumer that iterates states filters `role == auxiliary` out: respring candidates, 3.3 R1-R3 eligibility and geometry edges, P7a neighbour lists and chain edges, retirement, ladder respace (`centre_key` never groups a worker with its parent), rung replication of adds, top-up partners (top-ups refused anyway), coverage counts, convergence weak-edge counts. Each filter is tested; a central `is_auxiliary_state(state)` predicate is the only test used.

## 9. Testing and acceptance

**Unit (synthetic, CPU):** features vs mdtraj and the aux runtime's angle convention (sign, wrap); discovery/evaluation find a planted hidden mode at fixed (CV1, CV2) and return `insufficient_evidence` on a negative control; z3_search recovers a planted torsion direction and rejects a CV1-correlated one; a test asserts no native readout is ever loaded; placement port reproduces c10 `placement.json` exactly (all 888 candidates and the chosen four) from c10's frozen inputs; applier `admit_aux` (validation, reserve slice, k3 cap, role and instance columns, ledger resume); backfill (alignment incl. resumed XTC segments, completeness refusal, parity with recorded `aux_z`); union build with mixed phases equals direct energies, missing z3 refuses; crosscheck gate pass/fail; worker-invisibility filters; CLI refusals; off-path byte identity.

**CPU end-to-end:** small adaptive campaign on the GA-dipeptide fixture (Reference platform) with a test-only injection of a known z3 and placement: worker admitted at the boundary, pulled, next phase runs, union pools all phases, kill/resume across the admission epoch neither double-admits nor loses states.

**c10 replay (read-only, scratch copies):** the full hook on c10's real data at the epoch_001 boundary (train 000, confirm 001) and the epoch_002 boundary (train 000-001, confirm 002). Required: completes within a recorded frame/memory/time budget, deterministic on rerun, full report. The outcome is compared in a document with c10's manual result; matching the post-hoc hand choice is not required, a large divergence goes to the user before c11 launches.

**Before chignolin_11 launch:** scratch CODE_DIR on aurum2 (never c10's trees); GPU parity runbook passes; code review + board; validation scripts written (`scripts/cvaux_validation/`: finite timestep at 3.5 fs HMR vs a shorter-step reference up to `k3_max`, NPT controlled distribution, 236-context MPS cost) and run before the epoch_002 boundary, producing `aux_validation.json`.

## 10. chignolin_11 deliverables

- `RUNS/chignolin_11.yaml`: chignolin_10.yaml with out dir `chignolin_11`, `ap_aux_discovery: true`, `ap_aux_reserve_slots: 4`; traj and distance interval 300 (as c10); everything else identical.
- `RUNS/chignolin_11.sh`: c10 launcher with `PEPTIDE`, paths and `CODE_DIR` = the scratch tree.
- Out dir seeded with c10's `swarm/round_000` (copy, not link); analysis re-run in place.
- Report to the user at launch and at every admission attempt.

## 11. Out of scope

Shams and W/B attribution; more than one model per phase; re-evaluating, retiring or replacing workers; workers on lambda > 0; top-ups with aux; respring of k3; the legacy exception-crash double count (separate PR); CMAP/virtual-site support.

## 12. Risks

- Post-hoc origin: c10's z3 came from human choices among exploratory analyses; the automated rule may choose differently or admit nothing. The replay quantifies this before launch.
- Circularity: placement and the readout use the same evaluation partition (as c10). Holdout confirmation limits but does not remove optimism.
- No shams: any change in sampling after admission is confounded with adding four replicas.
- Backfill cost: every pre-admission frame is decoded once (c9 census: decoding dominates; budget measured in the replay).
- Deviation from gibbs-spec staging (U1): the workers enter a production campaign without a prior screen or confirmation.
