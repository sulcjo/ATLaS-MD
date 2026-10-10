# CVaux repairs, B/C/D scope — implementation plan

Status: in execution (2026-10-10). Branch `fix/cvaux-repair-bcd` off `fbe9500` (PR #141 head).

Spec: `docs/superpowers/specs/2026-10-10-cvaux-correctness-repair-spec.md` (F01–F13). **This plan overrides the spec's scope.** The user (2026-10-10) chose the parts categorised B (real scientific problem), C (coding issue) and D (bookkeeping accuracy). Parts categorised A (extreme security measure) and E (other) are OUT. When the spec and this plan disagree, this plan wins. Do not implement an OUT item because the spec calls it required.

## Scope table

| F | IN (B/C/D) | OUT (A/E) |
|---|---|---|
| F01 | Per-carrier burn-in exclusion; burn-in phases chosen from evidence (pull happened), incl. final after pool exhaustion and seed-mismatch fallbacks; one shared selection helper for both union paths; exclusion record | Whole-population exclusion; admission transaction IDs with crash-atomic equilibration record |
| F02 | Failed or missing-when-due spot check blocks pooling; post-admission phase run with the admitted model must have recorded z (no backfill fallback); backfill sha256 verified on read and kept current on re-admission; final_pdbs fallback needs step evidence | Proven XTC quantization error bound; reconstruction certificates; certified-subset/engineering analysis modes |
| F03 | Typed numeric validation of `aux_validation.json` (NaN/inf/bool/non-positive refused); per-check evidence required, not a bare "pass" string; same validator at admission and driver start | v2 certificate with full identity set; `aux_pending/` proposal workflow |
| F04 | logsumexp/expit overlap in `_forecast` | — |
| F05 | Absent/constant/nonfinite conditioning coordinates; no hard-coded 2 axes | — |
| F06 | Phase-local worker episodes; carrier instances not called independent lineages; unavailable diagnostics never PASS | — |
| F07 | Aux z always read from the aux force on each replica's Context, independent of the CV1/CV2 fast path; parity compares it with the position evaluator | GPU evidence run itself (controller follow-up, needs aurum2) |
| F08 | Pre-validate aux values before any `setParameter` in `set_window` | Pathological-input guards (1e200 centres, overflow-range checks) |
| F09 | Log-space Gibbs proposal + MH for aux runs; ledger writes authoritative log-q | Legacy gibbs-walk change (separate branch, user sign-off; Task 14) |
| F10 | Strict duplicate keys incl. phase; masked/fractional ids refused; float32-aware lambda comparison | float64 lambda writer schema |
| F11 | ΔU not labelled ΔH when volume missing (legacy analyzer; separate branch, Task 15) | Per-sample volume storage |
| F12 | Crosscheck labelled heuristic, never a statistical PASS; row-blocked `_target_logw` (no full N×K temporaries); `null_uninformative_trapped_lineages` status; repeated discovery attempts recorded | Paired block bootstrap |
| F13 | Pep-GaMD fixture stage lengths; parity assertion tolerance | CI jobs, constraints file, GPU/timestep/NPT evidence |

## Global constraints (bind every task)

1. **Opt-in.** With no aux flag, every output, manifest, kernel identity, Parquet schema and resume behaviour must be unchanged. Every change on a shared path (`production.py`, `windows.py`, `store.py`, union loaders, `offline.py`, `cli.py`) must be gated on aux being active and must be covered by a test that the flag-off path is unchanged. Existing off-path tests: `tests/test_aux_analyzer_unavailable.py::test_legacy_summary_verdict_and_bytes_unchanged`, `tests/test_aux_cv_state_table.py::test_legacy_table_has_no_aux_rows_and_does_not_import_the_aux_package`, `tests/test_aux_admission_registry.py::test_no_aux_csv_has_no_aux_columns`, `tests/test_aux_cv_cli.py::test_off_by_default`.
2. **TDD.** Write the regression first, show it fails on the unrepaired code for the intended reason, then fix the production function the regression exercises.
3. Never turn a required test into a skip, a failed check into a warning-only pass, or an invalid energy into zero.
4. Every MBAR solve goes through `gareus/adaptive/mbar_solve.py` / `gareus.mbar_analysis.solvers.solve_mbar`.
5. Run targeted tests only (`python -m pytest tests/<file> -q`); never the whole suite.
6. Commit messages: conventional commits, `fix(cvaux): ...` etc., ending with the attribution lines the controller gives.
7. Abbreviations: AP = `gareus/adaptive_production.py`, PR = `gareus/production.py`. Line numbers are as of `fbe9500` and will drift.

## Task 1: F13 test fixtures

Files: `tests/conftest.py`, `tests/test_aux_parity_report.py`, `gareus/cli.py` (only if needed).

- `tests/test_aux_parity_report.py:11` asserts `row["max_abs_dz"] == 0.0`. The baseline measured `1.1102230246251565e-16`. Replace the exact-zero check with the documented double-precision reduced-energy/z tolerance the parity report itself uses (read `gareus/auxiliary_cv/parity_report.py` for the constant; do not invent a looser one). Add a test where a physically significant disagreement (e.g. dz = 1e-3) gives `ok is False`.
- `tests/conftest.py:145-150` `_SMALL_CAMPAIGN_PEP_GAMD` uses `--gamd-cmd-steps 100 --equil-steps 100 --gamd-averaging-window 50`, but `gareus/cli.py:1889-1890` hard-codes `args.gamd_cmd_prep_steps = 5000` and `args.gamd_equil_prep_steps = 5000`, which conflict with 100-step stages under the upstream GaMD stage validation. Make the fixture valid: either make the two prep values derive from / be capped by the configured stage lengths ONLY when a new explicit CLI option is given (default behaviour byte-identical), or add explicit CLI options `--gamd-cmd-prep-steps` / `--gamd-equil-prep-steps` defaulting to 5000 and set them in the fixture. Keep the upstream stage validation enabled. The fixture's e2e test (`tests/test_aux_admission_e2e.py`, parametrised `pep_gamd` True/False, marked slow) must assert that the Pep-GaMD integrator was actually used (e.g. integrator class or recorded boost type in the phase manifest), not merely that the run did not crash.
- Acceptance: both parity tests pass; `python -m pytest tests/test_aux_admission_e2e.py -q -m slow` (or the marker the file uses) passes for both parametrisations; `tests/test_aux_cv_cli.py` passes.

## Task 2: F04 stable forecast overlap

File: `gareus/adaptive/aux_discovery/placement.py` (`_forecast`, ~line 19-28).

Current code: `w = np.exp(-(du - du.min()))`, `df = -np.log(np.mean(np.exp(-du)))`, `O = mean(1/(1+exp(clip(du - df, -50, 50))))`. With `du` ~ 800–1000 the `exp(-du)` underflows to 0, `df` becomes inf and O is wrong.

Replace with (float64):
- `lse = scipy.special.logsumexp(-du)`; `df = log(N) - lse`; `log_w = -du - lse`; `O = mean(scipy.special.expit(df - du))`.
- Validate: `du` 1-D, finite, non-empty; RT finite positive; z/c/k finite. Refuse nonfinite energy construction (raise a clear error naming the parent/candidate) — no clipping.
- Keep thresholds and the existing four-decimal gate/ranking policy unchanged. Keep `w` semantics for any other consumer (use `exp(log_w)` normalised).

Tests (`tests/test_aux_discovery_placement.py`):
- `du = [800] + [1000]*99` → O ≈ `0.009900990099` (rel 1e-9); assert this fails on the old code.
- Adding a finite common constant to `du` leaves O unchanged.
- Identical energies (du all equal) → O = 0.5 exactly within 1e-12.
- Extreme separation → O → 1/N, finite.
- Any NaN/inf in du → refused.

Controller follow-up after this task (not the implementer): re-run the real c10 epoch-1 replay and record before → after admission decisions.

## Task 3: F05 conditioning coordinate contract

Files: `gareus/adaptive/aux_discovery/{partitions,frames,pipeline,z3_search}.py`.

Facts: `pipeline.py:32` always stacks `np.c_[ft.cv1, ft.cv2]`; `partitions.standardise_s` (:44-48) has no NaN/zero-sd guard (NaN reaches sklearn KNN and errors); `partitions.s_bins` (:72-74) hard-codes 2 axes; `fit_partition` (:255-284) and `FrozenPartition.predict` (:213-217) repeat the 2-axis arithmetic; `partitions.py:271` and `z3_search.py:104` use `s_bins ** 2`; `frames.load_phase_samples` selects `cv2` explicitly (:74-75).

Requirements:
- Resolve declared conditioning dimensions from metadata (CV2 absent in a CV1-only campaign = supported 1-D case), not by treating arbitrary NaNs as absence. An unexpected NaN/inf in a declared active coordinate = invalid input with a specific status, not a crash.
- On training data only, classify each declared coordinate as usable / constant (scale-aware tolerance, e.g. sd <= 1e-12 * max(1, |mean|), document it) / invalid. Drop constant ones. No usable dimension → return status `insufficient_conditioning_evidence` (no crash, no invented dimension).
- Persist selected dimensions, training means/sds, constancy threshold and bin edges in the frozen partition (versioned field; legacy frozen partitions without it must still load as 2-D). Apply the identical transform to holdout/future data; never refit scales/edges on holdout.
- Replace `s_bins ** 2` and two-axis indexing with the actual conditioning-bin count (n_bins ** n_dims). Tied quantiles must not create phantom populated bins (use unique edges).
- z3 redundancy/correlation guards use the selected dimensions only, so a dropped coordinate does not trigger a later mandatory-correlation failure.
- `frames.load_phase_samples`: a schema without `cv2` (CV1-only campaign) must load with cv2 absent, not fail the query.

Tests (extend `tests/test_aux_discovery_partitions.py`, `tests/test_aux_discovery_pipeline.py`, `tests/test_aux_discovery_z3_search.py` or the existing equivalents): absent CV2 through the real `pipeline` discovery call (not an injected result); constant CV2; constant CV1 with useful CV2; tiny finite variance; one unexpected NaN; all descriptors constant; too few training rows; tied quantiles; train/holdout schema mismatch; assert no training statistic is recomputed from holdout. Existing 2-D results must be unchanged on the existing fixtures.

## Task 4: F03 validation-record contracts

Files: `gareus/adaptive/aux_discovery/validation.py` (:24-46), `gareus/adaptive/aux_discovery/settings.py` (`from_mapping` :70-79), `gareus/adaptive/aux_admission_io.py` (:210).

Facts: `check_validation_record` accepts NaN `timestep_fs` (`abs(nan - x) > 1e-9` is False), `true` (→ 1.0), and NaN/inf/0/negative/bool `k3_max_validated`; each check passes if its value equals the string `"pass"`.

Requirements:
- Typed validation: numbers must be real (`int`/`float`, not `bool`), finite; `timestep_fs > 0`; `k3_max_validated > 0` and finite; `temperature_k` if present > 0. Refuse unknown/missing `schema` values the module does not know (keep accepting the current schema id).
- Each of `finite_timestep`, `npt`, `cost` must be a mapping `{"status": "pass", "evidence": <non-empty string or non-empty list of strings>}`. A bare `"pass"` string is refused with reason `validation_evidence_missing:<name>`. (No v1 records exist in any campaign yet; c11 has no `aux_validation.json`.)
- `aux_settings.json` `from_mapping`: type and range checks for every numeric field (finite, booleans refused, positive where a count/size).
- The same validator runs at admission and at driver start (where `aux_discovery_incompatibilities` runs, AP:8853-8864) so a malformed record fails before MD, not at the first boundary.
- Update the runbook/docs that describe the `aux_validation.json` format (grep `k3_max_validated` in `docs/`).

Tests: NaN timestep, `true` timestep, infinite/zero/negative/bool k3, bare-pass check, empty evidence, unknown schema, malformed mapping, valid record passes.

## Task 5: F06 phase-local worker diagnostics

Files: `gareus/adaptive/aux_worker_table.py` (`worker_table` :104-121), `analyze_gareus_mbar.py` (~:5292), `gareus_report.py` (:235-248).

Facts: `worker_table` groups by replica only and sorts by phase-local step across all phases, interleaving chains; residence weights are all ones; `return_label_change_fraction` already groups by `(phase, replica)` lineage; analyzer calls `worker_table` without frames so return-label is always `frames_unavailable`.

Requirements:
- Group episodes by phase-local trajectory `(phase, replica)` (use the same `lineage = phase:replica` key as `frames.py:63-64`); sort by the phase's step clock; never create transitions across phase boundaries; preserve gaps.
- Report counts as `carrier_instances` (phase-local) and physical replica ids separately; never label either as independent lineages.
- Return-label diagnostics only when frames are supplied; otherwise status `unavailable` with a reason. An empty worker table or missing evidence must never grade PASS in `gareus_report.py` (grade NA/CAUTION with the reason).

Tests (`tests/test_aux_worker_table.py`): phase A `[ordinary, ordinary]`, phase B `[worker, worker]`, replica 0 in both, local steps `[1,2]` in both → 0 entries / 0 exits (assert old code gives 2 entries / 1 exit); real within-phase crossing counted; reused replica ids; gap preserved; empty table not PASS.

## Task 6: F10 observation identity and lambda comparison

Files: `gareus/auxiliary_cv/offline.py` (`refuse_duplicate_sample_keys` :192-213, `pool_aux_segments` :274, origins cast :292), `gareus/correctness/export.py` (lambda check :112-117).

Facts: `refuse_duplicate_sample_keys` returns early when `step`/`replica` are missing, drops masks with `getdata`, truncates fractional ids with `astype(int64)`, and has no phase key. It is only called from `pool_aux_segments` (aux pooling on the `load_parquet` path). The strict exporter compares `gamd_lambda` (stored `pa.float32()`) with exact equality against float64 frozen lambdas, so 0.1 refuses.

Requirements:
- Strict identity validation: required keys present (missing key = refusal on the aux path, not a silent return), unmasked, integer-valued (fractional → refuse; do not `astype(int)` first), within int range, non-negative. Duplicate key = `(phase/source id, replica, step)`; segment id is provenance, not identity. Two equal `(replica, step)` in different legitimate phases are not duplicates.
- Conflicting duplicates refuse (no silent dedupe); adding a duplicate must refuse, not change `N_k` from `[4,4]` to `[5,4]`.
- Share one identity validator between `pool_aux_segments` and `correctness/export.py` (`build_export_arrays` path).
- Lambda: compare stored values with the frozen lambda correctly rounded to the stored dtype (`np.float32(frozen) == stored` for float32 columns, exact for float64), then use the frozen float64 value. No loose global `isclose`. Must not conflate adjacent distinct float32-representable rungs.
- Flag-off: `load_parquet` on a non-aux run must not call any new code (assert in a test).

Tests (`tests/test_aux_export.py`, `tests/test_aux_mbar_duplicates.py`): identical and conflicting duplicates, missing/masked/fractional/overflow keys, equal steps in different phases, lambda 0 / 1 / 0.1 round trip via float32, genuinely wrong lambda refused, adjacent float32 rungs kept distinct, `N_k` unchanged-or-refused.

## Task 7: F08 aux parameter prevalidation

Files: `gareus/windows.py` (`set_window` :224-242), `gareus/auxiliary_cv/force.py` (`set_aux_parameters` :66-75).

Facts: `set_window` sets `r0`, `k`, `ss0`, `ss_k` first, then aux; an aux validation failure leaves the Context partly updated. `set_aux_parameters` validates k before centre, then sets.

Requirements:
- When the window has aux terms, validate every value that will be set (converted `k * 4.184` finite and >= 0, centre finite when k > 0; for k == 0 do not evaluate the centre) BEFORE the first `setParameter` call of `set_window`. On failure raise with the window/state provenance and leave the Context unchanged.
- Non-aux windows: identical call sequence to today (byte-identical; test with a recording fake Context that the sequence of `setParameter` calls is unchanged).

Tests (`tests/test_aux_cv_runtime.py` or a new `tests/test_aux_set_window_prevalidation.py`): fake Context records calls; invalid aux k (NaN, inf, negative) → no calls made; valid → same values as before; non-aux sequence unchanged.

## Task 8: F09 log-space Gibbs for aux runs

Files: PR `_gibbs_window_proposal_distribution` (:1142-1192), `_gibbs_mh_acceptance_probability` (:1195-1217), `GibbsProposal` (:1344-1358), `gibbs_propose_one_replica` (:1361-1422), call site (~:8992-9029); `gareus/auxiliary_cv/runtime_io.py` (`aux_event_fields` :58-61); aux ledger schema/replay (`gareus/auxiliary_cv/ledger.py` if present).

Facts: log weights are clipped to ±745, probabilities exponentiated; `qf <= 0 or qr <= 0 → 0.0`; ledger logs `log(q)` of rounded probabilities (-inf when q underflows). The same functions serve legacy gibbs-walk (c10). Aux-active is reachable at the call site via `_aux_io` / `args._aux_runtime`.

Requirements:
- New algorithm `gibbs_softmax_log_v2`, used ONLY when aux is active (aux runtime present). Legacy gibbs-walk keeps the current algorithm and RNG draw order byte-identically (a later separate branch fixes legacy; not here).
- v2: `log_q = -beta*delta - logsumexp(-beta*delta)` with no clipping; keep log-probabilities through proposal selection, the reverse proposal (evaluated against the hypothetical post-swap assignment, as today) and MH: `log_alpha = min(0, -beta*dU + log_q_rev - log_q_fwd)`. Exponentiate only for the sampler API. Draw one uniform u and accept iff `log(u) < log_alpha` (define u == 0 handling explicitly: accept). Validate finite reduced energies before proposals (nonfinite → raise with provenance; no candidate masking).
- `GibbsProposal` gains authoritative `log_q_forward`, `log_q_reverse`, `log_p_accept` (keep existing fields for legacy callers; audit positional construction).
- `aux_event_fields` writes those logs directly; the aux ledger records `proposal_algorithm` (`gibbs_softmax_log_v2`). Ledger readers accept records without the field as the legacy algorithm and do not claim a recoverable finite log-q from a historical `-inf`.

Tests (`tests/test_gibbs_walk.py`, `tests/test_aux_cv_exchange.py`, `tests/test_aux_ledger.py` or equivalents): F09 counterexample — beta = 1, `c = [sqrt(10), 0, 0.1]`, `z = [0, sqrt(10), -0.1]`, `u[s,r] = 50*(c[s]-z[r])**2`, identity assignment, carrier 0 proposing state 1 → acceptance ≈ `0.26894142137` (rel 1e-9) through the production function with aux active; finite logged reverse log-q even when its exponential underflows; exact small-permutation enumeration showing detailed balance for v2; legacy path (aux off) gives identical proposals/acceptances/RNG consumption as before on a fixed seed; old/new ledger record loading.

## Task 9: F07 force-side aux z observer

Files: `gareus/auxiliary_cv/runtime.py` (`observe_aux_z` :158-161, `make_aux_z_observer` :183-202), `gareus/auxiliary_cv/runtime_io.py` (`make_aux_record_observer` :117-140), `gareus/auxiliary_cv/parity_report.py` (:42-46), PR observer wiring (:7845-7873, :8189-8191, :8467-8472).

Facts: on the fast path (CustomCVForce CV1) z comes from `force.getCollectiveVariableValues(replica Context)` of the aux force; on the slow path (distance CV1, a CustomBondForce) z comes from the NumPy position evaluator and the aux force is never read, so runtime parity compares NumPy with NumPy. The parity report's "runtime" side is the stored `aux_z_00`.

Requirements:
- In aux runs, always obtain runtime z from the actual aux force (`ATLaSAuxCVUmbrella`) of each replica's own System/Context via `getCollectiveVariableValues`, regardless of `_use_fast_cv_path`. Resolve the force per replica System (do not assume the setup force object). Preserve full projection: `z = (offset + sum(cv values ... per model)) / scale` exactly as the model defines (cover pure-offset models).
- The independent comparison value comes from positions (`z_from_positions`) / stored torsions. If the force-side evaluator is unavailable, return an explicit unavailable/failure status — never substitute the NumPy evaluator on both sides.
- Record the evaluation source (`force` / `positions`) in the parity diagnostics.
- Keep existing precision-specific tolerances.
- Flag-off: no new Context calls on non-aux runs.

Tests (`tests/test_aux_runtime_io.py`, `tests/test_aux_cv_runtime.py`, `tests/test_aux_parity_report.py`): Reference-platform system with distance-primary CV1 (slow path) and contact-primary CV1 (fast path): perturb the force-side value only (e.g. monkeypatch/alter the aux force's offset parameter or the getCollectiveVariableValues result) while positions are unchanged → both paths detect the disagreement; unperturbed → parity ok; cloned Contexts each read their own force; inactive (k = 0) state still observed; pure-offset model.

GPU evidence on aurum2 is a controller follow-up (spec F07/T15), not part of this task.

## Task 10: F02 reconstruction governs pooling

Files: `gareus/kernel_identity.py` (`aux_admission_allows_pooling` :246-265), `gareus/adaptive/aux_pooling.py` (`phase_z` :94-129), `gareus/adaptive/aux_backfill.py` (`read_phase_backfill` :143-149, `write_phase_backfill` :138-140, final_pdbs fallback :118-127, `check_backfill_against_recorded` :174-230), `gareus/adaptive/aux_admission_io.py` (spot check :468-505, re-admission backfill :301-303, record :339-342), callers AP:3846-3848 and `gareus/mbar_analysis/loaders_union_parquet.py:327-328`.

Requirements:
- Spot check governs pooling: `aux_admission_allows_pooling` (one shared function, both union paths) refuses (`AuxPoolingRefused`, reason names the check) when the admission record's `spot_check` has `ok` false, or when the spot check was due (a phase with epoch > admission epoch exists on disk) but no spot_check record exists. `_post_admission_spot_check` must record a failure entry when it cannot run (exception → `{"ok": false, "error": ...}`), never leave nothing.
- Post-admission recorded z required: a phase whose window snapshots name the admitted model must carry recorded `aux_z` columns; if missing → refuse (integrity failure), no backfill fallback. Phases run without the admitted model (pre-admission, or before a re-admission's new model) legitimately use backfill.
- Backfill hash verified on read: `read_phase_backfill` (or `phase_z`) recomputes sha256 of `aux_z_backfill.parquet` and compares with the admission record's entry for that phase; mismatch or missing entry → refuse. Re-admission backfill (`backfill_all`) must update `aux_admission.json["backfill"]` atomically so the record never goes stale.
- final_pdbs fallback: use a final_pdbs frame only when evidence ties it to that sample: the sample step must be greater than the replica's max XTC step AND equal the phase's final production step recorded in that phase's checkpoint manifest (or equivalent durable record); otherwise refuse that phase's backfill with a reason (closes parked item 3).
- Engineering override flag is OUT of scope; refusal is the behaviour.

Tests (`tests/test_aux_backfill.py`, `tests/test_aux_union_pooling*.py`, `tests/test_aux_final_fix_wave.py` equivalents): `spot_check.ok=false, max_energy_err_kt=100` fixture refuses in both driver and analyzer union paths; due-but-missing spot check refuses; spot-check exception recorded as failure; missing recorded z post-admission refuses; modified backfill (byte change) refuses; stale admission backfill list after re-admission fixed; stale final_pdbs (step mismatch) refuses; legacy (no aux_admission.json) path unchanged.

## Task 11: F01 per-carrier burn-in exclusion

Files: `gareus/adaptive/aux_pooling.py` (`burnin_keep` :132-140, currently unused), AP driver loop (:3950-3966), `gareus/mbar_analysis/loaders_union_parquet.py` (:676-689), `gareus/adaptive/aux_admission_io.py` (`burnin_phase_epoch` :254, :309), PR seeding/continuation (:7304-7404, :6720-6813), `gareus/adaptive/aux_discovery/frames.py` (`phase_epoch` :34-37).

Facts: today the burn-in phase is always `admission epoch + 1`; both loops drop rows by sampled state only, so a replica that carried a pulled worker transient into an ordinary state keeps those rows; the final phase (`phase_epoch` None) is never filtered, so a worker first appearing in final after MD-pool exhaustion pools unfiltered and extensions continue its chain. Without `--ap-continue-states` workers are re-pulled every phase (each pull is a new transient); with it (c11) a worker is pulled in its first phase only, except on the `SeedMismatchError` fallback that pulls every window. Parquet rows carry `replica`, `window_id`, `step` (phase-local) — enough with the phase's window→state map; exchanges every 3000 steps vs samples every 2500 (c11), so any residence spans a sample.

Rule (binding):
- **Burn-in phases from evidence.** A phase is a burn-in phase for worker state w iff w was started in that phase by a US pull (not continued from an end state). Record this durably at phase setup: PR writes `<phase>/aux_seeding_record.json` listing, per aux worker state id, `"pulled"` or `"continued"` (and the source), written before production starts. For phases without the record (old data), treat every active worker as pulled (conservative).
- **Per-carrier exclusion inside a burn-in phase.** For each phase-local trajectory `(phase, replica)`, exclude every row from the first row at which that replica occupies any worker state pulled in that phase, to the end of the phase. Rows before that first visit are kept. Worker-state rows in that phase are therefore always excluded. Replicas that never visit such a worker keep all rows.
- Applies to numbered epochs, scheduled segments, final and extensions alike (an extension that falls back to pulling is a burn-in phase; a continued extension is not).
- **One implementation.** Replace both loops with one shared selection helper in `gareus/adaptive/aux_pooling.py` (`burnin_selection(...)` returning keep mask + exclusion record by phase, state, reason) called by both the driver union build and `loaders_union_parquet`. Remove `burnin_phase_epoch`-based filtering (keep the field for compatibility/reporting only). Delete or replace the unused `burnin_keep`.
- **Exclusion record.** Write per union build an exclusion summary (phase, state, replica count, rows excluded, reason `aux_burnin_carrier`) into the existing union diagnostics output / analyzer summary.
- A replica column missing in a phase that needs per-carrier exclusion → exclude that whole phase's rows (conservative) with reason `aux_burnin_no_replica`.

Tests: exchange trajectory where a pulled worker configuration moves into an ordinary state → both its worker rows and its later ordinary rows excluded, its earlier rows kept, other replicas kept; continued phase → nothing excluded; final phase as first worker phase → filtered; seed-mismatch fallback phase → filtered; old phase without record → conservative; driver and analyzer union paths select identical observation keys (aligned by key, not row order); flag-off unchanged. Exact-state counterexample (spec F01): ordinary two-state uniform target, aux state favouring one configuration, worker starting in the other; show a single MH-valid swap leaves the ordinary marginal off-equilibrium and that the rule excludes the transient rows.

## Task 12: F12 crosscheck honesty, memory, discovery labels

Files: `gareus/mbar_analysis/crosscheck.py` (`_target_logw` :252-260, `aux_ordinary_crosscheck` :279-343), `analyze_gareus_mbar.py` (:5258-5296), `gareus_report.py` (:215-232), `gareus/adaptive/aux_discovery/{z3_search,pipeline}.py`, discovery history in `gareus/adaptive/aux_admission_io.py`.

Requirements:
- Crosscheck statuses become `heuristic_pass`, `heuristic_fail`, `skipped`, `unavailable`, `error`; the payload states `method: "raw_count_heuristic"`. `gareus_report.py`: `heuristic_fail` → FAIL, `heuristic_pass` → CAUTION with text "heuristic agreement, no statistical test", others → CAUTION/NA; no status ever grades PASS; unknown status → CAUTION, never PASS. Integrity failures (pooling refusal) take precedence.
- Memory: `_target_logw` computes per row block (configurable block size, default e.g. 65,536 rows) without building full N×K `a`/`where` temporaries; `aux_ordinary_crosscheck` avoids extra full N×K copies where possible (ordinary subset may still be materialised once). Results identical (≤1e-12) to the dense version on small data.
- Null gate: when labels are constant within every lineage so shifts cannot discriminate, the discovery report records status `null_uninformative_trapped_lineages` (non-admission unchanged; do not weaken thresholds).
- Discovery history: each discovery attempt (boundary, epoch, outcome, holdout boundaries) appended to the discovery history so repeated attempts/holdout reuse are visible; reports must not claim campaign-wide false-discovery control.

Tests: `tests/test_aux_crosscheck*.py` (or new): heuristic labels, report grading table incl. unknown → CAUTION, blocked vs dense parity, trapped-lineage label, history appends across two boundaries.

## Task 13: parked bookkeeping items

- `aux_discovery_incompatibilities` (`gareus/adaptive/aux_discovery/settings.py:130-134`) at driver start reads the job's `--ap-aux-reserve-slots`; on a resume it must read the frozen `policy.aux_reserve_slots` (frozen key AP:656) when a frozen decision-settings record exists. Test: resume passing 0 with frozen 4 is not refused.
- `aux_admission_io.py:407-410` `rewritten["refused_workers"] = list(...) + dropped` — dedupe by worker identity so kill-replay does not append twice. Test: apply twice → one entry.

## Task 14 (separate branch `fix/legacy-gibbs-logspace` off `origin/main`, NOT merged without user sign-off): F09 for legacy gibbs-walk

Apply Task 8's log-space algorithm to the legacy path on its own branch from `origin/main`, with the same counterexample tests, an ordinary-energy-range regression showing identical acceptance decisions on a fixed seed for c10-like energies (document any last-ulp differences), and a note that c10 resumes would change exchange decisions only in the underflow regime. Do not push to main.

## Task 15 (same separate branch as Task 14): F11 ΔU label

`gareus/mbar_analysis/thermo.py` and report consumers: for NPT runs without aligned per-sample volume, label the energy difference `delta_U` (not `delta_H`), mark ΔH and the derived −TΔS as unavailable, and label NVT outputs as Helmholtz. Tests: labels for NVT, NPT-without-volume. No volume storage (OUT).
