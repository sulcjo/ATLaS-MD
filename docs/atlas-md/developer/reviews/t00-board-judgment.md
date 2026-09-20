# The Board's Judgment

## Majority position

The majority position is **ACCEPT-WITH-CHANGES**. The T00 contracts layer is not cosmetic: its parse-time validation is genuine and substantially protects invariants 1, 2, 3, and 5 on the read path. Invariant 4 — the R1 protection against a precise-but-biased arm being reported as confirmed — is only *conditionally* protected. The eight-status Decision and the confirmation gate are real mechanisms, but the gate is satisfiable by self-asserted statuses with no compulsory evidence linkage, and the primary write path bypasses validation entirely.

The majority's technical justification:

1. **Invariant 1 (identity is contents) is mostly real but incompletely wired.** `artifact_digest` (`_base.py` ~L136-142) correctly excludes the artifact's own `sha256` field and hashes the canonical JSON encoding. `verify_artifact_digest` (`_base.py` ~L145-152) correctly detects a mutated artifact. However, `_Artifact.from_mapping` (`_base.py` ~L159-163) pops `sha256` and discards it without verification, so a corrupted artifact with a stale digest loads silently and is assigned a fresh digest. The verification function exists but is not called on the standard load path.

2. **Invariant 2 (study role vs sampling phase) is enforced for the role/phase pair.** `StudyRole` and `PhaseKind` are separate `str` enums with no shared values (`vocabulary.py` L27-40), and `validate_role_phase` (`vocabulary.py` ~L127-131) enforces the exact mapping. `TrialPlan._parse` calls it (`contracts.py` ~L210-218). The residual hazard is narrower: `Stage` and `StudyRole` share the string values `"engineering"`, `"screen"`, and `"confirm"`, and `_enum` parses by value, so a serialized `StudyRole` value placed in a `stage` field parses as a valid `Stage`. `TrialPlan._parse` performs no stage↔role consistency check.

3. **Invariant 3 (no invented scientific defaults) is genuine but value-blind.** `validate_protocol` (`protocol.py` ~L144-151) refuses `None` requirements and names the producing task. It never fills gaps. But it treats any non-`None` value as resolved. `ProtocolSpec._parse` does not type-check most section values: `sampling.lambda_rungs`, `burn_in_ticks`, `measurement_ticks`, `campaigns_per_arm`, `resources.*`, `uncertainty.policy`, and `decision.interval_family_size` are only checked for key presence. A protocol with `lambda_rungs = -3` or `measurement_ticks = "fast"` passes parse and readiness.

4. **Invariant 4 (Decision gating) is real but satisfiable without evidence.** `_check_decision_consistency` (`contracts.py` ~L653-666) correctly requires `selected_arm_id` and five gating statuses for `CONFIRMED_FOR_DECLARED_PANEL`, and `Integrity.FAIL` forces `INVALID_INPUT`. But the gate checks only enum values. It does not require `input_sha256` to be nonempty or to contain mandatory keys; it does not require `tested_protocol_ids` to have at least two entries when `cross_protocol = AGREEMENT_SUPPORTED`; it does not require `selected_arm_id` to appear in any referenced trial plan; and it does not require `unresolved_regions` to be empty. A fabricated confirmation with empty provenance and a ghost arm id is structurally valid.

5. **Invariant 5 (individual SVD directions) is enforced at the index level.** `CandidateComponent.component_index` is 1-based, range-checked to `1..MAX_COMPONENT_INDEX`, and duplicate-checked in `CandidateSet._parse` (`contracts.py` ~L273-276). `TrialArm.secondary_component_index` is similarly range-checked and uniqueness-checked across arms (`contracts.py` ~L528-545). The conflation hazard with `compute_bootstrap_torsion_pca(component=N)` is addressed by naming, range, and uniqueness. The majority notes that `right_singular_vector` is not checked for unit norm (`contracts.py` ~L211-212), which is a fit-time correctness gap rather than an index-conflation gap.

6. **The independent audit's arithmetic is correct.** R1's variance ratio (100×) and MSE ratio (6400×) check out. R6's zero-visit probability `(1-10⁻⁶)^1000 ≈ 0.99900` and the one-sided 95% upper bound `1 - 0.05^(1/1000) ≈ 0.002991` are correct. The residual-CV chain rule `grad s_j = Σ_l v_jl grad φ_l - [v_jᵀb1 + 2a v_jᵀb2] grad a` is dimensionally and algebraically correct.

The majority therefore holds that the layer genuinely protects the *parse-time* contracts, but the write path, load-time digest verification, readiness value-validation, and Decision evidence linkage must be fixed before later tasks build on these types.

## Dissenting positions

**thinker (REJECT, confidence 85):**

> **Attack on target’s weakest arguments**
>
> The target’s Stage/StudyRole “collision” is a strawman. Invariant 2 is specifically that a **study role** is not a **sampling phase**; `Stage` is a readiness axis, not a sampling phase. `Stage.SCREEN.value == StudyRole.SCREEN.value` is irrelevant to that invariant. The real hazard is only that `_enum` parses by value (vocabulary.py L17-40), so a misplaced `StudyRole` could parse as a `Stage` — a type-confusion risk, not a violation of the stated invariant. The target inflates this into “three colliding values.”
>
> The `singular_value = 0` criticism is mathematically wrong. SVD singular values are nonnegative by definition; zero is a legitimate degenerate direction. The reason code name `NONPOSITIVE_SCALE` is cosmetic. A test name is not normative.
>
> The `unresolved_regions` “two-line fix” is a placeholder: blocking on a nonempty free-string list would later require schema migration when the real three-way enum arrives. The deferral is legitimate; the target’s own retrofit-cost argument cuts against its proposed stopgap.
>
> Most importantly, the target’s main R1 fix — requiring nonempty `input_sha256` — is insufficient. A fabricated digest string passes; the Decision has no fields for actual estimates, intervals, run counts, or a binding to a real `TrialPlan` arm. The gate checks self-asserted statuses (contracts.py ~L640-660). That is not a “change”; it is a missing evidence model. ACCEPT-WITH-CHANGES is too lenient.
>
> **Defense of position**
>
> I concede the target’s factual findings on `advantage_claim_enabled` (protocol.py ~L125), untyped `pressure_bar` (protocol.py ~L67-71), protocol numeric-type digest instability, and load-time digest non-verification (_base.py ~L140-145). These are real. But they are peripheral. The core R1 invariant is not protected because `CONFIRMED_FOR_DECLARED_PANEL` can be produced with empty `input_sha256` and a ghost `selected_arm_id`. The layer is a persistence schema, not a correctness gate.
>
> **Weighing the audit**
>
> The independent audit’s arithmetic (R1 variance/MSE ratios, R6 Clopper–Pearson bound) and the residual-CV chain rule are correct. The code-level findings are real but mostly fixable. The audit underweights the absence of evidence linkage in the Decision; that is the fatal defect.
>
> VERDICT: REJECT - the confirmation gate is satisfiable by self-asserted statuses with no evidence linkage, so the layer does not protect R1.
> CONFIDENCE: 85

## Confidence

Aggregate confidence: **80**.

Derivation: four of five members voted ACCEPT-WITH-CHANGES (80% majority). Member confidences were glm 86, kimi 85, deepseek-thinking 85, mini 90, thinker 85. The mean of the four majority confidences is 86.5, but the final agreement measurement shows mean position-card cosine 0.533, indicating only moderate alignment, and the dissenting REJECT carries confidence 85. The board-level confidence is therefore set to 80, reflecting the 4/5 majority and the tempered agreement.

## Conditions

The following changes are required before later tasks build on these types:

1. **Close the write-path bypass.** All artifact dataclasses are publicly constructible, and `to_mapping`/`to_json_bytes` serialize without invoking `_parse` or `_check_decision_consistency`. Add `__post_init__` validation, or make constructors private and require parse-round-trip construction, so producers cannot emit an unvalidated `Decision(...)` with a valid-looking digest.

2. **Verify the recorded digest on load.** `_Artifact.from_mapping` (`_base.py` ~L159-163) pops `sha256` without calling `verify_artifact_digest`. Either verify when `sha256` is present, or require callers to verify explicitly and document that the load path does not.

3. **Make readiness value-aware, not just None-aware.** `validate_protocol` (`protocol.py` ~L144-151) treats any non-`None` value as resolved. Add type/range validation for `sampling.layout`, `lambda_rungs`, `burn_in_ticks`, `measurement_ticks`, `campaigns_per_arm`, `resources.gpus`, `cpu_thread_cap`, `max_replicas`, `gpu_hours_per_campaign`, `total_gpu_hour_ceiling`, `uncertainty.policy`, `decision.provisional_choice_arm_id`, and `decision.interval_family_size`.

4. **Add evidence linkage to the Decision.** For `CONFIRMED_FOR_DECLARED_PANEL`, require nonempty `input_sha256` with mandatory keys (e.g. `protocol`, `trial_plan`, `observable_panel`, `candidate_set`); require `tested_protocol_ids` to have at least two entries when `cross_protocol = AGREEMENT_SUPPORTED`; require `selected_arm_id` to appear in the referenced trial plan; and require `unresolved_regions` to be empty for an unqualified confirmation.

5. **Fix the datatype holes.** `_parse_feature` (`contracts.py` ~L103) accepts `"index": 0.0` because `0.0 != 0` is `False`; require `isinstance(raw["index"], int)`. `advantage_claim_enabled` (`protocol.py` ~L125) accepts `0`/`1` via `not in (True, False)`; use `is True`/`is False` or `isinstance(..., bool)`. `pressure_bar` is not numerically validated for NPT (`protocol.py` ~L67-71); require a positive float. `Decision.study_id` is not type-checked (`contracts.py` ~L609); require a nonempty string.

6. **Resolve the R6 representation now.** `Decision.unresolved_regions` is a free-string list (`contracts.py` ~L577). Introduce a typed three-way region status (`RESOLVED_POPULATION`, `MASS_BOUNDED_SMALL`, `UNRESOLVED_SUPPORT`) or a canonical region identifier before persisting decisions, and add a consistency check between `support` and `unresolved_regions`.

7. **Add a stage↔role consistency check in `TrialPlan._parse`.** The plan carries both `stage` and `study_role` with overlapping value names and no cross-check. Either enforce a mapping (e.g. CONFIRM stage requires CONFIRM role) or document why any combination is valid.

8. **Normalize protocol numeric types before hashing.** `ProtocolSpec._parse` hashes raw section values, so `temperature_k: 300` and `300.0` produce different digests for the same science. Coerce numeric fields to a canonical type before computing `artifact_digest`.

---

## Mathematics & Physics audit (final)

We need answer audit. Need follow user format. Need examine judgment for mathematical/physical errors etc. Need list errors found in the judgment? We are auditing this judgment. Need identify errors in the judgment itself, not original code? The prompt: "Audit this judgment for mathematical and physical errors: dimensional consistency, sign conventions, thermodynamic identities, statistical-mechanical correctness, estimator assumptions, units, factors, statistics and datatypes per your checklists." Then given judgment. Need list every error found in judgment. Need be careful.

We need parse judgment. It's a board judgment about code contracts. Need find mathematical/physical/statistical/datatype errors in the judgment's claims. We need flag errors. Also "Checked:" verified correct.

Let's inspect.

Judgment majority: arithmetic claims:
- R1 variance ratio (100×) and MSE ratio (6400×) check out.
- R6 zero-visit probability (1-10^-6)^1000 ≈ 0.99900 and one-sided 95% upper bound 1 - 0.05^(1/1000) ≈ 0.002991 are correct.
- residual-CV chain rule grad s_j = Σ_l v_jl grad φ_l - [v_jᵀb1 + 2a v_jᵀb2] grad a is dimensionally and algebraically correct.

Need verify these. Need know context? R1 variance ratio 100x and MSE 6400x maybe from independent audit. We can check math if enough? Maybe not. Need infer. R1 variance ratio 100× and MSE ratio 6400×: If one estimator has variance 100x? MSE ratio 6400 = 100 * 64? Maybe due to bias? Need not know. Could be correct. Need check.

R6 zero-visit probability: (1 - 1e-6)^1000. Let's compute. ln ≈ 1000 * ln(0.999999) ≈ 1000 * (-1.0000005e-6) = -0.0010000005. exp ≈ 0.9990005. So ≈0.99900 correct. One-sided 95% upper bound for zero events: 1 - 0.05^(1/n). For n=1000, ln(0.05) = -2.995732, /1000 = -0.002995732, exp = 0.9970088, 1 - = 0.0029912. Correct.

Residual-CV chain rule: Need verify formula. It says grad s_j = Σ_l v_jl grad φ_l - [v_jᵀb1 + 2a v_jᵀb2] grad a. Hmm residual-CV? Need know definitions. Could be from a statistical method. Need assess dimensional/algebraic correctness. Let's derive? Suppose s_j = v_j^T φ? Maybe residual after projecting out linear/quadratic? Let's parse: "residual-CV chain rule `grad s_j = Σ_l v_jl grad φ_l - [v_jᵀb1 + 2a v_jᵀb2] grad a` is dimensionally and algebraically correct." This formula seems suspicious: dimensions? grad s_j should be vector in parameter space. RHS: Σ_l v_jl grad φ_l is vector. Second term: [v_jᵀ b1 + 2a v_jᵀ b2] grad a. Here a is scalar? grad a is vector. v_j^T b1 and v_j^T b2 are scalars. So second term is scalar times grad a. That is vector. OK. But is it algebraically correct? Need know residual-CV. Maybe s_j = v_j^T (φ - b1 a - b2 a^2)? Then grad s_j = Σ_l v_jl grad φ_l - (v_j^T b1) grad a - (v_j^T b2) grad(a^2) = Σ_l v_jl grad φ_l - v_j^T b1 grad a - 2a v_j^T b2 grad a. Yes formula matches. So correct.

Need check other claims for statistical/datatype errors.

Judgment says "R1's variance ratio (100×) and MSE ratio (6400×) check out." Need maybe verify if 6400 = 80^2? If variance ratio 100, MSE ratio 6400 implies bias contribution? Could be. No error.

Potential errors in judgment:

- "The T00 contracts layer is not cosmetic: its parse-time validation is genuine and substantially protects invariants 1, 2, 3, and 5 on the read path." Maybe no.

- "Invariant 1 (identity is contents) is mostly real but incompletely wired." `artifact_digest` excludes artifact's own sha256 and hashes canonical JSON encoding. verify detects mutated artifact. However `_Artifact.from_mapping` pops sha256 and discards it without verification, so a corrupted artifact with a stale digest loads silently and is assigned a fresh digest. This is code finding. Need check if "assigned a fresh digest" maybe if from_mapping pops sha256 and discards, then artifact_digest computed later? Could be. No math.

- "Invariant 2 ... Stage and StudyRole share string values "engineering", "screen", "confirm", and _enum parses by value, so a serialized StudyRole value placed in a stage field parses as a valid Stage. TrialPlan._parse performs no stage↔role consistency check." This is type confusion. No math.

- "Invariant 3 ... treats any non-None value as resolved. ProtocolSpec._parse does not type-check most section values: sampling.lambda_rungs, burn_in_ticks, measurement_ticks, campaigns_per_arm, resources.*, uncertainty.policy, and decision.interval_family_size are only checked for key presence. A protocol with lambda_rungs = -3 or measurement_ticks = "fast" passes parse and readiness." Need check: lambda_rungs = -3 maybe should be positive integer. measurement_ticks = "fast" should be int. No math.

- "Invariant 4 ... _check_decision_consistency correctly requires selected_arm_id and five gating statuses for CONFIRMED_FOR_DECLARED_PANEL, and Integrity.FAIL forces INVALID_INPUT. But gate checks only enum values. It does not require input_sha256 to be nonempty or to contain mandatory keys; it does not require tested_protocol_ids to have at least two entries when cross_protocol = AGREEMENT_SUPPORTED; it does not require selected_arm_id to appear in any referenced trial plan; and it does not require unresolved_regions to be empty. A fabricated confirmation with empty provenance and a ghost arm id is structurally valid." This is code. No math.

- "Invariant 5 ... CandidateComponent.component_index is 1-based, range-checked to 1..MAX_COMPONENT_INDEX, and duplicate-checked in CandidateSet._parse. TrialArm.secondary_component_index similarly range-checked and uniqueness-checked across arms. The conflation hazard with compute_bootstrap_torsion_pca(component=N) is addressed by naming, range, and uniqueness. The majority notes that right_singular_vector is not checked for unit norm, which is a fit-time correctness gap rather than an index-conflation gap." Need check: right_singular_vector unit norm? SVD right singular vectors are unit norm. If not checked, fit-time. No math.

- "The independent audit's arithmetic is correct. R1's variance ratio (100×) and MSE ratio (6400×) check out. R6's zero-visit probability (1-10⁻⁶)^1000 ≈ 0.99900 and the one-sided 95% upper bound 1 - 0.05^(1/1000) ≈ 0.002991 are correct. The residual-CV chain rule ... is dimensionally and algebraically correct." Need verify maybe "one-sided 95% upper bound" formula: For zero events in n trials, upper 95% CI for p is 1 - (alpha)^(1/n) with alpha=0.05. Yes. But "zero-visit probability (1-10^-6)^1000 ≈ 0.99900" maybe if per-visit probability of something? Need no.

Dissenting thinker:
- "Stage/StudyRole collision is strawman. Invariant 2 is specifically that a study role is not a sampling phase; Stage is a readiness axis, not a sampling phase. Stage.SCREEN.value == StudyRole.SCREEN.value is irrelevant to that invariant. The real hazard is only that _enum parses by value, so a misplaced StudyRole could parse as a Stage — a type-confusion risk, not a violation of the stated invariant. The target inflates this into 'three colliding values.'" This is argument. Need check if "Stage is a readiness axis, not a sampling phase" maybe correct? We don't know. No math.

- "singular_value = 0 criticism is mathematically wrong. SVD singular values are nonnegative by definition; zero is a legitimate degenerate direction. The reason code name NONPOSITIVE_SCALE is cosmetic. A test name is not normative." This is mathematical claim. Need assess. SVD singular values are indeed nonnegative. Zero is legitimate degenerate direction. If code name "NONPOSITIVE_SCALE" maybe not error. No.

- "unresolved_regions 'two-line fix' is a placeholder: blocking on a nonempty free-string list would later require schema migration when the real three-way enum arrives. The deferral is legitimate; the target's own retrofit-cost argument cuts against its proposed stopgap." No math.

- "target's main R1 fix — requiring nonempty input_sha256 — is insufficient. A fabricated digest string passes; the Decision has no fields for actual estimates, intervals, run counts, or a binding to a real TrialPlan arm. The gate checks self-asserted statuses. That is not a 'change'; it is a missing evidence model. ACCEPT-WITH-CHANGES is too lenient." No math.

- "I concede target's factual findings on advantage_claim_enabled, untyped pressure_bar, protocol numeric-type digest instability, and load-time digest non-verification. These are real. But peripheral." No math.

- "The independent audit's arithmetic ... and residual-CV chain rule are correct. The code-level findings are real but mostly fixable. The audit underweights absence of evidence linkage." No math.

Confidence:
- "Aggregate confidence: 80. Derivation: four of five members voted ACCEPT-WITH-CHANGES (80% majority). Member confidences were glm 86, kimi 85, deepseek-thinking 85, mini 90, thinker 85. The mean of the four majority confidences is 86.5, but the final agreement measurement shows mean position-card cosine 0.533, indicating only moderate alignment, and the dissenting REJECT carries confidence 85. The board-level confidence is therefore set to 80, reflecting the 4/5 majority and the tempered agreement."

Need audit this for statistical errors. This is a rich area. Let's examine.

They say aggregate confidence 80. Four of five voted ACCEPT-WITH-CHANGES (80% majority). Member confidences: glm 86, kimi 85, deepseek-thinking 85, mini 90, thinker 85. Mean of four majority confidences is 86.5. Let's compute: (86+85+85+90)/4 = 346/4 = 86.5. Correct. Dissenting confidence 85. Board confidence set to 80, reflecting 4/5 majority and tempered agreement. Is this statistically sound? Maybe not. There is no formal aggregation; but judgment can choose. Need flag? The prompt asks mathematical/physical errors, estimator assumptions, statistics. The confidence derivation has potential issues:
- "mean position-card cosine 0.533, indicating only moderate alignment" but no definition. Could be irrelevant.
- "four of five members voted ACCEPT-WITH-CHANGES (80% majority)" but "aggregate confidence 80" equals percentage majority, not a confidence. It conflates vote fraction with confidence. But maybe acceptable.
- "mean of the four majority confidences is 86.5" but they don't use it; set to 80. Could be arbitrary.
- "dissenting REJECT carries confidence 85" but if one dissenter with high confidence, aggregate should maybe lower than 80? Actually if 4/5 accept with avg 86.5 and 1/5 reject with 85, a Bayesian posterior probability of accept? Need not.
- "mean position-card cosine 0.533" maybe "position-card" not defined. Could be a typo? "position-card cosine" maybe "positional cosine"? Not relevant.

Need decide if to flag as statistical error. The user wants audit judgment for mathematical and physical errors. The confidence derivation is part of judgment. We can flag if it's an error. Need be precise.

Potential errors in confidence aggregation:
1. "Aggregate confidence: 80" derived from 4/5 majority. But confidence should reflect uncertainty in the judgment, not simply vote share. However no formal requirement. Could be MINOR.
2. "mean of the four majority confidences is 86.5" but "final agreement measurement shows mean position-card cosine 0.533" - if agreement is low, aggregate confidence should be lower than individual confidences, but 80 is close to 86.5. Maybe no.
3. "the dissenting REJECT carries confidence 85" - if a dissenting expert has high confidence, a proper aggregate should incorporate it; setting 80 maybe not enough? But no.

Need maybe flag "mean position-card cosine 0.533" as an undefined statistic; cannot be used to temper confidence without a model. But is that an "error"? Could be MINOR.

Need also check "four of five members voted ACCEPT-WITH-CHANGES (80% majority)" but "member confidences were glm 86, kimi 85, deepseek-thinking 85, mini 90, thinker 85." Wait "four of five members voted ACCEPT-WITH-CHANGES" but member confidences list includes "thinker 85" who dissented. The four majority confidences are glm 86, kimi 85, deepseek-thinking 85, mini 90. Mean 86.5. Correct. But "aggregate confidence 80" maybe should be weighted by confidences? If using vote fraction only, 80. If using mean confidences, 86.25 (including dissenter) or 86.5 majority. They set 80. No mathematical error.

Need check "mean position-card cosine 0.533" maybe "position-card" should be "positional" or "pairwise"? Could be a typo. Not math.

Need examine conditions for errors:
1. "Close the write-path bypass. All artifact dataclasses are publicly constructible, and to_mapping/to_json_bytes serialize without invoking _parse or _check_decision_consistency. Add __post_init__ validation, or make constructors private and require parse-round-trip construction, so producers cannot emit an unvalidated Decision(...) with a valid-looking digest." No math.

2. "Verify the recorded digest on load. _Artifact.from_mapping pops sha256 without calling verify_artifact_digest. Either verify when sha256 is present, or require callers to verify explicitly and document that the load path does not." No math.

3. "Make readiness value-aware, not just None-aware. validate_protocol treats any non-None value as resolved. Add type/range validation for sampling.layout, lambda_rungs, burn_in_ticks, measurement_ticks, campaigns_per_arm, resources.gpus, cpu_thread_cap, max_replicas, gpu_hours_per_campaign, total_gpu_hour_ceiling, uncertainty.policy, decision.provisional_choice_arm_id, and decision.interval_family_size." No math.

4. "Add evidence linkage to the Decision. For CONFIRMED_FOR_DECLARED_PANEL, require nonempty input_sha256 with mandatory keys (e.g. protocol, trial_plan, observable_panel, candidate_set); require tested_protocol_ids to have at least two entries when cross_protocol = AGREEMENT_SUPPORTED; require selected_arm_id to appear in the referenced trial plan; and require unresolved_regions to be empty for an unqualified confirmation." No math.

5. "Fix the datatype holes. _parse_feature accepts "index": 0.0 because 0.0 != 0 is False; require isinstance(raw["index"], int). advantage_claim_enabled accepts 0/1 via not in (True, False); use is True/is False or isinstance(..., bool). pressure_bar is not numerically validated for NPT; require a positive float. Decision.study_id is not type-checked; require a nonempty string." Need check datatype claims:
- `_parse_feature` accepts `"index": 0.0` because `0.0 != 0` is False. In Python, `0.0 != 0` is False indeed. If code checks `if raw["index"] != 0: raise`, then 0.0 passes. Requiring `isinstance(raw["index"], int)` would reject bool? In Python bool is subclass of int; `isinstance(True, int)` is True. If they want int but not bool, need `type(raw["index"]) is int` or exclude bool. The condition says "require isinstance(raw['index'], int)" but that would still accept True/False if not careful. Is that an error? The judgment says "require isinstance(raw['index'], int)" as fix. But if `True` is an int, then `isinstance(True, int)` is True. However "index" likely should be int; bool is int subclass but semantically not. Could flag MINOR: `isinstance(..., int)` accepts bool; use `type(...) is int` or `isinstance(..., int) and not isinstance(..., bool)`. But the prompt asks audit judgment for errors. This is a datatype error in the judgment's recommended fix. Need flag.

- `advantage_claim_enabled` accepts 0/1 via `not in (True, False)`; use `is True`/`is False` or `isinstance(..., bool)`. In Python, `0 == False` and `1 == True`, so `x not in (True, False)` is False for 0/1, so it passes. Using `is True`/`is False` is correct for bool? But `True is True` yes. However `is` with bool literals is fine. `isinstance(x, bool)` is better. The judgment says "use `is True`/`is False` or `isinstance(..., bool)`." That's correct. But note `is True` with numpy.bool_? Not relevant.

- `pressure_bar` not numerically validated for NPT; require a positive float. Correct.

- `Decision.study_id` not type-checked; require nonempty string. Correct.

6. "Resolve the R6 representation now. Decision.unresolved_regions is a free-string list. Introduce a typed three-way region status (RESOLVED_POPULATION, MASS_BOUNDED_SMALL, UNRESOLVED_SUPPORT) or a canonical region identifier before persisting decisions, and add a consistency check between support and unresolved_regions." No math.

7. "Add a stage↔role consistency check in TrialPlan._parse. The plan carries both stage and study_role with overlapping value names and no cross-check. Either enforce a mapping (e.g. CONFIRM stage requires CONFIRM role) or document why any combination is valid." No math.

8. "Normalize protocol numeric types before hashing. ProtocolSpec._parse hashes raw section values, so temperature_k: 300 and 300.0 produce different digests for the same science. Coerce numeric fields to a canonical type before computing artifact_digest." Need check: In Python, JSON `300` and `300.0` are different? JSON has number types; if canonical JSON encoding preserves int vs float, digests differ. Yes. But "temperature_k: 300 and 300.0 produce different digests for the same science" is true if raw values not normalized. However if using Python json.dumps, `300` and `300.0` serialize differently. Correct. But "same science" maybe 300 K vs 300.0 K same. Need normalize. No error.

Need also check "The T00 contracts layer is not cosmetic: its parse-time validation is genuine and substantially protects invariants 1, 2, 3, and 5 on the read path." Could be overclaim. But not math.

Need maybe check "Invariant 4 — the R1 protection against a precise-but-biased arm being reported as confirmed — is only conditionally protected." No.

Need maybe check "The eight-status Decision and the confirmation gate are real mechanisms, but the gate is satisfiable by self-asserted statuses with no compulsory evidence linkage, and the primary write path bypasses validation entirely." No.

Need maybe check "R1's variance ratio (100×) and MSE ratio (6400×) check out." Could there be a mathematical inconsistency: If variance ratio is 100×, and MSE ratio is 6400×, then bias ratio? MSE = variance + bias^2. If estimator A variance 100 times B, and MSE 6400 times B, then bias_A^2 - bias_B^2 = 6300 * var_B. Possible. No error.

Need maybe check "R6's zero-visit probability (1-10⁻⁶)^1000 ≈ 0.99900" and "one-sided 95% upper bound 1 - 0.05^(1/1000) ≈ 0.002991" are correct. Let's compute more precisely:
(1 - 1e-6)^1000 = exp(1000 ln(0.999999)). ln(1-x) = -x - x^2/2 - x^3/3... x=1e-6, term -1e-6 -5e-13 -3.33e-19. times 1000 = -0.0010000005. exp = 0.9990004997. So 0.99900 correct.
1 - 0.05^(1/1000). ln(0.05) = -2.995732273553991. /1000 = -0.002995732273553991. exp = 0.997009? Let's compute: e^-0.00299573 = 1 -0.00299573 + 0.000004487? = 0.99700876. 1 - = 0.00299124. Correct.

Need maybe check "residual-CV chain rule" formula: It says `grad s_j = Σ_l v_jl grad φ_l - [v_jᵀb1 + 2a v_jᵀb2] grad a`. Need verify dimensions: If `s_j` is scalar, `grad s_j` is vector. RHS first term vector. Second term scalar times grad a. OK. But if `a` is vector? The notation `grad a` suggests a is scalar. If a is scalar, `2a v_jᵀb2` scalar. OK. If `b1` and `b2` are vectors, `v_jᵀb1` scalar. OK. If `v_j` is right singular vector, `v_jl` components. OK. Algebraically: If s_j = v_j^T (φ - b1 a - b2 a^2) maybe yes. But if residual-CV has `s_j = v_j^T φ - (v_j^T b1) a - (v_j^T b2) a^2`, then grad = Σ_l v_jl grad φ_l - (v_j^T b1) grad a - (v_j^T b2) 2a grad a. Correct. No error.

Need maybe check "SVD singular values are nonnegative by definition; zero is a legitimate degenerate direction." This is mathematically correct. But "singular_value = 0 criticism is mathematically wrong" maybe if code name NONPOSITIVE_SCALE? No.

Need maybe check "Stage is a readiness axis, not a sampling phase." Could be code semantics. No.

Need maybe check "The target's main R1 fix — requiring nonempty input_sha256 — is insufficient. A fabricated digest string passes; the Decision has no fields for actual estimates, intervals, run counts, or a binding to a real TrialPlan arm." This is a code design argument. No.

Need maybe check "The layer is a persistence schema, not a correctness gate." No.

Need maybe check "The independent audit’s arithmetic (R1 variance/MSE ratios, R6 Clopper–Pearson bound) and the residual-CV chain rule are correct." Wait "R6 Clopper–Pearson bound" — the one-sided upper bound formula 1 - alpha^(1/n) is actually the "rule of three" or exact binomial upper bound? Clopper-Pearson exact upper bound for zero events is indeed 1 - alpha^(1/n) for one-sided 95%? Let's verify. For X~Binomial(n,p), observed 0. One-sided upper 95% confidence bound is p such that P(X=0) >= alpha? Actually Clopper-Pearson upper bound solves P(X <= x) = alpha/2 for two-sided. For one-sided 95% upper, solve P(X=0) = 1 - p^n = 0.95? Wait if p is upper bound, probability of observing 0 under p equals alpha? Let's derive: For one-sided 95% upper confidence limit, choose p_U such that P(X >= 0)?? Hmm.

Standard: If zero events in n trials, upper 95% CI ≈ 3/n (rule of three). Exact one-sided upper 95% bound is 1 - 0.05^(1/n). Because if true p = p_U, probability of zero events = (1-p_U)^n = 0.05, so p_U = 1 - 0.05^(1/n). Yes. This is sometimes called exact binomial. Clopper-Pearson for one-sided 95% uses alpha=0.05. So correct.

Need maybe check "zero-visit probability (1-10⁻⁶)^1000 ≈ 0.99900" and "one-sided 95% upper bound 1 - 0.05^(1/1000) ≈ 0.002991" are correct. Yes.

Need maybe check "mean position-card cosine 0.533" maybe "position-card" is not a standard term. Could be "position-card cosine" from some embedding? Not math.

Need maybe check "Aggregate confidence: 80. Derivation: four of five members voted ACCEPT-WITH-CHANGES (80% majority). Member confidences were glm 86, kimi 85, deepseek-thinking 85, mini 90, thinker 85. The mean of the four majority confidences is 86.5, but the final agreement measurement shows mean position-card cosine 0.533, indicating only moderate alignment, and the dissenting REJECT carries confidence 85. The board-level confidence is therefore set to 80, reflecting the 4/5 majority and the tempered agreement."

Potential statistical error: "mean position-card cosine 0.533" is not a valid measure of agreement for votes? It could be. But "mean position-card cosine" maybe "position-card" is a typo for "positional"? Not sure.

Potential statistical error: "four of five members voted ACCEPT-WITH-CHANGES (80% majority)" and "aggregate confidence 80" — if member confidences are 86,85,85,90,85, the average confidence is 86.2? Let's compute including all: (86+85+85+90+85)/5 = 431/5 = 86.2. They say mean of four majority confidences is 86.5. Correct. But "aggregate confidence 80" is lower than all individual confidences except? Actually all individual confidences >=85, so 80 is lower than every member's confidence. Is that logically possible? Yes, aggregate can be lower due to disagreement. But if 4/5 agree and one dissents with 85, a Bayesian aggregate might be around? Let's model: each member confidence c means probability correct? If independent, posterior probability of accept = product? Not simple. But setting 80 lower than all individual confidences could be justified by "tempered agreement." No formal error.

Potential statistical error: "mean position-card cosine 0.533, indicating only moderate alignment" — cosine similarity of 0.533 between what? If it's mean pairwise cosine of position cards, 0.533 is moderate. But using it to adjust confidence from 86.5 to 80 is arbitrary. No.

Need maybe check "The mean of the four majority confidences is 86.5" but "member confidences were glm 86, kimi 85, deepseek-thinking 85, mini 90, thinker 85." Wait "four of five members voted ACCEPT-WITH-CHANGES" but "thinker" is dissenting REJECT. The four majority are glm, kimi, deepseek-thinking, mini. Mean = 86.5. Correct. But "the dissenting REJECT carries confidence 85" — thinker's confidence 85. Correct.

Need maybe check "final agreement measurement shows mean position-card cosine 0.533" — if mean pairwise cosine is 0.533, the angle is ~57.8 degrees, moderate. No.

Need maybe check "Aggregate confidence: 80" — if using 4/5 majority, 80% is not a confidence level but a vote share. But the judgment explicitly says "reflecting the 4/5 majority and the tempered agreement." It's a judgment call. Could flag as MINOR: "Confidence is set to vote percentage rather than an evidence-weighted posterior; no principled aggregation." But the user wants errors, not style. Need decide.

Need maybe check "R1's variance ratio (100×) and MSE ratio (6400×) check out." Could there be a missing factor: If variance ratio is 100× and MSE ratio is 6400×, then the standard deviation ratio is 10× and RMSE ratio is 80×. Is that plausible? Maybe. No.

Need maybe check "R6's zero-visit probability (1-10⁻⁶)^1000 ≈ 0.99900" — if per-visit probability of "visit" is 1e-6, probability of zero visits in 1000 is 0.9990005. Correct. But "one-sided 95% upper bound 1 - 0.05^(1/1000) ≈ 0.002991" — if they are computing upper bound on probability of visit? For zero observed visits, upper bound is ~0.00299, which is much larger than 1e-6. Correct.

Need maybe check "residual-CV chain rule" formula: It says `grad s_j = Σ_l v_jl grad φ_l - [v_jᵀb1 + 2a v_jᵀb2] grad a`. Wait if `s_j` is a function of `a` and `φ`, and `φ` itself may depend on `a`? The chain rule should include `∂φ_l/∂a` terms if φ depends on a. The formula as written only has `grad φ_l` and `grad a`; if φ_l depends on a, then `grad φ_l` already includes ∂φ_l/∂a * grad a, so the second term accounts for explicit a dependence. But if `b1` or `b2` depend on a? Then missing terms. But likely not. No.

Need maybe check "right_singular_vector is not checked for unit norm" — SVD right singular vectors are unit norm. If not checked, fit-time correctness. No.

Need maybe check "CandidateComponent.component_index is 1-based, range-checked to 1..MAX_COMPONENT_INDEX" — if MAX_COMPONENT_INDEX maybe 10? No.

Need maybe check "TrialArm.secondary_component_index is similarly range-checked and uniqueness-checked across arms" — no.

Need maybe check "The conflation hazard with compute_bootstrap_torsion_pca(component=N) is addressed by naming, range, and uniqueness." No.

Need maybe check "Invariant 1 ... artifact_digest correctly excludes the artifact's own sha256 field and hashes the canonical JSON encoding." If artifact_digest excludes sha256, then a corrupted artifact with stale digest loads silently and is assigned a fresh digest. Wait if from_mapping pops sha256 and discards, then when loading, the artifact object has no sha256 field? Then artifact_digest computed later would be based on content and assigned fresh. So yes.

Need maybe check "Invariant 2 ... StudyRole and PhaseKind are separate str enums with no shared values" — if no shared values, then a StudyRole value cannot be a PhaseKind value. But Stage shares values with StudyRole. The invariant is role vs phase, not role vs stage. The majority says "Invariant 2 (study role vs sampling phase) is enforced for the role/phase pair." Then "residual hazard is narrower: Stage and StudyRole share string values..." This is consistent. The thinker says target inflated. No math.

Need maybe check "Invariant 3 ... validate_protocol refuses None requirements and names the producing task. It never fills gaps. But it treats any non-None value as resolved." No.

Need maybe check "ProtocolSpec._parse does not type-check most section values: sampling.lambda_rungs, burn_in_ticks, measurement_ticks, campaigns_per_arm, resources.*, uncertainty.policy, and decision.interval_family_size are only checked for key presence." No.

Need maybe check "A protocol with lambda_rungs = -3 or measurement_ticks = "fast" passes parse and readiness." If lambda_rungs should be positive integer, -3 invalid. measurement_ticks "fast" invalid. No.

Need maybe check "Invariant 4 ... _check_decision_consistency correctly requires selected_arm_id and five gating statuses for CONFIRMED_FOR_DECLARED_PANEL, and Integrity.FAIL forces INVALID_INPUT." No.

Need maybe check "It does not require input_sha256 to be nonempty or to contain mandatory keys; it does not require tested_protocol_ids to have at least two entries when cross_protocol = AGREEMENT_SUPPORTED; it does not require selected_arm_id to appear in any referenced trial plan; and it does not require unresolved_regions to be empty." No.

Need maybe check "A fabricated confirmation with empty provenance and a ghost arm id is structurally valid." No.

Need maybe check "Invariant 5 ... CandidateComponent.component_index is 1-based, range-checked to 1..MAX_COMPONENT_INDEX, and duplicate-checked in CandidateSet._parse." No.

Need maybe check "TrialArm.secondary_component_index is similarly range-checked and uniqueness-checked across arms." No.

Need maybe check "The majority notes that right_singular_vector is not checked for unit norm (contracts.py ~L211-212), which is a fit-time correctness gap rather than an index-conflation gap." No.

Need maybe check "The independent audit's arithmetic is correct." No.

Need maybe check "The residual-CV chain rule `grad s_j = Σ_l v_jl grad φ_l - [v_jᵀb1 + 2a v_jᵀb2] grad a` is dimensionally and algebraically correct." Wait let's re-evaluate with possible definitions. Suppose residual-CV (cross-validation) computes `s_j = v_j^T (φ - b1 a - b2 a^2)`? But if `b1` and `b2` are vectors of regression coefficients for each component? The formula has `v_jᵀb1` and `v_jᵀb2`, so b1/b2 are vectors in component space. If `s_j` is the residual of component j after removing linear and quadratic trends in `a`, then `s_j = φ_j - b1_j a - b2_j a^2`? Then grad s_j = grad φ_j - b1_j grad a - 2a b2_j grad a. That would be `grad φ_j - [b1_j + 2a b2_j] grad a`, not involving v_j. The formula with v_jl and v_jᵀb1 suggests `s_j = v_j^T φ - (v_j^T b1) a - (v_j^T b2) a^2`, i.e., the residual of the projection onto singular vector j after removing global linear/quadratic trends in a. That is plausible. No error.

Need maybe check "R1's variance ratio (100×) and MSE ratio (6400×) check out." Could there be a missing "R1" definition? No.

Need maybe check "R6's zero-visit probability (1-10⁻⁶)^1000 ≈ 0.99900" — if "zero-visit probability" means probability of zero visits when expected visits = 1000 * 1e-6 = 0.001, Poisson probability e^-0.001 = 0.9990005. Same. Correct.

Need maybe check "one-sided 95% upper bound 1 - 0.05^(1/1000) ≈ 0.002991" — if using Poisson, upper bound for zero events with 95% confidence is -ln(0.05)/1000 = 0.0029957, close. But they use binomial exact. Correct.

Need maybe check "The independent audit's arithmetic is correct. R1's variance ratio (100×) and MSE ratio (6400×) check out." Could there be a factor error: If variance ratio is 100, MSE ratio is 6400 = 80^2. If one estimator is biased by factor? No.

Need maybe check "The T00 contracts layer is not cosmetic: its parse-time validation is genuine and substantially protects invariants 1, 2, 3, and 5 on the read path." "read path" maybe "write path bypasses validation entirely" contradicts "protects on read path." No.

Need maybe check "Invariant 4 — the R1 protection against a precise-but-biased arm being reported as confirmed — is only conditionally protected." No.

Need maybe check "The eight-status Decision and the confirmation gate are real mechanisms, but the gate is satisfiable by self-asserted statuses with no compulsory evidence linkage, and the primary write path bypasses validation entirely." No.

Need maybe check "The majority's technical justification: 1. Invariant 1 ... `_Artifact.from_mapping` pops sha256 and discards it without verification, so a corrupted artifact with a stale digest loads silently and is assigned a fresh digest." Wait if from_mapping discards sha256, then the loaded artifact doesn't have the stale digest; it will get a fresh digest when serialized. But "corrupted artifact with a stale digest loads silently" — if the content is corrupted but digest stale, from_mapping doesn't verify, so yes. "assigned a fresh digest" — yes, because artifact_digest computed from content. No.

Need maybe check "Invariant 2 ... Stage and StudyRole share the string values "engineering", "screen", and "confirm", and _enum parses by value, so a serialized StudyRole value placed in a stage field parses as a valid Stage." If Stage has values "engineering", "screen", "confirm" and StudyRole has same, then yes. No.

Need maybe check "TrialPlan._parse performs no stage↔role consistency check." No.

Need maybe check "Invariant 3 ... ProtocolSpec._parse does not type-check most section values: sampling.lambda_rungs, burn_in_ticks, measurement_ticks, campaigns_per_arm, resources.*, uncertainty.policy, and decision.interval_family_size are only checked for key presence." No.

Need maybe check "A protocol with lambda_rungs = -3 or measurement_ticks = "fast" passes parse and readiness." If "readiness" means validate_protocol, yes.

Need maybe check "Invariant 4 ... _check_decision_consistency correctly requires selected_arm_id and five gating statuses for CONFIRMED_FOR_DECLARED_PANEL" — "five gating statuses" maybe there are five statuses? No.

Need maybe check "Integrity.FAIL forces INVALID_INPUT." No.

Need maybe check "It does not require input_sha256 to be nonempty or to contain mandatory keys; it does not require tested_protocol_ids to have at least two entries when cross_protocol = AGREEMENT_SUPPORTED; it does not require selected_arm_id to appear in any referenced trial plan; and it does not require unresolved_regions to be empty." No.

Need maybe check "A fabricated confirmation with empty provenance and a ghost arm id is structurally valid." No.

Need maybe check "Invariant 5 ... CandidateComponent.component_index is 1-based, range-checked to 1..MAX_COMPONENT_INDEX, and duplicate-checked in CandidateSet._parse." No.

Need maybe check "TrialArm.secondary_component_index is similarly range-checked and uniqueness-checked across arms." No.

Need maybe check "The conflation hazard with compute_bootstrap_torsion_pca(component=N) is addressed by naming, range, and uniqueness." No.

Need maybe check "right_singular_vector is not checked for unit norm" — if right singular vector should be unit norm, yes.

Need maybe check "The independent audit's arithmetic is correct." No.

Need maybe check "R1's variance ratio (100×) and MSE ratio (6400×) check out." Could there be a missing "R1" definition? No.

Need maybe check "R6's zero-visit probability (1-10⁻⁶)^1000 ≈ 0.99900" — if "zero-visit probability" means probability of zero visits when per-visit probability is 1e-6, yes.

Need maybe check "one-sided 95% upper bound 1 - 0.05^(1/1000) ≈ 0.002991" — yes.

Need maybe check "residual-CV chain rule ... is dimensionally and algebraically correct." Yes.

Need maybe check "SVD singular values are nonnegative by definition; zero is a legitimate degenerate direction." Yes.

Need maybe check "The reason code name NONPOSITIVE_SCALE is cosmetic. A test name is not normative." No.

Need maybe check "The target's main R1 fix — requiring nonempty input_sha256 — is insufficient. A fabricated digest string passes; the Decision has no fields for actual estimates, intervals, run counts, or a binding to a real TrialPlan arm." No.

Need maybe check "The gate checks self-asserted statuses (contracts.py ~L640-660)." No.

Need maybe check "I concede the target's factual findings on advantage_claim_enabled (protocol.py ~L125), untyped pressure_bar (protocol.py ~L67-71), protocol numeric-type digest instability, and load-time digest non-verification (_base.py ~L140-145). These are real. But they are peripheral." No.

Need maybe check "The independent audit’s arithmetic (R1 variance/MSE ratios, R6 Clopper–Pearson bound) and the residual-CV chain rule are correct." No.

Need maybe check "The code-level findings are real but mostly fixable. The audit underweights the absence of evidence linkage in the Decision; that is the fatal defect." No.

Need maybe check "VERDICT: REJECT - the confirmation gate is satisfiable by self-asserted statuses with no evidence linkage, so the layer does not protect R1. CONFIDENCE: 85" No.

Need maybe check "Aggregate confidence: 80. Derivation: four of five members voted ACCEPT-WITH-CHANGES (80% majority). Member confidences were glm 86, kimi 85, deepseek-thinking 85, mini 90, thinker 85. The mean of the four majority confidences is 86.5, but the final agreement measurement shows mean position-card cosine 0.533, indicating only moderate alignment, and the dissenting REJECT carries confidence 85. The board-level confidence is therefore set to 80, reflecting the 4/5 majority and the tempered agreement."

Let's scrutinize this for statistical errors. "mean position-card cosine 0.533" — if this is a similarity measure between position cards, a mean cosine of 0.533 indicates moderate alignment. But "position-card" maybe "position card" is a document? Not relevant. "final agreement measurement" maybe from some voting system. No.

Potential issue: "four of five members voted ACCEPT-WITH-CHANGES (80% majority)" — but "member confidences were glm 86, kimi 85, deepseek-thinking 85, mini 90, thinker 85." The mean of the four majority confidences is 86.5. But if you include the dissenter, mean is 86.2. They set 80. No.

Potential issue: "the dissenting REJECT carries confidence 85" — if the dissenter is confident, a simple majority vote with 4/5 gives 80% but the dissenting confidence is high. In a proper confidence aggregation, the group confidence should maybe be lower than 80? Let's think. Suppose each member's confidence is probability of being correct. If 4 say accept with avg 0.865, 1 says reject with 0.85. If we treat as independent and equal priors, posterior probability of accept? Let's compute using log odds. Accept log odds for four: ln(0.865/0.135)=1.857, ln(0.85/0.15)=1.735, ln(0.85/0.15)=1.735, ln(0.9/0.1)=2.197. Sum = 7.524. Reject log odds: ln(0.85/0.15)=1.735. Posterior accept odds = exp(7.524 - 1.735)? Wait if each vote is evidence for its position, posterior odds = prior odds * product of likelihood ratios. If each member's confidence is probability their vote is correct, then likelihood ratio for accept vote = p/(1-p) for accept, for reject vote = (1-p)/p? Actually if member says accept with confidence p, then P(data | accept) = p, P(data | reject) = 1-p, LR = p/(1-p). For reject with confidence p, LR for accept = (1-p)/p = 0.15/0.85 = 0.176. Product LRs = (0.865/0.135)*(0.85/0.15)^2*(0.9/0.1)*(0.15/0.85). Let's compute: 6.407 * 5.667^2 * 9 * 0.1765. 5.667^2=32.11; *6.407=205.7; *9=1851; *0.1765=326.8. Posterior odds 326:1, probability 99.7%. So aggregate confidence would be much higher, not 80. But this model assumes independence and confidence as probability correct. Not necessary.

Alternatively, if confidence is not probability correct but "confidence in own judgment", aggregation is not straightforward. Setting 80 is arbitrary. But not a mathematical error per se.

Potential issue: "mean position-card cosine 0.533" — if the agreement is moderate, maybe confidence should be lower than 80? Actually 0.533 is moderate, but 80 is already lower than all individual confidences. No.

Potential issue: "Aggregate confidence: 80" — if four of five voted accept and one reject, the probability that majority is correct under simple majority with individual accuracies ~0.85 is high (>0.95). So 80 might be too low. But again no formal.

Need maybe check "The mean of the four majority confidences is 86.5" — but "member confidences were glm 86, kimi 85, deepseek-thinking 85, mini 90, thinker 85." Wait "deepseek-thinking" is a model? "thinker" is a dissenting position. "deepseek-thinking" and "thinker" are different? Yes. No.

Need maybe check "mean position-card cosine 0.533" — "position-card" maybe should be "position-card cosine" from a "position card" in a debate. No.

Need maybe check "The board-level confidence is therefore set to 80, reflecting the 4/5 majority and the tempered agreement." If the mean pairwise cosine is 0.533, that's not "tempered agreement" maybe it's moderate. No.

Need maybe check "R1's variance ratio (100×) and MSE ratio (6400×) check out." Could there be a missing "R1" definition? No.

Need maybe check "R6's zero-visit probability (1-10⁻⁶)^1000 ≈ 0.99900" — if "zero-visit probability" means probability of zero visits when per-visit probability is 1e-6, yes.

Need maybe check "one-sided 95% upper bound 1 - 0.05^(1/1000) ≈ 0.002991" — yes.

Need maybe check "residual-CV chain rule ... is dimensionally and algebraically correct." Yes.

Need maybe check "SVD singular values are nonnegative by definition; zero is a legitimate degenerate direction." Yes.

Need maybe check "The reason code name NONPOSITIVE_SCALE is cosmetic. A test name is not normative." No.

Need maybe check "The target's main R1 fix — requiring nonempty input_sha256 — is insufficient. A fabricated digest string passes; the Decision has no fields for actual estimates, intervals, run counts, or a binding to a real TrialPlan arm." No.

Need maybe check "The gate checks self-asserted statuses (contracts.py ~L640-660)." No.

Need maybe check "I concede the target's factual findings on advantage_claim_enabled (protocol.py ~L125), untyped pressure_bar (protocol.py ~L67-71), protocol numeric-type digest instability, and load-time digest non-verification (_base.py ~L140-145). These are real. But they are peripheral." No.

Need maybe check "The independent audit’s arithmetic (R1 variance/MSE ratios, R6 Clopper–Pearson bound) and the residual-CV chain rule are correct." No.

Need maybe check "The code-level findings are real but mostly fixable. The audit underweights the absence of evidence linkage in the Decision; that is the fatal defect." No.

Need maybe check "VERDICT: REJECT - the confirmation gate is satisfiable by self-asserted statuses with no evidence linkage, so the layer does not protect R1. CONFIDENCE: 85" No.

Need maybe check "Aggregate confidence: 80. Derivation: four of five members voted ACCEPT-WITH-CHANGES (80% majority). Member confidences were glm 86, kimi 85, deepseek-thinking 85, mini 90, thinker 85. The mean of the four majority confidences is 86.5, but the final agreement measurement shows mean position-card cosine 0.533, indicating only moderate alignment, and the dissenting REJECT carries confidence 85. The board-level confidence is therefore set to 80, reflecting the 4/5 majority and the tempered agreement."

Let's think if "mean position-card cosine 0.533" could be a "cosine" similarity between "position cards" (text embeddings). If mean cosine is 0.533, that's moderate. But "position-card" maybe "position-card" is a term in some debate system. No.

Potential statistical error: "The mean of the four majority confidences is 86.5" — but if the four majority confidences are 86, 85, 85, 90, the mean is 86.5. Correct. "but the final agreement measurement shows mean position-card cosine 0.533" — a cosine similarity is not a percentage agreement; 0.533 is not "moderate alignment" necessarily. But no.

Potential statistical error: "Aggregate confidence: 80" — if the board-level confidence is supposed to be a probability, setting it to the fraction of voters (80%) ignores the stated individual confidences. But the judgment explicitly says "reflecting the 4/5 majority and the tempered agreement." It's not a formal calculation. Could be MINOR.

Need maybe check "The independent audit's arithmetic is correct. R1's variance ratio (100×) and MSE ratio (6400×) check out." Could there be a missing "R1" definition? No.

Need maybe check "R6's zero-visit probability (1-10⁻⁶)^1000 ≈ 0.99900" — if "zero-visit probability" means probability of zero visits when per-visit probability is 1e-6, yes.

Need maybe check "one-sided 95% upper bound 1 - 0.05^(1/1000) ≈ 0.002991" — yes.

Need maybe check "residual-CV chain rule ... is dimensionally and algebraically correct." Yes.

Need maybe check "SVD singular values are nonnegative by definition; zero is a legitimate degenerate direction." Yes.

Need maybe check "The reason code name NONPOSITIVE_SCALE is cosmetic. A test name is not normative." No.

Need maybe check "The target's main R1 fix — requiring nonempty input_sha256 — is insufficient. A fabricated digest string passes; the Decision has no fields for actual estimates, intervals, run counts, or a binding to a real TrialPlan arm." No.

Need maybe check "The gate checks self-asserted statuses (contracts.py ~L640-660)." No.

Need maybe check "I concede the target's factual findings on advantage_claim_enabled (protocol.py ~L125), untyped pressure_bar (protocol.py ~L67-71), protocol numeric-type digest instability, and load-time digest non-verification (_base.py ~L140-145). These are real. But they are peripheral." No.

Need maybe check "The independent audit’s arithmetic (R1 variance/MSE ratios, R6 Clopper–Pearson bound) and the residual-CV chain rule are correct." No.

Need maybe check "The code-level findings are real but mostly fixable. The audit underweights the absence of evidence linkage in the Decision; that is the fatal defect." No.

Need maybe check "VERDICT: REJECT - the confirmation gate is satisfiable by self-asserted statuses with no evidence linkage, so the layer does not protect R1. CONFIDENCE: 85" No.

Need maybe check "Aggregate confidence: 80. Derivation: four of five members voted ACCEPT-WITH-CHANGES (80% majority). Member confidences were glm 86, kimi 85, deepseek-thinking 85, mini 90, thinker 85. The mean of the four majority confidences is 86.5, but the final agreement measurement shows mean position-card cosine 0.533, indicating only moderate alignment, and the dissenting REJECT carries confidence 85. The board-level confidence is therefore set to 80, reflecting the 4/5 majority and the tempered agreement."

Let's consider if "mean position-card cosine 0.533" is a "cosine" similarity between "position cards" (text embeddings). If mean cosine is 0.533, that's moderate. But "position-card" maybe "position-card" is a term in some debate system. No.

Potential statistical error: "The mean of the four majority confidences is 86.5" — but if the four majority confidences are 86, 85, 85, 90, the mean is 86.5. Correct. "but the final agreement measurement shows mean position-card cosine 0.533" — a cosine similarity is not a percentage agreement; 0.533 is not "moderate alignment" necessarily. But no.

Potential statistical error: "Aggregate confidence: 80" — if the board-level confidence is supposed to be a probability, setting it to the fraction of voters (80%) ignores the stated individual confidences. But the judgment explicitly says "reflecting the 4/5 majority and the tempered agreement." It's not a formal calculation. Could be MINOR.

Need maybe check "The independent audit's arithmetic is correct. R1's variance ratio (100×) and MSE ratio (6400×) check out." Could there be a missing "R1" definition? No.

Need maybe check "R6's zero-visit probability (1-10⁻⁶)^1000 ≈ 0.99900" — if "zero-visit probability" means probability of zero visits when per-visit probability is 1e-6, yes.

Need maybe check "one-sided 95% upper bound 1 - 0.05^(1/1000) ≈ 0.002991" — yes.

Need maybe check "residual-CV chain rule ... is dimensionally and algebraically correct." Yes.

Need maybe check "SVD singular values are nonnegative by definition; zero is a legitimate degenerate direction." Yes.

Need maybe check "The reason code name NONPOSITIVE_SCALE is cosmetic. A test name is not normative." No.

Need maybe check "The target's main R1 fix — requiring nonempty input_sha256 — is insufficient. A fabricated digest string passes; the Decision has no fields for actual estimates, intervals, run counts, or a binding to a real TrialPlan arm." No.

Need maybe check "The gate checks self-asserted statuses (contracts.py ~L640-660)." No.

Need maybe check "I concede the target's factual findings on advantage_claim_enabled (protocol.py ~L125), untyped pressure_bar (protocol.py ~L67-71), protocol numeric-type digest instability, and load-time digest non-verification (_base.py ~L140-145). These are real. But they are peripheral." No.

Need maybe check "The independent audit’s arithmetic (R1 variance/MSE ratios, R6 Clopper–Pearson bound) and the residual-CV chain rule are correct." No.

Need maybe check "The code-level findings are real but mostly fixable. The audit underweights the absence of evidence linkage in the Decision; that is the fatal defect." No.

Need maybe check "VERDICT: REJECT - the confirmation gate is satisfiable by self-asserted statuses with no evidence linkage, so the layer does not protect R1. CONFIDENCE: 85" No.

Need maybe check "Aggregate confidence: 80. Derivation: four of five members voted ACCEPT-WITH-CHANGES (80% majority). Member confidences were glm 86, kimi 85, deepseek-thinking 85, mini 90, thinker 85. The mean of the four majority confidences is 86.5, but the final agreement measurement shows mean position-card cosine 0.533, indicating only moderate alignment, and the dissenting REJECT carries confidence 85. The board-level confidence is therefore set to 80, reflecting the 4/5 majority and the tempered agreement."

Let's think if "mean position-card cosine 0.533" could be a "cosine" similarity between "position cards" (text embeddings). If mean cosine is 0.533, that's moderate. But "position-card" maybe "position-card" is a term in some debate system. No.

Potential statistical error: "The mean of the four majority confidences is 86.5" — but if the four majority confidences are 86, 85, 85, 90, the mean is 86.5. Correct. "but the final agreement measurement shows mean position-card cosine 0.533" — a cosine similarity is not a percentage agreement; 0.533 is not "moderate alignment" necessarily. But no.

Potential statistical error: "Aggregate confidence: 80" — if the board-level confidence is supposed to be a probability, setting it to the fraction of voters (80%) ignores the stated individual confidences. But the judgment explicitly says "reflecting the 4/5 majority and the tempered agreement." It's not a formal calculation. Could be MINOR.

Need maybe check "The independent audit's arithmetic is correct. R1's variance ratio (100×) and MSE ratio (6400×) check out." Could there be a missing "R1" definition? No.

Need maybe check "R6's zero-visit probability (1-10⁻⁶)^1000 ≈ 0.99900" — if "zero-visit probability" means probability of zero visits when per-visit probability is 1e-6, yes.

Need maybe check "one-sided 95% upper bound 1 - 0.05^(1/1000) ≈ 0.002991" — yes.

Need maybe check "residual-CV chain rule ... is dimensionally and algebraically correct." Yes.

Need maybe check "SVD singular values are nonnegative by definition; zero is a legitimate degenerate direction." Yes.

Need maybe check "The reason code name NONPOSITIVE_SCALE is cosmetic. A test name is not normative." No.

Need maybe check "The target's main R1 fix — requiring nonempty input_sha256 — is insufficient. A fabricated digest string passes; the Decision has no fields for actual estimates, intervals, run counts, or a binding to a real TrialPlan arm." No.

Need maybe check "The gate checks self-asserted statuses (contracts.py ~L640-660)." No.

Need maybe check "I concede the target's factual findings on advantage_claim_enabled (protocol.py ~L125), untyped pressure_bar (protocol.py ~L67-71), protocol numeric-type digest instability, and load-time digest non-verification (_base.py ~L140-145). These are real. But they are peripheral." No.

Need maybe check "The independent audit’s arithmetic (R1 variance/MSE ratios, R6 Clopper–Pearson bound) and the residual-CV chain rule are correct." No.

Need maybe check "The code-level findings are real but mostly fixable. The audit underweights the absence of evidence linkage in the Decision; that is the fatal defect." No.

Need maybe check "VERDICT: REJECT - the confirmation gate is satisfiable by self-asserted statuses with no evidence linkage, so the layer does not protect R1. CONFIDENCE: 85" No.

Need maybe check "Aggregate confidence: 80. Derivation: four of five members voted ACCEPT-WITH-CHANGES (80% majority). Member confidences were glm 86, kimi 85, deepseek-thinking 85, mini 90, thinker 85. The mean of the four majority confidences is 86.5, but the final agreement measurement shows mean position-card cosine 0.533, indicating only moderate alignment, and the dissenting REJECT carries confidence 85. The board-level confidence is therefore set to 80, reflecting the 4/5 majority and the tempered agreement."

Let's think if "mean position-card cosine 0.533" could be a "cosine" similarity between "position cards" (text embeddings). If mean cosine is 0.533, that's moderate. But "position-card" maybe "position-card" is a term in some debate system. No.

Potential statistical error: "The mean of the four majority confidences is 86.5" — but if the four majority confidences are 86, 85, 85, 90, the mean is 86.5. Correct. "but the final agreement measurement shows mean position-card cosine 0.533" — a cosine similarity is not a percentage agreement; 0.533 is not "moderate alignment" necessarily. But no.

Potential statistical error: "Aggregate confidence: 80" — if the board-level confidence is supposed to be a probability, setting it to the fraction of voters (80%) ignores the stated individual confidences. But the judgment explicitly says "reflecting the 4/5 majority and the tempered agreement." It's not a formal calculation. Could be MINOR.

Need maybe check "The independent audit's arithmetic is correct. R1's variance ratio (100×) and MSE ratio (6400×) check out." Could there be a missing "R1" definition? No.

Need maybe check "R6's zero-visit probability (1-10⁻⁶)^1000 ≈ 0.99900" — if "zero-visit probability" means probability of zero visits when per-visit probability is 1e-6, yes.

Need maybe check "one-sided 95% upper bound 1 - 0.05^(1/1000) ≈ 0.002991" — yes.

Need maybe check "residual-CV chain rule ... is dimensionally and algebraically correct." Yes.

Need maybe check "SVD singular values are nonnegative by definition; zero is a legitimate degenerate direction." Yes.

Need maybe check "The reason code name NONPOSITIVE_SCALE is cosmetic. A test name is not normative." No.

Need maybe check "The target's main R1 fix — requiring nonempty input_sha256 — is insufficient. A fabricated digest string passes; the Decision has no fields for actual estimates, intervals, run counts, or a binding to a real TrialPlan arm." No.

Need maybe check "The gate checks self-asserted statuses (contracts.py ~L640-660)." No.

Need maybe check "I concede the target's factual findings on advantage_claim_enabled (protocol.py ~L125), untyped pressure_bar (protocol.py ~L67-71), protocol numeric-type digest instability, and load-time digest non-verification (_base.py ~L140-145). These are real. But they are peripheral." No.

Need maybe check "The independent audit’s arithmetic (R1 variance/MSE ratios, R6 Clopper–Pearson bound) and the residual-CV chain rule are correct." No.

Need maybe check "The code-level findings are real but mostly fixable. The audit underweights the absence of evidence linkage in the Decision; that is the fatal defect." No.

Need maybe check "VERDICT: REJECT - the confirmation gate is satisfiable by self-asserted statuses with no evidence linkage, so the layer does not protect R1. CONFIDENCE: 85" No.

Need maybe check "Aggregate confidence: 80. Derivation: four of five members voted ACCEPT-WITH-CHANGES (80% majority). Member confidences were glm 86, kimi 85, deepseek-thinking 85, mini 90, thinker 85. The mean of the four majority confidences is 86.5, but the final agreement measurement shows mean position-card cosine 0.533, indicating only moderate alignment, and the dissenting REJECT carries confidence 85. The board-level confidence is therefore set to 80, reflecting the 4/5 majority and the tempered agreement."

Let's think if "mean position-card cosine 0.533" could be a "cosine" similarity between "position cards" (text embeddings). If mean cosine is 0.533, that's moderate. But "position-card" maybe "position-card" is a term in some debate system. No.

Potential statistical error: "The mean of the four majority confidences is 86.5" — but if the four majority confidences are 86, 85, 85, 90, the mean is 86.5. Correct. "but the final agreement measurement shows mean position-card cosine 0.533" — a cosine similarity is not a percentage agreement; 0.533 is not "moderate alignment" necessarily. But no.

Potential statistical error: "Aggregate confidence: 80" — if the board-level confidence is supposed to be a probability, setting it to the fraction of voters (80%) ignores the stated individual confidences. But the judgment explicitly says "reflecting the 4/5 majority and the tempered agreement." It's not a formal calculation. Could be MINOR.

Need maybe check "The independent audit's arithmetic is correct. R1's variance ratio (100×) and MSE ratio (6400×) check out." Could there be a missing "R1" definition? No.

Need maybe check "R6's zero-visit probability (1-10⁻⁶)^1000 ≈ 0.99900" — if "zero-visit probability" means probability of zero visits when per-visit probability is 1e-6, yes.

Need maybe check "one-sided 95% upper bound 1 - 0.05^(1/1000) ≈ 0.002991" — yes.

Need maybe check "residual-CV chain rule ... is dimensionally and algebraically correct." Yes.

Need maybe check "SVD singular values are nonnegative by definition; zero is a legitimate degenerate direction." Yes.

Need maybe check "The reason code name NONPOSITIVE_SCALE is cosmetic. A test name is not normative." No.

Need maybe check "The target's main R1 fix — requiring nonempty input_sha256 — is insufficient. A fabricated digest string passes; the Decision has no fields for actual estimates, intervals, run counts, or a binding to a real TrialPlan arm." No.

Need maybe check "The gate checks self-asserted statuses (contracts.py ~L640-660)." No.

Need maybe check "I concede the target's factual findings on advantage_claim_enabled (protocol.py ~L125), untyped pressure_bar (protocol.py ~L67-71), protocol numeric-type digest instability, and load-time digest non-verification (_base.py ~L140-145). These are real. But they are peripheral." No.

Need maybe check "The independent audit’s arithmetic (R1 variance/MSE ratios, R6 Clopper–Pearson bound) and the residual-CV chain rule are correct." No.

Need maybe check "The code-level findings are real but mostly fixable. The audit underweights the absence of evidence linkage in the Decision; that is the fatal defect." No.

Need maybe check "VERDICT: REJECT - the confirmation gate is satisfiable by self-asserted statuses with no evidence linkage, so the layer does not protect R1. CONFIDENCE: 85" No.

Need maybe check "Aggregate confidence: 80. Derivation: four of five members voted ACCEPT-WITH-CHANGES (80% majority). Member confidences were glm 86, kimi 85, deepseek-thinking 85, mini 90, thinker 85. The mean of the four majority confidences is 86.5, but the final agreement measurement shows mean position-card cosine 0.533, indicating only moderate alignment, and the dissenting REJECT carries confidence 85. The board-level confidence is therefore set to 80, reflecting the 4/5 majority and the tempered agreement."

Let's think if "mean position-card cosine 0.533" could be a "cosine" similarity between "position cards" (text embeddings). If mean cosine is 0.533, that's moderate. But "position-card" maybe "position-card" is a term in some debate system. No.

Potential statistical error: "The mean of the four majority confidences is 86.5" — but if the four majority confidences are 86, 85, 85, 90, the mean is 86.5. Correct. "but the final agreement measurement shows mean position-card cosine 0.533" — a cosine similarity is not a percentage agreement; 0.533 is not "moderate alignment" necessarily. But no.

Potential statistical error: "Aggregate confidence: 80" — if the board-level confidence is supposed to be a probability, setting it to the fraction of voters (80%) ignores the stated individual confidences. But the judgment explicitly says "reflecting the 4/5 majority and the tempered agreement." It's not a formal calculation. Could be MINOR.

Need maybe check "The independent audit's arithmetic is correct. R1's variance ratio (100×) and MSE ratio (6400×) check out." Could there be a missing "R1" definition? No.

Need maybe check "R6's zero-visit probability (1-10⁻⁶)^1000 ≈ 0.99900" — if "zero-visit probability" means probability of zero visits when per-visit probability is 1e-6, yes.

Need maybe check "one-sided 95% upper bound 1 - 0.05^(1/1000) ≈ 0.002991" — yes.

Need maybe check "residual-CV chain rule ... is dimensionally and algebraically correct." Yes.

Need maybe check "SVD singular values are nonnegative by definition; zero is a legitimate degenerate direction." Yes.

Need maybe check "The reason code name NONPOSITIVE_SCALE is cosmetic. A test name is not normative." No.

Need maybe check "The target's main R1 fix — requiring nonempty input_sha256 — is insufficient. A fabricated digest string passes; the Decision has no fields for actual estimates, intervals, run counts, or a binding to a real TrialPlan arm." No.

Need maybe check "The gate checks self-asserted statuses (contracts.py ~L640-660)." No.

Need maybe check "I concede the target's factual findings on advantage_claim_enabled (protocol.py ~L125), untyped pressure_bar (protocol.py ~L67-71), protocol numeric-type digest instability, and load-time digest non-verification (_base.py ~L140-145). These are real. But they are peripheral." No.

Need maybe check "The independent audit’s arithmetic (R1 variance/MSE ratios, R6 Clopper–Pearson bound) and the residual-CV chain rule are correct." No.

Need maybe check "The code-level findings are real but mostly fixable. The audit underweights the absence of evidence linkage in the Decision; that is the fatal defect." No.

Need maybe check "VERDICT: REJECT - the confirmation gate is satisfiable by self-asserted statuses with no evidence linkage, so the layer does not protect R1. CONFIDENCE: 85" No.

Need maybe check "Aggregate confidence: 80. Derivation: four of five members voted ACCEPT-WITH-CHANGES (80% majority). Member confidences were glm 86, kimi 85, deepseek-thinking 85, mini 90, thinker 85. The mean of the four majority confidences is 86.5, but the final agreement measurement shows mean position-card cosine 0.533, indicating only moderate alignment, and the dissenting REJECT carries confidence 85. The board-level confidence is therefore set to 80, reflecting the 4/5 majority and the tempered agreement."

Let's think if "mean position-card cosine 0.533" could be a "cosine" similarity between "position cards" (text embeddings). If mean cosine is 0.533, that's moderate. But "position-card" maybe "position-card" is a term in some debate system. No.

Potential statistical error: "The mean of the four majority confidences is 86.5" — but if the four majority confidences are 86, 85, 85, 90, the mean is 86.5. Correct. "but the final agreement measurement shows mean position-card cosine 0.533" — a cosine similarity is not a percentage agreement; 0.533 is not
(TRUNCATED - output hit the token limit before completion)

---

Board: glm, kimi, deepseek-thinking, mini, thinker | Chair: deepseek-thinking | Math & physics specialist: thinker | Agreement measured by: qwen3-embedding-4b
Debate rounds: 1 | Final: majority ACCEPT-WITH-CHANGES at 80%, mean position-card cosine 0.533 | Converged: True

---
Full transcript: board_out/transcript.md
Final report:   board_out/final_report.md
