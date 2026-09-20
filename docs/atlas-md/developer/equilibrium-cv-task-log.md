# Equilibrium CV selection — per-task completion record

One entry per task from
[the implementation plan](equilibrium-cv-implementation-plan.md). Each entry
records what was actually run, not what was expected to pass.

---

## T00 — Baseline, contracts and test entry point

| Field | Value |
|---|---|
| Base SHA | `cdee6cc87ac540b0bac9b230f389c77d70383faf` (`main`) |
| Branch | `cv-select/t00-contracts` |
| Date | 2026-09-20 |

### Changed paths

| Path | Change |
|---|---|
| `gareus/cv_selection/__init__.py` | New package; documents the NumPy-only import policy |
| `gareus/cv_selection/_base.py` | New; primitive validation, digests, `_Artifact` base |
| `gareus/cv_selection/vocabulary.py` | New; stages, roles, phases and the eight decision statuses |
| `gareus/cv_selection/contracts.py` | New; the six artifacts, readiness validator, public façade |
| `tests/test_cv_selection_contracts.py` | New; rejection and round-trip cases |
| `tests/fixtures/cv_selection/generate.py` | New; regenerates the tiny artifacts |
| `tests/fixtures/cv_selection/*.json` | New; six generated artifacts pinning the encoding |
| `docs/atlas-md/developer/equilibrium-cv-implementation-plan.md` | Plan copied into the tracked docs path |
| `docs/atlas-md/developer/equilibrium-cv-selection-spec-v0.1.md` | Historical design evidence |
| `docs/atlas-md/developer/equilibrium-cv-selection-adversarial-review.md` | Historical design evidence |
| `docs/atlas-md/developer/equilibrium-cv-task-log.md` | This record |
| `mkdocs.yml` | Nav entries so the strict docs build keeps covering the new pages |

`pyproject.toml` needed **no** change: `[tool.setuptools.packages.find]` already
matches `gareus*`, so `gareus.cv_selection` is installed automatically, and the
contracts add no dependency beyond the existing NumPy core. The console script
`gareus-select-cvs` is deliberately left to T10, which owns the CLI.

### Module layout

`contracts.py` is the public façade — later tasks import everything from it, so
`from gareus.cv_selection import contracts as C` keeps working regardless of how
the internals move. The first cut was a single 1148-line module, over the
repository's 800-line ceiling; it was split into `_base.py` (mechanism:
validation primitives, digests, the `_Artifact` base) and `vocabulary.py` (the
controlled vocabularies and the role/phase rule), with the artifacts and the
readiness validator staying in `contracts.py`.

Each split was verified pure by regenerating the six fixtures: all `sha256`
values came back byte-identical, because digests are taken over JSON payloads
and never over module layout. (The fixtures' digests *did* change later, when
the adversarial round below added required fields to two schemas — that is a
deliberate schema change, not a refactor.)

### Interfaces introduced

* Artifacts: `FeatureSchema`, `CandidateSet`, `ObservablePanel`, `ProtocolSpec`,
  `TrialPlan`, `Decision` — each with `from_mapping`, `from_json_bytes`,
  `to_mapping`, `to_json_bytes` and a content `sha256`.
* Vocabularies: `Stage`, `StudyRole`, `PhaseKind`, `SearchMode`, the eight
  decision statuses, `ExitCode`, `ReasonCode`.
* Functions: `artifact_digest`, `verify_artifact_digest`, `require_feature_binding`,
  `measurement_phase_for`, `validate_role_phase`, `validate_protocol`.
* Error: `ContractError(IntegrityError)` carrying `.reason: ReasonCode`.

Digests reuse `gareus.correctness._io.json_bytes` (sorted keys, no NaN,
duplicate-key rejection), computed over the artifact body with its own `sha256`
field removed. `CandidateSet.primary_definition` is validated by the existing
`state_identity._canonical_cv`, which already refuses filename-based identity.

### Design decisions worth keeping

* **Study role is not a sampling phase.** `StudyRole` (discovery, calibration,
  engineering, screen, confirm) is disjoint from `PhaseKind`, which mirrors
  `gareus.correctness.sampling_policy` exactly. A screening campaign's
  measurement phase *is* `production`, so the existing eligibility invariant
  stays intact instead of being weakened to describe the selector.
* **Eight separate decision statuses.** Collapsing them would undo review
  finding R1: a reproducibly trapped arm has excellent precision.
  `CONFIRMED_FOR_DECLARED_PANEL` additionally requires a named arm and
  non-blocking values on all five gating statuses.
* **Readiness never supplies a default.** `validate_protocol(..., stage=)`
  enumerates each unresolved policy with the task that must produce it, and
  reports a declared-but-absent or declared-but-different artifact rather than
  assuming it.
* **Components are individual directions.** `component_index` is 1-based and
  unique; a `component_count` field is rejected as unknown, so the legacy
  `compute_bootstrap_torsion_pca(component=...)` count semantics can never be
  read into a selector artifact.

### Acceptance cases exercised

Unknown field/schema version, missing field, duplicate JSON key, non-finite
number, non-contiguous feature index, invalid trig/atom quadruplet, feature
identity and width mismatch, duplicate/out-of-range component index, incomplete
regression rows, zero scaling, filename identity, primary-panel size, duplicate
observable id, non-positive tolerance, out-of-range quantile, fold-specific
generator preset, disabled native-blind allowlist, historical data promoted to
confirmatory, NPT without pressure, artifact digest mismatch and absence,
role/phase conflict, zero-budget arm, unequal arm cost, duplicate arm id,
replica-cap and rung-divisibility violations, invalid status, free-text reason,
confirmation without a selected arm, confirmation over a protocol disagreement
or unresolved precision, integrity/outcome conflict, exit-code distinctness, and
a NumPy-only import with pyarrow/duckdb/pymbar/openmm/yaml/pandas/mdtraj blocked.

### Commands run and outcomes

| Command | Outcome |
|---|---|
| `python -m pytest -q tests/test_cv_selection_contracts.py tests/test_stage_phase_identity.py tests/test_packaging_root_modules.py tests/test_package_smoke.py` | 115 passed, 2 failed — both pre-existing in `test_package_smoke.py` |
| `python -m pytest -q tests/` (baseline at this SHA) | 3075 passed, 10 failed, 3 skipped, 19:36 |
| `python -m mkdocs build --strict` | exit 0 |
| `python tests/fixtures/cv_selection/generate.py` | six artifacts written |

### Pre-existing failures at `cdee6cc` (recorded, not fixed here)

None are touched by T00; the plan's task prompt keeps unrelated fixes separate.

1. `test_atlas_md_docs.py::test_atlas_md_has_mkdocs_configuration_and_required_pages` —
   asserts `site_name: ATLAS-MD`, the file says `ATLaS-MD` (casing drift).
2. `test_atlas_md_docs.py::test_atlas_md_documents_rendered_diagrams_and_synthetic_harness` —
   `index.md` no longer contains a ```` ```mermaid ```` block.
3. `test_example_configs.py::test_example_config_parses_without_unknown_keys[chignolin_genpept_contact_bias_sigma.yaml]` — `SystemExit: 2`.
4. `test_npt_pep_adapter_boost.py::test_finite_difference_boost_force_matches_scaling_factor_expression` —
   finite-difference vs integrator expression mismatch. See the verification
   note below: this one is **intrinsically nondeterministic** (~41% failure rate
   run in isolation), so it is not a fixed pre-existing failure at all.
5. `test_package_smoke.py::test_tiny_lambda_ladder_run_completes_end_to_end_slow` —
   `ArrowInvalid` reading `parquet_manifest.json` as Parquet.
6. `test_package_smoke.py::test_official_package_version_is_v07` — expects `0.8`,
   package is `0.8.1` since the `release: v0.8.1` commit.
7-10. `test_thermodynamic_validity_2d_rough.py` and
   `test_thermodynamic_validity_real_md.py`, `test_mutation_is_caught[...]` x2 each —
   `IntegrityError: beta must be positive`. What was observed is only this: the
   mutation battery injects `beta=0`/negative beta, and a strict guard raises
   before the oracle evaluates the mutation, so the test fails on a different
   channel than the one it asserts on. **Whether the mutation would still be
   caught downstream was not established here** — it needs its own run. These
   four are the teeth of the thermodynamic oracle suite, so the question matters
   and is left open rather than answered by inspection.

Provenance of each item:

* Items 5-10 also appear in the untracked `atlas_pytest_full_run.log` from
  2026-09-12, so they predate this branch.
* Items 1 and 2 are newer than that log and were confirmed pre-existing by
  reading their assertions: both check `site_name` casing and a ```` ```mermaid ````
  block in `index.md`, neither of which is the nav block this task edited. The
  strict docs build passes with the nav additions.
* Item 4 is newer than that log and is **not** explained by reading assertions —
  a Pep-GaMD finite-difference force mismatch has nothing to do with anything
  T00 touches, but that is an argument, not evidence. It was therefore run
  directly, and the result changed the conclusion. See below.

### Verification note: item 4 is an intrinsically flaky test (corrected)

**An earlier version of this record called item 4 cross-test state pollution.
That was wrong, and the evidence row it rested on does not reproduce.** An
independent re-measurement established:

| Tree | Invocation | Result |
|---|---|---|
| clean `cdee6cc` worktree | that file + `test_thermodynamic_validity_2d_rough.py` | the file is green; the two `beta must be positive` failures reproduce |
| `cv-select/t00-contracts` | the single node, run 17 times | **7 failures (~41%)** |
| `cv-select/t00-contracts` | whole suite | fails |

The original "27 passed alone" observation was a real run, but it is one draw
from roughly a 59%-pass distribution, not evidence of determinism. Two
independent re-runs of that file alone gave `1 failed, 26 passed`.

Bisection found no poisoning neighbour because there is none: splitting the
file's preceding tests yields a failing union whose two halves are each green,
which is the signature of chance rather than of a stateful neighbour.
Corroborating detail: failing runs finish in 3.6-6.6 s against ~7.9-8.4 s for
passing ones, and always abort on the first probe atom, so the nondeterminism
sits in context/system construction rather than in the finite-difference
numerics.

The earlier remark that the randomising order plugin is absent is true but
beside the point: the variable is inside the test, not in the ordering.

**Consequence.** The follow-up is to make the test deterministic (seed it, pin
the platform), not to hunt a polluting neighbour. Until then it must not be
counted among the pre-existing baseline failures, because it is not a fixed
quantity.

### Verification note: the baseline count is indicative, not exact

Treat "3075 passed, 10 failed, 3 skipped" as indicative. Two independent
collections put the baseline test set at **3087** items while that triple sums
to 3088; the discrepancy of one was not resolved. More importantly, one of the
ten failures is the ~41% coin flip above, so the failure count is not a
reproducible quantity. What is solid: the six failures also present in the
2026-09-12 log, and the two `test_atlas_md_docs.py` assertions, both confirmed
by reading them.

"Working tree clean" in this record means no *tracked* file was modified. Four
untracked entries (two `.docx` files, a log, and `chignolin_knowledge_base/`)
predate this branch and are not part of it.

### Adversarial review round (2026-09-20)

T00 was put through three independent fresh-context reviews and a five-model
adversarial panel. **Every finding acted on below was reproduced with a direct
probe first**; findings that did not reproduce are recorded as such rather than
acted on. The panel returned **4/5 ACCEPT-WITH-CHANGES** (confidences 85-90)
with one **REJECT** from the mathematics/physics seat (confidence 85);
aggregate 80, mean position-card cosine 0.533. The specialist audit of the
final judgment raised no flags.

#### Reproduced and fixed

| Finding | Why it mattered |
|---|---|
| Readiness gate tested only `is None` | `burn_in_ticks=-5`, `campaigns_per_arm="many"`, `gpus=-3`, `layout=""` all returned `ready=True, missing=()`. A scientifically impossible study was certified launch-ready — the exact failure R9 exists to prevent |
| Four refusals carried no reason code | Duplicate JSON key, non-finite number, path-based CV identity, unsupported units all raised a bare error, contradicting the module's promise that callers branch on codes |
| `.strip()` does not remove format characters | A CONFIRMED verdict could name an arm made only of U+200B/U+FEFF/U+2060/U+00AD/U+180E: invisible everywhere, yet unequal to `""`, so no downstream comparison can match it |
| Native-derived primary CV was screen-ready | Blindness was asserted by three protocol strings and never checked against the candidate set, so a primary CV of kind `rmsd-to-native-pdb` passed |
| The frozen panel was gameable | `halfwidth_tolerance: 1e9` made `precision: MET` free for every arm, and eight copies of one measurement satisfied the panel |
| Two disagreeing digests for one artifact | `artifact_digest` hashed the payload as written; `_parse` hashed rebuilt, coerced values. An artifact whose JSON said `1` where the schema means `1.0` could never be matched by a protocol declaring its on-disk digest, and a tampered payload was silently re-hashed |
| A confirmation could rest on nothing | Empty `input_sha256`, no tested protocols, or citing `CONFIRMATION_BLOCKED` among its own reasons |
| The writer bypassed every consistency rule | The rules lived only in `_parse`, so a producer could hand-build `integrity=FAIL` + `CONFIRMED` and serialize it with a valid digest; the contradiction surfaced only in whichever later task loaded it |
| `0.0` passed as a feature index | `0.0 != 0` is `False` |
| `1` passed as a boolean flag | `1 in (True, False)` is `True` |
| Unvalidated NPT pressure, unvalidated `study_id` | A negative pressure and a null study id both parsed |
| `Stage` and `StudyRole` share value names | A transposed pair parsed silently; now they must agree |
| `temperature_k: 300` vs `300.0` | The same target hashed two ways. Real-valued protocol fields are now coerced before hashing |

One refinement on the panel's condition 4. It asked for `unresolved_regions` to
be **empty** for an unqualified confirmation. What ships is stricter where it
matters and looser where it should be: a region of **unknown** mass
(`UNRESOLVED_SUPPORT`) blocks confirmation, while a region whose mass is
**bounded small** (`MASS_BOUNDED_SMALL`) does not. That is R6's actual
distinction — for an IID region of true probability 1e-6 with n=1000, zero
visits happens ~99.9% of the time and the exact one-sided 95% bound is ~0.003,
already inside a 0.02 tolerance, so demanding a visit there would reject an
adequate result.

#### Recorded, not acted on

* **The dissent stands.** The mathematics/physics seat voted REJECT on the
  grounds that the confirmation gate is satisfiable by *self-asserted* statuses:
  the `Decision` has no fields for estimates, intervals or run counts, so a
  fabricated digest string passes. Evidence linkage was strengthened here
  (mandatory artifact digests, two protocols for an agreement claim, no
  unknown-mass regions), but the dissent's deeper point is correct and not
  answered by T00: **a real evidence model belongs to the evaluation task**,
  which owns estimates and intervals. This is a known, deliberate limitation of
  the contracts layer, not an oversight.
* **Invariant 5 is not enforceable here.** Whether a stored vector is one SVD
  direction or a variance-weighted mixture of several cannot be decided at this
  layer: both are float vectors of the same width. A probe confirmed a mixture
  filed as `component_index: 3` is accepted verbatim. Index, range and
  uniqueness are enforced; *provenance* of the direction must be enforced by the
  fitting task. The module comment previously read as though the contract
  guarded it.
* **Cross-artifact binding is still absent from readiness.** `validate_protocol`
  compares declared digests against supplied ones but never calls
  `require_feature_binding`, never compares `physical_system_sha256` between
  protocol and candidate set, and never binds a `TrialPlan` to its protocol's
  layout, rungs, replica cap or budget ceiling. It receives digests, not
  objects, so closing this is an interface change and belongs with the task that
  owns trial planning.
* **`singular_value = 0` is legitimate.** The dissent is right that SVD singular
  values are non-negative by definition and a zero is a degenerate but valid
  direction. Only an all-zero *vector* is refused, because that defines no
  coordinate and its umbrella would apply no force.
* The new test file is **not in CI**. CI runs a curated three-file list plus the
  strict docs build; wiring the selector tests in belongs to the CLI/CI task.

### Statuses became summaries of numbers, not assertions (2026-09-20, later)

Stepping back from the findings to the goal: T00 exists so that a wrong
equilibrium population cannot be published without an alarm. The panel's
dissent named the residual self-deception precisely — the eight statuses were
*typed in* by the producer, so every consistency rule was policing the grammar
of a possible lie rather than its truth. Fixing the individual holes did not
touch that.

What changed:

* **`precision` and `cross_protocol` are now derived, not declared.** A
  `Decision` carries `evidence`: per arm, the estimate, half-width, tolerance
  and campaign count for every primary observable. On load the parser
  recomputes `precision` as `max_a (h_a/eps_a)^2 <= 1` over the selected arm
  and `cross_protocol` by the plan §6.4 interval rule over every pair of arms
  and every observable, and refuses a declared status that disagrees
  (`EVIDENCE_STATUS_MISMATCH`). One established disagreement anywhere
  dominates. A missing half-width derives `UNRESOLVED`; fewer than two arms
  derives `NOT_COMPARED`; a selected arm with no evidence is refused.
* **`MASS_BOUNDED_SMALL` must carry its bound.** A region record now holds
  `upper_bound` and `bound_method`, the latter from a fail-closed allowlist
  currently containing only `exact_iid_binomial_zero_count` — the
  `1 - alpha**(1/n)` bound that is exact for IID target draws and invalid for
  weighted replica-exchange samples. No Kish-ESS substitute is accepted. An
  `UNRESOLVED_SUPPORT` region may not carry a bound.
* **A probability tolerance must be below 0.5.** A half-width of 0.5 covers
  the unit interval, so any gate against it is free. The bound applies to panel
  half-widths, the cross-protocol band, the novelty band, and the tolerance
  recorded with each estimate.
* `dependence`, `reproducibility` and `support` remain **asserted**, and the
  module now says so by name (`ASSERTED_STATUSES`). They rest on batch
  covariances and per-campaign estimates this artifact does not carry; giving
  them the same treatment is the evaluation task's job and would need those
  numbers recorded first.

What this buys, stated exactly: fabricating a verdict now means fabricating
*numbers*, tied to artifact digests, that an auditor can re-derive and
cross-check. It does not make a fabricated number true. That is the most a
contract layer can do, and it is the line the dissent asked for.

Tests 114 -> 131. Fixture `decision.json` re-hashed (schema gained two fields).

### Unresolved and handed on

* `ProtocolSpec` sections are validated for exact key sets but their *values*
  are only shape-checked. Cross-field scientific validation (does the layout fit
  the replica cap, does the budget cover the arm count) belongs to T09, which
  owns the budget compiler.
* `contracts.py` imports `state_identity._canonical_cv`, a private name. It is
  the correct validator, so T01 — which owns the state-identity wiring — should
  either promote it to a public helper or keep the import and note it.
* `ReasonCode` already carries codes for later tasks (`INSUFFICIENT_SUPPORT`,
  `BUDGET_EXHAUSTED`) so those tasks do not each invent their own spelling.

### Next task readiness

T01 (state wiring) and T03 (residual models) are unblocked and independent.
Both must import their types from `gareus.cv_selection.contracts` rather than
re-declaring them.
