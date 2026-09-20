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
| `tests/test_cv_selection_contracts.py` | New; 67 rejection/round-trip cases |
| `tests/fixtures/cv_selection/generate.py` | New; regenerates the tiny artifacts |
| `tests/fixtures/cv_selection/*.json` | New; six generated artifacts pinning the encoding |
| `docs/atlas-md/developer/equilibrium-cv-implementation-plan.md` | Plan copied into the tracked docs path |
| `docs/atlas-md/developer/equilibrium-cv-selection-spec-v0.1.md` | Historical design evidence |
| `docs/atlas-md/developer/equilibrium-cv-selection-adversarial-review.md` | Historical design evidence |
| `mkdocs.yml` | Three nav entries so the strict docs build keeps covering them |

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

The split was verified pure by regenerating the six fixtures: all `sha256`
values are byte-identical before and after, because digests are taken over JSON
payloads and never over module layout.

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
   finite-difference vs integrator expression mismatch, **only when the whole
   suite runs**. See the verification note below: this one is not a plain
   pre-existing failure, it is a test-isolation failure.
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

### Verification note: item 4 is a test-isolation failure

A clean `git worktree` was created at `cdee6cc` with none of this branch's files
present, and `tests/test_npt_pep_adapter_boost.py` was run there:

| Tree | Invocation | Result |
|---|---|---|
| clean `cdee6cc` worktree | that file + `test_thermodynamic_validity_2d_rough.py` | `test_npt_pep_adapter_boost.py` fully green; the two `beta must be positive` failures reproduce |
| `cv-select/t00-contracts` | that file alone | 27 passed |
| `cv-select/t00-contracts` | whole suite | `test_finite_difference_boost_force_matches_scaling_factor_expression` fails |

So the finite-difference test **passes in isolation on both trees** and fails
only inside a whole-suite run. That is cross-test state pollution, not a
deterministic defect and not attributable to T00, whose modules are on no import
path this test touches. `pytest-randomly` is not installed in this environment,
so test ordering is fixed and is *not* the variable — the likely culprit is
global simulation state (platform, force groups, or an integrator left
configured) surviving from an earlier test in the session.

This is worth its own investigation before T04, which builds residual-CV forces
and will want this file's finite-difference check to be trustworthy in CI. It is
recorded here rather than fixed, per the plan's rule on keeping unrelated fixes
separate.

The two `beta must be positive` failures (items 7-8 of the list above, in
`test_thermodynamic_validity_2d_rough.py`) **did** reproduce in the clean
worktree, confirming those as genuinely pre-existing.

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
