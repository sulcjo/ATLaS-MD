# CVaux Stage C: Storage, Resume and Strict Pooling — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Revision:** v3, 2026-10-08. Revised against the code as built at `feat/cvaux` HEAD `d4415b1` (Stages A and B implemented). It applies the pre-flight scan (`.superpowers/sdd/2026-10-07-cvaux-stage-c-storage-resume-pooling/preflight.md`) and the binding "Pre-flight rulings" in that directory's `progress.md` (B1–B5, H1–H4 and the other-defect fixes; B4 is a user ruling). v2 history (verifier findings C1–C19, S1–S15, decisions D1–D14) is kept in the self-review record. The **Revision record (v3)** at the end maps every ruling to the task that applies it.

**Goal:** Ensure no active auxiliary term can disappear through a writer, loader, early return, resume or state-hash path, and lift Stage B's engineering-run restrictions.
- **Record:** every thermodynamic sample stores the runtime exchange-matrix z (float64) and the complete ordered torsion basis. Every exchange decision is an ordered, replayable event.
- **Resume:** checkpoints bind the auxiliary state table, the models, the Context parameters (verified *before* they are re-applied), the topology, the kernel and the ledger anchor. Every `--resume` (fast or not) refuses an auxiliary-capability mismatch. In an auxiliary run, a crashed parent segment is re-sealed at the checkpoint.
- **Pool:** the strict fixed-state exporter and `load_parquet` evaluate auxiliary energies offline from the stored features. Parity with the stored runtime z is checked at a tolerance chosen from the recorded platform precision. Whole-row exclusion is counted and reported. Every other loader refuses auxiliary data.

**Architecture:**
- **Storage format.** New pure-Python helpers live in `gareus/auxiliary_cv/`: `sample_schema.py`, `ledger.py`, `checkpoint.py`, `offline.py`, `runtime_definition.py`, `runtime_io.py`, `resume_check.py`, `parity_report.py`. The existing writers (`gareus/store.py`), the Parquet manifest and the window snapshot gain *optional* auxiliary payloads. Pure-JSON auxiliary *detection* helpers live in `gareus/kernel_identity.py` (import-light), so legacy loaders never import the `gareus.auxiliary_cv` package (Stage B pins that off-path rule for the window loader).
- **No legacy behaviour change.** Every production edit is gated by an auxiliary condition (`args._aux_runtime`, `_aux_io`, `args.aux_cv_model`). The resume re-seal of a crashed parent (Task 5) runs **only for auxiliary runs** (user ruling B4); the legacy seal code stays verbatim. Known legacy residual, left for a separate PR: after an exception crash plus resume, a legacy run still counts post-checkpoint rows of the parent (sealed at the crash step) and of the child.
- **Thermodynamic pooling.**
  - Pooling goes through the strict exporter (`gareus/correctness/export.py`) and the single-run Parquet loader (`gareus/mbar_analysis/loaders.py:load_parquet`).
  - `load_npz`, `load_csv`, round augmentation, the adaptive branch of `load_data`, the adaptive union loaders (including every pilot dir) and the legacy NPZ export refuse auxiliary runs. MVP pools only fixed-state phases (spec Sections 8 and 14).
  - In an auxiliary run, `load_parquet` refuses segments of a non-auxiliary kernel (mixed v2/v3_aux directory, ruling H3) and segments without a window snapshot (ruling H2) with no override; Stage-B-era auxiliary segments without stored features are refused unless `exclude_segments_without_aux_features=True`.
- **Kernel eligibility.** Stage B classifies every auxiliary snapshot as `aux_unpersisted`. Stage C makes a segment VERIFIED only when its samples manifest carries the `atlas-aux-samples-v1` payload and its snapshot is a frozen v2 snapshot naming the same model. The samples manifest is read only for auxiliary snapshots (ruling B5).
- **Order.** Tasks 1–11 are library code (Stage A/B names only). Task 12 binds resume (unit-testable, no MD). Task 13 wires production I/O. Tasks 14–15 are real runs: CPU end-to-end resume control, and GPU parity plus cost on aurum2. Task 16 is the exit gate and handoff.

**Tech Stack:** Python 3, NumPy, pyarrow, duckdb, OpenMM 8.5 (Reference/CPU in unit tests; CUDA/OpenCL on aurum2 in Task 15), pytest.

**Spec:** `docs/superpowers/specs/2026-10-07-auxiliary-cv-gibbs-production-spec.md` (main dc30285), Sections 6, 7, 8, 9, 10, 15 (Storage / Offline energies / Resume rows), 16 Stage C and 17. Read Sections 6–9 before starting.

**Depends on (built, at HEAD `d4415b1`):**
- **Stage A** (`gareus/auxiliary_cv/{model,features,evaluate,force}.py`, `gareus/correctness/state_identity.py` v2): `AuxModel` (`scale`, `periodic_imaging`, `identity_mapping()`, `to_mapping()`), `aux_models` stores identity bodies, every v2 row carries `instance` (`_check_instances` refuses a row without one), `spawn_parent_state_id` is a `state_instance_id` string, roles are enforced against `aux_k`, `normalize_windows` maps an inactive axis (k = 0) to centre 0.
- **Stage B** (`gareus/auxiliary_cv/{runtime,state_table}.py`, `gareus/kernel_identity.py`, `run_gareus` wiring). The names this plan consumes are listed below with their HEAD locations.

**Test runner (user policy):** tests in this repo are run by the local free runner `opencode`, never directly. Each "Run:" step below names the pytest arguments. Dispatch them as
`opencode run "In /run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/.claude/worktrees/cvaux: run python -m pytest -q <arguments> and report pass/fail/error counts and every failing test id. Read-only: do not edit, commit or fix anything."`
and treat its report as the result. A Bash hook in this harness refuses commands that call the test runner directly.

**Targeted tests only** (user preference): run the files a task names, never the whole suite.

## Global Constraints

- **Off path byte-identical, pinned.** With the auxiliary capability off, all of the following are unchanged:
  - the Parquet sample and exchange schemas, including dtypes (pinned as `schema.to_string()` literals, Tasks 3 and 4);
  - the manifest key sets;
  - window-snapshot bytes (sha pins, Task 8);
  - the strict-export `input_signature` (sha pin, Task 8);
  - checkpoint manifests (`aux` key absent, Task 6);
  - the legacy resume seal (`seal_segment(..., status="interrupted")` under `_parent_was_running`, Task 5, ruling B4);
  - eligibility classification of every non-auxiliary segment (ruling B5).

  Pins were computed on main dc30285 by `/home/sulcjo/.claude/jobs/969c720f/tmp/verify_C/pins.py` and **re-computed at HEAD `d4415b1` on 2026-10-08** (read-only snippet `/tmp/stagec_pins/pins.py`): the legacy sample and exchange schema strings, the manifest key set, the snapshot shas `45a1984c…`/`23b5638e…` and the export `input_signature` `4bb8e35b…` are unchanged.
- **Strict opt-in (c10 resumes on this code).** Every new `run_gareus` call that builds, reads or writes auxiliary state sits under an `if`/`elif` whose test names `aux` (Stage B's gating test, extended in Task 13 to the Stage C names). The only legacy-path effects, all output-identical and accepted by ruling:
  - read-only JSON scans for auxiliary evidence (window snapshots at fixed depth, `run_manifest.json`, raw sample-manifest `payload_schema`) in `load_npz`/`load_csv`/`load_parquet`/`load_data`, the union loader and builder, `export_analysis_arrays_npz` and `segment_eligibility` (no file validation, never a new failure mode);
  - `refuse_resume_of_aux_campaign(out_dir, args)` on every `--resume` (a no-op without auxiliary records or a model);
  - `load_production_checkpoint(..., aux_pre_apply=...)` always passed; with the capability off it only checks, inline and without importing `gareus.auxiliary_cv`, that the checkpoint has no `aux` block (the one check that stops a non-aux job from resuming an aux checkpoint);
  - two default-valued CLI keys (`aux_phase_kind`, `aux_equilibrium_eligible`) in the recorded args of every job (same class as rulings C6/C6-ext: provenance only).
- **No zero-fill.**
  - A missing or nonfinite auxiliary observation is never replaced by 0, by the current state's scalar z, or by a placeholder.
  - `gareus.query._concat_numpy_dicts` back-fills a column missing from one segment with masked placeholders (NaN downstream), so every pooling path decides per segment which rows may enter before using them (Task 9 refuses every featureless segment in an auxiliary pool).
  - `mbar_analysis.data.clean()` silently drops non-finite rows (`data.py:249-262`). In the auxiliary path, every non-finite row is either refused or excluded *with a report* before `clean()` runs (Task 9).
- **Use float64 for auxiliary z and torsions in storage and offline assembly** (spec Section 7).
- **What z is stored.** It is the runtime exchange-matrix z (Stage B's observer: fast-path force value or slow-path positions), written as shape (M,) with M = 1 (D8). The z recomputed from the stored torsions is used for parity only.
- **Parity tolerance comes from the recorded platform precision** (D9): 1e-6 reduced energy for `double`, 1e-4 for `mixed`/`single`. The precision is recorded per segment in the sample payload's `runtime` block. Task 15 measures it on GPU (spec Section 17). The runtime parity check never raises for a population without an active auxiliary state (sham-only arms record, Stage B contract).
- **Identity is content.**
  - Models are referenced by the full 64-hex `model_sha256`.
  - The torsion basis has its own `basis_sha256`.
  - Column names index into a recorded ordered list (`aux_z_00`, `tor_000`); the list is the identity.
- **Observation key** = `(run_id, segment_id, carrier_id, absolute_step, observation_phase)`.
  - `carrier_id` is the `replica` column.
  - `observation_phase` is the constant column `"pre_exchange"`.
  - Never join on step alone or on floating-point time.
- **Exchange ledger.** It extends the existing `ParquetExchangeWriter` dataset (`exchanges/<seg>/`). There is no second ledger. Every Gibbs decision writes exactly one event (`swap`/`stay`/`no_candidates`/`skip`).
- **One fixed state table per pooled solve.** Every pooled auxiliary segment's frozen snapshot must be fixed-state compatible (`validate_fixed_state_segments`).
- **Every auxiliary state table carries instance metadata on every row** (ruling M2): `load_aux_state_table` refuses an instance-less table, so a v2 state definition can always be built and a sham can never silently become "ordinary".
- **MBAR solves in tests** go through `gareus.adaptive.mbar_solve.solve_rows` (gareus-analyze `solve_mbar`). No hand-rolled solver (user rule).
- **Test fixtures** (Stage A): `tests/aux_cv_fixture.model_payload(...)` supplies `scale` (default 1.0) and `periodic_imaging` ("none"). Every v2 row built in this plan's tests carries an `instance` block through `tests/aux_c_fixture.py` (Task 2).

## Names consumed from Stage A/B (at HEAD `d4415b1`; reuse, never duplicate)

| Name | Where (HEAD) | Used by |
|---|---|---|
| `AuxStateTable(model, centers, k_kcal, instances)` (`instances` = per-row dict or None), `.n`, `.window_rows()`; `load_aux_state_table(model_path, aux_rows)` | `gareus/auxiliary_cv/state_table.py:59`, `:81` | Tasks 6, 11, 12 |
| `AuxRuntime(table, info, force_index, topology_sha256=None)`; `args._aux_runtime` (per process: reset to None at every `run_gareus` call, `production.py:7126`) | `gareus/auxiliary_cv/runtime.py:49` | Tasks 5, 11, 13 |
| `observe_aux_z`, `check_aux_geometry`, `make_aux_z_observer` (closure `_aux_z_for_replica`, `production.py:7779`), `AuxObservationError` | `runtime.py:147`, `:163`, `:183`, `:26` | Task 13 |
| `aux_snapshot_rows` (D5a fields in legacy snapshot rows), `refuse_aux_population_change(n_expected, n_now, *, cause)`, `canonical_topology_sha256(topology, model)` (+ CLI `python -m gareus.auxiliary_cv.runtime topology-sha PDB MODEL`) | `runtime.py:96`, `:88`, `:113` | Tasks 9, 12, 14 |
| `exchange_energy_version_for_args`, `EXCHANGE_ENERGY_VERSION`, `EXCHANGE_ENERGY_VERSION_AUX`, `is_aux_kernel_record`, `kernel_identity_for_run` (`aux_model_sha256`, `aux_topology_sha256`), `ELIGIBLE_AUX_UNPERSISTED`, `AUX_PERSISTED_SNAPSHOT_KEY` (deleted in Task 7) | `gareus/kernel_identity.py:21-108` | Tasks 7, 9, 12, 13 |
| `refuse_resume_of_aux_campaign(out_dir)` (first statement of `if fast_resume:`), `_latest_segment_kernel_identity(out_dir)`, `verify_kernel_identity_on_resume`, `_aux_window_order_key`, locals `_aux_table`/`_aux_window_key` | `production.py:702`/`:6835`, `:643`, `:660`, `:2382`, `:6829-6830` | Task 12 |
| Stage B engineering WARNING naming `--aux-cv-allow-unpersisted` | `production.py:6901-6903` | Task 12 (deleted) |
| `_refuse_aux_run(prod, meta)` (RuntimeError, pinned by `test_npz_and_csv_loaders_refuse_aux_runs`) | `gareus/mbar_analysis/loaders.py:243`, called at `:269`, `:390` | Task 9 (extended, ruling B3) |
| `_add_aux_cv_args`, `_validate_aux_cv_args`, `--aux-cv-model`, `--aux-cv-allow-unpersisted`, the `--resume`/`--extend` refusal | `gareus/cli.py:1431`, `:1443` | Task 11 |
| Explicit-window CSV aux columns `AUX_CSV_COLUMNS`, `INSTANCE_CSV_COLUMNS` | `gareus/windows.py:57-59` | Task 14 |
| Stage B test helpers `_setup`, `_with_table`, `_degenerate` | `tests/test_aux_cv_observation.py` | Task 13 |

**Stage B tests this plan edits** (each in the task that changes the pinned behaviour): `tests/test_aux_cv_kernel_identity.py` (Tasks 7, 12), `tests/test_aux_cv_production_wiring.py` (Tasks 12, 13), `tests/test_aux_cv_cli.py` (Task 11).

Stage C defines the storage adapter (`aux_io_runtime`, Task 11). Stage B code changes only where a task says so: `runtime.py` none; `state_table.py` (Task 11, ruling M2); `production.py` resume guards (Task 12); `loaders.py` `_refuse_aux_run` (Task 9).

## Review Focus

1. **A historical segment without feature columns, pooled with auxiliary states.** It must fail closed and name the segment, never become silently excluded placeholder-NaN rows. A legacy-kernel segment and a snapshot-less segment refuse with no override; a Stage-B-era auxiliary segment refuses unless explicitly excluded. Task 9 pins all three.
2. **Several Gibbs decisions at one step, including skips.** Every selected carrier yields exactly one event; replay follows `attempt_seq`; duplicates and legacy rows mixed into the ledger raise. Tasks 4, 5, 13 and 14 pin this.
3. **Resume after a crash by exception (auxiliary run).** The parent segment is sealed at the crash step by `finalize_segment`, then re-sealed at the checkpoint on resume. Replay from the ledger anchor reproduces the checkpoint assignments. The resumed run matches the uninterrupted control. Tasks 5, 6, 13 and 14 pin this; the legacy seal is untouched (Task 5 pin).
4. **GPU mixed-precision data.** The parity tolerance comes from the segment's recorded precision, never a hard-coded 1e-6. Tasks 2, 8 and 15 pin this.
5. **NaN aux z in `load_parquet`.** It is refused, or excluded with a report, before `clean()`. The Task 9 test pins this.
6. **Capability mismatch on any resume.** A fast or non-fast `--resume` of an auxiliary campaign without its model (or with another model), and of a non-auxiliary campaign with a model, are refused before any Context is built. Task 12 pins both branches.
7. **Opt-in.** No legacy output or behaviour changes. Tasks 3, 4, 5, 6, 7, 8, 12 and 13 pin it.

---

## File Structure

| File | Responsibility |
|---|---|
| Modify `gareus/parquet_manifest.py` | Preserve an optional `payload_schema` object through validate/append/replace; refuse a schema change within a segment (Task 1) |
| Create `gareus/auxiliary_cv/sample_schema.py` | `AuxSampleSchema`, `build_sample_schema`, `observe_carrier`, `AuxObservation`, runtime precision helpers (Task 2) |
| Modify `gareus/store.py` | `ParquetSampleWriter(aux_schema=, aux_runtime=)` (Task 3); `ParquetExchangeWriter(event_schema=)` + `write_event` (Task 4); `reseal_parent_for_resume` (Task 5); `WindowSnapshot.snapshot(state_definition=...)` (Task 8) |
| Create `gareus/auxiliary_cv/ledger.py` | `assignment_sha256`, `replay_assignments`, `EXCHANGE_EVENT_SCHEMA`, `EVENT_KINDS` (Tasks 4, 5) |
| Create `gareus/auxiliary_cv/checkpoint.py` | `aux_checkpoint_block`, `verify_aux_resume`, `verify_aux_ledger`, `expected_aux_parameters`, `read_aux_parameters`, `topology_identity_sha256`, `aux_table_from_checkpoint` (Task 6) |
| Modify `gareus/kernel_identity.py` | `classify_segment_kernel(..., sample_payload_schema=)`, `AUX_SAMPLES_PAYLOAD_SCHEMA`, `raw_sample_payload_schema`, delete `AUX_PERSISTED_SNAPSHOT_KEY` (Task 7); `snapshot_has_aux`, `aux_snapshot_hits`, `run_has_aux`, `refuse_aux_snapshots`, `AuxPoolingRefused` (Task 9) |
| Modify `gareus/query.py` | `segment_eligibility` (aux-gated payload read, H2) (Task 7); `reconstruct_bias_matrix(aux_z=)` pass-through (Task 8); `export_analysis_arrays_npz` refusal (Task 9) |
| Create `gareus/auxiliary_cv/offline.py` | `parity_context`, `parity_violation`, `aux_z_from_samples`, `exclusion_report`, `registry_model` (Task 8); `segment_aux_schemas`, `segment_aux_runtime`, `classify_pool_segments`, `pool_aux_segments` (Task 9); `merge_identical_hamiltonians` (Task 10) |
| Modify `gareus/correctness/export.py` | Auxiliary evaluation inside `build_export_arrays`, `on_incomplete=`, per-segment parity tolerance, kept/excluded counts (Task 8) |
| Modify `gareus/mbar_analysis/loaders.py` | `load_parquet` auxiliary evaluation; `_refuse_aux_run` extended and hoisted in `load_npz`; `load_data` auto/augment/adaptive guards (Task 9) |
| Modify `gareus/mbar_analysis/loaders_union_parquet.py`, `gareus/adaptive_production.py` | Refuse auxiliary runs, including every pilot dir (Task 9) |
| Create `gareus/auxiliary_cv/runtime_definition.py` | `physical_system_sha256`, `embed_cv_definition`, `build_runtime_state_definition`, `AuxIORuntime`, `aux_io_runtime` (Task 11) |
| Modify `gareus/auxiliary_cv/state_table.py` | Refuse instance-less tables at load (Task 11, ruling M2) |
| Modify `gareus/cli.py` | Remove `--aux-cv-allow-unpersisted` and the resume refusal; add `--aux-phase-kind`, `--aux-equilibrium-eligible` (Task 11) |
| Modify `gareus/production.py` | `save_production_checkpoint(aux_block=)`, `load_production_checkpoint(aux_pre_apply=)` (Task 6); aux-gated resume re-seal (Task 5); resume guards (Task 12); call-site wiring (Task 13) |
| Create `gareus/auxiliary_cv/runtime_io.py` | `check_exchange_boundary_alignment`, `ExchangeEventCounter`, `event_from_gibbs`, `aux_event_fields`, `check_runtime_parity`, `make_aux_record_observer` (Task 13) |
| Create `gareus/auxiliary_cv/resume_check.py`, `gareus/auxiliary_cv/parity_report.py` | Run-directory comparison (Task 14); GPU parity and cost report (Task 15) |
| Tests (new) | `tests/aux_c_fixture.py`, `test_aux_manifest_payload_schema.py`, `test_aux_sample_schema.py`, `test_aux_store_writers.py`, `test_aux_ledger.py`, `test_aux_checkpoint.py`, `test_aux_eligibility.py`, `test_aux_export.py`, `test_aux_window_snapshot.py`, `test_aux_loaders.py`, `test_aux_mbar_duplicates.py`, `test_aux_runtime_definition.py`, `test_aux_resume_guards.py`, `test_aux_runtime_io.py`, `test_aux_resume_check.py`, `test_aux_parity_report.py` |
| Tests (Stage B, edited) | `test_aux_cv_kernel_identity.py`, `test_aux_cv_production_wiring.py`, `test_aux_cv_cli.py` |

---

### Task 1: Parquet manifest carries an optional payload schema

**Files:**
- Modify: `gareus/parquet_manifest.py:111-198` (`validate_manifest`), `:229-258` (`append_file_to_manifest`, `replace_files_in_manifest`)
- Test: `tests/test_aux_manifest_payload_schema.py`

**Interfaces:**
- Produces:
  - `append_file_to_manifest(segment_dir, *, kind, record, next_chunk_index, payload_schema: dict | None = None)`;
  - `replace_files_in_manifest(..., payload_schema: dict | None = None)`;
  - validated manifests contain `"payload_schema"` **only** when it was present. This is needed because `validate_manifest` today rebuilds the dict and drops unknown keys (verified).
- Rules:
  - A `payload_schema` must be a dict with a nonempty string `"schema"`.
  - Appending with a payload schema different from the recorded one raises `ParquetManifestError("payload schema changed ...")`.
  - Appending with `None` to a segment that has one keeps it.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_aux_manifest_payload_schema.py
import json

import pytest

from gareus.parquet_manifest import (ParquetManifestError, append_file_to_manifest, load_manifest,
                                     manifest_path, replace_files_in_manifest)


def _chunk(tmp_path, name="chunk_000001.parquet"):
    import pyarrow as pa
    import pyarrow.parquet as pq
    p = tmp_path / name
    pq.write_table(pa.table({"step": pa.array([1, 2], type=pa.uint64())}), p)
    from gareus.parquet_manifest import file_record
    return file_record(p, rows=2, first_step=1, last_step=2)


def test_legacy_manifest_key_set_is_pinned(tmp_path):
    append_file_to_manifest(tmp_path, kind="samples", record=_chunk(tmp_path), next_chunk_index=2)
    raw = json.loads(manifest_path(tmp_path).read_text())
    assert "payload_schema" not in raw
    assert set(raw) == {"schema", "kind", "generation", "n_rows", "next_chunk_index", "files"}
    assert set(raw["files"][0]) == {"first_step", "last_step", "path", "rows", "sha256", "size_bytes"}


def test_payload_schema_survives_append_and_replace(tmp_path):
    ps = {"schema": "atlas-aux-samples-v1", "basis_sha256": "a" * 64}
    append_file_to_manifest(tmp_path, kind="samples", record=_chunk(tmp_path), next_chunk_index=2,
                            payload_schema=ps)
    append_file_to_manifest(tmp_path, kind="samples", record=_chunk(tmp_path, "chunk_000002.parquet"),
                            next_chunk_index=3)
    assert load_manifest(tmp_path)["payload_schema"] == ps
    rec = _chunk(tmp_path, "data.parquet")
    replace_files_in_manifest(tmp_path, kind="samples", records=[rec], next_chunk_index=3)
    assert load_manifest(tmp_path)["payload_schema"] == ps


def test_payload_schema_change_is_refused(tmp_path):
    append_file_to_manifest(tmp_path, kind="samples", record=_chunk(tmp_path), next_chunk_index=2,
                            payload_schema={"schema": "atlas-aux-samples-v1", "basis_sha256": "a" * 64})
    with pytest.raises(ParquetManifestError, match="payload schema changed"):
        append_file_to_manifest(tmp_path, kind="samples", record=_chunk(tmp_path, "chunk_000002.parquet"),
                                next_chunk_index=3,
                                payload_schema={"schema": "atlas-aux-samples-v1", "basis_sha256": "b" * 64})


def test_payload_schema_added_to_populated_legacy_segment_is_refused(tmp_path):
    append_file_to_manifest(tmp_path, kind="samples", record=_chunk(tmp_path), next_chunk_index=2)
    with pytest.raises(ParquetManifestError, match="payload schema changed"):
        append_file_to_manifest(tmp_path, kind="samples", record=_chunk(tmp_path, "chunk_000002.parquet"),
                                next_chunk_index=3, payload_schema={"schema": "atlas-aux-samples-v1"})


@pytest.mark.parametrize("bad", [[], {"schema": ""}, {"no_schema": 1}])
def test_malformed_payload_schema_rejected_on_load(tmp_path, bad):
    append_file_to_manifest(tmp_path, kind="samples", record=_chunk(tmp_path), next_chunk_index=2)
    raw = json.loads(manifest_path(tmp_path).read_text())
    raw["payload_schema"] = bad
    manifest_path(tmp_path).write_text(json.dumps(raw))
    with pytest.raises(ParquetManifestError, match="payload_schema"):
        load_manifest(tmp_path)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `tests/test_aux_manifest_payload_schema.py`
Expected: FAIL. `test_payload_schema_survives_append_and_replace` raises `TypeError: ... unexpected keyword argument 'payload_schema'`. `test_legacy_manifest_key_set_is_pinned` passes, as a regression guard.

- [ ] **Step 3: Implement**

In `validate_manifest`, replace the final `return {...}` with:

```python
    out = {
        "schema": SCHEMA,
        "kind": kind,
        "generation": generation,
        "n_rows": n_rows,
        "next_chunk_index": next_chunk_index,
        "files": normalized,
    }
    if "payload_schema" in manifest:
        payload_schema = manifest["payload_schema"]
        if (not isinstance(payload_schema, dict) or not isinstance(payload_schema.get("schema"), str)
                or not payload_schema["schema"]):
            raise ParquetManifestError("Parquet manifest payload_schema must be an object with a nonempty 'schema'")
        out["payload_schema"] = payload_schema
    return out
```

Add a helper above `append_file_to_manifest`:

```python
def _merge_payload_schema(current: dict[str, Any], requested: Optional[dict[str, Any]]) -> dict[str, Any]:
    new_manifest = dict(current)
    if requested is None:
        return new_manifest
    recorded = current.get("payload_schema")
    if recorded is None and current["files"]:
        raise ParquetManifestError("payload schema changed: segment already holds rows without one")
    if recorded is not None and recorded != requested:
        raise ParquetManifestError(f"payload schema changed within a segment: {recorded!r} -> {requested!r}")
    new_manifest["payload_schema"] = dict(requested)
    return new_manifest
```

Change `append_file_to_manifest` to:

```python
def append_file_to_manifest(segment_dir: Path, *, kind: str, record: dict[str, Any], next_chunk_index: int,
                            payload_schema: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    current = load_manifest(segment_dir, expected_kind=kind, verify_hashes=False)
    if current is None:
        current = empty_manifest(kind)
    new_manifest = _merge_payload_schema(current, payload_schema)
    new_manifest["generation"] = int(current["generation"]) + 1
    new_manifest["files"] = [dict(x) for x in current["files"]] + [dict(record)]
    new_manifest["n_rows"] = int(current["n_rows"]) + int(record["rows"])
    new_manifest["next_chunk_index"] = int(next_chunk_index)
    return publish_manifest(segment_dir, new_manifest)
```

Give `replace_files_in_manifest` the same keyword. Replace `new_manifest = dict(current)` with `new_manifest = _merge_payload_schema(current, payload_schema)`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `tests/test_aux_manifest_payload_schema.py tests/test_parquet_manifest_transactions.py tests/test_store.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add gareus/parquet_manifest.py tests/test_aux_manifest_payload_schema.py
git commit -m "feat(cvaux): Parquet manifests carry an optional immutable payload schema"
```

---

### Task 2: Auxiliary sample schema, carrier observation, runtime precision; shared test fixture

**Files:**
- Create: `gareus/auxiliary_cv/sample_schema.py`
- Create: `tests/aux_c_fixture.py`
- Modify: `gareus/auxiliary_cv/__init__.py`
- Test: `tests/test_aux_sample_schema.py`

**Interfaces:**
- Consumes: Stage A `AuxModel`, `openmm_dihedrals`, `unique_torsions`, `z_from_dihedrals`; `gareus.mbar_analysis.thermo_frames.backbone_torsion_quads`; `tests/aux_cv_fixture.py` (`dipeptide`, `model_payload`).
- Produces:
  - `AUX_SAMPLES_SCHEMA = "atlas-aux-samples-v1"`, `TORSION_CONVENTION = "openmm_theta_radians"`;
  - `PRECISIONS = ("double", "mixed", "single")`, `PARITY_TOLERANCE = {"double": 1e-6, "mixed": 1e-4, "single": 1e-4}`;
  - `@dataclass(frozen=True) class AuxSampleSchema(torsion_quads, torsion_labels, model_shas, basis_sha256)`, with `.to_payload()`, `.from_payload(d)` (ignores a `runtime` key), `.torsion_columns`, `.z_columns`, `.model_basis_index(model)`;
  - `build_sample_schema(topology, peptide_atoms, models) -> AuxSampleSchema`;
  - `@dataclass(frozen=True) class AuxObservation(torsions: np.ndarray, z: np.ndarray)`;
  - `observe_carrier(xyz_nm, schema, models) -> AuxObservation`;
  - `runtime_precision(platform_name: str, platform=None, context=None) -> str`;
  - `runtime_info(platform_name, precision) -> dict`;
  - `payload_with_runtime(schema, info) -> dict`;
  - `runtime_from_payload(payload) -> dict | None`.
- `tests/aux_c_fixture.py` produces:
  - `instance(i, role, *, parent=None, slot=None, obs=None) -> dict`, a valid Stage A instance block;
  - `rows(spec) -> list[dict]`, where each spec item is `(center1, k1, aux_k, aux_center, role)` and every row carries an instance;
  - `definition(rows, model, *, cv2=None) -> dict`, which builds a v2 definition with model registry `{sha: model.to_mapping()}`. Stage A canonicalises the registry to the identity body (D2).
- Rules:
  - The torsion basis is **every** backbone phi/psi in `backbone_torsion_quads` order.
  - Each model's torsions must lie in the basis.
  - At most 1 model (MVP).
  - Angles are stored in the OpenMM θ convention, float64.
  - A degenerate torsion gives NaN, which is kept.
  - Precision: `Reference` → `double`. `CPU` → `mixed` (conservative: CPU platform force kernels are single precision). CUDA/OpenCL/HIP → the platform's `Precision` property (`double|mixed|single`). Anything else → `single`.

- [ ] **Step 1: Write the shared fixture**

```python
# tests/aux_c_fixture.py
"""Stage C test builders: v2 rows with instance blocks (Stage A D3) and definitions."""
from __future__ import annotations

from gareus.correctness.state_identity import make_state_definition

BOX = [[3, 0, 0], [0, 3, 0], [0, 0, 3]]
CV1 = {"kind": "contacts", "units": "dimensionless", "definition": {"r0": 4.5}}


def instance(i, role, *, parent=None, slot=None, obs=None):
    return {"state_instance_id": f"s{i}", "state_role": role, "spawn_parent_state_id": parent,
            "spawn_source_observation": obs, "matched_additional_slot_id": slot}


def rows(spec, model_sha=None):
    out = []
    for i, (c1, k1, aux_k, aux_c, role) in enumerate(spec):
        row = {"window_id": i, "center1": c1, "k1": k1, "center2": 0.0, "k2": 0.0, "gamd_lambda": 0.0,
               "aux_k": aux_k, "instance": instance(i, role, parent="s0" if role != "ordinary" else None,
                                                    slot="slot-0" if role != "ordinary" else None)}
        if aux_k > 0:
            row.update(aux_model_sha256=model_sha, aux_center=aux_c)
        out.append(row)
    return out


def definition(row_list, model, *, cv2=None):
    return make_state_definition(row_list, physical_system_sha256="a" * 64, ensemble="NVT", temperature_k=300.0,
                                 fixed_box_vectors_nm=BOX, cv1=CV1, cv2=cv2,
                                 aux_models={model.model_sha256: model.to_mapping()})
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_aux_sample_schema.py
import numpy as np
import pytest

from aux_cv_fixture import dipeptide, model_payload
from gareus.auxiliary_cv.evaluate import z_from_positions
from gareus.auxiliary_cv.model import AuxModel
from gareus.auxiliary_cv.sample_schema import (AUX_SAMPLES_SCHEMA, PARITY_TOLERANCE, AuxSampleSchema,
                                               build_sample_schema, observe_carrier, payload_with_runtime,
                                               runtime_from_payload, runtime_info, runtime_precision)
from gareus.correctness._io import IntegrityError


def _model(d, which):
    blocks = [lab.split("-")[0] for lab in d["labels"]]
    coeffs = np.zeros(2 * len(d["quads"]))
    coeffs[which] = 1.0
    return AuxModel.from_mapping(model_payload(d["quads"], coeffs, offset=0.1, blocks=blocks))


def _peptide_atoms(d):
    return sorted({a for q in d["quads"] for a in q})


def test_schema_round_trip_and_identity():
    d = dipeptide()
    m = _model(d, 0)
    s = build_sample_schema(d["topology"], _peptide_atoms(d), [m])
    again = AuxSampleSchema.from_payload(s.to_payload())
    assert again == s and s.to_payload()["schema"] == AUX_SAMPLES_SCHEMA
    assert s.model_shas == (m.model_sha256,)
    assert s.torsion_columns == tuple(f"tor_{i:03d}" for i in range(len(s.torsion_quads)))
    assert s.z_columns == ("aux_z_00",)


def test_runtime_block_is_not_schema_identity():
    d = dipeptide()
    s = build_sample_schema(d["topology"], _peptide_atoms(d), [_model(d, 0)])
    p = payload_with_runtime(s, runtime_info("CUDA", "mixed"))
    assert AuxSampleSchema.from_payload(p) == s
    assert runtime_from_payload(p) == {"platform": "CUDA", "precision": "mixed"}
    assert runtime_from_payload(s.to_payload()) is None
    with pytest.raises(IntegrityError, match="precision"):
        runtime_info("CUDA", "quad")


def test_basis_sha_changes_with_basis():
    d = dipeptide()
    s = build_sample_schema(d["topology"], _peptide_atoms(d), [_model(d, 0)])
    payload = s.to_payload()
    payload["torsion_labels"] = list(reversed(payload["torsion_labels"]))
    with pytest.raises(IntegrityError, match="basis_sha256"):
        AuxSampleSchema.from_payload(payload)


def test_observation_matches_stage_a_evaluator():
    d = dipeptide()
    m = _model(d, 3)
    s = build_sample_schema(d["topology"], _peptide_atoms(d), [m])
    obs = observe_carrier(d["positions_nm"], s, {m.model_sha256: m})
    assert obs.torsions.dtype == np.float64 and obs.z.dtype == np.float64
    assert obs.torsions.shape == (len(s.torsion_quads),) and obs.z.shape == (1,)
    assert obs.z[0] == z_from_positions(d["positions_nm"], m)[0]


def test_model_outside_basis_is_refused():
    d = dipeptide()
    other = AuxModel.from_mapping(model_payload([(0, 1, 2, 3)], [1.0, 0.0]))
    with pytest.raises(IntegrityError, match="basis"):
        build_sample_schema(d["topology"], _peptide_atoms(d), [other])


def test_more_than_one_model_is_stage_f():
    d = dipeptide()
    with pytest.raises(IntegrityError, match="Stage F"):
        build_sample_schema(d["topology"], _peptide_atoms(d), [_model(d, 0), _model(d, 1)])


def test_observation_requires_every_schema_model():
    d = dipeptide()
    m = _model(d, 0)
    s = build_sample_schema(d["topology"], _peptide_atoms(d), [m])
    with pytest.raises(IntegrityError, match="model"):
        observe_carrier(d["positions_nm"], s, {})


def test_runtime_precision_of_reference_and_cpu_contexts():
    import openmm as mm
    s = mm.System(); s.addParticle(1.0)
    for name, expected in (("Reference", "double"), ("CPU", "mixed")):
        plat = mm.Platform.getPlatformByName(name)
        ctx = mm.Context(s, mm.VerletIntegrator(0.001), plat)
        assert runtime_precision(plat.getName(), plat, ctx) == expected
    assert runtime_precision("SomethingElse") == "single"
    assert PARITY_TOLERANCE == {"double": 1e-6, "mixed": 1e-4, "single": 1e-4}
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `tests/test_aux_sample_schema.py`
Expected: FAIL `ModuleNotFoundError: No module named 'gareus.auxiliary_cv.sample_schema'`

- [ ] **Step 4: Implement**

```python
# gareus/auxiliary_cv/sample_schema.py
"""What every auxiliary-capable sample row stores: the full ordered backbone torsion basis
(OpenMM theta, float64) and z of every phase model (float64), schema ``atlas-aux-samples-v1``.

The basis is ALL backbone phi/psi, so a model fitted later can still be evaluated (spec Section 7).
Column names (tor_000, aux_z_00) are positions in the recorded lists; the lists and basis_sha256
are the identity. The payload's optional ``runtime`` block (platform, precision) is per segment and
selects the parity tolerance; it is not part of the schema identity.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from ..correctness._io import IntegrityError, digest, json_bytes
from .evaluate import z_from_dihedrals
from .features import openmm_dihedrals, unique_torsions
from .model import AuxModel

AUX_SAMPLES_SCHEMA = "atlas-aux-samples-v1"
TORSION_CONVENTION = "openmm_theta_radians"
PRECISIONS = ("double", "mixed", "single")
#: Reduced-energy parity tolerance between stored runtime z and offline z (spec Section 17).
PARITY_TOLERANCE = {"double": 1e-6, "mixed": 1e-4, "single": 1e-4}


def _basis_sha(quads, labels) -> str:
    return digest(json_bytes({"convention": TORSION_CONVENTION, "quads": [list(q) for q in quads],
                              "labels": list(labels)}))


@dataclass(frozen=True)
class AuxSampleSchema:
    torsion_quads: tuple[tuple[int, int, int, int], ...]
    torsion_labels: tuple[str, ...]
    model_shas: tuple[str, ...]
    basis_sha256: str

    @property
    def torsion_columns(self) -> tuple[str, ...]:
        return tuple(f"tor_{i:03d}" for i in range(len(self.torsion_quads)))

    @property
    def z_columns(self) -> tuple[str, ...]:
        return tuple(f"aux_z_{i:02d}" for i in range(len(self.model_shas)))

    def to_payload(self) -> dict[str, Any]:
        return {"schema": AUX_SAMPLES_SCHEMA, "torsion_convention": TORSION_CONVENTION,
                "torsion_quads": [list(q) for q in self.torsion_quads],
                "torsion_labels": list(self.torsion_labels),
                "model_shas": list(self.model_shas), "basis_sha256": self.basis_sha256}

    @classmethod
    def from_payload(cls, raw: Mapping[str, Any]) -> "AuxSampleSchema":
        if raw.get("schema") != AUX_SAMPLES_SCHEMA or raw.get("torsion_convention") != TORSION_CONVENTION:
            raise IntegrityError(f"aux sample schema must be {AUX_SAMPLES_SCHEMA} / {TORSION_CONVENTION}")
        quads = tuple(tuple(int(a) for a in q) for q in raw["torsion_quads"])
        labels = tuple(str(x) for x in raw["torsion_labels"])
        if len(quads) != len(labels) or any(len(q) != 4 for q in quads):
            raise IntegrityError("aux sample schema torsion_quads/labels are inconsistent")
        shas = tuple(str(x) for x in raw["model_shas"])
        if any(len(s) != 64 for s in shas):
            raise IntegrityError("aux sample schema model_shas must be full sha256 digests")
        expected = _basis_sha(quads, labels)
        if raw.get("basis_sha256") != expected:
            raise IntegrityError(f"aux sample schema basis_sha256 {raw.get('basis_sha256')} != content {expected}")
        return cls(quads, labels, shas, expected)

    def model_basis_index(self, model: AuxModel) -> np.ndarray:
        where = {q: i for i, q in enumerate(self.torsion_quads)}
        quads, _idx = unique_torsions(model)
        missing = [q for q in quads if q not in where]
        if missing:
            raise IntegrityError(f"aux model {model.model_sha256} uses torsions outside the recorded basis: {missing}")
        return np.asarray([where[q] for q in quads], dtype=np.int64)


def build_sample_schema(topology, peptide_atoms, models: Sequence[AuxModel]) -> AuxSampleSchema:
    from ..mbar_analysis.thermo_frames import backbone_torsion_quads
    shas: list[str] = []
    for m in models:
        if m.model_sha256 not in shas:
            shas.append(m.model_sha256)
    if len(shas) > 1:
        raise IntegrityError("More than one auxiliary model per phase is Stage F (spec Section 16); MVP supports one")
    quads, labels = backbone_torsion_quads(topology, peptide_atoms)
    quads = tuple(tuple(int(a) for a in q) for q in quads)
    schema = AuxSampleSchema(quads, tuple(labels), tuple(shas), _basis_sha(quads, labels))
    for m in models:
        schema.model_basis_index(m)
    return schema


@dataclass(frozen=True)
class AuxObservation:
    torsions: np.ndarray
    z: np.ndarray


def observe_carrier(xyz_nm, schema: AuxSampleSchema, models: Mapping[str, AuxModel]) -> AuxObservation:
    theta = openmm_dihedrals(xyz_nm, schema.torsion_quads)[0].astype(np.float64)
    z = np.empty(len(schema.model_shas), dtype=np.float64)
    for i, sha in enumerate(schema.model_shas):
        if sha not in models:
            raise IntegrityError(f"no loaded aux model for schema model {sha}")
        model = models[sha]
        z[i] = z_from_dihedrals(theta[schema.model_basis_index(model)][None, :], model)[0]
    return AuxObservation(theta, z)


def runtime_precision(platform_name: str, platform=None, context=None) -> str:
    name = str(platform_name)
    if name == "Reference":
        return "double"
    if name == "CPU":
        return "mixed"
    if platform is not None and context is not None:
        try:
            value = str(platform.getPropertyValue(context, "Precision")).strip().lower()
        except Exception:
            value = ""
        if value in PRECISIONS:
            return value
    return "single"


def runtime_info(platform_name: str, precision: str) -> dict[str, str]:
    if precision not in PRECISIONS:
        raise IntegrityError(f"runtime precision must be one of {PRECISIONS}, got {precision!r}")
    return {"platform": str(platform_name), "precision": str(precision)}


def payload_with_runtime(schema: AuxSampleSchema, info: Mapping[str, str]) -> dict[str, Any]:
    payload = schema.to_payload()
    payload["runtime"] = runtime_info(info["platform"], info["precision"])
    return payload


def runtime_from_payload(payload: Mapping[str, Any]) -> Optional[dict[str, str]]:
    block = payload.get("runtime")
    if block is None:
        return None
    return runtime_info(block.get("platform", ""), block.get("precision", ""))
```

Add to `gareus/auxiliary_cv/__init__.py`: `from .sample_schema import AUX_SAMPLES_SCHEMA, AuxObservation, AuxSampleSchema, build_sample_schema, observe_carrier`, and extend `__all__`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `tests/test_aux_sample_schema.py tests/test_aux_cv_evaluate.py`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add gareus/auxiliary_cv/sample_schema.py gareus/auxiliary_cv/__init__.py tests/aux_c_fixture.py tests/test_aux_sample_schema.py
git commit -m "feat(cvaux): auxiliary sample schema, carrier observation and runtime precision"
```

---

### Task 3: Sample writer records torsions, runtime z and runtime precision; legacy schema pinned

**Files:**
- Modify: `gareus/store.py:29-159` (`ParquetSampleWriter`)
- Test: `tests/test_aux_store_writers.py` (sample part)

**Interfaces:**
- Consumes: Task 1 manifest keywords; Task 2 `AuxSampleSchema`, `payload_with_runtime`.
- Produces:
  - `ParquetSampleWriter(out_dir, flush_rows=5000, aux_schema=None, aux_runtime: Mapping[str, str] | None = None)`. With `aux_schema`, `aux_runtime` (`{"platform", "precision"}`) is required.
  - `write_sample(..., gamd_lambda=0.0, aux_z: Sequence[float] | None = None, torsions: Sequence[float] | None = None)`.
- Behaviour with `aux_schema`:
  - Each row gets float64 `tor_###` and `aux_z_##` columns and the string column `observation_phase="pre_exchange"`.
  - The table metadata `b"atlas_aux_samples"` and the manifest `payload_schema` both hold `payload_with_runtime(schema, aux_runtime)`.
  - Missing or wrong-length values raise `ValueError` before buffering.
  - Nonfinite values are stored as-is.
- Behaviour without `aux_schema`: passing `aux_z`/`torsions` raises; the schema and dtypes equal the pinned legacy string.
- Construction-time refusals:
  - a mismatched recorded `payload_schema`;
  - an aux writer on a populated legacy segment;
  - a legacy writer on an aux segment.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_aux_store_writers.py
import json

import numpy as np
import pytest

from gareus.auxiliary_cv.sample_schema import AuxSampleSchema, _basis_sha, runtime_from_payload
from gareus.parquet_manifest import ParquetManifestError, load_manifest

#: Pinned on main dc30285 (tmp/verify_C/pins.py): legacy sample schema with dtypes.
LEGACY_SAMPLE_SCHEMA = ("step: uint64\nreplica: uint16\nwindow_id: uint16\ncv1: float\ncv2: float\n"
                        "potential: float\ngamd_boost_total: float\ngamd_boost_dihedral: float\n"
                        "gamd_boost_nonbonded: float\nv_pep_kj_mol: float\nv_dih_kj_mol: float\ngamd_lambda: float")
LEGACY_COLUMNS = [line.split(":")[0] for line in LEGACY_SAMPLE_SCHEMA.split("\n")]
QUADS = ((0, 1, 2, 3), (1, 2, 3, 4), (2, 3, 4, 5))
LABELS = ("phi-A1", "psi-A1", "phi-B2")
SCHEMA = AuxSampleSchema(QUADS, LABELS, ("e" * 64,), _basis_sha(QUADS, LABELS))
RUNTIME = {"platform": "Reference", "precision": "double"}


def _row(step, replica=0):
    return dict(step=step, replica=replica, window_id=replica, cv1=0.1, cv2=None, potential=-1.0,
                boost_total=None, boost_dihedral=None, boost_nonbonded=None)


def test_legacy_sample_schema_and_dtypes_pinned(tmp_path):
    import pyarrow.parquet as pq
    from gareus.store import ParquetSampleWriter
    w = ParquetSampleWriter(tmp_path)
    w.write_sample(**_row(10))
    w.close()
    tbl = pq.read_table(tmp_path / "data.parquet")
    assert tbl.schema.to_string(show_schema_metadata=False) == LEGACY_SAMPLE_SCHEMA
    assert tbl.schema.metadata is None or b"atlas_aux_samples" not in tbl.schema.metadata
    assert "payload_schema" not in load_manifest(tmp_path)


def test_legacy_writer_refuses_aux_values(tmp_path):
    from gareus.store import ParquetSampleWriter
    w = ParquetSampleWriter(tmp_path)
    with pytest.raises(ValueError, match="aux"):
        w.write_sample(**_row(10), aux_z=[1.0], torsions=[0.0, 0.0, 0.0])


def test_aux_writer_requires_runtime(tmp_path):
    from gareus.store import ParquetSampleWriter
    with pytest.raises(ValueError, match="aux_runtime"):
        ParquetSampleWriter(tmp_path, aux_schema=SCHEMA)


def test_aux_columns_float64_metadata_runtime_survive_consolidation(tmp_path):
    import pyarrow.parquet as pq
    from gareus.store import ParquetSampleWriter
    w = ParquetSampleWriter(tmp_path, flush_rows=1, aux_schema=SCHEMA, aux_runtime=RUNTIME)
    w.write_sample(**_row(10), aux_z=[0.123456789012345], torsions=[0.1, -3.1, np.nan])
    w.write_sample(**_row(20, 1), aux_z=[1.5], torsions=[0.2, 3.1, 1.0])
    w.close()
    tbl = pq.read_table(tmp_path / "data.parquet")
    assert tbl.column_names == LEGACY_COLUMNS + ["observation_phase", "tor_000", "tor_001", "tor_002", "aux_z_00"]
    assert str(tbl.schema.field("aux_z_00").type) == "double" and str(tbl.schema.field("tor_000").type) == "double"
    assert tbl.column("aux_z_00").to_pylist()[0] == 0.123456789012345
    assert np.isnan(tbl.column("tor_002").to_pylist()[0])
    assert set(tbl.column("observation_phase").to_pylist()) == {"pre_exchange"}
    meta = json.loads(tbl.schema.metadata[b"atlas_aux_samples"])
    assert AuxSampleSchema.from_payload(meta) == SCHEMA and runtime_from_payload(meta) == RUNTIME
    payload = load_manifest(tmp_path)["payload_schema"]
    assert AuxSampleSchema.from_payload(payload) == SCHEMA and runtime_from_payload(payload) == RUNTIME


@pytest.mark.parametrize("kw", [dict(aux_z=None, torsions=[0, 0, 0]), dict(aux_z=[1.0], torsions=None),
                                dict(aux_z=[1.0, 2.0], torsions=[0, 0, 0]), dict(aux_z=[1.0], torsions=[0, 0])])
def test_aux_writer_requires_complete_rows(tmp_path, kw):
    from gareus.store import ParquetSampleWriter
    w = ParquetSampleWriter(tmp_path, aux_schema=SCHEMA, aux_runtime=RUNTIME)
    with pytest.raises(ValueError, match="aux"):
        w.write_sample(**_row(10), **kw)


def test_aux_writer_refuses_populated_legacy_segment(tmp_path):
    from gareus.store import ParquetSampleWriter
    w = ParquetSampleWriter(tmp_path)
    w.write_sample(**_row(10))
    w.flush()
    with pytest.raises(ParquetManifestError, match="payload schema"):
        ParquetSampleWriter(tmp_path, aux_schema=SCHEMA, aux_runtime=RUNTIME)


def test_legacy_writer_refuses_aux_segment(tmp_path):
    from gareus.store import ParquetSampleWriter
    w = ParquetSampleWriter(tmp_path, aux_schema=SCHEMA, aux_runtime=RUNTIME)
    w.write_sample(**_row(10), aux_z=[1.0], torsions=[0.0, 0.0, 0.0])
    w.flush()
    with pytest.raises(ParquetManifestError, match="payload schema"):
        ParquetSampleWriter(tmp_path)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `tests/test_aux_store_writers.py`
Expected: FAIL `TypeError: ... unexpected keyword argument 'aux_schema'`. `test_legacy_sample_schema_and_dtypes_pinned` passes, as a regression guard.

- [ ] **Step 3: Implement in `ParquetSampleWriter`**

Constructor. Keep the existing lines and add the auxiliary block:

```python
    def __init__(self, out_dir: Path, flush_rows: int = 5000, aux_schema=None, aux_runtime=None) -> None:
        self._out_dir = Path(out_dir)
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._flush_rows = max(1, flush_rows)
        self._buf: Dict[str, list] = defaultdict(list)
        self._manifest = load_manifest(self._out_dir, expected_kind="samples")
        if self._manifest is None and list(self._out_dir.glob("*.parquet")):
            raise ParquetManifestError(f"cannot append to legacy Parquet segment without manifest: {self._out_dir}")
        self._chunk_idx = (self._manifest["next_chunk_index"] - 1) if self._manifest else 0
        self._aux_schema = aux_schema
        self._aux_payload = None
        if aux_schema is not None:
            if aux_runtime is None:
                raise ValueError("an aux sample writer needs aux_runtime={'platform', 'precision'}")
            from .auxiliary_cv.sample_schema import payload_with_runtime
            self._aux_payload = payload_with_runtime(aux_schema, aux_runtime)
        recorded = (self._manifest or {}).get("payload_schema")
        populated = bool(self._manifest and self._manifest["files"])
        if populated and recorded != self._aux_payload:
            raise ParquetManifestError(
                f"payload schema of {self._out_dir} is {recorded!r}; this writer would write {self._aux_payload!r}")
```

In `write_sample`, add the keyword parameters `aux_z=None, torsions=None` and insert at the top of the body:

```python
        if self._aux_schema is None:
            if aux_z is not None or torsions is not None:
                raise ValueError("aux_z/torsions given to a writer without an aux sample schema")
        else:
            nz, nt = len(self._aux_schema.z_columns), len(self._aux_schema.torsion_columns)
            if aux_z is None or torsions is None or len(aux_z) != nz or len(torsions) != nt:
                raise ValueError(f"aux writer needs aux_z ({nz}) and torsions ({nt}) on every sample")
```

After the existing appends, before the flush check:

```python
        if self._aux_schema is not None:
            b["observation_phase"].append("pre_exchange")
            for name, value in zip(self._aux_schema.torsion_columns, torsions):
                b[name].append(float(value))
            for name, value in zip(self._aux_schema.z_columns, aux_z):
                b[name].append(float(value))
```

In `flush`, after `tbl = pa.table({...})`:

```python
        if self._aux_schema is not None:
            extra = {"observation_phase": pa.array(b["observation_phase"], type=pa.string())}
            for name in self._aux_schema.torsion_columns + self._aux_schema.z_columns:
                extra[name] = pa.array(b[name], type=pa.float64())
            for name, column in extra.items():
                tbl = tbl.append_column(name, column)
            tbl = tbl.replace_schema_metadata({b"atlas_aux_samples": json.dumps(self._aux_payload, sort_keys=True).encode()})
```

Changes elsewhere:
- Pass `payload_schema=self._aux_payload` to `append_file_to_manifest(...)` in `flush`.
- In `_consolidate`, after `tbl = tbl.sort_by(...)`, re-attach the metadata when `self._aux_payload is not None` (same `replace_schema_metadata` line), and pass `payload_schema=self._aux_payload` to `replace_files_in_manifest(...)`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `tests/test_aux_store_writers.py tests/test_store.py tests/test_parquet_manifest_transactions.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add gareus/store.py tests/test_aux_store_writers.py
git commit -m "feat(cvaux): sample writer records torsions, runtime z and platform precision; legacy schema pinned"
```

---

### Task 4: Ordered exchange event ledger

**Files:**
- Create: `gareus/auxiliary_cv/ledger.py` (schema, kinds, checksum; replay in Task 5)
- Modify: `gareus/store.py:162-273` (`ParquetExchangeWriter`)
- Test: `tests/test_aux_store_writers.py` (exchange part)

**Interfaces:**
- Produces:
  - `EXCHANGE_EVENT_SCHEMA = "atlas-exchange-events-v1"`;
  - `EVENT_KINDS = ("swap", "stay", "no_candidates", "skip")`. `skip` is a Gibbs proposal that the swap kernel declined to attempt (`swap_candidate_replicas(...) is None` or `apply_window_swap(...) is None`, `production.py:8548-8565`);
  - `assignment_sha256(assignments) -> str`;
  - `ParquetExchangeWriter(out_dir, flush_rows=1000, event_schema: str | None = None)`;
  - `write_event(*, step, attempt_seq, selected_replica, replica_i, replica_j, window_i, window_j, kind, delta_e_kj, accepted, log_q_forward, log_q_reverse, p_accept, energy_version, assignments_after)`.
- Event-mode columns are appended after the 7 legacy ones:

  | Column | Type |
  |---|---|
  | `attempt_seq` | uint32 |
  | `selected_replica` | int32 |
  | `kind` | string |
  | `delta_e_kj` | float64 |
  | `log_q_forward` | float64 |
  | `log_q_reverse` | float64 |
  | `p_accept` | float64 |
  | `energy_version` | string |
  | `assignment_sha256_after` | string |

  The legacy `delta_e` stays filled. `write_exchange` raises in event mode. The manifest records `payload_schema={"schema": EXCHANGE_EVENT_SCHEMA}`.
- Legacy mode is unchanged (schema pinned), and `write_event` raises there.
- Field rules:
  - Non-`swap` kinds have `accepted=False`, `replica_i = replica_j = selected` and `window_i = window_j = current`.
  - `assignments_after` is the assignment after the decision.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_aux_store_writers.py`)

```python
from gareus.auxiliary_cv.ledger import EVENT_KINDS, EXCHANGE_EVENT_SCHEMA, assignment_sha256

#: Pinned on main dc30285 (tmp/verify_C/pins.py).
LEGACY_EXCHANGE_SCHEMA = ("step: uint64\nreplica_i: uint16\nreplica_j: uint16\nwindow_i: uint16\n"
                          "window_j: uint16\ndelta_e: float\naccepted: bool")
LEGACY_EXCHANGE_COLUMNS = [line.split(":")[0] for line in LEGACY_EXCHANGE_SCHEMA.split("\n")]


def _event(step, seq, **kw):
    base = dict(step=step, attempt_seq=seq, selected_replica=0, replica_i=0, replica_j=1, window_i=0,
                window_j=1, kind="swap", delta_e_kj=-0.25, accepted=True, log_q_forward=np.log(0.4),
                log_q_reverse=np.log(0.3), p_accept=0.9, energy_version="state_bias_matrix_v3_aux",
                assignments_after=[1, 0, 2])
    base.update(kw)
    return base


def test_legacy_exchange_schema_pinned(tmp_path):
    import pyarrow.parquet as pq
    from gareus.store import ParquetExchangeWriter
    w = ParquetExchangeWriter(tmp_path)
    w.write_exchange(step=5, replica_i=0, replica_j=1, window_i=0, window_j=1, delta_e=0.5, accepted=False)
    w.close()
    assert pq.read_table(tmp_path / "data.parquet").schema.to_string(show_schema_metadata=False) == LEGACY_EXCHANGE_SCHEMA
    assert "payload_schema" not in load_manifest(tmp_path)
    with pytest.raises(ValueError, match="event"):
        w.write_event(**_event(6, 0))


def test_event_rows_round_trip(tmp_path):
    import pyarrow.parquet as pq
    from gareus.store import ParquetExchangeWriter
    w = ParquetExchangeWriter(tmp_path, event_schema=EXCHANGE_EVENT_SCHEMA)
    w.write_event(**_event(100, 0))
    w.write_event(**_event(100, 1, kind="skip", selected_replica=2, replica_i=2, replica_j=2, window_i=2,
                           window_j=2, accepted=False, assignments_after=[1, 0, 2]))
    w.close()
    t = pq.read_table(tmp_path / "data.parquet")
    assert t.column_names == LEGACY_EXCHANGE_COLUMNS + [
        "attempt_seq", "selected_replica", "kind", "delta_e_kj", "log_q_forward", "log_q_reverse",
        "p_accept", "energy_version", "assignment_sha256_after"]
    assert t.column("attempt_seq").to_pylist() == [0, 1] and t.column("kind").to_pylist() == ["swap", "skip"]
    assert t.column("assignment_sha256_after").to_pylist()[0] == assignment_sha256([1, 0, 2])
    assert load_manifest(tmp_path)["payload_schema"] == {"schema": EXCHANGE_EVENT_SCHEMA}
    with pytest.raises(ValueError, match="write_event"):
        w.write_exchange(step=5, replica_i=0, replica_j=1, window_i=0, window_j=1, delta_e=0.5, accepted=False)
    assert EVENT_KINDS == ("swap", "stay", "no_candidates", "skip")


@pytest.mark.parametrize("bad", [dict(kind="teleport"), dict(attempt_seq=-1),
                                 dict(kind="stay", accepted=True), dict(kind="skip", accepted=True),
                                 dict(energy_version="")])
def test_event_validation(tmp_path, bad):
    from gareus.store import ParquetExchangeWriter
    w = ParquetExchangeWriter(tmp_path, event_schema=EXCHANGE_EVENT_SCHEMA)
    with pytest.raises(ValueError):
        w.write_event(**_event(1, 0, **bad))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `tests/test_aux_store_writers.py -k "exchange or event"`
Expected: FAIL `ModuleNotFoundError: No module named 'gareus.auxiliary_cv.ledger'`

- [ ] **Step 3: Implement `ledger.py` (constants and checksum)**

```python
# gareus/auxiliary_cv/ledger.py
"""Ordered exchange-event ledger: schema, assignment checksums and replay (spec Section 7)."""
from __future__ import annotations

from typing import Sequence

from ..correctness._io import digest, json_bytes

EXCHANGE_EVENT_SCHEMA = "atlas-exchange-events-v1"
#: skip = a proposal the swap kernel declined to attempt (no holder / no outcome).
EVENT_KINDS = ("swap", "stay", "no_candidates", "skip")


def assignment_sha256(assignments: Sequence[int]) -> str:
    """Checksum of the replica -> window assignment list (index = replica)."""
    return digest(json_bytes([int(x) for x in assignments]))
```

- [ ] **Step 4: Implement the writer's event mode**

In `ParquetExchangeWriter.__init__`, add `event_schema=None` and after the existing body:

```python
        self._event_schema = event_schema
        recorded = (self._manifest or {}).get("payload_schema")
        wanted = {"schema": event_schema} if event_schema is not None else None
        if self._manifest and self._manifest["files"] and recorded != wanted:
            raise ParquetManifestError(f"payload schema of {self._out_dir} is {recorded!r}, writer wants {wanted!r}")
```

At the top of `write_exchange`:

```python
        if self._event_schema is not None:
            raise ValueError("event-mode writer requires write_event (ordered ledger), not write_exchange")
```

Add the method:

```python
    def write_event(self, *, step, attempt_seq, selected_replica, replica_i, replica_j, window_i, window_j,
                    kind, delta_e_kj, accepted, log_q_forward, log_q_reverse, p_accept, energy_version,
                    assignments_after) -> None:
        from .auxiliary_cv.ledger import EVENT_KINDS, assignment_sha256
        if self._event_schema is None:
            raise ValueError("write_event needs a writer constructed with event_schema")
        if kind not in EVENT_KINDS:
            raise ValueError(f"exchange event kind must be one of {EVENT_KINDS}, got {kind!r}")
        if int(attempt_seq) < 0:
            raise ValueError("attempt_seq must be >= 0")
        if kind != "swap" and bool(accepted):
            raise ValueError("only a swap event can be accepted")
        if not isinstance(energy_version, str) or not energy_version:
            raise ValueError("exchange event needs its energy/schema version")
        b = self._buf
        b["step"].append(int(step)); b["replica_i"].append(int(replica_i)); b["replica_j"].append(int(replica_j))
        b["window_i"].append(int(window_i)); b["window_j"].append(int(window_j))
        b["delta_e"].append(float(delta_e_kj)); b["accepted"].append(bool(accepted))
        b["attempt_seq"].append(int(attempt_seq)); b["selected_replica"].append(int(selected_replica))
        b["kind"].append(kind); b["delta_e_kj"].append(float(delta_e_kj))
        b["log_q_forward"].append(float(log_q_forward)); b["log_q_reverse"].append(float(log_q_reverse))
        b["p_accept"].append(float(p_accept)); b["energy_version"].append(energy_version)
        b["assignment_sha256_after"].append(assignment_sha256(assignments_after))
        if len(b["step"]) >= self._flush_rows:
            self.flush()
```

In `flush`, after building `tbl`:

```python
        if self._event_schema is not None:
            extra = {
                "attempt_seq": pa.array(b["attempt_seq"], type=pa.uint32()),
                "selected_replica": pa.array(b["selected_replica"], type=pa.int32()),
                "kind": pa.array(b["kind"], type=pa.string()),
                "delta_e_kj": pa.array(b["delta_e_kj"], type=pa.float64()),
                "log_q_forward": pa.array(b["log_q_forward"], type=pa.float64()),
                "log_q_reverse": pa.array(b["log_q_reverse"], type=pa.float64()),
                "p_accept": pa.array(b["p_accept"], type=pa.float64()),
                "energy_version": pa.array(b["energy_version"], type=pa.string()),
                "assignment_sha256_after": pa.array(b["assignment_sha256_after"], type=pa.string()),
            }
            for name, column in extra.items():
                tbl = tbl.append_column(name, column)
```

Other changes:
- Pass `payload_schema=({"schema": self._event_schema} if self._event_schema else None)` to `append_file_to_manifest` and `replace_files_in_manifest`.
- In `_consolidate`, sort event-mode tables by step, then `attempt_seq`:

```python
        keys = [("step", "ascending")]
        if self._event_schema is not None:
            keys.append(("attempt_seq", "ascending"))
        tbl = tbl.sort_by(keys)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `tests/test_aux_store_writers.py tests/test_store.py tests/test_parquet_manifest_transactions.py`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add gareus/auxiliary_cv/ledger.py gareus/store.py tests/test_aux_store_writers.py
git commit -m "feat(cvaux): ordered exchange-event ledger (swap/stay/no_candidates/skip); legacy schema pinned"
```

---

### Task 5: Ledger replay; re-seal crashed parents at the checkpoint on resume (auxiliary runs only)

**Files:**
- Modify: `gareus/auxiliary_cv/ledger.py` (`replay_assignments`)
- Modify: `gareus/store.py` (new `reseal_parent_for_resume`, after `finalize_segment` at `:362`)
- Modify: `gareus/production.py:8892-8900` (resume branch inside `if manifest is not None:`): an aux-gated `reseal_parent_for_resume` branch in front of the **verbatim** legacy `_parent_was_running` seal (user ruling B4)
- Test: `tests/test_aux_ledger.py`

**Interfaces:**
- Produces:
  - `replay_assignments(events, start_assignments, *, after_step, up_to_step) -> list[int]`:
    - selects rows with `after_step < step <= up_to_step`, ordered by (step, attempt_seq);
    - raises on a duplicate (step, attempt_seq);
    - an accepted `swap` checks the holders and exchanges windows;
    - every row's checksum must equal `assignment_sha256_after`;
    - masked values in event columns in the selected range raise `IntegrityError("legacy exchange rows mixed into an event ledger ...")`;
    - with no event columns at all it raises `IntegrityError("no ordered event columns")`.
  - `reseal_parent_for_resume(registry, parent_segment_id, checkpoint_absolute_step) -> dict | None`. If the parent's status is `running` or `interrupted`, it seals `interrupted` at `min(end_step, checkpoint_absolute_step)` (end_step None counts as +inf) and returns a record. It does nothing for `complete`, `abandoned` or a missing parent.

**Why, and the scope (user ruling B4).**
- After an in-process exception, the `finally` block's `finalize_segment(..., end_step=calib_steps + prod_done)` (`production.py:9255`, `store.py:362-397`) seals the parent `interrupted` at the *crash* step.
- Today's resume re-seals only parents still marked `running` (`production.py:8895`).
- So post-checkpoint samples and exchange rows of an exception crash stay visible and are re-run in the child segment. Samples are double-counted, and an event ledger would hold duplicate (step, attempt_seq).
- **Scope:** the re-seal runs only when this process built an auxiliary runtime (`args._aux_runtime is not None`; per process, never inferred from directory contents). Every other resume keeps the legacy seal code byte-for-byte. The legacy exception-crash double count is a **known residual** filed for a separate PR; it is not changed here.
- Gate on `_aux_runtime`, not on `_aux_io`: `_aux_io` exists only after Task 13, and `_aux_runtime` is set at base-system build (`production.py:7126`), before `load_production_checkpoint` runs. Task 12's resume refusal has already refused any capability mismatch by then, so an aux runtime implies an aux campaign.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_aux_ledger.py
import numpy as np
import pytest

from gareus.auxiliary_cv.ledger import EXCHANGE_EVENT_SCHEMA, assignment_sha256, replay_assignments
from gareus.correctness._io import IntegrityError


def _writer(run_dir, seg_id):
    from gareus.store import ParquetExchangeWriter
    return ParquetExchangeWriter(run_dir / "exchanges" / seg_id, flush_rows=1, event_schema=EXCHANGE_EVENT_SCHEMA)


def _swap(w, step, seq, ri, rj, wi, wj, after, accepted=True):
    w.write_event(step=step, attempt_seq=seq, selected_replica=ri, replica_i=ri, replica_j=rj, window_i=wi,
                  window_j=wj, kind="swap", delta_e_kj=0.0, accepted=accepted, log_q_forward=0.0,
                  log_q_reverse=0.0, p_accept=1.0, energy_version="v3", assignments_after=after)


def _quiet(w, step, seq, r, win, after, kind="stay"):
    w.write_event(step=step, attempt_seq=seq, selected_replica=r, replica_i=r, replica_j=r, window_i=win,
                  window_j=win, kind=kind, delta_e_kj=0.0, accepted=False, log_q_forward=0.0,
                  log_q_reverse=float("nan"), p_accept=float("nan"), energy_version="v3", assignments_after=after)


def _crashed_parent(tmp_path):
    """Checkpoint at 200 ([1,2,0]); an accepted swap at 300; then an EXCEPTION exit (finalize_segment)."""
    from gareus.store import SegmentRegistry, finalize_segment
    reg = SegmentRegistry(tmp_path)
    seg = reg.open_segment("run", None, 1)
    w = _writer(tmp_path, seg)
    _swap(w, 100, 0, 0, 1, 0, 1, [1, 0, 2])
    _swap(w, 100, 1, 1, 2, 0, 2, [1, 2, 0])        # second accepted swap at the SAME step
    _quiet(w, 200, 0, 0, 1, [1, 2, 0])
    _quiet(w, 200, 1, 2, 0, [1, 2, 0], kind="skip")
    _swap(w, 200, 2, 0, 2, 1, 0, [1, 2, 0], accepted=False)
    _swap(w, 300, 0, 0, 1, 1, 2, [2, 1, 0])        # after the checkpoint: phantom once resumed
    w.flush()
    finalize_segment(reg, seg, completed_cleanly=False, writers_ok=True, end_step=300)
    return reg, seg


def test_exception_exit_seals_at_crash_step_then_resume_reseals_at_checkpoint(tmp_path):
    from gareus.query import load_exchanges
    from gareus.store import reseal_parent_for_resume
    reg, seg = _crashed_parent(tmp_path)
    assert reg.get_segment(seg)["end_step"] == 300 and int(np.max(load_exchanges(tmp_path)["step"])) == 300
    rec = reseal_parent_for_resume(reg, seg, 200)
    assert rec == {"segment_id": seg, "previous_status": "interrupted", "previous_end_step": 300, "end_step": 200}
    ev = load_exchanges(tmp_path)
    assert int(np.max(ev["step"])) == 200
    assert replay_assignments(ev, [0, 1, 2], after_step=0, up_to_step=10**9) == [1, 2, 0]


def test_reseal_running_parent_and_leave_complete_alone(tmp_path):
    from gareus.store import SegmentRegistry, reseal_parent_for_resume
    reg = SegmentRegistry(tmp_path)
    a = reg.open_segment("run", None, 1)
    assert reseal_parent_for_resume(reg, a, 500)["end_step"] == 500        # running, end None
    b = reg.open_segment("run", a, 1)
    reg.close_segment(b, 900)
    assert reseal_parent_for_resume(reg, b, 400) is None and reg.get_segment(b)["status"] == "complete"
    assert reseal_parent_for_resume(reg, "nope", 1) is None


def test_crash_during_flush_leaves_orphan_tmp_ignored(tmp_path):
    from gareus.query import load_exchanges
    from gareus.store import reseal_parent_for_resume
    reg, seg = _crashed_parent(tmp_path)
    reseal_parent_for_resume(reg, seg, 200)
    (tmp_path / "exchanges" / seg / "chunk_000099.parquet.tmp.4242").write_bytes(b"partial")
    assert replay_assignments(load_exchanges(tmp_path), [0, 1, 2], after_step=0, up_to_step=200) == [1, 2, 0]


def test_checksum_mismatch_raises(tmp_path):
    from gareus.query import load_exchanges
    _crashed_parent(tmp_path)
    ev = load_exchanges(tmp_path)
    ev["assignment_sha256_after"] = np.asarray(ev["assignment_sha256_after"], dtype=object).copy()
    ev["assignment_sha256_after"][1] = assignment_sha256([9, 9, 9])
    with pytest.raises(IntegrityError, match="step 100 seq 1"):
        replay_assignments(ev, [0, 1, 2], after_step=0, up_to_step=200)


def test_window_mismatch_raises(tmp_path):
    from gareus.query import load_exchanges
    _crashed_parent(tmp_path)
    with pytest.raises(IntegrityError, match="window"):
        replay_assignments(load_exchanges(tmp_path), [2, 1, 0], after_step=0, up_to_step=200)


def test_duplicate_step_seq_raises(tmp_path):
    from gareus.query import load_exchanges
    _crashed_parent(tmp_path)
    ev = load_exchanges(tmp_path)
    ev = {k: np.concatenate([np.asarray(v), np.asarray(v)[:1]]) for k, v in ev.items()}
    with pytest.raises(IntegrityError, match="duplicate"):
        replay_assignments(ev, [0, 1, 2], after_step=0, up_to_step=200)


def test_legacy_rows_mixed_into_event_ledger_raise(tmp_path):
    from gareus.query import load_exchanges
    from gareus.store import ParquetExchangeWriter, SegmentRegistry
    reg = SegmentRegistry(tmp_path)
    s0 = reg.open_segment("run", None, 1)
    w0 = ParquetExchangeWriter(tmp_path / "exchanges" / s0)
    w0.write_exchange(step=50, replica_i=0, replica_j=1, window_i=0, window_j=1, delta_e=0.5, accepted=True)
    w0.close(); reg.close_segment(s0, 50)
    s1 = reg.open_segment("run", s0, 1)
    w1 = _writer(tmp_path, s1)
    _swap(w1, 100, 0, 0, 1, 1, 0, [0, 1, 2])
    w1.close(); reg.close_segment(s1, 100)
    with pytest.raises(IntegrityError, match="legacy exchange rows mixed"):
        replay_assignments(load_exchanges(tmp_path), [1, 0, 2], after_step=0, up_to_step=100)


def test_float_nan_event_columns_raise_before_any_int_cast(tmp_path):
    from gareus.query import load_exchanges
    reg_seg = "seg_001"
    w = _writer(tmp_path, reg_seg)
    _swap(w, 100, 0, 0, 1, 1, 0, [1, 0, 2])
    _swap(w, 200, 0, 0, 1, 1, 0, [0, 1, 2])
    w.close()
    ev = dict(load_exchanges(tmp_path))
    seq = np.asarray(np.ma.getdata(ev["attempt_seq"]), dtype=np.float64).copy()
    seq[1] = np.nan
    ev["attempt_seq"] = seq
    with pytest.raises(IntegrityError, match="legacy exchange rows mixed.*NaN"):
        replay_assignments(ev, [0, 1, 2], after_step=0, up_to_step=200)
    sha = np.asarray(ev["assignment_sha256_after"], dtype=object).copy()
    sha[0] = float("nan")
    ev["attempt_seq"] = np.asarray(np.ma.getdata(load_exchanges(tmp_path)["attempt_seq"]))
    ev["assignment_sha256_after"] = sha
    with pytest.raises(IntegrityError, match="legacy exchange rows mixed.*None/NaN"):
        replay_assignments(ev, [0, 1, 2], after_step=0, up_to_step=200)


def test_legacy_exchanges_cannot_be_replayed(tmp_path):
    from gareus.query import load_exchanges
    from gareus.store import ParquetExchangeWriter
    w = ParquetExchangeWriter(tmp_path / "exchanges" / "seg_001")
    w.write_exchange(step=5, replica_i=0, replica_j=1, window_i=0, window_j=1, delta_e=0.5, accepted=True)
    w.close()
    with pytest.raises(IntegrityError, match="ordered event"):
        replay_assignments(load_exchanges(tmp_path), [0, 1], after_step=0, up_to_step=10)


def test_production_resume_reseal_is_aux_gated_and_legacy_seal_is_unchanged():
    """B4 (user ruling): re-seal for auxiliary runs only; the legacy checkpoint-resume seal stays verbatim."""
    import ast
    import inspect
    import gareus.production as production
    tree = ast.parse(inspect.getsource(production.run_gareus))
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}

    def calls(name):
        return [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                and getattr(n.func, "attr", getattr(n.func, "id", None)) == name]

    def if_tests(node):
        out = []
        while node in parents:
            node = parents[node]
            if isinstance(node, ast.If):
                out.append(ast.unparse(node.test))
        return out

    reseal = calls("reseal_parent_for_resume")
    assert len(reseal) == 1 and any("aux" in t for t in if_tests(reseal[0]))
    interrupted = [c for c in calls("seal_segment") if any(
        k.arg == "status" and isinstance(k.value, ast.Constant) and k.value.value == "interrupted"
        for k in c.keywords)]
    assert len(interrupted) == 1, "the legacy checkpoint-resume seal must stay"
    assert if_tests(interrupted[0])[0] == "_parent_was_running and _parent_seg_id is not None"
    assert ast.unparse(interrupted[0]) == (
        "_seg_registry.seal_segment(_parent_seg_id, absolute_end_step=int(manifest.get('absolute_step', 0)), "
        "status='interrupted')")
```

`test_legacy_rows_mixed_into_event_ledger_raise` relies on `query._concat_numpy_dicts` back-filling the legacy segment's missing event columns with masked placeholders. If duckdb instead returns NaN/None arrays for them, `_col` must still detect the missing values: treat `None` entries in object arrays and NaN in `attempt_seq`/`assignment_sha256_after` the same way.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `tests/test_aux_ledger.py`
Expected: FAIL `ImportError: cannot import name 'replay_assignments'`

- [ ] **Step 3: Implement replay**

Append to `gareus/auxiliary_cv/ledger.py`:

```python
from typing import Mapping

import numpy as np

from ..correctness._io import IntegrityError

_EVENT_COLUMNS = ("attempt_seq", "kind", "assignment_sha256_after", "selected_replica")


def _col(events, name, rows=None):
    value = events[name]
    if np.ma.isMaskedArray(value):
        mask = np.ma.getmaskarray(value)
        if rows is not None and name in _EVENT_COLUMNS and mask[rows].any():
            raise IntegrityError(f"legacy exchange rows mixed into an event ledger (column {name} has "
                                 f"{int(mask[rows].sum())} missing values in the replay range)")
        value = np.ma.getdata(value)
    arr = np.asarray(value)
    if rows is not None and name in _EVENT_COLUMNS:
        # Board condition 2: a float column (pandas/NumPy upcast of a missing int) carries NaN, and
        # NaN.astype(int64) is garbage. Detect it before any cast, deterministically.
        if arr.dtype.kind == "f" and np.isnan(arr[rows]).any():
            raise IntegrityError(f"legacy exchange rows mixed into an event ledger (column {name} has NaN)")
        if arr.dtype == object and any(x is None or (isinstance(x, float) and x != x) for x in arr[rows]):
            raise IntegrityError(f"legacy exchange rows mixed into an event ledger (column {name} has None/NaN)")
    return arr


def replay_assignments(events: Mapping[str, np.ndarray], start_assignments: Sequence[int], *,
                       after_step: int, up_to_step: int) -> list[int]:
    if "attempt_seq" not in events or "assignment_sha256_after" not in events:
        raise IntegrityError("exchange data has no ordered event columns; replay needs an event-mode ledger")
    step = np.asarray(np.ma.getdata(events["step"])).astype(np.int64)
    rows = np.flatnonzero((step > int(after_step)) & (step <= int(up_to_step)))
    seq_all = _col(events, "attempt_seq", rows)
    seq = seq_all.astype(np.int64)
    idx = rows[np.lexsort((seq[rows], step[rows]))]
    pairs = list(zip(step[idx].tolist(), seq[idx].tolist()))
    if len(set(pairs)) != len(pairs):
        raise IntegrityError("duplicate (step, attempt_seq) in the exchange ledger")
    kind = _col(events, "kind", rows)
    accepted = _col(events, "accepted").astype(bool)
    ri = _col(events, "replica_i").astype(np.int64); rj = _col(events, "replica_j").astype(np.int64)
    wi = _col(events, "window_i").astype(np.int64); wj = _col(events, "window_j").astype(np.int64)
    sha = _col(events, "assignment_sha256_after", rows)
    current = [int(x) for x in start_assignments]
    for n in idx:
        label = f"step {int(step[n])} seq {int(seq[n])}"
        if kind[n] == "swap" and accepted[n]:
            a, b = int(ri[n]), int(rj[n])
            if current[a] != int(wi[n]) or current[b] != int(wj[n]):
                raise IntegrityError(f"{label}: ledger window pair ({int(wi[n])},{int(wj[n])}) does not match "
                                     f"replicas {a},{b} holding ({current[a]},{current[b]})")
            current[a], current[b] = current[b], current[a]
        if assignment_sha256(current) != str(sha[n]):
            raise IntegrityError(f"{label}: assignment checksum after the event does not match the ledger")
    return current
```

- [ ] **Step 4: Implement `reseal_parent_for_resume` in `gareus/store.py`** (after `finalize_segment`)

```python
def reseal_parent_for_resume(registry: "SegmentRegistry", parent_segment_id: Optional[str],
                             checkpoint_absolute_step: int) -> Optional[Dict[str, Any]]:
    """On resume of an AUXILIARY run, cut a non-complete parent segment back to its checkpoint.

    ``finalize_segment`` seals an exception exit at the crash step, and a killed job leaves the
    parent ``running``. Either way, rows after the checkpoint come from a state that was rolled
    back; the resumed segment re-runs those steps. Without this cut they would be pooled twice and
    the event ledger would hold duplicate (step, attempt_seq). Legacy runs keep their own seal
    (production resume branch, ruling B4).
    """
    seg = registry.get_segment(parent_segment_id) if parent_segment_id is not None else None
    if seg is None or seg.get("status") not in ("running", "interrupted"):
        return None
    previous_end = seg.get("end_step")
    cut = int(checkpoint_absolute_step) if previous_end is None else min(int(previous_end), int(checkpoint_absolute_step))
    previous_status = seg.get("status")
    registry.seal_segment(parent_segment_id, absolute_end_step=cut, status="interrupted")
    return {"segment_id": parent_segment_id, "previous_status": previous_status,
            "previous_end_step": previous_end, "end_step": cut}
```

- [ ] **Step 5: Gate it in the resume branch** (`gareus/production.py`)

Import `reseal_parent_for_resume` with the other `.store` names (line 35). In the resume branch, replace

```python
                if _parent_was_running and _parent_seg_id is not None:
                    _seg_registry.seal_segment(
                        _parent_seg_id,
                        absolute_end_step=int(manifest.get("absolute_step", 0)),
                        status="interrupted",
                    )
```

with (the `elif` body is the old code, unchanged):

```python
                if getattr(args, "_aux_runtime", None) is not None:
                    # Auxiliary runs only (ruling B4): also cut a parent that an exception exit sealed
                    # at its crash step, so no post-checkpoint row or ledger event is pooled twice.
                    _reseal_record = reseal_parent_for_resume(
                        _seg_registry, _parent_seg_id, int(manifest.get("absolute_step", 0)))
                    if _reseal_record is not None and _reseal_record["previous_end_step"] not in (
                            None, _reseal_record["end_step"]):
                        print(f"    Resume: parent segment {_reseal_record['segment_id']} re-sealed at checkpoint "
                              f"step {_reseal_record['end_step']} (was {_reseal_record['previous_status']} at "
                              f"{_reseal_record['previous_end_step']}); later rows were rolled back.", flush=True)
                elif _parent_was_running and _parent_seg_id is not None:
                    _seg_registry.seal_segment(
                        _parent_seg_id,
                        absolute_end_step=int(manifest.get("absolute_step", 0)),
                        status="interrupted",
                    )
```

Keep the two `abandoned` branches unchanged. Task 13 adds the ledger cross-check inside the auxiliary branch.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `tests/test_aux_ledger.py tests/test_store.py tests/test_segment_finalization.py tests/test_query.py tests/test_resume_and_poincare_regressions.py`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add gareus/auxiliary_cv/ledger.py gareus/store.py gareus/production.py tests/test_aux_ledger.py
git commit -m "feat(cvaux): ledger replay; re-seal crashed parent segments at the resume checkpoint (aux runs only)"
```

---

### Task 6: Checkpoint binding, save-time and pre-re-apply verification, ledger cross-check

**Files:**
- Create: `gareus/auxiliary_cv/checkpoint.py`
- Modify: `gareus/production.py:5255-5370` (`save_production_checkpoint`, keyword `aux_block=None` written as `manifest["aux"]` just before `publish_generation` at `:5368`)
- Modify: `gareus/production.py:5522-5652` (`load_production_checkpoint`, keyword `aux_pre_apply=None`; the assignment-count check is at `:5624-5626`, `def _apply_assignment` at `:5628`). It is a callable `aux_pre_apply(manifest, assignments)`, called after `assignments` is read and its length checked, **before** the `_apply_assignment` loop.
- Test: `tests/test_aux_checkpoint.py`

**Interfaces:**
- Consumes:
  - Stage A `AuxForceInfo`, `build_aux_force`, `set_aux_parameters`, `KJ_PER_KCAL`, `state_definition_hash`;
  - Task 5 `replay_assignments`;
  - Stage B `AuxStateTable`.
- Produces:
  - `AUX_CHECKPOINT_SCHEMA = "atlas-aux-checkpoint-v1"`;
  - `expected_aux_parameters(state_definition, window_id) -> (k_kJ, centre)`, giving (0.0, 0.0) when inactive;
  - `read_aux_parameters(context, info) -> (k, c)`;
  - `topology_identity_sha256(topology) -> str`, a sha of (atom index, atom name, residue name, residue index, chain index) plus sorted bond index pairs;
  - `aux_checkpoint_block(*, state_definition, force_info, assignments, observed_params, topology_sha256, kernel_identity_digest, segment_id, ledger_anchor, data_boundary) -> dict`. It **raises** if `observed_params` differ from `expected_aux_parameters` (save-time verification, C5/S8). A Context that already disagrees is never checkpointed as good.
  - `verify_aux_resume(manifest, *, aux_enabled, state_definition, force_info, assignments, observed_params, topology_sha256, kernel_identity_digest) -> None`. It is called with Context parameters read **after `loadCheckpoint` and before the re-apply**. It compares them to the expected values *and* to `block["observed_params"]`. A changed state definition is refused with the first differing fields named (`compare_state_definitions`), so a resume-time drift of the CV1/CV2 definitions, the boost envelope, the box or the physical system is diagnosable (ruling M3);
  - `verify_aux_ledger(manifest, events) -> None`. It replays from `block["ledger_anchor"]` (`start_step`, `start_assignments`) to `manifest["absolute_step"]` and requires the manifest assignments;
  - `aux_table_from_checkpoint(manifest, *, model) -> AuxStateTable` (every row carries its instance: v2 requires one).
- `ledger_anchor = {"segment_id", "start_step", "start_assignments"}` is the segment's first absolute step and assignments when its production loop began.
- `data_boundary = {"samples": {"generation", "n_rows"}, "exchanges": {"generation", "n_rows"}}` is read from the writers' manifests right after `flush_scalar_writers()`.
- Refusals (each an `IntegrityError` naming the field):
  - the capability mismatches;
  - the state-table hash changed (the message names the first differing fields);
  - the force identity changed;
  - the carrier count differs;
  - the assignment checksum differs;
  - the pre-apply Context parameters differ from the expected values or from the saved `observed_params`;
  - the topology identity differs;
  - the kernel identity digest differs;
  - ledger replay does not reproduce the manifest assignments.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_aux_checkpoint.py
import numpy as np
import pytest

from aux_c_fixture import definition, rows
from aux_cv_fixture import dipeptide, model_payload
from gareus.auxiliary_cv.checkpoint import (aux_checkpoint_block, aux_table_from_checkpoint,
                                            expected_aux_parameters, read_aux_parameters,
                                            topology_identity_sha256, verify_aux_ledger, verify_aux_resume)
from gareus.auxiliary_cv.force import AuxForceInfo, build_aux_force, set_aux_parameters
from gareus.auxiliary_cv.ledger import EXCHANGE_EVENT_SCHEMA
from gareus.auxiliary_cv.model import AuxModel
from gareus.correctness._io import IntegrityError

MODEL = AuxModel.from_mapping(model_payload([(0, 1, 2, 3)], [1.0, 0.5], offset=0.1))
INFO = AuxForceInfo("ATLaSAuxCVUmbrella", 7, MODEL.model_sha256, "aux_k", "aux_c", ("aux_cos_neg", "aux_sin_neg"))
TOPO = "c" * 64
KID = "d" * 64


def _defn(model=MODEL, aux_k=1.2):
    spec = [(0.0, 10.0, 0.0, 0.0, "ordinary"), (0.2, 10.0, 0.0, 0.0, "ordinary"), (0.4, 10.0, aux_k, 1.5, "auxiliary")]
    return definition(rows(spec, model.model_sha256), model)


def _observed(defn, assignments):
    return [expected_aux_parameters(defn, w) for w in assignments]


def _block(d, a, **kw):
    base = dict(state_definition=d, force_info=INFO, assignments=a, observed_params=_observed(d, a),
                topology_sha256=TOPO, kernel_identity_digest=KID, segment_id="seg_001",
                ledger_anchor={"segment_id": "seg_001", "start_step": 0, "start_assignments": [0, 1, 2]},
                data_boundary={"samples": {"generation": 3, "n_rows": 30}, "exchanges": {"generation": 2, "n_rows": 9}})
    base.update(kw)
    return aux_checkpoint_block(**base)


def _verify_kw(d, a):
    return dict(aux_enabled=True, state_definition=d, force_info=INFO, assignments=a,
                observed_params=_observed(d, a), topology_sha256=TOPO, kernel_identity_digest=KID)


def test_expected_parameters_units():
    d = _defn()
    assert expected_aux_parameters(d, 0) == (0.0, 0.0)
    assert expected_aux_parameters(d, 2) == pytest.approx((1.2 * 4.184, 1.5), rel=1e-15)


def test_save_time_verification_refuses_a_bad_context():
    d, a = _defn(), [2, 0, 1]
    with pytest.raises(IntegrityError, match="save"):
        _block(d, a, observed_params=[(0.0, 0.0)] * 3)


def test_block_round_trip_verifies():
    d, a = _defn(), [2, 0, 1]
    verify_aux_resume({"aux": _block(d, a)}, **_verify_kw(d, a))


@pytest.mark.parametrize("case, match", [
    ("changed_model", "state_definition_sha256"),
    ("force_group", "force_info"),
    ("carrier_count", "replica"),
    ("pre_apply_params", "before re-apply"),
    ("assignments", "assignment"),
    ("aux_off", "capability"),
    ("topology", "topology"),
    ("kernel", "kernel"),
])
def test_resume_refusals(case, match):
    d, a = _defn(), [2, 0, 1]
    manifest = {"aux": _block(d, a)}
    kw = _verify_kw(d, a)
    if case == "changed_model":
        other = AuxModel.from_mapping(model_payload([(0, 1, 2, 3)], [1.0, 0.5000001], offset=0.1))
        kw["state_definition"] = _defn(other)
    elif case == "force_group":
        kw["force_info"] = AuxForceInfo(INFO.name, 8, INFO.model_sha256, INFO.global_k, INFO.global_c, INFO.sub_cv_names)
    elif case == "carrier_count":
        kw["assignments"], kw["observed_params"] = a[:2], _observed(d, a[:2])
    elif case == "pre_apply_params":
        kw["observed_params"] = [(0.0, 0.0)] * 3        # checkpoint restored an inactive aux force
    elif case == "assignments":
        kw["assignments"] = [0, 2, 1]
        kw["observed_params"] = _observed(d, [0, 2, 1])
    elif case == "aux_off":
        kw["aux_enabled"] = False
    elif case == "topology":
        kw["topology_sha256"] = "e" * 64
    elif case == "kernel":
        kw["kernel_identity_digest"] = "f" * 64
    with pytest.raises(IntegrityError, match=match):
        verify_aux_resume(manifest, **kw)


def test_aux_enabled_against_legacy_manifest_refuses():
    d, a = _defn(), [2, 0, 1]
    with pytest.raises(IntegrityError, match="capability"):
        verify_aux_resume({}, **_verify_kw(d, a))


def test_ledger_cross_check(tmp_path):
    from gareus.query import load_exchanges
    from gareus.store import ParquetExchangeWriter
    d, a = _defn(), [1, 0, 2]
    w = ParquetExchangeWriter(tmp_path / "exchanges" / "seg_001", flush_rows=1, event_schema=EXCHANGE_EVENT_SCHEMA)
    w.write_event(step=100, attempt_seq=0, selected_replica=0, replica_i=0, replica_j=1, window_i=0, window_j=1,
                  kind="swap", delta_e_kj=0.0, accepted=True, log_q_forward=0.0, log_q_reverse=0.0, p_accept=1.0,
                  energy_version="v3", assignments_after=[1, 0, 2])
    w.close()
    manifest = {"aux": _block(d, a), "absolute_step": 100, "assignments": a}
    verify_aux_ledger(manifest, load_exchanges(tmp_path))
    with pytest.raises(IntegrityError, match="ledger"):
        verify_aux_ledger(dict(manifest, assignments=[0, 1, 2]), load_exchanges(tmp_path))


def test_topology_identity_is_stable_and_sensitive():
    d = dipeptide()
    assert topology_identity_sha256(d["topology"]) == topology_identity_sha256(d["topology"])
    from openmm import app
    t2 = app.Topology()
    c = t2.addChain(); r = t2.addResidue("ALA", c); t2.addAtom("CA", app.element.carbon, r)
    assert topology_identity_sha256(t2) != topology_identity_sha256(d["topology"])


def test_aux_table_from_checkpoint_round_trip():
    d, a = _defn(), [2, 0, 1]
    table = aux_table_from_checkpoint({"aux": _block(d, a)}, model=MODEL)
    assert table.n == 3 and table.k_kcal == (0.0, 0.0, 1.2) and table.centers[2] == 1.5
    assert table.instances[2]["state_role"] == "auxiliary"
    other = AuxModel.from_mapping(model_payload([(0, 1, 2, 3)], [0.0, 1.0]))
    with pytest.raises(IntegrityError, match="model"):
        aux_table_from_checkpoint({"aux": _block(d, a)}, model=other)


def test_read_parameters_from_a_real_context():
    import openmm as mm
    d = dipeptide()
    blocks = [lab.split("-")[0] for lab in d["labels"]]
    m = AuxModel.from_mapping(model_payload(d["quads"], [1.0] + [0.0] * (2 * len(d["quads"]) - 1), blocks=blocks))
    s = mm.System()
    for _ in range(len(d["positions_nm"])):
        s.addParticle(1.0)
    force, info = build_aux_force(mm, m, force_group=5)
    s.addForce(force)
    ctx = mm.Context(s, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference"))
    set_aux_parameters(ctx, info, center=0.7, k_kcal=2.0)
    assert read_aux_parameters(ctx, info) == pytest.approx((2.0 * 4.184, 0.7), rel=1e-15)
    chk = ctx.createCheckpoint()
    set_aux_parameters(ctx, info, center=0.0, k_kcal=0.0)
    ctx.loadCheckpoint(chk)                               # checkpoints restore global parameters
    assert read_aux_parameters(ctx, info) == pytest.approx((2.0 * 4.184, 0.7), rel=1e-15)


def test_state_change_names_the_differing_field():
    """M3: a resume-time drift names its field, so a non-reproducible input is diagnosable."""
    d, a = _defn(), [2, 0, 1]
    kw = _verify_kw(d, a)
    kw["state_definition"] = _defn(aux_k=1.3)
    with pytest.raises(IntegrityError, match=r"windows\[2\]\.aux_k"):
        verify_aux_resume({"aux": _block(d, a)}, **kw)


def test_aux_block_survives_publish_generation(tmp_path):
    """Preflight T6: the published root manifest keeps the aux block (checkpoint_store preserves caller fields)."""
    import json
    from gareus.correctness.checkpoint_store import checkpoint_manifest_path, publish_generation
    d, a = _defn(), [2, 0, 1]
    block = _block(d, a)
    fields = {"assignments": a, "prod_done": 0, "absolute_step": 0, "parity": 0, "attempt": 0,
              "next_exchange": 0, "next_log": 0, "rng_state": np.random.default_rng(0).bit_generator.state,
              "rng_bit_generator": "PCG64", "exchange_stats": {}, "aux": block}
    publish_generation(tmp_path, [b"r0", b"r1", b"r2"], fields)
    root = json.loads(checkpoint_manifest_path(tmp_path).read_text())
    assert root["aux"] == json.loads(json.dumps(block))
    verify_aux_resume(root, **_verify_kw(d, a))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `tests/test_aux_checkpoint.py`
Expected: FAIL `ModuleNotFoundError: No module named 'gareus.auxiliary_cv.checkpoint'`

- [ ] **Step 3: Implement `checkpoint.py`**

```python
# gareus/auxiliary_cv/checkpoint.py
"""Bind auxiliary state to production checkpoints and verify it on resume (spec Section 9).

A resume never guesses a state from CV centres. It checks the frozen state table hash, the force
identity, the carrier count, the assignment checksum, the topology identity, the kernel digest,
every Context's auxiliary parameters as restored by loadCheckpoint (before anything re-applies
them), and that the exchange ledger replays to the checkpoint's assignments.
"""
from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any, Mapping, Sequence

from ..correctness._io import IntegrityError, digest, json_bytes, json_loads
from ..correctness.bias import KJ_PER_KCAL
from ..correctness.state_identity import (canonical_state_definition, compare_state_definitions,
                                          state_definition_hash)
from .ledger import assignment_sha256, replay_assignments

AUX_CHECKPOINT_SCHEMA = "atlas-aux-checkpoint-v1"
_REL = 1e-12


def expected_aux_parameters(state_definition: Mapping[str, Any], window_id: int) -> tuple[float, float]:
    found = [r for r in state_definition["windows"] if r["window_id"] == int(window_id)]
    if len(found) != 1:
        raise IntegrityError(f"no window {window_id} in the auxiliary state table")
    k = float(found[0].get("aux_k", 0.0))
    if k == 0.0:
        return 0.0, 0.0
    return k * KJ_PER_KCAL, float(found[0]["aux_center"])


def read_aux_parameters(context, info) -> tuple[float, float]:
    return float(context.getParameter(info.global_k)), float(context.getParameter(info.global_c))


def topology_identity_sha256(topology) -> str:
    atoms = [[a.index, a.name, a.residue.name, a.residue.index, a.residue.chain.index] for a in topology.atoms()]
    bonds = sorted(sorted([b[0].index, b[1].index]) for b in topology.bonds())
    return digest(json_bytes({"atoms": atoms, "bonds": bonds}))


def _force_record(info) -> dict[str, Any]:
    record = asdict(info)
    record["sub_cv_names"] = list(record["sub_cv_names"])
    return record


# Board condition 8: strict on purpose. Measured 2026-10-08 (OpenMM 8.5.1, Reference and CPU):
# Context.createCheckpoint/loadCheckpoint restores CustomCVForce globals bit-exactly, both into the
# same Context and into a fresh one (scratch verify_C/board_checks.py). CUDA/OpenCL round-trip
# precision is measured in Task 15 Step 4; if a GPU platform is not bit-exact, record the measured
# deviation there and stop (spec Section 17) -- do not loosen _REL here.
def _check_params(state, assignments, observed, label) -> None:
    for r, (window, got) in enumerate(zip(assignments, observed)):
        expect = expected_aux_parameters(state, window)
        if not all(math.isclose(float(o), e, rel_tol=_REL, abs_tol=0.0) for o, e in zip(got, expect)):
            raise IntegrityError(f"{label}: replica {r} (window {window}) Context parameters {tuple(got)} != "
                                 f"state table {expect}")


def aux_checkpoint_block(*, state_definition, force_info, assignments: Sequence[int],
                         observed_params: Sequence[tuple[float, float]], topology_sha256: str,
                         kernel_identity_digest: str, segment_id: str, ledger_anchor: Mapping[str, Any],
                         data_boundary: Mapping[str, Any]) -> dict[str, Any]:
    state = canonical_state_definition(state_definition)
    if len(observed_params) != len(assignments):
        raise IntegrityError("save-time check: observed parameters do not cover every replica")
    _check_params(state, assignments, observed_params, "save-time check")
    return {
        "schema": AUX_CHECKPOINT_SCHEMA,
        "state_definition": state,
        "state_definition_sha256": state_definition_hash(state),
        "force_info": _force_record(force_info),
        "n_replicas": len(assignments),
        "assignment_sha256": assignment_sha256(assignments),
        "observed_params": [[float(k), float(c)] for k, c in observed_params],
        "topology_sha256": str(topology_sha256),
        "kernel_identity_digest": str(kernel_identity_digest),
        "segment_id": str(segment_id),
        "ledger_anchor": {"segment_id": str(ledger_anchor["segment_id"]),
                          "start_step": int(ledger_anchor["start_step"]),
                          "start_assignments": [int(x) for x in ledger_anchor["start_assignments"]]},
        "data_boundary": json_loads(json_bytes(dict(data_boundary))),
    }


def verify_aux_resume(manifest: Mapping[str, Any], *, aux_enabled: bool, state_definition, force_info,
                      assignments: Sequence[int], observed_params: Sequence[tuple[float, float]],
                      topology_sha256: str, kernel_identity_digest: str) -> None:
    """Call with the Context parameters as restored by loadCheckpoint, BEFORE any re-apply."""
    block = manifest.get("aux")
    if bool(aux_enabled) != (block is not None):
        raise IntegrityError("auxiliary capability differs between checkpoint and this run; a model or "
                             "population change starts a new phase, it is not a resume")
    if block is None:
        return
    if block.get("schema") != AUX_CHECKPOINT_SCHEMA:
        raise IntegrityError(f"unknown auxiliary checkpoint schema {block.get('schema')!r}")
    current = state_definition_hash(state_definition)
    if block["state_definition_sha256"] != current:
        diff = compare_state_definitions(canonical_state_definition(block["state_definition"]),
                                         canonical_state_definition(state_definition))
        fields = ", ".join(f"{d['field']}: {d['reference']!r} -> {d['candidate']!r}" for d in diff[:3])
        raise IntegrityError(f"state_definition_sha256 changed: checkpoint {block['state_definition_sha256']} vs run "
                             f"{current}; first differing fields: {fields}")
    if json_loads(json_bytes(block["force_info"])) != json_loads(json_bytes(_force_record(force_info))):
        raise IntegrityError(f"force_info changed: checkpoint {block['force_info']} vs run {_force_record(force_info)}")
    if block["topology_sha256"] != str(topology_sha256):
        raise IntegrityError("topology identity changed between checkpoint and this run")
    if block["kernel_identity_digest"] != str(kernel_identity_digest):
        raise IntegrityError("kernel identity digest changed between checkpoint and this run")
    if int(block["n_replicas"]) != len(assignments) or len(observed_params) != len(assignments):
        raise IntegrityError(f"replica/carrier set incomplete: checkpoint {block['n_replicas']}, run {len(assignments)}")
    if block["assignment_sha256"] != assignment_sha256(assignments):
        raise IntegrityError("resumed assignment does not match the checkpoint's assignment checksum")
    state = canonical_state_definition(state_definition)
    _check_params(state, assignments, observed_params, "before re-apply")
    for r, (got, saved) in enumerate(zip(observed_params, block["observed_params"])):
        if not all(math.isclose(float(o), float(s), rel_tol=_REL, abs_tol=0.0) for o, s in zip(got, saved)):
            raise IntegrityError(f"before re-apply: replica {r} restored parameters {tuple(got)} != saved {tuple(saved)}")


def verify_aux_ledger(manifest: Mapping[str, Any], events: Mapping[str, Any]) -> None:
    block = manifest.get("aux")
    if block is None:
        return
    anchor = block["ledger_anchor"]
    got = replay_assignments(events, anchor["start_assignments"], after_step=int(anchor["start_step"]),
                             up_to_step=int(manifest["absolute_step"]))
    if [int(x) for x in got] != [int(x) for x in manifest["assignments"]]:
        raise IntegrityError(f"exchange ledger replays to {got}, checkpoint holds {list(manifest['assignments'])}")


def aux_table_from_checkpoint(manifest: Mapping[str, Any], *, model):
    """Rebuild Stage B's AuxStateTable from the checkpoint block (the resume path's table source)."""
    from .state_table import AuxStateTable
    block = manifest.get("aux")
    if block is None:
        raise IntegrityError("checkpoint has no auxiliary block to resume an auxiliary run from")
    state = canonical_state_definition(block["state_definition"])
    registry = state.get("aux_models") or {}
    if model.model_sha256 not in registry:
        raise IntegrityError(f"--aux-cv-model {model.model_sha256} is not the checkpoint's model {sorted(registry)}")
    windows = sorted(state["windows"], key=lambda r: r["window_id"])
    return AuxStateTable(model, tuple(float(r["aux_center"]) for r in windows),
                         tuple(float(r["aux_k"]) for r in windows),
                         tuple(dict(r["instance"]) for r in windows))       # v2: an instance on every row
```

`verify_aux_ledger` raises the `replay_assignments` error on a broken ledger. When the replay ends in a different assignment, its own message says "exchange ledger replays to ...", which matches `match="ledger"`.

- [ ] **Step 4: Wire the optional blocks into `production.py`**

In `save_production_checkpoint`, add the keyword `aux_block: Optional[dict] = None` after `keep_generations`. Just before `from .correctness.checkpoint_store import publish_generation` (`:5368`):

```python
    if aux_block is not None:
        manifest["aux"] = aux_block
```

In `load_production_checkpoint`, add the keyword `aux_pre_apply=None`. Right after

```python
    if len(assignments) != len(sims):
        raise RuntimeError("Checkpoint assignment count does not match replica count")
```

and before `def _apply_assignment`:

```python
    if aux_pre_apply is not None:
        aux_pre_apply(manifest, assignments)       # parameters as restored by loadCheckpoint
```

Legacy: both keywords default to None, the manifest gains no key and nothing extra is called (Task 13 always passes `aux_pre_apply`; with the capability off it only checks that the manifest has no `aux` block). Add a structural test (append to `tests/test_aux_checkpoint.py`) pinning the order and the off-path invariance:

```python
def test_production_checkpoint_hooks_are_optional_and_ordered():
    import inspect
    import gareus.production as production
    save = inspect.getsource(production.save_production_checkpoint)
    assert "if aux_block is not None:\n        manifest[\"aux\"] = aux_block" in save
    load = inspect.getsource(production.load_production_checkpoint)
    assert load.index("aux_pre_apply(manifest, assignments)") < load.index("def _apply_assignment")
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `tests/test_aux_checkpoint.py tests/test_checkpoint_keep_generations_wiring.py tests/test_npt_production_integration.py tests/test_sample_before_exchange_ordering.py` (every test file that calls `save_production_checkpoint`/`load_production_checkpoint`)
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add gareus/auxiliary_cv/checkpoint.py gareus/production.py tests/test_aux_checkpoint.py
git commit -m "feat(cvaux): checkpoint binding with save-time and pre-re-apply parameter checks and ledger cross-check"
```

---

### Task 7: Kernel eligibility recognises persisted auxiliary segments

**Files:**
- Modify: `gareus/kernel_identity.py` (`classify_segment_kernel:108`; delete `AUX_PERSISTED_SNAPSHOT_KEY:51`; add `AUX_SAMPLES_PAYLOAD_SCHEMA`, `raw_sample_payload_schema`)
- Modify: `gareus/query.py:172-206` (`segment_eligibility`)
- Modify: `tests/test_aux_cv_kernel_identity.py` (Stage B): import line and `test_v3_aux_is_never_verified_in_stage_b`
- Test: `tests/test_aux_eligibility.py`

**Interfaces:**
- Produces:
  - `AUX_SAMPLES_PAYLOAD_SCHEMA = "atlas-aux-samples-v1"` in the import-light `kernel_identity` (equal to Task 2's `AUX_SAMPLES_SCHEMA`; a test pins the equality).
  - `raw_sample_payload_schema(run_dir, segment_id) -> dict | None`: the `payload_schema` object of `samples/<segment>/` `parquet_manifest.json`, read as raw JSON. It never validates files (no stat, no hash) and never raises: an unreadable or absent manifest gives None. This keeps eligibility from acquiring a new failure mode (ruling B5).
  - `classify_segment_kernel(window_snapshot, *, sample_payload_schema: dict | None = None) -> (status, reason)`.
- An auxiliary snapshot is one whose `kernel_identity` satisfies Stage B's `is_aux_kernel_record` (v3_aux exchange version, `aux_model_sha256` or `aux_cv_model_sha256`). For it:
  - `ELIGIBLE_AUX_UNPERSISTED` unless:
    - (a) `sample_payload_schema["schema"] == "atlas-aux-samples-v1"`, **and**
    - (b) the snapshot carries a frozen `state_definition` whose `aux_models` contains `kernel_identity["aux_model_sha256"]`, **and**
    - (c) the payload's `model_shas` contains the same sha.
  - When all three hold: non-residual CV2 gives `ELIGIBLE_VERIFIED` ("auxiliary features recorded"). Residual CV2 falls through to the residual rule, which accepts `EXCHANGE_ENERGY_VERSION_AUX` as the exchange version.
- Non-auxiliary snapshots: unchanged, byte for byte.
- **Replaces Stage B's marker (deliberate override of the Stage B handoff item "write `aux_sample_schema` into snapshots").**
  - Stage B classifies a v3_aux snapshot that carries the key `AUX_PERSISTED_SNAPSHOT_KEY` (`"aux_sample_schema"`) as `ELIGIBLE_UNKNOWN`.
  - This task deletes that constant and its branch: persistence is decided only by the sample manifest's `payload_schema` (rules (a)–(c)), which is written with the rows themselves and so cannot disagree with them.
  - Snapshots never carry an `aux_sample_schema` key; a stray one is ignored.
  - `tests/test_aux_cv_kernel_identity.py` imports the constant at module level, so its import line changes in the same commit, and `test_v3_aux_is_never_verified_in_stage_b` becomes `test_v3_aux_without_payload_stays_unpersisted`.
- `segment_eligibility` (rulings B5, H2):
  - It reads each snapshot once (the existing first pass) and caches it.
  - For a segment **with** a snapshot, it reads the samples payload only when the snapshot's kernel identity is auxiliary (B5). Legacy snapshots are classified exactly as before, with no extra file read.
  - For a segment **without** a snapshot: if the run is auxiliary, the segment is `ELIGIBLE_AUX_UNPERSISTED` with the reason "auxiliary run segment without a window snapshot" (H2; previously `not_applicable`, and its rows loaded without A_s). The run is auxiliary when any cached snapshot's kernel identity is auxiliary, or `run_manifest.json` `method_settings` is an auxiliary record, or that segment's own raw payload is `atlas-aux-samples-v1`. `run_manifest.json` and the raw payload are read only for snapshot-less segments. Otherwise the legacy rules (`unknown` when the latest snapshot is residual, else `not_applicable`) are unchanged.
- `_load_segmented_parquet` already excludes everything not VERIFIED or NOT_APPLICABLE, so Stage-B-era and snapshot-less auxiliary data stay excluded and reported here; `load_parquet` refuses them explicitly (Task 9).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_aux_eligibility.py
import json

import pytest

from gareus.kernel_identity import (AUX_SAMPLES_PAYLOAD_SCHEMA, ELIGIBLE_AUX_UNPERSISTED, ELIGIBLE_NOT_APPLICABLE,
                                    ELIGIBLE_VERIFIED, EXCHANGE_ENERGY_VERSION_AUX, classify_segment_kernel,
                                    raw_sample_payload_schema)

SHA = "e" * 64
PAYLOAD = {"schema": "atlas-aux-samples-v1", "model_shas": [SHA]}


def _snap(*, cv2="none", frozen=True, sha=SHA):
    snap = {"segment_id": "seg_001", "cv1_type": "contacts", "cv2_type": cv2,
            "kernel_identity": {"exchange_energy_version": EXCHANGE_ENERGY_VERSION_AUX, "aux_model_sha256": sha}}
    if frozen:
        snap["state_definition"] = {"aux_models": {SHA: {"schema": "atlas-aux-cv-model-v1"}}}
    return snap


def _segment_with_payload(run, seg_id, payload):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from gareus.parquet_manifest import append_file_to_manifest, file_record
    seg = run / "samples" / seg_id
    seg.mkdir(parents=True)
    p = seg / "chunk_000001.parquet"
    pq.write_table(pa.table({"step": pa.array([1], type=pa.uint64())}), p)
    append_file_to_manifest(seg, kind="samples", record=file_record(p, rows=1, first_step=1, last_step=1),
                            next_chunk_index=2, payload_schema=payload)


def _registry(run, *ids):
    (run / "segments.json").write_text(json.dumps([{"segment_id": s, "status": "complete"} for s in ids]))


def test_payload_constant_matches_the_sample_schema():
    from gareus.auxiliary_cv.sample_schema import AUX_SAMPLES_SCHEMA
    assert AUX_SAMPLES_PAYLOAD_SCHEMA == AUX_SAMPLES_SCHEMA


def test_unpersisted_without_payload():
    assert classify_segment_kernel(_snap())[0] == ELIGIBLE_AUX_UNPERSISTED
    assert classify_segment_kernel(_snap(), sample_payload_schema=None)[0] == ELIGIBLE_AUX_UNPERSISTED


def test_verified_with_payload_and_frozen_snapshot():
    assert classify_segment_kernel(_snap(), sample_payload_schema=PAYLOAD)[0] == ELIGIBLE_VERIFIED


@pytest.mark.parametrize("snap, payload", [
    (_snap(frozen=False), PAYLOAD),
    (_snap(sha="f" * 64), PAYLOAD),
    (_snap(), {"schema": "atlas-aux-samples-v1", "model_shas": ["f" * 64]}),
    (_snap(), {"schema": "something-else", "model_shas": [SHA]}),
])
def test_any_missing_binding_stays_unpersisted(snap, payload):
    assert classify_segment_kernel(snap, sample_payload_schema=payload)[0] == ELIGIBLE_AUX_UNPERSISTED


def test_legacy_snapshots_unchanged():
    assert classify_segment_kernel({"cv2_type": "none"})[0] == ELIGIBLE_NOT_APPLICABLE
    assert classify_segment_kernel({"cv2_type": "none"}, sample_payload_schema=PAYLOAD)[0] == ELIGIBLE_NOT_APPLICABLE


def test_segment_eligibility_reads_payload_for_aux_snapshots(tmp_path):
    from gareus.query import segment_eligibility
    _registry(tmp_path, "seg_001")
    (tmp_path / "windows").mkdir()
    (tmp_path / "windows" / "seg_001.json").write_text(json.dumps(_snap()))
    _segment_with_payload(tmp_path, "seg_001", PAYLOAD)
    assert segment_eligibility(tmp_path)["seg_001"]["eligibility"] == ELIGIBLE_VERIFIED


def test_legacy_eligibility_never_reads_the_samples_manifest(tmp_path, monkeypatch):
    """B5: a legacy segment's broken or missing manifest is not a new failure mode."""
    import gareus.kernel_identity as ki
    from gareus.query import segment_eligibility
    _registry(tmp_path, "seg_001")
    (tmp_path / "windows").mkdir()
    (tmp_path / "windows" / "seg_001.json").write_text(json.dumps({"segment_id": "seg_001", "cv2_type": None,
                                                                  "windows": []}))
    (tmp_path / "samples" / "seg_001").mkdir(parents=True)
    (tmp_path / "samples" / "seg_001" / "parquet_manifest.json").write_text("{ not json")

    def _boom(*a, **k):
        raise AssertionError("legacy eligibility read a samples manifest")
    monkeypatch.setattr(ki, "raw_sample_payload_schema", _boom)
    assert segment_eligibility(tmp_path)["seg_001"]["eligibility"] == ELIGIBLE_NOT_APPLICABLE


def test_snapshotless_segment_of_an_aux_run_is_unpersisted(tmp_path):
    """H2: a missing snapshot in an auxiliary run is never 'not_applicable'."""
    from gareus.query import segment_eligibility
    _registry(tmp_path, "seg_001", "seg_002")
    (tmp_path / "windows").mkdir()
    (tmp_path / "windows" / "seg_002.json").write_text(json.dumps(dict(_snap(), segment_id="seg_002")))
    out = segment_eligibility(tmp_path)
    assert out["seg_001"]["eligibility"] == ELIGIBLE_AUX_UNPERSISTED
    assert "without a window snapshot" in out["seg_001"]["reason"]


def test_snapshotless_segment_with_aux_payload_is_unpersisted(tmp_path):
    from gareus.query import segment_eligibility
    _registry(tmp_path, "seg_001")
    _segment_with_payload(tmp_path, "seg_001", PAYLOAD)
    assert raw_sample_payload_schema(tmp_path, "seg_001") == PAYLOAD
    assert segment_eligibility(tmp_path)["seg_001"]["eligibility"] == ELIGIBLE_AUX_UNPERSISTED


def test_snapshotless_segment_of_a_legacy_run_is_unchanged(tmp_path):
    from gareus.query import segment_eligibility
    _registry(tmp_path, "seg_001")
    assert segment_eligibility(tmp_path)["seg_001"]["eligibility"] == ELIGIBLE_NOT_APPLICABLE
    assert raw_sample_payload_schema(tmp_path, "seg_001") is None
```

Edit Stage B's `tests/test_aux_cv_kernel_identity.py` in the same commit:
- In the module import, replace `from gareus.kernel_identity import (AUX_PERSISTED_SNAPSHOT_KEY, ELIGIBLE_AUX_UNPERSISTED, ...` with the same list minus `AUX_PERSISTED_SNAPSHOT_KEY`.
- Replace `test_v3_aux_is_never_verified_in_stage_b` with:

```python
def test_v3_aux_without_payload_stays_unpersisted():
    """Stage C: persistence comes from the samples manifest payload, never from a snapshot key."""
    snap = {"cv2_type": "residual-torsion-pc", "aux_sample_schema": {"schema": "x"},
            "kernel_identity": {"cv_evaluator_version": RESIDUAL_EVALUATOR_VERSION,
                                "exchange_energy_version": EXCHANGE_ENERGY_VERSION_AUX}}
    assert classify_segment_kernel(snap)[0] == ELIGIBLE_AUX_UNPERSISTED
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `tests/test_aux_eligibility.py`
Expected: FAIL `ImportError: cannot import name 'AUX_SAMPLES_PAYLOAD_SCHEMA'`.

- [ ] **Step 3: Implement in `gareus/kernel_identity.py`**

Delete `AUX_PERSISTED_SNAPSHOT_KEY` and its comment. Add after `EXCHANGE_ENERGY_VERSION_AUX`:

```python
#: Payload schema a Stage C samples manifest records when every row stores the auxiliary z and the
#: full torsion basis (gareus.auxiliary_cv.sample_schema.AUX_SAMPLES_SCHEMA; kept here import-light).
AUX_SAMPLES_PAYLOAD_SCHEMA = "atlas-aux-samples-v1"


def raw_sample_payload_schema(run_dir, segment_id) -> "dict | None":
    """``payload_schema`` of ``samples/<segment>/`` as raw JSON; never validates files, never raises."""
    import json
    from pathlib import Path
    from .parquet_manifest import manifest_path
    try:
        raw = json.loads(manifest_path(Path(run_dir) / "samples" / str(segment_id)).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    payload = raw.get("payload_schema") if isinstance(raw, dict) else None
    return payload if isinstance(payload, dict) else None
```

Replace `classify_segment_kernel` with:

```python
def classify_segment_kernel(window_snapshot: dict, *, sample_payload_schema: "dict | None" = None) -> tuple:
    """(eligibility, reason) of one segment from its ``windows/<segment>.json`` payload.

    Non-residual segments are not affected by the residual fast-path defect and stay
    eligible under their own documented rules; a residual segment is verified only when its
    recorded evaluator and exchange versions equal the current ones. An auxiliary-CV segment is
    verified only when its samples manifest records the atlas-aux-samples-v1 payload and its frozen
    snapshot and that payload name the kernel's model (``sample_payload_schema``, Stage C).
    """
    snap = dict(window_snapshot or {})
    cv2 = str(snap.get("cv2_type") or "none")
    identity = snap.get("kernel_identity") or {}
    if is_aux_kernel_record(identity):
        sha = identity.get("aux_model_sha256")
        payload = dict(sample_payload_schema or {})
        state = snap.get("state_definition") or {}
        if payload.get("schema") != AUX_SAMPLES_PAYLOAD_SCHEMA:
            return ELIGIBLE_AUX_UNPERSISTED, ("auxiliary-CV segment whose samples carry no atlas-aux-samples-v1 "
                                              "payload (Stage B engineering run): its bias cannot be reconstructed")
        if not sha or sha not in (state.get("aux_models") or {}) or sha not in (payload.get("model_shas") or []):
            return ELIGIBLE_AUX_UNPERSISTED, ("auxiliary model of the kernel identity is not bound by the frozen "
                                              "snapshot and the sample schema")
        if cv2 != RESIDUAL_MODE:
            return ELIGIBLE_VERIFIED, "auxiliary features recorded (atlas-aux-samples-v1) with a frozen state table"
    elif cv2 != RESIDUAL_MODE:
        return ELIGIBLE_NOT_APPLICABLE, "secondary CV is not residual-torsion-pc"
    if not identity:
        return ELIGIBLE_UNKNOWN, "residual segment without a kernel_identity record (written before F01)"
    ev = identity.get("cv_evaluator_version")
    ex = identity.get("exchange_energy_version")
    if ev == RESIDUAL_EVALUATOR_VERSION and ex in (EXCHANGE_ENERGY_VERSION, EXCHANGE_ENERGY_VERSION_AUX):
        return ELIGIBLE_VERIFIED, "kernel matches the current evaluator and exchange assembly"
    if ev in (None, "affected_pre_f01"):
        return ELIGIBLE_AFFECTED, "recorded coordinates came from the two-term fall-through (review I01)"
    return ELIGIBLE_UNKNOWN, f"kernel {ev!r}/{ex!r} is not the current {RESIDUAL_EVALUATOR_VERSION!r}/{EXCHANGE_ENERGY_VERSION!r}"
```

A legacy residual snapshot can only record `EXCHANGE_ENERGY_VERSION` (v3_aux is an auxiliary record, handled above), so widening the accepted set changes no legacy classification.

- [ ] **Step 4: Implement in `gareus/query.py` `segment_eligibility`**

```python
def segment_eligibility(run_dir) -> dict:
    """Per-segment sample eligibility from ``windows/<segment>.json`` (spec F04).

    Returns ``{segment_id: {"eligibility": ..., "reason": ...}}`` for every segment in
    ``segments.json``; segments without a window snapshot are ``aux_unpersisted`` in an auxiliary
    run (Stage C ruling H2), else ``unknown`` when the run's latest snapshot says residual, else
    ``not_applicable``. The samples manifest is read only for auxiliary snapshots and for
    snapshot-less segments (raw JSON, never validated; ruling B5).
    """
    from . import kernel_identity as _ki

    run_dir = Path(run_dir)
    seg_json = run_dir / "segments.json"
    if not seg_json.exists():
        return {}
    segs = json.loads(seg_json.read_text(encoding="utf-8"))
    out = {}
    latest_residual = False
    snapshots: dict = {}
    for seg in segs:
        p = run_dir / "windows" / f"{seg['segment_id']}.json"
        if p.exists():
            snapshots[str(seg["segment_id"])] = json.loads(p.read_text(encoding="utf-8"))
            latest_residual = str((snapshots[str(seg["segment_id"])] or {}).get("cv2_type")) == "residual-torsion-pc"
    run_is_aux = None

    def _run_is_aux() -> bool:
        nonlocal run_is_aux
        if run_is_aux is None:
            manifest = read_json_file(run_dir / "run_manifest.json", {}) or {}
            run_is_aux = (any(_ki.is_aux_kernel_record((s or {}).get("kernel_identity")) for s in snapshots.values())
                          or (isinstance(manifest, dict) and _ki.is_aux_kernel_record(manifest.get("method_settings"))))
        return run_is_aux

    for seg in segs:
        seg_id = str(seg["segment_id"])
        if seg_id in snapshots:
            snap = snapshots[seg_id] or {}
            payload = (_ki.raw_sample_payload_schema(run_dir, seg_id)
                       if _ki.is_aux_kernel_record(snap.get("kernel_identity")) else None)
            status, reason = _ki.classify_segment_kernel(snap, sample_payload_schema=payload)
        elif _run_is_aux() or ((_ki.raw_sample_payload_schema(run_dir, seg_id) or {}).get("schema")
                               == _ki.AUX_SAMPLES_PAYLOAD_SCHEMA):
            status, reason = (_ki.ELIGIBLE_AUX_UNPERSISTED,
                              "auxiliary run segment without a window snapshot: its state table is unknown (repair the run)")
        elif latest_residual:
            status, reason = _ki.ELIGIBLE_UNKNOWN, "no window snapshot for a residual-CV run"
        else:
            status, reason = _ki.ELIGIBLE_NOT_APPLICABLE, "no window snapshot; not a residual-CV run"
        out[seg_id] = {"eligibility": status, "reason": reason}
    return out
```

`read_json_file` is already imported in `query.py` (line 28). The module-attribute calls (`_ki.raw_sample_payload_schema`) let the B5 test observe that no legacy read happens. Snapshots are now read once instead of twice; the classification of every legacy segment is unchanged.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `tests/test_aux_eligibility.py tests/test_aux_cv_kernel_identity.py tests/test_query.py` plus every other file listed by `grep -l "classify_segment_kernel\|segment_eligibility" tests/*.py`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add gareus/kernel_identity.py gareus/query.py tests/test_aux_eligibility.py tests/test_aux_cv_kernel_identity.py
git commit -m "feat(cvaux): auxiliary segments become verified only with persisted features and a frozen table"
```

---

### Task 8: Strict exporter evaluates auxiliary energies offline; frozen v2 snapshots; legacy pins

**Files:**
- Create: `gareus/auxiliary_cv/offline.py` (`parity_context`, `parity_violation`, `aux_z_from_samples`, `exclusion_report`)
- Modify: `gareus/correctness/export.py:63-163` (`build_export_arrays`), `:189` (`export_fixed_state_npz`)
- Modify: `gareus/query.py:377-395` (`reconstruct_bias_matrix(aux_z=None)` pass-through)
- Modify: `gareus/store.py` (`WindowSnapshot.snapshot(state_definition=None, phase_kind="production", equilibrium_analysis_eligible=None)`)
- Test: `tests/test_aux_export.py`, `tests/test_aux_window_snapshot.py`

**Interfaces:**
- Produces:
  - `parity_context(windows, beta) -> dict(beta, k_max_kcal, centers)`.
  - `parity_violation(stored, recomputed, *, beta, k_max_kcal: float, centers) -> np.ndarray`. It returns the per-row conservative reduced-energy disagreement `beta * KJ_PER_KCAL * k * |dz| * (max_c |z_off - c| + |dz|)`, with NaN where either side is nonfinite. The exact bound uses |dz|/2, so this one is conservative.
  - `aux_z_from_samples(samples, schema, models, *, beta, k_max_kcal, centers, parity_reduced_tol: float | np.ndarray) -> dict[str, np.ndarray]`. `parity_reduced_tol` may be per-row, chosen from each row's segment precision.
  - `exclusion_report(...)`, unchanged from v1.
  - `build_export_arrays(..., aux_sample_schema=None, aux_segment_precision: Mapping[str, str] | None = None, on_incomplete="refuse", time_block_steps=None)`.
  - `WindowSnapshot.snapshot(..., state_definition=None, phase_kind="production", equilibrium_analysis_eligible=None)`.
- Exporter rules:
  - With active aux rows, the samples need the schema torsion columns, and `aux_sample_schema` must cover every active model.
  - `aux_segment_precision` must give a precision for every selected segment, else `IntegrityError("... precision not recorded ...")`.
  - z is computed from the torsions for **every** row, including ordinary-origin rows. The tolerance is `PARITY_TOLERANCE[precision of the row's segment]`.
  - `on_incomplete="refuse"` (default) raises on any incomplete row. `"exclude_and_report"` filters every per-sample array, recomputes `N_k`, and records `exclusion_report_json` plus `manifest["exclusion_report"]`, `manifest["n_kept"]` and `manifest["n_excluded"]`. `manifest["n_samples"]` stays the input count.
  - The torsion and z columns enter `source_arrays`.
  - Legacy v1 output is unchanged: the input signature is pinned at `4bb8e35b…` (re-verified at HEAD `d4415b1`).
- Snapshot rules:
  - With `state_definition`, `cv1_type` must equal `state_definition["cv1"]["kind"]`, and `cv2_type` must equal `state_definition["cv2"]["kind"]` (or both are None). Otherwise `ValueError`. This keeps `classify_segment_kernel`'s residual F01 gate from being bypassed (C19).
  - The payload is `freeze_snapshot(...)` plus `kernel_identity`, written by `write_frozen_snapshot` (immutable).
  - Legacy snapshots are byte-identical: sha pins `45a1984c…` (no identity) and `23b5638e…` (with identity).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_aux_window_snapshot.py
import hashlib
import json

import pytest

from aux_c_fixture import definition, rows
from aux_cv_fixture import model_payload
from gareus.auxiliary_cv.model import AuxModel
from gareus.correctness._io import IntegrityError
from gareus.correctness.state_identity import validate_fixed_state_segments

MODEL = AuxModel.from_mapping(model_payload([(0, 1, 2, 3)], [1.0, 0.5], offset=0.1))


def _defn():
    return definition(rows([(0.2, 10.0, 0.0, 0.0, "ordinary"), (0.2, 10.0, 2.0, 1.0, "auxiliary")],
                           MODEL.model_sha256), MODEL)


def test_legacy_snapshot_bytes_pinned(tmp_path):
    from gareus.store import WindowSnapshot
    WindowSnapshot(tmp_path).snapshot("seg_001", [{"window_id": 0, "center1": 0.1, "k1": 1.0}], "contacts", None)
    assert hashlib.sha256((tmp_path / "windows" / "seg_001.json").read_bytes()).hexdigest() == \
        "45a1984cff98cceb61f95a4ecaba684571e0a172f691f9f61769670d2f1d8cda"
    WindowSnapshot(tmp_path).snapshot("seg_002", [{"window_id": 0, "center1": 0.1, "k1": 1.0}], "contacts", None,
                                      kernel_identity={"kernel_identity_version": "kernel_identity_v1", "digest": "d"})
    assert hashlib.sha256((tmp_path / "windows" / "seg_002.json").read_bytes()).hexdigest() == \
        "23b5638ed577d59eab82e811c101b30e8d13063397551e18590f1e7053840af8"


def test_aux_snapshot_is_a_frozen_v2_snapshot(tmp_path):
    from gareus.query import load_windows
    from gareus.store import WindowSnapshot
    WindowSnapshot(tmp_path).snapshot("seg_001", [], "contacts", None, state_definition=_defn(),
                                      equilibrium_analysis_eligible=True)
    payload = json.loads((tmp_path / "windows" / "seg_001.json").read_text())
    table = validate_fixed_state_segments({"seg_001": payload})
    assert table.definition["aux_models"]
    assert load_windows(tmp_path, "seg_001")[1]["aux_k"] == 2.0


def test_aux_snapshot_cv_kinds_must_match_definition(tmp_path):
    from gareus.store import WindowSnapshot
    with pytest.raises(ValueError, match="cv2"):
        WindowSnapshot(tmp_path).snapshot("seg_001", [], "contacts", "residual-torsion-pc", state_definition=_defn(),
                                          equilibrium_analysis_eligible=True)
    with pytest.raises(ValueError, match="cv1"):
        WindowSnapshot(tmp_path).snapshot("seg_001", [], "distance", None, state_definition=_defn(),
                                          equilibrium_analysis_eligible=True)


def test_aux_snapshot_is_immutable(tmp_path):
    from gareus.store import WindowSnapshot
    s = WindowSnapshot(tmp_path)
    s.snapshot("seg_001", [], "contacts", None, state_definition=_defn(), equilibrium_analysis_eligible=True)
    s.snapshot("seg_001", [], "contacts", None, state_definition=_defn(), equilibrium_analysis_eligible=True)
    with pytest.raises(IntegrityError, match="immutable"):
        s.snapshot("seg_001", [], "contacts", None, state_definition=_defn(), equilibrium_analysis_eligible=False)
```

```python
# tests/test_aux_export.py
import json

import numpy as np
import pytest

from aux_c_fixture import definition, rows
from aux_cv_fixture import model_payload
from gareus.auxiliary_cv.evaluate import z_from_dihedrals
from gareus.auxiliary_cv.model import AuxModel
from gareus.auxiliary_cv.offline import (aux_z_from_samples, exclusion_report, parity_context,
                                         parity_violation)
from gareus.auxiliary_cv.sample_schema import AuxSampleSchema, _basis_sha
from gareus.correctness._io import IntegrityError
from gareus.correctness.export import R_KJ_MOL_K, build_export_arrays
from gareus.correctness.state_identity import freeze_snapshot, make_state_definition

QUADS = ((0, 1, 2, 3), (1, 2, 3, 4))
LABELS = ("phi-A1", "psi-A1")
MODEL = AuxModel.from_mapping(model_payload(list(QUADS), [1.0, 0.0, 0.0, -0.7], offset=0.2,
                                            blocks=["phi", "psi"]))
SCHEMA = AuxSampleSchema(QUADS, LABELS, (MODEL.model_sha256,), _basis_sha(QUADS, LABELS))
BETA = 1.0 / (R_KJ_MOL_K * 300.0)          # the exporter's own constant (C8)
VIEW = {"kind": "immutable", "boundary_id": "b1"}
PREC = {"seg_001": "double"}


def _defn():
    return definition(rows([(0.2, 10.0, 0.0, 0.0, "ordinary"), (0.2, 10.0, 2.0, 0.5, "auxiliary")],
                           MODEL.model_sha256), MODEL)


def _samples(n=8, seed=0):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(-np.pi, np.pi, size=(n, 2))
    z = z_from_dihedrals(theta, MODEL)
    return {"cv1": rng.uniform(0, 0.4, n), "window_id": np.array([0, 1] * (n // 2)),
            "segment_id": np.array(["seg_001"] * n), "step": np.arange(n) * 100,
            "replica": np.array([0, 1] * (n // 2)), "gamd_lambda": np.zeros(n),
            "tor_000": theta[:, 0], "tor_001": theta[:, 1], "aux_z_00": z}


def _snap():
    return {"seg_001": freeze_snapshot("seg_001", _defn(), equilibrium_analysis_eligible=True, phase_kind="production")}


def _export(s, **kw):
    return build_export_arrays(s, _snap(), BETA, sample_view=VIEW, aux_sample_schema=SCHEMA,
                               aux_segment_precision=PREC, **kw)


def test_ordinary_origin_rows_get_auxiliary_cross_energy_in_lambda_zero_table():
    s = _samples()
    u = _export(s)["umbrella_reduced_bias_nk"]
    z = z_from_dihedrals(np.stack([s["tor_000"], s["tor_001"]], axis=1), MODEL)
    np.testing.assert_allclose(u[:, 1] - u[:, 0], BETA * 4.184 * 0.5 * 2.0 * (z - 0.5) ** 2, rtol=1e-12)
    assert np.all(u[s["window_id"] == 0, 1] > u[s["window_id"] == 0, 0])


def test_missing_features_fail_closed():
    s = _samples()
    for key in ("tor_000", "tor_001", "aux_z_00"):
        s.pop(key)
    with pytest.raises(IntegrityError, match="auxiliary features"):
        _export(s)


def test_schema_and_precision_required():
    with pytest.raises(IntegrityError, match="aux_sample_schema"):
        build_export_arrays(_samples(), _snap(), BETA, sample_view=VIEW)
    other = AuxSampleSchema(QUADS, LABELS, ("f" * 64,), _basis_sha(QUADS, LABELS))
    with pytest.raises(IntegrityError, match="model"):
        build_export_arrays(_samples(), _snap(), BETA, sample_view=VIEW, aux_sample_schema=other,
                            aux_segment_precision=PREC)
    with pytest.raises(IntegrityError, match="precision"):
        build_export_arrays(_samples(), _snap(), BETA, sample_view=VIEW, aux_sample_schema=SCHEMA)


def test_parity_tolerance_follows_segment_precision():
    s = _samples()
    s["aux_z_00"] = s["aux_z_00"] + 3e-6           # GPU-scale disagreement
    with pytest.raises(IntegrityError, match="parity"):
        _export(s)                                    # double: 1e-6 reduced
    a = build_export_arrays(s, _snap(), BETA, sample_view=VIEW, aux_sample_schema=SCHEMA,
                            aux_segment_precision={"seg_001": "mixed"})
    assert a["umbrella_reduced_bias_nk"].shape == (8, 2)


def test_parity_violation_is_conservative_and_dimensionless():
    du = parity_violation(np.array([1.0 + 1e-3]), np.array([1.0]), beta=BETA, k_max_kcal=2.0, centers=[0.5])
    exact = BETA * 4.184 * 0.5 * 2.0 * abs((1.0 + 1e-3 - 0.5) ** 2 - 0.5 ** 2)
    assert du[0] >= exact and du[0] < 2.5 * exact


def test_refuse_vs_exclude_and_report():
    s = _samples(n=8)
    s["tor_000"] = s["tor_000"].copy(); s["aux_z_00"] = s["aux_z_00"].copy()
    s["tor_000"][[0, 2]] = np.nan
    s["aux_z_00"][[0, 2]] = np.nan
    with pytest.raises(IntegrityError, match="Incomplete"):
        _export(s)
    a = _export(s, on_incomplete="exclude_and_report", time_block_steps=400)
    assert a["umbrella_reduced_bias_nk"].shape[0] == 6 and int(a["N_k"].sum()) == 6
    rep = json.loads(str(a["exclusion_report_json"].item()))
    assert rep["n_excluded"] == 2 and rep["by_origin_state"] == {"0": 2}
    assert rep["by_carrier"] == {"0": 2} and rep["by_time_block"] == {"seg_001:0": 2}
    assert rep["by_z_range"][MODEL.model_sha256]["nonfinite"] == 2
    assert rep["by_structural_group"].startswith("unavailable")
    man = json.loads(str(a["export_manifest_json"].item()))
    assert man["n_samples"] == 8 and man["n_kept"] == 6 and man["n_excluded"] == 2
    # Board condition 9: every exported per-sample array is aligned with the kept rows.
    n_kept = man["n_kept"]
    for key, value in a.items():
        arr = np.asarray(value)
        if arr.ndim >= 1 and arr.shape[0] in (8, n_kept) and key not in ("N_k",):
            assert arr.shape[0] == n_kept, f"exported array {key} has {arr.shape[0]} rows, expected {n_kept}"


def test_registry_lookup_is_integrity_error_not_keyerror():
    from gareus.auxiliary_cv.offline import registry_model
    with pytest.raises(IntegrityError, match="missing from the frozen state definition"):
        registry_model({"aux_models": {}}, "f" * 64, where="test")


def test_exclusion_report_bins_finite_z():
    excluded = np.array([True, False, True, False])
    rep = exclusion_report(excluded, origin_ids=np.array([1, 1, 1, 0]), replicas=np.array([3, 3, 4, 4]),
                           steps=np.array([0, 100, 900, 950]), segment_ids=np.array(["s"] * 4),
                           aux_z={"e" * 64: np.array([0.0, 1.0, 2.0, 3.0])}, time_block_steps=500, z_bins=3)
    assert rep["by_time_block"] == {"s:0": 1, "s:1": 1} and rep["by_carrier"] == {"3": 1, "4": 1}
    assert sum(rep["by_z_range"]["e" * 64]["counts"]) == 2 and rep["fraction"] == 0.5
    assert rep["above_audit_threshold"] is True


def test_legacy_v1_export_signature_pinned():
    rows_v1 = [{"window_id": 0, "center1": 0.2, "k1": 10.0, "center2": 0.0, "k2": 0.0, "gamd_lambda": 0.0}]
    d = make_state_definition(rows_v1, physical_system_sha256="a" * 64, ensemble="NVT", temperature_k=300.0,
                              fixed_box_vectors_nm=[[3, 0, 0], [0, 3, 0], [0, 0, 3]],
                              cv1={"kind": "contacts", "units": "dimensionless", "definition": {"r0": 4.5}}, cv2=None)
    snaps = {"seg_001": freeze_snapshot("seg_001", d, equilibrium_analysis_eligible=True, phase_kind="production")}
    s = {"cv1": np.array([0.1, 0.3]), "window_id": np.array([0, 0]), "segment_id": np.array(["seg_001"] * 2)}
    a = build_export_arrays(s, snaps, BETA, sample_view=VIEW)
    man = json.loads(str(a["export_manifest_json"].item()))
    assert man["input_signature"] == "4bb8e35b09a426d57c9723fc06ef70d9864b2553d4af28a49942a5e3be707b91"
    assert sorted(a) == ["N_k", "column_window_ids", "cv_A", "export_manifest_json", "original_window_id",
                         "segment_id", "umbrella_reduced_bias_nk", "window"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `tests/test_aux_export.py tests/test_aux_window_snapshot.py`
Expected: FAIL `ModuleNotFoundError: No module named 'gareus.auxiliary_cv.offline'` and `TypeError: snapshot() got an unexpected keyword argument 'state_definition'`. The two legacy pin tests already pass.

- [ ] **Step 3: Implement `offline.py` (first part)**

```python
# gareus/auxiliary_cv/offline.py
"""Offline auxiliary energies from stored features, parity, exclusion audits and pooling guards."""
from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

import numpy as np

from ..correctness._io import IntegrityError
from ..correctness.bias import KJ_PER_KCAL
from .evaluate import z_from_dihedrals
from .model import AuxModel
from .sample_schema import AuxSampleSchema

AUDIT_THRESHOLD_FRACTION = 1e-3


def _float_column(samples, name, n) -> np.ndarray:
    raw = samples[name]
    arr = np.ma.filled(np.ma.asarray(raw, dtype=np.float64), np.nan) if np.ma.isMaskedArray(raw) \
        else np.asarray(raw, dtype=np.float64)
    if arr.shape != (n,):
        raise IntegrityError(f"sample column {name} has shape {arr.shape}, expected ({n},)")
    return arr


def registry_model(definition: Mapping[str, Any], sha: str, *, where: str) -> AuxModel:
    """The frozen registry's model for ``sha``; a diagnostic IntegrityError, never a KeyError (board 7)."""
    registry = (definition or {}).get("aux_models") or {}
    if sha not in registry:
        raise IntegrityError(f"{where}: auxiliary model {sha} is recorded in the sample schema but missing "
                             f"from the frozen state definition's aux_models registry {sorted(registry)}")
    return AuxModel.from_mapping(registry[sha])


def parity_context(windows, beta: float) -> dict[str, Any]:
    """beta, the strongest active k per model and that model's active centres, from a state table."""
    k_max: dict[str, float] = {}
    centers: dict[str, list[float]] = {}
    for w in windows:
        if float(w.get("aux_k", 0.0)) > 0:
            sha = w["aux_model_sha256"]
            k_max[sha] = max(k_max.get(sha, 0.0), float(w["aux_k"]))
            centers.setdefault(sha, []).append(float(w["aux_center"]))
    return {"beta": float(beta), "k_max_kcal": k_max, "centers": centers}


def parity_violation(stored, recomputed, *, beta: float, k_max_kcal: float, centers: Sequence[float]) -> np.ndarray:
    """Conservative reduced-energy disagreement per row (exact bound uses |dz|/2; this uses |dz|)."""
    stored = np.asarray(stored, dtype=np.float64)
    recomputed = np.asarray(recomputed, dtype=np.float64)
    dz = np.abs(stored - recomputed)
    cs = np.asarray(list(centers) or [0.0], dtype=np.float64)
    dev = np.max(np.abs(recomputed[:, None] - cs[None, :]), axis=1)
    return float(beta) * KJ_PER_KCAL * float(k_max_kcal) * dz * (dev + dz)


def aux_z_from_samples(samples: Mapping[str, Any], schema: AuxSampleSchema,
                       models: Mapping[str, AuxModel], *, beta: float,
                       k_max_kcal: Mapping[str, float], centers: Mapping[str, Sequence[float]],
                       parity_reduced_tol) -> dict[str, np.ndarray]:
    n = len(np.asarray(samples["cv1"]))
    missing = [c for c in schema.torsion_columns if c not in samples]
    if missing:
        raise IntegrityError(f"historical rows lack auxiliary features (columns {missing[:3]}...); "
                             "they cannot enter a pool with active auxiliary states")
    tol = np.broadcast_to(np.asarray(parity_reduced_tol, dtype=np.float64), (n,))
    theta = np.stack([_float_column(samples, c, n) for c in schema.torsion_columns], axis=1)
    out: dict[str, np.ndarray] = {}
    for col, sha in zip(schema.z_columns, schema.model_shas):
        if sha not in models:
            raise IntegrityError(f"no aux model loaded for schema model {sha}")
        z = z_from_dihedrals(theta[:, schema.model_basis_index(models[sha])], models[sha]).astype(np.float64)
        if col in samples:
            stored = _float_column(samples, col, n)
            if np.any(np.isfinite(stored) != np.isfinite(z)):
                raise IntegrityError(f"stored {col} fails offline parity against model {sha} (finite mismatch)")
            k = float(k_max_kcal.get(sha, 0.0))
            both = np.isfinite(stored) & np.isfinite(z)
            if k > 0 and np.any(both):
                du = parity_violation(stored[both], z[both], beta=beta, k_max_kcal=k, centers=centers[sha])
                bad = du > tol[both]
                if np.any(bad):
                    raise IntegrityError(f"stored {col} fails offline parity against model {sha}: "
                                         f"max reduced-energy disagreement {du.max():.3g} > tolerance "
                                         f"{float(tol[both][bad].min()):g} on {int(bad.sum())} rows")
        out[sha] = z
    return out
```

Append `exclusion_report` exactly as in v1. It is unchanged; its code is:

```python
def exclusion_report(excluded, *, origin_ids, replicas, steps, segment_ids, aux_z: Mapping[str, np.ndarray],
                     time_block_steps: int, z_bins: int = 10) -> dict[str, Any]:
    excluded = np.asarray(excluded, dtype=bool)
    if int(time_block_steps) <= 0:
        raise IntegrityError("exclusion_report needs a positive time_block_steps")
    def counts(values):
        vals, cnt = np.unique(np.asarray(values)[excluded].astype(str), return_counts=True)
        return {str(v): int(c) for v, c in zip(vals, cnt)}
    blocks = np.asarray([f"{s}:{int(t) // int(time_block_steps)}" for s, t in zip(segment_ids, steps)])
    by_z = {}
    for sha, z in aux_z.items():
        z = np.asarray(z, dtype=np.float64)
        finite_all = z[np.isfinite(z)]
        lo, hi = (float(finite_all.min()), float(finite_all.max())) if finite_all.size else (0.0, 1.0)
        edges = np.linspace(lo, hi if hi > lo else lo + 1.0, int(z_bins) + 1)
        sel = z[excluded]
        hist, _ = np.histogram(sel[np.isfinite(sel)], bins=edges)
        by_z[sha] = {"edges": edges.tolist(), "counts": hist.astype(int).tolist(),
                     "nonfinite": int(np.count_nonzero(~np.isfinite(sel)))}
    n, k = int(excluded.size), int(excluded.sum())
    return {"n_total": n, "n_excluded": k, "fraction": (k / n) if n else 0.0,
            "by_origin_state": counts(origin_ids), "by_carrier": counts(replicas),
            "by_time_block": counts(blocks), "by_z_range": by_z,
            "by_structural_group": "unavailable: no frozen structural labels before Stage D",
            "audit_threshold_fraction": AUDIT_THRESHOLD_FRACTION,
            "above_audit_threshold": bool(n and k / n > AUDIT_THRESHOLD_FRACTION)}
```

- [ ] **Step 4: Implement the exporter changes in `gareus/correctness/export.py`**

Signature:

```python
def build_export_arrays(
    samples: Mapping[str, Any], snapshots: Mapping[str, Mapping[str, Any]], beta: float,
    *, sample_view: Mapping[str, Any], envelope_factory: Callable | None = None,
    reconstruct: Callable = reconstruct_bias_matrix, aux_sample_schema=None,
    aux_segment_precision: Mapping[str, str] | None = None,
    on_incomplete: str = "refuse", time_block_steps: int | None = None,
) -> dict[str, np.ndarray]:
    ...   # body: today's, with the edits below
```

Right after `envelope = ...`:

```python
    if on_incomplete not in ("refuse", "exclude_and_report"):
        raise IntegrityError(f"on_incomplete must be 'refuse' or 'exclude_and_report', got {on_incomplete!r}")
    active_aux = sorted({w["aux_model_sha256"] for w in table.windows if w.get("aux_k", 0.0) > 0})
    aux_z = None
    if active_aux:
        from ..auxiliary_cv.offline import aux_z_from_samples, parity_context
        from ..auxiliary_cv.sample_schema import PARITY_TOLERANCE
        if aux_sample_schema is None:
            raise IntegrityError("active auxiliary states need the run's aux_sample_schema to evaluate z")
        uncovered = [sha for sha in active_aux if sha not in aux_sample_schema.model_shas]
        if uncovered:
            raise IntegrityError(f"aux_sample_schema does not record active model(s) {uncovered}")
        precision = dict(aux_segment_precision or {})
        unknown_prec = [s for s in selected_ids if precision.get(s) not in PARITY_TOLERANCE]
        if unknown_prec:
            raise IntegrityError(f"platform precision not recorded for segment(s) {unknown_prec}; "
                                 "parity tolerance cannot be chosen")
        row_tol = np.asarray([PARITY_TOLERANCE[precision[s]] for s in segments], dtype=np.float64)
        from ..auxiliary_cv.offline import registry_model
        models = {sha: registry_model(table.definition, sha, where="strict exporter")
                  for sha in aux_sample_schema.model_shas}
        aux_z = aux_z_from_samples(samples, aux_sample_schema, models,
                                   parity_reduced_tol=row_tol, **parity_context(table.windows, beta))
    extra = {"aux_z": aux_z} if aux_z is not None else {}
```

`AuxModel.from_mapping` accepts the registry's identity body (Stage A D2). If Stage A's identity body lacks `label`/`provenance`, those fields are optional in `from_mapping`.

Change the reconstruct call to `reconstruct(cv1, cv2, list(table.windows), beta, v_pep=pep, v_dih=dih, envelope=envelope, **extra)`. Replace the `complete = ...` block with:

```python
    complete = np.all(np.isfinite(matrix), axis=1)
    report = None
    if not complete.all():
        counts = np.bincount(origins[~complete], minlength=len(table.windows)).tolist()
        if on_incomplete == "refuse":
            raise IntegrityError(
                f"Incomplete cross-state energies for {int((~complete).sum())} samples; "
                f"counts by origin column={counts}. No rows were silently discarded."
            )
        if time_block_steps is None or "step" not in samples or "replica" not in samples:
            raise IntegrityError("exclude_and_report needs step, replica and time_block_steps for the audit")
        from ..auxiliary_cv.offline import exclusion_report
        report = exclusion_report(~complete, origin_ids=origin_ids, replicas=_integer_vector(samples["replica"], "replica", n),
                                  steps=_integer_vector(samples["step"], "step", n), segment_ids=segments,
                                  aux_z=aux_z or {}, time_block_steps=int(time_block_steps))
```

Do **not** move the existing `for name in ("step", "replica")` loop (moving it would change the NPZ member order of legacy exports; preflight §2 T8). Insert the keep block after the `source_arrays` block (`export.py:141-147`, including the aux features below) and before `signature_payload = {...}`, so `source_arrays` hashes the complete pre-exclusion input (board condition 9) while `arrays`, and so `array_hashes`, hold the kept rows plus `exclusion_report_json`:

```python
    if report is not None:
        keep = complete
        for key in ("cv_A", "window", "original_window_id", "segment_id", "umbrella_reduced_bias_nk",
                    "secondary_cv", "step", "replica"):
            if key in arrays:
                arrays[key] = arrays[key][keep]
        arrays["N_k"] = np.bincount(arrays["window"], minlength=len(table.windows)).astype(np.int64)
        arrays["exclusion_report_json"] = np.asarray(json_bytes(report).decode("utf-8"))
```

Remaining exporter edits:
- In `source_arrays`, right after its existing `step`/`replica` loop, add the features when `aux_z is not None` (legacy `source_arrays` are unchanged):

```python
    if aux_z is not None:
        from ..auxiliary_cv.offline import _float_column
        for name in aux_sample_schema.torsion_columns + aux_sample_schema.z_columns:
            if name in samples:
                source_arrays[name] = _float_column(samples, name, n)
        for sha, z in aux_z.items():
            source_arrays[f"aux_z:{sha}"] = z
        source_arrays["aux_tolerance"] = row_tol
```
- **Board condition 9 resolution: `source_arrays` is deliberately not masked.**
  - In the exporter (`gareus/correctness/export.py:141-153`), `source_arrays` holds only input-provenance hashes. `manifest["source_arrays"] = {key: array_hash(value)}` hashes the complete pre-exclusion input, as `cv1`, `origins` and `segments` already do. No per-sample array from it is written to the NPZ.
  - Masking it would make the signature describe a sample set that never existed on disk.
  - The rows actually exported are the ones in the `keep` loop. `exclusion_report` together with `n_kept`/`n_excluded` ties the hashed input to the exported subset.
  - The alignment test below pins that every exported per-sample array has length `n_kept`.
- When `report is not None`, add `"exclusion_report": report`, `"n_kept": int(complete.sum())` and `"n_excluded": int((~complete).sum())` to `manifest`. Add `"on_incomplete": on_incomplete` to `signature_payload` only when it differs from `"refuse"`.
- Thread `aux_sample_schema`, `aux_segment_precision`, `on_incomplete` and `time_block_steps` through `export_fixed_state_npz`.

- [ ] **Step 5: Implement `WindowSnapshot.snapshot(state_definition=...)` and the query pass-through**

In `gareus/store.py` `WindowSnapshot.snapshot`, add the keywords `state_definition=None, phase_kind: str = "production", equilibrium_analysis_eligible=None`. At the top of the body:

```python
        if state_definition is not None:
            from .correctness.state_identity import freeze_snapshot, write_frozen_snapshot
            cv1_def = state_definition.get("cv1") or {}
            cv2_def = state_definition.get("cv2")
            if (cv1_def.get("kind") if cv1_def else None) != cv1_type:
                raise ValueError(f"cv1_type {cv1_type!r} != state definition cv1 kind {cv1_def.get('kind')!r}")
            if (cv2_def["kind"] if cv2_def else None) != cv2_type:
                raise ValueError(f"cv2_type {cv2_type!r} != state definition cv2 kind "
                                 f"{cv2_def['kind'] if cv2_def else None!r}")
            if equilibrium_analysis_eligible is None:
                raise ValueError("an auxiliary-capable snapshot must state its equilibrium eligibility explicitly")
            frozen = freeze_snapshot(segment_id, state_definition,
                                     equilibrium_analysis_eligible=bool(equilibrium_analysis_eligible),
                                     phase_kind=phase_kind)
            if kernel_identity is not None:
                frozen["kernel_identity"] = dict(kernel_identity)
            write_frozen_snapshot(self._win_dir / f"{segment_id}.json", frozen)
            return
```

`validate_fixed_state_segments` does not inspect extra top-level snapshot keys, so `kernel_identity` is accepted. In `gareus/query.py`, add `aux_z=None` to `reconstruct_bias_matrix` and pass it on to `_strict_bias`.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `tests/test_aux_export.py tests/test_aux_window_snapshot.py tests/test_query.py tests/test_query_reconstruct_bias_matrix_nan_guard.py tests/test_store.py`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add gareus/auxiliary_cv/offline.py gareus/correctness/export.py gareus/query.py gareus/store.py tests/test_aux_export.py tests/test_aux_window_snapshot.py
git commit -m "feat(cvaux): strict exporter evaluates aux energies with precision-chosen parity; frozen v2 snapshots"
```

---

### Task 9: Loaders evaluate or refuse; every other pooling path refuses auxiliary runs

**Files:**
- Modify: `gareus/kernel_identity.py` (import-light detection: `snapshot_has_aux`, `aux_snapshot_hits`, `run_has_aux`, `refuse_aux_snapshots`, `AuxPoolingRefused`)
- Modify: `gareus/auxiliary_cv/offline.py` (`segment_aux_schemas`, `segment_aux_runtime`, `classify_pool_segments`, `pool_aux_segments`)
- Modify: `gareus/mbar_analysis/loaders.py`:
  - `_refuse_aux_run:243` extended (ruling B3) and given `depth=`; in `load_npz:264` its call moves above `_load_merged_arrays`; `load_csv:388` keeps its call;
  - `load_parquet:537-665`, new keywords `exclude_segments_without_aux_features=False`, `allow_ineligible_aux_segments=False`, `aux_on_incomplete="refuse"`, `aux_time_block_steps=None`;
  - `load_data:1078`: adaptive branch guarded (ruling H4); plain branch forces Parquet for an auxiliary run and refuses auxiliary round directories.
- Modify: `gareus/query.py:398` (`export_analysis_arrays_npz`)
- Modify: `gareus/mbar_analysis/loaders_union_parquet.py:303` (`load_parquet_adaptive_union`), `gareus/adaptive_production.py:3715` (`build_union_state_mbar_inputs`: `adaptive_dir` **and every `pilot_dirs` entry**)
- Test: `tests/test_aux_loaders.py`

**Interfaces:**
- Produces in `gareus/kernel_identity.py` (pure JSON, no NumPy, no `gareus.auxiliary_cv` import, so legacy loads stay off the aux package):
  - `class AuxPoolingRefused(RuntimeError)`;
  - `snapshot_has_aux(payload) -> bool`: True when the snapshot's `kernel_identity` is an auxiliary record (`is_aux_kernel_record`), its frozen `state_definition` has `aux_models`, or any row carries `aux_model_sha256` or `aux_k > 0` (Stage B D5a rows included);
  - `aux_snapshot_hits(root, *, depth=0) -> list[str]`: snapshot paths with auxiliary states under the fixed-depth globs `windows/*.json` (depth 0), plus `*/windows/*.json` and `*/*/windows/*.json` (depth 2: adaptive phases and their top-ups, e.g. `epoch_002/topup_003_4812000/windows`, measured on `RUNS/chignolin_7`). Never `rglob`: c10's union build runs every epoch over trees holding checkpoint generations and trajectories (L1);
  - `run_has_aux(prod) -> bool` for one run directory: `run_manifest.json` `method_settings` is an auxiliary record, or `aux_snapshot_hits(prod, depth=0)`, or any `samples/<seg>/` raw payload is `atlas-aux-samples-v1` (H2);
  - `refuse_aux_snapshots(root, *, where, depth=2) -> None`, raising `AuxPoolingRefused`.
- Produces in `offline.py` (imported lazily, auxiliary branches only):
  - `segment_aux_schemas(run_dir) -> dict[str, AuxSampleSchema | None]`, `segment_aux_runtime(run_dir) -> dict[str, dict | None]` (the `runtime` block per segment);
  - `classify_pool_segments(run_dir, segment_ids) -> dict[str, list[str]]` with keys `aux` (auxiliary snapshot, eligibility VERIFIED), `unpersisted` (auxiliary snapshot, `aux_unpersisted`: Stage B era), `ineligible_kernel` (auxiliary snapshot, any other eligibility), `no_snapshot`, `legacy` (snapshot without auxiliary states);
  - `pool_aux_segments(prod, samples, beta, meta, *, exclude_segments_without_aux_features, allow_ineligible_aux_segments) -> (samples, windows, aux_z | None)`.
- `load_parquet` rules, when `run_has_aux(prod)`:
  1. **Rows.** Samples are loaded with `include_ineligible=True`, so the auxiliary branch sees every segment it must judge (the segment-status rules still apply). Legacy calls are unchanged (`include_ineligible=False`).
  2. **Segment classes** (`classify_pool_segments` over the segments that hold rows):
     - `legacy` → `IntegrityError` naming the segments, **no override** (ruling H3: a campaign directory mixing v2 and v3_aux segments is split manually);
     - `no_snapshot` → `IntegrityError`, **no override** (ruling H2: a damaged auxiliary run cannot be analysed without repair);
     - `ineligible_kernel` → `IntegrityError`, no override;
     - `unpersisted` → `IntegrityError` unless `exclude_segments_without_aux_features=True`, in which case their rows are dropped and a note gives the row count per segment.
  3. **Fixed state.** `validate_fixed_state_segments(..., require_eligible=not allow_ineligible_aux_segments)` over the `aux` segments' snapshots gives the table (C10). Its windows replace `load_windows(prod)`. Ineligible segments raise unless allowed; when allowed, a note records "engineering analysis, not an equilibrium estimate".
  4. **Features and z.** The `aux` segments must share one `AuxSampleSchema`. z comes from `aux_z_from_samples` with per-row tolerances from `segment_aux_runtime` (missing runtime → `IntegrityError`).
  5. **Incomplete rows.** After `u_nk`, before `clean()`, any row with a nonfinite `u_nk` row or cv is `aux_on_incomplete="refuse"` → `IntegrityError` with counts by origin. `"exclude_and_report"` attaches `exclusion_report(...)` to `meta["aux_exclusion_report"]` and needs `aux_time_block_steps`. Then `clean()` drops exactly those rows (C7, S10).
  6. `meta["aux_models"]` and `meta["aux_feature_segments"]` are recorded.
- Without auxiliary evidence, `load_parquet` output is unchanged (it pays the read-only `run_has_aux` scan, accepted by ruling).
- Refusals for auxiliary runs:
  - `load_npz`, `load_csv`: Stage B's `_refuse_aux_run` (RuntimeError, pinned), extended with `snapshot_has_aux` so frozen v2 snapshots and D5a rows are seen; in `load_npz` it runs before `_load_merged_arrays`, so an aux Parquet-only run refuses instead of raising `FileNotFoundError` (ruling B3);
  - `load_data` adaptive branch: `_refuse_aux_run(prod, {}, depth=2)` first (rulings B3/H4, covers `load_union_npz`, `load_epoch_csv_adaptive` and `load_parquet_adaptive_union`);
  - `load_data` plain branch: `source="auto"` selects Parquet for an auxiliary run; an explicit `npz`/`csv` request reaches the refusing loader; round augmentation refuses an auxiliary round directory and otherwise is skipped with a note;
  - `export_analysis_arrays_npz`, `load_parquet_adaptive_union`, `build_union_state_mbar_inputs` for `adaptive_dir` and every pilot dir (C9): `refuse_aux_snapshots` (`AuxPoolingRefused`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_aux_loaders.py
import json

import numpy as np
import pytest

from aux_c_fixture import definition, rows
from aux_cv_fixture import model_payload
from gareus.auxiliary_cv.evaluate import z_from_dihedrals
from gareus.auxiliary_cv.model import AuxModel
from gareus.auxiliary_cv.offline import segment_aux_schemas
from gareus.auxiliary_cv.sample_schema import AuxSampleSchema, _basis_sha
from gareus.correctness._io import IntegrityError
from gareus.kernel_identity import AuxPoolingRefused, refuse_aux_snapshots, snapshot_has_aux

QUADS = ((0, 1, 2, 3), (1, 2, 3, 4))
LABELS = ("phi-A1", "psi-A1")
MODEL = AuxModel.from_mapping(model_payload(list(QUADS), [1.0, 0.0, 0.0, -0.7], offset=0.2, blocks=["phi", "psi"]))
SCHEMA = AuxSampleSchema(QUADS, LABELS, (MODEL.model_sha256,), _basis_sha(QUADS, LABELS))
RUNTIME = {"platform": "Reference", "precision": "double"}
KI = {"kernel_identity_version": "kernel_identity_v1", "exchange_energy_version": "state_bias_matrix_v3_aux",
      "aux_model_sha256": MODEL.model_sha256}
STAGE_B_ROWS = [{"window_id": 0, "center1": 0.2, "k1": 10.0, "gamd_lambda": 0.0,
                 "aux_model_sha256": None, "aux_center": 0.0, "aux_k": 0.0},
                {"window_id": 1, "center1": 0.2, "k1": 10.0, "gamd_lambda": 0.0,
                 "aux_model_sha256": MODEL.model_sha256, "aux_center": 0.5, "aux_k": 2.0}]


def _defn():
    return definition(rows([(0.2, 10.0, 0.0, 0.0, "ordinary"), (0.2, 10.0, 2.0, 0.5, "auxiliary")],
                           MODEL.model_sha256), MODEL)


def _write_segment(run, seg, aux, rng, *, nan_rows=()):
    from gareus.store import ParquetSampleWriter
    w = ParquetSampleWriter(run / "samples" / seg, aux_schema=SCHEMA if aux else None,
                            aux_runtime=RUNTIME if aux else None)
    for i in range(6):
        theta = rng.uniform(-np.pi, np.pi, size=2)
        if i in nan_rows:
            theta[0] = np.nan
        kw = {}
        if aux:
            kw = dict(torsions=theta, aux_z=[float(z_from_dihedrals(theta[None, :], MODEL)[0])])
        w.write_sample(step=100 * i, replica=i % 2, window_id=i % 2, cv1=0.1, cv2=None, potential=0.0,
                       boost_total=None, boost_dihedral=None, boost_nonbonded=None, **kw)
    w.close()


def _run(tmp_path, *, historical=False, eligible=True, nan_rows=()):
    """historical: False | "legacy" (v2 kernel) | "stage_b" (aux kernel, no features) | "aux_no_snapshot"."""
    from gareus.store import SegmentRegistry, WindowSnapshot
    rng = np.random.default_rng(1)
    reg = SegmentRegistry(tmp_path)
    segs = []
    if historical:
        s0 = reg.open_segment("run", None, 1)
        _write_segment(tmp_path, s0, historical == "aux_no_snapshot", rng)
        reg.close_segment(s0, 600)
        if historical == "legacy":
            WindowSnapshot(tmp_path).snapshot(s0, [{"window_id": 0, "center1": 0.2, "k1": 10.0},
                                                   {"window_id": 1, "center1": 0.2, "k1": 10.0}], "contacts", None)
        elif historical == "stage_b":
            WindowSnapshot(tmp_path).snapshot(s0, STAGE_B_ROWS, "contacts", None, kernel_identity=KI)
        segs.append(s0)
    s1 = reg.open_segment("run", segs[-1] if segs else None, 1)
    _write_segment(tmp_path, s1, True, rng, nan_rows=nan_rows); reg.close_segment(s1, 1200)
    WindowSnapshot(tmp_path).snapshot(s1, [], "contacts", None, kernel_identity=KI, state_definition=_defn(),
                                      equilibrium_analysis_eligible=eligible,
                                      phase_kind="production" if eligible else "pilot")
    (tmp_path / "gareus_metadata.json").write_text(json.dumps({"temperature_K": 300.0}))
    return segs + [s1]


def test_segment_schemas(tmp_path):
    s0, s1 = _run(tmp_path, historical="legacy")
    sch = segment_aux_schemas(tmp_path)
    assert sch[s0] is None and sch[s1] == SCHEMA


def test_snapshot_has_aux_sees_kernel_identity_and_stage_b_rows():
    assert snapshot_has_aux({"kernel_identity": {"exchange_energy_version": "state_bias_matrix_v3_aux"}})
    assert snapshot_has_aux({"windows": [{"window_id": 0, "aux_k": 0.0, "aux_model_sha256": None, "aux_center": 0.0}]})
    assert not snapshot_has_aux({"windows": [{"window_id": 0, "center1": 0.1, "k1": 1.0}]})
    assert not snapshot_has_aux([])


def test_detection_never_imports_the_aux_package_for_a_legacy_run(tmp_path):
    import subprocess
    import sys
    import textwrap
    from gareus.store import WindowSnapshot
    WindowSnapshot(tmp_path).snapshot("seg_001", [{"window_id": 0, "center1": 0.2, "k1": 1.0}], "contacts", None)
    code = textwrap.dedent(f"""
        import sys
        from gareus.kernel_identity import refuse_aux_snapshots, run_has_aux
        assert not run_has_aux({str(tmp_path)!r})
        refuse_aux_snapshots({str(tmp_path)!r}, where="x")
        assert "gareus.auxiliary_cv" not in sys.modules
    """)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_load_parquet_evaluates_aux_for_every_row(tmp_path):
    from gareus.mbar_analysis.loaders import load_parquet
    _run(tmp_path)
    d = load_parquet(tmp_path)
    assert np.isfinite(d.u_nk).all() and np.all(d.u_nk[:, 1] >= d.u_nk[:, 0])
    assert d.meta["aux_models"] == [MODEL.model_sha256]


def test_legacy_kernel_segment_fails_closed_even_with_the_opt_out(tmp_path):
    """H3: a v2 segment in an auxiliary campaign directory is refused, never silently excluded."""
    from gareus.mbar_analysis.loaders import load_parquet
    s0, _s1 = _run(tmp_path, historical="legacy")
    for kw in ({}, {"exclude_segments_without_aux_features": True}):
        with pytest.raises(IntegrityError, match=s0):
            load_parquet(tmp_path, **kw)


def test_snapshotless_segment_fails_closed_even_with_the_opt_out(tmp_path):
    """H2: a segment without a window snapshot in an auxiliary run cannot be analysed without repair."""
    from gareus.mbar_analysis.loaders import load_parquet
    s0, _s1 = _run(tmp_path, historical="aux_no_snapshot")
    for kw in ({}, {"exclude_segments_without_aux_features": True}):
        with pytest.raises(IntegrityError, match=s0):
            load_parquet(tmp_path, **kw)


def test_stage_b_segment_refused_unless_explicitly_excluded(tmp_path):
    from gareus.mbar_analysis.loaders import load_parquet
    s0, _s1 = _run(tmp_path, historical="stage_b")
    with pytest.raises(IntegrityError, match=s0):
        load_parquet(tmp_path)
    d = load_parquet(tmp_path, exclude_segments_without_aux_features=True)
    assert d.u_nk.shape[0] == 6
    assert any(s0 in note and "6" in note for note in d.meta["load_notes"])


def test_ineligible_aux_segment_refused_unless_allowed(tmp_path):
    from gareus.mbar_analysis.loaders import load_parquet
    _run(tmp_path, eligible=False)
    with pytest.raises(IntegrityError, match="eligib"):
        load_parquet(tmp_path)
    d = load_parquet(tmp_path, allow_ineligible_aux_segments=True)
    assert any("engineering" in n for n in d.meta["load_notes"])


def test_nan_aux_rows_refused_or_excluded_with_report_never_silently(tmp_path):
    from gareus.mbar_analysis.loaders import load_parquet
    _run(tmp_path, nan_rows=(1, 3))
    with pytest.raises(IntegrityError, match="Incomplete"):
        load_parquet(tmp_path)
    d = load_parquet(tmp_path, aux_on_incomplete="exclude_and_report", aux_time_block_steps=300)
    assert d.u_nk.shape[0] == 4
    assert d.meta["aux_exclusion_report"]["n_excluded"] == 2


def test_other_pooling_paths_refuse_aux(tmp_path):
    from gareus.mbar_analysis.loaders import load_csv, load_data, load_npz
    from gareus.query import export_analysis_arrays_npz
    _run(tmp_path)
    with pytest.raises(AuxPoolingRefused, match="fixed-state exporter"):
        refuse_aux_snapshots(tmp_path, where="adaptive union")
    with pytest.raises(AuxPoolingRefused, match="fixed-state exporter"):
        export_analysis_arrays_npz(tmp_path, beta=0.4)
    with pytest.raises(RuntimeError, match="auxiliary"):         # Stage B guard, hoisted (B3)
        load_npz(tmp_path)
    with pytest.raises(RuntimeError, match="auxiliary"):
        load_csv(tmp_path)
    (tmp_path / "samples.csv").write_text("step\n1\n")          # tempt auto-selection
    d = load_data(tmp_path, None, source="auto", no_augment=True)
    assert d.meta["aux_models"] == [MODEL.model_sha256]


def test_load_data_default_path_works_for_aux_run_without_rounds(tmp_path):
    from gareus.mbar_analysis.loaders import load_data
    _run(tmp_path)
    d = load_data(tmp_path, None, source="auto")                 # default no_augment=False (board 4)
    assert d.meta["aux_models"] == [MODEL.model_sha256]


def test_load_data_refuses_aux_round_dir_and_skips_plain_round_dir(tmp_path):
    from gareus.mbar_analysis.loaders import load_data
    _run(tmp_path)
    rd = tmp_path / "adaptive_feedback_round_01"
    (rd / "analysis_chunks").mkdir(parents=True)
    (rd / "umbrella_windows.csv").write_text("window\n0\n")
    (rd / "analysis_chunks" / "chunk_000.npz").write_bytes(b"")
    d = load_data(tmp_path, None, source="auto")
    assert any("augmentation skipped" in n for n in d.meta.get("load_notes", []))
    _run(rd)                                                     # the round itself carries aux states
    with pytest.raises(AuxPoolingRefused, match="adaptive round augmentation"):
        load_data(tmp_path, None, source="auto")


def test_adaptive_branch_refuses_aux_phase_dirs(tmp_path):
    """H4: the adaptive branch of load_data (union NPZ, epoch CSV, union Parquet) refuses aux phases."""
    from gareus.mbar_analysis.loaders import load_data
    ap = tmp_path / "adaptive_production"
    (ap / "epoch_000").mkdir(parents=True)
    _run(ap / "epoch_000")
    (ap / "state_registry.csv").write_text("state_id\n0\n")
    with pytest.raises(RuntimeError, match="auxiliary"):
        load_data(tmp_path, None)


def test_union_builder_scans_pilot_dirs(tmp_path):
    from gareus.adaptive_production import build_union_state_mbar_inputs
    pilot = tmp_path / "pilot"; pilot.mkdir()
    _run(pilot)
    (tmp_path / "ap").mkdir()
    with pytest.raises(AuxPoolingRefused, match="fixed-state exporter"):
        build_union_state_mbar_inputs(tmp_path / "ap", registry=None, pilot_dirs=[pilot])


def test_refuse_is_silent_without_aux(tmp_path):
    from gareus.store import WindowSnapshot
    WindowSnapshot(tmp_path).snapshot("seg_001", [{"window_id": 0, "center1": 0.2, "k1": 1.0}], "contacts", None)
    refuse_aux_snapshots(tmp_path, where="x")
```

`build_union_state_mbar_inputs(..., registry=None)` is safe in the test because the refusals are the first statements after `adaptive_dir = Path(adaptive_dir)`. In `test_adaptive_branch_refuses_aux_phase_dirs`, `state_registry.csv` plus `epoch_000/samples/**/*.parquet` makes `prod_dir_of` resolve to `adaptive_production` (`loaders.py:668-690`).

- [ ] **Step 2: Run the tests to verify they fail**

Run: `tests/test_aux_loaders.py`
Expected: FAIL `ImportError: cannot import name 'AuxPoolingRefused'`

- [ ] **Step 3: Implement the detection helpers in `gareus/kernel_identity.py`** (after `raw_sample_payload_schema`)

```python
class AuxPoolingRefused(RuntimeError):
    """An auxiliary-CV run reached a pooling path that cannot reconstruct its bias (CVaux Stage C)."""


#: Window-snapshot globs per depth: the run itself, then adaptive phases and their top-ups
#: (adaptive_production/epoch_NNN/windows, adaptive_production/epoch_NNN/topup_*/windows). Never rglob.
_SNAPSHOT_GLOBS = ("windows/*.json", "*/windows/*.json", "*/*/windows/*.json")


def _positive(value) -> bool:
    try:
        return float(value or 0.0) > 0.0
    except (TypeError, ValueError):
        return False


def snapshot_has_aux(payload) -> bool:
    """A windows/<segment>.json payload that carries auxiliary states (kernel identity, frozen v2
    aux_models, or any row with auxiliary fields: Stage B D5a rows included)."""
    if not isinstance(payload, dict):
        return False
    if is_aux_kernel_record(payload.get("kernel_identity")):
        return True
    state = payload.get("state_definition")
    state = state if isinstance(state, dict) else {}
    if state.get("aux_models"):
        return True
    rows = list(payload.get("windows") or []) + list(state.get("windows") or [])
    return any(isinstance(r, dict) and ("aux_model_sha256" in r or _positive(r.get("aux_k"))) for r in rows)


def aux_snapshot_hits(root, *, depth: int = 0) -> list:
    import json
    from pathlib import Path
    root = Path(root)
    hits = []
    for pattern in _SNAPSHOT_GLOBS[: int(depth) + 1]:
        for path in sorted(root.glob(pattern)):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if snapshot_has_aux(payload):
                hits.append(str(path))
    return hits


def run_has_aux(prod) -> bool:
    """Auxiliary evidence for ONE run directory (read-only JSON; legacy output unchanged)."""
    import json
    from pathlib import Path
    prod = Path(prod)
    try:
        manifest = json.loads((prod / "run_manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        manifest = {}
    if isinstance(manifest, dict) and is_aux_kernel_record(manifest.get("method_settings")):
        return True
    if aux_snapshot_hits(prod, depth=0):
        return True
    samples = prod / "samples"
    if samples.is_dir():
        for seg_dir in sorted(p for p in samples.iterdir() if p.is_dir()):
            if (raw_sample_payload_schema(prod, seg_dir.name) or {}).get("schema") == AUX_SAMPLES_PAYLOAD_SCHEMA:
                return True
    return False


def refuse_aux_snapshots(root, *, where: str, depth: int = 2) -> None:
    hits = aux_snapshot_hits(root, depth=depth)
    if hits:
        raise AuxPoolingRefused(f"{where}: auxiliary states pool only through the strict fixed-state exporter or "
                                f"load_parquet in the MVP (spec Sections 8/14); found {hits[:3]}")
```

- [ ] **Step 4: Implement the pooling helpers in `offline.py`**

```python
def segment_aux_schemas(run_dir) -> dict[str, "AuxSampleSchema | None"]:
    return {seg: (AuxSampleSchema.from_payload(p) if p else None) for seg, p in _segment_payloads(run_dir).items()}


def segment_aux_runtime(run_dir) -> dict[str, "dict | None"]:
    from .sample_schema import runtime_from_payload
    return {seg: (runtime_from_payload(p) if p else None) for seg, p in _segment_payloads(run_dir).items()}


def _segment_payloads(run_dir) -> dict[str, "dict | None"]:
    from pathlib import Path
    from ..parquet_manifest import load_manifest
    out = {}
    root = Path(run_dir) / "samples"
    if not root.exists():
        return out
    for seg_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        manifest = load_manifest(seg_dir, expected_kind="samples")      # verify_hashes=False (default)
        out[seg_dir.name] = (manifest or {}).get("payload_schema")
    return out


POOL_CLASSES = ("aux", "unpersisted", "ineligible_kernel", "no_snapshot", "legacy")
_ALWAYS_REFUSED = (
    ("legacy", "ran without auxiliary states: a campaign directory mixing v2 and v3_aux segments cannot be "
               "pooled (ruling H3); split the run directory"),
    ("no_snapshot", "have no window snapshot in an auxiliary run: their state table is unknown (ruling H2); "
                    "repair the run"),
    ("ineligible_kernel", "have an ineligible kernel (segment_eligibility); they cannot enter an equilibrium pool"),
)


def classify_pool_segments(run_dir, segment_ids) -> dict[str, list[str]]:
    from ..kernel_identity import ELIGIBLE_AUX_UNPERSISTED, ELIGIBLE_VERIFIED, snapshot_has_aux
    from ..query import load_windows_metadata, segment_eligibility
    eligibility = segment_eligibility(run_dir)
    out: dict[str, list[str]] = {key: [] for key in POOL_CLASSES}
    for seg in sorted(str(s) for s in segment_ids):
        snap = load_windows_metadata(run_dir, seg)
        status = (eligibility.get(seg) or {}).get("eligibility")
        if not snap:
            out["no_snapshot"].append(seg)
        elif not snapshot_has_aux(snap):
            out["legacy"].append(seg)
        elif status == ELIGIBLE_VERIFIED:
            out["aux"].append(seg)
        elif status == ELIGIBLE_AUX_UNPERSISTED:
            out["unpersisted"].append(seg)
        else:
            out["ineligible_kernel"].append(seg)
    return out


def pool_aux_segments(prod, samples, beta, meta, *, exclude_segments_without_aux_features: bool = False,
                      allow_ineligible_aux_segments: bool = False):
    """load_parquet's auxiliary branch: refuse or exclude segment classes, one eligible fixed state,
    one sample schema, offline z with per-segment parity. Returns (samples, windows, aux_z or None)."""
    from ..correctness.state_identity import validate_fixed_state_segments
    from ..query import load_windows_metadata
    from .sample_schema import PARITY_TOLERANCE
    seg_col = np.asarray(samples["segment_id"]).astype(str)
    groups = classify_pool_segments(prod, set(seg_col.tolist()))
    for key, why in _ALWAYS_REFUSED:
        if groups[key]:
            raise IntegrityError(f"segments {groups[key]} {why}")
    if groups["unpersisted"]:
        if not exclude_segments_without_aux_features:
            raise IntegrityError(f"segments {groups['unpersisted']} ran auxiliary states without stored features "
                                 "(Stage B era) and cannot be evaluated; pass exclude_segments_without_aux_features=True "
                                 "to drop them with a note")
        n = seg_col.size
        keep = ~np.isin(seg_col, groups["unpersisted"])
        dropped = {s: int(np.count_nonzero(seg_col == s)) for s in groups["unpersisted"]}
        samples = {k: (v[keep] if hasattr(v, "__len__") and len(v) == n else v) for k, v in samples.items()}
        seg_col = seg_col[keep]
        meta.setdefault("load_notes", []).append(f"excluded auxiliary segments without stored features (rows): {dropped}")
    aux_segs = groups["aux"]
    if not aux_segs or seg_col.size == 0:
        raise IntegrityError(f"{prod}: no auxiliary segment with stored features remains to pool")
    snaps = {s: load_windows_metadata(prod, s) for s in aux_segs}
    try:
        table = validate_fixed_state_segments(snaps, require_eligible=not allow_ineligible_aux_segments)
    except IntegrityError as exc:
        raise IntegrityError(f"auxiliary segments are not one eligible fixed state: {exc}") from exc
    if allow_ineligible_aux_segments:
        meta.setdefault("load_notes", []).append(
            "engineering analysis: ineligible auxiliary segments allowed, not an equilibrium estimate")
    schemas, runtimes = segment_aux_schemas(prod), segment_aux_runtime(prod)
    distinct = {schemas.get(s) for s in aux_segs}
    if None in distinct or len(distinct) != 1:
        raise IntegrityError(f"auxiliary sample schema differs between (or is missing from) segments {aux_segs}")
    schema = distinct.pop()
    bad_rt = [s for s in aux_segs if (runtimes.get(s) or {}).get("precision") not in PARITY_TOLERANCE]
    if bad_rt:
        raise IntegrityError(f"platform precision not recorded for segment(s) {bad_rt}")
    row_tol = np.asarray([PARITY_TOLERANCE[runtimes[s]["precision"]] for s in seg_col], dtype=np.float64)
    # Every schema model must be in the frozen registry: a silent filter would drop an active term.
    models = {sha: registry_model(table.definition, sha, where=f"load_parquet ({prod})") for sha in schema.model_shas}
    windows = [dict(w) for w in table.windows]
    aux_z = None
    if any(float(w.get("aux_k", 0.0)) > 0 for w in windows):
        aux_z = aux_z_from_samples(samples, schema, models, parity_reduced_tol=row_tol, **parity_context(windows, beta))
    meta["aux_models"] = list(schema.model_shas)
    meta["aux_feature_segments"] = list(aux_segs)
    return samples, windows, aux_z
```

`v[keep]` keeps masked arrays masked (no `np.asarray`).

- [ ] **Step 5: Implement `load_parquet` and the guards** (`gareus/mbar_analysis/loaders.py`)

`load_parquet` signature:

```python
def load_parquet(prod: Path, *, exclude_segments_without_aux_features: bool = False,
                 allow_ineligible_aux_segments: bool = False, aux_on_incomplete: str = 'refuse',
                 aux_time_block_steps: Optional[int] = None) -> Data:
    ...   # body: today's, with the edits below
```

Replace `samples = load_samples(prod)` with:

```python
    from gareus.kernel_identity import run_has_aux
    aux_present = run_has_aux(prod)
    samples = load_samples(prod, include_ineligible=aux_present)
```

After `temp, beta = infer_temp_beta(prod, meta)` and before `cv = samples['cv1'].astype(np.float64)`:

```python
    aux_z = None
    if aux_present:
        from gareus.auxiliary_cv.offline import pool_aux_segments
        samples, windows, aux_z = pool_aux_segments(
            prod, samples, beta, meta,
            exclude_segments_without_aux_features=exclude_segments_without_aux_features,
            allow_ineligible_aux_segments=allow_ineligible_aux_segments)
```

Every later array read comes from the possibly filtered `samples`, and `windows` is the frozen table's. Change the bias call to `u_nk = reconstruct_bias_matrix(cv, cv2_for_nk, windows_for_nk, beta, v_pep=v_pep, v_dih=v_dih, envelope=envelope, meta=meta, aux_z=aux_z)` (Task 8 pass-through; `aux_z=None` is the legacy behaviour). Right after it:

```python
    if aux_present:
        from gareus.correctness._io import IntegrityError
        incomplete = ~(np.isfinite(cv) & np.all(np.isfinite(u_nk), axis=1))
        if incomplete.any():
            counts = np.bincount(window.astype(np.int64)[incomplete], minlength=len(windows)).tolist()
            if aux_on_incomplete == 'refuse':
                raise IntegrityError(f'Incomplete cross-state energies for {int(incomplete.sum())} auxiliary-run samples; '
                                     f'counts by origin={counts}; pass aux_on_incomplete="exclude_and_report"')
            if aux_on_incomplete != 'exclude_and_report' or not aux_time_block_steps:
                raise IntegrityError('aux_on_incomplete must be "refuse" or "exclude_and_report" (with aux_time_block_steps)')
            from gareus.auxiliary_cv.offline import exclusion_report
            meta['aux_exclusion_report'] = exclusion_report(
                incomplete, origin_ids=window, replicas=replica, steps=step,
                segment_ids=np.asarray(samples['segment_id']).astype(str), aux_z=aux_z or {},
                time_block_steps=int(aux_time_block_steps))
            meta.setdefault('load_notes', []).append(
                f'auxiliary run: {int(incomplete.sum())} incomplete rows excluded with report (aux_exclusion_report)')
```

`clean()` then drops exactly those rows: same mask, `np.isfinite(d.cv) & finite_rows(d.u_nk)`.

**N_k after exclusion (board condition 6), verified against current code:**
- `clean()` (`gareus/mbar_analysis/data.py:249-262`) slices `d.window` with the same mask it applies to `d.u_nk`, along with `cv`, `cv2`, `replica`, `step` and `boost_kj`, the optional per-sample arrays and `_epoch_source`.
- `Data` stores no separate `N_k`. Every solver recomputes it from the filtered origins at solve time (`np.bincount(window[(window >= 0) & (window < K)], minlength=K)` in `gareus/mbar_analysis/solvers.py`).
- So origin counts cannot drift from the retained rows. Pin this in `test_aux_loaders.py`:

```python
def test_origin_counts_follow_exclusion(tmp_path):
    from gareus.mbar_analysis.data import clean
    from gareus.mbar_analysis.loaders import load_parquet
    _run(tmp_path, nan_rows=(1, 3))   # two rows with NaN torsions
    d = clean(load_parquet(tmp_path, aux_on_incomplete="exclude_and_report", aux_time_block_steps=300))
    K = d.u_nk.shape[1]
    nk = np.bincount(d.window, minlength=K)
    assert int(nk.sum()) == d.u_nk.shape[0] == d.cv.size
    assert np.isfinite(d.u_nk).all()
```

Guards:
- `_refuse_aux_run` (ruling B3: extend Stage B's guard, keep its RuntimeError contract):

```python
def _refuse_aux_run(prod: Path, meta: dict, *, depth: int = 0) -> None:
    """Refuse an auxiliary-CV run in every loader except load_parquet (CVaux Stage B ruling C3, Stage C B3/H4).

    Their stored matrices / reconstructed biases omit the auxiliary restraint term. Fires on records naming
    the v3_aux kernel or an auxiliary model digest, and on any window snapshot carrying auxiliary states
    (kernel_identity.snapshot_has_aux: kernel identity, frozen v2 aux_models, D5a rows); depth=2 also scans
    phase directories for the adaptive branch. Legacy runs: read-only JSON, unchanged output.
    """
    from gareus.kernel_identity import AuxPoolingRefused, aux_snapshot_hits, is_aux_kernel_record
    manifest = rjson(prod/'run_manifest.json', {})
    records = [meta, (meta or {}).get('kernel_identity') if isinstance(meta, dict) else None,
               manifest.get('method_settings') if isinstance(manifest, dict) else None]
    hits = aux_snapshot_hits(prod, depth=depth)
    if hits or any(is_aux_kernel_record(r) for r in records):
        raise AuxPoolingRefused(f'{prod}: auxiliary-CV run (state_bias_matrix_v3_aux / aux model recorded'
                                f'{"; " + hits[0] if hits else ""}); its bias pools only through load_parquet or '
                                'the strict fixed-state exporter (CVaux Stage C).')
```

- `load_npz`: move its first lines so the guard runs before any data read:

```python
def load_npz(prod: Path) -> Data:
    meta=rjson(prod/'analysis_arrays_metadata.json',{}); meta.update(rjson(prod/'umbrella_pymbar_metadata.json',{}))
    _refuse_aux_run(prod, meta)  # CVaux Stage B (ruling C3), hoisted in Stage C (B3): before any data read
    arr, load_notes = _load_merged_arrays(prod)
    if not arr:
        raise FileNotFoundError(f'No samples in {prod}/analysis_arrays.npz or analysis_chunks/')
```

  (the old `meta=...` and `_refuse_aux_run` lines below `if not arr:` are deleted; legacy behaviour and output are identical).
- `load_csv`: unchanged (its guard is already its second line).
- `load_data`, adaptive branch: first statement inside `if prod.name == 'adaptive_production':` is `_refuse_aux_run(prod, {}, depth=2)` (H4).
- `load_data`, plain branch: right after `requested=str(source or 'auto').strip().lower()`:

```python
    from gareus.kernel_identity import refuse_aux_snapshots, run_has_aux
    aux_run = run_has_aux(prod)
    if aux_run and requested == 'auto':
        requested = 'parquet'      # the only per-segment feature-checked loader
```

  and replace the augmentation tail

```python
    if not no_augment:
        run_dir = prod.parent if prod.name == 'final_production' else prod
        if _find_gareus_round_dirs(run_dir):
            d = _augment_with_adaptive_rounds(d, run_dir)
            if out is not None: d.out_dir = Path(out)
    return d
```

  with (the `elif` body is the old code, unchanged):

```python
    run_dir = prod.parent if prod.name == 'final_production' else prod
    if aux_run:
        if not no_augment:
            _rounds = _find_gareus_round_dirs(run_dir)      # adaptive_feedback_round_* with analysis chunks only
            for _rd in _rounds:
                refuse_aux_snapshots(Path(_rd), where=f'adaptive round augmentation ({Path(_rd).name})', depth=0)
            if _rounds:
                d.meta.setdefault('load_notes', []).append(
                    f'Auxiliary run: adaptive round augmentation skipped for {len(_rounds)} round dir(s); their rows '
                    'carry no auxiliary features and cannot join an auxiliary pool.')
    elif not no_augment:
        if _find_gareus_round_dirs(run_dir):
            d = _augment_with_adaptive_rounds(d, run_dir)
            if out is not None: d.out_dir = Path(out)
    return d
```

  `refuse_aux_snapshots` is never called on the run directory itself (its own snapshots are auxiliary; board condition 4).
- `gareus/query.py` `export_analysis_arrays_npz`: first body line `from .kernel_identity import refuse_aux_snapshots; refuse_aux_snapshots(Path(run_dir), where="legacy analysis_arrays.npz export", depth=0)`.
- `load_parquet_adaptive_union` (`loaders_union_parquet.py:303`): first body line `from gareus.kernel_identity import refuse_aux_snapshots; refuse_aux_snapshots(Path(adaptive_dir), where="adaptive union loader")`.
- `build_union_state_mbar_inputs` (`adaptive_production.py:3715`), right after `adaptive_dir = Path(adaptive_dir)`:

```python
    from .kernel_identity import refuse_aux_snapshots
    refuse_aux_snapshots(adaptive_dir, where="adaptive union build")
    for _pilot in (pilot_dirs or []):
        refuse_aux_snapshots(Path(_pilot), where=f"adaptive union build (pilot dir {_pilot})")
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `tests/test_aux_loaders.py tests/test_aux_cv_kernel_identity.py tests/test_query.py tests/test_lambda_ladder_mbar.py tests/test_low_memory_complete.py tests/test_loader_masked_cv2_nan.py tests/test_aux_cv_state_table.py`
Expected: PASS (`test_aux_cv_kernel_identity.py::test_npz_and_csv_loaders_refuse_aux_runs` keeps passing: `AuxPoolingRefused` is a `RuntimeError` whose message contains "auxiliary").

- [ ] **Step 7: Commit**

```bash
git add gareus/kernel_identity.py gareus/auxiliary_cv/offline.py gareus/mbar_analysis/loaders.py gareus/mbar_analysis/loaders_union_parquet.py gareus/adaptive_production.py gareus/query.py tests/test_aux_loaders.py
git commit -m "feat(cvaux): load_parquet pools one eligible fixed aux state with audited exclusion; other paths refuse"
```

---

### Task 10: Duplicate Hamiltonians: separate and merged MBAR agree (library utility)

**Files:**
- Modify: `gareus/auxiliary_cv/offline.py` (`merge_identical_hamiltonians`)
- Test: `tests/test_aux_mbar_duplicates.py`

**Scope decision (C18/S13).** MVP pools shams as **separate origins**, which spec Section 4.3 allows. `merge_identical_hamiltonians` is a library utility for the Stage D pilot analysis and is deliberately **not wired** into the exporter or the loaders here. Its test shows that the two approaches recover the same target weights on unequal, independent data. Columns may merge only when bitwise identical.

**Interfaces:**
- Consumes: Stage A `hamiltonian_sha256(definition, window_id)`; `gareus.adaptive.mbar_solve.solve_rows(u_nk, window) -> (f, logw)`.
- Produces: `merge_identical_hamiltonians(u_nk, origins, hamiltonian_ids) -> (merged_u, merged_origins, groups)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_aux_mbar_duplicates.py
import numpy as np
import pytest

from aux_c_fixture import definition, rows
from aux_cv_fixture import model_payload
from gareus.adaptive.mbar_solve import solve_rows
from gareus.auxiliary_cv.model import AuxModel
from gareus.auxiliary_cv.offline import merge_identical_hamiltonians
from gareus.correctness._io import IntegrityError
from gareus.correctness.state_identity import hamiltonian_sha256

MODEL = AuxModel.from_mapping(model_payload([(0, 1, 2, 3)], [1.0, 0.5], offset=0.1))


def _defn():
    spec = [(-1.0, 4.0, 0.0, 0.0, "ordinary"), (0.0, 4.0, 0.0, 0.0, "ordinary"),
            (1.0, 4.0, 0.0, 0.0, "ordinary"), (0.0, 4.0, 0.0, 0.0, "sham")]
    return definition(rows(spec, MODEL.model_sha256), MODEL)


def _data(seed=4, counts=(1500, 2000, 1500, 900)):
    """Exact samples of harmonic umbrellas on a flat line; the sham (col 3) has its OWN, fewer samples."""
    rng = np.random.default_rng(seed)
    beta, k = 1.0, 4.0
    centers = np.array([-1.0, 0.0, 1.0, 0.0])
    x = np.concatenate([rng.normal(c, 1.0 / np.sqrt(beta * k), n) for c, n in zip(centers, counts)])
    origins = np.repeat(np.arange(4), counts)
    u = 0.5 * beta * k * (x[:, None] - centers[None, :]) ** 2
    return x, u, origins


def test_sham_and_parent_share_hamiltonian_id_and_merge():
    d = _defn()
    ids = [hamiltonian_sha256(d, w) for w in range(4)]
    assert ids[1] == ids[3] and len(set(ids)) == 3
    _x, u, origins = _data()
    merged_u, merged_origins, groups = merge_identical_hamiltonians(u, origins, ids)
    assert merged_u.shape == (u.shape[0], 3) and groups == [[0], [1, 3], [2]]
    f_sep, logw_sep = solve_rows(u, origins)
    f_mrg, logw_mrg = solve_rows(merged_u, merged_origins)
    assert f_sep[3] - f_sep[1] == pytest.approx(0.0, abs=1e-8)
    np.testing.assert_allclose(f_sep[[0, 1, 2]] - f_sep[0], f_mrg - f_mrg[0], atol=1e-8)
    w_sep = np.exp(logw_sep - logw_sep.max()); w_sep /= w_sep.sum()
    w_mrg = np.exp(logw_mrg - logw_mrg.max()); w_mrg /= w_mrg.sum()
    np.testing.assert_allclose(w_sep, w_mrg, rtol=1e-8, atol=1e-14)


def test_merge_refuses_non_identical_columns_with_same_id():
    _x, u, origins = _data()
    u = u.copy()
    u[0, 3] += 1e-9
    with pytest.raises(IntegrityError, match="bitwise"):
        merge_identical_hamiltonians(u, origins, ["a", "b", "c", "b"])
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `tests/test_aux_mbar_duplicates.py`
Expected: FAIL `ImportError: cannot import name 'merge_identical_hamiltonians'`

- [ ] **Step 3: Implement** (append to `offline.py`)

```python
def merge_identical_hamiltonians(u_nk, origins, hamiltonian_ids):
    """Library utility (Stage D pilot analysis); MVP pools shams as separate origins instead."""
    u = np.asarray(u_nk, dtype=np.float64)
    origins = np.asarray(origins, dtype=np.int64)
    ids = [str(x) for x in hamiltonian_ids]
    if u.ndim != 2 or u.shape[1] != len(ids):
        raise IntegrityError(f"u_nk has {u.shape[1] if u.ndim == 2 else '?'} columns for {len(ids)} Hamiltonian ids")
    groups: list[list[int]] = []
    where: dict[str, int] = {}
    for col, hid in enumerate(ids):
        if hid not in where:
            where[hid] = len(groups)
            groups.append([col])
        else:
            first = groups[where[hid]][0]
            if not np.array_equal(u[:, first], u[:, col], equal_nan=True):
                raise IntegrityError(f"columns {first} and {col} share Hamiltonian {hid} but are not bitwise equal")
            groups[where[hid]].append(col)
    merged = u[:, [g[0] for g in groups]].copy()
    col_to_group = {c: gi for gi, g in enumerate(groups) for c in g}
    merged_origins = np.asarray([col_to_group[int(o)] for o in origins], dtype=np.int64)
    return merged, merged_origins, groups
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `tests/test_aux_mbar_duplicates.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add gareus/auxiliary_cv/offline.py tests/test_aux_mbar_duplicates.py
git commit -m "feat(cvaux): bitwise-checked identical-Hamiltonian merge utility; separate and merged MBAR agree"
```

---

### Task 11: Runtime state definition, adapter, instance-complete tables and CLI changes

**Files:**
- Create: `gareus/auxiliary_cv/runtime_definition.py`
- Modify: `gareus/auxiliary_cv/state_table.py:81` (`load_aux_state_table`: refuse an instance-less table, ruling M2)
- Modify: `gareus/cli.py:1431-1478` (Stage B's `_add_aux_cv_args` / `_validate_aux_cv_args`)
- Modify: Stage B's `tests/test_aux_cv_cli.py` (drop the removed flag and the resume/extend refusal cases)
- Test: `tests/test_aux_runtime_definition.py`

**Interfaces:**
- Consumes:
  - Stage B `AuxRuntime(table, info, force_index, topology_sha256=None)`, `AuxStateTable`, `exchange_energy_version_for_args`;
  - Stage A `make_state_definition(..., aux_models=)`, `normalize_windows`, `_canonical_cv`;
  - `gareus.energy_decomposition.peptide_atom_groups_from_topology`;
  - Task 2 `build_sample_schema`, `runtime_precision`, `runtime_info`;
  - Task 6 `topology_identity_sha256`.
- Produces:
  - `physical_system_sha256(openmm, system) -> str`, the sha of `XmlSerializer.serialize` of a copy of `system` whose default periodic box is set to the unit cell. No existing production helper provides physical-system identity (verified: only swarm's `file_digest(base_system.xml)`, `gareus/swarm/analyze.py:839`). Production calls it on the base system **before** umbrella and aux forces are added (Task 13). **The default box is excluded on purpose:** a fresh run builds the System from the in-memory solvated topology, a resume from `01_solvated_start.pdb` (`gareus/checkpoints.py:77-82`), whose CRYST1 box is rounded; the box identity is `fixed_box_vectors_nm` (NVT) or a sample variable (NPT).
  - `embed_cv_definition(kind, units, raw) -> dict`, a canonical `{kind, units, definition}`. Path-like keys (ending `_path`/`_file`/`_filename`, or named `path`/`file`/`filename`) are replaced by `<stem>_content_sha256` of the file's bytes; `None`/empty values are dropped. A path to a missing file raises `IntegrityError`. The result passes `state_identity._canonical_cv`.
  - `build_runtime_state_definition(*, physical_system_sha256, ensemble, temperature_k, pressure_bar, box_vectors_nm, cv1, cv2, boost, snapshot_rows, aux_table, applied_window_key) -> dict`:
    - rows are the `snapshot_rows` merged with `aux_table.window_rows()`; every row must carry the table's `instance` (v2 requires one on every row; ruling M2 makes loaded tables instance-complete), else `IntegrityError`;
    - it builds the definition with `make_state_definition(..., aux_models={sha: model.to_mapping()})`;
    - **applied-restraint check (L3):** `applied_window_key` is `production._aux_window_order_key(centers_a, k_list, secondary_cv_centers, secondary_cv_k_kcal_list, nrep)`, i.e. the arrays `set_window` applies. The definition rows' `center1/k1/center2/k2` must equal `normalize_windows` of rows built from that key (an inactive axis compares as inactive, whatever placeholder centre it carries; `None` = NaN from the NaN-safe key, Task 12). This is not a tautology: it compares the canonicalised definition with the restraint arrays, not with the rows it was built from. No test-only keyword.
  - `@dataclass(frozen=True) class AuxIORuntime(state_definition, models, force_info, sample_schema, runtime, energy_version, phase_kind, equilibrium_eligible, topology_sha256)`. `topology_sha256` is Stage C's whole-topology `topology_identity_sha256`; Stage B's `canonical_topology_sha256` (model atom map) reaches the checkpoint through the kernel identity digest (`aux_topology_sha256`).
  - `aux_io_runtime(runtime, *, state_definition, topology, args, platform, context) -> AuxIORuntime`:
    - peptide atoms come from `args.pep_gamd_peptide_atoms` when set (Pep-GaMD), else `peptide_atom_groups_from_topology(topology, "all-peptide")[1]` (any run mode, including `--run-mode cmd`; Stage B handoff item);
    - precision comes from `runtime_precision`;
    - `energy_version` comes from `exchange_energy_version_for_args(args)`;
    - `phase_kind`/eligibility come from the CLI.
- `load_aux_state_table` (Stage B) additionally refuses a table whose rows carry no instance columns at all (the existing all-or-none check still refuses a partial table first). The refusal fires when the explicit CSV is loaded, before any Context is built. Consequence, recorded: a sham row can no longer silently become an "ordinary" row of a v2 definition.
- CLI (D6, D10):
  - **Remove:** `--aux-cv-allow-unpersisted`, its validation, the acknowledgement error and the `--resume`/`--extend` refusal. Resume binding moves to Task 12 (production, every `--resume`).
  - **Add:** `--aux-phase-kind {production,pilot,exploration,equilibration}` (default `pilot`) and `--aux-equilibrium-eligible` (store_true).
  - **Validate:** eligibility requires `production` (`freeze_snapshot`'s rule, enforced at parse, as the first check after the `aux_cv_model` early return).
  - **Keep:** every other Stage B refusal.
  - **Recorded args (ruling, same class as C6):** `aux_phase_kind`/`aux_equilibrium_eligible` appear with default values in `resolved_args`/effective config of every job, including non-aux jobs (provenance only, no behaviour). `aux_cv_allow_unpersisted` disappears from them; c10 never ran Stage B code, so no recorded c10 key is affected.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_aux_runtime_definition.py
import pytest

from aux_cv_fixture import model_payload
from gareus.auxiliary_cv.model import AuxModel
from gareus.auxiliary_cv.runtime_definition import (build_runtime_state_definition, embed_cv_definition,
                                                    physical_system_sha256)
from gareus.correctness._io import IntegrityError

MODEL = AuxModel.from_mapping(model_payload([(0, 1, 2, 3)], [1.0, 0.5], offset=0.1))
BOX = [[3, 0, 0], [0, 3, 0], [0, 0, 3]]
ORD = {"state_instance_id": "w0000", "state_role": "ordinary", "spawn_parent_state_id": None,
       "spawn_source_observation": None, "matched_additional_slot_id": None}
AUX = {"state_instance_id": "aux-0", "state_role": "auxiliary", "spawn_parent_state_id": "w0000",
       "spawn_source_observation": None, "matched_additional_slot_id": "slot-0"}
SNAP = [{"window_id": 0, "center1": 0.2, "k1": 10.0, "gamd_lambda": 0.0},
        {"window_id": 1, "center1": 0.2, "k1": 10.0, "gamd_lambda": 0.0}]
KEY = ((0.2, 10.0), (0.2, 10.0))
CV1 = {"kind": "contacts", "units": "dimensionless", "definition": {"r0": 4.5}}


def _table(k=(0.0, 2.0), inst=(ORD, AUX)):
    from gareus.auxiliary_cv.state_table import AuxStateTable
    return AuxStateTable(MODEL, (0.0, 1.5), tuple(k), tuple(inst))


def _build(**kw):
    base = dict(physical_system_sha256="a" * 64, ensemble="NVT", temperature_k=300.0, pressure_bar=None,
                box_vectors_nm=BOX, cv1=CV1, cv2=None, boost=None, snapshot_rows=SNAP, aux_table=_table(),
                applied_window_key=KEY)
    base.update(kw)
    return build_runtime_state_definition(**base)


def test_definition_merges_snapshot_rows_and_aux_table():
    d = _build()
    rows = {r["window_id"]: r for r in d["windows"]}
    assert rows[0]["instance"]["state_role"] == "ordinary" and rows[0]["aux_k"] == 0.0
    assert rows[1]["aux_k"] == 2.0 and rows[1]["instance"]["state_instance_id"] == "aux-0"
    assert list(d["aux_models"]) == [MODEL.model_sha256]


def test_row_without_instance_is_refused():
    with pytest.raises(IntegrityError, match="instance"):
        _build(aux_table=_table(inst=(None, AUX)))


def test_definition_must_match_the_restraints_set_window_applies():
    with pytest.raises(IntegrityError, match="set_window applies"):
        _build(applied_window_key=((0.2, 10.0), (0.2, 11.0)))
    with pytest.raises(IntegrityError, match="set_window applies"):
        _build(applied_window_key=((0.2, 10.0), (0.3, 10.0)))
    # an inactive axis is inactive whatever placeholder centre (or NaN -> None) it carries
    snap = [dict(SNAP[0], k1=0.0), SNAP[1]]
    _build(snapshot_rows=snap, applied_window_key=((None, 0.0), (0.2, 10.0)))


def test_instance_less_table_is_refused_at_load(tmp_path):
    from gareus.auxiliary_cv.state_table import load_aux_state_table
    path = tmp_path / "m.json"
    MODEL.write(path)
    with pytest.raises(IntegrityError, match="instance"):
        load_aux_state_table(path, [{"aux_k_kcal_mol": "0"}, {"aux_k_kcal_mol": "2.0", "aux_center": "1.5"}])


def test_embed_cv_definition_replaces_paths_by_content(tmp_path):
    f = tmp_path / "m.json"; f.write_text("{}")
    out = embed_cv_definition("contact-map", "dimensionless", {"cv1_model_path": str(f), "r0": 8.0, "note_file": None})
    assert "cv1_model_content_sha256" in out["definition"] and "cv1_model_path" not in out["definition"]
    assert "note_file" not in out["definition"]
    with pytest.raises(IntegrityError, match="missing"):
        embed_cv_definition("x", "dimensionless", {"model_file": str(tmp_path / "nope.json")})


def test_physical_system_sha_is_deterministic_and_bias_sensitive():
    import openmm as mm
    from pep_gamd_fixture import _fresh_system
    s1, s2 = _fresh_system(), _fresh_system()
    assert physical_system_sha256(mm, s1) == physical_system_sha256(mm, s2)
    s3 = _fresh_system()
    a, b, c = s3.getDefaultPeriodicBoxVectors()
    s3.setDefaultPeriodicBoxVectors(a * 1.0001, b * 1.0001, c * 1.0001)       # PDB-rounded box on resume
    assert physical_system_sha256(mm, s3) == physical_system_sha256(mm, s1)
    s2.addForce(mm.CustomExternalForce("0"))
    assert physical_system_sha256(mm, s1) != physical_system_sha256(mm, s2)


def test_cli_drops_unpersisted_flag_and_adds_phase_inputs(tmp_path, capsys):
    from gareus.cli import parse_args
    base = ["--seq", "GA", "--out", str(tmp_path / "o")]
    with pytest.raises(SystemExit):
        parse_args(base + ["--aux-cv-allow-unpersisted"])
    assert "unrecognized arguments" in capsys.readouterr().err
    a = parse_args(base)
    assert a.aux_phase_kind == "pilot" and a.aux_equilibrium_eligible is False
    with pytest.raises(SystemExit):
        parse_args(base + ["--aux-cv-model", "m.json", "--aux-equilibrium-eligible"])   # needs production
    assert "aux-phase-kind production" in capsys.readouterr().err


def test_resume_is_no_longer_refused_at_parse(tmp_path):
    import inspect
    import gareus.cli as cli
    src = inspect.getsource(cli._validate_aux_cv_args)
    assert "allow_unpersisted" not in src and "cannot --resume" not in src
```

Edit Stage B's `tests/test_aux_cv_cli.py` in the same commit:
- `OK`: delete `"--aux-cv-allow-unpersisted"`.
- `test_off_by_default`: `assert a.aux_cv_model is None and a.aux_phase_kind == "pilot" and a.aux_equilibrium_eligible is False`.
- `test_flags_are_known_config_keys`: `assert {"aux_cv_model", "aux_phase_kind", "aux_equilibrium_eligible"} <= dests and "aux_cv_allow_unpersisted" not in dests`.
- `test_refusals`: delete the `--resume`, `--extend` and `--aux-cv-allow-unpersisted` rows; keep every other row.
- Delete `test_ack_without_model_is_refused`.
- Add:

```python
def test_resume_is_accepted_at_parse(tmp_path):
    assert _parse(OK + ["--resume"], tmp_path).resume is True
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `tests/test_aux_runtime_definition.py`
Expected: FAIL `ModuleNotFoundError: No module named 'gareus.auxiliary_cv.runtime_definition'`

- [ ] **Step 3: Implement `runtime_definition.py`**

```python
# gareus/auxiliary_cv/runtime_definition.py
"""Build the frozen v2 state definition and the storage-facing runtime of an auxiliary run (D10)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..correctness._io import IntegrityError, digest, file_digest, json_bytes, json_loads
from ..correctness.bias import normalize_windows
from ..correctness.state_identity import _canonical_cv, make_state_definition

_PATH_SUFFIXES = ("_path", "_file", "_filename")
_PATH_KEYS = {"path", "file", "filename"}
_RESTRAINT_FIELDS = ("center1", "k1", "center2", "k2")


def physical_system_sha256(openmm, system) -> str:
    """Identity of the physical System (call BEFORE any umbrella/auxiliary bias force is added).

    The default periodic box is replaced by the unit cell before hashing: a resume rebuilds the System from
    01_solvated_start.pdb (rounded CRYST1), a fresh run from the in-memory topology. Box identity lives in
    the state definition (fixed_box_vectors_nm for NVT; a sample variable for NPT).
    """
    copy = openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(system))
    copy.setDefaultPeriodicBoxVectors(openmm.Vec3(1, 0, 0), openmm.Vec3(0, 1, 0), openmm.Vec3(0, 0, 1))
    return digest(openmm.XmlSerializer.serialize(copy).encode("utf-8"))


def _embed(value):
    if isinstance(value, dict):
        out = {}
        for key, val in value.items():
            if key.endswith(_PATH_SUFFIXES) or key in _PATH_KEYS:
                if val in (None, ""):
                    continue
                path = Path(str(val))
                if not path.is_file():
                    raise IntegrityError(f"cannot embed {key}: missing file {path}")
                stem = key[: -len(next(s for s in _PATH_SUFFIXES if key.endswith(s)))] if key not in _PATH_KEYS else key
                out[f"{stem}_content_sha256"] = file_digest(path)
            else:
                out[key] = _embed(val)
        return out
    if isinstance(value, list):
        return [_embed(v) for v in value]
    return value


def embed_cv_definition(kind: str, units: str, raw: Mapping[str, Any]) -> dict[str, Any]:
    definition = _embed(json_loads(json_bytes(dict(raw))))
    return _canonical_cv({"kind": str(kind), "units": str(units), "definition": definition}, f"cv[{kind}]")


def _applied_rows(applied_window_key) -> list[dict[str, Any]]:
    rows = []
    for w, entry in enumerate(applied_window_key):
        row = {"window_id": w, "center1": entry[0], "k1": entry[1]}
        if len(entry) == 4:
            row.update(center2=entry[2], k2=entry[3])
        rows.append(row)
    return normalize_windows(rows)


def build_runtime_state_definition(*, physical_system_sha256: str, ensemble: str, temperature_k: float,
                                   pressure_bar, box_vectors_nm, cv1, cv2, boost, snapshot_rows: Sequence[Mapping],
                                   aux_table, applied_window_key: Sequence[Sequence]) -> dict[str, Any]:
    if not (len(snapshot_rows) == aux_table.n == len(applied_window_key)):
        raise IntegrityError(f"snapshot has {len(snapshot_rows)} windows, auxiliary table {aux_table.n}, "
                             f"applied restraints {len(applied_window_key)}")
    merged = []
    for i, (snap, aux) in enumerate(zip(snapshot_rows, aux_table.window_rows())):
        if "instance" not in aux:
            raise IntegrityError(f"window {i}: the auxiliary state table carries no instance provenance; a v2 "
                                 "state definition needs it on every row (ruling M2)")
        row = dict(snap)
        row.update({k: v for k, v in aux.items() if k != "instance"})
        row["instance"] = dict(aux["instance"])
        merged.append(row)
    model = aux_table.model
    definition = make_state_definition(
        merged, physical_system_sha256=physical_system_sha256, ensemble=ensemble, temperature_k=float(temperature_k),
        pressure_bar=pressure_bar, fixed_box_vectors_nm=box_vectors_nm, cv1=cv1, cv2=cv2, boost=boost,
        aux_models={model.model_sha256: model.to_mapping()})
    applied = _applied_rows(applied_window_key)
    for want, have in zip(applied, sorted(definition["windows"], key=lambda r: r["window_id"])):
        diff = [k for k in _RESTRAINT_FIELDS if float(want[k]) != float(have[k])]
        if diff or want["window_id"] != have["window_id"]:
            raise IntegrityError(f"state definition window {have['window_id']} {diff} disagrees with the restraints "
                                 f"set_window applies: applied {[want[k] for k in diff]}, "
                                 f"definition {[have[k] for k in diff]}")
    return definition


@dataclass(frozen=True)
class AuxIORuntime:
    state_definition: dict
    models: Mapping[str, Any]
    force_info: Any
    sample_schema: Any
    runtime: Mapping[str, str]
    energy_version: str
    phase_kind: str
    equilibrium_eligible: bool
    topology_sha256: str


def aux_io_runtime(runtime, *, state_definition, topology, args, platform, context) -> AuxIORuntime:
    from ..energy_decomposition import peptide_atom_groups_from_topology
    from ..kernel_identity import exchange_energy_version_for_args
    from .checkpoint import topology_identity_sha256
    from .sample_schema import build_sample_schema, runtime_info, runtime_precision
    atoms = getattr(args, "pep_gamd_peptide_atoms", None)
    if not atoms:
        atoms = peptide_atom_groups_from_topology(topology, "all-peptide")[1]
    model = runtime.table.model
    schema = build_sample_schema(topology, list(atoms), [model])
    info = runtime_info(platform.getName(), runtime_precision(platform.getName(), platform, context))
    return AuxIORuntime(state_definition, {model.model_sha256: model}, runtime.info, schema, info,
                        exchange_energy_version_for_args(args), str(args.aux_phase_kind),
                        bool(args.aux_equilibrium_eligible), topology_identity_sha256(topology))
```

`file_digest` already exists in `gareus/correctness/_io.py` (line 78).

- [ ] **Step 4: Refuse instance-less tables** (`gareus/auxiliary_cv/state_table.py`, `load_aux_state_table`)

Right after the existing all-or-none check (`if with_instance and len(with_instance) != len(parsed): raise ...`):

```python
    if not with_instance:
        raise IntegrityError("auxiliary state tables need the instance columns (state_instance_id, state_role, "
                             "spawn_parent_state_id, matched_additional_slot_id) on every row: a frozen v2 state "
                             "definition requires instance metadata on every window (Stage C ruling M2)")
```

Every Stage B table test already supplies instances (`tests/test_aux_cv_state_table.py` `ROWS`), and the partial-table test still hits the all-or-none message first.

- [ ] **Step 5: CLI edits** (`gareus/cli.py`)

In `_add_aux_cv_args`, delete the `--aux-cv-allow-unpersisted` argument and add:

```python
    p.add_argument("--aux-phase-kind", default="pilot",
                   choices=["production", "pilot", "exploration", "equilibration"],
                   help="Sampling-policy phase kind frozen into an auxiliary run's window snapshot (spec Section 6). "
                        "Only 'production' segments can be marked equilibrium-eligible.")
    p.add_argument("--aux-equilibrium-eligible", action="store_true", default=False,
                   help="Mark this auxiliary run's segments eligible for equilibrium analysis (needs "
                        "--aux-phase-kind production and an equilibrated, prespecified retained segment).")
```

In `_validate_aux_cv_args`:
- The early return becomes plain `if not getattr(args, "aux_cv_model", None): return` (the acknowledgement-without-model error goes).
- Delete the `--resume`/`--extend` refusal and the final `aux_cv_allow_unpersisted` acknowledgement error.
- Add this as the **first** check after the early return, so its message wins over Stage B's later refusals:

```python
    if getattr(args, "aux_equilibrium_eligible", False) and str(getattr(args, "aux_phase_kind", "pilot")) != "production":
        p.error("--aux-equilibrium-eligible needs --aux-phase-kind production (freeze_snapshot rule)")
```

Stage B registers `_add_aux_cv_args` on both parsers (`cli.py:2140`, `:2175`), so the new flags reach `build_gareus_parser` too (S14).

- [ ] **Step 6: Run the tests to verify they pass**

Run: `tests/test_aux_runtime_definition.py tests/test_aux_cv_cli.py tests/test_aux_cv_state_table.py tests/test_aux_checkpoint.py`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add gareus/auxiliary_cv/runtime_definition.py gareus/auxiliary_cv/state_table.py gareus/cli.py tests/test_aux_runtime_definition.py tests/test_aux_cv_cli.py
git commit -m "feat(cvaux): runtime v2 state definition, instance-complete tables, storage adapter; lift the unpersisted flag"
```

---

### Task 12: Resume guards: capability binding on every resume, checkpoint table, args-aware kernel check

**Files:**
- Modify: `gareus/production.py`:
  - `refuse_resume_of_aux_campaign:702` → `refuse_resume_of_aux_campaign(out_dir, args=None)`, mismatch-only (ruling B1);
  - `run_gareus` fast branch `:6832-6866`: first statement becomes `refuse_resume_of_aux_campaign(out_dir, args)`; after `window_metadata = dict(...)` (`:6865`) the auxiliary table comes from the checkpoint and `_aux_window_key` is set (ruling B2);
  - `run_gareus` non-fast branch `:6872`: first statement `if bool(getattr(args, "resume", False)): refuse_resume_of_aux_campaign(out_dir, args)` (rulings B1/H3);
  - `_aux_window_order_key:2382`: NaN-safe (NaN → None) (ruling B2);
  - `verify_kernel_identity_on_resume:660`: compares against `exchange_energy_version_for_args(args)` (ruling H1); import it at `:121`;
  - delete Stage B's engineering WARNING (`:6901-6903`, names the removed flag).
- Modify: Stage B tests `tests/test_aux_cv_kernel_identity.py` (resume refusal tests), `tests/test_aux_cv_production_wiring.py` (refusal-first pin, population-cause pin)
- Test: `tests/test_aux_resume_guards.py`

**Interfaces and rules:**
- `refuse_resume_of_aux_campaign(out_dir, args=None) -> None` (B1). `recorded` = `_latest_segment_kernel_identity(out_dir)` (newest snapshot with a kernel identity); `model` = `args.aux_cv_model` (None when `args` is None):
  - `recorded is None` (no snapshot, or snapshots predating kernel identity): no-op. A first job that passes `--resume` into an empty directory must start. Such a directory is still covered elsewhere: a fast resume with `--aux-cv-model` fails in `aux_table_from_checkpoint` (no `aux` block), a non-aux job fast-resuming an aux checkpoint fails in `verify_aux_resume` (Task 13, `aux_pre_apply`), and pooling refuses mixed directories (Task 9).
  - recorded auxiliary, no model → `RuntimeError` ("…resume it with the same --aux-cv-model…").
  - recorded auxiliary, model whose `model_sha256` differs from the recorded `aux_model_sha256` → `RuntimeError` (a model change starts a new phase, spec Section 6).
  - recorded non-auxiliary, model given → `RuntimeError` (a non-aux campaign cannot continue as an auxiliary one).
  - recorded non-auxiliary, no model: no-op. Every legacy resume is unchanged.
- Placement:
  - **Fast resume:** still the **first statement** of `if fast_resume:` (before `load_resume_run_definition`), now with `args`.
  - **Non-fast `--resume`** (no checkpoint manifest, or one without replica files): first statement of the `else:` branch, before the `equil_state` check (H3 resume half).
  - A fresh run without `--resume` into a directory that already holds the other capability is not refused here; it fails closed at pooling (`load_parquet` refuses the mixed directory, Task 9, ruling H3).
- **Checkpoint table (B2).** In the fast branch, with `--aux-cv-model`, `_aux_table = aux_table_from_checkpoint(resume_manifest, model=AuxModel.load(args.aux_cv_model))` (Task 6), then `refuse_aux_population_change(_aux_table.n, len(centers_a), cause="checkpoint resume")` and `_aux_window_key = _aux_window_order_key(centers_a, k_list, secondary_cv_centers, secondary_cv_k_kcal_list, len(centers_a))`. Without the key, the runtime resolution check (`:7768`) compares against None and raises "window order changed".
- **NaN-safe key (B2).** `_aux_window_order_key` maps NaN to None. Two separately built NaN floats compare unequal, so a 2-D table with NaN CV2 placeholders would always fail the window-order check (Stage B minor "NaN self-inequality").
- **Args-aware kernel check (H1).** `verify_kernel_identity_on_resume` refuses when the recorded `exchange_energy_version` differs from `exchange_energy_version_for_args(args)`. Legacy is identical (it returns `state_bias_matrix_v2` without `--aux-cv-model`). `provenance._method_settings` records `exchange_energy_version_for_args(args)` (`provenance.py:345-346`), so an aux campaign records v3_aux.
- Every new call in `run_gareus` sits under an `if` whose test names `aux` (`getattr(args, "aux_cv_model", None)`), so Stage B's gating test covers `refuse_aux_population_change` and `aux_table_from_checkpoint` (the latter is added to its list in Task 13).
- **Known residual (not fixed here):** `main()` rewrites `run_manifest.json` (`method_settings`, `resolved_args`) for every job before `run_gareus` runs, so a refused `--aux-cv-model --resume` into a non-aux directory still stamps `aux_cv_model_sha256` into that directory's `method_settings`. `run_has_aux`, `_refuse_aux_run` and `segment_eligibility`'s H2 test read that as auxiliary evidence, so the legacy directory's non-Parquet loaders then refuse and its snapshot-less segments become `aux_unpersisted`. Fail-closed, not silent. Remedy (a later change): run the B1 check in `main()` before `initialize_run_manifest`, or restore the previous manifest on refusal. Also: `main()` refuses `--resume` without a production checkpoint before `run_gareus` (`cli.py:2680-2706`), so the non-fast guard covers only a manifest without replica files and internal callers.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_aux_resume_guards.py
import ast
import inspect
import json
import types

import pytest

from aux_cv_fixture import model_payload
from gareus.auxiliary_cv.model import AuxModel
from gareus.kernel_identity import EXCHANGE_ENERGY_VERSION, EXCHANGE_ENERGY_VERSION_AUX


def _model(tmp_path, name="aux.json", c=1.0):
    m = AuxModel.from_mapping(model_payload([(0, 1, 2, 3)], [c, 0.0]))
    path = tmp_path / name
    m.write(path)
    return m, str(path)


def _snapshot_dir(tmp_path, identity):
    run = tmp_path / "run"
    (run / "windows").mkdir(parents=True)
    (run / "windows" / "seg_001.json").write_text(json.dumps({"segment_id": "seg_001", "windows": [],
                                                              "kernel_identity": identity}))
    return run


def _args(model_path=None):
    return types.SimpleNamespace(aux_cv_model=model_path)


def test_resume_refusal_is_two_sided_and_model_bound(tmp_path):
    from gareus.production import refuse_resume_of_aux_campaign
    m, path = _model(tmp_path)
    _other, other_path = _model(tmp_path, "other.json", c=2.0)
    aux_run = _snapshot_dir(tmp_path / "a", {"exchange_energy_version": EXCHANGE_ENERGY_VERSION_AUX,
                                             "aux_model_sha256": m.model_sha256})
    legacy_run = _snapshot_dir(tmp_path / "l", {"exchange_energy_version": EXCHANGE_ENERGY_VERSION})
    refuse_resume_of_aux_campaign(aux_run, _args(path))                        # same model: resumes
    refuse_resume_of_aux_campaign(legacy_run, _args(None))                     # legacy: unchanged
    refuse_resume_of_aux_campaign(tmp_path / "empty", _args(path))             # nothing recorded yet
    with pytest.raises(RuntimeError, match="same --aux-cv-model"):
        refuse_resume_of_aux_campaign(aux_run, _args(None))
    with pytest.raises(RuntimeError, match="is not this campaign's auxiliary model"):
        refuse_resume_of_aux_campaign(aux_run, _args(other_path))
    with pytest.raises(RuntimeError, match="ran without auxiliary-CV states"):
        refuse_resume_of_aux_campaign(legacy_run, _args(path))


def test_kernel_identity_resume_check_is_args_aware(tmp_path):
    from gareus.production import verify_kernel_identity_on_resume
    _m, path = _model(tmp_path)
    meta = {"enabled": True, "mode": "alpha"}
    aux = types.SimpleNamespace(secondary_cv="alpha", aux_cv_model=path)
    plain = types.SimpleNamespace(secondary_cv="alpha")

    def _record(version):
        (tmp_path / "run_manifest.json").write_text(json.dumps({"method_settings": {"exchange_energy_version": version}}))

    _record(EXCHANGE_ENERGY_VERSION_AUX)
    verify_kernel_identity_on_resume(aux, tmp_path, meta)
    with pytest.raises(RuntimeError, match="resume refused"):
        verify_kernel_identity_on_resume(plain, tmp_path, meta)
    _record(EXCHANGE_ENERGY_VERSION)
    verify_kernel_identity_on_resume(plain, tmp_path, meta)
    with pytest.raises(RuntimeError, match="resume refused"):
        verify_kernel_identity_on_resume(aux, tmp_path, meta)


def test_window_order_key_is_nan_safe():
    from gareus.production import _aux_window_order_key
    nan = float("nan")
    a = _aux_window_order_key([0.1, 0.2], [1.0, 1.0], [nan, 0.5], [0.0, 2.0], 2)
    assert a == _aux_window_order_key([0.1, 0.2], [1.0, 1.0], [nan, 0.5], [0.0, 2.0], 2)
    assert a[0][2] is None


def _run_gareus_tree():
    import gareus.production as production
    return ast.parse(inspect.getsource(production.run_gareus))


def _calls(node, name):
    return [n for n in ast.walk(node) if isinstance(n, ast.Call)
            and getattr(n.func, "attr", getattr(n.func, "id", None)) == name]


def _fast_resume_branch(tree):
    """The `if fast_resume:` whose first statement is the refusal (run_gareus has other fast_resume ifs)."""
    hits = [n for n in ast.walk(tree) if isinstance(n, ast.If) and isinstance(n.test, ast.Name)
            and n.test.id == "fast_resume" and n.body and isinstance(n.body[0], ast.Expr)
            and _calls(n.body[0], "refuse_resume_of_aux_campaign")]
    assert len(hits) == 1
    return hits[0]


def test_refusal_runs_first_on_every_resume_branch():
    branch = _fast_resume_branch(_run_gareus_tree())
    fast, other = branch.body[0], branch.orelse[0]
    assert [ast.unparse(a) for a in _calls(fast, "refuse_resume_of_aux_campaign")[0].args] == ["out_dir", "args"]
    assert isinstance(other, ast.If) and "resume" in ast.unparse(other.test)
    assert _calls(other, "refuse_resume_of_aux_campaign")


def test_fast_resume_takes_the_aux_table_from_the_checkpoint_and_sets_the_window_key():
    body = ast.Module(body=_fast_resume_branch(_run_gareus_tree()).body, type_ignores=[])
    assert _calls(body, "aux_table_from_checkpoint")
    causes = {k.value.value for c in _calls(body, "refuse_aux_population_change") for k in c.keywords
              if k.arg == "cause" and isinstance(k.value, ast.Constant)}
    assert causes == {"checkpoint resume"}
    assert any(isinstance(n, ast.Assign) and any(getattr(t, "id", None) == "_aux_window_key" for t in n.targets)
               for n in ast.walk(body))


def test_stage_b_engineering_warning_is_gone():
    import gareus.production as production
    assert "aux-cv-allow-unpersisted" not in inspect.getsource(production.run_gareus)
```

Edit Stage B's tests in the same commit:
- `tests/test_aux_cv_kernel_identity.py`:
  - replace `test_resume_of_an_aux_campaign_is_refused_unconditionally` with

```python
def test_resume_of_an_aux_campaign_without_its_model_is_refused(tmp_path):
    from gareus.production import refuse_resume_of_aux_campaign
    run = _run_dir_with_snapshot(tmp_path, {"exchange_energy_version": EXCHANGE_ENERGY_VERSION_AUX,
                                            "aux_model_sha256": "d" * 64})
    with pytest.raises(RuntimeError, match="auxiliary"):
        refuse_resume_of_aux_campaign(run, types.SimpleNamespace(aux_cv_model=None))
```

  - in `test_resume_of_a_legacy_campaign_is_not_refused`, pass `types.SimpleNamespace(aux_cv_model=None)` as the second argument of each of the three calls.
- `tests/test_aux_cv_production_wiring.py`:
  - `test_resume_refusal_runs_first_in_the_fast_resume_branch`: `i_refuse = src.find("refuse_resume_of_aux_campaign(out_dir")` (prefix: the call now passes `args`);
  - `test_run_gareus_builds_aux_runtime_before_contexts_and_guards_population`: `assert causes == {"seed-reachability filter", "--max-replicas", "US auto-drop", "checkpoint resume"}`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `tests/test_aux_resume_guards.py`
Expected: FAIL (`refuse_resume_of_aux_campaign() takes 1 positional argument`; the NaN key compares unequal; the branch pins find no `args`).

- [ ] **Step 3: Implement in `gareus/production.py`**

Add `exchange_energy_version_for_args` to the `.kernel_identity` import at `:121`.

Replace `refuse_resume_of_aux_campaign`:

```python
def refuse_resume_of_aux_campaign(out_dir, args=None) -> None:
    """Every --resume continues the auxiliary capability its campaign recorded (spec 2026-10-07, D6; Stage C B1).

    Snapshot-based on purpose: a Context checkpoint of an aux system loads SILENTLY into a Context without
    the aux force (verified 2026-10-08), and the run manifest's method_settings are rebuilt from the
    CURRENT args every job. Refused: an auxiliary campaign resumed without --aux-cv-model or with another
    model, and a non-auxiliary campaign resumed with one. Nothing recorded (no kernel-identity snapshot):
    no-op; the checkpoint binding (aux_table_from_checkpoint / verify_aux_resume) and pooling cover it.
    Legacy campaigns resumed without a model pass silently.
    """
    recorded = _latest_segment_kernel_identity(out_dir)
    if recorded is None:
        return
    model_path = getattr(args, "aux_cv_model", None) if args is not None else None
    if is_aux_kernel_record(recorded):
        recorded_sha = str(recorded.get("aux_model_sha256") or "")
        if not model_path:
            raise RuntimeError(
                "resume/extend refused: this campaign ran auxiliary-CV states "
                f"(model {recorded_sha[:12]}); resume it with the same --aux-cv-model, or start a new run "
                "directory. Without it every auxiliary slot would continue as an ordinary state.")
        from .auxiliary_cv.model import AuxModel
        sha = AuxModel.load(model_path).model_sha256
        if sha != recorded_sha:
            raise RuntimeError(
                f"resume/extend refused: --aux-cv-model {sha[:12]} is not this campaign's auxiliary model "
                f"{recorded_sha[:12]}; a model change starts a new phase (spec Section 6).")
        return
    if model_path:
        raise RuntimeError(
            "resume/extend refused: --aux-cv-model given, but this campaign's newest segment ran without "
            "auxiliary-CV states; an auxiliary capability change starts a new run directory.")
```

Make `_aux_window_order_key` NaN-safe:

```python
def _aux_window_order_key(centers_a, k_list, secondary_centers, secondary_k_list, n) -> tuple:
    """(center1, k1[, center2, k2]) per window, in order: the identity an auxiliary state table is
    bound to when loaded, re-checked when the runtime is resolved (auxiliary-CV runs only).
    NaN placeholders map to None, so two keys built from the same table compare equal."""
    def _num(value):
        value = float(value)
        return None if value != value else value

    has_secondary = secondary_centers is not None and secondary_k_list is not None
    return tuple(
        (_num(centers_a[w]), _num(k_list[w]))
        + ((_num(secondary_centers[w]), _num(secondary_k_list[w])) if has_secondary else ())
        for w in range(int(n)))
```

In `verify_kernel_identity_on_resume`, replace the exchange-version refusal with:

```python
    expected_ex = exchange_energy_version_for_args(args)
    if recorded_ex is not None and str(recorded_ex) != expected_ex:
        raise RuntimeError(
            f"resume refused: this campaign's samples were exchanged with {recorded_ex!r}; this job "
            f"implements {expected_ex!r}. A changed kernel starts a new segment from these "
            "coordinates with fresh equilibration; it never appends to the old one (spec F04).")
```

In `run_gareus`:
- fast branch, first statement: `refuse_resume_of_aux_campaign(out_dir, args)` (update the comment above it: "Every resume (also --extend): auxiliary capability mismatch refused (B1); legacy no-op").
- fast branch, directly after `window_metadata = dict(resume_def.get("window_metadata", {}))`:

```python
        if getattr(args, "aux_cv_model", None):
            # Stage C (B2): the frozen auxiliary table comes from the checkpoint, never from today's CSV.
            from .auxiliary_cv.checkpoint import aux_table_from_checkpoint
            from .auxiliary_cv.model import AuxModel
            from .auxiliary_cv.runtime import refuse_aux_population_change
            _aux_table = aux_table_from_checkpoint(resume_manifest, model=AuxModel.load(args.aux_cv_model))
            refuse_aux_population_change(_aux_table.n, len(centers_a), cause="checkpoint resume")
            _aux_window_key = _aux_window_order_key(centers_a, k_list, secondary_cv_centers,
                                                    secondary_cv_k_kcal_list, len(centers_a))
```

- non-fast branch: insert as the first statement of `else:` (before `if equil_state is None:`):

```python
        if bool(getattr(args, "resume", False)):
            refuse_resume_of_aux_campaign(out_dir, args)       # non-fast --resume (no checkpoint): B1/H3
```

- delete the `print("WARNING: auxiliary-CV states active (--aux-cv-allow-unpersisted): ...")` statement (3 lines) in the CSV branch.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `tests/test_aux_resume_guards.py tests/test_aux_cv_kernel_identity.py tests/test_aux_cv_production_wiring.py tests/test_aux_cv_final_review_pins.py tests/test_resume_and_poincare_regressions.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add gareus/production.py tests/test_aux_resume_guards.py tests/test_aux_cv_kernel_identity.py tests/test_aux_cv_production_wiring.py
git commit -m "feat(cvaux): bind every resume to the recorded aux capability; checkpoint aux table; args-aware kernel check"
```

---

### Task 13: Production I/O wiring (samples, events, snapshot, checkpoints, runtime parity)

**Files:**
- Create: `gareus/auxiliary_cv/runtime_io.py`
- Modify: `gareus/production.py` (sites listed in Step 4, all inside `run_gareus`)
- Modify: Stage B's `tests/test_aux_cv_production_wiring.py` (`test_aux_runtime_construction_in_run_gareus_is_gated_by_an_aux_condition`: Stage C names, `ast.IfExp`)
- Test: `tests/test_aux_runtime_io.py`

**Interfaces:**
- Produces:
  - `check_exchange_boundary_alignment(*, exchange_interval, distance_interval, traj_interval, calib_steps) -> list[str]`. It takes the **resolved** values (C14/S9): `distance_interval` after `production.py:8782-8785`, `traj_interval` = `effective_traj_interval` (`:7491-7493`). It raises unless `exchange_interval % distance_interval == 0`. When `traj_interval > 0`, it also raises unless `exchange_interval % traj_interval == 0` and `calib_steps % traj_interval == 0`, because reporters fire on the absolute step grid (`npt_driver.register_reporter`, `describe_reporter`). With `traj_interval == 0` it returns a note: boundary frames are absent, which is a Stage D prerequisite.
  - `ExchangeEventCounter.next(step) -> int`.
  - `event_from_gibbs(prop, *, step, seq, selected_replica, accepted, energy_version, assignments_after) -> dict` covers `stay`/`no_candidates` (`GibbsProposal` fields, `production.py:1325`).
  - `aux_event_fields(prop) -> dict` gives the `selected_replica`-free q/p fields for a Gibbs move. `_log(q)` is `-inf` for q ≤ 0 (C11).
  - `check_runtime_parity(runtime_z, positions_z, *, beta, k_max_kcal, centers, tolerance) -> None` raises Stage B's `AuxObservationError` when `parity_violation` exceeds the tolerance or finiteness differs. **With no active state (`k_max_kcal <= 0`) it returns first and never raises** (ruling M5: sham-only arms record, Stage B contract; the fast path sees a finite theta where positions give NaN).
  - `context_box_nm(context, unit) -> list[list[float]]` (one Context's box vectors, nm) and `check_fixed_box(boxes, fixed_box, *, label) -> None` (every box must equal the fixed NVT box exactly, else `IntegrityError` naming the replica).
  - `make_aux_record_observer(runtime, *, use_fast_path, fast_forces, unit, schema, models)` → `observe(r, sim) -> (z_runtime: float, torsions: np.ndarray, z_positions: float)`. **One** `getState(getPositions=True)` per carrier per sample gives the geometry check (only with an active state, Stage B's `check_aux_geometry`), the slow-path runtime z (Stage B's `observe_aux_z`), the stored torsion basis and the positions z used for runtime parity (Task 2's `observe_carrier`). On the fast path the runtime z is the force's own value, exactly as Stage B's observer computes it (ruling M4: no second positions read).
- Opt-in: every new statement in `run_gareus` is under `if _aux_io is not None:` / `if getattr(args, "_aux_runtime", None) is not None:` / `if getattr(args, "aux_cv_model", None):`, or is a keyword whose legacy value reproduces today's call (`**{}`, a helper returning None, two trailing `None` in the fetch tuple). Stage B's gating test is extended with the Stage C names (Step 5).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_aux_runtime_io.py
import ast
import inspect
import math
import types
from types import SimpleNamespace

import numpy as np
import pytest

from gareus.auxiliary_cv.runtime_io import (ExchangeEventCounter, aux_event_fields, check_exchange_boundary_alignment,
                                            check_runtime_parity, event_from_gibbs)
from gareus.correctness._io import IntegrityError


def test_alignment_uses_resolved_intervals_and_calib_offset():
    assert check_exchange_boundary_alignment(exchange_interval=3000, distance_interval=300, traj_interval=3000,
                                             calib_steps=6000) == []
    with pytest.raises(IntegrityError, match="traj_interval"):
        check_exchange_boundary_alignment(exchange_interval=3000, distance_interval=300, traj_interval=2000, calib_steps=0)
    with pytest.raises(IntegrityError, match="calib_steps"):
        check_exchange_boundary_alignment(exchange_interval=3000, distance_interval=300, traj_interval=3000, calib_steps=1000)
    with pytest.raises(IntegrityError, match="distance_interval"):
        check_exchange_boundary_alignment(exchange_interval=3000, distance_interval=700, traj_interval=3000, calib_steps=0)
    notes = check_exchange_boundary_alignment(exchange_interval=3000, distance_interval=300, traj_interval=0, calib_steps=0)
    assert notes and "Stage D" in notes[0]


def test_counter_orders_within_step():
    c = ExchangeEventCounter()
    assert [c.next(10), c.next(10), c.next(20), c.next(20), c.next(20)] == [0, 1, 0, 1, 2]


def test_event_from_gibbs_stay_and_no_candidates_and_zero_q():
    stay = SimpleNamespace(current_window=2, proposed_window=2, delta_kj=0.0, q_forward=0.6, q_reverse=0.0,
                           pacc=0.0, stayed=True, no_candidates=False)
    ev = event_from_gibbs(stay, step=100, seq=4, selected_replica=7, accepted=False, energy_version="v",
                          assignments_after=[0, 1])
    assert ev["kind"] == "stay" and ev["replica_i"] == ev["replica_j"] == 7 and ev["log_q_reverse"] == float("-inf")
    none = SimpleNamespace(current_window=2, proposed_window=2, delta_kj=0.0, q_forward=0.0, q_reverse=0.0,
                           pacc=0.0, stayed=False, no_candidates=True)
    assert event_from_gibbs(none, step=1, seq=0, selected_replica=7, accepted=False, energy_version="v",
                            assignments_after=[0])["kind"] == "no_candidates"
    move = SimpleNamespace(q_forward=0.25, q_reverse=0.0, pacc=0.4)
    f = aux_event_fields(move)
    assert f["log_q_forward"] == pytest.approx(math.log(0.25)) and f["log_q_reverse"] == float("-inf")
    assert f["p_accept"] == 0.4


def test_runtime_parity():
    from gareus.auxiliary_cv.runtime import AuxObservationError
    check_runtime_parity(np.array([1.0]), np.array([1.0 + 1e-9]), beta=0.4, k_max_kcal=2.0, centers=[0.5], tolerance=1e-6)
    with pytest.raises(AuxObservationError, match="parity"):
        check_runtime_parity(np.array([1.0]), np.array([1.01]), beta=0.4, k_max_kcal=2.0, centers=[0.5], tolerance=1e-6)
    with pytest.raises(AuxObservationError, match="finite"):
        check_runtime_parity(np.array([1.0]), np.array([np.nan]), beta=0.4, k_max_kcal=2.0, centers=[0.5], tolerance=1e-6)


def test_fixed_box_check():
    from gareus.auxiliary_cv.runtime_io import check_fixed_box
    box = [[3.0, 0, 0], [0, 3.0, 0], [0, 0, 3.0]]
    check_fixed_box([box, [row[:] for row in box]], box, label="t")
    with pytest.raises(IntegrityError, match="replica 1"):
        check_fixed_box([box, [[3.0001, 0, 0], [0, 3.0, 0], [0, 0, 3.0]]], box, label="t")


def test_runtime_parity_never_raises_without_an_active_state():
    """M5: a sham-only population records (fast path finite theta vs positions NaN), never raises."""
    check_runtime_parity(np.array([0.3]), np.array([np.nan]), beta=0.4, k_max_kcal=0.0, centers=[], tolerance=1e-6)
    check_runtime_parity(np.array([0.3]), np.array([5.0]), beta=0.4, k_max_kcal=0.0, centers=[], tolerance=1e-6)


def _record_setup(k_kcal):
    from test_aux_cv_observation import _setup, _with_table
    from gareus.auxiliary_cv.sample_schema import build_sample_schema
    ctx, rt, force, d = _setup()
    runtime = _with_table(rt, k_kcal)
    atoms = sorted({a for q in d["quads"] for a in q})
    schema = build_sample_schema(d["topology"], atoms, [rt.table.model])
    return ctx, runtime, force, d, schema, {rt.table.model.model_sha256: rt.table.model}


@pytest.mark.parametrize("fast", [True, False])
def test_record_observer_one_read_gives_runtime_z_torsions_and_positions_z(fast):
    from openmm import unit
    from gareus.auxiliary_cv.runtime_io import make_aux_record_observer
    from gareus.auxiliary_cv.sample_schema import observe_carrier
    ctx, runtime, force, d, schema, models = _record_setup(2.0)
    observe = make_aux_record_observer(runtime, use_fast_path=fast, fast_forces=[force], unit=unit,
                                       schema=schema, models=models)
    z_rt, tors, z_pos = observe(0, types.SimpleNamespace(context=ctx))
    ref = observe_carrier(d["positions_nm"], schema, models)
    np.testing.assert_allclose(tors, ref.torsions, atol=1e-12)
    assert z_pos == pytest.approx(float(ref.z[0]), abs=1e-12) and z_rt == pytest.approx(z_pos, abs=1e-9)


@pytest.mark.parametrize("fast", [True, False])
def test_record_observer_checks_geometry_only_with_an_active_state(fast):
    from openmm import unit
    from test_aux_cv_observation import _degenerate
    from gareus.auxiliary_cv.runtime import AuxObservationError
    from gareus.auxiliary_cv.runtime_io import make_aux_record_observer
    for k, raises in ((2.0, True), (0.0, False)):
        ctx, runtime, force, d, schema, models = _record_setup(k)
        _degenerate(ctx, runtime, d)
        observe = make_aux_record_observer(runtime, use_fast_path=fast, fast_forces=[force], unit=unit,
                                           schema=schema, models=models)
        if raises:
            with pytest.raises(AuxObservationError, match="degenerate"):
                observe(0, types.SimpleNamespace(context=ctx))
        else:
            observe(0, types.SimpleNamespace(context=ctx))


def _func(tree, name):
    return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)


def _named_calls(node, name):
    return [n for n in ast.walk(node) if isinstance(n, ast.Call)
            and getattr(n.func, "attr", getattr(n.func, "id", None)) == name]


def test_every_gibbs_decision_and_every_swap_early_return_writes_an_event():
    import gareus.production as production
    tree = ast.parse(inspect.getsource(production.run_gareus))
    swap = _func(tree, "_attempt_window_swap")
    # D1: count CALL sites (the def line is not a call): one before each of the three early returns
    assert len(_named_calls(swap, "_aux_write_skip")) == 3
    early = [n for n in ast.walk(swap) if isinstance(n, ast.Return) and isinstance(n.value, ast.Name)
             and n.value.id == "attempt"]
    assert len(early) == 3
    src = ast.unparse(tree)
    gibbs_block = src[src.index("if mode == 'gibbs-walk'"):]
    gibbs_block = gibbs_block[: gibbs_block.index("raise ValueError")]
    assert gibbs_block.count("event_from_gibbs(") >= 2 and "aux_event=" in gibbs_block


def test_checkpoint_saves_and_load_carry_the_aux_hooks():
    import gareus.production as production
    tree = ast.parse(inspect.getsource(production.run_gareus))
    saves = _named_calls(tree, "save_production_checkpoint")
    assert len(saves) == 2 and all(any(k.arg == "aux_block" for k in c.keywords) for c in saves)
    loads = _named_calls(tree, "load_production_checkpoint")
    assert len(loads) == 1 and any(k.arg == "aux_pre_apply" for k in loads[0].keywords)
    assert len(_named_calls(tree, "verify_aux_ledger")) == 1
```

The structural tests pin the shape of the wiring. The behavioural checks (one event per selected carrier, ledger replay, resume parity) are the Task 14 real runs.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `tests/test_aux_runtime_io.py`
Expected: FAIL `ModuleNotFoundError: No module named 'gareus.auxiliary_cv.runtime_io'`

- [ ] **Step 3: Implement `runtime_io.py`**

```python
# gareus/auxiliary_cv/runtime_io.py
"""Small, testable glue between run_gareus and the auxiliary storage layer (CVaux Stage C)."""
from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from ..correctness._io import IntegrityError


def check_exchange_boundary_alignment(*, exchange_interval: int, distance_interval: int, traj_interval: int,
                                      calib_steps: int) -> list[str]:
    ex = int(exchange_interval)
    if int(distance_interval) <= 0 or ex % int(distance_interval) != 0:
        raise IntegrityError(f"auxiliary runs need distance_interval ({distance_interval}, resolved) to divide "
                             f"exchange_interval ({ex}) so samples exist on every exchange boundary (spec Section 7)")
    if int(traj_interval) <= 0:
        return ["trajectory frames are off: no structural frames on the exchange-boundary grid (Stage D prerequisite)"]
    if ex % int(traj_interval) != 0:
        raise IntegrityError(f"auxiliary runs need traj_interval ({traj_interval}) to divide exchange_interval ({ex})")
    if int(calib_steps) % int(traj_interval) != 0:
        raise IntegrityError(f"auxiliary runs need calib_steps ({calib_steps}) to be a multiple of traj_interval "
                             f"({traj_interval}): reporters fire on the absolute step grid")
    return []


class ExchangeEventCounter:
    def __init__(self) -> None:
        self._step = None
        self._seq = 0

    def next(self, step: int) -> int:
        if step != self._step:
            self._step, self._seq = step, 0
        seq = self._seq
        self._seq += 1
        return seq


def _log(q: float) -> float:
    return math.log(q) if q > 0 else float("-inf")


def aux_event_fields(prop) -> dict[str, float]:
    return {"log_q_forward": _log(float(prop.q_forward)), "log_q_reverse": _log(float(prop.q_reverse)),
            "p_accept": float(prop.pacc)}


def event_from_gibbs(prop, *, step: int, seq: int, selected_replica: int, accepted: bool, energy_version: str,
                     assignments_after: Sequence[int]) -> dict[str, Any]:
    kind = "no_candidates" if prop.no_candidates else ("stay" if prop.stayed else "swap")
    return dict(step=int(step), attempt_seq=int(seq), selected_replica=int(selected_replica),
                replica_i=int(selected_replica), replica_j=int(selected_replica),
                window_i=int(prop.current_window), window_j=int(prop.proposed_window if kind == "swap" else prop.current_window),
                kind=kind, delta_e_kj=float(prop.delta_kj), accepted=bool(accepted),
                energy_version=str(energy_version), assignments_after=[int(x) for x in assignments_after],
                **aux_event_fields(prop))


def check_runtime_parity(runtime_z, positions_z, *, beta: float, k_max_kcal: float, centers, tolerance: float) -> None:
    if float(k_max_kcal) <= 0:
        return                       # no active state: record only, never raise (ruling M5)
    from .offline import parity_violation
    from .runtime import AuxObservationError
    runtime_z = np.asarray(runtime_z, dtype=np.float64)
    positions_z = np.asarray(positions_z, dtype=np.float64)
    if np.any(np.isfinite(runtime_z) != np.isfinite(positions_z)):
        raise AuxObservationError("runtime/positions aux z parity: finite mismatch with an active auxiliary state")
    du = parity_violation(runtime_z, positions_z, beta=beta, k_max_kcal=k_max_kcal, centers=centers)
    if du.size and np.nanmax(du) > float(tolerance):
        raise AuxObservationError(f"runtime/positions aux z parity: reduced disagreement {np.nanmax(du):.3g} > {tolerance:g}")


def context_box_nm(context, unit) -> list:
    return np.asarray(context.getState().getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer),
                      dtype=np.float64).tolist()


def check_fixed_box(boxes, fixed_box, *, label: str) -> None:
    """An NVT state definition has ONE fixed box: every replica's box must equal it exactly."""
    want = np.asarray(fixed_box, dtype=np.float64)
    for r, box in enumerate(boxes):
        if not np.array_equal(np.asarray(box, dtype=np.float64), want):
            raise IntegrityError(f"{label}: replica {r} box {np.asarray(box).tolist()} != the fixed NVT box "
                                 f"{want.tolist()}; an NVT auxiliary state definition needs one fixed box")


def make_aux_record_observer(runtime, *, use_fast_path: bool, fast_forces, unit, schema, models):
    """Per-sample observation of an auxiliary run: ONE positions read per carrier gives the geometry check,
    the stored torsion basis and the positions z for runtime parity; the runtime exchange z is computed as
    Stage B's make_aux_z_observer does (fast path: the force's own value; slow path: these positions)."""
    from .runtime import check_aux_geometry, observe_aux_z
    from .sample_schema import observe_carrier
    any_active = any(float(k) > 0.0 for k in runtime.table.k_kcal)

    def observe(r, sim):
        pos = sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        if any_active:
            check_aux_geometry(pos, runtime, replica=r)
        if use_fast_path:
            z_runtime = observe_aux_z(sim.context, runtime, force=fast_forces[r])
        else:
            z_runtime = observe_aux_z(sim.context, runtime, positions_nm=pos)
        obs = observe_carrier(pos, schema, models)
        return float(z_runtime), obs.torsions, float(obs.z[0])

    return observe
```

- [ ] **Step 4: Wire `production.py`** (anchors at HEAD `d4415b1`; line numbers move as earlier tasks land, so anchor on the quoted code)

1. **Physical-system identity.** Directly after `prepare_pep_gamd_args(args, topology)` (`:7122`) and before `secondary_cv_force_info = add_umbrella_cv_forces(...)`:

```python
    _aux_physical_sha = None
    if getattr(args, "aux_cv_model", None):
        from .auxiliary_cv.runtime_definition import physical_system_sha256
        _aux_physical_sha = physical_system_sha256(openmm, base_system)
```

2. **Adapter, record observer and snapshot.** Between `_cv2_type = ...` (`:8045`) and the `WindowSnapshot(out_dir).snapshot(...)` call (`:8046`), after Stage B's `aux_snapshot_rows`/`_check_instances` block (which stays unchanged, so `_check_instances` keeps exactly one call site):

```python
    _aux_io = None
    _aux_record_for_replica = None
    if getattr(args, "_aux_runtime", None) is not None:
        from .auxiliary_cv.runtime_definition import (aux_io_runtime, build_runtime_state_definition,
                                                      embed_cv_definition)
        from .auxiliary_cv.runtime_io import check_fixed_box, context_box_nm, make_aux_record_observer
        from .cv import primary_cv_is_dimensionless
        _ens = str(getattr(args, "production_ensemble", "npt")).upper()
        _box = None
        if _ens == "NVT" and fast_resume:
            # The Contexts have not loaded their checkpoints yet (their box is the System default, from the
            # PDB-reloaded topology): the fixed box is the checkpoint's; _aux_pre_apply verifies every
            # restored box against it before the re-apply.
            _box = (((resume_manifest or {}).get("aux") or {}).get("state_definition") or {}).get("fixed_box_vectors_nm")
        elif _ens == "NVT":
            # Board condition 3: an OpenMM State always carries the box (verified 2026-10-08, 8.5.1
            # Reference/CPU). Fresh path: every replica got its start box in _build_context_i; they must agree.
            _boxes = [_sim_pool.submit(r, context_box_nm, sims[r].context, unit).result() for r in range(nrep)]
            check_fixed_box(_boxes, _boxes[0], label="NVT production replicas at segment start")
            _box = _boxes[0]
        _cv1 = embed_cv_definition(primary_cv_mode(args),
                                   "dimensionless" if primary_cv_is_dimensionless(args) else "angstrom",
                                   _json_ready(primary_cv_def))
        _cv2 = (embed_cv_definition(_cv2_type, "dimensionless", _json_ready(secondary_cv_metadata))
                if _cv2_type is not None else None)
        _boost = ({"kind": str(getattr(args, "gamd_boost_type", "") or "gamd"),
                   "envelope": _json_ready(shared_gamd_globals_all)} if use_gamd else None)
        _aux_definition = build_runtime_state_definition(
            physical_system_sha256=_aux_physical_sha, ensemble=_ens, temperature_k=float(args.temperature_k),
            pressure_bar=float(args.pressure_bar) if _ens == "NPT" else None, box_vectors_nm=_box,
            cv1=_cv1, cv2=_cv2, boost=_boost, snapshot_rows=_win_snapshot_windows,
            aux_table=args._aux_runtime.table,
            applied_window_key=_aux_window_order_key(centers_a, k_list, secondary_cv_centers,
                                                     secondary_cv_k_kcal_list, nrep))
        _aux_io = aux_io_runtime(args._aux_runtime, state_definition=_aux_definition, topology=topology, args=args,
                                 platform=platform, context=sims[0].context)
        _aux_record_for_replica = make_aux_record_observer(
            args._aux_runtime, use_fast_path=_use_fast_cv_path, fast_forces=_fast_aux_forces, unit=unit,
            schema=_aux_io.sample_schema, models=_aux_io.models)
```

   Then make the snapshot call conditional (the `else` call is today's, unchanged):

```python
    if _aux_io is not None:
        WindowSnapshot(out_dir).snapshot(_seg_id, _win_snapshot_windows, cv1_type=primary_cv_mode(args), cv2_type=_cv2_type,
                                         kernel_identity=kernel_identity_for_run(args, secondary_cv_metadata),
                                         state_definition=_aux_io.state_definition, phase_kind=_aux_io.phase_kind,
                                         equilibrium_analysis_eligible=_aux_io.equilibrium_eligible)
    else:
        WindowSnapshot(out_dir).snapshot(_seg_id, _win_snapshot_windows, cv1_type=primary_cv_mode(args), cv2_type=_cv2_type,
                                         kernel_identity=kernel_identity_for_run(args, secondary_cv_metadata))
```

   `embed_cv_definition` uses the same kind strings the snapshot passes (`primary_cv_mode(args)`, `_cv2_type`), so Task 8's kind check holds by construction. `beta` (`:7916`), `_use_fast_cv_path` (`:7752`) and `_fast_aux_forces` (`:7762`) are already bound here.

3. **Writers** (`:8048-8055`). Add the keywords only for an auxiliary run, so the legacy calls are textually today's:

```python
    parquet_sample_writer = ParquetSampleWriter(
        out_dir / "samples" / _seg_id,
        flush_rows=int(getattr(args, "parquet_flush_rows", 5000) or 5000),
        **({"aux_schema": _aux_io.sample_schema, "aux_runtime": _aux_io.runtime} if _aux_io is not None else {}),
    )
    _exchange_seq = None
    _aux_parity = None
    _aux_event_kw = {}
    if _aux_io is not None:
        from .auxiliary_cv.ledger import EXCHANGE_EVENT_SCHEMA
        from .auxiliary_cv.offline import parity_context
        from .auxiliary_cv.runtime_io import ExchangeEventCounter
        _exchange_seq = ExchangeEventCounter()
        _aux_parity = parity_context(_aux_io.state_definition["windows"], beta)
        _aux_event_kw = {"event_schema": EXCHANGE_EVENT_SCHEMA}
    parquet_exchange_writer = ParquetExchangeWriter(
        out_dir / "exchanges" / _seg_id,
        flush_rows=int(getattr(args, "parquet_flush_rows", 5000) or 5000),
        **_aux_event_kw,
    )
```

4. **Sample fetch** (`_fetch_state`, `:8240-8258`). Next to Stage B's `aux_z = np.full(nrep, np.nan, ...)` add `aux_torsions = [None] * nrep` and `aux_pos_z = np.full(nrep, np.nan, dtype=np.float64)`. Both return statements of `_fetch_state` become:

```python
                    return (r, cv, ss, pe, v_pep, v_dih,
                            *(_aux_record_for_replica(r, sim) if _aux_io is not None
                              else (_aux_z_for_replica(r, sim), None, None)))
```

   (the second return has one less indentation level). The bare `_aux_z_for_replica(r, sim)` call stays in `_fetch_state`, so Stage B's `test_both_fetch_closures_observe_z_through_one_helper` holds; in an auxiliary run the record observer replaces it, so each carrier is read once (M4). The unpacking loop becomes:

```python
            for r, cv, ss, pe, v_pep, v_dih, z_aux, tor_aux, zpos_aux in _fetched_states:
                primary_values[r], ss_values[r], potentials_kj[r] = cv, ss, pe
                v_pep_kj[r], v_dih_kj[r] = v_pep, v_dih
                aux_z[r] = z_aux
                aux_torsions[r] = tor_aux
                if zpos_aux is not None:
                    aux_pos_z[r] = zpos_aux
            if _aux_io is not None:
                from .auxiliary_cv.runtime_io import check_runtime_parity
                from .auxiliary_cv.sample_schema import PARITY_TOLERANCE
                _sha = _aux_io.sample_schema.model_shas[0]
                check_runtime_parity(aux_z, aux_pos_z, beta=float(beta),
                                     k_max_kcal=_aux_parity["k_max_kcal"].get(_sha, 0.0),
                                     centers=_aux_parity["centers"].get(_sha, []),
                                     tolerance=PARITY_TOLERANCE[_aux_io.runtime["precision"]])
```

   `_fetch_exchange_state` is unchanged: it observes only the exchange z (Stage B's observer); Step 4 item 8 makes every exchange boundary a sample boundary, so the stored z is the exchange-matrix z (D8). Cost note (spec Section 10): with the fast path and no active state Stage B read no positions; the record observer reads them once per carrier per sample (torsions are required). Task 15 measures it with `--production-phase-timers` (`sample.fetch`).

5. **Sample write** (`:8379`). Add `**({"aux_z": [float(aux_z[r])], "torsions": aux_torsions[r]} if _aux_io is not None else {})` as the last argument of `parquet_sample_writer.write_sample(...)`. The stored z is the runtime exchange-matrix z (D8).

6. **`_attempt_window_swap`** (`:8544`). Add the keyword `aux_event: Optional[dict] = None` and define the helper at the top of the body:

```python
            def _aux_write_skip():
                if _aux_io is not None and aux_event is not None:
                    parquet_exchange_writer.write_event(
                        step=int(absolute_step), attempt_seq=_exchange_seq.next(int(absolute_step)),
                        selected_replica=int(aux_event["selected_replica"]),
                        replica_i=int(aux_event["selected_replica"]), replica_j=int(aux_event["selected_replica"]),
                        window_i=int(wi), window_j=int(wi), kind="skip", delta_e_kj=0.0, accepted=False,
                        log_q_forward=aux_event["log_q_forward"], log_q_reverse=aux_event["log_q_reverse"],
                        p_accept=aux_event["p_accept"], energy_version=_aux_io.energy_version,
                        assignments_after=list(assignments))
```

   Each of the three early returns (`if wi == wj: return attempt`, `if swap_candidate_replicas(...) is None: return attempt`, `if outcome is None: return attempt`) becomes `_aux_write_skip()` followed by `return attempt`. The helper is a no-op unless a Gibbs move passed `aux_event` (pair modes write events only for attempted swaps; their skipped pairs change no assignment). Replace the `parquet_exchange_writer.write_exchange(...)` call with:

```python
            if _aux_io is not None:
                _f = aux_event or {"selected_replica": i, "log_q_forward": float("nan"),
                                   "log_q_reverse": float("nan"), "p_accept": float("nan")}
                parquet_exchange_writer.write_event(
                    step=int(absolute_step), attempt_seq=_exchange_seq.next(int(absolute_step)),
                    selected_replica=int(_f["selected_replica"]), replica_i=int(i), replica_j=int(j),
                    window_i=int(wi), window_j=int(wj), kind="swap", delta_e_kj=float(outcome.delta_kj),
                    accepted=bool(outcome.accepted), log_q_forward=_f["log_q_forward"],
                    log_q_reverse=_f["log_q_reverse"], p_accept=_f["p_accept"],
                    energy_version=_aux_io.energy_version, assignments_after=list(assignments))
            else:
                parquet_exchange_writer.write_exchange(
                    step=int(absolute_step),
                    replica_i=int(i),
                    replica_j=int(j),
                    window_i=int(wi),
                    window_j=int(wj),
                    delta_e=float(outcome.delta_kj),
                    accepted=bool(outcome.accepted),
                )
```

7. **Gibbs branch** (`:8746-8779`).
   - Before each `continue` (no-candidates and stay):

```python
                        if _aux_io is not None:
                            from .auxiliary_cv.runtime_io import event_from_gibbs
                            parquet_exchange_writer.write_event(**event_from_gibbs(
                                prop, step=int(absolute_step), seq=_exchange_seq.next(int(absolute_step)),
                                selected_replica=rep, accepted=False, energy_version=_aux_io.energy_version,
                                assignments_after=list(assignments)))
```

   - Pass the move's proposal fields to the swap: `attempt = _attempt_window_swap(prop.current_window, prop.proposed_window, primary_values, bias_matrix_kj, absolute_step, attempt, p_override=prop.pacc, aux_event=_gibbs_aux_event)`, with, directly above it:

```python
                    _gibbs_aux_event = None
                    if _aux_io is not None:
                        from .auxiliary_cv.runtime_io import aux_event_fields
                        _gibbs_aux_event = {"selected_replica": rep, **aux_event_fields(prop)}
```

8. **Alignment.** Directly after `distance_interval = max(1, distance_interval)` (`:8785`):

```python
        if _aux_io is not None:
            from .auxiliary_cv.runtime_io import check_exchange_boundary_alignment
            for _note in check_exchange_boundary_alignment(
                    exchange_interval=int(args.exchange_interval), distance_interval=distance_interval,
                    traj_interval=effective_traj_interval, calib_steps=int(calib_steps)):
                print(f"WARNING: {_note}", flush=True)
```

9. **Checkpoint hooks.** Directly above `if bool(getattr(args, "resume", False)):` (`:8846`), define two closures. Every statement that touches auxiliary state is inside `if _aux_io is not None:` (gating test):

```python
        def _aux_checkpoint_block_now():
            if _aux_io is not None:
                from .auxiliary_cv.checkpoint import aux_checkpoint_block, read_aux_parameters
                from .parquet_manifest import load_manifest

                def _boundary(kind):
                    # M1: a manifest exists only after the first non-empty flush (e.g. SIGTERM before any sample).
                    m = load_manifest(out_dir / kind / _seg_id, expected_kind=kind) or {"generation": 0, "n_rows": 0}
                    return {"generation": int(m["generation"]), "n_rows": int(m["n_rows"])}

                return aux_checkpoint_block(
                    state_definition=_aux_io.state_definition, force_info=_aux_io.force_info,
                    assignments=list(assignments),
                    observed_params=[_sim_pool.submit(r, read_aux_parameters, sims[r].context, _aux_io.force_info).result()
                                     for r in range(nrep)],
                    topology_sha256=_aux_io.topology_sha256,
                    kernel_identity_digest=kernel_identity_for_run(args, secondary_cv_metadata)["digest"],
                    segment_id=_seg_id, ledger_anchor=_aux_anchor,
                    data_boundary={"samples": _boundary("samples"), "exchanges": _boundary("exchanges")})
            return None

        def _aux_pre_apply(manifest_in, assignments_in):
            """Context parameters as restored by loadCheckpoint, BEFORE the re-apply (C5/S8)."""
            if _aux_io is not None:
                from .auxiliary_cv.checkpoint import read_aux_parameters, verify_aux_resume
                from .auxiliary_cv.runtime_io import check_fixed_box, context_box_nm
                verify_aux_resume(
                    manifest_in, aux_enabled=True, state_definition=_aux_io.state_definition,
                    force_info=_aux_io.force_info, assignments=assignments_in,
                    observed_params=[_sim_pool.submit(r, read_aux_parameters, sims[r].context, _aux_io.force_info).result()
                                     for r in range(len(assignments_in))],
                    topology_sha256=_aux_io.topology_sha256,
                    kernel_identity_digest=kernel_identity_for_run(args, secondary_cv_metadata)["digest"])
                if _aux_io.state_definition["ensemble"] == "NVT":
                    check_fixed_box([_sim_pool.submit(r, context_box_nm, sims[r].context, unit).result()
                                     for r in range(len(assignments_in))],
                                    _aux_io.state_definition["fixed_box_vectors_nm"],
                                    label="boxes restored by loadCheckpoint")
            elif manifest_in.get("aux") is not None:
                # Legacy path: an inline check, no gareus.auxiliary_cv import (off path stays off the package).
                # The one guard that stops a non-aux job from resuming an aux checkpoint.
                raise RuntimeError("resume refused: this checkpoint binds auxiliary-CV states but this job has no "
                                   "--aux-cv-model; resume it with the same --aux-cv-model (spec D6)")
```

   - Pass `aux_pre_apply=_aux_pre_apply` to `load_production_checkpoint(...)` (`:8847`). C17: Context reads go through `_sim_pool.submit`.
   - Pass `aux_block=_aux_checkpoint_block_now()` to both `save_production_checkpoint(...)` calls (graceful shutdown `:8940`, periodic `:9044`); `flush_scalar_writers()` already runs first at both. Legacy: the helper returns None and the manifest gains no key.
   - In Task 5's auxiliary resume branch, after the re-seal print, add the ledger cross-check (the re-seal must run first: until then a `running` non-last parent is skipped by `_load_segmented_parquet`):

```python
                    from .auxiliary_cv.checkpoint import verify_aux_ledger
                    from .query import load_exchanges
                    verify_aux_ledger(manifest, load_exchanges(out_dir))
```

   - On resume, `verify_aux_resume` also checks that the rebuilt `state_definition` hashes equal to the checkpoint's and names the first differing fields (Task 6, M3: CV1/CV2 definitions, boost envelope, box, physical system). Path differences already handled: the physical sha excludes the default box (Task 11; resume rebuilds the System from `01_solvated_start.pdb`), and an NVT resume takes the fixed box from the checkpoint and verifies every restored box (`check_fixed_box`). If Task 14 still shows a refusal on identical inputs, the diff names the drifting field: fix its source (e.g. a path or a reconciliation that differs between the fresh and the resume path); never skip the check.

10. **Ledger anchor and test hook.** Directly before `_phase_timers.begin()` / `while prod_done < prod_total:` (`:8935`):

```python
        _aux_anchor = None
        _aux_test_fail_step = None
        if _aux_io is not None:
            _aux_anchor = {"segment_id": _seg_id, "start_step": int(calib_steps + prod_done),
                           "start_assignments": list(assignments)}
            if os.environ.get("GAREUS_TEST_FAIL_AT_PROD_STEP"):
                _aux_test_fail_step = int(os.environ["GAREUS_TEST_FAIL_AT_PROD_STEP"])
```

    and, as the first statement inside the loop body after the `_graceful_shutdown` block:

```python
            if _aux_test_fail_step is not None and prod_done == _aux_test_fail_step:
                raise RuntimeError(f"GAREUS_TEST_FAIL_AT_PROD_STEP={_aux_test_fail_step}: injected test failure")
```

    The environment is read once, only for auxiliary runs (ruling); legacy runs pay one `is not None` test per iteration and never read the variable.

- [ ] **Step 5: Extend Stage B's gating test** (`tests/test_aux_cv_production_wiring.py::test_aux_runtime_construction_in_run_gareus_is_gated_by_an_aux_condition`)

Replace its name tuple and ancestor test with:

```python
    for name in ("add_aux_cv_force", "load_aux_state_table", "aux_snapshot_rows", "check_feature_atoms",
                 "refuse_aux_population_change", "_check_instances", "canonical_topology_sha256",
                 # Stage C
                 "reseal_parent_for_resume", "verify_aux_ledger", "aux_checkpoint_block", "read_aux_parameters",
                 "aux_table_from_checkpoint", "physical_system_sha256", "build_runtime_state_definition",
                 "aux_io_runtime", "make_aux_record_observer", "write_event", "event_from_gibbs",
                 "aux_event_fields", "check_runtime_parity", "check_exchange_boundary_alignment", "parity_context",
                 "verify_aux_resume", "check_fixed_box", "context_box_nm"):
        for call in _calls(tree, name):
            node, gated = call, False
            while node in parents:
                node = parents[node]
                if isinstance(node, (ast.If, ast.IfExp)) and "aux" in ast.unparse(node.test):
                    gated = True
                    break
            assert gated, f"{name} at line {call.lineno} is not under an aux condition"
```

Also add `tests/test_sample_before_exchange_ordering.py` to the run below: `test_every_site_that_writes_assignments_is_accounted_for` counts writes to `assignments`; `write_event` never writes it, so no expected set changes (L5).

- [ ] **Step 6: Run the tests**

Run: `tests/test_aux_runtime_io.py tests/test_aux_cv_production_wiring.py tests/test_aux_cv_observation.py tests/test_aux_cv_exchange.py tests/test_aux_resume_guards.py tests/test_aux_ledger.py tests/test_aux_checkpoint.py tests/test_sample_before_exchange_ordering.py tests/test_store.py tests/test_aux_store_writers.py`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add gareus/auxiliary_cv/runtime_io.py gareus/production.py tests/test_aux_runtime_io.py tests/test_aux_cv_production_wiring.py
git commit -m "feat(cvaux): wire aux sample/event/snapshot/checkpoint I/O, runtime parity and ledger check into production"
```

---

### Task 14: End-to-end resume control on CPU (spec Section 16 Stage C exit)

**Files:**
- Create: `gareus/auxiliary_cv/resume_check.py`
- Test: `tests/test_aux_resume_check.py`
- Evidence: `docs/superpowers/plans/cvaux_stage_c_exit_evidence.md` (force-added)

**Interfaces:**
- Produces:
  - `ledger_completeness(events, *, selected_per_step: int | None) -> dict`. For a `gibbs-walk` ledger it counts events per exchange step. With `selected_per_step` given (nrep, or `--exchange-max-pairs-per-interval` when > 0), it requires equality and returns `{"ok", "steps", "bad_steps"}`.
  - `compare_runs(control_dir, resumed_dir, *, z_atol=1e-9, torsion_atol=1e-9) -> dict`. It loads `samples` (`load_samples`) and the exchange ledgers of both runs. It reports:
    - ledger equality as equal multisets of `(step, attempt_seq, assignment_sha256_after)` — a sound proxy for replay equality, since each checksum is the assignment after that event (L4: no replay from an anchor is claimed here; `verify_aux_ledger` replays on every resume);
    - ledger duplicates (must be 0);
    - sample row counts and duplicate `(step, replica, window_id)` keys per run, and key-set equality;
    - max |Δ aux z| and max |Δ torsion| (wrapped to (-π, π]) on common rows;
    - `ok`.

- [ ] **Step 1: Write the failing unit tests** (synthetic run dirs; no MD)

```python
# tests/test_aux_resume_check.py
import numpy as np

from gareus.auxiliary_cv.ledger import EXCHANGE_EVENT_SCHEMA
from gareus.auxiliary_cv.resume_check import compare_runs, ledger_completeness


def _ledger(run, decisions):
    from gareus.store import ParquetExchangeWriter, SegmentRegistry
    reg = SegmentRegistry(run)
    sid = reg.open_segment("run", None, 1)
    w = ParquetExchangeWriter(run / "exchanges" / sid, flush_rows=1, event_schema=EXCHANGE_EVENT_SCHEMA)
    for step, seq, kind, r in decisions:
        w.write_event(step=step, attempt_seq=seq, selected_replica=r, replica_i=r, replica_j=r, window_i=r,
                      window_j=r, kind=kind, delta_e_kj=0.0, accepted=False, log_q_forward=0.0,
                      log_q_reverse=0.0, p_accept=0.0, energy_version="v3", assignments_after=[0, 1])
    w.close(); reg.close_segment(sid, 1000)


def test_ledger_completeness_counts_one_event_per_selected_carrier(tmp_path):
    from gareus.query import load_exchanges
    _ledger(tmp_path, [(100, 0, "stay", 0), (100, 1, "skip", 1), (200, 0, "stay", 1)])
    rep = ledger_completeness(load_exchanges(tmp_path), selected_per_step=2)
    assert rep["ok"] is False and rep["bad_steps"] == {200: 1}


def test_compare_runs_identical(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(); b.mkdir()
    for run in (a, b):
        _ledger(run, [(100, 0, "stay", 0), (100, 1, "stay", 1)])
    rep = compare_runs(a, b)
    assert rep["ok"] and rep["ledger_duplicates"] == 0


def _patched(monkeypatch, control, resumed):
    ev = {"step": np.array([100]), "attempt_seq": np.array([0]), "assignment_sha256_after": np.array(["a"])}
    monkeypatch.setattr("gareus.query.load_exchanges", lambda p: ev)
    monkeypatch.setattr("gareus.query.load_samples", lambda p: control if "ctl" in str(p) else resumed)


def test_compare_runs_detects_duplicate_sample_rows(monkeypatch, tmp_path):
    import gareus.auxiliary_cv.resume_check as rc
    base = {"step": np.array([100, 200]), "replica": np.array([0, 0]), "window_id": np.array([0, 1])}
    _patched(monkeypatch, base, {k: np.concatenate([v, v[-1:]]) for k, v in base.items()})
    out = rc.compare_runs(tmp_path / "ctl", tmp_path / "res")
    assert out["sample_duplicate_keys"]["resumed"] == 1 and out["sample_rows"] == {"control": 2, "resumed": 3}
    assert out["ok"] is False


def test_compare_runs_detects_torsion_drift_with_wrapping(monkeypatch, tmp_path):
    import gareus.auxiliary_cv.resume_check as rc
    base = {"step": np.array([100, 200]), "replica": np.array([0, 0]), "window_id": np.array([0, 1]),
            "tor_000": np.array([np.pi - 1e-12, 0.2]), "aux_z_00": np.array([0.5, 0.6])}
    same = dict(base, tor_000=np.array([-np.pi + 1e-12, 0.2]))           # equal angle across the branch cut
    _patched(monkeypatch, base, same)
    assert rc.compare_runs(tmp_path / "ctl", tmp_path / "res")["ok"] is True
    drift = dict(base, tor_000=np.array([np.pi - 1e-12, 0.3]))
    _patched(monkeypatch, base, drift)
    out = rc.compare_runs(tmp_path / "ctl", tmp_path / "res")
    assert out["ok"] is False and abs(out["max_abs_dtorsion"] - 0.1) < 1e-12
```

- [ ] **Step 2: Implement `resume_check.py`**

```python
# gareus/auxiliary_cv/resume_check.py
"""Compare an interrupted+resumed auxiliary run with its uninterrupted control (spec Section 16 Stage C)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np


def ledger_completeness(events, *, selected_per_step: Optional[int]) -> dict[str, Any]:
    step = np.asarray(np.ma.getdata(events["step"])).astype(np.int64)
    steps, counts = np.unique(step, return_counts=True)
    bad = {int(s): int(c) for s, c in zip(steps, counts)
           if selected_per_step is not None and int(c) != int(selected_per_step)}
    return {"ok": not bad, "steps": int(steps.size), "bad_steps": bad}


def _keys(samples) -> list[tuple]:
    return list(zip(np.asarray(samples["step"]).tolist(), np.asarray(samples["replica"]).tolist(),
                    np.asarray(samples["window_id"]).tolist()))


def _key_rows(samples) -> dict[tuple, int]:
    return {k: i for i, k in enumerate(_keys(samples))}


def compare_runs(control_dir, resumed_dir, *, z_atol: float = 1e-9, torsion_atol: float = 1e-9) -> dict[str, Any]:
    from ..query import load_exchanges, load_samples
    out: dict[str, Any] = {}
    ec, er = load_exchanges(Path(control_dir)), load_exchanges(Path(resumed_dir))

    def _events(ev):
        return list(zip(np.asarray(np.ma.getdata(ev["step"])).tolist(),
                        np.asarray(np.ma.getdata(ev["attempt_seq"])).tolist(),
                        np.asarray(ev["assignment_sha256_after"]).tolist()))

    pc, pr = _events(ec), _events(er)
    out["ledger_duplicates"] = len(pr) - len({(s, q) for s, q, _ in pr})
    out["ledger_equal"] = sorted(pc) == sorted(pr)
    sc, sr = load_samples(Path(control_dir)), load_samples(Path(resumed_dir))
    if sc and sr:
        keys_c, keys_r = _keys(sc), _keys(sr)
        # Board condition 5: a dict/set comparison cannot see phantom duplicate rows (a reseal regression
        # with a clean ledger). Compare row counts and duplicate keys explicitly.
        out["sample_rows"] = {"control": len(keys_c), "resumed": len(keys_r)}
        out["sample_duplicate_keys"] = {"control": len(keys_c) - len(set(keys_c)),
                                        "resumed": len(keys_r) - len(set(keys_r))}
        kc, kr = _key_rows(sc), _key_rows(sr)
        out["sample_keys_equal"] = (set(kc) == set(kr) and len(keys_c) == len(keys_r)
                                    and out["sample_duplicate_keys"]["resumed"] == 0
                                    and out["sample_duplicate_keys"]["control"] == 0)
        common = sorted(set(kc) & set(kr))
        ic, ir = [kc[k] for k in common], [kr[k] for k in common]
        if "aux_z_00" in sc and "aux_z_00" in sr and common:
            zc = np.asarray(sc["aux_z_00"], dtype=float)[ic]
            zr = np.asarray(sr["aux_z_00"], dtype=float)[ir]
            out["max_abs_dz"] = float(np.nanmax(np.abs(zc - zr)))
        tor = sorted(c for c in sc if c.startswith("tor_") and c in sr)
        if tor and common:
            d = [np.abs(np.angle(np.exp(1j * (np.asarray(sc[c], dtype=float)[ic] - np.asarray(sr[c], dtype=float)[ir]))))
                 for c in tor]
            out["max_abs_dtorsion"] = float(np.nanmax(np.stack(d)))
    else:
        out["sample_keys_equal"] = True
    out["ok"] = (out["ledger_duplicates"] == 0 and out["ledger_equal"] and out["sample_keys_equal"]
                 and out.get("max_abs_dz", 0.0) <= z_atol and out.get("max_abs_dtorsion", 0.0) <= torsion_atol)
    return out
```

- [ ] **Step 3: Run the unit tests**

Run: `tests/test_aux_resume_check.py`
Expected: PASS

- [ ] **Step 4: Real CPU control runs** (dispatch through `opencode`; no repo edits; scratch directory `S=/tmp/cvaux_stage_c_e2e`, never under `RUNS/`)

All commands run in the worktree with `python -m gareus`. Common flags `F` (smoke-parsed at HEAD on 2026-10-08 with `parse_args`; `--production-steps` reaches `gamd_production_steps`, `cli.py:1846`):

```
--seq GA --cv1 distance --window-mode manual --exchange-mode gibbs-walk
--platform CPU --seed 7 --minimize-iterations 200 --npt-steps 2000 --us-pull-steps-per-window 500
--production-steps 6000 --checkpoint-interval 2000 --exchange-interval 500 --distance-output-interval 250
--report-interval 250 --traj-interval 500
```

Every interval is aligned (`exchange % distance == 0`, `exchange % traj == 0`, Task 13 item 8); the exception step 3500 is on the 250-step loop grid; checkpoints fall at 2000 and 4000.

1. **Prep run (non-aux) for the production topology.** Write `S/W0.csv` (header `primary_cv_center,primary_cv_k_kcal`; rows `4.0,2.0` / `5.0,2.0` / `5.0,2.0` / `5.0,2.0`) and run `python -m gareus F --run-mode cmd --windows-2d-csv S/W0.csv --out S/RUN_P` (with `--production-steps 500` overriding F's value). It leaves `S/RUN_P/01_solvated_start.pdb`, the file every resume rebuilds the topology from (`gareus/checkpoints.py:77-82`).
2. **Model bound to that topology** (`check_feature_atoms(topology_sha256=canonical_topology_sha256(...))` refuses any other):

```python
# S/make_model.py   (run: python S/make_model.py S/RUN_P/01_solvated_start.pdb S/M.json)
import sys
sys.path.insert(0, "tests")
from openmm import app
from aux_cv_fixture import model_payload
from gareus.auxiliary_cv.model import AuxModel
from gareus.auxiliary_cv.runtime import canonical_topology_sha256
from gareus.energy_decomposition import peptide_atom_groups_from_topology
from gareus.mbar_analysis.thermo_frames import backbone_torsion_quads

top = app.PDBFile(sys.argv[1]).topology
quads, labels = backbone_torsion_quads(top, peptide_atom_groups_from_topology(top, "all-peptide")[1])
blocks = [lab.split("-")[0] for lab in labels]
coeffs = [1.0] + [0.0] * (2 * len(quads) - 1)
draft = AuxModel.from_mapping(model_payload(quads, coeffs, blocks=blocks, label="stage-c-e2e"))
sha = canonical_topology_sha256(top, draft)
AuxModel.from_mapping(model_payload(quads, coeffs, blocks=blocks, label="stage-c-e2e",
                                    topology_sha256=sha)).write(sys.argv[2])
print(sha, labels)
```

   If a fresh aux run below refuses with "aux model was built for topology X, this run's topology is Y", the in-memory topology and its PDB round trip differ in a peptide-chain atom, residue or chain index. Then a resume (PDB topology) and a fresh run (in-memory topology) can never both pass Stage B's binding: record X and Y and **stop** (a Stage B binding defect to fix first, e.g. computing the canonical digest from `01_solvated_start.pdb` on both paths). Do not edit the check. The same applies if a resume is refused on `topology identity changed` (Task 6, whole-topology `topology_identity_sha256`).
3. **State tables.** `S/W.csv` reuses Stage B's tested slot layout (`tests/test_aux_cv_state_table.py` `HEADER`/`ROWS`), with distance centres:

```
primary_cv_center,primary_cv_k_kcal,aux_center,aux_k_kcal_mol,state_instance_id,state_role,spawn_parent_state_id,matched_additional_slot_id
4.0,2.0,,0.0,ord-0,ordinary,,
5.0,2.0,,0.0,ord-1,ordinary,,
5.0,2.0,0.0,1.2,aux-0,auxiliary,ord-1,slot-0
5.0,2.0,,0.0,sham-0,sham,ord-1,slot-0
```

   `S/W2.csv` adds the CV2 columns `secondary_cv_center,secondary_cv_k_kcal_mol` with `0.5,5.0` on every row (the sham's baseline equals its auxiliary partner's, as `_check_slot_pairing` requires).
4. **Configurations** (all with `F --aux-cv-model S/M.json`; production ensemble NPT, the default):
   - **A** (CV1 only, cMD): `--run-mode cmd --windows-2d-csv S/W.csv`.
   - **B** (CV2, cMD; exercises H1 and the CV2 part of the state-definition hash): `--run-mode cmd --cv2 alpha --windows-2d-csv S/W2.csv`.
   - **C** (CV2 + Pep-GaMD; exercises the boost-envelope part of the hash): `--run-mode gamd --gamd-boost-type pep-gamd-lower-dual --cv2 alpha --windows-2d-csv S/W2.csv --equil-steps 2000 --gamd-cmd-steps 2000 --gamd-averaging-window 1000 --gamd-multiwindow-recon-prep-steps 200 --gamd-multiwindow-recon-cmd-steps 1000 --gamd-multiwindow-recon-steps 1000 --gamd-recon-boosted-iters 1`. If the run refuses at startup because `calib_steps` is not a multiple of `--traj-interval` (Task 13 alignment), record the printed `calib_steps` and rerun all C runs with a `--traj-interval` that divides both it and 500. CPU wall time of C is not known in advance; dispatch it in the background and poll.
   - **D** (NVT fixed box): configuration A plus `--production-ensemble nvt`. Either the fresh run refuses at segment start with "an NVT auxiliary state definition needs one fixed box" (the replicas' pulled start boxes differ: record it; the NVT box rule then holds by refusal), or it runs and its exception-path resume must pass like A (the box comes from the checkpoint and every restored box is verified).
5. **Runs per configuration** (`<cfg>` = A, B, C):
   - **Control:** `--out S/RUN_<cfg>_C`.
   - **Interrupted by exception:** `GAREUS_TEST_FAIL_AT_PROD_STEP=3500 python -m gareus ... --out S/RUN_<cfg>_X`; the job exits non-zero at prod step 3500 (`finalize_segment` seals the parent at the crash step). Record the parent's `end_step` from `S/RUN_<cfg>_X/segments.json`. Then `env -u GAREUS_TEST_FAIL_AT_PROD_STEP python -m gareus ... --out S/RUN_<cfg>_X --resume` (the variable must be unset, or the resumed run fails again at 3500). This drives `reseal_parent_for_resume` (cut to the 2000 checkpoint), `verify_aux_resume` (parameters before re-apply, state-definition hash), `verify_aux_ledger` and, for B and C, `verify_kernel_identity_on_resume`. Record the parent's `end_step` again.
   - **Interrupted by graceful signal (A only):** start with `--out S/RUN_A_R`; send SIGTERM once `S/RUN_A_R/progress.jsonl` shows a production step ≥ 3000; the graceful path checkpoints at the current step and exits; rerun with `--resume`.
   - **Refusals (A only), on copies:** every job rewrites `run_manifest.json` (`main()` → `initialize_run_manifest`) before `run_gareus` refuses, so run them against `cp -a` copies, never the directories checked below. `cp -a S/RUN_A_X S/RUN_A_X_ref && python -m gareus F --run-mode cmd --windows-2d-csv S/W0.csv --out S/RUN_A_X_ref --resume` (no model) must exit with "resume it with the same --aux-cv-model"; `cp -a S/RUN_P S/RUN_P_ref && python -m gareus F --run-mode cmd --windows-2d-csv S/W.csv --aux-cv-model S/M.json --out S/RUN_P_ref --resume` must exit with "ran without auxiliary-CV states" (ruling B1, both sides). Record that `S/RUN_P_ref/run_manifest.json` now carries `aux_cv_model_sha256` (Task 12 residual).
6. **Checks** (per configuration that ran):

```python
# S/check.py   (run: python S/check.py S/RUN_A_C S/RUN_A_X 4)
import json
import sys
from pathlib import Path

import numpy as np

from gareus.auxiliary_cv.resume_check import compare_runs, ledger_completeness
from gareus.mbar_analysis.loaders import load_parquet
from gareus.query import load_exchanges

control, resumed, nrep = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
print("compare", compare_runs(control, resumed))
print("ledger", ledger_completeness(load_exchanges(resumed), selected_per_step=nrep))
shas = {p.name: json.loads(p.read_text())["state_definition_sha256"] for p in sorted((resumed / "windows").glob("seg_*.json"))}
print("state_definition_sha256 per segment", shas, "equal:", len(set(shas.values())) == 1)
print("segments", json.loads((resumed / "segments.json").read_text()))
d = load_parquet(resumed, allow_ineligible_aux_segments=True)        # pilot / ineligible by default
print("u_nk", d.u_nk.shape, "finite:", bool(np.isfinite(d.u_nk).all()))
```

   Gate: `compare.ok` True, `ledger.ok` True, one `state_definition_sha256` across the parent and child snapshots (M3: CV1, CV2 and boost definitions resume-stable), finite `u_nk`. On CPU, `max_abs_dz` and `max_abs_dtorsion` should be 0. A bitwise difference after the resume step with equal ledgers means a nondeterministic CPU reduction: record it in the evidence file, never loosen the tolerances silently. If a resume is refused on `state_definition_sha256`, the message names the drifting field (Task 6): record it and fix its source in a follow-up task before the exit gate.

Record every output in `docs/superpowers/plans/cvaux_stage_c_exit_evidence.md`, with a **"Resume re-seal scope"** paragraph (ruling B4): the re-seal is auxiliary-only; the legacy exception-crash double count is a known residual left for a separate PR; evidence = the `RUN_*_X` parent `end_step` before (3500 + calib) and after (2000 + calib) the resume.

- [ ] **Step 5: Commit**

```bash
git add gareus/auxiliary_cv/resume_check.py tests/test_aux_resume_check.py
git add -f docs/superpowers/plans/cvaux_stage_c_exit_evidence.md
git commit -m "test(cvaux): resume control vs uninterrupted run (exception and graceful paths, CV2 and Pep-GaMD)"
```

---

### Task 15: GPU parity and cost on aurum2 (D9, spec Sections 10 and 17)

**Files:**
- Create: `gareus/auxiliary_cv/parity_report.py`
- Test: `tests/test_aux_parity_report.py`

**Interfaces:**
- Produces: `parity_report(run_dir) -> dict`. Per segment it gives the recorded platform/precision, rows, max |stored z − offline z|, and the max conservative reduced disagreement at the run's strongest k (`parity_violation`). It also gives the tolerance that applies, and `ok`.

- [ ] **Step 1: Unit test on a synthetic segment** (reuse `tests/test_aux_loaders.py`'s builders)

```python
# tests/test_aux_parity_report.py
import numpy as np

from test_aux_loaders import _run          # synthetic aux run with Reference/double runtime
from gareus.auxiliary_cv.parity_report import parity_report


def test_parity_report_on_exact_data(tmp_path):
    _run(tmp_path, historical=False)
    rep = parity_report(tmp_path)
    (seg, row), = rep["segments"].items()
    assert row["precision"] == "double" and row["max_abs_dz"] == 0.0 and row["ok"] is True
```

- [ ] **Step 2: Implement `parity_report.py`**

```python
# gareus/auxiliary_cv/parity_report.py
"""Measured stored-vs-offline aux z parity per segment (GPU validation, spec Section 17)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def parity_report(run_dir) -> dict[str, Any]:
    from ..correctness._io import IntegrityError
    from ..correctness.export import R_KJ_MOL_K
    from ..correctness.state_identity import canonical_state_definition
    from ..query import load_samples, load_windows_metadata
    from .offline import (_float_column, parity_context, parity_violation, registry_model,
                          segment_aux_runtime, segment_aux_schemas)
    from .evaluate import z_from_dihedrals
    from .sample_schema import PARITY_TOLERANCE
    run_dir = Path(run_dir)
    schemas, runtimes = segment_aux_schemas(run_dir), segment_aux_runtime(run_dir)
    out: dict[str, Any] = {"segments": {}}
    for seg, schema in schemas.items():
        if schema is None:
            continue
        # Board condition 1. load_windows_metadata(run_dir, segment_id=None) (gareus/query.py:354) takes
        # both arities, but the one-argument form returns only the LATEST segment's snapshot. Each segment
        # is graded against its own frozen state definition.
        meta = load_windows_metadata(run_dir, seg) or {}
        if not meta.get("state_definition"):
            raise IntegrityError(f"segment {seg} has auxiliary samples but no frozen state_definition snapshot")
        state = canonical_state_definition(meta["state_definition"])
        beta = 1.0 / (R_KJ_MOL_K * float(state["temperature_k"]))
        ctx = parity_context(state["windows"], beta)
        s = load_samples(run_dir, segment_ids=[seg])
        n = len(np.asarray(s["cv1"]))
        theta = np.stack([_float_column(s, c, n) for c in schema.torsion_columns], axis=1)
        sha = schema.model_shas[0]
        model = registry_model(state, sha, where=f"parity_report ({seg})")
        z_off = z_from_dihedrals(theta[:, schema.model_basis_index(model)], model)
        z_run = _float_column(s, schema.z_columns[0], n)
        both = np.isfinite(z_off) & np.isfinite(z_run)
        du = parity_violation(z_run[both], z_off[both], beta=beta, k_max_kcal=ctx["k_max_kcal"].get(sha, 0.0),
                              centers=ctx["centers"].get(sha, []))
        prec = (runtimes.get(seg) or {}).get("precision")
        tol = PARITY_TOLERANCE.get(prec, 0.0)
        out["segments"][seg] = {"platform": (runtimes.get(seg) or {}).get("platform"), "precision": prec,
                                "rows": int(n), "max_abs_dz": float(np.max(np.abs(z_run[both] - z_off[both]))) if both.any() else 0.0,
                                "max_reduced": float(du.max()) if du.size else 0.0, "tolerance": tol,
                                "ok": bool(du.size == 0 or du.max() <= tol)}
    out["ok"] = all(r["ok"] for r in out["segments"].values())
    return out
```

`load_samples(run_dir, segment_ids=[seg])` uses the existing `segment_ids` filter of `_load_segmented_parquet`.

- [ ] **Step 3: Run the unit test**

Run: `tests/test_aux_parity_report.py`
Expected: PASS

- [ ] **Step 4: aurum2 measurement** (user or opencode; never the live chignolin_10 CODE_DIR)

1. **Deploy.** Rsync this branch to a scratch CODE_DIR on aurum2, following the aurum2 deploy procedure (`reference_aurum2_deploy`: rsync, never git push, never `--delete`). Do not touch `~/2026_peptide_sampler` while c10 uses it.
2. **Parity runs.** Run Task 14's configuration A control command three times: `--platform CUDA` with precision `mixed` (the default), `--platform OpenCL`, and `--platform CUDA` with `double`. Use a 24-window CSV (aux:ordinary 4:20, every row with instance columns, each auxiliary row's parent an ordinary row), a model built for that run's topology (Task 14 Step 4.2) and `--production-phase-timers`.
   - **GPU checkpoint round trip:** repeat Task 14's exception path (configuration A, `GAREUS_TEST_FAIL_AT_PROD_STEP`, then `--resume`) once on CUDA `mixed`. `verify_aux_resume` passes only if `loadCheckpoint` restores the auxiliary globals within `_REL = 1e-12` (Task 6, board condition 8). If it refuses, record the measured deviation and stop (spec Section 17); do not loosen `_REL`.
3. **Cost runs.** Run the same 24 windows with `--aux-cv-model` omitted (same CSV minus the aux and instance columns) as the cost baseline.
4. **Record.** Write to the evidence file:
   - `parity_report(...)` for each run;
   - `sample.fetch` and `md` seconds per 2000 steps from `production_phase_timers.json`, aux vs baseline;
   - node ns/day.
5. **Gate.** Every parity run must be `ok` at its recorded precision's tolerance (1e-6 double, 1e-4 mixed). If mixed precision exceeds 1e-4, do **not** loosen `PARITY_TOLERANCE` here. Record the measured value and stop; a tolerance change is a spec decision (Section 17: "loosened only after a documented numerical-error study").
6. **Commit.** Commit the evidence file update (`git add -f`).

---

### Task 16: Stage C exit gate, handoff note and Stage D prerequisites

**Files:**
- Modify: `CLAUDE.md`: add a `## CVaux Stage C` section directly after the `## CVaux Stage B` section, and update the stale Stage B bullets.

- [ ] **Step 1: Run the Stage C set, the touched Stage B tests and the legacy regressions**

Run: `tests/test_aux_manifest_payload_schema.py tests/test_aux_sample_schema.py tests/test_aux_store_writers.py tests/test_aux_ledger.py tests/test_aux_checkpoint.py tests/test_aux_eligibility.py tests/test_aux_export.py tests/test_aux_window_snapshot.py tests/test_aux_loaders.py tests/test_aux_mbar_duplicates.py tests/test_aux_runtime_definition.py tests/test_aux_resume_guards.py tests/test_aux_runtime_io.py tests/test_aux_resume_check.py tests/test_aux_parity_report.py tests/test_aux_cv_cli.py tests/test_aux_cv_kernel_identity.py tests/test_aux_cv_production_wiring.py tests/test_aux_cv_exchange.py tests/test_aux_cv_final_review_pins.py tests/test_aux_cv_state_table.py tests/test_aux_cv_observation.py tests/test_segment_finalization.py tests/test_store.py tests/test_parquet_manifest_transactions.py tests/test_query.py tests/test_query_reconstruct_bias_matrix_nan_guard.py tests/test_lambda_ladder_mbar.py tests/test_low_memory_complete.py tests/test_loader_masked_cv2_nan.py tests/test_sample_before_exchange_ordering.py tests/test_resume_and_poincare_regressions.py tests/test_checkpoint_keep_generations_wiring.py tests/test_npt_production_integration.py`
Expected: all PASS. Also required: the Task 14 evidence (`compare_runs ... ok: True` on both interruption paths for configuration A, on the exception path for B and C, one `state_definition_sha256` per resumed run, both B1 refusals) and the Task 15 parity evidence. Together these are spec Section 16's Stage C exit.

- [ ] **Step 2: Update `CLAUDE.md`**

In the `## CVaux Stage B` section:
- first bullet: replace "(+ required `--aux-cv-allow-unpersisted` until Stage C)" with "(Stage C removed `--aux-cv-allow-unpersisted`; resume/extend allowed, bound by `refuse_resume_of_aux_campaign(out_dir, args)`)", and drop `--resume/--extend` from its refused list;
- the "Stage-B data is unanalysable" bullet: append "Stage C keeps Stage-B-era segments `aux_unpersisted` (no `atlas-aux-samples-v1` payload); `load_parquet` refuses them unless `exclude_segments_without_aux_features=True`; the snapshot key `aux_sample_schema` was never written and Stage C deleted `AUX_PERSISTED_SNAPSHOT_KEY` (persistence = the samples manifest `payload_schema`)";
- the "No end-to-end `run_gareus` MD smoke test" bullet: replace "Stage C Task 13" with "Stage C Task 14";
- the "Stage C must" bullet: replace it with "Stage C must: done -- see the CVaux Stage C section (each item mapped there)."

Then add after the Stage B section:

```markdown
## CVaux Stage C (storage, resume, strict pooling; plan `docs/superpowers/plans/2026-10-07-cvaux-stage-c-storage-resume-pooling.md` v3)

- Samples (aux runs only): `tor_###` (full backbone basis, OpenMM theta, float64) + `aux_z_##` (= runtime exchange-matrix z, float64) + `observation_phase`; manifest/table payload `atlas-aux-samples-v1` with per-segment `runtime` {platform, precision} selecting the parity tolerance (1e-6 double / 1e-4 mixed, reduced energy). One positions read per carrier per sample (`runtime_io.make_aux_record_observer`) gives torsions + positions z; runtime parity is checked every sample and never raises without an active state (sham arms record).
- Exchanges: ordered events `atlas-exchange-events-v1` (swap/stay/no_candidates/skip, attempt_seq, log q fwd/rev, p_accept, energy_version, assignment_sha256_after; one event per Gibbs decision; `ledger.replay_assignments`).
- Checkpoints: `manifest["aux"]` (`atlas-aux-checkpoint-v1`: frozen state table + hash, force identity, Context params verified at save AND after loadCheckpoint before re-apply, whole-topology identity, kernel digest, ledger anchor, data boundary). A changed state definition is refused naming the first differing fields. Ledger replay must reproduce the checkpoint assignments.
- Resume: `refuse_resume_of_aux_campaign(out_dir, args)` first on every `--resume` (fast and non-fast): aux campaign without/with another model, or non-aux campaign with a model, refused; legacy no-op. Fast resume takes the aux table from the checkpoint (`aux_table_from_checkpoint`), sets the NaN-safe window key. `verify_kernel_identity_on_resume` compares against `exchange_energy_version_for_args(args)`. Re-seal of a crashed parent at the checkpoint (`store.reseal_parent_for_resume`) runs for aux runs only (user ruling B4); legacy seal verbatim. **Known legacy residual (separate PR):** after an exception crash + resume a legacy run still pools the parent's post-checkpoint rows.
- Eligibility: aux segments VERIFIED only with the samples payload + a frozen v2 snapshot naming the model; the samples manifest is read (raw JSON) only for aux snapshots and snapshot-less segments; a snapshot-less segment of an aux run is `aux_unpersisted`.
- Pooling: strict exporter + `load_parquet` only (one eligible fixed state; `exclude_segments_without_aux_features` only for Stage-B-era segments; legacy-kernel segments in an aux directory (mixed v2/v3_aux) and snapshot-less segments always refused; `allow_ineligible_aux_segments`, `aux_on_incomplete`). `load_npz`/`load_csv` (`_refuse_aux_run`, RuntimeError), the adaptive branch of `load_data`, round augmentation, union loader/builder (incl. pilot dirs) and the legacy NPZ export refuse aux runs (`kernel_identity.AuxPoolingRefused`); detection is pure JSON at fixed depth (`windows/`, `*/windows/`, `*/*/windows/`), never imports `gareus.auxiliary_cv` on legacy runs.
- Tables: `load_aux_state_table` refuses instance-less tables (v2 needs instances on every row). CLI: `--aux-phase-kind` (default pilot), `--aux-equilibrium-eligible` (production only); both appear default-valued in every job's recorded args.
- Evidence: `docs/superpowers/plans/cvaux_stage_c_exit_evidence.md` (CPU resume control A/B/C, GPU parity and cost). Tests `tests/test_aux_*.py` (Stage C) plus the edited Stage B pins.
- Stage D prerequisites: see the plan's table (finite-timestep validation, NPT controlled distribution, production-scale cost, diagnostics audit, equilibration policy, boundary frames, stuck-rescue refusal, legacy double-count PR).
```

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs(cvaux): Stage C handoff note; Stage B bullets updated"
```

---

## Stage D prerequisites (explicit deferrals with owners, D13)

| Item | Spec | Owner / when |
|---|---|---|
| Finite-timestep validation of strong aux restraints against a shorter-step reference | 3.4 | Stage D, before any efficacy run; the timestep stays unchanged until then |
| NPT controlled-distribution test (aux restraint, `PepGamdLowerDualNptTargetAdapter`) | 3.4, 17 | Stage D engineering run; Stage B documents that molecular-centroid scaling leaves torsions invariant |
| Multi-Context cost benchmark at production scale (236 contexts, MPS) | 10 | Stage D engineering run; Task 15 measures only a 24-window run |
| Diagnostics audit: `ladder_overlap`, `pmf_ladder_crosscheck` (complete biases on both sides plus an ordinary-λ0 crosscheck, §8), `other_rung_same_centre`, `gareus_report` | 15 | Stage D; until then these report aux runs as unavailable (they never see aux data: their loaders refuse) |
| Equilibration and phase_kind policy (preparation ramp, excluded equilibration segment, burn-in sensitivity) | 6 | Stage D admission; Stage C defaults new aux runs to `pilot` / ineligible |
| Structural-label boundary stream / exchange-grid frames when trajectories are off | 7, 12 | Stage D (Task 13 warns) |
| `stuck-rescue` (`production.py` ~9000, `setPositions` from another replica) is a non-equilibrium move; refuse it with `--aux-cv-model` | 4, 6 | Stage D (preflight L2) |
| Legacy exception-crash double count after resume (re-seal of non-running interrupted parents for non-aux runs) | — | Separate PR (user ruling B4) |
| Model binding to the run topology, two hashes kept: Stage B `canonical_topology_sha256` (model atom map: chains holding feature atoms, solvent excluded; stored in `AuxRuntime.topology_sha256` and the kernel identity as `aux_topology_sha256`, so the checkpoint's kernel digest covers it) and Stage C `topology_identity_sha256` (whole topology, `AuxIORuntime.topology_sha256`, checked on resume) | 5, 9 | Done in Stages B/C; Stage D ports the c10 model with the canonical digest |

---

## Self-review record

**Verifier finding → change (v2; still valid unless the v3 record below supersedes it):**

| Finding | Change |
|---|---|
| C1/S2 adapter unspecified | Task 11 `build_runtime_state_definition` + `aux_io_runtime` (physical identity, CV embedding, boost, ensemble, rows + instances, applied-restraint check, phase/eligibility, energy version, topology identity) |
| C2/S6 no positions on the fast path | Task 13 `make_aux_record_observer`: one positions read per carrier per sample (M4); Task 15 benchmarks it |
| C3/S3 resume blocked | Task 11 removes the flag and the CLI refusal; Task 12 replaces the unconditional production refusal by a mismatch-only one on every resume (B1) and loads the table from the checkpoint (B2); Task 14 runs real resumes against controls on both interruption paths |
| C4/S7 crash tails | Task 5 `reseal_parent_for_resume`, driven by `finalize_segment` in the test; production uses it **for auxiliary runs only** (user ruling B4); legacy residual filed for a separate PR |
| C5/S8 tautological resume check | Task 6 checks at save time, then reads before re-apply and compares to expected and saved values (`aux_pre_apply`); `verify_aux_ledger` |
| C6/S4 mixed precision unreachable | Task 2/3 `runtime` block per segment; Tasks 8/9 per-row tolerance; Task 15 GPU measurement |
| S5 stored-z self-contradiction | Global Constraint plus Task 13 item 5: the stored z is the runtime exchange z; positions z only for runtime parity |
| C7 NaN rows dropped by `clean()` | Task 9 `aux_on_incomplete` before `clean()`, with a test |
| C8 beta constant | Task 8 `BETA = 1/(R_KJ_MOL_K*300)` |
| C9 pilot dirs | Task 9 scans every pilot dir, with a test |
| C10 no fixed-state check / S10 npz/csv bypass | Task 9 `validate_fixed_state_segments` across aux segments; refusals in `load_npz`/`load_csv`/`load_data`/augmentation |
| C11 early returns / log(0) | Task 4 `skip` kind; Task 13 `_aux_write_skip` on all three early returns (call count pinned, D1); `_log` |
| C12 checkpoint bindings | Task 6 topology identity, kernel digest, segment id, ledger anchor, data boundary |
| C13 masked legacy events | Task 5 explicit masked/None detection, with a test |
| C14/S9 alignment inputs | Task 13 resolved `distance_interval`, `effective_traj_interval`, `calib_steps` |
| C15 pins | Task 3/4 schema strings with dtypes; Task 8 snapshot sha and export `input_signature` pins (re-verified at HEAD); Task 1 manifest key sets; Task 6 off-path hook test; Task 5 legacy-seal pin |
| C16 n kept/excluded | Task 8 manifest `n_kept`/`n_excluded` |
| C17 off-worker reads | Task 13 save and load reads use `_sim_pool.submit` |
| C18/S13 merge unwired | Task 10: library utility, explicitly not wired in MVP (separate origins); test on unequal independent data |
| C19 cv2 kind | Task 8 snapshot asserts cv1/cv2 kind equality |
| S1/D5 Stage B data misread | Task 7 verifies only persisted, bound segments; Task 9 `snapshot_has_aux` keys on kernel identity and Stage B rows; Stage-B-era segments refused unless explicitly excluded |
| S11 topology hash | Task 6 binds `topology_identity_sha256`; the model-to-topology binding is Stage B's canonical digest (Stage D table) |
| S12 unowned items | Stage D prerequisites table; peptide atoms under `--run-mode cmd` in Task 11 |
| S14 build_gareus_parser | Stage B registers `_add_aux_cv_args` on both parsers; Task 11 edits that helper, so the new flags are registered too |

**Not applied, or applied differently:**
- **`merge_identical_hamiltonians` stays unwired.** The decision allowed this explicitly, and spec 4.3 permits separate origins.
- **Real resume control is a CLI run (Task 14 Step 4), not a pytest unit.** The repo has no in-test MD production fixture. The comparison tool is unit-tested.
- **A test-only failure hook (`GAREUS_TEST_FAIL_AT_PROD_STEP`) drives the exception path.** It is read once, only for auxiliary runs, and is inert unless set.
- **Stage B tests edited by this plan:** `tests/test_aux_cv_kernel_identity.py` (Task 7: import line, `test_v3_aux_is_never_verified_in_stage_b` → `test_v3_aux_without_payload_stays_unpersisted`; Task 12: the two resume-refusal tests), `tests/test_aux_cv_production_wiring.py` (Task 12: refusal-first prefix, population causes; Task 13: gating names and `ast.IfExp`), `tests/test_aux_cv_cli.py` (Task 11: removed flag, resume/extend cases).

**Spec coverage, Stage C:**

| Requirement | Task |
|---|---|
| Observations | 2, 3, 13 |
| Ordered events | 4, 5, 13 |
| Model-aware manifests | 1, 3, 8 |
| Restart binding | 6, 11, 12, 13 |
| Interrupted/restarted vs uninterrupted controls | 5, 14 |
| Strict pooling | 8, 9 |
| Legacy compatibility | pinned in 1, 3, 4, 5, 7, 8, 12, 13 |
| Duplicates | 10 |
| GPU parity | 15 |

---

## Revision record (v3, 2026-10-08): pre-flight rulings → where applied

| Ruling / finding | Applied in |
|---|---|
| **B1** refusal becomes mismatch-only `refuse_resume_of_aux_campaign(out_dir, args)`, first on ANY `--resume` | Task 12 (function, fast-branch first statement, non-fast first statement; two-sided unit test; Stage B tests `test_resume_of_an_aux_campaign_*`, refusal-first prefix pin); Task 14 Step 4.5 refusal runs |
| **B2** fast resume sets `_aux_window_key` + `refuse_aux_population_change(..., cause="checkpoint resume")`; NaN-safe key | Task 12 (fast branch, `_aux_window_order_key`, causes pin updated) |
| **B3** reuse/extend `_refuse_aux_run` (RuntimeError) instead of new loader guards; first in `load_npz`; adaptive branch | Task 9 (`_refuse_aux_run(..., depth=)` + `snapshot_has_aux`, hoisted in `load_npz`, `depth=2` in the adaptive branch); `AuxPoolingRefused` subclasses RuntimeError so the Stage B pin holds |
| **B4** (user) crash re-seal aux-gated on `args._aux_runtime`, legacy seal verbatim, legacy double count → separate PR | Task 5 (gated branch, AST pin of the verbatim legacy seal, commit `feat(cvaux):`); Architecture; Global Constraints; Task 14 evidence paragraph; Task 16 CLAUDE.md; Stage D table; self-review C4/S7 |
| **B5** `segment_eligibility` reads the samples manifest only for aux snapshots | Task 7 (`raw_sample_payload_schema` only on aux snapshots / snapshot-less segments; monkeypatched no-read test) |
| **H1** `verify_kernel_identity_on_resume` args-aware | Task 12 (`exchange_energy_version_for_args(args)`, unit test); Task 14 configurations B/C exercise it |
| **H2** snapshot-less segment of an aux run refused, not `not_applicable` | Task 7 (`aux_unpersisted`, run identified by snapshots / run_manifest / payload); Task 9 (`no_snapshot` class refused with no override, test) |
| **H3** mixed v2/v3_aux segments refused for pooling (+ B1 resume half) | Task 9 (`legacy` class refused with no override, test with the opt-out); Task 12 (non-fast resume guard; fresh non-resume run fails at pooling) |
| **H4** adaptive `load_data` branch guarded (via B3) | Task 9 (`_refuse_aux_run(prod, {}, depth=2)`, test through `prod_dir_of`) |
| New CLI flags in recorded args of non-aux jobs (C6 class) | Global Constraints; Task 11 interface note |
| Legacy snapshot-scan cost accepted | Global Constraints; Task 9 (fixed-depth globs instead of `rglob`, L1) |
| Task 13 (now 14) test hook aux-gated | Task 13 item 10 (env read once under `_aux_io`) |
| Skip-writer test counts call sites (D1) | Task 13 `test_every_gibbs_decision_...` (`ast.Call` count == 3) |
| Snapshot-equality check reachable without a test override (L3) | Task 11 `applied_window_key` compared with `normalize_windows` rows; `_rows_override` removed |
| Reuse the observer's positions read (M4) | Task 13 `make_aux_record_observer`; `_fetch_state` keeps the bare `_aux_z_for_replica` call for the Stage B pin |
| `data_boundary` safe before first flush (M1) | Task 13 item 9 (`load_manifest(...) or {"generation": 0, "n_rows": 0}`) |
| Instance-less aux tables refused at load (M2) | Task 11 (`load_aux_state_table`; `build_runtime_state_definition` has no default instance) |
| Resume state-definition hash covers CV2 defs and boost envelope; Task 13 (now 14) checks it (M3) | Task 6 (`compare_state_definitions` field names in the refusal); Task 13 item 9 note; Task 14 configurations B (CV2) and C (Pep-GaMD) with per-segment `state_definition_sha256` equality |
| Runtime parity non-fatal for sham-only populations (M5) | Task 13 `check_runtime_parity` (early return before the finite check; test) |
| Task 13 (now 14) explicit full flag set | Task 14 Step 4 (flags smoke-parsed at HEAD, model bound to the production topology, CSVs with instances) |
| Preflight T8: do not move the step/replica loop | Task 8 Step 4 (keep block after `source_arrays`) |
| Preflight T6: `aux` survives `publish_generation` | Task 6 round-trip unit test |
| Preflight L4: `compare_runs` interface | Task 14 (Δtorsion implemented, wrapped; ledger proxy stated, no anchor replay claimed) |
| Preflight L5: speculative ordering-test step | Dropped (Task 13 Step 5 note) |
| Preflight L6: self-review claim about `aux_io_runtime` | Stage D table row and Task 11 interface rewritten |
| Preflight §2 T15: Stage C section placement, stale Stage B bullets, run list | Task 16 |
| Duplication (`_is_aux_identity`, `snapshot_has_aux` identity part) | Task 7/9 use Stage B's `is_aux_kernel_record` |
| Task split | Old Task 12 split into Task 12 (resume guards, no MD) and Task 13 (I/O wiring); old Tasks 13/14/15 → 14/15/16 |
| Final review: fresh vs resume build different inputs (topology from `01_solvated_start.pdb` on resume; NVT replica boxes set only on the fresh path) | Task 11 physical sha excludes the default box (test); Task 13 NVT box from all fresh replicas (must agree) or from the checkpoint on resume, every restored box verified (`check_fixed_box`); Task 14 model built from `01_solvated_start.pdb`, configurations in NPT plus NVT configuration D, stop rule for a topology-digest mismatch |
| Final review: legacy `aux_pre_apply` must not import the aux package | Task 13 item 9 (inline `manifest.get("aux")` check; `verify_aux_resume` only in the aux branch, now on the gating list); Global Constraints |
| Final review: refusal runs rewrite `run_manifest.json` | Task 14 refusals on `cp -a` copies; Task 12 residual note |
