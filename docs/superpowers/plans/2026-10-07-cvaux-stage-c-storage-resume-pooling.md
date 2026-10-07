# CVaux Stage C: Storage, Resume and Strict Pooling — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ensure no active auxiliary term can disappear through a writer, loader, early return or state-hash path.
- **Record:** every thermodynamic sample stores model-indexed z (float64) and the complete ordered torsion basis. Every exchange decision is an ordered, replayable event.
- **Resume:** checkpoints bind the auxiliary state table, the models and the Context parameters.
- **Pool:** the strict fixed-state exporter evaluates auxiliary energies offline from the stored features, with whole-row exclusion counted and reported.

**Architecture:**
- **Storage format.** New pure-Python helpers live in `gareus/auxiliary_cv/`: `sample_schema.py`, `ledger.py`, `checkpoint.py`, `offline.py`, `runtime_io.py`. The existing writers (`gareus/store.py`), the Parquet manifest and the window snapshot gain *optional* auxiliary payloads. Every new field is written only when the auxiliary capability is on, so with it off every byte is unchanged.
- **Thermodynamic pooling.** Pooling goes through the strict exporter (`gareus/correctness/export.py`) and the single-run Parquet loader (`gareus/mbar_analysis/loaders.py:load_parquet`). The adaptive union loaders refuse auxiliary states outright: MVP pools only fixed-state pilot phases (spec Sections 8 and 14).
- **Order.** Production call-site wiring (Task 10) is the last task. It depends on Stage B's runtime objects.

**Tech Stack:** Python 3, NumPy, pyarrow, duckdb, OpenMM 8.5 (Reference platform, one test), pytest.

**Spec:** `docs/superpowers/specs/2026-10-07-auxiliary-cv-gibbs-production-spec.md` (main dc30285), Sections 7, 8, 9, 15 (Storage / Offline energies / Resume rows), 16 Stage C and 17. Read Sections 7–9 before starting.

**Depends on:**
- **Stage A plan** `docs/superpowers/plans/2026-10-07-cvaux-stage-a-definitions-evaluators.md`. Its tasks must be merged first.
- **Stage B plan** (sibling, `docs/superpowers/plans/2026-10-07-cvaux-stage-b-*.md`). Its runtime tasks must land before this plan's Task 10. Tasks 1–9 depend only on Stage A.

**Test runner (user policy):** tests in this repo are run by the local free runner `opencode`, never directly. Each "Run:" step below names the pytest arguments. Dispatch them as
`opencode run "In /run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler: run python -m pytest -q <arguments> and report pass/fail/error counts and every failing test id. Read-only: do not edit, commit or fix anything."`
and treat its report as the result. A Bash hook in this harness refuses commands that call the test runner directly.

**Targeted tests only** (user preference): run the files a task names, never the whole suite.

## Global Constraints

- **Off path byte-identical.** With the auxiliary capability off:
  - the Parquet sample/exchange column sets and dtypes do not change;
  - `parquet_manifest.json` gains no key;
  - window snapshots, checkpoint manifests and NPZ exports are unchanged.
  - Every task that touches a writer carries a legacy-equivalence test.
- **No zero-fill.** A missing or nonfinite auxiliary observation is never replaced by 0, by the current state's scalar z, or by a placeholder.
  - `gareus.query._concat_numpy_dicts` back-fills a column missing from one segment with masked placeholders (NaN downstream). Every loader in this plan therefore checks feature presence **per segment** before concatenation and fails closed.
- **Use float64 for auxiliary z and torsions in storage and offline assembly** (spec Section 7). Lower precision is out of scope.
- **Identity is content.**
  - Models are referenced by full 64-hex `model_sha256`, never abbreviated in executable records.
  - The torsion basis has its own `basis_sha256`.
  - Column names index into a recorded ordered list (`aux_z_00`, `tor_000`); the list is the identity, not the column name.
- **Observation key** = `(run_id, segment_id, carrier_id, absolute_step, observation_phase)`.
  - `carrier_id` is the existing `replica` column. `segment_id` comes from the loader. `run_id` is the run directory.
  - `observation_phase` is the new constant column `"pre_exchange"`. Samples are written before exchange (`tests/test_sample_before_exchange_ordering.py`).
  - Never join on step alone or on floating-point time.
- **Exchange ledger.** It extends the existing `ParquetExchangeWriter` dataset (`exchanges/<seg>/`). There is no second ledger.
- **One fixed state table per pooled solve.** Rows lacking features never enter a pool containing active auxiliary states unless the caller explicitly excludes them, and the exclusion is reported.
- **MBAR solves in tests** go through `gareus.adaptive.mbar_solve.solve_rows`, which wraps gareus-analyze's `gareus.mbar_analysis.solvers.solve_mbar`. No hand-rolled solver (user rule).

## Interfaces expected from Stage B (call-site contract)

Stage B builds the runtime. This plan's Task 10 wires the following calls into `gareus/production.py`; Stage B's code must provide the named values.

| Stage B provides (in `run_gareus` scope) | Type | Stage C call that consumes it |
|---|---|---|
| `aux_runtime` (None when the capability is off) | object with `.state_definition: dict` (canonical `atlas-fixed-state-v2`), `.models: Mapping[str, AuxModel]`, `.force_info: AuxForceInfo`, `.sample_schema: AuxSampleSchema` (built by Stage B with this plan's `build_sample_schema`) | `ParquetSampleWriter(..., aux_schema=aux_runtime.sample_schema)`, `ParquetExchangeWriter(..., event_schema=EXCHANGE_EVENT_SCHEMA)`, `WindowSnapshot.snapshot(..., state_definition=aux_runtime.state_definition, ...)` |
| Per sample step, per carrier r: `aux_obs[r]` | `AuxObservation(torsions: ndarray (T,), z: ndarray (M,))`, from this plan's `observe_carrier(xyz_nm, schema, models)` on the carrier's sample-step positions | `parquet_sample_writer.write_sample(..., aux_z=aux_obs[r].z, torsions=aux_obs[r].torsions)` |
| Per exchange decision (accepted, rejected, stay, no-candidates) | Proposal outcome plus a within-step attempt counter | `parquet_exchange_writer.write_event(step=, attempt_seq=, selected_replica=, replica_i=, replica_j=, window_i=, window_j=, kind=, delta_e_kj=, accepted=, log_q_forward=, log_q_reverse=, p_accept=, energy_version=, assignments_after=)` |
| At checkpoint save and after the resume assignment | `observed_params: list[tuple[float, float]]` per replica, from `read_aux_parameters(sim.context, aux_runtime.force_info)` | `save_production_checkpoint(..., aux_block=aux_checkpoint_block(...))`, `verify_aux_resume(manifest, ...)` |

If Stage B evaluates exchange-matrix z some other way than `observe_carrier`, Stage B must prove that its z equals `observe_carrier(...).z` bitwise for the same positions. The stored z must be the z used in the exchange matrix (spec Section 7: "runtime force calculations remain authoritative; offline feature computation must use the same conventions").

## Review Focus

1. **A historical segment without feature columns, pooled with auxiliary states.** It must fail closed and name the segment. The placeholder-NaN path in `_concat_numpy_dicts` must not turn it into silently excluded rows. The Task 8 test pins this.
2. **Several accepted swaps at one step.** Replay must follow `attempt_seq` order. Out-of-order or duplicate `(step, attempt_seq)` raises. The Task 5 test pins this.
3. **Resume after the model file changed on disk.** The coefficients differ, so the sha differs, so resume refuses. Equally, an auxiliary manifest resumed by a run with the capability off (or the reverse) refuses. The Task 6 test pins this.
4. **Consolidation dropping Parquet schema metadata.** `_consolidate` rewrites chunks into `data.parquet`. The auxiliary schema must survive in both the file metadata and the manifest. The Task 3 test pins this.
5. **Exclusion that is numerically rare but concentrated** (all in one origin state or one z range). The report must break it down by origin state, carrier, time block and z range. The Task 7 test pins this.

---

## File Structure

| File | Responsibility |
|---|---|
| Modify `gareus/parquet_manifest.py` | Preserve an optional `payload_schema` object through validate/append/replace; refuse a schema change within a segment |
| Create `gareus/auxiliary_cv/sample_schema.py` | `AuxSampleSchema` (torsion basis and model order), `build_sample_schema`, `observe_carrier`, `AuxObservation` |
| Modify `gareus/store.py` | `ParquetSampleWriter(aux_schema=)` and `write_sample(aux_z=, torsions=)`; `ParquetExchangeWriter(event_schema=)` and `write_event(...)`; `WindowSnapshot.snapshot(state_definition=...)` |
| Create `gareus/auxiliary_cv/ledger.py` | `assignment_sha256`, `replay_assignments`, `EXCHANGE_EVENT_SCHEMA` |
| Create `gareus/auxiliary_cv/checkpoint.py` | `aux_checkpoint_block`, `verify_aux_resume`, `expected_aux_parameters`, `read_aux_parameters` |
| Modify `gareus/production.py` | `save_production_checkpoint(aux_block=)`, `load_production_checkpoint(aux_resume=)` (Task 6); call-site wiring (Task 10) |
| Create `gareus/auxiliary_cv/offline.py` | `segment_aux_schemas`, `aux_z_from_samples`, `exclusion_report`, `merge_identical_hamiltonians`, `refuse_aux_snapshots` |
| Modify `gareus/correctness/export.py` | Auxiliary evaluation inside `build_export_arrays`, `on_incomplete=` policy, feature arrays in the signature |
| Modify `gareus/query.py` | `reconstruct_bias_matrix(aux_z=)` pass-through; `export_analysis_arrays_npz` refusal |
| Modify `gareus/mbar_analysis/loaders.py` | `load_parquet` auxiliary evaluation plus per-segment feature check |
| Modify `gareus/mbar_analysis/loaders_union_parquet.py`, `gareus/adaptive_production.py` | Refuse auxiliary snapshots (MVP scope) |
| Create `gareus/auxiliary_cv/runtime_io.py` | `check_exchange_boundary_alignment` and the production wiring helpers used in Task 10 |
| Tests | `tests/test_aux_manifest_payload_schema.py`, `test_aux_sample_schema.py`, `test_aux_store_writers.py`, `test_aux_window_snapshot.py`, `test_aux_ledger.py`, `test_aux_checkpoint.py`, `test_aux_export.py`, `test_aux_loaders.py`, `test_aux_mbar_duplicates.py`, `test_aux_runtime_io.py` |

---

### Task 1: Parquet manifest carries an optional payload schema

**Files:**
- Modify: `gareus/parquet_manifest.py:113-198` (`validate_manifest`), `:229-258` (`append_file_to_manifest`, `replace_files_in_manifest`)
- Test: `tests/test_aux_manifest_payload_schema.py`

**Interfaces:**
- Produces:
  - `append_file_to_manifest(segment_dir, *, kind, record, next_chunk_index, payload_schema: dict | None = None)`;
  - `replace_files_in_manifest(..., payload_schema: dict | None = None)`;
  - validated manifests contain `"payload_schema"` **only** when it was present.
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


def test_legacy_manifest_has_no_payload_schema_key(tmp_path):
    append_file_to_manifest(tmp_path, kind="samples", record=_chunk(tmp_path), next_chunk_index=2)
    raw = json.loads(manifest_path(tmp_path).read_text())
    assert "payload_schema" not in raw
    assert set(raw) == {"schema", "kind", "generation", "n_rows", "next_chunk_index", "files"}


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

Run: `pytest tests/test_aux_manifest_payload_schema.py`
Expected: FAIL. `test_payload_schema_survives_append_and_replace` raises `TypeError: ... unexpected keyword argument 'payload_schema'`. `test_legacy_manifest_has_no_payload_schema_key` already passes.

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

Run: `pytest tests/test_aux_manifest_payload_schema.py tests/test_parquet_manifest_transactions.py tests/test_store.py`
Expected: PASS. The last two files are the legacy regression.

- [ ] **Step 5: Commit**

```bash
git add gareus/parquet_manifest.py tests/test_aux_manifest_payload_schema.py
git commit -m "feat(cvaux): Parquet manifests carry an optional immutable payload schema"
```

---

### Task 2: Auxiliary sample schema and carrier observation

**Files:**
- Create: `gareus/auxiliary_cv/sample_schema.py`
- Modify: `gareus/auxiliary_cv/__init__.py`
- Test: `tests/test_aux_sample_schema.py`

**Interfaces:**
- Consumes: Stage A `AuxModel`, `openmm_dihedrals`, `unique_torsions`, `z_from_dihedrals`; `gareus.mbar_analysis.thermo_frames.backbone_torsion_quads`; `tests/aux_cv_fixture.py` (`dipeptide`, `model_payload`).
- Produces:
  - `AUX_SAMPLES_SCHEMA = "atlas-aux-samples-v1"`;
  - `@dataclass(frozen=True) class AuxSampleSchema` with fields `torsion_quads: tuple[tuple[int,int,int,int], ...]`, `torsion_labels: tuple[str, ...]`, `model_shas: tuple[str, ...]`, `basis_sha256: str`;
  - methods `.to_payload() -> dict`, `AuxSampleSchema.from_payload(d) -> AuxSampleSchema`, `.torsion_columns -> tuple[str, ...]` (`tor_000`…), `.z_columns -> tuple[str, ...]` (`aux_z_00`…), `.model_basis_index(model) -> np.ndarray`;
  - `build_sample_schema(topology, peptide_atoms, models: Sequence[AuxModel]) -> AuxSampleSchema`;
  - `@dataclass(frozen=True) class AuxObservation(torsions: np.ndarray, z: np.ndarray)`;
  - `observe_carrier(xyz_nm, schema, models: Mapping[str, AuxModel]) -> AuxObservation`.
- Rules:
  - The torsion basis is **every** backbone phi/psi of the peptide in `backbone_torsion_quads` order, so later models stay evaluable (spec Section 7).
  - Each model's torsions must be a subset of the basis, else `IntegrityError`.
  - The models in the schema are ordered by first appearance.
  - At most 1 model in MVP (spec Section 1.2), mirroring Stage A's schema rule.
  - Torsions are stored in the OpenMM convention (θ), float64.
  - A degenerate torsion gives NaN in `torsions` and therefore in `z`; storage keeps the NaN, and the exporter decides.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_aux_sample_schema.py
import numpy as np
import pytest

from aux_cv_fixture import dipeptide, model_payload
from gareus.auxiliary_cv.evaluate import z_from_positions
from gareus.auxiliary_cv.model import AuxModel
from gareus.auxiliary_cv.sample_schema import (AUX_SAMPLES_SCHEMA, AuxSampleSchema, build_sample_schema,
                                               observe_carrier)
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


def test_basis_sha_changes_with_basis():
    d = dipeptide()
    m = _model(d, 0)
    s = build_sample_schema(d["topology"], _peptide_atoms(d), [m])
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_aux_sample_schema.py`
Expected: FAIL `ModuleNotFoundError: No module named 'gareus.auxiliary_cv.sample_schema'`

- [ ] **Step 3: Implement**

```python
# gareus/auxiliary_cv/sample_schema.py
"""What every auxiliary-capable sample row stores: the full ordered backbone torsion basis
(OpenMM theta, float64) and z of every phase model (float64), schema ``atlas-aux-samples-v1``.

The basis is ALL backbone phi/psi, not just the active model's torsions, so a model fitted later
can still be evaluated on these rows (spec Section 7). Column names (tor_000, aux_z_00) are
positions in the recorded lists; the lists and basis_sha256 are the identity.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from ..correctness._io import IntegrityError, digest, json_bytes
from .evaluate import z_from_dihedrals
from .features import openmm_dihedrals, unique_torsions
from .model import AuxModel

AUX_SAMPLES_SCHEMA = "atlas-aux-samples-v1"
TORSION_CONVENTION = "openmm_theta_radians"


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
        """Column index in the basis of every unique torsion of ``model`` (Stage A order)."""
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
        schema.model_basis_index(m)      # raises if a model torsion is outside the basis
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
```

Add to `gareus/auxiliary_cv/__init__.py`: `from .sample_schema import AUX_SAMPLES_SCHEMA, AuxObservation, AuxSampleSchema, build_sample_schema, observe_carrier`, and extend `__all__`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_aux_sample_schema.py tests/test_aux_cv_evaluate.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add gareus/auxiliary_cv/sample_schema.py gareus/auxiliary_cv/__init__.py tests/test_aux_sample_schema.py
git commit -m "feat(cvaux): auxiliary sample schema (full torsion basis + model z) and carrier observation"
```

---

### Task 3: Sample writer records torsions and model z; legacy bytes unchanged

**Files:**
- Modify: `gareus/store.py:29-148` (`ParquetSampleWriter`)
- Test: `tests/test_aux_store_writers.py` (sample part)

**Interfaces:**
- Consumes: Task 1 `append_file_to_manifest(..., payload_schema=)`, `replace_files_in_manifest(..., payload_schema=)`; Task 2 `AuxSampleSchema`.
- Produces:
  - `ParquetSampleWriter(out_dir, flush_rows=5000, aux_schema: AuxSampleSchema | None = None)`;
  - `write_sample(..., gamd_lambda=0.0, aux_z: Sequence[float] | None = None, torsions: Sequence[float] | None = None)`.
- With `aux_schema`:
  - Each row gets float64 `tor_###` and `aux_z_##` columns and the string column `observation_phase="pre_exchange"`.
  - The table schema metadata key `b"atlas_aux_samples"` holds the payload JSON.
  - The manifest gets `payload_schema`.
  - Missing or wrong-length `aux_z`/`torsions` raises `ValueError` before buffering.
  - Nonfinite values are stored as-is (NaN), never replaced.
- Without `aux_schema`: passing `aux_z`/`torsions` raises `ValueError`, and the output is byte-identical to today.
- Constructor checks:
  - Opening an aux writer on a segment whose manifest records a different `payload_schema`, or a populated legacy segment, raises `ParquetManifestError` at construction. This catches the problem before the first flush.
  - Opening a legacy writer on a segment that has a `payload_schema` also raises.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_aux_store_writers.py
import json

import numpy as np
import pytest

from gareus.auxiliary_cv.sample_schema import AuxSampleSchema, _basis_sha
from gareus.parquet_manifest import ParquetManifestError, load_manifest

LEGACY_COLUMNS = ["step", "replica", "window_id", "cv1", "cv2", "potential", "gamd_boost_total",
                  "gamd_boost_dihedral", "gamd_boost_nonbonded", "v_pep_kj_mol", "v_dih_kj_mol", "gamd_lambda"]
QUADS = ((0, 1, 2, 3), (1, 2, 3, 4), (2, 3, 4, 5))
LABELS = ("phi-A1", "psi-A1", "phi-B2")
SCHEMA = AuxSampleSchema(QUADS, LABELS, ("e" * 64,), _basis_sha(QUADS, LABELS))


def _row(step, replica=0):
    return dict(step=step, replica=replica, window_id=replica, cv1=0.1, cv2=None, potential=-1.0,
                boost_total=None, boost_dihedral=None, boost_nonbonded=None)


def test_legacy_sample_schema_unchanged(tmp_path):
    import pyarrow.parquet as pq
    from gareus.store import ParquetSampleWriter
    w = ParquetSampleWriter(tmp_path)
    w.write_sample(**_row(10))
    w.close()
    tbl = pq.read_table(tmp_path / "data.parquet")
    assert tbl.column_names == LEGACY_COLUMNS
    assert tbl.schema.metadata is None or b"atlas_aux_samples" not in tbl.schema.metadata
    assert "payload_schema" not in load_manifest(tmp_path)


def test_legacy_writer_refuses_aux_values(tmp_path):
    from gareus.store import ParquetSampleWriter
    w = ParquetSampleWriter(tmp_path)
    with pytest.raises(ValueError, match="aux"):
        w.write_sample(**_row(10), aux_z=[1.0], torsions=[0.0, 0.0, 0.0])


def test_aux_columns_float64_metadata_and_manifest_survive_consolidation(tmp_path):
    import pyarrow.parquet as pq
    from gareus.store import ParquetSampleWriter
    w = ParquetSampleWriter(tmp_path, flush_rows=1, aux_schema=SCHEMA)
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
    assert AuxSampleSchema.from_payload(meta) == SCHEMA
    assert AuxSampleSchema.from_payload(load_manifest(tmp_path)["payload_schema"]) == SCHEMA


@pytest.mark.parametrize("kw", [dict(aux_z=None, torsions=[0, 0, 0]), dict(aux_z=[1.0], torsions=None),
                                dict(aux_z=[1.0, 2.0], torsions=[0, 0, 0]), dict(aux_z=[1.0], torsions=[0, 0])])
def test_aux_writer_requires_complete_rows(tmp_path, kw):
    from gareus.store import ParquetSampleWriter
    w = ParquetSampleWriter(tmp_path, aux_schema=SCHEMA)
    with pytest.raises(ValueError, match="aux"):
        w.write_sample(**_row(10), **kw)


def test_aux_writer_refuses_populated_legacy_segment(tmp_path):
    from gareus.store import ParquetSampleWriter
    w = ParquetSampleWriter(tmp_path)
    w.write_sample(**_row(10))
    w.flush()
    with pytest.raises(ParquetManifestError, match="payload schema"):
        ParquetSampleWriter(tmp_path, aux_schema=SCHEMA)


def test_legacy_writer_refuses_aux_segment(tmp_path):
    from gareus.store import ParquetSampleWriter
    w = ParquetSampleWriter(tmp_path, aux_schema=SCHEMA)
    w.write_sample(**_row(10), aux_z=[1.0], torsions=[0.0, 0.0, 0.0])
    w.flush()
    with pytest.raises(ParquetManifestError, match="payload schema"):
        ParquetSampleWriter(tmp_path)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_aux_store_writers.py`
Expected: FAIL `TypeError: ... unexpected keyword argument 'aux_schema'`. The two legacy tests fail on `aux_z` too.

- [ ] **Step 3: Implement in `ParquetSampleWriter`**

Change the constructor signature and body (keep the existing lines):

```python
    def __init__(self, out_dir: Path, flush_rows: int = 5000, aux_schema=None) -> None:
        self._out_dir = Path(out_dir)
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._flush_rows = max(1, flush_rows)
        self._buf: Dict[str, list] = defaultdict(list)
        self._manifest = load_manifest(self._out_dir, expected_kind="samples")
        if self._manifest is None and list(self._out_dir.glob("*.parquet")):
            raise ParquetManifestError(f"cannot append to legacy Parquet segment without manifest: {self._out_dir}")
        self._chunk_idx = (self._manifest["next_chunk_index"] - 1) if self._manifest else 0
        self._aux_schema = aux_schema
        self._aux_payload = aux_schema.to_payload() if aux_schema is not None else None
        recorded = (self._manifest or {}).get("payload_schema")
        populated = bool(self._manifest and self._manifest["files"])
        if populated and recorded != self._aux_payload:
            raise ParquetManifestError(
                f"payload schema of {self._out_dir} is {recorded!r}; this writer would write {self._aux_payload!r}")
```

In `write_sample`, add the two keyword parameters `aux_z=None, torsions=None` and insert at the top of the body:

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

In `flush`, after building `tbl = pa.table({...})`:

```python
        if self._aux_schema is not None:
            extra = {"observation_phase": pa.array(b["observation_phase"], type=pa.string())}
            for name in self._aux_schema.torsion_columns + self._aux_schema.z_columns:
                extra[name] = pa.array(b[name], type=pa.float64())
            for name, column in extra.items():
                tbl = tbl.append_column(name, column)
            tbl = tbl.replace_schema_metadata({b"atlas_aux_samples": json.dumps(self._aux_payload, sort_keys=True).encode()})
```

Pass `payload_schema=self._aux_payload` to the `append_file_to_manifest(...)` call in `flush`. In `_consolidate`, after `tbl = tbl.sort_by(...)`:

```python
        if self._aux_payload is not None:
            tbl = tbl.replace_schema_metadata({b"atlas_aux_samples": json.dumps(self._aux_payload, sort_keys=True).encode()})
```

Then pass `payload_schema=self._aux_payload` to `replace_files_in_manifest(...)`. Clearing `b.values()` already clears the new lists.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_aux_store_writers.py tests/test_store.py tests/test_parquet_manifest_transactions.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add gareus/store.py tests/test_aux_store_writers.py
git commit -m "feat(cvaux): sample writer records the torsion basis and model z (float64), off path unchanged"
```

---

### Task 4: Ordered exchange event ledger

**Files:**
- Create: `gareus/auxiliary_cv/ledger.py` (schema constant and assignment checksum only; replay comes in Task 5)
- Modify: `gareus/store.py:150-270` (`ParquetExchangeWriter`)
- Test: `tests/test_aux_store_writers.py` (exchange part)

**Interfaces:**
- Produces:
  - `EXCHANGE_EVENT_SCHEMA = "atlas-exchange-events-v1"`;
  - `EVENT_KINDS = ("swap", "stay", "no_candidates")`;
  - `assignment_sha256(assignments: Sequence[int]) -> str` (sha256 of `json_bytes([int,...])`);
  - `ParquetExchangeWriter(out_dir, flush_rows=1000, event_schema: str | None = None)`;
  - `write_event(*, step, attempt_seq, selected_replica, replica_i, replica_j, window_i, window_j, kind, delta_e_kj, accepted, log_q_forward, log_q_reverse, p_accept, energy_version, assignments_after)`.
- Event-mode columns, appended after the 7 legacy ones: `attempt_seq` (uint32), `selected_replica` (int32, -1 = none), `kind` (string), `delta_e_kj` (float64), `log_q_forward`, `log_q_reverse`, `p_accept` (float64, NaN when not applicable), `energy_version` (string), `assignment_sha256_after` (string).
- In event mode:
  - the legacy `delta_e` float32 column is still filled (`float(delta_e_kj)`), so existing readers keep working;
  - `write_exchange(...)` raises `ValueError` ("event-mode writer requires write_event");
  - the manifest records `payload_schema={"schema": EXCHANGE_EVENT_SCHEMA}`.
- Legacy mode is unchanged, and calling `write_event` raises.
- Field rules for each event kind:

  | Field | `stay` / `no_candidates` | `swap` |
  |---|---|---|
  | `replica_i` / `replica_j` | both = selected replica | the pair |
  | `window_i` / `window_j` | both = current window | the pair |
  | `accepted` | False | the decision |
  | `assignments_after` | the unchanged assignments | the assignments after this decision |

- [ ] **Step 1: Write the failing tests** (append to `tests/test_aux_store_writers.py`)

```python
from gareus.auxiliary_cv.ledger import EXCHANGE_EVENT_SCHEMA, assignment_sha256

LEGACY_EXCHANGE_COLUMNS = ["step", "replica_i", "replica_j", "window_i", "window_j", "delta_e", "accepted"]


def _event(step, seq, **kw):
    base = dict(step=step, attempt_seq=seq, selected_replica=0, replica_i=0, replica_j=1, window_i=0,
                window_j=1, kind="swap", delta_e_kj=-0.25, accepted=True, log_q_forward=np.log(0.4),
                log_q_reverse=np.log(0.3), p_accept=0.9, energy_version="state_bias_matrix_v3_aux",
                assignments_after=[1, 0, 2])
    base.update(kw)
    return base


def test_legacy_exchange_schema_unchanged(tmp_path):
    import pyarrow.parquet as pq
    from gareus.store import ParquetExchangeWriter
    w = ParquetExchangeWriter(tmp_path)
    w.write_exchange(step=5, replica_i=0, replica_j=1, window_i=0, window_j=1, delta_e=0.5, accepted=False)
    w.close()
    assert pq.read_table(tmp_path / "data.parquet").column_names == LEGACY_EXCHANGE_COLUMNS
    assert "payload_schema" not in load_manifest(tmp_path)
    with pytest.raises(ValueError, match="event"):
        w.write_event(**_event(6, 0))


def test_event_rows_round_trip(tmp_path):
    import pyarrow.parquet as pq
    from gareus.store import ParquetExchangeWriter
    w = ParquetExchangeWriter(tmp_path, event_schema=EXCHANGE_EVENT_SCHEMA)
    w.write_event(**_event(100, 0))
    w.write_event(**_event(100, 1, kind="stay", replica_j=0, window_j=0, accepted=False,
                           log_q_reverse=float("nan"), p_accept=float("nan"), assignments_after=[1, 0, 2]))
    w.close()
    t = pq.read_table(tmp_path / "data.parquet")
    assert t.column_names == LEGACY_EXCHANGE_COLUMNS + [
        "attempt_seq", "selected_replica", "kind", "delta_e_kj", "log_q_forward", "log_q_reverse",
        "p_accept", "energy_version", "assignment_sha256_after"]
    assert t.column("attempt_seq").to_pylist() == [0, 1]
    assert t.column("assignment_sha256_after").to_pylist()[0] == assignment_sha256([1, 0, 2])
    assert t.column("delta_e_kj").to_pylist()[0] == -0.25
    assert load_manifest(tmp_path)["payload_schema"] == {"schema": EXCHANGE_EVENT_SCHEMA}
    with pytest.raises(ValueError, match="write_event"):
        w.write_exchange(step=5, replica_i=0, replica_j=1, window_i=0, window_j=1, delta_e=0.5, accepted=False)


@pytest.mark.parametrize("bad", [dict(kind="teleport"), dict(attempt_seq=-1),
                                 dict(kind="stay", accepted=True), dict(energy_version="")])
def test_event_validation(tmp_path, bad):
    from gareus.store import ParquetExchangeWriter
    w = ParquetExchangeWriter(tmp_path, event_schema=EXCHANGE_EVENT_SCHEMA)
    with pytest.raises(ValueError):
        w.write_event(**_event(1, 0, **bad))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_aux_store_writers.py -k "exchange or event"`
Expected: FAIL `ModuleNotFoundError: No module named 'gareus.auxiliary_cv.ledger'`

- [ ] **Step 3: Implement `ledger.py` (constants and checksum)**

```python
# gareus/auxiliary_cv/ledger.py
"""Ordered exchange-event ledger: schema, assignment checksums and replay (spec Section 7)."""
from __future__ import annotations

from typing import Sequence

from ..correctness._io import digest, json_bytes

EXCHANGE_EVENT_SCHEMA = "atlas-exchange-events-v1"
EVENT_KINDS = ("swap", "stay", "no_candidates")


def assignment_sha256(assignments: Sequence[int]) -> str:
    """Checksum of the replica -> window assignment list (index = replica)."""
    return digest(json_bytes([int(x) for x in assignments]))
```

- [ ] **Step 4: Implement the writer's event mode**

In `ParquetExchangeWriter.__init__`, add the parameter `event_schema=None` and after the existing body:

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

Pass `payload_schema=({"schema": self._event_schema} if self._event_schema else None)` to `append_file_to_manifest` and to `replace_files_in_manifest`. In `_consolidate`, sort event-mode tables by `[("step", "ascending"), ("attempt_seq", "ascending")]` so within-step order survives compaction:

```python
        keys = [("step", "ascending")]
        if self._event_schema is not None:
            keys.append(("attempt_seq", "ascending"))
        tbl = tbl.sort_by(keys)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_aux_store_writers.py tests/test_store.py tests/test_parquet_manifest_transactions.py`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add gareus/auxiliary_cv/ledger.py gareus/store.py tests/test_aux_store_writers.py
git commit -m "feat(cvaux): ordered exchange-event ledger in the existing exchange dataset"
```

---

### Task 5: Ledger replay, checkpoint consistency and crash truncation

**Files:**
- Modify: `gareus/auxiliary_cv/ledger.py`
- Test: `tests/test_aux_ledger.py`

**Interfaces:**
- Consumes: Task 4 columns; `gareus.query.load_exchanges`; `gareus.store.SegmentRegistry`.
- Produces: `replay_assignments(events: Mapping[str, np.ndarray], start_assignments: Sequence[int], *, after_step: int, up_to_step: int) -> list[int]`. `events` is the dict returned by `load_exchanges`. Its rules:
  - It uses rows with `after_step < step <= up_to_step`, ordered by (step, attempt_seq). A duplicate (step, attempt_seq) raises `IntegrityError`.
  - An accepted `swap` exchanges the windows of `replica_i` and `replica_j`, after checking that their current windows are `window_i` and `window_j`.
  - After every row, `assignment_sha256(current) == assignment_sha256_after`, else `IntegrityError` naming the step and seq.
  - `stay` / `no_candidates` / rejected rows must leave the checksum unchanged.
  - Legacy (non-event) data raises `IntegrityError("no ordered event columns")`.
- Crash semantics (existing segment machinery, `gareus/store.py:315-340`):
  - An interrupted segment is read only up to its sealed `end_step`, the last checkpoint.
  - Events after a crash but beyond the checkpoint are dropped by `load_exchanges`.
  - Replay from checkpoint A to checkpoint B must reproduce B's manifest assignments.

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


def _stay(w, step, seq, r, win, after):
    w.write_event(step=step, attempt_seq=seq, selected_replica=r, replica_i=r, replica_j=r, window_i=win,
                  window_j=win, kind="stay", delta_e_kj=0.0, accepted=False, log_q_forward=0.0,
                  log_q_reverse=float("nan"), p_accept=float("nan"), energy_version="v3", assignments_after=after)


def _run(tmp_path):
    from gareus.store import SegmentRegistry
    reg = SegmentRegistry(tmp_path)
    seg = reg.open_segment("run", None, 1)
    w = _writer(tmp_path, seg)
    # start [0,1,2]; two accepted swaps at the SAME step, in order, then a stay and a rejection
    _swap(w, 100, 0, 0, 1, 0, 1, [1, 0, 2])
    _swap(w, 100, 1, 1, 2, 0, 2, [1, 2, 0])
    _stay(w, 200, 0, 0, 1, [1, 2, 0])
    _swap(w, 200, 1, 0, 2, 1, 0, [1, 2, 0], accepted=False)
    # checkpoint at 200 (assignments [1,2,0]); crash after an accepted swap at 300 (never checkpointed)
    _swap(w, 300, 0, 0, 1, 1, 2, [2, 1, 0])
    w.flush()
    reg.seal_segment(seg, absolute_end_step=200, status="interrupted")
    return seg


def test_replay_same_step_order_reproduces_checkpoint(tmp_path):
    from gareus.query import load_exchanges
    _run(tmp_path)
    ev = load_exchanges(tmp_path)
    assert replay_assignments(ev, [0, 1, 2], after_step=0, up_to_step=200) == [1, 2, 0]


def test_crash_after_accepted_swap_beyond_checkpoint_is_truncated(tmp_path):
    from gareus.query import load_exchanges
    _run(tmp_path)
    ev = load_exchanges(tmp_path)
    assert int(np.max(ev["step"])) == 200          # the step-300 swap is a phantom past the checkpoint
    assert replay_assignments(ev, [0, 1, 2], after_step=0, up_to_step=10**9) == [1, 2, 0]


def test_crash_during_flush_leaves_orphan_tmp_ignored(tmp_path):
    from gareus.query import load_exchanges
    seg = _run(tmp_path)
    (tmp_path / "exchanges" / seg / "chunk_000099.parquet.tmp.4242").write_bytes(b"partial")
    ev = load_exchanges(tmp_path)
    assert replay_assignments(ev, [0, 1, 2], after_step=0, up_to_step=200) == [1, 2, 0]


def test_checksum_mismatch_raises(tmp_path):
    from gareus.query import load_exchanges
    _run(tmp_path)
    ev = load_exchanges(tmp_path)
    ev["assignment_sha256_after"] = np.asarray(ev["assignment_sha256_after"], dtype=object).copy()
    ev["assignment_sha256_after"][1] = assignment_sha256([9, 9, 9])
    with pytest.raises(IntegrityError, match="step 100 seq 1"):
        replay_assignments(ev, [0, 1, 2], after_step=0, up_to_step=200)


def test_window_mismatch_raises(tmp_path):
    from gareus.query import load_exchanges
    _run(tmp_path)
    with pytest.raises(IntegrityError, match="window"):
        replay_assignments(load_exchanges(tmp_path), [2, 1, 0], after_step=0, up_to_step=200)


def test_duplicate_step_seq_raises(tmp_path):
    from gareus.query import load_exchanges
    _run(tmp_path)
    ev = load_exchanges(tmp_path)
    ev = {k: np.concatenate([np.asarray(v), np.asarray(v)[:1]]) for k, v in ev.items()}
    with pytest.raises(IntegrityError, match="duplicate"):
        replay_assignments(ev, [0, 1, 2], after_step=0, up_to_step=200)


def test_legacy_exchanges_cannot_be_replayed(tmp_path):
    from gareus.query import load_exchanges
    from gareus.store import ParquetExchangeWriter
    w = ParquetExchangeWriter(tmp_path / "exchanges" / "seg_001")
    w.write_exchange(step=5, replica_i=0, replica_j=1, window_i=0, window_j=1, delta_e=0.5, accepted=True)
    w.close()
    with pytest.raises(IntegrityError, match="ordered event"):
        replay_assignments(load_exchanges(tmp_path), [0, 1], after_step=0, up_to_step=10)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_aux_ledger.py`
Expected: FAIL `ImportError: cannot import name 'replay_assignments'`

- [ ] **Step 3: Implement**

Append to `gareus/auxiliary_cv/ledger.py`:

```python
from typing import Mapping

import numpy as np

from ..correctness._io import IntegrityError


def _col(events, name):
    value = events[name]
    return np.ma.filled(value, None) if np.ma.isMaskedArray(value) else np.asarray(value)


def replay_assignments(events: Mapping[str, np.ndarray], start_assignments: Sequence[int], *,
                       after_step: int, up_to_step: int) -> list[int]:
    if "attempt_seq" not in events or "assignment_sha256_after" not in events:
        raise IntegrityError("exchange data has no ordered event columns; replay needs an event-mode ledger")
    step = _col(events, "step").astype(np.int64)
    seq = _col(events, "attempt_seq").astype(np.int64)
    keep = (step > int(after_step)) & (step <= int(up_to_step))
    idx = np.flatnonzero(keep)
    idx = idx[np.lexsort((seq[idx], step[idx]))]
    pairs = list(zip(step[idx].tolist(), seq[idx].tolist()))
    if len(set(pairs)) != len(pairs):
        raise IntegrityError("duplicate (step, attempt_seq) in the exchange ledger")
    kind = _col(events, "kind"); accepted = _col(events, "accepted").astype(bool)
    ri = _col(events, "replica_i").astype(np.int64); rj = _col(events, "replica_j").astype(np.int64)
    wi = _col(events, "window_i").astype(np.int64); wj = _col(events, "window_j").astype(np.int64)
    sha = _col(events, "assignment_sha256_after")
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

Two notes for the implementer:
- `load_exchanges` returns string columns through duckdb `fetchnumpy` as object arrays or masked arrays. `_col` handles both.
- If duckdb returns `attempt_seq` as a masked array for a legacy segment mixed into the same run, the test `test_legacy_exchanges_cannot_be_replayed` covers only the no-column case. A mixed legacy+event run is refused because the masked values become `None`, and `astype(np.int64)` raises a `TypeError`. Wrap the two `astype` calls so a `TypeError` becomes `IntegrityError("legacy exchange rows mixed into an event ledger")`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_aux_ledger.py tests/test_aux_store_writers.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add gareus/auxiliary_cv/ledger.py tests/test_aux_ledger.py
git commit -m "feat(cvaux): exchange-ledger replay with checksums, same-step ordering and crash truncation"
```

---

### Task 6: Checkpoint binding and resume verification

**Files:**
- Create: `gareus/auxiliary_cv/checkpoint.py`
- Modify: `gareus/production.py:5216-5323` (`save_production_checkpoint`: keyword `aux_block=None`, written as `manifest["aux"]`)
- Modify: `gareus/production.py:5483-5600` (`load_production_checkpoint`: keyword `aux_resume=None`, a callable `aux_resume(manifest, assignments) -> None`, called after the `_apply_assignment` loop and before the RNG restore)
- Test: `tests/test_aux_checkpoint.py`

**Interfaces:**
- Consumes:
  - Stage A `AuxForceInfo` (`name`, `force_group`, `model_sha256`, `global_k`, `global_c`), `build_aux_force`, `set_aux_parameters`, `KJ_PER_KCAL`;
  - `state_definition_hash` and v2 rows (`aux_model_sha256`, `aux_center`, `aux_k` in kcal/mol per z²).
- Produces:
  - `AUX_CHECKPOINT_SCHEMA = "atlas-aux-checkpoint-v1"`;
  - `expected_aux_parameters(state_definition, window_id) -> tuple[float, float]` returns (k in kJ/mol per z², centre), and (0.0, 0.0) when inactive;
  - `read_aux_parameters(context, info) -> tuple[float, float]`;
  - `aux_checkpoint_block(*, state_definition, force_info, assignments, observed_params) -> dict`;
  - `verify_aux_resume(manifest, *, aux_enabled, state_definition, force_info, assignments, observed_params) -> None`.
- `verify_aux_resume` raises `IntegrityError` naming the field in each of these cases:
  - an aux manifest with the capability off, or the reverse;
  - a differing `state_definition_sha256` (changed coefficients change the embedded model, so the hash changes);
  - a differing `force_info` (name, group, model, globals);
  - a replica count that differs from the manifest (an incomplete carrier set);
  - observed Context parameters (k, c) that differ from `expected_aux_parameters` for the resumed assignment (`math.isclose`, rel 1e-12, abs 0);
  - an assignment checksum that differs from the manifest's `assignment_sha256`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_aux_checkpoint.py
import copy

import numpy as np
import pytest

from aux_cv_fixture import dipeptide, model_payload
from gareus.auxiliary_cv.checkpoint import (aux_checkpoint_block, expected_aux_parameters,
                                            read_aux_parameters, verify_aux_resume)
from gareus.auxiliary_cv.force import AuxForceInfo, build_aux_force, set_aux_parameters
from gareus.auxiliary_cv.model import AuxModel
from gareus.correctness._io import IntegrityError
from gareus.correctness.state_identity import make_state_definition

MODEL = AuxModel.from_mapping(model_payload([(0, 1, 2, 3)], [1.0, 0.5], offset=0.1))
INFO = AuxForceInfo("ATLaSAuxCVUmbrella", 7, MODEL.model_sha256, "aux_k", "aux_c", ("aux_cos_neg", "aux_sin_neg"))


def _defn(model=MODEL, aux_k=1.2):
    rows = [{"window_id": i, "center1": 0.2 * i, "k1": 10.0, "center2": 0.0, "k2": 0.0, "gamd_lambda": 0.0,
             "aux_k": 0.0} for i in range(3)]
    rows[2].update(aux_model_sha256=model.model_sha256, aux_center=1.5, aux_k=aux_k)
    return make_state_definition(rows, physical_system_sha256="a" * 64, ensemble="NVT", temperature_k=300.0,
                                 fixed_box_vectors_nm=[[3, 0, 0], [0, 3, 0], [0, 0, 3]],
                                 cv1={"kind": "contacts", "units": "dimensionless", "definition": {"r0": 4.5}},
                                 cv2=None, aux_models={model.model_sha256: model.to_mapping()})


def _observed(defn, assignments):
    return [expected_aux_parameters(defn, w) for w in assignments]


def test_expected_parameters_units():
    d = _defn()
    assert expected_aux_parameters(d, 0) == (0.0, 0.0)
    assert expected_aux_parameters(d, 2) == pytest.approx((1.2 * 4.184, 1.5), rel=1e-15)


def test_block_round_trip_verifies():
    d, a = _defn(), [2, 0, 1]
    block = aux_checkpoint_block(state_definition=d, force_info=INFO, assignments=a, observed_params=_observed(d, a))
    verify_aux_resume({"aux": block}, aux_enabled=True, state_definition=d, force_info=INFO,
                      assignments=a, observed_params=_observed(d, a))


@pytest.mark.parametrize("case, match", [
    ("changed_model", "state_definition_sha256"),
    ("force_group", "force_info"),
    ("carrier_count", "replica"),
    ("params", "Context parameters"),
    ("assignments", "assignment"),
    ("aux_off", "capability"),
])
def test_resume_refusals(case, match):
    d, a = _defn(), [2, 0, 1]
    manifest = {"aux": aux_checkpoint_block(state_definition=d, force_info=INFO, assignments=a,
                                            observed_params=_observed(d, a))}
    kw = dict(aux_enabled=True, state_definition=d, force_info=INFO, assignments=a, observed_params=_observed(d, a))
    if case == "changed_model":
        other = AuxModel.from_mapping(model_payload([(0, 1, 2, 3)], [1.0, 0.5000001], offset=0.1))
        kw["state_definition"] = _defn(other)
    elif case == "force_group":
        kw["force_info"] = AuxForceInfo(INFO.name, 8, INFO.model_sha256, INFO.global_k, INFO.global_c, INFO.sub_cv_names)
    elif case == "carrier_count":
        kw["assignments"], kw["observed_params"] = a[:2], _observed(d, a[:2])
    elif case == "params":
        kw["observed_params"] = [(0.0, 0.0)] * 3          # aux force left inactive on the aux carrier
    elif case == "assignments":
        kw["assignments"] = [0, 2, 1]
        kw["observed_params"] = _observed(d, [0, 2, 1])
    elif case == "aux_off":
        kw["aux_enabled"] = False
    with pytest.raises(IntegrityError, match=match):
        verify_aux_resume(manifest, **kw)


def test_aux_enabled_against_legacy_manifest_refuses():
    d, a = _defn(), [2, 0, 1]
    with pytest.raises(IntegrityError, match="capability"):
        verify_aux_resume({}, aux_enabled=True, state_definition=d, force_info=INFO, assignments=a,
                          observed_params=_observed(d, a))


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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_aux_checkpoint.py`
Expected: FAIL `ModuleNotFoundError: No module named 'gareus.auxiliary_cv.checkpoint'`

- [ ] **Step 3: Implement `checkpoint.py`**

```python
# gareus/auxiliary_cv/checkpoint.py
"""Bind auxiliary state to production checkpoints and verify it on resume (spec Section 9).

A resume never guesses a state from CV centres: it checks the frozen state table hash, the
force identity, the carrier count, the assignment checksum and every Context's observed
auxiliary parameters against the assignment-derived expectation.
"""
from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any, Mapping, Sequence

from ..correctness._io import IntegrityError, json_loads, json_bytes
from ..correctness.bias import KJ_PER_KCAL
from ..correctness.state_identity import canonical_state_definition, state_definition_hash
from .ledger import assignment_sha256

AUX_CHECKPOINT_SCHEMA = "atlas-aux-checkpoint-v1"


def expected_aux_parameters(state_definition: Mapping[str, Any], window_id: int) -> tuple[float, float]:
    rows = [r for r in state_definition["windows"] if r["window_id"] == int(window_id)]
    if len(rows) != 1:
        raise IntegrityError(f"no window {window_id} in the auxiliary state table")
    k = float(rows[0].get("aux_k", 0.0))
    if k == 0.0:
        return 0.0, 0.0
    return k * KJ_PER_KCAL, float(rows[0]["aux_center"])


def read_aux_parameters(context, info) -> tuple[float, float]:
    return float(context.getParameter(info.global_k)), float(context.getParameter(info.global_c))


def _force_record(info) -> dict[str, Any]:
    record = asdict(info)
    record["sub_cv_names"] = list(record["sub_cv_names"])
    return record


def aux_checkpoint_block(*, state_definition, force_info, assignments: Sequence[int],
                         observed_params: Sequence[tuple[float, float]]) -> dict[str, Any]:
    state = canonical_state_definition(state_definition)
    return {
        "schema": AUX_CHECKPOINT_SCHEMA,
        "state_definition": state,
        "state_definition_sha256": state_definition_hash(state),
        "force_info": _force_record(force_info),
        "n_replicas": len(assignments),
        "assignment_sha256": assignment_sha256(assignments),
        "observed_params": [[float(k), float(c)] for k, c in observed_params],
    }


def verify_aux_resume(manifest: Mapping[str, Any], *, aux_enabled: bool, state_definition, force_info,
                      assignments: Sequence[int], observed_params: Sequence[tuple[float, float]]) -> None:
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
        raise IntegrityError(f"state_definition_sha256 changed: checkpoint {block['state_definition_sha256']} vs run {current}")
    if json_loads(json_bytes(block["force_info"])) != json_loads(json_bytes(_force_record(force_info))):
        raise IntegrityError(f"force_info changed: checkpoint {block['force_info']} vs run {_force_record(force_info)}")
    if int(block["n_replicas"]) != len(assignments) or len(observed_params) != len(assignments):
        raise IntegrityError(f"replica/carrier set incomplete: checkpoint {block['n_replicas']}, run {len(assignments)}")
    if block["assignment_sha256"] != assignment_sha256(assignments):
        raise IntegrityError("resumed assignment does not match the checkpoint's assignment checksum")
    state = canonical_state_definition(state_definition)
    for r, (window, observed) in enumerate(zip(assignments, observed_params)):
        expect = expected_aux_parameters(state, window)
        if not all(math.isclose(o, e, rel_tol=1e-12, abs_tol=0.0) for o, e in zip(observed, expect)):
            raise IntegrityError(f"replica {r} (window {window}) Context parameters {tuple(observed)} != "
                                 f"state table {expect}")
```

- [ ] **Step 4: Wire the optional blocks into `production.py`**

In `save_production_checkpoint`, add the keyword `aux_block: Optional[dict] = None` after `keep_generations`. Just before `from .correctness.checkpoint_store import publish_generation`, add:

```python
    if aux_block is not None:
        manifest["aux"] = aux_block
```

In `load_production_checkpoint`, add the keyword `aux_resume=None`. After the `for r, sim in enumerate(sims): ... _apply_assignment(r)` loop and before the `try: if manifest.get("rng_state")` block:

```python
    if aux_resume is not None:
        aux_resume(manifest, assignments)
```

With both keywords `None`, the code path and manifest bytes are unchanged. The existing checkpoint tests pin this: run `ls tests | grep -i checkpoint` and include the files that call `save_production_checkpoint` or `load_production_checkpoint` (`grep -l "save_production_checkpoint\|load_production_checkpoint" tests/*.py`).

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_aux_checkpoint.py` plus the files listed by `grep -l "save_production_checkpoint\|load_production_checkpoint" tests/*.py`.
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add gareus/auxiliary_cv/checkpoint.py gareus/production.py tests/test_aux_checkpoint.py
git commit -m "feat(cvaux): bind auxiliary state to checkpoints and verify Context parameters on resume"
```

---

### Task 7: Strict exporter evaluates auxiliary energies offline, with an exclusion report

**Files:**
- Create: `gareus/auxiliary_cv/offline.py` (`aux_z_from_samples`, `exclusion_report`)
- Modify: `gareus/correctness/export.py:63-163` (`build_export_arrays`), `:189` (`export_fixed_state_npz`)
- Modify: `gareus/query.py:377-395` (`reconstruct_bias_matrix` gains the `aux_z=None` pass-through)
- Modify: `gareus/store.py` (`WindowSnapshot.snapshot`, keyword `state_definition=None`, `phase_kind="production"`, `equilibrium_analysis_eligible=None`)
- Test: `tests/test_aux_export.py`, `tests/test_aux_window_snapshot.py`

**Interfaces:**
- Consumes: Task 2 `AuxSampleSchema`; Stage A `z_from_dihedrals`, `AuxModel`, `reconstruct_bias_matrix(aux_z=)`; `state_identity.freeze_snapshot`, `STATE_SCHEMA_V2`.
- Produces:
  - `aux_z_from_samples(samples, schema, models, *, parity_tol=1e-12) -> dict[str, np.ndarray]`. It recomputes z from the `tor_###` columns for every schema model. Where a stored `aux_z_##` exists and both values are finite, they must agree within `parity_tol` (absolute), else `IntegrityError`. A stored value that is finite where the recomputation is NaN, or the reverse, also raises. The output is float64, NaN where undefined.
  - `exclusion_report(excluded, *, origin_ids, replicas, steps, segment_ids, aux_z, time_block_steps, z_bins=10) -> dict`.
  - `build_export_arrays(..., aux_sample_schema: AuxSampleSchema | None = None, on_incomplete: str = "refuse", time_block_steps: int | None = None)`.
  - `WindowSnapshot.snapshot(..., state_definition=None, phase_kind="production", equilibrium_analysis_eligible=None)`. With `state_definition`, the payload becomes `{**freeze_snapshot(segment_id, state_definition, ...), "kernel_identity": ...}`, so `validate_fixed_state_segments` accepts it. The rows are the canonical v2 rows, so `load_windows` returns v2 rows. A rewrite with different bytes raises `IntegrityError`.
- Exporter rules:
  - If the frozen table has any active aux row (`aux_k > 0`), samples must carry the schema's torsion columns, else `IntegrityError` ("historical rows lack auxiliary features").
  - `aux_sample_schema` must be given, and its `model_shas` must cover every active model.
  - z is computed from torsions for **every** row, including ordinary-origin rows, and passed as `aux_z=`.
  - `on_incomplete="refuse"`: identical to today, any incomplete row raises.
  - `on_incomplete="exclude_and_report"`: drops incomplete rows from every per-sample array, recomputes `N_k`, and adds `exclusion_report_json` (array) plus `manifest["exclusion_report"]`. It requires `time_block_steps`.
  - The torsion and z columns enter `source_arrays`, so a changed feature invalidates the export signature.
  - With no active aux rows (legacy v1 or an all-sham v2), behaviour and output bytes are unchanged apart from v2's state definition.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_aux_window_snapshot.py
import json

import pytest

from aux_cv_fixture import model_payload
from gareus.auxiliary_cv.model import AuxModel
from gareus.correctness._io import IntegrityError
from gareus.correctness.state_identity import make_state_definition, validate_fixed_state_segments

MODEL = AuxModel.from_mapping(model_payload([(0, 1, 2, 3)], [1.0, 0.5], offset=0.1))


def _defn():
    rows = [{"window_id": 0, "center1": 0.2, "k1": 10.0, "center2": 0.0, "k2": 0.0, "gamd_lambda": 0.0, "aux_k": 0.0},
            {"window_id": 1, "center1": 0.2, "k1": 10.0, "center2": 0.0, "k2": 0.0, "gamd_lambda": 0.0,
             "aux_model_sha256": MODEL.model_sha256, "aux_center": 1.0, "aux_k": 2.0}]
    return make_state_definition(rows, physical_system_sha256="a" * 64, ensemble="NVT", temperature_k=300.0,
                                 fixed_box_vectors_nm=[[3, 0, 0], [0, 3, 0], [0, 0, 3]],
                                 cv1={"kind": "contacts", "units": "dimensionless", "definition": {"r0": 4.5}},
                                 cv2=None, aux_models={MODEL.model_sha256: MODEL.to_mapping()})


def test_legacy_snapshot_bytes_unchanged(tmp_path):
    from gareus.store import WindowSnapshot
    WindowSnapshot(tmp_path).snapshot("seg_001", [{"window_id": 0, "center1": 0.1, "k1": 1.0}], "contacts", None)
    assert json.loads((tmp_path / "windows" / "seg_001.json").read_text()) == {
        "segment_id": "seg_001", "cv1_type": "contacts", "cv2_type": None,
        "windows": [{"window_id": 0, "center1": 0.1, "k1": 1.0}]}


def test_aux_snapshot_is_a_frozen_v2_snapshot(tmp_path):
    from gareus.query import load_windows
    from gareus.store import WindowSnapshot
    WindowSnapshot(tmp_path).snapshot("seg_001", [], "contacts", None, state_definition=_defn(),
                                      equilibrium_analysis_eligible=True)
    payload = json.loads((tmp_path / "windows" / "seg_001.json").read_text())
    table = validate_fixed_state_segments({"seg_001": payload})
    assert table.definition["aux_models"]
    assert load_windows(tmp_path, "seg_001")[1]["aux_k"] == 2.0


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
import numpy as np
import pytest

from aux_cv_fixture import model_payload
from gareus.auxiliary_cv.evaluate import z_from_dihedrals
from gareus.auxiliary_cv.model import AuxModel
from gareus.auxiliary_cv.offline import aux_z_from_samples, exclusion_report
from gareus.auxiliary_cv.sample_schema import AuxSampleSchema, _basis_sha
from gareus.correctness._io import IntegrityError
from gareus.correctness.export import build_export_arrays
from gareus.correctness.state_identity import freeze_snapshot, make_state_definition

QUADS = ((0, 1, 2, 3), (1, 2, 3, 4))
LABELS = ("phi-A1", "psi-A1")
MODEL = AuxModel.from_mapping(model_payload(list(QUADS), [1.0, 0.0, 0.0, -0.7], offset=0.2,
                                            blocks=["phi", "psi"]))
SCHEMA = AuxSampleSchema(QUADS, LABELS, (MODEL.model_sha256,), _basis_sha(QUADS, LABELS))
R = 0.0083144626
BETA = 1.0 / (R * 300.0)
VIEW = {"kind": "immutable", "boundary_id": "b1"}


def _defn(all_lambda_zero=True):
    rows = [{"window_id": 0, "center1": 0.2, "k1": 10.0, "center2": 0.0, "k2": 0.0, "gamd_lambda": 0.0, "aux_k": 0.0},
            {"window_id": 1, "center1": 0.2, "k1": 10.0, "center2": 0.0, "k2": 0.0, "gamd_lambda": 0.0,
             "aux_model_sha256": MODEL.model_sha256, "aux_center": 0.5, "aux_k": 2.0}]
    return make_state_definition(rows, physical_system_sha256="a" * 64, ensemble="NVT", temperature_k=300.0,
                                 fixed_box_vectors_nm=[[3, 0, 0], [0, 3, 0], [0, 0, 3]],
                                 cv1={"kind": "contacts", "units": "dimensionless", "definition": {"r0": 4.5}},
                                 cv2=None, aux_models={MODEL.model_sha256: MODEL.to_mapping()})


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


def test_ordinary_origin_rows_get_auxiliary_cross_energy_in_lambda_zero_table():
    s = _samples()
    a = build_export_arrays(s, _snap(), BETA, sample_view=VIEW, aux_sample_schema=SCHEMA)
    u = a["umbrella_reduced_bias_nk"]
    z = z_from_dihedrals(np.stack([s["tor_000"], s["tor_001"]], axis=1), MODEL)
    extra = BETA * 4.184 * 0.5 * 2.0 * (z - 0.5) ** 2
    np.testing.assert_allclose(u[:, 1] - u[:, 0], extra, rtol=1e-12)
    assert np.all(u[s["window_id"] == 0, 1] > u[s["window_id"] == 0, 0])


def test_missing_features_fail_closed():
    s = _samples()
    for key in ("tor_000", "tor_001", "aux_z_00"):
        s.pop(key)
    with pytest.raises(IntegrityError, match="auxiliary features"):
        build_export_arrays(s, _snap(), BETA, sample_view=VIEW, aux_sample_schema=SCHEMA)


def test_schema_required_and_must_cover_active_models():
    with pytest.raises(IntegrityError, match="aux_sample_schema"):
        build_export_arrays(_samples(), _snap(), BETA, sample_view=VIEW)
    other = AuxSampleSchema(QUADS, LABELS, ("f" * 64,), _basis_sha(QUADS, LABELS))
    with pytest.raises(IntegrityError, match="model"):
        build_export_arrays(_samples(), _snap(), BETA, sample_view=VIEW, aux_sample_schema=other)


def test_stored_z_parity_is_checked():
    s = _samples()
    s["aux_z_00"] = s["aux_z_00"] + 1e-6
    with pytest.raises(IntegrityError, match="parity"):
        aux_z_from_samples(s, SCHEMA, {MODEL.model_sha256: MODEL})


def test_refuse_vs_exclude_and_report():
    s = _samples(n=8)
    s["tor_000"] = s["tor_000"].copy(); s["aux_z_00"] = s["aux_z_00"].copy()
    s["tor_000"][[0, 2]] = np.nan          # two ordinary-origin rows lose their aux coordinate
    s["aux_z_00"][[0, 2]] = np.nan
    with pytest.raises(IntegrityError, match="Incomplete"):
        build_export_arrays(s, _snap(), BETA, sample_view=VIEW, aux_sample_schema=SCHEMA)
    a = build_export_arrays(s, _snap(), BETA, sample_view=VIEW, aux_sample_schema=SCHEMA,
                            on_incomplete="exclude_and_report", time_block_steps=400)
    assert a["umbrella_reduced_bias_nk"].shape[0] == 6 and int(a["N_k"].sum()) == 6
    import json
    rep = json.loads(str(a["exclusion_report_json"].item()))
    assert rep["n_excluded"] == 2 and rep["by_origin_state"] == {"0": 2}
    assert rep["by_carrier"] == {"0": 2} and rep["by_time_block"] == {"seg_001:0": 2}
    assert rep["by_z_range"][MODEL.model_sha256]["nonfinite"] == 2
    assert rep["by_structural_group"].startswith("unavailable")


def test_exclusion_report_bins_finite_z():
    excluded = np.array([True, False, True, False])
    rep = exclusion_report(excluded, origin_ids=np.array([1, 1, 1, 0]), replicas=np.array([3, 3, 4, 4]),
                           steps=np.array([0, 100, 900, 950]), segment_ids=np.array(["s"] * 4),
                           aux_z={"e" * 64: np.array([0.0, 1.0, 2.0, 3.0])}, time_block_steps=500, z_bins=3)
    assert rep["by_time_block"] == {"s:0": 1, "s:1": 1} and rep["by_carrier"] == {"3": 1, "4": 1}
    assert sum(rep["by_z_range"]["e" * 64]["counts"]) == 2 and rep["fraction"] == 0.5
    assert rep["above_audit_threshold"] is True


def test_legacy_v1_export_unchanged_without_schema():
    rows = [{"window_id": 0, "center1": 0.2, "k1": 10.0, "center2": 0.0, "k2": 0.0, "gamd_lambda": 0.0}]
    d = make_state_definition(rows, physical_system_sha256="a" * 64, ensemble="NVT", temperature_k=300.0,
                              fixed_box_vectors_nm=[[3, 0, 0], [0, 3, 0], [0, 0, 3]],
                              cv1={"kind": "contacts", "units": "dimensionless", "definition": {"r0": 4.5}}, cv2=None)
    snaps = {"seg_001": freeze_snapshot("seg_001", d, equilibrium_analysis_eligible=True, phase_kind="production")}
    s = {"cv1": np.array([0.1, 0.3]), "window_id": np.array([0, 0]), "segment_id": np.array(["seg_001"] * 2)}
    a = build_export_arrays(s, snaps, BETA, sample_view=VIEW)
    assert "exclusion_report_json" not in a
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_aux_export.py tests/test_aux_window_snapshot.py`
Expected: FAIL `ModuleNotFoundError: No module named 'gareus.auxiliary_cv.offline'` and `TypeError: snapshot() got an unexpected keyword argument 'state_definition'`

- [ ] **Step 3: Implement `offline.py` (first part)**

```python
# gareus/auxiliary_cv/offline.py
"""Offline auxiliary energies from stored features, exclusion audits and MVP pooling guards."""
from __future__ import annotations

import json
from typing import Any, Mapping

import numpy as np

from ..correctness._io import IntegrityError
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


def aux_z_from_samples(samples: Mapping[str, Any], schema: AuxSampleSchema,
                       models: Mapping[str, AuxModel], *, parity_tol: float = 1e-12) -> dict[str, np.ndarray]:
    n = len(np.asarray(samples["cv1"]))
    missing = [c for c in schema.torsion_columns if c not in samples]
    if missing:
        raise IntegrityError(f"historical rows lack auxiliary features (columns {missing[:3]}...); "
                             "they cannot enter a pool with active auxiliary states")
    theta = np.stack([_float_column(samples, c, n) for c in schema.torsion_columns], axis=1)
    out: dict[str, np.ndarray] = {}
    for col, sha in zip(schema.z_columns, schema.model_shas):
        if sha not in models:
            raise IntegrityError(f"no aux model loaded for schema model {sha}")
        z = z_from_dihedrals(theta[:, schema.model_basis_index(models[sha])], models[sha]).astype(np.float64)
        if col in samples:
            stored = _float_column(samples, col, n)
            both = np.isfinite(stored) & np.isfinite(z)
            if np.any(np.isfinite(stored) != np.isfinite(z)) or np.any(np.abs(stored[both] - z[both]) > parity_tol):
                raise IntegrityError(f"stored {col} fails offline parity against model {sha}")
        out[sha] = z
    return out


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

Change the signature:

```python
def build_export_arrays(
    samples: Mapping[str, Any], snapshots: Mapping[str, Mapping[str, Any]], beta: float,
    *, sample_view: Mapping[str, Any], envelope_factory: Callable | None = None,
    reconstruct: Callable = reconstruct_bias_matrix, aux_sample_schema=None,
    on_incomplete: str = "refuse", time_block_steps: int | None = None,
) -> dict[str, np.ndarray]:
```

Right after `envelope = ...`, add:

```python
    if on_incomplete not in ("refuse", "exclude_and_report"):
        raise IntegrityError(f"on_incomplete must be 'refuse' or 'exclude_and_report', got {on_incomplete!r}")
    active_aux = sorted({w["aux_model_sha256"] for w in table.windows if w.get("aux_k", 0.0) > 0})
    aux_z = None
    if active_aux:
        from ..auxiliary_cv.model import AuxModel
        from ..auxiliary_cv.offline import aux_z_from_samples
        if aux_sample_schema is None:
            raise IntegrityError("active auxiliary states need the run's aux_sample_schema to evaluate z")
        uncovered = [sha for sha in active_aux if sha not in aux_sample_schema.model_shas]
        if uncovered:
            raise IntegrityError(f"aux_sample_schema does not record active model(s) {uncovered}")
        models = {sha: AuxModel.from_mapping(table.definition["aux_models"][sha]) for sha in aux_sample_schema.model_shas}
        aux_z = aux_z_from_samples(samples, aux_sample_schema, models)
    extra = {"aux_z": aux_z} if aux_z is not None else {}
```

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

After building `arrays` (before `source_arrays`), when `report is not None`, filter every per-sample array and recompute `N_k`:

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

Move the `for name in ("step", "replica"): if name in samples: arrays[name] = ...` loop above this filter so they are filtered too. In `source_arrays`, add the features when `aux_z is not None`:

```python
    if aux_z is not None:
        for col in aux_sample_schema.torsion_columns:
            source_arrays[col] = numeric_vector(samples[col], col, n)
        for sha, z in aux_z.items():
            source_arrays[f"aux_z:{sha}"] = z
```

Add `"exclusion_report": report` to `manifest` only when `report is not None`, and `"on_incomplete": on_incomplete` to `signature_payload` only when it differs from `"refuse"`. Legacy signatures stay byte-identical. `exclusion_report_json` is included in `array_hashes` automatically because it is in `arrays`. Thread the three new keywords through `export_fixed_state_npz`.

- [ ] **Step 5: Implement `WindowSnapshot.snapshot(state_definition=...)` and the query pass-through**

In `gareus/store.py` `WindowSnapshot.snapshot`, add the keywords `state_definition=None, phase_kind: str = "production", equilibrium_analysis_eligible=None`. At the top of the body:

```python
        if state_definition is not None:
            from .correctness.state_identity import freeze_snapshot, write_frozen_snapshot
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

`write_frozen_snapshot` refuses a differing rewrite ("Refusing to overwrite immutable state snapshot"). It calls `validate_fixed_state_segments` on the frozen keys. If its check rejects the extra `kernel_identity` key, keep `kernel_identity` out of `frozen` and record it inside `state_definition`'s boost or CV metadata instead. Do not weaken the validator.

In `gareus/query.py`, add `aux_z=None` to `reconstruct_bias_matrix` and pass `aux_z=aux_z` to `_strict_bias`.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `pytest tests/test_aux_export.py tests/test_aux_window_snapshot.py tests/test_query.py tests/test_query_reconstruct_bias_matrix_nan_guard.py tests/test_store.py`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add gareus/auxiliary_cv/offline.py gareus/correctness/export.py gareus/query.py gareus/store.py tests/test_aux_export.py tests/test_aux_window_snapshot.py
git commit -m "feat(cvaux): strict exporter evaluates auxiliary energies from stored features with exclusion audit"
```

---

### Task 8: Loaders evaluate or refuse; historical segments fail closed

**Files:**
- Modify: `gareus/auxiliary_cv/offline.py` (`segment_aux_schemas`, `refuse_aux_snapshots`)
- Modify: `gareus/mbar_analysis/loaders.py:514-642` (`load_parquet`, keyword `exclude_segments_without_aux_features=False`)
- Modify: `gareus/query.py:398` (`export_analysis_arrays_npz` refuses aux snapshots)
- Modify: `gareus/mbar_analysis/loaders_union_parquet.py:303` (`load_parquet_adaptive_union`), `gareus/adaptive_production.py:3715` (`build_union_state_mbar_inputs`): call `refuse_aux_snapshots` at entry
- Test: `tests/test_aux_loaders.py`

**Interfaces:**
- Produces:
  - `segment_aux_schemas(run_dir) -> dict[str, AuxSampleSchema | None]`, one entry per sample segment, from the manifest's `payload_schema`;
  - `refuse_aux_snapshots(root, *, where: str) -> None`, which scans `root.rglob("windows/*.json")` and raises `IntegrityError(f"{where}: auxiliary states pool only through the fixed-state exporter or load_parquet in the MVP ...")` if any snapshot has a v2 `state_definition` with active aux rows or any window with `aux_k > 0`.
- `load_parquet` rules:
  - Read the latest snapshot payload (`load_windows_metadata(prod)`). If its windows include an active aux row:
    - (a) group the segments by `segment_aux_schemas`;
    - (b) segments with `None`: raise `IntegrityError` naming them, unless `exclude_segments_without_aux_features=True`, in which case drop their rows and append a load note with segment ids and row counts;
    - (c) more than one distinct schema among the remaining segments raises;
    - (d) build the models from the payload's `state_definition["aux_models"]`, compute `aux_z_from_samples`, and pass `aux_z=` to `reconstruct_bias_matrix`;
    - (e) record `meta["aux_models"]` and `meta["aux_feature_segments"]`.
  - The check happens on per-segment manifests before the rows are used. The placeholder back-fill in `_concat_numpy_dicts` can never reach `reconstruct_bias_matrix` as data.
  - Without active aux rows: unchanged.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_aux_loaders.py
import json

import numpy as np
import pytest

from aux_cv_fixture import model_payload
from gareus.auxiliary_cv.evaluate import z_from_dihedrals
from gareus.auxiliary_cv.model import AuxModel
from gareus.auxiliary_cv.offline import refuse_aux_snapshots, segment_aux_schemas
from gareus.auxiliary_cv.sample_schema import AuxSampleSchema, _basis_sha
from gareus.correctness._io import IntegrityError
from gareus.correctness.state_identity import make_state_definition

QUADS = ((0, 1, 2, 3), (1, 2, 3, 4))
LABELS = ("phi-A1", "psi-A1")
MODEL = AuxModel.from_mapping(model_payload(list(QUADS), [1.0, 0.0, 0.0, -0.7], offset=0.2, blocks=["phi", "psi"]))
SCHEMA = AuxSampleSchema(QUADS, LABELS, (MODEL.model_sha256,), _basis_sha(QUADS, LABELS))


def _defn():
    rows = [{"window_id": 0, "center1": 0.2, "k1": 10.0, "center2": 0.0, "k2": 0.0, "gamd_lambda": 0.0, "aux_k": 0.0},
            {"window_id": 1, "center1": 0.2, "k1": 10.0, "center2": 0.0, "k2": 0.0, "gamd_lambda": 0.0,
             "aux_model_sha256": MODEL.model_sha256, "aux_center": 0.5, "aux_k": 2.0}]
    return make_state_definition(rows, physical_system_sha256="a" * 64, ensemble="NVT", temperature_k=300.0,
                                 fixed_box_vectors_nm=[[3, 0, 0], [0, 3, 0], [0, 0, 3]],
                                 cv1={"kind": "contacts", "units": "dimensionless", "definition": {"r0": 4.5}},
                                 cv2=None, aux_models={MODEL.model_sha256: MODEL.to_mapping()})


def _write_segment(run, seg, aux, rng):
    from gareus.store import ParquetSampleWriter
    w = ParquetSampleWriter(run / "samples" / seg, aux_schema=SCHEMA if aux else None)
    for i in range(6):
        theta = rng.uniform(-np.pi, np.pi, size=2)
        kw = {}
        if aux:
            kw = dict(torsions=theta, aux_z=[float(z_from_dihedrals(theta[None, :], MODEL)[0])])
        w.write_sample(step=100 * i, replica=i % 2, window_id=i % 2, cv1=0.1, cv2=None, potential=0.0,
                       boost_total=None, boost_dihedral=None, boost_nonbonded=None, **kw)
    w.close()


def _run(tmp_path, *, historical):
    from gareus.store import SegmentRegistry, WindowSnapshot
    rng = np.random.default_rng(1)
    reg = SegmentRegistry(tmp_path)
    segs = []
    if historical:
        s0 = reg.open_segment("run", None, 1); _write_segment(tmp_path, s0, False, rng); reg.close_segment(s0, 600)
        WindowSnapshot(tmp_path).snapshot(s0, [{"window_id": 0, "center1": 0.2, "k1": 10.0},
                                               {"window_id": 1, "center1": 0.2, "k1": 10.0}], "contacts", None)
        segs.append(s0)
    s1 = reg.open_segment("run", segs[-1] if segs else None, 1)
    _write_segment(tmp_path, s1, True, rng); reg.close_segment(s1, 1200)
    WindowSnapshot(tmp_path).snapshot(s1, [], "contacts", None, state_definition=_defn(),
                                      equilibrium_analysis_eligible=True)
    (tmp_path / "gareus_metadata.json").write_text(json.dumps({"temperature_K": 300.0}))
    return segs + [s1]


def test_segment_schemas(tmp_path):
    s0, s1 = _run(tmp_path, historical=True)
    sch = segment_aux_schemas(tmp_path)
    assert sch[s0] is None and sch[s1] == SCHEMA


def test_load_parquet_evaluates_aux_for_every_row(tmp_path):
    from gareus.mbar_analysis.loaders import load_parquet
    _run(tmp_path, historical=False)
    d = load_parquet(tmp_path)
    assert np.isfinite(d.u_nk).all()
    assert np.all(d.u_nk[:, 1] >= d.u_nk[:, 0])
    assert d.meta["aux_models"] == [MODEL.model_sha256]


def test_historical_segment_without_features_fails_closed(tmp_path):
    from gareus.mbar_analysis.loaders import load_parquet
    s0, _s1 = _run(tmp_path, historical=True)
    with pytest.raises(IntegrityError, match=s0):
        load_parquet(tmp_path)


def test_explicit_opt_out_excludes_and_reports(tmp_path):
    from gareus.mbar_analysis.loaders import load_parquet
    s0, _s1 = _run(tmp_path, historical=True)
    d = load_parquet(tmp_path, exclude_segments_without_aux_features=True)
    assert d.u_nk.shape[0] == 6
    assert any(s0 in note and "6" in note for note in d.meta["load_notes"])


def test_union_and_legacy_npz_paths_refuse_aux(tmp_path):
    from gareus.query import export_analysis_arrays_npz
    _run(tmp_path, historical=False)
    with pytest.raises(IntegrityError, match="fixed-state exporter"):
        refuse_aux_snapshots(tmp_path, where="adaptive union")
    with pytest.raises(IntegrityError, match="fixed-state exporter"):
        export_analysis_arrays_npz(tmp_path, beta=0.4)


def test_refuse_is_silent_without_aux(tmp_path):
    from gareus.store import WindowSnapshot
    WindowSnapshot(tmp_path).snapshot("seg_001", [{"window_id": 0, "center1": 0.2, "k1": 1.0}], "contacts", None)
    refuse_aux_snapshots(tmp_path, where="x")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_aux_loaders.py`
Expected: FAIL `ImportError: cannot import name 'refuse_aux_snapshots'`

- [ ] **Step 3: Implement the helpers in `offline.py`**

```python
def segment_aux_schemas(run_dir) -> dict[str, "AuxSampleSchema | None"]:
    from pathlib import Path
    from ..parquet_manifest import load_manifest
    out = {}
    root = Path(run_dir) / "samples"
    if not root.exists():
        return out
    for seg_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        manifest = load_manifest(seg_dir, expected_kind="samples")
        payload = (manifest or {}).get("payload_schema")
        out[seg_dir.name] = AuxSampleSchema.from_payload(payload) if payload else None
    return out


def _snapshot_has_active_aux(payload: Mapping[str, Any]) -> bool:
    rows = list(payload.get("windows") or [])
    state = payload.get("state_definition") or {}
    rows += list(state.get("windows") or [])
    return any(float(r.get("aux_k", 0.0) or 0.0) > 0 for r in rows)


def refuse_aux_snapshots(root, *, where: str) -> None:
    from pathlib import Path
    hits = []
    for path in sorted(Path(root).rglob("windows/*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(payload, dict) and _snapshot_has_active_aux(payload):
            hits.append(str(path))
    if hits:
        raise IntegrityError(f"{where}: auxiliary states pool only through the fixed-state exporter or "
                             f"load_parquet in the MVP (spec Sections 8/14); found {hits[:3]}")
```

`refuse_aux_snapshots` skips unreadable snapshots deliberately. Corrupt-snapshot handling stays with the existing loaders, which already raise on them. Rows reading `aux_k` here are dict rows.

- [ ] **Step 4: Implement `load_parquet` and the three guards**

In `gareus/mbar_analysis/loaders.py` `load_parquet`:
- Change the signature to `def load_parquet(prod: Path, *, exclude_segments_without_aux_features: bool = False) -> Data:`.
- After `windows = load_windows(prod)` and the metadata/beta lines, insert:

```python
    from gareus.query import load_windows_metadata
    payload = load_windows_metadata(prod) or {}
    active_aux = any(float(w.get('aux_k', 0.0) or 0.0) > 0 for w in windows)
    aux_z = None
    if active_aux:
        from gareus.auxiliary_cv.model import AuxModel
        from gareus.auxiliary_cv.offline import aux_z_from_samples, segment_aux_schemas
        from gareus.correctness._io import IntegrityError
        schemas = segment_aux_schemas(prod)
        seg_col = np.asarray(samples['segment_id']).astype(str)
        present = sorted(set(seg_col.tolist()))
        missing = [s for s in present if schemas.get(s) is None]
        if missing and not exclude_segments_without_aux_features:
            raise IntegrityError(f'segments {missing} lack auxiliary features but the state table has active '
                                 'auxiliary states; pass exclude_segments_without_aux_features=True to drop them')
        if missing:
            keep = ~np.isin(seg_col, missing)
            dropped = {s: int(np.count_nonzero(seg_col == s)) for s in missing}
            samples = {k: (v[keep] if hasattr(v, '__len__') and len(v) == len(seg_col) else v) for k, v in samples.items()}
            meta.setdefault('load_notes', []).append(f'excluded featureless segments (rows): {dropped}')
        distinct = {schemas[s] for s in present if s not in missing}
        if len(distinct) != 1:
            raise IntegrityError(f'auxiliary sample schema differs between segments: {sorted(present)}')
        schema = distinct.pop()
        state = payload.get('state_definition') or {}
        models = {sha: AuxModel.from_mapping(state['aux_models'][sha]) for sha in schema.model_shas}
        aux_z = aux_z_from_samples(samples, schema, models)
        meta['aux_models'] = list(schema.model_shas)
        meta['aux_feature_segments'] = [s for s in present if s not in missing]
```

Every later array read (`cv`, `cv2_raw`, `window`, …) must come from the possibly filtered `samples`, so place this block before `cv = samples['cv1']...`. Pass `aux_z=aux_z` to the `reconstruct_bias_matrix(...)` call (the query wrapper from Task 7).

In `gareus/query.py` `export_analysis_arrays_npz`, add as the first line of the body:

```python
    from .auxiliary_cv.offline import refuse_aux_snapshots
    refuse_aux_snapshots(Path(run_dir), where="legacy analysis_arrays.npz export")
```

At the start of `load_parquet_adaptive_union`'s body (`loaders_union_parquet.py:303`), add `refuse_aux_snapshots(Path(adaptive_dir), where="adaptive union loader")`. Add the same to `build_union_state_mbar_inputs` (`adaptive_production.py:3715`), right after `adaptive_dir = Path(adaptive_dir)`. Import from `gareus.auxiliary_cv.offline` in each file.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_aux_loaders.py tests/test_query.py tests/test_lambda_ladder_mbar.py tests/test_low_memory_complete.py`
Expected: PASS. The last three are the legacy loader regressions.

- [ ] **Step 6: Commit**

```bash
git add gareus/auxiliary_cv/offline.py gareus/mbar_analysis/loaders.py gareus/mbar_analysis/loaders_union_parquet.py gareus/adaptive_production.py gareus/query.py tests/test_aux_loaders.py
git commit -m "feat(cvaux): loaders evaluate auxiliary energies or fail closed on featureless segments"
```

---

### Task 9: Duplicate Hamiltonians: separate and merged MBAR agree

**Files:**
- Modify: `gareus/auxiliary_cv/offline.py` (`merge_identical_hamiltonians`)
- Test: `tests/test_aux_mbar_duplicates.py`

**Interfaces:**
- Consumes: Stage A `hamiltonian_sha256(definition, window_id)`; `gareus.adaptive.mbar_solve.solve_rows(u_nk, window) -> (f, logw)`.
- Produces: `merge_identical_hamiltonians(u_nk, origins, hamiltonian_ids) -> tuple[np.ndarray, np.ndarray, list[list[int]]]`. It returns the merged u (N, K'), the merged origins, and `groups` (the original columns per merged column).
  - Columns sharing a Hamiltonian id must be **bitwise equal**, NaN-equal included; otherwise `IntegrityError`, because a hash collision or a stale table would otherwise merge distinct states.
  - It never merges on CV centres (spec Section 4.3).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_aux_mbar_duplicates.py
import numpy as np
import pytest

from gareus.adaptive.mbar_solve import solve_rows
from gareus.auxiliary_cv.offline import merge_identical_hamiltonians
from gareus.correctness._io import IntegrityError
from gareus.correctness.state_identity import hamiltonian_sha256, make_state_definition


def _defn():
    rows = []
    for i, (c, role) in enumerate([(-1.0, "ordinary"), (0.0, "ordinary"), (1.0, "ordinary"), (0.0, "sham")]):
        rows.append({"window_id": i, "center1": c, "k1": 4.0, "center2": 0.0, "k2": 0.0, "gamd_lambda": 0.0,
                     "aux_k": 0.0, "instance": {"state_instance_id": f"s{i}", "state_role": role,
                                                "spawn_parent_state_id": 1 if role == "sham" else None,
                                                "spawn_source_observation": None,
                                                "matched_additional_slot_id": "slot-0" if role == "sham" else None}})
    return make_state_definition(rows, physical_system_sha256="a" * 64, ensemble="NVT", temperature_k=300.0,
                                 fixed_box_vectors_nm=[[3, 0, 0], [0, 3, 0], [0, 0, 3]],
                                 cv1={"kind": "x", "units": "dimensionless", "definition": {"a": 1}}, cv2=None,
                                 aux_models={})


def _data(seed=4):
    """Exact samples of a harmonic umbrella on a flat 1-D landscape: x ~ N(c, 1/(beta k))."""
    rng = np.random.default_rng(seed)
    beta, k = 1.0, 4.0
    centers = np.array([-1.0, 0.0, 1.0, 0.0])
    x = np.concatenate([rng.normal(c, 1.0 / np.sqrt(beta * k), 2000) for c in centers])
    origins = np.repeat(np.arange(4), 2000)
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
    # target (unbiased) weights per sample agree
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

Two notes on the test:
- `solve_rows` returns `logw` as the per-sample log target weight (`gareus/adaptive/mbar_solve.py:24-47`). If its convention is unnormalised log weights, normalising as above makes the comparison convention-free.
- `aux_models={}` builds a v2 table whose rows are all inactive. If Stage A refuses an empty registry, use the one-model registry from `tests/test_aux_export.py`'s `_defn()`. The test only needs `instance` metadata and `aux_k = 0`.

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_aux_mbar_duplicates.py`
Expected: FAIL `ImportError: cannot import name 'merge_identical_hamiltonians'`

- [ ] **Step 3: Implement**

Append to `gareus/auxiliary_cv/offline.py`:

```python
def merge_identical_hamiltonians(u_nk, origins, hamiltonian_ids):
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
            a, b = u[:, first], u[:, col]
            if not np.array_equal(a, b, equal_nan=True):
                raise IntegrityError(f"columns {first} and {col} share Hamiltonian {hid} but are not bitwise equal")
            groups[where[hid]].append(col)
    merged = u[:, [g[0] for g in groups]].copy()
    col_to_group = {c: gi for gi, g in enumerate(groups) for c in g}
    merged_origins = np.asarray([col_to_group[int(o)] for o in origins], dtype=np.int64)
    return merged, merged_origins, groups
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_aux_mbar_duplicates.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add gareus/auxiliary_cv/offline.py tests/test_aux_mbar_duplicates.py
git commit -m "feat(cvaux): merge identical Hamiltonians bitwise-checked; separate and merged MBAR agree"
```

---

### Task 10: Production wiring (after Stage B's runtime tasks)

**Prerequisite:** Stage B is merged. `run_gareus` must then have `aux_runtime` and per-carrier `aux_obs` at the sample step, plus the Gibbs/neighbour decision objects (see "Interfaces expected from Stage B"). If Stage B named them differently, adapt the names here and nowhere else.

**Files:**
- Create: `gareus/auxiliary_cv/runtime_io.py`
- Modify: `gareus/production.py`:
  - writer construction (`:7899-7906`);
  - sample write (`:8225-8239`);
  - swap event (`:8427-8435`);
  - Gibbs stays and no-candidate skips (`:8597-8612`);
  - snapshot (`:7897`);
  - checkpoint save calls (find with `grep -n "save_production_checkpoint(" gareus/production.py`);
  - checkpoint load call (`grep -n "load_production_checkpoint(" gareus/production.py`).
- Test: `tests/test_aux_runtime_io.py`, plus the existing ordering test `tests/test_sample_before_exchange_ordering.py`

**Interfaces:**
- Produces:
  - `check_exchange_boundary_alignment(*, exchange_interval: int, distance_output_interval: int, traj_interval: int) -> None`. With aux on, both `distance_output_interval` and `traj_interval` must divide `exchange_interval`, so samples and structural frames exist on every exchange boundary (spec Section 7). Otherwise it raises `IntegrityError` naming the values.
  - `ExchangeEventCounter`: `next(step) -> int` returns the within-step attempt sequence, resetting when the step changes.
  - `event_from_gibbs(prop, *, step, seq, selected_replica, replica_i, replica_j, accepted, energy_version, assignments_after) -> dict`: `write_event` keywords from a `GibbsProposal` (`current_window`, `proposed_window`, `delta_kj`, `q_forward`, `q_reverse`, `pacc`, `stayed`, `no_candidates`; `gareus/production.py:1324-1395`). `log q` = `log(q)` when q > 0, else `-inf`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_aux_runtime_io.py
import math
from types import SimpleNamespace

import pytest

from gareus.auxiliary_cv.runtime_io import (ExchangeEventCounter, check_exchange_boundary_alignment,
                                            event_from_gibbs)
from gareus.correctness._io import IntegrityError


def test_alignment():
    check_exchange_boundary_alignment(exchange_interval=3000, distance_output_interval=300, traj_interval=3000)
    with pytest.raises(IntegrityError, match="traj_interval"):
        check_exchange_boundary_alignment(exchange_interval=3000, distance_output_interval=300, traj_interval=2000)
    with pytest.raises(IntegrityError, match="distance_output_interval"):
        check_exchange_boundary_alignment(exchange_interval=3000, distance_output_interval=700, traj_interval=3000)


def test_counter_orders_within_step():
    c = ExchangeEventCounter()
    assert [c.next(10), c.next(10), c.next(20), c.next(20), c.next(20)] == [0, 1, 0, 1, 2]


def test_event_from_gibbs_swap_and_stay():
    prop = SimpleNamespace(current_window=2, proposed_window=5, delta_kj=-1.5, q_forward=0.25, q_reverse=0.5,
                           pacc=1.0, stayed=False, no_candidates=False)
    ev = event_from_gibbs(prop, step=100, seq=3, selected_replica=7, replica_i=7, replica_j=1, accepted=True,
                          energy_version="state_bias_matrix_v3_aux", assignments_after=[0, 1])
    assert ev["kind"] == "swap" and ev["window_i"] == 2 and ev["window_j"] == 5 and ev["attempt_seq"] == 3
    assert ev["log_q_forward"] == pytest.approx(math.log(0.25)) and ev["log_q_reverse"] == pytest.approx(math.log(0.5))
    stay = SimpleNamespace(current_window=2, proposed_window=2, delta_kj=0.0, q_forward=0.6, q_reverse=0.0,
                           pacc=0.0, stayed=True, no_candidates=False)
    ev = event_from_gibbs(stay, step=100, seq=4, selected_replica=7, replica_i=7, replica_j=7, accepted=False,
                          energy_version="v", assignments_after=[0, 1])
    assert ev["kind"] == "stay" and ev["window_j"] == 2 and ev["log_q_reverse"] == float("-inf")
    none = SimpleNamespace(current_window=2, proposed_window=2, delta_kj=0.0, q_forward=0.0, q_reverse=0.0,
                           pacc=0.0, stayed=False, no_candidates=True)
    assert event_from_gibbs(none, step=1, seq=0, selected_replica=7, replica_i=7, replica_j=7, accepted=False,
                            energy_version="v", assignments_after=[0])["kind"] == "no_candidates"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_aux_runtime_io.py`
Expected: FAIL `ModuleNotFoundError: No module named 'gareus.auxiliary_cv.runtime_io'`

- [ ] **Step 3: Implement `runtime_io.py`**

```python
# gareus/auxiliary_cv/runtime_io.py
"""Small, testable glue between run_gareus and the auxiliary storage layer."""
from __future__ import annotations

import math
from typing import Any, Sequence

from ..correctness._io import IntegrityError


def check_exchange_boundary_alignment(*, exchange_interval: int, distance_output_interval: int,
                                      traj_interval: int) -> None:
    ex = int(exchange_interval)
    for name, value in (("distance_output_interval", distance_output_interval), ("traj_interval", traj_interval)):
        if int(value) <= 0 or ex % int(value) != 0:
            raise IntegrityError(f"auxiliary runs need {name} ({value}) to divide exchange_interval ({ex}) so "
                                 "observations exist on every exchange boundary (spec Section 7)")


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


def event_from_gibbs(prop, *, step: int, seq: int, selected_replica: int, replica_i: int, replica_j: int,
                     accepted: bool, energy_version: str, assignments_after: Sequence[int]) -> dict[str, Any]:
    kind = "no_candidates" if prop.no_candidates else ("stay" if prop.stayed else "swap")
    return dict(step=int(step), attempt_seq=int(seq), selected_replica=int(selected_replica),
                replica_i=int(replica_i), replica_j=int(replica_j), window_i=int(prop.current_window),
                window_j=int(prop.proposed_window), kind=kind, delta_e_kj=float(prop.delta_kj),
                accepted=bool(accepted), log_q_forward=_log(float(prop.q_forward)),
                log_q_reverse=_log(float(prop.q_reverse)), p_accept=float(prop.pacc),
                energy_version=str(energy_version), assignments_after=[int(x) for x in assignments_after])
```

- [ ] **Step 4: Wire `production.py`** (every change is guarded by `aux_runtime is not None`)

1. **Before `rng = np.random.default_rng(args.seed)` (`:7883`):**

```python
    if aux_runtime is not None:
        from .auxiliary_cv.runtime_io import check_exchange_boundary_alignment
        check_exchange_boundary_alignment(exchange_interval=int(args.exchange_interval),
                                          distance_output_interval=int(getattr(args, "distance_output_interval", 0) or args.exchange_interval),
                                          traj_interval=int(args.traj_interval))
```

2. **Snapshot (`:7897`).** When `aux_runtime is not None`, call `WindowSnapshot(out_dir).snapshot(_seg_id, _win_snapshot_windows, cv1_type=..., cv2_type=..., kernel_identity=..., state_definition=aux_runtime.state_definition, phase_kind=aux_runtime.phase_kind, equilibrium_analysis_eligible=aux_runtime.equilibrium_eligible)`. Stage B provides `phase_kind` and `equilibrium_eligible`; the pilot uses `"production"` plus an explicit eligibility. Otherwise the call is unchanged.

3. **Writers (`:7899-7906`).** Add `aux_schema=aux_runtime.sample_schema if aux_runtime is not None else None` to `ParquetSampleWriter(...)`, and `event_schema=EXCHANGE_EVENT_SCHEMA if aux_runtime is not None else None` to `ParquetExchangeWriter(...)`. Import `EXCHANGE_EVENT_SCHEMA` from `.auxiliary_cv.ledger`. Create `_exchange_seq = ExchangeEventCounter()`.

4. **Sample write (`:8225`).** Add `**({"aux_z": aux_obs[r].z, "torsions": aux_obs[r].torsions} if aux_runtime is not None else {})` to the `write_sample(...)` keywords.

5. **Swap event (`:8427`).** When `aux_runtime is not None`, replace the `write_exchange(...)` call with:

```python
                parquet_exchange_writer.write_event(
                    step=int(absolute_step), attempt_seq=_exchange_seq.next(int(absolute_step)),
                    selected_replica=int(i), replica_i=int(i), replica_j=int(j), window_i=int(wi), window_j=int(wj),
                    kind="swap", delta_e_kj=float(outcome.delta_kj), accepted=bool(outcome.accepted),
                    log_q_forward=_aux_logq_f, log_q_reverse=_aux_logq_r, p_accept=_aux_pacc,
                    energy_version=aux_runtime.energy_version, assignments_after=list(assignments))
```

   `_aux_logq_f`, `_aux_logq_r` and `_aux_pacc` are new keyword parameters of `_attempt_window_swap` (defaults `float("nan")`). The Gibbs branch passes `math.log(prop.q_forward)`, `math.log(prop.q_reverse)` (or `-inf`) and `prop.pacc`; the neighbour branches leave them NaN. `selected_replica` is the Gibbs-selected `rep` there. Pass it as a further keyword `_aux_selected` (default `i`).

6. **Gibbs stays and no-candidates (`:8600-8606`).** When `aux_runtime is not None`, before each `continue` write `parquet_exchange_writer.write_event(**event_from_gibbs(prop, step=int(absolute_step), seq=_exchange_seq.next(int(absolute_step)), selected_replica=rep, replica_i=rep, replica_j=rep, accepted=False, energy_version=aux_runtime.energy_version, assignments_after=list(assignments)))`.

7. **Checkpoint save.** At every `save_production_checkpoint(...)` call, add:

```python
aux_block=(aux_checkpoint_block(state_definition=aux_runtime.state_definition, force_info=aux_runtime.force_info,
                                assignments=assignments,
                                observed_params=[read_aux_parameters(s.context, aux_runtime.force_info) for s in sims])
           if aux_runtime is not None else None)
```

   Read the parameters on each replica's own worker (`_sim_pool.submit(r, read_aux_parameters, sims[r].context, info).result()`). This follows the existing Context-affinity rule.

8. **Checkpoint load.** At the `load_production_checkpoint(...)` call, pass:

```python
aux_resume=lambda manifest, a: verify_aux_resume(
    manifest, aux_enabled=aux_runtime is not None,
    state_definition=aux_runtime.state_definition if aux_runtime else None,
    force_info=aux_runtime.force_info if aux_runtime else None, assignments=a,
    observed_params=[read_aux_parameters(s.context, aux_runtime.force_info) for s in sims] if aux_runtime else [])
```

   Always pass it, even when aux is off: it then verifies that the manifest has no `aux` block, which closes the "aux checkpoint resumed by a non-aux run" hole.

- [ ] **Step 5: Extend the source-inspection test**

`tests/test_sample_before_exchange_ordering.py:209` (`test_every_site_that_writes_assignments_is_accounted_for`) enumerates writer call sites by source inspection. Run it. If it fails because the new `write_event` call sites are unaccounted, add them to its expected set. That is the test's purpose: it must name every exchange-record write.

- [ ] **Step 6: Run the tests**

Run: `pytest tests/test_aux_runtime_io.py tests/test_sample_before_exchange_ordering.py tests/test_store.py tests/test_aux_store_writers.py tests/test_aux_checkpoint.py`
Expected: PASS. The plain-run end-to-end check (short real run with aux on, then resume, then `load_parquet`) belongs to Stage B's exit gate, which owns the runtime fixture. Run it after this task once Stage B's harness exists, and record the result in the Task 11 note.

- [ ] **Step 7: Commit**

```bash
git add gareus/auxiliary_cv/runtime_io.py gareus/production.py tests/test_aux_runtime_io.py tests/test_sample_before_exchange_ordering.py
git commit -m "feat(cvaux): wire auxiliary sample, exchange-event, snapshot and checkpoint I/O into production"
```

---

### Task 11: Stage C exit gate and handoff note

**Files:**
- Modify: `CLAUDE.md` (append to the "CVaux Stage A" section written by Stage A Task 7)

- [ ] **Step 1: Run the Stage C set and the legacy regressions**

Run: `pytest tests/test_aux_manifest_payload_schema.py tests/test_aux_sample_schema.py tests/test_aux_store_writers.py tests/test_aux_window_snapshot.py tests/test_aux_ledger.py tests/test_aux_checkpoint.py tests/test_aux_export.py tests/test_aux_loaders.py tests/test_aux_mbar_duplicates.py tests/test_aux_runtime_io.py tests/test_store.py tests/test_parquet_manifest_transactions.py tests/test_query.py tests/test_query_reconstruct_bias_matrix_nan_guard.py tests/test_lambda_ladder_mbar.py tests/test_low_memory_complete.py tests/test_sample_before_exchange_ordering.py`
Expected: all PASS. This is spec Section 16's Stage C exit: no active auxiliary term can disappear through a writer, loader, early return or state-hash path.

- [ ] **Step 2: Append the handoff paragraph to `CLAUDE.md`**

```markdown
- **Stage C (storage/resume/pooling):** samples record `tor_###` (full backbone basis, OpenMM theta, float64) + `aux_z_##` (float64) + `observation_phase` when `aux_schema` given (`atlas-aux-samples-v1`, in Parquet metadata `atlas_aux_samples` and manifest `payload_schema`); exchanges record ordered events (`atlas-exchange-events-v1`: attempt_seq, kind swap/stay/no_candidates, log q fwd/rev, p_accept, energy_version, assignment_sha256_after; `ledger.replay_assignments`); checkpoints carry `manifest["aux"]` (`atlas-aux-checkpoint-v1`, verified on resume incl. Context params); snapshots with `state_definition` are frozen v2 snapshots; strict exporter + `load_parquet` evaluate z from stored torsions (parity vs stored z), `on_incomplete="exclude_and_report"` audits by origin/carrier/time block/z range; featureless segments fail closed (opt-out `exclude_segments_without_aux_features`); adaptive union loaders and legacy NPZ export refuse aux snapshots (MVP). Off path byte-identical (pinned tests). Tests `tests/test_aux_*.py`.
```

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs(cvaux): Stage C handoff note"
```

---

## Self-review record

- **Spec coverage, Stage C** (spec Section 16: "complete thermodynamic observations, ordered event identities, model-aware manifests and restart binding; interrupted/restarted trajectories vs uninterrupted controls; strict MBAR pooling and legacy compatibility; no active auxiliary term can disappear through a writer, loader, early return or state-hash path"):

  | Requirement | Task |
  |---|---|
  | Observations | 2, 3 |
  | Ordered events | 4, 5 |
  | Model-aware manifests | 1, 3, 7 (frozen v2 snapshot) |
  | Restart binding | 6 |
  | Interrupted / restarted behaviour | 5: crash after accepted swap, crash during flush, replay to checkpoint |
  | Strict pooling | 7, 8 |
  | Duplicate Hamiltonians | 9 |
  | Legacy compatibility | 1, 3, 4, 7 (pinned bytes) and the regression files in 11 |
  | Early return | Stage A Task 6, with the λ = 0 table in Task 7 |

- **Section 17 rows covered here:**
  - Ordinary-origin samples (Task 7);
  - Lambda-zero early return through the exporter (Task 7);
  - Duplicate Hamiltonians, separate and merged (Task 9);
  - Missing feature or changed model fails closed (Tasks 6, 7, 8);
  - Multiple swaps at one timestamp, as ledger order (Task 5; the episode-parser side is Stage D).
- **Section 9 crash cases:**
  - before an accepted swap: a checkpoint followed by rejections, replay;
  - after an accepted swap: the step-300 phantom swap;
  - during flush: an orphan tmp chunk.
  - "During an accepted swap" in the transactional sense (half-applied `set_window`) is a runtime property: Stage B's swap transaction. The ledger can only show the swap committed or not. Task 6's Context-parameter verification catches a half-applied swap on resume.
- **Ambiguities resolved:**
  1. **Torsions are stored as OpenMM θ (radians, float64), not as sin/cos features.** The angles are lossless for the feature basis and half the width.
  2. **Pooling scope:** MVP pools only through the fixed-state exporter and `load_parquet`. The adaptive union loaders refuse auxiliary snapshots. This follows spec Section 8 ("default pilot union contains only complete, eligible pilot phases") and Section 14 (adaptive integration comes after confirmation).
  3. **The exclusion report's structural-group breakdown is reported as unavailable until Stage D's frozen labels exist.** It is never synthesised.
  4. **Assignment checksums are written after every ledger event**, not only at checkpoints. This is stronger than the spec and makes replay self-verifying.
  5. **`run_id` in the observation key is the run directory**, implicit at the loader level. It is not stored per row, matching the existing samples layout.
- **Deliberately absent:**
  - the structural-label boundary stream (Stage D);
  - z-evaluation parity with the runtime force (Stage B);
  - multi-model schemas (Stage F; refused here).
