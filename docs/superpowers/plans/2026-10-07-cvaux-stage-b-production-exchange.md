# CVaux Stage B: Production Hamiltonians and Unrestricted Exchange — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the exact Stage A auxiliary restraint into every production Context of a plain (non-adaptive) run, so that dynamics, every state assignment, the sample-path and exchange-path bias matrices, and NPT trial energies all use the same complete Hamiltonian, while keeping unrestricted Gibbs proposals with their MH correction.

**Architecture:**
- One new runtime module, `gareus/auxiliary_cv/runtime.py`, holds four things:
  - a force-group audit and allocator;
  - `add_aux_cv_force`, which builds the Stage A force into a System before any integrator exists;
  - `observe_aux_z`, which reads z through the fast or slow path;
  - `aux_bias_matrix_kcal`, which builds the [state, replica] auxiliary bias.
- A second new module, `gareus/auxiliary_cv/state_table.py`, turns the explicit window CSV's aux and instance columns into a frozen `AuxStateTable`.
- In `production.py` / `windows.py`:
  - `set_window` gains `aux_state=`;
  - `assemble_bias_matrices` gains `aux_bias_kcal=`;
  - the two `run_gareus` fetch closures share one z helper.
- The exchange kernel itself (`_gibbs_window_proposal_distribution`, `gibbs_propose_one_replica`, `apply_window_swap`) is **not** changed. It already consumes only the complete matrix.
- Storage (Parquet z/feature columns), checkpoint/resume binding and pooled analysis are Stage C. Until Stage C lands, an aux run must be acknowledged as an engineering run (`--aux-cv-allow-unpersisted`) and cannot resume.

**Tech Stack:** Python 3, NumPy, OpenMM 8.5 (Reference/CPU in tests), pytest.

**Spec:** `docs/superpowers/specs/2026-10-07-auxiliary-cv-gibbs-production-spec.md` (main dc30285). Read Sections 3.3, 3.4, 4 (all), 5 (last paragraph), 10, 15 and 16 Stage B, plus the Section 17 rows listed in the self-review.
**Depends on:** `docs/superpowers/plans/2026-10-07-cvaux-stage-a-definitions-evaluators.md`, which must be merged first. This plan consumes these Stage A names exactly:
- `AuxModel` (`.load`, `.model_sha256`, `.offset`);
- `build_aux_force(openmm, model, *, force_group) -> (force, AuxForceInfo)` and `AuxForceInfo(name, force_group, model_sha256, global_k, global_c, sub_cv_names)`;
- `set_aux_parameters(context, info, *, center, k_kcal)` and `AUX_FORCE_NAME`;
- `z_from_positions(xyz_nm, model)`, `aux_forces_kj_nm`, `check_feature_atoms(model, topology)`;
- `gareus.correctness.bias._aux_term`, `gareus.correctness.state_identity._canonical_instance`;
- `tests/aux_cv_fixture.py` (`model_payload(..., blocks=)`, `dipeptide()`).

**Test runner (user policy):** tests in this repo are run by the local free runner `opencode`, never directly. Each "Run:" step below names the pytest arguments. Dispatch them as
`opencode run "In /run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler: run python -m pytest -q <arguments> and report pass/fail/error counts and every failing test id. Read-only: do not edit, commit or fix anything."`
and treat its report as the result. A Bash hook in this harness refuses commands that call the test runner directly. **Targeted tests only:** never the whole suite.

## Global Constraints

- **Off path byte-identical.** With `--aux-cv-model` unset, nothing changes:
  - no new force;
  - the same `bias_matrix_kj` arithmetic (`assemble_bias_matrices` keeps its exact old expression when `aux_bias_kcal is None`);
  - the same `EXCHANGE_ENERGY_VERSION` `state_bias_matrix_v2`;
  - the same kernel-identity digest;
  - the same `method_settings`.
- Energies are kJ/mol in OpenMM and in the exchange kernel; state tables use kcal/mol. Convert **once** by 4.184.
- **Inactive restraint** (`aux_k == 0`): exactly zero energy and force, and the centre global is reset to 0.0 (Stage A `set_aux_parameters`).
- **Every carrier Context carries the auxiliary force capability.** State assignment activates or deactivates it; a Context is never rebuilt on entry to an auxiliary state (spec 3.3).
- **No fixed force group.** The group is allocated by auditing the System's existing groups, excluding 0, 1 and 2 (Pep-GaMD physical/aux), `umbrella_force_group` and `secondary_cv_force_group`. It is recorded in the runtime info.
- **The auxiliary force enters the integrator once, at full strength, outside the boost channels.** Stock gamd-openmm boost types are refused, because their factory calls `set_all_forces_to_group(system, 0)` (`gamd/integrator_factory.py:198`, verified 2026-10-07), which would put the restraint inside the boosted group.
- **A missing or nonfinite z needed by an active state in the live exchange matrix is fatal** (`AuxObservationError`). It is never zero-filled and never removes a candidate.
- **Gibbs eligibility is unrestricted.** No parent-only mask, neighbour mask or overlap filter. `--exchange-mode neighbor` (geometry graph) is refused with an aux model; `gibbs-walk`, `all-pair-sweep` and `random-pair` are allowed.
- **The state population is frozen for the run.** Window auto-drop and the seed-reachability filter must not renumber an aux state table; either one dropping a window is fatal.
- **Version gating.** Kernel version `state_bias_matrix_v3_aux` only when an aux model is configured, even if every `aux_k` is 0 (the sham arm). Legacy runs keep v2.
- Float64 for every auxiliary value.

## Review Focus

1. **Active → ordinary → active on one Context.** After leaving an auxiliary state, the previous centre and strength must not linger. Covered by the Task 4 test.
2. **Duplicate Hamiltonians (sham = parent).** Identical bias rows. The Gibbs kernel must stay detailed-balanced and both instances must remain separate slots. Covered by the Task 7 enumeration with a duplicate row.
3. **A carrier in an ordinary state still contributes its z to active-state cross-energies.** The z of every replica is observed whether or not its current state feels the force. Covered by the Task 6 test (z observed for an ordinary-state replica) and the Task 7 direct cross-energy test.
4. **Slow path vs fast path disagreeing.** z read from the force's cached sub-CVs and z from positions must match. Covered by the Task 6 parity test.
5. **Window renumbering after load.** The seed-reachability filter or US auto-drop would desynchronise the aux table from window indices. Covered by the Task 2 CLI refusal plus the Task 8 test of the post-filter count guard.

---

## File Structure

| File | Responsibility |
|---|---|
| Create `gareus/auxiliary_cv/state_table.py` | `AuxStateTable`; parse aux and instance CSV columns; load and validate the table against the model |
| Create `gareus/auxiliary_cv/runtime.py` | Force-group audit and allocator, `AuxRuntime`, `add_aux_cv_force`, `observe_aux_z`, `aux_bias_matrix_kcal`, `AuxObservationError` |
| Modify `gareus/cli.py` | `--aux-cv-model`, `--aux-cv-allow-unpersisted`, `_validate_aux_cv_args` |
| Modify `gareus/windows.py` | `set_window(..., aux_state=None)`; `load_explicit_2d_window_csv` collects aux and instance columns into `window_metadata["aux_rows"]` |
| Modify `gareus/production.py` | `assemble_bias_matrices(..., aux_bias_kcal=None)`; aux force on base and starting-structure systems; `aux_state=` at all five `set_window` calls; shared z helper in both fetch closures; aux matrix in sample and exchange assembly; population guards; Stage B warning |
| Modify `gareus/kernel_identity.py`, `gareus/provenance.py` | `exchange_energy_version_for_args`, `state_bias_matrix_v3_aux`, aux model sha in the identity and method settings |
| Create `tests/test_aux_cv_cli.py`, `test_aux_cv_state_table.py`, `test_aux_cv_runtime.py`, `test_aux_cv_composition.py`, `test_aux_cv_set_window.py`, `test_aux_cv_observation.py`, `test_aux_cv_exchange.py`, `test_aux_cv_kernel_identity.py`, `test_aux_cv_production_wiring.py` | One test file per task |

---

### Task 1: CLI flags and refusals

**Files:**
- Modify: `gareus/cli.py` (new `_add_aux_cv_args`, `_validate_aux_cv_args`; register in `parse_args` at `gareus/cli.py:2099-2150`)
- Test: `tests/test_aux_cv_cli.py`

**Interfaces:**
- Produces: `args.aux_cv_model: str | None` and `args.aux_cv_allow_unpersisted: bool`. The YAML keys `aux_cv_model` / `aux_cv_allow_unpersisted` come for free through the generic dest mapping.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_aux_cv_cli.py
import pytest

from gareus.cli import parse_args

BASE = ["--seq", "GA", "--cv1", "contacts"]
OK = ["--aux-cv-model", "m.json", "--windows-2d-csv", "w.csv", "--run-mode", "cmd",
      "--exchange-mode", "gibbs-walk", "--aux-cv-allow-unpersisted"]


def _parse(extra, tmp_path):
    return parse_args(BASE + ["--out", str(tmp_path / "o")] + extra)


def test_off_by_default(tmp_path):
    a = _parse([], tmp_path)
    assert a.aux_cv_model is None and a.aux_cv_allow_unpersisted is False


def test_accepted_configuration(tmp_path):
    a = _parse(OK, tmp_path)
    assert a.aux_cv_model == "m.json"


def test_pep_gamd_boost_is_accepted(tmp_path):
    extra = [x for x in OK if x not in ("--run-mode", "cmd")] + ["--run-mode", "gamd",
                                                                 "--gamd-boost-type", "pep-gamd-lower-dual"]
    assert _parse(extra, tmp_path).aux_cv_model == "m.json"


@pytest.mark.parametrize("change, message", [
    (lambda o: [x for x in o if x not in ("--windows-2d-csv", "w.csv")], "windows-2d-csv"),
    (lambda o: [x for x in o if x not in ("--run-mode", "cmd")] + ["--run-mode", "gamd",
                                                                   "--gamd-boost-type", "lower-dihedral"], "group 0"),
    (lambda o: [x for x in o if x not in ("--exchange-mode", "gibbs-walk")] + ["--exchange-mode", "neighbor"], "neighbor"),
    (lambda o: o + ["--resume"], "Stage C"),
    (lambda o: o + ["--extend"], "Stage C"),
    (lambda o: o + ["--us-auto-drop-bad-windows"], "auto-drop"),
    (lambda o: [x for x in o if x != "--aux-cv-allow-unpersisted"], "aux-cv-allow-unpersisted"),
])
def test_refusals(change, message, tmp_path, capsys):
    with pytest.raises(SystemExit):
        _parse(change(list(OK)), tmp_path)
    assert message in capsys.readouterr().err


def test_ack_without_model_is_refused(tmp_path, capsys):
    with pytest.raises(SystemExit):
        _parse(["--aux-cv-allow-unpersisted"], tmp_path)
    assert "needs --aux-cv-model" in capsys.readouterr().err
```

- [ ] **Step 2: Run to verify failure**

Run: `tests/test_aux_cv_cli.py`
Expected: FAIL (`unrecognized arguments: --aux-cv-model`).

- [ ] **Step 3: Implement**

Add to `gareus/cli.py` (next to `_validate_gamd_args`):

```python
def _add_aux_cv_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--aux-cv-model", default=None, metavar="PATH",
                   help="Frozen auxiliary-CV model (atlas-aux-cv-model-v1 JSON). Enables auxiliary-CV "
                        "states: every row of --windows-2d-csv must state aux_k_kcal_mol (0 = ordinary or "
                        "sham) and active rows aux_center. Plain runs only. Spec "
                        "docs/superpowers/specs/2026-10-07-auxiliary-cv-gibbs-production-spec.md.")
    p.add_argument("--aux-cv-allow-unpersisted", action="store_true", default=False,
                   help="Acknowledge that this version does not yet write auxiliary z values to the "
                        "sample store: the run is an engineering run whose samples cannot be pooled "
                        "(Stage C removes this flag).")


def _validate_aux_cv_args(p: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Refuse every configuration the Stage B auxiliary-state machinery cannot run exactly."""
    if not getattr(args, "aux_cv_model", None):
        if getattr(args, "aux_cv_allow_unpersisted", False):
            p.error("--aux-cv-allow-unpersisted needs --aux-cv-model")
        return
    if not getattr(args, "windows_2d_csv", None):
        p.error("--aux-cv-model needs --windows-2d-csv: auxiliary states are rows of an explicit state table")
    run_mode = str(getattr(args, "run_mode", "gamd") or "gamd")
    boost = str(getattr(args, "gamd_boost_type", "") or "")
    if run_mode in ("gamd", "hmr-gamd") and not boost.startswith("pep-gamd"):
        p.error(f"--aux-cv-model with --gamd-boost-type {boost!r}: stock gamd-openmm integrators move every "
                "force to group 0 (gamd/integrator_factory.py set_all_forces_to_group), which would boost the "
                "auxiliary restraint; use a pep-gamd-* boost type or --run-mode cmd")
    if str(getattr(args, "exchange_mode", "neighbor")) == "neighbor":
        p.error("--aux-cv-model needs unrestricted exchange candidates (gibbs-walk, all-pair-sweep or "
                "random-pair): --exchange-mode neighbor builds its graph from CV1/CV2 geometry and cannot "
                "represent auxiliary states")
    if getattr(args, "resume", False) or getattr(args, "extend", False):
        p.error("--aux-cv-model cannot --resume/--extend yet: checkpoint binding of auxiliary states is Stage C")
    if bool(getattr(args, "us_auto_drop_bad_windows", False)):
        p.error("--aux-cv-model refuses --us-auto-drop-bad-windows: the state table is frozen and a population "
                "change is a new phase (spec Section 6)")
    if str(getattr(args, "window_mode", "")) == "adaptive-production":
        p.error("--aux-cv-model is plain-run only; adaptive admission is a later stage")
    if not getattr(args, "aux_cv_allow_unpersisted", False):
        p.error("--aux-cv-model does not yet write auxiliary z to the sample store (Stage C); pass "
                "--aux-cv-allow-unpersisted to acknowledge an engineering run whose samples cannot be pooled")
```

In `parse_args`, call `_add_aux_cv_args(p)` right after `_add_window_args(p)`, and `_validate_aux_cv_args(p, args)` right after `_validate_cv_selection_args(p, args)`.

- [ ] **Step 4: Run to verify pass**

Run: `tests/test_aux_cv_cli.py tests/test_gamd_boost_default.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add gareus/cli.py tests/test_aux_cv_cli.py
git commit -m "feat(cvaux): --aux-cv-model flags and Stage B refusals"
```

---

### Task 2: Auxiliary state table from the explicit window CSV

**Files:**
- Create: `gareus/auxiliary_cv/state_table.py`
- Modify: `gareus/windows.py` (`load_explicit_2d_window_csv`, currently `gareus/windows.py:1242-1473`)
- Test: `tests/test_aux_cv_state_table.py`

**Interfaces:**
- Consumes: `AuxModel.load`; `gareus.correctness.bias._aux_term(window, label) -> (sha|None, center, k)`; `gareus.correctness.state_identity._canonical_instance(raw, state_id)` (Stage A Task 5).
- Produces:
  - `AUX_CSV_COLUMNS = ("aux_center", "aux_k_kcal_mol", "aux_model_sha256")` and `INSTANCE_CSV_COLUMNS = ("state_instance_id", "state_role", "spawn_parent_state_id", "matched_additional_slot_id", "spawn_source_observation_json")`;
  - `@dataclass(frozen=True) class AuxStateTable(model: AuxModel, centers: tuple[float, ...], k_kcal: tuple[float, ...], instances: tuple[dict | None, ...])` with property `n` and method `window_rows() -> list[dict]` (the v2 schema fields per window);
  - `parse_aux_csv_row(row: Mapping[str, str], offset: int, model_sha256: str) -> dict`;
  - `load_aux_state_table(model_path, aux_rows: list[dict]) -> AuxStateTable`;
  - the loader writes `window_metadata["aux_rows"]`: one dict per window holding the raw strings of every aux/instance column, or no key when the table has no aux columns. `normalized_rows` gains those columns only when present.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_aux_cv_state_table.py
import json

import pytest

from aux_cv_fixture import model_payload
from gareus.auxiliary_cv.model import AuxModel
from gareus.auxiliary_cv.state_table import load_aux_state_table, parse_aux_csv_row
from gareus.correctness._io import IntegrityError

QUADS = [(0, 1, 2, 3)]


def _model(tmp_path):
    m = AuxModel.from_mapping(model_payload(QUADS, [1.0, 0.5], offset=0.1))
    path = tmp_path / "aux.json"
    m.write(path)
    return m, path


def _csv(tmp_path, rows, header):
    path = tmp_path / "w.csv"
    path.write_text("\n".join([",".join(header)] + [",".join(map(str, r)) for r in rows]) + "\n")
    return path


def _args(tmp_path, model_path):
    from gareus.cli import parse_args
    a = parse_args(["--seq", "GA", "--out", str(tmp_path / "o"), "--cv1", "contacts"])
    a.aux_cv_model = str(model_path)
    return a


HEADER = ["primary_cv_center", "primary_cv_k_kcal", "aux_center", "aux_k_kcal_mol", "state_instance_id",
          "state_role", "spawn_parent_state_id", "matched_additional_slot_id"]
ROWS = [[0.2, 25.0, "", 0.0, "ord-0", "ordinary", "", ""],
        [0.4, 25.0, "", 0.0, "ord-1", "ordinary", "", ""],
        [0.4, 25.0, 1.5, 1.2, "aux-0", "auxiliary", 1, "slot-0"],
        [0.4, 25.0, "", 0.0, "sham-0", "sham", 1, "slot-0"]]


def test_loader_collects_aux_rows_and_table_parses(tmp_path):
    from gareus.windows import load_explicit_2d_window_csv
    m, mpath = _model(tmp_path)
    *_rest, wmeta = load_explicit_2d_window_csv(_args(tmp_path, mpath), _csv(tmp_path, ROWS, HEADER))
    table = load_aux_state_table(mpath, wmeta["aux_rows"])
    assert table.n == 4
    assert table.k_kcal == (0.0, 0.0, 1.2, 0.0) and table.centers == (0.0, 0.0, 1.5, 0.0)
    assert [i["state_instance_id"] for i in table.instances] == ["ord-0", "ord-1", "aux-0", "sham-0"]
    rows = table.window_rows()
    assert rows[2]["aux_model_sha256"] == m.model_sha256 and rows[0]["aux_model_sha256"] is None


def test_legacy_table_has_no_aux_rows(tmp_path):
    from gareus.cli import parse_args
    from gareus.windows import load_explicit_2d_window_csv
    a = parse_args(["--seq", "GA", "--out", str(tmp_path / "o"), "--cv1", "contacts"])
    *_rest, wmeta = load_explicit_2d_window_csv(a, _csv(tmp_path, [[0.2, 25.0], [0.4, 25.0]],
                                                         ["primary_cv_center", "primary_cv_k_kcal"]))
    assert "aux_rows" not in wmeta
    assert "aux_k_kcal_mol" not in wmeta["normalized_rows"][0]


def test_aux_columns_without_model_flag_are_refused(tmp_path):
    from gareus.cli import parse_args
    from gareus.windows import load_explicit_2d_window_csv
    a = parse_args(["--seq", "GA", "--out", str(tmp_path / "o"), "--cv1", "contacts"])
    with pytest.raises(ValueError, match="--aux-cv-model"):
        load_explicit_2d_window_csv(a, _csv(tmp_path, ROWS, HEADER))


def test_model_flag_without_aux_columns_is_refused(tmp_path):
    from gareus.windows import load_explicit_2d_window_csv
    _m, mpath = _model(tmp_path)
    with pytest.raises(ValueError, match="aux_k_kcal_mol"):
        load_explicit_2d_window_csv(_args(tmp_path, mpath),
                                    _csv(tmp_path, [[0.2, 25.0]], ["primary_cv_center", "primary_cv_k_kcal"]))


@pytest.mark.parametrize("row, message", [
    ({"aux_k_kcal_mol": ""}, "aux_k_kcal_mol"),
    ({"aux_k_kcal_mol": "-1"}, "aux_k"),
    ({"aux_k_kcal_mol": "1.0"}, "aux_center"),
    ({"aux_k_kcal_mol": "1.0", "aux_center": "0.2", "aux_model_sha256": "a" * 64}, "model"),
    ({"aux_k_kcal_mol": "1.0", "aux_center": "0.2", "state_instance_id": "x", "state_role": "sham",
      "spawn_parent_state_id": "", "matched_additional_slot_id": "", "spawn_source_observation_json": ""}, "role"),
    ({"aux_k_kcal_mol": "0", "state_instance_id": "x", "state_role": "auxiliary",
      "spawn_parent_state_id": "", "matched_additional_slot_id": "", "spawn_source_observation_json": ""}, "role"),
])
def test_row_refusals(row, message, tmp_path):
    m, _ = _model(tmp_path)
    with pytest.raises((IntegrityError, ValueError), match=message):
        parse_aux_csv_row(row, 2, m.model_sha256)


def test_spawn_source_observation_json_round_trips(tmp_path):
    m, _ = _model(tmp_path)
    obs = {"run": "r", "segment": "s", "carrier": 3, "state": 1, "checkpoint": "c", "step": 100}
    out = parse_aux_csv_row({"aux_k_kcal_mol": "1", "aux_center": "0.5", "state_instance_id": "aux-0",
                             "state_role": "auxiliary", "spawn_parent_state_id": "1",
                             "matched_additional_slot_id": "slot-0",
                             "spawn_source_observation_json": json.dumps(obs)}, 2, m.model_sha256)
    assert out["instance"]["spawn_source_observation"] == obs


def test_table_refuses_partial_instances_and_duplicate_ids(tmp_path):
    m, mpath = _model(tmp_path)
    a = {"aux_k_kcal_mol": "0", "state_instance_id": "x", "state_role": "ordinary",
         "spawn_parent_state_id": "", "matched_additional_slot_id": "", "spawn_source_observation_json": ""}
    with pytest.raises(IntegrityError, match="duplicate"):
        load_aux_state_table(mpath, [a, dict(a)])
    with pytest.raises(IntegrityError, match="every row"):
        load_aux_state_table(mpath, [a, {"aux_k_kcal_mol": "0"}])
```

- [ ] **Step 2: Run to verify failure**

Run: `tests/test_aux_cv_state_table.py`
Expected: FAIL (`ModuleNotFoundError: gareus.auxiliary_cv.state_table`).

- [ ] **Step 3: Implement `gareus/auxiliary_cv/state_table.py`**

```python
"""Auxiliary-state table of a plain run: one frozen (centre, k, instance) per window.

Parsed from the explicit window CSV (``--windows-2d-csv``) columns ``aux_center``,
``aux_k_kcal_mol`` (explicit on every row; 0 = ordinary or sham), optional ``aux_model_sha256``
(must equal the loaded model) and the optional instance columns. Canonicalisation reuses
the Stage A helpers so the runtime table and the v2 state schema can never disagree.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..correctness._io import IntegrityError
from ..correctness.bias import _aux_term
from ..correctness.state_identity import _canonical_instance
from .model import AuxModel

AUX_CSV_COLUMNS = ("aux_center", "aux_k_kcal_mol", "aux_model_sha256")
INSTANCE_CSV_COLUMNS = ("state_instance_id", "state_role", "spawn_parent_state_id",
                        "matched_additional_slot_id", "spawn_source_observation_json")


def _cell(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    return "" if value is None else str(value).strip()


def parse_aux_csv_row(row: Mapping[str, Any], offset: int, model_sha256: str) -> dict:
    label = f"window table row {offset}"
    k_raw = _cell(row, "aux_k_kcal_mol")
    if not k_raw:
        raise IntegrityError(f"{label}: aux_k_kcal_mol is required on every row (0 = ordinary or sham state)")
    raw: dict[str, Any] = {"aux_k": float(k_raw)}
    if _cell(row, "aux_center"):
        raw["aux_center"] = float(_cell(row, "aux_center"))
    sha = _cell(row, "aux_model_sha256")
    if sha and sha != model_sha256:
        raise IntegrityError(f"{label}: aux_model_sha256 {sha} is not the loaded model {model_sha256}")
    raw["aux_model_sha256"] = sha or model_sha256
    model, center, k = _aux_term(raw, label)
    instance = None
    if any(_cell(row, key) for key in INSTANCE_CSV_COLUMNS):
        parent = _cell(row, "spawn_parent_state_id")
        obs = _cell(row, "spawn_source_observation_json")
        instance = _canonical_instance({
            "state_instance_id": _cell(row, "state_instance_id"),
            "state_role": _cell(row, "state_role"),
            "spawn_parent_state_id": int(parent) if parent else None,
            "spawn_source_observation": json.loads(obs) if obs else None,
            "matched_additional_slot_id": _cell(row, "matched_additional_slot_id") or None,
        }, offset)
        if (instance["state_role"] == "auxiliary") != (k > 0):
            raise IntegrityError(f"{label}: state_role {instance['state_role']!r} contradicts aux_k {k} "
                                 "(auxiliary iff aux_k > 0)")
    return {"aux_model_sha256": model, "aux_center": center, "aux_k": k, "instance": instance}


@dataclass(frozen=True)
class AuxStateTable:
    model: AuxModel
    centers: tuple[float, ...]
    k_kcal: tuple[float, ...]
    instances: tuple[dict | None, ...]

    @property
    def n(self) -> int:
        return len(self.k_kcal)

    def window_rows(self) -> list[dict]:
        rows = []
        for w in range(self.n):
            active = self.k_kcal[w] > 0
            row = {"aux_model_sha256": self.model.model_sha256 if active else None,
                   "aux_center": self.centers[w], "aux_k": self.k_kcal[w]}
            if self.instances[w] is not None:
                row["instance"] = dict(self.instances[w])
            rows.append(row)
        return rows


def load_aux_state_table(model_path: Path | str, aux_rows: list[Mapping[str, Any]]) -> AuxStateTable:
    model = AuxModel.load(model_path)
    if not aux_rows:
        raise IntegrityError("--aux-cv-model needs a window table with aux_k_kcal_mol on every row")
    parsed = [parse_aux_csv_row(row, offset, model.model_sha256)
              for offset, row in enumerate(aux_rows, start=2)]
    with_instance = [p for p in parsed if p["instance"] is not None]
    if with_instance and len(with_instance) != len(parsed):
        raise IntegrityError("instance columns must be filled on every row or on none")
    ids = [p["instance"]["state_instance_id"] for p in with_instance]
    if len(set(ids)) != len(ids):
        raise IntegrityError(f"duplicate state_instance_id in the window table: {sorted(ids)}")
    return AuxStateTable(model, tuple(p["aux_center"] for p in parsed), tuple(p["aux_k"] for p in parsed),
                         tuple(p["instance"] for p in parsed))
```

- [ ] **Step 4: Collect the columns in `load_explicit_2d_window_csv`**

In `gareus/windows.py`, before the row loop (next to `type_counts: dict[str, int] = {}`), add:

```python
    from .auxiliary_cv.state_table import AUX_CSV_COLUMNS, INSTANCE_CSV_COLUMNS
    _aux_cols = AUX_CSV_COLUMNS + INSTANCE_CSV_COLUMNS
    aux_rows: list[dict] = []
    any_aux = False
    all_aux = True
```

Inside the loop, after `gamd_lambdas.append(lam)`:

```python
        _aux_cells = {c: ("" if row.get(c) is None else str(row.get(c)).strip()) for c in _aux_cols}
        _has_aux = any(_aux_cells.values())
        any_aux = any_aux or _has_aux
        all_aux = all_aux and _has_aux
        aux_rows.append(_aux_cells)
```

After `normalized_rows.append({...})`, add:

```python
        if _has_aux:
            normalized_rows[-1].update({c: v for c, v in _aux_cells.items() if v})
```

After the `if any_secondary and not secondary_cv_enabled(args):` check, add:

```python
    if any_aux and not all_aux:
        raise ValueError("--windows-2d-csv mixes rows with and without auxiliary-CV columns; "
                         "aux_k_kcal_mol must be stated on every row (0 = ordinary or sham state)")
    if any_aux and not getattr(args, "aux_cv_model", None):
        raise ValueError("--windows-2d-csv carries auxiliary-CV columns but no --aux-cv-model was given")
    if getattr(args, "aux_cv_model", None) and not any_aux:
        raise ValueError("--aux-cv-model needs aux_k_kcal_mol (and aux_center for active rows) in --windows-2d-csv")
```

After `window_metadata = {...}` is built, add `if any_aux: window_metadata["aux_rows"] = aux_rows`. Leave the return tuple unchanged: aux travels in `window_metadata` so no caller's 6-tuple unpacking changes.

- [ ] **Step 5: Run to verify pass**

Run: `tests/test_aux_cv_state_table.py tests/test_contact_map_cv1_production.py::test_window_table_kind_must_match_a_contact_map_run`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add gareus/auxiliary_cv/state_table.py gareus/windows.py tests/test_aux_cv_state_table.py
git commit -m "feat(cvaux): auxiliary state table from the explicit window CSV"
```

---

### Task 3: Force-group audit, allocation and `add_aux_cv_force`

**Files:**
- Create: `gareus/auxiliary_cv/runtime.py`
- Test: `tests/test_aux_cv_runtime.py`

**Interfaces:**
- Consumes: `build_aux_force`, `AuxForceInfo`, `AUX_FORCE_NAME` (Stage A); `AuxStateTable` (Task 2).
- Produces:
  - `RESERVED_PHYSICAL_GROUPS = frozenset({0, 1, 2})`;
  - `force_group_audit(system) -> list[dict]` with keys `index, class, name, group`;
  - `allocate_free_force_group(system, reserved=()) -> int`;
  - `@dataclass(frozen=True) class AuxRuntime(table: AuxStateTable, info: AuxForceInfo, force_index: int)`;
  - `add_aux_cv_force(openmm, system, table, args, *, force_group: int | None = None) -> AuxRuntime`;
  - `class AuxObservationError(IntegrityError)`.

  Tasks 5 and 6 add `observe_aux_z` and `aux_bias_matrix_kcal` to this module.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_aux_cv_runtime.py
import types

import numpy as np
import pytest

from aux_cv_fixture import dipeptide, model_payload
from gareus.auxiliary_cv.force import AUX_FORCE_NAME
from gareus.auxiliary_cv.model import AuxModel
from gareus.auxiliary_cv.runtime import (RESERVED_PHYSICAL_GROUPS, add_aux_cv_force,
                                         allocate_free_force_group, force_group_audit)
from gareus.auxiliary_cv.state_table import AuxStateTable
from gareus.correctness._io import IntegrityError

ARGS = types.SimpleNamespace(umbrella_force_group=31, secondary_cv_force_group=29)


def _table():
    d = dipeptide()
    blocks = [lab.split("-")[0] for lab in d["labels"]]
    m = AuxModel.from_mapping(model_payload(d["quads"], [1.0] + [0.3] * (2 * len(d["quads"]) - 1),
                                            offset=0.2, blocks=blocks))
    return AuxStateTable(m, (0.0, 1.0), (0.0, 2.0), (None, None))


def _bare_system(groups):
    import openmm as mm
    s = mm.System()
    for _ in range(4):
        s.addParticle(1.0)
    for g in groups:
        f = mm.CustomExternalForce("0")
        f.setForceGroup(g)
        s.addForce(f)
    return s


def test_allocator_skips_used_and_reserved_groups():
    s = _bare_system([0, 3, 4])
    assert allocate_free_force_group(s, reserved=(5,)) == 6
    assert not ({allocate_free_force_group(s)} & RESERVED_PHYSICAL_GROUPS)


def test_allocator_raises_when_full():
    with pytest.raises(IntegrityError, match="free force group"):
        allocate_free_force_group(_bare_system(range(32)))


def test_audit_lists_every_force():
    s = _bare_system([0, 7])
    audit = force_group_audit(s)
    assert [a["group"] for a in audit] == [0, 7] and all(a["class"] == "CustomExternalForce" for a in audit)


def test_add_aux_cv_force_allocates_and_records():
    from pep_gamd_fixture import _fresh_system
    import openmm as mm
    system = _fresh_system()
    before = {a["group"] for a in force_group_audit(system)}
    rt = add_aux_cv_force(mm, system, _table(), ARGS)
    assert system.getForce(rt.force_index).getName() == AUX_FORCE_NAME
    assert rt.info.force_group not in before | RESERVED_PHYSICAL_GROUPS | {29, 31}


def test_explicit_group_must_be_free():
    from pep_gamd_fixture import _fresh_system
    import openmm as mm
    system = _fresh_system()
    used = force_group_audit(system)[0]["group"]
    with pytest.raises(IntegrityError, match="force group"):
        add_aux_cv_force(mm, system, _table(), ARGS, force_group=used)


def test_same_audit_gives_same_group_on_a_copy():
    """Starting-structure systems must receive the base system's group (production passes it explicitly)."""
    from pep_gamd_fixture import _fresh_system
    import openmm as mm
    a, b = _fresh_system(), _fresh_system()
    assert add_aux_cv_force(mm, a, _table(), ARGS).info.force_group == \
        add_aux_cv_force(mm, b, _table(), ARGS).info.force_group
```

- [ ] **Step 2: Run to verify failure**

Run: `tests/test_aux_cv_runtime.py`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement `gareus/auxiliary_cv/runtime.py`**

```python
"""Production runtime of auxiliary-CV states: force placement, observation and bias matrices.

The auxiliary restraint must enter the integrator once, at full strength, outside every boost
channel. Its force group is therefore allocated from an audit of the System, never assumed:
groups 0/1/2 belong to Pep-GaMD's physical and auxiliary-nonbonded channels, and the two
umbrella groups are reserved even when the shared layout leaves one empty.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from ..correctness._io import IntegrityError
from .force import AuxForceInfo, build_aux_force
from .state_table import AuxStateTable

RESERVED_PHYSICAL_GROUPS = frozenset({0, 1, 2})


class AuxObservationError(IntegrityError):
    """A live auxiliary z needed by an active state is missing or non-finite."""


def force_group_audit(system) -> list[dict]:
    out = []
    for i in range(system.getNumForces()):
        f = system.getForce(i)
        out.append({"index": i, "class": f.__class__.__name__, "name": f.getName(),
                    "group": int(f.getForceGroup())})
    return out


def allocate_free_force_group(system, reserved: Iterable[int] = ()) -> int:
    blocked = ({a["group"] for a in force_group_audit(system)} | set(RESERVED_PHYSICAL_GROUPS)
               | {int(g) for g in reserved})
    for group in range(3, 32):
        if group not in blocked:
            return group
    raise IntegrityError(f"no free force group for the auxiliary restraint (blocked: {sorted(blocked)})")


@dataclass(frozen=True)
class AuxRuntime:
    table: AuxStateTable
    info: AuxForceInfo
    force_index: int


def _reserved_groups(args) -> set[int]:
    return {int(getattr(args, "umbrella_force_group", 31)), int(getattr(args, "secondary_cv_force_group", 29))}


def add_aux_cv_force(openmm, system, table: AuxStateTable, args, *,
                     force_group: Optional[int] = None) -> AuxRuntime:
    """Add the auxiliary restraint capability to ``system``; call before any integrator is built."""
    reserved = _reserved_groups(args)
    if force_group is None:
        group = allocate_free_force_group(system, reserved)
    else:
        group = int(force_group)
        used = {a["group"] for a in force_group_audit(system)}
        if group in used or group in RESERVED_PHYSICAL_GROUPS or group in reserved:
            raise IntegrityError(f"auxiliary force group {group} is not free in this system "
                                 f"(used {sorted(used)}, reserved {sorted(RESERVED_PHYSICAL_GROUPS | reserved)})")
    force, info = build_aux_force(openmm, table.model, force_group=group)
    index = int(system.addForce(force))
    return AuxRuntime(table, info, index)
```

- [ ] **Step 4: Run to verify pass**

Run: `tests/test_aux_cv_runtime.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add gareus/auxiliary_cv/runtime.py tests/test_aux_cv_runtime.py
git commit -m "feat(cvaux): force-group audit, allocation and add_aux_cv_force"
```

---

### Task 4: `set_window` applies the complete target, and force-composition proofs

**Files:**
- Modify: `gareus/windows.py:193-224` (`set_window`)
- Modify: `gareus/production.py`, the five `set_window(` calls at `:5592` (`_apply_assignment` in `load_production_checkpoint`), `:5891` (`run_multiwindow_gamd_recon`), `:6128` (`apply_joint_envelope_gamd_calibration`), `:7522` (`_build_context_i`) and `:8420` (`_apply_swap_to_replica`)
- Test: `tests/test_aux_cv_set_window.py`, `tests/test_aux_cv_composition.py`

**Interfaces:**
- Consumes: `AuxRuntime` (Task 3); `set_aux_parameters` (Stage A).
- Produces:
  - `set_window(context, centers_nm, ks_kj_nm2, window_index, secondary_centers=None, secondary_ks_kj=None, *, aux_state=None)`. When `aux_state` is an `AuxRuntime`, it sets the window's complete aux target. A missing global raises; there is no try/except.
  - Every production call passes `aux_state=getattr(args, "_aux_runtime", None)`. `load_production_checkpoint` may receive `args=None`, so it uses `None if args is None else getattr(args, "_aux_runtime", None)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_aux_cv_set_window.py
import ast
import inspect
import types

import numpy as np
import pytest

from aux_cv_fixture import dipeptide, model_payload
from gareus.auxiliary_cv.model import AuxModel
from gareus.auxiliary_cv.runtime import add_aux_cv_force
from gareus.auxiliary_cv.state_table import AuxStateTable
from gareus.windows import set_window

ARGS = types.SimpleNamespace(umbrella_force_group=31, secondary_cv_force_group=29)


def _ctx_and_runtime():
    import openmm as mm
    from openmm import unit
    from pep_gamd_fixture import _fresh_system
    d = dipeptide()
    blocks = [lab.split("-")[0] for lab in d["labels"]]
    m = AuxModel.from_mapping(model_payload(d["quads"], [1.0] + [0.4] * (2 * len(d["quads"]) - 1),
                                            offset=0.1, blocks=blocks))
    table = AuxStateTable(m, (0.0, 0.7, 0.0), (0.0, 3.0, 0.0), (None, None, None))
    system = _fresh_system()
    umb = mm.CustomExternalForce("0.5*k*(x-r0)^2")
    umb.addGlobalParameter("k", 0.0)
    umb.addGlobalParameter("r0", 0.0)
    umb.setForceGroup(31)
    system.addForce(umb)
    rt = add_aux_cv_force(mm, system, table, ARGS)
    ctx = mm.Context(system, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference"))
    ctx.setPositions(d["positions_nm"])
    return ctx, rt, unit


def _aux_energy(ctx, rt, unit):
    return ctx.getState(getEnergy=True, groups={rt.info.force_group}).getPotentialEnergy().value_in_unit(
        unit.kilojoule_per_mole)


def test_active_ordinary_active_is_exact():
    ctx, rt, unit = _ctx_and_runtime()
    centers, ks = [0.0] * 3, [0.0] * 3
    set_window(ctx, centers, ks, 1, aux_state=rt)
    e_active = _aux_energy(ctx, rt, unit)
    assert e_active > 0.0
    set_window(ctx, centers, ks, 0, aux_state=rt)
    assert _aux_energy(ctx, rt, unit) == 0.0
    assert ctx.getParameter(rt.info.global_k) == 0.0 and ctx.getParameter(rt.info.global_c) == 0.0
    set_window(ctx, centers, ks, 1, aux_state=rt)
    assert _aux_energy(ctx, rt, unit) == e_active


def test_missing_aux_globals_raise():
    import openmm as mm
    s = mm.System()
    s.addParticle(1.0)
    f = mm.CustomExternalForce("0.5*k*(x-r0)^2")
    f.addGlobalParameter("k", 0.0)
    f.addGlobalParameter("r0", 0.0)
    f.addParticle(0, [])
    s.addForce(f)
    ctx = mm.Context(s, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference"))
    _ctx, rt, _unit = _ctx_and_runtime()
    with pytest.raises(Exception):
        set_window(ctx, [0.0, 0.0], [0.0, 0.0], 1, aux_state=rt)


def test_every_production_set_window_call_passes_aux_state():
    import gareus.production as production
    tree = ast.parse(inspect.getsource(production))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == "set_window"]
    assert len(calls) >= 5, "set_window call sites moved; re-anchor this test"
    missing = [n.lineno for n in calls if not any(k.arg == "aux_state" for k in n.keywords)]
    assert not missing, f"set_window calls without aux_state= at lines {missing}"
```

```python
# tests/test_aux_cv_composition.py
"""The auxiliary force must enter the integrator once, at full strength, outside the boost (spec 3.3)."""
import types

import numpy as np

from aux_cv_fixture import dipeptide, model_payload
from gareus.auxiliary_cv.evaluate import aux_energy_kj, aux_forces_kj_nm, z_from_positions
from gareus.auxiliary_cv.model import AuxModel
from gareus.auxiliary_cv.runtime import add_aux_cv_force
from gareus.auxiliary_cv.state_table import AuxStateTable
from gareus.imports import import_openmm

ARGS = types.SimpleNamespace(umbrella_force_group=31, secondary_cv_force_group=29)
K_KCAL, OFFSET_TO_CENTER = 4.0, 0.35


def _table():
    d = dipeptide()
    blocks = [lab.split("-")[0] for lab in d["labels"]]
    m = AuxModel.from_mapping(model_payload(d["quads"], [1.0] + [0.5] * (2 * len(d["quads"]) - 1),
                                            offset=0.0, blocks=blocks))
    z0 = float(z_from_positions(d["positions_nm"], m)[0])
    return AuxStateTable(m, (z0 - OFFSET_TO_CENTER,), (K_KCAL,), (None,)), d


def _with_active_aux(system, table):
    openmm, _app, _unit = import_openmm()
    rt = add_aux_cv_force(openmm, system, table, ARGS)
    force = system.getForce(rt.force_index)
    force.setGlobalParameterDefaultValue(0, K_KCAL * 4.184)          # aux_k (kJ/mol per z^2)
    force.setGlobalParameterDefaultValue(1, table.centers[0])          # aux_c
    return rt


def _constant_force(system, forces_kj_nm, group):
    openmm, _app, _unit = import_openmm()
    f = openmm.CustomExternalForce("-(fx*x+fy*y+fz*z)")
    for name in ("fx", "fy", "fz"):
        f.addPerParticleParameter(name)
    for i, vec in enumerate(forces_kj_nm):
        if np.any(vec):
            f.addParticle(i, [float(v) for v in vec])
    f.setForceGroup(group)
    system.addForce(f)


def test_group_energies_reconstruct_the_total_and_aux_is_a_bias_group():
    from pep_gamd_fixture import _fresh_system
    from gareus import pep_gamd
    openmm, _app, unit = import_openmm()
    table, d = _table()
    system = _fresh_system()
    pep_gamd.ensure_pep_gamd_partition(system, dipeptide_peptide())
    rt = _with_active_aux(system, table)
    assert rt.info.force_group in pep_gamd.pep_gamd_bias_force_groups(system)
    ctx = openmm.Context(system, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName("Reference"))
    ctx.setPositions(d["positions_nm"])
    groups = sorted({system.getForce(i).getForceGroup() for i in range(system.getNumForces())})
    e = {g: ctx.getState(getEnergy=True, groups={g}).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
         for g in groups}
    total = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    assert abs(sum(e.values()) - total) < 1e-6 * max(1.0, abs(total))
    z = z_from_positions(d["positions_nm"], table.model)
    assert abs(e[rt.info.force_group] - aux_energy_kj(z, table.centers[0], K_KCAL)[0]) < 1e-8


def dipeptide_peptide():
    from pep_gamd_fixture import solvated_dipeptide
    return solvated_dipeptide()["peptide"]


def test_pep_gamd_applies_the_aux_gradient_once_unscaled():
    """One boosted step with the real aux force == one step with a constant external force of the
    same value in the same (bias) group; the same constant force in boosted group 0 differs.

    Warm-up rule (CLAUDE.md): gamd-openmm's first step of a fresh Context moves nothing, so the
    helper steps once, re-seats the state, then measures.
    """
    from pep_gamd_fixture import _fresh_system, solvated_dipeptide
    from gareus import pep_gamd
    from test_pep_gamd_boost import _gamd_kwargs, _one_step_positions, _assert_moved
    openmm, _app, unit = import_openmm()
    table, d = _table()
    fx = solvated_dipeptide()
    f_aux = aux_forces_kj_nm(d["positions_nm"], table.model, table.centers[0], K_KCAL)
    assert np.abs(f_aux).max() > 1.0, "the auxiliary force must be large enough to move atoms measurably"

    def _system(kind):
        s = _fresh_system()
        pep_gamd.ensure_pep_gamd_partition(s, fx["peptide"])
        if kind == "aux":
            _with_active_aux(s, table)
        elif kind == "ext_bias":
            _constant_force(s, f_aux, group=_with_active_aux(_fresh_system_copy(), table).info.force_group)
        elif kind == "ext_boosted":
            _constant_force(s, f_aux, group=0)
        return s

    def _fresh_system_copy():
        s = _fresh_system()
        pep_gamd.ensure_pep_gamd_partition(s, fx["peptide"])
        return s

    probe = openmm.Context(_system("none"), openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName("Reference"))
    probe.setPositions(fx["positions"])
    v_pep = pep_gamd.peptide_essential_energy_kj(probe, unit)
    v_dih = probe.getState(getEnergy=True, groups={pep_gamd.DIHEDRAL_GROUP}).getPotentialEnergy().value_in_unit(
        unit.kilojoule_per_mole)
    del probe
    # Production stage with an ACTIVE boost: FSF = 1 - k0 (Vmax - V)/(Vmax - Vmin) = 0.75 on both channels.
    stage = {"stepCount": 50, "stage": 5,
             "k0_Total": 0.5, "Vmax_Total": v_pep + 50.0, "Vmin_Total": v_pep - 50.0, "threshold_energy_Total": v_pep + 50.0,
             "k0_Dihedral": 0.5, "Vmax_Dihedral": v_dih + 50.0, "Vmin_Dihedral": v_dih - 50.0,
             "threshold_energy_Dihedral": v_dih + 50.0}
    kw = _gamd_kwargs(unit)
    out = {}
    for kind in ("aux", "ext_bias", "ext_boosted", "none"):
        s = _system(kind)
        integ = pep_gamd.PepGaMDLowerDualIntegrator(pep_gamd.DIHEDRAL_GROUP,
                                                    bias_force_groups=pep_gamd.pep_gamd_bias_force_groups(s), **kw)
        out[kind] = _one_step_positions(s, integ, fx["positions"], openmm, unit, stage_globals=stage)
    _assert_moved(out["aux"], fx["positions"], unit)
    assert np.max(np.abs(out["aux"] - out["ext_bias"])) < 1e-9, "aux gradient is not applied once at full strength"
    assert np.max(np.abs(out["aux"] - out["none"])) > 1e-7, "the aux force did not move anything; test is vacuous"
    assert np.max(np.abs(out["ext_boosted"] - out["ext_bias"])) > 1e-7, \
        "a boosted copy is indistinguishable: the comparison cannot detect FSF scaling"


def test_npt_trial_energy_includes_aux_at_the_trial_geometry():
    from pep_gamd_fixture import _fresh_system
    from gareus import npt
    from gareus.pep_gamd import ConventionalNptTargetAdapter
    openmm, _app, unit = import_openmm()
    table, d = _table()
    system = _fresh_system()
    rt = _with_active_aux(system, table)
    adapter = ConventionalNptTargetAdapter(system)
    assert rt.info.force_group in adapter._bias_groups
    ctx = openmm.Context(system, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName("Reference"))
    ctx.setPositions(d["positions_nm"])
    before = adapter.evaluate(ctx, None)
    z0 = z_from_positions(d["positions_nm"], table.model)
    assert abs(before.bias_kj_mol - aux_energy_kj(z0, table.centers[0], K_KCAL)[0]) < 1e-8
    mol_ids, mol_sizes = npt._molecule_index_arrays(npt._molecules_from_context(ctx), system.getNumParticles())
    trial = npt._scale_about_molecule_centroids(d["positions_nm"], mol_ids, mol_sizes, 0.02)
    a, b, c = ctx.getState().getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)
    ctx.setPeriodicBoxVectors(a * 1.02, b * 1.02, c * 1.02)
    ctx.setPositions(trial)
    after = adapter.evaluate(ctx, None)
    zt = z_from_positions(trial, table.model)
    assert abs(after.bias_kj_mol - aux_energy_kj(zt, table.centers[0], K_KCAL)[0]) < 1e-8
```

In `test_pep_gamd_applies_the_aux_gradient_once_unscaled`, the `ext_bias` system needs the constant force in the **same group the aux force would get**. The test computes that group from an identical fresh partitioned system; `test_same_audit_gives_same_group_on_a_copy` in Task 3 proves the allocation is deterministic. Python resolves `_fresh_system_copy` at call time, so its definition after `_system` is fine.

- [ ] **Step 2: Run to verify failure**

Run: `tests/test_aux_cv_set_window.py tests/test_aux_cv_composition.py`
Expected: `test_aux_cv_set_window.py` FAILs (`set_window() got an unexpected keyword argument 'aux_state'`). The composition tests may already PASS, because they exercise Stage A plus Task 3 only: they are proofs, not a red/green pair. If any composition test fails, stop. It means the auxiliary force is scaled by the boost or missing from NPT, and that is a release blocker (spec 19, A06).

- [ ] **Step 3: Implement**

`gareus/windows.py`: extend the signature with `*, aux_state=None` and append to the body (after the secondary block):

```python
    if aux_state is not None:
        # Auxiliary-CV state (spec 3.3): set the COMPLETE target every time, including the reset
        # to zero when the window is ordinary. No try/except: a Context built without the
        # auxiliary capability while a runtime is configured is a fatal inconsistency.
        from .auxiliary_cv.force import set_aux_parameters
        w = int(window_index)
        set_aux_parameters(context, aux_state.info, center=aux_state.table.centers[w],
                           k_kcal=aux_state.table.k_kcal[w])
```

In `gareus/production.py`, add `aux_state=...` to each of the five calls. Exact replacements:

```python
# :5592 (_apply_assignment inside load_production_checkpoint)
        set_window(sim.context, centers_nm, ks_kj_nm2, int(assignments[r]), secondary_centers, secondary_ks_kj,
                   aux_state=None if args is None else getattr(args, "_aux_runtime", None))
# :5891 (run_multiwindow_gamd_recon)
        set_window(sim_i.context, centers_nm, ks_kj_nm2, i, secondary_cv_centers, secondary_cv_ks_kj,
                   aux_state=getattr(args, "_aux_runtime", None))
# :6128 (apply_joint_envelope_gamd_calibration)
        set_window(_check_sim.context, centers_nm, ks_kj_nm2, 0, secondary_cv_centers, secondary_cv_ks_kj,
                   aux_state=getattr(args, "_aux_runtime", None))
# :7522 (_build_context_i)
                set_window(sim_i.context, centers_nm, ks_kj_nm2, i, secondary_cv_centers, secondary_cv_ks_kj,
                           aux_state=getattr(args, "_aux_runtime", None))
# :8420 (_apply_swap_to_replica)
                    set_window(sims[replica_index].context, centers_nm, ks_kj_nm2, assignments[replica_index],
                               secondary_cv_centers, secondary_cv_ks_kj,
                               aux_state=getattr(args, "_aux_runtime", None))
```

The swap's two `set_window` calls plus the map change already run as one transaction: `_apply_swap_to_replica` runs on both replicas before `observable_cache.clear()`, and no integration happens in between. Keep that ordering.

- [ ] **Step 4: Run to verify pass**

Run: `tests/test_aux_cv_set_window.py tests/test_aux_cv_composition.py tests/test_sample_before_exchange_ordering.py tests/test_pep_gamd_boost.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add gareus/windows.py gareus/production.py tests/test_aux_cv_set_window.py tests/test_aux_cv_composition.py
git commit -m "feat(cvaux): set_window applies the complete aux target; composition, gradient-once and NPT proofs"
```

---

### Task 5: Observation of z for every carrier

**Files:**
- Modify: `gareus/auxiliary_cv/runtime.py` (add `observe_aux_z`)
- Test: `tests/test_aux_cv_observation.py`

**Interfaces:**
- Produces: `observe_aux_z(context, runtime: AuxRuntime, *, force=None, positions_nm=None) -> float`.
  - With `force`, z is read from the force's sub-CVs: z = offset + Σ sub-CV values, valid because each sub-CV is a weighted trig sum (Stage A Task 4).
  - With `positions_nm`, z = `z_from_positions`.
  - Exactly one of the two must be given.
  - A non-finite z raises `AuxObservationError`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_aux_cv_observation.py
import types

import numpy as np
import pytest

from aux_cv_fixture import dipeptide, model_payload
from gareus.auxiliary_cv.evaluate import z_from_positions
from gareus.auxiliary_cv.model import AuxModel
from gareus.auxiliary_cv.runtime import AuxObservationError, add_aux_cv_force, observe_aux_z
from gareus.auxiliary_cv.state_table import AuxStateTable

ARGS = types.SimpleNamespace(umbrella_force_group=31, secondary_cv_force_group=29)


def _setup(conventions=None):
    import openmm as mm
    from pep_gamd_fixture import _fresh_system
    d = dipeptide()
    blocks = [lab.split("-")[0] for lab in d["labels"]]
    rng = np.random.default_rng(5)
    m = AuxModel.from_mapping(model_payload(d["quads"], rng.normal(size=2 * len(d["quads"])), offset=-0.3,
                                            conventions=conventions, blocks=blocks))
    system = _fresh_system()
    rt = add_aux_cv_force(mm, system, AuxStateTable(m, (0.0,), (0.0,), (None,)), ARGS)
    ctx = mm.Context(system, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference"))
    ctx.setPositions(d["positions_nm"])
    return ctx, rt, system.getForce(rt.force_index), d


@pytest.mark.parametrize("mixed", [False, True])
def test_fast_and_slow_paths_agree_for_an_ordinary_state(mixed):
    """k = 0 everywhere: an ordinary carrier still yields its z for cross-evaluation (spec 3.1)."""
    d0 = dipeptide()
    conv = ["negated" if k % 2 == 0 else "direct" for k in range(len(d0["quads"]))] if mixed else None
    ctx, rt, force, d = _setup(conv)
    fast = observe_aux_z(ctx, rt, force=force)
    slow = observe_aux_z(ctx, rt, positions_nm=d["positions_nm"])
    assert fast == pytest.approx(slow, abs=1e-9)
    assert slow == pytest.approx(z_from_positions(d["positions_nm"], rt.table.model)[0], abs=1e-12)


def test_exactly_one_source():
    ctx, rt, force, d = _setup()
    with pytest.raises(ValueError):
        observe_aux_z(ctx, rt)
    with pytest.raises(ValueError):
        observe_aux_z(ctx, rt, force=force, positions_nm=d["positions_nm"])


def test_nonfinite_z_is_fatal():
    ctx, rt, _force, d = _setup()
    bad = d["positions_nm"].copy()
    q = rt.table.model.feature_schema.features[0].atom_indices
    bad[q[0]] = bad[q[1]]                       # collapse two torsion atoms -> undefined dihedral
    with pytest.raises(AuxObservationError):
        observe_aux_z(ctx, rt, positions_nm=bad)
```

- [ ] **Step 2: Run to verify failure**

Run: `tests/test_aux_cv_observation.py`
Expected: FAIL (`ImportError: observe_aux_z`).

- [ ] **Step 3: Implement**

Append to `gareus/auxiliary_cv/runtime.py`:

```python
import math

import numpy as np

from .evaluate import z_from_positions


def observe_aux_z(context, runtime: AuxRuntime, *, force=None, positions_nm=None) -> float:
    """z of one carrier under the phase's auxiliary model, whatever state it occupies."""
    if (force is None) == (positions_nm is None):
        raise ValueError("observe_aux_z needs exactly one of force= (fast path) or positions_nm= (slow path)")
    if force is not None:
        values = np.asarray(force.getCollectiveVariableValues(context), dtype=np.float64)
        z = float(runtime.table.model.offset + values.sum())
    else:
        z = float(z_from_positions(positions_nm, runtime.table.model)[0])
    if not math.isfinite(z):
        raise AuxObservationError(f"auxiliary z is not finite ({z!r}); a degenerate torsion or broken "
                                  "geometry stops the segment rather than dropping the state")
    return z
```

Move these imports to the top of the module with the others.

- [ ] **Step 4: Run to verify pass**

Run: `tests/test_aux_cv_observation.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add gareus/auxiliary_cv/runtime.py tests/test_aux_cv_observation.py
git commit -m "feat(cvaux): observe_aux_z fast/slow paths, fatal on non-finite z"
```

---

### Task 6: Auxiliary bias matrix in the sample and exchange paths

**Files:**
- Modify: `gareus/auxiliary_cv/runtime.py` (add `aux_bias_matrix_kcal`)
- Modify: `gareus/production.py`:
  - `assemble_bias_matrices` (`:2209`);
  - fast-path setup (`:7625-7655`);
  - the sample closure `_fetch_state` and its assembly (`:8080-8130`);
  - `sampled_umbrella_bias_kj` (`:8149`);
  - `_current_exchange_arrays` (`:8438-8498`).
- Test: `tests/test_aux_cv_production_wiring.py`

**Interfaces:**
- Produces:
  - `aux_bias_matrix_kcal(z_values, table) -> np.ndarray` of shape (n_states, n_replicas), kcal/mol. Inactive rows are exact zeros. If any state is active, every z must be finite, else `AuxObservationError`.
  - `assemble_bias_matrices(distance_bias_kcal, ss_bias_kcal, boost_bias_kj, aux_bias_kcal=None)`.
  - In `run_gareus`, one closure `_aux_z_for_replica(r, sim) -> float` (NaN when no runtime) called by **both** `_fetch_state` and `_fetch_exchange_state`.
  - `observable_cache["aux_z"]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_aux_cv_production_wiring.py
import ast
import inspect

import numpy as np
import pytest

from aux_cv_fixture import model_payload
from gareus.auxiliary_cv.model import AuxModel
from gareus.auxiliary_cv.runtime import AuxObservationError, aux_bias_matrix_kcal
from gareus.auxiliary_cv.state_table import AuxStateTable
from gareus.production import assemble_bias_matrices

M = AuxModel.from_mapping(model_payload([(0, 1, 2, 3)], [1.0, 0.0]))
TABLE = AuxStateTable(M, (0.0, 1.0, 0.0), (0.0, 2.0, 0.0), (None, None, None))


def test_aux_matrix_values_and_exact_zero_rows():
    z = np.array([0.5, 1.5, -1.0, 0.25])
    out = aux_bias_matrix_kcal(z, TABLE)
    assert out.shape == (3, 4)
    np.testing.assert_array_equal(out[0], 0.0)
    np.testing.assert_array_equal(out[2], 0.0)
    np.testing.assert_allclose(out[1], 0.5 * 2.0 * (z - 1.0) ** 2, rtol=1e-15)


def test_nonfinite_z_is_fatal_when_any_state_is_active():
    with pytest.raises(AuxObservationError):
        aux_bias_matrix_kcal(np.array([0.5, np.nan]), TABLE)


def test_all_zero_table_ignores_z():
    zero = AuxStateTable(M, (0.0, 0.0), (0.0, 0.0), (None, None))
    np.testing.assert_array_equal(aux_bias_matrix_kcal(np.array([np.nan, 1.0]), zero), 0.0)


def test_assemble_without_aux_is_bitwise_legacy():
    rng = np.random.default_rng(1)
    d, s, b = rng.normal(size=(3, 3)), rng.normal(size=(3, 3)), rng.normal(size=(3, 3))
    legacy_kcal = d + s + b / 4.184
    kcal, kj = assemble_bias_matrices(d, s, b)
    assert np.array_equal(kcal, legacy_kcal) and np.array_equal(kj, 4.184 * legacy_kcal)
    kcal0, _ = assemble_bias_matrices(d, s, b, aux_bias_kcal=np.zeros((3, 3)))
    assert np.array_equal(kcal0, kcal)


def test_assemble_adds_aux():
    rng = np.random.default_rng(2)
    d, s, b, a = (rng.normal(size=(2, 2)) for _ in range(4))
    kcal, kj = assemble_bias_matrices(d, s, b, aux_bias_kcal=a)
    np.testing.assert_allclose(kcal, d + s + a + b / 4.184, rtol=1e-15)
    np.testing.assert_allclose(kj, 4.184 * kcal, rtol=1e-15)


def _run_gareus_source():
    import gareus.production as production
    return ast.parse(inspect.getsource(production.run_gareus))


def _func(tree, name):
    hits = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(hits) == 1, f"{name} not found exactly once in run_gareus; re-anchor"
    return hits[0]


def _calls(node, name):
    return [n for n in ast.walk(node) if isinstance(n, ast.Call)
            and ((isinstance(n.func, ast.Name) and n.func.id == name)
                 or (isinstance(n.func, ast.Attribute) and n.func.attr == name))]


def test_both_fetch_closures_observe_z_through_one_helper():
    tree = _run_gareus_source()
    for name in ("_fetch_state", "_fetch_exchange_state"):
        assert _calls(_func(tree, name), "_aux_z_for_replica"), f"{name} does not observe aux z"


def test_sample_and_exchange_assembly_both_include_the_aux_matrix():
    tree = _run_gareus_source()
    exch = _func(tree, "_current_exchange_arrays")
    assert _calls(exch, "aux_bias_matrix_kcal")
    assert any(k.arg == "aux_bias_kcal" for c in _calls(exch, "assemble_bias_matrices") for k in c.keywords)
    all_assemble = _calls(tree, "assemble_bias_matrices")
    assert len(all_assemble) == 2 and all(any(k.arg == "aux_bias_kcal" for k in c.keywords) for c in all_assemble)
    assert len(_calls(tree, "aux_bias_matrix_kcal")) == 2
```

- [ ] **Step 2: Run to verify failure**

Run: `tests/test_aux_cv_production_wiring.py`
Expected: FAIL (`ImportError: aux_bias_matrix_kcal`).

- [ ] **Step 3: Implement the matrix and assembly**

Append to `gareus/auxiliary_cv/runtime.py`:

```python
def aux_bias_matrix_kcal(z_values, table: AuxStateTable) -> "np.ndarray":
    """[state, replica] auxiliary bias in kcal/mol; inactive states contribute exact zeros.

    Every replica's z enters every active row, including replicas whose own state is ordinary
    (spec 4.2). A non-finite z with any active state is fatal: the matrix never drops a state.
    """
    z = np.asarray(z_values, dtype=np.float64)
    k = np.asarray(table.k_kcal, dtype=np.float64)
    c = np.asarray(table.centers, dtype=np.float64)
    out = np.zeros((k.size, z.size), dtype=np.float64)
    active = k > 0.0
    if not np.any(active):
        return out
    if not np.all(np.isfinite(z)):
        bad = np.flatnonzero(~np.isfinite(z)).tolist()
        raise AuxObservationError(f"non-finite auxiliary z for replica(s) {bad} with active auxiliary states")
    d = z[np.newaxis, :] - c[active][:, np.newaxis]
    out[active] = 0.5 * k[active][:, np.newaxis] * d * d
    return out
```

Replace `assemble_bias_matrices` in `gareus/production.py` (`:2209-2225`) with:

```python
def assemble_bias_matrices(distance_bias_kcal, ss_bias_kcal, boost_bias_kj, aux_bias_kcal=None):
    """Combine the umbrella components, optional auxiliary-CV bias and the Pep-GaMD boost.

    Both unit matrices carry the SAME quantity, so ``bias_kj == 4.184 * bias_kcal`` holds.
    ``boost_bias_kj`` is all-zero when the λ-ladder is inactive. ``aux_bias_kcal`` None keeps the
    exact pre-auxiliary arithmetic (off path byte-identical). Shared by both assembly sites in
    ``run_gareus`` (sample path and ``_current_exchange_arrays``).
    """
    distance_bias_kcal = np.asarray(distance_bias_kcal, dtype=np.float64)
    ss_bias_kcal = np.asarray(ss_bias_kcal, dtype=np.float64)
    boost_bias_kj = np.asarray(boost_bias_kj, dtype=np.float64)
    boost_bias_kcal = boost_bias_kj / 4.184
    if aux_bias_kcal is None:
        bias_kcal = distance_bias_kcal + ss_bias_kcal + boost_bias_kcal
    else:
        bias_kcal = distance_bias_kcal + ss_bias_kcal + np.asarray(aux_bias_kcal, dtype=np.float64) + boost_bias_kcal
    bias_kj = 4.184 * bias_kcal
    return bias_kcal, bias_kj
```

`test_assemble_without_aux_is_bitwise_legacy` also checks `aux_bias_kcal=np.zeros` against no aux. Adding +0.0 to a finite float is exact, so those compare equal bitwise.

- [ ] **Step 4: Wire `run_gareus`**

After the fast-path block (just after `_use_fast_cv_path = (...)` and its `print`, `gareus/production.py:7647-7655`), add:

```python
    # ── Auxiliary-CV states (spec 3.3/4.2): one z per carrier, whatever state it occupies ──
    _aux_rt = getattr(args, "_aux_runtime", None)
    _fast_aux_forces = [None] * nrep
    if _aux_rt is not None:
        from .auxiliary_cv.force import AUX_FORCE_NAME as _AUX_CV_FORCE_NAME
        from .auxiliary_cv.runtime import aux_bias_matrix_kcal, observe_aux_z
        _fast_aux_forces = [sim.system.getForce(_aux_rt.force_index) for sim in sims]
        if any(f.getName() != _AUX_CV_FORCE_NAME for f in _fast_aux_forces):
            raise RuntimeError("replica systems do not carry the auxiliary-CV force at the base system's index")

    def _aux_z_for_replica(r, sim) -> float:
        """The single z observation both the sample writer and the exchange kernel use."""
        if _aux_rt is None:
            return float("nan")
        if _use_fast_cv_path:
            return observe_aux_z(sim.context, _aux_rt, force=_fast_aux_forces[r])
        pos = sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        return observe_aux_z(sim.context, _aux_rt, positions_nm=pos)
```

Sample path (`_fetch_state`, `:8090-8112`):
- Both return statements become `return r, cv, ss, pe, v_pep, v_dih, _aux_z_for_replica(r, sim)`.
- Before the closure, next to `v_dih_kj = np.empty(...)`, add `aux_z = np.full(nrep, np.nan, dtype=np.float64)`.
- Change the unpacking loop to:

```python
            for r, cv, ss, pe, v_pep, v_dih, z_aux in _fetched_states:
                primary_values[r], ss_values[r], potentials_kj[r] = cv, ss, pe
                v_pep_kj[r], v_dih_kj[r] = v_pep, v_dih
                aux_z[r] = z_aux
```

Replace the sample assembly (`bias_matrix_kcal, bias_matrix_kj = assemble_bias_matrices(...)`, `:8124`) with:

```python
            aux_bias_matrix_kcal_now = aux_bias_matrix_kcal(aux_z, _aux_rt.table) if _aux_rt is not None else None
            bias_matrix_kcal, bias_matrix_kj = assemble_bias_matrices(
                distance_bias_matrix_kcal, ss_bias_matrix_kcal, boost_bias_matrix_kj,
                aux_bias_kcal=aux_bias_matrix_kcal_now,
            )
```

and add `observable_cache["aux_z"] = aux_z` next to `observable_cache["v_pep_kj"] = v_pep_kj`. Replace the line `sampled_umbrella_bias_kj = float(4.184 * (all_distance_bias_kcal[w] + all_ss_bias_kcal[w]))` (`:8149`) with:

```python
                if aux_bias_matrix_kcal_now is None:
                    sampled_umbrella_bias_kj = float(4.184 * (all_distance_bias_kcal[w] + all_ss_bias_kcal[w]))
                else:
                    sampled_umbrella_bias_kj = float(4.184 * (all_distance_bias_kcal[w] + all_ss_bias_kcal[w]
                                                              + aux_bias_matrix_kcal_now[w, r]))
```

Exchange path (`_current_exchange_arrays`, `:8438-8498`):
- Add `aux_z = np.full(nrep, np.nan, dtype=np.float64)` next to `v_dih_kj = np.empty(...)`.
- `_fetch_exchange_state` returns `r, cv, ss, v_pep, v_dih, _aux_z_for_replica(r, sim)` on both branches, and the unpacking loop sets `aux_z[r] = z_aux`.
- Replace the final assembly with:

```python
            aux_bias_kcal = aux_bias_matrix_kcal(aux_z, _aux_rt.table) if _aux_rt is not None else None
            _, bias_matrix_kj = assemble_bias_matrices(distance_bias_kcal, ss_bias_kcal, boost_bias_matrix_kj,
                                                       aux_bias_kcal=aux_bias_kcal)
            return primary_values, bias_matrix_kj
```

The cached-observable shortcut at the top of `_current_exchange_arrays` returns the sample path's `bias_matrix_kj`. That matrix already contains the aux term, so the shortcut stays correct.

- [ ] **Step 5: Run to verify pass**

Run: `tests/test_aux_cv_production_wiring.py tests/test_sample_before_exchange_ordering.py tests/test_exchange_kernel_exact.py tests/test_lambda_ladder_states.py`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add gareus/auxiliary_cv/runtime.py gareus/production.py tests/test_aux_cv_production_wiring.py
git commit -m "feat(cvaux): auxiliary bias in both sample and exchange matrices"
```

---

### Task 7: Direct cross-energies and exact Gibbs permutation invariance

**Files:**
- Test only: `tests/test_aux_cv_exchange.py`. These tests prove existing kernel semantics against matrices that contain auxiliary rows, so no production code changes. If any test fails, the defect is in Tasks 3-6.

**Interfaces:**
- Consumes:
  - `tests/test_exchange_kernel_exact.py` helpers: `_states`, `_pi`, `_pair_kernel`, `_gibbs_kernel`, `_assert_kernel_ok`, `BETA`;
  - `gareus.production._gibbs_window_proposal_distribution`;
  - Tasks 3-6.

- [ ] **Step 1: Write the tests**

```python
# tests/test_aux_cv_exchange.py
"""Spec 17: direct cross-energy check, exact Gibbs permutations with ordinary/aux/duplicate states,
global candidates. The exchange kernel is unchanged; these pin it against auxiliary matrices."""
import itertools
import types

import numpy as np
import pytest

from aux_cv_fixture import dipeptide, model_payload
from gareus.auxiliary_cv.force import set_aux_parameters
from gareus.auxiliary_cv.model import AuxModel
from gareus.auxiliary_cv.runtime import add_aux_cv_force, aux_bias_matrix_kcal, observe_aux_z
from gareus.auxiliary_cv.state_table import AuxStateTable
from gareus.production import _gibbs_window_proposal_distribution
from test_exchange_kernel_exact import BETA, _assert_kernel_ok, _gibbs_kernel, _pair_kernel, _pi, _states

ARGS = types.SimpleNamespace(umbrella_force_group=31, secondary_cv_force_group=29)


def _carriers(n_carriers=2):
    """Distinct real configurations: carrier r is the fixture after r * 20 Langevin steps."""
    import openmm as mm
    from openmm import unit
    from pep_gamd_fixture import _fresh_system
    d = dipeptide()
    blocks = [lab.split("-")[0] for lab in d["labels"]]
    m = AuxModel.from_mapping(model_payload(d["quads"], [1.2] + [0.6] * (2 * len(d["quads"]) - 1),
                                            offset=0.0, blocks=blocks))
    table = AuxStateTable(m, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (None, None, None))
    ctxs, rt = [], None
    for r in range(n_carriers):
        system = _fresh_system()
        rt = add_aux_cv_force(mm, system, table, ARGS)
        integ = mm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 0.002 * unit.picoseconds)
        integ.setRandomNumberSeed(11 + r)
        ctx = mm.Context(system, integ, mm.Platform.getPlatformByName("Reference"))
        ctx.setPositions(d["positions_nm"])
        ctx.setVelocitiesToTemperature(300 * unit.kelvin, 11 + r)
        integ.step(20 * r)
        ctxs.append(ctx)
    return ctxs, rt, unit


def test_assembled_matrix_equals_direct_context_energies_and_all_four_swap_terms():
    ctxs, rt, unit = _carriers(2)
    z = np.array([observe_aux_z(c, rt, force=c.getSystem().getForce(rt.force_index)) for c in ctxs])
    assert abs(z[0] - z[1]) > 1e-3, "carriers must differ in z for a meaningful cross-energy check"
    table = AuxStateTable(rt.table.model, (z[0] + 0.4, 0.0, z[1] - 0.2), (2.5, 0.0, 1.0), (None, None, None))
    matrix_kj = 4.184 * aux_bias_matrix_kcal(z, table)
    direct = np.zeros_like(matrix_kj)
    for s in range(table.n):
        for r, ctx in enumerate(ctxs):
            set_aux_parameters(ctx, rt.info, center=table.centers[s], k_kcal=table.k_kcal[s])
            direct[s, r] = ctx.getState(getEnergy=True, groups={rt.info.force_group}).getPotentialEnergy(
            ).value_in_unit(unit.kilojoule_per_mole)
    np.testing.assert_allclose(matrix_kj, direct, rtol=1e-9, atol=1e-9)
    for a, b in itertools.permutations(range(table.n), 2):
        d_matrix = matrix_kj[b, 0] + matrix_kj[a, 1] - matrix_kj[a, 0] - matrix_kj[b, 1]
        d_direct = direct[b, 0] + direct[a, 1] - direct[a, 0] - direct[b, 1]
        assert d_matrix == pytest.approx(d_direct, abs=1e-8)


def _aux_bias(n, with_duplicate):
    """[state, replica] kJ: random CV1-like umbrella part + an auxiliary row; state 3 duplicates state 1."""
    rng = np.random.default_rng(20261007)
    base = rng.uniform(0.0, 5.0, size=(n, n))
    z = rng.normal(0.0, 1.0, size=n)
    k = np.zeros(n)
    c = np.zeros(n)
    k[2], c[2] = 3.0, 0.5                     # state 2 is auxiliary
    table = AuxStateTable(AuxModel.from_mapping(model_payload([(0, 1, 2, 3)], [1.0, 0.0])),
                          tuple(c), tuple(k), (None,) * n)
    bias = base + 4.184 * aux_bias_matrix_kcal(z, table)
    if with_duplicate:
        bias[3] = bias[1]                     # sham: identical Hamiltonian to its parent, separate slot
    return bias


@pytest.mark.parametrize("n, dup", [(3, False), (4, False), (4, True)])
def test_gibbs_and_pair_kernels_are_detailed_balanced_with_aux_and_sham_states(n, dup):
    states = _states(n)
    bias = _aux_bias(n, dup and n >= 4)
    pi = _pi(states, bias)
    for rep in range(n):
        _assert_kernel_ok(_gibbs_kernel(states, bias, rep), pi, f"gibbs rep={rep} n={n} dup={dup}")
    for wi, wj in itertools.combinations(range(n), 2):
        _assert_kernel_ok(_pair_kernel(states, bias, wi, wj), pi, f"pair {wi},{wj} n={n} dup={dup}")


def test_composed_gibbs_sweep_is_stationary_with_a_duplicate_state():
    n = 4
    states, bias = _states(n), _aux_bias(n, True)
    pi = _pi(states, bias)
    sweep = np.eye(len(states))
    for rep in range(n):
        sweep = sweep @ _gibbs_kernel(states, bias, rep)
    assert np.abs(pi @ sweep - pi).max() < 1e-12


def test_auxiliary_carrier_sees_every_occupied_state_as_a_candidate():
    """No parent-only or neighbour mask: a carrier in the aux state may propose any ordinary state."""
    n = 4
    bias = _aux_bias(n, True)
    replica_of_window = np.arange(n, dtype=np.int64)
    prop = _gibbs_window_proposal_distribution(BETA, bias, replica_index=2, current_window=2,
                                               replica_of_window=replica_of_window)
    assert sorted(prop["windows"].tolist()) == list(range(n))
    assert np.all(prop["probabilities"] > 0.0)
```

- [ ] **Step 2: Run**

Run: `tests/test_aux_cv_exchange.py tests/test_exchange_kernel_exact.py`
Expected: PASS. A failure here is a Task 3-6 bug: fix it there, never weaken these tests.

- [ ] **Step 3: Commit**

```bash
git add tests/test_aux_cv_exchange.py
git commit -m "test(cvaux): direct cross-energies and exact Gibbs permutations with aux and sham states"
```

---

### Task 8: Kernel identity `state_bias_matrix_v3_aux` and provenance

**Files:**
- Modify: `gareus/kernel_identity.py` (`EXCHANGE_ENERGY_VERSION` stays; add the aux constant, the resolver and the known set; `kernel_identity_for_run` and `classify_segment_kernel`)
- Modify: `gareus/provenance.py:345-346`
- Modify: `gareus/production.py:686-690` (`verify_kernel_identity_on_resume` compares against the args-aware version)
- Test: `tests/test_aux_cv_kernel_identity.py`

**Interfaces:**
- Produces:
  - `EXCHANGE_ENERGY_VERSION_AUX = "state_bias_matrix_v3_aux"`;
  - `KNOWN_EXCHANGE_ENERGY_VERSIONS = frozenset({EXCHANGE_ENERGY_VERSION, EXCHANGE_ENERGY_VERSION_AUX})`;
  - `exchange_energy_version_for_args(args) -> str`;
  - `kernel_identity_for_run`: the identity gains `"aux_model_sha256"` **only** when an aux model is configured, so the off-path digest is unchanged;
  - `method_settings["aux_cv_model_sha256"]`, only when configured.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_aux_cv_kernel_identity.py
import types

from aux_cv_fixture import model_payload
from gareus.auxiliary_cv.model import AuxModel
from gareus.kernel_identity import (EXCHANGE_ENERGY_VERSION, EXCHANGE_ENERGY_VERSION_AUX,
                                    classify_segment_kernel, exchange_energy_version_for_args,
                                    kernel_identity_for_run)

LEGACY_ARGS = types.SimpleNamespace(secondary_cv="none", production_ensemble="npt",
                                    gamd_boost_type="pep-gamd-lower-dual")
# Digest of the legacy identity for LEGACY_ARGS, computed on main dc30285 (kernel_identity.py unchanged).
LEGACY_DIGEST = "543f92e628d1777bdf6f9f228744eeb64159023e8ae16a36b2d775115d640bf6"


def _aux_args(tmp_path):
    m = AuxModel.from_mapping(model_payload([(0, 1, 2, 3)], [1.0, 0.0]))
    path = tmp_path / "aux.json"
    m.write(path)
    return types.SimpleNamespace(**vars(LEGACY_ARGS), aux_cv_model=str(path)), m


def test_legacy_identity_unchanged():
    ident = kernel_identity_for_run(LEGACY_ARGS, {})
    assert ident["exchange_energy_version"] == EXCHANGE_ENERGY_VERSION == "state_bias_matrix_v2"
    assert "aux_model_sha256" not in ident
    assert ident["digest"] == LEGACY_DIGEST


def test_aux_identity_and_version(tmp_path):
    args, m = _aux_args(tmp_path)
    assert exchange_energy_version_for_args(args) == EXCHANGE_ENERGY_VERSION_AUX == "state_bias_matrix_v3_aux"
    ident = kernel_identity_for_run(args, {})
    assert ident["exchange_energy_version"] == EXCHANGE_ENERGY_VERSION_AUX
    assert ident["aux_model_sha256"] == m.model_sha256


def test_residual_segment_with_aux_kernel_classifies_verified():
    from gareus.kernel_identity import RESIDUAL_EVALUATOR_VERSION, ELIGIBLE_VERIFIED
    snap = {"cv2_type": "residual-torsion-pc",
            "kernel_identity": {"cv_evaluator_version": RESIDUAL_EVALUATOR_VERSION,
                                "exchange_energy_version": EXCHANGE_ENERGY_VERSION_AUX}}
    assert classify_segment_kernel(snap)[0] == ELIGIBLE_VERIFIED


def test_method_settings_record_aux_sha_only_when_configured(tmp_path):
    from gareus.provenance import _method_settings
    from gareus.cli import parse_args
    a = parse_args(["--seq", "GA", "--out", str(tmp_path / "o"), "--cv1", "contacts"])
    s = _method_settings(a)
    assert s["exchange_energy_version"] == "state_bias_matrix_v2" and "aux_cv_model_sha256" not in s
    args, m = _aux_args(tmp_path)
    a.aux_cv_model = args.aux_cv_model
    s2 = _method_settings(a)
    assert s2["exchange_energy_version"] == "state_bias_matrix_v3_aux"
    assert s2["aux_cv_model_sha256"] == m.model_sha256
```

- [ ] **Step 1b: Confirm the legacy pin on unmodified code**

`LEGACY_DIGEST` was computed on main dc30285. Before editing, write this script to a file (the hook scans command text) and run it with `python <file>`. It must print `543f92e628d1777bdf6f9f228744eeb64159023e8ae16a36b2d775115d640bf6`. If it prints anything else, `kernel_identity.py` has changed since this plan was written: stop and report, do not re-pin.

```python
import types
from gareus.kernel_identity import kernel_identity_for_run
print(kernel_identity_for_run(types.SimpleNamespace(secondary_cv="none", production_ensemble="npt",
                                                    gamd_boost_type="pep-gamd-lower-dual"), {})["digest"])
```

- [ ] **Step 2: Run to verify failure**

Run: `tests/test_aux_cv_kernel_identity.py`
Expected: FAIL (`ImportError: EXCHANGE_ENERGY_VERSION_AUX`).

- [ ] **Step 3: Implement**

In `gareus/kernel_identity.py`, under `EXCHANGE_ENERGY_VERSION`:

```python
#: The same shared assembly plus the exact auxiliary-CV restraint term evaluated for every
#: carrier under every active state (spec 2026-10-07 auxiliary CV, Section 5). Used whenever an
#: auxiliary model is configured, including sham arms whose strengths are all zero.
EXCHANGE_ENERGY_VERSION_AUX = "state_bias_matrix_v3_aux"
KNOWN_EXCHANGE_ENERGY_VERSIONS = frozenset({EXCHANGE_ENERGY_VERSION, EXCHANGE_ENERGY_VERSION_AUX})


def exchange_energy_version_for_args(args) -> str:
    return EXCHANGE_ENERGY_VERSION_AUX if getattr(args, "aux_cv_model", None) else EXCHANGE_ENERGY_VERSION
```

In `kernel_identity_for_run`, replace `"exchange_energy_version": EXCHANGE_ENERGY_VERSION,` with `"exchange_energy_version": exchange_energy_version_for_args(args),`. Immediately before `body = json.dumps(...)`, add:

```python
    if getattr(args, "aux_cv_model", None):
        from .auxiliary_cv.model import AuxModel   # lazy: keep this module import-light
        identity["aux_model_sha256"] = AuxModel.load(args.aux_cv_model).model_sha256
```

In `classify_segment_kernel`, replace `ex == EXCHANGE_ENERGY_VERSION` with `ex in KNOWN_EXCHANGE_ENERGY_VERSIONS`.

In `gareus/provenance.py`, replace lines 345-346 with:

```python
    from .kernel_identity import RESIDUAL_EVALUATOR_VERSION, exchange_energy_version_for_args
    settings["exchange_energy_version"] = exchange_energy_version_for_args(args)
    if getattr(args, "aux_cv_model", None):
        from .auxiliary_cv.model import AuxModel
        settings["aux_cv_model_sha256"] = AuxModel.load(args.aux_cv_model).model_sha256
```

In `gareus/production.py`'s `verify_kernel_identity_on_resume`, make three changes:
- Import `exchange_energy_version_for_args` with the other kernel-identity names (`:120`).
- Add `current_ex = exchange_energy_version_for_args(args)` at the top of the function.
- Replace both `EXCHANGE_ENERGY_VERSION` uses in the refusal (`:686-690`) with `current_ex`.

- [ ] **Step 4: Run to verify pass**

Run: `tests/test_aux_cv_kernel_identity.py`, plus the existing kernel-identity regressions found with `grep -l "kernel_identity" tests/*.py` (run each file listed)
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add gareus/kernel_identity.py gareus/provenance.py gareus/production.py tests/test_aux_cv_kernel_identity.py
git commit -m "feat(cvaux): state_bias_matrix_v3_aux kernel identity when an aux model is configured"
```

---

### Task 9: `run_gareus` builds the runtime; population guards; exit gate

**Files:**
- Modify: `gareus/production.py`:
  - window loading (`:6819-6835`);
  - base system (`:7044-7047`);
  - starting-structure systems (`:7155`, `:7204`).
- Modify: `CLAUDE.md` (handoff section)
- Test: `tests/test_aux_cv_production_wiring.py` (append)

**Interfaces:**
- Consumes: Tasks 2-8.
- Produces: `args._aux_runtime: AuxRuntime | None`, set before any Context is built. The aux force sits at the same group and index in the base system and every replica copy; starting-structure systems get the same group.

- [ ] **Step 1: Write the failing structural tests (append)**

```python
def test_run_gareus_builds_aux_runtime_before_contexts_and_guards_population():
    import gareus.production as production
    src = inspect.getsource(production.run_gareus)
    tree = ast.parse(src)
    add_calls = _calls(tree, "add_aux_cv_force")
    assert len(add_calls) == 3, "base system + two starting-structure systems must all carry the aux force"
    assert sum(any(k.arg == "force_group" for k in c.keywords) for c in add_calls) == 2
    base_line = min(c.lineno for c in add_calls)
    # Replica Contexts are built by the nested _build_context_i (dispatched through the pool, so
    # it is not a direct call); the base-system aux force must precede its definition.
    assert base_line < _func(tree, "_build_context_i").lineno
    assert "_aux_runtime" in src and "population" in src
    assert _calls(tree, "load_aux_state_table") and _calls(tree, "check_feature_atoms")
```

- [ ] **Step 2: Run to verify failure**

Run: `tests/test_aux_cv_production_wiring.py`
Expected: the new test FAILs (`add_aux_cv_force` not called).

- [ ] **Step 3: Implement**

Near the top of `run_gareus` (before the `if <checkpoint manifest found>` branch at `:6810`), add `_aux_table = None`.

In the window-loading branch, right after the `filter_explicit_2d_windows_by_seed_reachability(...)` call (`:6825-6829`), add:

```python
            if getattr(args, "aux_cv_model", None):
                if len(centers_a) != _n_windows_before_reachability_filter:
                    raise RuntimeError(
                        f"auxiliary-CV state population changed: the seed-reachability filter dropped "
                        f"{_n_windows_before_reachability_filter - len(centers_a)} window(s); an auxiliary state "
                        "table is frozen and cannot be renumbered (spec Section 6). Fix the table instead.")
                from .auxiliary_cv.state_table import load_aux_state_table
                from .auxiliary_cv.features import check_feature_atoms
                _aux_table = load_aux_state_table(args.aux_cv_model, (window_metadata or {}).get("aux_rows") or [])
                if _aux_table.n != len(centers_a):
                    raise RuntimeError(f"auxiliary state table has {_aux_table.n} rows for {len(centers_a)} windows")
                check_feature_atoms(_aux_table.model, topology)
                print("WARNING: auxiliary-CV states active (--aux-cv-allow-unpersisted): z values are not "
                      "written to the sample store yet, so this is an engineering run whose samples cannot "
                      "be pooled (Stage C).", flush=True)
```

Replace the base-system block (`:7044-7047`) with:

```python
    prepare_pep_gamd_args(args, topology)
    secondary_cv_force_info = add_umbrella_cv_forces(
        openmm, base_system, topology, primary_cv_def, args,
        secondary_enabled=bool((secondary_cv_metadata or {}).get("enabled")))
    args._aux_runtime = None
    if getattr(args, "aux_cv_model", None):
        if _aux_table is None:
            raise RuntimeError("--aux-cv-model set but no auxiliary state table was loaded (resume paths are Stage C)")
        from .auxiliary_cv.runtime import add_aux_cv_force
        args._aux_runtime = add_aux_cv_force(openmm, base_system, _aux_table, args)
        print(f"    Auxiliary-CV restraint: model {args._aux_runtime.info.model_sha256[:12]}, force group "
              f"{args._aux_runtime.info.force_group}, {sum(k > 0 for k in _aux_table.k_kcal)} active of "
              f"{_aux_table.n} states")
```

After each of the two `add_umbrella_cv_forces(openmm, starting_structure_system, ...)` calls (`:7155`, `:7204`), add:

```python
                if args._aux_runtime is not None:
                    from .auxiliary_cv.runtime import add_aux_cv_force
                    add_aux_cv_force(openmm, starting_structure_system, _aux_table, args,
                                     force_group=args._aux_runtime.info.force_group)
```

Indent to match each site. Then search with `grep -n "drop_bad_us_windows_and_rebuild(" gareus/production.py` (`:7217`). Directly before that call, add the population guard:

```python
            if getattr(args, "_aux_runtime", None) is not None:
                raise RuntimeError("auxiliary-CV state population is frozen: US auto-drop must not run "
                                   "(the CLI refuses --us-auto-drop-bad-windows with --aux-cv-model)")
```

Place this guard inside the branch that performs the drop (the `if` that decides a drop is needed), so the off path is untouched.

`_build_context_i` deserializes `base_system` for each replica, so replicas inherit the aux force at `force_index`; Task 6's name check enforces this. Confirm that `prepare_pep_gamd_args` stays before the aux force: it only records peptide atoms; `ensure_pep_gamd_partition` runs inside `make_gamd_integrator`, after all bias forces exist, which `verify_pep_gamd_bias_force_groups` checks.

- [ ] **Step 4: Run the Stage B exit gate**

Run: `tests/test_aux_cv_cli.py tests/test_aux_cv_state_table.py tests/test_aux_cv_runtime.py tests/test_aux_cv_set_window.py tests/test_aux_cv_composition.py tests/test_aux_cv_observation.py tests/test_aux_cv_production_wiring.py tests/test_aux_cv_exchange.py tests/test_aux_cv_kernel_identity.py`, plus the Stage A set (`tests/test_aux_cv_model.py tests/test_aux_cv_features.py tests/test_aux_cv_evaluate.py tests/test_aux_cv_force.py tests/test_aux_cv_state_schema.py tests/test_aux_cv_bias.py`), plus the legacy regressions `tests/test_exchange_kernel_exact.py tests/test_sample_before_exchange_ordering.py tests/test_lambda_ladder_states.py tests/test_pep_gamd_boost.py tests/test_npt_coupled_target.py tests/test_contact_map_cv1_production.py tests/test_gamd_boost_default.py`.
Expected: all PASS.

- [ ] **Step 5: Add the handoff section to `CLAUDE.md`** (after the Stage A section)

```markdown
## CVaux Stage B (plain-run production wiring; spec 2026-10-07-auxiliary-cv-gibbs-production-spec Sections 3.3, 3.4, 4)

- `--aux-cv-model PATH` (+ required `--aux-cv-allow-unpersisted` until Stage C) on a plain run with `--windows-2d-csv` rows carrying `aux_k_kcal_mol` (explicit 0 for ordinary/sham), `aux_center`, optional instance columns. Refused: stock gamd boost types (their factory moves every force to group 0), `--exchange-mode neighbor`, `--resume/--extend`, `--us-auto-drop-bad-windows`, adaptive production. A seed-reachability drop is fatal.
- `gareus/auxiliary_cv/runtime.py`: force group allocated by audit (never 0-2, 29, 31), `add_aux_cv_force` on base + starting-structure systems before any integrator, `observe_aux_z` (fast = offset + sum of sub-CVs; slow = positions; non-finite fatal), `aux_bias_matrix_kcal`. `set_window(..., aux_state=)` sets the complete target at all five production call sites. `assemble_bias_matrices(..., aux_bias_kcal=)` in both sample and exchange paths (AST-pinned); off path bitwise legacy.
- Kernel `state_bias_matrix_v3_aux` + `aux_model_sha256` in the identity/method settings only with an aux model; legacy digest pinned.
- Proven: aux gradient applied once at full strength under an active Pep-GaMD boost (vs a constant external force in the same group; boosted copy differs), NPT trial energy includes aux at trial geometry, matrix = direct Context energies incl. all four swap terms, exact Gibbs/pair detailed balance and sweep stationarity with aux + duplicate sham states, no candidate mask.
- Not yet: z/feature storage, checkpoint binding, pooled MBAR (Stage C); US-pull ramp of aux (Stage D admission).
```

- [ ] **Step 6: Commit**

```bash
git add gareus/production.py CLAUDE.md tests/test_aux_cv_production_wiring.py
git commit -m "feat(cvaux): run_gareus builds the auxiliary runtime; frozen-population guards; Stage B handoff"
```

---

## Self-review record

- **Spec coverage, Stage B** (spec 16: "Wire the exact term into every Context, complete state application, exchange matrices, sampling and NPT. Preserve the existing Gibbs proposal semantics. Add canonical direct-energy tests and exact small-permutation invariance tests."):

  | Requirement | Task |
  |---|---|
  | Every Context carries the capability | Tasks 3, 9 |
  | Complete state application | Task 4 |
  | Exchange matrices, sample = exchange | Task 6 |
  | Sampling observation of z per carrier | Tasks 5, 6 |
  | NPT | Task 4 |
  | Gibbs semantics | Task 7 (unchanged kernel, proven on aux matrices) |
  | Direct energies | Task 7 |
  | Permutations | Task 7 |
  | Force-group audit | Task 3 |
  | Integrator expression audit | Task 4 gradient-once |
  | Missing z fatal | Tasks 5, 6 |
  | Loader parses aux + instance columns | Task 2 |
  | v3_aux only with capability | Task 8 |

- **Section 17 rows covered:** zero auxiliary strength on every state (Tasks 6, 8, plus the Stage A zero-k tests); active → ordinary → active (Task 4); force-group composition (Task 4); direct cross-energy check (Task 7); ordinary-origin samples (Task 5/7: z observed for k = 0 carriers and used in active rows); exact Gibbs permutations (Task 7); global candidates (Task 7); duplicate Hamiltonians in exchange (Task 7); NPT trial (Task 4).
- **Rows left to later stages:**
  - Duplicate-Hamiltonian MBAR merging, missing feature or changed model on resume/analysis, multiple swaps at one timestamp in the episode parser: Stages C/D.
  - Model parity on c10 frames: Stage D (needs the recovered artifact).
  - Reduced auxiliary-energy precision on a mixed-precision platform: not measured here; Reference/CPU only. A CUDA/OpenCL parity run belongs in the Stage C exit gate on aurum2.
- **Placeholder scan:** none. `LEGACY_DIGEST` is pinned (computed on dc30285), and Task 8 Step 1b only re-confirms it.
- **Type consistency:**
  - `AuxRuntime(table, info, force_index)` is used identically in Tasks 3-9.
  - `aux_bias_matrix_kcal(z_values, table)` takes the `AuxStateTable`, not the runtime, in Tasks 6 and 7.
  - `set_window(..., aux_state=AuxRuntime)` in Tasks 4 and 9.
  - `observe_aux_z(context, runtime, *, force=None, positions_nm=None)` in Tasks 5-7.
- **Findings recorded for the parent:**
  1. The stock gamd-openmm factory really does call `set_all_forces_to_group(system, 0)` (`gamd/integrator_factory.py:198`). The existing umbrella and CV2 forces are therefore also in group 0, and boosted, under stock boost types, and `fast_cv_force_indices`' group lookup cannot find them there. That is pre-existing and outside this plan; the aux feature simply refuses stock types.
  2. Spec 3.3 asks to "invalidate cached state-dependent values" after a state update. The code already does this: `observable_cache.clear()` runs after an accepted swap, and the sample path recomputes everything.
  3. With aux on, a Stage B run's Parquet rows carry no z. The Stage C plan must make analysis refuse `state_bias_matrix_v3_aux` segments without z columns, and must remove `--aux-cv-allow-unpersisted`.
