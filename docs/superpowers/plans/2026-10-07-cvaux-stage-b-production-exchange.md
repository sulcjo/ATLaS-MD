# CVaux Stage B: Production Hamiltonians and Unrestricted Exchange — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Revision 2 (2026-10-08), after adversarial verification.**
- Binding cross-stage decisions: `/home/sulcjo/.claude/jobs/969c720f/tmp/findings/DECISIONS.md` (D1-D14).
- Verifier findings this revision answers: `stage_B_verifier.md` 1-13 and `seams_verifier.md` 1, 10, 11, 12, 14, 15 (same directory).
- The finding-to-change map is in the self-review record at the end.

**Goal:** Wire the exact Stage A auxiliary restraint into every production Context of a plain (non-adaptive) run. Dynamics, every state assignment, the sample-path and exchange-path bias matrices, and NPT trial energies must all use the same complete Hamiltonian. Gibbs proposals stay unrestricted, with their MH correction. Data written by such a run is marked **unanalysable**: Stage B stores no z, so no loader may treat its samples as eligible. Stage C lifts that.

**Architecture:**
- **New module `gareus/auxiliary_cv/runtime.py`:**
  - force-group audit and allocator;
  - `add_aux_cv_force`, which builds the Stage A force into a System before any integrator exists;
  - `observe_aux_z` (fast or slow path), which never raises;
  - `aux_bias_matrix_kcal`, the [state, replica] auxiliary bias, which raises on non-finite z needed by an active state;
  - `deactivate_aux_parameters`, used by GaMD recon/calibration;
  - `refuse_aux_population_change`, the frozen-population guard;
  - `aux_snapshot_rows`, which adds aux fields to the legacy window snapshot rows;
  - `canonical_topology_sha256`, the run topology identity of the model's atom map.
- **New module `gareus/auxiliary_cv/state_table.py`:** turns the explicit window CSV's aux and instance columns into a frozen `AuxStateTable`.
- **`production.py` / `windows.py` changes:**
  - `set_window` gains `aux_state=`;
  - `assemble_bias_matrices` gains `aux_bias_kcal=`;
  - the two `run_gareus` fetch closures share one z helper.
- **Unchanged:** the exchange kernel (`_gibbs_window_proposal_distribution`, `gibbs_propose_one_replica`, `apply_window_swap`). It already consumes only the complete matrix.
- **Stage-B-era data is never analysable (D5):**
  - the window snapshot rows carry the aux fields, so strict reconstruction fails closed with `MissingCoordinateError`;
  - the kernel classifier returns a new ineligible status `aux_unpersisted` for every `state_bias_matrix_v3_aux` segment;
  - resume or extend of an aux campaign is refused unconditionally (D6).
- **Deferred to Stage C:** storage (Parquet z/feature columns), checkpoint/resume binding, pooled analysis. Until Stage C lands, an aux run must be acknowledged as an engineering run (`--aux-cv-allow-unpersisted`).

**Tech Stack:** Python 3, NumPy, OpenMM 8.5 (Reference/CPU in tests), pytest.

**Spec:** `docs/superpowers/specs/2026-10-07-auxiliary-cv-gibbs-production-spec.md` (main dc30285). Read Sections 3.3, 3.4, 4 (all), 5 (last paragraph), 10, 15 and 16 Stage B, plus the Section 17 rows listed in the self-review.

**Depends on:** `docs/superpowers/plans/2026-10-07-cvaux-stage-a-definitions-evaluators.md` (revision 2, after D1-D4), which must be merged first. This plan consumes these Stage A names exactly:

| Name | Notes |
|---|---|
| `AuxModel` | `.load`, `.model_sha256`, `.offset`, `.scale` (D1: z = (offset + Σ c_j f_j) / scale), `.feature_schema` |
| `build_aux_force(openmm, model, *, force_group) -> (force, AuxForceInfo)` | |
| `AuxForceInfo(name, force_group, model_sha256, global_k, global_c, sub_cv_names)` | |
| `set_aux_parameters(context, info, *, center, k_kcal)`, `AUX_FORCE_NAME` | |
| `z_from_positions(xyz_nm, model)`, `aux_energy_kj`, `aux_forces_kj_nm` | |
| `check_feature_atoms(model, topology, *, topology_sha256=None)` | |
| `gareus.correctness.bias._aux_term` | private Stage A helper; Stage B keeps the dependency deliberately, so both stages canonicalise identically |
| `gareus.correctness.state_identity._canonical_instance` | private, same reason as `_aux_term`. D3: `spawn_parent_state_id` is the parent's `state_instance_id` string or null |
| `gareus.correctness.bias.reconstruct_bias_matrix(..., aux_z=)` | |
| `tests/aux_cv_fixture.py` | `model_payload(..., blocks=, scale=1.0)`, `dipeptide()` |

**Test runner (user policy):** tests in this repo are run by the local free runner `opencode`, never directly. Each "Run:" step below names the pytest arguments. Dispatch them as
`opencode run "In /run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler: run python -m pytest -q <arguments> and report pass/fail/error counts and every failing test id. Read-only: do not edit, commit or fix anything."`
and treat its report as the result. A Bash hook in this harness refuses commands that call the test runner directly. **Targeted tests only:** never the whole suite.

## Global Constraints

- **Off path byte-identical.** With `--aux-cv-model` unset, nothing changes:
  - no new force;
  - the same `bias_matrix_kj` arithmetic (`assemble_bias_matrices` keeps its exact old expression when `aux_bias_kcal is None`);
  - the same window snapshot rows;
  - the same `EXCHANGE_ENERGY_VERSION` `state_bias_matrix_v2`;
  - the same kernel-identity digest, pinned;
  - the same `method_settings`;
  - the same `classify_segment_kernel` result for every existing snapshot.
- **Units.** Energies are kJ/mol in OpenMM and in the exchange kernel; state tables use kcal/mol. Convert **once** by 4.184.
- **Inactive restraint** (`aux_k == 0`): exactly zero energy, and zero force at non-degenerate geometry (D4). The centre global is reset to 0.0 (Stage A `set_aux_parameters`).
- **Every carrier Context carries the auxiliary force capability.** State assignment activates or deactivates it; a Context is never rebuilt on entry to an auxiliary state (spec 3.3).
- **No fixed force group.** The group is allocated by auditing the System's existing groups. Excluded: 0, 1 and 2 (Pep-GaMD physical/aux), `umbrella_force_group` and `secondary_cv_force_group`. The group is recorded in the runtime info.
- **The auxiliary force enters the integrator once, at full strength, outside the boost channels.**
  - Stock gamd-openmm boost types are refused. `GamdIntegratorFactory.get_integrator` calls `set_all_forces_to_group(system, 0)` for every stock type (`gamd/integrator_factory.py:198`, verified 2026-10-07). That destroys the auxiliary force's dedicated group; for total and dual types it also boosts the restraint.
  - Pep-GaMD (`pep-gamd-*`) and `--run-mode cmd` are allowed.
- **A z needed by an active state must be finite.** A non-finite z needed by an active state in the live exchange or sample matrix is fatal (`AuxObservationError`, raised by `aux_bias_matrix_kcal`). It is never zero-filled and never removes a candidate.
- **Live degenerate geometry fails the segment (spec §3.3, board condition 1).** Whenever any state has `aux_k > 0`, every z observation, on both the sample path and the exchange path, also checks the model's torsions for degeneracy on that carrier's positions (`check_aux_geometry`, Task 5).
  - The rule is the same as Stage A's `openmm_dihedrals` NaN rule: |b1×b2|² or |b2×b3|² below `DEGENERATE_CROSS2_NM4`.
  - A degenerate torsion raises `AuxObservationError` and fails the segment with diagnostics.
  - A sham-only or ordinary-only population (no active state) does not read positions for this and never raises.
  - `observe_aux_z` records the value whatever it is (a sham arm with every k = 0 may record NaN).
  - The fast path cannot detect a degenerate torsion: OpenMM returns a finite `theta` there. Detection of such frames is Stage C's offline parity: positions-derived NaN vs stored finite gives a finite-mismatch refusal.
- **Gibbs eligibility is unrestricted.** No parent-only mask, neighbour mask or overlap filter. `--exchange-mode neighbor` (geometry graph) is refused with an aux model; `gibbs-walk`, `all-pair-sweep` and `random-pair` are allowed.
- **Aux is plain-run only (D7).**
  - Allowed: `--window-mode adaptive` or `manual` with `--windows-2d-csv`.
  - Refused: every other window mode (`adaptive-feedback`, `adaptive-production`, `double-adaptive`, `delaunay-feedback`) and `--swarm-stage` other than `off`.
- **The state population is frozen for the run.** Each of these is fatal through `refuse_aux_population_change`:
  - window auto-drop;
  - the seed-reachability filter dropping a window;
  - `--max-replicas` truncation below the table size.
- **GaMD envelope recon and calibration run with the auxiliary restraint inactive** (k = 0 on every recon/calibration Context, D7). W and B arms therefore calibrate identical envelopes from identical inputs. This is recorded as `method_settings["aux_envelope_calibration"] = "aux_inactive"`.
- **Version gating.** Kernel version `state_bias_matrix_v3_aux` applies only when an aux model is configured, even if every `aux_k` is 0 (the sham arm). Legacy runs keep v2.
  - `classify_segment_kernel` returns `aux_unpersisted`, ineligible, for every v3_aux snapshot that lacks Stage C's `aux_sample_schema` snapshot key.
  - It never returns `verified` for v3_aux in Stage B.
- **Stage B refuses resume/extend of an aux campaign (D6).** The check reads the newest segment's `windows/<seg>.json` `kernel_identity` unconditionally. It does not depend on the run manifest, which `initialize_run_manifest` rebuilds from the current args every job, and it does not depend on CV2 being enabled.
- Float64 for every auxiliary value.

## Review Focus

1. **Active → ordinary → active on one Context.** After leaving an auxiliary state, the previous centre and strength must not linger. Task 4 test.
2. **Stage B data entering an analysis.** The off-path analysis and the strict reconstruction must both refuse it. Task 8 test (classifier: `aux_unpersisted` for residual and non-residual snapshots; `segment_eligibility` excludes it) and Task 9 test (snapshot rows with aux fields make strict reconstruction raise `MissingCoordinateError`).
3. **Resume or extend of an aux campaign without `--aux-cv-model`.** Context checkpoints load silently into a Context without the aux force (verified 2026-10-08). Only the snapshot-based refusal stops it. Task 8 test (refusal helper) and Task 9 structural test (called first in the `fast_resume` branch).
4. **Duplicate Hamiltonians (sham = parent).** Identical bias rows. The Gibbs kernel must stay detailed-balanced and both instances must remain separate slots. Task 7 enumeration with a duplicate row.
5. **Window renumbering after load.** Seed-reachability drop, US auto-drop or `--max-replicas` truncation would desynchronise the aux table from window indices. Task 1 CLI refusal, plus the Task 3 unit tests of `refuse_aux_population_change` and the Task 9 structural test that all three sites call it.

---

## File Structure

| File | Responsibility |
|---|---|
| Create `gareus/auxiliary_cv/state_table.py` | `AuxStateTable`; parse aux and instance CSV columns; load and validate the table against the model (role, unique ids, known parent ids) |
| Create `gareus/auxiliary_cv/runtime.py` | Force-group audit and allocator, `AuxRuntime`, `add_aux_cv_force`, `observe_aux_z`, `aux_bias_matrix_kcal`, `deactivate_aux_parameters`, `refuse_aux_population_change`, `aux_snapshot_rows`, `canonical_topology_sha256`, `AuxObservationError` |
| Modify `gareus/cli.py` | `--aux-cv-model`, `--aux-cv-allow-unpersisted`, `_validate_aux_cv_args`; registered in both `parse_args` and `build_gareus_parser` |
| Modify `gareus/windows.py` | `AUX_CSV_COLUMNS` / `INSTANCE_CSV_COLUMNS` constants (import-light home); `set_window(..., aux_state=None)`; `load_explicit_2d_window_csv` collects aux and instance columns into `window_metadata["aux_rows"]` |
| Modify `gareus/production.py` | `assemble_bias_matrices(..., aux_bias_kcal=None)`; aux force on base and starting-structure systems; `aux_state=` at all six window applications; recon/calibration deactivation; shared z helper in both fetch closures; aux matrix in sample and exchange assembly; snapshot rows with aux fields; population guards; `refuse_resume_of_aux_campaign`; Stage B warning; loaded-table CSV fieldnames union |
| Modify `gareus/kernel_identity.py`, `gareus/provenance.py` | `exchange_energy_version_for_args`, `state_bias_matrix_v3_aux`, `ELIGIBLE_AUX_UNPERSISTED`, aux model/topology sha in the identity, aux sha + envelope policy in method settings |
| Create `tests/test_aux_cv_cli.py`, `test_aux_cv_state_table.py`, `test_aux_cv_runtime.py`, `test_aux_cv_composition.py`, `test_aux_cv_set_window.py`, `test_aux_cv_observation.py`, `test_aux_cv_exchange.py`, `test_aux_cv_kernel_identity.py`, `test_aux_cv_production_wiring.py` | One test file per task |

---

### Task 1: CLI flags and refusals

**Files:**
- Modify: `gareus/cli.py`:
  - new `_add_aux_cv_args` and `_validate_aux_cv_args`;
  - register in `parse_args` (`gareus/cli.py:2099-2185`) **and** `build_gareus_parser` (`gareus/cli.py:2072-2096`).
- Test: `tests/test_aux_cv_cli.py`

**Interfaces:**
- Produces: `args.aux_cv_model: str | None` and `args.aux_cv_allow_unpersisted: bool`.
- The YAML keys `aux_cv_model` / `aux_cv_allow_unpersisted` are valid because `build_gareus_parser` registers the flags. That function is what config-key validation enumerates.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_aux_cv_cli.py
import pytest

from gareus.cli import build_gareus_parser, parse_args

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


def test_manual_window_mode_is_accepted(tmp_path):
    assert _parse(OK + ["--window-mode", "manual"], tmp_path).aux_cv_model == "m.json"


def test_pep_gamd_boost_is_accepted(tmp_path):
    extra = [x for x in OK if x not in ("--run-mode", "cmd")] + ["--run-mode", "gamd",
                                                                 "--gamd-boost-type", "pep-gamd-lower-dual"]
    assert _parse(extra, tmp_path).aux_cv_model == "m.json"


def test_flags_are_known_config_keys():
    dests = {a.dest for a in build_gareus_parser()._actions}
    assert {"aux_cv_model", "aux_cv_allow_unpersisted"} <= dests


@pytest.mark.parametrize("change, message", [
    (lambda o: [x for x in o if x not in ("--windows-2d-csv", "w.csv")], "windows-2d-csv"),
    (lambda o: [x for x in o if x not in ("--run-mode", "cmd")] + ["--run-mode", "gamd",
                                                                   "--gamd-boost-type", "lower-dihedral"], "group 0"),
    (lambda o: [x for x in o if x not in ("--run-mode", "cmd")] + ["--run-mode", "gamd",
                                                                   "--gamd-boost-type", "lower-dual"], "group 0"),
    (lambda o: [x for x in o if x not in ("--exchange-mode", "gibbs-walk")] + ["--exchange-mode", "neighbor"], "neighbor"),
    (lambda o: o + ["--resume"], "Stage C"),
    (lambda o: o + ["--extend"], "Stage C"),
    (lambda o: o + ["--us-auto-drop-bad-windows"], "auto-drop"),
    (lambda o: o + ["--window-mode", "adaptive-production"], "plain-run"),
    (lambda o: o + ["--window-mode", "adaptive-feedback"], "plain-run"),
    (lambda o: o + ["--window-mode", "double-adaptive"], "plain-run"),
    (lambda o: o + ["--window-mode", "delaunay-feedback"], "plain-run"),
    (lambda o: o + ["--swarm-stage", "run"], "swarm"),
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
_AUX_PLAIN_WINDOW_MODES = ("adaptive", "manual")


def _add_aux_cv_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--aux-cv-model", default=None, metavar="PATH",
                   help="Frozen auxiliary-CV model (atlas-aux-cv-model-v1 JSON). Enables auxiliary-CV "
                        "states: every row of --windows-2d-csv must state aux_k_kcal_mol (0 = ordinary or "
                        "sham) and active rows aux_center. Plain runs only. Spec "
                        "docs/superpowers/specs/2026-10-07-auxiliary-cv-gibbs-production-spec.md.")
    p.add_argument("--aux-cv-allow-unpersisted", action="store_true", default=False,
                   help="Acknowledge that this version does not yet write auxiliary z values to the "
                        "sample store: the run is an engineering run whose samples are marked "
                        "ineligible for analysis (Stage C removes this flag).")


def _validate_aux_cv_args(p: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Refuse every configuration the Stage B auxiliary-state machinery cannot run exactly."""
    if not getattr(args, "aux_cv_model", None):
        if getattr(args, "aux_cv_allow_unpersisted", False):
            p.error("--aux-cv-allow-unpersisted needs --aux-cv-model")
        return
    window_mode = str(getattr(args, "window_mode", "adaptive") or "adaptive")
    if window_mode not in _AUX_PLAIN_WINDOW_MODES:
        p.error(f"--aux-cv-model is plain-run only (--window-mode {' or '.join(_AUX_PLAIN_WINDOW_MODES)} with "
                f"--windows-2d-csv); --window-mode {window_mode} rewrites or regenerates the window table "
                "without auxiliary columns. Adaptive admission is a later stage")
    if str(getattr(args, "swarm_stage", "off") or "off") != "off":
        p.error("--aux-cv-model cannot run inside the swarm stage (--swarm-stage must be off)")
    if not getattr(args, "windows_2d_csv", None):
        p.error("--aux-cv-model needs --windows-2d-csv: auxiliary states are rows of an explicit state table")
    run_mode = str(getattr(args, "run_mode", "gamd") or "gamd")
    boost = str(getattr(args, "gamd_boost_type", "") or "")
    if run_mode in ("gamd", "hmr-gamd") and not boost.startswith("pep-gamd"):
        p.error(f"--aux-cv-model with --gamd-boost-type {boost!r}: every stock gamd-openmm integrator first "
                "moves all forces to group 0 (gamd/integrator_factory.py set_all_forces_to_group), which "
                "destroys the auxiliary restraint's own force group (and, for total/dual boost types, boosts "
                "it); use a pep-gamd-* boost type or --run-mode cmd")
    if str(getattr(args, "exchange_mode", "neighbor")) == "neighbor":
        p.error("--aux-cv-model needs unrestricted exchange candidates (gibbs-walk, all-pair-sweep or "
                "random-pair): --exchange-mode neighbor builds its graph from CV1/CV2 geometry and cannot "
                "represent auxiliary states")
    if getattr(args, "resume", False) or getattr(args, "extend", False):
        p.error("--aux-cv-model cannot --resume/--extend yet: checkpoint binding of auxiliary states is Stage C")
    if bool(getattr(args, "us_auto_drop_bad_windows", False)):
        p.error("--aux-cv-model refuses --us-auto-drop-bad-windows: the state table is frozen and a population "
                "change is a new phase (spec Section 6)")
    if not getattr(args, "aux_cv_allow_unpersisted", False):
        p.error("--aux-cv-model does not yet write auxiliary z to the sample store (Stage C); pass "
                "--aux-cv-allow-unpersisted to acknowledge an engineering run whose samples are marked "
                "ineligible for analysis")
```

In `parse_args`:
- call `_add_aux_cv_args(p)` right after `_add_window_args(p)`;
- call `_validate_aux_cv_args(p, args)` immediately after `_apply_v2_compat_shims(args)`, **before** the other validators. A window-mode or swarm refusal must not be pre-empted by an unrelated validator's `ValueError`.

In `build_gareus_parser`, call `_add_aux_cv_args(p)` right after `_add_window_args(p)`.

- [ ] **Step 4: Run to verify pass**

Run: `tests/test_aux_cv_cli.py tests/test_gamd_boost_default.py`, plus `grep -l "build_gareus_parser" tests/*.py` (run each file listed; they pin the known-key set and help output).
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
- Modify: `gareus/windows.py`:
  - add the column constants near the top, after the imports;
  - update `load_explicit_2d_window_csv` (currently `gareus/windows.py:1242-1473`).
- Modify: `gareus/production.py:6843-6850`. The `explicit_2d_window_table_loaded.csv` writer builds its fieldnames from all rows, not row 0.
- Test: `tests/test_aux_cv_state_table.py`

**Interfaces:**
- Consumes:
  - `AuxModel.load`;
  - `gareus.correctness.bias._aux_term(window, label) -> (sha|None, center, k)`;
  - `gareus.correctness.state_identity._canonical_instance(raw, state_id)` (Stage A Task 5, D3).
- Produces:
  - in `gareus/windows.py` (import-light: windows.py must not import `gareus.auxiliary_cv`; Stage A's package import pulls in `cv_selection.contracts` → `gareus.production`):
    - `AUX_CSV_COLUMNS = ("aux_center", "aux_k_kcal_mol", "aux_model_sha256")`;
    - `INSTANCE_CSV_COLUMNS = ("state_instance_id", "state_role", "spawn_parent_state_id", "matched_additional_slot_id", "spawn_source_observation_json")`;
  - `state_table.py` re-exports both constants;
  - `@dataclass(frozen=True) class AuxStateTable(model: AuxModel, centers: tuple[float, ...], k_kcal: tuple[float, ...], instances: tuple[dict | None, ...])`, with property `n` and method `window_rows() -> list[dict]` (the v2 schema fields per window);
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
        [0.4, 25.0, 1.5, 1.2, "aux-0", "auxiliary", "ord-1", "slot-0"],
        [0.4, 25.0, "", 0.0, "sham-0", "sham", "ord-1", "slot-0"]]


def _inst(**over):
    base = {"aux_k_kcal_mol": "0", "state_instance_id": "x", "state_role": "ordinary",
            "spawn_parent_state_id": "", "matched_additional_slot_id": "", "spawn_source_observation_json": ""}
    base.update(over)
    return base


def test_loader_collects_aux_rows_and_table_parses(tmp_path):
    from gareus.windows import load_explicit_2d_window_csv
    m, mpath = _model(tmp_path)
    *_rest, wmeta = load_explicit_2d_window_csv(_args(tmp_path, mpath), _csv(tmp_path, ROWS, HEADER))
    table = load_aux_state_table(mpath, wmeta["aux_rows"])
    assert table.n == 4
    assert table.k_kcal == (0.0, 0.0, 1.2, 0.0) and table.centers == (0.0, 0.0, 1.5, 0.0)
    assert [i["state_instance_id"] for i in table.instances] == ["ord-0", "ord-1", "aux-0", "sham-0"]
    assert table.instances[2]["spawn_parent_state_id"] == "ord-1"
    rows = table.window_rows()
    assert rows[2]["aux_model_sha256"] == m.model_sha256 and rows[0]["aux_model_sha256"] is None


def test_loaded_rows_keep_aux_cells_of_every_row(tmp_path):
    """Row 0 is ordinary (no aux_center cell); the active row's aux_center must survive normalisation."""
    from gareus.windows import load_explicit_2d_window_csv
    _m, mpath = _model(tmp_path)
    *_rest, wmeta = load_explicit_2d_window_csv(_args(tmp_path, mpath), _csv(tmp_path, ROWS, HEADER))
    keys = set().union(*(r.keys() for r in wmeta["normalized_rows"]))
    assert {"aux_center", "aux_k_kcal_mol", "state_instance_id"} <= keys


def test_legacy_table_has_no_aux_rows_and_does_not_import_the_aux_package(tmp_path):
    import subprocess, sys, textwrap
    code = textwrap.dedent(f"""
        import sys
        from gareus.cli import parse_args
        from gareus.windows import load_explicit_2d_window_csv
        from pathlib import Path
        p = Path({str(tmp_path)!r}) / "legacy.csv"
        p.write_text("primary_cv_center,primary_cv_k_kcal\\n0.2,25.0\\n0.4,25.0\\n")
        a = parse_args(["--seq", "GA", "--out", {str(tmp_path / "o")!r}, "--cv1", "contacts"])
        *_r, wmeta = load_explicit_2d_window_csv(a, p)
        assert "aux_rows" not in wmeta
        assert "aux_k_kcal_mol" not in wmeta["normalized_rows"][0]
        assert "gareus.auxiliary_cv" not in sys.modules, "off path imported the aux package"
    """)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


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
    (_inst(aux_k_kcal_mol="1.0", aux_center="0.2", state_role="sham"), "role"),
    (_inst(aux_k_kcal_mol="1.0", aux_center="0.2", state_role="ordinary"), "role"),
    (_inst(state_role="auxiliary"), "role"),
    (_inst(spawn_source_observation_json='{"run": "r"}'), "spawn_source_observation"),
])
def test_row_refusals(row, message, tmp_path):
    m, _ = _model(tmp_path)
    with pytest.raises((IntegrityError, ValueError), match=message):
        parse_aux_csv_row(row, 2, m.model_sha256)


def test_spawn_source_observation_json_round_trips(tmp_path):
    m, _ = _model(tmp_path)
    obs = {"run": "r", "segment": "s", "carrier": 3, "state": 1, "checkpoint": "c", "step": 100}
    out = parse_aux_csv_row(_inst(aux_k_kcal_mol="1", aux_center="0.5", state_instance_id="aux-0",
                                  state_role="auxiliary", spawn_parent_state_id="ord-1",
                                  matched_additional_slot_id="slot-0",
                                  spawn_source_observation_json=json.dumps(obs)), 2, m.model_sha256)
    assert out["instance"]["spawn_source_observation"] == obs
    assert out["instance"]["spawn_parent_state_id"] == "ord-1"


def test_table_refuses_partial_instances_duplicate_ids_and_unknown_parents(tmp_path):
    _m, mpath = _model(tmp_path)
    with pytest.raises(IntegrityError, match="duplicate"):
        load_aux_state_table(mpath, [_inst(), _inst()])
    with pytest.raises(IntegrityError, match="every row"):
        load_aux_state_table(mpath, [_inst(), {"aux_k_kcal_mol": "0"}])
    with pytest.raises(IntegrityError, match="parent"):
        load_aux_state_table(mpath, [_inst(state_instance_id="a"),
                                     _inst(state_instance_id="b", spawn_parent_state_id="nope")])
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
from ..windows import AUX_CSV_COLUMNS, INSTANCE_CSV_COLUMNS  # noqa: F401  (re-exported)
from .model import AuxModel


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
        obs = _cell(row, "spawn_source_observation_json")
        instance = _canonical_instance({
            "state_instance_id": _cell(row, "state_instance_id"),
            "state_role": _cell(row, "state_role"),
            "spawn_parent_state_id": _cell(row, "spawn_parent_state_id") or None,
            "spawn_source_observation": json.loads(obs) if obs else None,
            "matched_additional_slot_id": _cell(row, "matched_additional_slot_id") or None,
        }, offset)
        # D3: auxiliary <=> aux_k > 0; ordinary and sham require aux_k == 0. Stage A enforces the
        # same rule in the v2 canonicalisation; checking here gives a CSV-row error message.
        if (instance["state_role"] == "auxiliary") != (k > 0):
            raise IntegrityError(f"{label}: state_role {instance['state_role']!r} contradicts aux_k {k} "
                                 "(auxiliary iff aux_k > 0; ordinary and sham need aux_k == 0)")
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
    known = set(ids)
    for p in with_instance:
        parent = p["instance"]["spawn_parent_state_id"]
        if parent is not None and parent not in known:
            raise IntegrityError(f"state {p['instance']['state_instance_id']!r}: spawn parent {parent!r} is "
                                 "not a state_instance_id of this (frozen, plain-run) table")
    return AuxStateTable(model, tuple(p["aux_center"] for p in parsed), tuple(p["aux_k"] for p in parsed),
                         tuple(p["instance"] for p in parsed))
```

- [ ] **Step 4: Columns in `gareus/windows.py` and collection in `load_explicit_2d_window_csv`**

Near the top of `gareus/windows.py`, after the imports, add:

```python
#: Auxiliary-CV columns of the explicit window CSV (spec 2026-10-07 auxiliary CV, Section 5).
#: Defined here, not in gareus.auxiliary_cv, so the legacy loader never imports that package.
AUX_CSV_COLUMNS = ("aux_center", "aux_k_kcal_mol", "aux_model_sha256")
INSTANCE_CSV_COLUMNS = ("state_instance_id", "state_role", "spawn_parent_state_id",
                        "matched_additional_slot_id", "spawn_source_observation_json")
```

Before the row loop (next to `type_counts: dict[str, int] = {}`), add:

```python
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

After `window_metadata = {...}` is built, add `if any_aux: window_metadata["aux_rows"] = aux_rows`. Leave the return tuple unchanged: aux travels in `window_metadata`, so no caller's 6-tuple unpacking changes. The loader parses nothing aux-specific; parsing happens in `run_gareus` (Task 9).

- [ ] **Step 5: Fieldnames union in the loaded-table artifact (`gareus/production.py:6843-6850`)**

Replace `writer = csv.DictWriter(handle, fieldnames=list(norm_rows[0].keys()), extrasaction="ignore")` with:

```python
                        _fields = list(dict.fromkeys(k for row in norm_rows for k in row.keys()))
                        writer = csv.DictWriter(handle, fieldnames=_fields, extrasaction="ignore")
```

On a legacy table every row has the same keys, so the header is unchanged. Rows lacking a key get an empty cell (`DictWriter` restval `""`).

- [ ] **Step 6: Run to verify pass**

Run: `tests/test_aux_cv_state_table.py tests/test_contact_map_cv1_production.py::test_window_table_kind_must_match_a_contact_map_run`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add gareus/auxiliary_cv/state_table.py gareus/windows.py gareus/production.py tests/test_aux_cv_state_table.py
git commit -m "feat(cvaux): auxiliary state table from the explicit window CSV"
```

---

### Task 3: Runtime module: force group, population guard, snapshot rows, topology identity

**Files:**
- Create: `gareus/auxiliary_cv/runtime.py`
- Test: `tests/test_aux_cv_runtime.py`

**Interfaces:**
- Consumes: `build_aux_force`, `AuxForceInfo`, `AUX_FORCE_NAME`, `set_aux_parameters` (Stage A); `AuxStateTable` (Task 2).
- Produces:
  - `RESERVED_PHYSICAL_GROUPS = frozenset({0, 1, 2})`;
  - `force_group_audit(system) -> list[dict]` with keys `index, class, name, group`;
  - `allocate_free_force_group(system, reserved=()) -> int`;
  - `@dataclass(frozen=True) class AuxRuntime(table: AuxStateTable, info: AuxForceInfo, force_index: int, topology_sha256: str | None = None)`;
  - `add_aux_cv_force(openmm, system, table, args, *, force_group: int | None = None, topology_sha256: str | None = None) -> AuxRuntime`;
  - `class AuxObservationError(IntegrityError)`;
  - `deactivate_aux_parameters(context, runtime: AuxRuntime | None) -> None`;
  - `refuse_aux_population_change(n_expected: int, n_now: int, *, cause: str) -> None`, which raises `RuntimeError`;
  - `aux_snapshot_rows(rows: list[dict], table: AuxStateTable) -> list[dict]`, which returns new rows with `aux_model_sha256`/`aux_center`/`aux_k` added;
  - `canonical_topology_sha256(topology, model) -> str` and `canonical_topology_sha256_from_pdb(path, model) -> str`, plus a `python -m gareus.auxiliary_cv.runtime topology-sha PDB MODEL` entry point.

  Tasks 5 and 6 add `observe_aux_z` and `aux_bias_matrix_kcal` to this module.

**Topology identity.** Production has no topology digest helper. The swarm's feature schemas carry `file_digest(topology.pdb)` of the swarm's own file (`gareus/swarm/analyze.py:840`), which no production run reproduces. The binding is therefore content-based:
- `canonical_topology_sha256` hashes the ordered `(index, name, element, residue name, residue index, chain index)` of every atom in the chains that hold a model feature atom.
- Solvent count does not enter, so the atom map is bound without depending on box size.
- A model deployed in production must record this digest as its `feature_schema.topology_sha256`. Stage D's model export/port computes it with the entry point above.
- `run_gareus` passes it to `check_feature_atoms(..., topology_sha256=)`, which refuses a mismatch, and records it in the kernel identity for Stage C's resume binding.

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
                                         allocate_free_force_group, aux_snapshot_rows,
                                         canonical_topology_sha256, deactivate_aux_parameters,
                                         force_group_audit, refuse_aux_population_change)
from gareus.auxiliary_cv.state_table import AuxStateTable
from gareus.correctness._io import IntegrityError

ARGS = types.SimpleNamespace(umbrella_force_group=31, secondary_cv_force_group=29)


def _table(k=(0.0, 2.0), c=(0.0, 1.0)):
    d = dipeptide()
    blocks = [lab.split("-")[0] for lab in d["labels"]]
    m = AuxModel.from_mapping(model_payload(d["quads"], [1.0] + [0.3] * (2 * len(d["quads"]) - 1),
                                            offset=0.2, blocks=blocks))
    return AuxStateTable(m, tuple(c), tuple(k), (None,) * len(k))


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
    rt = add_aux_cv_force(mm, system, _table(), ARGS, topology_sha256="b" * 64)
    assert system.getForce(rt.force_index).getName() == AUX_FORCE_NAME
    assert rt.info.force_group not in before | RESERVED_PHYSICAL_GROUPS | {29, 31}
    assert rt.topology_sha256 == "b" * 64


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


def test_deactivate_sets_k_and_center_to_zero_and_ignores_none():
    from pep_gamd_fixture import _fresh_system
    import openmm as mm
    from gareus.auxiliary_cv.force import set_aux_parameters
    system = _fresh_system()
    rt = add_aux_cv_force(mm, system, _table(), ARGS)
    ctx = mm.Context(system, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference"))
    set_aux_parameters(ctx, rt.info, center=1.0, k_kcal=2.0)
    deactivate_aux_parameters(ctx, rt)
    assert ctx.getParameter(rt.info.global_k) == 0.0 and ctx.getParameter(rt.info.global_c) == 0.0
    deactivate_aux_parameters(ctx, None)        # off path: no-op


@pytest.mark.parametrize("n_expected, n_now", [(4, 3), (4, 5)])
def test_population_change_is_refused(n_expected, n_now):
    with pytest.raises(RuntimeError, match="--max-replicas"):
        refuse_aux_population_change(n_expected, n_now, cause="--max-replicas")


def test_unchanged_population_passes():
    refuse_aux_population_change(4, 4, cause="seed-reachability filter")


def test_snapshot_rows_carry_aux_fields_and_strict_reconstruction_fails_closed():
    from gareus.correctness.bias import MissingCoordinateError, reconstruct_bias_matrix
    table = _table()
    legacy = [{"window_id": 0, "center1": 0.2, "k1": 10.0, "gamd_lambda": 0.0},
              {"window_id": 1, "center1": 0.4, "k1": 10.0, "gamd_lambda": 0.0}]
    rows = aux_snapshot_rows(legacy, table)
    assert "aux_k" not in legacy[0], "input rows must not be mutated"
    assert rows[0]["aux_k"] == 0.0 and rows[0]["aux_model_sha256"] is None
    assert rows[1]["aux_k"] == 2.0 and rows[1]["aux_model_sha256"] == table.model.model_sha256
    with pytest.raises(MissingCoordinateError):
        reconstruct_bias_matrix(np.array([0.3]), None, rows, 0.4)


def test_snapshot_rows_length_must_match():
    with pytest.raises(RuntimeError, match="rows"):
        aux_snapshot_rows([{"window_id": 0, "center1": 0.0, "k1": 1.0, "gamd_lambda": 0.0}], _table())


def test_canonical_topology_sha_is_deterministic_and_content_sensitive():
    import openmm.app as app
    d = dipeptide()
    m = _table().model
    a = canonical_topology_sha256(d["topology"], m)
    assert a == canonical_topology_sha256(d["topology"], m) and len(a) == 64

    def _toy(name):
        top = app.Topology()
        ch = top.addChain()
        res = top.addResidue("ALA", ch)
        for nm in ("N", "CA", "C", name):
            top.addAtom(nm, app.element.carbon, res)
        return top

    m4 = AuxModel.from_mapping(model_payload([(0, 1, 2, 3)], [1.0, 0.0]))
    assert canonical_topology_sha256(_toy("O"), m4) != canonical_topology_sha256(_toy("OXT"), m4)
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

import hashlib
import json
from dataclasses import dataclass
from typing import Iterable, Optional

from ..correctness._io import IntegrityError
from .force import AuxForceInfo, build_aux_force, set_aux_parameters
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
    topology_sha256: Optional[str] = None


def _reserved_groups(args) -> set[int]:
    return {int(getattr(args, "umbrella_force_group", 31)), int(getattr(args, "secondary_cv_force_group", 29))}


def add_aux_cv_force(openmm, system, table: AuxStateTable, args, *,
                     force_group: Optional[int] = None,
                     topology_sha256: Optional[str] = None) -> AuxRuntime:
    """Add the auxiliary restraint capability to ``system``; call before any integrator is built.

    The force is created with aux_k = aux_c = 0 (inactive) as its global defaults, so every Context
    built from this system starts with the restraint off until ``set_window(..., aux_state=)``.
    """
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
    return AuxRuntime(table, info, index, topology_sha256)


def deactivate_aux_parameters(context, runtime: Optional[AuxRuntime]) -> None:
    """Restraint off (k = 0, centre 0) on a recon/calibration Context; no-op without a runtime."""
    if runtime is not None:
        set_aux_parameters(context, runtime.info, center=0.0, k_kcal=0.0)


def refuse_aux_population_change(n_expected: int, n_now: int, *, cause: str) -> None:
    if int(n_now) != int(n_expected):
        raise RuntimeError(
            f"auxiliary-CV state population changed by {cause}: {n_expected} -> {n_now} windows. "
            "An auxiliary state table is frozen and cannot be renumbered (spec Section 6); "
            "fix the window table instead.")


def aux_snapshot_rows(rows: list[dict], table: AuxStateTable) -> list[dict]:
    """Legacy window-snapshot rows plus the auxiliary fields (D5): strict readers then fail closed."""
    if len(rows) != table.n:
        raise RuntimeError(f"window snapshot has {len(rows)} rows for {table.n} auxiliary states")
    out = []
    for row, aux in zip(rows, table.window_rows()):
        new = dict(row)
        new.update(aux_model_sha256=aux["aux_model_sha256"], aux_center=float(aux["aux_center"]),
                   aux_k=float(aux["aux_k"]))
        out.append(new)
    return out


def _feature_atoms(model) -> set[int]:
    return {int(a) for f in model.feature_schema.features for a in f.atom_indices}


def canonical_topology_sha256(topology, model) -> str:
    """Content identity of the atom map the model's torsions index: every atom of the chains holding a
    feature atom, as (index, name, element, residue name, residue index, chain index)."""
    wanted = _feature_atoms(model)
    atoms = list(topology.atoms())
    if wanted and max(wanted) >= len(atoms):
        raise IntegrityError(f"aux model indexes atom {max(wanted)} beyond the topology ({len(atoms)} atoms)")
    chains = {atoms[i].residue.chain.index for i in wanted}
    body = [[a.index, a.name, (a.element.symbol if a.element is not None else ""),
             a.residue.name, a.residue.index, a.residue.chain.index]
            for a in atoms if a.residue.chain.index in chains]
    payload = json.dumps({"schema": "atlas-aux-topology-v1", "atoms": body},
                         sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def canonical_topology_sha256_from_pdb(path, model) -> str:
    from openmm import app
    return canonical_topology_sha256(app.PDBFile(str(path)).topology, model)


# Entry point. ASSUMPTION (board caveat 2): the digest includes each atom's residue and chain index, so it
# must be computed on the PRODUCTION topology (same chain order, same solvent/ion chains before the
# peptide). Solvent atoms never enter the body, but inserting or reordering chains ahead of the peptide's
# chain shifts the chain/residue indices and gives a different digest. Stage D computes it from the
# production topology.pdb of the run that will deploy the model.
if __name__ == "__main__":  # python -m gareus.auxiliary_cv.runtime topology-sha PDB MODEL
    import sys
    from .model import AuxModel
    if len(sys.argv) != 4 or sys.argv[1] != "topology-sha":
        raise SystemExit("usage: python -m gareus.auxiliary_cv.runtime topology-sha PDB MODEL")
    print(canonical_topology_sha256_from_pdb(sys.argv[2], AuxModel.load(sys.argv[3])))
```

- [ ] **Step 4: Run to verify pass**

Run: `tests/test_aux_cv_runtime.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add gareus/auxiliary_cv/runtime.py tests/test_aux_cv_runtime.py
git commit -m "feat(cvaux): force-group audit, population guard, snapshot rows, topology identity"
```

---

### Task 4: `set_window` applies the complete target at all six sites; recon inactive; force-composition proofs

**Files:**
- Modify: `gareus/windows.py:193-224` (`set_window`)
- Modify: `gareus/production.py`. Six window applications:
  - `:5592` `_apply_assignment` in `load_production_checkpoint`;
  - `:5763` `run_production_probe`, which passes `set_window` as a callable to `_run`;
  - `:5891` `run_multiwindow_gamd_recon`;
  - `:6128` `apply_joint_envelope_gamd_calibration`;
  - `:7522` `_build_context_i`;
  - `:8420` `_apply_swap_to_replica`.
- Test: `tests/test_aux_cv_set_window.py`, `tests/test_aux_cv_composition.py`

**Interfaces:**
- Consumes: `AuxRuntime`, `deactivate_aux_parameters` (Task 3); `set_aux_parameters` (Stage A).
- Produces:
  - `set_window(context, centers_nm, ks_kj_nm2, window_index, secondary_centers=None, secondary_ks_kj=None, *, aux_state=None)`. When `aux_state` is an `AuxRuntime`, it sets the window's complete aux target. A missing global raises; there is no try/except.
  - Production sites pass `aux_state=` as follows:
    - `_apply_assignment`, `run_production_probe`, `_build_context_i` and `_apply_swap_to_replica` pass `aux_state=getattr(args, "_aux_runtime", None)`. `load_production_checkpoint` may receive `args=None` and uses `None if args is None else getattr(...)`.
    - The two GaMD recon/calibration sites pass `aux_state=None` and then call `deactivate_aux_parameters(context, getattr(args, "_aux_runtime", None))`. Envelopes are calibrated with the restraint off (D7).

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


def _production_tree():
    import gareus.production as production
    return ast.parse(inspect.getsource(production))


def test_every_window_application_passes_aux_state():
    """Direct set_window(...) calls AND calls that pass set_window as a callable (e.g. _run(r, set_window, ...))."""
    tree = _production_tree()
    sites = []
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        direct = isinstance(n.func, ast.Name) and n.func.id == "set_window"
        passed = any(isinstance(a, ast.Name) and a.id == "set_window" for a in n.args)
        if direct or passed:
            sites.append(n)
    assert len(sites) >= 6, "set_window applications moved; re-anchor this test"
    missing = [n.lineno for n in sites if not any(k.arg == "aux_state" for k in n.keywords)]
    assert not missing, f"set_window applications without aux_state= at lines {missing}"
    # Every bare Name reference to set_window is one of the sites above (no hidden callable use).
    names = [n for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id == "set_window"]
    assert len(names) == len(sites), "a set_window reference is neither a call nor a callable argument of one"


def _func_source(name):
    import gareus.production as production
    return inspect.getsource(getattr(production, name))


@pytest.mark.parametrize("fn", ["run_multiwindow_gamd_recon", "apply_joint_envelope_gamd_calibration"])
def test_gamd_recon_and_calibration_keep_aux_inactive(fn):
    tree = ast.parse(_func_source(fn))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == "set_window"]
    assert calls and all(any(k.arg == "aux_state" and isinstance(k.value, ast.Constant) and k.value.value is None
                             for k in c.keywords) for c in calls)
    assert any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "deactivate_aux_parameters"
               for n in ast.walk(tree)), f"{fn} must deactivate the auxiliary restraint (D7)"
```

```python
# tests/test_aux_cv_composition.py
"""The auxiliary force must enter the integrator once, at full strength, outside the boost (spec 3.3).

Design note (verifier finding B4): adding or removing a force GROUP changes the Pep-GaMD integrator
program and therefore its random stream. One step at 300 K then shifts every atom by ~1e-3 nm,
which swamps the ~1e-6 nm the restraint itself causes. Every arm below has the SAME group
structure: the aux force sits in its group in every system (k = 0 where inactive), and every
constant external force has a zero-parameter twin in the same group. Differences between paired
arms are therefore exactly the effect of the force under test.
"""
import types

import numpy as np

from aux_cv_fixture import dipeptide, model_payload
from gareus.auxiliary_cv.evaluate import aux_energy_kj, aux_forces_kj_nm, z_from_positions
from gareus.auxiliary_cv.model import AuxModel
from gareus.auxiliary_cv.runtime import add_aux_cv_force
from gareus.auxiliary_cv.state_table import AuxStateTable
from gareus.imports import import_openmm

ARGS = types.SimpleNamespace(umbrella_force_group=31, secondary_cv_force_group=29)
K_KCAL, OFFSET_TO_CENTER = 40.0, 0.5


def _table():
    d = dipeptide()
    blocks = [lab.split("-")[0] for lab in d["labels"]]
    m = AuxModel.from_mapping(model_payload(d["quads"], [1.0] + [0.5] * (2 * len(d["quads"]) - 1),
                                            offset=0.0, blocks=blocks))
    z0 = float(z_from_positions(d["positions_nm"], m)[0])
    return AuxStateTable(m, (z0 - OFFSET_TO_CENTER,), (K_KCAL,), (None,)), d


def _with_aux(system, table, active):
    openmm, _app, _unit = import_openmm()
    rt = add_aux_cv_force(openmm, system, table, ARGS)
    force = system.getForce(rt.force_index)
    force.setGlobalParameterDefaultValue(0, K_KCAL * 4.184 if active else 0.0)   # aux_k (kJ/mol per z^2)
    force.setGlobalParameterDefaultValue(1, table.centers[0] if active else 0.0)  # aux_c
    return rt


def _constant_force(system, forces_kj_nm, positions_nm, group, scale):
    """F = scale * forces_kj_nm, energy -(F . (x - x0)): zero energy at the start positions, so the
    Total-channel energy (and therefore the FSF) is identical between a pair of arms."""
    openmm, _app, _unit = import_openmm()
    f = openmm.CustomExternalForce("-(fx*(x-x0)+fy*(y-y0)+fz*(z-z0))")
    for name in ("fx", "fy", "fz", "x0", "y0", "z0"):
        f.addPerParticleParameter(name)
    for i, vec in enumerate(forces_kj_nm):
        if np.any(vec):
            f.addParticle(i, [float(scale * v) for v in vec] + [float(c) for c in positions_nm[i]])
    f.setForceGroup(group)
    system.addForce(f)


def dipeptide_peptide():
    from pep_gamd_fixture import solvated_dipeptide
    return solvated_dipeptide()["peptide"]


def test_group_energies_reconstruct_the_total_and_aux_is_a_bias_group():
    from pep_gamd_fixture import _fresh_system
    from gareus import pep_gamd
    openmm, _app, unit = import_openmm()
    table, d = _table()
    system = _fresh_system()
    pep_gamd.ensure_pep_gamd_partition(system, dipeptide_peptide())
    rt = _with_aux(system, table, active=True)
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


def test_pep_gamd_applies_the_aux_gradient_once_unscaled():
    """Paired, structure-matched arms (see module docstring), one Pep-GaMD step with an ACTIVE boost:

      aux:  (aux active)                  - (aux k = 0)                     == effect of A_s
      ext:  (const F_aux in aux group)    - (zero const in aux group)       == effect of an unboosted bias force
      bst:  (const F_aux in group 0)      - (zero const in group 0)         == effect of a Total-boosted force

    Required: aux == ext (applied once, unscaled), bst == FSF_Total * ext (the comparison can see the
    boost), and all effects non-negligible. Warm-up rule (CLAUDE.md): _one_step_positions steps once,
    re-seats the start state, then measures.
    """
    from pep_gamd_fixture import _fresh_system, solvated_dipeptide
    from gareus import pep_gamd
    from test_pep_gamd_boost import _gamd_kwargs, _one_step_positions, _assert_moved
    openmm, _app, unit = import_openmm()
    table, d = _table()
    fx = solvated_dipeptide()
    x0 = d["positions_nm"]
    f_aux = aux_forces_kj_nm(x0, table.model, table.centers[0], K_KCAL)
    assert np.abs(f_aux).max() > 10.0, "the auxiliary force must be large enough to move atoms measurably"

    def _system(kind):
        s = _fresh_system()
        pep_gamd.ensure_pep_gamd_partition(s, fx["peptide"])
        rt = _with_aux(s, table, active=(kind == "aux_on"))
        if kind in ("ext_on", "ext_off"):
            _constant_force(s, f_aux, x0, rt.info.force_group, 1.0 if kind == "ext_on" else 0.0)
        elif kind in ("bst_on", "bst_off"):
            _constant_force(s, f_aux, x0, 0, 1.0 if kind == "bst_on" else 0.0)
        return s

    probe_sys = _system("aux_off")
    probe = openmm.Context(probe_sys, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName("Reference"))
    probe.setPositions(fx["positions"])
    v_pep = pep_gamd.peptide_essential_energy_kj(probe, unit)
    v_dih = probe.getState(getEnergy=True, groups={pep_gamd.DIHEDRAL_GROUP}).getPotentialEnergy().value_in_unit(
        unit.kilojoule_per_mole)
    del probe
    stage = {"stepCount": 50, "stage": 5,
             "k0_Total": 0.5, "Vmax_Total": v_pep + 50.0, "Vmin_Total": v_pep - 50.0, "threshold_energy_Total": v_pep + 50.0,
             "k0_Dihedral": 0.5, "Vmax_Dihedral": v_dih + 50.0, "Vmin_Dihedral": v_dih - 50.0,
             "threshold_energy_Dihedral": v_dih + 50.0}
    kw = _gamd_kwargs(unit)
    out, fsf_total = {}, {}
    for kind in ("aux_on", "aux_off", "ext_on", "ext_off", "bst_on", "bst_off"):
        s = _system(kind)
        integ = pep_gamd.PepGaMDLowerDualIntegrator(pep_gamd.DIHEDRAL_GROUP,
                                                    bias_force_groups=pep_gamd.pep_gamd_bias_force_groups(s), **kw)
        out[kind] = _one_step_positions(s, integ, fx["positions"], openmm, unit, stage_globals=stage)
        fsf_total[kind] = integ.getGlobalVariableByName("ForceScalingFactor_Total")
    _assert_moved(out["aux_on"], fx["positions"], unit)
    d_aux = out["aux_on"] - out["aux_off"]
    d_ext = out["ext_on"] - out["ext_off"]
    d_bst = out["bst_on"] - out["bst_off"]
    scale = np.abs(d_ext).max()
    assert scale > 1e-8, f"the bias-group force moved nothing measurable ({scale:.3g} nm): test is vacuous"
    assert np.abs(d_aux - d_ext).max() < 1e-6 * scale, "aux gradient is not applied once at full strength"
    fsf = fsf_total["bst_on"]
    assert fsf_total["bst_off"] == fsf, "pair arms must see the same Total-channel FSF"
    assert abs(1.0 - fsf) > 0.05, f"FSF_Total {fsf:.3f} is ~1: the boosted copy could not be told apart"
    assert np.abs(d_bst - fsf * d_ext).max() < 1e-3 * scale, "boosted copy is not FSF_Total-scaled"


def test_npt_trial_energy_includes_aux_and_tracks_geometry():
    """Pep-GaMD NPT adapter: the aux group is a bias group evaluated on the live coordinates.

    A molecular-centroid volume trial translates whole molecules, so intramolecular torsions -- and
    therefore A_s -- are INVARIANT under it: an adapter that read old coordinates would still pass a
    before/after-scaling check. The geometry dependence is therefore checked by displacing one
    torsion atom (a configuration change the adapter must see). The controlled-distribution NPT test
    of spec 17 is a Stage D prerequisite (see the handoff list).
    """
    from pep_gamd_fixture import _fresh_system, solvated_dipeptide
    from gareus import npt, pep_gamd
    from gareus.pep_gamd import PepGamdLowerDualNptTargetAdapter
    from test_pep_gamd_boost import _gamd_kwargs
    openmm, _app, unit = import_openmm()
    table, d = _table()
    fx = solvated_dipeptide()
    system = _fresh_system()
    pep_gamd.ensure_pep_gamd_partition(system, fx["peptide"])
    rt = _with_aux(system, table, active=True)
    integ = pep_gamd.PepGaMDLowerDualIntegrator(pep_gamd.DIHEDRAL_GROUP,
                                                bias_force_groups=pep_gamd.pep_gamd_bias_force_groups(system),
                                                **_gamd_kwargs(unit))
    adapter = PepGamdLowerDualNptTargetAdapter(system, integ)
    assert rt.info.force_group in adapter._bias_groups
    ctx = openmm.Context(system, integ, openmm.Platform.getPlatformByName("Reference"))
    ctx.setPositions(d["positions_nm"])
    snap = adapter.snapshot(ctx, integ)
    other_bias = adapter.evaluate(ctx, snap).bias_kj_mol - aux_energy_kj(
        z_from_positions(d["positions_nm"], table.model), table.centers[0], K_KCAL)[0]
    # Molecular scaling leaves the aux energy unchanged (documented cancellation).
    mol_ids, mol_sizes = npt._molecule_index_arrays(npt._molecules_from_context(ctx), system.getNumParticles())
    trial = npt._scale_about_molecule_centroids(d["positions_nm"], mol_ids, mol_sizes, 0.02)
    assert abs(float(z_from_positions(trial, table.model)[0]) - float(z_from_positions(d["positions_nm"], table.model)[0])) < 1e-9
    # Displace one torsion atom: the adapter's bias must follow the new A_s exactly.
    moved = d["positions_nm"].copy()
    atom = table.model.feature_schema.features[0].atom_indices[0]
    moved[atom] += np.array([0.02, -0.015, 0.01])
    ctx.setPositions(moved)
    after = adapter.evaluate(ctx, snap)
    expected = aux_energy_kj(z_from_positions(moved, table.model), table.centers[0], K_KCAL)[0]
    other_after = after.bias_kj_mol - expected
    assert abs(expected - aux_energy_kj(z_from_positions(d["positions_nm"], table.model), table.centers[0], K_KCAL)[0]) > 1e-3
    # The non-aux bias groups (umbrellas) are absent here, so the remainder is unchanged (0 both times).
    assert abs(other_after - other_bias) < 1e-8
```

Notes for the implementer:
- `K_KCAL = 40` and a 0.5 offset give |F_aux| ≫ 10 kJ/mol/nm on the dipeptide torsion atoms. That puts the one-step displacement effect far above floating-point noise.
- Every arm adds the aux force through `_with_aux`, so all six systems share one bias-group set and one integrator program.
- `ForceScalingFactor_Total` is the gamd-openmm global name (`_append_group_name("ForceScalingFactor", "Total")`, `gamd/stage_integrator.py:605`). It is read after the measured step, from the same pre-step energies in both arms of a pair, so it is identical within a pair; the test asserts this.
- The verifier measured FSF_Total ≈ 0.78 with these stage globals. The Total channel sees b_d inside its square, so it is not 0.75; the test does not hard-code it.
- `PepGamdLowerDualNptTargetAdapter.evaluate(context, snapshot)` and `.snapshot(context, integrator)` are the adapter API at `gareus/pep_gamd.py:879-978`.

- [ ] **Step 2: Run to verify failure**

Run: `tests/test_aux_cv_set_window.py tests/test_aux_cv_composition.py`

Expected:
- `test_aux_cv_set_window.py` FAILs (`set_window() got an unexpected keyword argument 'aux_state'`).
- The composition tests may already PASS: they exercise Stage A plus Task 3 only, so they are proofs, not a red/green pair. If any composition test fails, stop. It means the auxiliary force is scaled by the boost or missing from NPT. That is a release blocker (spec 19, A06).

- [ ] **Step 3: Implement**

`gareus/windows.py`: extend the signature with `*, aux_state=None` and append to the body, after the secondary block:

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

In `gareus/production.py`, apply these exact replacements:

```python
# :5592 (_apply_assignment inside load_production_checkpoint)
        set_window(sim.context, centers_nm, ks_kj_nm2, int(assignments[r]), secondary_centers, secondary_ks_kj,
                   aux_state=None if args is None else getattr(args, "_aux_runtime", None))
# :5763 (run_production_probe; loadCheckpoint just restored the aux globals, but the complete target is
#        re-applied like every other window application)
            _run(r, set_window, sim.context, centers_nm, ks_kj_nm2, int(assignments[r]),
                 aux_state=getattr(args, "_aux_runtime", None))
# :5891 (run_multiwindow_gamd_recon): the envelope is calibrated with the auxiliary restraint OFF (D7)
        set_window(sim_i.context, centers_nm, ks_kj_nm2, i, secondary_cv_centers, secondary_cv_ks_kj,
                   aux_state=None)
        deactivate_aux_parameters(sim_i.context, getattr(args, "_aux_runtime", None))
# :6128 (apply_joint_envelope_gamd_calibration): same policy
        set_window(_check_sim.context, centers_nm, ks_kj_nm2, 0, secondary_cv_centers, secondary_cv_ks_kj,
                   aux_state=None)
        deactivate_aux_parameters(_check_sim.context, getattr(args, "_aux_runtime", None))
# :7522 (_build_context_i)
                set_window(sim_i.context, centers_nm, ks_kj_nm2, i, secondary_cv_centers, secondary_cv_ks_kj,
                           aux_state=getattr(args, "_aux_runtime", None))
# :8420 (_apply_swap_to_replica)
                    set_window(sims[replica_index].context, centers_nm, ks_kj_nm2, assignments[replica_index],
                               secondary_cv_centers, secondary_cv_ks_kj,
                               aux_state=getattr(args, "_aux_runtime", None))
```

Add `from .auxiliary_cv.runtime import deactivate_aux_parameters` as a **local import inside** `run_multiwindow_gamd_recon` and `apply_joint_envelope_gamd_calibration`, at the top of each function body. Do not add it at module top: `production.py` must not import the aux package on the off path.

On the off path `getattr(args, "_aux_runtime", None)` is None, so the deactivation is a no-op. On the aux path the recon/calibration Contexts are built from `base_system`, whose aux globals default to 0. The explicit call states the policy and protects against future default changes.

The swap's two `set_window` calls plus the map change already run as one transaction: `_apply_swap_to_replica` runs on both replicas before `observable_cache.clear()`, and no integration happens in between. Keep that ordering.

- [ ] **Step 4: Run to verify pass**

Run: `tests/test_aux_cv_set_window.py tests/test_aux_cv_composition.py tests/test_sample_before_exchange_ordering.py tests/test_pep_gamd_boost.py tests/test_npt_coupled_target.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add gareus/windows.py gareus/production.py tests/test_aux_cv_set_window.py tests/test_aux_cv_composition.py
git commit -m "feat(cvaux): set_window applies the complete aux target at all six sites; recon aux-inactive; composition proofs"
```

---

### Task 5: Observation of z for every carrier

**Files:**
- Modify: `gareus/auxiliary_cv/runtime.py` (add `observe_aux_z`)
- Test: `tests/test_aux_cv_observation.py`

**Interfaces:**
- Produces: `observe_aux_z(context, runtime: AuxRuntime, *, force=None, positions_nm=None) -> float`.
  - With `force`, z is read from the force's sub-CVs: z = (offset + Σ sub-CV values) / scale. This is valid because each sub-CV is a weighted trig sum (Stage A Task 4, D1).
  - With `positions_nm`, z = `z_from_positions`.
  - Exactly one of the two must be given.
  - It **never raises on a non-finite z**: it records what it observed. Fatality is decided by `aux_bias_matrix_kcal`, only when an active state needs the value.
  - The fast path returns a finite value even at a degenerate torsion (OpenMM's `theta` is finite there). The slow path returns NaN. Neither is relied on for fail-closed behaviour: that is `check_aux_geometry`'s job (below).
- Produces: `check_aux_geometry(positions_nm, runtime, *, replica=None) -> None`. It raises `AuxObservationError` naming the degenerate quads and the replica.
  - It covers every unique torsion of the model, including zero-coefficient ones: Stage A's conservative rule.
  - Its threshold and semantics are exactly those of Stage A `openmm_dihedrals`, which it calls.
- Produces: `make_aux_z_observer(runtime, *, use_fast_path: bool, fast_forces, unit) -> Callable[[int, sim], float]`. This is the single observer `run_gareus` installs as `_aux_z_for_replica` for both fetch closures (Task 6).
  - With any active state, it reads the carrier's positions once and calls `check_aux_geometry`. It then returns the fast-path z (the force's own value, authoritative) or the slow-path z.
  - With no active state, it reads positions only on the slow path, and never raises.
- **Degeneracy option chosen, and its cost (board condition 1).** Fast-path z plus a positions-based degeneracy check, rather than switching z itself to `z_from_positions`.
  - Both options need the same one `getState(getPositions=True)` per carrier per observation. OpenMM has no subset-positions read, so reading "only the torsion atoms" is not cheaper.
  - Once the positions are on the host, the check is a vectorised cross product over the model's ≤ 36 torsions: negligible.
  - Keeping the fast-path z keeps the stored/exchanged z equal to the value the integrated force actually used (Stage C D8). Switching z to positions would decouple the two.
  - Added cost: one positions transfer per carrier per sample interval, plus one per exchange interval that is not a sample step. The cached-observable shortcut reuses the sample-step observation.
  - Only aux-active populations pay it. Stage C reads the same positions for torsion storage (D8), so in Stage C the read is shared, not doubled. Stage C Task 14 benchmarks it.
- **Spec §3.3 compliance note.** The OpenMM force evaluates a finite `theta` at a degenerate torsion, so between two observations the dynamics can integrate a meaningless A_s.
  - The segment is failed at the next observation: the next sample or exchange interval, whichever comes first. This bounds the exposure to at most one observation interval of MD.
  - The step that fails is never sampled or exchanged, so no observation from a degenerate configuration enters the sample store or the exchange kernel.
  - Checking every MD step would need an in-integrator guard, which is out of scope. This bounded-exposure behaviour is the plan's reading of "fail the affected production segment with diagnostics".

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_aux_cv_observation.py
import math
import types

import numpy as np
import pytest

from aux_cv_fixture import dipeptide, model_payload
from gareus.auxiliary_cv.evaluate import z_from_positions
from gareus.auxiliary_cv.model import AuxModel
from gareus.auxiliary_cv.runtime import add_aux_cv_force, observe_aux_z
from gareus.auxiliary_cv.state_table import AuxStateTable

ARGS = types.SimpleNamespace(umbrella_force_group=31, secondary_cv_force_group=29)


def _setup(conventions=None, scale=1.0):
    import openmm as mm
    from pep_gamd_fixture import _fresh_system
    d = dipeptide()
    blocks = [lab.split("-")[0] for lab in d["labels"]]
    rng = np.random.default_rng(5)
    m = AuxModel.from_mapping(model_payload(d["quads"], rng.normal(size=2 * len(d["quads"])), offset=-0.3,
                                            conventions=conventions, blocks=blocks, scale=scale))
    system = _fresh_system()
    rt = add_aux_cv_force(mm, system, AuxStateTable(m, (0.0,), (0.0,), (None,)), ARGS)
    ctx = mm.Context(system, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference"))
    ctx.setPositions(d["positions_nm"])
    return ctx, rt, system.getForce(rt.force_index), d


@pytest.mark.parametrize("mixed, scale", [(False, 1.0), (True, 1.0), (False, 3.181)])
def test_fast_and_slow_paths_agree_for_an_ordinary_state(mixed, scale):
    """k = 0 everywhere: an ordinary carrier still yields its z for cross-evaluation (spec 3.1)."""
    d0 = dipeptide()
    conv = ["negated" if k % 2 == 0 else "direct" for k in range(len(d0["quads"]))] if mixed else None
    ctx, rt, force, d = _setup(conv, scale)
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


def test_degenerate_geometry_is_recorded_not_raised():
    """Slow path: NaN is returned (fatality belongs to aux_bias_matrix_kcal, only for active states)."""
    ctx, rt, _force, d = _setup()
    bad = d["positions_nm"].copy()
    q = rt.table.model.feature_schema.features[0].atom_indices
    bad[q[0]] = bad[q[1]]                       # collapse two torsion atoms -> undefined dihedral
    assert math.isnan(observe_aux_z(ctx, rt, positions_nm=bad))


def _with_table(rt, k_kcal):
    """Same runtime, one-state table with the given strength (0 = sham/ordinary, > 0 = active)."""
    import dataclasses
    return dataclasses.replace(rt, table=AuxStateTable(rt.table.model, (0.5,), (float(k_kcal),), (None,)))


def _degenerate(ctx, rt, d):
    bad = d["positions_nm"].copy()
    q = rt.table.model.feature_schema.features[0].atom_indices
    bad[q[0]] = bad[q[1]]
    ctx.setPositions(bad)
    return bad


@pytest.mark.parametrize("fast", [True, False])
def test_live_degenerate_geometry_with_an_active_state_raises_on_both_paths(fast):
    """Board condition 1 / spec 3.3: the fast path alone sees a finite theta, so the observer checks geometry."""
    from openmm import unit
    from gareus.auxiliary_cv.runtime import AuxObservationError, make_aux_z_observer
    ctx, rt, force, d = _setup()
    _degenerate(ctx, rt, d)
    active = _with_table(rt, 2.0)
    observe = make_aux_z_observer(active, use_fast_path=fast, fast_forces=[force], unit=unit)
    with pytest.raises(AuxObservationError, match="degenerate"):
        observe(0, types.SimpleNamespace(context=ctx))


@pytest.mark.parametrize("fast", [True, False])
def test_sham_only_population_never_raises_on_degenerate_geometry(fast):
    from openmm import unit
    from gareus.auxiliary_cv.runtime import make_aux_z_observer
    ctx, rt, force, d = _setup()
    _degenerate(ctx, rt, d)
    sham = _with_table(rt, 0.0)
    z = make_aux_z_observer(sham, use_fast_path=fast, fast_forces=[force], unit=unit)(0, types.SimpleNamespace(context=ctx))
    assert (math.isfinite(z) if fast else math.isnan(z))      # recorded, never raised


@pytest.mark.parametrize("fast", [True, False])
def test_regular_geometry_with_an_active_state_returns_the_path_z(fast):
    from openmm import unit
    from gareus.auxiliary_cv.runtime import make_aux_z_observer
    ctx, rt, force, d = _setup()
    active = _with_table(rt, 2.0)
    z = make_aux_z_observer(active, use_fast_path=fast, fast_forces=[force], unit=unit)(0, types.SimpleNamespace(context=ctx))
    assert z == pytest.approx(z_from_positions(d["positions_nm"], rt.table.model)[0], abs=1e-9)


def test_set_aux_parameters_absolute_energy_is_kcal_times_4184():
    """Board dissent (thinker): pin the kcal -> kJ conversion of set_aux_parameters in absolute terms."""
    from openmm import unit
    from gareus.auxiliary_cv.force import set_aux_parameters
    ctx, rt, _force, d = _setup()
    z = z_from_positions(d["positions_nm"], rt.table.model)[0]
    k_kcal, c = 2.5, z - 0.4
    set_aux_parameters(ctx, rt.info, center=c, k_kcal=k_kcal)
    e = ctx.getState(getEnergy=True, groups={rt.info.force_group}).getPotentialEnergy() \
           .value_in_unit(unit.kilojoule_per_mole)
    assert e == pytest.approx(0.5 * k_kcal * 4.184 * (z - c) ** 2, rel=1e-9)
```

- [ ] **Step 2: Run to verify failure**

Run: `tests/test_aux_cv_observation.py`
Expected: FAIL (`ImportError: observe_aux_z`). `test_set_aux_parameters_absolute_energy_is_kcal_times_4184` is a regression guard on Stage A's `set_aux_parameters`; it would pass on its own once the import succeeds.

- [ ] **Step 3: Implement**

Add to the imports of `gareus/auxiliary_cv/runtime.py`:

```python
import numpy as np

from .evaluate import z_from_positions
from .features import openmm_dihedrals, unique_torsions
```

Append:

```python
def observe_aux_z(context, runtime: AuxRuntime, *, force=None, positions_nm=None) -> float:
    """z of one carrier under the phase's auxiliary model, whatever state it occupies.

    Never raises on a non-finite value: the observation is recorded as is, and
    ``aux_bias_matrix_kcal`` refuses it when an active state needs it (sham arms may record NaN).
    The fast path cannot see a degenerate torsion (OpenMM returns a finite theta there).
    """
    if (force is None) == (positions_nm is None):
        raise ValueError("observe_aux_z needs exactly one of force= (fast path) or positions_nm= (slow path)")
    model = runtime.table.model
    if force is not None:
        values = np.asarray(force.getCollectiveVariableValues(context), dtype=np.float64)
        return float((model.offset + values.sum()) / model.scale)
    return float(z_from_positions(positions_nm, model)[0])


def check_aux_geometry(positions_nm, runtime: AuxRuntime, *, replica=None) -> None:
    """Fail closed when a model torsion is degenerate (spec 3.3).

    Same rule as Stage A ``openmm_dihedrals`` (NaN when |b1 x b2|^2 or |b2 x b3|^2 < DEGENERATE_CROSS2_NM4),
    over every unique torsion of the model, zero-coefficient ones included (Stage A's conservative rule).
    """
    quads, _idx = unique_torsions(runtime.table.model)
    theta = openmm_dihedrals(positions_nm, quads)[0]
    if np.isnan(theta).any():
        bad = [quads[t] for t in np.flatnonzero(np.isnan(theta))]
        where = "" if replica is None else f" on replica {replica}"
        raise AuxObservationError(f"degenerate auxiliary torsion(s) {bad}{where} while auxiliary states are "
                                  "active; failing the segment (spec 3.3)")


def make_aux_z_observer(runtime: AuxRuntime, *, use_fast_path: bool, fast_forces, unit):
    """The single per-carrier z observer for both the sample and the exchange path.

    With any active state: one positions read, a geometry check, then the fast-path z (the force's
    own value) or the slow-path z. With no active state: positions only on the slow path; never raises.
    """
    any_active = any(float(k) > 0.0 for k in runtime.table.k_kcal)

    def observe(r, sim) -> float:
        pos = None
        if any_active or not use_fast_path:
            pos = sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        if any_active:
            check_aux_geometry(pos, runtime, replica=r)
        if use_fast_path:
            return observe_aux_z(sim.context, runtime, force=fast_forces[r])
        return observe_aux_z(sim.context, runtime, positions_nm=pos)

    return observe
```

- [ ] **Step 4: Run to verify pass**

Run: `tests/test_aux_cv_observation.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add gareus/auxiliary_cv/runtime.py tests/test_aux_cv_observation.py
git commit -m "feat(cvaux): observe_aux_z fast/slow paths, live degeneracy check for active populations"
```

---

### Task 6: Auxiliary bias matrix in the sample and exchange paths

**Files:**
- Modify: `gareus/auxiliary_cv/runtime.py` (add `aux_bias_matrix_kcal`)
- Modify: `gareus/production.py`:
  - `assemble_bias_matrices` (`:2209`);
  - fast-path setup (`:7625-7655`);
  - the sample closure `_fetch_state` and its assembly (`:8080-8130`);
  - the `pymbar_metadata["samples_columns_for_mbar"]` descriptions (`:7862-7880`);
  - `_current_exchange_arrays` (`:8438-8498`).
- Test: `tests/test_aux_cv_production_wiring.py`

**Interfaces:**
- Produces:
  - `aux_bias_matrix_kcal(z_values, table) -> np.ndarray` of shape (n_states, n_replicas), kcal/mol. Inactive rows are exact zeros. If any state is active, every z must be finite, else `AuxObservationError`.
  - `assemble_bias_matrices(distance_bias_kcal, ss_bias_kcal, boost_bias_kj, aux_bias_kcal=None)`.
  - In `run_gareus`, one closure `_aux_z_for_replica(r, sim) -> float` (NaN when no runtime), called by **both** `_fetch_state` and `_fetch_exchange_state`.
  - `observable_cache["aux_z"]`.
- **`sampled_umbrella_bias_kj` is unchanged:** it stays primary + secondary only, matching its existing description at `production.py:7873`. The auxiliary term enters the totals (`umbrella_bias_kcal_mol`, `umbrella_bias_all_windows_*`, the `AnalysisArrayWriter` total vectors) through `assemble_bias_matrices`. The component vectors (distance, secondary) do not include it.
  - With an aux model, the field-description dictionary gains one key, `"auxiliary_cv_bias_note"`, saying so.
  - Stage C gates every loader that reads those totals (D5).

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
    """Sham arm: observe_aux_z may record NaN; with no active state the matrix is exact zeros."""
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


def test_sampled_umbrella_bias_stays_umbrella_only():
    src = inspect.getsource(__import__("gareus.production", fromlist=["run_gareus"]).run_gareus)
    line = [l for l in src.splitlines() if "sampled_umbrella_bias_kj = float(" in l]
    assert line and "aux" not in line[0], "sampled_umbrella_bias_kj must remain primary + secondary only"
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
        from .auxiliary_cv.runtime import aux_bias_matrix_kcal, make_aux_z_observer
        _fast_aux_forces = [sim.system.getForce(_aux_rt.force_index) for sim in sims]
        if any(f.getName() != _AUX_CV_FORCE_NAME for f in _fast_aux_forces):
            raise RuntimeError("replica systems do not carry the auxiliary-CV force at the base system's index")
        # The single z observation both the sample writer and the exchange kernel use; with any
        # active state it also fails the segment on a degenerate torsion (spec 3.3, Task 5).
        _aux_z_for_replica = make_aux_z_observer(_aux_rt, use_fast_path=_use_fast_cv_path,
                                                 fast_forces=_fast_aux_forces, unit=unit)
    else:
        def _aux_z_for_replica(r, sim) -> float:
            return float("nan")
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

Then:
- Add `observable_cache["aux_z"] = aux_z` next to `observable_cache["v_pep_kj"] = v_pep_kj`.
- Leave `sampled_umbrella_bias_kj` (`:8149`) unchanged.
- The sample-column descriptions are `pymbar_metadata["samples_columns_for_mbar"]`. The `pymbar_metadata = {` literal opens at `:7791`; `"samples_columns_for_mbar": {` opens at `:7862`. Directly before `write_json(out_dir / "umbrella_pymbar_metadata.json", pymbar_metadata)` (`:7880`), add:

```python
    if getattr(args, "_aux_runtime", None) is not None:
        pymbar_metadata["samples_columns_for_mbar"]["auxiliary_cv_bias_note"] = (
            "auxiliary-CV runs: umbrella_bias_kcal_mol / umbrella_bias_kj_mol and every "
            "umbrella_bias_all_windows_* total include the auxiliary restraint A_s of each state; "
            "sampled_umbrella_bias_kj and the distance/secondary component vectors do not.")
```

The off-path bytes of `umbrella_pymbar_metadata.json` stay identical, because the key is added only with an aux runtime.

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

### Task 7: Direct cross-energies through the production assembly, and exact Gibbs permutations

**Files:**
- Test only: `tests/test_aux_cv_exchange.py`. These tests prove existing kernel semantics, and the production assembly path, against states that contain auxiliary rows. No production code changes. If any test fails, the defect is in Tasks 3-6.

**Interfaces:**
- Consumes:
  - `tests/test_exchange_kernel_exact.py` helpers: `_states`, `_pi`, `_pair_kernel`, `_gibbs_kernel`, `_assert_kernel_ok`, `BETA`;
  - `gareus.production._gibbs_window_proposal_distribution`, `umbrella_bias_matrix_kcal`, `assemble_bias_matrices`;
  - `gareus.windows.set_window`;
  - Tasks 3-6.

- [ ] **Step 1: Write the tests**

```python
# tests/test_aux_cv_exchange.py
"""Spec 17: direct cross-energy check through the production assembly (set_window(aux_state=),
umbrella_bias_matrix_kcal, assemble_bias_matrices), exact Gibbs permutations with ordinary/aux/
duplicate states, global candidates. The exchange kernel is unchanged; these pin it against
auxiliary matrices."""
import itertools
import types

import numpy as np
import pytest

from aux_cv_fixture import dipeptide, model_payload
from gareus.auxiliary_cv.model import AuxModel
from gareus.auxiliary_cv.runtime import add_aux_cv_force, aux_bias_matrix_kcal, observe_aux_z
from gareus.auxiliary_cv.state_table import AuxStateTable
from gareus.production import _gibbs_window_proposal_distribution, assemble_bias_matrices, umbrella_bias_matrix_kcal
from gareus.windows import set_window
from test_exchange_kernel_exact import BETA, _assert_kernel_ok, _gibbs_kernel, _pair_kernel, _pi, _states

ARGS = types.SimpleNamespace(umbrella_force_group=31, secondary_cv_force_group=29)
KJ = 4.184


def _umbrella_forces(system):
    """CV1 = x of atom 0 (group 31, globals r0/k), CV2 = y of atom 1 (group 29, globals ss0/ss_k), nm units."""
    import openmm as mm
    f1 = mm.CustomExternalForce("0.5*k*(x-r0)^2")
    f1.addGlobalParameter("k", 0.0)
    f1.addGlobalParameter("r0", 0.0)
    f1.addParticle(0, [])
    f1.setForceGroup(31)
    system.addForce(f1)
    f2 = mm.CustomExternalForce("0.5*ss_k*(y-ss0)^2")
    f2.addGlobalParameter("ss_k", 0.0)
    f2.addGlobalParameter("ss0", 0.0)
    f2.addParticle(1, [])
    f2.setForceGroup(29)
    system.addForce(f2)


def _carriers(table, n_carriers=3):
    """Distinct real configurations: carrier r is the fixture after 20*r Langevin steps (k = 0 everywhere)."""
    import openmm as mm
    from openmm import unit
    from pep_gamd_fixture import _fresh_system
    d = dipeptide()
    ctxs, rt = [], None
    for r in range(n_carriers):
        system = _fresh_system()
        _umbrella_forces(system)
        rt = add_aux_cv_force(mm, system, table, ARGS)
        integ = mm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 0.002 * unit.picoseconds)
        integ.setRandomNumberSeed(11 + r)
        ctx = mm.Context(system, integ, mm.Platform.getPlatformByName("Reference"))
        ctx.setPositions(d["positions_nm"])
        ctx.setVelocitiesToTemperature(300 * unit.kelvin, 11 + r)
        if r:
            integ.step(20 * r)
        ctxs.append(ctx)
    return ctxs, rt, unit


def _group_energy_kj(ctx, groups, unit):
    return ctx.getState(getEnergy=True, groups=set(groups)).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)


def test_assembled_matrix_equals_direct_context_energies_and_all_four_swap_terms():
    d = dipeptide()
    blocks = [lab.split("-")[0] for lab in d["labels"]]
    m = AuxModel.from_mapping(model_payload(d["quads"], [1.2] + [0.6] * (2 * len(d["quads"]) - 1),
                                            offset=0.0, blocks=blocks))
    probe_table = AuxStateTable(m, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (None, None, None))
    ctxs, rt, unit = _carriers(probe_table)
    pos = [c.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer) for c in ctxs]
    z = np.array([observe_aux_z(c, rt, force=c.getSystem().getForce(rt.force_index)) for c in ctxs])
    assert np.ptp(z) > 1e-3, "carriers must differ in z for a meaningful cross-energy check"
    x = np.array([p[0, 0] for p in pos])          # CV1 values (nm)
    y = np.array([p[1, 1] for p in pos])          # CV2 values (nm)
    # Three states: ordinary, auxiliary, and a sham duplicate of the ordinary state.
    c1, k1 = np.array([x[0] + 0.01, x[1] - 0.02, x[0] + 0.01]), np.array([800.0, 600.0, 800.0])   # kcal/mol/nm^2
    c2, k2 = np.array([y[1], y[2] + 0.03, y[1]]), np.array([400.0, 500.0, 400.0])
    table = AuxStateTable(m, (0.0, float(z.mean()) + 0.3, 0.0), (0.0, 2.5, 0.0), (None, None, None))
    # Contexts must carry the SAME table the matrix uses: rebuild the runtime view with it.
    rt_states = type(rt)(table, rt.info, rt.force_index, rt.topology_sha256)
    # Production assembly path.
    d_kcal, s_kcal = umbrella_bias_matrix_kcal(x, y, c1, k1, c2, k2)
    _, matrix_kj = assemble_bias_matrices(d_kcal, s_kcal, np.zeros_like(d_kcal),
                                          aux_bias_kcal=aux_bias_matrix_kcal(z, table))
    # Direct: apply each state's complete target with set_window and read groups {31, 29, aux}.
    groups = (31, 29, rt.info.force_group)
    direct = np.zeros_like(matrix_kj)
    for s in range(3):
        for r, ctx in enumerate(ctxs):
            set_window(ctx, c1, KJ * k1, s, c2, KJ * k2, aux_state=rt_states)
            direct[s, r] = _group_energy_kj(ctx, groups, unit)
    np.testing.assert_allclose(matrix_kj, direct, rtol=1e-9, atol=1e-8)
    np.testing.assert_array_equal(matrix_kj[0], matrix_kj[2])       # sham == parent row
    for (a, b), (i, j) in itertools.product(itertools.permutations(range(3), 2),
                                            itertools.combinations(range(3), 2)):
        d_matrix = matrix_kj[b, i] + matrix_kj[a, j] - matrix_kj[a, i] - matrix_kj[b, j]
        d_direct = direct[b, i] + direct[a, j] - direct[a, i] - direct[b, j]
        assert d_matrix == pytest.approx(d_direct, abs=1e-7)


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
    bias = base + KJ * aux_bias_matrix_kcal(z, table)
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

Notes:
- `_gibbs_kernel` / `_pair_kernel` call the production `gibbs_propose_one_replica` / `apply_window_swap`, with the reverse proposal evaluated on the post-swap holder map. The verifier confirmed they are not reimplementations.
- The direct test uses a positions-independent umbrella on single coordinates so that `umbrella_bias_matrix_kcal`'s inputs are exact.
- Every term of the four-term swap difference is compared, for every ordered state pair and every carrier pair.

- [ ] **Step 2: Run**

Run: `tests/test_aux_cv_exchange.py tests/test_exchange_kernel_exact.py`
Expected: PASS. A failure here is a Task 3-6 bug: fix it there, never weaken these tests.

- [ ] **Step 3: Commit**

```bash
git add tests/test_aux_cv_exchange.py
git commit -m "test(cvaux): direct cross-energies via set_window + production assembly; exact Gibbs permutations with aux and sham states"
```

---

### Task 8: Kernel identity, `aux_unpersisted` eligibility, resume refusal, provenance

**Files:**
- Modify: `gareus/kernel_identity.py`:
  - `EXCHANGE_ENERGY_VERSION` stays;
  - add the aux constant, the resolver, `ELIGIBLE_AUX_UNPERSISTED` and `AUX_PERSISTED_SNAPSHOT_KEY`;
  - update `kernel_identity_for_run` and `classify_segment_kernel`.
- Modify: `gareus/provenance.py:345-346`
- Modify: `gareus/production.py`: add `refuse_resume_of_aux_campaign` next to `verify_kernel_identity_on_resume` (`:660`).
- Test: `tests/test_aux_cv_kernel_identity.py`

**Interfaces:**
- Produces:
  - `EXCHANGE_ENERGY_VERSION_AUX = "state_bias_matrix_v3_aux"`;
  - `ELIGIBLE_AUX_UNPERSISTED = "aux_unpersisted"`;
  - `AUX_PERSISTED_SNAPSHOT_KEY = "aux_sample_schema"`. Stage C writes this key into a segment's `windows/<seg>.json` when that segment stores z and torsions, and Stage C owns the classification of such segments.
  - `exchange_energy_version_for_args(args) -> str`.
  - `kernel_identity_for_run`: the identity gains `"aux_model_sha256"` (plus `"aux_topology_sha256"` when the runtime knows it) **only** when an aux model is configured, so the off-path digest is unchanged. The sha comes from `args._aux_runtime` when it exists; the model file is read only when no runtime exists yet (provenance at parse time).
  - `classify_segment_kernel` checks this **before** the residual branch: a v3_aux snapshot without `AUX_PERSISTED_SNAPSHOT_KEY` gives `ELIGIBLE_AUX_UNPERSISTED`; with the key, `ELIGIBLE_UNKNOWN` ("needs the Stage C reader"). It never gives `verified` for v3_aux. The query loader excludes every status other than `verified`/`not_applicable` (`gareus/query.py:227-231`), so `aux_unpersisted` segments are excluded and reported.
  - `method_settings["aux_cv_model_sha256"]` and `method_settings["aux_envelope_calibration"] = "aux_inactive"`, only when configured.
  - `refuse_resume_of_aux_campaign(out_dir) -> None` (in `production.py`). It reads the newest segment snapshot's `kernel_identity` unconditionally. If that identity is v3_aux or carries `aux_model_sha256`, it raises `RuntimeError`.
- `verify_kernel_identity_on_resume` is **not** changed in Stage B: aux resume never reaches it (D6). Stage C makes it args-aware when it lifts the refusal.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_aux_cv_kernel_identity.py
import json
import types

import pytest

from aux_cv_fixture import model_payload
from gareus.auxiliary_cv.model import AuxModel
from gareus.kernel_identity import (AUX_PERSISTED_SNAPSHOT_KEY, ELIGIBLE_AUX_UNPERSISTED, ELIGIBLE_NOT_APPLICABLE,
                                    ELIGIBLE_UNKNOWN, ELIGIBLE_VERIFIED, EXCHANGE_ENERGY_VERSION,
                                    EXCHANGE_ENERGY_VERSION_AUX, RESIDUAL_EVALUATOR_VERSION,
                                    classify_segment_kernel, exchange_energy_version_for_args,
                                    kernel_identity_for_run)

LEGACY_ARGS = types.SimpleNamespace(secondary_cv="none", production_ensemble="npt",
                                    gamd_boost_type="pep-gamd-lower-dual")
# Digest of the legacy identity for LEGACY_ARGS, computed on main dc30285 (kernel_identity.py unchanged);
# re-confirmed by the verifier on unmodified code (2026-10-08).
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


def test_identity_uses_the_runtime_model_and_topology_without_rereading_the_file(tmp_path):
    args, m = _aux_args(tmp_path)
    args.aux_cv_model = str(tmp_path / "gone.json")            # the file is not read when a runtime exists
    args._aux_runtime = types.SimpleNamespace(info=types.SimpleNamespace(model_sha256=m.model_sha256),
                                              topology_sha256="c" * 64)
    ident = kernel_identity_for_run(args, {})
    assert ident["aux_model_sha256"] == m.model_sha256 and ident["aux_topology_sha256"] == "c" * 64


@pytest.mark.parametrize("cv2", ["residual-torsion-pc", "none", "torsion-pca"])
def test_unpersisted_aux_segments_are_ineligible_for_every_cv2(cv2):
    snap = {"cv2_type": cv2,
            "kernel_identity": {"cv_evaluator_version": RESIDUAL_EVALUATOR_VERSION,
                                "exchange_energy_version": EXCHANGE_ENERGY_VERSION_AUX}}
    assert classify_segment_kernel(snap)[0] == ELIGIBLE_AUX_UNPERSISTED


def test_v3_aux_is_never_verified_in_stage_b():
    snap = {"cv2_type": "residual-torsion-pc", AUX_PERSISTED_SNAPSHOT_KEY: {"schema": "x"},
            "kernel_identity": {"cv_evaluator_version": RESIDUAL_EVALUATOR_VERSION,
                                "exchange_energy_version": EXCHANGE_ENERGY_VERSION_AUX}}
    assert classify_segment_kernel(snap)[0] == ELIGIBLE_UNKNOWN


def test_legacy_classification_unchanged():
    res = {"cv2_type": "residual-torsion-pc",
           "kernel_identity": {"cv_evaluator_version": RESIDUAL_EVALUATOR_VERSION,
                               "exchange_energy_version": EXCHANGE_ENERGY_VERSION}}
    assert classify_segment_kernel(res)[0] == ELIGIBLE_VERIFIED
    assert classify_segment_kernel({"cv2_type": "none", "kernel_identity": {}})[0] == ELIGIBLE_NOT_APPLICABLE


def _run_dir_with_snapshot(tmp_path, identity):
    run = tmp_path / "run"
    (run / "windows").mkdir(parents=True)
    (run / "segments.json").write_text(json.dumps([{"segment_id": "seg_001"}]))
    (run / "windows" / "seg_001.json").write_text(json.dumps({"segment_id": "seg_001", "cv1_type": "contacts",
                                                              "cv2_type": None, "windows": [],
                                                              "kernel_identity": identity}))
    return run


def test_segment_eligibility_excludes_unpersisted_aux_segments(tmp_path):
    from gareus.query import segment_eligibility
    run = _run_dir_with_snapshot(tmp_path, {"exchange_energy_version": EXCHANGE_ENERGY_VERSION_AUX,
                                            "aux_model_sha256": "d" * 64})
    assert segment_eligibility(run)["seg_001"]["eligibility"] == ELIGIBLE_AUX_UNPERSISTED


def test_resume_of_an_aux_campaign_is_refused_unconditionally(tmp_path):
    from gareus.production import refuse_resume_of_aux_campaign
    run = _run_dir_with_snapshot(tmp_path, {"exchange_energy_version": EXCHANGE_ENERGY_VERSION_AUX,
                                            "aux_model_sha256": "d" * 64})
    with pytest.raises(RuntimeError, match="auxiliary"):
        refuse_resume_of_aux_campaign(run)


def test_resume_of_a_legacy_campaign_is_not_refused(tmp_path):
    from gareus.production import refuse_resume_of_aux_campaign
    refuse_resume_of_aux_campaign(_run_dir_with_snapshot(tmp_path, {"exchange_energy_version": EXCHANGE_ENERGY_VERSION}))
    refuse_resume_of_aux_campaign(tmp_path / "empty")          # no windows/ at all


def test_method_settings_record_aux_only_when_configured(tmp_path):
    from gareus.provenance import _method_settings
    from gareus.cli import parse_args
    a = parse_args(["--seq", "GA", "--out", str(tmp_path / "o"), "--cv1", "contacts"])
    s = _method_settings(a)
    assert s["exchange_energy_version"] == "state_bias_matrix_v2"
    assert "aux_cv_model_sha256" not in s and "aux_envelope_calibration" not in s
    args, m = _aux_args(tmp_path)
    a.aux_cv_model = args.aux_cv_model
    s2 = _method_settings(a)
    assert s2["exchange_energy_version"] == "state_bias_matrix_v3_aux"
    assert s2["aux_cv_model_sha256"] == m.model_sha256 and s2["aux_envelope_calibration"] == "aux_inactive"
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

In `gareus/kernel_identity.py`, under `ELIGIBLE_NOT_APPLICABLE`:

```python
#: Auxiliary-CV segment whose samples carry no z / torsion features (Stage B engineering runs):
#: its bias cannot be reconstructed for any analysis, whatever its CV2 mode.
ELIGIBLE_AUX_UNPERSISTED = "aux_unpersisted"

#: The same shared assembly plus the exact auxiliary-CV restraint term evaluated for every
#: carrier under every active state (spec 2026-10-07 auxiliary CV, Section 5). Used whenever an
#: auxiliary model is configured, including sham arms whose strengths are all zero.
EXCHANGE_ENERGY_VERSION_AUX = "state_bias_matrix_v3_aux"

#: Window-snapshot key Stage C writes when a segment stores auxiliary z and torsion features.
AUX_PERSISTED_SNAPSHOT_KEY = "aux_sample_schema"


def exchange_energy_version_for_args(args) -> str:
    return EXCHANGE_ENERGY_VERSION_AUX if getattr(args, "aux_cv_model", None) else EXCHANGE_ENERGY_VERSION
```

In `kernel_identity_for_run`, replace `"exchange_energy_version": EXCHANGE_ENERGY_VERSION,` with `"exchange_energy_version": exchange_energy_version_for_args(args),`. Immediately before `body = json.dumps(...)`, add:

```python
    if getattr(args, "aux_cv_model", None):
        runtime = getattr(args, "_aux_runtime", None)
        if runtime is not None:
            identity["aux_model_sha256"] = str(runtime.info.model_sha256)
            if getattr(runtime, "topology_sha256", None):
                identity["aux_topology_sha256"] = str(runtime.topology_sha256)
        else:
            from .auxiliary_cv.model import AuxModel   # lazy: keep this module import-light
            identity["aux_model_sha256"] = AuxModel.load(args.aux_cv_model).model_sha256
```

In `classify_segment_kernel`, insert directly after `identity = snap.get("kernel_identity") or {}`:

```python
    if identity.get("exchange_energy_version") == EXCHANGE_ENERGY_VERSION_AUX or identity.get("aux_model_sha256"):
        if not snap.get(AUX_PERSISTED_SNAPSHOT_KEY):
            return ELIGIBLE_AUX_UNPERSISTED, ("auxiliary-CV segment without stored z/torsion features "
                                              "(Stage B engineering run): its bias cannot be reconstructed")
        return ELIGIBLE_UNKNOWN, "auxiliary-CV segment: classification needs the Stage C reader"
```

The residual branch keeps comparing `ex == EXCHANGE_ENERGY_VERSION` exactly. Do not add v3_aux to it.

In `gareus/provenance.py`, replace lines 345-346 with:

```python
    from .kernel_identity import RESIDUAL_EVALUATOR_VERSION, exchange_energy_version_for_args
    settings["exchange_energy_version"] = exchange_energy_version_for_args(args)
    if getattr(args, "aux_cv_model", None):
        from .auxiliary_cv.model import AuxModel
        settings["aux_cv_model_sha256"] = AuxModel.load(args.aux_cv_model).model_sha256
        # GaMD envelope recon/calibration ran with the auxiliary restraint off (k = 0), so arms that
        # differ only in auxiliary strength calibrate identical envelopes (spec 3, 11.2).
        settings["aux_envelope_calibration"] = "aux_inactive"
```

Read the two lines being replaced first. If line 345 already binds `RESIDUAL_EVALUATOR_VERSION` into `settings`, keep that assignment as well.

In `gareus/production.py`, add after `verify_kernel_identity_on_resume`:

```python
def refuse_resume_of_aux_campaign(out_dir) -> None:
    """Stage B: an auxiliary-CV campaign cannot be resumed or extended (spec 2026-10-07, D6).

    Unconditional and snapshot-based on purpose. A Context checkpoint of an aux system loads
    SILENTLY into a Context without the aux force (verified 2026-10-08), the run manifest's
    method_settings are rebuilt from the CURRENT args every job, and verify_kernel_identity_on_resume
    only runs when CV2 is enabled -- none of those can stop an aux slot resuming as an ordinary one.
    """
    recorded = _latest_segment_kernel_identity(out_dir) or {}
    if recorded.get("exchange_energy_version") == EXCHANGE_ENERGY_VERSION_AUX or recorded.get("aux_model_sha256"):
        raise RuntimeError(
            "resume/extend refused: this campaign ran auxiliary-CV states "
            f"(model {str(recorded.get('aux_model_sha256'))[:12]}). Resuming them is Stage C "
            "(checkpoint binding of auxiliary states); without it every auxiliary slot would continue as "
            "an ordinary state. Start a new run directory instead.")
```

Import `EXCHANGE_ENERGY_VERSION_AUX` with the other kernel-identity names (`:120`).

- [ ] **Step 4: Run to verify pass**

Run: `tests/test_aux_cv_kernel_identity.py`, plus the existing kernel-identity regressions found with `grep -l "kernel_identity\|segment_eligibility" tests/*.py` (run each file listed).
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add gareus/kernel_identity.py gareus/provenance.py gareus/production.py tests/test_aux_cv_kernel_identity.py
git commit -m "feat(cvaux): v3_aux kernel identity, aux_unpersisted eligibility, aux resume refusal"
```

---

### Task 9: `run_gareus` builds the runtime; snapshot rows; population guards; exit gate

**Files:**
- Modify: `gareus/production.py`:
  - resume branch (`:6775`, `if fast_resume:`);
  - window loading (`:6819-6835`);
  - `--max-replicas` truncation (`:6920-6933`);
  - base system (`:7044-7047`);
  - starting-structure systems (`:7155`, `:7204`);
  - US auto-drop (`:7217`);
  - window snapshot (`:7892-7898`).
- Modify: `CLAUDE.md` (handoff section)
- Test: `tests/test_aux_cv_production_wiring.py` (append)

**Interfaces:**
- Consumes: Tasks 2-8.
- Produces: `args._aux_runtime: AuxRuntime | None`, set before any Context is built. The aux force sits at the same group and index in the base system and in every replica copy; starting-structure systems get the same group.

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
    assert _calls(tree, "load_aux_state_table")
    checks = _calls(tree, "check_feature_atoms")
    assert checks and all(any(k.arg == "topology_sha256" for k in c.keywords) for c in checks)
    causes = {k.value.value for c in _calls(tree, "refuse_aux_population_change") for k in c.keywords
              if k.arg == "cause" and isinstance(k.value, ast.Constant)}
    assert causes == {"seed-reachability filter", "--max-replicas", "US auto-drop"}
    assert _calls(tree, "aux_snapshot_rows"), "Stage B snapshots must carry aux fields (D5)"


def test_resume_refusal_runs_first_in_the_fast_resume_branch():
    import gareus.production as production
    src = inspect.getsource(production.run_gareus)
    i_refuse = src.find("refuse_resume_of_aux_campaign(out_dir)")
    i_load = src.find("resume_def = load_resume_run_definition(")
    assert 0 <= i_refuse < i_load, "the aux resume refusal must precede loading the resume definition"
    assert src.find("if fast_resume:") < i_refuse
```

- [ ] **Step 2: Run to verify failure**

Run: `tests/test_aux_cv_production_wiring.py`
Expected: the new tests FAIL (`add_aux_cv_force` not called; refusal missing).

- [ ] **Step 3: Implement**

Resume branch: make the first statement inside `if fast_resume:` (`:6775`)

```python
        refuse_resume_of_aux_campaign(out_dir)
```

`--extend` routes through `args.resume = True` (`_resolve_and_apply_extend_mode`), so it reaches the same branch.

Near the top of `run_gareus` (before `if fast_resume:`), add `_aux_table = None`.

In the window-loading branch, right after the `filter_explicit_2d_windows_by_seed_reachability(...)` call (`:6825-6829`), add the block below.

`_n_windows_before_reachability_filter` already exists on main and needs no new definition. It is set at `production.py:6823` as `_n_windows_before_reachability_filter = int(len(centers_a))`, immediately before the filter call, in the same `if getattr(args, "windows_2d_csv", None):` branch (board caveat 3). Re-confirm that line before editing; if it has moved or been removed, define it the same way directly before the filter call.

```python
            if getattr(args, "aux_cv_model", None):
                from .auxiliary_cv.runtime import refuse_aux_population_change
                refuse_aux_population_change(_n_windows_before_reachability_filter, len(centers_a),
                                             cause="seed-reachability filter")
                from .auxiliary_cv.state_table import load_aux_state_table
                _aux_table = load_aux_state_table(args.aux_cv_model, (window_metadata or {}).get("aux_rows") or [])
                refuse_aux_population_change(_aux_table.n, len(centers_a), cause="seed-reachability filter")
                print("WARNING: auxiliary-CV states active (--aux-cv-allow-unpersisted): z values are not "
                      "written to the sample store yet, so this is an engineering run; its segments are "
                      "classified 'aux_unpersisted' and excluded from every analysis (Stage C).", flush=True)
```

In the `--max-replicas` truncation (`:6924`), make the first statement of the `elif _max_replicas > 0 and len(centers_a) > _max_replicas:` branch:

```python
        if _aux_table is not None:
            from .auxiliary_cv.runtime import refuse_aux_population_change
            refuse_aux_population_change(len(centers_a), _max_replicas, cause="--max-replicas")
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
            raise RuntimeError("--aux-cv-model set but no auxiliary state table was loaded")
        from .auxiliary_cv.features import check_feature_atoms
        from .auxiliary_cv.runtime import add_aux_cv_force, canonical_topology_sha256
        _aux_topology_sha = canonical_topology_sha256(topology, _aux_table.model)
        check_feature_atoms(_aux_table.model, topology, topology_sha256=_aux_topology_sha)
        args._aux_runtime = add_aux_cv_force(openmm, base_system, _aux_table, args,
                                             topology_sha256=_aux_topology_sha)
        print(f"    Auxiliary-CV restraint: model {args._aux_runtime.info.model_sha256[:12]}, force group "
              f"{args._aux_runtime.info.force_group}, {sum(k > 0 for k in _aux_table.k_kcal)} active of "
              f"{_aux_table.n} states; topology {_aux_topology_sha[:12]}")
```

After each of the two `add_umbrella_cv_forces(openmm, starting_structure_system, ...)` calls (`:7155`, `:7204`), add:

```python
                if args._aux_runtime is not None:
                    from .auxiliary_cv.runtime import add_aux_cv_force
                    add_aux_cv_force(openmm, starting_structure_system, _aux_table, args,
                                     force_group=args._aux_runtime.info.force_group)
```

Indent to match each site. These systems drive the US pull. The aux force there keeps its default k = 0; ramping it is Stage D admission.

US auto-drop: find it with `grep -n "drop_bad_us_windows_and_rebuild(" gareus/production.py` (`:7217`). Inside the branch that decides a drop is needed, directly before the call, add:

```python
            if getattr(args, "_aux_runtime", None) is not None:
                from .auxiliary_cv.runtime import refuse_aux_population_change
                refuse_aux_population_change(len(centers_a), len(centers_a) - 1, cause="US auto-drop")
```

The CLI already refuses `--us-auto-drop-bad-windows`. This is the runtime backstop, and it always raises because any drop changes the count.

Window snapshot (`:7892-7898`): directly after `_win_snapshot_windows = snapshot_window_rows(...)`, add:

```python
    if getattr(args, "_aux_runtime", None) is not None:
        from .auxiliary_cv.runtime import aux_snapshot_rows
        # D5: every Stage-B-era snapshot names its auxiliary restraints, so a strict reader fails closed
        # (MissingCoordinateError) instead of reconstructing the bias without A_s.
        _win_snapshot_windows = aux_snapshot_rows(_win_snapshot_windows, args._aux_runtime.table)
```

`kernel_identity_for_run(args, ...)` at `:7898` now records v3_aux, the model sha and the topology sha from the runtime.

`_build_context_i` deserializes `base_system` for each replica, so replicas inherit the aux force at `force_index`; Task 6's name check enforces this. Keep `prepare_pep_gamd_args` before the aux force: it only records peptide atoms. `ensure_pep_gamd_partition` runs inside `make_gamd_integrator`, after all bias forces exist, which `verify_pep_gamd_bias_force_groups` checks.

- [ ] **Step 4: Run the Stage B exit gate**

Run, together:
- Stage B: `tests/test_aux_cv_cli.py tests/test_aux_cv_state_table.py tests/test_aux_cv_runtime.py tests/test_aux_cv_set_window.py tests/test_aux_cv_composition.py tests/test_aux_cv_observation.py tests/test_aux_cv_production_wiring.py tests/test_aux_cv_exchange.py tests/test_aux_cv_kernel_identity.py`
- Stage A: `tests/test_aux_cv_model.py tests/test_aux_cv_features.py tests/test_aux_cv_evaluate.py tests/test_aux_cv_force.py tests/test_aux_cv_state_schema.py tests/test_aux_cv_bias.py`
- Legacy regressions: `tests/test_exchange_kernel_exact.py tests/test_sample_before_exchange_ordering.py tests/test_lambda_ladder_states.py tests/test_pep_gamd_boost.py tests/test_npt_coupled_target.py tests/test_contact_map_cv1_production.py tests/test_gamd_boost_default.py`

Expected: all PASS.

- [ ] **Step 5: Add the handoff section to `CLAUDE.md`** (after the Stage A section)

```markdown
## CVaux Stage B (plain-run production wiring; spec 2026-10-07-auxiliary-cv-gibbs-production-spec Sections 3.3, 3.4, 4)

- `--aux-cv-model PATH` (+ required `--aux-cv-allow-unpersisted` until Stage C) on a plain run (`--window-mode adaptive|manual`) with `--windows-2d-csv` rows carrying `aux_k_kcal_mol` (explicit 0 for ordinary/sham), `aux_center`, optional instance columns (parent = `state_instance_id`). Refused: stock gamd boost types (factory moves every force to group 0), `--exchange-mode neighbor`, `--resume/--extend`, `--us-auto-drop-bad-windows`, other window modes, `--swarm-stage`. Population frozen: reachability drop, `--max-replicas` truncation, US auto-drop all fatal.
- `gareus/auxiliary_cv/runtime.py`: force group by audit (never 0-2, 29, 31); `add_aux_cv_force` on base + starting-structure systems before any integrator; `observe_aux_z` (fast = (offset + sum of sub-CVs)/scale; slow = positions; records, never raises); `aux_bias_matrix_kcal` (non-finite z with an active state fatal); `set_window(..., aux_state=)` at all six window applications (AST-pinned incl. callable use); GaMD recon/calibration with aux OFF (`deactivate_aux_parameters`, method_settings `aux_envelope_calibration: aux_inactive`); `assemble_bias_matrices(..., aux_bias_kcal=)` in both sample and exchange paths; off path bitwise legacy. `sampled_umbrella_bias_kj` stays primary+secondary; totals include A_s.
- Stage-B data is unanalysable: snapshot rows carry aux_model_sha256/aux_center/aux_k (strict reconstruction raises MissingCoordinateError); `classify_segment_kernel` -> `aux_unpersisted` for every v3_aux snapshot without `aux_sample_schema`; `refuse_resume_of_aux_campaign` reads the newest snapshot unconditionally (checkpoints load silently into a non-aux Context).
- Kernel `state_bias_matrix_v3_aux` + `aux_model_sha256` (+ `aux_topology_sha256` = `canonical_topology_sha256`, content of the feature atoms' chains) only with an aux model; legacy digest pinned. A production model must record that canonical digest as `feature_schema.topology_sha256` (`python -m gareus.auxiliary_cv.runtime topology-sha PDB MODEL`).
- Proven: aux gradient applied once at full strength under an active Pep-GaMD boost (structure-matched paired arms; boosted copy = FSF_Total-scaled), Pep-GaMD NPT adapter sees aux as a bias group and tracks geometry (A_s is invariant under molecular-centroid scaling), production assembly = direct Context energies incl. all four swap terms with a sham row, exact Gibbs/pair detailed balance and sweep stationarity with aux + duplicate sham states, no candidate mask.
- Stage D prerequisites (explicit deferrals, owners). Until done, each is reported as **unavailable, never as a pass** (spec §15), and none is part of Stage B's exit gate. Listed: finite-timestep validation of strong aux restraints (spec 3.4; Stage D); NPT controlled-distribution test (spec 17; Stage D); multi-Context cost benchmark incl. zero-strength overhead (spec 10; Stage D, Stage C measures the torsion read); diagnostics audit (spec 15: ladder_overlap, pmf_ladder_crosscheck with an ordinary-λ0 crosscheck, other_rung_same_centre, gareus_report; Stage D); pre-production equilibration and phase_kind (spec 6; Stage C records phase_kind, Stage D owns the protocol); aux pull ramp in admission (Stage D); c10 model port with the canonical topology digest (Stage D).
- Stage C must: write `aux_sample_schema` into snapshots that store z/torsions and classify them; gate load_parquet/load_npz/load_csv/union loaders (incl. pilot dirs)/export on aux presence; lift the resume refusal and `--aux-cv-allow-unpersisted`; make `verify_kernel_identity_on_resume` args-aware; source `peptide_atoms` for `build_sample_schema` under `--run-mode cmd` (Stage B only sets it via `prepare_pep_gamd_args`).
```

- [ ] **Step 6: Commit**

```bash
git add gareus/production.py CLAUDE.md tests/test_aux_cv_production_wiring.py
git commit -m "feat(cvaux): run_gareus builds the auxiliary runtime; unanalysable Stage B snapshots; frozen-population guards; handoff"
```

---

## Self-review record

**Spec coverage, Stage B** (spec 16: "Wire the exact term into every Context, complete state application, exchange matrices, sampling and NPT. Preserve the existing Gibbs proposal semantics. Add canonical direct-energy tests and exact small-permutation invariance tests."):

| Requirement | Task |
|---|---|
| Every Context carries the capability | Tasks 3, 9 |
| Complete state application | Task 4, all six sites |
| Exchange matrices, sample = exchange | Task 6 |
| Sampling observation of z per carrier | Tasks 5, 6 |
| NPT | Task 4, with the Pep-GaMD adapter |
| Gibbs semantics | Task 7: unchanged kernel, proven on aux matrices |
| Direct energies through the production assembly | Task 7 |
| Permutations | Task 7 |
| Force-group audit | Task 3 |
| Integrator expression audit | Task 4, structure-matched gradient-once test |
| Missing z fatal when needed | Task 6 |
| Loader parses aux + instance columns | Task 2 |
| v3_aux only with capability, never verified in B | Task 8 |

**Section 17 rows covered:**
- zero auxiliary strength on every state (Tasks 6, 8, plus the Stage A zero-k tests);
- active → ordinary → active (Task 4);
- force-group composition (Task 4);
- direct cross-energy check (Task 7);
- ordinary-origin samples (Tasks 5 and 7: z observed for k = 0 carriers and used in active rows);
- exact Gibbs permutations (Task 7);
- global candidates (Task 7);
- duplicate Hamiltonians in exchange (Task 7);
- NPT trial, geometry dependence (Task 4).

**Rows left to later stages:**

| Row | Owner |
|---|---|
| Duplicate-Hamiltonian MBAR merging | Stage C |
| Missing feature or changed model on resume/analysis | Stage C |
| Multiple swaps at one timestamp in the episode parser | Stage D |
| Model parity on c10 frames | Stage D (needs the recovered artifact) |
| Reduced auxiliary-energy precision on a mixed-precision platform | Stage C exit gate on aurum2, D9 |
| NPT controlled distribution | Stage D (handoff list) |

**Finding → change map (revision 2):**

| Finding | Change |
|---|---|
| B1 | Task 8: `ELIGIBLE_AUX_UNPERSISTED` before the residual branch, v3_aux never verified, test inverted. Task 9: snapshot rows carry aux fields (D5) |
| B2 | Task 8 `refuse_resume_of_aux_campaign` (unconditional, snapshot-based), Task 9 calls it first in `fast_resume`. The dead `verify_kernel_identity_on_resume` edit was removed (D6) |
| B3 | Task 1: window modes other than adaptive/manual and `--swarm-stage` refused; the validator runs first. Task 9: `--max-replicas` guard (D7) |
| B4 | Task 4 gradient-once test rewritten with structure-matched paired arms, an FSF_Total scaling assertion and the measured-FSF note |
| B5 | Task 4 NPT test uses the Pep-GaMD adapter. Molecular-scaling cancellation documented, plus a perturbed-geometry check. Distribution test deferred to Stage D |
| B6 | Task 7 direct test goes through `set_window(aux_state=)` + `umbrella_bias_matrix_kcal` + `assemble_bias_matrices`, with 3 Contexts, group energies {31, 29, aux}, a sham row, and all swap terms |
| B7 / seams 11 | Task 3 `canonical_topology_sha256`. Task 9 passes it to `check_feature_atoms` and into the runtime; Task 8 puts it in the kernel identity |
| B8 | Task 4: recon/calibration sites use `aux_state=None` + `deactivate_aux_parameters`; provenance records `aux_envelope_calibration` |
| B9 | Task 3 `refuse_aux_population_change` + unit tests. Task 9 calls it at the reachability, `--max-replicas` and US-auto-drop sites (AST-checked causes) |
| B10 | Task 4: sixth site (`run_production_probe`); AST pin covers callable use and every Name reference |
| B11 / seams 15 | Task 5: `observe_aux_z` records and never raises (sham arm). Fatality only in `aux_bias_matrix_kcal`. Fast-path degenerate behaviour documented |
| B12 | Task 2: loaded-table CSV fieldnames union (+ test); column constants in `windows.py` so the off path never imports the aux package (subprocess test) |
| B13 | Refusal message fixed; flags in `build_gareus_parser` (+ test); kernel identity uses the runtime sha; the slow-path position re-fetch stays (one extra `getState`, slow path only); spec 3.4/10 in the Stage D list |
| Seams 1 | = B1 + D5 snapshot rows |
| Seams 10 (B part) | `sampled_umbrella_bias_kj` stays umbrella-only (no meaning change); totals include A_s; description note added with aux; loader gating is Stage C's |
| Seams 12 | Stage D prerequisites list in the handoff (D13) |
| Seams 14 | = B13 parser registration |
| Seams 15 (B part) | the role rule matches Stage A D3 (auxiliary ⇔ k > 0; ordinary/sham k = 0); private Stage A helpers kept deliberately (noted in Depends on) |

**Board conditions applied (rev_stage_B, ACCEPT-WITH-CHANGES, unanimous):**

| Board item | Change |
|---|---|
| Condition 1: live degenerate-geometry fail-closed on sample and exchange paths | Task 5 `check_aux_geometry` + `make_aux_z_observer`, installed as `_aux_z_for_replica` (Task 6). Fast-path z is kept, plus a positions-based degeneracy check whenever any state is active. Tests on both paths, a sham-only no-raise test, and a §3.3 compliance note on bounded exposure |
| Thinker partial dissent: absolute aux-energy unit test | Task 5 `test_set_aux_parameters_absolute_energy_is_kcal_times_4184` |
| Caveat 2: topology digest chain-index assumption | Documented at the `topology-sha` entry point |
| Caveat 3: `_n_windows_before_reachability_filter` | Already defined at `production.py:6823`; noted in Task 9 with a re-confirm instruction |
| Caveat 4: deferred validations | Handoff list states they are reported unavailable, never pass, and are outside Stage B's exit gate |
| Caveat 5: `LEGACY_DIGEST` re-confirm | Task 8 Step 1b kept unchanged |

**Placeholder scan:** none. `LEGACY_DIGEST` is pinned (computed on dc30285, re-confirmed by the verifier), and Task 8 Step 1b only re-confirms it.

**Type consistency:**
- `AuxRuntime(table, info, force_index, topology_sha256=None)` is used identically in Tasks 3-9.
- Task 7 rebuilds a view with `type(rt)(table, info, force_index, topology_sha256)`.
- `aux_bias_matrix_kcal(z_values, table)` takes the `AuxStateTable`, not the runtime.
- `set_window(..., aux_state=AuxRuntime | None)`.
- `observe_aux_z(context, runtime, *, force=None, positions_nm=None)` in Tasks 5-7.

**Findings recorded for the parent:**
1. The stock gamd-openmm factory calls `set_all_forces_to_group(system, 0)` for every stock boost type (`gamd/integrator_factory.py:198`). Under stock types the existing umbrella and CV2 forces are also in group 0: boosted for total/dual types, and invisible to `fast_cv_force_indices`' group lookup. That is pre-existing and outside this plan; the aux feature refuses stock types.
2. Spec 3.3's cache invalidation already exists: `observable_cache.clear()` runs after an accepted swap.
3. `refuse_resume_of_aux_campaign` is an unconditional, snapshot-based guard. Stage C replaces it with checkpoint binding (D12).
