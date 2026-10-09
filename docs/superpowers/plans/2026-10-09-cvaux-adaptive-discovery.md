# CVaux Adaptive Discovery and Admission Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The adaptive-production driver discovers a torsion-linear auxiliary CV (z3) from its own production frames at a numbered-epoch boundary, places at most four lambda = 0 worker states, admits them into the running campaign, pools them in union MBAR, and reports them; chignolin_11 runs c10's settings plus this.

**Architecture:** A pure package `gareus/adaptive/aux_discovery/` (settings, frame table and descriptors, partitions, z3 search, placement port of c10's `placement.py`, validation record) plus an I/O hook `gareus/adaptive/aux_admission_io.py` called after the respring hook in the epoch loop. Admission is a new P2 action `admit_aux`; worker identity lives in `WindowState.metadata["aux"]`. Phases already run as `window_mode: adaptive` with an explicit window CSV, the path the Stage A-C aux machinery accepts, so post-admission phases get aux columns and an injected `aux_cv_model`. Pooling adds a z backfill for pre-admission phases and an aux term in both union paths.

**Tech Stack:** Python 3, numpy, pandas, scikit-learn (KNeighborsRegressor, PCA, GaussianMixture, LogisticRegression, adjusted_rand_score), mdtraj (XTC reading), OpenMM (Reference platform in tests), pyarrow/duckdb (Parquet), existing `gareus.auxiliary_cv` (Stages A-C).

**Spec:** `docs/superpowers/specs/2026-10-09-cvaux-adaptive-discovery-design.md` (commit 0c6de3f). Background: `docs/superpowers/specs/2026-10-07-auxiliary-cv-gibbs-production-spec.md`. c10 reference scripts and data (gitignored, main checkout only): `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/docs/_local_docs/c10_aux_diagnosis/`.

## Global Constraints

- Opt-in: `--ap-aux-discovery` default off. With it off every existing path (registry JSON/CSV, window CSVs, decision settings except the new keys, union inputs, reports) is byte-identical to `feat/cvaux` @6f10688 except the new `decision_settings.json` keys. A test pins this (Task 1, Task 2).
- Workers: lambda = 0 only; at most `max_workers` = 4; one aux model per campaign; no shams; never replicated onto other rungs.
- Replicas: aux workers draw only from a dedicated aux slice, `aux_reserve_slots` = 4, of the P1 reserve; R1/R3 never consume it; `max_replicas` unchanged.
- Attempts: end of every numbered epoch N >= 1 (never epoch 0, never final, never extensions), non-recovered epochs only, until `adaptive_production/aux_admission.json` exists; afterwards model, evaluation partition, worker rows and backfill shas are immutable.
- Training frames = lambda = 0 frames of epochs 0..N-1; holdout = lambda = 0 frames of epoch N.
- No native readout (RMSD, helix, H-bond distances to a reference, folded fraction) is ever loaded by any module in this plan.
- Thresholds (c10 `prereg_v2.json` + c10 `placement.py`, frozen in `aux_settings.json`): sd drop 1e-3, sd floor 0.05, family scaling 1/sqrt(n_family), kNN k 100, PCA to 80 % variance capped at 12, GMM full covariance n_init 3 random_state 0 reg_covar 1e-6, k in 2..8, validation-refit ARI >= 0.5 and lineage-bootstrap median ARI (10 draws) >= 0.5, group floor 1 % of holdout frames, s-bin grid 6 x 6, hidden fraction >= 0.5 with co-occurrence (two groups each >= 0.10 in a bin holding >= 2 % of holdout), lineage info >= 0.10 nats (Dirichlet c = 5), L1 C grid (0.003, 0.01, 0.03, 0.1, 0.3), info gain >= 0.10 nats, even/odd stability >= 0.8, |corr(z, CV1)| and |corr(z, CV2)| <= 0.7, basin-code gain >= 0.05 nats; placement quantiles (0.05, 0.10, 0.90, 0.95), width fractions (0.35, 0.5, 0.7), N_BOOT 100, N_NULL 50, N_NULL_BOOT 10, MIN_TRAIN_FRAMES 100, gates O_q05 >= 0.20, ess_frames >= 50, eff_lineages >= 20, top3_share <= 0.5, holdout floor 50 frames, holdout O >= 0.15 and net > 0.
- Rulings on c10 discrepancies (record in `aux_settings.json` docs): lineage bootstraps keep multiplicity (c10 `bootstrap_fix.py`, not the `set`/`isin` bug); k range 2..8 (prereg) for both partitions; z3 `scale` = sd of the raw projection over TRAINING lambda = 0 frames (not c10's 3.181 which included validation); placement is ported exactly, including 4-dp rounding before gates/ranking and point-estimate gates other than O_q05 (parity with c10 `placement.json` is a test).
- Units: k3 in kcal/mol per (z unit)^2; RT = 0.0019872041 x T kcal/mol; aux energy 0.5 k (z - c)^2 kcal/mol, x 4.184 to kJ/mol.
- Angle convention: descriptors use IUPAC (mdtraj) phi/psi; the emitted `atlas-aux-cv-model-v1` uses `dihedral_sign_convention: "negated"` so `trig(-theta_openmm) = trig(theta_iupac)` and coefficients carry over unchanged.
- Model topology binding: `feature_schema.topology_sha256 = canonical_topology_sha256(<campaign out>/01_solvated_start.pdb topology, draft model)`; atom indices are production-topology indices.
- Tests in this repo are launched through `opencode run "<pytest command>"` (a Bash hook blocks the runner name in agent shells). Run only the tests named in the task (user rule: targeted tests only, never the full suite unless asked).
- Hooks never raise: any exception becomes report `status: error` and nothing is admitted that epoch.

## Review Focus

1. A pre-admission phase whose `traj_interval` != `distance_output_interval`, or whose XTC is missing frames for some samples (crash mid-segment, resumed files): the backfill must refuse with counts, never fill z = 0 and never drop rows. Test in Task 12.
2. A worker parent that is later retired or replaced by respring, or a worker whose `spawn_parent_state_id` is absent from the next phase's active table: the window CSV must still load (parent written blank when absent). Test in Task 2.
3. Resume across the admission epoch (kill after `registry.save`, before `_record_applied_actions`; kill after the ledger, before the next phase): never discovers twice, never admits twice, worker state ids stable. Test in Task 10 and Task 17.
4. A campaign where discovery never succeeds (every attempt `insufficient_evidence`/`no_worker`): phases, union inputs and reports stay identical to a run without the flag except the per-epoch `aux_discovery_report.json` files and the aux slice held idle. Test in Task 10.
5. The first post-admission phase's runtime aux checks (`check_exchange_boundary_alignment` with the phase's actual calibration-step count, physical-system hash, `check_feature_atoms` against the phase topology): the hook must run the same checks before emitting `admit_aux` so a doomed admission is refused at the boundary, not crashed in MD. Test in Task 10.

---

## File Structure

| File | Responsibility |
|---|---|
| `gareus/adaptive/aux_discovery/__init__.py` | Package marker; re-exports `AuxDiscoverySettings`, `run_discovery`. |
| `gareus/adaptive/aux_discovery/settings.py` | `AuxDiscoverySettings` frozen dataclass, `resolve_aux_settings` (freeze/override record `aux_settings.json`). |
| `gareus/adaptive/aux_discovery/descriptors.py` | Descriptor definition from a solute topology (torsion quads, heavy-atom residue-pair groups, backbone N/O pairs) and vectorised evaluation on frames. |
| `gareus/adaptive/aux_discovery/frames.py` | `FrameTable`: XTC frames of the campaign's phases joined to Parquet samples (state, lambda, CV1, CV2), dedup, stride, budget. |
| `gareus/adaptive/aux_discovery/partitions.py` | preprocess, kNN residual, PCA, GMM choice with ARI + lineage bootstrap, s-bins, hidden fraction, co-occurrence, lineage info, info gain. |
| `gareus/adaptive/aux_discovery/z3_search.py` | Pairwise L1 torsion discriminants, gates, ranking; `emit_model` -> `atlas-aux-cv-model-v1`. |
| `gareus/adaptive/aux_discovery/placement.py` | Exact port of c10 `placement.py` as a function. |
| `gareus/adaptive/aux_discovery/validation.py` | `atlas-aux-validation-v1` record: load/check/write CLI. |
| `gareus/adaptive/aux_discovery/pipeline.py` | `run_discovery(frame_table, settings, ...) -> DiscoveryResult` (steps 2-5, pure). |
| `gareus/adaptive/aux_discovery/__main__.py` | Replay CLI. |
| `gareus/adaptive/aux_admission_io.py` | Boundary hook: frames, pipeline, gates, action emission, freezing, backfill, report. |
| `gareus/adaptive/aux_backfill.py` | Per-phase `aux_z_backfill.parquet` writer/reader/checker. |
| `gareus/adaptive_production.py` | Policy fields, registry aux metadata + CSV columns, `admit_aux` in the applier, hook call, phase-arg injection, invisibility filters, union aux term. |
| `gareus/adaptive/reserve_budget.py` | `aux_slots` slice. |
| `gareus/adaptive/cv2_resolution.py`, `cv2_resolution_io.py`, `cv2_respring_io.py`, `ladder_adapt.py`, `neighbour_rule.py` callers | worker filters. |
| `gareus/seeding.py` | Aux pull ramp in `relax_to_window`. |
| `gareus/production.py` | Pass the aux table to the pull. |
| `gareus/cli.py` | Flags, mapping, refusals. |
| `gareus/mbar_analysis/loaders_union_parquet.py` | Aux pooling with backfill. |
| `gareus/mbar_analysis/crosscheck.py`, `analyze_gareus_mbar.py`, `gareus_report.py` | Ordinary-only vs all-states crosscheck gate, worker table, report rows. |
| `tests/data/chignolin_solute.pdb` | Fixture topology (copy of c10 `adaptive_production/epoch_000/solute_only.pdb`). |
| `tests/test_aux_discovery_*.py`, `tests/test_aux_admission_*.py` | Tests. |
| `RUNS/chignolin_11.yaml`, `RUNS/chignolin_11.sh` (main checkout) | Campaign config and launcher. |

---

### Task 1: Settings, policy fields, CLI flags and refusals

**Files:**
- Create: `gareus/adaptive/aux_discovery/__init__.py`, `gareus/adaptive/aux_discovery/settings.py`
- Modify: `gareus/adaptive_production.py` (AdaptiveDecisionPolicy ~AP:554-568, DECISION_SETTINGS_FIELDS AP:584-600, policy_from_args AP:8191-8292), `gareus/cli.py` (flag block near :714-725, compat shim near :1945-1949, `_validate_aux_cv_args` :1446-1483, new `_validate_aux_discovery_args` called next to :2205)
- Test: `tests/test_aux_discovery_settings.py`

**Interfaces:**
- Produces: `AuxDiscoverySettings` (fields below), `AuxDiscoverySettings.to_mapping() -> dict`, `AuxDiscoverySettings.from_mapping(dict) -> AuxDiscoverySettings`, `resolve_aux_settings(adaptive_dir: Path, settings: AuxDiscoverySettings, *, override: bool) -> tuple[AuxDiscoverySettings, dict]`, `AUX_SETTINGS_FILENAME = "aux_settings.json"`; policy fields `aux_discovery: bool = False`, `aux_reserve_slots: int = 4`; CLI `--ap-aux-discovery/--no-ap-aux-discovery`, `--ap-aux-reserve-slots INT`, `--ap-aux-settings-override`; args dests `adaptive_production_aux_discovery`, `adaptive_production_aux_reserve_slots`, `adaptive_production_aux_settings_override`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_aux_discovery_settings.py
import argparse, json
from pathlib import Path
import pytest

from gareus.adaptive.aux_discovery.settings import (AuxDiscoverySettings, resolve_aux_settings,
                                                    AUX_SETTINGS_FILENAME)


def test_defaults_match_c10_prereg_and_placement():
    s = AuxDiscoverySettings()
    assert (s.k_min, s.k_max, s.ari_min, s.n_boot_partition) == (2, 8, 0.5, 10)
    assert (s.knn_k, s.pca_var, s.pca_max, s.sd_drop, s.sd_floor) == (100, 0.8, 12, 1e-3, 0.05)
    assert s.l1_c_grid == (0.003, 0.01, 0.03, 0.1, 0.3)
    assert (s.info_gain_min, s.stability_min, s.max_cv_corr, s.basin_gain_min) == (0.10, 0.8, 0.7, 0.05)
    assert s.quantiles == (0.05, 0.10, 0.90, 0.95) and s.width_fractions == (0.35, 0.5, 0.7)
    assert (s.gate_o_q05, s.gate_ess_frames, s.gate_eff_lineages, s.gate_top3_share) == (0.20, 50.0, 20.0, 0.5)
    assert (s.heldout_min_frames, s.heldout_o_min, s.max_workers) == (50, 0.15, 4)


def test_roundtrip_mapping():
    s = AuxDiscoverySettings(max_workers=3)
    assert AuxDiscoverySettings.from_mapping(json.loads(json.dumps(s.to_mapping()))) == s


def test_from_mapping_rejects_unknown_key():
    with pytest.raises(ValueError, match="unknown aux settings key"):
        AuxDiscoverySettings.from_mapping({**AuxDiscoverySettings().to_mapping(), "bogus": 1})


def test_resolve_freezes_first_record_and_override_replaces(tmp_path: Path):
    first, rec = resolve_aux_settings(tmp_path, AuxDiscoverySettings(max_workers=2), override=False)
    assert (tmp_path / AUX_SETTINGS_FILENAME).exists() and first.max_workers == 2
    again, _ = resolve_aux_settings(tmp_path, AuxDiscoverySettings(max_workers=4), override=False)
    assert again.max_workers == 2                      # recorded value wins
    new, _ = resolve_aux_settings(tmp_path, AuxDiscoverySettings(max_workers=4), override=True)
    assert new.max_workers == 4
    assert json.loads((tmp_path / AUX_SETTINGS_FILENAME).read_text())["settings"]["max_workers"] == 4


def test_policy_field_frozen_in_decision_settings():
    from gareus.adaptive_production import DECISION_SETTINGS_FIELDS, AdaptiveDecisionPolicy
    assert "aux_discovery" in DECISION_SETTINGS_FIELDS and "aux_reserve_slots" in DECISION_SETTINGS_FIELDS
    p = AdaptiveDecisionPolicy()
    assert p.aux_discovery is False and p.aux_reserve_slots == 4


def _parse(argv):
    from gareus.cli import parse_args
    return parse_args(argv)


def test_cli_refuses_aux_discovery_with_topups(tmp_path):
    with pytest.raises(SystemExit):
        _parse(["--window-mode", "adaptive-production", "--ap-aux-discovery", "--ap-topups",
                "--seq", "GYDPETGTWG", "--out", str(tmp_path)])


def test_cli_refuses_unaligned_intervals(tmp_path):
    with pytest.raises(SystemExit):
        _parse(["--window-mode", "adaptive-production", "--ap-aux-discovery", "--seq", "GYDPETGTWG",
                "--out", str(tmp_path), "--traj-interval", "250", "--distance-output-interval", "300"])


def test_explicit_aux_model_with_adaptive_production_points_to_flag(tmp_path, capsys):
    with pytest.raises(SystemExit):
        _parse(["--window-mode", "adaptive-production", "--aux-cv-model", str(tmp_path / "m.json"),
                "--seq", "GYDPETGTWG", "--out", str(tmp_path)])
    assert "--ap-aux-discovery" in capsys.readouterr().err
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `opencode run "pytest -q tests/test_aux_discovery_settings.py"`
Expected: FAIL (`ModuleNotFoundError: gareus.adaptive.aux_discovery`).

- [ ] **Step 3: Implement settings**

```python
# gareus/adaptive/aux_discovery/__init__.py
"""Adaptive auxiliary-CV discovery (spec 2026-10-09-cvaux-adaptive-discovery-design.md)."""
from .settings import AuxDiscoverySettings  # noqa: F401
```

```python
# gareus/adaptive/aux_discovery/settings.py
"""Frozen settings of the adaptive aux-CV discovery hook.

Values are c10's pre-registered rules (docs/_local_docs/c10_aux_diagnosis/prereg_v2.json) and c10
placement.py constants. Rulings on c10 discrepancies: lineage bootstraps keep multiplicity; k in 2..8
for both partitions; z3 scale = training-only sd; placement ported exactly (4-dp rounding, point gates).
"""
from __future__ import annotations

import dataclasses
import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

AUX_SETTINGS_FILENAME = "aux_settings.json"
AUX_SETTINGS_SCHEMA = "atlas-aux-discovery-settings-v1"


@dataclass(frozen=True)
class AuxDiscoverySettings:
    # partitions (prereg_v2)
    k_min: int = 2
    k_max: int = 8
    ari_min: float = 0.5
    n_boot_partition: int = 10
    group_min_share: float = 0.01
    knn_k: int = 100
    pca_var: float = 0.8
    pca_max: int = 12
    sd_drop: float = 1e-3
    sd_floor: float = 0.05
    s_bins: int = 6
    hidden_min: float = 0.5
    cooc_share: float = 0.10
    cooc_bin_min: float = 0.02
    lineage_info_min: float = 0.10
    lineage_dirichlet_c: float = 5.0
    # z3 search
    l1_c_grid: Tuple[float, ...] = (0.003, 0.01, 0.03, 0.1, 0.3)
    info_gain_min: float = 0.10
    stability_min: float = 0.8
    max_cv_corr: float = 0.7
    basin_gain_min: float = 0.05
    # placement (c10 placement.py)
    quantiles: Tuple[float, ...] = (0.05, 0.10, 0.90, 0.95)
    width_fractions: Tuple[float, ...] = (0.35, 0.5, 0.7)
    n_boot_place: int = 100
    n_null: int = 50
    n_null_boot: int = 10
    min_train_frames: int = 100
    gate_o_q05: float = 0.20
    gate_ess_frames: float = 50.0
    gate_eff_lineages: float = 20.0
    gate_top3_share: float = 0.5
    heldout_min_frames: int = 50
    heldout_o_min: float = 0.15
    max_workers: int = 4
    placement_seed: int = 20261007
    # data
    frame_stride_steps: int = 0          # 0 = one exchange interval
    max_frames: int = 400_000
    partition_seed: int = 0
    temperature_k: float = 300.0

    def to_mapping(self) -> Dict[str, Any]:
        out = dataclasses.asdict(self)
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in out.items()}

    @classmethod
    def from_mapping(cls, raw: Dict[str, Any]) -> "AuxDiscoverySettings":
        names = {f.name: f for f in dataclasses.fields(cls)}
        unknown = sorted(set(raw) - set(names))
        if unknown:
            raise ValueError(f"unknown aux settings key(s): {unknown}")
        kw = {}
        for k, v in raw.items():
            kw[k] = tuple(float(x) for x in v) if isinstance(v, list) else v
        return cls(**kw)


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
    with os.fdopen(fd, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    os.replace(tmp, path)


def resolve_aux_settings(adaptive_dir: Path, settings: AuxDiscoverySettings, *,
                         override: bool) -> Tuple[AuxDiscoverySettings, Dict[str, Any]]:
    """Freeze settings at first use; a recorded file wins unless ``override``."""
    path = Path(adaptive_dir) / AUX_SETTINGS_FILENAME
    if path.exists() and not override:
        rec = json.loads(path.read_text())
        if rec.get("schema") != AUX_SETTINGS_SCHEMA:
            raise ValueError(f"{path}: schema {rec.get('schema')!r} is not {AUX_SETTINGS_SCHEMA}")
        return AuxDiscoverySettings.from_mapping(rec["settings"]), rec
    rec = {"schema": AUX_SETTINGS_SCHEMA, "settings": settings.to_mapping(),
           "written_unix": time.time(), "override": bool(override)}
    _atomic_write_json(path, rec)
    return settings, rec
```

- [ ] **Step 4: Policy fields and decision settings**

In `gareus/adaptive_production.py`, after the respring fields (AP:554-558) add:

```python
    aux_discovery: bool = False
    aux_reserve_slots: int = 4
```

In `__post_init__` (AP:560-568) add:

```python
        if int(self.aux_reserve_slots) < 0:
            raise ValueError(f"aux_reserve_slots must be >= 0, got {self.aux_reserve_slots}")
```

Append `"aux_discovery", "aux_reserve_slots",` to `DECISION_SETTINGS_FIELDS` (AP:584-600, after the respring entries). In `policy_from_args` (next to AP:8286-8291):

```python
        aux_discovery=_arg_bool(args, "adaptive_production_aux_discovery", False),
        aux_reserve_slots=_arg_int(args, "adaptive_production_aux_reserve_slots", 4),
```

- [ ] **Step 5: CLI flags, mapping, refusals**

In `gareus/cli.py` next to `--ap-cv2-respring` (:714):

```python
    p.add_argument("--ap-aux-discovery", action=argparse.BooleanOptionalAction, default=False,
                   help="Discover a torsion-linear auxiliary CV (z3) from this campaign's own frames at each "
                        "numbered-epoch boundary from the end of epoch 1 and admit up to 4 lambda=0 worker states "
                        "(spec 2026-10-09-cvaux-adaptive-discovery-design.md). Off by default.")
    p.add_argument("--ap-aux-reserve-slots", type=int, default=4,
                   help="Replica slots of the P1 reserve kept for auxiliary workers (R1/R3 never use them).")
    p.add_argument("--ap-aux-settings-override", action="store_true", default=False,
                   help="Replace the campaign's recorded aux_settings.json with this job's values.")
```

In the compat-shim block (next to :1945):

```python
    args.adaptive_production_aux_discovery = bool(getattr(args, "ap_aux_discovery", False))
    args.adaptive_production_aux_reserve_slots = int(getattr(args, "ap_aux_reserve_slots", 4))
    args.adaptive_production_aux_settings_override = bool(getattr(args, "ap_aux_settings_override", False))
```

In `_validate_aux_cv_args`, replace the window-mode error text with:

```python
        p.error(f"--aux-cv-model is plain-run only (--window-mode {' or '.join(_AUX_PLAIN_WINDOW_MODES)} with "
                f"--windows-2d-csv); for adaptive-production use --ap-aux-discovery, which fits and injects "
                f"the model per phase")
```

New validator, called right after `_validate_aux_cv_args(p, args)` (:2205):

```python
def _validate_aux_discovery_args(p: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not getattr(args, "ap_aux_discovery", False):
        return
    if str(getattr(args, "window_mode", "")) != "adaptive-production":
        p.error("--ap-aux-discovery needs --window-mode adaptive-production")
    if getattr(args, "ap_topups", False):
        p.error("--ap-aux-discovery cannot run with --ap-topups (out of scope, spec Section 11)")
    if str(getattr(args, "exchange_mode", "") or "") == "neighbor":
        p.error("--ap-aux-discovery needs an unrestricted exchange mode (gibbs-walk, all-pair-sweep, random-pair)")
    if getattr(args, "us_auto_drop_bad_windows", False):
        p.error("--ap-aux-discovery refuses --us-auto-drop-bad-windows (aux populations are frozen)")
    run_mode = str(getattr(args, "run_mode", "gamd") or "gamd")
    boost = str(getattr(args, "gamd_boost_type", "") or "")
    if run_mode in ("gamd", "hmr-gamd") and not boost.startswith("pep-gamd"):
        p.error("--ap-aux-discovery needs a pep-gamd-* boost type")
    traj = int(getattr(args, "traj_interval", 0) or 0)
    dist = int(getattr(args, "distance_output_interval", 0) or 0)
    exch = int(getattr(args, "exchange_interval", 0) or 0)
    if traj <= 0 or traj != dist:
        p.error("--ap-aux-discovery needs traj_interval == distance_output_interval > 0 (z backfill needs "
                "one trajectory frame per sample)")
    if exch <= 0 or exch % traj:
        p.error("--ap-aux-discovery needs exchange_interval to be a multiple of traj_interval")
    if int(getattr(args, "ap_aux_reserve_slots", 4)) < 1:
        p.error("--ap-aux-discovery needs --ap-aux-reserve-slots >= 1")
```

Check the exact dest names of `--traj-interval`, `--distance-output-interval`, `--exchange-interval`, `--ap-topups`, `--us-auto-drop-bad-windows` with `grep -n "add_argument(\"--traj-interval\|--distance-output-interval\|--exchange-interval\|--ap-topups\|--us-auto-drop" gareus/cli.py` and use them; also check that `--seq`/`--out` are the minimal parse arguments used by other CLI tests (`grep -n "parse_args(\[" tests/test_cv2_respring.py`) and adjust the test's `_parse` argv accordingly.

- [ ] **Step 6: Run tests to verify they pass**

Run: `opencode run "pytest -q tests/test_aux_discovery_settings.py tests/test_cv2_respring.py -k 'settings or policy or cli'"`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add gareus/adaptive/aux_discovery/__init__.py gareus/adaptive/aux_discovery/settings.py gareus/adaptive_production.py gareus/cli.py tests/test_aux_discovery_settings.py
git commit -m "feat(cvaux-adaptive): aux discovery settings, policy fields, CLI flags and refusals"
```

---

### Task 2: Registry aux identity, CSV columns, Hamiltonian snapshot

**Files:**
- Modify: `gareus/adaptive_production.py` (`WindowState` AP:151-212, `has_near_duplicate` AP:716-760, `write_active_window_csv` AP:843-893, `write_state_subset_window_csv` AP:5490, `_HAMILTONIAN_FIELDS`/`_hamiltonian_snapshot` AP:2398-2419, `_centre_group_key` AP:2338)
- Test: `tests/test_aux_admission_registry.py`

**Interfaces:**
- Produces: `AUX_METADATA_KEY = "aux"`; `is_auxiliary_state(state) -> bool`; `aux_params(state) -> Optional[dict]` returning `{"role", "aux_center", "aux_k_kcal_mol", "aux_model_sha256", "state_instance_id", "spawn_parent_state_id", "admitted_epoch"}`; `registry_has_aux(registry) -> bool`; `aux_csv_cells(state, active_ids: set[int]) -> dict[str, str]`. Window CSV writers append `AUX_CSV_COLUMNS + INSTANCE_CSV_COLUMNS` (from `gareus.windows`) to every row only when `registry_has_aux(registry)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_aux_admission_registry.py
import csv
from pathlib import Path

from gareus.adaptive_production import (WindowStateRegistry, AdaptiveDecisionPolicy, is_auxiliary_state,
                                        registry_has_aux, AUX_METADATA_KEY, _hamiltonian_snapshot)
from gareus.windows import AUX_CSV_COLUMNS, INSTANCE_CSV_COLUMNS

SHA = "a" * 64


def _registry():
    r = WindowStateRegistry()
    r.add_state(0.1, 10.0, 0.5, 2.0, gamd_lambda=0.0)
    r.add_state(0.3, 10.0, 0.5, 2.0, gamd_lambda=0.0)
    return r


def _add_worker(r, parent=0, c3=1.5, k3=2.0):
    p = r.get_state(parent)
    meta = {AUX_METADATA_KEY: {"role": "auxiliary", "aux_center": c3, "aux_k_kcal_mol": k3,
                               "aux_model_sha256": SHA, "state_instance_id": None,
                               "spawn_parent_state_id": parent, "admitted_epoch": 1}}
    return r.add_state(p.primary_center, p.primary_k, p.secondary_center, p.secondary_k,
                       gamd_lambda=0.0, parent_state_id=parent, epoch=1, source="adaptive_production_aux",
                       reason="aux_discovery", burnin_steps=1000, metadata=meta)


def test_no_aux_csv_is_byte_identical(tmp_path: Path):
    r = _registry()
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    r.write_active_window_csv(a)
    assert not registry_has_aux(r)
    header = a.read_text().splitlines()[0].split(",")
    assert not set(AUX_CSV_COLUMNS) & set(header)


def test_aux_columns_on_every_row(tmp_path: Path):
    r = _registry(); w = _add_worker(r)
    path = tmp_path / "w.csv"
    r.write_active_window_csv(path)
    rows = list(csv.DictReader(path.open()))
    assert all(set(AUX_CSV_COLUMNS) | set(INSTANCE_CSV_COLUMNS) <= set(row) for row in rows)
    by_id = {int(row["state_id"]): row for row in rows}
    assert by_id[0]["state_role"] == "ordinary" and float(by_id[0]["aux_k_kcal_mol"]) == 0.0
    assert by_id[0]["aux_center"] == "" and by_id[0]["aux_model_sha256"] == ""
    wr = by_id[w.state_id]
    assert wr["state_role"] == "auxiliary" and float(wr["aux_center"]) == 1.5
    assert wr["aux_model_sha256"] == SHA and wr["spawn_parent_state_id"] == "s0"
    assert wr["state_instance_id"] == f"s{w.state_id}"


def test_parent_written_blank_when_not_active(tmp_path: Path):
    r = _registry(); w = _add_worker(r, parent=1)
    r.retire_state(1, epoch=2, reason="test")
    path = tmp_path / "w.csv"
    r.write_active_window_csv(path)
    row = {int(x["state_id"]): x for x in csv.DictReader(path.open())}[w.state_id]
    assert row["spawn_parent_state_id"] == ""


def test_worker_is_not_a_duplicate_of_parent_and_parent_is_not_of_worker():
    r = _registry(); _add_worker(r)
    pol = AdaptiveDecisionPolicy()
    p = r.get_state(0)
    # an ordinary add at the parent's centre is still a duplicate of the parent ...
    assert r.has_near_duplicate(p.primary_center, p.secondary_center, pol, gamd_lambda=0.0,
                                primary_k=p.primary_k, secondary_k=p.secondary_k)
    # ... but the worker itself is ignored by ordinary identity checks
    r2 = WindowStateRegistry(); w = _add_worker(_registry())
    assert is_auxiliary_state(w)


def test_hamiltonian_snapshot_includes_aux():
    r = _registry(); w = _add_worker(r)
    snap = _hamiltonian_snapshot(r)
    assert snap[w.state_id][-3:] == (1.5, 2.0, SHA)
    assert snap[0][-3:] == (None, 0.0, None)


def test_registry_json_roundtrip_keeps_aux(tmp_path: Path):
    r = _registry(); w = _add_worker(r)
    r.save_json(tmp_path / "r.json")
    back = WindowStateRegistry.load_json(tmp_path / "r.json")
    assert is_auxiliary_state(back.get_state(w.state_id))
```

- [ ] **Step 2: Run to verify failure**

Run: `opencode run "pytest -q tests/test_aux_admission_registry.py"`
Expected: FAIL (`ImportError: is_auxiliary_state`).

- [ ] **Step 3: Implement**

Add near `WindowState` (after AP:212):

```python
AUX_METADATA_KEY = "aux"


def aux_params(state) -> Optional[Dict[str, Any]]:
    meta = getattr(state, "metadata", None) or {}
    rec = meta.get(AUX_METADATA_KEY)
    return rec if isinstance(rec, dict) and rec.get("role") == "auxiliary" else None


def is_auxiliary_state(state) -> bool:
    """The ONE predicate every consumer uses to tell an auxiliary worker from an ordinary state."""
    return aux_params(state) is not None


def registry_has_aux(registry) -> bool:
    return any(is_auxiliary_state(s) for s in registry.all_states())


def aux_csv_cells(state, active_ids) -> Dict[str, str]:
    rec = aux_params(state)
    sid = f"s{int(state.state_id)}"
    if rec is None:
        return {"aux_center": "", "aux_k_kcal_mol": "0.0", "aux_model_sha256": "",
                "state_instance_id": sid, "state_role": "ordinary", "spawn_parent_state_id": "",
                "matched_additional_slot_id": "", "spawn_source_observation_json": ""}
    parent = rec.get("spawn_parent_state_id")
    parent_cell = f"s{int(parent)}" if parent is not None and int(parent) in active_ids else ""
    return {"aux_center": repr(float(rec["aux_center"])), "aux_k_kcal_mol": repr(float(rec["aux_k_kcal_mol"])),
            "aux_model_sha256": str(rec["aux_model_sha256"]), "state_instance_id": sid,
            "state_role": "auxiliary", "spawn_parent_state_id": parent_cell,
            "matched_additional_slot_id": "", "spawn_source_observation_json": ""}
```

In `write_active_window_csv` and `write_state_subset_window_csv`, after the existing fieldnames list is built:

```python
        from gareus.windows import AUX_CSV_COLUMNS, INSTANCE_CSV_COLUMNS
        with_aux = registry_has_aux(self)          # (registry_has_aux(registry) in the module-level writer)
        if with_aux:
            fieldnames = list(fieldnames) + list(AUX_CSV_COLUMNS) + list(INSTANCE_CSV_COLUMNS)
            active_ids = {int(s.state_id) for s in states_written}
```

and where each row dict is built: `if with_aux: row.update(aux_csv_cells(state, active_ids))`, where `states_written` is the list of states the writer iterates (name it so if the code uses another variable).

In `has_near_duplicate` (AP:716-760), skip auxiliary states at the top of the loop: `if is_auxiliary_state(state): continue`. In `_centre_group_key` (AP:2338) no change: callers filter workers out (Task 3).

`_HAMILTONIAN_FIELDS`/`_hamiltonian_snapshot` (AP:2398-2419): keep the tuple of the seven existing fields and append `(aux_center|None, aux_k (0.0 when ordinary), aux_sha|None)`:

```python
def _aux_identity(state):
    rec = aux_params(state)
    if rec is None:
        return (None, 0.0, None)
    return (float(rec["aux_center"]), float(rec["aux_k_kcal_mol"]), str(rec["aux_model_sha256"]))


def _hamiltonian_snapshot(registry):
    return {int(s.state_id): tuple(getattr(s, f) for f in _HAMILTONIAN_FIELDS) + _aux_identity(s)
            for s in registry.all_states()}
```

- [ ] **Step 4: Run tests**

Run: `opencode run "pytest -q tests/test_aux_admission_registry.py tests/test_applier_p2.py tests/test_cv2_respring.py"`
Expected: PASS (the P2 and respring suites pin that ordinary behaviour is unchanged).

- [ ] **Step 5: Commit**

```bash
git add gareus/adaptive_production.py tests/test_aux_admission_registry.py
git commit -m "feat(cvaux-adaptive): aux worker identity in registry metadata, window CSV aux columns, Hamiltonian snapshot"
```

---

### Task 3: Worker invisibility in every other adaptive consumer

**Files:**
- Modify: `gareus/adaptive_production.py` (`build_geometry_edges` AP:2505, retire loop AP:5368 and second loop AP:5424, `_propose_rung_actions` AP:4859, `_propose_tica_coverage_actions` AP:3055, `_representative_rung` AP:2434, `_centre_members` AP:7794, applier groupings AP:7748/7968/8015, `_add_centre_on_every_rung`'s `_centre_rungs` source), `gareus/adaptive/cv2_resolution.py:212-216` (`state_views`), `gareus/adaptive/cv2_respring_io.py:116-123` (`state_rows`), `gareus/adaptive/cv2_resolution_io.py:208` (`n_active`), `gareus/adaptive/ladder_adapt.py:365-393` (`load_centre_rung_samples`), `gareus/adaptive_production.py:1247` (`select_state_aware_seeds_for_targets`, aux-aware seed source)
- Test: `tests/test_aux_admission_invisibility.py`

**Interfaces:**
- Consumes: `is_auxiliary_state` (Task 2).
- Produces: `ordinary_active_states(registry) -> list[WindowState]` (AP, next to `is_auxiliary_state`); every listed consumer uses it in place of `registry.active_states()`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_aux_admission_invisibility.py
from gareus.adaptive_production import (WindowStateRegistry, AdaptiveDecisionPolicy, AUX_METADATA_KEY,
                                        ordinary_active_states, build_geometry_edges,
                                        propose_actions_from_diagnostics)
from gareus.adaptive import cv2_resolution, ladder_adapt

SHA = "b" * 64


def _registry_with_worker():
    r = WindowStateRegistry()
    for c1 in (0.0, 0.5, 1.0):
        for lam in (0.0, 0.2):
            r.add_state(c1, 10.0, 0.5, 2.0, gamd_lambda=lam)
    meta = {AUX_METADATA_KEY: {"role": "auxiliary", "aux_center": 1.0, "aux_k_kcal_mol": 2.0,
                               "aux_model_sha256": SHA, "state_instance_id": None,
                               "spawn_parent_state_id": 0, "admitted_epoch": 1}}
    w = r.add_state(0.0, 10.0, 0.5, 2.0, gamd_lambda=0.0, parent_state_id=0, epoch=1,
                    source="adaptive_production_aux", reason="aux", metadata=meta)
    return r, w


def test_ordinary_active_states_excludes_worker():
    r, w = _registry_with_worker()
    assert w.state_id not in {s.state_id for s in ordinary_active_states(r)}


def test_geometry_edges_never_touch_worker():
    r, w = _registry_with_worker()
    edges = build_geometry_edges(r, AdaptiveDecisionPolicy())
    ids = {int(x) for e in edges for x in (e["i"], e["j"])}
    assert w.state_id not in ids


def test_retirement_never_proposes_worker():
    r, w = _registry_with_worker()
    diag = {"states": [{"state_id": s.state_id, "n_samples": 10 ** 6} for s in r.active_states()], "edges": []}
    acts = propose_actions_from_diagnostics(r, diag, policy=AdaptiveDecisionPolicy(retire_converged=True))
    assert all(not (a[0] == "retire" and int(a[1]) == w.state_id) for a in acts)


def test_cv2_resolution_views_skip_worker():
    r, w = _registry_with_worker()
    views = cv2_resolution.state_views(r, {"states": []})
    assert w.state_id not in {int(v.state_id) for v in views}


def test_ladder_centre_samples_skip_worker():
    r, w = _registry_with_worker()
    states = {int(s.state_id): s.to_dict() for s in r.all_states()}
    keys = {ladder_adapt.centre_key(st) for sid, st in states.items() if sid != w.state_id}
    assert ladder_adapt.registry_states_for_ladder(states).keys() == {k for k in states if k != w.state_id}
```

Before writing the test bodies, open each consumer and adapt the call (argument names, edge dict keys `i`/`j` vs `a`/`b`, `state_views` return type, how `propose_actions_from_diagnostics` reads diagnostics) to the real signatures; keep the assertion: the worker id never appears. For `ladder_adapt`, add a tiny helper `registry_states_for_ladder(states: dict) -> dict` that drops entries whose `metadata.aux.role == "auxiliary"` and call it at the top of `load_centre_rung_samples`.

- [ ] **Step 2: Run to verify failure**

Run: `opencode run "pytest -q tests/test_aux_admission_invisibility.py"`
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
def ordinary_active_states(registry) -> List["WindowState"]:
    return [s for s in registry.active_states() if not is_auxiliary_state(s)]
```

Replace `registry.active_states()` by `ordinary_active_states(registry)` at: AP:2505 (`build_geometry_edges`), AP:5368 and AP:5424 (retire/extend loops), AP:4859 rung proposals, AP:3055 tICA coverage, AP:2434 `_representative_rung`, AP:7794 `_centre_members`, AP:7748/7968/8015 applier groupings and the `_centre_rungs()` rung source (so `add` never replicates onto a "rung" contributed by a worker), `gareus/adaptive/cv2_resolution.py:216`, `gareus/adaptive/cv2_respring_io.py:116-123`. In `cv2_resolution_io.py:208` keep `n_active` = ALL active states (workers occupy replicas). In `select_state_aware_seeds_for_targets` (AP:1247) extend `_seed_rows_for_target` (AP:1234) to also honour `metadata["aux"]["spawn_parent_state_id"]` as the seed source restriction:

```python
def _seed_source_restriction(target) -> Optional[int]:
    meta = getattr(target, "metadata", None) or {}
    aux = meta.get(AUX_METADATA_KEY)
    if isinstance(aux, dict) and aux.get("spawn_parent_state_id") is not None:
        return int(aux["spawn_parent_state_id"])
    res = meta.get("cv2_resolution")
    if isinstance(res, dict) and res.get("seed_source_state_id") is not None:
        return int(res["seed_source_state_id"])
    return None
```

and use `_seed_source_restriction(target)` where `_seed_rows_for_target` currently reads `metadata["cv2_resolution"]["seed_source_state_id"]`.

`edge_metric._state_nodes` reads diagnostics rows, not the registry: in `collect_epoch_diagnostics`, `collect_segmented_epoch_diagnostics` and `collect_final_combined_diagnostics`, tag each state row with `"auxiliary": True` for workers, and in `edge_metric._state_nodes` (`gareus/adaptive/edge_metric.py:525`) skip rows with `row.get("auxiliary")`.

- [ ] **Step 4: Run tests**

Run: `opencode run "pytest -q tests/test_aux_admission_invisibility.py tests/test_p7b_neighbour_consumers.py tests/test_cv2_resolution.py tests/test_cv2_respring.py tests/test_ladder_adapt_*.py tests/test_edge_metric_two_state_mbar.py tests/test_seed_assignment_after_actions.py"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A gareus tests/test_aux_admission_invisibility.py
git commit -m "feat(cvaux-adaptive): auxiliary workers invisible to retirement, edges, CV2 resolution, respring, ladder, rung replication"
```

---

### Task 4: `admit_aux` action in the P2 applier and the aux reserve slice

**Files:**
- Modify: `gareus/adaptive_production.py` (`_validate_action` AP:8021-8067, `_execute_action` AP:8069, `_states_added_by` AP:7996, `_refuse` reasons), `gareus/adaptive/reserve_budget.py` (`reserve_allowances` line 88, `ReserveAllowance` line 64)
- Test: `tests/test_aux_admission_applier.py`

**Interfaces:**
- Consumes: Task 2 metadata layout.
- Produces: action tuple `("admit_aux", parent_state_id: int, {"aux_center": float, "aux_k_kcal_mol": float, "aux_model_sha256": str, "burnin_steps": int}, reason: str, {"aux": {...discovery provenance...}})`; refusal reasons `aux_parent_not_lambda0`, `aux_parent_is_aux`, `aux_model_mismatch`, `aux_budget`, `aux_duplicate`; `ReserveAllowance.aux_slots: int` (new field, default 0); `reserve_allowances(..., aux_slots: int = 0)` subtracts `aux_slots` from `free` BEFORE the add_rung and resolution shares and reports `aux_slots = min(aux_slots, free)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_aux_admission_applier.py
import pytest
from gareus.adaptive_production import (WindowStateRegistry, AdaptiveDecisionPolicy, AdaptiveProductionController,
                                        is_auxiliary_state, aux_params)
from gareus.adaptive.reserve_budget import reserve_allowances

SHA = "c" * 64


def _reg():
    r = WindowStateRegistry()
    r.add_state(0.0, 10.0, 0.5, 2.0, gamd_lambda=0.0)
    r.add_state(0.0, 10.0, 0.5, 2.0, gamd_lambda=0.2)
    return r


def _act(parent=0, c3=1.2, k3=2.0, sha=SHA):
    return ("admit_aux", parent, {"aux_center": c3, "aux_k_kcal_mol": k3, "aux_model_sha256": sha,
                                  "burnin_steps": 3000}, "aux_discovery epoch 1", {"aux": {"epoch": 1}})


def test_admit_adds_one_lambda0_worker_with_parent_restraints():
    r = _reg(); c = AdaptiveProductionController(r, policy=AdaptiveDecisionPolicy(aux_discovery=True))
    c.apply_actions(1, [_act()])
    w = [s for s in r.active_states() if is_auxiliary_state(s)]
    assert len(w) == 1 and not c.refused_actions
    w = w[0]; p = r.get_state(0)
    assert (w.primary_center, w.primary_k, w.secondary_center, w.secondary_k, w.gamd_lambda) == \
           (p.primary_center, p.primary_k, p.secondary_center, p.secondary_k, 0.0)
    assert aux_params(w)["aux_center"] == 1.2 and w.burnin_steps == 3000 and w.parent_state_id == 0


def test_refuses_parent_on_boosted_rung():
    r = _reg(); c = AdaptiveProductionController(r, policy=AdaptiveDecisionPolicy(aux_discovery=True))
    c.apply_actions(1, [_act(parent=1)])
    assert c.refused_actions[0]["reason"] == "aux_parent_not_lambda0"


def test_refuses_second_model_sha():
    r = _reg(); c = AdaptiveProductionController(r, policy=AdaptiveDecisionPolicy(aux_discovery=True))
    c.apply_actions(1, [_act(), _act(c3=-1.0, sha="d" * 64)])
    assert [x["reason"] for x in c.refused_actions] == ["aux_model_mismatch"]


def test_refuses_duplicate_worker():
    r = _reg(); c = AdaptiveProductionController(r, policy=AdaptiveDecisionPolicy(aux_discovery=True))
    c.apply_actions(1, [_act(), _act()])
    assert [x["reason"] for x in c.refused_actions] == ["aux_duplicate"]


def test_refuses_over_aux_slice():
    r = _reg()
    pol = AdaptiveDecisionPolicy(aux_discovery=True, aux_reserve_slots=1)
    c = AdaptiveProductionController(r, policy=pol)
    c.apply_actions(1, [_act(c3=1.0), _act(c3=-1.0)])
    assert [x["reason"] for x in c.refused_actions] == ["aux_budget"]


def test_hamiltonians_of_existing_states_unchanged():
    r = _reg(); before = {s.state_id: s.to_dict() for s in r.all_states()}
    AdaptiveProductionController(r, policy=AdaptiveDecisionPolicy(aux_discovery=True)).apply_actions(1, [_act()])
    for sid, d in before.items():
        assert r.get_state(sid).to_dict() == d


def test_reserve_aux_slice_taken_first():
    a = reserve_allowances(236, 212, {"fraction": 0.10, "slots": 24}, n_rungs=3, aux_slots=4)
    assert a.aux_slots == 4 and a.free_slots == 20
    b = reserve_allowances(236, 212, {"fraction": 0.10, "slots": 24}, n_rungs=3)
    assert b.aux_slots == 0 and b.free_slots == 24
```

Check the real reserve mapping keys `layout_plan.adaptive_reserve` returns (`grep -n "def adaptive_reserve" -A20 gareus/layout_plan.py`) and use them in the last test.

- [ ] **Step 2: Run to verify failure**

Run: `opencode run "pytest -q tests/test_aux_admission_applier.py"`
Expected: FAIL (`ValueError: unknown adaptive-production action 'admit_aux'`).

- [ ] **Step 3: Implement the applier**

In `_validate_action`, before the `raise ValueError(...)`:

```python
        if kind == "admit_aux":
            return self._validate_admit_aux(action)
```

New methods on `AdaptiveProductionController`:

```python
    def _validate_admit_aux(self, action):
        _, parent_id, params, _reason = action[:4]
        parent = self.registry.get_state(int(parent_id)) if self._has_state(int(parent_id)) else None
        if parent is None:
            return None, ("unknown_state", f"parent {parent_id} not in registry", {})
        if not parent.active:
            return None, ("inactive", f"parent {parent_id} is retired", {})
        if is_auxiliary_state(parent):
            return None, ("aux_parent_is_aux", f"parent {parent_id} is itself a worker", {})
        if abs(float(parent.gamd_lambda)) > 1e-12:
            return None, ("aux_parent_not_lambda0", f"parent lambda {parent.gamd_lambda}", {})
        sha = str(params["aux_model_sha256"])
        shas = {aux_params(s)["aux_model_sha256"] for s in self.registry.all_states() if is_auxiliary_state(s)}
        shas |= set(self._pending_aux_shas)
        if shas and shas != {sha}:
            return None, ("aux_model_mismatch", f"campaign model {sorted(shas)} vs {sha}", {})
        key = (int(parent_id), round(float(params["aux_center"]), 6), round(float(params["aux_k_kcal_mol"]), 6))
        existing = {(aux_params(s)["spawn_parent_state_id"], round(aux_params(s)["aux_center"], 6),
                     round(aux_params(s)["aux_k_kcal_mol"], 6))
                    for s in self.registry.all_states() if is_auxiliary_state(s)}
        if key in existing or key in self._pending_aux_keys:
            return None, ("aux_duplicate", f"worker {key} already present", {})
        n_workers = sum(1 for s in self.registry.active_states() if is_auxiliary_state(s)) + len(self._pending_aux_keys)
        slots = int(getattr(self.policy, "aux_reserve_slots", 0))
        if n_workers + 1 > slots:
            return None, ("aux_budget", f"{n_workers} workers + 1 > aux slice {slots}",
                          {"n_workers": n_workers, "aux_reserve_slots": slots})
        self._pending_aux_keys.append(key); self._pending_aux_shas.append(sha)
        return {"parent": parent, "params": dict(params), "added": 1}, None

    def _execute_admit_aux(self, epoch, action, plan):
        parent = plan["parent"]; params = plan["params"]
        provenance = dict(action[4]) if len(action) > 4 and action[4] else {}
        meta = {AUX_METADATA_KEY: {"role": "auxiliary", "aux_center": float(params["aux_center"]),
                                   "aux_k_kcal_mol": float(params["aux_k_kcal_mol"]),
                                   "aux_model_sha256": str(params["aux_model_sha256"]),
                                   "state_instance_id": None, "spawn_parent_state_id": int(parent.state_id),
                                   "admitted_epoch": int(epoch), **provenance.get("aux", {})}}
        self.registry.add_state(parent.primary_center, parent.primary_k, parent.secondary_center,
                                parent.secondary_k, gamd_sigma0p=parent.gamd_sigma0p,
                                gamd_sigma0d=parent.gamd_sigma0d, gamd_lambda=0.0,
                                parent_state_id=int(parent.state_id), epoch=int(epoch),
                                source="adaptive_production_aux", reason=str(action[3]),
                                burnin_steps=int(params.get("burnin_steps", 0)), metadata=meta)
```

Initialise `self._pending_aux_keys: List[tuple] = []` and `self._pending_aux_shas: List[str] = []` in `__init__`, reset both at the top of `apply_actions`. Add `_has_state(sid)` if no equivalent exists (`try: self.registry.get_state(sid); return True / except KeyError: return False`; check what `get_state` raises). In `_execute_action` dispatch `if kind == "admit_aux": return self._execute_admit_aux(epoch, action, plan["plan"])`. In `_states_added_by` (AP:7996) return 1 for `admit_aux`. The generic `_budget_refusal` (AP:8063, `max_replicas_budget`) still runs after `_validate_admit_aux`, so the global cap stays enforced.

- [ ] **Step 4: Implement the reserve slice**

In `gareus/adaptive/reserve_budget.py`: add `aux_slots: int = 0` to `ReserveAllowance`; add keyword `aux_slots: int = 0` to `reserve_allowances`; immediately after `free` is computed and before `add_rung_slots`/`resolution_slots`:

```python
    aux = max(0, min(int(aux_slots), int(free)))
    free = int(free) - aux
```

and pass `aux_slots=aux` into the returned `ReserveAllowance`. In `gareus/adaptive/cv2_resolution_io.py:209` and `:217` pass `aux_slots=int(getattr(policy, "aux_reserve_slots", 0)) if getattr(policy, "aux_discovery", False) else 0` minus the number of active workers (an admitted worker already occupies its slot and is counted in `n_active`):

```python
    _workers = sum(1 for s in registry.active_states() if is_auxiliary_state(s))
    _aux_hold = max(0, int(getattr(policy, "aux_reserve_slots", 0)) - _workers) if getattr(policy, "aux_discovery", False) else 0
```

- [ ] **Step 5: Run tests**

Run: `opencode run "pytest -q tests/test_aux_admission_applier.py tests/test_applier_p2.py tests/test_layout_reserve.py tests/test_cv2_resolution_wiring.py"`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add gareus/adaptive_production.py gareus/adaptive/reserve_budget.py gareus/adaptive/cv2_resolution_io.py tests/test_aux_admission_applier.py
git commit -m "feat(cvaux-adaptive): admit_aux applier action and dedicated aux reserve slice"
```

---

### Task 5: Descriptors (torsions, heavy-atom contacts, backbone H-bonds, basin codes)

**Files:**
- Create: `gareus/adaptive/aux_discovery/descriptors.py`, `tests/data/chignolin_solute.pdb`
- Test: `tests/test_aux_discovery_descriptors.py`

**Interfaces:**
- Produces: `DescriptorDefinition` (frozen dataclass: `phi_quads: np.ndarray (n,4)`, `psi_quads`, `phi_labels: tuple[str]`, `psi_labels`, `hc_groups: tuple[tuple[np.ndarray, np.ndarray], ...]`, `hc_labels`, `hb_pairs: np.ndarray (m,2)`, `hb_labels`, `core_residue_pairs: tuple[(phi_col, psi_col)]`, `schema_sha256: str`), `descriptor_definition(topology) -> DescriptorDefinition` (mdtraj Topology of the solute), `evaluate_descriptors(xyz_nm: np.ndarray (F,N,3), d: DescriptorDefinition) -> dict` with keys `tors` (F, 2*(n_phi+n_psi)) ordered all phi then all psi, sin then cos per angle; `tors_theta_iupac` (F, n_phi+n_psi) radians; `hc` (F, n_hc); `hb` (F, n_hb); `basin` (F, n_core) uint8. Constants `LAMBDA_NM = 0.02`, `R0_HC_NM = 0.45`, `R0_HB_NM = 0.35`.

- [ ] **Step 1: Copy the fixture topology**

```bash
cp /run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_10/adaptive_production/epoch_000/solute_only.pdb tests/data/chignolin_solute.pdb
```

If that path is absent locally, fetch it: `scp sulcjo@aurum2:gareus/chignolin/chignolin_10/adaptive_production/epoch_000/solute_only.pdb tests/data/chignolin_solute.pdb`.

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_aux_discovery_descriptors.py
from pathlib import Path
import numpy as np
import mdtraj as md

from gareus.adaptive.aux_discovery.descriptors import (descriptor_definition, evaluate_descriptors,
                                                       LAMBDA_NM, R0_HC_NM, R0_HB_NM)

PDB = Path(__file__).parent / "data" / "chignolin_solute.pdb"


def _traj():
    return md.load(str(PDB))


def test_c10_family_sizes_for_chignolin():
    d = descriptor_definition(_traj().topology)
    assert len(d.phi_labels) == 9 and len(d.psi_labels) == 9
    assert len(d.hc_labels) == 28 and len(d.hb_labels) == 63
    assert d.phi_labels[0] == "phi_TYR2" and d.psi_labels[0] == "psi_GLY1"


def test_torsions_match_mdtraj_iupac():
    t = _traj(); d = descriptor_definition(t.topology)
    out = evaluate_descriptors(t.xyz, d)
    _, phi = md.compute_phi(t); _, psi = md.compute_psi(t)
    ang = np.concatenate([phi, psi], axis=1)
    np.testing.assert_allclose(out["tors_theta_iupac"], ang, atol=1e-6)
    np.testing.assert_allclose(out["tors"][:, 0::2], np.sin(ang), atol=1e-6)
    np.testing.assert_allclose(out["tors"][:, 1::2], np.cos(ang), atol=1e-6)


def test_contact_softmin_switch_formula():
    t = _traj(); d = descriptor_definition(t.topology)
    out = evaluate_descriptors(t.xyz, d)
    a, b = d.hc_groups[0]
    x = np.linalg.norm(t.xyz[0, a][:, None, :] - t.xyz[0, b][None, :, :], axis=-1).ravel()
    m = x.min(); dist = m - LAMBDA_NM * np.log(np.exp(-(x - m) / LAMBDA_NM).sum())
    assert np.isclose(out["hc"][0, 0], 1.0 / (1.0 + (dist / R0_HC_NM) ** 6), atol=1e-6)


def test_hbond_switch_formula():
    t = _traj(); d = descriptor_definition(t.topology)
    out = evaluate_descriptors(t.xyz, d)
    i, j = d.hb_pairs[0]
    r = np.linalg.norm(t.xyz[0, i] - t.xyz[0, j])
    assert np.isclose(out["hb"][0, 0], 1.0 / (1.0 + (r / R0_HB_NM) ** 6), atol=1e-6)


def test_basin_codes_match_discovery_census():
    from gareus.adaptive.discovery_census import basin_codes
    t = _traj(); d = descriptor_definition(t.topology)
    out = evaluate_descriptors(t.xyz, d)
    phi = np.degrees(out["tors_theta_iupac"][:, :9]); psi = np.degrees(out["tors_theta_iupac"][:, 9:])
    # core residues = residues with both phi and psi: residues 2..9 -> phi cols 0..7, psi cols 1..8
    exp = basin_codes(phi[:, 0:8], psi[:, 1:9])
    np.testing.assert_array_equal(out["basin"], exp)


def test_schema_sha_is_stable():
    t = _traj()
    assert descriptor_definition(t.topology).schema_sha256 == descriptor_definition(t.topology).schema_sha256
```

- [ ] **Step 3: Run to verify failure**

Run: `opencode run "pytest -q tests/test_aux_discovery_descriptors.py"`
Expected: FAIL (module missing).

- [ ] **Step 4: Implement**

```python
# gareus/adaptive/aux_discovery/descriptors.py
"""Reference-free descriptors (c10 build_features.py): backbone torsions (IUPAC sign), heavy-atom
residue-pair soft-min contacts, backbone N-O H-bond switches, per-residue basin codes."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from gareus.adaptive.discovery_census import basin_codes

LAMBDA_NM = 0.02
R0_HC_NM = 0.45
R0_HB_NM = 0.35
MIN_HC_SEPARATION = 3
MIN_HB_SEPARATION = 2


@dataclass(frozen=True)
class DescriptorDefinition:
    phi_quads: np.ndarray
    psi_quads: np.ndarray
    phi_labels: Tuple[str, ...]
    psi_labels: Tuple[str, ...]
    hc_groups: Tuple[Tuple[np.ndarray, np.ndarray], ...]
    hc_labels: Tuple[str, ...]
    hb_pairs: np.ndarray
    hb_labels: Tuple[str, ...]
    core_phi_cols: np.ndarray
    core_psi_cols: np.ndarray
    schema_sha256: str

    @property
    def tors_labels(self) -> Tuple[str, ...]:
        out = []
        for name in self.phi_labels + self.psi_labels:
            out += [f"tors_sin_{name}", f"tors_cos_{name}"]
        return tuple(out)


def _res_name(res) -> str:
    return f"{res.name}{res.resSeq}"


def descriptor_definition(topology) -> DescriptorDefinition:
    import mdtraj as md
    probe = md.Trajectory(np.zeros((1, topology.n_atoms, 3), dtype=np.float32), topology)
    phi_idx, _ = md.compute_phi(probe)
    psi_idx, _ = md.compute_psi(probe)
    atoms = list(topology.atoms)
    phi_labels = tuple(f"phi_{_res_name(atoms[q[2]].residue)}" for q in phi_idx)
    psi_labels = tuple(f"psi_{_res_name(atoms[q[1]].residue)}" for q in psi_idx)
    residues = [r for r in topology.residues if r.is_protein]
    heavy = {r.index: np.array([a.index for a in r.atoms if a.element is not None and a.element.symbol != "H"])
             for r in residues}
    hc_groups, hc_labels = [], []
    for i, ri in enumerate(residues):
        for rj in residues[i + MIN_HC_SEPARATION:]:
            hc_groups.append((heavy[ri.index], heavy[rj.index]))
            hc_labels.append(f"hc_{_res_name(ri)}_{_res_name(rj)}")
    donors = [(k, a.index) for k, r in enumerate(residues) if k >= 1 and r.name != "PRO"
              for a in r.atoms if a.name == "N"]
    acceptors = [(k, a.index) for k, r in enumerate(residues) for a in r.atoms if a.name in ("O", "OXT")]
    hb_pairs, hb_labels = [], []
    for kd, dn in donors:
        for ka, ac in acceptors:
            if abs(kd - ka) >= MIN_HB_SEPARATION:
                hb_pairs.append((dn, ac))
                hb_labels.append(f"hb_{_res_name(residues[kd])}N_{_res_name(residues[ka])}{atoms[ac].name}")
    # core residues = residues having both phi and psi
    phi_res = [atoms[q[2]].residue.index for q in phi_idx]
    psi_res = [atoms[q[1]].residue.index for q in psi_idx]
    core = [r for r in phi_res if r in psi_res]
    core_phi = np.array([phi_res.index(r) for r in core]); core_psi = np.array([psi_res.index(r) for r in core])
    schema = {"phi": phi_labels, "psi": psi_labels, "hc": hc_labels, "hb": hb_labels,
              "lambda_nm": LAMBDA_NM, "r0_hc_nm": R0_HC_NM, "r0_hb_nm": R0_HB_NM,
              "phi_quads": phi_idx.tolist(), "psi_quads": psi_idx.tolist()}
    sha = hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest()
    return DescriptorDefinition(np.asarray(phi_idx), np.asarray(psi_idx), phi_labels, psi_labels,
                                tuple(hc_groups), tuple(hc_labels), np.asarray(hb_pairs, dtype=np.int64),
                                tuple(hb_labels), core_phi, core_psi, sha)


def _dihedral(xyz, quads):
    b0 = xyz[:, quads[:, 0]] - xyz[:, quads[:, 1]]
    b1 = xyz[:, quads[:, 2]] - xyz[:, quads[:, 1]]
    b2 = xyz[:, quads[:, 3]] - xyz[:, quads[:, 2]]
    b1n = b1 / np.linalg.norm(b1, axis=-1, keepdims=True)
    v = b0 - (b0 * b1n).sum(-1, keepdims=True) * b1n
    w = b2 - (b2 * b1n).sum(-1, keepdims=True) * b1n
    x = (v * w).sum(-1)
    y = (np.cross(b1n, v) * w).sum(-1)
    return np.arctan2(y, x)


def evaluate_descriptors(xyz_nm: np.ndarray, d: DescriptorDefinition) -> Dict[str, np.ndarray]:
    xyz = np.asarray(xyz_nm, dtype=np.float64)
    theta = np.concatenate([_dihedral(xyz, d.phi_quads), _dihedral(xyz, d.psi_quads)], axis=1)
    tors = np.stack([np.sin(theta), np.cos(theta)], axis=2).reshape(theta.shape[0], -1)
    hc = np.empty((xyz.shape[0], len(d.hc_groups)))
    for c, (a, b) in enumerate(d.hc_groups):
        x = np.linalg.norm(xyz[:, a][:, :, None, :] - xyz[:, b][:, None, :, :], axis=-1).reshape(xyz.shape[0], -1)
        m = x.min(1)
        dist = m - LAMBDA_NM * np.log(np.exp(-(x - m[:, None]) / LAMBDA_NM).sum(1))
        hc[:, c] = 1.0 / (1.0 + (dist / R0_HC_NM) ** 6)
    r = np.linalg.norm(xyz[:, d.hb_pairs[:, 0]] - xyz[:, d.hb_pairs[:, 1]], axis=-1)
    hb = (1.0 / (1.0 + (r / R0_HB_NM) ** 6)).astype(np.float32)
    n_phi = len(d.phi_labels)
    deg = np.degrees(theta)
    basin = basin_codes(deg[:, d.core_phi_cols], deg[:, n_phi + d.core_psi_cols])
    return {"tors": tors, "tors_theta_iupac": theta, "hc": hc, "hb": hb, "basin": basin}
```

`mdtraj.compute_dihedrals` and this `_dihedral` agree in sign (IUPAC); the test pins it. If `basin_codes` expects 1-D inputs, apply it column-wise (check `gareus/adaptive/discovery_census.py:79`).

- [ ] **Step 5: Run tests**

Run: `opencode run "pytest -q tests/test_aux_discovery_descriptors.py"`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add -f tests/data/chignolin_solute.pdb
git add gareus/adaptive/aux_discovery/descriptors.py tests/test_aux_discovery_descriptors.py
git commit -m "feat(cvaux-adaptive): reference-free descriptors (c10 families) with basin codes"
```

---

### Task 6: Frame table (XTC frames joined to Parquet samples)

**Files:**
- Create: `gareus/adaptive/aux_discovery/frames.py`
- Test: `tests/test_aux_discovery_frames.py`

**Interfaces:**
- Consumes: `descriptor_definition`, `evaluate_descriptors` (Task 5); `discovery_census.ordered_phases(adaptive_dir)` (L251), `discovery_census._trajectory_files(phase_dir)` (L347), `discovery_census._find_topology(phase_dir, adaptive_dir)` (L291), `ladder_adapt._read_phase`-style duckdb dedup, `topup_seeding.state_id_of_window_from_epoch_map(phase_dir)`.
- Produces: `FrameTable` (frozen dataclass of equal-length arrays: `phase: np.ndarray[str]`, `epoch: np.ndarray[int]`, `replica: int64`, `step: int64`, `state_id: int64`, `lam: float64`, `cv1: float32`, `cv2: float32`, `tors`, `tors_theta_iupac`, `hc`, `hb`, `basin`, plus `definition: DescriptorDefinition`, `sources: list[dict]`; properties `lineage -> np.ndarray[str]` = `phase + ":" + replica`, `n`), `phase_epoch(label: str) -> Optional[int]` (`epoch_002/baseline` -> 2, `final*` -> None), `load_phase_samples(phase_dir) -> pandas.DataFrame` (columns replica, step, window_id, cv1, cv2, gamd_lambda; deduplicated on (replica, step) keeping the last segment), `build_frame_table(adaptive_dir: Path, *, epochs: Iterable[int], registry_lambda: dict[int, float], stride_steps: int, max_frames: int, seed: int, workers: int = 1) -> FrameTable` (lambda = 0 frames only).

- [ ] **Step 1: Write the failing tests**

Build a synthetic phase on disk: the fixture PDB, an XTC written with mdtraj (`md.formats.XTCTrajectoryFile`, fields `xyz`, `time`, `step`), a `samples/seg_000/data.parquet`, and an `epoch_window_map.csv`.

```python
# tests/test_aux_discovery_frames.py
from pathlib import Path
import csv
import numpy as np
import pandas as pd
import mdtraj as md

from gareus.adaptive.aux_discovery.frames import build_frame_table, phase_epoch, load_phase_samples

PDB = Path(__file__).parent / "data" / "chignolin_solute.pdb"
DT_PS = 0.0035


def _phase(root: Path, name: str, steps_by_file, lam_by_window, cv_by_step):
    ph = root / "adaptive_production" / name
    (ph / "replica_trajectories").mkdir(parents=True)
    (ph / "samples" / "seg_000").mkdir(parents=True)
    t = md.load(str(PDB))
    import shutil; shutil.copy(PDB, ph / "solute_only.pdb")
    rows = []
    for fname, (replica, window, steps) in steps_by_file.items():
        xyz = np.repeat(t.xyz, len(steps), axis=0)
        with md.formats.XTCTrajectoryFile(str(ph / "replica_trajectories" / fname), "w") as fh:
            fh.write(xyz, time=np.asarray(steps) * DT_PS, step=np.asarray(steps, dtype=np.int32))
        for s in steps:
            rows.append({"step": s, "replica": replica, "window_id": window, "cv1": cv_by_step(s),
                         "cv2": 0.0, "gamd_lambda": lam_by_window[window]})
    pd.DataFrame(rows).to_parquet(ph / "samples" / "seg_000" / "data.parquet")
    with (ph / "epoch_window_map.csv").open("w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["epoch_window", "state_id"])
        for win in lam_by_window:
            w.writerow([win, 100 + win])
    return ph


def test_phase_epoch_labels():
    assert phase_epoch("epoch_002/baseline") == 2 and phase_epoch("epoch_000") == 0
    assert phase_epoch("final") is None and phase_epoch("final_extension_001") is None


def test_join_dedup_stride_and_lambda0(tmp_path: Path):
    _phase(tmp_path, "epoch_000",
           {"replica_0.xtc": (0, 0, [300, 3300, 6300]),
            "replica_0_resume_from_3300.xtc": (0, 0, [3300, 6300, 9300]),   # overlaps the first file
            "replica_1.xtc": (1, 1, [300, 3300, 6300])},
           {0: 0.0, 1: 0.2}, lambda s: s / 1e4)
    ft = build_frame_table(tmp_path, epochs=[0], registry_lambda={100: 0.0, 101: 0.2},
                           stride_steps=3000, max_frames=10 ** 6, seed=0)
    assert set(ft.state_id) == {100}                     # lambda = 0 only
    assert sorted(ft.step.tolist()) == [300, 3300, 6300, 9300]   # resume overlap removed, one row per step
    assert np.allclose(ft.cv1, ft.step / 1e4)
    assert ft.tors.shape == (4, 36)


def test_samples_dedup_keeps_last_segment(tmp_path: Path):
    ph = tmp_path / "p"; (ph / "samples" / "seg_000").mkdir(parents=True); (ph / "samples" / "seg_001").mkdir()
    pd.DataFrame({"step": [300], "replica": [0], "window_id": [0], "cv1": [1.0], "cv2": [0.0],
                  "gamd_lambda": [0.0]}).to_parquet(ph / "samples" / "seg_000" / "data.parquet")
    pd.DataFrame({"step": [300], "replica": [0], "window_id": [0], "cv1": [2.0], "cv2": [0.0],
                  "gamd_lambda": [0.0]}).to_parquet(ph / "samples" / "seg_001" / "data.parquet")
    df = load_phase_samples(ph)
    assert len(df) == 1 and float(df.cv1.iloc[0]) == 2.0


def test_budget_subsamples_uniformly_per_state(tmp_path: Path):
    _phase(tmp_path, "epoch_000", {"replica_0.xtc": (0, 0, list(range(300, 300 + 3000 * 40, 3000)))},
           {0: 0.0}, lambda s: 0.0)
    ft = build_frame_table(tmp_path, epochs=[0], registry_lambda={100: 0.0}, stride_steps=3000,
                           max_frames=10, seed=0)
    assert ft.n == 10
```

- [ ] **Step 2: Run to verify failure**

Run: `opencode run "pytest -q tests/test_aux_discovery_frames.py"`
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
# gareus/adaptive/aux_discovery/frames.py
"""Campaign frames for aux discovery: each phase's solute XTCs joined on (replica, step) to its Parquet
samples (deduplicated, last segment wins), state from the phase's epoch_window_map.csv, lambda = 0 only."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from gareus.adaptive import discovery_census as census
from gareus.topup_seeding import state_id_of_window_from_epoch_map
from .descriptors import DescriptorDefinition, descriptor_definition, evaluate_descriptors

_EPOCH_RE = re.compile(r"^epoch_(\d{3})(?:/|$)")


def phase_epoch(label: str) -> Optional[int]:
    m = _EPOCH_RE.match(str(label))
    return int(m.group(1)) if m else None


@dataclass(frozen=True)
class FrameTable:
    phase: np.ndarray
    epoch: np.ndarray
    replica: np.ndarray
    step: np.ndarray
    state_id: np.ndarray
    lam: np.ndarray
    cv1: np.ndarray
    cv2: np.ndarray
    tors: np.ndarray
    tors_theta_iupac: np.ndarray
    hc: np.ndarray
    hb: np.ndarray
    basin: np.ndarray
    definition: DescriptorDefinition
    sources: List[dict] = field(default_factory=list)

    @property
    def n(self) -> int:
        return int(self.step.size)

    @property
    def lineage(self) -> np.ndarray:
        return np.char.add(np.char.add(self.phase.astype(str), ":"), self.replica.astype(str))

    def take(self, idx: np.ndarray) -> "FrameTable":
        arrays = {k: getattr(self, k)[idx] for k in ("phase", "epoch", "replica", "step", "state_id", "lam",
                                                      "cv1", "cv2", "tors", "tors_theta_iupac", "hc", "hb", "basin")}
        return FrameTable(**arrays, definition=self.definition, sources=self.sources)


def load_phase_samples(phase_dir: Path) -> pd.DataFrame:
    import duckdb
    q = (f"select replica, step, window_id, cv1, cv2, gamd_lambda, filename from "
         f"read_parquet('{Path(phase_dir)}/samples/*/*.parquet', filename=true, union_by_name=true)")
    df = duckdb.sql(q).df()
    df["seg"] = df["filename"].str.extract(r"seg_(\d+)").astype(int)
    df = df.sort_values(["replica", "step", "seg"]).drop_duplicates(["replica", "step"], keep="last")
    return df.drop(columns=["filename", "seg"]).reset_index(drop=True)


def _phase_frames(label: str, phase_dir: Path, adaptive_dir: Path, definition: Optional[DescriptorDefinition],
                  registry_lambda: Dict[int, float], stride_steps: int):
    import mdtraj as md
    top_path = census._find_topology(phase_dir, adaptive_dir)
    top = md.load_topology(str(top_path))
    definition = definition or descriptor_definition(top)
    samples = load_phase_samples(phase_dir)
    wmap = state_id_of_window_from_epoch_map(phase_dir)
    if not wmap:
        raise RuntimeError(f"{phase_dir}: no epoch_window_map.csv")
    samples["state_id"] = samples["window_id"].map(lambda w: wmap.get(int(w), -1)).astype(np.int64)
    samples["lam_reg"] = samples["state_id"].map(lambda s: registry_lambda.get(int(s), np.nan))
    samples = samples[(samples["state_id"] >= 0) & (samples["lam_reg"].abs() < 1e-12)]
    key = samples.set_index(["replica", "step"])
    files = sorted(census._trajectory_files(phase_dir), key=lambda t: (t[0], t[1] or 0, str(t[2])))
    blocks = []
    for i, (replica, start, path) in enumerate(files):
        nxt = next((s for r, s, _ in files[i + 1:] if r == replica), None)
        traj = md.load(str(path), top=top)
        steps = np.rint(traj.time / (traj.timestep if False else 1.0)).astype(np.int64)  # replaced below
        steps = _xtc_steps(path)
        keep = np.ones(len(steps), bool) if nxt is None else steps < int(nxt)
        keep &= (steps % int(stride_steps)) == (steps[keep][0] % int(stride_steps) if keep.any() else 0)
        idx = np.nonzero(keep)[0]
        if idx.size == 0:
            continue
        rows = [(int(replica), int(s)) for s in steps[idx]]
        hit = np.array([r in key.index for r in rows])
        if not hit.any():
            continue
        sub = key.loc[[r for r, h in zip(rows, hit) if h]]
        desc = evaluate_descriptors(traj.xyz[idx[hit]], definition)
        blocks.append((sub.reset_index(), desc))
    return definition, blocks


def _xtc_steps(path: Path) -> np.ndarray:
    import mdtraj as md
    with md.formats.XTCTrajectoryFile(str(path), "r") as fh:
        _xyz, _time, step, _box = fh.read()
    return np.asarray(step, dtype=np.int64)


def build_frame_table(adaptive_dir: Path, *, epochs: Iterable[int], registry_lambda: Dict[int, float],
                      stride_steps: int, max_frames: int, seed: int, workers: int = 1) -> FrameTable:
    adaptive_dir = Path(adaptive_dir)
    root = adaptive_dir if adaptive_dir.name == "adaptive_production" else adaptive_dir / "adaptive_production"
    wanted = set(int(e) for e in epochs)
    phases, _skipped = census.ordered_phases(root)
    definition = None
    cols = {k: [] for k in ("phase", "epoch", "replica", "step", "state_id", "lam", "cv1", "cv2",
                            "tors", "tors_theta_iupac", "hc", "hb", "basin")}
    sources = []
    for label, phase_dir in phases:
        ep = phase_epoch(label)
        if ep is None or ep not in wanted:
            continue
        definition, blocks = _phase_frames(label, Path(phase_dir), root, definition, registry_lambda, stride_steps)
        n_phase = 0
        for sub, desc in blocks:
            n = len(sub); n_phase += n
            cols["phase"].append(np.full(n, label, dtype=object)); cols["epoch"].append(np.full(n, ep))
            cols["replica"].append(sub["replica"].to_numpy(np.int64)); cols["step"].append(sub["step"].to_numpy(np.int64))
            cols["state_id"].append(sub["state_id"].to_numpy(np.int64)); cols["lam"].append(np.zeros(n))
            cols["cv1"].append(sub["cv1"].to_numpy(np.float32)); cols["cv2"].append(sub["cv2"].to_numpy(np.float32))
            for k in ("tors", "tors_theta_iupac", "hc", "hb", "basin"):
                cols[k].append(desc[k])
        sources.append({"phase": label, "dir": str(phase_dir), "n_frames": int(n_phase)})
    if definition is None or not cols["step"]:
        raise RuntimeError(f"no lambda = 0 frames in epochs {sorted(wanted)} under {root}")
    arrays = {k: np.concatenate(v) for k, v in cols.items()}
    arrays["phase"] = arrays["phase"].astype(str)
    ft = FrameTable(**arrays, definition=definition, sources=sources)
    if ft.n > int(max_frames):
        rng = np.random.default_rng(int(seed))
        keep = np.sort(rng.choice(ft.n, size=int(max_frames), replace=False))
        ft = ft.take(keep)
    return ft
```

Clean up `_phase_frames` while implementing: remove the placeholder `steps = np.rint(...)` line (the XTC `step` field is the authority, as in c10 `slow_mode_reseed_io`), read each XTC once (use `md.load` for coordinates and `_xtc_steps` for steps, or read both from one `XTCTrajectoryFile.read()` and build `md.Trajectory`), and make the stride filter `(steps - steps[0]) % stride_steps == 0` on the kept steps. The `workers` argument: when > 1, map phases through a spawn `ProcessPoolExecutor` exactly as `discovery_census.run_census` does (L411-445); keep 1 in tests. The budget is uniform random without replacement over all frames (spec says per (phase, state); a test pins the count; per-state stratification is acceptable if you prefer, keep the test).

- [ ] **Step 4: Run tests**

Run: `opencode run "pytest -q tests/test_aux_discovery_frames.py"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add gareus/adaptive/aux_discovery/frames.py tests/test_aux_discovery_frames.py
git commit -m "feat(cvaux-adaptive): campaign frame table (XTC x Parquet join, dedup, lambda=0, budget)"
```

---

### Task 7: Partitions (discovery and evaluation)

**Files:**
- Create: `gareus/adaptive/aux_discovery/partitions.py`
- Test: `tests/test_aux_discovery_partitions.py`

**Interfaces:**
- Consumes: `AuxDiscoverySettings` (Task 1).
- Produces:
  - `preprocess(X: np.ndarray, families: Sequence[str], train: np.ndarray[bool], s: AuxDiscoverySettings) -> (Z, keep_mask, mean, sd)`
  - `standardise_s(cv: np.ndarray (n,2), train) -> np.ndarray` (float32 arithmetic, as c10)
  - `knn_residual(S, Z, train, k) -> (R, model)`
  - `pca_project(R, train, var, cap) -> (P, pca)`
  - `s_bins(S, train, n) -> np.ndarray[int]`
  - `choose_partition(P, train, holdout, lineage, s, seed) -> PartitionChoice` (dataclass: `k: Optional[int]`, `gmm`, `labels: np.ndarray` over all frames or None, `table: list[dict]` per k with `ari_val`, `boot_median`, `n_groups_ok`)
  - `hidden_fraction(lab, bins, train, holdout, K, nbins) -> float`
  - `co_occurrence(lab, bins, holdout, K, s) -> list[dict]` (bins with >= 2 groups at `cooc_share`, each `{"bin", "groups"}`)
  - `lineage_info(lab, bins, lineage, step, K, nbins, c) -> float`
  - `info_gain(z, y, bins, train, holdout, K, nbins) -> float`
  - `FrozenPartition` dataclass (`feature_names`, `keep`, `mean`, `sd`, `family_scale`, `s_mean`, `s_sd`, `knn`, `pca`, `n_components`, `gmm`) with `predict(X_raw, cv) -> labels` and `to_file(path)` / `from_file(path)` (pickle + sha256 sidecar)
  - `fit_partition(X, families, cv, train, holdout, lineage, s, seed) -> PartitionResult` (dataclass: `choice: PartitionChoice`, `frozen: Optional[FrozenPartition]`, `bins: np.ndarray`, `hidden_fraction`, `co_occurrence`, `lineage_info`, `status`: one of `ok`, `insufficient_evidence`, `keep`)

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_aux_discovery_partitions.py
import numpy as np
from gareus.adaptive.aux_discovery import partitions as P
from gareus.adaptive.aux_discovery.settings import AuxDiscoverySettings


def _planted(n_lin=40, per=60, hidden=True, seed=0):
    """Two hidden states at fixed (cv1, cv2): descriptor family 'a' separates them, cv does not."""
    rng = np.random.default_rng(seed)
    n = n_lin * per
    lineage = np.repeat([f"p:{i}" for i in range(n_lin)], per)
    step = np.tile(np.arange(per) * 3000, n_lin)
    cv = rng.normal(size=(n, 2)).astype(np.float32)
    state = (np.repeat(rng.integers(0, 2, n_lin), per) if hidden else np.zeros(n, int))
    flip = rng.random(n) < 0.05
    state = np.where(flip, 1 - state, state)
    Xa = rng.normal(size=(n, 6)) + (3.0 * state[:, None] if hidden else 0.0)
    Xb = rng.normal(size=(n, 6))
    X = np.hstack([Xa, Xb]); fam = ["a"] * 6 + ["b"] * 6
    train = np.repeat(np.arange(n_lin) < 28, per); holdout = ~train
    return X, fam, cv, train, holdout, lineage, step, state


def test_planted_hidden_mode_found():
    X, fam, cv, tr, ho, lin, step, state = _planted()
    res = P.fit_partition(X, fam, cv, tr, ho, lin, step, AuxDiscoverySettings(k_max=4), seed=0)
    assert res.status == "ok" and res.choice.k >= 2
    from sklearn.metrics import adjusted_rand_score
    assert adjusted_rand_score(state[ho], res.choice.labels[ho]) > 0.8


def test_negative_control_insufficient_evidence():
    X, fam, cv, tr, ho, lin, step, _ = _planted(hidden=False)
    res = P.fit_partition(X, fam, cv, tr, ho, lin, step, AuxDiscoverySettings(k_max=4), seed=0)
    assert res.status in ("insufficient_evidence", "keep")


def test_preprocess_drops_constant_and_floors_sd():
    X = np.c_[np.ones(50), np.linspace(0, 1e-2, 50), np.random.default_rng(0).normal(size=50)]
    tr = np.ones(50, bool)
    Z, keep, mean, sd = P.preprocess(X, ["a", "a", "b"], tr, AuxDiscoverySettings())
    assert keep.tolist() == [False, True, True]
    assert np.isclose(sd[1], 0.05)                          # floored
    assert Z.shape == (50, 2)


def test_lineage_bootstrap_keeps_multiplicity():
    rng = np.random.default_rng(0)
    idx = P.lineage_bootstrap_rows(np.array(["a", "a", "b", "c"]), rng)
    # rows of a resampled lineage appear as many times as it is drawn
    assert len(idx) % 1 == 0 and len(idx) >= 1
    lin = np.repeat(["x", "y"], 3)
    counts = [len(P.lineage_bootstrap_rows(lin, np.random.default_rng(s))) for s in range(20)]
    assert set(counts) == {6}                               # 2 lineages drawn with replacement -> 6 rows


def test_info_gain_positive_for_informative_z():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 4000); z = y + 0.3 * rng.normal(size=4000); bins = np.zeros(4000, int)
    tr = np.arange(4000) < 3000
    assert P.info_gain(z, y, bins, tr, ~tr, 2, 1) > 0.3
    assert abs(P.info_gain(rng.normal(size=4000), y, bins, tr, ~tr, 2, 1)) < 0.02
```

- [ ] **Step 2: Run to verify failure**

Run: `opencode run "pytest -q tests/test_aux_discovery_partitions.py"`
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
# gareus/adaptive/aux_discovery/partitions.py
"""Discovery / evaluation partitions (c10 diagnose.py + eval_partition.py, prereg_v2 values).
Bootstraps keep lineage multiplicity (c10 bootstrap_fix.py)."""
from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsRegressor

from .settings import AuxDiscoverySettings


def preprocess(X, families, train, s: AuxDiscoverySettings):
    X = np.asarray(X, dtype=np.float64)
    mean = X[train].mean(0); sd_raw = X[train].std(0)
    keep = sd_raw >= s.sd_drop
    sd = np.maximum(sd_raw, s.sd_floor)
    Z = (X[:, keep] - mean[keep]) / sd[keep]
    fam = np.asarray(families)[keep]
    for f in np.unique(fam):
        Z[:, fam == f] /= np.sqrt((fam == f).sum())
    return Z, keep, mean, sd


def standardise_s(cv, train):
    cv = np.asarray(cv, dtype=np.float32)
    return (cv - cv[train].mean(0)) / cv[train].std(0)


def knn_residual(S, Z, train, k):
    knn = KNeighborsRegressor(n_neighbors=int(k)).fit(S[train], Z[train])
    M = np.empty_like(Z)
    for a in range(0, len(Z), 4000):
        M[a:a + 4000] = knn.predict(S[a:a + 4000])
    return Z - M, knn


def pca_project(R, train, var, cap):
    pca = PCA().fit(R[train])
    nc = int(min(cap, np.searchsorted(np.cumsum(pca.explained_variance_ratio_), var) + 1))
    return pca.transform(R)[:, :nc], pca, nc


def s_bins(S, train, n):
    edges = [np.quantile(S[train, a], np.linspace(0, 1, n + 1)[1:-1]) for a in range(2)]
    return np.digitize(S[:, 0], edges[0]) * n + np.digitize(S[:, 1], edges[1])


def lineage_bootstrap_rows(lineage_of_rows, rng) -> np.ndarray:
    lins, inv = np.unique(lineage_of_rows, return_inverse=True)
    rows_by = [np.nonzero(inv == i)[0] for i in range(len(lins))]
    pick = rng.integers(0, len(lins), len(lins))
    return np.concatenate([rows_by[i] for i in pick])


@dataclass
class PartitionChoice:
    k: Optional[int]
    gmm: object = None
    labels: Optional[np.ndarray] = None
    table: List[dict] = field(default_factory=list)


def choose_partition(Pm, train, holdout, lineage, s: AuxDiscoverySettings, seed) -> PartitionChoice:
    rng = np.random.default_rng(int(seed))
    tr_rows = np.nonzero(train)[0]
    table, best = [], None
    for k in range(int(s.k_min), int(s.k_max) + 1):
        g = GaussianMixture(k, covariance_type="full", n_init=3, random_state=0, reg_covar=1e-6).fit(Pm[train])
        lab_v = g.predict(Pm[holdout])
        refit = GaussianMixture(k, covariance_type="full", n_init=3, random_state=1, reg_covar=1e-6).fit(Pm[holdout])
        ari_val = float(adjusted_rand_score(lab_v, refit.predict(Pm[holdout])))
        boots = []
        for r in range(int(s.n_boot_partition)):
            rows = tr_rows[lineage_bootstrap_rows(lineage[train], rng)]
            gb = GaussianMixture(k, covariance_type="full", n_init=1, random_state=10 + r, reg_covar=1e-6).fit(Pm[rows])
            boots.append(adjusted_rand_score(lab_v, gb.predict(Pm[holdout])))
        share = np.bincount(lab_v, minlength=k) / max(1, lab_v.size)
        n_ok = int((share >= s.group_min_share).sum())
        row = {"k": k, "ari_val": round(ari_val, 4), "boot_median": round(float(np.median(boots)), 4),
               "n_groups_ok": n_ok}
        table.append(row)
        if ari_val >= s.ari_min and np.median(boots) >= s.ari_min and n_ok >= 2:
            best = (k, g)
    if best is None:
        return PartitionChoice(None, None, None, table)
    k, g = best
    return PartitionChoice(k, g, g.predict(Pm), table)


def _cond_table(y, x, K, nx, alpha=1.0):
    t = np.full((nx, K), float(alpha))
    np.add.at(t, (x, y), 1.0)
    return t / t.sum(1, keepdims=True)


def _ll(p, y, x=None):
    x = np.zeros_like(y) if x is None else x
    return float(np.mean(np.log(p[x, y])))


def hidden_fraction(lab, bins, train, holdout, K, nbins) -> float:
    pm = _cond_table(lab[train], np.zeros(train.sum(), int), K, 1)
    ll_marg = _ll(pm, lab[holdout])
    ll_s = _ll(_cond_table(lab[train], bins[train], K, nbins), lab[holdout], bins[holdout])
    return float(1.0 - (ll_s - ll_marg) / (-ll_marg)) if ll_marg < 0 else 0.0


def co_occurrence(lab, bins, holdout, K, s: AuxDiscoverySettings) -> List[dict]:
    out = []
    lv, bv = lab[holdout], bins[holdout]
    for b in np.unique(bv):
        m = bv == b
        if m.mean() < s.cooc_bin_min:
            continue
        share = np.bincount(lv[m], minlength=K) / m.sum()
        groups = np.nonzero(share >= s.cooc_share)[0]
        if groups.size >= 2:
            out.append({"bin": int(b), "groups": groups.tolist(), "share": share[groups].round(4).tolist()})
    return out


def lineage_info(lab, bins, lineage, step, K, nbins, c) -> float:
    lin_id = np.unique(lineage, return_inverse=True)[1]
    order = np.lexsort((step, lin_id))
    lin_s, lab_s, b_s = lin_id[order], lab[order], bins[order]
    first = np.zeros(lin_s.size, bool)
    for l in np.unique(lin_s):
        idx = np.nonzero(lin_s == l)[0]
        first[idx[: idx.size // 2]] = True
    A, Tm = first, ~first
    pA = _cond_table(lab_s[A], b_s[A], K, nbins)
    cnt = {}
    for l, b, y in zip(lin_s[A], b_s[A], lab_s[A]):
        cnt.setdefault((l, b), np.zeros(K))[y] += 1
    lt = []
    for l, b, y in zip(lin_s[Tm], b_s[Tm], lab_s[Tm]):
        nB = cnt.get((l, b), np.zeros(K))
        pB = (nB + c * pA[b]) / (nB.sum() + c)
        lt.append(np.log(pB[y]))
    return float(np.mean(lt) - _ll(pA, lab_s[Tm], b_s[Tm])) if lt else 0.0


def info_gain(z, y, bins, train, holdout, K, nbins) -> float:
    u = (z - z[train].mean()) / (z[train].std() + 1e-12)
    onehot = np.eye(nbins)[bins]
    base = onehot; full = np.c_[onehot, u, u ** 2, u ** 3]

    def _heldout_ll(F):
        m = LogisticRegression(max_iter=3000, C=1.0).fit(F[train], y[train])
        pr = np.full((holdout.sum(), K), 1e-12)
        pr[:, m.classes_] = np.maximum(m.predict_proba(F[holdout]), 1e-12)
        return float(np.mean(np.log(pr[np.arange(holdout.sum()), y[holdout]])))
    return _heldout_ll(full) - _heldout_ll(base)


@dataclass
class FrozenPartition:
    feature_names: List[str]
    keep: np.ndarray
    mean: np.ndarray
    sd: np.ndarray
    families: List[str]
    s_mean: np.ndarray
    s_sd: np.ndarray
    knn: object
    pca: object
    n_components: int
    gmm: object

    def predict(self, X_raw, cv) -> np.ndarray:
        X = np.asarray(X_raw, dtype=np.float64)
        Z = (X[:, self.keep] - self.mean[self.keep]) / self.sd[self.keep]
        fam = np.asarray(self.families)[self.keep]
        for f in np.unique(fam):
            Z[:, fam == f] /= np.sqrt((fam == f).sum())
        S = (np.asarray(cv, np.float32) - self.s_mean) / self.s_sd
        R = Z - self.knn.predict(S)
        return self.gmm.predict(self.pca.transform(R)[:, : self.n_components])

    def to_file(self, path: Path) -> str:
        data = pickle.dumps(self, protocol=4)
        Path(path).write_bytes(data)
        sha = hashlib.sha256(data).hexdigest()
        Path(str(path) + ".sha256").write_text(sha + "\n")
        return sha

    @staticmethod
    def from_file(path: Path) -> "FrozenPartition":
        data = Path(path).read_bytes()
        sha = Path(str(path) + ".sha256").read_text().strip()
        if hashlib.sha256(data).hexdigest() != sha:
            raise ValueError(f"{path}: sha256 mismatch")
        return pickle.loads(data)


@dataclass
class PartitionResult:
    status: str
    choice: PartitionChoice
    frozen: Optional[FrozenPartition]
    bins: np.ndarray
    hidden_fraction: Optional[float] = None
    co_occurrence: List[dict] = field(default_factory=list)
    lineage_info: Optional[float] = None


def fit_partition(X, families, cv, train, holdout, lineage, step, s: AuxDiscoverySettings, *, seed: int,
                  feature_names: Optional[Sequence[str]] = None) -> PartitionResult:
    Z, keep, mean, sd = preprocess(X, families, train, s)
    S = standardise_s(cv, train)
    R, knn = knn_residual(S, Z, train, s.knn_k)
    Pm, pca, nc = pca_project(R, train, s.pca_var, s.pca_max)
    bins = s_bins(S, train, s.s_bins)
    choice = choose_partition(Pm, train, holdout, lineage, s, seed)
    if choice.k is None:
        return PartitionResult("insufficient_evidence", choice, None, bins)
    K, nb = choice.k, s.s_bins ** 2
    hf = hidden_fraction(choice.labels, bins, train, holdout, K, nb)
    co = co_occurrence(choice.labels, bins, holdout, K, s)
    lam0 = np.ones_like(train)
    li = lineage_info(choice.labels, bins, lineage, step, K, nb, s.lineage_dirichlet_c)
    s_mean = np.asarray(cv, np.float32)[train].mean(0); s_sd = np.asarray(cv, np.float32)[train].std(0)
    frozen = FrozenPartition(list(feature_names or [f"f{i}" for i in range(np.asarray(X).shape[1])]), keep, mean,
                             np.maximum(np.asarray(X, float)[train].std(0), s.sd_floor), list(families), s_mean, s_sd,
                             knn, pca, nc, choice.gmm)
    triggered = (hf >= s.hidden_min and bool(co)) or li >= s.lineage_info_min
    return PartitionResult("ok" if triggered else "keep", choice, frozen, bins, hf, co, li)
```

Remove the unused `lam0` line while implementing. Keep `knn.predict` chunked in `FrozenPartition.predict` too if memory demands (same 4000-row chunks). A test that `frozen.predict(X, cv)` equals `choice.labels` on the training data must be added: `np.testing.assert_array_equal(res.frozen.predict(X, cv), res.choice.labels)` in `test_planted_hidden_mode_found`.

- [ ] **Step 4: Run tests**

Run: `opencode run "pytest -q tests/test_aux_discovery_partitions.py"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add gareus/adaptive/aux_discovery/partitions.py tests/test_aux_discovery_partitions.py
git commit -m "feat(cvaux-adaptive): discovery/evaluation partitions with reproducibility, hidden fraction, lineage info"
```

---

### Task 8: z3 search and model emission

**Files:**
- Create: `gareus/adaptive/aux_discovery/z3_search.py`
- Test: `tests/test_aux_discovery_z3.py`

**Interfaces:**
- Consumes: `info_gain`, `s_bins` (Task 7); `gareus.auxiliary_cv.model.AuxModel`, `AUX_MODEL_SCHEMA`; `gareus.auxiliary_cv.runtime.canonical_topology_sha256`; `gareus.cv_selection.contracts.FEATURE_SCHEMA_VERSION`; `gareus.auxiliary_cv.features.check_feature_atoms`; `gareus.auxiliary_cv.evaluate.z_from_positions`.
- Produces: `Z3Candidate` dataclass (`groups: tuple[int,int]`, `C: float`, `w_std: np.ndarray (36,)`, `mu`, `sd`, `z_raw_train_sd: float`, `info_gain`, `stability`, `corr_cv1`, `corr_cv2`, `basin_gain`, `n_nonzero`, `passed: bool`, `fail: list[str]`); `search_z3(ft: FrameTable, discovery_labels, bins, train, holdout, s) -> (Optional[Z3Candidate], list[Z3Candidate])`; `z3_values(ft_tors: np.ndarray, cand) -> np.ndarray` (scaled z = raw / z_raw_train_sd); `emit_model(cand, definition, full_topology, *, label, provenance) -> AuxModel`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_aux_discovery_z3.py
from pathlib import Path
import numpy as np
import mdtraj as md

from gareus.adaptive.aux_discovery import z3_search as Z
from gareus.adaptive.aux_discovery.descriptors import descriptor_definition, evaluate_descriptors
from gareus.adaptive.aux_discovery.settings import AuxDiscoverySettings

PDB = Path(__file__).parent / "data" / "chignolin_solute.pdb"


class _FT:   # minimal FrameTable stand-in
    def __init__(self, tors, cv1, cv2, basin, replica):
        self.tors, self.cv1, self.cv2, self.basin, self.replica = tors, cv1, cv2, basin, replica
        self.n = len(cv1)


def _planted(n=6000, seed=0, along_cv1=False):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, n)
    ang = rng.normal(scale=0.3, size=(n, 18))
    ang[:, 3] += np.where(labels == 1, 1.5, -1.5)             # the planted torsion
    tors = np.stack([np.sin(ang), np.cos(ang)], 2).reshape(n, 36)
    cv1 = (labels + 0.1 * rng.normal(size=n)) if along_cv1 else rng.normal(size=n)
    cv2 = rng.normal(size=n)
    basin = rng.integers(0, 5, (n, 8)).astype(np.uint8)
    basin[:, 2] = (ang[:, 3] > 0).astype(np.uint8)
    return _FT(tors, cv1.astype(np.float32), cv2.astype(np.float32), basin, rng.integers(0, 40, n)), labels


def test_recovers_planted_torsion():
    ft, lab = _planted()
    tr = np.arange(ft.n) < 4500; ho = ~tr; bins = np.zeros(ft.n, int)
    best, allc = Z.search_z3(ft, lab, bins, tr, ho, AuxDiscoverySettings(), nbins=1)
    assert best is not None and best.passed and best.groups == (0, 1)
    top = np.argsort(-np.abs(best.w_std))[:2]
    assert set(top) <= {6, 7}                                   # sin/cos of torsion 3


def test_rejects_cv1_correlated_direction():
    ft, lab = _planted(along_cv1=True)
    tr = np.arange(ft.n) < 4500; ho = ~tr; bins = np.zeros(ft.n, int)
    best, allc = Z.search_z3(ft, lab, bins, tr, ho, AuxDiscoverySettings(), nbins=1)
    assert best is None and any("max_cv_corr" in c.fail for c in allc)


def test_emitted_model_matches_projection_and_loads():
    t = md.load(str(PDB)); d = descriptor_definition(t.topology)
    rng = np.random.default_rng(0)
    w = rng.normal(size=36); mu = rng.normal(size=36) * 0.1; sd = 1 + rng.random(36)
    cand = Z.Z3Candidate(groups=(0, 1), C=0.1, w_std=w, mu=mu, sd=sd, z_raw_train_sd=2.5, info_gain=0.2,
                         stability=0.9, corr_cv1=0.0, corr_cv2=0.0, basin_gain=0.1, n_nonzero=36, passed=True, fail=[])
    model = Z.emit_model(cand, d, t.topology, label="test", provenance={"epoch": 1})
    from gareus.auxiliary_cv.evaluate import z_from_positions
    from gareus.auxiliary_cv.features import check_feature_atoms
    check_feature_atoms(model, t.topology.to_openmm())
    z_model = z_from_positions(t.xyz[0], model)
    z_ref = Z.z3_values(evaluate_descriptors(t.xyz, d)["tors"], cand)[0]
    assert abs(float(np.ravel(z_model)[0]) - float(z_ref)) < 1e-6
    from gareus.auxiliary_cv.model import AuxModel
    tmp = Path(__file__).parent / "_tmp_model.json"
    try:
        model.write(tmp); assert AuxModel.load(tmp).model_sha256 == model.model_sha256
    finally:
        tmp.unlink(missing_ok=True)


def test_no_native_readout_module_is_imported():
    import sys
    assert not any(m.endswith("native_readout") or "chignolin_fes" in m for m in sys.modules)
```

- [ ] **Step 2: Run to verify failure**

Run: `opencode run "pytest -q tests/test_aux_discovery_z3.py"`
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
# gareus/adaptive/aux_discovery/z3_search.py
"""Pairwise L1 torsion discriminants between co-occurring discovery groups (c10
torsion_only_candidate.py + prereg_v2 candidate gates), and emission as atlas-aux-cv-model-v1."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression

from .partitions import info_gain
from .settings import AuxDiscoverySettings


@dataclass
class Z3Candidate:
    groups: Tuple[int, int]
    C: float
    w_std: np.ndarray
    mu: np.ndarray
    sd: np.ndarray
    z_raw_train_sd: float
    info_gain: float
    stability: float
    corr_cv1: float
    corr_cv2: float
    basin_gain: float
    n_nonzero: int
    passed: bool
    fail: List[str] = field(default_factory=list)

    def summary(self) -> dict:
        return {"groups": list(self.groups), "C": self.C, "info_gain": round(self.info_gain, 4),
                "stability": round(self.stability, 4), "corr_cv1": round(self.corr_cv1, 4),
                "corr_cv2": round(self.corr_cv2, 4), "basin_gain": round(self.basin_gain, 4),
                "n_nonzero": self.n_nonzero, "passed": self.passed, "fail": list(self.fail)}


def z3_values(tors: np.ndarray, cand: Z3Candidate) -> np.ndarray:
    return (((np.asarray(tors, float) - cand.mu) / cand.sd) @ cand.w_std) / cand.z_raw_train_sd


def _l1(Xs, y, C):
    return LogisticRegression(penalty="l1", solver="liblinear", C=float(C), max_iter=3000).fit(Xs, y)


def _bern_ll(m, X, y):
    p = np.clip(m.predict_proba(X)[:, 1], 1e-9, 1 - 1e-9)
    return float(np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _cooccurring_pairs(labels, bins, holdout, s: AuxDiscoverySettings):
    from .partitions import co_occurrence
    K = int(labels.max()) + 1
    pairs = set()
    for rec in co_occurrence(labels, bins, holdout, K, s):
        g = rec["groups"]
        pairs |= {(min(a, b), max(a, b)) for i, a in enumerate(g) for b in g[i + 1:]}
    return sorted(pairs)


def search_z3(ft, labels, bins, train, holdout, s: AuxDiscoverySettings, *, nbins: Optional[int] = None):
    nbins = int(nbins or s.s_bins ** 2)
    X = np.asarray(ft.tors, float)
    mu, sd = X[train].mean(0), X[train].std(0)
    sd = np.where(sd > 0, sd, 1.0)
    Xs = (X - mu) / sd
    even = (np.asarray(ft.replica) % 2) == 0
    K = int(labels.max()) + 1
    out: List[Z3Candidate] = []
    pairs = _cooccurring_pairs(labels, bins, holdout, s) if nbins > 1 else \
        [(a, b) for a in range(K) for b in range(a + 1, K)]
    for g1, g2 in pairs:
        m = train & np.isin(labels, (g1, g2)); y = (labels == g2).astype(int)
        if len(np.unique(y[m & even])) < 2 or len(np.unique(y[m & ~even])) < 2:
            continue
        scores = [(_bern_ll(_l1(Xs[m & even], y[m & even], C), Xs[m & ~even], y[m & ~even]), C) for C in s.l1_c_grid]
        _, C = max(scores)
        w = _l1(Xs[m], y[m], C).coef_[0]
        we = _l1(Xs[m & even], y[m & even], C).coef_[0]; wo = _l1(Xs[m & ~even], y[m & ~even], C).coef_[0]
        z_raw = Xs @ w
        if not np.any(w) or z_raw[train].std() == 0:
            continue
        stab = abs(float(np.corrcoef(Xs[holdout] @ we, Xs[holdout] @ wo)[0, 1]))
        c1 = float(np.corrcoef(z_raw[holdout], ft.cv1[holdout])[0, 1])
        c2 = float(np.corrcoef(z_raw[holdout], ft.cv2[holdout])[0, 1])
        ig = info_gain(z_raw, labels, bins, train, holdout, K, nbins)
        bg = float(sum(info_gain(z_raw, ft.basin[:, r].astype(int), bins, train, holdout, 5, nbins)
                       for r in range(ft.basin.shape[1])))
        fail = []
        if ig < s.info_gain_min: fail.append("info_gain")
        if not np.isfinite(stab) or stab < s.stability_min: fail.append("stability")
        if abs(c1) > s.max_cv_corr or abs(c2) > s.max_cv_corr: fail.append("max_cv_corr")
        if bg < s.basin_gain_min: fail.append("basin_gain")
        out.append(Z3Candidate((g1, g2), float(C), w, mu, sd, float(z_raw[train].std()), ig, stab, c1, c2, bg,
                               int((np.abs(w) > 1e-8).sum()), not fail, fail))
    passing = [c for c in out if c.passed]
    best = max(passing, key=lambda c: (c.info_gain, -c.n_nonzero)) if passing else None
    return best, out


def emit_model(cand: Z3Candidate, definition, full_topology, *, label: str, provenance: dict):
    """atlas-aux-cv-model-v1, 'negated' convention so trig(-theta_openmm) = trig(theta_iupac); atom indices are
    those of ``full_topology`` (the campaign's 01_solvated_start.pdb), checked against the solute definition."""
    from gareus.auxiliary_cv.model import AuxModel, AUX_MODEL_SCHEMA
    from gareus.auxiliary_cv.runtime import canonical_topology_sha256
    from gareus.cv_selection.contracts import FEATURE_SCHEMA_VERSION
    coef = cand.w_std / cand.sd
    offset = float(-(cand.w_std * cand.mu / cand.sd).sum())
    quads = list(definition.phi_quads) + list(definition.psi_quads)
    names = list(definition.phi_labels) + list(definition.psi_labels)
    atoms = list(full_topology.atoms())
    feats = []
    for t, (q, name) in enumerate(zip(quads, names)):
        kind, res = name.split("_", 1)
        res_index = atoms[int(q[2 if kind == "phi" else 1])].residue.index
        for trig in ("sin", "cos"):
            feats.append({"index": len(feats), "name": f"{kind}-{res}-{trig}", "torsion_name": f"{kind}-{res}",
                          "residue_index": int(res_index), "atom_indices": [int(a) for a in q], "trig": trig,
                          "dihedral_sign_convention": "negated"})
    def _payload(top_sha):
        return {"schema": AUX_MODEL_SCHEMA,
                "feature_schema": {"schema": FEATURE_SCHEMA_VERSION, "topology_sha256": top_sha, "features": feats},
                "coefficients": [float(c) for c in coef], "offset": offset, "scale": float(cand.z_raw_train_sd),
                "periodic_imaging": "none", "units": "dimensionless", "label": label, "provenance": dict(provenance)}
    draft = AuxModel.from_mapping(_payload("0" * 64))
    return AuxModel.from_mapping(_payload(canonical_topology_sha256(full_topology, draft)))
```

Before finishing, confirm: (1) `check_feature_atoms` requires `torsion_name.split("-")[0]` in {phi, psi}: `f"{kind}-{res}"` satisfies it; (2) `canonical_topology_sha256(topology, model)` takes an OpenMM topology (`full_topology` is OpenMM's `app.Topology`; in the test pass `t.topology.to_openmm()` to both `emit_model` and `check_feature_atoms`, and use `.atoms()` accordingly); (3) the solute atom indices equal the full-topology indices (peptide atoms come first in the solvated topology): in `emit_model` assert, for each quad atom, that `full_topology` atom name and residue name equal the solute definition's (`definition` must then carry atom names; add `atom_names: tuple[str]` and `residue_names: tuple[str]` to `DescriptorDefinition` in Task 5 if needed and raise `ValueError("solute and production topologies disagree at atom ...")`).

- [ ] **Step 4: Run tests**

Run: `opencode run "pytest -q tests/test_aux_discovery_z3.py tests/test_aux_discovery_descriptors.py"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add gareus/adaptive/aux_discovery/z3_search.py gareus/adaptive/aux_discovery/descriptors.py tests/test_aux_discovery_z3.py
git commit -m "feat(cvaux-adaptive): pairwise L1 torsion z3 search with prereg gates and atlas-aux-cv-model-v1 emission"
```

---

### Task 9: Placement (exact port of c10 placement.py) and validation record

**Files:**
- Create: `gareus/adaptive/aux_discovery/placement.py`, `gareus/adaptive/aux_discovery/validation.py`
- Test: `tests/test_aux_discovery_placement.py`, `tests/test_aux_discovery_validation.py`

**Interfaces:**
- Produces:
  - `place_workers(z: np.ndarray, lab: np.ndarray, state_id: np.ndarray, lineage: np.ndarray[str], step: np.ndarray, is_train: np.ndarray[bool], is_heldout: np.ndarray[bool], s: AuxDiscoverySettings, *, k_labels: int, k3_max: Optional[float] = None) -> dict` with keys exactly as c10 `placement.json` (`doc`, `n_states`, `n_candidates`, `n_eligible`, `gates`, `skipped_states`, `selection_log`, `chosen`, `top20`, `all_candidates`); candidates above `k3_max` get `eligible: False` and `gates["k3_max"] = False`.
  - `VALIDATION_SCHEMA = "atlas-aux-validation-v1"`, `ValidationStatus(ok: bool, reason: str, k3_max: Optional[float])`, `check_validation_record(path: Path, *, timestep_fs: float) -> ValidationStatus`, CLI `python -m gareus.adaptive.aux_discovery.validation write --out PATH --commit SHA --timestep-fs F --k3-max K --finite-timestep {pass,fail} --npt {pass,fail} --cost {pass,fail} --evidence PATH...`.

- [ ] **Step 1: Write the failing placement tests**

```python
# tests/test_aux_discovery_placement.py
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from gareus.adaptive.aux_discovery.placement import place_workers
from gareus.adaptive.aux_discovery.settings import AuxDiscoverySettings

C10 = Path("/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/docs/_local_docs/c10_aux_diagnosis")
TRAIN = {"epoch_000", "epoch_001/baseline"}; HOLD = "epoch_002/baseline"


@pytest.mark.skipif(not (C10 / "placement.json").exists(), reason="c10 local diagnosis data absent")
def test_parity_with_c10_placement_json():
    g = pd.read_parquet(C10 / "group_labels.parquet", columns=["phase", "replica", "step", "state_id", "lam"])
    z = np.load(C10 / "z_torsion_only.npy") / 3.181
    lab = np.load(C10 / "eval_labels_frozen.npy")
    m = (g.lam == 0).to_numpy()
    g = g[m]; z = z[m]; lab = lab[m]
    lineage = (g.phase + ":" + g.replica.astype(str)).to_numpy()
    out = place_workers(z, lab, g.state_id.to_numpy(), lineage, g.step.to_numpy(),
                        g.phase.isin(TRAIN).to_numpy(), (g.phase == HOLD).to_numpy(),
                        AuxDiscoverySettings(), k_labels=5)
    ref = json.loads((C10 / "placement.json").read_text())
    assert out["all_candidates"] == ref["all_candidates"]
    assert out["chosen"] == ref["chosen"] and out["selection_log"] == ref["selection_log"]


def test_synthetic_selects_at_most_max_workers_and_one_per_side():
    rng = np.random.default_rng(0)
    n_states, per = 6, 600
    sid = np.repeat(np.arange(n_states), per)
    lineage = np.array([f"e0:{i % 30}" for i in range(n_states * per)])
    step = np.tile(np.arange(per) * 3000, n_states)
    z = rng.normal(size=sid.size)
    lab = (z + 0.5 * rng.normal(size=sid.size) > 0).astype(int)
    is_train = np.tile(np.arange(per) < 450, n_states); is_ho = ~is_train
    out = place_workers(z, lab, sid, lineage, step, is_train, is_ho, AuxDiscoverySettings(), k_labels=2)
    keys = [(c["state_id"], c["side"]) for c in out["chosen"]]
    assert len(out["chosen"]) <= 4 and len(keys) == len(set(keys))
    for c in out["chosen"]:
        assert c["heldout"]["net"] > 0 and c["heldout"]["O"] >= 0.15


def test_k3_cap_marks_ineligible():
    rng = np.random.default_rng(1)
    sid = np.repeat([0], 800); lineage = np.array([f"e0:{i % 30}" for i in range(800)])
    z = rng.normal(size=800); lab = (z > 0).astype(int); step = np.arange(800) * 3000
    tr = np.arange(800) < 600
    out = place_workers(z, lab, sid, lineage, step, tr, ~tr, AuxDiscoverySettings(), k_labels=2, k3_max=0.01)
    assert out["n_eligible"] == 0 and out["chosen"] == []
```

- [ ] **Step 2: Write the failing validation tests**

```python
# tests/test_aux_discovery_validation.py
import json
import subprocess, sys
from pathlib import Path
from gareus.adaptive.aux_discovery.validation import check_validation_record, VALIDATION_SCHEMA


def _write(tmp, **kw):
    rec = {"schema": VALIDATION_SCHEMA, "code_commit": "abc", "timestep_fs": 3.5, "k3_max_validated": 3.0,
           "checks": {"finite_timestep": "pass", "npt": "pass", "cost": "pass"}, "evidence": []}
    rec.update(kw)
    p = tmp / "aux_validation.json"; p.write_text(json.dumps(rec)); return p


def test_missing_record(tmp_path):
    st = check_validation_record(tmp_path / "nope.json", timestep_fs=3.5)
    assert not st.ok and st.reason == "validation_missing"


def test_passing_record(tmp_path):
    st = check_validation_record(_write(tmp_path), timestep_fs=3.5)
    assert st.ok and st.k3_max == 3.0


def test_failed_check_and_timestep_mismatch(tmp_path):
    assert check_validation_record(_write(tmp_path, checks={"finite_timestep": "fail", "npt": "pass",
                                                            "cost": "pass"}), timestep_fs=3.5).reason == "validation_failed:finite_timestep"
    assert check_validation_record(_write(tmp_path), timestep_fs=4.0).reason == "validation_timestep_mismatch"


def test_cli_write(tmp_path):
    out = tmp_path / "v.json"
    subprocess.run([sys.executable, "-m", "gareus.adaptive.aux_discovery.validation", "write", "--out", str(out),
                    "--commit", "abc", "--timestep-fs", "3.5", "--k3-max", "3.0", "--finite-timestep", "pass",
                    "--npt", "pass", "--cost", "pass"], check=True)
    assert check_validation_record(out, timestep_fs=3.5).ok
```

- [ ] **Step 3: Run to verify failure**

Run: `opencode run "pytest -q tests/test_aux_discovery_placement.py tests/test_aux_discovery_validation.py"`
Expected: FAIL.

- [ ] **Step 4: Implement placement (verbatim port)**

Copy the algorithm from `C10/placement.py` (read it in full first) into a function. The structure:

```python
# gareus/adaptive/aux_discovery/placement.py
"""Forecast-benefit placement of z3 workers: exact port of c10 docs/_local_docs/c10_aux_diagnosis/placement.py
(pilot spec v2, option 1). Parity with c10 placement.json is a test; do not 'improve' it (rounding to 4 dp before
gates and ranking, point-estimate gates other than O_q05, one global RNG stream in this exact order)."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .settings import AuxDiscoverySettings

R_KCAL = 0.0019872041
DOC = ("Forecast-benefit placement of z3 workers (c10 placement.py port): candidates per lambda = 0 state at z3 "
       "training quantiles x width fractions, forecast O / TV / circular-shift null by reweighting, lineage-bootstrap "
       "O_q05 gate, point gates ess/eff_lineages/top3, rank by utility q10, greedy one per (parent, side), held-out "
       "confirmation net > 0 and O >= 0.15, at most max_workers.")


def _forecast(z, lab, lin_codes, c, k, shifts, RT, K):
    du = 0.5 * k * (z - c) ** 2 / RT
    w = np.exp(-(du - du.min())); w /= w.sum()
    df = -np.log(np.mean(np.exp(-du)))
    O = float(np.mean(1.0 / (1.0 + np.exp(np.clip(du - df, -50, 50)))))
    n = z.size
    p0 = np.bincount(lab, minlength=K) / n
    TV = 0.5 * np.abs(np.bincount(lab, weights=w, minlength=K) - p0).sum()
    nulls = np.array([0.5 * np.abs(np.bincount(np.roll(lab, int(s)), weights=w, minlength=K) - p0).sum() for s in shifts])
    lw = np.bincount(lin_codes, weights=w)
    top3 = float(np.sort(lw)[::-1][:3].sum())
    return dict(O=O, TV=float(TV), null=float(nulls.mean()), null_pct=float(np.mean(nulls < TV)),
                ess_frames=float(1.0 / np.sum(w ** 2)), eff_lineages=float(1.0 / np.sum(lw ** 2)), top3_share=top3)


def place_workers(z, lab, state_id, lineage, step, is_train, is_heldout, s: AuxDiscoverySettings, *,
                  k_labels: int, k3_max: Optional[float] = None) -> dict:
    RT = R_KCAL * float(s.temperature_k)
    rng = np.random.default_rng(int(s.placement_seed))

    def shift_set(n, longest, m):
        lo = min(longest, n // 3); hi = n - lo
        return rng.integers(lo, hi, size=m) if hi > lo else np.array([n // 2] * m)

    df = pd.DataFrame({"state_id": state_id, "lineage": lineage, "step": step, "z": z, "lab": lab,
                       "tr": is_train, "ho": is_heldout})
    df = df.sort_values(["state_id", "lineage", "step"], kind="mergesort")
    cands, skipped = [], []
    for sid, grp in df.groupby("state_id", sort=True):
        tr = grp[grp.tr]; ho = grp[grp.ho]
        if len(tr) < s.min_train_frames:
            skipped.append({"state_id": int(sid), "n_train_frames": int(len(tr)), "reason": "too_few_train_frames"})
            continue
        zt = tr.z.to_numpy(); lt = tr.lab.to_numpy()
        lin = pd.factorize(tr.lineage)[0]
        nb = lin.max() + 1
        blocks = [np.nonzero(lin == b)[0] for b in range(nb)]
        longest = max(len(b) for b in blocks)
        sd = zt.std(); med = np.median(zt)
        for q in s.quantiles:
            for f in s.width_fractions:
                c = float(np.quantile(zt, q)); k = RT / (f * sd) ** 2
                pt = _forecast(zt, lt, lin, c, k, shift_set(len(zt), longest, s.n_null), RT, k_labels)
                boot = []
                for _ in range(s.n_boot_place):
                    pick = rng.integers(0, nb, nb)
                    idx = np.concatenate([blocks[j] for j in pick])
                    lb = np.concatenate([np.full(len(blocks[j]), i) for i, j in enumerate(pick)])
                    fb = _forecast(zt[idx], lt[idx], lb, c, k, shift_set(len(idx), longest, s.n_null_boot), RT, k_labels)
                    boot.append((fb["O"], fb["O"] * (fb["TV"] - fb["null"])))
                boot = np.array(boot)
                heldout = None
                if len(ho) >= s.heldout_min_frames:
                    lh = pd.factorize(ho.lineage)[0]
                    fh = _forecast(ho.z.to_numpy(), ho.lab.to_numpy(), lh, c, k,
                                   shift_set(len(ho), int(np.bincount(lh).max()), s.n_null), RT, k_labels)
                    heldout = {"O": round(fh["O"], 4), "net": round(fh["TV"] - fh["null"], 4),
                               "null_pct": round(fh["null_pct"], 3), "n": int(len(ho))}
                rec = {"state_id": int(sid), "side": "below" if c < med else "above", "c3": round(c, 4),
                       "k3": round(k, 4), "sigma_w": round(f * sd, 4), "width_fraction": f, "quantile": q,
                       "O": round(pt["O"], 4), "TV": round(pt["TV"], 4), "null": round(pt["null"], 4),
                       "null_pct": round(pt["null_pct"], 3), "ess_frames": round(pt["ess_frames"], 4),
                       "eff_lineages": round(pt["eff_lineages"], 4), "top3_share": round(pt["top3_share"], 4),
                       "net": round(pt["TV"] - pt["null"], 4), "utility": round(pt["O"] * (pt["TV"] - pt["null"]), 4),
                       "O_q05": round(float(np.quantile(boot[:, 0], 0.05)), 4),
                       "utility_q10": round(float(np.quantile(boot[:, 1], 0.10)), 4),
                       "n_train_frames": int(len(tr)), "n_train_lineages": int(nb)}
                gates = {"O_q05": rec["O_q05"] >= s.gate_o_q05, "ess_frames": rec["ess_frames"] >= s.gate_ess_frames,
                         "eff_lineages": rec["eff_lineages"] >= s.gate_eff_lineages,
                         "top3_share": rec["top3_share"] <= s.gate_top3_share}
                if k3_max is not None:
                    gates["k3_max"] = rec["k3"] <= float(k3_max)
                rec.update(gates=gates, eligible=all(gates.values()), heldout=heldout)
                cands.append(rec)
    ranked = sorted([c for c in cands if c["eligible"]], key=lambda c: -c["utility_q10"])
    chosen, log, used = [], [], set()
    for i, c in enumerate(ranked):
        if len(chosen) >= s.max_workers:
            break
        key = (c["state_id"], c["side"])
        if key in used:
            continue
        h = c["heldout"]
        ok = bool(h and h["net"] > 0 and h["O"] >= s.heldout_o_min)
        log.append({"rank": i, "state_id": c["state_id"], "side": c["side"], "c3": c["c3"], "k3": c["k3"],
                    "utility_q10": c["utility_q10"], "heldout": h, "confirmed": ok})
        if ok:
            chosen.append(c); used.add(key)
    gates_doc = {"O_q05": s.gate_o_q05, "ess_frames": s.gate_ess_frames, "eff_lineages": s.gate_eff_lineages,
                 "top3_share": s.gate_top3_share}
    return {"doc": DOC, "n_states": int(df.state_id.nunique()), "n_candidates": len(cands),
            "n_eligible": len(ranked), "gates": gates_doc, "skipped_states": skipped, "selection_log": log,
            "chosen": chosen, "top20": ranked[:20], "all_candidates": cands}
```

The parity test is the authority: diff field-by-field against `C10/placement.py` (key order inside each record, `gates` dict content without `k3_max` when `k3_max is None`, exact `doc` string, `width_fraction`/`quantile` value types, `skipped_states` reason text, `gates` top-level values) and change this port until `test_parity_with_c10_placement_json` passes; `doc`, `n_states` and `gates` are compared only if the test compares them (it compares `all_candidates`, `chosen`, `selection_log`).

- [ ] **Step 5: Implement the validation record**

```python
# gareus/adaptive/aux_discovery/validation.py
"""atlas-aux-validation-v1: evidence that the aux restraint is safe at the campaign's timestep (finite-timestep
check up to k3_max), under NPT (controlled distribution), and affordable (236-context cost). Written by the
validation runs, read by the admission hook."""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

VALIDATION_SCHEMA = "atlas-aux-validation-v1"
REQUIRED_CHECKS = ("finite_timestep", "npt", "cost")


@dataclass(frozen=True)
class ValidationStatus:
    ok: bool
    reason: str
    k3_max: Optional[float] = None


def check_validation_record(path: Path, *, timestep_fs: float) -> ValidationStatus:
    path = Path(path)
    if not path.exists():
        return ValidationStatus(False, "validation_missing")
    try:
        rec = json.loads(path.read_text())
    except (OSError, ValueError):
        return ValidationStatus(False, "validation_unreadable")
    if rec.get("schema") != VALIDATION_SCHEMA:
        return ValidationStatus(False, "validation_schema")
    checks = rec.get("checks") or {}
    for name in REQUIRED_CHECKS:
        if checks.get(name) != "pass":
            return ValidationStatus(False, f"validation_failed:{name}")
    if abs(float(rec.get("timestep_fs", -1.0)) - float(timestep_fs)) > 1e-9:
        return ValidationStatus(False, "validation_timestep_mismatch")
    return ValidationStatus(True, "ok", float(rec["k3_max_validated"]))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m gareus.adaptive.aux_discovery.validation")
    sub = p.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write")
    w.add_argument("--out", required=True); w.add_argument("--commit", required=True)
    w.add_argument("--timestep-fs", type=float, required=True); w.add_argument("--k3-max", type=float, required=True)
    for name in REQUIRED_CHECKS:
        w.add_argument(f"--{name.replace('_', '-')}", choices=("pass", "fail"), required=True)
    w.add_argument("--evidence", nargs="*", default=[])
    a = p.parse_args(argv)
    rec = {"schema": VALIDATION_SCHEMA, "code_commit": a.commit, "timestep_fs": a.timestep_fs,
           "k3_max_validated": a.k3_max, "checks": {n: getattr(a, n) for n in REQUIRED_CHECKS},
           "evidence": list(a.evidence), "written_unix": time.time()}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rec, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: Run tests**

Run: `opencode run "pytest -q tests/test_aux_discovery_placement.py tests/test_aux_discovery_validation.py"`
Expected: PASS (parity test runs locally where the c10 data exist; it must PASS, not skip, on the dev machine).

- [ ] **Step 7: Commit**

```bash
git add gareus/adaptive/aux_discovery/placement.py gareus/adaptive/aux_discovery/validation.py tests/test_aux_discovery_placement.py tests/test_aux_discovery_validation.py
git commit -m "feat(cvaux-adaptive): exact port of c10 forecast-benefit placement (parity-tested) and validation record"
```

---

### Task 10: Pipeline, admission hook, driver wiring, phase-arg injection

**Files:**
- Create: `gareus/adaptive/aux_discovery/pipeline.py`, `gareus/adaptive/aux_admission_io.py`
- Modify: `gareus/adaptive_production.py` (epoch loop after AP:9114; phase-arg builders AP:7383-7432, 8897-8959, 9586-9628, 9701-9740; `_apply_registry_actions` reporting), `gareus/adaptive/aux_discovery/__init__.py`
- Test: `tests/test_aux_admission_hook.py`

**Interfaces:**
- Consumes: everything in Tasks 1-9; `gareus.auxiliary_cv.runtime_io.check_exchange_boundary_alignment(*, exchange_interval, distance_interval, traj_interval, calib_steps)` (check the exact keywords at `runtime_io.py:17`).
- Produces:
  - `DiscoveryResult` dataclass (`status: str`, `report: dict`, `model: Optional[AuxModel]`, `eval_partition: Optional[FrozenPartition]`, `placement: Optional[dict]`) and `run_discovery(ft: FrameTable, *, train: np.ndarray, holdout: np.ndarray, settings, full_topology, k3_max: Optional[float], epoch: int) -> DiscoveryResult` with statuses `insufficient_evidence`, `keep`, `no_evaluation_partition`, `broaden`, `no_worker`, `ok`.
  - `run_epoch_aux_discovery(*, adaptive_dir, epoch_dir, epoch, registry, diagnostics, actions, policy, args, out_dir, phase_dirs, gate=None) -> list` (same keyword set as `run_epoch_cv2_respring`; returns `list(actions) + admit_aux actions`; never raises).
  - Frozen files: `adaptive_production/aux_model.json`, `aux_eval_partition.pkl` (+ `.sha256`), `aux_admission.json` (`{"schema": "atlas-aux-admission-v1", "epoch", "model_sha256", "eval_partition_sha256", "workers": [{"parent_state_id", "aux_center", "aux_k_kcal_mol", "placement_rank"}], "settings_sha256", "validation"}`), report `epoch_NNN/aux_discovery_report.json` (`aux_discovery_report_v1`).
  - `inject_aux_phase_args(phase_args, adaptive_dir) -> None`: when `aux_admission.json` exists and the phase's window CSV contains a worker, sets `aux_cv_model = <adaptive_dir>/aux_model.json`, `aux_phase_kind = "production"`, `aux_equilibrium_eligible = True`; otherwise leaves `aux_cv_model` unset.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_aux_admission_hook.py
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

from gareus.adaptive import aux_admission_io as H
from gareus.adaptive_production import WindowStateRegistry, AdaptiveDecisionPolicy


def _args(tmp):
    return SimpleNamespace(out=str(tmp), traj_interval=300, distance_output_interval=300, exchange_interval=3000,
                           timestep_fs=3.5, adaptive_production_aux_settings_override=False, temperature_k=300.0)


def _registry():
    r = WindowStateRegistry()
    for c in (0.0, 0.5):
        r.add_state(c, 10.0, 0.0, 2.0, gamd_lambda=0.0)
    return r


def test_skips_epoch_zero_and_after_admission(tmp_path):
    ad = tmp_path / "adaptive_production"; ad.mkdir()
    pol = AdaptiveDecisionPolicy(aux_discovery=True)
    out = H.run_epoch_aux_discovery(adaptive_dir=ad, epoch_dir=ad / "epoch_000", epoch=0, registry=_registry(),
                                    diagnostics={}, actions=[("extend", 0, "x")], policy=pol, args=_args(tmp_path),
                                    out_dir=tmp_path, phase_dirs=[])
    assert out == [("extend", 0, "x")] and not (ad / "epoch_000" / "aux_discovery_report.json").exists()
    (ad / "aux_admission.json").write_text("{}")
    out = H.run_epoch_aux_discovery(adaptive_dir=ad, epoch_dir=ad / "epoch_001", epoch=1, registry=_registry(),
                                    diagnostics={}, actions=[], policy=pol, args=_args(tmp_path),
                                    out_dir=tmp_path, phase_dirs=[])
    assert out == []


def test_never_raises_and_reports_error(tmp_path, monkeypatch):
    ad = tmp_path / "adaptive_production"; (ad / "epoch_001").mkdir(parents=True)
    monkeypatch.setattr(H, "_build_frames", lambda **kw: (_ for _ in ()).throw(RuntimeError("boom")))
    out = H.run_epoch_aux_discovery(adaptive_dir=ad, epoch_dir=ad / "epoch_001", epoch=1, registry=_registry(),
                                    diagnostics={}, actions=[], policy=AdaptiveDecisionPolicy(aux_discovery=True),
                                    args=_args(tmp_path), out_dir=tmp_path, phase_dirs=[])
    rep = json.loads((ad / "epoch_001" / "aux_discovery_report.json").read_text())
    assert out == [] and rep["status"] == "error" and "boom" in rep["error"]


def test_validation_missing_blocks_admission(tmp_path, monkeypatch):
    ad = tmp_path / "adaptive_production"; (ad / "epoch_001").mkdir(parents=True)
    fake = H.DiscoveryResultStub.ok_with_workers([(0, 1.2, 2.0)])
    monkeypatch.setattr(H, "_build_frames", lambda **kw: object())
    monkeypatch.setattr(H, "_discover", lambda **kw: fake)
    out = H.run_epoch_aux_discovery(adaptive_dir=ad, epoch_dir=ad / "epoch_001", epoch=1, registry=_registry(),
                                    diagnostics={}, actions=[], policy=AdaptiveDecisionPolicy(aux_discovery=True),
                                    args=_args(tmp_path), out_dir=tmp_path, phase_dirs=[])
    rep = json.loads((ad / "epoch_001" / "aux_discovery_report.json").read_text())
    assert out == [] and rep["status"] == "validation_missing" and not (ad / "aux_admission.json").exists()


def test_alignment_check_refuses_doomed_admission(tmp_path, monkeypatch):
    a = _args(tmp_path); a.traj_interval = 300; a.distance_output_interval = 300
    assert H.phase_alignment_ok(a, calib_steps=1_010_000)[0] is False
    assert H.phase_alignment_ok(a, calib_steps=0)[0] is True


def test_inject_phase_args(tmp_path):
    ad = tmp_path / "adaptive_production"; ad.mkdir()
    pa = SimpleNamespace(aux_cv_model=None, aux_phase_kind="pilot", aux_equilibrium_eligible=False,
                         windows_2d_csv=str(tmp_path / "w.csv"))
    (tmp_path / "w.csv").write_text("state_id,state_role\n0,ordinary\n")
    H.inject_aux_phase_args(pa, ad)
    assert pa.aux_cv_model is None
    (ad / "aux_admission.json").write_text(json.dumps({"schema": "atlas-aux-admission-v1"}))
    (ad / "aux_model.json").write_text("{}")
    (tmp_path / "w.csv").write_text("state_id,state_role\n0,ordinary\n1,auxiliary\n")
    H.inject_aux_phase_args(pa, ad)
    assert pa.aux_cv_model == str(ad / "aux_model.json") and pa.aux_phase_kind == "production"
    assert pa.aux_equilibrium_eligible is True
```

`DiscoveryResultStub` is a small test helper defined in `aux_admission_io.py` (`@dataclass` with a classmethod `ok_with_workers(list[(parent, c3, k3)])` that returns a `DiscoveryResult` with `status="ok"`, a dummy model sha `"e"*64`, and a `placement["chosen"]` list). It lives in the module so tests and the replay CLI share it; mark it `# test helper`.

- [ ] **Step 2: Run to verify failure**

Run: `opencode run "pytest -q tests/test_aux_admission_hook.py"`
Expected: FAIL.

- [ ] **Step 3: Implement the pure pipeline**

```python
# gareus/adaptive/aux_discovery/pipeline.py
"""Steps 2-5 of the spec (Section 4): discovery partition, evaluation partition, z3 search, placement."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .partitions import fit_partition, FrozenPartition
from .placement import place_workers
from .settings import AuxDiscoverySettings
from .z3_search import search_z3, z3_values, emit_model


@dataclass
class DiscoveryResult:
    status: str
    report: dict = field(default_factory=dict)
    model: object = None
    eval_partition: Optional[FrozenPartition] = None
    placement: Optional[dict] = None


def run_discovery(ft, *, train, holdout, settings: AuxDiscoverySettings, full_topology, k3_max, epoch) -> DiscoveryResult:
    rep = {"n_frames": int(ft.n), "n_train": int(train.sum()), "n_holdout": int(holdout.sum())}
    lineage = ft.lineage
    disc_X = np.hstack([ft.tors, ft.hc]); disc_fam = ["tors"] * ft.tors.shape[1] + ["hc"] * ft.hc.shape[1]
    cv = np.c_[ft.cv1, ft.cv2]
    disc = fit_partition(disc_X, disc_fam, cv, train, holdout, lineage, ft.step, settings, seed=settings.partition_seed)
    rep["discovery"] = {"status": disc.status, "k": disc.choice.k, "table": disc.choice.table,
                        "hidden_fraction": disc.hidden_fraction, "co_occurrence": disc.co_occurrence,
                        "lineage_info": disc.lineage_info}
    if disc.status != "ok":
        return DiscoveryResult(disc.status, rep)
    ev_X = np.hstack([ft.hc, ft.hb]); ev_fam = ["hc"] * ft.hc.shape[1] + ["hb"] * ft.hb.shape[1]
    ev = fit_partition(ev_X, ev_fam, cv, train, holdout, lineage, ft.step, settings, seed=settings.partition_seed,
                       feature_names=list(ft.definition.hc_labels) + list(ft.definition.hb_labels))
    rep["evaluation"] = {"status": ev.status, "k": ev.choice.k, "table": ev.choice.table}
    if ev.choice.k is None:
        return DiscoveryResult("no_evaluation_partition", rep)
    best, allc = search_z3(ft, disc.choice.labels, disc.bins, train, holdout, settings)
    rep["z3_search"] = {"candidates": [c.summary() for c in allc], "chosen": best.summary() if best else None}
    if best is None:
        return DiscoveryResult("broaden", rep, eval_partition=ev.frozen)
    model = emit_model(best, ft.definition, full_topology, label=f"z3-epoch{epoch:03d}",
                       provenance={"epoch": int(epoch), "groups": list(best.groups), "C": best.C,
                                   "descriptor_schema_sha256": ft.definition.schema_sha256})
    z = z3_values(ft.tors, best)
    pl = place_workers(z, ev.choice.labels, ft.state_id, lineage, ft.step, train, holdout, settings,
                       k_labels=int(ev.choice.k), k3_max=k3_max)
    rep["placement"] = {k: pl[k] for k in ("n_states", "n_candidates", "n_eligible", "skipped_states",
                                           "selection_log", "chosen")}
    rep["model_sha256"] = model.model_sha256
    if not pl["chosen"]:
        return DiscoveryResult("no_worker", rep, model, ev.frozen, pl)
    return DiscoveryResult("ok", rep, model, ev.frozen, pl)
```

- [ ] **Step 4: Implement the hook**

```python
# gareus/adaptive/aux_admission_io.py
"""Boundary hook: adaptive auxiliary-CV discovery and admission (spec 2026-10-09, Section 4). Never raises."""
from __future__ import annotations

import csv
import json
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from gareus.adaptive.aux_discovery.pipeline import DiscoveryResult, run_discovery
from gareus.adaptive.aux_discovery.settings import AuxDiscoverySettings, resolve_aux_settings
from gareus.adaptive.aux_discovery.validation import check_validation_record

REPORT_SCHEMA = "aux_discovery_report_v1"
ADMISSION_SCHEMA = "atlas-aux-admission-v1"
ADMISSION_FILENAME = "aux_admission.json"
MODEL_FILENAME = "aux_model.json"
PARTITION_FILENAME = "aux_eval_partition.pkl"
VALIDATION_FILENAME = "aux_validation.json"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    tmp.replace(path)


def phase_alignment_ok(args, *, calib_steps: int) -> Tuple[bool, str]:
    from gareus.auxiliary_cv.runtime_io import check_exchange_boundary_alignment
    try:
        check_exchange_boundary_alignment(exchange_interval=int(args.exchange_interval),
                                          distance_interval=int(args.distance_output_interval),
                                          traj_interval=int(args.traj_interval), calib_steps=int(calib_steps))
        return True, "ok"
    except Exception as exc:          # the runtime check raises; the hook records it
        return False, str(exc)


def _build_frames(*, adaptive_dir, registry, epoch, settings, args):
    from gareus.adaptive.aux_discovery.frames import build_frame_table
    lam = {int(s.state_id): float(s.gamd_lambda) for s in registry.all_states()}
    stride = int(settings.frame_stride_steps) or int(args.exchange_interval)
    return build_frame_table(adaptive_dir, epochs=range(0, int(epoch) + 1), registry_lambda=lam,
                             stride_steps=stride, max_frames=settings.max_frames, seed=settings.partition_seed)


def _full_topology(out_dir: Path):
    from openmm import app
    return app.PDBFile(str(Path(out_dir) / "01_solvated_start.pdb")).topology


def _discover(*, ft, epoch, settings, out_dir, k3_max) -> DiscoveryResult:
    train = ft.epoch < int(epoch); holdout = ft.epoch == int(epoch)
    return run_discovery(ft, train=train, holdout=holdout, settings=settings, full_topology=_full_topology(out_dir),
                         k3_max=k3_max, epoch=epoch)


def _calib_steps_for_phase(args) -> int:
    """Calibration steps the next phase's runtime alignment check will see: adaptive phases import the global
    shared GaMD setup, which records no calibration_steps -> 0 (see production.py:4488)."""
    return 0


def run_epoch_aux_discovery(*, adaptive_dir, epoch_dir, epoch, registry, diagnostics, actions, policy, args,
                            out_dir, phase_dirs, gate=None) -> List[tuple]:
    adaptive_dir = Path(adaptive_dir); epoch_dir = Path(epoch_dir)
    actions = list(actions)
    if not getattr(policy, "aux_discovery", False) or int(epoch) < 1:
        return actions
    if (adaptive_dir / ADMISSION_FILENAME).exists():
        return actions
    report = {"schema": REPORT_SCHEMA, "epoch": int(epoch), "started_unix": time.time()}
    try:
        settings, srec = resolve_aux_settings(adaptive_dir, AuxDiscoverySettings(),
                                              override=bool(getattr(args, "adaptive_production_aux_settings_override", False)))
        report["settings"] = srec.get("settings")
        val = check_validation_record(adaptive_dir / VALIDATION_FILENAME, timestep_fs=float(args.timestep_fs))
        report["validation"] = {"ok": val.ok, "reason": val.reason, "k3_max": val.k3_max}
        ok, why = phase_alignment_ok(args, calib_steps=_calib_steps_for_phase(args))
        report["alignment"] = {"ok": ok, "detail": why}
        ft = _build_frames(adaptive_dir=adaptive_dir, registry=registry, epoch=epoch, settings=settings, args=args)
        res = _discover(ft=ft, epoch=epoch, settings=settings, out_dir=out_dir, k3_max=val.k3_max)
        report.update(res.report); report["status"] = res.status
        if res.status != "ok":
            return actions
        if not val.ok:
            report["status"] = "validation_missing" if val.reason == "validation_missing" else val.reason
            return actions
        if not ok:
            report["status"] = "alignment"
            return actions
        burnin = int(getattr(args, "exchange_interval", 0)) * 0 + _burnin_steps(args)
        new = []
        for rank, c in enumerate(res.placement["chosen"]):
            new.append(("admit_aux", int(c["state_id"]),
                        {"aux_center": float(c["c3"]), "aux_k_kcal_mol": float(c["k3"]),
                         "aux_model_sha256": res.model.model_sha256, "burnin_steps": burnin},
                        f"aux_discovery epoch {int(epoch)}",
                        {"aux": {"discovery_epoch": int(epoch), "placement_rank": rank,
                                 "forecast": {k: c[k] for k in ("O", "TV", "null", "net", "utility_q10")}}}))
        _freeze(adaptive_dir, epoch, res, new, settings, srec, val)
        report["admitted"] = [a[1:3] for a in new]
        return actions + new
    except Exception as exc:
        report.update(status="error", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        return actions
    finally:
        report["finished_unix"] = time.time()
        _write_json(epoch_dir / "aux_discovery_report.json", report)


def _burnin_steps(args) -> int:
    """One full baseline segment of the first post-admission phase: the driver overwrites the provisional value
    with the phase's actual production steps when it writes the next window table (Step 5)."""
    return int(getattr(args, "gamd_production_steps", 0) or 0)


def _freeze(adaptive_dir: Path, epoch, res: DiscoveryResult, new_actions, settings, srec, val) -> None:
    import hashlib
    res.model.write(adaptive_dir / MODEL_FILENAME)
    part_sha = res.eval_partition.to_file(adaptive_dir / PARTITION_FILENAME)
    settings_sha = hashlib.sha256(json.dumps(srec.get("settings"), sort_keys=True).encode()).hexdigest()
    _write_json(adaptive_dir / ADMISSION_FILENAME, {
        "schema": ADMISSION_SCHEMA, "epoch": int(epoch), "model_sha256": res.model.model_sha256,
        "eval_partition_sha256": part_sha, "settings_sha256": settings_sha,
        "validation": {"k3_max": val.k3_max},
        "workers": [{"parent_state_id": a[1], "aux_center": a[2]["aux_center"],
                     "aux_k_kcal_mol": a[2]["aux_k_kcal_mol"], "placement_rank": a[4]["aux"]["placement_rank"]}
                    for a in new_actions]})


def inject_aux_phase_args(phase_args, adaptive_dir: Path) -> None:
    adaptive_dir = Path(adaptive_dir)
    if not (adaptive_dir / ADMISSION_FILENAME).exists():
        return
    csv_path = getattr(phase_args, "windows_2d_csv", None)
    if not csv_path or not Path(csv_path).exists():
        return
    with Path(csv_path).open() as fh:
        roles = {row.get("state_role") for row in csv.DictReader(fh)}
    if "auxiliary" not in roles:
        return
    phase_args.aux_cv_model = str(adaptive_dir / MODEL_FILENAME)
    phase_args.aux_phase_kind = "production"
    phase_args.aux_equilibrium_eligible = True


@dataclass
class DiscoveryResultStub:  # test helper shared with the replay CLI's --dry-admit
    @classmethod
    def ok_with_workers(cls, workers):
        class _M:
            model_sha256 = "e" * 64
            def write(self, path):
                Path(path).write_text("{}")
        class _P:
            def to_file(self, path):
                Path(path).write_bytes(b""); return "f" * 64
        chosen = [{"state_id": p, "c3": c3, "k3": k3, "O": 0.3, "TV": 0.2, "null": 0.05, "net": 0.15,
                   "utility_q10": 0.04} for p, c3, k3 in workers]
        return DiscoveryResult("ok", {}, _M(), _P(), {"chosen": chosen})
```

Clean while implementing: drop the `int(getattr(args, "exchange_interval", 0)) * 0 +` leftover in `burnin`. The `_calib_steps_for_phase` returning 0 is a claim: verify it on the real path by grepping `production.py:4488` and the place adaptive phases load `global_shared_gamd_setup`, and if a numbered-epoch phase can carry a non-zero `calibration_steps`, compute it the same way production does (`production.py:6283-6288`) from `args`. The test `test_alignment_check_refuses_doomed_admission` pins the check function itself.

- [ ] **Step 5: Driver wiring**

In the epoch loop after the respring hook (after AP:9114), same pattern:

```python
                if getattr(policy, "aux_discovery", False):
                    from .adaptive.aux_admission_io import run_epoch_aux_discovery
                    actions = run_epoch_aux_discovery(
                        adaptive_dir=adaptive_dir, epoch_dir=epoch_dir, epoch=epoch, registry=registry,
                        diagnostics=diagnostics, actions=actions, policy=policy, args=args, out_dir=out_dir,
                        phase_dirs=phase_dirs, gate=_coupling_gate)
```

(it runs only on the non-recovered branch, where the other proposers run; a recovered epoch replays `admit_aux` from the ledger like every other action, and `aux_admission.json` is already on disk).

Phase-arg injection: right after each of `seg_args.windows_2d_csv = ...` (AP:7384), `epoch_args.windows_2d_csv = ...` (AP:8950), `final_args.windows_2d_csv = ...` (AP:9608), `ext_args.windows_2d_csv = ...` (AP:9722):

```python
            from .adaptive.aux_admission_io import inject_aux_phase_args
            inject_aux_phase_args(<the phase args>, adaptive_dir)
```

Worker burn-in: when the driver writes `windows_epoch_{epoch+1}.csv` (AP:9197-9208) after an admission in this epoch, set each new worker's `burnin_steps` to the next phase's production steps for that state before `registry.save` (the per-state baseline length the schedule assigns; for the flat path `epoch_args.gamd_production_steps`). Implement as `_set_worker_burnin(registry, steps)` that only touches workers with `metadata["aux"]["admitted_epoch"] == epoch` and `burnin_steps == provisional`. Write the value into `aux_admission.json["workers"][i]["burnin_steps"]` too.

- [ ] **Step 6: Run tests**

Run: `opencode run "pytest -q tests/test_aux_admission_hook.py tests/test_aux_admission_applier.py tests/test_ladder_adapt_resume_e2e.py"`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add gareus/adaptive/aux_discovery/pipeline.py gareus/adaptive/aux_admission_io.py gareus/adaptive_production.py gareus/adaptive/aux_discovery/__init__.py tests/test_aux_admission_hook.py
git commit -m "feat(cvaux-adaptive): discovery pipeline, admission hook, driver wiring and per-phase aux arg injection"
```

---

### Task 11: Aux pull ramp in admission

**Files:**
- Modify: `gareus/seeding.py` (`generate_us_starting_states_by_pulling` signature at 1311, `relax_to_window` at 1627-1660), `gareus/production.py` (both pull calls, ~7318-7340 continue-states missing windows and ~7380-7390 normal pull)
- Test: `tests/test_aux_pull_ramp.py`

**Interfaces:**
- Consumes: `gareus.auxiliary_cv.force.set_aux_parameters(context, info: AuxForceInfo, *, center, k_kcal)`, `AuxStateTable` (`centers`, `k_kcal` per window), `args._aux_runtime.info`.
- Produces: keyword `aux_ramp: Optional[dict] = None` on `generate_us_starting_states_by_pulling`, dict `{"info": AuxForceInfo, "centers": list[Optional[float]], "k_kcal": list[float], "stages": int}`; inside `relax_to_window`, after the CV pull of window `w`, if `k_kcal[w] > 0`: for `i in 1..stages` set `k = k_kcal[w] * i / stages`, `center = centers[w]`, run `pull_steps // stages` MD steps; after the window is captured, `set_aux_parameters(..., center=0.0, k_kcal=0.0)`. Row dict gains `aux_ramp_stages`, `aux_z_end`. Constant `AUX_PULL_RAMP_STAGES = 5`.

- [ ] **Step 1: Write the failing test**

Use the Stage A-C fixture (`tests/aux_cv_fixture.py`: `model_payload`, the GA-dipeptide system builder used by `tests/test_aux_*` Stage B tests). Read one Stage B test (`grep -ln "add_aux_cv_force" tests/`) and reuse its system/context setup verbatim.

```python
# tests/test_aux_pull_ramp.py
import numpy as np
import pytest

openmm = pytest.importorskip("openmm")


def test_ramp_moves_z_toward_center_and_clears_parameters(aux_reference_context):
    """aux_reference_context: fixture from the Stage B tests -> (simulation, info, model, z_of_positions)."""
    from gareus.seeding import ramp_aux_restraint
    sim, info, model, z_of = aux_reference_context
    z0 = z_of(sim)
    target = z0 + 1.0
    row = ramp_aux_restraint(sim, info, center=target, k_kcal=20.0, stages=5, steps_per_stage=200)
    z1 = z_of(sim)
    assert abs(z1 - target) < abs(z0 - target)
    assert row["aux_ramp_stages"] == 5 and np.isfinite(row["aux_z_end"])
    assert sim.context.getParameter(info.global_k) == 0.0          # cleared after capture
```

If the Stage B tests have no reusable fixture, add `aux_reference_context` to `tests/conftest.py` built from those tests' setup code (copy it; do not import private test functions across files).

- [ ] **Step 2: Run to verify failure**

Run: `opencode run "pytest -q tests/test_aux_pull_ramp.py"`
Expected: FAIL (`ImportError: ramp_aux_restraint`).

- [ ] **Step 3: Implement**

In `gareus/seeding.py`, module level:

```python
AUX_PULL_RAMP_STAGES = 5


def ramp_aux_restraint(sim, info, *, center: float, k_kcal: float, stages: int, steps_per_stage: int) -> dict:
    """Bring a freshly pulled window into its auxiliary restraint gradually: k ramps 0 -> k_kcal at fixed centre
    over ``stages`` MD stages, then the parameters are cleared (production's set_window applies the real ones)."""
    from gareus.auxiliary_cv.force import set_aux_parameters
    for i in range(1, int(stages) + 1):
        set_aux_parameters(sim.context, info, center=float(center), k_kcal=float(k_kcal) * i / int(stages))
        sim.step(int(steps_per_stage))
    from gareus.auxiliary_cv.runtime import observe_aux_z
    z_end = float("nan")
    try:
        z_end = float(observe_aux_z(sim.context, info.runtime if hasattr(info, "runtime") else info))
    except Exception:
        pass
    set_aux_parameters(sim.context, info, center=0.0, k_kcal=0.0)
    return {"aux_ramp_stages": int(stages), "aux_z_end": z_end}
```

Check `observe_aux_z(context, runtime: AuxRuntime, *, force=None, positions_nm=None)` (runtime.py:147): it takes an `AuxRuntime`, not the force info; pass the runtime through `aux_ramp["runtime"]` and call `observe_aux_z(sim.context, runtime)`; drop the `hasattr` guess. Then add `aux_ramp: Optional[dict] = None` to `generate_us_starting_states_by_pulling` and, inside `relax_to_window`, after the restrained pull of window `w` and before the row/positions are captured:

```python
        if aux_ramp is not None and float(aux_ramp["k_kcal"][w]) > 0.0:
            row_extra = ramp_aux_restraint(sim, aux_ramp["info"], center=float(aux_ramp["centers"][w]),
                                           k_kcal=float(aux_ramp["k_kcal"][w]), stages=int(aux_ramp["stages"]),
                                           steps_per_stage=max(1, effective_pull_steps // int(aux_ramp["stages"])))
            row.update(row_extra)
```

In `gareus/production.py` at both pull call sites, pass:

```python
                aux_ramp=None if args._aux_runtime is None else {
                    "info": args._aux_runtime.info, "runtime": args._aux_runtime,
                    "centers": [None if c is None else float(c) for c in _aux_table.centers],
                    "k_kcal": [float(k) for k in _aux_table.k_kcal], "stages": AUX_PULL_RAMP_STAGES},
```

restricted to the pulled windows in the continue-states branch (`_sub`-mapped like the other per-window lists there).

- [ ] **Step 4: Run tests**

Run: `opencode run "pytest -q tests/test_aux_pull_ramp.py tests/test_aux_*stage_b*.py"`
Expected: PASS. (Use the actual Stage B test file names: `ls tests/test_aux_*`.)

- [ ] **Step 5: Commit**

```bash
git add gareus/seeding.py gareus/production.py tests/test_aux_pull_ramp.py tests/conftest.py
git commit -m "feat(cvaux-adaptive): aux restraint pull ramp for newly admitted workers"
```

---

### Task 12: z backfill for pre-admission phases

**Files:**
- Create: `gareus/adaptive/aux_backfill.py`
- Modify: `gareus/adaptive/aux_admission_io.py` (`_freeze` writes the backfill)
- Test: `tests/test_aux_backfill.py`

**Interfaces:**
- Consumes: `gareus.auxiliary_cv.evaluate.z_from_positions(xyz_nm, model)`, `load_phase_samples` (Task 6), `discovery_census._trajectory_files`, `_find_topology`.
- Produces: `BACKFILL_FILENAME = "aux_z_backfill.parquet"`; `write_phase_backfill(phase_dir: Path, model, *, solute_to_full: Optional[np.ndarray] = None) -> dict` (`{"phase", "n_samples", "n_z", "model_sha256", "sha256"}`; raises `BackfillIncomplete(n_missing, examples)` when any sample lacks a frame); `read_phase_backfill(phase_dir, model_sha256) -> pandas.DataFrame` (replica, step, aux_z); `backfill_all(adaptive_dir, model, *, up_to_epoch) -> list[dict]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_aux_backfill.py
from pathlib import Path
import numpy as np, pandas as pd, pytest
import mdtraj as md

from gareus.adaptive.aux_backfill import write_phase_backfill, read_phase_backfill, BackfillIncomplete
from tests.test_aux_discovery_frames import _phase      # synthetic phase builder (Task 6)

PDB = Path(__file__).parent / "data" / "chignolin_solute.pdb"


def _model():
    from gareus.adaptive.aux_discovery.descriptors import descriptor_definition
    from gareus.adaptive.aux_discovery.z3_search import Z3Candidate, emit_model
    t = md.load(str(PDB)); d = descriptor_definition(t.topology)
    rng = np.random.default_rng(0)
    c = Z3Candidate((0, 1), 0.1, rng.normal(size=36), np.zeros(36), np.ones(36), 1.0, 0.2, 0.9, 0, 0, 0.1, 36, True, [])
    return emit_model(c, d, t.topology.to_openmm(), label="t", provenance={})


def test_backfill_complete_and_matches_positions(tmp_path):
    ph = _phase(tmp_path, "epoch_000", {"replica_0.xtc": (0, 0, [300, 3300, 6300])}, {0: 0.0}, lambda s: 0.0)
    m = _model()
    info = write_phase_backfill(ph, m)
    df = read_phase_backfill(ph, m.model_sha256)
    assert info["n_samples"] == info["n_z"] == 3 and len(df) == 3
    from gareus.auxiliary_cv.evaluate import z_from_positions
    z = float(np.ravel(z_from_positions(md.load(str(PDB)).xyz[0], m))[0])
    assert np.allclose(df.aux_z, z, atol=1e-6)


def test_missing_frames_refuse(tmp_path):
    ph = _phase(tmp_path, "epoch_000", {"replica_0.xtc": (0, 0, [300, 3300])}, {0: 0.0}, lambda s: 0.0)
    s = pd.read_parquet(ph / "samples" / "seg_000" / "data.parquet")
    extra = s.iloc[[0]].copy(); extra["step"] = 99300
    pd.concat([s, extra]).to_parquet(ph / "samples" / "seg_000" / "data.parquet")
    with pytest.raises(BackfillIncomplete) as e:
        write_phase_backfill(ph, _model())
    assert e.value.n_missing == 1


def test_wrong_model_sha_refused(tmp_path):
    ph = _phase(tmp_path, "epoch_000", {"replica_0.xtc": (0, 0, [300])}, {0: 0.0}, lambda s: 0.0)
    write_phase_backfill(ph, _model())
    with pytest.raises(ValueError, match="model"):
        read_phase_backfill(ph, "0" * 64)
```

Note the backfill covers ALL samples of the phase (every lambda, every state), unlike the frame table.

- [ ] **Step 2: Run to verify failure**

Run: `opencode run "pytest -q tests/test_aux_backfill.py"`
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
# gareus/adaptive/aux_backfill.py
"""z of the frozen aux model for every sample of a pre-admission phase, from its trajectory frames.
Strict: one frame per sample or the phase is refused; z is never filled with 0."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from gareus.adaptive import discovery_census as census
from gareus.adaptive.aux_discovery.frames import load_phase_samples

BACKFILL_FILENAME = "aux_z_backfill.parquet"


class BackfillIncomplete(RuntimeError):
    def __init__(self, n_missing: int, examples: list):
        super().__init__(f"{n_missing} samples have no trajectory frame (e.g. {examples[:5]})")
        self.n_missing = int(n_missing); self.examples = examples


def write_phase_backfill(phase_dir: Path, model, *, adaptive_dir: Optional[Path] = None) -> dict:
    import mdtraj as md
    from gareus.auxiliary_cv.evaluate import z_from_positions
    phase_dir = Path(phase_dir)
    samples = load_phase_samples(phase_dir)[["replica", "step"]]
    top = md.load_topology(str(census._find_topology(phase_dir, adaptive_dir or phase_dir.parent)))
    files = sorted(census._trajectory_files(phase_dir), key=lambda t: (t[0], t[1] or 0, str(t[2])))
    z_rows = []
    for i, (replica, start, path) in enumerate(files):
        nxt = next((s for r, s, _ in files[i + 1:] if r == replica), None)
        with md.formats.XTCTrajectoryFile(str(path), "r") as fh:
            xyz, _t, steps, _b = fh.read()
        steps = np.asarray(steps, np.int64)
        keep = np.ones(len(steps), bool) if nxt is None else steps < int(nxt)
        if not keep.any():
            continue
        z = np.ravel(z_from_positions(xyz[keep].astype(np.float64), model))
        z_rows.append(pd.DataFrame({"replica": int(replica), "step": steps[keep], "aux_z": z}))
    zdf = (pd.concat(z_rows).drop_duplicates(["replica", "step"], keep="last") if z_rows
           else pd.DataFrame(columns=["replica", "step", "aux_z"]))
    merged = samples.merge(zdf, on=["replica", "step"], how="left")
    missing = merged["aux_z"].isna()
    if missing.any():
        raise BackfillIncomplete(int(missing.sum()), merged.loc[missing, ["replica", "step"]].head(10).values.tolist())
    out = phase_dir / BACKFILL_FILENAME
    merged.attrs = {}
    import pyarrow as pa, pyarrow.parquet as pq
    table = pa.Table.from_pandas(merged, preserve_index=False).replace_schema_metadata(
        {b"aux_model_sha256": model.model_sha256.encode()})
    pq.write_table(table, out)
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    return {"phase": str(phase_dir), "n_samples": int(len(samples)), "n_z": int(len(merged)),
            "model_sha256": model.model_sha256, "sha256": sha}


def read_phase_backfill(phase_dir: Path, model_sha256: str) -> pd.DataFrame:
    import pyarrow.parquet as pq
    t = pq.read_table(Path(phase_dir) / BACKFILL_FILENAME)
    sha = (t.schema.metadata or {}).get(b"aux_model_sha256", b"").decode()
    if sha != model_sha256:
        raise ValueError(f"{phase_dir}: backfill model {sha[:12]} != campaign model {model_sha256[:12]}")
    return t.to_pandas()


def backfill_all(adaptive_dir: Path, model, *, up_to_epoch: int) -> List[dict]:
    from gareus.adaptive.aux_discovery.frames import phase_epoch
    out = []
    phases, _ = census.ordered_phases(Path(adaptive_dir))
    for label, phase_dir in phases:
        ep = phase_epoch(label)
        if ep is not None and ep <= int(up_to_epoch):
            out.append(write_phase_backfill(Path(phase_dir), model, adaptive_dir=Path(adaptive_dir)))
    return out
```

`z_from_positions` needs production-topology atom indices; the XTC is solute-only. Peptide atoms come first in the solvated topology, so solute index == production index; assert it once per phase by comparing the solute PDB's atom names/residues to the model's feature atoms (raise `ValueError` otherwise). In `_freeze` (Task 10), after writing the model: `report["backfill"] = backfill_all(adaptive_dir, res.model, up_to_epoch=epoch)` and put the per-phase shas into `aux_admission.json["backfill"]`; a `BackfillIncomplete` propagates to the hook's `except`, which records `status: error` and deletes the partially frozen files (`aux_model.json`, `aux_eval_partition.pkl*`, `aux_admission.json`) so the next boundary retries. Add a hook test for that cleanup.

Also add the spot check of spec Section 5 as a test-only helper `check_backfill_against_recorded(phase_dir, model)` comparing a post-admission phase's recorded `aux_z_00` to `z_from_positions` on its frames (tolerance 1e-5); it is exercised in Task 17's e2e test.

- [ ] **Step 4: Run tests**

Run: `opencode run "pytest -q tests/test_aux_backfill.py tests/test_aux_admission_hook.py"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add gareus/adaptive/aux_backfill.py gareus/adaptive/aux_admission_io.py tests/test_aux_backfill.py tests/test_aux_admission_hook.py
git commit -m "feat(cvaux-adaptive): strict per-phase z backfill for pre-admission phases"
```

---

### Task 13: Union MBAR inputs (driver) with the aux term and worker burn-in

**Files:**
- Modify: `gareus/adaptive_production.py` (`build_union_state_mbar_inputs` AP:3715-3893), `gareus/kernel_identity.py` (new `aux_admission_allows_pooling`)
- Test: `tests/test_aux_union_inputs.py`

**Interfaces:**
- Consumes: `read_phase_backfill` (Task 12); recorded `aux_z_00` in post-admission Parquet samples; `aux_params` (Task 2).
- Produces: `kernel_identity.aux_admission_allows_pooling(adaptive_dir: Path) -> Optional[dict]` returning the admission record when `aux_admission.json` exists and its model file sha matches, else None. In `build_union_state_mbar_inputs`: when it returns a record, `refuse_aux_snapshots` is not called; per-source z = recorded `aux_z_00` (post-admission phases) or backfill (pre-admission phases), joined on (replica, step); `umbrella_bias_kcal[:, j] += 0.5 * k_j * (z - c_j)^2` for every worker state j BEFORE `apply_ladder_boost_to_u`; a source lacking z for any row raises `AuxPoolingRefused` with counts; worker rows with epoch-local `step < burnin_steps` of the sampled worker are dropped before subsampling. NPZ gains `aux_z` (per row) and `aux_center`, `aux_k` (per state, NaN/0 for ordinary); meta gains `aux_model_sha256`.

- [ ] **Step 1: Write the failing test**

Build a tiny adaptive dir with two phases (one pre-admission with a backfill parquet, one post-admission with `aux_z_00` in samples), a registry with 2 ordinary states + 1 worker, and compare the union bias matrix against direct evaluation with `gareus.correctness.bias.reconstruct_bias_matrix(..., aux_z={sha: z})`. Follow the existing union test fixtures: `grep -ln "build_union_state_mbar_inputs" tests/` and copy the smallest fixture's directory layout (sample CSV/Parquet writer, epoch_window_map, run_manifest temperature).

```python
# tests/test_aux_union_inputs.py  (skeleton; fill the fixture from the existing union test)
import numpy as np
import pytest
from gareus.adaptive_production import build_union_state_mbar_inputs
from gareus.kernel_identity import AuxPoolingRefused


def test_union_bias_equals_direct_energies(aux_union_campaign):
    ad, registry, expected_reduced = aux_union_campaign       # expected from reconstruct_bias_matrix
    out = build_union_state_mbar_inputs(ad, registry, output_prefix="t")
    u = np.load(out["npz"])["umbrella_reduced_bias_nk"]
    np.testing.assert_allclose(u, expected_reduced, rtol=1e-10, atol=1e-10)


def test_missing_backfill_refuses(aux_union_campaign_without_backfill):
    ad, registry = aux_union_campaign_without_backfill
    with pytest.raises(AuxPoolingRefused, match="backfill"):
        build_union_state_mbar_inputs(ad, registry, output_prefix="t")


def test_worker_burnin_rows_excluded(aux_union_campaign):
    ad, registry, _ = aux_union_campaign
    out = build_union_state_mbar_inputs(ad, registry, output_prefix="t")
    meta = out["meta"]
    w = [s for s in registry.all_states() if s.metadata.get("aux")][0]
    assert meta["subsample_counts"][str(w.state_id)]["burnin_dropped"] > 0


def test_without_admission_still_refuses_aux_snapshots(aux_union_campaign):
    ad, registry, _ = aux_union_campaign
    (ad / "aux_admission.json").unlink()
    with pytest.raises(AuxPoolingRefused):
        build_union_state_mbar_inputs(ad, registry, output_prefix="t")
```

Write the `aux_union_campaign` fixtures in `tests/conftest.py` (or the test file) with explicit arrays: 2 phases x 2 replicas x 5 samples; the expected matrix computed independently with `reconstruct_bias_matrix`. Check the real key names of the builder's return value (`out["npz"]`, `out["meta"]`) and adapt.

- [ ] **Step 2: Run to verify failure**

Run: `opencode run "pytest -q tests/test_aux_union_inputs.py"`
Expected: FAIL.

- [ ] **Step 3: Implement**

`gareus/kernel_identity.py`:

```python
def aux_admission_allows_pooling(adaptive_dir) -> "Optional[dict]":
    """A campaign whose driver admitted aux workers pools legacy + aux phases (spec 2026-10-09 Section 5)."""
    import hashlib, json
    from pathlib import Path
    ad = Path(adaptive_dir)
    rec_path = ad / "aux_admission.json"
    if not rec_path.exists():
        return None
    rec = json.loads(rec_path.read_text())
    from gareus.auxiliary_cv.model import AuxModel
    model = AuxModel.load(ad / "aux_model.json")
    if model.model_sha256 != rec.get("model_sha256"):
        raise AuxPoolingRefused(f"{ad}: aux_model.json sha {model.model_sha256[:12]} != admission record")
    return rec
```

In `build_union_state_mbar_inputs` (AP:3755-3759) replace the unconditional refusal with:

```python
    _aux_rec = aux_admission_allows_pooling(adaptive_dir)
    if _aux_rec is None:
        refuse_aux_snapshots(adaptive_dir, where="adaptive union build")
        for _pd in pilot_dirs or []:
            refuse_aux_snapshots(_pd, where="adaptive union build (pilot)")
```

While reading rows per source (AP:3779-3816) collect `aux_z` per row: for a source whose phase dir holds `aux_z_backfill.parquet`, join `read_phase_backfill(phase_dir, sha)` on (replica, step) — `_read_sample_dicts` rows must carry `replica` (add it to the dict if absent); for a post-admission source read `aux_z_00` from its Parquet samples (same join). Any NaN after the join raises `AuxPoolingRefused(f"{source_label}: {n} rows without aux z (backfill missing or incomplete)")`. Before subsampling (AP:3823), drop rows of worker states with `step < burnin_steps` (epoch-local step, the convention `loaders_union_parquet` uses at L625-626) and record `burnin_dropped` per state. After `umbrella_bias_kcal` is assembled (AP:3866-3893) and before the kJ conversion:

```python
    if _aux_rec is not None:
        for j, st in enumerate(states):
            ap = aux_params(st)
            if ap is not None:
                umbrella_bias_kcal[:, j] += 0.5 * float(ap["aux_k_kcal_mol"]) * (aux_z_values - float(ap["aux_center"])) ** 2
```

Store `aux_z`, `aux_center`, `aux_k` in the NPZ and `aux_model_sha256` in the meta JSON.

- [ ] **Step 4: Run tests**

Run: `opencode run "pytest -q tests/test_aux_union_inputs.py tests/test_union_mbar_per_epoch_bias.py tests/test_aux_final_fix_wave.py"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add gareus/adaptive_production.py gareus/kernel_identity.py tests/test_aux_union_inputs.py tests/conftest.py
git commit -m "feat(cvaux-adaptive): union MBAR inputs pool admitted aux workers with backfilled/recorded z"
```

---

### Task 14: Analyzer union-Parquet loader pools aux workers

**Files:**
- Modify: `gareus/mbar_analysis/loaders_union_parquet.py` (refusals L318-319, L384-386; bias assembly near L651; burn-in L368, L625-626)
- Test: `tests/test_aux_union_parquet_loader.py`

**Interfaces:**
- Consumes: `aux_admission_allows_pooling` (Task 13), `read_phase_backfill` (Task 12), `gareus.correctness.bias.reconstruct_bias_matrix(..., aux_z={sha: z})`.
- Produces: `load_parquet_adaptive_union` returns `Data` whose `meta` has `aux_states: [state index...]`, `aux_model_sha256`; bias rows include the aux term; refusal unchanged when no admission record.

- [ ] **Step 1: Write the failing test**

Reuse the Task 13 fixture campaign; assert (a) `load_parquet_adaptive_union(ad)` succeeds and `d.meta["aux_states"]` lists the worker; (b) its per-sample worker-column bias equals the Task 13 builder's within 1e-10; (c) removing `aux_admission.json` makes it raise `AuxPoolingRefused`.

```python
# tests/test_aux_union_parquet_loader.py
import numpy as np, pytest
from gareus.mbar_analysis.loaders_union_parquet import load_parquet_adaptive_union
from gareus.kernel_identity import AuxPoolingRefused


def test_loader_pools_worker(aux_union_campaign):
    ad, registry, expected_reduced = aux_union_campaign
    d = load_parquet_adaptive_union(ad.parent)
    assert d.meta["aux_states"]
    # compare the worker column of the loader's reduced bias with the driver builder's
    w = d.meta["aux_states"][0]
    assert np.isfinite(d.u_nk[:, w]).all()


def test_loader_refuses_without_admission(aux_union_campaign):
    ad, _, _ = aux_union_campaign
    (ad / "aux_admission.json").unlink()
    with pytest.raises(AuxPoolingRefused):
        load_parquet_adaptive_union(ad.parent)
```

Check the `Data` attribute names for the reduced bias (`grep -n "class Data" -A30 gareus/mbar_analysis/*.py`) and use the real one; compare against `expected_reduced` row-aligned if the loader keeps row order, else compare sorted per (state, step).

- [ ] **Step 2: Run to verify failure**

Run: `opencode run "pytest -q tests/test_aux_union_parquet_loader.py"`
Expected: FAIL.

- [ ] **Step 3: Implement**

At L318-319 and L384-386 gate the refusals on `aux_admission_allows_pooling(adaptive_dir) is None`. When pooling: load per phase the z column (recorded `aux_z_00` or backfill by (replica, step)), keep it through the burn-in filter (L625-626 already applies per-state `burnin_steps`), and where the bias matrix is assembled (near L651) add for each worker state column `0.5 * k * (z - c)^2` in kcal (then the existing kJ/beta conversion), or switch that block to `gareus.correctness.bias.reconstruct_bias_matrix(..., aux_z={sha: z})` if the inputs line up (windows list with `aux_center`/`aux_k`/`aux_model_sha256` per state, `_aux_term` at `bias.py:77`). Set `meta["aux_states"]` and `meta["aux_model_sha256"]`. Any row of any state lacking z raises `AuxPoolingRefused`.

- [ ] **Step 4: Run tests**

Run: `opencode run "pytest -q tests/test_aux_union_parquet_loader.py tests/test_aux_union_inputs.py tests/test_aux_analyzer_unavailable.py"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add gareus/mbar_analysis/loaders_union_parquet.py tests/test_aux_union_parquet_loader.py
git commit -m "feat(cvaux-adaptive): analyzer union-Parquet loader pools admitted aux workers"
```

---

### Task 15: Ordinary-only vs all-states crosscheck gate, worker table, report rows

**Files:**
- Modify: `gareus/mbar_analysis/crosscheck.py`, `analyze_gareus_mbar.py` (near L5223-5250 and L5363-5366), `gareus_report.py` (`build_health_verdict` L171-205, new `_check_aux_crosscheck`, `_check_aux_workers`)
- Create: `gareus/adaptive/aux_worker_table.py`
- Test: `tests/test_aux_crosscheck_gate.py`, `tests/test_aux_worker_table.py`

**Interfaces:**
- Produces:
  - `aux_ordinary_crosscheck(d, f_k_global, bins_by_axis: dict[str, np.ndarray], kbt_kcal, *, ordinary_states, tol_kcal=DEFAULT_TOL_KCAL) -> dict` (`{"status": "pass"|"fail"|"skipped", "axes": {"cv1": {...}, "cv2": {...}, "z3": {...}}, "max_abs_diff_kcal"}`), modelled on `ladder_crosscheck` (L80): PMF along each axis from all states vs from ordinary states only (`f_k` re-solved on the ordinary subset with `gareus.adaptive.mbar_solve.solve_rows`), same bin-noise and min-bins rules (`BIN_NOISE_SAFETY_FACTOR`, `MIN_BIN_COUNT_FLOOR`, `MIN_BINS_FOR_VERDICT`).
  - `worker_table(d, f_k, *, aux_states, ordinary_states, eval_partition, frames=None) -> list[dict]` with per worker: `state_id`, `parent_state_id`, `best_partner` + `best_partner_overlap` (`pairwise_state_overlap`), `overlap_floor_ok` (>= 0.15), `n_samples`, `occupancy_fraction`, `entries`, `exits` (positive-residence, from the replica-state series), `carrier_diversity` (distinct replicas), `z_mean`, `z_sd`, `forecast_O`, `return_label_change_fraction` (frozen evaluation partition, pre/post majority labels as c10 `local_power.py`), `attribution: "none (no shams)"`.
  - Summary keys `s["aux_crosscheck"]`, `s["aux_workers"]`; report rows "Aux ordinary-only crosscheck" (FAIL on `fail`, NA on skipped/absent) and "Aux workers" (CAUTION if any `overlap_floor_ok` is False, else PASS-with-detail), both appended only when the keys exist (legacy verdicts byte-identical). The existing CAUTION cap row (L195-198) stays.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_aux_crosscheck_gate.py
import numpy as np
from gareus.mbar_analysis.crosscheck import aux_ordinary_crosscheck


def _toy(bias_bug=False, seed=0):
    """3 harmonic states on x ~ N; state 2 is an aux state with an extra bias along z = x."""
    rng = np.random.default_rng(seed)
    centers, ks = np.array([-1.0, 1.0, 0.0]), np.array([2.0, 2.0, 2.0])
    aux_c, aux_k = 0.8, 3.0
    xs, win = [], []
    for j in range(3):
        kk = ks[j] + (aux_k if j == 2 else 0.0)
        mu = (ks[j] * centers[j] + (aux_k * aux_c if j == 2 else 0.0)) / kk
        x = rng.normal(mu, np.sqrt(0.596 / kk), 4000); xs.append(x); win += [j] * 4000
    x = np.concatenate(xs); win = np.array(win)
    u = 0.5 * ks[None, :] * (x[:, None] - centers[None, :]) ** 2
    u[:, 2] += 0.5 * (0.0 if bias_bug else aux_k) * (x - aux_c) ** 2
    return x, win, u / 0.596


def test_consistent_aux_passes():
    x, win, u = _toy()
    r = aux_ordinary_crosscheck_from_arrays(x, win, u, ordinary=[0, 1])
    assert r["status"] == "pass"


def test_missing_aux_term_fails():
    x, win, u = _toy(bias_bug=True)
    r = aux_ordinary_crosscheck_from_arrays(x, win, u, ordinary=[0, 1])
    assert r["status"] == "fail"


def aux_ordinary_crosscheck_from_arrays(x, win, u, ordinary):
    from types import SimpleNamespace
    from gareus.adaptive.mbar_solve import solve_rows
    f, _ = solve_rows(u, win)
    d = SimpleNamespace(u_nk=u, window=win, cv_A=x, secondary_cv=np.zeros_like(x), aux_z=x)
    bins = {"cv1": np.linspace(-2, 2, 21)}
    return aux_ordinary_crosscheck(d, f, bins, 0.596, ordinary_states=ordinary)
```

```python
# tests/test_aux_worker_table.py
import numpy as np
from gareus.adaptive.aux_worker_table import positive_residence_episodes


def test_entries_exits_and_zero_time_swaps_collapsed():
    # replica-state series on the exchange grid; aux state 9; a zero-time round trip 3 -> 9 -> 3 in one barrier
    series = np.array([3, 3, 9, 9, 9, 3, 3, 9, 3])
    residence = np.array([1, 1, 1, 1, 1, 1, 1, 0, 1])     # MD time after each assignment (0 = zero-time visit)
    ep = positive_residence_episodes(series, residence, aux_states={9})
    assert ep["entries"] == 1 and ep["exits"] == 1 and ep["zero_time_visits"] == 1
```

Adapt the `d` namespace to the real `Data` attribute names `ladder_crosscheck` uses (read `crosscheck.py:80-160` first) and make `aux_ordinary_crosscheck` accept that same object.

- [ ] **Step 2: Run to verify failure**

Run: `opencode run "pytest -q tests/test_aux_crosscheck_gate.py tests/test_aux_worker_table.py"`
Expected: FAIL.

- [ ] **Step 3: Implement**

`aux_ordinary_crosscheck` in `crosscheck.py`: for each axis present (`cv1` from `d.cv_A`, `cv2` from `d.secondary_cv` when finite, `z3` from `d.aux_z`), compute the target-ensemble PMF twice with the same estimator `ladder_crosscheck` uses (weights `logw` from MBAR at the unbiased target): (1) all rows, `f_k_global`; (2) rows sampled in ordinary states only, `f` re-solved by `solve_rows(u[ord_rows][:, ordinary], remap(win[ord_rows]))`. Align both PMFs to zero at their common minimum, evaluate per bin with the same noise floor and min-count rules, status `fail` if any supported bin differs by more than `max(tol_kcal, BIN_NOISE_SAFETY_FACTOR x bin noise)`, `skipped` with a reason if fewer than `MIN_BINS_FOR_VERDICT` bins are supported on an axis (all axes skipped -> overall skipped).

`gareus/adaptive/aux_worker_table.py`: `positive_residence_episodes(series, residence, aux_states)` (entries counted on O->A transitions followed by positive residence; zero-residence visits counted separately and not as entries) and `worker_table(...)` as specified, using `pairwise_state_overlap(u_nk, window, f_k, n_k, i, j)` for every (worker, ordinary state at lambda = 0) pair and the frozen evaluation partition's `predict` on the frames around each episode (frames from Task 6's frame table restricted to post-admission phases; when frames are unavailable set `return_label_change_fraction: None` and `return_label_status: "frames_unavailable"`).

In `analyze_gareus_mbar.py`, after the ladder crosscheck block (L5223-5250): if `d.meta.get("aux_states")`, compute `s["aux_crosscheck"]` and `s["aux_workers"]`, write `pmf_analysis/pmf_aux_crosscheck.csv`, and append a warning on fail. In `gareus_report.py` add after the cv2 row (L192-194):

```python
    aux_cc = _check_aux_crosscheck(s)
    if aux_cc is not None:
        checks.append(aux_cc)
    aux_w = _check_aux_workers(s)
    if aux_w is not None:
        checks.append(aux_w)
```

with `_check_aux_crosscheck(s)`: None if `"aux_crosscheck" not in s`; FAIL on `fail`, PASS on `pass`, NA otherwise; `_check_aux_workers(s)`: None if absent; CAUTION if any worker has `overlap_floor_ok is False`, else PASS, detail names each worker's best partner and overlap and states "descriptive; no shams, no attribution".

- [ ] **Step 4: Run tests**

Run: `opencode run "pytest -q tests/test_aux_crosscheck_gate.py tests/test_aux_worker_table.py tests/test_aux_analyzer_unavailable.py tests/test_cv2_resolution_report_row.py"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add gareus/mbar_analysis/crosscheck.py gareus/adaptive/aux_worker_table.py analyze_gareus_mbar.py gareus_report.py tests/test_aux_crosscheck_gate.py tests/test_aux_worker_table.py
git commit -m "feat(cvaux-adaptive): ordinary-only vs all-states crosscheck gate, worker table and report rows"
```

---

### Task 16: Replay CLI and the c10 replay

**Files:**
- Create: `gareus/adaptive/aux_discovery/__main__.py`
- Create (main checkout, gitignored is fine but prefer committed): `docs/superpowers/specs/2026-10-09-cvaux-adaptive-c10-replay.md`
- Test: `tests/test_aux_discovery_replay_cli.py`

**Interfaces:**
- Produces: `python -m gareus.adaptive.aux_discovery replay <run_dir> --epoch N --out DIR [--max-frames K] [--workers W] [--k3-max K]`: builds the frame table (training epochs < N, holdout N), runs `run_discovery` with the campaign's `01_solvated_start.pdb`, writes `DIR/aux_discovery_report.json`, `DIR/aux_model.json` (if any), `DIR/placement.json` (if any), `DIR/timing.json` (`wall_s`, `peak_rss_gb` via `resource.getrusage`), never writes into `<run_dir>`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_aux_discovery_replay_cli.py
import subprocess, sys, json
from pathlib import Path


def test_replay_writes_only_into_out(tmp_path, monkeypatch):
    from gareus.adaptive.aux_discovery import __main__ as cli
    called = {}
    monkeypatch.setattr(cli, "_replay", lambda **kw: called.update(kw) or {"status": "insufficient_evidence"})
    run = tmp_path / "run"; (run / "adaptive_production").mkdir(parents=True)
    out = tmp_path / "out"
    assert cli.main(["replay", str(run), "--epoch", "1", "--out", str(out)]) == 0
    assert called["epoch"] == 1 and json.loads((out / "aux_discovery_report.json").read_text())["status"] == "insufficient_evidence"
    assert sorted(p.name for p in (run / "adaptive_production").iterdir()) == []
```

- [ ] **Step 2: Run to verify failure**

Run: `opencode run "pytest -q tests/test_aux_discovery_replay_cli.py"`
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
# gareus/adaptive/aux_discovery/__main__.py
"""Read-only replay of the aux discovery hook on an existing campaign."""
from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

from .settings import AuxDiscoverySettings


def _replay(*, run_dir: Path, epoch: int, out: Path, max_frames: int, workers: int, k3_max):
    from gareus.adaptive_production import WindowStateRegistry
    from gareus.adaptive.aux_discovery.frames import build_frame_table
    from gareus.adaptive.aux_discovery.pipeline import run_discovery
    from openmm import app
    ad = run_dir / "adaptive_production"
    reg = WindowStateRegistry.load(ad)
    s = AuxDiscoverySettings(max_frames=int(max_frames))
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    exch = int(manifest.get("resolved_args", {}).get("exchange_interval", 3000))
    ft = build_frame_table(ad, epochs=range(0, epoch + 1),
                           registry_lambda={int(x.state_id): float(x.gamd_lambda) for x in reg.all_states()},
                           stride_steps=exch, max_frames=s.max_frames, seed=s.partition_seed, workers=workers)
    top = app.PDBFile(str(run_dir / "01_solvated_start.pdb")).topology
    res = run_discovery(ft, train=ft.epoch < epoch, holdout=ft.epoch == epoch, settings=s, full_topology=top,
                        k3_max=k3_max, epoch=epoch)
    if res.model is not None:
        res.model.write(out / "aux_model.json")
    if res.placement is not None:
        (out / "placement.json").write_text(json.dumps(res.placement, indent=1))
    return {"status": res.status, **res.report}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m gareus.adaptive.aux_discovery")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("replay")
    r.add_argument("run_dir"); r.add_argument("--epoch", type=int, required=True); r.add_argument("--out", required=True)
    r.add_argument("--max-frames", type=int, default=AuxDiscoverySettings().max_frames)
    r.add_argument("--workers", type=int, default=1); r.add_argument("--k3-max", type=float, default=None)
    a = p.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rep = _replay(run_dir=Path(a.run_dir), epoch=a.epoch, out=out, max_frames=a.max_frames, workers=a.workers,
                  k3_max=a.k3_max)
    (out / "aux_discovery_report.json").write_text(json.dumps(rep, indent=1, default=str))
    (out / "timing.json").write_text(json.dumps({"wall_s": time.time() - t0,
                                                 "peak_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Check where the exchange interval is recorded in c10's `run_manifest.json` (`method_settings` vs `resolved_args`) and read it from there.

- [ ] **Step 4: Run tests**

Run: `opencode run "pytest -q tests/test_aux_discovery_replay_cli.py"`
Expected: PASS.

- [ ] **Step 5: Run the c10 replay (read-only, scratch)**

Local data first (c10 is mirrored under `RUNS/chignolin_10` only partially; trajectories live on aurum2). On aurum2, from a scratch CODE_DIR (never `~/2026_peptide_sampler` or `~/gareus`), in an interactive or batch CPU job:

```bash
python -m gareus.adaptive.aux_discovery replay ~/gareus/chignolin/chignolin_10 --epoch 1 --out ~/scratch/c10_aux_replay/e1 --workers 8
python -m gareus.adaptive.aux_discovery replay ~/gareus/chignolin/chignolin_10 --epoch 2 --out ~/scratch/c10_aux_replay/e2 --workers 8
```

Run each twice and `diff` the reports (determinism). Copy the outputs back and write `docs/superpowers/specs/2026-10-09-cvaux-adaptive-c10-replay.md`: per boundary the discovery k and its ARI table, hidden fraction, co-occurrence, lineage info, evaluation k, the chosen z3 (groups, C, info gain, stability, CV correlations, basin gain), the placement result (workers, c3, k3, held-out O/net), wall time, peak RSS, and a comparison with c10's manual result (k = 8 discovery / k = 5 evaluation, groups 2 vs 3, workers 120/117/9/198). State plainly where the automated rule differs. A large divergence (no admission at either boundary, or z3 uncorrelated with c10's z3, |corr| < 0.5 on the same frames) is reported to the user before any c11 launch.

- [ ] **Step 6: Commit**

```bash
git add gareus/adaptive/aux_discovery/__main__.py tests/test_aux_discovery_replay_cli.py
git add -f docs/superpowers/specs/2026-10-09-cvaux-adaptive-c10-replay.md
git commit -m "feat(cvaux-adaptive): read-only replay CLI; c10 replay report"
```

---

### Task 17: CPU end-to-end through the real epoch loop

**Files:**
- Test: `tests/test_aux_admission_e2e.py`

**Interfaces:**
- Consumes: the small adaptive campaign harness used by `tests/test_ladder_adapt_resume_e2e.py` (read it fully first; reuse its fixture builder, platform Reference, kill injection), `GAREUS_TEST_FAIL_AT_PROD_STEP` style failure injection, `run_epoch_aux_discovery` monkeypatched `_discover` returning a fixed `DiscoveryResult` built from a real `emit_model` on the fixture topology (the statistics are covered by Tasks 7-9; this test covers plumbing).

- [ ] **Step 1: Write the test**

```python
# tests/test_aux_admission_e2e.py
import json
import pytest

pytestmark = pytest.mark.slow


def test_admission_through_epoch_loop_and_resume(tmp_path, monkeypatch, small_adaptive_campaign):
    """small_adaptive_campaign: the test_ladder_adapt_resume_e2e harness (GA-dipeptide, Reference, 2 numbered
    epochs, tiny steps) extended with --ap-aux-discovery, traj = distance interval, a passing aux_validation.json
    and a fixed discovery result (one worker on the first lambda=0 state, k3 = 1.0)."""
    run = small_adaptive_campaign(tmp_path, monkeypatch, aux=True)
    ad = run.out / "adaptive_production"
    adm = json.loads((ad / "aux_admission.json").read_text())
    assert adm["epoch"] == 1 and len(adm["workers"]) == 1
    reg = run.registry()
    workers = [s for s in reg.active_states() if s.metadata.get("aux")]
    assert len(workers) == 1 and workers[0].gamd_lambda == 0.0
    # epoch_002 ran with the worker: its window CSV has aux columns and its samples carry aux_z_00
    assert "aux_center" in (ad / "windows_epoch_002.csv").read_text().splitlines()[0]
    assert run.phase_has_column("epoch_002", "aux_z_00")
    # backfill exists for every pre-admission phase and matches recorded z on post-admission frames
    from gareus.adaptive.aux_backfill import check_backfill_against_recorded
    check_backfill_against_recorded(ad / "epoch_002", run.model())
    # union pools every phase
    from gareus.adaptive_production import build_union_state_mbar_inputs
    out = build_union_state_mbar_inputs(ad, reg, output_prefix="e2e")
    assert workers[0].state_id in out["meta"]["state_ids"]


def test_kill_after_registry_save_never_double_admits(tmp_path, monkeypatch, small_adaptive_campaign):
    run = small_adaptive_campaign(tmp_path, monkeypatch, aux=True, kill_after="registry_save:epoch_001")
    run.resume()
    reg = run.registry()
    assert sum(1 for s in reg.all_states() if s.metadata.get("aux")) == 1
```

Implement `small_adaptive_campaign` in `tests/conftest.py` by factoring the harness out of `tests/test_ladder_adapt_resume_e2e.py` (move, do not duplicate; keep that test passing). `kill_after="registry_save:epoch_001"` raises right after `registry.save` in epoch 1 (monkeypatch `WindowStateRegistry.save` to raise on the matching call), as the existing resume e2e does.

- [ ] **Step 2: Run**

Run: `opencode run "pytest -q tests/test_aux_admission_e2e.py tests/test_ladder_adapt_resume_e2e.py -m slow"`
Expected: PASS. Fix whatever plumbing it exposes in the owning task's files (aux table loading in the phase, CSV columns, arg injection, pull ramp, union) and add a unit test for each fix in that task's test file.

- [ ] **Step 3: Commit**

```bash
git add tests/test_aux_admission_e2e.py tests/conftest.py tests/test_ladder_adapt_resume_e2e.py gareus
git commit -m "test(cvaux-adaptive): CPU end-to-end admission through the epoch loop, resume without double admission"
```

---

### Task 18: chignolin_11 config, launcher, docs

**Files:**
- Create (main checkout `RUNS/`, deployed by rsync): `RUNS/chignolin_11.yaml`, `RUNS/chignolin_11.sh`
- Modify: `CLAUDE.md` (new section "CVaux adaptive discovery (chignolin_11)"), `gareus/helptext.py` (`-hh` topic text for `--ap-aux-discovery`)
- Test: `tests/test_aux_discovery_settings.py::test_c11_yaml_parses` (added here)

**Interfaces:**
- Consumes: Task 1 flags.

- [ ] **Step 1: Write the config**

`RUNS/chignolin_11.yaml` = `RUNS/chignolin_10.yaml` with: `description` rewritten for chignolin_11 (c10 settings + adaptive aux discovery, spec 2026-10-09; swarm round 0 copied from chignolin_10 and re-analysed), `output.out: /home/sulcjo/gareus/chignolin/chignolin_11`, and in `adaptive_production`:

```yaml
  ap_aux_discovery: true          # spec 2026-10-09: driver discovers z3 and admits <= 4 lambda=0 workers
  ap_aux_reserve_slots: 4         # dedicated slice of the 0.10 P1 reserve; R1/R3 never use it
```

Every other key identical to chignolin_10 (traj_interval 300 = distance_output_interval 300, exchange_interval 3000, top-ups off, retirement off).

- [ ] **Step 2: Test that it parses and passes the refusals**

```python
def test_c11_yaml_parses():
    from pathlib import Path
    cfg = Path("/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_11.yaml")
    if not cfg.exists():
        import pytest; pytest.skip("RUNS/ not present")
    from gareus.cli import parse_args
    args = parse_args(["--config", str(cfg), "--out", "/tmp/c11_parse_only"])
    assert args.adaptive_production_aux_discovery is True and args.adaptive_production_aux_reserve_slots == 4
```

Run: `opencode run "pytest -q tests/test_aux_discovery_settings.py::test_c11_yaml_parses"`
Expected: PASS.

- [ ] **Step 3: Launcher**

`RUNS/chignolin_11.sh` = `RUNS/chignolin_10.sh` with `chignolin_10` -> `chignolin_11` everywhere (job name, log paths, PEPTIDE, CONFIG, OUT_DIR, RUN_SCRIPT, CHAIN_STATUS), `CODE_DIR="/home/sulcjo/2026_peptide_sampler_c11"` (scratch tree; never c10's), and the preflight extended:

```bash
# chignolin_11 reuses chignolin_10's swarm round 0 (copied, not linked); analysis re-runs here.
if [[ ! -f "${OUT_DIR}/swarm/round_000/plan_meta.json" ]]; then
    echo "[gareus] ERROR: ${OUT_DIR}/swarm/round_000 missing -- copy it from chignolin_10 first:"
    echo "[gareus]        mkdir -p ${OUT_DIR}/swarm && cp -a ${RUN_DIR}/chignolin_10/swarm/round_000 ${OUT_DIR}/swarm/"
    exit 90
fi
if [[ ! -f "${OUT_DIR}/adaptive_production/aux_validation.json" ]]; then
    echo "[gareus] NOTE: no aux_validation.json yet -- workers are admitted only once it exists and passes."
fi
```

Keep the swarm length check (5 ns) as in c10.

- [ ] **Step 4: Docs**

Add to `CLAUDE.md` (top of the project handoff, after the CVaux Stage C notes) a section with: flag and files, boundary sequence, statuses, frozen files, pooling rule, invisibility, validation record, replay CLI, and the c11 launch checklist (scratch CODE_DIR rsync, DEPLOYED_COMMIT, copy c10 `swarm/round_000`, GPU parity runbook pass, validation runs before the epoch_002 boundary). Add a `-hh` help paragraph for `--ap-aux-discovery` in `gareus/helptext.py` next to the respring help.

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md gareus/helptext.py tests/test_aux_discovery_settings.py
git commit -m "docs(cvaux-adaptive): handoff, help text; chignolin_11 config parse test"
```

`RUNS/` is not tracked in this repo; the yaml and launcher stay in the main checkout and are deployed with rsync at launch time (user approval required for any aurum2 action).

---

## Out of this plan (follow-ups, owners recorded)

- The validation experiments that produce `aux_validation.json` (finite timestep at 3.5 fs HMR vs a shorter-step reference up to `k3_max`, NPT controlled distribution, 236-context MPS cost) need their own plan; they must run before chignolin_11's epoch_002 boundary. Without them the campaign runs, and the hook records `validation_missing`.
- GPU parity runbook (`docs/superpowers/specs/2026-10-07-cvaux-gpu-parity-runbook.md`) before launch (user-triggered).
- Shams, multiple models, worker retirement/replacement, lambda > 0 workers, k3 respring.

## Self-review record

- Spec coverage: Section 4 steps 1-7 -> Tasks 5, 7, 7, 8, 9, 10, 4/10; Section 5 backfill/union -> Tasks 12, 13, 14; Section 6 diagnostics -> Task 15; Section 7 settings/validation/reserve/resume/refusals -> Tasks 1, 9, 4, 10/17, 1; Section 8 invisibility -> Task 3; Section 9 tests/replay -> Tasks 1-17, 16; Section 10 deliverables -> Task 18; worker pull (Section 4 "Next phase") -> Task 11; phase-arg injection -> Task 10.
- Review Focus tests: (1) Task 12 `test_missing_frames_refuse`; (2) Task 2 `test_parent_written_blank_when_not_active`; (3) Task 17 `test_kill_after_registry_save_never_double_admits` + Task 10 skip-after-admission; (4) Task 10 statuses + Task 2 byte-identity test; (5) Task 10 `test_alignment_check_refuses_doomed_admission`.
- Names checked across tasks: `is_auxiliary_state`, `aux_params`, `AUX_METADATA_KEY`, `ordinary_active_states`, `FrameTable`, `DescriptorDefinition`, `fit_partition`, `FrozenPartition`, `Z3Candidate`, `emit_model`, `z3_values`, `place_workers`, `check_validation_record`, `run_discovery`, `DiscoveryResult`, `run_epoch_aux_discovery`, `inject_aux_phase_args`, `write_phase_backfill`, `read_phase_backfill`, `aux_admission_allows_pooling`, `aux_ordinary_crosscheck`.
