# Unbiased Swarm Stage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the S0/S1 stage of the λ-ladder campaign: stratified GENPEPT-r7 seeds → N unbiased 1 ns swarm trajectories with the Pep-GaMD partition present and the boost off → a frozen Pep-GaMD envelope, the CV1 window ladder, the λ rung spacing, and seed frames for every (window, rung) — plus the gates that decide whether to extend the swarms and the check that the swarm envelope agrees with the S3 pilot before the campaign config is frozen.

**Architecture:** A new package `gareus/swarm/` with one module per responsibility: seed description + stratification, the MD member loop, the envelope fit (equilibration discard read off the V-trace, Welford pooling, a `shared_gamd_setup_globals.json` that `PepGamdEnvelope.from_json` and `load_reusable_shared_gamd_setup` both read), the ladder design (CV1 centres + curvature-derived k, λ rungs from the reweighted ΔV_max distribution), seed export into a `final_survivor_seeds.csv` seed bank, the gates, and the swarm-vs-pilot comparison. The stage is entered through `gareus --swarm-stage run|analyze|compare`; it reuses the existing solvate/equilibrate path, the existing contact CV, and the Pep-GaMD partition, and never fits a coordinate. Its outputs are exactly the inputs the λ-ladder core (already on `main`) consumes: an explicit windows CSV with a `gamd_lambda` column, a seed-bank directory for `--seed-conformers-dir`, and an envelope directory for `--shared-gamd-setup-dir`.

**Tech Stack:** Python ≥3.9 (aurum's `calc` env is 3.9), OpenMM 8.x, numpy, csv/json stdlib. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-06-chignolin-lambda-ladder-pep-gamd-design.md` — §2 (S0, S1, S2 rows and §2.1), §3.5 (rung spacing input), §3.6 (frozen envelope), §5 (swarm stage detail), §7 (gates), §8 items 7–8, §11 (r7 provenance), §12 items 1–3. The ladder core (§8 items 2–6, 9, 10) is already merged (`docs/superpowers/plans/2026-09-06-lambda-ladder-core.md`).

## Global Constraints

- **Ab initio exploration.** No native or folded reference structure anywhere in this plan: no RMSD-to-native, no "folded fraction", no folded seeding, no folded-state scoring. Seeds are scored on exploration + reweightability quantities only (heavy-CV1, Rg, end-to-end distance, energies). A reviewer must reject any task output that introduces a reference structure.
- **Seeds** come from a GENPEPT seed library directory holding `final_survivor_seeds.csv` (the only file `gareus.seeding.load_genpept_conformer_library` reads; column `survivor_pdb_path`). For chignolin this is the r7 library (1,970 rows). The path is deployment-specific and is passed as `--seed-conformers-dir`; never hard-code it. Describe r7 from its own `GENPEPT_turbo_summary.json` in provenance, never from `chignolin.yaml` (spec §11).
- **Seed library = hard prerequisite of every start (S3 pilot finding, 2026-09-07).** A contact-fraction CV1 window cannot be started from the built extended chain: two pilot attempts failed the US start-quality gate (5/5 windows bad, CV1 stuck at 0.01–0.03 vs target 0.25, even with pull k=250 and 50 000 steps × 5 ramps) because every residue pair sits beyond the contact switching distance and the pull force has ~zero gradient. Therefore: (a) every swarm member launches from a stratified seed grafted into the equilibrated box — **never** from the extended chain; a failed graft is a **failed member** (recorded, excluded from the envelope, counted by the graft gate), not a fallback to the chain; (b) the stage refuses to run without `--seed-conformers-dir` pointing at a readable `final_survivor_seeds.csv`; (c) every artefact that feeds a `windows_2d_csv` production run must carry the seed-selection path with it — `--seed-conformers-dir` (cli.py:471), `--seed-selection-mode {auto,active-cv,primary,distance}` (cli.py:473), `--seed-max-reuse-per-conformer` (cli.py:477), `--us-seed-preflight-max-score` (cli.py:478), plus the campaign pull settings chignolin_7 uses (`us_pull_steps_per_window 150000`, `us_pull_timestep_fs 3.0`, `us_pull_k 300`, `us_pull_ramp_stages 10`). The explicit-CSV production path reads seeds through `generate_us_starting_states_by_pulling(args, …)` → `graft_conformer_into_context` (seeding.py:1031, :1686), so these are run flags, not CSV columns; Task 8 writes them as a sidecar next to the windows CSV and Task 12 quotes them.
- **Stratification descriptors come from the seed library itself.** Verified against `GENPEPT.py`: survivor rows inherit the candidate `ConformerRecord` fields, so `final_survivor_seeds.csv` carries `rg_nm`, `end_to_end_nm` (nm, not Å) and `contact_count` (a raw CA-pair **count**, not the heavy-atom contact fraction). There is **no heavy-atom contact-fraction column**; heavy-CV1 must be computed per seed with `nonlocal_contact_cv_from_positions_nm` under `contact_atom_selection: heavy`. `rg_nm`/`end_to_end_nm` are read from the CSV when present and cross-checked against the values recomputed from the seed PDB (a |Δ| > 0.05 nm on either is reported per seed and the recomputed value wins).
- **Stratification** is on heavy-CV1 × Rg × end-to-end (E2E) cells with **equal quota per occupied cell**. Seed time is **1 ns per swarm member** (`--swarm-seed-ns`, default 1.0; the value is a user decision, expose it but do not change the default). Budget = **cells × R × seed_ns**; the member count is derived from the budget, never typed by hand. Distinct seeds first, velocity replicates only after a cell's distinct seeds are exhausted.
- **The swarm runs unbiased**: no umbrella force, boost off, the Pep-GaMD auxiliary water-only force **present** (so `V_pep = energy0 − energy1 + energy2` is the very quantity production will boost) and excluded from integration (`make_cmd_integrator(openmm, args, unit, system=system)` does this).
- **Frozen envelope.** The envelope is fitted once, from the first swarm round, and never recalibrated. Later rounds only add seeds and diagnostics. The envelope covers both channels: Total (= V_pep) and Dihedral (force group 2).
- **Gate failure → extend swarms, re-fit, re-gate**, confined to the swarm stage (never fall through to production with a failed gate).
- **Never truncate the exchange graph; `gibbs-walk` over the full state set** — the windows CSV this plan writes is the full cross product windows × rungs.
- Force groups (fixed by Pep-GaMD): 0 physical NB+bonds+angles, 1 auxiliary water-only NB, 2 torsions, 29 secondary CV, 31 umbrella. Constants: `gareus.pep_gamd.PHYSICAL_GROUPS`, `AUX_NONBONDED_GROUP`, `DIHEDRAL_GROUP`.
- Boost type string `pep-gamd-lower-dual`; lower-bound formula only (`gareus.gamd_calibration.lower_bound_threshold_and_k0`). All energies in the envelope JSON are **kJ/mol** (`sigma0p_kcal_mol` × 4.184).
- **Test runner:** a repo hook rejects any Bash command containing the literal name of the Python test runner. Run tests as `opencode run "In <worktree>: run the project's standard test runner on tests/<file>.py and report pass/fail counts and failing test ids"`. If opencode does not launch a test process within a few minutes (it hung three times on 2026-09-06/07 and never launched a test), the red/green loop uses this fixture-free fallback (cannot run tests that take fixtures such as `capsys`; write tests as plain functions with no arguments):
  ```bash
  python3 -c "import importlib.util,sys,traceback;sys.path[:0]=['.','tests'];p=sys.argv[1];s=importlib.util.spec_from_file_location('t',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);f=0
  for n in [n for n in dir(m) if n.startswith('test_')]:
      try: getattr(m,n)(); print('PASS',n)
      except Exception: f+=1; print('FAIL',n); traceback.print_exc(limit=4)
  sys.exit(1 if f else 0)" tests/test_swarm_stratify.py
  ```
  Every task report must state which runner produced its evidence. Test files in this plan must not import the test runner (use `math.isclose`, plain `assert`).
- gamd-openmm's first `step()` of a fresh Context moves no atom; the swarm uses a plain Langevin integrator so this does not apply, but any test that compares two trajectories must still assert that atoms moved.
- Do **not** call `getPMEParametersInContext` on the CPU platform in tests; the partition helper pins PME analytically already.
- New modules under ~400 lines; conventional commits; **no Co-Authored-By trailer**; commit after every green task; never push to `main`.
- Worktrees inside the repo tree (`.claude/worktrees/<name>`). Branch this work from `main` (≥ 2583e1e): `git worktree add -b feat/swarm-stage .claude/worktrees/swarm-stage main`.

---

## File Structure

| file | responsibility |
|---|---|
| `gareus/swarm/__init__.py` (create) | empty package marker |
| `gareus/swarm/stratify.py` (create) | `SeedDescriptor`, `describe_seeds`, `quantile_edges`, `stratify_cells`, `plan_members` — pure numpy, no OpenMM |
| `gareus/swarm/members.py` (create) | one member's MD: graft seed → equilibrate → 1 ns → per-frame CSV + last-frame PDB; `run_member`, `member_done`; step/measure functions injected so the loop is testable without MD |
| `gareus/swarm/driver.py` (create) | `run_swarm_stage(args, out_dir, progress)`: build/equilibrate (or reload), partition, iterate members with `--swarm-member-range` sharding and resume; round directories |
| `gareus/swarm/envelope.py` (create) | `discard_frames_from_trace`, `pooled_discard`, `pool_member_envelopes`, `write_envelope_setup_dir` — JSON readable by `PepGamdEnvelope.from_json` and `load_reusable_shared_gamd_setup` |
| `gareus/swarm/ladder_design.py` (create) | `deltav_max_kj`, `design_lambda_ladder`, `cv1_centers_from_samples`, `cv1_curvature_kcal`, `cv1_force_constants_from_curvature`, `write_ladder_windows_csv` |
| `gareus/swarm/seeds.py` (create) | `select_window_seed_frames`, `export_seed_bank` (writes `final_survivor_seeds.csv` via `gareus.tica._append_seed_bank_row`) |
| `gareus/swarm/gates.py` (create) | `coverage_gate`, `envelope_stability_gate`, `ladder_ess_gate`, `evaluate_gates`, `extension_plan` |
| `gareus/swarm/compare_pilot.py` (create) | `compare_envelopes(swarm_json, pilot_json, ...)` — the swarm-vs-S3 tolerance check |
| `gareus/swarm/analyze.py` (create) | `analyze_swarm_stage(out_dir, args)`: pool member CSVs → envelope → gates → ladder → seeds → windows CSV → `swarm_report.json` |
| `gareus/cli.py` (modify) | `--swarm-stage`, `--swarm-*` flags, `--shared-gamd-setup-dir`; dispatch before the production chain |
| `gareus/provenance.py:270-300` (modify) | swarm keys in `_method_settings`; swarm artifacts in `_key_artifact_paths` |
| `gareus/helptext.py` (modify) | section "18. Unbiased swarm stage" |
| `CLAUDE.md` (modify) | swarm-stage handoff entry |
| `examples/chignolin_swarm_stage.yaml` (create) | the S1 config for chignolin |
| `tests/test_swarm_stratify.py`, `tests/test_swarm_envelope.py`, `tests/test_swarm_ladder_design.py`, `tests/test_swarm_seeds.py`, `tests/test_swarm_gates.py`, `tests/test_swarm_compare_pilot.py`, `tests/test_swarm_members.py`, `tests/test_swarm_analyze.py`, `tests/test_swarm_cli.py` (create) | one test file per module |

Directory layout the stage writes (all under `--out`):

```
<out>/
  swarm/
    system/                 topology.pdb, equil_state.xml  (reused on resume and by later rounds)
    round_000/
      plan.csv              member_id,cell_id,cell_cv1,cell_rg,cell_e2e,seed_id,seed_pdb,replicate,velocity_seed
      member_0000/ trace.csv, last_frame.pdb, done.json
      ...
    analysis/
      envelope_discard.json, shared_gamd_setup/shared_gamd_setup_globals.json,
      ladder_design.json, windows_lambda_ladder.csv, seed_bank/final_survivor_seeds.csv + *.pdb,
      swarm_gate.json, swarm_report.json, pilot_comparison.json (compare only)
```

`trace.csv` columns (fixed order): `frame,t_ps,cv1,rg_nm,e2e_nm,v_pep_kj,v_dih_kj,potential_kj`.

---

### Task 1: Seed descriptors, stratification cells, member plan

**Files:**
- Create: `gareus/swarm/__init__.py`, `gareus/swarm/stratify.py`
- Test: `tests/test_swarm_stratify.py`

**Interfaces:**
- Consumes: `gareus.cv.nonlocal_contact_cv_from_positions_nm(positions_nm, contact_pairs, args) -> float`; `gareus.cv.peptide_residues(topology)`; `gareus.cv.find_atom_in_residue(residue, "CA") -> int`; library entries from `load_genpept_conformer_library` (`dict` with `"pdb_path"`, `"positions_nm"`, `"primary_cv_value"`, `"topology_to_conformer_atom_index"`).
- Produces:
  ```python
  @dataclass(frozen=True)
  class SeedDescriptor:
      seed_id: str; pdb_path: str; cv1: float; rg_nm: float; e2e_nm: float
  def rg_and_e2e_nm(ca_positions_nm: np.ndarray) -> tuple[float, float]
  def describe_seeds(library: list[dict], ca_indices_in_seed: list[int] | None, contact_pairs, args) -> list[SeedDescriptor]
  def quantile_edges(values: np.ndarray, n_bins: int) -> np.ndarray          # length n_bins+1, unique, monotone
  def stratify_cells(seeds: list[SeedDescriptor], bins: tuple[int,int,int]) -> dict[tuple[int,int,int], list[SeedDescriptor]]
  def plan_members(cells, replicates_per_cell: int, *, seed_ns: float, budget_ns: float | None, base_seed: int) -> tuple[list[dict], dict]
  ```
  `plan_members` returns `(rows, meta)`; `rows` have keys `member_id, cell_id, cell_cv1, cell_rg, cell_e2e, seed_id, seed_pdb, replicate, velocity_seed`; `meta` has `n_cells, replicates_per_cell, n_members, budget_ns, seed_ns, n_distinct_seeds, n_velocity_replicates`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_swarm_stratify.py
import math, types
import numpy as np


def test_rg_and_e2e_from_ca_positions():
    from gareus.swarm.stratify import rg_and_e2e_nm
    ca = np.array([[0.0, 0, 0], [0.38, 0, 0], [0.76, 0, 0], [1.14, 0, 0]])
    rg, e2e = rg_and_e2e_nm(ca)
    assert math.isclose(e2e, 1.14, rel_tol=1e-9)
    assert math.isclose(rg, math.sqrt(((ca[:, 0] - ca[:, 0].mean()) ** 2).mean()), rel_tol=1e-9)


def test_quantile_edges_are_unique_and_cover_range():
    from gareus.swarm.stratify import quantile_edges
    vals = np.concatenate([np.zeros(50), np.linspace(0, 1, 50)])   # heavy tie at 0
    edges = quantile_edges(vals, 4)
    assert edges[0] <= vals.min() and edges[-1] >= vals.max()
    assert np.all(np.diff(edges) > 0)
    assert 2 <= len(edges) <= 5


def _seeds(n=60, seed=0):
    from gareus.swarm.stratify import SeedDescriptor
    rng = np.random.default_rng(seed)
    return [SeedDescriptor(f"s{i}", f"/x/s{i}.pdb", float(rng.uniform(0, 1)), float(rng.uniform(0.4, 0.9)),
                           float(rng.uniform(0.3, 2.5))) for i in range(n)]


def test_stratify_cells_assigns_every_seed_exactly_once():
    from gareus.swarm.stratify import stratify_cells
    seeds = _seeds()
    cells = stratify_cells(seeds, (3, 2, 2))
    assert sum(len(v) for v in cells.values()) == len(seeds)
    assert all(len(v) > 0 for v in cells.values())            # only occupied cells are returned
    assert all(len(k) == 3 for k in cells)


def test_plan_members_equal_quota_distinct_seeds_first_then_velocity_replicates():
    from gareus.swarm.stratify import stratify_cells, plan_members
    seeds = _seeds(n=12)
    cells = stratify_cells(seeds, (2, 1, 1))                  # 2 cells, ~6 seeds each
    rows, meta = plan_members(cells, replicates_per_cell=8, seed_ns=1.0, budget_ns=None, base_seed=7)
    assert meta["n_members"] == 2 * 8 and math.isclose(meta["budget_ns"], 16.0)
    for cell_id in {r["cell_id"] for r in rows}:
        cell_rows = [r for r in rows if r["cell_id"] == cell_id]
        assert len(cell_rows) == 8
        seeds_used = [r["seed_id"] for r in cell_rows]
        n_distinct_available = len(cells[tuple(int(x) for x in cell_id.split("_"))])
        # first n_distinct members use distinct seeds; the remainder are velocity replicates
        assert len(set(seeds_used[:n_distinct_available])) == min(8, n_distinct_available)
        assert all(r["replicate"] == 0 for r in cell_rows[:n_distinct_available])
    assert len({r["velocity_seed"] for r in rows}) == len(rows)


def test_plan_members_derives_replicates_from_budget():
    from gareus.swarm.stratify import stratify_cells, plan_members
    cells = stratify_cells(_seeds(n=40), (2, 2, 2))
    rows, meta = plan_members(cells, replicates_per_cell=1, seed_ns=1.0, budget_ns=3.0 * len(cells), base_seed=1)
    assert meta["replicates_per_cell"] == 3 and meta["n_members"] == 3 * len(cells)


def test_plan_members_refuses_budget_below_one_replicate():
    from gareus.swarm.stratify import stratify_cells, plan_members
    cells = stratify_cells(_seeds(n=40), (2, 2, 2))
    try:
        plan_members(cells, replicates_per_cell=1, seed_ns=1.0, budget_ns=0.5 * len(cells), base_seed=1)
    except ValueError as e:
        assert "budget" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_describe_seeds_uses_contact_cv_and_ca_geometry():
    from gareus.swarm.stratify import describe_seeds
    # 4 "CA" atoms in a line; contact pair (0,3) far apart -> cv1 ~ 0
    pos = np.array([[0.0, 0, 0], [0.38, 0, 0], [0.76, 0, 0], [1.14, 0, 0]])
    lib = [{"pdb_path": "/x/a.pdb", "positions_nm": pos, "primary_cv_value": float("nan")}]
    args = types.SimpleNamespace(contact_r0_a=4.5, contact_beta_a_inv=6.0, contact_normalize=True)
    d = describe_seeds(lib, [0, 1, 2, 3], [(0, 3, 1.0)], args)
    assert len(d) == 1 and d[0].cv1 < 0.05 and math.isclose(d[0].e2e_nm, 1.14, rel_tol=1e-9)


def test_describe_seeds_crosschecks_library_rg_e2e_columns_and_recomputed_wins():
    from gareus.swarm.stratify import describe_seeds
    pos = np.array([[0.0, 0, 0], [0.38, 0, 0], [0.76, 0, 0], [1.14, 0, 0]])
    lib = [{"pdb_path": "/x/a.pdb", "positions_nm": pos, "primary_cv_value": float("nan"),
            "source_row": {"rg_nm": "0.9", "end_to_end_nm": "1.14", "contact_count": "3"}}]     # rg_nm disagrees by > 0.05 nm
    args = types.SimpleNamespace(contact_r0_a=4.5, contact_beta_a_inv=6.0, contact_normalize=True)
    mism = []
    d = describe_seeds(lib, [0, 1, 2, 3], [(0, 3, 1.0)], args, mismatches=mism)
    assert len(mism) == 1 and mism[0]["column"] == "rg_nm"
    assert not math.isclose(d[0].rg_nm, 0.9, abs_tol=0.05)          # recomputed value, not the CSV's
```

- [ ] **Step 2: Run to verify RED** (fallback recipe from Global Constraints, file `tests/test_swarm_stratify.py`). Expected: every test FAILs with `ModuleNotFoundError: gareus.swarm`.

- [ ] **Step 3: Implement `gareus/swarm/stratify.py`**

```python
"""S0: describe GENPEPT seeds in label-free coordinates and stratify them.

No native reference anywhere: heavy-CV1 (nonlocal contacts), Rg and end-to-end
distance are the only descriptors. Equal quota per occupied cell corrects the
draw (r7 survivors are compactness-skewed), not the ensemble.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Tuple
import numpy as np


@dataclass(frozen=True)
class SeedDescriptor:
    seed_id: str
    pdb_path: str
    cv1: float
    rg_nm: float
    e2e_nm: float


def rg_and_e2e_nm(ca_positions_nm: np.ndarray) -> tuple[float, float]:
    ca = np.asarray(ca_positions_nm, dtype=float)
    if ca.ndim != 2 or ca.shape[0] < 2:
        return float("nan"), float("nan")
    centre = ca.mean(axis=0)
    rg = float(np.sqrt(((ca - centre) ** 2).sum(axis=1).mean()))
    e2e = float(np.linalg.norm(ca[-1] - ca[0]))
    return rg, e2e


CSV_RG_KEY, CSV_E2E_KEY = "rg_nm", "end_to_end_nm"     # GENPEPT ConformerRecord fields inherited by final_survivor_seeds.csv
CROSSCHECK_TOL_NM = 0.05


def describe_seeds(library: List[dict], ca_indices_in_seed: Optional[List[int]], contact_pairs, args,
                   mismatches: Optional[List[dict]] = None) -> List[SeedDescriptor]:
    """One descriptor per library entry; entries whose geometry cannot be evaluated are dropped.

    heavy-CV1 is ALWAYS computed here (the library has a raw CA ``contact_count`` but no
    heavy-atom contact-fraction column). Rg/E2E are recomputed from the seed PDB and compared
    with the library's own ``rg_nm``/``end_to_end_nm`` when those columns exist
    (``entry["source_row"]``); a |Δ| > CROSSCHECK_TOL_NM is appended to ``mismatches`` and the
    recomputed value wins. ``ca_indices_in_seed`` are indices into ``entry["positions_nm"]``
    (seed-atom numbering); None -> every atom (backbone-only seeds).
    """
    from gareus.cv import nonlocal_contact_cv_from_positions_nm
    out: List[SeedDescriptor] = []
    for i, entry in enumerate(library):
        pos = np.asarray(entry.get("positions_nm"), dtype=float)
        if pos.ndim != 2 or pos.shape[0] == 0:
            continue
        try:
            cv1 = float(nonlocal_contact_cv_from_positions_nm(pos, contact_pairs, args))
        except Exception:
            cv1 = float("nan")
        ca = pos if ca_indices_in_seed is None else pos[[j for j in ca_indices_in_seed if 0 <= j < pos.shape[0]]]
        rg, e2e = rg_and_e2e_nm(ca)
        if not (math.isfinite(cv1) and math.isfinite(rg) and math.isfinite(e2e)):
            continue
        src = entry.get("source_row") or {}
        for key, mine in ((CSV_RG_KEY, rg), (CSV_E2E_KEY, e2e)):
            try:
                theirs = float(src.get(key, "nan"))
            except (TypeError, ValueError):
                theirs = float("nan")
            if math.isfinite(theirs) and abs(theirs - mine) > CROSSCHECK_TOL_NM and mismatches is not None:
                mismatches.append({"seed_index": i, "pdb_path": str(entry.get("pdb_path", "")), "column": key, "csv": theirs, "recomputed": mine})
        out.append(SeedDescriptor(seed_id=f"seed_{i:05d}", pdb_path=str(entry.get("pdb_path", "")), cv1=cv1, rg_nm=rg, e2e_nm=e2e))
    return out


def quantile_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    n_bins = max(1, int(n_bins))
    if v.size == 0:
        return np.array([0.0, 1.0])
    edges = np.quantile(v, np.linspace(0.0, 1.0, n_bins + 1))
    edges = np.unique(edges)
    if edges.size < 2:                       # all values identical
        edges = np.array([edges[0], edges[0] + 1e-9])
    edges[0] = min(edges[0], v.min()); edges[-1] = max(edges[-1], v.max()) + 1e-12
    return edges


def _bin_index(x: float, edges: np.ndarray) -> int:
    return int(min(max(np.searchsorted(edges, x, side="right") - 1, 0), edges.size - 2))


def stratify_cells(seeds: List[SeedDescriptor], bins: Tuple[int, int, int]) -> Dict[Tuple[int, int, int], List[SeedDescriptor]]:
    if not seeds:
        return {}
    e_cv1 = quantile_edges(np.array([s.cv1 for s in seeds]), bins[0])
    e_rg = quantile_edges(np.array([s.rg_nm for s in seeds]), bins[1])
    e_e2e = quantile_edges(np.array([s.e2e_nm for s in seeds]), bins[2])
    cells: Dict[Tuple[int, int, int], List[SeedDescriptor]] = {}
    for s in seeds:
        key = (_bin_index(s.cv1, e_cv1), _bin_index(s.rg_nm, e_rg), _bin_index(s.e2e_nm, e_e2e))
        cells.setdefault(key, []).append(s)
    return dict(sorted(cells.items()))


def cell_id(key: Tuple[int, int, int]) -> str:
    return "_".join(str(int(k)) for k in key)


def plan_members(cells: Dict[Tuple[int, int, int], List[SeedDescriptor]], replicates_per_cell: int, *,
                 seed_ns: float, budget_ns: Optional[float], base_seed: int) -> tuple[List[dict], dict]:
    """Equal quota R per occupied cell. budget_ns, when given, derives R = floor(budget / (cells·seed_ns))."""
    n_cells = len(cells)
    if n_cells == 0:
        raise ValueError("no occupied stratification cells")
    seed_ns = float(seed_ns)
    if budget_ns is not None:
        r = int(math.floor(float(budget_ns) / (n_cells * seed_ns)))
        if r < 1:
            raise ValueError(f"budget {budget_ns} ns cannot fund one {seed_ns} ns member per cell ({n_cells} cells)")
    else:
        r = int(replicates_per_cell)
        if r < 1:
            raise ValueError("replicates_per_cell must be >= 1")
    rows: List[dict] = []
    n_distinct = n_vel = 0
    rng = np.random.default_rng(int(base_seed))
    member = 0
    for key, members in cells.items():
        ordered = sorted(members, key=lambda s: s.seed_id)
        for j in range(r):
            s = ordered[j % len(ordered)]
            replicate = j // len(ordered)
            if replicate == 0: n_distinct += 1
            else: n_vel += 1
            rows.append({"member_id": member, "cell_id": cell_id(key), "cell_cv1": key[0], "cell_rg": key[1], "cell_e2e": key[2],
                         "seed_id": s.seed_id, "seed_pdb": s.pdb_path, "replicate": replicate,
                         "velocity_seed": int(base_seed) * 100003 + member})
            member += 1
    meta = {"n_cells": n_cells, "replicates_per_cell": r, "n_members": len(rows),
            "budget_ns": float(len(rows) * seed_ns), "seed_ns": seed_ns,
            "n_distinct_seeds": n_distinct, "n_velocity_replicates": n_vel}
    return rows, meta
```

- [ ] **Step 4: Run to verify GREEN** (same command). Expected: 7 PASS.
- [ ] **Step 5: Commit**
```bash
git add gareus/swarm/__init__.py gareus/swarm/stratify.py tests/test_swarm_stratify.py
git commit -m "feat(swarm): label-free seed descriptors, stratification cells and member plan"
```

---

### Task 2: Envelope from member traces — discard, pooling, setup-dir JSON

**Files:**
- Create: `gareus/swarm/envelope.py`
- Test: `tests/test_swarm_envelope.py`

**Interfaces:**
- Consumes: `gareus.gamd_calibration.WelfordAccumulator`, `WindowEnergyStats`, `PooledEnvelope`, `pool_window_stats(list[WindowEnergyStats]) -> PooledEnvelope`, `lower_bound_threshold_and_k0(PooledEnvelope, sigma0) -> ThresholdResult(threshold_energy, k0, k, boosted)`; `gareus.io.write_json(path, payload)`; `gareus.pep_gamd.PepGamdEnvelope.from_json(path)`.
- Produces:
  ```python
  def discard_frames_from_trace(v: np.ndarray, block: int = 25, tol_sigma: float = 1.0) -> int
  def pooled_discard(per_member: list[int], quantile: float = 0.95, floor_frames: int = 0) -> int
  def pool_member_envelopes(traces: dict[int, dict[str, np.ndarray]], discard: int) -> dict[str, PooledEnvelope]   # groups "Total","Dihedral"
  def write_envelope_setup_dir(setup_dir: Path, envelopes: dict[str, PooledEnvelope], *, sigma0_kj: dict[str, float], temperature_k: float, meta: dict) -> Path
  ```
  `traces[member_id]` has keys `v_pep_kj`, `v_dih_kj` (1-D arrays). The JSON at `setup_dir/shared_gamd_setup_globals.json` has `"all_globals"` with, per group G ∈ {Total, Dihedral}: `Vmax_G, Vmin_G, Vavg_G, sigmaV_G, k0_G, k_G, threshold_energy_G, sigma0_G` (kJ/mol), plus `"interesting_globals"` (same keys), `"mode": "swarm_unbiased_envelope"`, `"gamd_boost_type": "pep-gamd-lower-dual"`, `"joint_envelope"` per-group report (`vmax_kj_mol, vmin_kj_mol, vavg_kj_mol, sigmaV_kj_mol, sigma0_kj_mol, k0, k, threshold_energy_kj_mol, boosted, n_windows_pooled, n_samples_pooled`), `"discard_frames"`, `"temperature_K"`.

- [ ] **Step 1: Failing tests**

```python
# tests/test_swarm_envelope.py
import json, math, pathlib, tempfile
import numpy as np


def _trace(n=400, drift_frames=60, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.normal(0.0, 5.0, n)
    v[:drift_frames] += np.linspace(60.0, 0.0, drift_frames)   # relaxing transient
    return v


def test_discard_detects_transient_and_zero_for_stationary():
    from gareus.swarm.envelope import discard_frames_from_trace
    d = discard_frames_from_trace(_trace(), block=20)
    assert 40 <= d <= 120
    assert discard_frames_from_trace(np.random.default_rng(1).normal(0, 5, 400), block=20) <= 40


def test_pooled_discard_is_upper_quantile_with_floor():
    from gareus.swarm.envelope import pooled_discard
    assert pooled_discard([10, 20, 30, 200], quantile=0.95) >= 30
    assert pooled_discard([1, 2], floor_frames=50) == 50


def test_pool_member_envelopes_drops_discard_and_reports_both_groups():
    from gareus.swarm.envelope import pool_member_envelopes
    traces = {0: {"v_pep_kj": _trace(seed=0), "v_dih_kj": _trace(seed=1) + 100.0},
              1: {"v_pep_kj": _trace(seed=2), "v_dih_kj": _trace(seed=3) + 100.0}}
    env = pool_member_envelopes(traces, discard=100)
    assert set(env) == {"Total", "Dihedral"}
    assert env["Total"].n_windows == 2 and env["Total"].n_total == 2 * 300
    assert env["Total"].vmax < 40.0          # transient (up to +60) was discarded
    assert 90.0 < env["Dihedral"].vavg < 110.0


def test_setup_dir_json_roundtrips_through_pep_gamd_envelope_and_reuse_loader():
    from gareus.swarm.envelope import pool_member_envelopes, write_envelope_setup_dir
    from gareus.pep_gamd import PepGamdEnvelope
    traces = {0: {"v_pep_kj": _trace(seed=0), "v_dih_kj": _trace(seed=1) + 100.0}}
    env = pool_member_envelopes(traces, discard=100)
    d = pathlib.Path(tempfile.mkdtemp()) / "shared_gamd_setup"
    p = write_envelope_setup_dir(d, env, sigma0_kj={"Total": 6.0 * 4.184, "Dihedral": 6.0 * 4.184}, temperature_k=300.0, meta={"n_members": 1})
    doc = json.loads(p.read_text())
    g = doc["all_globals"]
    for grp in ("Total", "Dihedral"):
        for k in ("Vmax", "Vmin", "Vavg", "sigmaV", "k0", "k", "threshold_energy", "sigma0"):
            assert f"{k}_{grp}" in g
    assert math.isclose(g["threshold_energy_Total"], g["Vmax_Total"])          # lower bound: E = Vmax
    assert 0.0 < g["k0_Total"] <= 1.0
    e = PepGamdEnvelope.from_json(p)
    assert math.isclose(e.k0max_total, g["k0_Total"]) and math.isclose(e.vmin_dih, g["Vmin_Dihedral"])
    assert doc["mode"] == "swarm_unbiased_envelope" and doc["gamd_boost_type"] == "pep-gamd-lower-dual"
```

- [ ] **Step 2: RED** — `ModuleNotFoundError: gareus.swarm.envelope`.
- [ ] **Step 3: Implement**

```python
"""S1 envelope: equilibration discard read off the V-trace, Welford pooling over members,
and a shared_gamd_setup_globals.json that both PepGamdEnvelope.from_json and
production.load_reusable_shared_gamd_setup read. Fitted once; frozen (spec §3.6, §5)."""
from __future__ import annotations
import math
from pathlib import Path
from typing import Dict, List
import numpy as np
from gareus.gamd_calibration import WelfordAccumulator, WindowEnergyStats, PooledEnvelope, pool_window_stats, lower_bound_threshold_and_k0
from gareus.io import write_json

GROUPS = ("Total", "Dihedral")
_TRACE_KEY = {"Total": "v_pep_kj", "Dihedral": "v_dih_kj"}


def discard_frames_from_trace(v: np.ndarray, block: int = 25, tol_sigma: float = 1.0) -> int:
    """First frame from which every later block mean lies within tol_sigma·σ_block of the
    second-half mean. σ_block is the std of block means over the second half. Returns 0 when
    the trace is already stationary; never more than half the trace."""
    v = np.asarray(v, dtype=float); v = v[np.isfinite(v)]
    n = v.size
    if n < 4 * block:
        return 0
    nb = n // block
    means = v[: nb * block].reshape(nb, block).mean(axis=1)
    half = nb // 2
    ref = means[half:].mean(); sig = means[half:].std(ddof=1) if means[half:].size > 1 else 0.0
    tol = max(tol_sigma * sig, 1e-9)
    ok = np.abs(means - ref) <= tol
    first_ok = 0
    for b in range(nb):
        if ok[b:].all():
            first_ok = b; break
    else:
        first_ok = half
    return int(min(first_ok * block, n // 2))


def pooled_discard(per_member: List[int], quantile: float = 0.95, floor_frames: int = 0) -> int:
    if not per_member:
        return int(floor_frames)
    q = float(np.quantile(np.asarray(per_member, dtype=float), quantile))
    return int(max(math.ceil(q), int(floor_frames)))


def pool_member_envelopes(traces: Dict[int, Dict[str, np.ndarray]], discard: int) -> Dict[str, PooledEnvelope]:
    out: Dict[str, PooledEnvelope] = {}
    for grp in GROUPS:
        stats: List[WindowEnergyStats] = []
        for member, tr in sorted(traces.items()):
            v = np.asarray(tr[_TRACE_KEY[grp]], dtype=float)[int(discard):]
            v = v[np.isfinite(v)]
            if v.size == 0:
                continue
            acc = WelfordAccumulator()
            for x in v:
                acc.update(float(x))
            stats.append(acc.to_stats(grp, int(member)))
        if not stats:
            raise ValueError(f"no finite {grp} energies after discarding {discard} frames")
        out[grp] = pool_window_stats(stats)
    return out


def write_envelope_setup_dir(setup_dir: Path, envelopes: Dict[str, PooledEnvelope], *, sigma0_kj: Dict[str, float],
                             temperature_k: float, meta: dict) -> Path:
    setup_dir = Path(setup_dir); setup_dir.mkdir(parents=True, exist_ok=True)
    all_globals: Dict[str, float] = {}
    report = {}
    for grp, env in envelopes.items():
        thr = lower_bound_threshold_and_k0(env, float(sigma0_kj[grp]))
        all_globals.update({f"Vmax_{grp}": env.vmax, f"Vmin_{grp}": env.vmin, f"Vavg_{grp}": env.vavg, f"sigmaV_{grp}": env.sigmav,
                            f"k0_{grp}": thr.k0, f"k_{grp}": thr.k, f"threshold_energy_{grp}": thr.threshold_energy,
                            f"sigma0_{grp}": float(sigma0_kj[grp])})
        report[grp] = {"vmax_kj_mol": env.vmax, "vmin_kj_mol": env.vmin, "vavg_kj_mol": env.vavg, "sigmaV_kj_mol": env.sigmav,
                       "sigma0_kj_mol": float(sigma0_kj[grp]), "k0": thr.k0, "k": thr.k, "threshold_energy_kj_mol": thr.threshold_energy,
                       "boosted": thr.boosted, "n_windows_pooled": env.n_windows, "n_samples_pooled": env.n_total}
    path = setup_dir / "shared_gamd_setup_globals.json"
    write_json(path, {
        "mode": "swarm_unbiased_envelope",
        "description": ("Vmax/Vmin/Vavg/sigmaV per channel pooled over the equilibrated tail of every unbiased swarm member "
                        "(Pep-GaMD partition present, boost off). k0/threshold from gamd-openmm's lower-bound formula. "
                        "Frozen for the whole campaign; production consumes it via --shared-gamd-setup-dir."),
        "gamd_boost_type": "pep-gamd-lower-dual",
        "temperature_K": float(temperature_k),
        "joint_envelope": report,
        "interesting_globals": dict(all_globals),
        "all_globals": all_globals,
        **{k: v for k, v in meta.items()},
    })
    return path
```

- [ ] **Step 4: GREEN.** Then read `gareus/production.py` around the consumer of `load_reusable_shared_gamd_setup` (grep its call site) and the loop at `production.py:379` that applies globals by name to an integrator. **Confirm** that applying a dict containing only the physics keys above sets those globals and leaves the integrator's own bookkeeping globals (`stage`, `stepCount`, `windowCount`, `ForceScalingFactor`…) at their defaults. If that loop instead requires every stock global (raises on a missing key, or copies `stage`), add a guard in the consumer: when `payload["mode"] == "swarm_unbiased_envelope"`, merge via `gareus.gamd_calibration.overwrite_physics_globals(all_integrator_globals(integrator), calibrations)` semantics — i.e. only overwrite keys present in both. Write a test for whichever branch you touched (in `tests/test_swarm_envelope.py`, using a fake integrator object with `getGlobalVariableName/…ByName` if the real one is heavy). Record the finding in the task report.
- [ ] **Step 5: Commit** — `feat(swarm): unbiased envelope fit with V-trace discard and reusable setup JSON`.

---

### Task 3: Ladder design — λ rungs from the ΔV_max distribution, CV1 centres and curvature-derived k, windows CSV

**Files:**
- Create: `gareus/swarm/ladder_design.py`
- Test: `tests/test_swarm_ladder_design.py`

**Interfaces:**
- Consumes: `gareus.pep_gamd.PepGamdEnvelope`, `pep_gamd_boost_kj(v_pep_kj, v_dih_kj, lam, env)` (vectorised over samples), `gareus.windows.load_explicit_2d_window_csv(args, path)` (reads back the CSV; returns `centers, ks, sec_c, sec_k, meta, ...` with `meta["gamd_lambdas"]`).
- Produces:
  ```python
  R_KJ_MOL_K = 0.008314462618
  def deltav_max_kj(v_pep_kj, v_dih_kj, env) -> np.ndarray                                  # boost at λ = 1 per frame
  def design_lambda_ladder(deltav_kj, temperature_k, *, target_beta_sigma=1.0, min_rungs=3, max_rungs=12, ess_floor=50) -> dict
      # {"lambdas": [...], "sigma_kj_per_rung": [...], "ess_per_rung": [...], "extrapolated_from_rung": int|None, "target_beta_sigma": float}
  def cv1_centers_from_samples(cv1, n_windows=16, lo_q=0.005, hi_q=0.995) -> np.ndarray
  def cv1_curvature_kcal(cv1, centers, temperature_k, *, n_hist=60, smooth_bins=2) -> np.ndarray   # F'' in kcal/mol/CV² at each centre; 0 where p is empty
  def cv1_force_constants_from_curvature(centers, curvature_kcal, temperature_k, *, overlap_sigma=1.5, k_min_kcal=5.0, k_max_kcal=1200.0) -> list[float]
  def write_ladder_windows_csv(path, centers, ks_kcal, lambdas) -> Path      # header: window,primary_cv_mode,primary_cv_center,primary_cv_k_kcal,gamd_lambda
  ```

**Physics (write these formulas into the module docstring):** for the lower-bound boost `ΔV_λ(x) = λ·ΔV_max(x)` (the threshold `E = Vmax` does not depend on k0), so the reduced-potential difference between adjacent rungs is `β·Δλ·ΔV_max(x)`. Adjacent-rung acceptance is controlled by `Δλ·β·σ_λ(ΔV_max)`, where σ_λ is the standard deviation of ΔV_max **under the ensemble at rung λ**. The swarm samples λ = 0 only; σ_λ is estimated by reweighting swarm frames with `w ∝ exp(−β λ ΔV_max)` (exact given overlap; the effective sample size `ESS = (Σw)²/Σw²` says how far that can be trusted). The ladder is grown from λ = 0: `Δλ_k = target / (β σ_{λ_k})`, `λ_{k+1} = min(1, λ_k + Δλ_k)`, until 1 — this yields the geometric-like widening of spec §3.5 from the data rather than by assumption. Rungs beyond the first with `ESS < ess_floor` are marked `extrapolated_from_rung`; S3 must confirm them.

For k(CV1): with an umbrella `k` on top of a free-energy curvature `F''`, the window's CV1 width is `σ_w² = kT / (k + F'')`. Target `σ_w = spacing / overlap_sigma` (spec S2 row: 1.5σ on CV1) → `k = kT/σ_w² − F''`, clamped to `[k_min, k_max]`. Where the swarm histogram is empty at a centre, `F'' = 0` and the spacing-only k results (same formula as `gareus.windows.adaptive_contact_force_constants_kcal`).

- [ ] **Step 1: Failing tests**

```python
# tests/test_swarm_ladder_design.py
import csv, math, pathlib, tempfile, types
import numpy as np


def _env():
    from gareus.pep_gamd import PepGamdEnvelope
    # Total: V in [-100, 100], k0max 0.6; Dihedral: V in [0, 50], k0max 0.4
    return PepGamdEnvelope(100.0, -100.0, 100.0, 0.6, 50.0, 0.0, 50.0, 0.4)


def test_deltav_max_is_boost_at_lambda_one_and_nonnegative():
    from gareus.swarm.ladder_design import deltav_max_kj
    from gareus.pep_gamd import pep_gamd_boost_kj
    rng = np.random.default_rng(0)
    vp = rng.uniform(-90, 90, 500); vd = rng.uniform(5, 45, 500)
    dv = deltav_max_kj(vp, vd, _env())
    assert dv.shape == (500,) and np.all(dv >= 0)
    assert np.allclose(dv, np.asarray(pep_gamd_boost_kj(vp, vd, 1.0, _env())))


def test_design_lambda_ladder_starts_at_zero_ends_at_one_and_widens():
    from gareus.swarm.ladder_design import design_lambda_ladder
    rng = np.random.default_rng(1)
    dv = np.abs(rng.normal(30.0, 12.0, 5000))          # kJ/mol, σ ≈ 12 → βσ ≈ 4.8 at 300 K
    d = design_lambda_ladder(dv, 300.0, target_beta_sigma=1.0)
    lam = d["lambdas"]
    assert lam[0] == 0.0 and lam[-1] == 1.0 and all(np.diff(lam) > 0)
    assert 3 <= len(lam) <= 12
    steps = np.diff(lam)
    assert steps[-1] >= steps[0] * 0.99                  # spacing does not shrink toward λ = 1
    assert len(d["ess_per_rung"]) == len(lam)


def test_design_lambda_ladder_respects_rung_bounds_and_flags_extrapolation():
    from gareus.swarm.ladder_design import design_lambda_ladder
    dv = np.abs(np.random.default_rng(2).normal(200.0, 80.0, 40))   # tiny, wide sample → poor ESS
    d = design_lambda_ladder(dv, 300.0, target_beta_sigma=1.0, max_rungs=6, ess_floor=1000)
    assert len(d["lambdas"]) <= 6 and d["lambdas"][-1] == 1.0
    assert d["extrapolated_from_rung"] is not None


def test_cv1_centers_span_observed_range_with_n_windows():
    from gareus.swarm.ladder_design import cv1_centers_from_samples
    c = cv1_centers_from_samples(np.random.default_rng(3).beta(2, 5, 20000), n_windows=16)
    assert c.shape == (16,) and 0.0 <= c[0] < c[-1] <= 1.0 and np.all(np.diff(c) > 0)


def test_curvature_positive_in_a_well_and_zero_where_empty():
    from gareus.swarm.ladder_design import cv1_curvature_kcal
    cv1 = np.random.default_rng(4).normal(0.5, 0.05, 50000)      # Gaussian well: F'' = kT/σ² > 0
    centers = np.array([0.5, 0.95])
    f2 = cv1_curvature_kcal(cv1, centers, 300.0)
    kT = 0.0019872041 * 300.0
    assert math.isclose(f2[0], kT / 0.05 ** 2, rel_tol=0.35)
    assert f2[1] == 0.0


def test_force_constants_from_curvature_subtract_and_clamp():
    from gareus.swarm.ladder_design import cv1_force_constants_from_curvature
    centers = np.linspace(0.1, 0.85, 16)
    kT = 0.0019872041 * 300.0
    spacing = centers[1] - centers[0]
    k_flat = cv1_force_constants_from_curvature(centers, np.zeros(16), 300.0)
    assert math.isclose(k_flat[5], kT / (spacing / 1.5) ** 2, rel_tol=1e-6)
    k_curved = cv1_force_constants_from_curvature(centers, np.full(16, 30.0), 300.0)
    assert all(kc < kf for kc, kf in zip(k_curved, k_flat))
    k_huge = cv1_force_constants_from_curvature(centers, np.full(16, 1e6), 300.0, k_min_kcal=5.0)
    assert all(k == 5.0 for k in k_huge)
    assert all(k <= 1200.0 for k in cv1_force_constants_from_curvature(np.array([0.1, 0.101]), np.zeros(2), 300.0))


def _args():
    return types.SimpleNamespace(primary_cv="contacts", secondary_cv="none",
                                 explicit_2d_primary_center_column="primary_cv_center",
                                 explicit_2d_primary_k_column="primary_cv_k_kcal",
                                 explicit_2d_secondary_center_column="secondary_cv_center",
                                 explicit_2d_secondary_k_column="secondary_cv_k_kcal_mol",
                                 explicit_2d_primary_cv_mode_column="primary_cv_mode",
                                 explicit_2d_secondary_cv_mode_column="secondary_cv_mode",
                                 explicit_2d_window_schema="generic")


def test_windows_csv_is_full_cross_product_and_loads_with_core_reader():
    from gareus.swarm.ladder_design import write_ladder_windows_csv
    from gareus.windows import load_explicit_2d_window_csv
    p = pathlib.Path(tempfile.mkdtemp()) / "windows.csv"
    write_ladder_windows_csv(p, np.array([0.2, 0.4, 0.6]), [100.0, 110.0, 120.0], [0.0, 0.5, 1.0])
    rows = list(csv.DictReader(p.open()))
    assert len(rows) == 9 and rows[0].keys() >= {"window", "primary_cv_mode", "primary_cv_center", "primary_cv_k_kcal", "gamd_lambda"}
    centers, ks, _sc, _sk, meta, *_ = load_explicit_2d_window_csv(_args(), p)
    assert len(centers) == 9 and sorted(set(meta["gamd_lambdas"])) == [0.0, 0.5, 1.0]
```

- [ ] **Step 2: RED.**
- [ ] **Step 3: Implement**

```python
"""S1 → S2 design inputs, all from the unbiased swarm (spec §2 S1 products ②③④, §3.5, §8 item 8).

λ rungs: ΔV_λ(x) = λ·ΔV_max(x) for the lower-bound boost, so adjacent-rung acceptance is set by
Δλ·β·σ_λ(ΔV_max). σ_λ is estimated from the λ=0 swarm by reweighting w ∝ exp(−βλΔV_max);
ESS = (Σw)²/Σw² bounds how far that estimate can be trusted. Grow the ladder from 0 with
Δλ = target/(βσ_λ) until 1.

k(CV1): σ_w² = kT/(k + F'') with σ_w = spacing/overlap_sigma → k = kT/σ_w² − F'', clamped.
"""
from __future__ import annotations
import csv, math
from pathlib import Path
from typing import List, Optional
import numpy as np
from gareus.pep_gamd import PepGamdEnvelope, pep_gamd_boost_kj

R_KJ_MOL_K = 0.008314462618
R_KCAL_MOL_K = 0.0019872041


def deltav_max_kj(v_pep_kj, v_dih_kj, env: PepGamdEnvelope) -> np.ndarray:
    return np.asarray(pep_gamd_boost_kj(np.asarray(v_pep_kj, float), np.asarray(v_dih_kj, float), 1.0, env), dtype=float)


def _reweighted_sigma_and_ess(dv: np.ndarray, beta: float, lam: float) -> tuple[float, float]:
    logw = -beta * lam * dv
    w = np.exp(logw - logw.max()); w /= w.sum()
    mean = float((w * dv).sum()); var = float((w * (dv - mean) ** 2).sum())
    ess = float(1.0 / (w ** 2).sum())
    return math.sqrt(max(var, 0.0)), ess


def design_lambda_ladder(deltav_kj, temperature_k: float, *, target_beta_sigma: float = 1.0,
                         min_rungs: int = 3, max_rungs: int = 12, ess_floor: int = 50) -> dict:
    dv = np.asarray(deltav_kj, dtype=float); dv = dv[np.isfinite(dv)]
    if dv.size < 2:
        raise ValueError("need at least two finite ΔV_max samples")
    beta = 1.0 / (R_KJ_MOL_K * float(temperature_k))
    lambdas = [0.0]; sigmas = []; esss = []; extrapolated: Optional[int] = None
    while lambdas[-1] < 1.0 and len(lambdas) < int(max_rungs):
        lam = lambdas[-1]
        sig, ess = _reweighted_sigma_and_ess(dv, beta, lam)
        sigmas.append(sig); esss.append(ess)
        if ess < ess_floor and extrapolated is None and lam > 0.0:
            extrapolated = len(lambdas) - 1
        step = float(target_beta_sigma) / (beta * sig) if sig > 0 else 1.0
        lambdas.append(min(1.0, lam + step))
    if lambdas[-1] < 1.0:                       # max_rungs reached: force the top rung
        lambdas[-1] = 1.0
    while len(lambdas) < int(min_rungs):        # too few: insert midpoints of the widest gap
        gaps = np.diff(lambdas); i = int(np.argmax(gaps))
        lambdas.insert(i + 1, 0.5 * (lambdas[i] + lambdas[i + 1]))
    # per-rung statistics for the final list (re-evaluate so lengths match)
    sigmas, esss = [], []
    for lam in lambdas:
        sig, ess = _reweighted_sigma_and_ess(dv, beta, lam); sigmas.append(sig); esss.append(ess)
    if extrapolated is None:
        low = [i for i, e in enumerate(esss) if i > 0 and e < ess_floor]
        extrapolated = low[0] if low else None
    return {"lambdas": [float(x) for x in lambdas], "sigma_kj_per_rung": sigmas, "ess_per_rung": esss,
            "extrapolated_from_rung": extrapolated, "target_beta_sigma": float(target_beta_sigma),
            "beta_sigma_lambda0": beta * sigmas[0], "n_samples": int(dv.size)}


def cv1_centers_from_samples(cv1, n_windows: int = 16, lo_q: float = 0.005, hi_q: float = 0.995) -> np.ndarray:
    v = np.asarray(cv1, dtype=float); v = v[np.isfinite(v)]
    lo, hi = float(np.quantile(v, lo_q)), float(np.quantile(v, hi_q))
    lo = max(0.0, lo); hi = min(1.0, hi)
    if hi - lo < 0.05:
        hi = min(1.0, lo + 0.10)
    return np.linspace(lo, hi, int(n_windows))


def cv1_curvature_kcal(cv1, centers, temperature_k: float, *, n_hist: int = 60, smooth_bins: int = 2) -> np.ndarray:
    v = np.asarray(cv1, dtype=float); v = v[np.isfinite(v)]
    kT = R_KCAL_MOL_K * float(temperature_k)
    hist, edges = np.histogram(v, bins=int(n_hist), range=(0.0, 1.0))
    p = hist.astype(float)
    if smooth_bins > 0:
        kern = np.exp(-0.5 * (np.arange(-3 * smooth_bins, 3 * smooth_bins + 1) / smooth_bins) ** 2); kern /= kern.sum()
        p = np.convolve(p, kern, mode="same")
    mid = 0.5 * (edges[:-1] + edges[1:]); h = mid[1] - mid[0]
    F = np.where(p > 0, -kT * np.log(np.where(p > 0, p, 1.0)), np.nan)
    F2 = np.full_like(F, np.nan)
    F2[1:-1] = (F[2:] - 2 * F[1:-1] + F[:-2]) / h ** 2
    out = np.zeros(len(centers), dtype=float)
    for i, c in enumerate(centers):
        j = int(np.clip(np.searchsorted(mid, c), 1, len(mid) - 2))
        out[i] = F2[j] if np.isfinite(F2[j]) else 0.0
    return out


def cv1_force_constants_from_curvature(centers, curvature_kcal, temperature_k: float, *, overlap_sigma: float = 1.5,
                                       k_min_kcal: float = 5.0, k_max_kcal: float = 1200.0) -> List[float]:
    c = np.asarray(centers, dtype=float); f2 = np.asarray(curvature_kcal, dtype=float)
    kT = R_KCAL_MOL_K * float(temperature_k)
    if c.size < 2:
        return [float(k_max_kcal)]
    sp = np.diff(c); local = np.empty_like(c); local[0] = sp[0]; local[-1] = sp[-1]
    if c.size > 2:
        local[1:-1] = 0.5 * (sp[:-1] + sp[1:])
    ks = []
    for spacing, curv in zip(local, f2):
        sigma_w = max(1e-5, float(spacing) / float(overlap_sigma))
        k = kT / sigma_w ** 2 - (float(curv) if math.isfinite(curv) else 0.0)
        ks.append(float(min(float(k_max_kcal), max(float(k_min_kcal), k))))
    return ks


def write_ladder_windows_csv(path, centers, ks_kcal, lambdas) -> Path:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["window", "primary_cv_mode", "primary_cv_center", "primary_cv_k_kcal", "gamd_lambda"])
        n = 0
        for c, k in zip(centers, ks_kcal):
            for lam in lambdas:
                w.writerow([n, "contacts", f"{float(c):.6f}", f"{float(k):.4f}", f"{float(lam):.6f}"]); n += 1
    return path
```

- [ ] **Step 4: GREEN** (adjust `rel_tol` in the curvature test only if the smoothing kernel biases it — record the measured value in the report; do not loosen past 0.5).
- [ ] **Step 5: Commit** — `feat(swarm): λ-rung spacing from the reweighted ΔV_max distribution, CV1 centres and curvature-derived k`.

---

### Task 4: Seed frames per window and the seed bank

**Files:**
- Create: `gareus/swarm/seeds.py`
- Test: `tests/test_swarm_seeds.py`

**Interfaces:**
- Consumes: `gareus.tica._append_seed_bank_row(seed_bank_dir, row)` (fieldnames `seed_name, survivor_pdb_path, source_run_dir, source_pdb_path, source_label, source_state_id, source_epoch_window, primary_cv_value, secondary_cv_value`); member `trace.csv` (`frame,t_ps,cv1,...`) and per-member frame PDBs (see Task 5: the member loop writes `frames/frame_XXXXX.pdb` at `--swarm-seed-frame-interval-ps`, default 20 ps, solute only is **not** acceptable — seeds are grafted, so write full peptide atoms; water is not needed: write peptide atoms only via `write_solute_only_topology_pdb`-style selection).
- Produces:
  ```python
  def select_window_seed_frames(frames: list[dict], centers, *, per_window: int = 3, discard_frames: int = 0) -> dict[int, list[dict]]
      # frames: [{"member_id", "frame", "cv1", "pdb_path"}]; returns window index -> up to per_window frames nearest the centre, from distinct members first
  def export_seed_bank(seed_bank_dir: Path, selection: dict[int, list[dict]], centers, *, round_index: int) -> Path
      # copies PDBs into seed_bank_dir, appends rows; primary_cv_value = frame cv1; secondary_cv_value = ""; source_label = f"swarm_round_{round_index:03d}"
  ```
  Seeds are rung-independent: the same frames seed every λ at a window (the ladder differs only in k0).

- [ ] **Step 1: Failing tests**

```python
# tests/test_swarm_seeds.py
import csv, pathlib, tempfile
import numpy as np


def _frames(tmp, n_members=4, n_frames=50):
    rng = np.random.default_rng(0); out = []
    for m in range(n_members):
        for f in range(n_frames):
            p = tmp / f"m{m}_f{f}.pdb"; p.write_text("ATOM      1  CA  GLY A   1       0.000   0.000   0.000  1.00  0.00           C\nEND\n")
            out.append({"member_id": m, "frame": f, "cv1": float(rng.uniform(0, 1)), "pdb_path": str(p)})
    return out


def test_selection_prefers_distinct_members_and_nearest_frames():
    from gareus.swarm.seeds import select_window_seed_frames
    tmp = pathlib.Path(tempfile.mkdtemp())
    frames = _frames(tmp)
    sel = select_window_seed_frames(frames, [0.2, 0.8], per_window=3, discard_frames=10)
    assert set(sel) == {0, 1}
    for w, c in zip((0, 1), (0.2, 0.8)):
        assert 1 <= len(sel[w]) <= 3
        assert len({fr["member_id"] for fr in sel[w]}) == len(sel[w])          # distinct members
        assert all(fr["frame"] >= 10 for fr in sel[w])
        nearest = min(abs(fr["cv1"] - c) for fr in frames if fr["frame"] >= 10)
        assert abs(sel[w][0]["cv1"] - c) <= nearest + 1e-12


def test_export_seed_bank_writes_rows_load_genpept_library_can_read():
    from gareus.swarm.seeds import select_window_seed_frames, export_seed_bank
    tmp = pathlib.Path(tempfile.mkdtemp())
    sel = select_window_seed_frames(_frames(tmp), [0.5], per_window=2)
    bank = export_seed_bank(tmp / "seed_bank", sel, [0.5], round_index=0)
    rows = list(csv.DictReader((bank / "final_survivor_seeds.csv").open()))
    assert len(rows) == len(sel[0]) and all(pathlib.Path(r["survivor_pdb_path"]).exists() for r in rows)
    assert all(r["source_label"] == "swarm_round_000" for r in rows)
    assert all(r["secondary_cv_value"] == "" for r in rows) and all(float(r["primary_cv_value"]) >= 0 for r in rows)
```

- [ ] **Step 2: RED. Step 3: Implement** (`gareus/swarm/seeds.py`, ~70 lines): sort candidate frames by `|cv1 − centre|`, walk them taking the first frame from each not-yet-used member until `per_window`; `export_seed_bank` copies each PDB to `seed_bank_dir/w{window:02d}_m{member:04d}_f{frame:05d}.pdb` and calls `_append_seed_bank_row` with `seed_name` = that stem, `source_run_dir` = the member dir, `source_state_id` = window index, `source_epoch_window` = `f"round{round_index:03d}"`. Also write `seed_bank_dir/selection.json` (window → frames chosen, centre, |Δcv1|).
- [ ] **Step 4: GREEN. Step 5: Commit** — `feat(swarm): per-window seed frames exported as a GENPEPT-compatible seed bank`.

---

### Task 5: Member loop — one unbiased 1 ns trajectory with the Pep-GaMD partition, boost off

**Files:**
- Create: `gareus/swarm/members.py`
- Test: `tests/test_swarm_members.py`

**Interfaces:**
- Consumes: `gareus.pep_gamd.ensure_pep_gamd_partition(system, peptide_atoms) -> int`, `peptide_essential_energy_kj(context, unit)`, `DIHEDRAL_GROUP`, `physical_potential_energy_kj(context, system, unit)`; `gareus.production.make_cmd_integrator(openmm, args, unit, system=system)` (excludes group 1); `gareus.seeding.graft_conformer_into_context(sim, topology, conformer, cv_atom1, cv_atom2, temperature_k, unit, minimize_iters, seed) -> dict` (`{"fallback": bool, ...}`); `gareus.cv.build_nonlocal_contact_pairs(topology, args)`, `nonlocal_contact_cv_from_positions_nm`, `peptide_residues`, `find_atom_in_residue`, `solute_atom_indices(topology)`; `gareus.system_setup.run_steps_safely(sim, nsteps, label, out_dir, app, topology, unit, ...)`, `write_state_pdb(path, app, topology, positions)`; `gareus.swarm.stratify.rg_and_e2e_nm`.
- Produces:
  ```python
  TRACE_COLUMNS = ["frame", "t_ps", "cv1", "rg_nm", "e2e_nm", "v_pep_kj", "v_dih_kj", "potential_kj"]
  def measure_frame(context, system, unit, positions_nm, contact_pairs, ca_indices, args) -> dict   # one trace row (without frame/t_ps)
  def run_member_loop(*, n_equil_steps, n_prod_steps, steps_per_frame, seed_frame_every, step_fn, measure_fn, write_frame_fn, trace_path) -> dict
      # pure bookkeeping, injected callables: step_fn(n) advances; measure_fn() -> row dict; write_frame_fn(frame_index) writes a PDB
  def run_member(args, member_row: dict, member_dir: Path, *, openmm, app, unit, topology, base_system_xml: str, equil_state, conformer: dict | None, platform, props, contact_pairs, progress=None) -> dict
  def member_done(member_dir: Path) -> bool           # done.json exists and parses
  ```
  `run_member`: deserialise `base_system_xml` (a fresh System per member so the barostat/aux force state is clean), `ensure_pep_gamd_partition(system, solute_atom_indices(topology))`, `integrator, _ = make_cmd_integrator(openmm, args, unit, system=system)` with `args.seed` temporarily set to `member_row["velocity_seed"]`, `sim = app.Simulation(topology, system, integrator, platform, props)`, set positions/box from `equil_state` (the box and water only — the peptide coordinates are replaced by the graft), graft `conformer` (**required**, never None in practice: `run_member` raises `ValueError("swarm member needs a seed conformer")` when it is None). If `graft_conformer_into_context` returns `{"fallback": True, ...}` the member **fails**: write `done.json` with `{"status": "graft_failed", "graft_fallback_reason": ...}`, write no trace, return — the extended-chain coordinates are never simulated (Global Constraints, seed prerequisite). Otherwise `setVelocitiesToTemperature(T, velocity_seed)`, then `run_member_loop` with `step_fn = lambda n: run_steps_safely(sim, n, "swarm", member_dir, app, topology, unit, ...)`. Equilibration length `--swarm-equil-ps` (default 100 ps) is **discarded by construction** and never enters the trace; the trace covers only the `--swarm-seed-ns` production part, at `--swarm-output-interval-ps` (default 2.0). Writes `trace.csv`, `frames/frame_XXXXX.pdb` (peptide atoms only) every `--swarm-seed-frame-interval-ps` (default 20), `last_frame.pdb`, `done.json` (`{"member_id", "n_frames", "graft_fallback", "wall_s", "ns_per_day"}`).

- [ ] **Step 1: Failing tests** — the bookkeeping loop with fakes, plus the measured-row contract:

```python
# tests/test_swarm_members.py
import csv, pathlib, tempfile


def test_run_member_loop_writes_expected_frames_and_seed_pdbs():
    from gareus.swarm.members import run_member_loop, TRACE_COLUMNS
    tmp = pathlib.Path(tempfile.mkdtemp())
    stepped = []; frames_written = []
    state = {"t": 0}
    def step_fn(n): stepped.append(n); state["t"] += n
    def measure_fn(): return {"cv1": 0.3, "rg_nm": 0.6, "e2e_nm": 1.0, "v_pep_kj": -50.0 - state["t"], "v_dih_kj": 20.0, "potential_kj": -1e4}
    def write_frame_fn(i): frames_written.append(i)
    summary = run_member_loop(n_equil_steps=100, n_prod_steps=1000, steps_per_frame=100, seed_frame_every=5,
                              step_fn=step_fn, measure_fn=measure_fn, write_frame_fn=write_frame_fn, trace_path=tmp / "trace.csv")
    rows = list(csv.DictReader((tmp / "trace.csv").open()))
    assert list(rows[0].keys()) == TRACE_COLUMNS
    assert len(rows) == 10 and summary["n_frames"] == 10
    assert stepped[0] == 100 and sum(stepped) == 1100                      # equilibration first, never in the trace
    assert float(rows[0]["v_pep_kj"]) == -50.0 - 200.0                     # first trace row is after equil + first prod chunk
    assert frames_written == [4, 9]                                        # every 5th frame (0-based index 4, 9)
    assert float(rows[-1]["t_ps"]) > float(rows[0]["t_ps"])


def test_run_member_loop_rejects_non_divisible_chunking():
    from gareus.swarm.members import run_member_loop
    try:
        run_member_loop(n_equil_steps=0, n_prod_steps=1001, steps_per_frame=100, seed_frame_every=1,
                        step_fn=lambda n: None, measure_fn=lambda: {}, write_frame_fn=lambda i: None,
                        trace_path=pathlib.Path(tempfile.mkdtemp()) / "t.csv")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for n_prod_steps not divisible by steps_per_frame")


def test_measure_frame_returns_all_energy_and_geometry_keys_on_tiny_real_system():
    # Reference platform, alanine dipeptide-like tiny system from the repo's Pep-GaMD test fixture.
    from tests.pep_gamd_fixture import tiny_solvated_system   # grep tests/pep_gamd_fixture.py for the real helper name and adapt
    from gareus.swarm.members import measure_frame, TRACE_COLUMNS
    import types
    openmm, app, unit, topology, system, positions = tiny_solvated_system()
    from gareus.pep_gamd import ensure_pep_gamd_partition
    from gareus.cv import solute_atom_indices, build_nonlocal_contact_pairs, peptide_residues, find_atom_in_residue
    ensure_pep_gamd_partition(system, solute_atom_indices(topology))
    args = types.SimpleNamespace(contact_atom_selection="heavy", contact_scheme="residue-balanced", contact_min_sequence_separation=1,
                                 contact_r0_a=4.5, contact_beta_a_inv=6.0, contact_normalize=True, contact_pair_warning_threshold=5000)
    pairs = build_nonlocal_contact_pairs(topology, args)
    ca = [find_atom_in_residue(r, "CA") for r in peptide_residues(topology)]
    ctx = openmm.Context(system, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName("Reference"))
    ctx.setPositions(positions)
    import numpy as np
    pos_nm = np.asarray(ctx.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer))
    row = measure_frame(ctx, system, unit, pos_nm, pairs, ca, args)
    assert set(row) == set(TRACE_COLUMNS) - {"frame", "t_ps"}
    assert all(np.isfinite(float(row[k])) for k in row)
```

- [ ] **Step 2: RED.** (For the third test, first read `tests/pep_gamd_fixture.py` and use its real builder; if it exposes only a System without topology/positions, build the tiny system the way `tests/test_pep_gamd_boost.py` does and factor a helper into `tests/pep_gamd_fixture.py`. Never add a fixture-parameter to the test function.)
- [ ] **Step 3: Implement** — `run_member_loop`:

```python
def run_member_loop(*, n_equil_steps: int, n_prod_steps: int, steps_per_frame: int, seed_frame_every: int,
                    step_fn, measure_fn, write_frame_fn, trace_path: Path, timestep_ps: float = 0.004) -> dict:
    if n_prod_steps % steps_per_frame:
        raise ValueError(f"n_prod_steps={n_prod_steps} not divisible by steps_per_frame={steps_per_frame}")
    trace_path = Path(trace_path); trace_path.parent.mkdir(parents=True, exist_ok=True)
    if n_equil_steps > 0:
        step_fn(int(n_equil_steps))                       # discarded by construction
    n_frames = n_prod_steps // steps_per_frame
    with trace_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TRACE_COLUMNS); w.writeheader()
        for i in range(n_frames):
            step_fn(int(steps_per_frame))
            row = dict(measure_fn()); row["frame"] = i; row["t_ps"] = (i + 1) * steps_per_frame * timestep_ps
            w.writerow({k: row.get(k, "") for k in TRACE_COLUMNS}); f.flush()
            if seed_frame_every > 0 and (i + 1) % seed_frame_every == 0:
                write_frame_fn(i)
    return {"n_frames": n_frames, "n_equil_steps": int(n_equil_steps), "n_prod_steps": int(n_prod_steps)}
```
  `measure_frame`: `v_pep = peptide_essential_energy_kj(context, unit)`; `v_dih = context.getState(getEnergy=True, groups={DIHEDRAL_GROUP}).getPotentialEnergy() / kJ/mol`; `potential = physical_potential_energy_kj(context, system, unit)`; `cv1 = nonlocal_contact_cv_from_positions_nm(positions_nm, contact_pairs, args)`; `rg, e2e = rg_and_e2e_nm(positions_nm[ca_indices])`. `run_member` as specified in Interfaces; timestep from `args.timestep_fs`; `steps_per_frame = round(output_interval_ps*1000/timestep_fs)`; assert `n_prod_steps % steps_per_frame == 0` by rounding `n_prod_steps` up to a multiple and recording the actual ns in `done.json`.
- [ ] **Step 4: GREEN. Step 5: Commit** — `feat(swarm): unbiased member trajectory with the Pep-GaMD partition present and the boost off`.

---

### Task 6: Driver — system build/reload, member plan, sharding, resume, rounds

**Files:**
- Create: `gareus/swarm/driver.py`
- Test: `tests/test_swarm_driver.py` (create; bookkeeping only, no MD)

**Interfaces:**
- Consumes: `gareus.system_setup.minimize_and_npt_equilibrate(args, out_dir, progress) -> (openmm, app, unit, forcefield, topology, final_state)`, `create_system(app, unit, forcefield, topology, args, include_barostat=True)`, `setup_platform_and_properties(openmm, args) -> (platform, props)`, `make_forcefield(app, water_model)`, `write_state_pdb`; `gareus.seeding.load_genpept_conformer_library(seed_dir, primary_cv_def=..., args=args, topology=topology)`; `gareus.cv.prepare_primary_cv_definition(topology, args)` (grep its return keys for the contact-pair list); Task 1 `describe_seeds/stratify_cells/plan_members`; Task 5 `run_member/member_done`.
- Produces:
  ```python
  def swarm_root(out_dir) -> Path                                      # out_dir/"swarm"
  def round_dir(out_dir, round_index) -> Path
  def latest_round_index(out_dir) -> int | None
  def parse_member_range(spec: str | None, n_members: int) -> range     # "a:b" half-open, None -> all
  def ensure_system(args, out_dir, progress) -> dict                     # builds or reloads swarm/system/{topology.pdb,equil_state.xml,base_system.xml}
  def build_or_load_plan(args, out_dir, round_index, *, topology, contact_pairs) -> tuple[list[dict], dict]   # plan.csv + plan_meta.json, deterministic
  def run_swarm_stage(args, out_dir, progress=None) -> dict
  ```
  Behaviour: `run_swarm_stage` first resolves `args.seed_conformers_dir`; if it is unset or `<dir>/final_survivor_seeds.csv` is missing it raises `SystemExit("--swarm-stage run needs --seed-conformers-dir <GENPEPT library> (final_survivor_seeds.csv); a contact-CV start from the extended chain is not possible")` before any MD (unit-test this with a fake args object). `ensure_system` writes `topology.pdb` (`write_state_pdb` of the equilibrated positions, full system), `equil_state.xml` (`XmlSerializer.serialize(final_state)`), `base_system.xml` (serialised System **without** the aux force — the partition is added per member). On reload it uses `app.PDBFile(topology.pdb)` for topology and deserialises the state/system; it never re-solvates. Round 0 seeds come from `--seed-conformers-dir`; a later round (`--swarm-round N`, N ≥ 1, `--swarm-seed-source production-frames --swarm-production-seed-csv PATH`) takes rows `pdb_path,cv1,rg_nm,e2e_nm` from that CSV and stratifies them with the **same bin edges as round 0** (persist edges in `round_000/plan_meta.json`; later rounds read them — the coordinate is frozen after the first swarm, spec decision). Members whose `done.json` exists are skipped (resume). `--swarm-member-range a:b` runs a shard; the analysis (Task 8) pools whatever is done and reports missing members.

- [ ] **Step 1: Failing tests** (bookkeeping only):

```python
# tests/test_swarm_driver.py
import json, pathlib, tempfile


def test_parse_member_range_half_open_and_default_all():
    from gareus.swarm.driver import parse_member_range
    assert list(parse_member_range("3:6", 10)) == [3, 4, 5]
    assert list(parse_member_range(None, 4)) == [0, 1, 2, 3]
    assert list(parse_member_range("8:20", 10)) == [8, 9]


def test_round_dirs_and_latest_round_index():
    from gareus.swarm.driver import round_dir, latest_round_index
    out = pathlib.Path(tempfile.mkdtemp())
    assert latest_round_index(out) is None
    round_dir(out, 0).mkdir(parents=True); round_dir(out, 2).mkdir(parents=True)
    assert latest_round_index(out) == 2
    assert round_dir(out, 1).name == "round_001"


def test_later_round_reuses_round0_bin_edges():
    from gareus.swarm.driver import stratify_with_frozen_edges
    from gareus.swarm.stratify import SeedDescriptor
    edges = {"cv1": [0.0, 0.5, 1.0], "rg": [0.0, 1.0], "e2e": [0.0, 3.0]}
    seeds = [SeedDescriptor("a", "/a", 0.1, 0.5, 1.0), SeedDescriptor("b", "/b", 0.9, 0.5, 1.0)]
    cells = stratify_with_frozen_edges(seeds, edges)
    assert set(cells) == {(0, 0, 0), (1, 0, 0)}
```

- [ ] **Step 2: RED. Step 3: Implement** the driver (~250 lines). `stratify_with_frozen_edges(seeds, edges)` reuses `stratify._bin_index`; `stratify_cells` from Task 1 is used for round 0 and its edges are persisted (`quantile_edges` outputs). For seed CA indices in seed-atom numbering use `conformer["topology_to_conformer_atom_index"]` to map topology CA indices → seed indices (fall back to all atoms when the map is empty). Per member print one line: `member 0012/0096 cell 2_1_0 seed seed_00345 rep 0 … 412 ns/day`. Emit `progress` events `{"event": "swarm_member_done", ...}` when `progress` is not None.
- [ ] **Step 4: GREEN. Step 5: Commit** — `feat(swarm): stage driver with system reuse, deterministic member plan, sharding, resume and frozen-edge rounds`.

---

### Task 7: Gates and the extension plan

**Files:**
- Create: `gareus/swarm/gates.py`
- Test: `tests/test_swarm_gates.py`

**Interfaces:**
- Consumes: Task 2 `pool_member_envelopes`, Task 3 `design_lambda_ladder`.
- Produces:
  ```python
  def coverage_gate(plan_rows, done_ids: set[int], *, min_done_fraction=0.9) -> dict     # every cell has ≥1 done member; ≥ fraction of members done
  def envelope_stability_gate(traces, discard, *, sigma_rel_tol=0.10, extrema_sigma_tol=1.0) -> dict   # odd vs even member halves
  def ladder_ess_gate(ladder: dict, *, ess_floor=50) -> dict                             # no extrapolated rung
  def graft_gate(done_summaries: list[dict], *, max_fallback_fraction=0.10) -> dict   # fraction of members with status "graft_failed"; those members never ran (no extended-chain fallback exists)
  def evaluate_gates(...) -> dict     # {"status": "pass"|"fail", "gates": {...}, "reasons": [...]}
  def extension_plan(plan_meta: dict, gate: dict) -> dict     # {"extra_replicates_per_cell": int, "reason": str} — doubles R (cap 4×) on stability/ESS failure, +1 on coverage failure
  ```

- [ ] **Step 1: Failing tests**

```python
# tests/test_swarm_gates.py
import numpy as np


def _traces(n=8, sigma=5.0, seed=0):
    rng = np.random.default_rng(seed)
    return {m: {"v_pep_kj": rng.normal(-50, sigma, 500), "v_dih_kj": rng.normal(20, 2.0, 500)} for m in range(n)}


def test_coverage_gate_fails_on_empty_cell_and_low_completion():
    from gareus.swarm.gates import coverage_gate
    rows = [{"member_id": i, "cell_id": "0_0_0" if i < 4 else "1_0_0"} for i in range(8)]
    assert coverage_gate(rows, set(range(8)))["ok"]
    assert not coverage_gate(rows, {0, 1, 2, 3})["ok"]                 # cell 1_0_0 empty
    assert not coverage_gate(rows, {0, 1, 2, 3, 4, 5}, min_done_fraction=0.9)["ok"]


def test_envelope_stability_passes_iid_and_fails_shifted_halves():
    from gareus.swarm.gates import envelope_stability_gate
    assert envelope_stability_gate(_traces(), discard=0)["ok"]
    tr = _traces()
    for m in tr:
        if m % 2: tr[m]["v_pep_kj"] = tr[m]["v_pep_kj"] + 40.0        # odd members shifted by 8σ
    g = envelope_stability_gate(tr, discard=0)
    assert not g["ok"] and "Total" in " ".join(g["reasons"])


def test_ladder_ess_gate_and_extension_plan():
    from gareus.swarm.gates import ladder_ess_gate, extension_plan
    assert ladder_ess_gate({"lambdas": [0, 0.5, 1], "ess_per_rung": [500, 200, 120], "extrapolated_from_rung": None})["ok"]
    bad = ladder_ess_gate({"lambdas": [0, 0.5, 1], "ess_per_rung": [500, 20, 5], "extrapolated_from_rung": 1})
    assert not bad["ok"]
    plan = extension_plan({"replicates_per_cell": 3}, {"status": "fail", "gates": {"ladder_ess": bad, "coverage": {"ok": True}, "envelope_stability": {"ok": True}, "graft": {"ok": True}}})
    assert plan["extra_replicates_per_cell"] == 3
```

- [ ] **Step 2: RED. Step 3: Implement** (~120 lines; `envelope_stability_gate` pools odd and even member ids separately with `pool_member_envelopes` and compares `sigmav` relative difference and `|Vmax_a − Vmax_b|`, `|Vmin_a − Vmin_b|` in units of the pooled `sigmav`, per group). `evaluate_gates` writes nothing; Task 8 persists `swarm_gate.json`.
- [ ] **Step 4: GREEN. Step 5: Commit** — `feat(swarm): coverage, envelope-stability, ladder-ESS and graft gates with an extension plan`.

---

### Task 8: Analysis entry — pool → envelope → gates → ladder → seeds → windows CSV → report

**Files:**
- Create: `gareus/swarm/analyze.py`
- Test: `tests/test_swarm_analyze.py`

**Interfaces:**
- Consumes: Tasks 2–4, 6, 7; `gareus.pep_gamd.PepGamdEnvelope.from_json`.
- Produces: `analyze_swarm_stage(out_dir, args) -> dict` writing under `swarm/analysis/`: `envelope_discard.json` (per-member discard, pooled discard, frames→ps), `shared_gamd_setup/shared_gamd_setup_globals.json` (**only for round 0**; later rounds read the frozen one and add `out_of_envelope_fraction` per channel to the report), `ladder_design.json` (Task 3 output + centres + curvature + k list + `n_states = n_windows × n_rungs`), `windows_lambda_ladder.csv`, **`ladder_run_args.yaml`** (the seed-selection sidecar, see below), `seed_bank/`, `swarm_gate.json`, `swarm_report.json` (everything above summarised + `missing_members` + `graft_failed_members` + `ns_per_day` median + `budget_ns_done`). Members whose `done.json` has `status == "graft_failed"` contribute nothing to the envelope, seeds or ladder and are counted for Task 7's `graft_gate`. On gate `fail`, still writes every artefact, sets `report["status"]="fail"` and `report["extension"]` from `extension_plan`, and **does not** write `windows_lambda_ladder.csv` or `ladder_run_args.yaml` (a failed gate must not hand S2 a ladder).
- `ladder_run_args.yaml` (written with `gareus.config._write_yaml_or_json`) is the production `starting_structures` + `windows` + `gamd` fragment a `windows_2d_csv` run must consume, so the seed path travels with the CSV:
  ```yaml
  windows:
    window_mode: manual
    windows_2d_csv: <abs path>/swarm/analysis/windows_lambda_ladder.csv
  starting_structures:
    seed_conformers_dir: <abs path>/swarm/analysis/seed_bank      # swarm seeds (round 0) — cli.py:471
    seed_selection_mode: active-cv                                  # cli.py:473
    seed_max_reuse_per_conformer: 0                                 # cli.py:477
    us_seed_preflight_max_score: 1.2                                # cli.py:478
    us_pull_steps_per_window: 150000                                # chignolin_7 campaign values
    us_pull_timestep_fs: 3.0
    us_pull_k: 300.0
    us_pull_ramp_stages: 10
  gamd:
    gamd_boost_type: pep-gamd-lower-dual
    shared_gamd_setup_dir: <abs path>/swarm/analysis/shared_gamd_setup
  ```
  The dests above must be verified against `gareus/cli.py` (grep each `us_pull_*` dest before writing; use the real names). Test: `test_analyze_writes_ladder_run_args_sidecar_with_seed_path` asserts the YAML loads (`gareus.config._load_config_file`) and `starting_structures.seed_conformers_dir` points at the exported seed bank.

- [ ] **Step 1: Failing test** with synthetic member directories:

```python
# tests/test_swarm_analyze.py
import csv, json, pathlib, tempfile, types
import numpy as np
from gareus.swarm.members import TRACE_COLUMNS


def _fake_round(out, n_members=6, n_frames=300, shift_odd=0.0):
    rng = np.random.default_rng(0)
    rd = out / "swarm" / "round_000"; rd.mkdir(parents=True)
    with (rd / "plan.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["member_id", "cell_id", "cell_cv1", "cell_rg", "cell_e2e", "seed_id", "seed_pdb", "replicate", "velocity_seed"]); w.writeheader()
        for m in range(n_members):
            w.writerow({"member_id": m, "cell_id": f"{m % 2}_0_0", "cell_cv1": m % 2, "cell_rg": 0, "cell_e2e": 0, "seed_id": f"s{m}", "seed_pdb": "/x", "replicate": 0, "velocity_seed": m})
    json.dump({"n_cells": 2, "replicates_per_cell": 3, "n_members": n_members, "budget_ns": 6.0, "seed_ns": 1.0,
               "edges": {"cv1": [0, 0.5, 1], "rg": [0, 2], "e2e": [0, 4]}}, (rd / "plan_meta.json").open("w"))
    for m in range(n_members):
        md = rd / f"member_{m:04d}"; (md / "frames").mkdir(parents=True)
        with (md / "trace.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=TRACE_COLUMNS); w.writeheader()
            for i in range(n_frames):
                w.writerow({"frame": i, "t_ps": 2.0 * (i + 1), "cv1": float(rng.beta(2, 4)), "rg_nm": 0.6, "e2e_nm": 1.0,
                            "v_pep_kj": float(rng.normal(-50 + (shift_odd if m % 2 else 0), 5)), "v_dih_kj": float(rng.normal(20, 2)), "potential_kj": -1e4})
                if (i + 1) % 10 == 0:
                    (md / "frames" / f"frame_{i:05d}.pdb").write_text("ATOM      1  CA  GLY A   1       0.000   0.000   0.000  1.00  0.00           C\nEND\n")
        json.dump({"member_id": m, "n_frames": n_frames, "graft_fallback": False, "wall_s": 10.0, "ns_per_day": 400.0}, (md / "done.json").open("w"))
    return rd


def _args():
    return types.SimpleNamespace(temperature_k=300.0, sigma0p_kcal_mol=6.0, sigma0d_kcal_mol=6.0, swarm_n_windows=8, swarm_overlap_sigma=1.5,
                                 swarm_target_beta_sigma=1.0, swarm_min_rungs=3, swarm_max_rungs=12, swarm_ess_floor=50,
                                 swarm_seeds_per_window=2, swarm_discard_block_frames=20, swarm_min_discard_ps=0.0,
                                 swarm_output_interval_ps=2.0, contact_adaptive_max_k_kcal=1200.0, contact_adaptive_min_k_kcal=5.0, swarm_round=0)


def test_analyze_writes_every_artifact_and_passes_on_clean_data():
    from gareus.swarm.analyze import analyze_swarm_stage
    from gareus.pep_gamd import PepGamdEnvelope
    out = pathlib.Path(tempfile.mkdtemp()); _fake_round(out)
    rep = analyze_swarm_stage(out, _args())
    an = out / "swarm" / "analysis"
    for name in ("envelope_discard.json", "shared_gamd_setup/shared_gamd_setup_globals.json", "ladder_design.json",
                 "windows_lambda_ladder.csv", "ladder_run_args.yaml", "seed_bank/final_survivor_seeds.csv", "swarm_gate.json", "swarm_report.json"):
        assert (an / name).exists(), name
    from gareus.config import _load_config_file
    side = _load_config_file(an / "ladder_run_args.yaml")
    assert side["starting_structures"]["seed_conformers_dir"] == str((an / "seed_bank").resolve())
    assert side["starting_structures"]["seed_selection_mode"] == "active-cv" and side["gamd"]["gamd_boost_type"] == "pep-gamd-lower-dual"
    assert rep["status"] == "pass"
    env = PepGamdEnvelope.from_json(an / "shared_gamd_setup/shared_gamd_setup_globals.json")
    assert env.vmax_total > env.vmin_total
    ld = json.load((an / "ladder_design.json").open())
    assert ld["n_states"] == 8 * len(ld["lambdas"]) and ld["lambdas"][0] == 0.0 and ld["lambdas"][-1] == 1.0
    assert len(list(csv.DictReader((an / "windows_lambda_ladder.csv").open()))) == ld["n_states"]


def test_analyze_fails_gate_and_withholds_windows_csv_on_shifted_halves():
    from gareus.swarm.analyze import analyze_swarm_stage
    out = pathlib.Path(tempfile.mkdtemp()); _fake_round(out, shift_odd=40.0)
    rep = analyze_swarm_stage(out, _args())
    assert rep["status"] == "fail" and rep["extension"]["extra_replicates_per_cell"] >= 3
    assert not (out / "swarm" / "analysis" / "windows_lambda_ladder.csv").exists()
```

- [ ] **Step 2: RED. Step 3: Implement** (~200 lines). Frame→ps conversion uses `swarm_output_interval_ps`. Read traces with `csv.DictReader`, coerce to float arrays. The seed selection passes `discard_frames` so no transient frame becomes a seed. `sigma0_kj = {"Total": 4.184*args.sigma0p_kcal_mol, "Dihedral": 4.184*args.sigma0d_kcal_mol}`. Later rounds (`args.swarm_round ≥ 1`): load the frozen envelope from round 0's analysis dir, compute `out_of_envelope_fraction[g] = mean(v > Vmax_g or v < Vmin_g)`, skip envelope and ladder writing, still export seeds (into `seed_bank_round_NNN/`) and report.
- [ ] **Step 4: GREEN. Step 5: Commit** — `feat(swarm): analysis entry producing envelope, gates, ladder design, windows CSV and seed bank`.

---

### Task 9: Swarm-vs-pilot envelope comparison

**Files:**
- Create: `gareus/swarm/compare_pilot.py`
- Test: `tests/test_swarm_compare_pilot.py`

**Interfaces:**
- Consumes: two `shared_gamd_setup_globals.json` files (`all_globals` + `joint_envelope[g]["sigmaV_kj_mol"]`), read through `PepGamdEnvelope.from_json` for Vmax/Vmin/k0max and `json` for sigmaV.
- Produces:
  ```python
  def compare_envelopes(swarm_json, pilot_json, *, sigma_rel_tol=0.25, extrema_sigma_tol=2.0, k0_ratio_bounds=(0.7, 1.4)) -> dict
      # per group: sigma_rel_diff, vmax_diff_in_pilot_sigma, vmin_diff_in_pilot_sigma, k0_ratio, ok; overall "status": "pass"|"fail", "freeze_allowed": bool, "reasons"
  ```
  **Tolerance (this is the campaign-freeze criterion the user asked for):** per channel, `|σ_swarm − σ_pilot| / σ_pilot ≤ 0.25`, `|Vmax_swarm − Vmax_pilot| ≤ 2 σ_pilot`, `|Vmin_swarm − Vmin_pilot| ≤ 2 σ_pilot`, `0.7 ≤ k0max_swarm / k0max_pilot ≤ 1.4`. The pilot's envelope is measured under one umbrella window with a boosted recon; the swarm's is unbiased and unboosted — they sample different ensembles, so this is a plausibility check, not an identity test. **The swarm envelope is the authoritative one for the campaign (spec §2 S1, §3.6).** A `fail` means: do not freeze the campaign config; inspect which channel disagrees; the fix is more swarm (extension), never editing the envelope by hand.

- [ ] **Step 1: Failing tests**

```python
# tests/test_swarm_compare_pilot.py
import json, pathlib, tempfile


def _write(path, vmax_t, vmin_t, sig_t, k0_t, vmax_d=50.0, vmin_d=0.0, sig_d=4.0, k0_d=0.5):
    g = {"Vmax_Total": vmax_t, "Vmin_Total": vmin_t, "Vavg_Total": 0.5 * (vmax_t + vmin_t), "sigmaV_Total": sig_t, "k0_Total": k0_t,
         "k_Total": k0_t / (vmax_t - vmin_t), "threshold_energy_Total": vmax_t, "sigma0_Total": 25.1,
         "Vmax_Dihedral": vmax_d, "Vmin_Dihedral": vmin_d, "Vavg_Dihedral": 25.0, "sigmaV_Dihedral": sig_d, "k0_Dihedral": k0_d,
         "k_Dihedral": k0_d / (vmax_d - vmin_d), "threshold_energy_Dihedral": vmax_d, "sigma0_Dihedral": 25.1}
    json.dump({"all_globals": g, "joint_envelope": {"Total": {"sigmaV_kj_mol": sig_t}, "Dihedral": {"sigmaV_kj_mol": sig_d}}}, path.open("w"))


def test_compare_passes_within_tolerance_and_fails_outside():
    from gareus.swarm.compare_pilot import compare_envelopes
    d = pathlib.Path(tempfile.mkdtemp())
    _write(d / "swarm.json", 100.0, -100.0, 20.0, 0.6); _write(d / "pilot.json", 110.0, -95.0, 22.0, 0.55)
    r = compare_envelopes(d / "swarm.json", d / "pilot.json")
    assert r["status"] == "pass" and r["freeze_allowed"] and r["groups"]["Total"]["ok"]
    _write(d / "pilot2.json", 100.0, -100.0, 40.0, 0.3)             # σ doubled, k0 halved
    r2 = compare_envelopes(d / "swarm.json", d / "pilot2.json")
    assert r2["status"] == "fail" and not r2["freeze_allowed"] and any("Total" in s for s in r2["reasons"])
```

- [ ] **Step 2: RED. Step 3: Implement** (~80 lines; also `main(argv)` so it can be called as `python -m gareus.swarm.compare_pilot swarm.json pilot.json --out pilot_comparison.json`).
- [ ] **Step 4: GREEN. Step 5: Commit** — `feat(swarm): swarm-vs-pilot envelope comparison as the campaign-freeze criterion`.

---

### Task 10: CLI flags, dispatch, config example

**Files:**
- Modify: `gareus/cli.py` — new argument group in the parser (next to `_add_output_args`, ~line 573) and a dispatch block in `main()` right after `initialize_run_manifest(args, out_dir, argv=argv_list)` (~line 1662) and before the `--extend` resolution
- Create: `examples/chignolin_swarm_stage.yaml`
- Test: `tests/test_swarm_cli.py`

**Flags** (all config-drivable: the YAML loader flattens any section key that matches an argparse dest — see `gareus/config.py:_flatten_config_mapping`):

| flag | default | meaning |
|---|---|---|
| `--swarm-stage {off,run,analyze,compare}` | off | enter the swarm stage instead of the production chain |
| `--swarm-seed-ns` | 1.0 | production length per member (user decision; do not change the default) |
| `--swarm-replicates-per-cell` | 3 | R |
| `--swarm-budget-ns` | None | when set, derives R = floor(budget/(cells·seed_ns)) and overrides `--swarm-replicates-per-cell` (print the derivation) |
| `--swarm-bins` | `4,3,3` | cells on heavy-CV1 × Rg × E2E (quantile edges) |
| `--swarm-equil-ps` | 100.0 | discarded-by-construction equilibration per member |
| `--swarm-output-interval-ps` | 2.0 | trace cadence (spec §5: 2 ps) |
| `--swarm-seed-frame-interval-ps` | 20.0 | PDB frame cadence for seed export |
| `--swarm-member-range` | None | `a:b` shard |
| `--swarm-round` | 0 | round index; ≥1 requires `--swarm-seed-source production-frames` |
| `--swarm-seed-source {genpept,production-frames}` | genpept | |
| `--swarm-production-seed-csv` | None | `pdb_path,cv1,rg_nm,e2e_nm` for later rounds |
| `--swarm-n-windows` | 16 | CV1 centres |
| `--swarm-overlap-sigma` | 1.5 | window width target = spacing/overlap_sigma |
| `--swarm-target-beta-sigma` | 1.0 | rung acceptance control `Δλ·βσ` |
| `--swarm-min-rungs` / `--swarm-max-rungs` | 3 / 12 | |
| `--swarm-ess-floor` | 50 | reweighting ESS below which a rung is "extrapolated" |
| `--swarm-seeds-per-window` | 3 | |
| `--swarm-discard-block-frames` | 25 | V-trace block for the discard detector |
| `--swarm-min-discard-ps` | 0.0 | floor on the discard |
| `--swarm-pilot-globals` | None | pilot `shared_gamd_setup_globals.json` for `compare` |
| `--shared-gamd-setup-dir` | None | **public** flag for production: reuse an envelope directory (dest `shared_gamd_setup_dir`, already read by `production.load_reusable_shared_gamd_setup`) |

Dispatch in `main()`:

```python
    _swarm_stage = str(getattr(args, "swarm_stage", "off") or "off")
    if _swarm_stage != "off":
        from gareus.swarm.driver import run_swarm_stage
        from gareus.swarm.analyze import analyze_swarm_stage
        from gareus.swarm.compare_pilot import compare_envelopes
        progress = GuiProgressSink(out_dir, args)
        if _swarm_stage == "run":
            result = run_swarm_stage(args, out_dir, progress)
        elif _swarm_stage == "analyze":
            result = analyze_swarm_stage(out_dir, args)
        else:
            pilot = getattr(args, "swarm_pilot_globals", None)
            if not pilot:
                raise SystemExit("--swarm-stage compare needs --swarm-pilot-globals PATH")
            result = compare_envelopes(out_dir / "swarm" / "analysis" / "shared_gamd_setup" / "shared_gamd_setup_globals.json", Path(pilot))
            write_json(out_dir / "swarm" / "analysis" / "pilot_comparison.json", result)
        print(json.dumps({k: v for k, v in result.items() if k in ("status", "n_members", "missing_members", "freeze_allowed", "reasons")}, indent=2, default=str))
        return
```

- [ ] **Step 1: Failing test**

```python
# tests/test_swarm_cli.py
def test_swarm_flags_parse_with_defaults_and_budget():
    from gareus.cli import parse_args
    a = parse_args(["--seq", "GYDPETGTWG", "--out", "/tmp/x", "--swarm-stage", "run", "--swarm-budget-ns", "120"])
    assert a.swarm_stage == "run" and a.swarm_seed_ns == 1.0 and a.swarm_budget_ns == 120.0
    assert a.swarm_bins == "4,3,3" and a.swarm_output_interval_ps == 2.0 and a.shared_gamd_setup_dir is None


def test_swarm_example_config_flattens_into_swarm_dests():
    from gareus.cli import parse_args
    a = parse_args(["--config", "examples/chignolin_swarm_stage.yaml"])
    assert a.swarm_stage == "run" and a.seq == "GYDPETGTWG" and a.gamd_boost_type == "pep-gamd-lower-dual"
```

- [ ] **Step 2: RED. Step 3: Implement** flags + dispatch + the example YAML (copy the structure of `examples/chignolin_lambda_ladder_pilot.yaml`: same `sequence/output/platform/simulation/cvs/contact_cv` blocks, `gamd: {gamd_boost_type: pep-gamd-lower-dual, sigma0p: 6.0, sigma0d: 6.0}`, a `swarm:` section with `swarm_stage: run, swarm_replicates_per_cell: 3, swarm_bins: "4,3,3", swarm_seed_ns: 1.0, swarm_equil_ps: 100, swarm_output_interval_ps: 2.0, swarm_n_windows: 16`; a `starting_structures:` section with `seed_selection_mode: active-cv`, `seed_max_reuse_per_conformer: 0`, `us_seed_preflight_max_score: 1.2` and **no** `seed_conformers_dir` value (the header comment states the r7 seed dir is deployment-specific and must be passed as `--seed-conformers-dir` on the command line; the stage refuses to start without it, and a contact-CV start from the extended chain is impossible — S3 pilot finding 2026-09-07); the header also states that budget = cells × R × 1 ns is printed at plan time. The header must also say: **no native reference is used anywhere in this stage.**
- [ ] **Step 4: GREEN. Step 5: Commit** — `feat(swarm): --swarm-stage run|analyze|compare, swarm flags, public --shared-gamd-setup-dir, chignolin example`.

---

### Task 11: Tiny real end-to-end swarm (slow), provenance, helptext, handoff

**Files:**
- Modify: `gareus/provenance.py:270-300` (`_method_settings`: add every `swarm_*` dest and `shared_gamd_setup_dir`), `:350-372` (`_key_artifact_paths`: `swarm/analysis/shared_gamd_setup/shared_gamd_setup_globals.json`, `swarm/analysis/ladder_design.json`, `swarm/analysis/windows_lambda_ladder.csv`, `swarm/analysis/swarm_gate.json`, `swarm/analysis/swarm_report.json`)
- Modify: `gareus/helptext.py` — new section "18. Unbiased swarm stage (S0/S1)" after "17. Inter-epoch tICA…": what runs (unbiased, partition present, boost off), what it produces, the three consumer flags for S2/S4 (`--windows-2d-csv swarm/analysis/windows_lambda_ladder.csv`, `--seed-conformers-dir swarm/analysis/seed_bank`, `--shared-gamd-setup-dir swarm/analysis/shared_gamd_setup`), the gates and the extension rule, the pilot comparison and its tolerance, and the sentence "No native reference or folded-state label is used anywhere in this stage."
- Modify: `CLAUDE.md` — a "Swarm stage" entry: directory layout, the frozen-envelope rule, that a failed gate withholds the windows CSV, and the S3-comparison freeze criterion.
- Test: `tests/test_swarm_provenance.py` (create) and the slow e2e appended to `tests/test_swarm_members.py`

- [ ] **Step 1: Failing tests**

```python
# tests/test_swarm_provenance.py
import types
def test_method_settings_record_swarm_keys():
    from gareus.provenance import _method_settings
    a = types.SimpleNamespace(swarm_stage="run", swarm_seed_ns=1.0, swarm_replicates_per_cell=3, swarm_bins="4,3,3",
                              swarm_budget_ns=None, swarm_output_interval_ps=2.0, shared_gamd_setup_dir="x/y")
    ms = _method_settings(a)
    for k in ("swarm_stage", "swarm_seed_ns", "swarm_replicates_per_cell", "swarm_bins", "swarm_output_interval_ps", "shared_gamd_setup_dir"):
        assert k in ms
```

and, marked slow (the repo uses a `slow` marker in `pyproject.toml`; with the fallback runner, guard the body with `if os.environ.get("GAREUS_RUN_SLOW") != "1": return`):

```python
def test_tiny_real_swarm_two_members_produce_traces_and_envelope():
    import os, pathlib, tempfile, subprocess, sys, json
    if os.environ.get("GAREUS_RUN_SLOW") != "1":
        return
    out = pathlib.Path(tempfile.mkdtemp())
    # Reference/CPU platform, dipeptide GA, 2 members × (2 ps equil + 4 ps prod), 0.5 ps output → 8 frames each.
    seed_dir = out / "seeds"; seed_dir.mkdir()
    # a 2-row survivor CSV pointing at two copies of the built peptide PDB is produced by the test from
    # gareus.system_setup.build_peptide_pdb("GA", ...) with two different (phi, psi) — no native reference involved.
    ...
    cmd = [sys.executable, "-m", "gareus", "--seq", "GA", "--out", str(out), "--platform", "CPU", "--setup-platform", "CPU",
           "--run-mode", "cmd", "--timestep-fs", "2.0", "--padding-nm", "0.6", "--swarm-stage", "run", "--seed-conformers-dir", str(seed_dir),
           "--swarm-replicates-per-cell", "1", "--swarm-bins", "2,1,1", "--swarm-seed-ns", "0.004", "--swarm-equil-ps", "2",
           "--swarm-output-interval-ps", "0.5", "--swarm-seed-frame-interval-ps", "1.0", "--gamd-boost-type", "pep-gamd-lower-dual",
           "--tui-mode", "none", "--progress-mode", "none"]
    subprocess.run(cmd, check=True, timeout=1800)
    subprocess.run(cmd[:cmd.index("--swarm-stage") + 1] + ["analyze"] + cmd[cmd.index("--swarm-stage") + 2:], check=True, timeout=600)
    rep = json.load((out / "swarm" / "analysis" / "swarm_report.json").open())
    assert rep["n_members_done"] == 2
    assert (out / "swarm" / "analysis" / "shared_gamd_setup" / "shared_gamd_setup_globals.json").exists()
```

  (Fill the `...` with the survivor-CSV construction: call `build_peptide_pdb("GA", seed_dir/"a.pdb", phi_deg=-60, psi_deg=-45)` and `build_peptide_pdb("GA", seed_dir/"b.pdb", phi_deg=-120, psi_deg=130)`, write `final_survivor_seeds.csv` with header `survivor_pdb_path` and the two paths. The gate will report `fail` on 2 members — assert only the artefacts above and `rep["status"] in ("pass","fail")`.)

- [ ] **Step 2: RED. Step 3: Implement** provenance keys, helptext section, `CLAUDE.md` entry. Run the slow test once with `GAREUS_RUN_SLOW=1` via the fallback recipe; paste its runtime into the task report.
- [ ] **Step 4: GREEN. Step 5: Commit** — `docs(swarm): provenance keys, helptext section 18, handoff notes; slow tiny e2e`.

---

### Task 12: Hand-off to S2/S3/S4 (documentation task, no code)

- [ ] Write `docs/superpowers/plans/2026-09-07-swarm-stage-handoff.md` (≤ 60 lines) with the exact command sequence for chignolin on aurum:
  1. `gareus --config examples/chignolin_swarm_stage.yaml --seed-conformers-dir <r7 dir> --out <run>/` (round 0; print of `n_cells, R, n_members, budget_ns`).
  2. `… --swarm-stage analyze` → read `swarm_report.json`; on `fail`, `… --swarm-stage run --swarm-replicates-per-cell <R + extra>` then analyze again (extension loop, confined to S1).
  3. `… --swarm-stage compare --swarm-pilot-globals <S3 pilot>/shared_gamd_setup_globals.json` → `freeze_allowed` must be true.
  4. Production: `gareus --config <campaign>.yaml --config <run>/swarm/analysis/ladder_run_args.yaml` (or the equivalent flags: `--windows-2d-csv <run>/swarm/analysis/windows_lambda_ladder.csv --seed-conformers-dir <run>/swarm/analysis/seed_bank --seed-selection-mode active-cv --us-seed-preflight-max-score 1.2 --shared-gamd-setup-dir <run>/swarm/analysis/shared_gamd_setup` plus the chignolin_7 pull settings) with `gamd_boost_type: pep-gamd-lower-dual`, `exchange_mode: gibbs-walk`. State in the doc that **every window start goes through the seed graft + pull** (cli.py:471/473/477/478) — a `windows_2d_csv` run without `--seed-conformers-dir` will fail the US start-quality gate on every contact window (S3 pilot, 2026-09-07). If `--config` cannot be repeated, merge the sidecar into the campaign YAML by hand and say so.
  5. Later rounds for S5 re-seeding: `--swarm-round 1 --swarm-seed-source production-frames --swarm-production-seed-csv <frames.csv>`; the envelope is **not** refitted.
  State explicitly which spec numbers each output replaces (§3.5 M and spacing ← `ladder_design.json`; §12 items 1–3 ← `ladder_design.json`, `plan_meta.json`, `envelope_discard.json`).
- [ ] Commit — `docs(swarm): hand-off command sequence from swarm outputs to the ladder campaign`.

---

## Self-review

**Spec coverage.** §2 S0 → Task 1 (+ Task 6 plan). §2 S1 products: ① envelopes → Task 2; ② CV1 → 16 centres → Task 3 `cv1_centers_from_samples`; ③ curvature → k → Task 3; ④ ΔV_max → rung spacing → Task 3 (`design_lambda_ladder`, §8 item 8); ⑤ seed frames per (window, rung) → Task 4 (rung-independent by construction). §2.1 (aux force present, boost off) → Task 5. §3.5/§12.1 → Task 3 + Task 9 (S3 confirms). §3.6 frozen envelope → Tasks 2, 6, 8 (round ≥ 1 never refits; out-of-envelope fraction reported). §5 seeds path via `--seed-conformers-dir`, stratification, 1 ns after a separate equilibration, tail-only pooling with a discard read off the V-trace, 2 ps interval in ps **and** frames (`envelope_discard.json`, `trace.csv` `frame,t_ps`) → Tasks 1, 2, 5, 8, 10. §7 gates (swarm-stage ones) → Task 7; the S3→S4 acceptance gate stays with the pilot. §8 item 7 → Tasks 1–8; item 8 → Task 3. §11 provenance → Task 11 (`GENPEPT_turbo_summary.json` hash belongs in `_input_file_hashes`; Task 11 must add the seed dir's summary JSON there). §12.2 R → Task 1 (`plan_members`); §12.3 discard → Task 2. User decisions: ab initio (Global Constraints + Task 10 header sentence + Task 11 helptext sentence); equal quota, budget-derived count, 1 ns (Task 1, Task 10 defaults); swarms recur with frozen coordinate (Task 6 frozen edges, Task 8 round ≥ 1); gate failure → extend/re-fit/re-gate (Tasks 7, 8, 12); pilot comparison with a stated tolerance (Task 9).

**Placeholders.** Task 11's slow test has one `...` that the step text fills in explicitly (two `build_peptide_pdb` calls + a two-row CSV). Task 5's third test names a fixture helper to grep for and says what to do if it is absent. No TBD/TODO.

**Type consistency.** `SeedDescriptor(seed_id, pdb_path, cv1, rg_nm, e2e_nm)` used identically in Tasks 1, 6. `plan_members` row keys match `plan.csv` header in Task 6/8. `TRACE_COLUMNS` from Task 5 is the header Task 8's fake writes. `pool_member_envelopes(traces, discard)` takes `{member_id: {"v_pep_kj","v_dih_kj"}}` in Tasks 2, 7, 8. `design_lambda_ladder` return keys (`lambdas, sigma_kj_per_rung, ess_per_rung, extrapolated_from_rung`) are the ones Task 7's `ladder_ess_gate` reads. `write_ladder_windows_csv(path, centers, ks_kcal, lambdas)` header equals the one `load_explicit_2d_window_csv` parses (verified in Task 3's test). Envelope JSON keys match `PepGamdEnvelope.from_integrator_globals` (`Vmax_Total … k0_Dihedral`) and `load_reusable_shared_gamd_setup` (`all_globals`, `interesting_globals`).

**Seed prerequisite (coordinator constraint, S3 pilot 2026-09-07).** No task starts MD from the extended chain: Task 5 fails the member on graft fallback, Task 6 refuses to run without the library, Task 7 gates the failure fraction, Task 8 excludes failed members and ships `ladder_run_args.yaml` with the seed flags, Tasks 10/12 carry them into the example config and the hand-off. Descriptors: heavy-CV1 computed (no fraction column exists in the library — verified in `GENPEPT.py`), `rg_nm`/`end_to_end_nm` read and cross-checked (Task 1).

**Spec gaps the plan could not resolve (left as open items, not guessed):**
1. Production's application of a physics-only `all_globals` dict (Task 2 Step 4 verifies and, if needed, patches) — the spec assumes the envelope "is copied to every replica" without saying which keys.
2. The spec does not say how many seed frames per window S2 needs; `--swarm-seeds-per-window 3` is a default to be confirmed by the S3 pilot's seeding behaviour.
3. The spec's S3→S4 acceptance gate (≥ 0.2 per λ edge) is measured by the pilot, not the swarm; `--swarm-target-beta-sigma 1.0` is the design knob that should reproduce it, and only the pilot can confirm.
