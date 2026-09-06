# λ-Ladder Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make GaREUS run a grid of (CV1 umbrella window × GaMD boost strength λ) thermodynamic states with Pep-GaMD, exchange over the full grid, store what MBAR needs to reweight every sample to (λ=0, w=0) exactly, and cross-check the full-ladder PMF against the λ=0 rungs alone.

**Architecture:** A state is (window *i*, rung *k*). The existing exchange machinery already treats "window" as "thermodynamic state" and consumes one `bias_matrix_kj[state, replica]`; the ladder adds one term to that matrix — the Pep-GaMD boost energy of replica *r*'s configuration under state *s*'s λ — and sets the replica's integrator `k0` globals whenever its state changes. Because every rung shares one frozen envelope, the boost under any λ is a closed-form function of two stored per-sample energies (`v_pep`, `v_dih`), so the MBAR loader reconstructs `u_nk` for all states offline. The λ=0 rungs are plain umbrella sampling and give the cross-check.

**Tech Stack:** Python 3.14, OpenMM 8.5.1 (`CustomIntegrator`), gamd-openmm 0.9.2, numpy, pyarrow (Parquet samples), pymbar via `gareus.mbar_analysis`.

**Spec:** `docs/superpowers/specs/2026-09-06-chignolin-lambda-ladder-pep-gamd-design.md` (§3 ladder, §4 Pep-GaMD, §6 analysis, §7 gates, §8 items 2–6, 9, 10). The swarm stage (§5, §8 items 7–8) is a separate plan.

## Global Constraints

- Base branch: **`feat/pep-gamd-boost`** (Pep-GaMD integrator, partition, wiring, 22 tests). Branch the ladder work from it: `git worktree add -b feat/lambda-ladder .claude/worktrees/lambda-ladder feat/pep-gamd-boost`. Keep worktrees **inside** the repo tree (the sanctioned test runner rejects outside paths).
- Force groups are fixed by Pep-GaMD: 0 physical (NB + bonds + angles), 1 auxiliary water-only NB, 2 torsions, 29 secondary CV, 31 umbrella. Never put a `Custom*` force in 0–2.
- Boost type string: `pep-gamd-lower-dual` (`gareus.pep_gamd.PEP_GAMD_BOOST_TYPE`). Lower-bound formula only: `threshold_energy = Vmax`, `k = k0/(Vmax−Vmin)`.
- Envelope (Vmax, Vmin, threshold, k0_max per channel) is **frozen** after calibration for the whole campaign. Never recalibrate mid-campaign.
- Per-rung integrator globals: `k0_Total = λ·k0_max_Total`, `k0_Dihedral = λ·k0_max_Dihedral`. Nothing else changes between rungs.
- Exchange mode `gibbs-walk` over the **full** state set; never truncate the neighbour graph.
- Every sample stores raw `v_pep_kj_mol`, `v_dih_kj_mol`, `gamd_lambda` and a physical `potential_kj_mol` (group 1 excluded).
- **Test runner:** a repo hook rejects any Bash command containing the literal name of the Python test runner. Run tests as `opencode run "In <worktree>: run the project's standard test runner on tests/<file>.py and report pass/fail counts and failing test ids"`. If opencode does not launch a test process within a few minutes (it hung twice on 2026-09-06), the red/green loop may use this fixture-free fallback (cannot run tests that take fixtures such as `capsys`):
  ```bash
  python3 -c "import importlib.util,sys,traceback;sys.path[:0]=['.','tests'];p=sys.argv[1];s=importlib.util.spec_from_file_location('t',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);f=0
  for n in [n for n in dir(m) if n.startswith('test_')]:
      try: getattr(m,n)(); print('PASS',n)
      except Exception: f+=1; print('FAIL',n); traceback.print_exc(limit=4)
  sys.exit(1 if f else 0)" tests/test_lambda_ladder_boost.py
  ```
- gamd-openmm's first `step()` of a fresh Context moves no atom. Any test comparing trajectories or reading `BoostPotential_*` after one step must warm up one step, re-seat positions/velocities, then measure.
- Do **not** call `getPMEParametersInContext` on the CPU platform in tests; use the Reference platform.
- Commit after every green task with a conventional-commit message; no Co-Authored-By trailer (attribution is disabled in this repo).

---

## File Structure

| file | responsibility |
|---|---|
| `gareus/pep_gamd.py` (modify) | add `PepGamdEnvelope`, `pep_gamd_boost_kj`, `pep_gamd_boost_matrix_kj`, `set_replica_lambda` — the closed-form boost and the per-rung integrator hook |
| `gareus/adaptive_production.py:144` `WindowState` (modify) | add `gamd_lambda: float = 0.0`; union MBAR inputs gain the boost term |
| `gareus/windows.py:1173` `load_explicit_2d_window_csv` (modify) | parse optional `gamd_lambda` column into `meta["gamd_lambdas"]` and `normalized_rows` |
| `gareus/production.py` (modify) | thread `state_lambdas` + envelope; set λ at replica build and on every accepted swap; add the boost term to the exchange/log bias matrix; write new sample columns |
| `gareus/store.py` (modify) | `ParquetSampleWriter.write_sample` gains `v_pep`, `v_dih`, `gamd_lambda`; schema columns |
| `gareus/query.py:253` `reconstruct_bias_matrix` (modify) | optional boost term from raw energies |
| `gareus/mbar_analysis/loaders.py` (modify) | load the new columns; mark `meta["gamd_ladder"]` |
| `gareus/mbar_analysis/crosscheck.py` (create) | λ=0-only PMF vs full-ladder PMF |
| `gareus/system_setup.py:467` (modify) | replace the tautological PBC guard |
| `gareus/helptext.py`, `CLAUDE.md` (modify) | ladder section; handoff notes |
| `tests/test_lambda_ladder_boost.py` (create) | closed-form boost vs integrator oracle; envelope; λ hook |
| `tests/test_lambda_ladder_states.py` (create) | registry/CSV/rows; sample writer; exchange kernel with boost term |
| `tests/test_lambda_ladder_mbar.py` (create) | `u_nk` reconstruction; cross-check |
| `tests/test_box_audit_guard.py` (create) | the guard fires on an undersized box and stays silent on an adequate one |

---

### Task 1: Closed-form Pep-GaMD boost from raw energies, verified against the integrator

**Files:**
- Modify: `gareus/pep_gamd.py` (append)
- Test: `tests/test_lambda_ladder_boost.py`

**Interfaces:**
- Consumes: `gareus.pep_gamd.PepGaMDLowerDualIntegrator`, `ensure_pep_gamd_partition`, `peptide_essential_energy_kj` (all on `feat/pep-gamd-boost`); fixture `tests/pep_gamd_fixture.py: solvated_dipeptide(), _fresh_system()`.
- Produces:
  ```python
  @dataclass(frozen=True)
  class PepGamdEnvelope:
      vmax_total: float; vmin_total: float; threshold_total: float; k0max_total: float
      vmax_dih: float;   vmin_dih: float;   threshold_dih: float;   k0max_dih: float
      @classmethod
      def from_integrator_globals(cls, g: dict) -> "PepGamdEnvelope": ...   # keys Vmax_Total, Vmin_Total, threshold_energy_Total, k0_Total, and _Dihedral
      @classmethod
      def from_json(cls, path) -> "PepGamdEnvelope": ...                    # shared_gamd_setup_globals.json
  def pep_gamd_boost_kj(v_pep_kj, v_dih_kj, lam, env) -> float | np.ndarray   # total boost ΔV_pep + ΔV_dih under rung λ, vectorised over samples
  def pep_gamd_boost_matrix_kj(v_pep_kj, v_dih_kj, lambdas, env) -> np.ndarray  # shape (len(lambdas), len(v_pep)) — states × samples
  ```

The formula is gamd-openmm's dependent dual boost (stage_integrator `_do_boost_updates`, `_add_dihedral_boost_to_total_energy`, base_integrator `_add_gamd_pre_calc_step`), with `k0_c = λ·k0max_c`:

```
def _one_channel(V, E, Vmax, Vmin, k0):
    rng = Vmax - Vmin
    scale = max(abs(E), abs(V), 1.0)               # energy_scale
    if abs(rng) <= 0.001*scale: return 0.0         # degenerate guard
    b = 0.5*k0*(E - V)**2 / rng
    return b if (b + V) < E else 0.0               # step(E - (b + V))
boost_dih = _one_channel(v_dih, E_dih, Vmax_dih, Vmin_dih, λ·k0max_dih)
boost_tot = _one_channel(v_pep + boost_dih, E_tot, Vmax_tot, Vmin_tot, λ·k0max_tot)   # dihedral boost is INSIDE the Total square
return boost_dih + boost_tot
```

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_lambda_ladder_boost.py
import json, types
import numpy as np
from gareus.imports import import_openmm
from pep_gamd_fixture import solvated_dipeptide, _fresh_system


def _env():
    from gareus.pep_gamd import PepGamdEnvelope
    return PepGamdEnvelope(vmax_total=50.0, vmin_total=-50.0, threshold_total=50.0, k0max_total=0.8,
                           vmax_dih=50.0, vmin_dih=-50.0, threshold_dih=50.0, k0max_dih=0.6)


def test_boost_is_zero_at_lambda_zero_and_monotone_in_lambda():
    from gareus.pep_gamd import pep_gamd_boost_kj
    env = _env()
    assert pep_gamd_boost_kj(10.0, 5.0, 0.0, env) == 0.0
    b = [pep_gamd_boost_kj(10.0, 5.0, lam, env) for lam in (0.1, 0.25, 0.5, 1.0)]
    assert all(x > 0 for x in b) and b == sorted(b)


def test_boost_vanishes_above_threshold():
    from gareus.pep_gamd import pep_gamd_boost_kj
    env = _env()
    assert pep_gamd_boost_kj(60.0, 60.0, 1.0, env) == 0.0   # both channels above E = Vmax


def test_boost_matrix_shape_and_rows():
    from gareus.pep_gamd import pep_gamd_boost_kj, pep_gamd_boost_matrix_kj
    env = _env()
    v_pep = np.array([10.0, 20.0, 30.0]); v_dih = np.array([5.0, 6.0, 7.0]); lams = np.array([0.0, 0.5, 1.0])
    M = pep_gamd_boost_matrix_kj(v_pep, v_dih, lams, env)
    assert M.shape == (3, 3)
    assert np.allclose(M[0], 0.0)
    assert np.isclose(M[2, 1], pep_gamd_boost_kj(20.0, 6.0, 1.0, env))


def test_envelope_from_integrator_globals_roundtrip():
    from gareus.pep_gamd import PepGamdEnvelope
    g = {"Vmax_Total": 50.0, "Vmin_Total": -50.0, "threshold_energy_Total": 50.0, "k0_Total": 0.8,
         "Vmax_Dihedral": 40.0, "Vmin_Dihedral": -40.0, "threshold_energy_Dihedral": 40.0, "k0_Dihedral": 0.6}
    env = PepGamdEnvelope.from_integrator_globals(g)
    assert (env.vmax_total, env.k0max_dih, env.threshold_dih) == (50.0, 0.6, 40.0)


def test_closed_form_matches_the_integrator_at_lambda_one_and_half():
    """Oracle: the integrator computes BoostPotential_* every step. Force a production stage
    with a known envelope and compare."""
    from gareus import pep_gamd
    openmm, _app, unit = import_openmm()
    fx = solvated_dipeptide()
    for lam in (1.0, 0.5):
        system = _fresh_system()
        pep_gamd.ensure_pep_gamd_partition(system, fx["peptide"])
        integ = pep_gamd.PepGaMDLowerDualIntegrator(
            pep_gamd.DIHEDRAL_GROUP, dt=0.002 * unit.picoseconds, ntcmdprep=2, ntcmd=4, ntebprep=2, nteb=4,
            nstlim=100, ntave=2, sigma0p=6.0 * unit.kilocalories_per_mole, sigma0d=6.0 * unit.kilocalories_per_mole,
            collision_rate=1.0 / unit.picoseconds, temperature=300.0 * unit.kelvin)
        ctx = openmm.Context(system, integ, openmm.Platform.getPlatformByName("Reference"))
        ctx.setPositions(fx["positions"]); ctx.setVelocitiesToTemperature(300.0 * unit.kelvin, 3)
        integ.step(1)                                    # first step of a fresh Context is a no-op move
        ctx.setPositions(fx["positions"])
        v_pep = pep_gamd.peptide_essential_energy_kj(ctx, unit)
        v_dih = ctx.getState(getEnergy=True, groups={pep_gamd.DIHEDRAL_GROUP}).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        env = pep_gamd.PepGamdEnvelope(vmax_total=v_pep + 200.0, vmin_total=v_pep - 200.0, threshold_total=v_pep + 200.0, k0max_total=0.8,
                                       vmax_dih=v_dih + 100.0, vmin_dih=v_dih - 100.0, threshold_dih=v_dih + 100.0, k0max_dih=0.6)
        for k, v in {"stepCount": 50, "stage": 5,
                     "k0_Total": lam * env.k0max_total, "Vmax_Total": env.vmax_total, "Vmin_Total": env.vmin_total, "threshold_energy_Total": env.threshold_total,
                     "k0_Dihedral": lam * env.k0max_dih, "Vmax_Dihedral": env.vmax_dih, "Vmin_Dihedral": env.vmin_dih, "threshold_energy_Dihedral": env.threshold_dih}.items():
            integ.setGlobalVariableByName(k, v)
        integ.step(1)
        got = integ.getGlobalVariableByName("BoostPotential_Total") + integ.getGlobalVariableByName("BoostPotential_Dihedral")
        want = pep_gamd.pep_gamd_boost_kj(v_pep, v_dih, lam, env)
        assert abs(got - want) < 1e-3, (lam, got, want)
```

- [ ] **Step 2: Run to verify RED** — expected: `ImportError`/`AttributeError` for `PepGamdEnvelope`, `pep_gamd_boost_kj`.

- [ ] **Step 3: Implement** (append to `gareus/pep_gamd.py`)

```python
from dataclasses import dataclass
import json as _json
import numpy as _np


@dataclass(frozen=True)
class PepGamdEnvelope:
    """Frozen per-channel GaMD envelope; k0max_* is the top rung (λ = 1)."""
    vmax_total: float; vmin_total: float; threshold_total: float; k0max_total: float
    vmax_dih: float;   vmin_dih: float;   threshold_dih: float;   k0max_dih: float

    @classmethod
    def from_integrator_globals(cls, g: dict) -> "PepGamdEnvelope":
        f = lambda k: float(g[k])
        return cls(f("Vmax_Total"), f("Vmin_Total"), f("threshold_energy_Total"), f("k0_Total"),
                   f("Vmax_Dihedral"), f("Vmin_Dihedral"), f("threshold_energy_Dihedral"), f("k0_Dihedral"))

    @classmethod
    def from_json(cls, path) -> "PepGamdEnvelope":
        doc = _json.loads(open(path).read())
        for cand in (doc, doc.get("globals"), doc.get("integrator_globals"), doc.get("shared_gamd_globals_all")):
            if isinstance(cand, dict) and "k0_Total" in cand:
                return cls.from_integrator_globals(cand)
        raise KeyError(f"{path}: no dict with k0_Total/Vmax_Total/... found")


def _channel_boost(v, e, vmax, vmin, k0):
    v = _np.asarray(v, dtype=float)
    rng = vmax - vmin
    scale = _np.maximum(_np.maximum(abs(e), _np.abs(v)), 1.0)
    b = 0.5 * k0 * (e - v) ** 2 / rng
    b = _np.where(_np.abs(rng) <= 0.001 * scale, 0.0, b)
    return _np.where((b + v) < e, b, 0.0)


def pep_gamd_boost_kj(v_pep_kj, v_dih_kj, lam, env: PepGamdEnvelope):
    """gamd-openmm's dependent dual boost under rung λ: dihedral first, then Total with the
    dihedral boost added to the Total energy before the square (stage_integrator
    _add_dihedral_boost_to_total_energy)."""
    lam = float(lam)
    b_dih = _channel_boost(v_dih_kj, env.threshold_dih, env.vmax_dih, env.vmin_dih, lam * env.k0max_dih)
    b_tot = _channel_boost(_np.asarray(v_pep_kj, dtype=float) + b_dih, env.threshold_total, env.vmax_total, env.vmin_total, lam * env.k0max_total)
    out = b_dih + b_tot
    return float(out) if out.ndim == 0 else out


def pep_gamd_boost_matrix_kj(v_pep_kj, v_dih_kj, lambdas, env: PepGamdEnvelope) -> _np.ndarray:
    """(n_states, n_samples): boost of each sample's configuration under each state's λ."""
    v_pep = _np.asarray(v_pep_kj, dtype=float); v_dih = _np.asarray(v_dih_kj, dtype=float)
    return _np.vstack([_np.asarray(pep_gamd_boost_kj(v_pep, v_dih, float(l), env), dtype=float) for l in lambdas])
```

- [ ] **Step 4: Run to verify GREEN.** The oracle test must pass at both λ values; if it fails only at λ=0.5, the dependent structure (dihedral boost inside the Total square) is wrong — do not loosen the tolerance.

- [ ] **Step 5: Amend the spec** — in `docs/superpowers/specs/2026-09-06-chignolin-lambda-ladder-pep-gamd-design.md` §3.1 replace the sentence "every rung's boost is a scalar rescale of one reference" with: "each channel's boost is a scalar rescale of one reference; because the dual boost is *dependent* (the dihedral boost enters the Total square), the total boost under λ is the closed form `pep_gamd_boost_kj(v_pep, v_dih, λ, envelope)`, reconstructible exactly from the two stored raw energies." Keep the `u_ik` line.

- [ ] **Step 6: Commit**

```bash
git add gareus/pep_gamd.py tests/test_lambda_ladder_boost.py docs/superpowers/specs/2026-09-06-chignolin-lambda-ladder-pep-gamd-design.md
git commit -m "feat: closed-form Pep-GaMD boost under any rung λ, verified against the integrator"
```

---

### Task 2: The rung dimension in the state model (registry, explicit CSV, assignment rows)

**Files:**
- Modify: `gareus/adaptive_production.py:144-183` (`WindowState`)
- Modify: `gareus/windows.py:1173` (`load_explicit_2d_window_csv`), and the `normalized_rows.append({...})` block near `windows.py:1285`
- Modify: `gareus/production.py` `window_assignment_rows` (defined right after `make_gamd_integrator`, ~line 1665)
- Test: `tests/test_lambda_ladder_states.py`

**Interfaces:**
- Produces: `WindowState.gamd_lambda: float` (default 0.0, round-trips through `to_dict`/`from_dict`); `load_explicit_2d_window_csv` returns `meta["gamd_lambdas"]: list[float]` (zeros when the column is absent) and each normalized row carries `"gamd_lambda"`; `window_assignment_rows(..., gamd_lambdas=None)` writes a `gamd_lambda` column.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_lambda_ladder_states.py
import csv, io, types, tempfile, pathlib
import numpy as np


def test_window_state_carries_gamd_lambda_and_roundtrips():
    from gareus.adaptive_production import WindowState
    s = WindowState(state_id=3, primary_center=0.2, primary_k=500.0, gamd_lambda=0.25)
    d = s.to_dict()
    assert d["gamd_lambda"] == 0.25
    assert WindowState.from_dict(d).gamd_lambda == 0.25
    assert WindowState.from_dict({"state_id": 1, "primary_center": 0.1, "primary_k": 1.0}).gamd_lambda == 0.0
    assert WindowState.from_dict({"state_id": 1, "primary_center": 0.1, "primary_k": 1.0, "gamd_lambda": ""}).gamd_lambda == 0.0


def _write_csv(rows, header):
    d = pathlib.Path(tempfile.mkdtemp()); p = d / "windows.csv"
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header); w.writeheader(); [w.writerow(r) for r in rows]
    return p


def _args():
    return types.SimpleNamespace(primary_cv="contacts", secondary_cv="none",
                                 explicit_2d_primary_center_column="primary_cv_center",
                                 explicit_2d_primary_k_column="primary_cv_k_kcal",
                                 explicit_2d_secondary_center_column="secondary_cv_center",
                                 explicit_2d_secondary_k_column="secondary_cv_k_kcal_mol",
                                 explicit_2d_primary_cv_mode_column="primary_cv_mode",
                                 explicit_2d_secondary_cv_mode_column="secondary_cv_mode",
                                 explicit_2d_window_schema="generic")


def test_explicit_csv_parses_gamd_lambda_column():
    from gareus.windows import load_explicit_2d_window_csv
    p = _write_csv([{"primary_cv_center": 0.1, "primary_cv_k_kcal": 100, "gamd_lambda": 0.0},
                    {"primary_cv_center": 0.1, "primary_cv_k_kcal": 100, "gamd_lambda": 0.5},
                    {"primary_cv_center": 0.3, "primary_cv_k_kcal": 100, "gamd_lambda": 1.0}],
                   ["primary_cv_center", "primary_cv_k_kcal", "gamd_lambda"])
    centers, ks, sec_c, sec_k, meta, *_rest = load_explicit_2d_window_csv(_args(), p)
    assert list(meta["gamd_lambdas"]) == [0.0, 0.5, 1.0]
    assert [r["gamd_lambda"] for r in meta["normalized_rows"]] == [0.0, 0.5, 1.0]
    assert len(centers) == 3


def test_explicit_csv_without_gamd_lambda_defaults_to_zero():
    from gareus.windows import load_explicit_2d_window_csv
    p = _write_csv([{"primary_cv_center": 0.1, "primary_cv_k_kcal": 100}], ["primary_cv_center", "primary_cv_k_kcal"])
    *_x, meta, *_rest = load_explicit_2d_window_csv(_args(), p)
    assert list(meta["gamd_lambdas"]) == [0.0]


def test_explicit_csv_rejects_lambda_outside_unit_interval():
    from gareus.windows import load_explicit_2d_window_csv
    p = _write_csv([{"primary_cv_center": 0.1, "primary_cv_k_kcal": 100, "gamd_lambda": 1.5}],
                   ["primary_cv_center", "primary_cv_k_kcal", "gamd_lambda"])
    try:
        load_explicit_2d_window_csv(_args(), p)
    except ValueError as exc:
        assert "gamd_lambda" in str(exc)
    else:
        raise AssertionError("λ outside [0, 1] must be rejected")


def test_window_assignment_rows_write_gamd_lambda():
    from gareus.production import window_assignment_rows
    rows = window_assignment_rows(np.array([0.1, 0.1]), [100.0, 100.0], 300.0, gamd_lambdas=[0.0, 1.0])
    assert [r["gamd_lambda"] for r in rows] == [0.0, 1.0]
    rows = window_assignment_rows(np.array([0.1]), [100.0], 300.0)
    assert rows[0]["gamd_lambda"] == 0.0
```

- [ ] **Step 2: Run to verify RED** — `TypeError: unexpected keyword 'gamd_lambda'`, `KeyError: 'gamd_lambdas'`, `TypeError: unexpected keyword 'gamd_lambdas'`.

- [ ] **Step 3: Implement**

`gareus/adaptive_production.py` — in `WindowState` add after `gamd_sigma0d`:
```python
    gamd_lambda: float = 0.0
```
and in `from_dict`, after the `for key in (...)` loop that maps `""`/`None` to `None`, add:
```python
        lam = data.get("gamd_lambda")
        data["gamd_lambda"] = 0.0 if lam in ("", "None", None) else float(lam)
```
(Check `to_dict` already serialises via `asdict` — it does, nothing to add.)

`gareus/windows.py` `load_explicit_2d_window_csv` — inside the `for offset, row in enumerate(rows, start=2):` loop, alongside the primary/secondary parsing, add:
```python
        lam_raw = row.get("gamd_lambda", "")
        lam = 0.0 if lam_raw in ("", None) else float(lam_raw)
        if not (0.0 <= lam <= 1.0):
            raise ValueError(f"--windows-2d-csv row {offset}: gamd_lambda={lam} must lie in [0, 1]")
        gamd_lambdas.append(lam)
```
with `gamd_lambdas: list[float] = []` declared next to `centers_a`/`k_list` before the loop; add `"gamd_lambda": float(lam),` to the `normalized_rows.append({...})` dict; and before the function's `return`, put `meta["gamd_lambdas"] = gamd_lambdas` and `meta["normalized_rows"] = normalized_rows` (the latter if not already present — grep the function's return statement to see which dict it returns and add both keys to it).

`gareus/production.py` `window_assignment_rows(centers_a, k_list, temperature_k, secondary_centers=None, secondary_k_list=None, args=None)` — add parameter `gamd_lambdas=None` and, in the per-window row dict, `"gamd_lambda": float(gamd_lambdas[i]) if gamd_lambdas is not None else 0.0`.

- [ ] **Step 4: Run to verify GREEN.**

- [ ] **Step 5: Thread the lambdas into `run_gareus`.** At the single call site of `load_explicit_2d_window_csv` in `gareus/production.py` (grep `load_explicit_2d_window_csv(`), after unpacking, add `args.state_gamd_lambdas = list(meta.get("gamd_lambdas") or [0.0] * len(centers_a))`. In `run_gareus`, right after `centers_a`/`k_list` are final and `nrep` is known, add:
```python
    state_lambdas = np.asarray(getattr(args, "state_gamd_lambdas", None) or [0.0] * nrep, dtype=float)
    if state_lambdas.size != nrep:
        raise ValueError(f"state_gamd_lambdas has {state_lambdas.size} entries for {nrep} states")
    ladder_active = bool(np.any(state_lambdas > 0.0))
    if ladder_active and not is_pep_gamd(args):
        raise ValueError("a gamd_lambda ladder requires --gamd-boost-type pep-gamd-lower-dual")
```
Pass `gamd_lambdas=state_lambdas.tolist()` where `window_assignment_rows(...)` is called for the run's `umbrella_windows.csv`.

- [ ] **Step 6: Commit**
```bash
git add gareus/adaptive_production.py gareus/windows.py gareus/production.py tests/test_lambda_ladder_states.py
git commit -m "feat: gamd_lambda rung dimension in WindowState, explicit window CSV and assignment rows"
```

---

### Task 3: Per-rung integrator globals — at replica build and on every accepted swap

**Files:**
- Modify: `gareus/pep_gamd.py` (append `set_replica_lambda`, `k0max_from_globals`)
- Modify: `gareus/production.py` `_build_context_i` (the `set_integrator_globals_from_dict(integrator_i, shared_gamd_globals_all)` call, ~line 5597) and `_attempt_window_swap` (~line 6427, the two `set_window(...)` calls after `if outcome.accepted:`)
- Test: `tests/test_lambda_ladder_boost.py` (append)

**Interfaces:**
- Produces:
  ```python
  def k0max_from_globals(shared_globals: dict) -> dict[str, float]      # {"Total": k0_Total, "Dihedral": k0_Dihedral}
  def set_replica_lambda(integrator, lam: float, k0max: dict[str, float]) -> None   # sets k0_Total, k0_Dihedral = λ·k0max
  ```

- [ ] **Step 1: Write the failing tests**

```python
def test_set_replica_lambda_scales_both_k0_globals():
    from gareus import pep_gamd
    openmm, _app, unit = import_openmm()
    fx = solvated_dipeptide(); system = _fresh_system()
    pep_gamd.ensure_pep_gamd_partition(system, fx["peptide"])
    integ = pep_gamd.PepGaMDLowerDualIntegrator(
        pep_gamd.DIHEDRAL_GROUP, dt=0.002 * unit.picoseconds, ntcmdprep=2, ntcmd=4, ntebprep=2, nteb=4, nstlim=100, ntave=2,
        sigma0p=6.0 * unit.kilocalories_per_mole, sigma0d=6.0 * unit.kilocalories_per_mole,
        collision_rate=1.0 / unit.picoseconds, temperature=300.0 * unit.kelvin)
    ctx = openmm.Context(system, integ, openmm.Platform.getPlatformByName("Reference"))
    k0max = {"Total": 0.8, "Dihedral": 0.6}
    pep_gamd.set_replica_lambda(integ, 0.25, k0max)
    assert abs(integ.getGlobalVariableByName("k0_Total") - 0.2) < 1e-12
    assert abs(integ.getGlobalVariableByName("k0_Dihedral") - 0.15) < 1e-12
    pep_gamd.set_replica_lambda(integ, 0.0, k0max)
    assert integ.getGlobalVariableByName("k0_Total") == 0.0 and integ.getGlobalVariableByName("k0_Dihedral") == 0.0


def test_k0max_from_globals_reads_both_channels():
    from gareus.pep_gamd import k0max_from_globals
    assert k0max_from_globals({"k0_Total": 0.8, "k0_Dihedral": 0.6, "Vmax_Total": 1.0}) == {"Total": 0.8, "Dihedral": 0.6}


def test_set_replica_lambda_rejects_out_of_range():
    from gareus import pep_gamd
    try:
        pep_gamd.set_replica_lambda(types.SimpleNamespace(setGlobalVariableByName=lambda *a: None), 1.2, {"Total": 1.0, "Dihedral": 1.0})
    except ValueError:
        pass
    else:
        raise AssertionError("λ > 1 must be rejected")
```

- [ ] **Step 2: Run to verify RED** — `AttributeError: set_replica_lambda`.

- [ ] **Step 3: Implement** (append to `gareus/pep_gamd.py`)

```python
def k0max_from_globals(shared_globals: dict) -> dict:
    return {"Total": float(shared_globals["k0_Total"]), "Dihedral": float(shared_globals["k0_Dihedral"])}


def set_replica_lambda(integrator, lam: float, k0max: dict) -> None:
    """Put a replica on rung λ: k0_c = λ·k0max_c for both channels. Nothing else differs between rungs."""
    lam = float(lam)
    if not (0.0 <= lam <= 1.0):
        raise ValueError(f"gamd_lambda={lam} must lie in [0, 1]")
    integrator.setGlobalVariableByName("k0_Total", lam * float(k0max["Total"]))
    integrator.setGlobalVariableByName("k0_Dihedral", lam * float(k0max["Dihedral"]))
```

- [ ] **Step 4: Wire it into production.**

In `run_gareus`, after the shared GaMD setup has produced `shared_gamd_globals_all` (the dict copied into replicas) and before the replica-building loop, add:
```python
    k0max_by_channel = k0max_from_globals(shared_gamd_globals_all) if (use_gamd and ladder_active) else None
```
In `_build_context_i`, immediately after `copied, copied_skipped = set_integrator_globals_from_dict(integrator_i, shared_gamd_globals_all)`, add:
```python
                    if k0max_by_channel is not None:
                        set_replica_lambda(integrator_i, float(state_lambdas[i]), k0max_by_channel)
```
In `_attempt_window_swap`, after the two `set_window(...)` calls inside `if outcome.accepted:`, add:
```python
                if k0max_by_channel is not None:
                    set_replica_lambda(sims[i].integrator, float(state_lambdas[assignments[i]]), k0max_by_channel)
                    set_replica_lambda(sims[j].integrator, float(state_lambdas[assignments[j]]), k0max_by_channel)
```
Also grep `production.py` for every other place `set_window(sims[` or `set_window(sim_i.context` is called with an assignment (fast-resume restore, epoch re-seeding: lines ~4312, ~4421, ~5630) and add the matching `set_replica_lambda` call right after each, guarded by `k0max_by_channel is not None`. Import `k0max_from_globals, set_replica_lambda` in the existing `from .pep_gamd import (...)` block.

- [ ] **Step 5: Run to verify GREEN**, then run `tests/test_pep_gamd_boost.py` and `tests/test_pep_gamd_wiring.py` to confirm nothing regressed.

- [ ] **Step 6: Commit**
```bash
git add gareus/pep_gamd.py gareus/production.py tests/test_lambda_ladder_boost.py
git commit -m "feat: per-rung k0 globals set at replica build and on every accepted exchange"
```

---

### Task 4: The boost term in the exchange bias matrix

**Files:**
- Modify: `gareus/production.py` — the log/sample path that builds `bias_matrix_kcal` (~lines 6130-6148, `distance_bias_matrix_kcal + ss_bias_matrix_kcal`) and `_current_exchange_arrays` (~line 6440), plus the two `_fetch_*` closures that read per-replica observables
- Test: `tests/test_lambda_ladder_states.py` (append)

**Interfaces:**
- Consumes: `pep_gamd_boost_matrix_kj`, `PepGamdEnvelope`, `peptide_essential_energy_kj`, `state_lambdas`, `apply_window_swap` (module-level exchange kernel already used by `_attempt_window_swap`).
- Produces: `bias_matrix_kj[s, r]` = umbrella + `pep_gamd_boost_kj(v_pep_r, v_dih_r, λ_s, env)`; `observable_cache["v_pep_kj"]`, `["v_dih_kj"]`, `["boost_bias_matrix_kj"]`.

- [ ] **Step 1: Write the failing test** (kernel-level; no MD)

```python
def test_exchange_delta_between_rungs_is_the_boost_difference():
    """Two states with identical umbrellas, λ=0 and λ=1, holding replicas 0 and 1.
    Swapping them costs exactly boost(x0;λ=1) - boost(x1;λ=1) (the λ=0 terms are zero)."""
    from gareus.production import apply_window_swap
    from gareus.pep_gamd import pep_gamd_boost_matrix_kj, PepGamdEnvelope
    env = PepGamdEnvelope(50.0, -50.0, 50.0, 0.8, 50.0, -50.0, 50.0, 0.6)
    lambdas = np.array([0.0, 1.0]); v_pep = np.array([10.0, 30.0]); v_dih = np.array([5.0, 7.0])
    umbrella_kj = np.zeros((2, 2))
    bias = umbrella_kj + pep_gamd_boost_matrix_kj(v_pep, v_dih, lambdas, env)
    assignments = np.array([0, 1]); replica_of_window = np.array([0, 1])
    out = apply_window_swap(bias, 1.0 / 2.494, assignments, replica_of_window, 0, 1, None, force_accept=True)
    expected = float(bias[1, 0] + bias[0, 1] - bias[0, 0] - bias[1, 1])
    assert abs(out.delta_kj - expected) < 1e-12
    assert out.accepted and list(assignments) == [1, 0]


def test_boost_matrix_term_is_zero_for_a_pure_umbrella_ladder():
    from gareus.pep_gamd import pep_gamd_boost_matrix_kj, PepGamdEnvelope
    env = PepGamdEnvelope(50.0, -50.0, 50.0, 0.8, 50.0, -50.0, 50.0, 0.6)
    M = pep_gamd_boost_matrix_kj(np.array([1.0, 2.0]), np.array([1.0, 2.0]), np.zeros(4), env)
    assert M.shape == (4, 2) and np.all(M == 0.0)
```
(`apply_window_swap`'s exact signature is at `gareus/production.py` — grep `def apply_window_swap`; adjust the positional call to match, keeping `force_accept=True` and no random draw.)

- [ ] **Step 2: Run to verify RED** — the first test fails only if `apply_window_swap` is missing/renamed; if both pass immediately, the kernel already does the right thing and the test stands as a regression guard — proceed.

- [ ] **Step 3: Implement the production wiring.**

In `run_gareus`, next to `ladder_active`, build the envelope once from the shared globals:
```python
    pep_env = PepGamdEnvelope.from_integrator_globals(shared_gamd_globals_all) if (use_gamd and ladder_active) else None
```
In `_fetch_state` (log/sample path) and `_fetch_exchange_state` (exchange path) extend the returned tuple with `v_pep` and `v_dih` when `pep_env is not None`:
```python
                v_pep = peptide_essential_energy_kj(sim.context, unit) if pep_env is not None else float("nan")
                v_dih = (sim.context.getState(getEnergy=True, groups={DIHEDRAL_GROUP}).getPotentialEnergy()
                         .value_in_unit(unit.kilojoule_per_mole)) if pep_env is not None else float("nan")
```
collect them into `v_pep_kj = np.empty(nrep)`, `v_dih_kj = np.empty(nrep)` alongside `primary_values`, and after `bias_matrix_kcal = distance_bias_matrix_kcal + ss_bias_matrix_kcal` (and the equivalent line in `_current_exchange_arrays`) add:
```python
            boost_bias_matrix_kj = (pep_gamd_boost_matrix_kj(v_pep_kj, v_dih_kj, state_lambdas, pep_env)
                                    if pep_env is not None else np.zeros((nrep, nrep)))
            bias_matrix_kj = 4.184 * bias_matrix_kcal + boost_bias_matrix_kj
            reduced_bias_matrix = float(beta) * bias_matrix_kj
```
(replace the existing `bias_matrix_kj = 4.184 * bias_matrix_kcal`). Store `observable_cache["v_pep_kj"] = v_pep_kj`, `["v_dih_kj"] = v_dih_kj`, `["boost_bias_matrix_kj"] = boost_bias_matrix_kj`. The per-sample `sampled_bias_kj = all_bias_kj[w]` now includes the boost, which is correct: it is the reduced-potential difference MBAR needs.

- [ ] **Step 4: Run GREEN**, plus the existing exchange-kernel tests (grep `tests/` for `apply_window_swap`, `gibbs_propose_one_replica`) to confirm no regression.

- [ ] **Step 5: Commit**
```bash
git add gareus/production.py tests/test_lambda_ladder_states.py
git commit -m "feat: exchange bias matrix carries the Pep-GaMD boost under each state's λ"
```

---

### Task 5: Sample columns `v_pep_kj_mol`, `v_dih_kj_mol`, `gamd_lambda`

**Files:**
- Modify: `gareus/store.py:33-75` (`ParquetSampleWriter.write_sample`, `flush`)
- Modify: `gareus/production.py` — the `row = {...}` dict (~line 6165) and the `parquet_sample_writer.write_sample(...)` call (~line 6235); the `RowBuffer`-style collector at ~3493-3581 that exposes `potential_kj_mol`/`gamd_boost_total_kj_mol` arrays
- Test: `tests/test_lambda_ladder_states.py` (append)

- [ ] **Step 1: Write the failing test**

```python
def test_parquet_sample_writer_stores_raw_channel_energies_and_lambda():
    import pyarrow.parquet as pq
    from gareus.store import ParquetSampleWriter
    d = pathlib.Path(tempfile.mkdtemp())
    w = ParquetSampleWriter(d, flush_rows=10)
    w.write_sample(step=1, replica=0, window_id=0, cv1=0.1, cv2=None, potential=-5.0,
                   boost_total=1.0, boost_dihedral=0.5, boost_nonbonded=0.0,
                   v_pep=12.5, v_dih=3.25, gamd_lambda=0.5)
    w.write_sample(step=2, replica=0, window_id=0, cv1=0.1, cv2=None, potential=-5.0,
                   boost_total=0.0, boost_dihedral=0.0, boost_nonbonded=0.0)   # defaults: NaN, NaN, 0.0
    w.flush()
    t = pq.read_table(sorted(d.glob("chunk_*.parquet"))[0]).to_pydict()
    assert t["v_pep_kj_mol"][0] == 12.5 and t["v_dih_kj_mol"][0] == 3.25 and t["gamd_lambda"][0] == 0.5
    assert np.isnan(t["v_pep_kj_mol"][1]) and t["gamd_lambda"][1] == 0.0
```
(Check `ParquetSampleWriter.__init__`'s parameter names at `store.py:19-31` and adapt the constructor call.)

- [ ] **Step 2: RED** — `TypeError: unexpected keyword 'v_pep'`.

- [ ] **Step 3: Implement** — in `write_sample` add parameters `v_pep: float = float("nan"), v_dih: float = float("nan"), gamd_lambda: float = 0.0`, append to `b["v_pep_kj_mol"]`, `b["v_dih_kj_mol"]`, `b["gamd_lambda"]`; in `flush` add the three columns (`pa.float32()` for the energies, `pa.float32()` for λ). In production's `write_sample(...)` call pass `v_pep=float(observable_cache["v_pep_kj"][r])` etc. when `pep_env is not None`, and `gamd_lambda=float(state_lambdas[w])`; add the same three keys to the CSV `row` dict; add them to the array collector so `analysis_arrays.npz` carries them.

- [ ] **Step 4: GREEN. Step 5: Commit** — `feat: store raw v_pep/v_dih and gamd_lambda per sample`.

---

### Task 6: MBAR reduced potentials with the ladder term (union path + legacy loader)

**Files:**
- Modify: `gareus/adaptive_production.py` `build_union_state_mbar_inputs` (~lines 2986-3134): `sample_rows` gain `v_pep_kj_mol`, `v_dih_kj_mol`; states expose `gamd_lambda`; after `umbrella_bias_kj = 4.184 * umbrella_bias_kcal` add the boost term
- Modify: `gareus/query.py:253` `reconstruct_bias_matrix(cv_A, cv2, windows, beta, *, v_pep=None, v_dih=None, envelope=None)`
- Modify: `gareus/mbar_analysis/loaders.py` (Parquet loader ~lines 439-471 and CSV loader ~320-385): read the new columns into `Data`; set `meta["gamd_ladder"] = bool(any λ > 0)`
- Test: `tests/test_lambda_ladder_mbar.py`

**Interfaces:**
- Produces: `u_nk[n, k] = β·(umbrella_kj[n,k] + pep_gamd_boost_kj(v_pep_n, v_dih_n, λ_k, env))`; the npz/inputs additionally carry `gamd_boost_kj_nk` for audit; `Data.v_pep_kj`, `Data.v_dih_kj`, `Data.state_lambdas`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_lambda_ladder_mbar.py
import numpy as np


def test_reconstruct_bias_matrix_adds_boost_term_per_state_lambda():
    from gareus.query import reconstruct_bias_matrix
    from gareus.pep_gamd import PepGamdEnvelope, pep_gamd_boost_kj
    env = PepGamdEnvelope(50.0, -50.0, 50.0, 0.8, 50.0, -50.0, 50.0, 0.6)
    windows = [{"center": 0.1, "k": 100.0, "gamd_lambda": 0.0}, {"center": 0.1, "k": 100.0, "gamd_lambda": 1.0}]
    cv = np.array([0.1, 0.2]); v_pep = np.array([10.0, 30.0]); v_dih = np.array([5.0, 7.0]); beta = 1.0 / 2.494
    u = reconstruct_bias_matrix(cv, None, windows, beta, v_pep=v_pep, v_dih=v_dih, envelope=env)
    u0 = reconstruct_bias_matrix(cv, None, windows, beta)
    assert np.allclose(u[:, 0], u0[:, 0])                                     # λ = 0 state unchanged
    assert np.allclose(u[:, 1] - u0[:, 1], beta * np.array([pep_gamd_boost_kj(10.0, 5.0, 1.0, env), pep_gamd_boost_kj(30.0, 7.0, 1.0, env)]))


def test_reconstruct_bias_matrix_refuses_lambda_states_without_energies():
    from gareus.query import reconstruct_bias_matrix
    windows = [{"center": 0.1, "k": 100.0, "gamd_lambda": 0.5}]
    try:
        reconstruct_bias_matrix(np.array([0.1]), None, windows, 0.4)
    except ValueError as exc:
        assert "v_pep" in str(exc)
    else:
        raise AssertionError("a λ>0 state without raw energies cannot be reweighted and must fail loudly")
```
(Read `reconstruct_bias_matrix`'s docstring for the exact window-dict keys it uses for centre/k — `center`/`k` vs `center1`/`k1` — and use those.)

- [ ] **Step 2: RED** — `TypeError: unexpected keyword 'v_pep'`.

- [ ] **Step 3: Implement** in `gareus/query.py` — add the keyword-only parameters; after the umbrella matrix is assembled:
```python
    lambdas = np.asarray([float(w.get("gamd_lambda", 0.0) or 0.0) for w in windows], dtype=float)
    if np.any(lambdas > 0.0):
        if v_pep is None or v_dih is None or envelope is None:
            raise ValueError("windows carry gamd_lambda > 0 but v_pep/v_dih/envelope were not supplied; "
                             "the ladder cannot be reweighted without the raw channel energies")
        from .pep_gamd import pep_gamd_boost_matrix_kj
        u = u + float(beta) * pep_gamd_boost_matrix_kj(v_pep, v_dih, lambdas, envelope).T
    return u
```
In `build_union_state_mbar_inputs`: add `"v_pep_kj_mol"`/`"v_dih_kj_mol"` to each `sample_rows` entry (read from the sample row like `gamd_boost_total_kj_mol`), build `state_lambdas = np.asarray([float(getattr(s, "gamd_lambda", 0.0)) for s in states])`, load the envelope from `adaptive_dir.parent / "global_shared_gamd_setup" / "shared_gamd_setup_globals.json"` via `PepGamdEnvelope.from_json` when any λ > 0, and change
```python
    umbrella_bias_kj = 4.184 * umbrella_bias_kcal
```
to
```python
    umbrella_bias_kj = 4.184 * umbrella_bias_kcal
    gamd_boost_kj_nk = np.zeros_like(umbrella_bias_kj)
    if np.any(state_lambdas > 0.0):
        gamd_boost_kj_nk = pep_gamd_boost_matrix_kj(v_pep_values, v_dih_values, state_lambdas, envelope).T
    total_bias_kj = umbrella_bias_kj + gamd_boost_kj_nk
```
and use `total_bias_kj` where `umbrella_bias_kj` fed `umbrella_reduced_bias_nk`; store `gamd_boost_kj_nk` in the returned inputs so `adaptive_union_mbar.npz` carries it. In `loaders.py`, read the new Parquet columns with the same `_fill_masked_nan` treatment as `gamd_boost_dihedral`, attach `v_pep_kj`, `v_dih_kj`, `state_lambdas` to `Data`, and set `meta["gamd_ladder"]`.

- [ ] **Step 4: Estimator selection.** In `gareus/mbar_analysis/pmf.py` `select_unbiased_method(...)`: when the caller passes `gamd_ladder=True`, return `('umbrella_only', 'λ ladder: boost is inside u_nk, MBAR is exact; no cumulant')`. Thread `gamd_ladder=d.meta.get("gamd_ladder", False)` from the analyzer's call site (grep `select_unbiased_method(` in `analyze_gareus_mbar.py`). Add a test:
```python
def test_ladder_selects_the_exact_umbrella_only_path():
    from gareus.mbar_analysis.pmf import select_unbiased_method
    method, reason = select_unbiased_method(exp_ess=1.0, n_samples=10, gamd_ladder=True)
    assert method == 'umbrella_only' and 'ladder' in reason
```
(match the existing positional/keyword parameters of `select_unbiased_method`; the default `gamd_ladder=False` must leave every existing test unchanged).

- [ ] **Step 5: GREEN**, then run `tests/test_unbiased_method_selection.py` and any test touching `reconstruct_bias_matrix` / `build_union_state_mbar_inputs` (grep `tests/`).

- [ ] **Step 6: Commit** — `feat: MBAR reduced potentials include the ladder boost from stored raw energies`.

---

### Task 7: λ=0-only vs full-ladder PMF cross-check

**Files:**
- Create: `gareus/mbar_analysis/crosscheck.py`
- Modify: `analyze_gareus_mbar.py` — call the cross-check when `d.meta["gamd_ladder"]`; write `pmf_analysis/pmf_ladder_crosscheck.csv`, `.png`, and a `ladder_crosscheck` block in `pmf_summary.json`
- Test: `tests/test_lambda_ladder_mbar.py` (append)

**Interfaces:**
- Consumes: `gareus.mbar_analysis.solvers._subset_logw_from_global_fk(d_subset, f_k_global)`, `gareus.mbar_analysis.pmf.pmf_from_weights(cv, w, bins, kbt_kcal)`.
- Produces:
  ```python
  def ladder_crosscheck(d, f_k_global, bins, kbt_kcal) -> dict
  # {"pmf_full": ..., "pmf_lambda0": ..., "max_abs_diff_kcal": float, "n_lambda0_samples": int, "status": "pass"|"fail"|"skipped"}
  ```

- [ ] **Step 1: Write the failing test**

```python
def test_ladder_crosscheck_agrees_when_lambda_zero_samples_are_representative():
    """Synthetic Data: all samples effectively unbiased, half tagged λ=0, half λ=1 with zero boost
    (v above threshold). The two PMFs must coincide."""
    from gareus.mbar_analysis.crosscheck import ladder_crosscheck
    from gareus.mbar_analysis.loaders import Data
    rng = np.random.default_rng(0); n = 4000
    cv = rng.normal(0.3, 0.05, n); win = np.repeat([0, 1], n // 2)
    u = np.zeros((n, 2)); beta = 1.0 / 2.494
    d = Data.__new__(Data)                              # construct minimally; fill the fields the cross-check reads
    d.cv, d.win, d.u, d.beta = cv, win, u, beta
    d.state_lambdas = np.array([0.0, 1.0]); d.meta = {"gamd_ladder": True}
    f_k = np.zeros(2)
    out = ladder_crosscheck(d, f_k, bins=20, kbt_kcal=0.596)
    assert out["status"] == "pass" and out["max_abs_diff_kcal"] < 0.15 and out["n_lambda0_samples"] == n // 2


def test_ladder_crosscheck_skips_without_lambda_zero_states():
    from gareus.mbar_analysis.crosscheck import ladder_crosscheck
    from gareus.mbar_analysis.loaders import Data
    d = Data.__new__(Data); d.cv = np.zeros(10); d.win = np.zeros(10, int); d.u = np.zeros((10, 1)); d.beta = 0.4
    d.state_lambdas = np.array([1.0]); d.meta = {"gamd_ladder": True}
    assert ladder_crosscheck(d, np.zeros(1), bins=5, kbt_kcal=0.596)["status"] == "skipped"
```
(Inspect `Data`'s constructor in `loaders.py` and either construct it properly or keep the `__new__` shortcut; the cross-check must only touch `cv`, `win`, `u`, `beta`, `state_lambdas`, `meta`.)

- [ ] **Step 2: RED** — `ModuleNotFoundError: gareus.mbar_analysis.crosscheck`.

- [ ] **Step 3: Implement**

```python
# gareus/mbar_analysis/crosscheck.py
"""λ=0-only vs full-ladder PMF: the built-in check that the boost reweighting is right.

The λ=0 rungs are plain umbrella sampling. Their PMF, computed with the global f_k but
only their own samples (subset N_k, never renormalised as if self-consistent — see
_subset_logw_from_global_fk), must agree with the full-ladder PMF within error."""
import numpy as np
from .pmf import pmf_from_weights
from .solvers import _subset_logw_from_global_fk


def _subset(d, mask):
    s = d.__class__.__new__(d.__class__)
    s.cv = d.cv[mask]; s.win = d.win[mask]; s.u = d.u[mask]; s.beta = d.beta
    s.state_lambdas = d.state_lambdas; s.meta = dict(d.meta)
    return s


def ladder_crosscheck(d, f_k_global, bins, kbt_kcal, tol_kcal: float = 0.5) -> dict:
    lambdas = np.asarray(getattr(d, "state_lambdas", np.zeros(d.u.shape[1])), dtype=float)
    lam0_states = np.where(lambdas == 0.0)[0]
    if lam0_states.size == 0:
        return {"status": "skipped", "reason": "no λ=0 states", "n_lambda0_samples": 0}
    full_logw = _subset_logw_from_global_fk(d, np.asarray(f_k_global, dtype=float))
    pmf_full = pmf_from_weights(d.cv, np.exp(full_logw - full_logw.max()), bins, kbt_kcal)
    mask = np.isin(d.win, lam0_states)
    sub = _subset(d, mask)
    sub_logw = _subset_logw_from_global_fk(sub, np.asarray(f_k_global, dtype=float))
    pmf_lam0 = pmf_from_weights(sub.cv, np.exp(sub_logw - sub_logw.max()), bins, kbt_kcal)
    both = np.isfinite(pmf_full) & np.isfinite(pmf_lam0)
    diff = (pmf_full - pmf_full[both].min()) - (pmf_lam0 - pmf_lam0[both].min())
    max_abs = float(np.nanmax(np.abs(diff[both]))) if both.any() else float("nan")
    return {"status": "pass" if max_abs <= tol_kcal else "fail", "max_abs_diff_kcal": max_abs,
            "n_lambda0_samples": int(mask.sum()), "pmf_full": pmf_full, "pmf_lambda0": pmf_lam0,
            "tolerance_kcal": tol_kcal}
```
(Check `pmf_from_weights`'s return shape and whether it wants raw or normalised weights — read `pmf.py:99-118` — and adapt; the bins argument must match what the analyzer already passes.) In `analyze_gareus_mbar.py`, after `f_k` is solved and when `d.meta.get("gamd_ladder")`, call it, write the two PMFs to `pmf_ladder_crosscheck.csv`, plot both curves to `pmf_ladder_crosscheck.png`, and put `{status, max_abs_diff_kcal, n_lambda0_samples, tolerance_kcal}` under `pmf_summary["ladder_crosscheck"]`. A `"fail"` must appear in `gareus_report.py`'s warnings (grep how other `pmf_summary` warnings are surfaced and add one line).

- [ ] **Step 4: GREEN. Step 5: Commit** — `feat: λ=0-only vs full-ladder PMF cross-check`.

---

### Task 8: Replace the tautological PBC guard

**Files:**
- Modify: `gareus/system_setup.py:455-475` (`_write_box_audit`)
- Test: `tests/test_box_audit_guard.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_box_audit_guard.py
import json, pathlib, tempfile, types
import numpy as np


def _audit(box_nm, contour_nm, cutoff_nm):
    from gareus.system_setup import _write_box_audit
    d = pathlib.Path(tempfile.mkdtemp())
    pos = np.zeros((3, 3)); ca = np.zeros((2, 3))
    args = types.SimpleNamespace(padding_nm=1.0, nonbonded_cutoff_nm=cutoff_nm, seq="GYDPETGTWG")
    _write_box_audit(d, pos, ca, contour_nm, box_nm, args)
    return json.loads((d / "box_audit.json").read_text())


def test_guard_is_silent_for_the_chignolin_box():
    a = _audit(box_nm=5.82, contour_nm=3.82, cutoff_nm=0.9)
    assert a["pbc_self_contact_warning"] is False
    assert abs(a["min_image_gap_nm"] - 2.0) < 1e-9


def test_guard_fires_when_the_gap_is_below_the_cutoff():
    a = _audit(box_nm=4.5, contour_nm=3.82, cutoff_nm=0.9)     # gap 0.68 < 0.9
    assert a["pbc_self_contact_warning"] is True
```

- [ ] **Step 2: RED** — `KeyError: 'min_image_gap_nm'` / warning `True` for the adequate box.

- [ ] **Step 3: Implement** — in `_write_box_audit` replace
```python
    pbc_self_contact_warning = bool(sequence_contour_estimate_nm > box_nm / 2.0 - padding_nm)
```
with
```python
    cutoff_nm = float(getattr(args, "nonbonded_cutoff_nm", 1.0) or 1.0)
    min_image_gap_nm = float(box_nm - sequence_contour_estimate_nm)
    pbc_self_contact_warning = bool(min_image_gap_nm < cutoff_nm)
```
add `"min_image_gap_nm": min_image_gap_nm, "nonbonded_cutoff_nm": cutoff_nm` to the JSON, and reword the warning message to quote the gap and the cutoff. Keep `minimum_margin_nm` as is.

- [ ] **Step 4: GREEN. Step 5: Commit** — `fix: box_audit PBC guard compares the min-image gap to the cutoff instead of to itself`.

---

### Task 9: Provenance, helptext, handoff

**Files:**
- Modify: `gareus/provenance.py` (the manifest field list ~lines 274-300) — add `state_gamd_lambdas`, `pep_gamd_envelope_path`
- Modify: `gareus/helptext.py` — new section "λ ladder over Pep-GaMD boost strength" after the Pep-GaMD section
- Modify: `CLAUDE.md` — a "λ ladder" entry under the Pep-GaMD one
- Test: `tests/test_lambda_ladder_states.py` (append)

- [ ] **Step 1: Failing test**
```python
def test_manifest_records_state_lambdas():
    from gareus.provenance import manifest_method_settings   # grep the function that builds method_settings; use its real name
    args = types.SimpleNamespace(state_gamd_lambdas=[0.0, 0.5, 1.0], gamd_boost_type="pep-gamd-lower-dual")
    ms = manifest_method_settings(args)
    assert ms["state_gamd_lambdas"] == [0.0, 0.5, 1.0]
```
- [ ] **Step 2: RED. Step 3: Implement** — add the two keys wherever `gamd_boost_type` is recorded (`provenance.py:278`). Helptext: one paragraph with the `u_ik` formula, the frozen-envelope rule, the cross-check, and the stored-column requirement (copy §3.1–3.3 of the spec, condensed). `CLAUDE.md`: state grid = (window, λ); k0 override points; where the envelope lives; the cross-check file; "never recalibrate mid-campaign".
- [ ] **Step 4: GREEN. Step 5: Commit** — `docs: λ-ladder provenance, helptext and handoff`.

---

### Task 10: Pilot configuration and a tiny end-to-end run

**Files:**
- Create: `examples/chignolin_lambda_ladder_pilot.yaml` and `examples/chignolin_lambda_ladder_pilot_windows.csv`
- Test: extend the existing "Tiny real workflow test" (see `gareus -hh` section of that name for the command it runs; the test file is found with `grep -l "tiny" tests/*.py`) with a ladder variant, marked slow

- [ ] **Step 1: Write the windows CSV** — 1 CV1 centre × 5 rungs:
```
window,primary_cv_mode,primary_cv_center,primary_cv_k_kcal,gamd_lambda
0,contacts,0.25,800,0.0
1,contacts,0.25,800,0.1
2,contacts,0.25,800,0.25
3,contacts,0.25,800,0.5
4,contacts,0.25,800,1.0
```
- [ ] **Step 2: Write the YAML** — copy `examples/chignolin_genpept_contact_bias_sigma.yaml`'s structure; set `gamd_boost_type: pep-gamd-lower-dual`, `contact_atom_selection: heavy`, `contact_adaptive_max_k_kcal: 1200`, `exchange_mode: gibbs-walk`, `exchange_interval: 400`, `sample_potential_energy: true`, `windows_2d_csv: examples/chignolin_lambda_ladder_pilot_windows.csv`, `cv2: none`, `production_steps: 500000` (2 ns at 4 fs), `md_budget_ns` sized for 5 replicas.
- [ ] **Step 3: Tiny run test** — with the tiny fixture peptide (`GA`), 2 states λ ∈ {0, 1}, `production_steps` ~2000, `exchange_interval` 200; assert the run completes, the sample Parquet has `v_pep_kj_mol` finite for every row and `gamd_lambda ∈ {0,1}`, `exchange` records contain at least one attempted swap, and `pmf_summary.json["ladder_crosscheck"]["status"] != "fail"` (it may be `"pass"` or, with tiny data, `"skipped"` is not allowed — both states must exist). Mark slow.
- [ ] **Step 4: Run via opencode; fix what breaks; commit** — `feat: λ-ladder pilot config and end-to-end smoke test`.

---

### Task 11: Merge order

- [ ] Open a PR for `feat/pep-gamd-boost` → `main` (title: "Pep-GaMD: boost only the peptide essential potential"). Body: the four invariants, the two quirks, test list, and that the sanctioned runner has not executed the three `capsys` tests.
- [ ] Open a PR for `feat/lambda-ladder` → `main` **based on** `feat/pep-gamd-boost`; merge after it.
- [ ] After merge: run the S3 pilot (`examples/chignolin_lambda_ladder_pilot.yaml`) and record in `CLAUDE.md`: per-edge acceptance, ΔV per rung, σ_V(pep) vs σ_V(total), λ=0 CV1 autocorrelation vs a no-exchange control, and the per-step cost of the second PME. These numbers replace every "estimate" in the spec.

---

## Self-review

**Spec coverage.** §3.1–3.3 → Tasks 1, 5, 6. §3.4 exchange → Tasks 3, 4. §3.5 spacing → Task 10 pilot (the swarm-derived estimate is the swarm plan). §3.6 envelope frozen → Global Constraints + Task 3 (no recalibration path added). §4 → base branch, Task 11. §6 cross-check → Task 7; 2-D post-hoc landscape needs no code (existing reprojection tools work on stored trajectories); thermodynamics `⟨V⟩(CV1)` uses the existing `potential_kj_mol` column and MBAR weights — no new task, noted for the analysis plan. §7 gates → Task 7 (quoting gate) + Task 10 (pilot gates measured, thresholds recorded). §8 items 2–6, 9, 10 → Tasks 2–9. §5 swarm and §8 items 7–8 → **separate plan**, deliberately.

**Placeholders.** None: every step has code or an exact edit. Two places ask the implementer to read a signature before adapting a call (`apply_window_swap`, `Data.__init__`, `select_unbiased_method`, `pmf_from_weights`) — those name the file and line to read.

**Type consistency.** `PepGamdEnvelope` field order `(vmax_total, vmin_total, threshold_total, k0max_total, vmax_dih, vmin_dih, threshold_dih, k0max_dih)` is used positionally in Tasks 4, 6, 7 — matches Task 1. `pep_gamd_boost_matrix_kj` returns `(n_states, n_samples)`; Task 6 transposes it (`.T`) to `(n_samples, n_states)` for `u_nk` — consistent. `set_replica_lambda(integrator, lam, k0max)` and `k0max_from_globals` are used identically in Task 3's wiring. `state_lambdas` is a numpy array of length `nrep` everywhere.
