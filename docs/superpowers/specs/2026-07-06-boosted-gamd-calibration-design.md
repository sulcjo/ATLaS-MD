# Self-consistent boosted GaMD calibration

Date: 2026-07-06
Status: approved

## Problem

The joint-envelope GaMD calibration (`gareus/gamd_calibration.py`,
`gareus/production.py:run_multiwindow_gamd_recon` /
`apply_joint_envelope_gamd_calibration`) measures the dihedral-energy spread
`sigmaV` under **conventional MD** (`make_cmd_integrator`, boost OFF). Production
then runs with the GaMD boost ON, which flattens the landscape and broadens the
sampled energy distribution.

Evidence from `RUNS/chignolin/chignolin/chignolin_2d_run3` (epoch_000):

- Calibration (unboosted recon, 1200 pooled samples): `sigmaV_Dihedral = 18.98 kJ`.
- Production (3.12M frames, reconstructed): `sigmaV ~ 31.4 kJ` (+65%).
- The boost is `DV = 1/2 k (E-V)^2` (lower-bound, `E=Vmax`). A squared transform
  is near-Gaussian only when `a = (E - <V>)/sigmaV` is large. Setup `a ~= 4.9`
  (predicts skew ~0.6); production `a ~= 2.3` (predicts skew ~1.20, observed 1.25).
- Result: boost anharmonicity 1.7-2.1 (BAD), cumulant2 reweighting FAIL,
  exp-reweight ESS = 4 samples.

Re-calibrating each adaptive round does not help: every round re-measures on the
same too-narrow unboosted ensemble (round_00 sigmaV 18.98 -> round_03 20.79).

## Goal

Measure `sigmaV` under the boost the run will actually use, so the frozen boost
parameters (Vmax/Vmin/Vavg/sigmaV/k0/k/threshold) match the boosted production
ensemble. Approach chosen by the user:

- **Re-measure under fixed boost** (reuse existing verified helpers).
- **Iterate to convergence** on `sigmaV`.
- **On by default.**
- **Rebalance the step budget**: short cMD seed, longer boosted passes.

## Design

### Flow

1. **cMD seed pass** (short). Run the existing per-window cMD recon to get an
   initial `PooledEnvelope` per boost group (initial Vmax/Vmin/Vavg/sigmaV).
   This seeds the boost; the GaMD integrator needs starting extrema (chicken-and-egg).
2. **Boosted convergence loop.** Repeat up to `gamd_recon_boosted_iters`:
   - Compute calibration from the current envelope
     (`compute_group_calibration`) and overwrite the physics globals
     (`overwrite_physics_globals`) to produce a full seed-globals dict with
     `stage=5.0` (fixed-boost production stage).
   - Run a per-window **boosted** recon: build `make_gamd_integrator`, seed it
     via `set_integrator_globals_from_dict` (same pattern as the existing
     50-step verify block, `production.py:2890-2909`), step
     `gamd_multiwindow_recon_steps`, Welford-accumulate boost-group PE, pool.
   - Compare new pooled `sigmaV` to previous (per group). Stop when the max
     relative change across boosted groups `< gamd_recon_boosted_tol`, or at
     `max_iters`.
3. **Finalize.** Overwrite globals from the converged calibration, run the
   existing 50-step finite-energy verification, write the report.

The loop is a negative feedback: larger `sigmaV` -> smaller `k0'` -> weaker
boost -> smaller `sigmaV`. It converges to a fixed point. `max_iters` bounds cost.

### Units and boundaries

- **`gamd_calibration.py` (pure, no OpenMM)** — new:
  - `envelope_converged(prev: PooledEnvelope, curr: PooledEnvelope, tol) -> bool`
    (relative `sigmaV` change).
  - `run_calibration_convergence(measure_fn, boost_type, sigma0, seed_envelope,
    max_iters, tol) -> ConvergenceResult` where `measure_fn(calibration) ->
    PooledEnvelope` is injected (the boosted MD). Returns final
    `GroupCalibration`, per-iteration `sigmaV` trace, `converged` bool, `iters`.
    Unit-testable with a synthetic `measure_fn`.
- **`production.py`** —
  - Generalize `run_multiwindow_gamd_recon` to accept an integrator factory +
    step count, so the same per-window loop serves both cMD seed and boosted
    passes (or a sibling `run_multiwindow_gamd_recon_boosted`).
  - Rework `apply_joint_envelope_gamd_calibration` to: short cMD seed ->
    per-group `run_calibration_convergence` (measure_fn runs the boosted recon
    with the current seed globals) -> overwrite -> verify -> report.
  - Add a warning when a converged boosted group has `k0 >= 0.999` (the
    `sigmaV <= sigma0` guardrail is inert; `sigma_DV` uncontrolled; consider a
    smaller `sigma0`).
- **`cli.py` / `helptext.py`** — new flags below.
- **Report** — `shared_gamd_setup_globals.json` gains a `boosted_calibration`
  block: per-group per-iteration `sigmaV` trace, `converged`, `iters`,
  `cmd_seed_sigmaV`. Fix the stale "exactly as before this change" description.

### Flags

| Flag | Default | Meaning |
|---|---|---|
| `--gamd-multiwindow-recon-cmd-steps` | 20000 | short cMD seed pass per window |
| `--gamd-multiwindow-recon-steps` | 20000 | **boosted** steps per window per iteration (repurposed) |
| `--gamd-recon-boosted-iters` | 4 | max boosted iterations (0 = legacy cMD-only) |
| `--gamd-recon-boosted-tol` | 0.05 | relative `sigmaV` convergence threshold |

`--gamd-multiwindow-recon-prep-steps` (existing) still applies as pre-recon
relaxation for both cMD and boosted passes. The chignolin config already sets
`gamd_multiwindow_recon_steps: 300000`, which now becomes the boosted per-iter
budget ("longer gamd"); the cMD seed drops to the new short default
("shorten cMD") automatically.

### Cost

~nwin x (cmd_steps + up to iters x recon_steps). For chignolin
(6 windows, 20k cMD + up to 4x300k boosted) ~ 7.3M steps vs current 1.8M.
Bounded by `max_iters` and early convergence exit.

### Out of scope (flagged, not fixed)

`k0` will likely still clamp at 1.0 for chignolin because `sigma0 = 5 kcal/mol`
is too large to bind (`k0' = 1` at `sigma0 ~= 1.15 kcal/mol`). This is a config
tuning choice, not a calibration bug; addressed only by the new k0-saturation
warning. Correct-ensemble calibration is the prerequisite; `sigma0` tuning is a
separate follow-up.

## Testing (TDD)

Pure unit tests in `tests/` (no OpenMM):

- `envelope_converged`: below/above tol; zero-guard on prev sigmaV.
- `run_calibration_convergence` with synthetic `measure_fn`:
  - monotone approach -> converges, correct final calibration + trace.
  - oscillation -> stops at max_iters, `converged=False`.
  - already-converged (measure_fn returns seed) -> 1 iteration.
  - `max_iters=0` -> returns cMD-seed calibration unchanged (legacy path).

Existing `gamd_calibration` formula/pooling tests stay green.
`py_compile` touched modules; parser + help smoke for new flags.
```
pytest -q tests/test_gamd_calibration*.py tests/test_bootstrap_torsion_cv.py
python -m py_compile gareus/gamd_calibration.py gareus/production.py gareus/cli.py gareus/helptext.py
python -m gareus -h ; python -m gareus -hh
```
