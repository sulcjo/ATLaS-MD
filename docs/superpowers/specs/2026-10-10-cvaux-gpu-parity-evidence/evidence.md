# CVaux GPU parity evidence (runbook 2026-10-07, run 2026-10-10)

Code: `feat/cvaux-adaptive` @ `f6f44b03e9f0720d6cee92643cc91e67c68d0c7f` (= PR #141 head `fbe9500` minus one docs-only commit), rsynced to the aurum2 scratch CODE_DIR
`/home/sulcjo/2026_peptide_sampler_cvaux_parity` (DEPLOYED_COMMIT written). Outputs: `/home/sulcjo/cvaux_parity_runs`.
c10's trees (`~/2026_peptide_sampler`, `~/gareus`) were not touched. Raw reports, timers, model and job scripts are in `raw/`.

Jobs (one L40S each, non-exclusive, driver 595.71.05): prep 2898973, cuda_mixed 2898974 (d090), opencl 2898975 (d075),
cuda_double 2898976 (d090), roundtrip 2898977 (d075), noaux 2898978 (d075). All COMPLETED, exit 0.
Model: topology sha `645bffed...` = the Task 14 CPU control's model; torsions psi-GLY1, phi-ALA2.

## Deviation from the runbook (user-approved)

All legs ran with `--us-allow-bad-windows`. For GA, `--cv1 distance` is the terminal CA-CA distance of adjacent
residues, which stays rigid at about 3.9 A. W24 centres from about 5.5 A up miss by 1.7-4.3 A, and the US
start-quality gate refused 14 of 24 windows (prep 2898887). The CSVs, model and slot table are unchanged.
Earlier failed attempts: 2898745, a launcher bug (conda `activate` read `$1`); 2898887, the quality gate.
Follow-up: re-centre W24 onto reachable CV1 values in the runbook.

## Parity (`python -m gareus.auxiliary_cv.parity_report`)

| run | segment | platform / precision | rows | max abs dz | max reduced | k_max | tol | bound_vacuous | ok |
|---|---|---|---|---|---|---|---|---|---|
| CUDA mixed | seg_001 | CUDA mixed | 576 | 0.0 | 0.0 | 1.2 | 1e-4 | false | true |
| OpenCL | seg_001 | OpenCL mixed | 576 | 0.0 | 0.0 | 1.2 | 1e-4 | false | true |
| CUDA double | seg_001 | CUDA double | 576 | 0.0 | 0.0 | 1.2 | 1e-6 | false | true |
| round trip | seg_001 | CUDA mixed | 336 | 0.0 | 0.0 | 1.2 | 1e-4 | false | true |
| round trip | seg_002 | CUDA mixed | 384 | 0.0 | 0.0 | 1.2 | 1e-4 | false | true |

`bound_vacuous` is written only when true, so it is absent from these reports.

Round trip: leg 1 died with `RuntimeError: GAREUS_TEST_FAIL_AT_PROD_STEP=3500: injected test failure` (exit 1,
string-checked before resuming). The `--resume` leg restored the step-2000 checkpoint and ran seg_002 from 2000 to 6000.
`verify_aux_resume` (`production.py:9120`) passed, meaning it did not refuse at `_REL` 1e-12. It records no
measured deviation, only pass or refuse.

## LIMITATION: this configuration cannot detect a GPU aux-force z error

`max_abs_dz` is exactly 0.0 on all platforms by construction, not because the GPU matches:

- Distance CV1 is a plain `CustomBondForce`. `fast_cv_force_indices` therefore finds no primary CustomCVForce, and
  `_use_fast_cv_path` is False.
- On the slow path the stored `aux_z_00` (`z_runtime`) is computed in NumPy from the positions read
  (`runtime.observe_aux_z(positions_nm=)`). The stored torsions come from the same positions, and
  `parity_report` recomputes z from those torsions. Both sides are the same double-precision NumPy computation.
- In-run `check_runtime_parity` compares the same two quantities.

What this run does show on CUDA mixed, OpenCL and CUDA double: aux runs complete; the stored torsion and z
columns are finite and consistent; the active aux state (k_max 1.2) is present; the injected crash plus resume
works on GPU with `verify_aux_resume` passing.

What it does not show is whether the GPU `CustomCVForce` value (`getCollectiveVariableValues`, fast path) agrees
with offline z within 1e-4 / 1e-6. c11 (`RUNS/chignolin_11.yaml`: `cv1: contact-map`, `cv2: auto`) will run the
fast path. c10 has the same CV kinds, and its latest production log on aurum2 (`chignolin_10_2884283.log`) has no
`has no verified fast-path scalar` line, so it observes on the fast path. On the fast path
the exchange-matrix z is the GPU value. On c11 that path is guarded only in-run by `check_runtime_parity`, which
fails closed: it raises and stops the segment. It fires only once a state with k > 0 exists. In c11 that is the
first sample after the epoch-1 admission, so a mismatch would show up as a crash hours into the run, not at launch.
A GPU parity check of the fast path needs a configuration whose CV1 has a CustomCVForce. The cheapest option:
`tests/test_aux_cv_observation.py::test_fast_and_slow_paths_agree_for_an_ordinary_state` exists but runs on
CPU/Reference only. A GPU-parametrized copy, run on aurum2 on OpenCL and CUDA, would measure exactly this quantity.

## Cost (s per 2000 steps from `production_phase_timers.json`; 24 replicas, 6000 steps, 2 fs)

| run | node | md | sample | sample.fetch | exchange | wall s | ns/day (node, 24 rep) |
|---|---|---|---|---|---|---|---|
| CUDA mixed, aux | d090 | 9.135 | 0.221 | 0.213 | 0.076 | 29.0 | 858 |
| no-aux baseline | d075 | 8.871 | 0.071 | 0.066 | 0.065 | 27.6 | 901 |
| OpenCL, aux | d075 | 10.001 | 0.243 | 0.236 | 0.073 | 31.7 | 784 |
| CUDA double, aux | d090 | 46.126 | 0.308 | 0.300 | 0.077 | 140.2 | 178 |

Aux vs baseline: md +3.0 %, sample.fetch 3.2x (+0.15 s/2000, slow-path positions read plus torsions),
ns/day -4.7 %. Indicative only: same GPU model but different nodes, non-exclusive, 30 s runs.

## Gate

The runbook's pass criterion is met: every segment is ok at its precision's tolerance, bound_vacuous is false,
and the round trip resumes with verify_aux_resume passing. No tolerance was changed. Given the limitation above,
this does not validate the fast-path GPU aux z that c11 will use.

## Fast-path GPU check (job 2900637, d088 L40S, 2026-10-10)

`raw/fastpath_gpu.py` runs from the CODE_DIR root with `PYTHONPATH=.:tests`; the job script is `raw/fastpath_job.sh`.
- Fixture and model: the solvated GA dipeptide and the model of `tests/test_aux_cv_observation.py` (random
  coefficients, offset -0.3), in three variants: plain, mixed sign conventions, scale 3.181.
- An ACTIVE restraint (k 1.2 kcal/mol, centre = initial z) acts during Langevin MD at 300 K, 2 fs.
- 200 samples every 50 steps. At each sample both paths are read from one Context state: the fast path
  (`CustomCVForce.getCollectiveVariableValues`) and the slow path (NumPy from the same positions).
- The aux force-group energy is checked against 0.5 k (z_slow - c)^2.

Worst over the three model variants (z sd 0.22-0.63 across samples):

| platform | precision | max abs dz | max reduced | tol | max abs dE (kJ/mol) | ok |
|---|---|---|---|---|---|---|
| Reference | double | 9.7e-16 | 3.3e-15 | 1e-6 | 1.1e-14 | 3/3 |
| CUDA | mixed | 3.1e-6 | 6.5e-6 | 1e-4 | 1.6e-5 | 3/3 |
| CUDA | single | 4.8e-7 | 1.6e-6 | 1e-4 | 3.9e-6 | 3/3 |
| CUDA | double | 6.0e-8 | 1.7e-7 | 1e-6 | 4.2e-7 | 3/3 |
| OpenCL | mixed | 4.5e-6 | 6.3e-6 | 1e-4 | 1.6e-5 | 3/3 |
| OpenCL | single | 6.6e-7 | 1.8e-6 | 1e-4 | 4.5e-6 | 3/3 |
| OpenCL | double | 6.2e-8 | 1.9e-7 | 1e-6 | 4.8e-7 | 3/3 |

Margins:
- Mixed: about 15x under 1e-4.
- Double: about 5x under 1e-6.
- The reduced bound scales as k x |z - c| x |dz|. A campaign with a larger aux k, or wider z excursions, shrinks
  these margins proportionally. Example: k = 10 kcal/mol at the same excursions leaves mixed at about 2x.
- Mixed dz comes out about 5x larger than single on both platforms. The cause is not investigated; it is within
  tolerance.

With this check, the fast-path GPU aux z that c11 uses agrees with the offline evaluator within the spec tolerances
on CUDA and OpenCL, and the GPU aux energy matches 0.5 k (z - c)^2. Still not covered: a full production run on
the fast path. No CVaux MD has run with a CustomCVForce CV1.
