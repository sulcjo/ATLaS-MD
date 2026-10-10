# CVaux Stage C Task 15: GPU parity and cost runbook (aurum2)

Run by the user. Nothing here was executed by the implementer. Measures stored-vs-offline auxiliary z parity
(spec Sections 10 and 17) and the aux cost against a no-aux baseline.

## 0. Hard rules

- **NEVER rsync to `~/2026_peptide_sampler` or `~/gareus` while chignolin_10 runs there.** Those are the live
  launcher CODE_DIR and deployed tree. Use only the scratch dir below.
- Never `git push` for deployment, never `rsync --delete`.
- **Never loosen `PARITY_TOLERANCE`** (1e-6 double, 1e-4 mixed/single, `gareus/auxiliary_cv/sample_schema.py`) **or
  `_REL`** (1e-12, `gareus/auxiliary_cv/checkpoint.py`). On a miss: record the measured value and STOP. A change is
  a spec decision (Section 17: only after a documented numerical-error study).

## 1. Deploy to a scratch CODE_DIR

Branch `feat/cvaux-t15` (or the merged feat/cvaux tip), from the local worktree:

```
SCR=cvaux_parity_scratch            # on aurum2; must NOT be ~/2026_peptide_sampler or ~/gareus
rsync -a --exclude .git --exclude RUNS --exclude '*.log' \
  /run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/.claude/worktrees/cvaux-t15/ \
  sulcjo@aurum2:"$SCR"/
```

(no `--delete`; `ssh sulcjo@aurum2`, not `ssh aurum`). Run everything below with `cd $SCR`.

## 2. Inputs (from Task 14)

Task 14 defines configuration A, the 24-window CSV and the model build. Use:

- 24-window CSV: aux:ordinary 4:20, every row with instance columns, each auxiliary row's parent an ordinary row.
- Model built for this run's topology: Task 14 Step 4.2.
- Configuration A control command: Task 14.

Inputs are in the repo: `scripts/cvaux_parity/W24.csv` (20 ordinary + 2 auxiliary + 2 sham, each aux/sham pair on one
slot with an ordinary parent), `scripts/cvaux_parity/W24_noaux.csv` (same centres, no aux/instance columns) and
`scripts/cvaux_parity/make_model.py`. Configuration A from Task 14 (GA dipeptide, distance CV1, gibbs-walk), scaled to
24 windows:

```bash
F="--seq GA --cv1 distance --window-mode manual --exchange-mode gibbs-walk --seed 7 --minimize-iterations 200 \
   --npt-steps 2000 --us-pull-steps-per-window 500 --production-steps 6000 --checkpoint-interval 2000 \
   --exchange-interval 500 --distance-output-interval 250 --report-interval 250 --traj-interval 500 \
   --production-phase-timers"
# 1. non-aux prep run, then a model bound to ITS 01_solvated_start.pdb
python -m gareus $F --platform CUDA --production-steps 500 --run-mode cmd \
   --windows-2d-csv scripts/cvaux_parity/W24_noaux.csv --out $SCR/RUN_P
PYTHONPATH=. python scripts/cvaux_parity/make_model.py $SCR/RUN_P/01_solvated_start.pdb $SCR/M.json
CMD_A="python -m gareus $F --run-mode cmd --windows-2d-csv scripts/cvaux_parity/W24.csv --aux-cv-model $SCR/M.json"
CMD_A_NOAUX="python -m gareus $F --run-mode cmd --windows-2d-csv scripts/cvaux_parity/W24_noaux.csv"
```

Always use `PYTHONPATH=.` (repo root) for helper scripts so the scratch CODE_DIR's `gareus` is the one imported.

Common flags on every run (put them in `$CMD_A` once): `--production-phase-timers`, plus `--aux-cv-model <MODEL>` on aux runs. Below `$CMD_A` stands for
Task 14's configuration A command, `$OUT_*` for distinct output dirs.

## 3. Parity runs (three, same CSV and model)

```
# CUDA mixed (default precision)
$CMD_A --platform CUDA --out $OUT_CUDA_MIXED
# OpenCL
$CMD_A --platform OpenCL --out $OUT_OPENCL
# CUDA double
$CMD_A --platform CUDA --precision double --out $OUT_CUDA_DOUBLE
```

Flags confirmed in `gareus/cli.py`: `--platform`, `--precision` (default `mixed`), `--out`.

Report, after each finishes:

```
scripts/cvaux_gpu_parity.sh $OUT_CUDA_MIXED $OUT_OPENCL $OUT_CUDA_DOUBLE
```

Pass criterion: every segment `ok: true` at its recorded precision (double 1e-6, mixed 1e-4). The script is
read-only and launches nothing.

## 4. GPU checkpoint round trip (CUDA mixed)

Repeat Task 14's exception path (configuration A):

```
GAREUS_TEST_FAIL_AT_PROD_STEP=3500 $CMD_A --platform CUDA --out $OUT_RT   # dies at step 3500, exit 1
env -u GAREUS_TEST_FAIL_AT_PROD_STEP $CMD_A --platform CUDA --out $OUT_RT --resume
```

N = 3500 (between the 2000 and 4000 checkpoints, as in Task 14). `verify_aux_resume` must pass: `loadCheckpoint` restores the
auxiliary globals within `_REL = 1e-12` (Task 6, board condition 8). If it refuses, record the measured deviation
and stop. Do not loosen `_REL`. Then run `scripts/cvaux_gpu_parity.sh $OUT_RT`.

## 5. No-aux cost baseline

Same 24 windows, `--aux-cv-model` omitted, CSV minus the aux and instance columns, CUDA mixed,
`--production-phase-timers`, same step count and GPU allocation:

```
$CMD_A_NOAUX --platform CUDA --out $OUT_BASE
```

Note: top-level `ok` does not surface `bound_vacuous`. Check every segment's `bound_vacuous` flag; a vacuous segment (no active aux state, k_max 0) proves only finiteness, not parity.

## 6. Record (evidence file)

Write to the evidence file (commit with `git add -f`; *.md is gitignored):

1. `parity_report` output for each run (CUDA mixed, OpenCL, CUDA double, round-trip).
2. `sample.fetch` and `md` seconds per 2000 steps from `<run>/production_phase_timers.json`, aux vs baseline.
3. Node ns/day, aux vs baseline.
4. Any measured deviation on a failed gate, verbatim.

## 7. Gate

- All parity runs `ok` at their precision's tolerance, round trip resumes: Task 15 passes.
- Mixed above 1e-4, or `verify_aux_resume` refusal: record measured value, STOP, escalate. No tolerance edits.

## Note on resume parity (Stage C F3)

A resumed run is not bitwise equal to an uninterrupted one under default concurrent replica stepping, with or without
CVaux (CPU control: cv1 up to ~0.16-0.2 A). Bitwise resume was shown for one aux run on Reference with
`--active-replicas-per-gpu 1`. The GPU round trip above therefore checks `verify_aux_resume` (aux globals within `_REL`)
and the parity report, not trajectory equality.
