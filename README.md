# GAREUS peptide sampler

OpenMM explicit-solvent Amber14/PME peptide umbrella-sampling workflow with optional HMR-GaMD, REUS, adaptive window feedback, epoch-based adaptive production, and post-hoc MBAR/PMF outputs.

## Install

```bash
pip install -e ".[all]"   # openmm, pdbfixer, PeptideBuilder, pymbar, PyYAML, pytest
pip install -e ".[dev]"   # pytest only (no MD stack)
```

`gamd-openmm` is external and required only for `--run-mode gamd` / `hmr-gamd`.

## Quick start

```bash
gareus --seq CLN025 --out run_cln025
gareus --write-config-template chignolin.yaml
gareus --config chignolin.yaml --seq CLN025 --out run_cln025
```

## Help

```
gareus -h     # concise operational help (reproduced below)
gareus -hh    # method encyclopedia, equations, complete flag list
```

---

```
GAREUS peptide workflow
=======================

Run a peptide OpenMM workflow combining peptide setup, staged equilibration,
umbrella sampling, optional GaMD calibration/production, replica exchange between
umbrella states, adaptive window feedback, and MBAR/PMF-ready outputs.

Minimal command
---------------
    gareus --seq CLN025 --out run_cln025

Useful starting points
----------------------
    # write an editable YAML template, then run from it
    gareus --write-config-template chignolin.yaml
    gareus --config chignolin.yaml --seq CLN025 --out run_cln025

    # adaptive distance windows plus standard GaMD/REUS production
    gareus --seq CLN025 --window-mode adaptive --out run_distance

    # conventional umbrella/REUS MD without gamd-openmm
    gareus --seq CLN025 --run-mode cmd --window-mode adaptive --out run_cmd

    # HMR + conventional umbrella/REUS MD; defaults to 4 fs
    gareus --seq CLN025 --run-mode hmr-cmd --window-mode adaptive --out run_hmr_cmd

    # HMR + GaMD convenience mode; defaults to 4 fs
    gareus --seq CLN025 --run-mode hmr-gamd --out run_hmr_gamd

    # automatic pilot rounds that refine windows before final production
    gareus --seq CLN025 --window-mode adaptive-feedback --out run_feedback

    # epoch-based adaptive production with a global aggregate-MD runtime pool
    gareus --seq CLN025 --run-mode hmr-gamd --cv1 contacts --cv2 rama-map \
           --window-mode adaptive-production \
           --md-budget-ns 840 \
           --out run_adaptive_pool

    # double-adaptive: feedback pilots seed adaptive-production epochs/topups
    gareus --seq CLN025 --run-mode hmr-gamd --cv1 contacts --cv2 rama-map \
           --window-mode double-adaptive \
           --adaptive-rounds 3 \
           --ap-epochs 10 \
           --out run_double_adaptive

    # nonlocal-contact primary CV, with automatic contact-window calibration
    gareus --seq CLN025 --cv1 contacts --window-mode adaptive-feedback --out run_contacts

    # 2D primary-CV x secondary-structure workflow, automatic secondary centers
    gareus --seq CLN025 --cv1 distance --cv2 rama-map --out run_2d_rama_map

    # seed each umbrella window from GENPEPT survivors scored in active CV space
    python GENPEPT.py --config chignolin.yaml
    gareus --config chignolin.yaml --seed-conformers-dir chignolin_genpept_seeds

Common flags
------------
    --seq SEQUENCE                      One-letter peptide sequence. Required for runs.
    --out DIR                           Output directory.
    --config FILE                       YAML/JSON config; CLI flags override config values.
    --write-config-template [FILE]      Write a starter YAML config and exit.
    --resume                            Resume from checkpoints or setup states under --out.
    --seed INT                          Random seed.

    --run-mode MODE                     cmd, hmr-cmd, gamd, or hmr-gamd. cmd/hmr-cmd need no gamd-openmm.
    --window-mode MODE                  manual, adaptive, adaptive-feedback, adaptive-production, double-adaptive, or delaunay-feedback.
                                        delaunay-feedback: KDE basin anchors + Delaunay triangulation on pilot samples,
                                        iterates each round until stable; add --delaunay-coverage-scaffold to fill gaps.
    --cv1 MODE                          Primary CV: distance or contacts.
    --cv2 MODE                          Secondary CV; non-none auto-enables 2D centers.
    --cv2-centers C ...                 Override auto 2D centers (e.g. -0.8 0.0 0.8).
    --windows-a A ...                   Manual distance centers in Angstrom (--window-mode manual).
    --seed-conformers-dir DIR           Use GENPEPT survivors for CV-aware window starts.
    --seed-selection-mode MODE          auto/active-cv, primary, or legacy distance seed scoring.

    --box-shape SHAPE                   Solvent box shape: dodecahedron (default), cube, octahedron.
    --padding-nm NM                     Solvent padding around the peptide.

    --production-steps N                Production steps per replica.
    --exchange-mode MODE                neighbor, random-pair, all-pair-sweep, or gibbs-walk.
    --exchange-interval N               Steps between exchange attempts.
    --md-budget-ns NS                   Aggregate adaptive-production MD pool over all states/replicas.
    --ap-epochs N                       Maximum adaptive-production epochs before frozen final.
    --ap-final-pool-fraction F          Fraction of the runtime pool reserved for frozen final production.
    --traj-format FORMAT                dcd, xtc, or none.
    --no-sample-potential-energy        Skip live total-PE diagnostics.

    --platform NAME                     auto, CUDA, HIP, OpenCL, CPU, or Reference.
    --device-index LIST                 GPU device token(s), e.g. 0 or 0,1,2,3.
    --replica-device-mode MODE          auto, round-robin, single-context-split, or manual.
    --cpu-budget N                      Total CPU cores; distributed as floor(N/n_replicas) threads/replica (CPU platform only).
    --max-cpu-per-replica N             Cap threads per replica regardless of --cpu-budget.
    --max-replicas N                    Hard cap on replicas/windows for any run mode; windows truncated after generation/loading. 0 = no cap.
    --setup-cpu-threads N               CPU threads for setup/minimization/equilibration phase.
    --scratchdir DIR                    Use fast local scratch and mirror to --out at checkpoints.

    --self-test-primary-cv-force        Validate the selected primary CV force and exit.
    --progress-mode MODE                none, console, jsonl, or both.
    --tui-mode MODE                     dashboard, interactive, line, or none.

Post-hoc CV suggestion report
-----------------------------
    gareus-suggest-cvs --run-dir run_cln025 --print

Post-hoc energy decomposition
-----------------------------
    gareus-energy-decompose --run-dir run_cln025 --out run_cln025/energy_decomposition.csv
    gareus-energy-decompose --run-dir run_cln025 --include-peptide-environment --write-total-forcefield-energy
    gareus-energy-decompose -hh

Tiny real workflow test
-----------------------
    gareus-test-run --dry-run
    gareus-test-run --run-mode cmd --dry-run
    gareus-test-run --check-deps --skip-if-missing
    gareus-test-run --out tiny_real_test --force

Core outputs
------------
    effective_config.yaml/json          Reproducible resolved configuration.
    run_manifest.json/yaml              Environment, dependency, source, platform, and artifact hashes.
    umbrella_windows.csv                Final umbrella centers and force constants.
    umbrella_explicit_windows.csv       Canonical window-major table, including sparse 2D cases.
    segments.json                       Restart/resume chain; one entry per gareus invocation.
    windows/<seg_id>.json               Per-segment window snapshot (centers, k, CV types).
    samples/<seg_id>/chunk_*.parquet    Per-step CV and energy data (~44 bytes/sample).
    exchanges/<seg_id>/chunk_*.parquet  Exchange events (step, replicas, delta_E, accepted).
    exchange_tuning_report.md/json      Practical exchange diagnostics.
    final_run_report.md/json            Final validation and output summary.

Post-hoc analysis
-----------------
    from gareus.query import export_analysis_arrays_npz
    export_analysis_arrays_npz(run_dir, beta=1/(8.314e-3 * 300))

    from gareus.query import load_samples, load_windows, reconstruct_bias_matrix
    samples = load_samples(run_dir)
    windows = load_windows(run_dir)
    nk = reconstruct_bias_matrix(samples['cv1'], samples['cv2'], windows, beta)

Use -hh for the method encyclopedia, equations, design notes, and the complete flag list.
```

---

## Resume

`--resume` auto-detects run stage and continues from the furthest completed checkpoint:

```bash
gareus --config chignolin.yaml --resume
```

Covers: epoch-boundary resume for adaptive-production and double-adaptive modes, production checkpoint resume for plain REUS/GaMD runs. The equilibration state (`03_npt_equilibrated_state.xml`) is saved after NPT for reliable resume; for older runs without it, the nearest available GaMD setup state is used as a fallback.

## Combined GENPEPT → GAREUS workflow

```yaml
conformer_generation:
  seq: GYDPETGTWG
  out: chignolin_genpept_seeds/
  n: 50000
  two_stage: true
  basin_hop: true

starting_structures:
  seed_conformers_dir: chignolin_genpept_seeds/
```

```bash
python GENPEPT.py --config chignolin.yaml
gareus --config chignolin.yaml
```

GENPEPT survivors are scored against active CV1/CV2 windows. Audit files written to `us_starting_structures/seed_selection_report.*`.

## Smoke checks

```bash
python -m py_compile gareus/*.py gareus_peptide.py GENPEPT.py
python -m gareus --help
pytest

# real MD (requires OpenMM + GPU/CPU)
gareus-test-run --check-deps --skip-if-missing
gareus-test-run --out tiny_real_test --force
gareus-test-run --run-mode cmd --out tiny_cmd --force
```

## Package layout

| Module | Role |
|---|---|
| `gareus/cli.py` | Argument parser and workflow dispatch |
| `gareus/system_setup.py` | Peptide build, solvation, NVT/NPT equilibration |
| `gareus/cv.py`, `forces.py` | CV observation and OpenMM force construction |
| `gareus/windows.py`, `adaptive_feedback.py` | Umbrella windows, 2D grids, adaptive/Delaunay feedback |
| `gareus/adaptive_production.py` | Epoch-based adaptive production, runtime pool, registry |
| `gareus/production.py` | CMD/GaMD/REUS loop, exchanges, checkpoints |
| `gareus/seeding.py`, `genpept_prescan.py` | GENPEPT survivor scoring and prescan prior |
| `gareus/store.py`, `query.py` | Parquet/DuckDB sample storage and query layer |
| `gareus/analysis.py`, `energy_decomposition.py` | Validation and post-hoc analyses |
| `gareus/checkpoints.py` | Checkpoint and resume helpers |
| `gareus/provenance.py` | Run manifest, source/input/output hashes |
| `gareus/legacy.py`, `gareus_peptide.py` | Backward compatibility |
| `GENPEPT.py` | Conformer generation, BH, NMA, PCA frontier |
