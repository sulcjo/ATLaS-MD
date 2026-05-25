# GAREUS peptide package

This repository is the modular package form of the historical `gareus_peptide.py`
workflow.  The top-level shim remains intentionally tiny so older commands keep
working while the implementation lives under the `gareus` package namespace.

## Entry points

All of these are intended to reach the same command-line workflow:

```bash
python gareus_peptide.py --help      # concise operational help
python -m gareus --help               # same concise help
python -m gareus -hh                  # method encyclopedia plus full option reference
gareus-peptide --help
gareus-energy-decompose -h            # energy-decomposition operational help
gareus-energy-decompose -hh           # energy equations, grouping definitions, caveats
gareus-test-run --dry-run              # print the tiny real workflow test command
gareus-test-run --check-deps           # check optional MD runtime dependencies
gareus-suggest-cvs --run-dir RUN       # heuristic CV/window suggestions from production arrays
```

The package also installs the shorter `gareus` console script.  Post-hoc peptide energy decomposition is exposed separately as `gareus-energy-decompose`.  A tiny end-to-end runtime test helper is exposed as `gareus-test-run`.  Analysis-driven CV/window suggestions are exposed as `gareus-suggest-cvs`.

## High-level run modes and friendly CV switches

The package now has a single high-level mode switch for the computational dynamics path:

```bash
# Historical/default path: GaMD + umbrella/REUS
gareus --seq CLN025 --run-mode gamd --out run_gamd

# Conventional umbrella/REUS MD; does not import or require gamd-openmm
gareus --seq CLN025 --run-mode cmd --out run_cmd

# HMR + conventional umbrella/REUS MD; no gamd-openmm, 4 fs default
gareus --seq CLN025 --run-mode hmr-cmd --out run_hmr_cmd

# HMR + GaMD/REUS; 4 fs default
gareus --seq CLN025 --run-mode hmr-gamd --out run_hmr_gamd
```

`--run-mode cmd` keeps the same peptide setup, solvation, equilibration, umbrella forces, starting-structure pulling, replica construction, exchange logic, logging, checkpointing, and MBAR-ready bias outputs.  It only replaces the gamd-openmm production integrators with OpenMM `LangevinMiddleIntegrator` instances and skips shared GaMD calibration.  `--run-mode hmr-cmd` does the same thing with hydrogen mass repartitioning and a 4 fs production timestep by default.  `--run-mode hmr-gamd` likewise enables HMR and defaults to 4 fs for the GaMD path.  Explicit `--timestep-fs` values, or `timestep_fs` set in a config file, are always respected.  Use `--production-steps N` as the run-mode-neutral alias for the historical `--gamd-production-steps N`.

For CV selection, the shorter switches are now preferred for routine use:

```bash
# 1D terminal-distance umbrella
gareus --seq CLN025 --cv1 distance --out run_distance

# 1D nonlocal-contact umbrella
gareus --seq CLN025 --cv1 contacts --out run_contacts

# 2D distance x explicit Ramachandran basin-map umbrellas; centers are inserted automatically
gareus --seq CLN025 --cv1 distance --cv2 rama-map --out run_distance_rama_map

# 2D contact x alpha/coil/beta umbrellas
gareus --seq CLN025 --cv1 contacts --cv2 alpha-coil-beta --out run_contact_acb
```

`--cv1 distance` maps to the historical terminal-distance primary CV.  `--cv1 contacts` maps to `--primary-cv nonlocal-contacts`.  Any non-`none` `--cv2` value deliberately creates a 2D primary x secondary workflow when no explicit `--secondary-cv-centers` are supplied: `rama-map` uses explicit beta/PPII/right-alpha/left-alpha centers `[-1, -1/3, 1/3, 1]`, legacy `rama-regions` uses `[-1, -0.5, 0, 0.5, 1]`, `alpha-coil-beta` uses `[-0.8, 0, 0.8]`, and scalar content CVs use `[0.2, 0.5, 0.8]`.  Expert flags such as `--primary-cv`, `--secondary-cv`, and `--secondary-cv-centers` remain available for exact control.

## CV-aware GENPEPT seeding

GENPEPT survivor libraries can now seed GAREUS windows using the active CV definitions instead of only terminal distance.  Point GAREUS at a seed directory containing `final_survivor_seeds.csv`:

```bash
gareus --seq CLN025 --cv1 distance --cv2 rama-map \
  --seed-conformers-dir chignolin_genpept_seeds \
  --out run_seeded_distance_rama_map

# Contact-primary seeding is supported too
gareus --seq CLN025 --cv1 contacts --cv2 rama-map \
  --seed-conformers-dir chignolin_genpept_seeds \
  --out run_seeded_contacts_rama_map
```

Default behavior is:

```text
--seed-selection-mode auto     # equivalent to active-cv
```

`active-cv` scores each survivor against the target umbrella window using CV1 and, when CV2 is active and can be evaluated on the survivor, a scaled CV2 mismatch.  The old distance-only behavior remains available for reproducibility/debugging:

```bash
gareus --seq CLN025 --seed-conformers-dir seeds --seed-selection-mode distance
```

Useful controls:

```bash
--seed-selection-mode auto|active-cv|primary|distance
--seed-secondary-weight 1.0
--seed-max-reuse-per-conformer 0
```

The selector writes audit files under `us_starting_structures/`:

```text
seed_selection_report.csv
seed_selection_report.json
graft_report.json
```

The actual production ensemble is still defined by the explicit-solvent OpenMM system, umbrella forces, optional GaMD/CMD mode, and exchange.  GENPEPT seeds are smarter starting guesses, not a sneaky alternate thermodynamic state.

## Analysis-driven CV suggestion report

After a pilot or completed run, use:

```bash
gareus-suggest-cvs --run-dir run_cln025
gareus-suggest-cvs --run-dir run_cln025 --print
```

It writes:

```text
cv_suggestions.md
cv_suggestions.json
```

The report reads `analysis_arrays.npz` when available, otherwise `analysis_chunks_manifest.json`, and flags symptoms such as weak neighbor overlap, primary-CV edge pileup, missing secondary-CV coverage, and distance-only runs that should be cross-checked with contacts or Ramachandran-region CVs.  It is a heuristic next-run planner, not a substitute for MBAR uncertainty or independent-repeat convergence.


## Help tiers

The CLI intentionally exposes two levels of help:

- `-h` / `--help` prints a concise front door with common commands, key flags, and expected outputs.
- `-hh` / `--help-heavy` prints an encyclopedia-style reference covering the applied methods, equations, workflow design, the complete start-to-finish MD methodology chapter, and the complete argparse option list.

## Solvation default

Fresh builds now solvate with a dodecahedral periodic box by default:

```bash
gareus --seq CLN025 --box-shape dodecahedron
```

Use `--box-shape cube` to reproduce legacy cubic-box setup, or `--box-shape octahedron` when explicitly desired and supported by the installed OpenMM version.

## Science-neutral performance switches

Two optional switches reduce diagnostic overhead without changing the dynamics or exchange math:

```bash
gareus --seq CLN025 --no-sample-potential-energy --no-flush-every-log
```

- `--no-sample-potential-energy` skips diagnostic total-potential-energy reads in scalar sample/distance rows. Umbrella biases, GaMD state, exchange probabilities, trajectories, and forces are unchanged; potential-energy columns are left unavailable. Later peptide-pair decomposition still works from saved coordinates, so keep trajectory/final PDB outputs if you plan to analyze energies after the run.
- `--no-flush-every-log` lets scalar CSV/JSONL logs use the configured row buffers and checkpoint/final flushes instead of syncing after every reporting event. This improves slow-filesystem throughput but slightly reduces log durability after a hard crash.




## Tiny real workflow test

The ordinary unit tests avoid requiring OpenMM, PeptideBuilder, and `gamd-openmm`, so they can run on documentation/build machines.  The tiny test can also exercise `--run-mode cmd` and `--run-mode hmr-cmd`, which require OpenMM/PeptideBuilder but not gamd-openmm. To test the actual MD path on a configured workstation or cluster node, use:

```bash
gareus-test-run --out tiny_real_test --force
gareus-test-run --run-mode cmd --out tiny_cmd --force
gareus-test-run --run-mode hmr-cmd --out tiny_hmr_cmd --force
```

This command launches a deliberately small but real workflow:

```text
AA peptide build -> solvation -> minimization -> staged NVT/NPT equilibration ->
manual two-window umbrella setup -> umbrella starting-structure pull -> optional shared
GaMD setup -> four production steps -> at least one neighbor exchange attempt
```

It writes:

```text
tiny_integration_test_report.json
tiny_integration_test_report.md
```

The helper can be used safely on machines that do not have the MD stack installed:

```bash
gareus-test-run --dry-run
gareus-test-run --check-deps --skip-if-missing
```

Useful variants:

```bash
# Use CPU instead of Reference when available
gareus-test-run --platform CPU --setup-platform CPU --out tiny_cpu --force

# Exercise the conventional-MD path without gamd-openmm
gareus-test-run --run-mode cmd --out tiny_cmd --force
gareus-test-run --run-mode hmr-cmd --out tiny_hmr_cmd --force

# Print the exact underlying gareus invocation without running it
gareus-test-run --out tiny_real_test --dry-run

# Append raw arguments to the underlying gareus command
gareus-test-run --extra-arg=--box-shape --extra-arg cube --out tiny_cube --force
```

This is not a scientific validation run.  It is a plumbing test designed to catch broken imports, parser/runtime mismatches, OpenMM platform issues, file-output regressions, and tiny-run lifecycle problems before someone feeds the machine a real allocation and sacrifices a wall-clock goat.

## Provenance and reproducibility manifest

Every normal run now writes a machine-readable provenance manifest next to the older effective-config files:

```text
run_manifest.json
run_manifest.yaml
config/run_manifest.json
config/run_manifest.yaml
```

The older `effective_config.yaml/json` files remain the resolved-argument source of truth for rerunning or resuming.  The new manifest adds the information needed to audit *what actually ran*:

- run ID, start/end UTC timestamps, completion status, and error details if the run failed or was interrupted;
- full resolved CLI arguments, config source, command line, working directory, output directory, and scratch/main-dir mapping;
- Python/OS/hostname metadata and dependency versions for NumPy, OpenMM, `gamd-openmm`, PeptideBuilder, PyYAML, pdbfixer, and pymbar when available;
- available OpenMM platforms plus requested and resolved setup/production platform properties;
- force-field XML names, solvent model, PME/cutoff/constraint/HMR/run-mode settings, random seeds, CV/window/GaMD/exchange settings;
- combined source hash over `gareus/*.py`, `pyproject.toml`, `gareus_peptide.py`, and `GENPEPT.py`;
- hashes of input files such as `--config`, `--windows-2d-csv`, and GENPEPT seed tables;
- hashes of small state-defining artifacts such as built/solvated/equilibrated PDBs, umbrella-window tables, metadata, checkpoint manifests, and reports;
- trajectory and final-PDB directory summaries.

Large trajectory files are summarized by count and size rather than fully hashed by default, so provenance does not turn into a surprise second filesystem benchmark.

## Post-hoc peptide energy decomposition

`--no-sample-potential-energy` does **not** prevent later peptide energy analysis.  It only skips live total-PE diagnostics in `samples.csv`.  Intra/inter peptide energies are a different analysis problem: they must be reconstructed from saved coordinates plus a compatible topology after production.

Keep at least one coordinate source if you want frame-resolved energies later:

- `--traj-format dcd` or `--traj-format xtc` for production trajectories.
- `final_pdbs/*.pdb` for one-frame-per-replica final snapshots.
- Avoid `--traj-format none` if you expect a time series rather than final snapshots.

Basic use:

```bash
gareus-energy-decompose --run-dir run_cln025
```

The energy-decomposition command also has its own help tiers:

```bash
gareus-energy-decompose -h      # operational arguments
gareus-energy-decompose -hh     # equations, grouping definitions, caveats, output columns
```

Useful variants:

```bash
# Include peptide-vs-solvent/ion direct nonbonded terms; slower for solvated systems
gareus-energy-decompose --run-dir run_cln025 --include-peptide-environment

# Also evaluate the exact whole-system unbiased OpenMM force-field PE for each frame
gareus-energy-decompose --run-dir run_cln025 --write-total-forcefield-energy

# Analyze explicit files/globs
gareus-energy-decompose --topology run_cln025/02_npt_equilibrated.pdb \
  --trajectory 'run_cln025/replica_trajectories/*.dcd' \
  --out run_cln025/energy_decomposition.csv
```

### Energy-decomposition definitions

The default grouping is `--group-mode chains`, which treats each peptide chain as one peptide molecule:

- `intrapeptide_direct_nonbonded_kj_mol`: atom pairs within the same peptide group/chain.
- `interpeptide_direct_nonbonded_kj_mol`: atom pairs between different peptide groups/chains. This is expected to be zero for a single-peptide system.
- `peptide_peptide_direct_nonbonded_kj_mol`: intra + inter direct peptide-pair terms.
- `peptide_environment_direct_nonbonded_kj_mol`: optional peptide-vs-solvent/ion term from `--include-peptide-environment`.
- `total_forcefield_kj_mol`: optional exact whole-system unbiased OpenMM potential-energy scalar from `--write-total-forcefield-energy`.

Alternative grouping modes:

- `--group-mode all-peptide`: all peptide atoms are one group; useful for single-chain intrapeptide totals.
- `--group-mode residues`: each residue is a group; useful for residue-residue style diagnostics, but noisier and more expensive.

The direct pair terms use the OpenMM `NonbondedForce` particle and exception parameters.  For each selected atom pair, the code evaluates Coulomb plus Lennard-Jones direct-space terms:

```text
E_ij = k_e q_i q_j / r_ij + 4 epsilon_ij [(sigma_ij / r_ij)^12 - (sigma_ij / r_ij)^6]
```

Exception pairs use the explicit exception charge product, sigma, and epsilon. Non-exception pairs use Lorentz-Berthelot mixing. These terms are intentionally labeled `direct_nonbonded`: they are useful, reproducible peptide-pair diagnostics, but they are **not** an exact pair decomposition of PME reciprocal-space electrostatics. Use `--write-total-forcefield-energy` when you also want the exact whole-system unbiased OpenMM PE scalar per frame.

For DCD/XTC trajectories, install `mdtraj`; PDB snapshots under `final_pdbs/` work without it. Because apparently even reading coordinates needs a side quest.

## Package layout

- `gareus/core.py` - stable package entrypoint.
- `gareus/cli.py` - argument parser and top-level workflow dispatch.
- `gareus/system_setup.py` - peptide construction, platform selection, minimization/equilibration setup.
- `gareus/cv.py`, `gareus/windows.py`, `gareus/forces.py` - CV, window, and force-building helpers.
- `gareus/seeding.py` - umbrella-start seeding and GENPEPT-output integration.
- `gareus/production.py` - conventional/GaMD umbrella-REUS production loop, checkpointing, analysis array output.
- `gareus/energy_decomposition.py` - post-hoc peptide-pair direct nonbonded energy decomposition.
- `gareus/cv_discovery.py` - heuristic CV/window suggestions from production arrays.
- `gareus/provenance.py` - run manifest, environment snapshots, source/input/output hashes.
- `gareus/adaptive_feedback.py` - adaptive pilot/final workflow.
- `gareus/legacy.py` - compatibility re-export surface for old imports from `gareus_peptide`.

## Compatibility policy

`gareus_peptide.py` should stay a thin shim.  New implementation code belongs in
`gareus/*`, and compatibility exports should be routed through `gareus.legacy`
instead of copying monolithic code back into the shim.

## Minimal smoke checks

The lightweight checks do not require OpenMM:

```bash
python -m py_compile gareus/*.py gareus_peptide.py GENPEPT.py
python -m gareus --help
python -m gareus -hh
pytest
```

A real end-to-end plumbing check requires the OpenMM stack and appropriate GPU/CPU
platform configuration:

```bash
gareus-test-run --check-deps --skip-if-missing
gareus-test-run --out tiny_real_test --force
gareus-test-run --run-mode cmd --out tiny_cmd --force
gareus-test-run --run-mode hmr-cmd --out tiny_hmr_cmd --force
```

## Combined GENPEPT → GAREUS YAML workflow

The package can now keep the seed-generation and production settings in one YAML file.
Use a top-level `conformer_generation:` block for GENPEPT and the normal GAREUS
blocks for the production run. GAREUS ignores `conformer_generation`, while
`GENPEPT.py --config` reads that block.

Typical sequence:

```bash
python GENPEPT.py --config chignolin_2d_distance_with_genpept.yaml

gareus --config chignolin_2d_distance_with_genpept.yaml
```

The conformer stage should write `final_survivor_seeds.csv` under the configured
GENPEPT output directory. The GAREUS config should then point
`seed_conformers_dir` at that same directory. GAREUS will score survivors in the
active CV space (`cv1` plus optional `cv2`) and write `seed_selection_report.*`
for auditability. For example:

```yaml
conformer_generation:
  seq: GYDPETGTWG
  out: chignolin_genpept_seeds/
  n: 50000
  n_candidate_seeds: 100
  n_final_seeds: 32
  two_stage: true
  basin_hop: true

starting_structures:
  seed_conformers_dir: chignolin_genpept_seeds/
```

The GENPEPT and GAREUS terminal displays are not a single linked TUI. GENPEPT has
its own console/live-histogram display, while GAREUS uses `gareus/tui.py` and its
`--tui-mode`/`--progress-mode` dashboard. The data link between the two stages is
the seed directory, especially `final_survivor_seeds.csv`.
