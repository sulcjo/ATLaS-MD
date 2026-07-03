"""Human-oriented CLI help text for the GAREUS package.

The parser in :mod:`gareus.cli` intentionally has many expert flags.  This
module keeps the user-facing help split into two tiers:

* ``-h`` / ``--help``: concise operational help for starting a run.
* ``-hh`` / ``--help-heavy``: method encyclopedia plus the complete option list.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from typing import Iterable

__all__ = [
    "SimpleHelpAction",
    "HeavyHelpAction",
    "simple_help_text",
    "heavy_help_text",
]


_OVERVIEW = """
GAREUS peptide workflow
=======================

Run a peptide OpenMM workflow combining peptide setup, staged equilibration,
umbrella sampling, optional GaMD calibration/production, replica exchange between
umbrella states, adaptive window feedback, and MBAR/PMF-ready outputs.

The short help is deliberately practical.  The complete method reference and
full flag list are available with:

    gareus -hh
    python -m gareus -hh
"""


_SIMPLE_HELP = """
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
    --profile NAME                      Named bundle of boilerplate config keys (config_profiles.py),
                                        shared with GENPEPT.py. Explicit YAML/CLI keys override it.
    --write-config-template [FILE]      Write a starter YAML config and exit.
    --resume                            Resume from checkpoints or setup states under --out.
    --seed INT                          Random seed.

    --run-mode MODE                     cmd, hmr-cmd, gamd, or hmr-gamd. cmd/hmr-cmd need no gamd-openmm.
    --window-mode MODE                  manual, adaptive, adaptive-feedback, adaptive-production, double-adaptive, or delaunay-feedback.
                                        delaunay-feedback: KDE basin anchors + Delaunay triangulation on pilot samples,
                                        iterates each round until stable; add --delaunay-coverage-scaffold to fill gaps.
    --cv1 MODE                          Primary CV: distance or contacts.
    --contact-atom-selection MODE       heavy, ca, backbone-heavy, sidechain-heavy, sidechain-all, or all.
    --cv2 MODE                          Secondary CV: none, alpha, beta, alpha-coil-beta, rama-map, torsion-pca, custom, or tica-linear.
                                        Non-none auto-enables 2D centers.
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

    # Inter-epoch tICA secondary-CV update (all off by default):
    --tica-obs-interval N               Record backbone dihedral features every N samples (0=off).
    --tica-lag-frames N                 Lag tau for tIC1 generalised eigenvalue problem (default 50).
    --tica-epochs-per-cycle N           Auto-refit tICA every N completed epochs (0=off; OR with --tica-update-after-epochs).
    --tica-update-after-epochs E ...    Explicit epoch list for tICA refit, e.g. 2 5 8.
    --tica-min-eigenvalue V             Skip refit if tIC1 eigenvalue below V (default 0.0=always update).
    --tica-switch-cv2                   After first successful tICA fit, permanently switch CV2 type to tica-linear.
                                        One-shot: triggers once, persists through final production and on --ap-resume.
    --tica-linear-k-min V               cv2_k_min applied after the CV2 switch (default 5.0 kcal/mol).
    --tica-linear-k-max V               cv2_k_max applied after the CV2 switch (default 50.0 kcal/mol).

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
    # after a run, generate heuristic CV/window suggestions from production arrays
    gareus-suggest-cvs --run-dir run_cln025 --print

Post-hoc energy decomposition
-----------------------------
    # after a run, decompose saved coordinates into peptide intra/inter direct nonbonded terms
    gareus-energy-decompose --run-dir run_cln025 --out run_cln025/energy_decomposition.csv

    # include slower peptide-vs-environment terms and exact whole-system unbiased PE
    gareus-energy-decompose --run-dir run_cln025 --include-peptide-environment --write-total-forcefield-energy

    # deeper energy-decomposition reference, equations, grouping definitions, caveats
    gareus-energy-decompose -hh

Tiny real workflow test
-----------------------
    # print the exact miniature OpenMM/GaMD workflow command
    gareus-test-run --dry-run

    # print/run a miniature workflow that avoids gamd-openmm
    gareus-test-run --run-mode cmd --dry-run

    # check optional MD dependencies/platforms without running MD
    gareus-test-run --check-deps --skip-if-missing

    # on a configured MD machine, run the tiny end-to-end plumbing test
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
    # reconstruct analysis_arrays.npz on demand (auto-generated by validate/analyze commands)
    from gareus.query import export_analysis_arrays_npz
    export_analysis_arrays_npz(run_dir, beta=1/(8.314e-3 * 300))

    # query samples directly
    from gareus.query import load_samples, load_windows, reconstruct_bias_matrix
    samples = load_samples(run_dir)           # dict of numpy arrays
    windows = load_windows(run_dir)           # list of window dicts
    nk = reconstruct_bias_matrix(samples['cv1'], samples['cv2'], windows, beta)

Inter-epoch tICA secondary-CV update (YAML)
--------------------------------------------
tICA fits the slowest backbone-dihedral mode from accumulated epoch observations
and re-centers umbrella windows in that coordinate after each cycle.  CV2-type
agnostic: works with rama-map, contacts, tica-linear, or any other secondary CV.

YAML block (add under top-level section named "tica:", all keys are optional):

    tica:
      tica_obs_interval: 50        # record dihedral features every N samples; >0 to enable
      tica_lag_frames: 50          # lag tau (frames) for tIC1; increase for slower modes
      tica_epochs_per_cycle: 3     # refit every 3 completed epochs (recommended >= 3;
                                   # N=1 means each refit excludes all prior epochs from
                                   # MBAR pooling because tica_cv_version bumps every cycle)
      tica_min_eigenvalue: 0.0     # skip refit if tIC1 eigenvalue below threshold

    # OR use an explicit epoch list instead of (or in addition to) the cycle:
    tica:
      tica_obs_interval: 50
      tica_update_after_epochs: [2, 5, 8]

The tica: section name is cosmetic.  The config loader flattens any nested dict
so you may put the tica_* keys in any section (cvs:, windows:, etc.).
The tica_state_file key is managed automatically; leave it empty or absent.

Use -hh for the method encyclopedia, equations, design notes, and the complete flag list.
"""


_METHOD_ENCYCLOPEDIA = """
Method encyclopedia
===================

0. Exact MD methodology, start to finish
----------------------------------------
The current package implements an explicit-solvent Amber/OpenMM peptide MD
workflow with staged equilibration, harmonic umbrella sampling, optional shared
GaMD calibration, optional GaMD production replicas, REUS-style exchange between
umbrella states, optional adaptive-feedback pilot refinement, and post-hoc
energy decomposition.  In compact form:

    OpenMM explicit-solvent Amber14 PME peptide CMD/GaMD/REUS with harmonic
    umbrella windows, staged NVT/NPT equilibration, optional HMR at 4 fs,
    optional shared article-style GaMD calibration, NPT production replicas,
    Metropolis exchange between
    umbrella states, optional 2D secondary-structure umbrellas, optional
    adaptive-feedback window refinement, and post-hoc direct nonbonded
    intra/inter peptide energy decomposition.

Default invocation:

    gareus --seq CLN025 --out run_cln025

High-level dynamics modes:

    --run-mode gamd      Historical/default GaMD + umbrella/REUS workflow.
    --run-mode cmd       Conventional Langevin umbrella/REUS MD; skips gamd-openmm entirely.
                         Use --production-steps N for the production length.
    --run-mode hmr-cmd   Conventional HMR umbrella/REUS MD; skips gamd-openmm and
                         defaults to 4 fs unless --timestep-fs is explicit.
    --run-mode hmr-gamd  HMR + GaMD/REUS; defaults to 4 fs unless --timestep-fs is explicit.

CV shortcuts (schema v2.0):

    --cv1 distance       Primary terminal-distance umbrella CV.
    --cv1 contacts       Primary smooth nonlocal-contact fraction CV.
    --cv2 rama-map       Enable explicit Ramachandran basin-map CV; auto-inserts
                         centers [-1, -1/3, 1/3, 1] (beta/PPII/right-alpha/left-alpha).
    --cv2 torsion-pca    Fit bootstrap backbone-torsion PC1 from GENPEPT seed conformers
                         and bias the same shared linear torsion force used by tica-linear.
    --cv2 alpha-coil-beta
                         Signed alpha-minus-beta CV; auto-inserts [-0.8, 0, 0.8].
    --cv2-centers C ...  Override auto-inserted 2D centers explicitly.

--cv2 non-none deliberately creates a 2D primary x secondary window set.
Supply --cv2-centers to override the auto-inserted center ladder.

0.1 Peptide construction
~~~~~~~~~~~~~~~~~~~~~~~~
The peptide is built from the one-letter sequence with PeptideBuilder.  The
default initial backbone geometry is:

    phi   = -60 degrees
    psi   = -45 degrees
    omega = 180 degrees

The first structure is written as:

    00_built_peptide.pdb

The built residue sequence is checked against the requested sequence so a bad
construct does not silently continue into an expensive run.

0.2 Protonation, force field, and solvation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Hydrogens are added at:

    pH = 7.0

Default force-field files are:

    amber14-all.xml
    amber14/tip3p.xml

Default water model:

    TIP3P

Solvation uses OpenMM Modeller.addSolvent.  The current default periodic box is:

    rhombic dodecahedron

with:

    padding        = 1.0 nm
    ionic strength = 0.15 M
    neutralize     = True

The solvated starting system is written as:

    01_solvated_start.pdb

Available geometry overrides are:

    --box-shape cube
    --box-shape octahedron

Use cube only when reproducing older cubic-box behavior or when a platform/build
requires it.  The dodecahedron default typically reduces solvent count for
compact peptides while preserving the requested minimum padding.

0.3 OpenMM system definition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The default physical model is:

    force field:        Amber14
    nonbonded method:   PME
    cutoff:             0.8 nm
    Ewald tolerance:    1e-4
    constraints:        HBonds
    rigid water:        True
    HMR:                off by default, unless --run-mode hmr-cmd or hmr-gamd is used

Hydrogen mass repartitioning is enabled through the high-level run mode:

    --run-mode hmr-cmd
    --run-mode hmr-gamd

With HMR enabled, the repartitioned hydrogen mass is:

    3.024 amu (fixed internal default)

HMR run modes also default the production timestep to 4 fs when neither the CLI nor the config file sets --timestep-fs explicitly.

0.4 Staged equilibration
~~~~~~~~~~~~~~~~~~~~~~~~
The equilibration sequence is conservative and intentionally separates
minimization, restrained warming, density relaxation, and unrestrained NPT.

Stage 0.4.1: restrained energy minimization

    ensemble/system:  NVT-style system, no barostat
    restraints:       peptide heavy atoms
    restraint k:      2.0 kcal/mol/A^2
    minimization:     20,000 iterations
    output:           02_minimized.pdb

Stage 0.4.2: restrained NVT warmup

    integrator:       OpenMM LangevinMiddleIntegrator
    steps:            20,000
    timestep:         0.25 fs
    friction:         10 ps^-1
    temperature ramp: 50 K -> intermediate -> 300 K
    output:           02_nvt_warm.pdb

Stage 0.4.3: restrained NPT ramp

    barostat:             OpenMM MonteCarloBarostat
    pressure:             1 bar
    temperature:          300 K
    barostat frequency:   25 steps
    steps:                50,000
    timestep:             0.5 fs
    restraints:           still active
    friction:             10 ps^-1

Stage 0.4.4: unrestrained NPT continuation

    remaining steps:   npt_steps - npt_ramp_steps
    default:           100,000 - 50,000 = 50,000 steps
    final timestep:    min(--timestep-fs, 2 fs) when --npt-final-timestep-fs is 0
    default timestep:  2 fs
    restraints:        removed
    output:            02_npt_equilibrated.pdb

The relevant flag is:

    --npt-final-timestep-fs

where 0 means automatic min(--timestep-fs, 2 fs).

0.5 Primary collective variable
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The default primary CV is terminal C-alpha distance:

    --cv1 distance

Mathematically:

    x(q) = ||r_a(q) - r_b(q)||

Explicit atom selectors (expert):

    --cv-atom1 1:CA --cv-atom2 -1:CA

The alternative primary CV is a smooth nonlocal-contact fraction:

    --cv1 contacts

with:

    s_ij(r) = 0.5 * [1 - tanh(0.5 * beta * (r_ij - r0))]
    C(q)    = sum_ij w_ij s_ij / N

Default contact-CV settings are:

    r0                              = 4.5 A
    beta                            = 6.0 A^-1
    minimum sequence separation      = 4 residues
    atom selection                   = heavy atoms
    scheme                           = atom-pairs
    normalized                       = true

0.6 Umbrella windows
~~~~~~~~~~~~~~~~~~~~
Default window mode:

    --window-mode adaptive

For the default distance CV, the package estimates a terminal-distance range
from the sequence and constructs adaptive umbrella centers.  Default settings:

    --cv1-target-spacing       = 0 (auto: 2 A for distance)
    --cv1-range-min/max        = 0 (auto-detect from sequence/boundary pull)
    --cv1-boundary-pull-steps  = 5000 (steps per direction; 0 = skip)
    --cv1-boundary-pull-k      = 50.0 kcal/mol/CV² (aggressive pull force constant)

The historical manual centers are still available in manual mode:

    5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 21 A

selected by:

    --window-mode manual

For window i, the primary umbrella bias is:

    U_i(q) = 1/2 k_i [x(q) - x_i]^2

Distance-mode user units are:

    center: A
    k:      kcal/mol/A^2

OpenMM internal units are:

    center: nm
    k:      kJ/mol/nm^2

Adaptive k values use the spacing heuristic:

    sigma ~= spacing / overlap_sigma          (default overlap_sigma=2.3 → overlap ≈ 0.25)
    k     ~= RT / sigma^2

with default clamp:

    0.05 to 20 kcal/mol/A^2

0.7 Optional secondary CV and 2D umbrellas
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Default secondary CV mode is:

    --cv2 none  (omitted → 1D workflow)

If enabled, the package adds a second harmonic umbrella term on a smooth
backbone torsion-content score.  Available modes include:

    alpha
    beta
    alpha-coil-beta
    rama-map
    torsion-pca
    tica-linear
    custom

Secondary CVs use smooth periodic phi/psi scores:

    g(theta; theta0, sigma) = exp[-(1 - cos(theta - theta0)) / sigma^2]

Default angular width:

    sigma = 35 degrees

Multiple secondary centers create a 2D umbrella grid:

    primary centers x secondary (--cv2-centers)

Auto-inserted centers when --cv2 is set without --cv2-centers:
    rama-map:         [-1, -1/3, 1/3, 1]
    alpha-coil-beta:  [-0.8, 0, 0.8]

Explicit sparse 2D windows can instead be loaded with:

    --windows-2d-csv

The secondary umbrella uses the same harmonic form:

    U_i,2(q) = 1/2 k_i,2 [s(q) - s_i]^2

Default secondary force constant:

    50 kcal/mol/CV^2

0.8 Starting structures for umbrella replicas
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Default starting-structure mode is:

    --us-starting-structure-mode pull

The package therefore does not simply clone the NPT-equilibrated structure into
every window.  It generates one starting structure per umbrella window by
restrained CV pulling.  Default pull settings are:

    5,000 steps per window          (--us-pull-steps-per-window)
    pull k = 5 kcal/mol/CV^2       (--us-pull-k; unified for all CV types)
    pull friction = 10 ps^-1       (--us-pull-friction-per-ps)
    pull timestep = min(production timestep, 2 fs)
    minimization at pulled centers = 100 iterations
    ramp stages for contact / 2D secondary = 8  (--us-pull-ramp-stages)

For 2D CVs, staged relaxation can first relax the primary CV and then ramp the
secondary CV.  If --seed-conformers-dir points to a GENPEPT output, nearby
conformer seeds can be selected or grafted first, then the normal CV-pull
preparation is still performed.

0.9 Production system
~~~~~~~~~~~~~~~~~~~~~
Default production ensemble:

    --production-ensemble npt

Production therefore includes:

    MonteCarloBarostat
    pressure    = 1 bar
    temperature = 300 K
    barostat frequency = 25 steps unless overridden

The older fixed-box behavior is available explicitly with:

    --production-ensemble nvt

The production system includes:

    Amber force field
    PME electrostatics
    primary umbrella force
    optional secondary umbrella force
    GaMD integrator
    optional barostat

0.10 GaMD setup
~~~~~~~~~~~~~~~
GaMD is provided through gamd-openmm.  Default boost type:

    lower-dual

Default GaMD parameters:

    --sigma0p              = 6 kcal/mol   (primary boost sigma)
    --sigma0d              = 6 kcal/mol   (secondary boost sigma, dual modes)
    --equil-steps          = 50,000       (GaMD equilibration/calibration)
    --gamd-cmd-steps       = 250,000      (CMD pre-equil for Vmax/Vmin stats; longer reduces anharmonicity)
    --production-steps     = 500,000      (production per replica)
    --gamd-averaging-window = 5,000 steps
    cmd/equil prep steps   = 5,000/5,000  (hardcoded internal defaults)

The package uses an article-style shared GaMD setup:

    1. Build the production-like system.
    2. Include umbrella force objects, but set umbrella force constants to zero.
    3. Run one shared GaMD calibration/equilibration from an equilibrated structure.
    4. Save the GaMD integrator/global state.
    5. Copy that same GaMD state into every umbrella replica.
    6. Start all replicas directly in the GaMD production phase.

It does not run a full independent GaMD calibration for each umbrella window.
The intention is a common GaMD boost setup with different umbrella thermodynamic
states.

Relevant outputs include:

    shared_gamd_setup_globals.json
    shared_gamd_setup_context.chk
    shared_gamd_setup_state.xml
    shared_gamd_setup_final.pdb
    replica_shared_gamd_copy_report.json

0.11 Replica construction
~~~~~~~~~~~~~~~~~~~~~~~~~
The number of production replicas equals the number of umbrella windows.  Each
replica receives:

    same topology/system definition
    same shared GaMD setup state
    different umbrella center/k assignment
    window-specific starting coordinates from the pull stage

Default production integrator settings are:

    timestep    = 2 fs
    friction    = 1 ps^-1
    temperature = 300 K

Trajectory defaults are:

    format   = DCD
    interval = 5,000 steps

with optional:

    --traj-format xtc
    --traj-format none

0.12 GaREUS production and exchange
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Default exchange mode:

    --exchange-mode neighbor

Default exchange interval:

    5,000 steps

At exchange time, the package reads positions, computes current CV values,
builds the full window-by-replica umbrella bias matrix, and attempts swaps.  For
a proposed swap between windows i and j holding replicas a and b:

    old      = U_i(x_a) + U_j(x_b)
    new      = U_j(x_a) + U_i(x_b)
    Delta    = new - old
    P_accept = min(1, exp(-beta Delta))

Only umbrella bias differences enter this exchange criterion.  The shared GaMD
boost is not included in the exchange delta because all replicas share the same
GaMD thermodynamic setup; the exchange is between umbrella assignments.

Available exchange modes are:

    neighbor          adjacent ladder or 2D neighbor graph
    random-pair       random disjoint Metropolis pairs
    all-pair-sweep    randomized many/all-pair Metropolis sweep
    gibbs-walk        heat-bath-like long-jump reassignment

For explicit sparse 2D windows, neighbor exchange uses a geometry graph rather
than a flattened ladder.

0.13 Adaptive-feedback mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Default window mode is adaptive, not adaptive-feedback.  If run with:

    --window-mode adaptive-feedback

then the package performs:

    pilot rounds -> diagnose overlap/exchange/coverage -> propose windows -> final production

Default adaptive-feedback settings are:

    --adaptive-rounds        = 3
    --pilot-fraction         = 0.05 of production-steps
    --validation-steps       = -1 (auto: 1/10 of production-steps)
    --target-overlap         = 0.25
    --aggressiveness         = balanced

The final production lives under:

    final_production/

Pilot folders are diagnostics, not final PMF inputs.  Adaptive feedback can add
midpoint windows, shift centers, prune over-resolved regions, create sparse 2D
local patches, and adapt secondary-CV centers when enabled.

0.13b Delaunay-feedback mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
`--window-mode delaunay-feedback` extends adaptive-feedback with data-driven
window placement from pilot CV samples.  Starting from the designated pilot round:

    pilot round N samples (cv_A, secondary_cv)
    -> KDE density field (Scott bandwidth)
    -> greedy KDE peak picking  -> basin_anchor windows
    -> [optional] coverage scaffold: grid-inject in uncovered [0,1]^2 cells
    -> Delaunay triangulation   -> delaunay_bridge at long edges
    -> [optional] circumcenter_probe (density-filtered)
    -> anisotropic force constants from local neighbor spacing
    -> explicit 2D window CSV  -> fed into subsequent adaptive-feedback rounds
    -> repeat each round until anchor set is stable (iterative mode)

Placement re-runs every pilot round >= delaunay-after-round (default) and
stops updating the CSV once anchors shift less than --delaunay-stability-tol.
A WARNING is printed whenever any Delaunay edge is >=2x the median edge length,
indicating a possible coverage gap in CV space.

Key controls:

    --delaunay-after-round N          pilot round to start Delaunay (default 1)
    --delaunay-iterate/--no-delaunay-iterate
                                      re-run each round >= trigger (default: on)
    --delaunay-stability-tol F        stop iterating when max anchor shift < F
                                      in normalised space (default 0.05)
    --delaunay-coverage-scaffold      inject grid anchors in cells with no KDE peak
    --delaunay-coverage-grid-n N      N×N scaffold grid (default 3)
    --delaunay-n-anchors N            max KDE basin anchors (default 16)
    --delaunay-bridge-min-edge F      min normalised edge for bridge windows (default 0.20)
    --delaunay-density-floor-q F      KDE percentile floor (default 0.05)
    --delaunay-k-sigma-factor F       sigma = factor × local spacing for k (default 0.50)
    --delaunay-dedup-radius F         min normalised distance between anchors (default 0.10)
    --delaunay-circumcenter-probes    optional circumcenter probe windows (off by default)

The coverage scaffold mitigates the blind-spot problem: if the pilot REUS does
not visit a CV region (high barrier, short pilot), KDE finds no peak there and
no anchor is placed.  Scaffold mode divides the normalised CV square into an
N×N grid and injects a "coverage_scaffold" window in each cell that has no
KDE anchor within dedup-radius, providing baseline coverage at no simulation cost.

Rounds 2-N and final production use the existing explicit-sparse pipeline
(build_explicit_2d_neighbor_edges + adaptive-feedback sparse-patch refinement).
The adaptive_production block is unused; delaunay-feedback dispatches to the
same run_adaptive_feedback_auto_loop as adaptive-feedback.

0.14 Adaptive-production mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
`--window-mode adaptive-production` runs epoch-based adaptive production rather
than a single fixed final production.  The workflow is:

    initial windows / GENPEPT prior
    -> adaptive production epochs
    -> diagnose overlap, exchange, coverage, GaMD boost health, and sample counts
    -> add/extend/optionally retire states
    -> convergence gate
    -> frozen final production
    -> post-hoc union-state MBAR inputs and quality reports

Useful controls (v2.0 names):

    --ap-epochs                        max adaptive epochs
    --ap-epoch-steps                   per-state epoch steps (fallback hint)
    --md-budget-ns                     aggregate MD pool over all states
    --ap-final-pool-fraction           fraction reserved for frozen final
    --ap-min-final-pool-ns             minimum ns reserved for frozen final
    --ap-final-steps                   per-state frozen-final steps (fallback hint)
    --ap-resume                        resume from state_registry.json or epoch checkpoints
    --ap-write-mbar-inputs             post-hoc union-state MBAR arrays
    --ap-run-mbar                      run PyMBAR after final

The runtime pool is aggregate MD, not per-replica trajectory length:

    consumed_ns = n_states * steps_per_state * timestep_fs / 1e6

For example, 16 states run for 250,000 steps at 4 fs consume 16 aggregate ns.
Active replica count grows adaptively; the md-budget-ns pool provides the
overall resource constraint.

0.15 Outputs for analysis
~~~~~~~~~~~~~~~~~~~~~~~~~
Production outputs include:

    segments.json                       Restart/resume provenance chain.
    windows/<seg_id>.json               Window snapshot per segment: centers, k, CV types.
    samples/<seg_id>/chunk_*.parquet    Per-step: step, replica, window_id, cv1, cv2,
                                        potential, gamd_boost_total/dihedral/nonbonded.
                                        ~44 bytes/sample; 10-20x smaller than CSV.
    exchanges/<seg_id>/chunk_*.parquet  Exchange events per segment.
    umbrella_windows.csv                Final umbrella centers and force constants.
    umbrella_explicit_windows.csv       Canonical window-major table.
    exchange_tuning_report.md/json
    final_run_report.md/json
    replica_trajectories/*.dcd or *.xtc
    final_pdbs/*.pdb

The umbrella bias matrix (umbrella_reduced_bias_nk) is NOT stored directly.
It is reconstructed analytically at analysis time from cv1/cv2 values and
window centers/k using the harmonic form:

    U_k(cv) = 0.5 * k_k * (cv - center_k)^2   [kcal/mol]
    reduced_bias[n,k] = beta * 4.184 * U_k(cv[n])

This is lossless for harmonic umbrellas and eliminates the N×K storage cost.

On-demand NPZ reconstruction for downstream MBAR tools:

    from gareus.query import export_analysis_arrays_npz
    export_analysis_arrays_npz(run_dir, beta=1/(8.314462618e-3 * T_K))

validate_analysis_metadata_readiness() calls this automatically when
analysis_arrays.npz is absent but Parquet data is present.

Multi-restart runs are queried across all segments transparently:

    from gareus.query import load_samples, load_windows, reconstruct_bias_matrix
    samples = load_samples(run_dir)           # unions all seg_001, seg_002, ...
    windows = load_windows(run_dir)           # latest segment window set
    nk = reconstruct_bias_matrix(samples['cv1'], samples['cv2'], windows, beta)

0.16 Potential-energy handling and later decomposition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
By default, sampled total potential energy is logged live.  The optional speed
flag:

    --no-sample-potential-energy

skips that extra diagnostic energy read.  It does not change:

    forces
    integration
    GaMD boost
    umbrella bias
    exchange probabilities
    trajectories
    checkpoints

It only skips live total-PE diagnostics.

For later intra/inter peptide energy analysis, use:

    gareus-energy-decompose --run-dir run_cln025

The post-processing command analyzes saved coordinates and computes direct-space
peptide pair terms, including:

    intrapeptide_direct_nonbonded_kj_mol
    interpeptide_direct_nonbonded_kj_mol
    peptide_peptide_direct_nonbonded_kj_mol
    peptide_environment_direct_nonbonded_kj_mol   optional
    total_forcefield_kj_mol                       optional exact whole-system PE

The pair decomposition is direct Coulomb plus Lennard-Jones using OpenMM
NonbondedForce particle and exception parameters.  It is not an exact pairwise
PME reciprocal-space decomposition.  The option --write-total-forcefield-energy
adds the exact whole-system unbiased OpenMM potential-energy scalar.

1. Workflow map
---------------
GAREUS is organized as a production peptide-sampling pipeline:

    sequence -> peptide build -> dodecahedral-box solvation/ions -> minimization -> staged NVT/NPT
    -> umbrella-window construction -> optional starting-structure pulling
    -> shared GaMD setup -> replica production -> umbrella-state exchange
    -> diagnostics, adaptive-feedback proposals, and MBAR/PMF-ready arrays.

The package keeps a backward-compatible top-level `gareus_peptide.py` shim, but
active implementation code lives under `gareus/*`:

    gareus/cli.py                 argument parsing and top-level workflow dispatch
    gareus/system_setup.py        peptide construction, solvation, platforms, equilibration
    gareus/cv.py                  primary and secondary collective-variable definitions
    gareus/forces.py              OpenMM umbrella/contact force construction
    gareus/windows.py             window expansion, explicit 2D graphs, state assignment
    gareus/seeding.py             umbrella starting structures and GENPEPT seed integration
    gareus/production.py          GaMD/REUS production, checkpoints, outputs
    gareus/adaptive_feedback.py   pilot/final adaptive-window refinement
    gareus/diagnostics.py         GaMD and US/MBAR validation diagnostics

2. Solvation and periodic box geometry
--------------------------------------
By default, GAREUS solvates with OpenMM Modeller.addSolvent using:

    boxShape = dodecahedron

with user-facing controls:

    --box-shape dodecahedron | cube | octahedron
    --padding-nm P
    --water-model tip3p | tip3pfb | spce | tip4pew
    --ionic-strength-molar I

For a compact peptide, a rhombic dodecahedron usually needs fewer waters than a
cubic box at the same minimum padding because it better approximates the volume
of a sphere around the solute.  This reduces atom count and PME work without
changing the requested solute-to-boundary padding.  Use `--box-shape cube` for
legacy reproducibility with older runs.  Use `--box-shape octahedron` only when
your OpenMM version supports it and you want that periodic geometry explicitly.

3. Primary collective variables
-------------------------------
Distance CV:
    x(q) = ||r_a(q) - r_b(q)||

By default, the terminal-distance CV uses terminal C-alpha atoms.  With
`--cv-mode terminal-n-c`, it uses terminal N/C selectors.  Explicit selectors
can be supplied with `--cv-atom1` and `--cv-atom2`.

Nonlocal-contact CV:
    s_ij(r) = 0.5 * (1 - tanh(0.5 * beta * (r_ij - r0)))
    C(q)    = sum_ij w_ij s_ij / N       when --contact-normalize is enabled
    C(q)    = sum_ij w_ij s_ij           with --no-contact-normalize

Relevant controls:
    --cv1 contacts
    --contact-scheme atom-pairs | residue-balanced | ca-pairs
    --contact-atom-selection heavy | ca | backbone-heavy | sidechain-heavy | sidechain-all | all
    --contact-min-sequence-separation N
    --contact-r0-a A
    --contact-beta-a-inv B

`atom-pairs` counts every selected atom pair directly.  `residue-balanced`
downweights atom pairs so each residue pair contributes roughly equally.
`ca-pairs` restricts the contact graph to C-alpha pairs.

4. Umbrella bias
----------------
For a window i with center x_i and force constant k_i, the primary umbrella is:

    U_i(q) = 1/2 k_i [x(q) - x_i]^2

Distance centers are user-facing Angstrom values and are converted internally to
nanometers for OpenMM.  Distance k values use kcal/mol/A^2 at the CLI and are
converted to kJ/mol/nm^2 internally.  Contact centers are dimensionless by
default and k values use kcal/mol/CV^2 at the CLI.

Secondary-CV bias uses the same harmonic form:

    U_i,2(q) = 1/2 k_i,2 [s(q) - s_i]^2

When a secondary CV is active, the total umbrella bias is the sum of primary and
secondary terms.  The analysis arrays store the per-window bias matrix used for
exchange and downstream reweighting.

5. Adaptive window force constants
----------------------------------
Spacing-derived k values are estimated from local center spacing:

    sigma ~= spacing / cv1_adaptive_overlap_sigma    (default 2.3 → overlap ≈ 0.25)
    k     ~= RT / sigma^2

Clamped by --cv1-k-min / --cv1-k-max (primary CV) and --cv2-k-min / --cv2-k-max
(secondary CV).  All CV types (distance, contacts, secondary) use the same
heuristic with CV-appropriate units.  The goal is a practical initial overlap
heuristic, not a theorem.

6. Secondary-structure CVs
--------------------------
Secondary CVs are smooth torsion-content scores over backbone phi/psi torsions.
The elementary torsion score has the periodic Gaussian-like form:

    g(theta; theta0, sigma) = exp(-(1 - cos(theta - theta0)) / sigma^2)

Supported modes include:
    alpha                 alpha-like phi/psi content
    beta                  beta-like phi/psi content
    alpha-coil-beta       signed alpha-minus-beta transition coordinate
    rama-map              explicit Ramachandran basin-map coordinate
    torsion-pca           bootstrap seed-fit backbone torsion PC1
    tica-linear           learned tIC1 backbone torsion coordinate
    custom                target supplied by --cv2-phi0-deg/--cv2-psi0-deg

For `alpha-coil-beta`, alpha-like states are positive, beta-like states are
negative, and coil/disordered states tend toward zero.  For `rama-map`, the code
uses named smooth phi/psi basins: beta/extended (-135,+135), PPII/coil
(-75,+145), right-alpha (-60,-45), and left-alpha (+60,+40).  It remains a
one-dimensional secondary map coordinate crossed with CV1, not a dense phi/psi
grid.  `torsion-pca` and `tica-linear` both use the shared linear torsion force,
but `torsion-pca` is a bootstrap PC fit from seed conformers whereas
`tica-linear` is the inter-epoch learned slow-mode coordinate.

7. 2D and sparse windows
------------------------
With multiple `--cv2-centers`, GAREUS normally creates a rectangular
cross-product of primary centers x secondary centers.  `--windows-2d-csv` can
instead load an explicit per-window table, which supports sparse local patches
from adaptive feedback.

GENPEPT prescan (`--genpept-prescan`) reads GENPEPT output stages, rescores with
active CV1/CV2 definitions, and uses the occupancy map as a window prior and seed
library.  This is a sampling prior only, not PMF/MBAR evidence.

Neighbor exchange for explicit/sparse 2D tables uses a geometry graph with
k-nearest/radius edges (internal defaults; not user-facing in v2.0).

The graph is written to diagnostic files so the exchange topology is auditable.

8. Starting structures and seed integration
-------------------------------------------
`--us-starting-structure-mode pull` generates per-window starting conformers by
short restrained relaxation/pulling before production.  For 2D windows, staged
relaxation can first relax the primary CV and then ramp the secondary CV.

`--seed-conformers-dir` can point to a GENPEPT output directory containing
`final_survivor_seeds.csv`.  The package then chooses nearby conformer seeds in
the active GAREUS CV space before the normal CV-pull step.  In practical terms:

    --cv1 distance                  scores survivor terminal distance vs window center
    --cv1 contacts                  scores survivor nonlocal-contact CV vs window center
    --cv1 distance --cv2 rama-map
                                    scores both distance and secondary backbone basin

The selector writes:

    us_starting_structures/seed_selection_report.csv
    us_starting_structures/seed_selection_report.json

Controls:

    --seed-selection-mode auto|active-cv|primary|distance
    --seed-cv2-weight FLOAT          (weight of CV2 in active-cv scoring)
    --seed-max-reuse-per-conformer N

`auto` is currently equivalent to `active-cv`: score by CV1 and, when CV2 is
active and can be evaluated on the GENPEPT survivor, add a scaled CV2 mismatch.
`primary` ignores CV2.  `distance` preserves the old terminal-distance-only
behavior for reproducibility/debugging.  This improves initial coverage while
leaving the production umbrella definitions unchanged.

Combined GENPEPT -> GAREUS YAML files are supported.  Put GENPEPT settings in a
top-level `conformer_generation`, `genpept`, or `seed_generation` block.  GAREUS
ignores those blocks when parsing its config, while `python GENPEPT.py --config`
reads them.  A typical handoff is:

    python GENPEPT.py --config chignolin_2d_distance_with_genpept.yaml
    gareus --config chignolin_2d_distance_with_genpept.yaml

The two terminal displays are not one linked TUI.  GENPEPT uses its own
console/live-histogram display (`--live-hist`, `--hist-every`,
`--no-clear-screen`, `--no-color`).  GAREUS uses the package dashboard
(`--tui-mode`, `--progress-mode`).  The data link is the seed directory,
especially `final_survivor_seeds.csv`.

9. GaMD setup and production
----------------------------
GaMD setup uses the selected `--gamd-boost-type` and sigma controls:

    --gamd-boost-type
    --sigma0p
    --sigma0d
    --equil-steps
    --production-steps

The code performs a shared GaMD setup and copies compatible integrator global
state into production replicas.  Diagnostics compare copied global variables and
can be made strict with `--shared-gamd-copy-strict`.

10. Replica exchange between umbrella states
-------------------------------------------
For a proposed swap between windows i and j holding replicas a and b:

    old = U_i(x_a) + U_j(x_b)
    new = U_j(x_a) + U_i(x_b)
    Delta = new - old
    P_accept = min(1, exp(-beta Delta))

The code evaluates the vectorized bias matrix in kJ/mol.  Shared GaMD terms are
not part of the exchange delta because they are common to the swapped thermodynamic
assignment.  Exchange modes:

    neighbor          adjacent ladder or graph-neighbor swaps
    random-pair       arbitrary disjoint Metropolis window pairs
    all-pair-sweep    randomized sweep over many/all window pairs
    gibbs-walk        heat-bath-like long-jump state reassignment

For Gibbs-walk, candidate windows are sampled from normalized weights:

    w_j proportional to exp(-beta Delta_j)

11. Adaptive feedback
---------------------
`--window-mode adaptive-feedback` runs short pilot rounds, analyzes overlap,
exchange acceptance, center visitation, and coverage, then proposes a final fixed
window set.  It can add midpoint windows, shift centers, prune over-resolved
regions, and create sparse 2D local patches.

Important controls:
    --adaptive-rounds         (pilot refinement rounds)
    --pilot-fraction          (pilot steps as fraction of production-steps)
    --validation-steps        (last pre-production validation steps)
    --target-overlap          (target neighboring-window overlap)
    --aggressiveness          (conservative/balanced/aggressive/very-aggressive)
    --sparse-2d               (sparse local 2D midpoint patches)
    --region-memory           (stateful region classification between pilots)

Final production should be analyzed from `final_production/` when adaptive
feedback is used.  Pilot folders are diagnostics, not final PMF inputs.

12. Double-adaptive mode
------------------------
`--double-adaptive` is an alias for `--window-mode double-adaptive`.  It runs
adaptive-feedback pilot rounds first, intentionally skips adaptive-feedback's
`final_production/`, writes/uses a handoff window table, and then starts
`--window-mode adaptive-production` from that feedback-optimized proposal.

Output landmarks:

    adaptive_feedback_driver_summary.json
    double_adaptive_handoff_windows.csv       # for factorized feedback proposals
    double_adaptive_driver_summary.json
    adaptive_production/
    adaptive_production/final/                # clean frozen final phase

Use `adaptive_production/final/` for the conservative final PMF/MBAR stage.
There is no duplicate adaptive-feedback final run in this mode.

13. Adaptive production and global runtime pool
-----------------------------------------------
`--window-mode adaptive-production` turns production into an epoch-based
adaptive campaign.  Each adaptive epoch runs fixed-bias production, analyzes the
resulting samples/exchanges/overlaps, then updates the persistent state registry
before the next epoch.  Once the adaptive convergence gate passes, the window set
is frozen and the final clean production phase begins.

The persistent adaptive-production output directory contains:

    adaptive_production/state_registry.json/csv
    adaptive_production/lifecycle.jsonl
    adaptive_production/epoch_*/
    adaptive_production/final/
    adaptive_production/adaptive_runtime_pool.json/md
    adaptive_production/adaptive_union_mbar.*
    adaptive_production/adaptive_quality_gate*.json/md

The global runtime pool limits aggregate MD across the whole campaign:

    consumed_ns = n_states * steps * timestep_fs / 1e6

Thus 16 states for 250,000 steps at 4 fs cost 16 aggregate ns from the pool.
This is a resource budget, not the physical length of any single continuous
replica trajectory.  The final-pool fraction reserves part of the pool for the
frozen final phase, while adaptive epochs and topups consume the remainder.

Important controls:

    --md-budget-ns                     (aggregate MD pool over all states)
    --ap-final-pool-fraction           (fraction reserved for frozen final)
    --ap-min-final-pool-ns             (minimum ns reserved for frozen final)
    --ap-epochs                        (max adaptive epochs)
    --ap-epoch-steps                   (per-state epoch steps fallback hint)
    --ap-final-steps                   (per-state frozen-final steps fallback hint)
    --ap-resume                        (resume from state_registry.json or epoch checkpoints)
    --target-overlap / --ap-min-exchange
                                       (per-epoch edge health thresholds)
    --ap-write-mbar-inputs / --ap-run-mbar
                                       (post-hoc union-state MBAR outputs)

The current implementation uses safe epoch-worker execution rather than true
in-process OpenMM context reuse.  Context reuse flags write readiness reports and
fall back unless strict reuse is required.

13. Output and downstream analysis
----------------------------------
The run writes scalar CSV files for readability and chunked NPZ arrays for large
analyses.  Important outputs include:

    samples.csv                         scalar samples and per-frame diagnostics
    analysis_chunks/chunk_*.npz         chunked CV/bias arrays
    analysis_arrays.npz                 optional consolidated compatibility file
    umbrella_pymbar_metadata.json       MBAR/US metadata and validation information
    exchange_tuning_report.md/json      exchange health and tuning recommendations
    final_run_report.md/json            final run summary and warnings
    run_manifest.json/yaml              provenance manifest with versions, hashes, args, and status

The bias matrix stored for analysis uses:

    U_window(sample) = U_primary + U_secondary

with the same harmonic definitions used in exchange.  The outputs are meant to be
friendly to MBAR/PMF workflows, while retaining enough metadata to diagnose bad
window overlap before pretending the PMF is meaningful.  Science: proudly turning
uncertainty into files with names.

14. Analysis-driven CV suggestion report
-----------------------------------------
After a pilot or completed production run, `gareus-suggest-cvs` reads
`analysis_arrays.npz` or `analysis_chunks_manifest.json` and writes heuristic
recommendations for the next CV/window refinement pass:

    gareus-suggest-cvs --run-dir run_cln025
    gareus-suggest-cvs --run-dir run_cln025 --print

Outputs:

    cv_suggestions.md
    cv_suggestions.json

The report checks symptoms such as weak primary-window overlap, primary-CV
sampling piled up near edges, missing secondary-CV coverage, secondary CVs that
do not respond to their targets, and whether a distance-only run should be
compared with contacts or Ramachandran CVs.  It is deliberately heuristic:
it suggests what to test next, not what to publish as truth.  Apparently the
molecule still refuses to fill out a reaction-coordinate declaration form.

14. Provenance and reproducibility manifest
--------------------------------------------
Every normal run writes a richer provenance manifest in addition to the older
`effective_config.yaml/json` files:

    run_manifest.json
    run_manifest.yaml
    config/run_manifest.json
    config/run_manifest.yaml

The manifest records:

    run_id and start/end UTC timestamps
    completion status or failure/interruption error
    full resolved CLI arguments and config source
    Python, OS, hostname, executable, and working directory
    dependency versions: numpy, OpenMM, gamd-openmm, PeptideBuilder, PyYAML, pymbar
    available OpenMM platforms and requested/resolved platform properties
    force-field XML names and core OpenMM method settings
    package source hash over gareus/*.py, pyproject.toml, gareus_peptide.py, GENPEPT.py
    hashes of input files such as --config, --windows-2d-csv, and GENPEPT seed tables
    topology hashes for built, solvated, and equilibrated PDBs
    window table hashes for umbrella_windows.csv and umbrella_explicit_windows.csv
    analysis/checkpoint/report artifact hashes and trajectory/final-PDB summaries

Large trajectory files are summarized by size/count rather than fully hashed by
default, so provenance does not become a second large filesystem workload.  The
small files that define the scientific state -- topology, window tables, metadata,
checkpoint manifests, reports, and configs -- are hashed directly.

The manifest is best-effort and non-fatal: a failed optional version lookup or
missing optional artifact is recorded instead of aborting a long simulation.

15. Tiny real workflow test
----------------------------
The normal pytest smoke tests intentionally do not require OpenMM,
PeptideBuilder, pdbfixer, or gamd-openmm.  They verify parser/help/package
behavior, but they do not prove that the actual MD lifecycle runs on a given
machine.  For that, the package provides:

    gareus-test-run

This helper constructs and optionally executes a deliberately tiny real workflow:

    AA peptide build -> solvation -> minimization -> staged NVT/NPT equilibration
    -> manual two-window umbrella setup -> umbrella-start pulling -> shared GaMD
    setup -> four production steps -> one or more neighbor exchange attempts

Default underlying workflow command, in compact form:

    gareus --seq AA --window-mode manual --windows-a 3.5 4.5 \
      --equil-steps 1 --production-steps 4 --exchange-interval 2 \
      --traj-format none --progress-mode none --tui-mode none

Operational commands:

    gareus-test-run --dry-run
    gareus-test-run --check-deps --skip-if-missing
    gareus-test-run --out tiny_real_test --force
    gareus-test-run --platform CPU --setup-platform CPU --out tiny_cpu --force

The command writes:

    tiny_integration_test_report.json
    tiny_integration_test_report.md

and validates that key artifacts exist after completion, including:

    effective_config.yaml/json
    run_manifest.yaml/json
    00_built_peptide.pdb
    01_solvated_start.pdb
    02_minimized.pdb
    02_npt_equilibrated.pdb
    umbrella_windows.csv
    umbrella_explicit_windows.csv
    umbrella_pymbar_metadata.json
    samples.csv
    exchanges.csv
    final_run_report.md/json

This is a plumbing/integration test, not a scientifically meaningful sampling
run.  Its purpose is to catch broken optional imports, parser/runtime mismatches,
OpenMM platform/property problems, output regressions, and lifecycle failures
before a real production run burns walltime.

16. Stability and HPC notes
---------------------------
Staged equilibration uses conservative defaults: short-timestep NVT warmup,
NPT ramping, restraints, high friction, and small safe chunks.  Production has a
rollback probe to catch NaNs before consuming long walltime.

For HPC runs, use:
    --device-index 0,1,2,3
    --replica-device-mode auto
    --scratchdir /local/nvme/run_name
    --traj-format xtc or --traj-format none for I/O reduction

`--scratchdir` mirrors results back to `--out` at checkpoints, reducing pressure
on network filesystems during high-frequency trajectory/NPZ/CSV writes.

`--no-sample-potential-energy` skips the diagnostic potential-energy read in
sample/distance rows.  It does not change integration, forces, GaMD boost state,
umbrella biases, exchange probabilities, trajectories, or checkpoints; it only
leaves total potential-energy columns as NaN/blank.  Later peptide energy
decomposition still works if coordinate trajectories or final PDB snapshots are
kept.  Do not combine it with `--traj-format none` if you expect frame-by-frame
post-hoc energies later.

CSV/JSONL buffering thresholds are internal defaults in v2.0 (not user-facing).

16. Post-hoc intra/inter peptide energy analysis
------------------------------------------------
`gareus-energy-decompose` is an analysis-only post-processing command for saved
coordinates.  It is the intended route for later intrapeptide/interpeptide energy
analysis when live total-PE sampling was disabled during production:

    gareus-energy-decompose --run-dir run_cln025
    gareus-energy-decompose --run-dir run_cln025 --include-peptide-environment
    gareus-energy-decompose --run-dir run_cln025 --write-total-forcefield-energy
    gareus-energy-decompose --topology run/02_npt_equilibrated.pdb --trajectory 'run/replica_trajectories/*.dcd'
    gareus-energy-decompose -hh

Coordinate retention matters:

    keep --traj-format dcd or --traj-format xtc       frame-resolved time series
    keep final_pdbs/*.pdb                             final snapshots only
    avoid --traj-format none                          if later frame energies are required

The output table includes:

    intrapeptide_direct_nonbonded_kj_mol              within the same peptide group
    interpeptide_direct_nonbonded_kj_mol              between peptide groups/chains
    peptide_peptide_direct_nonbonded_kj_mol           intra + inter
    peptide_environment_direct_nonbonded_kj_mol       optional peptide-vs-solvent/ion
    total_forcefield_kj_mol                           optional exact whole-system scalar

Default grouping is `--group-mode chains`: each peptide chain is a peptide group.
`--group-mode all-peptide` treats all peptide atoms as one group, useful for a
single-chain intrapeptide total.  `--group-mode residues` treats each residue as
a group for residue-residue diagnostics.

For each selected pair, the direct nonbonded diagnostic is:

    E_ij = k_e q_i q_j / r_ij + 4 epsilon_ij [(sigma_ij / r_ij)^12 - (sigma_ij / r_ij)^6]

Exception pairs use OpenMM exception charge product, sigma, and epsilon.
Non-exception pairs use the particle charges and Lorentz-Berthelot mixing.
The diagnostic intentionally does not claim an exact PME reciprocal-space pair
decomposition.  Use `--write-total-forcefield-energy` when an exact whole-system
unbiased OpenMM potential-energy scalar is needed alongside the pair diagnostics.

17. Inter-epoch tICA secondary-CV update
-----------------------------------------
After each completed adaptive epoch the package can optionally refit the slowest
backbone-dihedral mode via time-lagged Independent Component Analysis (tICA) and
update the secondary_center for every active umbrella window.  This is an
inter-epoch operation: it does not change CV2 type or the bias force; it
recalibrates where each window's harmonic potential is centered so that the
umbrella biases track the true data-density of the tIC1 coordinate.

Algorithm
~~~~~~~~~
1. During production, backbone (phi, psi) dihedral angles are recorded every
   tica_obs_interval samples into epoch_dir/tica_obs/dihedral_obs_*.npz.

2. At the end of each trigger epoch the package solves the generalized eigenvalue
   problem for the time-lagged covariance matrices C(tau) and C(0):

       C(tau) v = lambda C(0) v

   using scipy.linalg.eigh with a Cholesky-based numpy fallback.  The leading
   eigenvector v_1 (largest eigenvalue lambda_1) defines the tIC1 projection.

3. Sign continuity is enforced against the previous model:
   if dot(v_1_new, v_1_prev) < 0 then v_1_new = -v_1_new.

4. Per-window tIC1 medians are computed from the epoch's observations.

5. state.secondary_center is updated in the registry for each active window.

6. The next-epoch window CSV is re-written with the updated centers, so the new
   umbrella positions take effect at the start of the following epoch.

7. The tICA model is versioned (tica_cv_version).  The cross-epoch MBAR guard
   excludes any epoch whose version differs from the current model, preventing
   incoherent bias matrices when the coordinate has changed.

CV2-type agnosticism
~~~~~~~~~~~~~~~~~~~~
The center update is agnostic to the secondary CV type.  If CV2 = tica-linear,
the updated center is directly the per-window tIC1 median.  For other CV2 types
(rama-map, contacts, etc.) the tICA model is fitted for information/diagnostics
but secondary_center still holds a value in the original CV2 coordinate; the
update reflects where the data landed in CV2 space, not tIC1 space.

CV2 auto-switch (tica_switch_cv2)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Setting tica_switch_cv2: true enables a one-shot automatic CV2 type change:
after the first successful tICA refit, the secondary CV type is permanently
replaced with tica-linear for all remaining epochs and final production.

Bootstrap torsion PCA
~~~~~~~~~~~~~~~~~~~~~
`cv2: torsion-pca` fits a first-epoch linear backbone torsion CV from GENPEPT
seed conformers before production MD starts.  Features are per-residue
sin(phi), cos(phi), sin(psi), and cos(psi).  By default, each feature is
residualized against seed CV1 before PCA so CV2 is less redundant with the
nonlocal contact axis.  The state file is written to
<out>/tica/bootstrap_torsion_cv.json and the OpenMM force uses the same linear
torsion machinery as `tica-linear`.

Recommended two-stage setup:

    cvs:
      cv1: contacts
      cv2: torsion-pca

    bootstrap_torsion_cv:
      bootstrap_torsion_source: seeds
      bootstrap_torsion_residualize_against_cv1: true
      bootstrap_torsion_component: 1
      bootstrap_torsion_min_seed_count: 20

    tica:
      tica_obs_interval: 50
      tica_lag_frames: 50
      tica_update_after_epochs: [0]
      tica_switch_cv2: true

This allows a single YAML config to run a two-stage strategy:

  Epoch 0:   cv2 = rama-map  (broad discrete exploration; cheap, no model needed)
  tICA fit:  collect obs from epoch 0, fit tIC1, update window centers
  Epoch 1+:  cv2 = tica-linear  (data-driven slowest-mode coordinate)
  Final:     cv2 = tica-linear  (high-quality MBAR samples)

The switch is permanent once triggered.  On --ap-resume the switch is
restored from the epoch summary JSON so the resumed run continues with
tica-linear rather than reverting to the original cv2 setting.

k bounds are separately adjustable for the tica-linear regime:

  tica_linear_k_min: 5.0    # default 5.0 kcal/mol (softer than rama-map's 20)
  tica_linear_k_max: 50.0   # default 50.0 kcal/mol

tICA projects onto [-6, 6] (dimensionless).  The wider range relative to
rama-map's [-1, 1] requires softer springs; the adaptive overlap system will
scale further if needed.

Trigger modes
~~~~~~~~~~~~~
The refit fires when either (or both) of the following conditions is true AND
tica_obs_interval > 0:

  a) epoch is in tica_update_after_epochs (explicit list, 0-indexed)
  b) (epoch + 1) % tica_epochs_per_cycle == 0  (periodic cycle, >=1)

Both conditions act as an OR union; set one or both.  If neither is set the
refit is disabled even when observation collection is active.

MBAR guard and cycle-length choice
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Each refit bumps tica_cv_version.  The MBAR guard then excludes all prior-version
epochs from cross-epoch MBAR pooling.  With tica_epochs_per_cycle = 1 every epoch
is a new version, so the MBAR pool is always just the current epoch.

Recommended cycle lengths:

    tica_epochs_per_cycle: 1   only if each epoch is already sufficient for MBAR
    tica_epochs_per_cycle: 3   practical default; pools 3 epochs before refit
    tica_epochs_per_cycle: 5   more data per refit; slower adaptation

With tica_switch_cv2: epoch 0 (rama-map) and epoch 1+ (tica-linear) have
different tica_cv_version values by construction.  The MBAR pool for final
production only includes post-switch epochs, which is the correct behavior.

YAML configuration
~~~~~~~~~~~~~~~~~~
Minimal (center-update only, same CV2 throughout):

    tica:
      tica_obs_interval: 50        # >0 required to enable; typically 50-200
      tica_lag_frames: 50          # lag tau; increase for slow folding peptides
      tica_epochs_per_cycle: 3     # recommended >= 3 for MBAR pooling
      tica_min_eigenvalue: 0.0     # skip refits with poor slow-mode separation
      # tica_update_after_epochs: [2, 5, 8]   # alternative: explicit epoch list
      # tica_state_file: ""                   # managed automatically; omit

Two-stage auto-switch (epoch 0 rama-map → epoch 1+ tica-linear):

    cvs:
      cv1: contacts
      cv2: rama-map               # starting CV2; switched automatically after epoch 0

    adaptive_production:
      ap_epochs: 2                # epoch 0: rama-map, epoch 1: tica-linear
      ap_final_pool_fraction: 0.60

    tica:
      tica_obs_interval: 50
      tica_lag_frames: 50
      tica_update_after_epochs: [0]  # fit after epoch 0, then switch
      tica_switch_cv2: true          # permanently switch CV2 to tica-linear
      tica_linear_k_min: 5.0         # k bounds for the tICA-linear regime
      tica_linear_k_max: 50.0

The tica: section name is cosmetic.  _flatten_config_mapping descends any nested
dict, so tica_* keys may appear under any section heading.  tica_state_file is
written by the adaptive loop; starting from a previous model requires setting it
explicitly to the path of the saved TICAResult JSON.
"""


def _dedent(text: str) -> str:
    return textwrap.dedent(text).strip() + "\n"


def simple_help_text(prog: str = "gareus") -> str:
    """Return the concise user-facing help page."""
    text = _dedent(_OVERVIEW) + "\n" + _dedent(_SIMPLE_HELP)
    if prog and prog != "gareus":
        text = text.replace("gareus ", f"{prog} ", 1)
    return text


def heavy_help_text(parser: argparse.ArgumentParser) -> str:
    """Return the encyclopedia-style help page plus complete argparse output."""
    method_text = _dedent(_METHOD_ENCYCLOPEDIA)
    full_options = parser.format_help()
    return (
        method_text
        + "\n\nComplete option reference\n"
        + "=========================\n\n"
        + full_options
    )


class SimpleHelpAction(argparse.Action):
    """Argparse action that prints concise help and exits."""

    def __init__(self, option_strings: Iterable[str], dest: str = argparse.SUPPRESS, default=argparse.SUPPRESS, help: str | None = None):
        super().__init__(option_strings=list(option_strings), dest=dest, nargs=0, default=default, help=help)

    def __call__(self, parser, namespace, values, option_string=None):  # noqa: D401 - argparse signature
        parser._print_message(simple_help_text(parser.prog), sys.stdout)
        parser.exit(0)


class HeavyHelpAction(argparse.Action):
    """Argparse action that prints method encyclopedia help and exits."""

    def __init__(self, option_strings: Iterable[str], dest: str = argparse.SUPPRESS, default=argparse.SUPPRESS, help: str | None = None):
        super().__init__(option_strings=list(option_strings), dest=dest, nargs=0, default=default, help=help)

    def __call__(self, parser, namespace, values, option_string=None):  # noqa: D401 - argparse signature
        parser._print_message(heavy_help_text(parser), sys.stdout)
        parser.exit(0)
