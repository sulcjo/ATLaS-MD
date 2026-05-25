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

    --run-mode MODE                    cmd, hmr-cmd, gamd, or hmr-gamd. cmd/hmr-cmd need no gamd-openmm.
    --window-mode MODE                  manual, adaptive, or adaptive-feedback.
    --cv1 MODE                          Friendly primary CV: distance or contacts.
    --cv2 MODE                          Friendly secondary CV; non-none auto-enables 2D centers.
    --primary-cv MODE                   Expert primary CV: distance or nonlocal-contacts.
    --windows-a A ...                   Manual distance centers in Angstrom.
    --contact-centers C ...             Manual contact-fraction centers.
    --secondary-cv MODE                 Expert secondary CV: none, alpha, beta, alpha-coil-beta, rama-map, rama-regions, custom.
    --secondary-cv-centers C ...        Create/override a 2D grid by crossing primary and secondary centers.
    --seed-conformers-dir DIR           Use GENPEPT survivors for CV-aware window starts.
    --seed-selection-mode MODE          auto/active-cv, primary, or legacy distance seed scoring.

    --box-shape SHAPE                   Solvent box shape: dodecahedron default, cube legacy.
    --padding-nm NM                     Solvent padding around the peptide.

    --production-steps N                Production steps per replica; alias for --gamd-production-steps.
    --gamd-production-steps N           Historical production-step flag, still supported.
    --exchange-mode MODE                neighbor, random-pair, all-pair-sweep, or gibbs-walk.
    --exchange-interval N               Steps between exchange attempts.
    --traj-format FORMAT                dcd, xtc, or none.
    --no-sample-potential-energy        Skip live total-PE diagnostics; later decomposition still works from saved coordinates.
    --no-flush-every-log                Buffer CSV/JSONL between checkpoints for faster filesystem I/O.

    --platform NAME                     auto, CUDA, HIP, OpenCL, CPU, or Reference.
    --device-index LIST                 GPU device token(s), e.g. 0 or 0,1,2,3.
    --replica-device-mode MODE          auto, round-robin, single-context-split, or manual.
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
    samples.csv                         Scalar sample log.
    analysis_chunks/*.npz               Chunked bias/CV arrays for downstream analysis.
    exchanges.csv                       Exchange attempts and acceptance probabilities.
    exchange_tuning_report.md/json      Practical exchange diagnostics.
    final_run_report.md/json            Final validation and output summary.

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

Friendly CV shortcuts:

    --cv1 distance       Primary terminal-distance umbrella CV.
    --cv1 contacts       Primary smooth nonlocal-contact fraction CV.
    --cv2 rama-map       Enable explicit Ramachandran basin-map CV and automatically
                         cross primary windows with beta/PPII/right-alpha/left-alpha
                         centers [-1, -1/3, 1/3, 1].
    --cv2 rama-regions   Legacy scalar Ramachandran-region ladder with
                         centers [-1, -0.5, 0, 0.5, 1].
    --cv2 alpha-coil-beta
                         Enable secondary alpha-minus-beta CV and automatically
                         cross primary windows with [-0.8, 0, 0.8].

The expert flags --primary-cv, --secondary-cv, and --secondary-cv-centers remain
available.  Supplying --secondary-cv without a center list keeps the historical
single secondary restraint; supplying --cv2 non-none deliberately creates a 2D
primary x secondary window set unless centers are explicitly overridden.

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

Hydrogen mass repartitioning can be enabled directly or through the high-level mode:

    --hmr
    --run-mode hmr-cmd
    --run-mode hmr-gamd

With HMR enabled, the default repartitioned hydrogen mass is:

    3.024 amu

unless overridden by --hydrogen-mass-amu. The high-level HMR run modes also default the production timestep to 4 fs when neither the CLI nor the config file sets --timestep-fs explicitly.

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

    --primary-cv distance
    --cv-mode terminal-ca

Mathematically:

    x(q) = ||r_a(q) - r_b(q)||

The terminal N/C selector is available with:

    --cv-mode terminal-n-c

and explicit atom selectors are available with:

    --cv-atom1 1:CA --cv-atom2 -1:CA

The alternative primary CV is a smooth nonlocal-contact fraction:

    --primary-cv nonlocal-contacts

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
from the sequence and constructs adaptive umbrella centers.  Default distance
adaptive settings are:

    target spacing   = 2 A
    minimum windows  = 4
    maximum windows  = 32
    adaptive prescan = off by default

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

    sigma ~= spacing / overlap_sigma
    k     ~= RT / sigma^2

with default clamp:

    0.05 to 20 kcal/mol/A^2

0.7 Optional secondary CV and 2D umbrellas
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Default secondary CV mode is:

    --secondary-cv none

If enabled, the package adds a second harmonic umbrella term on a smooth
backbone torsion-content score.  Available modes include:

    alpha
    beta
    alpha-coil-beta
    rama-map
    rama-regions
    custom

Secondary CVs use smooth periodic phi/psi scores:

    g(theta; theta0, sigma) = exp[-(1 - cos(theta - theta0)) / sigma^2]

Default angular width:

    sigma = 35 degrees

If multiple secondary centers are supplied, the package creates a 2D umbrella
grid:

    primary centers x secondary-CV centers

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

    5,000 steps per window
    pull k = 5 kcal/mol/A^2 for distance CV
    pull friction = 10 ps^-1
    pull timestep = min(production timestep, 2 fs)
    minimization at pulled centers = 100 iterations

For contact CVs, pulling is more conservative:

    max contact pull timestep = 1 fs
    minimum friction          = 20 ps^-1
    contact pull k is capped unless explicitly overridden

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

    sigma0p           = 6 kcal/mol
    sigma0d           = 6 kcal/mol
    cmd prep          = 5,000 steps
    cmd               = 50,000 steps
    equil prep        = 5,000 steps
    equil             = 50,000 steps
    production        = 500,000 steps
    averaging window  = 5,000 steps

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

    rounds                                = 3
    pilot fraction                        = 0.05 of final production
    validation fraction                   = 0.10 for final pre-production validation
    target neighboring-window overlap     = 0.25
    bootstrap samples                     = 50
    aggressiveness                        = balanced

The final production lives under:

    final_production/

Pilot folders are diagnostics, not final PMF inputs.  Adaptive feedback can add
midpoint windows, shift centers, prune over-resolved regions, create sparse 2D
local patches, and adapt secondary-CV centers when enabled.

0.14 Outputs for analysis
~~~~~~~~~~~~~~~~~~~~~~~~~
Production outputs include:

    samples.csv
    distances.csv / distances.jsonl
    exchanges.csv
    umbrella_windows.csv
    umbrella_explicit_windows.csv
    umbrella_pymbar_metadata.json
    analysis_chunks/*.npz
    analysis_arrays.npz
    exchange_tuning_report.md/json
    final_run_report.md/json
    replica_trajectories/*.dcd or *.xtc
    final_pdbs/*.pdb

The analysis chunks store CVs and the full umbrella bias matrix needed for
MBAR/PMF-style downstream analysis.

0.15 Potential-energy handling and later decomposition
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
    --primary-cv nonlocal-contacts
    --contact-scheme atom-pairs | residue-balanced | ca-pairs
    --contact-atom-selection heavy | ca | backbone-heavy | sidechain-heavy | all
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

    sigma ~= spacing / overlap_sigma
    k     ~= RT / sigma^2

The result is clamped by the corresponding min/max k settings.  This is used by
adaptive distance windows, contact windows, and secondary-CV spacing/adaptive k
modes.  The goal is not a sacred theorem carved onto a GPU, but a practical
initial overlap heuristic.

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
    rama-regions          legacy soft Ramachandran basin coordinate
    custom                target supplied by --secondary-cv-phi0-deg/psi0-deg

For `alpha-coil-beta`, alpha-like states are positive, beta-like states are
negative, and coil/disordered states tend toward zero.  For `rama-map`, the code
uses named smooth phi/psi basins: beta/extended (-135,+135), PPII/coil
(-75,+145), right-alpha (-60,-45), and left-alpha (+60,+40).  It remains a
one-dimensional secondary map coordinate crossed with CV1, not a dense phi/psi
grid.  Legacy `rama-regions` keeps the older beta/PPII/turn/alpha/left-alpha
scalar ladder.

7. 2D and sparse windows
------------------------
With multiple `--secondary-cv-centers`, GAREUS normally creates a rectangular
cross-product of primary centers x secondary centers.  `--windows-2d-csv` can
instead load an explicit per-window table, which supports sparse local patches
from adaptive feedback.

`--genpept-prior-enabled` builds a round-zero explicit 2D table from GENPEPT
output structures before the first adaptive-feedback pilot.  GAREUS rescoring
uses the active CV1/CV2 definitions, writes `genpept_prior/genpept_prior_windows_2d.csv`,
and feeds that table through the same sparse explicit-2D machinery.  This is a
sampling prior only: it can place initial windows and seed suggestions, but it is
not PMF/MBAR evidence.

Neighbor exchange for explicit/sparse 2D tables uses a geometry graph.  The graph
combines row/column-style relationships with k-nearest/radius edges controlled by:

    --explicit-2d-exchange-neighbor-k
    --explicit-2d-exchange-radius
    --explicit-2d-exchange-slots

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
    --seed-secondary-weight FLOAT
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
    --sigma0p-kcal-mol
    --sigma0d-kcal-mol
    --gamd-cmd-prep-steps / --gamd-cmd-steps
    --gamd-equil-prep-steps / --gamd-equil-steps
    --gamd-production-steps

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
    --adaptive-feedback-rounds
    --adaptive-feedback-pilot-fraction
    --adaptive-feedback-target-overlap
    --adaptive-window-aggressiveness
    --adaptive-min-total-windows / --adaptive-max-total-windows
    --contact-adaptive-min-total-windows / --contact-adaptive-max-total-windows

Final production should be analyzed from `final_production/` when adaptive
feedback is used.  Pilot folders are diagnostics, not final PMF inputs.

12. Output and downstream analysis
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

13. Analysis-driven CV suggestion report
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
compared with contacts or Ramachandran-region CVs.  It is deliberately heuristic:
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
      --gamd-cmd-prep-steps 1 --gamd-cmd-steps 1 \
      --gamd-equil-prep-steps 1 --gamd-equil-steps 1 \
      --gamd-production-steps 4 --exchange-interval 2 \
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
    --no-flush-every-log on slow/network filesystems

`--scratchdir` mirrors results back to `--out` at checkpoints, reducing pressure
on network filesystems during high-frequency trajectory/NPZ/CSV writes.

`--no-sample-potential-energy` skips the diagnostic potential-energy read in
sample/distance rows.  It does not change integration, forces, GaMD boost state,
umbrella biases, exchange probabilities, trajectories, or checkpoints; it only
leaves total potential-energy columns as NaN/blank.  Later peptide energy
decomposition still works if coordinate trajectories or final PDB snapshots are
kept.  Do not combine it with `--traj-format none` if you expect frame-by-frame
post-hoc energies later.

`--no-flush-every-log` lets `--csv-flush-rows`/`--jsonl-flush-rows` batch scalar
I/O between checkpoints and clean shutdown.  This changes output durability during
a hard crash, not the simulation trajectory or statistics.

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
