"""Command-line parser and top-level workflow for GAREUS."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Iterable, Optional

from .adaptive_feedback import run_adaptive_feedback_auto_loop
from .checkpoints import production_checkpoint_available, load_existing_openmm_setup_for_resume
from .colors import configure_color
from .config import (
    _argv_as_list,
    _apply_config_defaults_to_parser,
    _write_config_template,
    _write_reproducibility_files,
)
from .cv import contact_scheme, primary_cv_is_contacts, primary_cv_mode, secondary_cv_mode
from .forces import self_test_nonlocal_contact_force
from .io import write_json
from .lifecycle import _graceful_shutdown
from .production import run_gareus, sync_scratch_to_main
from .progress import GuiProgressSink
from .provenance import initialize_run_manifest, finalize_run_manifest
from .system_setup import minimize_and_npt_equilibrate, validate_sequence
from .helptext import SimpleHelpAction, HeavyHelpAction

__all__ = ["parse_args", "main"]


# ---------------------------------------------------------------------------
# parse_args helpers — each adds one logical group of CLI flags to a parser.
# None of these functions change argument names, defaults, or behavior;
# they are pure structural decompositions of the original monolithic function.
# ---------------------------------------------------------------------------

def _add_core_args(p: argparse.ArgumentParser) -> None:
    """Add help, config, I/O, sequence, and seed arguments."""
    p.add_argument("-h", "--help", action=SimpleHelpAction, help="Show concise practical help and exit.")
    p.add_argument("-hh", "--help-heavy", action=HeavyHelpAction, help="Show method encyclopedia, equations, design notes, and complete option reference; then exit.")
    p.add_argument("--config", default=None, help="YAML/JSON config file. Nested groups are allowed; leaf keys use argparse dest names. CLI flags override config values.")
    p.add_argument("--write-config-template", nargs="?", const="chignolin_adaptive_feedback.yaml", default=None, help="Write a ready-to-edit YAML config template and exit. Optional path argument supported.")
    p.add_argument("--write-effective-config", action="store_true", help="Force writing effective_config.yaml/json immediately at run start. These files are also written by default for every run.")
    p.add_argument("--seq", required=True, help="One-letter peptide sequence.")
    p.add_argument("--out", default="gareus_peptide_out", help="Output directory.")
    p.add_argument("--seed", type=int, default=2026)


def _add_system_args(p: argparse.ArgumentParser) -> None:
    """Add peptide system / OpenMM simulation setup arguments."""
    p.add_argument("--initial-phi", type=float, default=-60.0, help="Initial PeptideBuilder internal phi angle for residues where applicable.")
    p.add_argument("--initial-psi", type=float, default=-45.0, help="Initial PeptideBuilder previous-residue psi angle for residues where applicable.")
    p.add_argument("--ph", type=float, default=7.0)
    p.add_argument("--water-model", choices=["tip3p", "tip3pfb", "spce", "tip4pew"], default="tip3p")
    p.add_argument("--box-shape", choices=["dodecahedron", "cube", "octahedron"], default="dodecahedron", help="Periodic solvent box shape passed to OpenMM Modeller.addSolvent. dodecahedron reduces solvent count for compact peptides; cube is the legacy rectangular default; octahedron is also supported by recent OpenMM versions.")
    p.add_argument("--padding-nm", type=float, default=1.0)
    p.add_argument("--ionic-strength-molar", type=float, default=0.15)
    p.add_argument("--temperature-k", type=float, default=300.0)
    p.add_argument("--pressure-bar", type=float, default=1.0)
    p.add_argument("--barostat-frequency", type=int, default=25)
    p.add_argument("--production-ensemble", choices=["npt", "nvt"], default="npt", help="Production ensemble. Default npt uses OpenMM MonteCarloBarostat; nvt preserves the previous fixed-box production behavior.")
    p.add_argument("--production-barostat-frequency", type=int, default=0, help="MC barostat attempt frequency for production NPT. 0 reuses --barostat-frequency.")
    p.add_argument("--run-mode", choices=["cmd", "hmr-cmd", "gamd", "hmr-gamd"], default="gamd", help="High-level dynamics mode: cmd = conventional umbrella/REUS MD without gamd-openmm; hmr-cmd = conventional umbrella/REUS with HMR and 4 fs default; gamd = GaMD/REUS without automatic HMR; hmr-gamd = HMR + GaMD/REUS with 4 fs default. Existing expert flags remain available.")
    p.add_argument("--timestep-fs", type=float, default=2.0)
    p.add_argument("--friction-per-ps", type=float, default=1.0)
    p.add_argument("--nonbonded-cutoff-nm", type=float, default=0.8)
    p.add_argument("--ewald-error-tolerance", type=float, default=1e-4)
    p.add_argument("--minimize-iterations", type=int, default=20000)
    p.add_argument("--npt-steps", type=int, default=100000)

    # Safer staged equilibration. Defaults are intentionally conservative:
    # first prove the molecule can survive, then let production be fast.
    p.add_argument("--nvt-warmup-steps", type=int, default=20000)
    p.add_argument("--nvt-warmup-timestep-fs", type=float, default=0.25)
    p.add_argument("--nvt-start-temperature-k", type=float, default=50.0)
    p.add_argument("--npt-ramp-steps", type=int, default=50000)
    p.add_argument("--npt-ramp-timestep-fs", type=float, default=0.5)
    p.add_argument("--npt-final-timestep-fs", type=float, default=0.0, help="0 = auto min(--timestep-fs, 2 fs) during NPT equilibration.")
    p.add_argument("--equil-restraint-k-kcal-mol-a2", type=float, default=2.0)
    p.add_argument("--equil-friction-per-ps", type=float, default=10.0)
    p.add_argument("--equil-safe-chunk-steps", type=int, default=100)

    p.add_argument("--hmr", action="store_true", help="Enable hydrogen mass repartitioning.")
    p.add_argument("--hydrogen-mass-amu", type=float, default=0.0, help="Hydrogen mass for HMR; 0 means use 3.024 when --hmr is set.")


def _add_cv_args(p: argparse.ArgumentParser) -> None:
    """Add collective-variable (CV) arguments: primary and secondary CVs."""
    p.add_argument("--cv1", choices=["distance", "contacts", "nonlocal-contacts"], default=None, help="Friendly primary-CV selector. --cv1 distance maps to --primary-cv distance; --cv1 contacts maps to --primary-cv nonlocal-contacts.")
    p.add_argument("--cv2", choices=["none", "alpha", "beta", "alpha-coil-beta", "acb", "rama-map", "rama-regions", "rama", "ramachandran", "ramachandran-regions", "custom"], default=None, help="Friendly secondary-CV selector. Any non-none value enables a real 2D workflow; if no --secondary-cv-centers are supplied, sensible default centers are inserted automatically.")
    p.add_argument("--cv-mode", choices=["terminal-ca", "terminal-n-c"], default="terminal-ca")
    p.add_argument("--cv-atom1", default=None, help="Explicit selector, e.g. 1:CA or absolute atom index.")
    p.add_argument("--cv-atom2", default=None, help="Explicit selector, e.g. -1:CA or absolute atom index.")
    p.add_argument("--primary-cv", choices=["distance", "nonlocal-contacts"], default="distance", help="Primary umbrella CV. distance preserves historical terminal-distance behavior; nonlocal-contacts uses a reference-free smooth nonlocal contact fraction.")
    p.add_argument("--contact-scheme", choices=["atom-pairs", "residue-balanced", "ca-pairs"], default="atom-pairs", help="How nonlocal contacts are counted. atom-pairs preserves v6 behavior; residue-balanced averages atom contacts within each residue pair so every residue pair contributes equally; ca-pairs uses only CA atom pairs.")
    p.add_argument("--contact-atom-selection", choices=["heavy", "ca", "backbone-heavy", "sidechain-heavy", "all"], default="heavy", help="Atoms used to build the nonlocal-contact primary CV. Ignored by --contact-scheme ca-pairs.")
    p.add_argument("--contact-min-sequence-separation", type=int, default=4, help="Minimum peptide residue-index separation for atom pairs in the nonlocal-contact CV.")
    p.add_argument("--contact-r0-a", type=float, default=4.5, help="Smooth contact switching midpoint in Angstrom for --primary-cv nonlocal-contacts.")
    p.add_argument("--contact-beta-a-inv", type=float, default=6.0, help="Smooth contact switching steepness in inverse Angstrom for --primary-cv nonlocal-contacts.")
    p.add_argument("--contact-normalize", action=argparse.BooleanOptionalAction, default=True, help="Normalize nonlocal-contact CV by the number of selected pairs so the CV is approximately 0..1.")
    p.add_argument("--contact-centers", nargs="+", type=float, default=None, help="Manual primary-CV centers for --primary-cv nonlocal-contacts. Normalized values should usually be in [0, 1].")
    p.add_argument("--contact-k-kcal", nargs="*", type=float, default=None, help="Manual contact-CV harmonic force constants in kcal/mol/CV^2. One value is broadcast to all contact centers; default 25.")
    p.add_argument("--contact-adaptive-min", type=float, default=0.0, help="Adaptive lower bound for normalized nonlocal-contact primary CV.")
    p.add_argument("--contact-adaptive-max", type=float, default=0.80, help="Adaptive upper bound for normalized nonlocal-contact primary CV. Values near 1 can be unrealistically collapsed for heavy-atom contact fractions.")
    p.add_argument("--contact-adaptive-target-spacing", type=float, default=0.15, help="Initial adaptive spacing for contact-CV windows in dimensionless CV units.")
    p.add_argument("--contact-adaptive-min-windows", type=int, default=4, help="Legacy contact-axis minimum window count. In 2D contact x secondary workflows prefer --contact-adaptive-min-total-windows/replicas.")
    p.add_argument("--contact-adaptive-max-windows", type=int, default=12, help="Legacy contact-axis maximum window count. In 2D contact x secondary workflows prefer --contact-adaptive-max-total-windows/replicas.")
    p.add_argument("--contact-adaptive-min-total-windows", "--contact-adaptive-min-total-replicas", dest="contact_adaptive_min_total_windows", type=int, default=0, help="Minimum total contact-primary expanded windows/replicas after crossing with secondary-CV centers. Overrides legacy axis min when >0.")
    p.add_argument("--contact-adaptive-max-total-windows", "--contact-adaptive-max-total-replicas", dest="contact_adaptive_max_total_windows", type=int, default=0, help="Maximum total contact-primary expanded windows/replicas after crossing with secondary-CV centers. Overrides legacy axis max when >0.")
    p.add_argument("--contact-adaptive-k-mode", choices=["spacing", "fixed", "constant"], default="spacing", help="Contact-CV adaptive force-constant assignment. spacing uses local contact-center spacing; fixed/constant repeats --contact-adaptive-default-k-kcal.")
    p.add_argument("--contact-adaptive-default-k-kcal", type=float, default=25.0, help="Fixed/default contact-CV k in kcal/mol/CV^2 when adaptive k is disabled or only one center is present.")
    p.add_argument("--contact-adaptive-min-k-kcal", type=float, default=5.0, help="Lower clamp for adaptive contact-CV k in kcal/mol/CV^2.")
    p.add_argument("--contact-adaptive-max-k-kcal", type=float, default=120.0, help="Upper clamp for adaptive contact-CV k in kcal/mol/CV^2.")
    p.add_argument("--contact-adaptive-overlap-sigma", type=float, default=1.25, help="Overlap-sigma factor for spacing-derived contact-CV k proposals.")
    p.add_argument("--contact-adaptive-min-sigma", type=float, default=0.02, help="Minimum contact-CV harmonic sigma used when deriving adaptive k.")
    p.add_argument("--contact-adaptive-k-scale", type=float, default=1.0, help="Scale factor applied to spacing-derived contact-CV k proposals.")
    p.add_argument("--contact-adaptive-hit-radius", type=float, default=0.08, help="Contact-CV hit radius for adaptive-feedback center visitation diagnostics.")
    p.add_argument("--contact-adaptive-tight-hit-radius", type=float, default=0.04, help="Tight contact-CV hit radius for adaptive-feedback diagnostics.")
    p.add_argument("--contact-adaptive-max-center-shift", type=float, default=0.08, help="Maximum contact-CV center shift per adaptive-feedback round.")
    p.add_argument("--contact-adaptive-min-new-spacing", type=float, default=0.04, help="Minimum separation between contact-CV centers proposed by adaptive-feedback.")
    p.add_argument("--contact-adaptive-coverage-gap-width", type=float, default=0.10, help="Minimum contact-CV interval width for adaptive-feedback coverage-gap additions.")
    p.add_argument("--contact-autocalibrate-windows", action=argparse.BooleanOptionalAction, default=True, help="Before adaptive/adaptive-feedback contact runs, run a short unbiased contact-CV prescan and shrink the initial window range to a reachable first-round interval.")
    p.add_argument("--contact-autocalibration-steps", type=int, default=None, help="Steps for contact-CV autocalibration prescan; 0 disables the prescan even when --contact-autocalibrate-windows is true. When unset, falls back to --adaptive-prescan-steps if > 0, else 5000.")
    p.add_argument("--contact-autocalibration-timestep-fs", type=float, default=None, help="Timestep for contact-CV autocalibration prescan. When unset, falls back to --adaptive-prescan-timestep-fs, else 1.0 fs.")
    p.add_argument("--contact-autocalibration-sample-interval", type=int, default=100, help="Steps between contact-CV samples during autocalibration.")
    p.add_argument("--contact-autocalibration-safe-chunk-steps", type=int, default=50, help="Maximum MD chunk size during contact-CV autocalibration.")
    p.add_argument("--contact-autocalibration-friction-per-ps", type=float, default=20.0, help="Langevin friction for contact-CV autocalibration prescan.")
    p.add_argument("--contact-autocalibration-percentile", type=float, default=99.0, help="Observed percentile used when estimating the first-round contact adaptive upper bound.")
    p.add_argument("--contact-autocalibration-margin", type=float, default=0.02, help="Extra contact-CV margin added above observed high-percentile values.")
    p.add_argument("--contact-autocalibration-expand-factor", type=float, default=4.0, help="Multiplier for observed contact-CV spread when extrapolating first-round adaptive range.")
    p.add_argument("--contact-autocalibration-max-multiple", type=float, default=4.0, help="Maximum multiple of observed contact-CV max used as a first-round adaptive upper-bound candidate.")
    p.add_argument("--contact-autocalibration-min-span", type=float, default=0.10, help="Minimum first-round contact adaptive span after autocalibration.")
    p.add_argument("--contact-autocalibration-required", action=argparse.BooleanOptionalAction, default=False, help="If true, abort when contact-CV autocalibration fails instead of falling back to configured bounds.")
    p.add_argument("--contact-pair-warning-threshold", type=int, default=5000, help="Warn when the nonlocal-contact CV contains more atom pairs than this threshold.")
    p.add_argument("--self-test-primary-cv-force", action="store_true", help="Construct and evaluate the selected primary-CV OpenMM force in a tiny test Context, then exit. Currently most useful for validating --primary-cv nonlocal-contacts on a given OpenMM installation.")
    p.add_argument("--secondary-cv", choices=["none", "alpha", "beta", "alpha-coil-beta", "acb", "rama-map", "rama-regions", "rama", "ramachandran", "ramachandran-regions", "custom"], default="none", help="Optional second umbrella CV based on smooth backbone phi/psi secondary-structure content. alpha and beta are 0..1 content scores; alpha-coil-beta/acb is a signed alpha-minus-beta coordinate; rama-map is an explicit beta/PPII/right-alpha/left-alpha Ramachandran basin map; rama-regions is the older signed soft region coordinate; custom uses --secondary-cv-phi0-deg/--secondary-cv-psi0-deg.")
    p.add_argument("--secondary-cv-center", type=float, default=None, help="Secondary-CV target for all windows when --secondary-cv-centers is not provided. For alpha/beta/custom this is roughly 0..1; for alpha-coil-beta, rama-map, and rama-regions it is -1..1 and defaults internally to 0.0 if no center list is supplied.")
    p.add_argument("--secondary-cv-centers", nargs="*", type=float, default=None, help="Secondary-CV targets. One value restrains all distance windows to that target; multiple values create a 2D grid by crossing every distance window with every secondary-CV target. For alpha-coil-beta use e.g. -0.8 0.0 0.8; for rama-map use the default basin centers -1 -0.333333 0.333333 1; for rama-regions use e.g. -1 -0.5 0 0.5 1.")
    p.add_argument("--secondary-cv-k-kcal", type=float, default=50.0, help="Fixed secondary-CV harmonic force constant in kcal/mol/CV^2, or the one-center fallback when --secondary-cv-k-mode spacing is requested. This is not per-Angstrom; the CV is dimensionless.")
    p.add_argument("--secondary-cv-k-mode", choices=["fixed", "constant", "spacing", "adaptive"], default="fixed", help="How to assign secondary-CV force constants. fixed/constant repeats --secondary-cv-k-kcal; spacing/adaptive mirrors the primary CV adaptive-k formula in dimensionless CV units using secondary-CV center spacing.")
    p.add_argument("--secondary-cv-adaptive-overlap-sigma", type=float, default=0.0, help="Spacing/adaptive secondary-CV k parameter: sigma_CV ~= spacing/this value. 0 = reuse --adaptive-overlap-sigma. Larger values make stronger secondary-CV k.")
    p.add_argument("--secondary-cv-adaptive-min-k-kcal", type=float, default=0.0, help="Minimum spacing/adaptive secondary-CV k in kcal/mol/CV^2.")
    p.add_argument("--secondary-cv-adaptive-max-k-kcal", type=float, default=500.0, help="Maximum spacing/adaptive secondary-CV k in kcal/mol/CV^2.")
    p.add_argument("--secondary-cv-adaptive-min-sigma", type=float, default=0.02, help="Lower bound on spacing-derived secondary-CV sigma before computing kBT/sigma^2; prevents extreme k for nearly duplicate centers.")
    p.add_argument("--secondary-cv-adaptive-k-scale", type=float, default=1.0, help="Extra multiplier for spacing/adaptive secondary-CV k values after kBT/sigma^2.")
    p.add_argument("--secondary-cv-sigma-deg", type=float, default=35.0, help="Angular width in degrees for the smooth phi/psi content score.")
    p.add_argument("--secondary-cv-phi0-deg", type=float, default=-60.0, help="Custom secondary-CV phi target in degrees, used with --secondary-cv custom.")
    p.add_argument("--secondary-cv-psi0-deg", type=float, default=-45.0, help="Custom secondary-CV psi target in degrees, used with --secondary-cv custom.")
    p.add_argument("--secondary-cv-force-group", type=int, default=29, help="OpenMM force group for the optional secondary-structure CV bias. Must be 0..31; default 29 avoids the primary umbrella group 31 and restraint group 30.")

    # Contact-CV frontier probing during adaptive-feedback.
    p.add_argument("--contact-frontier-enabled", action=argparse.BooleanOptionalAction, default=False, help="Enable contact-CV frontier probing during adaptive-feedback.")
    p.add_argument("--contact-frontier-percentile", type=float, default=99.0)
    p.add_argument("--contact-frontier-margin", type=float, default=0.02)
    p.add_argument("--contact-frontier-min-span", type=float, default=0.10)
    p.add_argument("--contact-frontier-probe-count", type=int, default=2)
    p.add_argument("--contact-frontier-probe-spacing", type=float, default=0.05)
    p.add_argument("--contact-frontier-max-probe-offset", type=float, default=0.15)
    p.add_argument("--contact-disable-gapfill-above-unvalidated-frontier", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--contact-frontier-confirm-rounds", type=int, default=2)
    p.add_argument("--contact-frontier-min-hit-fraction", type=float, default=0.02)
    p.add_argument("--contact-frontier-unreachable-deficit", type=float, default=0.08)
    p.add_argument("--contact-frontier-by-secondary-slice", action=argparse.BooleanOptionalAction, default=True)


def _add_window_args(p: argparse.ArgumentParser) -> None:
    """Add umbrella window layout, adaptive-feedback, and US pulling arguments."""
    p.add_argument("--us-starting-structure-mode", choices=["pull", "npt", "equilibrated", "same", "none"], default="pull", help="How to generate initial coordinates for each umbrella window. 'pull' performs a pre-production CV pulling walk and uses the resulting window conformers; 'npt' starts every replica from the same NPT state.")
    p.add_argument("--us-pull-steps-per-window", type=int, default=5000, help="Plain Langevin steps used to relax/pull into each umbrella starting structure before GaMD calibration.")
    p.add_argument("--us-pull-k-kcal-a2", type=float, default=5.0, help="Harmonic CV force constant for generating US starting conformers, in kcal/mol/A^2. This is only for pre-production pulling, not the production umbrella k values. In contact mode this legacy value is interpreted as kcal/mol/CV^2 unless contact-specific options override it.")
    p.add_argument("--contact-us-pull-k-kcal", type=float, default=None, help="Contact-mode-only pre-production starting-pull k in kcal/mol/CV^2. Overrides --us-pull-k-kcal-a2 for --primary-cv nonlocal-contacts.")
    p.add_argument("--contact-us-pull-max-k-kcal", type=float, default=20.0, help="Contact-mode safety cap for legacy --us-pull-k-kcal-a2 when --contact-us-pull-k-kcal is not set.")
    p.add_argument("--us-pull-timestep-fs", type=float, default=0.0, help="Pre-production US pulling timestep. 0 = min(--timestep-fs, 2 fs).")
    p.add_argument("--contact-us-pull-timestep-fs", type=float, default=1.0, help="Maximum timestep for contact-mode pre-production starting pulls. Contact pulls are capped to this value for stability.")
    p.add_argument("--contact-us-pull-ramp-stages", type=int, default=8, help="Number of r0/k ramp stages for contact-mode primary-CV starting pulls.")
    p.add_argument("--contact-us-pull-safe-chunk-steps", type=int, default=100, help="Maximum run_steps_safely chunk size for contact-mode starting pulls.")
    p.add_argument("--contact-us-pull-min-friction-per-ps", type=float, default=20.0, help="Minimum Langevin friction used during contact-mode starting pulls.")
    p.add_argument("--contact-us-pull-minimize-first-ramp", action=argparse.BooleanOptionalAction, default=True, help="Minimize briefly at the first contact primary-CV ramp stage.")
    p.add_argument("--us-pull-friction-per-ps", type=float, default=10.0, help="Langevin friction used only for pre-production US starting-structure pulling.")
    p.add_argument("--us-pull-minimize-iterations", type=int, default=100, help="Energy-minimization iterations at each pulled umbrella starting center.")
    p.add_argument("--us-2d-start-relax-mode", choices=["auto", "staged", "off", "single", "ramp"], default="auto", help="For secondary-CV windows, pre-relax starting structures with distance-only pulling followed by a gradual secondary-CV ramp. auto/staged enables this when CV2 is active; off/single preserves one-stage behavior.")
    p.add_argument("--us-2d-start-distance-fraction", type=float, default=0.50, help="Fraction of --us-pull-steps-per-window spent relaxing the distance CV before ramping the secondary CV in staged 2D starting-structure preparation.")
    p.add_argument("--us-2d-start-secondary-ramp-stages", type=int, default=3, help="Number of secondary-CV force-ramp stages after the distance-only stage for 2D starting structures.")
    p.add_argument("--us-2d-start-minimize-each-ramp", action="store_true", help="Minimize briefly at each secondary-CV ramp stage during 2D starting-structure preparation.")
    p.add_argument("--us-2d-start-secondary-warn-delta", type=float, default=0.35, help="Warn if the starting secondary CV is this far from its target after 2D starting-structure preparation.")
    p.add_argument("--us-2d-start-secondary-warn-bias-kcal", type=float, default=1.0, help="Warn if the starting secondary-CV bias exceeds this value in kcal/mol.")
    p.add_argument("--us-2d-start-secondary-bad-bias-kcal", type=float, default=5.0, help="Mark a starting structure bad if the starting secondary-CV bias exceeds this value in kcal/mol.")
    p.add_argument("--us-2d-start-secondary-k-pull-scale", type=float, default=1.0, help="Multiply the secondary-CV force constant by this factor during the 2D starting-structure pull ramp (production k is restored before saving). Values >1 push harder in fewer steps; try 5.0 with --us-2d-start-distance-fraction 0.05 to keep wall time near the 1D baseline.")
    p.add_argument("--window-mode", choices=["adaptive", "manual", "adaptive-feedback"], default="adaptive", help="manual: use --windows-a; adaptive: choose initial windows; adaptive-feedback: run short automatic feedback round(s), then a final full production run with the proposed fixed windows. In 2D secondary-CV runs, sparse local patch candidates are consumed automatically for final production when present.")
    p.add_argument("--adaptive-feedback-rounds", type=int, default=3, help="For --window-mode adaptive-feedback: number of short pilot refinement rounds before the final full production run. Default 3: usually enough for one broad diagnosis, one correction, and one validation pass; early convergence can skip remaining rounds.")
    p.add_argument("--adaptive-feedback-pilot-fraction", type=float, default=0.05, help="For --window-mode adaptive-feedback: pilot GaMD production fraction per refinement round relative to --gamd-production-steps. Default 0.05 = 1/20 of final production.")
    p.add_argument("--adaptive-feedback-validation-fraction", type=float, default=0.10, help="For --window-mode adaptive-feedback with more than one round: length of the last pre-production validation pilot as a fraction of --gamd-production-steps. The historical default is 0.10; set 0.05 to make it the same length as ordinary pilots, or use --adaptive-feedback-validation-steps 0 to disable the longer validation pass.")
    p.add_argument("--adaptive-feedback-validation-steps", type=int, default=-1, help="Exact step count for the last pre-production adaptive validation round. -1 uses --adaptive-feedback-validation-fraction, 0 disables the longer validation pass and uses ordinary pilot length, >0 uses that exact number of steps.")
    p.add_argument("--adaptive-feedback-target-overlap", type=float, default=0.25, help="Target neighboring-window CV histogram overlap for adaptive-feedback proposals; pairs below this are refined, and clearly over-resolved high-overlap/high-exchange regions can be reduced. Exchange/replica-reduction target is fixed at 30% acceptance to avoid extra flags.")
    p.add_argument("--adaptive-feedback-bootstrap-samples", type=int, default=50, help="Bootstrap resamples for adaptive-feedback overlap confidence intervals used in conservative replica pruning.")
    p.add_argument("--adaptive-window-aggressiveness", choices=["conservative", "balanced", "aggressive", "very-aggressive"], default="balanced", help="Bias automatic/adaptive-feedback window selection. balanced preserves old behavior; aggressive/very-aggressive start with wider spacing, add fewer windows, prune more readily, and tolerate lower bypass overlap to minimize replica count.")
    p.add_argument("--adaptive-window-count-weight", type=float, default=-1.0, help="Optional override for the adaptive-feedback score penalty per window. Negative = use the value implied by --adaptive-window-aggressiveness.")
    p.add_argument("--adaptive-secondary-cv", choices=["auto", "fixed"], default="auto", help="For adaptive-feedback with --secondary-cv: auto adapts the secondary-CV center ladder as a second dimension; fixed keeps the user-provided secondary center(s) unchanged.")
    p.add_argument("--windows-a", nargs="+", type=float, default=[5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 21], help="Manual umbrella centers in Angstrom.")
    p.add_argument("--window-k-kcal-a2", nargs="*", type=float, default=None, help="Per-window force constants in kcal/mol/A^2.")
    p.add_argument("--windows-2d-csv", default=None, help="Explicit per-window 2D umbrella table. Compatible with adaptive_feedback_explicit_window_candidates.csv. Generic columns primary_cv_center/primary_cv_k_kcal plus secondary_cv_center/secondary_cv_k_kcal_mol are accepted; legacy distance_center_A/distance_k_kcal_mol_A2 remain supported. Bypasses rectangular cross-product expansion.")
    p.add_argument("--sparse-2d-patches-enabled", action=argparse.BooleanOptionalAction, default=True, help="Enable generic sparse local 2D midpoint patches during adaptive-feedback. Works for any primary CV accepted by the primary-CV machinery; contact CVs get contact-specific units/bounds/K defaults.")
    p.add_argument("--contact-sparse-2d-patches-enabled", action=argparse.BooleanOptionalAction, default=True, help="Contact-CV specific gate for sparse local 2D patches. This is checked in addition to --sparse-2d-patches-enabled when the primary CV is nonlocal-contacts.")
    p.add_argument("--explicit-2d-window-schema", choices=["auto", "generic", "distance"], default="auto", help="Accepted schema for explicit 2D window CSVs. auto/generic accept primary_cv_* contact/distance-neutral fields and legacy distance_* aliases; distance documents the old alias names.")
    p.add_argument("--explicit-2d-exchange-neighbor-k", type=int, default=2, help="For --windows-2d-csv with --exchange-mode neighbor: add up to this many normalized k-nearest geometry edges per window in addition to row/column edges.")
    p.add_argument("--explicit-2d-exchange-radius", type=float, default=1.65, help="Normalized distance cutoff for extra k-nearest explicit-2D neighbor edges.")
    p.add_argument("--explicit-2d-exchange-slots", type=int, default=4, help="Round-robin slots for explicit-2D neighbor graph exchange scheduling; higher values reduce simultaneous edge attempts per interval.")
    p.add_argument("--default-window-k-kcal-a2", type=float, default=1.0)
    p.add_argument("--n-windows", type=int, default=0, help="Exact adaptive window count; 0 = choose from target spacing.")
    p.add_argument("--adaptive-target-spacing-a", type=float, default=2)
    p.add_argument("--adaptive-min-windows", type=int, default=4, help="Legacy primary-axis minimum window count for adaptive distance windows. In 2D workflows prefer --adaptive-min-total-windows/replicas.")
    p.add_argument("--adaptive-max-windows", type=int, default=32, help="Legacy primary-axis maximum window count for adaptive distance windows. In 2D workflows prefer --adaptive-max-total-windows/replicas.")
    p.add_argument("--adaptive-min-total-windows", "--adaptive-min-total-replicas", dest="adaptive_min_total_windows", type=int, default=0, help="Minimum total expanded windows/replicas after crossing primary and secondary CV centers. 0 disables.")
    p.add_argument("--adaptive-max-total-windows", "--adaptive-max-total-replicas", dest="adaptive_max_total_windows", type=int, default=0, help="Maximum total expanded windows/replicas after crossing primary and secondary CV centers. 0 disables.")
    p.add_argument("--adaptive-2d-max-local-patches", type=int, default=12, help="Maximum sparse local 2D midpoint patch windows to add per adaptive-feedback proposal.")
    p.add_argument("--adaptive-2d-min-edge-samples", type=int, default=5, help="Minimum per-window samples before a 2D local edge diagnostic is trusted.")
    p.add_argument("--adaptive-window-min-a", type=float, default=0.0, help="Override adaptive lower bound in A; <=0 means automatic.")
    p.add_argument("--adaptive-window-max-a", type=float, default=0.0, help="Override adaptive upper bound in A; <=0 means automatic.")
    p.add_argument("--adaptive-extension-fraction", type=float, default=0.95)
    p.add_argument("--adaptive-compact-floor-a", type=float, default=3.5)
    p.add_argument("--adaptive-k-mode", choices=["spacing", "constant"], default="spacing")
    p.add_argument("--adaptive-overlap-sigma", type=float, default=1.25)
    p.add_argument("--adaptive-min-k-kcal-a2", type=float, default=0.05)
    p.add_argument("--adaptive-max-k-kcal-a2", type=float, default=20.0)
    p.add_argument("--adaptive-prescan-steps", type=int, default=0)
    p.add_argument("--adaptive-prescan-timestep-fs", type=float, default=1.0)
    p.add_argument("--adaptive-prescan-margin-a", type=float, default=1.0)
    p.add_argument("--umbrella-force-group", type=int, default=31)

    # Stateful adaptive-feedback region classification and memory.
    p.add_argument("--adaptive-feedback-region-state-enabled", action=argparse.BooleanOptionalAction, default=False, help="Enable stateful adaptive-feedback region classification and memory.")
    p.add_argument("--adaptive-feedback-region-memory-decay", type=float, default=0.5)
    p.add_argument("--adaptive-feedback-min-effective-samples", type=int, default=50)
    p.add_argument("--adaptive-feedback-probe-unvalidated-regions", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--adaptive-feedback-prune-overscanned-regions", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--adaptive-feedback-extend-undersampled-regions", action=argparse.BooleanOptionalAction, default=True)

    # Explicit 2D window CSV column-name overrides.
    p.add_argument("--explicit-2d-primary-cv-mode-column", default="primary_cv_mode", help="Column name for primary-CV mode in explicit 2D window CSV.")
    p.add_argument("--explicit-2d-primary-center-column", default="primary_cv_center")
    p.add_argument("--explicit-2d-primary-k-column", default="primary_cv_k_kcal")
    p.add_argument("--explicit-2d-secondary-cv-mode-column", default="secondary_cv_mode")
    p.add_argument("--explicit-2d-secondary-center-column", default="secondary_cv_center")
    p.add_argument("--explicit-2d-secondary-k-column", default="secondary_cv_k_kcal_mol")


def _add_seeding_args(p: argparse.ArgumentParser) -> None:
    """Add GENPEPT seeding / starting-structure arguments."""
    p.add_argument(
        "--seed-conformers-dir", type=Path, default=None,
        help=(
            "Path to a GENPEPT output directory containing final_survivor_seeds.csv. "
            "When set, each umbrella window's starting structure is seeded from the "
            "nearest GENPEPT conformer in active CV space before the standard CV-pull step."
        ),
    )
    p.add_argument("--seed-selection-mode", choices=["auto", "active-cv", "primary", "distance"], default="auto", help="How GENPEPT survivors are scored against umbrella windows. auto/active-cv uses CV1 plus CV2 when active; primary ignores CV2; distance preserves legacy terminal-distance scoring.")
    p.add_argument("--seed-secondary-weight", type=float, default=1.0, help="Relative weight of CV2 in active-cv GENPEPT seed scoring. 0 makes active-cv equivalent to primary-only scoring.")
    p.add_argument("--seed-max-reuse-per-conformer", type=int, default=0, help="Maximum number of windows that may reuse the same GENPEPT survivor during seed selection. 0 means unlimited reuse.")


def _add_genpept_prescan_args(p: argparse.ArgumentParser) -> None:
    """Add GAREUS-side GENPEPT prescan prior arguments."""
    p.add_argument("--genpept-prescan-enabled", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--genpept-prescan-dir", type=Path, default=None)
    p.add_argument("--genpept-prescan-stages", nargs="*", default=None)
    p.add_argument("--genpept-prescan-rescore-active-cvs", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--genpept-prescan-contact-bin-width", type=float, default=0.025)
    p.add_argument("--genpept-prescan-rama-bin-width", type=float, default=0.25)
    p.add_argument("--genpept-prescan-min-hits-per-bin", type=int, default=2)
    p.add_argument("--genpept-prescan-frontier-stages", nargs="*", default=None)
    p.add_argument("--genpept-prescan-use-as-window-prior", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--genpept-prescan-use-as-seed-library", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--genpept-prescan-absence-means-unknown", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--genpept-prescan-output-prefix", default="genpept_prescan")
    p.add_argument("--genpept-prescan-write-maps", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--genpept-prescan-write-seed-assignments", action=argparse.BooleanOptionalAction, default=True)


def _add_genpept_prior_args(p: argparse.ArgumentParser) -> None:
    """Add GENPEPT round-zero window prior arguments."""
    p.add_argument("--genpept-prior-enabled", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--genpept-prior-dir", type=Path, default=None)
    p.add_argument("--genpept-prior-stages", nargs="*", default=None)
    p.add_argument("--genpept-prior-max-structures", type=int, default=50000)
    p.add_argument("--genpept-prior-min-points", type=int, default=10)
    p.add_argument("--genpept-prior-max-windows", type=int, default=48)
    p.add_argument("--genpept-prior-bins", default="24,12")
    p.add_argument("--genpept-prior-min-hits-per-bin", type=int, default=2)
    p.add_argument("--genpept-prior-basin-windows", type=int, default=0)
    p.add_argument("--genpept-prior-bridge-windows", type=int, default=0)
    p.add_argument("--genpept-prior-frontier-windows", type=int, default=0)
    p.add_argument("--genpept-prior-probe-windows", type=int, default=0)
    p.add_argument("--genpept-prior-absence-means-unknown", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--genpept-prior-snap-secondary-centers", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--genpept-prior-required", action=argparse.BooleanOptionalAction, default=False)


def _add_gamd_args(p: argparse.ArgumentParser) -> None:
    """Add GaMD integrator, exchange, and production control arguments."""
    p.add_argument("--gamd-boost-type", default="lower-dual", choices=[
        "gamd-cmd-base", "lower-total", "upper-total", "lower-dihedral", "upper-dihedral",
        "lower-dual", "upper-dual", "lower-nonbonded", "upper-nonbonded",
        "lower-dual-nonbonded-dihedral", "upper-dual-nonbonded-dihedral",
    ])
    p.add_argument("--sigma0p-kcal-mol", type=float, default=6.0, help="Primary sigma0 for gamd-openmm.")
    p.add_argument("--sigma0d-kcal-mol", type=float, default=6.0, help="Secondary sigma0 for dual boost gamd-openmm.")
    p.add_argument("--gamd-cmd-prep-steps", type=int, default=5000)
    p.add_argument("--gamd-cmd-steps", type=int, default=50000)
    p.add_argument("--gamd-equil-prep-steps", type=int, default=5000)
    p.add_argument("--gamd-equil-steps", type=int, default=50000)
    p.add_argument("--gamd-production-steps", "--production-steps", dest="gamd_production_steps", type=int, default=500000, help="Production steps per replica. Historical name retained; --production-steps is the run-mode-neutral alias.")
    p.add_argument("--gamd-averaging-window", type=int, default=5000)
    p.add_argument("--exchange-interval", type=int, default=5000)
    p.add_argument("--exchange-mode", choices=["neighbor", "random-pair", "all-pair-sweep", "gibbs-walk"], default="neighbor", help="Umbrella state exchange scheme. neighbor = adjacent window REUS ladder; random-pair = arbitrary disjoint Metropolis window swaps; all-pair-sweep = randomized Metropolis sweep over many/all window pairs; gibbs-walk = experimental heat-bath-like long-jump window swap update.")
    p.add_argument("--exchange-random-pairs", type=int, default=0, help="For --exchange-mode random-pair, number of disjoint arbitrary window pairs to try per exchange interval. 0 = floor(n_windows/2).")
    p.add_argument("--exchange-max-pairs-per-interval", type=int, default=0, help="Cap attempted pairs/moves per interval for all-pair-sweep or gibbs-walk. 0 = no cap; all-pair-sweep tries all window pairs, gibbs-walk visits all replicas once.")
    p.add_argument("--report-interval", type=int, default=5000)
    p.add_argument("--traj-interval", type=int, default=5000)
    p.add_argument("--traj-format", choices=["dcd", "xtc", "none"], default="dcd", help="Production trajectory format for replica_trajectories. dcd preserves historical behavior; xtc writes compressed XTC trajectories when OpenMM provides XTCReporter; none disables coordinate trajectory reporters without affecting scalar reports/samples.")
    p.add_argument("--adaptive-pilot-trajectories", action=argparse.BooleanOptionalAction, default=False, help="Write coordinate trajectory reporters during adaptive-feedback pilot rounds. Default false because pilot coordinates are diagnostic/disposable and can dominate filesystem I/O.")
    p.add_argument("--randomize-replica-velocities", action="store_true")
    p.add_argument("--checkpoint-interval", type=int, default=50000, help="Production steps between overwriting restart checkpoints; 0 disables checkpoint writing.")
    p.add_argument("--resume", action="store_true", help="Resume from --out checkpoints: load production Context checkpoints directly when available, otherwise reuse saved NPT state XML to skip minimization/equilibration. Appends samples/exchanges/distances when possible.")
    p.add_argument("--production-probe-steps", type=int, default=20, help="Rollback preflight steps per replica before production; catches NaNs without consuming production time. 0 disables.")
    p.add_argument("--production-probe-warn-only", action="store_true", help="Warn instead of aborting if the rollback production probe fails.")
    p.add_argument("--production-safe-chunk-steps", type=int, default=0, help="Optional maximum subchunk size for production stepping. 0 uses the normal next-event chunk. Smaller values make NaN diagnostics more local but add overhead.")
    p.add_argument("--production-nan-diagnostics", action=argparse.BooleanOptionalAction, default=True, help="When production stepping fails, scan replicas and write PRODUCTION_NAN_DIAGNOSTICS_*.json plus crash PDBs when possible.")
    p.add_argument("--shared-gamd-copy-strict", action="store_true", help="Abort if the copied shared GaMD integrator globals differ from the reference setup before production.")


def _add_output_args(p: argparse.ArgumentParser) -> None:
    """Add GUI/TUI progress, dashboard, distance, and analysis I/O arguments."""
    # GUI/progress/TUI output. The JSONL files are intentionally simple so an external GUI can tail them.
    p.add_argument("--progress-mode", choices=["none", "console", "jsonl", "both"], default="both")
    p.add_argument("--progress-jsonl", default="progress.jsonl")
    p.add_argument("--progress-update-interval-sec", type=float, default=0.25)
    p.add_argument("--progress-bar-width", type=int, default=36)
    p.add_argument("--color", choices=["auto", "always", "never"], default="auto")
    p.add_argument("--tui-mode", choices=["dashboard", "interactive", "line", "none"], default="dashboard")
    p.add_argument("--tui-clear-mode", choices=["auto", "always", "never"], default="always", help="In full-frame TUI modes, clear and redraw the visible terminal instead of stacking progress/TUI panels. auto is treated like always because MPI/SLURM/tee often hide TTY status; use never for plain logs.")
    p.add_argument("--dashboard-density", choices=["auto", "compact", "normal", "full"], default="auto", help="Dashboard information density. auto compacts on small terminals and expands on large terminals without hiding panels.")
    p.add_argument("--dashboard-wide-threshold", type=int, default=132, help="Terminal width at which wide 2D side-by-side dashboard layout becomes eligible.")
    p.add_argument("--dashboard-min-panel-width", type=int, default=30, help="Minimum readable panel width before dashboard rows stack vertically.")
    p.add_argument("--dashboard-max-height", type=int, default=0, help="Optional maximum visible dashboard lines before truncation; 0 uses terminal height.")
    p.add_argument("--dashboard-render-interval-sec", type=float, default=0.0, help="Minimum wall-clock seconds between live dashboard frame renders. 0 preserves historical render-every-log behavior.")
    p.add_argument("--dashboard-panels", choices=["minimal", "normal", "full"], default="normal", help="Live dashboard panel set. minimal keeps only the context/decision/CV panels; normal preserves the standard dashboard; full keeps all panels.")
    p.add_argument("--dashboard-heavy-panels-every", type=int, default=1, help="Render heavy dashboard panels only every N rendered frames. 1 preserves historical behavior.")

    p.add_argument("--distance-output-mode", choices=["none", "csv", "jsonl", "both"], default="both")
    p.add_argument("--distance-output-interval", type=int, default=1000, help="Steps between distance/TUI updates; 0 reuses min(report, exchange).")
    p.add_argument("--distance-csv", default="distances.csv")
    p.add_argument("--distance-jsonl", default="distances.jsonl")
    p.add_argument("--no-distance-gui-events", action="store_true", help="Do not mirror distance events into progress.jsonl.")
    p.add_argument("--distance-ascii-mode", choices=["none", "compact", "bars", "hist", "hist3d"], default="hist3d")
    p.add_argument("--distance-ascii-width", type=int, default=54)
    p.add_argument("--distance-ascii-max-replicas", type=int, default=32)
    p.add_argument("--distance-history-limit", type=int, default=4000)
    # No-database I/O optimization. Heavy N x K analysis matrices are written as
    # chunked NumPy files; CSV remains a scalar human-readable companion by default.
    p.add_argument("--analysis-chunk-size", type=int, default=10000, help="Number of production sample rows per analysis_chunks/chunk_XXXXXX.npz file.")
    p.add_argument("--analysis-chunk-dir", default="analysis_chunks", help="Directory name under --out for chunked NumPy analysis arrays.")
    p.add_argument("--write-analysis-chunks", action=argparse.BooleanOptionalAction, default=True, help="Write chunked NumPy analysis arrays during production.")
    p.add_argument("--analysis-write-consolidated-npz", action=argparse.BooleanOptionalAction, default=True, help="Also write legacy consolidated analysis_arrays.npz at the end for compatibility. Disable for very large production runs.")
    p.add_argument("--analysis-consolidated-max-elements", type=int, default=100_000_000, help="Skip writing legacy consolidated analysis_arrays.npz when n_samples*n_windows exceeds this value. 0 disables the guard.")
    p.add_argument("--analysis-array-dtype", choices=["float64", "float32"], default="float64", help="Floating dtype for consolidated analysis_arrays.npz. Chunk files remain full precision; float32 can reduce compatibility-NPZ size after validation.")
    p.add_argument("--analysis-npz-compressed", action=argparse.BooleanOptionalAction, default=False, help="Use compressed NPZ chunks/consolidated arrays. Smaller files, more CPU.")
    p.add_argument("--write-full-bias-csv-vectors", action="store_true", help="Write all-window bias JSON vectors into samples.csv. Default false keeps samples.csv scalar-only; full matrices are in analysis_chunks.")
    p.add_argument("--write-gamd-globals-json", action="store_true", help="Write full integrator diagnostic globals JSON into samples.csv. Default false reduces text I/O.")
    p.add_argument("--sample-potential-energy", action=argparse.BooleanOptionalAction, default=True, help="Evaluate and log potential energy for samples.csv/distances.csv rows. Disable with --no-sample-potential-energy to avoid an extra GPU energy evaluation when only CV/bias observables are needed; trajectories, forces, GaMD, and exchange probabilities are unchanged.")
    p.add_argument("--flush-every-log", action=argparse.BooleanOptionalAction, default=True, help="Flush scalar CSV writers after every sample/distance/exchange log event. Disable with --no-flush-every-log to rely on --csv-flush-rows buffering for faster network-filesystem runs; files are still flushed on checkpoints and clean shutdown.")
    p.add_argument("--csv-flush-rows", type=int, default=1000, help="Buffered CSV rows before flushing samples/exchanges/distances.")
    p.add_argument("--jsonl-flush-rows", type=int, default=500, help="Buffered JSONL events before flushing progress/distances.")


def _add_platform_args(p: argparse.ArgumentParser) -> None:
    """Add OpenMM platform / GPU device selection arguments."""
    p.add_argument("--platform", default="auto", help="CUDA, HIP, OpenCL, CPU, Reference, or auto. Used for production replicas.")
    p.add_argument("--precision", default="mixed", help="Precision for production replicas on CUDA/HIP/OpenCL.")
    p.add_argument("--device-index", default="0", help="OpenMM DeviceIndex for production. With --replica-device-mode auto/round-robin, comma lists such as 0,1,2,3 are assigned one token per replica.")
    p.add_argument("--replica-device-mode", choices=["auto", "round-robin", "single-context-split", "manual"], default="auto", help="How to interpret comma-separated --device-index for production replicas. auto uses round-robin when multiple device tokens are present; single-context-split gives each replica the full comma list; manual uses --replica-device-map.")
    p.add_argument("--replica-device-map", default="", help="Explicit repeating per-replica DeviceIndex map, e.g. 0,1,2,3,0,1. Used by --replica-device-mode manual, or as an override for round-robin.")
    p.add_argument("--cuda-use-cpu-pme", choices=["auto", "true", "false"], default="auto", help="Optional OpenMM CUDA/HIP UseCpuPme platform property. auto leaves the property unset.")
    p.add_argument("--cuda-use-blocking-sync", choices=["auto", "true", "false"], default="auto", help="Optional OpenMM CUDA/HIP UseBlockingSync platform property. auto leaves the property unset.")
    p.add_argument("--cuda-deterministic-forces", choices=["auto", "true", "false"], default="auto", help="Optional OpenMM CUDA/HIP DeterministicForces platform property. auto leaves the property unset.")
    p.add_argument("--platform-temp-directory", default="", help="Optional CUDA/HIP TempDirectory platform property for temporary files.")
    p.add_argument("--cpu-threads", type=int, default=1, help="CPU threads for production when --platform CPU is used.")
    p.add_argument("--setup-platform", default="", help="OpenMM platform for single-context setup/prep phases: minimization/equilibration, adaptive prescan, US pulling, and shared GaMD setup. Empty = --platform.")
    p.add_argument("--setup-precision", default="", help="Precision for setup on CUDA/HIP/OpenCL. Empty = --precision.")
    p.add_argument("--setup-device-index", default="", help="DeviceIndex for single-context setup/prep phases. Empty = first entry of --device-index, so --device-index 0,1,2,3 uses only GPU 0 for setup. Set this explicitly, e.g. 1 or 0,1,2,3, to override.")
    p.add_argument("--setup-cpu-threads", type=int, default=0, help="CPU threads for setup when --setup-platform CPU is used. 0 = --cpu-threads.")
    p.add_argument("--us-pull-workers", default="auto", help="Parallel workers for US starting-structure pulls. 'auto': on CUDA/OpenCL uses one worker per GPU device token (from --setup-device-index / --device-index), on CPU uses 1. Explicit N overrides. Each worker owns a separate OpenMM Context; threads run concurrently since OpenMM releases the GIL during step().")
    p.add_argument("--scratchdir", default="", help="Fast local scratch directory for all simulation I/O (e.g. /local/nvme/run). If set, --out becomes a mirror that is updated at each checkpoint via a full directory copy. Use on HPC nodes with local NVMe to avoid writing DCD/NPZ/CSV traffic over a network filesystem.")


# ---------------------------------------------------------------------------
# Post-parse processing helpers
# ---------------------------------------------------------------------------

def _resolve_cv_aliases(args: argparse.Namespace, argv_list: list) -> None:  # noqa: ARG001
    """Resolve --cv1/--cv2 friendly aliases onto the canonical CV flags."""
    # User-facing aliases.  The expert flags remain canonical in output metadata,
    # but --cv1/--cv2 make the common 1D/2D choices short and explicit.
    if getattr(args, "cv1", None) not in (None, ""):
        args.primary_cv = "nonlocal-contacts" if str(args.cv1).strip().lower().replace("_", "-") in {"contacts", "nonlocal-contacts"} else "distance"

    _cv2_requested = getattr(args, "cv2", None) not in (None, "")
    if _cv2_requested:
        args.secondary_cv = str(args.cv2)
    args.primary_cv = primary_cv_mode(args)
    args.secondary_cv = secondary_cv_mode(args)

    # Make --cv2 deliberately mean a 2D primary x secondary run.  The historical
    # --secondary-cv flag still supports a single fixed secondary restraint unless
    # the user supplies multiple --secondary-cv-centers themselves.
    if _cv2_requested and str(args.secondary_cv) != "none" and not getattr(args, "secondary_cv_centers", None):
        if args.secondary_cv == "rama-map":
            args.secondary_cv_centers = [-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0]
        elif args.secondary_cv == "rama-regions":
            args.secondary_cv_centers = [-1.0, -0.5, 0.0, 0.5, 1.0]
        elif args.secondary_cv == "alpha-coil-beta":
            args.secondary_cv_centers = [-0.8, 0.0, 0.8]
        else:
            args.secondary_cv_centers = [0.2, 0.5, 0.8]
        args._cv2_auto_centers = True
    else:
        args._cv2_auto_centers = False


def _normalize_run_mode(args: argparse.Namespace, argv_list: list) -> None:
    """Validate and normalise --run-mode; set HMR and auto-timestep flags."""
    _run_mode = str(getattr(args, "run_mode", "gamd") or "gamd").strip().lower().replace("_", "-")
    if _run_mode not in {"cmd", "hmr-cmd", "gamd", "hmr-gamd"}:
        raise ValueError("--run-mode must be one of: cmd, hmr-cmd, gamd, hmr-gamd")
    args.run_mode = _run_mode
    if args.run_mode in {"hmr-cmd", "hmr-gamd"}:
        args.hmr = True
        _specified_timestep_cli = any(str(tok) == "--timestep-fs" or str(tok).startswith("--timestep-fs=") for tok in argv_list)
        _specified_timestep_config = "timestep_fs" in (getattr(args, "_config_values", {}) or {})
        if not _specified_timestep_cli and not _specified_timestep_config:
            args.timestep_fs = 4.0
            args._run_mode_auto_timestep_fs = True
        else:
            args._run_mode_auto_timestep_fs = False
    else:
        args._run_mode_auto_timestep_fs = False


def _validate_contact_args(args: argparse.Namespace) -> None:
    """Validate contact-CV argument combinations when nonlocal-contacts is active.

    Assumes ``args.contact_scheme`` has already been set by the caller.
    """
    if args.primary_cv != "nonlocal-contacts":
        return
    if args.contact_scheme not in {"atom-pairs", "residue-balanced", "ca-pairs"}:
        raise ValueError("--contact-scheme must be atom-pairs, residue-balanced, or ca-pairs")
    # Explicit sparse 2D window tables are generic over the primary CV.
    # In contact mode the primary values are dimensionless contact-CV centers
    # even when legacy distance_* column aliases are used for compatibility.
    if not math.isfinite(float(getattr(args, "contact_r0_a", 4.5))) or float(getattr(args, "contact_r0_a", 4.5)) <= 0.0:
        raise ValueError("--contact-r0-a must be positive and finite")
    if not math.isfinite(float(getattr(args, "contact_beta_a_inv", 6.0))) or float(getattr(args, "contact_beta_a_inv", 6.0)) <= 0.0:
        raise ValueError("--contact-beta-a-inv must be positive and finite")
    if int(getattr(args, "contact_min_sequence_separation", 4) or 4) < 1:
        raise ValueError("--contact-min-sequence-separation must be at least 1")
    if bool(getattr(args, "contact_normalize", True)):
        cmin = float(getattr(args, "contact_adaptive_min", 0.0) or 0.0)
        cmax = float(getattr(args, "contact_adaptive_max", 0.80) or 0.80)
        if not (math.isfinite(cmin) and math.isfinite(cmax)) or cmax <= cmin:
            raise ValueError("--contact-adaptive-max must be finite and greater than --contact-adaptive-min")
        if cmin < -1.0e-8 or cmax > 1.0 + 1.0e-8:
            raise ValueError("Normalized contact adaptive bounds should be within [0, 1]")
    if float(getattr(args, "contact_adaptive_target_spacing", 0.15) or 0.15) <= 0.0:
        raise ValueError("--contact-adaptive-target-spacing must be positive")
    if int(getattr(args, "contact_adaptive_min_windows", 4) or 4) < 2:
        raise ValueError("--contact-adaptive-min-windows must be at least 2")
    if int(getattr(args, "contact_adaptive_max_windows", 12) or 12) < int(getattr(args, "contact_adaptive_min_windows", 4) or 4):
        raise ValueError("--contact-adaptive-max-windows must be >= --contact-adaptive-min-windows")
    _autocalib_steps = getattr(args, "contact_autocalibration_steps", None)
    if _autocalib_steps is not None and int(_autocalib_steps) < 0:
        raise ValueError("--contact-autocalibration-steps must be >= 0")
    _autocalib_ts = getattr(args, "contact_autocalibration_timestep_fs", None)
    if _autocalib_ts is not None and float(_autocalib_ts) <= 0.0:
        raise ValueError("--contact-autocalibration-timestep-fs must be positive")
    if int(getattr(args, "contact_autocalibration_sample_interval", 100) or 100) <= 0:
        raise ValueError("--contact-autocalibration-sample-interval must be positive")
    if float(getattr(args, "contact_autocalibration_min_span", 0.10) or 0.10) <= 0.0:
        raise ValueError("--contact-autocalibration-min-span must be positive")
    if (
        not bool(getattr(args, "resume", False))
        and not bool(getattr(args, "self_test_primary_cv_force", False))
        and str(getattr(args, "window_mode", "manual")) == "manual"
        and (getattr(args, "contact_centers", None) is None or len(args.contact_centers) == 0)
    ):
        raise ValueError("--primary-cv nonlocal-contacts with --window-mode manual requires --contact-centers; adaptive/adaptive-feedback can generate them.")


def _validate_total_window_bounds(args: argparse.Namespace) -> None:
    """Validate adaptive-min/max-total-windows consistency for both CV axes."""
    for _min_name, _max_name in (
        ("adaptive_min_total_windows", "adaptive_max_total_windows"),
        ("contact_adaptive_min_total_windows", "contact_adaptive_max_total_windows"),
    ):
        _min_total = int(getattr(args, _min_name, 0) or 0)
        _max_total = int(getattr(args, _max_name, 0) or 0)
        if _min_total < 0 or _max_total < 0:
            raise ValueError(f"--{_min_name.replace('_','-')} and --{_max_name.replace('_','-')} must be >= 0")
        if _min_total > 0 and _max_total > 0 and _min_total > _max_total:
            raise ValueError(f"--{_min_name.replace('_','-')} cannot exceed --{_max_name.replace('_','-')}")
        if _max_total > 0 and _max_total < 2:
            raise ValueError(f"--{_max_name.replace('_','-')} must be at least 2 when enabled")


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Iterable[str]] = None):
    argv_list = _argv_as_list(argv)

    # Pre-parse: read --config and --write-config-template before building the
    # full parser so config defaults can be injected and templates can be
    # written without requiring --seq.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    pre.add_argument("--write-config-template", nargs="?", const="chignolin_adaptive_feedback.yaml", default=None)
    pre_args, _pre_unknown = pre.parse_known_args(argv_list)

    p = argparse.ArgumentParser(
        prog="gareus",
        add_help=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="GAREUS peptide GaMD/REUS workflow. Use -h for concise help or -hh for the method encyclopedia.",
    )
    _add_core_args(p)
    _add_system_args(p)
    _add_cv_args(p)
    _add_window_args(p)
    _add_seeding_args(p)
    _add_genpept_prescan_args(p)
    _add_genpept_prior_args(p)
    _add_gamd_args(p)
    _add_output_args(p)
    _add_platform_args(p)

    if pre_args.write_config_template:
        _write_config_template(Path(pre_args.write_config_template))
        print(f"Wrote config template: {pre_args.write_config_template}")
        raise SystemExit(0)

    config_info = _apply_config_defaults_to_parser(p, pre_args.config)
    args = p.parse_args(argv_list)
    args._config_values = config_info.get("config_values", {})

    _resolve_cv_aliases(args, argv_list)
    _normalize_run_mode(args, argv_list)
    args.contact_scheme = contact_scheme(args)
    _validate_contact_args(args)
    _validate_total_window_bounds(args)

    if args.hmr and float(args.hydrogen_mass_amu) <= 0:
        args.hydrogen_mass_amu = 3.024
    return args


def main(argv: Optional[Iterable[str]] = None):
    _graceful_shutdown.clear()
    argv_list = _argv_as_list(argv)
    args = parse_args(argv_list)
    args.seq = validate_sequence(args.seq)
    out_dir = Path(os.path.expandvars(str(args.out)))
    _scratchdir = os.path.expandvars(str(getattr(args, "scratchdir", "") or "")).strip()
    if _scratchdir:
        _main_dir = out_dir
        out_dir = Path(_scratchdir)
        out_dir.mkdir(parents=True, exist_ok=True)
        args._main_dir = str(_main_dir)
        args.out = str(out_dir)
        _main_dir.mkdir(parents=True, exist_ok=True)
        print(f"[scratchdir] I/O → {out_dir}  (synced to {_main_dir} at each checkpoint)")
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
    configure_color(args.color)
    if bool(getattr(args, "self_test_primary_cv_force", False)):
        if primary_cv_is_contacts(args):
            result = self_test_nonlocal_contact_force(args)
            print(json.dumps(result, indent=2, sort_keys=True))
            return
        else:
            print(json.dumps({"ok": True, "primary_cv": primary_cv_mode(args), "note": "distance primary CV uses the historical CustomBondForce path; full validation occurs in normal smoke tests."}, indent=2, sort_keys=True))
            return
    _write_reproducibility_files(args, out_dir, argv=argv_list)
    public_args = {k: v for k, v in vars(args).items() if not str(k).startswith("_")}
    if bool(getattr(args, "resume", False)) and (out_dir / "run_args.json").exists():
        write_json(out_dir / "last_resume_args.json", public_args)
    else:
        write_json(out_dir / "run_args.json", public_args)
    initialize_run_manifest(args, out_dir, argv=argv_list)
    progress = GuiProgressSink(out_dir, args)
    _run_status = "started"
    _run_error = None
    try:
        progress.emit({"event": "run_start", "sequence": args.seq, "out": str(out_dir), "resume": bool(getattr(args, "resume", False))})

        if bool(getattr(args, "resume", False)):
            # --resume: skip all setup/pilots and go straight to production.
            #
            # For adaptive-feedback runs the production checkpoint lives in
            # final_production/ even when the user passes the parent directory as
            # --out.  Search both locations so either invocation works.
            resume_dir = None
            for _candidate in [out_dir, out_dir / "final_production"]:
                if production_checkpoint_available(_candidate):
                    resume_dir = _candidate
                    break

            if resume_dir is None:
                _manifest_paths = [
                    out_dir / "checkpoints" / "production_checkpoint_manifest.json",
                    out_dir / "final_production" / "checkpoints" / "production_checkpoint_manifest.json",
                ]
                print()
                print("ERROR: --resume requires a production checkpoint, but none was found.")
                print(f"  Searched: {out_dir}")
                print(f"            {out_dir / 'final_production'}")
                print("  Expected file in either location:")
                for _p in _manifest_paths:
                    _exists = _p.exists()
                    _chk_files_ok = False
                    if _exists:
                        try:
                            import json as _json
                            _m = _json.loads(_p.read_text())
                            _chk_files_ok = all((_p.parent / str(f)).exists() for f in _m.get("replica_checkpoint_files", []))
                        except Exception:
                            pass
                    print(f"    {_p}  [{'found but .chk files missing' if _exists and not _chk_files_ok else 'found' if _exists else 'NOT FOUND'}]")
                print()
                print("Possible causes:")
                print("  - Wrong --out path (does not match the original run directory)")
                print("  - Run was cancelled before the first checkpoint was written")
                print(f"    (default checkpoint interval is 50000 steps; pass --checkpoint-interval N to tune)")
                print("  - Checkpoint files were deleted or moved")
                print()
                print("To start a new run from scratch omit --resume.")
                progress.emit({"event": "resume_failed", "reason": "no_production_checkpoint", "out": str(out_dir)})
                return

            if resume_dir != out_dir:
                print(f"[resume] Production checkpoint found in {resume_dir}; using as effective output directory.")
                out_dir = resume_dir
                args.out = str(out_dir)
                _write_reproducibility_files(args, out_dir, argv=argv_list)
                initialize_run_manifest(args, out_dir, argv=argv_list)

            # The solvated topology PDB may live in a parent dir (e.g. when
            # resuming final_production/ whose 01_solvated_start.pdb is one
            # level up).  Search up to two levels.
            loaded_resume_setup = None
            for _pdb_dir in list(dict.fromkeys([out_dir, out_dir.parent, out_dir.parent.parent])):
                _setup = load_existing_openmm_setup_for_resume(args, _pdb_dir, require_equil_state=False)
                if _setup is not None:
                    loaded_resume_setup = _setup
                    break

            if loaded_resume_setup is None:
                print()
                print("ERROR: --resume found a production checkpoint but could not load the solvated topology.")
                print(f"  Searched for 01_solvated_start.pdb in:")
                for _pdb_dir in list(dict.fromkeys([out_dir, out_dir.parent, out_dir.parent.parent])):
                    _pdb = _pdb_dir / "01_solvated_start.pdb"
                    print(f"    {_pdb}  [{'found' if _pdb.exists() else 'NOT FOUND'}]")
                print()
                print("The solvated PDB is written at the start of every fresh run.")
                print("If it was deleted, start a new run from scratch without --resume.")
                progress.emit({"event": "resume_failed", "reason": "no_solvated_topology", "out": str(out_dir)})
                return

            openmm, app, unit, forcefield, topology, equil_state = loaded_resume_setup
            print("[resume] Reusing existing solvated topology and production checkpoints.")
            progress.emit({"event": "resume_setup_loaded", "production_checkpoint": True, "out": str(out_dir)})

            # Always call run_gareus directly — adaptive-feedback pilots are
            # already complete and must not be re-run on resume.
            run_gareus(args, out_dir, openmm, app, unit, forcefield, topology, equil_state, progress=progress)

        else:
            # Fresh start: equilibrate, then run the requested workflow.
            openmm, app, unit, forcefield, topology, equil_state = minimize_and_npt_equilibrate(args, out_dir, progress=progress)
            if str(getattr(args, "window_mode", "adaptive")) == "adaptive-feedback":
                run_adaptive_feedback_auto_loop(args, out_dir, openmm, app, unit, forcefield, topology, equil_state, progress=progress)
            else:
                run_gareus(args, out_dir, openmm, app, unit, forcefield, topology, equil_state, progress=progress)

        progress.emit({"event": "run_complete", "out": str(out_dir)})
        _run_status = "completed"
    except BaseException as exc:
        _run_status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        _run_error = exc
        raise
    finally:
        try:
            finalize_run_manifest(args, out_dir, status=_run_status, error=_run_error)
        except Exception as _prov_exc:
            try:
                print(f"WARNING: failed to finalize run provenance manifest: {_prov_exc}", flush=True)
            except Exception:
                pass
        progress.close()
        _final_main = getattr(args, "_main_dir", None)
        if _final_main:
            print(f"[scratchdir] final sync → {_final_main}")
            sync_scratch_to_main(out_dir, Path(_final_main))
