"""Command-line parser and top-level workflow for GAREUS (schema v2.0)."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import warnings
from pathlib import Path
from typing import Iterable, Optional

from .adaptive_feedback import run_adaptive_feedback_auto_loop
from .adaptive_production import run_adaptive_production_auto_loop
from .checkpoints import production_checkpoint_available, load_existing_openmm_setup_for_resume
from .colors import configure_color
from .config import (
    _argv_as_list,
    _apply_config_defaults_to_parser,
    _build_known_config_dests,
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
# Argument group builders
# ---------------------------------------------------------------------------

def _add_core_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("-h", "--help", action=SimpleHelpAction)
    p.add_argument("-hh", "--help-heavy", action=HeavyHelpAction)
    p.add_argument("--config", default=None, help="YAML/JSON config file.")
    p.add_argument("--profile", default=None,
                   help="Named bundle of boilerplate config keys (see config_profiles.py). "
                        "Explicit YAML/CLI keys override anything from the profile.")
    p.add_argument("--write-config-template", nargs="?", const="gareus_template.yaml", default=None)
    p.add_argument("--write-effective-config", action="store_true")
    p.add_argument("--seq", required=True, help="One-letter peptide sequence.")
    p.add_argument("--input-pdb", default=None,
                   help="Use this prebuilt PDB verbatim (skip PeptideBuilder + addHydrogens). "
                        "For capped/non-standard structures, e.g. Ace-Ala-Nme. Relative paths "
                        "resolve against the launch directory.")
    p.add_argument("--out", default="gareus_out", help="Output directory.")
    p.add_argument("--seed", type=int, default=2026)


def _add_system_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--water-model", choices=["tip3p", "tip3pfb", "spce", "tip4pew"], default="tip3p")
    p.add_argument("--box-shape", choices=["dodecahedron", "cube", "octahedron"], default="dodecahedron")
    p.add_argument("--padding-nm", type=float, default=1.0)
    p.add_argument("--ionic-strength-molar", type=float, default=0.15)
    p.add_argument("--temperature-k", type=float, default=300.0)
    p.add_argument("--pressure-bar", type=float, default=1.0)
    p.add_argument("--barostat-frequency", type=int, default=100)
    p.add_argument("--production-ensemble", choices=["npt", "nvt"], default="npt")
    p.add_argument("--minimize-iterations", type=int, default=20000)
    p.add_argument("--npt-steps", type=int, default=100000)
    p.add_argument("--nvt-warmup-steps", type=int, default=20000)
    p.add_argument("--equil-restraint-k-kcal-mol-a2", type=float, default=2.0)
    p.add_argument("--equil-friction-per-ps", type=float, default=10.0)
    # Simulation
    p.add_argument("--run-mode", choices=["cmd", "hmr-cmd", "gamd", "hmr-gamd"], default="gamd",
                   help="cmd/gamd = no HMR; hmr-cmd/hmr-gamd = HMR + 4 fs default timestep.")
    p.add_argument("--timestep-fs", type=float, default=2.0)
    p.add_argument("--friction-per-ps", type=float, default=1.0)
    p.add_argument("--nonbonded-cutoff-nm", type=float, default=0.8)
    p.add_argument("--ewald-error-tolerance", type=float, default=1e-4)
    # Expert equil ramp (rarely changed, kept for completeness)
    p.add_argument("--nvt-warmup-timestep-fs", type=float, default=0.25)
    p.add_argument("--npt-ramp-steps", type=int, default=50000)
    p.add_argument("--npt-ramp-timestep-fs", type=float, default=0.5)


def _add_cv_args(p: argparse.ArgumentParser) -> None:
    # Primary / secondary CV selectors
    p.add_argument("--cv1", choices=["distance", "contacts", "nonlocal-contacts"], default=None,
                   help="Primary CV: distance or contacts.")
    p.add_argument("--cv2", choices=["none", "alpha", "beta", "alpha-coil-beta", "acb",
                                      "rama-map", "rama", "custom", "tica", "tica-linear",
                                      "torsion-pca", "bootstrap-torsion", "bootstrap-linear",
                                      "torsion-linear"], default=None,
                   help="Secondary CV for 2D workflow. Default centers inserted automatically.")
    # Expert atom override
    p.add_argument("--cv-atom1", default=None, help="Explicit primary CV atom, e.g. 1:CA.")
    p.add_argument("--cv-atom2", default=None, help="Explicit primary CV atom, e.g. -1:CA.")

    # Contact CV physics (contact-specific, no distance analog)
    p.add_argument("--contact-scheme", choices=["atom-pairs", "residue-balanced", "ca-pairs"],
                   default="atom-pairs")
    p.add_argument("--contact-atom-selection",
                   choices=["heavy", "ca", "backbone-heavy", "sidechain-heavy", "sidechain-all", "all"],
                   default="heavy")
    p.add_argument("--contact-min-sequence-separation", type=int, default=4)
    p.add_argument("--contact-r0-a", type=float, default=4.5,
                   help="Contact switching midpoint in Å.")
    p.add_argument("--contact-beta-a-inv", type=float, default=6.0,
                   help="Contact switching steepness in 1/Å.")
    p.add_argument("--contact-normalize", action=argparse.BooleanOptionalAction, default=True)

    # CV1 adaptive — unified regardless of CV type
    p.add_argument("--cv1-range-min", type=float, default=0.0,
                   help="Adaptive lower bound (Å for distance, dimensionless for contacts). 0 = auto.")
    p.add_argument("--cv1-range-max", type=float, default=0.0,
                   help="Adaptive upper bound. 0 = auto.")
    p.add_argument("--cv1-target-spacing", type=float, default=0.0,
                   help="Initial adaptive window spacing (units match CV1). 0 = auto.")
    p.add_argument("--cv1-k-default", type=float, default=0.0,
                   help="Default/fixed CV1 force constant. 0 = auto (spacing-derived).")
    p.add_argument("--cv1-k-min", type=float, default=0.0,
                   help="Lower clamp for adaptive CV1 k. 0 = auto.")
    p.add_argument("--cv1-k-max", type=float, default=0.0,
                   help="Upper clamp for adaptive CV1 k. 0 = auto.")
    p.add_argument("--cv1-k-mode", choices=["spacing", "fixed", "constant"], default="spacing",
                   help="CV1 force-constant assignment mode.")
    p.add_argument("--cv1-adaptive-overlap-sigma", type=float, default=2.3,
                   help="Overlap-sigma for spacing-derived CV1 k. "
                        "spacing/sigma_gaussian ratio; erfc(sigma/(2*sqrt(2))) gives actual overlap. "
                        "Default 2.3 yields ~0.25 fractional overlap, matching --target-overlap default.")
    p.add_argument("--cv1-boundary-pull-steps", type=int, default=5000,
                   help="Steps per direction for CV boundary pull (0 = skip, use configured range).")
    p.add_argument("--cv1-boundary-pull-k", type=float, default=50.0,
                   help="Force constant for CV boundary pulls in kcal/mol/CV².")
    p.add_argument("--cv1-boundary-pull-margin", type=float, default=0.0,
                   help="Extra margin added beyond achieved extreme when setting window bounds.")
    p.add_argument("--cv1-frontier", action=argparse.BooleanOptionalAction, default=False,
                   help="Enable CV1 frontier probing during adaptive-feedback.")
    p.add_argument("--cv1-frontier-probe-count", type=int, default=2,
                   help="Number of frontier probe windows.")

    # CV2 adaptive — secondary CV parameters
    p.add_argument("--cv2-centers", nargs="*", type=float, default=None,
                   help="Secondary CV targets. One value = all windows same; multiple = 2D grid.")
    p.add_argument("--cv2-n-centers", type=int, default=3,
                   help="Number of data-derived CV2 quantile centers for torsion-pca "
                        "auto-centering (ignored if --cv2-centers is set explicitly). Must be >= 3.")
    p.add_argument("--cv2-k-default", type=float, default=50.0,
                   help="Default CV2 force constant in kcal/mol/CV².")
    p.add_argument("--cv2-k-mode", choices=["fixed", "constant", "spacing", "adaptive"],
                   default="fixed")
    p.add_argument("--cv2-k-min", type=float, default=0.0,
                   help="Min adaptive CV2 k. 0 = no lower clamp.")
    p.add_argument("--cv2-k-max", type=float, default=500.0,
                   help="Max adaptive CV2 k.")
    p.add_argument("--cv2-adaptive-overlap-sigma", type=float, default=0.0,
                   help="Overlap-sigma for spacing-derived CV2 k. 0 = reuse cv1 value.")
    p.add_argument("--cv2-k-scale", type=float, default=1.0,
                   help="Extra multiplier for spacing/adaptive CV2 k.")
    # Custom secondary CV angles
    p.add_argument("--cv2-sigma-deg", type=float, default=35.0,
                   help="Angular width in degrees for phi/psi content score. "
                        "Only active when --cv2 custom; ignored for all other CV2 types.")
    p.add_argument("--cv2-phi0-deg", type=float, default=-60.0,
                   help="Custom CV2 phi target in degrees. Only active when --cv2 custom.")
    p.add_argument("--cv2-psi0-deg", type=float, default=-45.0,
                   help="Custom CV2 psi target in degrees. Only active when --cv2 custom.")

    # Debug
    p.add_argument("--self-test-primary-cv-force", action="store_true",
                   help="Test primary CV OpenMM force and exit.")


def _add_window_args(p: argparse.ArgumentParser) -> None:
    # Mode
    p.add_argument("--window-mode",
                   choices=["adaptive", "manual", "adaptive-feedback",
                             "adaptive-production", "double-adaptive",
                             "delaunay-feedback"],
                   default="adaptive")

    # Adaptive-feedback tuning
    p.add_argument("--aggressiveness",
                   choices=["conservative", "balanced", "aggressive", "very-aggressive"],
                   default="balanced", help="Window selection bias.")
    p.add_argument("--adaptive-rounds", type=int, default=3,
                   help="Pilot refinement rounds before final production.")
    p.add_argument("--pilot-fraction", type=float, default=0.05,
                   help="Pilot steps as fraction of production-steps per round.")
    p.add_argument("--pilot-steps", type=int, default=-1,
                   help="Absolute pilot steps per round. Overrides --pilot-fraction when > 0. "
                        "Use this when epoch/production steps are short and the fraction would give "
                        "too few steps for reliable overlap statistics.")
    p.add_argument("--validation-steps", type=int, default=-1,
                   help="Last pre-production validation steps. -1 = 1/10 of production-steps.")
    p.add_argument("--target-overlap", type=float, default=0.25,
                   help="Target neighboring-window CV histogram overlap.")
    p.add_argument("--adaptive-feedback-min-rounds", type=int, default=2,
                   help="Minimum adaptive-feedback rounds regardless of convergence criteria.")
    p.add_argument("--adaptive-feedback-reuse-shared-gamd", action=argparse.BooleanOptionalAction, default=True,
                   help="Calibrate the shared GaMD envelope once in the first adaptive-feedback "
                        "pilot round and reuse it for all later rounds and final production, instead "
                        "of re-running the expensive multiwindow GaMD recon every round. Default on.")
    p.add_argument("--pilot-min-cv1-coverage", type=float, default=0.60,
                   help="Required CV1 coverage fraction across achievable [lo,hi] range before "
                        "pilot is accepted.")
    p.add_argument("--pilot-min-cv2-coverage", type=float, default=0.70,
                   help="Required CV2 coverage fraction across secondary-CV center span before "
                        "pilot is accepted.")
    p.add_argument("--sparse-2d", action=argparse.BooleanOptionalAction, default=True,
                   help="Enable sparse local 2D midpoint patches during adaptive-feedback.")
    p.add_argument("--max-2d-patches", type=int, default=12,
                   help="Max sparse 2D patch windows per adaptive-feedback proposal.")
    p.add_argument("--max-total-windows", type=int, default=0,
                   help="Hard cap on TOTAL umbrella windows/replicas across the 2D CV1xCV2 grid "
                        "(0 = uncapped). Adaptive generation thins BOTH axes (axis_max = "
                        "floor(max_total / N_cv2_centers)) to honour this budget instead of growing "
                        "replicas unbounded. Requires max_total >= 2*N_cv2_centers; reduce "
                        "--cv2-centers if the cap is too small for the secondary axis.")
    p.add_argument("--min-total-windows", type=int, default=0,
                   help="Minimum TOTAL umbrella windows/replicas across the 2D grid (0 = no floor).")

    # Delaunay-feedback window placement
    p.add_argument("--delaunay-after-round", type=int, default=0,
                   help="Pilot round whose samples trigger Delaunay placement (0 = disabled; "
                        "auto-set to 1 when --window-mode delaunay-feedback).")
    p.add_argument("--delaunay-n-anchors", type=int, default=16,
                   help="Max KDE-peak basin anchors for Delaunay placement.")
    p.add_argument("--delaunay-dedup-radius", type=float, default=0.10,
                   help="Min normalized-space distance between Delaunay anchors.")
    p.add_argument("--delaunay-bridge-min-edge", type=float, default=0.20,
                   help="Min normalized Delaunay edge length to spawn a bridge window.")
    p.add_argument("--delaunay-density-floor-q", type=float, default=0.05,
                   help="Pilot KDE percentile below which candidate positions are rejected.")
    p.add_argument("--delaunay-kde-grid-res", type=int, default=64,
                   help="Resolution of the KDE evaluation grid for peak picking.")
    p.add_argument("--delaunay-circumcenter-probes", action=argparse.BooleanOptionalAction, default=False,
                   help="Enable optional circumcenter probe windows (density-filtered).")
    p.add_argument("--delaunay-circumcenter-max-radius", type=float, default=2.0,
                   help="Circumradius cutoff as multiple of median Delaunay edge length.")
    p.add_argument("--delaunay-k-sigma-factor", type=float, default=0.50,
                   help="sigma = factor × neighbor spacing for Delaunay force constant.")
    p.add_argument("--delaunay-min-pilot-samples", type=int, default=50,
                   help="Minimum pilot samples required to proceed with Delaunay placement.")
    p.add_argument("--delaunay-iterate", action=argparse.BooleanOptionalAction, default=None,
                   help="Re-run Delaunay on every pilot round >= delaunay-after-round until anchor "
                        "set stabilises. Default: on for delaunay-feedback, off otherwise.")
    p.add_argument("--delaunay-stability-tol", type=float, default=0.05,
                   help="Max normalised anchor shift below which Delaunay layout is considered "
                        "stable and the CSV is not rewritten (used with --delaunay-iterate).")
    p.add_argument("--delaunay-coverage-scaffold", action=argparse.BooleanOptionalAction, default=False,
                   help="Inject uniform grid anchors in [0,1]^2 cells that have no KDE-peak "
                        "anchor within dedup-radius, preventing blind spots in unvisited CV regions.")
    p.add_argument("--delaunay-coverage-grid-n", type=int, default=3,
                   help="N for the N×N coverage scaffold grid (default 3 → ≤9 scaffold points).")

    # Stuck-replica rescue (contact CV)
    p.add_argument("--cv1-stuck-reseed", action=argparse.BooleanOptionalAction, default=None,
                   help="Detect replicas pinned at CV1≈0 (fully-extended dead zone) and rescue by "
                        "copying positions from a non-stuck replica. Default: on when contact CV active, "
                        "off otherwise.")
    p.add_argument("--cv1-stuck-threshold", type=float, default=0.03,
                   help="Contact fraction below which a replica is considered potentially stuck (default 0.03).")
    p.add_argument("--cv1-stuck-detect-intervals", type=int, default=500,
                   help="Consecutive exchange intervals a replica must stay below --cv1-stuck-threshold "
                        "before rescue fires (default 500; at exchange_interval=100 steps = 50,000 steps = 200 ps).")
    p.add_argument("--rescue-in-final-production", action="store_true", default=False,
                   dest="rescue_in_final_production",
                   help="Allow contact-CV stuck rescue during final MBAR-quality production. "
                        "Disabled by default — rescue events are non-equilibrium and can "
                        "contaminate final equilibrium statistics.")

    # Region memory
    p.add_argument("--region-memory", action=argparse.BooleanOptionalAction, default=False,
                   help="Enable stateful region classification and memory.")
    p.add_argument("--region-memory-decay", type=float, default=0.5)
    p.add_argument("--region-probe-unvalidated", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--region-prune-overscanned", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--region-extend-undersampled", action=argparse.BooleanOptionalAction, default=True)

    # Manual / explicit window tables
    p.add_argument("--windows-a", nargs="+", type=float,
                   default=[5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 21],
                   help="Manual umbrella centers in Å (--window-mode manual).")
    p.add_argument("--window-k-kcal-a2", nargs="*", type=float, default=None)
    p.add_argument("--windows-2d-csv", default=None,
                   help="Explicit per-window 2D umbrella table CSV.")
    p.add_argument("--explicit-2d-window-schema", choices=["auto", "generic", "distance"],
                   default="auto")

    # Explicit 2D CSV column-name overrides (data-format, keep for compatibility)
    p.add_argument("--explicit-2d-primary-cv-mode-column", default="primary_cv_mode")
    p.add_argument("--explicit-2d-primary-center-column", default="primary_cv_center")
    p.add_argument("--explicit-2d-primary-k-column", default="primary_cv_k_kcal")
    p.add_argument("--explicit-2d-secondary-cv-mode-column", default="secondary_cv_mode")
    p.add_argument("--explicit-2d-secondary-center-column", default="secondary_cv_center")
    p.add_argument("--explicit-2d-secondary-k-column", default="secondary_cv_k_kcal_mol")

    # Epoch-based adaptive production (ap_* prefix)
    p.add_argument("--ap-epochs", type=int, default=3,
                   help="Max adaptive-production epochs before frozen final.")
    p.add_argument("--ap-epoch-steps", type=int, default=0,
                   help="Steps per epoch. 0 = auto (~1/20 of production-steps).")
    p.add_argument("--ap-final-steps", type=int, default=0,
                   help="Frozen final steps. 0 = reuse production-steps.")
    p.add_argument("--ap-epoch0-step-fraction", type=float, default=0.5,
                   help="Fraction of epoch 0's fair MD-pool share it actually consumes (default "
                        "0.5, since epoch 0 only needs a short look to bootstrap tICA/GaMD "
                        "recalibration). Unused budget is redistributed to later epochs, so with "
                        "the default and ap_epochs=2 the split is 1:3, not 1:1 -- set to 1.0 for "
                        "an even split across epochs.")
    p.add_argument("--ap-resume", action=argparse.BooleanOptionalAction, default=False,
                   help="Resume adaptive-production from state_registry.json or epoch checkpoints.")
    p.add_argument("--ap-include-epoch-samples", action=argparse.BooleanOptionalAction,
                   default=False, help="Include epoch samples in post-hoc MBAR.")
    p.add_argument("--ap-write-mbar-inputs", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--ap-run-mbar", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--ap-fes-bins", default="80,40",
                   help="Diagnostic histogram bins: primary[,secondary].")
    p.add_argument("--ap-seed-bank", action=argparse.BooleanOptionalAction, default=True,
                   help="Propagate seed bank between epochs.")
    p.add_argument("--ap-seed-bank-max", type=int, default=1,
                   help="Max seeds per state per epoch.")
    p.add_argument("--ap-min-exchange", type=float, default=0.08)
    p.add_argument("--ap-min-samples", type=int, default=50)
    p.add_argument("--ap-retire-min-samples", type=int, default=200,
                   help="[compat alias] Minimum saved sample rows per window per adaptive round "
                        "before retirement is considered. Prefer --ap-min-samples-per-window.")
    p.add_argument("--ap-min-samples-per-window", type=int, default=200,
                   help="Minimum saved sample rows per window per adaptive round (same unit as "
                        "ap_retire_min_samples: rows written by ParquetSampleWriter, one row per "
                        "distance_output_interval steps). Total steps >= value * distance_output_interval.")
    p.add_argument("--ap-final-min-samples-per-window", type=int, default=100,
                   help="Minimum saved sample rows per window in the final frozen production phase "
                        "(same unit as ap_min_samples_per_window).")
    p.add_argument("--ap-max-new-windows", type=int, default=4)
    p.add_argument("--ap-retire-converged", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--ap-gamd-boost-sd-warn", type=float, default=6.0)
    p.add_argument("--ap-write-reports", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--md-budget-ns", type=float, default=0.0,
                   help="Max aggregate MD simulation time in ns. 0 disables.")
    p.add_argument("--ap-final-pool-fraction", type=float, default=0.50)
    p.add_argument("--ap-min-final-pool-ns", type=float, default=0.0)
    p.add_argument("--ap-extend-rounds", dest="ap_extend_rounds", type=int, default=1,
                   help="Extension rounds / extra epochs for --extend (frozen: extension rounds; adaptive: extra epochs).")
    p.add_argument("--ap-extend-steps", dest="ap_extend_steps", type=int, default=0,
                   help="Steps per extension round for --extend frozen mode. 0 = reuse final-phase steps.")


def _add_us_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--us-starting-structure-mode",
                   choices=["pull", "npt", "equilibrated", "same", "none"], default="pull")
    p.add_argument("--us-pull-steps-per-window", type=int, default=5000)
    p.add_argument("--us-pull-k", type=float, default=5.0,
                   help="Pre-production pull force constant (kcal/mol/CV²).")
    p.add_argument("--us-pull-timestep-fs", type=float, default=0.0,
                   help="Pull timestep. 0 = min(timestep-fs, 2 fs).")
    p.add_argument("--us-pull-friction-per-ps", type=float, default=10.0)
    p.add_argument("--us-pull-minimize-iterations", type=int, default=1000)
    p.add_argument("--us-pull-ramp-stages", type=int, default=8,
                   help="Ramp stages for contact pulls and 2D secondary CV ramp.")
    p.add_argument("--us-2d-relax-mode",
                   choices=["auto", "staged", "off", "single", "ramp"], default="auto")
    p.add_argument("--us-2d-secondary-k-scale", type=float, default=1.0,
                   help="Multiply secondary CV k during 2D pull ramp.")
    p.add_argument("--us-allow-bad-windows", action=argparse.BooleanOptionalAction, default=False,
                   help="Start production even if US starting-structure quality-control flags windows "
                        "as 'bad' (start far enough from its production center that the full-strength "
                        "restraint force at step 0 can blow up the simulation). Default: refuse and "
                        "raise, listing the offending windows.")


def _add_seeding_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--seed-conformers-dir", type=Path, default=None,
                   help="GENPEPT output directory with final_survivor_seeds.csv.")
    p.add_argument("--seed-selection-mode",
                   choices=["auto", "active-cv", "primary", "distance"], default="auto")
    p.add_argument("--seed-cv2-weight", type=float, default=1.0,
                   help="Weight of CV2 in active-cv seed scoring.")
    p.add_argument("--seed-max-reuse-per-conformer", type=int, default=0)


def _add_genpept_prescan_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--genpept-prescan", action=argparse.BooleanOptionalAction, default=False,
                   help="Enable GAREUS-side GENPEPT prescan.")
    p.add_argument("--genpept-prescan-dir", type=Path, default=None)
    p.add_argument("--genpept-prescan-stages", nargs="*", default=None)
    p.add_argument("--genpept-prescan-frontier-stages", nargs="*", default=None)
    p.add_argument("--genpept-prescan-use-as-window-prior",
                   action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--genpept-prescan-use-as-seed-library",
                   action=argparse.BooleanOptionalAction, default=True)


def _add_gamd_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--production-steps", dest="production_steps",
                   type=int, default=500000, help="Production steps per replica.")
    # Default is the group-boost variant (nonbonded+dihedral), NOT `lower-dual`.
    # `lower-dual`'s total-boost portion reads the full potential, which includes
    # the umbrella restraint (gamd-openmm folds all forces into the boosted group),
    # making the boost window-dependent -> misspecifies MBAR + breaks REUS detailed
    # balance (audit finding C1). Group boost leaves the umbrella unboosted.
    p.add_argument("--gamd-boost-type", default="lower-dual-nonbonded-dihedral", choices=[
        "gamd-cmd-base", "lower-total", "upper-total", "lower-dihedral", "upper-dihedral",
        "lower-dual", "upper-dual", "lower-nonbonded", "upper-nonbonded",
        "lower-dual-nonbonded-dihedral", "upper-dual-nonbonded-dihedral",
    ])
    p.add_argument("--sigma0p", type=float, default=6.0,
                   help="Primary sigma0 in kcal/mol for GaMD.")
    p.add_argument("--sigma0d", type=float, default=6.0,
                   help="Secondary sigma0 in kcal/mol for dual-boost GaMD.")
    p.add_argument("--equil-steps", type=int, default=50000,
                   help="GaMD equilibration steps for boost calibration.")
    p.add_argument("--gamd-cmd-steps", type=int, default=250000,
                   help="GaMD CMD pre-equilibration steps for Vmax/Vmin statistics (longer → better calibration, lower anharmonicity).")
    p.add_argument("--gamd-averaging-window", type=int, default=5000)
    p.add_argument("--gamd-multiwindow-recon-prep-steps", type=int, default=2000,
                   help="Unrecorded relaxation steps per window before multi-window GaMD recon starts collecting statistics (applies to both the cMD seed and the boosted passes).")
    p.add_argument("--gamd-multiwindow-recon-cmd-steps", type=int, default=20000,
                   help="Per-window conventional-MD (boost OFF) seed-recon steps. Seeds the initial Vmax/Vmin/Vavg/sigmaV the GaMD boost needs before it can turn on. Keep short -- it only bootstraps the boosted passes.")
    p.add_argument("--gamd-multiwindow-recon-steps", type=int, default=20000,
                   help="Per-window BOOSTED recon steps per self-consistency iteration (boost ON, seeded from the current calibration). Measures sigmaV under the boost the run actually uses, so the frozen boost matches the boosted production ensemble.")
    p.add_argument("--gamd-recon-boosted-iters", type=int, default=4,
                   help="Max boosted self-consistency iterations for GaMD calibration. Each iteration re-measures sigmaV under the boost and recomputes k0/threshold; the loop stops early when every boost group's sigmaV settles within --gamd-recon-boosted-tol. 0 = legacy cMD-only calibration (boost measured OFF).")
    p.add_argument("--gamd-recon-boosted-tol", type=float, default=0.05,
                   help="Relative sigmaV convergence tolerance for the boosted GaMD calibration loop (0.05 = 5%%). The loop stops once each boost group's sigmaV changes by less than this between successive boosted iterations.")
    p.add_argument("--gamd-multiwindow-recon-report-interval", type=int, default=0,
                   help="Potential-energy sampling cadence (steps) during multi-window GaMD recon. 0 = auto (max(1, recon_steps // 200)).")
    p.add_argument("--exchange-interval", type=int, default=5000)
    p.add_argument("--exchange-mode",
                   choices=["neighbor", "random-pair", "all-pair-sweep", "gibbs-walk"],
                   default="neighbor")
    p.add_argument("--report-interval", type=int, default=5000)
    p.add_argument("--traj-interval", type=int, default=5000)
    p.add_argument("--traj-format", choices=["dcd", "xtc", "none"], default="dcd")
    p.add_argument("--traj-solute-only", action=argparse.BooleanOptionalAction, default=False,
                   help="Record only solute (non-water/ion) atoms in trajectories, and write a "
                        "companion solute_only.pdb. Shrinks dense-sampling trajectories ~100x for "
                        "a peptide in water. Analysis must use solute_only.pdb as topology.")
    p.add_argument("--adaptive-pilot-trajectories",
                   action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--adaptive-production-trajectories",
                   action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--randomize-replica-velocities", action="store_true")
    p.add_argument("--checkpoint-interval", type=int, default=50000)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--extend", action="store_true", default=False,
                   help="Extend a completed run instead of resuming an interrupted one.")
    p.add_argument("--extend-mode", dest="extend_mode",
                   choices=["auto", "regular", "adaptive", "frozen", "topup"],
                   default="auto",
                   help="Extension mode: auto detects last completed phase.")


def _add_output_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--progress-mode", choices=["none", "console", "jsonl", "both"], default="console")
    p.add_argument("--tui-mode", choices=["dashboard", "interactive", "line", "none"],
                   default="dashboard")
    p.add_argument("--distance-output-mode", choices=["none", "csv", "jsonl", "both"],
                   default="none")
    p.add_argument("--distance-output-interval", type=int, default=1000)
    p.add_argument("--parquet-flush-rows", type=int, default=200000,
                   help="Rows buffered per parquet writer before flushing a chunk to disk (default 200000). "
                        "Larger values → fewer, bigger parquet parts per run.")
    p.add_argument("--flush-every-log", action=argparse.BooleanOptionalAction, default=False,
                   help="Flush parquet writers after every logging step (creates many tiny chunks). "
                        "Default False: rely on --parquet-flush-rows threshold instead.")
    p.add_argument("--write-analysis-chunks", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--analysis-write-consolidated-npz",
                   action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--analysis-chunk-size", type=int, default=10000)
    p.add_argument("--analysis-chunk-dir", default="analysis_chunks")
    p.add_argument("--sample-potential-energy",
                   action=argparse.BooleanOptionalAction, default=True)


def _add_tica_args(p: argparse.ArgumentParser) -> None:
    """Args for inter-epoch tICA CVaux (all default to OFF; only active when tica_obs_interval > 0)."""
    p.add_argument("--tica-obs-interval", type=int, default=0,
                   help="Record backbone dihedral features every N samples for tICA refitting. "
                        "0 = disabled (default).")
    p.add_argument("--tica-update-after-epochs", type=int, nargs="*", default=None,
                   metavar="EPOCH",
                   help="Epoch numbers after which to refit tIC1 and update the secondary CV. "
                        "Epoch indexing is 0-based. Default: None (disabled).")
    p.add_argument("--tica-lag-frames", type=int, default=50,
                   help="Lag in frames for tICA generalised eigenvalue problem (default 50).")
    p.add_argument("--tica-state-file", default="",
                   help="Path to a tICA state JSON file (TICAResult) for the tica-linear secondary CV. "
                        "Set automatically by the adaptive production loop; leave empty for first epoch.")
    p.add_argument("--tica-min-eigenvalue", type=float, default=0.0,
                   help="Skip tICA update if the fitted eigenvalue is below this threshold (default 0.0 = always update).")
    p.add_argument("--tica-epochs-per-cycle", type=int, default=0, dest="tica_epochs_per_cycle",
                   help="Auto-trigger tICA refit every N completed epochs (0 = off). "
                        "Acts as an OR condition with --tica-update-after-epochs. "
                        "Note: each refit bumps tica_cv_version so prior-version epochs "
                        "are excluded from cross-epoch MBAR pooling.")
    p.add_argument("--tica-switch-cv2", action="store_true", default=False, dest="tica_switch_cv2",
                   help="After a successful tICA fit, permanently switch the secondary CV type to "
                        "tica-linear for all remaining epochs and final production. "
                        "The switch is one-shot: once triggered it is not re-triggered. "
                        "Requires tica_obs_interval > 0. k bounds default to "
                        "--tica-linear-k-min/max; supply explicit values if the defaults "
                        "do not fit your system.")
    p.add_argument("--tica-linear-k-min", type=float, default=5.0, dest="tica_linear_k_min",
                   help="cv2_k_min applied immediately after the tica-linear CV2 switch "
                        "(default 5.0 kcal/mol). tICA projects onto [-6, 6]; softer than the "
                        "rama-map default (20 kcal/mol) to avoid over-constraining the wider range.")
    p.add_argument("--tica-linear-k-max", type=float, default=50.0, dest="tica_linear_k_max",
                   help="cv2_k_max applied immediately after the tica-linear CV2 switch "
                        "(default 50.0 kcal/mol).")
    p.add_argument("--tica-component-count", type=int, default=1, dest="tica_component_count",
                   help="Number of top tICA modes to combine (eigenvalue-weighted, "
                        "1..N by slowness rank) into the single cv2=tica-linear direction "
                        "each time tICA is (re)fit. 1 means tIC1 alone (default, unchanged "
                        "behavior).")
    p.add_argument("--bootstrap-torsion-source", default="seeds",
                   choices=["seeds"],
                   help="Source ensemble for cv2=torsion-pca bootstrap model.")
    p.add_argument("--bootstrap-torsion-residualize-against-cv1",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="For cv2=torsion-pca, regress seed torsion features against seed CV1 before PCA.")
    p.add_argument("--bootstrap-torsion-component", type=int, default=5,
                   help="Number of top PCA components to combine (variance-weighted, "
                        "1..N by variance rank) into the single cv2=torsion-pca direction. "
                        "1 means PC1 alone.")
    p.add_argument("--bootstrap-torsion-min-seed-count", type=int, default=20,
                   help="Minimum usable GENPEPT survivors needed for cv2=torsion-pca.")
    p.add_argument("--bootstrap-torsion-state-file", default="",
                   help="Bootstrap torsion PCA state JSON. Empty means <out>/tica/bootstrap_torsion_cv.json.")


def _add_platform_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--platform", default="auto")
    p.add_argument("--precision", default="mixed")
    p.add_argument("--device-index", default="0")
    p.add_argument("--replica-device-mode",
                   choices=["auto", "round-robin", "single-context-split", "manual"],
                   default="auto")
    p.add_argument("--replica-device-map", default="")
    p.add_argument("--cuda-use-cpu-pme", choices=["auto", "true", "false"], default="auto")
    p.add_argument("--cuda-use-blocking-sync", choices=["auto", "true", "false"], default="auto")
    p.add_argument("--cuda-deterministic-forces", choices=["auto", "true", "false"], default="auto")
    p.add_argument("--cuda-disable-pme-stream", choices=["auto", "true", "false"], default="auto",
                   help="OpenMM 8.3+: disable the dedicated CUDA PME stream. 'false' explicitly "
                        "keeps the stream enabled so PME overlaps direct-space computation on GPU. "
                        "'auto' lets OpenMM choose the default for the installed version.")
    p.add_argument("--cuda-mps", action="store_true", default=False,
                   help="Apply CUDA MPS-optimal settings as soft defaults: UseBlockingSync=false, "
                        "DeterministicForces=false. Individual --cuda-* flags override these. "
                        "Requires CUDA MPS running on the node (nvidia-cuda-mps-control -d).")
    p.add_argument("--platform-temp-directory", default="")
    p.add_argument("--cpu-threads", type=int, default=1,
                   help="Threads per replica context (platform CPU only). "
                        "Overridden by --cpu-budget when set.")
    p.add_argument("--cpu-budget", type=int, default=0,
                   help="Total CPU cores to distribute evenly across replicas "
                        "(floor(budget/n_replicas), min 1). 0 = use --cpu-threads directly.")
    p.add_argument("--max-cpu-per-replica", type=int, default=0,
                   help="Cap per-replica CPU thread count regardless of --cpu-budget. 0 = no cap.")
    p.add_argument("--max-replicas", type=int, default=0,
                   help="Hard cap on the number of replicas/windows for any run mode. "
                        "Windows are truncated to this count after generation/loading. 0 = no cap.")
    p.add_argument("--setup-platform", default="")
    p.add_argument("--setup-precision", default="")
    p.add_argument("--setup-device-index", default="")
    p.add_argument("--setup-cpu-threads", type=int, default=0)
    p.add_argument("--us-pull-workers", default="auto")
    p.add_argument("--scratchdir", default="")


# ---------------------------------------------------------------------------
# Post-parse resolution helpers
# ---------------------------------------------------------------------------

def _resolve_cv_aliases(args: argparse.Namespace, argv_list: list) -> None:
    # cv1 → primary_cv
    if getattr(args, "cv1", None) not in (None, ""):
        args.primary_cv = ("nonlocal-contacts"
                           if str(args.cv1).strip().lower().replace("_", "-")
                           in {"contacts", "nonlocal-contacts"}
                           else "distance")

    # cv2 is canonical secondary CV attr; secondary_cv_mode() reads args.secondary_cv
    _cv2_val = getattr(args, "cv2", None)
    _cv2_requested = _cv2_val not in (None, "")
    if _cv2_requested:
        args.secondary_cv = str(_cv2_val)
    else:
        args.secondary_cv = getattr(args, "secondary_cv", "none")

    args.primary_cv = primary_cv_mode(args)
    args.secondary_cv = secondary_cv_mode(args)
    args.cv2 = args.secondary_cv  # keep cv2 in sync

    # Auto-insert cv2_centers when cv2 active but no centers provided.
    # torsion-pca centers are data-derived after seed PCA is fitted in run_gareus.
    if _cv2_requested and str(args.cv2) != "none" and not getattr(args, "cv2_centers", None):
        if args.cv2 == "rama-map":
            args.cv2_centers = [-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0]
        elif args.cv2 in {"alpha-coil-beta", "acb"}:
            args.cv2_centers = [-0.8, 0.0, 0.8]
        elif args.cv2 == "torsion-pca":
            args.cv2_centers = None
        else:
            args.cv2_centers = [0.2, 0.5, 0.8]
        args._cv2_auto_centers = True
    else:
        args._cv2_auto_centers = False


def _normalize_run_mode(args: argparse.Namespace, argv_list: list) -> None:
    _rm = str(getattr(args, "run_mode", "gamd") or "gamd").strip().lower().replace("_", "-")
    if _rm not in {"cmd", "hmr-cmd", "gamd", "hmr-gamd"}:
        raise ValueError("--run-mode must be cmd, hmr-cmd, gamd, or hmr-gamd")
    args.run_mode = _rm
    if args.run_mode in {"hmr-cmd", "hmr-gamd"}:
        args.hmr = True
        _specified_ts_cli = any(
            str(t) == "--timestep-fs" or str(t).startswith("--timestep-fs=")
            for t in argv_list
        )
        _specified_ts_cfg = "timestep_fs" in (getattr(args, "_config_values", {}) or {})
        if not _specified_ts_cli and not _specified_ts_cfg:
            args.timestep_fs = 4.0
            args._run_mode_auto_timestep_fs = True
        else:
            args._run_mode_auto_timestep_fs = False
    else:
        args.hmr = False
        args._run_mode_auto_timestep_fs = False


def _validate_contact_args(args: argparse.Namespace) -> None:
    """Validate contact-CV args. Runs after compat shims so old attr names are available."""
    if args.primary_cv != "nonlocal-contacts":
        return
    if args.contact_scheme not in {"atom-pairs", "residue-balanced", "ca-pairs"}:
        raise ValueError("--contact-scheme must be atom-pairs, residue-balanced, or ca-pairs")
    if not math.isfinite(float(getattr(args, "contact_r0_a", 4.5))) or float(getattr(args, "contact_r0_a", 4.5)) <= 0:
        raise ValueError("--contact-r0-a must be positive and finite")
    if not math.isfinite(float(getattr(args, "contact_beta_a_inv", 6.0))) or float(getattr(args, "contact_beta_a_inv", 6.0)) <= 0:
        raise ValueError("--contact-beta-a-inv must be positive and finite")
    if int(getattr(args, "contact_min_sequence_separation", 4) or 4) < 1:
        raise ValueError("--contact-min-sequence-separation must be >= 1")
    if bool(getattr(args, "contact_normalize", True)):
        cmin = float(getattr(args, "contact_adaptive_min", 0.0) or 0.0)
        cmax = float(getattr(args, "contact_adaptive_max", 0.80) or 0.80)
        if not (math.isfinite(cmin) and math.isfinite(cmax)) or cmax <= cmin:
            raise ValueError("cv1-range-max must be finite and > cv1-range-min for contacts")
        if cmin < -1e-8 or cmax > 1.0 + 1e-8:
            raise ValueError("Normalized contact adaptive bounds should be within [0, 1]")
    if float(getattr(args, "contact_adaptive_target_spacing", 0.15) or 0.15) <= 0:
        raise ValueError("--cv1-target-spacing must be positive")
    if int(getattr(args, "cv1_boundary_pull_steps", 5000) or 0) < 0:
        raise ValueError("--cv1-boundary-pull-steps must be >= 0")
    if (
        not bool(getattr(args, "resume", False))
        and not bool(getattr(args, "self_test_primary_cv_force", False))
        and str(getattr(args, "window_mode", "manual")) == "manual"
        and (getattr(args, "contact_centers", None) is None or len(args.contact_centers) == 0)
    ):
        raise ValueError("contacts + --window-mode manual requires --contact-centers")


# The upstream `gamd` package's integrator_factory.get_integrator only forwards
# sigma0d to the dual-boost integrators (lower-dual, upper-dual, and their
# nonbonded-dihedral variants). Every single-boost type is calibrated entirely
# from sigma0p (see create_lower_dihedral_boost_integrator etc. in
# gamd/integrator_factory.py, which accept a single `sigma0` positional filled
# from sigma0p) and silently ignores sigma0d.
_SINGLE_BOOST_GAMD_TYPES = frozenset({
    "gamd-cmd-base", "lower-total", "upper-total",
    "lower-dihedral", "upper-dihedral",
    "lower-nonbonded", "upper-nonbonded",
})


def _validate_gamd_args(args: argparse.Namespace) -> None:
    """Warn when --sigma0d is set but the selected GaMD boost type can't use it."""
    boost_type = str(getattr(args, "gamd_boost_type", "") or "")
    if boost_type not in _SINGLE_BOOST_GAMD_TYPES:
        return
    sigma0d = float(getattr(args, "sigma0d_kcal_mol", None) if getattr(args, "sigma0d_kcal_mol", None) is not None else getattr(args, "sigma0d", 6.0))
    if abs(sigma0d - 6.0) > 1e-9:
        print(
            f"WARNING: --sigma0d={sigma0d:.3f} kcal/mol is set, but --gamd-boost-type={boost_type!r} "
            "is a single-boost mode -- sigma0d is silently ignored for this boost type (only "
            "dual-boost modes use it). --sigma0p is the parameter that actually governs GaMD "
            "calibration here; set that instead.",
            flush=True,
        )


def _shim_cv(args: argparse.Namespace) -> None:
    """CV2 (secondary CV) schema-v2 -> legacy attr names."""
    args.secondary_cv = args.cv2
    cv2_centers = getattr(args, "cv2_centers", None)
    args.secondary_cv_centers = cv2_centers
    args.secondary_cv_center = (cv2_centers[0] if cv2_centers and len(cv2_centers) == 1 else None)
    args.secondary_cv_k_kcal = args.cv2_k_default
    args.secondary_cv_k_mode = args.cv2_k_mode
    args.secondary_cv_adaptive_min_k_kcal = args.cv2_k_min
    args.secondary_cv_adaptive_max_k_kcal = args.cv2_k_max
    args.secondary_cv_adaptive_overlap_sigma = args.cv2_adaptive_overlap_sigma
    args.secondary_cv_adaptive_k_scale = args.cv2_k_scale
    args.secondary_cv_sigma_deg = args.cv2_sigma_deg
    args.secondary_cv_phi0_deg = args.cv2_phi0_deg
    args.secondary_cv_psi0_deg = args.cv2_psi0_deg
    args.secondary_cv_adaptive_min_sigma = 0.02
    args.secondary_cv_force_group = 29


def _shim_cv1_legacy_names(args: argparse.Namespace) -> None:
    """CV1 (primary CV) adaptive-window schema-v2 -> legacy distance/contact attr names."""
    _cv1_range_min = args.cv1_range_min
    _cv1_range_max = args.cv1_range_max
    _cv1_spacing = args.cv1_target_spacing
    _cv1_k_def = args.cv1_k_default
    _cv1_k_min = args.cv1_k_min
    _cv1_k_max = args.cv1_k_max
    _cv1_k_mode = args.cv1_k_mode
    _cv1_sigma = args.cv1_adaptive_overlap_sigma

    # distance-mode legacy attrs
    args.adaptive_window_min_a = _cv1_range_min
    args.adaptive_window_max_a = _cv1_range_max
    args.adaptive_target_spacing_a = _cv1_spacing if _cv1_spacing > 0 else 2.0
    args.default_window_k_kcal_a2 = _cv1_k_def if _cv1_k_def > 0 else 1.0
    args.adaptive_min_k_kcal_a2 = _cv1_k_min if _cv1_k_min > 0 else 0.05
    args.adaptive_max_k_kcal_a2 = _cv1_k_max if _cv1_k_max > 0 else 20.0
    args.adaptive_k_mode = _cv1_k_mode
    args.adaptive_overlap_sigma = _cv1_sigma
    args.cv1_boundary_pull_steps = args.cv1_boundary_pull_steps
    args.cv1_boundary_pull_k = args.cv1_boundary_pull_k
    args.cv1_boundary_pull_margin = args.cv1_boundary_pull_margin
    args.cv1_boundary_pull_timestep_fs = 1.0

    # contact-mode legacy attrs
    args.contact_adaptive_min = _cv1_range_min
    args.contact_adaptive_max = _cv1_range_max if _cv1_range_max > 0 else 0.80
    args.contact_adaptive_target_spacing = _cv1_spacing if _cv1_spacing > 0 else 0.15
    args.contact_adaptive_default_k_kcal = _cv1_k_def if _cv1_k_def > 0 else 25.0
    args.contact_adaptive_min_k_kcal = _cv1_k_min if _cv1_k_min > 0 else 5.0
    args.contact_adaptive_max_k_kcal = _cv1_k_max if _cv1_k_max > 0 else 120.0
    args.contact_adaptive_k_mode = _cv1_k_mode
    args.contact_adaptive_overlap_sigma = _cv1_sigma
    # boundary pull replaces old autocalibration — no compat shims needed
    args.contact_frontier_enabled = args.cv1_frontier
    args.contact_frontier_probe_count = args.cv1_frontier_probe_count

    # Dropped contact adaptive tuning → hardcoded defaults
    args.contact_adaptive_min_sigma = 0.02
    args.contact_adaptive_k_scale = 1.0
    args.contact_adaptive_hit_radius = 0.08
    args.contact_adaptive_tight_hit_radius = 0.04
    args.contact_adaptive_max_center_shift = 0.08
    args.contact_adaptive_min_new_spacing = 0.04
    args.contact_adaptive_coverage_gap_width = 0.10
    args.contact_pair_warning_threshold = 5000
    args.contact_frontier_percentile = 99.0
    args.contact_frontier_margin = 0.02
    args.contact_frontier_min_span = 0.10
    args.contact_frontier_probe_spacing = 0.05
    args.contact_frontier_max_probe_offset = 0.15
    args.contact_disable_gapfill_above_unvalidated_frontier = True
    args.contact_frontier_confirm_rounds = 2
    args.contact_frontier_min_hit_fraction = 0.02
    args.contact_frontier_unreachable_deficit = 0.08
    args.contact_frontier_by_secondary_slice = True

    # Dropped legacy per-axis window count bounds (still used as fallbacks when no
    # total-window budget is set).
    args.adaptive_min_windows = 4
    args.adaptive_max_windows = 32
    args.contact_adaptive_min_windows = 4
    args.contact_adaptive_max_windows = 12
    # Total-window/replica budget — user-settable via --max-total-windows /
    # --min-total-windows (mapped onto both the generic and contact-CV budget
    # attrs that gareus.windows consumes). 0 = uncapped. This lets adaptive
    # generation honour an overall replica budget instead of growing unbounded.
    _min_total_windows = int(getattr(args, "min_total_windows", 0) or 0)
    _max_total_windows = int(getattr(args, "max_total_windows", 0) or 0)
    args.adaptive_min_total_windows = _min_total_windows
    args.adaptive_max_total_windows = _max_total_windows
    args.contact_adaptive_min_total_windows = _min_total_windows
    args.contact_adaptive_max_total_windows = _max_total_windows
    args.n_windows = 0
    args.adaptive_compact_floor_a = 3.5
    args.adaptive_extension_fraction = 0.95
    args.umbrella_force_group = 31
    args.adaptive_secondary_cv = "auto"


def _shim_us_pulling(args: argparse.Namespace) -> None:
    """Umbrella-sampling pull-ramp schema-v2 -> legacy attr names."""
    args.us_pull_k_kcal_a2 = args.us_pull_k
    args.contact_us_pull_k_kcal = args.us_pull_k
    args.contact_us_pull_max_k_kcal = max(args.us_pull_k * 2.0, 20.0)
    args.contact_us_pull_ramp_stages = args.us_pull_ramp_stages
    args.us_2d_start_relax_mode = args.us_2d_relax_mode
    args.us_2d_start_secondary_k_pull_scale = args.us_2d_secondary_k_scale
    args.us_2d_start_secondary_ramp_stages = args.us_pull_ramp_stages
    # Dropped US pull tuning
    args.contact_us_pull_timestep_fs = 1.0
    args.contact_us_pull_safe_chunk_steps = 100
    args.contact_us_pull_min_friction_per_ps = 20.0
    args.contact_us_pull_minimize_first_ramp = True
    args.us_2d_start_distance_fraction = 0.50
    args.us_2d_start_minimize_each_ramp = False
    args.us_2d_start_secondary_warn_delta = 0.35
    args.us_2d_start_secondary_warn_bias_kcal = 1.0
    args.us_2d_start_secondary_bad_bias_kcal = 5.0


def _shim_windows(args: argparse.Namespace) -> None:
    """Adaptive-window-placement schema-v2 -> legacy attr names."""
    args.adaptive_window_aggressiveness = args.aggressiveness
    args.adaptive_feedback_rounds = args.adaptive_rounds
    args.adaptive_feedback_pilot_fraction = args.pilot_fraction
    args.adaptive_feedback_pilot_steps = args.pilot_steps
    args.adaptive_feedback_validation_steps = args.validation_steps
    args.adaptive_feedback_target_overlap = args.target_overlap
    args.sparse_2d_patches_enabled = args.sparse_2d
    args.contact_sparse_2d_patches_enabled = args.sparse_2d
    args.adaptive_2d_max_local_patches = args.max_2d_patches
    args.adaptive_feedback_region_state_enabled = args.region_memory
    args.adaptive_feedback_region_memory_decay = args.region_memory_decay
    args.adaptive_feedback_probe_unvalidated_regions = args.region_probe_unvalidated
    args.adaptive_feedback_prune_overscanned_regions = args.region_prune_overscanned
    args.adaptive_feedback_extend_undersampled_regions = args.region_extend_undersampled
    # Dropped window tuning
    args.adaptive_window_count_weight = -1.0
    args.adaptive_feedback_bootstrap_samples = 50
    args.adaptive_feedback_validation_fraction = 0.10
    args.adaptive_feedback_min_effective_samples = 50
    args.adaptive_2d_min_edge_samples = 5
    args.explicit_2d_exchange_neighbor_k = 2
    args.explicit_2d_exchange_radius = 1.65
    args.explicit_2d_exchange_slots = 4


def _shim_seeding(args: argparse.Namespace) -> None:
    """Seeding schema-v2 -> legacy attr names."""
    args.seed_secondary_weight = args.seed_cv2_weight


def _shim_genpept_prescan(args: argparse.Namespace) -> None:
    """GENPEPT-prescan (live feature) schema-v2 -> legacy attr names."""
    args.genpept_prescan_enabled = args.genpept_prescan
    # Dropped prescan tuning
    args.genpept_prescan_rescore_active_cvs = True
    args.genpept_prescan_contact_bin_width = 0.025
    args.genpept_prescan_rama_bin_width = 0.25
    args.genpept_prescan_min_hits_per_bin = 2
    args.genpept_prescan_absence_means_unknown = True
    args.genpept_prescan_output_prefix = "genpept_prescan"
    args.genpept_prescan_write_maps = True
    args.genpept_prescan_write_seed_assignments = True


def _shim_genpept_prior_disabled(args: argparse.Namespace) -> None:
    """GENPEPT-prior window-seeding pipeline (gareus/genpept_window_prior.py) — deliberately
    retired, not exposed via any CLI/YAML knob. ``genpept_prior_enabled`` is unconditionally
    False here on purpose: the pipeline is still imported and called by
    gareus/adaptive_feedback.py (it takes the disabled early-return branch every time), it is
    not dead code, just permanently gated off. Flip ``genpept_prior_enabled`` back to a real
    CLI flag only as a deliberate product decision to revive the feature, not as a bugfix.
    """
    args.genpept_prior_enabled = False
    args.genpept_prior_dir = None
    args.genpept_prior_stages = None
    args.genpept_prior_max_structures = 50000
    args.genpept_prior_min_points = 10
    args.genpept_prior_max_windows = 48
    args.genpept_prior_bins = "24,12"
    args.genpept_prior_min_hits_per_bin = 2
    args.genpept_prior_basin_windows = 0
    args.genpept_prior_bridge_windows = 0
    args.genpept_prior_frontier_windows = 0
    args.genpept_prior_probe_windows = 0
    args.genpept_prior_absence_means_unknown = True
    args.genpept_prior_snap_secondary_centers = True
    args.genpept_prior_required = False


def _shim_gamd(args: argparse.Namespace) -> None:
    """GaMD schema-v2 -> legacy attr names."""
    args.gamd_production_steps = args.production_steps
    args.sigma0p_kcal_mol = args.sigma0p
    args.sigma0d_kcal_mol = args.sigma0d
    args.gamd_equil_steps = args.equil_steps
    # Dropped GaMD prep steps
    args.gamd_cmd_prep_steps = 5000
    args.gamd_equil_prep_steps = 5000
    # Dropped exchange tuning
    args.exchange_random_pairs = 0
    args.exchange_max_pairs_per_interval = 0
    # Dropped production tuning
    args.production_probe_steps = 0
    args.production_probe_warn_only = False
    args.production_safe_chunk_steps = 0
    args.production_nan_diagnostics = True
    args.shared_gamd_copy_strict = False
    args.shared_gamd_setup_dir = ""
    args.shared_gamd_export_dir = ""


def _shim_adaptive_production(args: argparse.Namespace) -> None:
    """Adaptive-production schema-v2 -> legacy attr names."""
    args.adaptive_production_epochs = args.ap_epochs
    args.adaptive_production_epoch_steps = args.ap_epoch_steps
    args.adaptive_production_final_steps = args.ap_final_steps
    args.adaptive_production_resume = args.ap_resume
    args.adaptive_production_use_epoch_samples_for_mbar = args.ap_include_epoch_samples
    args.adaptive_production_write_union_mbar_inputs = args.ap_write_mbar_inputs
    args.adaptive_production_run_union_mbar_analysis = args.ap_run_mbar
    args.adaptive_production_union_fes_bins = args.ap_fes_bins
    args.adaptive_production_propagate_seed_bank = args.ap_seed_bank
    args.adaptive_production_seed_bank_max_per_state = args.ap_seed_bank_max
    args.adaptive_production_target_overlap = args.target_overlap
    args.adaptive_production_min_exchange = args.ap_min_exchange
    args.adaptive_production_min_samples = args.ap_min_samples
    # ap_min_samples_per_window is the preferred name; ap_retire_min_samples is the legacy alias.
    # Start from the new preferred key, then let the legacy alias override when it was set
    # explicitly (i.e. differs from its argparse default of 200).
    args.adaptive_production_retire_min_samples = args.ap_min_samples_per_window
    if args.ap_retire_min_samples != 200:
        # Legacy alias was explicitly set — honour it for backward compatibility.
        warnings.warn(
            "ap_retire_min_samples is deprecated; use ap_min_samples_per_window instead.",
            DeprecationWarning, stacklevel=2,
        )
        args.adaptive_production_retire_min_samples = args.ap_retire_min_samples
    args.adaptive_production_final_min_samples_per_state = args.ap_final_min_samples_per_window
    args.adaptive_production_max_new_windows_per_epoch = args.ap_max_new_windows
    args.adaptive_production_retire_converged = args.ap_retire_converged
    args.adaptive_production_max_gamd_boost_sd_kcal_mol = args.ap_gamd_boost_sd_warn
    args.adaptive_production_write_action_reports = args.ap_write_reports
    args.adaptive_production_total_md_pool_ns = args.md_budget_ns
    args.adaptive_production_final_pool_fraction = args.ap_final_pool_fraction
    args.adaptive_production_min_final_pool_ns = args.ap_min_final_pool_ns
    # Dropped adaptive production tuning → hardcoded defaults
    args.adaptive_production_allocation_scheduler = True
    args.adaptive_production_global_shared_gamd = True
    args.adaptive_production_context_reuse = False
    args.adaptive_production_context_reuse_require = False
    args.adaptive_production_context_reuse_mode = "off"
    args.adaptive_production_epoch_step_budget = 0
    args.adaptive_production_min_state_steps = 0
    args.adaptive_production_max_state_steps = 0
    args.adaptive_production_new_state_steps = 0
    args.adaptive_production_final_allocation_scheduler = True
    args.adaptive_production_final_step_budget = 0
    args.adaptive_production_state_aware_seed_filtering = True
    args.adaptive_production_scheduled_final_segments = True
    args.adaptive_production_convergence_min_samples_per_state = 50
    args.adaptive_production_convergence_max_weak_edges = 0
    args.adaptive_production_convergence_allow_extend_actions = True
    args.adaptive_production_require_convergence_before_final = False
    args.adaptive_production_pool_hard_stop = True
    # Epoch 0 also bootstraps tICA from real GaMD-boosted sampling, so it doubles
    # as the cheapest place to recalibrate the shared GaMD envelope for epoch 1+
    # (see gareus/adaptive_production.py:_maybe_recalibrate_gamd_boost) and can run
    # at half length -- the unused MD-pool budget is redistributed to later epochs.
    args.adaptive_production_gamd_recalibrate_after_epoch0 = True
    args.adaptive_production_epoch0_step_fraction = args.ap_epoch0_step_fraction
    args.adaptive_production_final_connectivity_required = True
    # final_min_samples_per_state is set earlier from ap_final_min_samples_per_window (default 100).
    args.adaptive_production_quality_min_primary_coverage_fraction = 0.25
    args.adaptive_production_quality_hard_fail = False
    args.adaptive_production_final_quality_extension_rounds = 0
    args.adaptive_production_final_quality_extension_steps = 0
    # extend support
    args.extend = bool(getattr(args, "extend", False))
    args.extend_mode = str(getattr(args, "extend_mode", "auto"))
    args.adaptive_production_topup_only = False


def _shim_output(args: argparse.Namespace) -> None:
    """Output/logging/progress schema-v2 -> legacy attr names."""
    args.color = "auto"
    args.progress_jsonl = "progress.jsonl"
    args.progress_update_interval_sec = 0.25
    args.progress_bar_width = 36
    args.tui_clear_mode = "always"
    args.dashboard_density = "auto"
    args.dashboard_wide_threshold = 132
    args.dashboard_min_panel_width = 30
    args.dashboard_max_height = 0
    args.dashboard_render_interval_sec = 0.0
    args.dashboard_panels = "normal"
    args.dashboard_heavy_panels_every = 1
    args.distance_csv = "distances.csv"
    args.distance_jsonl = "distances.jsonl"
    args.no_distance_gui_events = False
    args.distance_ascii_mode = "hist3d"
    args.distance_ascii_width = 54
    args.distance_ascii_max_replicas = 32
    args.distance_history_limit = 4000
    # flush_every_log is a real, documented user flag (default False: "rely on
    # --parquet-flush-rows threshold instead", see --flush-every-log's own help text) —
    # it must NOT be forced here. Leave args.flush_every_log exactly as argparse set it.
    args.csv_flush_rows = 1000
    args.jsonl_flush_rows = 500
    args.analysis_consolidated_max_elements = 100_000_000
    args.analysis_array_dtype = "float64"
    args.analysis_npz_compressed = False
    args.write_full_bias_csv_vectors = False
    args.write_gamd_globals_json = False


def _shim_system(args: argparse.Namespace) -> None:
    """System-setup schema-v2 -> legacy attr names."""
    args.initial_phi = -60.0
    args.initial_psi = -45.0
    args.ph = 7.0
    args.hydrogen_mass_amu = 3.024 if getattr(args, "hmr", False) else 0.0
    args.production_barostat_frequency = 0
    args.equil_safe_chunk_steps = 100
    args.nvt_start_temperature_k = 50.0
    args.npt_final_timestep_fs = 0.0


def _shim_misc(args: argparse.Namespace) -> None:
    """Miscellaneous schema-v2 -> legacy attr names."""
    args.double_adaptive = (args.window_mode == "double-adaptive")
    args.cv_mode = "terminal-ca"
    # primary_cv alias kept for cv.py compatibility
    if not hasattr(args, "primary_cv"):
        args.primary_cv = "distance"


def _validate_config_sanity(args: argparse.Namespace) -> None:
    """Cross-flag sanity warnings/errors, run after all v2-compat shims are applied."""
    # R5: cv1_k_default silently ignored in non-fixed mode
    if getattr(args, "cv1_k_mode", "spacing") not in ("fixed", "constant") and float(getattr(args, "cv1_k_default", 0.0) or 0.0) != 0.0:
        warnings.warn(
            f"cv1_k_default={args.cv1_k_default} is set but cv1_k_mode='{args.cv1_k_mode}' — "
            "cv1_k_default is only used in 'fixed' mode. Use cv1_k_min/cv1_k_max for spacing mode.",
            UserWarning, stacklevel=2,
        )
    # R6: cv2_k_default silently ignored when cv2_k_mode is not fixed/constant
    _cv2_k_mode = getattr(args, "cv2_k_mode", "fixed")
    if _cv2_k_mode not in ("fixed", "constant") and float(getattr(args, "cv2_k_default", 50.0) or 50.0) != 50.0:
        warnings.warn(
            f"cv2_k_default={args.cv2_k_default} is set but cv2_k_mode='{_cv2_k_mode}' — "
            "cv2_k_default is only used in 'fixed' mode. Use cv2_k_min/cv2_k_max for other modes.",
            UserWarning, stacklevel=2,
        )
    # R8: tICA trigger conflict
    _tica_cycle = int(getattr(args, "tica_epochs_per_cycle", 0) or 0)
    _tica_explicit = getattr(args, "tica_update_after_epochs", None)
    if _tica_cycle > 0 and _tica_explicit is not None:
        raise ValueError(
            "tica_epochs_per_cycle and tica_update_after_epochs are mutually exclusive. "
            f"Got tica_epochs_per_cycle={_tica_cycle} and tica_update_after_epochs={_tica_explicit}. "
            "Use one trigger mechanism only."
        )
    # R9: genpept_prescan_dir fallback
    if getattr(args, "genpept_prescan", False) and getattr(args, "genpept_prescan_dir", None) is None:
        _fallback = getattr(args, "seed_conformers_dir", None)
        if _fallback is not None:
            args.genpept_prescan_dir = _fallback


def _apply_v2_compat_shims(args: argparse.Namespace) -> None:
    """Map schema-v2 attr names to legacy internal names expected by consumer modules.

    This lets the public CLI/YAML API be clean and minimal while all downstream
    code continues to work without changes.  Consumer modules can be updated to
    the new names incrementally.

    Split into one function per subsystem so each override is a named, visible
    thing instead of a line buried in one 340-line function — that's what let
    ``args.flush_every_log = True`` silently defeat a real, documented user flag
    for as long as it did. Order matters (later sections may read attrs set by
    earlier ones) and must match the original single-function order exactly.
    """
    _shim_cv(args)
    _shim_cv1_legacy_names(args)
    _shim_us_pulling(args)
    _shim_windows(args)
    _shim_seeding(args)
    _shim_genpept_prescan(args)
    _shim_genpept_prior_disabled(args)
    _shim_gamd(args)
    _shim_adaptive_production(args)
    _shim_output(args)
    _shim_system(args)
    _shim_misc(args)
    _validate_config_sanity(args)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def build_gareus_parser() -> argparse.ArgumentParser:
    """Return a bare ArgumentParser with all GAREUS arguments registered.

    Used by config validation and tests to enumerate valid YAML keys without
    triggering config-file loading or arg parsing.
    """
    p = argparse.ArgumentParser(
        prog="gareus",
        add_help=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_core_args(p)
    _add_system_args(p)
    _add_cv_args(p)
    _add_window_args(p)
    _add_us_args(p)
    _add_seeding_args(p)
    _add_genpept_prescan_args(p)
    _add_gamd_args(p)
    _add_output_args(p)
    _add_platform_args(p)
    _add_tica_args(p)
    return p


def parse_args(argv: Optional[Iterable[str]] = None):
    argv_list = _argv_as_list(argv)

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    pre.add_argument("--profile", default=None)
    pre.add_argument("--write-config-template", nargs="?", const="gareus_template.yaml",
                     default=None)
    pre_args, _ = pre.parse_known_args(argv_list)

    p = argparse.ArgumentParser(
        prog="gareus",
        add_help=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="GAREUS peptide GaMD/REUS workflow. Use -h for help or -hh for method details.",
    )
    _add_core_args(p)
    _add_system_args(p)
    _add_cv_args(p)
    _add_window_args(p)
    _add_us_args(p)
    _add_seeding_args(p)
    _add_genpept_prescan_args(p)
    _add_gamd_args(p)
    _add_output_args(p)
    _add_platform_args(p)
    _add_tica_args(p)

    if pre_args.write_config_template:
        _write_config_template(Path(pre_args.write_config_template))
        print(f"Wrote config template: {pre_args.write_config_template}")
        raise SystemExit(0)

    config_info = _apply_config_defaults_to_parser(p, pre_args.config, cli_profile=pre_args.profile)
    args = p.parse_args(argv_list)
    args._config_values = config_info.get("config_values", {})
    args._known_dests = _build_known_config_dests(p)

    _resolve_cv_aliases(args, argv_list)
    _normalize_run_mode(args, argv_list)

    # Apply compat shims before validation (validators use legacy attr names)
    _apply_v2_compat_shims(args)

    args.contact_scheme = contact_scheme(args)
    _validate_contact_args(args)
    _validate_gamd_args(args)

    return args


def _float_list_from_summary(value, *, label: str) -> list[float]:
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"double-adaptive handoff field {label!r} is not a list")
    out = []
    for item in value:
        try:
            val = float(item)
        except Exception as exc:
            raise ValueError(f"double-adaptive handoff field {label!r} contains non-numeric value {item!r}") from exc
        if not math.isfinite(val):
            raise ValueError(f"double-adaptive handoff field {label!r} contains non-finite value {item!r}")
        out.append(val)
    return out


def _write_double_adaptive_factorized_handoff_csv(args, out_dir: Path, feedback_summary: dict) -> Path | None:
    out_dir = Path(out_dir)
    sparse_csv = str(
        feedback_summary.get("handoff_windows_2d_csv")
        or feedback_summary.get("latest_sparse_windows_2d_csv_for_next_pilot_or_final")
        or ""
    ).strip()
    if sparse_csv:
        path = Path(sparse_csv)
        if path.exists():
            return path
        raise FileNotFoundError(f"double-adaptive handoff referenced sparse window CSV that does not exist: {path}")

    centers = _float_list_from_summary(
        feedback_summary.get("handoff_centers_A", feedback_summary.get("latest_proposed_centers_A")),
        label="handoff/latest primary centers",
    )
    k_values = _float_list_from_summary(
        feedback_summary.get("handoff_k_kcal_mol_A2", feedback_summary.get("latest_proposed_k_kcal_mol_A2")),
        label="handoff/latest primary force constants",
    )
    if not centers:
        return None
    if not k_values:
        k_values = [float(getattr(args, "cv1_k_default", None) or 1.0)] * len(centers)
    if len(k_values) == 1 and len(centers) > 1:
        k_values = k_values * len(centers)
    if len(k_values) != len(centers):
        raise ValueError(
            f"double-adaptive feedback handoff has {len(centers)} centers but {len(k_values)} force constants"
        )

    secondary_centers = _float_list_from_summary(
        feedback_summary.get("handoff_secondary_cv_centers", feedback_summary.get("latest_proposed_secondary_cv_centers")),
        label="handoff/latest secondary centers",
    )
    secondary_k = _float_list_from_summary(
        feedback_summary.get("handoff_secondary_cv_k_kcal_mol", feedback_summary.get("latest_proposed_secondary_cv_k_kcal_mol")),
        label="handoff/latest secondary force constants",
    )
    if secondary_centers and not secondary_k:
        secondary_k = [float(getattr(args, "cv2_k_default", None) or 50.0)] * len(secondary_centers)
    if secondary_k and len(secondary_k) == 1 and len(secondary_centers) > 1:
        secondary_k = secondary_k * len(secondary_centers)
    if secondary_centers and len(secondary_k) != len(secondary_centers):
        raise ValueError(
            f"double-adaptive feedback handoff has {len(secondary_centers)} secondary centers but {len(secondary_k)} secondary force constants"
        )

    csv_path = out_dir / "double_adaptive_handoff_windows.csv"
    rows = []
    window = 0
    if secondary_centers:
        for i, center in enumerate(centers):
            for j, sec in enumerate(secondary_centers):
                rows.append({
                    "window": int(window),
                    "primary_cv_mode": str(primary_cv_mode(args)),
                    "primary_cv_center": float(center),
                    "primary_cv_k_kcal": float(k_values[i]),
                    "distance_center_A": float(center),
                    "distance_k_kcal_mol_A2": float(k_values[i]),
                    "secondary_cv_mode": str(secondary_cv_mode(args)),
                    "secondary_cv_center": float(sec),
                    "secondary_cv_k_kcal_mol": float(secondary_k[j]),
                    "window_type": "double_adaptive_feedback_handoff",
                    "source": "adaptive_feedback_factorized_proposal",
                })
                window += 1
    else:
        for i, center in enumerate(centers):
            rows.append({
                "window": int(window),
                "primary_cv_mode": str(primary_cv_mode(args)),
                "primary_cv_center": float(center),
                "primary_cv_k_kcal": float(k_values[i]),
                "distance_center_A": float(center),
                "distance_k_kcal_mol_A2": float(k_values[i]),
                "secondary_cv_mode": "",
                "secondary_cv_center": "",
                "secondary_cv_k_kcal_mol": "",
                "window_type": "double_adaptive_feedback_handoff",
                "source": "adaptive_feedback_factorized_proposal",
            })
            window += 1

    with csv_path.open("w", newline="") as handle:
        fieldnames = list(rows[0].keys()) if rows else ["window", "primary_cv_center", "primary_cv_k_kcal"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def _propagate_pilot_shared_gamd_dir(prod_args, out_dir: Path) -> None:
    """Point adaptive-production at the pilot feedback stage's shared GaMD calibration.

    Without this, adaptive-production resolves its own empty shared_gamd_setup_dir
    to a fresh adaptive_production/global_shared_gamd_setup/ and recalibrates from
    scratch, even though the feedback stage already calibrated the same campaign-
    global envelope once in its own first pilot round. GaMD calibration is a
    property of the system, not the workflow stage, so double-adaptive runs should
    calibrate it exactly once -- in the pilot stage, before adaptive-production's
    epoch 0 -- and reuse it thereafter. Only call this before adaptive-production
    has run any epoch of its own; a run that already has epoch state committed to
    its own shared dir must keep using that dir, not switch mid-campaign.
    """
    if str(getattr(prod_args, "shared_gamd_setup_dir", "") or "").strip():
        return
    policy_path = Path(out_dir) / "adaptive_feedback_shared_gamd_policy.json"
    if not policy_path.exists():
        return
    try:
        policy = json.loads(policy_path.read_text())
        pilot_shared_gamd_dir = str(policy.get("shared_gamd_dir", "") or "").strip()
    except Exception:
        return
    if not pilot_shared_gamd_dir:
        return
    if not (Path(pilot_shared_gamd_dir) / "shared_gamd_setup_globals.json").exists():
        return
    prod_args.shared_gamd_setup_dir = pilot_shared_gamd_dir


def _committed_shared_gamd_dir(out_dir: Path) -> str:
    """Return the shared_gamd_setup_dir a prior adaptive-production invocation
    already committed epoch state against, so resuming an in-progress campaign
    reuses that calibration instead of silently defaulting to the (empty,
    never-calibrated) adaptive_production/global_shared_gamd_setup/ path.
    """
    summary_path = Path(out_dir) / "adaptive_production" / "adaptive_production_driver_summary.json"
    if not summary_path.exists():
        return ""
    try:
        summary = json.loads(summary_path.read_text())
    except Exception:
        return ""
    return str(summary.get("global_shared_gamd_setup_dir", "") or "").strip()


def run_double_adaptive_auto_loop(args, out_dir: Path, openmm, app, unit, forcefield, topology, equil_state, progress: Optional[GuiProgressSink] = None) -> dict:
    out_dir = Path(out_dir)
    summary_path = out_dir / "double_adaptive_driver_summary.json"
    adaptive_registry = out_dir / "adaptive_production" / "state_registry.json"

    feedback_driver_summary_path = out_dir / "adaptive_feedback_driver_summary.json"
    feedback_completed = False
    if feedback_driver_summary_path.exists():
        try:
            _fds = json.loads(feedback_driver_summary_path.read_text())
            feedback_completed = bool(_fds.get("final_production_skipped") or _fds.get("handoff_to_adaptive_production"))
        except Exception:
            pass

    if bool(getattr(args, "adaptive_production_resume", False)) and adaptive_registry.exists():
        prod_args = copy.deepcopy(args)
        prod_args.window_mode = "adaptive-production"
        prod_args.resume = False
        prod_args.adaptive_production_resume = True
        if not str(getattr(prod_args, "shared_gamd_setup_dir", "") or "").strip():
            _committed_dir = _committed_shared_gamd_dir(out_dir)
            if _committed_dir:
                prod_args.shared_gamd_setup_dir = _committed_dir
        prod_summary = run_adaptive_production_auto_loop(
            prod_args, out_dir, openmm, app, unit, forcefield, topology, equil_state, progress=progress
        )
        payload = {
            "schema_version": "double_adaptive_driver_summary_v1",
            "status": "completed",
            "mode": "double-adaptive",
            "resume_mode": "adaptive_production_registry",
            "adaptive_production": prod_summary,
        }
        write_json(summary_path, payload)
        return payload

    if bool(getattr(args, "adaptive_production_resume", False)) and feedback_completed and not adaptive_registry.exists():
        print("[resume] Feedback pilots already completed; skipping to adaptive-production stage.")
        try:
            feedback_summary = json.loads(feedback_driver_summary_path.read_text())
        except Exception:
            feedback_summary = {}
        handoff_csv = _write_double_adaptive_factorized_handoff_csv(args, out_dir, feedback_summary)
        prod_args = copy.deepcopy(args)
        prod_args.window_mode = "adaptive-production"
        prod_args.resume = False
        prod_args.adaptive_production_resume = True
        prod_args.adaptive_feedback_enabled = False
        prod_args.adaptive_feedback_pilot = False
        prod_args.adaptive_feedback_final_production = False
        _propagate_pilot_shared_gamd_dir(prod_args, out_dir)
        if handoff_csv is not None:
            prod_args.windows_2d_csv = str(handoff_csv)
        prod_summary = run_adaptive_production_auto_loop(
            prod_args, out_dir, openmm, app, unit, forcefield, topology, equil_state, progress=progress
        )
        payload = {
            "schema_version": "double_adaptive_driver_summary_v1",
            "status": "completed",
            "mode": "double-adaptive",
            "resume_mode": "feedback_completed_production_resume",
            "adaptive_feedback": feedback_summary,
            "handoff_windows_csv": str(handoff_csv) if handoff_csv is not None else "",
            "adaptive_production": prod_summary,
        }
        write_json(summary_path, payload)
        return payload

    print("Double-adaptive workflow")
    print("    stage 1: adaptive-feedback pilots only")
    print("    stage 2: adaptive-production initialized from feedback proposal")

    feedback_args = copy.deepcopy(args)
    feedback_args.window_mode = "adaptive-feedback"
    feedback_args.resume = False
    feedback_args.double_adaptive_handoff = True
    feedback_args.adaptive_feedback_skip_final_production = True
    feedback_summary = run_adaptive_feedback_auto_loop(
        feedback_args, out_dir, openmm, app, unit, forcefield, topology, equil_state, progress=progress
    )
    if _graceful_shutdown.is_set():
        payload = {
            "schema_version": "double_adaptive_driver_summary_v1",
            "status": "interrupted_after_feedback",
            "mode": "double-adaptive",
            "adaptive_feedback": feedback_summary,
        }
        write_json(summary_path, payload)
        return payload

    handoff_csv = _write_double_adaptive_factorized_handoff_csv(args, out_dir, feedback_summary or {})
    prod_args = copy.deepcopy(args)
    prod_args.window_mode = "adaptive-production"
    prod_args.resume = False
    prod_args.adaptive_production_resume = False
    prod_args.adaptive_feedback_enabled = False
    prod_args.adaptive_feedback_pilot = False
    prod_args.adaptive_feedback_final_production = False
    _propagate_pilot_shared_gamd_dir(prod_args, out_dir)
    if handoff_csv is not None:
        prod_args.windows_2d_csv = str(handoff_csv)
        print(f"    double-adaptive handoff window table: {handoff_csv}")
    else:
        print("    double-adaptive handoff: no feedback proposal CSV; adaptive-production will use its own initial adaptive windows")

    prod_summary = run_adaptive_production_auto_loop(
        prod_args, out_dir, openmm, app, unit, forcefield, topology, equil_state, progress=progress
    )
    payload = {
        "schema_version": "double_adaptive_driver_summary_v1",
        "status": "completed",
        "mode": "double-adaptive",
        "adaptive_feedback": feedback_summary,
        "handoff_windows_csv": str(handoff_csv) if handoff_csv is not None else "",
        "adaptive_production": prod_summary,
        "final_dir": str((out_dir / "adaptive_production" / "final")),
    }
    write_json(summary_path, payload)
    return payload


def main(argv: Optional[Iterable[str]] = None):
    _graceful_shutdown.clear()
    argv_list = _argv_as_list(argv)
    args = parse_args(argv_list)
    # For --input-pdb runs the sequence is nominal (real topology comes from the
    # supplied structure), so the >=2-residue terminal-CV gate does not apply.
    args.seq = validate_sequence(args.seq, require_min_two=not bool(getattr(args, "input_pdb", None)))
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
            print(json.dumps({"ok": True, "primary_cv": primary_cv_mode(args),
                              "note": "distance CV uses CustomBondForce; validated in smoke tests."}, indent=2, sort_keys=True))
            return
    _write_reproducibility_files(args, out_dir, argv=argv_list)
    public_args = {k: v for k, v in vars(args).items() if not str(k).startswith("_")}
    if bool(getattr(args, "resume", False)) and (out_dir / "run_args.json").exists():
        write_json(out_dir / "last_resume_args.json", public_args)
    else:
        write_json(out_dir / "run_args.json", public_args)
    initialize_run_manifest(args, out_dir, argv=argv_list)

    # Resolve --extend mode BEFORE the dispatch chain so that extend_mode='regular'
    # can set args.resume=True and fall through to the elif resume: branch below.
    if bool(getattr(args, "extend", False)):
        from gareus.adaptive_production import _resolve_and_apply_extend_mode
        _resolved_extend_mode = _resolve_and_apply_extend_mode(args, out_dir)
        _ap_window_modes = {"adaptive-production", "double-adaptive"}
        _current_window_mode = str(getattr(args, "window_mode", "adaptive"))
        if _resolved_extend_mode != "regular" and _current_window_mode not in _ap_window_modes:
            print()
            print(f"ERROR: --extend with extend_mode='{_resolved_extend_mode}' requires window_mode in")
            print(f"  {sorted(_ap_window_modes)}, but got window_mode='{_current_window_mode}'.")
            print("  Use --extend-mode regular (or omit --extend) to resume a non-adaptive-production run.")
            return

    progress = GuiProgressSink(out_dir, args)
    _run_status = "started"
    _run_error = None
    try:
        progress.emit({"event": "run_start", "sequence": args.seq, "out": str(out_dir),
                       "resume": bool(getattr(args, "resume", False))})

        if bool(getattr(args, "resume", False)) and str(getattr(args, "window_mode", "adaptive")) in {"adaptive-production", "double-adaptive"}:
            setattr(args, "adaptive_production_resume", True)
            setattr(args, "ap_resume", True)
            _resume_window_mode = str(getattr(args, "window_mode", "adaptive"))
            setattr(args, "resume", False)
            loaded_resume_setup = None
            for _pdb_dir in list(dict.fromkeys([out_dir, out_dir.parent, out_dir.parent.parent])):
                _setup = load_existing_openmm_setup_for_resume(args, _pdb_dir, require_equil_state=False)
                if _setup is not None:
                    loaded_resume_setup = _setup
                    break
            if loaded_resume_setup is None:
                print()
                print("ERROR: adaptive-production --resume could not load the solvated topology.")
                print("  Expected 01_solvated_start.pdb in the run directory or one of its parents.")
                progress.emit({"event": "resume_failed", "reason": "no_solvated_topology", "out": str(out_dir)})
                return
            openmm, app, unit, forcefield, topology, equil_state = loaded_resume_setup
            print("[resume] Adaptive-production driver resume; continuing from registry/driver summary.")
            progress.emit({"event": "adaptive_production_resume_setup_loaded", "out": str(out_dir)})
            if _resume_window_mode == "double-adaptive":
                run_double_adaptive_auto_loop(args, out_dir, openmm, app, unit, forcefield, topology, equil_state, progress=progress)
            else:
                run_adaptive_production_auto_loop(args, out_dir, openmm, app, unit, forcefield, topology, equil_state, progress=progress)

        elif bool(getattr(args, "extend", False)) and str(getattr(args, "window_mode", "adaptive")) in {"adaptive-production", "double-adaptive"}:
            # Extend a completed adaptive-production run.
            setattr(args, "adaptive_production_resume", True)
            setattr(args, "ap_resume", True)
            _extend_window_mode = str(getattr(args, "window_mode", "adaptive"))
            loaded_extend_setup = None
            for _pdb_dir in list(dict.fromkeys([out_dir, out_dir.parent, out_dir.parent.parent])):
                _setup = load_existing_openmm_setup_for_resume(args, _pdb_dir, require_equil_state=False)
                if _setup is not None:
                    loaded_extend_setup = _setup
                    break
            if loaded_extend_setup is None:
                print()
                print("ERROR: --extend could not load the solvated topology.")
                print("  Expected 01_solvated_start.pdb in the run directory or one of its parents.")
                progress.emit({"event": "extend_failed", "reason": "no_solvated_topology", "out": str(out_dir)})
                return
            openmm, app, unit, forcefield, topology, equil_state = loaded_extend_setup
            print("[extend] Adaptive-production extension; final-phase skip guard active.")
            progress.emit({"event": "adaptive_production_extend_setup_loaded", "out": str(out_dir)})
            if _extend_window_mode == "double-adaptive":
                run_double_adaptive_auto_loop(args, out_dir, openmm, app, unit, forcefield, topology, equil_state, progress=progress)
            else:
                run_adaptive_production_auto_loop(args, out_dir, openmm, app, unit, forcefield, topology, equil_state, progress=progress)

        elif bool(getattr(args, "resume", False)):
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
                for _p in _manifest_paths:
                    _exists = _p.exists()
                    _chk_ok = False
                    if _exists:
                        try:
                            _m = json.loads(_p.read_text(encoding="utf-8"))
                            _chk_ok = all((_p.parent / str(f)).exists() for f in _m.get("replica_checkpoint_files", []))
                        except Exception:
                            pass
                    print(f"    {_p}  [{'found but .chk files missing' if _exists and not _chk_ok else 'found' if _exists else 'NOT FOUND'}]")
                progress.emit({"event": "resume_failed", "reason": "no_production_checkpoint", "out": str(out_dir)})
                return

            if resume_dir != out_dir:
                print(f"[resume] Production checkpoint found in {resume_dir}; using as effective output directory.")
                out_dir = resume_dir
                args.out = str(out_dir)
                _write_reproducibility_files(args, out_dir, argv=argv_list)
                initialize_run_manifest(args, out_dir, argv=argv_list)

            loaded_resume_setup = None
            for _pdb_dir in list(dict.fromkeys([out_dir, out_dir.parent, out_dir.parent.parent])):
                _setup = load_existing_openmm_setup_for_resume(args, _pdb_dir, require_equil_state=False)
                if _setup is not None:
                    loaded_resume_setup = _setup
                    break

            if loaded_resume_setup is None:
                print()
                print("ERROR: --resume found a production checkpoint but could not load the solvated topology.")
                for _pdb_dir in list(dict.fromkeys([out_dir, out_dir.parent, out_dir.parent.parent])):
                    _pdb = _pdb_dir / "01_solvated_start.pdb"
                    print(f"    {_pdb}  [{'found' if _pdb.exists() else 'NOT FOUND'}]")
                progress.emit({"event": "resume_failed", "reason": "no_solvated_topology", "out": str(out_dir)})
                return

            openmm, app, unit, forcefield, topology, equil_state = loaded_resume_setup
            print("[resume] Reusing existing solvated topology and production checkpoints.")
            progress.emit({"event": "resume_setup_loaded", "production_checkpoint": True, "out": str(out_dir)})
            run_gareus(args, out_dir, openmm, app, unit, forcefield, topology, equil_state, progress=progress)

        else:
            openmm, app, unit, forcefield, topology, equil_state = minimize_and_npt_equilibrate(args, out_dir, progress=progress)
            if str(getattr(args, "window_mode", "adaptive")) in {"adaptive-feedback", "delaunay-feedback"}:
                run_adaptive_feedback_auto_loop(args, out_dir, openmm, app, unit, forcefield, topology, equil_state, progress=progress)
            elif str(getattr(args, "window_mode", "adaptive")) == "adaptive-production":
                run_adaptive_production_auto_loop(args, out_dir, openmm, app, unit, forcefield, topology, equil_state, progress=progress)
            elif str(getattr(args, "window_mode", "adaptive")) == "double-adaptive":
                run_double_adaptive_auto_loop(args, out_dir, openmm, app, unit, forcefield, topology, equil_state, progress=progress)
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
