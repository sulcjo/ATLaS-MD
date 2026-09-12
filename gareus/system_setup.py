"""
System preparation and equilibration utilities.

This module contains functions that build and prepare an OpenMM system
for a peptide simulation.  The helpers here were originally part of
``gareus_peptide.py`` but have been extracted into their own module to
improve modularity.  They cover sequence validation, peptide
construction, force field and platform selection, solvation, restraint
application, minimisation, NVT/NPT equilibration, and basic trajectory
reporting.  Where appropriate, these functions delegate to other
submodules such as ``gareus.imports`` for external imports, ``gareus.units``
for unit conversions, ``gareus.constants`` for residue name mappings and
``gareus.progress`` for progress reporting.

The functions here are designed to mirror their counterparts in
``gareus_peptide.py`` as closely as possible, retaining the same
signatures and behaviour.  Consumers may call them directly via
``gareus.system_setup`` or, for backwards compatibility, via the
legacy names in ``gareus_peptide.py`` which alias to this module.
"""

from __future__ import annotations

import collections
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from .constants import AA3, WATER_RESNAMES, ION_RESNAMES
from .imports import import_openmm, import_peptidebuilder
from .units import kcal_a2_to_kj_nm2
from .progress import GuiProgressSink

__all__ = [
    "validate_sequence",
    "read_pdb_residue_sequence",
    "resolve_input_pdb",
    "build_peptide_pdb",
    "platform_and_properties",
    "extra_platform_properties",
    "replica_platform_properties",
    "_first_device_index",
    "setup_platform_and_properties",
    "platform_summary",
    "make_forcefield",
    "make_forcefield_from_args",
    "forcefield_xml_paths",
    "forcefield_selection_from_args",
    "create_system",
    "prepare_solvated_system",
    "make_langevin_integrator",
    "add_position_restraints",
    "make_trajectory_reporter",
    "write_solute_only_topology_pdb",
    "write_solute_only_pdb",
    "write_state_pdb",
    "run_steps_safely",
    "minimize_and_npt_equilibrate",
    "_write_box_audit",
    "BarostatOwnership",
    "resolve_barostat_ownership",
    "preflight_barostat_ownership",
]


def validate_sequence(seq: str, require_min_two: bool = True) -> str:
    """Validate a peptide one‑letter sequence and return the upper‑case string.

    The sequence must consist solely of canonical residues defined in
    ``gareus.constants.AA3`` and must be non‑empty.  By default it must also
    contain at least two residues to allow a terminal distance collective
    variable; pass ``require_min_two=False`` to relax this (e.g. for
    ``--input-pdb`` runs where ``seq`` is nominal and the real topology comes
    from the supplied structure).  If validation fails, a ``ValueError`` is
    raised.
    """
    seq = seq.strip().upper()
    bad = sorted(set(seq) - set(AA3))
    if not seq:
        raise ValueError("Empty sequence.")
    if bad:
        raise ValueError(f"Unsupported residues: {bad}. Only canonical one‑letter residues are supported.")
    if require_min_two and len(seq) < 2:
        raise ValueError("Need at least 2 residues for a terminal‑distance CV.")
    return seq


def read_pdb_residue_sequence(pdb_path: Path) -> list[str]:
    """Read unique canonical protein residue names from a PDB in file order.

    The returned list contains three‑letter residue codes in the order
    they appear in the PDB, omitting water and ion residues.  Only
    canonical residues present in ``AA3.values()`` are considered.
    """
    residues: list[tuple[str, str, str, str]] = []
    seen = set()
    with Path(pdb_path).open() as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            resname = line[17:20].strip().upper()
            if resname not in set(AA3.values()):
                continue
            chain = line[21].strip()
            resseq = line[22:26].strip()
            icode = line[26].strip()
            key = (chain, resseq, icode, resname)
            if key not in seen:
                seen.add(key)
                residues.append(key)
    return [r[3] for r in residues]


def resolve_input_pdb(args, base_dir=None) -> Optional[Path]:
    """Return the resolved absolute path to an explicit input PDB, or None.

    When ``args.input_pdb`` is set, the structure is used verbatim and the
    PeptideBuilder build + ``addHydrogens`` step are bypassed. Relative paths are
    resolved against ``base_dir`` (default: current working directory), matching
    how gareus resolves config-relative paths at launch.
    """
    raw = getattr(args, "input_pdb", None)
    if not raw:
        return None
    p = Path(raw)
    if not p.is_absolute():
        base = Path(base_dir) if base_dir is not None else Path.cwd()
        p = base / p
    if not p.exists():
        raise FileNotFoundError(
            f"--input-pdb {raw!r} not found (resolved to {p})"
        )
    return p


def build_peptide_pdb(seq: str, out_pdb: Path, phi_deg: float = -60.0, psi_deg: float = -45.0) -> Path:
    """Build a simple peptide PDB from a one‑letter sequence with PeptideBuilder.

    The ``PeptideBuilder.Geometry.geometry()`` function expects one‑letter
    residue codes.  Passing three‑letter codes can silently create
    glycines.  After constructing the peptide, the output PDB is
    validated to ensure the built residue sequence matches the input.
    """
    PeptideBuilder, Geometry, PDBIO = import_peptidebuilder()
    seq = validate_sequence(seq)
    out_pdb = Path(out_pdb)
    out_pdb.parent.mkdir(parents=True, exist_ok=True)

    # Start with the first residue, then iteratively add the rest with
    # specified backbone torsions.  Explicitly set phi/psi/omega where
    # attributes exist to avoid unexpected default values.
    structure = PeptideBuilder.initialize_res(Geometry.geometry(seq[0]))
    for aa in seq[1:]:
        geo = Geometry.geometry(aa)
        if hasattr(geo, "phi"):
            geo.phi = float(phi_deg)
        if hasattr(geo, "psi_im1"):
            geo.psi_im1 = float(psi_deg)
        if hasattr(geo, "omega"):
            geo.omega = 180.0
        PeptideBuilder.add_residue(structure, geo)

    # Attempt to add a terminal OXT atom; ignore failures on older versions.
    try:
        PeptideBuilder.add_terminal_OXT(structure)
    except Exception:
        pass

    io = PDBIO()
    io.set_structure(structure)
    io.save(str(out_pdb))

    expected = [AA3[a] for a in seq]
    built = read_pdb_residue_sequence(out_pdb)
    if built != expected:
        raise RuntimeError(
            "PeptideBuilder produced the wrong sequence. "
            f"Expected {expected}, got {built}. "
            "Check the installed PeptideBuilder version/residue code handling."
        )
    return out_pdb


def _tri_state_platform_property(value) -> Optional[str]:
    """Return OpenMM's lowercase string form for true/false/auto platform flags."""
    text = str(value or "auto").strip().lower()
    if text in {"", "auto", "none", "default"}:
        return None
    if text in {"1", "true", "yes", "on"}:
        return "true"
    if text in {"0", "false", "no", "off"}:
        return "false"
    raise ValueError(f"Expected auto/true/false for platform property, got {value!r}")


def extra_platform_properties(platform_name: str, args=None) -> dict[str, str]:
    """Return optional OpenMM platform properties requested through CLI flags.

    Defaults intentionally omit every optional property so historical behavior is
    preserved.  The properties are only injected when the user explicitly sets
    the corresponding option.  CUDA and HIP share most property names in OpenMM;
    OpenCL support is more platform-dependent, so only precision/device routing
    is set there unless the user extends this function.

    ``--cuda-mps`` applies soft defaults (UseBlockingSync=false, DeterministicForces=false)
    that individual ``--cuda-*`` flags can override.  These settings are required for
    CUDA Multi-Process Service (MPS) to allow concurrent multi-replica GPU utilization;
    enable MPS on the node with ``nvidia-cuda-mps-control -d`` before launching.
    """
    if args is None:
        return {}
    name = str(platform_name or "").strip()
    props: dict[str, str] = {}
    accelerated = name in {"CUDA", "HIP"}
    if accelerated:
        mps = bool(getattr(args, "cuda_mps", False))
        # Soft defaults applied when --cuda-mps is set and the flag is not explicitly overridden.
        mps_defaults: dict[str, str] = (
            {"cuda_use_blocking_sync": "false", "cuda_deterministic_forces": "false"} if mps else {}
        )
        mappings = [
            ("cuda_use_cpu_pme", "UseCpuPme"),
            ("cuda_use_blocking_sync", "UseBlockingSync"),
            ("cuda_deterministic_forces", "DeterministicForces"),
            # OpenMM 8.3+: DisablePmeStream=false keeps the dedicated PME CUDA stream
            # enabled so PME reciprocal-space work overlaps direct-space on the GPU.
            # Default (auto) lets OpenMM decide; set to 'false' to explicitly enable.
            ("cuda_disable_pme_stream", "DisablePmeStream"),
        ]
        for attr, prop_name in mappings:
            raw = getattr(args, attr, "auto")
            if str(raw or "auto").strip().lower() in {"", "auto", "none", "default"} and attr in mps_defaults:
                raw = mps_defaults[attr]
            val = _tri_state_platform_property(raw)
            if val is not None:
                props[prop_name] = val
        temp_dir = str(getattr(args, "platform_temp_directory", "") or "").strip()
        if temp_dir:
            props["TempDirectory"] = temp_dir
    return props


def platform_and_properties(openmm, platform_name: str, precision: str, device_index: str, cpu_threads: int, args=None):
    """Return an OpenMM platform and a properties dict for the requested configuration."""
    platform_name = str(platform_name or "auto")
    if platform_name.lower() == "auto":
        order = ["CUDA", "HIP", "OpenCL", "CPU", "Reference"]
        names = [openmm.Platform.getPlatform(i).getName() for i in range(openmm.Platform.getNumPlatforms())]
        for name in order:
            if name in names:
                platform_name = name
                break
    platform = openmm.Platform.getPlatformByName(platform_name)
    props: dict[str, str] = {}
    if platform_name in {"CUDA", "HIP", "OpenCL"}:
        if precision:
            props["Precision"] = str(precision)
        if device_index != "":
            props["DeviceIndex"] = str(device_index)
        props.update(extra_platform_properties(platform_name, args=args))
    if platform_name == "CPU" and int(cpu_threads) > 0:
        props["Threads"] = str(int(cpu_threads))
    return platform, props


def _first_device_index(device_index: str) -> str:
    """Return the first device token from a comma‑separated device index string."""
    tokens = [x.strip() for x in str(device_index or "").split(",") if x.strip() != ""]
    return tokens[0] if tokens else ""


def replica_platform_properties(platform, base_props: dict, args, replica_index: int) -> dict[str, str]:
    """Return per-replica OpenMM platform properties.

    ``--device-index`` accepts a comma-separated list.  By default, accelerated
    platforms assign one device token per replica in round-robin order, which is
    usually better for many independent peptide replicas than asking every
    single Context to split across every GPU.  Use
    ``--replica-device-mode single-context-split`` to preserve the old
    comma-list-as-one-Context behavior, or ``--replica-device-map`` for an
    explicit repeating map.
    """
    props = dict(base_props or {})
    try:
        platform_name = str(platform.getName())
    except Exception:
        platform_name = str(getattr(args, "platform", "") or "")
    if platform_name not in {"CUDA", "HIP", "OpenCL"}:
        return props

    tokens = [x.strip() for x in str(getattr(args, "device_index", "") or "").split(",") if x.strip()]
    map_tokens = [x.strip() for x in str(getattr(args, "replica_device_map", "") or "").split(",") if x.strip()]
    mode = str(getattr(args, "replica_device_mode", "auto") or "auto").strip().lower().replace("_", "-")
    if mode == "auto":
        mode = "round-robin" if len(tokens) > 1 or map_tokens else "single-context-split"
    if mode == "manual":
        if not map_tokens:
            raise ValueError("--replica-device-mode manual requires --replica-device-map")
        props["DeviceIndex"] = map_tokens[int(replica_index) % len(map_tokens)]
    elif mode == "round-robin":
        route = map_tokens if map_tokens else tokens
        if route:
            props["DeviceIndex"] = route[int(replica_index) % len(route)]
    elif mode == "single-context-split":
        if tokens:
            props["DeviceIndex"] = ",".join(tokens)
    else:
        raise ValueError(f"Unsupported --replica-device-mode {mode!r}")
    return props


def setup_platform_and_properties(openmm, args):
    """Platform and properties for single‑context setup and preparation phases.

    This helper follows ``--platform``/``--precision`` but uses only the first
    entry of ``--device-index``.  This keeps setup simulations from being
    inadvertently distributed across multiple GPUs when a comma‑separated list
    is provided.  The ``args`` object should expose attributes like
    ``setup_platform``, ``platform``, ``setup_precision``, ``precision``,
    ``setup_device_index``, ``device_index`` and ``setup_cpu_threads``.
    """
    setup_platform = getattr(args, "setup_platform", "") or getattr(args, "platform", "auto")
    setup_precision = getattr(args, "setup_precision", "") or getattr(args, "precision", "mixed")
    setup_device_index = getattr(args, "setup_device_index", "")
    if setup_device_index is None or str(setup_device_index).strip() == "":
        setup_device_index = _first_device_index(getattr(args, "device_index", ""))
    setup_cpu_threads = int(getattr(args, "setup_cpu_threads", 0) or getattr(args, "cpu_threads", 1) or 1)
    return platform_and_properties(
        openmm,
        setup_platform,
        setup_precision,
        str(setup_device_index),
        setup_cpu_threads,
        args=args,
    )


def resolve_cpu_threads_for_replicas(args, n_replicas: int) -> int:
    """Return effective per-replica CPU thread count.

    If --cpu-budget is set, distributes evenly: floor(cpu_budget / n_replicas),
    clamped to at least 1. Capped by --max-cpu-per-replica when non-zero.
    Falls back to --cpu-threads when no budget is specified.
    """
    budget = int(getattr(args, "cpu_budget", 0) or 0)
    max_per = int(getattr(args, "max_cpu_per_replica", 0) or 0)
    base = int(getattr(args, "cpu_threads", 1) or 1)

    if budget > 0:
        n = max(1, int(n_replicas))
        per_replica = max(1, budget // n)
    else:
        per_replica = base

    if max_per > 0:
        per_replica = min(per_replica, max_per)

    return max(1, per_replica)


def platform_summary(platform, props: dict) -> str:
    """Return a compact log string for an OpenMM platform and its selected properties."""
    try:
        name = platform.getName()
    except Exception:
        name = str(platform)
    clean = {str(k): str(v) for k, v in dict(props or {}).items()}
    return f"{name} {clean}" if clean else str(name)


# Single source of truth for which XMLs a run loads. `provenance` reads this
# too: the two used to carry duplicate water dicts, and updating one without the
# other makes the provenance record disagree with what actually ran.
FORCEFIELD_XML = {
    "ff14SB": "amber14-all.xml",
    "ff19SB": "amber19-all.xml",
}

# The amber14/ and amber19/ copies of every one of these files are byte-identical
# in OpenMM 8.5.1, so the directory prefix is cosmetic; keeping the amber14/ paths
# avoids churning the provenance record for zero numerical change.
WATER_XML = {
    "tip3p": "amber14/tip3p.xml",
    "tip3pfb": "amber14/tip3pfb.xml",
    "spce": "amber14/spce.xml",
    "tip4pew": "amber14/tip4pew.xml",
    "opc": "amber14/opc.xml",
}

# ff19SB was parameterized against OPC. Any other water is a known mismatch.
_REQUIRED_WATER = {"ff19SB": "opc"}


def forcefield_xml_paths(
    forcefield: str = "ff14SB",
    water_model: str = "tip3p",
    *,
    allow_mismatch: bool = False,
) -> list[str]:
    """Resolve (force field, water model) to the XML list OpenMM should load.

    Raises rather than defaulting on an unknown name: a typo that silently
    produced an amber14/TIP3P run would be invisible in the output and wrong in
    the free energies.
    """
    ff = str(forcefield or "ff14SB")
    water = str(water_model or "tip3p")
    if ff not in FORCEFIELD_XML:
        raise ValueError(
            f"unknown forcefield {ff!r}; choose one of {sorted(FORCEFIELD_XML)}")
    if water not in WATER_XML:
        raise ValueError(
            f"unknown water model {water!r}; choose one of {sorted(WATER_XML)}")
    required = _REQUIRED_WATER.get(ff)
    if required is not None and water != required and not allow_mismatch:
        raise ValueError(
            f"{ff} was parameterized against {required} water; refusing to pair it "
            f"with {water!r}. Pass allow_mismatch=True "
            f"(--allow-forcefield-water-mismatch) only if you intend this."
        )
    return [FORCEFIELD_XML[ff], WATER_XML[water]]


def forcefield_selection_from_args(args) -> tuple[str, str, bool]:
    """Read the selection off `args` so every construction site agrees.

    `checkpoints.py` and `swarm/driver.py` build force fields independently of
    the production path. If each carried its own default, a resume could rebuild
    an ff19SB system as ff14SB without complaint -- invisible and unrecoverable.
    """
    return (
        str(getattr(args, "forcefield", None) or "ff14SB"),
        str(getattr(args, "water_model", None) or "tip3p"),
        bool(getattr(args, "allow_forcefield_water_mismatch", False)),
    )


def make_forcefield(app, water_model: str = "tip3p", forcefield: str = "ff14SB",
                    *, allow_mismatch: bool = False):
    """Construct the force field for the selected protein FF and water model."""
    return app.ForceField(*forcefield_xml_paths(
        forcefield, water_model, allow_mismatch=allow_mismatch))


def make_forcefield_from_args(app, args):
    """Preferred entry point: one selection, used identically everywhere."""
    ff, water, allow = forcefield_selection_from_args(args)
    return make_forcefield(app, water, ff, allow_mismatch=allow)


def create_system(app, unit, forcefield, topology, args, include_barostat: bool, barostat_frequency: Optional[int] = None):
    """Create an OpenMM System with optional barostat and hydrogen mass repartitioning."""
    nonbonded_cutoff = args.nonbonded_cutoff_nm * unit.nanometer
    system_kwargs = dict(
        nonbondedMethod=app.PME,
        nonbondedCutoff=nonbonded_cutoff,
        constraints=app.HBonds,
        rigidWater=True,
        ewaldErrorTolerance=args.ewald_error_tolerance,
    )
    # Hydrogen mass repartitioning: allow user to set a custom hydrogen mass
    # or fall back to a default if ``--hmr`` is specified.
    if getattr(args, "hmr", False) or float(getattr(args, "hydrogen_mass_amu", 0.0) or 0.0) > 0:
        hm = float(getattr(args, "hydrogen_mass_amu", 3.024) or 3.024)
        system_kwargs["hydrogenMass"] = hm * unit.amu
    system = forcefield.createSystem(topology, **system_kwargs)
    if include_barostat:
        # Defer heavy imports to ``gareus.imports.import_openmm`` to avoid
        # importing OpenMM at module load time.
        openmm, _app, _unit = import_openmm()
        frequency = int(barostat_frequency if barostat_frequency is not None else getattr(args, "barostat_frequency", 100))
        barostat = openmm.MonteCarloBarostat(
            args.pressure_bar * unit.bar,
            args.temperature_k * unit.kelvin,
            frequency,
        )
        system.addForce(barostat)
    return system


# ---------------------------------------------------------------------------
# Barostat ownership (NPT correction, spec sections 5-6)
#
# The decision of WHO owns volume sampling must be made BEFORE any Context is
# created from a System: the application-controlled backend requires the native
# MonteCarloBarostat to be absent from the System it drives, and the native
# backend requires exactly one barostat and no application controller.
# ---------------------------------------------------------------------------

# Barostat families this correction explicitly does not support (spec section
# 4: no anisotropic or membrane barostats; their proposal/Jacobian differs).
# MonteCarloFlexibleBarostat (flexible constraints) and the RPMD variant are
# excluded for the same reason: only the isotropic rigid-constraint
# MonteCarloBarostat / biased-MC proposal has a verified Jacobian here.
_UNSUPPORTED_BAROSTAT_FORCE_NAMES = (
    "MonteCarloAnisotropicBarostat",
    "MonteCarloMembraneBarostat",
    "MonteCarloFlexibleBarostat",
    "RPMDMonteCarloBarostat",
)


class BarostatOwnership:
    """Resolved barostat ownership for one System-construction site.

    ``backend`` is one of ``"none"`` (NVT: no volume controller of any kind),
    ``"native"`` (the OpenMM MonteCarloBarostat stays in the System) or
    ``"biased_mc"`` (the application-controlled
    ``gareus.npt.BiasedMCBarostatController`` owns volume moves and the native
    barostat must never be added to the System).
    """

    def __init__(
        self,
        backend: str,
        *,
        requested: str,
        ensemble: str,
        run_mode: str,
        boost_type: str,
        barostat_frequency: int,
        pressure_bar: float,
        temperature_k: float,
        volume_step_fraction: float,
    ):
        self.backend = str(backend)
        self.requested = str(requested)
        self.ensemble = str(ensemble)
        self.run_mode = str(run_mode)
        self.boost_type = str(boost_type)
        self.barostat_frequency = int(barostat_frequency)
        self.pressure_bar = float(pressure_bar)
        self.temperature_k = float(temperature_k)
        self.volume_step_fraction = float(volume_step_fraction)

    @property
    def include_native_barostat(self) -> bool:
        """What to pass as ``create_system(include_barostat=...)``.

        Only ``native`` keeps the native barostat; ``biased_mc`` systems are
        built without it from the start (removal before Context creation).
        """
        return self.backend == "native"

    def describe(self) -> str:
        return (
            f"backend={self.backend} (requested={self.requested}, ensemble={self.ensemble}, "
            f"run_mode={self.run_mode}, boost_type={self.boost_type or 'none'}), "
            f"P={self.pressure_bar:g} bar, T={self.temperature_k:g} K, "
            f"frequency={self.barostat_frequency} steps"
            + (
                f", volume_step_fraction={self.volume_step_fraction:g}"
                if self.backend == "biased_mc"
                else ""
            )
        )


def _production_barostat_frequency(args) -> int:
    """Existing production override semantics, unchanged by this correction.

    ``--production-barostat-frequency`` (nonzero) wins over
    ``--barostat-frequency``; both default to 100 today.
    """
    freq = int(getattr(args, "production_barostat_frequency", 0) or 0)
    if freq <= 0:
        freq = int(getattr(args, "barostat_frequency", 100) or 100)
    if freq <= 0:
        raise ValueError(
            "barostat frequency must be a positive number of integration steps "
            f"(got production_barostat_frequency={getattr(args, 'production_barostat_frequency', 0)!r}, "
            f"barostat_frequency={getattr(args, 'barostat_frequency', 100)!r})"
        )
    return freq


def resolve_barostat_ownership(args, *, ensemble: str, run_mode: Optional[str] = None, boost_type: Optional[str] = None) -> BarostatOwnership:
    """Decide barostat ownership BEFORE any Context exists.

    This is the single decision point shared by the production base system and
    the swarm base system.  It delegates the actual backend choice to the
    frozen ``gareus.npt.resolve_npt_backend`` contract, which fails loudly on
    the two forbidden silent outcomes: explicit ``native`` with boosted
    dynamics, and an unsupported boosted mode silently downgraded to NVT.

    NVT never needs the resolver (there is no volume controller regardless of
    backend); requesting an explicit backend together with NVT is a
    contradiction and is rejected here.
    """
    from . import npt

    requested = str(getattr(args, "npt_barostat_backend", "auto") or "auto").strip().lower()
    if requested not in {"auto", "native", "biased_mc"}:
        raise ValueError(
            f"npt_barostat_backend must be auto, native or biased_mc (got {requested!r})"
        )
    ensemble = str(ensemble).strip().lower()
    if ensemble not in {"npt", "nvt"}:
        raise ValueError(f"ensemble must be npt or nvt (got {ensemble!r})")
    mode = str(run_mode if run_mode is not None else getattr(args, "run_mode", "cmd") or "cmd")
    boost = str(boost_type if boost_type is not None else getattr(args, "gamd_boost_type", "") or "")
    fraction = float(getattr(args, "barostat_volume_step_fraction", 0.01) or 0.01)

    if ensemble == "nvt":
        if requested != "auto":
            raise ValueError(
                f"--npt-barostat-backend {requested!r} was requested together with "
                "--production-ensemble nvt: an NVT run has no volume controller; "
                "either drop the backend flag or select the npt ensemble"
            )
        backend = "none"
    else:
        # The frozen contract owns the auto/native/biased_mc dispatch, including
        # the loud failures for explicit-native-with-boost and unsupported
        # boosted modes (never a silent ensemble downgrade).
        try:
            backend = npt.resolve_npt_backend(
                ensemble=ensemble,
                requested=requested,
                run_mode=mode,
                boost_type=boost,
            )
        except NotImplementedError:
            # Interim state while the NPT correction's package 1 (the biased-MC
            # controller itself) has not landed: the contract stub raises.  Per
            # the spec this configuration maps to the application-controlled
            # biased-MC backend, which does not exist yet -- and silently
            # keeping the native barostat would preserve exactly the known-
            # wrong acceptance energy the correction exists to remove.  Fail
            # loudly with an actionable message instead.
            raise RuntimeError(
                f"The requested run (ensemble={ensemble!r}, run_mode={mode!r}, "
                f"boost_type={boost!r}, npt_barostat_backend={requested!r}) resolves to "
                "the application-controlled biased-MC NPT backend, whose controller "
                "is not implemented in this build (gareus/npt.py raises "
                "NotImplementedError: NPT correction package 1). Running it with the "
                "native MonteCarloBarostat would sample the wrong volume distribution "
                "(acceptance energy omits the boost and includes the Pep-GaMD auxiliary "
                "force), so this is refused rather than silently downgraded. Use "
                "--production-ensemble nvt, a conventional (cmd/hmr-cmd) run mode, or "
                "land NPT correction package 1 before running boosted NPT production."
            ) from None
        if backend not in {"none", "native", "biased_mc"}:
            raise RuntimeError(
                f"gareus.npt.resolve_npt_backend returned an unknown backend {backend!r}"
            )

    ownership = BarostatOwnership(
        backend,
        requested=requested,
        ensemble=ensemble,
        run_mode=mode,
        boost_type=boost,
        barostat_frequency=_production_barostat_frequency(args) if backend != "none" else 0,
        pressure_bar=float(getattr(args, "pressure_bar", 1.0)),
        temperature_k=float(getattr(args, "temperature_k", 300.0)),
        volume_step_fraction=fraction,
    )
    print(f"[npt] Barostat ownership resolved: {ownership.describe()}")
    return ownership


def preflight_barostat_ownership(system, ownership: BarostatOwnership) -> dict:
    """Assert exactly one volume controller and run the physical preflight.

    Called on the fully-assembled application-controlled System (all umbrella
    and secondary-CV forces added), before the first Context is created from
    it.  Checks:

    * no unsupported barostat family (anisotropic/membrane) anywhere;
    * ``native`` backend: exactly one MonteCarloBarostat-family force, and the
      stepping integrator is conventional (boosted dynamics would make its
      acceptance energy wrong -- resolve_npt_backend already refuses this, the
      preflight is the belt to that braces);
    * ``biased_mc`` backend: zero native barostats (the controller is the one
      volume controller and is attached per replica later), a periodic box
      (volume moves need it) and no immobile non-virtual-site particle (the
      molecule-translation Jacobian cannot count it).
    """
    from . import npt

    for i in range(system.getNumForces()):
        name = system.getForce(i).__class__.__name__
        if name in _UNSUPPORTED_BAROSTAT_FORCE_NAMES:
            raise ValueError(
                f"Force {i} is {name}: anisotropic/membrane barostats are not "
                "supported by the NPT correction; use the isotropic "
                "MonteCarloBarostat / biased-MC backends"
            )
    n_native = int(npt.count_native_barostats(system))
    backend = str(ownership.backend)
    if backend == "native":
        if n_native != 1:
            raise RuntimeError(
                f"native barostat backend requires exactly one MonteCarloBarostat "
                f"in the System, found {n_native}"
            )
        if str(ownership.run_mode) in {"gamd", "hmr-gamd"}:
            raise RuntimeError(
                "native barostat backend with boosted dynamics (run_mode="
                f"{ownership.run_mode!r}) samples the wrong volume distribution: "
                "the Context potential excludes the boost and includes the "
                "Pep-GaMD auxiliary energy. Use the biased_mc backend."
            )
    elif n_native != 0:
        raise RuntimeError(
            f"{backend!r} barostat backend requires the native MonteCarloBarostat "
            "to be removed from application-controlled Systems before Context "
            f"creation; found {n_native}"
        )
    if backend == "biased_mc":
        if not bool(system.usesPeriodicBoundaryConditions()):
            raise RuntimeError("biased_mc NPT requires a periodic System")
        openmm, _app, _unit = import_openmm()
        for i in range(system.getNumParticles()):
            mass = system.getParticleMass(i)
            if float(mass.value_in_unit(_unit.dalton)) == 0.0 and not system.isVirtualSite(i):
                raise RuntimeError(
                    f"particle {i} is massless but not a virtual site: immobile "
                    "particles break the volume-move molecule Jacobian"
                )
    return {
        "backend": backend,
        "native_barostats_in_system": n_native,
        "pressure_bar": float(ownership.pressure_bar),
        "temperature_k": float(ownership.temperature_k),
        "barostat_frequency_steps": int(ownership.barostat_frequency),
        "volume_step_fraction": float(ownership.volume_step_fraction),
    }


def _write_box_audit(
    out_dir: Path,
    pos_nm: "np.ndarray",
    ca_pos_nm: "np.ndarray",
    contour_nm: float,
    box_nm: float,
    args,
) -> dict:
    """Compute and write a PBC box-size audit JSON to *out_dir/box_audit.json*.

    All inputs are pure numpy arrays — no OpenMM dependency in this function.
    The audit checks whether the sequence contour estimate (fully-extended
    peptide length) might cause self-contact through the periodic boundary.

    Parameters
    ----------
    out_dir:
        Directory where ``box_audit.json`` is written.
    pos_nm:
        All solute atom positions in nm, shape (N, 3).
    ca_pos_nm:
        Cα positions in chain order, shape (M, 3).  Caller must sort by
        (chain, residue, atom) index before passing.
    contour_nm:
        The *box-sizing* contour used by ``prepare_solvated_system`` (may be
        the actual-extent value for input_pdb runs — NOT used here for the
        sequence estimate).
    box_nm:
        Final box edge length in nm (cubic / equivalent).
    args:
        Namespace with ``padding_nm`` and ``seq`` attributes.

    Returns
    -------
    dict
        The audit record that was written to disk.
    """
    # --- actual extent --------------------------------------------------
    extent = pos_nm.max(axis=0) - pos_nm.min(axis=0)
    actual_extent_nm = [float(v) for v in extent]

    # --- sequence contour estimate: always from residue count -----------
    # Use CA count (topology-aware) when available; fall back to seq length.
    n_residues = len(ca_pos_nm) if len(ca_pos_nm) > 0 else len(args.seq)
    sequence_contour_estimate_nm = float((n_residues - 1) * 0.38 + 0.40)

    # --- backbone path length -------------------------------------------
    if len(ca_pos_nm) >= 2:
        diffs = ca_pos_nm[1:] - ca_pos_nm[:-1]
        backbone_path_length_nm = float(np.sqrt((diffs ** 2).sum(axis=1)).sum())
    else:
        backbone_path_length_nm = 0.0

    # --- margin and warning ---------------------------------------------
    padding_nm = float(args.padding_nm)
    minimum_margin_nm = float(box_nm / 2.0 - float(max(actual_extent_nm)) / 2.0)
    cutoff_nm = float(getattr(args, "nonbonded_cutoff_nm", 1.0) or 1.0)
    min_image_gap_nm = float(box_nm - sequence_contour_estimate_nm)
    pbc_self_contact_warning = bool(min_image_gap_nm < cutoff_nm)
    if pbc_self_contact_warning:
        warning_message = (
            f"PBC warning: min-image gap {min_image_gap_nm:.2f} nm "
            f"(box {box_nm:.2f} − contour {sequence_contour_estimate_nm:.2f}) "
            f"is below the nonbonded cutoff {cutoff_nm:.2f} nm; "
            "the peptide can interact with its own image."
        )
    else:
        warning_message = None

    audit = {
        "actual_extent_nm": actual_extent_nm,
        "sequence_contour_estimate_nm": sequence_contour_estimate_nm,
        "backbone_path_length_nm": backbone_path_length_nm,
        "box_size_nm": float(box_nm),
        "padding_nm": padding_nm,
        "minimum_margin_nm": minimum_margin_nm,
        "min_image_gap_nm": min_image_gap_nm,
        "nonbonded_cutoff_nm": cutoff_nm,
        "pbc_self_contact_warning": pbc_self_contact_warning,
        "warning_message": warning_message,
    }

    out_dir = Path(out_dir)
    with (out_dir / "box_audit.json").open("w") as fh:
        json.dump(audit, fh, indent=2)

    if pbc_self_contact_warning:
        print(f"[setup] WARNING: {warning_message}", flush=True)

    return audit


def prepare_solvated_system(args, out_dir: Path):
    """Build and solvate a peptide system, writing intermediate PDB files.

    Returns a tuple ``(openmm, app, unit, forcefield, modeller, raw_pdb, solvated_pdb)``.
    ``raw_pdb`` is the generated peptide PDB before solvation; ``solvated_pdb`` is
    the solvated starting PDB written to disk.  The parent directory of
    ``out_dir`` will be created if necessary.
    """
    openmm, app, unit = import_openmm()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    forcefield = make_forcefield_from_args(app, args)
    input_pdb = resolve_input_pdb(args, base_dir=getattr(args, "config_base_dir", None))
    if input_pdb is not None:
        raw_pdb = out_dir / "00_input_structure.pdb"
        shutil.copyfile(input_pdb, raw_pdb)
        pdb = app.PDBFile(str(raw_pdb))
        modeller = app.Modeller(pdb.topology, pdb.positions)
        print(
            f"[setup] input_pdb bypass: loaded {input_pdb} "
            f"({modeller.topology.getNumAtoms()} atoms, "
            f"residues={[r.name for r in modeller.topology.residues()]}); "
            "skipping PeptideBuilder + addHydrogens"
        )
    else:
        raw_pdb = build_peptide_pdb(args.seq, out_dir / "00_built_peptide.pdb", args.initial_phi, args.initial_psi)
        pdb = app.PDBFile(str(raw_pdb))
        modeller = app.Modeller(pdb.topology, pdb.positions)
        modeller.addHydrogens(forcefield, pH=args.ph)

    # Size the box against the fully-extended conformation, not the compact
    # alpha-helical starting structure.  Treat the contour length (3.8 Å per
    # residue in a fully-extended beta-strand + 4 Å terminal radii) as the
    # effective peptide size, then add the user-requested padding on each side.
    # Always extract positions here so they are available for the audit below
    # regardless of whether we are on the input_pdb or generated-peptide path.
    _pos_nm = np.asarray(modeller.positions.value_in_unit(unit.nanometer), dtype=float)
    if input_pdb is not None:
        # Size the box from the actual solute extent; the seq-based contour
        # estimate is meaningless for an externally supplied fixture.
        _contour_nm = float((_pos_nm.max(axis=0) - _pos_nm.min(axis=0)).max()) + 0.40
    else:
        _contour_nm = (len(args.seq) - 1) * 0.38 + 0.40
    _box_nm = _contour_nm + 2.0 * float(args.padding_nm)

    # Extract Cα positions in chain order for backbone path-length audit.
    _ca_pos_nm = []
    for atom in modeller.topology.atoms():
        if atom.name == "CA":
            _ca_pos_nm.append(_pos_nm[atom.index])
    _ca_pos_nm = np.array(_ca_pos_nm) if _ca_pos_nm else np.empty((0, 3))

    _write_box_audit(out_dir, _pos_nm, _ca_pos_nm, _contour_nm, _box_nm, args)

    print(
        f"[setup] Box size from extended conformation: "
        f"seq={args.seq} ({len(args.seq)} res), contour={_contour_nm:.2f} nm, "
        f"padding={args.padding_nm:.2f} nm → boxSize={_box_nm:.2f} nm"
    )

    solvent_kwargs = {
        "model": args.water_model,
        "boxSize": openmm.Vec3(_box_nm, _box_nm, _box_nm) * unit.nanometer,
        "ionicStrength": args.ionic_strength_molar * unit.molar,
        "neutralize": True,
    }
    box_shape = getattr(args, "box_shape", "dodecahedron")
    try:
        modeller.addSolvent(forcefield, boxShape=box_shape, **solvent_kwargs)
    except TypeError as exc:
        # OpenMM versions before boxShape support accepted padding but always built
        # a cubic box.  Do not silently turn a requested dodecahedron into a cube:
        # that changes atom count, PME setup, and reproducibility metadata.
        if "boxShape" in str(exc):
            raise RuntimeError(
                "This OpenMM Modeller.addSolvent implementation does not support "
                "boxShape=. Use a newer OpenMM build for --box-shape "
                f"{box_shape!r}, or rerun explicitly with --box-shape cube only "
                "after accepting the legacy cubic-box behavior."
            ) from exc
        raise
    solvated_pdb = out_dir / "01_solvated_start.pdb"
    with solvated_pdb.open("w") as handle:
        app.PDBFile.writeFile(modeller.topology, modeller.positions, handle, keepIds=True)
    return openmm, app, unit, forcefield, modeller, raw_pdb, solvated_pdb


def make_langevin_integrator(openmm, unit, args, timestep_fs: Optional[float] = None, temperature_k: Optional[float] = None, friction_per_ps: Optional[float] = None, seed_offset: int = 0):
    """Construct a LangevinMiddleIntegrator with reproducible seeding."""
    ts = float(args.timestep_fs if timestep_fs is None else timestep_fs)
    temp = float(args.temperature_k if temperature_k is None else temperature_k)
    friction = float(args.friction_per_ps if friction_per_ps is None else friction_per_ps)
    integ = openmm.LangevinMiddleIntegrator(
        temp * unit.kelvin,
        friction / unit.picosecond,
        ts * unit.femtosecond,
    )
    integ.setRandomNumberSeed(int(args.seed) + int(seed_offset))
    return integ


def add_position_restraints(openmm, unit, system, topology, positions, k_kcal_mol_a2: float, force_group: int = 30) -> int:
    """Restrained heavy peptide atoms; returns restrained atom count."""
    k = kcal_a2_to_kj_nm2(float(k_kcal_mol_a2))
    if k <= 0:
        return 0
    force = openmm.CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    force.addGlobalParameter("k", k)
    force.addPerParticleParameter("x0")
    force.addPerParticleParameter("y0")
    force.addPerParticleParameter("z0")
    pos_nm = positions.value_in_unit(unit.nanometer)
    n = 0
    for atom in topology.atoms():
        resname = atom.residue.name.upper()
        if resname in WATER_RESNAMES or resname in ION_RESNAMES:
            continue
        if atom.element is not None and atom.element.symbol == "H":
            continue
        x, y, z = pos_nm[atom.index]
        force.addParticle(atom.index, [float(x), float(y), float(z)])
        n += 1
    force.setForceGroup(int(force_group))
    system.addForce(force)
    return n


def make_trajectory_reporter(app, base_path: Path, interval: int, args, atom_subset=None):
    """Create a trajectory reporter using the requested trajectory format.

    ``base_path`` should be a path without a suffix.  DCD remains the default
    for backward compatibility; XTC can be requested with ``--traj-format xtc``.
    The function deliberately returns None for format ``none`` so callers can
    keep the same reporter‑append pattern.

    ``atom_subset`` (ascending list of atom indices, e.g. from
    :func:`gareus.cv.solute_atom_indices`) records only those atoms — used by
    ``--traj-solute-only`` to keep dense-sampling trajectories tiny. OpenMM's
    own ``app.DCDReporter``/``app.XTCReporter`` never accept an ``atomSubset``
    kwarg (that's an ``mdtraj.reporters`` feature, in any OpenMM version), so
    when a subset is requested we dispatch to ``mdtraj.reporters`` instead. A
    companion ``solute_only.pdb`` topology must be written for analysis to
    read the file.
    """
    interval = int(interval)
    if interval <= 0:
        return None
    fmt = str(getattr(args, "traj_format", "dcd") or "dcd").strip().lower()
    if fmt in {"none", "off", "disabled"}:
        return None
    base_path = Path(base_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)
    subset = list(atom_subset) if atom_subset is not None else None

    if subset is not None:
        try:
            import mdtraj.reporters as mdtraj_reporters
        except ImportError as exc:
            raise RuntimeError(
                "--traj-solute-only requires mdtraj (for mdtraj.reporters.DCDReporter/"
                "XTCReporter, which support atomSubset); OpenMM's own reporters do not."
            ) from exc
        if fmt == "dcd":
            return mdtraj_reporters.DCDReporter(str(base_path.with_suffix(".dcd")), interval, atomSubset=subset)
        if fmt == "xtc":
            return mdtraj_reporters.XTCReporter(str(base_path.with_suffix(".xtc")), interval, atomSubset=subset)
        raise ValueError(f"Unsupported --traj-format {fmt!r}; use dcd, xtc, or none")

    if fmt == "dcd":
        return app.DCDReporter(str(base_path.with_suffix(".dcd")), interval)
    if fmt == "xtc":
        xtc_reporter = getattr(app, "XTCReporter", None)
        if xtc_reporter is None:
            raise RuntimeError(
                "--traj-format xtc was requested, but this OpenMM installation does not expose "
                "openmm.app.XTCReporter. Upgrade OpenMM or use --traj-format dcd."
            )
        return xtc_reporter(str(base_path.with_suffix(".xtc")), interval)
    raise ValueError(f"Unsupported --traj-format {fmt!r}; use dcd, xtc, or none")


def write_solute_only_pdb(path: Path, app, topology, positions, solute_indices) -> Path:
    """Write a PDB at ``path`` containing only ``solute_indices`` (ascending).

    Deleting every non-solute atom preserves the kept atoms in ascending topology
    order, which matches an ``atomSubset``-restricted reporter's output exactly, so
    a trimmed trajectory (or a peptide-only seed-bank frame) can be read against
    this PDB as its topology.
    """
    keep = {int(i) for i in solute_indices}
    modeller = app.Modeller(topology, positions)
    to_delete = [a for a in topology.atoms() if int(a.index) not in keep]
    modeller.delete(to_delete)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        app.PDBFile.writeFile(modeller.topology, modeller.positions, handle, keepIds=True)
    return path


def write_solute_only_topology_pdb(out_dir: Path, app, topology, positions, solute_indices) -> Path:
    """Write ``<out_dir>/solute_only.pdb`` containing only ``solute_indices`` (ascending).

    Companion topology for ``--traj-solute-only`` trajectories. Thin wrapper over
    :func:`write_solute_only_pdb` that fixes the conventional filename.
    """
    return write_solute_only_pdb(Path(out_dir) / "solute_only.pdb", app, topology, positions, solute_indices)


def write_state_pdb(path: Path, app, topology, positions):
    """Write positions to a PDB file."""
    with Path(path).open("w") as handle:
        app.PDBFile.writeFile(topology, positions, handle, keepIds=True)


def run_steps_safely(
    sim,
    nsteps: int,
    label: str,
    out_dir: Path,
    app,
    topology,
    unit,
    chunk_size: int = 100,
    progress: Optional[GuiProgressSink] = None,
    progress_total: Optional[int] = None,
    progress_offset: int = 0,
    timestep_fs: Optional[float] = None,
    n_replicas: int = 1,
    message: str = "",
):
    """Run simulation steps in small chunks so crash PDBs contain the last finite coordinates."""
    nsteps = int(nsteps)
    if nsteps <= 0:
        return
    done = 0
    last_positions = None
    if progress is not None:
        progress.progress(label, progress_offset, progress_total or nsteps, message=message, timestep_fs=timestep_fs, n_replicas=n_replicas, force=True)
    while done < nsteps:
        chunk = min(int(chunk_size), nsteps - done)
        try:
            state = sim.context.getState(getPositions=True, enforcePeriodicBox=True)
            last_positions = state.getPositions()
            sim.step(int(chunk))
            done += chunk
            if progress is not None:
                progress.progress(
                    label,
                    progress_offset + done,
                    progress_total or nsteps,
                    message=message,
                    timestep_fs=timestep_fs,
                    n_replicas=n_replicas,
                )
        except Exception:
            crash_path = Path(out_dir) / f"CRASH_{label}_before_nan_step_{done}.pdb"
            if last_positions is not None:
                try:
                    write_state_pdb(crash_path, app, topology, last_positions)
                    print(f"WARNING: wrote last finite coordinates before {label} crash to {crash_path}")
                except Exception as exc:
                    print(f"WARNING: failed to write crash coordinates for {label}: {exc}")
            if progress is not None:
                progress.emit({"event": "error", "phase": label, "step": progress_offset + done, "message": str(sys.exc_info()[1])})
            raise
    if progress is not None:
        progress.progress(label, progress_offset + done, progress_total or nsteps, message=message, timestep_fs=timestep_fs, n_replicas=n_replicas, force=True)


def minimize_and_npt_equilibrate(args, out_dir: Path, progress: Optional[GuiProgressSink] = None):
    """Minimize and equilibrate a solvated system using restrained NVT and NPT phases."""
    # Build and solvate the peptide system
    openmm, app, unit, forcefield, modeller, raw_pdb, solvated_pdb = prepare_solvated_system(args, out_dir)
    platform, props = setup_platform_and_properties(openmm, args)

    print(f"[setup] OpenMM setup platform: {platform_summary(platform, props)}")
    print(f"[1/4] Generated peptide: {raw_pdb}")
    print(f"      sequence check: {'-'.join(read_pdb_residue_sequence(raw_pdb))}")
    print(f"      solvated system: {solvated_pdb}")

    # Stage 1: minimize and NVT without barostat.
    system_nvt = create_system(app, unit, forcefield, modeller.topology, args, include_barostat=False)
    restrained = add_position_restraints(
        openmm, unit, system_nvt, modeller.topology, modeller.positions,
        float(args.equil_restraint_k_kcal_mol_a2),
    )
    integrator_nvt = make_langevin_integrator(
        openmm, unit, args,
        timestep_fs=float(args.nvt_warmup_timestep_fs),
        temperature_k=float(args.nvt_start_temperature_k),
        friction_per_ps=float(args.equil_friction_per_ps),
        seed_offset=11,
    )
    sim_nvt = app.Simulation(modeller.topology, system_nvt, integrator_nvt, platform, props)
    sim_nvt.context.setPositions(modeller.positions)

    print(f"[2/4] Minimizing without barostat ({restrained} restrained peptide heavy atoms)...")
    sim_nvt.minimizeEnergy(maxIterations=int(args.minimize_iterations))
    minimized_state = sim_nvt.context.getState(getPositions=True, getEnergy=True, enforcePeriodicBox=True)
    write_state_pdb(out_dir / "02_minimized.pdb", app, modeller.topology, minimized_state.getPositions())

    # Restrained NVT warmup: gentle temperature ramp
    if int(args.nvt_warmup_steps) > 0:
        print(f"[3a/4] Restrained NVT warmup: {args.nvt_warmup_steps} steps at {args.nvt_warmup_timestep_fs} fs")
        chunks = [
            (float(args.nvt_start_temperature_k), max(1, int(args.nvt_warmup_steps) // 3)),
            (0.5 * (float(args.nvt_start_temperature_k) + float(args.temperature_k)), max(1, int(args.nvt_warmup_steps) // 3)),
            (float(args.temperature_k), int(args.nvt_warmup_steps) - 2 * max(1, int(args.nvt_warmup_steps) // 3)),
        ]
        sim_nvt.context.setVelocitiesToTemperature(chunks[0][0] * unit.kelvin, int(args.seed) + 101)
        for temp_k, nsteps in chunks:
            if nsteps <= 0:
                continue
            try:
                sim_nvt.integrator.setTemperature(float(temp_k) * unit.kelvin)
            except Exception:
                pass
            run_steps_safely(
                sim_nvt, int(nsteps), f"nvt_warmup_{int(temp_k)}K",
                out_dir, app, modeller.topology, unit,
                chunk_size=int(args.equil_safe_chunk_steps),
                progress=progress,
                progress_total=int(args.nvt_warmup_steps),
                progress_offset=sum(c[1] for c in chunks[:chunks.index((temp_k, nsteps))]) if (temp_k, nsteps) in chunks else 0,
                timestep_fs=float(args.nvt_warmup_timestep_fs),
                message=f"NVT warmup at {temp_k:g} K",
                n_replicas=1,
            )

    nvt_state = sim_nvt.context.getState(
        getPositions=True, getVelocities=True, getEnergy=True, enforcePeriodicBox=True,
    )
    write_state_pdb(out_dir / "02_nvt_warm.pdb", app, modeller.topology, nvt_state.getPositions())

    # Stage 2: restrained NPT ramp with conservative timestep
    system_npt_ramp = create_system(app, unit, forcefield, modeller.topology, args, include_barostat=True)
    add_position_restraints(
        openmm, unit, system_npt_ramp, modeller.topology, nvt_state.getPositions(),
        float(args.equil_restraint_k_kcal_mol_a2),
    )
    integrator_npt_ramp = make_langevin_integrator(
        openmm, unit, args,
        timestep_fs=float(args.npt_ramp_timestep_fs),
        temperature_k=float(args.temperature_k),
        friction_per_ps=float(args.equil_friction_per_ps),
        seed_offset=22,
    )
    sim_npt = app.Simulation(modeller.topology, system_npt_ramp, integrator_npt_ramp, platform, props)
    sim_npt.context.setPeriodicBoxVectors(*nvt_state.getPeriodicBoxVectors())
    sim_npt.context.setPositions(nvt_state.getPositions())
    sim_npt.context.setVelocities(nvt_state.getVelocities())

    equil_log = out_dir / "02_npt_equil.csv"
    sim_npt.reporters.append(app.StateDataReporter(
        str(equil_log), max(1, int(args.report_interval)),
        step=True, potentialEnergy=True, kineticEnergy=True,
        temperature=True, volume=True, density=True, speed=True, separator=",",
    ))

    ramp_steps = min(int(args.npt_ramp_steps), int(args.npt_steps))
    if ramp_steps > 0:
        print(f"[3b/4] Restrained NPT ramp: {ramp_steps} steps at {args.npt_ramp_timestep_fs} fs")
        run_steps_safely(
            sim_npt, ramp_steps, "npt_ramp", out_dir, app, modeller.topology, unit,
            chunk_size=int(args.equil_safe_chunk_steps),
            progress=progress,
            progress_total=ramp_steps,
            timestep_fs=float(args.npt_ramp_timestep_fs),
            message="restrained NPT ramp",
            n_replicas=1,
        )

    ramp_state = sim_npt.context.getState(
        getPositions=True, getVelocities=True, getEnergy=True, enforcePeriodicBox=True,
    )

    # Stage 3: unrestrained NPT continuation.  Keep timestep conservative during ramp.
    npt_steps = int(args.npt_steps)
    if npt_steps > ramp_steps:
        npt_final_timestep_fs = float(getattr(args, "npt_final_timestep_fs", 0.0) or 0.0)
        if npt_final_timestep_fs <= 0.0:
            npt_final_timestep_fs = min(float(args.timestep_fs), 2.0)
        integrator_npt = make_langevin_integrator(
            openmm, unit, args,
            timestep_fs=npt_final_timestep_fs,
            temperature_k=float(args.temperature_k),
            friction_per_ps=float(args.equil_friction_per_ps),
            seed_offset=33,
        )
        system_npt = create_system(app, unit, forcefield, modeller.topology, args, include_barostat=True)
        sim_npt_cont = app.Simulation(modeller.topology, system_npt, integrator_npt, platform, props)
        sim_npt_cont.context.setPeriodicBoxVectors(*ramp_state.getPeriodicBoxVectors())
        sim_npt_cont.context.setPositions(ramp_state.getPositions())
        sim_npt_cont.context.setVelocities(ramp_state.getVelocities())
        remaining_steps = npt_steps - ramp_steps
        print(f"[3c/4] Unrestrained NPT: {remaining_steps} steps at {npt_final_timestep_fs} fs")
        run_steps_safely(
            sim_npt_cont, remaining_steps, "npt", out_dir, app, modeller.topology, unit,
            chunk_size=int(args.equil_safe_chunk_steps),
            progress=progress,
            progress_total=remaining_steps,
            timestep_fs=npt_final_timestep_fs,
            message="unrestrained NPT",
            n_replicas=1,
        )
        final_state = sim_npt_cont.context.getState(
            getPositions=True, getVelocities=True, getEnergy=True, enforcePeriodicBox=True,
        )
    else:
        final_state = ramp_state

    write_state_pdb(out_dir / "02_npt_equilibrated.pdb", app, modeller.topology, final_state.getPositions())
    (out_dir / "03_npt_equilibrated_state.xml").write_text(
        openmm.XmlSerializer.serialize(final_state), encoding="utf-8"
    )

    # Keep the fresh-run return contract identical to
    # load_existing_openmm_setup_for_resume(): callers need the OpenMM
    # modules, force field, solvated topology, and final equilibration state
    # for window generation, seeding/grafting, GaMD setup, and production.
    return openmm, app, unit, forcefield, modeller.topology, final_state