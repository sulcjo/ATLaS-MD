"""Post-hoc peptide energy-decomposition helpers.

This module is intentionally analysis-only.  It never modifies the production
OpenMM system or any integrator state.  The core production run can therefore
skip diagnostic potential-energy sampling while still preserving the trajectory
and topology needed for later energy analysis.

The robust, portable decomposition implemented here is a direct-space
NonbondedForce pair decomposition over peptide atoms:

* intrapeptide_direct_nonbonded_kj_mol: peptide atoms within the same peptide
  group, normally the same chain.
* interpeptide_direct_nonbonded_kj_mol: peptide atoms in different peptide
  groups/chains.
* peptide_environment_direct_nonbonded_kj_mol: optional peptide vs solvent/ion
  interactions.

The direct-space terms include Lennard-Jones and Coulomb contributions using the
OpenMM NonbondedForce particle and exception parameters.  They deliberately do
not claim to be an exact PME reciprocal-space decomposition.  Optionally, the
full unbiased force-field potential of each frame can also be evaluated with a
normal OpenMM Context as an exact whole-system scalar diagnostic.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence

import numpy as np

from .constants import ION_RESNAMES, WATER_RESNAMES
from .helptext import page_text, render_encyclopedia_help
from .imports import import_openmm
from .system_setup import make_forcefield

__all__ = [
    "EnergyDecompositionConfig",
    "peptide_atom_groups_from_topology",
    "direct_nonbonded_pair_energies",
    "main",
]

COULOMB_KJ_MOL_NM_E2 = 138.935456


_ENERGY_DECOMP_HEAVY_HELP = """
Energy-decomposition method reference
=====================================

Purpose
-------
`gareus-energy-decompose` is an analysis-only command.  It re-reads saved
coordinates and a compatible topology after production and writes peptide
intra/inter direct nonbonded energy diagnostics.  It does not modify a running
simulation, alter forces, change GaMD state, or change exchange probabilities.

This command exists so production can use `--no-sample-potential-energy` to avoid
extra live diagnostic energy evaluations while still preserving a clean path to
later intrapeptide/interpeptide energy analysis.

Required inputs
---------------
Use either a completed run directory:

    gareus-energy-decompose --run-dir run_cln025

or explicit files/globs:

    gareus-energy-decompose \
      --topology run_cln025/02_npt_equilibrated.pdb \
      --trajectory 'run_cln025/replica_trajectories/*.dcd' \
      --out run_cln025/energy_decomposition.csv

Frame-resolved analysis requires saved coordinate trajectories such as DCD or
XTC.  If only `final_pdbs/*.pdb` are available, the command can analyze final
snapshots but cannot reconstruct the time series.  Do not use `--traj-format none`
for production runs if you plan to compute frame-by-frame energies later.

Grouping definitions
--------------------
The default `--group-mode chains` treats each peptide chain as one peptide group:

    intrapeptide = atom pairs within the same peptide group/chain
    interpeptide = atom pairs between different peptide groups/chains

For a single-chain peptide simulation, `interpeptide_direct_nonbonded_kj_mol` is
therefore expected to be zero.  Alternative grouping modes are:

    --group-mode all-peptide    all peptide atoms form one group
    --group-mode residues       each residue is a separate group

`all-peptide` is useful for a single total intrapeptide term.  `residues` can be
useful for residue-residue style diagnostics, but it is more expensive and the
result is a diagnostic partition, not a different force field.

Direct nonbonded equation
-------------------------
For selected atom pair i,j, the diagnostic pair term is:

    E_ij = k_e q_i q_j / r_ij
           + 4 epsilon_ij [(sigma_ij / r_ij)^12 - (sigma_ij / r_ij)^6]

with k_e = 138.935456 kJ mol^-1 nm e^-2.  Normal non-exception pairs use OpenMM
particle charges and Lorentz-Berthelot mixing:

    sigma_ij   = 0.5 (sigma_i + sigma_j)
    epsilon_ij = sqrt(epsilon_i epsilon_j)

OpenMM exception pairs use the explicit exception charge product, sigma, and
epsilon.  Exclusions are treated as zeroed exception terms.

Output columns
--------------
The CSV includes frame metadata plus these energy columns:

    intrapeptide_direct_nonbonded_kj_mol
    interpeptide_direct_nonbonded_kj_mol
    peptide_peptide_direct_nonbonded_kj_mol
    peptide_environment_direct_nonbonded_kj_mol      only populated with --include-peptide-environment
    total_forcefield_kj_mol                          only populated with --write-total-forcefield-energy

`peptide_environment_direct_nonbonded_kj_mol` covers peptide-vs-solvent/ion
direct nonbonded terms and can be much slower in solvated systems because it
contains many more pairs.

PME caveat
----------
The direct pair columns are intentionally named `direct_nonbonded`.  They are
reproducible peptide-pair diagnostics based on OpenMM NonbondedForce parameters,
but they are not an exact pair decomposition of PME reciprocal-space
electrostatics.  Exact reciprocal-space pair partitioning is not generally
available from a standard OpenMM PME energy evaluation.  Use
`--write-total-forcefield-energy` to add the exact whole-system unbiased OpenMM
potential energy as a scalar per frame.

Practical recipes
-----------------
Fast production, later decomposition:

    gareus --seq CLN025 --traj-format xtc --no-sample-potential-energy --no-flush-every-log
    gareus-energy-decompose --run-dir run_cln025

Include environment and whole-system scalar:

    gareus-energy-decompose --run-dir run_cln025 \
      --include-peptide-environment \
      --write-total-forcefield-energy

Subsample a large trajectory:

    gareus-energy-decompose --run-dir run_cln025 --stride 10 --start 1000 --stop 10000
"""


class _EnergyDecompHeavyHelpAction(argparse.Action):
    """Argparse action for the energy-decomposition method reference.

    Shares gareus.helptext's auto-numbered-TOC/pager machinery with the main
    ``gareus -hh`` so both heavy-help surfaces behave the same way: paged
    (via $PAGER, default less) when run at a terminal, plain when piped, and
    jumpable straight to one topic with ``-hh <number>`` / ``-hh <keyword>``.
    """

    def __init__(self, option_strings, dest=argparse.SUPPRESS, default=argparse.SUPPRESS, help=None):
        super().__init__(option_strings=list(option_strings), dest=dest, nargs="?", default=default, help=help, metavar="TOPIC")

    def __call__(self, parser, namespace, values, option_string=None):  # noqa: D401 - argparse signature
        page_text(render_encyclopedia_help(parser, _ENERGY_DECOMP_HEAVY_HELP, topic=values), parser)
        parser.exit(0)


@dataclass(frozen=True)
class EnergyDecompositionConfig:
    """Configuration for post-hoc peptide energy decomposition."""

    topology: Path
    trajectories: tuple[Path, ...]
    output: Path
    water_model: str = "tip3p"
    group_mode: str = "chains"
    stride: int = 1
    start: int = 0
    stop: Optional[int] = None
    include_peptide_environment: bool = False
    write_total_forcefield_energy: bool = False
    nonbonded_cutoff_nm: float = 0.8
    ewald_error_tolerance: float = 1.0e-4
    platform: str = "auto"
    direct_cutoff_nm: Optional[float] = None


def _is_peptide_residue(residue) -> bool:
    name = str(getattr(residue, "name", "")).upper()
    return name not in WATER_RESNAMES and name not in {x.upper() for x in ION_RESNAMES}


def peptide_atom_groups_from_topology(topology, mode: str = "chains") -> tuple[list[list[int]], list[int], list[int]]:
    """Return peptide atom groups, peptide atoms, and non-peptide atoms.

    Parameters
    ----------
    topology:
        OpenMM topology object.
    mode:
        ``chains`` groups peptide residues by chain, ``all-peptide`` treats all
        peptide atoms as one molecule, and ``residues`` treats every peptide
        residue as a group.  ``chains`` is the usual definition of peptide
        molecules for inter/intra-peptide decomposition.
    """

    mode = str(mode or "chains").strip().lower().replace("_", "-")
    if mode not in {"chains", "all-peptide", "residues"}:
        raise ValueError("group mode must be chains, all-peptide, or residues")

    groups: list[list[int]] = []
    peptide_atoms: list[int] = []
    peptide_atom_set: set[int] = set()

    if mode == "all-peptide":
        one_group: list[int] = []
        for residue in topology.residues():
            if not _is_peptide_residue(residue):
                continue
            for atom in residue.atoms():
                idx = int(atom.index)
                one_group.append(idx)
                peptide_atoms.append(idx)
                peptide_atom_set.add(idx)
        if one_group:
            groups.append(sorted(one_group))
    elif mode == "residues":
        for residue in topology.residues():
            if not _is_peptide_residue(residue):
                continue
            group = [int(atom.index) for atom in residue.atoms()]
            if group:
                groups.append(sorted(group))
                peptide_atoms.extend(group)
                peptide_atom_set.update(group)
    else:
        for chain in topology.chains():
            group: list[int] = []
            for residue in chain.residues():
                if not _is_peptide_residue(residue):
                    continue
                for atom in residue.atoms():
                    idx = int(atom.index)
                    group.append(idx)
                    peptide_atoms.append(idx)
                    peptide_atom_set.add(idx)
            if group:
                groups.append(sorted(group))
        if not groups:
            # Some generated topologies can have odd chain bookkeeping.  Fall
            # back to a single peptide group rather than failing mysteriously.
            return peptide_atom_groups_from_topology(topology, mode="all-peptide")

    all_atoms = [int(atom.index) for atom in topology.atoms()]
    nonpeptide_atoms = [idx for idx in all_atoms if idx not in peptide_atom_set]
    return groups, sorted(peptide_atoms), nonpeptide_atoms


def _minimum_image_delta(delta_nm: np.ndarray, box_vectors_nm: Optional[np.ndarray]) -> np.ndarray:
    """Apply a triclinic minimum-image convention to a displacement vector."""

    if box_vectors_nm is None:
        return delta_nm
    box = np.asarray(box_vectors_nm, dtype=np.float64)
    if box.shape != (3, 3) or not np.all(np.isfinite(box)):
        return delta_nm
    try:
        fractional = np.linalg.solve(box.T, np.asarray(delta_nm, dtype=np.float64))
    except np.linalg.LinAlgError:
        return delta_nm
    fractional -= np.round(fractional)
    return fractional @ box


def _quantity_value(value, target_unit) -> float:
    try:
        return float(value.value_in_unit(target_unit))
    except AttributeError:
        return float(value)


def _find_nonbonded_force(system):
    for idx in range(system.getNumForces()):
        force = system.getForce(idx)
        if force.__class__.__name__ == "NonbondedForce":
            return force
    raise RuntimeError("Could not find an OpenMM NonbondedForce in the reconstructed system.")


def _extract_nonbonded_parameters(system, unit) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[tuple[int, int], tuple[float, float, float]]]:
    """Extract charges, sigma, epsilon, and exception parameters from OpenMM."""

    nb = _find_nonbonded_force(system)
    n = int(system.getNumParticles())
    charges = np.zeros(n, dtype=np.float64)
    sigmas = np.zeros(n, dtype=np.float64)
    epsilons = np.zeros(n, dtype=np.float64)
    for i in range(n):
        charge, sigma, epsilon = nb.getParticleParameters(i)
        charges[i] = _quantity_value(charge, unit.elementary_charge)
        sigmas[i] = _quantity_value(sigma, unit.nanometer)
        epsilons[i] = _quantity_value(epsilon, unit.kilojoule_per_mole)
    exceptions: dict[tuple[int, int], tuple[float, float, float]] = {}
    for eidx in range(nb.getNumExceptions()):
        i, j, chargeprod, sigma, epsilon = nb.getExceptionParameters(eidx)
        key = tuple(sorted((int(i), int(j))))
        try:
            qprod = float(chargeprod.value_in_unit(unit.elementary_charge * unit.elementary_charge))
        except Exception:
            qprod = float(chargeprod)
        exceptions[key] = (
            qprod,
            _quantity_value(sigma, unit.nanometer),
            _quantity_value(epsilon, unit.kilojoule_per_mole),
        )
    return charges, sigmas, epsilons, exceptions


def _iter_pairs_within(groups: Sequence[Sequence[int]]) -> Iterator[tuple[int, int]]:
    for group in groups:
        ordered = list(group)
        for a_pos in range(len(ordered)):
            i = int(ordered[a_pos])
            for j in ordered[a_pos + 1 :]:
                yield i, int(j)


def _iter_pairs_between(groups: Sequence[Sequence[int]]) -> Iterator[tuple[int, int]]:
    for g1 in range(len(groups)):
        for g2 in range(g1 + 1, len(groups)):
            for i in groups[g1]:
                for j in groups[g2]:
                    yield int(i), int(j)


def _iter_cross_pairs(left: Sequence[int], right: Sequence[int]) -> Iterator[tuple[int, int]]:
    for i in left:
        for j in right:
            yield int(i), int(j)


def _direct_nonbonded_energy_for_pairs(
    positions_nm: np.ndarray,
    pairs: Iterable[tuple[int, int]],
    charges: np.ndarray,
    sigmas: np.ndarray,
    epsilons: np.ndarray,
    exceptions: dict[tuple[int, int], tuple[float, float, float]],
    box_vectors_nm: Optional[np.ndarray] = None,
    cutoff_nm: Optional[float] = None,
) -> float:
    energy = 0.0
    cutoff2 = None if cutoff_nm is None or cutoff_nm <= 0 else float(cutoff_nm) ** 2
    pos = np.asarray(positions_nm, dtype=np.float64)
    for i, j in pairs:
        key = tuple(sorted((int(i), int(j))))
        delta = _minimum_image_delta(pos[int(j)] - pos[int(i)], box_vectors_nm)
        r2 = float(np.dot(delta, delta))
        if r2 <= 0.0:
            continue
        if cutoff2 is not None and r2 > cutoff2:
            continue
        r = math.sqrt(r2)
        if key in exceptions:
            qprod, sigma, epsilon = exceptions[key]
        else:
            qprod = float(charges[int(i)] * charges[int(j)])
            sigma = 0.5 * float(sigmas[int(i)] + sigmas[int(j)])
            epsilon = math.sqrt(max(0.0, float(epsilons[int(i)] * epsilons[int(j)])))
        coulomb = COULOMB_KJ_MOL_NM_E2 * qprod / r
        if epsilon == 0.0 or sigma == 0.0:
            lj = 0.0
        else:
            sr = sigma / r
            sr6 = sr ** 6
            lj = 4.0 * epsilon * (sr6 * sr6 - sr6)
        energy += coulomb + lj
    return float(energy)


def direct_nonbonded_pair_energies(
    positions_nm: np.ndarray,
    peptide_groups: Sequence[Sequence[int]],
    nonpeptide_atoms: Sequence[int],
    charges: np.ndarray,
    sigmas: np.ndarray,
    epsilons: np.ndarray,
    exceptions: dict[tuple[int, int], tuple[float, float, float]],
    box_vectors_nm: Optional[np.ndarray] = None,
    include_peptide_environment: bool = False,
    cutoff_nm: Optional[float] = None,
) -> dict[str, float]:
    """Compute direct-space peptide nonbonded decomposition for one frame."""

    peptide_atoms = sorted({int(i) for group in peptide_groups for i in group})
    intra = _direct_nonbonded_energy_for_pairs(
        positions_nm, _iter_pairs_within(peptide_groups), charges, sigmas, epsilons, exceptions,
        box_vectors_nm=box_vectors_nm, cutoff_nm=cutoff_nm,
    )
    inter = _direct_nonbonded_energy_for_pairs(
        positions_nm, _iter_pairs_between(peptide_groups), charges, sigmas, epsilons, exceptions,
        box_vectors_nm=box_vectors_nm, cutoff_nm=cutoff_nm,
    )
    peptide_env = math.nan
    if include_peptide_environment:
        peptide_env = _direct_nonbonded_energy_for_pairs(
            positions_nm, _iter_cross_pairs(peptide_atoms, nonpeptide_atoms), charges, sigmas, epsilons, exceptions,
            box_vectors_nm=box_vectors_nm, cutoff_nm=cutoff_nm,
        )
    return {
        "intrapeptide_direct_nonbonded_kj_mol": float(intra),
        "interpeptide_direct_nonbonded_kj_mol": float(inter),
        "peptide_environment_direct_nonbonded_kj_mol": float(peptide_env),
        "peptide_peptide_direct_nonbonded_kj_mol": float(intra + inter),
    }


def _expand_trajectory_paths(run_dir: Optional[Path], patterns: Sequence[str]) -> tuple[Path, ...]:
    search_patterns: list[str] = []
    if patterns:
        search_patterns.extend(str(p) for p in patterns)
    elif run_dir is not None:
        search_patterns.extend([
            str(run_dir / "replica_trajectories" / "*.dcd"),
            str(run_dir / "replica_trajectories" / "*.xtc"),
            str(run_dir / "final_pdbs" / "*.pdb"),
        ])
    paths: list[Path] = []
    for pattern in search_patterns:
        expanded = sorted(glob.glob(pattern))
        if expanded:
            paths.extend(Path(x) for x in expanded)
        else:
            path = Path(pattern)
            if path.exists():
                paths.append(path)
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        resolved = str(path.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return tuple(unique)


def _infer_topology_path(run_dir: Optional[Path], topology: Optional[str]) -> Path:
    if topology:
        path = Path(topology)
        if not path.exists():
            raise FileNotFoundError(f"Topology file does not exist: {path}")
        return path
    if run_dir is None:
        raise ValueError("Provide --topology or --run-dir so the topology PDB can be inferred.")
    for name in ("02_npt_equilibrated.pdb", "01_solvated_start.pdb", "00_built_peptide.pdb"):
        path = run_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Could not infer topology under {run_dir}; expected 02_npt_equilibrated.pdb, "
        "01_solvated_start.pdb, or 00_built_peptide.pdb."
    )


def _infer_output_path(run_dir: Optional[Path], output: Optional[str]) -> Path:
    if output:
        return Path(output)
    if run_dir is not None:
        return run_dir / "energy_decomposition.csv"
    return Path("energy_decomposition.csv")


def _frame_label(path: Path, local_frame: int) -> tuple[str, int]:
    name = path.name
    match = re.search(r"replica_(\d+)", name)
    replica = int(match.group(1)) if match else -1
    return f"{name}:{local_frame}", replica


def _box_vectors_from_mdtraj_frame(frame) -> Optional[np.ndarray]:
    vectors = getattr(frame, "unitcell_vectors", None)
    if vectors is None:
        return None
    arr = np.asarray(vectors, dtype=np.float64)
    if arr.ndim == 3 and arr.shape[0] >= 1:
        return arr[0]
    if arr.shape == (3, 3):
        return arr
    return None


def _iter_frames(paths: Sequence[Path], topology_path: Path, stride: int, start: int, stop: Optional[int]) -> Iterator[tuple[Path, int, np.ndarray, Optional[np.ndarray]]]:
    """Yield path, frame index, positions in nm, and optional box vectors in nm."""

    _openmm, app, unit = import_openmm()
    del _openmm
    stride = max(1, int(stride or 1))
    start = max(0, int(start or 0))
    global_index = 0
    for path in paths:
        suffix = path.suffix.lower()
        if suffix == ".pdb":
            if global_index >= start and (stop is None or global_index < stop):
                pdb = app.PDBFile(str(path))
                yield path, 0, np.asarray(pdb.positions.value_in_unit(unit.nanometer), dtype=np.float64), None
            global_index += 1
            continue
        try:
            import mdtraj as md  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                f"Reading {suffix} trajectories requires mdtraj. Install the package analysis extra or "
                "run the decomposition on PDB snapshots/final_pdbs instead."
            ) from exc
        traj = md.load(str(path), top=str(topology_path), stride=stride)
        for local_idx in range(traj.n_frames):
            if global_index < start:
                global_index += 1
                continue
            if stop is not None and global_index >= stop:
                return
            xyz_nm = np.asarray(traj.xyz[local_idx], dtype=np.float64)
            frame = traj[local_idx]
            yield path, local_idx, xyz_nm, _box_vectors_from_mdtraj_frame(frame)
            global_index += 1


def _make_forcefield_system(app, unit, topology, cfg: EnergyDecompositionConfig):
    ff = make_forcefield(app, cfg.water_model)
    nonbonded_cutoff = float(cfg.nonbonded_cutoff_nm) * unit.nanometer
    return ff.createSystem(
        topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=nonbonded_cutoff,
        constraints=app.HBonds,
        rigidWater=True,
        ewaldErrorTolerance=float(cfg.ewald_error_tolerance),
    )


def _create_total_energy_context(openmm, app, unit, topology, system, cfg: EnergyDecompositionConfig):
    integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)
    if str(cfg.platform).lower() == "auto":
        return app.Simulation(topology, system, integrator)
    platform = openmm.Platform.getPlatformByName(str(cfg.platform))
    return app.Simulation(topology, system, integrator, platform)


def _set_context_positions_and_box(sim, positions_nm: np.ndarray, box_vectors_nm: Optional[np.ndarray], openmm, unit) -> None:
    sim.context.setPositions([openmm.Vec3(float(x), float(y), float(z)) for x, y, z in positions_nm] * unit.nanometer)
    if box_vectors_nm is not None:
        vecs = [openmm.Vec3(float(v[0]), float(v[1]), float(v[2])) for v in np.asarray(box_vectors_nm, dtype=np.float64)]
        sim.context.setPeriodicBoxVectors(vecs[0] * unit.nanometer, vecs[1] * unit.nanometer, vecs[2] * unit.nanometer)


def run_energy_decomposition(cfg: EnergyDecompositionConfig) -> Path:
    """Run post-hoc energy decomposition and write a CSV file."""

    openmm, app, unit = import_openmm()
    topology_pdb = app.PDBFile(str(cfg.topology))
    topology = topology_pdb.topology
    system = _make_forcefield_system(app, unit, topology, cfg)
    charges, sigmas, epsilons, exceptions = _extract_nonbonded_parameters(system, unit)
    peptide_groups, peptide_atoms, nonpeptide_atoms = peptide_atom_groups_from_topology(topology, mode=cfg.group_mode)
    if not peptide_groups:
        raise RuntimeError("No peptide atoms were identified in the topology.")

    total_sim = None
    if cfg.write_total_forcefield_energy:
        total_sim = _create_total_energy_context(openmm, app, unit, topology, system, cfg)

    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "frame_source",
        "source_file",
        "source_frame",
        "replica",
        "n_peptide_groups",
        "n_peptide_atoms",
        "n_nonpeptide_atoms",
        "intrapeptide_direct_nonbonded_kj_mol",
        "interpeptide_direct_nonbonded_kj_mol",
        "peptide_peptide_direct_nonbonded_kj_mol",
        "peptide_environment_direct_nonbonded_kj_mol",
        "total_forcefield_kj_mol",
        "decomposition_note",
    ]
    with cfg.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for path, local_frame, positions_nm, box_vectors_nm in _iter_frames(cfg.trajectories, cfg.topology, cfg.stride, cfg.start, cfg.stop):
            label, replica = _frame_label(path, local_frame)
            decomp = direct_nonbonded_pair_energies(
                positions_nm,
                peptide_groups,
                nonpeptide_atoms,
                charges,
                sigmas,
                epsilons,
                exceptions,
                box_vectors_nm=box_vectors_nm,
                include_peptide_environment=bool(cfg.include_peptide_environment),
                cutoff_nm=cfg.direct_cutoff_nm,
            )
            total = math.nan
            if total_sim is not None:
                _set_context_positions_and_box(total_sim, positions_nm, box_vectors_nm, openmm, unit)
                state = total_sim.context.getState(getEnergy=True)
                total = float(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
            writer.writerow({
                "frame_source": label,
                "source_file": str(path),
                "source_frame": int(local_frame),
                "replica": int(replica),
                "n_peptide_groups": int(len(peptide_groups)),
                "n_peptide_atoms": int(len(peptide_atoms)),
                "n_nonpeptide_atoms": int(len(nonpeptide_atoms)),
                **decomp,
                "total_forcefield_kj_mol": total,
                "decomposition_note": "direct-space LJ+Coulomb peptide-pair terms; PME reciprocal-space is not pair-decomposed",
            })
    return cfg.output


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gareus-energy-decompose",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Post-process saved GAREUS coordinates into peptide intrapeptide/interpeptide "
            "direct nonbonded energy components. Use -hh for equations, grouping definitions, "
            "PME caveats, and output-column semantics."
        ),
    )
    parser.add_argument("-hh", "--help-heavy", action=_EnergyDecompHeavyHelpAction, help="Show energy-decomposition method reference, equations, grouping definitions, caveats, and complete options; then exit.")
    parser.add_argument("--run-dir", default=None, help="Completed GAREUS run directory. Used to infer topology, trajectories, and output path.")
    parser.add_argument("--topology", default=None, help="Topology PDB, usually 02_npt_equilibrated.pdb or 01_solvated_start.pdb.")
    parser.add_argument("--trajectory", nargs="*", default=None, help="Trajectory/PDB path(s) or glob(s). Defaults to run-dir replica_trajectories/*.dcd/*.xtc, then final_pdbs/*.pdb.")
    parser.add_argument("--out", default=None, help="Output CSV path. Defaults to run-dir/energy_decomposition.csv.")
    parser.add_argument("--water-model", choices=["tip3p", "tip3pfb", "spce", "tip4pew"], default="tip3p", help="Water model XML used to reconstruct the unbiased OpenMM system.")
    parser.add_argument("--group-mode", choices=["chains", "all-peptide", "residues"], default="chains", help="How peptide molecules/groups are defined for intra/inter terms.")
    parser.add_argument("--stride", type=int, default=1, help="Read every Nth frame for DCD/XTC trajectories. PDB snapshots are one frame each.")
    parser.add_argument("--start", type=int, default=0, help="First global frame index to analyze.")
    parser.add_argument("--stop", type=int, default=None, help="Stop before this global frame index.")
    parser.add_argument("--include-peptide-environment", action="store_true", help="Also compute peptide-vs-solvent/ion direct nonbonded energy. Can be much slower.")
    parser.add_argument("--write-total-forcefield-energy", action="store_true", help="Also evaluate exact whole-system unbiased OpenMM force-field potential for each frame.")
    parser.add_argument("--nonbonded-cutoff-nm", type=float, default=0.8, help="Cutoff used when reconstructing the total PME force-field system.")
    parser.add_argument("--ewald-error-tolerance", type=float, default=1.0e-4, help="PME tolerance used for optional total force-field energy.")
    parser.add_argument("--platform", default="auto", help="OpenMM platform for optional total force-field energy: auto, CPU, CUDA, OpenCL, HIP, or Reference.")
    parser.add_argument("--direct-cutoff-nm", type=float, default=None, help="Optional cutoff for direct-space pair decomposition. Default uses all selected peptide-pair distances.")
    return parser


def parse_args(argv: Optional[Iterable[str]] = None) -> EnergyDecompositionConfig:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    run_dir = Path(args.run_dir) if args.run_dir else None
    topology = _infer_topology_path(run_dir, args.topology)
    trajectories = _expand_trajectory_paths(run_dir, args.trajectory or [])
    if not trajectories:
        raise SystemExit("No trajectory/PDB inputs found. Provide --trajectory or a --run-dir with saved coordinates.")
    output = _infer_output_path(run_dir, args.out)
    return EnergyDecompositionConfig(
        topology=topology,
        trajectories=trajectories,
        output=output,
        water_model=str(args.water_model),
        group_mode=str(args.group_mode),
        stride=int(args.stride),
        start=int(args.start),
        stop=args.stop,
        include_peptide_environment=bool(args.include_peptide_environment),
        write_total_forcefield_energy=bool(args.write_total_forcefield_energy),
        nonbonded_cutoff_nm=float(args.nonbonded_cutoff_nm),
        ewald_error_tolerance=float(args.ewald_error_tolerance),
        platform=str(args.platform),
        direct_cutoff_nm=None if args.direct_cutoff_nm is None else float(args.direct_cutoff_nm),
    )


def main(argv: Optional[Iterable[str]] = None) -> int:
    cfg = parse_args(argv)
    path = run_energy_decomposition(cfg)
    print(f"Energy decomposition written to {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
