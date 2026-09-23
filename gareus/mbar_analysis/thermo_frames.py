"""Offline peptide-only PME energies of saved frames, for the exact pp/pe split.

Spec: docs/superpowers/specs/2026-09-23-thermo-energy-decomposition/spec.md §11 (phase 2).

Ewald/PME electrostatics is a quadratic form in the charges, E = 1/2 q^T A q, so with
q = q_pep + q_env the nonbonded energy splits exactly into pp + pe + ee for one alpha and one grid;
LJ pairs split the same way. The production kernel records v_pep = E0 - E1 + E2 = bonded_pep +
NB_pp + NB_pe + E_dih (+ the dispersion-correction difference). A System holding ONLY the peptide
atoms, in the same box, with the physical force's alpha and grid, gives

    V_pp = bonded_pep + E_dih + NB_pp          (peptide with its own periodic images)
    V_pe = v_pep - V_pp                         (peptide-environment nonbonded, + LRC difference)

Only peptide coordinates and the box are needed, which is what a solute-only trajectory stores.
The dispersion correction is OFF in the peptide-only System (convention: it stays in V_pe).
Every physical term with atoms both inside and outside the peptide is refused: the split would
otherwise be undefined.
"""
from __future__ import annotations

import math
from typing import Iterable, Optional

import numpy as np

#: Force groups inside the peptide-only System.
NB_GROUP, BONDED_GROUP, DIHEDRAL_GROUP = 0, 1, 2
AUX_FORCE_NAME = "PepGaMDWaterOnlyNonbonded"


def _physical_nonbonded(system, openmm):
    found = [f for f in system.getForces()
             if isinstance(f, openmm.NonbondedForce) and f.getName() != AUX_FORCE_NAME]
    if len(found) != 1:
        raise ValueError(f"need exactly one physical NonbondedForce, found {len(found)}")
    return found[0]


def resolved_pme_parameters(system, openmm, unit) -> tuple[float, int, int, int]:
    """alpha (1/nm) and grid of the physical force, as production pins them.

    Production calls ``pep_gamd._pin_pme_parameters``: explicit parameters when set, else OpenMM's
    tolerance formula on the System's default box. Reproduce exactly that.
    """
    from gareus.pep_gamd import pme_parameters_from_tolerance
    nb = _physical_nonbonded(system, openmm)
    alpha, nx, ny, nz = nb.getPMEParameters()
    a = alpha.value_in_unit(unit.nanometer ** -1) if hasattr(alpha, "value_in_unit") else float(alpha)
    if a > 0.0:
        return float(a), int(nx), int(ny), int(nz)
    cutoff = float(nb.getCutoffDistance().value_in_unit(unit.nanometer))
    box = [[float(v[i].value_in_unit(unit.nanometer)) for i in range(3)] for v in system.getDefaultPeriodicBoxVectors()]
    return pme_parameters_from_tolerance(cutoff, float(nb.getEwaldErrorTolerance()), box)


def build_peptide_only_system(system, peptide_atoms: Iterable[int], openmm, unit, *, pme=None):
    """Peptide-only System in the full System's box. Returns (system, index_map, pme_params).

    ``index_map[old] = new`` for peptide atoms, new indices in sorted order of old ones.
    """
    pep = sorted(int(i) for i in peptide_atoms)
    if not pep:
        raise ValueError("empty peptide atom set")
    new_of = {old: k for k, old in enumerate(pep)}
    inside = set(pep)
    out = openmm.System()
    for old in pep:
        out.addParticle(system.getParticleMass(old))
    out.setDefaultPeriodicBoxVectors(*system.getDefaultPeriodicBoxVectors())

    def _all_in(atoms):
        n_in = sum(a in inside for a in atoms)
        if 0 < n_in < len(atoms):
            raise ValueError(f"a physical term couples peptide and non-peptide atoms {tuple(atoms)}")
        return n_in == len(atoms)

    phys_nb = _physical_nonbonded(system, openmm)
    alpha, nx, ny, nz = pme if pme is not None else resolved_pme_parameters(system, openmm, unit)
    bonds = openmm.HarmonicBondForce(); angles = openmm.HarmonicAngleForce(); tors = openmm.PeriodicTorsionForce()
    cmap = None
    for f in system.getForces():
        if isinstance(f, openmm.HarmonicBondForce):
            for i in range(f.getNumBonds()):
                a, b, r0, k = f.getBondParameters(i)
                if _all_in((a, b)):
                    bonds.addBond(new_of[a], new_of[b], r0, k)
        elif isinstance(f, openmm.HarmonicAngleForce):
            for i in range(f.getNumAngles()):
                a, b, c, th, k = f.getAngleParameters(i)
                if _all_in((a, b, c)):
                    angles.addAngle(new_of[a], new_of[b], new_of[c], th, k)
        elif isinstance(f, openmm.PeriodicTorsionForce):
            for i in range(f.getNumTorsions()):
                a, b, c, d, per, ph, k = f.getTorsionParameters(i)
                if _all_in((a, b, c, d)):
                    tors.addTorsion(new_of[a], new_of[b], new_of[c], new_of[d], per, ph, k)
        elif isinstance(f, openmm.CMAPTorsionForce):
            if cmap is None:
                cmap = openmm.CMAPTorsionForce()
            offset = cmap.getNumMaps()  # every CMAP force's maps kept, re-indexed into one force
            for m in range(f.getNumMaps()):
                size, energy = f.getMapParameters(m)
                cmap.addMap(size, energy)
            for i in range(f.getNumTorsions()):
                m, *atoms = f.getTorsionParameters(i)
                if _all_in(atoms):
                    cmap.addTorsion(offset + m, *[new_of[a] for a in atoms])
        elif isinstance(f, openmm.NonbondedForce) or isinstance(f, (openmm.CMMotionRemover, openmm.MonteCarloBarostat)):
            continue
        elif isinstance(f, openmm.CustomCVForce) or f.getForceGroup() >= 3:
            continue  # umbrella/CV bias forces: not physical energy
        else:
            raise ValueError(f"unsupported physical force {type(f).__name__} in group {f.getForceGroup()}")
    nb = openmm.NonbondedForce()
    nb.setNonbondedMethod(openmm.NonbondedForce.PME)
    nb.setCutoffDistance(phys_nb.getCutoffDistance())
    nb.setEwaldErrorTolerance(phys_nb.getEwaldErrorTolerance())
    nb.setPMEParameters(alpha, nx, ny, nz)
    nb.setUseSwitchingFunction(phys_nb.getUseSwitchingFunction())
    if phys_nb.getUseSwitchingFunction():
        nb.setSwitchingDistance(phys_nb.getSwitchingDistance())
    nb.setUseDispersionCorrection(False)
    nb.setExceptionsUsePeriodicBoundaryConditions(phys_nb.getExceptionsUsePeriodicBoundaryConditions())
    if phys_nb.getNumParticleParameterOffsets() or phys_nb.getNumExceptionParameterOffsets():
        raise ValueError("physical NonbondedForce uses parameter offsets: unsupported")
    for old in pep:
        q, s, e = phys_nb.getParticleParameters(old)
        nb.addParticle(q, s, e)
    for i in range(phys_nb.getNumExceptions()):
        a, b, qq, s, e = phys_nb.getExceptionParameters(i)
        if _all_in((a, b)):
            nb.addException(new_of[a], new_of[b], qq, s, e)
    nb.setForceGroup(NB_GROUP)
    for f in (bonds, angles):
        f.setForceGroup(BONDED_GROUP)
        out.addForce(f)
    tors.setForceGroup(DIHEDRAL_GROUP); out.addForce(tors)
    if cmap is not None:
        cmap.setForceGroup(DIHEDRAL_GROUP); out.addForce(cmap)
    out.addForce(nb)
    return out, new_of, (float(alpha), int(nx), int(ny), int(nz))


class PeptideOnlyEvaluator:
    """Single-point peptide-only energies for many frames on one Context."""

    def __init__(self, system, peptide_atoms, openmm, unit, *, platform: str = "CPU", pme=None):
        self.openmm, self.unit = openmm, unit
        self.system, self.index_map, self.pme = build_peptide_only_system(system, peptide_atoms, openmm, unit, pme=pme)
        integ = openmm.VerletIntegrator(0.001 * unit.picoseconds)
        self.context = openmm.Context(self.system, integ, openmm.Platform.getPlatformByName(platform))
        self.n_atoms = self.system.getNumParticles()

    def evaluate(self, positions_nm: np.ndarray, box_nm: Optional[np.ndarray]) -> tuple[float, float, float]:
        """(E_nb, E_bonded, E_dih) in kJ/mol for peptide coordinates (n_pep, 3) nm and a (3, 3) box."""
        openmm, unit = self.openmm, self.unit
        if box_nm is not None:
            b = np.asarray(box_nm, dtype=float)
            self.context.setPeriodicBoxVectors(*(openmm.Vec3(*b[i]) * unit.nanometer for i in range(3)))
        self.context.setPositions(np.asarray(positions_nm, dtype=float) * unit.nanometer)
        e = []
        for g in (NB_GROUP, BONDED_GROUP, DIHEDRAL_GROUP):
            st = self.context.getState(getEnergy=True, groups={g})
            e.append(float(st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)))
        return e[0], e[1], e[2]


def backbone_torsion_quads(topology, peptide_atoms: Iterable[int]) -> tuple[list[tuple[int, int, int, int]], list[str]]:
    """(phi, psi) atom quadruples over the peptide's residues, in topology atom indices, with labels.

    phi_i = C(i-1)-N(i)-CA(i)-C(i); psi_i = N(i)-CA(i)-C(i)-N(i+1). Terminal torsions without a
    partner residue are skipped (caps such as ACE/NME are fine as long as they carry C/N).
    """
    inside = set(int(i) for i in peptide_atoms)
    residues = [r for r in topology.residues() if any(a.index in inside for a in r.atoms())]
    names = [{a.name: a.index for a in r.atoms()} for r in residues]
    quads, labels = [], []
    for k, r in enumerate(residues):
        cur = names[k]
        if k > 0 and all(x in cur for x in ("N", "CA", "C")) and "C" in names[k - 1]:
            quads.append((names[k - 1]["C"], cur["N"], cur["CA"], cur["C"])); labels.append(f"phi-{r.name}{r.id}")
        if k + 1 < len(residues) and all(x in cur for x in ("N", "CA", "C")) and "N" in names[k + 1]:
            quads.append((cur["N"], cur["CA"], cur["C"], names[k + 1]["N"])); labels.append(f"psi-{r.name}{r.id}")
    return quads, labels


def dihedral_angles(xyz_nm: np.ndarray, quads) -> np.ndarray:
    """(n_frames, n_quads) dihedrals in radians from (n_frames, n_atoms, 3) coordinates (IUPAC sign)."""
    xyz = np.asarray(xyz_nm, dtype=np.float64)
    q = np.asarray(quads, dtype=np.int64)
    p0, p1, p2, p3 = (xyz[:, q[:, i], :] for i in range(4))
    b0 = p0 - p1; b1 = p2 - p1; b2 = p3 - p2
    b1n = b1 / np.linalg.norm(b1, axis=-1, keepdims=True)
    v = b0 - np.sum(b0 * b1n, axis=-1, keepdims=True) * b1n
    w = b2 - np.sum(b2 * b1n, axis=-1, keepdims=True) * b1n
    x = np.sum(v * w, axis=-1)
    y = np.sum(np.cross(b1n, v) * w, axis=-1)
    return np.arctan2(y, x)


def peptide_energy_oracle_ok(e_dih_offline, v_dih_recorded, tol_kj: float = 2.0) -> dict:
    """Frame-alignment check: offline torsion energy from saved coordinates vs the recorded v_dih.

    A frame matched to the wrong sample differs by ~sigma(v_dih) (tens of kJ/mol); a correctly
    matched frame differs only by coordinate rounding (XTC ~1e-3 nm).
    """
    d = np.asarray(e_dih_offline, dtype=float) - np.asarray(v_dih_recorded, dtype=float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return {"ok": False, "n": 0, "median_abs_kj": float("nan"), "p99_abs_kj": float("nan")}
    med = float(np.median(np.abs(d))); p99 = float(np.percentile(np.abs(d), 99))
    return {"ok": bool(med <= tol_kj), "n": int(d.size), "median_abs_kj": med, "p99_abs_kj": p99,
            "mean_kj": float(np.mean(d)), "tol_kj": float(tol_kj)}


# --- run reconstruction and frame alignment ---------------------------------------------------

import json as _json
import re as _re
from pathlib import Path as _Path
from types import SimpleNamespace as _NS

_SEG_RE = _re.compile(r"^replica_(\d+)(?:_resume_from_(\d+))?\.xtc$")


def find_run_root(prod_dir) -> _Path:
    """Nearest directory (prod dir or up to two parents) holding run_args.json and 01_solvated_start.pdb."""
    base = _Path(prod_dir)
    for d in (base, base.parent, base.parent.parent):
        if (d / "run_args.json").is_file() and (d / "01_solvated_start.pdb").is_file():
            return d
    raise FileNotFoundError(f"no run_args.json + 01_solvated_start.pdb at or above {base}")


def rebuild_physical_system(run_root, openmm, app, unit):
    """(System, topology, run_args, peptide_atoms) exactly as production's create_system builds it.

    The default box is the solvated topology's (01_solvated_start.pdb), which is the box production's
    ``_pin_pme_parameters`` uses for alpha and the PME grid.
    """
    from gareus.system_setup import make_forcefield_from_args, create_system
    from gareus.energy_decomposition import peptide_atom_groups_from_topology
    run_root = _Path(run_root)
    run_args = _json.loads((run_root / "run_args.json").read_text())
    ns = _NS(**run_args)
    pdb = app.PDBFile(str(run_root / "01_solvated_start.pdb"))
    ff = make_forcefield_from_args(app, ns)
    system = create_system(app, unit, ff, pdb.topology, ns, include_barostat=False)
    _groups, pep, _non = peptide_atom_groups_from_topology(pdb.topology, "all-peptide")
    return system, pdb.topology, run_args, sorted(int(i) for i in pep)


def trajectory_segments(epoch_dir) -> list[tuple[int, int, _Path]]:
    """(replica, resume_start, path) of every non-empty XTC under ``replica_trajectories``, oldest first."""
    tdir = _Path(epoch_dir) / "replica_trajectories"
    out = []
    for p in sorted(tdir.glob("replica_*.xtc")) if tdir.is_dir() else []:
        m = _SEG_RE.match(p.name)
        if m is None or p.stat().st_size == 0:
            continue
        out.append((int(m.group(1)), int(m.group(2) or 0), p))
    return sorted(out, key=lambda t: (t[0], t[1]))


def _traj_peptide_indices(traj_topology, full_topology, pep_full) -> list[int]:
    """Indices of the peptide atoms inside the trajectory's own atom order, checked by name."""
    full_atoms = list(full_topology.atoms())
    traj_atoms = list(traj_topology.atoms())
    if len(traj_atoms) == len(full_atoms):
        local = list(pep_full)
    else:
        if len(traj_atoms) < len(pep_full):
            raise ValueError(f"trajectory holds {len(traj_atoms)} atoms, fewer than the {len(pep_full)} peptide atoms")
        local = list(range(len(pep_full)))  # solute-only XTC: peptide atoms in ascending full-index order
    for i_loc, i_full in zip(local, pep_full):
        a, b = traj_atoms[i_loc], full_atoms[i_full]
        if (a.name, a.residue.name) != (b.name, b.residue.name):
            raise ValueError(f"trajectory atom {i_loc} ({a.residue.name}:{a.name}) != full atom {i_full} "
                             f"({b.residue.name}:{b.name}): trajectory atom order does not match the rebuilt system")
    return local


def evaluate_sample_frames(d, *, epoch_dirs=None, platform: str = "CPU", stride: int = 1, progress=None) -> dict:
    """Per-sample peptide-only energies and backbone torsions from saved frames.

    Frames are matched to samples by (epoch source, replica, step), with step = round(time / dt) from
    the XTC and dt from run_args; a later resume segment overrides an earlier one for the same step.
    ``stride`` keeps every stride-th trajectory interval (deterministic in step). Returns NaN-filled
    per-sample arrays plus diagnostics.
    """
    import mdtraj as md
    from gareus.imports import import_openmm
    openmm, app, unit = import_openmm()
    n = int(np.asarray(d.cv).size)
    meta = d.meta or {}
    _src = meta.get("_epoch_source")
    src = np.asarray(np.zeros(n, int) if _src is None else _src, dtype=np.int64)
    if src.size != n:
        raise ValueError("meta['_epoch_source'] is not aligned with the samples")
    if epoch_dirs is None:
        if meta.get("_epoch_source_run_dirs"):
            epoch_dirs = [_Path(p) for p in meta["_epoch_source_run_dirs"]]
        elif meta.get("_epoch_source") is not None:
            listed = [_Path(p) for p in meta.get("adaptive_epoch_run_dirs") or []]
            if not listed or int(src.max()) >= len(listed) or len(set(src.tolist())) != len(listed):
                raise ValueError("cannot map _epoch_source to phase directories: the loader recorded no "
                                 "_epoch_source_run_dirs and adaptive_epoch_run_dirs does not have one entry per source")
            epoch_dirs = listed
        else:
            epoch_dirs = [_Path(d.prod_dir)]
    if src.size and int(src.max()) >= len(epoch_dirs):
        raise ValueError(f"_epoch_source reaches {int(src.max())} but only {len(epoch_dirs)} phase directories are known")
    run_root = find_run_root(d.prod_dir)
    system, full_top, run_args, pep_full = rebuild_physical_system(run_root, openmm, app, unit)
    dt_ps = float(run_args["timestep_fs"]) / 1000.0
    interval = int(run_args.get("traj_interval") or 1)
    steps = np.asarray(d.step, dtype=np.int64); reps = np.asarray(d.replica, dtype=np.int64)
    lookup = {}
    order = np.lexsort((steps, reps, src))
    keys = np.stack([src[order], reps[order]], axis=1)
    cut = np.flatnonzero(np.any(np.diff(keys, axis=0) != 0, axis=1)) + 1
    for part in np.split(order, cut):
        if part.size:
            lookup[(int(src[part[0]]), int(reps[part[0]]))] = (steps[part], part)

    evaluator = PeptideOnlyEvaluator(system, pep_full, openmm, unit, platform=platform)
    quads_full, labels = backbone_torsion_quads(full_top, pep_full)
    pos_of = {a: k for k, a in enumerate(pep_full)}
    quads_pep = [tuple(pos_of[a] for a in q) for q in quads_full]
    e_nb = np.full(n, np.nan); e_bond = np.full(n, np.nan); e_dih = np.full(n, np.nan)
    theta = np.full((n, len(quads_pep)), np.nan)
    diag = {"run_root": str(run_root), "platform": platform, "stride": int(stride), "pme": list(evaluator.pme),
            "n_segments": 0, "n_frames_read": 0, "n_frames_matched": 0, "n_frames_unmatched": 0,
            "torsion_labels": labels, "timestep_ps": dt_ps, "traj_interval_steps": interval}
    for e_idx, edir in enumerate(epoch_dirs):
        segs = trajectory_segments(edir)
        if not segs:
            continue
        top_path = _Path(edir) / "solute_only.pdb"
        if not top_path.is_file():
            top_path = run_root / "01_solvated_start.pdb"
        mtop = md.load_topology(str(top_path))
        local = _traj_peptide_indices(mtop.to_openmm(), full_top, pep_full)
        for rep, _start, path in segs:
            entry = lookup.get((e_idx, rep))
            try:
                tr = md.load(str(path), top=mtop)
            except Exception as exc:  # a corrupted segment loses its frames, loudly
                diag.setdefault("unreadable_segments", []).append(f"{path}: {exc}")
                continue
            diag["n_segments"] += 1
            diag["n_frames_read"] += tr.n_frames
            if entry is None:
                diag["n_frames_unmatched"] += tr.n_frames
                continue
            s_steps, s_idx = entry
            f_steps = np.rint(np.asarray(tr.time, dtype=np.float64) / dt_ps).astype(np.int64)
            pos = np.clip(np.searchsorted(s_steps, f_steps), 0, len(s_steps) - 1)
            hit = s_steps[pos] == f_steps
            if stride > 1:
                hit &= (f_steps // interval) % int(stride) == 0
            diag["n_frames_unmatched"] += int(np.count_nonzero(s_steps[pos] != f_steps))
            xyz = tr.xyz[:, local, :].astype(np.float64)
            boxes = tr.unitcell_vectors
            fr = np.flatnonzero(hit)
            if fr.size:
                theta[s_idx[pos[fr]]] = dihedral_angles(xyz[fr], quads_pep)
            for f in fr:
                i = s_idx[pos[f]]
                box = None if boxes is None else np.asarray(boxes[f], dtype=np.float64)
                e_nb[i], e_bond[i], e_dih[i] = evaluator.evaluate(xyz[f], box)
            diag["n_frames_matched"] += int(fr.size)
            if progress is not None:
                progress(diag)
    diag["n_samples_with_frames"] = int(np.count_nonzero(np.isfinite(e_nb)))
    return {"e_nb": e_nb, "e_bond": e_bond, "e_dih": e_dih, "theta": theta, "diag": diag,
            "full_system_frames": False}
