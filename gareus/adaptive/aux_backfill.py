"""z of the frozen aux model for every sample of a pre-admission phase, from its trajectory frames.
Strict: one frame per sample or the phase is refused; z is never filled with 0 and no row is dropped."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from gareus.adaptive import discovery_census as census
from gareus.adaptive.aux_discovery.frames import _read_xtc, load_phase_samples, phase_epoch, xtc_step_offset

BACKFILL_FILENAME = "aux_z_backfill.parquet"
_SHA_KEY = b"aux_model_sha256"


CHECKPOINT_MANIFEST = Path("checkpoints") / "production_checkpoint_manifest.json"


class BackfillIncomplete(RuntimeError):
    def __init__(self, n_missing: int, examples: list, reasons: Optional[list] = None):
        why = f"; {'; '.join(reasons[:5])}" if reasons else ""
        super().__init__(f"{n_missing} samples have no trajectory frame (e.g. {examples[:5]}){why}")
        self.n_missing = int(n_missing)
        self.examples = examples
        self.reasons = list(reasons or [])


def phase_label(phase_dir, adaptive_dir) -> Optional[str]:
    """Phase label relative to the campaign (``epoch_001``, ``epoch_001/baseline``, ``final``); None outside it.
    The key of a backfill entry: a copied or moved campaign keeps its labels, not its absolute paths."""
    try:
        return Path(phase_dir).resolve().relative_to(Path(adaptive_dir).resolve()).as_posix()
    except ValueError:
        return None


def final_production_step(phase_dir: Path) -> Optional[int]:
    """The phase's last production step on the samples' clock (``absolute_step`` of its checkpoint manifest,
    which production writes at the end of the loop, together with final_pdbs/); None when not recorded."""
    import json
    path = Path(phase_dir) / CHECKPOINT_MANIFEST
    try:
        value = json.loads(path.read_text()).get("absolute_step")
    except (OSError, ValueError, AttributeError):
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return int(value)


def check_solute_indices(solute_pdb: Path, model) -> None:
    """Solute-only XTC atom i must be production atom i for every feature atom: the solute PDB's atom
    names/residues at those indices must be the phi/psi the model's features claim."""
    import mdtraj as md
    from gareus.auxiliary_cv.features import check_feature_atoms
    top = md.load_topology(str(solute_pdb))
    atoms = list(top.atoms)
    for f in model.feature_schema.features:
        quad = [int(a) for a in f.atom_indices]
        if max(quad) >= len(atoms):
            raise ValueError(f"{solute_pdb}: feature {f.name} atom {max(quad)} beyond solute ({len(atoms)} atoms)")
        kind = f.torsion_name.split("-")[0]
        res = atoms[quad[2 if kind == "phi" else 1]].residue
        want = f"{kind}-{res.name}{res.resSeq}"
        if f.torsion_name != want:
            raise ValueError(f"{solute_pdb}: feature {f.name} expects {f.torsion_name} but atom {quad[2 if kind == 'phi' else 1]} "
                             f"({atoms[quad[2 if kind == 'phi' else 1]].name}) is in residue {res.name}{res.resSeq}")
    try:
        check_feature_atoms(model, top.to_openmm())
    except Exception as exc:
        raise ValueError(f"{solute_pdb}: solute atom indices do not match the model's feature atoms: {exc}") from exc


def frame_z(phase_dir: Path, model, adaptive_dir: Optional[Path] = None) -> pd.DataFrame:
    """z for every XTC frame of the phase, (replica, step, aux_z) with step on the samples' clock
    (``xtc_step_offset``); a later resume file wins."""
    from gareus.auxiliary_cv.evaluate import z_from_positions
    phase_dir = Path(phase_dir)
    top_path = census._find_topology(phase_dir, Path(adaptive_dir) if adaptive_dir else phase_dir.parent)
    if top_path is None:
        raise FileNotFoundError(f"{phase_dir}: no solute_only.pdb")
    check_solute_indices(top_path, model)
    import mdtraj as md
    n_solute = md.load_topology(str(top_path)).n_atoms
    files = sorted(census._trajectory_files(phase_dir), key=lambda t: (t[0], t[1] or 0, str(t[2])))
    offset = xtc_step_offset(phase_dir)          # samples are keyed by calib_steps + prod_done
    blocks = []
    for i, (replica, _start, path) in enumerate(files):
        if Path(path).stat().st_size == 0:
            continue
        nxt = next((s for r, s, _ in files[i + 1:] if r == replica), None)
        xyz, steps = _read_xtc(path)
        if steps.size == 0:
            continue
        # A resume file named ``_resume_from_R`` restarts at the checkpoint step R but its reporter writes the
        # first frame at R + interval (npt_driver.register_reporter), so the earlier file's frame AT R is the
        # only one there: keep it (<=). Should a resume file also re-emit R, drop_duplicates keeps the later one.
        keep = np.ones(steps.size, bool) if nxt is None else steps <= int(nxt)
        if not keep.any():
            continue
        if xyz.shape[1] != n_solute:
            raise ValueError(f"{path}: {xyz.shape[1]} atoms per frame, solute PDB {top_path} has {n_solute}")
        z = np.ravel(z_from_positions(xyz[keep].astype(np.float64), model))
        blocks.append(pd.DataFrame({"replica": int(replica), "step": steps[keep] + offset, "aux_z": z}))
    if not blocks:
        return pd.DataFrame({"replica": np.zeros(0, np.int64), "step": np.zeros(0, np.int64), "aux_z": np.zeros(0)})
    return pd.concat(blocks).drop_duplicates(["replica", "step"], keep="last").reset_index(drop=True)


def _final_pdb_z(phase_dir: Path, replica: int, window: int, model, top_path: Path) -> Optional[float]:
    """z of ``final_pdbs/replica_RRR_window_WWW.pdb`` (production writes it at the phase's last step from the
    replica's live Context; a replica's earlier stop at the same window is overwritten by it). None when the
    file does not exist; raises when its leading atoms are not the solute's."""
    import mdtraj as md
    from gareus.auxiliary_cv.evaluate import z_from_positions
    path = Path(phase_dir) / "final_pdbs" / f"replica_{int(replica):03d}_window_{int(window):03d}.pdb"
    if not path.exists():
        return None
    solute = md.load_topology(str(top_path))
    t = md.load(str(path))
    n = solute.n_atoms
    if t.n_atoms < n or any((a.name, a.residue.name, a.residue.resSeq) != (b.name, b.residue.name, b.residue.resSeq)
                            for a, b in zip(list(t.topology.atoms)[:n], solute.atoms)):
        raise ValueError(f"{path}: final_pdbs file's first {n} atoms are not the solute of {top_path}")
    return float(np.ravel(z_from_positions(t.xyz[0, :n].astype(np.float64), model))[0])


def write_phase_backfill(phase_dir: Path, model, *, adaptive_dir: Optional[Path] = None) -> dict:
    """z for every sample of the phase from its XTC frames. The one exception is each replica's LAST sample when
    it has no frame: the phase's step total (from the MD pool) is generally off the frame grid, so production
    logs a final sample at that step with no XTC frame; its configuration is ``final_pdbs/replica_RRR_window_WWW
    .pdb`` (W = the sample's own window), used only when the sample's step is the phase's final production step
    (checkpoint manifest ``absolute_step``) and after the replica's last XTC frame. Any other missing frame, a
    missing final PDB, or a final PDB without that step evidence refuses the phase (``BackfillIncomplete.reasons``).
    With ``adaptive_dir`` the entry carries the campaign-relative ``label`` the admission record is matched on."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    phase_dir = Path(phase_dir)
    loaded = load_phase_samples(phase_dir)
    samples = loaded[["replica", "step"]]
    zdf = frame_z(phase_dir, model, adaptive_dir)
    merged = samples.merge(zdf, on=["replica", "step"], how="left")
    missing = merged["aux_z"].isna().to_numpy()
    n_pdb = 0
    reasons: List[str] = []
    if missing.any():
        last = (loaded["step"].to_numpy() == loaded.groupby("replica")["step"].transform("max").to_numpy())
        top_path = census._find_topology(phase_dir, Path(adaptive_dir) if adaptive_dir else phase_dir.parent)
        z = merged["aux_z"].to_numpy(dtype=np.float64).copy()
        end_step = final_production_step(phase_dir)
        xtc_max = zdf.groupby("replica")["step"].max().to_dict() if len(zdf) else {}
        for i in np.flatnonzero(missing & last):
            replica, step = int(loaded["replica"].iat[i]), int(loaded["step"].iat[i])
            why = _final_pdb_evidence(replica, step, end_step, xtc_max.get(replica))
            if why is not None:
                reasons.append(why)
                continue
            zi = _final_pdb_z(phase_dir, replica, int(loaded["window_id"].iat[i]), model, top_path)
            if zi is not None:
                z[i] = zi
                n_pdb += 1
        merged["aux_z"] = z
        missing = merged["aux_z"].isna().to_numpy()
    if missing.any():
        raise BackfillIncomplete(int(missing.sum()), merged.loc[missing, ["replica", "step"]].head(10).values.tolist(),
                                 reasons)
    out = phase_dir / BACKFILL_FILENAME
    table = pa.Table.from_pandas(merged, preserve_index=False).replace_schema_metadata(
        {_SHA_KEY: model.model_sha256.encode()})
    tmp = out.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp)
    tmp.replace(out)
    label = phase_label(phase_dir, adaptive_dir) if adaptive_dir is not None else None
    return {"phase": str(phase_dir), **({"label": label} if label is not None else {}),
            "n_samples": int(len(samples)), "n_z": int(merged["aux_z"].notna().sum()),
            "n_from_xtc": int(len(samples)) - n_pdb, "n_from_final_pdb": int(n_pdb),
            "model_sha256": model.model_sha256, "sha256": hashlib.sha256(out.read_bytes()).hexdigest()}


def _final_pdb_evidence(replica: int, step: int, end_step: Optional[int], xtc_max: Optional[int]) -> Optional[str]:
    """None when final_pdbs/ provably holds this sample's configuration, else why not: the sample must be the
    phase's final production step (checkpoint manifest) and lie after every XTC frame of its replica. A replica
    with no XTC frame at all (``xtc_max`` None) is allowed on the manifest step alone: there is no frame to order
    against, and the manifest step is the step final_pdbs/ was written at."""
    if end_step is None:
        return (f"replica {replica} step {step}: no checkpoint manifest ({CHECKPOINT_MANIFEST}) records the "
                "phase's final production step, so final_pdbs/ cannot be tied to the sample")
    if step != end_step:
        return (f"replica {replica} step {step} is not the phase's final production step {end_step} "
                "(checkpoint manifest): final_pdbs/ does not hold it")
    if xtc_max is not None and step <= int(xtc_max):
        return f"replica {replica} step {step} is not after its last XTC frame (step {int(xtc_max)})"
    return None


def read_phase_backfill(phase_dir: Path, model_sha256: str, *, expected_sha256: str) -> pd.DataFrame:
    """The phase's backfill, after checking the sha256 of the bytes parsed against ``expected_sha256`` (the
    admission record's entry for the phase) and the model the file was computed with."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    path = Path(phase_dir) / BACKFILL_FILENAME
    data = path.read_bytes()
    got = hashlib.sha256(data).hexdigest()
    if got != str(expected_sha256):
        raise ValueError(f"{path}: backfill sha256 {got[:12]} != admission record {str(expected_sha256)[:12]} "
                         "(the file changed after the admission recorded it)")
    t = pq.read_table(pa.BufferReader(data))
    sha = (t.schema.metadata or {}).get(_SHA_KEY, b"").decode()
    if sha != model_sha256:
        raise ValueError(f"{phase_dir}: backfill model {sha[:12]} != campaign model {model_sha256[:12]}")
    return t.to_pandas()


def remove_backfills(adaptive_dir: Path) -> None:
    """Delete every aux_z_backfill.parquet under the campaign (a failed freeze must leave none behind)."""
    for p in Path(adaptive_dir).rglob(BACKFILL_FILENAME):
        try:
            p.unlink()
        except OSError as exc:
            print(f"WARNING: could not remove {p}: {exc}")


def backfill_all(adaptive_dir: Path, model, *, up_to_epoch: int) -> List[dict]:
    out = []
    phases, _ = census.ordered_phases(Path(adaptive_dir))
    for label, phase_dir in phases:
        ep = phase_epoch(label)
        if ep is not None and ep <= int(up_to_epoch):
            out.append(write_phase_backfill(Path(phase_dir), model, adaptive_dir=Path(adaptive_dir)))
    return out


def merge_backfill_entries(old: list, new: list) -> list:
    """The admission record's backfill list after ``new`` entries rewrote their phases' files: every phase
    matched by label or path, the new entry wins, campaign order of ``new`` kept."""
    labels = {e.get("label") for e in new if e.get("label") is not None}
    phases = {str(e.get("phase")) for e in new}
    kept = [e for e in (old or []) if isinstance(e, dict)
            and e.get("label") not in labels and str(e.get("phase")) not in phases]
    return kept + list(new)


R_KCAL_MOL_K = 0.0019872041


def worker_energy_error_kt(z_recorded, z_frame, workers, *, temperature_k: float = 300.0) -> List[dict]:
    """Per worker (c3, k3 kcal/mol per z^2): the largest error the frame-derived z would make in the worker's
    restraint energy, 0.5 k3 |(z1 - c3)^2 - (z2 - c3)^2| / RT, over the samples within 2 sigma_w of c3
    (sigma_w = sqrt(RT / k3); either z). The pooled bias only matters where the worker samples, so that is
    where the tolerance is judged (final fix wave I3: z units carry no physical scale)."""
    RT = R_KCAL_MOL_K * float(temperature_k)
    z1 = np.asarray(z_recorded, dtype=np.float64)
    z2 = np.asarray(z_frame, dtype=np.float64)
    out = []
    for c3, k3 in workers:
        c3, k3 = float(c3), float(k3)
        sw = float(np.sqrt(RT / k3)) if k3 > 0 else float("inf")
        near = (np.abs(z1 - c3) <= 2.0 * sw) | (np.abs(z2 - c3) <= 2.0 * sw)
        de = 0.5 * k3 * np.abs((z1 - c3) ** 2 - (z2 - c3) ** 2) / RT
        out.append({"aux_center": c3, "aux_k_kcal_mol": k3, "sigma_w": sw, "n_within_2sigma": int(near.sum()),
                    "max_abs_dz_within": float(np.abs(z1 - z2)[near].max()) if near.any() else None,
                    "max_energy_err_kt": float(de[near].max()) if near.any() else 0.0})
    return out


def check_backfill_against_recorded(phase_dir: Path, model, *, tol: float = 0.05,
                                    adaptive_dir: Optional[Path] = None, raise_on_fail: bool = False,
                                    workers=None, temperature_k: float = 300.0, tol_kt: float = 1.0) -> dict:
    """Post-admission phase: recorded ``aux_z_00`` vs z_from_positions on its XTC frames (the backfill's path).

    Always returns max_abs_dev (z units), n_compared, tol, ok. With ``workers`` [(c3, k3), ...] the verdict is
    physical: ``per_worker`` energy errors (``worker_energy_error_kt``), ``max_energy_err_kt`` and ok iff that
    is <= ``tol_kt`` (default 1 kT); without workers ok iff max_abs_dev <= ``tol`` (z units; XTC 0.001 nm
    precision gives ~1e-2). Raises on a failed verdict only with ``raise_on_fail``; raises when no sample could
    be compared."""
    import duckdb
    phase_dir = Path(phase_dir)
    rec = duckdb.connect().execute(
        f"select replica, step, aux_z_00, filename from read_parquet('{phase_dir}/samples/*/*.parquet', "
        f"filename=true, union_by_name=true)").df()
    rec["seg"] = [int(m.group(1)) if (m := re.search(r"seg_(\d+)", f)) else 0 for f in rec["filename"]]
    rec = rec.sort_values(["replica", "step", "seg"], kind="stable").drop_duplicates(["replica", "step"], keep="last")
    joined = rec.merge(frame_z(phase_dir, model, adaptive_dir), on=["replica", "step"], how="inner")
    joined = joined[joined["aux_z_00"].notna()]
    if joined.empty:
        raise ValueError(f"{phase_dir}: no sample with a recorded aux_z_00 and a frame to compare")
    dev = float((joined["aux_z_00"] - joined["aux_z"]).abs().max())
    out = {"phase": str(phase_dir), "n_compared": int(len(joined)), "max_abs_dev": dev, "tol": float(tol)}
    if workers:
        per = worker_energy_error_kt(joined["aux_z_00"].to_numpy(np.float64), joined["aux_z"].to_numpy(np.float64),
                                     workers, temperature_k=temperature_k)
        err = max(p["max_energy_err_kt"] for p in per)
        ok = bool(np.isfinite(err) and err <= float(tol_kt))
        out.update(per_worker=per, max_energy_err_kt=float(err), tol_kt=float(tol_kt), temperature_k=float(temperature_k))
        what = f"implies a worker energy error of {err:.3g} kT (> {tol_kt} kT)"
    else:
        ok = dev <= tol
        what = f"differs from z_from_positions by {dev:.3g} (> {tol})"
    out["ok"] = bool(ok)
    if not ok and raise_on_fail:
        raise ValueError(f"{phase_dir}: recorded aux_z_00 {what}")
    return out
