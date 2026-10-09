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
from gareus.adaptive.aux_discovery.frames import _read_xtc, load_phase_samples, phase_epoch

BACKFILL_FILENAME = "aux_z_backfill.parquet"
_SHA_KEY = b"aux_model_sha256"


class BackfillIncomplete(RuntimeError):
    def __init__(self, n_missing: int, examples: list):
        super().__init__(f"{n_missing} samples have no trajectory frame (e.g. {examples[:5]})")
        self.n_missing = int(n_missing)
        self.examples = examples


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
    """z for every XTC frame of the phase, (replica, step, aux_z); a later resume file wins."""
    from gareus.auxiliary_cv.evaluate import z_from_positions
    phase_dir = Path(phase_dir)
    top_path = census._find_topology(phase_dir, Path(adaptive_dir) if adaptive_dir else phase_dir.parent)
    if top_path is None:
        raise FileNotFoundError(f"{phase_dir}: no solute_only.pdb")
    check_solute_indices(top_path, model)
    import mdtraj as md
    n_solute = md.load_topology(str(top_path)).n_atoms
    files = sorted(census._trajectory_files(phase_dir), key=lambda t: (t[0], t[1] or 0, str(t[2])))
    blocks = []
    for i, (replica, _start, path) in enumerate(files):
        if Path(path).stat().st_size == 0:
            continue
        nxt = next((s for r, s, _ in files[i + 1:] if r == replica), None)
        xyz, steps = _read_xtc(path)
        if steps.size == 0:
            continue
        keep = np.ones(steps.size, bool) if nxt is None else steps < int(nxt)
        if not keep.any():
            continue
        if xyz.shape[1] != n_solute:
            raise ValueError(f"{path}: {xyz.shape[1]} atoms per frame, solute PDB {top_path} has {n_solute}")
        z = np.ravel(z_from_positions(xyz[keep].astype(np.float64), model))
        blocks.append(pd.DataFrame({"replica": int(replica), "step": steps[keep], "aux_z": z}))
    if not blocks:
        return pd.DataFrame({"replica": np.zeros(0, np.int64), "step": np.zeros(0, np.int64), "aux_z": np.zeros(0)})
    return pd.concat(blocks).drop_duplicates(["replica", "step"], keep="last").reset_index(drop=True)


def write_phase_backfill(phase_dir: Path, model, *, adaptive_dir: Optional[Path] = None) -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq
    phase_dir = Path(phase_dir)
    samples = load_phase_samples(phase_dir)[["replica", "step"]]
    zdf = frame_z(phase_dir, model, adaptive_dir)
    merged = samples.merge(zdf, on=["replica", "step"], how="left")
    missing = merged["aux_z"].isna()
    if missing.any():
        raise BackfillIncomplete(int(missing.sum()), merged.loc[missing, ["replica", "step"]].head(10).values.tolist())
    out = phase_dir / BACKFILL_FILENAME
    table = pa.Table.from_pandas(merged, preserve_index=False).replace_schema_metadata(
        {_SHA_KEY: model.model_sha256.encode()})
    tmp = out.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp)
    tmp.replace(out)
    return {"phase": str(phase_dir), "n_samples": int(len(samples)), "n_z": int(merged["aux_z"].notna().sum()),
            "model_sha256": model.model_sha256, "sha256": hashlib.sha256(out.read_bytes()).hexdigest()}


def read_phase_backfill(phase_dir: Path, model_sha256: str) -> pd.DataFrame:
    import pyarrow.parquet as pq
    t = pq.read_table(Path(phase_dir) / BACKFILL_FILENAME)
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


def check_backfill_against_recorded(phase_dir: Path, model, *, tol: float = 0.05,
                                    adaptive_dir: Optional[Path] = None, raise_on_fail: bool = False) -> dict:
    """Post-admission phase: recorded ``aux_z_00`` vs z_from_positions on its frames. Raises on a
    deviation above ``tol`` only with ``raise_on_fail``; always returns
    max_abs_dev, n_compared, tol, ok. (XTC 0.001 nm precision gives ~1e-2 z differences, hence tol 0.05.)
    Raises when no sample could be compared."""
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
    ok = dev <= tol
    if not ok and raise_on_fail:
        raise ValueError(f"{phase_dir}: recorded aux_z_00 differs from z_from_positions by {dev:.3g} (> {tol})")
    return {"phase": str(phase_dir), "n_compared": int(len(joined)), "max_abs_dev": dev, "tol": float(tol), "ok": bool(ok)}
