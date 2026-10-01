"""Swarm side of the contact-map CV1 fit (spec ``2026-10-01-contact-map-cv1.md`` section 3).

Builds :class:`gareus.cv_selection.contact_map_cv1.CV1Data` from a swarm round and writes
``analysis/cv1_model.json`` (only when a mode is selected) and ``analysis/cv1_selection_report.json``.
Diagnostic for now: nothing downstream reads the model (the anchor kind and production mode are a
later step), so the CV1 design of the round is unchanged.

Row source:

* ``trace`` -- every ok member recorded ``contact_map_features.npy`` with one shared definition: the
  fit uses every selection row.
* ``pdb_frames`` -- otherwise (swarms that predate the recording, or a mixed round): the map is
  computed from each member's ``frames/frame_*.pdb`` (one per ``--swarm-seed-frame-interval-ps``)
  for the rows that have one, with a warning; lags are rounded to multiples of that stride.

C-alpha clusters for the breadth always come from the PDB frames (greedy leader, 2 A, the 30 most
populated + "other"); per-residue backbone basins from the torsion features (phi/psi paired by their
shared C-alpha).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from ..correctness._io import digest, file_digest
from ..cv_selection import contact_map_cv1 as CM1
from ..cv_selection.independence import frame_partition
from . import contact_map as CMAP

CA_CUTOFF_NM = 0.2
N_TOP_CA_CLUSTERS = 30
N_COARSE_CELLS = 8                       # = SelectionConfig.n_cells_coarse (the design measure)
SOURCE_TRACE = "trace"
SOURCE_PDB = "pdb_frames"


def read_pdb_atoms(path) -> Tuple[np.ndarray, List[str]]:
    """(positions in A (n, 3), atom names) of ATOM/HETATM records in file order."""
    xyz, names = [], []
    with open(path) as fh:
        for line in fh:
            if line.startswith(("ATOM", "HETATM")):
                xyz.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
                names.append(line[12:16].strip())
    return np.asarray(xyz, dtype=np.float64), names


def basin_labels(features, schema) -> np.ndarray:
    """(n, n_core) backbone basin index per residue with both phi and psi (paired by C-alpha)."""
    from ..adaptive.discovery_census import basin_codes
    X = np.asarray(features, dtype=np.float64)
    torsions: Dict[str, Dict[str, Any]] = {}
    for f in schema.features:
        t = torsions.setdefault(f.torsion_name, {"atoms": tuple(f.atom_indices)})
        t[f.trig] = int(f.index)
    phi_by_ca, psi_by_ca = {}, {}
    for name, t in torsions.items():
        angle = -np.degrees(np.arctan2(X[:, t["sin"]], X[:, t["cos"]]))      # "negated" convention
        if name.startswith("phi"):
            phi_by_ca[t["atoms"][2]] = angle                                 # C(i-1) N CA C
        elif name.startswith("psi"):
            psi_by_ca[t["atoms"][1]] = angle                                 # N CA C N(i+1)
    core = sorted(set(phi_by_ca) & set(psi_by_ca))
    if not core:
        raise ValueError("no residue has both phi and psi features")
    return np.column_stack([basin_codes(phi_by_ca[c], psi_by_ca[c]).astype(np.int64) for c in core])


def _top_labels(cluster_ids, n_top: int) -> np.ndarray:
    ids, counts = np.unique(cluster_ids, return_counts=True)
    top = ids[np.argsort(-counts, kind="stable")[:n_top]]
    out = np.full(np.asarray(cluster_ids).shape, len(top), dtype=np.int64)
    for k, u in enumerate(top):
        out[np.asarray(cluster_ids) == u] = k
    return out


def _row_lookup(dataset) -> Dict[Tuple[int, int], int]:
    return {(int(m), int(f)): i for i, (m, f) in enumerate(zip(dataset.member_ids, dataset.frame_index))}


def _pdb_rows(frame_candidates, ok_traces, rows_of) -> List[Tuple[int, str]]:
    """(selection row, pdb path) for every exported frame that is a selection row."""
    out = []
    for fr in frame_candidates:
        m = int(fr["member_id"])
        hits = np.flatnonzero(np.round(ok_traces[m]["frame"]).astype(int) == int(fr["frame"]))
        if hits.size and (m, int(hits[0])) in rows_of:
            out.append((rows_of[(m, int(hits[0]))], str(fr["pdb_path"])))
    return sorted(out)


def map_atoms(args) -> str:
    """The contact-map atom selection the CV1 is fitted on (``--swarm-cv1-contact-map-atoms``)."""
    return str(getattr(args, "swarm_cv1_contact_map_atoms", CMAP.ATOMS_CA) or CMAP.ATOMS_CA)


#: Rational-switch r0 (A) by atom selection: CA 8 A (2026-10-01 calibration, G0 CA info 0.283 nats),
#: heavy soft-min 4.5 A (the first calibration).
DEFAULT_R0_BY_ATOMS = {CMAP.ATOMS_CA: 8.0, CMAP.ATOMS_HEAVY: CM1.DEFAULT_SWITCH_R0_ANGSTROM}


def _native_maps(rd: Path, members, ok_traces, warnings: List[str], atoms: str = CMAP.ATOMS_CA) -> Optional[Tuple[Dict[int, np.ndarray], dict]]:
    """Every member's recorded map with one shared definition and one row per trace row, else None."""
    maps, definition = {}, None
    for m in members:
        got = CMAP.load_member_contact_map(rd / f"member_{int(m):04d}", atoms)
        if got is None:
            return None
        X, d = got
        n_trace = len(ok_traces[int(m)]["frame"])
        if X.shape[0] != n_trace:
            warnings.append(f"cv1 contact map: member {int(m)} recorded {X.shape[0]} map rows vs {n_trace} "
                            "trace rows; not aligned, using the PDB frames")
            return None
        if definition is not None and d["sha256"] != definition["sha256"]:
            return None
        maps[int(m)], definition = X, d
    return maps, definition


def _check_pdb_atoms(names: List[str], definition: Mapping[str, Any], topology) -> None:
    atoms = list(topology.atoms())
    used = sorted({a for p in definition["pairs"] for a in p["atoms_i"] + p["atoms_j"]})
    if used and used[-1] >= len(names):
        raise ValueError(f"PDB frames hold {len(names)} atoms; the contact map needs index {used[-1]}")
    bad = [a for a in used if names[a] != atoms[a].name]
    if bad:
        raise ValueError(f"PDB frame atom names differ from the swarm topology at indices {bad[:5]}")


def _baselines(dataset, ok_traces, rows) -> Dict[str, np.ndarray]:
    m, f = np.asarray(dataset.member_ids)[rows], np.asarray(dataset.frame_index)[rows]
    contacts = [ok_traces[int(a)].get("cv1_heavy_contacts", ok_traces[int(a)]["cv1"])[int(b)] for a, b in zip(m, f)]
    e2e = [10.0 * ok_traces[int(a)]["e2e_nm"][int(b)] for a, b in zip(m, f)]
    return {"contacts": np.asarray(contacts, dtype=np.float64), "e2e": np.asarray(e2e, dtype=np.float64)}


def _shape_columns(dataset, ok_traces) -> np.ndarray:
    m, f = np.asarray(dataset.member_ids), np.asarray(dataset.frame_index)
    return np.asarray([(ok_traces[int(a)]["rg_nm"][int(b)], ok_traces[int(a)]["e2e_nm"][int(b)])
                       for a, b in zip(m, f)], dtype=np.float64)


def _stride(member_ids, frame_index) -> int:
    diffs = [np.diff(np.sort(frame_index[member_ids == m])) for m in np.unique(member_ids)]
    diffs = np.concatenate([d for d in diffs if d.size]) if diffs else np.zeros(0)
    return int(np.median(diffs)) if diffs.size else 1


def assemble(dataset, ok_traces, frame_candidates, rd: Path, topology_pdb: Path, args,
             warnings: List[str]) -> Tuple[CM1.CV1Data, dict, dict]:
    """(data, contact-map definition, training record) for the selection rows of ``dataset``."""
    from openmm import app  # noqa: WPS433
    rows_of = _row_lookup(dataset)
    pdb = _pdb_rows(frame_candidates, ok_traces, rows_of)
    topology = app.PDBFile(str(topology_pdb)).topology
    atoms = map_atoms(args)
    native = _native_maps(Path(rd), np.unique(dataset.member_ids), ok_traces, warnings, atoms)
    if native is not None:
        maps, definition = native
        rows = np.arange(len(dataset.member_ids))
        D = np.vstack([maps[int(m)][int(f)] for m, f in zip(dataset.member_ids, dataset.frame_index)])
        source = SOURCE_TRACE
    else:
        definition = CMAP.contact_map_definition(
            topology, min_sequence_separation=int(getattr(args, "swarm_contact_map_min_separation", 3)),
            lambda_angstrom=float(getattr(args, "swarm_contact_map_lambda_a", 0.2)), atoms=atoms)
        rows = np.asarray([r for r, _ in pdb], dtype=np.int64)
        D = None
        source = SOURCE_PDB
        warnings.append(f"cv1 contact map: members did not all record {CMAP.file_names(atoms)[0]}; fitting on "
                        f"the {rows.size} rows with a PDB frame (contact_map_source: {SOURCE_PDB})")
    evaluator = CMAP.ContactMapEvaluator(definition)
    ca_rows, ca_xyz, pdb_maps = [], [], []
    for k, (r, path) in enumerate(pdb):
        xyz, names = read_pdb_atoms(path)
        if k == 0:
            _check_pdb_atoms(names, definition, topology)
        ca_rows.append(r)
        ca_xyz.append(xyz[[i for i, n in enumerate(names) if n == "CA"]] / 10.0)
        if D is None:
            pdb_maps.append(evaluator(xyz / 10.0))
    if D is None:
        D = np.vstack(pdb_maps) if pdb_maps else np.zeros((0, len(definition["pairs"])))
    return _finish(dataset, ok_traces, rows, D, definition, source, ca_rows, ca_xyz, topology_pdb)


def _finish(dataset, ok_traces, rows, D, definition, source, ca_rows, ca_xyz, topology_pdb):
    from ..adaptive.discovery_census import leader_cluster
    X = np.asarray(dataset.features, dtype=np.float64)
    # torsions + rg + e2e for every CV1 kind (a distance-anchored dataset drops e2e from its
    # shape features; the CV1 fit's design measure must not depend on which CV1 is configured)
    shape = _shape_columns(dataset, ok_traces)
    cells, _ = frame_partition(np.column_stack([X, shape]), n_cells=N_COARSE_CELLS)
    position = {int(r): k for k, r in enumerate(rows)}
    keep = [k for k, r in enumerate(ca_rows) if int(r) in position]
    ca_lab = None
    ca_sub = None
    if keep:
        _leaders, cl = leader_cluster(np.asarray([ca_xyz[k] for k in keep]), cutoff_nm=CA_CUTOFF_NM)
        ca_lab = _top_labels(np.asarray(cl), N_TOP_CA_CLUSTERS)
        ca_sub = np.asarray([position[int(ca_rows[k])] for k in keep], dtype=np.int64)
    members = np.asarray(dataset.member_ids)[rows]
    frames = np.asarray(dataset.frame_index)[rows]
    data = CM1.CV1Data(
        distances=np.asarray(D, dtype=np.float64), member_ids=members, frame_index=frames,
        groups=np.asarray(dataset.groups)[rows], cells=cells[rows],
        basin_labels=basin_labels(X, dataset.feature_schema)[rows],
        frame_dt_ps=float(dataset.frame_dt_ps),
        lag_stride=1 if source == SOURCE_TRACE else _stride(members, frames),
        ca_rows=ca_sub, ca_labels=ca_lab, baselines=_baselines(dataset, ok_traces, rows))
    training = {"contact_map_source": source, "n_rows": int(len(rows)),
                "rows_sha256": digest(np.ascontiguousarray(D, dtype=np.float64).tobytes()),
                "topology_sha256": file_digest(topology_pdb), "design_measure": f"balanced_frame_cells_{N_COARSE_CELLS}"}
    return data, definition, training


def fit_config(args) -> CM1.CV1FitConfig:
    return CM1.CV1FitConfig(
        switch_r0_angstrom=float(getattr(args, "swarm_cv1_contact_map_r0_a", None) or DEFAULT_R0_BY_ATOMS[map_atoms(args)]),
        tica_lag_ps=float(getattr(args, "cv_selection_tica_lag_ps", 50.0)),
        slowness_lag_ps=float(getattr(args, "cv_selection_slowness_lag_ps", 200.0)),
        min_slowness_rho=float(getattr(args, "cv_selection_min_slowness", 0.72)),
        half_split_min_corr=float(getattr(args, "cv_selection_half_split_min_corr", 0.8)),
        min_windows=int(getattr(args, "cv_selection_min_windows_cv1", 4)),
        overlap_sigma=float(getattr(args, "swarm_overlap_sigma", 1.5)),
        temperature_k=float(args.temperature_k),
        min_breadth_nats=float(getattr(args, "cv_selection_min_gain_nats", 0.02)),
        breadth_resamples=max(1, int(getattr(args, "cv_selection_gain_resamples", 8))),
        breadth_tie_sd=float(getattr(args, "cv_selection_breadth_tie_sd", 1.0)),
        # as for CV2: no resampling or pick "slowest" = every passing mode is in the tie-set
        use_breadth_tie=(int(getattr(args, "cv_selection_gain_resamples", 8)) > 0
                         and str(getattr(args, "cv_selection_pick", "breadth-tie-slowest")
                                 or "breadth-tie-slowest") == "breadth-tie-slowest"))


def clear_artifacts(an: Path) -> None:
    """Remove an earlier fit's model and report (a fallback or failure must never leave them)."""
    for name in (CM1.MODEL_NAME, CM1.REPORT_NAME):
        (Path(an) / name).unlink(missing_ok=True)


def run_fit(an: Path, dataset, ok_traces, frame_candidates, rd: Path, topology_pdb: Path, args,
            warnings: List[str]) -> Dict[str, Any]:
    """Fit, write the report (and the model when a mode is selected); return a summary.

    Earlier artifacts are removed first, so a fallback or failure never leaves an old model or report."""
    model_path, report_path = Path(an) / CM1.MODEL_NAME, Path(an) / CM1.REPORT_NAME
    clear_artifacts(an)
    from ..io import write_json
    data, definition, training = assemble(dataset, ok_traces, frame_candidates, rd, topology_pdb, args, warnings)
    model, report = CM1.select_contact_map_cv1(data, fit_config(args), definition=definition, training=training)
    if model is not None:
        CM1.write_model(model_path, model)
    write_json(report_path, report)
    summary = {k: report.get(k) for k in ("status", "selected", "selection_reason", "model_sha256",
                                          "breadth_tie_set", "tica_lag_ps", "n_rows")}
    summary["contact_map_source"] = training["contact_map_source"]
    return summary
