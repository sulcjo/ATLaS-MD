from __future__ import annotations

from types import SimpleNamespace

import mdtraj as md
import numpy as np
import pytest

from aux_discovery_fixture import PDB, make_phase
from gareus.adaptive.aux_backfill import _final_pdb_z, _read_xtc, check_solute_indices, frame_z
from gareus.auxiliary_cv.atom_mapping import topology_metadata
from gareus.auxiliary_cv.sidechain_core import Primitive, Projection


def _permuted_topology(source, order, *, omit=()):
    """Build fresh MDTraj topology in requested atom order, preserving stable atom identities."""
    src = source.topology
    selected = [int(i) for i in order if int(i) not in set(omit)]
    top = md.Topology()
    chain_map = {}
    residue_map = {}
    atom_map = {}
    for old in selected:
        atom = src.atom(old)
        chain = atom.residue.chain
        cid = str(getattr(chain, "chain_id", "") or "")
        if cid not in chain_map:
            chain_map[cid] = top.add_chain(cid)
        residue_key = (cid, atom.residue.index)
        if residue_key not in residue_map:
            residue_map[residue_key] = top.add_residue(atom.residue.name, chain_map[cid],
                                                       int(atom.residue.resSeq))
        atom_map[old] = top.add_atom(atom.name, atom.element, residue_map[residue_key])
    for a, b in src.bonds:
        if a.index in atom_map and b.index in atom_map:
            top.add_bond(atom_map[a.index], atom_map[b.index])
    xyz = source.xyz[:, selected, :]
    return md.Trajectory(xyz, top), selected


def _model_and_quad():
    source = md.load(str(PDB))
    keys, _bonds = topology_metadata(source.topology)
    atoms_by_residue = {}
    for atom in source.topology.atoms:
        atoms_by_residue.setdefault(atom.residue.index, {})[atom.name] = atom.index
    quad = next((atoms_by_residue[i - 1]["C"], atoms_by_residue[i]["N"],
                 atoms_by_residue[i]["CA"], atoms_by_residue[i]["C"])
                for i in sorted(atoms_by_residue) if i > 0 and "C" in atoms_by_residue[i - 1]
                and all(n in atoms_by_residue[i] for n in ("N", "CA", "C")))
    projection = Projection((Primitive((quad,), "sin"),), (0.6,), offset=0.1, scale=1.2)
    model = SimpleNamespace(schema_version=2, atom_keys=keys, projection=projection)
    return source, model


def test_v2_backfill_maps_reordered_xtc_topology_and_final_pdb(tmp_path):
    source, model = _model_and_quad()
    order = np.arange(source.n_atoms - 1, -1, -1)
    permuted, _selected = _permuted_topology(source, order)
    xtc_top = tmp_path / "reordered-solute.pdb"
    permuted.save_pdb(str(xtc_top))
    mapped = check_solute_indices(xtc_top, model)
    expected = model.projection.values(source.xyz[0])[0]
    assert mapped.values(permuted.xyz[0])[0] == pytest.approx(expected, abs=1e-12)

    phase = tmp_path / "phase"
    (phase / "final_pdbs").mkdir(parents=True)
    final_path = phase / "final_pdbs" / "replica_000_window_000.pdb"
    permuted.save_pdb(str(final_path))
    got = _final_pdb_z(phase, 0, 0, model, PDB)
    assert got == pytest.approx(expected, abs=1e-12)


def test_v2_frame_z_maps_reordered_xtc_atoms(tmp_path):
    source, model = _model_and_quad()
    order = np.arange(source.n_atoms - 1, -1, -1)
    permuted, _selected = _permuted_topology(source, order)
    phase = make_phase(tmp_path, "epoch_000", {"replica_0.xtc": (0, 0, [300])}, {0: 0.0}, lambda _s: 0.0)
    permuted.save_pdb(str(phase / "solute_only.pdb"))
    with md.formats.XTCTrajectoryFile(str(phase / "replica_trajectories" / "replica_0.xtc"), "w") as fh:
        fh.write(permuted.xyz.astype(np.float32), time=np.array([1.05]), step=np.array([300], dtype=np.int32))
    got = frame_z(phase, model)
    xyz, _steps = _read_xtc(phase / "replica_trajectories" / "replica_0.xtc")
    expected = model.projection.values(xyz[0, np.argsort(order)])[0]
    assert got.loc[0, "aux_z"] == pytest.approx(expected, abs=1e-12)


def test_v2_remap_refuses_missing_orbit_atom_without_fallback(tmp_path):
    source, model = _model_and_quad()
    missing_index = model.projection.primitives[0].orbit[0][2]
    permuted, _selected = _permuted_topology(source, np.arange(source.n_atoms), omit=(missing_index,))
    path = tmp_path / "missing-orbit-atom.pdb"
    permuted.save_pdb(str(path))
    with pytest.raises(ValueError, match="missing model atom"):
        check_solute_indices(path, model)


def test_v2_final_pdb_rejects_missing_orbit_atom(tmp_path):
    source, model = _model_and_quad()
    missing_index = model.projection.primitives[0].orbit[0][2]
    permuted, _selected = _permuted_topology(source, np.arange(source.n_atoms), omit=(missing_index,))
    phase = tmp_path / "phase"
    (phase / "final_pdbs").mkdir(parents=True)
    permuted.save_pdb(str(phase / "final_pdbs" / "replica_000_window_000.pdb"))
    with pytest.raises(ValueError, match="missing model atom"):
        _final_pdb_z(phase, 0, 0, model, PDB)
