"""Reference-free descriptors (c10 build_features.py): backbone torsions (IUPAC sign), heavy-atom
residue-pair soft-min contacts, backbone N-O H-bond switches, per-residue basin codes."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from gareus.adaptive.discovery_census import basin_codes

LAMBDA_NM = 0.02
R0_HC_NM = 0.45
R0_HB_NM = 0.35
MIN_HC_SEPARATION = 3
MIN_HB_SEPARATION = 2


@dataclass(frozen=True)
class DescriptorDefinition:
    phi_quads: np.ndarray
    psi_quads: np.ndarray
    phi_labels: Tuple[str, ...]
    psi_labels: Tuple[str, ...]
    hc_groups: Tuple[Tuple[np.ndarray, np.ndarray], ...]
    hc_labels: Tuple[str, ...]
    hb_pairs: np.ndarray
    hb_labels: Tuple[str, ...]
    core_phi_cols: np.ndarray
    core_psi_cols: np.ndarray
    atom_names: Tuple[str, ...]
    atom_residue_names: Tuple[str, ...]
    atom_residue_index: Tuple[int, ...]
    schema_sha256: str

    @property
    def tors_labels(self) -> Tuple[str, ...]:
        out = []
        for name in self.phi_labels + self.psi_labels:
            out += [f"tors_sin_{name}", f"tors_cos_{name}"]
        return tuple(out)


def _res_name(res) -> str:
    return f"{res.name}{res.resSeq}"


def descriptor_definition(topology) -> DescriptorDefinition:
    import mdtraj as md
    probe = md.Trajectory(np.zeros((1, topology.n_atoms, 3), dtype=np.float32), topology)
    phi_idx, _ = md.compute_phi(probe)
    psi_idx, _ = md.compute_psi(probe)
    atoms = list(topology.atoms)
    phi_labels = tuple(f"phi_{_res_name(atoms[q[2]].residue)}" for q in phi_idx)
    psi_labels = tuple(f"psi_{_res_name(atoms[q[1]].residue)}" for q in psi_idx)
    residues = [r for r in topology.residues if r.is_protein]
    heavy = {r.index: np.array([a.index for a in r.atoms if a.element is not None and a.element.symbol != "H"])
             for r in residues}
    hc_groups, hc_labels = [], []
    for i, ri in enumerate(residues):
        for rj in residues[i + MIN_HC_SEPARATION:]:
            hc_groups.append((heavy[ri.index], heavy[rj.index]))
            hc_labels.append(f"hc_{_res_name(ri)}_{_res_name(rj)}")
    donors = [(k, a.index) for k, r in enumerate(residues) if k >= 1 and r.name != "PRO"
              for a in r.atoms if a.name == "N"]
    acceptors = [(k, a.index) for k, r in enumerate(residues) for a in r.atoms if a.name in ("O", "OXT")]
    hb_pairs, hb_labels = [], []
    for kd, dn in donors:
        for ka, ac in acceptors:
            if abs(kd - ka) >= MIN_HB_SEPARATION:
                hb_pairs.append((dn, ac))
                hb_labels.append(f"hb_{_res_name(residues[kd])}N_{_res_name(residues[ka])}{atoms[ac].name}")
    # core residues = residues having both phi and psi
    phi_res = [atoms[q[2]].residue.index for q in phi_idx]
    psi_res = [atoms[q[1]].residue.index for q in psi_idx]
    core = [r for r in phi_res if r in psi_res]
    core_phi = np.array([phi_res.index(r) for r in core], dtype=np.int64)
    core_psi = np.array([psi_res.index(r) for r in core], dtype=np.int64)
    atom_names = tuple(a.name for a in atoms)
    atom_residue_names = tuple(a.residue.name for a in atoms)
    atom_residue_index = tuple(int(a.residue.index) for a in atoms)
    schema = {"phi": phi_labels, "psi": psi_labels, "hc": hc_labels, "hb": hb_labels,
              "lambda_nm": LAMBDA_NM, "r0_hc_nm": R0_HC_NM, "r0_hb_nm": R0_HB_NM,
              "phi_quads": phi_idx.tolist(), "psi_quads": psi_idx.tolist(),
              "atom_names": atom_names, "atom_residue_names": atom_residue_names,
              "atom_residue_index": atom_residue_index}
    sha = hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest()
    return DescriptorDefinition(np.asarray(phi_idx), np.asarray(psi_idx), phi_labels, psi_labels,
                                tuple(hc_groups), tuple(hc_labels), np.asarray(hb_pairs, dtype=np.int64).reshape(-1, 2),
                                tuple(hb_labels), core_phi, core_psi,
                                atom_names, atom_residue_names, atom_residue_index, sha)


def _dihedral(xyz, quads):
    b0 = xyz[:, quads[:, 0]] - xyz[:, quads[:, 1]]
    b1 = xyz[:, quads[:, 2]] - xyz[:, quads[:, 1]]
    b2 = xyz[:, quads[:, 3]] - xyz[:, quads[:, 2]]
    b1n = b1 / np.linalg.norm(b1, axis=-1, keepdims=True)
    v = b0 - (b0 * b1n).sum(-1, keepdims=True) * b1n
    w = b2 - (b2 * b1n).sum(-1, keepdims=True) * b1n
    x = (v * w).sum(-1)
    y = (np.cross(b1n, v) * w).sum(-1)
    return np.arctan2(y, x)


def evaluate_descriptors(xyz_nm: np.ndarray, d: DescriptorDefinition) -> Dict[str, np.ndarray]:
    xyz = np.asarray(xyz_nm, dtype=np.float64)
    theta = np.concatenate([_dihedral(xyz, d.phi_quads), _dihedral(xyz, d.psi_quads)], axis=1)
    tors = np.stack([np.sin(theta), np.cos(theta)], axis=2).reshape(theta.shape[0], -1)
    hc = np.empty((xyz.shape[0], len(d.hc_groups)))
    for c, (a, b) in enumerate(d.hc_groups):
        x = np.linalg.norm(xyz[:, a][:, :, None, :] - xyz[:, b][:, None, :, :], axis=-1).reshape(xyz.shape[0], -1)
        m = x.min(1)
        dist = m - LAMBDA_NM * np.log(np.exp(-(x - m[:, None]) / LAMBDA_NM).sum(1))
        hc[:, c] = 1.0 / (1.0 + (dist / R0_HC_NM) ** 6)
    r = np.linalg.norm(xyz[:, d.hb_pairs[:, 0]] - xyz[:, d.hb_pairs[:, 1]], axis=-1)
    hb = (1.0 / (1.0 + (r / R0_HB_NM) ** 6)).astype(np.float32)
    n_phi = len(d.phi_labels)
    deg = np.degrees(theta)
    basin = basin_codes(deg[:, d.core_phi_cols], deg[:, n_phi + d.core_psi_cols])
    return {"tors": tors, "tors_theta_iupac": theta, "hc": hc, "hb": hb, "basin": basin}
