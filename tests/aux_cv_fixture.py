"""Shared builders for the auxiliary-CV tests: model payloads over real backbone torsions."""
from __future__ import annotations

import numpy as np

from gareus.auxiliary_cv.model import AUX_MODEL_SCHEMA
from gareus.cv_selection.contracts import FEATURE_SCHEMA_VERSION


def feature_rows(quads, conventions=None, blocks=None):
    """sin then cos per torsion, in the swarm's canonical order (gareus/swarm/analyze.py:416-419).

    ``blocks`` gives each torsion's "phi"/"psi" (from backbone_torsion_quads labels) so
    topology checks see true names; without it the names alternate and only synthetic
    (topology-free) tests may use the payload.
    """
    rows = []
    for k, quad in enumerate(quads):
        conv = "negated" if conventions is None else conventions[k]
        block = blocks[k] if blocks is not None else ("phi" if k % 2 == 0 else "psi")
        for trig in ("sin", "cos"):
            rows.append({"index": len(rows), "name": f"{block}-{k}-{trig}", "torsion_name": f"{block}-{k}",
                         "residue_index": int(k), "atom_indices": [int(x) for x in quad],
                         "trig": trig, "dihedral_sign_convention": conv})
    return rows


def model_payload(quads, coefficients, offset=0.0, conventions=None, topology_sha256="0" * 64,
                  label="test", provenance=None, blocks=None, scale=1.0, periodic_imaging="none"):
    return {
        "schema": AUX_MODEL_SCHEMA,
        "feature_schema": {"schema": FEATURE_SCHEMA_VERSION, "topology_sha256": topology_sha256,
                           "features": feature_rows(quads, conventions, blocks)},
        "coefficients": [float(c) for c in coefficients],
        "offset": float(offset),
        "scale": float(scale),
        "periodic_imaging": periodic_imaging,
        "units": "dimensionless",
        "label": label,
        "provenance": provenance if provenance is not None else {"source": "manual"},
    }


def dipeptide():
    """GA dipeptide (solvated, minimised) from pep_gamd_fixture, with its backbone torsions."""
    from pep_gamd_fixture import solvated_dipeptide
    from gareus.imports import import_openmm
    from gareus.mbar_analysis.thermo_frames import backbone_torsion_quads

    _openmm, _app, unit = import_openmm()
    fx = solvated_dipeptide()
    quads, labels = backbone_torsion_quads(fx["topology"], fx["peptide"])
    pos = np.asarray(fx["positions"].value_in_unit(unit.nanometer), dtype=np.float64)
    return {"topology": fx["topology"], "positions_nm": pos, "quads": quads, "labels": labels}


def blocks_of(d):
    return [lab.split("-")[0] for lab in d["labels"]]


def perturbed_geometries(d, n=4, sigma_nm=0.02, seed=0):
    """The minimised geometry plus n copies whose torsion atoms are randomly displaced."""
    rng = np.random.default_rng(seed)
    atoms = sorted({a for q in d["quads"] for a in q})
    out = [d["positions_nm"].copy()]
    for _ in range(n):
        x = d["positions_nm"].copy()
        x[atoms] += rng.normal(scale=sigma_nm, size=(len(atoms), 3))
        out.append(x)
    return out


def four_atoms_at(rotation_rad):
    """4 atoms (nm) whose dihedral is controlled by rotating atom 3 about the 1-2 axis.

    rotation_rad = pi puts the torsion exactly at the +-pi branch cut of OpenMM's theta.
    """
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([0.0, 0.0, 0.15])
    p0 = p1 + np.array([0.15, 0.0, -0.05])
    p3 = p2 + np.array([0.15 * np.cos(rotation_rad), 0.15 * np.sin(rotation_rad), 0.05])
    return np.array([p0, p1, p2, p3])
