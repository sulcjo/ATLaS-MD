"""OpenMM force and position evaluator for the contact-map CV1 (spec 2026-10-01-contact-map-cv1.md
sections 4-5; production definition: the C-alpha map, user decision 2026-10-01).

The frozen model (``gareus.cv_selection.contact_map_cv1``, schema ``cv1_contact_map_v1``) defines

    c = sum_p w_p / (1 + (d_p / r0)^6) - offset

over residue pairs p. Production runs the C-alpha map, d_p = r(CA_i, CA_j): c is a weighted sum of
one switch per atom pair, so the CV is ONE CustomBondForce (per-bond weight w_p) inside the umbrella
CustomCVForce, exactly the shape of the contact CV1, with an O(1) value and a bounded gradient.

Why not the heavy-atom soft-min the map was first calibrated with (d_p = -lam ln sum exp(-r/lam)):
measured 2026-10-01 on OpenCL (RTX 3060 Ti), a CustomCVForce accumulates each sub-CV's force in
64-bit fixed point (saturating near 2.1e9 kJ/mol/nm, resolution 2^-32). Per-pair sums
exp(-(r - REF)/lam) span ~1e-40..1e27, so near contact the CV1 force was ~0 on GPU while its energy
was right. Exact alternatives cost too much: staggered-reference windows need 112 sub-CVs (OpenMM
allows 32 per CustomCVForce; nested, 9x slower MD), a CustomCompoundBondForce per pair needs ~1 min
of Lepton differentiation per Context. The CA map calibrated as well or better on chignolin (G0:
CA-cluster information 0.283 vs 0.261 nats, basins 0.41 vs 0.31, rho 0.963 vs 0.975), so multi-atom
maps are refused here.

The umbrella is 0.5 k (c - r0)^2 with the historical global parameter names ``k`` / ``r0``
(``set_window`` and exchange code unchanged).
"""
from __future__ import annotations

from typing import Any, List, Mapping, Optional

import numpy as np

NM_PER_ANGSTROM = 0.1
SUBCV_PREFIX = "cms"
CV_NAME = "cmap_cv1"
#: Global parameter holding the model's offset, so c = sub-CV 0 - cmap_offset can be read from the
#: force's cached collective variables (``production.observe_fast_path``) without positions.
OFFSET_PARAM = "cmap_offset"
_CHECKED: set = set()


def _model_constants(model: Mapping[str, Any]) -> tuple:
    from .cv_selection.contact_map_cv1 import check_model
    key = (id(model), str(model.get("sha256")))
    if key not in _CHECKED:                 # re-hashing the model on every observation is wasted work
        check_model(model)
        _CHECKED.add(key)
    lam_nm = float(model["contact_map"]["lambda_angstrom"]) * NM_PER_ANGSTROM
    r0_nm = float(model["switch"]["r0_angstrom"]) * NM_PER_ANGSTROM
    return lam_nm, r0_nm, [float(w) for w in model["weights"]], float(model["offset"])


def require_single_atom_pairs(model: Mapping[str, Any]) -> None:
    """Production force exists only for one-atom-per-residue maps (the CA map)."""
    bad = [k for k, p in enumerate(model["contact_map"]["pairs"])
           if len(p["atoms_i"]) != 1 or len(p["atoms_j"]) != 1]
    if bad:
        raise ValueError("this cv1 model is a multi-atom (soft-min) contact map: it has no GPU-exact force "
                         "of acceptable cost in plain OpenMM; fit the CA map "
                         "(--swarm-cv1-contact-map-atoms ca, the default) for production")


def subcv_force(openmm, model: Mapping[str, Any]):
    """sum_p w_p / (1 + (r_p / r0)^6) over the residue pairs' C-alpha pairs (= c + offset), non-periodic."""
    _lam, r0_nm, weights, _off = _model_constants(model)
    require_single_atom_pairs(model)
    f = openmm.CustomBondForce(f"cmw/(1+(r/{r0_nm:.17g})^6)")
    f.addPerBondParameter("cmw")
    for p, w in zip(model["contact_map"]["pairs"], weights):
        f.addBond(int(p["atoms_i"][0]), int(p["atoms_j"][0]), [float(w)])
    return f


def add_subcvs(openmm, cv_force, model: Mapping[str, Any], prefix: str = SUBCV_PREFIX) -> str:
    """Add the CV1 sub-CV to ``cv_force``; return the Lepton definition of ``CV_NAME``."""
    _lam, _r0, _w, offset = _model_constants(model)
    cv_force.addCollectiveVariable(f"{prefix}0", subcv_force(openmm, model))
    cv_force.addGlobalParameter(OFFSET_PARAM, float(offset))
    return f"{CV_NAME} = {prefix}0-{OFFSET_PARAM}"


def add_contact_map_umbrella_force(openmm, system, model: Mapping[str, Any], force_group: int = 31):
    """The CV1 umbrella 0.5 k (c - r0)^2 (k in kJ/mol per unit^2, r0 in model units)."""
    force = openmm.CustomCVForce("0")
    defs = add_subcvs(openmm, force, model)
    force.setEnergyFunction(f"0.5*k*({CV_NAME}-r0)^2; " + defs)
    force.addGlobalParameter("k", 0.0)
    force.addGlobalParameter("r0", 0.0)
    force.setForceGroup(int(force_group))
    system.addForce(force)
    return force


def cv_from_positions_nm(model: Mapping[str, Any], positions_nm, atom_map: Optional[Mapping[int, int]] = None) -> float:
    """c at positions (nm) in double precision: the model's own soft-min definition (= the swarm/fit
    evaluator; for the CA map d_p is the CA-CA distance).

    ``atom_map`` (model atom index -> row of ``positions_nm``) reads a structure with another atom
    numbering, e.g. a peptide-only seed conformer."""
    lam_nm, r0_nm, weights, offset = _model_constants(model)
    x = np.asarray(positions_nm, dtype=np.float64)

    def rows(atoms):
        idx = [int(a) for a in atoms] if atom_map is None else [int(atom_map[int(a)]) for a in atoms]
        return x[np.asarray(idx, dtype=np.int64)]

    total = 0.0
    for p, w in zip(model["contact_map"]["pairs"], weights):
        r = np.linalg.norm(rows(p["atoms_j"])[None, :, :] - rows(p["atoms_i"])[:, None, :], axis=-1).ravel()
        m = float(r.min())
        d = m - lam_nm * float(np.log(np.sum(np.exp(-(r - m) / lam_nm))))
        total += w / (1.0 + (d / r0_nm) ** 6)
    return float(total - offset)


__all__ = ["CV_NAME", "OFFSET_PARAM", "SUBCV_PREFIX", "add_contact_map_umbrella_force", "add_subcvs", "cv_from_positions_nm",
           "require_single_atom_pairs", "subcv_force"]
