# gareus/adaptive/aux_discovery/z3_search.py
"""Pairwise L1 torsion discriminants between co-occurring discovery groups (c10
torsion_only_candidate.py + prereg_v2 candidate gates), and emission as atlas-aux-cv-model-v1."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression

from .partitions import info_gain
from .settings import AuxDiscoverySettings


@dataclass
class Z3Candidate:
    groups: Tuple[int, int]
    C: float
    w_std: np.ndarray
    mu: np.ndarray
    sd: np.ndarray
    z_raw_train_sd: float
    info_gain: float
    stability: float
    corr_cv1: float
    corr_cv2: float
    basin_gain: float
    n_nonzero: int
    passed: bool
    fail: List[str] = field(default_factory=list)

    def summary(self) -> dict:
        return {"groups": list(self.groups), "C": self.C, "info_gain": round(self.info_gain, 4),
                "stability": round(self.stability, 4), "corr_cv1": round(self.corr_cv1, 4),
                "corr_cv2": round(self.corr_cv2, 4), "basin_gain": round(self.basin_gain, 4),
                "n_nonzero": self.n_nonzero, "passed": self.passed, "fail": list(self.fail)}


def z3_values(tors: np.ndarray, cand: Z3Candidate) -> np.ndarray:
    return (((np.asarray(tors, float) - cand.mu) / cand.sd) @ cand.w_std) / cand.z_raw_train_sd


def _l1(Xs, y, C):
    return LogisticRegression(penalty="l1", solver="liblinear", C=float(C), max_iter=3000).fit(Xs, y)


def _bern_ll(m, X, y):
    p = np.clip(m.predict_proba(X)[:, 1], 1e-9, 1 - 1e-9)
    return float(np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _cooccurring_pairs(labels, bins, holdout, s: AuxDiscoverySettings):
    from .partitions import co_occurrence
    K = int(labels.max()) + 1
    pairs = set()
    for rec in co_occurrence(labels, bins, holdout, K, s):
        g = rec["groups"]
        pairs |= {(min(a, b), max(a, b)) for i, a in enumerate(g) for b in g[i + 1:]}
    return sorted(pairs)


def search_z3(ft, labels, bins, train, holdout, s: AuxDiscoverySettings, *, nbins: Optional[int] = None):
    nbins = int(nbins or s.s_bins ** 2)
    X = np.asarray(ft.tors, float)
    mu, sd = X[train].mean(0), X[train].std(0)
    sd = np.where(sd > 0, sd, 1.0)
    Xs = (X - mu) / sd
    even = (np.asarray(ft.replica) % 2) == 0
    K = int(labels.max()) + 1
    out: List[Z3Candidate] = []
    pairs = _cooccurring_pairs(labels, bins, holdout, s) if nbins > 1 else \
        [(a, b) for a in range(K) for b in range(a + 1, K)]
    for g1, g2 in pairs:
        m = train & np.isin(labels, (g1, g2)); y = (labels == g2).astype(int)
        if len(np.unique(y[m & even])) < 2 or len(np.unique(y[m & ~even])) < 2:
            continue
        scores = [(_bern_ll(_l1(Xs[m & even], y[m & even], C), Xs[m & ~even], y[m & ~even]), C) for C in s.l1_c_grid]
        _, C = max(scores)
        w = _l1(Xs[m], y[m], C).coef_[0]
        we = _l1(Xs[m & even], y[m & even], C).coef_[0]; wo = _l1(Xs[m & ~even], y[m & ~even], C).coef_[0]
        z_raw = Xs @ w
        if not np.any(w) or z_raw[train].std() == 0:
            continue
        stab = abs(float(np.corrcoef(Xs[holdout] @ we, Xs[holdout] @ wo)[0, 1]))
        c1 = float(np.corrcoef(z_raw[holdout], ft.cv1[holdout])[0, 1])
        c2 = float(np.corrcoef(z_raw[holdout], ft.cv2[holdout])[0, 1])
        ig = info_gain(z_raw, labels, bins, train, holdout, K, nbins)
        bg = float(sum(info_gain(z_raw, ft.basin[:, r].astype(int), bins, train, holdout, 5, nbins)
                       for r in range(ft.basin.shape[1])))
        fail = []
        if ig < s.info_gain_min: fail.append("info_gain")
        if not np.isfinite(stab) or stab < s.stability_min: fail.append("stability")
        if abs(c1) > s.max_cv_corr or abs(c2) > s.max_cv_corr: fail.append("max_cv_corr")
        if bg < s.basin_gain_min: fail.append("basin_gain")
        out.append(Z3Candidate((g1, g2), float(C), w, mu, sd, float(z_raw[train].std()), ig, stab, c1, c2, bg,
                               int((np.abs(w) > 1e-8).sum()), not fail, fail))
    passing = [c for c in out if c.passed]
    best = max(passing, key=lambda c: (c.info_gain, -c.n_nonzero)) if passing else None
    return best, out


def emit_model(cand: Z3Candidate, definition, full_topology, *, label: str, provenance: dict):
    """atlas-aux-cv-model-v1, "direct" convention: the descriptors' theta equals OpenMM CustomTorsionForce theta (verified); atom indices are
    those of ``full_topology`` (the campaign's 01_solvated_start.pdb), checked against the solute definition."""
    from gareus.auxiliary_cv.model import AuxModel, AUX_MODEL_SCHEMA
    from gareus.auxiliary_cv.runtime import canonical_topology_sha256
    from gareus.cv_selection.contracts import FEATURE_SCHEMA_VERSION
    coef = cand.w_std / cand.sd
    offset = float(-(cand.w_std * cand.mu / cand.sd).sum())
    quads = list(definition.phi_quads) + list(definition.psi_quads)
    names = list(definition.phi_labels) + list(definition.psi_labels)
    atoms = list(full_topology.atoms())
    for q in quads:
        for a in q:
            a = int(a)
            if (a >= len(atoms) or atoms[a].name != definition.atom_names[a]
                    or atoms[a].residue.name != definition.atom_residue_names[a]):
                raise ValueError(f"solute and production topologies disagree at atom {a}")
    feats = []
    for t, (q, name) in enumerate(zip(quads, names)):
        kind, res = name.split("_", 1)
        res_index = atoms[int(q[2 if kind == "phi" else 1])].residue.index
        for trig in ("sin", "cos"):
            feats.append({"index": len(feats), "name": f"{kind}-{res}-{trig}", "torsion_name": f"{kind}-{res}",
                          "residue_index": int(res_index), "atom_indices": [int(a) for a in q], "trig": trig,
                          "dihedral_sign_convention": "direct"})
    def _payload(top_sha):
        return {"schema": AUX_MODEL_SCHEMA,
                "feature_schema": {"schema": FEATURE_SCHEMA_VERSION, "topology_sha256": top_sha, "features": feats},
                "coefficients": [float(c) for c in coef], "offset": offset, "scale": float(cand.z_raw_train_sd),
                "periodic_imaging": "none", "units": "dimensionless", "label": label, "provenance": dict(provenance)}
    draft = AuxModel.from_mapping(_payload("0" * 64))
    return AuxModel.from_mapping(_payload(canonical_topology_sha256(full_topology, draft)))