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
    corr_cv1: Optional[float]
    corr_cv2: Optional[float]
    basin_gain: float
    n_nonzero: int
    passed: bool
    fail: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def summary(self) -> dict:
        return {"groups": list(self.groups), "C": self.C, "info_gain": round(self.info_gain, 4),
                "stability": round(self.stability, 4), "corr_cv1": None if self.corr_cv1 is None else round(self.corr_cv1, 4),
                "corr_cv2": None if self.corr_cv2 is None else round(self.corr_cv2, 4), "basin_gain": round(self.basin_gain, 4),
                "n_nonzero": self.n_nonzero, "passed": self.passed, "fail": list(self.fail)}


def z3_values(tors: np.ndarray, cand: Z3Candidate) -> np.ndarray:
    return (((np.asarray(tors, float) - cand.mu) / cand.sd) @ cand.w_std) / cand.z_raw_train_sd


def _sklearn_has_l1_ratio_api() -> bool:
    import sklearn
    major, minor = (int(v) for v in sklearn.__version__.split(".")[:2])
    return (major, minor) >= (1, 8)


def _l1(Xs, y, C):
    if _sklearn_has_l1_ratio_api():
        m = LogisticRegression(l1_ratio=1.0, solver="liblinear", C=float(C), max_iter=3000)
    else:
        m = LogisticRegression(penalty="l1", solver="liblinear", C=float(C), max_iter=3000)
    return m.fit(Xs, y)


def _cv_corr(z, cv, holdout):
    """|corr| of z with cv on holdout rows where cv is finite; None when cv has < 2 distinct finite values."""
    cv = np.asarray(cv, float)
    ok = holdout & np.isfinite(cv) & np.isfinite(z)
    if ok.sum() < 3 or np.unique(cv[ok]).size < 2:
        return None
    return float(np.corrcoef(z[ok], cv[ok])[0, 1])


def pick_best(candidates):
    passing = [c for c in candidates if c.passed]
    return max(passing, key=lambda c: (c.info_gain, -c.n_nonzero)) if passing else None


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


def search_z3(ft, labels, bins, train, holdout, s: AuxDiscoverySettings, *, nbins: Optional[int] = None,
              all_pairs: bool = False):
    nbins = int(nbins or s.s_bins ** 2)
    X = np.asarray(ft.tors, float)
    mu, sd = X[train].mean(0), X[train].std(0)
    sd = np.where(sd > 0, sd, 1.0)
    Xs = (X - mu) / sd
    even = (np.asarray(ft.replica) % 2) == 0
    K = int(labels.max()) + 1
    out: List[Z3Candidate] = []
    pairs = _cooccurring_pairs(labels, bins, holdout, s) if not all_pairs else \
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
        c1 = _cv_corr(z_raw, ft.cv1, holdout)
        c2 = _cv_corr(z_raw, ft.cv2, holdout)
        ig = info_gain(z_raw, labels, bins, train, holdout, K, nbins)
        bg = float(sum(info_gain(z_raw, ft.basin[:, r].astype(int), bins, train, holdout, 5, nbins)
                       for r in range(ft.basin.shape[1])))
        fail, notes = [], []
        for name, v in (("info_gain", ig), ("stability", stab), ("basin_gain", bg)):
            if not np.isfinite(v):
                fail.append(f"nonfinite_{name}")
        if c1 is None:
            fail.append("nonfinite_corr_cv1")
        if c2 is None:
            notes.append("cv2_not_applicable")
        if np.isfinite(ig) and ig < s.info_gain_min: fail.append("info_gain")
        if np.isfinite(stab) and stab < s.stability_min: fail.append("stability")
        for name, c in (("cv1", c1), ("cv2", c2)):
            if c is not None and not np.isfinite(c):
                fail.append(f"nonfinite_corr_{name}")
            elif c is not None and abs(c) > s.max_cv_corr and "max_cv_corr" not in fail:
                fail.append("max_cv_corr")
        if np.isfinite(bg) and bg < s.basin_gain_min: fail.append("basin_gain")
        out.append(Z3Candidate((g1, g2), float(C), w, mu, sd, float(z_raw[train].std()), ig, stab, c1, c2, bg,
                               int((np.abs(w) > 1e-8).sum()), not fail, fail, notes))
    return pick_best(out), out


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
    for q, name in zip(quads, names):
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
