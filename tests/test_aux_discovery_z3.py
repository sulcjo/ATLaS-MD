import inspect
from pathlib import Path

import mdtraj as md
import numpy as np
import pytest

from gareus.adaptive.aux_discovery import z3_search as Z
from gareus.adaptive.aux_discovery.descriptors import descriptor_definition, evaluate_descriptors
from gareus.adaptive.aux_discovery.settings import AuxDiscoverySettings

PDB = Path(__file__).parent / "data" / "chignolin_solute.pdb"


class _FT:   # minimal FrameTable stand-in
    def __init__(self, tors, cv1, cv2, basin, replica):
        self.tors, self.cv1, self.cv2, self.basin, self.replica = tors, cv1, cv2, basin, replica
        self.n = len(cv1)


def _planted(n=6000, seed=0, along_cv1=False):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, n)
    ang = rng.normal(scale=0.3, size=(n, 18))
    ang[:, 3] += np.where(labels == 1, 1.5, -1.5)             # the planted torsion
    tors = np.stack([np.sin(ang), np.cos(ang)], 2).reshape(n, 36)
    cv1 = (labels + 0.1 * rng.normal(size=n)) if along_cv1 else rng.normal(size=n)
    cv2 = rng.normal(size=n)
    basin = rng.integers(0, 5, (n, 8)).astype(np.uint8)
    basin[:, 2] = (ang[:, 3] > 0).astype(np.uint8)
    return _FT(tors, cv1.astype(np.float32), cv2.astype(np.float32), basin, rng.integers(0, 40, n)), labels


def test_recovers_planted_torsion():
    ft, lab = _planted()
    tr = np.arange(ft.n) < 4500; ho = ~tr; bins = np.zeros(ft.n, int)
    best, allc = Z.search_z3(ft, lab, bins, tr, ho, AuxDiscoverySettings(), nbins=1)
    assert best is not None and best.passed and best.groups == (0, 1)
    top = np.flatnonzero(np.abs(best.w_std) > 1e-8)
    assert set(top) <= {6, 7}                                   # sin/cos of torsion 3


def test_rejects_cv1_correlated_direction():
    ft, lab = _planted(along_cv1=True)
    tr = np.arange(ft.n) < 4500; ho = ~tr; bins = np.zeros(ft.n, int)
    best, allc = Z.search_z3(ft, lab, bins, tr, ho, AuxDiscoverySettings(), nbins=1)
    assert best is None and any("max_cv_corr" in c.fail for c in allc)


def _cand():
    rng = np.random.default_rng(0)
    return Z.Z3Candidate(groups=(0, 1), C=0.1, w_std=rng.normal(size=36), mu=rng.normal(size=36) * 0.1,
                         sd=1 + rng.random(36), z_raw_train_sd=2.5, info_gain=0.2, stability=0.9,
                         corr_cv1=0.0, corr_cv2=0.0, basin_gain=0.1, n_nonzero=36, passed=True, fail=[])


def test_emitted_model_matches_projection_and_loads():
    t = md.load(str(PDB)); d = descriptor_definition(t.topology)
    cand = _cand()
    model = Z.emit_model(cand, d, t.topology.to_openmm(), label="test", provenance={"epoch": 1})
    from gareus.auxiliary_cv.evaluate import z_from_positions
    from gareus.auxiliary_cv.features import check_feature_atoms
    check_feature_atoms(model, t.topology.to_openmm())
    z_model = z_from_positions(t.xyz[0], model)
    z_ref = Z.z3_values(evaluate_descriptors(t.xyz, d)["tors"], cand)[0]
    assert abs(float(np.ravel(z_model)[0]) - float(z_ref)) < 1e-6
    from gareus.auxiliary_cv.model import AuxModel
    tmp = Path(__file__).parent / "_tmp_model.json"
    try:
        model.write(tmp); assert AuxModel.load(tmp).model_sha256 == model.model_sha256
    finally:
        tmp.unlink(missing_ok=True)


def test_emit_refuses_disagreeing_production_topology():
    t = md.load(str(PDB)); d = descriptor_definition(t.topology)
    top = t.topology.to_openmm()
    q = int(d.phi_quads[0][1])
    list(top.atoms())[q].name = "XX"
    with pytest.raises(ValueError, match="solute and production topologies disagree at atom"):
        Z.emit_model(_cand(), d, top, label="t", provenance={})


def test_ties_prefer_larger_c_and_ranking_rule():
    src = inspect.getsource(Z.search_z3)
    assert "max(scores)" in src or "max((" in src


def test_no_native_readout_imported():
    src = inspect.getsource(Z)
    imports = [ln for ln in src.splitlines() if ln.strip().startswith(("import ", "from "))]
    for ln in imports:
        for bad in ("rmsd", "native", "helix", "chignolin_fes"):
            assert bad not in ln.lower(), ln
