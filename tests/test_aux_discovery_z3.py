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
    best, allc = Z.search_z3(ft, lab, bins, tr, ho, AuxDiscoverySettings(), nbins=1, all_pairs=True)
    assert best is not None and best.passed and best.groups == (0, 1)
    top = np.flatnonzero(np.abs(best.w_std) > 1e-8)
    assert set(top) <= {6, 7}                                   # sin/cos of torsion 3


def test_rejects_cv1_correlated_direction():
    ft, lab = _planted(along_cv1=True)
    tr = np.arange(ft.n) < 4500; ho = ~tr; bins = np.zeros(ft.n, int)
    best, allc = Z.search_z3(ft, lab, bins, tr, ho, AuxDiscoverySettings(), nbins=1, all_pairs=True)
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
    rng = np.random.default_rng(1)
    xyz = np.repeat(t.xyz, 5, axis=0) + rng.normal(scale=0.05, size=(5, t.xyz.shape[1], 3))
    # trig features are smooth through +-pi, so large perturbations (which cross it for some torsions) must agree
    zs = np.array([float(np.ravel(z_from_positions(x, model))[0]) for x in xyz])
    zr = Z.z3_values(evaluate_descriptors(xyz, d)["tors"], cand)
    assert np.abs(zs - zr).max() < 1e-6
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


def test_exact_c_ties_go_to_larger_c(monkeypatch):
    ft, lab = _planted()
    tr = np.arange(ft.n) < 4500; ho = ~tr; bins = np.zeros(ft.n, int)
    monkeypatch.setattr(Z, "_bern_ll", lambda m, X, y: 0.0)
    s = AuxDiscoverySettings()
    best, allc = Z.search_z3(ft, lab, bins, tr, ho, s, nbins=1, all_pairs=True)
    assert allc and all(c.C == max(s.l1_c_grid) for c in allc)


def _c(ig, nz, passed=True):
    return Z.Z3Candidate((0, 1), 0.1, np.ones(36), np.zeros(36), np.ones(36), 1.0, ig, 0.9, 0.0, 0.0, 0.1,
                         nz, passed, [])


def test_pick_best_ranking():
    assert Z.pick_best([_c(0.3, 5), _c(0.3, 2), _c(0.2, 1)]).n_nonzero == 2
    assert Z.pick_best([_c(0.2, 1), _c(0.3, 9)]).n_nonzero == 9
    assert Z.pick_best([_c(0.9, 1, passed=False)]) is None


def test_l1_is_warning_free_and_sparse():
    import warnings
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 10)); y = (X[:, 3] > 0).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        m = Z._l1(X, y, 0.1)
    assert 0 < (np.abs(m.coef_[0]) > 1e-8).sum() < 10


def test_nan_cv2_is_not_applicable_and_can_pass():
    ft, lab = _planted()
    ft.cv2 = np.full(ft.n, np.nan, np.float32)
    tr = np.arange(ft.n) < 4500; ho = ~tr; bins = np.zeros(ft.n, int)
    best, allc = Z.search_z3(ft, lab, bins, tr, ho, AuxDiscoverySettings(), nbins=1, all_pairs=True)
    assert best is not None and best.corr_cv2 is None and "cv2_not_applicable" in best.notes


def test_nonfinite_info_gain_fails(monkeypatch):
    ft, lab = _planted()
    tr = np.arange(ft.n) < 4500; ho = ~tr; bins = np.zeros(ft.n, int)
    monkeypatch.setattr(Z, "info_gain", lambda *a, **k: float("nan"))
    best, allc = Z.search_z3(ft, lab, bins, tr, ho, AuxDiscoverySettings(), nbins=1, all_pairs=True)
    assert best is None and allc and "nonfinite_info_gain" in allc[0].fail


def test_no_native_readout_imported():
    import importlib, pkgutil, sys
    import gareus.adaptive.aux_discovery as pkg
    for mi in pkgutil.iter_modules(pkg.__path__):
        importlib.import_module(f"{pkg.__name__}.{mi.name}")
    bad = [m for m in sys.modules if m.startswith("gareus") and
           any(k in m for k in ("native", "chignolin_fes", "readout"))]
    assert not bad, bad
    src = inspect.getsource(Z)
    imports = [ln for ln in src.splitlines() if ln.strip().startswith(("import ", "from "))]
    for ln in imports:
        for bad in ("rmsd", "native", "helix", "chignolin_fes"):
            assert bad not in ln.lower(), ln


# ---- U12: null-calibrated gate, multi-k sources --------------------------------------------------------------
def _with_lineage(ft, n_lin=20):
    ft.lineage = np.array([f"p:{r % n_lin}" for r in range(ft.n)])
    ft.step = np.arange(ft.n) // n_lin
    return ft


def _shift_within_lineage(arr, lineage, step, rng):
    out = np.array(arr, copy=True)
    for lin in np.unique(lineage):
        idx = np.nonzero(lineage == lin)[0]
        idx = idx[np.argsort(step[idx], kind="stable")]
        out[idx] = np.roll(arr[idx], int(rng.integers(1, idx.size)), axis=0)
    return out


def test_null_gate_passes_planted_signal():
    ft, lab = _planted(n=3000); _with_lineage(ft)
    bins = np.zeros(ft.n, int); tr = np.arange(ft.n) < 2200; ho = ~tr
    s = AuxDiscoverySettings(n_null_z3=4)
    sources = [(2, lab, [(0, 1)])]
    best, allc = Z.search_z3_sources(ft, sources, bins, tr, ho, s, nbins=1)
    gate = Z.z3_null_gate(ft, sources, best, bins, tr, ho, s, nbins=1)
    assert best is not None and gate["passed"] and gate["null_margin"] > 0
    assert gate["n_null"] == 4 and len(gate["null_best"]) == 4
    assert gate["real_best"] == pytest.approx(best.info_gain)
    assert {"min", "median", "max"} <= set(gate["null_summary"])


def test_null_gate_fails_noise_control():
    ft, lab = _planted(n=3000); _with_lineage(ft)
    rng = np.random.default_rng(7)   # a fixed draw; under no link the gate's false-pass rate is 1/(n_null+1)
    ft.tors = _shift_within_lineage(ft.tors, ft.lineage, ft.step, rng)   # torsions no longer linked to labels
    bins = np.zeros(ft.n, int); tr = np.arange(ft.n) < 2200; ho = ~tr
    s = AuxDiscoverySettings(n_null_z3=8)
    sources = [(2, lab, [(0, 1)])]
    best, _ = Z.search_z3_sources(ft, sources, bins, tr, ho, s, nbins=1)
    gate = Z.z3_null_gate(ft, sources, best, bins, tr, ho, s, nbins=1)
    assert not gate["passed"] and gate["null_margin"] <= 0


def test_removed_thresholds_do_not_gate_candidates():
    ft, lab = _planted(); _with_lineage(ft)
    tr = np.arange(ft.n) < 4500; ho = ~tr; bins = np.zeros(ft.n, int)
    _, allc = Z.search_z3(ft, lab, bins, tr, ho, AuxDiscoverySettings(), nbins=1, all_pairs=True)
    assert allc and all(not ({"info_gain", "stability", "basin_gain"} & set(c.fail)) for c in allc)
    assert all(np.isfinite(c.info_gain) and np.isfinite(c.stability) and np.isfinite(c.basin_gain) for c in allc)


def test_sources_collect_candidates_from_every_k():
    ft, lab = _planted(n=3000); _with_lineage(ft)
    bins = np.zeros(ft.n, int); tr = np.arange(ft.n) < 2200; ho = ~tr
    lab3 = lab.copy(); lab3[::7] = 2
    best, allc = Z.search_z3_sources(ft, [(2, lab, [(0, 1)]), (3, lab3, [(0, 1), (0, 2)])], bins, tr, ho,
                                     AuxDiscoverySettings(), nbins=1)
    assert {c.k for c in allc} == {2, 3}
    assert {(c.k, c.groups) for c in allc} >= {(2, (0, 1)), (3, (0, 1)), (3, (0, 2))}


def test_scramble_preserves_counts_and_is_seeded():
    lin = np.repeat(["a", "b"], 50); step = np.tile(np.arange(50), 2)
    lab = np.arange(100) % 3
    a = Z.scramble_labels(lab, lin, step, np.random.default_rng(1))
    b = Z.scramble_labels(lab, lin, step, np.random.default_rng(1))
    assert (a == b).all() and (a != lab).any()
    for l in ("a", "b"):
        assert np.bincount(a[lin == l], minlength=3).tolist() == np.bincount(lab[lin == l], minlength=3).tolist()


def test_choose_partition_returns_every_passing_k():
    from gareus.adaptive.aux_discovery.partitions import choose_partition
    rng = np.random.default_rng(0)
    n = 1200
    Pm = np.vstack([rng.normal(c, 0.4, (n // 3, 2)) for c in ((0, 0), (6, 0), (0, 6))])
    lineage = np.array([f"p:{i % 12}" for i in range(n)])
    tr = rng.random(n) < 0.6; ho = ~tr
    ch = choose_partition(Pm, tr, ho, lineage, AuxDiscoverySettings(k_max=4, n_boot_partition=3), 0)
    ks = [c.k for c in ch.choices]
    assert 3 in ks and ks == sorted(ks) and ch.k == ks[-1]
    assert all(c.labels is not None and c.gmm is not None for c in ch.choices)


def test_removed_settings_rejected():
    for key in ("info_gain_min", "stability_min", "basin_gain_min"):
        with pytest.raises(ValueError, match="unknown aux settings key"):
            AuxDiscoverySettings.from_mapping({**AuxDiscoverySettings().to_mapping(), key: 0.1})


def test_null_gate_exits_early_on_failure():
    ft, lab = _planted(n=3000); _with_lineage(ft)
    ft.tors = _shift_within_lineage(ft.tors, ft.lineage, ft.step, np.random.default_rng(7))
    bins = np.zeros(ft.n, int); tr = np.arange(ft.n) < 2200; ho = ~tr
    s = AuxDiscoverySettings(n_null_z3=20); src = [(2, lab, [(0, 1)])]
    best, _ = Z.search_z3_sources(ft, src, bins, tr, ho, s, nbins=1)
    gate = Z.z3_null_gate(ft, src, best, bins, tr, ho, s, nbins=1)
    assert not gate["passed"] and gate["n_null_run"] < 20 and len(gate["null_best"]) == gate["n_null_run"]


def test_failing_null_gate_gives_broaden(monkeypatch):
    from types import SimpleNamespace
    from gareus.adaptive.aux_discovery import pipeline as P
    ft, lab = _planted(n=600); _with_lineage(ft)
    ft.hc = ft.hb = np.zeros((ft.n, 1)); ft.definition = SimpleNamespace(hc_labels=["a"], hb_labels=["b"]); ft.state_id = np.zeros(ft.n, int)
    part = SimpleNamespace(status="ok", choice=SimpleNamespace(k=2, table=[]), hidden_fraction=1.0, co_occurrence=[],
                           lineage_info=1.0, bins=np.zeros(ft.n, int), frozen=None,
                           per_k=[{"k": 2, "labels": lab, "hidden_fraction": 1.0, "co_occurrence": [],
                                   "lineage_info": 1.0, "triggered": True}])
    monkeypatch.setattr(P, "fit_partition", lambda *a, **k: part)
    monkeypatch.setattr(P, "_cooccurring_pairs", lambda *a, **k: [(0, 1)])
    monkeypatch.setattr(P, "z3_null_gate", lambda *a, **k: {"passed": False, "null_margin": -1.0})
    tr = np.arange(ft.n) < 400
    res = P.run_discovery(ft, train=tr, holdout=~tr, settings=AuxDiscoverySettings(), full_topology=None,
                          k3_max=3, epoch=1)
    assert res.status == "broaden"
    assert "passing_k" in res.report["partition"] and "null_gate" in res.report["z3_search"]


def test_correlation_guard_uses_only_the_selected_conditioning_dimensions():
    ft, lab = _planted()
    ft.cv1 = np.full(ft.n, 3.0, np.float32)                      # constant: a dropped coordinate
    tr = np.arange(ft.n) < 4500; ho = ~tr; bins = np.zeros(ft.n, int)
    _, dflt = Z.search_z3(ft, lab, bins, tr, ho, AuxDiscoverySettings(), nbins=1, all_pairs=True)
    assert dflt and all("nonfinite_corr_cv1" in c.fail for c in dflt)          # undeclared drop still fails
    best, allc = Z.search_z3(ft, lab, bins, tr, ho, AuxDiscoverySettings(), nbins=1, all_pairs=True,
                             cond_dims=["cv2"])
    assert best is not None and best.passed and "cv1_not_applicable" in best.notes
    ft.cv2 = None                                                # CV1-only campaign, cv1 the only dimension
    ft.cv1 = np.random.default_rng(1).normal(size=ft.n).astype(np.float32)
    best, _ = Z.search_z3(ft, lab, bins, tr, ho, AuxDiscoverySettings(), nbins=1, all_pairs=True, cond_dims=["cv1"])
    assert best is not None and best.passed and "cv2_not_applicable" in best.notes
