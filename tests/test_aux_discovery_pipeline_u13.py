"""U13: one contacts+H-bond partition; torsions never enter it."""
from types import SimpleNamespace

import numpy as np

from gareus.adaptive.aux_discovery import pipeline as P
from gareus.adaptive.aux_discovery.settings import AuxDiscoverySettings


def _ft(n=3000, seed=0, tors_linked=True, n_lin=20):
    rng = np.random.default_rng(seed)
    lab = rng.integers(0, 2, n)
    hc = rng.normal(size=(n, 12)) + 3.0 * lab[:, None] * (np.arange(12) < 6)
    hb = rng.normal(size=(n, 6))
    ang = rng.normal(scale=0.3, size=(n, 18))
    if tors_linked:
        ang[:, 3] += np.where(lab == 1, 1.5, -1.5)
    tors = np.stack([np.sin(ang), np.cos(ang)], 2).reshape(n, 36)
    lineage = np.array([f"p:{r % n_lin}" for r in range(n)]); step = np.arange(n) // n_lin
    ft = SimpleNamespace(
        n=n, tors=tors, hc=hc, hb=hb, cv1=rng.normal(size=n).astype(np.float32),
        cv2=rng.normal(size=n).astype(np.float32), basin=rng.integers(0, 5, (n, 8)).astype(np.uint8),
        replica=rng.integers(0, 40, n), lineage=lineage, step=step, state_id=np.zeros(n, int),
        definition=SimpleNamespace(hc_labels=[f"c{i}" for i in range(12)], hb_labels=[f"h{i}" for i in range(6)],
                                   schema_sha256="0" * 64))
    return ft


def _split(n):
    tr = np.arange(n) < int(0.7 * n)
    return tr, ~tr


def test_torsions_never_enter_the_partition(monkeypatch):
    ft = _ft(n=400)
    seen = []

    def fake(X, fam, *a, **k):
        seen.append((np.asarray(X).shape, list(fam), k.get("multi_k"), k.get("feature_names")))
        return SimpleNamespace(status="insufficient_evidence", choice=SimpleNamespace(k=None, table=[]),
                               hidden_fraction=None, co_occurrence=None, lineage_info=None, per_k=None,
                               bins=None, frozen=None)

    monkeypatch.setattr(P, "fit_partition", fake)
    tr, ho = _split(ft.n)
    res = P.run_discovery(ft, train=tr, holdout=ho, settings=AuxDiscoverySettings(), full_topology=None,
                          k3_max=3, epoch=1)
    assert len(seen) == 1
    shape, fam, multi, names = seen[0]
    assert shape == (ft.n, ft.hc.shape[1] + ft.hb.shape[1])
    assert set(fam) == {"hc", "hb"} and multi is True and len(names) == shape[1]
    assert res.status == "insufficient_evidence"
    assert res.report["partition"] is res.report["discovery"] and "evaluation" not in res.report


def test_noise_control_is_broaden():
    ft = _ft(tors_linked=False, seed=1)
    tr, ho = _split(ft.n)
    res = P.run_discovery(ft, train=tr, holdout=ho, settings=AuxDiscoverySettings(), full_topology=None,
                          k3_max=3, epoch=1)
    assert res.status == "broaden"
    assert res.model is None and res.placement is None
    assert not res.report["z3_search"].get("null_gate", {"passed": False})["passed"]


def test_planted_torsion_predicting_contact_groups_passes(monkeypatch):
    ft = _ft(tors_linked=True, seed=2)
    tr, ho = _split(ft.n)
    monkeypatch.setattr(P, "emit_model", lambda *a, **k: SimpleNamespace(model_sha256="x"))
    monkeypatch.setattr(P, "place_workers", lambda *a, **k: {
        "n_states": 0, "n_candidates": 0, "n_eligible": 0, "skipped_states": [], "selection_log": [],
        "chosen": [{"state_id": 0}]})
    res = P.run_discovery(ft, train=tr, holdout=ho, settings=AuxDiscoverySettings(), full_topology=None,
                          k3_max=3, epoch=1)
    assert res.status == "ok", res.report.get("partition")
    assert res.report["z3_search"]["null_gate"]["passed"]
    assert res.eval_partition is not None
