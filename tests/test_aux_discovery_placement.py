import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gareus.adaptive.aux_discovery.placement import place_workers
from gareus.adaptive.aux_discovery.settings import AuxDiscoverySettings

C10 = Path("/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/docs/_local_docs/c10_aux_diagnosis")
TRAIN = {"epoch_000", "epoch_001/baseline"}
HOLD = "epoch_002/baseline"


@pytest.mark.skipif(not C10.is_dir(), reason="c10 local diagnosis data absent")
def test_parity_with_c10_placement_json():
    g = pd.read_parquet(C10 / "group_labels.parquet", columns=["phase", "replica", "step", "state_id", "lam"])
    z = np.load(C10 / "z_torsion_only.npy") / 3.181
    lab = np.load(C10 / "eval_labels_frozen.npy")
    m = (g.lam == 0).to_numpy()
    g = g[m]
    z = z[m]
    lab = lab[m]
    lineage = (g.phase + ":" + g.replica.astype(str)).to_numpy()
    out = place_workers(z, lab, g.state_id.to_numpy(), lineage, g.step.to_numpy(),
                        g.phase.isin(TRAIN).to_numpy(), (g.phase == HOLD).to_numpy(),
                        AuxDiscoverySettings(), k_labels=5)
    ref = json.loads((C10 / "placement.json").read_text())
    assert out["all_candidates"] == ref["all_candidates"]
    assert out["chosen"] == ref["chosen"] and out["selection_log"] == ref["selection_log"]
    assert out["top20"] == ref["top20"] and out["skipped_states"] == ref["skipped_states"]
    assert out["doc"] == ref["doc"] and out["gates"] == ref["gates"]
    assert [k for k in out] == [k for k in ref]
    assert (out["n_states"], out["n_candidates"], out["n_eligible"]) == (
        ref["n_states"], ref["n_candidates"], ref["n_eligible"])


def test_synthetic_selects_at_most_max_workers_and_one_per_side():
    rng = np.random.default_rng(0)
    n_states, per = 6, 600
    sid = np.repeat(np.arange(n_states), per)
    lineage = np.array([f"e0:{i % 30}" for i in range(n_states * per)])
    step = np.tile(np.arange(per) * 3000, n_states)
    z = rng.normal(size=sid.size)
    lab = (z + 0.5 * rng.normal(size=sid.size) > 0).astype(int)
    is_train = np.tile(np.arange(per) < 450, n_states)
    is_ho = ~is_train
    out = place_workers(z, lab, sid, lineage, step, is_train, is_ho, AuxDiscoverySettings(), k_labels=2)
    keys = [(c["state_id"], c["side"]) for c in out["chosen"]]
    assert len(out["chosen"]) <= 4 and len(keys) == len(set(keys))
    for c in out["chosen"]:
        assert c["heldout"]["net"] > 0 and c["heldout"]["O"] >= 0.15


def test_k3_cap_marks_ineligible():
    rng = np.random.default_rng(1)
    sid = np.repeat([0], 800)
    lineage = np.array([f"e0:{i % 30}" for i in range(800)])
    z = rng.normal(size=800)
    lab = (z > 0).astype(int)
    step = np.arange(800) * 3000
    tr = np.arange(800) < 600
    out = place_workers(z, lab, sid, lineage, step, tr, ~tr, AuxDiscoverySettings(), k_labels=2, k3_max=0.01)
    assert out["n_eligible"] == 0 and out["chosen"] == []
    assert all(c["gates"]["k3_max"] is False and c["eligible"] is False for c in out["all_candidates"])


def _overlap(du, what="test"):
    from gareus.adaptive.aux_discovery.placement import _forecast
    du = np.asarray(du, dtype=float)
    z = np.sqrt(du) if np.all(np.isfinite(du)) else du
    n = len(du)
    out = _forecast(z, np.zeros(n, dtype=int), np.zeros(n, dtype=int), 0.0, 2.0, [0], 1.0, 1)
    return out["O"]


def test_forecast_overlap_stable_when_exp_minus_du_underflows():
    assert _overlap([800.0] + [1000.0] * 99) == pytest.approx(0.009900990099, rel=1e-9)


def test_forecast_overlap_invariant_to_common_energy_constant():
    base = np.array([1.0, 2.5, 4.0, 7.0, 0.5])
    assert _overlap(base + 900.0) == pytest.approx(_overlap(base), rel=1e-9)


def test_forecast_overlap_identical_energies_is_one_half():
    assert _overlap([3.0] * 50) == pytest.approx(0.5, abs=1e-12)


def test_forecast_overlap_extreme_separation_is_finite_and_tends_to_1_over_n_plus_1():
    o = _overlap([0.0] + [1e6] * 9)
    assert np.isfinite(o) and o == pytest.approx(1 / 11, rel=1e-6)


def test_forecast_refuses_nonfinite_energy():
    from gareus.adaptive.aux_discovery.placement import _forecast
    for bad in (np.nan, np.inf):
        z = np.array([0.0, 1.0, bad])
        with pytest.raises(ValueError, match="parent 7"):
            _forecast(z, np.zeros(3, dtype=int), np.zeros(3, dtype=int), 0.0, 2.0, [0], 1.0, 1, what="parent 7")
