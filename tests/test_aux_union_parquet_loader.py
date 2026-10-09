"""Task 14: the analyzer's union-Parquet loader pools admitted aux workers."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aux_union_fixture import MODEL, build_campaign, expected_reduced  # noqa: E402

from gareus.kernel_identity import AuxPoolingRefused  # noqa: E402
from gareus.mbar_analysis.loaders_union_parquet import load_parquet_adaptive_union  # noqa: E402


def _load(camp, **kw):
    return load_parquet_adaptive_union(camp.ad, n_workers=1, **kw)


def test_loader_pools_worker(tmp_path):
    camp = build_campaign(tmp_path)
    d = _load(camp)
    assert d.meta["aux_states"] == [2]
    assert d.meta["aux_model_sha256"] == MODEL.model_sha256
    assert d.meta["aux_parent_state"] == {2: 0}
    assert d.meta["aux_forecast_O"] == {2: None}
    assert d.aux_z is not None and d.aux_z.shape == d.cv.shape and np.isfinite(d.aux_z).all()
    zt = {k[2]: z for k, z in camp.truth.items()}
    np.testing.assert_allclose(d.aux_z, [zt[int(s)] for s in d.step], rtol=0, atol=1e-12)
    # burn-in: the worker keeps only its final-phase rows; ordinary states keep all three phases
    win = np.asarray(d.window)
    assert (win == 2).sum() == 40
    assert (win == 0).sum() == 3 * 40 and (win == 1).sum() == 3 * 40
    assert d.meta["aux_burnin_dropped"] == {"2": 40}
    # reduced bias of every row equals the strict shared reconstruction from the truth z (steps are unique)
    z_by_step = {k[2]: z for k, z in camp.truth.items()}
    z = np.asarray([z_by_step[int(s)] for s in d.step])
    expected = expected_reduced(d.cv, z, d.beta)
    np.testing.assert_allclose(np.asarray(d.u_nk), expected, rtol=1e-10, atol=1e-10)


def test_loader_equals_driver_builder(tmp_path):
    from gareus.adaptive_production import build_union_state_mbar_inputs
    camp = build_campaign(tmp_path)
    d = _load(camp)
    meta = build_union_state_mbar_inputs(camp.ad, camp.registry, output_prefix="drv")
    npz = np.load(meta["arrays_npz"])
    # the builder subsamples; compare every builder row against the loader's row with the same step
    import csv
    with open(meta["samples_csv"], newline="") as fh:
        steps = [int(r["step"]) for r in csv.DictReader(fh)]
    pos = {int(s): i for i, s in enumerate(d.step)}
    idx = [pos[s] for s in steps]
    np.testing.assert_allclose(np.asarray(d.u_nk)[idx], npz["umbrella_reduced_bias_nk"], rtol=1e-10, atol=1e-10)


def test_loader_refuses_without_admission(tmp_path):
    camp = build_campaign(tmp_path, admission=False)
    with pytest.raises(AuxPoolingRefused):
        _load(camp)


def test_loader_refuses_missing_backfill(tmp_path):
    camp = build_campaign(tmp_path, backfill=False)
    with pytest.raises(AuxPoolingRefused, match="backfill"):
        _load(camp)


def test_loader_low_memory_matches(tmp_path):
    camp = build_campaign(tmp_path)
    a = _load(camp)
    b = _load(camp, low_memory=True)
    np.testing.assert_allclose(np.asarray(b.u_nk), np.asarray(a.u_nk), rtol=1e-12, atol=1e-12)


def test_loader_refuses_when_registry_json_missing(tmp_path):
    camp = build_campaign(tmp_path)
    (camp.ad / "state_registry.json").unlink()
    with pytest.raises(AuxPoolingRefused, match="state_registry.json is missing"):
        _load(camp)


def test_loader_refuses_when_admitted_worker_not_in_registry(tmp_path):
    import json
    camp = build_campaign(tmp_path)
    p = camp.ad / "aux_admission.json"
    rec = json.loads(p.read_text())
    rec["workers"][0]["aux_center"] = 9.9
    p.write_text(json.dumps(rec))
    with pytest.raises(AuxPoolingRefused, match="no matching worker state"):
        _load(camp)
