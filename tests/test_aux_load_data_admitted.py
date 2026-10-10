"""Final fix wave C3/I1: the analyzer's load_data pools an admitted aux campaign through the union-Parquet
loader (never refuses it up front), never through the union-NPZ snapshot, and the analysis is capped
(aux_models set -> ladder/overlap rows unavailable, banner never PASS) while the aux crosscheck still runs."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aux_union_fixture import MODEL, build_campaign  # noqa: E402

from gareus.kernel_identity import AuxPoolingRefused  # noqa: E402
from gareus.mbar_analysis.loaders import load_data  # noqa: E402


def _campaign(tmp_path, **kw):
    camp = build_campaign(tmp_path, **kw)
    prod = tmp_path / "adaptive_production"
    camp.ad.rename(prod)
    return prod


def test_load_data_pools_an_admitted_campaign(tmp_path):
    prod = _campaign(tmp_path)
    d = load_data(prod.parent, tmp_path / "out", n_workers=1)
    assert d.meta["aux_states"] == [2]
    assert d.meta["aux_models"] == [MODEL.model_sha256]
    assert d.aux_z is not None and np.isfinite(d.aux_z).all()


def test_load_data_admitted_low_memory_never_reads_the_union_npz(tmp_path):
    prod = _campaign(tmp_path)
    (prod / "adaptive_union_mbar.npz").write_bytes(b"not an npz")     # would raise if it were read
    d = load_data(prod.parent, tmp_path / "out", n_workers=1, low_memory=True)
    assert d.meta["aux_models"] == [MODEL.model_sha256]


def test_load_data_without_admission_still_refuses(tmp_path):
    prod = _campaign(tmp_path, admission=False)
    with pytest.raises(AuxPoolingRefused, match="auxiliary-CV run"):
        load_data(prod.parent, tmp_path / "out", n_workers=1)


def test_admitted_aux_analysis_is_never_pass(tmp_path):
    import analyze_gareus_mbar as agm
    from test_per_regime_full_analysis import _args
    prod = _campaign(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    d = load_data(prod.parent, out, n_workers=1)
    args = _args()
    args.no_extra_pmfs = True
    args.no_convergence = True
    args.no_basin_tracking = True
    agm.analyze(d, args)
    s = json.loads((Path(d.out_dir) / "pmf_summary.json").read_text())
    assert s["aux_states"]["models"] == [MODEL.model_sha256]
    assert s["health"]["overall"] != "PASS"
    assert any(c["name"] == "Auxiliary states" and c["status"] == "caution" for c in s["health"]["checks"])
    assert "aux_crosscheck" in s and "aux_workers" in s          # the aux-specific checks still run
    excl = s["aux_burnin_exclusions"]                             # F01: the exclusion record reaches the summary
    assert excl["rule"] == "aux_burnin_carrier_v1"
    assert [(r["phase"], r["state_id"], r["reason"]) for r in excl["records"]] == [("epoch_001", 2, "aux_burnin_carrier")]
