"""Task 13: the driver's union MBAR inputs pool admitted aux workers."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aux_union_fixture import MODEL, build_campaign, expected_reduced  # noqa: E402

from gareus.adaptive_production import build_union_state_mbar_inputs  # noqa: E402
from gareus.kernel_identity import AuxPoolingRefused  # noqa: E402


def _truth_z(camp, meta):
    import csv
    with open(meta["samples_csv"], newline="") as fh:
        recs = list(csv.DictReader(fh))
    return recs


def test_union_bias_equals_direct_energies(tmp_path):
    camp = build_campaign(tmp_path)
    meta = build_union_state_mbar_inputs(camp.ad, camp.registry)
    d = np.load(meta["arrays_npz"])
    u = d["umbrella_reduced_bias_nk"]
    assert u.shape[0] > 20 and u.shape[1] == 3
    # recorded/backfilled z in the NPZ equals the truth of that sample (source, step)
    recs = _truth_z(camp, meta)
    by_step = {k[2]: z for k, z in camp.truth.items()}      # steps are unique across the campaign
    z_truth = np.asarray([by_step[int(r["step"])] for r in recs])
    np.testing.assert_allclose(d["aux_z"], z_truth, rtol=0, atol=1e-12)
    assert len(recs) == u.shape[0]
    expected = expected_reduced(d["cv_A"], d["aux_z"], meta["beta_1_over_kJ_mol"])
    np.testing.assert_allclose(u, expected, rtol=1e-10, atol=1e-10)
    assert list(d["aux_k"]) == [0.0, 0.0, 2.0] and np.isnan(d["aux_center"][:2]).all() and d["aux_center"][2] == 0.4
    assert meta["aux_model_sha256"] == MODEL.model_sha256
    # pre-admission rows really were evaluated with the backfilled z (worker column is not the CV1 term alone)
    assert any(r["source"] == "epoch_000" for r in recs)


def test_missing_backfill_refuses(tmp_path):
    camp = build_campaign(tmp_path, backfill=False)
    with pytest.raises(AuxPoolingRefused, match="backfill"):
        build_union_state_mbar_inputs(camp.ad, camp.registry)


def test_incomplete_backfill_refuses_with_counts(tmp_path):
    import pyarrow.parquet as pq
    from gareus.adaptive.aux_backfill import BACKFILL_FILENAME
    camp = build_campaign(tmp_path)
    p = camp.ad / "epoch_000" / BACKFILL_FILENAME
    t = pq.read_table(p)
    pq.write_table(t.slice(0, t.num_rows - 3), p)
    import hashlib
    import json
    rp = camp.ad / "aux_admission.json"           # the record hashes the truncated file: incompleteness decides
    rec = json.loads(rp.read_text())
    rec["backfill"][0]["sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
    rp.write_text(json.dumps(rec))
    with pytest.raises(AuxPoolingRefused, match=r"3 rows without aux z"):
        build_union_state_mbar_inputs(camp.ad, camp.registry)


def test_worker_burnin_rows_excluded(tmp_path):
    camp = build_campaign(tmp_path)
    meta = build_union_state_mbar_inputs(camp.ad, camp.registry)
    rec = meta["subsample_counts_per_state"][str(camp.worker_id)]
    assert rec["burnin_dropped"] == 40            # all of epoch_001's worker rows
    assert rec["raw"] == 40                       # only the final-phase worker rows remain
    d = np.load(meta["arrays_npz"])
    import csv
    with open(meta["samples_csv"], newline="") as fh:
        w_rows = [r for r in csv.DictReader(fh) if int(r["sampled_state_id"]) == camp.worker_id]
    assert w_rows and {r["source"] for r in w_rows} == {"final"}
    assert d["N_k"][2] == len(w_rows)
    # ordinary states keep their epoch_001 rows and carry no burnin_dropped record
    assert "burnin_dropped" not in meta["subsample_counts_per_state"]["0"]


def test_without_admission_still_refuses_aux_snapshots(tmp_path):
    camp = build_campaign(tmp_path, admission=False)
    with pytest.raises(AuxPoolingRefused):
        build_union_state_mbar_inputs(camp.ad, camp.registry)


def test_admission_with_wrong_model_sha_refuses(tmp_path):
    import json
    camp = build_campaign(tmp_path)
    p = camp.ad / "aux_admission.json"
    rec = json.loads(p.read_text())
    rec["model_sha256"] = "0" * 64
    p.write_text(json.dumps(rec))
    with pytest.raises(AuxPoolingRefused, match="aux_model.json"):
        build_union_state_mbar_inputs(camp.ad, camp.registry)


def test_no_workers_outputs_unchanged(tmp_path):
    """No worker in the registry: an admission record adds no aux arrays/keys and no z requirement."""
    from gareus.adaptive_production import WindowStateRegistry
    import json
    camp = build_campaign(tmp_path)
    p = camp.ad / "aux_admission.json"
    rec = json.loads(p.read_text())
    rec["workers"] = []
    p.write_text(json.dumps(rec))
    reg = WindowStateRegistry()
    reg.add_state(0.2, 10.0)
    reg.add_state(0.5, 10.0)
    meta = build_union_state_mbar_inputs(camp.ad, reg, include_epochs=False, output_prefix="plain")
    d = np.load(meta["arrays_npz"])
    assert "aux_z" not in d.files and "aux_model_sha256" not in meta
    assert all("burnin_dropped" not in v for v in meta["subsample_counts_per_state"].values())
    assert "aux_burnin_exclusions" not in meta


def test_builder_refuses_when_admitted_worker_not_in_registry(tmp_path):
    from gareus.adaptive_production import WindowStateRegistry
    camp = build_campaign(tmp_path)
    reg = WindowStateRegistry()
    reg.add_state(0.2, 10.0)
    reg.add_state(0.5, 10.0)
    with pytest.raises(AuxPoolingRefused, match="no matching worker state"):
        build_union_state_mbar_inputs(camp.ad, reg, output_prefix="x")
