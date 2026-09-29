"""Epoch rung-sample loader and the dry-run ladder replay (fixtures: ladder_adapt_fixture)."""
import json

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from gareus.adaptive.ladder_adapt import load_centre_rung_samples, phase_dirs, replay
from ladder_adapt_fixture import ENV, _sample_at  # noqa: E402  (tests dir on sys.path)


def _fixture(tmp_path):
    ad = tmp_path / "adaptive_production"
    ep = ad / "epoch_000"
    (ep / "samples" / "seg_001").mkdir(parents=True)
    rows, wmap, states, w = [], [], [], 0
    for c, cv1 in enumerate((0.3, 0.7)):
        for i, lam in enumerate((0.0, 0.5, 1.0)):
            vp, vd = _sample_at(lam, 800, 1000 + 10 * c + i)
            for t in range(vp.size):
                rows.append(dict(replica=w, step=250 * (t + 1), window_id=w, cv1=cv1, cv2=0.0,
                                 gamd_lambda=lam, v_pep_kj_mol=vp[t], v_dih_kj_mol=vd[t]))
            wmap.append(dict(epoch_window=w, state_id=w, primary_center=cv1, secondary_center=0.0))
            states.append(dict(state_id=w, active=True, primary_center=cv1, primary_k=300.0,
                               secondary_center=0.0, secondary_k=1.0, gamd_lambda=lam))
            w += 1
    rows.append(dict(rows[0]))                                   # duplicated (replica, step)
    pq.write_table(pa.Table.from_pandas(pd.DataFrame(rows)), ep / "samples" / "seg_001" / "data.parquet")
    (ep / "segments.json").write_text(json.dumps({"segments": []}))
    pd.DataFrame(wmap).to_csv(ep / "epoch_window_map.csv", index=False)
    (ad / "state_registry.json").write_text(json.dumps({"schema_version": 1, "states": states}))
    (ad / "global_shared_gamd_setup").mkdir()
    (ad / "global_shared_gamd_setup" / "shared_gamd_setup_globals.json").write_text(json.dumps({
        "temperature_K": 300.0,
        "all_globals": {"Vmax_Dihedral": ENV.vmax_dih, "Vmin_Dihedral": ENV.vmin_dih,
                        "threshold_energy_Dihedral": ENV.threshold_dih, "k0_Dihedral": ENV.k0max_dih}}))
    return ad, states


def test_loader_groups_by_centre_and_rung_and_dedupes(tmp_path):
    ad, states = _fixture(tmp_path)
    got = load_centre_rung_samples(phase_dirs(ad), states)
    assert len(got) == 2 and all(sorted(v) == [0.0, 0.5, 1.0] for v in got.values())
    assert sum(len(v[0.0][1]) for v in got.values()) == 1600      # 2 centres x 800, duplicate once


def test_replay_reports_current_ladder_and_a_design(tmp_path):
    ad, _ = _fixture(tmp_path)
    rep = replay(ad, target=0.25)
    assert rep["current"] == [0.0, 0.5, 1.0]
    assert rep["n_centres"] == 2
    assert rep["design"]["lambdas"][0] == 0.0 and rep["design"]["lambdas"][-1] == 1.0
    assert rep["design"]["feasible"] is True
    json.dumps(rep)                                                # JSON-able


def test_phase_filter_selects_named_phases(tmp_path):
    ad, _ = _fixture(tmp_path)
    assert len(phase_dirs(ad, names=["epoch_000"])) == 1
    assert phase_dirs(ad, names=["final"]) == []
