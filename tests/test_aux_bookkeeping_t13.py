import argparse
import json
from pathlib import Path

from gareus.adaptive.aux_admission_io import ADMISSION_FILENAME, reconcile_admission_with_registry
from gareus.adaptive.aux_discovery.settings import aux_discovery_incompatibilities
from test_aux_admission_registry import _add_worker, _registry


def _args(**kw):
    base = dict(window_mode="adaptive-production", exchange_mode="gibbs-walk", run_mode="gamd",
                gamd_boost_type="pep-gamd-lower-dual", traj_interval=250, distance_output_interval=250,
                exchange_interval=500, ap_topups=False, us_auto_drop_bad_windows=False)
    base.update(kw)
    return argparse.Namespace(**base)


def test_resume_reads_frozen_reserve_slots_not_job_flag():
    args = _args(ap_aux_reserve_slots=0)
    assert any("reserve-slots" in b for b in aux_discovery_incompatibilities(args))
    assert aux_discovery_incompatibilities(args, reserve_slots=4) == []
    assert any("reserve-slots" in b for b in aux_discovery_incompatibilities(_args(ap_aux_reserve_slots=4), reserve_slots=0))


def _worker(parent, c3):
    return {"parent_state_id": parent, "aux_center": c3, "aux_k_kcal_mol": 2.0}


def test_refused_workers_not_duplicated_on_replay(tmp_path: Path):
    r = _registry()
    _add_worker(r, parent=0, c3=1.5)
    w1, w2 = _worker(0, 1.5), _worker(1, 2.5)
    path = tmp_path / ADMISSION_FILENAME
    path.write_text(json.dumps({"workers": [w1, w2], "refused_workers": [w2]}))
    reconcile_admission_with_registry(tmp_path, r)
    first = json.loads(path.read_text())
    assert first["refused_workers"] == [w2]
    # kill-replay: the record again lists the refused worker among its workers
    path.write_text(json.dumps({**first, "workers": [w1, w2]}))
    reconcile_admission_with_registry(tmp_path, r)
    assert json.loads(path.read_text())["refused_workers"] == [w2]
