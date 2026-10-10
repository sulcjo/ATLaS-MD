"""F12: heuristic crosscheck labels, report grading, blocked-vs-dense parity, trapped-lineage null, history."""
import json

import numpy as np
import pytest
from scipy.special import logsumexp

from gareus.mbar_analysis.crosscheck import _target_logw


def _dense(u, window, f):
    n_k = np.bincount(window, minlength=u.shape[1]).astype(float)
    with np.errstate(divide="ignore"):
        logn = np.log(n_k)
    a = logn[None, :] + np.asarray(f, float)[None, :] - u
    a = np.where(np.isfinite(a), a, -np.inf)
    return -logsumexp(a, axis=1)


@pytest.mark.parametrize("block", [1, 7, 65536])
def test_blocked_target_logw_matches_dense(block):
    rng = np.random.default_rng(1)
    u = rng.normal(size=(100, 5)); u[rng.random(u.shape) < 0.1] = np.inf
    u[:, 0] = np.nan_to_num(u[:, 0], posinf=0.0)
    window = rng.integers(0, 5, 100); f = rng.normal(size=5)
    assert np.max(np.abs(_target_logw(u, window, f, block_rows=block) - _dense(u, window, f))) <= 1e-12


def _row(s, name="Aux ordinary-only crosscheck"):
    from gareus_report import build_health_verdict
    v = build_health_verdict(s)
    return next(c for c in v["checks"] if c["name"] == name), v


@pytest.mark.parametrize("status,expected", [("heuristic_fail", "fail"), ("heuristic_pass", "caution"),
                                              ("error", "caution"), ("skipped", "na"), ("unavailable", "na"),
                                              ("pass", "caution"), ("weird", "caution")])
def test_grading_table_never_pass(status, expected):
    r, _ = _row({"aux_crosscheck": {"status": status, "reason": "x", "max_abs_diff_kcal": 0.1}})
    assert r["status"] == expected
    if status == "heuristic_pass":
        assert "heuristic agreement, no statistical test" in r["detail"]


def test_integrity_failure_takes_precedence():
    r, v = _row({"aux_crosscheck": {"status": "heuristic_pass"}, "aux_integrity_failure": "pooling refused"})
    assert r["status"] == "fail" and v["overall"] == "FAIL"


def test_trapped_lineages_null_label():
    from types import SimpleNamespace
    from gareus.adaptive.aux_discovery.z3_search import NULL_UNINFORMATIVE_STATUS, z3_null_gate
    lineage = np.repeat([0, 1, 2], 10)
    labels = np.repeat([0, 1, 1], 10)           # constant within every lineage
    ft = SimpleNamespace(lineage=lineage, step=np.arange(30))
    best = SimpleNamespace(info_gain=0.5)
    g = z3_null_gate(ft, [(2, labels, [])], best, None, None, None, SimpleNamespace(n_null_z3=3, partition_seed=0))
    assert g["status"] == NULL_UNINFORMATIVE_STATUS and g["passed"] is False


def test_history_appends_across_boundaries(tmp_path):
    from gareus.adaptive.aux_admission_io import HISTORY_FILENAME, append_discovery_history
    append_discovery_history(tmp_path, {"status": "broaden", "n_holdout": 10}, 1)
    append_discovery_history(tmp_path, {"status": "ok", "n_holdout": 12}, 2)
    rec = json.loads((tmp_path / HISTORY_FILENAME).read_text())
    assert [a["boundary_epoch"] for a in rec["attempts"]] == [1, 2]
    assert [a["status"] for a in rec["attempts"]] == ["broaden", "ok"]
    assert "NO campaign-wide false-discovery control" in rec["note"]


def test_history_kill_replay_replaces_the_same_boundary_entry(tmp_path):
    from gareus.adaptive.aux_admission_io import HISTORY_FILENAME, append_discovery_history
    append_discovery_history(tmp_path, {"status": "broaden", "n_holdout": 10}, 1)
    append_discovery_history(tmp_path, {"status": "error", "n_holdout": 12}, 2)
    append_discovery_history(tmp_path, {"status": "ok", "n_holdout": 12}, 2)       # epoch 2 replayed after a kill
    rec = json.loads((tmp_path / HISTORY_FILENAME).read_text())
    assert [a["boundary_epoch"] for a in rec["attempts"]] == [1, 2]
    assert [a["status"] for a in rec["attempts"]] == ["broaden", "ok"]
    assert [a["attempt_index"] for a in rec["attempts"]] == [1, 2]


@pytest.mark.parametrize("payload", ["{not json", json.dumps({"schema": "other", "attempts": []}),
                                     json.dumps({"schema": "aux_discovery_history_v1", "attempts": {"a": 1}})])
def test_unreadable_history_is_never_overwritten(tmp_path, capsys, payload):
    from gareus.adaptive.aux_admission_io import HISTORY_FILENAME, append_discovery_history
    path = tmp_path / HISTORY_FILENAME
    path.write_text(payload)
    append_discovery_history(tmp_path, {"status": "ok", "n_holdout": 12}, 2)       # never raises
    assert path.read_text() == payload
    err = capsys.readouterr().out
    assert "WARNING" in err and HISTORY_FILENAME in err


def test_history_note_says_holdout_becomes_training():
    from gareus.adaptive.aux_admission_io import HISTORY_NOTE
    assert "boundary epoch" in HISTORY_NOTE and "training" in HISTORY_NOTE
    assert "NO campaign-wide false-discovery control" in HISTORY_NOTE
