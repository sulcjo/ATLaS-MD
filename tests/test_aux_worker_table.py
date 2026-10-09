import numpy as np
from types import SimpleNamespace

from gareus.adaptive.aux_worker_table import positive_residence_episodes, worker_table


def test_entries_exits_and_zero_time_swaps_collapsed():
    series = np.array([3, 3, 9, 9, 9, 3, 3, 9, 3])
    residence = np.array([1, 1, 1, 1, 1, 1, 1, 0, 1])
    ep = positive_residence_episodes(series, residence, aux_states={9})
    assert ep["entries"] == 1 and ep["exits"] == 1 and ep["zero_time_visits"] == 1


def test_worker_table_toy():
    rng = np.random.default_rng(1)
    n = 3000
    win = np.repeat([0, 1, 2], n)
    x = rng.normal([0.0, 0.5, 0.25][0], 1.0, 3 * n)
    u = np.zeros((3 * n, 3))
    rep = np.tile(np.arange(4), 3 * n // 4 + 1)[:3 * n]
    d = SimpleNamespace(u_nk=u, window=win, replica=rep, step=np.arange(3 * n),
                        aux_z=x, meta={"aux_parent_state": {2: 0}})
    rows = worker_table(d, np.zeros(3), aux_states=[2], ordinary_states=[0, 1])
    r = rows[0]
    assert r["state_id"] == 2 and r["parent_state_id"] == 0
    assert r["best_partner_overlap"] > 0.15 and r["overlap_floor_ok"] is True
    assert r["n_samples"] == n and r["carrier_diversity"] == 4
    assert r["return_label_status"] == "frames_unavailable"
    assert "no shams" in r["attribution"]
