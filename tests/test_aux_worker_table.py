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


def _two_worker_data(lams=None):
    # one replica; ordinary 0 (lam 0), 1 (lam 0.5); workers 2 and 3
    series = [0, 0, 2, 2, 0, 3, 0, 2, 0, 3, 3, 0]
    win = np.array(series * 300)
    n = win.size
    rng = np.random.default_rng(0)
    u = np.zeros((n, 4))
    return SimpleNamespace(u_nk=u, window=win, replica=np.zeros(n, int), step=np.arange(n), meta={})


def test_entries_exits_are_per_worker():
    d = _two_worker_data()
    rows = {r["state_id"]: r for r in worker_table(d, np.zeros(4), aux_states=[2, 3], ordinary_states=[0])}
    # per 12-block: worker 2 enters twice, worker 3 twice; but block boundary 0->0 so counts scale
    assert rows[2]["entries"] == 2 * 300 and rows[2]["exits"] == 2 * 300
    assert rows[3]["entries"] == 2 * 300 and rows[3]["exits"] == 2 * 300
    assert "zero_time_visits" not in rows[2]


def test_distinct_per_worker_counts():
    series = [0, 2, 0, 2, 0, 2, 0, 3, 0]
    win = np.array(series)
    n = win.size
    d = SimpleNamespace(u_nk=np.zeros((n, 4)), window=win, replica=np.zeros(n, int), step=np.arange(n), meta={})
    rows = {r["state_id"]: r for r in worker_table(d, np.zeros(4), aux_states=[2, 3], ordinary_states=[0])}
    assert rows[2]["entries"] == 3 and rows[3]["entries"] == 1


def test_best_partner_restricted_to_lambda_zero():
    rng = np.random.default_rng(2)
    n = 2000
    win = np.repeat([0, 1, 2], n)
    # u: state 1 identical to worker 2 (full overlap), state 0 far (offset in u)
    x = rng.normal(0, 1, 3 * n)
    u = np.c_[0.5 * (x - 3.0) ** 2, 0.5 * x ** 2 + 0.0, 0.5 * x ** 2]
    u[n:2 * n, 0] = 0.5 * (x[n:2 * n] - 3.0) ** 2
    d = SimpleNamespace(u_nk=u, window=win, replica=np.tile(np.arange(2), 3 * n // 2), step=np.arange(3 * n), meta={})
    from gareus.adaptive.mbar_solve import solve_rows
    f, _ = solve_rows(u, win)
    r = worker_table(d, f, aux_states=[2], ordinary_states=[0, 1], state_lambdas=[0.0, 0.5, 0.0])[0]
    assert r["best_partner"] == 0
    r2 = worker_table(d, f, aux_states=[2], ordinary_states=[0, 1], state_lambdas=[0.0, 0.0, 0.0])[0]
    assert r2["best_partner"] == 1
