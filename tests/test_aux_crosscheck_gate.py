import numpy as np
from types import SimpleNamespace

from gareus.adaptive.mbar_solve import solve_rows
from gareus.mbar_analysis.crosscheck import aux_ordinary_crosscheck


def _toy(bias_bug=False, seed=0):
    """3 harmonic states on x ~ N; state 2 is an aux state with an extra bias along z = x."""
    rng = np.random.default_rng(seed)
    centers, ks = np.array([-1.0, 1.0, 0.0]), np.array([2.0, 2.0, 2.0])
    aux_c, aux_k = 0.8, 3.0
    xs, win = [], []
    for j in range(3):
        kk = ks[j] + (aux_k if j == 2 else 0.0)
        mu = (ks[j] * centers[j] + (aux_k * aux_c if j == 2 else 0.0)) / kk
        x = rng.normal(mu, np.sqrt(0.596 / kk), 4000); xs.append(x); win += [j] * 4000
    x = np.concatenate(xs); win = np.array(win)
    u = 0.5 * ks[None, :] * (x[:, None] - centers[None, :]) ** 2
    u[:, 2] += 0.5 * (0.0 if bias_bug else aux_k) * (x - aux_c) ** 2
    return x, win, u / 0.596


def _run(x, win, u, ordinary):
    f, _ = solve_rows(u, win)
    d = SimpleNamespace(u_nk=u, window=win, cv=x, cv2=np.full_like(x, np.nan), aux_z=x)
    return aux_ordinary_crosscheck(d, f, {"cv1": np.linspace(-2, 2, 21)}, 0.596, ordinary_states=ordinary)


def test_consistent_aux_passes():
    r = _run(*_toy(), ordinary=[0, 1])
    assert r["status"] == "heuristic_pass" and r["method"] == "raw_count_heuristic", r["axes"]["cv1"].get("max_abs_diff_kcal")


def test_missing_aux_term_fails():
    r = _run(*_toy(bias_bug=True), ordinary=[0, 1])
    assert r["status"] == "heuristic_fail"


def test_too_few_bins_is_skipped():
    x, win, u = _toy()
    f, _ = solve_rows(u, win)
    d = SimpleNamespace(u_nk=u, window=win, cv=x, cv2=np.full_like(x, np.nan))
    r = aux_ordinary_crosscheck(d, f, {"cv1": np.linspace(50, 60, 5)}, 0.596, ordinary_states=[0, 1])
    assert r["status"] == "skipped"


def _row(v, name):
    return next((c for c in v["checks"] if c["name"] == name), None)


def test_report_rows_absent_without_aux_keys():
    from gareus_report import build_health_verdict
    v = build_health_verdict({})
    assert _row(v, "Aux ordinary-only crosscheck") is None and _row(v, "Aux workers") is None


def test_report_rows_grade_fail_and_caution():
    from gareus_report import build_health_verdict
    v = build_health_verdict({"aux_crosscheck": {"status": "heuristic_fail", "max_abs_diff_kcal": 1.2},
                              "aux_workers": {"workers": [{"state_id": 7, "best_partner": 3,
                                                           "best_partner_overlap": 0.05,
                                                           "overlap_floor_ok": False}]}})
    assert _row(v, "Aux ordinary-only crosscheck")["status"] == "fail" and v["overall"] == "FAIL"
    w = _row(v, "Aux workers")
    assert w["status"] == "caution" and "no shams" in w["detail"]
    v2 = build_health_verdict({"aux_crosscheck": {"status": "skipped", "reason": "x"},
                               "aux_workers": {"workers": []}})
    assert _row(v2, "Aux ordinary-only crosscheck")["status"] == "na"
    assert _row(v2, "Aux workers")["status"] == "caution"  # empty evidence never grades PASS (F06)


def test_report_row_error_is_caution():
    from gareus_report import build_health_verdict
    v = build_health_verdict({"aux_crosscheck": {"status": "error", "reason": "boom"}})
    assert _row(v, "Aux ordinary-only crosscheck")["status"] == "caution"
