"""Report v3 follow-ups (h): R3 flag-only mode, the replica-path crossing count, R2 same-column
contributor counting and the autocorrelation-length bootstrap blocks."""
import dataclasses
import json
import math
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import lfilter

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_cv2_resolution as base  # noqa: E402

import gareus.adaptive_production as ap  # noqa: E402
from gareus.adaptive import cv2_coverage as cov  # noqa: E402
from gareus.adaptive import cv2_resolution as cr  # noqa: E402
from gareus.adaptive import cv2_resolution_grade as grade  # noqa: E402
from gareus.adaptive import cv2_resolution_io as cio  # noqa: E402
from gareus.adaptive import cv2_resolution_rules as rules  # noqa: E402
from gareus.adaptive import cv2_resolution_summary as summ  # noqa: E402
from gareus.cli import parse_args  # noqa: E402

T = base.T
A, B = -1.0, 1.0
BOUNDS = (-0.5, 0.5)


# ---- defaults, validation, wiring ----------------------------------------------------------

def test_v3_defaults_and_mirrored_choice_tuples():
    s = cr.ResolutionSettings()
    assert (s.refine_transition_count, s.refine_r3_mode, s.coverage_count) == ("replica-path", "flag", "same-column")
    assert s.refine_pmf_sigma_kT == 0.25 and s.coverage_min_windows == 2.0
    assert ap.AdaptiveDecisionPolicy().refine_pmf_sigma_kT == 0.25
    assert ap.policy_from_args(Namespace()).refine_pmf_sigma_kT == 0.25
    p = ap.AdaptiveDecisionPolicy()
    assert (p.refine_transition_count, p.refine_r3_mode, p.coverage_count) == ("replica-path", "flag", "same-column")
    assert ap.REFINE_TRANSITION_COUNTS == cr.TRANSITION_COUNTS
    assert ap.REFINE_R3_MODES == cr.R3_MODES and ap.COVERAGE_COUNTS == cr.COVERAGE_COUNTS
    for f in ("refine_r3_mode", "coverage_count", "refine_transition_count"):
        assert f in ap.DECISION_SETTINGS_FIELDS and f in cr.DEFAULTS
    assert cr.SCHEMA_VERSION == "cv2_resolution_report_v3"


@pytest.mark.parametrize("field", ["refine_transition_count", "refine_r3_mode", "coverage_count"])
def test_bad_choice_fails_at_policy_and_settings_construction(field):
    with pytest.raises(ValueError, match=field):
        ap.AdaptiveDecisionPolicy(**{field: "bogus"})
    with pytest.raises(ValueError, match=field):
        cr.ResolutionSettings(**{field: "bogus"})
    with pytest.raises(ValueError, match=field):
        ap.policy_from_args(Namespace(**{f"adaptive_production_{field}": "bogus"}))


def test_cli_and_yaml_reach_the_policy(tmp_path):
    args = parse_args(["--seq", "GYDPETGTWG", "--out", str(tmp_path / "r"), "--ap-refine-r3-mode", "insert",
                       "--ap-coverage-count", "any", "--ap-refine-transition-count", "replica"])
    p = ap.policy_from_args(args)
    assert (p.refine_r3_mode, p.coverage_count, p.refine_transition_count) == ("insert", "any", "replica")
    cfg = tmp_path / "c.yaml"
    cfg.write_text("adaptive_production:\n  ap_refine_r3_mode: insert\n  ap_coverage_count: any\n"
                   "  ap_refine_transition_count: state-series\n")
    p = ap.policy_from_args(parse_args(["--config", str(cfg), "--seq", "GYDPETGTWG", "--out", str(tmp_path / "r2")]))
    assert (p.refine_r3_mode, p.coverage_count, p.refine_transition_count) == ("insert", "any", "state-series")
    st = cr.ResolutionSettings.from_policy(p)
    assert (st.refine_r3_mode, st.coverage_count) == ("insert", "any")


def test_help_topic_documents_the_new_flags():
    from gareus.helptext import _METHOD_ENCYCLOPEDIA
    for key in ("--ap-refine-r3-mode", "--ap-coverage-count", "replica-path", "r3_flag_only"):
        assert key in _METHOD_ENCYCLOPEDIA


# ---- R3 flag vs insert ---------------------------------------------------------------------

def _r3(mode, count="replica-path"):
    reg = base._registry()
    parent = base._sid(reg, 0.2, 0.0)
    z = base._bimodal()
    sub = {parent: {"cv2": z, "cv1": np.full(z.size, 0.2), "source_index": np.zeros(z.size, int)}}
    alt = base._alternating_runs()
    runs = {parent: {"source": "parquet", "replica_runs": alt, "replica_paths": alt, "state_runs": alt}}
    settings = dataclasses.replace(base.SETTINGS, refine_r3_mode=mode, refine_transition_count=count)
    new, report, _h = base._propose(reg, base._payload(reg, samples={parent: z}), subsamples=sub, runs=runs,
                                    settings=settings)
    (cand,) = [c for c in report["candidates"] if c["rule"] == "R3" and c["state_ids"] == [parent]]
    return reg, parent, new, report, cand


def test_flag_mode_records_the_would_be_children_but_never_inserts(tmp_path):
    reg, parent, new, report, cand = _r3("flag")
    assert cand["decision"] == "flagged" and cand["reason"].startswith(cr.R3_FLAG_ONLY)
    assert cand["refusal"] is None and cand["cost_states"] == 0
    assert cand["metrics"]["r3_gate"] == "passed" and cand["metrics"]["trapped_or_orthogonal"] is False
    assert cand["metrics"]["would_be"] == {"decision": "proposed", "refusal": None, "r3_mode": "flag"}
    kids = cand["proposal"]["children"]
    assert len(kids) == 2 and all(k["k2"] > 0 and k["refusal"] is None for k in kids)
    assert not [a for a in new if a[0] == "insert"] and new == []
    assert report["budget"]["spent_states"] == 0
    assert report["summary"]["n_blocking"] == 0 and report["summary"]["n_proposed"] == 0
    assert report["summary"]["n_r3_flag_only"] == 1 and cr.is_flag_only(cand)
    (tmp_path / cr.REPORT_NAME).write_text(json.dumps(report))
    assert cio.is_blocking(tmp_path) == 0                      # convergence gate not blocked


def test_insert_mode_is_the_pre_v3_behaviour_with_the_same_children():
    _reg, parent, new, report, cand = _r3("insert")
    _r, _p, _n, _rep, flagged = _r3("flag")
    assert cand["decision"] == "proposed" and cand["metrics"]["would_be"]["decision"] == "proposed"
    (ins,) = [a for a in new if a[0] == "insert"]
    assert ins[1] == parent and report["summary"]["n_blocking"] == 1 and report["summary"]["n_r3_flag_only"] == 0
    assert [k["k2"] for k in cand["proposal"]["children"]] == [k["k2"] for k in flagged["proposal"]["children"]]


def test_flag_mode_keeps_a_spring_cap_refusal_refused():
    view = cr.StateView(0, 0.2, 800.0, 0.0, 5.0, 0.0)
    comp = {"mean": -0.5, "sd": 0.02, "variance": 0.0004, "n_members": 30, "weight": 0.5}
    modes = {"pair": [comp, {**comp, "mean": 0.5}], "fit": {"pooled_variance": 0.3},
             "depth": {"depth_kT": 3.0, "barrier_z": 0.0}, "gate_values": {}}
    rec = {"source": "parquet", "replica_runs": base._alternating_runs(sep=0.5),
           "replica_paths": base._alternating_runs(sep=0.5), "state_runs": []}
    st = cr.ResolutionSettings(temperature_k=T)
    cand = rules._r3_decide(view, modes, rec, st, None)
    assert cand["decision"] == "refused" and cand["refusal"] == "k2_capped_below_compression"
    assert cand["metrics"]["would_be"]["decision"] == "refused" and not cr.is_flag_only(cand)


def test_summary_and_grade_count_flag_only_windows():
    _reg, parent, _new, report, _cand = _r3("flag")
    row = {"state_id": parent, "sample_count": 5000,
           "paired_cv": {"n_pairs": 2000, "cv2": {"mean": 0.0, "var": 0.4}}}
    s = summ.build_summary({"states": [row], "edges": []}, report, label="epoch_001", temperature_k=T)
    assert s["schema_version"] == "cv2_resolution_summary_v2"
    assert s["counts"]["n_r3_flag_only"] == 1
    (st,) = s["states"]
    assert st["r3_flag_only"] is True and st["r3_would_be"]["decision"] == "proposed"
    assert st["transitions_estimator"] == "replica-path" and st["transitions_replica_path"] == 40
    out = grade.check_cv2_resolution({"cv2_resolution": s})
    assert out["status"] == grade.CAUTION and "flagged only" in out["detail"]


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_summary_reads_v1_and_v2_reports(version):
    key = "transitions_lower_bound" if version == "v1" else "transitions_state_series_lower_bound"
    cand = {"rule": "R3", "state_ids": [3], "decision": "flagged", "reason": "trapped_or_orthogonal: x",
            "metrics": {"transitions": 4, "transitions_state_series": 90, "trapped_or_orthogonal": True}}
    cand2 = {"rule": "R3", "state_ids": [4], "decision": "flagged", "reason": "transitions_unavailable",
             "metrics": {"transitions": None, key: 7}}
    rep = {"schema_version": f"cv2_resolution_report_{version}", "status": "ok", "candidates": [cand, cand2],
           "summary": {"n_proposed": 0, "n_blocking": 0}}
    rows = [{"state_id": i, "paired_cv": {"n_pairs": 10, "cv2": {"mean": 0.0, "var": 0.1}}} for i in (3, 4)]
    s = summ.build_summary({"states": rows, "edges": []}, rep, label="x", temperature_k=T)
    by = {r["state_id"]: r for r in s["states"]}
    assert (by[3]["transitions"], by[3]["transitions_estimator"]) == (4, "replica")
    assert (by[4]["transitions"], by[4]["transitions_estimator"]) == (7, "state_series_lower_bound")
    assert by[3]["transitions_replica_path"] is None and by[3]["r3_flag_only"] is False
    assert s["counts"]["n_r3_flag_only"] == 0 and s["counts"]["n_trapped_or_orthogonal"] == 1


# ---- replica-path count --------------------------------------------------------------------

def _data(rows):
    """rows: (step, segment, replica, cv2); every row at window 0 (-> state 7)."""
    step, seg, rep, z = (np.asarray(x) for x in zip(*rows))
    return {"step": step.astype(np.int64), "segment_id": seg.astype(str), "replica": rep.astype(np.int64),
            "cv2": z.astype(float), "window_id": np.zeros(step.size, np.int64)}


def _counts(rows):
    rec = {**cio._runs_one_source(_data(rows), {0: 7}, {7})[7], "source": "parquet"}
    return {c: rules._transitions(rec, BOUNDS, c)["transitions"] for c in cr.TRANSITION_COUNTS}


def test_replica_path_on_a_hand_built_series():
    rows = [(0, "a", 1, A), (1, "a", 1, A), (2, "a", 2, B), (3, "a", 2, B),       # r1 A A | r2 B B
            (4, "a", 1, B), (5, "a", 1, np.nan), (6, "a", 1, B),                   # r1 back in B (NaN dropped)
            (7, "a", 2, A), (8, "a", 2, 0.0), (9, "a", 2, A),                      # r2 back in A (0 = between cores)
            (100, "b", 1, A), (101, "b", 1, A)]                                     # new segment: never joined
    got = _counts(rows)
    # r1 path in a: A A B B -> 1; r2: B B A A -> 1; segment b adds nothing (not joined to a's B)
    assert got["replica-path"] == 2
    # residences: no change inside any contiguous residence
    assert got["replica"] == 0
    # state series in a: A A B B B | NaN breaks | B A 0 A -> 1 + 1 = 2; segment b 0
    assert got["state-series"] == 2


def test_replica_path_matches_the_t2_harness_definition():
    rng = np.random.default_rng(5)
    n, R = 400, 3
    rep = np.zeros(n, int)
    z = np.zeros(n)
    pos = rng.choice([A, B], R)
    for t in range(n):
        rep[t] = rng.integers(R)
        if rng.uniform() < 0.1:
            pos[rep[t]] = -pos[rep[t]]
        z[t] = pos[rep[t]] + rng.normal(0, 0.1)
    got = _counts([(t, "s", rep[t], z[t]) for t in range(n)])["replica-path"]
    want = 0
    for r in range(R):
        lab = np.where(z[rep == r] <= BOUNDS[0], 0, np.where(z[rep == r] >= BOUNDS[1], 1, -1))
        lab = lab[lab >= 0]
        want += int(np.count_nonzero(np.diff(lab)))
    assert got == want > 0


@pytest.mark.parametrize("seed", range(20))
def test_ordering_replica_is_below_both_others(seed):
    rng = np.random.default_rng(seed)
    n, R = 300, int(rng.integers(1, 5))
    pos = rng.choice([A, B], R)
    rows, cur = [], int(rng.integers(R))
    for t in range(n):
        if rng.uniform() < 0.3:
            cur = int(rng.integers(R))
        for r in range(R):
            if rng.uniform() < 0.05:
                pos[r] = -pos[r]
        seg = "a" if t < n // 2 else "b"
        step = t + (7 if rng.uniform() < 0.02 else 0)
        rows.append((step, seg, cur, pos[cur] + rng.normal(0, 0.4)))
    got = _counts(sorted(rows))
    assert got["replica"] <= got["replica-path"]
    assert got["replica"] <= got["state-series"]


def test_replica_path_and_state_series_are_not_ordered():
    # two walkers, each fixed in one mode, alternating residences: many state-series switches, no crossing
    swaps = [(t, "a", t // 2 % 2, A if t // 2 % 2 == 0 else B) for t in range(40)]
    got = _counts(swaps)
    assert got["state-series"] > 0 == got["replica-path"]
    # state series A A B B whose two replicas each go A -> B: state-series 1 < replica-path 2
    got = _counts([(0, "a", 1, A), (1, "a", 2, A), (2, "a", 1, B), (3, "a", 2, B)])
    assert got["state-series"] == 1 < got["replica-path"] == 2


def test_replica_path_without_paths_or_replica_column_is_unavailable():
    no_paths = {"source": "parquet", "replica_runs": [np.array([A, B])], "state_runs": [np.array([A, B])]}
    out = rules._transitions(no_paths, BOUNDS, "replica-path")
    assert out["transitions"] is None and out["transitions_state_series_lower_bound"] == 1
    out = rules._transitions({"source": "no_replica_column", "state_runs": [np.array([A, B, A])]}, BOUNDS)
    assert out["transitions"] is None and out["transitions_state_series_lower_bound"] == 2


# ---- R2 same-column --------------------------------------------------------------------------

K1, K2 = 800.0, 5.0


def _grid_case():
    """2 x 2 grid (columns 0.2 / 0.4, rows -1 / +1). The CV2 = 0 gap of column 0.2 is filled
    ONLY by rows of the neighbouring column's windows (sitting inside column 0.2's slab) and by
    a CV1-unrestrained window at the placeholder c1 = 0.2."""
    rng = np.random.default_rng(0)
    views = [cr.StateView(i, c1, K1, c2, K2, 0.0) for i, (c1, c2) in enumerate(
        [(0.2, -1.0), (0.2, 1.0), (0.4, -1.0), (0.4, 1.0)])]
    views.append(cr.StateView(4, 0.2, 0.0, 0.0, K2, 0.0))        # CV1-unrestrained, placeholder c1
    cv1, cv2, sidx = [], [], []
    for v in views[:4]:
        n = 600
        cv1.append(np.full(n, v.c1) + rng.normal(0, 0.01, n))
        cv2.append(np.full(n, v.c2) + rng.normal(0, 0.1, n))
        sidx.append(np.full(n, v.state_id))
    for s in (2, 3, 4):                                            # the gap rows
        n = 300
        cv1.append(np.full(n, 0.21) + rng.normal(0, 0.005, n))
        cv2.append(rng.normal(0.0, 0.08, n))
        sidx.append(np.full(n, s))
    cv1, cv2, sidx = (np.concatenate(x) for x in (cv1, cv2, sidx))
    w = np.full(cv1.size, 1.0 / cv1.size)
    cols = {v.state_id: v for v in views[:4]}
    return cv1, cv2, sidx, w, views, cols


def _gap_holes(count):
    cv1, cv2, sidx, w, views, cols = _grid_case()
    st = cr.ResolutionSettings(temperature_k=T, coverage_count=count, coverage_min_windows=2.0,
                               refine_pmf_sigma_kT=float("inf"))
    cands = cov.coverage_holes(cv1, cv2, sidx, w, views, cols, st)
    return [c for c in cands if abs(c["metrics"]["c1"] - 0.2) < 1e-9
            and c["metrics"]["interval"][0] < 0.0 < c["metrics"]["interval"][1]]


def test_same_column_count_finds_the_gap_a_neighbour_column_fills():
    assert _gap_holes("any") == []                      # 2 other-column centres contribute: not a hole
    (hole,) = _gap_holes("same-column")
    m = hole["metrics"]
    assert m["coverage_count"] == "same-column"
    assert m["n_contributing_windows"] == m["n_contributing_windows_same_column"] == 0
    assert m["n_contributing_windows_any"] >= 2


def test_cv1_unrestrained_windows_are_in_no_column():
    _cv1, _cv2, _s, _w, views, _c = _grid_case()
    centre, column = cov.centre_columns(views, cr.ResolutionSettings(temperature_k=T))
    assert column[centre[4]] is None
    assert [column[centre[i]] for i in range(4)] == [0.2, 0.2, 0.4, 0.4]


# ---- bootstrap blocks from the autocorrelation ---------------------------------------------

def _ar1(phi, n, seed):
    rng = np.random.default_rng(seed)
    return lfilter([math.sqrt(1 - phi ** 2)], [1, -phi], rng.normal(size=n + 2000))[2000:]


def _true_sigma_f(phi, n):
    """Exact sd of F = -ln p_hat, p_hat = fraction of x > 0 for a stationary Gaussian AR(1)
    (P(x0 > 0, xt > 0) = 1/4 + arcsin(rho_t) / 2 pi), to first order."""
    t = np.arange(1, n)
    var = 0.25 + 2.0 * np.sum((1.0 - t / n) * np.arcsin(phi ** t) / (2 * np.pi))
    return 2.0 * math.sqrt(var / n)


def test_block_length_tracks_the_autocorrelation_time():
    phi, n = 0.9, 20000
    x = _ar1(phi, n, 0)
    ids, info = cov.autocorrelation_block_ids(np.zeros(n, int), [x, x])
    rec = info["states"]["0"]
    g_true = (1 + phi) / (1 - phi)
    assert 0.8 * g_true < rec["g"] < 1.25 * g_true
    assert rec["block_rows"] == math.ceil(cr.BOOT_BLOCK_G_MULTIPLE * rec["g"])
    assert rec["n_blocks"] == np.unique(ids).size and not rec["min_blocks_bound"]


def test_bootstrap_sigma_is_calibrated_on_ar1():
    """g5 blocks: mean bootstrap sigma within 15 % of the exact sd (measured 0.95 +- 0.01 at
    phi 0.9, n 20,000; a 5 g block reads ~5 % low by construction)."""
    phi, n = 0.9, 20000
    sig = []
    for seed in range(12):
        x = _ar1(phi, n, seed)
        w = np.full(n, 1.0 / n)
        ids, _ = cov.autocorrelation_block_ids(np.zeros(n, int), [x, x])
        sig.append(cov._boot_sigma(np.where(x > 0, w, 0.0)[None, :], w, ids, np.random.default_rng(seed))[0])
    ratio = float(np.mean(sig)) / _true_sigma_f(phi, n)
    assert 0.85 < ratio < 1.1


def test_blocks_never_cross_a_source_and_the_min_block_guard_binds():
    x = _ar1(0.995, 600, 1)
    src = np.repeat([0, 1], 300)
    ids, info = cov.autocorrelation_block_ids(np.zeros(600, int), [x, x], src)
    rec = info["states"]["0"]
    assert rec["n_sources"] == 2 and rec["n_blocks"] >= cr.BOOT_MIN_BLOCKS and rec["min_blocks_bound"]
    assert not set(ids[:300]) & set(ids[300:])
    short = cov.autocorrelation_block_ids(np.zeros(15, int), [np.arange(15.0)])[1]["states"]["0"]
    assert short["g_status"] != "ok" and short["g"] == 1.0          # too short to estimate: g = 1, recorded


def test_union_row_sources_read_from_the_sidecar(tmp_path):
    npz = tmp_path / "topup_union_mbar.npz"
    (tmp_path / "topup_union_mbar.samples.csv").write_text("source,source_dir,step\na,/x/a,1\nb,/x/b,2\na,/x/a,3\n")
    src, note = cio.union_row_sources(npz, 3)
    assert src.tolist() == [0, 1, 0] and note.endswith(".samples.csv")
    assert cio.union_row_sources(npz, 4)[0] is None
    assert cio.union_row_sources(tmp_path / "missing.npz", 3)[0] is None
