"""X5: allocate top-up MD by effective samples (tau-corrected deficits).

Spec docs/superpowers/specs/2026-09-29-adaptive-cv2-resolution-design.md, Section 12 X5.
"""
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gareus.adaptive import effective_samples as es
from gareus.adaptive.topup_allocator import plan_topup
from gareus.adaptive.union_diagnostics import UnionDiagnostics
import gareus.adaptive_production as ap


def _ar1(phi, n, rng, mean=0.0):
    x = np.empty(n)
    x[0] = rng.normal() / math.sqrt(1.0 - phi * phi)
    eps = rng.normal(size=n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + eps[i]
    return x + mean


def _g_true(phi):
    return (1.0 + phi) / (1.0 - phi)


# ---------------------------------------------------------------- estimator

@pytest.mark.parametrize("phi", [0.5, 0.9, 0.97])
def test_ar1_statistical_inefficiency_matches_the_analytic_value(phi):
    rng = np.random.default_rng(1)
    est = es.pooled_inefficiency([_ar1(phi, 200_000, rng)])
    assert est.status == "ok"
    assert est.g == pytest.approx(_g_true(phi), rel=0.12)


def test_white_noise_has_inefficiency_one():
    rng = np.random.default_rng(2)
    est = es.pooled_inefficiency([rng.normal(size=50_000)])
    assert est.status == "ok" and 1.0 <= est.g < 1.15


def test_segments_with_different_means_are_not_stitched():
    """Each contiguous segment is centred on its own mean: offsets between phases are
    nonstationarity, not correlation, and must not inflate g the way a stitched trace does."""
    rng = np.random.default_rng(3)
    segs = [_ar1(0.8, 4000, rng, mean=m) for m in (0.0, 5.0, -3.0, 8.0)]
    pooled = es.pooled_inefficiency(segs)
    stitched = es.pooled_inefficiency([np.concatenate(segs)])
    assert pooled.g == pytest.approx(_g_true(0.8), rel=0.2)
    assert stitched.g > 3 * pooled.g
    assert pooled.n_segments == 4 and pooled.n_segments_used == 4 and pooled.n_samples == 16000


def test_many_moderate_segments_pool_to_the_right_value():
    rng = np.random.default_rng(4)
    segs = [_ar1(0.9, 1000, rng) for _ in range(100)]          # ~50 g per run
    est = es.pooled_inefficiency(segs)
    assert est.g == pytest.approx(_g_true(0.9), rel=0.15)


def test_runs_of_only_a_few_g_are_biased_low_not_high():
    """Known limitation: centring each run on its own mean removes ~g/n of every rho,
    so runs of ~10 g underestimate g (never overestimate). Real state-indexed runs span
    a whole sample segment (thousands of samples)."""
    rng = np.random.default_rng(4)
    segs = [_ar1(0.9, 200, rng) for _ in range(400)]
    est = es.pooled_inefficiency(segs)
    assert 0.5 * _g_true(0.9) < est.g < _g_true(0.9)


def test_segments_below_the_floor_are_skipped_but_counted():
    rng = np.random.default_rng(5)
    segs = [_ar1(0.5, 5000, rng), rng.normal(size=10), rng.normal(size=3)]
    est = es.pooled_inefficiency(segs)
    assert est.n_segments == 3 and est.n_segments_used == 1 and est.n_samples == 5013
    assert est.status == "ok"


def test_too_short_falls_back_to_one_with_a_reason():
    est = es.pooled_inefficiency([np.arange(10.0), np.arange(5.0)])
    assert est.g == 1.0 and est.status == "too_short" and est.reason


def test_constant_series_falls_back_to_one_with_a_reason():
    est = es.pooled_inefficiency([np.full(500, 2.5)])
    assert est.g == 1.0 and est.status == "zero_variance" and est.reason


def test_no_data_falls_back_to_one():
    est = es.pooled_inefficiency([])
    assert est.g == 1.0 and est.status == "no_samples"


def test_contiguous_runs_split_at_step_gaps_and_nonfinite_values():
    steps = np.array([250, 500, 750, 1000, 2000, 2250, 2500, 2750])
    vals = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0])
    runs = es.contiguous_runs(steps, vals)
    assert [list(r) for r in runs] == [[1.0, 2.0], [4.0], [5.0, 6.0, 7.0, 8.0]]


# ---------------------------------------------------------------- per-state series

def _fake_source(n_steps, windows, phi_by_window, rng, stride=250, replicas_swap=True, dup=0):
    """A sample dict in load_samples' shape: one row per (step, window), replicas permuted."""
    steps, win, rep, cv1, cv2, seg = [], [], [], [], [], []
    series = {w: (_ar1(phi_by_window[w][0], n_steps, rng), _ar1(phi_by_window[w][1], n_steps, rng))
              for w in windows}
    for t in range(n_steps):
        perm = rng.permutation(len(windows)) if replicas_swap else np.arange(len(windows))
        for j, w in enumerate(windows):
            steps.append((t + 1) * stride); win.append(w); rep.append(int(perm[j]))
            cv1.append(series[w][0][t]); cv2.append(series[w][1][t]); seg.append("seg_001")
    d = {"step": np.array(steps, dtype=np.uint64), "window_id": np.array(win, dtype=np.uint16),
         "replica": np.array(rep, dtype=np.uint16), "cv1": np.array(cv1), "cv2": np.array(cv2),
         "segment_id": np.array(seg, dtype=object)}
    if dup:
        for k in list(d):
            d[k] = np.concatenate([d[k], d[k][:dup]])
    return d


def test_state_series_follow_the_state_not_the_replica_and_use_the_window_map():
    rng = np.random.default_rng(6)
    src = _fake_source(20_000, [0, 1], {0: (0.95, 0.5), 1: (0.5, 0.5)}, rng)
    out = es.state_effective_samples(
        [("final", Path("/x/final/baseline"))],
        window_map_for=lambda d: {0: 17, 1: 42},
        restrained_axes={17: (True, True), 42: (True, True)},
        load=lambda d: src)
    assert set(out) == {17, 42}
    assert out[17].g == pytest.approx(_g_true(0.95), rel=0.2)       # replica swaps do not break the series
    assert out[42].g == pytest.approx(_g_true(0.5), rel=0.2)
    assert out[17].n_samples == 20_000
    assert out[17].n_eff == pytest.approx(out[17].n_samples / out[17].g)


def test_g_is_the_slowest_restrained_axis_and_an_unrestrained_axis_is_ignored():
    rng = np.random.default_rng(7)
    src = _fake_source(20_000, [0, 1], {0: (0.5, 0.95), 1: (0.5, 0.95)}, rng)
    out = es.state_effective_samples(
        [("final", Path("/x"))], window_map_for=lambda d: {0: 0, 1: 1},
        restrained_axes={0: (True, True), 1: (True, False)}, load=lambda d: src)
    assert out[0].g == pytest.approx(_g_true(0.95), rel=0.2) and out[0].g == out[0].g_cv2
    assert out[1].g == pytest.approx(_g_true(0.5), rel=0.2) and out[1].g_cv2 is None


def test_duplicate_rows_are_dropped_before_estimating():
    rng = np.random.default_rng(8)
    src = _fake_source(5_000, [0], {0: (0.5, 0.5)}, rng, dup=2_000)
    out = es.state_effective_samples([("final", Path("/x"))], window_map_for=lambda d: {0: 0},
                                     restrained_axes={0: (True, True)}, load=lambda d: src)
    assert out[0].n_samples == 5_000 and out[0].g == pytest.approx(3.0, rel=0.3)


def test_a_state_across_two_phases_pools_without_stitching():
    rng = np.random.default_rng(9)
    a = _fake_source(8_000, [0], {0: (0.8, 0.8)}, rng)
    b = _fake_source(8_000, [3], {3: (0.8, 0.8)}, rng)
    b["cv1"] = b["cv1"] + 10.0
    srcs = {"/a": a, "/b": b}
    out = es.state_effective_samples([("e0", Path("/a")), ("final", Path("/b"))],
                                     window_map_for=lambda d: {0: 5} if str(d) == "/a" else {3: 5},
                                     restrained_axes={5: (True, False)}, load=lambda d: srcs[str(d)])
    assert out[5].n_samples == 16_000 and out[5].n_segments == 2
    assert out[5].g == pytest.approx(_g_true(0.8), rel=0.2)


def test_a_loader_failure_never_raises_and_falls_back_to_one():
    def boom(d):
        raise OSError("unreadable parquet")
    out = es.state_effective_samples([("final", Path("/x"))], window_map_for=lambda d: {0: 0},
                                     restrained_axes={0: (True, True)}, load=boom)
    assert out == {}
    est = es.effective_g_for_states([0, 1], out)
    assert est[0].g == 1.0 and est[0].status == "no_samples" and est[0].reason


# ---------------------------------------------------------------- allocator

POL = ap.AdaptiveDecisionPolicy(topups_enabled=True)          # target sigma 0.10
IV = 500


def _diag(sigma, g_builder=2.0, n=100):
    ids = tuple(sorted(sigma))
    return UnionDiagnostics(state_ids=ids, n_k={s: n for s in ids}, sigma_kcal=dict(sigma),
                            unconverged=frozenset(), inefficiency={s: g_builder for s in ids},
                            edge_overlap={}, f_kT={})


def _chain(n):
    nb = {s: [x for x in (s - 1, s + 1) if 0 <= x < n] for s in range(n)}
    return nb, {s: [] for s in range(n)}


def _plan(diag, effective_g=None, n=6):
    nb, rp = _chain(n)
    return plan_topup(diag, state_ids_in_order=list(diag.state_ids), neighbours=nb, rung_partners=rp,
                      policy=POL, report_interval=IV, timestep_fs=4.0, n_gpus=4, budget_hours=1e6,
                      effective_g=effective_g)


def test_raw_mode_is_unchanged_by_the_new_argument():
    sig = {s: 0.05 for s in range(6)}; sig[2] = 0.15
    d = _diag(sig)
    assert _plan(d) == _plan(d, effective_g=None)
    # the builder's own g is a no-op correction
    assert _plan(d, effective_g={s: 2.0 for s in range(6)}) == _plan(d)


def test_a_slow_state_the_builder_undercounted_becomes_a_deficit():
    sig = {s: 0.08 for s in range(6)}
    assert _plan(_diag(sig)).reason == "healthy"
    g = {s: 2.0 for s in range(6)}; g[4] = 8.0                 # 4x slower -> sigma x2 = 0.16
    p = _plan(_diag(sig), effective_g=g)
    assert p.reason == "planned" and p.deficit_state_ids == (4,)


def test_a_lower_effective_g_never_makes_a_state_look_better_sampled():
    """One-sided correction: the contiguity-aware g is a lower bound for slow states,
    so g_x5 < g_builder leaves sigma (and the deficit set) as the builder has it."""
    sig = {s: 0.05 for s in range(6)}; sig[1] = 0.12; sig[4] = 0.12
    raw = _plan(_diag(sig))
    assert raw.deficit_state_ids == (1, 4)
    g = {s: 2.0 for s in range(6)}; g[1] = 1.0
    assert _plan(_diag(sig), effective_g=g) == raw
    from gareus.adaptive.topup_allocator import decision_sigma
    assert decision_sigma(0.12, 2.0, 1.0) == 0.12 and decision_sigma(0.12, 2.0, 8.0) == 0.24


def test_steps_move_to_a_slow_state_away_from_the_rest():
    sig = {s: 0.05 for s in range(6)}; sig[1] = 0.12
    raw = _plan(_diag(sig))
    g = {s: 2.0 for s in range(6)}; g[4] = 10.0                # 0.05 -> 0.112: the second deficit
    p = _plan(_diag(sig), effective_g=g)
    assert set(p.deficit_state_ids) == {1, 4} and es.allocation_shift(raw, p) > 0.0


def test_required_length_scales_with_the_corrected_deficit_and_keeps_the_cap():
    sig = {s: 0.05 for s in range(6)}; sig[2] = 0.15
    raw = _plan(_diag(sig))
    g = {s: 2.0 for s in range(6)}; g[2] = 4.0                 # sigma_eff = 0.15*sqrt(2)
    p = _plan(_diag(sig), effective_g=g)
    assert p.steps > raw.steps
    # the MAX_STEP_MULTIPLE cap is on the steps-worth of data held, which g does not change
    from gareus.adaptive.topup_allocator import MAX_STEP_MULTIPLE
    g[2] = 1000.0
    capped = _plan(_diag(sig), effective_g=g)
    assert capped.steps == MAX_STEP_MULTIPLE * 100 * IV * 2


def test_predictions_stay_on_the_builder_scale_for_calibration():
    sig = {s: 0.05 for s in range(6)}; sig[2] = 0.15
    g = {s: 2.0 for s in range(6)}; g[2] = 4.0
    p = _plan(_diag(sig), effective_g=g)
    assert p.sigma_before[2] == pytest.approx(0.15)
    scale = p.sample_scale_steps[2]
    assert scale == pytest.approx(100 * IV * 2.0)
    assert p.predicted_sigma[2] == pytest.approx(0.15 * math.sqrt(scale / (scale + p.steps)))


def test_a_nonfinite_effective_g_is_ignored_for_that_state():
    sig = {s: 0.05 for s in range(6)}; sig[2] = 0.15
    g = {s: 2.0 for s in range(6)}; g[2] = float("nan")
    assert _plan(_diag(sig), effective_g=g) == _plan(_diag(sig))


def test_allocation_shift_is_half_the_l1_distance_of_step_shares():
    from gareus.adaptive.topup_allocator import TopupPlan
    a = TopupPlan(state_ids=(0, 1), steps=100)
    b = TopupPlan(state_ids=(1, 2), steps=300)
    assert es.allocation_shift(a, b) == pytest.approx(0.5)
    assert es.allocation_shift(a, a) == 0.0
    assert es.allocation_shift(TopupPlan(), TopupPlan()) == 0.0


# ---------------------------------------------------------------- policy, flag, P8

def test_the_policy_defaults_to_raw_and_the_field_is_frozen():
    assert ap.AdaptiveDecisionPolicy().allocation_weight == "raw"
    assert "allocation_weight" in ap.DECISION_SETTINGS_FIELDS


def test_the_flag_reaches_the_policy():
    from gareus.cli import build_gareus_parser, parse_args
    assert build_gareus_parser().parse_args(["--seq", "AA"]).ap_allocation_weight == "raw"
    args = parse_args(["--seq", "AA", "--ap-allocation-weight", "ess"])
    assert ap.policy_from_args(args).allocation_weight == "ess"
    with pytest.raises(SystemExit):
        parse_args(["--seq", "AA", "--ap-allocation-weight", "neff"])


def test_a_campaign_recorded_before_the_field_existed_resumes(tmp_path):
    fields = [f for f in ap.DECISION_SETTINGS_FIELDS if f != "allocation_weight"]
    base = ap.AdaptiveDecisionPolicy()
    (tmp_path / ap.DECISION_SETTINGS_FILENAME).write_text(json.dumps(
        {"schema_version": "adaptive_decision_settings_v1",
         "settings": {f: getattr(base, f) for f in fields}, "created_unix": 1.0}))
    pol, record = ap._resolve_decision_settings(tmp_path, base)
    assert pol.allocation_weight == "raw" and record["settings"]["allocation_weight"] == "raw"


# ---------------------------------------------------------------- wiring

def _registry(tmp_path, n=4):
    reg = ap.WindowStateRegistry()
    for i in range(n):
        reg.add_state(primary_center=0.1 * i, primary_k=500.0, secondary_center=0.0, secondary_k=0.0)
    adaptive_dir = tmp_path / "adaptive_production"
    adaptive_dir.mkdir()
    return reg, adaptive_dir


def _args():
    return SimpleNamespace(report_interval=500, device_index="0", timestep_fs=4.0, temperature_k=300.0)


def _wire(monkeypatch, sig, g_x5):
    diag = _diag(sig)
    monkeypatch.setattr(ap, "_phase_union_diagnostics", lambda *a, **k: diag)

    def fake(adaptive_dir, registry, state_ids, **kw):
        return {s: es.StateEffectiveSamples(state_id=s, n_samples=1000, g=g_x5[s], g_cv1=g_x5[s], g_cv2=None,
                                            n_eff=1000 / g_x5[s], n_segments=1, status="ok", reason="")
                for s in state_ids}
    monkeypatch.setattr(es, "campaign_effective_samples", fake)


def test_ess_mode_changes_the_plan_and_writes_the_audit(tmp_path, monkeypatch):
    reg, adaptive_dir = _registry(tmp_path)
    epoch_dir = adaptive_dir / "final"; epoch_dir.mkdir()
    sig = {s: 0.08 for s in range(4)}
    _wire(monkeypatch, sig, {0: 2.0, 1: 2.0, 2: 8.0, 3: 2.0})
    pol = ap.AdaptiveDecisionPolicy(topups_enabled=True, allocation_weight="ess")
    plan = ap._topup_plan_for_phase(_args(), epoch_dir, reg, pol, full_steps=10_000_000)
    assert plan.deficit_state_ids == (2,)
    audit = json.loads((epoch_dir / es.REPORT_NAME).read_text())
    assert audit["allocation_weight"] == "ess"
    row = audit["states"]["2"]
    assert row["g"] == 8.0 and row["g_builder"] == 2.0
    assert row["sigma_decision_kcal"] == pytest.approx(0.16)
    assert audit["raw_plan"]["reason"] == "healthy" and audit["allocation_shift"] == pytest.approx(1.0)


def test_raw_mode_writes_no_audit_and_never_estimates(tmp_path, monkeypatch):
    reg, adaptive_dir = _registry(tmp_path)
    epoch_dir = adaptive_dir / "final"; epoch_dir.mkdir()
    _wire(monkeypatch, {s: 0.08 for s in range(4)}, {0: 2.0, 1: 2.0, 2: 8.0, 3: 2.0})
    monkeypatch.setattr(es, "campaign_effective_samples", lambda *a, **k: pytest.fail("raw must not estimate"))
    plan = ap._topup_plan_for_phase(_args(), epoch_dir, reg, ap.AdaptiveDecisionPolicy(topups_enabled=True),
                                    full_steps=10_000_000)
    assert plan.reason == "healthy" and not (epoch_dir / es.REPORT_NAME).exists()


def test_an_estimation_failure_plans_as_raw_and_records_why(tmp_path, monkeypatch):
    reg, adaptive_dir = _registry(tmp_path)
    epoch_dir = adaptive_dir / "final"; epoch_dir.mkdir()
    sig = {s: 0.05 for s in range(4)}; sig[1] = 0.15
    _wire(monkeypatch, sig, {s: 2.0 for s in range(4)})
    monkeypatch.setattr(es, "campaign_effective_samples",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("duckdb gone")))
    pol = ap.AdaptiveDecisionPolicy(topups_enabled=True, allocation_weight="ess")
    plan = ap._topup_plan_for_phase(_args(), epoch_dir, reg, pol, full_steps=10_000_000)
    assert plan.deficit_state_ids == (1,)
    audit = json.loads((epoch_dir / es.REPORT_NAME).read_text())
    assert audit["status"] == "failed" and "duckdb gone" in audit["reason"]
