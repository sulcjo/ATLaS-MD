import argparse

from gareus.dashboard.context import build_context
from gareus.dashboard.sidecar import SidecarSnapshot
from gareus.dashboard.spine import COMPACT_SPINE_LINES, FULL_SPINE_LINES, bucket_strip, spine_lines
from gareus.logger import DistanceLogger
from gareus.tui import strip_ansi, strip_ansi_len

CENTERS = tuple(4.0 + 0.55 * i for i in range(25))


def _ctx(tmp_path, *, term_w=140, term_h=45, sidecar=None, n=25):
    logger = DistanceLogger(tmp_path, argparse.Namespace(timestep_fs=2.0), no_file_persistence=True)
    for w in range(n):
        logger.history_by_window[w] = [CENTERS[w] + 0.1 * (i % 5 - 2) for i in range(50)]
    rows = [{"replica": w, "window": w, "center_A": CENTERS[w], "k_kcal_mol_A2": 2.5,
             "cv_A": CENTERS[w] + 0.05} for w in range(n)]
    return build_context(
        logger=logger, rows=rows, phase="gareus_production", step=12_450_000,
        total_steps=37_500_000, summary={}, dashboard_info={
            "centers_a": list(CENTERS[:n]), "n_windows": n, "k_list": [2.5] * n,
            "exchange_stats": {f"{i}-{i+1}": {"attempts": 400, "accepted": 120}
                               for i in range(n - 1)},
            "primary_cv_label": "contacts", "primary_cv_units": "A",
            "adaptive_phase": {"epoch_index": 1, "epoch_total": 2,
                               "segment_name": "epoch_001/topup_002"}},
        sidecar=sidecar or SidecarSnapshot(), term_w=term_w, term_h=term_h, now=1000.0,
        view="progress", glyphs="unicode",
    )


def test_bucket_strip_renders_one_cell_per_value_when_it_fits():
    assert strip_ansi_len(bucket_strip([1.0, 2.0, 3.0], ["ok"] * 3, cells=8)) == 3


def test_bucket_strip_aggregates_when_there_are_more_values_than_cells():
    assert strip_ansi_len(bucket_strip([1.0] * 364, ["ok"] * 364, cells=100)) == 100


def test_bucket_strip_keeps_a_bad_window_visible_after_bucketing():
    statuses = ["ok"] * 364
    statuses[200] = "BAD"
    for mode in ("ascii", "unicode"):
        strip = strip_ansi(bucket_strip([1.0] * 364, statuses, cells=100, glyphs=mode))
        assert "X" in strip, f"{mode}: a bad bucket must survive ANSI stripping"


def test_bucket_strip_never_renders_a_present_value_as_blank():
    """The lowest bucket is the starved window -- the whole point of the strip."""
    varied = strip_ansi(bucket_strip([1.0, 3.0, 9.0, 4.0], ["ok"] * 4, cells=4))
    assert " " not in varied
    uniform = strip_ansi(bucket_strip([60.0] * 8, ["ok"] * 8, cells=8))
    assert " " not in uniform
    assert len(set(uniform)) == 1           # equal sampling reads as an even strip


def test_bucket_strip_marks_a_dead_pair_rather_than_leaving_a_gap():
    """A dead pair sits at the bottom of the acceptance range, so a ramp starting
    at a space would render the single most important cell as nothing."""
    rates = [0.30, 0.31, 0.0, 0.29]
    statuses = ["ok", "ok", "BAD", "ok"]
    strip = strip_ansi(bucket_strip(rates, statuses, cells=4))
    assert strip[2] == "X"
    assert " " not in strip


def test_bucket_strip_handles_an_empty_input():
    assert bucket_strip([], [], cells=20) == ""


def test_bucket_strip_survives_every_value_being_nan():
    """Reproduces a real crash found wiring the screen into DistanceLogger: the
    exchange strip on a brand-new run (zero exchange attempts yet) is built from
    `acceptance_by_pair`, which returns NaN -- never 0.0 -- for every untried
    pair (see its docstring: 'not tried yet' must not rank the same as 'tried
    and always rejected'). `hi`/`lo` default to 1.0/0.0 when nothing is finite,
    which made `span` look positive even though there was nothing real to place
    on it, and `int(round(nan))` from averaging an all-NaN bucket raised
    ValueError outright -- i.e. the live dashboard would have crashed on its
    very first frame, before a single exchange had been attempted."""
    nan = float("nan")
    strip = bucket_strip([nan, nan, nan, nan], ["ok"] * 4, cells=4)
    assert strip_ansi_len(strip) == 4


def test_bucket_strip_survives_a_partially_nan_series():
    """Same bug, narrower trigger: only some buckets are entirely NaN while the
    strip as a whole has real finite values elsewhere (a run partway through
    filling in its exchange pairs one at a time)."""
    nan = float("nan")
    strip = bucket_strip([0.30, nan, nan, 0.29], ["ok"] * 4, cells=4)
    assert strip_ansi_len(strip) == 4


def test_spine_lines_returns_the_real_1d_line_count_with_no_blank_padding(tmp_path):
    """Nine line-kinds exist in the full tier; a 1D run (no CV2 line) has
    eight, all real content. `spine_lines` no longer pads the return value up
    to `FULL_SPINE_LINES` -- that constant is only the budget `render_screen`
    reserves for the spine, not a line count this function fills with blanks
    (the design mockup's tenth row is `render_screen`'s own footer divider).
    """
    lines = spine_lines(_ctx(tmp_path), FULL_SPINE_LINES)
    assert len(lines) == 8
    assert all(strip_ansi_len(l) <= 140 - 2 for l in lines)
    assert all(strip_ansi(l).strip() != "" for l in lines)


def test_spine_lines_names_the_phase_epoch_and_window_count(tmp_path):
    text = strip_ansi("\n".join(spine_lines(_ctx(tmp_path), FULL_SPINE_LINES)))
    assert "gareus_production" in text
    assert "1/2" in text
    assert "25 win" in text


def test_spine_shows_the_pool_budget_when_the_sidecar_has_it(tmp_path):
    sidecar = SidecarSnapshot(pool={"total_ns": 15000.0, "used_ns": 7500.0,
                                    "remaining_ns": 7500.0, "events": []})
    text = strip_ansi("\n".join(spine_lines(_ctx(tmp_path, sidecar=sidecar), FULL_SPINE_LINES)))
    assert "7500" in text and "15000" in text


def test_pool_line_never_prints_a_literal_nan_perf_value(tmp_path):
    """I1: the shared `_ctx` fixture passes no `eta_start_wall`, so elapsed_s is
    measured against the real wall clock and comes out 0 for a `now` this far in
    the past -- `_ns_per_day` is nan whenever that happens (real trigger: a
    logger whose first frame renders before any wall time has elapsed). A bare
    `{ns_day:.0f}` used to print the literal token "nan" in the pool line's own
    perf field, in a run that otherwise has real, populated pool data.
    """
    sidecar = SidecarSnapshot(pool={"total_ns": 15000.0, "used_ns": 7500.0,
                                    "remaining_ns": 7500.0, "events": []})
    lines = [strip_ansi(l) for l in spine_lines(_ctx(tmp_path, sidecar=sidecar), FULL_SPINE_LINES)]
    pool_line = next(l for l in lines if l.startswith("pool"))
    assert "nan" not in pool_line
    assert "perf —" in pool_line


def test_spine_says_the_pool_is_unavailable_rather_than_faking_a_bar(tmp_path):
    text = strip_ansi("\n".join(spine_lines(_ctx(tmp_path), FULL_SPINE_LINES)))
    assert "pool" in text and "unavailable" in text


def test_spine_reports_gamd_saturation_as_text_not_only_colour(tmp_path):
    sidecar = SidecarSnapshot(gamd={"joint_envelope": {"Dihedral": {
        "sigma0_kj_mol": 12.552, "sigmaV_kj_mol": 11.039, "k0": 1.0}}})
    text = strip_ansi("\n".join(spine_lines(_ctx(tmp_path, sidecar=sidecar), FULL_SPINE_LINES)))
    assert "SATURATED" in text


def test_gamd_line_reports_missing_k0_as_unreported_not_ok(tmp_path):
    """I1: an absent/unmeasured k0 must not fall into the `>= _K0_SATURATED`
    comparison -- `nan >= 0.999` is False in Python, so the old bare threshold
    chain silently classified "not reported" as healthy ("k0 nan ok" in green),
    the same bug class `boost_envelope_panel` already guards against.
    """
    sidecar = SidecarSnapshot(gamd={"joint_envelope": {"Dihedral": {}}})
    text = strip_ansi("\n".join(spine_lines(_ctx(tmp_path, sidecar=sidecar), FULL_SPINE_LINES)))
    gamd_line = next(l for l in text.splitlines() if l.startswith("gamd"))
    assert "nan" not in gamd_line
    assert "k0 — not reported" in gamd_line
    assert "SATURATED" not in gamd_line


def test_compact_spine_returns_five_lines_and_keeps_verdict_and_progress(tmp_path):
    lines = spine_lines(_ctx(tmp_path, term_w=100, term_h=24), COMPACT_SPINE_LINES)
    assert len(lines) == COMPACT_SPINE_LINES
    text = strip_ansi("\n".join(lines))
    assert "gareus_production" in text
    assert "%" in text                       # progress survives the compact tier


def test_spine_omits_the_cv2_line_for_a_1d_run(tmp_path):
    lines = spine_lines(_ctx(tmp_path), FULL_SPINE_LINES)
    assert not any(strip_ansi(l).startswith("cv2") for l in lines)


def test_spine_lines_returns_the_real_2d_line_count_with_no_blank_padding(tmp_path):
    """A 2D run has all nine line-kinds (the eight 1D ones plus CV2), all real
    content -- the mirror case of the 1D count test above, pinning the other
    number a later task's frame geometry needs (Task 11/14 both assert
    against these exact counts).
    """
    n = 10
    centers = [4.0 + 0.5 * i for i in range(n)]
    sec_centers = [float(i - 4) for i in range(n)]   # varying -> ctx.is_2d True
    logger = DistanceLogger(tmp_path, argparse.Namespace(timestep_fs=2.0), no_file_persistence=True)
    for w in range(n):
        logger.history_by_window[w] = [centers[w] + 0.02 * (i % 5 - 2) for i in range(30)]
        logger.secondary_history_by_window[w] = [sec_centers[w] + 0.02 * (i % 5 - 2) for i in range(30)]
    rows = [{"replica": w, "window": w, "center_A": centers[w], "k_kcal_mol_A2": 2.5,
             "cv_A": centers[w] + 0.01} for w in range(n)]
    exchange_stats = {f"{i}-{i+1}": {"attempts": 400, "accepted": 120} for i in range(n - 1)}
    ctx = build_context(
        logger=logger, rows=rows, phase="gareus_production", step=1000, total_steps=100000,
        summary={}, dashboard_info={
            "centers_a": centers, "n_windows": n, "k_list": [2.5] * n,
            "secondary_cv_centers": sec_centers, "secondary_cv": {"type": "torsion-pca"},
            "exchange_stats": exchange_stats,
            "primary_cv_label": "contacts", "primary_cv_units": "A"},
        sidecar=SidecarSnapshot(), term_w=140, term_h=45, now=1000.0,
        view="progress", glyphs="unicode",
    )
    assert ctx.is_2d
    lines = spine_lines(ctx, FULL_SPINE_LINES)
    assert len(lines) == 9
    assert all(strip_ansi(l).strip() != "" for l in lines)
    assert any(strip_ansi(l).startswith("cv2") for l in lines)


# --- Regression coverage for defects found while implementing this task -----
#
# The tests below were added on top of the task brief's own 11 -- each pins a
# real bug that the given 11 tests do not (and, for the truncation case,
# cannot) catch, since `spine_lines`' own final `_ansi_truncate` safety net
# silently rescues an over-length line by clipping it.


def test_verdict_reflects_actual_decision_health_not_hardcoded_ok(tmp_path):
    """The fixture's rows carry no `potential_kj_mol`, which the shared
    `_dashboard_decision_state` health check (gareus/logger.py) treats as a
    non-finite reading for every replica -- so this run's real health is BAD,
    not OK. `_dashboard_decision_state` returns `{"health", "issues", ...}`;
    reading nonexistent `"status"`/`"issue_count"` keys instead would make the
    verdict silently and permanently read "OK" regardless of ctx.decision.
    """
    text = strip_ansi("\n".join(spine_lines(_ctx(tmp_path), FULL_SPINE_LINES)))
    assert "BAD" in text


def test_verdict_reads_as_unavailable_not_ok_when_the_decision_call_raises(tmp_path, monkeypatch):
    """I2: a swallowed exception in `_dashboard_decision_state` must not render
    as the healthiest possible value. `build_context`'s own `except Exception`
    used to fall back to a bare `{}`, and `_verdict`'s `.get("health", "OK")`
    default then rendered a green "✓ OK" for a run whose health could not even
    be assessed -- an absence read as the best case, not as an absence.
    """
    logger = DistanceLogger(tmp_path, argparse.Namespace(timestep_fs=2.0), no_file_persistence=True)
    for w in range(25):
        logger.history_by_window[w] = [CENTERS[w] + 0.1 * (i % 5 - 2) for i in range(50)]

    def _boom(*_args, **_kwargs):
        raise RuntimeError("boom: decision state unavailable")

    monkeypatch.setattr(logger, "_dashboard_decision_state", _boom)
    rows = [{"replica": w, "window": w, "center_A": CENTERS[w], "k_kcal_mol_A2": 2.5,
             "cv_A": CENTERS[w] + 0.05} for w in range(25)]
    ctx = build_context(
        logger=logger, rows=rows, phase="gareus_production", step=12_450_000,
        total_steps=37_500_000, summary={}, dashboard_info={
            "centers_a": list(CENTERS[:25]), "n_windows": 25, "k_list": [2.5] * 25,
            "exchange_stats": {f"{i}-{i+1}": {"attempts": 400, "accepted": 120}
                               for i in range(24)},
            "primary_cv_label": "contacts", "primary_cv_units": "A"},
        sidecar=SidecarSnapshot(), term_w=140, term_h=45, now=1000.0,
        view="progress", glyphs="unicode",
    )
    assert ctx.decision.get("health") == "UNAVAILABLE"
    text = strip_ansi("\n".join(spine_lines(ctx, FULL_SPINE_LINES)))
    assert "OK" not in text
    assert "unavailable" in text.lower()


def test_win_and_exchange_strips_are_offset_by_one_column(tmp_path):
    """Lines 6/7 share an x-axis by design: pair i (between windows i and i+1)
    must read as a seam, not sit directly under either window's own glyph. With
    25 windows and no bucketing, that means the exchange strip's first cell
    must start exactly one column to the right of the window strip's first cell.

    Uses a non-uniform acceptance pattern (not the shared `_ctx` fixture's flat
    0.3 everywhere) so the exchange strip's own cell 0 renders a real density
    glyph rather than a blank " " level -- a uniform strip would make an
    un-shifted prefix indistinguishable from a shifted one by pure coincidence.
    """
    accepted = [200 if i == 0 else (20 if i % 2 == 0 else 380) for i in range(24)]
    logger = DistanceLogger(tmp_path, argparse.Namespace(timestep_fs=2.0), no_file_persistence=True)
    for w in range(25):
        logger.history_by_window[w] = [CENTERS[w] + 0.1 * (i % 5 - 2) for i in range(50)]
    rows = [{"replica": w, "window": w, "center_A": CENTERS[w], "k_kcal_mol_A2": 2.5,
             "cv_A": CENTERS[w] + 0.05} for w in range(25)]
    ctx = build_context(
        logger=logger, rows=rows, phase="gareus_production", step=12_450_000,
        total_steps=37_500_000, summary={}, dashboard_info={
            "centers_a": list(CENTERS[:25]), "n_windows": 25, "k_list": [2.5] * 25,
            "exchange_stats": {f"{i}-{i+1}": {"attempts": 400, "accepted": accepted[i]}
                               for i in range(24)},
            "primary_cv_label": "contacts", "primary_cv_units": "A",
            "adaptive_phase": {"epoch_index": 1, "epoch_total": 2,
                               "segment_name": "epoch_001/topup_002"}},
        sidecar=SidecarSnapshot(), term_w=140, term_h=45, now=1000.0,
        view="progress", glyphs="unicode",
    )
    lines = spine_lines(ctx, FULL_SPINE_LINES)
    plain = [strip_ansi(l) for l in lines]
    win_line = next(l for l in plain if l.startswith("win"))
    exch_line = next(l for l in plain if l.startswith("exch"))
    # The window strip's own cell 0 begins right after the "win   |" label+rule;
    # find that column directly (not by stripping leading spaces off either
    # prefix, since a bucket's own first glyph can legitimately be a blank " "
    # level, which would be indistinguishable from prefix padding).
    win_strip_start = win_line.index("|") + 1
    assert win_line[:win_strip_start] == "win   |"
    assert exch_line[win_strip_start] == " "          # the extra offset column
    assert exch_line[win_strip_start + 1] != " "       # pair 0's own real glyph
    assert exch_line[:win_strip_start + 1] == "exch    "


def test_alert_line_carries_the_literal_status_word_not_just_the_reason(tmp_path):
    """`SATURATED`/`ok` already appear as literal text on the gamd line; `BAD`
    now does too on the identity line's verdict (see the fix above). The alert
    line itself used to print only `s.reasons` ("low accept 0.12"), never the
    window's own `BAD`/`WARN` status word -- this fixture builds a window pair
    with real overlap (so the overlap check doesn't out-rank it) and a low but
    not-dead acceptance rate, landing it squarely in WARN, to pin that the
    alert line surfaces the status word itself.
    """
    n = 5
    centers = [0.3 * i for i in range(n)]
    logger = DistanceLogger(tmp_path, argparse.Namespace(timestep_fs=2.0), no_file_persistence=True)
    for w in range(n):
        logger.history_by_window[w] = [centers[w] + 0.15 * (i % 5 - 2) for i in range(50)]
    rows = [{"replica": w, "window": w, "center_A": centers[w], "k_kcal_mol_A2": 2.5,
             "cv_A": centers[w] + 0.01} for w in range(n)]
    exchange_stats = {f"{i}-{i+1}": {"attempts": 400, "accepted": 200} for i in range(n - 1)}
    exchange_stats["0-1"] = {"attempts": 400, "accepted": 50}   # 0.125: WARN, not dead
    ctx = build_context(
        logger=logger, rows=rows, phase="gareus_production", step=1000, total_steps=100000,
        summary={}, dashboard_info={
            "centers_a": centers, "n_windows": n, "k_list": [2.5] * n,
            "exchange_stats": exchange_stats,
            "primary_cv_label": "contacts", "primary_cv_units": "A"},
        sidecar=SidecarSnapshot(), term_w=140, term_h=45, now=1000.0,
        view="progress", glyphs="unicode",
    )
    text = strip_ansi("\n".join(spine_lines(ctx, FULL_SPINE_LINES)))
    assert "WARN" in text


def test_run_and_pool_bars_fit_without_needing_ellipsis_truncation(tmp_path):
    """The elastic middle must be sized from the *measured* width of the fixed
    parts including the bracket characters -- omitting the brackets from the
    budget calculation makes every run/pool line exactly 2 columns too long,
    silently rescued (and truncated, dropping real content) by the final
    `_ansi_truncate` safety net instead of fitting cleanly.
    """
    sidecar = SidecarSnapshot(pool={"total_ns": 15000.0, "used_ns": 7500.0,
                                    "remaining_ns": 7500.0, "events": []})
    lines = spine_lines(_ctx(tmp_path, sidecar=sidecar), FULL_SPINE_LINES)
    assert not any(strip_ansi(l).endswith("…") for l in lines)


# --- stage identity in the header -------------------------------------------
# The terminal `final` stage used to render `ep 0/?`: the producer's bare
# `except` mapped an unparseable directory name to a valid-looking index 0,
# and `epoch_total` was a hardcoded "?". Three epochs were complete at the
# time, so the header stated the opposite of the truth.

def _phase_ctx(tmp_path, adaptive_phase, *, n=25, term_w=140):
    logger = DistanceLogger(tmp_path, argparse.Namespace(timestep_fs=2.0), no_file_persistence=True)
    for w in range(n):
        logger.history_by_window[w] = [CENTERS[w] + 0.1 * (i % 5 - 2) for i in range(50)]
    rows = [{"replica": w, "window": w, "center_A": CENTERS[w], "k_kcal_mol_A2": 2.5,
             "cv_A": CENTERS[w] + 0.05} for w in range(n)]
    return build_context(
        logger=logger, rows=rows, phase="gareus_production", step=12_450_000,
        total_steps=37_500_000, summary={}, dashboard_info={
            "centers_a": list(CENTERS[:n]), "n_windows": n, "k_list": [2.5] * n,
            "exchange_stats": {f"{i}-{i+1}": {"attempts": 400, "accepted": 120}
                               for i in range(n - 1)},
            "primary_cv_label": "contacts", "primary_cv_units": "A",
            "adaptive_phase": adaptive_phase},
        sidecar=SidecarSnapshot(), term_w=term_w, term_h=45, now=1000.0,
        view="progress", glyphs="unicode",
    )


def test_final_stage_header_never_claims_epoch_zero(tmp_path):
    ctx = _phase_ctx(tmp_path, {"is_final_stage": True, "segment_name": "topup_002_6910000"})
    text = strip_ansi("\n".join(spine_lines(ctx, FULL_SPINE_LINES)))
    assert "ep 0" not in text, "the final stage is not epoch 0"
    assert "0/?" not in text
    assert "final" in text, "the terminal stage must name itself"


def test_topup_inside_a_real_epoch_names_the_run_length(tmp_path):
    ctx = _phase_ctx(tmp_path, {"epoch_index": 1, "epoch_total": 3,
                                "segment_name": "topup_002_2407000"})
    text = strip_ansi("\n".join(spine_lines(ctx, FULL_SPINE_LINES)))
    assert "ep 1/3" in text
    assert "1/?" not in text, "max_epochs is known and must be threaded through"


def test_epoch_zero_still_renders_as_epoch_zero(tmp_path):
    """The fix must not overcorrect a genuine first epoch into the final stage."""
    ctx = _phase_ctx(tmp_path, {"epoch_index": 0, "epoch_total": 3, "segment_name": "baseline"})
    text = strip_ansi("\n".join(spine_lines(ctx, FULL_SPINE_LINES)))
    assert "ep 0/3" in text


def test_run_health_badge_names_its_unit_so_it_cannot_read_as_a_window_count(tmp_path):
    """`✗ BAD 3` sat directly above a window table listing 4 BAD windows, and
    read as a contradictory count of the same thing. The badge counts run-health
    issues from `_dashboard_decision_state` (replica finiteness, stuck traces,
    pool, connectivity) -- never windows. Naming the unit removes the collision;
    the health word itself stays, since that IS the run's health.
    """
    ctx = _phase_ctx(tmp_path, {"epoch_index": 1, "epoch_total": 3, "segment_name": "baseline"})
    text = strip_ansi("\n".join(spine_lines(ctx, FULL_SPINE_LINES)))
    assert "issue" in text, "the badge must say what it is counting"


def test_identity_line_starts_with_the_versioned_product_label(tmp_path):
    from gareus.branding import product_label

    first = strip_ansi(spine_lines(_ctx(tmp_path), FULL_SPINE_LINES)[0])
    assert first.startswith(product_label() + "  ")


def test_identity_line_never_truncates_the_verdict_on_a_long_run_label():
    from gareus.dashboard.spine import _identity_with_verdict

    verdict = "\x1b[31m✗ BAD 1 issue\x1b[0m"
    identity = "ATLaS-MD v0.8.3  " + "a_very_long_campaign_directory_name_" * 4 + "   final  25 win"
    line = _identity_with_verdict(identity, verdict, 100)
    assert strip_ansi_len(line) <= 100
    assert strip_ansi(line).endswith("✗ BAD 1 issue")
    assert "…" in strip_ansi(line)


def test_identity_line_keeps_the_wide_gap_when_everything_fits():
    from gareus.dashboard.spine import _identity_with_verdict

    assert _identity_with_verdict("ATLaS-MD v0.8.3  run", "✓ OK", 140) == "ATLaS-MD v0.8.3  run      ✓ OK"
