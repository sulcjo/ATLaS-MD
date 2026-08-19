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
            "exchange_stats": {f"{i}-{i+1}": {"attempts": 40, "accepted": 12}
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
    strip = bucket_strip([1.0] * 364, statuses, cells=100, glyphs="ascii")
    assert "X" in strip                     # ascii mode encodes status in the glyph


def test_bucket_strip_handles_an_empty_input():
    assert bucket_strip([], [], cells=20) == ""


def test_spine_lines_returns_exactly_ten_lines_and_fits_the_width(tmp_path):
    lines = spine_lines(_ctx(tmp_path), FULL_SPINE_LINES)
    assert len(lines) == FULL_SPINE_LINES
    assert all(strip_ansi_len(l) <= 140 - 2 for l in lines)


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


def test_spine_says_the_pool_is_unavailable_rather_than_faking_a_bar(tmp_path):
    text = strip_ansi("\n".join(spine_lines(_ctx(tmp_path), FULL_SPINE_LINES)))
    assert "pool" in text and "unavailable" in text


def test_spine_reports_gamd_saturation_as_text_not_only_colour(tmp_path):
    sidecar = SidecarSnapshot(gamd={"joint_envelope": {"Dihedral": {
        "sigma0_kj_mol": 12.552, "sigmaV_kj_mol": 11.039, "k0": 1.0}}})
    text = strip_ansi("\n".join(spine_lines(_ctx(tmp_path, sidecar=sidecar), FULL_SPINE_LINES)))
    assert "SATURATED" in text


def test_compact_spine_returns_five_lines_and_keeps_verdict_and_progress(tmp_path):
    lines = spine_lines(_ctx(tmp_path, term_w=100, term_h=24), COMPACT_SPINE_LINES)
    assert len(lines) == COMPACT_SPINE_LINES
    text = strip_ansi("\n".join(lines))
    assert "gareus_production" in text
    assert "%" in text                       # progress survives the compact tier


def test_spine_omits_the_cv2_line_for_a_1d_run(tmp_path):
    lines = spine_lines(_ctx(tmp_path), FULL_SPINE_LINES)
    assert not any(strip_ansi(l).startswith("cv2") for l in lines)


# --- Regression coverage for defects found while implementing this task -----
#
# The three tests below were added on top of the task brief's own 11 -- each
# pins a real bug that the given 11 tests do not (and, for the truncation
# case, cannot) catch, since `spine_lines`' own final `_ansi_truncate` safety
# net silently rescues an over-length line by clipping it.


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
    accepted = [20 if i == 0 else (2 if i % 2 == 0 else 38) for i in range(24)]
    logger = DistanceLogger(tmp_path, argparse.Namespace(timestep_fs=2.0), no_file_persistence=True)
    for w in range(25):
        logger.history_by_window[w] = [CENTERS[w] + 0.1 * (i % 5 - 2) for i in range(50)]
    rows = [{"replica": w, "window": w, "center_A": CENTERS[w], "k_kcal_mol_A2": 2.5,
             "cv_A": CENTERS[w] + 0.05} for w in range(25)]
    ctx = build_context(
        logger=logger, rows=rows, phase="gareus_production", step=12_450_000,
        total_steps=37_500_000, summary={}, dashboard_info={
            "centers_a": list(CENTERS[:25]), "n_windows": 25, "k_list": [2.5] * 25,
            "exchange_stats": {f"{i}-{i+1}": {"attempts": 40, "accepted": accepted[i]}
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
    exchange_stats = {f"{i}-{i+1}": {"attempts": 40, "accepted": 20} for i in range(n - 1)}
    exchange_stats["0-1"] = {"attempts": 40, "accepted": 5}   # 0.125: WARN, not dead
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
