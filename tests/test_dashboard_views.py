import argparse
import dataclasses
import re

from gareus.dashboard import view_physics, view_progress, view_windows
from gareus.dashboard.context import build_context
from gareus.dashboard.sidecar import SidecarSnapshot
from gareus.logger import DistanceLogger
from gareus.tui import strip_ansi

# Matches the potential-energy histogram bar's "kJ |<bar>|" suffix -- see
# tests/test_dashboard_scaling_integration.py's identical pattern, which pins
# the same glyph set against `logger._render_screen_frame`'s WINDOWS view
# (`_render_dashboard`, which this once described, is gone -- removed once
# the screen engine was wired in).
_PE_BAR_RE = re.compile(r"kJ \|([ ·░▒▓█●]+)\|")

CENTERS = tuple(4.0 + 0.55 * i for i in range(6))
# A fixed wall clock six hours after the segment started. Without an explicit
# `eta_start_wall` the context would measure elapsed time against the logger's real
# `start_wall`, giving elapsed_s == 0 and a NaN throughput -- which no projection
# assertion could ever satisfy.
NOW = 1_700_000_000.0

# Shaped like the real ledger: the three extension-loop segments interleave
# rather than repeating consecutively (epoch_001/baseline appears 3x here,
# separated by the two final/* segments each time).
POOL = {
    "total_ns": 15000.0, "used_ns": 7500.0, "remaining_ns": 7500.0, "timestep_fs": 4.0,
    "events": [
        {"label": "epoch_000", "kind": "adaptive_epoch", "consumed_ns": 1875.0, "n_states": 27},
        {"label": "epoch_001/baseline", "kind": "scheduled_epoch", "consumed_ns": 1391.7, "n_states": 29},
        {"label": "final/baseline", "kind": "scheduled_final", "consumed_ns": 14.5, "n_states": 29},
        {"label": "epoch_001/baseline", "kind": "scheduled_epoch", "consumed_ns": 1029.3, "n_states": 29},
        {"label": "final/baseline", "kind": "scheduled_final", "consumed_ns": 14.5, "n_states": 29},
        {"label": "epoch_001/baseline", "kind": "scheduled_epoch", "consumed_ns": 757.5, "n_states": 29},
        {"label": "final/topup_001_375000", "kind": "scheduled_final", "consumed_ns": 43.5, "n_states": 29},
    ],
}
GAMD = {"joint_envelope": {"Dihedral": {"sigma0_kj_mol": 12.552, "sigmaV_kj_mol": 11.039, "k0": 1.0}}}


def _ctx(tmp_path, *, view="progress", sidecar=None, exchange=None, n=6, term_w=140, term_h=45):
    logger = DistanceLogger(tmp_path, argparse.Namespace(timestep_fs=4.0), no_file_persistence=True)
    for w in range(n):
        logger.history_by_window[w] = [CENTERS[w] + 0.1 * (i % 5 - 2) for i in range(40)]
    logger.boost_history_all = [2.0 + 0.4 * (i % 6) for i in range(120)]
    rows = [{"replica": w, "window": w, "center_A": CENTERS[w], "k_kcal_mol_A2": 2.5,
             "cv_A": CENTERS[w] + 0.05, "umbrella_bias_kcal_mol": 0.0,
             "umbrella_pull_kcal_mol_A": 0.0} for w in range(n)]
    stats = exchange if exchange is not None else {
        f"{i}-{i+1}": {"attempts": 40, "accepted": 12} for i in range(n - 1)}
    return build_context(
        logger=logger, rows=rows, phase="gareus_production", step=1_000_000,
        total_steps=4_000_000, summary={}, dashboard_info={
            "centers_a": list(CENTERS[:n]), "n_windows": n, "k_list": [2.5] * n,
            "exchange_stats": stats, "primary_cv_label": "contacts",
            "primary_cv_units": "A", "primary_k_units": "kcal/mol/A^2",
            "eta_start_wall": NOW - 6 * 3600.0},
        sidecar=sidecar or SidecarSnapshot(), term_w=term_w, term_h=term_h, now=NOW,
        view=view, glyphs="unicode",
    )


def _text(rows):
    return strip_ansi("\n".join(line for row in rows for p in row.panels for line in p.lines))


def test_progress_view_lists_pool_events_chronologically(tmp_path):
    rows = view_progress.build(_ctx(tmp_path, sidecar=SidecarSnapshot(pool=POOL)))
    text = _text(rows)
    assert text.index("epoch_000") < text.index("epoch_001")
    assert "1875" in text


def test_progress_view_groups_repeated_segments_into_one_row_each(tmp_path):
    """Mirrors the real ledger's shape: repeats interleave rather than run
    consecutively, so grouping by label is what keeps the panel readable."""
    rows = view_progress.build(_ctx(tmp_path, sidecar=SidecarSnapshot(pool=POOL)))
    text = _text(rows)
    assert text.count("epoch_001/baseline") == 1        # one row, not three
    assert "x3" in text                                 # with its occurrence count
    assert "3178" in text or "3178.5" in text           # and its summed ns


def test_collapse_extension_rounds_matches_the_real_ledger_shape():
    """The chignolin_5 ledger: 27 events, 6 labels, three of them 8x interleaved."""
    events = []
    for _ in range(8):
        events.append({"label": "epoch_001/baseline", "consumed_ns": 100.0, "n_states": 29})
        events.append({"label": "final/baseline", "consumed_ns": 14.5, "n_states": 29})
        events.append({"label": "final/topup_001_375000", "consumed_ns": 43.5, "n_states": 29})
    events.insert(0, {"label": "epoch_000", "consumed_ns": 1875.0, "n_states": 27})
    grouped = view_progress.collapse_extension_rounds(events)
    assert len(grouped) == 4                            # four distinct labels here
    labels = [row[0] for row in grouped]
    assert labels[0] == "epoch_000"                     # first-appearance order
    by_label = {row[0]: row for row in grouped}
    assert by_label["epoch_001/baseline"][1] == 8       # occurrences
    assert by_label["epoch_001/baseline"][2] == 800.0   # summed ns
    assert by_label["final/topup_001_375000"][4] is True    # holds the final event


def test_progress_view_explains_a_missing_pool_ledger(tmp_path):
    text = _text(view_progress.build(_ctx(tmp_path)))
    assert "adaptive_runtime_pool.json" in text


def test_progress_view_projects_remaining_budget_in_gpu_days(tmp_path):
    text = _text(view_progress.build(_ctx(tmp_path, sidecar=SidecarSnapshot(pool=POOL))))
    assert "7500" in text
    assert "GPU-day" in text


def test_timeline_labels_never_silently_cut_a_step_count(tmp_path):
    """`epoch_001/topup_001_3388000` truncated to `...338800` reads as a real
    step count and is wrong; two segments can also collide on the same cut."""
    from gareus.dashboard.view_progress import _fit_label
    assert _fit_label("epoch_001/topup_001_3388000", 26) == "epoch_001/topup_001"
    assert _fit_label("epoch_001/topup_002_8178000", 26) == "epoch_001/topup_002"
    assert _fit_label("epoch_000", 26) == "epoch_000"
    long_non_numeric = "a" * 40
    assert _fit_label(long_non_numeric, 26).endswith("…")
    assert len(_fit_label(long_non_numeric, 26)) <= 26


def test_timeline_reserve_row_aligns_with_the_rows_above_it(tmp_path):
    """The panel exists so bar lengths can be compared by eye."""
    rows = view_progress.build(_ctx(tmp_path, sidecar=SidecarSnapshot(pool=POOL)))
    body = [strip_ansi(l) for row in rows for p in row.panels for l in p.lines]
    bars = [l.index("█") for l in body if "█" in l]
    reserves = [l.index("░") for l in body if "░" in l]
    assert bars and reserves
    assert len(set(bars)) == 1                  # every segment bar starts alike
    assert reserves[0] == bars[0]               # and the reserve starts there too


def test_progress_view_returns_rows_of_panels(tmp_path):
    rows = view_progress.build(_ctx(tmp_path, sidecar=SidecarSnapshot(pool=POOL)))
    assert rows and all(row.panels for row in rows)
    assert {p.key for row in rows for p in row.panels} >= {"timeline", "projection", "throughput"}


from gareus.dashboard.view_physics import connected_components


def test_connected_components_returns_one_set_for_a_fully_coupled_chain():
    pairs = {(i, i + 1): 0.30 for i in range(5)}
    comps = connected_components(6, pairs)
    assert len(comps) == 1 and comps[0] == frozenset(range(6))


def test_connected_components_splits_at_a_dead_pair():
    pairs = {(0, 1): 0.30, (1, 2): 0.001, (2, 3): 0.30}
    comps = connected_components(4, pairs)
    assert sorted(len(c) for c in comps) == [2, 2]


def test_connected_components_does_not_link_windows_over_an_unattempted_pair():
    """An unattempted pair carries a NaN rate and must never read as a link.

    This does not, by itself, prove the implementation's `math.isfinite` guard is
    what excludes it: `nan >= threshold` is already `False` under Python's own
    float-comparison semantics for any real threshold, so no test built purely
    from a NaN input can discriminate whether that guard is even present. The
    guard's presence is proven separately, below, with a non-finite value (`inf`)
    that WOULD pass the bare `>=` comparison. What this test still pins is real
    and worth keeping regardless of mechanism: an unattempted pair must never
    silently become a link.
    """
    pairs = {(0, 1): float("nan")}
    assert len(connected_components(2, pairs)) == 2


def test_connected_components_ignores_a_non_finite_rate_even_when_it_would_pass_the_threshold():
    """Unlike NaN, `inf` DOES pass a bare `rate >= threshold` comparison
    (`inf >= 0.02` is `True`), so this is the case that actually requires the
    `math.isfinite` guard to be present: without it, a corrupt/infinite rate
    would incorrectly link two windows."""
    pairs = {(0, 1): float("inf")}
    assert len(connected_components(2, pairs)) == 2


def test_connectivity_verdict_holds_judgement_before_any_exchange_is_attempted(tmp_path):
    """Zero attempts is the state of every run's first frames. Announcing
    "1/6 connected" there would be a false claim of total MBAR disconnection,
    and would pin the auto view to PHYSICS until exchanges accumulate."""
    ctx = _ctx(tmp_path, view="physics", exchange={
        f"{i}-{i+1}": {"attempts": 0, "accepted": 0} for i in range(5)})
    state, _largest, measured = view_physics.connectivity_verdict(ctx)
    assert state == "unmeasured"
    assert measured == 0
    text = _text(view_physics.build(ctx))
    assert "not yet measurable" in text
    assert "connected" not in text.split("not yet measurable")[0].splitlines()[-1]


def test_connectivity_verdict_reports_coverage_when_only_some_pairs_are_measured(tmp_path):
    ctx = _ctx(tmp_path, view="physics", exchange={
        "0-1": {"attempts": 40, "accepted": 12},
        "1-2": {"attempts": 0, "accepted": 0},
        "2-3": {"attempts": 0, "accepted": 0},
        "3-4": {"attempts": 0, "accepted": 0},
        "4-5": {"attempts": 0, "accepted": 0}})
    state, _largest, measured = view_physics.connectivity_verdict(ctx)
    assert state == "partial"
    assert measured == 1
    assert "holding judgement" in _text(view_physics.build(ctx))


def test_connectivity_verdict_is_connected_on_a_spanning_set_whatever_the_mode(tmp_path):
    """A spanning set proves connectivity. There is no fixed expected pair count:
    --exchange-mode offers neighbor, random-pair, all-pair-sweep and gibbs-walk,
    so counting measured pairs against n-1 would assert a verdict after 5 of 15
    pairs on a 6-window all-pair run."""
    ctx = _ctx(tmp_path, view="physics", exchange={
        f"{i}-{i+1}": {"attempts": 40, "accepted": 12} for i in range(5)})
    state, largest, _measured = view_physics.connectivity_verdict(ctx)
    assert state == "connected"
    assert largest == 6


def test_connectivity_verdict_holds_judgement_while_a_window_is_untried(tmp_path):
    """Non-neighbour modes measure arbitrary pairs, so a window with no measured
    pair yet must read as missing evidence, not as an isolated state."""
    ctx = _ctx(tmp_path, view="physics", exchange={
        "0-3": {"attempts": 40, "accepted": 12},
        "1-4": {"attempts": 40, "accepted": 9}})
    state, _largest, measured = view_physics.connectivity_verdict(ctx)
    assert state == "partial"
    assert measured == 2
    assert "not yet tried" in _text(view_physics.build(ctx))


def test_connectivity_verdict_asserts_a_split_only_once_every_window_is_tried(tmp_path):
    ctx = _ctx(tmp_path, view="physics", exchange={
        "0-1": {"attempts": 40, "accepted": 12}, "1-2": {"attempts": 40, "accepted": 0},
        "2-3": {"attempts": 40, "accepted": 12}, "3-4": {"attempts": 40, "accepted": 12},
        "4-5": {"attempts": 40, "accepted": 12}})
    state, largest, measured = view_physics.connectivity_verdict(ctx)
    assert state == "split"
    assert measured == 5
    assert largest == 4       # {0,1} and {2,3,4,5}: every window tried, still in pieces


def test_physics_view_reports_connectivity_and_names_isolated_windows(tmp_path):
    ctx = _ctx(tmp_path, view="physics", exchange={
        "0-1": {"attempts": 40, "accepted": 12}, "1-2": {"attempts": 40, "accepted": 0},
        "2-3": {"attempts": 40, "accepted": 12}, "3-4": {"attempts": 40, "accepted": 12},
        "4-5": {"attempts": 40, "accepted": 12}})
    text = _text(view_physics.build(ctx))
    assert "connected" in text
    assert "w02" in text or "w01" in text


def test_physics_view_compares_restraint_sigma_with_window_spacing(tmp_path):
    text = _text(view_physics.build(_ctx(tmp_path, view="physics")))
    assert "spacing" in text
    assert "σ" in text or "sigma" in text


def test_physics_view_includes_overlap_and_boost_panels(tmp_path):
    rows = view_physics.build(_ctx(tmp_path, view="physics",
                                   sidecar=SidecarSnapshot(gamd=GAMD)))
    assert {p.key for row in rows for p in row.panels} >= {"overlap", "boost"}


def test_sigma_panel_reports_no_overlap_data_rather_than_zero_overlap(tmp_path):
    """`ctx.overlap_pairs` is legitimately empty for a run's first few samples per
    window (`_hist_overlap` needs >=5 finite samples per side). Empty must render
    as "no data yet", never a fabricated "0.00" overlap -- the same false-alarm
    class the connectivity verdict machinery exists to avoid."""
    ctx = dataclasses.replace(_ctx(tmp_path, view="physics"), overlap_pairs={})
    text = _text(view_physics.build(ctx))
    after = text.split("observed median overlap")[1]
    assert "0.00" not in after.splitlines()[0]


from gareus.dashboard.ranking import OK as _OK
from gareus.dashboard.ranking import WindowStatus as _WindowStatus
from gareus.dashboard.ranking import rank_windows as _rank
from gareus.dashboard.view_windows import select_window


def _statuses(ctx, acceptance):
    return _rank(n_windows=ctx.n_windows, centers_a=ctx.centers_a, k_list=ctx.k_list,
                 acceptance_by_window=acceptance, overlap_by_pair=ctx.overlap_pairs,
                 delta_by_window=ctx.deltas, temperature_k=ctx.temperature_k)


def test_select_window_picks_the_worst_ranked_window(tmp_path):
    ctx = _ctx(tmp_path, view="windows")
    statuses = _statuses(ctx, {0: 0.30, 1: 0.30, 2: 0.001, 3: 0.30, 4: 0.30, 5: 0.30})
    assert select_window(ctx, statuses) == 2


def test_select_window_falls_back_to_window_zero_when_all_are_healthy(tmp_path):
    ctx = _ctx(tmp_path, view="windows")
    statuses = _statuses(ctx, {w: 0.30 for w in range(6)})
    assert select_window(ctx, statuses) == 0


def test_select_window_true_fallback_line_fires_when_every_status_is_actually_ok(tmp_path):
    """Companion to the test above: that one's shared `_ctx` fixture builds
    non-overlapping per-window histories on purpose (window spacing 0.55A vs. a
    +/-0.2A sample range per window -- see CENTERS/`_ctx` at the top of this
    file), so `ctx.overlap_pairs` there is 0.0 for every neighbour pair. That
    makes `rank_windows` mark *every* window BAD via the dead-overlap branch
    even with all-healthy acceptance, and window 0 wins `select_window`'s loop
    on the first (worst-ranked, tie-broken-by-index) iteration -- never
    reaching this function's actual `statuses[0].window` fallback line at all.
    A broken fallback line would not fail that test. This one builds genuinely
    all-OK statuses directly, bypassing rank_windows/`_ctx` overlap entirely, so
    the fallback line itself is what is pinned.
    """
    ctx = _ctx(tmp_path, view="windows")
    statuses = tuple(_WindowStatus(w, 0, _OK, ()) for w in range(6))
    assert select_window(ctx, statuses) == 0


def test_windows_view_orders_rows_table_then_maps_then_detail(tmp_path):
    rows = view_windows.build(_ctx(tmp_path, view="windows"))
    keys = [tuple(p.key for p in row.panels) for row in rows]
    assert keys[0] == ("windows",)
    assert keys[1] == ("cv_map", "pe_map")
    assert keys[2][0].startswith("detail-w")


def test_windows_view_gives_the_maps_row_lower_priority_than_table_and_detail(tmp_path):
    rows = view_windows.build(_ctx(tmp_path, view="windows"))
    by_key = {p.key: p for row in rows for p in row.panels}
    assert by_key["windows"].priority == 1
    assert by_key["cv_map"].priority == 2
    assert by_key["pe_map"].priority == 3
    assert min(p.priority for p in rows[2].panels) == 1


def test_windows_view_keeps_the_historical_cv_pe_weights(tmp_path):
    rows = view_windows.build(_ctx(tmp_path, view="windows"))
    by_key = {p.key: p for row in rows for p in row.panels}
    assert (by_key["cv_map"].weight, by_key["pe_map"].weight) == (2.4, 0.9)


def test_windows_view_map_bar_widths_match_the_pinned_pe_arithmetic(tmp_path):
    """`tests/test_dashboard_scaling_integration.py::
    test_pe_histogram_bar_width_pinned_at_160x40` pins the historical PE-bar
    arithmetic through `logger._render_screen_frame`'s WINDOWS view and
    `_weighted_panel_widths` directly -- neither one ever reaches this view's
    own `_map_bar_widths`, so nothing else in the suite catches a regression in
    this view's own bar-width plumbing. Pin it here too.
    """
    from gareus.dashboard.view_windows import _map_bar_widths
    ctx = _ctx(tmp_path, view="windows", term_w=160)
    _cv_bar_w, pe_bar_w = _map_bar_widths(ctx)
    assert pe_bar_w == 18


def test_windows_view_pe_map_renders_a_bar_at_the_pinned_width(tmp_path):
    """Companion, end-to-end: seeds real PE history and confirms the bar
    `pe_map_panel` actually renders through `view_windows.build` is 18 columns
    wide at 160 columns, not just that the width arithmetic alone says so."""
    ctx = dataclasses.replace(
        _ctx(tmp_path, view="windows", term_w=160),
        pe_history_by_replica={w: tuple(100.0 + w + i for i in range(20)) for w in range(6)},
    )
    text = _text(view_windows.build(ctx))
    matches = _PE_BAR_RE.findall(text)
    assert matches, "expected a rendered 'kJ |<bar>|' PE histogram bar"
    assert len(matches[0]) == 18


def test_windows_view_drops_the_maps_row_before_table_or_detail_on_a_short_terminal(tmp_path):
    """Measures the allocator against this view's real three rows, rather than
    inferring the drop order from priority numbers alone."""
    from gareus.tui_screen import allocate_rows, row_min

    rows = view_windows.build(_ctx(tmp_path, view="windows"))
    table_row, maps_row, detail_row = rows
    budget = row_min(table_row) + row_min(detail_row)
    allocated, dropped = allocate_rows(rows, budget=budget)
    assert set(dropped) == {p.key for p in maps_row.panels}
    kept_keys = {p.key for row, _body in allocated for p in row.panels}
    assert kept_keys == {p.key for p in table_row.panels} | {p.key for p in detail_row.panels}
