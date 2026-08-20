import argparse

from gareus.dashboard import view_physics, view_progress
from gareus.dashboard.context import build_context
from gareus.dashboard.sidecar import SidecarSnapshot
from gareus.logger import DistanceLogger
from gareus.tui import strip_ansi

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


def test_connected_components_treats_unattempted_pairs_as_unlinked():
    pairs = {(0, 1): float("nan")}
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


def test_connectivity_verdict_asserts_a_split_only_once_every_pair_is_measured(tmp_path):
    ctx = _ctx(tmp_path, view="physics", exchange={
        "0-1": {"attempts": 40, "accepted": 12}, "1-2": {"attempts": 40, "accepted": 0},
        "2-3": {"attempts": 40, "accepted": 12}, "3-4": {"attempts": 40, "accepted": 12},
        "4-5": {"attempts": 40, "accepted": 12}})
    state, largest, measured = view_physics.connectivity_verdict(ctx)
    assert state == "split"
    assert measured == 5
    assert largest == 4


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
