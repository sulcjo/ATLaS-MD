import argparse
import dataclasses

import pytest

from gareus.dashboard.context import build_context
from gareus.dashboard.panels import (
    boost_envelope_panel,
    overlap_panel,
    panel,
    window_detail_panel,
    window_table_panel,
)
from gareus.dashboard.ranking import BAD, rank_windows
from gareus.dashboard.sidecar import SidecarSnapshot
from gareus.logger import DistanceLogger
from gareus.tui import strip_ansi

CENTERS = (4.0, 4.55, 5.10, 5.65)


def _ctx(tmp_path, *, sidecar=None, histories=True, exchange=True, secondary=None):
    args = argparse.Namespace(timestep_fs=2.0, temperature_k=300.0)
    logger = DistanceLogger(tmp_path, args, no_file_persistence=True)
    if histories:
        for w, c in enumerate(CENTERS):
            logger.history_by_window[w] = [c + 0.1 * (i % 5 - 2) for i in range(60)]
        logger.boost_history_all = [2.0 + 0.5 * (i % 7) for i in range(200)]
    stats = {f"{i}-{i+1}": {"attempts": 40, "accepted": 12} for i in range(3)} if exchange else {}
    rows = [{"replica": w, "window": w, "center_A": c, "k_kcal_mol_A2": 2.5,
             "cv_A": c + 0.05, "umbrella_bias_kcal_mol": 0.01,
             "umbrella_pull_kcal_mol_A": 0.1} for w, c in enumerate(CENTERS)]
    return build_context(
        logger=logger, rows=rows, phase="gareus_production", step=10, total_steps=100,
        summary={}, dashboard_info={"centers_a": list(CENTERS), "n_windows": 4,
                                    "k_list": [2.5] * 4, "exchange_stats": stats,
                                    "primary_cv_label": "contacts", "primary_cv_units": "A",
                                    **({"secondary_cv_centers": list(secondary)}
                                       if secondary is not None else {})},
        sidecar=sidecar or SidecarSnapshot(), term_w=140, term_h=45, now=1000.0,
        view="physics", glyphs="unicode",
    )


def test_panel_helper_builds_a_frozen_panel_with_the_given_budget():
    p = panel("k", "title", ["a", "b"], min_lines=1, want_lines=4, priority=2, weight=1.5)
    assert (p.key, p.title, p.lines, p.min_lines, p.want_lines, p.priority, p.weight) == (
        "k", "title", ("a", "b"), 1, 4, 2, 1.5)


def test_overlap_panel_lists_pairs_worst_first_with_a_severity_tail(tmp_path):
    ctx = _ctx(tmp_path)
    p = overlap_panel(ctx)
    text = strip_ansi("\n".join(p.lines))
    assert "w00-w01" in text
    values = [float(line.split()[1]) for line in strip_ansi("\n".join(p.lines)).splitlines()
              if line.strip().startswith("w")]
    assert values == sorted(values)                 # ascending overlap == worst first
    assert p.priority == 1 and p.min_lines == 5


def test_overlap_panel_states_that_it_has_no_samples_yet(tmp_path):
    p = overlap_panel(_ctx(tmp_path, histories=False))
    assert "insufficient samples" in strip_ansi("\n".join(p.lines))


def test_boost_envelope_panel_reports_sigma_target_and_k0_saturation(tmp_path):
    sidecar = SidecarSnapshot(gamd={"joint_envelope": {"Dihedral": {
        "sigma0_kj_mol": 12.552, "sigmaV_kj_mol": 11.039, "k0": 1.0}}})
    text = strip_ansi("\n".join(boost_envelope_panel(_ctx(tmp_path, sidecar=sidecar)).lines))
    assert "11.04" in text and "12.55" in text
    assert "SATURATED" in text                      # k0 at its ceiling, stated as text


def test_boost_envelope_panel_does_not_call_an_unmeasurable_score_healthy(tmp_path):
    """`nan > 1.0` is False, so a naive threshold chain reports "could not be
    computed" as OK -- in the panel whose job is flagging a bad boost."""
    ctx = _ctx(tmp_path, sidecar=SidecarSnapshot(gamd={"joint_envelope": {"Dihedral": {
        "sigma0_kj_mol": 12.552, "sigmaV_kj_mol": float("nan"), "k0": float("nan")}}}))
    # One sample: a variance-free input makes skew/kurtosis and the score undefined.
    ctx = dataclasses.replace(ctx, boost_history_all=(2.4,))
    text = strip_ansi("\n".join(boost_envelope_panel(ctx).lines))
    assert "nan" not in text
    assert "not measurable" in text or "not reported" in text
    anharm_line = next(l for l in text.splitlines() if "anharmonicity" in l)
    assert " ok" not in anharm_line.lower()


def test_boost_envelope_panel_says_so_when_the_run_has_no_gamd(tmp_path):
    text = strip_ansi("\n".join(boost_envelope_panel(_ctx(tmp_path)).lines))
    assert "no GaMD" in text


def test_window_table_panel_never_prints_a_literal_nan(tmp_path):
    """`nan` means "not measured"; printing it puts a token in a numeric column
    that reads as data. A 1D run has no cv2 column at all, and the first window
    has no left neighbour, so both are absences rather than measurements."""
    ctx = _ctx(tmp_path)
    statuses = rank_windows(
        n_windows=ctx.n_windows, centers_a=ctx.centers_a, k_list=ctx.k_list,
        acceptance_by_window=ctx.acceptance_windows, overlap_by_pair=ctx.overlap_pairs,
        delta_by_window=ctx.deltas, temperature_k=ctx.temperature_k)
    text = strip_ansi("\n".join(window_table_panel(ctx, statuses).lines))
    assert "nan" not in text
    assert "cv2 ctr" not in text           # 1D run: column dropped, not filled
    assert "—" in text                     # w00 has no left neighbour


def test_window_table_panel_shows_the_cv2_column_for_a_fully_populated_2d_run(tmp_path):
    """The conditional branch this guard added needs its own coverage, and the
    column must carry real numbers rather than collapse to dashes."""
    ctx = _ctx(tmp_path, secondary=[-2.0, -1.0, 0.0, 1.0])
    statuses = rank_windows(
        n_windows=ctx.n_windows, centers_a=ctx.centers_a, k_list=ctx.k_list,
        acceptance_by_window=ctx.acceptance_windows, overlap_by_pair=ctx.overlap_pairs,
        delta_by_window=ctx.deltas, temperature_k=ctx.temperature_k)
    text = strip_ansi("\n".join(window_table_panel(ctx, statuses).lines))
    assert "cv2 ctr" in text
    assert "-2.00" in text and "1.00" in text        # real values, not all dashes
    assert "nan" not in text


def test_window_table_panel_drops_the_cv2_column_when_only_some_windows_have_one(tmp_path):
    """A short secondary list would otherwise put an earlier window's centre on a
    later window's row -- plausible-looking data attributed to the wrong window."""
    ctx = _ctx(tmp_path, secondary=[-2.0, -1.0])      # 2 centres, 4 windows
    statuses = rank_windows(
        n_windows=ctx.n_windows, centers_a=ctx.centers_a, k_list=ctx.k_list,
        acceptance_by_window=ctx.acceptance_windows, overlap_by_pair=ctx.overlap_pairs,
        delta_by_window=ctx.deltas, temperature_k=ctx.temperature_k)
    text = strip_ansi("\n".join(window_table_panel(ctx, statuses).lines))
    assert "cv2 ctr" not in text


def test_window_table_panel_still_prints_real_measurements(tmp_path):
    """Guards the inverse regression: a `_num` that em-dashed everything would
    satisfy every "no nan" assertion in this file."""
    ctx = _ctx(tmp_path)
    statuses = rank_windows(
        n_windows=ctx.n_windows, centers_a=ctx.centers_a, k_list=ctx.k_list,
        acceptance_by_window=ctx.acceptance_windows, overlap_by_pair=ctx.overlap_pairs,
        delta_by_window=ctx.deltas, temperature_k=ctx.temperature_k)
    text = strip_ansi("\n".join(window_table_panel(ctx, statuses).lines))
    assert "4.00" in text                               # w00's centre
    assert text.count("—") < text.count(".")            # dashes are the exception


def test_window_detail_panel_never_prints_a_literal_nan_before_any_samples(tmp_path):
    """Reachable at the start of every run: no samples yet means no mean."""
    ctx = _ctx(tmp_path, histories=False)
    text = strip_ansi("\n".join(window_detail_panel(ctx, 0).lines))
    assert "nan" not in text
    assert "—" in text


def test_window_table_panel_puts_the_worst_window_first(tmp_path):
    ctx = _ctx(tmp_path)
    statuses = rank_windows(
        n_windows=ctx.n_windows, centers_a=ctx.centers_a, k_list=ctx.k_list,
        acceptance_by_window={0: 0.30, 1: 0.30, 2: 0.001, 3: 0.30},
        overlap_by_pair=ctx.overlap_pairs, delta_by_window=ctx.deltas,
        temperature_k=ctx.temperature_k,
    )
    lines = strip_ansi("\n".join(window_table_panel(ctx, statuses).lines)).splitlines()
    # Body rows are indented two spaces ("  w00  ..."); the header is flush-left
    # ("win   cv1 ctr ..."). A plain `.strip().startswith("w")` can no longer
    # tell them apart now that the header reads "win" instead of "#win" (see
    # task-11 fix report: the leading "#" was a hack to dodge exactly this
    # collision, removed because it read as a stray typo in a rendered frame).
    body = [l for l in lines if l.startswith("  w")]
    assert body[0].split()[0] == "w02"
    assert BAD in body[0]


def test_window_detail_panel_reads_the_trace_of_the_replica_in_that_window(tmp_path):
    """The trace map is replica-keyed, so a fixture where replica != window is
    the only shape that can catch indexing it by window index."""
    args = argparse.Namespace(timestep_fs=2.0, temperature_k=300.0)
    logger = DistanceLogger(tmp_path, args, no_file_persistence=True)
    logger.window_trace_by_replica[7] = [2, 3, 2]      # replica 7 sits in window 2
    logger.window_trace_by_replica[2] = [9, 9, 9]      # decoy: replica 2's own trail
    rows = [{"replica": 7, "window": 2, "center_A": 5.10, "k_kcal_mol_A2": 2.5,
             "cv_A": 5.15, "umbrella_bias_kcal_mol": 0.0, "umbrella_pull_kcal_mol_A": 0.0}]
    ctx = build_context(
        logger=logger, rows=rows, phase="gareus_production", step=1, total_steps=10,
        summary={}, dashboard_info={"centers_a": list(CENTERS), "n_windows": 4,
                                   "k_list": [2.5] * 4},
        sidecar=SidecarSnapshot(), term_w=140, term_h=45, now=1000.0,
        view="windows", glyphs="unicode",
    )
    text = strip_ansi("\n".join(window_detail_panel(ctx, 2).lines))
    assert "r07" in text
    assert "w09" not in text          # replica 2's decoy trail must not appear


def test_window_detail_panel_skips_occupancy_when_the_row_has_no_replica_key(tmp_path):
    """A row that matches the window filter but lacks a "replica" key must not
    raise -- replica_table_panel already degrades gracefully the same way via
    `r.get("replica", 0)`, and this lookup should match that convention."""
    args = argparse.Namespace(timestep_fs=2.0, temperature_k=300.0)
    logger = DistanceLogger(tmp_path, args, no_file_persistence=True)
    logger.window_trace_by_replica[7] = [2, 3, 2]
    rows = [{"window": 2, "center_A": 5.10, "k_kcal_mol_A2": 2.5,
             "cv_A": 5.15, "umbrella_bias_kcal_mol": 0.0, "umbrella_pull_kcal_mol_A": 0.0}]
    ctx = build_context(
        logger=logger, rows=rows, phase="gareus_production", step=1, total_steps=10,
        summary={}, dashboard_info={"centers_a": list(CENTERS), "n_windows": 4,
                                   "k_list": [2.5] * 4},
        sidecar=SidecarSnapshot(), term_w=140, term_h=45, now=1000.0,
        view="windows", glyphs="unicode",
    )
    text = strip_ansi("\n".join(window_detail_panel(ctx, 2).lines))
    assert "occupancy" not in text


def test_window_detail_panel_names_the_window_and_its_restraint(tmp_path):
    ctx = _ctx(tmp_path)
    text = strip_ansi("\n".join(window_detail_panel(ctx, 2).lines))
    assert "5.10" in text                           # its centre
    assert "2.5" in text                            # its k
    assert "w03" in text or "w01" in text           # its exchange partners
