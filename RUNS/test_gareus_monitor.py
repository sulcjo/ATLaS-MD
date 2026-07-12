import io
import importlib.util
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gareus_monitor import (
    BOOST_TARGET_PER_WINDOW,
    PeptideState,
    _live_boost_window_view,
    _min_boost_samples_per_window,
    _plan_yaml_path,
    _rich_detail_panel,
    boost_values_from_entries,
    build_run_snapshot,
    discover,
    group_boost_values_by_window,
    match_slurm_jobs_to_states,
    parse_distances_samples,
    parse_squeue_rows,
    read_boost_samples,
    render_detail,
    summarize_boost_values,
)

_HAS_RICH = importlib.util.find_spec("rich") is not None


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")


def _append_jsonl(path: Path, *payloads: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for payload in payloads:
            f.write(json.dumps(payload) + "\n")


def _render_rich(renderable) -> str:
    from rich.console import Console

    console = Console(file=io.StringIO(), width=160, record=True, color_system=None, force_terminal=False)
    console.print(renderable)
    return console.export_text()


class PeptideStateBudgetTests(unittest.TestCase):
    def test_completed_segment_uses_pool_budget_without_double_counting_last_progress(self):
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "PEP_2d_run"
            _write_json(
                run_dir / "adaptive_production" / "adaptive_runtime_pool.json",
                {
                    "total_ns": 100.0,
                    "used_ns": 25.0,
                    "events": [
                        {
                            "label": "epoch_000",
                            "kind": "adaptive_epoch",
                            "consumed_ns": 25.0,
                        }
                    ],
                },
            )
            _append_jsonl(
                run_dir / "progress.jsonl",
                {
                    "event": "progress",
                    "phase": "gareus_production",
                    "percent": 100.0,
                    "aggregate_sim_time_ns": 25.0,
                    "wall_time_s": 1000.0,
                },
                {"event": "run_complete", "wall_time_s": 1001.0},
            )

            state = PeptideState.from_rundir(run_dir)
            state.refresh()

            self.assertEqual(state.committed_ns, 25.0)
            self.assertEqual(state.agg_ns, 0.0)
            self.assertEqual(state.global_percent, 25.0)

    def test_running_epoch_zero_adds_live_budget_and_displays_zero_of_total_epochs(self):
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "PEP_2d_run"
            (run_dir / "adaptive_production" / "epoch_000").mkdir(parents=True)
            _write_json(
                run_dir / "adaptive_production" / "adaptive_runtime_pool.json",
                {"total_ns": 100.0, "used_ns": 0.0, "events": []},
            )
            _write_json(
                run_dir / "config" / "effective_config.json",
                {"args": {"ap_epochs": 4}},
            )
            _append_jsonl(
                run_dir / "progress.jsonl",
                {
                    "event": "progress",
                    "phase": "gareus_production",
                    "percent": 50.0,
                    "aggregate_sim_time_ns": 12.5,
                    "wall_time_s": 2000.0,
                },
            )

            state = PeptideState.from_rundir(run_dir)
            state.refresh()

            self.assertEqual(state.phase, "gareus_production")
            self.assertEqual(state.committed_ns, 0.0)
            self.assertEqual(state.agg_ns, 12.5)
            self.assertEqual(state.global_percent, 12.5)
            self.assertEqual(state.epochs_completed, 0)
            self.assertEqual(state.total_epochs, 4)


class SlurmMonitorTests(unittest.TestCase):
    def test_parse_squeue_rows_keeps_running_job_fields_and_workdir(self):
        rows = parse_squeue_rows(
            "12345|chignolin_2d_run|RUNNING|12:03|1|gpu-a01|/runs/chignolin/chignolin_2d_run\n"
            "12346|DYKDDDDK-prod|PENDING|0:00|1|Priority|\n"
        )

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0].job_id, "12345")
        self.assertEqual(rows[0].name, "chignolin_2d_run")
        self.assertEqual(rows[0].state, "RUNNING")
        self.assertEqual(rows[0].elapsed, "12:03")
        self.assertEqual(rows[0].nodes, "1")
        self.assertEqual(rows[0].reason, "gpu-a01")
        self.assertEqual(rows[0].workdir, "/runs/chignolin/chignolin_2d_run")
        self.assertEqual(rows[1].workdir, "")

    def test_match_slurm_jobs_prefers_workdir_and_falls_back_to_job_name(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            chig = PeptideState.from_rundir(root / "chignolin" / "chignolin_2d_run")
            dyk = PeptideState.from_rundir(root / "DYKDDDDK" / "DYKDDDDK_2d_run")
            states = [chig, dyk]
            jobs = parse_squeue_rows(
                f"12345|some_wrapper|RUNNING|12:03|1|gpu-a01|{chig.run_dir}\n"
                "12346|DYKDDDDK-prod|PENDING|0:00|1|Priority|\n"
                "99999|unrelated|RUNNING|1:00|1|node-x|\n"
            )

            unmatched = match_slurm_jobs_to_states(states, jobs, fleet_roots=[root])

            self.assertEqual([j.job_id for j in chig._slurm_jobs], ["12345"])
            self.assertEqual([j.job_id for j in dyk._slurm_jobs], ["12346"])
            self.assertEqual([j.job_id for j in unmatched], [])

    def test_match_slurm_jobs_keeps_chignolin_tica_separate_from_chignolin(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            chig = PeptideState.from_rundir(root / "chignolin" / "chignolin_2d_run")
            tica = PeptideState.from_rundir(root / "chignolin_tica" / "chignolin_tica_2d_run")
            states = [chig, tica]
            jobs = parse_squeue_rows(
                "12345|chignolin-prod|RUNNING|12:03|1|gpu-a01|\n"
                "12346|chignolin_tica-prod|RUNNING|12:04|1|gpu-a02|\n"
            )

            unmatched = match_slurm_jobs_to_states(states, jobs, fleet_roots=[root])

            self.assertEqual([j.job_id for j in chig._slurm_jobs], ["12345"])
            self.assertEqual([j.job_id for j in tica._slurm_jobs], ["12346"])
            self.assertEqual([j.job_id for j in unmatched], [])

    def test_run_snapshot_includes_matched_slurm_jobs(self):
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "PEP_2d_run"
            state = PeptideState.from_rundir(run_dir)
            jobs = parse_squeue_rows(f"222|PEP_2d_run|RUNNING|1:00|1|node-a|{run_dir}\n")
            match_slurm_jobs_to_states([state], jobs, fleet_roots=[Path(td)])

            snap = build_run_snapshot(state)

            self.assertEqual([j.job_id for j in snap.slurm_jobs], ["222"])


class GaMDBoostSummaryTests(unittest.TestCase):
    def test_parse_distances_samples_preserves_explicit_window_fields(self):
        samples = parse_distances_samples([
            {
                "event": "distances",
                "distances": [
                    {
                        "primary_cv_value": 0.1,
                        "secondary_cv": 0.2,
                        "gamd_boost_total_kcal_mol": 1.0,
                        "window": 7,
                        "replica": 3,
                    },
                ],
            },
        ])

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["window"], 7)
        self.assertEqual(samples[0]["replica"], 3)

    def test_group_boost_values_by_window_uses_only_explicit_ids(self):
        groups = group_boost_values_by_window([
            {"boost": 1.0, "window": 1},
            {"boost": 2.0, "window": 1},
            {"boost": 4.0, "state_id": 9},
            {"boost": 5.0},
            {"boost": None, "window": 2},
        ])

        self.assertEqual(groups, [("W1", [1.0, 2.0])])

    def test_group_boost_values_by_window_ignores_replica_only_samples(self):
        groups = group_boost_values_by_window([
            {"boost": 3.0, "replica": 7},
            {"boost": 4.0, "replica": 7, "window": None},
            {"boost": 5.0, "window": 2},
        ])

        self.assertEqual(groups, [("W2", [5.0])])

    def test_default_per_window_limit_is_at_least_two_thousand(self):
        self.assertGreaterEqual(BOOST_TARGET_PER_WINDOW, 2000)

    def test_group_boost_values_by_window_keeps_up_to_two_thousand_samples(self):
        samples = [{"boost": float(i), "window": 1} for i in range(2500)]

        groups = group_boost_values_by_window(samples)

        self.assertEqual(len(groups), 1)
        label, vals = groups[0]
        self.assertEqual(label, "W1")
        # keeps the target depth, dropping only the oldest overflow
        self.assertEqual(len(vals), BOOST_TARGET_PER_WINDOW)
        self.assertEqual(vals[-1], 2499.0)

    def test_group_boost_values_by_window_keeps_all_when_under_target(self):
        samples = [{"boost": float(i), "window": 1} for i in range(500)]

        groups = group_boost_values_by_window(samples)

        self.assertEqual(len(groups[0][1]), 500)

    def test_min_boost_samples_per_window_reports_shallowest_window(self):
        samples = (
            [{"boost": 1.0, "window": 1}] * 5
            + [{"boost": 2.0, "window": 2}] * 3
            + [{"boost": None, "window": 3}]
        )

        self.assertEqual(_min_boost_samples_per_window(samples), 3)

    def test_min_boost_samples_per_window_zero_without_explicit_ids(self):
        self.assertEqual(_min_boost_samples_per_window([{"boost": 1.0}]), 0)

    def test_read_boost_samples_reads_all_distances_from_small_file(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "progress.jsonl"
            events = [
                {
                    "event": "distances",
                    "distances": [
                        {"primary_cv_value": 0.1, "secondary_cv": 0.2,
                         "gamd_boost_total_kcal_mol": float(i), "window": 1},
                    ],
                }
                for i in range(700)
            ]
            _append_jsonl(path, *events)

            samples = read_boost_samples(path)

            self.assertEqual(len(samples), 700)
            self.assertEqual(_min_boost_samples_per_window(samples), 700)

    def test_live_boost_window_view_keeps_more_than_old_four_hundred_cap(self):
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "PEP_2d_run"
            events = [
                {
                    "event": "distances",
                    "distances": [
                        {"primary_cv_value": 0.1, "secondary_cv": 0.2,
                         "gamd_boost_total_kcal_mol": float(i), "window": 1},
                    ],
                }
                for i in range(700)
            ]
            progress = {
                "event": "progress",
                "phase": "gareus_production",
                "percent": 50.0,
                "aggregate_sim_time_ns": 4.0,
                "wall_time_s": 2000.0,
            }
            # progress lines interleave with distances in real runs; the tail
            # gate scans only recent entries, so keep one near the end too
            _append_jsonl(run_dir / "progress.jsonl", progress, *events, progress)
            state = PeptideState.from_rundir(run_dir)
            state.refresh()

            view = _live_boost_window_view(state)

            self.assertIsNotNone(view)
            self.assertTrue(view["has_explicit_ids"])
            groups = dict(view["groups"])
            self.assertIn("W1", groups)
            self.assertEqual(len(groups["W1"]), 700)

    def test_live_boost_values_aggregate_reads_beyond_old_four_thousand_cap(self):
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "PEP_2d_run"
            events = [
                {
                    "event": "distances",
                    "distances": [
                        {"primary_cv_value": 0.1, "secondary_cv": 0.2,
                         "gamd_boost_total_kcal_mol": float(i)},
                    ],
                }
                for i in range(4200)
            ]
            progress = {
                "event": "progress",
                "phase": "gareus_production",
                "percent": 50.0,
                "aggregate_sim_time_ns": 4.0,
                "wall_time_s": 2000.0,
            }
            _append_jsonl(run_dir / "progress.jsonl", progress, *events, progress)
            state = PeptideState.from_rundir(run_dir)
            state.refresh()

            self.assertEqual(len(state.live_boost_values()), 4200)

    def test_peptide_state_live_boost_values_reads_progress_tail(self):
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "PEP_2d_run"
            _append_jsonl(
                run_dir / "progress.jsonl",
                {
                    "event": "progress",
                    "phase": "gareus_production",
                    "percent": 50.0,
                    "aggregate_sim_time_ns": 12.5,
                    "wall_time_s": 1000.0,
                },
                {
                    "event": "distances",
                    "distances": [
                        {"primary_cv_value": 0.1, "secondary_cv": 0.2, "gamd_boost_total_kcal_mol": 1.0},
                        {"primary_cv_value": 0.3, "secondary_cv": 0.4, "gamd_boost_total_kcal_mol": 2.0},
                    ],
                },
                {
                    "event": "distances",
                    "distances": [
                        {"primary_cv_value": 0.5, "secondary_cv": 0.6, "gamd_boost_total_kcal_mol": 3.0},
                    ],
                },
            )
            state = PeptideState.from_rundir(run_dir)

            self.assertEqual(state.live_boost_values(2), [2.0, 3.0])

    def test_peptide_state_live_boost_values_returns_empty_after_terminal_progress(self):
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "PEP_2d_run"
            _append_jsonl(
                run_dir / "progress.jsonl",
                {
                    "event": "progress",
                    "phase": "gareus_production",
                    "percent": 50.0,
                    "aggregate_sim_time_ns": 12.5,
                    "wall_time_s": 1000.0,
                },
                {
                    "event": "distances",
                    "distances": [
                        {"primary_cv_value": 0.1, "secondary_cv": 0.2, "gamd_boost_total_kcal_mol": 1.0},
                    ],
                },
                {"event": "run_complete", "wall_time_s": 1001.0},
            )
            state = PeptideState.from_rundir(run_dir)

            self.assertEqual(state.live_boost_values(), [])

    def test_boost_values_from_entries_keeps_recent_distances_values_only(self):
        entries = [
            {"event": "progress", "gamd_boost_mean_kcal_mol": 3.0},
            {
                "event": "distances",
                "distances": [
                    {"primary_cv_value": 1.0, "secondary_cv": 2.0, "gamd_boost_total_kcal_mol": 1.5},
                    {"primary_cv_value": 1.1, "secondary_cv": 2.1, "gamd_boost_total_kcal_mol": 2.5},
                ],
            },
            {
                "event": "distances",
                "distances": [
                    {"primary_cv_value": 1.2, "secondary_cv": 2.2, "gamd_boost_total_kcal_mol": 3.5},
                ],
            },
        ]

        self.assertEqual(boost_values_from_entries(entries, limit=2), [2.5, 3.5])

    def test_boost_values_from_entries_skips_missing_boost_samples_instead_of_treating_them_as_zero(self):
        entries = [
            {
                "event": "distances",
                "distances": [
                    {"primary_cv_value": 1.0, "secondary_cv": 2.0},
                    {"primary_cv_value": 1.1, "secondary_cv": 2.1, "gamd_boost_total_kcal_mol": 0.0},
                    {"primary_cv_value": 1.2, "secondary_cv": 2.2, "gamd_boost_total_kcal_mol": 1.5},
                ],
            },
        ]

        self.assertEqual(boost_values_from_entries(entries), [0.0, 1.5])

    def test_summarize_boost_values_reports_percentiles_and_bounds(self):
        stats = summarize_boost_values([1.0, 2.0, 3.0, 4.0, 5.0])

        self.assertEqual(stats["n"], 5)
        self.assertEqual(stats["min"], 1.0)
        self.assertEqual(stats["p10"], 1.0)
        self.assertEqual(stats["p50"], 3.0)
        self.assertEqual(stats["p90"], 5.0)
        self.assertEqual(stats["max"], 5.0)

    def test_summarize_boost_values_returns_empty_summary_when_no_samples_exist(self):
        self.assertEqual(summarize_boost_values([]), {})

    def test_render_detail_shows_gamd_boost_distribution_when_live_samples_exist(self):
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "PEP_2d_run"
            _append_jsonl(
                run_dir / "progress.jsonl",
                {
                    "event": "progress",
                    "phase": "gareus_production",
                    "percent": 50.0,
                    "aggregate_sim_time_ns": 4.0,
                    "gamd_boost_mean_kcal_mol": 2.5,
                    "gamd_boost_sd_kcal_mol": 0.8,
                    "gamd_boost_anharmonicity_score": 0.1,
                    "wall_time_s": 2000.0,
                },
                {
                    "event": "distances",
                    "distances": [
                        {"primary_cv_value": 0.1, "secondary_cv": 0.2, "gamd_boost_total_kcal_mol": 1.0},
                        {"primary_cv_value": 0.3, "secondary_cv": 0.4, "gamd_boost_total_kcal_mol": 2.0},
                        {"primary_cv_value": 0.5, "secondary_cv": 0.6, "gamd_boost_total_kcal_mol": 4.0},
                    ],
                },
            )
            state = PeptideState.from_rundir(run_dir)
            state.refresh()

            out = render_detail(state)

            self.assertIn("GaMD boost dist", out)
            self.assertIn("p10=", out)
            self.assertIn("p50=", out)
            self.assertIn("p90=", out)

    def test_render_detail_shows_per_window_boost_rows_when_explicit_ids_exist(self):
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "PEP_2d_run"
            _append_jsonl(
                run_dir / "progress.jsonl",
                {
                    "event": "progress",
                    "phase": "gareus_production",
                    "percent": 50.0,
                    "aggregate_sim_time_ns": 4.0,
                    "wall_time_s": 2000.0,
                },
                {
                    "event": "distances",
                    "distances": [
                        {"primary_cv_value": 0.1, "secondary_cv": 0.2, "gamd_boost_total_kcal_mol": 1.0, "window": 1},
                        {"primary_cv_value": 0.3, "secondary_cv": 0.4, "gamd_boost_total_kcal_mol": 2.0, "window": 1},
                        {"primary_cv_value": 0.5, "secondary_cv": 0.6, "gamd_boost_total_kcal_mol": 4.0, "window": 2},
                    ],
                },
            )
            state = PeptideState.from_rundir(run_dir)
            state.refresh()

            out = render_detail(state)

            self.assertIn("per-window boost dist", out)
            self.assertIn("W1", out)
            self.assertIn("W2", out)

    def test_render_detail_reports_unavailable_when_no_explicit_ids_exist(self):
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "PEP_2d_run"
            _append_jsonl(
                run_dir / "progress.jsonl",
                {
                    "event": "progress",
                    "phase": "gareus_production",
                    "percent": 50.0,
                    "aggregate_sim_time_ns": 4.0,
                    "wall_time_s": 2000.0,
                },
                {
                    "event": "distances",
                    "distances": [
                        {"primary_cv_value": 0.1, "secondary_cv": 0.2, "gamd_boost_total_kcal_mol": 1.0},
                    ],
                },
            )
            state = PeptideState.from_rundir(run_dir)
            state.refresh()

            out = render_detail(state)

            self.assertIn("per-window boost dist", out)
            self.assertIn("no explicit window ids", out)
            self.assertNotIn("explicit window ids present but no usable boost values", out)

    def test_render_detail_reports_unavailable_when_explicit_ids_have_no_usable_boost_values(self):
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "PEP_2d_run"
            _append_jsonl(
                run_dir / "progress.jsonl",
                {
                    "event": "progress",
                    "phase": "gareus_production",
                    "percent": 50.0,
                    "aggregate_sim_time_ns": 4.0,
                    "wall_time_s": 2000.0,
                },
                {
                    "event": "distances",
                    "distances": [
                        {"primary_cv_value": 0.1, "secondary_cv": 0.2, "gamd_boost_total_kcal_mol": None, "window": 7},
                        {"primary_cv_value": 0.3, "secondary_cv": 0.4, "gamd_boost_total_kcal_mol": float("nan"), "window": 7},
                    ],
                },
            )
            state = PeptideState.from_rundir(run_dir)
            state.refresh()

            out = render_detail(state)

            self.assertIn("per-window boost dist", out)
            self.assertIn("explicit window ids present but no usable boost values", out)
            self.assertNotIn("no explicit window ids", out)

    def test_render_detail_hides_per_window_section_after_terminal_progress(self):
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "PEP_2d_run"
            _append_jsonl(
                run_dir / "progress.jsonl",
                {
                    "event": "progress",
                    "phase": "gareus_production",
                    "percent": 100.0,
                    "aggregate_sim_time_ns": 4.0,
                    "wall_time_s": 2000.0,
                },
                {
                    "event": "distances",
                    "distances": [
                        {"primary_cv_value": 0.1, "secondary_cv": 0.2, "gamd_boost_total_kcal_mol": 1.0, "window": 1},
                    ],
                },
                {"event": "run_complete", "wall_time_s": 2001.0},
            )
            state = PeptideState.from_rundir(run_dir)
            state.refresh()

            out = render_detail(state)

            self.assertNotIn("per-window boost dist", out)
            self.assertNotIn("W1", out)

    def test_render_detail_reports_no_live_boost_samples_when_absent(self):
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "PEP_2d_run"
            _append_jsonl(
                run_dir / "progress.jsonl",
                {
                    "event": "progress",
                    "phase": "gareus_production",
                    "percent": 50.0,
                    "aggregate_sim_time_ns": 4.0,
                    "gamd_boost_mean_kcal_mol": 2.5,
                    "gamd_boost_sd_kcal_mol": 0.8,
                    "gamd_boost_anharmonicity_score": 0.1,
                    "wall_time_s": 2000.0,
                },
            )
            state = PeptideState.from_rundir(run_dir)
            state.refresh()

            out = render_detail(state)

            self.assertIn("GaMD boost dist", out)
            self.assertIn("no live boost samples", out)

    @unittest.skipUnless(_HAS_RICH, "rich not installed")
    def test_rich_detail_panel_shows_per_window_boost_section_when_ids_exist(self):
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "PEP_2d_run"
            _append_jsonl(
                run_dir / "progress.jsonl",
                {
                    "event": "progress",
                    "phase": "gareus_production",
                    "percent": 50.0,
                    "aggregate_sim_time_ns": 4.0,
                    "wall_time_s": time.time(),
                },
                {
                    "event": "distances",
                    "distances": [
                        {"primary_cv_value": 0.1, "secondary_cv": 0.2, "gamd_boost_total_kcal_mol": 1.0, "window": 1},
                        {"primary_cv_value": 0.3, "secondary_cv": 0.4, "gamd_boost_total_kcal_mol": 2.0, "window": 2},
                    ],
                },
            )
            state = PeptideState.from_rundir(run_dir)
            state.refresh()
            panel = _rich_detail_panel(state, build_run_snapshot(state))
            rendered = _render_rich(panel)

            self.assertIn("per-window boost dist", rendered)
            self.assertIn("W1", rendered)
            self.assertIn("W2", rendered)

    @unittest.skipUnless(_HAS_RICH, "rich not installed")
    def test_rich_detail_panel_shows_unavailable_note_without_ids(self):
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "PEP_2d_run"
            _append_jsonl(
                run_dir / "progress.jsonl",
                {
                    "event": "progress",
                    "phase": "gareus_production",
                    "percent": 50.0,
                    "aggregate_sim_time_ns": 4.0,
                    "wall_time_s": time.time(),
                },
                {
                    "event": "distances",
                    "distances": [
                        {"primary_cv_value": 0.1, "secondary_cv": 0.2, "gamd_boost_total_kcal_mol": 1.0},
                    ],
                },
            )
            state = PeptideState.from_rundir(run_dir)
            state.refresh()
            panel = _rich_detail_panel(state, build_run_snapshot(state))
            rendered = _render_rich(panel)

            self.assertIn("per-window boost dist", rendered)
            self.assertIn("no explicit window ids", rendered)


class DiscoveryTests(unittest.TestCase):
    def test_discover_keeps_chignolin_tica_autoswitch_yaml_as_distinct_run(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "chignolin").mkdir()
            (root / "chignolin" / "chignolin.yaml").write_text("md_budget_ns: 10\n")
            (root / "chignolin_tica").mkdir()
            (root / "chignolin_tica" / "chignolin_tica_autoswitch.yaml").write_text("md_budget_ns: 20\n")

            states = discover(root)

            self.assertEqual([s.name for s in states], ["chignolin", "chignolin_tica"])
            self.assertEqual(_plan_yaml_path(states[1]).name, "chignolin_tica_autoswitch.yaml")

    @staticmethod
    def _mkrun(d: Path, markers):
        d.mkdir(parents=True, exist_ok=True)
        for name in markers:
            (d / name).write_text("{}")
        return d

    def test_discovers_descriptively_named_run(self):
        """A run dir not matching PEPTIDE_2d_run<N> is still resolved by markers."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            foo = root / "foo"
            foo.mkdir()
            (foo / "foo.yaml").write_text("md_budget_ns: 10\n")
            # Early/hung run: only setup artifacts written, no progress yet.
            self._mkrun(foo / "foo_fullrun",
                        ("effective_config.json", "run_manifest.json", "00_built_peptide.pdb"))

            states = {s.name: Path(s.run_dir).name for s in discover(root)}
            self.assertEqual(states.get("foo"), "foo_fullrun")

    def test_config_and_genpept_siblings_not_mistaken_for_run(self):
        """A run's config/ provenance subdir and the genpept sibling never win."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            foo = root / "foo"
            foo.mkdir()
            (foo / "foo.yaml").write_text("md_budget_ns: 10\n")
            run = self._mkrun(foo / "foo_fullrun", ("effective_config.json", "progress.jsonl"))
            self._mkrun(run / "config", ("effective_config.json", "run_manifest.json"))
            (foo / "foo_genpept").mkdir()
            (foo / "foo_genpept" / "candidate_seeds").mkdir()

            from gareus_monitor import _find_container_rundir
            self.assertEqual(_find_container_rundir(foo, "foo").name, "foo_fullrun")
            # Pointing directly at the container lists the run, never config/.
            sub = {s.run_dir.name for s in discover(foo)}
            self.assertIn("foo_fullrun", sub)
            self.assertNotIn("config", sub)

    def test_active_run_preferred_over_legacy_2d_run(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            foo = root / "foo"
            foo.mkdir()
            (foo / "foo.yaml").write_text("md_budget_ns: 10\n")
            self._mkrun(foo / "foo_2d_run2", ("effective_config.json",))       # legacy, no progress
            self._mkrun(foo / "foo_fullrun", ("effective_config.json", "progress.jsonl"))  # active

            states = {s.name: Path(s.run_dir).name for s in discover(root)}
            self.assertEqual(states.get("foo"), "foo_fullrun")

    def test_collection_dir_not_collapsed_into_single_state(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            coll = root / "v09_runs"
            coll.mkdir()
            bar = coll / "bar"
            bar.mkdir()
            (bar / "bar.yaml").write_text("md_budget_ns: 10\n")
            self._mkrun(bar / "bar_prodrun", ("progress.jsonl",))

            names = [s.name for s in discover(root)]
            self.assertNotIn("v09_runs", names)

    def test_empty_container_uses_legacy_fallback_path(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            emp = root / "emp"
            emp.mkdir()
            (emp / "emp.yaml").write_text("md_budget_ns: 10\n")
            from gareus_monitor import _find_container_rundir
            self.assertEqual(_find_container_rundir(emp, "emp").name, "emp_2d_run")


class SourceCompatibilityTests(unittest.TestCase):
    def test_postpones_annotations_for_python39_generic_alias_unions(self):
        source = Path(__file__).with_name("gareus_monitor.py").read_text()

        self.assertIn("tuple[SlurmJob, ...] | list[SlurmJob]", source)
        self.assertIn(
            "from __future__ import annotations",
            source.splitlines()[:30],
            "Python 3.9 evaluates tuple[...] | list[...] annotations at import time.",
        )


if __name__ == "__main__":
    unittest.main()
