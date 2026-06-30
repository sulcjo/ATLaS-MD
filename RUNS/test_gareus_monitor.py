import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gareus_monitor import PeptideState


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")


def _append_jsonl(path: Path, *payloads: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for payload in payloads:
            f.write(json.dumps(payload) + "\n")


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


if __name__ == "__main__":
    unittest.main()
