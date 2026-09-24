"""`--no-ap-topups`: a scheduled phase runs its all-state baseline only.

On a lambda-ladder campaign the scheduled top-ups cost more than they buy
(chignolin_9 epoch_001, 2026-09-24): the "weak edges" driving them were almost
all unmeasured rung edges, every one of 236 states got extra steps (3x the
baseline in total), and each top-up batch held a single rung, so no lambda
exchange was possible inside it, and each re-pulled its windows from scratch.
With top-ups off the baseline carries the phase's whole per-state budget (the
mean of the allocator's requested steps), so the same MD is spent on every
state together, continuing from the baseline's own checkpoint.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import gareus.adaptive_production as ap
import gareus.production as prod
from gareus.lifecycle import _graceful_shutdown

from test_scheduled_final_interruption import _scheduled_final_campaign


@pytest.fixture(autouse=True)
def _clear_shutdown():
    _graceful_shutdown.clear()
    yield
    _graceful_shutdown.clear()


def _drive(monkeypatch, args, out):
    calls = []

    def _fake_run_gareus(seg_args, run_dir, *rest, **kw):
        calls.append((Path(run_dir).name, int(seg_args.gamd_production_steps)))

    monkeypatch.setattr(prod, "run_gareus", _fake_run_gareus)
    ap.run_adaptive_production_auto_loop(args, out, None, None, None, None, None, None)
    return calls


def test_no_topups_runs_only_the_baseline_at_the_mean_requested_steps(tmp_path, monkeypatch):
    args, out = _scheduled_final_campaign(tmp_path)
    args.adaptive_production_topups = False
    calls = _drive(monkeypatch, args, out)

    final_calls = [c for c in calls if c[0] in ("baseline",) or c[0].startswith("topup_")]
    assert final_calls and all(name == "baseline" for name, _ in final_calls), calls
    rows = json.loads((Path(out) / "adaptive_production" / "final" / "epoch_schedule.json").read_text())["rows"]
    requested = [int(r["requested_steps"]) for r in rows if int(r.get("requested_steps", 0) or 0) > 0]
    mean_q = ap._quantized_extra_steps(int(round(sum(requested) / len(requested))))
    assert final_calls[-1][1] == max(min(int(r["baseline_steps"]) for r in rows
                                         if int(r.get("requested_steps", 0) or 0) > 0), mean_q)
    assert not list((Path(out) / "adaptive_production" / "final").glob("topup_*"))


def test_the_flag_parses_and_defaults_on():
    from gareus.cli import build_gareus_parser

    p = build_gareus_parser()
    assert p.parse_args(["--seq", "AA"]).ap_topups is True
    assert p.parse_args(["--seq", "AA", "--no-ap-topups"]).ap_topups is False
