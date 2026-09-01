"""An interrupted scheduled final must not be reported as a completed campaign.

Found on `RUNS/chignolin_6`. Its `final/scheduled_epoch_summary.json` says
`status: interrupted_after_checkpoint` -- the phase ran 2,613,200 of the
15,201,246 steps per state it had scheduled, 17%, and was cut short. The
campaign's own `adaptive_production_driver_summary.json` says
`status: completed`, and the quality gate underneath it says
`needs_more_sampling` with one state holding zero samples.

`run_scheduled_adaptive_epoch` reports the interruption correctly. The defect is
one branch in its caller: of the three sites that run a phase, only the scheduled
final fails to check.

    epoch loop            adaptive_production.py:6189   checks, returns
    non-scheduled final   adaptive_production.py:6840   checks, returns
    scheduled final       adaptive_production.py:6779   DOES NOT CHECK

so execution falls through the quality gate and the union-MBAR build to the
unconditional `"status": "completed"` at the end of the driver.

The second defect is in the same place. A completed scheduled final books its MD
to the runtime pool; an interrupted one books nothing, because the interrupted
return happens before the consume. chignolin_6's ledger therefore shows
1000.86 ns used of 10000 and no `final` event at all, while the final actually
burned roughly 376 ns -- so "90% of the pool is left" is an overstatement of what
is really available, and a resumed run would over-allocate against it.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import gareus.adaptive_production as ap
import gareus.production as prod
from gareus.lifecycle import _graceful_shutdown

from test_epoch_window_map_rewrite_after_drop import _five_state_resumed_campaign


@pytest.fixture(autouse=True)
def _clear_shutdown():
    """The event is module-global; a leaked set() would poison every later test."""
    _graceful_shutdown.clear()
    yield
    _graceful_shutdown.clear()


def _scheduled_final_campaign(tmp_path):
    """A campaign whose next action is the SCHEDULED final phase.

    The driver summary records epoch 0 as done, so `start_epoch` is 1; setting
    `adaptive_production_epochs = 1` empties `range(1, 1)` and the epoch loop is
    skipped, putting the final phase first in line. Unlike the frozen-final
    fixtures elsewhere in the suite, the scheduler is left ON so the phase goes
    through `run_scheduled_adaptive_epoch`.
    """
    args, out, _epoch = _five_state_resumed_campaign(tmp_path)
    args.adaptive_production_epochs = 1
    args.adaptive_production_final_allocation_scheduler = True
    args.adaptive_production_scheduled_final_segments = True
    args.adaptive_production_total_md_pool_ns = 100.0
    return args, out


def _drive_interrupting_the_final(monkeypatch, args, out):
    """Run the real driver; the final phase's first segment is cut short.

    The fake `run_gareus` writes nothing and sets the shutdown event, which is
    exactly what a SIGTERM between checkpoints looks like to the driver: the
    segment returns normally and the interruption is discovered afterwards.
    """
    seen = {}

    def _fake_run_gareus(_args, run_dir, *rest, **kw):
        seen.setdefault("dirs", []).append(str(run_dir))
        _graceful_shutdown.set()

    monkeypatch.setattr(prod, "run_gareus", _fake_run_gareus)
    result = ap.run_adaptive_production_auto_loop(
        args, out, None, None, None, None, None, None
    )
    return result, seen


def test_an_interrupted_scheduled_final_is_not_reported_as_completed(tmp_path, monkeypatch):
    args, out = _scheduled_final_campaign(tmp_path)
    result, seen = _drive_interrupting_the_final(monkeypatch, args, out)

    assert seen.get("dirs"), "the drive never reached the final phase at all"
    assert "final" in seen["dirs"][-1], f"expected a final phase, got {seen['dirs']}"
    assert result.get("status") == "interrupted_after_checkpoint", (
        f"driver reported {result.get('status')!r} for a final phase that was cut "
        "short. A campaign that stopped mid-final must not claim completion -- on "
        "the real run this hid a final that had done 17% of its schedule."
    )


def test_the_interruption_reaches_the_driver_summary_on_disk(tmp_path, monkeypatch):
    """The JSON is what a human or a resume reads, not the return value."""
    args, out = _scheduled_final_campaign(tmp_path)
    _drive_interrupting_the_final(monkeypatch, args, out)

    summary = json.loads(
        (Path(out) / "adaptive_production" / "adaptive_production_driver_summary.json").read_text()
    )
    assert summary.get("status") == "interrupted_after_checkpoint", (
        f"on-disk driver summary says {summary.get('status')!r}"
    )
    assert "final" in str(summary.get("interrupted_segment", "")), (
        "the summary must name the final phase as the interrupted segment"
    )


def test_an_interrupted_final_still_books_its_md_to_the_runtime_pool(tmp_path, monkeypatch):
    """Steps that were actually run must be charged, interrupted or not.

    Otherwise the pool reports budget that has already been spent, and a resumed
    campaign allocates against a figure that is too large. On chignolin_6 the
    ledger claimed 8999 ns remaining while the final had already burned ~376.
    """
    args, out = _scheduled_final_campaign(tmp_path)
    _drive_interrupting_the_final(monkeypatch, args, out)

    pool = json.loads(
        (Path(out) / "adaptive_production" / "adaptive_runtime_pool.json").read_text()
    )
    labels = [e.get("label") for e in pool.get("events", [])]
    assert any("final" in str(lbl) for lbl in labels), (
        f"runtime pool has no final event; recorded {labels}. The interrupted "
        "final's MD was run but never charged to the budget."
    )
    assert float(pool.get("used_ns") or 0.0) > 0.0, "pool records no usage at all"
