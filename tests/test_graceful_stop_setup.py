"""A SIGTERM during a setup phase (US pull) stops the process within one chunk.

chignolin_10 job 2795169 (2026-10-03): SIGTERM arrived 40 min into a 212-window pull,
the pull ignored it, SLURM killed the job at the walltime and the chain never resubmitted.
"""
from __future__ import annotations

import concurrent.futures
import threading
import time
import types

import pytest

from gareus import lifecycle
from gareus.lifecycle import GracefulStop, _graceful_shutdown
from gareus.seeding import _completed_or_cancel_on_stop
from gareus.system_setup import run_steps_safely


@pytest.fixture(autouse=True)
def _clear_flag():
    _graceful_shutdown.clear()
    yield
    _graceful_shutdown.clear()


class _FakeState:
    def getPositions(self):
        return []


class _FakeSim:
    def __init__(self, stop_after=None):
        self.steps = 0
        self.stop_after = stop_after
        self.context = types.SimpleNamespace(getState=lambda **kw: _FakeState())

    def step(self, n):
        self.steps += n
        if self.stop_after is not None and self.steps >= self.stop_after:
            _graceful_shutdown.set()


def _run(sim, stop_check):
    run_steps_safely(sim, 1000, "us_starting_pull", None, None, None, None, chunk_size=100,
                     stop_check=stop_check)


def test_stop_check_raises_at_the_next_chunk():
    sim = _FakeSim(stop_after=300)
    with pytest.raises(GracefulStop):
        _run(sim, stop_check=True)
    assert sim.steps == 300


def test_without_stop_check_the_segment_runs_to_the_end():
    sim = _FakeSim(stop_after=300)
    _run(sim, stop_check=False)
    assert sim.steps == 1000


def test_graceful_stop_passes_through_except_exception_recovery():
    def recovering():
        try:
            lifecycle.raise_if_stop_requested("pull")
        except Exception:  # the pull's crash-recovery shape
            return "recovered"
        return "ran"

    _graceful_shutdown.set()
    with pytest.raises(GracefulStop):
        recovering()


def test_concurrent_pull_cancels_queued_windows_on_stop():
    started = []
    lock = threading.Lock()

    def task(w):
        lifecycle.raise_if_stop_requested(f"window {w}")
        with lock:
            started.append(w)
        if w == 3:
            _graceful_shutdown.set()
        time.sleep(0.01)
        return w

    with pytest.raises(GracefulStop):
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            futures = {ex.submit(task, w): w for w in range(200)}
            for fut in _completed_or_cancel_on_stop(futures):
                fut.result()
    assert len(started) < 10                       # 200 queued, almost none did work
    # the rest were cancelled or exited at their entry check -- none still pending
    assert all(f.cancelled() or isinstance(f.exception(), GracefulStop) or f.result() in started
               for f in futures)


def test_core_main_exits_cleanly_on_graceful_stop(monkeypatch, capsys):
    import gareus.cli
    from gareus import core

    def fake_main(argv=None):
        raise GracefulStop("US pull before window 90/212")

    monkeypatch.setattr(gareus.cli, "main", fake_main)
    assert core.main([]) is None
    assert "stopped on request" in capsys.readouterr().out
