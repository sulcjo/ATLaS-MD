"""Wide swarm concurrency (174 member threads, 2026-10-02 job 2788266): grafts gated, coarse snapshots."""
from __future__ import annotations

import inspect
import threading
import time

from gareus.swarm import members as M


def test_graft_is_gated_and_crash_snapshots_are_per_frame():
    src = inspect.getsource(M.run_member)
    assert "with _GRAFT_GATE:" in src
    assert src.index("with _GRAFT_GATE:") < src.index("graft_conformer_into_context(")
    assert "chunk_size=max(100, int(steps_per_frame))" in src


def test_graft_gate_bounds_concurrency_at_many_threads():
    active, peak, lock = [0], [0], threading.Lock()

    def member():
        with M._GRAFT_GATE:
            with lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            time.sleep(0.01)
            with lock:
                active[0] -= 1

    threads = [threading.Thread(target=member) for _ in range(60)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert peak[0] == M.GRAFT_CONCURRENCY
