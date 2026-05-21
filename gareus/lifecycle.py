"""
Lifecycle helpers for graceful shutdown.

In high‑performance computing (HPC) environments, jobs are often
terminated via signals (e.g. ``SIGTERM``).  This module encapsulates
the global shutdown event used by the production loop and provides a
function to register a signal handler that sets that event.
"""

from __future__ import annotations

import threading
import signal
from typing import Optional, Any

__all__ = ["_graceful_shutdown", "_register_graceful_shutdown"]

# Module‑level event used to indicate that a graceful shutdown has been requested.
_graceful_shutdown: threading.Event = threading.Event()


def _register_graceful_shutdown() -> None:
    """Register a ``SIGTERM`` handler for clean HPC wall‑time exits.

    When a ``SIGTERM`` is received, the handler sets
    :data:`_graceful_shutdown`.  The production loop polls this event at
    the top of each exchange‑interval iteration; when set, it writes a
    checkpoint and exits gracefully.
    """
    def _handler(signum: int, frame: Optional[Any]) -> None:  # noqa: ARG001
        _graceful_shutdown.set()
        # Immediately inform any log‑tailing user what happened.
        try:
            print("\nSIGTERM received — will checkpoint and exit after the current MD chunk.", flush=True)
        except Exception:
            pass
    try:
        signal.signal(signal.SIGTERM, _handler)
    except (OSError, ValueError):
        # Either not running in the main thread or the platform does not support SIGTERM.
        pass