"""Best-effort process memory checkpoints for long analyses."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional


class MemoryReporter:
    def __init__(self, path: Optional[Path]):
        self.path = Path(path) if path else None
        self._fh = None
        if self.path is not None:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._fh = self.path.open('a', encoding='utf-8')
            except OSError:
                self._fh = None

    def _read_proc(self) -> dict:
        rss = 0
        with Path('/proc/self/status').open(encoding='utf-8') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    rss = int(line.split()[1]) * 1024
                    break
        available = None
        with Path('/proc/meminfo').open(encoding='utf-8') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    available = int(line.split()[1]) * 1024
                    break
        swap_free = swap_total = 0
        with Path('/proc/meminfo').open(encoding='utf-8') as f:
            for line in f:
                if line.startswith('SwapTotal:'):
                    swap_total = int(line.split()[1]) * 1024
                elif line.startswith('SwapFree:'):
                    swap_free = int(line.split()[1]) * 1024
        return {'rss_bytes': rss, 'mem_available_bytes': available,
                'swap_total_bytes': swap_total, 'swap_free_bytes': swap_free}

    def record(self, phase: str) -> None:
        if self._fh is None:
            return
        try:
            row = {'timestamp': time.time(), 'phase': str(phase), 'pid': os.getpid()}
            row.update(self._read_proc())
            self._fh.write(json.dumps(row, sort_keys=True) + '\n')
            self._fh.flush()
        except Exception:
            # Diagnostics must never change analysis behavior.
            return

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            except OSError:
                pass
            self._fh = None
