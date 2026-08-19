"""
I/O helper routines.

This module bundles together simple JSON and CSV utilities originally
implemented in ``gareus_peptide.py``.  It includes a JSON encoder
capable of serialising NumPy types, convenience functions to read and
write JSON files, and buffered writers for CSV and JSONL output.
"""

from __future__ import annotations

import atexit
import json
import csv
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np


class _NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that converts NumPy types into Python primitives."""

    def default(self, o: Any) -> Any:  # type: ignore[override]
        if isinstance(o, Path):
            return str(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        return super().default(o)


def write_json(path: Path, payload: Any) -> None:
    """Write a JSON payload to a file.

    The directory is created if necessary and NumPy types are converted
    transparently via :class:`_NumpyEncoder`. The write is atomic: the
    payload is written to a temporary file in the same directory and then
    renamed onto the final path, so a process killed mid-write cannot leave
    a torn/partial file behind.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, cls=_NumpyEncoder), encoding="utf-8")
    tmp.replace(path)


def read_json_file(path: Path, default: Optional[Any] = None) -> Any:
    """Read a JSON file if it exists; return ``default`` on error or missing file.

    A missing file is the normal case for a fresh campaign and is handled
    silently. A file that exists but fails to parse (corrupt/truncated
    JSON) is a recoverable-but-suspicious situation, so it prints a loud
    warning identifying the path and the parse error before falling back
    to ``default`` rather than raising.
    """
    try:
        path = Path(path)
        exists = path.exists()
    except Exception:
        return default
    if not exists:
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"WARNING: failed to parse JSON file {path}: {exc}")
        return default


def acquire_run_lock(out_dir: Path) -> None:
    """Acquire an exclusive run lock in ``out_dir``.

    Creates ``.gareus_run.lock`` in ``out_dir`` containing this process's
    PID, so a second campaign accidentally pointed at the same output
    directory fails fast instead of corrupting shared state. If a lock
    file already exists, its PID is checked: if that process is still
    alive, raises ``RuntimeError`` naming the directory and the live PID;
    if it is gone (stale lock left by a crash), the stale lock is removed
    and a fresh one is created. On success, registers an ``atexit`` hook
    to remove the lock file (best-effort; only removes it if it still
    contains our own PID).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lock_path = out_dir / ".gareus_run.lock"
    pid = os.getpid()

    def _pid_is_alive(other_pid: int) -> bool:
        try:
            os.kill(other_pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        existing_pid: Optional[int] = None
        try:
            existing_pid = int(lock_path.read_text(encoding="utf-8").strip())
        except Exception:
            existing_pid = None
        if existing_pid is not None and _pid_is_alive(existing_pid):
            raise RuntimeError(
                f"Another gareus run appears to be active in {out_dir} (PID {existing_pid} is still running); refusing to start."
            )
        # Stale lock left behind by a crashed/killed process: remove and retry.
        try:
            lock_path.unlink()
        except OSError:
            pass
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)

    with os.fdopen(fd, "w") as handle:
        handle.write(str(pid))

    def _release_lock() -> None:
        try:
            if lock_path.read_text(encoding="utf-8").strip() == str(pid):
                lock_path.unlink()
        except Exception:
            pass

    atexit.register(_release_lock)


def _finite_positive_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out) or out <= 0.0:
        return None
    return out


def resolve_run_temperature_k(run_dir: Path) -> Optional[float]:
    """Resolve a run directory's simulation temperature in Kelvin.

    gareus_metadata.json never carries a temperature field in practice (it's
    CV/window bookkeeping only), so that lookup alone is a silent no-op. Fall
    back to run_manifest.json's resolved_args/method_settings, which every
    run directory's provenance writer (gareus/provenance.py) always
    populates with temperature_k. Returns None if no source has a usable
    value.
    """
    run_dir = Path(run_dir)
    meta = read_json_file(run_dir / "gareus_metadata.json", {}) or {}
    temp = _finite_positive_float(meta.get("temperature_K", meta.get("temperature_k")))
    if temp is not None:
        return temp
    manifest = read_json_file(run_dir / "run_manifest.json", {}) or {}
    resolved_args = manifest.get("resolved_args", {}) or {}
    method_settings = manifest.get("method_settings", {}) or {}
    return _finite_positive_float(resolved_args.get("temperature_k", method_settings.get("temperature_k")))


def _json_ready(obj: Any) -> Any:
    """Recursively convert objects into JSON‑serialisable forms.

    Numpy types, lists and dictionaries are converted to plain Python
    equivalents.  Keys starting with an underscore are omitted to avoid
    serialising private or cached data (e.g. cached NumPy arrays stored
    in CV definitions).
    """
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _json_ready(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, (list, tuple)):
        return [_json_ready(v) for v in obj]
    if isinstance(obj, set):
        return sorted(_json_ready(v) for v in obj)
    if isinstance(obj, np.ndarray):
        return [_json_ready(v) for v in obj.tolist()]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


class BufferedCsvDictWriter:
    """Batched CSV writer for scalar logs.

    Rows are buffered and written in batches to reduce the overhead of
    writing many small rows to network storage.  It supports a subset
    of the ``csv.DictWriter`` interface used by the Gareus code.
    """

    def __init__(
        self,
        path: Path,
        fieldnames: List[str],
        append: bool = False,
        flush_rows: int = 1000,
        extrasaction: str = "ignore",
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.flush_rows = max(1, int(flush_rows or 1))
        self.handle = self.path.open("a" if append else "w", newline="")
        self.writer = csv.DictWriter(self.handle, fieldnames=fieldnames, extrasaction=extrasaction)
        self.buffer: List[Dict[str, Any]] = []
        if not append:
            self.writer.writeheader()
            self.handle.flush()

    def writerow(self, row: Dict[str, Any]) -> None:
        self.buffer.append(dict(row))
        if len(self.buffer) >= self.flush_rows:
            self.flush()

    def writerows(self, rows: Iterable[Dict[str, Any]]) -> None:
        for row in rows:
            self.writerow(row)

    def flush(self) -> None:
        if self.buffer:
            self.writer.writerows(self.buffer)
            self.buffer.clear()
        try:
            self.handle.flush()
        except Exception:
            pass

    def close(self) -> None:
        try:
            self.flush()
        finally:
            try:
                self.handle.close()
            except Exception:
                pass


class BufferedJsonlWriter:
    """Batched JSONL writer.

    This writer buffers JSON lines and writes them in batches to reduce
    file system overhead.  Use ``write_json`` to append a dictionary
    serialised as a JSON object on a single line, or ``write`` to append
    arbitrary strings.
    """

    def __init__(self, path: Path, append: bool = True, flush_rows: int = 500) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.flush_rows = max(1, int(flush_rows or 1))
        self.handle = self.path.open("a" if append else "w", buffering=1)
        self.buffer: List[str] = []

    def write_json(self, payload: Dict[str, Any]) -> None:
        self.buffer.append(json.dumps(payload, sort_keys=True) + "\n")
        if len(self.buffer) >= self.flush_rows:
            self.flush()

    def write(self, text: str) -> None:
        self.buffer.append(str(text))
        if len(self.buffer) >= self.flush_rows:
            self.flush()

    def flush(self) -> None:
        if self.buffer:
            self.handle.writelines(self.buffer)
            self.buffer.clear()
        try:
            self.handle.flush()
        except Exception:
            pass

    def close(self) -> None:
        try:
            self.flush()
        finally:
            try:
                self.handle.close()
            except Exception:
                pass


__all__ = [
    "_NumpyEncoder",
    "write_json",
    "read_json_file",
    "acquire_run_lock",
    "resolve_run_temperature_k",
    "_json_ready",
    "BufferedCsvDictWriter",
    "BufferedJsonlWriter",
]