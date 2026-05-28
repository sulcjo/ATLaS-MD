"""
I/O helper routines.

This module bundles together simple JSON and CSV utilities originally
implemented in ``gareus_peptide.py``.  It includes a JSON encoder
capable of serialising NumPy types, convenience functions to read and
write JSON files, and buffered writers for CSV and JSONL output.
"""

from __future__ import annotations

import json
import csv
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
    transparently via :class:`_NumpyEncoder`.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, cls=_NumpyEncoder), encoding="utf-8")


def read_json_file(path: Path, default: Optional[Any] = None) -> Any:
    """Read a JSON file if it exists; return ``default`` on error or missing file."""
    try:
        path = Path(path)
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


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
    "_json_ready",
    "BufferedCsvDictWriter",
    "BufferedJsonlWriter",
]