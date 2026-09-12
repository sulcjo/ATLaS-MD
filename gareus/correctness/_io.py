"""Strict JSON, durable publication, and safe relative paths.

Atomic replace is a single-filesystem guarantee, not a universal network-
filesystem or power-loss guarantee. Writers must serialize through writer_lock.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Any, Iterator


class IntegrityError(ValueError):
    """Persisted input cannot safely be interpreted as the claimed state."""


def _pairs_no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise IntegrityError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _bad_constant(text):
    raise IntegrityError(f"Nonfinite JSON number: {text}")


def json_loads(data: bytes | str) -> Any:
    try:
        return _plain(json.loads(data, object_pairs_hook=_pairs_no_duplicates,
                          parse_constant=_bad_constant))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise IntegrityError(f"Invalid UTF-8 JSON: {exc}") from exc


def _plain(value: Any) -> Any:
    # Deliberately reject arbitrary objects instead of serializing repr().
    # NumPy scalar/array support is local to keep the filesystem layer light.
    try:
        import numpy as np
        if isinstance(value, np.ndarray):
            return _plain(value.tolist())
        if isinstance(value, np.generic):
            return _plain(value.item())
    except ImportError:
        pass
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise IntegrityError("Nonfinite number in persisted metadata")
        return value
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        if any(not isinstance(k, str) for k in value):
            raise IntegrityError("JSON metadata keys must be strings")
        return {k: _plain(v) for k, v in value.items()}
    raise IntegrityError(f"Unsupported metadata value: {type(value).__name__}")


def json_bytes(value: Any) -> bytes:
    return (json.dumps(_plain(value), sort_keys=True, separators=(",", ":"),
                       allow_nan=False, ensure_ascii=True) + "\n").encode("utf-8")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def safe_relative(text: str) -> PurePosixPath:
    if not isinstance(text, str) or not text or "\\" in text or "\x00" in text:
        raise IntegrityError(f"Invalid relative path: {text!r}")
    path = PurePosixPath(text)
    if path.is_absolute() or any(p in {"", ".", ".."} for p in text.split("/")):
        raise IntegrityError(f"Path must be normalized and relative: {text!r}")
    if any(":" in p for p in path.parts):
        raise IntegrityError(f"Drive-qualified path is forbidden: {text!r}")
    return path


def contained_file(root: Path, relative: str) -> Path:
    """Reject symlinks and traversal beneath root (trusted root, nonhostile FS).

    This guards corrupt manifests; it is not a sandbox against a malicious
    process racing filesystem mutations between lstat and open.
    """
    root = Path(root).resolve()
    rel = safe_relative(relative)
    target = root
    for part in rel.parts:
        target = target / part
        if target.is_symlink():
            raise IntegrityError(f"Symlink forbidden in committed path: {relative}")
    if not target.resolve().is_relative_to(root):
        raise IntegrityError(f"Path escapes storage root: {relative}")
    return target


def fsync_directory(path: Path) -> None:
    # Windows has no portable directory-fsync interface. Do not claim the same
    # durability there. The transaction writer currently requires POSIX locks.
    if os.name != "posix":
        raise OSError("Durable checkpoint publication requires POSIX directory fsync")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def atomic_bytes(path: Path, payload: bytes) -> None:
    """Write in the destination directory, then replace; preserve old on error."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise IntegrityError(f"Refusing to replace symlink: {path}")
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(tmp)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def writer_lock(directory: Path) -> Iterator[None]:
    """Serialize publishers; advisory locking requires cooperating writers."""
    if os.name != "posix":
        raise OSError("Checkpoint writer requires POSIX flock; no unsafe fallback")
    import fcntl
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    lock_path = directory / ".publication.lock"
    if lock_path.is_symlink():
        raise IntegrityError("Publication lock must not be a symlink")
    with lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
