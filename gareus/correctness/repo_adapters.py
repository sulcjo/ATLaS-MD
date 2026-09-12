"""Narrow adapters for the existing production/checkpoint APIs.

These preserve the existing driver and exchange ordering. They do not implement
#1, a full transactional Parquet reader, or exploratory coordinate rescue.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
import shutil
import tempfile

from ._io import IntegrityError, contained_file, fsync_directory, writer_lock
from .checkpoint_store import (CheckpointError, checkpoint_manifest_path,
                               copy_committed_generation, read_validated_generation)


def checkpoint_globals(integrator, reader):
    """Conventional integrators have no CustomIntegrator globals; real failures raise."""
    if not hasattr(integrator, "getNumGlobalVariables"):
        return {}
    return reader(integrator)


def validate_rng_restore(manifest, rng) -> None:
    expected = manifest.get("rng_bit_generator")
    actual = type(rng.bit_generator).__name__
    if expected != actual:
        raise CheckpointError(f"RNG kind mismatch: saved {expected!r}, current {actual!r}")
    # Validate without changing or consuming the production RNG stream.
    probe = copy.deepcopy(rng.bit_generator)
    try:
        probe.state = manifest["rng_state"]
    except (TypeError, ValueError, KeyError) as exc:
        raise CheckpointError(f"Saved RNG state cannot be restored: {exc}") from exc


def _copy_atomic(source: Path, destination: Path) -> None:
    if destination.is_symlink():
        raise IntegrityError(f"Refusing to overwrite destination symlink: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    temporary = Path(name)
    try:
        with source.open("rb") as src, os.fdopen(fd, "wb") as dst:
            shutil.copyfileobj(src, dst, length=1024*1024)
            dst.flush()
            os.fsync(dst.fileno())
        os.replace(temporary, destination)
        fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)


def sync_run_tree_quiescent(source, destination) -> None:
    """Copy a single paused run tree; publish checkpoint files/manifest LAST.

    Checkpoint ensemble is transactional. Each other copied file is atomic, but
    the entire collection of mutable sample/registry files is NOT transactional.
    Thus this is NOT the full output-watermark/index protocol from the plan.
    Call only at the existing writer-flushed driver barrier, not in background.
    Canonical files are copied; symlink convenience views/lock files are skipped.
    Nested independent checkpoint trees require their own explicit sync call.
    """
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination:
        return
    if source.is_relative_to(destination) or destination.is_relative_to(source):
        raise IntegrityError("Source and destination backup trees must be disjoint")
    if not source.is_dir():
        raise FileNotFoundError(source)
    pinned = read_validated_generation(source)
    if pinned is None:
        raise CheckpointError("Scratch synchronization needs a committed generation")
    destination.mkdir(parents=True, exist_ok=True)
    root_bytes = checkpoint_manifest_path(source).read_bytes()
    # Own the source checkpoint-publication lock while copying. The caller must
    # additionally ensure its sample writers/compactor are quiescent.
    with writer_lock(source / "checkpoints"):
        if checkpoint_manifest_path(source).read_bytes() != root_bytes:
            raise CheckpointError("Source checkpoint changed before synchronization")
        files = []
        for directory, dirnames, filenames in os.walk(source, followlinks=False):
            base = Path(directory)
            if base != source and "checkpoints" in dirnames:
                raise CheckpointError("Nested checkpoint trees need separate synchronization")
            dirnames[:] = [d for d in dirnames if d != "checkpoints" and not (base / d).is_symlink()]
            for name in filenames:
                candidate = base / name
                if (candidate.is_symlink() or name.endswith(".lock") or ".tmp" in name
                        or name.startswith(".publication")):
                    continue
                files.append(candidate)
        for path in files:
            relative = path.relative_to(source).as_posix()
            dest = contained_file(destination, relative)
            _copy_atomic(path, dest)
        if checkpoint_manifest_path(source).read_bytes() != root_bytes:
            raise CheckpointError("Source changed during synchronization; no checkpoint published")
        copy_committed_generation(source, destination)
