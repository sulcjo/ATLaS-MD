"""Immutable ensemble checkpoint generations with validate-before-load semantics.

A generation binds binary checkpoints and Python bookkeeping. This module never
loads or steps an OpenMM Context. Capture must happen at an ensemble barrier.
Its copy operation copies *checkpoint generations only*, not a transaction over
mutable samples/trajectories elsewhere in the run directory.

No generation garbage collection is performed: in-flight readers and previous
commits retain valid filenames. Filesystems must support POSIX locks, same-FS
rename, and directory fsync. Network-storage durability must be validated locally.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import uuid
from typing import Any, Callable, Iterable, Mapping
import warnings

from ._io import (IntegrityError, atomic_bytes, contained_file, digest,
                  fsync_directory, json_bytes, json_loads, safe_relative,
                  write_new, writer_lock)

SCHEMA = "gareus_production_checkpoint_v2"
LEGACY_SCHEMA = "gareus_production_checkpoint_v1"
MANIFEST_NAME = "production_checkpoint_manifest.json"
EventHook = Callable[[str, Mapping[str, Any]], None]


class CheckpointError(IntegrityError):
    """Checkpoint exists but is not safe to use as a complete ensemble."""


class LegacyCheckpointError(CheckpointError):
    """Legacy files have no immutable-generation evidence."""


@dataclass(frozen=True)
class ValidatedGeneration:
    manifest: dict[str, Any]
    replica_payloads: tuple[bytes, ...]
    artifact_payloads: dict[str, bytes]
    validation_level: str


@dataclass(frozen=True)
class CheckpointInspection:
    status: str
    generation_id: str | None = None
    reason: str | None = None


def checkpoint_manifest_path(out_dir: Path | str) -> Path:
    return Path(out_dir) / "checkpoints" / MANIFEST_NAME


def _event(hook: EventHook | None, event: str, **details: Any) -> None:
    if hook is not None:
        hook(event, details)


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CheckpointError(f"{name} must be an integer >= {minimum}; got {value!r}")
    return value


def _bookkeeping(manifest: dict[str, Any], n: int) -> None:
    if n <= 0:
        raise CheckpointError("Cannot publish an empty ensemble")
    assignments = manifest.get("assignments")
    if not isinstance(assignments, list) or len(assignments) != n:
        raise CheckpointError("assignments must have one entry per replica")
    for i, state in enumerate(assignments):
        _integer(state, f"assignments[{i}]")
    # This is the present ATLaS-MD permutation-ensemble contract, not a general
    # requirement for arbitrary multi-walker algorithms.
    if sorted(assignments) != list(range(n)):
        raise CheckpointError("assignments must be a permutation of 0..n_replicas-1")
    for field in ("prod_done", "absolute_step", "parity", "attempt",
                  "next_exchange", "next_log"):
        _integer(manifest.get(field), field)
    if manifest["absolute_step"] < manifest["prod_done"]:
        raise CheckpointError("absolute_step precedes prod_done")
    if not isinstance(manifest.get("rng_state"), dict):
        raise CheckpointError("rng_state must be captured with the ensemble")
    if not isinstance(manifest.get("rng_bit_generator"), str) or not manifest["rng_bit_generator"]:
        raise CheckpointError("rng_bit_generator must name the saved NumPy generator")
    if not isinstance(manifest.get("exchange_stats"), dict):
        raise CheckpointError("exchange_stats must be captured with the ensemble")
    for field in ("replica_integrator_globals_all", "replica_integrator_global_counts",
                  "replica_context_metadata"):
        if field in manifest and (not isinstance(manifest[field], list)
                                  or len(manifest[field]) != n):
            raise CheckpointError(f"{field} must have one entry per replica")
    if "replica_integrator_globals_all" in manifest:
        globals_all = manifest["replica_integrator_globals_all"]
        if any(not isinstance(g, dict) for g in globals_all):
            raise CheckpointError("Invalid per-replica globals")
        counts = manifest.get("replica_integrator_global_counts")
        if counts is not None and counts != [len(g) for g in globals_all]:
            raise CheckpointError("Recorded integrator-global counts disagree")
    if "next_checkpoint" in manifest:
        _integer(manifest["next_checkpoint"], "next_checkpoint")
    # Optional participant_state is owned by its caller, not this module. It
    # allows another controller to persist state without implementing its physics.
    if "participant_state" in manifest and not isinstance(manifest["participant_state"], dict):
        raise CheckpointError("participant_state must be a versioned mapping")


def _storage_root(out_dir: Path | str) -> Path:
    root = Path(out_dir) / "checkpoints"
    if root.is_symlink():
        raise CheckpointError("Checkpoint directory must not be a symlink")
    return root


def _record(payload: bytes) -> dict[str, Any]:
    return {"size": len(payload), "sha256": digest(payload)}


def _root_bytes(root: Path) -> bytes | None:
    path = root / MANIFEST_NAME
    if path.is_symlink():
        raise CheckpointError("Root manifest must not be a symlink")
    try:
        return path.read_bytes()
    except FileNotFoundError:
        return None


def publish_generation(
    out_dir: Path | str,
    replica_payloads: Iterable[bytes],
    manifest_fields: Mapping[str, Any],
    *,
    artifacts: Mapping[str, bytes] | None = None,
    fault_hook: EventHook | None = None,
) -> dict[str, Any]:
    """Publish all replicas or none; old committed files are never overwritten.

    Input objects are detached immediately. fault_hook exists for deterministic
    crash tests; production callers should leave it unset. A failure after root
    replacement can leave a *complete new generation* committed, never a mixture.
    """
    fields = json_loads(json_bytes(dict(manifest_fields)))
    raw_payloads = tuple(replica_payloads)
    if any(not isinstance(p, (bytes, bytearray, memoryview)) for p in raw_payloads):
        raise CheckpointError("Checkpoint payloads must be bytes-like, not numbers or strings")
    payloads = tuple(bytes(p) for p in raw_payloads)
    if any(not p for p in payloads):
        raise CheckpointError("Empty binary checkpoint")
    _bookkeeping(fields, len(payloads))
    artifact_data = {}
    for logical, payload in (artifacts or {}).items():
        safe_relative(logical)
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise CheckpointError(f"Artifact {logical!r} must be bytes-like")
        artifact_data[logical] = bytes(payload)
    root = _storage_root(out_dir)
    with writer_lock(root):
        generations = contained_file(root, "generations")
        generations.mkdir(exist_ok=True)
        fsync_directory(root)
        generation_id = uuid.uuid4().hex
        staging = root / f".staging-{generation_id}"
        staging.mkdir()
        final = generations / generation_id
        prefix = f"generations/{generation_id}"
        records = {}
        replica_files = []
        artifact_files = {}
        previous = _root_bytes(root)
        # Reject malformed old metadata. The previous binaries are not reread on
        # save: this new generation is captured from the live ensemble, not rebuilt
        # from those files. A reader still validates every referenced byte.
        previous_id = None
        if previous is not None:
            prev = json_loads(previous)
            if not isinstance(prev, dict):
                raise CheckpointError("Previous root manifest is not an object")
            previous_id = prev.get("generation_id")
        try:
            _event(fault_hook, "staging_created", generation_id=generation_id)
            for i, payload in enumerate(payloads):
                name = f"replica_{i:03d}.chk"
                write_new(staging / name, payload)
                rel = f"{prefix}/{name}"
                records[rel] = _record(payload)
                replica_files.append(rel)
                _event(fault_hook, "replica_written", replica=i)
            for logical, payload in sorted(artifact_data.items()):
                name = f"artifacts/{logical}"
                write_new(staging / name, payload)
                rel = f"{prefix}/{name}"
                records[rel] = _record(payload)
                artifact_files[logical] = rel
                _event(fault_hook, "artifact_written", logical=logical)
            # Discard v1 filename/schema fields; preserve all other caller-owned
            # state, including future participant/controller fields.
            fields.update({
                "schema": SCHEMA,
                "generation_id": generation_id,
                "previous_generation_id": previous_id,
                "replica_checkpoint_files": replica_files,
                "replica_count": len(payloads),
                "artifact_files": artifact_files,
                "file_records": records,
            })
            encoded = json_bytes(fields)
            write_new(staging / "manifest.json", encoded)
            _event(fault_hook, "manifest_written")
            # Synchronize every newly created nested directory, bottom up.
            for directory, _, _ in os.walk(staging, topdown=False):
                fsync_directory(Path(directory))
            _event(fault_hook, "generation_synced")
            os.rename(staging, final)
            fsync_directory(generations)
            fsync_directory(root)
            _event(fault_hook, "generation_published")
            atomic_bytes(root / MANIFEST_NAME, encoded)
            _event(fault_hook, "root_published")
            return fields
        finally:
            # A hard kill may leave staging behind; the reader never enumerates
            # it. Completed but unpublished generations are deliberately kept.
            if staging.exists():
                shutil.rmtree(staging)


def _checked_payload(root: Path, path: str, record: Any) -> bytes:
    if not isinstance(record, dict):
        raise CheckpointError(f"Missing checksum record: {path}")
    try:
        payload = contained_file(root, path).read_bytes()
    except OSError as exc:
        raise CheckpointError(f"Cannot read committed file {path}: {exc}") from exc
    size = _integer(record.get("size"), f"{path}.size")
    sha = record.get("sha256")
    if not isinstance(sha, str) or len(sha) != 64:
        raise CheckpointError(f"Invalid checksum record: {path}")
    if len(payload) != size or digest(payload) != sha:
        raise CheckpointError(f"Checksum/length mismatch in {path}")
    return payload


def read_validated_generation(
    out_dir: Path | str, *, allow_legacy: bool = False,
) -> ValidatedGeneration | None:
    """Read root once, verify every byte, then return the exact validated bytes.

    Callers must load these payloads rather than reopening their paths, which
    would discard the validate-before-load guarantee. None means genuinely absent.
    """
    root = _storage_root(out_dir)
    encoded = _root_bytes(root)
    if encoded is None:
        return None
    manifest = json_loads(encoded)
    if not isinstance(manifest, dict):
        raise CheckpointError("Root manifest must be a JSON object")
    paths = manifest.get("replica_checkpoint_files")
    if not isinstance(paths, list) or not paths or len(set(map(str, paths))) != len(paths):
        raise CheckpointError("Invalid or duplicated replica paths")
    _bookkeeping(manifest, len(paths))
    schema = manifest.get("schema")
    if schema == LEGACY_SCHEMA:
        if not allow_legacy:
            raise LegacyCheckpointError(
                "Legacy v1 checkpoint cannot prove ensemble atomicity. Use explicit "
                "allow_legacy=True only after reviewing its history; it stays unverified."
            )
        payloads = tuple(contained_file(root, p).read_bytes() for p in paths)
        if any(not p for p in payloads):
            raise CheckpointError("Empty legacy checkpoint")
        warnings.warn("Loading an unverified v1 checkpoint generation", RuntimeWarning,
                      stacklevel=2)
        manifest["unverified_legacy_ancestry"] = True
        return ValidatedGeneration(manifest, payloads, {}, "legacy_unverified")
    if schema != SCHEMA:
        raise CheckpointError(f"Unsupported checkpoint schema: {schema!r}")
    generation_id = manifest.get("generation_id")
    if (not isinstance(generation_id, str) or len(generation_id) != 32
            or any(c not in "0123456789abcdef" for c in generation_id)):
        raise CheckpointError("Malformed generation ID")
    if _integer(manifest.get("replica_count"), "replica_count", 1) != len(paths):
        raise CheckpointError("Replica count disagrees with file list")
    prefix = f"generations/{generation_id}/"
    immutable_manifest = contained_file(root, prefix + "manifest.json")
    try:
        generation_bytes = immutable_manifest.read_bytes()
    except OSError as exc:
        raise CheckpointError(f"Missing immutable manifest: {exc}") from exc
    if generation_bytes != encoded:
        raise CheckpointError("Root metadata differs from its immutable generation")
    artifacts = manifest.get("artifact_files", {})
    records = manifest.get("file_records", {})
    if not isinstance(artifacts, dict) or not isinstance(records, dict):
        raise CheckpointError("Malformed artifact/checksum table")
    all_paths = paths + list(artifacts.values())
    if (any(not isinstance(p, str) or not p.startswith(prefix) for p in all_paths)
            or len(set(all_paths)) != len(all_paths)
            or set(records) != set(all_paths)):
        raise CheckpointError("Mixed generations, duplicated paths, or incomplete checksum table")
    for logical in artifacts:
        safe_relative(logical)
    data = {p: _checked_payload(root, p, records[p]) for p in all_paths}
    return ValidatedGeneration(manifest, tuple(data[p] for p in paths),
                               {k: data[p] for k, p in artifacts.items()}, "valid_v2")


def inspect_generation(out_dir: Path | str) -> CheckpointInspection:
    try:
        result = read_validated_generation(out_dir)
    except LegacyCheckpointError as exc:
        return CheckpointInspection("legacy_unverified", reason=str(exc))
    except (IntegrityError, OSError, TypeError) as exc:
        return CheckpointInspection("corrupt", reason=str(exc))
    if result is None:
        return CheckpointInspection("absent")
    return CheckpointInspection("valid_v2", result.manifest["generation_id"])


def require_available(out_dir: Path | str) -> bool:
    """False only for absence; corrupt/unverified must not trigger fresh setup."""
    result = inspect_generation(out_dir)
    if result.status == "absent":
        return False
    if result.status != "valid_v2":
        raise CheckpointError(f"Checkpoint is {result.status}: {result.reason}")
    return True


def copy_committed_generation(
    source: Path | str, destination: Path | str, *,
    fault_hook: EventHook | None = None,
) -> dict[str, Any]:
    """Copy one pinned, verified generation; publish destination manifest last.

    Does NOT copy mutable run-level output. Callers needing an output transaction
    must pass its immutable reader index/files as artifacts when publishing, and
    teach the reader to use that index. Never describe this as whole-run backup.
    """
    pinned = read_validated_generation(source)
    if pinned is None:
        raise CheckpointError("Source has no committed checkpoint")
    source_root = _storage_root(source)
    dest_root = _storage_root(destination)
    if source_root.resolve() == dest_root.resolve():
        return pinned.manifest
    manifest = pinned.manifest
    generation_id = manifest["generation_id"]
    encoded = json_bytes(manifest)
    # Root and generation bytes were canonical at publication. Keep their exact
    # representation for checking, including forward-compatible fields.
    src_encoded = contained_file(source_root,
        f"generations/{generation_id}/manifest.json").read_bytes()
    if json_loads(src_encoded) != manifest:
        raise CheckpointError("Pinned source generation changed during copy")
    encoded = src_encoded
    payloads = dict(zip(manifest["replica_checkpoint_files"], pinned.replica_payloads))
    payloads.update({manifest["artifact_files"][key]: data
                     for key, data in pinned.artifact_payloads.items()})
    with writer_lock(dest_root):
        generations = contained_file(dest_root, "generations")
        generations.mkdir(exist_ok=True)
        final = contained_file(dest_root, f"generations/{generation_id}")
        if final.exists():
            if (final / "manifest.json").read_bytes() != encoded:
                raise CheckpointError("Destination generation ID has different contents")
            for path, data in payloads.items():
                existing = _checked_payload(dest_root, path, manifest["file_records"][path])
                if existing != data:
                    raise CheckpointError("Destination bytes disagree with pinned source")
        else:
            staging = dest_root / f".staging-copy-{uuid.uuid4().hex}"
            staging.mkdir()
            prefix = f"generations/{generation_id}/"
            try:
                for i, (path, data) in enumerate(sorted(payloads.items())):
                    write_new(staging / path.removeprefix(prefix), data)
                    _event(fault_hook, "file_copied", index=i, path=path)
                write_new(staging / "manifest.json", encoded)
                for directory, _, _ in os.walk(staging, topdown=False):
                    fsync_directory(Path(directory))
                _event(fault_hook, "copy_synced")
                os.rename(staging, final)
                fsync_directory(generations)
                fsync_directory(dest_root)
                _event(fault_hook, "copy_generation_published")
            finally:
                if staging.exists():
                    shutil.rmtree(staging)
        atomic_bytes(dest_root / MANIFEST_NAME, encoded)
        _event(fault_hook, "copy_root_published")
    return manifest
