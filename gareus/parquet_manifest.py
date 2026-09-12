"""Transactional Parquet segment manifests.

The manifest is the authoritative list of Parquet files that comprise one
samples/ or exchanges/ segment. Files present on disk but absent from the
manifest are uncommitted/orphaned and must be ignored by scientific readers.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Optional

MANIFEST_NAME = "parquet_manifest.json"
SCHEMA = "gareus_parquet_segment_v1"
VALID_KINDS = frozenset({"samples", "exchanges"})


class ParquetManifestError(RuntimeError):
    """Manifest-backed Parquet segment failed an integrity check."""


@dataclass(frozen=True)
class ParquetSnapshot:
    kind: str
    generation: int
    n_rows: int
    next_chunk_index: int
    files: tuple[dict[str, Any], ...]


def manifest_path(segment_dir: Path) -> Path:
    return Path(segment_dir) / MANIFEST_NAME


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while True:
            block = fh.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _fsync_file(path: Path) -> None:
    try:
        with Path(path).open("rb") as fh:
            os.fsync(fh.fileno())
    except OSError:
        pass


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(str(Path(path)), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _safe_relative_name(raw: str) -> str:
    text = str(raw or "")
    p = Path(text)
    if not text or p.is_absolute() or len(p.parts) != 1 or p.name != text or text in {".", ".."}:
        raise ParquetManifestError(
            f"invalid manifest Parquet path {raw!r}; expected one relative basename"
        )
    return text


def file_record(
    path: Path,
    *,
    rows: int,
    first_step: Optional[int],
    last_step: Optional[int],
) -> dict[str, Any]:
    path = Path(path)
    return {
        "path": path.name,
        "rows": int(rows),
        "size_bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
        "first_step": int(first_step) if first_step is not None else None,
        "last_step": int(last_step) if last_step is not None else None,
    }


def empty_manifest(kind: str) -> dict[str, Any]:
    kind = str(kind)
    if kind not in VALID_KINDS:
        raise ValueError(
            f"unsupported parquet segment kind {kind!r}; expected one of {sorted(VALID_KINDS)}"
        )
    return {
        "schema": SCHEMA,
        "kind": kind,
        "generation": 0,
        "n_rows": 0,
        "next_chunk_index": 1,
        "files": [],
    }


def validate_manifest(
    segment_dir: Path,
    manifest: dict[str, Any],
    *,
    expected_kind: Optional[str] = None,
    verify_hashes: bool = False,
) -> dict[str, Any]:
    segment_dir = Path(segment_dir)
    if not isinstance(manifest, dict):
        raise ParquetManifestError("Parquet segment manifest is not a JSON object")
    if manifest.get("schema") != SCHEMA:
        raise ParquetManifestError(
            f"unsupported Parquet manifest schema {manifest.get('schema')!r}; expected {SCHEMA!r}"
        )
    kind = str(manifest.get("kind", ""))
    if kind not in VALID_KINDS:
        raise ParquetManifestError(f"invalid Parquet manifest kind {kind!r}")
    if expected_kind is not None and kind != str(expected_kind):
        raise ParquetManifestError(
            f"Parquet manifest kind {kind!r} does not match expected {expected_kind!r}"
        )
    try:
        generation = int(manifest["generation"])
        n_rows = int(manifest["n_rows"])
        next_chunk_index = int(manifest["next_chunk_index"])
    except Exception as exc:
        raise ParquetManifestError(
            "Parquet manifest generation/n_rows/next_chunk_index must be integers"
        ) from exc
    if generation < 0 or n_rows < 0 or next_chunk_index < 1:
        raise ParquetManifestError(
            "Parquet manifest counters must be non-negative and next_chunk_index >= 1"
        )

    files = manifest.get("files")
    if not isinstance(files, list):
        raise ParquetManifestError("Parquet manifest files must be a list")
    seen: set[str] = set()
    row_sum = 0
    normalized: list[dict[str, Any]] = []
    for idx, record in enumerate(files):
        if not isinstance(record, dict):
            raise ParquetManifestError(
                f"Parquet manifest file record {idx} is not an object"
            )
        name = _safe_relative_name(record.get("path"))
        if name in seen:
            raise ParquetManifestError(
                f"Parquet manifest references {name!r} more than once"
            )
        seen.add(name)
        file_path = segment_dir / name
        if not file_path.is_file():
            raise ParquetManifestError(
                f"Parquet manifest references missing file {file_path}"
            )
        try:
            rows = int(record["rows"])
            size_bytes = int(record["size_bytes"])
        except Exception as exc:
            raise ParquetManifestError(
                f"invalid rows/size for manifest file {name!r}"
            ) from exc
        if rows < 0 or size_bytes < 0:
            raise ParquetManifestError(
                f"negative rows/size for manifest file {name!r}"
            )
        actual_size = int(file_path.stat().st_size)
        if actual_size != size_bytes:
            raise ParquetManifestError(
                f"Parquet file size mismatch for {name!r}: manifest {size_bytes}, actual {actual_size}"
            )
        expected_hash = str(record.get("sha256", ""))
        if not expected_hash:
            raise ParquetManifestError(
                f"Parquet manifest file {name!r} has no sha256"
            )
        if verify_hashes:
            actual_hash = sha256_file(file_path)
            if actual_hash != expected_hash:
                raise ParquetManifestError(
                    f"Parquet file checksum mismatch for {name!r}: manifest {expected_hash}, actual {actual_hash}"
                )
        row_sum += rows
        normalized.append({
            "path": name,
            "rows": rows,
            "size_bytes": size_bytes,
            "sha256": expected_hash,
            "first_step": record.get("first_step"),
            "last_step": record.get("last_step"),
        })
    if row_sum != n_rows:
        raise ParquetManifestError(
            f"Parquet manifest row total {n_rows} does not equal file-row sum {row_sum}"
        )
    return {
        "schema": SCHEMA,
        "kind": kind,
        "generation": generation,
        "n_rows": n_rows,
        "next_chunk_index": next_chunk_index,
        "files": normalized,
    }


def load_manifest(
    segment_dir: Path,
    *,
    expected_kind: Optional[str] = None,
    verify_hashes: bool = False,
) -> Optional[dict[str, Any]]:
    path = manifest_path(segment_dir)
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ParquetManifestError(f"could not read {path}: {exc}") from exc
    return validate_manifest(
        segment_dir,
        raw,
        expected_kind=expected_kind,
        verify_hashes=verify_hashes,
    )


def publish_manifest(segment_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    segment_dir = Path(segment_dir)
    segment_dir.mkdir(parents=True, exist_ok=True)
    normalized = validate_manifest(
        segment_dir,
        manifest,
        expected_kind=str(manifest.get("kind", "")),
        verify_hashes=False,
    )
    path = manifest_path(segment_dir)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(normalized, indent=2, sort_keys=True), encoding="utf-8")
    _fsync_file(tmp)
    os.replace(tmp, path)
    _fsync_dir(segment_dir)
    return normalized


def committed_files(
    segment_dir: Path,
    *,
    expected_kind: str,
    verify_hashes: bool = False,
) -> Optional[list[str]]:
    manifest = load_manifest(
        segment_dir,
        expected_kind=expected_kind,
        verify_hashes=verify_hashes,
    )
    if manifest is None:
        return None
    return [str(Path(segment_dir) / rec["path"]) for rec in manifest["files"]]


def append_file_to_manifest(
    segment_dir: Path,
    *,
    kind: str,
    record: dict[str, Any],
    next_chunk_index: int,
) -> dict[str, Any]:
    current = load_manifest(segment_dir, expected_kind=kind, verify_hashes=False)
    if current is None:
        current = empty_manifest(kind)
    new_manifest = dict(current)
    new_manifest["generation"] = int(current["generation"]) + 1
    new_manifest["files"] = [dict(x) for x in current["files"]] + [dict(record)]
    new_manifest["n_rows"] = int(current["n_rows"]) + int(record["rows"])
    new_manifest["next_chunk_index"] = int(next_chunk_index)
    return publish_manifest(segment_dir, new_manifest)


def replace_files_in_manifest(
    segment_dir: Path,
    *,
    kind: str,
    records: Iterable[dict[str, Any]],
    next_chunk_index: int,
) -> dict[str, Any]:
    current = load_manifest(segment_dir, expected_kind=kind, verify_hashes=False)
    if current is None:
        raise ParquetManifestError(
            "cannot replace files in a segment with no manifest"
        )
    records = [dict(r) for r in records]
    new_manifest = dict(current)
    new_manifest["generation"] = int(current["generation"]) + 1
    new_manifest["files"] = records
    new_manifest["n_rows"] = sum(int(r["rows"]) for r in records)
    new_manifest["next_chunk_index"] = int(next_chunk_index)
    return publish_manifest(segment_dir, new_manifest)


def manifest_snapshot(
    segment_dir: Path,
    *,
    expected_kind: str,
) -> Optional[ParquetSnapshot]:
    manifest = load_manifest(
        segment_dir,
        expected_kind=expected_kind,
        verify_hashes=False,
    )
    if manifest is None:
        return None
    return ParquetSnapshot(
        kind=str(manifest["kind"]),
        generation=int(manifest["generation"]),
        n_rows=int(manifest["n_rows"]),
        next_chunk_index=int(manifest["next_chunk_index"]),
        files=tuple(dict(x) for x in manifest["files"]),
    )
