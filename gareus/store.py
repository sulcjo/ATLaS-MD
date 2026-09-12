"""
Parquet-based timeseries storage for GAREUS production runs.

Replaces CSV/NPZ output with a columnar format:
  samples/<seg_id>/chunk_XXXXXX.parquet  — per-step CV + energy data
  exchanges/<seg_id>/chunk_XXXXXX.parquet — exchange events
  windows/<seg_id>.json                  — window snapshot for the segment
  segments.json                          — restart/segment chain
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from .parquet_manifest import (
    ParquetManifestError,
    append_file_to_manifest,
    file_record,
    load_manifest,
    replace_files_in_manifest,
)


def _fsync_path(path: Path) -> None:
    """Best-effort fsync for a newly written Parquet file."""
    try:
        with Path(path).open("rb") as fh:
            os.fsync(fh.fileno())
    except OSError:
        pass


class _ManifestParquetWriter:
    """Shared crash-safe file publication for sample/exchange writers."""

    kind: str
    sort_columns: tuple[tuple[str, str], ...]

    def _init_manifest_state(self) -> None:
        manifest = load_manifest(self._out_dir, expected_kind=self.kind, verify_hashes=False)
        if manifest is None:
            # A writer must never append blindly to a legacy segment: that was
            # the route by which a consolidated data.parquet later acquired a
            # second overlapping chunk set. Legacy directories stay readable,
            # but require an explicit repair/migration before further writes.
            legacy = sorted(self._out_dir.glob("*.parquet"))
            if legacy:
                raise ParquetManifestError(
                    f"refusing to append to legacy {self.kind} segment {self._out_dir}: "
                    f"found {len(legacy)} Parquet file(s) but no parquet_manifest.json"
                )
            self._chunk_idx = 0
            return
        self._chunk_idx = int(manifest["next_chunk_index"]) - 1

    def _publish_table(self, tbl) -> Path:
        import pyarrow.parquet as pq

        self._chunk_idx += 1
        chunk_path = self._out_dir / f"chunk_{self._chunk_idx:06d}.parquet"
        if chunk_path.exists():
            raise ParquetManifestError(
                f"refusing to overwrite committed/unclassified Parquet file {chunk_path}"
            )
        for stale in self._out_dir.glob(f"{chunk_path.name}.tmp.*"):
            stale.unlink(missing_ok=True)
        tmp_path = chunk_path.with_name(f"{chunk_path.name}.tmp.{os.getpid()}")
        pq.write_table(tbl, tmp_path, compression="zstd", compression_level=3)
        _fsync_path(tmp_path)
        os.replace(tmp_path, chunk_path)
        _fsync_path(chunk_path)

        steps = tbl.column("step") if "step" in tbl.column_names else None
        first_step = int(steps[0].as_py()) if steps is not None and len(steps) else None
        last_step = int(steps[-1].as_py()) if steps is not None and len(steps) else None
        record = file_record(
            chunk_path,
            rows=int(tbl.num_rows),
            first_step=first_step,
            last_step=last_step,
        )
        # The manifest is published last. A kill between the Parquet rename
        # and this replace leaves an orphan file, not a scientifically visible
        # committed file for manifest-aware readers.
        append_file_to_manifest(
            self._out_dir,
            kind=self.kind,
            record=record,
            next_chunk_index=self._chunk_idx + 1,
        )
        return chunk_path

    def _consolidate(self) -> None:
        """Transactionally compact the currently committed file set.

        Old files are not removed until the new compact file has been written,
        validated and made authoritative by an atomic manifest replacement.
        A crash during best-effort cleanup therefore leaves harmless unreferenced
        Parquet files rather than two logical copies of the same rows.
        """
        import pyarrow.dataset as ds
        import pyarrow.parquet as pq

        manifest = load_manifest(self._out_dir, expected_kind=self.kind, verify_hashes=True)
        if manifest is None or len(manifest["files"]) <= 1:
            return
        old_paths = [self._out_dir / rec["path"] for rec in manifest["files"]]
        table = ds.dataset([str(p) for p in old_paths], format="parquet").to_table()
        if self.sort_columns:
            table = table.sort_by(list(self.sort_columns))
        if int(table.num_rows) != int(manifest["n_rows"]):
            raise ParquetManifestError(
                f"refusing to compact {self.kind} segment {self._out_dir}: "
                f"read {table.num_rows} rows, manifest declares {manifest['n_rows']}"
            )

        compact_name = f"compact_{int(manifest['generation']) + 1:06d}.parquet"
        compact_path = self._out_dir / compact_name
        if compact_path.exists():
            raise ParquetManifestError(f"compaction destination already exists: {compact_path}")
        tmp = compact_path.with_name(f"{compact_path.name}.tmp.{os.getpid()}")
        pq.write_table(table, tmp, compression="zstd", compression_level=3)
        _fsync_path(tmp)
        os.replace(tmp, compact_path)
        _fsync_path(compact_path)

        steps = table.column("step") if "step" in table.column_names else None
        record = file_record(
            compact_path,
            rows=int(table.num_rows),
            first_step=int(steps[0].as_py()) if steps is not None and len(steps) else None,
            last_step=int(steps[-1].as_py()) if steps is not None and len(steps) else None,
        )
        replace_files_in_manifest(
            self._out_dir,
            kind=self.kind,
            records=[record],
            next_chunk_index=self._chunk_idx + 1,
        )

        # Cleanup is deliberately after the manifest switch. Failures here are
        # safe: readers ignore any old file no longer referenced by the manifest.
        for path in old_paths:
            if path != compact_path:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass


class ParquetSampleWriter(_ManifestParquetWriter):
    """Buffers per-step sample data and publishes immutable Parquet chunks."""

    kind = "samples"
    sort_columns = (("step", "ascending"), ("replica", "ascending"))

    def __init__(self, out_dir: Path, flush_rows: int = 5000) -> None:
        self._out_dir = Path(out_dir)
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._flush_rows = max(1, flush_rows)
        self._buf: Dict[str, list] = defaultdict(list)
        self._init_manifest_state()

    def write_sample(
        self,
        step: int,
        replica: int,
        window_id: int,
        cv1: float,
        cv2: Optional[float],
        potential: float,
        boost_total: float,
        boost_dihedral: float,
        boost_nonbonded: float,
        v_pep: float = float("nan"),
        v_dih: float = float("nan"),
        gamd_lambda: float = 0.0,
    ) -> None:
        b = self._buf
        b["step"].append(step)
        b["replica"].append(replica)
        b["window_id"].append(window_id)
        b["cv1"].append(cv1)
        b["cv2"].append(cv2)
        b["potential"].append(potential)
        b["gamd_boost_total"].append(boost_total)
        b["gamd_boost_dihedral"].append(boost_dihedral)
        b["gamd_boost_nonbonded"].append(boost_nonbonded)
        b["v_pep_kj_mol"].append(v_pep)
        b["v_dih_kj_mol"].append(v_dih)
        b["gamd_lambda"].append(gamd_lambda)
        if len(b["step"]) >= self._flush_rows:
            self.flush()

    def flush(self) -> None:
        if not self._buf["step"]:
            return
        import pyarrow as pa

        b = self._buf
        tbl = pa.table({
            "step":                pa.array(b["step"],                type=pa.uint64()),
            "replica":             pa.array(b["replica"],             type=pa.uint16()),
            "window_id":           pa.array(b["window_id"],           type=pa.uint16()),
            "cv1":                 pa.array(b["cv1"],                 type=pa.float32()),
            "cv2":                 pa.array(b["cv2"],                 type=pa.float32()),
            "potential":           pa.array(b["potential"],           type=pa.float32()),
            "gamd_boost_total":    pa.array(b["gamd_boost_total"],    type=pa.float32()),
            "gamd_boost_dihedral": pa.array(b["gamd_boost_dihedral"], type=pa.float32()),
            "gamd_boost_nonbonded":pa.array(b["gamd_boost_nonbonded"],type=pa.float32()),
            "v_pep_kj_mol":        pa.array(b["v_pep_kj_mol"],        type=pa.float32()),
            "v_dih_kj_mol":        pa.array(b["v_dih_kj_mol"],        type=pa.float32()),
            "gamd_lambda":         pa.array(b["gamd_lambda"],         type=pa.float32()),
        })
        self._publish_table(tbl)
        for values in b.values():
            values.clear()

    def close(self) -> None:
        self.flush()
        self._consolidate()


class ParquetExchangeWriter(_ManifestParquetWriter):
    """Buffers exchange events and publishes immutable Parquet chunks."""

    kind = "exchanges"
    sort_columns = (("step", "ascending"),)

    def __init__(self, out_dir: Path, flush_rows: int = 1000) -> None:
        self._out_dir = Path(out_dir)
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._flush_rows = max(1, flush_rows)
        self._buf: Dict[str, list] = defaultdict(list)
        self._init_manifest_state()

    def write_exchange(
        self,
        step: int,
        replica_i: int,
        replica_j: int,
        window_i: int,
        window_j: int,
        delta_e: float,
        accepted: bool,
    ) -> None:
        b = self._buf
        b["step"].append(step)
        b["replica_i"].append(replica_i)
        b["replica_j"].append(replica_j)
        b["window_i"].append(window_i)
        b["window_j"].append(window_j)
        b["delta_e"].append(delta_e)
        b["accepted"].append(accepted)
        if len(b["step"]) >= self._flush_rows:
            self.flush()

    def flush(self) -> None:
        if not self._buf["step"]:
            return
        import pyarrow as pa

        b = self._buf
        tbl = pa.table({
            "step":      pa.array(b["step"],      type=pa.uint64()),
            "replica_i": pa.array(b["replica_i"], type=pa.uint16()),
            "replica_j": pa.array(b["replica_j"], type=pa.uint16()),
            "window_i":  pa.array(b["window_i"],  type=pa.uint16()),
            "window_j":  pa.array(b["window_j"],  type=pa.uint16()),
            "delta_e":   pa.array(b["delta_e"],   type=pa.float32()),
            "accepted":  pa.array(b["accepted"],  type=pa.bool_()),
        })
        self._publish_table(tbl)
        for values in b.values():
            values.clear()

    def close(self) -> None:
        self.flush()
        self._consolidate()


class SegmentRegistry:
    """Manages segments.json — the restart/segment chain for a run.

    Each gareus invocation is one segment. Segments link via parent_segment_id
    so the full sample history can be reconstructed by reading all segments
    belonging to a run in order.
    """

    def __init__(self, run_dir: Path) -> None:
        self._path = Path(run_dir) / "segments.json"
        self._segments: List[Dict[str, Any]] = []
        if self._path.exists():
            self._segments = json.loads(self._path.read_text(encoding="utf-8"))

    def _next_id(self) -> str:
        return f"seg_{len(self._segments) + 1:03d}"

    def _save(self) -> None:
        tmp = self._path.with_name(f"{self._path.name}.tmp.{os.getpid()}")
        tmp.write_text(json.dumps(self._segments, indent=2), encoding="utf-8")
        tmp.replace(self._path)

    def open_segment(self, run_id: str, parent_id: Optional[str], round_id: int) -> str:
        seg_id = self._next_id()
        self._segments.append({
            "segment_id": seg_id,
            "run_id": run_id,
            "parent_segment_id": parent_id,
            "round_id": round_id,
            "start_step": None,
            "end_step": None,
            "status": "running",
        })
        self._save()
        return seg_id

    def close_segment(self, segment_id: str, end_step: int) -> None:
        for seg in self._segments:
            if seg["segment_id"] == segment_id:
                seg["end_step"] = end_step
                seg["status"] = "complete"
                break
        self._save()

    def seal_segment(self, segment_id: str, absolute_end_step: int, status: str = "interrupted") -> None:
        """Mark a previously-running segment as interrupted or abandoned.

        interrupted: crashed mid-run; absolute_end_step is the last valid checkpoint
            step — rows with step > absolute_end_step in its Parquet files are
            phantom frames from a simulation state that was rolled back.
        abandoned: crashed before first checkpoint or fresh start after crash;
            absolute_end_step=-1 means no valid data boundary known; query layer
            skips the segment entirely.

        Does NOT overwrite segments that are already "complete".
        """
        for seg in self._segments:
            if seg["segment_id"] == segment_id:
                if seg.get("status") == "complete":
                    return
                seg["end_step"] = absolute_end_step
                seg["status"] = status
                break
        self._save()

    def set_segment_start_step(self, segment_id: str, start_step: int) -> None:
        """Record the absolute simulation step at which this segment begins."""
        for seg in self._segments:
            if seg["segment_id"] == segment_id:
                seg["start_step"] = start_step
                break
        self._save()

    def get_segment(self, segment_id: str) -> Optional[Dict[str, Any]]:
        for seg in self._segments:
            if seg["segment_id"] == segment_id:
                return seg
        return None

    def get_latest_segment(self) -> Optional[Dict[str, Any]]:
        return self._segments[-1] if self._segments else None

    def all_segments(self) -> List[Dict[str, Any]]:
        return list(self._segments)


def finalize_segment(
    registry: "SegmentRegistry",
    seg_id: str,
    *,
    completed_cleanly: bool,
    writers_ok: bool,
    end_step: int,
) -> None:
    """Atomically decide segment fate and update the registry.

    Marks the segment as ``"complete"`` only when the production loop finished
    without exception *and* all parquet writers flushed and closed without
    error.  Any other outcome seals the segment as ``"interrupted"`` so the
    next resume knows to pick up from the last valid checkpoint.
    """
    if completed_cleanly and writers_ok:
        registry.close_segment(seg_id, end_step=end_step)
    else:
        registry.seal_segment(seg_id, absolute_end_step=end_step, status="interrupted")


class WindowSnapshot:
    """Writes a per-segment window definition snapshot."""

    def __init__(self, run_dir: Path) -> None:
        self._win_dir = Path(run_dir) / "windows"
        self._win_dir.mkdir(parents=True, exist_ok=True)

    def snapshot(
        self,
        segment_id: str,
        windows: List[Dict[str, Any]],
        cv1_type: str,
        cv2_type: Optional[str],
    ) -> None:
        payload = {
            "segment_id": segment_id,
            "cv1_type": cv1_type,
            "cv2_type": cv2_type,
            "windows": windows,
        }
        target = self._win_dir / f"{segment_id}.json"
        tmp = target.with_name(f"{target.name}.tmp.{os.getpid()}")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(target)

    def load(self, segment_id: str) -> Dict[str, Any]:
        return json.loads(
            (self._win_dir / f"{segment_id}.json").read_text(encoding="utf-8")
        )


def parse_gamd_boost_components(
    boost_total: Optional[float],
    boost_components: Dict[str, float],
) -> tuple:
    """Extract (total, dihedral, nonbonded) from a GaMD boost components dict."""
    if not boost_components:
        return boost_total, None, None

    dihedral: Optional[float] = None
    nonbonded: Optional[float] = None
    unmatched: list = []

    for key, val in boost_components.items():
        k = key.lower()
        if any(s in k for s in ("dihedral", "torsion")):
            dihedral = float(val)
        elif any(s in k for s in ("nonbond", "lj", "vdw", "elec")):
            nonbonded = float(val)
        else:
            unmatched.append((key, float(val)))

    if dihedral is None and nonbonded is None and len(unmatched) >= 2:
        unmatched.sort(key=lambda x: x[0])
        dihedral = unmatched[0][1]
        nonbonded = unmatched[1][1]

    return boost_total, dihedral, nonbonded


__all__ = [
    "ParquetSampleWriter",
    "ParquetExchangeWriter",
    "SegmentRegistry",
    "WindowSnapshot",
    "parse_gamd_boost_components",
]
