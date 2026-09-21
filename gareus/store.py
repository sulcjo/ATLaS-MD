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
    _fsync_dir,
    _fsync_file,
    append_file_to_manifest,
    file_record,
    load_manifest,
    replace_files_in_manifest,
)


class ParquetSampleWriter:
    """Buffers per-step sample data and flushes to Parquet chunks.

    Each flush produces one atomic chunk_XXXXXX.parquet via tmp→rename,
    so partial flushes on crash leave no corrupt files.
    """

    def __init__(self, out_dir: Path, flush_rows: int = 5000) -> None:
        self._out_dir = Path(out_dir)
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._flush_rows = max(1, flush_rows)
        self._buf: Dict[str, list] = defaultdict(list)
        self._manifest = load_manifest(self._out_dir, expected_kind="samples")
        if self._manifest is None and list(self._out_dir.glob("*.parquet")):
            raise ParquetManifestError(f"cannot append to legacy Parquet segment without manifest: {self._out_dir}")
        self._chunk_idx = (self._manifest["next_chunk_index"] - 1) if self._manifest else 0

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
        import pyarrow.parquet as pq

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

        self._chunk_idx += 1
        chunk_path = self._out_dir / f"chunk_{self._chunk_idx:06d}.parquet"
        # Per-PID tmp names avoid a real collision between two concurrent
        # writers, but orphan a tmp file forever if THIS process gets killed
        # mid-write.  Sweep any stale tmp for this exact chunk index left by a
        # prior killed attempt before writing our own - self-healing on the
        # next successful flush rather than accumulating forever.
        for stale in self._out_dir.glob(f"{chunk_path.name}.tmp.*"):
            stale.unlink(missing_ok=True)
        tmp_path = chunk_path.with_name(f"{chunk_path.name}.tmp.{os.getpid()}")
        pq.write_table(tbl, tmp_path, compression="zstd", compression_level=3)
        tmp_path.rename(chunk_path)
        _fsync_file(chunk_path)
        _fsync_dir(self._out_dir)
        record = file_record(
            chunk_path,
            rows=tbl.num_rows,
            first_step=min(b["step"]),
            last_step=max(b["step"]),
        )
        self._manifest = append_file_to_manifest(
            self._out_dir,
            kind="samples",
            record=record,
            next_chunk_index=self._chunk_idx + 1,
        )

        for lst in b.values():
            lst.clear()

    def close(self) -> None:
        self.flush()
        self._consolidate()

    def _consolidate(self) -> None:
        """Publish compacted Parquet through manifest switch, then collect sources."""
        manifest = load_manifest(self._out_dir, expected_kind="samples")
        if manifest is None or not any(str(r["path"]).startswith("chunk_") for r in manifest["files"]):
            return
        source_records = list(manifest["files"])
        chunks = [self._out_dir / r["path"] for r in source_records]
        import pyarrow.dataset as ds
        import pyarrow.parquet as pq
        tbl = ds.dataset(chunks, format="parquet").to_table()
        tbl = tbl.sort_by([("step", "ascending"), ("replica", "ascending")])
        has_compact = any(str(r["path"]).startswith("data") for r in source_records)
        name = f"data_{manifest['generation'] + 1:06d}.parquet" if has_compact else "data.parquet"
        output = self._out_dir / name
        tmp = output.with_name(f"{output.name}.tmp.{os.getpid()}")
        pq.write_table(tbl, tmp, compression="zstd", compression_level=3)
        _fsync_file(tmp)
        tmp.rename(output)
        _fsync_file(output)
        _fsync_dir(self._out_dir)
        record = file_record(output, rows=tbl.num_rows, first_step=int(tbl["step"][0].as_py()), last_step=int(tbl["step"][-1].as_py()))
        self._manifest = replace_files_in_manifest(
            self._out_dir, kind="samples", records=[record], next_chunk_index=self._chunk_idx + 1
        )
        for source in chunks:
            source.unlink(missing_ok=True)
        _fsync_dir(self._out_dir)


class ParquetExchangeWriter:
    """Buffers exchange events and flushes to Parquet chunks."""

    def __init__(self, out_dir: Path, flush_rows: int = 1000) -> None:
        self._out_dir = Path(out_dir)
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._flush_rows = max(1, flush_rows)
        self._buf: Dict[str, list] = defaultdict(list)
        self._manifest = load_manifest(self._out_dir, expected_kind="exchanges")
        if self._manifest is None and list(self._out_dir.glob("*.parquet")):
            raise ParquetManifestError(f"cannot append to legacy Parquet segment without manifest: {self._out_dir}")
        self._chunk_idx = (self._manifest["next_chunk_index"] - 1) if self._manifest else 0

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
        import pyarrow.parquet as pq

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

        self._chunk_idx += 1
        chunk_path = self._out_dir / f"chunk_{self._chunk_idx:06d}.parquet"
        # Per-PID tmp names avoid a real collision between two concurrent
        # writers, but orphan a tmp file forever if THIS process gets killed
        # mid-write.  Sweep any stale tmp for this exact chunk index left by a
        # prior killed attempt before writing our own - self-healing on the
        # next successful flush rather than accumulating forever.
        for stale in self._out_dir.glob(f"{chunk_path.name}.tmp.*"):
            stale.unlink(missing_ok=True)
        tmp_path = chunk_path.with_name(f"{chunk_path.name}.tmp.{os.getpid()}")
        pq.write_table(tbl, tmp_path, compression="zstd", compression_level=3)
        tmp_path.rename(chunk_path)
        _fsync_file(chunk_path)
        _fsync_dir(self._out_dir)
        record = file_record(
            chunk_path,
            rows=tbl.num_rows,
            first_step=min(b["step"]),
            last_step=max(b["step"]),
        )
        self._manifest = append_file_to_manifest(
            self._out_dir,
            kind="exchanges",
            record=record,
            next_chunk_index=self._chunk_idx + 1,
        )

        for lst in b.values():
            lst.clear()

    def close(self) -> None:
        self.flush()
        self._consolidate()

    def _consolidate(self) -> None:
        """Publish compacted Parquet through manifest switch, then collect sources."""
        manifest = load_manifest(self._out_dir, expected_kind="exchanges")
        if manifest is None or not any(str(r["path"]).startswith("chunk_") for r in manifest["files"]):
            return
        source_records = list(manifest["files"])
        chunks = [self._out_dir / r["path"] for r in source_records]
        import pyarrow.dataset as ds
        import pyarrow.parquet as pq
        tbl = ds.dataset(chunks, format="parquet").to_table()
        tbl = tbl.sort_by([("step", "ascending")])
        has_compact = any(str(r["path"]).startswith("data") for r in source_records)
        name = f"data_{manifest['generation'] + 1:06d}.parquet" if has_compact else "data.parquet"
        output = self._out_dir / name
        tmp = output.with_name(f"{output.name}.tmp.{os.getpid()}")
        pq.write_table(tbl, tmp, compression="zstd", compression_level=3)
        _fsync_file(tmp)
        tmp.rename(output)
        _fsync_file(output)
        _fsync_dir(self._out_dir)
        record = file_record(output, rows=tbl.num_rows, first_step=int(tbl["step"][0].as_py()), last_step=int(tbl["step"][-1].as_py()))
        self._manifest = replace_files_in_manifest(
            self._out_dir, kind="exchanges", records=[record], next_chunk_index=self._chunk_idx + 1
        )
        for source in chunks:
            source.unlink(missing_ok=True)
        _fsync_dir(self._out_dir)


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

    Parameters
    ----------
    registry:
        The :class:`SegmentRegistry` for the current run.
    seg_id:
        Segment ID to finalize.
    completed_cleanly:
        ``True`` iff the production loop ran to completion (``prod_done >=
        prod_total``) without raising an exception.
    writers_ok:
        ``True`` iff all parquet writer flush/close calls succeeded.
    end_step:
        ``calib_steps + prod_done`` — the absolute step boundary recorded in
        the registry entry.  For interrupted segments this is the *last known
        step*; phantom frames beyond a checkpoint boundary are filtered by the
        resume-time safety net.
    """
    if completed_cleanly and writers_ok:
        registry.close_segment(seg_id, end_step=end_step)
    else:
        registry.seal_segment(seg_id, absolute_end_step=end_step, status="interrupted")


class WindowSnapshot:
    """Writes a per-segment window definition snapshot.

    Windows can change between adaptive feedback rounds, so each segment
    records its own window set in windows/<segment_id>.json.
    """

    def __init__(self, run_dir: Path) -> None:
        self._win_dir = Path(run_dir) / "windows"
        self._win_dir.mkdir(parents=True, exist_ok=True)

    def snapshot(
        self,
        segment_id: str,
        windows: List[Dict[str, Any]],
        cv1_type: str,
        cv2_type: Optional[str],
        kernel_identity: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload = {
            "segment_id": segment_id,
            "cv1_type": cv1_type,
            "cv2_type": cv2_type,
            "windows": windows,
        }
        if kernel_identity is not None:
            # Which arithmetic produced this segment's coordinates and exchange energies
            # (gareus.kernel_identity); readers classify sample eligibility from it.
            payload["kernel_identity"] = dict(kernel_identity)
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
    """Extract (total, dihedral, nonbonded) from a GaMD boost components dict.

    Component key names vary by gamd-openmm version. Uses name heuristics:
    - dihedral: key contains "dihedral" or "torsion"
    - nonbonded: key contains "nonbond", "lj", "vdw", or "elec"
    - two unnamed components: assign sorted-key order (first=dihedral, second=nonbonded)
    - one or zero components: dihedral=None, nonbonded=None
    """
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
