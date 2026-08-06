"""Regression test: _prepare_adaptive_merged_traj_dir dropped a phase's own
internal resume segments when merging trajectories across multiple
adaptive-production phases.

A single scheduled phase (e.g. epoch_001/baseline) can itself be
interrupted and auto-resumed multiple times (SIGTERM + checkpoint restart),
writing a fresh replica_NNN_resume_from_<local_step>.xtc file each time --
independent of which phase this is. The link-naming scheme used to key
purely off `epoch_idx` for anything past the first file seen per replica,
so every one of a phase's OWN later resume segments collided on the same
merged-dir link name and only the first survived.

Confirmed on a real run (chignolin_5): epoch_001/baseline alone had 4 real
trajectory segments per replica (1 base + 3 internal resumes); the merged
dir kept only 1, discarding ~75% of that phase's coordinate data. Overall,
chignolin-FES/Rg/PCA analyses only ever matched 856,655 of 3,348,831
frames actually present in the (buggy) merged directory (25.6%).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("analyze_gareus_mbar")

from analyze_gareus_mbar import (
    Data,
    _MERGED_TRAJ_STEP_STRIDE,
    _prepare_adaptive_merged_traj_dir,
)


def _write_traj(path: Path, n_bytes: int = 128) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x00" * n_bytes)


def _make_data(prod_dir: Path, run_dirs: list) -> Data:
    n = 4
    return Data(
        prod_dir=prod_dir, out_dir=prod_dir / "pmf_analysis",
        cv=np.zeros(n), cv2=np.full(n, np.nan), rg_A=np.full(n, np.nan),
        window=np.zeros(n, dtype=np.int32), replica=np.zeros(n, dtype=np.int32),
        step=np.arange(n, dtype=np.int64), u_nk=np.zeros((n, 1)),
        centers=np.zeros(1), k_kcal=np.zeros(1), beta=1.0, temp=300.0,
        boost_kj=np.zeros(n), potential_kj=None, source="test",
        meta={"adaptive_epoch_run_dirs": [str(rd) for rd in run_dirs]},
    )


def test_every_segment_in_every_phase_gets_a_distinct_link(tmp_path):
    prod_dir = tmp_path / "adaptive_production"
    phase0 = prod_dir / "epoch_000"
    phase1 = prod_dir / "epoch_001" / "baseline"

    # Phase 0: base + 1 internal resume (2 real segments for replica_000).
    _write_traj(phase0 / "replica_trajectories" / "replica_000.xtc")
    _write_traj(phase0 / "replica_trajectories" / "replica_000_resume_from_000500000.xtc")

    # Phase 1: base + 3 internal resumes (4 real segments for replica_000) --
    # the exact shape of the real chignolin_5/epoch_001/baseline case.
    _write_traj(phase1 / "replica_trajectories" / "replica_000.xtc")
    _write_traj(phase1 / "replica_trajectories" / "replica_000_resume_from_003736350.xtc")
    _write_traj(phase1 / "replica_trajectories" / "replica_000_resume_from_007976400.xtc")
    _write_traj(phase1 / "replica_trajectories" / "replica_000_resume_from_011997844.xtc")

    d = _make_data(prod_dir, [phase0, phase1])
    merged = _prepare_adaptive_merged_traj_dir(d)

    assert merged is not None
    links = sorted(p.name for p in merged.iterdir())
    # 2 segments from phase 0 + 4 segments from phase 1 = 6 distinct links.
    assert len(links) == 6, links

    STRIDE = _MERGED_TRAJ_STEP_STRIDE
    expected = {
        "replica_000.xtc",  # phase 0's own base -> global_resume == 0
        f"replica_000_resume_from_{500000}.xtc",  # phase 0's internal resume
        f"replica_000_resume_from_{1 * STRIDE}.xtc",  # phase 1's own base
        f"replica_000_resume_from_{1 * STRIDE + 3736350}.xtc",
        f"replica_000_resume_from_{1 * STRIDE + 7976400}.xtc",
        f"replica_000_resume_from_{1 * STRIDE + 11997844}.xtc",
    }
    assert set(links) == expected


def test_zero_byte_files_are_skipped(tmp_path):
    prod_dir = tmp_path / "adaptive_production"
    phase0 = prod_dir / "epoch_000"
    _write_traj(phase0 / "replica_trajectories" / "replica_000.xtc", n_bytes=64)
    (phase0 / "replica_trajectories" / "replica_000_resume_from_000500000.xtc").touch()  # 0 bytes

    d = _make_data(prod_dir, [phase0])
    merged = _prepare_adaptive_merged_traj_dir(d)

    links = sorted(p.name for p in merged.iterdir())
    assert links == ["replica_000.xtc"]


def test_rebuilds_from_scratch_discarding_stale_links(tmp_path):
    """A merged dir from a run predating this fix (or from a since-changed
    phase list) must never leave incorrect leftover links in place."""
    prod_dir = tmp_path / "adaptive_production"
    phase0 = prod_dir / "epoch_000"
    _write_traj(phase0 / "replica_trajectories" / "replica_000.xtc")

    merged_dir = prod_dir / "_merged_replica_trajectories"
    merged_dir.mkdir(parents=True)
    stale = merged_dir / "replica_999_stale_leftover.xtc"
    stale.write_bytes(b"\x00" * 32)
    assert stale.exists()

    d = _make_data(prod_dir, [phase0])
    merged = _prepare_adaptive_merged_traj_dir(d)

    assert merged is not None
    assert not stale.exists()
    assert sorted(p.name for p in merged.iterdir()) == ["replica_000.xtc"]
