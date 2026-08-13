import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import analyze_gareus_mbar as agm


def _make_data_stub(replica, epoch_source=None):
    """Minimal object exposing only the fields _sample_block_ids reads."""
    class _Stub:
        pass
    d = _Stub()
    d.replica = np.asarray(replica)
    d.meta = {}
    if epoch_source is not None:
        d.meta['_epoch_source'] = list(epoch_source)
    return d


def test_block_ids_group_by_replica_when_no_epoch_source():
    d = _make_data_stub(replica=[0, 0, 1, 1, 2])
    block_ids = agm._sample_block_ids(d)
    # Same replica -> same block id; different replica -> different block id.
    assert block_ids[0] == block_ids[1]
    assert block_ids[2] == block_ids[3]
    assert len({block_ids[0], block_ids[2], block_ids[4]}) == 3


def test_block_ids_distinguish_same_replica_across_epoch_sources():
    # Replica 0 in epoch source 0 and replica 0 in epoch source 1 must be
    # DIFFERENT blocks -- adaptive-production runs don't guarantee trajectory
    # continuity across epoch boundaries.
    d = _make_data_stub(replica=[0, 0, 0, 0], epoch_source=[0, 0, 1, 1])
    block_ids = agm._sample_block_ids(d)
    assert block_ids[0] == block_ids[1]
    assert block_ids[2] == block_ids[3]
    assert block_ids[0] != block_ids[2]


def test_block_ids_are_compact_zero_based():
    d = _make_data_stub(replica=[5, 5, 9, 9, 5])
    block_ids = agm._sample_block_ids(d)
    assert set(np.unique(block_ids)) == {0, 1}
    assert block_ids.dtype == np.int64
