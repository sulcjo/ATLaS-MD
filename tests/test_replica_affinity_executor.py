"""Thread-affinity invariant of `gareus.production._ReplicaAffinityExecutor`.

The class exists for one reason, spelled out in its own docstring: a replica's
CUDA Context must be first-touched, and then stepped for the rest of the run,
from a *single* OS thread.  Migrating a Context to another thread on first use
is a driver-level SIGSEGV hazard (observed as a segfault firing the instant
after replica_construction reached N/N).  So the property under test here is
observed thread identity - `threading.get_ident()` collected across repeated
dispatches - and not construction arguments: an executor that dutifully builds
one `max_workers=1` pool per replica but routes replica 3's task into replica
0's pool satisfies every argument-shaped assertion while reintroducing exactly
the crash this class prevents.

Note on `get_ident()`: idents are recycled once a thread dies, so every pool
whose idents are being compared must still be alive.  Each test below builds
its own executor and only shuts it down after all comparisons are done (or, for
the shutdown test, asserts nothing about idents afterwards).
"""

from __future__ import annotations

import threading

import gareus.production as production


_NREP = 4
_TIMEOUT = 30


def _ident() -> int:
    return threading.get_ident()


def _submit_idents(pool, replica_index: int, n_calls: int) -> list[int]:
    """Sequentially dispatch `n_calls` no-op tasks to one replica's pool."""
    return [
        pool.submit(replica_index, _ident).result(timeout=_TIMEOUT)
        for _ in range(n_calls)
    ]


def test_submit_pins_each_replica_to_one_private_thread() -> None:
    """Repeated submit(i, ...) always runs on the same thread, and replicas
    never share a thread with each other."""
    pool = production._ReplicaAffinityExecutor(_NREP)
    try:
        per_replica = {i: set(_submit_idents(pool, i, 3)) for i in range(_NREP)}

        # Affinity: one and only one thread ever served replica i.
        for i, idents in per_replica.items():
            assert len(idents) == 1, f"replica {i} ran on {len(idents)} threads: {idents}"

        # Isolation: no two replicas were served by the same thread.  With one
        # ident per replica this is just a cardinality check, and it is what
        # would fail if submit() ignored replica_index (all tasks landing on a
        # single shared worker) or if the pools shared a multi-worker executor.
        all_idents = {next(iter(s)) for s in per_replica.values()}
        assert len(all_idents) == _NREP, f"replica threads not disjoint: {per_replica}"

        # ... and none of them is the calling thread.
        assert _ident() not in all_idents
    finally:
        pool.shutdown(wait=True)


def test_map_dispatches_item_j_to_the_same_thread_submit_j_uses() -> None:
    """map() is a second, independent dispatch path from submit().

    A map() that funnelled every item through one pool would leave
    test_submit_pins_each_replica_to_one_private_thread green while breaking
    affinity for the production step loop, which drives replicas exclusively
    through _sim_pool.map (see run_gareus's _step_item / _fetch_state calls).
    """
    pool = production._ReplicaAffinityExecutor(_NREP)
    try:
        submit_idents = [_submit_idents(pool, i, 1)[0] for i in range(_NREP)]

        # map() passes each item positionally to the pool at the same index, so
        # item j must come back on replica j's already-established thread.
        map_idents = pool.map(lambda item: _ident(), list(range(_NREP)))
        assert list(map_idents) == submit_idents

        # Stable across calls, and a shorter item list still starts at pool 0.
        assert list(pool.map(lambda item: _ident(), list(range(_NREP)))) == submit_idents
        assert list(pool.map(lambda item: _ident(), [0, 1])) == submit_idents[:2]
    finally:
        pool.shutdown(wait=True)


def test_exception_in_one_replica_does_not_poison_its_own_or_another_pool() -> None:
    """A raising task propagates through .result() but must not kill the worker.

    If the failing replica's thread were replaced, its Context would be
    first-touched again from a *new* thread on the next step - precisely the
    migration this class exists to prevent - so the same-ident assertion after
    the failure is the real content here, not the raised-exception assertion.
    """
    pool = production._ReplicaAffinityExecutor(_NREP)
    try:
        before = {i: _submit_idents(pool, i, 1)[0] for i in range(_NREP)}

        def _boom():
            raise ValueError("replica 1 blew up")

        fut = pool.submit(1, _boom)
        try:
            fut.result(timeout=_TIMEOUT)
        except ValueError as exc:
            assert "replica 1 blew up" in str(exc)
        else:  # pragma: no cover - defensive
            raise AssertionError("exception did not propagate through .result()")

        # The failed replica keeps its own thread ...
        assert _submit_idents(pool, 1, 2) == [before[1], before[1]]
        # ... and every other replica keeps its own, unchanged.
        for i in range(_NREP):
            if i == 1:
                continue
            assert _submit_idents(pool, i, 1)[0] == before[i]

        # map() over all replicas still works after the failure.
        assert list(pool.map(lambda item: _ident(), list(range(_NREP)))) == [
            before[i] for i in range(_NREP)
        ]
    finally:
        pool.shutdown(wait=True)


def test_shutdown_closes_every_replica_pool_not_just_the_first() -> None:
    """shutdown(wait=True) must reject later work for *every* replica index.

    Asserted per index: a shutdown() that only closed self._pools[0] would pass
    an index-0-only check while leaving nrep-1 live threads holding CUDA
    Contexts after the run believes it has released them.
    """
    pool = production._ReplicaAffinityExecutor(_NREP)
    # Touch every pool first, so each has a real, running worker thread to close.
    for i in range(_NREP):
        assert pool.submit(i, _ident).result(timeout=_TIMEOUT)

    pool.shutdown(wait=True)

    for i in range(_NREP):
        try:
            pool.submit(i, _ident)
        except RuntimeError:
            pass
        else:  # pragma: no cover - defensive
            raise AssertionError(f"pool for replica {i} still accepted work after shutdown")

    # Idempotent: run_gareus's shutdown path can be reached more than once
    # (normal completion plus the graceful-shutdown handler).
    pool.shutdown(wait=True)


def test_pools_are_single_worker_and_built_from_the_module_threadpool_name() -> None:
    """One max_workers=1 pool per replica, from `production.ThreadPoolExecutor`.

    This is the argument-shaped half of the invariant - necessary but not
    sufficient, which is why the affinity tests above assert on observed thread
    identity instead.  Kept because it also pins the executor to the
    module-level, eagerly-bound `ThreadPoolExecutor` name guarded by
    tests/test_package_smoke.py::test_production_threadpool_import_is_py314_safe.
    """
    pool = production._ReplicaAffinityExecutor(_NREP)
    try:
        assert len(pool._pools) == _NREP
        for p in pool._pools:
            assert isinstance(p, production.ThreadPoolExecutor)
            assert p._max_workers == 1
    finally:
        pool.shutdown(wait=True)
