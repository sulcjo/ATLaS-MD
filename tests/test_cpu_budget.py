"""Tests for CPU budget distribution across replicas."""
from __future__ import annotations

import types


def _args(**kwargs):
    defaults = dict(cpu_budget=0, max_cpu_per_replica=0, cpu_threads=1)
    defaults.update(kwargs)
    return types.SimpleNamespace(**defaults)


def test_budget_distributes_evenly():
    from gareus.system_setup import resolve_cpu_threads_for_replicas
    args = _args(cpu_budget=32)
    assert resolve_cpu_threads_for_replicas(args, 8) == 4


def test_budget_floored_not_rounded():
    from gareus.system_setup import resolve_cpu_threads_for_replicas
    args = _args(cpu_budget=10)
    assert resolve_cpu_threads_for_replicas(args, 3) == 3  # floor(10/3) = 3


def test_max_per_replica_caps_budget():
    from gareus.system_setup import resolve_cpu_threads_for_replicas
    args = _args(cpu_budget=32, max_cpu_per_replica=2)
    assert resolve_cpu_threads_for_replicas(args, 8) == 2  # min(4, 2)


def test_max_per_replica_caps_explicit_threads():
    from gareus.system_setup import resolve_cpu_threads_for_replicas
    args = _args(cpu_threads=8, max_cpu_per_replica=3)
    assert resolve_cpu_threads_for_replicas(args, 4) == 3  # min(8, 3)


def test_no_budget_falls_back_to_cpu_threads():
    from gareus.system_setup import resolve_cpu_threads_for_replicas
    args = _args(cpu_threads=4)
    assert resolve_cpu_threads_for_replicas(args, 8) == 4


def test_budget_minimum_one():
    from gareus.system_setup import resolve_cpu_threads_for_replicas
    args = _args(cpu_budget=1)
    assert resolve_cpu_threads_for_replicas(args, 100) == 1  # floor(1/100)=0 → clamp to 1


def test_zero_replicas_fallback():
    from gareus.system_setup import resolve_cpu_threads_for_replicas
    args = _args(cpu_budget=32, cpu_threads=4)
    # n_replicas=0 is degenerate; should not crash and should fallback gracefully
    result = resolve_cpu_threads_for_replicas(args, 0)
    assert result >= 1
