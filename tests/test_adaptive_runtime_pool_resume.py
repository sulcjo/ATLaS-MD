"""Smoke tests for adaptive-production runtime-pool resume accounting.

These tests are intentionally OpenMM-free.  They validate the ledger logic used
by --adaptive-production-total-md-pool-ns and --adaptive-production-resume.
"""

from pathlib import Path
from types import SimpleNamespace
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gareus.adaptive_production import (
    AdaptiveDecisionPolicy,
    AdaptiveRuntimePool,
    _adaptive_runtime_pool_validate,
    _load_or_initialize_runtime_pool,
    evaluate_context_reuse_readiness,
)


def test_runtime_pool_ledger_roundtrip(tmp_path: Path):
    pool = AdaptiveRuntimePool(total_ns=10.0, timestep_fs=4.0)
    pool.consume(label="epoch_000", kind="epoch", n_states=10, steps=1000, path=tmp_path / "epoch_000")
    payload = _adaptive_runtime_pool_validate(tmp_path, pool, label="runtime_pool_test")
    assert payload["status"] in {"ok", "warning"}
    assert abs(payload["summed_event_ns"] - 0.04) < 1.0e-12
    assert abs(payload["used_ns"] - 0.04) < 1.0e-12


def test_runtime_pool_resume_loads_previous_ledger(tmp_path: Path):
    args = SimpleNamespace(adaptive_production_total_md_pool_ns=5.0, timestep_fs=2.0)
    policy = AdaptiveDecisionPolicy(total_md_pool_ns=5.0)
    pool = AdaptiveRuntimePool(total_ns=5.0, timestep_fs=2.0)
    pool.consume(label="epoch_000", kind="epoch", n_states=4, steps=1000, path=tmp_path / "epoch_000")
    from gareus.io import write_json
    write_json(tmp_path / "adaptive_runtime_pool.json", pool.to_dict())
    loaded, validation = _load_or_initialize_runtime_pool(tmp_path, args, policy, resume_requested=True)
    assert abs(loaded.used_ns - pool.used_ns) < 1.0e-12
    assert validation["status"] in {"ok", "warning"}


def test_context_reuse_required_errors(tmp_path: Path):
    args = SimpleNamespace(
        adaptive_production_context_reuse=True,
        adaptive_production_context_reuse_require=True,
        adaptive_production_context_reuse_mode="inprocess-experimental",
    )
    try:
        evaluate_context_reuse_readiness(args, tmp_path, registry=None)
    except RuntimeError:
        pass
    else:
        raise AssertionError("required unsupported context reuse should raise")


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        test_runtime_pool_ledger_roundtrip(Path(d) / "a")
    with tempfile.TemporaryDirectory() as d:
        test_runtime_pool_resume_loads_previous_ledger(Path(d) / "b")
    with tempfile.TemporaryDirectory() as d:
        test_context_reuse_required_errors(Path(d) / "c")
    print("adaptive runtime-pool/resume smoke tests passed")
