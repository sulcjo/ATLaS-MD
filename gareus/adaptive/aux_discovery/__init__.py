"""Adaptive auxiliary-CV discovery (spec 2026-10-09-cvaux-adaptive-discovery-design.md)."""
from .settings import AuxDiscoverySettings  # noqa: F401


def __getattr__(name):  # lazy: the pipeline pulls in sklearn/openmm-side modules
    if name in ("DiscoveryResult", "run_discovery"):
        from . import pipeline
        return getattr(pipeline, name)
    raise AttributeError(name)
