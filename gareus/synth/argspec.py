"""Build a faithful ``args`` namespace for the real adaptive-feedback dispatcher.

The dispatcher reads many attributes off ``args`` (mostly via ``getattr`` with
defaults).  This module constructs a ``SimpleNamespace`` carrying the contact /
rama-map 2D configuration the harness uses, so the real decision code runs
unchanged against synthetic samples.  The attribute set mirrors
``tests/test_delaunay_windows.py:_make_args`` plus the adaptive-feedback knobs
verified in ``gareus/adaptive_feedback.py`` and ``gareus/windows.py``.
"""
from __future__ import annotations

import types
from typing import Optional


def make_feedback_args(*, target_overlap: float = 0.30, round_index: int = 0,
                       aggressiveness: str = "balanced",
                       secondary_cv_centers=(-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0),
                       contact_min: float = 0.0, contact_max: float = 0.80,
                       adaptive_memory: Optional[dict] = None,
                       **overrides) -> types.SimpleNamespace:
    """Return a ``SimpleNamespace`` of args for ``run_adaptive_feedback_dispatcher_2d``."""
    d = dict(
        primary_cv="nonlocal-contacts",
        secondary_cv="rama-map",
        secondary_cv_centers=list(secondary_cv_centers),
        contact_normalize=True,
        adaptive_window_aggressiveness=aggressiveness,
        adaptive_feedback_target_overlap=float(target_overlap),
        adaptive_feedback_round_index=int(round_index),
        adaptive_feedback_memory=adaptive_memory,
        # contact-mode adaptive knobs
        contact_adaptive_min=float(contact_min),
        contact_adaptive_max=float(contact_max),
        contact_adaptive_hit_radius=0.08,
        contact_adaptive_tight_hit_radius=0.04,
        contact_adaptive_max_center_shift=0.08,
        contact_adaptive_min_new_spacing=0.04,
        contact_adaptive_coverage_gap_width=0.10,
        contact_adaptive_target_spacing=0.08,
        contact_adaptive_min_k_kcal=10.0,
        contact_adaptive_max_k_kcal=200.0,
        contact_adaptive_default_k_kcal=25.0,
        contact_k_kcal=None,
        # secondary-CV knobs
        secondary_cv_k_kcal=25.0,
        secondary_cv_k_mode="fixed",
        secondary_cv_adaptive_min_k_kcal=5.0,
        secondary_cv_adaptive_max_k_kcal=100.0,
        # generic window k + explicit-2d graph knobs
        default_window_k_kcal_a2=50.0,
        explicit_2d_exchange_neighbor_k=2,
        explicit_2d_exchange_radius=1.65,
    )
    d.update(overrides)
    return types.SimpleNamespace(**d)
