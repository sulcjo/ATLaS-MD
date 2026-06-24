"""Synthetic adaptive-window test harness.

Drives the real GAREUS adaptive decision code (window create/move/remove/k/2D
patch in feedback; top-up/retire in production) with synthetic samples drawn
from an analytic free-energy surface, so the adaptive behavior can be validated
and tuned against a known answer without running OpenMM MD.

No OpenMM / PeptideBuilder / gamd-openmm dependency.
"""
from __future__ import annotations

from .landscapes import LANDSCAPES, Landscape, mixture_wells
from .sampler import Window, sample_window, sample_window_exact
from .exchange import build_exchange_stats
from .oracle import (grid_overlap, ideal_axis_ladder, low_f_mask, reference_pmf,
                     oracle_k_for_r, oracle_optimal_centers_1d,
                     oracle_optimal_centers_2d, placement_comparison_1d)
from .drivers import (EpochRecord, RoundRecord, feedback_campaign,
                      production_campaign, sample_final_layout)
from .metrics import campaign_metrics

__all__ = [
    "Landscape", "LANDSCAPES", "mixture_wells",
    "Window", "sample_window", "sample_window_exact",
    "build_exchange_stats",
    "grid_overlap", "ideal_axis_ladder", "low_f_mask", "reference_pmf",
    "oracle_k_for_r", "oracle_optimal_centers_1d",
    "oracle_optimal_centers_2d", "placement_comparison_1d",
    "RoundRecord", "EpochRecord", "feedback_campaign", "production_campaign",
    "sample_final_layout", "campaign_metrics",
]
