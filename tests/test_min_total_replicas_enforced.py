"""An EXPLICITLY configured replica minimum must be a creation floor, not a
report-only note. Regression for the chignolin 2.5 µs run: min_total_windows=12
was configured but the 2D double-adaptive finalizer only *noted* "below minimum"
and returned the grid unchanged (finalized 9). These tests pin:

  1. An explicit min expands the finalized factorized grid up to it, by
     subdividing the existing accessible span (capped by max_total, soft-clamp,
     no raise).
  2. With NO explicit min there is NO implicit floor -- the adaptive dispatcher's
     own count is kept (it prunes overdense windows and converges on its own; a
     default floor would fight that, cf. test_synth_drivers_feedback).
  3. The adaptive-k clamp ceiling defaults to 1000 so a tight ladder can reach
     the target neighbour overlap.

No OpenMM / PeptideBuilder imports (unit-test rule).
"""
import argparse

import pytest

import gareus.cli as cli
from gareus import windows


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    for fn in (
        "_add_core_args", "_add_system_args", "_add_cv_args", "_add_window_args",
        "_add_us_args", "_add_seeding_args", "_add_genpept_prescan_args",
        "_add_gamd_args", "_add_output_args", "_add_platform_args",
    ):
        getattr(cli, fn)(p)
    return p


def _parsed(argv):
    args = _build_parser().parse_args(argv)
    cli._apply_v2_compat_shims(args)
    return args


BASE = ["--seq", "AAAAA", "--cv1", "contacts", "--cv2", "rama-map"]


def test_no_implicit_floor_when_min_unset():
    # No explicit min -> the finalizer must NOT expand; the collapsed grid is
    # left for the dispatcher to manage. (A default floor here broke the
    # pruning/economy synth tests.)
    args = _parsed(BASE)
    primary = [0.0, 0.0559, 0.1119]
    secondary = [-0.37, -0.34, -0.28]
    new_primary, new_secondary, meta = windows._apply_total_window_budget_to_factorized_grid(
        args, primary, secondary
    )
    assert len(new_primary) * len(new_secondary) == 9  # unchanged
    assert meta.get("expanded_to_floor") is False


def test_explicit_min_expands_collapsed_grid():
    # The exact chignolin failure: 3 primary x 3 secondary = 9, explicit min 14.
    args = _parsed(BASE + ["--min-total-windows", "14", "--max-total-windows", "24"])
    primary = [0.0, 0.0559, 0.1119]
    secondary = [-0.37, -0.34, -0.28]
    new_primary, new_secondary, meta = windows._apply_total_window_budget_to_factorized_grid(
        args, primary, secondary
    )
    assert 14 <= len(new_primary) * len(new_secondary) <= 24
    # Expansion stays WITHIN the accessible span (no widening past observed bounds).
    assert min(new_primary) >= min(primary) - 1e-9
    assert max(new_primary) <= max(primary) + 1e-9
    assert meta.get("expanded_to_floor") is True


def test_explicit_min_soft_clamps_to_smaller_max_no_raise():
    # An explicit min above the max budget clamps down to the cap in the finalizer
    # (never raises there); the count stays within max.
    args = _parsed(BASE + ["--min-total-windows", "20", "--max-total-windows", "8"])
    primary = [0.0, 0.1, 0.2, 0.3]
    secondary = [-1.0, 1.0]
    new_primary, new_secondary, _meta = windows._apply_total_window_budget_to_factorized_grid(
        args, primary, secondary
    )
    assert len(new_primary) * len(new_secondary) <= 8


def test_adaptive_k_max_ceiling_defaults_to_1000():
    # The adaptive-k clamp must allow k up to 1000 so a tightly-spaced ladder can
    # reach the target neighbour overlap (the old 120 / 500 ceilings capped it,
    # leaving windows over-overlapped). Applies to both CVs when unset.
    args = _parsed(BASE)
    assert args.contact_adaptive_max_k_kcal == 1000.0
    assert args.secondary_cv_adaptive_max_k_kcal == 1000.0


def test_explicit_min_exceeding_max_still_raises_in_axis_bounds():
    # The axis-bounds helper (used at bootstrap) still treats an explicit
    # min>max as a genuine conflict and raises.
    args = _parsed(BASE + ["--max-total-windows", "8", "--min-total-windows", "20"])
    with pytest.raises(ValueError):
        windows._axis_count_bounds_from_total_window_budget(
            args, n_secondary=4, legacy_min_axis=4, legacy_max_axis=12, label="contact"
        )
