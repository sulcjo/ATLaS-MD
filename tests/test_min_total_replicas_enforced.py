"""The replica minimum must be a CREATION floor, not a report-only note.

Regression for the chignolin 2.5 µs run: min_total_windows=12 was configured but
the 2D double-adaptive path finalized only 9 windows (a 3x3 grid) because
`_apply_total_window_budget_to_factorized_grid` merely *noted* "below requested
minimum" and never expanded the grid. These tests pin:

  1. A default replica floor (DEFAULT_MIN_TOTAL_REPLICAS == 14) applies when the
     user sets no explicit min.
  2. When a factorized grid is below the floor, it is EXPANDED (within the
     existing axis span -> tighter spacing) up to the floor, capped by max_total.
  3. The default floor soft-clamps to an explicit (smaller) max_total and never
     raises; an explicit min that genuinely exceeds max_total still raises.

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


def test_default_replica_floor_is_14():
    assert windows.DEFAULT_MIN_TOTAL_REPLICAS == 14
    # With no explicit min-total (and no cap), a 2D factorized grid below the
    # default floor is expanded up to 14. Scoped to the 2D finalizer so 1D /
    # legacy paths are unaffected.
    args = _parsed(BASE)
    new_primary, new_secondary, meta = windows._apply_total_window_budget_to_factorized_grid(
        args, [0.0, 0.0559, 0.1119], [-0.37, -0.34, -0.28]
    )
    assert len(new_primary) * len(new_secondary) >= 14
    assert meta.get("expanded_to_floor") is True


def test_collapsed_grid_is_expanded_up_to_floor():
    # The exact chignolin failure: 3 primary x 3 secondary = 9, floor 14.
    args = _parsed(BASE + ["--min-total-windows", "14", "--max-total-windows", "24"])
    primary = [0.0, 0.0559, 0.1119]
    secondary = [-0.37, -0.34, -0.28]
    new_primary, new_secondary, meta = windows._apply_total_window_budget_to_factorized_grid(
        args, primary, secondary
    )
    assert len(new_primary) * len(new_secondary) >= 14
    assert len(new_primary) * len(new_secondary) <= 24
    # Expansion stays WITHIN the accessible span (no widening past observed bounds).
    assert min(new_primary) >= min(primary) - 1e-9
    assert max(new_primary) <= max(primary) + 1e-9
    assert meta.get("expanded_to_floor") is True


def test_default_floor_soft_clamps_to_smaller_max_no_raise():
    # Default floor (14) must never raise against an explicit smaller max budget;
    # it clamps down to what the max permits.
    args = _parsed(BASE + ["--max-total-windows", "8"])
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


def test_explicit_min_exceeding_max_still_raises():
    args = _parsed(BASE + ["--max-total-windows", "8", "--min-total-windows", "20"])
    with pytest.raises(ValueError):
        windows._axis_count_bounds_from_total_window_budget(
            args, n_secondary=4, legacy_min_axis=4, legacy_max_axis=12, label="contact"
        )
