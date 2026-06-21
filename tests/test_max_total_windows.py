"""Tests for the user-settable total-window/replica budget (--max-total-windows).

The budget machinery already lives in gareus.windows; these tests pin:
  1. CLI/config wiring: --max-total-windows / --min-total-windows flow through
     _apply_v2_compat_shims onto the internal adaptive_*_total_windows budget
     attrs that windows.py consumes (previously hardcoded to 0 = unsettable).
  2. Enforcement: a factorized CV1 x CV2 grid is thinned to honour the cap.

No OpenMM / PeptideBuilder imports (unit-test rule).
"""
import argparse

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


def test_max_total_windows_is_a_real_settable_arg():
    args = _parsed(BASE + ["--max-total-windows", "8"])
    assert args.max_total_windows == 8
    # Maps onto both generic and contact budget attrs consumed by windows.py.
    assert args.adaptive_max_total_windows == 8
    assert args.contact_adaptive_max_total_windows == 8


def test_min_total_windows_is_a_real_settable_arg():
    args = _parsed(BASE + ["--min-total-windows", "6"])
    assert args.min_total_windows == 6
    assert args.adaptive_min_total_windows == 6
    assert args.contact_adaptive_min_total_windows == 6


def test_budget_defaults_to_uncapped_zero():
    args = _parsed(BASE)
    assert args.adaptive_max_total_windows == 0
    assert args.contact_adaptive_max_total_windows == 0
    assert args.adaptive_min_total_windows == 0


def test_max_total_windows_is_a_known_config_dest():
    # Must be a real argparse dest, else YAML configs naming it are rejected
    # as "unknown config key".
    from gareus.config import _build_known_config_dests
    known = _build_known_config_dests(_build_parser())
    assert "max_total_windows" in known
    assert "min_total_windows" in known


def test_factorized_grid_thinned_to_budget():
    # 5 primary x 4 secondary = 20 windows, cap to 8 -> primary thinned to 2.
    args = _parsed(BASE + ["--max-total-windows", "8"])
    args.secondary_cv_centers = [-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0]
    primary = [0.0, 0.05, 0.10, 0.15, 0.20]
    secondary = list(args.secondary_cv_centers)
    new_primary, new_secondary, meta = windows._apply_total_window_budget_to_factorized_grid(
        args, primary, secondary
    )
    assert len(new_primary) * len(new_secondary) <= 8
    assert meta["enabled"] is True
    assert meta["max_total_windows"] == 8
    # endpoints preserved when thinning the primary axis
    assert new_primary[0] == 0.0 and new_primary[-1] == 0.20


def test_axis_bounds_derive_from_total_budget():
    # axis_max = floor(max_total / n_secondary): 12 / 4 = 3 primary centers max.
    args = _parsed(BASE + ["--max-total-windows", "12"])
    axis_min, axis_max, meta = windows._axis_count_bounds_from_total_window_budget(
        args, n_secondary=4, legacy_min_axis=4, legacy_max_axis=12, label="contact"
    )
    assert axis_max == 3
    assert meta["enabled"] is True
    # legacy per-axis min (4) must NOT win over an explicit tight total budget:
    # axis_min is pulled down to axis_max instead of raising.
    assert axis_min <= axis_max


def test_tight_budget_overrides_legacy_axis_min_no_raise():
    # 12 total / 4 secondary -> 3 primary; legacy min 4 would conflict, but an
    # explicit max-total budget (without explicit min-total) must override it.
    args = _parsed(BASE + ["--max-total-windows", "12"])
    axis_min, axis_max, _ = windows._axis_count_bounds_from_total_window_budget(
        args, n_secondary=4, legacy_min_axis=4, legacy_max_axis=12, label="contact"
    )
    assert (axis_min, axis_max) == (3, 3)


def test_explicit_min_total_still_validates_conflict():
    # If the user sets an explicit min-total that genuinely exceeds the max-total
    # capacity, that is a real error and must still raise.
    args = _parsed(BASE + ["--max-total-windows", "8", "--min-total-windows", "20"])
    import pytest
    with pytest.raises(ValueError):
        windows._axis_count_bounds_from_total_window_budget(
            args, n_secondary=4, legacy_min_axis=4, legacy_max_axis=12, label="contact"
        )


def test_sparse_patches_respect_total_budget():
    # The total-window budget must bound the FINAL count (base grid + sparse
    # patches), not just the factorized base grid. 10 base + 4 candidate patches,
    # cap 12 -> at most 2 patches survive.
    from gareus import adaptive_feedback as af
    args = _parsed(BASE + ["--max-total-windows", "12"])
    args.adaptive_2d_max_local_patches = 32  # patch-count cap must NOT be the binding one
    base_primary = [0.0, 0.1, 0.2, 0.3, 0.4]
    base_secondary = [-1.0, 1.0]
    base_grid_rows = [
        {"primary_cv_center": pc, "primary_cv_k_kcal": 50.0,
         "secondary_cv_center": sc, "secondary_cv_k_kcal_mol": 25.0}
        for pc in base_primary for sc in base_secondary
    ]
    assert len(base_grid_rows) == 10
    defects = [
        {"local_patch_primary_cv_center": pc, "local_patch_secondary_cv_center": 0.0,
         "edge_status": "low_overlap_low_exchange", "decision_overlap": 0.0,
         "target_overlap": 0.25, "exchange_acceptance": 0.0}
        for pc in (0.05, 0.15, 0.25, 0.35)
    ]
    patch_rows, explicit_rows, _skipped = af._adaptive_feedback_2d_sparse_patch_candidates(
        defects, base_grid_rows, base_primary, [50.0] * len(base_primary),
        base_secondary, [25.0] * len(base_secondary), args,
    )
    assert len(patch_rows) <= max(0, 12 - len(base_grid_rows))  # <= 2
    assert len(explicit_rows) <= 12
