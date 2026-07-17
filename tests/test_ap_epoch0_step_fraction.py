"""Tests for --ap-epoch0-step-fraction.

`adaptive_production_epoch0_step_fraction` used to be hardcoded to 0.5 in
`gareus/cli.py`'s `_apply_v2_compat_shims`, with no argparse flag reaching it --
so there was no config-only way to get epoch 0 and epoch 1 an equal MD-pool
share. With `ap_epochs=2`, `_epoch0_scaled_steps` (gareus/adaptive_production.py)
gives epoch 0 exactly 1/4 and epoch 1 exactly 3/4 of the epoch-portion budget
(epoch 0's unused half is redistributed to epoch 1, which then recomputes its
share against the pool's fresh remaining_ns). Setting the fraction to 1.0
disables that redistribution and gives an even 1:1 split.

No OpenMM, PeptideBuilder, or gamd-openmm imports.
"""
from __future__ import annotations

from gareus.adaptive_production import _epoch0_scaled_steps


def test_ap_epoch0_step_fraction_defaults_to_half():
    from gareus.cli import parse_args

    args = parse_args(["--seq", "DPETG"])
    assert args.ap_epoch0_step_fraction == 0.5
    assert args.adaptive_production_epoch0_step_fraction == 0.5


def test_ap_epoch0_step_fraction_is_overridable():
    from gareus.cli import parse_args

    args = parse_args(["--seq", "DPETG", "--ap-epoch0-step-fraction", "1.0"])
    assert args.ap_epoch0_step_fraction == 1.0
    assert args.adaptive_production_epoch0_step_fraction == 1.0


def test_epoch0_scaled_steps_default_gives_one_to_three_split_for_two_epochs():
    """Reproduces the exact epoch0:epoch1 ratio the pool's live recompute
    produces for ap_epochs=2 at the default fraction=0.5: epoch 0 consumes
    E/4, epoch 1's fresh share against the remaining pool is 3E/4."""
    epoch_portion = 1200  # arbitrary "fair share per epoch" unit before scaling
    fraction = 0.5

    epoch0_actual = _epoch0_scaled_steps(epoch_portion, epoch=0, fraction=fraction)
    assert epoch0_actual == round(epoch_portion * 0.5)

    # Epoch 1 is not epoch 0, so _epoch0_scaled_steps is a no-op for it --
    # its actual share comes from the pool recomputing against what's left,
    # which is the remaining 3/4 of the total epoch-portion budget.
    total_epoch_portion = epoch_portion * 2
    epoch1_share_from_pool = total_epoch_portion - epoch0_actual
    epoch1_actual = _epoch0_scaled_steps(epoch1_share_from_pool, epoch=1, fraction=fraction)

    assert epoch0_actual == total_epoch_portion // 4
    assert epoch1_actual == 3 * total_epoch_portion // 4
    assert epoch0_actual + epoch1_actual == total_epoch_portion


def test_epoch0_scaled_steps_fraction_one_gives_even_split():
    epoch_portion = 1200
    epoch0_actual = _epoch0_scaled_steps(epoch_portion, epoch=0, fraction=1.0)
    assert epoch0_actual == epoch_portion  # unscaled -- epoch 0 keeps its full fair share

    total_epoch_portion = epoch_portion * 2
    epoch1_share_from_pool = total_epoch_portion - epoch0_actual
    epoch1_actual = _epoch0_scaled_steps(epoch1_share_from_pool, epoch=1, fraction=1.0)

    assert epoch0_actual == epoch1_actual == epoch_portion
