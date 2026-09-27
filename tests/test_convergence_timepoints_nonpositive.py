"""--convergence-timepoints <= 0 must disable convergence, not crash.

chignolin_9 rerun with --convergence-timepoints 0 crashed in
checkpoint_steps_from_data (np.linspace(..., 0) -> empty idx -> idx[-1]
IndexError) after every earlier stage had finished.
"""
import numpy as np
import pytest

import analyze_gareus_mbar as agm


@pytest.mark.parametrize("n", ["0", "-3"])
def test_nonpositive_timepoints_disable_convergence(n):
    args = agm.parse_args(["some_run", "--convergence-timepoints", n])
    assert args.no_convergence is True


def test_positive_timepoints_keep_convergence_enabled():
    args = agm.parse_args(["some_run", "--convergence-timepoints", "5"])
    assert args.no_convergence is False


@pytest.mark.parametrize("n", [0, -1])
def test_checkpoint_steps_nonpositive_returns_empty(n):
    steps = agm.checkpoint_steps_from_data(np.arange(100), n)
    assert steps.size == 0
