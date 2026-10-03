"""--swarm-lambda-max caps the swarm-designed lambda ladder below full boost.

chignolin_10 epoch_000 (2026-10-03): the forced lambda = 1 rung boosted LESS than lambda = 0.42
(mean 3.3 vs 3.6 kcal/mol) at reweighting ESS 29, while its 53 states were what the CV2 layout
lacked under the 236-replica cap.
"""
from __future__ import annotations

import numpy as np
import pytest

from gareus.cli import parse_args
from gareus.swarm.ladder_design import design_lambda_ladder


def _dv(seed=0, n=4000):
    return np.random.default_rng(seed).normal(40.0, 8.0, n)


def test_default_still_ends_at_full_boost():
    out = design_lambda_ladder(_dv(), 300.0, target_beta_sigma=2.5, min_rungs=4, max_rungs=4)
    assert out["lambdas"][0] == 0.0 and out["lambdas"][-1] == 1.0
    assert out["lambda_max"] == 1.0


def test_default_matches_explicit_lambda_max_one():
    a = design_lambda_ladder(_dv(), 300.0, target_beta_sigma=1.0, min_rungs=3, max_rungs=12)
    b = design_lambda_ladder(_dv(), 300.0, target_beta_sigma=1.0, min_rungs=3, max_rungs=12, lambda_max=1.0)
    assert a == b


@pytest.mark.parametrize("lam_max,n_rungs", [(0.5, 3), (0.45, 3), (0.3, 4)])
def test_top_rung_is_lambda_max(lam_max, n_rungs):
    out = design_lambda_ladder(_dv(), 300.0, target_beta_sigma=2.5, min_rungs=n_rungs, max_rungs=n_rungs,
                               lambda_max=lam_max)
    lams = out["lambdas"]
    assert len(lams) == n_rungs
    assert lams[0] == 0.0 and lams[-1] == pytest.approx(lam_max)
    assert all(b > a for a, b in zip(lams, lams[1:]))
    assert out["lambda_max"] == pytest.approx(lam_max)


def test_spacing_below_the_cap_is_unchanged():
    """Rungs the growth reaches before the cap are the same as without a cap."""
    full = design_lambda_ladder(_dv(), 300.0, target_beta_sigma=1.0, min_rungs=3, max_rungs=12)["lambdas"]
    capped = design_lambda_ladder(_dv(), 300.0, target_beta_sigma=1.0, min_rungs=3, max_rungs=12,
                                  lambda_max=0.5)["lambdas"]
    below = [x for x in full if x < 0.5]
    assert capped[:len(below)] == below
    assert capped[-1] == 0.5


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5, float("nan")])
def test_invalid_lambda_max_refused(bad):
    with pytest.raises(ValueError):
        design_lambda_ladder(_dv(), 300.0, lambda_max=bad)


def test_cli_flag_default_and_validation(capsys):
    args = parse_args(["--seq", "GYDPETGTWG", "--swarm-stage", "analyze", "--out", "/tmp/x"])
    assert args.swarm_lambda_max == 1.0
    args = parse_args(["--seq", "GYDPETGTWG", "--swarm-stage", "analyze", "--out", "/tmp/x", "--swarm-lambda-max", "0.5"])
    assert args.swarm_lambda_max == 0.5
    with pytest.raises(SystemExit):
        parse_args(["--seq", "GYDPETGTWG", "--swarm-stage", "analyze", "--out", "/tmp/x", "--swarm-lambda-max", "1.2"])
