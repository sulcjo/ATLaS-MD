import math
import pytest

from gareus.gamd_calibration import WelfordAccumulator, WindowEnergyStats, pool_window_stats


def test_welford_accumulator_matches_known_mean_variance():
    acc = WelfordAccumulator()
    values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    for v in values:
        acc.update(v)
    assert acc.n == 8
    assert acc.mean == pytest.approx(5.0)
    assert acc.variance == pytest.approx(4.571428571, rel=1e-9)
    assert acc.vmax == 9.0
    assert acc.vmin == 2.0


def test_welford_accumulator_single_sample_has_zero_variance():
    acc = WelfordAccumulator()
    acc.update(3.5)
    assert acc.n == 1
    assert acc.variance == 0.0
    assert acc.vmax == acc.vmin == 3.5


def test_welford_to_stats_raises_on_no_samples():
    acc = WelfordAccumulator()
    with pytest.raises(ValueError, match="no samples"):
        acc.to_stats("NonBonded", window=0)


def test_pool_window_stats_joint_extrema_and_pooled_mean_variance():
    # Two windows, equal N: window 0 in [0, 10] mean 5 var 8.333..;
    # window 1 in [8, 20] mean 14 var 8.333... (both computed by hand below).
    w0 = WindowEnergyStats(group="NonBonded", window=0, vmax=10.0, vmin=0.0, mean=5.0, var=8.333333333333334, n=6)
    w1 = WindowEnergyStats(group="NonBonded", window=1, vmax=20.0, vmin=8.0, mean=14.0, var=8.333333333333334, n=6)
    pooled = pool_window_stats([w0, w1])
    assert pooled.group == "NonBonded"
    assert pooled.vmax == 20.0
    assert pooled.vmin == 0.0
    assert pooled.n_total == 12
    assert pooled.n_windows == 2
    # mu_pooled = (6*5 + 6*14) / 12 = 9.5
    assert pooled.vavg == pytest.approx(9.5)
    # M2 = (5*8.3333333)*2 + 6*(5-9.5)^2 + 6*(14-9.5)^2 = 83.33333 + 121.5 + 121.5 = 326.33333
    # sigma_pooled = sqrt(326.33333 / 11)
    assert pooled.sigmav == pytest.approx(math.sqrt(326.33333333333337 / 11), rel=1e-9)


def test_pool_window_stats_rejects_mixed_groups():
    w0 = WindowEnergyStats(group="NonBonded", window=0, vmax=1.0, vmin=0.0, mean=0.5, var=0.1, n=4)
    w1 = WindowEnergyStats(group="Dihedral", window=1, vmax=1.0, vmin=0.0, mean=0.5, var=0.1, n=4)
    with pytest.raises(ValueError, match="single boost group"):
        pool_window_stats([w0, w1])


def test_pool_window_stats_rejects_empty_input():
    with pytest.raises(ValueError, match="at least one window"):
        pool_window_stats([])
