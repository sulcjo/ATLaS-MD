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


from gareus.gamd_calibration import (
    PooledEnvelope,
    lower_bound_threshold_and_k0,
    upper_bound_threshold_and_k0,
    threshold_and_k0,
    compute_group_calibration,
    overwrite_physics_globals,
)


def test_lower_bound_threshold_and_k0_matches_hand_computed_values():
    # Vmax=100, Vmin=-200, Vavg=-50, sigmaV=40, sigma0=6 (kcal/mol-consistent units, formula is unit-agnostic)
    envelope = PooledEnvelope(group="NonBonded", vmax=100.0, vmin=-200.0, vavg=-50.0, sigmav=40.0, n_total=1000, n_windows=10)
    result = lower_bound_threshold_and_k0(envelope, sigma0=6.0)
    assert result.threshold_energy == 100.0
    # k0prime = (6/40) * (100 - -200) / (100 - -50) = 0.15 * 300/150 = 0.30
    assert result.k0 == pytest.approx(0.30)
    # k = k0 / (Vmax - Vmin) = 0.30 / 300
    assert result.k == pytest.approx(0.30 / 300.0)
    assert result.boosted is True


def test_lower_bound_threshold_and_k0_clips_at_one():
    envelope = PooledEnvelope(group="Dihedral", vmax=10.0, vmin=-10.0, vavg=-9.0, sigmav=1.0, n_total=100, n_windows=5)
    # k0prime = (6/1) * (20)/(19) = huge >> 1
    result = lower_bound_threshold_and_k0(envelope, sigma0=6.0)
    assert result.k0 == 1.0


def test_lower_bound_threshold_and_k0_degenerate_when_vmax_equals_vmin():
    envelope = PooledEnvelope(group="Total", vmax=5.0, vmin=5.0, vavg=5.0, sigmav=40.0, n_total=100, n_windows=5)
    result = lower_bound_threshold_and_k0(envelope, sigma0=6.0)
    assert result.boosted is False
    assert result.k0 == 0.0
    assert result.k == 0.0


def test_upper_bound_threshold_and_k0_matches_hand_computed_values():
    # Choose numbers giving k0doubleprime strictly inside (0, 1):
    # Vmax=100, Vmin=-200, Vavg=50, sigmaV=200, sigma0=6
    # k0'' = (1 - 6/200) * (100 - -200) / (50 - -200) = 0.97 * 300/250 = 1.164 -> outside (0,1) -> falls back
    # Pick sigma0 larger so (1 - sigma0/sigmaV) is smaller: sigma0=190
    envelope = PooledEnvelope(group="Total", vmax=100.0, vmin=-200.0, vavg=50.0, sigmav=200.0, n_total=500, n_windows=8)
    result = upper_bound_threshold_and_k0(envelope, sigma0=190.0)
    # k0'' = (1 - 190/200) * 300/250 = 0.05 * 1.2 = 0.06
    assert result.k0 == pytest.approx(0.06)
    # threshold_energy = Vmin + (Vmax-Vmin)/k0 = -200 + 300/0.06
    assert result.threshold_energy == pytest.approx(-200.0 + 300.0 / 0.06)
    assert result.k == pytest.approx(0.06 / 300.0)
    assert result.boosted is True


def test_upper_bound_threshold_and_k0_falls_back_to_lower_bound_formula():
    # k0doubleprime computed as >= 1 here (from the docstring-derived example above: 1.164)
    envelope = PooledEnvelope(group="Total", vmax=100.0, vmin=-200.0, vavg=50.0, sigmav=200.0, n_total=500, n_windows=8)
    upper_result = upper_bound_threshold_and_k0(envelope, sigma0=6.0)
    lower_result = lower_bound_threshold_and_k0(envelope, sigma0=6.0)
    assert upper_result.threshold_energy == pytest.approx(lower_result.threshold_energy)
    assert upper_result.k0 == pytest.approx(lower_result.k0)
    assert upper_result.k == pytest.approx(lower_result.k)


def test_threshold_and_k0_dispatches_by_prefix():
    envelope = PooledEnvelope(group="NonBonded", vmax=100.0, vmin=-200.0, vavg=-50.0, sigmav=40.0, n_total=1000, n_windows=10)
    lower = threshold_and_k0("lower-dual-nonbonded-dihedral", envelope, sigma0=6.0)
    upper = threshold_and_k0("upper-total", envelope, sigma0=6.0)
    assert lower.k0 == pytest.approx(0.30)
    assert isinstance(upper.k0, float)


def test_threshold_and_k0_rejects_unknown_prefix():
    envelope = PooledEnvelope(group="NonBonded", vmax=1.0, vmin=0.0, vavg=0.5, sigmav=0.1, n_total=10, n_windows=2)
    with pytest.raises(ValueError, match="Unknown gamd_boost_type"):
        threshold_and_k0("gamd-cmd-base", envelope, sigma0=6.0)


def test_compute_group_calibration_bundles_envelope_and_threshold():
    envelope = PooledEnvelope(group="NonBonded", vmax=100.0, vmin=-200.0, vavg=-50.0, sigmav=40.0, n_total=1000, n_windows=10)
    calib = compute_group_calibration("lower-dual-nonbonded-dihedral", envelope, sigma0=6.0)
    assert calib.group == "NonBonded"
    assert calib.vmax == 100.0
    assert calib.k0 == pytest.approx(0.30)


def test_overwrite_physics_globals_only_touches_matching_group_keys():
    globals_all = {
        "stepCount": 305000.0, "stage": 5.0, "windowCount": 0.0,
        "Vmax_NonBonded": 1.0, "Vmin_NonBonded": -1.0, "Vavg_NonBonded": 0.0, "sigmaV_NonBonded": 0.5,
        "k0_NonBonded": 0.1, "k_NonBonded": 0.001, "threshold_energy_NonBonded": 1.0,
        "sigma0_NonBonded": 6.0,
    }
    calib = compute_group_calibration("lower-nonbonded", PooledEnvelope(
        group="NonBonded", vmax=100.0, vmin=-200.0, vavg=-50.0, sigmav=40.0, n_total=1000, n_windows=10,
    ), sigma0=6.0)
    out = overwrite_physics_globals(globals_all, {"NonBonded": calib})
    assert out["Vmax_NonBonded"] == 100.0
    assert out["Vmin_NonBonded"] == -200.0
    assert out["Vavg_NonBonded"] == -50.0
    assert out["sigmaV_NonBonded"] == 40.0
    assert out["k0_NonBonded"] == pytest.approx(0.30)
    assert out["threshold_energy_NonBonded"] == 100.0
    # bookkeeping globals untouched
    assert out["stepCount"] == 305000.0
    assert out["stage"] == 5.0
    assert out["sigma0_NonBonded"] == 6.0
    # original dict not mutated
    assert globals_all["Vmax_NonBonded"] == 1.0


def test_overwrite_physics_globals_raises_when_no_keys_match():
    with pytest.raises(RuntimeError, match="matched no global names"):
        overwrite_physics_globals({"stepCount": 1.0}, {"NonBonded": compute_group_calibration(
            "lower-nonbonded",
            PooledEnvelope(group="NonBonded", vmax=1.0, vmin=0.0, vavg=0.5, sigmav=0.1, n_total=10, n_windows=2),
            sigma0=6.0,
        )})
