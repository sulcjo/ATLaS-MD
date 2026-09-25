import pytest

from gareus.adaptive.throughput import node_ns_per_day, wall_hours

TABLE = ((16.0, 3154.0), (59.0, 2300.0))


def test_interpolates_between_measured_points():
    assert node_ns_per_day(64, 4, TABLE) == pytest.approx(3154.0)          # 16/GPU
    assert node_ns_per_day(236, 4, TABLE) == pytest.approx(2300.0)         # 59/GPU
    mid = node_ns_per_day(4 * 37, 4, TABLE)                                 # 37/GPU
    assert 2300.0 < mid < 3154.0


def test_is_flat_outside_the_measured_range():
    assert node_ns_per_day(8, 4, TABLE) == pytest.approx(3154.0)            # 2/GPU: no extrapolated gain
    assert node_ns_per_day(400, 4, TABLE) == pytest.approx(2300.0)


def test_wall_hours_scales_with_steps_and_states():
    # 236 states x 1e6 steps x 4 fs = 944 ns aggregate at 2300 ns/day -> 9.85 h
    assert wall_hours(1_000_000, 236, 4.0, 4, TABLE) == pytest.approx(944.0 / 2300.0 * 24.0)
    # the same per-state ns on a 64-state patch costs less wall time per state-ns
    per_state_small = wall_hours(1_000_000, 64, 4.0, 4, TABLE) / 64
    per_state_full = wall_hours(1_000_000, 236, 4.0, 4, TABLE) / 236
    assert per_state_small < per_state_full


def test_rejects_an_empty_table():
    with pytest.raises(ValueError):
        node_ns_per_day(10, 4, ())
