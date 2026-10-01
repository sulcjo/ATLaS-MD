"""The US starting-structure gate never fails a window on its distance from a k = 0 placeholder centre."""
from gareus.seeding import classify_primary_start_delta


def test_unrestrained_primary_axis_is_never_judged_on_its_placeholder_distance():
    assert classify_primary_start_delta(0.38, 0.0, 0.15, 0.30) is None
    assert classify_primary_start_delta(0.38, -1.0, 0.15, 0.30) is None
    assert classify_primary_start_delta(0.38, float("nan"), 0.15, 0.30) is None


def test_restrained_primary_axis_keeps_the_old_thresholds():
    assert classify_primary_start_delta(0.38, 349.8, 0.15, 0.30) == "bad"
    assert classify_primary_start_delta(0.20, 349.8, 0.15, 0.30) == "warn"
    assert classify_primary_start_delta(0.10, 349.8, 0.15, 0.30) is None
    assert classify_primary_start_delta(float("nan"), 349.8, 0.15, 0.30) is None
