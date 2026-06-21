from gareus.adaptive_feedback import _minimum_valid_overlap


def test_floor_hits_point_two_at_default_target():
    # default measured target_overlap = 0.30 -> band value 0.20
    assert abs(_minimum_valid_overlap(0.30) - 0.20) < 1e-9


def test_hard_floor_never_below_point_one():
    # tiny target must not drop below the 0.10 hard floor
    assert _minimum_valid_overlap(0.05) == 0.10


def test_scales_up_to_ceiling():
    # large target saturates at the 0.25 ceiling
    assert abs(_minimum_valid_overlap(0.60) - 0.25) < 1e-9


def test_aggressive_override_floor():
    # aggressive modes may lower the floor explicitly
    val = _minimum_valid_overlap(0.30, {"minimum_valid_overlap_floor": 0.12,
                                         "minimum_valid_overlap_ceiling": 0.25,
                                         "minimum_valid_overlap_factor": 0.67})
    assert abs(val - 0.201) < 1e-3
