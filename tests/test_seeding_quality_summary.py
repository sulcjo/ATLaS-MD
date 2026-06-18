import math

from gareus.seeding import _safe_max_finite


def test_empty_strings_treated_as_missing():
    # cv2:none runs write "" for the secondary-CV bias; must not raise.
    rows = [
        {"k": ""},
        {"k": ""},
    ]
    assert math.isnan(_safe_max_finite(rows, "k"))


def test_max_over_finite_values_ignores_blanks_and_none():
    rows = [
        {"k": 1.5},
        {"k": ""},
        {"k": None},
        {"k": 3.25},
        {"k": float("nan")},
    ]
    assert _safe_max_finite(rows, "k") == 3.25


def test_missing_key_is_nan():
    assert math.isnan(_safe_max_finite([{"other": 1.0}], "k"))


def test_empty_rows_is_nan():
    assert math.isnan(_safe_max_finite([], "k"))
