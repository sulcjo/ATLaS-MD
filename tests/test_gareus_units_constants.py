import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gareus.units import (
    KJ_PER_KCAL,
    K_B_KJ_PER_MOL_K,
    kcal_a2_to_kj_nm2,
    kj_nm2_to_kcal_a2,
    kcal_to_kj,
    bias_energy_kj,
)


def test_kj_per_kcal_is_the_exact_conversion_factor():
    assert KJ_PER_KCAL == 4.184


def test_k_b_matches_codata_value():
    assert K_B_KJ_PER_MOL_K == 0.00831446261815324


def test_k_b_and_kj_per_kcal_round_trip_kbt_at_300k():
    # Same check this codebase already validated numerically elsewhere this
    # session: at T=300K, kT should be ~0.596 kcal/mol.
    beta = 1.0 / (K_B_KJ_PER_MOL_K * 300.0)
    kbt_kj = 1.0 / beta
    kbt_kcal = kbt_kj / KJ_PER_KCAL
    assert abs(kbt_kcal - 0.5961612775922496) < 1e-9


def test_conversion_functions_unchanged_after_refactor():
    # Regression guard: refactoring these to reference the named constant
    # instead of an inline literal must not change any output.
    assert kcal_a2_to_kj_nm2(5.0) == 5.0 * 4.184 / 0.01
    assert kj_nm2_to_kcal_a2(2092.0) == 2092.0 * 0.01 / 4.184
    assert kcal_to_kj(1.0) == 4.184
    assert bias_energy_kj(2.5) == kcal_to_kj(2.5)
