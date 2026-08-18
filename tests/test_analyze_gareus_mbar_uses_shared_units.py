import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gareus.units
import analyze_gareus_mbar as agm


def test_analyze_gareus_mbar_kj_per_kcal_is_the_shared_constant():
    # `is`, not `==` -- proves the import actually replaced the local
    # definition rather than merely producing a coincidentally-equal copy.
    assert agm.KJ_PER_KCAL is gareus.units.KJ_PER_KCAL


def test_analyze_gareus_mbar_k_b_is_the_shared_constant():
    assert agm.K_B_KJ_PER_MOL_K is gareus.units.K_B_KJ_PER_MOL_K


def test_values_are_unchanged():
    assert agm.KJ_PER_KCAL == 4.184
    assert agm.K_B_KJ_PER_MOL_K == 0.00831446261815324
