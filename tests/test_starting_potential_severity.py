"""A physically impossible starting energy must be droppable, not just noted.

Real incident (chignolin_6 final phase, 2026-09-01): windows 5 and 6 entered
production at +2.51e17 kJ/mol against a -267,632 kJ/mol median (MAD 1,831) --
a catastrophic steric clash. The quality check DID measure it, reporting
``potential energy robust-z 92499365528798.6``, but the potential-energy branch
had only one severity tier: ``warn``, at any magnitude. So
``--us-auto-drop-bad-windows`` could not drop them, they NaN'd 200 steps into
the final phase, and all 36 replicas died with
``OpenMMException: Particle coordinate is NaN``. The chain stopped and the
campaign was lost.

Meanwhile window 32 -- which had the *most negative*, i.e. most stable, energy
of all 37 starts -- was the one window classified ``bad`` (on unrelated CV/bias
criteria) and dropped.
"""
from __future__ import annotations

import math

from gareus.seeding import (
    STARTING_PE_BAD_Z,
    STARTING_PE_WARN_Z,
    classify_starting_potential_energy,
)

# The real distribution from the incident.
MED = -267632.0
MAD = 1830.94


def _status(pe, med=MED, mad=MAD):
    return classify_starting_potential_energy(pe, med, mad)[1]


def test_the_incident_energy_is_bad_not_warn():
    """+2.51e17 kJ/mol against a -2.7e5 median must be droppable."""
    z, status, msg = classify_starting_potential_energy(2.51094e17, MED, MAD)
    assert status == "bad", f"got {status!r} for the energy that killed the run"
    assert msg and "positive" in msg.lower()


def test_a_non_finite_starting_energy_is_bad():
    """Previously this fell through and recorded nothing at all -- silent."""
    for pe in (float("nan"), float("inf")):
        z, status, msg = classify_starting_potential_energy(pe, MED, MAD)
        assert status == "bad", f"{pe} classified {status!r}"
        assert msg


def test_a_hugely_strained_but_still_negative_start_is_bad():
    """Upward deviation past the bad tier, without needing a positive total."""
    pe = MED + (STARTING_PE_BAD_Z + 5.0) * 1.4826 * MAD
    assert pe < 0.0, "fixture must stay negative so it tests the z tier, not the sign"
    assert _status(pe) == "bad"


def test_an_ordinary_start_is_clean():
    assert _status(MED) is None
    assert _status(MED + 2.0 * 1.4826 * MAD) is None


def test_the_historical_warn_tier_is_unchanged():
    """Between warn_z and bad_z stays a warning, exactly as before."""
    pe = MED + (STARTING_PE_WARN_Z + 1.0) * 1.4826 * MAD
    assert _status(pe) == "warn"


def test_an_unusually_stable_start_is_never_bad():
    """Window 32's shape: the most negative energy in the set.

    Downward deviation means a more stable start. It must never be droppable,
    however far out it sits -- only upward deviation is a hazard.
    """
    for z in (10.0, 23.2, 1000.0, 1e13):
        pe = MED - z * 1.4826 * MAD
        assert _status(pe) != "bad", f"stable start at -{z} z was classified bad"


def test_the_stable_tail_still_warns_so_no_existing_warning_is_lost():
    """w16/w32/w34 sat at |z| ~= 23 below the median and warned; keep that."""
    pe = MED - 23.2 * 1.4826 * MAD
    assert _status(pe) == "warn"


def test_a_degenerate_spread_does_not_invent_a_z_score():
    """All starts identical: MAD 0. No z, and nothing to escalate on."""
    z, status, msg = classify_starting_potential_energy(-267632.0, MED, 0.0)
    assert z == ""
    assert status is None


def test_a_degenerate_spread_still_catches_the_physical_checks():
    """Sign and finiteness do not depend on the spread being measurable."""
    assert classify_starting_potential_energy(2.51094e17, MED, 0.0)[1] == "bad"
    assert classify_starting_potential_energy(float("nan"), MED, 0.0)[1] == "bad"


def test_the_reported_z_stays_the_two_sided_magnitude():
    """Back-compat: `potential_robust_z` in the artifact is |z|, as before."""
    z_up, _, _ = classify_starting_potential_energy(MED + 9.0 * 1.4826 * MAD, MED, MAD)
    z_dn, _, _ = classify_starting_potential_energy(MED - 9.0 * 1.4826 * MAD, MED, MAD)
    assert math.isclose(z_up, 9.0, rel_tol=1e-9)
    assert math.isclose(z_dn, 9.0, rel_tol=1e-9)
