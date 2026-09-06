"""Connectivity of the union ladder MBAR is finally handed.

Includes the negative result that defines this module's scope: run on the
chignolin_6 registry it PASSES, because that campaign's split was a CV2
definition mismatch rather than a geometric gap. That case is pinned here so a
later reader does not mistake a pass for proof that a pooled solve is sound.
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gareus.ladder_connectivity import (  # noqa: E402
    assess_ladder_connectivity,
    format_connectivity_report,
    pair_overlap,
)

RT = 0.00198720425864083 * 300.0


def _grid(c1, s1, c2, s2, tag=""):
    out = []
    for i, a in enumerate(c1):
        for j, b in enumerate(c2):
            out.append({"state_id": f"{tag}{i}_{j}",
                        "primary_mean": float(a), "primary_sigma": s1,
                        "secondary_mean": float(b), "secondary_sigma": s2})
    return out


# --- the physics the check rests on ----------------------------------------

def test_axis_overlap_matches_the_one_dimensional_form():
    """A pair differing on a single axis must reproduce erfc(d/(2*sqrt(2)))."""
    sig = 0.04
    for dsig in (1.0, 1.5, 2.0, 3.0):
        d = dsig * sig
        got = pair_overlap([0.0, 0.0], [sig, sig], [d, 0.0], [sig, sig], RT)
        assert got == pytest.approx(math.erfc(dsig / (2 * math.sqrt(2))), rel=1e-9)


def test_axes_combine_as_a_product():
    sig = 0.04
    d = 1.5 * sig
    one = pair_overlap([0, 0], [sig, sig], [d, 0], [sig, sig], RT)
    both = pair_overlap([0, 0], [sig, sig], [d, d], [sig, sig], RT)
    assert both == pytest.approx(one * one, rel=1e-9)


def test_an_unrestrained_axis_contributes_no_separation():
    ov = pair_overlap([0.0, 0.0], [0.05, 0.0], [0.0, 99.0], [0.05, 0.0], RT)
    assert ov == pytest.approx(1.0)


# --- the intended use: validating a designed ladder ------------------------

def test_the_proposed_16x8_ladder_is_comfortably_connected():
    """CV1 16 windows over 0.043-0.581 at sigma 0.0239; CV2 8 windows over
    -1.07..2.25 at sigma 0.316 -- both at 1.5 sigma spacing."""
    res = assess_ladder_connectivity(
        _grid(np.linspace(0.043, 0.581, 16), 0.0239,
              np.linspace(-1.07, 2.25, 8), 0.316), 300.0)
    assert res["connected"]
    assert res["n_states"] == 128
    # every window has an axis-adjacent neighbour at 1.5 sigma, and
    # erfc(1.5/(2*sqrt(2))) = 0.453 -- so the bottleneck is that, not the floor
    assert res["weakest_link"] == pytest.approx(
        math.erfc(1.5 / (2 * math.sqrt(2))), rel=0.02)
    assert res["weakest_link"] > 15 * res["floor"]


def test_a_genuine_spacing_blunder_is_caught():
    """The failure this module exists for: an axis spaced so far that no pair of
    windows across the gap can be stitched."""
    lo = _grid(np.linspace(0.0, 0.1, 3), 0.02, [0.0], 1.0, tag="lo")
    hi = _grid(np.linspace(1.0, 1.1, 3), 0.02, [0.0], 1.0, tag="hi")
    res = assess_ladder_connectivity(lo + hi, 300.0)
    assert not res["connected"]
    assert res["n_components"] == 2
    assert sorted(len(c) for c in res["components"]) == [3, 3]
    assert "DISCONNECTED" in format_connectivity_report(res)


def test_too_stiff_for_the_spacing_disconnects():
    """Raising k without re-spacing is how a redesign breaks a ladder."""
    ok = assess_ladder_connectivity(
        _grid(np.linspace(0.0, 0.5, 8), 0.030, [0.0], 1.0), 300.0)
    stiff = assess_ladder_connectivity(
        _grid(np.linspace(0.0, 0.5, 8), 0.008, [0.0], 1.0), 300.0)
    assert ok["connected"]
    assert not stiff["connected"]


def test_measured_widths_are_preferred_over_nominal():
    nominal = [{"state_id": i, "primary_center": float(c), "primary_k": 200.0,
                "secondary_center": 0.0, "secondary_k": 0.0}
               for i, c in enumerate(np.linspace(0.0, 0.5, 6))]
    res_nom = assess_ladder_connectivity(nominal, 300.0)
    assert not res_nom["used_measured"]
    # the same ladder, but the windows actually sampled far narrower
    measured = [dict(r, primary_sigma=0.004) for r in nominal]
    res_meas = assess_ladder_connectivity(measured, 300.0)
    assert res_meas["used_measured"]
    assert res_nom["connected"] and not res_meas["connected"], (
        "narrow realised widths must be able to disconnect a nominally fine ladder")


def test_report_names_the_basis_it_used():
    nominal = [{"state_id": i, "primary_center": float(c), "primary_k": 200.0}
               for i, c in enumerate(np.linspace(0.0, 0.3, 5))]
    assert "NOMINAL" in format_connectivity_report(
        assess_ladder_connectivity(nominal, 300.0))
    measured = [dict(r, primary_sigma=0.05) for r in nominal]
    assert "measured" in format_connectivity_report(
        assess_ladder_connectivity(measured, 300.0))


# --- the documented blind spot ---------------------------------------------

_REG = Path('/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/'
            'chignolin_6/adaptive_production/final_registry_used_for_mbar.csv')


@pytest.mark.skipif(not _REG.exists(), reason='chignolin_6 registry not on this machine')
def test_chignolin_6_passes_because_its_split_was_not_geometric():
    """Pinned negative result. chignolin_6's overlap graph demonstrably split,
    yet this check reports one component: its CV2 was redefined mid-campaign, so
    the two groups' centres are numbers on different axes and the separation is
    in the units, not the distance. A pass here is not evidence that a pooled
    solve is sound."""
    rows = []
    for r in csv.DictReader(open(_REG)):
        if str(r.get('usable_for_mbar', 'True')).lower() not in ('true', '1'):
            continue
        rows.append({'state_id': int(r['state_id']),
                     'primary_center': float(r['primary_center']),
                     'primary_k': float(r['primary_k']),
                     'secondary_center': float(r['secondary_center']),
                     'secondary_k': float(r['secondary_k'])})
    res = assess_ladder_connectivity(rows, 300.0)
    assert res["connected"], 'documented blind spot: geometry alone does not see it'
    # and its most isolated window is far worse attached than the proposed
    # ladder's, so even the geometry was marginal
    assert res["weakest_link"] < 0.30
