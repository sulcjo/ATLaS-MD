"""F03 geometry acceptance for gareus.solvent_repair (spec F03; findings N01, N02).

Every RESOLVED claim in these tests is re-checked with the brute-force all-pairs oracle,
never taken from the solver's own bookkeeping.
"""
from __future__ import annotations

import numpy as np
import pytest

from fixtures.thermodynamic_repair import oscillating_displacement_case, solvent_collision_case
from gareus.solvent_repair import (STATUS_INVALID_INPUT, STATUS_RESOLVED, STATUS_UNRESOLVED, RepairPolicy,
                                   _CellList, brute_force_clashes, min_image, repair_solvent_clashes,
                                   validate_reduced_box)

POLICY = RepairPolicy()


def _tip3p(origin):
    return np.array([[0.0, 0.0, 0.0], [0.0, 0.0957, 0.0], [0.0, -0.024, 0.0927]]) + np.asarray(origin, dtype=float)


def _rigid_preserved(before, after, groups):
    for g in groups:
        db = np.linalg.norm(before[g][1:] - before[g][0], axis=1)
        da = np.linalg.norm(after[g][1:] - after[g][0], axis=1)
        np.testing.assert_allclose(da, db, atol=1e-12)


def test_two_waters_between_two_peptide_atoms_do_not_end_coincident():
    c = solvent_collision_case()
    res = repair_solvent_clashes(c.positions_nm, c.box_nm, c.peptide, c.groups, c.heavy, seed=3)
    assert res.status == STATUS_RESOLVED, res.as_record()
    oo = np.linalg.norm(min_image(res.positions_nm[2] - res.positions_nm[5], c.box_nm))
    assert oo >= POLICY.solvent_heavy_heavy_nm - 1e-9
    assert brute_force_clashes(res.positions_nm, c.box_nm, c.peptide, c.groups, c.heavy) == (0, 0)
    np.testing.assert_array_equal(res.positions_nm[c.peptide], c.positions_nm[c.peptide])
    _rigid_preserved(c.positions_nm, res.positions_nm, c.groups)
    assert res.n_solvent_solvent_clashes == 0 and res.n_peptide_solvent_clashes == 0
    assert res.minimum_clearance_by_pair_class["solvent_heavy_heavy"] >= POLICY.solvent_heavy_heavy_nm - 1e-9


def test_two_obstacle_geometry_escapes_non_radially_or_reports_unresolved():
    c = oscillating_displacement_case()
    res = repair_solvent_clashes(c.positions_nm, c.box_nm, c.peptide, c.groups, c.heavy)
    if res.status == STATUS_RESOLVED:
        # independently verified: clear of BOTH obstacles (the old solver oscillated between them
        # and returned at 0.05 nm; any real escape -- past the pair or sideways -- is acceptable)
        assert brute_force_clashes(res.positions_nm, c.box_nm, c.peptide, c.groups, c.heavy) == (0, 0)
        assert res.minimum_clearance_by_pair_class["peptide_heavy_heavy"] >= c.required_nm - 1e-9
        d = np.linalg.norm(min_image(res.positions_nm[2] - c.positions_nm[:2], c.box_nm), axis=1)
        assert d.min() >= c.required_nm - 1e-9
    else:
        assert res.status == STATUS_UNRESOLVED
        assert res.n_peptide_solvent_clashes >= 1
        assert res.reason_code.startswith("exhausted") or res.reason_code.startswith("unresolved")
    assert res.n_rounds <= POLICY.max_rounds


def test_no_clash_means_no_change_and_resolved():
    pos = np.vstack([[0.0, 0.0, 0.0], _tip3p([1.0, 0.0, 0.0])])
    res = repair_solvent_clashes(pos, np.eye(3) * 4.0, [0], [np.arange(1, 4)], np.array([1, 1, 0, 0], bool))
    assert res.status == STATUS_RESOLVED and res.n_groups_moved == 0
    np.testing.assert_array_equal(res.positions_nm, pos)


def test_exact_coincidence_gets_a_reproducible_direction():
    pos = np.vstack([[1.0, 1.0, 1.0], _tip3p([1.0, 1.0, 1.0])])
    heavy = np.array([1, 1, 0, 0], bool)
    r1 = repair_solvent_clashes(pos, np.eye(3) * 4.0, [0], [np.arange(1, 4)], heavy, seed=5)
    r2 = repair_solvent_clashes(pos, np.eye(3) * 4.0, [0], [np.arange(1, 4)], heavy, seed=5)
    assert r1.status == STATUS_RESOLVED
    np.testing.assert_array_equal(r1.positions_nm, r2.positions_nm)
    assert brute_force_clashes(r1.positions_nm, np.eye(3) * 4.0, [0], [np.arange(1, 4)], heavy) == (0, 0)


def test_clash_across_the_periodic_wall_is_seen_and_the_result_is_image_equivalent():
    L = 3.0
    box = np.eye(3) * L
    pep = np.array([[0.02, 1.5, 1.5]])
    water = _tip3p([2.99, 1.5, 1.5])                       # 0.03 nm from the peptide through the wall
    pos = np.vstack([pep, water]); heavy = np.array([1, 1, 0, 0], bool)
    res = repair_solvent_clashes(pos, box, [0], [np.arange(1, 4)], heavy)
    assert res.status == STATUS_RESOLVED
    assert brute_force_clashes(res.positions_nm, box, [0], [np.arange(1, 4)], heavy) == (0, 0)
    # the same system translated by a lattice vector repairs to the same relative geometry
    shifted = pos + np.array([L, 0.0, 0.0])
    res2 = repair_solvent_clashes(shifted, box, [0], [np.arange(1, 4)], heavy)
    d1 = min_image(res.positions_nm[1] - res.positions_nm[0], box)
    d2 = min_image(res2.positions_nm[1] - res2.positions_nm[0], box)
    np.testing.assert_allclose(d1, d2, atol=1e-9)


def test_translation_invariance_of_the_repair():
    c = solvent_collision_case()
    shift = np.array([0.37, -0.81, 1.23])
    r1 = repair_solvent_clashes(c.positions_nm, c.box_nm, c.peptide, c.groups, c.heavy, seed=1)
    r2 = repair_solvent_clashes(c.positions_nm + shift, c.box_nm, c.peptide, c.groups, c.heavy, seed=1)
    np.testing.assert_allclose(r2.positions_nm - shift, r1.positions_nm, atol=1e-9)


def test_skewed_reduced_box_uses_true_lattice_images():
    box = np.array([[3.0, 0.0, 0.0], [1.2, 3.0, 0.0], [-0.9, 1.1, 3.0]])
    validate_reduced_box(box)
    from gareus.solvent_repair import safe_radius_nm
    rng = np.random.default_rng(0)
    radius = safe_radius_nm(box)
    assert 0.5 < radius < 1.5
    # Within the safe radius the sequential min-image recovers the true short vector exactly,
    # whatever lattice image it was handed. That is the regime every clearance check runs in;
    # the repair refuses boxes whose safe radius is below its cutoff.
    short = rng.normal(size=(200, 3)); short *= (rng.uniform(0.01, 0.95 * radius, size=200) / np.linalg.norm(short, axis=1))[:, None]
    shifts = rng.integers(-2, 3, size=(200, 3)) @ box
    m = min_image(short + shifts, box)
    np.testing.assert_allclose(m, short, atol=1e-9)
    # a peptide-water clash through the skewed wall is repaired
    pep = np.array([[0.01, 0.01, 0.01]]); water = _tip3p(box.sum(0) - np.array([0.02, 0.02, 0.02]))
    pos = np.vstack([pep, water]); heavy = np.array([1, 1, 0, 0], bool)
    res = repair_solvent_clashes(pos, box, [0], [np.arange(1, 4)], heavy)
    assert res.status == STATUS_RESOLVED
    assert brute_force_clashes(res.positions_nm, box, [0], [np.arange(1, 4)], heavy) == (0, 0)


def test_unreduced_or_invalid_boxes_and_bad_groups_are_invalid_input():
    pos = np.vstack([[0.0, 0.0, 0.0], _tip3p([0.05, 0.0, 0.0])]); heavy = np.array([1, 1, 0, 0], bool)
    bad_box = np.array([[3.0, 0.0, 0.0], [2.9, 3.0, 0.0], [0.0, 0.0, 3.0]])     # b_x > a/2
    assert repair_solvent_clashes(pos, bad_box, [0], [np.arange(1, 4)], heavy).status == STATUS_INVALID_INPUT
    assert repair_solvent_clashes(pos, np.eye(3) * 4.0, [0], [np.arange(0, 3)], heavy).status == STATUS_INVALID_INPUT  # overlaps peptide
    nan_pos = pos.copy(); nan_pos[2, 0] = np.nan
    assert repair_solvent_clashes(nan_pos, np.eye(3) * 4.0, [0], [np.arange(1, 4)], heavy).status == STATUS_INVALID_INPUT
    assert repair_solvent_clashes(pos, np.eye(3) * 4.0, [0], [np.array([1, 2, 9])], heavy).status == STATUS_INVALID_INPUT


def test_cell_list_neighbour_search_agrees_with_brute_force_on_random_boxes():
    rng = np.random.default_rng(42)
    for trial in range(5):
        box = np.eye(3) * rng.uniform(1.5, 3.0)
        n = 60
        pts = rng.uniform(0, box[0, 0], size=(n, 3))
        cells = _CellList(pts, box, 0.3)
        q = rng.uniform(0, box[0, 0], size=(4, 3))
        near = set(cells.neighbours(q).tolist())
        d = np.linalg.norm(min_image(q[:, None, :] - pts[None, :, :], box), axis=-1)
        within = set(np.where((d < 0.3).any(axis=0))[0].tolist())
        assert within <= near, "cell list missed a neighbour inside the cutoff"


def test_massless_virtual_sites_ride_along_but_do_not_count():
    # a 4-site water: O, H, H, M(virtual). Its M sits inside the peptide clearance while O/H are clear.
    pep = np.array([[0.0, 0.0, 0.0]])
    water = np.vstack([_tip3p([0.30, 0.0, 0.0]), [[0.10, 0.0, 0.0]]])
    pos = np.vstack([pep, water]); heavy = np.array([1, 1, 0, 0, 0], bool)
    massless = np.array([0, 0, 0, 0, 1], bool)
    res = repair_solvent_clashes(pos, np.eye(3) * 4.0, [0], [np.arange(1, 5)], heavy, massless_mask=massless)
    assert res.status == STATUS_RESOLVED and res.n_groups_moved == 0
    np.testing.assert_array_equal(res.positions_nm, pos)


def test_result_record_is_json_ready_and_names_the_policy():
    c = solvent_collision_case()
    res = repair_solvent_clashes(c.positions_nm, c.box_nm, c.peptide, c.groups, c.heavy, seed_id="seed_07546", seed=9)
    rec = res.as_record()
    assert rec["repair_policy_version"] == POLICY.version and rec["seed_id"] == "seed_07546" and rec["repair_seed"] == 9
    assert set(rec["minimum_clearance_by_pair_class"]) == {"peptide_heavy_heavy", "peptide_hydrogen",
                                                           "solvent_heavy_heavy", "solvent_hydrogen"}
    import json
    json.dumps(rec)
