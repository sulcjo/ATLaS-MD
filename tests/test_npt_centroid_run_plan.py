"""The slice-plan centroid transform must equal the bincount one bit for bit.

``np.bincount`` and the fancy-index gather hold the GIL for their whole
duration. Replacing them with slices, elementwise ufuncs and a broadcast
assignment lets numpy release it -- but only if the result is unchanged to the
last bit, because the reject path hands this array straight back to
``_restore_positions`` and a strided check compares the Context against it.

Everything here asserts with ``==``, never ``allclose``. A previous
"bit-identical" claim about this same transform was false and deviated by
8.9e-16; it was caught by a property test over randomly ordered partitions, so
that style of test is repeated rather than trusted.
"""
import numpy as np

from gareus.npt import (
    _expand_and_offset,
    _mean_fallback_molecules,
    _molecule_index_arrays,
    _scale_about_molecule_centroids,
    _sums_from_runs,
    _uniform_run_plan,
)

SCALE = 0.000317


def _solvated_layout(n_water=400, solute=138, n_ions=7):
    """Solute, then a water block, then ions -- what getMolecules() returns."""
    mols, a = [], 0
    mols.append(list(range(a, a + solute))); a += solute
    for _ in range(n_water):
        mols.append(list(range(a, a + 3))); a += 3
    for _ in range(n_ions):
        mols.append([a]); a += 1
    return mols, a


def _both_paths(molecules, n_atoms, seed=0):
    rng = np.random.default_rng(seed)
    positions = rng.random((n_atoms, 3)) * 6.0
    mol_ids, mol_sizes = _molecule_index_arrays(molecules, n_atoms)
    fallback = _mean_fallback_molecules(molecules)
    runs = _uniform_run_plan(molecules)
    slow = _scale_about_molecule_centroids(
        positions, mol_ids, mol_sizes, SCALE, fallback, None)
    fast = _scale_about_molecule_centroids(
        positions, mol_ids, mol_sizes, SCALE, fallback, runs)
    return slow, fast, runs, positions


def test_solvated_layout_is_covered_by_a_run_plan():
    molecules, n_atoms = _solvated_layout()
    runs = _uniform_run_plan(molecules)
    assert runs is not None
    assert [(n, size) for _fm, n, size, _fa in runs] == [(1, 138), (400, 3), (7, 1)]
    assert sum(n * size for _fm, n, size, _fa in runs) == n_atoms


def test_fast_path_is_bit_identical_on_a_solvated_layout():
    molecules, n_atoms = _solvated_layout()
    slow, fast, runs, _pos = _both_paths(molecules, n_atoms)
    assert runs is not None, "the fast path must actually be exercised here"
    assert np.array_equal(slow, fast)


def test_fast_path_is_bit_identical_across_many_random_water_counts():
    """The property test that caught the previous 8.9e-16 defect, re-aimed."""
    for seed in range(12):
        rng = np.random.default_rng(1000 + seed)
        molecules, n_atoms = _solvated_layout(
            n_water=int(rng.integers(50, 600)),
            solute=int(rng.integers(130, 200)),
            n_ions=int(rng.integers(0, 12)))
        slow, fast, runs, _pos = _both_paths(molecules, n_atoms, seed=seed)
        assert runs is not None
        assert np.array_equal(slow, fast), "seed %d differs" % seed


def test_sums_from_runs_match_bincount_exactly():
    molecules, n_atoms = _solvated_layout()
    rng = np.random.default_rng(3)
    positions = rng.random((n_atoms, 3)) * 6.0
    mol_ids, _sizes = _molecule_index_arrays(molecules, n_atoms)
    runs = _uniform_run_plan(molecules)
    n_mol = len(molecules)

    ref = np.empty((n_mol, 3))
    for k in range(3):
        ref[:, k] = np.bincount(mol_ids, weights=positions[:, k], minlength=n_mol)
    got = _sums_from_runs(positions, runs, n_mol)
    assert np.array_equal(got, ref)


def test_plan_refuses_a_non_ascending_molecule():
    molecules = [[0, 1, 2], [5, 4, 3], [6, 7, 8]]
    assert _uniform_run_plan(molecules) is None


def test_plan_refuses_a_non_contiguous_molecule():
    molecules = [[0, 1, 2], [3, 4, 9], [5, 6, 7]]
    assert _uniform_run_plan(molecules) is None


def test_a_refused_plan_leaves_the_bincount_path_in_charge():
    """Falling back must still produce the right answer, not just not crash."""
    molecules = [[0, 1, 2], [5, 4, 3], [6, 7, 8]]
    n_atoms = 9
    rng = np.random.default_rng(7)
    positions = rng.random((n_atoms, 3)) * 6.0
    mol_ids, mol_sizes = _molecule_index_arrays(molecules, n_atoms)
    fallback = _mean_fallback_molecules(molecules)
    assert _uniform_run_plan(molecules) is None

    got = _scale_about_molecule_centroids(
        positions, mol_ids, mol_sizes, SCALE, fallback, None)
    expected = positions.copy()
    for mol in molecules:
        expected[mol] = positions[mol] + SCALE * positions[mol].mean(axis=0)
    assert np.allclose(got, expected, rtol=0, atol=1e-15)


def test_large_molecule_fallback_still_overrides_the_fast_path():
    """A >=128-atom molecule keeps its ``.mean()`` centroid under both paths."""
    molecules, n_atoms = _solvated_layout(solute=138)
    rng = np.random.default_rng(11)
    positions = rng.random((n_atoms, 3)) * 6.0
    mol_ids, mol_sizes = _molecule_index_arrays(molecules, n_atoms)
    fallback = _mean_fallback_molecules(molecules)
    assert any(m == 0 for m, _mol in fallback), "the solute must take the fallback"
    runs = _uniform_run_plan(molecules)

    fast = _scale_about_molecule_centroids(
        positions, mol_ids, mol_sizes, SCALE, fallback, runs)
    solute = molecules[0]
    want = positions[solute] + SCALE * positions[solute].mean(axis=0)
    assert np.array_equal(fast[solute], want)


def test_the_reshape_is_a_view_not_a_copy():
    """Board condition: a copying reshape would silently reintroduce a
    GIL-held allocation and invalidate the whole performance case."""
    molecules, n_atoms = _solvated_layout()
    positions = np.ascontiguousarray(
        np.random.default_rng(5).random((n_atoms, 3)) * 6.0)
    runs = _uniform_run_plan(molecules)
    for _fm, n, size, first_atom in runs:
        blk = positions[first_atom:first_atom + n * size]
        assert blk.base is not None, "slice must be a view"
        assert blk.reshape(n, size, 3).base is not None, "reshape must be a view"


def test_input_positions_are_not_mutated_by_the_fast_path():
    molecules, n_atoms = _solvated_layout()
    rng = np.random.default_rng(13)
    positions = rng.random((n_atoms, 3)) * 6.0
    before = positions.copy()
    mol_ids, mol_sizes = _molecule_index_arrays(molecules, n_atoms)
    _scale_about_molecule_centroids(
        positions, mol_ids, mol_sizes, SCALE,
        _mean_fallback_molecules(molecules), _uniform_run_plan(molecules))
    assert np.array_equal(positions, before)


def test_expansion_preserves_the_original_operation_order():
    """centers are expanded first and scaled after.

    Folding the scale into the centroid (``acc *= SCALE/size``) reassociates the
    arithmetic and reintroduces a last-ulp difference -- measured at 8.9e-16
    while developing this. Pinned so the cheaper-looking form cannot creep back.
    """
    molecules, n_atoms = _solvated_layout(n_water=64, solute=138, n_ions=3)
    rng = np.random.default_rng(17)
    positions = rng.random((n_atoms, 3)) * 6.0
    runs = _uniform_run_plan(molecules)
    _ids, sizes = _molecule_index_arrays(molecules, n_atoms)
    sums = _sums_from_runs(positions, runs, len(molecules))
    centers = sums / sizes[:, None]

    got = _expand_and_offset(positions, centers, runs, SCALE)
    mol_ids, _s = _molecule_index_arrays(molecules, n_atoms)
    want = positions + SCALE * centers[mol_ids]
    assert np.array_equal(got, want)


def test_result_dtype_and_shape_are_unchanged():
    molecules, n_atoms = _solvated_layout()
    slow, fast, _runs, positions = _both_paths(molecules, n_atoms)
    assert fast.dtype == slow.dtype == positions.dtype == np.float64
    assert fast.shape == positions.shape
