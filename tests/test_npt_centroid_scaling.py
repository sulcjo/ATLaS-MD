"""Bit-identity of the vectorized centroid translation against the loop it replaces.

Assertions here use == deliberately. The claim is that the vectorized form
produces the same IEEE-754 doubles as the per-molecule loop, not that it
produces close ones; a tolerance would not test that claim.
"""
import numpy as np
import pytest

from gareus.npt import (
    _MEAN_FALLBACK_MIN_ATOMS,
    _mean_fallback_molecules,
    _molecule_index_arrays,
    _scale_about_molecule_centroids,
)


def _reference_loop(positions, molecules, scale_minus_one):
    """The implementation being replaced, verbatim in behaviour."""
    new_positions = positions.copy()
    for mol in molecules:
        center = positions[mol].mean(axis=0)
        new_positions[mol] = positions[mol] + scale_minus_one * center
    return new_positions


def _large(molecules):
    return _mean_fallback_molecules(molecules)


def _check(molecules, n_atoms, scale_minus_one=0.0003, seed=0):
    rng = np.random.default_rng(seed)
    positions = rng.random((n_atoms, 3)) * 6.0
    mol_ids, mol_sizes = _molecule_index_arrays(molecules, n_atoms)
    got = _scale_about_molecule_centroids(
        positions, mol_ids, mol_sizes, scale_minus_one, _large(molecules))
    want = _reference_loop(positions, molecules, scale_minus_one)
    assert got.shape == want.shape
    assert np.array_equal(got, want), f"max abs diff {np.abs(got - want).max():.3e}"


def test_production_shape_waters_ions_and_one_solute():
    molecules = [list(range(174))]
    nxt = 174
    for _ in range(36):
        molecules.append([nxt])
        nxt += 1
    while nxt + 3 <= 19008:
        molecules.append([nxt, nxt + 1, nxt + 2])
        nxt += 3
    _check(molecules, nxt)


def test_molecules_listed_out_of_order_with_noncontiguous_indices():
    rng = np.random.default_rng(7)
    perm = rng.permutation(300)
    molecules = [sorted(perm[i:i + 3].tolist()) for i in range(0, 300, 3)]
    rng.shuffle(molecules)
    _check(molecules, 300)


def test_single_atom_molecules_only():
    _check([[i] for i in range(50)], 50)


def test_one_molecule_spanning_the_system():
    _check([list(range(500))], 500)


def test_single_atom_system():
    _check([[0]], 1)


@pytest.mark.parametrize("size", [4, 7, 16, 63, 127])
def test_bincount_window_summation_order_is_exact(size):
    """The sizes that actually traverse bincount: 1 to 127.

    With the >=128 fallback in place, testing 174/1000/10000 atoms would compare
    ndarray.mean against itself and prove nothing. This window is where bincount
    really runs and where a pairwise-vs-sequential divergence would bite. Both
    contiguous and shuffled layouts, since ordering is what changes summation
    order.

    If this ever fails, match numpy's summation explicitly -- do NOT relax to a
    tolerance, which would silently downgrade a bit-identical change into a
    statistically-equivalent one.
    """
    rng = np.random.default_rng(size)
    _check([list(range(size)), list(range(size, size + 3))], size + 3)
    order = rng.permutation(size + 3).tolist()
    _check([order[:size], order[size:]], size + 3, seed=size)


@pytest.mark.parametrize("size", [174, 1000, 10000])
def test_large_molecules_bypass_bincount_entirely(size):
    """Documents WHY testing bincount at these sizes would be vacuous.

    These molecules take the mean fallback, so identity is by construction. The
    assertion is about routing, not arithmetic: if the threshold were ever
    raised above one of these sizes, the guarantee silently becomes empirical
    again and this test says so.
    """
    molecules = [list(range(size))]
    assert _large(molecules) == [(0, molecules[0])], (
        f"size {size} no longer takes the mean fallback; the bit-identity "
        "guarantee for it is now empirical, not by construction")
    _check(molecules, size)


def test_large_molecule_fallback_actually_fires():
    """The guarantee only holds if the fallback is reached. Prove it is."""
    size = _MEAN_FALLBACK_MIN_ATOMS + 1
    molecules = [list(range(size)), [size, size + 1, size + 2]]
    n_atoms = size + 3
    assert _large(molecules) == [(0, molecules[0])]

    rng = np.random.default_rng(5)
    positions = rng.random((n_atoms, 3)) * 6.0
    mol_ids, mol_sizes = _molecule_index_arrays(molecules, n_atoms)

    with_fallback = _scale_about_molecule_centroids(
        positions, mol_ids, mol_sizes, 0.01, _large(molecules))
    without = _scale_about_molecule_centroids(
        positions, mol_ids, mol_sizes, 0.01, ())

    assert np.array_equal(with_fallback, _reference_loop(positions, molecules, 0.01))
    # Small molecules sit below any pairwise blocksize, so both paths agree
    # there; what matters is that the large molecule took the fallback.
    assert np.array_equal(with_fallback[size:], without[size:])


def test_small_molecules_are_below_the_fallback_threshold():
    """Waters and ions must never take the slow path -- that is the whole point."""
    molecules = [[0, 1, 2], [3], [4, 5]]
    assert _large(molecules) == []


def test_result_dtype_is_float64_like_the_original():
    """np.bincount returns float64 regardless of input.

    A float32 position array would be silently upcast and break identity.
    Production cannot hit this -- _read_positions_and_box builds the array with
    dtype=float -- but the helper must not quietly change dtype on its own.
    """
    molecules = [[0, 1, 2], [3]]
    mol_ids, mol_sizes = _molecule_index_arrays(molecules, 4)
    positions = np.random.default_rng(0).random((4, 3))
    assert positions.dtype == np.float64
    out = _scale_about_molecule_centroids(
        positions, mol_ids, mol_sizes, 0.01, _large(molecules))
    assert out.dtype == np.float64


def test_index_arrays_cover_every_atom_exactly_once():
    molecules = [[0, 1, 2], [3], [4, 5]]
    mol_ids, mol_sizes = _molecule_index_arrays(molecules, 6)
    assert mol_ids.tolist() == [0, 0, 0, 1, 2, 2]
    assert mol_sizes.tolist() == [3.0, 1.0, 2.0]
    assert mol_ids.dtype == np.int32


def test_zero_scale_is_the_identity():
    rng = np.random.default_rng(3)
    positions = rng.random((30, 3)) * 6.0
    molecules = [[i, i + 1, i + 2] for i in range(0, 30, 3)]
    mol_ids, mol_sizes = _molecule_index_arrays(molecules, 30)
    got = _scale_about_molecule_centroids(
        positions, mol_ids, mol_sizes, 0.0, _large(molecules))
    assert np.array_equal(got, positions)


def test_input_positions_are_not_mutated():
    """The reject path hands this same array back to _restore_positions."""
    molecules = [[0, 1, 2], [3, 4, 5]]
    mol_ids, mol_sizes = _molecule_index_arrays(molecules, 6)
    rng = np.random.default_rng(0)
    positions = rng.random((6, 3))
    before = positions.copy()
    _scale_about_molecule_centroids(
        positions, mol_ids, mol_sizes, 0.01, _large(molecules))
    assert np.array_equal(positions, before)


@pytest.mark.parametrize("seed", range(25))
def test_property_random_valid_partitions_are_bit_identical(seed):
    """Random total, non-overlapping partitions of random size and shape."""
    rng = np.random.default_rng(1000 + seed)
    n_atoms = int(rng.integers(1, 400))
    order = rng.permutation(n_atoms).tolist()
    molecules = []
    i = 0
    while i < n_atoms:
        take = min(int(rng.integers(1, 9)), n_atoms - i)
        molecules.append(order[i:i + take])
        i += take
    rng.shuffle(molecules)
    _check(molecules, n_atoms,
           scale_minus_one=float(rng.uniform(-0.01, 0.01)), seed=seed)
