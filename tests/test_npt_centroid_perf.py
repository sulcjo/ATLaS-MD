"""Guard against the per-molecule loop silently returning.

Not a benchmark -- a regression tripwire with a wide margin, so it does not
flake on a loaded machine.
"""
import time

import numpy as np

from gareus.npt import (
    _mean_fallback_molecules,
    _molecule_index_arrays,
    _scale_about_molecule_centroids,
)


def test_vectorized_scaling_is_far_faster_than_a_python_loop():
    n_mol = 6000
    molecules = [[3 * m, 3 * m + 1, 3 * m + 2] for m in range(n_mol)]
    n_atoms = 3 * n_mol
    rng = np.random.default_rng(0)
    positions = rng.random((n_atoms, 3)) * 6.0
    mol_ids, mol_sizes = _molecule_index_arrays(molecules, n_atoms)
    fallback = _mean_fallback_molecules(molecules)
    assert fallback == [], "3-atom ascending molecules must take the fast path"

    def loop():
        new = positions.copy()
        for mol in molecules:
            new[mol] = positions[mol] + 0.0003 * positions[mol].mean(axis=0)
        return new

    def vec():
        return _scale_about_molecule_centroids(
            positions, mol_ids, mol_sizes, 0.0003, fallback)

    assert np.array_equal(loop(), vec())

    loop()
    vec()
    t0 = time.perf_counter()
    for _ in range(3):
        loop()
    loop_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(3):
        vec()
    vec_s = time.perf_counter() - t0

    assert vec_s < 0.05 * loop_s, f"vectorized {vec_s:.4f}s vs loop {loop_s:.4f}s"
