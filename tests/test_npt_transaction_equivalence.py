"""The vectorized volume move must be indistinguishable from the loop it replaced.

Not just the translated array: the whole transaction. Same acceptance decision,
same random stream position, same coordinates left in the Context, across
accepted moves, rejected moves and geometrically invalid proposals.

Cross-module test imports follow the existing convention in this suite (see
tests/test_thermodynamic_validity_2d.py:137); tests/ has no __init__.py and
pythonpath = ["."] is set in pyproject.toml.
"""
import numpy as np
import pytest

openmm = pytest.importorskip("openmm")

import gareus.npt as npt_mod  # noqa: E402
from test_npt_controller_mechanics import (  # noqa: E402
    VolumeStubAdapter,
    _make_controller,
    _rigid_molecule_system,
    _snapshot_state,
)


def _loop_reference_for(molecules):
    """The replaced implementation, iterating the ORIGINAL molecule lists.

    Reconstructing molecules with np.flatnonzero(mol_ids == m) would be wrong:
    flatnonzero returns atom indices in ascending order, while the original loop
    iterated self._molecules in whatever order getMolecules() gave. For a
    non-contiguous or unsorted molecule that is a different summation order, so
    such a reference would compare the new code against a *third*
    implementation rather than against the code being replaced.
    """
    def impl(positions, mol_ids, mol_sizes, scale_minus_one, large=(), runs=None):
        # `runs` is the slice plan the production transform may receive; the
        # loop reference deliberately ignores it, because the point of this
        # test is to compare the transaction against the ORIGINAL per-molecule
        # loop, whatever fast path the production code has grown.
        new_positions = positions.copy()
        for mol in molecules:
            center = positions[mol].mean(axis=0)
            new_positions[mol] = positions[mol] + scale_minus_one * center
        return new_positions
    return impl


def _molecules_of(system):
    ctx = openmm.Context(
        system, openmm.VerletIntegrator(0.001),
        openmm.Platform.getPlatformByName("Reference"))
    return [[int(i) for i in mol] for mol in ctx.getMolecules()]


def _run(ctrl, n):
    out = []
    step = ctrl.next_due_step
    for _ in range(n):
        res = ctrl.attempt_due(step)
        out.append((res.proposed_volume_nm3, res.log_acceptance,
                    res.accepted, res.reason))
        step = ctrl.next_due_step
    return out


def _rng_state(ctrl):
    return ctrl.state_dict()["rng"]


@pytest.mark.parametrize("physical", [
    lambda v: 5.0 * (v - 8.5) ** 2,   # mixes accepts and rejects
    lambda v: 0.0,                     # accepts almost everything
    lambda v: 1.0e4 * v,               # rejects almost everything
])
def test_transaction_matches_the_loop_implementation(monkeypatch, physical):
    ctx_a, ctrl_a = _make_controller(
        _rigid_molecule_system(n_mol=12), VolumeStubAdapter(physical=physical),
        seed=4242)
    vectorized = _run(ctrl_a, 30)
    pos_a, box_a = _snapshot_state(ctx_a)
    rng_a = _rng_state(ctrl_a)

    system_b = _rigid_molecule_system(n_mol=12)
    monkeypatch.setattr(npt_mod, "_scale_about_molecule_centroids",
                        _loop_reference_for(_molecules_of(system_b)))
    ctx_b, ctrl_b = _make_controller(
        system_b, VolumeStubAdapter(physical=physical), seed=4242)
    looped = _run(ctrl_b, 30)
    pos_b, box_b = _snapshot_state(ctx_b)
    rng_b = _rng_state(ctrl_b)

    assert vectorized == looped
    assert rng_a == rng_b
    assert np.array_equal(pos_a, pos_b)
    assert np.array_equal(box_a, box_b)
    assert len(vectorized) == 30


def test_both_acceptance_branches_are_actually_exercised():
    """Guards the test above: a run where nothing is ever accepted proves nothing."""
    _ctx, ctrl = _make_controller(
        _rigid_molecule_system(n_mol=12),
        VolumeStubAdapter(physical=lambda v: 5.0 * (v - 8.5) ** 2), seed=4242)
    outcomes = {r[2] for r in _run(ctrl, 30)}
    assert outcomes == {True, False}


def test_production_positions_are_float64():
    """np.bincount returns float64 regardless of input dtype.

    If the Context ever handed back float32 the vectorized path would upcast and
    bit-identity would break silently. _read_positions_and_box builds the array
    with dtype=float; this pins that behaviour against a real Context.
    """
    ctx, _ctrl = _make_controller(
        _rigid_molecule_system(n_mol=4), VolumeStubAdapter(), seed=1)
    positions, box = npt_mod._read_positions_and_box(ctx)
    assert positions.dtype == np.float64
    assert box.dtype == np.float64


def test_controller_precomputes_index_arrays_covering_the_partition():
    ctx, ctrl = _make_controller(
        _rigid_molecule_system(n_mol=4, atoms_per_mol=3), VolumeStubAdapter(),
        seed=1)
    core = ctrl._core
    assert core._mol_ids.dtype == np.int32
    assert core._mol_ids.shape == (12,)
    assert core._mol_sizes.tolist() == [3.0, 3.0, 3.0, 3.0]
    # Contiguous ascending molecules from getMolecules(): nothing needs the
    # mean fallback, which is the fast path this change exists to take.
    assert core._mean_fallback == []
