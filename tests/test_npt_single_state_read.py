"""Coordinates and box come from ONE State, not two.

``_read_positions_and_box`` used to fetch ``getState(getPositions=True)`` and
then call ``_box_matrix_nm(context)``, which fetches a SECOND State purely for
the box vectors a State already carries.

Under 32 replicas sharing 4 GPUs through MPS a getState is queue wait rather
than transfer -- 8.0 ms per attempt for ~240 KB, about 30 MB/s, orders below
PCIe -- and this sits on the reject path where ~78% of attempts go. Removing the
second call is a pure sync saving with no change to any returned value.

These tests pin both halves: the values are bit-identical to what the two-call
form produced, and the second call is really gone.
"""
import numpy as np

from gareus import npt
from gareus.imports import import_openmm


def _periodic_context():
    """Two particles in a triclinic-ish periodic box, on the Reference platform.

    The box is deliberately not cubic so a transposed or mis-ordered vector
    conversion cannot pass by symmetry.
    """
    openmm, _app, unit = import_openmm()
    system = openmm.System()
    system.addParticle(1.0)
    system.addParticle(1.0)
    system.setDefaultPeriodicBoxVectors(
        openmm.Vec3(2.5, 0.0, 0.0) * unit.nanometer,
        openmm.Vec3(0.3, 2.7, 0.0) * unit.nanometer,
        openmm.Vec3(0.1, 0.2, 2.9) * unit.nanometer,
    )
    force = openmm.HarmonicBondForce()
    force.addBond(0, 1, 0.15, 1000.0)
    system.addForce(force)
    integ = openmm.VerletIntegrator(0.001 * unit.picoseconds)
    ctx = openmm.Context(system, integ, openmm.Platform.getPlatformByName("Reference"))
    ctx.setPositions([[0.11, 0.22, 0.33], [0.24, 0.19, 0.41]])
    return ctx


class _CountingContext:
    """Transparent proxy that counts getState calls."""

    def __init__(self, ctx):
        self._ctx = ctx
        self.get_state_calls = 0

    def getState(self, *args, **kwargs):
        self.get_state_calls += 1
        return self._ctx.getState(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._ctx, name)


def test_reading_positions_and_box_costs_exactly_one_get_state():
    """The regression guard. Two calls here is the defect being fixed."""
    ctx = _CountingContext(_periodic_context())
    npt._read_positions_and_box(ctx)
    assert ctx.get_state_calls == 1, (
        "expected a single getState; got %d (the box must come from the State "
        "already fetched, not a second one)" % ctx.get_state_calls)


def test_borrowed_box_is_bit_identical_to_a_freshly_read_one():
    """Value-preservation, asserted with == rather than allclose.

    Both paths now run the same ``_box_matrix_from_state`` conversion, so this
    holds by construction. The assertion is what would catch a conversion-path
    slip if either side were changed independently.
    """
    ctx = _periodic_context()
    _pos, borrowed = npt._read_positions_and_box(ctx)
    fresh = npt._box_matrix_nm(ctx)
    assert borrowed.shape == (3, 3)
    assert np.array_equal(borrowed, fresh)
    assert borrowed.dtype == fresh.dtype


def test_positions_are_bit_identical_to_a_direct_state_read():
    ctx = _periodic_context()
    openmm, _app, unit = import_openmm()
    pos, _box = npt._read_positions_and_box(ctx)
    direct = np.array(
        ctx.getState(getPositions=True).getPositions(asNumpy=True)
        .value_in_unit(unit.nanometer), dtype=float)
    assert np.array_equal(pos, direct)


def test_the_returned_positions_own_their_buffer():
    """The snapshot must not alias a State-owned array.

    ``np.asarray`` would skip the copy when the dtype already matches, leaving
    the returned array pointing at memory the State owns. This snapshot is what
    the restore path compares the Context against, so it owning its buffer is a
    correctness property, not a style preference.
    """
    ctx = _periodic_context()
    pos, _box = npt._read_positions_and_box(ctx)
    assert pos.flags.owndata, "positions snapshot must own its buffer"

    before = pos.copy()
    pos[0, 0] += 1.0  # scribble on the snapshot
    again, _box2 = npt._read_positions_and_box(ctx)
    assert np.array_equal(again, before), (
        "mutating the returned snapshot changed what the Context reports back; "
        "the array is aliasing State-owned memory")


def test_box_matrix_rows_are_the_box_vectors_in_order():
    """Guards the conversion itself: rows are a, b, c -- not the transpose."""
    ctx = _periodic_context()
    _pos, box = npt._read_positions_and_box(ctx)
    assert np.allclose(box[0], [2.5, 0.0, 0.0])
    assert np.allclose(box[1], [0.3, 2.7, 0.0])
    assert np.allclose(box[2], [0.1, 0.2, 2.9])


def test_volume_is_unchanged_by_the_single_read():
    """The acceptance rule consumes the box only through its volume."""
    ctx = _periodic_context()
    _pos, borrowed = npt._read_positions_and_box(ctx)
    assert npt._box_volume_nm3(borrowed) == npt._box_volume_nm3(npt._box_matrix_nm(ctx))
