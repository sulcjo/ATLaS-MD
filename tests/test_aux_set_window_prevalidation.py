import math
from types import SimpleNamespace

import pytest

from gareus.correctness._io import IntegrityError
from gareus.windows import set_window


class Rec:
    def __init__(self):
        self.calls = []

    def setParameter(self, name, value):
        self.calls.append((name, value))


def aux(centers, ks):
    return SimpleNamespace(
        info=SimpleNamespace(global_k="aux_k", global_c="aux_c"),
        table=SimpleNamespace(centers=centers, k_kcal=ks))


def run(ctx, aux_state=None, w=0):
    set_window(ctx, [1.0, 2.0], [10.0, 20.0], w, [3.0, 4.0], [5.0, 6.0], aux_state=aux_state)


def test_non_aux_sequence_unchanged():
    ctx = Rec()
    run(ctx, w=1)
    assert ctx.calls == [("r0", 2.0), ("k", 20.0), ("ss0", 4.0), ("ss_k", 6.0)]


@pytest.mark.parametrize("bad", [math.nan, math.inf, -1.0])
def test_invalid_aux_k_makes_no_calls(bad):
    ctx = Rec()
    with pytest.raises(IntegrityError, match="window index 0"):
        run(ctx, aux([0.5], [bad]))
    assert ctx.calls == []


def test_nonfinite_centre_with_positive_k_makes_no_calls():
    ctx = Rec()
    with pytest.raises(IntegrityError):
        run(ctx, aux([math.nan], [2.0]))
    assert ctx.calls == []


def test_valid_active_values():
    ctx = Rec()
    run(ctx, aux([0.5], [2.0]))
    assert ctx.calls[:4] == [("r0", 1.0), ("k", 10.0), ("ss0", 3.0), ("ss_k", 5.0)]
    assert ctx.calls[4:] == [("aux_k", 2.0 * 4.184), ("aux_c", 0.5)]


def test_inactive_k0_ignores_centre_and_resets():
    ctx = Rec()
    run(ctx, aux([math.nan], [0.0]))
    assert ctx.calls[4:] == [("aux_k", 0.0), ("aux_c", 0.0)]
