"""Synthetic exchange-stat generator tests. No OpenMM."""
from __future__ import annotations

import numpy as np

from gareus.synth.exchange import build_exchange_stats
from gareus.synth.landscapes import mixture_wells
from gareus.synth.sampler import Window, sample_window_exact


def _samples(lp, ws, rng):
    return {i: sample_window_exact(lp, w, 1500, beta=1.0, res=120, rng=rng)
            for i, w in enumerate(ws)}


def test_exchange_schema_and_keys():
    lp = mixture_wells()
    rng = np.random.default_rng(0)
    ws = [Window(0.2, 60, 0.0, 60), Window(0.4, 60, 0.0, 60), Window(0.6, 60, 0.0, 60)]
    st = build_exchange_stats(ws, _samples(lp, ws, rng), beta=1.0, rng=rng)
    assert set(st) >= {"attempts", "accepted", "pairs", "mode"}
    assert set(st["pairs"]) == {"0-1", "1-2"}
    for v in st["pairs"].values():
        assert v["attempts"] > 0 and 0 <= v["accepted"] <= v["attempts"]


def test_exchange_acceptance_drops_with_separation():
    lp = mixture_wells()
    rng = np.random.default_rng(3)
    near = [Window(0.40, 80, 0.0, 80), Window(0.44, 80, 0.0, 80)]
    far = [Window(0.40, 80, 0.0, 80), Window(0.80, 80, 0.0, 80)]
    a = build_exchange_stats(near, _samples(lp, near, rng), beta=1.0, rng=rng)["pairs"]["0-1"]
    b = build_exchange_stats(far, _samples(lp, far, rng), beta=1.0, rng=rng)["pairs"]["0-1"]
    assert a["accepted"] / a["attempts"] > b["accepted"] / b["attempts"]
