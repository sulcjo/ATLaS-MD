"""Plot/report tests (skipped without matplotlib)."""
from __future__ import annotations

import pytest

pytest.importorskip("matplotlib")

from gareus.synth.drivers import RoundRecord
from gareus.synth.landscapes import mixture_wells
from gareus.synth.report import plot_campaign


def test_plot_campaign_writes_pngs(tmp_path):
    lp = mixture_wells()
    recs = [RoundRecord(0, [0.2, 0.5, 0.8], [50.0] * 3, [0.0], [60.0], 3,
                        0.3, 0.3, False),
            RoundRecord(1, [0.2, 0.4, 0.6, 0.8], [50.0] * 4, [0.0], [60.0], 4,
                        0.31, 0.31, True)]
    paths = plot_campaign(lp, recs, tmp_path)
    assert len(paths) >= 1
    for p in paths:
        assert p.exists() and p.stat().st_size > 0


def test_plot_cv_space_writes_panel(tmp_path):
    import numpy as np
    from gareus.synth.report import plot_cv_space
    from gareus.synth.sampler import Window, sample_window_exact
    lp = mixture_wells()
    rng = np.random.default_rng(0)
    samples, windows = {}, {}
    wi = 0
    for c1 in (0.2, 0.5, 0.8):
        for c2 in (-0.5, 0.5):
            w = Window(float(c1), 40.0, float(c2), 40.0)
            windows[wi] = w
            samples[wi] = sample_window_exact(lp, w, 800, beta=1.0, res=80, rng=rng)
            wi += 1
    out = plot_cv_space(lp, samples, windows, tmp_path / "cv_space.png", res=80)
    assert out.exists() and out.stat().st_size > 0
