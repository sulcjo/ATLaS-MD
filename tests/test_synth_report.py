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
