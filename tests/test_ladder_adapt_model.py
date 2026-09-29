"""Per-centre rung model: predicted overlap between any two lambda values (fixtures: ladder_adapt_fixture)."""
import numpy as np
import pytest

from gareus.adaptive.ladder_adapt import fit_centre_model
from ladder_adapt_fixture import BETA, DUAL, ENV, _exact_overlap, _sample_at  # noqa: E402  (tests dir on sys.path)


def _model(lams, n=8000, seed=1):
    return fit_centre_model({l: _sample_at(l, n, seed + i) for i, l in enumerate(lams)}, ENV, BETA)


def test_identical_states_overlap_one_half():
    m = _model((0.0, 0.3))
    assert m.overlap(0.3, 0.3) == pytest.approx(0.5, abs=1e-9)


def test_sampled_pair_overlap_matches_exact():
    m = _model((0.0, 0.5, 1.0))
    assert m.overlap(0.0, 0.5) == pytest.approx(_exact_overlap(0.0, 0.5), abs=0.02)
    assert m.overlap(0.5, 1.0) == pytest.approx(_exact_overlap(0.5, 1.0), abs=0.02)


def test_unsampled_lambda_is_predicted_from_neighbouring_rungs():
    m = _model((0.0, 0.5, 1.0))
    assert m.overlap(0.5, 0.75) == pytest.approx(_exact_overlap(0.5, 0.75), abs=0.02)
    assert m.overlap(0.0, 0.25) == pytest.approx(_exact_overlap(0.0, 0.25), abs=0.02)
    assert m.overlap(0.25, 0.75) == pytest.approx(_exact_overlap(0.25, 0.75), abs=0.02)


def test_overlap_decreases_with_lambda_separation():
    m = _model((0.0, 0.5, 1.0))
    vals = [m.overlap(0.0, l) for l in (0.1, 0.3, 0.5, 0.8, 1.0)]
    assert all(x > y for x, y in zip(vals, vals[1:]))


def test_ess_is_high_inside_and_falls_outside_the_sampled_range():
    m = _model((0.0, 0.3), n=5000)
    assert m.ess(0.15) > 1000
    assert m.ess(1.0) < m.ess(0.15)


def test_nan_channel_samples_are_dropped():
    vp, vd = _sample_at(0.0, 3000, 1)
    vd = vd.copy(); vd[:1000] = np.nan
    m = fit_centre_model({0.0: (vp, vd), 0.5: _sample_at(0.5, 3000, 2)}, ENV, BETA)
    assert m.n_k[0] == 2000


def test_fewer_than_two_rungs_gives_none():
    assert fit_centre_model({0.0: _sample_at(0.0, 1000, 1)}, ENV, BETA) is None


def test_dual_boost_envelope_fits_and_is_monotone():
    rng = np.random.default_rng(5)
    data = {l: (rng.normal(-3000.0, 150.0, 3000), rng.normal(440.0 + 40.0 * l, 20.0, 3000)) for l in (0.0, 0.5, 1.0)}
    m = fit_centre_model(data, DUAL, BETA)
    assert m is not None and m.overlap(0.0, 0.2) > m.overlap(0.0, 0.8)
