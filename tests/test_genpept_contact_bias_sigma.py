"""GENPEPT --contact-bias-sigma: a contact bias that means the same thing at any length.

The legacy --contact-bias-strength measures shortfall against
_contact_bias_max_ccount, a combinatorial bound m(m+1)/2 that grows as O(n^2)
while the contact counts a real peptide reaches grow far slower. On a 10-mer the
bound (21) sits just under what is reached (24), so the bias is a gentle
suppression; on a 25-mer the bound is 231 against a measured mean of 28, so
acceptance collapses to ~1e-7 and the run yields nothing. Because rejected
proposals are discarded rather than retried, that failure costs a full
allocation and produces an empty seed bank.

--contact-bias-sigma instead measures shortfall in standard deviations of the
sequence's OWN contact-count distribution, referenced to its 95th percentile.
Both are calibrated at startup from a deterministic subsample of the run's own
conformers, so the same sigma gives the same acceptance and the same enrichment
regardless of chain length.

Legacy behaviour is unchanged and is covered by test_genpept_contact_bias.py.
"""

import math

import numpy as np
import pytest

import GENPEPT


class _Cfg:
    """Minimal stand-in for GenConfig for the pure acceptance-maths tests."""

    def __init__(self, seq="GYDPETGTWG", contact_min_sep=3,
                 contact_bias_strength=0.0, contact_bias_sigma=0.0,
                 contact_bias_ref=0.0, contact_bias_scale=0.0):
        self.seq = seq
        self.contact_min_sep = contact_min_sep
        self.contact_bias_strength = contact_bias_strength
        self.contact_bias_sigma = contact_bias_sigma
        self.contact_bias_ref = contact_bias_ref
        self.contact_bias_scale = contact_bias_scale


def _gen_cfg(seq, n_requested=400, **kw):
    return GENPEPT.GenConfig(
        seq=seq,
        n_requested=n_requested,
        angle_sd_deg=20.0,
        clash_cutoff_A=1.6,
        contact_cutoff_A=8.0,
        contact_min_sep=3,
        max_clashes=0,
        seed=1,
        write_pdbs=False,
        **kw,
    )


# --- the defect this change exists to fix -----------------------------------

def test_legacy_reference_does_not_scale_with_chain_length():
    """The combinatorial bound outruns reachable contacts as the chain grows."""
    ref10 = GENPEPT._contact_bias_max_ccount(10, 3)
    ref25 = GENPEPT._contact_bias_max_ccount(25, 3)
    assert (ref10, ref25) == (21, 231)
    # Quadratic in length, so a strength tuned at one length is meaningless at another.
    assert ref25 / ref10 > 10


def test_legacy_strength_from_a_short_peptide_annihilates_a_longer_one():
    """0.15 is benign on a 10-mer and total rejection on a 25-mer."""
    short = _Cfg(seq="G" * 10, contact_bias_strength=0.15)
    long_ = _Cfg(seq="G" * 25, contact_bias_strength=0.15)
    # A typical 10-mer conformer (~12 contacts, ref 21) is merely suppressed.
    assert GENPEPT.contact_bias_accept_prob(short, 12) > 1e-2
    # A typical 25-mer conformer (~28 contacts, ref 231) is annihilated.
    assert GENPEPT.contact_bias_accept_prob(long_, 28) < 1e-6


# --- the fix ----------------------------------------------------------------

def test_sigma_zero_accepts_everything():
    cfg = _Cfg(contact_bias_sigma=0.0)
    for ccount in (0, 5, 50, 500):
        assert GENPEPT.contact_bias_accept_prob(cfg, ccount) == 1.0


def test_sigma_accepts_at_and_above_the_reference():
    cfg = _Cfg(contact_bias_sigma=1.0, contact_bias_ref=20.0, contact_bias_scale=5.0)
    assert GENPEPT.contact_bias_accept_prob(cfg, 20) == 1.0
    assert GENPEPT.contact_bias_accept_prob(cfg, 40) == 1.0


def test_sigma_suppresses_by_standard_deviations_of_shortfall():
    cfg = _Cfg(contact_bias_sigma=1.0, contact_bias_ref=20.0, contact_bias_scale=5.0)
    # One sd below the reference costs exactly a factor of e.
    assert GENPEPT.contact_bias_accept_prob(cfg, 15) == pytest.approx(math.exp(-1.0), rel=1e-9)
    assert GENPEPT.contact_bias_accept_prob(cfg, 10) == pytest.approx(math.exp(-2.0), rel=1e-9)


def test_sigma_is_length_invariant_where_strength_is_not():
    """Same sigma, same acceptance, on distributions 2.5x apart in scale.

    Stand-ins for the two measured distributions: chignolin (mean 3.6, sd 3.6,
    p95 11) and CB1 ICL3 (mean 28.3, sd 24.0, p95 75).
    """
    rng = np.random.default_rng(0)
    short_counts = np.clip(rng.normal(3.6, 3.6, 20000), 0, None)
    long_counts = np.clip(rng.normal(28.3, 24.0, 20000), 0, None)

    def mean_accept(counts, cfg):
        return float(np.mean([GENPEPT.contact_bias_accept_prob(cfg, c) for c in counts]))

    for sigma in (0.5, 1.0, 2.0):
        s_cfg = _Cfg(contact_bias_sigma=sigma,
                     contact_bias_ref=float(np.percentile(short_counts, 95)),
                     contact_bias_scale=float(short_counts.std()))
        l_cfg = _Cfg(contact_bias_sigma=sigma,
                     contact_bias_ref=float(np.percentile(long_counts, 95)),
                     contact_bias_scale=float(long_counts.std()))
        a_short, a_long = mean_accept(short_counts, s_cfg), mean_accept(long_counts, l_cfg)
        assert abs(a_short - a_long) < 0.05, (sigma, a_short, a_long)

    # The legacy knob fails the same comparison badly.
    s_leg = _Cfg(seq="G" * 10, contact_bias_strength=0.15)
    l_leg = _Cfg(seq="G" * 25, contact_bias_strength=0.15)
    assert mean_accept(short_counts, s_leg) / max(mean_accept(long_counts, l_leg), 1e-300) > 1e3


def test_sigma_does_not_overflow_at_extreme_values():
    cfg = _Cfg(contact_bias_sigma=1e6, contact_bias_ref=20.0, contact_bias_scale=5.0)
    assert GENPEPT.contact_bias_accept_prob(cfg, 100) == 1.0
    assert GENPEPT.contact_bias_accept_prob(cfg, 0) == pytest.approx(math.exp(-50.0))


# --- calibration ------------------------------------------------------------

def test_calibration_is_deterministic_and_tracks_chain_length():
    short = _gen_cfg("G" * 10)
    long_ = _gen_cfg("G" * 25)
    ref_s, scale_s = GENPEPT.contact_bias_calibration(short, n_sample=200)
    ref_l, scale_l = GENPEPT.contact_bias_calibration(long_, n_sample=200)
    # Reproducible for a given (seq, seed, n_requested).
    assert GENPEPT.contact_bias_calibration(short, n_sample=200) == (ref_s, scale_s)
    # A longer chain reaches more contacts, and the measured reference follows it
    # rather than the combinatorial bound.
    assert ref_l > ref_s
    assert ref_l < GENPEPT._contact_bias_max_ccount(25, 3)
    assert scale_s > 0 and scale_l > 0


def test_calibration_sample_is_bounded_by_requested_trials():
    cfg = _gen_cfg("G" * 12, n_requested=37)
    counts = GENPEPT.contact_bias_sample_ccounts(cfg, n_sample=1000)
    assert 0 < counts.size <= 37


# --- the guard --------------------------------------------------------------

def test_preflight_returns_none_when_no_bias_is_set():
    assert GENPEPT.contact_bias_preflight(_gen_cfg("G" * 12), 10, 1000) is None


def test_preflight_aborts_on_a_legacy_strength_carried_across_lengths():
    cfg = _gen_cfg("G" * 25, n_requested=400, contact_bias_strength=0.15)
    with pytest.raises(SystemExit) as excinfo:
        GENPEPT.contact_bias_preflight(cfg, n_candidate_seeds=10000, n_trials=2_000_000)
    msg = str(excinfo.value)
    assert "refusing to start" in msg
    assert "contact_bias_sigma" in msg          # points at the portable knob
    assert "O(n^2)" in msg                      # explains why


def test_preflight_passes_and_reports_for_a_sane_sigma():
    cfg = _gen_cfg("G" * 25, n_requested=400, contact_bias_sigma=1.0)
    cfg.contact_bias_ref, cfg.contact_bias_scale = GENPEPT.contact_bias_calibration(cfg, n_sample=200)
    info = GENPEPT.contact_bias_preflight(cfg, n_candidate_seeds=10, n_trials=400)
    assert info is not None
    assert info["mode"] == "sigma"
    assert 0.0 < info["accept_fraction"] <= 1.0
    assert info["expected_conformers"] >= 10
