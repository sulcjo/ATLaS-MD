"""GENPEPT --contact-bias-strength (Metropolis-style contact-count acceptance bias).

GENPEPT's per-residue Ramachandran sampling has no compactness/contact bias
anywhere in generation, basin-hopping, or NMA expansion (all downstream
filtering is energy/sterics only). This adds an opt-in soft bias, applied at
generation time in generate_one, that raises acceptance probability for
conformers with more native contacts:

    accept_prob = min(1, exp(strength * (ccount - ref)))

where ref is the combinatorial max contact_count for the sequence/min-sep when
strength > 0 (so the best-possible conformer is always accepted and lower-contact
ones are progressively suppressed), or 0 when strength < 0 (mirrored).
"""

import numpy as np

import GENPEPT


class _Cfg:
    def __init__(self, contact_bias_strength, seq="GYDPETGTWG", contact_min_sep=3):
        self.contact_bias_strength = contact_bias_strength
        self.seq = seq
        self.contact_min_sep = contact_min_sep


def test_zero_strength_always_accepts_regardless_of_contacts():
    rng = np.random.default_rng(0)
    cfg = _Cfg(contact_bias_strength=0.0)
    for ccount in (0, 1, 5, 100):
        assert GENPEPT._contact_bias_accept(rng, cfg, ccount) is True


def test_default_genconfig_has_no_bias():
    cfg = GENPEPT.GenConfig(
        seq="GYDPETGTWG",
        n_requested=10,
        angle_sd_deg=20.0,
        clash_cutoff_A=1.6,
        contact_cutoff_A=8.0,
        contact_min_sep=3,
        max_clashes=0,
        seed=1,
    )
    assert cfg.contact_bias_strength == 0.0


def test_max_ccount_matches_combinatorial_pair_count():
    # chignolin: N=10, min_sep=3 -> valid gaps 4..9, pairs = 6+5+4+3+2+1 = 21.
    assert GENPEPT._contact_bias_max_ccount(10, 3) == 21
    assert GENPEPT._contact_bias_max_ccount(4, 3) == 0
    assert GENPEPT._contact_bias_max_ccount(0, 3) == 0


def _accept_rate(cfg, ccount, seed, n_trials=4000):
    rng = np.random.default_rng(seed)
    accepted = sum(GENPEPT._contact_bias_accept(rng, cfg, ccount) for _ in range(n_trials))
    return accepted / n_trials


def test_positive_strength_favors_more_contacts_statistically():
    cfg = _Cfg(contact_bias_strength=0.5, seq="GYDPETGTWG", contact_min_sep=3)
    max_ccount = GENPEPT._contact_bias_max_ccount(10, 3)

    low = _accept_rate(cfg, ccount=0, seed=42)
    mid = _accept_rate(cfg, ccount=max_ccount // 2, seed=42)
    high = _accept_rate(cfg, ccount=max_ccount, seed=42)

    assert high == 1.0  # exponent is exactly 0 at the combinatorial max.
    assert low < mid < high


def test_negative_strength_favors_fewer_contacts_statistically():
    cfg = _Cfg(contact_bias_strength=-0.5, seq="GYDPETGTWG", contact_min_sep=3)
    max_ccount = GENPEPT._contact_bias_max_ccount(10, 3)

    low = _accept_rate(cfg, ccount=0, seed=7)
    mid = _accept_rate(cfg, ccount=max_ccount // 2, seed=7)
    high = _accept_rate(cfg, ccount=max_ccount, seed=7)

    assert low == 1.0  # exponent is exactly 0 at ccount=0 for negative strength.
    assert high < mid < low


def test_large_strength_times_ccount_does_not_overflow():
    cfg = _Cfg(contact_bias_strength=1000.0, seq="GYDPETGTWG", contact_min_sep=3)
    rng = np.random.default_rng(1)
    max_ccount = GENPEPT._contact_bias_max_ccount(10, 3)
    # At the combinatorial max the exponent is exactly 0 regardless of strength.
    assert GENPEPT._contact_bias_accept(rng, cfg, max_ccount) is True
    # Far below it, exp() would overflow a naive call; must be clamped, not raise.
    assert GENPEPT._contact_bias_accept(rng, cfg, 0) is False
