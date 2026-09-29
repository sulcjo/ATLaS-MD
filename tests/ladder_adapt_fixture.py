"""Shared fixtures for the adaptive lambda ladder tests.

Reference ensembles use a dihedral-only envelope, where the rung ensemble is exact:
p_lambda(v) ~ N(v; MU, SD^2) * exp(-beta * lambda * k * (E - v)^2 / 2) is a Gaussian (the boost is
active for all v < E, and v + boost < E holds over the sampled range), so samples are drawn
exactly and the reference overlap is a numerical integral.
"""
import numpy as np

from gareus.pep_gamd import PepGamdEnvelope

BETA = 1.0 / (0.0083144626 * 300.0)
MU, SD = 440.0, 20.0
ENV = PepGamdEnvelope(vmax_total=0.0, vmin_total=0.0, threshold_total=0.0, k0max_total=0.0,
                      vmax_dih=545.0, vmin_dih=371.0, threshold_dih=545.0, k0max_dih=1.0, has_total=False)
K = ENV.k0max_dih / (ENV.vmax_dih - ENV.vmin_dih)
DUAL = PepGamdEnvelope(vmax_total=-2400.0, vmin_total=-3700.0, threshold_total=-2400.0, k0max_total=0.05,
                       vmax_dih=545.0, vmin_dih=371.0, threshold_dih=545.0, k0max_dih=0.3)


def _gauss_params(lam):
    prec = 1.0 / SD**2 + BETA * lam * K
    mean = (MU / SD**2 + BETA * lam * K * ENV.threshold_dih) / prec
    return mean, np.sqrt(1.0 / prec)


def _sample_at(lam, n, seed):
    m, s = _gauss_params(lam)
    vd = np.random.default_rng(seed).normal(m, s, n)
    return np.full(n, np.nan), vd                      # v_pep unused for a dihedral-only boost


def _exact_overlap(lam_a, lam_b):
    v = np.linspace(300.0, 540.0, 200001)
    ma, sa = _gauss_params(lam_a); mb, sb = _gauss_params(lam_b)
    pa = np.exp(-0.5 * ((v - ma) / sa) ** 2) / sa; pb = np.exp(-0.5 * ((v - mb) / sb) ** 2) / sb
    pa /= np.trapezoid(pa, v); pb /= np.trapezoid(pb, v)
    return float(np.trapezoid(pa * pb / (pa + pb), v))
