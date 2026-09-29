"""Reference curves: two-state (pairwise MBAR, 0..0.5) overlap of two equal-width Gaussians.

O(d) = int p_a p_b / (p_a + p_b) dx for N_a = N_b, d = separation / sd (1D, or the Mahalanobis
distance for isotropic Gaussians of any dimension -- the integrand depends only on the
log-ratio, which is 1D-distributed along the separation). BAR asymptotic variance for
N_a = N_b = n iid samples: var(df) = (1/n) (1/O - 2) (Shirts et al. 2003 with the equal-N
optimal constant). Prints the table used in the T3 report.
"""
import math

import numpy as np


def gaussian_pair_overlap(d: float) -> float:
    x = np.linspace(-12.0, 12.0 + d, 200001)
    pa = np.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)
    pb = np.exp(-0.5 * (x - d) ** 2) / math.sqrt(2 * math.pi)
    with np.errstate(invalid="ignore", divide="ignore"):
        integrand = np.where(pa + pb > 0, pa * pb / (pa + pb), 0.0)
    return float(np.trapz(integrand, x))


def d_for_overlap(o: float) -> float:
    lo, hi = 0.0, 12.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if gaussian_pair_overlap(mid) > o:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


if __name__ == "__main__":
    print("d/sd  O(d)   sd(df)*sqrt(n) [kT]")
    for d in (0.0, 0.5, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 3.5, 4.0):
        o = gaussian_pair_overlap(d)
        v = (1.0 / o - 2.0) if o > 0 else float("inf")
        print(f"{d:4.2f}  {o:.4f}  {math.sqrt(max(v, 0.0)):.3f}")
    print("O target -> d/sd, n needed for sd(df) <= 0.1 kT")
    for o in (0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30):
        d = d_for_overlap(o)
        print(f"{o:.2f} -> d={d:.3f}  n(0.1kT)={(1 / o - 2) / 0.01:.0f}")
