"""Per-rung GaMD boost statistics for the run report."""
from __future__ import annotations

import numpy as np


def _anharm(x: np.ndarray, bins: int = 200) -> float:
    """Gaussian entropy minus actual entropy, in nats. 0.0 means Gaussian.

    Miao's criterion for cumulant reweighting is < 0.01; chignolin_7's top rung
    measured 0.107, which is why that run needs the ladder rather than CE2.
    """
    if x.size == 0 or float(np.var(x)) <= 0.0:
        return 0.0
    hist, edges = np.histogram(x, bins=bins, density=True)
    width = float(edges[1] - edges[0])
    p = hist[hist > 0] * width
    entropy = -float(np.sum(p * np.log(p / width)))
    return float(0.5 * np.log(2 * np.pi * np.e * float(np.var(x))) - entropy)


def boost_report_rows(dv_kcal, lambdas, kt_kcal: float) -> list[dict]:
    dv = np.asarray(dv_kcal, dtype=np.float64)
    lam = np.asarray(lambdas, dtype=np.float64)
    rows = []
    for value in np.unique(np.round(lam[np.isfinite(lam)], 6)):
        sel = dv[np.abs(lam - value) < 1.0e-9]
        sel = sel[np.isfinite(sel)]
        sd = float(np.std(sel)) if sel.size else 0.0
        mean = float(np.mean(sel)) if sel.size else 0.0
        skew = (float(np.mean(((sel - mean) / sd) ** 3)) if sd > 0 else 0.0)
        rows.append({
            "lambda": float(value),
            "n": int(sel.size),
            "mean_dv_kcal": mean,
            "sd_dv_kcal": sd,
            "mean_dv_kt": mean / float(kt_kcal),
            "anharm_nats": _anharm(sel),
            "skew": skew,
        })
    return rows
