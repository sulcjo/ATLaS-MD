"""Per-rung GaMD boost statistics for the run report."""
from __future__ import annotations

from typing import Optional

import numpy as np

from gareus.units import KJ_PER_KCAL


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
    # Round ONCE and reuse the same rounded array for both binning (np.unique)
    # and selection (==). Binning on a rounded value but then selecting
    # against the raw array with a 1e-9 tolerance (the original bug here)
    # matches nothing whenever a real lambda's true value differs from its
    # own 6-dp rounding by more than 1e-9 -- true for almost any value that
    # isn't already <=6 decimal digits, e.g. adaptive bisection's 1/128 ==
    # 0.0078125. That silently produced n=0 for the affected rung, which is
    # indistinguishable from a legitimate unboosted rung.
    lam_r = np.round(lam, 6)
    rows = []
    for value in np.unique(lam_r[np.isfinite(lam_r)]):
        sel = dv[lam_r == value]
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


def gamd_boost_by_rung_report(d, kbt_kcal: float) -> tuple[Optional[list], list]:
    """Per-rung GaMD boost/reweighting diagnostics for one ``Data``, isolated
    from the report-writer wiring so it is directly unit-testable.

    Returns ``(rows, warnings)``:

    - ``rows is None`` when ``d.meta['gamd_ladder']`` is falsy -- callers
      must NOT set ``gamd_boost_by_rung`` in that case, so a non-ladder run's
      summary is provably unaffected (nothing here even runs).
    - Otherwise ``rows`` is the list from :func:`boost_report_rows` (never
      ``None``, possibly ``[]``), and ``warnings`` carries zero or more
      human-readable strings: one if fewer than 2 distinct rungs were found
      (an all-zero/collapsed ``state_lambdas`` would otherwise silently
      reproduce the pooled-across-rungs problem this report exists to
      expose), or exactly one if anything above raised -- this function
      never raises.
    - ``d.boost_kj`` absent/all-NaN and ``d.state_lambdas is None`` both
      degrade to zero/empty rows via :func:`boost_report_rows`, not a raise.
    """
    if not (getattr(d, "meta", None) or {}).get("gamd_ladder"):
        return None, []
    try:
        lam_per_sample = (
            d.state_lambdas[np.asarray(d.window, dtype=np.int64)]
            if d.state_lambdas is not None else np.array([], dtype=np.float64)
        )
        dv_kcal = np.asarray(d.boost_kj, dtype=np.float64) / KJ_PER_KCAL
        rows = boost_report_rows(dv_kcal, lam_per_sample, kbt_kcal)
        warnings: list = []
        if len(rows) < 2:
            warnings.append(
                "gamd_ladder is active but fewer than 2 distinct lambda rungs were found "
                f"in per-sample lambda ({len(rows)} found); gamd_boost_by_rung cannot "
                "distinguish rungs -- check state_lambdas/state_registry.csv for this run."
            )
        return rows, warnings
    except Exception as exc:
        return [], [f"gamd boost/rung report failed: {exc}"]
