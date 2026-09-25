"""Cost of one ``union_diagnostics_from_npz`` call at chignolin scale (not a test).

Builds ``n_centres`` x 4 rungs synthetic states (default 59 x 4 = 236, chignolin_9's
layout size) on an 8 x 8 grid of centres, draws ``n_rows`` samples in total, writes the
union NPZ with ``topup_study``'s writer and times one real diagnostics call. Prints wall
seconds and the process's peak RSS before and after the call (``ru_maxrss``).

    python -m gareus.synth.union_solve_bench --n-rows 250000
"""
from __future__ import annotations

import argparse
import json
import resource
import tempfile
import time
from pathlib import Path

import numpy as np

from . import topup_study as T
from .landscapes import LANDSCAPES
from .sampler import Window, sample_window_exact


def _peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 ** 2   # Linux: KiB


def bench(n_rows: int, n_centres: int = 59, landscape: str = "rugged-2d", seed: int = 0) -> dict:
    from ..adaptive.union_diagnostics import union_diagnostics_from_npz
    ls = LANDSCAPES[landscape]
    grid = [(float(a), float(b)) for a in np.linspace(0.05, 0.95, 8) for b in np.linspace(-0.9, 0.9, 8)]
    windows = [Window(c1, T.K_WINDOW, c2, T.K_WINDOW, lam=lam) for c1, c2 in grid[:n_centres] for lam in T.RUNGS]
    edges, _nb, _rp, policy = T._layout(windows)
    per_state = max(1, n_rows // len(windows))
    camp = T._Campaign(len(windows))
    rng = np.random.default_rng(seed)
    for k, w in enumerate(windows):
        camp.rows_x.append(sample_window_exact(ls, w, per_state, rng=rng))
        camp.rows_w.append(np.full(per_state, k, dtype=np.int64))
        camp.kept[k] = per_state
        camp.eff[k] = per_state
        camp.steps[k] = per_state * T.REPORT_INTERVAL
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "union.npz"
        counts = T._write_union_npz(path, ls, camp, windows)
        rss_before = _peak_rss_gb()
        t0 = time.perf_counter()
        diag = union_diagnostics_from_npz(path, edges, kt_kcal=T.KT_KCAL, subsample_counts=counts,
                                          min_effect_kcal=float(policy.topup_min_effect))
        wall = time.perf_counter() - t0
    return {"n_states": len(windows), "n_rows": per_state * len(windows), "wall_s": round(wall, 2),
            "peak_rss_gb_before_call": round(rss_before, 2), "peak_rss_gb_after_call": round(_peak_rss_gb(), 2),
            "solved": diag is not None}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--n-rows", type=int, default=250_000)
    p.add_argument("--n-centres", type=int, default=59)
    a = p.parse_args()
    print(json.dumps(bench(a.n_rows, a.n_centres)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
