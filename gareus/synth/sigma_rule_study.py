"""Which per-state sigma rule tracks the true local PMF error? (Ruling 23; a measurement, not a test.)

For uniform (arm A) campaigns of ``topup_study`` it solves the union with the production
solver (``union_diagnostics._solve``) and, per state k, computes candidate sigma rules
from the pairwise uncertainty matrix dDelta_f (kT):

* ``all_edges_min``     -- min over every union edge neighbour (build_geometry_edges; the shipped rule);
* ``spatial_min``       -- min over same-rung spatial neighbours (layout_neighbours), rung partners if none;
* ``spatial_max``       -- max over the same set (the least certain local link);
* ``reference``         -- dDelta_f to one fixed reference state (the lambda=0 state with most samples).

Local error of state k (kT): mean over its same-rung spatial neighbours j of
|(f_j - f_k)_MBAR - (f_j - f_k)_true|, where f_true = -ln sum_grid exp(-(F + U + dV))
on a 200 x 200 grid of the analytic surface (the source of ``reference_pmf``).
Reports Spearman rho of each rule against that error, pooled over seeds x states.

    python -m gareus.synth.sigma_rule_study --out <dir>
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from . import topup_study as T
from .landscapes import LANDSCAPES

RULES = ("all_edges_min", "spatial_min", "spatial_max", "reference")


def _true_f(landscape, windows, res: int = 200) -> np.ndarray:
    g1, g2 = np.meshgrid(np.linspace(*landscape.cv1_bounds, res), np.linspace(*landscape.cv2_bounds, res),
                         indexing="ij")
    x = np.column_stack([g1.ravel(), g2.ravel()])
    f = landscape.energy(x[:, 0], x[:, 1])
    u = T._reduced_potentials(landscape, windows, x) + f[:, None]
    m = u.min(axis=0)
    return -(np.log(np.exp(-(u - m)).sum(axis=0))) + m


def _campaign(landscape, seed: int, hours: float):
    from ..adaptive.throughput import wall_hours
    windows = T._ladder(landscape)
    edges, nb, rp, policy = T._layout(windows)
    ids = list(range(len(windows)))
    per = wall_hours(1, len(ids), T.TIMESTEP_FS, T.N_GPUS, policy.topup_throughput_table)
    steps = int(hours / per // T.REPORT_INTERVAL) * T.REPORT_INTERVAL
    camp = T._Campaign(len(ids))
    T._sample_segment(landscape, windows, ids, steps, T._Streams(landscape, windows, seed),
                      T._partners_in(ids, nb, rp), camp, policy.topup_throughput_table)
    return windows, edges, nb, rp, camp


def measure(landscape_name: str, seed: int, hours: float) -> list:
    from ..adaptive.union_diagnostics import _solve
    ls = LANDSCAPES[landscape_name]
    windows, edges, nb, rp, camp = _campaign(ls, seed, hours)
    x = np.concatenate(camp.rows_x); w = np.concatenate(camp.rows_w)
    u = T._reduced_potentials(ls, windows, x)
    f, dmat, n_k = _solve(u, w, len(windows))
    ft = _true_f(ls, windows)
    edge_nb = {k: set() for k in range(len(windows))}
    for a, b in edges:
        edge_nb[a].add(b); edge_nb[b].add(a)
    lam0 = [k for k in range(len(windows)) if windows[k].lam == 0.0]
    ref = max(lam0, key=lambda k: (n_k[k], -k))
    rows = []
    for k in range(len(windows)):
        sp = nb.get(k, []) or rp.get(k, [])
        if not sp or n_k[k] == 0:
            continue
        err = float(np.mean([abs((f[j] - f[k]) - (ft[j] - ft[k])) for j in nb.get(k, sp)]))
        vals = {"all_edges_min": min(dmat[j, k] for j in edge_nb[k]) if edge_nb[k] else np.nan,
                "spatial_min": min(dmat[j, k] for j in sp), "spatial_max": max(dmat[j, k] for j in sp),
                "reference": dmat[ref, k] if k != ref else np.nan}
        rows.append({"landscape": landscape_name, "seed": seed, "hours": hours, "state": k,
                     "local_error_kT": err, **{r: float(v) for r, v in vals.items()}})
    return rows


def spearman(rows: list) -> dict:
    from scipy.stats import spearmanr
    out = {}
    for r in RULES:
        pts = [(q[r], q["local_error_kT"]) for q in rows if np.isfinite(q[r])]
        rho, p = spearmanr([a for a, _ in pts], [b for _, b in pts])
        out[r] = {"rho": float(rho), "p": float(p), "n": len(pts)}
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--hours", type=float, nargs="+", default=[0.3, 5.0])
    p.add_argument("--landscapes", nargs="+", default=list(T.HETEROGENEOUS))
    a = p.parse_args()
    rows = [row for name in a.landscapes for h in a.hours for s in range(a.n_seeds) for row in measure(name, s, h)]
    summary = {"pooled": spearman(rows),
               **{f"hours={h}": spearman([q for q in rows if q["hours"] == h]) for h in a.hours},
               **{name: spearman([q for q in rows if q["landscape"] == name]) for name in a.landscapes}}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "sigma_rule_study.json").write_text(json.dumps({"summary": summary}, indent=1))
    with gzip.open(a.out / "sigma_rule_study_rows.json.gz", "wt") as fh:
        json.dump(rows, fh)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
