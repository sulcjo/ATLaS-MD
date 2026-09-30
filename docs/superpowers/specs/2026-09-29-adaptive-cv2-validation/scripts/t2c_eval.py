"""T2 calibration: evaluate the shipped R2/R3 code on one simulation, at every knob value.

Sample once, evaluate many: R3 is run once per window with refine_min_transitions = 0
(so every bimodal window reaches the spring code and records both crossing counts), and the
threshold, estimator and 4x growth cap are applied post hoc from the recorded metrics. R2's
interval statistics (weight share, contributing centres, bootstrap sigma) are computed once
per column with the module's own functions; the two R2 knobs only change the flag test.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from gareus.adaptive import cv2_coverage as cov
from gareus.adaptive import cv2_resolution as cr
from gareus.adaptive import cv2_resolution_rules as rules
from gareus.adaptive.edge_metric import blocking_inefficiency
from gareus.swarm.ladder_design import R_KCAL_MOL_K

import t2c_sim as S

T_UNIT = 1.0 / R_KCAL_MOL_K          # temperature at which RT = 1 (harness units)
P4_MAX_PAIRS = 2000
MIN_TRANSITIONS = (1, 2, 3, 5, 10, 20, 50, 100)
GROWTH = (2.0, 4.0, 8.0, 16.0, 32.0, float("inf"))
COVERAGE_MIN_WINDOWS = (1.0, 1.5, 2.0, 2.5, 3.0)
PMF_SIGMA = (0.25, 0.5, 1.0, 2.0, float("inf"))


def settings(**kw) -> cr.ResolutionSettings:
    return cr.ResolutionSettings(temperature_k=T_UNIT, **kw)


def _moments(z: np.ndarray) -> Dict[str, float]:
    m, v = float(z.mean()), float(z.var())
    sd = math.sqrt(v) if v > 0 else 0.0
    sk = float(np.mean(((z - m) / sd) ** 3)) if sd else 0.0
    ku = float(np.mean(((z - m) / sd) ** 4) - 3.0) if sd else 0.0
    return {"n": int(z.size), "mean": m, "var": v, "skewness": sk, "kurtosis_excess": ku}


def views_for(sc: S.Scenario, X: np.ndarray) -> Dict[int, cr.StateView]:
    out = {}
    for i, w in enumerate(sc.windows):
        z = X[:, i, 1]
        mom = _moments(z)
        out[i] = cr.StateView(i, float(w.center1), float(w.k1), float(w.center2), float(w.k2), 0.0, 0,
                              mom["mean"], mom["var"], int(z.size), mom)
    return out


def replica_runs(z: np.ndarray, rep: np.ndarray) -> List[np.ndarray]:
    """Contiguous residences of one replica at the state (the production provider's runs)."""
    brk = np.flatnonzero(np.diff(rep) != 0) + 1
    return [seg for seg in np.split(z, brk) if seg.size]


def replica_path_count(z: np.ndarray, rep: np.ndarray, bounds) -> int:
    """Candidate estimator (NOT in gareus): each replica's samples AT this state in time order,
    joined across its absences; a core-to-core label change between consecutive visits counts.
    A swap relabels, it never moves coordinates, so every counted change is a real crossing by
    that replica's own dynamics (possibly made while it sat in another window)."""
    ua, lb = bounds
    total = 0
    for r in np.unique(rep):
        zr = z[rep == r]
        lab = np.where(zr <= ua, 0, np.where(zr >= lb, 1, -1))
        lab = lab[lab >= 0]
        if lab.size > 1:
            total += int(np.count_nonzero(np.diff(lab)))
    return total


def eval_r3(sc: S.Scenario, sim: S.Sim, n: int, truth: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    X, R = sim.x[:n], sim.rep[:n]
    views = views_for(sc, X)
    ids = sorted(views)
    stride = max(1, int(math.ceil(n / P4_MAX_PAIRS)))
    subs = {i: {"cv2": X[::stride, i, 1], "source_index": np.zeros(X[::stride].shape[0], dtype=np.int64)}
            for i in ids}

    def runs_for(state_ids):
        return {int(s): {"source": "parquet", "state_runs": [X[:, s, 1]],
                         "replica_runs": replica_runs(X[:, s, 1], R[:, s])} for s in state_ids}

    st = settings(refine_min_transitions=0, refine_transition_count="replica")
    cands = rules.propose_r3(views, ids, subs, runs_for, st, epoch=0)
    out = []
    for c in cands:
        s = c["state_ids"][0]
        m = c.get("metrics") or {}
        w = sc.windows[s]
        g = blocking_inefficiency(X[:, s, 1])
        rec = {"state": s, "c1": w.center1, "c2": w.center2, "k2": w.k2, "gate": m.get("r3_gate"),
               "decision_at_min0": c["decision"], "refusal_at_cap4": c.get("refusal"),
               "exact_resolvable": bool(truth[s]["resolvable"]), "exact_depth_kT": truth[s]["depth_kT"],
               "g_cv2": float(g["g"]), "n": int(n)}
        tr = truth[s]
        if tr.get("valley") is not None:
            # mode population error vs exact: ln-odds of the lower basin (exact valley split)
            zlo = X[:, s, 1] <= tr["valley"]
            wh = float(zlo.mean())
            we = float(tr["masses"][0] / (tr["masses"][0] + tr["masses"][1]))
            lo = lambda p_: math.log(max(p_, 1e-4) / max(1.0 - p_, 1e-4))  # noqa: E731
            rec.update(pop_lower_sampled=wh, pop_lower_exact=we, pop_lnodds_err=abs(lo(wh) - lo(we)))
        if m.get("r3_gate") == "passed" or m.get("transitions") is not None:
            b = m.get("core_bounds")
            rec.update(depth_kT=(m.get("depth") or {}).get("depth_kT"), core_bounds=b,
                       t_replica=m.get("transitions"), t_state_series=m.get("transitions_state_series"),
                       t_replica_path=replica_path_count(X[:, s, 1], R[:, s], b) if b else None)
            kids = (c.get("proposal") or {}).get("children") or []
            rec["children"] = [{"f2_est": k["f2_est"], "k2": k["k2"], "sigma_target": k["sigma_target"],
                                "sigma_used": k["sigma_used"], "secondary_center": k["secondary_center"],
                                "compression_floor_k2": k["compression_floor_k2"],
                                "growth_needed": k["compression_floor_k2"] / w.k2, "refusal": k.get("refusal")}
                               for k in kids]
        out.append(rec)
    return out


def _slab_error(est_f: np.ndarray, exact_f: np.ndarray, frac: np.ndarray) -> np.ndarray:
    """Per-interval |F_est - F_exact| (kT) after a weight-averaged shift; NaN where unsampled."""
    ok = np.isfinite(est_f) & np.isfinite(exact_f)
    err = np.full(est_f.shape, np.nan)
    if ok.sum() < 2:
        return err
    wt = np.exp(-exact_f[ok]); wt /= wt.sum()
    d = est_f[ok] - exact_f[ok]
    d -= np.sum(wt * d)
    err[ok] = np.abs(d)
    return err


def eval_r2(sc: S.Scenario, sim: S.Sim, n: int, seed: int) -> Dict[str, Any]:
    X = sim.x[:n]
    views = views_for(sc, X)
    ids = sorted(views)
    K = len(ids)
    cv1 = X[:, :, 0].T.reshape(-1)
    cv2 = X[:, :, 1].T.reshape(-1)
    sidx = np.repeat(np.arange(K), n)
    vlist = [views[i] for i in ids]
    u = cov.reduced_umbrella(cv1, cv2, vlist, 1.0)
    n_k = np.bincount(sidx, minlength=K).astype(float)
    info: Dict[str, Any] = {}
    f = cov.solve_mbar(u, n_k, info=info)
    lw = cov.log_weights(u, n_k, f)
    w = np.exp(lw - lw.max()); w /= w.sum()
    keys = [(round(v.c1, 6), round(v.c2, 6)) for v in vlist]
    uniq = {k: i for i, k in enumerate(dict.fromkeys(keys))}
    centre_of_state = np.asarray([uniq[k] for k in keys], dtype=np.int64)
    rng = np.random.default_rng([3303, int(seed)])
    base = settings()
    cols = []
    for col in cov.columns(views, ids, base):
        col = cov._extend_to_weight(col, cv1, cv2, w)
        # pre-v3 fixed 20 blocks per state, so Section 9.6 reproduces (v3's default is autocorrelation blocks)
        frac, n_eff, n_contrib, sigma = cov._interval_stats(col, cv1, cv2, sidx, w, centre_of_state, rng,
                                                            block_ids=cov._block_ids(sidx, cov.BOOT_BLOCKS))
        edges = col["edges"]
        tot = frac.sum()
        with np.errstate(divide="ignore"):
            est_f = -np.log(frac / tot) if tot > 0 else np.full(frac.shape, np.nan)
        est_f = np.where(np.isfinite(est_f), est_f, np.nan)
        exact_f = S.exact_slab_cv2(sc, col["c1"], col["slab_half_width"], edges)
        err = _slab_error(est_f, exact_f, frac)
        mids = 0.5 * (edges[1:] + edges[:-1])
        in_hole = np.zeros(frac.size, dtype=bool)
        if sc.hole is not None and (sc.hole[0] is None or abs(sc.hole[0] - col["c1"]) < 1e-6):
            in_hole = (mids >= sc.hole[1][0]) & (mids <= sc.hole[1][1])
        combos = {}
        for cmw in COVERAGE_MIN_WINDOWS:
            for ps in PMF_SIGMA:
                stg = settings(coverage_min_windows=cmw, refine_pmf_sigma_kT=ps)
                heavy = frac >= cov.MIN_INTERVAL_WEIGHT
                by_c = heavy & (n_contrib < cmw)
                by_s = heavy & (sigma > ps)
                flags = by_c | by_s
                runs = cov._hole_runs(flags, edges, [float(v.c2) for v in col["members"]])
                props = []
                for i0, i1 in runs:
                    cand = cov._hole_candidate(col, i0, i1, frac, n_eff, n_contrib, sigma, stg)
                    seg = np.arange(i0, i1 + 1)
                    props.append({"decision": cand["decision"], "refusal": cand.get("refusal"),
                                  "interval": cand["metrics"]["interval"], "in_hole": bool(in_hole[seg].any()),
                                  "by_contrib": bool(by_c[seg].any()), "by_sigma": bool(by_s[seg].any()),
                                  "max_err_kT": float(np.nanmax(err[seg])) if np.isfinite(err[seg]).any() else None,
                                  "child": (cand.get("proposal") or {}).get("children", [None])[0]})
                combos[f"{cmw}|{ps}"] = props
        cols.append({"c1": col["c1"], "n_members": len(col["members"]), "frac": frac.tolist(),
                     "n_contrib": n_contrib.tolist(), "sigma": [None if not math.isfinite(x) else float(x) for x in sigma],
                     "err_kT": [None if not math.isfinite(x) else float(x) for x in err], "in_hole": in_hole.tolist(),
                     "edges": edges.tolist(), "combos": combos})
    return {"mbar": info, "columns": cols}
