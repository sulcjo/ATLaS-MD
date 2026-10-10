"""Forecast-benefit placement of z3 workers: exact port of c10 docs/_local_docs/c10_aux_diagnosis/placement.py
(pilot spec v2, option 1). Parity with c10 placement.json is a test; do not 'improve' it (rounding to 4 dp before
gates and ranking, point-estimate gates other than O_q05, one global RNG stream in this exact order)."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy.special import expit, logsumexp

from .settings import AuxDiscoverySettings

R_KCAL = 0.0019872041
DOC = "Forecast-benefit placement of z3 workers (pilot spec v2, option 1). Generic: no named groups, no restraint-pattern\npreference, no hand-picked states. Inputs: per-frame z3 (frozen model), per-frame evaluation label (frozen contact+H-bond\npartition), state id, lambda, replica lineage, phase.\n\nFor every lambda = 0 state P (training frames = epoch_000 + epoch_001):\n  candidates: c3 in the 5/10/90/95 % training quantiles of z3 in P; worker width sigma_w = f * sd_P(z3), f in {0.35, 0.5, 0.7},\n              k3 = RT / sigma_w^2 (kcal/mol per z3 unit^2).\n  forecast by reweighting P's frames with exp(-du), du = 0.5 k3 (z - c3)^2 / RT:\n    O    = E_p[1 / (1 + exp(du - df))], df = -log E_p[exp(-du)]   (pairwise-MBAR overlap, identical states = 0.5)\n    TV   = 0.5 sum_g |p_worker(g) - p_P(g)|                         (shift of the evaluation-label distribution)\n    null = mean TV after circularly shifting z3 against the labels along P's lineage-ordered series (offsets >= longest\n           lineage, so within-lineage dependence of each series is kept and their alignment is broken)\n    net  = TV - null ; utility = O * net\n  gates (all from a lineage bootstrap, 100 draws, null recomputed inside each draw):\n    O q05 >= 0.20; weighted ESS >= 50 frames; effective lineages >= 20; top-3 lineage weight share <= 0.5\n  rank: utility q10 (bootstrap 10th percentile).\nSelection: greedy down the ranking, at most one worker per (parent, side of the parent's median); each pick must CONFIRM on\nheld-out epoch_002 frames of P with the frozen (c3, k3): held-out net > 0 and held-out O >= 0.15. Stop at 4 confirmed\nworkers or when the ranking is exhausted (fewer than 4 is allowed).\nWrites placement.json (every candidate with its forecasts, gates, rank, held-out check, selection).\n"


def _forecast(z, lab, lin_codes, c, k, shifts, RT, K, what="candidate"):
    z = np.asarray(z, dtype=np.float64)
    if z.ndim != 1 or z.size == 0:
        raise ValueError(f"{what}: z must be a non-empty 1-D array, got shape {z.shape}")
    if not (np.isfinite(RT) and RT > 0 and np.isfinite(c) and np.isfinite(k)):
        raise ValueError(f"{what}: RT, centre and k must be finite with RT > 0 (RT={RT}, c={c}, k={k})")
    du = 0.5 * k * (z - c) ** 2 / RT
    if not np.all(np.isfinite(du)):
        raise ValueError(f"{what}: non-finite forecast energy (non-finite z or overflow); refusing, no clipping")
    lse = float(logsumexp(-du))
    df = np.log(du.size) - lse
    w = np.exp(-du - lse)
    O = float(np.mean(expit(df - du)))
    p0 = np.bincount(lab, minlength=K) / len(lab)
    tv = 0.5 * np.abs(np.bincount(lab, weights=w, minlength=K) - p0).sum()
    nulls = [0.5 * np.abs(np.bincount(np.roll(lab, s), weights=w, minlength=K) - p0).sum() for s in shifts]
    lw = np.bincount(lin_codes, weights=w)
    top3 = float(np.sort(lw)[::-1][:3].sum())
    return {"O": O, "TV": float(tv), "null": float(np.mean(nulls)),
            "null_pct": float(np.mean(np.array(nulls) < tv)),
            "ess_frames": float(1 / np.sum(w ** 2)), "eff_lineages": float(1 / np.sum(lw ** 2)),
            "top3_share": top3}


def place_workers(z, lab, state_id, lineage, step, is_train, is_heldout, s: AuxDiscoverySettings, *,
                  k_labels: int, k3_max: Optional[float] = None, eligible_parents=None) -> dict:
    """``eligible_parents`` (None = any): the states a worker may be spawned from (active, ordinary, lambda = 0
    in the live registry after the epoch's other actions); a ranked candidate of another state is logged
    ``parent_not_eligible`` and never chosen. At most ``s.max_workers`` are chosen (0 = none)."""
    RT = R_KCAL * float(s.temperature_k)
    rng = np.random.default_rng(int(s.placement_seed))

    def shift_set(n, longest, m):
        lo = min(longest, n // 3)
        hi = n - lo
        return rng.integers(lo, hi, size=m) if hi > lo else np.array([n // 2] * m)

    df = pd.DataFrame({"state_id": np.asarray(state_id), "lineage": np.asarray(lineage), "step": np.asarray(step),
                       "z": np.asarray(z), "lab": np.asarray(lab), "tr": np.asarray(is_train, dtype=bool),
                       "ho": np.asarray(is_heldout, dtype=bool)})
    df = df.sort_values(["state_id", "lineage", "step"], kind="mergesort").reset_index(drop=True)
    cands, skipped = [], []
    for sid, grp in df.groupby("state_id"):
        tr = grp[grp.tr]
        ho = grp[grp.ho]
        if len(tr) < s.min_train_frames:
            skipped.append({"state_id": int(sid), "n_train_frames": len(tr), "reason": "too_few_training_frames"})
            continue
        zt, lt = tr.z.to_numpy(), tr.lab.to_numpy()
        lin = pd.factorize(tr.lineage)[0]
        longest = int(np.bincount(lin).max())
        sd = float(zt.std())
        if not np.isfinite(sd) or sd <= 0.0:
            skipped.append({"state_id": int(sid), "n_train_frames": len(tr), "reason": "zero_variance"})
            continue
        med = float(np.median(zt))
        blocks = [np.flatnonzero(lin == b) for b in range(lin.max() + 1)]
        for q in s.quantiles:
            c = float(np.quantile(zt, q))
            for f in s.width_fractions:
                k = RT / (f * sd) ** 2
                point = _forecast(zt, lt, lin, c, k, shift_set(len(zt), longest, s.n_null), RT, k_labels,
                                what=f"parent {int(sid)} c3={c:.4g} f={f}")
                boot = []
                for _ in range(s.n_boot_place):
                    pick = rng.integers(0, len(blocks), len(blocks))
                    idx = np.concatenate([blocks[i] for i in pick])
                    lb = np.concatenate([[j] * len(blocks[i]) for j, i in enumerate(pick)])
                    b = _forecast(zt[idx], lt[idx], lb, c, k, shift_set(len(idx), longest, s.n_null_boot),
                                  RT, k_labels, what=f"parent {int(sid)} c3={c:.4g} f={f} (bootstrap)")
                    boot.append((b["O"], b["O"] * (b["TV"] - b["null"])))
                boot = np.array(boot)
                rec = {"state_id": int(sid), "side": "below" if c < med else "above", "c3": round(c, 4),
                       "k3": round(k, 4), "sigma_w": round(f * sd, 4), "width_fraction": f, "quantile": q,
                       **{k_: round(v, 4) for k_, v in point.items()},
                       "net": round(point["TV"] - point["null"], 4),
                       "utility": round(point["O"] * (point["TV"] - point["null"]), 4),
                       "O_q05": round(float(np.quantile(boot[:, 0], 0.05)), 4),
                       "utility_q10": round(float(np.quantile(boot[:, 1], 0.10)), 4),
                       "n_train_frames": len(zt), "n_train_lineages": len(blocks)}
                rec["gates"] = {"O_q05": rec["O_q05"] >= s.gate_o_q05,
                                "ess_frames": rec["ess_frames"] >= s.gate_ess_frames,
                                "eff_lineages": rec["eff_lineages"] >= s.gate_eff_lineages,
                                "top3_share": rec["top3_share"] <= s.gate_top3_share}
                if k3_max is not None:
                    rec["gates"]["k3_max"] = bool(rec["k3"] <= float(k3_max))
                rec["eligible"] = all(rec["gates"].values())
                if len(ho) >= s.heldout_min_frames:
                    lh = pd.factorize(ho.lineage)[0]
                    h = _forecast(ho.z.to_numpy(), ho.lab.to_numpy(), lh, c, k,
                                  shift_set(len(ho), int(np.bincount(lh).max()), s.n_null), RT, k_labels,
                                  what=f"parent {int(sid)} c3={c:.4g} f={f} (held-out)")
                    rec["heldout"] = {"O": round(h["O"], 4), "net": round(h["TV"] - h["null"], 4),
                                      "null_pct": round(h["null_pct"], 3), "n": len(ho)}
                else:
                    rec["heldout"] = None
                cands.append(rec)
    ranked = sorted([c for c in cands if c["eligible"]], key=lambda c: -c["utility_q10"])
    chosen, used, log = [], set(), []
    allowed = None if eligible_parents is None else {int(x) for x in eligible_parents}
    for r, c in enumerate(ranked):
        if len(chosen) >= int(s.max_workers):
            break
        key = (c["state_id"], c["side"])
        if key in used:
            continue
        if allowed is not None and int(c["state_id"]) not in allowed:
            log.append({"rank": r, "state_id": c["state_id"], "side": c["side"], "c3": c["c3"], "k3": c["k3"],
                        "utility_q10": c["utility_q10"], "heldout": c["heldout"], "confirmed": False,
                        "reason": "parent_not_eligible"})
            continue
        h = c["heldout"]
        ok = bool(h and h["net"] > 0 and h["O"] >= s.heldout_o_min)
        log.append({"rank": r, "state_id": c["state_id"], "side": c["side"], "c3": c["c3"], "k3": c["k3"],
                    "utility_q10": c["utility_q10"], "heldout": h, "confirmed": ok})
        if ok:
            chosen.append(c)
            used.add(key)
    gates = {"O_q05": s.gate_o_q05, "ess_frames": s.gate_ess_frames, "eff_lineages": s.gate_eff_lineages,
             "top3_share": s.gate_top3_share}
    return {"doc": DOC, "n_states": int(df.state_id.nunique()), "n_candidates": len(cands),
            "n_eligible": len(ranked), "gates": gates, "skipped_states": skipped, "selection_log": log,
            "chosen": chosen, "top20": ranked[:20], "all_candidates": cands}
