"""Basin enthalpy/entropy split and solvent share from recorded per-sample energies.

Spec: docs/superpowers/specs/2026-09-23-thermo-energy-decomposition/spec.md.

Channels (kJ/mol, one value per MBAR sample):

    W_own = u_nk[n, w_n]/beta - Delta(lambda_{w_n}; v_pep, v_dih)   own-window umbrella
    U     = potential - W_own          physical potential energy
    V_pep = v_pep                      E0 - E1 + E2 (exact Pep-GaMD kernel only)
    U_ee  = U - V_pep                  solvent/ion-solvent/ion nonbonded (= E1)
    V_dih = v_dih; V_nt = V_pep - V_dih

Basin differences use the same unbiased MBAR weights as the PMF: dG from the basin
population ratio, dH from conditional means of U, -TdS = dG - dH, and -TdS' = dG - dV_pep,
which follows from the solvent-reorganization identity (dU_ee enters dH and TdS equally).

Pure analysis: never touches OpenMM. Imports only from ``gareus`` (never the
``analyze_gareus_mbar`` script, see CLAUDE.md on second script copies).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

from gareus.math_helpers import ess
from gareus.units import KJ_PER_KCAL

from .data import _sample_block_ids, wjson
from .pmf import make_bins
from .solvers import norm_logw
from .writers import _write_csv_rows

#: The only boost type whose recorded v_pep is E0 - E1 + E2 with E1 the exact water-only PME.
EXACT_SPLIT_BOOST_TYPE = "pep-gamd-lower-dual"
TOTAL_CHANNELS = ("U",)
SPLIT_CHANNELS = ("V_pep", "U_ee", "V_dih", "V_nt")
#: A harmonic bias cannot be negative; allow float noise from the reduced-energy round trip.
NEGATIVE_UMBRELLA_TOL_KJ = 1e-6
#: Per-state median of beta*W_own above this is the self-bias HIGH threshold (chignolin_6 report).
SELF_BIAS_HIGH_KT = 10.0
MIN_BIN_SAMPLES = 50
#: Per-sample channels that only exist when saved frames were evaluated (spec §11).
FRAME_CHANNELS = ("V_pp", "V_pp_nb", "V_pp_bonded", "V_pe")
#: An auto split needs a barrier at least this many kT above the higher of the two minima.
AUTO_BASIN_MIN_BARRIER_KT = 1.0
OUT_DIR_NAME = "thermo_decomposition"


# --- inputs -----------------------------------------------------------------------------------

def resolve_boost_type(prod_dir) -> Optional[str]:
    """``gamd_boost_type`` from the nearest run manifest (prod dir, then up to two parents)."""
    base = Path(prod_dir)
    for d in (base, base.parent, base.parent.parent):
        for name in ("run_manifest.json", "run_args.json"):
            path = d / name
            if not path.is_file():
                continue
            try:
                doc = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            for section in (doc.get("method_settings"), doc.get("resolved_args"), doc):
                if isinstance(section, dict) and section.get("gamd_boost_type"):
                    return str(section["gamd_boost_type"])
    return None


def split_supported(boost_type: Optional[str]) -> bool:
    """The V_pep/U_ee split needs the exact kernel by name: an RS surrogate or stock boost is refused."""
    return boost_type == EXACT_SPLIT_BOOST_TYPE


def own_state_umbrella_kj(u_nk, window, beta, *, v_pep, v_dih, state_lambdas, envelope) -> np.ndarray:
    """Umbrella energy of each sample in its own window, with the ladder boost taken back out.

    ``apply_ladder_boost_to_u`` adds ``beta*pep_gamd_boost_kj(v_pep, v_dih, lambda_k, env)`` to
    every column k; this undoes exactly that for the sample's own column, with the same function.
    """
    u_nk = np.asarray(u_nk, dtype=np.float64)
    window = np.asarray(window, dtype=np.int64)
    w_own = u_nk[np.arange(window.size), window] / float(beta)
    if state_lambdas is None:
        return w_own
    lam_own = np.asarray(state_lambdas, dtype=np.float64)[window]
    active = lam_own > 0.0
    if not np.any(active):
        return w_own
    if envelope is None:
        raise ValueError("ladder states with gamd_lambda > 0 need the frozen Pep-GaMD envelope to recover W_own")
    v_pep = np.asarray(v_pep, dtype=np.float64)
    v_dih = np.asarray(v_dih, dtype=np.float64)
    out = w_own.copy()
    from gareus.pep_gamd import pep_gamd_boost_kj
    for lam in np.unique(lam_own[active]):
        sel = lam_own == lam
        out[sel] = w_own[sel] - np.asarray(pep_gamd_boost_kj(v_pep[sel], v_dih[sel], float(lam), envelope), dtype=np.float64)
    return out


def own_umbrella_diagnostics(w_own_kj, window, beta) -> dict:
    """Guards on W_own: never negative; per-state median of beta*W_own near 1 for a 2-DOF bias."""
    w_own_kj = np.asarray(w_own_kj, dtype=np.float64)
    window = np.asarray(window, dtype=np.int64)
    finite = np.isfinite(w_own_kj)
    neg = finite & (w_own_kj < -NEGATIVE_UMBRELLA_TOL_KJ)
    medians = {}
    for k in np.unique(window[finite]):
        medians[int(k)] = float(np.median(float(beta) * w_own_kj[finite & (window == k)]))
    vals = np.array(list(medians.values())) if medians else np.array([np.nan])
    return {
        "ok": bool(not np.any(neg)),
        "n_negative": int(np.count_nonzero(neg)),
        "min_kj": float(np.nanmin(w_own_kj)) if np.any(finite) else float("nan"),
        "median_reduced_by_state": medians,
        "median_reduced_max": float(np.nanmax(vals)),
        "n_states_self_bias_high": int(np.count_nonzero(vals > SELF_BIAS_HIGH_KT)),
    }


def observable_weights(d, logw):
    """Per-sample unbiased weights for energy averages, or None with a reason."""
    logw = np.asarray(logw, dtype=np.float64)
    if bool((d.meta or {}).get("gamd_ladder", False)):
        return norm_logw(logw), "mbar_ladder_exact", ""
    boost = np.asarray(d.boost_kj, dtype=np.float64) if d.boost_kj is not None else np.zeros(0)
    finite = boost[np.isfinite(boost)]
    if finite.size == 0 or float(np.max(np.abs(finite))) <= 1e-9:
        return norm_logw(logw), "mbar_unboosted", ""
    return None, "unsupported", ("stock GaMD without a lambda-ladder: no exact per-sample weights "
                                  "(exponential reweighting has near-zero ESS for sigma(U) ~ 500 kJ/mol)")


# --- basins -----------------------------------------------------------------------------------

def parse_basin_specs(specs: Optional[Sequence[str]]) -> list[tuple[str, float, float]]:
    out: list[tuple[str, float, float]] = []
    seen = set()
    for spec in specs or []:
        parts = str(spec).split(":")
        if len(parts) != 3:
            raise ValueError(f"--thermo-basin expects NAME:LO:HI, got {spec!r}")
        name, lo, hi = parts[0].strip(), float(parts[1]), float(parts[2])
        if not name or not (math.isfinite(lo) and math.isfinite(hi)) or lo >= hi:
            raise ValueError(f"--thermo-basin {spec!r}: need a name and finite LO < HI")
        if name in seen:
            raise ValueError(f"--thermo-basin: duplicate basin name {name!r}")
        seen.add(name)
        out.append((name, lo, hi))
    return out


def auto_basins(cv, w, n_bins, kbt_kcal, min_barrier_kt: float = AUTO_BASIN_MIN_BARRIER_KT) -> list[tuple[str, float, float]]:
    """Split CV1 at the highest PMF barrier between the two deepest minima; [] if only one.

    A barrier lower than ``min_barrier_kt`` above the higher minimum is histogram noise, not two basins.
    """
    cv = np.asarray(cv, dtype=np.float64)
    edges = make_bins(cv, int(n_bins), None, None)
    prob, _ = np.histogram(cv, bins=edges, weights=w)
    with np.errstate(divide="ignore"):
        F = -float(kbt_kcal) * np.log(prob / prob.sum())
    idx = [i for i in range(len(F)) if np.isfinite(F[i])
           and (i == 0 or not np.isfinite(F[i - 1]) or F[i] < F[i - 1])
           and (i == len(F) - 1 or not np.isfinite(F[i + 1]) or F[i] <= F[i + 1])]
    idx = [i for i in idx if 0 < i < len(F) - 1]  # interior minima only: an edge bin is not a basin
    if len(idx) < 2:
        return []
    a, b = sorted(sorted(idx, key=lambda i: F[i])[:2])
    if b - a < 2:
        return []
    between = np.arange(a + 1, b)
    top = int(between[np.nanargmax(F[between])])
    if not F[top] - max(F[a], F[b]) >= float(min_barrier_kt) * float(kbt_kcal):
        return []
    s = 0.5 * (edges[top] + edges[top + 1])
    return [("auto_low_cv", float(edges[0]), float(s)), ("auto_high_cv", float(s), float(edges[-1]))]


# --- estimator --------------------------------------------------------------------------------

def _region_sums(region, blocks, w, X, n_regions, n_blocks):
    """Per (region, block) sums of w and w*X_c. region = -1 means outside every region."""
    ok = region >= 0
    key = region[ok] * n_blocks + blocks[ok]
    size = n_regions * n_blocks
    S0 = np.bincount(key, weights=w[ok], minlength=size).reshape(n_regions, n_blocks)
    S1 = {c: np.bincount(key, weights=(w * x)[ok], minlength=size).reshape(n_regions, n_blocks) for c, x in X.items()}
    return S0, S1


def _stat(point, reps):
    reps = np.asarray(reps, dtype=np.float64)
    reps = reps[np.isfinite(reps)]
    if reps.size < 2:
        return {"estimate": float(point), "se": float("nan"), "ci95": [float("nan"), float("nan")]}
    return {"estimate": float(point), "se": float(np.std(reps, ddof=1)),
            "ci95": [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))]}


def _channels(d, boost_type, envelope_loader):
    """Per-sample channels plus availability notes; raises nothing, reports reasons."""
    notes: dict = {}
    if d.potential_kj is None or not np.any(np.isfinite(np.asarray(d.potential_kj, dtype=float))):
        return None, {"reason": "no finite per-sample potential energy (sample_potential_energy off?)"}
    lambdas = d.state_lambdas
    has_rungs = lambdas is not None and np.any(np.asarray(lambdas, dtype=float) > 0.0)
    envelope = envelope_loader(d.prod_dir) if has_rungs else None
    try:
        w_own = own_state_umbrella_kj(d.u_nk, d.window, d.beta, v_pep=d.v_pep_kj, v_dih=d.v_dih_kj,
                                      state_lambdas=lambdas if has_rungs else None, envelope=envelope)
    except ValueError as exc:
        return None, {"reason": f"own-window umbrella energy not recoverable: {exc}"}
    diag = own_umbrella_diagnostics(w_own, d.window, d.beta)
    notes["own_umbrella"] = diag
    if not diag["ok"]:
        return None, {"reason": (f"{diag['n_negative']} samples have negative own-window umbrella energy "
                                 f"(min {diag['min_kj']:.3g} kJ/mol): ladder-boost subtraction or u_nk is inconsistent"),
                      "own_umbrella": diag}
    X = {"U": np.asarray(d.potential_kj, dtype=np.float64) - w_own}
    split = split_supported(boost_type)
    notes["split_available"] = False
    if not split:
        notes["split_reason"] = (f"boost type {boost_type!r} is not {EXACT_SPLIT_BOOST_TYPE!r}: recorded v_pep is not "
                                 "E0 - E1 + E2 with an exact water-only PME, so U_ee = U - v_pep is undefined")
    elif (d.v_pep_kj is None or d.v_dih_kj is None
          or not np.any(np.isfinite(np.asarray(d.v_pep_kj, dtype=float)))
          or not np.any(np.isfinite(np.asarray(d.v_dih_kj, dtype=float)))):
        notes["split_reason"] = "no finite v_pep_kj_mol and v_dih_kj_mol recorded"
    else:
        v_pep = np.asarray(d.v_pep_kj, dtype=np.float64); v_dih = np.asarray(d.v_dih_kj, dtype=np.float64)
        X.update({"V_pep": v_pep, "U_ee": X["U"] - v_pep, "V_dih": v_dih, "V_nt": v_pep - v_dih})
        notes["split_available"] = True
        notes["split_reason"] = ""
    return X, notes


def _basin_regions(cv, basins):
    """One region array per basin: 0 inside the half-open [lo, hi), -1 outside.

    Half-open so an auto split point belongs to exactly one basin; explicit basins may still overlap.
    """
    return [np.where((cv >= lo) & (cv < hi), 0, -1) for _, lo, hi in basins]


def compute_thermo(d, logw, *, boost_type, basins, n_bins, n_boot, seed, min_basin_ess, min_basin_blocks,
                   envelope_loader: Callable, kbt_kcal: Optional[float] = None,
                   extra_channels: Optional[dict] = None, torsions: Optional[np.ndarray] = None,
                   entropy_bins: int = 24) -> dict:
    """Everything in spec §4-§6 (and §11 with frame inputs). ``basins`` empty -> CV profiles only.

    ``extra_channels`` adds per-sample energies (NaN where unknown), e.g. V_pp from saved frames;
    with V_pp and V_pep present, V_pe = V_pep - V_pp is derived. ``torsions`` (n_samples, M, radians,
    NaN rows where no frame) adds the per-basin MIE configurational entropy on the same bootstrap draws.
    Samples lacking any requested channel are excluded from every quantity, so all differences in one
    result describe one population.
    """
    w_all, weight_method, weight_reason = observable_weights(d, logw)
    if w_all is None:
        return {"available": False, "reason": weight_reason, "weights": weight_method}
    X, notes = _channels(d, boost_type, envelope_loader)
    if X is None:
        return {"available": False, **notes, "weights": weight_method, "boost_type": boost_type}
    for name, arr in (extra_channels or {}).items():
        X[name] = np.asarray(arr, dtype=np.float64)
    if "V_pp" in X and "V_pep" in X:
        X["V_pe"] = X["V_pep"] - X["V_pp"]
    cv = np.asarray(d.cv, dtype=np.float64)
    valid = np.isfinite(cv) & np.isfinite(w_all) & (w_all > 0)
    for x in X.values():
        valid &= np.isfinite(x)
    tors_bins = None
    if torsions is not None:
        from .thermo_entropy import torsion_bins
        torsions = np.asarray(torsions, dtype=np.float64)
        valid &= np.all(np.isfinite(torsions), axis=1)
    n_dropped = int(valid.size - np.count_nonzero(valid))
    if np.count_nonzero(valid) < 10:
        return {"available": False, "reason": "fewer than 10 samples with finite weights and energies",
                "weights": weight_method, "boost_type": boost_type}
    beta = float(d.beta)
    kT = 1.0 / beta
    kbt_kcal = kT / KJ_PER_KCAL if kbt_kcal is None else float(kbt_kcal)
    w = w_all[valid]
    cv_v = cv[valid]
    Xv = {c: x[valid] for c, x in X.items()}
    blocks_all = _sample_block_ids(d)
    _, blocks = np.unique(blocks_all[valid], return_inverse=True)
    B = int(blocks.max()) + 1
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(B, np.full(B, 1.0 / B), size=int(n_boot)).astype(np.float64) if n_boot > 0 else np.zeros((0, B))
    ones = np.ones(B)
    if torsions is not None:
        tors_bins = torsion_bins(torsions[valid], int(entropy_bins))

    result: dict = {
        "available": True, "weights": weight_method, "boost_type": boost_type,
        "split_available": bool(notes.get("split_available")), "split_reason": notes.get("split_reason", ""),
        "own_umbrella": notes.get("own_umbrella"), "n_samples": int(valid.size), "n_used": int(np.count_nonzero(valid)),
        "n_dropped_nonfinite": n_dropped, "n_blocks": B, "n_bootstrap": int(n_boot), "seed": seed,
        "channels": list(X), "temperature_K": float(d.temp), "pdV_neglected": True,
    }

    # basins
    basin_rows, basin_stats = [], []
    for (name, lo, hi), region in zip(basins, _basin_regions(cv_v, basins)):
        S0, S1 = _region_sums(region, blocks, w, Xv, 1, B)
        P = S0[0] @ ones
        inside = region >= 0
        e = ess(w[inside] / w[inside].sum()) if P > 0 else 0.0
        nb = int(np.count_nonzero(S0[0] > 0))
        ok = P > 0 and e >= min_basin_ess and nb >= min_basin_blocks
        means = {c: (S1[c][0] @ ones) / P if P > 0 else float("nan") for c in Xv}
        P_reps = S0[0] @ counts.T
        with np.errstate(divide="ignore", invalid="ignore"):  # a replicate may draw no block of this basin -> NaN
            mean_reps = {c: (S1[c][0] @ counts.T) / P_reps for c in Xv} if n_boot > 0 else {c: np.zeros(0) for c in Xv}
        ent = None
        if tors_bins is not None:
            from .thermo_entropy import basin_entropy
            ent = basin_entropy(tors_bins, w, blocks, counts, inside, int(entropy_bins))
        basin_stats.append({"P": P, "P_reps": P_reps, "means": means, "mean_reps": mean_reps, "ok": ok, "ent": ent})
        row = {"basin": name, "cv_lo": lo, "cv_hi": hi, "n_samples": int(np.count_nonzero(inside)),
               "weight_fraction": float(P / w.sum()), "ess": float(e), "n_blocks": nb, "sampling_ok": bool(ok)}
        if ent is not None:
            row.update({"S1_over_k": ent["point"]["S1"], "MI_over_k": ent["point"]["MI"], "S2_over_k": ent["point"]["S2"]})
        for c in Xv:
            st = _stat(means[c], mean_reps[c])
            row[f"mean_{c}_kj"] = st["estimate"]; row[f"se_{c}_kj"] = st["se"]
        basin_rows.append(row)
    result["basins"] = basin_rows

    diffs, replicates = [], {}
    for i in range(len(basins)):
        for j in range(i + 1, len(basins)):
            a, b = basin_stats[i], basin_stats[j]
            pair = f"{basins[i][0]}->{basins[j][0]}"
            with np.errstate(divide="ignore", invalid="ignore"):
                dG = -kT * math.log(b["P"] / a["P"]) if a["P"] > 0 and b["P"] > 0 else float("nan")
                dG_r = -kT * np.log(b["P_reps"] / a["P_reps"])
            q_point = {"dG_kj": dG, "dH_kj": b["means"]["U"] - a["means"]["U"]}
            q_reps = {"dG_kj": dG_r, "dH_kj": b["mean_reps"]["U"] - a["mean_reps"]["U"]}
            q_point["minus_TdS_kj"] = q_point["dG_kj"] - q_point["dH_kj"]
            q_reps["minus_TdS_kj"] = q_reps["dG_kj"] - q_reps["dH_kj"]
            if result["split_available"]:
                for c, key in (("V_pep", "dV_pep_kj"), ("U_ee", "dU_ee_kj"), ("V_dih", "dV_dih_kj"), ("V_nt", "dV_nt_kj")):
                    q_point[key] = b["means"][c] - a["means"][c]
                    q_reps[key] = b["mean_reps"][c] - a["mean_reps"][c]
                q_point["minus_TdS_prime_kj"] = q_point["dG_kj"] - q_point["dV_pep_kj"]
                q_reps["minus_TdS_prime_kj"] = q_reps["dG_kj"] - q_reps["dV_pep_kj"]
            for c in FRAME_CHANNELS:
                if c in Xv:
                    q_point[f"d{c}_kj"] = b["means"][c] - a["means"][c]
                    q_reps[f"d{c}_kj"] = b["mean_reps"][c] - a["mean_reps"][c]
            if a["ent"] is not None and b["ent"] is not None:
                for s in ("S1", "S2"):
                    q_point[f"TdS_conf_{s}_kj"] = kT * (b["ent"]["point"][s] - a["ent"]["point"][s])
                    q_reps[f"TdS_conf_{s}_kj"] = kT * (b["ent"]["reps"][s] - a["ent"]["reps"][s])
                # solvent (and every non-torsional) entropy by difference, with the best (S2) estimate
                q_point["minus_TdS_solv_kj"] = q_point["minus_TdS_kj"] + q_point["TdS_conf_S2_kj"]
                q_reps["minus_TdS_solv_kj"] = q_reps["minus_TdS_kj"] + q_reps["TdS_conf_S2_kj"]
                if "minus_TdS_prime_kj" in q_point:
                    q_point["minus_TdS_prime_solv_kj"] = q_point["minus_TdS_prime_kj"] + q_point["TdS_conf_S2_kj"]
                    q_reps["minus_TdS_prime_solv_kj"] = q_reps["minus_TdS_prime_kj"] + q_reps["TdS_conf_S2_kj"]
            entry = {"pair": pair, "status": "ok" if (a["ok"] and b["ok"]) else "inconclusive"}
            for k in q_point:
                entry[k] = _stat(q_point[k], q_reps[k])
            for k in ("dG", "dH", "minus_TdS"):
                entry[f"{k}_kcal"] = {"estimate": entry[f"{k}_kj"]["estimate"] / KJ_PER_KCAL,
                                      "se": entry[f"{k}_kj"]["se"] / KJ_PER_KCAL}
            entry["dS_J_per_mol_K"] = -1000.0 * entry["minus_TdS_kj"]["estimate"] / float(d.temp)
            diffs.append(entry)
            replicates[pair] = q_reps
    result["differences"] = diffs
    result["_replicates"] = replicates

    # CV1 profiles
    edges = make_bins(cv_v, int(n_bins), None, None)
    bi = np.clip(np.digitize(cv_v, edges[1:-1]), 0, len(edges) - 2)
    nb_ = len(edges) - 1
    S0, S1 = _region_sums(bi, blocks, w, Xv, nb_, B)
    n_in_bin = np.bincount(bi, minlength=nb_)
    P = S0 @ ones
    P_reps = S0 @ counts.T
    with np.errstate(divide="ignore", invalid="ignore"):
        F = -kbt_kcal * np.log(P / P.sum())
    usable = (n_in_bin >= MIN_BIN_SAMPLES) & (P > 0)
    ref = int(np.nanargmax(np.where(usable, P, -np.inf))) if np.any(usable) else 0
    Fmin = np.nanmin(F[usable]) if np.any(usable) else 0.0
    profile_rows = []
    for k in range(nb_):
        row = {"cv_center": float(0.5 * (edges[k] + edges[k + 1])), "n_samples": int(n_in_bin[k]),
               "pmf_kcal": float(F[k] - Fmin) if usable[k] else float("nan"), "usable": bool(usable[k]),
               "is_reference": k == ref}
        for c in Xv:
            with np.errstate(divide="ignore", invalid="ignore"):
                mk = S1[c][k] @ ones / P[k] - S1[c][ref] @ ones / P[ref]
                rk = (S1[c][k] @ counts.T) / P_reps[k] - (S1[c][ref] @ counts.T) / P_reps[ref]
            st = _stat(mk, rk) if usable[k] else {"estimate": float("nan"), "se": float("nan")}
            row[f"d_{c}_kj"] = st["estimate"]; row[f"se_d_{c}_kj"] = st["se"]
        profile_rows.append(row)
    result["cv_profiles"] = profile_rows
    result["cv_profile_reference_bin_center"] = profile_rows[ref]["cv_center"]
    return result


# --- driver -----------------------------------------------------------------------------------

def _plot_profiles(result, path, warnings):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # plotting is optional; the CSV/JSON are the deliverable
        warnings.append(f"thermo_decomposition: plot skipped ({exc})")
        return None
    rows = [r for r in result["cv_profiles"] if r["usable"]]
    if not rows:
        return None
    x = np.array([r["cv_center"] for r in rows])
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
    ax0.plot(x, [r["pmf_kcal"] for r in rows], color="black")
    ax0.set_ylabel("PMF (kcal/mol)")
    for c in result["channels"]:
        y = np.array([r[f"d_{c}_kj"] for r in rows]); se = np.array([r[f"se_d_{c}_kj"] for r in rows])
        ax1.plot(x, y, label=c)
        if np.any(np.isfinite(se)):
            ax1.fill_between(x, y - se, y + se, alpha=0.2)
    ax1.axhline(0.0, color="grey", lw=0.8)
    ax1.set_ylabel("<X>(s) - <X>(ref) (kJ/mol)")
    ax1.set_xlabel("CV1")
    ax1.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return str(path)


def _flatten_diff(entry):
    row = {"pair": entry["pair"], "status": entry["status"]}
    for k, v in entry.items():
        if isinstance(v, dict) and "estimate" in v:
            row[k] = v["estimate"]; row[f"{k}_se"] = v.get("se")
            if "ci95" in v:
                row[f"{k}_ci_lo"], row[f"{k}_ci_hi"] = v["ci95"]
    row["dS_J_per_mol_K"] = entry.get("dS_J_per_mol_K")
    return row


def analyze_thermo_decomposition(d, args, m, *, kbt_kcal, out, warnings, envelope_loader=None) -> dict:
    """Stage entry point for analyze_gareus_mbar.py; returns the pmf_summary.json block."""
    if bool(getattr(args, "no_thermo_decomposition", False)):
        return {"available": False, "reason": "disabled via --no-thermo-decomposition"}
    if envelope_loader is None:
        from .ladder import load_pep_gamd_envelope as envelope_loader
    boost_type = resolve_boost_type(d.prod_dir)
    try:
        basins = parse_basin_specs(getattr(args, "thermo_basin", None))
    except ValueError as exc:
        warnings.append(f"thermo_decomposition: {exc}")
        return {"available": False, "reason": str(exc)}
    n_bins = int(getattr(args, "thermo_bins", None) or getattr(args, "bins", 50) or 50)
    basin_source = "explicit" if basins else "auto"
    if not basins:
        w, _, _ = observable_weights(d, m["logw"])
        if w is not None:
            ok = np.isfinite(d.cv) & np.isfinite(w)
            basins = auto_basins(np.asarray(d.cv)[ok], w[ok], n_bins, kbt_kcal)
        if not basins:
            basin_source = "none"
    res = compute_thermo(d, m["logw"], boost_type=boost_type, basins=basins, n_bins=n_bins,
                         n_boot=int(getattr(args, "thermo_bootstrap", 200)), seed=int(getattr(args, "thermo_seed", 0)),
                         min_basin_ess=float(getattr(args, "thermo_min_basin_ess", 1000)),
                         min_basin_blocks=int(getattr(args, "thermo_min_basin_blocks", 8)),
                         envelope_loader=envelope_loader, kbt_kcal=kbt_kcal)
    res.pop("_replicates", None)
    res["basin_source"] = basin_source
    if not res.get("available"):
        warnings.append(f"thermo_decomposition unavailable: {res.get('reason')}")
        return res
    if basin_source == "auto":
        warnings.append("thermo_decomposition: basins split automatically at the CV1 PMF barrier (label 'auto'); "
                        "pass --thermo-basin NAME:LO:HI for headline numbers")
    if res.get("n_dropped_nonfinite", 0) > 0.01 * res["n_samples"]:
        warnings.append(f"thermo_decomposition: {res['n_dropped_nonfinite']} of {res['n_samples']} samples dropped for "
                        "non-finite energies/weights; basin weights are the global MBAR weights restricted to the rest")
    for dd in res["differences"]:
        if dd["status"] != "ok":
            warnings.append(f"thermo_decomposition: {dd['pair']} inconclusive (basin ESS/blocks below gate)")
    out_dir = Path(out) / OUT_DIR_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv_rows(out_dir / "thermo_basins.csv", res["basins"])
    _write_csv_rows(out_dir / "thermo_differences.csv", [_flatten_diff(x) for x in res["differences"]])
    _write_csv_rows(out_dir / "thermo_cv_profiles.csv", res["cv_profiles"])
    files = {"thermo_basins_csv": str(out_dir / "thermo_basins.csv"),
             "thermo_differences_csv": str(out_dir / "thermo_differences.csv"),
             "thermo_cv_profiles_csv": str(out_dir / "thermo_cv_profiles.csv"),
             "thermo_summary_json": str(out_dir / "thermo_summary.json")}
    png = _plot_profiles(res, out_dir / "thermo_cv_profiles.png", warnings)
    if png:
        files["thermo_cv_profiles_png"] = png
    if bool(getattr(args, "thermo_frames", False)):
        res["frames"] = _frame_decomposition(d, args, m, res, basins, n_bins, boost_type, envelope_loader,
                                             kbt_kcal, out_dir, files, warnings)
    res["files"] = files
    wjson(out_dir / "thermo_summary.json", res)
    return res


def _pick_platform(requested: Optional[str]) -> str:
    if requested and requested != "auto":
        return requested
    try:
        import openmm
        names = {openmm.Platform.getPlatform(i).getName() for i in range(openmm.Platform.getNumPlatforms())}
    except Exception:
        return "CPU"
    for name in ("CUDA", "OpenCL", "CPU"):
        if name in names:
            return name
    return "Reference"


def _frame_decomposition(d, args, m, res, basins, n_bins, boost_type, envelope_loader, kbt_kcal, out_dir, files, warnings):
    """Spec §11: peptide-only PME energies and torsion entropy from saved frames, gated by an alignment oracle."""
    from .thermo_frames import evaluate_sample_frames, peptide_energy_oracle_ok
    platform = _pick_platform(getattr(args, "thermo_frame_platform", "auto"))
    try:
        fr = evaluate_sample_frames(d, platform=platform, stride=int(getattr(args, "thermo_frame_stride", 1) or 1))
    except Exception as exc:
        warnings.append(f"thermo_decomposition frames unavailable: {exc}")
        return {"available": False, "reason": str(exc)}
    have = np.isfinite(fr["e_dih"])
    v_dih = np.asarray(d.v_dih_kj, dtype=float) if d.v_dih_kj is not None else np.full(have.size, np.nan)
    oracle = peptide_energy_oracle_ok(fr["e_dih"][have], v_dih[have])
    info = {"frame_alignment": oracle, "diag": fr["diag"]}
    if not oracle["ok"]:
        warnings.append(f"thermo_decomposition frames refused: offline torsion energy vs recorded v_dih median |diff| "
                        f"{oracle['median_abs_kj']:.3g} kJ/mol > {oracle.get('tol_kj', 2.0)} (frames misaligned with samples?)")
        return {"available": False, "reason": "frame/sample alignment oracle failed", **info}
    v_pp = fr["e_nb"] + fr["e_bond"] + fr["e_dih"]
    theta = None if bool(getattr(args, "thermo_no_entropy", False)) else fr["theta"]
    sub = compute_thermo(d, m["logw"], boost_type=boost_type, basins=basins, n_bins=n_bins,
                         n_boot=int(getattr(args, "thermo_bootstrap", 200)), seed=int(getattr(args, "thermo_seed", 0)),
                         min_basin_ess=float(getattr(args, "thermo_min_basin_ess", 1000)),
                         min_basin_blocks=int(getattr(args, "thermo_min_basin_blocks", 8)),
                         envelope_loader=envelope_loader, kbt_kcal=kbt_kcal,
                         extra_channels={"V_pp": v_pp, "V_pp_nb": fr["e_nb"], "V_pp_bonded": fr["e_bond"]},
                         torsions=theta, entropy_bins=int(getattr(args, "thermo_entropy_bins", 24)))
    sub.pop("_replicates", None)
    sub.update(info)
    sub["torsion_labels"] = fr["diag"].get("torsion_labels")
    sub["lra_electrostatic_solvation"] = {"available": False, "reason": (
        "needs the electrostatic part of U_pe, i.e. solvent coordinates; solute-only trajectories carry "
        "V_pe = coulomb + LJ + dispersion-correction difference only")}
    if not sub.get("available"):
        warnings.append(f"thermo_decomposition frames unavailable: {sub.get('reason')}")
        return sub
    np.savez_compressed(out_dir / "thermo_frame_energies.npz", e_nb=fr["e_nb"], e_bond=fr["e_bond"], e_dih=fr["e_dih"],
                        theta=fr["theta"], step=np.asarray(d.step), replica=np.asarray(d.replica))
    _write_csv_rows(out_dir / "thermo_frame_basins.csv", sub["basins"])
    _write_csv_rows(out_dir / "thermo_frame_differences.csv", [_flatten_diff(x) for x in sub["differences"]])
    files.update({"thermo_frame_basins_csv": str(out_dir / "thermo_frame_basins.csv"),
                  "thermo_frame_differences_csv": str(out_dir / "thermo_frame_differences.csv"),
                  "thermo_frame_energies_npz": str(out_dir / "thermo_frame_energies.npz")})
    return sub
