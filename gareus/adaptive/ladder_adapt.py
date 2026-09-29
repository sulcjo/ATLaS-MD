"""Adaptive lambda ladder: predict rung overlaps from data and re-place interior rungs.

At a fixed umbrella centre the rung states differ only in boost strength, so every sample's
reduced energy at any lambda is beta * pep_gamd_boost_kj(v_pep, v_dih, lambda, env) plus terms
common to all rungs of that centre (they cancel). A per-centre MBAR over the sampled rungs then
gives the ensemble at ANY lambda by reweighting, and the pairwise overlap between any two lambda
values. The boost is not linear in lambda (the dependent dual boost adds the dihedral boost
inside the Total channel's square), so energies are always recomputed, never interpolated.

Overlap scale: O(a, b) = integral p_a p_b / (p_a + p_b), 0.5 for identical ensembles -- the
two-state, equal-weight pairwise MBAR overlap (``mbar_analysis.ladder.pairwise_state_overlap``,
``min_rung_overlap`` / ``target_rung_overlap``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from gareus.pep_gamd import PepGamdEnvelope, pep_gamd_boost_kj


def _logsumexp(a, axis=None):
    a = np.asarray(a, dtype=float)
    m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    out = np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True)) + m
    return np.squeeze(out, axis=axis) if axis is not None else float(out.squeeze())


@dataclass(frozen=True)
class CentreRungModel:
    """Pooled rung samples of one umbrella centre and their MBAR free energies."""

    lambdas: np.ndarray            # sampled rungs, sorted
    n_k: np.ndarray                # samples per sampled rung
    f_k: np.ndarray                # reduced free energies (f_k[0] = 0)
    v_pep: np.ndarray              # pooled per-sample channel energies (kJ/mol)
    v_dih: np.ndarray
    beta: float                    # 1/(kJ/mol)
    env: PepGamdEnvelope
    _denom: np.ndarray = field(repr=False, compare=False)   # log sum_k N_k exp(f_k - u_k(n))
    _cache: dict = field(default_factory=dict, repr=False, compare=False)   # lambda -> log-weights

    def reduced(self, lam: float) -> np.ndarray:
        """beta * boost of every pooled sample under rung ``lam``."""
        return self.beta * np.asarray(pep_gamd_boost_kj(self.v_pep, self.v_dih, float(lam), self.env), dtype=float)

    def log_weights(self, lam: float) -> np.ndarray:
        """Normalised log-weights of the pooled samples in the ensemble at ``lam`` (cached)."""
        key = round(float(lam), 9)
        hit = self._cache.get(key)
        if hit is None:
            lw = -self.reduced(lam) - self._denom
            hit = lw - _logsumexp(lw)
            self._cache[key] = hit
        return hit

    def ess(self, lam: float) -> float:
        """Kish effective sample size of the reweighted ensemble at ``lam``."""
        return float(np.exp(-_logsumexp(2.0 * self.log_weights(lam))))

    def overlap(self, lam_a: float, lam_b: float) -> float:
        """O(a, b) = integral p_a p_b / (p_a + p_b); 0.5 for identical ensembles."""
        la, lb = self.log_weights(lam_a), self.log_weights(lam_b)
        # both are densities w.r.t. the same pooled mixture, so p_a/p_b = exp(la - lb) per sample
        return float(np.sum(np.exp(la) / (1.0 + np.exp(la - lb))))


def _mbar_f(u_kn: np.ndarray, n_k: np.ndarray, tol: float = 1e-10, max_iter: int = 20000) -> np.ndarray:
    """Self-consistent MBAR free energies (reduced units, f[0] = 0)."""
    f = np.zeros(u_kn.shape[0])
    log_n = np.log(n_k)
    for _ in range(max_iter):
        denom = _logsumexp(log_n[:, None] + f[:, None] - u_kn, axis=0)
        f_new = -_logsumexp(-u_kn - denom[None, :], axis=1)
        f_new = f_new - f_new[0]
        if np.max(np.abs(f_new - f)) < tol:
            return f_new
        f = f_new
    return f


def fit_centre_model(samples: Dict[float, Tuple[np.ndarray, np.ndarray]], env: PepGamdEnvelope, beta: float,
                     *, max_per_rung: int = 5000) -> Optional[CentreRungModel]:
    """Model of one centre from ``{lambda: (v_pep, v_dih)}``; None with fewer than two usable rungs.

    Samples with a non-finite channel the envelope uses are dropped per sample; each rung is
    thinned evenly to at most ``max_per_rung`` samples.
    """
    lams, vps, vds, ns = [], [], [], []
    for lam in sorted(samples):
        vp, vd = (np.asarray(x, dtype=float) for x in samples[lam])
        ok = np.isfinite(vd)
        if env.has_total:
            ok &= np.isfinite(vp)
        vp, vd = vp[ok], vd[ok]
        if vp.size == 0:
            continue
        if vp.size > max_per_rung:
            idx = np.linspace(0, vp.size - 1, max_per_rung).astype(int)
            vp, vd = vp[idx], vd[idx]
        lams.append(float(lam)); vps.append(vp); vds.append(vd); ns.append(vp.size)
    if len(lams) < 2:
        return None
    v_pep, v_dih = np.concatenate(vps), np.concatenate(vds)
    lambdas, n_k = np.asarray(lams), np.asarray(ns, dtype=float)
    u_kn = np.vstack([beta * np.asarray(pep_gamd_boost_kj(v_pep, v_dih, l, env), dtype=float) for l in lambdas])
    f_k = _mbar_f(u_kn, n_k)
    denom = _logsumexp(np.log(n_k)[:, None] + f_k[:, None] - u_kn, axis=0)
    return CentreRungModel(lambdas, n_k, f_k, v_pep, v_dih, float(beta), env, denom)


# ---------------------------------------------------------------------------
# Designer and planner
# ---------------------------------------------------------------------------

def _supports(m: CentreRungModel, lam: float, min_ess: float) -> bool:
    """A centre can predict ``lam`` only inside its sampled range and with enough reweighting ESS."""
    return float(m.lambdas[0]) - 1e-9 <= lam <= float(m.lambdas[-1]) + 1e-9 and m.ess(lam) >= min_ess


def aggregate_overlap(models, lam_a: float, lam_b: float, *, quantile: float = 0.10,
                      min_ess: float = 200.0) -> Optional[float]:
    """Quantile over centres of O(lam_a, lam_b); centres that cannot predict either value are skipped."""
    vals = [m.overlap(lam_a, lam_b) for m in models
            if m is not None and _supports(m, lam_a, min_ess) and _supports(m, lam_b, min_ess)]
    return float(np.quantile(vals, quantile)) if vals else None


@dataclass(frozen=True)
class LadderDesign:
    lambdas: Tuple[float, ...]
    predicted: Tuple[Optional[float], ...]     # adjacent-pair aggregate overlaps
    min_overlap: Optional[float]
    feasible: bool
    reason: str


def _predicted(models, lams, quantile, min_ess):
    return tuple(aggregate_overlap(models, a, b, quantile=quantile, min_ess=min_ess) for a, b in zip(lams, lams[1:]))


class _Grid:
    """Aggregate overlaps on a fixed lambda grid (the designer only places rungs on it)."""

    def __init__(self, models, lam_max, step, quantile, min_ess):
        n = max(2, int(round(lam_max / step)) + 1)
        self.lam = np.round(np.linspace(0.0, lam_max, n), 9)
        self.models, self.q, self.min_ess = models, quantile, min_ess
        self._memo: Dict[Tuple[int, int], Optional[float]] = {}

    def ov(self, i: int, j: int) -> Optional[float]:
        key = (i, j)
        if key not in self._memo:
            self._memo[key] = aggregate_overlap(self.models, float(self.lam[i]), float(self.lam[j]),
                                                quantile=self.q, min_ess=self.min_ess)
        return self._memo[key]

    def ok(self, i: int, j: int, level: float) -> bool:
        o = self.ov(i, j)
        return o is not None and o >= level

    def next_idx(self, i: int, level: float) -> int:
        """Largest j > i with overlap(i, j) >= level (overlap falls with distance); i if none."""
        last = len(self.lam) - 1
        if self.ok(i, last, level):
            return last
        lo, hi = i, last                       # ok(lo) (trivially for lo = i), not ok(hi)
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if self.ok(i, mid, level):
                lo = mid
            else:
                hi = mid
        return lo

    def greedy(self, level: float, max_rungs: int) -> Optional[List[int]]:
        idx, last = [0], len(self.lam) - 1
        while idx[-1] < last:
            nxt = self.next_idx(idx[-1], level)
            if nxt == idx[-1] or len(idx) >= max_rungs:
                return None
            idx.append(nxt)
        return idx


def design_ladder(models, *, lam_max: float = 1.0, target: float = 0.25, quantile: float = 0.10,
                  min_ess: float = 200.0, max_rungs: int = 8, grid_step: float = 0.005) -> LadderDesign:
    """Fewest rungs from 0 to ``lam_max`` with every adjacent aggregate overlap >= ``target``,
    interior rungs then placed so the adjacent overlaps are as equal (and high) as possible."""
    models = [m for m in models if m is not None]
    ends = [0.0, float(lam_max)]
    if not models or aggregate_overlap(models, 0.0, 0.0, quantile=quantile, min_ess=min_ess) is None \
            or aggregate_overlap(models, lam_max, lam_max, quantile=quantile, min_ess=min_ess) is None:
        return LadderDesign(tuple(ends), (None,), None, False, "no centre has support at both endpoints")
    g = _Grid(models, float(lam_max), float(grid_step), quantile, min_ess)
    first = g.greedy(target, max_rungs)
    if first is None:
        if g.greedy(target, 10_000) is not None:
            reason = f"target {target} not reachable within max_rungs={max_rungs}"
        else:
            reason = f"insufficient reweighting support to reach lambda={lam_max} at overlap {target}"
        return LadderDesign(tuple(ends), _predicted(models, ends, quantile, min_ess), None, False, reason)
    n, best = len(first), first
    lo, hi = float(target), 0.5
    for _ in range(25):                        # highest common level still reaching lam_max in n rungs
        mid = 0.5 * (lo + hi)
        cand = g.greedy(mid, n)
        if cand is not None:
            lo, best = mid, cand
        else:
            hi = mid
    lams = [float(g.lam[i]) for i in best]
    pred = _predicted(models, lams, quantile, min_ess)
    return LadderDesign(tuple(lams), pred, min(p for p in pred if p is not None), True, "ok")


@dataclass(frozen=True)
class LadderChange:
    drop: Tuple[float, ...]
    add: Tuple[float, ...]
    current_min: Optional[float]
    design: LadderDesign
    reason: str


def _min_pred(models, lams, quantile, min_ess) -> Optional[float]:
    pred = _predicted(models, sorted(lams), quantile, min_ess)
    if any(p is None for p in pred):
        return None
    return min(pred) if pred else None


def plan_ladder_change(current: Sequence[float], models, design: LadderDesign, *, target: float = 0.25,
                       quantile: float = 0.10, min_ess: float = 200.0, hysteresis: float = 0.03,
                       max_moves: int = 2, match_tol: float = 0.02) -> LadderChange:
    """Rung changes that move the current ladder toward ``design``, applied atomically.

    The whole change is applied at one epoch boundary, so intermediate ladders never run.
    A move is one interior rung changed (an add paired with a drop, or a lone add/drop); if the
    design needs at most ``max_moves`` moves the ladder becomes the design exactly (current rungs
    within ``match_tol`` of a design rung are kept as they are). Otherwise a partial change is
    chosen greedily: adds into the weakest gaps first, then drops that keep every adjacent
    overlap >= min(target, current minimum), until ``max_moves`` moves are used. Endpoints are
    never dropped. No change when the current ladder already meets the target with the design's
    rung count and the design's minimum overlap is not better by more than ``hysteresis``.
    """
    cur = sorted(float(x) for x in current)
    cur_min = _min_pred(models, cur, quantile, min_ess)
    if not design.feasible:
        return LadderChange((), (), cur_min, design, f"design infeasible: {design.reason}")
    ok_now = cur_min is not None and cur_min >= target
    if ok_now and len(cur) == len(design.lambdas) and (design.min_overlap or 0.0) - cur_min <= hysteresis:
        return LadderChange((), (), cur_min, design, "current ladder meets the target")
    ends = (cur[0], cur[-1])
    wanted = [y for y in design.lambdas[1:-1] if all(abs(y - x) > match_tol for x in cur)]
    spare = [x for x in cur[1:-1] if all(abs(x - y) > match_tol for y in design.lambdas)]
    if max(len(wanted), len(spare)) <= max(0, int(max_moves)):
        return LadderChange(tuple(sorted(spare)), tuple(sorted(wanted)), cur_min, design, "respace to design")
    floor = target if cur_min is None else min(target, cur_min)
    budget, new, added, dropped = max(0, int(max_moves)), list(cur), [], []

    def gap_overlap(y, lams):
        lams = sorted(lams)
        k = int(np.searchsorted(lams, y))
        o = aggregate_overlap(models, lams[k - 1], lams[min(k, len(lams) - 1)], quantile=quantile, min_ess=min_ess)
        return -1.0 if o is None else o

    for y in sorted(wanted, key=lambda v: gap_overlap(v, new)):
        if budget == 0:
            break
        new.append(y); added.append(y)
        # pair the add with a drop when one is admissible, so a move keeps the rung count
        best = None
        for x in spare:
            m = _min_pred(models, [v for v in new if v != x], quantile, min_ess)
            if m is not None and m >= floor and (best is None or m > best[1]):
                best = (x, m)
        if best is not None:
            new.remove(best[0]); spare.remove(best[0]); dropped.append(best[0])
        budget -= 1
    while budget > 0 and spare:
        best = None
        for x in spare:
            m = _min_pred(models, [v for v in new if v != x], quantile, min_ess)
            if m is not None and m >= floor and (best is None or m > best[1]):
                best = (x, m)
        if best is None:
            break
        new.remove(best[0]); spare.remove(best[0]); dropped.append(best[0]); budget -= 1
    assert ends[0] not in dropped and ends[1] not in dropped
    return LadderChange(tuple(sorted(dropped)), tuple(sorted(added)), cur_min, design,
                        "partial respace toward design" if (added or dropped) else "no admissible change")


# ---------------------------------------------------------------------------
# Loading rung samples from a campaign, and the dry-run replay
# ---------------------------------------------------------------------------

R_KJ_PER_MOL_K = 0.0083144626


def _restrained(k) -> bool:
    """k None = not recorded, i.e. the run's default restraint (same rule as the registry)."""
    if k is None:
        return True
    k = float(k)
    return bool(np.isfinite(k) and k > 0.0)


def centre_key(state: dict) -> tuple:
    """Umbrella centre identity with the restraint pattern (rungs of one centre share it).

    The coordinate of an unrestrained axis is a placeholder and never part of the identity
    (spec P6)."""
    on1 = _restrained(state.get("primary_k"))
    sc = state.get("secondary_center")
    on2 = sc is not None and np.isfinite(float(sc)) and _restrained(state.get("secondary_k"))
    return (round(float(state["primary_center"]), 6) if on1 else None, on1,
            round(float(sc), 6) if on2 else None, on2)


def phase_dirs(adaptive_dir, *, names: Optional[Sequence[str]] = None) -> List[tuple]:
    """``(phase_dir, window_map_path)`` for every campaign phase with Parquet samples.

    ``names`` filters by phase path relative to ``adaptive_dir`` (``"epoch_001"`` matches
    ``epoch_001`` and its ``baseline``/``topup_*`` sub-phases; ``"final/baseline"`` one phase).
    """
    from pathlib import Path
    from gareus.mbar_analysis.loaders_adaptive import _find_adaptive_epoch_dirs
    ad = Path(adaptive_dir)
    found = list(_find_adaptive_epoch_dirs(ad))
    for ext in sorted(ad.glob("final_extension_*")):          # extensions: flat phase directories
        if (ext / "samples").is_dir() and (ext / "epoch_window_map.csv").exists() \
                and all(Path(p) != ext for p, _ in found):
            found.append((ext, ext / "epoch_window_map.csv"))
    if names is None:
        return found
    keep = []
    for p, w in found:
        rel = str(Path(p).relative_to(ad))
        if any(rel == n or rel.startswith(n.rstrip("/") + "/") for n in names):
            keep.append((p, w))
    return keep


def _read_phase(phase_dir) -> "pd.DataFrame":
    import duckdb
    import re
    q = (f"select replica, step, window_id, gamd_lambda, v_pep_kj_mol, v_dih_kj_mol, filename "
         f"from read_parquet('{phase_dir}/samples/*/*.parquet', filename=true, union_by_name=true)")
    df = duckdb.connect().execute(q).df()
    df["seg"] = [int(m.group(1)) if (m := re.search(r"seg_(\d+)", f)) else 0 for f in df["filename"]]
    df = df.sort_values(["replica", "step", "seg"]).drop_duplicates(["replica", "step"], keep="last")
    return df.drop(columns=["filename", "seg"])


def load_centre_rung_samples(phases: Sequence[tuple], registry_states: Sequence[dict]
                             ) -> Dict[tuple, Dict[float, Tuple[np.ndarray, np.ndarray]]]:
    """``{centre_key: {lambda: (v_pep, v_dih)}}`` pooled over ``phases``.

    Rows are deduplicated on (replica, step), keeping the latest segment; local window ids go
    through each phase's (repaired) window map to state ids, and each state's lambda and centre
    come from the registry.
    """
    import csv
    from gareus.mbar_analysis.loaders_adaptive import _validate_and_repair_epoch_window_map
    by_id = {int(s["state_id"]): s for s in registry_states}
    acc: Dict[tuple, Dict[float, List[Tuple[np.ndarray, np.ndarray]]]] = {}
    for phase_dir, wmap_path in phases:
        with open(wmap_path) as fh:
            rows = list(csv.DictReader(fh))
        rows, _notes = _validate_and_repair_epoch_window_map(phase_dir, rows)
        local_to_state = {int(r["epoch_window"]): int(r["state_id"]) for r in rows}
        df = _read_phase(phase_dir)
        for w, grp in df.groupby("window_id"):
            sid = local_to_state.get(int(w))
            st = by_id.get(sid) if sid is not None else None
            if st is None:
                continue
            lam = round(float(st.get("gamd_lambda") or 0.0), 9)
            vp = grp["v_pep_kj_mol"].to_numpy(dtype=float, na_value=np.nan)
            vd = grp["v_dih_kj_mol"].to_numpy(dtype=float, na_value=np.nan)
            acc.setdefault(centre_key(st), {}).setdefault(lam, []).append((vp, vd))
    return {k: {lam: (np.concatenate([a for a, _ in parts]), np.concatenate([b for _, b in parts]))
                for lam, parts in rungs.items()} for k, rungs in acc.items()}


def _campaign_beta(adaptive_dir) -> float:
    import json
    from pathlib import Path
    ad = Path(adaptive_dir)
    for cand in (ad / "global_shared_gamd_setup" / "shared_gamd_setup_globals.json",
                 ad / "shared_gamd_setup_globals.json"):
        if cand.exists():
            t = json.loads(cand.read_text()).get("temperature_K")
            if t:
                return 1.0 / (R_KJ_PER_MOL_K * float(t))
    from gareus.io import resolve_run_temperature_k
    for p, _ in phase_dirs(ad):
        t = resolve_run_temperature_k(p)
        if t:
            return 1.0 / (R_KJ_PER_MOL_K * float(t))
    raise ValueError(f"{ad}: no temperature found for the campaign")


def replay(adaptive_dir, *, phases: Optional[Sequence[str]] = None, target: float = 0.25,
           quantile: float = 0.10, min_ess: float = 200.0, max_rungs: int = 8,
           max_moves: int = 2, max_per_rung: int = 5000) -> dict:
    """Read-only: what the adaptive ladder would conclude from a campaign's samples."""
    import json
    from pathlib import Path
    from gareus.mbar_analysis.ladder import load_pep_gamd_envelope
    ad = Path(adaptive_dir)
    env = load_pep_gamd_envelope(ad)
    if env is None:
        raise ValueError(f"{ad}: no shared_gamd_setup_globals.json (not a lambda-ladder campaign)")
    beta = _campaign_beta(ad)
    states = json.loads((ad / "state_registry.json").read_text())["states"]
    active = [s for s in states if s.get("active", True)]
    current = sorted({round(float(s.get("gamd_lambda") or 0.0), 9) for s in active})
    samples = load_centre_rung_samples(phase_dirs(ad, names=phases), states)
    models = [fit_centre_model(r, env, beta, max_per_rung=max_per_rung) for r in samples.values()]
    models = [m for m in models if m is not None]
    lam_max = float(current[-1])
    design = design_ladder(models, lam_max=lam_max, target=target, quantile=quantile, min_ess=min_ess,
                           max_rungs=max_rungs)
    change = plan_ladder_change(current, models, design, target=target, quantile=quantile,
                                min_ess=min_ess, max_moves=max_moves)
    median = tuple(aggregate_overlap(models, a, b, quantile=0.5, min_ess=min_ess) for a, b in zip(current, current[1:]))
    return {
        "adaptive_dir": str(ad), "phases": list(phases) if phases else "all",
        "n_centres": len(models), "current": [float(x) for x in current],
        "predicted_q": [None if v is None else float(v) for v in _predicted(models, current, quantile, min_ess)],
        "predicted_median": [None if v is None else float(v) for v in median],
        "settings": {"target": target, "quantile": quantile, "min_ess": min_ess, "max_rungs": max_rungs,
                     "max_moves": max_moves},
        "design": {"lambdas": [float(x) for x in design.lambdas],
                   "predicted": [None if v is None else float(v) for v in design.predicted],
                   "min_overlap": design.min_overlap, "feasible": design.feasible, "reason": design.reason},
        "change": {"drop": list(change.drop), "add": list(change.add), "reason": change.reason},
    }


def _main(argv=None) -> int:
    import argparse
    import json
    p = argparse.ArgumentParser(prog="python -m gareus.adaptive.ladder_adapt",
                                description="Adaptive lambda ladder: read-only replay on a campaign.")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("replay", help="what the adaptive ladder would propose from recorded samples")
    r.add_argument("adaptive_dir")
    r.add_argument("--phases", nargs="*", default=None, help="phase paths, e.g. final/baseline final_extension_001")
    r.add_argument("--target", type=float, default=0.25)
    r.add_argument("--quantile", type=float, default=0.10)
    r.add_argument("--min-ess", type=float, default=200.0)
    r.add_argument("--max-rungs", type=int, default=8)
    r.add_argument("--max-moves", type=int, default=2)
    r.add_argument("--json", default=None, help="write the report here")
    a = p.parse_args(argv)
    rep = replay(a.adaptive_dir, phases=a.phases, target=a.target, quantile=a.quantile, min_ess=a.min_ess,
                 max_rungs=a.max_rungs, max_moves=a.max_moves)
    text = json.dumps(rep, indent=2)
    if a.json:
        open(a.json, "w").write(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
