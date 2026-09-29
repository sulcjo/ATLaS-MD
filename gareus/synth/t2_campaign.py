"""T2 matched-budget campaigns: today's edge metric (marginal) vs spec 3.1 (pairwise-mbar).

Each arm runs the same epoch loop on the same starting layout and the same per-epoch MD
budget (total samples split uniformly over the arm's active states, as the topups-off
production allocator does):

    MALA chains  ->  adapter writes the phase  ->  REAL collect_epoch_diagnostics
    ->  REAL propose_actions_from_diagnostics  ->  REAL _apply_registry_actions

Arms (``ARMS``): ``marginal`` and ``pairwise-mbar`` differ only in
``AdaptiveDecisionPolicy.edge_metric``; ``marginal-noexchange`` is the marginal rule with no
exchange rows written (its acceptance < 0.08 test never fires: an exchange scheme whose
acceptance is inflated, e.g. gibbs-walk, or an edge the exchange graph does not attempt).
The synthetic acceptance is an independent-sample two-window Metropolis estimate (200
attempts per geometry edge), i.e. the best case for today's rule. Every state's chain
continues from its own last position; a state created by an action starts from the pooled
sample nearest its restraint centre (in sigma_w, the state-aware seed assignment's rule)
after a declared burn-in. Reported per epoch and arm: actions by kind, weak edges under
both metrics, the worst edge (analytic and estimated), connected components (the
collector's verdict and the analytic truth), pre-union vs post-union verdict flips, and
MBAR PMF errors of the cumulative union against the analytic surface.

Spec 3.3 (R1-R3) does not exist yet: a proposer that sees a CV2 gap can only place
today's midpoint bridge. ``RESOLUTION_TOLERANCE`` is the pre-set end-to-end criterion for
``slow-cv2-double-branch`` (both branches' PMF error), checked by a skipped test until
3.3 lands.

``python -m gareus.synth.t2_campaign --out DIR [--landscapes ...] [--seeds 0 1 2]``
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import collector_adapter as A
from .landscapes import LANDSCAPES
from .sampler import Window
from .vec_langevin import run_chains

D_COEF = (1.0, 0.05)
STEPS_PER_SAMPLE = 20
NEW_STATE_BURN_IN = 500
SAMPLES_PER_STATE_EPOCH0 = 2000
N_EPOCHS = 4
# pre-set before any campaign ran (kT); "resolved" also needs one analytic component
RESOLUTION_TOLERANCE = {"branch_delta_f_err_kT": 0.5, "cv2_pmf_rmse_lowf_kT": 0.5}
THRESHOLD = 0.15


def _grid(c1s, k1, c2s, k2) -> List[Window]:
    return [Window(float(a), float(k1), float(b), float(k2)) for a in c1s for b in c2s]


def _k_for_spacing(spacing: float) -> float:
    """Today's per-gap rule: sigma_w = spacing / 1.5, k = 1 / sigma_w^2 (reduced units)."""
    return 1.0 / (spacing / 1.5) ** 2


def split_spec(spec: str) -> Tuple[str, str]:
    """``"landscape:variant"`` -> (landscape, variant); no variant -> (landscape, "")."""
    name, _, variant = spec.partition(":")
    return name, variant


def initial_layout(spec: str) -> List[Window]:
    """A today-like uniform grid for each landscape (CV1 and CV2 per-gap springs).

    ``slow-cv2-double-branch:k2xF`` puts two CV2 rows at +-0.4, either side of the 6 kBT
    saddle, at F x the per-gap spring: the double well pushes both into the branches.
    Truth CV2-edge overlap: 0.28 (x1), 0.087 (x2, a gap MBAR still crosses), 0.008 (x4,
    disconnected). Default x4.
    """
    name, variant = split_spec(spec)
    if name in ("slow-cv2-double-branch", "slow-cv2-double-branch-asym"):
        factor = float(variant[3:]) if variant.startswith("k2x") else 4.0
        return _grid(np.linspace(0.2, 0.8, 5), _k_for_spacing(0.15), [-0.4, 0.4], factor * _k_for_spacing(0.8))
    if name == "gated-barrier":
        return _grid(np.linspace(0.1, 0.9, 7), _k_for_spacing(0.8 / 6), [-0.6, 0.0, 0.6], _k_for_spacing(0.6))
    if name in ("harmonic-bowl", "stiff-bowl"):
        return _grid(np.linspace(0.15, 0.85, 6), _k_for_spacing(0.14), [-0.5, 0.5], _k_for_spacing(1.0))
    if name == "narrow-cv2-band-at-high-cv1":
        return _grid(np.linspace(0.1, 0.95, 6), _k_for_spacing(0.17), [-1.4, 0.0, 1.4], _k_for_spacing(1.4))
    if name == "plateau-walls":
        return _grid(np.linspace(0.15, 0.85, 6), _k_for_spacing(0.14), [-0.6, -0.2, 0.2, 0.6],
                     _k_for_spacing(0.4))
    raise KeyError(name)


# ---------------------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------------------

def _dt(windows: Sequence[Window], stiff_extra: float = 0.0) -> np.ndarray:
    k1 = np.array([w.k1 for w in windows]); k2 = np.array([w.k2 or 0.0 for w in windows])
    return np.minimum(2.0e-3, 0.3 / np.maximum(D_COEF[0] * np.maximum(k1, 1.0), D_COEF[1] * (k2 + stiff_extra)))


def _nearest_start(w: Window, pool: np.ndarray) -> np.ndarray:
    d = w.k1 * (pool[:, 0] - w.center1) ** 2
    if w.center2 is not None and (w.k2 or 0) > 0:
        d = d + w.k2 * (pool[:, 1] - w.center2) ** 2
    return pool[int(np.argmin(d))]


def sample_epoch(name: str, windows: Dict[int, Window], starts: Dict[int, np.ndarray], n_per_state: int,
                 rng: np.random.Generator, burn_in: Dict[int, int], steps_per_sample: int = STEPS_PER_SAMPLE) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    ls = LANDSCAPES[name]
    sids = sorted(windows)
    ws = [windows[s] for s in sids]
    x0 = np.array([starts[s] for s in sids])
    extra = 2000.0 if name == "plateau-walls" else (400.0 if name == "stiff-bowl" else 0.0)
    kw = dict(c1=[w.center1 for w in ws], k1=[w.k1 for w in ws],
              c2=[w.center2 if w.center2 is not None else 0.0 for w in ws],
              k2=[w.k2 if w.center2 is not None and w.k2 else 0.0 for w in ws],
              D=D_COEF, dt=_dt(ws, extra), steps_per_sample=int(steps_per_sample))
    bi = max(burn_in.values()) if burn_in else 0
    if bi:   # one shared burn-in pass (continuing chains get it too: harmless, MALA is exact)
        x0 = run_chains(ls, x0, 1, burn_in=max(0, bi - int(steps_per_sample)), rng=rng, **kw).final
    run = run_chains(ls, x0, n_per_state, rng=rng, **kw)
    return ({s: run.samples[i] for i, s in enumerate(sids)}, {s: run.final[i] for i, s in enumerate(sids)})


# ---------------------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------------------

def _log_weights(u, window, f, n_k):
    from scipy.special import logsumexp  # noqa: PLC0415
    log_den = logsumexp(np.log(n_k)[None, :] + f[None, :] - u, axis=1)
    lw = -log_den
    return lw - logsumexp(lw)


def pmf_errors(name: str, pooled: np.ndarray, log_w: np.ndarray, res: int = 80) -> Dict[str, Any]:
    """Low-F-weighted RMSE (kT) of the MBAR PMF vs the analytic surface: cv1, cv2 marginals and 2D."""
    ls = LANDSCAPES[name]
    c1g, c2g, f = ls.grid(res=res)
    w = np.exp(log_w)
    out: Dict[str, Any] = {}
    for axis, (lo, hi), grid_axis in (("cv1", ls.cv1_bounds, 1), ("cv2", ls.cv2_bounds, 0)):
        edges = np.linspace(lo, hi, res + 1)
        x = pooled[:, 0 if axis == "cv1" else 1]
        dens, _ = np.histogram(x, bins=edges, weights=w)
        raw, _ = np.histogram(x, bins=edges)
        centres = 0.5 * (edges[1:] + edges[:-1])
        from scipy.special import logsumexp  # noqa: PLC0415
        ref = -logsumexp(-f, axis=grid_axis)
        ref = np.interp(centres, c1g if axis == "cv1" else c2g, ref)
        ref -= ref.min()
        ok = (raw >= 5) & (dens > 0)
        if ok.sum() < 3:
            out[f"{axis}_pmf_rmse_lowf_kT"] = None
            continue
        est = -np.log(dens[ok])
        diff = est - ref[ok]
        wt = np.exp(-ref[ok]); wt /= wt.sum()
        diff -= np.sum(wt * diff)
        out[f"{axis}_pmf_rmse_lowf_kT"] = round(float(np.sqrt(np.sum(wt * diff ** 2))), 4)
        lowf = ref <= 5.0
        out[f"{axis}_lowf_coverage"] = round(float(np.mean((raw >= 5)[lowf])), 3)
    # branch free-energy difference (CV2 sign split) vs exact
    p_pos_est = float(np.sum(w[pooled[:, 1] > 0.0]))
    pz = np.exp(-f)
    p_pos = float(pz[:, c2g > 0].sum() / pz.sum())
    if 0.0 < p_pos_est < 1.0:
        out["branch_delta_f_err_kT"] = round(abs(math.log(p_pos_est / (1 - p_pos_est)) - math.log(p_pos / (1 - p_pos))), 4)
    else:
        out["branch_delta_f_err_kT"] = None
    return out


def _components(n_ids: Sequence[int], edges: Sequence[Tuple[int, int]]) -> int:
    parent = {i: i for i in n_ids}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for a, b in edges:
        if a in parent and b in parent:
            parent[find(a)] = find(b)
    return len({find(i) for i in n_ids})


def edge_report(name: str, diag_pw: Dict[str, Any], diag_arm: Dict[str, Any], arm_policy, pw_policy,
                windows: Dict[int, Window], union) -> Dict[str, Any]:
    """Weak edges (both metrics), worst edges, components, pre/post-union flips."""
    from gareus.adaptive_production import _edge_is_measured_weak  # noqa: PLC0415
    ls = LANDSCAPES[name]
    sids, pooled, u, window, f, n_k = union
    idx = {s: i for i, s in enumerate(sids)}
    spatial = [e for e in diag_pw["edges"] if str(e.get("edge_type")) != "rung"]
    truth, est, flips, graph_true = [], [], 0, []
    truth_geom, est_geom, edge_rows = [], [], []
    n_post_weak = 0
    for e in spatial:
        a, b = int(e["state_i"]), int(e["state_j"])
        if a not in windows or b not in windows:
            continue
        t = A.exact_pair_overlap(ls, windows[a], windows[b])["overlap"]
        truth.append(t)
        if t >= THRESHOLD:
            graph_true.append((a, b))
        pm = e.get("pairwise_mbar") or {}
        if pm.get("overlap") is not None:
            est.append(pm["overlap"])
        geometry = str(e.get("edge_type")) not in ("neighbour", "spanning")
        if geometry:
            truth_geom.append(t)
            if pm.get("overlap") is not None:
                est_geom.append(pm["overlap"])
            edge_rows.append([a, b, str(e.get("edge_type")), round(t, 4),
                              None if e.get("overlap") is None else round(float(e["overlap"]), 3),
                              e.get("exchange_acceptance"),
                              None if pm.get("overlap") is None else round(pm["overlap"], 4),
                              None if pm.get("overlap_upper") is None else round(pm["overlap_upper"], 4),
                              pm.get("status")])
        pre = _edge_is_measured_weak(e, pw_policy)
        if a in idx and b in idx:
            e2 = copy.deepcopy(e)
            e2["mbar_overlap"] = A.union_pairwise_overlap(u, window, f, n_k, idx[a], idx[b])
            post = _edge_is_measured_weak(e2, pw_policy)
            n_post_weak += int(post)
            flips += int(pre != post)
    comps = (diag_pw.get("edge_metric") or {}).get("components") or {}
    geom_arm = [e for e in diag_arm["edges"] if str(e.get("edge_type")) != "rung"]
    return {
        "n_edges_graded": len(spatial),
        # the marginal predicate reads only overlap / exchange_acceptance, identical in both payloads
        "weak_marginal": len(A.weak_edges(diag_pw, type(pw_policy)())),
        "weak_pairwise": len(A.weak_edges(diag_pw, pw_policy)),
        "weak_pairwise_post_union": n_post_weak,
        "verdict_flips_pre_vs_post_union": flips,
        "worst_truth_overlap": None if not truth else round(min(truth), 4),
        "worst_estimated_overlap": None if not est else round(min(est), 4),
        "n_truth_below_threshold": int(sum(1 for t in truth if t < THRESHOLD)),
        "worst_truth_overlap_geometry": None if not truth_geom else round(min(truth_geom), 4),
        "worst_estimated_overlap_geometry": None if not est_geom else round(min(est_geom), 4),
        "n_truth_below_threshold_geometry": int(sum(1 for t in truth_geom if t < THRESHOLD)),
        "n_geometry_unmeasured_pairwise": int(sum(1 for r in edge_rows if r[8] != "ok")),
        "geometry_edges": edge_rows,
        "components_verdict": comps.get("n_components"),
        "components_truth": _components(sorted(windows), graph_true),
        "n_geometry_edges": len(geom_arm),
    }


# ---------------------------------------------------------------------------------------
# campaign
# ---------------------------------------------------------------------------------------

# arm -> (policy overrides, write exchange rows)
ARMS = {"marginal": ({"edge_metric": "marginal"}, True),
        "marginal-noexchange": ({"edge_metric": "marginal"}, False),
        "pairwise-mbar": ({"edge_metric": "pairwise-mbar"}, True)}
# extra arms, run on request (--arms): the proposed N_eff floor, and a no-action baseline
# (the same sampling budget with no adds or retirements: "just sample more")
EXTRA_ARMS = {"pairwise-mbar-neff100": ({"edge_metric": "pairwise-mbar", "min_edge_neff": 100.0}, True),
              "no-action": ({"edge_metric": "marginal", "max_new_windows_per_epoch": 0,
                             "retire_converged": False, "max_new_rungs_per_epoch": 0}, True)}
ALL_ARMS = {**ARMS, **EXTRA_ARMS}


def run_arm(name: str, arm: str, seed: int, *, n_epochs: int = N_EPOCHS, steps_per_sample: int = STEPS_PER_SAMPLE,
            samples_per_state0: int = SAMPLES_PER_STATE_EPOCH0, workdir: Optional[Path] = None,
            fmt: str = "csv") -> Dict[str, Any]:
    """One arm. ``marginal-noexchange`` writes no exchange rows (the marginal rule's
    acceptance test then never fires: the case of an exchange scheme whose acceptance is
    inflated, e.g. gibbs-walk, or of an edge the exchange graph does not attempt)."""
    from gareus.adaptive_production import AdaptiveDecisionPolicy, _apply_registry_actions  # noqa: PLC0415
    overrides, with_exchange = ALL_ARMS[arm]
    metric = overrides["edge_metric"]
    policy = AdaptiveDecisionPolicy(**overrides)
    pw_policy = policy if metric == "pairwise-mbar" else AdaptiveDecisionPolicy(edge_metric="pairwise-mbar")
    layout = initial_layout(name)
    spec, name = name, split_spec(name)[0]
    registry = A.registry_from_windows(layout)
    budget = samples_per_state0 * len(layout)
    base = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix=f"t2camp_{name}_{metric}_"))
    all_samples: Dict[int, List[np.ndarray]] = {}
    all_windows: Dict[int, Window] = {}
    last_pos: Dict[int, np.ndarray] = {}
    records = []
    for epoch in range(n_epochs):
        rng = np.random.default_rng([seed, epoch])
        active = registry.active_states()
        windows = {int(s.state_id): A.window_of_state(s) for s in active}
        all_windows.update(windows)
        pool = np.concatenate([np.concatenate(v) for v in all_samples.values()]) if all_samples else None
        starts, burn = {}, {}
        for sid, w in windows.items():
            if sid in last_pos:
                starts[sid] = last_pos[sid]
            elif pool is not None:
                starts[sid] = _nearest_start(w, pool); burn[sid] = NEW_STATE_BURN_IN
            else:
                starts[sid] = np.array([w.center1, w.center2 if w.center2 is not None else 0.0])
                burn[sid] = NEW_STATE_BURN_IN
        n_per = max(50, budget // len(windows))
        samples, final = sample_epoch(name, windows, starts, n_per, rng, burn, steps_per_sample)
        last_pos.update(final)
        for sid, x in samples.items():
            all_samples.setdefault(sid, []).append(x)
        edir = base / f"epoch_{epoch:03d}"
        A.write_phase_dir(edir, samples, windows, rng=rng, fmt=fmt,
                          exchange_pairs=A.geometry_pairs(registry) if with_exchange else ())
        diag_arm = A.collect(edir, registry, policy)
        diag_pw = diag_arm if metric == "pairwise-mbar" else A.collect(edir, registry, pw_policy)
        cum = {s: np.concatenate(v) for s, v in all_samples.items()}
        union = A.union_mbar(cum, all_windows)
        sids, pooled, u, window, f, n_k = union
        pmf = pmf_errors(name, pooled, _log_weights(u, window, f, n_k))
        edges = edge_report(name, diag_pw, diag_arm, policy, pw_policy, windows, union)
        actions = A.propose(registry, diag_arm, policy)
        refused = _apply_registry_actions(registry, actions, epoch + 1, policy=policy)
        records.append({"epoch": epoch, "n_states": len(windows), "samples_per_state": int(n_per),
                        "samples_total": int(n_per * len(windows)), "actions": A.action_counts(actions),
                        "n_refused": len(refused), **edges, **pmf,
                        "added_states": [[round(float(a[2][0]), 4), float(a[2][1]),
                                          None if a[2][2] is None else round(float(a[2][2]), 4),
                                          None if a[2][3] is None else float(a[2][3])]
                                         for a in actions if a and a[0] == "add"]})
    if workdir is None:
        shutil.rmtree(base, ignore_errors=True)
    final = records[-1]
    resolved = is_resolved(final)
    return {"landscape": spec, "arm": arm, "metric": metric, "seed": seed, "epochs": records,
            "resolved_per_tolerance": bool(resolved)}


def is_resolved(record: Dict[str, Any]) -> bool:
    """The pre-set end-to-end criterion: both PMF tolerances met and one analytic component."""
    return bool(all(record.get(k) is not None and record[k] <= tol for k, tol in RESOLUTION_TOLERANCE.items())
                and record.get("components_truth") == 1)


def _job(args):
    name, metric, seed, sps = args
    t0 = time.time()
    out = run_arm(name, metric, seed, steps_per_sample=sps)
    out["steps_per_sample"] = sps
    out["wall_s"] = round(time.time() - t0, 1)
    return out


DEFAULT_LANDSCAPES = ("slow-cv2-double-branch:k2x2", "slow-cv2-double-branch:k2x4",
                      "slow-cv2-double-branch-asym:k2x2", "slow-cv2-double-branch-asym:k2x4", "gated-barrier", "harmonic-bowl", "stiff-bowl",
                      "narrow-cv2-band-at-high-cv1", "plateau-walls")


SUMMARY_KEYS = ("n_states", "weak_marginal", "weak_pairwise", "weak_pairwise_post_union",
                "verdict_flips_pre_vs_post_union", "worst_truth_overlap_geometry",
                "worst_estimated_overlap_geometry", "n_truth_below_threshold_geometry",
                "n_geometry_unmeasured_pairwise", "components_verdict", "components_truth",
                "cv1_pmf_rmse_lowf_kT", "cv2_pmf_rmse_lowf_kT", "branch_delta_f_err_kT")


def summarise(results: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Per landscape x arm: seed means of the epoch-0 and final-epoch metrics, total actions."""
    out = []
    for spec in sorted({r["landscape"] for r in results}):
        for arm in ALL_ARMS:
            runs = [r for r in results if r["landscape"] == spec and r["arm"] == arm]
            if not runs:
                continue
            row: Dict[str, Any] = {"landscape": spec, "arm": arm, "n_seeds": len(runs)}
            for when, idx in (("epoch0", 0), ("final", -1)):
                for k in SUMMARY_KEYS:
                    vals = [r["epochs"][idx].get(k) for r in runs]
                    vals = [float(v) for v in vals if v is not None]
                    row[f"{when}_{k}"] = None if not vals else round(float(np.mean(vals)), 4)
            tot: Dict[str, float] = {}
            for r in runs:
                for e in r["epochs"]:
                    for kind, n in e["actions"].items():
                        tot[kind] = tot.get(kind, 0) + n / len(runs)
            row["actions_per_campaign"] = {k: round(v, 2) for k, v in sorted(tot.items())}
            row["resolved_fraction_epoch0"] = float(np.mean([is_resolved(r["epochs"][0]) for r in runs]))
            row["resolved_fraction"] = float(np.mean([is_resolved(r["epochs"][-1]) for r in runs]))
            out.append(row)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", required=True)
    p.add_argument("--landscapes", nargs="*", default=list(DEFAULT_LANDSCAPES))
    p.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--steps-per-sample", type=int, default=STEPS_PER_SAMPLE,
                   help="MALA steps between recorded samples (sets the per-state N_eff regime)")
    p.add_argument("--arms", nargs="*", default=None, choices=sorted(ALL_ARMS),
                   help=f"default: {sorted(ARMS)}")
    p.add_argument("--summarise-only", action="store_true", help="re-summarise an existing campaigns.json")
    a = p.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.summarise_only:
        results = json.loads((out / "campaigns.json").read_text())
    else:
        jobs = [(n, m, s, a.steps_per_sample) for n in a.landscapes for s in a.seeds for m in (a.arms or ARMS)]
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            results = list(ex.map(_job, jobs))
        (out / "campaigns.json").write_text(json.dumps(results, indent=1, default=float))
    (out / "campaigns_summary.json").write_text(json.dumps(summarise(results), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
