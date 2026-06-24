"""Campaign drivers: loop over the REAL adaptive decision code with synthetic data.

The harness owns only the thin round/epoch *glue* that ``*_auto_loop`` wraps
around the decision functions (window bookkeeping, proposal application, round
accounting).  The decisions themselves are made by shipped, unmodified code:

* feedback:   ``gareus.adaptive_feedback.run_adaptive_feedback_dispatcher_2d``
* production: ``gareus.adaptive_production.WindowStateRegistry`` + policy retire

This glue is the harness's own and is explicitly out of test scope; tests assert
on the decision-function outputs, not on the glue.
"""
from __future__ import annotations

import csv
import math
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .argspec import make_feedback_args
from .exchange import build_exchange_stats
from .ess import effective_count, tau_int, thin_to_ess
from .landscapes import Landscape
from .sampler import BIAS, Window, sample_window


# ---------------------------------------------------------------------------
# Feedback campaign (create / remove / shift / recompute-k / 2D patch)
# ---------------------------------------------------------------------------

@dataclass
class RoundRecord:
    round_index: int
    centers1: list
    k1: list
    centers2: list
    k2: list
    n_windows: int
    overlap_mean: float
    exchange_frac: float
    converged: bool
    added: list = field(default_factory=list)
    removed: list = field(default_factory=list)
    shifted: list = field(default_factory=list)


def _cross(centers1, k1, centers2, k2):
    """Rectangular grid flattened in ``window = ip*n_secondary + js`` order."""
    full_c1, full_k1, full_c2, full_k2 = [], [], [], []
    ns = len(centers2)
    for ip, (c1, kk1) in enumerate(zip(centers1, k1)):
        for js, (c2, kk2) in enumerate(zip(centers2, k2)):
            full_c1.append(float(c1))
            full_k1.append(float(kk1))
            full_c2.append(float(c2))
            full_k2.append(float(kk2))
    return full_c1, full_k1, full_c2, full_k2, len(centers1), ns


def _write_samples_csv(path: Path, samples_by_window: dict) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["window", "cv_A", "secondary_cv"])
        w.writeheader()
        for wi in sorted(samples_by_window):
            for (cv1, cv2) in samples_by_window[wi]:
                w.writerow({"window": wi, "cv_A": float(cv1), "secondary_cv": float(cv2)})


def _mean_neighbor_overlap_from_samples(samples_by_window, n_primary, n_secondary):
    """Empirical CV1 histogram-intersection overlap between adjacent columns."""
    cols = {}
    for ip in range(n_primary):
        vals = []
        for js in range(n_secondary):
            wi = ip * n_secondary + js
            s = samples_by_window.get(wi)
            if s is not None and len(s):
                vals.extend(np.asarray(s)[:, 0].tolist())
        cols[ip] = np.asarray(vals, float)
    ovs = []
    for ip in range(n_primary - 1):
        a, b = cols.get(ip), cols.get(ip + 1)
        if a is None or b is None or a.size == 0 or b.size == 0:
            continue
        lo = min(a.min(), b.min())
        hi = max(a.max(), b.max())
        if hi <= lo:
            continue
        bins = np.linspace(lo, hi, 41)
        ha, _ = np.histogram(a, bins=bins, density=True)
        hb, _ = np.histogram(b, bins=bins, density=True)
        width = bins[1] - bins[0]
        ovs.append(float(np.sum(np.minimum(ha, hb)) * width))
    return float(np.mean(ovs)) if ovs else float("nan")


def feedback_campaign(landscape: Landscape, *, initial_centers1, initial_k1,
                      initial_centers2, initial_k2, mode: str = "exact",
                      beta: float = 1.0, n_rounds: int = 5,
                      samples_per_window: int = 1500, res: int = 120, seed: int = 0,
                      target_overlap: float = 0.30, aggressiveness: str = "balanced",
                      tmp_dir: Optional[str] = None, sampler_kwargs: Optional[dict] = None) -> list:
    """Run synthetic adaptive-feedback rounds against the real 2D dispatcher.

    Returns a list of ``RoundRecord``; stops on ``converged`` or ``n_rounds``.
    """
    from gareus.adaptive_feedback import run_adaptive_feedback_dispatcher_2d

    rng = np.random.default_rng(seed)
    centers1 = [float(x) for x in initial_centers1]
    k1 = [float(x) for x in initial_k1]
    centers2 = [float(x) for x in initial_centers2]
    k2 = [float(x) for x in initial_k2]
    adaptive_memory = None
    skw = dict(sampler_kwargs or {})
    records: list = []
    base_tmp = Path(tmp_dir) if tmp_dir else Path(tempfile.mkdtemp(prefix="synth_fb_"))
    base_tmp.mkdir(parents=True, exist_ok=True)

    for r in range(n_rounds):
        fc1, fk1, fc2, fk2, n_primary, n_secondary = _cross(centers1, k1, centers2, k2)
        samples_by_window = {}
        windows = []
        for wi in range(len(fc1)):
            w = Window(fc1[wi], fk1[wi], fc2[wi], fk2[wi])
            windows.append(w)
            samples_by_window[wi] = sample_window(
                landscape, w, samples_per_window, mode=mode, beta=beta, rng=rng, res=res, **skw)

        round_dir = base_tmp / f"round_{r:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        _write_samples_csv(round_dir / "samples.csv", samples_by_window)
        exchange_stats = build_exchange_stats(windows, samples_by_window, beta=beta,
                                              rng=rng, n_primary=n_primary,
                                              n_secondary=n_secondary)

        args = make_feedback_args(target_overlap=target_overlap, round_index=r + 1,
                                  aggressiveness=aggressiveness,
                                  secondary_cv_centers=tuple(centers2),
                                  adaptive_memory=adaptive_memory)
        sec_meta = {"range_min": -1.0, "range_max": 1.0, "explicit_2d_windows": False}
        # The dispatcher indexes secondary[window_id], so pass the FULL per-window
        # secondary arrays (length nwin); it derives unique axis centers itself.
        proposal = run_adaptive_feedback_dispatcher_2d(
            args, round_dir, fc1, fk1, exchange_stats, fc2, fk2, sec_meta,
            fallback_history_by_window=None)

        ex_frac = (exchange_stats["accepted"] / exchange_stats["attempts"]
                   if exchange_stats["attempts"] else float("nan"))
        ov = _mean_neighbor_overlap_from_samples(samples_by_window, n_primary, n_secondary)

        if not proposal:
            records.append(RoundRecord(r, list(centers1), list(k1), list(centers2),
                                       list(k2), len(fc1), ov, ex_frac, True))
            break

        new_c1 = [float(x) for x in proposal.get("proposed_centers_A", centers1)] or centers1
        new_k1 = [float(x) for x in proposal.get("proposed_k_kcal_mol_A2", k1)] or k1
        new_c2 = [float(x) for x in proposal.get("proposed_secondary_cv_centers", centers2)] or centers2
        new_k2 = [float(x) for x in proposal.get("proposed_secondary_cv_k_kcal_mol", k2)] or k2
        # keep k lists aligned with centers (axis-factorized proposal)
        if len(new_k1) != len(new_c1):
            new_k1 = [float(np.median(k1))] * len(new_c1)
        if len(new_k2) != len(new_c2):
            new_k2 = [float(np.median(k2))] * len(new_c2)
        adaptive_memory = proposal.get("adaptive_memory_after", adaptive_memory)

        records.append(RoundRecord(
            r, list(new_c1), list(new_k1), list(new_c2), list(new_k2),
            len(new_c1) * max(1, len(new_c2)), ov, ex_frac,
            bool(proposal.get("converged", False)),
            added=list(proposal.get("added_centers_A", proposal.get("added_centers", [])) or []),
            removed=list(proposal.get("removed_windows", []) or []),
            shifted=list(proposal.get("shifted_centers", []) or [])))

        centers1, k1, centers2, k2 = new_c1, new_k1, new_c2, new_k2
        if bool(proposal.get("converged", False)):
            break

    return records


def sample_final_layout(landscape: Landscape, record, *, mode: str = "exact",
                        beta: float = 1.0, samples_per_window: int = 1500,
                        res: int = 120, seed: int = 12345, sampler_kwargs=None):
    """Sample the windows of a finished feedback layout (for PMF-recovery metric).

    Returns ``(samples_by_window, windows_by_window)`` where both are dicts keyed
    by window index; ``windows_by_window[i]`` is the ``Window`` so the umbrella
    bias can be removed in MBAR reweighting.
    """
    rng = np.random.default_rng(seed)
    fc1, fk1, fc2, fk2, _np_, _ns_ = _cross(record.centers1, record.k1,
                                            record.centers2, record.k2)
    skw = dict(sampler_kwargs or {})
    samples, windows = {}, {}
    for wi in range(len(fc1)):
        w = Window(fc1[wi], fk1[wi], fc2[wi], fk2[wi])
        windows[wi] = w
        samples[wi] = sample_window(landscape, w, samples_per_window, mode=mode,
                                    beta=beta, rng=rng, res=res, **skw)
    return samples, windows


# ---------------------------------------------------------------------------
# Production campaign (top-up / retire) -- Task 9
# ---------------------------------------------------------------------------

@dataclass
class EpochRecord:
    epoch: int
    active_ids: list
    n_active: int
    retired_this_epoch: list = field(default_factory=list)
    added_this_epoch: list = field(default_factory=list)
    extended_this_epoch: list = field(default_factory=list)
    sample_counts: dict = field(default_factory=dict)


def _default_production_ladder(landscape):
    """Over-dense CV1 ladder with one near-duplicate so retire has work."""
    lo, hi = landscape.cv1_bounds
    base = list(np.linspace(lo + 0.05 * (hi - lo), hi - 0.05 * (hi - lo), 10))
    # inject a near-duplicate of the middle state
    mid = base[5]
    base.insert(6, mid + 1e-3 * (hi - lo))
    return [(float(c), 50.0) for c in base]


def _pair_overlap_exchange(wi: Window, wj: Window, si, sj, *, beta, rng,
                           hist_overlap):
    """Real histogram-intersection overlap + synthetic Metropolis acceptance for a pair.

    Overlap is computed on the axis along which the two windows are actually
    neighbors (the larger center separation), using the SHIPPED overlap
    functional so it matches what the real diagnostics use.
    """
    si = np.asarray(si, float)
    sj = np.asarray(sj, float)
    d1 = abs(float(wi.center1) - float(wj.center1))
    d2 = abs(float(wi.center2 or 0.0) - float(wj.center2 or 0.0))
    axis = 0 if d1 >= d2 else 1
    a, b = si[:, axis], sj[:, axis]
    lo = float(min(a.min(), b.min()))
    hi = float(max(a.max(), b.max()))
    overlap = float(hist_overlap(a, b, lo, hi))
    # synthetic Metropolis swap acceptance (F cancels)
    n = int(min(200, a.size, b.size))
    ai = rng.integers(0, si.shape[0], n)
    aj = rng.integers(0, sj.shape[0], n)
    xi, xj = si[ai], sj[aj]
    delta = (BIAS(wi, xj[:, 0], xj[:, 1]) + BIAS(wj, xi[:, 0], xi[:, 1])
             - BIAS(wi, xi[:, 0], xi[:, 1]) - BIAS(wj, xj[:, 0], xj[:, 1]))
    acc = np.minimum(1.0, np.exp(-beta * delta))
    accepted = int(np.sum(rng.uniform(0.0, 1.0, n) < acc))
    exch = float(accepted) / float(n) if n else float("nan")
    return overlap, exch


def production_campaign(landscape: Landscape, *, initial_states=None, policy=None,
                        mode: str = "exact", beta: float = 1.0, n_epochs: int = 4,
                        samples_per_window: int = 1500, res: int = 120, seed: int = 0,
                        target_overlap: float = 0.30,
                        sample_count_per_epoch=None, initial_cumulative=None) -> list:
    """Run synthetic adaptive-production epochs against the REAL decision pipeline.

    Each epoch samples every active state, builds a faithful epoch-diagnostics
    dict (real geometry edges + the shipped histogram-overlap functional), then
    calls the real ``propose_actions_from_diagnostics`` and
    ``_apply_registry_actions``.  The add / retire / extend (top-up) decisions,
    graph articulation/connectivity checks, and the ``min_active_states`` floor
    are all shipped code -- the harness only assembles the diagnostics + samples.

    Sample accounting (drives top-up): each state accumulates
    ``sample_count_per_epoch`` (default ``samples_per_window``) per epoch it is
    active, starting at ``initial_cumulative`` (default ``samples_per_window``).
    A freshly added state starts at 0, so it is reported under the retire floor
    until it has accrued enough -- which is what makes the real decision emit
    ``extend`` (top-up) actions.  ``samples_per_window`` is the number of points
    drawn for overlap/exchange statistics, independent of the accrual.
    """
    from gareus.adaptive_production import (AdaptiveDecisionPolicy,
                                            WindowStateRegistry,
                                            build_geometry_edges,
                                            propose_actions_from_diagnostics,
                                            _apply_registry_actions)
    from gareus.math_helpers import _adaptive_hist_overlap

    rng = np.random.default_rng(seed)
    policy = policy or AdaptiveDecisionPolicy()
    accrual = int(sample_count_per_epoch if sample_count_per_epoch is not None else samples_per_window)
    init_cum = int(initial_cumulative if initial_cumulative is not None else samples_per_window)
    registry = WindowStateRegistry()
    states = initial_states if initial_states is not None else _default_production_ladder(landscape)
    cumulative: dict = {}
    for (c1, k1) in states:
        st = registry.add_state(primary_center=float(c1), primary_k=float(k1),
                                secondary_center=0.0, secondary_k=60.0, epoch=0,
                                source="seed")
        cumulative[int(st.state_id)] = init_cum

    records: list = []
    for epoch in range(1, n_epochs + 1):
        active = registry.active_states()
        if not active:
            break

        # sample each active state; accrue its cumulative sample_count
        win_by_id, samp_by_id, count_by_id = {}, {}, {}
        for st in active:
            sid = int(st.state_id)
            w = Window(st.primary_center, st.primary_k, st.secondary_center, st.secondary_k)
            win_by_id[sid] = w
            samp_by_id[sid] = sample_window(landscape, w, samples_per_window,
                                            mode=mode, beta=beta, rng=rng, res=res)
            cumulative[sid] = cumulative.get(sid, 0) + accrual  # fresh states start at 0
            count_by_id[sid] = cumulative[sid]

        # build faithful diagnostics dict for the real decision function
        edges = []
        for a, b, etype, nd in build_geometry_edges(registry):
            if a not in samp_by_id or b not in samp_by_id:
                continue
            ov, exch = _pair_overlap_exchange(
                win_by_id[a], win_by_id[b], samp_by_id[a], samp_by_id[b],
                beta=beta, rng=rng, hist_overlap=_adaptive_hist_overlap)
            edges.append({"state_i": a, "state_j": b, "edge_type": etype,
                          "normalized_distance": nd, "overlap": ov,
                          "exchange_acceptance": exch})
        diagnostics = {
            "schema_version": "adaptive_production_epoch_diagnostics_v1",
            "states": [{"state_id": sid, "sample_count": count_by_id[sid],
                        "gamd_boost_sd_kcal_mol": 0.0} for sid in samp_by_id],
            "edges": edges,
        }

        actions = propose_actions_from_diagnostics(registry, diagnostics, policy)
        retired = [a[1] for a in actions if a[0] == "retire"]
        added_parents = [a[1] for a in actions if a[0] == "add"]
        extended = [a[1] for a in actions if a[0] == "extend"]
        _apply_registry_actions(registry, actions, epoch)

        active_after = registry.active_state_ids()
        records.append(EpochRecord(
            epoch=epoch, active_ids=sorted(active_after), n_active=len(active_after),
            retired_this_epoch=retired, added_this_epoch=added_parents,
            extended_this_epoch=extended, sample_counts=dict(count_by_id)))

    return records


# ---------------------------------------------------------------------------
# Doubly-adaptive workflow: feedback layout -> production epochs (top-up/retire/add)
# ---------------------------------------------------------------------------

def double_adaptive_campaign(landscape: Landscape, *, layout=None, feedback_cfg=None,
                             policy=None, mode: str = "exact", beta: float = 1.0,
                             n_epochs: int = 5, epoch_raw_budget: int = 900,
                             res: int = 90, seed: int = 0, use_ess: bool = True,
                             default_steps: int = 150, total_budget=None,
                             sampler_kwargs=None) -> dict:
    """Run the full doubly-adaptive workflow against the real production decisions.

    Stage 1 (feedback): if ``layout`` is given (frozen handoff for a fair policy
    comparison) it is used directly; otherwise ``feedback_campaign`` is run with
    ``feedback_cfg`` and its converged layout handed off.

    Stage 2 (production): seed a real WindowStateRegistry from the layout, then run
    epochs that drive the SHIPPED budget allocation (build_adaptive_epoch_schedule),
    decision (propose_actions_from_diagnostics), convergence gate
    (evaluate_adaptive_convergence_gate), and registry application. A fixed raw
    budget per epoch is distributed across states by the strategy's policy; raw
    samples accumulate per state across epochs (top-up), and the ESS cost model
    (gareus.synth.ess) converts raw -> effective so allocation actually matters.

    Returns a dict with the handoff layout, per-epoch records (budget, PMF-vs-budget,
    active count, actions, weak edges, connectivity), and hard-gate / churn flags.
    """
    from gareus.adaptive_production import (AdaptiveDecisionPolicy,
                                            WindowStateRegistry,
                                            build_adaptive_epoch_schedule,
                                            propose_actions_from_diagnostics,
                                            evaluate_adaptive_convergence_gate,
                                            active_graph_connected,
                                            build_geometry_edges,
                                            _apply_registry_actions)
    from gareus.math_helpers import _adaptive_hist_overlap
    from .metrics import pmf_recovery
    from pathlib import Path
    import tempfile

    rng = np.random.default_rng(seed)
    policy = policy or AdaptiveDecisionPolicy()
    skw = dict(sampler_kwargs or {})

    # ---- Stage 1: obtain the handoff layout ------------------------------
    if layout is None:
        fcfg = dict(feedback_cfg or {})
        recs = feedback_campaign(
            landscape, initial_centers1=fcfg.get("initial_centers1", [0.2, 0.5, 0.8]),
            initial_k1=fcfg.get("initial_k1", [50.0, 50.0, 50.0]),
            initial_centers2=fcfg.get("initial_centers2", [-0.5, 0.5]),
            initial_k2=fcfg.get("initial_k2", [50.0, 50.0]),
            mode=mode, n_rounds=fcfg.get("n_rounds", 4),
            samples_per_window=fcfg.get("samples_per_window", 1000),
            res=res, seed=seed, target_overlap=fcfg.get("target_overlap", 0.30),
            aggressiveness=fcfg.get("aggressiveness", "balanced"), sampler_kwargs=skw)
        last = recs[-1]
        layout = {"centers1": list(last.centers1), "k1": list(last.k1),
                  "centers2": list(last.centers2), "k2": list(last.k2)}

    # ---- Stage 2: seed registry from the layout (the handoff) ------------
    registry = WindowStateRegistry()
    for c1, k1 in zip(layout["centers1"], layout["k1"]):
        for c2, k2 in zip(layout["centers2"], layout["k2"]):
            registry.add_state(primary_center=float(c1), primary_k=float(k1),
                               secondary_center=float(c2), secondary_k=float(k2),
                               epoch=0, source="handoff")

    raw_by_id: dict = {}          # accumulated raw sample arrays per state
    created_epoch: dict = {sid: 0 for sid in registry.active_state_ids()}
    win_by_id: dict = {}
    retired_centers: list = []    # (primary, secondary) of retired states (churn detection)
    epoch_records: list = []
    budget_spent = 0
    churn_events = 0

    def window_of(st):
        return Window(st.primary_center, st.primary_k, st.secondary_center, st.secondary_k)

    def ess_samples():
        """ESS-thinned samples + effective counts for all active states."""
        sb, eff = {}, {}
        for st in registry.active_states():
            sid = int(st.state_id)
            raw = raw_by_id.get(sid)
            if raw is None or len(raw) == 0:
                sb[sid] = np.empty((0, 2)); eff[sid] = 0
                continue
            if use_ess:
                tau = tau_int(landscape, win_by_id[sid])
                ne = int(effective_count(len(raw), tau, is_new=(created_epoch.get(sid, 0) > 0)))
                sb[sid] = thin_to_ess(raw, ne)
                eff[sid] = ne
            else:
                sb[sid] = raw; eff[sid] = len(raw)
        return sb, eff

    def union_pmf():
        sb, _ = ess_samples()
        wins = {sid: win_by_id[sid] for sid in sb if sid in win_by_id}
        sb2 = {sid: s for sid, s in sb.items() if len(s) and sid in wins}
        if len(sb2) < 2:
            return float("nan")
        return pmf_recovery(landscape, sb2, wins, axis="cv1", res=res)["rmse_lowf_weighted"]

    def build_diag(sb, eff):
        """Full epoch-diagnostics dict (states + geometry edges with real overlap).

        The decision gates (min_samples_for_add / min_samples_for_retire) key off
        the RAW accumulated MD sample count per state -- matching the real
        collect_epoch_diagnostics, which reports raw counts, not ESS. ESS-thinned
        samples (``sb``) are still used for the edge OVERLAP and for PMF recovery;
        only the gating sample_count is raw. (Feeding ESS here is what made retire
        never fire, since ESS stayed below the 200-sample retirement floor.)
        """
        edges = []
        for a, b, etype, nd in build_geometry_edges(registry):
            sa, sbb = sb.get(a), sb.get(b)
            if sa is None or sbb is None or len(sa) == 0 or len(sbb) == 0:
                continue
            ov, exch = _pair_overlap_exchange(win_by_id[a], win_by_id[b], sa, sbb,
                                              beta=beta, rng=rng, hist_overlap=_adaptive_hist_overlap)
            edges.append({"state_i": a, "state_j": b, "edge_type": etype,
                          "normalized_distance": nd, "overlap": ov, "exchange_acceptance": exch})
        return {"schema_version": "adaptive_production_epoch_diagnostics_v1",
                "states": [{"state_id": sid,
                            "sample_count": int(len(raw_by_id.get(sid, [])) or 0),  # RAW, not ESS
                            "gamd_boost_sd_kcal_mol": 0.0} for sid in sb],
                "edges": edges}

    for epoch in range(1, n_epochs + 1):
        active = registry.active_states()
        if not active:
            break
        for st in active:
            win_by_id[int(st.state_id)] = window_of(st)

        # ---- allocate this epoch's raw budget via the SHIPPED scheduler ----
        # Schedule from the PRIOR-epoch diagnostics (with edges, so weak_edge_bonus
        # and articulation actually inform allocation).
        sb_prev, eff_prev = ess_samples()
        diag_prev = build_diag(sb_prev, eff_prev)
        # Inject synthetic-scale scheduler floors so the per-epoch raw budget is
        # governed by ``epoch_raw_budget`` (not the production-scale 1000-step
        # default min_state_steps). Strategies may override any of these.
        from dataclasses import replace
        overrides = {}
        if int(getattr(policy, "epoch_step_budget", 0) or 0) == 0:
            overrides["epoch_step_budget"] = int(epoch_raw_budget)
        if int(getattr(policy, "min_state_steps", 0) or 0) == 0:
            overrides["min_state_steps"] = max(20, default_steps // 5)
        if int(getattr(policy, "max_state_steps", 0) or 0) == 0:
            overrides["max_state_steps"] = max(default_steps, 4 * default_steps)
        pol_epoch = replace(policy, **overrides) if overrides else policy
        schedule = build_adaptive_epoch_schedule(
            registry, diag_prev, pol_epoch, epoch=epoch, default_steps=default_steps)

        # ---- draw raw samples per the schedule (top-up = accumulate) -------
        budget_exhausted = False
        for row in schedule:
            sid = int(row["state_id"])
            n_draw = int(row.get("requested_steps", default_steps))
            if total_budget is not None:
                remaining = int(total_budget) - budget_spent
                if remaining <= 0:
                    budget_exhausted = True
                    break
                n_draw = min(n_draw, remaining)
            if n_draw <= 0:
                continue
            st = registry.get_state(sid)
            w = win_by_id.get(sid) or window_of(st)
            new = sample_window(landscape, w, n_draw, mode=mode, beta=beta, rng=rng, res=res, **skw)
            raw_by_id[sid] = np.vstack([raw_by_id[sid], new]) if sid in raw_by_id and len(raw_by_id[sid]) else new
            budget_spent += n_draw

        # ---- build diagnostics from ESS-thinned samples --------------------
        sb, eff = ess_samples()
        diagnostics = build_diag(sb, eff)
        edges = diagnostics["edges"]

        # ---- real decision + convergence gate + apply ----------------------
        actions = propose_actions_from_diagnostics(registry, diagnostics, policy)
        gate = evaluate_adaptive_convergence_gate(
            Path(tempfile.mkdtemp(prefix="synth_da_")), epoch, registry, diagnostics, actions, policy)
        retired = [a[1] for a in actions if a[0] == "retire"]
        added_parents = [a[1] for a in actions if a[0] == "add"]
        extended = [a[1] for a in actions if a[0] == "extend"]
        # churn: an add whose midpoint lands near a previously-retired center
        for a in actions:
            if a[0] == "add":
                pc = float(a[2][0])
                if any(abs(pc - rc[0]) < 0.06 for rc in retired_centers):
                    churn_events += 1
        for sid in retired:
            st = registry.get_state(sid)
            if st is not None:
                retired_centers.append((float(st.primary_center),
                                        float(st.secondary_center or 0.0)))
        overlaps = [e["overlap"] for e in edges if e["overlap"] is not None]
        weak = [e for e in edges if e["overlap"] is None or e["overlap"] < policy.target_overlap]
        _apply_registry_actions(registry, actions, epoch)
        for st in registry.active_states():
            sid = int(st.state_id)
            created_epoch.setdefault(sid, epoch)   # new states stamped this epoch
            win_by_id[sid] = window_of(st)

        epoch_records.append({
            "epoch": epoch, "budget_cumulative": int(budget_spent),
            "n_active": len(registry.active_state_ids()),
            "added": added_parents, "retired": retired, "extended": extended,
            "n_weak_edges": len(weak), "connected": bool(active_graph_connected(registry)),
            "mean_overlap": float(np.mean(overlaps)) if overlaps else float("nan"),
            "min_overlap": float(np.min(overlaps)) if overlaps else float("nan"),
            "pmf_rmse_lowf": union_pmf(),
            "gate_status": gate.get("status", "unknown"),
        })
        if budget_exhausted or gate.get("status") in {"converged", "warning_stop_allowed"}:
            break

    final = epoch_records[-1] if epoch_records else {}
    # PMF-vs-budget efficiency (lower area = better); normalize by budget
    bs = [r["budget_cumulative"] for r in epoch_records]
    pm = [r["pmf_rmse_lowf"] for r in epoch_records if math.isfinite(r["pmf_rmse_lowf"])]
    _trap = getattr(np, "trapezoid", np.trapz)
    auc = float(_trap([r["pmf_rmse_lowf"] for r in epoch_records], bs) / (bs[-1] - bs[0])) \
        if len(bs) > 1 and bs[-1] > bs[0] and all(math.isfinite(r["pmf_rmse_lowf"]) for r in epoch_records) else float("nan")
    return {
        "layout": layout,
        "epochs": epoch_records,
        "final_n_active": final.get("n_active", 0),
        "final_pmf_rmse_lowf": final.get("pmf_rmse_lowf", float("nan")),
        "pmf_budget_auc": auc,
        "total_raw_budget": int(budget_spent),
        "all_connected": all(r["connected"] for r in epoch_records),
        "min_overlap_floor": float(np.nanmin([r["min_overlap"] for r in epoch_records])) if epoch_records else float("nan"),
        "retire_churn_events": int(churn_events),
        "converged_epoch": next((r["epoch"] for r in epoch_records
                                 if r["gate_status"] in {"converged", "warning_stop_allowed"}), None),
    }
