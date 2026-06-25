"""Oracle CVT vs Linspace: 16-replica comparison on chaos_2d, 5 epochs.

chaos_2d: 24 sub-basins (8 CV1 × 3 CV2 wells), 18 kBT main CV1 barrier,
10 kBT secondary CV1 barrier (CV2-dependent), 10 kBT CV2 barrier, 5 kBT
CV2 saddle, 10 noise/trap Gaussians — much harder than rugged_2d.

Four strategies, R=16, 5 epochs, 30M total budget (6M/epoch):
  FIXED-LINSPACE  : farthest-point initial, no adaptation (accumulate samples)
  FIXED-ORACLE    : Lloyd CVT initial, no adaptation
  ADAPTIVE-LINSPACE: farthest-point initial + real adaptive add/retire, 5 epochs
  ADAPTIVE-ORACLE : Lloyd CVT initial + real adaptive add/retire, 5 epochs

Output: docs/GAREUS_oracle_16rep_chaos2d_5epoch.docx + PNGs.
"""
from __future__ import annotations

import time, sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo not in sys.path:
    sys.path.insert(0, _repo)

from gareus.synth.landscapes import chaos_2d
from gareus.synth.replica import (
    place_windows_2d, calibrate_k_2d, replica_run_2d_adaptive,
)
from gareus.synth.oracle import oracle_optimal_centers_2d
from gareus.synth.sampler import sample_window_exact, Window
from gareus.synth.ess import tau_int, effective_count, thin_to_ess, BURN_IN
from gareus.synth.metrics import pmf_recovery_2d

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
LAND = chaos_2d()
R = 16
N_EPOCHS = 5
EPOCH_BUDGET = 6_000_000       # 6M per epoch → 30M total
TOTAL_BUDGET = EPOCH_BUDGET * N_EPOCHS
MBAR_CAP = 15_000
SEED = 0
RES = 90
ORACLE_RES = 60
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

STRATEGY_COLORS = {
    "fixed_lin":  "steelblue",
    "fixed_orc":  "firebrick",
    "adapt_lin":  "darkorange",
    "adapt_orc":  "mediumseagreen",
}
STRATEGY_LABELS = {
    "fixed_lin":  "Fixed Linspace",
    "fixed_orc":  "Fixed Oracle CVT",
    "adapt_lin":  "Adaptive Linspace",
    "adapt_orc":  "Adaptive Oracle CVT",
}


# ---------------------------------------------------------------------------
# Fixed-epoch runner (accumulate samples over N_EPOCHS)
# ---------------------------------------------------------------------------
def fixed_epoch_run(land, centers, k, n_epochs, epoch_budget, mbar_cap, seed, res):
    """Run fixed windows for n_epochs, accumulating samples. Return per-epoch RMSE/coverage."""
    rng = np.random.default_rng(seed)
    n_per = max(1, epoch_budget // max(1, len(centers)))
    windows = {i: Window(float(c1), float(k), float(c2), float(k))
               for i, (c1, c2) in enumerate(centers)}
    all_samples = {i: np.empty((0, 2)) for i in range(len(centers))}
    per_epoch = []
    for ep in range(1, n_epochs + 1):
        for i in range(len(centers)):
            raw = sample_window_exact(land, windows[i], n_per, beta=1.0, res=res, rng=rng)
            tau = tau_int(land, windows[i])
            ne = int(effective_count(n_per, tau, burn_in=BURN_IN, is_new=(ep == 1)))
            thinned = thin_to_ess(raw, ne)
            all_samples[i] = np.vstack([all_samples[i], thinned]) if len(all_samples[i]) else thinned
        nonempty = {i: s for i, s in all_samples.items() if len(s) > 0}
        wins_ne = {i: windows[i] for i in nonempty}
        rec = pmf_recovery_2d(land, nonempty, wins_ne, res=res, max_per_window=mbar_cap) \
            if len(nonempty) >= 1 else {"rmse_lowf": float("nan"), "coverage": 0.0}
        per_epoch.append({
            "epoch": ep,
            "rmse": float(rec["rmse_lowf"]),
            "coverage": float(rec["coverage"]),
            "n_windows": len(centers),
            "budget_spent": ep * epoch_budget,
        })
        print(f"    epoch {ep}/{n_epochs}  RMSE={per_epoch[-1]['rmse']:.4f}  "
              f"cov={per_epoch[-1]['coverage']:.3f}", flush=True)
    return per_epoch, all_samples, wins_ne


# ---------------------------------------------------------------------------
# Adaptive-epoch runner (wraps replica_run_2d_adaptive with per-epoch tracking)
# ---------------------------------------------------------------------------
def adaptive_epoch_run(land, centers, n_epochs, epoch_budget, mbar_cap, seed, res):
    """Run adaptive add/retire for n_epochs, recording per-epoch RMSE.

    Runs one epoch at a time by calling replica_run_2d_adaptive with n_epochs=1
    in a loop, passing accumulated samples forward via raw_by_id.
    """
    from dataclasses import replace as dc_replace
    from gareus.adaptive_production import (AdaptiveDecisionPolicy, WindowStateRegistry,
                                            build_adaptive_epoch_schedule,
                                            propose_actions_from_diagnostics,
                                            _apply_registry_actions, build_geometry_edges)
    from gareus.math_helpers import _adaptive_hist_overlap
    from gareus.synth.drivers import _pair_overlap_exchange

    rng = np.random.default_rng(seed)
    policy = AdaptiveDecisionPolicy()
    total_budget = n_epochs * epoch_budget

    k, _, _ = calibrate_k_2d(land, centers, overlap_target=0.30, res=res)
    reg = WindowStateRegistry()
    for (c1, c2) in centers:
        reg.add_state(primary_center=float(c1), primary_k=float(k),
                      secondary_center=float(c2), secondary_k=float(k), epoch=0, source="seed")

    raw_by_id: dict = {}
    created: dict = {}
    win_by_id: dict = {}
    spent = 0
    total_retired = total_added = 0
    per_epoch = []

    def win_of(st):
        return Window(st.primary_center, st.primary_k, st.secondary_center, st.secondary_k)

    def thinned(sid):
        raw = raw_by_id.get(sid)
        if raw is None or len(raw) == 0:
            return np.empty((0, 2)), 0
        tau = tau_int(land, win_by_id[sid])
        ne = int(effective_count(len(raw), tau, burn_in=BURN_IN, is_new=(created.get(sid, 0) > 0)))
        return thin_to_ess(raw, ne), ne

    def build_diag(active):
        for st in active:
            win_by_id[int(st.state_id)] = win_of(st)
        sb = {}
        for st in active:
            sb[int(st.state_id)], _ = thinned(int(st.state_id))
        edges = []
        for a, b, etype, nd in build_geometry_edges(reg):
            sa, sbb = sb.get(a), sb.get(b)
            if sa is None or sbb is None or len(sa) == 0 or len(sbb) == 0:
                continue
            ov, exch = _pair_overlap_exchange(win_by_id[a], win_by_id[b], sa, sbb,
                                              beta=1.0, rng=rng,
                                              hist_overlap=_adaptive_hist_overlap)
            edges.append({"state_i": a, "state_j": b, "edge_type": etype,
                          "normalized_distance": nd, "overlap": ov, "exchange_acceptance": exch})
        states = [{"state_id": int(st.state_id),
                   "sample_count": int(len(raw_by_id.get(int(st.state_id), []))),
                   "gamd_boost_sd_kcal_mol": 0.0} for st in active]
        return {"schema_version": "adaptive_production_epoch_diagnostics_v1",
                "states": states, "edges": edges}, sb

    for epoch in range(1, n_epochs + 1):
        active = reg.active_states()
        if not active or spent >= total_budget:
            break
        for st in active:
            win_by_id[int(st.state_id)] = win_of(st)
        diag_prev, _ = build_diag(active)
        pol_e = dc_replace(
            policy,
            epoch_step_budget=epoch_budget,
            min_state_steps=max(20, epoch_budget // (4 * max(1, len(active)))),
            max_state_steps=int(total_budget),
        )
        sched = build_adaptive_epoch_schedule(
            reg, diag_prev, pol_e, epoch=epoch,
            default_steps=max(50, epoch_budget // max(1, len(active))))
        for row in sched:
            sid = int(row["state_id"])
            n = int(row.get("requested_steps", 0))
            n = min(n, total_budget - spent)
            if n <= 0:
                continue
            w = win_by_id.get(sid) or win_of(reg.get_state(sid))
            new = sample_window_exact(land, w, n, beta=1.0, res=res, rng=rng)
            raw_by_id[sid] = np.vstack([raw_by_id[sid], new]) \
                if sid in raw_by_id and len(raw_by_id[sid]) else new
            spent += n

        active2 = reg.active_states()
        diag2, _ = build_diag(active2)
        acts = propose_actions_from_diagnostics(reg, diag2, policy)
        total_retired += sum(1 for a in acts if a[0] == "retire")
        total_added += sum(1 for a in acts if a[0] == "add")
        _apply_registry_actions(reg, acts, epoch)
        for st in reg.active_states():
            created.setdefault(int(st.state_id), epoch)
            win_by_id[int(st.state_id)] = win_of(st)

        # Per-epoch PMF recovery
        sb_ep, wn_ep = {}, {}
        for st in reg.active_states():
            sid = int(st.state_id)
            win_by_id[sid] = win_of(st)
            s, _ = thinned(sid)
            if len(s):
                sb_ep[sid] = s
                wn_ep[sid] = win_by_id[sid]
        rec = pmf_recovery_2d(land, sb_ep, wn_ep, res=res, max_per_window=mbar_cap) \
            if len(sb_ep) >= 1 else {"rmse_lowf": float("nan"), "coverage": 0.0}
        n_active = len(reg.active_state_ids())
        per_epoch.append({
            "epoch": epoch,
            "rmse": float(rec["rmse_lowf"]),
            "coverage": float(rec["coverage"]),
            "n_windows": n_active,
            "budget_spent": spent,
            "retired": total_retired,
            "added": total_added,
        })
        print(f"    epoch {epoch}/{n_epochs}  RMSE={per_epoch[-1]['rmse']:.4f}  "
              f"cov={per_epoch[-1]['coverage']:.3f}  "
              f"W={n_active}  +{total_added}/-{total_retired}", flush=True)
        if spent >= total_budget:
            break

    return per_epoch, sb_ep, wn_ep


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_landscape(land, out_dir, res=100):
    """Show the chaos_2d FES + basins."""
    c1, c2, f = land.grid(res=res)
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.pcolormesh(c2, c1, np.clip(f, 0, 20), cmap="viridis_r", vmin=0, vmax=20)
    plt.colorbar(im, ax=ax, label="F (kBT)")
    for (b1, b2) in land.basins:
        ax.plot(b2, b1, "wx", ms=7, mew=1.5)
    ax.set_xlabel("CV2"); ax.set_ylabel("CV1")
    ax.set_title("chaos_2d FES — 24 sub-basins, 18 kBT CV1 barrier + traps (clipped at 20 kBT)")
    fig.tight_layout()
    path = os.path.join(out_dir, "chaos2d_fig0_landscape.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    return path


def fig_initial_placements(land, lin_c, orc_c, k_lin, k_orc, out_dir, res=100):
    """Show linspace vs oracle initial window positions on the FES."""
    c1, c2, f = land.grid(res=res)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, centers, k, label, col in [
        (axes[0], lin_c, k_lin, f"Linspace (farthest-pt, k={k_lin:.0f})", "deepskyblue"),
        (axes[1], orc_c, k_orc, f"Oracle CVT (Lloyd, k={k_orc:.0f})", "tomato"),
    ]:
        ax.pcolormesh(c2, c1, np.clip(f, 0, 20), cmap="viridis_r", vmin=0, vmax=20)
        for (bc1, bc2) in centers:
            ax.plot(bc2, bc1, "o", color=col, ms=9, mew=1.5, markeredgecolor="white")
        ax.set_title(f"R={R} — {label}", fontsize=9)
        ax.set_xlabel("CV2"); ax.set_ylabel("CV1")
    fig.suptitle("chaos_2d: Initial window placement for R=16", fontsize=11)
    fig.tight_layout()
    path = os.path.join(out_dir, "chaos2d_fig1_initial_placement.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    return path


def fig_epoch_convergence(epoch_data, out_dir):
    """Per-epoch RMSE and coverage for all 4 strategies."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    for strat, data in epoch_data.items():
        epochs = [d["epoch"] for d in data]
        rmse = [d["rmse"] for d in data]
        cov = [d["coverage"] for d in data]
        kw = dict(color=STRATEGY_COLORS[strat], label=STRATEGY_LABELS[strat], lw=2, ms=7)
        ax1.plot(epochs, rmse, "o-", **kw)
        ax2.plot(epochs, cov, "s-", **kw)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("PMF RMSE (kBT)")
    ax1.set_title("PMF RMSE convergence over 5 epochs")
    ax1.legend(fontsize=8)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Low-F coverage (fraction)")
    ax2.set_title("Coverage convergence over 5 epochs")
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=8)
    fig.suptitle("chaos_2d 5-epoch comparison: Oracle CVT vs Linspace, Fixed vs Adaptive (R=16, 30M samples)",
                 fontsize=10)
    fig.tight_layout()
    path = os.path.join(out_dir, "chaos2d_fig2_epoch_convergence.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    return path


def fig_final_bar(epoch_data, out_dir):
    """Final RMSE and coverage bar chart."""
    strategies = list(epoch_data.keys())
    final_rmse = [epoch_data[s][-1]["rmse"] for s in strategies]
    final_cov = [epoch_data[s][-1]["coverage"] for s in strategies]
    labels = [STRATEGY_LABELS[s] for s in strategies]
    colors = [STRATEGY_COLORS[s] for s in strategies]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(len(strategies))
    bars1 = ax1.bar(x, final_rmse, color=colors, edgecolor="k", linewidth=0.8)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax1.set_ylabel("PMF RMSE (kBT)")
    ax1.set_title("Final RMSE after 5 epochs")
    for b, v in zip(bars1, final_rmse):
        ax1.text(b.get_x() + b.get_width()/2, v + 0.003, f"{v:.4f}", ha="center", fontsize=7)

    bars2 = ax2.bar(x, final_cov, color=colors, edgecolor="k", linewidth=0.8)
    ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax2.set_ylabel("Low-F coverage")
    ax2.set_title("Final coverage after 5 epochs")
    ax2.set_ylim(0, 1.1)
    for b, v in zip(bars2, final_cov):
        ax2.text(b.get_x() + b.get_width()/2, v + 0.01, f"{v:.3f}", ha="center", fontsize=7)

    fig.suptitle("chaos_2d: Final quality after 5 epochs (R=16, 30M samples)", fontsize=10)
    fig.tight_layout()
    path = os.path.join(out_dir, "chaos2d_fig3_final_bar.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    return path


def fig_fes_recovery(land, sample_data, out_dir, res=90):
    """2D FES recovery for all 4 strategies."""
    from gareus.synth.metrics import _mbar_weights
    c1ax = np.linspace(*land.cv1_bounds, res)
    c2ax = np.linspace(*land.cv2_bounds, res)
    g1, g2 = np.meshgrid(c1ax, c2ax, indexing="ij")
    f_true = land.energy(g1, g2); f_true -= f_true.min()

    strategies = list(sample_data.keys())
    fig, axes = plt.subplots(3, len(strategies), figsize=(4 * len(strategies), 11))

    def _recovered(sb, wn):
        pooled, log_w = _mbar_weights(wn, sb, beta=1.0)
        b1 = np.linspace(*land.cv1_bounds, res + 1)
        b2 = np.linspace(*land.cv2_bounds, res + 1)
        w = np.exp(log_w)
        dens, _, _ = np.histogram2d(pooled[:, 0], pooled[:, 1], bins=[b1, b2], weights=w)
        f_est = np.full_like(dens, float("nan"))
        pos = dens > 0
        f_est[pos] = -np.log(dens[pos])
        if np.any(pos):
            f_est[pos] -= np.nanmin(f_est[pos])
        return f_est

    vmax = 15.0
    for col, strat in enumerate(strategies):
        sb, wn, rmse = sample_data[strat]
        f_rec = _recovered(sb, wn)
        err = np.abs(f_rec - f_true)

        # Row 0: true FES + window positions
        axes[0, col].pcolormesh(c2ax, c1ax, np.clip(f_true, 0, vmax), cmap="viridis_r", vmin=0, vmax=vmax)
        for w in wn.values():
            axes[0, col].plot(w.center2, w.center1, "wx", ms=5, mew=1)
        axes[0, col].set_title(STRATEGY_LABELS[strat], fontsize=8)
        axes[0, col].set_ylabel("CV1") if col == 0 else None

        # Row 1: recovered FES
        axes[1, col].pcolormesh(c2ax, c1ax, np.nan_to_num(f_rec, nan=vmax),
                                cmap="viridis_r", vmin=0, vmax=vmax)
        axes[1, col].set_title(f"MBAR-recovered RMSE={rmse:.4f} kBT", fontsize=8)
        axes[1, col].set_ylabel("CV1") if col == 0 else None

        # Row 2: error
        axes[2, col].pcolormesh(c2ax, c1ax, np.nan_to_num(err, nan=0), cmap="Reds", vmin=0, vmax=5)
        axes[2, col].set_title("|ΔF| error", fontsize=8)
        axes[2, col].set_xlabel("CV2")
        axes[2, col].set_ylabel("CV1") if col == 0 else None

    fig.suptitle("chaos_2d 2D FES Recovery: 4 strategies after 5 epochs (R=16, 30M)", fontsize=10)
    fig.tight_layout()
    path = os.path.join(out_dir, "chaos2d_fig4_fes_recovery.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    return path


def fig_window_count(epoch_data, out_dir):
    """Window count over epochs (for adaptive strategies)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for strat, data in epoch_data.items():
        epochs = [d["epoch"] for d in data]
        n_win = [d["n_windows"] for d in data]
        ax.plot(epochs, n_win, "o-", color=STRATEGY_COLORS[strat],
                label=STRATEGY_LABELS[strat], lw=2, ms=7)
    ax.axhline(R, color="k", ls="--", lw=0.8, label=f"Seeded R={R}")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Active windows")
    ax.set_title("Active window count over 5 epochs")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(out_dir, "chaos2d_fig5_window_count.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Docx
# ---------------------------------------------------------------------------
def write_docx(epoch_data, sample_data, lin_c, orc_c, k_lin, k_orc,
               elapsed, fig_paths, out_path):
    import docx
    from docx.shared import Inches

    doc = docx.Document()
    doc.add_heading("Oracle CVT vs Linspace: 16-Replica chaos_2d Comparison, 5 Epochs", 0)
    doc.add_paragraph(
        f"chaos_2d (24 sub-basins, 18 kBT CV1 + 10 kBT CV2 barriers, 10 noise traps) · "
        f"R={R} · 5 epochs · {TOTAL_BUDGET/1e6:.0f}M total samples ({EPOCH_BUDGET/1e6:.0f}M/epoch) · "
        f"MBAR cap {MBAR_CAP//1000}k/window · 1 seed · {elapsed:.0f}s"
    )

    doc.add_heading("chaos_2d Landscape", 1)
    doc.add_paragraph(
        "8 CV1 basins (varied depth/width) × 3 CV2 wells = 24 sub-basins. "
        "Main CV1 barrier: 18 kBT (vs ~14 kBT in rugged_2d). "
        "Secondary CV1 barrier: 10 kBT, CV2-dependent (tallest at CV2~0, half-height at CV2 extremes — "
        "requires diagonal bridging, not just axis-aligned ladders). "
        "CV2 barrier: 10 kBT at 0 + 5 kBT saddle at +0.33 (4 effective CV2 zones). "
        "10 noise Gaussians (3–5 kBT) as reproducible metastable traps scattered across the surface."
    )
    if fig_paths.get("landscape"):
        doc.add_picture(fig_paths["landscape"], width=Inches(5.5))

    doc.add_heading("Initial Placement: R=16", 1)
    doc.add_paragraph(
        f"Linspace (farthest-point greedy): k={k_lin:.0f}. "
        f"Oracle CVT (Lloyd 30 iterations): k={k_orc:.0f}. "
        f"Both k values calibrated to their own centers via calibrate_k_2d (target overlap 0.30)."
    )
    if fig_paths.get("init"):
        doc.add_picture(fig_paths["init"], width=Inches(6.0))

    doc.add_heading("Epoch Convergence", 1)
    if fig_paths.get("convergence"):
        doc.add_picture(fig_paths["convergence"], width=Inches(6.5))
    if fig_paths.get("windows"):
        doc.add_picture(fig_paths["windows"], width=Inches(5.0))

    doc.add_heading("Final Results (after 5 epochs)", 1)
    if fig_paths.get("bar"):
        doc.add_picture(fig_paths["bar"], width=Inches(6.0))

    # Summary table
    tbl = doc.add_table(rows=1, cols=5)
    tbl.style = "Table Grid"
    hdr = tbl.rows[0].cells
    for i, h in enumerate(["Strategy", "Final RMSE (kBT)", "Final coverage",
                            "Final windows", "Budget spent"]):
        hdr[i].text = h
    for strat in ["fixed_lin", "fixed_orc", "adapt_lin", "adapt_orc"]:
        d = epoch_data[strat][-1]
        row = tbl.add_row().cells
        row[0].text = STRATEGY_LABELS[strat]
        row[1].text = f"{d['rmse']:.4f}"
        row[2].text = f"{d['coverage']:.3f}"
        row[3].text = str(d["n_windows"])
        row[4].text = f"{d['budget_spent']/1e6:.1f}M"

    doc.add_heading("2D FES Recovery", 1)
    if fig_paths.get("fes"):
        doc.add_picture(fig_paths["fes"], width=Inches(6.5))

    doc.add_heading("Answer", 1)
    # Compute key numbers
    rmses = {s: epoch_data[s][-1]["rmse"] for s in epoch_data}
    best = min(rmses, key=rmses.get)
    worst = max(rmses, key=rmses.get)
    oracle_gain_fixed = (rmses["fixed_lin"] - rmses["fixed_orc"]) / rmses["fixed_lin"] * 100
    oracle_gain_adapt = (rmses["adapt_lin"] - rmses["adapt_orc"]) / rmses["adapt_lin"] * 100
    adapt_gain_lin = (rmses["fixed_lin"] - rmses["adapt_lin"]) / rmses["fixed_lin"] * 100
    adapt_gain_orc = (rmses["fixed_orc"] - rmses["adapt_orc"]) / rmses["fixed_orc"] * 100

    doc.add_paragraph(
        f"Best strategy: {STRATEGY_LABELS[best]} (RMSE = {rmses[best]:.4f} kBT).\n"
        f"Worst strategy: {STRATEGY_LABELS[worst]} (RMSE = {rmses[worst]:.4f} kBT).\n\n"
        f"Oracle placement gain (fixed): {oracle_gain_fixed:+.1f}% vs fixed linspace "
        f"({rmses['fixed_orc']:.4f} vs {rmses['fixed_lin']:.4f} kBT).\n"
        f"Oracle placement gain (adaptive): {oracle_gain_adapt:+.1f}% vs adaptive linspace "
        f"({rmses['adapt_orc']:.4f} vs {rmses['adapt_lin']:.4f} kBT).\n"
        f"Adaptive gain for linspace initial: {adapt_gain_lin:+.1f}%.\n"
        f"Adaptive gain for oracle initial: {adapt_gain_orc:+.1f}%.\n\n"
        f"Key question — does oracle seeding help adaptive runs? "
        f"Oracle CVT covers sub-basins more uniformly at epoch 1; the adaptive loop "
        f"then corrects misplacements. If oracle seeding gives fewer corrections needed, "
        f"the budget is better spent sampling than repositioning."
    )

    doc.add_heading("Caveats", 1)
    doc.add_paragraph(
        f"1 seed (compute cost); exact sampler (no kinetic trapping). "
        f"MBAR cap {MBAR_CAP//1000}k/window; floor partly cap-limited. "
        f"chaos_2d has 24 sub-basins — with R=16 windows many cells share coverage, "
        f"so PMF RMSE reflects both coverage gaps AND over-smoothing at barriers. "
        f"The secondary CV1 barrier is CV2-dependent; purely CV1-axis placement misses "
        f"the diagonal bridging required to connect basins across both axes."
    )

    doc.save(out_path)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    t0 = time.time()
    print(f"chaos_2d 5-epoch comparison  R={R}  budget={TOTAL_BUDGET/1e6:.0f}M  epochs={N_EPOCHS}", flush=True)

    # Pre-compute centers
    print("\nPre-computing initial placements ...", flush=True)
    lin_c = place_windows_2d(LAND, R)
    orc_c = oracle_optimal_centers_2d(LAND, R, res=ORACLE_RES)
    k_lin, _, _ = calibrate_k_2d(LAND, lin_c, overlap_target=0.30, res=RES)
    k_orc, _, _ = calibrate_k_2d(LAND, orc_c, overlap_target=0.30, res=RES)
    print(f"  linspace k={k_lin:.1f}  oracle CVT k={k_orc:.1f}", flush=True)

    epoch_data = {}
    sample_data = {}     # strat -> (samples_by_win, windows_by_win, final_rmse)

    # FIXED-LINSPACE
    print("\n[1/4] Fixed Linspace ...", flush=True)
    ep, sb, wn = fixed_epoch_run(LAND, lin_c, k_lin, N_EPOCHS, EPOCH_BUDGET, MBAR_CAP, SEED, RES)
    epoch_data["fixed_lin"] = ep
    sample_data["fixed_lin"] = (sb, wn, ep[-1]["rmse"])

    # FIXED-ORACLE
    print("\n[2/4] Fixed Oracle CVT ...", flush=True)
    ep, sb, wn = fixed_epoch_run(LAND, orc_c, k_orc, N_EPOCHS, EPOCH_BUDGET, MBAR_CAP, SEED, RES)
    epoch_data["fixed_orc"] = ep
    sample_data["fixed_orc"] = (sb, wn, ep[-1]["rmse"])

    # ADAPTIVE-LINSPACE
    print("\n[3/4] Adaptive Linspace ...", flush=True)
    ep, sb, wn = adaptive_epoch_run(LAND, lin_c, N_EPOCHS, EPOCH_BUDGET, MBAR_CAP, SEED, RES)
    epoch_data["adapt_lin"] = ep
    sample_data["adapt_lin"] = (sb, wn, ep[-1]["rmse"])

    # ADAPTIVE-ORACLE
    print("\n[4/4] Adaptive Oracle CVT ...", flush=True)
    ep, sb, wn = adaptive_epoch_run(LAND, orc_c, N_EPOCHS, EPOCH_BUDGET, MBAR_CAP, SEED, RES)
    epoch_data["adapt_orc"] = ep
    sample_data["adapt_orc"] = (sb, wn, ep[-1]["rmse"])

    elapsed = time.time() - t0
    print(f"\nSweep done in {elapsed:.1f}s", flush=True)

    print("\nGenerating figures ...", flush=True)
    figs = {}
    figs["landscape"] = fig_landscape(LAND, OUT_DIR)
    figs["init"] = fig_initial_placements(LAND, lin_c, orc_c, k_lin, k_orc, OUT_DIR)
    figs["convergence"] = fig_epoch_convergence(epoch_data, OUT_DIR)
    figs["bar"] = fig_final_bar(epoch_data, OUT_DIR)
    figs["windows"] = fig_window_count(epoch_data, OUT_DIR)
    figs["fes"] = fig_fes_recovery(LAND, sample_data, OUT_DIR)

    out_docx = os.path.join(OUT_DIR, "GAREUS_oracle_16rep_chaos2d_5epoch.docx")
    write_docx(epoch_data, sample_data, lin_c, orc_c, k_lin, k_orc,
               elapsed, figs, out_docx)
    print("Done.", flush=True)
