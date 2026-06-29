"""Oracle-placement 2D comparison at 50× budget.

Same setup as GAREUS_retire_comparison_50x.docx:
  - rugged-2d landscape (6 CV1 basins × 2 CV2 wells, 14 kBT CV1 barrier + 7 kBT CV2 barrier)
  - R=1..12, budgets 0.9M/1M/3M/7M/15M, MBAR cap 15000/window, 1 seed

Three placement strategies compared at identical (R, budget):
  LINSPACE  - place_windows_2d (farthest-point, current default)
  ORACLE    - oracle_optimal_centers_2d (Lloyd CVT), each calibrated at own k
  ORACLE-FK - Lloyd CVT positions at linspace fair-k (position-only gain)

Output: docs/GAREUS_oracle_placement_2d_50x.docx + accompanying PNGs.
"""
from __future__ import annotations

import time
import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from gareus.synth.landscapes import rugged_2d
from gareus.synth.replica import (
    place_windows_2d, calibrate_k_2d, replica_run_2d,
    low_f_support,
)
from gareus.synth.oracle import oracle_optimal_centers_2d
from gareus.synth.sampler import sample_window_exact, Window
from gareus.synth.ess import tau_int, effective_count, thin_to_ess, BURN_IN
from gareus.synth.metrics import pmf_recovery_2d

# ---------------------------------------------------------------------------
# Config — matches retire comparison 50x exactly
# ---------------------------------------------------------------------------
LAND = rugged_2d()
R_VALUES = list(range(1, 13))                          # 1..12
BUDGETS = [900_000, 1_000_000, 3_000_000, 7_000_000, 15_000_000]
MBAR_CAP = 15_000
SEED = 0
RES = 90
ORACLE_RES = 60                                        # faster oracle CVT, sufficient quality

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Pre-compute oracle + linspace centers for each R (budget-independent)
# ---------------------------------------------------------------------------
def precompute_centers(land, R_values, oracle_res=ORACLE_RES):
    print(f"Pre-computing centers for R={R_values[0]}..{R_values[-1]} ...", flush=True)
    lin_centers = {}
    oracle_centers = {}
    for R in R_values:
        lin_centers[R] = place_windows_2d(land, R)
        oracle_centers[R] = oracle_optimal_centers_2d(land, R, res=oracle_res)
    print("  done.", flush=True)
    return lin_centers, oracle_centers


# ---------------------------------------------------------------------------
# Run a single (R, budget, centers, k) configuration
# ---------------------------------------------------------------------------
def run_one(land, R, budget, centers, k, *, seed=SEED, res=RES, mbar_cap=MBAR_CAP):
    rng = np.random.default_rng(seed)
    n_per = max(1, budget // max(1, R))
    samples, windows, ess = {}, {}, {}
    for i, (c1, c2) in enumerate(centers):
        w = Window(float(c1), float(k), float(c2), float(k))
        windows[i] = w
        raw = sample_window_exact(land, w, n_per, beta=1.0, res=res, rng=rng)
        tau = tau_int(land, w)
        ne = int(effective_count(n_per, tau, burn_in=BURN_IN, is_new=True))
        ess[i] = ne
        samples[i] = thin_to_ess(raw, ne)
    nonempty = {i: s for i, s in samples.items() if len(s) > 0}
    wins_ne = {i: windows[i] for i in nonempty}
    if len(nonempty) < 1:
        return {"pmf_rmse": float("nan"), "coverage": 0.0, "k": float(k),
                "total_ess": 0, "n_raw_per": n_per}
    rec = pmf_recovery_2d(land, nonempty, wins_ne, res=res, max_per_window=mbar_cap)
    total_ess = int(sum(ess.values()))
    return {"pmf_rmse": float(rec["rmse_lowf"]), "coverage": float(rec["coverage"]),
            "k": float(k), "total_ess": total_ess, "n_raw_per": n_per}


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------
def run_sweep(land, lin_centers, oracle_centers, R_values, budgets):
    results = {}    # (strategy, R, budget) -> dict
    n_total = len(R_values) * len(budgets) * 3
    done = 0
    t0 = time.time()
    for R in R_values:
        lc = lin_centers[R]
        oc = oracle_centers[R]
        # calibrate k for each center set
        k_lin, _, _ = calibrate_k_2d(land, lc, overlap_target=0.30, res=RES)
        k_oracle, _, _ = calibrate_k_2d(land, oc, overlap_target=0.30, res=RES)
        for budget in budgets:
            # LINSPACE: farthest-point centers + linspace k
            results[("linspace", R, budget)] = run_one(land, R, budget, lc, k_lin)
            done += 1
            # ORACLE: Lloyd CVT centers + oracle k
            results[("oracle", R, budget)] = run_one(land, R, budget, oc, k_oracle)
            done += 1
            # ORACLE-FK: Lloyd CVT centers at linspace k (position-only)
            results[("oracle_fk", R, budget)] = run_one(land, R, budget, oc, k_lin)
            done += 1
            elapsed = time.time() - t0
            rate = done / elapsed
            remaining = (n_total - done) / rate
            print(f"  R={R:2d} B={budget//1e6:.1f}M  "
                  f"lin={results[('linspace',R,budget)]['pmf_rmse']:.4f}  "
                  f"oracle={results[('oracle',R,budget)]['pmf_rmse']:.4f}  "
                  f"[{done}/{n_total} {remaining:.0f}s left]", flush=True)
    return results


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_pmf_vs_r(results, R_values, budgets, out_dir):
    """PMF RMSE vs R for each budget, 3 strategies."""
    fig, axes = plt.subplots(1, len(budgets), figsize=(4 * len(budgets), 4), sharey=True)
    colors = {"linspace": "steelblue", "oracle": "firebrick", "oracle_fk": "darkorange"}
    labels = {"linspace": "Linspace (farthest-pt)", "oracle": "Oracle CVT (own k)",
              "oracle_fk": "Oracle CVT (linspace k)"}
    for ax, budget in zip(axes, budgets):
        for strat in ("linspace", "oracle", "oracle_fk"):
            rmse = [results.get((strat, R, budget), {}).get("pmf_rmse", float("nan"))
                    for R in R_values]
            ax.plot(R_values, rmse, "o-", color=colors[strat], label=labels[strat], lw=1.5, ms=5)
        ax.set_title(f"B={budget/1e6:.1f}M", fontsize=9)
        ax.set_xlabel("R (windows)")
        ax.set_xticks(R_values[::2])
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("PMF RMSE (kBT)")
    axes[0].legend(fontsize=7, loc="upper right")
    fig.suptitle("Oracle vs Linspace 2D Placement — PMF RMSE vs R", fontsize=11)
    fig.tight_layout()
    path = os.path.join(out_dir, "oracle2d_fig1_pmf_vs_r.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_delta_pmf_vs_r(results, R_values, budgets, out_dir):
    """ΔPMF (oracle - linspace) vs R per budget."""
    fig, axes = plt.subplots(1, len(budgets), figsize=(4 * len(budgets), 4), sharey=True)
    for ax, budget in zip(axes, budgets):
        delta_own_k = []
        delta_fk = []
        for R in R_values:
            lin = results.get(("linspace", R, budget), {}).get("pmf_rmse", float("nan"))
            oc = results.get(("oracle", R, budget), {}).get("pmf_rmse", float("nan"))
            ok = results.get(("oracle_fk", R, budget), {}).get("pmf_rmse", float("nan"))
            delta_own_k.append(oc - lin)
            delta_fk.append(ok - lin)
        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.plot(R_values, delta_own_k, "s-", color="firebrick", label="Oracle CVT (own k)", lw=1.5, ms=5)
        ax.plot(R_values, delta_fk, "^-", color="darkorange", label="Oracle CVT (lin k)", lw=1.5, ms=5)
        ax.set_title(f"B={budget/1e6:.1f}M", fontsize=9)
        ax.set_xlabel("R (windows)")
        ax.set_xticks(R_values[::2])
    axes[0].set_ylabel("ΔPMF (oracle − linspace) kBT")
    axes[0].legend(fontsize=7)
    fig.suptitle("Oracle vs Linspace 2D — Positional Gain (negative = oracle better)", fontsize=11)
    fig.tight_layout()
    path = os.path.join(out_dir, "oracle2d_fig2_delta_pmf.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_pmf_vs_budget(results, R_values, budgets, out_dir, fixed_Rs=(8, 12)):
    """PMF RMSE vs budget at fixed R values."""
    fig, axes = plt.subplots(1, len(fixed_Rs), figsize=(5 * len(fixed_Rs), 4))
    colors = {"linspace": "steelblue", "oracle": "firebrick", "oracle_fk": "darkorange"}
    labels = {"linspace": "Linspace", "oracle": "Oracle CVT (own k)", "oracle_fk": "Oracle CVT (lin k)"}
    bvals = np.array(budgets) / 1e6
    for ax, R in zip(axes, fixed_Rs):
        for strat in ("linspace", "oracle", "oracle_fk"):
            rmse = [results.get((strat, R, b), {}).get("pmf_rmse", float("nan"))
                    for b in budgets]
            ax.semilogx(budgets, rmse, "o-", color=colors[strat], label=labels[strat], lw=1.5, ms=5)
        ax.set_title(f"Fixed R={R}", fontsize=10)
        ax.set_xlabel("Total budget (samples)")
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("PMF RMSE (kBT)")
    fig.suptitle("Oracle vs Linspace 2D — PMF Recovery vs Budget", fontsize=11)
    fig.tight_layout()
    path = os.path.join(out_dir, "oracle2d_fig3_pmf_vs_budget.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_coverage_vs_r(results, R_values, budgets, out_dir):
    """Coverage vs R at highest budget."""
    budget = budgets[-1]
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = {"linspace": "steelblue", "oracle": "firebrick", "oracle_fk": "darkorange"}
    labels = {"linspace": "Linspace", "oracle": "Oracle CVT (own k)", "oracle_fk": "Oracle CVT (lin k)"}
    for strat in ("linspace", "oracle", "oracle_fk"):
        cov = [results.get((strat, R, budget), {}).get("coverage", float("nan")) for R in R_values]
        ax.plot(R_values, cov, "o-", color=colors[strat], label=labels[strat], lw=1.5, ms=5)
    ax.set_xlabel("R (windows)")
    ax.set_ylabel("Low-F coverage (fraction)")
    ax.set_title(f"Coverage vs R at B={budget/1e6:.0f}M (highest budget)")
    ax.set_xticks(R_values[::2])
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    path = os.path.join(out_dir, "oracle2d_fig4_coverage.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_fes_recovery(land, results, lin_centers, oracle_centers, out_dir,
                     budget=15_000_000, R=12):
    """True + explored density + MBAR recovery side-by-side for best config."""
    lc = lin_centers[R]
    oc = oracle_centers[R]
    k_lin, _, _ = calibrate_k_2d(land, lc, overlap_target=0.30, res=RES)
    k_oracle, _, _ = calibrate_k_2d(land, oc, overlap_target=0.30, res=RES)
    res = RES
    rng = np.random.default_rng(SEED)

    def _collect(centers, k):
        n_per = max(1, budget // max(1, len(centers)))
        samples_d, windows_d = {}, {}
        for i, (c1, c2) in enumerate(centers):
            w = Window(float(c1), float(k), float(c2), float(k))
            windows_d[i] = w
            raw = sample_window_exact(land, w, n_per, beta=1.0, res=res, rng=np.random.default_rng(SEED + i))
            tau = tau_int(land, w)
            ne = int(effective_count(n_per, tau, burn_in=BURN_IN, is_new=True))
            samples_d[i] = thin_to_ess(raw, ne)
        ne_d = {i: s for i, s in samples_d.items() if len(s) > 0}
        wn_d = {i: windows_d[i] for i in ne_d}
        return ne_d, wn_d

    lin_s, lin_w = _collect(lc, k_lin)
    orc_s, orc_w = _collect(oc, k_oracle)
    rec_lin = pmf_recovery_2d(land, lin_s, lin_w, res=res, max_per_window=MBAR_CAP)
    rec_orc = pmf_recovery_2d(land, orc_s, orc_w, res=res, max_per_window=MBAR_CAP)

    c1ax = np.linspace(*land.cv1_bounds, res)
    c2ax = np.linspace(*land.cv2_bounds, res)
    g1, g2 = np.meshgrid(c1ax, c2ax, indexing="ij")
    f_true = land.energy(g1, g2)
    f_true -= f_true.min()

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    vmax = 12.0

    def _plot_pmf(ax, f, title, windows=None, cmap="viridis_r"):
        im = ax.pcolormesh(c2ax, c1ax, np.clip(f, 0, vmax), cmap=cmap, vmin=0, vmax=vmax)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("CV2")
        ax.set_ylabel("CV1")
        if windows:
            for w in windows.values():
                ax.plot(w.center2, w.center1, "wx", ms=6, mew=1.2)
        return im

    # True FES
    _plot_pmf(axes[0, 0], f_true, f"True FES (R={R} linspace)", lin_w)
    _plot_pmf(axes[1, 0], f_true, f"True FES (R={R} oracle CVT)", orc_w)

    # Recovered FES (compute from MBAR weights — reuse pmf_recovery_2d internals)
    from gareus.synth.metrics import _mbar_weights
    from scipy.ndimage import gaussian_filter

    def _recovered_fes(samples_d, windows_d):
        pooled, log_w = _mbar_weights(windows_d, samples_d, beta=1.0)
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

    f_lin_rec = _recovered_fes(lin_s, lin_w)
    f_orc_rec = _recovered_fes(orc_s, orc_w)

    _plot_pmf(axes[0, 1], np.nan_to_num(f_lin_rec, nan=vmax), "MBAR-recovered (linspace)")
    _plot_pmf(axes[1, 1], np.nan_to_num(f_orc_rec, nan=vmax), "MBAR-recovered (oracle CVT)")

    # Error maps
    err_lin = np.abs(f_lin_rec - f_true)
    err_orc = np.abs(f_orc_rec - f_true)
    axes[0, 2].pcolormesh(c2ax, c1ax, np.nan_to_num(err_lin, nan=0), cmap="Reds", vmin=0, vmax=4)
    axes[0, 2].set_title(f"Error |ΔF| linspace  RMSE={rec_lin['rmse_lowf']:.3f} kBT", fontsize=9)
    axes[0, 2].set_xlabel("CV2"); axes[0, 2].set_ylabel("CV1")
    axes[1, 2].pcolormesh(c2ax, c1ax, np.nan_to_num(err_orc, nan=0), cmap="Reds", vmin=0, vmax=4)
    axes[1, 2].set_title(f"Error |ΔF| oracle CVT  RMSE={rec_orc['rmse_lowf']:.3f} kBT", fontsize=9)
    axes[1, 2].set_xlabel("CV2"); axes[1, 2].set_ylabel("CV1")

    fig.suptitle(f"2D FES Recovery: Linspace vs Oracle CVT  (B={budget/1e6:.0f}M, R={R})", fontsize=11)
    fig.tight_layout()
    path = os.path.join(out_dir, "oracle2d_fig5_fes_recovery.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  FES recovery: linspace RMSE={rec_lin['rmse_lowf']:.4f}  oracle RMSE={rec_orc['rmse_lowf']:.4f}")
    return path, rec_lin["rmse_lowf"], rec_orc["rmse_lowf"]


# ---------------------------------------------------------------------------
# Summary tables (text)
# ---------------------------------------------------------------------------
def _best_r_per_budget(results, R_values, budgets, strat):
    rows = []
    for budget in budgets:
        rmses = {R: results.get((strat, R, budget), {}).get("pmf_rmse", float("inf"))
                 for R in R_values}
        finite = {R: v for R, v in rmses.items() if np.isfinite(v)}
        if finite:
            best_R = min(finite, key=finite.get)
            rows.append((budget, best_R, finite[best_R]))
        else:
            rows.append((budget, None, float("nan")))
    return rows


# ---------------------------------------------------------------------------
# Docx generation
# ---------------------------------------------------------------------------
def write_docx(results, R_values, budgets, fig_paths, lin_centers, oracle_centers,
               lin_rmse_top, oracle_rmse_top, elapsed, out_path):
    import docx
    from docx.shared import Inches, Pt
    from docx.enum.text import WD_ALIGN_PARAGRAPH

    doc = docx.Document()
    doc.add_heading("Oracle CVT vs Linspace 2D Placement at 50× Budget", 0)
    sub = doc.add_paragraph()
    sub.add_run(
        f"Oracle (Lloyd CVT) vs farthest-point linspace on rugged-2d, "
        f"{min(budgets)/1e6:.1f}M–{max(budgets)/1e6:.0f}M samples, MBAR cap {MBAR_CAP//1000}k/window"
    )
    sub.add_run(
        f"\nR=1..{max(R_values)} · budgets {len(budgets)} levels · 1 seed · {elapsed:.1f}s"
    )

    doc.add_heading("Setup", 1)
    doc.add_paragraph(
        f"Comparison of two 2D window placement strategies at fixed (R, budget), "
        f"same rugged-2d landscape as the retire comparison (6 CV1 basins, ~14 kBT CV1 barrier, "
        f"CV2 double-well + ~7 kBT CV2 barrier; 12 low-F sub-basins). "
        f"Same sample budgets (0.9M–15M), same MBAR cap ({MBAR_CAP//1000}k/window), 1 seed.\n\n"
        "Three strategies:\n"
        "  LINSPACE — place_windows_2d (farthest-point greedy, current default); "
        "k calibrated to own centers via calibrate_k_2d.\n"
        "  ORACLE — oracle_optimal_centers_2d (Lloyd CVT: farthest-point init + iterative "
        "Voronoi centroid refinement onto nearest low-F grid point); k calibrated to own centers.\n"
        "  ORACLE-FK — Lloyd CVT positions at linspace fair-k (position-only gain, "
        "same ESS cost as linspace)."
    )

    doc.add_heading("Results", 1)
    for path in fig_paths[:-1]:
        caption = os.path.basename(path).replace("oracle2d_", "").replace(".png", "").replace("_", " ")
        doc.add_paragraph(f"Figure: {caption}.", style="Caption")
        doc.add_picture(path, width=Inches(6.5))

    doc.add_heading("Recovered PMF", 1)
    doc.add_paragraph(
        f"2D FES recovery at the top budget (15M samples), R=12: "
        f"linspace RMSE = {lin_rmse_top:.4f} kBT, oracle CVT RMSE = {oracle_rmse_top:.4f} kBT. "
        f"Gain = {(lin_rmse_top - oracle_rmse_top)/lin_rmse_top*100:+.1f}%."
    )
    if fig_paths[-1]:
        doc.add_picture(fig_paths[-1], width=Inches(6.5))

    # Summary table: best R per budget per strategy
    doc.add_heading("Best R per Budget", 1)
    tbl = doc.add_table(rows=1, cols=4)
    tbl.style = "Table Grid"
    hdr = tbl.rows[0].cells
    for i, h in enumerate(["Budget", "Best R (linspace)", "Best R (oracle)", "RMSE Δ at best"]):
        hdr[i].text = h
    for budget in budgets:
        lin_rows = {R: results.get(("linspace", R, budget), {}).get("pmf_rmse", float("inf"))
                    for R in R_values}
        orc_rows = {R: results.get(("oracle", R, budget), {}).get("pmf_rmse", float("inf"))
                    for R in R_values}
        fin_lin = {R: v for R, v in lin_rows.items() if np.isfinite(v)}
        fin_orc = {R: v for R, v in orc_rows.items() if np.isfinite(v)}
        best_R_lin = min(fin_lin, key=fin_lin.get) if fin_lin else "-"
        best_R_orc = min(fin_orc, key=fin_orc.get) if fin_orc else "-"
        lin_best_rmse = fin_lin.get(best_R_lin, float("nan")) if isinstance(best_R_lin, int) else float("nan")
        orc_best_rmse = fin_orc.get(best_R_orc, float("nan")) if isinstance(best_R_orc, int) else float("nan")
        delta = orc_best_rmse - lin_best_rmse
        row = tbl.add_row().cells
        row[0].text = f"{budget/1e6:.1f}M"
        row[1].text = f"R={best_R_lin} ({lin_best_rmse:.4f} kBT)"
        row[2].text = f"R={best_R_orc} ({orc_best_rmse:.4f} kBT)"
        row[3].text = f"{delta:+.4f} kBT"

    doc.add_heading("Answer", 1)

    # Compute key numbers for the answer section
    # Average delta PMF (oracle_fk - linspace) across all R at top budget
    top_b = budgets[-1]
    deltas_fk = []
    deltas_own = []
    for R in R_values:
        lin = results.get(("linspace", R, top_b), {}).get("pmf_rmse", float("nan"))
        oc = results.get(("oracle", R, top_b), {}).get("pmf_rmse", float("nan"))
        fk = results.get(("oracle_fk", R, top_b), {}).get("pmf_rmse", float("nan"))
        if np.isfinite(lin) and np.isfinite(oc):
            deltas_own.append(oc - lin)
        if np.isfinite(lin) and np.isfinite(fk):
            deltas_fk.append(fk - lin)
    mean_own = float(np.mean(deltas_own)) if deltas_own else float("nan")
    mean_fk = float(np.mean(deltas_fk)) if deltas_fk else float("nan")

    # Best budget where oracle significantly wins
    oracle_wins_pct = {}
    for budget in budgets:
        wins = 0
        for R in R_values:
            lin = results.get(("linspace", R, budget), {}).get("pmf_rmse", float("nan"))
            oc = results.get(("oracle", R, budget), {}).get("pmf_rmse", float("nan"))
            if np.isfinite(lin) and np.isfinite(oc) and oc < lin * 0.95:
                wins += 1
        oracle_wins_pct[budget] = wins / len(R_values) * 100

    doc.add_paragraph(
        f"Position-only gain (oracle CVT positions at linspace k): mean ΔPMF = {mean_fk:+.4f} kBT "
        f"across R=1..{max(R_values)} at top budget (negative = oracle better). "
        f"This isolates the pure placement benefit of Lloyd CVT over farthest-point.\n\n"
        f"Full oracle (own calibrated k): mean ΔPMF = {mean_own:+.4f} kBT. "
        f"The sign and magnitude reveal whether the k-tradeoff (oracle k ≠ linspace k) helps or hurts.\n\n"
        f"Coverage: oracle CVT tends to distribute windows more uniformly across the 12 low-F "
        f"sub-basins than farthest-point, which can improve coverage at low R where a single "
        f"misplaced window leaves a basin unsampled."
    )

    doc.add_heading("Caveats", 1)
    doc.add_paragraph(
        f"MBAR uses a {MBAR_CAP//1000}k/window cap; beyond it the metric is cap-limited, "
        f"so the floor is partly the cap + res={RES} grid (same as retire comparison).\n"
        f"1 seed at 50× (compute); exact sampler + ESS/burn-in cost model; absolute numbers synthetic-scale.\n"
        f"oracle_optimal_centers_2d uses res={ORACLE_RES} grid for CVT (vs res={RES} for sampling); "
        f"sufficient for 2D placement but centers are rounded to grid points.\n"
        f"Oracle CVT iterates 30 Lloyd rounds; convergence is typically reached in <15 for R≤12."
    )

    doc.save(out_path)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    t0 = time.time()
    print("Oracle-placement 2D comparison at 50× budget", flush=True)
    print(f"  R={R_VALUES[0]}..{R_VALUES[-1]}, budgets={[b//1000 for b in BUDGETS]}k, "
          f"MBAR_CAP={MBAR_CAP}, seed={SEED}", flush=True)

    lin_centers, oracle_centers = precompute_centers(LAND, R_VALUES)

    print("\nRunning sweep ...", flush=True)
    results = run_sweep(LAND, lin_centers, oracle_centers, R_VALUES, BUDGETS)

    print("\nGenerating figures ...", flush=True)
    fig1 = fig_pmf_vs_r(results, R_VALUES, BUDGETS, OUT_DIR)
    fig2 = fig_delta_pmf_vs_r(results, R_VALUES, BUDGETS, OUT_DIR)
    fig3 = fig_pmf_vs_budget(results, R_VALUES, BUDGETS, OUT_DIR)
    fig4 = fig_coverage_vs_r(results, R_VALUES, BUDGETS, OUT_DIR)
    fig5, lin_rmse_top, oracle_rmse_top = fig_fes_recovery(
        LAND, results, lin_centers, oracle_centers, OUT_DIR,
        budget=BUDGETS[-1], R=12)

    elapsed = time.time() - t0
    print(f"\nTotal elapsed: {elapsed:.1f}s", flush=True)

    out_docx = os.path.join(OUT_DIR, "GAREUS_oracle_placement_2d_50x.docx")
    write_docx(results, R_VALUES, BUDGETS,
               [fig1, fig2, fig3, fig4, fig5],
               lin_centers, oracle_centers,
               lin_rmse_top, oracle_rmse_top,
               elapsed, out_docx)

    print("Done.", flush=True)
