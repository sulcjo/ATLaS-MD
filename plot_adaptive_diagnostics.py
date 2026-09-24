"""
Diagnostic plots for ATLaS-MD adaptive-production epoch/topup behavior.

Usage:
    python plot_adaptive_diagnostics.py <run_dir> [--out <out_dir>] [--stride N]

Generates:
  fig1_phase_coverage.png   -- per-phase 2D density maps
  fig2_window_layout.png    -- window ellipses + sample counts
  fig3_topup_timeline.png   -- cumulative samples, step allocation, overlap
  fig4_topup_targeting.png  -- state participation matrix per topup round
"""

import argparse
import csv as _csv
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

# The stale-`epoch_window_map.csv` guard, from the package that owns the shared
# entry points for it -- see the section header below. The predicate comes from
# the module that defines the note kinds it reads, not from `loaders`, which
# only re-exports it.
from gareus.mbar_analysis.loaders import figure_epoch_window_map_rows
from gareus.mbar_analysis.loaders_adaptive import window_map_note_rewrote_rows

KB_KCAL = 1.987204e-3  # kcal mol⁻¹ K⁻¹
TEMP_K   = 300.0


# ── utilities ─────────────────────────────────────────────────────────────────

def _rcsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _sigma(k_kcal: float) -> float:
    return math.sqrt((KB_KCAL * TEMP_K) / max(k_kcal, 1e-9))


def _step_from_name(name: str) -> int:
    """Extract cumulative step count from topup_NNN_MMMMM or topup_NNN_MMMMM_windows."""
    parts = name.replace("_windows", "").split("_")
    for tok in reversed(parts):
        if tok.isdigit():
            return int(tok)
    return 0


def _load_parquet_samples(samples_dir: Path, stride: int = 1) -> pd.DataFrame:
    """Load cv1, cv2, window_id from all Parquet chunks under samples_dir."""
    try:
        import pyarrow.dataset as ds
        dataset = ds.dataset(str(samples_dir), format="parquet")
        table = dataset.to_table(columns=["cv1", "cv2", "window_id"])
        df = table.to_pandas()
        if stride > 1:
            df = df.iloc[::stride].reset_index(drop=True)
        return df
    except Exception as exc:
        print(f"  [warn] load {samples_dir}: {exc}", file=sys.stderr)
        return pd.DataFrame(columns=["cv1", "cv2", "window_id"])


# ── stale window-map guard (shared with the MBAR loaders) ─────────────────────
#
# `epoch_window_map.csv` maps a phase's local window_id -> global state_id, and
# it can be STALE: `--us-auto-drop-bad-windows` prunes windows post-pull and
# renumbers the survivors 0..N-1, but the identity map the registry had already
# written over all *active* states was never rewritten, so every local index at
# or after the first dropped one names the wrong state
# (docs/chignolin_6_low_ess_root_cause.md). Every per-state figure here reads
# that file, so an affected run's sample counts and window ellipses were
# silently drawn against the wrong states -- in the one tool an operator
# reaches for when a run looks odd.
#
# Same guard as the MBAR loaders, deliberately different severity: this is a
# figure tool, so an unrepairable map is still drawn (from the stale rows) with
# a loud stderr warning and a red stamp on the figure, rather than refusing.
# The MBAR path fails closed instead, because there a wrong number is published
# as a free energy.
#
# Metadata-only check on purpose: the guard's other input, this phase's own
# sampled window_ids, would mean a full unstrided Parquet read (multi-GB on a
# real run) here at discovery time, and a *strided* read would undercount the
# window indices and manufacture false alarms. A phase that records no window
# count is therefore reported as unchecked rather than treated as healthy.

def _phase_window_map(phase_dir: Path) -> tuple[pd.DataFrame, str | None]:
    """``(wmap, warning)`` for one phase, validated/repaired by the shared guard.

    `warning` is None for the overwhelming majority of phases, and then `wmap`
    is exactly what `_rcsv` returned -- same rows, same dtypes, untouched. On a
    repairable stale map the frame comes back holding the windows the phase
    really ran, in the order it really ran them, with `epoch_window` renumbered
    0..N-1: a drop repair takes the never-run windows' rows out, and a
    permutation repair takes nothing out and reorders instead, so the row count
    alone says nothing about whether a repair happened (see the branch below).
    On an unrepairable one the frame comes back as-is with a warning saying so.
    """
    path = phase_dir / "epoch_window_map.csv"
    df = _rcsv(path)
    if df.empty:
        return df, None
    with path.open(newline="") as f:
        rows = list(_csv.DictReader(f))
    survivors, note = figure_epoch_window_map_rows(phase_dir, rows)
    if note is None:
        return df, None
    print(f"  [STALE MAP] {note}", file=sys.stderr)
    if not window_map_note_rewrote_rows(note):
        # The guard detected something but did not rewrite the rows (an
        # unrepairable fault, or a refusal loaded under
        # GAREUS_ALLOW_STALE_WINDOW_MAP): `survivors` is the map's own row list,
        # so the figures get the original frame plus the warning stamp.
        #
        # Asked of the note, not inferred from `len(survivors) == len(rows)`,
        # which is what stood here. That row-count proxy answered this question
        # correctly only while the only possible repair was REMOVING never-run
        # windows. A permutation repair -- same window set, wrong order -- has
        # nothing to remove, so the counts agree, this branch fired, and the
        # plotter returned the unrepaired frame under a note reading "REPAIRED
        # IN MEMORY": every per-state count and window label in the figure then
        # named a different state than its own caption claimed. Taken on the
        # note OBJECT, before anything copies its text: `kind` does not survive
        # an f-string, and a note that has lost it grades as not-rewritten,
        # which lands back on this (conservative) branch.
        return df, note
    # Re-select the *typed* frame's rows for the survivors, in the repaired
    # order, instead of rebuilding one from the guard's string rows, so every
    # downstream `float(row[...])` sees the same dtypes it always did. state_id
    # keys the join because it is the one field the repair never rewrites --
    # only ever which local window it sits at.
    #
    # That join is only sound while state_id really is unique per map row, so
    # the uniqueness is CHECKED here rather than assumed from the writers. It
    # holds on every real map (audited: all 152 epoch_window_map.csv files under
    # RUNS/), and every current writer enumerates distinct registry states -- but
    # a `pos` dict built last-wins over a duplicated state_id collapses silently:
    # every lookup below would still succeed, `len(keep) != len(survivors)` would
    # still pass, and `df.iloc[keep]` would hand the figures a frame with one row
    # duplicated and another dropped. A wrong frame that passes the guard is the
    # one outcome this whole file's guard exists to prevent, so the guard checks
    # the property it depends on, not a proxy for it.
    sid_col = next((c for c in df.columns if "state_id" in c.lower()), None)
    if sid_col is None:
        return df, note
    pos = {}
    duplicated = []
    for i, v in enumerate(df[sid_col].tolist()):
        try:
            sid = int(v)
        except (TypeError, ValueError):
            continue
        if sid in pos:
            duplicated.append(sid)
            continue
        pos[sid] = i
    if duplicated:
        return df, (note + f" (this phase's map carries state_id {sorted(set(duplicated))} on more "
                           f"than one row, so the repaired rows cannot be matched back onto the "
                           f"CSV's own columns unambiguously; the figures fall back to the STALE "
                           f"map.)")
    keep = []
    for r in survivors:
        try:
            keep.append(pos[int(r["state_id"])])
        except (KeyError, TypeError, ValueError):
            continue
    # Two nets, deliberately: the length check catches a survivor with no row in
    # the frame, and the distinctness check catches two survivors resolving to
    # one row. The second is the corruption shape itself rather than a symptom of
    # it, and it costs one set() over a handful of rows.
    if len(keep) != len(survivors) or len(set(keep)) != len(keep):
        return df, (note + " (the repaired rows could not be matched back onto the CSV's own "
                           "columns, so the figures fall back to the STALE map.)")
    out = df.iloc[keep].copy()
    out["epoch_window"] = list(range(len(out)))
    return out.reset_index(drop=True), note


def _phase_map_notes(phases: list[dict]) -> list[str]:
    """Every stale-map warning collected during phase discovery, in order."""
    return [ph["wmap_note"] for ph in phases if ph.get("wmap_note")]


def _annotate_map_warnings(fig, notes: list[str]) -> None:
    """Stamp stale-window-map warnings onto a figure.

    A figure that is quietly mislabelled is worse than an ugly one: whatever
    the guard could not silently fix has to travel with the picture, not just
    with the terminal it was generated in.
    """
    if not notes:
        return
    uniq = list(dict.fromkeys(notes))
    shown = [n if len(n) <= 240 else n[:239] + "…" for n in uniq[:3]]
    if len(uniq) > len(shown):
        shown.append(f"(+{len(uniq) - len(shown)} further phase(s) affected)")
    fig.text(0.005, 0.001,
             "STALE epoch_window_map.csv — per-state counts/labels in this figure are affected:\n"
             + "\n".join(shown),
             ha="left", va="bottom", fontsize=6, color="crimson", wrap=True)


# ── phase discovery ───────────────────────────────────────────────────────────

def _discover_subrun_phases(base_dir: Path, label_prefix: str) -> list[dict]:
    """baseline + topup_* sub-phases under one epoch/final directory, in true
    chronological order.

    Sorted by directory mtime, NOT by the topup directory's own step-count
    suffix (topup_NNN_MMMMM). MMMMM is that segment's own *extra-step
    duration* -- run_scheduled_adaptive_epoch (gareus/adaptive_production.py)
    groups states needing the same additional steps into one topup_NNN_MMMMM
    segment per call; when a quality-gate extension loop re-invokes this for
    the same epoch/final with a fresh single deficiency group, NNN is always
    1 and MMMMM shrinks round over round as the remaining gap closes. Sorting
    ascending by that number is the *reverse* of creation order whenever a
    phase got more than one extension round -- confirmed on a real run
    (chignolin_5/epoch_001: topup_001_34794000 created first at the largest
    remaining-gap value, then topup_001_25733000, then topup_001_18937000
    created last at the smallest).
    """
    subs = [d for d in base_dir.iterdir()
            if d.is_dir() and (d / "samples").is_dir()
            and (d.name == "baseline" or d.name.startswith("topup_"))]
    subs.sort(key=lambda d: (0 if d.name == "baseline" else 1, d.stat().st_mtime))
    phases = []
    topup_i = 0
    for sub in subs:
        if sub.name == "baseline":
            lbl = f"{label_prefix}\nBaseline"
            step = 0
        else:
            topup_i += 1
            step = _step_from_name(sub.name)
            ns = step * 4e-6
            suffix = f"Topup {topup_i}" if topup_i > 1 or len(subs) > 2 else "Topup"
            lbl = f"{label_prefix} {suffix}\n+{ns:.1f} ns"
        wmap, wmap_note = _phase_window_map(sub)
        phases.append({
            "name": f"{base_dir.name}/{sub.name}",
            "label": lbl,
            "path": sub,
            "wmap": wmap,
            "wmap_note": wmap_note,
            "step": step,
        })
    return phases


def discover_phases(ap_dir: Path) -> list[dict]:
    """
    Return ordered list of phase dicts, each with:
      name, label, path, wmap (DataFrame), step (this phase's own extra-step
      count if it's a topup, else 0 -- not a cumulative/global step count)
    Ordered chronologically: numbered epochs in whatever order/count they
    actually ran, then the final phase's baseline + topup_* segments.

    A numbered epoch can itself use either layout: a flat epoch_NNN/samples/
    (the unscheduled path, e.g. real epoch_000 runs) or -- when
    run_scheduled_adaptive_epoch's allocation_scheduler is active for that
    epoch too, not just for "final" -- its own baseline/topup_* sub-run
    layout, identical in shape to "final"'s. Missing that second case used
    to silently drop the entire epoch from every figure: on a real run
    (chignolin_5) epoch_001 used the scheduled layout and held 61% of the
    run's real samples, but the topup timeline jumped straight from
    "Epoch 0 (initial)" to "Final Baseline" as if epoch_001 never happened.

    As of 2026-07-27, run_adaptive_production_auto_loop's per-epoch
    convergence gate no longer hands the remaining MD-pool budget to
    "final" the moment it reports stop_adaptive -- it just continues to the
    next scheduled epoch, so the loop always runs the full --ap-epochs
    budget before final starts. (Historically it did break early there,
    which is why this function must not assume exactly N phases from
    --ap-epochs alone: it globs whatever epoch_* directories actually ran.)
    """
    phases = []

    epoch_dirs = sorted(
        (d for d in ap_dir.glob("epoch_*") if d.is_dir()),
        key=lambda d: d.name,
    )
    for i, ep in enumerate(epoch_dirs):
        try:
            n = int(ep.name.split("_")[-1])
        except ValueError:
            n = i
        if (ep / "samples").is_dir():
            lbl = f"Epoch {n}\n(initial)" if i == 0 else f"Epoch {n}"
            wmap, wmap_note = _phase_window_map(ep)
            phases.append({
                "name": ep.name,
                "label": lbl,
                "path": ep,
                "wmap": wmap,
                "wmap_note": wmap_note,
                "step": 0,
            })
        else:
            phases.extend(_discover_subrun_phases(ep, f"Epoch {n}"))

    final_dir = ap_dir / "final"
    if final_dir.is_dir():
        phases.extend(_discover_subrun_phases(final_dir, "Final"))

    return phases


_CV2_LABELS = {
    "torsion-pca": "torsion-PCA",
    "tica-linear": "tICA",
    "rama-map": "rama",
    "contacts": "contacts",
    "distance": "distance",
}


def _secondary_cv_label(phases: list[dict]) -> str:
    """Best-effort human label for the CV2 axis, from the last phase's
    run_manifest.json. CV2 type can switch mid-run (e.g. torsion-pca ->
    tica-linear after a tICA refit); the last phase is where most of the
    plotted production weight actually sits, so prefer its label over
    epoch 0's -- hardcoding "rama"/"rama-map" regardless of actual CV2 type
    mislabels every run that doesn't use rama-map (e.g. torsion-pca,
    tica-linear, or no secondary CV at all)."""
    import json as _json
    for ph in reversed(phases):
        manifest = ph["path"] / "run_manifest.json"
        if not manifest.exists():
            continue
        try:
            d = _json.loads(manifest.read_text())
            cv2 = d.get("method_settings", {}).get("secondary_cv")
        except Exception:
            continue
        if cv2:
            return _CV2_LABELS.get(cv2, str(cv2))
    return "secondary"


def _wmap_to_sid(wmap: pd.DataFrame) -> dict[int, int]:
    """epoch_window → state_id mapping from wmap DataFrame."""
    if wmap.empty:
        return {}
    sid_col = next((c for c in wmap.columns if "state_id" in c.lower()), None)
    if sid_col is None:
        return {}
    return {int(r["epoch_window"]): int(r[sid_col]) for _, r in wmap.iterrows()}


def _sample_count_grid(state_reg: pd.DataFrame, state_counts: dict[int, int]):
    """Build the primary x secondary sample-count grid for fig_window_layout's
    heatmap panel. Returns (p_vals, s_vals, grid) where grid[i, j] is the
    total sample count for the state at (p_vals[i], s_vals[j]).

    Dict keys are rounded the same way on both construction and lookup --
    secondary_center is a continuous CV projection with far more than 6
    significant digits, so keying by the raw unrounded float while looking
    up a rounded one made every lookup miss silently (grid stayed all-zero)."""
    p_vals = sorted(state_reg["primary_center"].unique())
    s_vals = sorted(state_reg["secondary_center"].unique())
    p_idx = {round(float(v), 6): i for i, v in enumerate(p_vals)}
    s_idx = {round(float(v), 6): i for i, v in enumerate(s_vals)}
    grid = np.zeros((len(p_vals), len(s_vals)))
    for _, row in state_reg.iterrows():
        sid = int(row["state_id"])
        pi = p_idx.get(round(float(row["primary_center"]), 6))
        si = s_idx.get(round(float(row["secondary_center"]), 6))
        if pi is not None and si is not None:
            grid[pi, si] = state_counts.get(sid, 0)
    return p_vals, s_vals, grid


# ── figure 1: per-phase 2D density maps ──────────────────────────────────────

def fig_phase_coverage(phases: list[dict], state_reg: pd.DataFrame,
                        stride: int, out_path: Path) -> None:
    """Small-multiples of 2D sample density per epoch/topup phase."""
    n = len(phases)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    sig1 = _sigma(250.0)
    sig2 = _sigma(100.0)
    cv2_label = _secondary_cv_label(phases)

    # compute global CV range from registry
    if not state_reg.empty:
        cv1_cen = state_reg["primary_center"].astype(float)
        cv2_cen = state_reg["secondary_center"].astype(float)
        x_edges = np.linspace(max(0, cv1_cen.min() - 4*sig1), cv1_cen.max() + 4*sig1, 65)
        y_edges = np.linspace(cv2_cen.min() - 3*sig2, cv2_cen.max() + 3*sig2, 65)
    else:
        x_edges = np.linspace(0, 0.3, 65)
        y_edges = np.linspace(-1.2, 1.2, 65)

    # load per-phase histograms
    hists, counts, colors = [], [], []
    phase_colors = plt.cm.tab10(np.linspace(0, 0.9, n))
    for ph, pc in zip(phases, phase_colors):
        sp = ph["path"] / "samples"
        if sp.is_dir():
            df = _load_parquet_samples(sp, stride=stride)
        else:
            df = pd.DataFrame(columns=["cv1", "cv2"])
        if len(df):
            h, _, _ = np.histogram2d(df["cv1"], df["cv2"], bins=[x_edges, y_edges])
        else:
            h = np.zeros((len(x_edges)-1, len(y_edges)-1))
        hists.append(h)
        counts.append(len(df))
        colors.append(pc)

    global_max = max(h.max() for h in hists) if hists else 1.0

    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4.5*nrows),
                              constrained_layout=True)
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    for ax, ph, h, n_samp, pc in zip(axes_flat, phases, hists, counts, colors):
        hm = np.ma.masked_where(h == 0, h)
        ax.pcolormesh(x_edges, y_edges, hm.T, cmap="YlOrRd",
                      vmin=0, vmax=global_max, rasterized=True)

        # window ellipses from wmap
        wmap = ph["wmap"]
        if not wmap.empty:
            for _, row in wmap.iterrows():
                cx = float(row.get("primary_center", row.get("primary_cv_center", np.nan)))
                cy = float(row.get("secondary_center", row.get("secondary_cv_center", np.nan)))
                if math.isfinite(cx) and math.isfinite(cy):
                    ell = mpatches.Ellipse((cx, cy), 2*sig1, 2*sig2,
                                           fill=False, edgecolor="white",
                                           linewidth=1.2, linestyle="--", alpha=0.85)
                    ax.add_patch(ell)
                    ax.plot(cx, cy, "w+", ms=5, mew=1.5, alpha=0.8)

        ax.set_xlim(x_edges[0], x_edges[-1])
        ax.set_ylim(y_edges[0], y_edges[-1])
        ax.set_xlabel("CV1 (contacts)", fontsize=8)
        ax.set_ylabel(f"CV2 ({cv2_label})", fontsize=8)
        ax.set_title(f"{ph['label']}\n{n_samp:,} pts (stride {stride})",
                     fontsize=9, color=pc)
        ax.tick_params(labelsize=7)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    sm = plt.cm.ScalarMappable(cmap="YlOrRd", norm=mcolors.Normalize(0, global_max))
    sm.set_array([])
    fig.colorbar(sm, ax=axes_flat[:n], shrink=0.6, label="Strided sample count per bin")
    fig.suptitle("Phase-space Coverage: each Epoch/Topup Phase\n"
                 "(dashed ellipses = ±1σ harmonic width)", fontsize=11)
    _annotate_map_warnings(fig, _phase_map_notes(phases))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


# ── figure 2: window layout ───────────────────────────────────────────────────

def fig_window_layout(state_reg: pd.DataFrame, phases: list[dict],
                       stride: int, out_path: Path) -> None:
    """Window ellipses on total-density background + sample count summary."""
    if state_reg.empty:
        print("[warn] state_registry empty", file=sys.stderr)
        return

    sig1 = _sigma(250.0)
    sig2 = _sigma(100.0)
    cv2_label = _secondary_cv_label(phases)

    # total samples per state_id
    state_counts: dict[int, int] = {}
    for ph in phases:
        sp = ph["path"] / "samples"
        if not sp.is_dir():
            continue
        df = _load_parquet_samples(sp, stride=1)
        ew2sid = _wmap_to_sid(ph["wmap"])
        for ew, sid in ew2sid.items():
            cnt = int((df["window_id"] == ew).sum())
            state_counts[sid] = state_counts.get(sid, 0) + cnt

    # all-sample density
    cv1_all, cv2_all = [], []
    for ph in phases:
        sp = ph["path"] / "samples"
        if sp.is_dir():
            df = _load_parquet_samples(sp, stride=stride)
            if len(df):
                cv1_all.append(df["cv1"].values)
                cv2_all.append(df["cv2"].values)

    fig = plt.figure(figsize=(15, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[3, 1], width_ratios=[2.2, 1, 1],
                          hspace=0.35, wspace=0.3)
    ax_main = fig.add_subplot(gs[0, :2])
    ax_bar  = fig.add_subplot(gs[0, 2])
    ax_heat = fig.add_subplot(gs[1, :])

    # ─ main: hexbin background + ellipses ─
    if cv1_all:
        cv1_cat = np.concatenate(cv1_all)
        cv2_cat = np.concatenate(cv2_all)
        hb = ax_main.hexbin(cv1_cat, cv2_cat, gridsize=70, cmap="Greys",
                            mincnt=1, linewidths=0, alpha=0.7, rasterized=True)

    vmax_cnt = max(state_counts.values()) if state_counts else 1
    vmin_cnt = max(1, min(state_counts.values())) if state_counts else 1
    norm_c = mcolors.LogNorm(vmin=vmin_cnt, vmax=max(vmin_cnt + 1, vmax_cnt))
    cmap_w = plt.cm.plasma

    for _, row in state_reg.iterrows():
        sid = int(row["state_id"])
        cx = float(row["primary_center"])
        cy = float(row["secondary_center"])
        k1 = float(row["primary_k"])
        k2 = float(row["secondary_k"])
        s1, s2 = _sigma(k1), _sigma(k2)
        cnt = state_counts.get(sid, 0)
        fc = cmap_w(norm_c(max(cnt, vmin_cnt)))
        ell = mpatches.Ellipse((cx, cy), 2*s1, 2*s2,
                               fill=True, facecolor=(*fc[:3], 0.40),
                               edgecolor=fc, linewidth=2.2)
        ax_main.add_patch(ell)
        ax_main.text(cx, cy, str(sid), ha="center", va="center",
                     fontsize=7.5, fontweight="bold",
                     color="white" if cnt > vmax_cnt * 0.3 else "black")

    sm = plt.cm.ScalarMappable(cmap=cmap_w, norm=norm_c)
    sm.set_array([])
    fig.colorbar(sm, ax=ax_main, label="Total samples per window (log)", shrink=0.8)

    cv1_cen = state_reg["primary_center"].astype(float)
    cv2_cen = state_reg["secondary_center"].astype(float)
    ax_main.set_xlim(max(0, cv1_cen.min() - 4*sig1), cv1_cen.max() + 4*sig1)
    ax_main.set_ylim(cv2_cen.min() - 3*sig2, cv2_cen.max() + 3*sig2)
    ax_main.set_xlabel("CV1 (nonlocal contact fraction)", fontsize=10)
    ax_main.set_ylabel(f"CV2 ({cv2_label})", fontsize=10)
    ax_main.set_title("Window Placement: ±1σ Harmonic Ellipses\n"
                       "Fill color = total samples (log scale)", fontsize=10)

    # ─ bar chart ─
    sids  = sorted(state_counts)
    cnts  = [state_counts[s] for s in sids]
    bclrs = [cmap_w(norm_c(max(c, vmin_cnt))) for c in cnts]
    ax_bar.barh(sids, cnts, color=bclrs, height=0.7)
    ax_bar.set_xlabel("Total samples", fontsize=9)
    ax_bar.set_ylabel("State ID", fontsize=9)
    ax_bar.set_title("Samples\nper State", fontsize=9)
    ax_bar.invert_yaxis()
    ax_bar.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k"))
    ax_bar.tick_params(labelsize=8)
    for s, c in zip(sids, cnts):
        ax_bar.text(c + max(cnts)*0.01, s, f"{c:,}", va="center", fontsize=6)

    # ─ 2D sample count grid ─
    p_vals, s_vals, grid = _sample_count_grid(state_reg, state_counts)

    gmin = grid[grid > 0].min() if grid.any() else 1
    im = ax_heat.imshow(grid.T, aspect="auto", origin="lower",
                         cmap="plasma", norm=mcolors.LogNorm(vmin=gmin, vmax=max(gmin+1, grid.max())))
    ax_heat.set_xticks(range(len(p_vals)))
    ax_heat.set_xticklabels([f"{v:.4f}" for v in p_vals], fontsize=8, rotation=40, ha="right")
    ax_heat.set_yticks(range(len(s_vals)))
    ax_heat.set_yticklabels([f"{v:.3f}" for v in s_vals], fontsize=8)
    ax_heat.set_xlabel("Primary center (contact fraction)", fontsize=9)
    ax_heat.set_ylabel(f"Secondary center ({cv2_label})", fontsize=9)
    ax_heat.set_title("Sample Count Grid: Primary × Secondary CV Centers", fontsize=9)
    for pi_i, pv in enumerate(p_vals):
        for si_i, sv in enumerate(s_vals):
            cnt = int(grid[pi_i, si_i])
            if cnt > 0:
                lbl = f"{cnt//1000}k" if cnt >= 1000 else str(cnt)
                ax_heat.text(pi_i, si_i, lbl, ha="center", va="center",
                             fontsize=7, color="white" if cnt > grid.max()*0.3 else "black")
    fig.colorbar(im, ax=ax_heat, label="Samples (log)", shrink=0.5)

    fig.suptitle("ATLaS-MD: Window Layout & Sample Distribution", fontsize=12)
    _annotate_map_warnings(fig, _phase_map_notes(phases))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


# ── figure 3: topup timeline + overlap ───────────────────────────────────────

def fig_topup_timeline(state_reg: pd.DataFrame, phases: list[dict],
                        ap_dir: Path, out_path: Path) -> None:
    """
    A. Cumulative samples per state across phases.
    B. Epoch-1 step allocation (from epoch_schedule.csv).
    C. Analytical window overlap matrix.
    """
    all_sids = sorted(state_reg["state_id"].astype(int).tolist()) if not state_reg.empty else []

    # per-phase sample counts per state
    phase_state_counts = []
    for ph in phases:
        sp = ph["path"] / "samples"
        if not sp.is_dir():
            phase_state_counts.append({})
            continue
        df = _load_parquet_samples(sp, stride=1)
        ew2sid = _wmap_to_sid(ph["wmap"])
        counts = {}
        for ew, sid in ew2sid.items():
            counts[sid] = counts.get(sid, 0) + int((df["window_id"] == ew).sum())
        phase_state_counts.append(counts)

    # cumulative
    cum = np.zeros((len(all_sids), len(phases)), dtype=np.int64)
    for pi, pc in enumerate(phase_state_counts):
        for si, sid in enumerate(all_sids):
            cum[si, pi] = pc.get(sid, 0)
    cum_c = np.cumsum(cum, axis=1)

    sched = _rcsv(ap_dir / "final" / "epoch_schedule.csv")

    fig, axes = plt.subplots(1, 3, figsize=(18, 7), constrained_layout=True)
    ax_cum, ax_sched, ax_ovlp = axes

    # ─ A: cumulative sample count ─
    colors_st = plt.cm.tab20(np.linspace(0, 1, max(len(all_sids), 1)))
    xlabels = [ph["label"].replace("\n", " ") for ph in phases]
    xpos = np.arange(len(phases))

    for si, (sid, color) in enumerate(zip(all_sids, colors_st)):
        vals = cum_c[si]
        ax_cum.plot(xpos, vals, "o-", color=color, label=f"S{sid}",
                    ms=5, lw=1.8, alpha=0.85)
        if vals[-1] > 0:
            ax_cum.text(len(phases) - 1 + 0.08, float(vals[-1]),
                        f" {sid}", va="center", fontsize=7, color=color)

    ax_cum.set_xticks(xpos)
    ax_cum.set_xticklabels(xlabels, rotation=40, ha="right", fontsize=7.5)
    ax_cum.set_ylabel("Cumulative samples", fontsize=9)
    ax_cum.set_title("A. Cumulative Samples per State\nacross Phases", fontsize=9)
    ax_cum.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v/1e3:.0f}k"))
    ax_cum.grid(True, alpha=0.3)

    # ─ B: epoch_schedule ─
    if not sched.empty:
        sids_s  = sched["state_id"].astype(int).tolist()
        bl_s    = sched["baseline_steps"].astype(float).tolist()
        ex_s    = sched["extra_steps"].astype(float).tolist()
        reasons = sched["allocation_reason"].tolist()
        ypos = np.arange(len(sids_s))
        bar_ec = ["#e07b39" if "graph_bridge" in r else "#6ab187" for r in reasons]
        ax_sched.barh(ypos, bl_s, color="#aecde8", height=0.55, label="Baseline")
        ax_sched.barh(ypos, ex_s, left=bl_s, color=bar_ec, height=0.55, alpha=0.9, label="Extra")
        ax_sched.set_yticks(ypos)
        ax_sched.set_yticklabels([f"S{s}" for s in sids_s], fontsize=8)
        ax_sched.set_xlabel("Requested steps", fontsize=9)
        ax_sched.set_title("B. Final-Phase Step Allocation\n(orange=bridge extra, teal=standard extra)", fontsize=9)
        ax_sched.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v/1e6:.1f}M"))
        ax_sched.invert_yaxis()
        ax_sched.grid(axis="x", alpha=0.3)
        handles = [mpatches.Patch(color="#aecde8", label="Baseline"),
                   mpatches.Patch(color="#6ab187", label="Standard extra"),
                   mpatches.Patch(color="#e07b39", label="Bridge-state extra")]
        ax_sched.legend(handles=handles, fontsize=8)
    else:
        ax_sched.text(0.5, 0.5, "No epoch_schedule.csv", ha="center", va="center",
                      transform=ax_sched.transAxes)

    # ─ C: analytical overlap matrix ─
    if not state_reg.empty:
        K = len(all_sids)
        sid_map = {int(r["state_id"]): r for _, r in state_reg.iterrows()}
        ovlp = np.zeros((K, K))
        for i, si in enumerate(all_sids):
            ri = sid_map.get(si)
            if ri is None:
                continue
            for j, sj in enumerate(all_sids):
                if i == j:
                    ovlp[i, j] = 1.0
                    continue
                rj = sid_map.get(sj)
                if rj is None:
                    continue
                s1i = _sigma(float(ri["primary_k"]))
                s2i = _sigma(float(ri["secondary_k"]))
                s1j = _sigma(float(rj["primary_k"]))
                s2j = _sigma(float(rj["secondary_k"]))
                d1 = abs(float(ri["primary_center"]) - float(rj["primary_center"])) / math.sqrt(s1i**2 + s1j**2)
                d2 = abs(float(ri["secondary_center"]) - float(rj["secondary_center"])) / math.sqrt(s2i**2 + s2j**2)
                ovlp[i, j] = math.exp(-0.5 * (d1**2 + d2**2))

        im = ax_ovlp.imshow(ovlp, cmap="hot_r", vmin=0, vmax=1, aspect="auto")
        ticks = [str(s) for s in all_sids]
        ax_ovlp.set_xticks(range(K)); ax_ovlp.set_xticklabels(ticks, fontsize=7, rotation=90)
        ax_ovlp.set_yticks(range(K)); ax_ovlp.set_yticklabels(ticks, fontsize=7)
        ax_ovlp.set_title("C. Analytical Window Overlap\nexp(−½|Δc|²/σ²) Gaussian metric", fontsize=9)
        fig.colorbar(im, ax=ax_ovlp, label="Overlap (0→1)", shrink=0.8)
        for i in range(K):
            for j in range(K):
                v = ovlp[i, j]
                if i != j and v >= 0.01:
                    ax_ovlp.text(j, i, f"{v:.2f}", ha="center", va="center",
                                 fontsize=5.5, color="white" if v > 0.5 else "black")

    fig.suptitle("ATLaS-MD: Topup Timeline, Allocation & Window Overlap", fontsize=12)
    _annotate_map_warnings(fig, _phase_map_notes(phases))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


# ── figure 4: topup participation × state matrix ─────────────────────────────

def fig_topup_targeting(ap_dir: Path, state_reg: pd.DataFrame, out_path: Path) -> None:
    """
    Left: which states were active in each topup (from topup_windows.csv).
    Right: actual samples accumulated per state per topup subdir (from Parquet).
    """
    all_sids = sorted(state_reg["state_id"].astype(int).tolist()) if not state_reg.empty else []
    sid_idx  = {s: i for i, s in enumerate(all_sids)}
    final_dir = ap_dir / "final"

    # ─ topup participation from windows CSV files ─
    sched = _rcsv(final_dir / "epoch_schedule.csv")
    sid_to_bl = {}
    if not sched.empty:
        for _, row in sched.iterrows():
            sid_to_bl[int(row["state_id"])] = float(row.get("baseline_steps", 0))

    topup_entries: list[dict] = []  # {label, step, present: set[int]}

    # baseline from baseline_windows.csv
    bl_wmap = _rcsv(final_dir / "baseline_windows.csv")
    if not bl_wmap.empty:
        sid_col = next((c for c in bl_wmap.columns if "state_id" in c.lower()), None)
        present = set(bl_wmap[sid_col].astype(int).tolist()) if sid_col else set()
        topup_entries.append({"label": "Baseline\n(0 ns)", "step": 0, "present": present})

    # each topup_*_windows.csv in true chronological (mtime) order -- the
    # step-count suffix is that topup's own extra-step duration, not a
    # cumulative/creation-order marker (see _discover_subrun_phases).
    topup_csvs = sorted(final_dir.glob("topup_*_windows.csv"), key=lambda p: p.stat().st_mtime)
    for tcsv in topup_csvs:
        df = _rcsv(tcsv)
        if df.empty:
            continue
        sid_col = next((c for c in df.columns if "state_id" in c.lower()), None)
        present = set(df[sid_col].astype(int).tolist()) if sid_col else set()
        step = _step_from_name(tcsv.stem)
        ns = step * 4e-6
        topup_entries.append({"label": f"Topup\n{ns:.1f} ns", "step": step, "present": present})

    # ─ actual samples per state per topup subdir (from Parquet) ─
    subdirs = [d for d in final_dir.iterdir()
               if d.is_dir() and (d/"samples").is_dir()
               and (d.name == "baseline" or d.name.startswith("topup_"))]
    subdirs.sort(key=lambda p: (0 if p.name == "baseline" else 1, p.stat().st_mtime))

    K = len(all_sids)
    T_part = len(topup_entries)
    T_samp = len(subdirs)

    part_mat = np.zeros((K, T_part))
    samp_mat = np.zeros((K, T_samp), dtype=np.int64)
    samp_xlabels = []
    map_notes: list[str] = []

    for ti, entry in enumerate(topup_entries):
        for sid in entry["present"]:
            if sid in sid_idx:
                part_mat[sid_idx[sid], ti] = 1

    for ti, sub in enumerate(subdirs):
        step = _step_from_name(sub.name) if sub.name != "baseline" else 0
        ns = step * 4e-6
        samp_xlabels.append(f"{sub.name[:12]}…\n{ns:.1f} ns" if len(sub.name) > 12 else f"{sub.name}\n{ns:.1f} ns")
        sp = sub / "samples"
        if not sp.is_dir():
            continue
        df = _load_parquet_samples(sp, stride=1)
        wmap, wmap_note = _phase_window_map(sub)
        if wmap_note:
            map_notes.append(wmap_note)
        ew2sid = _wmap_to_sid(wmap)
        for ew, sid in ew2sid.items():
            if sid in sid_idx:
                cnt = int((df["window_id"] == ew).sum())
                samp_mat[sid_idx[sid], ti] += cnt

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    ax_part, ax_samp = axes

    # ─ participation ─
    part_xlabels = [e["label"] for e in topup_entries]
    ylabels = [f"State {s}" for s in all_sids]

    ax_part.imshow(part_mat, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax_part.set_xticks(range(T_part))
    ax_part.set_xticklabels(part_xlabels, rotation=45, ha="right", fontsize=8)
    ax_part.set_yticks(range(K))
    ax_part.set_yticklabels(ylabels, fontsize=8)
    ax_part.set_title("State Participation per Topup Round\n"
                       "(from topup_*_windows.csv — blue = active)", fontsize=9)
    # annotate state centers
    for si, sid in enumerate(all_sids):
        row = state_reg[state_reg["state_id"].astype(int) == sid]
        if not row.empty:
            cx = float(row["primary_center"].iloc[0])
            cy = float(row["secondary_center"].iloc[0])
            ax_part.text(-0.6, si, f"({cx:.3f},{cy:.2f})",
                         ha="right", va="center", fontsize=6, color="gray",
                         transform=ax_part.get_yaxis_transform())

    # ─ actual sample counts ─
    smin = samp_mat[samp_mat > 0].min() if samp_mat.any() else 1
    smax = samp_mat.max()
    snorm = mcolors.LogNorm(vmin=smin, vmax=max(smin+1, smax))
    im2 = ax_samp.imshow(samp_mat, aspect="auto", cmap="plasma", norm=snorm)
    ax_samp.set_xticks(range(T_samp))
    ax_samp.set_xticklabels(samp_xlabels, rotation=45, ha="right", fontsize=8)
    ax_samp.set_yticks(range(K))
    ax_samp.set_yticklabels(ylabels, fontsize=8)
    ax_samp.set_title("Actual Samples per State per Topup\n(Parquet counts, log scale)", fontsize=9)
    for i in range(K):
        for j in range(T_samp):
            cnt = int(samp_mat[i, j])
            if cnt > 0:
                lbl = f"{cnt//1000}k" if cnt >= 1000 else str(cnt)
                ax_samp.text(j, i, lbl, ha="center", va="center", fontsize=6,
                             color="white" if samp_mat[i,j] > smax*0.2 else "black")
    fig.colorbar(im2, ax=ax_samp, label="Samples (log scale)", shrink=0.8)

    fig.suptitle("ATLaS-MD: Topup Window Targeting & Sample Accumulation", fontsize=12)
    _annotate_map_warnings(fig, map_notes)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


# ── main entry point ──────────────────────────────────────────────────────────

def run_all(run_dir: Path, out_dir: Path, stride: int = 20,
            skip_coverage: bool = False) -> None:
    """Generate all diagnostic figures. Called by analyze_gareus_mbar or standalone."""
    ap_dir = run_dir / "adaptive_production"
    if not ap_dir.is_dir():
        print(f"[adaptive_diag] no adaptive_production/ in {run_dir}", file=sys.stderr)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[adaptive_diag] run={run_dir.name}  out={out_dir}  stride={stride}")

    state_reg = _rcsv(ap_dir / "state_registry.csv")
    if state_reg.empty:
        state_reg = _rcsv(ap_dir / "final_registry_used_for_mbar.csv")
    print(f"  {len(state_reg)} states, discovering phases …")

    phases = discover_phases(ap_dir)
    print(f"  phases: {[p['name'] for p in phases]}")

    if not skip_coverage:
        print("[fig1] phase-coverage …")
        fig_phase_coverage(phases, state_reg, stride, out_dir / "adaptive_fig1_phase_coverage.png")

    print("[fig2] window layout …")
    fig_window_layout(state_reg, phases, stride, out_dir / "adaptive_fig2_window_layout.png")

    print("[fig3] topup timeline …")
    fig_topup_timeline(state_reg, phases, ap_dir, out_dir / "adaptive_fig3_topup_timeline.png")

    print("[fig4] topup targeting …")
    fig_topup_targeting(ap_dir, state_reg, out_dir / "adaptive_fig4_topup_targeting.png")

    print(f"  [adaptive_diag] done → {out_dir}/")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir")
    p.add_argument("--out", default=None)
    p.add_argument("--stride", type=int, default=20)
    p.add_argument("--no-coverage", action="store_true")
    args = p.parse_args()

    run_dir = Path(args.run_dir).resolve()
    out_dir = Path(args.out) if args.out else run_dir.parent / f"{run_dir.name}_adaptive_diag"
    run_all(run_dir, out_dir, stride=args.stride, skip_coverage=args.no_coverage)


if __name__ == "__main__":
    main()
