"""Spec 3.7 sample figure by explicit state coordinates (sparse / 2D shape layouts).

``plot_adaptive_diagnostics``'s fig2 heatmap keys states by the rounded (primary_center,
secondary_center) pair, which assumes a regular grid: a sparse or shape-based 2D layout
(chignolin_9: 59 centres, 59 distinct CV2 centres) becomes a mostly-empty matrix. This
figure places every active state at its own (c1, c2), one panel per lambda rung, marker by
restraint pattern, colour = samples (log), and marks trapped_or_orthogonal windows and
weak / unmeasured edges from a ``cv2_resolution_summary`` when one is given.

P6: on an axis the state does not restrain (k <= 0) the registry centre is a placeholder,
so such a state is drawn at its SAMPLED mean on that axis (summary ``cv1_mean``/``cv2_mean``)
with an open marker; without a sampled mean it stays at the placeholder, flagged.

``state_grid_points`` (the data prep) is unit-tested; ``render_state_grid`` is not.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

PATTERN_MARKERS = {(True, True): "o", (True, False): "s", (False, True): "^", (False, False): "D"}
PATTERN_LABELS = {(True, True): "CV1 + CV2 restrained", (True, False): "CV1 only (k2 = 0)",
                  (False, True): "CV2 only (k1 = 0)", (False, False): "unrestrained"}


def _f(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _is_active(row: Mapping[str, Any]) -> bool:
    a = row.get("active", True)
    return str(a).strip().lower() not in ("false", "0", "no", "") if not isinstance(a, bool) else a


def _axis(centre: Optional[float], k: Optional[float], sampled: Optional[float]) -> tuple:
    """(plotted coordinate, restrained, placeholder_left_in_place)."""
    restrained = k is None or k > 0.0          # P6: k not recorded counts as restrained
    if restrained or sampled is None:
        return centre, restrained, not restrained
    return sampled, restrained, False


def state_grid_points(registry_rows: Sequence[Mapping[str, Any]], counts: Mapping[int, int],
                      summary: Optional[Mapping[str, Any]] = None) -> List[Dict[str, Any]]:
    """One point per active registry state: x/y, rung, restraint pattern, samples, flags."""
    by_sid = {int(s["state_id"]): s for s in ((summary or {}).get("states") or [])}
    pts: List[Dict[str, Any]] = []
    for row in registry_rows:
        if not _is_active(row):
            continue
        sid = int(row["state_id"])
        srow = by_sid.get(sid, {})
        x, r1, ph1 = _axis(_f(row.get("primary_center")), _f(row.get("primary_k")), _f(srow.get("cv1_mean")))
        y, r2, ph2 = _axis(_f(row.get("secondary_center")), _f(row.get("secondary_k")), _f(srow.get("cv2_mean")))
        pts.append({"state_id": sid, "x": x, "y": y, "lambda": round(_f(row.get("gamd_lambda")) or 0.0, 6),
                    "pattern": (r1, r2), "marker": PATTERN_MARKERS[(r1, r2)],
                    "sampled_position": (not r1 and not ph1) or (not r2 and not ph2),
                    "placeholder": ph1 or ph2, "samples": int(counts.get(sid, 0) or 0),
                    "trapped": bool(srow.get("trapped_or_orthogonal"))})
    return pts


def flagged_edges(summary: Optional[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Graded edges worth drawing: weak (FAIL-level) and unmeasured (CAUTION-level)."""
    out = []
    for e in (summary or {}).get("edges") or []:
        if not e.get("graded"):
            continue
        if e.get("weak"):
            out.append({"i": int(e["state_i"]), "j": int(e["state_j"]), "kind": "weak"})
        elif e.get("measured") is False:
            out.append({"i": int(e["state_i"]), "j": int(e["state_j"]), "kind": "unmeasured"})
    return out


def _count_range(pts: Sequence[Mapping[str, Any]]) -> tuple:
    """Log colour range over the positive counts (equal counts, as in a lockstep layout, sit mid-scale)."""
    pos = [p["samples"] for p in pts if p["samples"] > 0] or [1]
    lo, hi = min(pos), max(pos)
    return (lo / 2.0, hi * 2.0) if lo == hi else (lo, hi)


def _draw_panel(ax, pts, edges, norm, cmap, lam) -> None:
    here = [p for p in pts if p["lambda"] == lam and p["x"] is not None and p["y"] is not None]
    pos = {p["state_id"]: (p["x"], p["y"]) for p in here}
    for e in edges:
        if e["i"] in pos and e["j"] in pos:
            (x0, y0), (x1, y1) = pos[e["i"]], pos[e["j"]]
            ax.plot([x0, x1], [y0, y1], color="#D55E00" if e["kind"] == "weak" else "#4D4D4D",
                    ls="-" if e["kind"] == "weak" else "--", lw=1.6, zorder=1)
    for pat, marker in PATTERN_MARKERS.items():
        grp = [p for p in here if p["pattern"] == pat]
        if not grp:
            continue
        cols = [cmap(norm(max(p["samples"], 1))) for p in grp]
        opened = [p["sampled_position"] or p["placeholder"] for p in grp]
        ax.scatter([p["x"] for p in grp], [p["y"] for p in grp], marker=marker, s=46, zorder=3,
                   c=["none" if o else c for c, o in zip(cols, opened)], edgecolors=cols, linewidths=1.3)
    trapped = [p for p in here if p["trapped"]]
    if trapped:
        ax.scatter([p["x"] for p in trapped], [p["y"] for p in trapped], s=170, facecolors="none",
                   edgecolors="black", linewidths=1.4, zorder=4)
        for p in trapped:
            ax.annotate(str(p["state_id"]), (p["x"], p["y"]), xytext=(5, 5), textcoords="offset points",
                        fontsize=7)
    ax.set_title(f"λ = {lam:g}  ({len(here)} states)", fontsize=9)
    ax.grid(True, alpha=0.25)


def render_state_grid(pts: Sequence[Mapping[str, Any]], edges: Sequence[Mapping[str, Any]], out_path: Path,
                      *, cv2_label: str = "CV2", title: str = "") -> Optional[Path]:
    """Write the figure; returns the path, or None when there is nothing to plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.lines as mlines
    import matplotlib.pyplot as plt
    if not pts:
        return None
    lams = sorted({p["lambda"] for p in pts})
    ncol = min(len(lams), 2)
    nrow = int(math.ceil(len(lams) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.2 * ncol, 4.6 * nrow), sharex=True, sharey=True,
                             squeeze=False, constrained_layout=True)
    norm, cmap = mcolors.LogNorm(*_count_range(pts)), plt.cm.viridis
    for ax, lam in zip(axes.flat, lams):
        _draw_panel(ax, pts, edges, norm, cmap, lam)
    for ax in list(axes.flat)[len(lams):]:
        ax.set_visible(False)
    for ax in axes[-1, :]:
        ax.set_xlabel("CV1 centre", fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel(f"{cv2_label} centre", fontsize=9)
    handles = [mlines.Line2D([], [], color="#4D4D4D", marker=PATTERN_MARKERS[k], ls="none", label=v)
               for k, v in PATTERN_LABELS.items() if any(p["pattern"] == k for p in pts)]
    handles += [mlines.Line2D([], [], color="#4D4D4D", marker="o", mfc="none", ls="none",
                              label="open: placed at sampled mean (P6 placeholder axis)"),
                mlines.Line2D([], [], color="black", marker="o", mfc="none", ms=11, ls="none",
                              label="trapped_or_orthogonal (3.3 R3)"),
                mlines.Line2D([], [], color="#D55E00", ls="-", label="weak edge (pairwise MBAR)"),
                mlines.Line2D([], [], color="#4D4D4D", ls="--", label="unmeasured edge")]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=8, frameon=False)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=axes, label="samples per state (log)", shrink=0.7)
    fig.suptitle(title or "State coordinates by rung: samples, restraint pattern, CV2-resolution flags",
                 fontsize=11)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


__all__ = ["PATTERN_LABELS", "PATTERN_MARKERS", "flagged_edges", "render_state_grid", "state_grid_points"]
