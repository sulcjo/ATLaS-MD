"""Figures for the pairwise MBAR overlap graph (see ``overlap_graph``).

Writes, into the analysis output directory:

  overlap_matrix.png            states at (CV1, CV2, lambda) labelled by state index,
                                edges coloured by pairwise overlap (red dashed below the
                                threshold), and translucent lambda sheets of the overlap
                                density; two view angles. CV1-only runs: a 2D (CV1, lambda)
                                version with the density as strips.
  overlap_density_layers.png    the same density sheets as flat panels, one per layer.
  overlap_pairs_mbar.csv        one row per edge.
  overlap_graph_3d.html         interactive version (only with CV2 and plotly importable).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from gareus.mbar_analysis.overlap_graph import (
    PAIR_KIND_GAP,
    PAIR_KIND_RUNG,
    graph_pairs,
    layer_value,
    pair_overlap_densities,
    write_pairs_csv,
)

EDGE_VMAX = 0.5          # two identical states give 0.5 on the pairwise scale
DENSITY_FLOOR = 1e-3          # fraction of the max below which a bin is left transparent
DENSITY_CMAP = "YlOrRd"
_VIEWS = ((22, -60), (22, 30))
_LAMBDA_DECIMALS = 6
OUTPUT_FILES = ("overlap_matrix.png", "overlap_density_layers.png", "overlap_pairs_mbar.csv",
                "overlap_graph_3d.html")
SAME_RUNG, RUNG_PAIR = "same", "rung"


def remove_overlap_graph_outputs(out: Path) -> None:
    """Delete this module's outputs so a skipped or failed render never leaves a previous run's files."""
    for name in OUTPUT_FILES:
        try:
            (Path(out) / name).unlink()
        except FileNotFoundError:
            pass


def _grid_edges(values: np.ndarray, centres: np.ndarray, bins: int) -> Optional[np.ndarray]:
    vals = values[np.isfinite(values)]
    cen = centres[np.isfinite(centres)]
    if vals.size == 0 and cen.size == 0:
        return None
    lo_hi = []
    if vals.size:
        lo_hi += list(np.percentile(vals, [0.5, 99.5]))
    if cen.size:
        lo_hi += [float(cen.min()), float(cen.max())]
    lo, hi = min(lo_hi), max(lo_hi)
    pad = 0.02 * (hi - lo) if hi > lo else 0.5
    return np.linspace(lo - pad, hi + pad, int(bins) + 1)


def _layers(results: list[dict]) -> dict[tuple[float, str], np.ndarray]:
    """Summed density per ``(lambda, 'same'|'rung')`` sheet.

    Rung pairs keep their own sheet even when their mid-lambda coincides with
    a real rung (a centre missing its intermediate rung), so a rung-pair
    density is never summed into, or labelled as, a same-rung sheet."""
    layers: dict[tuple[float, str], np.ndarray] = {}
    for r in results:
        key = (round(float(r["layer"]), _LAMBDA_DECIMALS), RUNG_PAIR if r["kind"] == PAIR_KIND_RUNG else SAME_RUNG)
        layers[key] = layers.get(key, 0.0) + r["density"]
    return dict(sorted(layers.items()))


def _density_norm(layers: dict[float, np.ndarray]):
    """Linear 0..max: real overlap densities vary within one decade, where a log scale shows no contrast."""
    from matplotlib.colors import Normalize
    vmax = max((float(np.max(d)) for d in layers.values()), default=0.0)
    if not vmax > 0:
        return None
    return Normalize(vmin=0.0, vmax=vmax)


def _masked(density: np.ndarray, norm) -> np.ma.MaskedArray:
    """Hide bins with (almost) no shared weight so empty CV space stays transparent."""
    return np.ma.masked_less_equal(density, norm.vmax * DENSITY_FLOOR)


def _edge_classes(results, threshold):
    """Split edges into measured-good, measured-weak and unmeasured (NaN) -- never weak."""
    good, weak, unmeasured = [], [], []
    for r in results:
        ov = r["overlap"]
        (unmeasured if not math.isfinite(ov) else good if ov >= threshold else weak).append(r)
    return good, weak, unmeasured


def _edge_collections(results, pos, threshold, three_d):
    """``(collections, counts)``: good edges coloured by overlap, weak ones red dashed,
    unmeasured ones grey dotted (an edge that could not be measured is not weak).

    ``counts`` holds each collection's segment count -- a Line3DCollection
    reports no segments from ``get_segments()`` until it has been projected, so
    callers must not test emptiness that way."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    if three_d:
        from mpl_toolkits.mplot3d.art3d import Line3DCollection as LC
    else:
        from matplotlib.collections import LineCollection as LC
    good, weak, unmeasured = _edge_classes(results, threshold)
    seg = lambda rows: [[pos[r["i"]], pos[r["j"]]] for r in rows]  # noqa: E731
    good_lc = LC(seg(good), cmap=plt.get_cmap("viridis"), norm=Normalize(0.0, EDGE_VMAX), linewidths=1.6)
    good_lc.set_array(np.asarray([r["overlap"] for r in good], dtype=float))
    weak_lc = LC(seg(weak), colors="#d62728", linestyles="--", linewidths=1.8)
    none_lc = LC(seg(unmeasured), colors="0.55", linestyles=":", linewidths=1.4)
    return (good_lc, weak_lc, none_lc), (len(good), len(weak), len(unmeasured))


def _label_states(ax, pos, three_d, fontsize=5):
    for w, p in enumerate(pos):
        if not np.all(np.isfinite(p)):
            continue
        if three_d:
            ax.text(p[0], p[1], p[2], str(w), fontsize=fontsize, color="black")
        else:
            ax.annotate(str(w), (p[0], p[1]), fontsize=fontsize, xytext=(2, 2), textcoords="offset points")


def _plot_3d(out, pos, results, layers, e1, e2, threshold, labels):
    import matplotlib.pyplot as plt
    norm = _density_norm(layers)
    x = 0.5 * (e1[:-1] + e1[1:]); y = 0.5 * (e2[:-1] + e2[1:])
    X, Y = np.meshgrid(x, y, indexing="ij")
    lam_values = sorted(set(pos[:, 2][np.isfinite(pos[:, 2])]) | {z for z, _g in layers})
    fig = plt.figure(figsize=(17, 8))
    good_lc = None
    for p, (elev, azim) in enumerate(_VIEWS):
        ax = fig.add_subplot(1, 2, p + 1, projection="3d", computed_zorder=False)  # draw order = call order
        if norm is not None:
            levels = np.linspace(norm.vmax * DENSITY_FLOOR, norm.vmax, 12)
            for (z, _group), dens in layers.items():
                if float(np.max(dens)) > norm.vmax * DENSITY_FLOOR:
                    cs = ax.contourf(X, Y, _masked(dens, norm), zdir="z", offset=z, levels=levels,
                                     norm=norm, cmap=DENSITY_CMAP, alpha=0.55, zorder=1)
        collections, counts = _edge_collections(results, pos, threshold, three_d=True)
        good_lc = collections[0]
        for lc, count in zip(collections, counts):
            if count:
                lc.set_zorder(3)
                ax.add_collection3d(lc, autolim=False)
        ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=10, c="black", depthshade=False, zorder=4)
        _label_states(ax, pos, three_d=True)
        ax.set_xlim(e1[0], e1[-1]); ax.set_ylim(e2[0], e2[-1])
        zpad = 0.05 * (max(lam_values) - min(lam_values)) if len(lam_values) > 1 else 0.5
        ax.set_zlim(min(lam_values) - zpad, max(lam_values) + zpad)
        ax.set_xlabel(labels[0]); ax.set_ylabel(labels[1]); ax.set_zlabel("GaMD λ")
        ax.view_init(elev=elev, azim=azim)
        ax.set_box_aspect((1.3, 1.0, 1.1), zoom=1.1)
    fig.colorbar(good_lc, cax=fig.add_axes([0.83, 0.2, 0.012, 0.6]), label="pairwise MBAR overlap √(O_ij·O_ji)")
    if norm is not None:
        fig.colorbar(cs, cax=fig.add_axes([0.91, 0.2, 0.012, 0.6]), label="overlap density (per bin)")
    fig.subplots_adjust(left=0.0, right=0.80, top=0.93, bottom=0.02, wspace=0.05)
    fig.suptitle(f"Pairwise MBAR state overlap (red dashed: < {threshold:g}; grey dotted: unmeasured); "
                 "sheets: overlap integrand, rung pairs at mid-λ")
    fig.savefig(out / "overlap_matrix.png", dpi=200, bbox_inches="tight"); plt.close(fig)


def _plot_2d(out, pos, results, layers, e1, threshold, labels):
    import matplotlib.pyplot as plt
    norm = _density_norm(layers)
    lam_values = sorted(set(pos[:, 1][np.isfinite(pos[:, 1])]) | {z for z, _g in layers})
    gaps = np.diff(lam_values)
    height = 0.35 * (float(gaps.min()) if gaps.size else 1.0)
    fig, ax = plt.subplots(figsize=(11, 3.5 + 1.2 * len(lam_values)))
    mesh = None
    if norm is not None:
        for (z, _group), dens in layers.items():
            mesh = ax.pcolormesh(e1, [z - height / 2, z + height / 2], _masked(dens, norm)[None, :],
                                 norm=norm, cmap=DENSITY_CMAP, alpha=0.9, shading="flat")
    collections, counts = _edge_collections(results, pos, threshold, three_d=False)
    good_lc = collections[0]
    for lc, count in zip(collections, counts):
        if count:
            ax.add_collection(lc, autolim=False)
    ax.scatter(pos[:, 0], pos[:, 1], s=12, c="black", zorder=3)
    _label_states(ax, pos, three_d=False, fontsize=6)
    ax.set_xlim(e1[0], e1[-1]); ax.set_ylim(min(lam_values) - height, max(lam_values) + height)
    ax.set_xlabel(labels[0]); ax.set_ylabel("GaMD λ")
    fig.colorbar(good_lc, ax=ax, pad=0.01, label="pairwise MBAR overlap √(O_ij·O_ji)")
    if mesh is not None:
        fig.colorbar(mesh, ax=ax, pad=0.02, label="overlap density (per bin)")
    ax.set_title(f"Pairwise MBAR state overlap (red dashed: < {threshold:g}; grey dotted: unmeasured); "
                 "strips: overlap integrand")
    fig.tight_layout(); fig.savefig(out / "overlap_matrix.png", dpi=200); plt.close(fig)


def _plot_layer_panels(out, pos, lam, layers, e1, e2, labels):
    import matplotlib.pyplot as plt
    norm = _density_norm(layers)
    keys = list(layers)
    ncols = min(4, max(1, len(keys)))
    nrows = int(math.ceil(len(keys) / ncols)) or 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.2 * nrows), squeeze=False,
                             gridspec_kw={"hspace": 0.45, "wspace": 0.3})
    rung_levels = sorted(set(lam))
    mesh = None
    for ax, key in zip(axes.flat, keys):
        z, group = key
        dens = layers[key]
        on_rung = group == SAME_RUNG
        base = z if on_rung else max((r for r in rung_levels if r < z), default=z)
        shown = np.flatnonzero(lam == base)
        if e2 is not None:
            if norm is not None:
                mesh = ax.pcolormesh(e1, e2, _masked(dens, norm).T, norm=norm, cmap=DENSITY_CMAP, shading="flat")
            ax.scatter(pos[shown, 0], pos[shown, 1], s=8, c="black")
            for w in shown:
                ax.annotate(str(w), (pos[w, 0], pos[w, 1]), fontsize=5, xytext=(2, 2), textcoords="offset points")
            ax.set_ylabel(labels[1])
        else:
            ax.plot(0.5 * (e1[:-1] + e1[1:]), dens, color="#6a3d9a")
            for w in shown:
                ax.axvline(pos[w, 0], color="0.7", lw=0.6)
                ax.annotate(str(w), (pos[w, 0], 0.0), fontsize=5, xytext=(1, 2), textcoords="offset points")
            ax.set_ylabel("overlap density")
        ax.set_xlabel(labels[0])
        ax.set_title(f"λ = {z:g}" + ("  (same-rung pairs)" if on_rung else "  (rung pairs, mid-λ)"), fontsize=9)
    for ax in list(axes.flat)[len(keys):]:
        ax.axis("off")
    if mesh is not None:
        fig.colorbar(mesh, ax=axes, shrink=0.8, label="overlap density (per bin)")
    fig.savefig(out / "overlap_density_layers.png", dpi=170, bbox_inches="tight"); plt.close(fig)


def _write_html(out, pos, results, layers, e1, e2, threshold, labels) -> Optional[str]:
    try:
        import plotly.graph_objects as go
    except Exception:
        return None
    x = 0.5 * (e1[:-1] + e1[1:]); y = 0.5 * (e2[:-1] + e2[1:])
    vmax = max((float(np.max(d)) for d in layers.values()), default=0.0)
    traces = []
    if vmax > 0:
        floor = vmax * DENSITY_FLOOR
        for (z, group), dens in layers.items():
            colour = np.where(dens > floor, dens, np.nan)
            traces.append(go.Surface(x=x, y=y, z=np.full((y.size, x.size), z), surfacecolor=colour.T,
                                     cmin=0.0, cmax=vmax, colorscale="YlOrRd", opacity=0.55,
                                     showscale=(len(traces) == 0),
                                     colorbar=dict(title="overlap density", x=1.08), name=f"λ={z:g} ({group})",
                                     hovertemplate=f"λ={z:g} ({group})<br>density=%{{surfacecolor:.3g}}<extra></extra>"))
    classes = _edge_classes(results, threshold)
    styles = (None, dict(color="#d62728", width=3, dash="dash"), dict(color="#8c8c8c", width=2, dash="dot"))
    names = ("overlap edges", f"< {threshold:g}", "unmeasured")
    for rows, style, label in zip(classes, styles, names):
        xs, ys, zs, cs, texts = [], [], [], [], []
        for r in rows:
            ov = r["overlap"]
            for w in (r["i"], r["j"]):
                xs.append(pos[w, 0]); ys.append(pos[w, 1]); zs.append(pos[w, 2]); cs.append(ov if math.isfinite(ov) else 0.0)
                texts.append(f"{r['i']}–{r['j']} ({r['kind']}): {ov:.3g}")
            xs.append(None); ys.append(None); zs.append(None); cs.append(0.0); texts.append("")
        if not texts:
            continue
        line = style or dict(color=cs, colorscale="Viridis", cmin=0.0, cmax=EDGE_VMAX, width=4,
                             colorbar=dict(title="pairwise overlap", x=1.0))
        traces.append(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=line, text=texts, hoverinfo="text",
                                   name=label))
    traces.append(go.Scatter3d(x=pos[:, 0], y=pos[:, 1], z=pos[:, 2], mode="markers+text",
                               text=[str(w) for w in range(len(pos))], textfont=dict(size=9),
                               marker=dict(size=3, color="black"), name="states"))
    fig = go.Figure(traces)
    fig.update_layout(scene=dict(xaxis_title=labels[0], yaxis_title=labels[1], zaxis_title="GaMD λ"),
                      title="Pairwise MBAR state overlap", legend=dict(x=0.0, y=1.0))
    path = out / "overlap_graph_3d.html"
    fig.write_html(str(path), include_plotlyjs=True)
    return str(path)


def _indistinguishable_states(centers, sec, lam) -> int:
    """How many states sit at a layout position another state also occupies."""
    keys = [(round(float(c), 6), round(float(s), 6) if sec is not None and math.isfinite(s) else None, float(l))
            for c, s, l in zip(centers, sec if sec is not None else [math.nan] * len(centers), lam)
            if math.isfinite(c)]
    counts: dict = {}
    for k in keys:
        counts[k] = counts.get(k, 0) + 1
    return sum(n for n in counts.values() if n > 1)


def render_overlap_graph(
    out: Path, *, u_nk, window, f_k, n_k, cv1, cv2, centers, secondary_centers,
    lambdas: Optional[Sequence[float]], k1, k2, temperature_k: float, threshold: float,
    bins: int, cv1_label: str, cv2_label: str, write_html: bool = True,
) -> dict:
    """Compute the pairwise overlap graph and write its figures/CSV. Never raises on bad layout input:
    returns ``{'available': False, 'reason': ...}`` instead."""
    out = Path(out)
    remove_overlap_graph_outputs(out)
    centers = np.asarray(centers, dtype=float)
    K = centers.size
    sec = np.asarray(secondary_centers, dtype=float) if secondary_centers is not None else np.full(K, np.nan)
    if not np.any(np.isfinite(centers)):
        return {"available": False, "reason": "no finite CV1 centres in the window table"}
    lam = np.asarray(lambdas, dtype=float) if lambdas is not None else np.zeros(K)
    lam = np.round(np.where(np.isfinite(lam), lam, 0.0), _LAMBDA_DECIMALS)   # one lambda spelling everywhere
    cv1 = np.asarray(cv1, dtype=float)
    cv2_arr = np.asarray(cv2, dtype=float) if cv2 is not None else None
    has_cv2 = (cv2_arr is not None and np.any(np.isfinite(sec))
               and float(np.mean(np.isfinite(cv2_arr))) >= 0.5)

    shared = _indistinguishable_states(centers, sec if has_cv2 else None, lam)
    if shared:
        if np.any(np.isfinite(sec)):
            cause = "identical states (same restraints and rung), e.g. a ladder analysed with its rungs collapsed"
        else:
            cause = "secondary centres missing from the window table?"
        return {"available": False,
                "reason": f"{shared} states share a (CV1 centre{', CV2 centre' if has_cv2 else ''}, lambda) "
                          f"position: {cause}"}
    pairs = graph_pairs(centers, sec if has_cv2 else np.full(K, np.nan), lam, k1,
                        k2 if has_cv2 else [math.nan] * K, temperature_k)
    if not pairs:
        return {"available": False, "reason": "no neighbour pairs in the state layout"}
    e1 = _grid_edges(cv1, centers, bins)
    e2 = _grid_edges(cv2_arr, sec, bins) if has_cv2 else None
    results = pair_overlap_densities(u_nk, window, f_k, n_k, cv1, cv2_arr if has_cv2 else None,
                                     pairs, e1, e2)
    for r in results:
        r["layer"] = layer_value(lam[r["i"]], lam[r["j"]])
    write_pairs_csv(results, out / "overlap_pairs_mbar.csv", threshold)

    layers = _layers(results)
    files = {"overlap_pairs_mbar": str(out / "overlap_pairs_mbar.csv")}
    import matplotlib.pyplot as plt  # noqa: F401  (fail here, not half-way through a figure)
    if has_cv2:
        pos = np.column_stack([centers, sec, lam])
        _plot_3d(out, pos, results, layers, e1, e2, threshold, (cv1_label, cv2_label))
    else:
        pos = np.column_stack([centers, lam])
        _plot_2d(out, pos, results, layers, e1, threshold, (cv1_label, cv2_label))
    files["overlap_matrix"] = str(out / "overlap_matrix.png")
    _plot_layer_panels(out, np.column_stack([centers, sec]) if has_cv2 else np.column_stack([centers, lam]),
                       lam, layers, e1, e2, (cv1_label, cv2_label))
    files["overlap_density_layers"] = str(out / "overlap_density_layers.png")
    if write_html and has_cv2:
        html = _write_html(out, np.column_stack([centers, sec, lam]), results, layers, e1, e2, threshold,
                           (cv1_label, cv2_label))
        if html:
            files["overlap_graph_3d"] = html

    vals = np.array([r["overlap"] for r in results], dtype=float)
    finite = vals[np.isfinite(vals)]
    return {
        "available": True, "files": files, "n_pairs": len(results),
        "n_rung_pairs": sum(r["kind"] == PAIR_KIND_RUNG for r in results),
        "n_gap_pairs": sum(r["kind"] == PAIR_KIND_GAP for r in results),
        "n_below_threshold": int(np.sum(finite < threshold)), "n_unmeasured": int(vals.size - finite.size),
        "min_overlap": float(finite.min()) if finite.size else None,
        "median_overlap": float(np.median(finite)) if finite.size else None,
        "threshold": float(threshold), "dimensions": "cv1_cv2_lambda" if has_cv2 else "cv1_lambda",
    }


__all__ = ["OUTPUT_FILES", "remove_overlap_graph_outputs", "render_overlap_graph"]
