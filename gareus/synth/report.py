"""Optional matplotlib plots for a synthetic adaptive campaign.

Headless (Agg backend). matplotlib is an optional dependency; importing this
module without it raises, so callers should import lazily and guard.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .landscapes import Landscape  # noqa: E402
from .oracle import reference_pmf, low_f_mask  # noqa: E402


def _recovered_fes_2d(landscape, samples_by_window, windows, res=80, beta=1.0):
    """MBAR-reweighted 2D free energy from per-window samples. Returns (c1,c2,F_est,mask)."""
    from .metrics import _mbar_weights
    pooled, log_w = _mbar_weights(windows, samples_by_window, beta=beta)
    lo1, hi1 = landscape.cv1_bounds
    lo2, hi2 = landscape.cv2_bounds
    b1 = np.linspace(lo1, hi1, res + 1)
    b2 = np.linspace(lo2, hi2, res + 1)
    if pooled.shape[0] == 0:
        empty = np.full((res, res), np.nan)
        return 0.5 * (b1[:-1] + b1[1:]), 0.5 * (b2[:-1] + b2[1:]), empty, np.zeros((res, res), bool)
    w = np.exp(log_w)
    dens, _, _ = np.histogram2d(pooled[:, 0], pooled[:, 1], bins=[b1, b2], weights=w)
    raw, _, _ = np.histogram2d(pooled[:, 0], pooled[:, 1], bins=[b1, b2])
    explored = raw > 0
    f = np.full_like(dens, np.nan)
    pos = dens > 0
    f[pos] = -np.log(dens[pos])
    if np.any(pos):
        f[pos] -= np.nanmin(f[pos])
    f[~explored] = np.nan
    return 0.5 * (b1[:-1] + b1[1:]), 0.5 * (b2[:-1] + b2[1:]), f, explored


def plot_cv_space(landscape: Landscape, samples_by_window: dict, windows: dict,
                  out_path, *, res: int = 100, title: str = None) -> Path:
    """2x2 CV1xCV2 panel: true FES + windows, explored density, recovered FES, error.

    ``samples_by_window`` / ``windows`` are dicts keyed by window index
    (windows[i] is a Window). Returns the written PNG path.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    c1, c2, f = landscape.grid(res=res)               # true FES, min 0 (kBT)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.4))
    name = title or landscape.name

    # pooled samples + window centers
    pooled = [np.asarray(s, float) for s in samples_by_window.values() if s is not None and len(s)]
    allpts = np.vstack(pooled) if pooled else np.empty((0, 2))
    wc1 = [float(w.center1) for w in windows.values()]
    wc2 = [float(w.center2 if w.center2 is not None else 0.0) for w in windows.values()]

    # (a) true FES + window placement
    ax = axes[0, 0]
    cf = ax.contourf(c1, c2, f.T, levels=24, cmap="viridis")
    fig.colorbar(cf, ax=ax, label="F (kBT)")
    ax.scatter(wc1, wc2, s=40, c="white", edgecolors="k", linewidths=0.6, marker="o", label="windows")
    for (b1, b2) in getattr(landscape, "basins", ()):
        ax.plot(b1, b2, "r*", ms=11, mec="k", mew=0.5)
    ax.set_title("True FES + window placement", fontsize=10, fontweight="bold")
    ax.set_xlabel("CV1 (contact fraction)"); ax.set_ylabel("CV2 (rama-map)")
    ax.legend(loc="upper right", fontsize=7)

    # (b) explored CV space (sample density) + low-F oracle contour
    ax = axes[0, 1]
    lo1, hi1 = landscape.cv1_bounds; lo2, hi2 = landscape.cv2_bounds
    if allpts.shape[0]:
        hb = ax.hexbin(allpts[:, 0], allpts[:, 1], gridsize=42, cmap="magma",
                       extent=(lo1, hi1, lo2, hi2), mincnt=1)
        fig.colorbar(hb, ax=ax, label="samples / bin")
    _, _, mask = low_f_mask(landscape, res=res, threshold_kbt=5.0)
    ax.contour(c1, c2, mask.T.astype(float), levels=[0.5], colors="cyan", linewidths=1.4)
    ax.set_xlim(lo1, hi1); ax.set_ylim(lo2, hi2)
    ax.set_title("Explored CV space (cyan = low-F region to cover)", fontsize=10, fontweight="bold")
    ax.set_xlabel("CV1 (contact fraction)"); ax.set_ylabel("CV2 (rama-map)")

    # (c) MBAR-recovered 2D FES
    ax = axes[1, 0]
    rc1, rc2, fest, explored = _recovered_fes_2d(landscape, samples_by_window, windows, res=80)
    vmax = float(np.nanmax(fest)) if np.any(np.isfinite(fest)) else 1.0
    cf3 = ax.contourf(rc1, rc2, np.ma.masked_invalid(fest.T), levels=24, cmap="viridis", vmin=0, vmax=vmax)
    fig.colorbar(cf3, ax=ax, label="F_est (kBT)")
    ax.set_title("MBAR-recovered FES (explored region)", fontsize=10, fontweight="bold")
    ax.set_xlabel("CV1 (contact fraction)"); ax.set_ylabel("CV2 (rama-map)")

    # (d) recovered - true error map
    ax = axes[1, 1]
    ftrue_i = np.interp(rc1, c1, np.zeros_like(c1))  # placeholder; do 2D interp below
    # interpolate true F onto the recovered grid
    from numpy import interp as _np_interp  # noqa
    gi1 = np.clip(np.searchsorted(c1, rc1) - 0, 0, len(c1) - 1)
    gi2 = np.clip(np.searchsorted(c2, rc2) - 0, 0, len(c2) - 1)
    ftrue = f[np.ix_(gi1, gi2)]
    err = np.where(explored, fest - ftrue, np.nan)
    if np.any(np.isfinite(err)):
        err -= np.nanmean(err[np.isfinite(err)])    # PMFs defined up to a constant
    amax = float(np.nanmax(np.abs(err))) if np.any(np.isfinite(err)) else 1.0
    cf4 = ax.pcolormesh(rc1, rc2, np.ma.masked_invalid(err.T), cmap="RdBu_r", vmin=-amax, vmax=amax)
    fig.colorbar(cf4, ax=ax, label="recovered - true (kBT)")
    ax.set_title("Recovery error (explored region)", fontsize=10, fontweight="bold")
    ax.set_xlabel("CV1 (contact fraction)"); ax.set_ylabel("CV2 (rama-map)")

    fig.suptitle(f"{name}: CV1xCV2 FES and space exploration", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=115); plt.close(fig)
    return out_path


def plot_campaign(landscape: Landscape, records, out_dir) -> list:
    """Write campaign plots; return the list of written PNG paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    # (1) window-center trajectory over the F(cv1,cv2) contour
    c1, c2, f = landscape.grid(res=120)
    fig, ax = plt.subplots(figsize=(6, 5))
    cs = ax.contourf(c1, c2, f.T, levels=20, cmap="viridis")
    fig.colorbar(cs, ax=ax, label="F (kBT)")
    for r, rec in enumerate(records):
        xs = sorted(set(round(float(x), 4) for x in rec.centers1))
        ys = [r] * len(xs)  # round index encoded by marker color shade
        ax.scatter(xs, [0.95 - 0.0 * y for y in ys], s=18,
                   c=[plt.cm.autumn(r / max(1, len(records) - 1))] * len(xs),
                   edgecolors="k", linewidths=0.3, label=f"round {r}" if r in (0, len(records) - 1) else None)
    ax.set_xlabel("CV1 (contact fraction)")
    ax.set_ylabel("CV2 (rama-map)")
    ax.set_title(f"{landscape.name}: window CV1 centers over rounds")
    ax.legend(loc="upper right", fontsize=7)
    p1 = out_dir / f"{landscape.name}_window_trajectory.png"
    fig.tight_layout(); fig.savefig(p1, dpi=110); plt.close(fig)
    written.append(p1)

    # (2) final neighbor-overlap bar vs target line
    from .oracle import grid_overlap
    from .sampler import Window
    last = records[-1]
    centers = sorted(set(round(float(x), 4) for x in last.centers1))
    if len(centers) >= 2:
        ov = [grid_overlap(landscape, Window(centers[i], 50.0),
                           Window(centers[i + 1], 50.0), res=120)
              for i in range(len(centers) - 1)]
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.bar(range(len(ov)), ov, color="steelblue")
        ax.axhline(0.30, color="crimson", ls="--", label="target 0.30")
        ax.set_xlabel("neighbor pair index")
        ax.set_ylabel("grid overlap")
        ax.set_title(f"{landscape.name}: final neighbor overlap")
        ax.legend(fontsize=8)
        p2 = out_dir / f"{landscape.name}_final_overlap.png"
        fig.tight_layout(); fig.savefig(p2, dpi=110); plt.close(fig)
        written.append(p2)

    # (3) reference PMF along CV1
    x, pmf = reference_pmf(landscape, axis="cv1", res=120)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(x, pmf, color="black", label="true PMF (CV1)")
    for c in centers:
        ax.axvline(c, color="seagreen", alpha=0.4, lw=1)
    ax.set_xlabel("CV1 (contact fraction)")
    ax.set_ylabel("PMF (kBT)")
    ax.set_title(f"{landscape.name}: true CV1 PMF + final window centers")
    ax.legend(fontsize=8)
    p3 = out_dir / f"{landscape.name}_pmf_cv1.png"
    fig.tight_layout(); fig.savefig(p3, dpi=110); plt.close(fig)
    written.append(p3)

    return written
