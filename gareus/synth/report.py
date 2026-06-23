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
from .oracle import reference_pmf  # noqa: E402


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
