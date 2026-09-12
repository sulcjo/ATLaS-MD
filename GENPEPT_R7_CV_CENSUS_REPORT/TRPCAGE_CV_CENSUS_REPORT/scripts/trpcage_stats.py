from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

BASE = Path("/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler")
REPORT = BASE / "TRPCAGE_CV_CENSUS_REPORT"
FIG = REPORT / "figures"
VALUES = REPORT / "trpcage_cv_census_values.csv"
OUT_STATS = REPORT / "trpcage_cv_census_stats.csv"
OUT_CORR = REPORT / "trpcage_cv_census_corr_pearson.csv"
OUT_MD = REPORT / "TRPCAGE_STATS.md"
OUT_JSON = REPORT / "trpcage_cv_census_stats.json"


def entropy_norm(v, n_bins=20):
    v = v[np.isfinite(v)]
    if v.max() == v.min():
        return 0.0
    h, _ = np.histogram(v, bins=n_bins, range=(v.min(), v.max()))
    p = h / h.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / math.log(n_bins))


def main():
    df = pd.read_csv(VALUES).set_index("seed")
    n = len(df)
    E = df["energy_kj_mol"].to_numpy()
    cage = df["cage_score"].to_numpy()
    like = df["trpcage_like"].to_numpy()

    rows = []
    for c in df.columns:
        if c in ("energy_kj_mol",):
            continue
        v = df[c].to_numpy()
        if not np.isfinite(v).all() or v.std() == 0:
            continue
        rows.append({
            "cv": c,
            "mean": float(v.mean()), "std": float(v.std()),
            "min": float(v.min()), "max": float(v.max()),
            "iqr": float(np.percentile(v, 75) - np.percentile(v, 25)),
            "entropy": entropy_norm(v),
            "spear_energy": float(spearmanr(v, E).correlation),
            "spear_cage": float(spearmanr(v, cage).correlation),
            "spear_trpcage_like": float(spearmanr(v, like).correlation),
        })
    stats = pd.DataFrame(rows)
    stats.to_csv(OUT_STATS, index=False)

    nums = df.select_dtypes(include=[np.floating]).drop(columns=["energy_kj_mol"])
    keep = [c for c in nums.columns if nums[c].std() > 0 and np.isfinite(nums[c]).all()]
    corr = nums[keep].corr(method="spearman")
    corr.to_csv(OUT_CORR)

    by_ent = stats.sort_values("entropy", ascending=False)
    by_cage = stats.reindex(stats["spear_cage"].abs().sort_values(ascending=False).index)
    by_E = stats.reindex(stats["spear_energy"].abs().sort_values(ascending=False).index)

    fig, axes = plt.subplots(1, 3, figsize=(17, 7))
    t = by_cage.head(20)[::-1]
    axes[0].barh(t["cv"], t["spear_cage"], color="teal")
    axes[0].set_xlabel("Spearman rho vs cage_score")
    axes[0].set_title("top CVs correlated with trp-cage packing", fontsize=9)
    axes[0].tick_params(labelsize=7)
    t = by_E.head(20)[::-1]
    axes[1].barh(t["cv"], t["spear_energy"], color="darkorange")
    axes[1].set_xlabel("Spearman rho vs GBn2 energy")
    axes[1].set_title("top CVs correlated with energy", fontsize=9)
    axes[1].tick_params(labelsize=7)
    t = by_ent.head(20)[::-1]
    axes[2].barh(t["cv"], t["entropy"], color="slateblue")
    axes[2].set_xlabel("normalized 20-bin entropy")
    axes[2].set_title("highest-entropy (broadest) CVs", fontsize=9)
    axes[2].tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "trpcage_stats_topcvs.png", dpi=150)
    plt.close(fig)

    lines = [f"# CV census statistics — trp-cage (TC5b) bank\n"]
    lines.append(f"n={n} survivors; {len(stats)} numeric CVs after dropping zero-variance columns.")
    lines.append("\n## Top CVs by |Spearman| vs cage_score (W6-Pro packing quality)\n")
    lines.append("| cv | rho | entropy | iqr |")
    lines.append("|---|---|---|---|")
    for _, r in by_cage.head(15).iterrows():
        lines.append(f"| {r['cv']} | {r['spear_cage']:+.3f} | {r['entropy']:.3f} | {r['iqr']:.3g} |")
    lines.append("\n## Top CVs by |Spearman| vs GBn2 minimized energy\n")
    lines.append("| cv | rho |")
    lines.append("|---|---|")
    for _, r in by_E.head(15).iterrows():
        lines.append(f"| {r['cv']} | {r['spear_energy']:+.3f} |")
    lines.append("\n## Broadest CVs (entropy)\n")
    lines.append("| cv | entropy |")
    lines.append("|---|---|")
    for _, r in by_ent.head(15).iterrows():
        lines.append(f"| {r['cv']} | {r['entropy']:.3f} |")
    lines.append("\n## Correlation of the fold metric\n")
    for c in ("c_heavy_s4_r012_b3", "rg_heavy", "e2e_nc_ca", "sasa_total", "rama_alpha"):
        if c in stats["cv"].values:
            r = stats[stats["cv"] == c].iloc[0]
            lines.append(f"- {c}: spear_cage {r['spear_cage']:+.3f}, spear_E {r['spear_energy']:+.3f}")
    n_like = int(like.sum())
    lines.append(f"\ntrpcage_like seeds: {n_like} / {n}")
    lines.append("")
    OUT_MD.write_text("\n".join(lines))
    json.dump({"n": n, "trpcage_like": n_like, "top_cage": by_cage.head(15).to_dict("records"),
               "top_energy": by_E.head(15).to_dict("records")},
              open(OUT_JSON, "w"), indent=1)
    print("wrote", OUT_STATS, OUT_CORR, OUT_MD)
    print(f"trpcage_like: {n_like}/{n}")


if __name__ == "__main__":
    main()
