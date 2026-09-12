from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

BASE = Path("/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler")
REPORT = BASE / "GENPEPT_R7_CV_CENSUS_REPORT"
FIG = REPORT / "figures"
CSV = REPORT / "genpept_rep3_cv_census_values.csv"
OUT_STATS = REPORT / "genpept_rep3_cv_census_stats.csv"
OUT_MD = REPORT / "REP3_STATS.md"

R_KCAL = 1.98720425864083e-3
T_K = 300.0


def entropy_norm(v, n_bins=20):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if v.max() - v.min() <= 0:
        return 0.0
    h, _ = np.histogram(v, bins=n_bins, range=(v.min(), v.max()))
    p = h / h.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / math.log(n_bins))


def n_resolvable(span, k=1200.0):
    return int(math.floor(span / (1.5 * math.sqrt(R_KCAL * T_K / k))))


def bc(v):
    n = len(v)
    g1 = stats.skew(v)
    g2 = stats.kurtosis(v)
    den = g2 + 3.0 * ((n - 1) ** 2) / ((n - 2) * (n - 3))
    return float((g1 ** 2 + 1) / den) if den > 0 else float("nan")


def family(name):
    if name.startswith("cresb_"):
        return "contact resbal"
    if name.startswith("crat_"):
        return "contact rational"
    if name.startswith("strand_"):
        return "strand packing"
    if name.startswith("pack_"):
        return "strand packing"
    if name.startswith("c_"):
        return "contact"
    if name.startswith("torpca"):
        return "torsion PCA"
    if name.startswith("restorpca"):
        return "res-torsion PCA"
    if name.startswith(("rama", "chi1", "cis")):
        return "torsion bins"
    if name.startswith("hbond"):
        return "H-bond"
    if name.startswith(("sasa", "burial")):
        return "SASA/burial"
    if name.startswith("rmsd"):
        return "reference"
    return "geometry"


df = pd.read_csv(CSV)
n = len(df)
cv_cols = [c for c in df.columns if c != "seed"]
rows = []
for c in cv_cols:
    v = df[c].to_numpy(float)
    p01, p05, p25, p50, p75, p95, p99 = np.percentile(v, [1, 5, 25, 50, 75, 95, 99])
    rows.append({"cv": c, "family": family(c), "mean": float(v.mean()), "std": float(v.std()),
                 "median": float(p50), "IQR": float(p75 - p25), "p05": float(p05), "p95": float(p95),
                 "span_01_99": float(p99 - p01), "entropy": entropy_norm(v), "BC": bc(v),
                 "nw_k1200": n_resolvable(float(p99 - p01))})
st = pd.DataFrame(rows).sort_values("entropy", ascending=False).reset_index(drop=True)
st.to_csv(OUT_STATS, index=False)
print("top 12:")
print(st.head(12)[["cv", "median", "IQR", "entropy", "BC", "nw_k1200"]].to_string(index=False))

pear = df[cv_cols].corr(method="pearson")
CV1_CANDS = ["c_bb_s4_r010_b3", "c_ca_s4_r010_b3", "c_heavy_s4_r012_b3", "c_sch_s4_r012_b3"]
CV2_CANDS = ["restorpca_pc1", "torpca_pc1", "torpca_pc2", "torpca_pc3", "hairpin_closure_ratio",
             "acylindricity", "shape_anisotropy", "chirality_ca", "turn3537_mean_ca", "pack_y2",
             "burial_d3", "burial_w9", "sasa_hydrophobic_frac", "contact_order_ca8", "e2e_nc_ca",
             "strat_col_npcol".replace("strat_col_npcol", "d_y2_w9_ca")]
orth = {}
for a in CV1_CANDS:
    for b in CV2_CANDS:
        orth.setdefault(a, {})[b] = {"pearson": float(pear.loc[a, b]),
                                     "spearman": float(df[[a, b]].corr(method="spearman").iloc[0, 1])}

def joint(x, y, nx=16, ny=8):
    xb = np.quantile(x, np.linspace(0.001, 0.999, nx + 1))
    yb = np.quantile(y, np.linspace(0.001, 0.999, ny + 1))
    H, _, _ = np.histogram2d(x, y, bins=[xb, yb])
    occ = (H > 0).mean()
    iqr_global = np.percentile(y, 75) - np.percentile(y, 25)
    vals = []
    for i in range(nx):
        m = (x >= xb[i]) & (x < xb[i + 1] if i < nx - 1 else x <= xb[i + 1])
        if m.sum() >= 30:
            vals.append(np.percentile(y[m], 75) - np.percentile(y[m], 25))
    vals = np.array(vals)
    return occ, int(H[H > 0].min()), float(np.median(vals / iqr_global)) if len(vals) else 0.0

pair_rows = []
for a in CV1_CANDS:
    for b in CV2_CANDS:
        occ, mn, cond = joint(df[a].to_numpy(), df[b].to_numpy())
        pair_rows.append({"cv1": a, "cv2": b, **orth[a][b], "occ": occ, "min_bin": mn, "cond_iqr": cond})
pairs = pd.DataFrame(pair_rows)

fig, axes = plt.subplots(6, 5, figsize=(19, 17))
top_hist = (st.head(18).cv.tolist()
            + ["hbond_bb_ge3", "hbond_crossstrand", "strand_mindist", "pack_y2", "chirality_ca",
               "strand_cross_far_r012_b3", "torpca_pc2", "restorpca_pc1", "sasa_hydrophobic_frac",
               "d_y2_w9_ring", "turn3537_mean_ca", "rmsd_ca_to_survivor000"])
top_hist = [c for c in dict.fromkeys(top_hist) if c in df.columns][:30]
for ax, c in zip(axes.ravel(), top_hist):
    ax.hist(df[c].to_numpy(), bins=50, color="steelblue", edgecolor="none")
    r = st[st.cv == c].iloc[0]
    ax.set_title(f"{c}\nmed={r['median']:.3f} IQR={r['IQR']:.3f} H={r['entropy']:.3f}", fontsize=7)
    ax.tick_params(labelsize=6)
for ax in axes.ravel()[len(top_hist):]:
    ax.axis("off")
fig.suptitle(f"rep3_massive (n={n}) — CV distributions", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.985])
fig.savefig(FIG / "cv_rep3_histograms.png", dpi=140)
plt.close(fig)

keep = st.cv.tolist()[:55]
sub = df[keep]
C = sub.corr().abs().to_numpy()
Dv = 1.0 - C
np.fill_diagonal(Dv, 0)
Z = hierarchy.linkage(squareform(Dv, checks=False), method="average")
order = hierarchy.leaves_list(Z)
labs = [keep[i] for i in order]
fig, ax = plt.subplots(figsize=(15, 12))
im = ax.imshow(C[np.ix_(order, order)], cmap="viridis", vmin=0, vmax=1)
ax.set_xticks(range(len(labs))); ax.set_yticks(range(len(labs)))
ax.set_xticklabels(labs, rotation=90, fontsize=5.5)
ax.set_yticklabels(labs, fontsize=5.5)
fig.colorbar(im, ax=ax, shrink=.8, label="|Pearson r|")
fig.suptitle("rep3 CV redundancy map (top 55, clustered)", fontsize=11)
fig.tight_layout()
fig.savefig(FIG / "cv_rep3_corr.png", dpi=150)
plt.close(fig)

selmaps = [("c_bb_s4_r010_b3", "torpca_pc2"), ("c_bb_s4_r010_b3", "restorpca_pc1"),
           ("c_heavy_s4_r012_b3", "torpca_pc2"), ("c_ca_s4_r010_b3", "torpca_pc2")]
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
for ax, (a, b) in zip(axes.ravel(), selmaps):
    x = df[a].to_numpy()
    y = df[b].to_numpy()
    ax.hexbin(x, y, gridsize=30, cmap="viridis", mincnt=1)
    r_ = df[[a, b]].corr().iloc[0, 1]
    occ_, mn_, cond_ = joint(x, y)
    ax.set_title(f"{a} x {b}\nr={r_:+.2f}, occ={occ_*100:.0f}%, cond-IQR {cond_:.2f}", fontsize=8)
fig.suptitle("rep3 CV1xCV2 joint maps (16x8 window feasibility)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(FIG / "cv_rep3_joint_maps.png", dpi=150)
plt.close(fig)

lines = [f"# rep3_massive stats summary (n={n}, 298 CVs)\n",
         "## top 15 by entropy"]
lines.append("| # | CV | family | median | IQR | entropy | BC | nw@k1200 |")
lines.append("|---|---|---|---|---|---|---|---|")
for i in range(15):
    r = st.iloc[i]
    lines.append(f"| {i+1} | {r.cv} | {r.family} | {r['median']:.3f} | {r['IQR']:.3f} | {r['entropy']:.3f} | {r['BC']:.3f} | {r.nw_k1200} |")

lines.append("\n## contact grid (entropy by selection/r0/beta at sep 4)")
lines.append("| selection | r0 | beta | median | IQR | entropy | nw |")
lines.append("|---|---|---|---|---|---|---|")
for sel in ("bb", "ca", "heavy", "sch", "sca"):
    for r0, beta in [(4.5, 6.0), (8.0, 3.0), (10.0, 1.5), (10.0, 3.0), (10.0, 6.0), (12.0, 1.5), (12.0, 3.0), (12.0, 6.0), (14.0, 3.0)]:
        cname = f"c_{sel}_s4_r0{r0:g}_b{beta:g}"
        if cname in df.columns:
            r = st[st.cv == cname].iloc[0]
            lines.append(f"| {sel} | {r0:g} | {beta:g} | {r['median']:.3f} | {r['IQR']:.3f} | {r['entropy']:.3f} | {r.nw_k1200} |")

lines.append("\n## orthogonality (Pearson / Spearman) and 16x8 joint-fill for champion CV1s")
for a in CV1_CANDS:
    lines.append(f"\n[{a}]")
    lines.append("| CV2 | r_p | r_s | occ | min_bin | cond_IQR_share |")
    lines.append("|---|---|---|---|---|---|")
    for _, r in pairs[pairs.cv1 == a].sort_values("spearman").iterrows():
        lines.append(f"| {r.cv2} | {r.pearson:+.2f} | {r.spearman:+.2f} | {r.occ*100:.0f}% | {r.min_bin} | {r.cond_iqr:.2f} |")

lines.append("\n## degenerate / notes")
hbc = st[st.family == "H-bond"].iloc[0]
lines.append(f"- H-bond family entropy range: {st[st.family=='H-bond'].entropy.min():.3f}..{st[st.family=='H-bond'].entropy.max():.3f}")
lines.append(f"- cresb_bb == c_bb exactly? max |diff|: {float((df['cresb_bb_s4_r010_b3'] - df['c_bb_s4_r010_b3']).abs().max()):.2e}")
lines.append(f"- legacy def c_heavy_s4_r04.5_b6: median {st[st.cv=='c_heavy_s4_r04.5_b6'].iloc[0]['median']:.4f}, nw {st[st.cv=='c_heavy_s4_r04.5_b6'].iloc[0].nw_k1200}")
lines.append("")
OUT_MD.write_text("\n".join(lines))
pairs.to_csv(REPORT / "genpept_rep3_orthogonal_pairs.csv", index=False)
print("wrote", OUT_STATS, OUT_MD, FIG / "cv_rep3_histograms.png", FIG / "cv_rep3_corr.png", FIG / "cv_rep3_joint_maps.png")
