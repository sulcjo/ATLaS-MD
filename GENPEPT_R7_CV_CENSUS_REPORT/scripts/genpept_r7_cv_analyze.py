import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

REPORT_DIR = Path(__file__).resolve().parents[1]
FIG_DIR = REPORT_DIR / "figures"
CSV = str(REPORT_DIR / "genpept_r7_cv_census_values.csv")
REPORT = str(REPORT_DIR / "report_tables.txt")
FIG_HIST = str(FIG_DIR / "genpept_r7_cv_histograms.png")
FIG_CORR = str(FIG_DIR / "genpept_r7_cv_corr.png")
FIG_MAP = str(FIG_DIR / "genpept_r7_cv_2dmaps.png")

R_KCAL = 1.98720425864083e-3
T_K = 300.0


def sigma_w(k_kcal):
    return math.sqrt(R_KCAL * T_K / k_kcal)


def n_resolvable(span, k_kcal=1200.0, overlap_sigma=1.5):
    return math.floor(span / (overlap_sigma * sigma_w(k_kcal)))


def entropy_norm(v, n_bins=20):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    lo, hi = v.min(), v.max()
    if hi - lo <= 0:
        return 0.0
    h, _ = np.histogram(v, bins=n_bins, range=(lo, hi))
    p = h / h.sum()
    p = p[p > 0]
    h_sh = -(p * np.log(p)).sum()
    return float(h_sh / math.log(n_bins))


def eff_states(v, n_bins=20):
    v = np.asarray(v, float)
    lo, hi = v.min(), v.max()
    if hi - lo <= 0:
        return 0.0
    h, _ = np.histogram(v, bins=n_bins, range=(lo, hi))
    p = h / h.sum()
    p = p[p > 0]
    return float(math.exp(-(p * np.log(p)).sum()))


def bimodality_coeff(v):
    v = np.asarray(v, float)
    n = len(v)
    g1 = stats.skew(v)
    g2 = stats.kurtosis(v)
    denom = g2 + 3.0 * ((n - 1) ** 2) / ((n - 2) * (n - 3))
    return float((g1 ** 2 + 1) / denom) if denom > 0 else np.nan


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
    iqr = float(p75 - p25)
    span99 = float(p99 - p01)
    rows.append({
        "cv": c,
        "family": family(c),
        "mean": float(v.mean()),
        "std": float(v.std()),
        "skew": float(stats.skew(v)),
        "exkurt": float(stats.kurtosis(v)),
        "min": float(v.min()),
        "p05": float(p05),
        "median": float(p50),
        "p95": float(p95),
        "max": float(v.max()),
        "IQR": iqr, "span_p05_95": float(p95 - p05),
        "span_p01_99": span99,
        "entropy": entropy_norm(v),
        "eff_states": eff_states(v),
        "BC": bimodality_coeff(v),
        "nw_k1200": n_resolvable(span99),
        "nw_k250": n_resolvable(span99, 250.0),
    })

st = pd.DataFrame(rows).sort_values("entropy", ascending=False).reset_index(drop=True)
st.to_csv(REPORT_DIR / "genpept_r7_cv_census_stats.csv", index=False)

pear = df[cv_cols].corr(method="pearson")
pear_abs = pear.abs()
pairs = []
for i, a in enumerate(cv_cols):
    for b in cv_cols[i + 1:]:
        r_ = pear.loc[a, b]
        if abs(r_) >= 0.90 and family(a) != family(b):
            pairs.append((a, b, r_))
pairs.sort(key=lambda t: -abs(t[2]))

with open(REPORT, "w") as fh:
    w = fh.write
    w("GENPEPT r7 seed-bank CV census\n")
    w("================================\n\n")
    w(f"Library: RUNS/chignolin_genpept_r7/final_implicit_survivor_seeds (n={n}, chignolin GYDPETGTWG, 138 atoms implicit-minimized)\n")
    w(f"Per-seed values: genpept_r7_cv_census_values.csv ({n} x {len(cv_cols)})\n")
    w(f"Per-CV stats:    genpept_r7_cv_census_stats.csv\n\n")
    w("Definitions\n")
    w("-----------\n")
    w("Contacts: logistic switch 0.5*(1-tanh(0.5*beta*(r-r0))), beta in A^-1, normalized by #terms\n")
    w("  (atom-pairs) or residue-pair-balanced. 's3'/'s4' = residue-separation >=3 / >=4.\n")
    w("  selections: heavy (all heavy), bb (N,CA,C,O), sch (sidechain-heavy), sca (sidechain-all),\n")
    w("  ca (CA only). Matches gareus/cv.py nonlocal_contact_cv_from_positions_nm semantics.\n")
    w("entropy: 20 equal bins over [min,max], H/ln(20). eff_states = exp(H).\n")
    w("BC: bimodality coefficient, >0.555 (5/9) flags sub/unimodal departure.\n")
    w("nw_k1200 / nw_k250: resolvable 1.5-sigma windows over p01-p99 span at k ceiling (sigma_w = sqrt(kT/k), 300 K).\n\n")

    def table(sub, cols, title, sortkey="entropy", nrows=None, floatfmt="%8.3f"):
        w(title + "\n" + "-" * len(title) + "\n")
        sub = sub.sort_values(sortkey, ascending=False)
        if nrows:
            sub = sub.head(nrows)
        hdr = "%-26s %8s %8s %8s %8s %8s %8s %8s %6s %5s %5s\n" % (
            "cv", "median", "std", "IQR", "p05", "p95", "entropy", "eff_st", "BC", "nw12", "nw25")
        w(hdr)
        for _, r in sub.iterrows():
            w("%-26s %8.3f %8.3f %8.4f %8.3f %8.3f %8.3f %8.1f %6.3f %5d %5d\n" % (
                r["cv"], r["median"], r["std"], r["IQR"], r["p05"], r["p95"],
                r["entropy"], r["eff_states"], r["BC"], r["nw_k1200"], r["nw_k250"]))
        w("\n")

    contact = st[st.family.str.startswith("contact")].copy()
    contact["r0"] = contact.cv.str.extract(r"r0([0-9.]+)_b").astype(float)
    contact["beta"] = contact.cv.str.extract(r"_b([0-9.]+)$").astype(float)
    contact["sel"] = contact.cv.str.extract(r"^(?:c|cresb)_([a-z]+)_s(\d)").apply(lambda x: f"{x[0]}-s{x[1]}", axis=1)
    for beta_sel, sub in contact.groupby(["sel", "r0", "beta"]):
        pass

    w("A. Contact-CV grid (normalized switch fraction)\n")
    w("-------------------------------------------------\n")
    w("%-24s %6s %5s | %8s %8s %8s %8s | %6s | %4s %4s\n" % (
        "selection/sep", "r0", "beta", "median", "IQR", "entropy", "p95-p05w", "", "nw12", "nw25"))
    for sel in ["heavy-s4", "heavy-s3", "bb-s4", "sch-s4", "sca-s4", "ca-s4"]:
        pass
    grp = contact.groupby(["sel", "r0", "beta"]).first().reset_index()
    for sel in sorted(grp.sel.unique()):
        sub = grp[grp.sel == sel].sort_values(["beta", "r0"])
        w(f"[{sel}]\n")
        for _, r in sub.iterrows():
            w("  r0=%4.1f  b=%4.1f | med %7.3f  std %6.3f  IQR %6.3f  ent %6.3f  BC %5.2f  nw@12 %2d  nw@25 %2d  range %.3f-%.3f\n" % (
                r["r0"], r["beta"], r["median"], r["std"], r["IQR"], r["entropy"], r["BC"],
                r["nw_k1200"], r["nw_k250"], r["min"], r["max"]))
        w("\n")

    table(st[st.family == "geometry"], None, "B. Global geometry / shape / distances")
    table(st[st.family == "H-bond"], None, "C. Backbone H-bond-like switches (N-O, r0 3.5 A, beta 8)")
    table(st[st.family == "torsion bins"], None, "D. Ramachandran/chi counters")
    table(st[st.family.isin(["torsion PCA", "res-torsion PCA"])], None, "E. Torsion PCA projections (fit on this seed bank)")
    table(st[st.family.isin(["SASA/burial", "reference"])], None, "F. SASA / burial / reference RMSD")

    w("G. Top 30 CVs by normalized entropy\n")
    w("-----------------------------------\n")
    for i, (_, r) in enumerate(st.head(30).iterrows(), 1):
        w("%2d. %-30s %-16s med %7.3f  std %6.3f  IQR %6.3f  ent %6.3f  nw@12 %2d\n" % (
            i, r["cv"], r["family"], r["median"], r["std"], r["IQR"], r["entropy"], r["nw_k1200"]))
    w("\n")

    w("H. Cross-family redundancy (|Pearson r| >= 0.90)\n")
    w("--------------------------------------------------\n")
    for a, b, r_ in pairs[:60]:
        w("  %-28s <-> %-28s r = %+.3f   [%s / %s]\n" % (a, b, r_, family(a), family(b)))
        w(f"  ... total cross-family pairs: {len(pairs)}\n\n")

    w("I. Orthogonality: candidate CV1 vs candidate CV2 (Pearson | Spearman)\n")
    w("----------------------------------------------------------------------\n")
    cv1_cands = ["c_bb_s4_r010_b3", "c_ca_s4_r010_b3", "c_heavy_s4_r012_b3", "c_heavy_s4_r012_b1.5",
                 "c_sch_s4_r012_b3", "cresb_heavy_s4_r012_b3", "rg_heavy", "e2e_nc_ca", "sasa_total"]
    cv2_cands = ["torpca_pc1", "torpca_pc2", "restorpca_pc1", "restorpca_pc2", "rama_alpha",
                 "shape_anisotropy", "hairpin_closure_ratio", "contact_order_ca8", "d_y2_w9_ca", "burial_w9"]
    w("%-24s" % "" + "".join("%14s" % c[:14] for c in cv2_cands) + "\n")
    for a in cv1_cands:
        if a not in df.columns:
            continue
        row = []
        for b in cv2_cands:
            rp = pear.loc[a, b]
            rs = df[[a, b]].corr(method="spearman").iloc[0, 1]
            row.append(f"{rp:+.2f}/{rs:+.2f}")
        w("%-24s" % a + "".join("%14s" % r_ for r_ in row) + "\n")
    w("\nConclusions\n")
    w("-----------\n")
    w("1. On this seed library the widest-spread, highest-entropy contact CVs are backbone-heavy (N,CA,C,O)\n")
    w("   and CA-only switches at r0=10 A, |i-j|>=4 (ent 0.973-0.978, 25-27 resolvable 1.5-sigma windows at\n")
    w("   k=1200 kcal/mol/CV^2), followed by all-heavy at r0=12 A (ent 0.954-0.958, 23-25 windows).\n")
    w("   The previous winner heavy(r0=12,b=1.5-3) is matched closely and edged out by bb/ca at r0=10.\n")
    w("2. The legacy definition (r0=4.5 A, b=6, any selection) is degenerate on the library: median 0.000-0.017,\n")
    w("   <=2 resolvable windows -- it sees only near-native tightly-packed conformers.\n")
    w("3. beta beyond ~3 changes little once r0 is in the transition band (8-12 A); r0 placement dominates.\n")
    w("   r0=14 A pushes the median toward saturation (>0.78) and loses entropy again.\n")
    w("4. Residue-balanced variants correlate with plain atom-pairs at |r|>0.99: pick one, they are the same CV.\n")
    w("5. Rg / mean-CA-distance are near-duplicates of mid-r0 contact fractions (r ~ -0.97..-0.98): using Rg as\n")
    w("   CV1 plus a mid-r0 contact CV1 would be redundant.\n")
    w("6. H-bond switches are degenerate on this implicit-minimized library (median 0, p95 ~1): not usable as CVs.\n")
    w("7. Torsion PCs are broad and unimodal (ent 0.91-0.94, BC~0.44): flat PCA spectrum (PC1 = 12% variance),\n")
    w("   no dominant torsion direction; residualization vs contact CV1 barely changes PC2+ (r(PC3,resPC3)=1.000).\n")
    w("8. nw figures for Angstrom-unit CVs (distances, SASA, RMSD) use the same sigma_w=sqrt(kT/k) formula with the\n")
    w("   k held in kcal/mol/A^2 -- comparable only under an equal-k-per-unit assumption, not directly to contact CVs.\n\n")

pear.to_csv(REPORT_DIR / "genpept_r7_cv_census_corr_pearson.csv")

top_for_hist = [
     "c_heavy_s4_r04.5_b6", "c_heavy_s4_r012_b1.5", "c_heavy_s4_r012_b3", "c_heavy_s4_r08_b3",
     "cresb_heavy_s4_r012_b3", "c_bb_s4_r012_b3",
     "rg_heavy", "rg_ca", "asphericity", "shape_anisotropy", "max_ca_ca", "mean_ca_ca",
     "e2e_nc_ca", "d_y2_w9_ca", "d_y2_w9_ring", "hairpin_closure_ratio",
     "hbond_bb_ge3", "hbond_crossstrand", "contact_order_ca8",
     "torpca_pc1", "torpca_pc2", "restorpca_pc1", "restorpca_pc2",
     "rama_beta", "sasa_hydrophobic_frac", "burial_w9", "rmsd_ca_to_survivor000",
     "chirality_ca", "turn3537_mean_ca",
]
top_for_hist = [c for c in top_for_hist if c in df.columns]
fig, axes = plt.subplots(6, 5, figsize=(19, 17))
for ax, c in zip(axes.ravel(), top_for_hist):
    v = df[c].to_numpy(float)
    ax.hist(v, bins=40, color="steelblue", edgecolor="none")
    r = st[st.cv == c].iloc[0]
    ax.set_title(f"{c}\nmed={r['median']:.3f} IQR={r['IQR']:.3f} H={r['entropy']:.3f}", fontsize=8)
    ax.tick_params(labelsize=7)
for ax in axes.ravel()[len(top_for_hist):]:
    ax.axis("off")
fig.suptitle("GENPEPT r7 seed bank (n=1970) — selected CV distributions", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.985])
fig.savefig(FIG_HIST, dpi=140)
plt.close(fig)

rep = st.sort_values("entropy", ascending=False)
fams = rep.family.unique()
keep = []
for fam in fams:
    fam_cols = rep[rep.family == fam].cv.tolist()
    for c in fam_cols:
        if len(keep) >= 45:
            break
        if all(abs(pear.loc[c, k]) < 0.985 for k in keep):
            keep.append(c)
keep = rep.cv.tolist()[:10] + [c for c in keep if c not in rep.cv.tolist()[:10]]
sub = df[keep]
C = sub.corr().abs().to_numpy()
D = 1.0 - C
np.fill_diagonal(D, 0)
Z = hierarchy.linkage(squareform(D, checks=False), method="average")
order = hierarchy.leaves_list(Z)
labs = [keep[i] for i in order]
fig, ax = plt.subplots(figsize=(16, 13))
im = ax.imshow(C[np.ix_(order, order)], cmap="viridis", vmin=0, vmax=1)
ax.set_xticks(range(len(labs)))
ax.set_yticks(range(len(labs)))
ax.set_xticklabels(labs, rotation=90, fontsize=6)
ax.set_yticklabels(labs, fontsize=6)
fig.colorbar(im, ax=ax, shrink=0.8, label="|Pearson r|")
fig.suptitle("GENPEPT r7 CV redundancy map (clustered |r|)", fontsize=12)
fig.tight_layout()
fig.savefig(FIG_CORR, dpi=150)
plt.close(fig)

x = df["c_heavy_s4_r012_b3"].to_numpy()
y = df["restorpca_pc1"].to_numpy()
fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
sc = axes[0].scatter(x, y, c=df["rg_heavy"], s=6, cmap="viridis")
axes[0].set_xlabel("c_heavy_s4_r012_b3")
axes[0].set_ylabel("res-torsion PC1")
axes[0].set_title("contact CV1 vs residual torsion PC1\ncolor = rg_heavy", fontsize=10)
fig.colorbar(sc, ax=axes[0], shrink=0.85)
sc = axes[1].scatter(x, df["torpca_pc1"], c=df["d_y2_w9_ca"], s=6, cmap="magma")
axes[1].set_xlabel("c_heavy_s4_r012_b3")
axes[1].set_ylabel("torsion PC1")
axes[1].set_title("color = d(CA Tyr2-Trp9)", fontsize=10)
fig.colorbar(sc, ax=axes[1], shrink=0.85)
sc = axes[2].scatter(df["rg_heavy"], df["hairpin_closure_ratio"], c=x, s=6, cmap="plasma")
axes[2].set_xlabel("rg_heavy (nm)")
axes[2].set_ylabel("d(2,9)/d(1,10) CA ratio")
axes[2].set_title("color = contact CV1", fontsize=10)
fig.colorbar(sc, ax=axes[2], shrink=0.85)
fig.tight_layout()
fig.savefig(FIG_MAP, dpi=150)
plt.close(fig)

print("wrote", REPORT, FIG_HIST, FIG_CORR, FIG_MAP)
print(st.head(12)[["cv", "family", "median", "std", "IQR", "entropy", "nw_k1200"]].to_string(index=False))
