from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, gaussian_kde
from scipy.signal import find_peaks

BASE = Path("/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler")
REPORT = BASE / "TRPCAGE_CV_CENSUS_REPORT"
FIG = REPORT / "figures"
OUT_MD = REPORT / "TRPCAGE_MI_STATES_ENERGY.md"
OUT_JSON = REPORT / "trpcage_mi_states_energy.json"


def kraskov_mi(x, y, k=3):
    from scipy.spatial import cKDTree
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    zx = (x - x.mean()) / x.std()
    zy = (y - y.mean()) / y.std()
    data = np.stack([zx, zy], axis=1)
    tree = cKDTree(data, leafsize=16)
    dist, _ = tree.query(data, k + 1, p=np.inf)
    eps = dist[:, -1] + 1e-15
    tx = cKDTree(zx.reshape(-1, 1), leafsize=16)
    ty = cKDTree(zy.reshape(-1, 1), leafsize=16)
    nx = np.empty(n)
    ny = np.empty(n)
    for i in range(n):
        nx[i] = len(tx.query_ball_point(zx[i], eps[i], p=np.inf)) - 1
        ny[i] = len(ty.query_ball_point(zy[i], eps[i], p=np.inf)) - 1
    from scipy.special import digamma
    return float(digamma(k) - (digamma(nx + 1) + digamma(ny + 1)).mean() + digamma(n))


def entropy_norm(v, n_bins=20):
    v = v[np.isfinite(v)]
    h, _ = np.histogram(v, bins=n_bins, range=(v.min(), v.max()))
    p = h / h.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / math.log(n_bins))


def main():
    census = pd.read_csv(REPORT / "trpcage_cv_census_values.csv").set_index("seed")
    n = len(census)

    cv1n = "c_bb_s4_r010_b3"
    cv1 = census[cv1n].to_numpy()
    cv2_cands = ["restorpca_pc1", "torpca_pc2", "torpca_pc1", "cage_score", "cage_w6_pros_r06_b4",
                 "dssp_helix_1_9", "rama_alpha_1_9", "pack_w6", "burial_w6",
                 "acylindricity", "chirality_ca", "contact_order_ca8", "rg_heavy"]
    mi_rows = []
    rng = np.random.default_rng(19)
    for c in cv2_cands:
        y = census[c].to_numpy()
        mi = kraskov_mi(cv1, y)
        mi_null = np.mean([kraskov_mi(cv1, y[rng.permutation(n)]) for _ in range(5)])
        rs = float(spearmanr(cv1, y).correlation)
        mi_rows.append({"cv2": c, "mi_nats": mi, "mi_null_mean": mi_null, "mi_excess": mi - mi_null,
                        "spearman": rs})
        print(f"MI({cv1n},{c}) = {mi:.3f} (null {mi_null:.3f}), excess {mi - mi_null:+.3f}, rho {rs:+.3f}")

    E = census["energy_kj_mol"].to_numpy()
    for c in ("cage_score", "dssp_helix_1_9", "rg_heavy", "c_heavy_s4_r012_b3", "restorpca_pc1"):
        rs = float(spearmanr(census[c].to_numpy(), E).correlation)
        print(f"Spearman(E, {c}) = {rs:+.3f}")

    cv2 = census["restorpca_pc1"].to_numpy()
    xs = np.linspace(cv2.min(), cv2.max(), 400)
    kde = gaussian_kde(cv2, bw_method=0.06)
    dens = kde(xs)
    peaks, props = find_peaks(dens, prominence=0.04 * dens.max())
    mins, _ = find_peaks(-dens, prominence=0.04 * dens.max())
    bounds = np.concatenate([[xs[0]], xs[mins], [xs[-1]]])
    states = []
    for i in range(len(bounds) - 1):
        m = (cv2 >= bounds[i]) & (cv2 <= bounds[i + 1])
        if m.sum() < 15:
            continue
        states.append({"i": i, "bounds": (float(bounds[i]), float(bounds[i + 1])),
                       "n": int(m.sum()),
                       "E_mean": float(E[m].mean()), "E_std": float(E[m].std()),
                       "E_min": float(E[m].min()),
                       "cage_score_mean": float(census["cage_score"].to_numpy()[m].mean()),
                       "helix19_mean": float(census["dssp_helix_1_9"].to_numpy()[m].mean()),
                       "trpcage_frac": float(census["trpcage_like"].to_numpy()[m].mean()),
                       "rg_mean": float(census["rg_heavy"].to_numpy()[m].mean())})
    print(f"states from restorpca_pc1 KDE: {len(states)}")

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    axes[0].bar([r["cv2"] for r in mi_rows], [r["mi_excess"] for r in mi_rows], color="steelblue")
    axes[0].set_ylabel("MI excess over null (nats)")
    axes[0].set_title(f"MI({cv1n}, CV2)", fontsize=9)
    axes[0].tick_params(axis="x", rotation=60, labelsize=7)
    axes[1].plot(xs, dens)
    axes[1].plot(xs[peaks], dens[peaks], "rv")
    axes[1].set_xlabel("restorpca_pc1"); axes[1].set_ylabel("KDE density")
    axes[1].set_title(f"state decomposition ({len(states)} states)", fontsize=9)
    axes[2].scatter(cv2, E, s=2, alpha=.15, c=census["cage_score"].to_numpy(), cmap="viridis")
    axes[2].set_xlabel("restorpca_pc1"); axes[2].set_ylabel("GBn2 energy (kJ/mol)")
    axes[2].set_title("energy vs residual torsion PC1 (color = cage_score)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "trpcage_mi_states_energy.png", dpi=150)
    plt.close(fig)

    lines = ["# MI, states and energy — trp-cage (TC5b) bank\n"]
    lines.append(f"n={n} seeds. CV1 = {cv1n}.\n")
    lines.append("## MI (Kraskov k=3, 5-permutation null) CV1 vs CV2 candidates\n")
    lines.append("| CV2 | MI nats | null | excess | Spearman |")
    lines.append("|---|---|---|---|---|")
    for r in sorted(mi_rows, key=lambda r: -r["mi_excess"]):
        lines.append(f"| {r['cv2']} | {r['mi_nats']:.3f} | {r['mi_null_mean']:.3f} | "
                     f"{r['mi_excess']:+.3f} | {r['spearman']:+.3f} |")
    lines.append("\n## States from restorpca_pc1 KDE (bw 0.06, 4% prominence)\n")
    lines.append("| state | n | E_mean | E_min | cage_score | helix(1-9) | trpcage% | Rg A |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for s in states:
        lines.append(f"| {s['i']} | {s['n']} | {s['E_mean']:.0f} | {s['E_min']:.0f} | "
                     f"{s['cage_score_mean']:.3f} | {s['helix19_mean']:.1f} | "
                     f"{100*s['trpcage_frac']:.1f}% | {s['rg_mean']:.1f} |")
    lines.append("\n## Energy correlations (Spearman)\n")
    for c in ("cage_score", "dssp_helix_1_9", "rg_heavy", "c_heavy_s4_r012_b3", "restorpca_pc1"):
        rs = float(spearmanr(census[c].to_numpy(), E).correlation)
        lines.append(f"- E vs {c}: rho = {rs:+.3f}")
    lines.append("")
    OUT_MD.write_text("\n".join(lines))
    json.dump({"mi": mi_rows, "states": states}, open(OUT_JSON, "w"), indent=1)
    print("wrote", OUT_MD, OUT_JSON)


if __name__ == "__main__":
    main()
