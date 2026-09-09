from __future__ import annotations

import glob
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import mdtraj as md
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path("/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler")
REPORT = BASE / "GENPEPT_R7_CV_CENSUS_REPORT"
FIG = REPORT / "figures"
XTC_DIR = BASE / "RUNS" / "chignolin_6" / "adaptive_production" / "epoch_000" / "replica_trajectories"
TOP = "/tmp/opencode/chignolin6_solute_top.pdb"
OUT_MD = REPORT / "DYNAMICS_VALIDATION.md"
OUT_JSON = REPORT / "genpept_r7_dynamics_cv1.json"

DEFS = {
    "bb_r010_b3": ("backbone-heavy", 4, 10.0, 3.0),
    "ca_r010_b3": ("ca", 4, 10.0, 3.0),
    "heavy_r012_b3": ("heavy", 4, 12.0, 3.0),
}
RES3 = ["G1", "Y2", "D3", "P4", "E5", "T6", "G7", "T8", "W9", "G10"]


def pair_lists(top, selection, sep):
    BB = {"N", "CA", "C", "O"}
    idxs = []
    res_of = []
    for a in top.atoms:
        keep = False
        if selection == "heavy":
            keep = a.element is not None and a.element.symbol != "H"
        elif selection == "backbone-heavy":
            keep = a.name in BB
        elif selection == "ca":
            keep = a.name == "CA"
        if keep:
            idxs.append(a.index)
        res_of.append(a.residue.index)
    idxs = np.array(idxs)
    res_of = np.array(res_of)
    ii, jj = np.triu_indices(len(idxs), k=1)
    keep = np.abs(res_of[idxs[ii]] - res_of[idxs[jj]]) >= sep
    return idxs[ii][keep], idxs[jj][keep]


def switch_cv(xyz, ii, jj, r0, beta):
    d = np.linalg.norm(xyz[:, ii] - xyz[:, jj], axis=2) * 10.0
    return (0.5 * (1.0 - np.tanh(0.5 * beta * (d - r0)))).mean(axis=1)


def half_life(x):
    x = x - np.mean(x)
    n = len(x)
    if n < 16 or np.allclose(x, 0):
        return float("nan")
    var = float(np.dot(x, x))
    if var <= 0:
        return float("nan")
    ac = np.correlate(x, x, mode="full")[n - 1:] / var
    below = np.where(ac < 0.5)[0]
    if len(below) == 0:
        return float(n)
    i0 = below[0]
    if i0 == 0:
        return 0.0
    return float(i0 - 1 + (ac[i0 - 1] - 0.5) / (ac[i0 - 1] - ac[i0]))


def main():
    top = md.load_pdb(TOP).topology
    files = sorted(glob.glob(str(XTC_DIR / "*.xtc")))
    pairs = {k: pair_lists(top, *v[:2]) for k, v in DEFS.items()}
    perfile = []
    series = {}
    dt_ps = None
    for f in files:
        t = md.load(f, top=TOP)
        if dt_ps is None:
            dt_ps = float(t.time[1] - t.time[0]) if t.n_frames > 1 else float("nan")
        row = {"file": Path(f).name, "n_frames": t.n_frames}
        for k, (sel, sep, r0, beta) in DEFS.items():
            ii, jj = pairs[k]
            v = switch_cv(t.xyz, ii, jj, r0, beta)
            series.setdefault(k, []).append(v)
            row[f"{k}_mean"] = float(v.mean())
            row[f"{k}_std"] = float(v.std())
            row[f"{k}_minmax"] = float(v.max() - v.min())
            row[f"{k}_halflife_ps"] = half_life(v) * (dt_ps if dt_ps == dt_ps else 1.0)
        perfile.append(row)
    df = pd.DataFrame(perfile)
    print(df.describe().loc[["mean", "50%"], [c for c in df.columns if c != "file"]].to_string())

    census = pd.read_csv(REPORT / "genpept_r7_cv_census_values.csv")
    bank = {k: census[c].to_numpy() for k, c in zip(DEFS, ("c_bb_s4_r010_b3", "c_ca_s4_r010_b3", "c_heavy_s4_r012_b3"))}
    pooled = {k: np.concatenate(v) for k, v in series.items()}

    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    for i, k in enumerate(DEFS):
        ax = axes[0, i]
        edges = np.linspace(min(pooled[k].min(), bank[k].min()),
                            max(pooled[k].max(), bank[k].max()), 61)
        ax.hist(bank[k], bins=edges, density=True, alpha=.5, label="r7 bank", color="#999999")
        ax.hist(pooled[k], bins=edges, density=True, alpha=.5, label="chignolin_6 epoch_000", color="#3182bd")
        ax.set_title(f"{k}  (dyn: range {pooled[k].min():.3f}-{pooled[k].max():.3f}, "
                     f"pooled std {pooled[k].std():.3f})", fontsize=8)
        ax.legend(fontsize=8)
        ax.set_xlabel("normalized contact CV")
        ax2 = axes[1, i]
        hl = df[f"{k}_halflife_ps"].to_numpy(float)
        ax2.hist(hl[np.isfinite(hl)], bins=20, color="#756bb1")
        ax2.set_xlabel("ACF half-time (ps)")
        ax2.set_title(f"{k} per-replica decorrelation\nmedian {np.nanmedian(hl):.1f} ps, "
                      f"max {np.nanmax(hl):.0f} ps", fontsize=8)
        ax2.grid(alpha=.3, axis="y")
    fig.suptitle("CV1 candidates on solvated MD (96 replica trajectories, epoch_000) vs the r7 bank", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIG / "cv_dynamics_cv1_validation.png", dpi=150)
    plt.close(fig)

    bank_cov = {k: float(np.quantile(bank[k], 0.995) - np.quantile(bank[k], 0.005)) for k in bank}
    dyn_cov = {k: float(np.quantile(pooled[k], 0.995) - np.quantile(pooled[k], 0.005)) for k in pooled}
    res = {
        "dt_ps": dt_ps,
        "n_files": len(files),
        "n_frames_total": int(df.n_frames.sum()),
        "coverage_p005_995": {"bank": bank_cov, "dynamics": dyn_cov},
        "pooled_mean_std": {k: [float(pooled[k].mean()), float(pooled[k].std())] for k in pooled},
        "halflife_ps_median": {k: float(np.nanmedian(df[f"{k}_halflife_ps"])) for k in DEFS},
        "halflife_ps_max": {k: float(np.nanmax(df[f"{k}_halflife_ps"])) for k in DEFS},
        "perfile_mean_minmax": {k: float(df[f"{k}_minmax"].mean()) for k in DEFS},
    }
    OUT_JSON.write_text(json.dumps(res, indent=1))

    lines = ["# CV1 candidates on solvated MD (chignolin_6 epoch_000)\n"]
    lines.append(f"{len(files)} replica xtc, {int(df.n_frames.sum())} frames, dt {dt_ps} ps; "
                 f"solute-only (138 atoms). Bank comparison: r7 (1,970).")
    lines.append("\n| def | p0.5–p99.5 bank | dynamics | pooled mean±std | median ACF t½ (ps) | max t½ | mean per-replica range |")
    lines.append("|---|---|---|---|---|---|---|")
    for k in DEFS:
        lines.append(f"| {k} | {bank_cov[k]:.3f} | {dyn_cov[k]:.3f} | {pooled[k].mean():.3f}±{pooled[k].std():.3f} "
                     f"| {res['halflife_ps_median'][k]:.1f} | {res['halflife_ps_max'][k]:.0f} | {res['perfile_mean_minmax'][k]:.3f} |")
    lines.append("")
    OUT_MD.write_text("\n".join(lines))
    print("wrote", OUT_MD, OUT_JSON)


if __name__ == "__main__":
    main()
