from __future__ import annotations

import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np

for p in ("/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler",):
    if p not in sys.path:
        sys.path.insert(0, p)

from gareus.cv import build_nonlocal_contact_pairs, secondary_structure_torsions
from gareus.imports import import_openmm
from gareus.tica import backbone_dihedral_features, compute_bootstrap_torsion_pca

BASE = Path("/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler")
ROOT = BASE / "RUNS" / "chignolin_genpept_r7"
OUT = BASE / "GENPEPT_R7_CV_CENSUS_REPORT" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

rows = list(csv.DictReader((ROOT / "final_survivor_seeds.csv").open()))
_openmm, app, unit = import_openmm()
positions, rg, e2e, topology = [], [], [], None
for row in rows:
    path = Path(row["survivor_pdb_path"])
    if not path.is_absolute():
        path = ROOT.parent / path
    pdb = app.PDBFile(str(path))
    topology = topology or pdb.topology
    positions.append(np.asarray(pdb.positions.value_in_unit(unit.nanometer), dtype=float))
    rg.append(float(row["rg_nm"]))
    e2e.append(float(row["end_to_end_nm"]))
pos = np.asarray(positions)
rg = np.asarray(rg)
e2e = np.asarray(e2e)
print(f"loaded {len(pos)} seeds", flush=True)

xedges = np.linspace(rg.min() - .01, rg.max() + .01, 41)
yedges = np.linspace(e2e.min() - .01, e2e.max() + .01, 41)
counts, _, _ = np.histogram2d(rg, e2e, bins=(xedges, yedges))
fes = np.full_like(counts, np.nan, dtype=float)
mask = counts > 0
fes[mask] = -np.log(counts[mask] / counts[mask].max())
xc = (xedges[:-1] + xedges[1:]) / 2
yc = (yedges[:-1] + yedges[1:]) / 2

pair_cache: dict = {}
dist_cache: dict = {}


def pairs_for(selection: str, sep: int):
    key = (selection, sep)
    if key not in pair_cache:
        args = SimpleNamespace(contact_scheme="atom-pairs", contact_atom_selection=SEL_GAREUS[selection],
                               contact_min_sequence_separation=sep, contact_r0_a=10.0,
                               contact_beta_a_inv=3.0, contact_normalize=True,
                               contact_pair_warning_threshold=10 ** 9)
        pairs = build_nonlocal_contact_pairs(topology, args)
        ii = np.asarray([p[0] for p in pairs], dtype=int)
        jj = np.asarray([p[1] for p in pairs], dtype=int)
        ww = np.asarray([p[2] for p in pairs], dtype=float)
        pair_cache[key] = (ii, jj, ww)
        dist_cache[key] = np.linalg.norm(pos[:, ii] - pos[:, jj], axis=2)
        print("pairs", key, len(ii), flush=True)
    return pair_cache[key], dist_cache[key]


def cv_vals(selection: str, sep: int, r0: float, beta: float) -> np.ndarray:
    (ii, jj, ww), d = pairs_for(selection, sep)
    return (0.5 * (1.0 - np.tanh(0.5 * beta * 10.0 * (d - r0 * .1))) * ww).sum(axis=1) / ww.sum()


SEL_LABEL = {"heavy": "all-heavy", "bb": "backbone-heavy", "ca": "CA-only"}
SEL_GAREUS = {"heavy": "heavy", "bb": "backbone-heavy", "ca": "ca"}


def fig_beta_pseudofes():
    betas = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
    panel_cv = [("heavy", 4, 12.0), ("bb", 4, 10.0), ("ca", 4, 10.0)]
    fig, axes = plt.subplots(len(panel_cv), 2, figsize=(13, 4.3 * len(panel_cv)), constrained_layout=True)
    colors_bl = plt.cm.viridis(np.linspace(0.08, 0.93, len(betas)))
    for r_i, (sel, sep, r0) in enumerate(panel_cv):
        ax, bx = axes[r_i]
        vals = {}
        for beta in betas:
            vals[beta] = cv_vals(sel, sep, r0, beta)
        for beta, color in zip(betas, colors_bl):
            v = vals[beta]
            edges = np.linspace(0.0, 1.0, 81)
            cnt, _ = np.histogram(v, bins=edges)
            centers = 0.5 * (edges[:-1] + edges[1:])
            valid = cnt > 0
            pseudo = -np.log(cnt[valid] / cnt[valid].max())
            ax.plot(centers[valid], pseudo, lw=2, color=color, label=f"β={beta:g} Å⁻¹")
        ax.set(xlim=(0, 1), ylim=(5.2, 0), xlabel=f"normalized {SEL_LABEL[sel]} contact CV",
               ylabel="seed-density pseudo-FES")
        ax.set_title(f"{SEL_LABEL[sel]} contacts, r≥{sep}, r₀={r0:g} Å")
        ax.legend(ncol=2, fontsize=8, frameon=False)
        if r_i == 0:
            ax.text(0.02, 0.03, "Search-bank density; not equilibrium free energy",
                    transform=ax.transAxes, fontsize=8, va="bottom")
        stds = [vals[b].std() for b in betas]
        iqrs = [np.quantile(vals[b], .75) - np.quantile(vals[b], .25) for b in betas]
        span = [np.quantile(vals[b], .95) - np.quantile(vals[b], .05) for b in betas]
        bx.plot(betas, stds, "o-", lw=2, label="std")
        bx.plot(betas, iqrs, "s-", lw=2, label="IQR")
        bx.plot(betas, span, "^-", lw=2, label="p05–p95 span")
        bx.set(xscale="log", xlabel="switch steepness β (Å⁻¹)", ylabel="normalized-CV spread",
               title="spread vs β")
        bx.set_xticks(betas, labels=[str(x) for x in betas])
        bx.grid(alpha=.25)
        if r_i == 0:
            bx.legend(frameon=False)
    p = OUT / "r7v2_contact_beta_pseudofes.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    print(p)


def fig_combo_maps():
    for sel in ("bb", "ca"):
        fig, axes = plt.subplots(3, 6, figsize=(17, 8.7), sharex=True, sharey=True, constrained_layout=True)
        for row_i, sep in enumerate((2, 3, 4)):
            for col_i, mid in enumerate((2, 4, 6, 8, 10, 12)):
                ax = axes[row_i, col_i]
                v = cv_vals(sel, sep, mid, 3.0)
                ax.pcolormesh(xedges, yedges, fes.T, cmap="Greys_r", shading="auto", vmin=0, vmax=5)
                ax.contour(xc, yc, np.ma.masked_invalid(fes).T, levels=(1, 2, 3),
                           colors="white", linewidths=.45, alpha=.8)
                h = ax.hexbin(rg, e2e, C=v, reduce_C_function=np.mean, gridsize=17, mincnt=1,
                              cmap="viridis", vmin=0, vmax=1, edgecolors="none")
                ax.set_title(f"r≥{sep}, r₀={mid} Å\nσ={v.std():.3f}", fontsize=9)
                if col_i == 0:
                    ax.set_ylabel("end-to-end (nm)")
                if row_i == 2:
                    ax.set_xlabel("Rg (nm)")
        cbar = fig.colorbar(h, ax=axes, shrink=.82, pad=.012)
        cbar.set_label(f"mean normalized {SEL_LABEL[sel]} contact CV\nβ=3 Å⁻¹")
        fig.suptitle(f"r7 GENPEPT map (v2): {SEL_LABEL[sel]} contacts; grey contours = seed-density pseudo-FES",
                     fontsize=15)
        p = OUT / f"r7v2_combo_map_{sel}_beta3.png"
        fig.savefig(p, dpi=200)
        plt.close(fig)
        print(p)


def fig_high_midpoints():
    betas = (1.5, 3, 4, 6, 8, 10)
    mids = (6, 7, 8, 9, 10, 11, 12)
    for sel in ("bb", "ca"):
        fig, axes = plt.subplots(len(betas), len(mids), figsize=(17, 13), sharex=True, sharey=True,
                                 constrained_layout=True)
        for i, beta in enumerate(betas):
            for j, mid in enumerate(mids):
                ax = axes[i, j]
                v = cv_vals(sel, 4, mid, beta)
                ax.pcolormesh(xedges, yedges, fes.T, cmap="Greys_r", shading="auto", vmin=0, vmax=5)
                ax.contour(xc, yc, np.ma.masked_invalid(fes).T, levels=(1, 2, 3),
                           colors="white", linewidths=.35, alpha=.7)
                h = ax.hexbin(rg, e2e, C=v, reduce_C_function=np.mean, gridsize=16, mincnt=1,
                              cmap="viridis", vmin=0, vmax=1, edgecolors="none")
                ax.set_title(f"β={beta}, r₀={mid}\nσ={v.std():.3f}", fontsize=7.5)
                if j == 0:
                    ax.set_ylabel("E2E (nm)", fontsize=8)
                if i == len(betas) - 1:
                    ax.set_xlabel("Rg (nm)", fontsize=8)
        c = fig.colorbar(h, ax=axes, shrink=.78, pad=.01)
        c.set_label(f"mean normalized {SEL_LABEL[sel]} contact CV; r≥4")
        fig.suptitle(f"r7 v2: {SEL_LABEL[sel]} contacts. Grey contours: seed-density pseudo-FES", fontsize=15)
        p = OUT / f"r7v2_midpoint_scan_{sel}_r4.png"
        fig.savefig(p, dpi=200)
        plt.close(fig)
        print(p)


def fig_colored_fes_maps():
    for sel, r0 in (("bb", 10.0), ("ca", 10.0)):
        v = cv_vals(sel, 4, r0, 3.0)
        fig, ax = plt.subplots(figsize=(8.2, 6.6), constrained_layout=True)
        mesh = ax.pcolormesh(xedges, yedges, fes.T, cmap="Greys_r", shading="auto", vmin=0, vmax=5)
        c0 = fig.colorbar(mesh, ax=ax, pad=.02)
        c0.set_label("GENPEPT seed-density pseudo-FES, −ln(count / peak)")
        ax.contour(xc, yc, np.ma.masked_invalid(fes).T, levels=(.5, 1, 2, 3, 4),
                   colors="white", linewidths=.8, alpha=.8)
        hx = ax.hexbin(rg, e2e, C=v, reduce_C_function=np.mean, gridsize=28, mincnt=1,
                       cmap="viridis", vmin=0, vmax=1, edgecolors="none", alpha=.95)
        c1 = fig.colorbar(hx, ax=ax, pad=.10)
        c1.set_label(f"mean normalized {SEL_LABEL[sel]} contact CV\nβ=3 Å⁻¹, r₀={r0:g} Å, r≥4")
        ax.set(xlabel="radius of gyration (nm)", ylabel="end-to-end distance (nm)",
               title=f"r7 v2 GENPEPT map colored by candidate CV1 ({SEL_LABEL[sel]}, r₀={r0:g})")
        ax.text(.02, .02, "Hexagons: mean CV. Grey/contours: seed-density pseudo-FES.\nSearch bank only; not equilibrium FES.",
                transform=ax.transAxes, fontsize=8, va="bottom", color="black",
                bbox={"facecolor": "white", "alpha": .75, "edgecolor": "none"})
        p = OUT / f"r7v2_pseudofes_cv_{sel}_beta3_r0_10_r4.png"
        fig.savefig(p, dpi=200)
        plt.close(fig)
        print(p)


def residual_model(cv1: np.ndarray):
    phi, psi = secondary_structure_torsions(topology)
    X = np.asarray([backbone_dihedral_features(p, phi, psi) for p in pos])
    model = compute_bootstrap_torsion_pca(X, cv1=cv1, residualize=True, component=5,
                                          phi_torsion_indices=phi, psi_torsion_indices=psi)
    return X, model


def fig_residual_pca_map():
    cv1 = cv_vals("bb", 4, 10.0, 3.0)
    X, model = residual_model(cv1)
    cv2 = X @ model.weights + model.offset
    lim = float(np.quantile(np.abs(cv2), .98))
    fig, ax = plt.subplots(figsize=(8.2, 6.6), constrained_layout=True)
    mesh = ax.pcolormesh(xedges, yedges, fes.T, cmap="Greys_r", shading="auto", vmin=0, vmax=5)
    c0 = fig.colorbar(mesh, ax=ax, pad=.02)
    c0.set_label("GENPEPT seed-density pseudo-FES, −ln(count / peak)")
    ax.contour(xc, yc, np.ma.masked_invalid(fes).T, levels=(.5, 1, 2, 3, 4),
               colors="white", linewidths=.8, alpha=.8)
    h = ax.hexbin(rg, e2e, C=cv2, reduce_C_function=np.mean, gridsize=28, mincnt=1,
                  cmap="coolwarm", norm=colors.TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim),
                  edgecolors="none")
    c1 = fig.colorbar(h, ax=ax, pad=.10)
    c1.set_label("mean residual torsion-PCA CV2\n(residualized on backbone-heavy r₀=10, β=3, r≥4; component=5)")
    ax.set(xlabel="radius of gyration (nm)", ylabel="end-to-end distance (nm)",
           title="r7 v2: seed-density pseudo-FES colored by residual torsion-PCA CV2")
    ax.text(.02, .02, "Hexagons: mean CV2. Grey/contours: seed-density pseudo-FES.\nSearch bank only; not equilibrium FES.",
            transform=ax.transAxes, fontsize=8, va="bottom", color="black",
            bbox={"facecolor": "white", "alpha": .75, "edgecolor": "none"})
    p = OUT / "r7v2_pseudofes_residual_torsion_pca_cv2.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    print(p)


fig_beta_pseudofes()
fig_combo_maps()
fig_high_midpoints()
fig_colored_fes_maps()
fig_residual_pca_map()
print("done")
