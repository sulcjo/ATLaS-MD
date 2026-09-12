from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import mdtraj as md
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path("/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler")
R4 = BASE / "chignolin_genpept_rep4_tuned"
if not R4.exists():
    R4 = Path("/tmp/opencode/chignolin_genpept_rep4_tuned")
NATIVE = Path("/tmp/opencode/1UAO.pdb")
REPORT = BASE / "GENPEPT_R7_CV_CENSUS_REPORT"
FIG = REPORT / "figures"
OUT_MD = REPORT / "REP4_FES.md"
OUT_JSON = REPORT / "genpept_rep4_fes_folded.json"

BANKS = {"rep4": R4}


def hbond_indices(top):
    n_d3 = [a.index for a in top.residue(2).atoms if a.name == "N"][0]
    o_g7 = [a.index for a in top.residue(6).atoms if a.name == "O"][0]
    o_t8 = [a.index for a in top.residue(7).atoms if a.name == "O"][0]
    return n_d3, o_g7, o_t8


def contact_bb(traj):
    BB = {"N", "CA", "C", "O"}
    idx = np.array([a.index for a in traj.topology.atoms if a.name in BB])
    res_of_arr = np.array([a.residue.index for a in traj.topology.atoms])
    ii, jj = np.triu_indices(len(idx), k=1)
    keep = np.abs(res_of_arr[idx[ii]] - res_of_arr[idx[jj]]) >= 4
    ii, jj = idx[ii][keep], idx[jj][keep]
    d = np.linalg.norm(traj.xyz[:, ii] - traj.xyz[:, jj], axis=2) * 10.0
    return (0.5 * (1.0 - np.tanh(0.5 * 3.0 * (d - 10.0)))).mean(axis=1)


def torsion_pc(traj):
    phis = md.compute_phi(traj)[1]
    psis = md.compute_psi(traj)[1]
    cols = []
    for arr in (phis, psis):
        for k in range(arr.shape[1]):
            cols += [np.sin(arr[:, k]), np.cos(arr[:, k])]
    X = np.stack(cols, axis=1)
    Xc = X - X.mean(0)
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[0], Xc @ Vt[1], (S ** 2) / (S ** 2).sum()


def e2e_nm_a(traj):
    n_term = [a.index for a in traj.topology.residue(0).atoms if a.name == "N"][0]
    c_last = [a.index for a in traj.topology.residue(traj.topology.n_residues - 1).atoms if a.name == "CA"][0]
    return np.linalg.norm(traj.xyz[:, n_term] - traj.xyz[:, c_last], axis=1) * 10.0


def fes2d(x, y, n=60):
    xe = np.linspace(x.min(), x.max(), n + 1)
    ye = np.linspace(y.min(), y.max(), n + 1)
    H, _, _ = np.histogram2d(x, y, bins=[xe, ye])
    F = np.full_like(H, np.nan, dtype=float)
    m = H > 0
    F[m] = -np.log(H[m] / H.max())
    return F, xe, ye, H


def draw_fes(ax, F, xe, ye, label):
    im = ax.pcolormesh(xe, ye, F.T, cmap="viridis_r", vmin=0, vmax=5)
    xc = (xe[:-1] + xe[1:]) / 2
    yc = (ye[:-1] + ye[1:]) / 2
    ax.contour(xc, yc, np.ma.masked_invalid(F).T, levels=(.5, 1, 2, 3, 4), colors="white", linewidths=.6, alpha=.75)
    return im


def main():
    native = md.load(NATIVE)
    n_idx, o7_idx, o8_idx = hbond_indices(native.topology)
    nat_d37 = np.linalg.norm(native.xyz[:, n_idx] - native.xyz[:, o7_idx], axis=1) * 10.0
    nat_d38 = np.linalg.norm(native.xyz[:, n_idx] - native.xyz[:, o8_idx], axis=1) * 10.0
    print(f"1UAO d(D3N,G7O): {nat_d37.min():.2f}..{nat_d37.max():.2f}  d(D3N,T8O): {nat_d38.min():.2f}..{nat_d38.max():.2f}")

    result = {"native": {"n_models": native.n_frames,
                         "d_D3N_G7O_A": [float(v) for v in nat_d37],
                         "d_D3N_T8O_A": [float(v) for v in nat_d38]}, "banks": {}}
    md_lines = ["# rep4 (tuning bundle) folded-state proximity audit + pseudo-FES maps\n",
                f"Reference: PDB 1UAO (chignolin GYDPETGTWG NMR ensemble, {native.n_frames} models). "
                f"Native pair distances: d(D3 N, G7 O) {nat_d37.min():.2f}..{nat_d37.max():.2f} A; "
                f"d(D3 N, T8 O) {nat_d38.min():.2f}..{nat_d38.max():.2f} A.\n"]

    for tag, root in BANKS.items():
        files = sorted((root / "final_implicit_survivor_seeds").glob("*.pdb"))
        traj = md.load(files)
        names = [f.stem for f in files]
        n = len(files)
        print(f"{tag}: {n} seeds", flush=True)

        ca_ref = [a.index for a in native.topology.atoms if a.name == "CA"]
        rmsd_stack = np.stack([md.rmsd(traj, native, frame=m, atom_indices=[a.index for a in traj.topology.atoms if a.name == "CA"], ref_atom_indices=ca_ref) for m in range(native.n_frames)], axis=1) * 10.0
        rmsd_min = rmsd_stack.min(axis=1)
        which_model = rmsd_stack.argmin(axis=1)

        nn_ = hbond_indices(traj.topology)
        d37 = np.linalg.norm(traj.xyz[:, nn_[0], :] - traj.xyz[:, nn_[1], :], axis=1) * 10.0
        d38 = np.linalg.norm(traj.xyz[:, nn_[0], :] - traj.xyz[:, nn_[2], :], axis=1) * 10.0

        cv1 = contact_bb(traj)
        pc1, pc2, evr = torsion_pc(traj)
        e2e = e2e_nm_a(traj)
        rg = md.compute_rg(traj) * 10.0
        dssp = md.compute_dssp(traj)
        cE = (dssp == "E").sum(axis=1)
        cH = (dssp == "H").sum(axis=1)

        fracs = {str(t): float((rmsd_min < t).mean()) for t in (2.0, 2.5, 3.0, 3.5, 4.0)}
        corner4 = ((d37 < 4.0) & (d38 < 4.0))
        corner5 = ((d37 < 5.0) & (d38 < 5.0))
        best = int(np.argmin(rmsd_min))
        result["banks"][tag] = {
            "n": n,
            "rmsd_frac_below": fracs,
            "rmsd_min_to_1UAO_A": float(rmsd_min.min()),
            "rmsd_median_A": float(np.median(rmsd_min)),
            "best_seed": names[best], "best_model_1uao": int(which_model[best]) + 1,
            "best_rmsd_A": float(rmsd_min[best]),
            "best_d37_d38_A": [float(d37[best]), float(d38[best])],
            "d37_min_A": float(d37.min()), "d38_min_A": float(d38.min()),
            "corner_both_lt4": {"count": int(corner4.sum()), "frac": float(corner4.mean())},
            "corner_both_lt5": {"count": int(corner5.sum()), "frac": float(corner5.mean())},
            "dssp_E_residue_mean": float(cE.mean()),
            "dssp_H_residue_mean": float(cH.mean()),
            "evr_plain": evr[:6].tolist(),
        }
        print(f"  RMSD fracs: {fracs}")
        print(f"  corner <4/4: {corner4.sum()}, <5/5: {corner5.sum()}; best seed {names[best]} rmsd {rmsd_min[best]:.2f} (model {which_model[best]+1})")

        md_lines.append(f"\n## {tag}: folded-proximity audit (vs 1UAO)\n")
        md_lines.append(f"- min CA-RMSD to any 1UAO model: **{rmsd_min.min():.2f} A** (seed `{names[best]}`, model {which_model[best]+1})")
        for t in (2.0, 2.5, 3.0, 3.5, 4.0):
            md_lines.append(f"  - CA-RMSD < {t} A: {fracs[str(t)]*100:.2f}% ({fracs[str(t)]*n:.0f} seeds)")
        md_lines.append(f"- median: {np.median(rmsd_min):.2f} A")
        md_lines.append(f"- folded corner (both d37, d38 < 4 A): **{int(corner4.sum())} seeds**; both < 5 A: {int(corner5.sum())}")
        md_lines.append(f"- minima: d37 min {d37.min():.2f} A, d38 min {d38.min():.2f} A")
        md_lines.append(f"- DSSP: mean strand (E) residues/seed {cE.mean():.2f}; helix (H) {cH.mean():.2f}")

        fig, axes = plt.subplots(2, 2, figsize=(13.5, 10))
        ax = axes[0, 0]
        ax.hist(rmsd_min, bins=80, color="#e6550d")
        for t in (2.5, 3.0, 3.5):
            ax.axvline(t, color="red", ls="--", lw=.8)
        ax.set_xlabel("min CA-RMSD to 1UAO ensemble (A)")
        ax.set_ylabel("seeds")
        ax.set_title(f"{tag}: proximity to experiment", fontsize=10)

        F, xe, ye, H = fes2d(d37, d38)
        ax = axes[0, 1]
        draw_fes(ax, F, xe, ye, "native pair")
        ax.scatter(nat_d37, nat_d38, marker="*", s=100, color="red", label="1UAO (18 models)", zorder=5)
        ax.axvspan(0, 4, color="red", alpha=.05)
        ax.axhspan(0, 4, color="red", alpha=.05)
        ax.set_xlabel("d(D3 N, G7 O), A")
        ax.set_ylabel("d(D3 N, T8 O), A")
        ax.set_title("native-pair pseudo-FES (search density)", fontsize=9)
        ax.legend(fontsize=8, loc="upper right")

        ax = axes[1, 0]
        hb = ax.hexbin(cv1, pc1, C=rmsd_min, reduce_C_function=np.min, gridsize=45, cmap="magma_r", mincnt=1)
        fig.colorbar(hb, ax=ax, shrink=.8, label="nearest-seed RMSD in bin (A)")
        ax.set_xlabel("bb r0=10 b3 contact CV1")
        ax.set_ylabel("bank plain torsion PC1")
        ax.set_title("where the near-native seeds sit in working CV space", fontsize=9)

        ax = axes[1, 1]
        ax.hist(rmsd_min[cE == 0], bins=60, alpha=.6, label=f"DSSP E=0 (n={int((cE==0).sum())})", color="#999999")
        if (cE > 0).any():
            ax.hist(rmsd_min[cE > 0], bins=60, alpha=.7, label=f"DSSP E>0 (n={int((cE>0).sum())})", color="#e6550d")
        ax.set_xlabel("min RMSD to 1UAO (A)")
        ax.legend(fontsize=8)
        ax.set_title("does strand assignment predict proximity?", fontsize=9)
        fig.suptitle(f"folded-state proximity audit — {tag} ({n} seeds)", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.975])
        fig.savefig(FIG / f"cv_{tag}_folded_proximity.png", dpi=150)
        plt.close(fig)

        fig2, axes = plt.subplots(1, 3, figsize=(17, 4.9))
        maps = [(cv1, pc2, "bb r0=10 b3 (CV1)", "bank plain torsion PC2 (CV2)"),
                (rg, e2e, "Rg (A)", "N terminus to C-alpha-10 (A)"),
                (cv1, pc1, "bb r0=10 b3 (CV1)", "bank plain torsion PC1")]
        for ax, (x, y, xl, yl) in zip(axes, maps):
            F, xe, ye, H = fes2d(x, y, n=60)
            draw_fes(ax, F, xe, ye, xl)
            ax.set_xlabel(xl, fontsize=9)
            ax.set_ylabel(yl, fontsize=9)
            ax.set_title(f"occupancy {float((H > 0).mean())*100:.1f}% of grid", fontsize=9)
        fig2.suptitle(f"rep4 pseudo-FES maps ({n} seeds, search density - not equilibrium FES)", fontsize=11)
        fig2.tight_layout(rect=[0, 0, 1, 0.97])
        fig2.savefig(FIG / "cv_rep4_pseudofes.png", dpi=150)
        plt.close(fig2)
        occ = float((H > 0).mean())
        md_lines.append("\n## rep4 pseudo-FES")
        md_lines.append(f"- grid occupancy on CV1xPC1 map: {occ*100:.1f}%")
        md_lines.append("- figures: cv_rep4_pseudofes.png (3 maps), cv_rep4_folded_proximity.png")

    OUT_JSON.write_text(json.dumps(result, indent=1))
    md_lines.append("")
    OUT_MD.write_text("\n".join(md_lines))
    print("wrote", OUT_MD, OUT_JSON)


if __name__ == "__main__":
    main()
