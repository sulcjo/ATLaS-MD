from __future__ import annotations

import json
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from mdtraj import load as mdload
from openmm import app, unit

BASE = Path("/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler")
R7 = BASE / "RUNS" / "chignolin_genpept_r7"
REPORT = BASE / "GENPEPT_R7_CV_CENSUS_REPORT"
FIG = REPORT / "figures"
AXES = [
    ("torpca_pc2", "plain torsion PC2 (CV2)"),
    ("acylindricity", "acylindricity (shape CV2)"),
    ("chirality_ca", "hairpin handedness (chirality, mA^3)"),
    ("hairpin_closure_ratio", "hairpin closure ratio d(2,9)/d(1,10)"),
]
FRAMES = 96


def render_morph(topology, pos, axis_vals, cv1_vals, tag, label, outbase, fps=24):
    target = float(np.median(cv1_vals))
    band = 0.035
    pool = np.where(np.abs(cv1_vals - target) <= band)[0]
    if pool.size < 20:
        pool = np.where(np.abs(cv1_vals - target) <= 0.06)[0]
    lo_q, hi_q = np.quantile(axis_vals[pool], [.03, .97])
    low = int(pool[np.argmin(np.where(axis_vals[pool] <= lo_q, axis_vals[pool], np.inf))])
    high = int(pool[np.argmax(np.where(axis_vals[pool] >= hi_q, axis_vals[pool], -np.inf))])

    solute = np.array(sorted([a.index for r in topology.residues() if r.name not in ("HOH", "WAT") for a in r.atoms()]))
    bb = np.array(sorted([a.index for r in topology.residues() if r.name not in ("HOH", "WAT")
                          for a in r.atoms() if a.name in {"N", "CA", "C"}]))

    def kabsch(mobile, reference):
        a, b = mobile[bb], reference[bb]
        ca, cb = a.mean(0), b.mean(0)
        u, _, vt = np.linalg.svd((a - ca).T @ (b - cb))
        r = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
        return (mobile - ca) @ r + cb

    start = pos[low].copy()
    end = kabsch(pos[high], start)
    bonds = set()
    for a, b in topology.bonds():
        if a.index in set(solute) and b.index in set(solute):
            bonds.add((a.index, b.index))
    atom_by_index = {a.index: a for a in topology.atoms()}
    colors = {"C": "#bdbdbd", "N": "#3182bd", "O": "#de2d26", "S": "#f1c40f", "H": "#ffffff"}
    atom_color = [colors.get(str(getattr(atom_by_index[i].element, "symbol", "C")), "#bdbdbd") for i in solute]
    xyz0 = np.vstack([start[solute], end[solute]])
    center = xyz0.mean(0)
    lim = float(np.max(np.abs(xyz0 - center))) * 1.28

    def smoothstep(t):
        return t * t * (3 - 2 * t)

    def t_for(n):
        return 0.0 if n < 12 else (1.0 if n >= 84 else smoothstep((n - 12) / 71))

    mp4 = outbase.with_suffix(".mp4")
    pdb_path = outbase.with_suffix(".pdb")
    with pdb_path.open("w") as h:
        app.PDBFile.writeHeader(topology, h)
        for n in range(FRAMES):
            t = t_for(n)
            xyz = (1 - t) * start + t * end
            app.PDBFile.writeModel(topology, xyz * unit.nanometer, h, modelIndex=n + 1)
        app.PDBFile.writeFooter(topology, h)

    fig = plt.figure(figsize=(8, 7), constrained_layout=True)
    ax = fig.add_subplot(projection="3d")
    ax.set_axis_off()
    ax.view_init(elev=18, azim=-63)
    writer = FFMpegWriter(fps=fps, codec="libx264", bitrate=4000,
                         metadata={"title": f"Chignolin morph along {label}"})
    hyper = {}
    def axis_eval(xyz_):
        def tors_feats(x_):
            from mdtraj import compute_phi, compute_psi, Trajectory
            tr = Trajectory(x_[None, :, :], None, coords_used="nanometers")
            ph_ = compute_phi(tr)[1]
            ps_ = compute_psi(tr)[1]
            return np.concatenate([np.sin(ph_.ravel()), np.cos(ph_.ravel()),
                                   np.sin(ps_.ravel()), np.cos(ps_.ravel())])
        v = tors_feats(xyz_) @ hyper["vt"] + hyper["offset"]
        return float(v)

    with writer.saving(fig, str(mp4), dpi=160):
        for n in range(FRAMES):
            t = t_for(n)
            xyz = (1 - t) * start + t * end
            ax.cla()
            ax.set_axis_off()
            ax.view_init(elev=18, azim=-63)
            for a, b in bonds:
                p, q = xyz[a], xyz[b]
                ax.plot([p[0], q[0]], [p[1], q[1]], [p[2], q[2]], color="#666666", lw=1.2, zorder=1)
            ax.scatter(xyz[solute, 0], xyz[solute, 1], xyz[solute, 2], s=38, c=atom_color,
                       edgecolors="#303030", linewidths=.3, depthshade=True, zorder=2)
            ax.set(xlim=(center[0] - lim, center[0] + lim), ylim=(center[1] - lim, center[1] + lim),
                   zlim=(center[2] - lim, center[2] + lim))
            ax.set_title(f"CV1 (bb r0=10 b3): {cv1_vals[low]:.3f} -> {cv1_vals[high]:.3f}   |   {label}: {axis_vals[low]:+.3f} -> {axis_vals[high]:+.3f}   {100*t:3.0f}%",
                         fontsize=11, pad=10)
            ax.text2D(.02, .02, "Geometric interpolation between r7 seeds — not molecular dynamics",
                      transform=ax.transAxes, fontsize=8)
            writer.grab_frame()

    prev = outbase.with_name(outbase.stem + "_preview.png")
    plt_save = plt.figure(figsize=(8, 7))
    axx = plt_save.add_subplot(projection="3d")
    axx.set_axis_off()
    axx.view_init(elev=18, azim=-63)
    t = 0.5
    xyz = (1 - t) * start + t * end
    for a, b in bonds:
        p, q = xyz[a], xyz[b]
        axx.plot([p[0], q[0]], [p[1], q[1]], [p[2], q[2]], color="#666666", lw=1.2)
    axx.scatter(xyz[solute, 0], xyz[solute, 1], xyz[solute, 2], s=38, c=atom_color,
                edgecolors="#303030", linewidths=.3, depthshade=True)
    axx.set(xlim=(center[0] - lim, center[0] + lim), ylim=(center[1] - lim, center[1] + lim),
            zlim=(center[2] - lim, center[2] + lim))
    axx.set_title(label + " (midpoint preview)")
    plt_save.savefig(prev, dpi=160)
    plt.close(plt_save)

    return {"low_index": low, "high_index": high, "cv1_low": float(cv1_vals[low]),
            "cv1_high": float(cv1_vals[high]), "axis_low": float(axis_vals[low]),
            "axis_high": float(axis_vals[high])}


def main():
    files = sorted((R7 / "final_implicit_survivor_seeds").glob("*.pdb"))
    names = [f.stem for f in files]
    census = pd.read_csv(REPORT / "genpept_r7_cv_census_values.csv").set_index("seed").loc[names]
    traj = mdload(files)
    topology = traj.topology.to_openmm()
    pos = traj.xyz
    cv1 = census["c_bb_s4_r010_b3"].to_numpy()
    results = {}
    for tag, label in AXES:
        vals = census[tag].to_numpy()
        out = FIG / f"r7v2_morph_{tag}"
        print(f"render {tag} -> {out}", flush=True)
        results[tag] = render_morph(topology, pos, vals, cv1, tag, label, out)
        print("  ", results[tag], flush=True)
    (FIG / "r7v2_morph_per_cv.json").write_text(json.dumps(results, indent=1))
    print("wrote all", json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
