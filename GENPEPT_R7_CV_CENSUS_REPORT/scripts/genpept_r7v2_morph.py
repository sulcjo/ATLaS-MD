from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter

from gareus.cv import build_nonlocal_contact_pairs, peptide_residues, secondary_structure_torsions
from gareus.imports import import_openmm
from gareus.tica import backbone_dihedral_features, compute_bootstrap_torsion_pca


from pathlib import Path as _P
root = _P("/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_genpept_r7")
out = _P("/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/GENPEPT_R7_CV_CENSUS_REPORT/figures/r7v2_residual_torsion_cv2_geometric_morph.mp4")
rows = list(csv.DictReader((root / "final_survivor_seeds.csv").open()))
_openmm, app, unit = import_openmm()
positions, topology = [], None
for row in rows:
    path = Path(row["survivor_pdb_path"])
    if not path.is_absolute(): path = root.parent / path
    pdb = app.PDBFile(str(path))
    topology = topology or pdb.topology
    positions.append(np.asarray(pdb.positions.value_in_unit(unit.nanometer), dtype=float))
pos = np.asarray(positions)
args = SimpleNamespace(contact_scheme="atom-pairs", contact_atom_selection="backbone-heavy", contact_min_sequence_separation=4,
    contact_r0_a=10., contact_beta_a_inv=3., contact_normalize=True, contact_pair_warning_threshold=10**9)
pairs = build_nonlocal_contact_pairs(topology, args)
ii, jj, ww = (np.asarray([p[n] for p in pairs], dtype=int if n < 2 else float) for n in range(3))


def primary(x: np.ndarray) -> float:
    d = np.linalg.norm(x[ii]-x[jj], axis=1)
    return float((.5*(1-np.tanh(15*(d-1.0)))*ww).sum()/ww.sum())


heavy = np.asarray([primary(x) for x in pos])
phi, psi = secondary_structure_torsions(topology)
X = np.asarray([backbone_dihedral_features(p, phi, psi) for p in pos])
model = compute_bootstrap_torsion_pca(X, cv1=heavy, residualize=True, component=5,
    phi_torsion_indices=phi, psi_torsion_indices=psi)
cv2 = X @ model.weights + model.offset

target = float(np.median(heavy)); band = .035
pool = np.where(np.abs(heavy-target) <= band)[0]
if pool.size < 20:
    band = .06; pool = np.where(np.abs(heavy-target) <= band)[0]
lo_q, hi_q = np.quantile(cv2[pool], [.03,.97])
low = int(pool[np.argmin(np.where(cv2[pool] <= lo_q, cv2[pool], np.inf))])
high = int(pool[np.argmax(np.where(cv2[pool] >= hi_q, cv2[pool], -np.inf))])

solute = [a.index for r in peptide_residues(topology) for a in r.atoms()]
solute = np.asarray(sorted(solute), dtype=int)
backbone = []
for r in peptide_residues(topology):
    backbone.extend(a.index for a in r.atoms() if a.name in {"N","CA","C"})
backbone = np.asarray(backbone, dtype=int)


def kabsch(mobile: np.ndarray, reference: np.ndarray) -> np.ndarray:
    a, b = mobile[backbone], reference[backbone]
    ca, cb = a.mean(0), b.mean(0)
    u, _, vt = np.linalg.svd((a-ca).T @ (b-cb))
    r = u @ np.diag([1,1,np.linalg.det(u@vt)]) @ vt
    return (mobile-ca) @ r + cb


start = pos[low].copy(); end = kabsch(pos[high], start)
atom_by_index = {a.index: a for a in topology.atoms()}
bonds = [(a.index,b.index) for a,b in topology.bonds() if a.index in set(solute) and b.index in set(solute)]
element_color = {"C":"#bdbdbd","N":"#3182bd","O":"#de2d26","S":"#f1c40f","H":"#ffffff"}
atom_color = [element_color.get(str(getattr(atom_by_index[i].element,"symbol","C")), "#bdbdbd") for i in solute]
xyz0 = np.vstack([start[solute], end[solute]])
center = xyz0.mean(0); lim = float(np.max(np.abs(xyz0-center))) * 1.28

fig = plt.figure(figsize=(8,7), constrained_layout=True)
ax = fig.add_subplot(projection="3d")
ax.set_axis_off(); ax.view_init(elev=18, azim=-63)
frames = 96


def smoothstep(t: float) -> float:
    return t*t*(3-2*t)


def t_for_frame(n: int) -> float:
    if n < 12: return 0.0
    if n >= 84: return 1.0
    return smoothstep((n-12)/71)


traj_pdb = out.with_suffix(".pdb")
with traj_pdb.open("w") as handle:
    app.PDBFile.writeHeader(topology, handle)
    for n in range(96):
        t = t_for_frame(n)
        xyz = (1-t)*start + t*end
        app.PDBFile.writeModel(topology, xyz * unit.nanometer, handle, modelIndex=n+1)
    app.PDBFile.writeFooter(topology, handle)


writer = FFMpegWriter(fps=24, codec="libx264", bitrate=4000, metadata={"title":"Chignolin residual torsion-PCA geometric interpolation (CV1=backbone-heavy r0=10)"})
with writer.saving(fig, str(out), dpi=180):
    for n in range(frames):
        t = t_for_frame(n); xyz = (1-t)*start + t*end
        ax.cla(); ax.set_axis_off(); ax.view_init(elev=18, azim=-63)
        for a,b in bonds:
            p,q=xyz[a],xyz[b]
            ax.plot([p[0],q[0]],[p[1],q[1]],[p[2],q[2]],color="#666666",lw=1.25,zorder=1)
        s=38
        ax.scatter(xyz[solute,0],xyz[solute,1],xyz[solute,2],s=s,c=atom_color,edgecolors="#303030",linewidths=.3,depthshade=True,zorder=2)
        ax.set(xlim=(center[0]-lim,center[0]+lim),ylim=(center[1]-lim,center[1]+lim),zlim=(center[2]-lim,center[2]+lim))
        h=primary(xyz)
        xx=backbone_dihedral_features(xyz,phi,psi); z=float(xx@model.weights+model.offset)
        ax.set_title(f"Backbone-heavy CV1: {h:.3f}    residual torsion-PCA CV2: {z:+.3f}",fontsize=13,pad=12)
        ax.text2D(.02,.02,"Geometric interpolation between r7 seeds — not molecular dynamics",transform=ax.transAxes,fontsize=9)
        ax.text2D(.02,.94,f"CV2 low seed → high seed    {100*t:3.0f}%",transform=ax.transAxes,fontsize=10)
        writer.grab_frame()

summary={"low_index":low,"high_index":high,"cv1_target":target,"cv1_band":band,
 "low_cv1":float(heavy[low]),"high_cv1":float(heavy[high]),"low_cv2":float(cv2[low]),"high_cv2":float(cv2[high]),
 "frames":frames,"fps":24,"trajectory_pdb":str(traj_pdb),"note":"rigidly aligned Cartesian geometric interpolation; not MD"}
out.with_suffix(".json").write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
