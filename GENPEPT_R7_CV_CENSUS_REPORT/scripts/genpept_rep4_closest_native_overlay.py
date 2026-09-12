from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import mdtraj as md
from PIL import Image, ImageDraw

BASE = Path("/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler")
R4 = BASE / "chignolin_genpept_rep4_tuned"
REPORT = BASE / "GENPEPT_R7_CV_CENSUS_REPORT"
FIG = REPORT / "figures"
WORK = Path("/tmp/opencode/overlay_seeds")
WORK.mkdir(parents=True, exist_ok=True)
N_TOP = 6
PYMOL_PNGS = [WORK / f"panel_{k}.png" for k in range(N_TOP)]
STITCHED = FIG / "cv_rep4_closest_to_native.png"


def main():
    native = md.load("/tmp/opencode/1UAO.pdb")
    files = sorted((R4 / "final_implicit_survivor_seeds").glob("*.pdb"))
    print(f"loading {len(files)} seeds...", flush=True)
    traj = md.load(files)
    ca = [a.index for a in traj.topology.atoms if a.name == "CA"]
    ca_ref = [a.index for a in native.topology.atoms if a.name == "CA"]
    rmsd_stack = np.stack([md.rmsd(traj, native, frame=m, atom_indices=ca, ref_atom_indices=ca_ref)
                           for m in range(native.n_frames)], axis=1) * 10.0
    rmsd_min = rmsd_stack.min(axis=1)
    best_model = rmsd_stack.argmin(axis=1)
    order = np.argsort(rmsd_min)

    panel_pdb = []
    for k, idx in enumerate(order[:N_TOP]):
        m = best_model[idx]
        t = traj[idx]
        nat_m = native[m]
        t.superpose(nat_m, 0, atom_indices=ca, ref_atom_indices=ca_ref)
        sp = WORK / f"seed_{k}.pdb"
        np_ = WORK / f"native_{k}.pdb"
        t.save_pdb(str(sp))
        nat_m.save_pdb(str(np_))
        panel_pdb.append({"k": k, "seed_pdb": str(sp), "native_pdb": str(np_),
                           "rmsd": float(rmsd_min[idx]), "model_1uao": int(m + 1),
                           "seed": files[idx].name.replace("_implicit_min", ""),
                           "out_png": str(PYMOL_PNGS[k])})
        print(f"#{k}: {files[idx].name} rmsd {rmsd_min[idx]:.2f} (model {m+1})", flush=True)

    (WORK / "panels.json").write_text(json.dumps(panel_pdb, indent=1))
    subprocess.run([sys.executable.replace("python", "pymol") if False else "pymol",
                   "-cq", str(Path(__file__).with_name("genpept_pymol_overlay.py"))],
                  check=True)

    ims = [Image.open(p) for p in PYMOL_PNGS]
    w, h = ims[0].size
    cols, rows = 3, 2
    pad, thead = 10, 46
    canvas = Image.new("RGB", (cols * w + (cols + 1) * pad,
                               rows * (h + thead) + (rows + 1) * pad), "white")
    draw = ImageDraw.Draw(canvas)
    for k, (im, pmeta) in enumerate(zip(ims, panel_pdb)):
        r, c = divmod(k, cols)
        x = pad + c * (w + pad)
        y = pad + r * (h + thead + pad)
        canvas.paste(im, (x, y))
        draw.text((x + 4, y + h + 8),
                  f"#{pmeta['k']}  {pmeta['rmsd']:.2f} A to 1UAO model {pmeta['model_1uao']}",
                  fill="black")
    canvas.save(STITCHED)
    print("wrote", STITCHED)

    doc = {"n_seeds": int(len(files)), "top": panel_pdb}
    (REPORT / "genpept_rep4_closest_to_native.json").write_text(json.dumps(doc, indent=1))
    lines = ["# rep4: closest seeds vs folded chignolin\n",
             f"Top {N_TOP} rep4 survivors by min CA-RMSD to the 18-model 1UAO ensemble,",
             "each aligned (CA least-squares) onto its closest native model and rendered fully",
             f"atomistically (heavy atoms; red spheres/sticks = native, blue sticks = seed, 45%",
             f"transparent). Figure: `figures/cv_rep4_closest_to_native.png`.\n"]
    for p in panel_pdb:
        lines.append(f"- **{p['rmsd']:.2f} A** (1UAO model {p['model_1uao']}) — `{p['seed']}`")
    lines.append(f"\nBank-wide best: **{rmsd_min.min():.2f} A**; median {np.median(rmsd_min):.2f} A.\n")
    (REPORT / "CLOSEST_SEEDS.md").write_text("\n".join(lines))
    print("wrote CLOSEST_SEEDS.md, json")


if __name__ == "__main__":
    main()
