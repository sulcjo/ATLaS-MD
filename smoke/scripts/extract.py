import numpy as np, mdtraj as md, os, json
OUT="/home/sulcjo/.claude/jobs/b27cb9c3/tmp/results"; os.makedirs(f"{OUT}/structures",exist_ok=True)
TOP="/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production/epoch_000/solute_only.pdb"
z=np.load(f"{OUT}/frame_metrics.npz",allow_pickle=True)
files=[str(x) for x in z["files"]]; r=z["rmsd_bb_min18"]; q=z["q"]
top=md.load(TOP).topology; native=md.load("/home/sulcjo/.claude/jobs/b27cb9c3/tmp/1uao.pdb")
nat_idx={(a.residue.resSeq,a.name):a.index for a in native.topology.atoms}
run_idx={(a.residue.resSeq,a.name):a.index for a in top.atoms}
bb=[k for k in sorted(set(nat_idx)&set(run_idx)) if k[1] in {"N","CA","C","O"} and 2<=k[0]<=9]
bb_run=np.array([run_idx[k] for k in bb]); bb_nat=np.array([nat_idx[k] for k in bb])
# 25 best frames, one per independent visit, superposed on 1UAO model 1 for viewing
order=np.argsort(r); picked=[]; seen=set()
for i in order:
    key=(int(z["src_file"][i]), int(z["src_frame"][i])//60)
    if key in seen: continue
    seen.add(key); picked.append(int(i))
    if len(picked)>=25: break
frames=[]
for i in picked:
    tr=md.load_frame(files[z["src_file"][i]], int(z["src_frame"][i]), top=top)
    tr.superpose(native, frame=0, atom_indices=bb_run, ref_atom_indices=bb_nat)
    frames.append(tr)
out=frames[0]
for t in frames[1:]: out=out.join(t)
out.save(f"{OUT}/structures/native_like_top25.pdb")
native.save(f"{OUT}/structures/1uao_reference.pdb")
meta=[{"rank":n+1,"rmsd_bb_A":float(r[i]),"Q":float(q[i]),"cv1":float(z["cv1"][i]),
       "rg_A":float(z["rg"][i]),"d1_A":float(z["d1"][i]),"d2_A":float(z["d2"][i]),
       "tyr2_trp9_A":float(z["yw"][i]),
       "source":os.path.realpath(files[z["src_file"][i]]),"frame":int(z["src_frame"][i])}
      for n,i in enumerate(picked)]
json.dump(meta,open(f"{OUT}/structures/native_like_top25.json","w"),indent=2)
print(f"wrote {len(frames)} models, rmsd {r[picked].min():.2f}-{r[picked].max():.2f} A")
for m in meta[:10]:
    print(f"  {m['rank']:2d} rmsd {m['rmsd_bb_A']:.2f} Q {m['Q']:.3f} "
          f"{m['source'].split('adaptive_production/')[-1]} frame {m['frame']}")
