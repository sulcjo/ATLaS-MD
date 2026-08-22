# System setup

`gareus` builds peptide from `--seq` or accepts `--input-pdb` verbatim for capped/non-standard structures. Default solvent uses explicit water, PME, NPT production, dodecahedral box, 1.0 nm padding, 300 K, and 0.15 M ionic strength.

```bash
gareus --seq GYDPETGTWG --water-model tip3pfb --box-shape dodecahedron \
  --padding-nm 1.0 --temperature-k 300 --out chignolin
```

Use `--input-pdb` when topology must be preserved. Relative PDB paths resolve from launch directory. Check `box_audit.json`, setup structures, and resolved config before production.

HMR modes (`hmr-cmd`, `hmr-gamd`) default to 4 fs timestep; CMD/GaMD modes default to 2 fs. Do not change timestep independently without validating constraints and integration stability.
