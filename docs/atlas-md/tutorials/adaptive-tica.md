# Tutorial: adaptive torsion PCA to tICA

Goal: use seed-fit `torsion-pca` before trajectory data exists, then switch to learned `tica-linear` only after successful update.

```bash
python GENPEPT.py --config examples/chignolin_runs3.yaml
gareus --config examples/chignolin_runs3.yaml
```

Inspect adaptive metadata and exact torsion state path after update. On resume, verify same readable state is restored. Do not force a switch from labels/centers alone.
