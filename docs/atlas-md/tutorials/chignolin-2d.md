# Tutorial: seeded 2D Chignolin

Goal: generate seed conformers then start 2D distance × Ramachandran sampling.

```bash
python GENPEPT.py --config examples/chignolin_2d_distance_with_genpept.yaml
gareus --config examples/chignolin_2d_distance_with_genpept.yaml
```

Review seed-selection report and explicit 2D window table. Diagnose sampling in both coordinates. A regularly spaced target grid does not establish population at every CV1/CV2 intersection.
