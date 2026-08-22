# GENPEPT seeding

GENPEPT provides seed conformers for CV-aware window starts. One YAML can drive both commands when it has conformer-generation and GAREUS sections.

```bash
python GENPEPT.py --config examples/chignolin_2d_distance_with_genpept.yaml
gareus --config examples/chignolin_2d_distance_with_genpept.yaml \
  --seed-conformers-dir chignolin_genpept_seeds
```

GAREUS scores survivors against active CV1/CV2 windows. Inspect `us_starting_structures/seed_selection_report.*`; broad seed diversity does not guarantee each restrained 2D target has support.
