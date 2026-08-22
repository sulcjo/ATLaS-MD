# Umbrella windows and REUS

Choose `manual` for supplied centers, `adaptive` for calibrated primary CV range, `adaptive-feedback` for pilot refinement, `adaptive-production` for epoch-based pool allocation, or `double-adaptive` for both feedback and production stages.

```bash
gareus --seq CLN025 --window-mode adaptive --cv1 distance \
  --cv1-target-spacing 0.2 --out distance_windows
```

Window force constants use kcal/mol/CV². Primary spacing-derived constants use `--cv1-adaptive-overlap-sigma`; CV2 may use fixed, spacing, or adaptive values. Inspect `umbrella_windows.csv` and `umbrella_explicit_windows.csv` rather than assuming generated grid is physically covered.

`--exchange-mode` supports `neighbor`, `random-pair`, `all-pair-sweep`, and `gibbs-walk`. Exchanges improve movement between restraints; they do not repair missing 2D overlap.
