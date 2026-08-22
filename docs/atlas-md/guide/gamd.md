# GaMD

GaMD modes are `gamd` and `hmr-gamd`. They require external `gamd-openmm`. CMD modes omit this dependency and are useful baseline/control workflows.

```bash
gareus --seq CLN025 --run-mode hmr-gamd --window-mode adaptive --out gamd_run
```

Treat GaMD boost diagnostics and reweighting quality as acceptance checks. Cumulant expansion can converge numerically while sampling remains poor. Compare unbiased support, boost distribution, overlap, and weighted ESS before reporting PMF differences.

Use same configuration except `--run-mode` for CMD-versus-GaMD control comparisons when scientific question requires it.
