# GaMD

GaMD modes are `gamd` and `hmr-gamd`. They require external `gamd-openmm`. CMD modes omit this dependency and are useful baseline/control workflows.

```bash
gareus --seq CLN025 --run-mode hmr-gamd --window-mode adaptive --out gamd_run
```

Treat GaMD boost diagnostics and reweighting quality as acceptance checks. Cumulant expansion can converge numerically while sampling remains poor. Compare unbiased support, boost distribution, overlap, and weighted ESS before reporting PMF differences.

GaMD samples `U' = U + U_bias - ΔV`. Target weights therefore contain `exp(βΔV)` after umbrella MBAR weights. Direct exponential correction is exact in principle but can be dominated by rare large boosts. Default `gamd_cumulant2` approximates each-bin log correction as `βμ + β²σ²/2`; `gamd_cumulant3` adds `β³κ₃/6` for skew. Both need finite boost values and acceptable distribution diagnostics. See [reweighting](../analysis/reweighting.md).

Do not compare CMD `umbrella_only` and GaMD-cumulant PMFs as interchangeable controls without reporting estimator, boost diagnostics, analyzed population, and uncertainty. PMF smoothness is not reweighting validity.

Use same configuration except `--run-mode` for CMD-versus-GaMD control comparisons when scientific question requires it.
