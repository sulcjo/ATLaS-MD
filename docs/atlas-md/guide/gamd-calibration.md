# GaMD calibration

GaMD parameters belong to sampling protocol. Calibration estimates potential-energy envelope per boost group, calculates harmonic boost parameters, then checks those parameters under boosted dynamics.

## Why pooled group calibration

`lower-dual-nonbonded-dihedral` is default. It boosts physical NonBonded and Dihedral groups while leaving umbrella restraint unboosted. This keeps boost definition compatible across windows and avoids using a window-dependent total potential in MBAR/REUS workflow.

For each group, ATLAS-MD pools window statistics: `Vmax=max(Vmax,i)`, `Vmin=min(Vmin,i)`, weighted mean `Vavg=ΣNᵢVavg,i/ΣNᵢ`, and pooled sample standard deviation:

\[
\sigma_V=\sqrt{\frac{\sum_i[(N_i-1)\sigma_i^2+N_i(\mu_i-\mu)^2]}{\sum_iN_i-1}}.
\]

Pooling is deliberate. Per-window boosts with incompatible envelopes distort cross-window comparison. Check calibration artifact for group names, sample counts, envelope, threshold, `k0`, and convergence trace.

## Threshold formulas

For lower-bound boost, code uses:

\[
k'_0=\frac{\sigma_0}{\sigma_V}\frac{V_{max}-V_{min}}{V_{max}-V_{avg}},\quad
k_0=\min(1,k'_0),\quad E=V_{max},\quad k=\frac{k_0}{V_{max}-V_{min}}.
\]

For upper-bound boost, candidate is:

\[
k''_0=\left(1-\frac{\sigma_0}{\sigma_V}\right)\frac{V_{max}-V_{min}}{V_{avg}-V_{min}}.
\]

If `k''₀` is outside `(0,1)`, code falls back to lower-bound calculation. Degenerate envelopes—negligible `σV`, range, or required denominator—disable boost (`k0=0`). This is safety behavior, not calibrated GaMD.

## Self-consistent boosted reconstruction

CMD reconstruction seeds envelope. `--gamd-recon-boosted-iters` then runs boosted reconstruction, remeasures `σV`, recomputes calibration, and stops when every group changes by less than `--gamd-recon-boosted-tol`:

\[
|\sigma_{V,t+1}-\sigma_{V,t}|/\sigma_{V,t}<\mathrm{tol}.
\]

`--gamd-recon-boosted-iters 0` keeps legacy CMD-only calibration. Record this choice. More calibration steps reduce estimator noise; they do not establish PMF convergence.

## Acceptance checks

1. All intended groups have finite, non-degenerate envelope statistics.
2. Joint boosted reconstruction reaches tolerance, or report non-convergence.
3. Group boost distribution and per-window ranges are compatible; inspect warnings.
4. Final production uses recorded frozen calibration.
5. PMF still passes full overlap, base ESS, and reweighting checks.

Calibration controls boost construction. It never validates sampling coverage or reweighting. See [GaMD](gamd.md) and [reweighting](../analysis/reweighting.md).
