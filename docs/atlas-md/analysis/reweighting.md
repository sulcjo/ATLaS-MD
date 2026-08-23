# Reweighting and free-energy estimators

Reweighting transforms biased samples into target equilibrium estimates. It cannot create support missing from sampled ensemble.

## Definitions

`n` indexes sample and `k` window. `N_k` is sample count in state `k`; `u_nk` is dimensionless reduced umbrella bias of sample `n` in window `k`:

\[
u_{nk}=\beta\,4.184\,[\tfrac12K_{1,k}(\xi_{1,n}-c_{1,k})^2+\tfrac12K_{2,k}(\xi_{2,n}-c_{2,k})^2].
\]

Second term exists only for finite CV2 center and positive CV2 constant. A sample with unavailable CV2 receives `NaN` for CV2-restrained windows; analysis does not invent zero displacement.

## MBAR umbrella weights

MBAR solves window offsets `f_k` self-consistently. For populated states, normalized unbiased sample weight is proportional to:

\[
w_n=\left[\sum_k N_k\exp(f_k-u_{nk})\right]^{-1},\qquad \sum_nw_n=1.
\]

`f_k` itself has arbitrary common offset. Empty states are omitted from solver algebra, but remain reported as coverage failures. Omitting them prevents numerical singularity; it adds no physical samples or overlap.

For bin `b`, weighted probability is `P_b = Σ_{n∈b}w_n`; PMF is:

\[
F_b=-k_BT\ln P_b+C,
\]

with `C` selected so minimum finite PMF is zero. Zero weighted probability gives infinite PMF, not high-confidence barrier.

## GaMD correction

With boost `ΔV_n` in kJ/mol, direct exponential estimator uses:

\[
w_n^{exp}\propto w_n\exp(\beta\Delta V_n).
\]

Direct exponential weights often collapse. ATLAS-MD therefore writes it as diagnostic/comparison output; it is not default plot unless selected.

For each PMF bin, cumulant estimators approximate conditional factor `⟨exp(βΔV)⟩_b`. Cumulant-2 uses:

\[
\ln r_b\approx \beta\mu_b+\tfrac12\beta^2\sigma_b^2,
\qquad P_b^{C2}\propto P_b^{umbrella}r_b.
\]

`μ_b` and `σ²_b` are MBAR-weighted boost mean and variance. Cumulant-3 adds `(β³/6)κ_{3,b}`, where `κ₃` is weighted third central moment. Cumulant expansion is approximation, not proof: compare estimator behavior and inspect boost anharmonicity. A sampled bin lacking finite boost gets `NaN` correction, never silent zero correction.

## Diagnostics before interpretation

| Quantity | Formula / meaning | Failure meaning |
| --- | --- | --- |
| Base ESS | `1 / Σ_n w_n²` | Umbrella-unbiased estimate dominated by few frames |
| Reweight ESS | `(Σ_n q_n)² / Σ_n q_n²`, `q_n=w_n exp(βΔV_n)` | Direct GaMD exponential correction dominated by few frames |
| Neighbor overlap | Sum of pairwise minimum normalized CV histograms | Weak shared support; inspect full 2D restraints |
| `N_k` | Raw analyzed count per window | Zero means intended state not sampled |
| Block bootstrap σ | Variation after resampling trajectory blocks | Low block count/correlation makes uncertainty unreliable |

Use ESS fraction (`ESS / N`), not count alone. Per-window GaMD ESS can look healthy while pooled ESS collapses because windows have different boost means. PMF solver convergence only validates fixed-point tolerance, not support.

## Uncertainty

`--pmf-uncertainty` resamples trajectory blocks, retains solved global `f_k`, recomputes subset MBAR denominators using subset `N_k`, then rebuilds selected PMF. This preserves within-block correlation more honestly than resampling individual frames. It remains conditional on sampled support, block definition, estimator, bins, and frozen analysis population. It cannot quantify systematic CV/model/force-field error.

## Recommended record

Archive `pmf_summary.json`, selected and alternate estimator CSVs, boost diagnostics, overlap plots, uncertainty output, exact analysis command, `effective_config.*`, explicit windows, and segment population. Record estimator, `β`/temperature, bins, `base_ess / n_samples`, worst overlap, number of bootstrap blocks, and all exclusions.

See [PMF validity](pmf-validity.md) for acceptance gate and [physical foundations](../guide/foundations.md) for ensemble/bias definitions.
