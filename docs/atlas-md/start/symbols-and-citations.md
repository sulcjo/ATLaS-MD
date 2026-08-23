# Symbols and background references

## Symbols

| Symbol | Meaning | Units |
| --- | --- | --- |
| `x`, `V` | coordinates, volume | coordinates / nm³ |
| `U(x)` | physical potential energy | kJ/mol |
| `U_bias,k` | window `k` restraint | kJ/mol after conversion |
| `ξ`, `z` | CV and its value | CV-specific |
| `c_k`, `K_k` | umbrella center and force constant | CV units; kcal mol⁻¹ CV⁻² input |
| `β` | `1/(RT)` | mol/kJ |
| `kBT` | thermal energy `RT` | kJ/mol or kcal/mol |
| `ΔV` | GaMD boost potential | kJ/mol |
| `u_nk` | sample/window reduced bias | dimensionless |
| `N_k`, `f_k` | window count, MBAR free-energy offset | count, dimensionless |
| `w_n` | normalized MBAR sample weight | dimensionless; sums to 1 |
| `F(z)` | projected PMF | energy, arbitrary additive zero |
| ESS | effective sample size | frames-equivalent |

## Background reading

- Shirts MR, Chodera JD. *Statistically optimal analysis of samples from multiple equilibrium states.* J Chem Phys 129, 124105 (2008). MBAR derivation.
- Kumar S et al. *The weighted histogram analysis method for free-energy calculations on biomolecules.* J Comput Chem 13, 1011–1021 (1992). Histogram free-energy analysis.
- Sugita Y, Okamoto Y. *Replica-exchange molecular dynamics method for protein folding.* Chem Phys Lett 314, 141–151 (1999). Replica exchange.
- Miao Y, Feher VA, McCammon JA. *Gaussian accelerated molecular dynamics: unconstrained enhanced sampling and free energy calculation.* J Chem Theory Comput 11, 3584–3595 (2015). GaMD.

Read references for theory; use ATLAS-MD pages for implemented conventions, artifact names, defaults, and guards. See [foundations](../guide/foundations.md), [reweighting](../analysis/reweighting.md), and [validation workflow](../analysis/validation-workflow.md).
