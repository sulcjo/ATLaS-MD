# Physical and statistical foundations

This page defines quantities used by ATLAS-MD. It is model documentation, not a replacement for system-specific validation.

## Ensemble and units

Production targets isothermal-isobaric sampling (NPT). For configuration `x`, volume `V`, potential energy `U(x;V)`, pressure `p`, and inverse temperature
`β = 1 / (RT)`, equilibrium probability is proportional to:

\[
\pi(x,V) \propto \exp[-\beta(U(x;V)+pV)].
\]

ATLAS-MD stores thermodynamic energies in kJ/mol. `R = 0.008314462618` kJ mol⁻¹ K⁻¹, so `β` has mol/kJ. At 300 K, `kBT = RT ≈ 2.494` kJ/mol (`≈ 0.596` kcal/mol). Umbrella input constants use kcal mol⁻¹ CV⁻²; analysis converts with `1 kcal/mol = 4.184 kJ/mol`. Distances are Å at CV interface, while OpenMM coordinate work uses nm.

Free-energy differences measure relative probability:

\[
\Delta F_{A\to B}=-k_BT\ln(P_B/P_A).
\]

Absolute PMF zero is arbitrary. Compare differences only after using same temperature, reference convention, estimator, bins, and population definition.

## Atomistic model and integrator

System setup builds explicit-water, PME, NPT peptide simulations by default. Read resolved `effective_config.*`, `box_audit.json`, and manifest: force field, water model, ionic strength, box, constraints, thermostat/barostat, and seeds belong in scientific record.

HMR-CMD/HMR-GaMD defaults to 4 fs; CMD/GaMD defaults to 2 fs. HMR changes masses, not equilibrium configurational distribution in ideal sampling, but changes stable integration limit and dynamical timescales. Never treat trajectory time from HMR as direct kinetic time without separate validation.

Constraints, finite timestep, thermostat, barostat, PME settings, and cutoff choices are numerical/model assumptions. Equilibrium reweighting cannot repair bad integration, failed equilibration, wrong protonation, finite-size artifacts, or inadequate force field.

## Collective variables and probability projection

A collective variable `ξ(x)` projects atomistic coordinates into one or two reported coordinates. Reported PMF is projection free energy:

\[
F(z)=-k_BT\ln P(\xi(x)=z)+C.
\]

It is not full molecular free-energy landscape. Distinct conformations can share one CV value; hidden barriers can prevent local equilibrium inside CV bins. Pick CVs that resolve scientific states and inspect orthogonal coordinates. See [collective variables](collective-variables.md).

## Harmonic umbrellas

Window `k` adds bias to physical potential. In one dimension:

\[
U_k^{bias}(x)=\tfrac12 K_{1,k}[\xi_1(x)-c_{1,k}]^2.
\]

For a restrained second CV, ATLAS-MD adds `½ K₂,k[ξ₂(x)-c₂,k]²`. A window's sampled density therefore differs from unbiased density by its complete bias; all active restraints must enter analysis. Width scale near harmonic region is `σ ≈ sqrt(kBT/K)`, but molecular PMFs and orthogonal coupling alter real overlap.

## Replica exchange

REUS swaps configurations between two restraint Hamiltonians. For replicas holding `x_i` and `x_j`, Metropolis acceptance is:

\[
p_{acc}=\min\{1,\exp[-\beta(U_i^{bias}(x_j)+U_j^{bias}(x_i)-U_i^{bias}(x_i)-U_j^{bias}(x_j))]\}.
\]

Unbiased physical energy cancels because temperature and physical Hamiltonian match. Exchange acceptance measures ability to swap given current samples; it does not prove every window covers intended CV intersection or that MBAR weights have usable ESS.

## GaMD in physical terms

GaMD adds non-negative boost `ΔV(x)` below configured energy thresholds, creating sampled potential `U'(x)=U(x)+U_bias(x)-ΔV(x)`. Low-energy barriers flatten, improving transitions. To recover target distribution, each sample needs factor `exp[βΔV(x)]` after umbrella unbiasing. Large or non-Gaussian boosts make this factor highly variable; [reweighting](../analysis/reweighting.md) states estimator limits.
