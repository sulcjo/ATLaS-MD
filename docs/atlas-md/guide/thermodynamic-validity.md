# Thermodynamic target and detailed-balance contract

ATLaS-MD is designed to preserve a well-defined extended ensemble of molecular configurations and thermodynamic-state assignments. This page states that target explicitly and separates three questions that are easy to conflate:

1. **What distribution is intended?**
2. **Do replica-exchange and volume-move kernels preserve that distribution?**
3. **Does the finite-timestep within-state propagator sample it exactly?**

The first two can be stated and tested algebraically. The third is necessarily approximate for finite-timestep Langevin/GaMD propagation and must be validated numerically.

## 1. Intended thermodynamic target

Let replica $r$ have Cartesian configuration $x_r$ and, for NPT simulations, periodic-box volume $V_r$. Let $\sigma(r)$ denote the thermodynamic state currently assigned to replica $r$.

For state $k$, define the effective configurational potential

$$
U_k^*(x,V)
=
U_{\mathrm{phys}}(x,V)
+
W_k(x,V)
+
\Delta V_k(x,V),
$$

where:

- $U_{\mathrm{phys}}$ is the physical force-field potential;
- $W_k$ contains state-dependent explicit biases such as umbrella and secondary-CV restraints;
- $\Delta V_k$ is the state-dependent GaMD/Pep-GaMD boost actually applied by the propagator.

ATLaS-MD stores energies as molar energies. Therefore, when energies are expressed in kJ/mol,

$$
\beta = \frac{1}{RT}.
$$

### NVT target

With respect to Cartesian configurational measure $dX=\prod_r dx_r$ and counting measure over state assignments,

$$
\Pi_{\mathrm{NVT}}(X,\sigma)
\propto
\prod_r
\exp\left[-\beta U_{\sigma(r)}^*(x_r)\right].
$$

The kinetic contribution is omitted here because the exchange machinery changes only thermodynamic-state labels. At common temperature and fixed masses, the momentum distribution is independent of the state assignment and therefore cancels from exchange ratios.

### NPT target

For common pressure $P$, the configurational NPT target with respect to Cartesian coordinates and volume measure $dx_r\,dV_r$ is

$$
\Pi_{\mathrm{NPT}}(X,V,\sigma)
\propto
\prod_r
\mathbf{1}_{x_r\in\Omega(V_r)}
\exp\left\{
-\beta\left[
U_{\sigma(r)}^*(x_r,V_r)+PV_r
\right]
\right\}.
$$

There is **no additional explicit factor of $V_r^{N_{\mathrm{mol}}}$** in this Cartesian-coordinate representation.

Such a factor appears when a volume-changing transformation is expressed in scaled molecular translational coordinates, or equivalently as the Jacobian of the molecular volume-scaling proposal.

For an isotropic trial

$$
V' = V + \delta V,
\qquad
s = \left(\frac{V'}{V}\right)^{1/3},
$$

ATLaS-MD translates the molecular reference position of each of the $N_{\mathrm{mol}}$ molecules by $s$, leaving its internal coordinates unchanged. The transformation Jacobian is

$$
J=s^{3N_{\mathrm{mol}}}
=
\left(\frac{V'}{V}\right)^{N_{\mathrm{mol}}}.
$$

Because ATLaS-MD proposes $\delta V$ from a fixed symmetric distribution,

$$
q(V'\mid V)=q(V\mid V'),
$$

and the Metropolis log acceptance ratio for the volume move is

$$
\ln A
=
-\beta\left[
U_k^*(x',V')-U_k^*(x,V)+P(V'-V)
\right]
+
N_{\mathrm{mol}}\ln\left(\frac{V'}{V}\right).
$$

Equivalently, in dimensionless scaled molecular translational coordinates, the configurational density contains the familiar factor

$$
V^{N_{\mathrm{mol}}}e^{-\beta(U^*+PV)}.
$$

These are two representations of the same target measure, not two different ensembles.

## 2. Replica exchange

Replica exchange changes the thermodynamic-state assignment $\sigma$ while leaving Cartesian coordinates, momenta and box volumes attached to their original simulation contexts.

For replicas $a$ and $b$ carrying states $i$ and $j$,

$$
(a:i,\;b:j)
\longrightarrow
(a:j,\;b:i).
$$

The target-density ratio is therefore determined by

$$
\Delta
=
U_j^*(x_a,V_a)
+
U_i^*(x_b,V_b)
-
U_i^*(x_a,V_a)
-
U_j^*(x_b,V_b).
$$

At common temperature and pressure, all state-independent terms cancel:

- the physical potential $U_{\mathrm{phys}}$;
- kinetic energy;
- the $PV$ contribution;
- any volume-measure or molecular-scaling factor that depends on $V_r$ but not on the exchanged state label.

Thus only thermodynamic-state-dependent Hamiltonian terms need to be cross-evaluated.

For a symmetric pair proposal,

$$
P_{\mathrm{acc}}
=
\min\left[1,e^{-\beta\Delta}\right].
$$

For a nonuniform proposal such as the Gibbs-walk exchange,

$$
P_{\mathrm{acc}}
=
\min\left[
1,
e^{-\beta\Delta}
\frac{q(\sigma\mid\sigma')}{q(\sigma'\mid\sigma)}
\right],
$$

where $q(\sigma'\mid\sigma)$ is the forward proposal probability and $q(\sigma\mid\sigma')$ is the reverse proposal probability.

### GaMD lambda ladders

For a GaMD $\lambda$-ladder, $\Delta V_k$ is part of the thermodynamic-state Hamiltonian and therefore does **not** generally cancel. The boost must be evaluated under every candidate state's $\lambda_k$, exactly like any other state-dependent Hamiltonian term.

The production implementation follows this rule: exchange uses a full state-by-configuration boost matrix, rather than assuming that a state-dependent boost cancels.

## 3. Detailed balance versus sweep invariance

ATLaS-MD distinguishes between detailed balance of an elementary exchange transition and invariance of a complete exchange sweep.

For an elementary pairwise Metropolis exchange kernel $K_{ij}$,

$$
\Pi(\sigma)K_{ij}(\sigma\rightarrow\sigma')
=
\Pi(\sigma')K_{ij}(\sigma'\rightarrow\sigma).
$$

Thus the elementary move satisfies detailed balance with respect to the intended extended-ensemble target.

For a state-dependent or otherwise nonuniform proposal, the Metropolis-Hastings factor

$$
\frac{q(\sigma\mid\sigma')}{q(\sigma'\mid\sigma)}
$$

restores the same relation.

Production exchange modes apply multiple valid elementary kernels sequentially. If

$$
K=K_1K_2\cdots K_n
$$

and each constituent kernel preserves $\Pi$,

$$
\Pi K_i = \Pi,
$$

then

$$
\Pi K = \Pi.
$$

A sequential composition of reversible kernels need not itself be reversible. In general,

$$
\Pi(x)K(x,y)\neq\Pi(y)K(y,x)
$$

even though

$$
\Pi K=\Pi.
$$

Therefore it is neither necessary nor generally correct to claim that an entire ordered exchange sweep itself satisfies detailed balance.

Likewise, a state-independent random mixture of valid kernels preserves the target distribution. If

$$
K=\sum_s p_sK_s,
$$

with scheduling probabilities $p_s$ independent of the current molecular configuration and every $K_s$ preserving $\Pi$, then

$$
\Pi K
=
\sum_s p_s\Pi K_s
=
\Pi.
$$

The precise statement for ATLaS-MD is therefore:

> **Each elementary exchange transition is constructed to satisfy detailed balance with respect to the intended extended-ensemble target. Sequential or randomly scheduled compositions of these transitions preserve the same stationary distribution, although a complete exchange sweep need not itself be reversible.**

## 4. What this does and does not prove

This thermodynamic contract establishes the target against which exchange and NPT moves are derived and tested. It does **not** imply mathematically exact end-to-end sampling at finite timestep.

Correct equilibrium sampling additionally requires that the within-state propagator preserve, to the accuracy appropriate for its finite timestep, the configurational distribution associated with the same $U_k^*$. OpenMM's Langevin-middle scheme and the GaMD custom integrators are finite-step stochastic propagators, so their stationary distribution is only exact in the appropriate limiting sense.

Accordingly:

- exchange correctness is tested independently of propagation accuracy;
- NPT acceptance mathematics is tested independently of the MD integrator;
- GaMD/Pep-GaMD force and energy bookkeeping must agree with the same $U_k^*$;
- adaptive epochs that change Hamiltonian parameters are treated as distinct phases rather than silently pooled as one fixed thermodynamic state;
- ergodicity and mixing are separate diagnostics from detailed balance.

## 5. Current validation status

The current codebase includes:

- exact finite-state transition-matrix tests for pairwise Metropolis and Gibbs-walk exchange kernels;
- tests that distinguish elementary detailed balance from stationary invariance of composed sweeps;
- explicit state-by-configuration GaMD boost evaluation for lambda ladders;
- an application-controlled NPT Metropolis move using the same molecular-volume Jacobian convention as OpenMM's native Monte Carlo barostat;
- independent analytic tests of the NPT acceptance expression and molecule-count Jacobian;
- stage-aware NPT target adapters that separate physical, bias, auxiliary and GaMD boost energies.

Remaining validation work is intentionally narrower than the contract itself: realistic zero-boost comparison against native OpenMM NPT statistics, pathological periodic-image tests for molecule-centroid volume scaling, and continued numerical validation of the finite-timestep GaMD propagator against the stated effective potential.
