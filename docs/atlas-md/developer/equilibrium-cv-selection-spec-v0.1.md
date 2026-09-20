# ATLaS-MD: automatic CV selection for equilibrium thermodynamics

Version: 0.1 — proposed implementation specification, 2026-09-19  
Repository baseline: `sulcjo/ATLaS-MD`, `main` at `cdee6cc87ac540b0bac9b230f389c77d70383faf`  
Status: design only; no selector implemented, candidates ranked, or simulations launched.

## 1. Decision and scope

Build an automatic, native-blind selector that chooses a pair of deployable collective variables by the precision and reproducibility of reweighted equilibrium estimates at a fixed computational budget. Return an explicit inconclusive or failure result when evidence is inadequate.

The user’s objective is equilibrium thermodynamics. Physical rates, transition-path fidelity, unbiased relaxation times, and recovery of physical kinetics are not objectives or acceptance gates. Sampling correlation remains relevant to uncertainty.

The selected pair is the best-supported choice **among the declared candidates, sampling protocols, observables, and budgets tested**. It is not a universal optimum. Two CVs do not have to parameterize the entire molecular configuration space to support correct thermodynamic sampling.

### Required outputs

1. A frozen, reproducible two-CV model, including units, atom mapping, all fitted coefficients, and provenance.
2. A supported window/boost design within the declared replica and resource limits.
3. Equilibrium-observable estimates with uncertainty and explicit support status.
4. A comparison against the strongest tested one-dimensional baseline.
5. A machine-readable decision with reasons, including permission to decline selection.

### Initial target

Use original chignolin `GYDPETGTWG` at 300 K under the intended ff14SB/TIP3P Hamiltonian and declared NPT settings. Record the complete target definition: termini, protonation, ions, force-field XML hashes, box treatment, pressure, constraints, and numerical integration settings. Do not substitute CLN025 or take another force field’s literature population as numerical ground truth.

Thermodynamic correctness here means correctness for that specified molecular model, within demonstrated numerical and sampling accuracy. Agreement with experiment is a separate validation question.

### Two operating modes

| Mode | Search space | Allowed conclusion |
|---|---|---|
| `joint_pair` — final objective | Both coordinates selected from the declared deployable dictionary; pair-conditioned residual candidates allowed | Best-supported tested pair |
| `fixed_primary` — initial integration milestone | Existing contact CV1 fixed; automatically select its companion | Best-supported tested companion to that CV1 |

The constrained milestone must not be reported as completing joint pair selection.

## 2. Repository findings that determine the implementation

These are observations from the pinned source, not claims that new behavior already exists.

| Existing location | Verified behavior | Consequence for this design |
|---|---|---|
| `smoke/README.md`, `smoke/pmf_analysis/README.md` | Old campaign has weak structural population support; some segments are excluded for incorrect ensemble provenance; coordinate observables were forward-filled | Use old results for discovery/debugging, never as an equilibrium reference or independent confirmation |
| `smoke/scripts/cv_hunt.py` | Fits six separate plain PCs and six PCs after **quadratic** regression on CV1; feature order is all sines followed by all cosines | Importing old projections requires explicit feature permutation and the exact regression polynomial |
| `gareus/tica.py::backbone_dihedral_features` | Runtime feature order interleaves sine/cosine for each torsion | Array width alone cannot establish projection compatibility |
| `gareus/tica.py::compute_bootstrap_torsion_pca` | `component` is a **count of leading PCs combined**, not an individual PC index; residualization is linear and followed by coefficient projection | Add a new explicit selection contract; never interpret `component=2` as the handoff’s residual PC2 |
| `gareus/cv.py`, `gareus/mbar_analysis/cv2_reprojection.py` | Existing linear torsion projection uses feature weights plus offset | A true polynomial-residual CV also needs its CV1-dependent term at runtime and in reprojection |
| `gareus/query.py::reconstruct_bias_matrix` | Delegates to the strict reconstruction implementation | Reuse the canonical path |
| `gareus/correctness/bias.py` | Constructs umbrella-plus-ladder reduced bias, with explicit inactive axes and missing-coordinate checks | Preserve strict errors; add no alternative bias formula in the selector |
| `gareus/mbar_analysis/ladder.py`, `estimators.py` | GaMD ladder boost is included in reduced potentials; extra exponential/cumulant corrections are excluded | The exact-ladder estimator is mandatory; legacy name `umbrella_only` does not mean the boost was omitted |
| `gareus/windows.py::load_explicit_2d_window_csv` | Accepts sparse per-window primary/secondary restraints | Use sparse layouts under the replica cap; a full Cartesian grid is unnecessary |
| `gareus/store.py`, `parquet_manifest.py` | Samples are stored by segment with committed-file manifests | Extend this storage contract; do not invent a parallel untracked sample pipeline |
| `gareus/tica.py::DihedralObsBuffer` | Records feature arrays, local steps, windows, and CV values; inactive CV2 may be NaN | Selection observations need stronger identity and complete CV measurements |
| `gareus/mbar_subsample.py` | Provides a strict uncertainty-facing API and a legacy fallback that can retain all rows | Use strict status handling; fallback data cannot receive a precision certificate |
| `gareus/cv_discovery.py` | Existing suggestion command is heuristic | Keep it compatible; add a separate selector rather than changing its meaning |

Source anchors: [handoff](https://github.com/sulcjo/ATLaS-MD/blob/cdee6cc87ac540b0bac9b230f389c77d70383faf/smoke/README.md), [candidate script](https://github.com/sulcjo/ATLaS-MD/blob/cdee6cc87ac540b0bac9b230f389c77d70383faf/smoke/scripts/cv_hunt.py), [torsion implementation](https://github.com/sulcjo/ATLaS-MD/blob/cdee6cc87ac540b0bac9b230f389c77d70383faf/gareus/tica.py), [strict bias](https://github.com/sulcjo/ATLaS-MD/blob/cdee6cc87ac540b0bac9b230f389c77d70383faf/gareus/correctness/bias.py).

Preserve the handoff’s existing η²/reweight-ratio experiment as a historical, separate hypothesis. This specification is a new prospective protocol. Failure to select PC2 does not reject all two-CV sampling designs. Selecting it on old data does not provide independent validation.

## 3. Thermodynamic contract

Let `x` include molecular coordinates and, for NPT, the volume/box variables. With a common reference measure, define the target reduced potential `u0(x)`. A fixed sampling state is

\[
u_k(x)=u_0(x)+\beta\{B_k[z_1(x),z_2(x)]+\Delta V_k(x)\},
\]

where the umbrella is

\[
B_k=\tfrac12 k_{1k}(z_1-c_{1k})^2+
     \tfrac12 k_{2k}(z_2-c_{2k})^2,
\]

and the positive GaMD boost `ΔV_k` is evaluated from that state’s actual frozen envelope and λ. Do not assume simple linear λ scaling unless the canonical boost implementation specifies it.

For equilibrium samples from these states, normalized target weights satisfy

\[
\widetilde w_n=
\frac{e^{-u_0(x_n)}}{\sum_k N_k e^{\widehat f_k-u_k(x_n)}},
\qquad
w_n=\frac{\widetilde w_n}{\sum_m\widetilde w_m},
\qquad
\widehat{\langle A\rangle}_0=\sum_n w_n A(x_n).
\]

This is the MBAR equilibrium-estimation basis, subject to adequate sampling and overlap; it is not an assertion of exact finite-sample results. [Shirts and Chodera](https://arxiv.org/abs/0801.1426)

At common temperature, pressure, physical Hamiltonian, and measure, subtracting `u0(x_n)` from every state column gives the repository’s reduced-bias convention. The target then has zero reduced bias. Common physical-energy, pressure-volume, and measure terms cancel. Changing temperature, pressure, constraints, or the physical Hamiltonian requires a full reduced-potential treatment; v1 must reject such a union rather than use the bias-only shortcut.

Mandatory rules:

- Apply the GaMD correction once, through the canonical reduced potentials. Never multiply ladder-MBAR weights by `exp(βΔV)` or a cumulant approximation.
- A changed CV, center, stiffness, envelope, or other Hamiltonian parameter creates a new state identity. A reused numeric window ID is not sufficient identity.
- Production measurements use frozen states. Exclude calibration, active adaptation, steering, and declared post-change equilibration from thermodynamic scoring.
- A record of a time-dependent bias is not, by itself, a justification for ordinary equilibrium MBAR on the evolving trajectory.
- Initial conformer counts and minimized bank energies do not define equilibrium populations. Incorrectly equilibrated starting distributions can contaminate finite runs despite a correct MBAR formula.
- Correct reweighting cannot compensate for an unvisited relevant region, an incorrect force, an invalid barostat/exchange kernel, or appreciable integrator bias.

For intuition, a bias that depends only on `s(x)` leaves the conditional distribution `p_B(x|s)=p_0(x|s)`. It does not directly distinguish structures sharing those CV values, although excursions through other `s` values may improve their interconversion. This identity applies to that CV-only bias; a separate GaMD term can change the conditional distribution.

## 4. Data partitions and native blindness

Use four data roles with separate manifests:

| Role | Data | Permitted use |
|---|---|---|
| Historical exploratory | Existing handoff/campaign | Interface debugging, candidate ideas, historical hypothesis reproduction |
| Discovery/training | New `broad` conformer bank plus declared discovery sampling | Fit features, clusters, candidate coefficients, window rules, shortlist |
| Pilot comparison | Fresh independent campaigns for every tested arm | Rank thermodynamic sampling performance |
| Confirmation | New seeds and random streams, after winner/protocol freeze | Confirm the selected pair and its comparison baseline |

Native references, RMSD to an experimental fold, native Q, folded labels, and residue pairs chosen from the known fold are excluded from selector inputs. Use an allowlist extractor; do not load a label-bearing historical NPZ wholesale and rely on later code to ignore labels.

The generic feature dictionary may include all eligible backbone donor–acceptor geometries. It may therefore contain a physically useful native contact without being told which one it is.

Split discovery folds by generator lineage, initial conformer family, and complete independent simulation campaign. Closely related conformers or velocity replicas of one seed stay together. Fit scaling, regression, PCA, cluster centroids, and shortlisting models inside training folds. Freeze final models after discovery and before pilot evaluation.

No new unbiased 300 K kinetic swarm is required. Discovery may use the existing enhanced-sampling machinery. High-temperature or other exploration is optional, separately identified, and is not automatically target-ensemble evidence.

## 5. Durable data and model contracts

### 5.1 Observation identity

Canonical observation key:

`(campaign_uuid, phase_uuid, segment_uuid, replica_slot, step_local, sample_event)`

Also record `exchange_cycle`, `walker_lineage` when available, exact pre/post-exchange convention, `configuration_id`, `state_hash`, `pair_model_hash`, `target_hash`, and feature schema version. Paths and `(step, replica)` alone are not unique identifiers across resumes.

Each scoring observation must carry, directly or through a verifiable join:

- Both selected CV values, including in states where either restraint is zero.
- Canonical torsion features and the frozen evaluation observables.
- Actual origin-state identity and all raw energy channels required by its GaMD envelope.
- Box information and enough configuration provenance to audit PBC/geometry.
- Validity status and a reason for every excluded record.

Record the inexpensive structural observables at the thermodynamic sample cadence. If some require coordinate files, use an exact matched subset with a declared configuration-independent recording schedule. Refit MBAR and state counts on that subset. Do not forward-fill, interpolate, or silently renormalize a convenience subset of global weights.

Missingness triggered by a conformation is not random thinning and cannot be repaired by merely refitting on the surviving rows. Treat it as an integrity failure until explained.

### 5.2 Frozen CV model

`cv_pair_model.json` must contain:

| Field group | Required contents |
|---|---|
| Identity | `schema_version`, content hash, target/topology hashes, training-manifest hash |
| Geometry | Atom identities and indices, torsion quadruplets, bond/contact definitions, PBC convention |
| Feature order | Explicit ordered feature names; never inferred from vector length |
| Coordinate definition | Primitive functions, weights, intercepts, residual polynomial coefficients, units |
| PCA semantics | `component_index_1based`, exact eigenvector, fit measure, eigenvalue diagnostics; mixtures use a different explicit field |
| Pair transform | Frozen affine transformation and offsets, design-measure identity, sign convention |
| Support | Training ranges and quantiles, extrapolation flags, numerical conditioning diagnostics |
| Reproducibility | Software/source hashes, random seeds, fitting parameters, immutable schema version |

The same canonical model evaluator feeds NumPy analysis, runtime force construction, recorded CV values, and MBAR reconstruction. Provide a round-trip assertion on every export/import. Do not edit a model in place during a run.

### 5.3 State and budget identity

Hash the complete bias-state definition: pair model, centers, force constants, λ, envelope, target, and unit convention. Keep runtime state labels as separate aliases.

Record allocated GPU-seconds, CPU-hours, wall time, setup/calibration time, retained measurement time, disk bytes, failures, and restarts. Pair-specific setup costs count against its comparison budget. Shared discovery cost is reported separately and included in total project cost.

## 6. Candidate construction and orthogonality

### 6.1 Initial dictionary

Start with a bounded, declared dictionary:

- Existing smooth nonlocal contact fractions: a small fixed set of atom selections and distance parameters.
- Radius of gyration and end-to-end distance, subject to differentiable/PBC-safe implementation.
- Smooth generic backbone donor–acceptor proximity summaries, without native residue choices.
- Individual torsion PCA components 1–6 from periodic sine/cosine features.
- Individual residual-torsion components 1–6 for each declared compaction anchor.

Use continuous contact functions, not the hard distance thresholds appearing in some exploratory analysis scripts. Candidate definitions are frozen before observing pilot performance. Initial cap: 32 primitive scalar candidates plus explicitly enumerated pair-conditioned residual candidates. Report the actual number of tested pairs.

`joint_pair` enumerates all deployable unordered primitive pairs and the declared anchor/residual pairs. `fixed_primary` enumerates companions only. A candidate that cannot be evaluated consistently with differentiable forces is not pilot-eligible.

### 6.2 Exact residual-coordinate definition

For periodic torsion feature vector `φ(x)` and anchor `a(x)`, fit weighted least squares on training data:

\[
m(a)=b_0+b_1a+b_2a^2,\qquad r(x)=\phi(x)-m[a(x)].
\]

The default prospective family uses degree 1; degree 2 is an explicit additional family if declared before trials. Historical PC2 reproduction must use degree 2 because the handoff script does.

Diagonalize the weighted residual covariance and create **separate** candidates

\[
s_j(x)=v_j^T[\phi(x)-b_0-b_1a(x)-b_2a(x)^2],
\quad j=1,\ldots,6.
\]

The runtime must include every term. In particular,

\[
\nabla s_j=\sum_l v_{jl}\nabla\phi_l
 -[v_j^Tb_1+2a\,v_j^Tb_2]\nabla a.
\]

Using `v_j·φ` alone changes the selected CV and its force. The existing projected-coefficient torsion-only method remains a separate, explicitly named candidate family; it is not an interchangeable implementation of this residual coordinate.

Degenerate eigenvalues make component numbers unstable. Save actual vectors, compare subspaces across discovery folds, and avoid claims that PC number has an invariant molecular meaning.

### 6.3 Define the measure used for orthogonality

Default `q_design`: a reproducible empirical discovery distribution, balanced across independent seed families/campaigns according to a frozen weighting rule. If bank and simulation data are combined, declare their mixture weights; recommended starting rule is equal total mass to each source class, then equal mass to families within each class.

Bank density, balanced state density, and target equilibrium density are different measures. Do not call `q_design`-orthogonality equilibrium orthogonality. An optional target-weighted fit is allowed only when target weights already pass support checks; it is not required to start discovery.

For each pair `(a,b)`, apply the frozen Gram–Schmidt transform

\[
z_1=(a-\mu_a)/\sigma_a,\quad
\gamma=\operatorname{Cov}_{q}(b,z_1),\quad
z_2=\frac{b-\mu_b-\gamma z_1}
{\sqrt{\operatorname{Var}_{q}(b)-\gamma^2}}.
\]

This provides zero training covariance and unit variances when nonsingular. It is an invertible pair transformation, not a claim of statistical independence. Fix coordinate ordering by a declared rule; when using an anchor/residual pair, the anchor comes first. The ordering is part of the tested bias protocol because sparse axis restraints depend on it.

Report held-out correlation and nonlinear predictability in both directions. Large held-out correlation is a distribution-shift/conditioning diagnostic, not an automatic scientific rejection. Reject collapse or near-deterministic duplication only when the declared numerical and structural checks agree; do not require complete independence of coupled molecular variables.

Evaluate the local gradient Gram matrix, after tangent projection for constraints where used:

\[
G_{ij}(x)=\nabla z_i^T M^{-1}\nabla z_j.
\]

Persistent rank deficiency flags two biases acting along effectively one local direction. Report its distribution, including low-population structural regions. A global average must not hide singularity where a relevant basin lies.

### 6.4 Runtime realization

Compile smooth primitive features and algebraic pair transforms into a single-level `CustomCVForce` where practical. A weighted sine/cosine sum can be supplied by a torsion force; contact sums by a bond-based force. Compose the residual polynomial in the outer expression, keeping both umbrella terms and their gradients consistent. Do not add the inner CV-defining energies as extra physical energies.

This uses OpenMM’s documented ability to form an energy from scalar functions supplied by force objects. The exact compiler layout and parameter naming require tests against the supported OpenMM/CUDA environment. [OpenMM CustomCVForce](https://docs.openmm.org/latest/api-python/generated/openmm.openmm.CustomCVForce.html)

The compiled umbrella must remain outside the physical GaMD energy channels. Adding/changing a CV must not change the meaning of `v_pep` or `v_dih`. Update the existing force-group accounting and NPT/exchange paths through tested adapters.

## 7. Cheap offline shortlisting

Offline scores are proposal tools. They cannot certify faster equilibrium convergence.

1. Build a shared, native-blind structural feature space from periodic torsions and generic contact patterns. Standardize feature families so the larger contact block cannot dominate by dimension alone.
2. Fit coarse and fine structural partitions on discovery training folds. Suggested sizes for chignolin are 8 and 24 clusters, with deterministic initialization and a frozen out-of-support rule. These are geometric cells, not asserted metastable states.
3. Measure held-out structural reconstruction and cluster prediction from each single coordinate and pair. Use the same fixed model class and regularization policy for all candidates; split by families.
4. For probabilistic cluster predictions, calculate the held-out cross-entropy `L(P)`. Define incremental information as `min[L(z1), L(z2)] - L(P)`. Fit single-coordinate baselines nonlinearly so a nonlinear copy does not win merely by enlarging the basis.
5. Rank by conservative held-out loss, incremental information, structural coverage, force conditioning, and evaluation cost. Keep these as separate columns with deterministic tie-breaking; do not hide them in an arbitrary weighted sum.
6. Keep at most three pairs: the two strongest candidates from distinct structural families, plus one predeclared exploration slot selected reproducibly among the remaining well-conditioned, nonredundant candidates. If fewer pass, retain fewer.

An optional ranking statistic is conditional cluster information `I(C; z2 | z1)`, estimated through held-out predictive loss. Label it information about the discovery partition, not equilibrium entropy or thermodynamic sampling efficiency.

Also select a promising single-coordinate baseline. Retain the incumbent contact baseline if it differs. The final comparator is the strongest adequately supported one-dimensional arm actually tested. Searching a restricted set of baselines does not establish optimality over all possible one-dimensional schemes.

Fail offline with `NO_DEPLOYABLE_PAIR` if there are no force-valid nonredundant pairs. If structural scores cannot discriminate, spend the bounded exploration slot rather than pretending variance or histogram entropy resolves the question.

## 8. Pilot experiment and resource plan

### 8.1 Freeze the tested sampling protocol

An experimental arm is `(CV model, window placement rule, boost calibration rule, resource allocation)`. A pair cannot be declared best independently of the protocol used to bias it.

Hold the physical target, integrator, exchange cadence, output cadence, GPU allocation policy, and calibration rules fixed across arms. Use identical independent seed-family cohorts across arms where possible, but fresh random streams for each campaign. Within one arm, replicas that exchange are not independent replicate experiments.

Recommended v1 sequence:

1. Generate a new generic `broad` bank and discovery dataset.
2. Fit and freeze candidates, evaluation observables, and the shortlist.
3. Perform a bounded common GaMD calibration covering the union of shortlisted structural regions, with equal allocation to the candidate arms. Freeze its envelope and λ schedule for the comparison.
4. Set up windows for every arm, relax into its fixed biases, and discard the prescribed equilibration interval.
5. Collect measurements without fitting new CVs or changing the bias states.
6. Score every arm at the same scheduled GPU-hour checkpoint.
7. Freeze the provisional winner and strongest baseline, then start independent confirmation.

If common calibration fails for an arm’s region, report failure or a new protocol version. Do not reuse an envelope known to be unsupported there, or silently recalibrate only the apparent winner. A later version may compare fully adaptive pipelines, with all setup costs and state changes accounted for explicitly.

The old `chignolin_8.yaml` is a useful configuration reference, not a ready-made two-CV trial. At the inspected commit it sets `cv2: none` and points to an existing seed directory. Generate new configurations with the declared bank and frozen models; do not merely change a PC number.

### 8.2 Sparse windows under the replica limit

Hardware envelope: 4 × L40S GPUs, up to 192 CPU threads, optional large RAM/scratch allocation; maximum 128 simultaneous replicas. Memory and disk quotas must be explicit run settings, not assumptions that the optional maximum is present.

Suggested comparison layout: 24 spatial states × 4 λ rungs = 96 replicas. One possible preregistered spatial allocation is:

| Spatial state type | Count | Restraints |
|---|---:|---|
| Bridge | 1 | Both umbrella constants zero |
| Axis 1 | 12 | `k1 > 0`, `k2 = 0` |
| Axis 2 | 7 | `k1 = 0`, `k2 > 0` |
| Joint patches | 4 | Both positive; supported cells selected from discovery |

The incumbent handoff’s `(16 + 8) × 4 = 96` separate-axis design remains a valid alternative protocol. Do not mix layouts within a comparison without naming separate arms. A full `16 × 8 × 4 = 512` grid exceeds the cap.

For one-dimensional arms, use the same state budget where feasible: one unrestrained bridge plus 23 supported one-axis windows, crossed with the same λ schedule. If fewer windows are adequate, allow fewer only under a predeclared allocation rule and compare the actual GPU-hours. The comparator is not artificially handicapped by unused replicas.

Window placement:

- Use supported discovery quantiles and structural cells, retaining tail/support diagnostics. Quantiles are a placement device, not assumed equilibrium masses.
- Start stiffness from `k = kBT / σ_window²` with `σ_window = local_spacing / η`, default `η = 1.25`; enforce declared force caps. Convert from kJ/mol to the repository’s kcal/mol/CV² interface explicitly.
- Relax each state and check that its distribution responds to its restraint without numerical failures.
- Use the actual sparse neighbor graph and MBAR overlap. Flat CSV adjacency is not geometric adjacency.
- A bridge state provides an overlap opportunity, not a guarantee of connectivity or structural mixing.
- Do not remove a poorly sampled, potentially relevant region just to make an overlap statistic pass. Reposition/add states within the fixed setup policy, or return a support failure.

### 8.3 Example bounded budget

The following is an adjustable engineering budget, **not a prediction that chignolin will converge within it**. No compute is authorized or launched by this document.

| Stage | Allocation | GPU-hours |
|---|---|---:|
| Shared discovery/calibration | Common pool | 24 |
| Screening | 3 pairs + 1 baseline; 3 independent campaigns each; 8 GPU-hours/campaign | 96 |
| Confirmation | Winner + strongest baseline; 4 new campaigns each; 16 GPU-hours/campaign | 128 |
| Declared contingency | Engineering recovery or equal extensions of all remaining compared arms | 40 |
| Total ceiling | Example with one baseline | 288 |

If a distinct incumbent and new one-dimensional candidate both need screening, add 24 GPU-hours or reduce all screening-arm budgets equally before launching. When all four GPUs are continuously allocated, 288 GPU-hours corresponds to 72 elapsed GPU-occupied hours; CPU setup, I/O, queueing, and idle time can increase wall time.

Use the example to configure a budget, not to infer microseconds from old throughput measurements. Measure throughput with the actual CV compiler, reporters, and state count. Budget exhaustion returns an inconclusive decision; it does not relax acceptance thresholds.

## 9. What equilibrium distributions must be tested

Freeze an evaluation panel independently of the competing CVs:

| Observable family | Initial implementation | Example desired precision |
|---|---|---|
| Coarse structural cells | 8 shared indicators | 95% simultaneous half-width ≤ 0.02 absolute probability |
| Supported fine cells | 24-cell partition; relevance/support list frozen from discovery | Same probability tolerance, with separate rare-cell status |
| Backbone geometry | Per-torsion sine/cosine expectations and periodic marginal histograms | Mean tolerance 0.05; histogram total-variation tolerance 0.05 |
| Contact geometry | Generic per-residue contact indicators/smooth values and selected joint marginals | Mean tolerance 0.03; histogram TV tolerance 0.05 |
| Global shape | Rg and end-to-end histograms with common bins | Histogram TV tolerance 0.05 |
| Thermodynamics | Declared physical-energy expectations, basin free-energy differences | User-specified energy tolerance; default basin ΔF tolerance 0.3 kcal/mol where both populations are supported |
| New structure detection | Distance to discovery support plus novelty-cell assignments | Relevant unresolved novel regions block certification |

These tolerances are defaults to calibrate on controls and freeze before the prospective experiment. Changing them after viewing candidate results creates a new protocol version.

Always include distributions outside the chosen CV pair. A converged CV1–CV2 histogram cannot certify the complete configuration distribution. A finite panel also cannot prove correctness for every possible molecular observable; report the panel and its limits.

Define a cell as prospectively required when it is a coarse partition cell, or when valid discovery equilibrium estimates support its relevance at the declared threshold (example: ≥1% population). Bank-only fine cells are structural challenges with unknown equilibrium relevance. They cannot be discarded on the basis of implicit minimized energies; unresolved relevance must be disclosed.

All arms use the same required cells. If a trial discovers a new region with plausible thermodynamic relevance, add it to a shared challenge log and return `DISCOVERY_INCOMPLETE` until a subsequent frozen protocol evaluates it consistently. Do not silently change just one arm’s scoring denominator.

For supported basins `a,b`, compute

\[
\Delta F_{ab}=-k_BT\ln(p_a/p_b).
\]

Use joint population uncertainty, including covariance. A zero-count basin gives unresolved free energy, not a finite pseudocount-derived estimate with a narrow error bar. Coordinate-density PMFs require bin-volume/Jacobian conventions; compare basin probabilities or fixed common observables across different CV parameterizations, not the depth of each pair’s own PMF minimum.

This selector does not by itself provide configurational entropy. Enthalpy/entropy decomposition would require an additional declared estimator and its own precision targets.

## 10. Uncertainty estimation that preserves replica-exchange dependence

The first implementation should use independent campaign estimates plus synchronized block resampling of the complete replica ensemble. Do not bootstrap saved frames independently.

### 10.1 Point estimates

For each independent campaign, reconstruct reduced potentials, apply the immutable eligibility mask, recompute origin-state counts, solve MBAR, and estimate every observable. Use all eligible measurements for the point estimate if the uncertainty route below handles their correlations. An alternative decorrelated-subsample estimate is a diagnostic, not a second independent experiment.

For equal-budget independent campaigns, the primary arm estimate is their equal-weight arithmetic mean. Do not inverse-variance weight campaigns: a trapped campaign can falsely report extremely small variance. A pooled-MBAR estimate is optional and must preserve all distinct state definitions and hashes; it is not used to conceal between-campaign disagreement.

### 10.2 Resampling units

1. Order each campaign by complete ensemble sampling events or exchange cycles.
2. Form contiguous blocks of **all replicas together**. This retains simultaneous inter-replica dependence and, within blocks, walker transfer history.
3. Never join blocks across discontinuities, unverified resumes, or Hamiltonian changes. A verified continuation may remain one series even if split across files.
4. Select block lengths from ensemble-level structural/weight diagnostics using a fixed rule. Require uncertainty stability when the block length is doubled. If fewer than 20 usable blocks remain, mark uncertainty unresolved; this minimum is an engineering diagnostic, not proof of independence.
5. For each resample, rebuild state counts and refit MBAR. Retain the pairing of observation, origin state, feature row, and reduced-potential row. Holding `f_k` fixed omits normalization uncertainty.
6. If a resample loses necessary support/connectivity, record it as failed. A material failure fraction blocks certification; do not compute intervals from successful resamples alone.

Use 200 block replicates for screening and 500 for confirmation by default. Repeated weight solves may use a validated solver warm start, but convergence must still be checked. CPU cost is measured and bounded.

For observable `a`, let `v_ra` be the per-campaign block variance and `s_a²` the sample variance of independent campaign estimates. With `R` equal-budget campaigns, a conservative diagnostic variance for their mean is

\[
v_a^{\mathrm{check}}=
\max\left\{\frac{1}{R^2}\sum_r v_{ra},\frac{s_a^2}{R}\right\}.
\]

This diagnostic is not a theorem guaranteeing coverage under metastability. Construct final intervals using the validated synchronized bootstrap and small-sample between-campaign uncertainty; adopt the wider result. Between-campaign disagreement is also a separate failure gate.

Concrete initial interval implementation: for `M` primary scalar observables, calculate a simultaneous maximum-deviation interval from the within-campaign block replicates. Also calculate a small-sample interval centered at the arm mean with half-width `t(R-1, 1-α/(2M)) × sqrt(v_check)`. Use the interval hull of these two results. The Student-t approximation assumes approximately normal independent campaign estimates; its adequacy must pass the repeated synthetic coverage tests. If it does not, increase independent replication within the declared budget or return insufficient uncertainty. Do not silently substitute a normal quantile or treat a zero between-run variance as a coverage guarantee.

MBAR uncertainty depends on sampling correlation. Correlated-chain error analysis provides a possible later efficiency improvement, but its independence assumptions must be checked before applying it to exchanging replicas. [Li et al.](https://arxiv.org/abs/2203.01227)

### 10.3 Simultaneous comparisons

Use resampled maximum standardized deviations to construct simultaneous intervals across the frozen primary observables. If the chosen bootstrap method cannot achieve calibrated coverage on the synthetic controls, use a conservative multiplicity-adjusted alternative or report insufficient evidence.

For independence/stability comparisons, use **equivalence margins**, not “failure to reject a difference.” Require the confidence interval on population differences to lie inside the declared tolerance band. For histograms, require an upper confidence bound on TV distance below the declared threshold. Compare independent campaigns and non-overlapping late blocks, with the covariance-aware procedure where data overlap.

Three screening campaigns and four confirmation campaigns provide limited replication. The method must be able to say the interval is too wide. Do not manufacture precision by treating the 96 exchanging replicas as 96 independent experiments.

### 10.4 ESS and support diagnostics

Report global Kish weight ESS, conditional weight concentration inside each required cell, support across independent campaigns and blocks, and correlation-aware observable uncertainty. They answer different questions. PyMBAR’s weight ESS does not automatically include time correlation. [PyMBAR MBAR API](https://pymbar.readthedocs.io/en/stable/mbar.html)

Examples of useful diagnostics are the largest normalized weight within a cell, the number of independent campaigns contributing to it, and whether removing one campaign changes its probability beyond tolerance. A λ round trip is not a new structural observation.

No uncertainty method can detect a basin absent from every input dataset. Diverse starts, a challenge pool, additional exploration, and eventual external validation limit this risk; they do not eliminate it.

## 11. Selection rule and pass/fail gates

### 11.1 Primary score

At the same total cost `C` per arm, let `h_a(P,C)` be the simultaneous 95% interval half-width for primary observable `a`, and `ε_a` its frozen tolerance. Define

\[
L(P,C)=\max_{a\in\mathcal A}
\left[\frac{h_a(P,C)}{\epsilon_a}\right]^2.
\]

Lower is better; `L ≤ 1` satisfies the declared scalar precision targets, subject to every other gate. Histogram equivalence, initialization dependence, unknown support, and correctness gates remain separate. The score is estimated precision, not a bound on unknown total error.

At equal cost, rank passing arms by `L`. Use resampling to assess ranking uncertainty. A practical superiority margin is `L(pair)/L(best_1D) ≤ 0.8`, with a one-sided 95% upper confidence bound below 1. This proposed 20% margin prevents promoting a complicated second coordinate for a negligible apparent gain.

Calculate ranking uncertainty by resampling complete independent campaign records, retaining each record's stored block-replicate distribution, then recomputing the entire arm score including the maximum over observables. Use paired cohort resampling only when the experiment deliberately matched independent seed cohorts across arms; never pair arbitrary replica IDs. With only three or four campaigns this ratio interval can be poorly resolved. Superiority must then be reported as undetermined even if the pair independently passes the absolute equilibrium gates. Repeated synthetic experiments must separately calibrate the false-superiority rate; this comparison is not justified solely by point-score ordering.

If both arms meet absolute targets and are statistically indistinguishable, choose the cheaper/simpler deployable option according to a frozen tie-breaker; return the validated pair as an alternative if appropriate. Do not claim a unique optimum. If only the pair meets targets while the baseline remains support-limited, report that fact; do not invent a finite speedup from an infinite/undefined baseline score.

### 11.2 Gate table

All numerical tolerances below are proposed defaults requiring control-system calibration. Algebraic/provenance rules are hard requirements.

| Gate | Requirement | Failure outcome |
|---|---|---|
| Target and state identity | All contributing rows have valid target/state/model/envelope hashes and compatible Hamiltonians | `INVALID_INPUT` |
| Bias reconstruction | Runtime bias and reconstructed bias agree on audited configurations and cross-states | `INVALID_SAMPLER` |
| CV implementation | NumPy/OpenMM values, analytic/finite-difference gradients, feature-order import, and serialized reload agree | `NO_DEPLOYABLE_PAIR` or `INVALID_SAMPLER` |
| Measurement identity | No ambiguous join, forward-fill, duplicate observation, or unexplained conformation-dependent loss | `INVALID_INPUT` |
| Frozen sampling | Only eligible stationary-phase records; strict equilibration diagnostics | `INSUFFICIENT_EQUILIBRATION` |
| Nonredundancy | No collapsed coordinate or persistent numerical rank deficiency; second axis separates observed structures | `NO_DEPLOYABLE_PAIR` |
| MBAR solution | Numerical residual meets solver tolerance; genuine state-overlap graph supported, including target connection | `INSUFFICIENT_OVERLAP` |
| Structural support | Required cells have repeated support across at least two independent campaigns and enough blocks to estimate uncertainty | `INSUFFICIENT_SUPPORT` |
| Weight concentration | Example: no individual observation contributes >10% of a required cell’s weight; violations cannot pass solely on large global ESS | `INSUFFICIENT_SUPPORT` |
| Uncertainty reliability | Block-length doubling stable within 20%; ≥20 blocks/campaign; bootstrap failure fraction ≤1% for confirmation | `INSUFFICIENT_UNCERTAINTY` |
| Initialization/stability | Simultaneous differences equivalent within the declared margins; histogram TV upper bounds meet tolerance | `NOT_CONVERGED` |
| Novel regions | No unresolved plausible relevant structure outside the frozen evaluation support | `DISCOVERY_INCOMPLETE` |
| Absolute precision | `L ≤ 1`, plus distribution and supported free-energy targets | `PRECISION_NOT_REACHED` |
| Independent confirmation | Winner passes on new campaigns with frozen definition and budget | `CONFIRMATION_FAILED` |

Do not apply a universal numerical MBAR-overlap cutoff borrowed from a marginal CV histogram. Calibrate a diagnostic edge floor on synthetic known-overlap controls, persist it in the protocol, and assess sensitivity to raising/lowering it. Solver convergence, structural support, and target-weight uncertainty remain required regardless of graph connectivity.

Build the usual overlap connectivity graph over **sampled** states. An unsampled target with `N_target = 0` is assessed by its target weights and coverage from the connected sampled ensemble; requiring an ordinary positive-count graph edge to that zero-count state would be an implementation error. The proposed unrestrained λ=0 bridge makes the target sampled when it is actually occupied.

High CV correlation by itself is not a thermodynamic failure. Likewise, there is no mandatory number of physical folding events, physical timescale plateau, or kinetic-model score.

### 11.3 Final decision states

| Decision | Meaning |
|---|---|
| `SELECTED_PAIR_CONFIRMED` | Pair meets equilibrium gates and confirmation; superiority evidence reported separately |
| `PAIR_VALID_NO_CLEAR_ADVANTAGE` | Pair passes but adds no demonstrated practical advantage over the tested baseline |
| `SELECTED_1D_CONFIRMED` | One-dimensional arm meets the targets and is preferred under the rule |
| `NO_DEPLOYABLE_PAIR` | No admissible candidate can be safely and consistently biased |
| `INSUFFICIENT_EVIDENCE` | Sampling/support/precision budget did not justify a winner; attach gate-specific reason codes |
| `INVALID_INPUT` / `INVALID_SAMPLER` | Scientific comparison is invalid until the identified defect is corrected |

Do not select the runner-up automatically after confirmation fails. A new candidate needs its own prospective confirmation allocation.

## 12. Proposed software architecture

Create a separate package `gareus/cv_selection/`. Names below are **new proposed interfaces**, except the existing integration points explicitly identified.

| New module | Responsibility | Core interface |
|---|---|---|
| `schema.py` | Validated protocol, model, observation, state and decision types | `SelectionProtocol`, `CVModel`, `PairModel`, `Decision` |
| `audit.py` | Input manifests, target compatibility, native-blind allowlist, exact joins | `audit_inputs(protocol) -> AuditReport` |
| `features.py` | Canonical PBC-safe feature extraction and topology mapping | `evaluate_features(configurations, feature_schema)` |
| `candidates.py` | PCA/residual models and explicit pair enumeration | `fit_candidates(training, spec) -> list[CVModel]` |
| `orthogonality.py` | Frozen pair transform, redundancy and gradient diagnostics | `fit_pair(a, b, design_measure) -> PairModel` |
| `force_compiler.py` | Compile primitive CVs and umbrella energy using the same model semantics | `compile_bias(pair, state, topology) -> ForceBundle` |
| `shortlist.py` | Grouped held-out structural ranking and bounded exploration slot | `shortlist_candidates(models, folds, policy)` |
| `observables.py` | Freeze structural partitions, bins, tolerances and novelty rule | `build_evaluation_panel(discovery, spec)` |
| `windows.py` | Sparse trial design and state budget | `design_trial_states(pair, discovery, policy)` |
| `runner.py` | Generate/launch/restart bounded campaigns and enforce phase freeze | `run_trial_plan(plan, budget_ledger)` |
| `equilibrium.py` | Strict adapter to canonical reduced potentials and MBAR | `estimate_campaign(dataset, panel) -> Estimate` |
| `uncertainty.py` | Synchronized blocks, MBAR refits, independent-run intervals | `estimate_arm_uncertainty(campaigns, panel, policy)` |
| `gates.py` | Non-negotiable correctness checks and prospective decision rule | `evaluate_gates(evidence, protocol)` |
| `decision.py` | Select winner/baseline and confirm without implicit retuning | `decide(arm_results, confirmation, protocol)` |
| `report.py` | JSON/Markdown report, comparisons and diagnostic plots | `write_report(result, output_dir)` |
| `cli.py` | Separate user command | `main()` |

### 12.1 Existing integration points

| Existing file | Required work |
|---|---|
| `pyproject.toml` | Add `gareus-select-cvs = gareus.cv_selection.cli:main`; keep current `gareus-suggest-cvs` unchanged |
| `gareus/config.py`, `cli.py` | Add explicit frozen-model/trial-manifest inputs with strict validation; prevent conflicting automatic CV updates |
| `gareus/cv.py`, `forces.py` | Add adapters for compiled model evaluation, units and force construction; preserve old modes |
| `gareus/tica.py` | Reuse geometry helpers; preserve legacy component-count behavior; provide explicit model import with feature-order conversion |
| `gareus/production.py` | Connect the compiled pair at existing CV/umbrella setup and sample collection points; record both CVs even when unrestrained |
| `gareus/adaptive_production.py` | Add a fixed-measurement trial path; prevent adaptation, top-up bias drift and automatic CV replacement after freeze |
| `gareus/store.py`, `parquet_manifest.py` | Add versioned selection observations and stable identity; maintain manifest/checkpoint transaction ordering |
| `gareus/query.py` | Strict identity-aware joins and exports; no unlabelled forward-fill |
| `gareus/windows.py` | Reuse sparse CSV loading and neighbor construction; extend pair metadata rather than silently relabel dimensionless values as Å |
| `gareus/correctness/bias.py`, `mbar_analysis/ladder.py` | Reuse unchanged mathematical reconstruction wherever possible; only explicit model adapters around it |
| `gareus/mbar_analysis/solvers.py` | Reuse solvers; add a validated multiplicity-aware interface if needed for efficient bootstrap refits |
| `gareus/mbar_analysis/cv2_reprojection.py` | Support new composite models through the canonical evaluator; keep old torsion-only loading explicit |
| `gareus/provenance.py`, `checkpoints.py` | Hash frozen models/protocols; resume must fail on a changed model or state definition |
| `gareus/npt.py`, `pep_gamd.py` | Integration validation for force groups, cross-state energies, λ=0 and barostat consistency; no new scientific formulas merely for selection |

The large production/adaptive modules were not audited line by line for this specification. During implementation, locate the exact setup/sampling call sites on the branch being changed and keep modifications localized. The interface constraints above, not guessed line numbers, are the binding contract.

### 12.2 Dataset export contracts

Create these per selection run:

| Artifact | Contents |
|---|---|
| `protocol.json` | Frozen algorithm, thresholds, budgets, feature definitions, split rules and hashes |
| `input_audit.json` | Accepted/rejected sources and precise reasons |
| `discovery_manifest.json` | Source/family weighting, fit folds and provenance |
| `candidate_registry.json` | Candidate definitions and parameter semantics |
| `shortlist.csv` | All candidate scores and exclusions, including the exploration-slot draw |
| `evaluation_panel.json` | Shared clusters, bins, observables, tolerances, support rules |
| `models/<pair_hash>.json` | Immutable deployable pair model |
| `trials/<arm>/trial_manifest.json` | Model/state/envelope identities and seed cohorts |
| `trials/<arm>/windows.csv` | Concrete sparse window table |
| `observations/*.parquet` | Exact-key structural observations aligned with sampling records |
| `estimates/*.parquet` | Probabilities, expectations, supported ΔF values, uncertainty and support statuses |
| `uncertainty.json` | Block lengths, resampling policy, solver failures and coverage diagnostics |
| `budget_ledger.json` | Actual resource usage by arm and phase |
| `decision.json` | Verdict, reason codes, confidence scope, comparison and confirmation status |
| `report.md` | Concise human-readable evidence and remaining limitations |

`decision.json` must distinguish `selected_pair_id`, `recommended_sampler`, `confirmation_passed`, `advantage_demonstrated`, `search_mode`, and `scope_of_equilibrium_claim`. A validated pair may exist without being the recommended sampler.

## 13. Proposed configuration example

The following is a **new selector schema**, not a YAML file supported by current `gareus` commands. Explicit unit suffixes and paths are part of the proposed contract. Replace input/model paths with resolved artifacts before execution.

```yaml
schema_version: equilibrium-cv-selection-v1
search_mode: joint_pair
target:
  sequence: GYDPETGTWG
  temperature_K: 300.0
  ensemble: NPT
  system_manifest: inputs/target_system.json
  require_common_physical_hamiltonian: true

blindness:
  generator_preset: broad
  require_native_blind_allowlist: true
  historical_data_role: exploratory_only

discovery:
  design_measure: balanced_source_and_family
  grouped_folds: 4
  torsion_components_1based: [1, 2, 3, 4, 5, 6]
  residual_polynomial_degrees: [1, 2]
  max_primitive_candidates: 32
  feature_order: interleaved_sin_cos_phi_then_psi

shortlist:
  max_pairs: 3
  ranked_distinct_families: 2
  exploration_slots: 1
  deterministic_seed: 20260919
  retain_incumbent_1d_if_distinct: true

sampling:
  common_frozen_envelope: required
  lambda_rungs: 4
  spatial_states: 24
  max_replicas: 128
  freeze_cv_windows_and_envelope_during_measurement: true
  measure_both_cvs_in_every_state: true
  synchronized_structural_observations: true
  forbid_forward_fill_for_scoring: true
  screening_campaigns_per_arm: 3
  confirmation_campaigns_per_arm: 4

resources:
  gpus: 4
  gpu_model: L40S
  cpu_thread_cap: 192
  analysis_memory_GiB: 32
  bootstrap_workers: 4
  shared_discovery_gpu_hours: 24
  screening_gpu_hours_per_campaign: 8
  confirmation_gpu_hours_per_campaign: 16
  contingency_gpu_hours: 40
  total_gpu_hour_ceiling: 288
  budget_mismatch_policy: error

evaluation:
  coarse_cells: 8
  fine_cells: 24
  relevant_population_threshold: 0.01
  probability_ci_halfwidth: 0.02
  torsion_mean_tolerance: 0.05
  contact_mean_tolerance: 0.03
  histogram_tv_tolerance: 0.05
  basin_deltaF_tolerance_kcal_mol: 0.3
  require_shared_out_of_pair_observables: true

uncertainty:
  method: synchronized_ensemble_blocks_plus_independent_campaigns
  refit_mbar_each_resample: true
  screening_resamples: 200
  confirmation_resamples: 500
  min_blocks_per_campaign: 20
  block_doubling_relative_tolerance: 0.20
  confirmation_max_failed_resample_fraction: 0.01
  confidence: 0.95
  simultaneous_primary_intervals: true

decision:
  require_absolute_precision: true
  require_independent_confirmation: true
  practical_loss_ratio: 0.80
  superiority_upper_confidence_bound: 1.0
  budget_exhaustion: INSUFFICIENT_EVIDENCE
```

The budget validator computes the number of arms after shortlisting. If the extra incumbent creates five arms, the example ceiling is insufficient for the unchanged allocations; the planner must produce a budget-consistent manifest before running. It must not silently exceed the ceiling.

Proposed CLI sequence:

```bash
gareus-select-cvs audit --protocol selection.yaml
gareus-select-cvs fit --protocol selection.yaml
gareus-select-cvs shortlist --protocol selection.yaml
gareus-select-cvs plan-trials --protocol selection.yaml
gareus-select-cvs run --trial-plan trial_plan.json
gareus-select-cvs evaluate --trial-plan trial_plan.json
gareus-select-cvs confirm --decision decision.json
gareus-select-cvs report --decision decision.json
```

`plan-trials` is side-effect-free with respect to simulations. `run` executes only a concrete, validated trial manifest when the user has requested execution. Resume reuses the manifest and hashes; it does not refit or adapt parameters implicitly.

## 14. Implementation milestones and dependencies

| Milestone | Deliverable | Required exit evidence |
|---|---|---|
| M0 — Freeze scientific protocol | Target manifest, budgets, observables, candidate semantics and validation controls | Reproducible protocol hash; no native-reference dependency |
| M1 — Reliable observation identity | Exact-key export, completeness report, inactive-axis measurement and strict masks | Resume/duplicate/forward-fill fixtures cannot masquerade as independent observations |
| M2 — Canonical models and force compiler | Explicit individual PCs, residual polynomials, pair transform and runtime model import | Energy/force/reprojection agreement, including feature permutation and λ=0 |
| M3 — Offline selector | Grouped discovery fitting, joint pair enumeration, shortlist and baseline registry | Synthetic redundancy/noise controls; deterministic split/shortlist; no held-out fitting |
| M4 — Frozen trial runner | Sparse windows, common calibration, bounded campaigns, immutable measurement phases | Small integration run preserves state/envelope/CV hashes across exchange and restart |
| M5 — Equilibrium evaluator | MBAR estimates, synchronized bootstrap, support/equivalence gates | Known-distribution recovery and calibrated interval coverage; injected traps rejected |
| M6 — Restricted chignolin pilot | `fixed_primary` run if useful to validate integration early | Correctly scoped companion-selection report; no claim of joint optimum |
| M7 — Joint selection and confirmation | Full tested pair comparison, new confirmation campaigns, decision/report | Final verdict meets the frozen rule or states an explicit insufficiency |
| M8 — Transfer validation | Same frozen selection algorithm on a declared additional peptide panel | Report transfer success/failure; no per-peptide threshold retuning hidden from the record |

Dependency order: M0 → M1/M2 → M3 → M4/M5 → M6 or M7. Joint mode is the requested completion target; M6 is optional. M8 is necessary for a general methodological claim, not for delivering a chignolin-specific decision.

Suggested review-sized change sets are M1 (data integrity), M2 (model/runtime), M3 (discovery), M4 (runner), and M5 (statistics/decision). Keep the existing heuristic command, legacy PCA setting, and non-selector production defaults backward compatible.

Do not run a costly CV competition until M1, M2, and the core M5 synthetic gates pass. Otherwise an apparent CV advantage could be a projection, data-join, or reweighting defect.

## 15. Verification plan

Tests are warranted here because the changes affect forces, thermodynamic weights, uncertainty and restart behavior. Prefer tests against mathematical/physical reference results over tests that repeat implementation formulas.

### 15.1 Model, force and storage tests

| Test | Failure it must expose |
|---|---|
| Individual PC versus leading-PC mixture | `component_index=2` accidentally reproduces a mixture of PC1 and PC2 |
| Known feature permutation | Handoff’s blocked sine/cosine order imported into runtime’s interleaved order without conversion |
| Polynomial residual on a known analytic system | Missing CV1 linear/quadratic terms or missing chain-rule forces |
| Scale/sign/axis transformations | Wrong transformed centers/stiffness/units; scores depending on arbitrary PCA sign |
| Finite-difference bias gradients | Runtime force disagrees with the serialized coordinate definition |
| PBC wrapping and peptide reconstruction | CV changes under equivalent box wrapping or a torsion atom-map error |
| λ=0 and inactive-axis states | Nonzero unintended boost or required CV left unmeasured |
| Cross-state reconstruction audit | Runtime bias differs from the canonical MBAR matrix, including zero-k rows |
| Force-group isolation | Umbrella energy leaks into a physical GaMD channel |
| Interrupted-write/resume fixture | Duplicated or uncommitted observations become extra evidence |
| Missing structural rows | Forward-fill or conformation-dependent row loss silently passes |
| Changed model/envelope on resume | Old state label reused for a new Hamiltonian |

Set initial Reference/double-precision coordinate and energy tolerances from analytical test systems; use scale-aware absolute plus relative tolerances. A reasonable starting target is relative `1e-6` for nonzero energies and `1e-4` for force finite differences, with explicit absolute floors near zero. Calibrate tighter/looser CUDA tolerances from numerical precision tests, not from chignolin’s desired population.

Existing regression suites to extend include `test_bootstrap_torsion_cv.py`, `test_cv2_reprojection.py`, `test_bias_reconstruction_nan_handling.py`, `test_parquet_manifest_transactions.py`, `test_union_mbar_per_epoch_bias.py`, `test_pmf_bootstrap_uncertainty.py`, and the `test_thermodynamic_validity_*` suites. Reuse `gareus/synth/` and its oracle infrastructure where compatible.

### 15.2 Thermodynamic and statistical controls

1. **Known 2D double well:** quadrature supplies exact basin probabilities. Useful pairs must recover them within declared uncertainty.
2. **Hidden orthogonal barrier:** a smooth, stable primary marginal coexists with incorrect populations in an omitted coordinate. The structural gate must reject the misleading pass.
3. **High-variance fast nuisance:** structural PCA alone favors an irrelevant degree of freedom. The equal-cost thermodynamic trial must be free to prefer another pair.
4. **Linear and nonlinear duplicates:** `(x, 2x)` and `(x, x²)` must not be treated as two independent local directions merely because covariance vanishes.
5. **Same potential, different sampling mobility:** alter friction/mobility without changing the invariant target. Preferred sampling coordinates may change; recovered equilibrium populations must not. This directly tests that physical kinetics are not the objective.
6. **Three-dimensional competing barriers:** no tested pair suffices within budget. Expect an honest insufficiency outcome.
7. **Trapped independent campaigns:** same apparent PMF stability, different seeded populations. Equivalence/support gates must fail.
8. **Disconnected MBAR bridge:** numerical solver convergence cannot create a valid global population ratio.
9. **Heavy-tailed weights:** a large global sample count must not hide a cell controlled by one or two observations.
10. **Wrong boost sign/double correction:** inject both defects and require failure against the known equilibrium distribution.
11. **Adaptation contamination:** adaptation-phase samples cannot enter a fixed-phase score by changing a metadata label.
12. **Uncertainty coverage:** across repeated cheap independent synthetic experiments, check actual simultaneous coverage and false promotion rates. Freeze thresholds only after these tests pass.

For the last test, start with 200 cheap experiments at the claimed 95% confidence level. Evaluate coverage using a binomial uncertainty interval and a predeclared tolerated undercoverage margin, rather than requiring exactly 190 successes. Increase the count only if the pass/fail conclusion remains unresolved.

An all-atom engineering check then verifies compiled forces, bias energy reconstruction, exchange/NPT integration and numerical stability on the intended system. This is not a proof of its equilibrium convergence. A larger-than-reference timestep needs a separate equilibrium-distribution comparison before attributing numerical bias to the CV selector.

## 16. Computational details and failure recovery

### 16.1 Analysis scaling

- Materializing an `N × K` float64 matrix costs `8NK` bytes: at 5 million rows and 96 states, about 3.84 GB before work arrays. Stream or memory-map large matrices and cap concurrent bootstrap workers.
- The 36 chignolin sine/cosine features require about 720 MB for 5 million float32 rows, before metadata/compression. Computing them at sample cadence need not require writing a full trajectory at the same cadence.
- Compute generic observables together from each retrieved configuration to avoid repeated coordinate transfers. Record actual reporter overhead; it is included in all arm timings.
- Avoid launching 192 BLAS threads inside every bootstrap worker. Divide a fixed CPU/memory budget across workers.
- For bootstrap efficiency, represent resampled blocks by row multiplicities rather than copying complete energy matrices. The MBAR solver must include those multiplicities in its estimating sums and origin-state counts. Validate against literal row replication on small problems before use.
- Keep identical data eligibility and observation ordering across point estimates, uncertainty refits and displayed plots.

### 16.2 Recovery rules

| Condition | Allowed next action |
|---|---|
| Corrupt identity, wrong bias, wrong force | Repair implementation; invalidate affected results; rerun required engineering gates |
| A pair is numerically unstable | Exclude with a recorded reason; do not lower its stiffness silently outside the declared setup policy |
| Pilot support is weak for all arms | Use the bounded discovery/extension reserve consistently; otherwise return insufficiency |
| Pair score improves but absolute precision remains poor | Report a promising provisional pair, not a confirmed thermodynamic result |
| Baseline succeeds and pairs add no clear benefit | Recommend the baseline; retain pair diagnostics |
| No second CV helps hidden structural sampling | Consider a separate CV-independent exploration protocol such as temperature or multicanonical sampling, with the corresponding valid reweighting model |
| New relevant structure appears | Start a revised discovery/evaluation version; keep old results and reasons |
| Confirmation fails | Report failure; a replacement winner requires fresh confirmation |

Historical folding-event counts may remain useful descriptive evidence of trapping. They are not numerical acceptance targets. In particular, reproducing a folding-event count from a different model/temperature is not required for equilibrium validity.

The handoff’s GaMD mechanism explanation also needs care: zero boost at a transition region does not imply zero acceleration. Raising the reactant well relative to that region can reduce the effective barrier. This specification therefore tunes/assesses the actual sampling and weight support, not a claimed speedup derived from the mean boost alone.

## 17. Acceptance criteria for the completed feature

The implementation is complete when:

1. A single protocol can generate, fit, enumerate, test and confirm a pair without native inputs or manual component selection.
2. Joint mode actually considers both coordinates; constrained mode is clearly labeled.
3. Fitted coordinates, runtime forces, measured CVs and cross-state bias reconstruction describe the same functions.
4. Independent campaigns and synchronized uncertainty analysis determine thermodynamic precision; preserved physical kinetics are never a gate.
5. The winner is evaluated on shared structural distributions beyond its own two coordinates.
6. State identity, complete observations, exact ladder reweighting and strict failure handling survive restart and analysis.
7. Budget exhaustion, hidden support failures, no useful second CV and failed confirmation are first-class outcomes.
8. The final report distinguishes implemented/verified behavior, measured results, provisional inferences and unresolved limitations.

For the first chignolin deployment, success means producing a defensible selection decision. It does not mean guaranteeing a particular PC, a folded-state percentage, or convergence within the example budget.

## 18. Methodological sources and evidence limits

- [Shirts and Chodera, MBAR](https://arxiv.org/abs/0801.1426): equilibrium estimates from multiple ensembles, normalization, reduced potentials and uncertainty foundation.
- [Li, Van Koten, Dinner and Thiede, correlated-sampling error in MBAR](https://arxiv.org/abs/2203.01227): sampling correlations contribute materially to estimator uncertainty. The synchronized replica-exchange bootstrap specified here is a proposed implementation choice, not a claim copied from that paper.
- [PyMBAR MBAR API](https://pymbar.readthedocs.io/en/stable/mbar.html) and [timeseries API](https://pymbar.readthedocs.io/en/master/timeseries.html): reference diagnostics and correlation handling. Pin actual package versions in execution manifests.
- [OpenMM CustomCVForce](https://docs.openmm.org/latest/api-python/generated/openmm.openmm.CustomCVForce.html): force-based collective variables and algebraic bias construction.
- [ATLaS-MD handoff](https://github.com/sulcjo/ATLaS-MD/blob/cdee6cc87ac540b0bac9b230f389c77d70383faf/smoke/README.md), [PMF diagnostic caveats](https://github.com/sulcjo/ATLaS-MD/blob/cdee6cc87ac540b0bac9b230f389c77d70383faf/smoke/pmf_analysis/README.md), [estimator exclusions](https://github.com/sulcjo/ATLaS-MD/blob/cdee6cc87ac540b0bac9b230f389c77d70383faf/gareus/mbar_analysis/estimators.py), [strict bias reconstruction](https://github.com/sulcjo/ATLaS-MD/blob/cdee6cc87ac540b0bac9b230f389c77d70383faf/gareus/correctness/bias.py): repository-specific constraints checked for this specification.

The shortlist score, window allocation, resource budget, thresholds, uncertainty implementation and decision policy in this document are proposed engineering/scientific design choices. They require the stated control tests and prospective molecular experiments. No literature source establishes that this complete selector is already validated.
