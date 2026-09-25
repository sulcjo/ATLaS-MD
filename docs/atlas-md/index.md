<p align="center">
  <img src="assets/atlas-md-logo.svg" alt="ATLaS-MD" width="720">
</p>

<p align="center"><strong>Adaptive Topological Landscape Sampling for Molecular Dynamics</strong></p>

ATLaS-MD (Adaptive Topological Landscape Sampling) is an OpenMM-based workflow for exploring peptide conformational landscapes. It combines explicit-solvent molecular dynamics, collective-variable (CV) umbrella states, replica exchange, optional Gaussian accelerated MD (GaMD/Pep-GaMD), adaptive state placement, and MBAR/PMF analysis.

In a typical run, you prepare a peptide, choose one or two CVs, construct umbrella states, propagate replicas and attempt exchanges, then analyze the recorded samples against their state definitions. Optional adaptive phases can change the state layout; the output keeps those phases and their provenance explicit. ATLaS-MD supplies the simulation workflow and analysis tools, but a successful run alone does not establish convergence or a valid free-energy estimate.

It is research software built around an explicit scientific contract: sampled Hamiltonians are state-defined, exchange kernels are independently validated, adaptive phases retain their own provenance, and estimator validity is kept separate from sampler correctness.

<p align="center">
  <img src="assets/workflow.svg" alt="ATLaS-MD workflow" width="100%">
</p>

## Standard workflow

1. **GENPEPT:** generate and rank candidate conformers.
2. **Swarm:** run parallel exploratory simulations and collect sampling evidence.
3. **CV selection:** choose collective variables (CVs) using the seed-bank and swarm evidence.
4. **Adaptive epochs:** refine the umbrella-state layout and allocate sampling using epoch feedback.
5. **Optional top-ups:** extend the run under the current regime when more sampling is needed, preserving phase and state provenance.
6. **MBAR:** analyze the recorded states and samples, then inspect PMF, overlap, effective sample size (ESS), and reweighting diagnostics.

This is the advertised end-to-end route; ATLaS-MD also supports simpler conventional-MD and manually specified-window runs.

## Performance improvements

| Optimization | Evidence and scope |
| --- | --- |
| Shared contact-sum calculation for CV1 and residual-torsion CV2 | **+14.6% node throughput** in a 236-context MPS production A/B, with equivalent bias energy and forces. |
| Fewer Pep-GaMD force evaluations and barostat state reads | **~1.26× campaign throughput** in a measured in-campaign A/B. |
| CUDA MPS for a high replica count | **2,307 vs 840 aggregate ns/day** at 236 replicas / 59 contexts per L40S, MPS on vs off (**2.75×** for that tested workload). |
| Multi-GPU umbrella pulls | `--us-pull-device-index` spreads setup workers across GPUs; this affects startup work, not production MD step rate. |
| PME stream choice | +8–11% in a no-MPS test; about −4% when enabled at 236 contexts with MPS. Measure against the intended regime. |

See the [detailed throughput benchmark record](developer/gpu-throughput-benchmark-todo.md). These results are hardware- and workload-specific, not general speedup guarantees.

The intended target and exactness limits are described in the [thermodynamic validity guide](guide/thermodynamic-validity.md); the [PMF validity guide](analysis/pmf-validity.md) covers checks required before interpreting a result.

## Start here

1. [Install](start/installation.md) dependencies.
2. Run the [quickstart](start/quickstart.md) with conventional MD.
3. Choose [collective variables](guide/collective-variables.md) and [umbrella/exchange states](guide/windows-and-exchange.md).
4. Read the [thermodynamic target and detailed-balance contract](guide/thermodynamic-validity.md) before modifying sampler logic.
5. Apply the [PMF validity](analysis/pmf-validity.md) gates before interpreting free energies.

## What ATLaS-MD combines

| Layer | Purpose |
| --- | --- |
| System setup | Explicit-solvent OpenMM peptide systems, equilibration, HMR options |
| State construction | 1D/2D umbrella states, adaptive placement, sparse state layouts |
| Acceleration | GaMD / Pep-GaMD and optional state-dependent lambda ladders |
| Exchange | Neighbor, random-pair, all-pair and MH-corrected Gibbs-walk kernels |
| Production | NVT or supported boosted-NPT workflows with checkpoint/resume |
| Storage | Phase-local state maps, Parquet samples, exchange ledgers, manifests |
| Inference | MBAR/PMF analysis, overlap/ESS diagnostics and validity gates |

## Scientific scope

ATLaS-MD writes reproducibility artifacts and diagnostics; it does not make a free-energy estimate valid merely by completing a run. Validate sampling, overlap, effective sample size, reweighting stability, and estimator assumptions for every scientific conclusion.

The code distinguishes:

- the **intended target distribution**;
- **transition-kernel correctness** for exchange and supported NPT moves;
- **finite-timestep propagation accuracy**;
- **post-hoc estimator/reweighting validity**.

Those are related, but they are not interchangeable claims.

## Command map

| Command | Purpose |
| --- | --- |
| `gareus` | Main setup, equilibration, production, REUS, adaptive workflow |
| `python GENPEPT.py` | Generate and rank peptide seed conformers |
| `gareus-analyze` | MBAR/PMF analysis package entry point |
| `gareus-suggest-cvs` | Post-hoc CV/window suggestions |
| `gareus-energy-decompose` | Coordinate-based energy decomposition |
| `gareus-test-run` | Dependency checks and tiny end-to-end workflow |

See the [command reference](reference/cli.md) for the exact interface.
