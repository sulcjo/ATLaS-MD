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

## How the pieces fit

| Stage | What you choose or get |
| --- | --- |
| Prepare | Peptide sequence or structure, solvent/system settings, and optional GENPEPT seeds |
| Define states | One or two CVs, umbrella centers and force constants, temperature, and supported acceleration settings |
| Sample | OpenMM trajectories, replica-exchange proposals, checkpoints, and phase-specific state maps |
| Analyze | MBAR/PMF inputs plus overlap, effective-sample-size (ESS), and reweighting diagnostics |

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
