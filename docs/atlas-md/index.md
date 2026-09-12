<p align="center">
  <img src="assets/atlas-md-logo.svg" alt="ATLaS-MD" width="720">
</p>

<p align="center"><strong>Adaptive Thermodynamic Landscape Sampling for molecular dynamics</strong></p>

ATLaS-MD is an OpenMM-based peptide sampling framework for explicit-solvent umbrella sampling, replica exchange, optional Gaussian accelerated MD (GaMD/Pep-GaMD), adaptive state placement, and MBAR/PMF analysis.

It is research software built around an explicit scientific contract: sampled Hamiltonians are state-defined, exchange kernels are independently validated, adaptive phases retain their own provenance, and estimator validity is kept separate from sampler correctness.

<p align="center">
  <img src="assets/workflow.svg" alt="ATLaS-MD workflow" width="100%">
</p>

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
