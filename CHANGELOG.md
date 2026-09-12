# Changelog

All notable changes to ATLaS-MD are documented here.

The project follows semantic-style release numbering where practical. Research-method changes that alter a sampled Hamiltonian, estimator, output contract, or thermodynamic assumption should be called out explicitly even when backwards compatibility is retained.

## [0.7] — 2026-09-12

### Added

- Explicit thermodynamic target and detailed-balance contract for NVT, NPT, replica exchange, Gibbs-walk proposals, and GaMD lambda ladders.
- Application-controlled NPT Metropolis volume moves for supported boosted Hamiltonians.
- Stage-aware effective-potential adapters that separate physical, bias, auxiliary, and GaMD boost energies.
- Exact finite-state exchange-kernel validation that distinguishes elementary detailed balance from stationary invariance of composed sweeps.
- State-by-configuration GaMD boost cross-evaluation for lambda/Hamiltonian replica exchange.
- Adaptive-production and double-adaptive workflows with phase-local state provenance.
- Parquet-backed sample/exchange storage and query tooling.
- Modernized documentation and project landing page.

### Correctness

- Native OpenMM barostat use is rejected for supported boosted paths where its acceptance energy would not match the propagated effective potential.
- Gibbs-walk proposals include the reverse/forward proposal correction required by Metropolis-Hastings.
- NPT volume acceptance uses the molecular Jacobian convention consistent with rigid-molecule isotropic scaling.
- Adaptive phases are tracked as distinct Hamiltonian epochs rather than silently pooled under reused local window identifiers.

### Validation scope

ATLaS-MD does not claim mathematically exact finite-timestep propagation. Exchange and NPT transition-kernel correctness are tested separately from numerical integration accuracy and from post-hoc reweighting validity.

[0.7]: https://github.com/sulcjo/ATLaS-MD/releases/tag/v0.7
