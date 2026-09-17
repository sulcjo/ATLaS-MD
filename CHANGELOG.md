# Changelog

All notable changes to ATLaS-MD are documented here.

The project follows semantic-style release numbering where practical. Research-method changes that alter a sampled Hamiltonian, estimator, output contract, or thermodynamic assumption should be called out explicitly even when backwards compatibility is retained.

## [0.8] — 2026-09-18

### Correctness

- `ladder_active` is now a property of the **campaign**, not of one sub-run. It was
  derived from `args.state_gamd_lambdas`, which only ever describes the states handed
  to a single adaptive sub-run. A top-up given nothing but lambda=0 states therefore
  concluded "no ladder here", with two consequences: `k0max_by_channel` stayed `None`
  so `set_replica_lambda_for_window()` no-oped and every replica **kept the shared
  calibration's full `k0`** instead of `lambda*k0max = 0` — running boosted while
  recorded as lambda=0 — and `pep_env` stayed `None` so `v_pep`/`v_dih` were written
  as NaN. `resolve_ladder_active()` now also consults the campaign state registry,
  and the parent walk is bounded at `adaptive_production/` so one campaign cannot
  switch on another's boost recording. Plain umbrella/REUS and plain GaMD are
  deliberately unaffected: the ladder path costs two extra Context energy reads per
  logged sample.
- Observed in the chignolin_7 campaign, where three lambda=0-only top-ups carried a
  mean recorded boost of 9.44-10.60 kJ/mol against exactly 0.000 for lambda=0 samples
  from mixed sub-runs and 9.57-10.16 for the genuine lambda=1 rung. The MBAR loader's
  existing refusal to fabricate a 0.0 boost for samples lacking raw channel energies
  excluded those 992,896 samples, which was protective rather than lossy.

### Performance

- Removed a redundant all-groups force evaluation from the Pep-GaMD integrator by
  deriving the bias force groups as the complement of the boosted set. Measured at
  22.7% of MD step cost by controlled A/B on the real 19,008-atom system.
- The application-controlled barostat reads positions and box vectors from a single
  `State` instead of two.
- Together ~1.26x campaign throughput, confirmed by in-campaign A/B on identical
  hardware.

### Known issues

- `tests/test_package_smoke.py::test_tiny_lambda_ladder_run_completes_end_to_end_slow`
  fails with `pyarrow.lib.ArrowInvalid: Could not open Parquet input source`. Present
  in 0.7 as well; unrelated to the changes above.

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
