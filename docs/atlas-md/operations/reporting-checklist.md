# Scientific reporting checklist

Use this checklist for every PMF, including exploratory results marked invalid.

## System and protocol

- Sequence/input structure, protonation/caps, force field, water model, ions, box, PME/cutoff/constraints.
- Temperature, pressure, timestep, HMR status, thermostat/barostat, OpenMM platform/precision.
- CV definitions, atom selections, units, bin edges, all window centers/constants, exchange mode/interval.
- Run mode, GaMD boost type, sigma targets, calibration reconstruction settings, boost convergence.
- Adaptive policy, epoch schedule, registry actions, final-pool boundary, final frozen layout.

## Estimator and population

- Exact analysis command, source revision, package/environment versions, and input artifact hashes.
- Selected segments/epochs, discarded equilibration, frame stride, sample count, window counts `N_k`.
- `β`/temperature, MBAR backend/tolerance, estimator (`umbrella_only`, `gamd_exponential`, `gamd_cumulant2`, or `gamd_cumulant3`).
- PMF reference convention, binning, smoothing, bootstrap block definition/count/seed.

## Evidence and decision

- Coverage: all intended final states sampled; explain every zero count.
- Overlap: worst neighboring edge and full 2D support where relevant.
- Weight quality: `base_ess`, ESS fraction, GaMD/reweighting diagnostics, boost anharmonicity.
- Uncertainty: finite bins, bootstrap spread, estimator sensitivity, convergence versus added sampling.
- Verdict: PASS/CAUTION/FAIL, thresholds, observed values, repair steps, remaining limits.

Archive `effective_config.*`, manifest, segment registry, window tables/snapshots, samples/exchanges, `pmf_summary.*`, PMF/overlap/boost/uncertainty files, and this completed record. Completion means outputs written; it does not mean PMF valid.
