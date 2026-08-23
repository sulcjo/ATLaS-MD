# Validation workflow without production examples

This workflow gives reproducible checks before a production campaign exists. Outputs are synthetic/interface evidence, not peptide free-energy results.

## Stage 1: interface and schema

```bash
pytest -q tests/test_atlas_md_docs.py tests/test_example_configs.py
gareus-test-run --dry-run --run-mode cmd
gareus-analyze --help
```

Expected: docs/config contracts pass, dry-run prints resolved command, analyzer help resolves. None runs MD or validates physics.

## Stage 2: analytic adaptive oracle

```bash
python -m gareus.synth --landscape mixture-wells --mode exact \
  --subsystem feedback --rounds 1 --samples-per-window 100 --res 40 --seed 0
python -m gareus.synth --landscape mixture-wells --mode exact \
  --subsystem production --rounds 2 --samples-per-window 100 --res 40 --seed 0
```

`exact` draws independent samples from known biased equilibrium density `exp[-β(F+U_bias)]`. Use this to test window placement, bridge proposals, retirement safeguards, and oracle PMF recovery. It excludes solvent, molecular force field, OpenMM integration, GaMD, and trajectory correlation.

## Stage 3: estimator sanity

Run focused numerical contracts before changing analysis:

```bash
pytest -q tests/test_cumulant_expansion.py tests/test_diagnostics.py \
  tests/test_pmf_bootstrap_uncertainty.py tests/test_thermodynamic_validity_2d.py
```

These protect formula implementation: cumulant terms, per-window versus pooled ESS, block bootstrap, 2D bias/reweighting. They do not prove target peptide samples equilibrium distribution.

## Stage 4: later production gate

When production exists, use this order:

1. Confirm final window registry and fresh frozen-final population.
2. Run MBAR with exact temperature, full restraints, selected segment/population.
3. Read `pmf_summary.json`: coverage, `mbar.n_k`, overlap, base ESS, GaMD diagnostics.
4. Add block-bootstrap uncertainty and compare estimator outputs.
5. Record acceptance decision and limits; only then interpret PMF.

Do not turn synthetic values into acceptance thresholds. Thresholds must be declared for physical system before final interpretation. See [PMF validity](pmf-validity.md).
