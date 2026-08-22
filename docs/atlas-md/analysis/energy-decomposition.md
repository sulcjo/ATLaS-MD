# Energy decomposition

`gareus-energy-decompose` evaluates saved coordinates into direct nonbonded peptide terms.

```bash
gareus-energy-decompose --run-dir run_cln025 \
  --out run_cln025/energy_decomposition.csv
```

Optional peptide-environment and whole-system force-field energies cost more:

```bash
gareus-energy-decompose --run-dir run_cln025 \
  --include-peptide-environment --write-total-forcefield-energy
```

Use `gareus-energy-decompose -hh` for equations, grouping definitions, and caveats. This analysis is coordinate/post-processing based; it does not substitute free-energy reweighting or PMF validation.
