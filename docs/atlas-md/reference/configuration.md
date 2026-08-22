# Configuration

GAREUS accepts YAML or JSON via `--config`. Generate starter YAML:

```bash
gareus --write-config-template chignolin.yaml
```

Precedence: explicit CLI flags override YAML/JSON values; named `--profile` supplies boilerplate before explicit overrides. Save `effective_config.yaml`/`.json` from every run; it is source of truth for resolved settings.

Existing working examples:

- `examples/chignolin_2d_distance_with_genpept.yaml`: combined seed generation + 2D workflow.
- `examples/chignolin_fulltreatment2.yaml`: contacts × Ramachandran workflow.
- `examples/chignolin_runs3.yaml`: double-adaptive torsion-PCA → tICA workflow.

Use units from CLI help: temperature K, pressure bar, lengths nm or Å as named, time fs, force constants kcal/mol/CV², and adaptive budget ns. Never copy a value without its option-specific unit.
