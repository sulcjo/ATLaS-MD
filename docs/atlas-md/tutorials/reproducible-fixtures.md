# Reproducible tutorial fixtures

Tutorials need two proof levels:

1. **Interface fixture:** commands/config parse and resolve without OpenMM/GPU.
2. **Runtime fixture:** tiny real MD only on configured machine.

Never present interface fixture as molecular-dynamics validation.

## Maintained fixture matrix

| Tutorial path | Fixture | What test proves | What it cannot prove |
| --- | --- | --- | --- |
| First CMD workflow | `gareus-test-run --dry-run --run-mode cmd` | Resolved tiny `gareus` command is constructible | OpenMM run, force-field stability, PMF validity |
| YAML configuration | `examples/*.yaml` + `tests/test_example_configs.py` | Shipped examples parse without unknown keys | Hardware/dependency availability, sampling quality |
| Main CLI reference | `gareus -h`, `gareus -hh list` | Help surfaces exist | Every scientific combination is valid |
| MBAR entry point | `gareus-analyze --help` | Installed console script resolves its analyzer module | Analysis of a particular run |
| Adaptive policy | `python -m gareus.synth ...` | Real decision functions run against oracle data | Peptide/OpenMM physics |

## Copyable no-MD checks

```bash
# Prints tiny conventional-MD command; does not execute MD.
gareus-test-run --dry-run --run-mode cmd

# Resolve shipped full-workflow YAML. Parsing only; no trajectory starts.
python -c 'from gareus.cli import parse_args; parse_args(["--config", "examples/chignolin_fulltreatment2.yaml"])'

# Run adaptive policy against analytic landscape; writes metrics JSON to stdout.
python -m gareus.synth --landscape mixture-wells --mode exact \
  --subsystem feedback --rounds 1 --samples-per-window 100 --res 40 --seed 0
```

Expected outcomes:

- Dry run prints a `gareus ...` command and exits 0.
- Config parser exits 0 with no unknown-key error.
- Synth command prints JSON metrics; dispatcher diagnostics appear on stderr.

## Real-MD fixture

Run only where optional MD stack/platform is installed:

```bash
gareus-test-run --check-deps --skip-if-missing
gareus-test-run --run-mode cmd --out tiny_cmd --force
```

Second command creates/replaces `tiny_cmd`; it performs MD. It validates tiny workflow plumbing, not converged free-energy science. Follow [PMF health field guide](../analysis/pmf-health-field-guide.md) for production interpretation.

## CI contract

Run lightweight fixtures with:

```bash
pytest -q tests/test_atlas_md_docs.py tests/test_example_configs.py
gareus-test-run --dry-run --run-mode cmd
python -m gareus.synth --landscape mixture-wells --mode exact \
  --subsystem feedback --rounds 1 --samples-per-window 100 --res 40 --seed 0
```

Add a fixture whenever tutorial gains command, YAML, or expected artifact claim. Keep fixture deterministic with explicit seed and bounded sample count.
