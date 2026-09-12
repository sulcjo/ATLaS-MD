# Contributing to ATLaS-MD

ATLaS-MD is research software. Contributions are welcome, but changes that affect sampled Hamiltonians, exchange probabilities, barostat behavior, reweighting, or state bookkeeping require a higher standard of validation than ordinary refactoring.

## Before opening a change

For user-facing behavior or methodological changes, please open an issue first and describe:

- the scientific or operational problem;
- the affected run mode(s), ensemble, CVs, and exchange mode;
- whether the change alters the sampler, estimator/reweighting, or only presentation/analysis;
- the expected invariant or mathematical relation the implementation should preserve.

## Development setup

```bash
python -m pip install -e ".[all]"
pytest
python -m mkdocs build --strict
```

For changes that do not require the MD stack, `pip install -e ".[dev]"` is sufficient for the fast test suite.

## Scientific correctness checklist

Changes that touch thermodynamic logic should answer all of the following explicitly:

1. What is the target distribution before the change?
2. What is the target distribution after the change?
3. Which state-dependent terms enter the reduced potential?
4. Does the proposal remain symmetric? If not, where is the Hastings ratio applied?
5. Are adaptation stages being treated as distinct Hamiltonians?
6. Does checkpoint/resume preserve all state needed to continue the same Markov process?
7. Is the new behavior tested with an independent oracle rather than by reusing the production helper under test?

See `docs/atlas-md/guide/thermodynamic-validity.md` for the current thermodynamic contract.

## Tests

At minimum, run:

```bash
pytest
python -m mkdocs build --strict
```

When changing exchange logic, also run the exact transition-kernel suite:

```bash
pytest tests/test_exchange_kernel_exact.py tests/test_gibbs_walk.py
```

When changing boosted NPT or GaMD bookkeeping, include the relevant NPT/GaMD tests and, where possible, a real OpenMM smoke test.

## Pull requests

Keep pull requests focused. Separate scientific-method changes from unrelated cleanup when practical.

A pull request should include:

- a concise explanation of the change;
- the affected thermodynamic assumptions, if any;
- tests added or updated;
- any migration or output-format implications;
- documentation changes for user-visible behavior.

## Style

Prefer explicit state and energy bookkeeping over implicit inference. Favor readable equations and named intermediate quantities over compressed cleverness in code paths that determine ensemble correctness.
