## Summary

Describe the change and why it is needed.

## Scientific impact

- [ ] No thermodynamic / methodological behavior changes
- [ ] Changes sampler / target distribution
- [ ] Changes exchange / proposal kernel
- [ ] Changes NPT / GaMD energy bookkeeping
- [ ] Changes estimator / reweighting
- [ ] Changes adaptive-state bookkeeping

If any scientific behavior changes, state the intended target distribution or invariant explicitly.

## Validation

List tests run and any independent oracle or analytic comparison used.

```text
pytest ...
```

- [ ] Fast tests pass
- [ ] Relevant exact-kernel / analytic tests pass
- [ ] Docs build with `python -m mkdocs build --strict`
- [ ] User-visible behavior is documented

## Reproducibility / compatibility

Note any impact on checkpoints, output schema, state IDs, manifests, or prior runs.
