# Testing and contributing

Run focused checks before broad suite:

```bash
pytest -q tests/test_atlas_md_docs.py
python -m mkdocs build --strict
python -m py_compile gareus/*.py GENPEPT.py
pytest
```

Documentation changes must keep nav, local links, commands, units, and default claims accurate. Add tests when doc tooling behavior changes. Do not convert an output-completion message into an unsupported scientific-validity claim.
