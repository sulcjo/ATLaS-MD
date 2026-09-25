# Installation

ATLaS-MD (the `gareus` command and Python package) requires Python 3.10+. Choose extras matching your task. Clone the current repository and install only the dependency groups you need:

```bash
git clone https://github.com/sulcjo/ATLaS-MD.git
cd ATLaS-MD
python -m pip install -e ".[dev]"       # parser, tests, lightweight tooling
python -m pip install -e ".[all]"       # OpenMM, config, MBAR/storage stack
python -m pip install -e ".[docs]"      # ATLAS-MD builder
```

`gamd-openmm` is external. Install/configure it only for `--run-mode gamd` or `hmr-gamd`; `cmd` and `hmr-cmd` avoid it.

## Verify environment

```bash
gareus --help
gareus-test-run --check-deps --skip-if-missing
python -m mkdocs build --strict
```

`--check-deps` reports missing optional MD components without starting a trajectory. Run real MD only after platform and device policy are confirmed.
