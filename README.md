# ATLaS-MD

**Adaptive Thermodynamic Landscape Sampling for molecular dynamics**

ATLaS-MD is an OpenMM-based peptide sampling framework for explicit-solvent umbrella sampling, replica exchange, GaMD/Pep-GaMD acceleration, adaptive state placement, and MBAR-ready analysis.

It is designed for workflows where the sampling protocol itself is part of the scientific method: state definitions are explicit, exchange kernels are tested against their target distribution, adaptive phases are tracked separately, and production data are written with enough provenance to reconstruct the sampled Hamiltonians later.

> **Current release:** `v0.7`  
> **Python:** `>=3.10`  
> **OpenMM:** `>=8`

## Why ATLaS-MD

ATLaS-MD combines several enhanced-sampling layers in one reproducible workflow:

| Capability | What it provides |
|---|---|
| **Umbrella sampling / REUS** | 1D and 2D biased state ensembles with explicit cross-state energy evaluation |
| **GaMD / Pep-GaMD** | Optional accelerated dynamics, including state-dependent lambda ladders |
| **Adaptive windows** | Pilot-driven refinement, sparse 2D layouts, Delaunay feedback, and epoch-based production |
| **NVT / NPT production** | Conventional OpenMM dynamics plus corrected application-controlled NPT moves for supported boosted Hamiltonians |
| **GENPEPT seeding** | Conformer generation and CV-aware starting-state selection |
| **MBAR-ready storage** | Per-sample CVs, energies, state mappings, exchange records, and phase-local Hamiltonian metadata |
| **Reproducibility** | Checkpoints, manifests, source/input hashes, segment tracking, and restart-safe output layout |

## Scientific contract

ATLaS-MD treats thermodynamic correctness as an explicit part of the implementation.

The intended state Hamiltonian is

\[
U_k^*(x,V)
=
U_{\mathrm{phys}}(x,V)
+
W_k(x,V)
+
\Delta V_k(x,V),
\]

with replica exchange evaluated by cross-pricing the state-dependent terms under the candidate thermodynamic states.

Elementary exchange moves are constructed to satisfy detailed balance with respect to the intended extended ensemble. Sequential exchange sweeps preserve the same stationary distribution, although a complete sweep need not itself be reversible.

For boosted NPT, ATLaS-MD does not rely on OpenMM's native barostat acceptance energy when that energy would omit the integrator-applied GaMD boost or include auxiliary bookkeeping forces. Supported boosted modes instead use an application-controlled Metropolis volume move against the effective potential actually being propagated.

See **[Thermodynamic target and detailed-balance contract](docs/atlas-md/guide/thermodynamic-validity.md)** for the derivation, assumptions, validation scope, and remaining limitations.

## Quick start

Install the full local stack:

```bash
pip install -e ".[all]"
```

`gamd-openmm` is an external dependency and is required only for `gamd` / `hmr-gamd` runs.

Run a minimal peptide workflow:

```bash
gareus --seq CLN025 --out run_cln025
```

Write an editable configuration first:

```bash
gareus --write-config-template chignolin.yaml
gareus --config chignolin.yaml --seq CLN025 --out run_cln025
```

Conventional REUS without GaMD:

```bash
gareus \
  --seq CLN025 \
  --run-mode cmd \
  --window-mode adaptive \
  --out run_cmd
```

HMR + GaMD with adaptive production:

```bash
gareus \
  --seq CLN025 \
  --run-mode hmr-gamd \
  --cv1 contacts \
  --cv2 rama-map \
  --window-mode adaptive-production \
  --md-budget-ns 840 \
  --out run_adaptive
```

## Workflow

```text
sequence / seeds
      |
      v
system construction + equilibration
      |
      v
CV definition + state construction
      |
      v
umbrella / GaMD / lambda state ensemble
      |
      v
REUS / Gibbs-walk exchange
      |
      +--> adaptive feedback / new epoch
      |
      v
frozen production
      |
      v
Parquet samples + exchange ledger
      |
      v
MBAR / PMF / diagnostics
```

Adaptive pilot and production epochs are intentionally tracked as distinct thermodynamic phases rather than silently pooled.

## Main run modes

### Conventional MD

```bash
gareus --seq CLN025 --run-mode cmd --out run_cmd
```

Use `hmr-cmd` for the HMR convenience path.

### GaMD

```bash
gareus --seq CLN025 --run-mode gamd --out run_gamd
```

Use `hmr-gamd` for the HMR convenience path. GaMD setup/calibration is shared across replicas where the selected workflow supports it.

### 2D sampling

```bash
gareus \
  --seq CLN025 \
  --cv1 contacts \
  --cv2 rama-map \
  --window-mode adaptive-feedback \
  --out run_2d
```

### Double-adaptive sampling

```bash
gareus \
  --seq CLN025 \
  --run-mode hmr-gamd \
  --cv1 contacts \
  --cv2 rama-map \
  --window-mode double-adaptive \
  --adaptive-rounds 3 \
  --ap-epochs 10 \
  --out run_double_adaptive
```

### GENPEPT-seeded workflow

```bash
python GENPEPT.py --config chignolin.yaml

gareus \
  --config chignolin.yaml \
  --seed-conformers-dir chignolin_genpept_seeds
```

GENPEPT survivors are scored in the active CV space and audited under `us_starting_structures/`.

## Exchange modes

ATLaS-MD currently supports:

- `neighbor` — nearest-neighbor REUS/HREX exchange;
- `random-pair` — randomly scheduled symmetric pair exchanges;
- `all-pair-sweep` — shuffled all-pair Metropolis sweeps;
- `gibbs-walk` — long-range heat-bath-like proposals with an explicit Metropolis-Hastings correction.

The exact finite-state validation suite checks detailed balance for elementary exchange kernels and stationary invariance for their sequential compositions.

## Output model

The production output is designed to remain analyzable after adaptive changes, restart events, and long runs.

```text
run/
├── effective_config.yaml
├── run_manifest.json
├── umbrella_windows.csv
├── umbrella_explicit_windows.csv
├── segments.json
├── windows/
│   └── <segment>.json
├── samples/
│   └── <segment>/chunk_*.parquet
├── exchanges/
│   └── <segment>/chunk_*.parquet
├── exchange_tuning_report.md
└── final_run_report.md
```

Important design rule: **state identity is explicit and phase-local**. Analysis should not infer thermodynamic state solely from a replica index or raw local window number.

## Analysis

CLI entry point:

```bash
gareus-analyze --help
```

Python query layer:

```python
from gareus.query import load_samples, load_windows, reconstruct_bias_matrix

samples = load_samples("run_cln025")
windows = load_windows("run_cln025")

beta = 1 / (8.314462618e-3 * 300.0)
u_nk = reconstruct_bias_matrix(
    samples["cv1"],
    samples["cv2"],
    windows,
    beta,
)
```

Additional tools:

```bash
gareus-suggest-cvs --run-dir run_cln025 --print
gareus-energy-decompose --run-dir run_cln025 --out energy_decomposition.csv
```

## Resume and long runs

ATLaS-MD supports production and epoch-level restart:

```bash
gareus --config chignolin.yaml --resume
```

Resume state includes exchange assignments, RNG state, production counters, segment provenance, and supported NPT-controller state. Interrupted output segments are sealed so rows beyond the last valid checkpoint are not silently reused as equilibrium data.

## Validation and testing

Fast test suite:

```bash
pytest
```

Basic package checks:

```bash
python -m py_compile gareus/*.py gareus_peptide.py GENPEPT.py
python -m gareus --help
```

Real workflow smoke test:

```bash
gareus-test-run --check-deps --skip-if-missing
gareus-test-run --out tiny_real_test --force
```

The validation suite includes checks for:

- exact exchange-kernel detailed balance on finite state spaces;
- Gibbs proposal/reverse-proposal correctness;
- lambda-ladder boost cross-evaluation;
- NPT acceptance algebra and molecular Jacobians;
- GaMD/Pep-GaMD force-group wiring;
- checkpoint and resume consistency;
- state/window-map provenance.

Finite-timestep propagation is not claimed to be mathematically exact. See the thermodynamic contract for the distinction between an exact transition-kernel statement and numerical sampling accuracy.

## Documentation

The full manual covers installation, tutorials, workflow design, scientific assumptions, analysis, operations, and developer internals:

**https://sulcjo.github.io/2026_peptide_sampler/**

Build it locally with:

```bash
pip install -e ".[docs]"
python -m mkdocs serve
```

Strict static build:

```bash
python -m mkdocs build --strict
```

Useful entry points:

- [Thermodynamic validity](docs/atlas-md/guide/thermodynamic-validity.md)
- [Umbrella windows and REUS](docs/atlas-md/guide/windows-and-exchange.md)
- [GaMD](docs/atlas-md/guide/gamd.md)
- [Adaptive workflows](docs/atlas-md/guide/adaptive-workflows.md)
- [PMF validity](docs/atlas-md/analysis/pmf-validity.md)
- [Architecture](docs/atlas-md/developer/architecture.md)

## Package architecture

| Module | Responsibility |
|---|---|
| `gareus/production.py` | Production stepping, exchange, sampling, checkpoints |
| `gareus/npt.py` | Application-controlled NPT volume moves |
| `gareus/pep_gamd.py` | Pep-GaMD partitioning, lambda ladder, effective-potential adapters |
| `gareus/windows.py` | State/window construction and exchange topology |
| `gareus/adaptive_feedback.py` | Pilot-driven state refinement |
| `gareus/adaptive_production.py` | Epoch-based adaptive production and runtime budgeting |
| `gareus/cv.py` / `gareus/forces.py` | Collective variables and OpenMM forces |
| `gareus/seeding.py` | CV-aware starting-state selection |
| `gareus/store.py` / `gareus/query.py` | Parquet storage and query layer |
| `gareus/mbar_analysis/` | MBAR/PMF analysis |
| `gareus/provenance.py` | Run manifests and reproducibility metadata |
| `GENPEPT.py` | Conformer generation and seed-library construction |

## Command reference

Use the CLI as the source of truth for available options:

```bash
gareus -h      # operational help
gareus -hh     # extended method / option reference
```

For reproducible production work, prefer version-controlled YAML configuration over long shell commands.
