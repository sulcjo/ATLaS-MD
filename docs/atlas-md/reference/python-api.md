# Python API

ATLAS-MD documents user-facing modules; internal functions can change.

```python
from gareus.query import load_samples, load_windows, reconstruct_bias_matrix

samples = load_samples("run_cln025")
windows = load_windows("run_cln025")
u_nk = reconstruct_bias_matrix(samples["cv1"], samples["cv2"], windows, beta)
```

Curated modules:

| Module | Use |
| --- | --- |
| `gareus.query` | Stored sample/window access and bias reconstruction |
| `gareus.mbar_analysis` | MBAR data, solvers, PMF/writers |
| `gareus.cv` | CV selection and mode utilities |
| `gareus.tica` | tICA state/fitting support |
| `gareus.provenance` | Run manifest/provenance helpers |

Read signatures and tests in current checkout before scripting against non-public helpers.
