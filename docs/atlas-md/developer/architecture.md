# Architecture

`gareus.core` dispatches `gareus.cli`. Main flow combines setup, CV force construction, windows, production, checkpoints, provenance, storage, and adaptive loops.

| Area | Modules |
| --- | --- |
| Setup | `system_setup`, `forces`, `cv` |
| Sampling | `production`, `windows`, `adaptive_feedback`, `adaptive_production` |
| State | `checkpoints`, `provenance`, `store`, `query` |
| Analysis | `mbar_analysis`, `analysis`, `diagnostics` |
| Interfaces | `cli`, `helptext`, dashboard modules |

Keep public behavior documented beside tests. Configuration resolves before execution and is written as reproducibility artifact.
