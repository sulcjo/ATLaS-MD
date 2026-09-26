# Output artifacts

| Artifact | Meaning |
| --- | --- |
| `effective_config.yaml/json` | Resolved configuration used by run |
| `run_manifest.json/yaml` | Environment, dependency, source/input/output hashes |
| `umbrella_windows.csv` | Final centers and force constants |
| `umbrella_explicit_windows.csv` | Canonical window-major table, including sparse 2D cases |
| `segments.json` | Invocation/restart chain |
| `windows/<segment>.json` | Per-segment CV/window snapshot |
| `samples/<segment>/chunk_*.parquet` | Per-step CV and energy samples |
| `exchanges/<segment>/chunk_*.parquet` | Exchange events and acceptance |
| `exchange_tuning_report.md/json` | Exchange diagnostics |
| `final_run_report.md/json` | Final validation/output summary |
| `overlap_matrix.png` (analysis) | Pairwise MBAR overlap graph over (CV1, CV2, λ); see [Overlap graph](../analysis/overlap-graph.md) |
| `overlap_density_layers.png`, `overlap_pairs_mbar.csv`, `overlap_graph_3d.html` (analysis) | Overlap density per λ layer, per-edge overlaps, interactive graph |
| `overlap_matrix_cv1_hist.png` (analysis) | CV1-marginal histogram overlap heatmap (was `overlap_matrix.png` before 2026-09-26) |

Artifacts record provenance and diagnostics. They are evidence inputs, not automatic scientific approval.
