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

Artifacts record provenance and diagnostics. They are evidence inputs, not automatic scientific approval.
