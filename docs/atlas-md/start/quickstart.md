# Quickstart

Start with conventional umbrella/REUS MD. It minimizes external GaMD requirements while exercising same system, CV, windows, exchanges, artifacts, and analysis paths.

```bash
gareus --seq CLN025 --run-mode cmd --window-mode adaptive --out run_cln025_cmd
```

Inspect `run_cln025_cmd/effective_config.yaml`, `run_manifest.json`, `umbrella_windows.csv`, and `final_run_report.md`. Then use [output artifacts](../reference/output-artifacts.md) and [analysis workflow](../analysis/overview.md).

## Config-first run

```bash
gareus --write-config-template chignolin.yaml
gareus --config chignolin.yaml --seq CLN025 --out run_cln025
```

CLI values override config values. Retain resolved config and manifest with results.

!!! warning "Do not read a PMF yet"
    A finished run or written PMF only proves workflow completion. Apply [PMF validity gates](../analysis/pmf-validity.md).
