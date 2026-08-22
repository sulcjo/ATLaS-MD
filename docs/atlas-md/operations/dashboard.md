# Progress and dashboard

Use `--progress-mode console`, `jsonl`, or `both`; use `--tui-mode dashboard`, `interactive`, `line`, or `none`.

```bash
gareus --seq CLN025 --progress-mode both --tui-mode dashboard --out observed_run
```

Treat dashboard values as operational telemetry. For adaptive budgets, distinguish committed work from live uncommitted progress and confirm final runtime from persisted run artifacts.
