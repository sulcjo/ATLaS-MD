# Tutorial: first CMD run

Goal: create minimal conventional umbrella/REUS run without GaMD dependency.

```bash
gareus --seq CLN025 --run-mode cmd --window-mode adaptive \
  --progress-mode jsonl --out tutorial_cmd
```

After completion:

```bash
ls tutorial_cmd/effective_config.yaml tutorial_cmd/run_manifest.json \
   tutorial_cmd/umbrella_windows.csv tutorial_cmd/final_run_report.md
```

Read artifacts, then run installed MBAR analysis help. Validate coverage/overlap/ESS before interpreting resulting PMF.
