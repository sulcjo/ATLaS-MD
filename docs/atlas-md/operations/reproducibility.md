# Scratch and reproducibility

Use `--scratchdir` for fast local I/O with checkpoint mirroring to `--out`.

```bash
gareus --seq CLN025 --scratchdir "$SCRATCH/gareus_cln025" --out durable_run
```

Archive output directory with `effective_config.*`, `run_manifest.*`, `segments.json`, window tables, samples, exchanges, analysis command lines, and generated reports. Record Git revision, OpenMM platform/device, external GaMD version, and exact input files.
