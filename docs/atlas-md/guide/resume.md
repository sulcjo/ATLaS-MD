# Resume and restart

Resume same output directory after interruption:

```bash
gareus --config chignolin.yaml --resume
```

GAREUS selects furthest viable checkpoint: epoch boundary for adaptive modes or production checkpoint for plain REUS/GaMD. It writes `segments.json` and per-segment window snapshots. Preserve directory structure; do not hand-copy partial artifacts into a new output directory.

For linear torsion CV modes, resume requires exact recorded state path. Confirm `effective_config.yaml`, run manifest, and adaptive metadata before trusting resumed CV2 behavior.
