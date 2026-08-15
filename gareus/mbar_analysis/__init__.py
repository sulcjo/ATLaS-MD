"""MBAR/PMF post-hoc analysis subpackage.

Progressively absorbing `analyze_gareus_mbar.py`'s logic via a strangler-fig
migration (see docs/superpowers/specs/2026-08-13-mbar-analysis-modularization-a1-design.md).
The data-loading domain (Data dataclass, all NPZ/CSV/Parquet/adaptive-production
loaders) now lives here (`data.py`, `loaders_adaptive.py`,
`loaders_union_parquet.py`, `loaders.py`) as of Plan A2. `cli.py` still
delegates to the top-level script for everything not yet relocated.
"""
