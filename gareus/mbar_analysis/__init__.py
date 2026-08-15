"""MBAR/PMF post-hoc analysis subpackage.

Currently a strangler-fig shell: `cli.py` delegates to the top-level
`analyze_gareus_mbar.py` script, which still holds all real analysis logic.
Later plans in the same modularization sequence (see
docs/superpowers/specs/2026-08-13-mbar-analysis-modularization-a1-design.md)
progressively move that logic into this subpackage.
"""
