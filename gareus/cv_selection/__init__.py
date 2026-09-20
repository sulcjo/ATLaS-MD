"""Equilibrium CV selection for ATLaS-MD (implementation plan tasks T00-T13).

This package compares candidate collective variables by the precision and
reproducibility of reweighted *equilibrium* estimates at matched cost. It never
treats physical kinetics, transition rates or relaxation times as an objective.

Import policy: the core contracts must import with a NumPy-only installation.
Modules that need OpenMM, pyarrow, duckdb or pymbar import them lazily inside
the functions that use them, so `validate`/`plan` work on an analysis host that
has no simulation stack installed.
"""
