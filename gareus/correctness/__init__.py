"""Auditable correctness primitives for ATLaS-MD (findings 2-7; never #1).

This package does not import OpenMM at module import time. Its numerical and
filesystem contracts can be tested independently of a GPU installation.
"""
