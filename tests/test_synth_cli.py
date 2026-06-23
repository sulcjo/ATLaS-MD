"""CLI smoke tests for the synthetic harness."""
from __future__ import annotations

import pytest

from gareus.synth.__main__ import run_cli


def test_cli_feedback_smoke():
    pytest.importorskip("scipy")
    m = run_cli(["--landscape", "mixture-wells", "--mode", "exact",
                 "--subsystem", "feedback", "--rounds", "2",
                 "--samples-per-window", "800", "--res", "80", "--seed", "0"])
    assert "economy" in m and "final_overlap" in m
    assert m["rounds"] >= 1


def test_cli_production_smoke():
    m = run_cli(["--landscape", "mixture-wells", "--mode", "exact",
                 "--subsystem", "production", "--rounds", "2",
                 "--samples-per-window", "800", "--res", "80", "--seed", "0"])
    assert m["subsystem"] == "production"
    assert m["final_n_active"] >= 8
