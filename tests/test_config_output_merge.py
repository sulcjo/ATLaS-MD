"""Test that the merged output dict in _basic_chignolin_config is correct."""

from __future__ import annotations


def test_basic_chignolin_config_has_single_output_key() -> None:
    """Verify that _basic_chignolin_config returns a dict with exactly one 'output' key."""
    from gareus.config import _basic_chignolin_config

    result = _basic_chignolin_config()

    # Count occurrences of "output" key
    output_keys = [k for k in result.keys() if k == "output"]
    assert len(output_keys) == 1, f"Expected exactly one 'output' key, found {len(output_keys)}"


def test_basic_chignolin_config_output_has_all_fields() -> None:
    """Verify that the merged output dict contains all expected fields."""
    from gareus.config import _basic_chignolin_config

    result = _basic_chignolin_config()
    output = result["output"]

    # Check for fields from the first output block (out, resume)
    assert output["out"] == "chignolin/", "Missing or incorrect 'out' field"
    assert output["resume"] is True, "Missing or incorrect 'resume' field"

    # Check for fields from the second output block (trajectory/report settings)
    assert output["traj_interval"] == 50, "Missing or incorrect 'traj_interval' field"
    assert output["traj_format"] == "xtc", "Missing or incorrect 'traj_format' field"
    assert output["adaptive_pilot_trajectories"] is False, "Missing or incorrect 'adaptive_pilot_trajectories' field"
    assert output["adaptive_production_trajectories"] is True, "Missing or incorrect 'adaptive_production_trajectories' field"
    assert output["sample_potential_energy"] is True, "Missing or incorrect 'sample_potential_energy' field"
    assert output["report_interval"] == 50, "Missing or incorrect 'report_interval' field"
    assert output["distance_output_interval"] == 50, "Missing or incorrect 'distance_output_interval' field"
    assert output["checkpoint_interval"] == 10, "Missing or incorrect 'checkpoint_interval' field"


def test_basic_chignolin_config_output_key_order() -> None:
    """Verify that 'out' and 'resume' appear first in the merged output dict."""
    from gareus.config import _basic_chignolin_config

    result = _basic_chignolin_config()
    output = result["output"]
    keys = list(output.keys())

    # Verify order: "out" and "resume" should be first
    assert keys[0] == "out", f"Expected first key to be 'out', got '{keys[0]}'"
    assert keys[1] == "resume", f"Expected second key to be 'resume', got '{keys[1]}'"
