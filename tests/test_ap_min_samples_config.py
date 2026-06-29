"""Tests for ap_min_samples_per_window and ap_final_min_samples_per_window config keys.

Verifies:
- Both new argparse args parse to int and have the correct defaults.
- ap_min_samples_per_window properly sets adaptive_production_retire_min_samples
  (i.e. the new key is NOT a no-op — the new preferred key works on its own).
- ap_retire_min_samples (legacy alias) overrides when set explicitly to a non-default value.
- ap_final_min_samples_per_window sets adaptive_production_final_min_samples_per_state.
- Both keys are accepted from YAML without unknown-key errors.
- policy_from_args picks up the values from both paths.

No OpenMM, PeptideBuilder, or gamd-openmm imports.
"""
from __future__ import annotations

import types

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_args(**kwargs) -> types.SimpleNamespace:
    """Build a minimal namespace that parse_args would produce after shims,
    then override with caller-supplied values.  Only the keys relevant to
    the min-samples-per-window feature are set here.
    """
    ns = types.SimpleNamespace(**kwargs)
    return ns


def _parse(argv: list[str]):
    """Run the real parse_args path (no OpenMM required)."""
    from gareus.cli import parse_args
    # --seq is required; use a short sequence that doesn't need OpenMM.
    base = ["--seq", "AA", "--run-mode", "cmd"]
    return parse_args(base + argv)


# ---------------------------------------------------------------------------
# Argparse defaults
# ---------------------------------------------------------------------------

class TestArgparseDefaults:
    def test_ap_min_samples_per_window_default_is_200(self):
        args = _parse([])
        assert args.ap_min_samples_per_window == 200

    def test_ap_final_min_samples_per_window_default_is_100(self):
        args = _parse([])
        assert args.ap_final_min_samples_per_window == 100

    def test_ap_min_samples_per_window_is_int(self):
        args = _parse(["--ap-min-samples-per-window", "300"])
        assert isinstance(args.ap_min_samples_per_window, int)
        assert args.ap_min_samples_per_window == 300

    def test_ap_final_min_samples_per_window_is_int(self):
        args = _parse(["--ap-final-min-samples-per-window", "500"])
        assert isinstance(args.ap_final_min_samples_per_window, int)
        assert args.ap_final_min_samples_per_window == 500


# ---------------------------------------------------------------------------
# Shim wiring: new key → internal dest
# ---------------------------------------------------------------------------

class TestShimWiring:
    def test_new_key_sets_retire_min_samples(self):
        """ap_min_samples_per_window ALONE must propagate to the internal dest."""
        args = _parse(["--ap-min-samples-per-window", "350"])
        assert args.adaptive_production_retire_min_samples == 350

    def test_final_new_key_sets_final_min_samples_per_state(self):
        """ap_final_min_samples_per_window must propagate to the internal dest."""
        args = _parse(["--ap-final-min-samples-per-window", "600"])
        assert args.adaptive_production_final_min_samples_per_state == 600

    def test_default_retire_wiring_is_200(self):
        """Without any override, retire_min_samples defaults to 200."""
        args = _parse([])
        assert args.adaptive_production_retire_min_samples == 200

    def test_default_final_wiring_is_100(self):
        """Without any override, final_min_samples_per_state defaults to 100."""
        args = _parse([])
        assert args.adaptive_production_final_min_samples_per_state == 100


# ---------------------------------------------------------------------------
# Legacy alias precedence
# ---------------------------------------------------------------------------

class TestLegacyAliasPrecedence:
    def test_legacy_alias_overrides_new_key_when_set_explicitly(self):
        """ap_retire_min_samples (legacy) should win when set to a non-default value."""
        args = _parse([
            "--ap-min-samples-per-window", "350",
            "--ap-retire-min-samples", "400",
        ])
        assert args.adaptive_production_retire_min_samples == 400

    def test_legacy_alias_default_does_not_clobber_new_key(self):
        """When only ap_min_samples_per_window is set, its value must survive
        (i.e. the legacy alias default of 200 must NOT override 350)."""
        args = _parse(["--ap-min-samples-per-window", "350"])
        assert args.adaptive_production_retire_min_samples == 350

    def test_both_at_default_gives_200(self):
        """When neither is set explicitly, result is the shared default of 200."""
        args = _parse([])
        assert args.adaptive_production_retire_min_samples == 200


# ---------------------------------------------------------------------------
# YAML acceptance (no unknown-key errors)
# ---------------------------------------------------------------------------

class TestYamlAcceptance:
    def test_ap_min_samples_per_window_accepted_from_yaml(self, tmp_path):
        """ap_min_samples_per_window must be accepted by parse_args from YAML."""
        cfg = tmp_path / "test.yaml"
        cfg.write_text(
            "sequence:\n  seq: AA\n"
            "adaptive_production:\n"
            "  ap_min_samples_per_window: 250\n"
        )
        from gareus.cli import parse_args
        args = parse_args(["--config", str(cfg), "--run-mode", "cmd"])
        assert args.ap_min_samples_per_window == 250
        assert args.adaptive_production_retire_min_samples == 250

    def test_ap_final_min_samples_per_window_accepted_from_yaml(self, tmp_path):
        """ap_final_min_samples_per_window must be accepted by parse_args from YAML."""
        cfg = tmp_path / "test.yaml"
        cfg.write_text(
            "sequence:\n  seq: AA\n"
            "adaptive_production:\n"
            "  ap_final_min_samples_per_window: 700\n"
        )
        from gareus.cli import parse_args
        args = parse_args(["--config", str(cfg), "--run-mode", "cmd"])
        assert args.ap_final_min_samples_per_window == 700
        assert args.adaptive_production_final_min_samples_per_state == 700

    def test_both_yaml_keys_together_no_error(self, tmp_path):
        """Both keys together in YAML must not raise unknown-key errors."""
        cfg = tmp_path / "test.yaml"
        cfg.write_text(
            "sequence:\n  seq: AA\n"
            "adaptive_production:\n"
            "  ap_min_samples_per_window: 300\n"
            "  ap_final_min_samples_per_window: 500\n"
        )
        from gareus.cli import parse_args
        # Should not raise
        args = parse_args(["--config", str(cfg), "--run-mode", "cmd"])
        assert args.ap_min_samples_per_window == 300
        assert args.ap_final_min_samples_per_window == 500


# ---------------------------------------------------------------------------
# policy_from_args integration
# ---------------------------------------------------------------------------

class TestPolicyFromArgs:
    def test_policy_reads_retire_min_samples(self):
        """policy_from_args picks up adaptive_production_retire_min_samples."""
        from gareus.adaptive_production import policy_from_args
        ns = types.SimpleNamespace(adaptive_production_retire_min_samples=350)
        p = policy_from_args(ns)
        assert p.min_samples_for_retire == 350

    def test_policy_reads_final_min_samples_per_state(self):
        """policy_from_args picks up adaptive_production_final_min_samples_per_state."""
        from gareus.adaptive_production import policy_from_args
        ns = types.SimpleNamespace(adaptive_production_final_min_samples_per_state=600)
        p = policy_from_args(ns)
        assert p.final_min_samples_per_state == 600

    def test_end_to_end_new_key_reaches_policy(self):
        """Full parse_args → policy_from_args path: ap_min_samples_per_window reaches policy."""
        from gareus.adaptive_production import policy_from_args
        from gareus.cli import parse_args
        args = parse_args(["--seq", "AA", "--run-mode", "cmd",
                           "--ap-min-samples-per-window", "420"])
        p = policy_from_args(args)
        assert p.min_samples_for_retire == 420

    def test_end_to_end_final_key_reaches_policy(self):
        """Full parse_args → policy_from_args path: ap_final_min_samples_per_window reaches policy."""
        from gareus.adaptive_production import policy_from_args
        from gareus.cli import parse_args
        args = parse_args(["--seq", "AA", "--run-mode", "cmd",
                           "--ap-final-min-samples-per-window", "800"])
        p = policy_from_args(args)
        assert p.final_min_samples_per_state == 800
