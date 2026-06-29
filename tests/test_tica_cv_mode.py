"""Tests for tica-linear integration into gareus.cv — no OpenMM required."""
import pytest
from unittest.mock import MagicMock
import argparse

from gareus.cv import (
    secondary_cv_mode,
    secondary_cv_is_transition,
    secondary_cv_range,
    secondary_cv_enabled,
)


def _args(**kw) -> argparse.Namespace:
    ns = argparse.Namespace()
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


class TestTICALinearMode:
    def test_tica_linear_canonical_from_alias(self):
        assert secondary_cv_mode("tica") == "tica-linear"
        assert secondary_cv_mode("tica-linear") == "tica-linear"
        assert secondary_cv_mode("tica_linear") == "tica-linear"

    def test_tica_linear_canonical_from_args(self):
        args = _args(secondary_cv="tica")
        assert secondary_cv_mode(args) == "tica-linear"

    def test_tica_linear_is_transition(self):
        assert secondary_cv_is_transition("tica-linear") is True
        assert secondary_cv_is_transition("tica") is True

    def test_tica_linear_range(self):
        lo, hi = secondary_cv_range("tica-linear")
        assert lo == pytest.approx(-6.0)
        assert hi == pytest.approx(6.0)

    def test_tica_linear_enabled(self):
        args = _args(secondary_cv="tica-linear")
        assert secondary_cv_enabled(args) is True

    def test_none_still_disabled(self):
        assert secondary_cv_enabled(_args(secondary_cv="none")) is False

    def test_alpha_coil_beta_range_unchanged(self):
        lo, hi = secondary_cv_range("alpha-coil-beta")
        assert lo == pytest.approx(-1.0)
        assert hi == pytest.approx(1.0)

    def test_tica_linear_metadata_schema(self):
        """build_tica_linear_metadata returns required schema keys when mode is tica-linear."""
        from gareus.cv import build_tica_linear_metadata
        meta = build_tica_linear_metadata(enabled=True, n_phi=5, n_psi=5, tica_state_path="tica.json")
        assert meta["enabled"] is True
        assert meta["mode"] == "tica-linear"
        assert "range_min" in meta
        assert "range_max" in meta
        assert meta["range_min"] == pytest.approx(-6.0)
        assert meta["range_max"] == pytest.approx(6.0)


class TestConfigKeyRejection:
    def test_tica_keys_not_rejected_by_config(self):
        """tica YAML keys must be argparse dests so config validation accepts them."""
        import sys
        from gareus.config import _build_known_config_dests
        from gareus.cli import build_gareus_parser

        parser = build_gareus_parser()
        dests = _build_known_config_dests(parser)
        for key in ("tica_obs_interval", "tica_update_after_epochs", "tica_lag_frames"):
            assert key in dests, f"YAML key '{key}' not in argparse dests — add --{key.replace('_','-')} to cli.py"
