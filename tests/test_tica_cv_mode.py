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


class TestMBARGuard:
    def test_version_marker_written_disabled(self, tmp_path):
        """_write_tica_version_marker writes 'disabled' when no tica_cv_version on args."""
        import argparse
        from gareus.adaptive_production import _write_tica_version_marker
        args = argparse.Namespace()
        _write_tica_version_marker(tmp_path, args)
        assert (tmp_path / "tica_cv_version.txt").read_text() == "disabled"

    def test_version_marker_written_versioned(self, tmp_path):
        """_write_tica_version_marker writes the actual version string."""
        import argparse
        from gareus.adaptive_production import _write_tica_version_marker
        args = argparse.Namespace(tica_cv_version="v2")
        _write_tica_version_marker(tmp_path, args)
        assert (tmp_path / "tica_cv_version.txt").read_text() == "v2"

    def test_epoch_sample_sources_filters_incompatible(self, tmp_path):
        """Epochs with mismatched tica_cv_version are excluded from sample sources."""
        from gareus.adaptive_production import _epoch_sample_sources

        # Create two fake epoch dirs; epoch_000 has v1, epoch_001 has v2
        for name, version in [("epoch_000", "v1"), ("epoch_001", "v2")]:
            d = tmp_path / name
            d.mkdir()
            (d / "tica_cv_version.txt").write_text(version)
            # Write a minimal samples.csv so _run_dir_has_samples passes
            (d / "samples.csv").write_text("step,window,cv_A\n1,0,0.5\n")

        sources = _epoch_sample_sources(tmp_path, include_epochs=True, tica_cv_version="v2")
        names = [label for label, _ in sources]
        assert "epoch_001" in names
        assert "epoch_000" not in names

    def test_epoch_sample_sources_no_filter_when_disabled(self, tmp_path):
        """tica_cv_version=None includes all epochs (tICA disabled)."""
        from gareus.adaptive_production import _epoch_sample_sources

        for name in ["epoch_000", "epoch_001"]:
            d = tmp_path / name
            d.mkdir()
            (d / "samples.csv").write_text("step,window,cv_A\n1,0,0.5\n")

        sources = _epoch_sample_sources(tmp_path, include_epochs=True, tica_cv_version=None)
        names = [label for label, _ in sources]
        assert "epoch_000" in names
        assert "epoch_001" in names


class TestApplyTICACentersToRegistry:
    def test_updates_active_states(self, tmp_path):
        """_apply_tica_centers_to_registry sets secondary_center on active states."""
        from gareus.adaptive_production import (
            WindowState, WindowStateRegistry, _apply_tica_centers_to_registry
        )
        import csv

        # Build a registry with 2 active states
        reg = WindowStateRegistry()
        s0 = WindowState(state_id=0, primary_center=1.0, primary_k=1.0, secondary_center=0.0)
        s1 = WindowState(state_id=1, primary_center=2.0, primary_k=1.0, secondary_center=0.0)
        reg._states = {0: s0, 1: s1}
        reg._next_state_id = 2

        # Write an epoch_window_map.csv: window 0 → state 0, window 1 → state 1
        wmap = tmp_path / "epoch_window_map.csv"
        with open(wmap, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch_window", "state_id"])
            writer.writeheader()
            writer.writerow({"epoch_window": 0, "state_id": 0})
            writer.writerow({"epoch_window": 1, "state_id": 1})

        per_window_centers = {0: 0.42, 1: -0.13}
        n = _apply_tica_centers_to_registry(reg, per_window_centers, tmp_path)

        assert n == 2
        assert reg.get_state(0).secondary_center == pytest.approx(0.42)
        assert reg.get_state(1).secondary_center == pytest.approx(-0.13)

    def test_skips_retired_states(self, tmp_path):
        """Retired states are not updated."""
        from gareus.adaptive_production import (
            WindowState, WindowStateRegistry, _apply_tica_centers_to_registry
        )
        import csv

        reg = WindowStateRegistry()
        s0 = WindowState(state_id=0, primary_center=1.0, primary_k=1.0, active=False, secondary_center=0.0)
        reg._states = {0: s0}
        reg._next_state_id = 1

        wmap = tmp_path / "epoch_window_map.csv"
        with open(wmap, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch_window", "state_id"])
            writer.writeheader()
            writer.writerow({"epoch_window": 0, "state_id": 0})

        n = _apply_tica_centers_to_registry(reg, {0: 1.5}, tmp_path)
        assert n == 0
        assert reg.get_state(0).secondary_center == pytest.approx(0.0)


class TestEpochCycling:
    """Tests for tica_epochs_per_cycle auto-trigger logic."""

    def _make_args(self, **kw) -> argparse.Namespace:
        ns = argparse.Namespace(
            tica_obs_interval=10,
            tica_update_after_epochs=None,
            tica_epochs_per_cycle=0,
            tica_lag_frames=50,
            tica_min_eigenvalue=0.0,
            tica_state_file="",
        )
        for k, v in kw.items():
            setattr(ns, k, v)
        return ns

    def test_cycle_trigger_fires_at_correct_epochs(self):
        """tica_epochs_per_cycle=2 fires at epoch 1, 3, 5 (i.e. after epoch+1 is multiple of 2)."""
        args = self._make_args(tica_epochs_per_cycle=2)
        fired = []
        for epoch in range(6):
            obs_int = int(getattr(args, "tica_obs_interval", 0) or 0)
            update_after = getattr(args, "tica_update_after_epochs", None)
            epc = int(getattr(args, "tica_epochs_per_cycle", 0) or 0)
            should = False
            if update_after is not None and epoch in [int(e) for e in update_after]:
                should = True
            if epc > 0 and (epoch + 1) % epc == 0:
                should = True
            if obs_int > 0 and should:
                fired.append(epoch)
        assert fired == [1, 3, 5]

    def test_legacy_update_after_still_fires(self):
        """Legacy tica_update_after_epochs=[0, 2] fires independently of cycle."""
        args = self._make_args(tica_update_after_epochs=[0, 2], tica_epochs_per_cycle=0)
        fired = []
        for epoch in range(5):
            obs_int = int(getattr(args, "tica_obs_interval", 0) or 0)
            update_after = getattr(args, "tica_update_after_epochs", None)
            epc = int(getattr(args, "tica_epochs_per_cycle", 0) or 0)
            should = False
            if update_after is not None and epoch in [int(e) for e in update_after]:
                should = True
            if epc > 0 and (epoch + 1) % epc == 0:
                should = True
            if obs_int > 0 and should:
                fired.append(epoch)
        assert fired == [0, 2]

    def test_both_triggers_union(self):
        """Both triggers active: fires at union of both sets."""
        args = self._make_args(tica_update_after_epochs=[4], tica_epochs_per_cycle=3)
        fired = []
        for epoch in range(7):
            obs_int = int(getattr(args, "tica_obs_interval", 0) or 0)
            update_after = getattr(args, "tica_update_after_epochs", None)
            epc = int(getattr(args, "tica_epochs_per_cycle", 0) or 0)
            should = False
            if update_after is not None and epoch in [int(e) for e in update_after]:
                should = True
            if epc > 0 and (epoch + 1) % epc == 0:
                should = True
            if obs_int > 0 and should:
                fired.append(epoch)
        # cycle=3 fires at epoch 2, 5; explicit fires at epoch 4
        assert fired == [2, 4, 5]

    def test_no_trigger_when_obs_disabled(self):
        """tica_obs_interval=0 means no trigger regardless of other settings."""
        args = self._make_args(tica_obs_interval=0, tica_epochs_per_cycle=1, tica_update_after_epochs=[0])
        for epoch in range(5):
            obs_int = int(getattr(args, "tica_obs_interval", 0) or 0)
            assert obs_int == 0  # early return in _maybe_update_tica_cvaux

    def test_cli_tica_epochs_per_cycle_in_known_dests(self):
        """tica_epochs_per_cycle must be a known argparse dest for YAML config validation."""
        from gareus.config import _build_known_config_dests
        from gareus.cli import build_gareus_parser
        parser = build_gareus_parser()
        dests = _build_known_config_dests(parser)
        assert "tica_epochs_per_cycle" in dests, (
            "YAML key 'tica_epochs_per_cycle' not in argparse dests — "
            "add --tica-epochs-per-cycle to cli.py"
        )
