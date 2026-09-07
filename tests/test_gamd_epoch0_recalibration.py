"""Tests for epoch-0 GaMD envelope recalibration and epoch-0 step scaling.

No OpenMM required: gareus.adaptive_production imports OpenMM lazily (only
inside functions that receive it as an argument), and
_maybe_recalibrate_gamd_boost / _epoch0_scaled_steps are pure Python + JSON I/O.
"""
import argparse
import json

import pytest

from gareus.adaptive_production import _epoch0_scaled_steps, _maybe_recalibrate_gamd_boost
from gareus.gamd_calibration import WindowEnergyStats, pool_window_stats, compute_group_calibration


def _args(**kw) -> argparse.Namespace:
    ns = argparse.Namespace(run_mode="gamd", gamd_boost_type="lower-total")
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


def _write_stats_file(path, group_window_stats: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "schema_version": "gamd_production_envelope_stats_v1",
        "gamd_boost_type": "lower-total",
        "epoch_index": 0,
        "per_group_window_stats": group_window_stats,
    }))


def _write_shared_globals(path, all_globals: dict, **extra) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"mode": "joint_envelope_gamd_calibration", "all_globals": all_globals, "interesting_globals": {}}
    payload.update(extra)
    path.write_text(json.dumps(payload))


class TestEpoch0ScaledSteps:
    def test_epoch_zero_is_halved_by_default_fraction(self):
        assert _epoch0_scaled_steps(10000, epoch=0, fraction=0.5) == 5000

    def test_later_epochs_unaffected(self):
        assert _epoch0_scaled_steps(10000, epoch=1, fraction=0.5) == 10000
        assert _epoch0_scaled_steps(10000, epoch=7, fraction=0.5) == 10000

    def test_fraction_one_is_passthrough(self):
        assert _epoch0_scaled_steps(12345, epoch=0, fraction=1.0) == 12345

    def test_never_scales_to_zero_steps(self):
        assert _epoch0_scaled_steps(1, epoch=0, fraction=0.01) == 1

    def test_zero_or_negative_input_steps_passthrough(self):
        assert _epoch0_scaled_steps(0, epoch=0, fraction=0.5) == 0

    def test_fraction_clamped_to_unit_interval(self):
        assert _epoch0_scaled_steps(10000, epoch=0, fraction=5.0) == 10000
        assert _epoch0_scaled_steps(10000, epoch=0, fraction=-1.0) == 1


class TestMaybeRecalibrateGamdBoostGating:
    def test_noop_for_epoch_not_zero(self, tmp_path):
        assert _maybe_recalibrate_gamd_boost(1, tmp_path, _args(), tmp_path) == {}

    def test_noop_when_shared_dir_is_none(self, tmp_path):
        assert _maybe_recalibrate_gamd_boost(0, tmp_path, _args(), None) == {}

    def test_noop_when_flag_disabled(self, tmp_path):
        args = _args(adaptive_production_gamd_recalibrate_after_epoch0=False)
        assert _maybe_recalibrate_gamd_boost(0, tmp_path, args, tmp_path) == {}

    def test_noop_when_run_mode_is_not_gamd(self, tmp_path):
        args = _args(run_mode="cmd")
        assert _maybe_recalibrate_gamd_boost(0, tmp_path, args, tmp_path) == {}

    def test_skips_when_no_stats_files_found(self, tmp_path):
        epoch_dir = tmp_path / "epoch_000"
        epoch_dir.mkdir()
        report = _maybe_recalibrate_gamd_boost(0, epoch_dir, _args(), tmp_path / "shared")
        assert report["status"] == "skipped_no_stats"

    def test_skips_when_shared_envelope_globals_missing(self, tmp_path):
        epoch_dir = tmp_path / "epoch_000"
        _write_stats_file(epoch_dir / "gamd_production_envelope_stats.json", {
            "Total": [{"group": "Total", "window": 0, "vmax": 100.0, "vmin": -200.0, "mean": -50.0, "var": 1600.0, "n": 500}],
        })
        shared_dir = tmp_path / "shared_gamd_empty"
        shared_dir.mkdir()
        report = _maybe_recalibrate_gamd_boost(0, epoch_dir, _args(), shared_dir)
        assert report["status"] == "skipped_no_shared_envelope"

    def test_skips_when_no_group_has_a_matching_sigma0(self, tmp_path):
        epoch_dir = tmp_path / "epoch_000"
        _write_stats_file(epoch_dir / "gamd_production_envelope_stats.json", {
            "Total": [{"group": "Total", "window": 0, "vmax": 100.0, "vmin": -200.0, "mean": -50.0, "var": 1600.0, "n": 500}],
        })
        shared_dir = tmp_path / "shared_gamd"
        _write_shared_globals(shared_dir / "shared_gamd_setup_globals.json", {"sigma0_Dihedral": 6.0})
        report = _maybe_recalibrate_gamd_boost(0, epoch_dir, _args(), shared_dir)
        assert report["status"] == "skipped_no_matching_groups"


class TestMaybeRecalibrateGamdBoostLadderGuard:
    """Final-review fix wave, C4. Recalibrating the campaign-shared envelope
    after epoch 0 overwrites Vmax/Vmin/k0/threshold for every later epoch.
    Under a λ-ladder that is not merely an efficiency choice: MBAR applies ONE
    envelope (load_pep_gamd_envelope) to every sample, so epoch-0 samples get
    reweighted under epoch-1's envelope, and because k0max changes, λ stops
    denoting the same thermodynamic state across epochs. Must no-op."""

    def _ready_epoch(self, tmp_path):
        """An epoch dir + shared envelope that WOULD recalibrate successfully,
        so a 'skipped_lambda_ladder' can only come from the ladder guard."""
        epoch_dir = tmp_path / "epoch_000"
        _write_stats_file(epoch_dir / "gamd_production_envelope_stats.json", {
            "Total": [{"group": "Total", "window": 0, "vmax": 100.0, "vmin": -200.0,
                       "mean": -50.0, "var": 1600.0, "n": 500}],
        })
        shared_dir = tmp_path / "shared_gamd"
        _write_shared_globals(shared_dir / "shared_gamd_setup_globals.json", {
            "sigma0_Total": 6.0, "k0_Total": 0.5, "Vmax_Total": 100.0,
            "Vmin_Total": -200.0, "threshold_energy_Total": 100.0,
        })
        return epoch_dir, shared_dir

    def test_recalibration_runs_without_a_ladder(self, tmp_path):
        """Control: the same fixture with no λ must NOT be skipped by the
        ladder guard -- otherwise the guard would be vacuous."""
        epoch_dir, shared_dir = self._ready_epoch(tmp_path)
        report = _maybe_recalibrate_gamd_boost(
            0, epoch_dir, _args(state_gamd_lambdas=[0.0, 0.0]), shared_dir)
        assert report.get("status") != "skipped_lambda_ladder", report

    def test_noop_when_args_state_gamd_lambdas_has_a_rung(self, tmp_path):
        epoch_dir, shared_dir = self._ready_epoch(tmp_path)
        report = _maybe_recalibrate_gamd_boost(
            0, epoch_dir, _args(state_gamd_lambdas=[0.0, 0.5, 1.0]), shared_dir)
        assert report["status"] == "skipped_lambda_ladder", report

    def test_noop_when_the_state_registry_carries_a_rung(self, tmp_path):
        """The adaptive campaign's λ lives in state_registry.csv, not
        necessarily on args -- the guard must read it too."""
        import csv as _csv
        epoch_dir, shared_dir = self._ready_epoch(tmp_path)
        with (tmp_path / "state_registry.csv").open("w", newline="") as fh:
            w = _csv.DictWriter(fh, fieldnames=["state_id", "gamd_lambda"])
            w.writeheader()
            w.writerow({"state_id": 0, "gamd_lambda": 0.0})
            w.writerow({"state_id": 1, "gamd_lambda": 1.0})
        report = _maybe_recalibrate_gamd_boost(0, epoch_dir, _args(), shared_dir)
        assert report["status"] == "skipped_lambda_ladder", report

    def test_guard_precedes_the_stats_scan(self, tmp_path):
        """The ladder verdict must not depend on whether epoch 0 happened to
        write stats files -- otherwise a ladder run reports
        'skipped_no_stats' and the real reason is invisible."""
        epoch_dir = tmp_path / "epoch_000"
        epoch_dir.mkdir()
        report = _maybe_recalibrate_gamd_boost(
            0, epoch_dir, _args(state_gamd_lambdas=[1.0]), tmp_path / "shared")
        assert report["status"] == "skipped_lambda_ladder", report

    def test_docstring_no_longer_claims_envelope_independent_reweighting(self):
        """The docstring asserted 'Reweighting validity does not depend on
        which envelope was active for a given frame' -- false under a ladder,
        and the sentence that justified the whole feature."""
        doc = _maybe_recalibrate_gamd_boost.__doc__ or ""
        assert "does not depend on which envelope was active" not in doc
        assert "ladder" in doc.lower()


class TestMaybeRecalibrateGamdBoostHappyPath:
    def _window_stats_payload(self):
        # Two "workers" (baseline + a topup segment) each contribute one window's
        # worth of samples for the same boost group; pooling them must match
        # gareus.gamd_calibration.pool_window_stats exactly.
        w0 = WindowEnergyStats(group="Total", window=0, vmax=100.0, vmin=-200.0, mean=-60.0, var=1500.0, n=400)
        w1 = WindowEnergyStats(group="Total", window=1, vmax=80.0, vmin=-180.0, mean=-40.0, var=1700.0, n=600)
        return w0, w1

    def test_recalibrates_from_two_segment_stats_files_and_bumps_version(self, tmp_path):
        w0, w1 = self._window_stats_payload()
        epoch_dir = tmp_path / "epoch_000"
        _write_stats_file(epoch_dir / "baseline" / "gamd_production_envelope_stats.json", {
            "Total": [{"group": w0.group, "window": w0.window, "vmax": w0.vmax, "vmin": w0.vmin, "mean": w0.mean, "var": w0.var, "n": w0.n}],
        })
        _write_stats_file(epoch_dir / "topup_0" / "gamd_production_envelope_stats.json", {
            "Total": [{"group": w1.group, "window": w1.window, "vmax": w1.vmax, "vmin": w1.vmin, "mean": w1.mean, "var": w1.var, "n": w1.n}],
        })
        shared_dir = tmp_path / "shared_gamd"
        globals_path = shared_dir / "shared_gamd_setup_globals.json"
        _write_shared_globals(globals_path, {
            "sigma0_Total": 6.0,
            "Vmax_Total": 999.0, "Vmin_Total": -999.0, "Vavg_Total": 0.0,
            "sigmaV_Total": 999.0, "k0_Total": 0.5, "threshold_energy_Total": 999.0,
            "stepCount": 12345.0,  # non-physics global -- must survive untouched
        })

        report = _maybe_recalibrate_gamd_boost(0, epoch_dir, _args(gamd_boost_type="lower-total"), shared_dir)

        assert report["status"] == "recalibrated"
        assert report["envelope_version"] == 2
        assert report["n_stats_files"] == 2
        assert "Total" in report["groups"]
        assert report["groups"]["Total"]["status"] == "recalibrated"

        # Independently recompute what the pooled/recalibrated values *should* be
        # using the exact same OpenMM-free primitives, and check the file on disk
        # matches -- this is what actually proves the merge+recalibrate did the
        # right physics, not just "some JSON got written".
        expected_envelope = pool_window_stats([w0, w1])
        expected_calib = compute_group_calibration("lower-total", expected_envelope, sigma0=6.0)

        on_disk = json.loads(globals_path.read_text())
        g = on_disk["all_globals"]
        assert g["Vmax_Total"] == pytest.approx(expected_calib.vmax)
        assert g["Vmin_Total"] == pytest.approx(expected_calib.vmin)
        assert g["Vavg_Total"] == pytest.approx(expected_calib.vavg)
        assert g["sigmaV_Total"] == pytest.approx(expected_calib.sigmav)
        assert g["k0_Total"] == pytest.approx(expected_calib.k0)
        assert g["threshold_energy_Total"] == pytest.approx(expected_calib.threshold_energy)
        # sigma0 is the target, not an output -- must be unchanged.
        assert g["sigma0_Total"] == pytest.approx(6.0)
        # Non-physics bookkeeping globals must survive untouched.
        assert g["stepCount"] == pytest.approx(12345.0)

        assert on_disk["gamd_envelope_version"] == 2
        assert len(on_disk["recalibration_history"]) == 1
        assert on_disk["recalibration_history"][0]["recalibrated_from_epoch"] == 0

    def test_second_recalibration_bumps_version_again(self, tmp_path):
        w0, w1 = self._window_stats_payload()
        epoch_dir = tmp_path / "epoch_000"
        _write_stats_file(epoch_dir / "gamd_production_envelope_stats.json", {
            "Total": [
                {"group": w0.group, "window": w0.window, "vmax": w0.vmax, "vmin": w0.vmin, "mean": w0.mean, "var": w0.var, "n": w0.n},
                {"group": w1.group, "window": w1.window, "vmax": w1.vmax, "vmin": w1.vmin, "mean": w1.mean, "var": w1.var, "n": w1.n},
            ],
        })
        shared_dir = tmp_path / "shared_gamd"
        globals_path = shared_dir / "shared_gamd_setup_globals.json"
        _write_shared_globals(globals_path, {
            "sigma0_Total": 6.0, "Vmax_Total": 0.0, "Vmin_Total": 0.0, "Vavg_Total": 0.0,
            "sigmaV_Total": 0.0, "k0_Total": 0.0, "threshold_energy_Total": 0.0,
        })

        args = _args(gamd_boost_type="lower-total")
        first = _maybe_recalibrate_gamd_boost(0, epoch_dir, args, shared_dir)
        second = _maybe_recalibrate_gamd_boost(0, epoch_dir, args, shared_dir)

        assert first["envelope_version"] == 2
        assert second["envelope_version"] == 3
        on_disk = json.loads(globals_path.read_text())
        assert len(on_disk["recalibration_history"]) == 2
