"""CLI wiring for the NPT-correction flags.

Covers: argparse defaults (auto / 0.01), the three argument-level validation
failures (NVT + explicit backend, native + boosted run mode, bad fraction),
YAML-config reachability, and provenance recording of both keys (resume must
not silently change who owns volume moves).
"""

from __future__ import annotations

import json

import pytest

from gareus.cli import parse_args

MINIMAL = ["--seq", "AA", "--out", "unused"]


def test_defaults_are_auto_and_one_percent():
    args = parse_args(MINIMAL)
    assert args.npt_barostat_backend == "auto"
    assert args.barostat_volume_step_fraction == 0.01
    assert args.production_ensemble == "npt"


def test_flags_reachable_from_cli():
    args = parse_args(MINIMAL + ["--npt-barostat-backend", "biased_mc",
                                  "--barostat-volume-step-fraction", "0.02"])
    assert args.npt_barostat_backend == "biased_mc"
    assert args.barostat_volume_step_fraction == 0.02


def test_nvt_with_explicit_backend_is_rejected_at_parse_time():
    with pytest.raises(ValueError, match="npt-barostat-backend"):
        parse_args(MINIMAL + ["--production-ensemble", "nvt",
                              "--npt-barostat-backend", "native"])


def test_explicit_native_with_boosted_run_mode_is_rejected_at_parse_time():
    with pytest.raises(ValueError, match="native"):
        parse_args(MINIMAL + ["--npt-barostat-backend", "native"])


def test_explicit_native_with_cmd_run_mode_is_accepted():
    args = parse_args(MINIMAL + ["--run-mode", "cmd",
                                 "--npt-barostat-backend", "native"])
    assert args.npt_barostat_backend == "native"


def test_bad_volume_step_fraction_is_rejected():
    for bad in ("0", "-0.1", "1.0", "nan"):
        with pytest.raises(ValueError, match="volume-step-fraction"):
            parse_args(MINIMAL + ["--barostat-volume-step-fraction", bad])


def test_flags_reachable_from_yaml_config(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "npt_barostat_backend: biased_mc\n"
        "barostat_volume_step_fraction: 0.05\n"
    )
    args = parse_args(MINIMAL + ["--config", str(cfg)])
    assert args.npt_barostat_backend == "biased_mc"
    assert args.barostat_volume_step_fraction == 0.05


def test_method_settings_records_the_npt_keys(tmp_path):
    from gareus.provenance import _method_settings

    args = parse_args(MINIMAL + ["--npt-barostat-backend", "biased_mc",
                                 "--barostat-volume-step-fraction", "0.03"])
    settings = _method_settings(args)
    assert settings["npt_barostat_backend"] == "biased_mc"
    assert settings["barostat_volume_step_fraction"] == 0.03
    # and both survive a JSON round trip (they land in run_manifest.json)
    assert json.loads(json.dumps(settings))["npt_barostat_backend"] == "biased_mc"
