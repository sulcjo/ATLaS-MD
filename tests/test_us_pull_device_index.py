"""The umbrella pull's worker GPUs are chosen independently of the single-GPU setup contexts.

Before --us-pull-device-index, gareus/seeding.py took the pull workers' device list from the
*setup* platform properties, which setup_platform_and_properties() deliberately restricts to the
first entry of --device-index. On chignolin_8 (4x L40S) that put all 28 pull workers on GPU 0 for
2h11 per phase start while GPUs 1-3 idled. Widening --setup-device-index instead would turn every
other setup context (minimisation, NPT equilibration, shared GaMD setup) into a multi-GPU context,
which is not what a 21k-atom system wants.
"""
from __future__ import annotations

import types

from gareus.seeding import us_pull_device_tokens


def _args(**over):
    a = types.SimpleNamespace(us_pull_device_index="", setup_device_index="", device_index="0,1,2,3")
    for k, v in over.items():
        setattr(a, k, v)
    return a


def test_explicit_pull_device_list_wins_over_setup_props():
    assert us_pull_device_tokens(_args(us_pull_device_index="0,1,2,3"), {"DeviceIndex": "0"}) == ["0", "1", "2", "3"]
    assert us_pull_device_tokens(_args(us_pull_device_index=" 2, 3 "), {"DeviceIndex": "0"}) == ["2", "3"]


def test_default_keeps_the_setup_platform_device():
    # Unchanged behaviour when the flag is not given: the setup context's own device.
    assert us_pull_device_tokens(_args(), {"DeviceIndex": "1"}) == ["1"]


def test_fallbacks_when_setup_props_carry_no_device():
    assert us_pull_device_tokens(_args(setup_device_index="3"), {}) == ["3"]
    assert us_pull_device_tokens(_args(device_index="1,2"), {}) == ["1"]          # first entry only, as before
    assert us_pull_device_tokens(_args(device_index=""), {}) == ["0"]


def test_cli_flag_exists_with_empty_default():
    from gareus.cli import parse_args
    assert parse_args(["--seq", "DPETG", "--us-pull-device-index", "0,1,2,3"]).us_pull_device_index == "0,1,2,3"
    assert parse_args(["--seq", "DPETG"]).us_pull_device_index == ""
