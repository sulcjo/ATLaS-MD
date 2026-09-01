"""The start-bias `bad` thresholds must be reachable from the CLI.

A window whose starting structure sits far from its umbrella centre is flagged
`bad`, and with `--us-auto-drop-bad-windows` a `bad` window is dropped before
production. The threshold that decides this was not settable:

* the primary-CV threshold was a bare literal `5.0` in the quality loop;
* the secondary-CV thresholds existed as `args` attributes but were assigned
  unconditionally by `_apply_v2_compat_shims`, with no argparse flag -- the same
  "dropped tuning knob" pattern CLAUDE.md records for
  `--ap-epoch0-step-fraction` and `require_convergence_before_final`.

Real incident (chignolin_6, 2026-09-01). The final phase dropped windows 15 and
32 for start biases of 6.67 and 6.77 kcal/mol -- about 11 kT, strained but
routinely relaxed within picoseconds -- while their potential energies were
entirely healthy (robust-z 0.26 and 0.18). Window 32 was the bridge window the
previous epoch had created specifically to repair weak edge 5-32, so dropping it
reproduced the zero-sample state and the three unmeasurable overlap edges that
had made the quality gate demand more sampling in the first place.

Raising the bar required editing the source. It should require a flag.
"""
from __future__ import annotations

import pytest

from gareus.cli import parse_args
from gareus.seeding import classify_starting_umbrella_bias

BASE = ["--seq", "GYDPETGTWG", "--out", "/tmp/_x"]


def test_the_primary_bad_bias_threshold_has_a_flag_and_keeps_its_old_default():
    assert parse_args(BASE).us_start_primary_bad_bias_kcal == pytest.approx(5.0)
    got = parse_args(BASE + ["--us-start-primary-bad-bias-kcal", "15"])
    assert got.us_start_primary_bad_bias_kcal == pytest.approx(15.0)


def test_the_secondary_bad_bias_threshold_survives_the_compat_shim():
    """The shim used to assign this unconditionally, clobbering any override."""
    assert parse_args(BASE).us_2d_start_secondary_bad_bias_kcal == pytest.approx(5.0)
    got = parse_args(BASE + ["--us-2d-start-secondary-bad-bias-kcal", "15"])
    assert got.us_2d_start_secondary_bad_bias_kcal == pytest.approx(15.0), (
        "the compat shim overwrote the parsed value"
    )


def test_the_warn_thresholds_are_reachable_too():
    a = parse_args(BASE + ["--us-start-primary-warn-bias-kcal", "2.5",
                           "--us-2d-start-secondary-warn-bias-kcal", "3.5"])
    assert a.us_start_primary_warn_bias_kcal == pytest.approx(2.5)
    assert a.us_2d_start_secondary_warn_bias_kcal == pytest.approx(3.5)


@pytest.mark.parametrize("bias,warn,bad,expect", [
    (0.5, 1.0, 5.0, None),      # comfortably on target
    (2.0, 1.0, 5.0, "warn"),    # strained
    (6.7, 1.0, 5.0, "bad"),     # the real chignolin_6 w15 value at the old bar
    (6.7, 1.0, 15.0, "warn"),   # ... and at the raised bar: kept, still flagged
    (20.0, 1.0, 15.0, "bad"),   # genuinely unreachable is still caught
])
def test_the_bias_classifier_respects_the_thresholds_it_is_given(bias, warn, bad, expect):
    status, message = classify_starting_umbrella_bias(bias, warn_kcal=warn, bad_kcal=bad)
    assert status == expect
    assert bool(message) == (expect is not None)


def test_a_non_finite_bias_is_not_silently_treated_as_on_target():
    status, _ = classify_starting_umbrella_bias(float("nan"), warn_kcal=1.0, bad_kcal=5.0)
    assert status is None, "NaN must not classify; the caller skips it as before"
