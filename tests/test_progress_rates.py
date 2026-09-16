"""Rates must describe this process's segment, not the whole campaign.

``step`` is cumulative across a resumed chain; ``elapsed`` restarts at zero each
job. Dividing one by the other reported 5,293 ns/day against an actual ~283 on
chignolin_7 job 2411974 -- high by a factor of ~19.
"""
import types

import pytest

from gareus.progress import GuiProgressSink


class _Cap:
    """Stands in for the JSONL writer so payloads are captured, not written."""

    def __init__(self, sink_list):
        self._out = sink_list

    def write_json(self, payload):
        self._out.append(dict(payload))

    def close(self):
        pass


def _sink(tmp_path):
    args = types.SimpleNamespace(progress_mode="jsonl", tui_mode="plain")
    sink = GuiProgressSink(tmp_path, args)
    captured = []
    sink.handle = _Cap(captured)
    return sink, captured


def _report(sink, captured, **kwargs):
    sink.progress(force=True, **kwargs)


def test_rates_use_only_steps_since_the_segment_started(tmp_path):
    sink, captured = _sink(tmp_path)
    _report(sink, captured, phase="prod", step=1_000_000,
            total_steps=2_000_000, timestep_fs=3.5, n_replicas=64)
    _report(sink, captured, phase="prod", step=1_010_000,
            total_steps=2_000_000, timestep_fs=3.5, n_replicas=64)

    last = captured[-1]
    assert last["segment_steps"] == 10_000
    assert last["sim_time_ns"] == pytest.approx(1_010_000 * 3.5 / 1e6)
    # Assert the identity the fix establishes, not a magnitude: the rate must be
    # segment_steps/elapsed. A bound like "< 1e9" is machine-speed dependent --
    # two calls microseconds apart legitimately give a huge rate, and that bound
    # passed alone but failed inside the full suite.
    assert last["steps_per_s"] == pytest.approx(
        last["segment_steps"] / last["elapsed_s"], rel=1e-9)
    # The defect being fixed used step_int, which is 101x larger here.
    assert last["steps_per_s"] < 0.5 * last["step"] / last["elapsed_s"]


def test_first_report_of_a_segment_omits_rates_rather_than_dividing_by_zero(tmp_path):
    sink, captured = _sink(tmp_path)
    _report(sink, captured, phase="prod", step=500_000,
            total_steps=2_000_000, timestep_fs=3.5, n_replicas=64)

    first = captured[0]
    assert first["segment_steps"] == 0
    assert "ns_per_day" not in first
    assert "steps_per_s" not in first
    assert first["eta_s"] is None


def test_cumulative_simulated_time_is_still_cumulative(tmp_path):
    sink, captured = _sink(tmp_path)
    _report(sink, captured, phase="prod", step=900_000,
            total_steps=2_000_000, timestep_fs=3.5, n_replicas=64)

    p = captured[0]
    assert p["sim_time_ns"] == pytest.approx(900_000 * 3.5 / 1e6)
    assert p["aggregate_sim_time_ns"] == pytest.approx(900_000 * 3.5 / 1e6 * 64)


def test_a_resumed_segment_does_not_inherit_the_previous_job_rate(tmp_path):
    """The regression that motivated this: a resume must not inflate the rate.

    Two sinks, same cumulative step range, different baselines: the one that
    starts mid-campaign must report the same rate as one starting from zero for
    the same number of steps actually run.
    """
    sink_a, cap_a = _sink(tmp_path)
    _report(sink_a, cap_a, phase="prod", step=0, total_steps=2_000_000,
            timestep_fs=3.5, n_replicas=64)
    _report(sink_a, cap_a, phase="prod", step=10_000, total_steps=2_000_000,
            timestep_fs=3.5, n_replicas=64)

    sink_b, cap_b = _sink(tmp_path)
    _report(sink_b, cap_b, phase="prod", step=1_500_000, total_steps=2_000_000,
            timestep_fs=3.5, n_replicas=64)
    _report(sink_b, cap_b, phase="prod", step=1_510_000, total_steps=2_000_000,
            timestep_fs=3.5, n_replicas=64)

    assert cap_a[-1]["segment_steps"] == cap_b[-1]["segment_steps"] == 10_000
    # Same work done => same order of magnitude rate, regardless of where in the
    # campaign the segment sits.
    ratio = cap_b[-1]["steps_per_s"] / cap_a[-1]["steps_per_s"]
    assert 0.1 < ratio < 10.0, f"resumed segment rate differs by {ratio:.1f}x"
