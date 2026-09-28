from __future__ import annotations

import json

from gareus.io import RotatingJsonlWriter


def _lines(p):
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


def test_writes_compact_lines_and_flushes(tmp_path):
    w = RotatingJsonlWriter(tmp_path / "live.jsonl", max_bytes=10_000)
    w.write_json({"b": 1, "a": 2})
    assert (tmp_path / "live.jsonl").read_text() == '{"a":2,"b":1}\n'
    w.close()


def test_rotates_into_single_previous_file(tmp_path):
    p = tmp_path / "live.jsonl"
    w = RotatingJsonlWriter(p, max_bytes=200)
    for i in range(40):
        w.write_json({"i": i, "pad": "x" * 20})
    w.close()
    prev = tmp_path / "live.1.jsonl"
    assert w.previous_path == prev
    assert p.stat().st_size <= 200 and prev.stat().st_size <= 200
    seq = [e["i"] for e in _lines(prev) + _lines(p)]
    assert seq == list(range(seq[0], 40))  # contiguous, newest last
    assert sorted(x.name for x in tmp_path.iterdir()) == ["live.1.jsonl", "live.jsonl"]


def test_appends_across_reopen(tmp_path):
    p = tmp_path / "live.jsonl"
    RotatingJsonlWriter(p, max_bytes=10_000).write_json({"i": 0})
    w = RotatingJsonlWriter(p, max_bytes=10_000)
    w.write_json({"i": 1})
    w.close()
    assert [e["i"] for e in _lines(p)] == [0, 1]


def test_single_line_larger_than_limit_is_still_written(tmp_path):
    p = tmp_path / "live.jsonl"
    w = RotatingJsonlWriter(p, max_bytes=10)
    w.write_json({"big": "y" * 100})
    w.write_json({"big": "z" * 100})
    w.close()
    assert len(_lines(p)) == 1 and len(_lines(tmp_path / "live.1.jsonl")) == 1
