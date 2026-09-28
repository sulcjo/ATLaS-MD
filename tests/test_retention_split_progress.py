from __future__ import annotations

import json

import gareus.retention as retention


def _write(p, events):
    p.write_text("".join(json.dumps(e) + "\n" for e in events))


def _sample():
    return [
        {"event": "progress", "step": 1, "aggregate_sim_time_ns": 0.1},
        {"event": "distances", "step": 250, "cv_mean_A": 0.4, "distances": [{"replica": 0}] * 50,
         "dashboard": {"exchange_stats": {"pairs": {"a": 1}}}},
        {"event": "run_complete"},
    ]


def test_dry_run_reports_and_keeps_file(tmp_path):
    p = tmp_path / "progress.jsonl"
    _write(p, _sample())
    before = p.read_bytes()
    r = retention.split_progress_jsonl(p, apply=False)
    assert r["status"] == "dry_run" and r["distances_lines"] == 1 and r["bytes_after"] < r["bytes_before"]
    assert p.read_bytes() == before


def test_apply_rewrites_distances_as_summary_and_keeps_others_byte_for_byte(tmp_path):
    p = tmp_path / "progress.jsonl"
    _write(p, _sample())
    lines_before = p.read_text().splitlines()
    retention.split_progress_jsonl(p, apply=True)
    lines_after = p.read_text().splitlines()
    assert lines_after[0] == lines_before[0] and lines_after[2] == lines_before[2]
    e = json.loads(lines_after[1])
    assert e["event"] == "distances_summary" and e["cv_mean_A"] == 0.4
    assert "distances" not in e and "dashboard" not in e


def test_aborts_if_file_grows_during_rewrite(tmp_path, monkeypatch):
    p = tmp_path / "progress.jsonl"
    _write(p, _sample())
    real = retention._rewrite_lines

    def growing(src, dst):
        out = real(src, dst)
        with p.open("a") as fh:
            fh.write(json.dumps({"event": "progress"}) + "\n")
        return out
    monkeypatch.setattr(retention, "_rewrite_lines", growing)
    r = retention.split_progress_jsonl(p, apply=True)
    assert r["status"] == "changed_during_rewrite"
    assert json.loads(p.read_text().splitlines()[1])["event"] == "distances"
    assert not list(tmp_path.glob("*.tmp"))
