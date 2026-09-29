import json

from gareus.mbar_analysis.memory import MemoryReporter


def test_memory_report_writes_required_fields(tmp_path):
    path = tmp_path / 'memory.jsonl'
    reporter = MemoryReporter(path)
    reporter.record('startup')
    reporter.close()
    row = json.loads(path.read_text().splitlines()[0])
    assert row['phase'] == 'startup'
    assert row['pid'] > 0
    assert isinstance(row['rss_bytes'], int)


def test_memory_report_is_best_effort_when_proc_unavailable(tmp_path, monkeypatch):
    path = tmp_path / 'memory.jsonl'
    reporter = MemoryReporter(path)
    monkeypatch.setattr(reporter, '_read_proc', lambda: (_ for _ in ()).throw(OSError('no proc')))
    reporter.record('startup')
    reporter.close()
    assert path.read_text() == ''
