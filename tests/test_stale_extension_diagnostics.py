"""A finished extension round whose samples outgrew its diagnostics is recollected on resume.

chignolin_9 final_extension_001: the round was re-entered with a larger target, its MD
finished right as SIGTERM arrived, the driver returned interrupted_after_checkpoint before
collecting, and the next job moved to round 2 trusting diagnostics built from 10,369 of
the 15,796 samples per state.
"""
import json
from pathlib import Path

import gareus.adaptive_production as ap


def _round(adaptive: Path, n: int, prod_done: int) -> Path:
    d = adaptive / f"final_extension_{n:03d}"
    (d / "checkpoints").mkdir(parents=True)
    (d / "checkpoints" / "production_checkpoint_manifest.json").write_text(json.dumps({"prod_done": prod_done}))
    return d


def _set_prod(d: Path, prod_done: int) -> None:
    (d / "checkpoints" / "production_checkpoint_manifest.json").write_text(json.dumps({"prod_done": prod_done}))


def test_a_round_is_fresh_after_its_diagnostics_are_recorded(tmp_path):
    d = _round(tmp_path, 1, 1000)
    ap._record_extension_diagnostics_state(d)
    assert ap._extension_diagnostics_are_stale(d) is False


def test_a_round_continued_after_its_diagnostics_is_stale(tmp_path):
    d = _round(tmp_path, 1, 1000)
    ap._record_extension_diagnostics_state(d)
    _set_prod(d, 12_110_000)
    assert ap._extension_diagnostics_are_stale(d) is True


def test_a_round_without_a_record_is_stale_only_if_it_has_a_checkpoint(tmp_path):
    assert ap._extension_diagnostics_are_stale(_round(tmp_path, 1, 1000)) is True
    bare = tmp_path / "final_extension_002"
    bare.mkdir()
    assert ap._extension_diagnostics_are_stale(bare) is False


def test_refresh_recollects_only_stale_rounds(tmp_path, monkeypatch):
    fresh = _round(tmp_path, 1, 1000)
    ap._record_extension_diagnostics_state(fresh)
    stale = _round(tmp_path, 2, 2000)
    collected = []
    monkeypatch.setattr(ap, "collect_segmented_epoch_diagnostics",
                        lambda d, reg, pol: collected.append(Path(d).name) or {})
    prior = [{"extension": 1, "dir": str(fresh)}, {"extension": 2, "dir": str(stale)}]
    assert ap._refresh_stale_extension_diagnostics(tmp_path, prior, None, None) == ["final_extension_002"]
    assert collected == ["final_extension_002"]
    assert ap._extension_diagnostics_are_stale(stale) is False
    assert ap._refresh_stale_extension_diagnostics(tmp_path, prior, None, None) == []


def test_refresh_falls_back_to_the_round_number_when_dir_is_foreign(tmp_path, monkeypatch):
    stale = _round(tmp_path, 1, 5)
    monkeypatch.setattr(ap, "collect_segmented_epoch_diagnostics", lambda d, reg, pol: {})
    prior = [{"extension": 1, "dir": "/home/elsewhere/final_extension_001"}]   # a synced copy's path
    assert ap._refresh_stale_extension_diagnostics(tmp_path, prior, None, None) == ["final_extension_001"]
    assert ap._extension_diagnostics_are_stale(stale) is False


def test_a_failed_refresh_warns_and_continues(tmp_path, monkeypatch, capsys):
    _round(tmp_path, 1, 5)

    def boom(*a, **k):
        raise OSError("disk")

    monkeypatch.setattr(ap, "collect_segmented_epoch_diagnostics", boom)
    assert ap._refresh_stale_extension_diagnostics(tmp_path, [{"extension": 1}], None, None) == []
    assert "could not refresh" in capsys.readouterr().out


def test_the_driver_refreshes_prior_rounds_and_records_each_collected_round():
    import inspect
    src = inspect.getsource(ap.run_adaptive_production_auto_loop)
    refresh = src.index("_refresh_stale_extension_diagnostics(adaptive_dir, prior_extension_summaries")
    assert refresh < src.index("for ext_index in range(start_ext_round, ext_round_total)")
    collect = src.index("ext_diag = collect_segmented_epoch_diagnostics(ext_dir, registry, policy)")
    assert src.index("_record_extension_diagnostics_state(ext_dir)", collect) - collect < 120
