import json

from gareus.dashboard.sidecar import SidecarCache, find_run_root


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _make_run(tmp_path):
    root = tmp_path / "chignolin_x"
    segment = root / "adaptive_production" / "epoch_001" / "baseline"
    segment.mkdir(parents=True)
    _write(root / "adaptive_production" / "adaptive_runtime_pool.json",
           {"total_ns": 15000.0, "used_ns": 7500.0, "remaining_ns": 7500.0, "events": []})
    _write(root / "global_shared_gamd_setup" / "shared_gamd_setup_globals.json",
           {"joint_envelope": {"Dihedral": {"sigma0_kj_mol": 12.552, "sigmaV_kj_mol": 10.77, "k0": 1.0}}})
    return root, segment


def test_find_run_root_walks_up_from_a_segment_directory(tmp_path):
    root, segment = _make_run(tmp_path)
    assert find_run_root(segment) == root


def test_find_run_root_falls_back_to_the_given_directory(tmp_path):
    lonely = tmp_path / "not_a_run"
    lonely.mkdir()
    assert find_run_root(lonely) == lonely


def test_snapshot_reads_pool_and_gamd_payloads(tmp_path):
    _root, segment = _make_run(tmp_path)
    snap = SidecarCache(segment).snapshot(now=1000.0)
    assert snap.pool["total_ns"] == 15000.0
    assert snap.gamd["joint_envelope"]["Dihedral"]["k0"] == 1.0
    assert snap.errors == ()


def test_snapshot_is_cached_until_the_min_interval_elapses(tmp_path):
    root, segment = _make_run(tmp_path)
    cache = SidecarCache(segment, min_interval_s=5.0)
    first = cache.snapshot(now=1000.0)
    _write(root / "adaptive_production" / "adaptive_runtime_pool.json",
           {"total_ns": 15000.0, "used_ns": 9999.0, "remaining_ns": 5001.0, "events": []})
    assert cache.snapshot(now=1002.0) is first          # inside the interval
    assert cache.snapshot(now=1006.0).pool["used_ns"] == 9999.0


def test_snapshot_reports_missing_files_as_none_not_an_error(tmp_path):
    bare = tmp_path / "bare"
    bare.mkdir()
    snap = SidecarCache(bare).snapshot(now=1.0)
    assert snap.pool is None and snap.gamd is None
    assert snap.errors == ()


def test_snapshot_records_a_corrupt_file_as_an_error_and_keeps_going(tmp_path):
    root, segment = _make_run(tmp_path)
    (root / "adaptive_production" / "adaptive_runtime_pool.json").write_text("{not json")
    snap = SidecarCache(segment).snapshot(now=1.0)
    assert snap.pool is None
    assert any("adaptive_runtime_pool.json" in e for e in snap.errors)
    assert snap.gamd is not None            # one bad file does not poison the rest


def test_snapshot_finds_per_epoch_seeding_quality(tmp_path):
    root, segment = _make_run(tmp_path)
    _write(segment.parent / "setup" / "us_starting_structure_quality.json",
           {"windows": [{"window": 17, "status": "bad", "direction": "pull_crash_fallback_unpulled"}]})
    snap = SidecarCache(segment).snapshot(now=1.0)
    assert snap.seeding_quality["windows"][0]["window"] == 17
