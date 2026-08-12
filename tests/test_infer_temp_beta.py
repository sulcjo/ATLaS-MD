"""infer_temp_beta (analyze_gareus_mbar.py) coverage-gap regression tests.

Found by a physics/math correctness audit: infer_temp_beta resolves
temperature/beta from (in order) an already-loaded `arrays` NPZ, then a
`meta` dict, then a `run_args.json` file search, then a hardcoded 300 K
default. Two real coverage gaps and one silent-default issue:

1. The `meta` dict branch only ever checked TOP-LEVEL keys
   (`temperature_K`/`temperature_k`/`temperature`). A real call site passes
   a parsed `run_manifest.json` as `meta` (see
   `load_parquet_adaptive_union`'s `infer_temp_beta(adaptive_dir, meta)`,
   where `meta = rjson(adaptive_dir.parent / 'run_manifest.json', {})`), and
   `run_manifest.json` only ever nests temperature under
   `method_settings.temperature_k` / `resolved_args.temperature_k`
   (`gareus/provenance.py`'s `_method_settings`/`_public_args`) -- never at
   the top level. So this branch always missed a real, present value and
   fell through to a less-reliable later branch.
2. The `run_args.json` file-fallback only checked `prod` and `prod.parent`,
   one directory level too shallow for an epoch_dir-style caller
   (`infer_temp_beta(epoch_dir, ep_meta)` in `load_parquet_adaptive_union`,
   where `epoch_dir` = `<run>/adaptive_production/epoch_NNN` and the real
   `run_args.json` lives at `<run>/`, i.e. `epoch_dir.parent.parent`).
3. The final hardcoded `t=300.0` fallback fired with zero warning. It now
   appends a note to `meta['load_notes']` (the file's existing
   loader-warnings convention -- see `load_npz`/`load_csv`) when `meta` is a
   dict, and also raises a `RuntimeWarning` via the stdlib `warnings`
   module so it surfaces to a user running the real CLI even when the
   particular `meta` object at that call site is discarded.
"""
import math

import pytest

pytest.importorskip("analyze_gareus_mbar")

from analyze_gareus_mbar import K_B_KJ_PER_MOL_K, infer_temp_beta


def _expected_beta(temp_k: float) -> float:
    return 1.0 / (K_B_KJ_PER_MOL_K * temp_k)


def test_resolves_temperature_nested_under_method_settings(tmp_path):
    """run_manifest.json-shaped meta: temperature ONLY nested, no top-level key.

    A conflicting run_args.json (275 K) sits right next to `prod` too, so this
    also locks precedence: the nested meta-dict branch must win outright
    rather than merely "not hitting the 300K default" -- proving it doesn't
    fall through to the (also-successful, but wrong-value) file search.
    """
    prod = tmp_path / "adaptive_production"
    prod.mkdir()
    (tmp_path / "run_args.json").write_text('{"temperature_k": 275.0}', encoding="utf-8")
    meta = {
        "resolved_args": {"seed": 1},
        "method_settings": {"temperature_k": 310.0},
    }

    temp, beta = infer_temp_beta(prod, meta)

    assert math.isclose(temp, 310.0, rel_tol=1e-9)
    assert math.isclose(beta, _expected_beta(310.0), rel_tol=1e-9)
    # The real value was found; the 300K-fallback note must NOT have fired.
    assert "load_notes" not in meta


def test_resolves_temperature_nested_under_resolved_args(tmp_path):
    """Same manifest shape, but temperature nested under resolved_args instead."""
    prod = tmp_path / "adaptive_production"
    prod.mkdir()
    meta = {"resolved_args": {"temperature_k": 315.5}, "method_settings": {}}

    temp, beta = infer_temp_beta(prod, meta)

    assert math.isclose(temp, 315.5, rel_tol=1e-9)
    assert math.isclose(beta, _expected_beta(315.5), rel_tol=1e-9)


def test_top_level_meta_temperature_still_resolves(tmp_path):
    """Regression guard: existing top-level meta-key resolution is unaffected."""
    prod = tmp_path / "final_production"
    prod.mkdir()
    meta = {"temperature_K": 305.0}

    temp, beta = infer_temp_beta(prod, meta)

    assert math.isclose(temp, 305.0, rel_tol=1e-9)
    assert math.isclose(beta, _expected_beta(305.0), rel_tol=1e-9)
    assert "load_notes" not in meta


def test_epoch_dir_falls_back_to_run_args_two_levels_up(tmp_path):
    """epoch_dir-style caller: real run_args.json sits two levels above epoch_dir.

    Layout: <run_root>/run_args.json, <run_root>/adaptive_production/epoch_000.
    epoch_dir.parent (adaptive_production/) has no run_args.json of its own;
    only epoch_dir.parent.parent (run_root) does.
    """
    run_root = tmp_path / "chignolin_run"
    adaptive_dir = run_root / "adaptive_production"
    epoch_dir = adaptive_dir / "epoch_000"
    epoch_dir.mkdir(parents=True)
    (run_root / "run_args.json").write_text('{"temperature_k": 298.0}', encoding="utf-8")

    temp, beta = infer_temp_beta(epoch_dir, {})

    assert math.isclose(temp, 298.0, rel_tol=1e-9)
    assert math.isclose(beta, _expected_beta(298.0), rel_tol=1e-9)


def test_file_search_prefers_shallower_run_args_json(tmp_path):
    """When run_args.json exists at both one and two levels up, the shallower
    (one-level-up) file wins -- locking the shallowest-first search order.
    """
    run_root = tmp_path / "chignolin_run"
    adaptive_dir = run_root / "adaptive_production"
    epoch_dir = adaptive_dir / "epoch_000"
    epoch_dir.mkdir(parents=True)
    (run_root / "run_args.json").write_text('{"temperature_k": 280.0}', encoding="utf-8")
    (adaptive_dir / "run_args.json").write_text('{"temperature_k": 320.0}', encoding="utf-8")

    temp, beta = infer_temp_beta(epoch_dir, {})

    assert math.isclose(temp, 320.0, rel_tol=1e-9)
    assert math.isclose(beta, _expected_beta(320.0), rel_tol=1e-9)


def test_final_fallback_warns_and_records_load_note(tmp_path):
    """Nothing resolvable anywhere: still returns 300.0, but now signals it fired."""
    prod = tmp_path / "isolated_run" / "final_production"
    prod.mkdir(parents=True)
    meta: dict = {}

    with pytest.warns(RuntimeWarning, match="falling back to default"):
        temp, beta = infer_temp_beta(prod, meta)

    assert math.isclose(temp, 300.0, rel_tol=1e-9)
    assert math.isclose(beta, _expected_beta(300.0), rel_tol=1e-9)
    # Observable signal #2: a note recorded via the file's existing
    # load_notes warnings convention (meta is mutated in place).
    assert any("falling back to default" in note for note in meta.get("load_notes", []))
