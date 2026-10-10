import json

import pytest
import subprocess
import sys

from gareus.adaptive.aux_discovery.validation import VALIDATION_SCHEMA, check_validation_record


_GOOD_CHECKS = {n: {"status": "pass", "evidence": "log.txt"} for n in ("finite_timestep", "npt", "cost")}


def _write(tmp, **kw):
    rec = {"schema": VALIDATION_SCHEMA, "code_commit": "abc", "timestep_fs": 3.5, "k3_max_validated": 3.0,
           "checks": {n: {"status": "pass", "evidence": ["log.txt"]} for n in ("finite_timestep", "npt", "cost")},
           "evidence": []}
    rec.update(kw)
    p = tmp / "aux_validation.json"
    p.write_text(json.dumps(rec))
    return p


def test_missing_record(tmp_path):
    st = check_validation_record(tmp_path / "nope.json", timestep_fs=3.5)
    assert not st.ok and st.reason == "validation_missing"


def test_passing_record(tmp_path):
    st = check_validation_record(_write(tmp_path), timestep_fs=3.5)
    assert st.ok and st.k3_max == 3.0


def test_failed_check_and_timestep_mismatch(tmp_path):
    bad = _write(tmp_path, checks={**_GOOD_CHECKS, "finite_timestep": {"status": "fail", "evidence": ["x"]}})
    assert check_validation_record(bad, timestep_fs=3.5).reason == "validation_failed:finite_timestep"
    assert check_validation_record(_write(tmp_path), timestep_fs=4.0).reason == "validation_timestep_mismatch"


def test_unreadable_and_schema(tmp_path):
    p = tmp_path / "x.json"
    p.write_text("{not json")
    assert check_validation_record(p, timestep_fs=3.5).reason == "validation_unreadable"
    assert check_validation_record(_write(tmp_path, schema="other"), timestep_fs=3.5).reason == "validation_schema"


def test_cli_write(tmp_path):
    out = tmp_path / "v.json"
    subprocess.run([sys.executable, "-m", "gareus.adaptive.aux_discovery.validation", "write", "--out", str(out),
                    "--commit", "abc", "--timestep-fs", "3.5", "--k3-max", "3.0", "--finite-timestep", "pass", "--finite-timestep-evidence", "a.log",
                    "--npt", "pass", "--npt-evidence", "b.log", "--cost", "pass", "--cost-evidence", "c.log"], check=True)
    assert check_validation_record(out, timestep_fs=3.5).ok


def _reason(tmp, raw_rec):
    p = tmp / "aux_validation.json"
    p.write_text(raw_rec if isinstance(raw_rec, str) else json.dumps(raw_rec))
    return check_validation_record(p, timestep_fs=3.5).reason


@pytest.mark.parametrize("bad", ["NaN", "true", "0", "-1.0", '"3.5"'])
def test_timestep_must_be_real_finite_positive(tmp_path, bad):
    txt = _write(tmp_path).read_text()
    txt = txt.replace('"timestep_fs": 3.5', f'"timestep_fs": {bad}')
    assert _reason(tmp_path, txt) == "validation_unreadable"


@pytest.mark.parametrize("bad", ["NaN", "Infinity", "-Infinity", "0", "-2.0", "true", "false", '"3"', "null"])
def test_k3_max_must_be_real_finite_positive(tmp_path, bad):
    txt = _write(tmp_path).read_text().replace('"k3_max_validated": 3.0', f'"k3_max_validated": {bad}')
    assert _reason(tmp_path, txt) == "validation_unreadable"


def test_temperature_checked_when_present(tmp_path):
    assert check_validation_record(_write(tmp_path, temperature_k=300.0), timestep_fs=3.5).ok
    for bad in (0, -1.0, True, "300"):
        assert _reason(tmp_path, {**json.loads(_write(tmp_path).read_text()), "temperature_k": bad}) == "validation_unreadable"


def test_bare_pass_string_refused(tmp_path):
    for name in ("finite_timestep", "npt", "cost"):
        checks = {**_GOOD_CHECKS, name: "pass"}
        assert _reason(tmp_path, {**json.loads(_write(tmp_path).read_text()), "checks": checks}) \
            == f"validation_evidence_missing:{name}"


@pytest.mark.parametrize("ev", ["", "  ", [], [""], [1], None, 3])
def test_empty_or_bad_evidence_refused(tmp_path, ev):
    checks = {**_GOOD_CHECKS, "npt": {"status": "pass", "evidence": ev}}
    assert _reason(tmp_path, {**json.loads(_write(tmp_path).read_text()), "checks": checks}) \
        == "validation_evidence_missing:npt"


def test_unknown_or_missing_schema_refused(tmp_path):
    assert _reason(tmp_path, {**json.loads(_write(tmp_path).read_text()), "schema": "atlas-aux-validation-v9"}) \
        == "validation_schema"
    rec = json.loads(_write(tmp_path).read_text())
    del rec["schema"]
    assert _reason(tmp_path, rec) == "validation_schema"


def test_malformed_mapping_refused(tmp_path):
    base = json.loads(_write(tmp_path).read_text())
    assert _reason(tmp_path, {**base, "checks": ["pass"]}) == "validation_unreadable"
    assert _reason(tmp_path, {**base, "checks": {}}) == "validation_evidence_missing:finite_timestep"
    assert _reason(tmp_path, [1, 2]) == "validation_schema"


def test_valid_record_with_list_evidence_passes(tmp_path):
    checks = {n: {"status": "pass", "evidence": ["a", "b"]} for n in ("finite_timestep", "npt", "cost")}
    assert check_validation_record(_write(tmp_path, checks=checks), timestep_fs=3.5).ok


def test_driver_start_refuses_malformed_record(tmp_path):
    import types
    from gareus.adaptive_production import _require_valid_aux_validation
    args = types.SimpleNamespace(timestep_fs=3.5)
    _require_valid_aux_validation(tmp_path, args)  # missing: left to admission
    _write(tmp_path)
    _require_valid_aux_validation(tmp_path, args)
    _write(tmp_path, timestep_fs=float("nan"))
    with pytest.raises(RuntimeError, match="validation_unreadable"):
        _require_valid_aux_validation(tmp_path, args)
