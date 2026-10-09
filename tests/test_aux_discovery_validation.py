import json
import subprocess
import sys

from gareus.adaptive.aux_discovery.validation import VALIDATION_SCHEMA, check_validation_record


def _write(tmp, **kw):
    rec = {"schema": VALIDATION_SCHEMA, "code_commit": "abc", "timestep_fs": 3.5, "k3_max_validated": 3.0,
           "checks": {"finite_timestep": "pass", "npt": "pass", "cost": "pass"}, "evidence": []}
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
    bad = _write(tmp_path, checks={"finite_timestep": "fail", "npt": "pass", "cost": "pass"})
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
                    "--commit", "abc", "--timestep-fs", "3.5", "--k3-max", "3.0", "--finite-timestep", "pass",
                    "--npt", "pass", "--cost", "pass"], check=True)
    assert check_validation_record(out, timestep_fs=3.5).ok
