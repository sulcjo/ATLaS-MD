"""Replica-admission settings: parsing, validation, one manifest representation.

Spec §3, §5, §7. The MPS tests are added in Task 4.
"""

from __future__ import annotations

import json

import pytest

from gareus.cli import parse_args
from gareus.provenance import (
    _method_settings,
    admission_manifest_record,
    initialize_run_manifest,
    record_replica_admission,
)

MINIMAL = ["--seq", "AA", "--out", "unused"]
FLAT_KEYS = ("active_replicas_per_gpu", "active_replica_turn_steps", "cuda_mps_active_thread_percentage")


def test_defaults():
    args = parse_args(MINIMAL)
    assert args.active_replicas_per_gpu == "all"
    assert args.active_replica_turn_steps == 50
    assert args.cuda_mps_active_thread_percentage == "inherit"


def test_cli_values():
    args = parse_args(MINIMAL + ["--active-replicas-per-gpu", "8", "--active-replica-turn-steps", "30",
                                 "--cuda-mps-active-thread-percentage", "25"])
    assert args.active_replicas_per_gpu == 8
    assert args.active_replica_turn_steps == 30
    assert args.cuda_mps_active_thread_percentage == 25


def test_yaml_values(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("active_replicas_per_gpu: 8\nactive_replica_turn_steps: 50\n"
                   "cuda_mps_active_thread_percentage: 25\n")
    args = parse_args(MINIMAL + ["--config", str(cfg)])
    assert args.active_replicas_per_gpu == 8
    assert args.active_replica_turn_steps == 50
    assert args.cuda_mps_active_thread_percentage == 25


def test_yaml_all_and_inherit_words(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("active_replicas_per_gpu: ALL\ncuda_mps_active_thread_percentage: Inherit\n")
    args = parse_args(MINIMAL + ["--config", str(cfg)])
    assert args.active_replicas_per_gpu == "all"
    assert args.cuda_mps_active_thread_percentage == "inherit"


@pytest.mark.parametrize("bad", ["0", "-1", "2.5", "eight", ""])
def test_bad_cap_rejected(bad):
    with pytest.raises(ValueError, match="--active-replicas-per-gpu"):
        parse_args(MINIMAL + ["--active-replicas-per-gpu", bad])


def test_yaml_boolean_cap_rejected(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("active_replicas_per_gpu: true\n")
    with pytest.raises(ValueError, match="--active-replicas-per-gpu"):
        parse_args(MINIMAL + ["--config", str(cfg)])


@pytest.mark.parametrize("bad", ["0", "-5"])
def test_bad_turn_steps_rejected(bad):
    with pytest.raises(ValueError, match="--active-replica-turn-steps"):
        parse_args(MINIMAL + ["--active-replica-turn-steps", bad])


@pytest.mark.parametrize("bad", ["0", "101", "-1", "25.5", "half"])
def test_bad_percentage_rejected(bad):
    with pytest.raises(ValueError, match="--cuda-mps-active-thread-percentage"):
        parse_args(MINIMAL + ["--cuda-mps-active-thread-percentage", bad])


def test_method_settings_has_nested_record_and_no_flat_keys():
    args = parse_args(MINIMAL + ["--active-replicas-per-gpu", "8"])
    settings = _method_settings(args)
    for key in FLAT_KEYS:
        assert key not in settings
    rec = settings["replica_admission"]
    assert rec["active_replicas_per_gpu"] == 8
    assert rec["active_replica_turn_steps"] == 50
    assert rec["effective_per_queue"] is None
    assert set(rec["cuda_mps_active_thread_percentage"]) == {"requested", "inherited_env"}
    json.dumps(settings)


def test_record_on_fresh_and_resumed_complete_manifest(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "111")
    job1 = parse_args(MINIMAL)
    initialize_run_manifest(job1, tmp_path, argv=[])
    record_replica_admission(tmp_path, job1, {"shared": 4})

    # Second job: a real resume against the now-complete manifest, cap switched on,
    # queues moved to real GPUs. The old "shared" key must not survive.
    monkeypatch.setenv("SLURM_JOB_ID", "222")
    job2 = parse_args(MINIMAL + ["--active-replicas-per-gpu", "8", "--resume"])
    initialize_run_manifest(job2, tmp_path, argv=[])
    record_replica_admission(tmp_path, job2, {"0": 8, "1": 8})

    payload = json.loads((tmp_path / "run_manifest.json").read_text())
    rec = payload["method_settings"]["replica_admission"]
    assert rec["active_replicas_per_gpu"] == 8
    assert rec["effective_per_queue"] == {"0": 8, "1": 8}
    for key in FLAT_KEYS:
        assert key not in payload["method_settings"]
    hist = payload["replica_admission_history"]
    assert [h["slurm_job_id"] for h in hist] == ["111", "222"]
    assert hist[0]["active_replicas_per_gpu"] == "all"
    assert hist[0]["effective_per_queue"] == {"shared": 4}
    assert all("recorded_utc" in h for h in hist)


def test_fresh_non_resume_start_resets_history(tmp_path):
    job1 = parse_args(MINIMAL)
    initialize_run_manifest(job1, tmp_path, argv=[])
    record_replica_admission(tmp_path, job1, {"shared": 4})

    # A fresh (non-resume) start into the same directory must reset the history,
    # unlike a real --resume (covered above).
    job2 = parse_args(MINIMAL + ["--active-replicas-per-gpu", "8"])
    initialize_run_manifest(job2, tmp_path, argv=[])
    record_replica_admission(tmp_path, job2, {"0": 8, "1": 8})

    payload = json.loads((tmp_path / "run_manifest.json").read_text())
    hist = payload["replica_admission_history"]
    assert len(hist) == 1
    assert hist[0]["active_replicas_per_gpu"] == 8
    assert hist[0]["effective_per_queue"] == {"0": 8, "1": 8}


def test_record_without_existing_manifest_creates_one(tmp_path):
    args = parse_args(MINIMAL)
    record_replica_admission(tmp_path, args, {"shared": 2})
    payload = json.loads((tmp_path / "run_manifest.json").read_text())
    assert payload["method_settings"]["replica_admission"]["effective_per_queue"] == {"shared": 2}
    assert len(payload["replica_admission_history"]) == 1


def test_record_reads_inherited_env_captured_by_mps_apply():
    args = parse_args(MINIMAL)
    args._cuda_mps_inherited_env = None
    rec = admission_manifest_record(args)
    assert rec["cuda_mps_active_thread_percentage"] == {"requested": "inherit", "inherited_env": "unset"}


# ------------------------------------------------------------------ MPS share

import subprocess  # noqa: E402
import sys  # noqa: E402
import textwrap  # noqa: E402

from gareus.mps_share import ENV_VAR, apply_mps_thread_percentage  # noqa: E402


def _args(pct):
    return parse_args(MINIMAL + ["--cuda-mps-active-thread-percentage", str(pct)])


def test_inherit_records_and_leaves_env_alone():
    env = {ENV_VAR: "40"}
    args = parse_args(MINIMAL)
    apply_mps_thread_percentage(args, environ=env, modules={})
    assert env == {ENV_VAR: "40"}
    assert args._cuda_mps_inherited_env == "40"


def test_sets_env_when_unset():
    env = {"CUDA_MPS_PIPE_DIRECTORY": "/tmp/pipe"}
    args = _args(25)
    apply_mps_thread_percentage(args, environ=env, modules={})
    assert env[ENV_VAR] == "25"
    assert args._cuda_mps_inherited_env is None


@pytest.mark.parametrize("inherited", ["25", " 25 "])
def test_equal_inherited_value_is_accepted(inherited):
    env = {ENV_VAR: inherited, "CUDA_MPS_PIPE_DIRECTORY": "/tmp/pipe"}
    apply_mps_thread_percentage(_args(25), environ=env, modules={})
    assert env[ENV_VAR].strip() == "25"


def test_conflicting_inherited_value_exits_naming_both():
    env = {ENV_VAR: "40"}
    with pytest.raises(SystemExit, match=r"40.*25|25.*40"):
        apply_mps_thread_percentage(_args(25), environ=env, modules={})
    assert env[ENV_VAR] == "40"


def test_openmm_already_imported_exits():
    with pytest.raises(SystemExit, match="openmm"):
        apply_mps_thread_percentage(_args(25), environ={}, modules={"openmm": object()})


def test_warns_when_mps_not_running(capsys):
    env = {}
    apply_mps_thread_percentage(_args(25), environ=env, modules={})
    assert "CUDA_MPS_PIPE_DIRECTORY" in capsys.readouterr().err
    assert env[ENV_VAR] == "25"


def test_real_entry_chain_loads_no_openmm_and_sets_env():
    script = textwrap.dedent(
        """
        import os, sys
        os.environ.pop("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", None)
        os.environ["CUDA_MPS_PIPE_DIRECTORY"] = "/tmp/unused-pipe"
        import gareus.core
        import gareus.cli
        from gareus.mps_share import apply_mps_thread_percentage
        args = gareus.cli.parse_args(["--seq", "AA", "--out", "unused",
                                      "--cuda-mps-active-thread-percentage", "25"])
        apply_mps_thread_percentage(args)
        print("openmm" in sys.modules, "gamd" in sys.modules,
              os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"))
        """
    )
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=300, check=True)
    assert out.stdout.split()[-3:] == ["False", "False", "25"]
