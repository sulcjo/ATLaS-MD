"""Regression test: --config effective_config.yaml --resume must not throw.

_write_reproducibility_files dumps the full vars(args) namespace (227 attrs
injected by _apply_v2_compat_shims) into effective_config.yaml's args: block,
but the config loader rejects any key that isn't a real argparse dest. Every
run's auto-generated resume_command.sh therefore failed immediately. The fix
filters the dump to known dests before writing.
"""

from __future__ import annotations

from pathlib import Path


def test_effective_config_resume_round_trip(tmp_path: Path) -> None:
    from gareus.cli import parse_args
    from gareus.config import _write_reproducibility_files

    out_dir = tmp_path / "run"
    out_dir.mkdir()
    args = parse_args(["--seq", "TESTSEQ", "--out", str(out_dir)])

    _write_reproducibility_files(args, out_dir, argv=["--seq", "TESTSEQ", "--out", str(out_dir)])

    effective_config = out_dir / "config" / "effective_config.yaml"
    assert effective_config.exists()

    # This is the documented resume path (resume_command.sh runs exactly this).
    resumed = parse_args(["--config", str(effective_config), "--resume"])
    assert resumed.seq == "TESTSEQ"
