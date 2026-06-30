"""Tests for the shared config_profiles module and its GAREUS/GENPEPT wiring.

A 'profile: <name>' YAML key expands to a hardcoded bundle of boilerplate
config keys (mirrors GENPEPT's existing diversity_bank_preset pattern), so a
combined YAML only needs profile + seq + a couple of output paths. None of
this changes behavior for configs that omit 'profile' (additive-only).
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_config_profiles_importable_without_heavy_deps() -> None:
    """config_profiles.py must be pure stdlib + PyYAML: no openmm/pandas/gareus."""
    code = (
        "import sys\n"
        "import config_profiles\n"
        "loaded = set(sys.modules)\n"
        "heavy = {'openmm', 'pandas', 'numpy', 'gareus', 'PeptideBuilder', 'pymbar'}\n"
        "hit = heavy & loaded\n"
        "assert not hit, f'config_profiles import pulled in: {hit}'\n"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_gareus_and_genpept_profiles_share_names() -> None:
    from config_profiles import GAREUS_PROFILES, GENPEPT_PROFILES

    assert set(GAREUS_PROFILES) == set(GENPEPT_PROFILES)
    assert "chignolin_tica" in GAREUS_PROFILES


def test_expand_profile_unknown_name_raises() -> None:
    from config_profiles import GAREUS_PROFILES, expand_profile

    with pytest.raises(KeyError, match="nope"):
        expand_profile(GAREUS_PROFILES, "nope", tool="gareus")


def test_expand_profile_returns_independent_copy() -> None:
    from config_profiles import GAREUS_PROFILES, expand_profile

    bundle = expand_profile(GAREUS_PROFILES, "chignolin_tica", tool="gareus")
    bundle["window_mode"] = "mutated"
    bundle2 = expand_profile(GAREUS_PROFILES, "chignolin_tica", tool="gareus")
    assert bundle2["window_mode"] != "mutated"


@pytest.mark.parametrize("name", ["chignolin_tica"])
def test_gareus_profile_keys_are_known_dests(name: str) -> None:
    from config_profiles import GAREUS_PROFILES
    from gareus.cli import build_gareus_parser

    known = {a.dest for a in build_gareus_parser()._actions if getattr(a, "dest", None)}
    bad = set(GAREUS_PROFILES[name]) - known
    assert not bad, f"GAREUS_PROFILES[{name!r}] has keys not in any registered dest: {bad}"


@pytest.mark.parametrize("name", ["chignolin_tica"])
def test_genpept_profile_keys_are_known_dests(name: str, tmp_path: Path) -> None:
    import GENPEPT
    from config_profiles import GENPEPT_PROFILES

    sargs = GENPEPT.parse_args(["--seq", "AAAA", "--out", str(tmp_path / "out")])
    known = set(vars(sargs).keys())
    bad = set(GENPEPT_PROFILES[name]) - known
    assert not bad, f"GENPEPT_PROFILES[{name!r}] has keys not in any registered dest: {bad}"


def test_assert_no_duplicate_keys_raises_on_top_level_dup() -> None:
    from config_profiles import assert_no_duplicate_keys

    text = "min_jobs: 1\nmin_jobs: 32\n"
    with pytest.raises(ValueError, match="min_jobs"):
        assert_no_duplicate_keys(text, "fake.yaml")


def test_assert_no_duplicate_keys_raises_on_nested_dup() -> None:
    from config_profiles import assert_no_duplicate_keys

    text = "genpept:\n  min_jobs: 1\n  min_jobs: 32\n"
    with pytest.raises(ValueError, match="min_jobs"):
        assert_no_duplicate_keys(text, "fake.yaml")


def test_assert_no_duplicate_keys_passes_clean_yaml() -> None:
    from config_profiles import assert_no_duplicate_keys

    text = "genpept:\n  min_jobs: 32\n  jobs: 64\n"
    assert_no_duplicate_keys(text, "fake.yaml")  # should not raise


def test_gareus_config_without_profile_unaffected(tmp_path: Path) -> None:
    """Additive-only: configs that omit 'profile' parse exactly as before."""
    from gareus.cli import parse_args

    cfg = tmp_path / "plain.yaml"
    cfg.write_text("seq: TESTSEQ\nout: " + str(tmp_path / "run") + "\n")
    args = parse_args(["--config", str(cfg)])
    assert args.window_mode == "adaptive"  # plain argparse default, not the bundle's value


def test_gareus_profile_expands_and_explicit_overrides_win(tmp_path: Path) -> None:
    from gareus.cli import parse_args

    cfg = tmp_path / "profiled.yaml"
    cfg.write_text(textwrap.dedent(f"""\
        profile: chignolin_tica
        seq: TESTSEQ
        out: {tmp_path / "run"}
        gamd:
          sigma0p: 9.9
        """))
    args = parse_args(["--config", str(cfg)])
    assert args.window_mode == "double-adaptive"  # inherited from the profile bundle
    assert abs(float(args.sigma0p) - 9.9) < 1e-12  # explicit key beats the profile


def test_seed_conformers_dir_derived_from_genpept_out(tmp_path: Path) -> None:
    """Fixes the cross-peptide seed-contamination bug: derive, don't copy-paste."""
    from gareus.cli import parse_args

    cfg = tmp_path / "derive.yaml"
    cfg.write_text(textwrap.dedent(f"""\
        seq: TESTSEQ
        out: {tmp_path / "run"}
        genpept:
          out: my_own_seeds
        genpept_prescan:
          genpept_prescan: true
        """))
    args = parse_args(["--config", str(cfg)])
    expected = str((tmp_path / "my_own_seeds").resolve())
    assert str(args.seed_conformers_dir) == expected  # resolved relative to the config file's dir
    assert str(args.genpept_prescan_dir) == expected  # chains via existing R9 fallback


def test_seed_conformers_dir_explicit_value_not_overridden(tmp_path: Path) -> None:
    from gareus.cli import parse_args

    cfg = tmp_path / "explicit.yaml"
    cfg.write_text(textwrap.dedent(f"""\
        seq: TESTSEQ
        out: {tmp_path / "run"}
        genpept:
          out: my_own_seeds
        starting_structures:
          seed_conformers_dir: explicitly_set_dir
        """))
    args = parse_args(["--config", str(cfg)])
    assert str(args.seed_conformers_dir) == "explicitly_set_dir"


def test_genpept_inherits_bare_top_level_seq(tmp_path: Path) -> None:
    """A minimal combined config shouldn't need seq written twice."""
    import GENPEPT

    cfg = tmp_path / "bareseq.yaml"
    cfg.write_text(textwrap.dedent(f"""\
        seq: TESTSEQ
        genpept:
          out: {tmp_path / "seeds"}
        """))
    sargs = GENPEPT.parse_args(["--config", str(cfg)])
    assert sargs.seq == "TESTSEQ"


def test_gareus_cli_only_profile_works_without_config(tmp_path: Path) -> None:
    """--profile must work as a bare CLI flag, with no --config file at all."""
    from gareus.cli import parse_args

    args = parse_args(["--seq", "TESTSEQ", "--out", str(tmp_path / "run"), "--profile", "chignolin_tica"])
    assert args.window_mode == "double-adaptive"  # inherited from the profile bundle


def test_genpept_cli_only_profile_works_without_config(tmp_path: Path) -> None:
    import GENPEPT

    sargs = GENPEPT.parse_args(["--seq", "TESTSEQ", "--out", str(tmp_path / "out"), "--profile", "chignolin_tica"])
    assert sargs.basin_hop is True  # inherited from the profile bundle


def test_cli_profile_overrides_yaml_profile_key(tmp_path: Path) -> None:
    """A bare CLI --profile takes precedence over the file's own profile: key."""
    from config_profiles import GAREUS_PROFILES
    from gareus.cli import parse_args

    GAREUS_PROFILES["_test_other"] = dict(GAREUS_PROFILES["chignolin_tica"])
    GAREUS_PROFILES["_test_other"]["window_mode"] = "manual"
    try:
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text(f"profile: _test_other\nseq: TESTSEQ\nout: {tmp_path / 'run'}\n")
        args = parse_args(["--config", str(cfg), "--profile", "chignolin_tica"])
        assert args.window_mode == "double-adaptive"  # CLI --profile wins over the YAML's profile: key
    finally:
        del GAREUS_PROFILES["_test_other"]


def test_genpept_profile_expands_and_explicit_overrides_win(tmp_path: Path) -> None:
    import GENPEPT

    cfg = tmp_path / "genpept_profiled.yaml"
    cfg.write_text(textwrap.dedent(f"""\
        profile: chignolin_tica
        seq: TESTSEQ
        genpept:
          out: {tmp_path / "seeds"}
          n: 999
        """))
    sargs = GENPEPT.parse_args(["--config", str(cfg)])
    assert sargs.basin_hop is True  # inherited from the profile bundle
    assert sargs.n_candidate_seeds == 5000  # inherited, not overridden
    assert sargs.n == 999  # explicit key beats the profile
