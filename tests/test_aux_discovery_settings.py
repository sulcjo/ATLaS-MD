# tests/test_aux_discovery_settings.py
import argparse, json
from pathlib import Path
import pytest

from gareus.adaptive.aux_discovery.settings import (AuxDiscoverySettings, resolve_aux_settings,
                                                    AUX_SETTINGS_FILENAME)


def test_defaults_match_c10_prereg_and_placement():
    s = AuxDiscoverySettings()
    assert (s.k_min, s.k_max, s.ari_min, s.n_boot_partition) == (2, 8, 0.5, 10)
    assert (s.knn_k, s.pca_var, s.pca_max, s.sd_drop, s.sd_floor) == (100, 0.8, 12, 1e-3, 0.05)
    assert s.l1_c_grid == (0.003, 0.01, 0.03, 0.1, 0.3)
    assert (s.n_null_z3, s.max_cv_corr) == (20, 0.7)
    assert s.quantiles == (0.05, 0.10, 0.90, 0.95) and s.width_fractions == (0.35, 0.5, 0.7)
    assert (s.gate_o_q05, s.gate_ess_frames, s.gate_eff_lineages, s.gate_top3_share) == (0.20, 50.0, 20.0, 0.5)
    assert (s.heldout_min_frames, s.heldout_o_min, s.max_workers) == (50, 0.15, 4)


def test_roundtrip_mapping():
    s = AuxDiscoverySettings(max_workers=3)
    assert AuxDiscoverySettings.from_mapping(json.loads(json.dumps(s.to_mapping()))) == s


def test_from_mapping_rejects_unknown_key():
    with pytest.raises(ValueError, match="unknown aux settings key"):
        AuxDiscoverySettings.from_mapping({**AuxDiscoverySettings().to_mapping(), "bogus": 1})


def test_resolve_freezes_first_record_and_override_replaces(tmp_path: Path):
    first, rec = resolve_aux_settings(tmp_path, AuxDiscoverySettings(max_workers=2), override=False)
    assert (tmp_path / AUX_SETTINGS_FILENAME).exists() and first.max_workers == 2
    again, _ = resolve_aux_settings(tmp_path, AuxDiscoverySettings(max_workers=4), override=False)
    assert again.max_workers == 2                      # recorded value wins
    new, _ = resolve_aux_settings(tmp_path, AuxDiscoverySettings(max_workers=4), override=True)
    assert new.max_workers == 4
    assert json.loads((tmp_path / AUX_SETTINGS_FILENAME).read_text())["settings"]["max_workers"] == 4


def test_policy_field_frozen_in_decision_settings():
    from gareus.adaptive_production import DECISION_SETTINGS_FIELDS, AdaptiveDecisionPolicy
    assert "aux_discovery" in DECISION_SETTINGS_FIELDS and "aux_reserve_slots" in DECISION_SETTINGS_FIELDS
    p = AdaptiveDecisionPolicy()
    assert p.aux_discovery is False and p.aux_reserve_slots == 4


def _parse(argv):
    from gareus.cli import parse_args
    return parse_args(argv)


def _base(tmp_path):
    return ["--window-mode", "adaptive-production", "--seq", "GYDPETGTWG", "--out", str(tmp_path), "--gamd-boost-type", "pep-gamd-lower-dual", "--exchange-mode", "gibbs-walk", "--traj-interval", "250", "--distance-output-interval", "250", "--exchange-interval", "500"]


def test_cli_accepts_valid_aux_discovery(tmp_path):
    args = _parse(_base(tmp_path) + ["--ap-aux-discovery"])
    assert args.adaptive_production_aux_discovery is True
    assert args.adaptive_production_aux_reserve_slots == 4
    assert args.adaptive_production_aux_settings_override is False
    from gareus.adaptive_production import policy_from_args
    assert policy_from_args(args).aux_discovery is True


def test_cli_refuses_aux_discovery_with_topups(tmp_path):
    with pytest.raises(SystemExit):
        _parse(_base(tmp_path) + ["--ap-aux-discovery", "--ap-topups"])


def test_cli_refuses_unaligned_intervals(tmp_path):
    with pytest.raises(SystemExit):
        _parse(_base(tmp_path) + ["--ap-aux-discovery", "--distance-output-interval", "300"])


def test_explicit_aux_model_with_adaptive_production_points_to_flag(tmp_path, capsys):
    with pytest.raises(SystemExit):
        _parse(["--window-mode", "adaptive-production", "--aux-cv-model", str(tmp_path / "m.json"),
                "--seq", "GYDPETGTWG", "--out", str(tmp_path)])
    assert "--ap-aux-discovery" in capsys.readouterr().err


def test_c11_yaml_parses():
    from pathlib import Path
    cfg = Path("/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_11.yaml")
    if not cfg.exists():
        import pytest
        pytest.skip("RUNS/ not present")
    from gareus.cli import parse_args
    try:
        args = parse_args(["--config", str(cfg), "--out", "/tmp/c11_parse_only"])
    except ValueError as exc:
        # c10's yaml needs ATLaS-MD >= 31fee1d (swarm_max_members); this branch predates it.
        if "swarm.swarm_max_members" in str(exc):
            import pytest
            pytest.skip("branch predates 31fee1d (swarm_max_members)")
        raise
    assert args.adaptive_production_aux_discovery is True
    assert args.adaptive_production_aux_reserve_slots == 4


@pytest.mark.parametrize("key,bad", [
    ("k_max", True), ("k_max", 0), ("k_max", -3), ("k_max", 2.5), ("k_max", "8"),
    ("ari_min", float("nan")), ("ari_min", float("inf")), ("ari_min", True), ("ari_min", 1.5),
    ("temperature_k", 0), ("temperature_k", -300.0), ("temperature_k", False),
    ("max_workers", 0), ("l1_c_grid", []), ("l1_c_grid", [0.1, float("nan")]), ("l1_c_grid", [0.1, True]),
    ("quantiles", [0.05, 1.5]), ("l1_c_grid", "0.1"),
])
def test_from_mapping_refuses_bad_values(key, bad):
    with pytest.raises(ValueError):
        AuxDiscoverySettings.from_mapping({key: bad})


def test_from_mapping_roundtrips_defaults():
    s = AuxDiscoverySettings()
    assert AuxDiscoverySettings.from_mapping(s.to_mapping()) == s
