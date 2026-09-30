"""The 3.1 edge metric and its neighbours are reachable from a YAML config (generic dest mapping)."""
from gareus.cli import parse_args


def test_yaml_keys_reach_the_adaptive_edge_metric_settings(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("ap_edge_metric: pairwise-mbar\n"
                   "ap_min_edge_neff: 150\n"
                   "layout_neighbour_rule: restraint-width\n")
    args = parse_args(["--config", str(cfg), "--out", str(tmp_path / "o"), "--seq", "GYDPETGTWG"])
    assert args.ap_edge_metric == "pairwise-mbar"
    assert args.adaptive_production_edge_metric == "pairwise-mbar"
    assert args.ap_min_edge_neff == 150
    assert args.layout_neighbour_rule == "restraint-width"


def test_edge_metric_defaults_to_marginal_without_the_key(tmp_path):
    args = parse_args(["--out", str(tmp_path / "o"), "--seq", "GYDPETGTWG"])
    assert args.adaptive_production_edge_metric == "marginal"
