"""Breadth tie-set, then slowest (``--cv-selection-pick breadth-tie-slowest``, default).

On chignolin's swarm the top CV2 candidates differ in breadth (fold-averaged information
gain) by less than its own noise, while the slowness order is stable. Breadth therefore only
defines the set of candidates indistinguishable from the broadest one; slowness decides in it.
"""
import pytest

from gareus.cv_selection.select_pair import PICK_RULES, SelectionConfig, _pick, breadth_tie_set


def _row(mean, sd, rho, deployable=True):
    return {"gain_nats_mean": mean, "gain_nats_sd": sd, "gain_nats": mean, "slowness_rho": rho,
            "deployable": deployable}


SCORES = {
    7: _row(0.25, 0.10, 0.986),   # tied with 8 in breadth, slowest
    8: _row(0.26, 0.10, 0.972),   # broadest
    2: _row(0.20, 0.08, 0.904),
    9: _row(-0.10, 0.03, 0.999),  # slowest of all but clearly narrower: outside the tie-set
}


def test_tie_set_membership_follows_combined_sd():
    cfg = SelectionConfig()
    tie = breadth_tie_set(SCORES, cfg)
    # 0.25 >= 0.26 - hypot(0.10, 0.10) = 0.119; 0.20 >= 0.26 - hypot(0.10, 0.08) = 0.132;
    # -0.10 < 0.26 - hypot(0.10, 0.03) = 0.156
    assert tie == {7, 8, 2}


def test_slowest_inside_the_tie_set_wins_not_the_slowest_overall():
    best, reason = _pick(SCORES, SelectionConfig(), ranking="slowness")
    assert best == 7 and "breadth tie-set [2, 7, 8]" in reason
    best_old, _ = _pick(SCORES, SelectionConfig(pick_rule="slowest"), ranking="slowness")
    assert best_old == 9                                   # the rule before: slowest overall


def test_zero_width_tie_set_is_the_broadest_alone():
    best, _ = _pick(SCORES, SelectionConfig(breadth_tie_sd=0.0), ranking="slowness")
    assert best == 8


def test_without_resampled_gains_every_candidate_is_tied():
    plain = {j: {k: v for k, v in r.items() if not k.startswith("gain_nats_")} | {"gain_nats": r["gain_nats"]}
             for j, r in SCORES.items()}
    assert breadth_tie_set(plain, SelectionConfig()) == set(plain)
    assert _pick(plain, SelectionConfig(), ranking="slowness")[0] == 9


def test_non_deployable_candidates_never_enter_and_validation():
    scores = dict(SCORES)
    scores[9] = _row(0.40, 0.01, 0.999, deployable=False)
    best, _ = _pick(scores, SelectionConfig(), ranking="slowness")
    assert best == 7
    assert PICK_RULES == ("breadth-tie-slowest", "slowest")
    with pytest.raises(ValueError):
        SelectionConfig(pick_rule="breadth")
    with pytest.raises(ValueError):
        SelectionConfig(breadth_tie_sd=-1.0)


def test_cli_and_yaml_reach_the_config(tmp_path):
    from gareus.cli import parse_args
    from gareus.swarm.analyze import _selection_config
    args = parse_args(["--seq", "GYDPETGTWG", "--out", str(tmp_path / "o")])
    cfg = _selection_config(args, 1200.0, 300.0)
    assert (cfg.pick_rule, cfg.breadth_tie_sd) == ("breadth-tie-slowest", 1.0)
    y = tmp_path / "c.yaml"
    y.write_text("cv_selection:\n  pick: slowest\n  breadth_tie_sd: 2.0\n")
    cfg = _selection_config(parse_args(["--config", str(y), "--seq", "GYDPETGTWG", "--out", str(tmp_path / "o")]),
                            1200.0, 300.0)
    assert (cfg.pick_rule, cfg.breadth_tie_sd) == ("slowest", 2.0)
