from gareus.tui import dashboard_body_budget


def test_dashboard_body_budget_scales_with_term_h():
    small = dashboard_body_budget(term_h=24, fraction=0.45, floor=10)
    typical = dashboard_body_budget(term_h=40, fraction=0.45, floor=10)
    large = dashboard_body_budget(term_h=300, fraction=0.45, floor=10)
    assert small < typical < large
    # No hard ceiling: a very tall terminal must get far more than the old
    # fixed caps (18/26/30 in the pre-rework code) ever allowed.
    assert large > 100


def test_dashboard_body_budget_respects_floor_on_tiny_terminal():
    assert dashboard_body_budget(term_h=10, fraction=0.45, floor=10) == 10
    assert dashboard_body_budget(term_h=1, fraction=0.30, floor=8) == 8


def test_dashboard_body_budget_matches_old_normal_tier_reference_point():
    # fraction is defined as old_normal_value / 40 (the stdlib terminal-size
    # fallback height); term_h=40 should reproduce the old "normal" tier
    # value exactly, by construction.
    assert dashboard_body_budget(term_h=40, fraction=18 / 40, floor=10) == 18
    assert dashboard_body_budget(term_h=40, fraction=12 / 40, floor=8) == 12
