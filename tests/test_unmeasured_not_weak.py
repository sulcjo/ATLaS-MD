from gareus.adaptive_production import AdaptiveDecisionPolicy, _edge_is_measured_weak


POL = AdaptiveDecisionPolicy()


def test_an_unmeasured_rung_edge_is_not_weak():
    assert _edge_is_measured_weak({"edge_type": "rung", "overlap": None, "mbar_overlap": None}, POL) is False


def test_a_measured_low_rung_edge_is_weak():
    assert _edge_is_measured_weak({"edge_type": "rung", "mbar_overlap": 0.05}, POL) is True


def test_a_spatial_edge_uses_its_cv_overlap():
    assert _edge_is_measured_weak({"edge_type": "nearest_2d", "overlap": 0.9}, POL) is False
    assert _edge_is_measured_weak({"edge_type": "nearest_2d", "overlap": 0.1}, POL) is True
    assert _edge_is_measured_weak({"edge_type": "nearest_2d", "overlap": None}, POL) is False


def test_low_measured_acceptance_still_counts_for_spatial_edges():
    assert _edge_is_measured_weak({"edge_type": "nearest_2d", "overlap": 0.9,
                                   "exchange_acceptance": 0.01}, POL) is True
