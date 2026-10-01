"""The US starting-structure gate never fails a window on its distance from a k = 0 placeholder centre."""
from gareus.seeding import classify_primary_start_delta


def test_unrestrained_primary_axis_is_never_judged_on_its_placeholder_distance():
    assert classify_primary_start_delta(0.38, 0.0, 0.15, 0.30) is None
    assert classify_primary_start_delta(0.38, -1.0, 0.15, 0.30) is None
    assert classify_primary_start_delta(0.38, float("nan"), 0.15, 0.30) is None


def test_restrained_primary_axis_keeps_the_old_thresholds():
    assert classify_primary_start_delta(0.38, 349.8, 0.15, 0.30) == "bad"
    assert classify_primary_start_delta(0.20, 349.8, 0.15, 0.30) == "warn"
    assert classify_primary_start_delta(0.10, 349.8, 0.15, 0.30) is None
    assert classify_primary_start_delta(float("nan"), 349.8, 0.15, 0.30) is None


def _conf(cv1, cv2):
    return {"primary_cv_value": cv1, "secondary_cv_value": cv2, "pdb_path": f"{cv1}_{cv2}.pdb"}


def _score(conf, **kw):
    from types import SimpleNamespace
    from gareus.seeding import _score_seed_conformer
    args = SimpleNamespace(primary_cv="contacts", cv1="contacts", contact_normalize=True)
    base = dict(window_index=4, seed_selection_mode="active-cv", primary_seed_scale=0.05, target_primary=0.577,
                target_secondary=-0.25, secondary_available=True, seed_secondary_weight=1.0,
                secondary_seed_scale=0.1, args=args)
    base.update(kw)
    return _score_seed_conformer(conf, **base)


def test_cv2_free_window_picks_the_seed_nearest_on_cv1_not_the_cv2_placeholder():
    near_cv1, near_placeholder = _conf(0.55, 1.4), _conf(0.20, -0.25)
    # legacy scoring (both axes counted) prefers the placeholder match -- the c10 failure
    assert _score(near_placeholder)[0] < _score(near_cv1)[0]
    s_cv1, c = _score(near_cv1, secondary_restrained=False)
    s_ph, _ = _score(near_placeholder, secondary_restrained=False)
    assert s_cv1 < s_ph and c["secondary_score"] == 0.0 and c["secondary_restrained"] is False


def test_cv1_free_window_ignores_the_cv1_placeholder():
    s, c = _score(_conf(0.95, -0.25), primary_restrained=False)
    assert c["primary_score"] == 0.0 and s == c["secondary_score"]


def test_window_axis_restrained_defaults_to_restrained_when_unknown():
    from gareus.seeding import _window_axis_restrained
    assert _window_axis_restrained([0.0, 5.0], 1) and not _window_axis_restrained([0.0, 5.0], 0)
    assert _window_axis_restrained(None, 0) and _window_axis_restrained([float("nan")], 0)
