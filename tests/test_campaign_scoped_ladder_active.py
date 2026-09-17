"""``ladder_active`` must be a property of the CAMPAIGN, not of one sub-run.

chignolin_7 made the cost of getting this wrong concrete. The adaptive scheduler
is free to top up any subset of states, and it legitimately chose the 16 lambda=0
states on three occasions. In each of those sub-runs ``np.any(state_lambdas > 0)``
was False, so:

* ``k0max_by_channel`` stayed None, ``set_replica_lambda_for_window`` no-oped,
  and every replica kept the shared calibration's ``k0 = k0max`` -- i.e. FULL
  boost while the registry recorded lambda=0. Measured on the real campaign:
  those samples carry mean boost 9.44-10.60 kJ/mol against 9.57-10.16 for the
  genuine lambda=1 rung, while lambda=0 samples from mixed sub-runs are exactly
  0.000; and
* ``pep_env`` stayed None, so ``_fetch_v_pep_v_dih`` returned (nan, nan) and
  992,896 samples became unusable for MBAR.

One flag caused both. These tests pin it to the registry, which knows every
state in the campaign, rather than to whichever subset a sub-run happens to hold.
"""
import numpy as np
import pytest

from gareus.production import (
    campaign_ladder_registry_lambda,
    resolve_ladder_active,
)


def _write_registry(path, lambdas, name="state_registry.csv"):
    path.mkdir(parents=True, exist_ok=True)
    target = path / name
    lines = ["state_id,active,gamd_lambda,usable_for_mbar"]
    for i, lam in enumerate(lambdas):
        lines.append(f"{i},True,{lam},True")
    target.write_text("\n".join(lines) + "\n")
    return target


def _campaign(tmp_path, lambdas, name="state_registry.csv"):
    """adaptive_production/ holding the registry, plus a lambda=0-only top-up."""
    ap = tmp_path / "chignolin_7" / "adaptive_production"
    _write_registry(ap, lambdas, name=name)
    sub = ap / "final" / "topup_002_8742000"
    sub.mkdir(parents=True, exist_ok=True)
    return ap, sub


def test_no_registry_anywhere_is_not_a_ladder(tmp_path):
    lone = tmp_path / "run" / "adaptive_production" / "final" / "topup_002"
    lone.mkdir(parents=True)
    assert campaign_ladder_registry_lambda(lone) is None


def test_registry_two_levels_up_is_found(tmp_path):
    _ap, sub = _campaign(tmp_path, [0.0, 0.0, 0.2323, 1.0])
    found = campaign_ladder_registry_lambda(sub)
    assert found is not None
    name, lam = found
    assert name == "state_registry.csv"
    assert lam == pytest.approx(0.2323)


def test_an_all_zero_registry_is_not_a_ladder(tmp_path):
    _ap, sub = _campaign(tmp_path, [0.0, 0.0, 0.0, 0.0])
    assert campaign_ladder_registry_lambda(sub) is None


def test_final_registry_used_for_mbar_is_also_consulted(tmp_path):
    _ap, sub = _campaign(
        tmp_path, [0.0, 1.0], name="final_registry_used_for_mbar.csv")
    found = campaign_ladder_registry_lambda(sub)
    assert found is not None
    assert found[0] == "final_registry_used_for_mbar.csv"


def test_the_walk_stops_at_adaptive_production(tmp_path):
    """A registry from an unrelated campaign further up must not leak in.

    Two campaigns under one parent directory is an ordinary layout here
    (``~/gareus/chignolin/`` holds several), so an unbounded parent walk would
    let one campaign's ladder switch on another's boost recording.
    """
    _write_registry(tmp_path, [1.0])  # sibling-level registry, outside the campaign
    _ap, sub = _campaign(tmp_path, [0.0, 0.0])
    assert campaign_ladder_registry_lambda(sub) is None


def test_malformed_rows_are_skipped_not_fatal(tmp_path):
    ap = tmp_path / "c" / "adaptive_production"
    ap.mkdir(parents=True)
    (ap / "state_registry.csv").write_text(
        "state_id,gamd_lambda\n0,\n1,None\n2,not-a-number\n3,0.6429\n")
    sub = ap / "final" / "topup_002"
    sub.mkdir(parents=True)
    found = campaign_ladder_registry_lambda(sub)
    assert found is not None
    assert found[1] == pytest.approx(0.6429)


# --- resolve_ladder_active: the decision production.py actually makes ---

def test_local_boosted_state_alone_activates_the_ladder(tmp_path):
    """Unchanged behaviour for a sub-run that holds a boosted state itself."""
    bare = tmp_path / "plain_run"
    bare.mkdir()
    assert resolve_ladder_active(np.array([0.0, 1.0]), bare) is True


def test_all_zero_sub_run_of_a_ladder_campaign_activates_the_ladder(tmp_path):
    """THE REGRESSION. This returned False and produced chignolin_7's defect."""
    _ap, sub = _campaign(tmp_path, [0.0, 0.2323, 0.6429, 1.0])
    assert resolve_ladder_active(np.zeros(16), sub) is True


def test_all_zero_sub_run_of_an_unladdered_campaign_stays_inactive(tmp_path):
    """Plain umbrella/REUS and plain GaMD must not start paying for the ladder.

    Activating it there would build pep_env and make every logged sample read
    two extra Context energies for a boost that is identically zero.
    """
    _ap, sub = _campaign(tmp_path, [0.0, 0.0, 0.0])
    assert resolve_ladder_active(np.zeros(8), sub) is False


def test_resolve_accepts_a_plain_list_of_lambdas(tmp_path):
    bare = tmp_path / "plain"
    bare.mkdir()
    assert resolve_ladder_active([0.0, 0.0], bare) is False
    assert resolve_ladder_active([0.0, 0.5], bare) is True


def test_resolve_returns_a_real_bool_not_numpy_bool(tmp_path):
    """It is written into JSON manifests; np.bool_ is not JSON-serializable."""
    bare = tmp_path / "plain"
    bare.mkdir()
    assert type(resolve_ladder_active(np.array([0.0, 1.0]), bare)) is bool
