"""Merging a stale final_registry_used_for_mbar.csv snapshot with the live
state_registry.csv.

final_registry_used_for_mbar.csv is written once, when a run first enters its
final phase. If the run is later resumed and the epoch loop adds more states
(e.g. adaptive splits) before re-entering final phase, this snapshot goes
stale: it's missing state_ids that later epochs' samples actually reference.
`load_parquet_adaptive_union` used to build its state_id -> k mapping from the
snapshot alone, so a sample referencing a missing-but-usable state_id crashed
with a bare KeyError deep in a numpy comprehension -- or, under a naive
"treat unknown as invalid" fix, would have been silently dropped instead,
biasing the recovered free energies. `_merge_missing_usable_states` fixes this
at the source: any usable state present in the live registry but absent from
the frozen snapshot is merged in.
"""
import pytest

pytest.importorskip("analyze_gareus_mbar")

from analyze_gareus_mbar import _merge_missing_usable_states, _is_usable_for_mbar


def _row(state_id, usable="True"):
    return {"state_id": str(state_id), "usable_for_mbar": usable}


def test_merges_usable_state_missing_from_primary():
    primary = [_row(0), _row(1)]
    live = [_row(0), _row(1), _row(2)]

    merged = _merge_missing_usable_states(primary, live)

    assert sorted(int(r["state_id"]) for r in merged) == [0, 1, 2]


def test_does_not_duplicate_state_already_in_primary():
    primary = [_row(0), _row(1)]
    live = [_row(0), _row(1)]

    merged = _merge_missing_usable_states(primary, live)

    assert sorted(int(r["state_id"]) for r in merged) == [0, 1]


def test_skips_unusable_state_missing_from_primary():
    primary = [_row(0)]
    live = [_row(0), _row(1, usable="False")]

    merged = _merge_missing_usable_states(primary, live)

    assert sorted(int(r["state_id"]) for r in merged) == [0]


def test_primary_row_wins_over_live_row_for_same_state_id():
    # Primary's own row data (e.g. its own burnin_steps) must not be
    # overwritten by a same-id row appearing in the live registry.
    primary = [{"state_id": "0", "usable_for_mbar": "True", "burnin_steps": "500"}]
    live = [{"state_id": "0", "usable_for_mbar": "True", "burnin_steps": "0"}]

    merged = _merge_missing_usable_states(primary, live)

    assert len(merged) == 1
    assert merged[0]["burnin_steps"] == "500"


@pytest.mark.parametrize("flag", ["True", "true", "1", "yes", "YES"])
def test_is_usable_for_mbar_accepts_truthy_variants(flag):
    assert _is_usable_for_mbar({"usable_for_mbar": flag})


@pytest.mark.parametrize("flag", ["False", "false", "0", "no", "", None])
def test_is_usable_for_mbar_rejects_falsy_variants(flag):
    assert not _is_usable_for_mbar({"usable_for_mbar": flag})
