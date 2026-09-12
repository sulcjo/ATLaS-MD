"""Stage identity for the TUI header: an epoch index that was never measured
must not render as a valid one.

`run_segment` serves both the numbered adaptive epochs (``epoch_000``,
``epoch_001``, ...) and the terminal ``final`` stage.  Its epoch index was
derived by parsing the directory name with a bare ``except`` that fell back to
``0``:

    try:    _epoch_idx = int(epoch_dir.name.rsplit("_", 1)[-1])
    except Exception:    _epoch_idx = 0

``"final".rsplit("_", 1)[-1]`` is ``"final"``, so ``int()`` raised and every
frame of the terminal stage rendered ``ep 0/?`` -- indistinguishable from the
first epoch of a run whose length is unknown, while three epochs were in fact
complete.  This is the same fabricated-sentinel class that
``gareus/dashboard/ranking.py`` already calls out for deltas ("`nan`, not
`0.0`: a missing delta is unmeasured, and defaulting it to a valid 'exactly on
target' reading is the same fabricated-zero-deviation mistake"): an
inapplicable value needs a representation that cannot be mistaken for a
measured one.

The second defect is independent: ``epoch_total`` was the hardcoded string
``"?"`` even though the epoch loop one function away has ``max_epochs``, so a
legitimate top-up inside epoch 1 of 3 also rendered ``ep 1/?``.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gareus.adaptive_production import stage_phase_identity


def test_numbered_epoch_reports_its_index_and_the_real_total():
    ident = stage_phase_identity("epoch_001", max_epochs=3)
    assert ident["epoch_index"] == 1
    assert ident["epoch_total"] == 3
    assert ident.get("is_final_stage", False) is False


def test_epoch_total_is_the_configured_count_not_a_question_mark():
    """A top-up inside a real epoch must name the run's length, not '?'."""
    ident = stage_phase_identity("epoch_000", max_epochs=3)
    assert ident["epoch_total"] == 3, "max_epochs is known here and must be threaded through"


def test_final_stage_carries_no_epoch_index_at_all():
    """`final` is not an epoch.  Omission, not a plausible-looking zero."""
    ident = stage_phase_identity("final", max_epochs=3)
    assert ident["is_final_stage"] is True
    assert "epoch_index" not in ident, "an absent index must be absent, not 0"
    assert "epoch_total" not in ident


def test_unparseable_directory_name_does_not_fabricate_epoch_zero():
    """Any future stage dir that is not `epoch_NNN` must fail the same way:
    by declining to state an index, never by inventing one."""
    for name in ("final", "rescue", "epoch_extra", ""):
        ident = stage_phase_identity(name, max_epochs=3)
        assert ident.get("epoch_index") != 0 or name == "epoch_000", (
            f"{name!r} produced a fabricated epoch_index 0"
        )


def test_epoch_zero_is_still_reported_as_a_real_zero():
    """The guard must not overcorrect: epoch_000 genuinely IS index 0."""
    ident = stage_phase_identity("epoch_000", max_epochs=3)
    assert ident["epoch_index"] == 0
    assert ident.get("is_final_stage", False) is False


# --- consumer side: the merge in gareus/production.py -------------------------
# Even with the producer fixed, the merge branch re-stamped a default:
#
#     adaptive_phase_info["epoch_index"] = int(_api.get("epoch_index", 0))
#
# which writes 0 onto an `is_final=True` dict whenever `_api` carries no index.
# A key that is absent upstream must stay absent downstream.

from gareus.production import merge_adaptive_phase_info


def test_merge_does_not_stamp_epoch_zero_when_the_key_is_absent():
    """The helper owns the epoch-identity keys only; the caller keeps its own
    fields (segment_name, is_topup, ...) and they must survive the merge."""
    base = {"is_final": True, "rounds": 3}
    merged = merge_adaptive_phase_info(base, {"is_final_stage": True, "segment_name": "topup_002"})
    assert "epoch_index" not in merged, "absent upstream must stay absent downstream"
    assert merged["is_final_stage"] is True
    assert merged["is_final"] is True and merged["rounds"] == 3


def test_merge_carries_a_real_epoch_through():
    base = {"is_adaptive_epoch": True}
    merged = merge_adaptive_phase_info(base, {"epoch_index": 2, "epoch_total": 3})
    assert merged["epoch_index"] == 2
    assert merged["epoch_total"] == 3


def test_merge_preserves_a_genuine_epoch_zero():
    merged = merge_adaptive_phase_info({}, {"epoch_index": 0, "epoch_total": 3})
    assert merged["epoch_index"] == 0, "epoch_000 is a real index and must survive"


def test_merge_leaves_base_untouched():
    base = {"is_final": True}
    merge_adaptive_phase_info(base, {"epoch_index": 1})
    assert "epoch_index" not in base, "merge must not mutate its input"
