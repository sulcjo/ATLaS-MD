"""F01 (Task 11): per-carrier worker burn-in exclusion, burn-in phases chosen from seeding evidence.

A phase is a burn-in phase for worker w iff w was started there by a US pull (``aux_seeding_record.json``;
no record = every worker pulled). Inside it, every (phase, replica) trajectory loses its rows from its first
visit to such a worker to the end of the phase; the next phase is not filtered.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aux_union_fixture import EXCHANGE_PERIOD, ROWS_PER_WINDOW, build_campaign  # noqa: E402

from gareus.adaptive.aux_pooling import burnin_selection  # noqa: E402
from gareus.adaptive.aux_seeding_record import (SEEDING_RECORD_FILENAME, read_burnin_workers,  # noqa: E402
                                                seeding_record_payload, write_seeding_record)
from gareus.kernel_identity import AuxPoolingRefused  # noqa: E402

WORKER = 9


# ---------------------------------------------------------------- the shared selection helper (pure)

def _exchange_rows():
    """One phase, three replicas. Replica 0 starts in the pulled worker, then carries that configuration into
    ordinary state 0; replica 1 sits in ordinary state 1, swaps into the worker at step 40 and back to state 1;
    replica 2 never visits the worker. Rows are shuffled: the rule must order by step, never by row."""
    rows = []
    for step in range(10, 101, 10):
        rows.append((0, step, WORKER if step <= 30 else 0))
        rows.append((1, step, WORKER if 40 <= step <= 50 else 1))
        rows.append((2, step, 2))
    order = np.random.default_rng(3).permutation(len(rows))
    rep, step, state = (np.asarray([rows[i][j] for i in order]) for j in range(3))
    return rep, step, state


def test_carrier_rows_after_first_worker_visit_are_excluded():
    rep, step, state = _exchange_rows()
    keep, rec = burnin_selection(np.full(rep.size, "epoch_003", dtype=object), rep, step, state,
                                 {"epoch_003": {WORKER}}, evidence={"epoch_003": "aux_seeding_record"})
    # replica 0: everything (worker rows from step 10 on, then its ordinary rows) excluded
    assert not keep[rep == 0].any()
    # replica 1: rows before its first worker visit kept; its worker rows AND its later ordinary rows excluded
    r1 = rep == 1
    assert keep[r1 & (step < 40)].all() and not keep[r1 & (step >= 40)].any()
    # replica 2 never visits a pulled worker: all rows kept
    assert keep[rep == 2].all()
    assert not keep[state == WORKER].any()
    phase = rec["phases"][0]
    assert phase["phase"] == "epoch_003" and phase["burnin_workers"] == [WORKER]
    assert phase["carriers"] == [{"replica": 0, "first_excluded_step": 10}, {"replica": 1, "first_excluded_step": 40}]
    by_state = {r["state_id"]: r for r in rec["records"]}
    assert by_state[WORKER] == {"phase": "epoch_003", "state_id": WORKER, "n_replicas": 2, "rows_excluded": 5,
                                "reason": "aux_burnin_carrier"}
    assert by_state[0]["rows_excluded"] == 7 and by_state[0]["n_replicas"] == 1
    assert by_state[1]["rows_excluded"] == 5 and by_state[1]["n_replicas"] == 1
    assert 2 not in by_state


def test_phase_without_burnin_workers_keeps_everything():
    rep, step, state = _exchange_rows()
    keep, rec = burnin_selection(np.full(rep.size, "final", dtype=object), rep, step, state, {"final": set()})
    assert keep.all() and rec["phases"] == [] and rec["records"] == []


def test_only_the_burnin_phase_is_filtered_never_the_next_one():
    rep, step, state = _exchange_rows()
    phase = np.asarray(["epoch_003"] * rep.size + ["epoch_004"] * rep.size, dtype=object)
    keep, _ = burnin_selection(phase, np.r_[rep, rep], np.r_[step, step], np.r_[state, state],
                               {"epoch_003": {WORKER}, "epoch_004": set()})
    assert keep[rep.size:].all() and not keep[:rep.size].all()


def test_missing_replica_column_excludes_the_whole_phase():
    rep, step, state = _exchange_rows()
    keep, rec = burnin_selection(np.full(rep.size, "final", dtype=object), None, step, state, {"final": {WORKER}})
    assert not keep.any()
    assert {r["reason"] for r in rec["records"]} == {"aux_burnin_no_replica"}
    assert sum(r["rows_excluded"] for r in rec["records"]) == rep.size
    # without a visit to a pulled worker the missing column needs no exclusion
    keep2, _ = burnin_selection(np.full(rep.size, "final", dtype=object), None, step, np.full(rep.size, 2),
                                {"final": {WORKER}})
    assert keep2.all()


# ---------------------------------------------------------------- exact-state counterexample (spec F01)

def test_exact_two_state_counterexample_one_mh_swap_and_the_rule():
    """Configurations {A, B}; ordinary state O uniform, aux worker W with pi_W(A) = p > 1/2. The ordinary replica
    is in equilibrium, the freshly pulled worker replica starts in B. A Metropolis swap preserves the
    equilibrium joint distribution, but from this transient start it leaves O holding B with probability 1.
    The rule excludes exactly the post-swap (transient-carrying) rows, so O's kept marginal is uniform."""
    p = 0.9
    pi_o = {"A": 0.5, "B": 0.5}
    pi_w = {"A": p, "B": 1.0 - p}

    def swap(joint):
        out = {}
        for (xo, xw), pr in joint.items():
            acc = min(1.0, pi_o[xw] * pi_w[xo] / (pi_o[xo] * pi_w[xw]))
            for (a, b), q in (((xw, xo), acc), ((xo, xw), 1.0 - acc)):
                out[(a, b)] = out.get((a, b), 0.0) + pr * q
        return out

    # (i) the move is MH-valid: it leaves the equilibrium product distribution invariant
    eq = {(a, b): pi_o[a] * pi_w[b] for a in "AB" for b in "AB"}
    assert all(abs(v - eq[k]) < 1e-12 for k, v in swap(eq).items())
    # (ii) from the transient start (worker in B) one swap leaves O off-equilibrium
    after = swap({("A", "B"): 0.5, ("B", "B"): 0.5})
    o_after_b = sum(v for (a, _), v in after.items() if a == "B")
    assert o_after_b == pytest.approx(1.0)
    # (iii) rows: each "world" (initial O configuration) is one phase-local trajectory pair; replica 0 starts in
    # O, replica 1 is the pulled worker. Pooled unfiltered, O's marginal of B is (0.5 + 1) / 2 = 0.75.
    pooled_b = kept_b = kept_n = 0.0
    for xo0, weight in (("A", 0.5), ("B", 0.5)):
        # step 10: before the swap; step 20: after it (replicas swapped when accepted, which here is always)
        rows = [(0, 10, "O", xo0), (1, 10, "W", "B"), (1, 20, "O", "B"), (0, 20, "W", xo0)]
        rep = np.asarray([r[0] for r in rows]); step = np.asarray([r[1] for r in rows])
        sid = np.asarray([0 if r[2] == "O" else WORKER for r in rows])
        keep, _ = burnin_selection(np.full(4, "final", dtype=object), rep, step, sid, {"final": {WORKER}})
        for (_, _, s, x), k in zip(rows, keep):
            if s == "O":
                pooled_b += weight * 0.5 * (x == "B")
                if k:
                    kept_b += weight * (x == "B"); kept_n += weight
        assert keep.tolist() == [True, False, False, False]
    assert pooled_b == pytest.approx(0.75)
    assert kept_b / kept_n == pytest.approx(pi_o["B"])


# ---------------------------------------------------------------- seeding evidence

def test_seeding_record_reader(tmp_path):
    ph = tmp_path / "final_extension_001"
    ph.mkdir()
    # old phase without a record: every worker counts as pulled (conservative)
    assert read_burnin_workers(ph, {5, 7}) == ({5, 7}, "no_seeding_record")
    write_seeding_record(ph, seeding_record_payload(
        worker_windows=[1, 3], state_of_window={1: 5, 3: 7}, pulled_windows={3},
        continued_source={1: "state_export"}, branch="continue_states"))
    assert read_burnin_workers(ph, {5, 7}) == ({7}, "aux_seeding_record")
    # a worker the record does not list is conservative too
    assert read_burnin_workers(ph, {5, 7, 8}) == ({7, 8}, "aux_seeding_record")


def test_seed_mismatch_fallback_records_every_worker_pulled(tmp_path):
    ph = tmp_path / "final_extension_002"
    ph.mkdir()
    payload = seeding_record_payload(worker_windows=[1, 3], state_of_window={1: 5, 3: 7}, pulled_windows={0, 1, 2, 3},
                                     continued_source={}, branch="pull", fallback="seed_mismatch")
    write_seeding_record(ph, payload)
    rec = json.loads((ph / SEEDING_RECORD_FILENAME).read_text())
    assert {w["seeding"] for w in rec["workers"]} == {"pulled"} and rec["fallback"] == "seed_mismatch"
    assert read_burnin_workers(ph, {5, 7})[0] == {5, 7}


def test_rewrite_never_downgrades_a_pulled_worker(tmp_path):
    ph = tmp_path / "epoch_004"
    ph.mkdir()
    kw = dict(worker_windows=[1], state_of_window={1: 5}, continued_source={1: "state_export"})
    write_seeding_record(ph, seeding_record_payload(pulled_windows={1}, branch="pull", **kw))
    write_seeding_record(ph, seeding_record_payload(pulled_windows=set(), branch="continue_states", **kw))
    assert read_burnin_workers(ph, {5})[0] == {5}


@pytest.mark.parametrize("payload", ["{not json", json.dumps({"schema": "other"}),
                                     json.dumps({"schema": "atlas-aux-seeding-record-v1",
                                                 "workers": [{"state_id": 5, "seeding": "maybe"}]})])
def test_malformed_seeding_record_refuses(tmp_path, payload):
    (tmp_path / SEEDING_RECORD_FILENAME).write_text(payload)
    with pytest.raises(AuxPoolingRefused):
        read_burnin_workers(tmp_path, {5})


# ---------------------------------------------------------------- both union paths

def _driver(camp):
    from gareus.adaptive_production import build_union_state_mbar_inputs
    return build_union_state_mbar_inputs(camp.ad, camp.registry, output_prefix="drv")


def _loader(camp, **kw):
    from gareus.mbar_analysis.loaders_union_parquet import load_parquet_adaptive_union
    return load_parquet_adaptive_union(camp.ad, n_workers=1, **kw)


def _loader_keys(d):
    labels = [Path(p).name for p in d.meta["_epoch_source_run_dirs"]]
    src = np.asarray(d.meta["_epoch_source"])
    return {(labels[int(s)], int(r), int(t)) for s, r, t in zip(src, np.asarray(d.replica), np.asarray(d.step))}


def _expected_kept(camp, carriers_by_phase):
    keys = set(camp.truth)
    out = set()
    for (label, r, t) in keys:
        first = carriers_by_phase.get(label, {}).get(r)
        if first is None or t < first:
            out.add((label, r, t))
    return out


def test_final_as_first_worker_phase_is_filtered_and_burnin_phase_epoch_is_ignored(tmp_path):
    # the admission record's burnin_phase_epoch says epoch 1, the evidence says the worker was pulled in final
    camp = build_campaign(tmp_path, seeding_records={"epoch_001": "continued", "final": "pulled"})
    d = _loader(camp)
    win = np.asarray(d.window)
    src = np.asarray(d.meta["_epoch_source"])
    labels = [Path(p).name for p in d.meta["_epoch_source_run_dirs"]]
    worker_phases = {labels[int(s)] for s in src[win == 2]}
    assert worker_phases == {"epoch_001"}                   # final's worker rows excluded, epoch_001's kept
    assert d.meta["aux_burnin_dropped"] == {"2": ROWS_PER_WINDOW}
    meta = _driver(camp)
    with open(meta["samples_csv"], newline="") as fh:
        w_rows = [r for r in csv.DictReader(fh) if int(r["sampled_state_id"]) == camp.worker_id]
    assert w_rows and {r["source"] for r in w_rows} == {"epoch_001"}


def test_continued_phases_exclude_nothing(tmp_path):
    camp = build_campaign(tmp_path, exchange=True, seeding_records={"epoch_001": "continued", "final": "continued"})
    d = _loader(camp)
    assert _loader_keys(d) == set(camp.truth)
    assert d.meta["aux_burnin_exclusions"]["records"] == []
    assert _driver(camp)["aux_burnin_exclusions"]["records"] == []


def test_old_phases_without_record_are_conservative(tmp_path):
    camp = build_campaign(tmp_path, seeding_records={})
    d = _loader(camp)
    assert (np.asarray(d.window) == 2).sum() == 0           # both post-admission phases count as pulled
    assert {p["evidence"] for p in d.meta["aux_burnin_exclusions"]["phases"]} == {"no_seeding_record"}


def test_driver_and_analyzer_select_identical_observation_keys(tmp_path):
    camp = build_campaign(tmp_path, exchange=True)           # epoch_001 pulled, final continued
    d = _loader(camp)
    # rotating schedule over 3 windows, worker = window 2: replica 2 starts there, replica 1 enters at
    # t = EXCHANGE_PERIOD, replica 0 at t = 2 EXCHANGE_PERIOD (step = step0 + 10 (t + 1))
    first = {2: 100_010, 1: 100_000 + 10 * (EXCHANGE_PERIOD + 1), 0: 100_000 + 10 * (2 * EXCHANGE_PERIOD + 1)}
    expected = _expected_kept(camp, {"epoch_001": first})
    assert _loader_keys(d) == expected
    rec_a = d.meta["aux_burnin_exclusions"]
    assert rec_a["phases"][0]["carriers"] == [{"replica": r, "first_excluded_step": first[r]} for r in (0, 1, 2)]
    # ordinary states lose their carriers' rows too
    assert set(d.meta["aux_burnin_dropped"]) == {"0", "1", "2"}
    meta = _driver(camp)
    assert meta["aux_burnin_exclusions"] == rec_a                # same rule, same record
    # the driver subsamples after the selection: every row it kept is a kept key (its CSV has no replica
    # column: (phase, step) is compared), and its burn-in drops count exactly the excluded keys
    with open(meta["samples_csv"], newline="") as fh:
        drv = {(r["source"], int(r["step"])) for r in csv.DictReader(fh)}
    assert drv and drv <= {(lab, t) for lab, _, t in expected}
    n_excl = len(set(camp.truth) - expected)
    assert sum(v.get("burnin_dropped", 0) for v in meta["subsample_counts_per_state"].values()) == n_excl


def test_flag_off_union_paths_carry_no_burnin_record(tmp_path):
    """No aux admission: neither union path reads a seeding record or writes an exclusion record."""
    from aux_union_fixture import _write_phase
    from gareus.adaptive_production import WindowStateRegistry, build_union_state_mbar_inputs
    ad = tmp_path / "adaptive"
    ad.mkdir()
    reg = WindowStateRegistry()
    reg.add_state(0.2, 10.0)
    reg.add_state(0.5, 10.0)
    reg.save(ad)
    _write_phase(ad / "epoch_000", 2, False, 0, np.random.default_rng(1), {}, "epoch_000")
    (ad / "epoch_000" / SEEDING_RECORD_FILENAME).write_text("{not json")     # never read without admission
    meta = build_union_state_mbar_inputs(ad, reg)
    assert "aux_burnin_exclusions" not in meta
    assert all("burnin_dropped" not in v for v in meta["subsample_counts_per_state"].values())
    d = _loader(type("C", (), {"ad": ad})())
    assert "aux_burnin_exclusions" not in d.meta and "aux_burnin_dropped" not in d.meta


# ---------------------------------------------------------------- production's record writer

def _phase_with_map(tmp_path, name):
    ph = tmp_path / name
    ph.mkdir()
    (ph / "epoch_window_map.csv").write_text("epoch_window,state_id\n0,10\n1,11\n2,12\n3,13\n")
    return ph


@pytest.mark.parametrize("branch,pulled,cont,pdb,fallback,expected", [
    # --ap-continue-states: the new worker (window 3) is pulled, the old one (window 2) continues its State
    ("continue_states", {3}, {0, 1, 2}, set(), None, {12: ("continued", "state_export"), 13: ("pulled", "us_pull")}),
    # frozen-final extension continued from final PDBs: not a burn-in phase
    ("extension", set(), set(), {0, 1, 2, 3}, None, {12: ("continued", "final_pdb"), 13: ("continued", "final_pdb")}),
    # top-ups continue every chain from the parent segment's export
    ("topup", set(), set(), set(), None, {12: ("continued", "topup_state_export"),
                                          13: ("continued", "topup_state_export")}),
    # SeedMismatchError fallback: every window pulled
    ("pull", {0, 1, 2, 3}, set(), set(), "seed_mismatch", {12: ("pulled", "us_pull"), 13: ("pulled", "us_pull")}),
])
def test_production_writes_the_seeding_record(tmp_path, branch, pulled, cont, pdb, fallback, expected):
    from types import SimpleNamespace
    from gareus.production import _write_aux_seeding_record
    ph = _phase_with_map(tmp_path, "final_extension_001")
    table = SimpleNamespace(n=4, k_kcal=(0.0, 0.0, 2.0, 3.0))
    _write_aux_seeding_record(ph, table, branch, pulled, cont, pdb, fallback)
    rec = json.loads((ph / SEEDING_RECORD_FILENAME).read_text())
    assert rec["branch"] == branch and rec["fallback"] == fallback
    assert {e["state_id"]: (e["seeding"], e["source"]) for e in rec["workers"]} == expected
    pulled_states = {sid for sid, (seeding, _) in expected.items() if seeding == "pulled"}
    assert read_burnin_workers(ph, {12, 13}) == (pulled_states, "aux_seeding_record")


def test_low_memory_stride_never_leaks_a_carrier_whose_first_visit_was_strided_out(tmp_path):
    camp = build_campaign(tmp_path, exchange=True)
    full = _loader_keys(_loader(camp))
    d = _loader(camp, low_memory=True, analysis_stride=2)
    strided = _loader_keys(d)
    assert strided and strided <= full
    # every carrier loses every row at or after its first visit, whether or not the stride kept that visit
    first = {c["replica"]: c["first_excluded_step"] for c in d.meta["aux_burnin_exclusions"]["phases"][0]["carriers"]}
    assert not any(lab == "epoch_001" and t >= first[r] for lab, r, t in strided)
    strided_rows = {(r, t) for lab, r, t in strided if lab == "epoch_001"}
    assert any((r, f) not in strided_rows for r, f in first.items())      # some first visit really was strided out
    # the counts describe the rows this strided pool lost
    assert d.meta["aux_burnin_exclusions"]["analysis_stride_applied_before_count"] is True
    n_loaded_epoch1 = sum(1 for (lab, r, t) in camp.truth if lab == "epoch_001")
    assert sum(r["rows_excluded"] for r in d.meta["aux_burnin_exclusions"]["records"]) < n_loaded_epoch1


# ---------------------------------------------------------------- resume supersession (controller ruling)

from gareus.adaptive.aux_pooling import resume_supersession  # noqa: E402

_ORDER = {"epoch_001": {"seg_001": 0, "seg_002": 1, "seg_003": 2}}


def _segments(*specs):
    """specs: (segment, replica, steps) -> phase/replica/step/segment columns."""
    rep, step, seg = [], [], []
    for s_, r, steps in specs:
        for t in steps:
            rep.append(r); step.append(t); seg.append(s_)
    n = len(step)
    return np.full(n, "epoch_001", dtype=object), np.asarray(rep), np.asarray(step), np.asarray(seg, dtype=object)


def test_restart_supersedes_the_whole_earlier_segment():
    # seg_002 restarted from step 100 (no checkpoint): seg_001 is excluded entirely, also its rows past seg_002's end
    ph, rep, step, seg = _segments(("seg_001", 0, range(100, 1100, 100)), ("seg_001", 1, range(100, 1100, 100)),
                                   ("seg_002", 0, range(100, 600, 100)), ("seg_002", 1, range(100, 600, 100)))
    keep, rec = resume_supersession(ph, rep, step, seg, _ORDER)
    assert not keep[seg == "seg_001"].any() and keep[seg == "seg_002"].all()
    assert rec == [{"phase": "epoch_001", "segment": "seg_001", "superseded_by": "seg_002", "rows_excluded": 20,
                    "reason": "superseded_by_resume"}]


def test_continuation_after_the_earlier_last_step_is_kept():
    ph, rep, step, seg = _segments(("seg_001", 0, range(100, 600, 100)), ("seg_002", 0, range(600, 1100, 100)),
                                   ("seg_003", 0, range(1100, 1300, 100)))
    keep, rec = resume_supersession(ph, rep, step, seg, _ORDER)
    assert keep.all() and rec == []


def test_restart_inside_the_earlier_segment_is_ambiguous_and_refuses():
    ph, rep, step, seg = _segments(("seg_001", 0, range(100, 1100, 100)), ("seg_002", 0, range(500, 900, 100)))
    with pytest.raises(AuxPoolingRefused, match="neither a restart nor a continuation"):
        resume_supersession(ph, rep, step, seg, _ORDER)


def test_overlapping_segment_without_durable_order_refuses():
    ph, rep, step, seg = _segments(("seg_001", 0, range(100, 600, 100)), ("seg_009", 0, range(100, 600, 100)))
    with pytest.raises(AuxPoolingRefused, match="no durable order"):
        resume_supersession(ph, rep, step, seg, _ORDER)


def test_duplicates_within_one_segment_still_refuse():
    from gareus.adaptive.aux_pooling import union_aux_selection
    from gareus.correctness.observation_keys import ObservationKeyRefusal
    ph, rep, step, seg = _segments(("seg_001", 0, [100, 200, 200]))
    with pytest.raises(ObservationKeyRefusal, match="duplicate"):
        union_aux_selection(ph, rep, step, seg, np.zeros(3, dtype=np.int64), {}, segment_orders=_ORDER)


def test_restarted_phase_both_paths_select_identical_keys(tmp_path):
    camp = build_campaign(tmp_path, exchange=True, restart_epoch_001=True)
    d = _loader(camp)
    meta = _driver(camp)
    rec = d.meta["aux_burnin_exclusions"]
    assert rec["superseded"] == [{"phase": "epoch_001", "segment": "seg_001", "superseded_by": "seg_002",
                                  "rows_excluded": 3 * ROWS_PER_WINDOW, "reason": "superseded_by_resume"}]
    assert meta["aux_burnin_exclusions"] == rec
    first = {2: 100_010, 1: 100_000 + 10 * (EXCHANGE_PERIOD + 1), 0: 100_000 + 10 * (2 * EXCHANGE_PERIOD + 1)}
    expected = _expected_kept(camp, {"epoch_001": first})
    assert _loader_keys(d) == expected                    # each key once: seg_001's copy superseded
    # the kept epoch_001 rows are seg_002's (its z is the truth the restart wrote)
    z = dict(zip(_loader_keys_list(d), np.asarray(d.aux_z).tolist()))
    assert all(z[k] == pytest.approx(camp.truth[k]) for k in expected if k[0] == "epoch_001")
    with open(meta["samples_csv"], newline="") as fh:
        drv = {(r["source"], int(r["step"])) for r in csv.DictReader(fh)}
    assert drv and drv <= {(lab, t) for lab, _, t in expected}


def _loader_keys_list(d):
    labels = [Path(p).name for p in d.meta["_epoch_source_run_dirs"]]
    src = np.asarray(d.meta["_epoch_source"])
    return [(labels[int(s)], int(r), int(t)) for s, r, t in zip(src, np.asarray(d.replica), np.asarray(d.step))]
