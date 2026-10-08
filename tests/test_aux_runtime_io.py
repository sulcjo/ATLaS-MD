# tests/test_aux_runtime_io.py
import ast
import inspect
import math
import types
from types import SimpleNamespace

import numpy as np
import pytest

from gareus.auxiliary_cv.runtime_io import (ExchangeEventCounter, aux_event_fields, check_exchange_boundary_alignment,
                                            check_runtime_parity, event_from_gibbs)
from gareus.correctness._io import IntegrityError


def test_alignment_uses_resolved_intervals_and_calib_offset():
    assert check_exchange_boundary_alignment(exchange_interval=3000, distance_interval=300, traj_interval=3000,
                                             calib_steps=6000) == []
    with pytest.raises(IntegrityError, match="traj_interval"):
        check_exchange_boundary_alignment(exchange_interval=3000, distance_interval=300, traj_interval=2000, calib_steps=0)
    with pytest.raises(IntegrityError, match="calib_steps"):
        check_exchange_boundary_alignment(exchange_interval=3000, distance_interval=300, traj_interval=3000, calib_steps=1000)
    with pytest.raises(IntegrityError, match="distance_interval"):
        check_exchange_boundary_alignment(exchange_interval=3000, distance_interval=700, traj_interval=3000, calib_steps=0)
    notes = check_exchange_boundary_alignment(exchange_interval=3000, distance_interval=300, traj_interval=0, calib_steps=0)
    assert notes and "Stage D" in notes[0]


def test_counter_orders_within_step():
    c = ExchangeEventCounter()
    assert [c.next(10), c.next(10), c.next(20), c.next(20), c.next(20)] == [0, 1, 0, 1, 2]


def test_event_from_gibbs_stay_and_no_candidates_and_zero_q():
    stay = SimpleNamespace(current_window=2, proposed_window=2, delta_kj=0.0, q_forward=0.6, q_reverse=0.0,
                           pacc=0.0, stayed=True, no_candidates=False)
    ev = event_from_gibbs(stay, step=100, seq=4, selected_replica=7, accepted=False, energy_version="v",
                          assignments_after=[0, 1])
    assert ev["kind"] == "stay" and ev["replica_i"] == ev["replica_j"] == 7 and ev["log_q_reverse"] == float("-inf")
    none = SimpleNamespace(current_window=2, proposed_window=2, delta_kj=0.0, q_forward=0.0, q_reverse=0.0,
                           pacc=0.0, stayed=False, no_candidates=True)
    assert event_from_gibbs(none, step=1, seq=0, selected_replica=7, accepted=False, energy_version="v",
                            assignments_after=[0])["kind"] == "no_candidates"
    move = SimpleNamespace(q_forward=0.25, q_reverse=0.0, pacc=0.4)
    f = aux_event_fields(move)
    assert f["log_q_forward"] == pytest.approx(math.log(0.25)) and f["log_q_reverse"] == float("-inf")
    assert f["p_accept"] == 0.4


def test_runtime_parity():
    from gareus.auxiliary_cv.runtime import AuxObservationError
    check_runtime_parity(np.array([1.0]), np.array([1.0 + 1e-9]), beta=0.4, k_max_kcal=2.0, centers=[0.5], tolerance=1e-6)
    with pytest.raises(AuxObservationError, match="parity"):
        check_runtime_parity(np.array([1.0]), np.array([1.01]), beta=0.4, k_max_kcal=2.0, centers=[0.5], tolerance=1e-6)
    with pytest.raises(AuxObservationError, match="finite"):
        check_runtime_parity(np.array([1.0]), np.array([np.nan]), beta=0.4, k_max_kcal=2.0, centers=[0.5], tolerance=1e-6)


def test_fixed_box_check():
    from gareus.auxiliary_cv.runtime_io import check_fixed_box
    box = [[3.0, 0, 0], [0, 3.0, 0], [0, 0, 3.0]]
    check_fixed_box([box, [row[:] for row in box]], box, label="t")
    with pytest.raises(IntegrityError, match="replica 1"):
        check_fixed_box([box, [[3.0001, 0, 0], [0, 3.0, 0], [0, 0, 3.0]]], box, label="t")


def test_runtime_parity_never_raises_without_an_active_state():
    """M5: a sham-only population records (fast path finite theta vs positions NaN), never raises."""
    check_runtime_parity(np.array([0.3]), np.array([np.nan]), beta=0.4, k_max_kcal=0.0, centers=[], tolerance=1e-6)
    check_runtime_parity(np.array([0.3]), np.array([5.0]), beta=0.4, k_max_kcal=0.0, centers=[], tolerance=1e-6)


def _record_setup(k_kcal):
    from test_aux_cv_observation import _setup, _with_table
    from gareus.auxiliary_cv.sample_schema import build_sample_schema
    ctx, rt, force, d = _setup()
    runtime = _with_table(rt, k_kcal)
    atoms = sorted({a for q in d["quads"] for a in q})
    schema = build_sample_schema(d["topology"], atoms, [rt.table.model])
    return ctx, runtime, force, d, schema, {rt.table.model.model_sha256: rt.table.model}


@pytest.mark.parametrize("fast", [True, False])
def test_record_observer_one_read_gives_runtime_z_torsions_and_positions_z(fast):
    from openmm import unit
    from gareus.auxiliary_cv.runtime_io import make_aux_record_observer
    from gareus.auxiliary_cv.sample_schema import observe_carrier
    ctx, runtime, force, d, schema, models = _record_setup(2.0)
    observe = make_aux_record_observer(runtime, use_fast_path=fast, fast_forces=[force], unit=unit,
                                       schema=schema, models=models)
    z_rt, tors, z_pos = observe(0, types.SimpleNamespace(context=ctx))
    ref = observe_carrier(d["positions_nm"], schema, models)
    np.testing.assert_allclose(tors, ref.torsions, atol=1e-12)
    assert z_pos == pytest.approx(float(ref.z[0]), abs=1e-12) and z_rt == pytest.approx(z_pos, abs=1e-9)


@pytest.mark.parametrize("fast", [True, False])
def test_record_observer_checks_geometry_only_with_an_active_state(fast):
    from openmm import unit
    from test_aux_cv_observation import _degenerate
    from gareus.auxiliary_cv.runtime import AuxObservationError
    from gareus.auxiliary_cv.runtime_io import make_aux_record_observer
    for k, raises in ((2.0, True), (0.0, False)):
        ctx, runtime, force, d, schema, models = _record_setup(k)
        _degenerate(ctx, runtime, d)
        observe = make_aux_record_observer(runtime, use_fast_path=fast, fast_forces=[force], unit=unit,
                                           schema=schema, models=models)
        if raises:
            with pytest.raises(AuxObservationError, match="degenerate"):
                observe(0, types.SimpleNamespace(context=ctx))
        else:
            observe(0, types.SimpleNamespace(context=ctx))


@pytest.mark.parametrize("fast", [True, False])
def test_record_observer_reads_positions_once_per_carrier(fast):
    """Carry-over 10: ONE getState(getPositions=True) per carrier per sample."""
    from openmm import unit
    from gareus.auxiliary_cv.runtime_io import make_aux_record_observer
    ctx, runtime, force, d, schema, models = _record_setup(2.0)
    calls = []

    class _Ctx:
        def getState(self, **kw):
            calls.append(kw)
            return ctx.getState(**kw)

    class _Force:      # the fast path reads the force's own value through the real Context
        def getCollectiveVariableValues(self, _c):
            return force.getCollectiveVariableValues(ctx)

    observe = make_aux_record_observer(runtime, use_fast_path=fast, fast_forces=[_Force()], unit=unit,
                                       schema=schema, models=models)
    z_rt, _tors, _zpos = observe(0, types.SimpleNamespace(context=_Ctx()))
    assert sum(1 for kw in calls if kw.get("getPositions")) == 1
    assert math.isfinite(z_rt)


def _func(tree, name):
    return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)


def _named_calls(node, name):
    return [n for n in ast.walk(node) if isinstance(n, ast.Call)
            and getattr(n.func, "attr", getattr(n.func, "id", None)) == name]


def test_every_gibbs_decision_and_every_swap_early_return_writes_an_event():
    import gareus.production as production
    tree = ast.parse(inspect.getsource(production.run_gareus))
    swap = _func(tree, "_attempt_window_swap")
    # D1: count CALL sites (the def line is not a call): one before each of the three early returns
    assert len(_named_calls(swap, "_aux_write_skip")) == 3
    early = [n for n in ast.walk(swap) if isinstance(n, ast.Return) and isinstance(n.value, ast.Name)
             and n.value.id == "attempt"]
    assert len(early) == 3
    src = ast.unparse(tree)
    gibbs_block = src[src.index("if mode == 'gibbs-walk'"):]
    gibbs_block = gibbs_block[: gibbs_block.index("raise ValueError")]
    assert gibbs_block.count("event_from_gibbs(") >= 2 and "aux_event=" in gibbs_block


def test_checkpoint_saves_and_load_carry_the_aux_hooks():
    import gareus.production as production
    tree = ast.parse(inspect.getsource(production.run_gareus))
    saves = _named_calls(tree, "save_production_checkpoint")
    assert len(saves) == 2 and all(any(k.arg == "aux_block" for k in c.keywords) for c in saves)
    loads = _named_calls(tree, "load_production_checkpoint")
    assert len(loads) == 1 and any(k.arg == "aux_pre_apply" for k in loads[0].keywords)
    assert len(_named_calls(tree, "verify_aux_ledger")) == 1


# ── Carry-overs (task-13-carryovers.md) ────────────────────────────────────────────────────────────

def _parents(tree):
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def _under_aux_condition(node, parents):
    while node in parents:
        node = parents[node]
        if isinstance(node, (ast.If, ast.IfExp)) and "aux" in ast.unparse(node.test):
            return True
    return False


def test_test_failure_hook_is_aux_gated():
    """Carry-over 14: the injected-failure environment variable is read only on an auxiliary run."""
    import gareus.production as production
    tree = ast.parse(inspect.getsource(production.run_gareus))
    parents = _parents(tree)
    hits = [n for n in ast.walk(tree) if isinstance(n, ast.Constant) and n.value == "GAREUS_TEST_FAIL_AT_PROD_STEP"]
    assert hits and all(_under_aux_condition(n, parents) for n in hits)
    raises = [n for n in ast.walk(tree) if isinstance(n, ast.Raise) and "GAREUS_TEST_FAIL_AT_PROD_STEP" in ast.unparse(n)]
    assert len(raises) == 1
    # the in-loop test reads a variable that is None unless the aux block above set it
    guard = parents[raises[0]]
    assert isinstance(guard, ast.If) and "_aux_test_fail_step is not None" in ast.unparse(guard.test)


def test_resume_ledger_check_is_scoped_and_runs_after_the_reseal():
    import gareus.production as production
    src = ast.unparse(ast.parse(inspect.getsource(production.run_gareus)))
    reseal = src.index("reseal_parent_for_resume(")
    ledger = src.index("verify_aux_ledger(")
    dup = src.index("refuse_duplicate_event_keys(")
    assert reseal < ledger and reseal < dup
    assert "anchor_ledger_events(out_dir, manifest)" in src


def test_data_boundary_is_safe_before_the_first_flush(tmp_path):
    """Carry-over 12: no manifest yet (SIGTERM before any sample) -> generation 0, 0 rows."""
    from gareus.auxiliary_cv.runtime_io import data_boundary
    from gareus.store import ParquetExchangeWriter, ParquetSampleWriter
    seg = "seg_001"
    ParquetSampleWriter(tmp_path / "samples" / seg)
    ParquetExchangeWriter(tmp_path / "exchanges" / seg)
    assert data_boundary(tmp_path, seg) == {"samples": {"generation": 0, "n_rows": 0},
                                            "exchanges": {"generation": 0, "n_rows": 0}}
    assert data_boundary(tmp_path, "seg_missing")["samples"] == {"generation": 0, "n_rows": 0}
    w = ParquetSampleWriter(tmp_path / "samples" / seg)
    for s in (10, 20):
        w.write_sample(step=s, replica=0, window_id=0, cv1=0.1, cv2=None, potential=0.0, boost_total=None,
                       boost_dihedral=None, boost_nonbonded=None)
    w.flush()
    got = data_boundary(tmp_path, seg)
    assert got["samples"]["n_rows"] == 2 and got["samples"]["generation"] >= 1
    assert got["exchanges"] == {"generation": 0, "n_rows": 0}


# Ledger scope (carry-over 2) ---------------------------------------------------------------------

def _write_events(run, seg, events):
    from gareus.auxiliary_cv.ledger import EXCHANGE_EVENT_SCHEMA
    from gareus.store import ParquetExchangeWriter
    w = ParquetExchangeWriter(run / "exchanges" / seg, event_schema=EXCHANGE_EVENT_SCHEMA)
    for ev in events:
        w.write_event(**ev)
    w.close()


def _swap(step, seq, i, j, wi, wj, after, accepted=True):
    return dict(step=step, attempt_seq=seq, selected_replica=i, replica_i=i, replica_j=j, window_i=wi, window_j=wj,
                kind="swap", delta_e_kj=0.0, accepted=accepted, log_q_forward=float("nan"),
                log_q_reverse=float("nan"), p_accept=float("nan"), energy_version="v", assignments_after=after)


def _manifest(seg, start_step, start, absolute_step, assignments):
    return {"absolute_step": absolute_step, "assignments": assignments,
            "aux": {"ledger_anchor": {"segment_id": seg, "start_step": start_step, "start_assignments": start}}}


def _registry(tmp_path, n):
    from gareus.store import SegmentRegistry
    reg = SegmentRegistry(tmp_path)
    segs = []
    for _ in range(n):
        s = reg.open_segment("run", segs[-1] if segs else None, 1)
        segs.append(s)
    return reg, segs


def test_ledger_is_replayed_from_the_anchor_segment_only(tmp_path):
    from gareus.auxiliary_cv.checkpoint import verify_aux_ledger
    from gareus.auxiliary_cv.runtime_io import anchor_ledger_events
    _reg, (s0, s1) = _registry(tmp_path, 2)
    # s0: another segment whose events at the same steps would break the replay if they leaked in
    _write_events(tmp_path, s0, [_swap(200, 0, 0, 1, 0, 1, [0, 1]) | {"assignments_after": [5, 5]}])
    _write_events(tmp_path, s1, [_swap(200, 0, 0, 1, 0, 1, [1, 0])])
    manifest = _manifest(s1, 100, [0, 1], 300, [1, 0])
    events = anchor_ledger_events(tmp_path, manifest)
    assert set(np.asarray(events["segment_id"]).astype(str)) == {s1}
    verify_aux_ledger(manifest, events)
    from gareus.query import load_exchanges
    with pytest.raises(IntegrityError):
        verify_aux_ledger(manifest, load_exchanges(tmp_path, segment_ids=[s0, s1]))


def test_ledger_boundary_start_step_exclusive_checkpoint_step_inclusive(tmp_path):
    from gareus.auxiliary_cv.checkpoint import verify_aux_ledger
    from gareus.auxiliary_cv.runtime_io import anchor_ledger_events
    _reg, (s1,) = _registry(tmp_path, 1)
    _write_events(tmp_path, s1, [
        # at start_step: belongs to the state BEFORE the anchor (its checksum would fail if replayed)
        _swap(100, 0, 0, 1, 1, 0, [9, 9]),
        _swap(200, 0, 0, 1, 0, 1, [1, 0]),
        # exactly at the checkpoint step: the exchange ran before the save, so it is replayed
        _swap(300, 0, 0, 1, 1, 0, [0, 1]),
        # after the checkpoint (rolled back): never replayed
        _swap(400, 0, 0, 1, 0, 1, [7, 7]),
    ])
    verify_aux_ledger(_manifest(s1, 100, [0, 1], 300, [0, 1]), anchor_ledger_events(tmp_path, _manifest(s1, 100, [0, 1], 300, [0, 1])))
    with pytest.raises(IntegrityError, match="replays"):
        verify_aux_ledger(_manifest(s1, 100, [0, 1], 300, [1, 0]),
                          anchor_ledger_events(tmp_path, _manifest(s1, 100, [0, 1], 300, [1, 0])))


def test_ledger_without_any_event_needs_the_anchor_assignment(tmp_path):
    """No exchange between the anchor and the checkpoint (e.g. SIGTERM before the first exchange)."""
    from gareus.auxiliary_cv.checkpoint import verify_aux_ledger
    from gareus.auxiliary_cv.runtime_io import anchor_ledger_events
    _reg, (s1,) = _registry(tmp_path, 1)
    m = _manifest(s1, 100, [0, 1], 150, [0, 1])
    events = anchor_ledger_events(tmp_path, m)
    verify_aux_ledger(m, events)
    with pytest.raises(IntegrityError, match="no exchange event"):
        verify_aux_ledger(_manifest(s1, 100, [0, 1], 150, [1, 0]), events)


# Duplicate refusal (carry-over 3) ----------------------------------------------------------------

def test_duplicate_event_keys_are_refused():
    from gareus.auxiliary_cv.ledger import refuse_duplicate_event_keys
    ok = {"step": np.array([10, 10, 20]), "attempt_seq": np.array([0, 1, 0]),
          "segment_id": np.array(["a", "a", "b"], dtype=object)}
    refuse_duplicate_event_keys(ok)
    refuse_duplicate_event_keys({})
    refuse_duplicate_event_keys({"step": np.array([10, 10]), "delta_e": np.array([0.0, 0.0])})   # legacy rows
    bad = {"step": np.array([10, 20, 10]), "attempt_seq": np.array([0, 0, 0]),
           "segment_id": np.array(["a", "a", "b"], dtype=object)}
    with pytest.raises(IntegrityError, match=r"duplicate \(step, attempt_seq\).*step 10.*\['a', 'b'\]"):
        refuse_duplicate_event_keys(bad)


def test_duplicate_sample_keys_are_refused():
    from gareus.auxiliary_cv.offline import refuse_duplicate_sample_keys
    refuse_duplicate_sample_keys({"step": np.array([10, 10, 20]), "replica": np.array([0, 1, 0]),
                                  "segment_id": np.array(["a", "a", "b"], dtype=object)})
    with pytest.raises(IntegrityError, match=r"duplicate \(step, replica\).*step 10, replica 1"):
        refuse_duplicate_sample_keys({"step": np.array([10, 10, 10]), "replica": np.array([0, 1, 1]),
                                      "segment_id": np.array(["a", "a", "b"], dtype=object)})


def test_aux_pool_refuses_duplicate_step_replica_rows(tmp_path):
    """The Task 5 no-checkpoint residual: a pooled crashed parent and its restarted child share steps."""
    import test_aux_loaders as tl
    from gareus.mbar_analysis.loaders import load_parquet
    from gareus.store import SegmentRegistry, WindowSnapshot
    (s1,) = tl._run(tmp_path)
    reg = SegmentRegistry(tmp_path)
    s2 = reg.open_segment("run", s1, 1)
    tl._write_segment(tmp_path, s2, True, np.random.default_rng(2))
    reg.close_segment(s2, 600)
    WindowSnapshot(tmp_path).snapshot(s2, [], "contacts", None, kernel_identity=tl.KI, state_definition=tl._defn(),
                                      equilibrium_analysis_eligible=True, phase_kind="production")
    with pytest.raises(IntegrityError, match=r"duplicate \(step, replica\)"):
        load_parquet(tmp_path)


# Row alignment (carry-over 9) and the resume hash (carry-over 15) -------------------------------

def _aux_defn(*, cv2=None, boost=None, c1=(0.0, 0.4)):
    from aux_c_fixture import BOX, CV1, rows
    from aux_cv_fixture import model_payload
    from gareus.auxiliary_cv.model import AuxModel
    from gareus.correctness.state_identity import make_state_definition
    model = AuxModel.from_mapping(model_payload([(0, 1, 2, 3)], [1.0, 0.5], offset=0.1))
    r = rows([(c1[0], 10.0, 0.0, 0.0, "ordinary"), (c1[1], 10.0, 1.2, 1.5, "auxiliary")], model.model_sha256)
    return make_state_definition(r, physical_system_sha256="a" * 64, ensemble="NVT", temperature_k=300.0,
                                 fixed_box_vectors_nm=BOX, cv1=CV1, cv2=cv2, boost=boost,
                                 aux_models={model.model_sha256: model.to_mapping()})


def test_checkpoint_rows_must_align_with_the_resumed_windows():
    from gareus.auxiliary_cv.checkpoint import check_checkpoint_rows_align
    manifest = {"aux": {"state_definition": _aux_defn()}}
    check_checkpoint_rows_align(manifest, [(0.0, 10.0), (0.4, 10.0)])
    with pytest.raises(IntegrityError, match="row 1"):
        check_checkpoint_rows_align(manifest, [(0.0, 10.0), (0.5, 10.0)])
    with pytest.raises(IntegrityError, match="rows"):
        check_checkpoint_rows_align(manifest, [(0.0, 10.0)])


@pytest.mark.parametrize("field,changed", [
    ("state.cv2.definition", dict(cv2={"kind": "torsion-pca", "units": "dimensionless", "definition": {"x": 2}})),
    ("state.boost.envelope", dict(boost={"kind": "pep", "envelope": {"Vmax": 2.0}})),
])
def test_resume_state_hash_covers_cv2_definition_and_boost_envelope(field, changed):
    from gareus.auxiliary_cv.checkpoint import verify_aux_resume
    from test_aux_checkpoint import INFO, KID, TOPO, _block, _observed
    base = dict(cv2={"kind": "torsion-pca", "units": "dimensionless", "definition": {"x": 1}},
                boost={"kind": "pep", "envelope": {"Vmax": 1.0}})
    saved = _aux_defn(**base)
    now = _aux_defn(**{**base, **changed})
    a = [1, 0]
    block = _block(saved, a, ledger_anchor={"segment_id": "s", "start_step": 0, "start_assignments": a})
    with pytest.raises(IntegrityError, match=field.replace(".", r"\.")):
        verify_aux_resume({"aux": block}, aux_enabled=True, state_definition=now, force_info=INFO, assignments=a,
                          observed_params=_observed(now, a), topology_sha256=TOPO, kernel_identity_digest=KID)


# physical identity (carry-over 7) ----------------------------------------------------------------

def test_physical_system_sha_excludes_the_barostat():
    import openmm
    from gareus.auxiliary_cv.runtime_definition import physical_system_sha256
    s = openmm.System()
    s.addParticle(1.0)
    s.addParticle(1.0)
    bond = openmm.HarmonicBondForce()
    bond.addBond(0, 1, 0.1, 100.0)
    s.addForce(bond)
    plain = physical_system_sha256(openmm, s)
    s.addForce(openmm.MonteCarloBarostat(1.0, 300.0, 25))
    assert physical_system_sha256(openmm, s) == plain
    s.addForce(openmm.CMMotionRemover())
    assert physical_system_sha256(openmm, s) != plain


# readiness NPZ (carry-over 4) --------------------------------------------------------------------

def test_final_report_skips_the_legacy_npz_validators_on_an_aux_run(tmp_path, monkeypatch):
    import gareus.production as production
    called = []
    for name in ("validate_analysis_metadata_readiness", "validate_us_mbar_inputs",
                 "compute_gamd_reweighting_diagnostics"):
        monkeypatch.setattr(production, name, lambda *a, _n=name, **k: called.append(_n) or {"status": "ok"})
    aux_args = SimpleNamespace(_aux_runtime=object(), temperature_k=300.0)
    mbar, legacy, gamd = production._final_report_validations(tmp_path, aux_args, 0.3, 0.3)
    assert called == []
    for rec in (mbar, legacy, gamd):
        assert rec["status"] == "skipped" and "auxiliary" in rec["reason"]
    assert not (tmp_path / "analysis_arrays.npz").exists()
    legacy_args = SimpleNamespace(_aux_runtime=None, temperature_k=300.0)
    production._final_report_validations(tmp_path, legacy_args, 0.3, 0.3)
    assert called == ["validate_analysis_metadata_readiness", "validate_us_mbar_inputs",
                      "compute_gamd_reweighting_diagnostics"]
