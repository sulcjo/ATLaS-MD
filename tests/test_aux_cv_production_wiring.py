# tests/test_aux_cv_production_wiring.py
import ast
import inspect

import numpy as np
import pytest

from aux_cv_fixture import model_payload
from gareus.auxiliary_cv.model import AuxModel
from gareus.auxiliary_cv.runtime import AuxObservationError, aux_bias_matrix_kcal
from gareus.auxiliary_cv.state_table import AuxStateTable
from gareus.production import assemble_bias_matrices

M = AuxModel.from_mapping(model_payload([(0, 1, 2, 3)], [1.0, 0.0]))
TABLE = AuxStateTable(M, (0.0, 1.0, 0.0), (0.0, 2.0, 0.0), (None, None, None))


def test_aux_matrix_values_and_exact_zero_rows():
    z = np.array([0.5, 1.5, -1.0, 0.25])
    out = aux_bias_matrix_kcal(z, TABLE)
    assert out.shape == (3, 4)
    np.testing.assert_array_equal(out[0], 0.0)
    np.testing.assert_array_equal(out[2], 0.0)
    np.testing.assert_allclose(out[1], 0.5 * 2.0 * (z - 1.0) ** 2, rtol=1e-15)


def test_nonfinite_z_is_fatal_when_any_state_is_active():
    with pytest.raises(AuxObservationError):
        aux_bias_matrix_kcal(np.array([0.5, np.nan]), TABLE)


def test_all_zero_table_ignores_z():
    """Sham arm: observe_aux_z may record NaN; with no active state the matrix is exact zeros."""
    zero = AuxStateTable(M, (0.0, 0.0), (0.0, 0.0), (None, None))
    np.testing.assert_array_equal(aux_bias_matrix_kcal(np.array([np.nan, 1.0]), zero), 0.0)


def test_assemble_without_aux_is_bitwise_legacy():
    rng = np.random.default_rng(1)
    d, s, b = rng.normal(size=(3, 3)), rng.normal(size=(3, 3)), rng.normal(size=(3, 3))
    legacy_kcal = d + s + b / 4.184
    kcal, kj = assemble_bias_matrices(d, s, b)
    assert np.array_equal(kcal, legacy_kcal) and np.array_equal(kj, 4.184 * legacy_kcal)
    kcal0, _ = assemble_bias_matrices(d, s, b, aux_bias_kcal=np.zeros((3, 3)))
    assert np.array_equal(kcal0, kcal)


def test_assemble_adds_aux():
    rng = np.random.default_rng(2)
    d, s, b, a = (rng.normal(size=(2, 2)) for _ in range(4))
    kcal, kj = assemble_bias_matrices(d, s, b, aux_bias_kcal=a)
    np.testing.assert_allclose(kcal, d + s + a + b / 4.184, rtol=1e-15)
    np.testing.assert_allclose(kj, 4.184 * kcal, rtol=1e-15)


def _run_gareus_source():
    import gareus.production as production
    return ast.parse(inspect.getsource(production.run_gareus))


def _func(tree, name):
    hits = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(hits) == 1, f"{name} not found exactly once in run_gareus; re-anchor"
    return hits[0]


def _calls(node, name):
    return [n for n in ast.walk(node) if isinstance(n, ast.Call)
            and ((isinstance(n.func, ast.Name) and n.func.id == name)
                 or (isinstance(n.func, ast.Attribute) and n.func.attr == name))]


def test_both_fetch_closures_observe_z_through_one_helper():
    tree = _run_gareus_source()
    for name in ("_fetch_state", "_fetch_exchange_state"):
        assert _calls(_func(tree, name), "_aux_z_for_replica"), f"{name} does not observe aux z"


def test_sample_and_exchange_assembly_both_include_the_aux_matrix():
    tree = _run_gareus_source()
    exch = _func(tree, "_current_exchange_arrays")
    assert _calls(exch, "aux_bias_matrix_kcal")
    assert any(k.arg == "aux_bias_kcal" for c in _calls(exch, "assemble_bias_matrices") for k in c.keywords)
    all_assemble = _calls(tree, "assemble_bias_matrices")
    assert len(all_assemble) == 2 and all(any(k.arg == "aux_bias_kcal" for k in c.keywords) for c in all_assemble)
    assert len(_calls(tree, "aux_bias_matrix_kcal")) == 2


def test_sampled_umbrella_bias_stays_umbrella_only():
    src = inspect.getsource(__import__("gareus.production", fromlist=["run_gareus"]).run_gareus)
    line = [l for l in src.splitlines() if "sampled_umbrella_bias_kj = float(" in l]
    assert line and "aux" not in line[0], "sampled_umbrella_bias_kj must remain primary + secondary only"


def test_run_gareus_builds_aux_runtime_before_contexts_and_guards_population():
    import gareus.production as production
    src = inspect.getsource(production.run_gareus)
    tree = ast.parse(src)
    add_calls = _calls(tree, "add_aux_cv_force")
    assert len(add_calls) == 3, "base system + two starting-structure systems must all carry the aux force"
    assert sum(any(k.arg == "force_group" for k in c.keywords) for c in add_calls) == 2
    base_line = min(c.lineno for c in add_calls)
    # Replica Contexts are built by the nested _build_context_i (dispatched through the pool, so
    # it is not a direct call); the base-system aux force must precede its definition.
    assert base_line < _func(tree, "_build_context_i").lineno
    assert _calls(tree, "load_aux_state_table")
    checks = _calls(tree, "check_feature_atoms")
    assert checks and all(any(k.arg == "topology_sha256" for k in c.keywords) for c in checks)
    causes = {k.value.value for c in _calls(tree, "refuse_aux_population_change") for k in c.keywords
              if k.arg == "cause" and isinstance(k.value, ast.Constant)}
    assert causes == {"seed-reachability filter", "--max-replicas", "US auto-drop", "checkpoint resume"}
    assert _calls(tree, "aux_snapshot_rows"), "Stage B snapshots must carry aux fields (D5)"


def test_resume_refusal_runs_first_in_the_fast_resume_branch():
    import gareus.production as production
    src = inspect.getsource(production.run_gareus)
    i_refuse = src.find("refuse_resume_of_aux_campaign(out_dir")
    i_load = src.find("resume_def = load_resume_run_definition(")
    assert 0 <= i_refuse < i_load, "the aux resume refusal must precede loading the resume definition"
    assert src.find("if fast_resume:") < i_refuse
    # Unconditional: the first statement of the branch, not nested under a CV2/aux condition.
    tree = _run_gareus_source()
    branch = [n for n in ast.walk(tree) if isinstance(n, ast.If) and isinstance(n.test, ast.Name)
              and n.test.id == "fast_resume" and n.body and isinstance(n.body[0], ast.Expr)
              and _calls(n.body[0], "refuse_resume_of_aux_campaign")]
    assert len(branch) == 1


def test_snapshot_rows_with_instances_go_through_stage_a_instance_checks():
    """C1: slot pairing and roles are enforced once, by Stage A's _check_instances."""
    tree = _run_gareus_source()
    checks = _calls(tree, "_check_instances")
    assert len(checks) == 1
    arg = checks[0].args[0]
    assert isinstance(arg, ast.Call) and _calls(arg, "normalize_windows")
    assert _calls(tree, "window_rows"), "instances must come from the aux table (snapshot rows drop them)"
    assert checks[0].lineno > min(c.lineno for c in _calls(tree, "aux_snapshot_rows"))


def test_us_auto_drop_guard_passes_the_real_drop_count():
    """L6: n_now is the survivor count, not len - 1."""
    tree = _run_gareus_source()
    guard = [c for c in _calls(tree, "refuse_aux_population_change") for k in c.keywords
             if k.arg == "cause" and isinstance(k.value, ast.Constant) and k.value.value == "US auto-drop"]
    assert len(guard) == 1
    n_now = ast.unparse(guard[0].args[1])
    assert "dropped_window_indices" in n_now and "- 1" not in n_now


def test_aux_runtime_resolution_checks_population_and_window_order():
    """Task-6 alignment: the table matches the replica count and the window order it was loaded in."""
    src = inspect.getsource(__import__("gareus.production", fromlist=["run_gareus"]).run_gareus)
    i_res = src.find('_aux_rt = getattr(args, "_aux_runtime", None)')
    assert i_res >= 0
    block = src[i_res:i_res + 2500]
    assert "_aux_rt.table.n != nrep" in block
    assert "_aux_window_key" in block


def test_assemble_refuses_aux_matrix_of_wrong_shape():
    from gareus.correctness._io import IntegrityError
    d = np.zeros((3, 3))
    with pytest.raises(IntegrityError, match="shape"):
        assemble_bias_matrices(d, d, d, aux_bias_kcal=np.zeros((3, 2)))


def test_aux_runtime_construction_in_run_gareus_is_gated_by_an_aux_condition():
    """Opt-in: every aux-runtime construction call in run_gareus sits under an aux condition."""
    tree = _run_gareus_source()
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    for name in ("add_aux_cv_force", "load_aux_state_table", "aux_snapshot_rows", "check_feature_atoms",
                 "refuse_aux_population_change", "_check_instances", "canonical_topology_sha256",
                 # Stage C
                 "reseal_parent_for_resume", "verify_aux_ledger", "aux_checkpoint_block", "read_aux_parameters",
                 "aux_table_from_checkpoint", "physical_system_sha256", "build_runtime_state_definition",
                 "aux_io_runtime", "make_aux_record_observer", "write_event", "event_from_gibbs",
                 "aux_event_fields", "check_runtime_parity", "check_exchange_boundary_alignment", "parity_context",
                 "verify_aux_resume", "check_fixed_box", "context_box_nm",
                 # Task 13 carry-overs
                 "solvated_start_topology_identities", "check_checkpoint_rows_align", "anchor_ledger_events",
                 "refuse_duplicate_event_keys", "data_boundary", "embed_cv_definition"):
        for call in _calls(tree, name):
            node, gated = call, False
            while node in parents:
                node = parents[node]
                if isinstance(node, (ast.If, ast.IfExp)) and "aux" in ast.unparse(node.test):
                    gated = True
                    break
            assert gated, f"{name} at line {call.lineno} is not under an aux condition"
