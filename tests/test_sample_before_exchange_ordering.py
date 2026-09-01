"""A sample must carry the window whose bias actually produced it.

`window_id` is what MBAR uses to pick a sample's bias when it builds `u_nk`. If a
sample logged at step t carried the *post*-swap label, every affected frame would
be reweighted against a restraint it never felt, and the PMF would be wrong in a
way no downstream check could see -- the numbers stay finite and plausible.

Production gets this right by ordering: inside the production loop, `sample()` is
called before `attempt_exchanges()`, and `sample()` reads `assignments[r]`
directly. So the label is the window under whose restraint the preceding interval
of dynamics ran.

This is pinned structurally rather than behaviourally. The loop lives inside the
`run_gareus` closure and cannot be driven from a test without booting the whole
MD driver -- even the real-MD oracles in
`tests/test_thermodynamic_validity_*.py` drive lower-level pieces
(`make_langevin_integrator`, `add_umbrella_force`, `set_window`) and never the
loop. The same AST-pinning technique is already used in this repo for the
window-map write ordering (see `tests/test_epoch_window_map_rewrite_after_drop.py`,
which asserts `repair_at < drop_at < open_at`).

What this proves: the two calls cannot be reordered, the writer's label cannot
be decoupled from the live `assignments` array, and no second site can start
permuting that array -- without a test failing.

Mutation battery (2026-09-01, isolated tree, reverted after each):

    CAUGHT  sample() genuinely moved after attempt_exchanges()
    CAUGHT  write_sample(window_id=...) fed a constant instead of the live label
    CAUGHT  a second site permuting `assignments`

An earlier draft of this file missed the first two. It compared `max()` of line
numbers across the whole file, so inserting an *extra* exchange call ahead of
sample() left a later one still preceded by a sample; and it grepped for the
string `w = int(assignments[r])`, which occurs twice, so mutating the writer's
own copy left the other occurrence to satisfy the check. Both are now AST
comparisons within the same loop body and at the writer call itself.

What this still does not prove: that the value reaches Parquet intact. That
would need a full `run_gareus` harness, which does not exist.
"""
from __future__ import annotations

import ast
import inspect
import pathlib

import pytest

import gareus.production as production


def _production_source() -> str:
    return pathlib.Path(inspect.getfile(production)).read_text()


def _call_lines(tree: ast.AST, func_name: str) -> list[int]:
    """Line numbers of every call to a bare name, e.g. `sample(...)`."""
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == func_name:
                out.append(node.lineno)
    return sorted(out)


def _stmt_index_containing(body, func_name):
    """Index of the statement in `body` whose subtree calls `func_name`."""
    for i, stmt in enumerate(body):
        for node in ast.walk(stmt):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == func_name):
                return i
    return None


def test_sample_is_called_before_exchange_in_the_production_loop():
    """The ordering that makes `window_id` mean the pre-swap window.

    Compares positions *within the same loop body*, not line numbers across the
    file: an earlier version of this test used max(line numbers) and a mutation
    that inserted an extra exchange call ahead of sample() slipped through it.
    """
    tree = ast.parse(_production_source())

    loops = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.For, ast.While)):
            continue
        ex = _stmt_index_containing(node.body, "attempt_exchanges")
        if ex is not None:
            loops.append((node, ex))
    assert loops, "no loop body calls attempt_exchanges() -- production loop restructured?"

    checked = 0
    for node, ex_idx in loops:
        sa_idx = _stmt_index_containing(node.body, "sample")
        if sa_idx is None:
            continue                      # a loop that exchanges but never samples
        checked += 1
        assert sa_idx < ex_idx, (
            f"in the loop at line {node.lineno}, sample() is statement {sa_idx} and "
            f"attempt_exchanges() is statement {ex_idx}: a sample taken after the "
            "swap carries the post-swap window_id, and MBAR would reweight it "
            "against a restraint it never felt"
        )
    assert checked, "found no loop body containing both calls -- re-anchor this test"


def test_the_sample_writer_labels_with_the_live_assignment():
    """`window_id` is the live `assignments[r]`, checked at the writer call.

    Not a substring search: `w = int(assignments[r])` appears more than once in
    the file, so grepping for it passes even when the writer's own copy is
    mutated. This walks to the write_sample() call and back to `w`'s binding in
    the same function.
    """
    tree = ast.parse(_production_source())

    writers = [
        (fn, node)
        for fn in ast.walk(tree) if isinstance(fn, ast.FunctionDef)
        for node in ast.walk(fn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "write_sample"
    ]
    assert writers, "no write_sample() call found -- sample writer restructured?"

    for fn, call in writers:
        kw = {k.arg: k.value for k in call.keywords}
        assert "window_id" in kw, f"write_sample() in {fn.name}() has no window_id"
        label = ast.unparse(kw["window_id"])
        # whatever name it uses, that name must be bound from assignments[...]
        bindings = [
            ast.unparse(a.value)
            for a in ast.walk(fn)
            if isinstance(a, ast.Assign)
            and any(ast.unparse(t) == label for t in a.targets)
        ]
        assert bindings, f"write_sample(window_id={label}) but {label} is never bound in {fn.name}()"
        assert any("assignments[" in b for b in bindings), (
            f"write_sample(window_id={label}) in {fn.name}(), but {label} is bound "
            f"from {bindings} -- not from the live assignments array. The label "
            "would no longer track the restraint that was applied."
        )


def test_only_the_swap_mutates_the_assignment():
    """If anything else reassigned windows, the ordering argument would not hold.

    `apply_window_swap` is the one place `assignments` is permuted; everything
    else reads it. Guarded here because the ordering guarantee above is only
    meaningful if the label is stable between samples.
    """
    tree = ast.parse(_production_source())
    swaps = []
    for node in ast.walk(tree):
        # the tuple-swap idiom: assignments[i], assignments[j] = assignments[j], assignments[i]
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Tuple):
            tgt = ast.unparse(node.targets[0])
            if "assignments[" in tgt:
                swaps.append(node.lineno)
    assert len(swaps) == 1, (
        f"expected exactly one site permuting `assignments`, found {len(swaps)} at "
        f"lines {swaps}. Each extra site is a place a sample's label can change "
        "without an exchange having happened."
    )
    # ...and it must live in the extracted kernel, which the exact-enumeration
    # test drives, not back inside an untestable closure.
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "apply_window_swap"
    )
    assert fn.lineno < swaps[0] < (fn.end_lineno or 10**9), (
        "the assignments permutation moved out of apply_window_swap; "
        "tests/test_exchange_kernel_exact.py no longer covers the bookkeeping"
    )


def test_apply_window_swap_leaves_labels_alone_when_it_rejects():
    """A rejected swap must not touch the label a later sample will record."""
    import numpy as np

    bias = np.array([[0.0, 50.0], [50.0, 0.0]])   # swapping is very unfavourable
    assignments = np.array([0, 1], dtype=np.int64)
    replica_of_window = np.array([0, 1], dtype=np.int64)

    out = production.apply_window_swap(
        bias, 0.4009, assignments, replica_of_window, 0, 1, uniform=0.999999,
    )
    assert out is not None and not out.accepted
    assert list(assignments) == [0, 1]
    assert list(replica_of_window) == [0, 1]
