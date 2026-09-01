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

Mutation battery (2026-09-01, applied to the shipped file, restored after each;
counts over the 16 tests in this file plus tests/test_exchange_kernel_exact.py):

    CAUGHT  sample() removed from the production loop             1 failed
    CAUGHT  an EXTRA sample() after attempt_exchanges()           1 failed
    CAUGHT  `w = 0` rebinding the label before write_sample()     1 failed
    CAUGHT  a second site permuting `assignments`                 1 failed
    CAUGHT  bulk `assignments[:] =` without the holder resync     1 failed
    CAUGHT  gibbs call site force_accept instead of p_override    1 failed

Three of these passed against the first draft of this file, and each failure was
a different kind of too-coarse comparison:

* `max()` of line numbers across the whole file -- an extra exchange call ahead
  of sample() left a later one still preceded by a sample.
* statement index within the loop body -- a sample() added inside the *same*
  `if` block as the exchange shares one top-level statement, so the indices are
  equal and `later than` is false. Now compared by line number within the loop.
* `any(binding is from assignments)` over the whole function -- a later
  `w = 0` leaves the good binding in place to satisfy it. Now the binding in
  effect at the call line.

A fourth hole was found by the fixed test itself rather than by review: matching
only tuple-target assignments made the resume path's `assignments[:] = ...`
invisible. That write is legitimate, but only because `_refresh_replica_of_window()`
follows it, which is now the stated and enforced invariant.

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


def _call_lines_within(node: ast.AST, func_name: str) -> list[int]:
    """Line numbers of every call to a bare name inside `node`'s subtree."""
    return sorted(
        n.lineno for n in ast.walk(node)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        and n.func.id == func_name
    )


def _stmt_indices_containing(body, func_name):
    """Indices of EVERY statement in `body` whose subtree calls `func_name`."""
    out = []
    for i, stmt in enumerate(body):
        for node in ast.walk(stmt):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == func_name):
                out.append(i)
                break
    return out


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

    Compares call LINE NUMBERS confined to the loop body. Two coarser versions
    were tried first and both had real holes:

    * `max()` of line numbers across the whole file -- an extra exchange call
      inserted ahead of sample() left a later one still preceded by a sample.
    * statement index within the loop body -- if a sample() call is added
      *inside the same `if` block* as attempt_exchanges(), both live in one
      top-level statement, the indices are equal, and `sample_idx > exch_idx`
      is false. Verified: that mutation passed.

    Line numbers inside the loop body are finer than both and still immune to
    the file-wide problem.

    What this cannot see: control flow. A sample() call textually above the
    exchange but in a branch that runs after it would pass. Nothing short of
    driving the real loop would catch that, and no such harness exists.
    """
    tree = ast.parse(_production_source())

    loops = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.For, ast.While)):
            continue
        ex = _call_lines_within(node, "attempt_exchanges")
        if ex:
            loops.append((node, ex))
    assert loops, "no loop body calls attempt_exchanges() -- production loop restructured?"

    checked = 0
    for node, ex_lines in loops:
        sa_lines = _call_lines_within(node, "sample")
        if not sa_lines:
            continue                      # a loop that exchanges but never samples
        checked += 1
        first_exchange = min(ex_lines)
        late = [ln for ln in sa_lines if ln > first_exchange]
        assert not late, (
            f"in the loop at line {node.lineno}, attempt_exchanges() is first called "
            f"at line {first_exchange} but sample() is also called at line(s) {late}: "
            "a sample taken after the swap carries the post-swap window_id, and MBAR "
            "would reweight it against a restraint it never felt"
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
        # The LAST binding that precedes the call, not any binding anywhere in
        # the function. `any(...)` over all bindings passes even when a later
        # `w = 0` overwrites the good one -- another real miss in the first
        # draft. Only the binding actually in effect at the call site counts.
        bindings = sorted(
            (a.lineno, ast.unparse(a.value))
            for a in ast.walk(fn)
            if isinstance(a, ast.Assign)
            and any(ast.unparse(t) == label for t in a.targets)
            and a.lineno < call.lineno
        )
        assert bindings, (
            f"write_sample(window_id={label}) at line {call.lineno} but {label} is "
            f"never bound before it in {fn.name}()"
        )
        eff_line, eff = bindings[-1]
        assert "assignments[" in eff, (
            f"write_sample(window_id={label}) in {fn.name}(): the binding in effect "
            f"at line {call.lineno} is `{label} = {eff}` (line {eff_line}), not the "
            "live assignments array. The label would no longer track the restraint "
            "that was applied."
        )


def test_every_site_that_writes_assignments_is_accounted_for():
    """Enumerate ALL writes to `assignments`, not just the tuple-swap idiom.

    The first draft matched only `ast.Assign` nodes with a Tuple target, so it
    saw the swap and nothing else. Production also does

        assignments[:] = [int(x) for x in manifest.get("assignments", ...)]

    on the resume path -- a whole-array rewrite that the check was structurally
    blind to. That one is legitimate, but only because of an invariant the test
    must now state: it is immediately followed by `_refresh_replica_of_window()`,
    which rebuilds the holder table. A bulk write WITHOUT that resync leaves the
    two arrays desynced, and `gibbs-walk` builds its next proposal from the
    holder table -- so the proposal distribution would silently be computed for
    the wrong replicas.
    """
    tree = ast.parse(_production_source())

    # Scope to the function that owns the driver's array -- the one running the
    # production loop. `assignments` in any other function is a different local
    # object (load_production_checkpoint builds its own and returns a manifest),
    # and conflating them makes this test fail on code that is fine.
    drivers = [
        fn for fn in ast.walk(tree)
        if isinstance(fn, ast.FunctionDef)
        and any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                and n.func.id == "attempt_exchanges" for n in ast.walk(fn))
    ]
    assert drivers, "no function calls attempt_exchanges() -- re-anchor this test"
    driver = max(drivers, key=lambda f: (f.end_lineno or 0) - f.lineno)

    swaps, bulk = [], []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        in_driver = driver.lineno <= node.lineno <= (driver.end_lineno or 0)
        for tgt in node.targets:
            txt = ast.unparse(tgt)
            if isinstance(tgt, ast.Tuple):
                # unparses as "(assignments[i], assignments[j])"
                if "assignments[" in txt:
                    swaps.append(node)
            elif txt == "assignments" or txt.startswith("assignments["):
                if in_driver:
                    bulk.append(node)

    assert len(swaps) == 1, (
        f"expected exactly one site permuting `assignments` element-wise, found "
        f"{len(swaps)} at lines {[n.lineno for n in swaps]}. Each extra site is a "
        "place a sample's label can change without an exchange having happened."
    )
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "apply_window_swap"
    )
    assert fn.lineno < swaps[0].lineno < (fn.end_lineno or 10**9), (
        "the assignments permutation moved out of apply_window_swap; "
        "tests/test_exchange_kernel_exact.py no longer covers the bookkeeping"
    )

    # Every bulk rewrite must resync the holder table within a few statements.
    src_lines = _production_source().splitlines()
    for node in bulk:
        # Initialising the array to the identity permutation is not a rewrite of
        # live state -- it happens once, before any replica has sampled.
        if ast.unparse(node.value).startswith("list(range("):
            continue
        window = "\n".join(src_lines[node.lineno - 1: node.lineno + 4])
        assert "_refresh_replica_of_window()" in window, (
            f"`{ast.unparse(node)}` at line {node.lineno} rewrites the whole "
            "assignment array but does not call _refresh_replica_of_window() "
            "right after it. assignments and replica_of_window would desync, and "
            "gibbs-walk reads the holder table to build its next proposal."
        )


def test_the_gibbs_branch_metropolis_corrects_rather_than_force_accepting():
    """`gibbs-walk` must hand its MH probability to the swap, not force-accept.

    The heat-bath proposal is nonuniform, so accepting a selected move
    unconditionally does not target pi. The numeric kernel test cannot see this
    one: it drives `gibbs_propose_one_replica` and `apply_window_swap` directly,
    while the substitution would happen at the call site *between* them, inside
    the run_gareus closure. Verified as a real gap -- with the numeric test
    alone, replacing `p_override=prop.pacc` with `force_accept=True` passed all
    11 assertions.
    """
    tree = ast.parse(_production_source())
    calls = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        and n.func.id == "_attempt_window_swap"
        and any(k.arg == "p_override" for k in n.keywords)
    ]
    assert calls, (
        "no _attempt_window_swap(..., p_override=...) call found -- the gibbs "
        "branch no longer applies its Metropolis-Hastings correction"
    )
    for c in calls:
        kw = {k.arg: ast.unparse(k.value) for k in c.keywords}
        assert "force_accept" not in kw, (
            f"_attempt_window_swap at line {c.lineno} passes both p_override and "
            f"force_accept={kw['force_accept']}: force_accept wins, so the MH "
            "correction is computed and then discarded."
        )
        assert ".pacc" in kw["p_override"], (
            f"_attempt_window_swap at line {c.lineno} has p_override="
            f"{kw['p_override']}, which is not a GibbsProposal.pacc -- the "
            "acceptance no longer comes from the corrected proposal."
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
