# NPT Barostat Performance (Stage 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the Python per-molecule loop that consumes ~77% of wall time in the biased-MC barostat, without changing a single number the simulation produces.

**Architecture:** The centroid translation inside `attempt_due` is extracted into two pure module-level functions and reimplemented with `np.bincount`, driven by index arrays precomputed once per controller. Per-phase timers are added to the controller so stage 2 can be decided on measurement. Separately, the throughput reporter is corrected — it currently divides cumulative simulated time by process-local elapsed time, reading ~19x high after any resume.

**Tech Stack:** Python 3.9, numpy 1.24.2 (aurum2 production env), OpenMM 8.3.1, pytest (delegated — see Global Constraints).

**Spec:** `docs/superpowers/specs/2026-09-15-npt-barostat-perf-design.md`

## Global Constraints

- **Bit-identical is the bar.** Every change either provably produces the same numbers or touches only measurement. Equality assertions use `==`, never `np.allclose` — a tolerance would not test the claim being made.
- **Do NOT bump `_CONTROLLER_SCHEMA_VERSION`** (`gareus/npt.py:215`, currently `1`). `BiasedMCBarostatController.restore` (`gareus/npt.py:702`) raises on any mismatch (`gareus/npt.py:730`), which would break resume for the live `chignolin_7` chain. Note the method is `restore`, not `from_state_dict`.
- **Do NOT widen `_RESTORE_VERIFY_STRIDE`** (`gareus/npt.py:600`, currently `256`). Explicit non-goal; see spec §7.
- **Do NOT change `barostat_frequency` defaults or the acceptance rule.** Sampling behaviour is out of scope.
- **Never run pytest directly.** A repo hook blocks it, and the hook matches the literal string *anywhere* in the command — including inside a `grep` pattern or a heredoc. Delegate every test run through `opencode run` exactly as written in each step. For the same reason, use the Write/Edit tools (not a Bash heredoc) for any file whose content contains that word.
- **Known-failing baseline:** the suite has 6 known failures plus 1 flaky test (`test_finite_difference_boost_force_matches_scaling_factor_expression`). These pre-exist this work. Record the baseline in Task 0 and compare against it — do not claim green against zero.
- **`*.md` is gitignored** (`.gitignore:32`), with only `docs/atlas-md/**` re-included. Committing any plan, spec, or doc requires `git add -f`. `git status` will not show them.
- **Deployment target is `/home/sulcjo/2026_peptide_sampler` on aurum2** — the tree named by the job's `PYTHONPATH`. `~/gareus/gareus/` is a stale divergent copy; never deploy there and never read it for line numbers.

---

### Task 0: Record the test baseline

**Files:**
- Create: none
- Modify: none
- Test: whole suite

**Interfaces:**
- Consumes: nothing
- Produces: a recorded baseline failure list that Tasks 1-5 compare against

- [ ] **Step 1: Run the full suite and record the result**

Run (generous timeout, the suite takes 10+ minutes):

```bash
opencode run "In /run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler: run exactly the following command and report the result: cd /run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler && python -m pytest -q. This is a read-only task - do not edit, commit, or fix anything. Report concisely: pass/fail/error counts, each failing test id with file:line, and total runtime."
```

Expected: ~2959 passed, 6 failed, 1 flaky. Write the exact failing test ids into a scratch note; every later task must show the same set, neither more nor fewer.

---

### Task 1: Pure, vectorized centroid scaling

Extract the hot loop into testable pure functions. No wiring yet — this task's deliverable stands alone.

**Files:**
- Modify: `gareus/npt.py` (add two module-level functions near `_molecules_from_context`, around line 288)
- Test: `tests/test_npt_centroid_scaling.py` (create)

**Interfaces:**
- Consumes: nothing
- Produces:
  - `_molecule_index_arrays(molecules: list[list[int]], n_atoms: int) -> tuple[np.ndarray, np.ndarray]` returning `(mol_ids: int32[n_atoms], mol_sizes: float64[n_mol])`
  - `_scale_about_molecule_centroids(positions: np.ndarray, mol_ids: np.ndarray, mol_sizes: np.ndarray, scale_minus_one: float) -> np.ndarray`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_npt_centroid_scaling.py`:

```python
"""Bit-identity of the vectorized centroid translation against the loop it replaces.

Assertions here use == deliberately. The claim is that the vectorized form
produces the same IEEE-754 doubles as the per-molecule loop, not that it
produces close ones; a tolerance would not test that claim.
"""
import numpy as np
import pytest

from gareus.npt import (
    _MEAN_FALLBACK_MIN_ATOMS,
    _molecule_index_arrays,
    _scale_about_molecule_centroids,
)


def _reference_loop(positions, molecules, scale_minus_one):
    """The implementation being replaced, verbatim in behaviour."""
    new_positions = positions.copy()
    for mol in molecules:
        center = positions[mol].mean(axis=0)
        new_positions[mol] = positions[mol] + scale_minus_one * center
    return new_positions


def _large(molecules):
    return [(m, mol) for m, mol in enumerate(molecules)
            if len(mol) >= _MEAN_FALLBACK_MIN_ATOMS]


def _check(molecules, n_atoms, scale_minus_one=0.0003, seed=0):
    rng = np.random.default_rng(seed)
    positions = rng.random((n_atoms, 3)) * 6.0
    mol_ids, mol_sizes = _molecule_index_arrays(molecules, n_atoms)
    got = _scale_about_molecule_centroids(
        positions, mol_ids, mol_sizes, scale_minus_one, _large(molecules))
    want = _reference_loop(positions, molecules, scale_minus_one)
    assert got.shape == want.shape
    assert np.array_equal(got, want), f"max abs diff {np.abs(got - want).max():.3e}"


def test_production_shape_waters_ions_and_one_solute():
    molecules = [list(range(174))]
    nxt = 174
    for _ in range(36):
        molecules.append([nxt]); nxt += 1
    while nxt + 3 <= 19008:
        molecules.append([nxt, nxt + 1, nxt + 2]); nxt += 3
    _check(molecules, nxt)


def test_molecules_listed_out_of_order_with_noncontiguous_indices():
    rng = np.random.default_rng(7)
    perm = rng.permutation(300)
    molecules = [sorted(perm[i:i + 3].tolist()) for i in range(0, 300, 3)]
    rng.shuffle(molecules)
    _check(molecules, 300)


def test_single_atom_molecules_only():
    _check([[i] for i in range(50)], 50)


def test_one_molecule_spanning_the_system():
    _check([list(range(500))], 500)


def test_single_atom_system():
    _check([[0]], 1)


@pytest.mark.parametrize("size", [4, 7, 16, 63, 127])
def test_bincount_window_summation_order_is_exact(size):
    """The sizes that actually traverse bincount: 1 to 127.

    With the >=128 fallback in place, testing 174/1000/10000 atoms would compare
    ndarray.mean against itself and prove nothing. This window is where bincount
    really runs and where a pairwise-vs-sequential divergence would bite. Both
    contiguous and shuffled layouts, since ordering is what changes summation
    order.

    If this ever fails, match numpy's summation explicitly -- do NOT relax to a
    tolerance, which would silently downgrade a bit-identical change into a
    statistically-equivalent one.
    """
    rng = np.random.default_rng(size)
    _check([list(range(size)), list(range(size, size + 3))], size + 3)
    order = rng.permutation(size + 3).tolist()
    _check([order[:size], order[size:]], size + 3, seed=size)


@pytest.mark.parametrize("size", [174, 1000, 10000])
def test_large_molecules_bypass_bincount_entirely(size):
    """Documents WHY the old large-size tests were vacuous.

    These molecules take the mean fallback, so identity is by construction. The
    assertion here is about routing, not arithmetic: if the threshold were ever
    raised above one of these sizes, the guarantee silently becomes empirical
    again and this test says so.
    """
    molecules = [list(range(size))]
    assert _large(molecules) == [(0, molecules[0])], (
        f"size {size} no longer takes the mean fallback; the bit-identity "
        "guarantee for it is now empirical, not by construction")
    _check(molecules, size)


def test_result_dtype_is_float64_like_the_original():
    """np.bincount returns float64 regardless of input.

    A float32 position array would be silently upcast and break identity.
    Production cannot hit this -- _read_positions_and_box builds the array with
    dtype=float -- but the helper must not quietly change dtype on its own.
    """
    molecules = [[0, 1, 2], [3]]
    mol_ids, mol_sizes = _molecule_index_arrays(molecules, 4)
    positions = np.random.default_rng(0).random((4, 3))
    assert positions.dtype == np.float64
    out = _scale_about_molecule_centroids(
        positions, mol_ids, mol_sizes, 0.01, _large(molecules))
    assert out.dtype == np.float64


def test_large_molecule_fallback_actually_fires_and_uses_mean():
    """The guarantee only holds if the fallback is reached. Prove it is.

    A bincount-only result is compared against one where the large molecule's
    centroid is forced to a sentinel; if the fallback were dead code the two
    would agree.
    """
    size = _MEAN_FALLBACK_MIN_ATOMS + 1
    molecules = [list(range(size)), [size, size + 1, size + 2]]
    n_atoms = size + 3
    assert _large(molecules) == [(0, molecules[0])]

    rng = np.random.default_rng(5)
    positions = rng.random((n_atoms, 3)) * 6.0
    mol_ids, mol_sizes = _molecule_index_arrays(molecules, n_atoms)

    with_fallback = _scale_about_molecule_centroids(
        positions, mol_ids, mol_sizes, 0.01, _large(molecules))
    without = _scale_about_molecule_centroids(
        positions, mol_ids, mol_sizes, 0.01, ())

    assert np.array_equal(with_fallback, _reference_loop(positions, molecules, 0.01))
    # Small molecules are below any pairwise blocksize, so the two paths agree
    # there; the assertion that matters is that the fallback path was taken.
    assert np.array_equal(with_fallback[size:], without[size:])


def test_small_molecules_are_below_the_fallback_threshold():
    """Waters and ions must never take the slow path -- that is the whole point."""
    molecules = [[0, 1, 2], [3], [4, 5]]
    assert _large(molecules) == []


def test_index_arrays_cover_every_atom_exactly_once():
    molecules = [[0, 1, 2], [3], [4, 5]]
    mol_ids, mol_sizes = _molecule_index_arrays(molecules, 6)
    assert mol_ids.tolist() == [0, 0, 0, 1, 2, 2]
    assert mol_sizes.tolist() == [3.0, 1.0, 2.0]


def test_zero_scale_is_the_identity():
    rng = np.random.default_rng(3)
    positions = rng.random((30, 3)) * 6.0
    molecules = [[i, i + 1, i + 2] for i in range(0, 30, 3)]
    mol_ids, mol_sizes = _molecule_index_arrays(molecules, 30)
    got = _scale_about_molecule_centroids(positions, mol_ids, mol_sizes, 0.0)
    assert np.array_equal(got, positions)


@pytest.mark.parametrize("seed", range(25))
def test_property_random_valid_partitions_are_bit_identical(seed):
    """Random total, non-overlapping partitions of random size and shape."""
    rng = np.random.default_rng(1000 + seed)
    n_atoms = int(rng.integers(1, 400))
    order = rng.permutation(n_atoms).tolist()
    molecules = []
    i = 0
    while i < n_atoms:
        take = min(int(rng.integers(1, 9)), n_atoms - i)
        molecules.append(order[i:i + take])
        i += take
    rng.shuffle(molecules)
    _check(molecules, n_atoms,
           scale_minus_one=float(rng.uniform(-0.01, 0.01)), seed=seed)
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
opencode run "In /run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler: run exactly the following command and report the result: cd /run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler && python -m pytest tests/test_npt_centroid_scaling.py -q. This is a read-only task - do not edit, commit, or fix anything. Report concisely: pass/fail/error counts, each failing test id with file:line, and total runtime."
```

Expected: collection error, `ImportError: cannot import name '_molecule_index_arrays' from 'gareus.npt'`.

- [ ] **Step 3: Write the implementation**

In `gareus/npt.py`, immediately after `_molecules_from_context` (which ends at line 288), add:

```python
def _molecule_index_arrays(molecules, n_atoms: int):
    """Flat per-atom molecule ids and per-molecule atom counts.

    Built once per controller. ``_molecules_from_context`` has already proven
    the partition is total and non-overlapping, so every atom appears in
    exactly one molecule and the ids are a complete labelling. No contiguity
    is assumed: molecules may be listed in any order and hold non-contiguous
    indices.
    """
    mol_ids = np.empty(int(n_atoms), dtype=np.int32)
    mol_sizes = np.empty(len(molecules), dtype=np.float64)
    for m, mol in enumerate(molecules):
        mol_ids[mol] = m
        mol_sizes[m] = float(len(mol))
    return mol_ids, mol_sizes


_MEAN_FALLBACK_MIN_ATOMS = 128


def _scale_about_molecule_centroids(positions, mol_ids, mol_sizes, scale_minus_one: float,
                                    large_molecules=()):
    """Translate every molecule by ``scale_minus_one`` times its centroid.

    Internal geometry is untouched: every atom of a molecule receives the same
    displacement. This replaces a per-molecule Python loop that cost 57.1 ms
    per attempt at 6,303 molecules against 0.314 ms here.

    ``bincount`` accumulates sequentially; ``ndarray.mean`` may sum pairwise
    above a blocksize. The two agree for the 1- and 3-atom molecules that make
    up almost the whole system, and agreed empirically for the 174-atom solute,
    but only the original call is *guaranteed* to reproduce the original value.
    Molecules of at least ``_MEAN_FALLBACK_MIN_ATOMS`` atoms therefore keep
    using ``positions[mol].mean(axis=0)``. There is one such molecule in this
    system, so the fallback costs nothing and converts the one case that could
    diverge from empirically identical into identical by construction.
    """
    n_mol = int(mol_sizes.shape[0])
    sums = np.empty((n_mol, 3), dtype=np.float64)
    for k in range(3):
        sums[:, k] = np.bincount(mol_ids, weights=positions[:, k], minlength=n_mol)
    centers = sums / mol_sizes[:, None]
    for m, mol in large_molecules:
        centers[m] = positions[mol].mean(axis=0)
    return positions + scale_minus_one * centers[mol_ids]
```

`large_molecules` is a precomputed sequence of `(molecule_index, atom_index_list)`
pairs — see Task 2, which builds it once and passes it from the controller.

The three-iteration loop is over spatial axes, not molecules — its cost does not grow with system size.

- [ ] **Step 4: Run the tests to verify they pass**

```bash
opencode run "In /run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler: run exactly the following command and report the result: cd /run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler && python -m pytest tests/test_npt_centroid_scaling.py -q. This is a read-only task - do not edit, commit, or fix anything. Report concisely: pass/fail/error counts, each failing test id with file:line, and total runtime."
```

Expected: 37 passed (10 named tests, 3 large-molecule parametrisations, 25 property seeds).

- [ ] **Step 5: Commit**

```bash
cd /run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler
git add gareus/npt.py tests/test_npt_centroid_scaling.py
git commit -m "perf(npt): add vectorized molecule-centroid scaling helpers

The per-molecule Python loop in attempt_due costs 57.1 ms per volume move at
6,303 molecules and holds the GIL on all 64 replica workers. These helpers
compute the same array with three vectorized operations, measured at 0.314 ms
and bit-identical. Not yet wired in.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01LR21pBRrL23jbmmPFEEeuZ"
```

---

### Task 2: Wire the vectorized path into the controller

**Files:**
- Modify: `gareus/npt.py:320-338` (`_ControllerCore.__init__`), `gareus/npt.py:519-523` (`attempt_due`)
- Test: `tests/test_npt_centroid_scaling.py` (append), `tests/test_npt_transaction_equivalence.py` (create)

**Interfaces:**
- Consumes: `_molecule_index_arrays`, `_scale_about_molecule_centroids` from Task 1
- Produces: `_ControllerCore._mol_ids`, `_ControllerCore._mol_sizes` attributes

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_npt_centroid_scaling.py`:

```python
def test_controller_precomputes_index_arrays_covering_the_partition():
    """The controller's arrays must describe exactly the partition it stores."""
    from gareus.npt import _ControllerCore

    molecules = [[0, 1, 2], [3, 4, 5], [6]]
    core = _ControllerCore.__new__(_ControllerCore)
    core._molecules = molecules
    core._n_mol = len(molecules)
    _ControllerCore._install_molecule_index_arrays(core, 7)

    assert core._mol_ids.tolist() == [0, 0, 0, 1, 1, 1, 2]
    assert core._mol_sizes.tolist() == [3.0, 3.0, 1.0]
    assert core._mol_ids.dtype == np.int32
```

Create `tests/test_npt_transaction_equivalence.py`. This is the test that
actually defends the bit-identity claim end-to-end: it runs the whole volume-move
transaction twice — once vectorized, once with the loop it replaced — and
compares results, RNG state, and the Context's final coordinates.

```python
"""The vectorized volume move must be indistinguishable from the loop it replaced.

Not just the translated array: the whole transaction. Same acceptance decision,
same random stream position, same coordinates left in the Context, across
accepted moves, rejected moves and geometrically invalid proposals.

Cross-module test imports follow the existing convention in this suite (see
tests/test_thermodynamic_validity_2d.py:137); tests/ has no __init__.py and
pythonpath = ["."] is set in pyproject.toml.
"""
import numpy as np
import pytest

openmm = pytest.importorskip("openmm")

import gareus.npt as npt_mod  # noqa: E402
from test_npt_controller_mechanics import (  # noqa: E402
    VolumeStubAdapter,
    _make_controller,
    _rigid_molecule_system,
    _snapshot_state,
)


def _loop_reference_for(molecules):
    """Exactly the replaced implementation, iterating the ORIGINAL molecule lists.

    Reconstructing molecules with np.flatnonzero(mol_ids == m) would be wrong:
    flatnonzero returns atom indices in ascending order, while the original loop
    iterated `self._molecules` in list order with whatever ordering getMolecules()
    gave. For a non-contiguous or unsorted molecule the summation order differs,
    so such a reference would compare the new code against a *third*
    implementation rather than against the code being replaced.
    """
    def impl(positions, mol_ids, mol_sizes, scale_minus_one, large=()):
        new_positions = positions.copy()
        for mol in molecules:
            center = positions[mol].mean(axis=0)
            new_positions[mol] = positions[mol] + scale_minus_one * center
        return new_positions
    return impl


def _run(ctrl, n):
    out = []
    step = ctrl.next_due_step
    for _ in range(n):
        res = ctrl.attempt_due(step)
        out.append((res.proposed_volume_nm3, res.log_acceptance,
                    res.accepted, res.reason))
        step = ctrl.next_due_step
    return out


def _rng_state(ctrl):
    return ctrl.state_dict()["rng"]


@pytest.mark.parametrize("physical", [
    lambda v: 5.0 * (v - 8.5) ** 2,   # mixes accepts and rejects
    lambda v: 0.0,                     # accepts almost everything
    lambda v: 1.0e4 * v,               # rejects almost everything
])
def test_transaction_matches_the_loop_implementation(monkeypatch, physical):
    ctx_a, ctrl_a = _make_controller(
        _rigid_molecule_system(n_mol=12), VolumeStubAdapter(physical=physical), seed=4242)
    vectorized = _run(ctrl_a, 30)
    pos_a, box_a = _snapshot_state(ctx_a)
    rng_a = _rng_state(ctrl_a)

    system_b = _rigid_molecule_system(n_mol=12)
    molecules_b = [[int(i) for i in mol] for mol in
                   npt_mod._molecules_from_context(
                       openmm.Context(system_b, openmm.VerletIntegrator(0.001),
                                      openmm.Platform.getPlatformByName("Reference")))]
    monkeypatch.setattr(npt_mod, "_scale_about_molecule_centroids",
                        _loop_reference_for(molecules_b))
    ctx_b, ctrl_b = _make_controller(
        system_b, VolumeStubAdapter(physical=physical), seed=4242)
    looped = _run(ctrl_b, 30)
    pos_b, box_b = _snapshot_state(ctx_b)
    rng_b = _rng_state(ctrl_b)

    assert vectorized == looped
    assert rng_a == rng_b
    assert np.array_equal(pos_a, pos_b)
    assert np.array_equal(box_a, box_b)
    # The parametrisation is worthless if every case lands on one branch.
    assert {r[2] for r in vectorized} != set() and len(vectorized) == 30


def test_both_acceptance_branches_are_actually_exercised():
    """Guards the test above: a suite where nothing is ever accepted proves nothing."""
    _ctx, ctrl = _make_controller(
        _rigid_molecule_system(n_mol=12),
        VolumeStubAdapter(physical=lambda v: 5.0 * (v - 8.5) ** 2), seed=4242)
    outcomes = {r[2] for r in _run(ctrl, 30)}
    assert outcomes == {True, False}


def test_positions_array_is_not_mutated_by_the_scaling():
    """The restore path hands back the same array it read; mutating it corrupts."""
    from gareus.npt import _molecule_index_arrays, _scale_about_molecule_centroids

    molecules = [[0, 1, 2], [3, 4, 5]]
    mol_ids, mol_sizes = _molecule_index_arrays(molecules, 6)
    rng = np.random.default_rng(0)
    positions = rng.random((6, 3))
    before = positions.copy()
    _scale_about_molecule_centroids(positions, mol_ids, mol_sizes, 0.01)
    assert np.array_equal(positions, before)
```

> `_make_controller`, `_rigid_molecule_system`, `VolumeStubAdapter` and
> `_snapshot_state` all exist in `tests/test_npt_controller_mechanics.py`
> (lines 27-98). Do not redefine them.

- [ ] **Step 2: Run the test to verify it fails**

```bash
opencode run "In /run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler: run exactly the following command and report the result: cd /run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler && python -m pytest tests/test_npt_centroid_scaling.py::test_controller_precomputes_index_arrays_covering_the_partition tests/test_npt_transaction_equivalence.py -q. This is a read-only task - do not edit, commit, or fix anything. Report concisely: pass/fail/error counts, each failing test id with file:line, and total runtime."
```

Expected: FAIL, `AttributeError: type object '_ControllerCore' has no attribute '_install_molecule_index_arrays'`. The transaction tests may pass at this point — the controller is still running the loop and is trivially equal to itself. They become meaningful after Step 4, which is where they are re-run.

- [ ] **Step 3: Add the precomputation**

In `gareus/npt.py`, inside `_ControllerCore`, add this method directly after `__init__` (which ends at line 338):

```python
    def _install_molecule_index_arrays(self, n_atoms: int) -> None:
        """Derive the flat index arrays the vectorized volume move needs.

        ``_large_molecules`` lists the few molecules big enough that numpy's
        mean may reduce pairwise; those keep the original ``.mean`` call so
        their centroid is identical by construction, not by observation.
        """
        self._mol_ids, self._mol_sizes = _molecule_index_arrays(self._molecules, n_atoms)
        self._large_molecules = [
            (m, mol) for m, mol in enumerate(self._molecules)
            if len(mol) >= _MEAN_FALLBACK_MIN_ATOMS
        ]
```

Then in `__init__`, immediately after the existing line `self._max_cutoff_nm = float(max_cutoff_nm)` (line 338), add:

```python
        self._install_molecule_index_arrays(context.getSystem().getNumParticles())
```

- [ ] **Step 4: Replace the hot loop**

In `attempt_due`, replace lines 519-523 exactly:

```python
        new_positions = positions.copy()
        scale_minus_one = s - 1.0
        for mol in self._molecules:
            center = positions[mol].mean(axis=0)
            new_positions[mol] = positions[mol] + scale_minus_one * center
```

with:

```python
        scale_minus_one = s - 1.0
        new_positions = _scale_about_molecule_centroids(
            positions, self._mol_ids, self._mol_sizes, scale_minus_one,
            self._large_molecules,
        )
```

The `positions.copy()` is deliberately dropped — `_scale_about_molecule_centroids` allocates its own result array, so the copy was dead. `positions` itself is never mutated, which the restore path depends on.

- [ ] **Step 5: Run the npt suite to verify nothing changed**

```bash
opencode run "In /run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler: run exactly the following command and report the result: cd /run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler && python -m pytest tests/test_npt_centroid_scaling.py tests/test_npt_transaction_equivalence.py tests/test_npt_controller_mechanics.py tests/test_npt_acceptance_math.py tests/test_npt_coupled_target.py -q. This is a read-only task - do not edit, commit, or fix anything. Report concisely: pass/fail/error counts, each failing test id with file:line, and total runtime."
```

Expected: all pass. Two independent guards fire here — `test_npt_transaction_equivalence.py` compares the whole transaction against the loop, and `test_npt_controller_mechanics.py` already asserts exact restoration and checkpoint round-trips.

- [ ] **Step 6: Commit**

```bash
cd /run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler
git add gareus/npt.py tests/test_npt_centroid_scaling.py tests/test_npt_transaction_equivalence.py
git commit -m "perf(npt): vectorize the volume move's molecule translation

Replaces the per-molecule Python loop in attempt_due with the precomputed
index arrays and bincount helpers. Same array, same trajectory; the loop was
~77% of production wall time with all four GPUs idle.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01LR21pBRrL23jbmmPFEEeuZ"
```

---

### Task 3: Per-phase timers

Stage 2 is gated on knowing how the remaining barostat cost splits. This is that instrument.

**Files:**
- Modify: `gareus/npt.py` (`_ControllerCore.__init__`, `attempt_due`, `state_dict`)
- Test: `tests/test_npt_controller_timings.py` (create)

**Interfaces:**
- Consumes: `_ControllerCore` from Task 2
- Produces: `_ControllerCore._timings: dict[str, float]`; `state_dict()["timings"]`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_npt_controller_timings.py`:

```python
"""Per-phase timers on the volume move, and their checkpoint compatibility.

Timers are per-process diagnostics: they are written into the checkpoint for
the record but never restored from it, so each job reports its own rates.
"""
import json

import numpy as np
import pytest

openmm = pytest.importorskip("openmm")

from gareus.npt import _CONTROLLER_SCHEMA_VERSION  # noqa: E402


def test_schema_version_is_unchanged_by_this_work():
    """Bumping it would break resume for the live chain at npt.py:730."""
    assert _CONTROLLER_SCHEMA_VERSION == 1


def test_timings_start_at_zero_and_have_the_expected_keys():
    from gareus.npt import _ControllerCore

    core = _ControllerCore.__new__(_ControllerCore)
    _ControllerCore._install_timings(core)
    assert set(core._timings) == {
        "read_s", "scale_s", "restore_s", "evaluate_s", "verify_s", "attempts",
    }
    assert all(v == 0 for v in core._timings.values())


def test_state_dict_carries_timings_and_stays_json_serialisable():
    from test_npt_controller_mechanics import (
        VolumeStubAdapter, _make_controller, _rigid_molecule_system)

    _ctx, ctrl = _make_controller(
        _rigid_molecule_system(), VolumeStubAdapter(), seed=11)
    ctrl.attempt_due(ctrl.next_due_step)

    sd = json.loads(json.dumps(ctrl.state_dict()))
    assert sd["schema_version"] == 1
    assert sd["timings"]["attempts"] == 1
    assert sd["timings"]["scale_s"] >= 0.0


def test_restore_accepts_a_checkpoint_written_before_timings_existed():
    """The live chain's in-flight checkpoints have no timings key."""
    from test_npt_controller_mechanics import (
        VolumeStubAdapter, _make_controller, _rigid_molecule_system)
    from gareus.npt import BiasedMCBarostatController

    ctx, ctrl = _make_controller(
        _rigid_molecule_system(), VolumeStubAdapter(), seed=11)
    ctrl.attempt_due(ctrl.next_due_step)

    sd = json.loads(json.dumps(ctrl.state_dict()))
    sd.pop("timings")  # exactly what an a03f3f7-era checkpoint looks like

    restored = BiasedMCBarostatController.restore(
        ctx, VolumeStubAdapter(), state=sd,
        expected_pressure_bar=0.0, expected_temperature_k=300.0)
    assert restored.next_due_step == ctrl.next_due_step
    assert restored.state_dict()["timings"]["attempts"] == 0


def test_schema_v1_readers_ignore_unknown_optional_keys():
    """Adding a key without a version bump is only legitimate if v1 says this.

    Otherwise the timings addition is an undocumented schema fork. This test is
    the documentation: v1 readers must ignore what they do not recognise, so
    future optional additions have a stated rule to follow.
    """
    from test_npt_controller_mechanics import (
        VolumeStubAdapter, _make_controller, _rigid_molecule_system)
    from gareus.npt import BiasedMCBarostatController

    ctx, ctrl = _make_controller(
        _rigid_molecule_system(), VolumeStubAdapter(), seed=11)
    sd = json.loads(json.dumps(ctrl.state_dict()))
    sd["a_key_from_a_future_version"] = {"nested": [1, 2, 3]}

    restored = BiasedMCBarostatController.restore(
        ctx, VolumeStubAdapter(), state=sd,
        expected_pressure_bar=0.0, expected_temperature_k=300.0)
    assert restored.next_due_step == ctrl.next_due_step
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
opencode run "In /run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler: run exactly the following command and report the result: cd /run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler && python -m pytest tests/test_npt_controller_timings.py -q. This is a read-only task - do not edit, commit, or fix anything. Report concisely: pass/fail/error counts, each failing test id with file:line, and total runtime."
```

Expected: FAIL, `AttributeError: ... has no attribute '_install_timings'`.

- [ ] **Step 3: Add the timers**

At the top of `gareus/npt.py`, ensure `import time` is present (add it to the existing stdlib imports if absent).

At `_CONTROLLER_SCHEMA_VERSION` (`gareus/npt.py:215`), add the rule that makes an unversioned optional key legitimate:

```python
# Schema v1 readers MUST ignore keys they do not recognise. `restore` reads
# named keys explicitly and never enumerates the mapping, so a writer may add
# optional diagnostic keys -- "timings" is the first -- without a version bump
# and without breaking a resume in either direction. Anything a reader must
# *act* on still requires a bump.
_CONTROLLER_SCHEMA_VERSION = 1
```

In `_ControllerCore`, add after `_install_molecule_index_arrays`:

```python
    def _install_timings(self) -> None:
        """Per-phase wall-time accumulators for one attempt's transaction.

        Diagnostics only: perf_counter costs tens of nanoseconds against phases
        measured in milliseconds, so the instrument does not perturb what it
        measures. Never restored from a checkpoint -- each job reports its own.
        """
        self._timings = {
            "read_s": 0.0, "scale_s": 0.0, "restore_s": 0.0,
            "evaluate_s": 0.0, "verify_s": 0.0, "attempts": 0,
        }
```

In `__init__`, after the `_install_molecule_index_arrays(...)` line added in Task 2:

```python
        self._install_timings()
```

In `attempt_due`, wrap the phases. The existing line 472 `old = self._adapter.evaluate(self._context, snapshot)` becomes:

```python
            _t0 = time.perf_counter()
            old = self._adapter.evaluate(self._context, snapshot)
            self._timings["evaluate_s"] += time.perf_counter() - _t0
```

Line 484 `positions, box = _read_positions_and_box(self._context)` becomes:

```python
        _t0 = time.perf_counter()
        positions, box = _read_positions_and_box(self._context)
        self._timings["read_s"] += time.perf_counter() - _t0
        self._timings["attempts"] += 1
```

The Task 2 translation block becomes:

```python
        _t0 = time.perf_counter()
        scale_minus_one = s - 1.0
        new_positions = _scale_about_molecule_centroids(
            positions, self._mol_ids, self._mol_sizes, scale_minus_one,
            self._large_molecules,
        )
        self._timings["scale_s"] += time.perf_counter() - _t0
```

Wrap each `self._restore_positions(...)` call site (lines 526, 530, 546, 555 in the pre-edit file) as:

```python
            _t0 = time.perf_counter()
            self._restore_positions(<existing args>)
            self._timings["restore_s"] += time.perf_counter() - _t0
```

and each `self._verify_restoration(...)` call site (lines 531, 547, 556) as:

```python
            _t0 = time.perf_counter()
            self._verify_restoration(<existing args>)
            self._timings["verify_s"] += time.perf_counter() - _t0
```

and the second `self._adapter.evaluate(...)` at line 528 the same way as the first, into `evaluate_s`.

In `state_dict`, add one entry to the returned dict, after `"counters": dict(self._counters),`:

```python
            "timings": dict(self._timings),
```

Do **not** touch `BiasedMCBarostatController.restore` (`gareus/npt.py:702`). Timings are written, never read back — a per-job reset is what makes them useful.

- [ ] **Step 4: Run the tests to verify they pass**

```bash
opencode run "In /run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler: run exactly the following command and report the result: cd /run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler && python -m pytest tests/test_npt_controller_timings.py tests/test_npt_controller_mechanics.py -q. This is a read-only task - do not edit, commit, or fix anything. Report concisely: pass/fail/error counts, each failing test id with file:line, and total runtime."
```

Expected: all pass. `test_npt_controller_mechanics.py` contains checkpoint round-trip tests; an extra key must not break them.

- [ ] **Step 5: Commit**

```bash
cd /run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler
git add gareus/npt.py tests/test_npt_controller_timings.py
git commit -m "feat(npt): record per-phase timings for the volume move

Accumulates read/scale/restore/evaluate/verify wall time per controller and
writes it into the checkpoint as an optional key. The schema version is
deliberately unchanged so a resume of an existing checkpoint still loads.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01LR21pBRrL23jbmmPFEEeuZ"
```

---

### Task 4: Fix the throughput meter

**Files:**
- Modify: `gareus/progress.py:69-90` (`GuiProgressSink.__init__`), `gareus/progress.py:145-179` (`report`)
- Test: `tests/test_progress_rates.py` (create)

**Interfaces:**
- Consumes: nothing
- Produces: `GuiProgressSink.baseline_step: dict[str, int]`; payload key `segment_steps`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_progress_rates.py`:

```python
"""Rates must describe this process's segment, not the whole campaign.

step is cumulative across a resumed chain; elapsed restarts at zero each job.
Dividing one by the other reported 5,293 ns/day against an actual ~283 on
chignolin_7 job 2411974 -- high by a factor of ~19.
"""
import types

import pytest

from gareus.progress import GuiProgressSink


def _sink(tmp_path):
    args = types.SimpleNamespace(progress_mode="console", tui_mode="plain")
    return GuiProgressSink(tmp_path, args)


def test_rates_use_only_steps_since_the_segment_started(tmp_path):
    sink = _sink(tmp_path)
    captured = []
    sink._emit_json = lambda payload: captured.append(payload)  # type: ignore[attr-defined]

    sink.report("prod", step=1_000_000, total_steps=2_000_000,
                timestep_fs=3.5, n_replicas=64, force=True)
    sink.report("prod", step=1_010_000, total_steps=2_000_000,
                timestep_fs=3.5, n_replicas=64, force=True)

    last = captured[-1]
    assert last["segment_steps"] == 10_000
    assert last["sim_time_ns"] == pytest.approx(1_010_000 * 3.5 / 1e6)
    assert last["steps_per_s"] < 10_000_000


def test_first_report_of_a_segment_omits_rates_rather_than_dividing_by_zero(tmp_path):
    sink = _sink(tmp_path)
    captured = []
    sink._emit_json = lambda payload: captured.append(payload)  # type: ignore[attr-defined]

    sink.report("prod", step=500_000, total_steps=2_000_000,
                timestep_fs=3.5, n_replicas=64, force=True)

    first = captured[0]
    assert first["segment_steps"] == 0
    assert "ns_per_day" not in first
    assert "steps_per_s" not in first


def test_cumulative_simulated_time_is_still_cumulative(tmp_path):
    sink = _sink(tmp_path)
    captured = []
    sink._emit_json = lambda payload: captured.append(payload)  # type: ignore[attr-defined]

    sink.report("prod", step=900_000, total_steps=2_000_000,
                timestep_fs=3.5, n_replicas=64, force=True)

    p = captured[0]
    assert p["sim_time_ns"] == pytest.approx(900_000 * 3.5 / 1e6)
    assert p["aggregate_sim_time_ns"] == pytest.approx(900_000 * 3.5 / 1e6 * 64)
```

> Before writing the implementation, open `gareus/progress.py` and find the method that writes a payload to the JSONL handle. If it is not named `_emit_json`, use its real name in the three monkeypatch lines above. Do not invent a method.

- [ ] **Step 2: Run the tests to verify they fail**

```bash
opencode run "In /run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler: run exactly the following command and report the result: cd /run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler && python -m pytest tests/test_progress_rates.py -q. This is a read-only task - do not edit, commit, or fix anything. Report concisely: pass/fail/error counts, each failing test id with file:line, and total runtime."
```

Expected: FAIL — `KeyError: 'segment_steps'`, and `steps_per_s` present on the first report.

- [ ] **Step 3: Write the implementation**

In `GuiProgressSink.__init__`, after `self.phase_start: dict[str, float] = {}` (line 81):

```python
        # First step seen in this process for each phase. step is cumulative
        # across a resumed chain while phase_start restarts every job, so any
        # rate must be computed from the difference, not from step itself.
        self.baseline_step: dict[str, int] = {}
```

In `report`, after `self.phase_start.setdefault(phase, now)` (line 147):

```python
        self.baseline_step.setdefault(phase, step_int)
        segment_steps = step_int - self.baseline_step[phase]
```

Replace the ETA computation on line 153:

```python
        eta: Optional[float] = (elapsed * (1.0 - frac) / frac) if frac > 0.0 and total > 0 else None
```

with:

```python
        eta: Optional[float] = None
        if segment_steps > 0 and elapsed > 0.0 and total > step_int:
            eta = (total - step_int) * elapsed / segment_steps
```

Add `segment_steps` to the payload dict, next to `"step": step_int,`:

```python
            "segment_steps": segment_steps,
```

Replace the rate block (lines 173-179):

```python
            if elapsed > 0 and sim_time_ns > 0:
                payload["ns_per_day"] = sim_time_ns / elapsed * 86400.0
                payload["aggregate_ns_per_day"] = aggregate_ns / elapsed * 86400.0
                payload["wall_s_per_ns"] = elapsed / sim_time_ns
                payload["wall_h_per_us"] = elapsed / sim_time_ns * 1000.0 / 3600.0
                payload["wall_ms_per_step"] = elapsed / max(1, step_int) * 1000.0
                payload["steps_per_s"] = step_int / elapsed
```

with:

```python
            if elapsed > 0 and segment_steps > 0:
                segment_ns = segment_steps * float(timestep_fs) / 1.0e6
                payload["ns_per_day"] = segment_ns / elapsed * 86400.0
                payload["aggregate_ns_per_day"] = (
                    segment_ns * max(1, int(n_replicas)) / elapsed * 86400.0
                )
                payload["wall_s_per_ns"] = elapsed / segment_ns
                payload["wall_h_per_us"] = elapsed / segment_ns * 1000.0 / 3600.0
                payload["wall_ms_per_step"] = elapsed / segment_steps * 1000.0
                payload["steps_per_s"] = segment_steps / elapsed
```

`sim_time_ns` and `aggregate_sim_time_ns` above this block are left alone — they are totals and are correct.

- [ ] **Step 4: Run the tests to verify they pass**

```bash
opencode run "In /run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler: run exactly the following command and report the result: cd /run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler && python -m pytest tests/test_progress_rates.py -q. This is a read-only task - do not edit, commit, or fix anything. Report concisely: pass/fail/error counts, each failing test id with file:line, and total runtime."
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
cd /run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler
git add gareus/progress.py tests/test_progress_rates.py
git commit -m "fix(progress): derive throughput rates from the current segment

step is cumulative across a resumed chain while elapsed restarts each job, so
every rate read high by the ratio of campaign age to job age -- 5,293 ns/day
reported against ~283 actual on chignolin_7. Rates now use steps since this
process started; cumulative totals are unchanged.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01LR21pBRrL23jbmmPFEEeuZ"
```

---

### Task 5: Production-shaped A/B harness — measure p before deploying

The unit tests prove the transform is equal and fast **single-threaded on toy
systems**. They do not establish what the campaign will experience: `p` (how
much of barostat time is the loop) is unmeasured, and 64 Python threads
contending for the GIL is the regime that actually matters. Deploying on the
strength of a 182x single-call ratio would be quoting the wrong number.

This harness settles it on a real GPU Context at production size, without
touching the campaign.

**Files:**
- Create: `tools/npt_ab_harness.py`
- Test: none (this is an instrument, not a unit test)

**Interfaces:**
- Consumes: `BiasedMCBarostatController`, `_scale_about_molecule_centroids`
- Produces: a phase breakdown (`p`) and old-vs-new throughput at 1 and 64 threads

- [ ] **Step 1: Write the harness**

Create `tools/npt_ab_harness.py`:

```python
"""Old-vs-new volume move on a production-sized Context, at 1 and 64 threads.

Answers the two questions the unit tests cannot: what fraction of barostat wall
time is the molecule loop (p), and whether the single-threaded 182x transform
speedup survives 64 workers contending for the GIL.

Run on a GPU node:  python tools/npt_ab_harness.py --threads 1 8 32 64
"""
import argparse
import statistics
import sys
import threading
import time

import numpy as np
import openmm
import openmm.app as app
import openmm.unit as unit

import gareus.npt as npt_mod
from gareus.npt import BiasedMCBarostatController, EnergyBreakdown


class _RealEnergyAdapter:
    """Forces a REAL potential-energy evaluation, like production's U* does.

    An analytic volume-only adapter would make `evaluate` free, and `evaluate`
    is one of the phases competing with the loop for the barostat's wall time.
    Measuring p with a free `evaluate` biases p upward and could pass the gate
    while production delivers far less. So this calls getState(getEnergy=True),
    which triggers the same PME force evaluation the real adapter pays for.

    It does not reproduce Pep-GaMD's boost channel -- that needs the full
    integrator -- so measured p remains a mild OVER-estimate. Stated rather than
    hidden: the true production p is at or below what this reports.
    """
    adapter_id = "ab-harness-real-energy"

    def snapshot(self, context, integrator):
        return {}

    def evaluate(self, context, snapshot):
        st = context.getState(getEnergy=True)
        phys = float(st.getPotentialEnergy().value_in_unit(
            unit.kilojoule_per_mole))
        return EnergyBreakdown(phys, 0.0, 0.0, 0.0, phys)


def _loop_reference_for(molecules):
    """The replaced implementation, iterating the ORIGINAL molecule lists.

    np.flatnonzero(mol_ids == m) would return ascending atom indices, which is
    not the order the original loop used; for a non-contiguous molecule that is
    a different summation order and therefore a different reference.
    """
    def impl(positions, mol_ids, mol_sizes, scale_minus_one, large=()):
        new_positions = positions.copy()
        for mol in molecules:
            new_positions[mol] = positions[mol] + scale_minus_one * positions[mol].mean(axis=0)
        return new_positions
    return impl


def build_context(device_index, platform_name="CUDA"):
    """~19k atoms of water: the production system's size and molecule count.

    device_index pins the Context to one GPU so the harness can reproduce
    production's topology -- 4 GPUs x 16 replicas -- rather than piling 64
    contexts onto one card, which would contend for memory bandwidth in a way
    production does not and depress the measured speedup.
    """
    ff = app.ForceField("amber14/tip3pfb.xml")
    modeller = app.Modeller(app.Topology(), [])
    modeller.addSolvent(ff, boxSize=openmm.Vec3(6.0, 6.0, 6.0) * unit.nanometer)
    system = ff.createSystem(modeller.topology, nonbondedMethod=app.PME,
                             nonbondedCutoff=1.0 * unit.nanometer,
                             constraints=app.HBonds, rigidWater=True)
    integ = openmm.LangevinMiddleIntegrator(
        300 * unit.kelvin, 1 / unit.picosecond, 0.002 * unit.picoseconds)
    platform = openmm.Platform.getPlatformByName(platform_name)
    props = {"Precision": "mixed", "DeviceIndex": str(device_index)} \
        if platform_name == "CUDA" else {}
    ctx = openmm.Context(system, integ, platform, props)
    ctx.setPositions(modeller.positions)
    return ctx, system


def time_moves(ctrl, n):
    step = ctrl.next_due_step
    t0 = time.perf_counter()
    for _ in range(n):
        ctrl.attempt_due(step)
        step = ctrl.next_due_step
    return time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 8, 32, 64])
    ap.add_argument("--moves", type=int, default=40)
    ap.add_argument("--platform", default="CUDA")
    ap.add_argument("--gpus", type=int, default=4,
                    help="spread contexts across this many GPUs, as production does")
    args = ap.parse_args()

    ctx0, system = build_context(0, args.platform)
    n_atoms = system.getNumParticles()
    n_mol = len(list(ctx0.getMolecules()))
    molecules = [[int(i) for i in mol] for mol in ctx0.getMolecules()]
    print(f"system: {n_atoms} atoms, {n_mol} molecules, "
          f"platform {args.platform}, {args.gpus} GPU(s)")

    for n_threads in args.threads:
        for label, impl in (("loop", _loop_reference_for(molecules)),
                            ("vectorized", npt_mod._scale_about_molecule_centroids)):
            original = npt_mod._scale_about_molecule_centroids
            npt_mod._scale_about_molecule_centroids = impl
            try:
                ctrls = []
                for i in range(n_threads):
                    c, _ = build_context(i % max(1, args.gpus), args.platform) \
                        if i else (ctx0, system)
                    ctrls.append(BiasedMCBarostatController.initialize(
                        c, _RealEnergyAdapter(), pressure_bar=1.0,
                        temperature_k=300.0, frequency_steps=1,
                        volume_step_fraction=0.01, seed=1234 + i))
                times = [None] * n_threads
                def run(i):
                    times[i] = time_moves(ctrls[i], args.moves)
                ts = [threading.Thread(target=run, args=(i,)) for i in range(n_threads)]
                t0 = time.perf_counter()
                for t in ts: t.start()
                for t in ts: t.join()
                wall = time.perf_counter() - t0
                per_move_ms = wall / args.moves * 1000.0
                print(f"  threads={n_threads:3d} {label:11s} "
                      f"wall {wall:7.3f}s  {per_move_ms:8.3f} ms/move/replica-set")
                tim = ctrls[0].state_dict().get("timings")
                if tim and label == "vectorized":
                    total = sum(v for k, v in tim.items() if k.endswith("_s"))
                    if total > 0:
                        print("    phases: " + "  ".join(
                            f"{k}={100.0 * v / total:5.1f}%"
                            for k, v in tim.items() if k.endswith("_s")))
            finally:
                npt_mod._scale_about_molecule_centroids = original


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run it on a GPU node**

```bash
scp tools/npt_ab_harness.py aurum2:/home/sulcjo/2026_peptide_sampler/tools/
ssh aurum2 'srun -p d192_384_gpu --gres=gpu:4 -t 00:40:00 --pty bash -c "cd /home/sulcjo/2026_peptide_sampler && python tools/npt_ab_harness.py --threads 1 8 32 64 --gpus 4"'
```

Do **not** run this on the node the campaign occupies. Request a separate allocation.

- [ ] **Step 3: Record the answers**

From the output, write down:

| quantity | where it comes from | decides |
| --- | --- | --- |
| `p` | `scale_s` as a share of the summed `*_s` phases, **loop build** | which row of the spec §2 table applies |
| single-thread ratio | loop vs vectorized ms/move at `threads=1` | whether 182x reproduces on a real Context |
| 64-thread ratio | loop vs vectorized ms/move at `threads=64` | the number that actually predicts campaign speedup |
| scaling shape | ms/move across 1 → 64 | how much of the cost was GIL convoy rather than arithmetic |

**Gate — downgraded to a measurement by explicit decision (2026-09-15).** The
review board's majority made deployment conditional on this number clearing
~1.5x, and on the current best estimate (p ≈ 0.42 → ~1.47x) it may not. The
project owner was told this and elected to proceed regardless.

So: **record the number, report it, and continue to Task 6.** Do not treat a
sub-1.5x result as a stop. The justification for proceeding anyway is that the
change is bit-identical — a disappointing speedup costs a deploy cycle, not a
campaign — and that the timers shipped alongside it are what make stage 2
designable. If the measured ratio is below ~1.2x, say so prominently in the
report; that would mean the loop is not the bottleneck and stage 2's target is
somewhere else entirely.

---

### Task 6: Full suite, deployment, and validation

**Files:**
- Create: `tests/test_npt_centroid_perf.py`
- Modify: none

**Interfaces:**
- Consumes: everything from Tasks 1-4
- Produces: a deployable commit

- [ ] **Step 1: Write the performance guard**

Create `tests/test_npt_centroid_perf.py`:

```python
"""Guard against the per-molecule loop silently returning.

Not a benchmark -- a regression tripwire with a wide margin, so it does not
flake on a loaded machine.
"""
import time

import numpy as np

from gareus.npt import _molecule_index_arrays, _scale_about_molecule_centroids


def test_vectorized_scaling_is_far_faster_than_a_python_loop():
    n_mol = 6000
    molecules = [[3 * m, 3 * m + 1, 3 * m + 2] for m in range(n_mol)]
    n_atoms = 3 * n_mol
    rng = np.random.default_rng(0)
    positions = rng.random((n_atoms, 3)) * 6.0
    mol_ids, mol_sizes = _molecule_index_arrays(molecules, n_atoms)

    def loop():
        new = positions.copy()
        for mol in molecules:
            new[mol] = positions[mol] + 0.0003 * positions[mol].mean(axis=0)
        return new

    def vec():
        return _scale_about_molecule_centroids(positions, mol_ids, mol_sizes, 0.0003)

    assert np.array_equal(loop(), vec())

    loop(); vec()
    t0 = time.perf_counter()
    for _ in range(3):
        loop()
    loop_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(3):
        vec()
    vec_s = time.perf_counter() - t0

    assert vec_s < 0.05 * loop_s, f"vectorized {vec_s:.4f}s vs loop {loop_s:.4f}s"
```

- [ ] **Step 2: Run the whole suite**

```bash
opencode run "In /run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler: run exactly the following command and report the result: cd /run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler && python -m pytest -q. This is a read-only task - do not edit, commit, or fix anything. Report concisely: pass/fail/error counts, each failing test id with file:line, and total runtime."
```

Expected: the Task 0 baseline failure set **exactly** — same test ids, no additions, none missing. Roughly 50 new tests pass on top of the baseline (39 centroid, 5 transaction, 5 timings, 3 progress, 1 perf), so ~3012 passed against ~2959 at baseline.

Compare the **set of failing test ids**, not the counts. A matching count with a different id is a regression wearing a disguise. Any new failure blocks deployment.

- [ ] **Step 3: Commit**

```bash
cd /run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler
git add tests/test_npt_centroid_perf.py
git commit -m "test(npt): guard the vectorized volume move against regression

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01LR21pBRrL23jbmmPFEEeuZ"
```

- [ ] **Step 4: Deploy to aurum2**

Never `rsync ./` — the untracked `.claude/` is ~1.3 GB.

**Extract to a staging tree and swap atomically.** Untarring directly over the
live `PYTHONPATH` tree risks a resubmit landing mid-extraction and importing a
half-written package — a failure with no good diagnostic. A rename within one
filesystem is atomic; use that.

```bash
cd /run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler
SHA=$(git rev-parse HEAD)
git archive HEAD --format=tar -o /tmp/gareus_deploy.tar
scp /tmp/gareus_deploy.tar aurum2:/tmp/gareus_deploy.tar

ssh aurum2 "set -e
  STAGE=/home/sulcjo/2026_peptide_sampler.staged_$SHA
  rm -rf \$STAGE && mkdir -p \$STAGE
  tar xf /tmp/gareus_deploy.tar -C \$STAGE
  echo $SHA > \$STAGE/DEPLOYED_COMMIT
  python -c 'import sys; sys.path.insert(0, \"'\$STAGE'\"); import gareus.npt' || {
      echo 'staged tree does not import; aborting'; exit 1; }
  mv /home/sulcjo/2026_peptide_sampler /home/sulcjo/2026_peptide_sampler.prev_$SHA
  mv \$STAGE /home/sulcjo/2026_peptide_sampler
  rm -f /tmp/gareus_deploy.tar
  cat /home/sulcjo/2026_peptide_sampler/DEPLOYED_COMMIT"
```

The two `mv`s are the only window, and they are metadata operations. The staged
tree is import-checked before the swap, so a broken archive never becomes live.
The previous tree is kept as `.prev_<sha>` — that is the rollback.

Best sequenced **immediately after an observed resubmit**, which leaves ~3h45m
of clear air before the next one:

```bash
ssh aurum2 'squeue -u $USER -h -n chignolin_7 -o "%i %M"'   # want a small elapsed time
```

The live chain re-imports on every resubmit (~3h51m), so this lands with no restart.

- [ ] **Step 5: Validate on the first resumed job**

Wait for the chain to resubmit, then:

```bash
ssh aurum2 'squeue -u $USER -h -n chignolin_7 -o "%i %N"'
```

With `<JOBID>` and `<NODE>` from that, check each row:

| check | command | expectation |
| --- | --- | --- |
| resumed cleanly | `ssh aurum2 'grep -a "Resumed GaREUS production" ~/gareus/chignolin/chignolin_7_<JOBID>.log'` | a step continuing the chain |
| GPUs busy | `ssh aurum2 'ssh <NODE> "nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader"'` | clearly non-zero, was 0% |
| loop gone | `ssh aurum2 'ssh <NODE> "/home/sulcjo/pyspy_tmp/bin/py-spy dump --pid \$(pgrep -f \"python -m gareus\") --nonblocking"'` | no `npt.py` frame in the centroid translation |
| acceptance | `ssh aurum2 'tail -c 300000 ~/gareus/chignolin/chignolin_7/progress.jsonl \| grep -o "\"accepted\": [0-9]*" \| tail -1'` | same ~22% band |
| stride | compare successive `replica_000_resume_from_*.xtc` names | ~191k → 237k-816k steps per job, ~281k at the expected p≈0.42 |

If any check fails, roll back: `git revert` the four commits, redeploy by the same archive route. Because the change is bit-identical, neither deploying nor reverting has any consequence for data already collected.

- [ ] **Step 6: Clear the two environment hazards (ask first — one is destructive)**

Both are recorded in spec §10. The rename is reversible; **confirm with the user
before running either**, since both touch their home directory.

```bash
ssh aurum2 'ls -d ~/gareus/gareus && mv ~/gareus/gareus ~/gareus/gareus.stale_20260915 && echo renamed'
ssh aurum2 'ls -l /tmp/inspect.py'   # inspect before removing; it is not ours
```

The first removes a divergent copy of the package that is *not* what runs
(`PYTHONPATH=/home/sulcjo/2026_peptide_sampler`) but parses cleanly and yields
confident wrong line numbers. The second shadows the stdlib `inspect` for any
Python process whose `sys.path[0]` is `/tmp`; the production job passes
`--platform-temp-directory /tmp`. Leave `/tmp/inspect.py` alone unless the user
confirms it is disposable.

- [ ] **Step 7: Report the stage-2 gate measurement**

Read `timings` out of the checkpoint manifest:

```bash
ssh aurum2 'python3 -c "
import json
p=\"/home/sulcjo/gareus/chignolin/chignolin_7/adaptive_production/epoch_000/checkpoints/production_checkpoint_manifest.json\"
d=json.load(open(p))
print(json.dumps(d.get(\"barostat\",{}).get(\"timings\",\"not found\"), indent=2))
"'
```

Report `scale_s`, `read_s`, `restore_s`, `evaluate_s`, `verify_s` as fractions of their sum. That breakdown decides whether stage 2 is needed to reach the ~10% overhead target, and which phase it must attack.

---

## Notes for the executor

- The spec this implements is `docs/superpowers/specs/2026-09-15-npt-barostat-perf-design.md`. Read it before Task 1; it explains why the barostat cannot simply be replaced by OpenMM's native one.
- Line numbers in this plan refer to `gareus/npt.py` and `gareus/progress.py` **as of commit `a03f3f7`**. They shift as you edit. Anchor on the surrounding code shown in each step, not on the number.
- If a step's described code does not match what you find, stop and report rather than improvising. A mismatch means the file moved under the plan.
