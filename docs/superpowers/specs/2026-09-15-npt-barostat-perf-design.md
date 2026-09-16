# ATLaS-MD: NPT barostat performance, stage 1

Status: proposed design for review; no implementation changes made.

Scope: remove the dominant cost of the application-controlled biased-MC volume
move, without altering the trajectory it produces. This is an engineering
change inside the contract frozen by
[2026-09-12-npt-correction-design.md](./2026-09-12-npt-correction-design.md).
No physics, ensemble, or acceptance-rule decision in that document is reopened.

Measurements below come from the live `chignolin_7` campaign on aurum2
(19,008 atoms, 6,303 molecules, 64 replicas, four L40S, jobs 2394315–2411974).

## 1. Problem

Introducing NPT made production roughly 5–8x slower than the same run without
it. The slowdown is not the volume move's physics; it is one Python loop.

Two independent measurements agree on the size:

- **End-to-end.** Changing `barostat_frequency` from 100 to 200 steps — halving
  the number of attempts and changing nothing else — gave a 1.77x speedup
  (checkpoint-to-checkpoint stride 108,000 → 191,250 steps per 3h51m job).
  A halving that buys 1.77x implies the barostat was consuming **~87% of wall
  time** at frequency 100, and consumes **~77%** at frequency 200.
- **Profile.** `py-spy` against job 2411974 (PID 760422, node d093), 10 dumps.
  Read the denominator carefully: the dumps contain 2,580 stack *frames*, but a
  worker's stack is about four frames deep
  (`_step_item → advance → attempt_due → attempt_due`), so they represent **631
  worker-thread observations**. Of those, **629 were innermost in
  `npt.py:521-523`** and **2 in `integrator.step()`**. All four GPUs read 0%
  utilisation; load average 7.4 on a 192-core exclusive node.

  Dividing 629 by 2,580 yields a spurious 24% and is a frame-versus-observation
  error; it is called out here because it is an easy one to make and it changes
  the conclusion by a factor of four.

The profile is nonetheless biased upward — threads blocked on the GIL are
sampled where they last held it, which flatters pure-Python frames against the
GPU calls that release it. The end-to-end figure does not have that weakness, so
**~77% is the number this design is built on.** What the profile cannot settle
is how that 77% divides, which §2 treats as the open question it is.

### The hot loop

`gareus/npt.py`, inside `_ControllerCore.attempt_due`:

```python
519:  new_positions = positions.copy()
520:  scale_minus_one = s - 1.0
521:  for mol in self._molecules:
522:      center = positions[mol].mean(axis=0)
523:      new_positions[mol] = positions[mol] + scale_minus_one * center
```

One Python iteration per molecule — 6,303 for this system — each doing a fancy
-index gather, a `.mean`, and a fancy-index scatter. Roughly 19,000 small numpy
calls per volume move, all holding the GIL, on all 64 replica worker threads at
once.

Measured in isolation on a reconstructed partition matching the production
system's size and composition — 19,008 atoms, 6,303 molecules, one 174-atom
solute, 36 single-atom ions, the rest 3-site waters. This is the production
system's *shape*, not a dump of its actual partition:

| form | ms per attempt |
| --- | --- |
| current Python loop | 57.114 |
| vectorized (`bincount`) | 0.314 |

182x, with output **bit-identical** (`maxdiff 0.0`).

### What is already optimised, and is not the problem

`a03f3f7` made `_verify_restoration` strided (`_RESTORE_VERIFY_STRIDE`) and
stopped re-reading coordinates per move. That readback was the largest frame in
the pre-fix profile; it does not appear in the current one. There is nothing
further to win there, and the guard's stride must not be widened — see §7.

## 2. Decision

Stage 1 is **bit-identical**: every change either provably produces the same
numbers, or touches only measurement. It can therefore be deployed into the
running `chignolin_7` chain at its next resubmit, with no restart, no lost
work, and no effect on results already sampled.

Stage 1 does **not** reach the ~10% overhead target, and its gain is a function
of one unmeasured quantity. Define it explicitly:

> **p** = the fraction of *barostat* wall time spent in the molecule loop.
> Baseline is the **current** configuration — attempt interval 200, barostat
> 77% of wall, MD 23%. Speedup is `1 / (0.23 + 0.77·[(1−p) + p/182])`.

| p | stage-1 speedup | barostat after | stride/job |
| --- | --- | --- | --- |
| 0.25 | **1.24x** | 71.6% | 237k |
| 0.33 | 1.34x | 68.5% | 255k |
| **0.42** | **1.47x** | **64.6%** | **281k** |
| 0.50 | 1.62x | 62.7% | 310k |
| 0.75 | 2.35x | 46.0% | 449k |
| 1.00 | 4.27x | 1.8% | 816k |

The curve is steep and no point on it may be quoted as if measured.

### MEASURED, 2026-09-16: p = 0.99, end-to-end 4.2x

Harness job 2415561 on d099 (4x L40S, MPS, 21,384 atoms / 7,128 molecules —
slightly larger than production, so if anything conservative). Old loop vs
vectorized, both driven through the real controller:

| threads | loop ms/move | vectorized ms/move | ratio | implied p | end-to-end |
| --- | --- | --- | --- | --- | --- |
| 1 | 100.4 | 2.50 | 40.2x | 0.98 | 3.73x |
| 8 | 1067.3 | 9.87 | 108.1x | 0.99 | 4.10x |
| 32 | 7109.9 | 48.36 | 147.0x | 0.99 | 4.16x |
| **64** | **15991.3** | **91.17** | **175.4x** | **0.99** | **4.19x** |

**The cycle-arithmetic estimate below was wrong, and this is why.** It assumed
the isolated 57.1 ms microbenchmark transfers to a contended runtime. It does
not: per replica the loop costs 100 ms at 1 thread but **250 ms at 64** — 4.4x
its isolated cost. That inflation *is* the GIL convoy, and it is what made
cycle arithmetic read p ≈ 0.42 when the true value is ≈ 0.99.

The measurement is robust to the harness's one known bias. Its `evaluate` is a
plain PME energy rather than the full Pep-GaMD U*, so production's is dearer;
but making it 2x/3x/5x dearer moves p only to 0.997/0.997/0.995 and the speedup
to 4.23x/4.22x/4.20x. The loop so dominates that nothing else matters.

**Expected stride is therefore ~816k steps/job, the top of the §2 table, not the
~281k that p ≈ 0.42 predicted.**

Historical note, kept because the reasoning matters more than the answer:

**Two independent estimates of p disagreed, and the disagreement was the whole
problem.**

- **py-spy says p ≈ 1.0.** Of 631 worker-thread stack observations, 629 were
  innermost in the loop. (The denominator is 631 observations, not the 2,580
  total frames — each worker stack contributes about four frames. Dividing by
  2,580 gives a spurious 24%.) This estimate is biased upward: a thread blocked
  on the GIL is sampled where it last held it, which flatters pure-Python frames
  against the GPU calls that release it.
- **Cycle arithmetic says p ≈ 0.42.** At the measured 17.87 steps/s/replica a
  200-step cycle is 11.19 s of wall, while 64 replicas × 57.114 ms of
  GIL-serialised loop is 3.66 s — a 32.7% loop share, hence p = 0.327/0.77 =
  0.42. This assumes the isolated 57.1 ms microbenchmark transfers to a
  64-thread contended runtime, which is exactly what it may not do.

The gap between 0.42 and 1.0 **is** the GIL convoy factor, and nothing currently
in hand measures it. Plan for the low end: at p ≈ 0.42 the change buys ~1.47x,
not the ~4x the profile suggests. **Change 2 exists precisely to measure p**, and
Task 5 of the implementation plan settles it.

Quoting the interval-100 baseline instead (barostat 87%) gives a different and
larger set of numbers — 1.28x (p=0.25), 1.76x (p=0.50), 2.85x (p=0.75),
7.42x (p=1.00). Both are correct arithmetic; only the interval-200 column
describes what this campaign will actually experience. Always state the baseline
with the number.

Closing whatever remains is stage 2, deliberately not designed here.

## 3. Change 1: vectorize the centroid translation

### Precomputation

In `_ControllerCore.__init__`, derive two arrays once from the existing
`self._molecules`:

- `self._mol_ids` — `int32[n_atoms]`, the molecule index owning each atom
- `self._mol_sizes` — `float64[n_mol]`, atom count per molecule

`_molecules_from_context` (`npt.py:271-288`) already rejects any partition that
overlaps or fails to cover every particle, so each atom appears in exactly one
molecule and `bincount` is well defined. **No contiguity assumption is made**:
molecules may be listed in any order and hold non-contiguous indices.

`self._molecules` is retained unchanged — `_molecule_fingerprint` and the
checkpoint `state_dict` both depend on it.

### Hot path

```python
scale_minus_one = s - 1.0
sums = np.empty((self._n_mol, 3), dtype=np.float64)
for k in range(3):
    sums[:, k] = np.bincount(self._mol_ids, weights=positions[:, k],
                             minlength=self._n_mol)
centers = sums / self._mol_sizes[:, None]
for m, mol in self._large_molecules:          # >= 128 atoms; one here
    centers[m] = positions[mol].mean(axis=0)
new_positions = positions + scale_minus_one * centers[self._mol_ids]
```

The three-iteration loop is over spatial axes, not molecules; it is constant
cost. `np.add.reduceat` would be marginally faster but only for atom-contiguous
molecules, which is an assumption this partition does not license.

The original's `positions.copy()` is dropped: the final expression allocates its
own array, and `positions` is left untouched — which the reject path depends on,
since it hands that same array back to `_restore_positions`.

### Why this is bit-identical

Both forms compute, per molecule `m` with atoms `A(m)`:

```
center(m) = (1/|A(m)|) * sum_{i in A(m)} positions[i]
new[i]    = positions[i] + (s-1) * center(m)   for i in A(m)
```

Verified empirically at this system's shape: `maxdiff 0.0`, exact equality
including the 174-atom solute.

### Periodic imaging: checked, and already correct

An arithmetic centroid over a molecule split across the periodic boundary is
meaningless, and would silently produce a wrong translation while every
microbenchmark passed. This project has hit exactly that class of bug before.

It does not apply here. `_read_positions_and_box` (`npt.py:622-627`) calls
`context.getState(getPositions=True)` **without** `enforcePeriodicBox`, which
defaults to `False`, so OpenMM returns unwrapped coordinates and every molecule
arrives whole. (`npt_driver.py:202` does pass `enforcePeriodicBox` — that is the
trajectory reporter, a different path, and does not feed the barostat.)

This is load-bearing and invisible at the call site. The vectorized form computes
the same centroid from the same array, so it neither introduces nor repairs
anything here — but if that `getState` call ever gains `enforcePeriodicBox=True`,
the volume move becomes wrong for both implementations. Noted so it is not
rediscovered the hard way.

**This is empirical, not a numpy guarantee, and the claim must be worded that
way.** `np.bincount` is documented as repeated `out[n] += weight[i]` —
sequential accumulation. `ndarray.mean` delegates to the reduction machinery,
which may use pairwise summation above a blocksize. Nothing in either API
promises they agree. They could diverge on a molecule large enough to trigger
pairwise reduction, under heavy cancellation, in float32, or after a numpy,
SIMD-path or architecture change.

The honest claim is therefore: **bit-identical under the pinned production
environment (numpy 1.24.2, x86-64, float64 coordinates), verified empirically for
molecules below 128 atoms, and identical by construction at or above 128 atoms
via the `ndarray.mean` fallback.** Not bit-identical in general.

Two limits of the evidence, stated rather than glossed:

- **The large-molecule trials are circular.** Sizes 174 / 1,000 / 10,000 /
  100,000 were checked against `ndarray.mean` — but with the fallback in place
  those molecules never reach `bincount`, so the test exercises `mean` against
  itself. The sizes that actually traverse `bincount` are 1 to 127.
- **The 4–127 atom window is the real untested region.** Waters and ions (1 and
  3 atoms) reduce in the same order under both implementations. Nothing in this
  system sits between 4 and 127 atoms, but the code accepts such molecules, so
  §6 tests that window explicitly rather than assuming it.

**Dtype.** `np.bincount` returns float64 regardless of input, so a float32
position array would be silently upcast and break identity. It cannot happen
here: `_read_positions_and_box` (`npt.py:625`) constructs the array with
`dtype=float`, i.e. float64, unconditionally. §6 asserts this rather than
relying on it.

Two things close most of that gap:

1. **Molecules above `_MEAN_FALLBACK_MIN_ATOMS` (128) use `positions[mol].mean(axis=0)`
   — the identical call the old code made**, so for them identity is guaranteed
   rather than observed. This system has exactly one such molecule (the 174-atom
   solute); the fallback loop is over a handful of entries and costs nothing.
   Waters and ions, at 1-3 atoms, are far below any pairwise blocksize and reduce
   sequentially in both implementations.
2. The A/B harness (plan Task 5) compares whole transactions on real production
   state — decisions, energies, box vectors, restored coordinates and RNG state —
   not merely the transformed array.

The concern was probed directly rather than assumed: molecule sizes of 3, 174,
1,000, 10,000 and 100,000 atoms, 200 random trials each, comparing
`bincount`-derived centroids against `ndarray.mean`. **Zero differing trials at
every size**, on both numpy 1.24.2 (the aurum2 production environment) and
numpy 2.3.5. The divergence does not occur in practice at any size this project
will encounter.

The test suite still asserts exact equality at molecule sizes well beyond
anything currently simulated (§6), so a numpy upgrade cannot reintroduce this
silently. If that assertion ever fails, `bincount` is
not a bit-identical substitute for that case and the implementation must match
numpy's summation explicitly rather than relax the assertion to a tolerance —
relaxing it would silently convert this from a bit-identical change into a
statistically-equivalent one, which is a decision reserved for stage 2.

## 4. Change 2: per-phase timers

Add a `self._timings` dict of `time.perf_counter` accumulators to
`_ControllerCore`, covering the phases of one attempt:

| key | spans |
| --- | --- |
| `read_s` | `_read_positions_and_box` |
| `scale_s` | the centroid/translation computation |
| `restore_s` | `_restore_positions` calls, trial and reject |
| `evaluate_s` | `self._adapter.evaluate` calls, old and new |
| `verify_s` | `_verify_restoration`, when the stride fires |
| `attempts` | attempts timed |

Surfaced through `state_dict()` under an optional `"timings"` key.

**`_CONTROLLER_SCHEMA_VERSION` is deliberately not bumped.**
`BiasedMCBarostatController.restore` (`npt.py:702`) raises on any version
mismatch (`npt.py:730`), which would break resume for the live chain. An added
key that old checkpoints simply lack is compatible in both directions: new code
resumes an old checkpoint, and old code ignores an unknown key.

Timings are written into `state_dict` but never read back by `restore` — they
are per-process diagnostics, and a per-job reset is what makes them useful.
`restore` therefore needs no change at all.

**Adding a key without a version bump is only legitimate if v1 is defined to
tolerate unknown optional keys; otherwise this is an undocumented schema fork.**
It is legitimate here — `restore` reads named keys explicitly and ignores
everything else — but that property was implicit. This change makes it explicit:
a comment at `_CONTROLLER_SCHEMA_VERSION` stating that v1 readers must ignore
unknown keys, and a test asserting a state carrying an unrecognised key still
restores. Future optional additions then have a stated rule to follow.

`perf_counter` costs tens of nanoseconds against phases measured in
milliseconds; the instrument does not perturb what it measures.

## 5. Change 3: fix the throughput meter

`gareus/progress.py:168` computes

```python
sim_time_ns = step_int * float(timestep_fs) / 1.0e6
...
payload["ns_per_day"] = sim_time_ns / elapsed * 86400.0
```

`step_int` is the run's **cumulative** step; `elapsed` is **this process's**
elapsed time. After a resume the numerator carries every prior job in the chain
while the denominator restarts at zero, so every rate reads high by the ratio of
campaign age to job age. Observed on 2411974: reported 5,293 ns/day aggregate
against an actual ~283 ns/day, a factor of ~19. The same defect reaches
`steps_per_s` (reported 68.25 = 936250/13718, actual ~15), `wall_s_per_ns`,
`wall_h_per_us`, `wall_ms_per_step` and `eta_s`.

Record `self.baseline_step` when the reporter is constructed. Rate quantities
derive from `step_int - baseline_step`; cumulative quantities
(`sim_time_ns`, `aggregate_sim_time_ns`) keep using `step_int`, which is correct
as a total. Rates are suppressed until at least one step of the new segment has
elapsed.

This is not cosmetic. Stage 2's gate is a measurement, and it cannot be read off
an instrument that is wrong by a factor of 19.

## 6. Testing

Tests are written before the implementation. Equality assertions use `==`, not
`np.allclose` — the claim is bit-identity, and a tolerance would not test it.

**Centroid translation**
- contiguous 3-site waters plus a multi-atom solute — the production shape
- molecules listed out of order, with non-contiguous atom indices
- single-atom molecules (ions) — exercises `size == 1`
- one molecule spanning the whole system
- a single-atom system
- **large-molecule summation order**: molecules of 174, 1,000 and 10,000 atoms,
  asserting exact equality — this is the assertion that would catch `bincount`
  and `ndarray.mean` diverging in the last ulp (§3)
- property test: random valid partitions, bit-equality against the loop

**Transaction**
- `attempt_due` against a fake context returns an identical `VolumeMoveResult`
  *and* leaves identical RNG bit-generator state versus the reference path, for
  an accepted move, a rejected move, and a `cutoff_domain` rejection
- the strided `_verify_restoration` still fires on exactly the same moves

**Checkpoint**
- a `state_dict` written without `timings` loads under the new code
- a `state_dict` written with `timings` loads under code that ignores it
- `_CONTROLLER_SCHEMA_VERSION` is unchanged by this work

**Meter**
- after a simulated resume at step N, `ns_per_day` reflects only steps since N
- `sim_time_ns` still reports cumulative simulated time
- rates are absent, not infinite, on the first report of a segment

**Performance guard**
- vectorized form under 5% of the loop's time at 6,000 molecules, so the
  regression cannot silently return

## 7. Non-goals

- **Widening `_RESTORE_VERIFY_STRIDE`.** The bug it caught (`7900ff0`,
  `db5c7be`) appeared only after accepted moves had accumulated, so any scheme
  that checks early and relaxes later is precisely inverted. Left alone.
- **Reducing `barostat_frequency` further.** That trades volume-relaxation
  quality for speed. This design buys speed without touching sampling.
- **Anything that breaks bit-identity** — batching moves across replicas,
  reordering RNG consumption, deferring restores. Stage 2, separately decided.
- **Reopening the native-barostat question.** The frozen contract establishes
  that OpenMM's `MonteCarloBarostat` accepts on the Context potential, which
  under Pep-GaMD excludes the integrator-applied boost and includes the
  auxiliary water-only force — wrong even at λ=0.

## 8. Deployment and validation

Deploy by `git archive HEAD` → tar → extract **to a staging directory, then
atomically swap**. Rewrite `DEPLOYED_COMMIT` afterwards. Never `rsync ./` — the
untracked `.claude/` is ~1.3 GB.

**The extraction must be atomic with respect to the resubmit.** Untarring
directly over `/home/sulcjo/2026_peptide_sampler` while a job may resubmit
mid-extraction would let a new process import a half-written package — a failure
mode with no good diagnostic. Extract to `2026_peptide_sampler.staged_<sha>`,
then `mv` the live tree aside and `mv` the staged tree into place (a rename is
atomic within a filesystem), or sequence the deploy immediately *after* an
observed resubmit, which leaves ~3h45m of clear air.

The chain re-imports on every resubmit (~3h51m), so the change lands without a
restart and without losing the segment in flight.

Validation on the first resumed job:

| check | expectation |
| --- | --- |
| acceptance ratio | unchanged band, ~22% |
| checkpoint stride | ~191k → **237k–816k** steps per job (the §2 table; ~281k at the expected p ≈ 0.42) |
| py-spy | `npt.py:521-523` absent; `integrator.step()` present |
| GPU utilisation | clearly non-zero on all four |
| `timings` | `scale_s` a small fraction of the attempt total |

Rollback is `git revert` plus redeploy. Because the change is bit-identical,
neither deploying nor reverting has any consequence for data already collected.

## 9. Stage 2 gate

After one full job under the new timers and the corrected meter, the per-phase
breakdown decides whether the ~10% overhead target needs the remaining work —
the three position transfers and the energy evaluations per attempt. That
design is written then, against measurements, not now against estimates.

## 10. Environment hazards observed

Neither is caused by this work; both cost real time during the investigation.

- **`~/gareus/gareus/` on aurum2 is a stale divergent copy of the package.** The
  running job imports from `/home/sulcjo/2026_peptide_sampler` via `PYTHONPATH`.
  The stale tree parses cleanly and has plausible-looking line numbers, so
  reading it yields confident, wrong conclusions — it produced two turns of
  incorrect analysis before being caught. Any note citing a bare `npt.py:NNN` is
  ambiguous until the tree is named. Recommend renaming or deleting it.
- **`/tmp/inspect.py`** (a Schrödinger `StructureReader` snippet) shadows the
  standard library's `inspect` for any Python process whose `sys.path[0]` is
  `/tmp`, failing at `import numpy`. The production job passes
  `--platform-temp-directory /tmp`.
