# Negative result: the barostat centroid transform cannot be usefully optimised

**Status: DO NOT MERGE this branch as an optimisation.** The implementation on
`perf/centroid-run-plan` is correct and bit-identical, and it is *slower* than
what it replaces at production thread counts. Kept only so the measurement is
not repeated.

Date 2026-09-17. Measured on aurum2 (d094/d095, L40S nodes, 48 CPUs in the
affinity mask), against `main` at 7119df1.

## What was proposed

`_scale_about_molecule_centroids` cost 29.25 ms/attempt in the live campaign
(32 replicas) = 0.146 ms/step = **5.0% of wall**. It used `np.bincount` (once
per axis) plus a fancy-index gather, both of which hold the GIL for their whole
duration. The proposal was to release the GIL, either with a compiled kernel
(numba/Cython/C) or a pure-numpy reformulation.

## What was built

A slice plan: maximal runs of equal-size, contiguous, ascending molecules
(`_uniform_run_plan`), summed with sequential `acc += blk[:, j, :]` and expanded
with a broadcast assignment. No `bincount`, no fancy index, no `np.repeat`.
Verified bit-identical with `==` over a solvated layout and 12 randomised
variants, with the `>= 128 atoms OR non-ascending -> .mean()` fallback preserved
and the whole plan refused unless it covers every molecule and atom.

## Why it must not ship

Production duty cycle is one transform per 200 MD steps at ~2.93 ms/step, i.e.
~586 ms apart, with threads otherwise inside `integrator.step()` holding no GIL.
Probe: each thread waits (GIL released) then runs one transform.

```
wait_ms  thr    old ms    new ms  speedup
6.0      32     0.9546    2.6630    0.36x
50.0     32     1.5433    2.3033    0.67x
200.0    32     2.3464    2.6470    0.89x
600.0    32     2.9410    3.1815    0.92x   <- production cadence
```

At 32 threads the rewrite is never faster. `bincount` is ONE C call that holds
the GIL briefly; the reformulation is ~15 numpy calls, each acquiring and
releasing the GIL, and each release is an opportunity for a context switch under
contention. Fewer GIL-held operations, but far more GIL transitions, and the
second effect dominates.

## The premise was wrong, three times over

1. **"94% GIL-serialized" was a misattribution.** It was 1 - parallel
   efficiency, not a serial fraction. From 1.84x aggregate at 64 workers Amdahl
   gives a serial fraction of ~0.54.
2. **A direct discriminator refuted the GIL diagnosis.** Operations whose GIL
   behaviour is known, timed at 1/8/32 threads:
   `pure_ufunc` (GIL released) inflated 17.2x, `inplace` (released, no
   allocation) 41.3x, `bincount` (held) 33.6x, a pure Python loop (held) 17.6x.
   GIL-releasing operations inflate as badly as GIL-held ones; at 32 threads the
   released ones sit at ~141 GB/s aggregate, i.e. memory-bandwidth saturation.
3. **The saturated probe showed 0.97x** for the GIL-free reformulation before
   the duty-cycle probe showed it was outright worse.

## The finding that actually matters

At production cadence the transform itself costs **2.94 ms**, while the live
campaign records **29.25 ms** for the same call. The 10x gap is GIL *wait*
caused by other threads' Python work -- logging, CV sampling, exchange,
reporters -- not by this function's own arithmetic.

So the 5%-of-wall "scale" cost is **not addressable by optimising this
function**. A hypothetical zero-cost transform recovers at most ~10% of it,
about 0.5% of wall. The addressable target is total Python/GIL pressure in the
process, i.e. process-per-replica, which the board classified tier (b) and
next-campaign work.

## Board input

The small board (kimi / mini / thinker, glm chairing) split 1/1/1 on labels but
both final positions converged on this pure-numpy reformulation, and rejected
the compiled options as "strictly dominated on risk" for ~5% of wall. Their
reasoning about numpy's GIL behaviour was locally correct -- the reduction and
broadcast do release the GIL where `bincount` and the gather do not -- but
neither modelled the cost of multiplying GIL transitions. The board's
condition 5 independently caught the 94% error.

Transcript: `~/.claude/jobs/ff96fa1f/tmp/sboard_out/`.
