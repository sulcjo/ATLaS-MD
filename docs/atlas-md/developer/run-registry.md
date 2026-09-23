# Chignolin run registry — what each campaign actually was

Maintained from 2026-09-22. Each run is named for **what it sampled**, which is not always what its
config asked for. Verified facts cite the artefact they were read from; anything not verifiable from
surviving artefacts is marked *(unverified)*.

| Run | What it actually was | Boost as applied | CVs / states | Hardware | Status / usable for |
|---|---|---|---|---|---|
| **chignolin_5** | GaMD-REUS, adaptive production, 2D, with a mid-run CV2 switch | single-channel GaMD, boost applied; type *(unverified, likely `lower-dihedral` like chignolin_6)*; envelope recalibrated once after epoch 0 | CV1 contacts; CV2 torsion-pca in epoch_000, tica-linear from epoch_001 on; ~27-28 states | MPS *(unverified)* | Completed early Aug 2026, ~8.2M samples. Run directory no longer on disk (local or aurum2). Its PMF was wrong until the 2026-08-04 union-MBAR per-epoch-bias fix; any number predating that is invalid |
| **chignolin_6** | GaMD-REUS, double-adaptive, 2D | `lower-dihedral` GaMD, **boost applied** (all checkpoints stage 5, FSF < 1 on every replica); sigma0 5 | CV1 contacts, CV2 torsion-pca; up to 64 replicas; 2 AP epochs + final with top-ups | 4 GPU, MPS | Ran 2026-08-28 to 09-04 (`RUNS/_old/chignolin_6`). ESS 0.74 % was the window-map drop bug (fixed 38af544; data repairable in the loader without new MD) |
| **chignolin_7** | **Pep-GaMD-REUS, 1D** | `pep-gamd-lower-dual` (peptide-only boost), **boost applied** (all checkpoints stage 5, FSF < 1 on 48/64 replicas; the rest sit above threshold, as GaMD allows); no lambda ladder | CV1 contacts only, no CV2; up to 128 replicas; 3 AP epochs + final | 4 GPU, MPS | Completed 2026-09-13 to 09-17 (`RUNS/chignolin_7`). CV1 topology-degenerate (529 file-local episodes); the reference MPS throughput run (3,200-3,400 ns/day/node) |
| **chignolin_8** | **Sparse pseudo-2D layout validation run — umbrella sampling only (no GaMD boost)** | **none**: the Pep-GaMD integrator never left gamd-openmm stage 2 (conventional MD collecting statistics). All 248 replicas `stage = 2`, FSF exactly 1.0 on both channels, Vmax drifted per replica | CV1 nonlocal contacts, CV2 residual-torsion-pc (auto-selected by the swarm stage); 62 CV centres x 4 lambda rungs = 248 states, where the four rungs are dynamically identical | 4 GPU, **no MPS** (248 > ~60 clients/GPU) | Production from 2026-09-22. **Stopped deliberately at the last checkpoint inside stage 2 (step 485,200)** by the launcher watchdog; kept as a pure-US reference dataset. See below |

## chignolin_8 in detail

**Classification:** validation run for the swarm-designed sparse pseudo-2D window layout, sampled as
plain umbrella sampling (the intended Pep-GaMD boost was never active).

- **Intended:** swarm-calibrated Pep-GaMD lower-dual boost with a frozen envelope and a 4-rung lambda
  ladder over a sparse 2D umbrella grid.
- **Actual:** the swarm-to-production hand-off (epoch-0 sidecar, 03e0d43) delivered the envelope's
  physics globals but not the integrator's `stepCount`/stage, and calibration steps were 0, so every
  replica started gamd-openmm's stage machine at step 0. Stage bounds solved from the stage counters
  at two checkpoints: stage 1 = steps 0-5,000, stage 2 = 5,000-500,000, stage 3 from 500,001, where
  each replica would recompute its own threshold and k0 from its drifted statistics.
- **What the data are:** 62-window 2D umbrella sampling with 4-fold redundancy per window.
- **Caveat before using it as US:** exchanges were priced with a boost that was never applied. Rung
  swaps between dynamically identical states were accepted only 7-21 % (should be 100 %), and
  cross-rung CV swaps carried a phantom boost term, so the exchange was not in detailed balance with
  the simulated Hamiltonians. Check that the four rungs' CV distributions coincide at each centre;
  if they do, analyse as 62 US states with the rungs collapsed and lambda ignored. The lambda > 0
  labels and the recorded boost-reweighting inputs must not be used.
- **Interim item-0 check, 2026-09-23 00:20, steps 5,250-265,000 (257,920 samples, ~1,040 per window;
  script `~/gareus/chignolin/c8_us_connectivity.py`, read-only):**
  - *Rung consistency: FAILS.* The 4 rungs of a centre should be identical, but 49/62 centres show a
    pairwise mean difference > 3 block-SE (27/62 > 5); median largest difference 0.55 window sigma
    (max 1.42), KS median 0.34. Null (two time-halves of the same rung): 7/62 > 3, median 0.16 sigma,
    KS 0.18. The null makes 2 comparisons per centre against 12 for the rungs, so it is not a perfect
    match, but rungs differ ~3x more than time-halves -- consistent with the phantom-boost exchange
    distorting per-rung ensembles. Do not collapse rungs yet; re-test at the 485,200 cutoff.
  - *MBAR connectivity (rungs collapsed, 62 states): connected.* One component at overlap >= 0.01;
    at 0.03 and 0.09 only c28 separates -- the zero-k unrestrained state, whose overlap is spread
    thinly over all others (O_28,28 = 0.02, 0.98 shared out), not a gap. Weakest restrained state
    best link 0.094; spectral gap 1 - lambda2 = 0.038 (connected, slow-mixing).
  - *Per-state quality: good.* Self-bias median 0.90 kT (max 2.60), no state > 10 kT; per-state Kish
    ESS >= 8,233; unbiased-ensemble ESS 91 %; CV2 neighbour spacing 1.50 window sigma everywhere
    (MBAR overlaps 0.10-0.21 across those steps).
  - *Verdict so far:* the sparse pseudo-2D layout itself is connected and reweightable; the open
    question is the rung (exchange) artefact, not the layout.
- **Final item-0 check at the cutoff, 2026-09-23 (steps 5,250-485,500; 476,656 samples, ~1,920 per
  window; boost zero in every sample):**
  - *MBAR connectivity (62 collapsed states): connected, unchanged from interim.* One component at
    overlap >= 0.01; at 0.03/0.09 only the zero-k unrestrained state c28 separates (O_28,28 = 0.019,
    0.981 spread over the others). Weakest restrained best link 0.102; spectral gap 0.0356.
  - *Per-state quality: good.* Self-bias median 0.85 kT (max 2.86), none > 10 kT; per-state Kish ESS
    >= 15,132; unbiased-ensemble ESS 87.9 %; CV2 neighbour spacing 1.50 window sigma, MBAR overlaps
    0.11-0.21 across CV2 steps.
  - *Rung consistency: still fails, smaller.* 56/62 centres > 3 SE (37/62 > 5); median largest rung
    difference 0.43 window sigma (was 0.55), KS 0.28 (was 0.34). Null (time-halves of one rung):
    10/62 > 3, 0.19 sigma, KS 0.16. The difference shrank with 2x the data but is still ~2.3x the null
    and more significant (z grows with sample size), i.e. a persistent modest distortion from the
    phantom-boost exchange on top of slow CV sampling. Rung-collapsed US estimates carry that caveat.
  - Run directory pulled locally to `RUNS/chignolin_8/` (with launcher, benchmarks and job logs).
- **Full analysis as US (2026-09-23, `RUNS/chignolin_8_US/`, lambda zeroed):** MBAR converged, ESS 87 %,
  joint overlap graph connected; CV1 PMF converged but nearly flat (topology-degenerate contact CV);
  chignolin H-bond FES shows no native basin -- folded (both H-bonds <= 5 A) 0.01 %, dG_fold ~ +5 kcal/mol
  vs about -0.5 to -1 experimentally, after 1.7 ns/replica unboosted. The layout samples and reweights;
  the unboosted dynamics never reached the hairpin. Summary: `RUNS/chignolin_8_US/ANALYSIS_SUMMARY.md`.
- **Cutoff:** `STOP_AT_PROD_DONE=485200` in `~/gareus/chignolin/chignolin_8.sh`; marker file
  `~/gareus/chignolin/chignolin_8.US_ONLY_STOPPED`; chain status line
  `STOPPED: US-only cutoff before GaMD stage 3`.
- **Engineering by-products (all valid):** chain checkpoint/resume fixes, skeleton-manifest resume
  fix, multi-GPU pull, the throughput study (`gpu-throughput-benchmark-todo.md`, T1-T8).
- **Must be fixed before chignolin_9:** frozen-envelope production must start in gamd stage 5 (build
  the integrator with zero cMD/equilibration stages, or seed `stepCount` past stage 4), with a
  regression test asserting FSF < 1 on a lambda = 1 replica after one warm step.

## Naming rule for future runs

Record, per run: boost type **as applied** (read stage and FSF from a production checkpoint, not
from the config), CV1/CV2, state count, MPS or not, and what the data are usable for.
