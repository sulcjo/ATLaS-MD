# Chignolin runs (ATLaS-MD) — what each campaign actually was

Updated 2026-09-22 (ATLaS-MD v0.8.2). Boost status is read from production checkpoints
(integrator stage and force-scaling factor), not from the run configuration.
Details and evidence: `docs/atlas-md/developer/run-registry.md`.

| Run | Classification | Summary |
|---|---|---|
| chignolin_5 | GaMD-REUS, 2D adaptive | CV1 contacts; CV2 torsion-pca in epoch 0, tica-linear afterwards; ~27-28 states. Run directory no longer exists, boost type unverified (likely lower-dihedral). PMFs computed before the 2026-08-04 union-MBAR fix are invalid. |
| chignolin_6 | GaMD-REUS, 2D double-adaptive (boosted) | lower-dihedral GaMD, stage 5 on every replica; CV1 contacts + CV2 torsion-pca; up to 64 replicas, MPS. Low ESS (0.74 %) was the window-map drop bug, fixed; data repairable in the loader. |
| chignolin_7 | Pep-GaMD-REUS, 1D (boosted) | peptide-only lower-dual boost, stage 5; CV1 contacts only; up to 128 replicas, MPS. Reference run for MPS throughput (3,200-3,400 ns/day/node). |
| chignolin_8 | **Sparse pseudo-2D layout validation run — umbrella sampling only (no GaMD boost)** | Tested the swarm-designed sparse 2D window layout: 62 CV1 x CV2 centres (nonlocal contacts x residual-torsion-pc) x 4 lambda rungs = 248 states, no MPS. The Pep-GaMD integrator never left gamd stage 2, so the dynamics are plain umbrella sampling and the four rungs are identical. Stopped at step 485,200, before stage 3. Usable as a US reference for the sparse layout after a rung-consistency check (exchanges were priced with a boost that was never applied). |
