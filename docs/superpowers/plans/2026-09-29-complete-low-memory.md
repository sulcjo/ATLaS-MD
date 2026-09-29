# Complete low-memory analysis

Extend the approved low-memory adaptive-analysis design to the full-data
failure demonstrated on chignolin_9. Preserve float64 energies, row alignment,
Hamiltonians, requested solver backend, and sample selection.

1. RED/GREEN: disk-backed row/column selection, finite-row checks, streamed
   NPZ extraction, temporary-storage ownership. Wire all row-filter consumers.
2. RED/GREEN: bounded umbrella reconstruction, GaMD correction and overlap.
   Record epoch, correction and cleanup boundaries.
3. RED/GREEN: bounded NumPy/Anderson/L-BFGS/SAMBAR reductions and active-column
   selection; preserve Numba kernels and solver stopping rules.
4. Verify focused suites, full suite, real full-data loader and solver memory
   probe. Integrate only requested paths; preserve dirty checkout edits.

Review focus: noncontiguous state selections; repeated row indices; invalid
raw energies; inactive states; log-domain reduction stability; memmap cleanup
on success and exception; no silent subsampling or backend replacement.

Ruling: Use isolated clone under /tmp because parent checkout is dirty and
read-only in sandbox. Existing temporary worktree predates current main.
Ruling: Chunk reductions may change floating-point summation order; compare
within float64 tolerances, not bitwise equality of solver results.
Ruling: Scratch belongs on the adaptive run filesystem, not default /tmp.
The host /tmp is a 32 GiB tmpfs; files there consume RAM and hit quota.
Ruling: Fail cleanly when filesystem space reservation is unsupported.
Checking free space before a sparse mapping cannot prevent later SIGBUS.
Ruling: When skip-first-n-frames is requested, postpone stride until after
that filter. This preserves sample selection while trading more disk I/O.
Ruling: In-place GaMD correction is restricted to process-owned scratch.
Caller-owned and copy-on-write mappings remain unchanged; copy-on-write
pages must not be discarded because that would erase private modifications.

## Progress

- Disk-backed selection/finite checks and streamed NPZ extraction implemented.
- Chunked umbrella/GaMD/overlap implemented; writable scratch corrected in place.
- All solver backends use bounded reductions or existing Numba kernels in
  bounded row blocks. Fortran/dtype normalization stays disk-backed.
- Review found mapping mode w+ discarded fallocate reservation; changed to r+.
  Reservation regression observed failing before fix.
- Broad focused verification: 135 passed, 5 fixture temperature warnings.
- Final ownership/COW regressions plus loader/ladder tests: 87 passed in
  isolated clone and installed source (installed warnings include cache permissions).
- Real chignolin_9 full-data probe passed: 11,569,664 loaded rows x 236 states;
  skip-first retains 11,569,428; epoch-zero split retains 11,031,112.
  Two SAMBAR/polish iterations, finite log weights, and 236 x 236 overlap
  completed; numerical convergence intentionally not tested. Kernel peak RSS
  2.419 GiB versus prior OOM process 38.3 GiB. Scratch cleanup verified.
- Compile and installed CLI help passed. Full suite: 4,301 passed, 9 failed,
  10 skipped, 1 xfailed (29m44s). Baseline reproduces two documentation
  failures, distance logger failure, swarm provenance failure, and tiny
  lambda-ladder manifest/Parquet smoke failure. Four thermodynamic mutation
  failures reject beta=0/negative in unchanged correctness/bias.py before
  MBAR. Direct baseline calls reproduce both IntegrityError messages.
  Baseline rough-surface rerun stopped after this failure cause was resolved.
  Final ownership/COW changes were tested separately after the suite started.
  Full-suite log: /tmp/low-memory-full-suite-clean.log.
