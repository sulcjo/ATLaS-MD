# rep5: T7 restraint-forming kicks — first movement of the folded-state metric

Clean A/B against rep4: same seed (81806), same T1–T6 bundle, same config otherwise —
the only difference is `--restraint-forming-kicks` (T7: flat-bottom N–O walls at 3.8 Å,
k = 1000 kJ/mol/nm², capture 6 Å, active only during the 675 K kick, released before
re-minimization).

Banks: `chignolin_genpept_rep4_tuned/` (9,745 survivors) vs `chignolin_genpept_rep5_t7/`
(9,886 survivors, 2 h 38 min on the RTX 3060 Ti, 16 jobs). Config:
`scripts/chignolin_genpept_rep5_t7.yaml`. Note: GENPEPT resolves a relative `out:` against the
config file's directory — the bank initially landed in `scripts/` and was moved to repo root.

## Mechanism provenance (production scale)

- 1,098 BH hop records, 1,084 success; 97.0% of parents captured restraint pairs
  (mean 7.2, median 7, max 22 per parent).
- `t7_pairs` column in `basin_hop_minima.csv` is the per-hop provenance.
- Smoke-scale validation (80k, `genpept_t7_smoke.yaml`): BH-endpoint registers
  median 1.0 (control) → 3.0 (T7); registers survive release + minimization.

## Headline battery (final survivor banks)

| metric | rep4 (T1–T6) | rep5 (+T7) |
|---|---|---|
| native-envelope seeds | 31 (0.32%) | **39 (0.39%)** |
| registers ≥ 3 (|i−j|≥3, N–O < 3.8 Å) | 28.1% | **35.2%** |
| registers ≥ 5 | 5.8% | **9.5%** |
| registers median / p90 / max | 2 / 4 / 10 | 2 / 4 / 11 |
| best CA-RMSD to 1UAO | 0.79 Å | 1.15 Å |
| frac RMSD < 2.0 / 2.5 / 3.5 Å | 0.84 / 5.45 / 39.84 % | 0.74 / 5.11 / 39.17 % |
| median RMSD | 3.75 Å | 3.77 Å |

(RMSD values corrected 2026-09-11 for the atom-correspondence bug — see
ADVERSARIAL_VERIFICATION.md round 3; earlier editions quoted scrambled-correspondence values
1.18/1.30 Å best. Comparative conclusion unchanged.)

A/B caveat: the proposal stream is bit-identical (same seed, same flags up to BH), but
GPU implicit minimization is not bitwise reproducible run-to-run — rep5 materialized 11,939
vs rep4's 11,935 viable minima and 285 vs 295 BH parents. The A/B is therefore clean at
proposal level, with a small platform-noise floor at the BH stage.

## Reading

- **Registers**: T7 does exactly what it was designed for — the register-rich channel
  grows ~25% (≥3) to ~64% (≥5). Highly significant at n≈9.8k.
- **Envelope**: 31 → 39 seeds (+26% relative). Consistent in direction with the register
  gain, but ~1.4σ on Poisson noise alone — supportive, not standalone-proof.
- **Near-native tail**: essentially flat (frac < 2.5 Å 5.45% → 5.11%; rep5's single best
  1.15 Å vs rep4's 0.79 Å — a deeper single champion on the rep4 side, within tail noise).
  Register-rich is not the same as native-registered: some of the new registers are
  non-native pairings. T7 enriches the *skeleton*, the minimizer still decides the topology.
- **Cost**: none measurable (survivor count and wall time comparable to rep4).

## Verdict

T7 is the first mechanism in the campaign that moves the folded-state needle (envelope
0.32 → 0.39%, registers ≥5 nearly doubled) — confirming the minimizer-reach diagnosis and
partially correcting it. The remaining deficit is now localized one level deeper: forming
non-native registers is as easy as forming native ones, so the next levers are
(a) native-register *selection* after T7 formation (e.g. envelope-consistent rewards at
final selection, not BH), or (b) the solvent model itself (GBn2's near-degeneracy between
native and misfolded registers — see THERMO_CONSENSUS_MODEL.md).

## Reproduction

```
python GENPEPT.py --config GENPEPT_R7_CV_CENSUS_REPORT/scripts/chignolin_genpept_rep5_t7.yaml \
  --strict-config --resume --jobs 16 --min-jobs 1 --platform OpenCL   # ~2.6 h
```
T7 implementation: `--restraint-forming-kicks --t7-wall-a 3.8 --t7-k 1000 --t7-capture-a 6.0
--t7-max-pairs 0` in GENPEPT.py (ReusableBasinHopper). Backup of pre-T7 GENPEPT.py:
`/tmp/opencode/verify_backup/GENPEPT_pre_T7.py`.
