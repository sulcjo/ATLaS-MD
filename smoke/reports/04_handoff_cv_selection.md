# Handoff: ab initio folded-state thermodynamics for chignolin — the CV-selection problem

**For:** fable, astra
**From:** Claude (session 2026-09-18/19)
**Status:** chignolin_7 analysed and closed out; chignolin_8 config deployed; **one open decision blocks the next run** (§10).
**Root:** `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler` (repo `sulcjo/ATLaS-MD`, local `main`, not pushed)
**Compute:** `aurum2` (UOCHB SLURM; `ssh -o BatchMode=yes aurum2`), run dir `~/gareus/chignolin/`

This document is self-standing. Everything needed to take over is here or at a path given here.

---

## 1. The goal

Reproduce **true thermodynamic populations of the chignolin folded state at 300 K, ab initio**, for a
selected force field — no native structure used anywhere in sampling. "Ab initio" constrains the
*sampling and CV construction*; validating against the experimental structure afterwards is required
and is what the reference papers themselves do.

The peptide is **original chignolin, `GYDPETGTWG`, PDB 1UAO** (18 NMR models). Not CLN025
(`YYDPETGTWY`, 2RVD/5AWL), which is a different and much more stable peptide — several literature
folding studies use CLN025, so check before comparing numbers.

---

## 2. Reproduction targets

Two papers, in `chignolin_knowledge_base/`. **Neither is ff14SB+TIP3P** — worth stating because the
campaign was described internally as reproducing an "ff14SB+TIP3P article".

### Satoh et al., FEBS Lett 580 (2006) 3422–3426 — `1-s2.0-S0014579306006028-main.pdf`
- Amber **parm99** (modified) + **GB/SA implicit solvent**, no cutoff, SHAKE, 2 fs.
- **Multicanonical MD**, 180 ns at 700 K, reweighted to 300 K.
- **159 folding events** (Cα RMSD ≥ 4.0 Å → < 1.0 Å), called "statistically sufficient".
- 300 K cluster populations: **A (native) 25.1 %**, B 24.9 %, C 10.6 %, D 2.1 %, E 1.3 %, rest 39.4 %.
- Native H-bonds: **Asp3O–Gly7N 3.12 ± 0.27 Å**, Asp3N–Thr8O 3.40 ± 0.77, Asp3OD–Glu5N 3.39 ± 0.54.
  Asp3O–Gly7N is called *"the primal force driving the β-hairpin formation"*.
- Misfolded clusters B/C form **Asp3N–Gly7O** instead — a one-residue register shift.
- **Fig 5B is the target figure**: free-energy contour in d(Asp3N–Gly7O) × d(Asp3N–Thr8O), wells "N" and "M".
- This is the only one of the two that reports 300 K populations. Implicit solvent typically
  over-stabilises hairpins, so treat 25 % as an upper bound.

### Fujisaki et al., Life 12 (2022) 1188 — `Non-Markov-Type_Analysis_and_Diffusion_Map_Analysi.pdf`
- AMBER 11, **ff99SB + TIP3P explicit**, 15 Å buffer, 2 Na⁺, 11,081 atoms.
- **420 K** (near T_f), 1 atm, 2 fs, Langevin γ = 2.0 ps⁻¹, **750 ns plain MD**, coords every 10 ps.
- Three states F/M/I; MFPTs **~10 ns** (F→M 13, M→F 7.8, F→I 5.9, I→F 3.7, M→I 4.9, I→M 6.0 ns).
- HB1 = Asp3O–Gly7N, HB2 = Asp3N–Thr8O; *"except HB2, the correlations are good, indicating HB1 is a 'good' CV"*.
- Independently finds **the ψ of glycine and threonine** most correlated with the slow modes.
- A kinetics paper at 420 K — it reports **no 300 K populations**. Use it as the system definition and
  a cheap validation anchor, not as the thermodynamic target.

**Measured on the 18 NMR models of 1UAO** (this session, `native_fold_audit_results/native_reference.json`):

| pair | 1UAO | Satoh's MD |
|---|---|---|
| Asp3O–Gly7N (HB1, native) | **2.84 ± 0.13 Å** | 3.12 ± 0.27 |
| Asp3N–Thr8O (HB2, native) | 3.57 ± 0.41 Å | 3.40 ± 0.77 |
| Asp3N–Gly7O (misfold register) | 6.73 ± 0.38 Å | — |
| Asp3OD–Glu5N (side chain) | 3.65 ± 0.61 Å | 3.39 ± 0.54 |

**The run's `chignolin_fes` tracks Asp3N–Thr8O and Asp3N–Gly7O, and never measures Asp3O–Gly7N** —
the one both papers call the native/"good" coordinate. Cheap gap to close.

---

## 3. The software

`gareus/` — ATLAS-MD / GAREUS. Umbrella sampling along CV1 × a GaMD λ ladder, replica exchange over
the full (window, rung) state set, MBAR over all states.

| concern | file |
|---|---|
| CV definitions, modes | `gareus/cv.py` (contact CV at `:881`), `gareus/cv_discovery.py` |
| GaMD calibration (2-stage) | `gareus/production.py:5358–5470` (`apply_joint_envelope_gamd_calibration`) |
| recon driver | `gareus/production.py:5231+` (`run_multiwindow_gamd_recon`) |
| exchange / gibbs-walk | `gareus/production.py:706–1000`, `:7382+`; 2-D neighbours `:1199` |
| bias matrix for MBAR | `gareus/query.py:1–18` |
| swarm (epoch 0) | `gareus/swarm/` — stratify, members, envelope, `ladder_design.py` |
| stage-length validation | `gareus/cli.py:1013` (`validate_gamd_stage_multiples`) |
| analysis | `analyze_gareus_mbar.py` (PCA ~`:1981`), `gareus_report.py` |

**The bias form that makes everything else possible** (`gareus/query.py`):

```
U_k(cv) = 0.5·k1_k·(cv1 − c1_k)²  +  0.5·k2_k·(cv2 − c2_k)²
```

per-state k and centre on both axes; the second term applies only when that window has finite
`c2_k`, finite `k2_k` and `k2_k > 0`. The runtime computes `secondary_cv_bias_all_windows_kcal`
(every window's bias for each sample), which is what gibbs-walk needs.

---

## 4. The chignolin_7 campaign

Config `aurum2:~/gareus/chignolin/chignolin.yaml` (schema 2.0, nested sections — **not** the flat
key list that `effective_config.yaml`'s `config_values` shows). Local record:
`RUNS/chignolin_7/effective_config.yaml`, `RUNS/chignolin_7/run_args.json`.

ff14SB + TIP3P, 300 K, NPT 1 bar, 3.5 fs with HMR (H = 3.024 amu), 0.15 M, dodecahedron, 1.0 nm
padding, **0.8 nm nonbonded cutoff**, Langevin 1.0 ps⁻¹, built from sequence (`input_pdb = None`).

CV1 = reference-free nonlocal contact fraction: heavy atoms, |i−j| ≥ 4, r₀ = 12.0 Å, β = 3.0 Å⁻¹,
normalized, atom-pairs. 16 centres × 4 λ rungs (0, 0.232349, 0.642932, 1.0) = **64 states**.
Exchange every 400 steps (1.4 ps), gibbs-walk. `cv2: none`.

**Scale:** 5,124,128 MBAR samples; **611,536 coordinate frames** (11.9 % of samples — `traj_interval`
was 2500; fixed to 250 in v0.8); 1,472 trajectory files; **5.351 µs** aggregate walker time.

---

## 5. What was measured — the audit

Full report: `RUNS/chignolin_7/adaptive_production/pmf_analysis/NATIVE_FOLD_AUDIT_2026-09-18.md`
(+ `.docx`). Adversarial self-review: `ADVERSARIAL_REVIEW_2026-09-18.md`. Data and scripts:
`pmf_analysis/native_fold_audit_results/`.

### 5.1 The fold was sampled but never scored
- **33 frames of 611,536 reach Q ≥ 0.80** against 1UAO's own 65-contact map; all 33 also satisfy
  backbone RMSD ≤ 1.0 Å and a formed Tyr2/Trp9 cluster.
- Tightest backbone **0.435 Å** (Q 0.752); highest **Q 0.932** (RMSD 0.485 Å) — *different frames*.
- 1UAO's own model-to-model spread: mean 0.77 Å, max 1.37 Å.
- 4,929 frames ≤ 1.5 Å across 33/64 replicas, 67/1472 files, 11 run segments.

### 5.2 No PMF minimum is at native — with the same sampling caveat as §5.7

⚠️ Every free energy in this subsection rests on the same ~1 folding event that forced the
ff14SB/TIP3P withdrawal in §5.7. They are reported because they are what the run produced, **not
because they are converged**. "No PMF minimum is at native" is a statement about *the run's PMF
products*; it is not evidence about the system's free-energy landscape. The native-cell
(+4.09 kcal/mol) and misfolded-cell (+3.57) values, and §5.7's "+4.06 independently reproduces
3.70", are all mutually dependent on that same deficient sampling — the agreement between them is
**not** independent corroboration.
Recomputed from matched rows with block-bootstrap CIs (first CIs in this campaign):

| coordinate | published | matched rows | 95 % CI | native | verdict |
|---|---|---|---|---|---|
| CV1 contacts | 0.4873 | — | — | 0.82–0.93 | outside |
| Rg (Å) | 6.594 | **6.572** | [5.98, 6.97] | 5.14–5.66 | outside |
| d1 Asp3N–Thr8O (Å) | 7.206 | **7.206** | [7.21, 18.12] | 2.84–4.22 | outside |
| d2 Asp3N–Gly7O (Å) | 8.377 | **8.377** | [8.38, 14.83] | 6.17–7.56 | outside |
| PCA1 / PCA2 | 1.439 / −5.289 | — | — | — | **not counted** (2 % surface) |

The d1/d2 surface reproduces **neither** well of Satoh Fig 5B: all 7 local minima have both H-bonds
broken. Native cell sits at +4.09 kcal/mol (112 frames), misfolded cell at +3.57.

### 5.3 CV1 is degenerate — and fails necessity too
At fixed CV1 the structure is not determined:

| CV1 bin | frames | RMSD min | median | max | % < 1.5 Å |
|---|---|---|---|---|---|
| [0.00, 0.20) | 81,979 | 2.72 | 5.02 | 6.45 | 0.00 |
| [0.20, 0.40) | 151,705 | 1.98 | 4.22 | 6.10 | 0.00 |
| [0.40, 0.60) | 148,703 | 0.87 | 3.69 | 5.56 | 0.06 |
| [0.60, 0.75) | 113,198 | 0.65 | 2.68 | 4.93 | 1.22 |
| [0.75, 0.82) | 56,208 | 0.47 | 2.61 | 4.55 | 2.34 |
| **[0.82, 0.93)** | **58,670** | **0.43** | **2.56** | **4.27** | **3.65** |
| [0.93, 1.00) | 1,073 | 1.25 | 2.44 | 3.85 | 1.03 |
| total | 611,536 | | | | |

Sufficiency: only 3.65 % of the native CV1 window is native. **Necessity: 56.6 % of near-native
frames fall OUTSIDE that window** (2,140 of 4,929 inside; the rest spread over CV1 0.53–0.94).

### 5.4 Only ~1 genuine folding event
Exchange records demultiplexed (§6). Counting Satoh's statistic along continuous walker series,
with hysteresis (a walker must fully unfold before it can re-fold):

| definition | events |
|---|---|
| Satoh thresholds on Cα RMSD (< 1.0 / ≥ 4.0 Å) | **1** |
| backbone res 2–9, min over 18 models | **4** |
| loose (< 1.5 / ≥ 4.0 Å) | **25** |
| **Satoh 2006, 180 ns multicanonical** | **159** |

Over the three definitions above the shortfall spans **34–851 µs**, not one figure:

| definition | events | rate | µs needed for 159 |
|---|---|---|---|
| Satoh-exact Cα | 1 | 0.19 /µs | **851** |
| backbone res 2–9 | 4 | 0.75 /µs | 213 |
| loose (≤ 1.5 Å) | 25 | 4.67 /µs | **34** |

i.e. the campaign is short by one to two and a half orders of magnitude in folding events.

**This is the campaign's actual defect.** An earlier version of the audit claimed "sampling is not
the blocker" on the strength of 529 contiguous native episodes; those are overwhelmingly *re-entries*
from partly-folded states, not folding events. That claim is withdrawn in §8 of the audit.

### 5.5 Why: the boost cannot cross the barrier
`boost`: mean **2.08 kcal/mol** (3.5 k_BT), sd 1.30, max 13.01, against `sigma0p = sigma0d = 6.0`
that therefore **never became active**. exp(⟨ΔV⟩/k_BT) ≈ 33× is **not** an upper bound — by Jensen's inequality it is a
*lower* bound on ⟨exp(ΔV/k_BT)⟩. The sharper and correct argument is structural: ΔV = ½k(E−V)² is
zero once V ≥ E, and E = V_max was frozen from **35 ps of basin fluctuation**, so the transition
state almost certainly lies *above* E. Then **ΔV(TS) ≈ 0 and the barrier enhancement is ≈ 1×** — the
boost fills wells it has already sampled and does nothing at the barrier. That, not the ⟨ΔV⟩
magnitude, is why 5.351 µs bought ~1 folding event.

Root cause, from the config: the multiwindow recon ran at `gamd_multiwindow_recon_cmd_steps: 5000`
(17.5 ps) and `gamd_multiwindow_recon_steps: 10000` (35 ps), with
**`gamd_recon_boosted_iters: 0`** — so stage 2 of the documented two-stage calibration ("the boost is
turned on and the per-window recon is repeated and pooled, iterating to self-consistency on sigmaV so
the frozen boost matches the boosted production ensemble (not the unboosted one)") **never ran**.
V_max from 35 ps is a basin-fluctuation width.

### 5.6 Estimator validity
Only **one** estimator is valid. The GaMD boost enters the MBAR reduced potential directly
(`pmf_summary.json` warning: *"λ ladder: boost is inside u_nk, MBAR is exact; no cumulant"*), so
`umbrella_only` needs no reweighting correction. The others are disqualified:
`boost_reweight_ess = 51.2` of 4,583,936 (1.1 × 10⁻⁵) kills `gamd_exponential`;
`anharmonicity_score = 0.799`, excess kurtosis 0.905 kills both cumulants. Their minima disagree
wildly (CV1: 0.101 / 0.166 / 0.487 / 0.745; Rg: 5.82 / 6.59 / 7.37 / 7.75 Å) — that spread is a
**disclosure, not corroboration**, since three of four are invalid.

### 5.7 Native-state support
| native definition | MBAR weight | ΔG above ensemble | Kish ESS | largest single weight |
|---|---|---|---|---|
| RMSD ≤ 1.5 Å | 1.10 × 10⁻³ | **+4.06 kcal/mol** | 215.3 | 1.6 % |
| RMSD ≤ 1.0 Å | 2.36 × 10⁻⁵ | +6.35 | **6.9** | **33 %** |
| Q ≥ 0.80 | **0 — see below** | — | — | — |

**Why the 33 best frames carry no MBAR weight — and why it matters more than it looks.** They are
not a join failure. **All 33 lie in `epoch_001/topup_002_3378000`**, one of the three λ=0-only
top-ups that ran fully boosted (defect D7); those rows are dropped as wrong-ensemble data, correctly.
Quantified over the whole archive:

| population | in the 3 wrong-ensemble segments | share |
|---|---|---|
| all frames | 99,264 / 611,536 | 16.2 % |
| RMSD ≤ 1.5 Å | 1,645 / 4,929 | 33.4 % |
| RMSD ≤ 1.0 Å | **415 / 481** | **86.3 %** |
| **Q ≥ 0.80** | **33 / 33** | **100 %** |

Those segments are 16.2 % of the campaign and hold 86.3 % of its tightest native frames — a 5.3×
enrichment. The correctly-configured 83.8 % of the campaign produced **66 frames ≤ 1.0 Å and zero at
Q ≥ 0.80.**

Three consequences. (i) The structural claim "the fold was sampled" survives — these are real
ff14SB configurations, merely generated under a boost that was not the intended one — but it rests
overwhelmingly on a mis-configured segment, and that must travel with the claim. (ii) No free energy
can be attached to them, by construction. (iii) **It is an accidental controlled experiment in
exactly the direction chignolin_8 bets on**: those segments ran with a real boost (9.44–10.60 kJ/mol)
where genuine λ=0 had exactly 0.000, and they are where the folding happened. That is the strongest
within-campaign evidence that the weak boost is the binding constraint.

Independently reproduces the audit's F = 3.70 kcal/mol at the native d1/d2 cell. **No population or
free energy should be quoted for the fold** — with ~1 folding event and ESS 6.9 at the tight
threshold, the number is not estimable. The earlier "ff14SB/TIP3P destabilises the fold" conclusion
is **withdrawn** for the same reason.

---

## 6. Demultiplexing (done — reusable)

Exchange records survive: 33 parquet files, **2,067,726 attempts**, columns
`step, replica_i, replica_j, window_i, window_j, delta_e, accepted`, under
`adaptive_production/*/exchanges/*/data.parquet`.

**Frame ↔ step is exact.** Every segment restarts its step counter at **1,010,400** and writes a
frame every 2500 steps, so frame *i* of a segment is at step `1,010,400 + (i+1)·2500`. Verified on `final/baseline`, stating the
convention unambiguously: with *i* 0-indexed, frame 0 is at step 1,012,900 and frame 1357 at
**4,405,400** ≤ 4,406,800 (the last exchange step), while frame 1358 would fall at 4,407,900 >
4,406,800 — so exactly **1358 frames**, which is what was written. (An earlier draft quoted
"(4,406,800 − 1,010,400)/2500 = 1358.56"; that was sloppy presentation of the same check, not a
different mapping.) The replica
index is a **continuous walker**; what changes under exchange is the state it occupies, and the
records report each replica's window at every attempt, so assignment is read directly, never replayed.

Result: **611,530 / 611,536 frames (99.999 %)** carry their state. Scripts `demux.py`, `events.py`.

**The ladder mixes well** (this is a statement about *sampling*, not about free-energy precision —
see below) — acceptance 0.8199; rung occupancy 24.2 / 25.5 / 26.1 / 24.3 %;
**42,724 λ=0↔λ=1 round trips**, median 39 per walker, 1 of 448 series with none. This settles the
"38 weak rung edges / RESULT HEALTH FAIL" alarm: low pairwise MBAR overlap coexists with excellent
traversal (bulk-vs-tail statistics on a heavy-tailed boost, plus 2-D overlap dilution against a
threshold calibrated on a 1-D 5-state ladder). **It does not clear the ladder for free-energy
precision**: round trips establish ergodic mixing across states, while MBAR variance is governed by
pairwise phase-space overlap, which the gate measures at 0.065–0.15 against a 0.15 target. Both hold
at once.

⚠️ **Steps are segment-local, not globally unique.** `rg_samples_with_weights.csv` has no segment
column, so `(step, replica)` is not a key there — this defeated a naive weight join. Use
`adaptive_union_mbar.samples.csv` (carries `source`/`source_dir`) or the parquet paths.

---

## 7. Defects found (all verified)

| # | defect | status |
|---|---|---|
| D1 | `gamd_recon_boosted_iters: 0` — calibration stage 2 never ran; frozen envelope is the *unboosted* one | **fixed in chignolin_8** |
| D2 | recon at 17.5/35 ps — V_max is a basin fluctuation width | **fixed in chignolin_8** |
| D3 | PCA FES built on **103,664 of 5,124,128 slots (2.02 %)**; cache reused because the guard tests `p1.shape == d.cv.shape` and NaNs pass a shape test (`analyze_gareus_mbar.py` ~:1993). Guard should test finite count. Ceiling is 11.9 % anyway (only 611,536 frames exist) | open |
| D4 | Rg and d1/d2 are **forward-filled** across ~10 sample rows each — `cv_A` changes every 250-step row while `rg_A` repeats in blocks. Nominal 5,110,208 support is really 611,536 | quantified; minima unchanged |
| D5 | `rg_samples_with_weights.csv` has no segment column → unjoinable to coordinates | open |
| D6 | CV1 is dimensionless but reported as `cv_min_A` / `pmf_minimum_cv_A` / "CV distance (A)" | open |
| D7 | λ=0 rung retains only **34.4 %** of its samples — three λ=0-only top-ups ran fully boosted (scheduler bug); 992,896 wrong-ensemble samples, correctly dropped; 14.2 % of the reduced bias matrix non-finite | fixed in v0.8 (`e565d0c`) |
| D8 | `adaptive_union_mbar_analysis` reports `PyMBAR unavailable ... scipy.linalg has no attribute 'tril'` and wrote coverage only | open |

**Sample-count reconciliation** (five totals circulate): 5,124,128 = Σ n_k over 64 states;
5,110,208 = finite subset used for FES; 4,583,936 = samples with finite boost = 5,124,128 − 540,192
(the epoch_000 count — exact but undocumented); 2,947,296 = rows in `adaptive_union_mbar.npz`;
2,387,808 = rows its analysis used = 2,947,296 − 559,488 (`gamd_ladder_samples_without_raw_energies`);
611,536 = coordinate frames.

---

## 8. The GENPEPT seed banks and CV census

GENPEPT generates a diverse conformer library from sequence alone (basin hopping + implicit
minimization + diversity selection). It supplies umbrella seeds and the CV-fitting data.

⚠️ **It is NOT fully ab initio as run.** The r7 bank used `diversity_bank_preset=chignolin`, and
`GENPEPT.py:8118` describes that preset as biasing *"toward hairpin/turn-rich banks"*. Against
`broad` (`GENPEPT.py:1146–1163`) it raises β+turn bank mass 0.55 → 0.65 and drops α 0.25 → 0.15.
Chignolin is a β-hairpin: **a preset named for the target upweights the target's fold class.** No
native coordinates enter, and the free energies are unaffected (the preset acts on seed generation,
not the Hamiltonian, and MBAR reweights from whatever was sampled) — but the *discovery* claim is
weakened, and the mode split of §10 must extend here:
**ab initio production → `diversity_bank_preset: broad`; validation runs → `chignolin`, recorded as such.**

Two adjacent knobs are clean and should not be lumped in: `contact_bias_strength = 0.15` resolves to
`exp(0.15·(ccount − ccount_max))` (`GENPEPT.py:1841–1850`), a generic compaction prior carrying no
fold information; and `adaptive_register_threshold.json` is fit from the library's own N–O histogram
(rep5_t7: `valley_detected: true`, τ = 5.31 Å over 86,400 distances), not from 1UAO — note it differs
per bank (rep4: `valley_detected: false`, fallback 3.8 Å).

### Banks (`RUNS/`)
`chignolin_genpept_r7` (the one in use), and panel banks for **CLN025, TRPZIP2, TC16bP12W,
RETROTRPCAGE, GPM12, YGGFL, NLYIQWLKDGGPSSGRPPPS**. Repo-root working copies:
`chignolin_genpept_rep2`, `rep3_massive`, `rep4_tuned`, `rep5_t7` (42–45 entries each: basin-hop
minima, implicit/NMA-minimized PDBs, candidate seeds, `final_survivor_seeds.csv`,
`GENPEPT_turbo_summary.json`, feature caches).

chignolin_7 seeded from `~/gareus/chignolin/chignolin_genpept_rep5_t7`.

### The census — `GENPEPT_R7_CV_CENSUS_REPORT/`
Dated 2026-09-08. **1,970 conformers**, **298 CVs** each. Verified independently
(`ADVERSARIAL_VERIFICATION.md`, 2026-09-10: funnel counts, comparison battery reproduced *bitwise*,
all audited numbers exact). Key documents:

| file | content |
|---|---|
| `README.md` | 18 headline findings — read this first |
| `CV2_DEEPDIVE.md` | residual-torsion PCA, diffusion content, joint maps |
| `ORTHOGONAL_PAIRS.md` | ranked CV1×CV2 pairs with a composite score |
| `DYNAMICS_VALIDATION.md` | CV1 candidates on chignolin_6 solvated MD (ACF t½ 172–190 ps) |
| `DIFFMAP.md`, `FOURIER.md`, `TURN_AXIS.md` | diffusion maps, circular harmonics, turn axis |
| `RIGOR_HANDOFF.md`, `THERMO_*.md`, `REP*.md` | per-bank comparisons |
| `TRPCAGE_CV_CENSUS_REPORT/` | same treatment for Trp-cage |
| `figures/`, `scripts/` | plots and re-derivation scripts |

**Findings that matter here:**
1. Best CV1 = backbone-heavy or CA logistic contact fraction at **r₀ = 10 Å, |i−j| ≥ 4**, any β ≥ 1.5
   (entropy 0.973–0.980, 25–27 resolvable windows at k = 1200). The run's all-heavy r₀ = 12 is
   statistically co-champion (0.954–0.958, 23–25 windows).
2. The legacy r₀ = 4.5 Å, β = 6 definition is **degenerate** (≤ 2 resolvable windows).
3. Rg is a near-duplicate of mid-r₀ contact fractions (|r| ≈ 0.94–0.98) — pairing them adds nothing.
4. Backbone H-bond switch counts are **degenerate on this library** (median 0).
5. Torsion-PCA CV2 has a **flat spectrum** (PC1 12.2 %, PC2 9.2 %), broad unimodal projections.
6. Residualization **rotates within one subspace**: |cos(plainPC1, resPC1)| = 0.166 but top-5
   principal-angle overlap = 0.986.
7. **Bank-fit CV2 axes are not diffusive** — Hess cosine content median ≈ 0.07, 0/32 traces ≥ 0.5.
   A seed-bank torsion PCA is a *structure-diversity* coordinate, **not a slow-motion coordinate**.
8. **Bank axes do not transfer across banks** — per-trace corr(bank resPC1, chignolin_6 stored
   `secondary_cv`) median −0.540, range [−0.817, +0.075]. Persist each bank's eigenvector with its run.
9. 2-D grid feasibility fine: 16×8 occupancy 94–97 %, per-CV1-bin CV2 IQR ≈ 1.35–1.44.
10. Diffusion maps identify the compaction axis as the first nontrivial coordinate (R² = 0.985);
    residual-torsion CV2 appears only at ψ₃. TwoNN intrinsic dimension ≈ 3.6–4.6 — **the manifold is
    ~4-D, so a 2-D grid is a cover, not a parameterization**.
11. The residual CV2 is physically a **strand-flank curl transition**: at matched CV1, ψ of Y2, P4,
    E5, T6, T8, W9 rotates from β-extended (+91…+149°) to curled (−7…−16°) while D3/G7 anchor the
    turn — it tightens the turn without changing compaction.

### Design document
`docs/Two-CV Design- Nonlocal Contact Topology + Orthogonalized Circular Torsion Mode.docx` —
specifies CV1 = smooth nonlocal contact count, CV2 = PCA/TICA of circular torsion features
**residualized against scalar CV1**; prescan → fit f(C) → residualize → project → **freeze before
production, never retrain mid-run**; implementation modes `torsion_raw_pca`,
`torsion_residual_vs_contact` (recommended default), `torsion_residual_vs_contact_topology`; risks
(over-orthogonalization, prescan coverage, non-differentiability, CV drift).

### Prior CV scoring — the accepted criterion
From `RUNS/chignolin_6/cv2_modes/` (memory `chignolin-cv2-both-definitions-weak`):
**score CVs on exploration leverage + reweightability, never on folded/unfolded separation.**

| CV | η² leverage | reweight ratio |
|---|---|---|
| contact **heavy** | **0.452** | 1.20 |
| Rg (floor) | 0.411 | 0.80 |
| CV2 tica-linear (stored) | 0.316 | 1.61 |
| contact sidechain-all | 0.303 | 0.68 |
| CV2 torsion-PCA (plain) | **0.081** | **0.66** |
| tICA refit, within-segment | — | **2.71** |

Two documented traps: `sidechain-all` silently drops glycines (12 pairs / 7 residues, excludes Gly7);
and **never measure implied timescales on umbrella data without removing each segment's mean** —
restrained replicas barely move along the restrained CV, so ITS grows linearly with lag instead of
converging.

---

## 9. The CV-selection problem (this session's new work)

All candidates computed on **all 611,536 frames**; torsion features = sin/cos of 9 φ + 9 ψ = 36
dims, matching the design doc. Scripts `cv_hunt.py`; data `cv_candidates.npz`; results
`cv_ranking.txt`, `cv_ranking_final.json`, `auto_select.txt`, `why_pc2.txt`.

Label for evaluation only: folded = backbone RMSD (res 2–9, min over 18 models) ≤ 1.5 Å.

### Standalone discrimination, folded vs everything else

| AUROC | CV | note |
|---|---|---|
| 0.9777 | cartesian CA PC1 | **circular with the label** — PC1 is linear in superposed CA coords, the label is an RMSD in the same space; it even beats native-referenced Q. Also needs a superposition reference to bias on. |
| 0.8591 | CV1 nonlocal contacts | the current CV1 |
| 0.8384 | Rg (heavy) | |
| **0.7891** | **torsion residual-vs-CV1 PC2** | corr with CV1 = **0.0000** |
| 0.7617 | nonlocal backbone H-bond count | |
| 0.6784 | torsion plain PC1 | |
| 0.5119 | torsion residual PC1 | chance |
| *0.9564* | *Q vs 1UAO* | *ceiling, not ab initio* |

Replicated EVR spectrum on production frames: plain [0.176, 0.083, 0.074, 0.067, 0.058, 0.047];
residual [0.111, 0.085, 0.080, 0.074, 0.061, 0.049] — same flatness the bank showed.

### As a second CV — what actually matters

| CV set | AUROC |
|---|---|
| CV1 alone | 0.8591 |
| CV1 + residual PC1 | 0.8607 |
| **CV1 + residual PC2** | **0.9230** |
| CV1 + residual PC1 + PC2 | 0.9227 |
| CV1 + residual PC2 + PC6 | 0.9259 |
| CV1 + Rg | 0.8766 |
| CV1 + nHB | 0.8618 |

Inside the degenerate native CV1 window (58,670 frames, 3.65 % folded), residual PC2's
**low quartile is 11.00 % folded (3.02×) and its high quartile is 0.00 %** — it splits the window
into a folded-enriched half and a folded-free half. The H-bond count does nothing there
(0.99× / 0.67×).

**Why not PC1, and why not both:** residual PC1 is at chance (0.512) *by construction* —
residualizing against CV1 removes precisely CV1's folding signal (corr(CV1, folded) = +0.111,
corr(resPC1, CV1) = 0.0000). Adding PC1 on top of PC2 *lowers* the score (0.9227 vs 0.9230). The
currently configured `bootstrap_torsion_component: 5` scores 0.566 — near chance.

### ⚠️ The blocker: no unsupervised rule picks PC2

| selection rule | picks |
|---|---|
| variance (EVR) | PC1 |
| conditional multimodality within CV1 bins | PC4 |
| slowness / implied timescale (per-series centred, fine lag grid, 320 series) | PC1 |
| **folded-vs-unfolded AUROC (supervised)** | **PC2** |

Implied timescales are **306–473 ps across all six components** (1.5× spread) and
**correlation(timescale, AUROC) = −0.033**. Slowness carries no information about folded/unfolded
discrimination here — consistent with census finding 7 (bank-fit torsion axes are structure-diversity
coordinates, not slow modes) and with the chignolin_6 note that "no candidate reaches large ratio, so
hidden slow modes remain whatever chosen".

The flat spectrum and uniform timescales mean **the residual torsion space has no natural best axis** —
a property of the system, which is why every criterion picks a different component.

**Never computed for the residual components: η² leverage and reweight ratio** — the project's own
accepted criterion. That is the missing input.

---

## 10. Open decision (blocks the next run)

**State this plainly first: for this system, ab initio selection of the second CV is currently
unsolved.** Not "pending a criterion choice" — measured. Variance picks PC1, conditional
multimodality PC4, implied timescale PC1, and correlation(timescale, AUROC) = −0.033 across six
components spanning only 306–473 ps. No unsupervised rule tested recovers the supervised pick. Do
not expect the η²/reweight scorer to land on PC2; if it picks PC1 or PC4, **that is the ab initio
answer and PC2 is simply not available to an ab initio campaign.**

With that said: the greedy structure — *pick axis 1, residualize, pick the axis maximising
incremental score* — is right and easy to implement, and the scoring function determines the answer
(PC1 vs PC2 vs PC4).

**Proposed resolution:** implement greedy selection with a **pluggable scorer**, and record in the
config which scorer ran, so an ab initio claim is never made by a run that used labels.

- **Ab initio production** → scorer = η² leverage + reweight ratio (chignolin_6 harness). Unsupervised,
  is the project's accepted criterion, and is the number still missing. It may or may not pick PC2.
- **Methods-validation runs (e.g. chignolin_8)** → scorer = supervised AUROC vs 1UAO. Picks PC2.
  Legitimate when reproducing a known landscape rather than claiming discovery.

**Pre-registered decision rule — commit to this before running the scorer, not after.**
If the η²-leverage + reweight-ratio scorer picks PC2, use it and the 2-D ab initio design stands.
**If it picks anything else** (PC1 and PC4 are the live alternatives), then no ab initio selector
recovers a folding-informative second axis, and the correct conclusion is that **the 2-D ab initio
CV design is falsified for this system** — not that a better scorer should be sought. Route instead
to **T-REMD (300–420 K) or multicanonical**, which obtain 300 K populations with *no CV at all* and
therefore sidestep the entire selection problem. §12 step 6 is that route.

**First action for whoever takes this:** compute η²-leverage + reweight ratio for residual components
1–6 and see whether an ab initio selector agrees with PC2. Everything needed is in
`cv_candidates.npz` + `frame_metrics.npz` + `demux.npz`.

### Adding the rows to the HREX array
Keep CV1 = contacts everywhere so `cv1` means one thing; new rows get **k1 = 0** (with a *finite*
dummy centre — the primary term is unconditional, and `0 × NaN = NaN`) and k2 > 0 along CV2, times
the same 4 λ rungs. Two hazards:
1. **cv2 must be observed on every sample in every state, including where k2 = 0.** `query.py`: *"A
   sample whose own cv2 is non-finite gets NaN for any window that DOES restrain CV2."* In
   chignolin_7 `secondary_cv_finite_fraction = 0.0`. Get this wrong and the two blocks decouple into
   disjoint MBAR problems sharing a walker pool.
2. The frozen GaMD envelope is pooled over the *initial* windows; new rows sampling elsewhere can hit
   V < V_min → FSF < 0 → **NaN at boost onset** (the documented S3-pilot failure). Cover the new rows
   in the recon or widen the envelope deliberately.

N = 8 → 96 states, under `max_replicas: 128`, but the 6000 ns pool is shared (~33 % less per state).

---

## 11. chignolin_8 — deployed, not yet launched

`aurum2:~/gareus/chignolin/chignolin_8.yaml` (md5 `cc595c7a741edaff06e7352e48713c2b`; local copy
`chignolin_8.yaml`). Patched from the **real** schema-2.0 config; exactly six values differ:

| knob | c7 | c8 | at 3.5 fs |
|---|---|---|---|
| `output.out` | chignolin_7 | chignolin_8 | |
| `recon_prep_steps` | 2,000 | 100,000 | 7 ps → 350 ps |
| `recon_cmd_steps` (stage 1, boost off) | 5,000 | 285,000 | 17.5 ps → ~1 ns |
| `recon_steps` (stage 2, per iter) | 10,000 | 285,000 | 35 ps → ~1 ns |
| `recon_boosted_iters` | **0** | **4** | stage 2 runs |
| `description` | — | notes the change | |

Rungs stay pinned at 4 (`swarm_min_rungs = swarm_max_rungs = 4`, `swarm_target_beta_sigma = 2.5`) so
the calibration fix is tested in isolation. Cost 6.75 ns/window → ~108 ns (16 windows) or ~431 ns
(64 states) against a 6000 ns budget.

**Why re-enabling stage 2 is safe now:** it was disabled after a NaN (`# pilot attempt 7: boosted
recon NaN-ed without per-window diagnostics`), and the mechanism is documented ten lines above in the
same file for the main stages — *"100 ps cMD let V_pep cross Vmin → FSF < 0 → NaN at boost onset"*.
The main stages were moved to ~1.75 ns for that reason while the recon stayed at 17.5/35 ps, inside
the proven NaN-prone regime. The length change is the fix for that NaN; the two edits are not
independent. `recon_report_interval: 0` auto-derives to `recon_steps // 200` = 200 diagnostic points
per window per iteration (c7 got 50).

**First checks on the c8 run**, in `chignolin_8/.../shared_gamd_setup_globals.json`: `sigmaV_trace`
should settle within 4 iterations, `boosted_calibration_report.converged` true. **If the final mean
boost is still ~2 kcal/mol, the envelope was never the limiter — stop before spending the pool.**

---

## 12. Recommended sequence

1. **420 K Fujisaki validation with plain MD.** Match their setup (ff99SB, TIP3P, 15 Å buffer, 2 Na⁺,
   11,081 atoms, 420 K, γ = 2.0 ps⁻¹, 2 fs, 750 ns). MFPTs ~10 ns → ~75 transitions with no enhanced
   sampling, no reweighting, no CV, no MBAR. Validates system build + force field + analysis chain
   end-to-end against a published result, and yields the V-statistics a correct envelope needs.
   Cheapest decisive experiment available.
2. **Add Asp3O–Gly7N** as an observable (ideally a CV). One pass over existing frames.
3. **Implement Satoh's restraint-satisfaction metric** from the 1UAO restraint list (their Fig 1A) —
   the reproduction's primary yardstick; there is no equivalent today.
4. **Native-seeded control**: unbiased runs from several of the 18 NMR models *and* from the 25
   extracted native-like frames (already equilibrated in this force field/box). Separates
   force-field from method. Decision rule fixed in advance: folded population ≫ 0.1 % ⟹ the
   enhanced-sampling estimate is a method artefact; decay toward ~0.1 % ⟹ the force field genuinely
   destabilises it.
5. **Resolve §10**, then build the CV2 rows.
6. For 300 K populations, the method must decouple barrier crossing from 300 K dynamics —
   **T-REMD 300–420 K** (anchored at the top by Fujisaki; each replica samples its own canonical
   ensemble, no reweighting variance) or **multicanonical** (Satoh's method, proven on this peptide).
   Fixing GaMD reweighting is *not* viable: ESS 51 of 4.58 M with anharmonicity 0.799 is not a tuning
   problem.
7. **Adopt a sufficiency gate**: ≥ 100 independent U→N transitions, measurable now via demultiplexing.
   The current gate (`adaptive_production_convergence_max_weak_edges = 0` plus overlap thresholds)
   passed a run with **one**.
8. **Generality** needs the panel, and the seed banks already exist: CLN025, TRPZIP2, TC16bP12W,
   RETROTRPCAGE, GPM12, YGGFL. "Consistently" cannot be established on n = 1, and chignolin is an
   atypically marginal test case.

---

## 12a. How to actually run it

- **Code**: `/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler`, local `main` at
  **`24acc314ddd329571869b45ed347162a78900450`** (tag `release: v0.8.1`), **4 files dirty, NOT
  pushed** — push and record the hash before handing over, or fable/astra cannot reproduce the tree.
  Aurum copy is a `git archive` extract at `~/2026_peptide_sampler` (chignolin_7 ran
  `DEPLOYED_COMMIT = a03f3f7`).
- **Environment**: `/home/sulcjo/conda-envs/calc/bin/python` on aurum (Python 3.9).
- **Launch** (chignolin_8; the c7 line with the config swapped):
  ```
  /home/sulcjo/conda-envs/calc/bin/python /home/sulcjo/2026_peptide_sampler/gareus/config.py \
    --config /home/sulcjo/gareus/chignolin/chignolin_8.yaml \
    --out /home/sulcjo/gareus/chignolin/chignolin_8 \
    --platform CUDA --device-index 0,1,2,3 --precision mixed \
    --cuda-disable-pme-stream true --cuda-mps --platform-temp-directory /tmp \
    --tui-mode dashboard --progress-mode both \
    --us-pull-workers 28 --us-start-primary-bad-bias-kcal 15.0
  ```
  Resume with `--config <out>/config/effective_config.yaml --resume`. The run self-resubmits on
  walltime; `MAX_RESUBMITS` caps the chain and it does **not** resubmit on exit code 1.
- **Force field for chignolin_8**: unchanged from c7 — **ff14SB + TIP3P**, deliberately, because c8
  isolates the calibration fix. The reproduction target force fields are parm99+GB/SA (Satoh) and
  **ff99SB**+TIP3P (Fujisaki); the project's forward decision is ff19SB+OPC.
- **D8 workaround** (`scipy.linalg has no attribute 'tril'`): scipy ≥ 1.13 removed
  `scipy.linalg.tril`, which pymbar calls. Either pin `scipy<1.13` in the analysis env, or
  monkeypatch before importing pymbar:
  ```python
  import numpy, scipy.linalg
  if not hasattr(scipy.linalg, "tril"): scipy.linalg.tril = numpy.tril
  ```
  Until then `adaptive_union_mbar_analysis` writes coverage only and runs no MBAR.
- **The §10 first action, made runnable.** η²-leverage and reweight-ratio for residual components
  1–6, from files already shipped: candidate scores in `cv_candidates.npz` (`resid`, 6 columns),
  CV1 and labels in `frame_metrics.npz`, state/λ assignment in `demux.npz`. η² = between-window
  variance of the component divided by its total variance, windows taken from `demux.npz['window']`.
  Reweight ratio follows the chignolin_6 harness in `RUNS/chignolin_6/cv2_modes/` — **remove each
  segment's mean first** (`seg`,`rep` in `demux.npz`), or you measure the restraint, not the dynamics.
- **Memories cited in §13 are not readable by fable/astra.** Everything load-bearing from them is
  inlined in this document; treat §13's memory list as provenance only.

## 13. File index

### Reports written this session
`RUNS/chignolin_7/adaptive_production/pmf_analysis/`
- `NATIVE_FOLD_AUDIT_2026-09-18.md` / `.docx` — the audit (12 pp, 21 tables, 2 figures)
- `ADVERSARIAL_REVIEW_2026-09-18.md` / `.docx` — 8 findings on the report, 8 on the run
- `AB_INITIO_FOLD_FINDING_2026-09-19.md` / `.docx` — method design, blind tests, demux (§8)

### Data and scripts — `pmf_analysis/native_fold_audit_results/`
`frame_metrics.npz` (611,536 × 11 metrics + provenance) · `native_reference.json` ·
`demux.npz` / `demux.json` / `demux_summary.txt` · `matched_rows.npz` · `frame_weights.npz` ·
`matched_pmf_rg.json` · `matched_pmf_d1d2.json` · `pca_all_frames.npz` / `.json` ·
`pca_analysis.json` · `auroc_fairness.json` · `cv_candidates.npz` · `cv_ranking.txt` /
`cv_ranking.json` / `cv_ranking_final.json` · `auto_select.txt` · `why_pc2.txt` ·
`q_decoy_null*.json` · `matched_null_corrected.json` · `blind_identification*.json` ·
`structures/native_like_top25.pdb` (+ `.json`) · `1uao_reference.pdb` ·
`pca_all_frames.png`, `pca_vs_cv1_discrimination.png` ·
scripts: `scan.py`, `report.py`, `where.py`, `extract.py`, `null.py`, `null2.py`, `null3.py`,
`null4.py`, `qtest.py`, `q_null.py`, `q_null2.py`, `pca_all.py`, `pca_analysis.py`, `pca_figure.py`,
`matched_rows.py`, `matched_pmf2.py`, `join_d1d2.py`, `d1d2_fes.py`, `demux.py`, `events.py`,
`blind_test.py`, `cv_hunt.py`, `build_docx.js`
Reviews: `board_review/`, `board_method_review/`, `small_board_review/`, `codex_review.log`,
`codex_method_review.log`

### Papers
`chignolin_knowledge_base/1-s2.0-S0014579306006028-main.pdf` (Satoh 2006, + `-gr5.jpg` = Fig 5)
`chignolin_knowledge_base/Non-Markov-Type_Analysis_and_Diffusion_Map_Analysi.pdf` (Fujisaki 2022)

### Configs
`aurum2:~/gareus/chignolin/chignolin.yaml` (c7) · `chignolin_8.yaml` (c8, deployed) ·
local `chignolin_8.yaml` · `RUNS/chignolin_7/effective_config.yaml` · `run_args.json`

### Relevant memories
`chignolin7-folded-state-not-identified`, `chignolin7-run-health-and-estimator`,
`chignolin7-lambda0-topups-ran-boosted`, `chignolin7-ladder-overlap-threshold`,
`chignolin-cv2-both-definitions-weak`, `ff19sb-opc-next-runs`, `aurum-remote-compute`

---

## 13a. Provenance limit

Every number in this document comes from **one campaign, on one peptide, with one force field**
(chignolin_7: `GYDPETGTWG`, ff14SB/TIP3P, 300 K). The CV rankings, the degeneracy measurements and
the folding-event counts are properties of that ensemble. Nothing here has been shown to transfer to
another peptide or force field — the census (§8, finding 8) explicitly measures that bank-fit CV axes
do *not* transfer across banks. Treat the method conclusions as hypotheses for the panel, not results.

## 14. Things that will mislead you if you skip them

1. `effective_config.yaml`'s `config_values` is **flat**; the real config is **nested schema 2.0**.
   Reconstructing from the flat list silently drops keys (`ap_min_samples_per_window`,
   `ap_max_new_windows`, seed-bank flags, `tui`).
2. **Steps are segment-local** — every segment restarts at 1,010,400.
3. **529 native "episodes" are not 529 folding events.** They are re-entries; the folding-event count
   is ~1–25.
4. **Do not read exchange acceptance 0.74–0.86 as "rungs too close."** Acceptance is a bulk pairwise
   statistic; MBAR overlap is tail-sensitive and globally normalised; the boost is heavy-tailed. For
   same-Hamiltonian λ states the physical energy cancels exactly in the swap criterion, leaving
   Δ = Δλ·ΔV — that is correct, not truncated.
5. **The λ=0 cross-check is low-power.** It reproduces the full-MBAR CV1 PMF to 0.289 kcal/mol
   (tolerance 0.5), but both estimates share the λ=0 samples, weak coupling partly *insulates* λ=0 so
   agreement is partly expected, the metric is unspecified as max or mean, and both rest on the same
   34.4 %-retained rung.
6. **Cartesian CA PC1's AUROC is circular** with an RMSD label. It is not evidence that PC1 is a good
   folding CV, and biasing on it needs a superposition reference that does not exist until after the run.
7. **GENPEPT as run is not fully ab initio** — `diversity_bank_preset: chignolin` encodes the
   target's fold class (§8). Use `broad` for any ab initio claim.
8. **Scoring CVs on folded/unfolded separation contradicts the project's standing criterion**
   (exploration + reweightability). It was used here because it was explicitly requested for this
   decision; §10 keeps the two modes separate so the distinction survives into the config.
