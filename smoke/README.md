# `smoke/` — ab initio peptide folding thermodynamics: the open problem

**Self-contained.** Everything needed to understand and continue this work is in this folder. No
other part of the repository, and no external memory, is required. Written for an external AI or
collaborator with no prior context.

Package assembled 2026-09-19 from the `chignolin_7` campaign. Parent project: **ATLaS-MD / GAREUS**
(`sulcjo/ATLaS-MD`), a peptide enhanced-sampling pipeline.

---

## 1. What we are trying to do

Obtain the **true thermodynamic population of a peptide's folded state at 300 K, ab initio** — from
sequence alone, with no native structure used anywhere in sampling or CV construction. The
experimental structure may be used *afterwards*, to validate; that is what the reference papers do.

Test system: **chignolin, `GYDPETGTWG`, PDB 1UAO** — a 10-residue designed β-hairpin, one of the
smallest peptides with a unique fold. Note this is the *original* chignolin, not the more stable
CLN025 variant (`YYDPETGTWY`, 2RVD/5AWL) used by many literature studies; do not compare numbers
across the two.

## 2. How the method works

Umbrella sampling along a collective variable **CV1**, crossed with a **GaMD boost λ ladder**, all
states coupled by Hamiltonian replica exchange, and combined with MBAR.

- **CV1** = reference-free nonlocal contact fraction: heavy atoms, |i−j| ≥ 4, r₀ = 12 Å, β = 3 Å⁻¹,
  normalized. 16 umbrella centres.
- **λ ladder** = 4 GaMD boost rungs (0, 0.232, 0.643, 1.0). 16 × 4 = **64 states**.
- Exchange every 400 steps (1.4 ps), gibbs-walk over the full state set.
- Seeds come from **GENPEPT**, a from-sequence conformer generator (§6).
- MBAR bias form (`gareus/query.py`), which is what makes heterogeneous states possible:
  `U_k = ½·k1_k·(cv1 − c1_k)² + ½·k2_k·(cv2 − c2_k)²`, per-state k and centre on both axes.

chignolin_7 ran ff14SB + TIP3P, 300 K, NPT, 3.5 fs with HMR, 0.8 nm cutoff, built from sequence
(`input_pdb = None`). **5,124,128 MBAR samples, 611,536 coordinate frames, 5.351 µs** of aggregate
walker time.

## 3. What went wrong

**The fold was sampled but never scored, and the campaign is short of converged populations by one
to two and a half orders of magnitude.**

Counting folding events the way the reference paper does (a walker must fully unfold, ≥ 4.0 Å, then
reach native, < 1.0 Å):

| definition | events | µs needed for the reference's 159 |
|---|---|---|
| Satoh-exact Cα | **1** | 851 |
| backbone res 2–9 | 4 | 213 |
| loose (≤ 1.5 Å) | 25 | 34 |

Satoh 2006 obtained **159** events from **180 ns** of multicanonical MD. We obtained ~1–25 from
5.351 µs.

**Root cause — the GaMD boost could not cross the barrier.** Measured boost: mean **2.08 kcal/mol**
(3.5 k_BT) against a σ₀ cap of 6.0 that never became active. The envelope (V_max) was calibrated
from **35 ps per window** of conventional MD, and the two-stage calibration's second stage
(`gamd_recon_boosted_iters`) was set to **0**, so it never ran. With E = V_max frozen from 35 ps of
basin fluctuation, the transition state almost certainly lies *above* E, giving **ΔV(TS) ≈ 0**: the
boost filled wells it had already sampled and did nothing at the barrier.

**Second problem — CV1 is degenerate.** At fixed CV1 the structure is not determined. In CV1's
native window (0.82–0.93), 58,670 frames span backbone RMSD 0.43–4.27 Å and **only 3.65 % are
native**. It also fails the other direction: **56.6 % of near-native frames fall outside that
window**. A PMF along CV1 therefore averages the fold together with ~25× more equally compact
non-native topologies.

**A caution that qualifies all of the above.** 16.2 % of frames came from three segments where λ=0
states accidentally ran fully boosted (a scheduler bug). Those segments hold **100 % of the
Q ≥ 0.80 frames and 86.3 % of the ≤ 1.0 Å frames**. The correctly configured 83.8 % of the campaign
produced 66 frames ≤ 1.0 Å and **zero** at Q ≥ 0.80. Those frames are real ff14SB configurations,
but they carry no MBAR weight (wrong ensemble, correctly dropped) — and their existence is the
strongest within-campaign evidence that a stronger boost is what finds the fold.

## 4. The open question — where you come in

We want a **second CV** to break CV1's degeneracy, added as new rows in the same replica-exchange
array (k1 = 0, k2 > 0, same 4 λ rungs). The design is a **circular torsion PCA residualized against
CV1**: take sin/cos of all 18 internal backbone φ/ψ (36 features), regress them on scalar CV1,
PCA the residuals, freeze the projection.

Computed on all 611,536 frames (`data/cv_candidates.npz`), scored by how well each separates folded
from unfolded:

| CV | AUROC (folded vs rest) |
|---|---|
| CV1 nonlocal contacts | 0.8591 |
| Rg (heavy) | 0.8384 |
| **torsion residual-vs-CV1 PC2** | **0.7891** (corr with CV1 = 0.0000) |
| nonlocal backbone H-bond count | 0.7617 |
| torsion plain PC1 | 0.6784 |
| torsion residual PC1 | 0.5119 (chance) |

**CV1 + residual PC2 together reach 0.9230**, versus 0.8591 for CV1 alone. Inside the degenerate CV1
window, residual PC2's low quartile is **11.00 % native (3.02×)** while its high quartile is
**0.00 %**. Residual PC1 is at chance *by construction*: residualizing against CV1 removes precisely
CV1's folding signal.

### ⚠️ The blocker

**PC2 was selected using the native structure.** No unsupervised rule tested recovers it:

| selection rule | picks |
|---|---|
| variance (explained variance ratio) | PC1 |
| conditional multimodality within CV1 bins | PC4 |
| slowness / implied timescale | PC1 |
| **folded-vs-unfolded AUROC (supervised)** | **PC2** |

Implied timescales span only 306–473 ps across all six components and
**correlation(timescale, AUROC) = −0.033**. The residual torsion spectrum is flat
(EVR 0.111 → 0.049). There is no natural "best" axis.

**So: constructing the CV is ab initio; choosing the component is not.** That is the unsolved
problem.

### What we want done, and the pre-registered rule

Compute **η²-leverage + reweight-ratio** (the project's own unsupervised criterion) for residual
components 1–6, from `data/cv_candidates.npz` (`resid`, 6 columns) + `data/demux.npz` (per-frame
state/window assignment) + `data/frame_metrics.npz` (CV1, labels). η² = between-window variance of
the component ÷ its total variance. **Remove each segment's mean first** (`seg`, `rep` in
`demux.npz`) or you measure the restraint, not the dynamics.

Committed in advance, so the answer cannot be rationalized after the fact:

- **If it picks PC2** → the 2-D ab initio design stands; use it.
- **If it picks anything else** (PC1 and PC4 are live) → no ab initio selector recovers a
  folding-informative second axis, and the correct conclusion is that **the 2-D ab initio CV design
  is falsified for this system**. Do not search for a better scorer. Route instead to **T-REMD
  (300–420 K) or multicanonical**, which obtain 300 K populations with *no CV at all*.

## 5. Reproduction targets

Two papers define what "correct" looks like. PDFs are not redistributed here; citations:

**Satoh, Shimizu, Nakamura, Terada, FEBS Lett 580 (2006) 3422–3426** — "Folding free-energy landscape
of a 10-residue mini-protein, chignolin". Amber parm99 + GB/SA implicit solvent, multicanonical MD,
180 ns at 700 K reweighted to 300 K. **159 folding events. Native cluster 25.1 % at 300 K**
(B 24.9 %, C 10.6 %, unfolded 39.4 %). Native H-bonds Asp3O–Gly7N 3.12 ± 0.27 Å, Asp3N–Thr8O
3.40 ± 0.77 Å; misfolded clusters form Asp3N–Gly7O instead (one-residue register shift). Fig 5B —
the free-energy contour in d(Asp3N–Gly7O) × d(Asp3N–Thr8O) — is the figure to reproduce.

**Fujisaki, Suetani, Maragliano, Mitsutake, Life 12 (2022) 1188** — "Non-Markov-Type Analysis and
Diffusion Map Analysis for Molecular Dynamics Trajectory of Chignolin at a High Temperature".
AMBER 11, **ff99SB + TIP3P**, 11,081 atoms, **420 K**, 750 ns plain MD. Three states F/M/I, MFPTs
**~10 ns**. Calls Asp3O–Gly7N the "good" CV, and finds the ψ of **glycine and threonine** most
correlated with the slow modes. A kinetics paper — it reports **no 300 K populations**.

Measured on the 18 NMR models of 1UAO (`data/native_reference.json`): Asp3O–Gly7N **2.84 ± 0.13 Å**,
Asp3N–Thr8O 3.57 ± 0.41 Å, Asp3N–Gly7O 6.73 ± 0.38 Å (not a native contact).
**The pipeline never measures Asp3O–Gly7N** — the coordinate both papers call the native one.
Cheap gap to close.

## 6. GENPEPT — the seed generator, and a caveat

GENPEPT builds a diverse conformer library from sequence (basin hopping → implicit minimization →
diversity selection). The chignolin r7 bank holds **1,970 conformers**, characterised across **298
CVs** in `genpept/` — independently re-derived and verified (`ADVERSARIAL_VERIFICATION.md`, some
results reproduced bitwise).

⚠️ **It is not fully ab initio as run.** The bank used `diversity_bank_preset=chignolin`, which the
CLI documents as biasing *"toward hairpin/turn-rich banks"*: versus the generic `broad` preset it
raises β+turn bank mass 0.55 → 0.65 and drops α 0.25 → 0.15. Chignolin is a β-hairpin, so a preset
named for the target upweights the target's fold class. Free energies are unaffected (it acts on
seed generation, not the Hamiltonian), but the *discovery* claim is weakened.
**Use `diversity_bank_preset: broad` for any ab initio claim.**

Census findings that matter most (`genpept/README.md` has all 18):
- Best CV1 = backbone-heavy or CA contact fraction at r₀ = 10 Å (entropy 0.973–0.980, 25–27
  resolvable windows). The run's all-heavy r₀ = 12 is statistically co-champion.
- Rg is a near-duplicate of mid-r₀ contact fractions (|r| ≈ 0.94–0.98) — pairing them adds nothing.
- Torsion PCA has a flat spectrum; residualization **rotates within one subspace**
  (|cos(PC1, resPC1)| = 0.166 but top-5 subspace overlap 0.986).
- **Bank-fit CV2 axes are not diffusive** (Hess cosine content ≈ 0.07) — a seed-bank torsion PCA is a
  *structure-diversity* coordinate, not a slow-motion one.
- **Bank axes do not transfer between banks** (per-trace correlation median −0.540, range
  [−0.817, +0.075]). Persist each bank's eigenvector with its own run.
- Intrinsic dimension ≈ 3.6–4.6 — the manifold is ~4-D, so a 2-D grid is a cover, not a
  parameterization.

## 7. What is in this folder

| path | content |
|---|---|
| `reports/01_native_fold_audit.md` | full audit of chignolin_7: was the fold found? |
| `reports/02_run_adversarial_review.md` | hostile review of the run and of report 01 |
| `reports/03_ab_initio_fold_finding.md` | method design, blind identification tests, demultiplexing |
| `reports/04_handoff_cv_selection.md` | **the deep reference** — everything, with all paths |
| `reports/05_handoff_adversarial_review.md` | hostile review of 04, with corrections applied |
| `reports/*.docx` | the two handoff documents in Word format |
| `genpept/` | the 298-CV census: 30+ analysis documents, JSON results, `two_cv_design.docx` |
| `data/cv_candidates.npz` | torsion PCA scores, plain and residualized, all 611,536 frames |
| `data/frame_metrics.npz` | per-frame CV1, Rg, RMSD, Q, H-bond distances, provenance |
| `data/demux.npz` | per-frame state/λ assignment from the exchange records |
| `data/native_reference.json` | 1UAO values for every metric, per model |
| `data/cv_ranking*.json`, `auto_select.txt`, `why_pc2.txt` | the CV comparison results |
| `data/native_like_top25.pdb` | the 25 best native-like frames, superposed on 1UAO |
| `data/1uao_reference.pdb` | the experimental reference (18 NMR models) |
| `scripts/` | every analysis script — all results here are reproducible from them |
| `config/chignolin_8.yaml` | the next run's config (calibration fix applied) |
| `config/chignolin_7_*` | exactly what the analysed campaign ran |

Key scripts: `cv_hunt.py` (builds the CV candidates), `demux.py` + `events.py` (exchange
demultiplexing and folding-event counting), `scan.py` (per-frame metrics), `pca_all.py`,
`blind_test.py` (unsupervised identification tests).

## 8. Facts that will mislead you if you skip them

1. **529 native "episodes" are not 529 folding events.** They are re-entries from partly-folded
   states; the folding-event count is ~1–25.
2. **Exchange acceptance of 0.74–0.86 does not mean "rungs too close."** Acceptance is a bulk
   pairwise statistic; MBAR overlap is tail-sensitive and globally normalized; the boost is
   heavy-tailed. For same-Hamiltonian λ states the physical energy cancels in the swap criterion,
   leaving Δ = Δλ·ΔV — that is correct, not truncated.
3. **Good λ mixing does not mean good MBAR overlap.** 42,724 round trips establish ergodic mixing;
   pairwise overlap sits at 0.065–0.15 against a 0.15 target. Both hold at once.
4. **Only one estimator is valid.** The GaMD boost enters the MBAR reduced potential directly, so
   `umbrella_only` is exact. `gamd_exponential` has effective sample size **51 of 4.58 M**;
   the cumulant expansions are invalid at anharmonicity 0.799. Their disagreement is a disclosure,
   not corroboration.
5. **Free energies here are not converged.** Every kcal/mol figure rests on ~1 folding event.
   The native-state Kish ESS is 6.9 at the tight threshold, with one frame carrying 33 % of the
   weight. Do not quote a folded population from this campaign.
6. **Trajectory-derived observables are forward-filled** across ~10 sample rows each, so their
   nominal 5.1 M support is really 611,536.
7. **Steps are segment-local** — every run segment restarts its counter at 1,010,400.
8. **Provenance limit:** one campaign, one peptide, one force field. The census explicitly measures
   that bank-fit CV axes do *not* transfer between banks. Treat everything here as hypotheses for a
   multi-peptide panel, not as general results.

## 9. If you only do one thing

Compute η²-leverage + reweight-ratio for residual components 1–6 (§4) and apply the pre-registered
rule. It decides whether this project has an ab initio second CV at all, and everything downstream
depends on the answer.
