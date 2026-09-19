# Adversarial verification audit of GENPEPT_R7_CV_CENSUS_REPORT

Date: 2026-09-10. Verifier: opencode (local runner), independent re-derivation from raw
artifacts — no report number taken on trust. Method: re-run the deterministic scripts and
diff bitwise; recompute statistics from the values CSVs and bank PDBs; re-execute the GAREUS
oracle; grep every quantitative claim in the MDs against the shipped JSONs.

## What was re-derived and PASSED

### rep4 (T1–T6 bundle) — all claims verified
- Funnel: 12,000 → 11,935 → 1,132 → 9,745 exactly (`GENPEPT_turbo_summary.json`).
- **Comparison battery re-run: `genpept_rep4_vs_rep3.json`, `REP4_TUNED_COMPARISON.md`,
  `cv_rep4_vs_rep3.png` all reproduced BITWISE IDENTICAL.** Envelope 31 (0.31/0.32%),
  best RMSD 1.16/1.18 Å, entropy 0.978/0.979/0.939 vs 0.972/0.973/0.953, windows 26/27/23,
  chirality 0.540/0.502, q05 SASA 12.29/12.31 — all confirmed.
- T1: production log shows **480,112/800,000 accepted = 60.0%** (rep3 legacy baseline log:
  55,996/800,000 = 7.0% vs claimed 6.9% — rounding).
- T2: 295 parents, 110 `bh_parent_cluster` basins, per-basin 2–6 (cap respected).
- T4: `adaptive_register_threshold.json` — 86,400 N–O distances, `valley_detected: false`,
  fallback 3.8 Å, exactly as claimed.
- T5: final-bank frac(+) 0.502 (re-run), `selection_balancing.json` records 0 swaps.
- T6: `selection_balancing.json` — final q05 12.300 ≤ pool 12.448 × 1.1, 0 swaps.

### rep3 / r7 banks
- REP3_MASSIVE funnel table: all 10 numbers across both banks exact (incl. NMA 9,536/3,840).
- REP3_FES_FOLDED: all 14 audited numbers exact (best RMSDs, RMSD fractions, corners,
  medians, native envelope bounds).
- REP3_STATS: top-8 entropy set, champion row (0.982/0.506/28) exact.
- REP3_ADVANCED: TwoNN 5.12/5.35, stability medians (0.998/0.994/0.991), generator
  n's (6027/3156/547), dmap lambdas — all exact.
- REP3_ENERGY_TERMS: all six CV1 trend slopes, compensation 0.979, near-native deltas — exact.
- SOLVATION_AXIS: native GB −1553 [−1782,−1414], bank frac 0.5255, SASA frac 0.0003,
  near-native n=31 (−1525.0 vs −1554.6) — exact.

### r7 census core
- Values CSV: 1,970 × 298, 0 NaN. Stats recompute from raw values: exact to 4 decimals
  (entropy/median/IQR for c_bb_s4_r010_b3, c_heavy_s4_r012_b3, torpca_pc1, rg_heavy).
- Corr matrix: 298×298, symmetric, unit diagonal.
- Catalog counts: contact 175, resbal 50, rational 18, strand 10, geometry 18, hbond 4,
  torsion bins 9, torpca 5, res-torsion 3, SASA 5, reference 1 = 298. ✓
- README findings 2, 3, 5, 6, 9, 10 (legacy degeneracy, β saturation, Rg redundancy,
  H-bond degeneracy, rational < logistic, strand = disguised duplicate at |r| = 0.998) —
  all re-derived and confirmed.
- §3 top-15 spot rows and §4 orthogonality entries — confirmed.
- **Verification anchors re-executed:** vs `HEAVY_CONTACTS_PARAM_TEST_CHIGNOLIN_GENPEPT.txt`
  (mean .5435 / std .1969 / p05 .1789 / p95 .8246 — exact); **GAREUS oracle re-run**
  (`build_nonlocal_contact_pairs` + `nonlocal_contact_cv_from_positions_nm`, 150 seeds ×
  5 selections): max |Δ| = 1.76×10⁻⁷ ≤ claimed 1.8×10⁻⁷. PASS.

### Derived analyses
- DIFFMAP (TwoNN 3.60/4.64 ± SE; LASSO R² 0.974/0.937/0.826/0.985/0.647/0.936) — exact.
- FOURIER (R1–R4 per torsion incl. multimodal ψ rows; winding nulls 36.32/38.67/35.98/36.03
  with nothing flagged; Fisher-Lee top pairs −0.347/−0.340/−0.321…) — exact.
- TURN_AXIS (ρ(turn3537) = −0.544, ρ(strand) = −0.034, CV1-matched deciles 0.433/0.474,
  n = 197/197) — exact.
- MI_STATES (MI 0.089/null −0.020/excess +0.109; KDE states sum 1,970; bond component stats) — exact.
- RIGOR (bootstrap CIs, cis-omega 0.0030/0.0000, provenance strata, bias audit
  7.97/7.64/4.79 and all four frac pairs) — exact.
- DYNAMICS_VALIDATION (coverages 0.917/0.931/0.813 dyn, pooled means, t½ medians/max) — exact.
- PLAIN_PC2 (deciles, Spearman −0.050, C1 median 0.052, 0/32 ≥ 0.5, KDE modes, ψ rotations
  −173/−166/−149/−148/+115) — exact.
- ORTHOGONAL_PAIRS top rows — exact vs CSV.
- **CV2_DEEPDIVE: script re-run — `CV2_DEEPDIVE.md` and `bank_torsion_pca_cv2.json`
  reproduced BITWISE IDENTICAL** (deterministic).
- REP4_FES (this audit's own addition) re-derived from the bank.
- All 28 README-listed figures exist with non-trivial sizes; compiled HTML 19.0 MB as claimed.

## Discrepancies found (the adversarial yield)

1. **README §1b finding 13 is STALE.** It states per-trace corr(bank resPC1, stored
   secondary_cv) median −0.514, range [−0.80, +0.13]. The reproducible CV2_DEEPDIVE.md
   (re-derived bitwise this audit) says **median −0.540, range [−0.817, +0.075]** — same
   quantity, different numbers, neither JSON-persisted, but the re-run sides with
   CV2_DEEPDIVE.md. → README claim 13 should be updated to −0.540 [−0.817, +0.075].

2. **STEERABILITY_STABILITY.md half-bank column is partially unbacked.** The shipped
   `genpept_r7_steer_stability.json` contains only `pc1` (median 0.993, min 0.975 — matches
   MD row 1) and `res1` (median 0.960, min 0.561). The MD's residual-vs-bb row states
   "0.90 (0.02–0.56)" — contradicts the JSON's 0.960/0.561; the plain-PC2 row
   (0.983/0.944) and residual-combined row (0.746/0.02) have **no JSON fields at all**.
   The generator-split columns (0.979/0.752 and 0.964/0.887) match the JSON exactly, so the
   MD likely came from a richer console output that was never persisted. → half-bank cells
   for rows 2–4 are unverifiable from shipped artifacts, and row 3 conflicts with the JSON.

3. **T3's "94.5% of candidates carry overlays" is a smoke-2 number, not a rep4 production
   measurement.** The production rep4 bank's `tt_source` column is entirely NaN (the run
   predates the provenance fix); 94.5% (1,418/1,500, all 5 turn-type windows active) is
   measured on the post-fix smoke-2 bank. REP4_TUNED.md discloses the column drop, but the
   mechanism-outcome table reads as a production result. The mechanism is
   flag-deterministic so it certainly applied in rep4 — but the exact fraction on rep4's
   12,000 candidates cannot be re-measured from the bank.

4. **README finding 4 overstates two ways.** (a) "byte-identical"/"exactly identical" for
   `cresb_bb ≡ c_bb`: actual max |Δ| = 1.4×10⁻⁷ (float roundoff; REP3_STATS.md states this
   honestly as 1.48×10⁻⁷). (b) "|r| > 0.99 for every matched pair": **false for the `sch`
   selection** — |r| = 0.83–0.89 at r₀ ≤ 10 Å (near-degenerate, low-spread CVs; the claim
   holds for bb/ca/heavy/sca and for all champions).

5. **pack_y2 independence bound slightly understated.** README finding 10 says |r| ≤ 0.33
   vs contact CV1s; actual max is **0.353** (vs `c_sch_s4_r012_b3`). Off by 0.02 —
   conclusion (mildly independent) unaffected.

6. Cosmetic: rep4's `GENPEPT_turbo_summary.json` `reports` paths point to the original
   `/tmp/opencode/...` run location, not the persisted bank location. The rep3-legacy
   acceptance baseline is 7.0% in the log vs 6.9% claimed (rounding).

## Verdict

Every claim that is re-derivable from raw artifacts — funnel counts, T-mechanism outcomes,
the full rep3/rep4 comparison battery, census statistics, correlation structure, GAREUS
parity, diffusion maps, Fourier content, MI/energy decompositions, bootstrap CIs —
reproduces exactly or bitwise. The audit's genuine yield is one stale README number
(finding 1), one partially-unpersisted table (finding 2), one smoke-scale number presented
in a production context (finding 3), and two wording overstatements (findings 4–5). None
of these changes any scientific conclusion of the campaign.

---

# Round 2 (same day): determinism sweep, full oracle, REP2, fixes

## Full GAREUS oracle — PASS
All **1,970 seeds × 6 configs** (heavy r₀12 β3/β1.5, bb r010 β3, ca r010 β3, sch r012 β3,
sca r012 β3) re-executed against `build_nonlocal_contact_pairs` +
`nonlocal_contact_cv_from_positions_nm`: max |Δ| = **1.76×10⁻⁷** ≤ 1.8×10⁻⁷ claim. Closed.

## HTML census — PASS
Compiled report contains exactly **38 embedded base64 figures** (as claimed), 19.0 MB;
63 non-index figure files on disk (a superset).

## Determinism sweep (re-run of all bundle scripts vs shipped artifacts)
- **Bitwise identical after re-run:** `REP2_COMPARISON.md` + `genpept_r7_vs_rep2.json`
  (also closes the REP2 verification gap — bank symlinked to repo-root
  `chignolin_genpept_rep2/`); `genpept_r7_cv2_deepdive` outputs; `genpept_r7_cv_census`
  values/stats/corr + analyze figures; diffmap/fourier/turn_axis/pairs/plain_pc2/
  mi_states/rigor **JSONs**; `genpept_rep3_energy_terms.json`; dynamics_cv
  (`DYNAMICS_VALIDATION.md` + JSON); r7v2 figure PNGs; rep4 compare battery (round 1).
- **rep3 census values:** identical to ≤ 2.8×10⁻¹⁴ float noise on 10019×298, **except 3
  NaNs** newly appearing in `crat_heavy_s4_{n6m10,n6m12,n8m14}_r08` — the rational-switch
  form `(1−(r/r₀)ⁿ)/(1−(r/r₀)ᵐ)` has a 0/0 singularity when a pair distance lands exactly
  on r₀ (correct continuous limit: n/m). The shipped run happened not to hit it; the re-run
  did, and `genpept_rep3_stats.py` crashes on the NaN (no guard). One more concrete reason
  the campaign's "keep the logistic" verdict is right.
- **Stale/downsized scripts vs shipped artifacts (structural finding):** 4 shipped scripts
  cannot regenerate the shipped MD/JSON content they are credited with:
  `genpept_r7_steer_stability.py` writes neither the steerability table nor the stability
  table (see finding 2); `genpept_r7_rigor.py` drops the `contact_bias_audit` JSON key and
  MD section (numbers themselves verified vs the shipped JSON in round 1);
  `genpept_rep3_fes_folded.py` drops `native_envelope` (breaking downstream
  `genpept_rep3_solvation.py` when run against its own output);
  `genpept_rep3_advanced.py` fails outright (NameError `sys0`, line 213). The richer
  sections in the shipped MDs were authored after generation (hand-expanded or from
  fuller script versions in `/tmp/opencode`). The bundle is therefore **not one-command
  reproducible** despite README §8 — data artifacts all reproduce, prose extras do not.

## Fixes applied (this audit)
1. README finding 13: −0.514 [−0.80, +0.13] → **−0.540 [−0.817, +0.075]** (matches the
   bitwise-reproducible CV2_DEEPDIVE).
2. README finding 4: "byte-identical" → "identical to float roundoff (max |Δ| 1.4×10⁻⁷)";
   |r| claim qualified with the `sch` exception. §3 note updated likewise.
3. README finding 10: pack_y2 |r| ≤ 0.33 → ≤ 0.35.
4. REP4_TUNED.md T3 row: smoke-2 provenance now stated explicitly.
5. STEERABILITY_STABILITY.md: provenance caveat added naming exactly which cells are
   artifact-backed and which are console-transcribed.
6. All sweep-clobbered files restored to shipped state (verified by diff vs the
   pre-sweep snapshot); only the fixes above and this audit file are new.


---

## Round 3 (2026-09-11) — rep5/T7 verification: one CRITICAL bug found and fixed

Target: everything added since round 2 (T7 implementation, rep5 bank, REP5_T7.md,
report section 3.3). Method: independent re-derivation with different code paths
(hand Kabsch, scipy `Rotation.align_vectors`, per-file loops) rather than re-running
the shipped scripts.

### C1 (CRITICAL, FIXED): scrambled atom correspondence in all CA-RMSD-to-1UAO numbers

- **Symptom**: my independent Kabsch battery disagreed with the campaign's
  `md.rmsd` battery (rep4 best 0.79 vs 1.18 Å; frac<2.5 Å 5.45% vs 9.04%).
- **Root cause**: seed PDBs (OpenMM-written, N H H2 H3 CA ... order) and 1UAO
  (NMR, N CA C O ... order) have different atom orderings. Every campaign script
  called `md.rmsd(traj, native, frame=m, atom_indices=ca)` where `ca` holds the
  SEED's CA indices; mdtraj applies `atom_indices` to the reference too unless
  `ref_atom_indices` is given, so native CAs were paired with seed H/C/O atoms.
  Verified directly: same seed/model pair gives 1.178 Å (scrambled) vs 0.900 Å
  (correct); scipy `Rotation.align_vectors` confirms 0.9001 Å to 4 decimals.
- **Scope**: all absolute RMSD claims (rep3/rep4/rep5 best-RMSD, <t Å fractions,
  closest-seed rankings and overlay alignments, guided-expansion RMSDs).
  NOT affected: envelope occupancy (named-atom distances), register counts,
  CV/SASA/chirality metrics, thermodynamic layer. Comparative A/B verdicts
  survive (both sides measured identically).
- **Fix**: `ref_atom_indices=ca_native` added to all five shipped RMSD call sites
  (genpept_rep3_fes_folded.py, genpept_rep4_fes_folded.py, genpept_rep4_compare.py,
  genpept_rep4_guided_expansion.py, genpept_rep4_closest_native_overlay.py);
  affected analyses re-run; report and REP5_T7.md corrected with a dated note.
- **Corrected headline numbers**: rep3 best 1.01 Å (was 1.16), r7 best 1.61 (was
  1.64), rep4 best 0.79 Å (was 1.18; same champion seed, model 9 not 8),
  rep5 best 1.15 Å (was 1.30); closest-seed top-6 now 0.79/1.14/1.18/1.24/1.29/1.36 Å.

### C2 (minor, recovered): downsized rep3_fes_folded script clobbered native_envelope

Re-running the (known-stale, audit round 2 item) shipped `genpept_rep3_fes_folded.py`
regenerated `genpept_rep3_fes_folded.json` without the `native_envelope` key that
downstream scripts expect. Re-derived and re-added: raw 18-model d37/d38 bounds
± 0.5 Å padding = d37 [5.666, 8.063], d38 [2.336, 4.720] — matching the original
artifact exactly. The key now also records `"padding_A": 0.5` for provenance.

### Verified claims (independent re-derivation)

- rep5 envelope 39/9886 vs rep4 31/9745: confirmed exactly (Kabsch path, named atoms).
- Register enrichment 28.1%→35.2% (≥3) and 5.8%→9.5% (≥5): confirmed exactly.
- T7 provenance: 1,098 hop rows, 97.0% parents with pairs, mean 7.2, max 22: confirmed
  from basin_hop_minima.csv.
- Poisson sigma of the envelope gain: (39−31)/√31 = 1.44σ: confirmed.
- Smoke A/B: BH endpoint registers median 1.0 (control) vs 3.0 (T7): re-derived, confirmed.
- Bank integrity after the scripts/ → repo-root move: 9,886 PDBs, basename sets match CSV,
  energies finite [−1690, −904] kJ/mol, no leftovers.
- Guided-expansion negative result re-run with correct RMSD: still negative
  (registers 2.03 → 2.03 both arms; mean RMSD 1.58 → 1.99 both arms).

### New finding: A/B not perfectly clean at BH stage

rep5 materialized 11,939 viable implicit minima vs rep4's 11,935 and 285 vs 295 BH
parents — GPU implicit minimization is not bitwise reproducible run-to-run. The
rep4-vs-rep5 comparison is clean at proposal level (12,000 candidates, same seed)
with a small platform-noise floor at BH. Recorded in REP5_T7.md.

### Audit-of-the-audit note

My own first independent script contained two bugs (unit mix nm/Å in the native CA
stack; raw instead of ±0.5 Å-padded envelope bounds) and my first Kabsch used a
transposed rotation. Each was caught by cross-checking against a second independent
implementation (mdtraj superpose, scipy align_vectors, synthetic ground truth).
Final numbers are triple-confirmed.
