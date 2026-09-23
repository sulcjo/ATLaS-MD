# Thermodynamic energy decomposition in `gareus-analyze`

Date: 2026-09-23  
Status: phase 1 implemented on branch `feat/thermo-energy-decomposition` (see §9); phase 2 not started.  
Baseline: `main` at `b45e6c9` (code identical to `fa42346`, deployed on aurum2).

## 1. Goal

From the per-sample energies an exact-kernel Pep-GaMD campaign already records, compute the
enthalpy/entropy split of a basin-to-basin free-energy difference and the solvent share of it,
using the **same** unbiased MBAR weights that produce the PMF. No new MD, no production change.

Target: chignolin_9 once finished (`pep-gamd-lower-dual`, λ-ladder, `sample_potential_energy: true`).
Development/validation data: chignolin_8_US (same kernel type, never boosted, same columns).

## 2. Recorded inputs

Per sample (Parquet columns → `Data` fields):

| Column | Field | Definition (exact kernel) |
|---|---|---|
| `potential` | `potential_kj` | Groups {0..31} \ {1}: physical E0 + E2 **plus umbrella W (groups 29/31)**; boost excluded |
| `v_pep_kj_mol` | `v_pep_kj` | E0 − E1 + E2: everything physical except solvent–solvent nonbonded |
| `v_dih_kj_mol` | `v_dih_kj` | E2: physical torsions (all peptide) |
| window/state | `window` | generating state (own window) |
| — | `u_nk[n, window[n]]` | β·W_own, plus β·Δ(λ_own) on ladder runs (`apply_ladder_boost_to_u`) |

E1 is the auxiliary water-only `NonbondedForce` (`PepGaMDWaterOnlyNonbonded`, group 1): solvent and
ions, peptide charges/ε and peptide exceptions zeroed, PME parameters pinned.

## 3. Derived channels

```text
W_own  = u_nk[n, w_n]/β − Δ(λ_{w_n}; v_pep_n, v_dih_n, envelope)     # Δ = 0 off-ladder
U      = potential − W_own                  # physical potential energy
V_pep  = v_pep                              # peptide-involving energy: pp + pe + peptide bonded + torsions
U_ee   = U − V_pep = E1                     # solvent/ion–solvent/ion nonbonded (PME exact)
V_nt   = v_pep − v_dih                      # V_pep without torsions
```

`Δ` is recomputed with `gareus.pep_gamd.pep_gamd_boost_kj`, the same function `apply_ladder_boost_to_u`
uses, and the frozen envelope from `gareus.mbar_analysis.ladder.load_pep_gamd_envelope(d.prod_dir)`
(the loader's own path).

## 4. Thermodynamic quantities

Weights `w_n` = `norm_logw(m["logw"])` over the full population (same array as `analyze_rg` and the 2D FES;
target = unbiased physical state). On adaptive campaigns the headline CV1 PMF excludes epoch_000
(`d_main`); this stage does not, because on ladder/unboosted runs the global weights are exact for every
epoch. ΔG here can therefore differ slightly from a ΔG read off the headline PMF. Basins are half-open [LO, HI).
Basin population `P_A = Σ_{n∈A} w_n`; conditional mean `⟨X⟩_A = Σ_{n∈A} w_n X_n / P_A`.

| Quantity | Formula | Meaning |
|---|---|---|
| ΔG_AB | −kT ln(P_B/P_A) | Free-energy difference, same weights as the PMF |
| ΔH_AB | ⟨U⟩_B − ⟨U⟩_A | Enthalpy difference; kinetic energy cancels at fixed T; PΔV (1 bar × ~0.1 nm³ ≈ 6e-3 J/mol) neglected and stated |
| −TΔS_AB | ΔG − ΔH | Total entropy term at one temperature |
| ΔV_pep, ΔU_ee, ΔV_dih, ΔV_nt | conditional-mean differences | Exact additive split of ΔH: ΔH = ΔV_pep + ΔU_ee |
| −TΔS′_AB | ΔG − ΔV_pep | Entropy term **excluding solvent reorganization** (derived, see below) |

**Solvent-reorganization identity** (Yu & Karplus 1988; Ben-Naim; Lazaridis): at fixed solute
configuration the solvent–solvent reorganization energy enters the solvation enthalpy and T·(solvation
entropy) identically and cancels in ΔG. Hence ΔU_ee = TΔS_reorg and ΔG = ΔV_pep − TΔS′. −TΔS′ is
**derived from that identity, not measured**. It contains peptide conformational entropy and the
uncompensated part of solvation entropy. The identity is exact for NVT; NPT adds PΔV, which is
negligible here.

Along CV1: per-bin conditional means ⟨U⟩(s), ⟨V_pep⟩(s), ⟨U_ee⟩(s), ⟨V_dih⟩(s), reported relative to
the bin of lowest PMF, plus the PMF itself.

## 5. Preconditions (fail-closed)

Channels degrade independently. The report always says which channels are available and why not.

| Channel group | Requires | Else |
|---|---|---|
| ΔG, ΔH, −TΔS | `potential` finite; W_own recoverable (off-ladder, or ladder with envelope + finite v_pep/v_dih); weights valid (see below) | `available: false` with reason |
| ΔV_pep/ΔU_ee/ΔV_dih/−TΔS′ | additionally boost type **exactly** `pep-gamd-lower-dual` (resolved from `run_manifest.json`) and finite `v_pep`, `v_dih` | split `available: false`. `pep-gamd-lower-dual-rs` (planned), stock boosts and unknown types are refused: their `v_pep` is not E0−E1+E2 |

**Weights policy.** λ-ladder runs (`meta["gamd_ladder"]`) and unboosted runs (boost identically zero or
absent) use `logw` directly. Stock GaMD without a ladder is **unsupported**: per-sample exponential
reweighting has near-zero ESS at σ(U) ≈ 500 kJ/mol, and cumulant methods provide no per-sample weights.

**Force inventory.** `U_ee = potential − W_own − v_pep` is valid only if `potential` contains nothing
but physical forces and the umbrella of the sample's own window. Audited in §8. The module checks at
run time that W_own ≥ −tol (a harmonic bias cannot be negative) and reports the per-state median of
β·W_own (≈ 1 for a 2-DOF harmonic bias). A broken ladder subtraction shows up as negative W_own or a
median that trends with λ.

**Basins.** Headline basin numbers need explicit `--thermo-basin NAME:LO:HI` (CV1 units). Without them
an `auto` split is made at the highest PMF barrier (≥ 1 kT above the higher minimum) between the two deepest minima and labelled
`auto` (it need not correspond to the folded state on a contact CV). One minimum only → profiles only.

**Sampling gate per basin.** Weight ESS ≥ `--thermo-min-basin-ess` (default 1000) and ≥
`--thermo-min-basin-blocks` (default 8) distinct (epoch, replica) blocks. Failing basins make every
difference that involves them `status: inconclusive`; the numbers are still written, flagged.

## 6. Uncertainty

Paired fixed-f_k block bootstrap (same convention as `--pmf-uncertainty`; blocks from
`_sample_block_ids`: one replica within one epoch/phase). Per-block sufficient statistics
(Σw, Σw·X per basin/bin/channel) are precomputed once; each replicate draws one multinomial count
vector over blocks and applies it to every basin, bin and channel, and ΔG, ΔH, −TΔS are computed inside
the replicate. SEs are never combined across separately bootstrapped quantities. Default 200
replicates, seeded. Reported: point estimate, bootstrap SE, 2.5/97.5 % percentiles.

Precision expectation: σ(U) ≈ 500 kJ/mol per sample (chignolin_8_US, per window), so ΔH to ±5 kJ/mol
needs ~2·10⁴ effectively independent samples per basin. Peptide-only channels (σ ≈ 140) converge
~13× faster than U_ee.

## 7. Outputs

`<out>/thermo_decomposition/`:

- `thermo_basins.csv`: per basin population, ESS, blocks, ⟨X⟩ ± SE per channel.
- `thermo_differences.csv`: per basin pair ΔG, ΔH, −TΔS, ΔV_pep, ΔU_ee, ΔV_dih, ΔV_nt, −TΔS′, each with
  SE and 95 % interval, status.
- `thermo_cv_profiles.csv`: per CV1 bin the PMF and ⟨X⟩ − ⟨X⟩_ref ± SE.
- `thermo_cv_profiles.png`: profiles and PMF.
- `thermo_summary.json`, merged into `pmf_summary.json["thermo_decomposition"]`.

Everything in kJ/mol, with kcal/mol columns for the three headline terms. Report-only: does not change
the health verdict.

CLI (`analyze_gareus_mbar.py` / `gareus-analyze`): `--no-thermo-decomposition`,
`--thermo-basin NAME:LO:HI` (repeatable), `--thermo-bins`, `--thermo-bootstrap N`, `--thermo-seed`,
`--thermo-min-basin-ess`, `--thermo-min-basin-blocks`.

## 8. Audit results (baseline code)

Read-only code audits at `b45e6c9`, run before implementation.

### 8.1 Force inventory (production System, `pep-gamd-lower-dual`)

| Force | Group | Energy | Added at |
|---|---|---|---|
| HarmonicBond/HarmonicAngle, physical NonbondedForce (PME), CMMotionRemover | 0 | yes (CMM: no) | `system_setup.py:472` (`createSystem`), grouped by `assign_pep_gamd_force_groups` |
| MonteCarloBarostat (native backend only; biased-MC omits it) | 0 | no | `system_setup.py:483` |
| `PepGaMDWaterOnlyNonbonded` | 1 | yes (E1) | `pep_gamd.py:183` |
| PeriodicTorsion, CMAPTorsion | 2 | yes | `createSystem`, regrouped |
| Secondary-CV CustomCVForce (CV2; plus CV1 under `SHARED_CONTACT_LAYOUT`) | 29 | yes (W) | `production.py` CV builders |
| Primary-CV umbrella CustomCVForce (split layout only) | 31 | yes (W) | `forces.py:60,102` |
| Position restraints (CustomExternalForce) | 30 | equilibration only, absent from the production System | `system_setup.py:917` |

Umbrella force expressions are the plain harmonic terms MBAR uses (no walls, wrapping or extra
scaling); `contact_norm` and the residual-CV clamp appear identically in force and recorded CV.
`assign_pep_gamd_force_groups` rejects unexpected Custom forces in groups 0..2. Hence
`potential = E0 + E2 + W_own` exactly, and U_ee = potential − W_own − v_pep = E1.

Assumption stated rather than checked at run time: rigid water with constraints (TIP3P; OpenMM omits
constrained bonds from HarmonicBondForce), so no solvent intramolecular energy sits in E0 and hence in
V_pep. A flexible-water model would move that energy from U_ee into V_pep.

### 8.2 Sample-row timing

`sample()` runs after `step_all(chunk)` and before `attempt_exchanges` (`production.py` ~8294–8308).
Each replica's `_fetch_state` reads, on one Context with no step, barostat move or swap in between:
CVs, `potential` (`physical_energy_groups_for_args` = all groups except 1, so W is included), then
`v_dih` and `v_pep` (`_fetch_v_pep_v_dih`). `window_id` is the pre-exchange assignment, the window the
sample was generated in. The exchange path's own v_pep reads are never written to sample rows.
Verdict: same logical state, so the per-sample channel algebra is valid.
`gamd_boost_total_kj_mol` is an integrator global from the last force evaluation; it is not used here.

## 9. Phases

**Phase 1 (this branch):** §3–§7 from recorded columns only. Module `gareus/mbar_analysis/thermo.py`
(pure; imports only `gareus.*`), wiring in `analyze_gareus_mbar.py`, tests
`tests/test_thermo_decomposition.py`.

**Phase 1 real-data check (chignolin_8_US, 2026-09-23).** `analyze_gareus_mbar.py RUNS/chignolin_8_US
--analysis-source parquet --selected-method umbrella_only --thermo-basin low:0.10:0.35 --thermo-basin
high:0.65:0.90` (convergence/Rg/PCA/extra/adaptive-diag stages off). 481,616 samples, 0 dropped, 248 blocks,
weights `mbar_unboosted`, both basins pass the gate (ESS 132k / 124k, 173 / 164 blocks). W_own: no negative
values, worst per-state median β·W_own 3.1 kT (matches the run's self-bias report).

| low → high contacts | kJ/mol (± bootstrap SE) |
|---|---|
| ΔG | 1.18 ± 0.33 (flat CV1 PMF, consistent) |
| ΔH | 1.0 ± 6.9 |
| ΔV_pep | +130.3 ± 12.3 |
| ΔU_ee | −129.3 ± 13.4 |
| ΔV_dih | +0.95 ± 0.82 |

The peptide loses ~130 kJ/mol of peptide-involving energy between the two contact basins and the
solvent recovers it almost exactly (the compensation §4 predicts); ΔH itself is zero within error.
This basin pair is a contact-fraction split on an unfolded ensemble (chignolin_8 never sampled the
native hairpin), not a folding free energy.

**Phase 2:** implemented in §11 (frame-based pp/pe split and configurational entropy). The electrostatic
linear-response estimate and the items below need solvent coordinates and are not reachable from
solute-only output:

- ΔG_solv,el ≈ ½Δ⟨U_pe,el⟩ (linear response): needs the Coulomb part of U_pe alone; V_pe from solute
  frames is Coulomb + LJ + dispersion-correction difference. Reported as unavailable with that reason.
- ΔCp (needs several temperatures); GIST/2PT and per-residue pe (need solvent coordinates;
  chignolin_9 writes `traj_solute_only: true`).

## 11. Phase 2: frame-based peptide–peptide / peptide–environment split and entropy

Opt-in `--thermo-frames` (one PME evaluation per frame; `--thermo-frame-stride N`,
`--thermo-frame-platform auto|CUDA|OpenCL|CPU`, `--thermo-entropy-bins`, `--thermo-no-entropy`).
Modules `gareus/mbar_analysis/thermo_frames.py` (OpenMM, trajectories) and `thermo_entropy.py` (numpy).

**Split.** Ewald/PME is a quadratic form in the charges and LJ with geometric-mean ε pairs is too, so
at one α and one grid NB = NB_pp + NB_pe + NB_ee exactly. A System of only the peptide atoms, same box,
α and grid pinned to the physical force (production's `_pin_pme_parameters`: tolerance formula on the
**solvated topology's** default box, `01_solvated_start.pdb`; for chignolin_8 α = 3.648 nm⁻¹, grid 90³),
dispersion correction off, gives per frame

```text
V_pp      = NB_pp + bonded_pep + E_dih          (peptide with its own periodic images)
V_pe      = v_pep − V_pp                         (NB_pe + dispersion-correction difference)
```

A term coupling peptide and non-peptide atoms, parameter offsets, or an unknown physical force is
refused. Only peptide coordinates and the per-frame box are needed (solute-only XTC).

**Frame alignment.** Samples are keyed by (epoch source, replica, step); frame step = round(time/dt)
from the XTC; a later resume segment overrides an earlier one for the same step; stride keeps
(step / traj_interval) mod N = 0. The existing `analyze_gareus_mbar._sample_to_segment_frame`
(first frame assumed at R + interval) is off by one frame for resume segments whose start is not a
multiple of the interval (3 of chignolin_8's 5 segments) and is **not** used. Oracle: the offline torsion
energy E_dih must match the recorded `v_dih_kj_mol` of the matched sample (median |Δ| ≤ 2 kJ/mol); a
misaligned frame differs by ~σ(v_dih). Failure refuses the whole frame block.

**Entropy.** Backbone φ/ψ (from the rebuilt topology), weighted histograms per basin, first-order
(S1 = Σ H_i) and second-order mutual-information expansion (S2 = S1 − Σ_{i<j} I_ij); bin-width constants
cancel in basin differences. TΔS_conf = kT·ΔS2/k. Solvent (and every non-backbone-torsion) entropy by
difference: −TΔS_solv = −TΔS + TΔS_conf; −TΔS′_solv = −TΔS′ + TΔS_conf (the part of solvation entropy
not compensated by ΔU_ee). Excludes side-chain torsions, bond/angle vibrations and higher MI orders;
histogram entropies are biased low at small ESS.

**Statistics.** The frame block is one `compute_thermo` call on the samples that have frames: every
quantity (ΔG, ΔH, ΔV_pp, ΔV_pe, ΔU_ee, TΔS_conf, −TΔS_solv) is computed on that one population, with
one paired block bootstrap (entropy replicates reweight samples by their block's multinomial count).
The phase-1 block (all samples) is reported separately.

**Validation (tests/test_thermo_frames_openmm.py, OpenMM Reference, solvated GA dipeptide):**

- builder = force-field-built water-deleted peptide System (NB, bonded, torsion to 1e-6 kJ/mol);
- builder Coulomb = pp term from charge scaling of the peptide inside the full system
  (E(s) = s²pp + s·pe + ee), to 1e-4 kJ/mol: PME bilinearity at the pinned grid;
- production `v_pep` (partitioned copy) − V_pp = pe(Coulomb, charge scaling) + pe(LJ, ε scaling) +
  dispersion-correction difference, to 2e-3 kJ/mol;
- real XTCs: base + overlapping resume segment starting at 1900 (off-multiple): per-sample energies equal
  direct evaluation of the resume frame, a frame-less sample stays NaN, stride keeps exactly the
  intended samples;
- coupling term refused; MIE entropy analytic cases (uniform, duplicated torsion, single bin);
  entropy and V_pe identities inside compute_thermo and its paired replicates.

**Real-data check (chignolin_8_US, 2026-09-23):** `--thermo-frames --thermo-frame-stride 10`, OpenCL on an
RTX 3060 Ti, whole analyzer run ≈ 4 min. 1,240 XTC segments, 481,616 frames read, all matched to a
sample (0 unmatched), 48,112 evaluated. PME pinned α = 3.6480 nm⁻¹, grid 90³ (from the 5.82 nm topology
box). Alignment oracle: median |E_dih − v_dih| = 0.73 kJ/mol, p99 2.9 kJ/mol (XTC rounding). 18 backbone
torsions, 24 bins; basins ESS 13.2k / 12.4k, 161 / 160 blocks.

| low → high contacts | kJ/mol (± bootstrap SE) |
|---|---|
| ΔG | 1.18 ± 0.35 |
| ΔH | 2.4 ± 6.8 |
| ΔV_pp (NB 165 ± 13 of it) | −154.3 ± 13.0 |
| ΔV_pe | +284.5 ± 25.6 |
| ΔU_ee | −127.8 ± 14.2 |
| TΔS_conf, MIE-1 / MIE-2 | +3.1 ± 0.8 / −10.1 ± 3.2 |
| −TΔS_solv (MIE-2) | −11.3 ± 7.1 |

Forming contacts trades peptide–water interactions (+284) for peptide–peptide (−154) and water–water
(−128) ones; ΔH nets to zero. Backbone conformational entropy drops once torsion correlations are
counted (MIE-1 and MIE-2 disagree in sign, so the second-order term is essential and higher orders may
matter); solvent entropy rises by about as much, a hydrophobic-like signature at 1.6σ only. Histogram MI
is biased upward by ~(cells − 1)/(2·ESS) per pair (≈ 3 nats summed here), similar in both basins because
their ESS is similar, so it largely cancels in the difference.

## 10. Caveats

- Energies, not free energies: only ΔG is a state function; component "free energies" are not unique.
- `V_pep` carries the difference of the two dispersion corrections, which depends on volume under NPT.
  U_ee includes ions.
- ff14SB/TIP3P enthalpies and entropies are qualitative against experiment (TIP3P's heat capacity and
  temperature response are off); relative trends are the robust output.
- Fixed-f_k bootstrap ignores f_k uncertainty, as `--pmf-uncertainty` does.
