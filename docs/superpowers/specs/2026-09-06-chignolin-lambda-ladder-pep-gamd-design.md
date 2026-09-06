# Chignolin GaREUS rebuild — λ-ladder × CV1 umbrella with Pep-GaMD

Date: 2026-09-06
Status: design approved in discussion; implementation plan not yet written
Supersedes: `2026-09-06-chignolin-gareus-epochal-rebuild-design.md` (tICA-CV2 route)
Depends on: branch `feat/pep-gamd-boost` (Pep-GaMD integrator + partition, built, unmerged)

## 0. The method's goals, and what this design does about each

| goal | mechanism here |
|---|---|
| complete conformational exploration driven by multi-boosted MD | a **ladder of GaMD boost strengths** λ ∈ [0, 1] exchanged as Hamiltonian replicas, crossed with a 16-window umbrella on heavy-atom contacts; the boost acts on the **peptide essential potential only** |
| exactly reweightable, folding recovered rather than imposed | MBAR over every (window, rung) state; `u_ik = β[V + λ_k ΔV_max(x) + w_i(x)]` is computable for every sample from stored per-sample energies; the target is (λ=0, w=0); the λ=0 rungs are plain umbrella sampling and give a built-in cross-check |
| thermodynamics | ΔF(CV1) from MBAR; ΔH(CV1) from `⟨V⟩` reweighted to the target; ΔS by subtraction; optional sparse temperature bracket at λ=0 |

No coordinate is fitted anywhere. Nothing can be redefined mid-campaign because nothing was defined from data.

## 1. Context

### 1.1 What failed in chignolin_6 (measured)

| defect | measurement |
|---|---|
| mid-campaign CV2 redefinition | `torsion-pca` → `tica-linear` centres are coordinates on different axes; the overlap graph split |
| CV1 too weak | `sidechain-all`: 12 residue pairs over 7 residues, all three glycines dropped incl. Gly7 in the PETGT turn |
| boost unreweightable | `sigma0 = 2.5` → `βσ = 3.57` → exact-estimator ESS `3.0e-6·N` = 2.9 of 940,788 samples |
| force constants clamped | all 37 windows at exactly the 200.0 cap |
| PMF root cause | `gamd_cumulant2` variance spike — the cumulant expansion is not perturbative at the measured `var(βΔV)` = 21–27 |

### 1.2 Why the tICA-CV2 route was dropped

- With 1 ns swarms the tICA lag is ≤ ~100 ps; tIC1 is then the slowest mode *visible on 1 ns* — turn/side-chain relaxation, not folding. Restraining a fast coordinate buys no barrier crossing, and it would have cost 7 of every 8 replicas.
- The best CV2 ever measured on this system had reweightability 2.71; "hidden slow modes remain whatever is chosen." A single restrained linear coordinate is the wrong tool class.
- A fitted coordinate needs a freeze gate, a re-gate loop confined to S1, and a provenance check against redefinition. A ladder needs none of that.

### 1.3 Why a single boost level was dropped

The databank (`gamd-boost-calibration`) records "accelerate vs. reweight can be simultaneously unsatisfiable" for one shared lower-bound boost on this peptide. A ladder decouples them: high-λ rungs accelerate, the λ=0 rung reweights exactly, MBAR weights the rest.

### 1.4 Why the boost is peptide-only

Stock gamd-openmm dual boosts bind the Total channel to bare `energy` (all 32 force groups — umbrella included, audit finding C1) and their velocity update applies only `f0` and the dihedral group's force, so a force moved to another group is *dropped*, not unboosted. Regrouping cannot fix that. And a system-total boost on a 19,008-atom box spends its σ budget on water (σ_V ≈ 140 kcal/mol, estimate). Pep-GaMD (Wang & Miao, *J. Chem. Phys.* 2020) boosts

```
V_pep = V_bonded(pep) + V_nb(pep–pep) + V_nb(pep–water)          water–water excluded
```

Peptide–water is included deliberately: hydrophobic collapse lives there.

### 1.5 Framing

Ab initio exploration. No structural knowledge of the target enters: no native reference, no RMSD-to-native, no folded seeding, no CV scored on folded/unfolded separation. A campaign that finds no compact basin has produced a result.

### 1.6 What survives from earlier design work

CV1 = heavy-atom nonlocal contacts, justified data-independently (28 pairs over all 10 residues vs 12 over 7; glycines kept). `contact_adaptive_max_k_kcal` 200 → 1200. Full Gibbs conditional over all states. `exchange_interval` 200 → 400. Stratified swarm seeding. The `box_audit` tautology fix (§9). r7 provenance notes (§11).

## 2. Architecture

| stage | what runs | products |
|---|---|---|
| **S0** seed prep | r7's 1,970 survivors binned on heavy-CV1 × Rg × E2E, equal quota per occupied cell | seed manifest, R replicates per cell |
| **S1** genesis swarm | N = cells × R unbiased trajectories × 1 ns, Pep-GaMD partition present, boost **off** | ① V_pep and V_dih envelopes (Vmax/Vmin/Vavg/σ_V per channel) ② heavy-CV1 distribution → 16 centres ③ CV1 curvature F″ → k(CV1) ④ ΔV_max distribution → rung spacing ⑤ seed frames for every (window, rung) |
| **S2** ladder build | 16 CV1 centres × M rungs λ ∈ {0, …, 1}, 1.5σ on CV1, acceptance-targeted on λ; MBAR-connectivity preflight | the state grid |
| **S3** pilot | one mid-range CV1 window × candidate rungs, a few ns each | acceptance per edge, ΔV per rung, σ_V(pep) vs σ_V(total), λ=0 CV1 autocorrelation vs no-exchange control, cost of the second PME |
| **S4** production | REUS over the full (i,k) grid with Pep-GaMD boost per rung; envelope frozen | trajectories, per-sample energies |
| **S5** tuning epochs | insert a rung or window where acceptance/overlap is weak; re-seed from swarms; never move an axis, never refit an envelope | extended grid |
| **S6** final | frozen grid, ≥ 70 % of budget | MBAR, PMF, cross-checks |

### 2.1 What the swarm is for now

Everything the tICA route needed it for, minus tICA: the boost envelope calibrated *before* any ladder exists (breaking the ladder → boost → ladder circularity), CV1 statistics, seeds — and now rung spacing, because the swarm's ΔV_max distribution is exactly the input to the exchange-acceptance estimate. The swarm runs with the auxiliary force present and the boost off, so its energies are the same quantities production will boost.

## 3. The λ ladder

### 3.1 Definition

For the lower-bound formula `threshold_energy = Vmax` **independent of k0**, and `k = k0/(Vmax−Vmin)` (`gareus/gamd_calibration.py:116-133`, reproducing gamd-openmm). So with a frozen envelope every rung's boost is a scalar rescale of one reference:

```
ΔV_c,k(x) = λ_k · ΔV_c,max(x),   ΔV_c,max(x) = ½ · k0_max/(Vmax_c − Vmin_c) · (Vmax_c − V_c(x))²  for V_c < Vmax_c, else 0
c ∈ {pep (Total channel), dih (Dihedral channel)}
u_ik(x) = β [ V(x) + λ_k (ΔV_pep,max(x) + ΔV_dih,max(x)) + w_i(CV1(x)) ]
```

One λ scales both channels; `k0_c,k = λ_k · k0_c,max`, thresholds and envelopes shared across rungs. `k0_max` comes from the S1 envelope at the campaign's `sigma0` (top rung = full GaMD). λ = 0 is plain umbrella.

### 3.2 Why it is exact

- ΔV is a potential-energy term (a function of x through V_pep, V_dih). Exchange between rungs is standard Hamiltonian RE; detailed balance holds.
- `u_ik` for **every** state is computable for **every** sample from `V(x)`, `V_pep(x)`, `V_dih(x)`, `CV1(x)`. No cumulant. MBAR is the estimator; `gamd_cumulant2` and its failure modes are irrelevant.
- λ=0 rungs are the databank's "statmech-sound configuration" running *inside* the campaign, fed conformations by exchange. Their ESS toward the target is ≈ 1 by construction.

### 3.3 Requirement on stored data

The realised boost at the sampling rung is not enough: for λ=0 rungs it is identically zero and ΔV_max is not recoverable from it. **Every sample stores the raw channel energies** `v_pep_kj_mol`, `v_dih_kj_mol` and the physical `potential_kj_mol` (group 1 excluded), alongside `cv1`. With the frozen (Vmax, Vmin, k0_max) per channel, `ΔV_max(x)` and hence every `u_ik` is reconstructible offline. Storing raw energies also protects against any later change of envelope convention.

### 3.4 Exchange

`gibbs-walk` over the full (i,k) grid — the full conditional, never a neighbour-truncated graph. The bias matrix gains the boost term: per replica, read `V_pep`, `V_dih` (two group-energy queries on the partitioned system — `boost_target_energy_kj` semantics) and compute `λ_k ΔV_max` for every k in O(1) each; add to the umbrella column. `boost_target_energy_kj(..., total_groups=total_energy_groups_for_args(args))` — the Total channel's groups are a property of the boost type, not of the stepping integrator.

### 3.5 Rung count and spacing (estimate, to be replaced by S3)

Acceptance between adjacent rungs needs `Δλ · β σ(ΔV_max) ≲ 1–1.5`. With a system-total boost the databank's `βσ_ΔV ≈ 7` gave Δλ ≲ 0.15, ~7 rungs. With Pep-GaMD σ_V(pep) is ~5–15 kcal/mol (estimate), so **M ≈ 4–5**, geometric spacing in λ, **64–80 replicas**. The S1 ΔV_max histogram gives the number; S3 measures it.

### 3.6 Envelope coverage

The envelope is frozen after S1. Production frames outside [Vmin, Vmax] do not break exactness (ΔV is still a defined function; the stored raw energies reconstruct it) — they only degrade acceleration (`V > Vmax` → boost off on that frame). Report the out-of-envelope fraction per rung as a diagnostic; do not re-calibrate.

## 4. Pep-GaMD (built on `feat/pep-gamd-boost`)

### 4.1 Partition

PME reciprocal space cannot be split by atom subset. An auxiliary water-only `NonbondedForce` (`PepGaMDWaterOnlyNonbonded`, force group 1; peptide q/ε and peptide exceptions zeroed; PME parameters pinned to the physical force's) gives

```
V_pep = energy0 − energy1 + energy2        (0: physical NB + bonds + angles, 2: torsions)
```

Verified by an independent oracle: the auxiliary reproduces a peptide-*deleted* water-only system's energy to < 0.05 kJ/mol.

### 4.2 Integrator

`PepGaMDLowerDualIntegrator` (subclass of gamd-openmm's dual `LowerBoundIntegrator`):

```
Total channel     StartingPotentialEnergy_Total    = energy0 − energy1 + energy2
Dihedral channel  StartingPotentialEnergy_Dihedral = energy2
applied force     (f0 − f1)·FSF_T + f2·FSF_T·FSF_D + f1 + (f − f0 − f1 − f2)
```

Water–water (`f1`) and every non-physical group (umbrella 31, secondary CV 29) are applied unscaled and excluded from every statistic. The integrator's own cMD stages are overridden likewise. OpenMM allows one force group per computation step, so each `fN`/`energyN` is copied into an intermediate first. Verified: statistics blind to a 5×10⁴ kJ/mol/nm² umbrella; cMD and k0=0 stages bit-identical against an all-zero auxiliary (with warm-up and motion assertions); force algebra at `FSF_T = 0.798, FSF_D = 0.775` matches the analytic expression to 1e-9.

### 4.3 Invariants the code enforces

1. The auxiliary force exists **only** in Systems handed to the Pep-GaMD integrator (added inside `make_gamd_integrator`, idempotent, found by name). A plain Langevin integrator built for such a System excludes group 1 (`make_cmd_integrator(system=)`), or water–water is counted twice.
2. Any `Custom*` force parked in groups 0–2 is rejected at build time.
3. Recorded potentials exclude group 1; recon and recalibration measure the Total channel by the boost type's definition.
4. `threshold_and_k0` strips the `pep-gamd-` prefix before its lower/upper dispatch.

### 4.4 Cost

One extra PME evaluation per step: ~+50–70 % per step (estimate; S3 measures it).

### 4.5 Two upstream quirks

- gamd-openmm's first `step()` of a fresh Context moves no atom. One-step comparisons must warm up, re-seat, then measure — and assert motion.
- OpenMM 8.5.1 CPU platform: once a Context holding two `NonbondedForce`s has been freed, every later `getPMEParametersInContext` in the process throws `std::bad_cast` (energies unaffected). PME parameters are therefore pinned from OpenMM's own formula (`α = √(−ln 2τ)/r_c`, `grid = ⌈2αL/(3τ^{1/5})⌉`), verified against a Reference-platform probe.

## 5. Swarm stage detail

- **Seeds:** `RUNS/chignolin_genpept_r7/final_survivor_seeds.csv` — 1,970 rows, the only set `load_genpept_conformer_library` reads. Survivor compactness is skewed (61/1,970 = 3.1 % at `contact_count ≥ 16`); stratification on heavy-CV1 × Rg × E2E with equal quota per occupied cell corrects the draw, not the ensemble.
- **Geometry:** 1 ns per seed after a separate equilibration stage; N = cells × R × 1 ns; distinct seeds first, then velocity replicates per seed once the 1,970 are exhausted.
- **Envelope:** `Vmax/Vmin/Vavg/σ_V` for both channels pooled over the equilibrated tail only. Extrema are the most transient-sensitive statistic; the discard length is read off the V-trace, not assumed. Pooling over a stratified (non-Boltzmann) ensemble inflates σ_V and so lowers `k0_max` for a given `sigma0`; under a ladder this only shifts where λ=1 sits, it does not bias any estimate.
- **Output interval:** fixed here, in ps *and* frames: 2 ps recommended.
- **Centring, tICA, freeze gate:** gone.

## 6. Analysis

- MBAR over all (i,k) with `u_ik` from §3.1; target (λ=0, w=0). Estimator selection collapses to MBAR; the `select_unbiased_method` gate is moot for this run type.
- **Cross-check (mandatory before quoting):** F(CV1) from λ=0 samples alone vs F(CV1) from the full ladder. Disagreement beyond bootstrap error means the boost reweighting is off.
- **2-D landscape without a 2-D bias:** any second coordinate — tICA on the λ=0 data included — may be fitted *after the fact* and F(CV1, ·) reweighted onto it. That is analysis, not bias; no freeze, no centring rule, no redefinition hazard.
- **Thermodynamics:** `ΔH(CV1) = ⟨V⟩(CV1) − ⟨V⟩(ref)` at the target via MBAR weights; `ΔS = (ΔH − ΔF)/T`. σ_V ≈ 140 kcal/mol per frame → ~1.8×10⁵ independent frames per bin for 1 kcal/mol at 3σ (bound). Optional: a sparse **temperature bracket at λ=0 only** (e.g. 3 temperatures, independent umbrella ladders, pooled in one MBAR via stored `V`) as a van't Hoff cross-check; gate on the databank's `mean_err/depth < 0.1` before any slope. A full T ladder as *exploration* axis is rejected (§10).

## 7. Gates (all label-free)

| gate | criterion |
|---|---|
| S3 → S4 | acceptance ≥ ~0.2 on every λ edge and every CV1 edge; λ=0 CV1 autocorrelation time drops vs the no-exchange control |
| S4 running | per-edge acceptance and per-window overlap stay above floor; out-of-envelope fraction reported |
| S6 quoting | λ=0-only vs full-ladder F(CV1) agree within bootstrap error; MBAR ESS toward target healthy (~27 % reference); ≥ 70 % of budget in the frozen final |

## 8. Code change surface

| # | change | status |
|---|---|---|
| 1 | Pep-GaMD integrator + partition + wiring + 22 tests | **done** on `feat/pep-gamd-boost`, reviewed, unmerged |
| 2 | rung dimension in the state registry (`gamd_lambda` column); per-replica `k0_Total`/`k0_Dihedral` override after the shared-globals copy (`production.py`, `_build_context_i`) | to do — the only structurally non-trivial change |
| 3 | exchange bias matrix gains `λ_k ΔV_max(x)`; needs `V_pep`, `V_dih` per replica at exchange time | to do, small |
| 4 | per-sample columns `v_pep_kj_mol`, `v_dih_kj_mol`; `sample_potential_energy` on, group 1 excluded | to do, small |
| 5 | MBAR loader: `u_nk += β λ_k ΔV_max` from stored raw energies + frozen envelope | to do, small |
| 6 | λ=0-only vs full-ladder cross-check in the analyzer | to do, small |
| 7 | swarm stage: stratified seed selection, replicate management, equilibration discard, envelope + CV1 stats + ΔV histogram | to do |
| 8 | rung-spacing estimator from the ΔV histogram | to do, small |
| 9 | `box_audit` guard fix (§9) | to do, trivial |
| 10 | provenance: boost type, envelope, λ per state, output interval in ps and frames | to do |
| — | tICA-from-swarm, freeze gate, CV2 source extension, uncertainty+frontier allocator | **deleted** from the plan |

## 9. The `box_audit` guard is a tautology (unchanged finding)

`system_setup.py:544` sizes `box = contour + 2·padding`; `:467` warns when `contour > box/2 − padding`, which reduces to `contour > contour/2` — true always. The correct test is `B − L ≥ cutoff`; for chignolin `5.82 − 3.82 = 2.0 nm ≥ 1.0 nm`, the box is adequate. Fix the guard; do not change padding.

## 10. Rejected alternatives

| rejected | why |
|---|---|
| tICA CV2 fitted on 1 ns swarms | fast-mode coordinate; 7/8 of replicas on an axis with no barrier-crossing power; needs freeze + gate + provenance |
| reuse chignolin_6's tICA | discarded; restraint-contaminated umbrella data |
| single shared boost level | accelerate-vs-reweight impasse (databank); cumulant reweighting |
| stock `lower-dual` | structurally entangled with the umbrella (energy = all groups; forces outside {0, dih} dropped) |
| `lower-dual-nonbonded-dihedral` | nonbonded channel 92 % exactly zero (databank) |
| system-total boost | water-dominated σ_V ≈ 140 kcal/mol |
| temperature ladder as the exploration axis | ΔT ≈ 1.3–2.5 K per rung at 19k atoms → 16–30 rungs per 40 K, 2–4× the λ ladder; envelope shifts with T |
| REST2 as the exploration axis | force scaling not in the codebase; intermediate states unphysical → no thermodynamics from them |
| native reference / folded seeding | ab initio framing; constructed initial conditions are Goodhart-gameable |
| recurring CV2 refits | coordinate redefinition — the chignolin_6 defect |
| neighbour-truncated exchange graph | truncates the Gibbs conditional |
| increase box padding | the alarm was a tautological guard |

## 11. Provenance notes on r7

Describe r7 from its own `GENPEPT_turbo_summary.json`, never from `chignolin.yaml` (edited after the bank was built; disagrees on six knobs). r7 predates the `contact_bias_sigma` fix (PR #68); its bias value is inside the working regime for a 10-mer.

## 12. Open items

1. **M and λ spacing** — from the S1 ΔV histogram, confirmed by S3.
2. **R** — needs the occupied-cell count from S0.
3. **Equilibration discard** for a 1 ns swarm — from the V-trace.
4. **Does dual peptide boost accelerate collapse enough?** — S3's λ=0 autocorrelation measurement decides; fallback is a stronger `sigma0` on the Total channel, not a different axis.
5. **Trajectory retention** — 45 % of chignolin_6's frames were unavailable to analysis; untraced.
6. **Stopping thresholds** for S6.
7. **Cost of the second PME** — S3 measures; if prohibitive, the only alternative that keeps the partition exact is fewer replicas, not a cheaper auxiliary.
8. Merge of `feat/pep-gamd-boost`; the authoritative test run through the sanctioned runner has not completed (it hung twice without launching a test process); 22/22 pass under a minimal runner.
