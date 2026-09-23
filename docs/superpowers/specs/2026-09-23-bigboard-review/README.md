# Big-board review of ATLaS-MD v0.8.2 (main @ dfeb8f9), 2026-09-23 00:59-04:35

Seven boards were run: 5 judges each (glm, kimi, deepseek-thinking, mini, thinker), a debate round,
a chair (deepseek-thinking), and a math/physics veto (thinker). Each board is in `<board>/` next to this file
(`final_report.md`, `transcript.md`, `board_summary.json`). Report only; no code was changed.

**Caveats on the board output itself.**
- Two chairs (b4, b6) hit the model's output-token limit and never wrote a judgment. For those two
  boards the verdicts below come from the judges' own final-round VERDICT lines.
- Several judges read unshown code and asserted things about it. Every headline claim was checked
  against the code or the OpenMM runtime in this session; the "Checked" column gives the result.

## Verdicts

| Board | Subsystem | Board result | Split | Mean position-card cosine |
|---|---|---|---|---|
| b1 | Pep-GaMD integrator, boost hand-off, pricing | **ACCEPT-WITH-CHANGES**, confidence 75 | 3-2 (mini REJECT 92, glm REJECT 88) | 0.634 |
| b2 | Replica exchange | **ACCEPT-WITH-CHANGES**, confidence 74 | 3-2 (kimi REJECT 85, mini REJECT 92) | 0.611 |
| b3 | NPT biased-MC barostat | **REJECT**, confidence 78 | 3-2 (glm AWC 76, kimi AWC 80) | 0.597 |
| b4 | Analysis / MBAR | no chair judgment (truncated): glm REJECT, kimi AWC, deepseek AWC, mini REJECT, thinker n/a | split | 0.594 |
| b5 | Swarm, envelope, sidecar hand-off | **REJECT**, confidence 80 | 4-1 (deepseek AWC 70) | 0.612 |
| b6 | Checkpoint / resume / recalibration | no chair judgment (truncated): glm, kimi, deepseek, mini all REJECT | 4 REJECT (80 %) | 0.572 |
| b7 | Method as a whole | **ACCEPT-WITH-CHANGES: SOUND WITH CONDITIONS, value SIMPLIFY**, confidence 74 | 4-1 (mini REJECT 88) | 0.759 |

Consensus that holds across all boards: **the core mathematics is right.**
- The Pep-GaMD force algebra is the exact gradient of the boosted potential.
- `_channel_boost` matches the gamd-openmm lower-bound kernel.
- The gibbs-walk Metropolis-Hastings kernel satisfies detailed balance.
- MBAR over (window, rung) states with a λ=0 anchor is asymptotically exact.

The failures are **hand-offs and silent fallbacks around that core**, which is the same class as the
chignolin_8 stage-2 bug.

## Ranked findings, with the check done in this session

| # | Finding | Location | Free-energy impact | Checked |
|---|---|---|---|---|
| 1 | **The CV-selection result is overwritten by the seed selection.** The `selection` variable (CV-selection result; `None` unless `cv2: auto`) is reassigned at L715 to `select_window_seed_frames(...)`, which is never `None`. The 1-D sidecar branch at L740 then always writes `cvs = {cv2: "none"}`, and `apply_epoch0_sidecar` overwrites `args.secondary_cv`. | `gareus/swarm/analyze.py` L601/L715/L740 | A swarm campaign with a user-configured (non-auto) CV2 silently runs 1-D. The requested surface is not the one sampled. | **VERIFIED** (code read) |
| 2 | **No applied-vs-priced boost check at runtime.** `sample()` has both the integrator's applied boost and the priced `pep_gamd_boost_kj` in hand and never compares them. | `gareus/production.py` sample path (~L7588) | Exactly the chignolin_8 class. The stage-5 check (c73224f) now catches the stage cause; any other applied/priced divergence still passes silently. | **VERIFIED** (absence) |
| 3 | **NaN is treated as zero bias, but only in exchange.** `umbrella_bias_matrix_kcal` zeroes a non-finite secondary displacement. `_channel_boost` maps NaN energy to 0 boost via `np.where((b+v)<e, b, 0)`. MBAR (`reconstruct_bias_matrix`, `apply_ladder_boost_to_u`) propagates NaN instead. | `production.py` L914-916; `pep_gamd.py` `_channel_boost` | Exchange and MBAR price different Hamiltonians for the same frame. Reachable only when CV2 or boost arguments are non-finite (rare on the fast path). | **VERIFIED** (code) |
| 4 | **The frozen envelope and seed bank are not digested.** The epoch-0 marker digests directories only through `final_survivor_seeds.csv`; the seed bank has no such index, and `shared_gamd_setup/` holds only the globals JSON. | `gareus/swarm/epoch0.py` `_artefact_digests` | A modified or stale envelope passes the epoch-0 gate. Production then boosts with an envelope different from the one the ladder was designed on. | reported by b5, not re-checked here |
| 5 | **The FSF floor is reported but not gated,** and the forced top rung (`lambdas[-1] = 1.0` at `max_rungs`) is never re-checked for adjacent-rung acceptance. | `gareus/swarm/ladder_design.py` | The top rung can run with FSF near 0 or negative (the NaN mechanism seen in S3 attempts 6-7), or be exchange-disconnected, while MBAR still prices it. | reported by b5 |
| 6 | **Silent success defaults:** `done.get("status", "ok")` pools a member without a status as successful. | `swarm/analyze.py` `_load_members` | A crashed or partial member can enter the envelope fit. | reported by b5 |
| 7 | **The native boost reader swallows exceptions.** The fallback `infer_gamd_boost_kj_from_globals` then returns only the last "total"-like global (drops the dihedral channel of a dual boost; `boosted_energy_*` can match). | `production.py` L344-365, L382-409 | Recorded `gamd_boost_*` columns (used by non-ladder GaMD reweighting) can be wrong. Low reachability: gamd-openmm does implement `get_boost_potentials`. | **VERIFIED** (code); reachability low |
| 8 | **Resume can relabel states without validating the window/λ table.** The kernel-identity refusal can be bypassed. | `production.py` `load_production_checkpoint` | Resumed samples are attributed to wrong states. | reported by b6 (4 REJECT), **not re-checked** |
| 9 | **λ per state is inferred by `nanmedian`, not read from one authoritative source.** A NaN registry λ can silently disable ladder reweighting. | `mbar_analysis` loaders | Ladder states are mis-priced as λ=0. Already an open item in CLAUDE.md. | known |
| 10 | **Sample and exchange use different predicates for the secondary axis.** `sample()` keys on `secondary_cv_k_kcal_list`; `_current_exchange_arrays()` keys on `secondary_cv_ks_kj`. | `production.py` | Divergent Hamiltonians if these ever disagree. Currently derived from the same config. | reported by b2 |

### Claims the boards made that are refuted

- **b3 pillar 1** (majority REJECT): "`Simulation.currentStep` is advanced only by `Simulation.step()`,
  so the barostat schedule desyncs." **FALSE.** In OpenMM 8.5.1 (local) and 8.3.1 (aurum),
  `currentStep` is a property returning `context.getStepCount()`. Tested: after
  `integrator.step(7)`, `sim.currentStep == 7`. b3's REJECT therefore rests on its remaining,
  weaker pillars: force-group hygiene asserts, and boost activation inferred rather than verified.
- **b7 dissent** (mini): "the volume-move acceptance omitted the PV term." **FALSE.**
  `npt.py:274-276` is `-beta*(dU* + P dV) + N_mol ln(V_new/V_old)`.
- **b5**: the `anchor_partials` 1/norm factor was disputed within the board and left unresolved.

## Method verdict (b7, relayed)

> **SOUND WITH CONDITIONS / SIMPLIFY.** "The core that survives scrutiny is the MBAR estimator over the
> (window × rung) state set, with the λ = 0 rung providing an unboosted, cumulant-free anchor …
> However, the majority does not regard any current chignolin FES as trustworthy … For a 10-residue
> peptide that folds in microseconds, the Pep-GaMD + λ-ladder + custom-barostat + adaptive-epoch stack
> is unearned complexity. The minimal trustworthy version is sparse pseudo-2D US/REUS on
> non-degenerate CVs, a zero-k unboosted stack, MBAR, native NPT, and validation against a long
> unbiased reference."

b7 conditions:
- replace or augment the degenerate CV1;
- drop the unearned layers for chignolin;
- mandatory validation gates before quoting any number: overlap matrix, ESS, identical-rung
  consistency, PMF invariance under window removal, agreement with a ≥100 μs unbiased reference or
  the Lindorff-Larsen chignolin FES, and a known-answer benchmark in CI;
- runtime invariants: Hamiltonian echo, swap-energy replay, per-state bookkeeping;
- no chignolin FES from the current pipeline may be quoted until the above are met.

Dissent, mini (REJECT 88): "the method contains irreparable ensemble and reweighting errors". Its
central PV-term argument is refuted above.

Note: b7's "native NPT" recommendation is only valid **without** a boost. Under GaMD a native
barostat samples the wrong ensemble, which is why the biased-MC barostat exists.
