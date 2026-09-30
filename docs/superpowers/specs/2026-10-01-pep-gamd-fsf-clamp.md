# Pep-GaMD force-scaling-factor (FSF) clamp

Status: implemented on `feat/pep-gamd-fsf-clamp` (2026-10-01). Off by default.

## Problem

gamd-openmm's lower-bound boost, per channel c (Total = V_pep + ΔV_dih, Dihedral = V_dih):

    d   = E - V                      (E = threshold = Vmax)
    k   = k0 / (Vmax - Vmin)
    ΔV  = ½ k d²                     (only while V + ΔV < E)
    FSF = 1 + dΔV/dV = 1 - k d

The envelope is frozen for the whole campaign. A frame with V < Vmin at k0 → 1 gives
FSF < 0: the boosted forces reverse and push atoms uphill. In Pep-GaMD, FSF_Total scales
every peptide bonded, peptide-peptide and peptide-water force, so FSF_Total → 0 also removes
the peptide's bond and angle restoring forces. S3 pilot attempts 6-7 crashed this way
("Particle coordinate is NaN"). A stronger boost (larger σ0, k0 → 1) is unusable without a
guard.

## Design

Each channel gets a floor f ∈ [0, 1). Past the crossover the boost continues linearly with
the slope the harmonic had there, so FSF never drops below f:

    a   = 1 - f
    d_c = a / k                       (FSF(d_c) = f)
    ΔV  = ½ k min(d, d_c)² + a max(0, d - d_c)
    FSF = max(f, 1 - k d)

- ΔV is C¹: value and slope match at d_c. FSF is exactly 1 + dΔV/dV, so the applied force is
  the exact gradient of U + ΔV. The boosted ensemble is a legitimate Hamiltonian, and MBAR
  over the λ ladder stays exact provided every evaluator uses this same ΔV.
- For V ≥ E, the upstream guards (`step(E - (V + ΔV))`, `check_boost`) are unchanged:
  ΔV = 0 and FSF = 1. In the linear branch V + ΔV = f V + const rises with V and equals
  V_c + ΔV(V_c) < E at d_c, so the guard never trips there.
- k = 0 (λ = 0 rung) makes d_c infinite, so ΔV = 0 and FSF = 1, as before.
- Nothing is divided by k in the expression: d_c = a (Vmax - Vmin) / max(k0, 1e-300).
- Dependent dual: the Total channel reads V_pep + ΔV_dih(clamped), and the dihedral force is
  scaled by FSF_T · FSF_D. This is the same structure as before; each FSF is clamped on its
  own channel.

## One source of truth: the frozen envelope

Floors are integrator globals `fsf_floor_Total` / `fsf_floor_Dihedral`, and they are recorded
in `shared_gamd_setup_globals.json` `all_globals` next to k0/Vmax/Vmin. Consumers:

| Consumer | How it gets the floors |
|---|---|
| `PepGaMDLowerDualIntegrator` | constructor kwargs from args; overwritten by the envelope copy |
| swarm envelope writer (`write_envelope_setup_dir`) | args → `all_globals` |
| per-replica / resume copy (`set_integrator_globals_from_dict`) | copies by name; `reconcile_fsf_floors` fails closed on any mismatch |
| analysis + in-run MBAR (`PepGamdEnvelope` → `pep_gamd_boost_kj`) | `from_integrator_globals` / `from_json` |
| NPT barostat (`_npt_lower_bound_channel_boost`) | fresh snapshot of the integrator globals |
| swarm ladder report (`fsf_floor_per_rung`) | `max(f, 1 - λ k0max)` |

`reconcile_fsf_floors` raises when:
- the envelope carries a floor the integrator lacks, which would otherwise be silently skipped
  and leave the integrator unclamped while analysis clamps;
- the integrator is clamped but an envelope dict (it has `k0_<channel>`) lacks the floor;
- the two floors differ.

A frozen campaign therefore can't mix clamped and unclamped physics.

## Interface

- `--pep-gamd-fsf-floor-total F`, `--pep-gamd-fsf-floor-dihedral F`. YAML keys
  `pep_gamd_fsf_floor_total` / `pep_gamd_fsf_floor_dihedral` (generic dest mapping).
- Default unset means legacy: the integrator program, envelope and analysis are byte-identical
  to before.
- Setting either floor enables the clamp on both channels; an unset one defaults to 0.0, which
  only forbids force reversal.
- Valid only with `--gamd-boost-type pep-gamd-lower-dual`; 0 ≤ F < 1.
- The floors are recorded in the run manifest method settings.

## Verification

- Closed form: C¹ continuity at d_c, FSF = 1 + dΔV/dV (finite difference), legacy identity
  when unset, k0 = 0 → 0.
- Integrator vs closed form: BoostPotential/FSF globals in stage 5, clamped region forced.
- Force = −∇(U + ΔV) by finite difference on the real Pep-GaMD system in the clamped region.
- NPT adapter vs closed form.
- Envelope round trip; reconcile guard; CLI validation; YAML mapping; legacy byte identity.
