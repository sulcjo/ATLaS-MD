# The residual-to-contacts motion, in context

The residual PC1 axis (CV2) is the component of bank torsion space **linearly independent of the
nonlocal-contact compaction axis (CV1)**. What it is physically, energetically, and why the
literature says this is exactly the coordinate that matters.

## Measured (this analysis, `genpept_r7_residual_therm.json`, figure `figures/cv2_thermo_dssp_energy.png`)

Fixed-CV1 band (|CV1 − 0.481| ≤ 0.02, 117 seeds, 24 evenly sampled across CV2), OpenMM GBn2
(amber14 + implicit/gbn2, NoCutoff) single-point decomposition.
Anchor: seed-0 total −1572.57 vs generator CSV −1572.62 kJ/mol.

- **total GBn2 energy is flat along the axis** (ρ(cv2, total) = −0.05), while components trade:
  bond strain worsens (ρ = +0.38), GB solvation improves (ρ = −0.37), torsion (ρ = −0.26) and
  direct nonbonded (ρ = +0.28), angle (ρ = +0.26). The curl is a **compensated motion** — enthalpy
  is repartitioned between strain and solvation at ~constant total.
- Globally, curled (high-CV2) seeds have **slightly lower GBn2 minimized energy** (ρ = −0.31),
  uncorrelated with the clash-proxy energy drop (ρ = +0.06).
- **DSSP along the axis**: the bank is overwhelmingly all-coil (`CCCCCCCCCC` 188/197 low,
  68/197 high decile); at high CV2 short helix nuclei appear in residues 3–6
  (`CCHHHHCCCC` ×20, `CCCCCHHHHC` ×17). **No residue carries DSSP 'E' (strand) at any point**
  (per-residue E fractions ≤ 1.5%) — the implicit bank contains hairpin-like compactness but
  no hydrogen-bonded strand registers.
- Solvation descriptors: SASA ↓ slightly with curl (ρ = −0.20), Asp3 burial ↑ (ρ = +0.23),
  Tyr2 burial ↓ (ρ = −0.21 — the aromatic ring *unpacks* as the flank curls), hydrophobic
  fraction ~neutral (ρ = +0.28), Rg unchanged (ρ = −0.03).

## Literature context (verified via NCBI E-utilities)

1. **Turn-directed "broken-zipper" folding** — Enemark & Rajagopalan, *Phys. Chem. Chem. Phys.*
   2012, 14, 12442, doi:10.1039/c2cp40285h. Unbiased MD of chignolin: turn nucleation → cooperative
   turn growth → H-bond formation → hydrophobic-core packing *last*; cross-strand sidechain packing
   can be **rate-limiting** and produces misfolded states.
2. **C-terminal roll-up** — Enemark, Kurniawan & Rajagopalan, *Sci. Rep.* 2012, 2, 649,
   doi:10.1038/srep00649. Prefolding dynamics is a staged, topologically guided roll-up.
3. **Turn rearrangements are causally dominant** — Sobieraj & Setny, *J. Chem. Theory Comput.*
   2022, 18, 1936, doi:10.1021/acs.jctc.1c00945. Granger causality on a CLN025 (*same sequence
   GYDPETGTWG*) folding/unfolding trajectory: turn-region rearrangements drive folding/unfolding;
   arm–arm interactions score low.
4. **Near-degenerate native vs misfolded energies** — Maruyama & Mitsutake, *J. Phys. Chem. B*
   2018, 122, 3801, doi:10.1021/acs.jpcb.8b00288 (3D-RISM atomic decomposition): native ≈ misfolded
   (−171.1 vs −171.2 kcal/mol) with *different components*; Thr6–Thr8 sidechain interaction is
   important for the π-turn.
5. **Aromatic Y2–W9 packing organizes folding; refolding is μs** — Amado et al.,
   *J. Phys. Chem. B* 2024, 128, 4898, doi:10.1021/acs.jpcb.3c08271 (experiment): β-hairpin stability
   is driven by Tyr2–Trp9 aromatics and Tyr2–Pro4 CH–π; refolding after pH-jump = 1.15 μs,
   volume-expansion +10.4 mL/mol.

## Reading the axis in context

- **CV1 (contact fraction)** is the *zipping/compaction* coordinate — the part the literature finds
  mechanistically **secondary** (Sobieraj–Setny: arms score low) and thermodynamically well-behaved.
- **CV2 (residual) is the *turn-state* coordinate** — the part the literature singles out as the
  causal/nucleation DOF. Its flank-level content (ψ of T6/G7/T8 flipping while D3 stays pinned)
  is precisely Maruyama's π-turn region machinery; its curled endpoint collects helix-like turn
  nuclei (`CCHHHHCCCC`). The measured near-degeneracy of total energy along it (with compensated
  bond/GB fluxes) is the same near-degeneracy Maruyama reports between native and misfolded states:
  **this motion cannot be discriminated by total energy — a dedicated CV is required to see it.**
- **Entropy/solvation side:** the curl lightly unpacks the Y2–W9 aromatics (the experimentally
  organizing interaction) and buries Asp3 — i.e. it exchanges aromatic packing for turn
  pre-organization. Bank-wise the GBn2 filter mildly prefers curled states already
  (ρ = −0.31 on E_min); GB solvation models are known to favor compact/ionic arrangements,
  so treat that bank-side preference as a *generator bias*, not physics.
- **Design consequence for the campaign:** CV1 spreads *where the arms are*; CV2 spreads
  *turn pre-organization* — the two DOFs that folding kinetics says are limiting. The orthogonal
  decomposition found here (compaction vs turn curl) mirrors the accepted β-hairpin mechanism
  decomposition (zipping vs turn nucleation), independently arrived at from the seed bank.
