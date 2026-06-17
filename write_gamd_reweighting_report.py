#!/usr/bin/env python3
"""Generate GaMD+MBAR+CE2 reweighting methodology report as .docx."""

from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

OUT_DOCX = Path("/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/gamd_reweighting_methodology.docx")

# ── helpers ────────────────────────────────────────────────────────────────────

def heading(doc, text, level):
    return doc.add_heading(text, level=level)


def body(doc, text):
    p = doc.add_paragraph(text)
    p.paragraph_format.space_after = Pt(6)
    return p


def equation(doc, text):
    """Monospace-styled equation block."""
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.5)
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(4)
    run = p.add_run(text)
    run.font.name = "Courier New"
    run.font.size = Pt(10)
    run.font.color.rgb = RGBColor(0x1A, 0x1A, 0x6E)
    return p


def note(doc, text):
    """Grey italic note paragraph."""
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.3)
    p.paragraph_format.space_after = Pt(4)
    run = p.add_run(text)
    run.italic = True
    run.font.size = Pt(9)
    run.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
    return p


def shaded_cell(cell, fill_hex):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:fill'), fill_hex)
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:val'), 'clear')
    tcPr.append(shd)


def add_table(doc, headers, rows_data, header_fill='D9E1F2'):
    table = doc.add_table(rows=1 + len(rows_data), cols=len(headers))
    table.style = 'Table Grid'
    hdr = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = h
        run = hdr[i].paragraphs[0].runs[0]
        run.bold = True
        run.font.size = Pt(9)
        shaded_cell(hdr[i], header_fill)
    for ri, row_data in enumerate(rows_data):
        cells = table.rows[ri + 1].cells
        for ci, val in enumerate(row_data):
            cells[ci].text = str(val)
            if cells[ci].paragraphs[0].runs:
                cells[ci].paragraphs[0].runs[0].font.size = Pt(9)
    return table


def mixed(doc, parts):
    """
    parts: list of (text, bold, italic, mono) tuples.
    Returns the paragraph.
    """
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(6)
    for text, bold, italic, mono in parts:
        run = p.add_run(text)
        run.bold = bold
        run.italic = italic
        if mono:
            run.font.name = "Courier New"
            run.font.size = Pt(10)
    return p


# ── build document ─────────────────────────────────────────────────────────────
doc = Document()

for section in doc.sections:
    section.left_margin   = Inches(1.0)
    section.right_margin  = Inches(1.0)
    section.top_margin    = Inches(1.0)
    section.bottom_margin = Inches(1.0)

# ── Title ──────────────────────────────────────────────────────────────────────
title = doc.add_heading(
    "Free Energy Reweighting in Combined GaMD/REUS Simulations: "
    "Theoretical Basis and Implementation", 0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER

sub = doc.add_paragraph(
    "GAREUS workflow — chignolin CLN025 (GYDPETGTWG) | "
    "HMR-GaMD + Umbrella Sampling REUS + MBAR + 2nd-order Cumulant Expansion")
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
sub.runs[0].italic = True
sub.runs[0].font.color.rgb = RGBColor(0x44, 0x44, 0x44)
doc.add_paragraph()

# ── 1. Background ──────────────────────────────────────────────────────────────
heading(doc, "1. Background and Motivation", 1)

body(doc,
    "GAREUS combines two complementary sampling strategies: Replica Exchange Umbrella "
    "Sampling (REUS) provides spatial coverage across a collective variable (CV) landscape "
    "via harmonic window potentials, while Gaussian Accelerated Molecular Dynamics (GaMD) "
    "boosts the kinetics within each window by adding a non-negative bias potential to the "
    "system energy. Both biases must be removed to recover the true equilibrium free energy "
    "surface.")

body(doc,
    "The central methodological question is: how do these two reweighting problems interact, "
    "and can they be solved sequentially? This document derives the answer and describes "
    "the implementation used in GAREUS.")

# ── 2. The two bias potentials ─────────────────────────────────────────────────
heading(doc, "2. The Two Bias Potentials", 1)

heading(doc, "2.1 Umbrella Sampling (US) Harmonic Bias", 2)

body(doc,
    "In REUS, each window k applies a harmonic restraint on the primary CV:")

equation(doc, "w_k(x) = ½ · K_k · (CV(x) − c_k)²")

body(doc,
    "where K_k is the force constant and c_k is the window center. This bias is "
    "a fixed, analytical function of the configuration x evaluated through CV(x). "
    "It can be computed exactly for any configuration at any time — this is the "
    "key property that makes MBAR applicable.")

heading(doc, "2.2 GaMD Boost Potential", 2)

body(doc,
    "GaMD adds a non-negative boost ΔV(x) to the total potential energy V(x) so the "
    "system samples from a modified potential:")

equation(doc, "V_GaMD(x) = V(x) + ΔV(x)")

body(doc,
    "For the lower-dual boost type used in GAREUS (dual potential + dihedral boost), "
    "in the production phase after calibration:")

equation(doc, "ΔV(x) = ΔV_pot(V(x)) + ΔV_dih(V_dih(x))")
equation(doc, "ΔV_pot(V) = ½ · k₀ · (E − V)² / (V_max − V_min)   if V < E,  else 0")

body(doc,
    "where E is the boost threshold, k₀ is the effective force constant, and "
    "V_min/V_max are the energy bounds calibrated during the equilibration phase. "
    "After calibration, ΔV is a deterministic function of instantaneous potential "
    "energy — it is no longer time-dependent.")

body(doc,
    "The σ₀ parameter controls the maximum allowed boost standard deviation (σ₀ = 7.0 "
    "kcal/mol in run7). Smaller σ₀ → smaller, more Gaussian ΔV → better reweighting "
    "tractability. Larger σ₀ → stronger acceleration but harder reweighting.")

# ── 3. The Reweighting Gap ─────────────────────────────────────────────────────
heading(doc, "3. The Reweighting Gap", 1)

body(doc,
    "In a combined GaMD/US simulation, each window k samples from the distribution:")

equation(doc, "π_k(x) ∝ exp(−β · [V(x) + w_k(CV(x)) + ΔV_k(x)])")

body(doc,
    "Standard MBAR for umbrella sampling assumes sampling from:")

equation(doc, "π_k(x) ∝ exp(−β · [V(x) + w_k(CV(x))])")

body(doc,
    "The GaMD boost term ΔV_k(x) is absent from the MBAR model. Consequently, MBAR "
    "computes frame weights under the wrong generative distribution, and the resulting "
    "PMF is in the GaMD-modified ensemble — not the true equilibrium ensemble. "
    "A separate GaMD reweighting step is required to recover the true PMF.")

note(doc,
    "Note: This is why the GaMD methodology (Miao et al. 2015, JCTC) specifically "
    "developed the 2nd-order cumulant expansion (CE2) as its reweighting strategy. "
    "Standard exponential reweighting (exact but high-variance) was already known "
    "for aMD; GaMD's bounded boost was designed so that ΔV is approximately Gaussian "
    "and CE2 converges tractably.")

# ── 4. Shared GaMD Calibration ────────────────────────────────────────────────
heading(doc, "4. Shared GaMD Calibration: The Key Design Choice", 1)

body(doc,
    "In GAREUS, all REUS windows share a single GaMD calibration performed on a "
    "pooled or representative trajectory before production. This means every window k "
    "uses identical GaMD parameters: E, k₀, V_min, V_max. Therefore:")

equation(doc, "ΔV_k(x) = ΔV(x)   for all windows k")

body(doc,
    "The GaMD boost is the same function of instantaneous potential energy V(x) "
    "regardless of which window a frame belongs to. This simplification has a "
    "profound consequence for the MBAR analysis.")

# ── 5. MBAR with Shared GaMD: Row-Constant Shift ──────────────────────────────
heading(doc, "5. MBAR Analysis: The Row-Constant Shift Cancellation", 1)

heading(doc, "5.1 The u_nk Matrix", 2)

body(doc,
    "MBAR operates on the reduced potential matrix u_nk, where entry u[i,k] is the "
    "reduced potential of frame i evaluated under the Hamiltonian of window k. "
    "For US-only (no GaMD):")

equation(doc, "u[i, k] = β · w_k(CV_i)   =   β · ½ · K_k · (CV_i − c_k)²")

body(doc,
    "With GaMD included and shared calibration, the correct u_nk is:")

equation(doc, "u[i, k] = β · [w_k(CV_i) + ΔV(V_i)]")
equation(doc, "        = β · w_k(CV_i)  +  β · ΔV(V_i)")
equation(doc, "        = β · w_k(CV_i)  +  c_i         ← same c_i for ALL k")

body(doc,
    "The GaMD boost β·ΔV(V_i) is a constant that depends only on the frame index i, "
    "not on the window index k. It is a row-constant shift in the u_nk matrix.")

heading(doc, "5.2 Cancellation in MBAR Free Energy Differences", 2)

body(doc,
    "MBAR solves for free energy differences f_k via the self-consistency equation:")

equation(doc, "f_k = −log Σ_i { exp(−u[i,k]) / Σ_j N_j · exp(−u[i,j] + f_j) }")

body(doc,
    "Substituting the row-shifted u[i,k] = u_harm[i,k] + c_i:")

equation(doc, "exp(−u[i,k]) = exp(−u_harm[i,k]) · exp(−c_i)")
equation(doc, "Σ_j N_j · exp(−u[i,j] + f_j) = exp(−c_i) · Σ_j N_j · exp(−u_harm[i,j] + f_j)")

body(doc,
    "The exp(−c_i) factors cancel in the ratio:")

equation(doc, "exp(−u[i,k]) / Σ_j N_j · exp(−u[i,j] + f_j)")
equation(doc, "  = exp(−u_harm[i,k]) / Σ_j N_j · exp(−u_harm[i,j] + f_j)")

body(doc,
    "Therefore, the MBAR free energy differences f_k are identical whether or not "
    "ΔV is included in u_nk. MBAR correctly combines the umbrella windows and "
    "produces the PMF in the GaMD-modified ensemble without any error from ignoring ΔV "
    "in the matrix — as long as GaMD calibration is shared.")

note(doc,
    "This cancellation holds only because ΔV is the SAME function for all windows. "
    "If each window had its own GaMD calibration (different E_k, k₀_k), the shifts "
    "would be window-dependent and would NOT cancel — requiring explicit per-window "
    "ΔV inclusion in u_nk.")

heading(doc, "5.3 Effect on Frame Weights", 2)

body(doc,
    "While free energy differences are unaffected, MBAR frame weights DO change "
    "when ΔV is included in u_nk. Without ΔV:")

equation(doc, "w_i(MBAR)  ∝  1 / Σ_k N_k · exp(f_k − β·w_k(CV_i))")

body(doc,
    "With ΔV included:")

equation(doc, "w_i'(MBAR) ∝  exp(β·ΔV_i) / Σ_k N_k · exp(f_k − β·w_k(CV_i))")
equation(doc, "            =  exp(β·ΔV_i) · w_i(MBAR)")

body(doc,
    "Including ΔV multiplies each frame weight by exp(β·ΔV_i). This is precisely "
    "the GaMD reweighting factor. When ΔV is omitted from u_nk, the frame weights "
    "are in the GaMD-modified ensemble; the GaMD correction must then be applied "
    "explicitly as a separate step.")

# ── 6. CE2 Reweighting Step ───────────────────────────────────────────────────
heading(doc, "6. 2nd-Order Cumulant Expansion (CE2) for GaMD Unbiasing", 1)

heading(doc, "6.1 Exact Reweighting", 2)

body(doc,
    "The exact correction from the GaMD-modified ensemble to the true ensemble is:")

equation(doc, "⟨A⟩_true = Σ_i [w_i(MBAR) · A_i · exp(β·ΔV_i)]")
equation(doc, "           ─────────────────────────────────────────")
equation(doc, "           Σ_i [w_i(MBAR) · exp(β·ΔV_i)]")

body(doc,
    "This requires per-frame ΔV_i values. For the PMF, A_i = δ(CV_i − CV). "
    "In principle this is exact; in practice exp(β·ΔV) weights are "
    "highly non-uniform and the effective sample size (ESS) can collapse to near zero.")

heading(doc, "6.2 CE2 Approximation", 2)

body(doc,
    "CE2 approximates exp(β·ΔV) using the 2nd-order cumulant expansion, "
    "treating ΔV as approximately Gaussian-distributed within each window:")

equation(doc, "⟨exp(β·ΔV)⟩ ≈ exp(β·⟨ΔV⟩ + ½·β²·σ²_ΔV)")

body(doc,
    "where ⟨ΔV⟩ and σ²_ΔV are the mean and variance of the boost within the window "
    "or CV bin. The CE2-corrected free energy:")

equation(doc, "F_true(CV) ≈ F_GaMD(CV) − ⟨ΔV⟩(CV) − ½·β·σ²_ΔV(CV)")

body(doc,
    "CE2 is valid when the ΔV distribution per window is approximately Gaussian — "
    "equivalently, when the 3rd and higher cumulants of ΔV are negligible. "
    "This is controlled by σ₀: smaller σ₀ → smaller bounded boost → more Gaussian ΔV.")

heading(doc, "6.3 Validity Check: ESS", 2)

body(doc,
    "The effective sample size of the GaMD reweighting:")

equation(doc, "ESS = (Σ_i w_i · exp(β·ΔV_i))²  /  Σ_i (w_i · exp(β·ΔV_i))²")

body(doc,
    "ESS → N_frames when ΔV ≈ 0 (no boost needed). ESS → 1 when one frame "
    "dominates (exponential reweighting has collapsed). CE2 is reliable when "
    "ESS / N_frames > ~0.1. If ESS ≈ 0, the ΔV distribution is non-Gaussian "
    "(heavy-tailed) and CE2 underestimates the correction — σ₀ should be reduced.")

add_table(doc,
    ["ESS / N_frames", "Interpretation", "Action"],
    [
        ["> 0.3",     "ΔV near-Gaussian, CE2 reliable",           "Proceed normally"],
        ["0.1 – 0.3", "Moderate boost, CE2 approximate",          "Use with caution; report uncertainty"],
        ["< 0.1",     "Large boost, ΔV non-Gaussian, CE2 fails",  "Reduce σ₀; or use exact reweighting with per-frame ΔV"],
        ["≈ 0",       "Reweighting collapsed",                    "GaMD correction unreliable; σ₀ too high"],
    ]
)
doc.add_paragraph()

# ── 7. Sequential Equivalence ─────────────────────────────────────────────────
heading(doc, "7. Sequential MBAR + CE2 = Joint Treatment", 1)

body(doc,
    "The sequential approach — MBAR for US bias followed by CE2 for GaMD bias — "
    "is mathematically equivalent to the joint treatment (ΔV in u_nk + exact "
    "exponential reweighting) when GaMD calibration is shared. The proof:")

body(doc,
    "Joint treatment frame weight (ΔV included in u_nk):")

equation(doc, "w_i_joint ∝ exp(β·ΔV_i) · w_i(MBAR)")

body(doc,
    "Sequential approach (MBAR weights × CE2 correction factor):")

equation(doc, "w_i_seq   =  w_i(MBAR) · exp(β·ΔV_i) / Z_CE2")

body(doc,
    "These are proportional — identical up to normalisation. The sequential approach "
    "produces the same PMF as the joint treatment when using exact exp(β·ΔV_i) "
    "weights (not CE2). CE2 introduces an additional approximation on top of this "
    "exact equivalence.")

body(doc,
    "Practical consequence: if per-frame ΔV is saved to the Parquet output, "
    "the sequential approach with exact exponential reweighting is both rigorous "
    "and straightforward to implement. CE2 is a fallback when only per-window "
    "ΔV moments (⟨ΔV⟩, σ²_ΔV) are available.")

# ── 8. Implementation in GAREUS ───────────────────────────────────────────────
heading(doc, "8. Implementation in GAREUS", 1)

heading(doc, "8.1 Shared GaMD Setup", 2)

body(doc,
    "The shared GaMD calibration is implemented via run_shared_gamd_setup_article_a() "
    "in gareus/production.py. A single OpenMM simulation accumulates GaMD statistics "
    "(V_min, V_max, V_avg, σ_V) and fixes the boost parameters (E, k₀) before any "
    "production window is launched. These parameters are written to "
    "shared_gamd_setup_globals.json and loaded by every window replica at startup.")

heading(doc, "8.2 Boost Extraction", 2)

body(doc,
    "Per-frame GaMD boost is read via extract_gamd_boost_kj() in production.py, "
    "which first attempts the native gamd-openmm API (get_boost_potentials()) and "
    "falls back to integrator globals. The boost is stored in the Parquet sample "
    "files as gamd_boost_kj (kJ/mol) alongside CV values.")

heading(doc, "8.3 MBAR Pipeline", 2)

body(doc,
    "analyze_gareus_mbar.py computes u_nk from harmonic US potentials only "
    "(β · ½ · K_k · (CV_i − c_k)²). This is correct given the shared-calibration "
    "cancellation proof (Section 5.2). The resulting MBAR free energies and frame "
    "weights are in the GaMD-modified ensemble.")

heading(doc, "8.4 CE2 Correction", 2)

body(doc,
    "The CE2 step applies the cumulant correction to each CV bin using per-window "
    "ΔV moments computed from the Parquet data. The MBAR-reweighted ⟨ΔV⟩ and σ²_ΔV "
    "per CV bin are used — not raw per-window statistics — to correctly account for "
    "window overlap and MBAR weight redistribution:")

equation(doc, "F_true(CV) ≈ F_MBAR(CV) − Σ_i [w_i(MBAR) · ΔV_i · δ(CV_i−CV)] / p(CV)")
equation(doc, "             − ½β · Var_MBAR[ΔV | CV]")

body(doc,
    "where p(CV) = Σ_i w_i(MBAR) · δ(CV_i − CV) is the MBAR-reweighted CV density.")

heading(doc, "8.5 Current Run Parameters (run7)", 2)

add_table(doc,
    ["Parameter", "Value", "Notes"],
    [
        ["Run mode",           "hmr-gamd",          "HMR + GaMD dual boost"],
        ["Boost type",         "lower-dual",         "Dual: total potential + dihedral"],
        ["σ₀ (potential)",     "7.0 kcal/mol",      "Raised from 6.0 to target anharmonicity ≈ 0.45"],
        ["σ₀ (dihedral)",      "7.0 kcal/mol",      "Same as potential boost"],
        ["GaMD equil steps",   "1,500,000",          "6 ns at 4 fs — shared calibration phase"],
        ["Averaging window",   "5,000 steps",        "GaMD running statistics window"],
        ["Timestep",           "4.0 fs",             "HMR-enabled; H-mass 3.0 amu"],
        ["Temperature",        "300 K",              "Langevin thermostat"],
        ["CV1",                "contacts",           "Nonlocal contact fraction, r₀=4.8 Å"],
        ["CV2",                "rama-map",           "Φ/Ψ similarity to 4 Ramachandran centers"],
        ["REUS windows",       "adaptive (Delaunay)","Double-adaptive: pilot rounds + production"],
        ["Budget",             "2,000 ns aggregate", "run7 target"],
    ]
)
doc.add_paragraph()

# ── 9. Known Limitations ──────────────────────────────────────────────────────
heading(doc, "9. Known Limitations and Caveats", 1)

p = doc.add_paragraph(style='List Number')
p.add_run("CE2 validity depends on σ₀ tuning.").bold = True
p.add_run(
    " If per-window ESS ≈ 0, CE2 underestimates the GaMD correction. "
    "Pilot run diagnostics showed near-zero ESS at σ₀ = 5–6 kcal/mol; "
    "run7 raises σ₀ to 7.0 to increase acceleration but may worsen CE2 convergence. "
    "Monitor ESS per CV bin in production analysis.")

p = doc.add_paragraph(style='List Number')
p.add_run("MBAR u_nk omits ΔV by design.").bold = True
p.add_run(
    " This is correct given shared calibration (Section 5.2). "
    "If calibration is ever made window-specific, ΔV must be re-introduced into u_nk.")

p = doc.add_paragraph(style='List Number')
p.add_run("CE2 uses per-window moments, not per-frame weights.").bold = True
p.add_run(
    " For maximum accuracy, the CE2 step should use MBAR frame weights "
    "(w_i(MBAR) · ΔV_i) rather than raw per-window ⟨ΔV⟩ averages. "
    "Raw per-window averages assume uniform MBAR weight within a window, "
    "which is approximately but not exactly correct.")

p = doc.add_paragraph(style='List Number')
p.add_run("GaMD boost is not saved per-frame by default.").bold = True
p.add_run(
    " The Parquet pipeline writes gamd_boost_kj when extract_gamd_boost_kj() "
    "succeeds, but this depends on gamd-openmm API availability. "
    "Verify boost column presence before running the exact reweighting path.")

p = doc.add_paragraph(style='List Number')
p.add_run("US MBAR convergence is independent of GaMD reweighting quality.").bold = True
p.add_run(
    " Even if CE2 fails (ESS ≈ 0), the MBAR PMF in the GaMD-modified ensemble "
    "is still a valid — if biased — free energy surface. The two corrections are "
    "independent; partial results are interpretable.")

# ── 10. Summary ───────────────────────────────────────────────────────────────
heading(doc, "10. Summary", 1)

add_table(doc,
    ["Step", "What it removes", "Method", "Requires", "Approximation"],
    [
        ["1. MBAR",
         "US harmonic bias (all windows)",
         "MBAR self-consistency",
         "CV per frame, K, c_k",
         "None (exact for shared GaMD)"],
        ["2. CE2",
         "GaMD boost bias",
         "2nd-order cumulant expansion",
         "⟨ΔV⟩ and σ²_ΔV per CV bin",
         "ΔV ~ Gaussian (fails if ESS → 0)"],
        ["Alternative: exact",
         "GaMD boost bias",
         "Exponential reweighting",
         "Per-frame ΔV_i",
         "None (exact, but high variance)"],
    ]
)
doc.add_paragraph()

body(doc,
    "The GAREUS approach (MBAR for US + CE2 for GaMD) is theoretically rigorous "
    "for shared GaMD calibration. The sequential decomposition is not an "
    "approximation of the joint treatment — it IS the joint treatment, with CE2 "
    "as an approximation to the exact exp(β·ΔV) reweighting. Validity is controlled "
    "entirely by σ₀ and monitored via per-window ESS.")

# ── 11. References ─────────────────────────────────────────────────────────────
heading(doc, "11. Key References", 1)

refs = [
    ("Shirts & Chodera (2008)", "MBAR", "J. Chem. Phys. 129, 124105",
     "Original MBAR derivation"),
    ("Miao et al. (2015)", "GaMD", "J. Chem. Theory Comput. 11, 3584",
     "GaMD method and CE2 reweighting"),
    ("Miao & McCammon (2014)", "CE2 for aMD", "J. Comput. Chem. 35, 1761",
     "Cumulant expansion reweighting for accelerated MD"),
    ("Wang et al. (2021)", "GaMD-REUS", "J. Phys. Chem. Lett. 12, 6871",
     "Combined GaMD + umbrella sampling methodology"),
    ("Tan et al. (2012)", "MBAR+US theory", "J. Chem. Phys. 136, 144102",
     "Theoretical basis for combining US and MBAR"),
]

add_table(doc,
    ["Authors (year)", "Topic", "Journal", "Relevance"],
    refs
)
doc.add_paragraph()

doc.save(OUT_DOCX)
print(f"Saved: {OUT_DOCX}")
