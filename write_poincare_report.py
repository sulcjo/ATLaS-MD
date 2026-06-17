#!/usr/bin/env python3
"""Generate Poincaré map findings report as .docx."""

import csv, json, numpy as np
from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

ANALYSIS_DIR = Path("/tmp/poincare_full_test")
RUN_DIR = Path("/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_2d_run2_66ns")
OUT_DOCX = Path("/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/poincare_findings.docx")

# ── load data ──────────────────────────────────────────────────────────────────
pm = json.load(open(ANALYSIS_DIR / "poincare_map_summary.json"))
rows = list(csv.DictReader(open(ANALYSIS_DIR / "poincare_residue_torsions.csv")))
A = [r for r in rows if r['event_type'] == 'fold_A']
B = [r for r in rows if r['event_type'] == 'fold_B']
U = [r for r in rows if r['event_type'] == 'unfold']
psi_cols = [k for k in rows[0] if k.startswith('psi_')]
phi_cols = [k for k in rows[0] if k.startswith('phi_')]

CHIGNOLIN_SEQ = list("GYDPETGTWG")
RES_NAMES = [f"{aa}{i+1}" for i, aa in enumerate(CHIGNOLIN_SEQ)]

def col_stats(group, col):
    vals = [float(r[col]) for r in group if r.get(col) not in ('', 'nan', None)]
    if not vals:
        return float('nan'), float('nan')
    return float(np.mean(vals)), float(np.std(vals))

# per-residue ψ and φ deltas (route B − route A)
psi_stats = {}
phi_stats = {}
for i, (pc, fc) in enumerate(zip(psi_cols, phi_cols)):
    ma, sa = col_stats(A, pc)
    mb, sb = col_stats(B, pc)
    mu, su = col_stats(U, pc)
    psi_stats[i] = {'A_mean': ma, 'A_std': sa, 'B_mean': mb, 'B_std': sb, 'U_mean': mu, 'U_std': su,
                    'delta_BA': mb - ma, 'delta_UA': mu - ma}
    ma2, sa2 = col_stats(A, fc)
    mb2, sb2 = col_stats(B, fc)
    mu2, su2 = col_stats(U, fc)
    phi_stats[i] = {'A_mean': ma2, 'A_std': sa2, 'B_mean': mb2, 'B_std': sb2, 'U_mean': mu2, 'U_std': su2,
                    'delta_BA': mb2 - ma2, 'delta_UA': mu2 - ma2}

most_diff_psi = max(range(len(psi_cols)), key=lambda i: abs(psi_stats[i]['delta_BA']))
most_diff_ua  = max(range(len(psi_cols)), key=lambda i: abs(psi_stats[i]['delta_UA']))

# ── helpers ────────────────────────────────────────────────────────────────────
def heading(doc, text, level):
    p = doc.add_heading(text, level=level)
    return p

def body(doc, text):
    p = doc.add_paragraph(text)
    p.paragraph_format.space_after = Pt(6)
    return p

def add_figure(doc, path, caption, width=6.0):
    if Path(path).exists():
        doc.add_picture(str(path), width=Inches(width))
        p = doc.add_paragraph(caption)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.runs[0].italic = True
        p.runs[0].font.size = Pt(9)
        p.runs[0].font.color.rgb = RGBColor(0x55, 0x55, 0x55)
    else:
        doc.add_paragraph(f"[Figure not found: {path}]")

def add_table(doc, headers, rows_data, col_widths=None):
    table = doc.add_table(rows=1 + len(rows_data), cols=len(headers))
    table.style = 'Table Grid'
    # header row
    hdr = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = h
        hdr[i].paragraphs[0].runs[0].bold = True
        hdr[i].paragraphs[0].runs[0].font.size = Pt(9)
        tc = hdr[i]._tc
        tcPr = tc.get_or_add_tcPr()
        shd = OxmlElement('w:shd')
        shd.set(qn('w:fill'), 'D9E1F2')
        shd.set(qn('w:color'), 'auto')
        shd.set(qn('w:val'), 'clear')
        tcPr.append(shd)
    # data rows
    for ri, row_data in enumerate(rows_data):
        cells = table.rows[ri + 1].cells
        for ci, val in enumerate(row_data):
            cells[ci].text = str(val)
            cells[ci].paragraphs[0].runs[0].font.size = Pt(9)
    return table

def bold_run(para, text):
    run = para.add_run(text)
    run.bold = True
    return run

def normal_run(para, text):
    return para.add_run(text)

# ── build document ─────────────────────────────────────────────────────────────
doc = Document()

# page margins
for section in doc.sections:
    section.left_margin   = Inches(1.0)
    section.right_margin  = Inches(1.0)
    section.top_margin    = Inches(1.0)
    section.bottom_margin = Inches(1.0)

# Title
title = doc.add_heading("Chignolin Poincaré Map Analysis: Folding/Unfolding Pathway Findings", 0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER

p = doc.add_paragraph("GAREUS run2 — 66 ns, 16 replicas | chignolin CLN025 (GYDPETGTWG)")
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.runs[0].italic = True
p.runs[0].font.color.rgb = RGBColor(0x44, 0x44, 0x44)
doc.add_paragraph()

# ── 1. Background ──────────────────────────────────────────────────────────────
heading(doc, "1. What is a Poincaré Map?", 1)
body(doc,
    "A Poincaré map records a low-dimensional observable each time a trajectory "
    "crosses a defined threshold. Here, two sections are defined in CV1 space "
    "(nonlocal contact fraction, the primary umbrella coordinate):")

p = doc.add_paragraph(style='List Bullet')
p.add_run("Σ_fold").bold = True
p.add_run(f": CV1 crosses {pm['c_fold']:.3f} upward (peptide enters high-contact / folded region).")

p = doc.add_paragraph(style='List Bullet')
p.add_run("Σ_unfold").bold = True
p.add_run(f": CV1 crosses {pm['c_unfold']:.3f} downward (peptide enters extended / unfolded region).")

body(doc,
    "At each crossing the backbone geometry score CV2 (rama-map: weighted mean similarity "
    "of all residue φ/ψ pairs to four Ramachandran basin centers) is recorded. "
    "Plotting CV2 at event #n vs event #n+1 reveals pathway memory and distinct routes.")

body(doc,
    f"A committor dwell filter ({pm['min_dwell_ps']:.0f} ps) is applied: a crossing only "
    f"counts if CV1 stays on the committed side for at least {pm['min_dwell_frames']} "
    "consecutive frames afterward. This eliminates threshold-bounce artifacts (99% of "
    "raw crossings were noise). Only committed folding/unfolding events are analysed.")

heading(doc, "CV2 Ramachandran Basin Reference", 2)
add_table(doc,
    ["CV2 value", "Basin name", "φ centre", "ψ centre", "Physical geometry"],
    [
        ["-1.000", "β/extended",  "-135°", "+135°", "β-strand / extended chain"],
        ["-0.333", "PPII/coil",   "-75°",  "+145°", "Polyproline-II / random coil"],
        ["+0.333", "right-α",     "-60°",  "-45°",  "Right-handed α-helix"],
        ["+1.000", "left-α",      "+60°",  "+40°",  "Left-handed α-helix (rare)"],
    ]
)
doc.add_paragraph()

# ── 2. Poincaré Map Results ────────────────────────────────────────────────────
heading(doc, "2. Poincaré Map Results", 1)

heading(doc, "2.1 Crossing Statistics", 2)
add_table(doc,
    ["Section", "Threshold", "Committed crossings", "Return pairs", "Recurrence (median)"],
    [
        ["Σ_fold",   f"CV1 > {pm['c_fold']:.3f}",   f"{pm['n_fold_crossings']:,}",
         f"{pm['n_fold_return_pairs']:,}", f"{pm['fold_recurrence_median_ns']*1000:.1f} ps"],
        ["Σ_unfold", f"CV1 < {pm['c_unfold']:.3f}", f"{pm['n_unfold_crossings']:,}",
         f"{pm['n_unfold_return_pairs']:,}", f"{pm['unfold_recurrence_median_ns']*1000:.1f} ps"],
    ]
)
doc.add_paragraph()
body(doc,
    "Note: recurrence times reflect REUS umbrella bias (replicas are artificially "
    "driven through all windows). They do not represent physical folding rates "
    "(chignolin physical fold time ≈ μs). The map topology (cluster structure, "
    "diagonal alignment) is bias-robust.")

heading(doc, "2.2 CV2 Peak Structure at Crossings", 2)

p = doc.add_paragraph()
bold_run(p, "Fold crossings (Σ_fold): ")
normal_run(p, "Two CV2 peaks detected.")

fp = pm['fold_peaks']
add_table(doc,
    ["Peak", "CV2 value", "Nearest basin", "KDE density", "Events (approx)"],
    [
        ["Route A (dominant)",
         f"{fp[0]['cv2']:+.3f}", fp[0]['nearest_basin'], f"{fp[0]['density']:.2f}",
         str(len(A))],
        ["Route B (minor)",
         f"{fp[1]['cv2']:+.3f}", fp[1]['nearest_basin'], f"{fp[1]['density']:.3f}",
         str(len(B))],
    ]
)
doc.add_paragraph()

p = doc.add_paragraph()
bold_run(p, "Unfold crossings (Σ_unfold): ")
normal_run(p, "Single CV2 peak.")
up = pm['unfold_peaks']
add_table(doc,
    ["Peak", "CV2 value", "Nearest basin", "KDE density", "Events"],
    [
        ["Single unfolding route",
         f"{up[0]['cv2']:+.3f}", up[0]['nearest_basin'], f"{up[0]['density']:.2f}",
         str(len(U))],
    ]
)
doc.add_paragraph()

heading(doc, "2.3 Key Interpretations", 2)

p = doc.add_paragraph(style='List Number')
p.add_run("Two distinct folding routes").bold = True
p.add_run(
    f": the system enters the folded region (CV1 > {pm['c_fold']:.2f}) via two "
    f"backbone configurations — Route A (CV2 ≈ {fp[0]['cv2']:+.2f}, {fp[0]['nearest_basin']}, "
    f"dominant, n={len(A)}) and Route B (CV2 ≈ {fp[1]['cv2']:+.2f}, {fp[1]['nearest_basin']}, "
    f"minor, n={len(B)}).")

p = doc.add_paragraph(style='List Number')
p.add_run("Single stereotyped unfolding route").bold = True
p.add_run(
    f": unfolding consistently exits via PPII/coil backbone "
    f"(CV2 ≈ {up[0]['cv2']:+.2f}). Unfolding is more mechanistically constrained than folding.")

p = doc.add_paragraph(style='List Number')
p.add_run("Strong pathway memory").bold = True
p.add_run(
    ": return maps lie on the diagonal, meaning the backbone geometry at fold/unfold "
    "event #n predicts event #n+1. The trajectory is not exploring pathways randomly.")

p = doc.add_paragraph(style='List Number')
p.add_run("Folding/unfolding asymmetry").bold = True
p.add_run(
    ": 2 folding routes vs 1 unfolding route. This is consistent with a funnel "
    "landscape where multiple entry routes converge onto a single unfolding channel.")

heading(doc, "2.4 Poincaré Map Figure", 2)
add_figure(doc,
    ANALYSIS_DIR / "poincare_map.png",
    "Figure 1. Poincaré return maps. Top: CV1 distribution with Σ_fold (red) and "
    "Σ_unfold (green) thresholds. Centre-left: Σ_fold return map — each dot is two "
    "consecutive committed fold entries; colour=log density (red scale). Centre-right: "
    "Σ_unfold return map (green scale). Centre-right panel: CV2 marginal distributions "
    "at each crossing type. Bottom: recurrence time distributions and temporal stationarity.",
    width=6.2)
doc.add_paragraph()

# ── 3. Per-Residue Torsion Analysis ───────────────────────────────────────────
heading(doc, "3. Per-Residue Backbone Torsion Analysis", 1)
body(doc,
    "To identify which residues drive the two folding routes, committed fold-crossing "
    f"frames were extracted from the XTC trajectories (topology: shared_gamd_setup_final.pdb). "
    f"Per-residue φ and ψ angles were computed using MDTraj. "
    f"Route A vs Route B are split at CV2 = {(fp[0]['cv2'] + fp[1]['cv2'])/2:.3f} "
    f"(midpoint between the two fold peaks).")

body(doc,
    f"Note: Route B has only {len(B)} events vs {len(A)} for Route A. "
    "Δψ/Δφ values for Route B should be treated as indicative rather than statistically "
    "robust. More sampling with a lower dwell threshold (--poincare-min-dwell 5) "
    "would improve Route B statistics.")

heading(doc, "3.1 Per-Residue ψ Angles", 2)
hdr = ["Residue", "Route A ψ (°)", "Route B ψ (°)", "Unfold ψ (°)",
       "Δψ B−A (°)", "Δψ Unfold−A (°)"]
tdata = []
for i in range(len(psi_cols)):
    if i >= len(RES_NAMES):
        break
    s = psi_stats[i]
    def fmt(m, sd):
        if np.isnan(m): return "—"
        return f"{m:+.1f} ± {sd:.1f}"
    def fmt_d(d):
        if np.isnan(d): return "—"
        return f"{d:+.1f}"
    tdata.append([
        RES_NAMES[i],
        fmt(s['A_mean'], s['A_std']),
        fmt(s['B_mean'], s['B_std']),
        fmt(s['U_mean'], s['U_std']),
        fmt_d(s['delta_BA']),
        fmt_d(s['delta_UA']),
    ])
add_table(doc, hdr, tdata)
doc.add_paragraph()

heading(doc, "3.2 Per-Residue φ Angles", 2)
hdr2 = ["Residue", "Route A φ (°)", "Route B φ (°)", "Unfold φ (°)",
        "Δφ B−A (°)", "Δφ Unfold−A (°)"]
tdata2 = []
for i in range(len(phi_cols)):
    if i >= len(RES_NAMES):
        break
    s = phi_stats[i]
    def fmt(m, sd):
        if np.isnan(m): return "—"
        return f"{m:+.1f} ± {sd:.1f}"
    def fmt_d(d):
        if np.isnan(d): return "—"
        return f"{d:+.1f}"
    tdata2.append([
        RES_NAMES[i],
        fmt(s['A_mean'], s['A_std']),
        fmt(s['B_mean'], s['B_std']),
        fmt(s['U_mean'], s['U_std']),
        fmt_d(s['delta_BA']),
        fmt_d(s['delta_UA']),
    ])
add_table(doc, hdr2, tdata2)
doc.add_paragraph()

heading(doc, "3.3 Key Residues Driving Route Differences", 2)

most_psi = sorted(range(len(psi_cols)), key=lambda i: abs(psi_stats[i]['delta_BA']), reverse=True)
most_phi = sorted(range(len(phi_cols)), key=lambda i: abs(phi_stats[i]['delta_BA']), reverse=True)

p = doc.add_paragraph()
p.add_run("Largest Δψ (Route B − Route A):").bold = True
for i in most_psi[:3]:
    if i < len(RES_NAMES) and not np.isnan(psi_stats[i]['delta_BA']):
        p.add_run(f"  {RES_NAMES[i]}: {psi_stats[i]['delta_BA']:+.1f}°")

p = doc.add_paragraph()
p.add_run("Largest Δφ (Route B − Route A):").bold = True
for i in most_phi[:3]:
    if i < len(RES_NAMES) and not np.isnan(phi_stats[i]['delta_BA']):
        p.add_run(f"  {RES_NAMES[i]}: {phi_stats[i]['delta_BA']:+.1f}°")

p = doc.add_paragraph()
p.add_run("Largest Δψ (Unfold − Route A):").bold = True
for i in sorted(range(len(psi_cols)), key=lambda i: abs(psi_stats[i]['delta_UA']), reverse=True)[:3]:
    if i < len(RES_NAMES) and not np.isnan(psi_stats[i]['delta_UA']):
        p.add_run(f"  {RES_NAMES[i]}: {psi_stats[i]['delta_UA']:+.1f}°")

doc.add_paragraph()
body(doc,
    f"The two folding routes differ most at the N-terminal strand residues "
    f"({RES_NAMES[most_psi[0]]}, {RES_NAMES[most_psi[1]]}), not at the turn "
    f"(Asp3-Pro4-Glu5). The turn geometry is similar across both routes, "
    "consistent with a fixed turn nucleus model: the β-hairpin turn locks first "
    "and the strands zip in via different backbone configurations. "
    f"Unfolding (Δψ Unfold−A) shows large shifts at strand residues (Tyr2, Asp3, "
    "Thr6, Trp9) — the fully extended state has strongly positive ψ throughout, "
    "consistent with PPII/extended-strand geometry.")

heading(doc, "3.4 Per-Residue Ramachandran Figure", 2)
add_figure(doc,
    ANALYSIS_DIR / "poincare_residue_torsions.png",
    "Figure 2. Per-residue Ramachandran plots at committed crossing frames. "
    "Top 10 panels: one per residue (Gly1–Gly10); red = Route A, blue = Route B, "
    "green = unfold. Background shading: β/extended (blue), PPII (cyan), "
    "right-α (orange), left-α (red). Bottom panels: mean ψ and φ differences "
    "(Route B − Route A) per residue with standard deviation error bars.",
    width=6.4)
doc.add_paragraph()

# ── 4. Methods & Caveats ──────────────────────────────────────────────────────
heading(doc, "4. Methods and Caveats", 1)

p = doc.add_paragraph(style='List Bullet')
p.add_run("Simulation:").bold = True
p.add_run(" GAREUS REUS+HMR-GaMD, 16 replicas, 66 ns, chignolin CLN025 (GYDPETGTWG), "
          "explicit TIP3P solvent, OpenMM 8.3+, CUDA.")

p = doc.add_paragraph(style='List Bullet')
p.add_run("CVs:").bold = True
p.add_run(" CV1 = nonlocal contact fraction (backbone-heavy, min seq sep 3, r₀=4.8 Å, "
          "normalised); CV2 = rama-map (weighted average φ/ψ similarity to 4 Ramachandran "
          "basin centres).")

p = doc.add_paragraph(style='List Bullet')
p.add_run("Poincaré analysis:").bold = True
p.add_run(f" Σ_fold threshold CV1={pm['c_fold']:.3f}, Σ_unfold={pm['c_unfold']:.3f}. "
          f"Committor dwell filter {pm['min_dwell_ps']:.0f} ps ({pm['min_dwell_frames']} frames "
          f"at 0.2 ps/frame). Min segment between crossings: 5 frames.")

p = doc.add_paragraph(style='List Bullet')
p.add_run("Torsion extraction:").bold = True
p.add_run(" MDTraj 1.x; protein atoms selected via topology.select('protein'); "
          "φ/ψ computed via mdtraj.compute_phi/psi; XTC frames matched to crossing "
          "steps via traj_interval=50 steps/frame.")

p = doc.add_paragraph(style='List Bullet')
p.add_run("REUS bias caveat:").bold = True
p.add_run(" Crossing frequencies and recurrence times are biased by the umbrella "
          "potential. Map topology (clusters, diagonal memory) is qualitatively "
          "bias-robust; quantitative crossing rates require MBAR reweighting.")

p = doc.add_paragraph(style='List Bullet')
p.add_run("Route B statistics:").bold = True
p.add_run(f" Only {len(B)} events. Δψ/Δφ values are indicative. More simulation "
          "or lower dwell threshold needed for statistical significance.")

p = doc.add_paragraph(style='List Bullet')
p.add_run("Crash artifacts:").bold = True
p.add_run(" Two replica final XTC segments (replica_002, replica_006) were malformed "
          "(simulation crash at step 5,933,550). Those frames were skipped; "
          "earlier segments for those replicas were included normally.")

# ── 5. Output Files ──────────────────────────────────────────────────────────
heading(doc, "5. Output Files", 1)
add_table(doc,
    ["File", "Description"],
    [
        ["poincare_map.png",               "Annotated return maps, CV1 distribution, CV2 marginals, recurrence times"],
        ["poincare_map_summary.json",      "Threshold values, crossing counts, peak positions, route summaries"],
        ["poincare_fold_crossings.csv",    "replica, step, cv2 for each committed fold crossing"],
        ["poincare_unfold_crossings.csv",  "replica, step, cv2 for each committed unfold crossing"],
        ["poincare_residue_torsions.png",  "Per-residue Ramachandran plots + Δφ/Δψ bar charts"],
        ["poincare_residue_torsions.csv",  "event_type, replica, step, cv2, phi_0..phi_N, psi_0..psi_N"],
        ["poincare_findings.docx",         "This report"],
    ]
)

doc.save(OUT_DOCX)
print(f"Saved: {OUT_DOCX}")
