#!/usr/bin/env python3
"""Generate the GAREUS engine statistical-mechanics / MD code-review report as .docx.

Scope: the GAREUS sampling engine only (GaMD boost + REUS umbrella sampling + CV
machinery + system setup + adaptive windows + seeding). Analysis, GENPEPT, and
helper/infra modules are out of scope.

Findings come from a multi-agent adversarial review pass; the highest-severity
findings were independently re-read and confirmed in the main thread, and the
load-bearing statistical mechanics was checked against a Codex/GPT consultation.
"""

from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

OUT_DOCX = Path("/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/gareus_statmech_review.docx")

# ── palette ──────────────────────────────────────────────────────────────────
SEV_COLOR = {"critical": "C00000", "high": "C55A11", "medium": "BF9000", "low": "548235"}
SEV_FILL = {"critical": "FCE4E4", "high": "FCEADB", "medium": "FDF3D8", "low": "E8F2DE"}
CAT_LABEL = {"science": "Scientific validity", "math-stability": "Numerical stability",
             "bug": "Correctness bug", "implementation": "Implementation"}
INK = RGBColor(0x1A, 0x1A, 0x6E)
GREY = RGBColor(0x55, 0x55, 0x55)


# ── helpers ────────────────────────────────────────────────────────────────────
def heading(doc, text, level):
    return doc.add_heading(text, level=level)


def body(doc, text):
    p = doc.add_paragraph(text)
    p.paragraph_format.space_after = Pt(6)
    return p


def equation(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.45)
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(3)
    run = p.add_run(text)
    run.font.name = "Courier New"
    run.font.size = Pt(9.5)
    run.font.color.rgb = INK
    return p


def code_block(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.35)
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(4)
    run = p.add_run(text)
    run.font.name = "Courier New"
    run.font.size = Pt(8)
    run.font.color.rgb = RGBColor(0x20, 0x20, 0x20)
    _shade_paragraph(p, "F2F2F2")
    return p


def note(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.3)
    p.paragraph_format.space_after = Pt(4)
    run = p.add_run(text)
    run.italic = True
    run.font.size = Pt(9)
    run.font.color.rgb = GREY
    return p


def callout(doc, text, fill="FFF2CC"):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(6)
    r = p.add_run(text)
    r.bold = True
    r.font.size = Pt(10)
    r.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
    _shade_paragraph(p, fill)
    return p


def _shade_paragraph(p, fill_hex):
    pPr = p._p.get_or_add_pPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:fill'), fill_hex)
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:val'), 'clear')
    pPr.append(shd)


def _shade_run(run, fill_hex):
    rPr = run._r.get_or_add_rPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:fill'), fill_hex)
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:val'), 'clear')
    rPr.append(shd)


def shaded_cell(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:fill'), fill_hex)
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:val'), 'clear')
    tcPr.append(shd)


def add_table(doc, headers, rows_data, header_fill='D9E1F2', font_size=8.5, sev_col=None, widths=None):
    table = doc.add_table(rows=1 + len(rows_data), cols=len(headers))
    table.style = 'Table Grid'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = h
        run = hdr[i].paragraphs[0].runs[0]
        run.bold = True
        run.font.size = Pt(font_size)
        shaded_cell(hdr[i], header_fill)
    for ri, row_data in enumerate(rows_data):
        cells = table.rows[ri + 1].cells
        for ci, val in enumerate(row_data):
            cells[ci].text = str(val)
            if cells[ci].paragraphs[0].runs:
                cells[ci].paragraphs[0].runs[0].font.size = Pt(font_size)
            if sev_col is not None and ci == sev_col:
                sev = str(val).lower()
                shaded_cell(cells[ci], SEV_FILL.get(sev, "FFFFFF"))
                if cells[ci].paragraphs[0].runs:
                    cells[ci].paragraphs[0].runs[0].bold = True
                    cells[ci].paragraphs[0].runs[0].font.color.rgb = RGBColor(*_hex_rgb(SEV_COLOR.get(sev, "333333")))
    if widths:
        for ci, w in enumerate(widths):
            for row in table.rows:
                row.cells[ci].width = Inches(w)
    return table


def severity_chip(doc, severity, category, fid, status):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(2)
    chip = p.add_run(f"  {severity.upper()}  ")
    chip.bold = True
    chip.font.size = Pt(9)
    chip.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    _shade_run(chip, SEV_COLOR.get(severity, "555555"))
    tail = p.add_run(f"   {fid} · {CAT_LABEL.get(category, category)}   ·   {status}")
    tail.font.size = Pt(9)
    tail.font.color.rgb = GREY
    return p


def finding_block(doc, f):
    severity_chip(doc, f["severity"], f["category"], f["id"], f.get("status", ""))
    h = doc.add_paragraph()
    h.paragraph_format.space_after = Pt(2)
    r = h.add_run(f["title"])
    r.bold = True
    r.font.size = Pt(11)
    r.font.color.rgb = RGBColor(*_hex_rgb(SEV_COLOR.get(f["severity"], "333333")))
    loc = doc.add_paragraph()
    loc.paragraph_format.space_after = Pt(2)
    lr = loc.add_run(f["file"] + ":" + f["lines"])
    lr.font.name = "Courier New"
    lr.font.size = Pt(9)
    lr.font.color.rgb = GREY
    _label(doc, "What / why")
    body(doc, f["description"])
    if f.get("snippet"):
        code_block(doc, f["snippet"])
    _label(doc, "Impact")
    body(doc, f["impact"])
    _label(doc, "Check")
    cp = doc.add_paragraph()
    cr = cp.add_run("✓  " + f["check"])
    cr.font.size = Pt(10)
    cr.font.color.rgb = RGBColor(0x1F, 0x49, 0x2E)
    _shade_paragraph(cp, SEV_FILL.get(f["severity"], "F2F2F2"))
    _hr(doc)


def _label(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(0)
    r = p.add_run(text)
    r.bold = True
    r.font.size = Pt(8.5)
    r.font.color.rgb = GREY
    return p


def _hex_rgb(h):
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _hr(doc):
    p = doc.add_paragraph()
    pPr = p._p.get_or_add_pPr()
    pbdr = OxmlElement('w:pBdr')
    bottom = OxmlElement('w:bottom')
    bottom.set(qn('w:val'), 'single')
    bottom.set(qn('w:sz'), '6')
    bottom.set(qn('w:space'), '1')
    bottom.set(qn('w:color'), 'BFBFBF')
    pbdr.append(bottom)
    pPr.append(pbdr)
    return p


# ============================================================================
# DATA
# ============================================================================
ST_CONFIRMED = "re-read & confirmed (main thread)"
ST_GPT = "physics confirmed via Codex/GPT"
ST_AGENT = "re-read & confirmed (main-thread verification)"
ST_REFUTED_RR = "REFUTED on main-thread re-read — see note"
ST_SELF = "self-verified in workflow"
ST_SRC = "CONFIRMED — gamd-openmm source + adversarial verify + GPT"
ST_RERUN = "confirmed by rerun adversarial verifier"

CRIT_HIGH = [
    {
        "id": "F1", "severity": "critical", "category": "science",
        "title": "CONFIRMED: GaMD total boost is computed on the full energy that INCLUDES the window-dependent umbrella + secondary-CV bias — the REUS exchange cancellation is invalid",
        "file": "gareus/production.py", "lines": "942-974, 2640-2643, 2770-2771; gamd-openmm integrator_factory.py:198, stage_integrator.py:224",
        "status": ST_SRC,
        "snippet": "# gamd-openmm get_integrator() FIRST calls:  set_all_forces_to_group(system, 0)\n#   -> umbrella (group 31) AND secondary-CV (group 29) are folded into boosted group 0\n# stage_integrator.py:224  StartingPotentialEnergy = energy   # FULL system PE, all groups\n# base_integrator.py:367   BoostPotential = 0.5*k0*(threshold - StartingPotentialEnergy)^2/(Vmax-Vmin)\n# => dV(x) = f(U_phys(x) + U_umbrella,k(x)),  but exchange Δ (production.py:3625) omits dV entirely",
        "description": (
            "The prior pass flagged this as an unverified assumption; the rerun verified it against the gamd-openmm source. "
            "For the default boost type lower-dual, gamd-openmm's get_integrator() FIRST calls set_all_forces_to_group(system, "
            "0), placing the umbrella restraint (force group 31) and the secondary-CV force (group 29) into the boosted group "
            "0. The total boost uses StartingPotentialEnergy = the full system potential, so the applied boost "
            "dV(x) = 0.5·k0·(E − (U_phys + U_umbrella,k))²/(Vmax−Vmin) is an explicit function of the per-window umbrella "
            "energy. The single shared calibration is run with k=0 (production.py:2347-2351) so Vmax/Vmin/Vavg/σV are "
            "calibrated on the UNBIASED peptide, then frozen and copied verbatim to every window (2810-2811) — but in "
            "production each window adds a different umbrella energy on top of the same frozen thresholds. GPT-confirmed "
            "precondition: omitting the boost from the exchange acceptance is valid only if the boost excludes the "
            "window-dependent groups; it does not. The defect is the COMBINATION of (i) one frozen free-peptide calibration "
            "reused for all windows and (ii) the exchange Δ being umbrella-only (production.py:3625) while the boost actually "
            "tracks U_umbrella,k."),
        "impact": (
            "Both the REUS exchange acceptance and the per-frame GaMD reweighting factor exp(βΔV) assume the boost is a "
            "function of the physical potential only. Here ΔV also depends on the window-dependent umbrella bias, so neither "
            "cancels in the swap ratio nor reweights cleanly across windows. This is the verified mechanism behind the "
            "documented 'GaMD reweighting ESS effectively zero' (boost pooled across windows whose boost actually differs "
            "because it tracks U_umbrella,k) and biases the sampled ensemble and all downstream free energies."),
        "check": (
            "FIX direction: either restrict the boost to physical force groups (re-issue setForceGroup(31)/(29) AFTER the "
            "factory and confirm gamd-openmm's TOTAL excludes them) so the exchange omission becomes valid and one global "
            "calibration is meaningful, OR recalibrate Vmax/Vmin per window. CHECKS: (1) integration test — build base_system "
            "with nonzero umbrella k, assert |StartingPotentialEnergy − PE_all_groups| < 1e-3 kJ/mol while "
            "|StartingPotentialEnergy − PE_excluding_31_29| ≈ U_umbrella ≫ 0 (proves the umbrella is currently boosted); "
            "(2) two windows, same config, different r0/k → assert get_boost_potentials()['BoostPotential_Total'] differs "
            "(currently it does; should be ~0 if umbrella-independent); (3) per-window ESS = (Σe^{βΔV})²/Σe^{2βΔV}/N, expect "
            ">~0.05 healthy, ~0 confirms the pathology; (4) correlate per-window mean βΔV with |center − r0_min| — a strong "
            "monotone trend confirms ΔV tracks U_umbrella."),
    },
    {
        "id": "F2", "severity": "high", "category": "science",
        "title": "Single global Vmax/Vmin calibrated on the free peptide is frozen and reused for every umbrella window",
        "file": "gareus/production.py", "lines": "2312-2353 (shared calib, k=0), 2810-2812 (copied to replicas)",
        "status": ST_CONFIRMED,
        "snippet": "shared_sim.context.setParameter(\"k\", 0.0)   # calibrate with umbrella OFF\n...\nset_integrator_globals_from_dict(integrator_i, shared_gamd_globals_all)  # same E,k0,Vmax,Vmin -> every window\n# dV = 0.5*k0*(E-V)^2/(Vmax-Vmin)  for V<E",
        "description": (
            "GaMD energy statistics (Vmax, Vmin, Vavg, sigmaV → E, k0) are determined ONCE with the umbrella disabled (k=0), "
            "i.e. from the unbiased free, mostly-extended peptide PE distribution, then frozen and copied to every umbrella "
            "replica. Each window restrains the peptide to a different CV basin with a different PE range, so a single Vmax "
            "produces boosts whose magnitude and variance differ systematically between windows (compact windows near Vmin get "
            "large boosts; extended windows near Vmax get near-zero boost). GPT confirmed per-window reweighting is formally "
            "unbiased given exact recorded ΔV and overlap, but the boost distributions are not commensurate across windows."),
        "impact": (
            "Cross-window boost heterogeneity. Per-window reweighting can still be valid, but any pooled cross-window "
            "exp(βΔV) reweight has wildly varying weights → effective-sample-size collapse, and the large per-window-σ "
            "warnings. Inherent GaREUS approximation, but applied with no per-window adjustment and no recorded caveat."),
        "check": (
            "Record calibration (Vmin,Vmax,Vavg) and per-window mean potential_kj_mol in the manifest; warn when a window's "
            "mean sampled PE falls outside [Vmin,Vmax] (boost saturates/vanishes there). Diagnostic: plot per-window mean "
            "boost vs window center; flag if max/median across windows exceeds ~3×. Prefer per-window or per-region GaMD "
            "calibration if heterogeneity is large."),
    },
    {
        "id": "F3", "severity": "high", "category": "math-stability",
        "title": "Reweighting ESS pools the boost across all windows, guaranteeing weight collapse even when per-window reweighting is fine",
        "file": "gareus/diagnostics.py", "lines": "190-210  (note: diagnostics is out of the engine scope, but it is the proximate cause of the 'ESS≈0' symptom)",
        "status": ST_AGENT,
        "snippet": "finite = boost_kj[np.isfinite(boost_kj)]            # pooled over ALL windows\nbeta = 1.0/(0.00831446261815324*T)\nlogw = beta*finite ; logw -= np.max(logw)            # log-sum-exp OK\nweights = np.exp(logw)\ness = (np.sum(weights)**2)/np.sum(weights*weights)   # single global ESS",
        "description": (
            "The reweighting ESS is computed on the boost array POOLED over every replica/window, not per window. Because of "
            "F2 the per-window mean boost differs substantially; concatenating makes the dV spread equal to intra-window "
            "variance PLUS inter-window mean differences, so a few high-boost windows dominate Σw and ESS collapses to ~0 even "
            "when each window individually has acceptable overlap. ESS for reweighting is only meaningful within a single "
            "thermodynamic state. GPT: this is a variance/overlap failure, not necessarily formal bias. The log-sum-exp "
            "max-subtraction itself is correct."),
        "impact": (
            "The headline GaMD reweighting health metric reads ~0 and fires a 'weights strongly degenerate' warning "
            "regardless of true per-window quality — masking usable GaMD data and likely the origin of the CLAUDE.md "
            "'ESS effectively zero' note."),
        "check": (
            "Add per-window ESS: group finite boost by window, max-subtract and compute ESS within each window, report "
            "min/median per-window ess_fraction alongside the pooled value. Unit test: two synthetic windows boost~N(5,0.5) "
            "and N(20,0.5) kcal/mol; assert pooled ess_fraction ≪ min per-window ess_fraction and each per-window "
            "ess_fraction > 0.5."),
    },
    {
        "id": "F4", "severity": "high", "category": "science",
        "title": "Divergent adaptive seed scorer uses raw un-normalized Euclidean distance — mis-weights CV1 vs CV2 by ~50-100x",
        "file": "gareus/adaptive_production.py", "lines": "792-804 (called at 670, 766)",
        "status": ST_CONFIRMED,
        "snippet": "dp = float(p) - float(target.primary_center)\nds = 0.0\nif target.secondary_center is not None and s is not None:\n    ds = float(s) - float(target.secondary_center)\nscore = dp*dp + ds*ds          # raw units, no normalization, no weight",
        "description": (
            "A second, divergent 2D seed scorer (the primary one in seeding.py normalizes each CV by its median grid spacing). "
            "Here the two CV deltas are summed in raw native units. In the current config CV1 is a contact fraction "
            "(centers ~0–0.26 → dp² ≤ ~0.07) while CV2 is the rama-map index in [−1,1] (ds² up to ~4). The secondary term "
            "dominates by ~50–100×, so the contact CV is effectively ignored during seed-to-state assignment; in legacy "
            "distance mode (Å, ~3–32) the reverse happens and CV2 is ignored. GPT confirmed raw dp²+ds² is scale-biased and "
            "indefensible when CVs have different numeric ranges. This is live code feeding state-aware seed filtering."),
        "impact": (
            "Adaptive-production seed assignment matches only the larger-magnitude CV, defeating 2D seeding and violating the "
            "project rule 'do not regress to distance-/single-CV-only logic when CV2 is active'. Windows get seeds far from "
            "target in the ignored CV, slowing equilibration and biasing initial sampling."),
        "check": (
            "Replace score with (dp/median_dcv1)² + (w·ds/median_dcv2)² reusing seeding.py's spacing scales and weight; assert "
            "parity with the seeding.py scorer on a shared fixture. Unit test: target=(0.20 contacts, 0.0 rama); seedA=(0.20,1.0), "
            "seedB=(0.05,0.0) — the raw scorer wrongly always prefers seedB (ds²=1 swamps dp²=0.0225); normalized scorer must not."),
    },
    {
        "id": "F5", "severity": "high", "category": "science",
        "title": "Seeds with a non-finite (missing) secondary CV are systematically favored in both scorers",
        "file": "gareus/seeding.py", "lines": "814-827  (also adaptive_production.py:798-801)",
        "status": ST_CONFIRMED,
        "snippet": "secondary_score = 0.0\nuse_secondary = (... and math.isfinite(secondary_value))\nif use_secondary:\n    secondary_score = seed_secondary_weight*secondary_delta/max(1e-12, scale)\ntotal = primary_score + secondary_score     # NaN secondary -> 0 penalty -> always wins",
        "description": (
            "secondary_score is ≥ 0 and stays 0.0 when secondary_value is NaN (use_secondary requires math.isfinite). So a "
            "conformer that FAILED secondary scoring always has total ≤ an otherwise-identical seed that matches the secondary "
            "target imperfectly, and is preferentially selected. The same defect exists in the adaptive scorer (ds defaults to "
            "0.0 when s is None). The library loader already reports a 'finite_sec' count, so this path is reachable."),
        "impact": (
            "2D seed selection is biased toward conformers whose secondary structure could not be evaluated — the least "
            "characterized seeds — exactly when secondary data is partly missing, undermining the active-CV seeding intent."),
        "check": (
            "Either skip NaN-secondary seeds when secondary data is available, or assign them a max/sentinel secondary penalty "
            "(weight·max_observed_secondary_delta/scale) so they are penalized, not rewarded. Unit test: confA finite-secondary "
            "far from target vs confB secondary=NaN identical primary — assert confB is NOT preferred."),
    },
    {
        "id": "F6", "severity": "high", "category": "bug",
        "title": "Live-production secondary-CV bias matrix does not mask NaN centers/values, unlike the adaptive path",
        "file": "gareus/production.py", "lines": "3297-3308 (live) vs adaptive_production.py:1314-1318 (masked)",
        "status": ST_CONFIRMED,
        "snippet": "ss_delta_matrix = ss_values[np.newaxis,:] - ss_centers_arr[:,np.newaxis]\nss_bias_matrix_kcal = 0.5*ss_k_arr[:,np.newaxis]*ss_delta_matrix*ss_delta_matrix\n# no np.isfinite mask; ss_centers_arr_global init np.full(...,nan) for windows w/o secondary center",
        "description": (
            "ss_centers_arr_global is initialized to NaN for windows with no secondary center, and ss_values can be NaN when "
            "the secondary score is unavailable or a replica goes unstable. The live sampling/bias path multiplies these "
            "directly with no finite-mask, so any NaN poisons that (window,replica) entry. The exchange-state helper "
            "(3625-3628) and the adaptive MBAR builder DO guard with isfinite + secondary_mask. A single NaN freezes that "
            "replica's exchanges (NaN comparison → always reject) with no counter/warning and writes NaN into the cached bias "
            "matrix consumed downstream."),
        "impact": (
            "Silent NaN in the Metropolis term biases/freezes replica exchange (all-reject for that replica → reduced mixing/"
            "overlap) and corrupts the cached bias matrix used by MBAR — biased or broken free energies, no error raised. "
            "Edge-case in the full 2D grid (all windows finite), real in sparse/mixed setups."),
        "check": (
            "After building bias_matrix_kj in the live loop, assert np.isfinite(bias_matrix_kj).all() with a dump of the "
            "offending (window,replica); or mirror the adaptive path (compute ss bias only where centers & values are finite, "
            "else 0). Unit test: one window secondary_center=NaN + one finite → bias matrix finite everywhere; track a "
            "nonfinite-bias counter so a stuck replica is observable."),
    },
    {
        "id": "F7", "severity": "high", "category": "science",
        "title": "rama-map collapses geometrically non-collinear basins onto a monotonic 1D label ladder; intermediate centers are unphysical mixtures",
        "file": "gareus/cv.py", "lines": "394-399 (labels), 486-505 (scalar)",
        "status": ST_GPT,
        "snippet": "beta -> -1.0 ; ppii -> -1/3 ; alpha_r -> +1/3 ; alpha_l(phi=+60,psi=+40) -> +1.0\nss = sum(value_k*score_k)/(eps + sum(score_k))   # clamped to [-1,1]",
        "description": (
            "rama-map maps φ/ψ to one scalar via a softmax-like weighted average with fixed labels [−1,−1/3,+1/3,+1] for "
            "basins ordered beta, PPII, right-α, left-α. The labels impose a 1D ordering on basins that do not lie on a 1D "
            "physical manifold: left-α (φ=+60,ψ=+40) sits at the +1 extreme adjacent to right-α at +1/3 yet is geometrically "
            "unrelated. A window centered at e.g. +0.66 is not a real Ramachandran intermediate; the same scalar can come from "
            "mixed per-residue populations or off-basin geometry. Codex confirmed: numerically well-behaved (continuous, "
            "differentiable, bounded) but physically lossy — fine as a categorical basin map at named centers, problematic as "
            "a dense 1D umbrella ladder or 1D FES path."),
        "impact": (
            "PMFs along intermediate rama-map values are not interpretable as physical Ramachandran transition states; adaptive "
            "windows at intermediate centers waste sampling on unphysical mixtures and can yield a misleading 1D FES. Does NOT "
            "corrupt MBAR (the bias is self-consistent), but the scientific interpretation of intermediate-center windows is unsound."),
        "check": (
            "Unit test: evaluate the score at each named basin geometry and assert it equals that basin's label within "
            "tolerance; then a 50/50 beta/alpha_l residue mixture must return ~0 and NOT coincide with any basin. Guard "
            "adaptive-feedback to snap rama-map centers to {−1,−1/3,+1/3,+1} unless the user explicitly opts into a dense ladder."),
    },
    {
        "id": "F8", "severity": "high", "category": "science",
        "title": "LJ nonbonded cutoff defaults to 0.8 nm with NO switching function (hard truncation)",
        "file": "gareus/system_setup.py", "lines": "351-360 (create_system); cli.py:71 default=0.8",
        "status": ST_CONFIRMED,
        "snippet": "system_kwargs = dict(nonbondedMethod=app.PME, nonbondedCutoff=0.8*nm,\n                     constraints=app.HBonds, rigidWater=True, ewaldErrorTolerance=...)\n# no setUseSwitchingFunction / setSwitchingDistance anywhere in the engine (grep empty)",
        "description": (
            "createSystem is called without any switching function and the default cutoff is 0.8 nm; a grep for "
            "setUseSwitchingFunction/switchDistance across all engine files returns nothing, and no YAML overrides "
            "nonbonded_cutoff_nm — so 0.8 nm hard-truncated LJ is what actually runs. PME fixes electrostatics, but "
            "Lennard-Jones is truncated abruptly at 8 Å. Amber/CHARMM force fields are parameterized for ~0.9–1.0 nm cutoffs; "
            "0.8 nm with no switch (a) loses medium-range dispersion that controls peptide-water vs peptide-peptide packing and "
            "(b) creates a force/energy discontinuity at the cutoff sphere. (OpenMM keeps the analytic dispersion correction on "
            "by default, so the bias is from the missing switch + short cutoff, not from a missing tail correction.) GPT "
            "concurs: moderate-to-high, likely a systematic sub-kcal to ~1 kcal/mol PMF bias for a small peptide."),
        "impact": (
            "Systematically biased PMF for chignolin — relative stability of compact vs extended states shifted by truncated "
            "dispersion, plus integration noise from the force discontinuity (partly masked by the thermostat). All GaMD/REUS "
            "PMFs inherit the bias; the suite is internally consistent but not converged to the FF's intended physics."),
        "check": (
            "Raise the default to ≥0.9 nm and enable a switching function. Startup assertion: after createSystem, find the "
            "NonbondedForce and assert getUseSwitchingFunction() is True with getSwitchingDistance() ≈ cutoff−0.1 nm; warn if "
            "nonbonded_cutoff_nm < 0.9. Sanity: recompute the chignolin primary-CV PMF at 0.8 vs 1.0 nm and report the shift."),
    },
    {
        "id": "N1", "severity": "high", "category": "math-stability",
        "title": "gibbs-walk exchange (the configured default) upper-clips selection weights to 0 then force-accepts — biased kernel that does not preserve the target distribution",
        "file": "gareus/production.py", "lines": "3738-3762  (config.py:436 default; chignolin_fulltreatment*.yaml:394)",
        "status": ST_RERUN,
        "snippet": "log_weights = np.clip(-beta*delta, -745, 0)   # WRONG: upper bound 0\n# favorable swap delta<0 -> -beta*delta>0 -> true weight >1, clamped to 1 (== stay)\n# then the sampled target is force_accept=True (no Metropolis rejection correction)",
        "description": (
            "gibbs-walk samples a target window with probability ∝ exp(log_weights) where log_weights = clip(−βΔ, −745, 0). A "
            "correct heat-bath/Gibbs weight is ∝ exp(−βΔ) with NO upper bound — for a favorable swap (Δ<0) the weight exceeds "
            "1, but the clip to 0 clamps every favorable swap to weight 1, identical to staying. The selection is therefore "
            "neither a valid Gibbs update (wrong weights) nor a valid Metropolis update (force_accept=True, no rejection), so "
            "the kernel does not leave π(assignment) invariant. GPT confirmed the upper clip is wrong (heat-bath weights are "
            "unbounded above before normalization). This is the CONFIGURED DEFAULT scheme and is set in the chignolin YAML, so "
            "every current CLN025 production run uses this biased exchange. Within-window MD still targets π_w(x); the certain "
            "harm is broken exchange invariance + degraded mixing (favorable long jumps under-selected, P(stay) inflated)."),
        "impact": (
            "The stationary distribution over window assignments is wrong and exchange mixing is degraded on the default "
            "scheme — silently, no crash/warning. Distorts which configurations accumulate per window (what MBAR/WHAM assume), "
            "plausibly contributing to ESS≈0 / poor convergence."),
        "check": (
            "Remove the upper bound: log_weights = np.clip(−β·Δ, a_min=−745.0, a_max=None) (lower underflow guard only). Unit "
            "test on a K=5 single-site heat-bath with random U: build the K×K selection transition matrix from the production "
            "formula and assert max|π·T − π| < 1e-12 with π ∝ exp(−βU) (current code residual ~0.115; fixed ~1e-16); also "
            "assert detailed balance max|π_i T_ij − π_j T_ji| < 1e-12."),
    },
    {
        "id": "N2", "severity": "high", "category": "science",
        "title": "2D adaptive add/remove decisions run on axis-pooled marginals — a single depopulated (cv1,cv2) cell can be wrongly pruned",
        "file": "gareus/adaptive_feedback.py", "lines": "1350-1351, 1403-1411, 1488-1494 (pooling at 240-247)",
        "status": ST_RERUN,
        "snippet": "_group_window_samples_by_axis(...)   # collapses every secondary slice into ONE primary-axis histogram\n# overlap-based removal of primary column i uses POOLED overlap with i-1/i+1\n# -> looks high even when one cell (primary_i, secondary_j) is a depopulated bottleneck",
        "description": (
            "In the rectangular 2D dispatcher, primary/secondary add/remove/shift proposals are computed from samples POOLED "
            "across the orthogonal axis. The overlap-based removal of a primary center at index i uses the pooled overlap of "
            "column i with i±1, which can read high (≥high_overlap) even when one specific (primary_i, secondary_j) cell is a "
            "depopulated bottleneck. The code comment acknowledges pooling can hide a broken local edge and relies on "
            "sparse_patch_rows — but that only blocks CONVERGENCE, it does not stop the pooled-overlap logic from REMOVING a "
            "primary column that is load-bearing for one secondary slice, orphaning that cell in the next grid."),
        "impact": (
            "Axis-pooled redundancy pruning can delete a window statistically necessary for one secondary slice, creating a 2D "
            "gap the per-axis MBAR sees as connected but which biases the 2D FES."),
        "check": (
            "Before accepting a primary-axis removal at i, verify every cell (i,j) over all secondary slices has "
            "min(left,right) per-cell overlap ≥ minimum_valid_overlap using _adaptive_feedback_2d_edge_diagnostics. Unit test: "
            "a 3×2 grid where cell (1,0) is depopulated but pooled column-1 overlap is high → assert column 1 is NOT proposed "
            "for removal."),
    },
    {
        "id": "N3", "severity": "high", "category": "science",
        "title": "very-aggressive remove_block_radius=0 lets two adjacent windows be removed in one round; the bypass test only validated single removal",
        "file": "gareus/adaptive_feedback.py", "lines": "2218-2285 (also local_redundancy 2155-2212); windows.py:315",
        "status": ST_RERUN,
        "snippet": "# very-aggressive: remove_block_radius = 0 (windows.py:315)\n# blocked.update(range(w, w+1)) blocks only w; adjacency guard abs(w-prev)<=0 blocks only identical index\n# => windows w and w+1 can BOTH be removed; bypass test for w assumed w+1 survives, and vice-versa",
        "description": (
            "The bypass-prune agent justifies removing interior window w by checking neighbors w−1 and w+1 stay connected. "
            "Selection blocks further removals within remove_block_radius of an accepted removal — radius 1 for "
            "aggressive/balanced/conservative, but very-aggressive sets 0, so adjacent removals are NOT blocked. Windows w and "
            "w+1 can both be removed in one round; the bypass test for w assumed w+1 survives and vice-versa, so the actual "
            "surviving neighbors become w−1 and w+2, whose pairwise overlap was never tested. The verifier widened the scope: "
            "the same hazard exists for the local_redundancy agent, and the only post-removal repair (midpoint gap-fill) is "
            "gated on contact_primary & contact_normalize, so the rama axis has NO gap-fill at all."),
        "impact": (
            "In very-aggressive mode the ladder can be cut at two adjacent points at once, creating an untested wide gap "
            "(w−1 to w+2) that statistically disconnects the ladder and biases/destabilizes MBAR — silent, manifests only as "
            "poor downstream convergence."),
        "check": (
            "Force remove_block_radius ≥ 1 (set very-aggressive to 1), OR after selecting removals recompute the bypass "
            "overlap between every pair of surviving neighbors bracketing ≥2 removed windows and reject removals whose "
            "resulting neighbor-overlap CI-low < minimum_valid_overlap. Assert no two entries of removed_windows are adjacent. "
            "Unit test: 5-window ladder, very-aggressive, windows 2 and 3 both bypass-removable → assert not both removed (or "
            "survivor pair (1,4) overlap re-validated ≥ floor 0.04)."),
    },
    {
        "id": "N4", "severity": "high", "category": "math-stability",
        "title": "Fresh adaptive segments re-seed the exchange RNG and integrators with the identical seed every segment — no fresh stochasticity across epochs",
        "file": "gareus/adaptive_production.py", "lines": "2830 (copy.copy), 2905 (run); production.py:3067, 923/969",
        "status": ST_RERUN,
        "snippet": "seg_args = copy.copy(args)         # seed never offset\n# each segment: rng = np.random.default_rng(args.seed)   (production.py:3067)\n# per-replica integrators all seeded args.seed (923 CMD / 969 GaMD)",
        "description": (
            "run_segment builds seg_args via shallow copy.copy(args) and never offsets seg_args.seed. Each segment then "
            "creates a brand-new exchange RNG default_rng(args.seed) and per-replica integrators seeded with args.seed (the "
            "Langevin/GaMD thermostat seed is shared across replicas). Because the seed is constant across all epochs/segments, "
            "the exchange-acceptance draw sequence is identical every segment and the thermostat streams repeat. (Initial "
            "velocities ARE differentiated per replica via seed+i+17; the shared quantity is the integrator thermostat seed "
            "and the cross-segment exchange RNG.) Adaptive-production segments are fresh (resume=False) by default, so the "
            "resume-path rng_state restore does not save them."),
        "impact": (
            "Multi-segment adaptive runs do not accumulate independent randomness: exchange decisions and thermal noise are "
            "systematically reproduced segment-to-segment, reducing effective independent sampling and biasing "
            "convergence/overlap statistics that assume fresh stochasticity per segment."),
        "check": (
            "Offset per segment: seg_args.seed = int(args.seed) + epoch_index*1000003 + segment_index, and seed each replica "
            "integrator with args.seed + i. Assert default_rng(seg_a.seed) and default_rng(seg_b.seed) give different first-10 "
            "draws across segments and that per-replica integrator seeds are all distinct; log per-segment/per-replica seeds."),
    },
]

MEDIUM = [
    ("F9", "bug", "Boost unit detection silently treats a bare float as kJ/mol; a kcal/mol return is mislabeled (4.184x off)",
     "state.py:74-89",
     "_energy_to_kj_mol falls through to float(value) when get_boost_potentials() returns a bare number instead of an OpenMM Quantity, accepting it AS-IS and labeling it kJ/mol. If those numbers are kcal/mol every gamd_boost_total_kj_mol is 4.184× off, scaling exp(βΔV) wrongly, with no error.",
     "Assert the raw result is a Quantity (hasattr value_in_unit); record boost_unit_source. Startup probe: compare to integrator_globals boost values, warn on ~4.184× mismatch. Unit-test _energy_to_kj_mol with a kcal/mol Quantity → returns kcal*4.184.", ST_AGENT),
    ("F10", "bug", "lower-dual boost total can double-count or mis-split when get_boost_potentials returns ambiguous component names",
     "production.py:329-339, 199-207",
     "Total boost = sum(comps.values()) over whatever keys are returned; if a version returns BOTH components AND an aggregate 'Total', the sum double-counts. Re-read: CORRECT for the default lower-dual ({Total, Dihedral} only); the double-count risk is real only for non-default boost types (GROUPS / dual-nonbonded-dihedral) that emit per-group + Total — same defect as N5.",
     "When both component keys and an aggregate are present, assert aggregate ≈ sum(components) and use the aggregate, else error. Unit test {'dihedral':3,'total':5,'TotalBoost':8} → recorded total 8, not 16.", ST_AGENT),
    ("F11", "science", "Recorded potential_kj_mol includes the umbrella bias, but the GaMD boost is a function of the UNbiased potential",
     "production.py:3283-3284, 3352",
     "getPotentialEnergy() returns all force groups including the active umbrella (k>0), so potential_kj_mol is U(x)+W_k(x), not U(x). The GaMD calibration ran unbiased (k=0). Any downstream code that recomputes dV from potential_kj_mol or uses it as 'V' is biased by the per-frame umbrella energy.",
     "Record an unbiased potential too (getState(groups=...) excluding the umbrella group 31) as potential_unbiased_kj_mol; assert potential_kj_mol − potential_unbiased_kj_mol ≈ recorded umbrella_bias_kj_mol.", ST_AGENT),
    ("F12", "science", "Per-frame boost read 'one stage behind' the CV/energy — REFUTED on re-read",
     "production.py:3282-3294, 3365-3367",
     "REFUTED: the main loop is strictly serialized step_all(chunk)→sample()→attempt_exchanges(); inside sample() the CV and the boost are read from the SAME sim object at the same point, so there is no frame desync. Code-fact accurate but not a defect. Retained for the record.",
     "(no action needed) — optional: a debug assertion that recomputed analytic dV ≈ get_boost_potentials confirms alignment.", ST_REFUTED_RR),
    ("F13", "implementation", "Boost recorded but never folded into the MBAR reduced potential, and the arrays don't flag it",
     "production.py:1464-1467, 1490-1491",
     "umbrella_reduced_bias_nk = β·(umbrella bias only). The boost is a separate column; no cumulant/exp reweight exists in-engine. Architecturally fine (boost is a post-hoc per-bin correction), but a consumer could run MBAR on these arrays WITHOUT the boost correction and get the boosted (flattened), not equilibrium, PMF.",
     "Add metadata reduced_bias_includes_gamd_boost=False + a gamd_reduced_boost_nk=β·boost array and a documented note; assert any in-repo MBAR caller applies a boost reweight when boost variance is nonzero.", ST_AGENT),
    ("F14", "math-stability", "Secondary-CV value feeding the reconstructed MBAR bias comes from two non-identical evaluators (eps/clamp differ)",
     "production.py:248-276, 3269-3306, 3594-3627; cv.py:501-505",
     "The fast path (_ss_scalar_from_sub_cv_values, denom+1e-8, no clamp) mirrors the OpenMM force; the slow Python path (denom guard 1e-12, clamp to [−1,1]) differs when scores are tiny (all residues far from basins). Mixing paths (sampling vs resume/fallback) makes the reconstructed bias ≠ the applied bias — small systematic MBAR bias concentrated in low-score/disordered configs.",
     "Make both paths use identical eps (1e-8) and clamp policy; unit-test bit-for-bit equality over a φ/ψ battery incl. all-far-from-basins; runtime assert max|ss_fast − ss_slow| < 1e-6.", ST_AGENT),
    ("F15", "math-stability", "Spacing→k rule duplicated in 3 functions with inconsistent overlap_sigma defaults/floors; no overlap guarantee",
     "windows.py:525-541, 588-612; cv.py:565-571",
     "k = kBT/σ², σ = spacing/overlap_sigma implemented thrice with divergent floors (σ floor 0.05 Å distance vs 0.02 contact) and clamps (adaptive_min/max vs 5/120). Re-read CORRECTION: the original 'overlap_sigma=1.25 → poor overlap' claim is backwards — d/σ=1.25 is GENEROUS overlap (σ≈0.8·spacing), more than the σ≈spacing/2 rule. The genuine defect is the DRY triplication + the min/max k clamps silently overriding the k–σ relation so the installed k no longer matches the intended overlap, with nothing re-checking neighbor overlap.",
     "After finalizing k, compute predicted neighbor overlap O = exp(−spacing²/(8·σ_eff²)), σ_eff=√(kBT/k), and warn/assert O ≥ ~0.1 per adjacent pair; emit it in the window rows. Unit-test that the three functions return identical k for the same (kBT, spacing, overlap_sigma).", ST_AGENT),
    ("F16", "bug", "set_window silently swallows secondary-CV parameter-assignment failures",
     "windows.py:205-213",
     "A bare except around setParameter('ss0'/'ss_k') assumes the only failure is 'force absent', but an IndexError from a length-mismatched secondary array (plausible in sparse-2D/resume) is also swallowed, leaving the replica in the WRONG (stale/zero) secondary state while the bias matrix assumes the intended one.",
     "Narrow the except to the 'parameter does not exist' case (cache membership of 'ss_k'); let IndexError raise. Assert len(secondary_ks_kj)==len(centers) at setup; after set_window read back ss_k and assert it equals the request.", ST_AGENT),
    ("F17", "implementation", "Contact-CV self-test computes manual vs OpenMM collective value but never asserts they agree",
     "forces.py:142-170",
     "The self-test builds both the OpenMM CV (CustomBondForce-as-CV) and the Python manual CV — the two paths that MUST agree for the bias/exchange matrices to be correct — but only checks each is finite. A normalization mismatch (raw sum vs /norm_denom) would silently bias the contact CV.",
     "Reconstruct the normalized OpenMM CV (collective_value/contact_norm) and assert abs(cv_openmm − manual_primary_cv) < 1e-6; fail loudly on mismatch.", ST_AGENT),
    ("F18", "math-stability", "gaussian_kde / np.percentile in Delaunay window placement fails silently on degenerate (collinear/all-equal) pilot samples",
     "windows.py:1359-1375",
     "gaussian_kde inverts the 2×2 sample covariance; near-collinear pilot data (e.g. an under-sampled window where CV2 barely moved) makes it singular → LinAlgError. The per-axis span floor (1e-8) only rescales, it cannot de-correlate, so singularity persists. The broad try/except returns ([], {error}), so the adaptive round silently produces zero anchors. (Self-verified in workflow: confirmed real; the separate 'out-of-range density_floor_q gives garbage' sub-claim is wrong — np.percentile raises ValueError, same swallowed outcome.)",
     "Pre-check rank(cov)==2 (or both column stds > 1e-6) and log a distinct 'degenerate_pilot_distribution' reason; add tiny isotropic jitter or fall back to marginal placement; clamp density_floor_q to [0,1). Unit-test all-equal CV2 → named degeneracy.", ST_SELF),
    ("F19", "math-stability", "Bias-matrix NaN is only absorbed by downstream finite-guards, never detected at the source",
     "production.py:3606-3628",
     "A NaN primary/secondary CV (unstable replica, or ss=NaN) poisons a whole column of bias_matrix_kj. _exchange_probability returns 0 (isfinite false) and the Gibbs walk masks non-finite log-weights, so swaps aren't corrupted — but the replica silently stops exchanging with no counter, and the NaN is cached in observable_cache['bias_matrix_kj'] as a latent trap.",
     "Count non-finite entries after building the matrix; increment exchange_stats['nonfinite_bias_columns'] keyed by replica and emit a one-time warning. Unit-test: a NaN entry → all swaps for that replica rejected AND the counter records it.", ST_AGENT),
    ("F20", "bug", "contact_min_sequence_separation and seed r0/β fall back to non-config defaults if metadata is missing",
     "cv.py:761,875; seeding.py:148-150",
     "build_nonlocal_contact_pairs defaults min_sep=4 while the chignolin config sets 3; the seeding proxy hardcodes r0=0.45 nm, β=60 nm⁻¹ (=4.5 Å, 6.0 Å⁻¹) which do NOT match the config (4.8 Å, 4.0 Å⁻¹). Only bites if args/metadata lack the attributes (partial proxy or resume), but then the contact fraction silently differs from the production force.",
     "Assert contact_min_sequence_separation/contact_r0_a/contact_beta_a_inv are present on args (raise if missing) rather than defaulting; in the seed proxy raise if _np_r0_nm/_np_beta_nm_inv absent. Resume test: rebuilt r0_nm/β match the original.", ST_AGENT),
    ("F21", "math-stability", "Seed-to-window assignment under a reuse cap is non-deterministic in the threaded pull path",
     "seeding.py:848-866",
     "With seed_max_reuse>0, selection depends on running seed_usage_counts mutated under a lock as windows complete; the multi-worker path processes windows in non-deterministic as_completed order, so which window gets a popular conformer varies run-to-run even with a fixed RNG seed, and differs from the serial path. (Reuse cap defaults to 0/disabled.)",
     "Compute all window scores first, then assign in deterministic order (sorted window index or a global optimal assignment) before incrementing counts. Test: run selection twice with seed_max_reuse=1 → identical window→pdb mapping.", ST_AGENT),
    ("F22", "implementation", "Adaptive seed scorer ignores the user-configurable seed_secondary_weight",
     "adaptive_production.py:797-801",
     "seeding.py exposes seed_secondary_weight (default 1.0) but the adaptive scorer hardcodes equal weighting and never reads it, so setting the weight to 0 or >1 has no effect on adaptive state-aware assignments — silent divergence between the two scoring paths.",
     "Thread the weight + spacing scales into select_state_aware_seeds_for_targets (ideally factor seeding.py._score into a shared helper used by both); test weight=0 → pure-primary ranking; large weight flips to the secondary-matching seed.", ST_AGENT),
    ("F23", "science", "GaMD boost statistics are calibrated under a fluctuating NPT barostat",
     "production.py:2629-2639, 2327-2328",
     "The shared GaMD calibration runs on a system that INCLUDES the MonteCarloBarostat (production_ensemble=='npt' default), so Vmax/Vmin/Vavg/σV reflect NPT volume fluctuations. Standard GaMD calibration is defined for the configurational potential at fixed volume; calibrating with the barostat active couples the boost params to volume moves and can widen the boost distribution → poorer cumulant reweighting.",
     "Calibrate GaMD in NVT (include_barostat=False) and enable the barostat only for production, or record/assert the calibration ensemble. Diagnostic: compare calibrated Vmax/Vmin under NVT vs NPT; assert boost std below the σ0 target.", ST_AGENT),
    ("F25", "math-stability", "rama-map OpenMM force lacks the [-1,1] clamp present in the Python evaluator; division by near-zero denom can blow up",
     "production.py:842 (cv_expr) vs cv.py:505",
     "The applied OpenMM secondary force uses (numer)/(1e-8+denom) with NO clamp, while the Python evaluator clamps to [−1,1]. When all φ/ψ are far from every basin, denom→0 and the CV is dominated by the 1e-8 regularizer, taking large/ill-defined values that are then squared in 0.5·ss_k·(cv−ss0)². A transient spike injects a huge restraint force on a real replica — a plausible 4 fs HMR instability trigger.",
     "Clamp the OpenMM expression to [−1,1] via min/max so it matches the documented range and the Python path; assert each torsion-score sub-CV ∈ [0,1] and denom ≥ a floor. Diagnostic: drive all torsions to ±90° and assert |cv| ≤ 1+tol.", ST_AGENT),
    ("N5", "bug", "extract_gamd_boost_kj sums ALL keys from get_boost_potentials() — correct for lower-dual but double-counts for boost types that emit both per-group and Total keys",
     "production.py:329-339, 142-163",
     "For lower-dual the dict is exactly {BoostPotential_Total, BoostPotential_Dihedral} so the sum is right. But for GROUPS / dual-nonbonded-dihedral types (all user-selectable via cli.py:344) get_boost_potentials() can return per-group entries AND a Total that already aggregates them; blind summing double-counts, inflating exp(βΔV).",
     "Boost-type-aware extraction: if a Total key is present and the method aggregates groups, return only Total (+ independent Dihedral). Unit test {Total:10,Dihedral:3,NonBonded:7} → 13, not 20; cross-check vs boosted_energy−StartingPotentialEnergy within 1e-6.", ST_RERUN),
    ("N6", "math-stability", "Adaptive convergence can be declared with unresolved low-overlap pairs on the 1D/fallback path (plateau override ignores unresolved_pairs)",
     "adaptive_feedback.py:2518-2527",
     "The score-plateau override sets converged=True when the proposal score stops improving and structure is unchanged, IGNORING unresolved_pairs/coverage_gaps — so a low-overlap gap that cannot be repaired (add caps, spacing, center clamp) is shipped to final production. Verifier-scoped to the 1D dispatcher (default 2D path uses the correct gate); fallback for single-CV/degenerate-secondary runs.",
     "Plateau override must AND-in `not unresolved_pairs and not coverage_gaps`; assert converged ⇒ min finite decision_overlap ≥ 0.9·target. Unit test: one pair at 0.5·target, no improvement, unchanged → converged False.", ST_RERUN),
    ("N7", "math-stability", "severe_center_miss and weak-visitation convergence thresholds hardcode 1.5 Å — dead for the dimensionless contact-fraction CV",
     "adaptive_feedback.py:2105, 2520",
     "The center-miss gate uses abs(mean−center) > 1.5 (Å). The contact axis is only ~0.8 wide, so this can never fire — the severe_center_miss convergence term is permanently empty and the weak-visitation flag never fires in contact mode, silently disabling a convergence guard for the CV currently in use.",
     "Scale the threshold by the contact hit_radius / axis range when primary_cv_is_contacts. Unit test: contact center 0.4, mean 0.7, hit 0 → counted in severe_center_misses (currently not).", ST_RERUN),
    ("N8", "math-stability", "Per-pair overlap range is taken from each pair's own data union, so the binned overlap is non-comparable across pairs/rounds yet compared to fixed thresholds",
     "adaptive_feedback.py:632-639",
     "Each pair's histogram overlap uses a pair- and round-dependent [lo,hi] with bins=80 fixed, so bin width varies; widely-separated pairs get coarse bins / inflated overlap, tight pairs fine bins / lower overlap, for the SAME true overlap. Couples add/remove/redundancy decisions to ladder geometry, not the distributions.",
     "Use a fixed bin WIDTH (or fixed global CV range) so overlap is comparable, or a KDE Bhattacharyya coefficient. Unit test: two pairs with identical standardized dists but different separation → equal overlap.", ST_RERUN),
    ("N9", "math-stability", "All replica integrators share one fixed thermostat seed (args.seed=2026); replicas not statistically independent",
     "production.py:923, 969, 2771",
     "Both integrator factories call setRandomNumberSeed(args.seed) with the same nonzero value for every replica (no auto-randomize), and --randomize-replica-velocities defaults False, so replicas share a common thermostat noise stream → pairwise correlation ρ>0 → N_eff↓. Verifier: does NOT bias PMF (each window's invariant measure is seed-independent); inflates uncertainty/overlap ESS; not the primary ESS≈0 cause (that is F1).",
     "Per-replica seed (args.seed+i) + default randomize-replica-velocities True; assert distinct getRandomNumberSeed() across replicas; two k=0 replicas from identical state must diverge, not be bit-identical.", ST_RERUN),
    ("N10", "bug", "Stuck-replica rescue teleports a configuration and reinitializes velocities without flagging post-rescue equilibration",
     "production.py:3962-3986",
     "Rescue copies positions from the nearest non-stuck replica into a stuck one mid-production; immediately after, that replica is far from equilibrium for its own umbrella ensemble, so samples right after a rescue are not draws from the target biased distribution. Only a counter/message is recorded — no per-sample rescue boundary for MBAR to discard an equilibration window. With F9 (shared seeds) the rescued and source replicas can track each other.",
     "Emit a per-replica rescued_at_step marker; analysis discards one exchange_interval after each rescue; ensure the reinit velocity seed differs from the source; report total excluded frames.", ST_RERUN),
]

LOW = [
    ("F24", "math-stability", "Production Langevin friction default (1.0/ps) is under-damped for the BAOAB middle scheme",
     "system_setup.py:442-447; cli.py:70",
     "γ=1.0/ps (equil uses 10/ps); under-damped, so combined with the 0.8 nm hard-truncated LJ and 4 fs HMR it gives slower temperature-error correction. Deliberate sampling-efficiency choice, not a correctness bug.",
     "Production diagnostic: log running mean kinetic temperature per replica; assert |T_mean − T| < ~3% over a window; make it easy to raise γ if the temperature diagnostic fails.", ST_AGENT),
    ("F26", "implementation", "Distance/contact umbrella init k=0/r0=0; a missed set_window leaves a wrong-but-finite (often unbiased) restraint",
     "forces.py:43-48, 89-93",
     "k=0,r0=0 init is deliberate (unbiased calibration), but a replica whose set_window was never called (new sparse-2D window, resume gap) silently runs unbiased (k=0) rather than failing; for the contact CV, r0=0 with k>0 pulls toward zero contacts.",
     "Init k to NaN (or add a 'k_armed' flag) and assert every production replica's k is finite and >0 before the first step (unless calibrating); log a (replica,k,r0) readback table matching the intended assignment.", ST_AGENT),
    ("F27", "bug", "Contact-CV norm-zero fallback returns an unnormalized count silently mislabeled as a fraction",
     "production.py:3607-3608",
     "On the fast exchange path cv = raw/norm if norm>0 else raw. The denominator can't be 0 in the canonical build, but if contact_norm is ever 0/negative the code returns the raw contact COUNT while the pipeline believes it's a [0,1] fraction — grossly wrong bias matrix, no error.",
     "When contact_normalize and norm≤0, raise or set cv=NaN (so finite-guards reject). Assert at setup that getParameter('contact_norm') == contact_normalization_denominator within 1e-9.", ST_AGENT),
    ("F28", "math-stability", "np.nanpercentile on an all-NaN bootstrap array warns and returns NaN with no explicit guard",
     "adaptive_feedback.py:1912-1913",
     "If every bootstrap overlap resample for a pair is NaN (near-empty windows), np.nanpercentile emits a RuntimeWarning. Result is correctly NaN but via a warning, unlike the explicit 'if not vals: return nan' pattern used elsewhere in the same file. Robustness/clarity only.",
     "Guard: finite=arr[isfinite]; ci=percentile(finite,q) if finite.size else nan. Unit-test all-NaN array → no warning, both CI fields NaN.", ST_AGENT),
    ("F29", "math-stability", "If every conformer has a non-finite primary CV, an inf-score garbage seed is still selected for grafting",
     "seeding.py:812-864",
     "Non-finite primary → delta=inf, sorted last (correct when finite seeds exist). But if the whole library is non-finite, ranked[0] still has total=inf and is grafted; no guard that ranked[0] is finite. Masks an upstream CV-evaluation failure.",
     "After ranking, assert math.isfinite(ranked[0][0]) or force NPT-pull fallback for that window. Unit-test all-NaN library → descriptive error or fallback flag, not a silent inf seed.", ST_AGENT),
    ("F30", "implementation", "The two seed scorers use different distance metrics (normalized L1 vs raw squared-L2)",
     "seeding.py:813-827 vs adaptive_production.py:801",
     "Beyond the normalization bug (F4), seeding.py uses normalized L1 while adaptive_production uses squared-L2, so even after a normalization fix the two paths can rank the same pool differently (L2 punishes a large single-axis miss more).",
     "Pick one metric and use it in both (ideally a shared helper); test that both return the same argmin on a shared 2D fixture with crossing tradeoffs.", ST_AGENT),
    ("F31", "math-stability", "Secondary-CV σ clamp differs: OpenMM force floors at 1°, Python evaluator has no degree floor",
     "production.py:778-779 vs cv.py:409,428-429",
     "Agree for any σ≥1° (default 35°), but a user σ<1° makes the applied bias (width 1°) and the recorded CV (true tiny width) use different kernels, breaking the bias/observable identity MBAR relies on. Latent.",
     "Apply the identical max(radians(1.0),…) clamp in the Python evaluator (shared helper); test σ_deg=0.5 → both use the same effective σ.", ST_AGENT),
    ("F32", "implementation", "rama-map OpenMM force lacks the partial-φ/ψ fallback present in the Python evaluator",
     "production.py:823-843 vs cv.py:494-497",
     "For a residue missing φ (N-term) or ψ (C-term) the OpenMM force forms 0.5·(φ+ψ) with the missing term =0; the factor-0.5 cancels in numer/denom so real multi-residue peptides agree. Latent inconsistency only for degenerate terminal-only cases.",
     "Regression test: build both on a topology with a terminal residue missing φ; assert |ss_openmm − ss_python| < 1e-6 to lock the cancellation invariant.", ST_AGENT),
    ("N11", "bug", "Fast-CV path is silently disabled in ALL GaMD run modes (factory resets umbrella/CV force groups to 0, so the group-31/29 lookup never matches)",
     "production.py:2849-2876, 3271-3279, 3603-3617",
     "make_gamd_integrator→factory.get_integrator calls set_all_forces_to_group(system,0), so the fast-CV lookup by group 31/29 never matches, _use_fast_cv_path stays False, and every sample/exchange interval falls back to the positions-based recompute. gamd/hmr-gamd only (CMD keeps the fast path). Performance regression in the primary production mode + a signal the force-group assumptions are stale post-factory (same root cause as F1).",
     "Warn + manifest-record if use_gamd and _fast_primary_force_idx<0; better, capture CV forces by identity before the factory or re-assign groups 31/29 after it. gareus-test-run gamd: assert _use_fast_cv_path True.", ST_RERUN),
    ("N12", "math-stability", "Overlap-histogram pad floor 0.1 is an absolute (Å-scale) constant misapplied to the dimensionless contact-fraction CV",
     "math_helpers.py:74-83",
     "_adaptive_hist_overlap pads the range by max(0.1, 0.02·(hi−lo)). Contact values live in [0,0.8] with adjacent-window ranges ~0.1–0.2 wide, so the 0.1 floor doubles/triples the range, squeezes samples into a few of the 80 bins and INFLATES the overlap estimate for tight contact windows — biasing redundancy/removal toward fewer windows than the data warrant.",
     "Make pad relative (max(frac·(hi−lo),0), no absolute floor) or pass a CV scale. Unit test: two narrow contact dists at 0.10/0.18 → relative-pad overlap < 0.1-floor overlap; distance case unchanged.", ST_RERUN),
    ("N13", "bug", "RNG-state restore on resume is silently swallowed — continues with a fresh, uncontinued stream",
     "production.py:2179-2183",
     "On --resume the exchange RNG state is restored from the manifest; if assignment fails (bit_generator class mismatch, malformed/legacy state) the exception is caught with only a warning and execution continues with the freshly seeded default_rng(args.seed), silently re-aligning post-resume draws to the original seed-2026 sequence. No check that rng_bit_generator matches the live generator.",
     "Assert manifest['rng_bit_generator']==rng.bit_generator.__class__.__name__ before restore; on mismatch raise (or re-seed with a derived value and record it). Round-trip unit test: save→restore→next 100 draws match an un-checkpointed reference.", ST_RERUN),
    ("N14", "implementation", "calib_steps double-counts prep steps vs gamd-openmm's stage model (ntcmd already includes ntcmdprep)",
     "production.py:945-967, 2321-2326, 397-403",
     "gamd-openmm treats ntcmd/nteb as TOTALS including prep, but calib_steps/total_steps ADD prep+main again, so the shared setup runs ~ntcmdprep+ntebprep steps past stage-5 start (defaults: 360000 vs stage_5_start 350001) and inflates the reported step budget. Harmless (stage 5 freezes stats) but a semantic miscount that wastes calibration and can confuse stage timing if equil is reduced.",
     "Assert stage_5_start ≤ calib_steps and calib_steps−stage_5_start ≤ one ntave window; document gamd_cmd/equil_steps are prep-inclusive; compute calib_steps from max(cmd+equil,…)+small margin.", ST_RERUN),
    ("N15", "implementation", "gibbs-walk force-accept records 100% acceptance — masks exchange health in tuning reports",
     "production.py:3561-3562, 3762",
     "gibbs-walk calls _attempt_window_swap with force_accept=True, so acceptance_fraction (consumed by write_exchange_tuning_report) reports ~1.0 regardless of true overlap. The standard REUS health check 'acceptance ~0.2–0.4' cannot detect poor overlap in the default mode — compounding/hiding N1.",
     "For gibbs-walk report a move/stay ratio + selection entropy instead of a trivially-1.0 acceptance; assert the report emits gibbs_moves/gibbs_stays/gibbs_choices (already tracked) when mode startswith 'gibbs'.", ST_RERUN),
    ("N16", "bug", "Silent reject on non-finite swap delta (NaN secondary CV) with no diagnostic counter in the Metropolis path",
     "production.py:3559-3561, 3627-3628",
     "A NaN secondary-CV value makes bias_matrix_kj NaN → delta NaN → _exchange_probability returns 0 → swap silently rejected. The gibbs path increments gibbs_all_nan_skips, but the neighbor/random/all-pair Metropolis paths have no counter, so NaN-driven rejections look like ordinary unfavorable swaps and silently suppress exchange. Given the project's NAN_DIAGNOSTICS crash history, plausible and currently invisible.",
     "On non-finite delta increment exchange_stats['nonfinite_delta_skips'] and debug-log the pair. Unit test: NaN bias entry → counter increments and swap rejected, for all non-gibbs modes.", ST_RERUN),
]

REFUTED = [
    ("Harmonic σ uses +inf sentinel for k≤0 (production) vs NaN (adaptive)", "production.py:992; adaptive_feedback.py:286-299",
     "Adversarial physics-refute verdict (high confidence): NOT a defect. Both sentinels feed only (a) a radius guard that rejects +inf and NaN identically (falls back) or (b) a cosmetic diagnostic column; no consumer distinguishes them. Physically +inf is the correct k→0 limit of √(RT/k); NaN is a benign 'undefined'. Zero behavioral/numerical consequence."),
    ("Per-frame boost not synchronized to the exact sampled frame", "production.py:3315-3390, 180-207",
     "Rerun verifier: code-fact true but is_real=false. The main loop is strictly serialized step_all(chunk)→sample()→attempt_exchanges(); inside sample() the CV and the boost are both read from the SAME sim object at the same point, so there is no frame desync. Not a defect."),
    ("2D grid neighbor pairing is not a perfect matching (a window appears in 2 swaps/sweep)", "production.py:512-528",
     "Rerun verifier: code-fact true but is_real=false. Confirmed empirically that windows recur across pairs, but sequential application with refreshed holder lookups keeps each pairwise swap a valid Metropolis move; non-disjoint pairing is an efficiency/transport property, not a correctness/detailed-balance error. (GPT concurred.)"),
    ("Early-stop coverage gate uses span, not density — internal valleys pass", "adaptive_feedback.py:3181-3221",
     "Rerun verifier: code-fact true (gate is span-based and cannot see an interior unsampled valley alone) but is_real downgraded to false — the span gate is one of several gates; the per-pair overlap/convergence machinery (when not on the 1D plateau-override path, N6) catches internal gaps, so the standalone span gate is not a genuine defect in the default 2D pipeline."),
    ("Contact gap-fill midpoint inserts windows with only spacing-based k", "adaptive_feedback.py:2360-2374, 796-811",
     "Rerun verifier: code-fact true but is_real=false. The midpoint fill assigns _adaptive_feedback_k_from_centers (spacing-based k), which is the same principled k=kBT/σ² rule used for all added windows; a geometric midpoint at halved spacing with spacing-derived k is correct, not a defect."),
    ("set_window swallows secondary-CV parameter assignment (≡ prior F16)", "windows.py:205-213",
     "Rerun verifier: code-fact true (the bare except IS over-broad, matching F16) but the headline impact — a normally-configured 2D run silently collapsing to 1D — is REFUTED: in the documented flow every window carries the secondary force and the arrays are length-matched, so the except does not fire. Net: a low-value robustness nit (narrow the except), not the medium-severity silent-bias claimed; F16 is downgraded accordingly."),
]

GAPS = [
    ("Boost force-group exclusion vs exchange cancellation (the precondition behind F1)",
     "If the dual boost touches umbrella groups 31/29, the 'GaMD cancels from exchange' design fails and every swap uses the wrong acceptance ratio — biasing all PMFs. GPT confirmed the cancellation requires ΔV identical across replicas.",
     "production.py:942-974, 2849-2876, 3556-3558; assert boosted-group bitmask ∩ {31,29} = ∅ at startup (see F1 check)."),
    ("NPT MonteCarloBarostat × GaMD × REUS three-way interaction",
     "Per-replica independent barostats put replicas at different volumes; the exchange Δ has no PV/volume term and the barostat MC acceptance may use the boosted or bare energy — either breaks the intended NPT Boltzmann target.",
     "system_setup.py:351-377, production.py:2629-2645. Record per-replica box volume; assert volume-distribution overlap between exchanging windows; verify gamd-openmm barostat compatibility, or calibrate/run NVT."),
    ("Gibbs-walk exchange always-accepts its sampled target (force_accept=True)",
     "The heat-bath choice never applies a Metropolis/Barker correction for proposal asymmetry; for a single-replica reassignment among windows occupied by other replicas, the full conditional must account for the displaced replica, or detailed balance fails and window occupancy is biased.",
     "production.py:3715-3764, _attempt_window_swap:3545-3578. Toy 3-window/3-replica with known stationary dist; assert empirical occupancy matches Boltzmann; compare to neighbor-Metropolis."),
    ("Adaptive-production re-calibrates GaMD every epoch (resume=False)",
     "Each epoch re-derives Vmax/Vmin/Vavg/σ from scratch, so per-frame boost magnitude is inconsistent across epochs; pooling exp(βΔV) weights across epochs (or comparing epoch PMFs) mixes different boost references — compounds F2/F3.",
     "adaptive_production.py:3083+, 3240-3274, 3427-3496. Record per-epoch calibration constants; assert reuse or never pool boost weights across differing references; warn on drift."),
    ("Contact-switch steepness × 4 fs HMR × 1.0/ps friction stability margin",
     "A stiff tanh contact switch summed over many pairs can produce large restraint forces near r0; combined with 4 fs HMR and weak thermostat coupling this is a plausible mechanism for the documented CRASH/NAN production artifacts.",
     "forces.py:51-94, cv.py:816-819; compute worst-case per-atom restraint force = k·dCV/dx at the steepest switch point and assert force·dt²/mass ≪ a stability bound; 4 fs integration test for NaN/energy blow-up."),
    ("NaN-secondary seeds (F5) propagate into window initialization",
     "Seeds favored precisely because their secondary CV is unevaluable become the starting structures for windows with a finite secondary center, starting far outside the secondary restraint (large initial bias energy → instability / long relaxation / under-equilibrated MBAR data).",
     "seeding.py:388+, 800-867. After selection assert each window's seed has a finite secondary CV within a few σ of its center; histogram initial secondary-bias energy per window, flag > kT."),
    ("Dual boost-extraction-path consistency (native sum vs globals-inferred total)",
     "If get_boost_potentials sum and the integrator-globals fallback use different conventions (one already summed, one single-component), the recorded 'total' boost silently changes scale with gamd-openmm version, biasing exp(βΔV).",
     "production.py:329-339, 142-160, 180-211, ~3378. Assert the two paths agree on the same frame and total == dihedral+nonbonded when both present; log boost_source and assert it doesn't switch mid-run."),
]

CHECKS = [
    ("C1", "startup assert", "F1 / gap-1", "Boost invariant to umbrella k (d(boost)/dk=0); boosted-group ∩ {29,31}=∅", "Boost double-counts umbrella → wrong exchange + reweight"),
    ("C2", "diagnostic", "F2 / F3", "Per-window ESS + per-window mean boost vs center; flag max/median > 3×", "Pooled-ESS artifact masking usable per-window data"),
    ("C3", "unit test", "F4 / F22 / F30", "Both seed scorers share one normalized, weighted helper; parity on a fixture", "Scale-biased / weight-ignoring seed assignment"),
    ("C4", "unit test", "F5 / gap-6", "NaN-secondary seed is penalized or skipped, never preferred", "Bias toward un-scorable seeds; bad window init"),
    ("C5", "runtime assert", "F6 / F19", "np.isfinite(bias_matrix_kj).all(); nonfinite-bias counter per replica", "Silent exchange freeze + cached NaN bias"),
    ("C6", "unit test", "F7", "Score==label at named basins; 50/50 mix ≠ any basin; snap dense rama centers", "Unphysical 1D-ladder intermediate windows"),
    ("C7", "startup assert", "F8", "NonbondedForce switching on, switchDist≈cutoff−0.1; cutoff ≥0.9 nm", "Truncated-dispersion PMF bias + force discontinuity"),
    ("C8", "startup/probe", "F9 / F10 / F12 / gap-7", "Boost is a Quantity; native==globals total; recomputed≈reported dV", "Mislabeled/ double-counted / stale boost units"),
    ("C9", "metadata+assert", "F11 / F13", "Record unbiased potential + reduced_bias_includes_gamd_boost flag", "MBAR run on boosted ensemble; wrong 'V' for dV"),
    ("C10", "unit test", "F14 / F31 / F32", "fast vs slow secondary-CV evaluators bit-for-bit equal (eps, clamp, σ floor)", "Bias≠observable mismatch in MBAR"),
    ("C11", "diagnostic", "F15", "Predicted neighbor overlap O≥~0.1 per adjacent window pair", "Marginal overlap → poor exchange / MBAR variance"),
    ("C12", "narrow except + assert", "F16 / F26", "set_window readback ss_k/k==request; k armed (NaN→finite>0) before prod", "Replica in wrong/stale thermodynamic state"),
    ("C13", "unit test", "F17 / F27 / C-norm", "self-test asserts OpenMM CV/contact_norm == manual CV within 1e-6; norm>0", "Contact CV bias inconsistency"),
    ("C14", "guard + test", "F18 / F28 / F29", "rank(cov)==2 before KDE (jitter/marginal fallback); finite guards on percentile/seed", "Silent loss of adaptive round; garbage seed"),
    ("C15", "config + diag", "F23 / F24 / gap-2 / gap-4", "Calibrate GaMD NVT; per-replica T and box-volume overlap; per-epoch calib constants", "Mis-scaled boost; NPT/epoch ensemble inconsistency"),
    ("C16", "toy + assert", "gap-3", "Gibbs-walk occupancy matches Boltzmann on a 3-window toy (detailed balance)", "Biased window occupancy → biased PMF"),
    ("C17", "integration test", "F25 / gap-5", "Clamp rama-map OpenMM CV to [−1,1]; 4 fs contact-primary run, no NaN/blow-up", "Restraint-force spike → integrator instability"),
    ("C18", "unit test", "N1", "gibbs-walk weights: no upper clip; π·T=π on a K=5 heat-bath; detailed balance < 1e-12", "Biased default exchange kernel → wrong window occupancy"),
    ("C19", "runtime assert", "N2 / N3 / N4", "2D removal checks per-cell (not pooled) overlap; remove_block_radius ≥1; re-validate survivor-pair overlap", "Orphaned 2D cell / untested wide ladder gap"),
    ("C20", "assert + test", "N9 / N4 / N10", "Distinct per-replica integrator seeds; per-segment seed offset; randomize-replica-velocities default True", "Correlated replicas / re-correlated segments → ESS↓"),
    ("C21", "warn + diagnostic", "N11 / N15 / N16", "Warn if GaMD fast-CV path disabled; gibbs reports move/stay not 1.0; nonfinite_delta_skips counter", "Silent perf regression / masked exchange health"),
]

VERIFIED_OK = [
    ("Contact-CV force-constant units", "cv.py:178 primary_k_to_openmm_value; production.py:619",
     "The contact CV is a dimensionless fraction; k is converted with kcal_to_kj (×4.184) in contact mode and "
     "kcal_a2_to_kj_nm2 (×418.4) only in distance mode — branched correctly by CV type, no 100× error."),
    ("CV-side ≡ force-side contact switch", "forces.py:81 vs cv.py:816/943",
     "Biased and recorded CV use the identical logistic switch 0.5·(1−tanh(0.5·β·(r−r0))) on both the OpenMM and Python "
     "sides, so MBAR sees a consistent CV; the tanh identity also avoids exp overflow."),
    ("Umbrella ½ factors and unit conversions", "forces.py:43,89; production.py bias matrices",
     "Every restraint energy carries the explicit 1/2 (distance, contact, secondary). Exchange/MBAR bias matrices recompute "
     "in user units and convert with the same 1/2 — algebraically identical to the OpenMM force in nm/kJ."),
    ("Periodic CV math (no wrap/averaging errors)", "math_helpers.py:115; cv.py:410; production.py:211",
     "Torsions use the Praxeolitic atan2 formula; rama/secondary scores use a von-Mises kernel exp(−(1−cos(θ−θ0))/σ²). No "
     "linear angle averaging and no squared-difference across the ±π wrap."),
    ("HMR via createSystem(hydrogenMass)", "system_setup.py:363-366",
     "HMR is delegated to OpenMM (hydrogenMass=3.024 amu), which conserves total mass and handles water/virtual sites — no "
     "hand-rolled repartitioning bug; HBonds + rigidWater + 4 fs is the standard recipe and valid for the dihedral boost."),
    ("Exchange numerical-stability guards", "production.py:3504-3517, 3738-3754",
     "REUS/Gibbs reweighting uses log-sum-exp with max-subtraction and clips reduced energies to [−745,0]; Metropolis guards "
     "isfinite and clamps acceptance to [0,1]. No overflow-prone bare exp on positive arguments in the kernels."),
]

# severity tally over detailed + tabled findings
ALL_SEV = ([f["severity"] for f in CRIT_HIGH] + [r[1] and "medium" for r in []]
           + ["medium"] * len(MEDIUM) + ["low"] * len(LOW))
TALLY = {s: ALL_SEV.count(s) for s in SEV_COLOR}

EXEC = (
    "The GAREUS engine is, at the level of locally-verifiable correctness, well built: umbrella restraint energies carry the "
    "correct 1/2 factor, force-constant unit conversions are branched correctly by CV type, the contact CV uses an "
    "overflow-free logistic switch implemented identically on the OpenMM and Python sides, periodic CVs use a proper "
    "von-Mises kernel (no ±π-wrap error), HMR is delegated to OpenMM, and the exchange kernels use log-sum-exp with "
    "max-subtraction. No factor-of-two or unit error was found in the bias energies. "
    "The real risks are CROSS-SUBSYSTEM and statistical-mechanical, where each piece is locally correct but the joint "
    "estimator is biased. The single most important issue (F1) is now CONFIRMED against the gamd-openmm source: the boost "
    "factory calls set_all_forces_to_group(system, 0), folding the per-window umbrella (group 31) and secondary-CV (group "
    "29) restraints into the boosted energy, so the GaMD boost is an explicit function of the window-dependent umbrella "
    "bias. The REUS exchange acceptance omits the boost (valid only if the boost excludes those groups — it does not) and a "
    "single free-peptide calibration is frozen and reused for every window. This is the verified mechanism behind the "
    "documented 'GaMD reweighting ESS effectively zero' and biases the sampled ensemble and all reweighted free energies. "
    "A second confirmed high-severity exchange defect: the CONFIGURED-DEFAULT 'gibbs-walk' scheme upper-clips its selection "
    "weights to 0 and force-accepts, so the kernel does not preserve the window-assignment distribution (every current CLN025 "
    "run uses it). Other confirmed highs: a divergent adaptive seed scorer that ignores CV normalization and weight, both "
    "seed scorers favoring un-scorable (NaN-secondary) seeds, the live bias matrix not masking NaN (silent exchange freeze), "
    "axis-pooled 2D window pruning that can orphan a depopulated cell, very-aggressive double-removal gaps, cross-segment RNG "
    "re-correlation, the rama-map 1D label ladder making intermediate windows physically uninterpretable, and a short 0.8 nm "
    "LJ cutoff with no switching function."
)

# ============================================================================
# BUILD
# ============================================================================
doc = Document()
for section in doc.sections:
    section.left_margin = Inches(0.85)
    section.right_margin = Inches(0.85)
    section.top_margin = Inches(0.85)
    section.bottom_margin = Inches(0.85)

title = doc.add_heading("GAREUS Sampling Engine — Statistical-Mechanics & MD Correctness Review", 0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
sub = doc.add_paragraph(
    "GaMD boost + REUS umbrella sampling + collective variables + system setup + adaptive windows + seeding\n"
    "chignolin CLN025 (GYDPETGTWG) · OpenMM workflow")
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
for r in sub.runs:
    r.italic = True
    r.font.color.rgb = RGBColor(0x44, 0x44, 0x44)
doc.add_paragraph()

# 1 Scope & method
heading(doc, "1. Scope and Method", 1)
body(doc,
     "This review targets the GAREUS sampling engine — the code that defines the biased ensembles and produces the raw "
     "biased samples: GaMD boost construction/calibration, REUS umbrella bias forces and replica-exchange acceptance, "
     "collective-variable definitions, HMR/integrator/thermostat setup, adaptive window placement, and GENPEPT-seed-to-window "
     "assignment. Post-hoc free-energy estimation (MBAR/CE2), the GENPEPT generator internals, and helper/infra modules "
     "(logging, TUI, provenance, I/O) are out of scope.")
add_table(doc, ["In scope", "Out of scope"],
          [["production.py, adaptive_production.py", "analysis.py, energy_decomposition.py"],
           ["adaptive_feedback.py, windows.py", "cv_discovery.py, diagnostics.py*"],
           ["cv.py, forces.py", "genpept_prescan.py, genpept_window_prior.py"],
           ["system_setup.py, seeding.py", "logger.py, tui.py, query.py, helptext.py"],
           ["math_helpers.py, config.py, units.py, constants.py", "io.py, store.py, provenance.py, cli.py"]],
          header_fill='D9E1F2', font_size=9)
note(doc, "*diagnostics.py is out of scope as a module, but F3 cites it because it is the proximate cause of the documented 'ESS≈0' symptom.")
doc.add_paragraph()
body(doc,
     "Method: ten physics/code dimensions were each reviewed against first-principles statistical mechanics (GaMD equations "
     "of Miao et al. 2015; umbrella-sampling and replica-exchange theory; circular statistics; log-sum-exp; HMR mass "
     "conservation), with a Codex/GPT consultation on the load-bearing derivations. Each candidate finding was to be "
     "adversarially verified by two lenses — a code-fact lens and a physics-refute lens.")
callout(doc,
        "Verification status: this report merges two passes. The first pass produced findings F1–F32 (six review dimensions); "
        "its automated verification was interrupted by an API session limit, so each CRITICAL/HIGH was re-read in the main "
        "thread and the load-bearing physics confirmed via a Codex/GPT derivation, while MEDIUM/LOW marked 'unverified' are "
        "agent-reported and code-cited but not independently re-read. A second (rerun) pass then completed the four killed "
        "review dimensions (gamd-boost, reus-exchange, adaptive-windows, code-bugs) → findings N1–N16, EACH adversarially "
        "verified by a combined code-fact + physics-refute lens (status 'confirmed by rerun'); 6 candidates were refuted (§7). "
        "Crucially the rerun confirmed F1 against the gamd-openmm source. Finally, all 22 prior MEDIUM/LOW findings (F9–F32) "
        "were independently re-read in the main thread: 21 are confirmed code-accurate with their severities upheld, F12 is "
        "REFUTED (no frame desync — serialized step→sample→exchange loop), and scope corrections are noted inline on F10 "
        "(double-count risk only for non-default boost types) and F15 (the overlap-direction claim was backwards). Net: every "
        "finding in this report is now verified — no 'unverified' items remain.", fill="FFF2CC")

heading(doc, "1.1 Review Dimensions", 2)
add_table(doc, ["#", "Dimension", "Core question", "Outcome"],
          [["1", "GaMD boost potential", "ΔV=½k(E−V)²/(Vmax−Vmin), threshold, dual boost, units", "RERUN: F1 confirmed via gamd-openmm source; +N5,N11,N14"],
           ["2", "GaMD reweighting data", "Per-frame ΔV; βΔV sign/units; ESS-collapse cause", "F2,F3,F9-F13 + GPT"],
           ["3", "Umbrella bias forces", "½k(CV−c)², units, periodic restraint, spacing", "core sound; F15,F16,F25,F26"],
           ["4", "Replica-exchange accept", "Metropolis Δ with bias; detailed balance; boost-aware", "RERUN: N1 (gibbs default biased),N15,N16; 1 refuted"],
           ["5", "CV definitions & circular math", "switch limits; rama map; atan2; ±π wrap", "sound; F7,F14,F20,F31,F32"],
           ["6", "HMR / integrator / thermostat", "mass conservation; constraints; 4 fs; Langevin/barostat", "sound; F8,F23,F24"],
           ["7", "Adaptive window placement", "overlap metric; no orphaned windows; convergence", "RERUN: N2,N3,N6,N7,N8,N12,F18; 2 refuted"],
           ["8", "Seed scoring (active-CV)", "2D uses both CVs; no single-CV regression; assignment", "F4,F5,F21,F22,F29,F30"],
           ["9", "Numerical stability", "exp overflow; log/sqrt domain; zero denom; empty reductions", "disciplined; F18,F19,F28 + verified OK"],
           ["10", "General correctness bugs", "indexing; checkpoint; aliasing; units; RNG", "RERUN: N4,N9,N10,N13; 1 refuted"]],
          header_fill='D9E1F2', font_size=8)
doc.add_paragraph()

# 2 Executive summary
heading(doc, "2. Executive Summary", 1)
body(doc, EXEC)
add_table(doc, ["Critical", "High", "Medium", "Low", "Total"],
          [[TALLY["critical"], TALLY["high"], TALLY["medium"], TALLY["low"],
            TALLY["critical"] + TALLY["high"] + TALLY["medium"] + TALLY["low"]]],
          header_fill='D9E1F2', font_size=10)
doc.add_paragraph()
callout(doc,
        "Top action: F1 is CONFIRMED — the gamd-openmm factory's set_all_forces_to_group(system, 0) boosts the umbrella "
        "groups 31/29, so the exchange-cancellation assumption is false and the reweighting is corrupted. Fix first: restrict "
        "the boost to physical force groups (or recalibrate Vmax/Vmin per window). Second: fix the default 'gibbs-walk' "
        "selection-weight upper-clip (N1). These two alone are the most likely cause of the observed ESS≈0 and tail-RMSE "
        "non-convergence.", fill="FCE4E4")

# 3 Critical & High
heading(doc, "3. Confirmed Findings — Critical & High", 1)
order = {"critical": 0, "high": 1}
for f in sorted(CRIT_HIGH, key=lambda x: order.get(x["severity"], 9)):
    finding_block(doc, f)

# 4 Medium
heading(doc, "4. Medium-Severity Findings", 1)
add_table(doc, ["ID", "Sev", "Cat", "Title / location", "Why it matters & check"],
          [[m[0], "medium", CAT_LABEL.get(m[1], m[1]).split()[0], m[2] + "\n[" + m[3] + "]  (" + m[6] + ")",
            m[4] + "  CHECK: " + m[5]] for m in MEDIUM],
          header_fill='FDF3D8', font_size=8, sev_col=1, widths=[0.4, 0.6, 0.8, 3.1, 3.6])
doc.add_paragraph()

# 5 Low
heading(doc, "5. Low-Severity Findings", 1)
add_table(doc, ["ID", "Sev", "Cat", "Title / location", "Why it matters & check"],
          [[l[0], "low", CAT_LABEL.get(l[1], l[1]).split()[0], l[2] + "\n[" + l[3] + "]  (" + l[6] + ")",
            l[4] + "  CHECK: " + l[5]] for l in LOW],
          header_fill='E8F2DE', font_size=8, sev_col=1, widths=[0.4, 0.5, 0.8, 3.1, 3.7])
doc.add_paragraph()

# 6 Verified correct
heading(doc, "6. Verified Correct (Reviewer Due Diligence)", 1)
body(doc, "Subtle physics specifically checked and found implemented correctly — recorded so the verification is on record.")
for area, where, txt in VERIFIED_OK:
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(1)
    r = p.add_run("✓ " + area)
    r.bold = True
    r.font.size = Pt(10)
    r.font.color.rgb = RGBColor(0x2E, 0x5E, 0x2E)
    wp = doc.add_paragraph()
    wp.paragraph_format.space_after = Pt(0)
    wr = wp.add_run(where)
    wr.font.name = "Courier New"
    wr.font.size = Pt(8.5)
    wr.font.color.rgb = GREY
    body(doc, txt)

# 7 Examined, not a defect
heading(doc, "7. Examined — Not a Defect (Refuted)", 1)
for title_txt, where, txt in REFUTED:
    p = doc.add_paragraph()
    r = p.add_run("✗ " + title_txt)
    r.bold = True
    r.font.size = Pt(10)
    r.font.color.rgb = GREY
    wp = doc.add_paragraph()
    wp.paragraph_format.space_after = Pt(0)
    wr = wp.add_run(where)
    wr.font.name = "Courier New"
    wr.font.size = Pt(8.5)
    wr.font.color.rgb = GREY
    body(doc, txt)

# 8 Cross-subsystem gaps
heading(doc, "8. Cross-Subsystem Open Questions", 1)
body(doc,
     "The per-dimension reviews are strong on local correctness but under-examine interactions between subsystems — where "
     "the 'each piece looks fine but the joint estimator is biased' bugs live. These need investigation (and four of the ten "
     "review dimensions lost their agent to the session limit, so these partly stand in for them).")
for i, (gap, why, where) in enumerate(GAPS, 1):
    p = doc.add_paragraph(style='List Number')
    rb = p.add_run(gap)
    rb.bold = True
    rb.font.size = Pt(10)
    wp = doc.add_paragraph()
    wp.paragraph_format.left_indent = Inches(0.3)
    wp.paragraph_format.space_after = Pt(1)
    wr = wp.add_run("Why: " + why)
    wr.font.size = Pt(9.5)
    lp = doc.add_paragraph()
    lp.paragraph_format.left_indent = Inches(0.3)
    lp.paragraph_format.space_after = Pt(4)
    lr = lp.add_run("Where / check: " + where)
    lr.font.size = Pt(9)
    lr.font.color.rgb = GREY

# 9 Check suite
heading(doc, "9. Proposed Check Suite", 1)
body(doc,
     "Concrete, mostly OpenMM-free checks to lock in correctness. Types: startup assertions (cheap always-on guards), unit "
     "tests (tests/, no OpenMM/PeptideBuilder), runtime asserts, and diagnostics (post-run sanity).")
add_table(doc, ["ID", "Type", "Targets", "Check", "Catches"],
          [[c[0], c[1], c[2], c[3], c[4]] for c in CHECKS],
          header_fill='D9E1F2', font_size=8, widths=[0.4, 1.0, 1.0, 3.0, 2.1])
doc.add_paragraph()

# 10 GaMD physics note
heading(doc, "10. GaMD/REUS Reweighting — Physics Note (Codex/GPT consultation)", 1)
body(doc, "The load-bearing statistical mechanics, confirmed by a Codex/GPT derivation:")
heading(doc, "10.1 Why the boost must exclude the umbrella groups", 2)
body(doc, "Each window samples π_k(x) ∝ exp(−β[U0(x) + w_k(CV) + ΔV_k(x)]). The boost cancels from the exchange acceptance ratio only if ΔV_i(x) = ΔV_j(x) = ΔV(x) for all windows — i.e. ΔV is the SAME function of configuration in every replica.")
equation(doc, "If ΔV_k(x) = g[U0(x) + w_k(x)]   (boost includes the umbrella):")
equation(doc, "Δboost = g[U0(x_j)+w_i(x_j)] + g[U0(x_i)+w_j(x_i)]")
equation(doc, "         − g[U0(x_i)+w_i(x_i)] − g[U0(x_j)+w_j(x_j)]   ≠ 0  in general")
body(doc, "Shared calibration constants do NOT make this cancel, because the boost argument still changes with (K_k, c_k). Therefore omitting the boost from the acceptance ratio is valid only when the boosted force-group set excludes the window-dependent umbrella groups (31, and 29 if the secondary bias is window-dependent).")
body(doc, "CONFIRMED status: the rerun verified against the gamd-openmm source that the boost factory calls set_all_forces_to_group(system, 0), so the umbrella groups 31/29 ARE inside the boosted energy. The precondition is therefore violated in the current build — this is finding F1, not a hypothetical.")
heading(doc, "10.2 Why pooled ESS collapses even when per-window reweighting is fine", 2)
body(doc, "Per-window unbiasing is formally exact given the actual recorded boost and adequate overlap:")
equation(doc, "P_unboosted,k(x) ∝ P_boosted,k(x) · exp(+β·ΔV(x))")
equation(doc, "ESS = (Σ_n W_n)² / Σ_n W_n² ,   W_n = exp(β·ΔV_n)")
body(doc, "But a single global Vmax/Vmin (F2) gives each window a different ΔV distribution; pooling the W_n across windows makes a few high-boost frames dominate, so ESS collapses. This is a variance/overlap failure, not necessarily a formal bias — the fix is per-window (or per-CV-bin) ESS and reweighting (F3), not discarding the data.")
heading(doc, "10.3 When 2nd-order cumulant reweighting stays valid", 2)
equation(doc, "⟨exp(β·ΔV)⟩ ≈ exp[β·⟨ΔV⟩ + (β²/2)·Var(ΔV)]   per CV bin")
body(doc, "Conditions: (i) the per-frame ΔV recorded is the actually-applied boost (F9,F10,F12); (ii) exchange sampled the correct stationary distribution (F1, gap 3); (iii) the boost is excluded from — or consistently included in — the exchange criterion; (iv) within each CV bin ΔV has modest, approximately-Gaussian fluctuations; (v) adequate umbrella overlap for MBAR (F15). The clean alternative is to treat the umbrella bias AND the boost as known bias energies in MBAR directly.")

# 11 References
heading(doc, "11. References", 1)
add_table(doc, ["Authors (year)", "Topic", "Journal"],
          [["Miao, Feher, McCammon (2015)", "GaMD method + 2nd-order cumulant reweighting", "JCTC 11, 3584"],
           ["Hamelberg, Mongan, McCammon (2004)", "Accelerated MD boost potential", "JCP 120, 11919"],
           ["Shirts & Chodera (2008)", "MBAR", "JCP 129, 124105"],
           ["Sugita, Kitao, Okamoto (2000)", "Replica-exchange umbrella sampling (REUS)", "JCP 113, 6042"],
           ["Hopkins, Le Grand, Walker, Roitberg (2015)", "Hydrogen mass repartitioning, 4 fs", "JCTC 11, 1864"],
           ["Wang, Miao et al. (2021)", "Combined GaMD + umbrella sampling", "JPCL 12, 6871"]],
          header_fill='D9E1F2', font_size=9)

doc.save(OUT_DOCX)
n_detail = len(CRIT_HIGH)
n_tab = len(MEDIUM) + len(LOW)
print(f"Saved: {OUT_DOCX}")
print(f"  detailed findings (crit/high): {n_detail}")
print(f"  tabled findings (med/low): {n_tab}")
print(f"  total findings: {n_detail + n_tab}  | checks: {len(CHECKS)} | gaps: {len(GAPS)} | verified-ok: {len(VERIFIED_OK)}")
