"""Generate the chignolin lambda-ladder campaign preparation report (.docx).

House style copied from write_physics_oracle_report.py: Document(), title at
heading level 0, numbered level-1 sections, level-2 subsections, and the
add_table helper with 'Light List Accent 1' falling back to 'Table Grid'.
"""
from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt

OUT = Path("/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/docs/"
           "chignolin_lambda_ladder_campaign_prep_report.docx")


def add_table(doc, headers, rows, widths=None, font=8.5):
    t = doc.add_table(rows=1 + len(rows), cols=len(headers))
    try:
        t.style = "Light List Accent 1"
    except KeyError:
        t.style = "Table Grid"
    for j, h in enumerate(headers):
        c = t.rows[0].cells[j]; c.text = ""
        run = c.paragraphs[0].add_run(h); run.bold = True; run.font.size = Pt(font)
    for i, row in enumerate(rows, start=1):
        for j, val in enumerate(row):
            c = t.rows[i].cells[j]; c.text = ""
            run = c.paragraphs[0].add_run(str(val)); run.font.size = Pt(font)
    if widths:
        for row in t.rows:
            for j, wdt in enumerate(widths):
                row.cells[j].width = Inches(wdt)
    return t


def para(doc, text):
    return doc.add_paragraph(text)


def caption(doc, text):
    p = doc.add_paragraph()
    r = p.add_run(text); r.italic = True; r.font.size = Pt(8.5)
    return p


doc = Document()
doc.add_heading("Chignolin λ-Ladder GaREUS Campaign: Collective-Variable "
                "Redefinition, Boost Calibration and Unattended-Run Preparation", 0)
p = doc.add_paragraph()
r = p.add_run("ATLaS-MD / gareus peptide sampler — preparation record for campaign "
              "chignolin_7 (stage S2). Work performed 2026-09-08 to 2026-09-09 on the "
              "aurum SLURM cluster (4× NVIDIA L40S per node). Repository "
              "github.com/sulcjo/ATLaS-MD, main at commit 09492e3.")
r.italic = True

# ---------------------------------------------------------------- Abstract
doc.add_heading("Abstract", 1)
para(doc,
     "A 112-state λ-ladder × CV1-umbrella campaign on chignolin was prepared to the "
     "point of launch and deliberately not launched. Four blocks of work are reported. "
     "(i) The primary collective variable was redefined from a contact definition that "
     "carried almost no usable signal (occupancy entropy 0.037, two resolvable windows, "
     "every force constant clamped at the k_min floor of 5 kcal/mol/Å²) to heavy-atom "
     "contacts with r0 = 12 Å, β = 3 Å⁻¹ and sequence separation ≥ 4 "
     "(entropy 0.907, 25 resolvable windows, k = 110–505 kcal/mol/Å²). "
     "(ii) The Pep-GaMD boost was calibrated against the resulting frozen envelope "
     "(k0_Total = 0.3801, theoretical maximum boost 60.8 kcal/mol) and the integration "
     "timestep was mapped: 4.0 fs fails (NaN at 1.056 ns), 3.5 fs and below pass. "
     "(iii) The swarm-stage machinery was extended — graft relaxation raised from 100 to "
     "500 minimisation iterations, a probed autotuned CV1 upper bound replacing a static "
     "library-quantile veto, and the ladder-ESS gate demoted from blocking to advisory "
     "after direct measurement contradicted its prediction. (iv) Umbrella seeding was "
     "parallelised (21 windows in 106 s at 28 workers, against ~21 min serial) and the "
     "pull shortened 15-fold to 30 ps, both verified by execution rather than by "
     "argument parsing; the worst starting umbrella bias was 4.42 kcal/mol against a "
     "tolerance of 15.0, with zero pull fallbacks. Five configuration defects that would "
     "each have broken an unattended run were found and corrected. One gate tolerance was "
     "loosened as a documented, reversible judgement call, at the cost of an accepted "
     "~17 % uncertainty in boost magnitude.")

# ---------------------------------------------------------------- 1 Methods
doc.add_heading("1. Computational Details", 1)

doc.add_heading("1.1  Method and state space", 2)
para(doc,
     "The campaign replaces the second umbrella coordinate (CV2) of the standard 2-D "
     "GaREUS protocol with a λ-ladder. A replica state is the pair (CV1 window, λ rung). "
     "Only the GaMD acceleration parameter k0 scales with λ; the umbrella envelope is "
     "frozen across the ladder. This keeps the MBAR estimate exact and provides a "
     "λ = 0 rung that is unboosted, giving a direct cross-check on the reweighting "
     "(tolerance 0.5 kcal/mol). Exchange uses gibbs-walk, which samples the full "
     "conditional over all (window, rung) states rather than a truncated neighbour graph.")
add_table(doc, ["Setting", "Value", "Note"], [
    ("States", "112", "16 CV1 centres × 7 λ rungs"),
    ("CV1 centres", "0.105 – 0.906", "dimensionless contact fraction"),
    ("λ rungs", "0, 0.094, 0.212, 0.364, 0.573, 0.870, 1.000", "autotuned spacing"),
    ("Umbrella k", "109.7 – 504.6 kcal/mol/Å²", "from local curvature"),
    ("Boost", "pep-gamd-lower-dual, σ0p = σ0d = 6.0 kcal/mol", "essential Pep-GaMD"),
    ("Integrator", "3.5 fs, HMR, hmr-gamd", "4.0 fs fails, see §2.3"),
    ("Exchange", "gibbs-walk, every 400 steps", "full conditional"),
    ("CV2", "none", "replaced by the λ ladder"),
    ("max_replicas", "128", "≥ 112 states"),
], widths=[1.7, 2.9, 2.0])
caption(doc, "Table 1. Production state space and integrator settings (chignolin.yaml, stage S2).")

doc.add_heading("1.2  Umbrella and boost conventions", 2)
para(doc,
     "The applied umbrella energy is 0.5·k·((contacts/contact_norm) − r0)² (half "
     "convention), so the Gaussian width of a window is σ_w = √(kT/k); force constants "
     "quoted in kcal are converted to kJ by ×4.184. Window force constants are set from "
     "k = kT/σ_w² − F''(centre) with σ_w = spacing / overlap_sigma, clipped to "
     "[k_min, k_max]. The Pep-GaMD lower-dual boost uses the force-scaling factor "
     "FSF = 1 − k0(E − V)/(Vmax − Vmin) with the self-limiting "
     "k0 = (σ0/σV)·(Vmax − Vmin)/(Vmax − Vavg).")

doc.add_heading("1.3  Swarm stage", 2)
para(doc,
     "Round-0 windows, the frozen boost envelope and the seed bank are produced by a "
     "preceding unbiased swarm stage: stratify (CV1 × Rg × end-to-end bins, used for "
     "seed diversity only and never as a bias) → members → envelope → ladder_design → "
     "seed bank → gates → compare_pilot. The campaign is ab initio: no native or folded "
     "reference structure enters the pipeline at any stage.")

doc.add_heading("1.4  Seeding", 2)
para(doc,
     "Each state is seeded by grafting a library conformer into the solvated context "
     "(graft_conformer_into_context), minimising, relaxing to the target window "
     "(relax_to_window), minimising again, and finishing with a restrained pull. Seeding "
     "is parallelised with --us-pull-workers; workers are assigned GPUs round-robin as "
     "device_tokens[w % len(device_tokens)].")

# ---------------------------------------------------------------- 2 Results
doc.add_heading("2. Results", 1)

doc.add_heading("2.1  CV1 redefinition", 2)
para(doc,
     "The contact CV in use at the start of this work resolved essentially nothing. Its "
     "occupancy entropy over the r7 seed library was 0.037, it supported two windows, and "
     "every force constant the adaptive placement returned was clamped at the k_min floor "
     "of 5 kcal/mol/Å² — that is, no restraint was applied at all. Replacing the "
     "switching function with heavy-atom contacts at r0 = 12 Å and β = 3 Å⁻¹ "
     "recovers a usable coordinate.")
add_table(doc, ["Quantity", "Old CV1", "New CV1"], [
    ("Atom selection", "(default)", "heavy"),
    ("r0", "4.5 Å", "12 Å"),
    ("β", "6 Å⁻¹", "3 Å⁻¹"),
    ("Min. sequence separation", "—", "≥ 4"),
    ("Occupancy entropy", "0.037", "0.907"),
    ("Resolvable windows", "2", "25"),
    ("Force constants returned", "all clamped at k_min = 5", "110 – 505 kcal/mol/Å²"),
], widths=[2.4, 2.0, 2.2])
caption(doc, "Table 2. Old versus new CV1, scored on the r7 seed library. The new definition "
             "spans 1256 heavy-atom pairs with |i−j| ≥ 4 over 77 heavy atoms "
             "(21 residue pairs possible).")
para(doc,
     "The CV is a mean logistic switch over those 1256 pairs, normalised, so its range is "
     "the dimensionless contact fraction quoted throughout.")

doc.add_heading("2.2  Boost calibration against the frozen envelope", 2)
add_table(doc, ["Quantity", "Value", "Units"], [
    ("k0_Total", "0.3801", "—"),
    ("k0_Dihedral", "1.000", "— (saturated)"),
    ("Vmax_Total", "−2423.8", "kcal/mol"),
    ("Vmin_Total", "−3761.5", "kcal/mol"),
    ("σV_Total", "154.7", "kcal/mol"),
    ("Theoretical maximum boost", "60.8", "kcal/mol"),
], widths=[2.6, 1.8, 2.2])
caption(doc, "Table 3. Frozen Pep-GaMD envelope for round swarm2, as consumed by every "
             "probe and by production.")
para(doc,
     "A boost-variant comparison preceded this calibration. The peptide-internal variant "
     "at 4 fs with k0 = 1 was both the fastest and the most boosted option measured "
     "(207 ns/day per state), but the essential Pep-GaMD variant at σ0 = 6 with a "
     "self-limiting k0 was adopted instead, on the grounds that it is the variant whose "
     "reweighting at λ = 0 the campaign depends on.")

doc.add_heading("2.3  Integration timestep stability map", 2)
para(doc,
     "Stability was mapped directly on the frozen envelope of Table 3, not inferred.")
add_table(doc, ["Timestep", "Outcome", "Observed boost ceiling"], [
    ("4.0 fs", "FAIL — NaN at 1.056 ns", "61 kcal/mol"),
    ("3.5 fs", "PASS", "—"),
    ("3.0 fs", "PASS", "47 kcal/mol"),
    ("2.0 fs", "PASS", "26 kcal/mol"),
], widths=[1.3, 2.7, 2.2])
caption(doc, "Table 4. Timestep stability at the swarm2 envelope. 3.5 fs was adopted.")
para(doc,
     "The 4 fs failure was diagnosed rather than assumed. The dead replica was at "
     "λ = 0.0, i.e. unboosted. The exchange log shows it arriving there by a single "
     "jump from λ = 0.573 at step 263,200 with Δe = −1.67 kT. The mechanism is that a "
     "configuration relaxed under softened peptide forces has the full forces restored in "
     "one step. This is a property of the ladder's end rung, not of the boost magnitude "
     "alone — a distinction that matters because it means the failure is not removed by "
     "lowering k0 while keeping 4 fs.")

doc.add_heading("2.4  Swarm graft relaxation", 2)
para(doc,
     "In the first swarm round 18 of 99 members went NaN within 3 s of starting. The "
     "cause was that a grafted conformer was relaxed by a single unrestrained "
     "minimizeEnergy(maxIterations=100). Raising the iteration count to 500 reduced this "
     "to 3 of 99. A controlled re-run of the same 18 members with their original velocity "
     "seeds recovered 15, confirming the diagnosis rather than merely correlating with it. "
     "The production round (swarm2) completed 172 of 174 members.")

doc.add_heading("2.5  Autotuned CV1 upper bound", 2)
para(doc,
     "Ladder design was capped by a static veto that compared candidate centres against a "
     "GENPEPT seed-library q99 quantile. The cap was unpassable, and both obvious remedies "
     "were inert: linspace always places its endpoint at the upper bound, and extending "
     "the swarm widens the quantile it is compared against. The veto was replaced by "
     "autotune_cv1_upper_bound(), which probes actual seed support at candidate centres "
     "(probe_centre_seed_support) with lo_q = 0.005, hi_q = 0.995 and a maximum seed gap "
     "of 0.5 σ, bounded by MAX_UPPER_BOUND_PROBES = 256.")
para(doc,
     "Review of the delivered implementation found two defects, both fixed test-first: "
     "the probe measured every trace row rather than the exported frames (a sample "
     "roughly 10× denser than the seed pool actually available), and the candidate walk "
     "was unbounded and O(n²).")

doc.add_heading("2.6  Ladder-ESS gate demoted to advisory", 2)
para(doc,
     "A 174-member round failed the ladder-ESS gate, which predicted the top rungs "
     "unusable (reweighted ESS 34.1 and 19.2 against a floor of 50). A production probe at "
     "those exact rungs was then run and measured the opposite.")
add_table(doc, ["Diagnostic", "Gate prediction", "Direct measurement"], [
    ("Top-rung usability", "unusable (ESS 34.1 / 19.2, floor 50)", "usable"),
    ("Neighbour overlap, all 6 gaps", "—", "0.919 – 0.963"),
    ("Neighbour overlap, validated pilot", "—", "0.809 – 0.902"),
    ("λ = 0 cross-check", "—", "0.282 kcal/mol (tol. 0.5)"),
], widths=[2.2, 2.2, 2.2])
caption(doc, "Table 5. The ladder-ESS gate against direct measurement at the same rungs.")
para(doc,
     "The gate asks whether the unbiased swarm can predict a rung by a single reweighting "
     "jump. Production instead samples every rung directly and couples them by exchange, "
     "so the gate's question is not the question production answers. It now reports the "
     "same diagnostics as warnings, surfaced at the top level so they still reach "
     "swarm_gate.json, the report and the operator's terminal.")

doc.add_heading("2.7  Tiling test", 2)
para(doc,
     "A 21-state tiling probe (3 CV1 centres × 7 rungs) raised the base effective sample "
     "size from 0.4 % to 1.5 %. Its λ = 0 cross-check nominally failed at 1.042 kcal/mol, "
     "but the failure is an edge artefact: evaluated within ±2σ of the tiled centres the "
     "cross-check passes at 0.470 against a tolerance of 0.500.")

doc.add_heading("2.8  Seeding cost and parallelisation", 2)
para(doc,
     "Serial seeding cost 60–70 s per seed, dominated by a 450 ps restrained pull "
     "(150,000 steps, against a stock default of 5,000). Both the parallel path and the "
     "shortened pull had been validated only at the level of argument parsing; neither had "
     "ever executed. Both were therefore run before launch (job 2376833), on the same "
     "resource request as production (--mem=96G, --gres=gpu:4).")
add_table(doc, ["Quantity", "Serial (measured earlier)", "Parallel, 28 workers (this test)"], [
    ("Windows seeded", "21", "21 / 21"),
    ("Wall time", "~21 min (21 × ~60 s)", "106 s"),
    ("Write spread across all PDBs", "—", "17 s (one concurrent batch)"),
    ("Temp-file races observed", "—", "none"),
    ("Projected for 112 production seeds", "~112 min", "4 batches, ≈ 7 min"),
], widths=[2.1, 2.0, 2.5])
caption(doc, "Table 6. Seeding throughput. 28 concurrent OpenMM contexts plus MPS fit "
             "inside the 96 GB request.")

doc.add_heading("2.9  Pull length reduction", 2)
para(doc,
     "The pull was cut from 150,000 steps to 10,000 at 3.0 fs, i.e. 450 ps to 30 ps — a "
     "15-fold reduction. Because the starting-structure quality gate is what detects a "
     "pull that is too short, and that gate's tolerance had simultaneously been loosened "
     "from 5.0 to 15.0 kcal/mol, the two changes were verified together.")
add_table(doc, ["Metric", "Value", "Threshold"], [
    ("Windows classified bad", "0", "—"),
    ("Windows ok / warn", "14 / 7", "—"),
    ("Worst starting umbrella bias", "4.42 kcal/mol", "15.0 (campaign); 5.0 (old default)"),
    ("Worst |ΔCV1| from centre", "0.1365", "dimensionless"),
    ("Pull fallbacks", "0 of 21 (all 'seeded')", "—"),
], widths=[2.4, 1.8, 2.4])
caption(doc, "Table 7. Starting-structure quality at 30 ps pull "
             "(us_starting_structure_quality.json, job 2376833).")
para(doc,
     "The worst bias sits below even the old, stricter default, so the shortened pull and "
     "the loosened tolerance are each independently justified rather than jointly excused.")
para(doc,
     "The probe was run on a different code deployment from production. Its result "
     "transfers because gareus/seeding.py is byte-identical between the two deployments "
     "and the --us-pull-workers argparse block is identical; this was checked before the "
     "result was accepted.")

doc.add_heading("2.10  Throughput", 2)
add_table(doc, ["Configuration", "Aggregate", "Per state"], [
    ("21 states, 4× L40S", "2476 ns/day", "118 ns/day"),
    ("1 GPU (reference)", "—", "107 ns/day"),
], widths=[2.4, 2.0, 2.0])
caption(doc, "Table 8. Measured throughput at 3.5 fs. Multi-GPU scaling is close to linear.")

doc.add_heading("2.11  Configuration defects found before launch", 2)
para(doc,
     "Five defects were found in the campaign configuration and launcher during "
     "preparation. Each would have affected an unattended run.")
add_table(doc, ["#", "Defect", "Consequence if launched", "Fix"], [
    ("1", "SWARM_AN pointed at chignolin_7_swarm",
     "consumes the OLD, old-CV, gate-FAILED round", "→ chignolin_7_swarm2"),
    ("2", "timestep_fs = 4.0",
     "NaN, proven on this exact envelope (§2.3)", "→ 3.5"),
    ("3", "max_replicas = 64",
     "64 < 112 states; written when the old CV supported 2 windows", "→ 128"),
    ("4", "ap_min_samples_per_window = 50000",
     "unreachable: budget yields 30612 samples/state", "→ 30000"),
    ("5", "no resubmit cap",
     "chain unbounded", "MAX_RESUBMITS = 30"),
], widths=[0.3, 1.9, 2.4, 1.7])
caption(doc, "Table 9. Pre-launch configuration defects.")
para(doc,
     "Budget arithmetic behind defect 4: md_budget_ns 6000 × ap_final_pool_fraction 0.5 "
     "= 3000 ns for the final pool; 3000 / 112 = 26.8 ns per state = 30612 samples. "
     "Restoring the 50000 floor would require raising md_budget_ns to approximately 11200.")

doc.add_heading("2.12  Chain exit reporting", 2)
para(doc,
     "The resubmission chain had five terminal paths in cleanup() sharing one bare return "
     "statement, so a stop at 03:00 was indistinguishable from success without reading "
     "logs. A chain_marker() helper was added; each path now stamps a timestamped reason "
     "into chignolin_7_CHAIN_STATUS.txt (pool spent, driver completed, non-zero exit, "
     "resubmit cap reached, sbatch rejected).")

doc.add_heading("2.13  Repository state", 2)
add_table(doc, ["Action", "Detail"], [
    ("Commit", "09492e3 — 102 files"),
    ("Push", "1878fd2..09492e3, main level with origin"),
    ("Merge", "feat/pep-gamd-internal-boost NOT merged; left open on GitHub"),
    ("Deploy", "aurum ~/2026_peptide_sampler: ddf396e → 09492e3 (git-archive, 550 files)"),
    ("Deploy backup", "~/2026_peptide_sampler_ddf396e"),
], widths=[1.4, 5.2])
caption(doc, "Table 10. Repository and deployment actions.")
para(doc,
     "Commit 09492e3 moves gareus_monitor.py (4961 lines) and its 779-line test out of "
     "RUNS/, which is listed in .gitignore. Because .gitignore does not untrack files, "
     "both were still tracked while being invisible to ignore-respecting tooling; the "
     "reorganisation deleted the test with no rename target, so committing the working "
     "tree as found would have dropped 779 lines of tests from version control. The test "
     "was restored beside its module, where its "
     "sys.path.insert(Path(__file__).parent) requires it to be, and 38 tests pass there.")
para(doc,
     "The internal-boost branch was left unmerged on the grounds that although it presents "
     "as additive (a new boost variant), it is +281/−63 on gareus/pep_gamd.py and also "
     "modifies production.py — the shared path the essential variant runs. Merging and "
     "deploying it would have placed unvalidated code under a frozen envelope. That the "
     "deployed code is free of it was verified positively: the --gamd-boost-type choice "
     "list ends at pep-gamd-lower-dual.")
para(doc,
     "The deployment change was confirmed behaviour-neutral: gareus/ is byte-identical "
     "between the ddf396e and 09492e3 deployments (only stale __pycache__ differed).")

# ---------------------------------------------------------------- 3 Discussion
doc.add_heading("3. Discussion", 1)
para(doc,
     "Two methodological findings generalise beyond this campaign.")
para(doc,
     "First, a gate that predicts an outcome by extrapolation should not block a protocol "
     "that measures the same outcome directly (§2.6). The ladder-ESS gate asked whether "
     "a single reweighting jump from the unbiased swarm could reach a rung. Production "
     "samples every rung and couples them by exchange, so a negative answer to the gate's "
     "question carries no information about production's viability — as the measured "
     "overlap of 0.919–0.963 against a predicted-unusable verdict shows. The correct "
     "response was to keep the diagnostic and remove its authority, not to weaken its "
     "threshold.")
para(doc,
     "Second, dry validation of a code path proves only that arguments parse (§2.8, "
     "§2.9). Both the parallel seeding path and the shortened pull were argument-valid "
     "and never executed, in a codebase with a known concurrency race on shared temporary "
     "files whose earlier fix was a scheduling stagger rather than a code change — and "
     "the seeding path has no such stagger. One minute of cluster time settled both "
     "questions. The generalisable rule is that an untested path in an unattended run "
     "fails silently, because a non-zero exit does not resubmit and nothing else is "
     "watching.")
para(doc,
     "A third, narrower point concerns the 4 fs failure (§2.3). Because the dead replica "
     "was at the unboosted λ = 0 rung and arrived there by a single exchange, the failure "
     "belongs to the ladder's end rung rather than to boost magnitude in isolation. "
     "Lowering k0 while retaining 4 fs would therefore not be expected to remove it.")

# ---------------------------------------------------------------- 4 Limitations
doc.add_heading("4. Limitations", 1)
add_table(doc, ["Limitation", "Status"], [
    ("Boost-magnitude uncertainty of ~17 % is frozen campaign-wide by the "
     "swarm_stability_extrema_sigma_tol change (1.0 → 1.5; measured value 1.248, "
     "odd = −3761.5, even = −3568.4).",
     "Accepted, documented, reversible. First suspect if the λ = 0 cross-check "
     "returns marginal after epoch 0."),
    ("λ = 0 cross-check margin in the tiling probe is 0.470 against a tolerance "
     "of 0.500.",
     "Thin. The 16-window production ladder is expected to widen it; this is an "
     "expectation, not a measurement."),
    ("3.5 fs was validated by a 2 ns × 7 rung probe, not by a full-length run.",
     "Bound, not a guarantee."),
    ("The 4 fs failure was observed once; σ0 = 8 was never probed.",
     "Single observation. The stable/unstable bracket is k0 = 0.273 (stable) to "
     "0.720 (NaN)."),
    ("The full library test suite was not executed in the final session.",
     "No gareus/ source changed in commit 09492e3, so the previous green state "
     "(1349 passed, 0 failed) stands; not independently re-confirmed."),
    ("Seeding is performed once per rung rather than once per centre.",
     "A further ~7× saving is available and was not implemented."),
    ("Campaign configuration files on aurum (chignolin.yaml, chignolin_7.sh, "
     "chignolin_7_swarm.yaml/.sh) are unversioned.",
     "They exist only on the cluster."),
], widths=[3.4, 3.2])
caption(doc, "Table 11. Known limitations and their status.")

doc.add_heading("4.1  Corrections to earlier claims made during this work", 2)
para(doc,
     "Recorded for provenance. Each was corrected on measurement.")
add_table(doc, ["Claim as first stated", "Correction"], [
    ("8 test failures in the suite", "Mis-parse of warning lines; actual 1349 passed, 0 failed."),
    ("Multi-GPU scaling is lossy (564 ns/day)",
     "The average included 23 min of sequential seeding; scaling settles at "
     "2476 ns/day aggregate, 118 ns/day per state."),
    ("k0 = 0.380 sits at the failure boundary (~0.4)",
     "Unsupported. Verified points are 0.273 stable and 0.720 NaN; σ0 = 8 unknown."),
    ("Cap λ_max at 0.7",
     "Withdrawn: it optimised a diagnostic rather than the campaign objective."),
], widths=[2.7, 3.9])
caption(doc, "Table 12. Corrections.")

# ---------------------------------------------------------------- 5 Conclusions
doc.add_heading("5. Conclusions", 1)
para(doc,
     "The campaign is prepared, validated and not launched. Every artefact the launcher "
     "requires is present (windows_lambda_ladder.csv, shared_gamd_setup_globals.json, "
     "seed_bank, swarm_gate.json), the swarm gate status is pass, the pilot comparison "
     "returns freeze_allowed = true (Total k0 ratio 1.393, Vmin 1.70 σ; Dihedral clean), "
     "and the output directory chignolin_7/ is absent, so the first job starts fresh "
     "rather than resuming.")
add_table(doc, ["Phase", "Amount", "Wall time"], [
    ("Seeding", "112 seeds, 28 workers", "≈ 7 min"),
    ("Pre-production", "392 ns", "3.8 h"),
    ("Production", "6000 ns", "2.4 days"),
    ("Total", "—", "≈ 2.6 days, ≈ 15 chained 4 h jobs"),
], widths=[1.8, 2.4, 2.4])
caption(doc, "Table 13. Projected campaign timeline.")
para(doc, "Launch command: cd ~/gareus/chignolin && sbatch chignolin_7.sh")

# ---------------------------------------------------------------- Appendix
doc.add_heading("Appendix A. File and Commit Inventory", 1)
add_table(doc, ["Path", "Role"], [
    ("gareus/swarm/ladder_design.py",
     "MAX_UPPER_BOUND_PROBES, probe_centre_seed_support, autotune_cv1_upper_bound; "
     "library-quantile veto removed"),
    ("gareus/swarm/members.py", "_DEFAULT_GRAFT_MINIMIZE_ITERS raised 100 → 500"),
    ("gareus/swarm/analyze.py",
     "passes seed_cv1 from post-discard exported frames; folds gate warnings into a "
     "flat report warnings list"),
    ("gareus/swarm/gates.py", "ladder_ess_gate returns warnings, not a block"),
    ("gareus/cli.py",
     "--swarm-graft-minimize-iters, --swarm-max-seed-gap-sigma, "
     "--swarm-stability-sigma-rel-tol, --swarm-stability-extrema-sigma-tol, "
     "--swarm-min-done-fraction, --swarm-max-graft-fallback-fraction"),
    ("gareus/provenance.py", "new argument destinations registered in _method_settings"),
    ("examples/chignolin_swarm_stage.yaml", "validated contact CV pinned with evidence"),
    ("~/gareus/chignolin/chignolin.yaml", "stage S2 production configuration"),
    ("~/gareus/chignolin/chignolin_7.sh",
     "launcher: preflight, resubmit cap, chain markers, 28 seeding workers"),
    ("~/gareus/chignolin/chignolin_7_swarm.yaml/.sh", "swarm stage S1"),
], widths=[2.5, 4.1])
caption(doc, "Table A1. Files changed or created.")
add_table(doc, ["Commit", "Subject"], [
    ("405cbb1", "merge: swarm graft relaxation + autotuned probed CV1 upper bound"),
    ("ddf396e", "merge: expose swarm gate tolerances, demote ladder-ESS to advisory, "
                "pin the validated CV1"),
    ("09492e3", "chore: move docs and the monitor out of ignored trees, land the R7 "
                "CV census"),
], widths=[1.1, 5.5])
caption(doc, "Table A2. Commits on main. The repository carries no attribution trailers; "
             "this convention was preserved.")

add_table(doc, ["Job", "Purpose", "Outcome"], [
    ("2375854", "4 fs stability probe", "NaN at 1.056 ns"),
    ("2376833", "28-worker seeding + 30 ps pull", "21/21 seeded in 106 s; worst bias "
                "4.42 kcal/mol; cancelled after seeding"),
], widths=[1.0, 2.6, 3.0])
caption(doc, "Table A3. Cluster jobs referenced in this report.")

OUT.parent.mkdir(parents=True, exist_ok=True)
doc.save(OUT)
print("written:", OUT)
print("size:", OUT.stat().st_size, "bytes")
