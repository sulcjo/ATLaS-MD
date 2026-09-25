"""Build the effective adaptive top-ups report (.docx).

    python docs/superpowers/specs/2026-09-24-effective-topups/build_report.py

House style from write_boltzmann_validation_report.py (Calibri 9.5, heading level 0 title,
numbered level-1 sections, add_table with 'Light List Accent 1' falling back to 'Table Grid',
monospace paragraphs for equations). Every number is read from a source file at build time and
checked against the value quoted in the SDD ruling ledger / CHANGELOG, or it is quoted from
those documents with the source named in the text. Figures go to FIGDIR.
"""
from __future__ import annotations

import csv
import gzip
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from docx import Document  # noqa: E402
from docx.enum.text import WD_BREAK  # noqa: E402
from docx.shared import Inches, Pt, RGBColor  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO))
from gareus.adaptive.throughput import node_ns_per_day, wall_hours  # noqa: E402
from gareus.adaptive.union_diagnostics import UNION_PEAK_BYTES_PER_CELL  # noqa: E402

STUDY = HERE / "synth_study"
DILUTION_CSV = Path("/home/sulcjo/.claude/jobs/afcc2200/tmp/dilution/rung_pair_table.csv")
FIGDIR = Path("/home/sulcjo/.claude/jobs/afcc2200/tmp/report_figs")
OUT = HERE / "effective_topups_report.docx"
BASE, HEAD = "4ea0814", "4cdc54d"

ACCENT = RGBColor(0x1F, 0x4E, 0x79)
BLUE, GREEN, VERM, PURPLE, ORANGE, SKY = ("#0072B2", "#009E73", "#D55E00", "#CC79A7",
                                          "#E69F00", "#56B4E9")
LANDSCAPES = ("rugged-2d", "gated-barrier", "slow-cv2-double-branch", "harmonic-bowl")
TABLE = ((16.0, 3154.0), (59.0, 2300.0))       # policy default, gareus/adaptive_production.py
N_GPUS, TIMESTEP_FS = 4, 4.0

_counts = {"tables": 0, "figures": 0}


# ── data ────────────────────────────────────────────────────────────────────

def load_data() -> dict:
    study = json.loads((STUDY / "topup_study.json").read_text())
    detail = json.load(gzip.open(STUDY / "topup_study_detail.json.gz"))
    sigma = json.loads((STUDY / "sigma_rule_study.json").read_text())["summary"]
    with DILUTION_CSV.open() as fh:
        pairs = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(fh)]
    commits = subprocess.run(["git", "log", "--reverse", "--format=%h|%ad|%s", "--date=short",
                              f"{BASE}..{HEAD}"], cwd=REPO, capture_output=True, text=True,
                             check=True).stdout.strip().splitlines()
    return {"study": study, "detail": detail, "sigma": sigma, "pairs": pairs,
            "commits": [c.split("|", 2) for c in commits]}


def check(cond: bool, what: str) -> None:
    if not cond:
        raise SystemExit(f"source check failed: {what}")


def dilution_stats(pairs) -> dict:
    full = [p["overlap_full_union_a"] for p in pairs]
    pw = [p["overlap_pairwise_b"] for p in pairs]
    ratio = [b / a for a, b in zip(full, pw)]
    s = {"n": len(pairs), "full_med": statistics.median(full), "pw_med": statistics.median(pw),
         "full_min": min(full), "full_max": max(full), "pw_min": min(pw), "pw_max": max(pw),
         "ratio_med": statistics.median(ratio), "full_lt15": sum(x < 0.15 for x in full),
         "pw_lt15": sum(x < 0.15 for x in pw), "full_lt25": sum(x < 0.25 for x in full),
         "pw_lt25": sum(x < 0.25 for x in pw)}
    # ledger ruling-25 check line
    check(s["n"] == 48, "48 rung pairs")
    check(abs(s["full_med"] - 0.0887) < 5e-4 and abs(s["pw_med"] - 0.2579) < 5e-4, "c7 medians")
    check(abs(s["ratio_med"] - 2.71) < 0.01, "c7 ratio median 2.71")
    check(s["full_lt15"] == 38 and s["pw_lt15"] == 0, "38/48 vs 0/48 below 0.15")
    check(round(s["full_min"], 3) == 0.065 and round(s["full_max"], 3) == 0.214, "full range")
    check(round(s["pw_min"], 3) == 0.197 and round(s["pw_max"], 3) == 0.428, "pairwise range")
    return s


def study_rows(study, detail) -> list:
    rows = []
    for ls in LANDSCAPES:
        s = study["landscapes"][ls]
        seeds = detail[ls]
        a = np.array([r["uniform"] for r in seeds]); b = np.array([r["topup"] for r in seeds])
        check(abs(np.median(a[:, 0]) - s["median_uniform"]["max_sigma"]) < 1e-12, f"{ls} A max σ")
        check(abs(np.median(b[:, 1]) - s["median_topup"]["pmf_rmse"]) < 1e-12, f"{ls} B RMSE")
        eps = [e for r in seeds for e in r["epochs"] if "plan" in e]
        planned = [e["plan"] for e in eps if e["plan"]["reason"] == "planned"]
        unconv = sum(e.get("sigma_baseline", {}).get("n_unconverged", 0) for e in eps)
        rows.append({"ls": ls, "s": s, "n_seed_epochs": len(eps), "n_planned": len(planned),
                     "patch_med": (int(np.median([p["patch"] for p in planned])) if planned else 0),
                     "n_unconverged": unconv, "a": a, "b": b})
    return rows


def gain_ratios(detail) -> dict:
    """Realised / predicted variance reduction per deficit state, (σb²-σr²)/(σb²-σp²)."""
    out = {}
    for ls in LANDSCAPES:
        vals = []
        for r in detail[ls]:
            for e in r["epochs"]:
                p = e.get("plan", {})
                if p.get("reason") != "planned" or "realised_sigma" not in p:
                    continue
                for k, sb in p["sigma_before"].items():
                    sp, sr = p["predicted_sigma"].get(k), p["realised_sigma"].get(k)
                    if sp is None or sr is None or sb <= sp:
                        continue
                    vals.append((sb ** 2 - sr ** 2) / (sb ** 2 - sp ** 2))
        out[ls] = np.array(vals)
    return out


# ── figures ─────────────────────────────────────────────────────────────────

def _save(fig, name) -> Path:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    p = FIGDIR / name
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
    return p


def fig_flow() -> Path:
    fig, ax = plt.subplots(figsize=(9.2, 6.0)); ax.set_axis_off()
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    boxes = {
        "phase": (0.3, 8.8, 3.6, 0.9, "Scheduled phase (epoch_NNN or final)\nfull_steps = quantised mean requested_steps"),
        "base": (0.3, 7.3, 3.6, 0.9, "Baseline, all active states, lockstep\nmax(1000, (1 − f)·full_steps) steps"),
        "diag": (0.3, 5.5, 3.6, 1.2, "Per-epoch union MBAR (all phases pooled)\nσ_k, split-halves flag, pairwise edge\noverlap, g_k  [memory guard first]"),
        "alloc": (0.3, 3.7, 3.6, 1.2, "plan_topup (pure function)\ndeficits, noise vs structural edges,\npartners, L = min(worst need, budget)"),
        "seed": (5.3, 3.7, 4.2, 1.2, "_seedable_patch: drop windows with no\nexported final State; corrupt index →\nseed_mismatch (top-up ends)"),
        "topup": (5.3, 1.9, 4.2, 1.2, "topup_001_L: patch only, lockstep,\neach window continued from its newest\nfinal_window_states export (no pull)"),
        "calib": (5.3, 0.2, 4.2, 1.2, "Post-top-up union solve → correction c,\nedge attempts, wall-time row\n(topup_state.json)"),
        "none": (0.3, 1.2, 3.6, 1.6, "No top-up: healthy / cap_too_small /\nno_diagnostics / no_seed_states\nNumbered epoch: withheld MD stays in pool\nFinal phase: baseline resumed to full_steps"),
        "struct": (5.3, 6.0, 4.2, 1.0, "Structural edges → bridge / add-state\nmachinery (never MD); logged + saved"),
    }
    for key, (x, y, w, h, text) in boxes.items():
        colour = {"none": "#f2e0d6", "struct": "#e6ddeb", "topup": "#d7ebf6", "seed": "#d7ebf6",
                  "calib": "#d7ebf6"}.get(key, "#e3eee8")
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=colour, ec="#333333", lw=0.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.4)

    def arrow(x0, y0, x1, y1, label=None):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", lw=0.9, color="#333333"))
        if label:
            ax.text((x0 + x1) / 2 + 0.08, (y0 + y1) / 2, label, fontsize=6.8, color="#333333")
    arrow(2.1, 8.8, 2.1, 8.2); arrow(2.1, 7.3, 2.1, 6.7); arrow(2.1, 5.5, 2.1, 4.9)
    arrow(3.9, 4.3, 5.3, 4.3, "planned"); arrow(2.1, 3.7, 2.1, 2.8, "other reason")
    arrow(7.4, 3.7, 7.4, 3.1); arrow(7.4, 1.9, 7.4, 1.4)
    arrow(3.9, 4.7, 5.3, 6.1, "structural edges")
    ax.set_title("One scheduled phase with --ap-topups (at most one top-up per phase)", fontsize=9)
    return _save(fig, "fig1_flow.png")


def fig_throughput() -> Path:
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.2))
    ctx = np.linspace(4, 70, 200)
    node = [node_ns_per_day(int(c * N_GPUS), N_GPUS, TABLE) for c in ctx]
    ax[0].plot(ctx, node, color=BLUE, label="node ns/day (model)")
    ax[0].plot([c for c, _ in TABLE], [v for _, v in TABLE], "o", color=VERM, label="measured (chignolin_8)")
    ax[0].set_xlabel("contexts per GPU"); ax[0].set_ylabel("aggregate ns/day/node")
    ax[0].set_title("(a) Throughput table, flat outside the points", fontsize=9)
    ax[0].legend(fontsize=7); ax[0].grid(alpha=0.3)
    n = np.arange(4, 237, 4)
    ax[1].plot(n, [wall_hours(100_000, int(k), TIMESTEP_FS, N_GPUS, TABLE) for k in n], color=BLUE)
    ax[1].set_xlabel("states in segment (4 GPUs)"); ax[1].set_ylabel("wall hours")
    ax[1].set_title("(b) Cost of 100,000 steps (0.4 ns/state at 4 fs)", fontsize=9)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, "fig2_throughput.png")


def fig_regime(study) -> Path:
    cal = study["meta"]["regime_calibration_arm_A_only"]
    fig, axes = plt.subplots(1, 4, figsize=(10.0, 2.8), sharey=True)
    for ax, ls in zip(axes, LANDSCAPES):
        tr = sorted(cal[ls], key=lambda t: t["hours"])
        h = [t["hours"] for t in tr]
        ax.axhspan(0.10, 0.40, color=ORANGE, alpha=0.18, lw=0)
        ax.plot(h, [t["deficit_fraction"] for t in tr], "o-", ms=3, color=VERM, label="deficit fraction")
        ax.set_title(ls, fontsize=8.5); ax.set_xlabel("wall hours (arm A)", fontsize=8); ax.grid(alpha=0.3)
        ax2 = ax.twinx()
        ax2.plot(h, [t["median_rows"] for t in tr], "s--", ms=3, color=BLUE, label="median rows/state")
        ax2.axhline(40, color=BLUE, lw=0.8, ls=":"); ax2.set_ylim(0, 70)
        if ls == LANDSCAPES[-1]:
            ax2.set_ylabel("median decorrelated rows/state", color=BLUE, fontsize=8)
        ax.axvline(study["meta"]["rows_floor_hours"][ls], color="#555555", lw=0.8)
    axes[0].set_ylabel("fraction of states above σ*", color=VERM, fontsize=8)
    axes[0].set_ylim(0, 0.45)
    fig.tight_layout()
    return _save(fig, "fig3_regime.png")


def fig_ab(rows) -> Path:
    fig, axes = plt.subplots(2, 4, figsize=(10.0, 5.0))
    for j, r in enumerate(rows):
        for i, (col, lab) in enumerate(((0, "max σ_k (kcal/mol)"), (1, "PMF RMSE (kBT)"))):
            ax = axes[i, j]
            x, y = r["a"][:, col], r["b"][:, col]
            lo, hi = min(x.min(), y.min()), max(x.max(), y.max()); pad = 0.04 * (hi - lo + 1e-12)
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#777777", lw=0.8)
            ax.plot([lo - pad, hi + pad], [0.99 * (lo - pad), 0.99 * (hi + pad)], color="#777777", lw=0.6, ls=":")
            ax.plot(x, y, "o", ms=3.5, color=BLUE if i == 0 else GREEN)
            ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad)
            ax.tick_params(labelsize=6.5); ax.grid(alpha=0.3)
            if i == 0:
                ax.set_title(r["ls"], fontsize=8.5)
            ax.set_xlabel(f"arm A (uniform) {lab}", fontsize=7)
            if j == 0:
                ax.set_ylabel(f"arm B (top-ups) {lab}", fontsize=7)
    fig.suptitle("Paired seeds at the rows-floor budget; below the diagonal = top-ups better "
                 "(dotted: 1 % win margin)", fontsize=8.5)
    fig.tight_layout()
    return _save(fig, "fig4_ab_paired.png")


def fig_sigma_rules(sigma) -> Path:
    rules = ("all_edges_min", "spatial_min", "spatial_max", "reference")
    groups = ("hours=0.3", "hours=5.0", "pooled")
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    w = 0.2
    for i, rule in enumerate(rules):
        vals = [sigma[g][rule]["rho"] for g in groups]
        ax.bar(np.arange(3) + (i - 1.5) * w, vals, w, color=(SKY, GREEN, BLUE, ORANGE)[i], label=rule)
    ax.set_xticks(np.arange(3)); ax.set_xticklabels(["0.3 h budget", "5.0 h budget", "pooled (both budgets)"])
    ax.set_ylabel("Spearman ρ vs true local Δf error"); ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=7, ncol=2); ax.set_ylim(0, 0.9)
    return _save(fig, "fig5_sigma_rules.png")


def fig_dilution(pairs) -> Path:
    fig, ax = plt.subplots(figsize=(8.0, 3.4))
    kinds = sorted({(p["lambda_i"], p["lambda_j"]) for p in pairs})
    cols = (BLUE, GREEN, PURPLE)
    for (li, lj), c in zip(kinds, cols):
        sel = [p for p in pairs if (p["lambda_i"], p["lambda_j"]) == (li, lj)]
        x = [p["primary_center"] for p in sel]
        ax.plot(x, [p["overlap_pairwise_b"] for p in sel], "o-", ms=3.5, lw=0.8, color=c,
                label=f"pairwise, λ {li:.2f}→{lj:.2f}")
        ax.plot(x, [p["overlap_full_union_a"] for p in sel], "x--", ms=4, lw=0.7, color=c,
                label=f"full union, λ {li:.2f}→{lj:.2f}")
    ax.axhline(0.15, color=VERM, lw=1.2, label="0.15 min_rung_overlap")
    ax.axhline(0.25, color=VERM, lw=1.0, ls="--", label="0.25 target_rung_overlap")
    ax.set_xlabel("CV1 centre of the rung pair"); ax.set_ylabel("sqrt(O_ij·O_ji)")
    ax.set_ylim(0, 0.5); ax.grid(alpha=0.3); ax.legend(fontsize=6.3, ncol=2, loc="upper left")
    return _save(fig, "fig6_c7_dilution.png")


def fig_gain(gr) -> Path:
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    data = [gr[ls] for ls in LANDSCAPES]
    ax.boxplot(data, showfliers=False)
    ax.set_xticks(range(1, 5)); ax.set_xticklabels([f"{ls}\n(n={len(v)})" for ls, v in zip(LANDSCAPES, data)], fontsize=7)
    ax.axhline(1.0, color="#555555", lw=0.8)
    ax.set_ylabel("realised / predicted\nvariance reduction", fontsize=8); ax.grid(alpha=0.3, axis="y")
    return _save(fig, "fig7_gain_model.png")


# ── document helpers ────────────────────────────────────────────────────────

def add_table(doc, headers, rows, widths=None, font=8.0):
    t = doc.add_table(rows=1, cols=len(headers))
    try:
        t.style = "Light List Accent 1"
    except KeyError:
        t.style = "Table Grid"
    for i, h in enumerate(headers):
        c = t.rows[0].cells[i]; c.text = ""
        r = c.paragraphs[0].add_run(h); r.bold = True; r.font.size = Pt(font)
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            r = cells[i].paragraphs[0].add_run(str(v)); r.font.size = Pt(font)
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Inches(w)
    _counts["tables"] += 1
    return t


def para(doc, text, size=9.5, italic=False, bold=False, space=4):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(space)
    r = p.add_run(text); r.font.size = Pt(size); r.italic = italic; r.bold = bold
    return p


def bullets(doc, items, size=9.5):
    for it in items:
        p = doc.add_paragraph(style="List Bullet")
        p.paragraph_format.space_after = Pt(2)
        r = p.add_run(it); r.font.size = Pt(size)


def code(doc, text, size=8.0):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(6)
    r = p.add_run(text); r.font.name = "Consolas"; r.font.size = Pt(size)
    return p


def caption(doc, text):
    return para(doc, text, size=8.0, italic=True, space=8)


def figure(doc, path, text, width=6.4):
    doc.add_picture(str(path), width=Inches(width))
    _counts["figures"] += 1
    caption(doc, text)


def f3(x):
    return f"{x:.3f}"


def pct(x):
    return f"{100 * x:+.1f} %"


# ── sections ────────────────────────────────────────────────────────────────

def title_and_summary(doc, D, rows, dil):
    doc.add_heading("Effective adaptive top-ups in ATLaS-MD: design as built, validation, and "
                    "the pairwise-overlap finding", 0)
    para(doc, f"Engineering report · 2026-09-25 · repository 2026_peptide_sampler (package gareus), "
              f"branch feat/effective-topups, {BASE}..{HEAD} ({len(D['commits'])} commits) · "
              "audience: PI and future developers", size=9, italic=True)
    para(doc, "Sources: design spec docs/superpowers/specs/2026-09-24-effective-topups-design.md; "
              "plan rev 2 docs/superpowers/plans/2026-09-24-effective-topups.md; SDD ruling ledger, "
              "small-board judgments and synthetic study outputs in "
              "docs/superpowers/specs/2026-09-24-effective-topups/; CHANGELOG [Unreleased]; the code at "
              f"{HEAD}. This document and its generator (build_report.py, same directory) read the study "
              "JSON files and the chignolin_7 overlap table directly and check every quoted number "
              "against the ledger before writing.", size=8.5, italic=True)
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

    doc.add_heading("Executive summary", 1)
    gb = next(r for r in rows if r["ls"] == "gated-barrier")["s"]
    sc = next(r for r in rows if r["ls"] == "slow-cv2-double-branch")["s"]
    bullets(doc, [
        "What was built. An optional top-up mechanism for scheduled adaptive-production phases. After each "
        "phase's all-state baseline, a union-MBAR solve over all data pooled so far measures, per state, a "
        "local free-energy uncertainty σ_k and a split-halves drift flag, and, per layout edge, a pairwise "
        "energy-space overlap. A pure allocator picks the deficient states plus minimal exchange partners and "
        "one lockstep length L, costed in wall-hours from a node-throughput table. The phase then runs at most "
        "one top-up segment. That segment continues every window from its own exported final State; there is "
        "no pull and no re-equilibration.",
        "Top-ups are off by default (--ap-topups defaulted to True before this branch). chignolin_9 runs "
        "without them.",
        "Results-changing for top-ups-OFF campaigns, once this branch is merged and deployed (it is neither "
        "yet): (1) each scheduled phase gives every active state a uniform share of its budget. The per-state "
        "score allocator, with its low-sample, weak-edge, frontier and high-boost bonuses and the 2x step "
        "multiplier for new states, is removed. (2) An unmeasured edge is never counted as weak in the gate, "
        "the proposer or the reports. (3) The campaign-end rung overlap and ladder health are computed "
        "pairwise (item 3 is already deployed; see next bullet). No final_window_states/ export is written "
        "with top-ups off. For chignolin_9, the final review judged a redeploy to keep the resumed final "
        "baseline length and the subsampling identical, and to change the gate in logging only.",
        f"Pairwise-overlap finding. The full-union MBAR overlap matrix dilutes each pair's overlap by the "
        f"number of other states sharing its region. On chignolin_7's 64-state union, the {dil['n']} "
        f"adjacent-rung pairs read {dil['full_med']:.3f} median as full-union entries and "
        f"{dil['pw_med']:.3f} median pairwise (ratio {dil['ratio_med']:.2f}). "
        f"{dil['full_lt15']}/{dil['n']} full-union entries fall below the 0.15 floor; "
        f"{dil['pw_lt15']}/{dil['n']} pairwise entries do. The campaign-end rung gate and "
        "analyze_gareus_mbar.py's ladder health now use the pairwise statistic. This fix alone was merged to "
        "main as PR #103 (merge 90b7eba) and deployed to aurum2 as dda8caa. The 0.15/0.25 thresholds have "
        "not been re-measured on the pairwise scale.",
        "Synthetic verdict (negative). Four analytic landscapes, a 112-state ladder, 20 paired seeds at equal "
        "modelled wall-hours; the real allocator and union solve were used. No landscape has a budget at which "
        "10-40 % of states are above target while the median state holds at least 40 decorrelated rows. Spec "
        "6.2's heterogeneous criterion is therefore untestable in this harness. For information only, at the "
        f"rows floor: top-ups lower the worst per-state σ on gated-barrier ({pct(gb['rel_diff']['max_sigma'])}, "
        f"{gb['wins_max_sigma']}/20 seeds) and slow-cv2-double-branch ({pct(sc['rel_diff']['max_sigma'])}, "
        f"{sc['wins_max_sigma']}/20). They improve PMF RMSE on no landscape; gated-barrier is "
        f"{pct(gb['rel_diff']['pmf_rmse'])}. Missing-bridge routing passes 20/20. Top-ups are not shown to help.",
        "Recommendation for chignolin_10. Run it as the real-MD A/B test (user decision 2026-09-24), with "
        "top-ups on against a uniform-extension control at equal wall-hours and equal contexts/GPU, with "
        "paired seeds and replicates. Before launch: (a) settle the union-diagnostics memory (todo T6). The "
        "8 GB guard admits about 550,000 kept rows at 236 states and chignolin_9 holds about 686,000 raw rows "
        "campaign-wide; whether a c9-sized campaign trips it depends on the subsampling ratio. The final "
        "re-review judged the default sensible, but a tripped guard would end phases as no_diagnostics and "
        "make the A/B compare uniform extension with itself; (b) re-measure the rung thresholds on the pairwise scale (T5); (c) run the "
        "real-file seeding spike (T3). Do not enable top-ups on any production campaign before that test "
        "reports.",
    ], size=9)


def background(doc, D):
    doc.add_heading("1  Background: why the old top-ups were ineffective", 1)
    para(doc, "Top-ups are extra MD on a subset of states inside a scheduled epoch or the final phase. The design "
              "spec measured the pre-branch mechanism on chignolin_9 epoch_001 and found four defects:")
    add_table(doc, ["#", "Defect (measured on chignolin_9 epoch_001)", "Where"], [
        ("D1", "An edge with no measurement counted as weak (weak = overlap is None or overlap < target). All 177 "
               "rung edges (no energy-space overlap before the campaign-end solve) counted weak, while the 84 "
               "spatial edges were healthy (median CV overlap 0.887, none < 0.1). The gate reported 184 weak edges.",
         "_weak_edge_touch_counts, convergence gate"),
        ("D2", "Every state started at score ≥ 1 and the whole remaining budget was distributed by score, so all 236 "
               "states received extra steps (3.0x the baseline in total).", "build_adaptive_epoch_schedule"),
        ("D3", "States were grouped into segments by identical extra-step size. On the ladder this put one rung "
               "per segment (topup_001 = 59 states at λ 1.0, topup_002 = 48 at λ 0.0), so no λ exchange could "
               "happen inside a top-up.", "run_scheduled_adaptive_epoch"),
        ("D4", "Each top-up re-seeded from the conformer bank and re-pulled its windows (about 25+ min for 59 "
               "windows) instead of continuing from the baseline's final replica states, which added an "
               "equilibration transient.", "top-up seeding in production.py"),
    ], widths=[0.4, 4.6, 1.7])
    caption(doc, "Table 1. Defects of the pre-branch top-ups (spec §2).")
    doc.add_heading("1.1  Lockstep segments and the throughput inversion", 2)
    para(doc, "A segment advances all its replicas by the same number of steps, so per-state targets inside a "
              "segment are impossible. A top-up of n states and length L costs its wall time, L/T(n). Node "
              "throughput depends on occupancy. On chignolin_8's node (4 GPUs, production integrator, MPS), "
              "16 contexts/GPU gave 3,154 ns/day/node and 59 contexts/GPU about 2,300 ns/day/node, with the CPU "
              "co-limiting. Per context that is about 5x faster at low occupancy; in aggregate about 1.37x. "
              "Occupancies below 16/GPU are unmeasured. The allocator uses these two points as its throughput "
              "table and interpolates linearly between them, with no extrapolated gain outside them "
              "(gareus/adaptive/throughput.py).")
    rows = []
    for c, v in TABLE:
        n = int(c * N_GPUS)
        rows.append((f"{int(c)}", f"{n}", f"{v:,.0f}", f"{v / n:.1f}"))
    add_table(doc, ["contexts/GPU", "states (4 GPUs)", "node ns/day", "ns/day per context"], rows,
              widths=[1.2, 1.3, 1.3, 1.6])
    caption(doc, "Table 2. The throughput table as shipped (--ap-topup-throughput-table default \"16:3154,59:2300\"). "
                 "The per-context column divides node throughput by 4 GPUs × contexts/GPU.")
    rows = []
    for n in (16, 59, 112, 236):
        rows.append((n, f"{n / N_GPUS:.1f}", f"{node_ns_per_day(n, N_GPUS, TABLE):,.0f}",
                     f"{wall_hours(100_000, n, TIMESTEP_FS, N_GPUS, TABLE):.2f}"))
    add_table(doc, ["states in segment", "contexts/GPU", "node ns/day (model)", "wall h for 100,000 steps"],
              rows, widths=[1.3, 1.1, 1.5, 1.8])
    caption(doc, "Table 3. Wall-hour cost of a 100,000-step lockstep segment (0.4 ns/state at 4 fs), computed with "
                 "gareus.adaptive.throughput.wall_hours on 4 GPUs. Below 16 contexts/GPU the model is flat, so a "
                 "small patch gets no modelled speed-up per context beyond the 16/GPU point.")
    figure(doc, fig_throughput(), "Figure 1. (a) The two measured throughput points and the model curve. (b) Wall "
                                  "hours for a fixed 100,000-step segment against patch size.")


def design(doc):
    doc.add_heading("2  Design as built", 1)
    para(doc, "This section describes the code at the branch head, not the first plan. Where the implementation "
              "departs from the spec, the departure and the ruling behind it are named.")
    figure(doc, fig_flow(), "Figure 2. Control flow of one scheduled phase with top-ups on. Blue: top-up path; green: "
                            "every phase; orange: no top-up this phase; purple: structural routing.", width=6.3)

    doc.add_heading("2.1  Diagnostics: per-epoch union MBAR", 2)
    para(doc, "After the baseline, _phase_union_diagnostics builds the union inputs over every sample the campaign "
              "holds (build_union_state_mbar_inputs). It uses each segment's native window centres, the per-state "
              "equilibration cut and the autocorrelation subsampling, and the same tICA-regime and pilot-directory "
              "filters as the campaign-end union. It solves MBAR (pymbar, initialize=\"zeros\", solver_protocol="
              "\"robust\"; warm start from the previous phase's f_k, filling missing entries). It does not use BAR "
              "initialisation, because the union state order (centres × rungs) is not overlap order. It returns "
              "UnionDiagnostics: n_k, σ_k, the unconverged set, g_k, the edge overlaps, f_k and sigma_argmax.")
    para(doc, "Pairwise edge overlap (ruling 22; gareus.mbar_analysis.ladder.pairwise_state_overlap). Only the "
              "pair's own samples enter, with a two-state mixture denominator and the union f_k held fixed:", space=2)
    code(doc, "W_nk  = exp(f_k − u_k(x_n)) / Σ_{l∈{i,j}} N_l exp(f_l − u_l(x_n)),   n ∈ samples of i or j,  k ∈ {i,j}\n"
              "O_ij  = N_j Σ_n W_ni W_nj,      reported edge overlap = sqrt(O_ij · O_ji)")
    para(doc, "Two identical states give 0.5 regardless of how many other states are in the union. The full-union "
              "matrix gives 1/K for K identical states. The value is exact only to the extent the passed f_k is "
              "converged. The scale matters for the thresholds: the maximum is 0.5, not 1.")
    para(doc, "Local uncertainty σ_k (ruling 23, then ruling 33/I4). From the MBAR pairwise uncertainty matrix "
              "δΔf (in kT), with S(k) the same-rung spatial layout neighbours of k:", space=2)
    code(doc, "σ_k = kT · max_{j ∈ S'(k)} δΔf_{jk}\n"
              "S'(k) = S(k) minus neighbours whose edge has used up its top-up attempts (edge_attempts ≥ max);\n"
              "        if empty, the same-centre other-rung partners of k\n"
              "fallback (no finite entry over S'(k)): min over the finite off-diagonal row δΔf_{k·};  unsampled k: NaN")
    para(doc, "The first cut used the minimum over all edge neighbours. A near-identical rung twin always won that "
              "minimum (median 0.007 kcal/mol against a 0.10 target), so no state ever looked deficient. The "
              "maximum over same-rung spatial neighbours was chosen by measurement (§4.3). Excluding tried-out "
              "edges (ruling 33, I4) stops both endpoints of a real gap from being topped up in every phase after "
              "the edge has been handed to the bridge machinery. The neighbour achieving the maximum is recorded as "
              "sigma_argmax and forced into the patch (I5).")
    para(doc, "Split-halves drift test (ruling 6). The spec's f_k-from-each-half check was replaced by a per-state "
              "local test. The review found the edge-based version flagged mostly healthy states (a 1σ drift in one "
              "of 9 states flagged 5) and leaked through the global solve.", space=2)
    code(doc, "own_k = u_k(x_n) for n in state k's own rows (chronological); halves a, b; each ≥ 20 rows (MIN_HALF)\n"
              "d = |mean(a) − mean(b)|,  se = sqrt(s_a²/n_a + s_b²/n_b)\n"
              "flag k  if  d > max(z*·se, min_effect/kT),  z* = Φ⁻¹(1 − α/(2m)),  α = 0.05,  m = number of states tested")
    para(doc, "The test cannot leak into another state's number, and it drops the two half-MBAR solves. The accepted "
              "cost: it is blind to a drift that changes f_k without changing the state's own-window bias "
              "distribution. min_effect is --ap-topup-min-effect (0.05 kcal/mol).")
    para(doc, "Statistical inefficiency g_k (plan rev 2, V3). The union NPZ is already decorrelated, so a Geyer pass "
              "over kept samples returns about 1. g_k is instead read from the builder's subsampling meta "
              "(subsample_counts_per_state: g, else (raw − t0)/kept, floored at 1). With N_eff = kept, the forward "
              "gain of L steps is L/(report_interval · g).")

    doc.add_heading("2.2  Allocator: plan_topup (pure function)", 2)
    bullets(doc, [
        "Deficits: sampled states (n_k > 0, finite σ) with σ_k > σ* (--ap-topup-target-sigma, 0.10 kcal/mol), or "
        "flagged by split-halves. Healthy states get zero; there is no default score and no distribution of "
        "left-over budget.",
        "Edge routing: each measured edge is compared with its threshold. Rung edges (same centre, other rung) are "
        "compared with the campaign's min_rung_overlap (0.15; ruling 11), spatial edges with "
        "--ap-topup-weak-overlap (0.15). A weak edge with an endpoint in the frozen initial deficit set (ruling 9) "
        "and fewer than --ap-topup-max-edge-attempts (2) previous top-ups is a noise edge, and both endpoints join "
        "the deficits. Any other weak edge is structural: it is left to the bridge/add-state machinery, logged "
        "(M2) and saved in topup_plan.json. An unmeasured edge is never weak.",
        "Partners: for each deficit, if no same-rung neighbour is already in the patch, add the highest-σ sampled "
        "same-rung neighbour (falling back to a non-deficit rung partner). If it has rung partners and none is in "
        "the patch, add the highest-σ rung partner. Then add every deficit's sigma_argmax neighbour. Never-sampled "
        "states are never partners.",
        "Gain model: scale_s = N_eff,s · interval · g_s / c_s (the steps-worth of data already held); "
        "σ_s(L) = σ_s · sqrt(scale_s/(scale_s + L)). This is a 1/sqrt(N_eff) surrogate, not the MBAR covariance.",
        "Required length per deficit: target σ* if σ_s > σ*, else σ_s/√2 (noise-edge endpoints and unconverged "
        "states double their data); need = ⌈N_eff((σ_s/target)² − 1)·interval·g_s/c_s⌉. The need is capped at 4x "
        "the state's current effective steps (MAX_STEP_MULTIPLE), so a severe deficit can remain above target "
        "after its one top-up.",
        "Length (ruling 8): L = min(max over deficits of required, budget_max), rounded to the report interval. "
        "budget_max is the largest L whose cost for the whole patch fits the budget. The budget is "
        "--ap-topup-max-fraction (0.3) × wall_hours(full_steps, all active states), taken from the un-shortened "
        "default (plan V4, ruling 33/I2). If L < one interval, the reason is cap_too_small; this is a budget-size "
        "trigger, not a small-deficit one. The first cut maximised benefit per cost over candidate lengths. That "
        "objective is concave and always picked the shortest candidate (measured: a σ = 0.30 state got 2,500 "
        "steps against a need of 400,000).",
        "Calibration (topup_state.update_after_topup): ratio = (σ_b² − σ_real²)/(σ_b² − σ_pred²); "
        "c ← clamp(0.7·c + 0.3·ratio, [0.1, 2.0]); if ratio < 0.5 then c ← max(0.1, 0.5·c). Each topped noise edge "
        "increments its attempt count.",
    ], size=9)

    doc.add_heading("2.3  Driver: one top-up per phase", 2)
    para(doc, "run_scheduled_adaptive_epoch computes full_steps as the quantised mean of the schedule's "
              "requested_steps (_schedule_full_steps). With top-ups off, that is the baseline length. With top-ups "
              "on, every active state's baseline is max(1000, quantised((1 − f)·full_steps)), so a healthy "
              "non-patch state gets less baseline MD than with top-ups off. The withheld fraction funds the one "
              "top-up. A resumed phase whose schedule was written by the removed score allocator stored each "
              "state's minimum in baseline_steps; reading requested_steps avoids a tiny baseline and budget (I2). "
              "_run_phase_topup plans, filters seeds and runs topup_001_<L> through the same run_segment used for "
              "baselines. If the MD pool skips the segment, the reason is pool_exhausted and no calibration is done. "
              "If the pool clips it, calibration uses predictions re-derived at the steps actually run (ruling 20).")
    add_table(doc, ["File", "Scope", "Contents"], [
        ("<phase>/topup_plan.json", "phase", "the persisted decision; once reason ≠ planned the phase never gets a "
                                             "second top-up; a resumed run reuses a planned plan"),
        ("<phase>/topup_union_overlap.json", "phase", "latest measured edge overlaps, reloaded on resume so the "
                                                      "gate and proposer see the same numbers"),
        ("<adaptive_dir>/topup_state.json", "campaign", "correction factors, edge attempts, warm-start f_k, "
                                                        "wall-time log (predicted_h, realised_h)"),
        ("<segment>/final_window_states/", "segment", "per-window OpenMM State XML + index.json (top-ups on only)"),
    ], widths=[2.0, 0.8, 3.9])
    caption(doc, "Table 4. Persistence. All JSON writes are atomic (tmp + replace); integer state-id keys and NaN "
                 "survive the round trip.")
    add_table(doc, ["reason", "Meaning", "Final-phase baseline extended?"], [
        ("healthy", "no deficit", "yes"),
        ("planned → completed", "top-up ran", "no"),
        ("pool_exhausted", "MD pool skipped the segment", "no (nothing left to extend with)"),
        ("cap_too_small", "budget affords < 1 report interval", "yes"),
        ("no_diagnostics", "union solve failed or memory guard tripped", "yes"),
        ("no_seed_states", "no deficit has an exported final State", "yes"),
        ("seed_mismatch", "restraint/CV mismatch at runtime, corrupt index.json, or no epoch_window_map.csv", "yes"),
        ("layout_changed", "a plan state left the active set after the segment already held data; partial "
                           "samples kept, no second directory", "yes"),
    ], widths=[1.5, 3.6, 1.6])
    caption(doc, "Table 5. Plan reasons. In a numbered epoch unspent MD stays in the campaign pool. In the final "
                 "phase, _final_phase_strands_topup_budget resumes the baseline from its own checkpoint to "
                 "full_steps (run_segment(..., force_resume=True)). Under force_resume only the delta is clipped "
                 "against the pool (ruling 34); ordinary resumes are unchanged.")

    doc.add_heading("2.4  Seeding: continuation, not re-pull", 2)
    para(doc, "With top-ups on, every adaptive-production segment (baseline or top-up) writes final_window_states/ on "
              "completion (should_export_final_window_states). A top-ups-off campaign, a pilot and a standalone "
              "run write nothing (I1). The export holds one portable OpenMM State per window "
              "(positions + velocities) and index.json, which records window, CV1, CV2, primary/secondary centre "
              "and k, and export_seq = time.time_ns(). A top-up loads, per window, the newest export across the "
              "phase's ancestor chain, ordered by export_seq; mtime is used only as a fallback, because copies "
              "scramble it (ruling 16). Before a seed is used, the recorded restraint must match the top-up's own "
              "window table (relative tolerance 1e-6; ruling 15) and the recomputed CVs must match the recorded "
              "ones (tolerance 1e-3; CV1 must be finite). Failure handling (ruling 18): a MISSING seed drops that "
              "window before launch. A runtime MISMATCH, or an unreadable index.json in any parent (ruling 19), "
              "ends the whole top-up; the campaign continues.")
    para(doc, "Departures from the spec and the board's condition 3. The spec asked for binary checkpoints, a check "
              "that the seeded frame's reduced potential equals the last logged u_k, and a burn-in fallback of "
              "5·τ_k. As built, seeds are portable State XML, the check is on restraint parameters and CVs, and "
              "there is no burn-in path. The index records no λ, so two states at one centre on different rungs "
              "have identical restraints and CVs, and the check cannot tell them apart. A window-to-state mis-map "
              "onto a rung twin would pass. The real-file spike (todo T3) is still open.")

    doc.add_heading("2.5  Memory guard and the unmeasured-never-weak invariant", 2)
    para(doc, "The union solve runs on the driver node, and an out-of-memory kill during it cannot be caught. "
              "_topup_union_size_guard runs inside the builder, after subsampling and before any rows × states "
              f"matrix is allocated. It estimates the peak as kept rows × states × 8 B × 7.7 "
              f"(UNION_PEAK_BYTES_PER_CELL = {UNION_PEAK_BYTES_PER_CELL:.1f} B). Above --ap-topup-diagnostics-max-gb "
              "(8.0) it logs a WARNING and the phase gets no_diagnostics. The raw sample dicts read before "
              "subsampling are not guarded.")
    para(doc, "_edge_is_measured_weak is the one predicate every weak-edge reader now routes through: the per-epoch "
              "gate, evaluate_adaptive_quality_gate, propose_actions_from_diagnostics, write_epoch_action_report and "
              "_adaptive_production_converged (rulings 12-13). Rung edges are judged on pairwise mbar_overlap "
              "against min_rung_overlap. Accepted cost: an edge that is unmeasured because an endpoint is unsampled "
              "no longer triggers a bridge or extension by itself; the min-samples checks still own coverage.")


NR = "not recorded in ledger"
RULINGS = [
    (1, "Execute Task 9 (seeding) before Task 8 (driver).", "T8's tests import T9's module; T9 is standalone.", "none"),
    (2, "All agents run tests via opencode run, logging to an untracked atlas_task<N>.log.", "pytest is hook-blocked in Bash.", "an agent wastes turns on a blocked command"),
    (3, "No aurum2 deploy or env change inside the plan; jax removal done separately at the c9 epoch_002→final boundary.", "user decision 2026-09-25", "none for the branch"),
    (4, "warn_if_pymbar_unusable fires once for every adaptive-production entry path.", "spec goal \"loud at startup\"; the plain AP path (chignolin_9) bypassed it.", "one redundant warning line"),
    (5, "Move restraint_sigma into gareus/math_helpers.py (re-export in dashboard); round λ with _CENTRE_DECIMALS in other_rung_same_centre.", "core allocator must not import the dashboard (circular cold import reproduced); one rung identity per module.", "one extra re-export line"),
    (6, "Replace edge-based split-halves with a per-state local Welch test, Bonferroni over tested states, effect floor in kT.", "edge version flagged mostly healthy states and leaked via the global solve; drops two half-MBAR solves.", "less sensitive to a drift that changes f_k without changing the own-window bias distribution"),
    (7, "WARNING on every None return; partial warm start; _local_sigma fallback = min over row; drop unused n_k; exc_info on catch-all; five named tests.", "review findings on Task 5", "none"),
    (8, "L = min(max required over deficits, budget_max), rounded; < interval → cap_too_small; benefit/score loop removed.", "concave score always chose the shortest L (σ 0.30 got 2,500 of 400,000 steps); one top-up per phase.", "a patch with one slow state runs longer (bounded by 4x cap and 0.3 budget)"),
    (9, "Classify every weak edge against the frozen initial deficit set, then add noise-edge endpoints.", "classification was order-dependent.", "none"),
    (10, "Round-1 minors: 3-rung layout test, exact predicted-σ assertion, 2.0 clamp exercised, _round_up tolerance, σ/√2 comment.", "review minors", NR),
    (11, "Rung edges in the allocator use policy.min_rung_overlap, not topup_weak_overlap.", "one rung-weakness threshold per campaign", "none (both default 0.15)"),
    (12, "Route the non-rung weak test of four more readers through _edge_is_measured_weak.", "spec invariant: an unmeasured edge is never weak; reports must agree with the gate.", "an edge unmeasured because an endpoint is unsampled no longer triggers a bridge/extension by itself"),
    (13, "Route rung edges in write_epoch_action_report and _adaptive_production_converged through the same predicate.", "\"rung always weak\" was the same bug; converged could never be true on a ladder run.", "report/summary flags change on ladder runs only, no MD decision"),
    (14, "Per-state XML written tmp+replace; unreadable/incomplete entries treated as missing (WARNING).", "seeding never stops the campaign", "none"),
    (15, "index.json records each window's restraint; seeding asserts it matches the top-up window (rel 1e-6); no epoch_window_map.csv fails closed.", "closes the chignolin_6 window-map mis-map class", "an old export without restraint fields cannot seed (none exist)"),
    (16, "Parent ordering by export_seq = time.time_ns() in index.json; mtime only as fallback.", "copies do not preserve mtime; a stale parent can replay a trajectory", "clock skew between nodes (<< the minutes-hours between exports)"),
    (17, "Minors: errors name window + state_id; seeded windows skip velocity randomisation; pool shutdown + lock release on mismatch; helpers moved; dead campaign-search helper deleted.", "review minors on Task 9", NR),
    (18, "Missing seed → window dropped pre-launch (_seedable_patch); runtime mismatch → whole top-up ends, campaign continues.", "dropping a replica after contexts are built needs a rebuild for a should-never-happen event", "one lost top-up in a rare event"),
    (19, "An unreadable index.json in any candidate parent raises SeedMismatchError (fail closed).", "a stale-parent fallback restarts a chain from an earlier point (replay)", "one lost top-up on a corrupt file"),
    (20, "run_segment's outcome reaches _run_phase_topup: skip → pool_exhausted, no calibration; clip → calibrate against the prediction at actual steps.", "the correction must measure the gain model, not the pool", "none"),
    (21, "Minors: persist/reload per-phase edge overlaps; layout_changed after a partial top-up keeps samples; crash-resume omits realised_h; extra tests; removed score fields warn once.", "review minors on Task 8", NR),
    (22, "Edge overlap = pairwise overlap with union f fixed, pair's samples, 2-state denominator; exact 3- vs 30-state test.", "full-union overlap diluted ~1/K; almost every synthetic edge read structural.", "thresholds need recalibration on the new scale"),
    (23, "Choose the σ_k rule empirically (min all edges / min or max same-rung / reference) by correlation with true local error.", "min rule was won by a near-identical rung twin", "none (measured)"),
    (24, "Study regime fixed from arm A only (max σ ≈ 1.5-2x target); strict 1 % win margin; two-sided bridge test; PMF RMSE co-criterion; if B does not beat A, report and stop tuning.", "prevent tuning the study to a pass", NR),
    (25, "Check the campaign-end full-union rung overlap for the same dilution on real data first (read-only).", "same computation as the diluted top-up statistic", NR),
    (26, "Separate vetoable commit switching the campaign-end gate and ladder health to the shared pairwise function; no deploy without the user.", "confirmed on chignolin_7 (38/48 vs 0/48 below 0.15)", NR),
    (27, "Bridge test keyed by (centre coords, λ); flanking edges routed with the gap and absent from the intact layout; report topped-then-routed count.", "state indices shift when centres are removed (104 vs 112 states)", NR),
    (28, "Regime by deficit fraction: budget where 10-40 % of arm-A states are above target AND median ≥ 40 decorrelated rows; else \"no targetable regime\".", "round 2 left every state deficient on 3/4 landscapes and ~5 rows/state", NR),
    (29, "Keep the patch τ penalty in the harness; the verdict must state it.", "fewer exchange partners in a sparse patch is a real cost", NR),
    (30, "Max-σ rule × structural routing interaction (a true gap costs max_edge_attempts top-ups before routing) deferred to final review.", "not changed mid-study", NR),
    (31, "Last study round; still-failing A/B tests become xfail(strict=True) citing the study JSON; per-seed detail gzipped.", "stop iterating the harness", NR),
    (32, "Fix all 8 doc inaccuracies + minors + --ap-topups help text; topups-todo T1 rewritten: chignolin_10 proceeds regardless of the synthetic result.", "user decision 2026-09-24", NR),
    (33, "One fix wave: I1 export only for top-ups-on, I2 budget from requested steps, I3 final-phase reclaim, I4 tried-out edges leave σ neighbours, I5 σ-setting neighbour in patch, I6 calibration comments, M1-M3, M5, M6, memory guard; M4, M7-M9 accepted/documented.", "final whole-branch review (opus)", NR),
    (34, "Fix I3 residual now: under force_resume clip only the delta (prior + clip(requested − prior)); exact exhaustion does not mark states skipped; summary reports extended length.", "the final-phase extension was a no-op under an enabled MD pool (20000→5999 reproduced)", NR),
]


def decisions(doc):
    doc.add_heading("3  Decisions log", 1)
    para(doc, "The subagent-driven execution recorded 34 controller rulings. Each is condensed below from "
              "sdd_ruling_ledger.md. The last column quotes the ledger's own \"cost if wrong\" where it gives one.")
    check(len(RULINGS) == 34 and [r[0] for r in RULINGS] == list(range(1, 35)), "34 rulings")
    add_table(doc, ["#", "Decision", "Reason", "Cost if wrong"], RULINGS, widths=[0.3, 3.0, 2.0, 1.5], font=7.0)
    caption(doc, "Table 6. Controller rulings 1-34.")


def synthetic(doc, D, rows, gr):
    study, meta = D["study"], D["study"]["meta"]
    doc.add_heading("4  Validation: synthetic harness", 1)
    doc.add_heading("4.1  Model", 2)
    bullets(doc, [
        f"Landscapes: four analytic 2D surfaces (rugged-2d, gated-barrier, slow-cv2-double-branch; homogeneous "
        f"control harmonic-bowl). Ladder: a uniform grid over 5-95 % of each CV axis, k = "
        f"{meta['k_window_kT']:.0f} kBT/CV² on both axes, centres 2 restraint widths apart (4 × 7 = 28 centres), "
        f"crossed with rungs λ ∈ {{0, 1/3, 2/3, 1}}: 112 states.",
        "Boost: a λ-scaled flattening of the surface below a reference energy, dV = λ·a·max(0, E_ref − F(cv)) "
        "(sampler.boost_dv). States at one centre are distinct Hamiltonians, so rung overlap is well defined. The "
        "reduced potential in the union NPZ is U_w + dV_λ.",
        "Mixing: τ from the harness's steepness model (ess.tau_int), scaled by 2/max(1, n_partners), where "
        "n_partners is the number of the window's spatial and rung partners present in its segment. A sparse "
        "patch therefore mixes worse (ruling 29). A segment of length L gives each member L/(interval·(1 + 2τ)) "
        "decorrelated samples, drawn exactly from the biased, boosted density.",
        "Common random numbers: window w's i-th decorrelated draw is the same in both arms, so the arms differ "
        "only in how many draws each state gets.",
        f"Lockstep and cost: segments advance all members equally, charged with wall_hours on the policy's default "
        f"throughput table, {meta['n_gpus']} GPUs, {meta['timestep_fs']:.0f} fs, report interval "
        f"{meta['report_interval']} steps, kT = {meta['kt_kcal']:.3f} kcal/mol.",
        f"Budget: the campaign's wall hours are split evenly over {meta['n_epochs']} epochs. Arm A runs each epoch "
        "as one all-state segment. Arm B runs a (1 − 0.3) baseline and then the real plan_topup with budget "
        "0.3 × epoch hours. Unused hours roll forward, and both arms spend the remainder in one final all-state "
        "segment. Arm B carries corrections, edge attempts and warm-start f_k across epochs.",
        "Measurement: the real union_diagnostics_from_npz (production pymbar solve, pairwise overlap, split-halves) "
        "and the real layout graphs (build_geometry_edges, layout_neighbours). Ground truth is the analytic "
        "reference PMF. PMF RMSE is the low-F-weighted RMSE (kBT) of the union-MBAR CV1 PMF against it.",
    ], size=9)

    doc.add_heading("4.2  Three study rounds, and why the first two were superseded", 2)
    add_table(doc, ["Round (commit)", "Set-up", "Result", "Why superseded"], [
        ("1 (af1a40e)", "5 h budget; full-union edge overlap; σ_k = min over edge neighbours",
         "B = A on every landscape",
         "Vacuous. Union sqrt(O_ij·O_ji) diluted ~1/K (isolated pairs 0.26-0.32 vs 0.02-0.09 inside the 112-state "
         "union), so almost every edge read structural. The min rule was always won by a rung twin (median 0.007 "
         "vs target 0.10). At 5 h every σ was 4-15x under target."),
        ("2 (f1e6a09, after 9858046 + 08f38cc)", "pairwise overlap; σ_k = max over same-rung neighbours; budget "
         "frozen from arm A at max σ ≈ 1.5-2x target",
         "PMF RMSE better nowhere. Max σ passes on gated-barrier only (20/20, p = 9.5e-7) with RMSE 9/20, p = 0.41. "
         "rugged-2d and slow-cv2 fail both; harmonic-bowl passes; two-sided bridge test fails.",
         "Review found the verdict untrustworthy. Every state was deficient on 3/4 landscapes, so B was whole-ladder "
         "extension split 70/30. The 0.7x epoch-0 baseline left 16 states with 0 rows. The bridge comparison was "
         "keyed by shifted indices. One metric was tautological. With ~5 decorrelated rows/state, split-halves "
         "could never fire."),
        ("3 (35cfacb)", "regime by deficit fraction 10-40 % AND ≥ 40 rows/state, chosen from arm A only "
         "(ruling 28); bridge keyed by coordinates (ruling 27)",
         "No landscape has a targetable regime; informational A/B at the rows floor (below)",
         "Final (ruling 31); accepted as is"),
    ], widths=[1.0, 1.7, 2.0, 2.0], font=7.3)
    caption(doc, "Table 7. Study history. Round 1 and 2 numbers are from the ledger's Task 10 entries.")

    doc.add_heading("4.3  Which σ_k rule tracks the true error", 2)
    sig = D["sigma"]
    rules = ("all_edges_min", "spatial_min", "spatial_max", "reference")
    order = ("hours=0.3", "hours=5.0", "pooled", "rugged-2d", "gated-barrier", "slow-cv2-double-branch")
    add_table(doc, ["group", "n"] + list(rules),
              [(g, sig[g]["spatial_max"]["n"]) + tuple(f"{sig[g][r]['rho']:.3f}" for r in rules) for g in order],
              widths=[1.6, 0.6, 1.1, 1.1, 1.1, 1.1])
    caption(doc, "Table 8. Spearman ρ of each candidate σ_k rule against the true local Δf error (mean |MBAR − "
                 "true| over same-rung neighbours), uniform-arm campaigns, 5 seeds × 3 heterogeneous landscapes × "
                 "2 budgets (sigma_rule_study.json). Landscape rows pool both budgets.")
    para(doc, f"The ledger quotes the pooled value: spatial_max {sig['pooled']['spatial_max']['rho']:.3f} against "
              f"all_edges_min {sig['pooled']['all_edges_min']['rho']:.3f}. That pooled number mostly reflects the "
              "difference between budgets: σ and error both shrink with more data, so any rule correlates across "
              "budgets. Within a budget the signal is weak. At 0.3 h spatial_max is clearly best "
              f"({sig['hours=0.3']['spatial_max']['rho']:.3f} against ≤ {max(sig['hours=0.3'][r]['rho'] for r in rules if r != 'spatial_max'):.3f}). "
              f"At 5 h it is tied with spatial_min ({sig['hours=5.0']['spatial_max']['rho']:.3f} vs "
              f"{sig['hours=5.0']['spatial_min']['rho']:.3f}), and the reference-state rule carries no information "
              f"(ρ = {sig['hours=5.0']['reference']['rho']:.3f}, p = {sig['hours=5.0']['reference']['p']:.2f}). "
              "The choice of the max rule is justified; the strength of the evidence is modest.")
    figure(doc, fig_sigma_rules(sig), "Figure 3. σ_k rule correlations by budget.", width=5.2)

    doc.add_heading("4.4  Regime calibration: no targetable regime", 2)
    para(doc, f"Ruling 28 requires a budget, chosen from arm A alone on calibration seeds 100-102, at which "
              f"{int(100 * meta['deficit_band'][0])}-{int(100 * meta['deficit_band'][1])} % of states are above "
              f"σ* = {meta['target_sigma_kcal']:.2f} kcal/mol AND the median state has ≥ {meta['min_rows']} "
              "decorrelated rows, the floor at which the split-halves test (≥ 20 rows per half) can fire. The rows "
              "floor is reached at about 2-2.5 wall-hours. At that budget, arm A has 0 % of states above target on "
              "every landscape. Budgets short enough to leave deficits (gated-barrier at 1-2 h: 33 % and 11 %) hold "
              "only 16-32 rows per state, too few for the drift test. So every landscape is reported as \"no "
              "targetable regime\" and none is graded pass/fail.")
    figure(doc, fig_regime(study), "Figure 4. Arm-A calibration traces. Red: fraction of states above σ*, with the "
                                   "10-40 % band shaded. Blue: median decorrelated rows per state, with the 40-row "
                                   "floor dotted. Vertical line: the rows-floor budget used for the informational A/B.")

    doc.add_heading("4.5  Informational A/B at the rows floor (20 seeds)", 2)
    body = []
    for r in rows:
        s = r["s"]
        body.append((r["ls"], f"{s['budget_hours']:.2f}",
                     f"{f3(s['median_uniform']['max_sigma'])} / {f3(s['median_topup']['max_sigma'])}",
                     f"{s['wins_max_sigma']}/20", f"{s['p_max_sigma']:.2g}",
                     f"{f3(s['median_uniform']['pmf_rmse'])} / {f3(s['median_topup']['pmf_rmse'])}",
                     f"{s['wins_pmf_rmse']}/20", f"{s['p_pmf_rmse']:.2g}",
                     f"{s['zero_row_states_epoch0']['uniform']}/{s['zero_row_states_epoch0']['topup']}",
                     f"{s['g_ratio_patch_over_all_state']:.2f}", f"{s['topped_then_routed_median']:.0f}"))
    gap = study["landscapes"]["structural_gap"]
    tr = gap["topped_then_routed_with_gap"]
    body.append((f"structural gap ({gap['landscape']}, pass removed)", f"{gap['budget_hours']:.2f}",
                 f"bridge test: {gap['seeds_passing']}/{gap['n_seeds']} seeds pass", "", "", "", "", "",
                 "", "", f"{min(tr)}-{max(tr)}"))
    add_table(doc, ["landscape", "budget h", "max σ A / B", "wins", "p", "RMSE A / B", "wins", "p",
                    "0-row A/B (epoch 0)", "g ratio", "topped→routed"], body,
              widths=[1.25, 0.5, 0.95, 0.45, 0.5, 0.95, 0.45, 0.45, 0.5, 0.45, 0.6], font=6.8)
    caption(doc, "Table 9. Medians over 20 paired seeds (max σ in kcal/mol; RMSE in kBT). A win means B is more than "
                 "1 % below A on that seed (WIN_MARGIN). p is a one-sided Wilcoxon signed-rank test, H1: B < A. A "
                 "small p with 0 wins (rugged-2d max σ) means B is consistently but less than 1 % lower. 0-row: "
                 "states with no row when epoch 0's diagnostics run (A/B); 16 in round 2, 0 here, and also 0 "
                 "at campaign end. Last row: the missing-bridge scenario (flanking edges routed structural with "
                 "the gap, not in the intact layout; topped→routed per seed). g ratio: median g(patch)/g(all-state) over topped "
                 "states, the patch τ penalty. topped→routed: median count of edges topped as noise in an earlier "
                 "epoch that end structural.")
    body = []
    for r in rows:
        body.append((r["ls"], r["n_seed_epochs"], r["n_planned"], r["patch_med"], r["n_unconverged"],
                     f"{r['s']['max_topup_md_fraction']:.3f}", f"{100 * r['s']['max_rel_hours_gap']:.2f} %"))
    add_table(doc, ["landscape", "seed-epochs", "top-ups planned", "median patch (of 112)",
                    "split-halves flags (total)", "max top-up MD fraction", "max |ΔA-B hours|"], body,
              widths=[1.4, 0.8, 0.9, 1.0, 1.0, 0.9, 0.8], font=7.3)
    caption(doc, "Table 10. What arm B actually did, counted from topup_study_detail.json.gz. Non-planned epochs "
                 "were all \"healthy\".")
    para(doc, "Reading Tables 9-10 together. On rugged-2d and harmonic-bowl, the top-ups that ran covered the whole "
              "112-state ladder (median patch 112). There B is uniform extension split into two segments, which is "
              "why A and B agree to within round-off on harmonic-bowl and why the g ratio is exactly 1 there. The "
              "allocator degenerates to uniform when every state is short, as designed. slow-cv2-double-branch was "
              "partly targeted (median patch 76). Only gated-barrier produced small patches (median 25 states). There the worst σ fell on every seed and the PMF RMSE rose by "
              "7 %: the model moves MD toward the states the σ metric ranks worst, and in this harness that does "
              "not buy PMF accuracy. The split-halves test almost never fired: " +
              ", ".join(f"{r['ls']} {r['n_unconverged']}" for r in rows) +
              " flagged states in total over 60 seed-epochs each (never more than one per solve). The missing-bridge scenario "
              f"passes {study['landscapes']['structural_gap']['seeds_passing']}/20: the edges flanking the removed "
              "pass are routed as structural with the gap and not in the intact layout. It takes "
              f"{min(study['landscapes']['structural_gap']['topped_then_routed_with_gap'])}-"
              f"{max(study['landscapes']['structural_gap']['topped_then_routed_with_gap'])} noise top-ups per seed "
              "before routing (ruling 30).")
    figure(doc, fig_ab(rows), "Figure 5. Per-seed arm A vs arm B at the rows-floor budget. Top row: max σ_k. "
                              "Bottom row: PMF RMSE.")
    med = {ls: float(np.median(v)) for ls, v in gr.items() if len(v)}
    para(doc, "Gain-model check. For every planned top-up, the realised variance reduction of each deficit state "
              "was divided by the reduction the allocator predicted. Median ratio: " +
              "; ".join(f"{ls} {m:.2f}" for ls, m in med.items()) +
              ". A ratio below 1 means the 1/sqrt(N_eff) model over-promised. The online correction is meant to "
              "absorb this; it is clamped to [0.1, 2.0].")
    figure(doc, fig_gain(gr), "Figure 6. Realised / predicted variance reduction per deficit state per planned "
                              "top-up (box: quartiles; whiskers: 1.5 IQR; outliers hidden).", width=5.4)

    doc.add_heading("4.6  Limitations", 2)
    para(doc, "The harness is not MD. Its samples are exact i.i.d. draws thinned by a modelled τ, so it cannot show "
              "unsampled basins, slow drift or seeding transients: the failure modes split-halves and continuation "
              "seeding were built for. The τ-vs-partners model and the throughput table are assumptions shared by "
              "both arms. The landscapes are small (28 centres); chignolin_9 has 236 states. The regime criterion "
              "(10-40 % deficits at ≥ 40 rows) is a controller choice, and a different rows floor could admit a "
              "regime. The study is frozen at 35cfacb. The final fix wave (ruling 33: tried-out edges leave the σ "
              "neighbour set, σ-setting neighbour forced into the patch) is in production code but not in the "
              "harness (topup_model._diagnose docstring), so the verdict describes the allocator before those two "
              "changes. The study neither supports nor rules out a benefit on real MD; chignolin_10 is the test.")


def real_data(doc, D, dil):
    pairs = D["pairs"]
    doc.add_heading("5  Validation: real data (chignolin_7 overlap dilution)", 1)
    para(doc, "Ruling 25 asked whether the campaign-end rung overlap, which indexed the full-union "
              "mbar_state_overlap matrix, had the same dilution the synthetic study exposed. A read-only check on "
              "RUNS/chignolin_7 answered yes. That union has K = 64 states (16 CV1 centres × 4 rungs) and 2.39 M "
              "finite rows. For each of the 48 adjacent-rung pairs, the check computed the full-union "
              "sqrt(O_ij·O_ji) and the pairwise value (same f_k).")
    add_table(doc, ["statistic", "full union", "pairwise"], [
        ("median", f3(dil["full_med"]), f3(dil["pw_med"])),
        ("range", f"{dil['full_min']:.3f}-{dil['full_max']:.3f}", f"{dil['pw_min']:.3f}-{dil['pw_max']:.3f}"),
        ("pairs below 0.15 (min_rung_overlap)", f"{dil['full_lt15']}/{dil['n']}", f"{dil['pw_lt15']}/{dil['n']}"),
        ("pairs below 0.25 (target_rung_overlap)", f"{dil['full_lt25']}/{dil['n']}", f"{dil['pw_lt25']}/{dil['n']}"),
        ("median ratio pairwise/full", f"{dil['ratio_med']:.2f}", ""),
    ], widths=[2.6, 1.4, 1.4])
    caption(doc, "Table 11. chignolin_7 adjacent-rung pairs (rung_pair_table.csv).")
    kinds = sorted({(p["lambda_i"], p["lambda_j"]) for p in pairs})
    body = []
    for li, lj in kinds:
        sel = [p for p in pairs if (p["lambda_i"], p["lambda_j"]) == (li, lj)]
        body.append((f"{li:.3f} → {lj:.3f}", len(sel), f"{int(sel[0]['n_samples_ij']):,}",
                     f3(statistics.median(p["overlap_full_union_a"] for p in sel)),
                     f3(statistics.median(p["overlap_pairwise_b"] for p in sel))))
    add_table(doc, ["rung pair (λ)", "pairs", "samples per pair", "full-union median", "pairwise median"], body,
              widths=[1.3, 0.6, 1.3, 1.4, 1.4])
    caption(doc, "Table 12. By rung pair. The top pair (0.643 → 1.0) is the best-overlapping on both scales. On the "
                 "pairwise scale no pair is below 0.15; the lowest pairs read about 0.20.")
    figure(doc, fig_dilution(pairs), "Figure 7. chignolin_7 rung-pair overlap along CV1. Circles: pairwise. Crosses: "
                                     "full-union matrix entries. Red lines: the 0.15 floor and 0.25 target.")
    para(doc, "Consequence. With the full-union statistic, the campaign-end gate graded 38 of 48 rung pairs below "
              "the rung floor, each an add_rung candidate, on a ladder where no pair is below it pairwise. Commit f7e007b switched rung_mbar_overlap_from_union (gate and add_rung) "
              "and analyze_gareus_mbar.py's ladder_overlap health check to the shared pairwise_state_overlap. The "
              "review reproduced the table (median 0.25788, 0/48 below 0.15, max difference 7e-16) at about 21 ms "
              "per edge. On chignolin_7 both ladder-health axes flip toward pass: lambda_direction goes from 38/48 "
              "to 0/48 pairs below 0.15, and cv1_direction from 46/60 to 0/60.")
    para(doc, "Caveat, not resolved. min_rung_overlap = 0.15 and target_rung_overlap = 0.25 were calibrated on the "
              "S3 pilot's 5-state full-matrix entries (0.298/0.250/0.240/0.273), which are themselves diluted. The "
              "pairwise statistic has a different scale (two identical states read 0.5). The thresholds were "
              "carried over unchanged and have not been re-measured pairwise (todo T5). A pairwise pass is "
              "therefore graded against a bar set on another scale; it is not a calibrated pass. A warning now "
              "fires once when ladder_overlap grades a full matrix (M6).")


def performance(doc):
    doc.add_heading("6  Performance and cost", 1)
    admit = 8.0e9 / (236 * UNION_PEAK_BYTES_PER_CELL)
    add_table(doc, ["quantity", "value", "source"], [
        ("union diagnostics, 236 states, 250,000 rows", "26.4 s, 4.06 GB peak RSS", "gareus.synth.union_solve_bench"),
        ("union diagnostics, 236 states, 1,000,000 rows", "102.7 s, 14.6 GB peak RSS", "gareus.synth.union_solve_bench"),
        ("peak model", "rows × states × 8 B × 7.7", "1e6 × 236 × 8 B = 1.888 GB → 7.7x"),
        ("--ap-topup-diagnostics-max-gb default", "8.0 GB", "cli.py"),
        ("kept rows admitted at 236 states", f"{admit:,.0f}", "8 GB / (236 × 61.6 B)"),
        ("chignolin_9 campaign-wide raw rows", "~686,000", "ledger, final-wave re-review"),
        ("solves per top-ups-on phase", "up to 2 (plan, post-top-up calibration)", "adaptive_production.py"),
        ("pairwise overlap per edge (c7, K = 64)", "~21 ms", "ledger, ruling 26 review"),
        ("final_window_states per State, 19k atoms", "~3 MB", "topup_seeding.py docstring"),
        ("final_window_states per 236-window segment", "~0.74 GB (≈ 3.1 MB/State)", "ledger, final review I1"),
    ], widths=[2.8, 2.0, 1.9])
    caption(doc, "Table 13. Measured and derived costs. The solve runs on the driver/analysis node, not a GPU.")
    para(doc, "The guard compares KEPT (post-subsampling) rows, while the chignolin_9 figure is RAW rows. Whether a "
              "c9-sized campaign trips the guard depends on its subsampling ratio. If it does, every phase ends as "
              "no_diagnostics and runs no top-up: safe, but an A/B on such a campaign would compare uniform "
              "extension with itself. Todo T6 asks for this to be settled (subsample the union input, or confirm "
              "node RAM and raise the limit) before chignolin_10 enables top-ups. With top-ups off, "
              "final_window_states is no longer written (I1); before that fix it would have cost ~0.74 GB per "
              "236-window segment on every campaign.")


TEST_NOTES = {
    "test_topup_allocator.py": "healthy → no top-up; unmeasured edges never deficient; minimal partners; structural vs noise routing; attempt escalation; never-sampled partners; cap_too_small; worst-deficit length and budget cap; degeneration to all states; correction clamps; determinism; σ-argmax in patch; rung edges vs min_rung_overlap",
    "test_topup_union_diagnostics.py": "σ shrinks with samples; split-halves sensitivity, specificity, false-alarm control at scale; g from builder meta; warm start; missing/NaN input warns; pairwise overlap invariant under union size (3 vs 30 states, identical-pair 0.5); sparse state not rescued by a rung twin; sigma_argmax",
    "test_topup_epoch_wiring.py": "uniform schedule; shortened baseline + exactly one top-up; budget from un-shortened default; seed-mismatch/corrupt-index end the top-up only; plan reuse, layout change, one per phase across resume; pool skip/clip calibration; top-up writes its epoch_window_map; overlaps reach diagnostics",
    "test_topup_final_wave.py": "full_steps for old/new schedules; final-phase baseline extension cases; export only for top-ups-on segments; tried-out edges leave σ neighbours; structural-edge log; memory-guard estimate and trip; extension under a real AdaptiveRuntimePool",
    "test_topup_seeding.py": "export/reload round trip (Reference platform); CV and restraint assertions; truncated XML / corrupt index handling; window-map fail-closed; export_seq ordering over mtime and name",
    "test_topup_state.py": "plan/state JSON round trip (int keys, NaN), atomic writes",
    "test_topup_throughput.py": "interpolation and flat extrapolation of the throughput table; wall_hours",
    "test_topup_flags.py": "--ap-topups off by default; knob defaults; malformed table exits",
    "test_unmeasured_not_weak.py": "_edge_is_measured_weak for rung/spatial edges; gate, proposer and converged ignore unmeasured edges",
    "test_pairwise_rung_overlap.py": "pairwise value identical in 4- and 40-state unions while full-matrix drops; campaign-end gate and ladder health use it",
    "test_layout_neighbours.py": "same-rung neighbours, other-rung partners, λ rounding",
    "test_pymbar_check.py": "pymbar status reports any import exception",
    "test_pymbar_warning_wiring.py": "warning printed once per process",
    "test_union_subsample_counts.py": "union meta records raw, t0, kept, g per state",
    "test_synth_topup_study.py": "boost and τ model; equal wall-hours between arms; homogeneous parity; bridge routing by coordinates; frozen regime; study writer; one strict xfail",
    "test_helptext_topups.py": "-hh top-ups topic exists and is well formed",
}


def tests(doc):
    doc.add_heading("7  Test coverage", 1)
    changed = subprocess.run(["git", "diff", "--name-status", f"{BASE}..{HEAD}", "--", "tests/"], cwd=REPO,
                             capture_output=True, text=True, check=True).stdout.split("\n")
    new = sorted(Path(l.split("\t")[1]).name for l in changed if l.startswith("A\t") and l.endswith(".py"))
    body = []
    for name in new:
        n = len(re.findall(r"^\s*def test_", (REPO / "tests" / name).read_text(), flags=re.M))
        if name in TEST_NOTES:
            body.append((name, n, TEST_NOTES[name]))
    add_table(doc, ["new test file", "tests", "what it pins"], body, widths=[2.0, 0.5, 4.2], font=7.3)
    caption(doc, "Table 14. Test files added on the branch (test counts by def test_ at the branch head). Six existing "
                 "files were adjusted: test_adaptive_ladder_rungs, test_adaptive_new_state_guards, "
                 "test_allocation_articulation_degeneracy (dead-helper tests removed, M5), test_ladder_overlap_axes, "
                 "test_no_topups, test_synth_double_adaptive.")
    para(doc, "Strict xfail. test_synth_topup_study.py::test_topups_beat_uniform_on_a_heterogeneous_landscape asserts "
              "that B beats A on both max σ and PMF RMSE on gated-barrier in at least 4 of 5 seeds. It is marked "
              "xfail(strict=True) under ruling 31, citing topup_study.json: no targetable regime, and at the rows "
              "floor B wins max σ but not RMSE. Being strict, it turns red if the harness or allocator ever starts "
              "passing it, which forces the verdict to be revisited rather than silently changing.")
    para(doc, "Recorded results are targeted batches only; no full-suite run was recorded for this branch (user "
              "preference: targeted tests). Final fix wave: 202 passed + 1 xfail. I3 residual fix: 152 passed "
              "(2 red on the old code). Task 10 final round: 108 passed + 1 xfail. Ruling 26: 73 passed. Deploy "
              "branch: 55 targeted tests passed. Pre-existing and unrelated: tests/test_atlas_md_docs.py has 2 "
              "failures (ATLAS-MD vs ATLaS-MD branding, a missing mermaid block in docs/index.md). The files involved "
              "are untouched since the branch base 4ea0814, and the branch does not change them.")


def operations(doc):
    doc.add_heading("8  Operations and open items", 1)
    doc.add_heading("8.1  aurum2 environment and deploy", 2)
    bullets(doc, [
        "pymbar was unusable in the aurum2 calc env: import failed on a scipy 1.13 / jax incompatibility, and the "
        "driver silently skipped subsampling and the union MBAR. Task 0 made this loud: pymbar_status catches any "
        "import exception, and warn_if_pymbar_unusable fires once on every adaptive-production entry path.",
        "2026-09-25 17:15, at chignolin_9's epoch_002 → final boundary (job 2647937): jax 0.3.0 and jaxlib 0.1.75 "
        "were uninstalled from the calc env. pymbar 4.0.1 now imports, MBAR and detect_equilibration work, and "
        "equilibrated_subsample reports status subsampled. No graceful stop was needed: the final schedule was "
        "already persisted and the next walltime resubmit picks up the fixed env. Revert: "
        "pip install jax==0.3.0 jaxlib==0.1.75.",
        "2026-09-25 20:40: the pairwise-overlap fix alone (branch deploy/pairwise-overlap = e74dd30 + cherry-pick "
        "of f7e007b, minus the top-up-only union_diagnostics.py → dda8caa) was rsynced to both aurum2 trees plus "
        "analyze_gareus_mbar.py. DEPLOYED_COMMIT dda8caa; the tree md5s match; 55 targeted tests passed. The "
        "running c9 job keeps the old code in memory and the next resubmit loads it. Merged to main as PR #103 "
        "(merge 90b7eba).",
        "Not deployed: everything else on feat/effective-topups, including the uniform-share allocation and the "
        "unmeasured-never-weak rule. They change chignolin_9's behaviour only after merge and a separate deploy.",
    ], size=9)
    doc.add_heading("8.2  How to enable top-ups", 2)
    add_table(doc, ["flag (YAML key = argparse dest: drop --, dashes → underscores)", "default", "meaning"], [
        ("--ap-topups / --no-ap-topups", "off", "enable at most one top-up per scheduled phase"),
        ("--ap-topup-target-sigma", "0.10 kcal/mol", "σ* per state"),
        ("--ap-topup-weak-overlap", "0.15", "spatial-edge weak threshold (rung edges use min_rung_overlap 0.15, not a CLI flag)"),
        ("--ap-topup-max-fraction", "0.3", "share of the phase's un-shortened wall-hour budget held for the top-up"),
        ("--ap-topup-min-effect", "0.05 kcal/mol", "split-halves effect floor"),
        ("--ap-topup-max-edge-attempts", "2", "noise top-ups an edge gets before it is structural"),
        ("--ap-topup-throughput-table", "16:3154,59:2300", "contexts_per_gpu:ns_per_day_node pairs"),
        ("--ap-topup-diagnostics-max-gb", "8.0", "skip diagnostics (no_diagnostics) above this estimated peak"),
    ], widths=[2.6, 1.1, 3.0])
    caption(doc, "Table 15. Top-up knobs (gareus/cli.py). A config that already sets ap_topups: true gets the new "
                 "mechanism, not the old one. The removed score-policy fields still load and warn once if non-default.")
    doc.add_heading("8.3  Open items (docs/atlas-md/developer/topups-todo.md)", 2)
    add_table(doc, ["id", "item", "status"], [
        ("T1", "chignolin_10: real-MD A/B of top-ups vs uniform extension at equal wall-hours; same contexts/GPU "
               "(or report targeting and throughput separately); paired seeds, distinct RNG after the branch; "
               "replicates. Success: lower max per-state σ and block-bootstrap PMF uncertainty, no worse wall-clock "
               "per ns, no more dead exchange pairs.", "not started; proceeds regardless of the synthetic result"),
        ("T2", "Throughput curve T(n) at 4, 8, 16, 24, 32, 48, 59 contexts/GPU on the production node, replacing "
               "the provisional two-point chignolin_8 table.", "not started"),
        ("T3", "Real-file seeding spike on chignolin_9 epoch_001/baseline exports; check the seeding assertion.",
         "not started"),
        ("T4", "Sequential-panel benchmark (~16 ctx/GPU panels vs one 236-context run; ~0.73x wall-time if "
               "throughput holds, no cross-panel exchange).", "not started"),
        ("T5", "Re-measure min_rung_overlap 0.15 / target_rung_overlap 0.25 on the pairwise scale.", "not started"),
        ("T6", "Union-diagnostics memory before chignolin_10: subsample the union input or size the node.",
         "not started"),
    ], widths=[0.4, 4.8, 1.5], font=7.5)
    caption(doc, "Table 16. Open follow-ups.")


def appendices(doc, D):
    doc.add_heading("Appendix A  Commits " + f"{BASE}..{HEAD}", 1)
    add_table(doc, ["commit", "date", "subject"], [tuple(c) for c in D["commits"]], widths=[0.7, 0.9, 5.1], font=7.0)
    caption(doc, f"Table A1. {len(D['commits'])} commits, oldest first (git log --reverse).")
    doc.add_heading("Appendix B  Glossary", 1)
    add_table(doc, ["term", "meaning"], [
        ("phase", "a scheduled adaptive-production unit: a numbered epoch_NNN or final"),
        ("baseline", "the phase's all-state segment"),
        ("top-up (segment)", "topup_001_<L>: extra lockstep MD on the patch only; at most one per phase"),
        ("lockstep", "all replicas of a segment advance the same number of steps; cost L/T(n)"),
        ("rung / λ", "a Pep-GaMD boost strength λ ∈ [0, 1]; a state is a (window, rung) pair"),
        ("same-rung neighbour", "nearest spatial layout neighbour at the same λ (gareus.layout_neighbours)"),
        ("rung partner / twin", "a state at the same centre on another rung"),
        ("σ_k", "local free-energy uncertainty: max MBAR δΔf to same-rung spatial neighbours (minus tried-out edges), × kT"),
        ("σ*", "target σ (--ap-topup-target-sigma, 0.10 kcal/mol)"),
        ("deficit", "a sampled state with σ_k > σ* or a split-halves flag"),
        ("split-halves flag", "per-state Welch test of own reduced bias, first vs second half, Bonferroni + effect floor"),
        ("g_k", "statistical inefficiency raw/kept from the subsampling meta; one decorrelated sample per g_k·interval steps"),
        ("N_eff", "kept (decorrelated) rows of a state in the union"),
        ("c_k", "per-state calibration correction in [0.1, 2.0], learned from realised vs predicted σ"),
        ("pairwise overlap", "sqrt(O_ij·O_ji) from the pair's own samples, 2-state denominator, union f_k fixed"),
        ("full-union overlap", "the same entry of the K-state MBAR overlap matrix; diluted by other states in the region"),
        ("noise edge", "weak edge touching an initial deficit, attempts < max: both endpoints topped up"),
        ("structural edge", "weak edge between adequate states, or tried out: left to bridge/add-state, never MD"),
        ("unmeasured edge", "no overlap value (e.g. unsampled endpoint); never weak"),
        ("patch", "deficits plus partners: the top-up's state set"),
        ("MD pool", "campaign-wide remaining MD budget; clips or skips segments"),
        ("final_window_states", "per-segment export of each window's State + index.json, seeds a later top-up"),
        ("export_seq", "time.time_ns() write stamp ordering parent exports"),
        ("arm A / arm B", "synthetic study: uniform extension / top-ups at equal modelled wall-hours"),
        ("win", "a seed on which B is more than 1 % below A"),
        ("rows floor", "budget at which the median state holds ≥ 40 decorrelated rows"),
        ("targetable regime", "a budget with 10-40 % of arm-A states above σ* at or above the rows floor"),
    ], widths=[1.6, 5.1], font=7.5)
    caption(doc, "Table B1. Glossary.")


def main() -> int:
    D = load_data()
    dil = dilution_stats(D["pairs"])
    rows = study_rows(D["study"], D["detail"])
    for r in rows:
        check(r["n_planned"] + sum(1 for x in D["detail"][r["ls"]] for e in x["epochs"]
                                   if e.get("plan", {}).get("reason") == "healthy") == r["n_seed_epochs"],
              f"{r['ls']}: only planned/healthy reasons")
    gb = next(r for r in rows if r["ls"] == "gated-barrier")["s"]
    check(gb["wins_max_sigma"] == 20 and abs(gb["p_max_sigma"] - 9.5367431640625e-07) < 1e-12, "gb max σ")
    check(gb["wins_pmf_rmse"] == 7 and abs(gb["rel_diff"]["pmf_rmse"] - 0.070) < 0.001, "gb RMSE +7 %")
    check(len(D["commits"]) == 43, "43 commits")
    gr = gain_ratios(D["detail"])

    doc = Document()
    doc.styles["Normal"].font.name = "Calibri"; doc.styles["Normal"].font.size = Pt(9.5)
    for s in doc.sections:
        s.left_margin = s.right_margin = Inches(0.8)
    title_and_summary(doc, D, rows, dil)
    background(doc, D)
    design(doc)
    decisions(doc)
    synthetic(doc, D, rows, gr)
    real_data(doc, D, dil)
    performance(doc)
    tests(doc)
    operations(doc)
    appendices(doc, D)
    doc.save(OUT)
    print(f"wrote {OUT}: {_counts['tables']} tables, {_counts['figures']} figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
