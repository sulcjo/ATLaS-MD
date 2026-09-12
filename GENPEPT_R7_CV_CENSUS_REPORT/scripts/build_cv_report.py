from __future__ import annotations

import base64
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler")
REPORT = BASE / "GENPEPT_R7_CV_CENSUS_REPORT"
FIG = REPORT / "figures"
OUT = REPORT / "GENPEPT_R7_CV_CAMPAIGN_REPORT.html"

st = pd.read_csv(REPORT / "genpept_r7_cv_census_stats.csv")
st = st.sort_values("entropy", ascending=False).reset_index(drop=True)
four = json.load(open(REPORT / "genpept_r7_fourier.json"))
dmap = json.load(open(REPORT / "genpept_r7_diffmap.json"))


def b64(name):
    return "data:image/png;base64," + base64.b64encode((FIG / name).read_bytes()).decode()


def fig(name, caption):
    return f'<figure><img src="{b64(name)}"><figcaption>{caption}</figcaption></figure>'


def row_stat(cv):
    return st[st.cv == cv].iloc[0]


def stat_cells(cv):
    r = row_stat(cv)
    return (f"<td>{cv}</td><td>{r['family']}</td><td>{r['median']:.3f}</td><td>{r['std']:.3f}</td>"
            f"<td>{r['IQR']:.3f}</td><td>{r['entropy']:.3f}</td><td>{r['BC']:.3f}</td><td>{int(r['nw_k1200'])}</td>")


top15_rows = "\n".join(f"<tr><td>{i+1}</td>{stat_cells(cv)}</tr>"
                       for i, cv in enumerate(st.cv.head(15)))


def twosafe(x):
    return f"{x:.2f}"


typ_sections = []
tw = dmap["twonn"]
spec_lines = "".join(
    f"<tr><td>{tag}</td>" + "".join(f"<td>{l:.3f}</td>" for _, l, _ in vals[:7]) + "</tr>"
    for tag, vals in dmap["spectra"].items())
lasso_lines = "".join(
    f"<tr><td>{k}</td><td>{v['r2']:.3f}</td><td>{'; '.join(f'{c} ({w:+.2f})' for c, w in v['top'][:4])}</td></tr>"
    for k, v in dmap["lasso"].items())
stab_lines = "".join(
    f"<tr><td>{tag}</td><td>{v.get('0.5x', float('nan')):.3f}</td><td>{v.get('2.0x', float('nan')):.3f}</td></tr>"
    for tag, v in dmap["psi1_stability"].items())

circ = four["circ"]
fourierA_lines = "".join(
    f"<tr><td>{lab}</td><td>{c['R1']:.3f}</td><td>{c['R2']:.3f}</td><td>{c['R3']:.3f}</td><td>{c['R4']:.3f}</td>"
    f"<td>{math.degrees(c['mean_dir']):+.0f}&deg;</td>"
    f"<td>{'<b>bimodal</b>' if (c['R1'] < 0.35 and c['R2'] > 0.25) else ''}</td></tr>"
    for lab, c in circ.items())
circ_top = four["circ_corr_top"][:10]
circC_lines = "".join(f"<tr><td>{a}</td><td>{b}</td><td>{r:+.3f}</td></tr>" for a, b, r in circ_top)
windB_lines = "".join(
    f"<tr><td>{o}</td><td>{v['threshold_p99']:.2f}</td><td>{'; '.join(f'{l} ({t:.2f})' for l, t, _, _ in v['flagged']) or '<b>none</b>'}</td></tr>"
    for o, v in four["winding"].items())


pairs_df = pd.read_csv(REPORT / "genpept_r7_orthogonal_pairs.csv")
def pair_rows(df_, n):
    out = []
    for i, (_, r) in enumerate(df_.head(n).iterrows(), 1):
        label2 = "(residual torsion PC1, re-fit vs this CV1)" if r.cv2 == "resPC1_refit" else r.cv2
        out.append(f"<tr><td>{i}</td><td>{r.cv1}</td><td>{label2}</td><td>{r.pear:+.2f}</td><td>{r.spear:+.2f}</td>"
                   f"<td>{r.occ*100:.0f}</td><td>{int(r.min_bin)}</td><td>{r.cond_iqr_med:.2f}</td>"
                   f"<td>{r.ent1:.3f}</td><td>{r.ent2:.3f}</td><td>{r.score:.3f}</td></tr>")
    return "\n".join(out)

pair_source = pairs_df[pairs_df.control.isna() | (pairs_df.control == "")]
top_pairs = pair_rows(pair_source.sort_values("score", ascending=False), 8)
ctrl_rows = "".join(
    f"<tr><td>{r.cv1}</td><td>{r.cv2}</td><td>{r.pear:+.2f}</td><td>{r.spear:+.2f}</td><td>{r.score:.3f}</td></tr>"
    for _, r in pairs_df[pairs_df.control == "NEGATIVE"].nlargest(4, "score").iterrows())
bb_pairs = pairs_df[pairs_df.cv1 == "c_bb_s4_r010_b3"].sort_values("score", ascending=False).head(6)
bb_rows = "".join(
    f"<tr><td>{r.cv2}</td><td>{r.spear:+.2f}</td><td>{r.occ*100:.0f}</td><td>{int(r.min_bin)}</td>"
    f"<td>{r.cond_iqr_med:.2f}</td><td>{r.score:.3f}</td></tr>"
    for _, r in bb_pairs.iterrows())


dyn = json.load(open(REPORT / "genpept_r7_dynamics_cv1.json"))
steer = json.load(open(REPORT / "genpept_r7_steer_stability.json"))
misj = json.load(open(REPORT / "genpept_r7_mi_states_energy.json"))
rig = json.load(open(REPORT / "genpept_r7_rigor.json"))

def fmt_def_row(k):
    cv = dyn["coverage_p005_995"]
    return (f"<tr><td>{k}</td><td>{cv['bank'][k]:.3f}</td><td>{cv['dynamics'][k]:.3f}</td>"
            f"<td>{dyn['pooled_mean_std'][k][0]:.3f}&plusmn;{dyn['pooled_mean_std'][k][1]:.3f}</td>"
            f"<td>{dyn['halflife_ps_median'][k]:.0f}</td><td>{steer['steerability'][k]['rms_grad_mean_dynamics_band']:.1e}</td></tr>")

dyn_rows = "".join(fmt_def_row(k) for k in ("bb_r010_b3", "ca_r010_b3", "heavy_r012_b3"))
legacy_steer = steer["steerability"]["legacy_heavy_r04.5_b6"]
stab = steer["half_bank"]
gsp = steer["generator_split"]

mi_rows_html = "".join(
    f"<tr><td>{r['cv2']}</td><td>{r['mi_nats']:.3f}</td><td>{r['mi_null_mean']:.3f}</td><td>{r['mi_excess']:+.3f}</td><td>{r['spearman']:+.2f}</td></tr>"
    for r in misj["mi"])
state_lines = "".join(
    f"<tr><td>S{i}</td><td>[{st_['lo']:.2f},{st_['hi']:.2f}]</td><td>{st_['n']}</td><td>{st_['frac']*100:.1f}%</td>"
    f"<td>{st_['psi_means'][list(st_['psi_means'])[0]]:+.0f} / {st_['psi_means'][list(st_['psi_means'])[2]]:+.0f} / {st_['psi_means'][list(st_['psi_means'])[4]]:+.0f}</td></tr>"
    for i, st_ in enumerate(misj["kde_states"]))
bt = rig["bootstrap"]
dn = dyn["n_frames_total"]
fz = f"{legacy_steer['frac_nearzero_grad_in_band']*100:.1f}"
pc1 = steer["half_bank"]["pc1_median"]
pc2 = steer["half_bank"]["res1_median"]
pc2min = steer["half_bank"]["res1_min"]
res1 = pc2
res1min = pc2min
geo_len = rig["geodesic"]["len"]
geo_ok = f"{rig['geodesic']['frac_edges_ok']*100:.0f}"
line_ok = f"{rig['geodesic']['frac_line_ok']*100:.0f}"

bt_rows = "".join(
    f"<tr><td>{c}</td><td>{v['entropy']:.3f}</td><td>[{v['ci95'][0]:.3f},{v['ci95'][1]:.3f}]</td><td>{v['nw_k1200']}</td><td>[{v['nw_ci95'][0]:.0f},{v['nw_ci95'][1]:.0f}]</td></tr>"
    for c, v in bt.items() if "entropy" in v)


rep2 = json.load(open(REPORT / "genpept_r7_vs_rep2.json"))
rep2_rows = "".join(
    f"<tr><td>{r['cv']}</td><td>{r['r7']['median']:.3f}</td><td>{r['r2']['median']:.3f}</td>"
    f"<td>{r['r7']['entropy']:.3f}</td><td>{r['r2']['entropy']:.3f}</td><td>{r['r7']['nw']}</td><td>{r['r2']['nw']}</td>"
    f"<td>{r['ks_stat']:.3f}</td></tr>"
    for r in rep2["per_cv"])
rep2_cos = rep2["axis_cos"]
ladder_repr = rep2["ladder_common_axis"]
lad_first = ladder_repr[0]
lad_last = ladder_repr[-1]


rep3_st = pd.read_csv(REPORT / "genpept_rep3_cv_census_stats.csv")
rep3_top_rows = "".join(
    f"<tr><td>{i+1}</td><td>{r.cv}</td><td>{r.family}</td><td>{r['median']:.3f}</td><td>{r['IQR']:.3f}</td>"
    f"<td>{r['entropy']:.3f}</td><td>{r['BC']:.3f}</td><td>{int(r.nw_k1200)}</td></tr>"
    for i, r in rep3_st.head(12).iterrows())
rep3_adv = json.load(open(REPORT / "genpept_rep3_advanced.json"))
rep3_mi_rows = "".join(
    f"<tr><td>{r['cv2']}</td><td>{r['mi']:.3f}</td><td>{r['excess']:+.3f}</td></tr>"
    for r in rep3_adv["mi"])
def _rep3_stab_row(k, v):
    if "median" in v:
        right = v.get("min", float("nan"))
    else:
        right = v.get("pc2", float("nan"))
    right_s = f"{right:.3f}" if right == right else "-"
    left = v.get("median", v.get("pc1", float("nan")))
    return f"<tr><td>{k}</td><td>{left:.3f}</td><td>{right_s}</td></tr>"

rep3_stab_rows = "".join(
    _rep3_stab_row(k, v) for k, v in rep3_adv["stability"].items() if "median" in v or "pc1" in v)
rep3_energy_rows = "".join(
    f"<tr><td>{t}</td><td>{v['median']:.1f}</td><td>{v['rho_cv1']:+.2f}</td><td>{v['rho_cv2']:+.2f}</td></tr>"
    for t, v in rep3_adv["energies"].items())
rep3_tw = rep3_adv["twonn_3k"]
rep3_dm = rep3_adv["dmap_3k"]
rep3_kde = rep3_adv["kde_states"]
rep3_folded = json.load(open(REPORT / "genpept_rep3_fes_folded.json"))["banks"]["rep3_massive"]


ent = json.load(open(REPORT / "genpept_rep3_energy_terms.json"))
def trend_line(axis):
    v = {t: ent["trend_slopes"][f"{t}_vs_{axis}"] for t in ("bond", "angle", "torsion", "nonbonded", "gb_solvation", "total")}
    return ("<tr><td>vs " + axis + "</td>"
            + "".join(f"<td>{v[t]['slope_kj_per_unit']:+.1f}</td>" for t in
                      ("bond", "angle", "torsion", "nonbonded", "gb_solvation", "total")) + "</tr>")
trend_rows = "".join(trend_line(a) for a in ("cv1", "cv2", "rg", "e2e"))
nnv = ent["near_native"]
nn_rows = "".join(
    f"<tr><td>{t}</td><td>{nnv['delta_mean_minus_rest'][t]:+.1f}</td><td>{nnv['perm_p'][t]:.3f}</td></tr>"
    for t in ("bond", "angle", "torsion", "nonbonded", "gb_solvation", "total"))

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>GENPEPT r7 seed bank — CV selection campaign, full report</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 0; background: #f7f7f5; color: #1d1d1b; }}
  main {{ max-width: 1080px; margin: 0 auto; background: #fff; padding: 2.4em 3em 4em; box-shadow: 0 0 18px #ddd; }}
  h1 {{ font-size: 1.55em; margin-bottom: .1em; }}
  h2 {{ border-bottom: 2px solid #2c5f8a; padding-bottom: .15em; margin-top: 2.1em; color: #2c5f8a; }}
  h3 {{ color: #444; margin-bottom: .3em; margin-top: 1.6em; }}
  h4 {{ margin-bottom: .2em; }}
  .sub {{ color: #666; margin-top: 0; }}
  .toc {{ background: #eef3f7; border: 1px solid #d5e0e8; border-radius: 6px; padding: .8em 1.2em; font-size: .93em; }}
  .toc ol {{ margin: .3em 0 0 1.2em; padding: 0; }}
  .toc .chp {{ margin: .55em 0 .1em 0; font-weight: 700; color: #2c5f8a; }}
  .toc .chp:first-child {{ margin-top: .2em; }}
  .chapter {{ background: linear-gradient(90deg, #2c5f8a, #3d7ab0); color: #fff; padding: .55em 1em; margin: 3em 0 1.2em 0; border-radius: 6px; }}
  .chapter .chn {{ font-size: .74em; letter-spacing: .16em; text-transform: uppercase; opacity: .8; margin: 0; }}
  .chapter h2 {{ border: none; color: #fff; margin: .1em 0 0 0; padding: 0; font-size: 1.25em; }}
  table {{ border-collapse: collapse; margin: .7em 0 1.2em; font-size: .83em; }}
  th, td {{ border: 1px solid #d8d8d8; padding: .28em .55em; text-align: right; }}
  th {{ background: #2c5f8a; color: #fff; font-weight: 600; }}
  td:first-child, th:first-child {{ text-align: left; font-family: ui-monospace, monospace; font-size: .95em; }}
  tr:nth-child(even) td {{ background: #f2f6fa; }}
  figure {{ margin: 1.2em 0; text-align: center; }}
  figure img {{ max-width: 100%; border: 1px solid #ddd; }}
  figcaption {{ font-size: .84em; color: #555; margin-top: .35em; }}
  .note {{ background: #fdf6e5; border-left: 4px solid #d8a000; padding: .5em .9em; font-size: .9em; }}
  .ok {{ background: #ecf6ec; border-left: 4px solid #479e47; padding: .5em .9em; font-size: .9em; }}
  .bad {{ background: #fbeeee; border-left: 4px solid #c0392b; padding: .5em .9em; font-size: .9em; }}
  code, .mono {{ font-family: ui-monospace, monospace; font-size: .92em; background: #f0f0ee; padding: .1em .3em; border-radius: 3px; }}
  pre {{ background: #2b2b28; color: #e8e8e4; padding: .9em 1.1em; border-radius: 6px; overflow-x: auto; font-size: .83em; }}
  .did {{ margin: .6em 0 .9em 1.1em; }}
  .did dt {{ font-weight: 600; color: #2c5f8a; }}
  .did dd {{ margin: .2em 0 .7em 0; }}
</style>
</head>
<body><main>
<h1>GENPEPT r7 seed bank — collective-variable selection campaign</h1>
<p class="sub">Complete report · compiled 2026-09-10 (chapter edition) · bank: <code>RUNS/chignolin_genpept_r7/final_implicit_survivor_seeds/</code>
(1,970 survivors, chignolin <code>GYDPETGTWG</code>, 138 atoms, GBn2-minimized) · all numbers re-derived from data files in <code>GENPEPT_R7_CV_CENSUS_REPORT/</code></p>

<div class="toc"><b>Contents</b>
<p class="chp">Chapter 1 — Foundations: the r7 bank, the CV census, CV selection</p>
<ol>
<li><a href="#exec">Executive summary</a></li>
<li><a href="#data">Dataset and measurement definitions</a></li>
<li><a href="#anchors">Correctness anchors (including the OXT defect)</a></li>
<li><a href="#census">Contact-CV census: 298 descriptors, rankings</a></li>
<li><a href="#cv2">CV2: bank torsion PCA, residualization, dynamics tests</a></li>
<li><a href="#maps">Pseudo-FES map figures (v2 family)</a></li>
<li><a href="#dmap">Diffusion maps on the bank</a></li>
<li><a href="#twonn">Intrinsic dimensionality (TwoNN)</a></li>
<li><a href="#fourier">Fourier analysis of dihedrals — three passes, each explained</a></li>
<li><a href="#turn">The physical turn axis of the residual CV2</a></li>
<li><a href="#pairs">Alternative orthogonal pairs</a></li>
<li><a href="#decision-rigor">Decision dynamics + rigor pack</a></li>
</ol>
<p class="chp">Chapter 2 — Scale-up and reproducibility</p>
<ol start="13">
<li><a href="#rep2">Bank-to-bank reproducibility (rep2)</a></li>
<li><a href="#folded">Folded-state proximity (1UAO) + rep3 pseudo-FES</a></li>
<li><a href="#rep3full">Full re-analysis on the 10k bank</a></li>
<li><a href="#energy-terms">Energy terms behavior on the FES maps</a></li>
</ol>
<p class="chp">Chapter 3 — The universal tuning bundle (rep4)</p>
<ol start="17">
<li><a href="#rep4">rep4: implemented tuning bundle — build, run, comparison</a></li>
<li><a href="#genpept-now">What GENPEPT is doing now — stage by stage</a></li>
<li><a href="#rep5">rep5: T7 restraint-forming kicks — first movement of the folded-state metric</a></li>
</ol>
<p class="chp">Chapter 4 — The thermodynamic layer on the bank</p>
<ol start="19">
<li><a href="#thermo-fes">From search density to a starting FES (basin level)</a></li>
<li><a href="#consensus">Consensus solvent-model check (GBn2 vs OBC2)</a></li>
<li><a href="#guided">Guided register-rewarded expansion — an honest negative</a></li>
<li><a href="#fes-starting">Starting FES on the working CV axes (CV1 × PC1)</a></li>
<li><a href="#fes-heavy">Starting FES on heavy contacts × Rg</a></li>
<li><a href="#closest">The closest bank seeds vs folded chignolin</a></li>
<li><a href="#fes-rep5">Starting FES on the rep5 (T7) bank</a></li>
</ol>
<p class="chp">Chapter 5 — Synthesis</p>
<ol start="25">
<li><a href="#residual-context">The residual motion in context</a></li>
<li><a href="#rec">Final recommendation</a></li>
<li><a href="#files">Files, reproduction, limitations</a></li>
</ol></div>

<div class="chapter"><p class="chn">Chapter 1</p><h2>Foundations: the r7 bank, the CV census, CV selection</h2></div>

<h2 id="exec">1.1 Executive summary</h2>
<ul>
<li><b>CV1 (primary) — backbone-heavy logistic contact fraction</b> (GAREUS <code>backbone-heavy</code>:
N,CA,C,O only; r≥4, atom-pairs, normalized), at <b>r&#8320; = 10 &#197;, β ≈ 3 &#197;&#8315;&#185;</b>:
entropy 0.979, IQR 0.385, 26–27 resolvable 1.5σ windows at k = 1200 — measured best of 298 candidates.
Substitutes: CA-only at r&#8320;10 (cheap, tied) and all-heavy at r&#8320;12 (swarm-side validated).</li>
<li><b>Legacy definition (r&#8320; 4.5, β 6) is degenerate</b> &le; 2 resolvable windows on every selection.</li>
<li><b>CV2 — plain torsion PC2</b>: a strand-register flip that (unlike the residual axis)
needs no residualization — it is orthogonal to every compaction CV by PCA construction, and the
bank-robustest scalar (half-bank |cos| 0.983 vs the residual scalar's collapse-prone 0.90).
Physically a hairpin-register/turn flip including the D3 anchor; also a diversity coordinate
(Hess C₁ ≈ 0.05), not a slow-motion one.</li>
<li><b>Manifold math agrees on CV1</b> (diffusion-map ψ₁ = compaction, LASSO R² = 0.985) and warns that
the bank is ~4-dimensional (TwoNN d ≈ 3.6–4.6) — a 2-D grid covers, does not parameterize.</li>
<li><b>Fourier harmonics expose the hidden signal</b>: interior ψ torsions are bimodal
(R&#8321; 0.13–0.34, R&#8322; 0.58–0.73) — the same flip the residual CV2 surfaces; no other periodic
signal survives a permutation null.</li>
</ul>

<h2 id="data">1.2 Dataset and measurement definitions</h2>
<p>The bank comes from GENPEPT run r7 (2026-07-16): 200,000 proposals with
<code>diversity_bank_preset=chignolin</code>, <code>contact_bias_strength=0.15</code>, clash cutoff 1.6 &#197;,
seed 12345 → 3,000 implicit candidates (+ basin-hopping, NMA, PCA-frontier) → 1,970 final GBn2-minimized
survivor PDBs. It is a <i>search</i> library, not a Boltzmann sample; every statistic here measures
coverage/differentiation of that library.</p>
<p><b>Contact-fraction CVs</b> use GAREUS's exact semantics
(<code>gareus.cv.nonlocal_contact_cv_from_positions_nm</code>): per selected atom pair with residue
separation ≥ s, c<sub>ij</sub> = 0.5·(1 − tanh(β·(r<sub>ij</sub> − r&#8320;)/2)), normalized by the number of
terms (or residue-balanced). Atom selections: <code>heavy</code> (all non-H), <code>bb</code> = backbone-heavy
(N,CA,C,O), <code>sch</code> = sidechain-heavy, <code>sca</code> = sidechain-all, <code>ca</code>.</p>
<p><b>Quality metrics per CV:</b> std, IQR, p05–p95; normalized 20-bin entropy H/ln 20; bimodality
coefficient BC (flag &gt; 5/9); resolvable-window count n = ⌊span<sub>p01–p99</sub> / (1.5·σ<sub>w</sub>)⌋ with
σ<sub>w</sub> = √(kT/k), k = 1200 kcal/mol/CV² at 300 K (same formula as
<code>gareus.swarm.ladder_design.n_resolvable_windows</code>).</p>

<h2 id="anchors">1.3 Correctness anchors (including the OXT defect)</h2>
<table><tr><th>check</th><th>result</th></tr>
<tr><td>anchor vs earlier parameter-test table (heavy r≥4 r&#8320;12 β1.5)</td>
<td>mean .5435, std .1969, p05 .1789, p95 .8246 — exact</td></tr>
<tr><td>single-seed oracle vs <code>gareus.cv.nonlocal_contact_cv_from_positions_nm</code></td><td>Δ = 1.5×10&#8315;&#8312;</td></tr>
<tr><td>per-seed parity vs GAREUS's pair builder on 6 spot configs × 1,970 seeds</td><td>max |Δ| ≤ 1.8×10&#8315;&#8311; on all five selections</td></tr>
</table>
<div class="note"><b>Defect found &amp; fixed during this campaign:</b> the census's backbone-heavy selection
initially included OXT. bb-CVs were off by up to 4.5×10&#8315;&#178;. After correction everything
matches GAREUS to 10&#8315;&#8311;, and residue-balanced ≡ atom-pairs <i>exactly</i> for backbone-heavy
(uniform 4×4 atom cells per residue pair). All tables/figures in this document are post-fix.</div>

<h2 id="census">1.4 Contact-CV census — 298 descriptors</h2>
<p>298 CVs computed per seed (logistic contacts across 5 selections × seps 2/3/4 × 14 (r&#8320;,β) sets;
residue-balanced variants; PLUMED rational switches; strand-packing descriptors; shape, H-bond, Ramachandran,
torsion-PCA, SASA/burial, reference CVs). Full tables: <code>report_tables.txt</code>,
<code>genpept_r7_cv_census_values.csv</code> (1,970 × 298, no NaN).</p>
<h3>Top 15 by normalized entropy</h3>
<table><tr><th>#</th><th>CV</th><th>family</th><th>median</th><th>std</th><th>IQR</th><th>entropy</th><th>BC</th><th>nw@k1200</th></tr>
{top15_rows}
</table>
<p class="note" style="background:none;border:none;padding-left:0"><code>cresb_bb_*</code> ≡ <code>c_bb_*</code>
byte-identical (uniform backbone cells).</p>
<h3>How the landscape reads</h3>
<ul>
<li><b>r&#8320; placement dominates.</b> Below ~6 &#197; every selection collapses (legacy: median ≤ 0.017, ≤ 2 windows).
The informative plateau is 8–12 &#197;; at 14 &#197; the median saturates (&gt; 0.78) and entropy falls again.</li>
<li><b>β is nearly free</b> once r&#8320; is in band (Δentropy &lt; 0.02 for β 1.5–16); β ≈ 3 is a fine default.</li>
<li><b>Residue-balanced ≈ atom-pairs</b> (|r| &gt; 0.99 everywhere; exact identity for bb). Pick one.</li>
<li><b>Rational switches lose</b> (best crat entropy 0.965 vs 0.980 logistic). Keep logistic.</li>
<li><b>Degenerate-by-design CVs</b>: backbone H-bond switch counts (implicit minima: median 0), strand minimum
distance (std 0.002 &#197;, steric wall), cis-ω, left-helical Ramachandran count, sidechain-all at any
tight setting (drops all three glycines — only 12 residue pairs over 7 residues).</li>
<li><b>Redundancy:</b> Rg/mean-Cα-dist ≈ mid-r&#8320; contact fraction (|r| 0.94–0.98); strand-interface fraction
≡ the all-heavy global fraction (r = 0.998, since a hairpin's nonlocal contacts ARE the strand pairing);
<code>contact_order_ca8</code> moves with every contact CV (r 0.82–0.93).</li>
</ul>
{fig("genpept_r7_cv_histograms.png", "28 representative CV distributions; titles carry median / IQR / entropy.")}
{fig("genpept_r7_cv_corr.png", "Clustered |Pearson r| redundancy map of representative CVs: bright blocks are single coordinates seen from several angles.")}
{fig("genpept_r7_cv_2dmaps.png", "CV1 × CV2 scatters: contact CV1 vs residual torsion PC1 (color = Rg); contact vs plain torsion PC1 (color = d(Y2–W9)); Rg vs hairpin closure (color = CV1).")}

<h2 id="cv2">1.5 CV2: bank torsion PCA, residualization, dynamics tests</h2>
<ul>
<li><b>Axis identity</b> (canonical <code>gareus.tica</code> interleaved sin/cos, fit on the bank): residual PC1
(residualized on the winning CV1) = PETGT-turn/strand ψ axis — cos ψ P4 −0.44, cos ψ W9 −0.43, cos/sin ψ D3
−0.37/−0.35. Residualization rotates <i>within</i> the same top-5 subspace (principal-angle overlap 0.986), i.e.,
it removes the compaction component rather than finding new motion.</li>
<li><b>Dynamics verdict (Hess cosine content, 32 chignolin_6 epoch-0 traces × 4,768 frames):</b> bank axes are
<i>not diffusive</i> — median C₁ ≈ 0.07, 0/32 ≥ 0.5; trace-own PC1 only reaches 0.388. A bank CV2 is a
diversity coordinate; a slow CV2 has to be fit on dynamics (REBUILD.md's frozen within-segment tICA).</li>
<li><b>No bank-to-bank axis transfer:</b> per-trace corr(bank axis, chignolin_6's stored bootstrap CV2) median
−0.514, mixed sign — eigenvectors must be persisted per run, never re-fit from a moved library.</li>
<li><b>Grid feasibility:</b> 16 × 8 quantile joint maps give 94–100% occupancy on bank and production ensemble;
per-CV1-bin CV2 IQR stays ≥ 0.74 — window grid is well-founded.</li>
</ul>
{fig("cv2_loadings.png", "Residual-PC1 top-18 torsion loadings and explained-variance spectrum (flat: PC1 ≈ 9.7%).")}
{fig("cv2_diffusion_content.png", "Hess C₁ of projected axes on 32 epoch-0 traces. Bank plain/residual PC1 ≈ 0.07; red line = diffusion criterion.")}
{fig("cv2_joint_maps.png", "CV1 × CV2 16×8 joint maps on bank and chignolin_6 epoch-0, occupancy labeled in each title.")}

<h2 id="maps">1.6 Pseudo-FES map figures (v2 family)</h2>
<p>Same layouts as the earlier parameter tests, regenerated on the census winners. The grey base is the
seed-density pseudo-FES (search density — not thermodynamics).</p>
{fig("r7v2_contact_beta_pseudofes.png", "β-scan pseudo-FES curves + spread panels: heavy r012 vs backbone-heavy r010 vs CA r010.")}
{fig("r7v2_combo_map_bb_beta3.png", "Combo map for backbone-heavy contacts (sep × midpoint grid) over the Rg/E2E pseudo-FES.")}
{fig("r7v2_combo_map_ca_beta3.png", "Combo map for CA-only contacts.")}
{fig("r7v2_pseudofes_cv_bb_beta3_r0_10_r4.png", "pseudo-FES colored by the winning CV1 (bb, r010, β3).")}
{fig("r7v2_pseudofes_residual_torsion_pca_cv2.png", "pseudo-FES colored by residual torsion-PCA CV2 (component=5, residualized on bb r010).")}
<p>(video companion: <code>figures/r7v2_residual_torsion_cv2_geometric_morph.mp4</code> — aligned geometric
interpolation along the residual CV2 axis at constant CV1 ≈ 0.49, seeds 73 → 1259, CV2 −2.00 → +1.98.)</p>
{fig("r7v2_residual_torsion_cv2_geometric_morph_preview.png", "Morph preview frame: flank (ψ β-extended) → curled but D3–G7 anchor unchanged.")}

<h2 id="dmap">1.7 Diffusion maps on the bank</h2>
<p><b>What this is:</b> instead of assuming a coordinate, build the manifold's own geometry: kernel
K<sub>ij</sub> = exp(−d(x<sub>i</sub>,x<sub>j</sub>)²/2ε²) on pairwise distances between seeds (two metrics:
z-scored interleaved torsion sin/cos 36-D, and z-scored Cα dRMSD 45-D), symmetric Markov normalization,
eigendecompose; ψ<sub>k</sub> with eigenvalue λ<sub>k</sub> are the manifold's orthogonal slow-geometry
coordinates (diffusion time 1/λ<sub>k</sub>). ε = median pairwise distance, robustness re-fit at 0.5× / 2× ε.</p>
<h3>Spectra (λ<sub>k</sub>, trivial ψ&#8320; shown first)</h3>
<table><tr><th>metric</th><th>λ&#8321;</th><th>λ&#8322;</th><th>λ&#8323;</th><th>λ&#8324;</th><th>λ&#8325;</th><th>λ&#8326;</th><th>λ&#8327;</th></tr>
{spec_lines}</table>
<h3>Eigenvector identification</h3>
<p>Two independent reads: Spearman correlations of each ψ against all 298 census CVs, and LASSO (5-fold CV)
of each ψ onto the interpretable descriptors. Both agree:</p>
<table><tr><th>eigenvector</th><th>LASSO R²</th><th>top interpretable features (coefficient)</th></tr>
{lasso_lines}</table>
<div class="ok">The bank's first diffusion coordinate <b>is</b> the compaction/contact axis
(LASSO R² = 0.985 on CA-dRMSD) — the hand-picked CV1 and the nonlinear manifold agree, which is the
strongest argument for the bb r&#8320;10 definition. The residual CV2 surfaces only at ψ&#8323; level:
higher-order, not hidden.</div>
<h3>ε robustness of ψ&#8321;</h3>
<table><tr><th>metric</th><th>|ρ| at 0.5×ε</th><th>|ρ| at 2.0×ε</th></tr>{stab_lines}</table>
{fig("cv_diffmap_spectra_maps.png", "Diffusion-map spectra and ψ₁×ψ₂ maps (row per metric), colored by contact CV1 (left) and residual torsion PC1 (right).")}

<h2 id="twonn">1.8 Intrinsic dimensionality (TwoNN)</h2>
<p><b>What this is:</b> Facco et al.'s TwoNE estimator. For every seed take the ratio μ of 2nd-to-1st
neighbor distances; on a d-dimensional manifold the CDF of μ is P(μ) = 1 − μ<sup>−d</sup>, so fitting
−ln(1 − P) = d·ln μ gives d with no embedding and no parameters. Two independent distance spaces:</p>
<ul>
<li>torsion space (z-scored 36-D sin/cos): <b>d = {tw['torsion']['d']:.2f} ± {2*tw['torsion']['se']:.2f}</b></li>
<li>Cα-dRMSD space (z-scored 45-D): <b>d = {tw['ca_drmsd']['d']:.2f} ± {2*tw['ca_drmsd']['se']:.2f}</b></li>
</ul>
<div class="note">The r7 bank's explorations occupy a <b>~4-dimensional</b> manifold. A 2-D window grid
(campaign design) is a cover of the dominant coordinates, not a parameterization of the space — expect
curvature pressure on the second axis rather than assuming both axes carry equal independent weight.</div>
{fig("cv_twonn_id.png", "TwoNN fits: −ln(1−P(μ)) vs ln μ, slope = intrinsic dimension, for both distance metrics.")}

<h2 id="fourier">1.9 Fourier analysis of dihedrals — three passes</h2>
<h3>9A. Circular Fourier harmonics (per torsion)</h3>
<dl class="did">
<dt>What we did:</dt>
<dd>Each backbone torsion is a circular variable θ; its distribution over the bank is a density on the circle.
Its natural Fourier decomposition is the family of circular moments
R<sub>m</sub> = |⟨e<sup>imθ</sup>⟩|, m = 1..4 — literally the magnitudes of the angular Fourier coefficients.
R&#8321; measures concentration around one direction; a LOW R&#8321; with HIGH R&#8322;/R&#8323;/R&#8324; is the signature
of a multi-lobe (multimodal) distribution. Computed for all 18 backbone torsions over 1,970 seeds.</dd>
<dt>What came out:</dt>
<dd>All φ concentrated (R&#8321; 0.61–0.98, mean ≈ −75..−85°). Every interior ψ is <b>bimodal</b>:
R&#8321; 0.13–0.34 with R&#8322; 0.58–0.73 — a two-lobe structure split between the extended β basin
(≈ +90..+150°) and a curled basin (≈ −10°). This bimodality is invisible to any linear (variance/mean)
statistic on the raw angle.</dd>
</dl>
<table><tr><th>torsion</th><th>R1</th><th>R2</th><th>R3</th><th>R4</th><th>mean dir</th><th></th></tr>{fourierA_lines}</table>

<h3>9B. Ordered-signal winding scan (hidden periodic pumps)</h3>
<dl class="did">
<dt>What we did:</dt>
<dd>A cyclic motion that winds while a manifold coordinate advances would look like noise to linear
correlation but show up as a systematic drift in the unwrapped angle series after sorting seeds by that
coordinate. For each ordering ξ ∈ {{torsion-ψ&#8321;, CA-dRMSD-ψ&#8321;, contact-CV1, residual-CV2}}:
sort the bank by ξ, unwrap each torsion's θ along the ordering, measure total winding
(|slope|·n/2π turns) plus FFT spectral concentration of the demeaned series. Significance = the max over
all 18 torsions compared against a permutation-null distribution of that same max statistic
(N = 300 orderings; threshold = max-statistic p99).</dd>
<dt>What came out:</dt>
<dd><b>Nothing above the null.</b> No torsion exhibits significant multi-turn winding along any tested
coordinate — there are no further hidden periodic signals beyond the bimodality found in 9A.</dd>
</dl>
<table><tr><th>ordering coordinate</th><th>null max p99 (turns)</th><th>torsions above threshold</th></tr>{windB_lines}</table>
{fig("cv_fourier_winding.png", "Winding counts per ordering vs the permutation p99 (red dashed): no torsion crosses.")}

<h3>9C. Circular–circular correlations (Fisher–Lee)</h3>
<dl class="did">
<dt>What we did:</dt>
<dd>Pearson correlation is undefined for angles (θ and θ+2π are the same point but numerically distant).
Fisher–Lee is the circular analogue: correlate sin(θ<sub>i</sub>−θ̄<sub>i</sub>) against sin(θ<sub>j</sub>−θ̄<sub>j</sub>),
with circular means. Computed for all 153 torsion pairs.</dd>
<dt>What came out:</dt>
<dd>Within-residue φ–ψ coupling dominates (ρ ≈ −0.35 to −0.24 — the textbook Ramachandran correlation);
between-residue couplings are all weak (|ρ| ≤ 0.19: ψ(Y2)–ψ(E5) +0.19, ψ(E5)–ψ(T6) +0.18, φ(W9)–ψ(T8) +0.18).
No hidden long-range torsion network.</dd>
</dl>
<table><tr><th>torsion i</th><th>torsion j</th><th>ρ</th></tr>{circC_lines}</table>
{fig("cv_fourier_harmonics_corrmatrix.png", "Left: circular harmonics R1–R4 per torsion. Right: Fisher–Lee circular correlation matrix.")}

<h2 id="turn">1.10 The physical turn axis of the residual CV2</h2>
<p>To say what the residual CV2 physically is, seeds were split at the CV2 deciles (low ≤ −1.44, n = 197;
high ≥ +1.31, n = 197), <b>matched on CV1</b> (bb r&#8320;10 means 0.433 vs 0.474 — same compaction), and every
backbone torsion's circular mean was compared low → high:</p>
<table><tr><th>residue</th><th>φ low → high (deg)</th><th>Δφ</th><th>ψ low → high (deg)</th><th>Δψ</th></tr>
<tr><td>Y2</td><td>−96 → −79</td><td>+17</td><td>+146 → −16</td><td><b>−163</b></td></tr>
<tr><td>D3</td><td>−79 → −75</td><td>+4</td><td>−42 → −39</td><td>+3</td></tr>
<tr><td>P4</td><td>−73 → −70</td><td>+3</td><td>+145 → −12</td><td><b>−157</b></td></tr>
<tr><td>E5</td><td>−87 → −72</td><td>+15</td><td>+91 → −16</td><td><b>−107</b></td></tr>
<tr><td>T6</td><td>−82 → −80</td><td>+2</td><td>+100 → −7</td><td><b>−107</b></td></tr>
<tr><td>G7</td><td>−104 → −75</td><td>+29</td><td>+54 → −9</td><td>−63</td></tr>
<tr><td>T8</td><td>−96 → −76</td><td>+20</td><td>+125 → −11</td><td><b>−136</b></td></tr>
<tr><td>W9</td><td>−101 → −83</td><td>+19</td><td>+149 → −12</td><td><b>−160</b></td></tr>
</table>
<p>Reading: the flank residues (Y2, P4, E5, T6, T8, W9) swing ψ from the β/pP-II plateau (+91..+149°)
to a curled conformation (−7..−16°, i.e. α<sub>R</sub>-edge/γ-turn) while the turn anchor D3 stays pinned at
ψ ≈ −40°. At constant compaction, the axis tightens the turn loop (ρ = −0.54 vs
<code>turn3537_mean_ca</code>, d(P4–G7) ρ = −0.29) <b>without moving strand–strand contacts</b>
(ρ ≈ 0 vs <code>strand_cross_far</code>, ≈ 0.10 vs CV1). This is exactly the orthogonal-to-CV1 motion
residualization is designed to keep — a strand-flank curl, not another compaction axis.</p>
{fig("cv2_turn_axis.png", "Per-residue Ramachandran colored by the residual CV2, per-residue Δφ/Δψ across deciles, decile histograms, and CV2 correlation with physical descriptors.")}

<h2 id="pairs">1.11 Alternative orthogonal pairs — 60 combinations tested</h2>
<dl class="did">
<dt>What we did:</dt>
<dd>Every candidate (CV1 &times; CV2) pair was scored for what a window grid actually needs: mutual
independence (Pearson r<sub>p</sub>, Spearman r<sub>s</sub>), joint 16&times;8 quantile-grid occupancy,
minimum per-bin count, and <b>conditional spread</b> &mdash; the median CV2 IQR inside CV1 bins relative to
the global IQR (does CV2 still differentiate once CV1 is pinned at a window centre?). Composite score =
min(ent&#8321;, ent&#8322;) &times; (1 &minus; |r<sub>s</sub>|) &times; occupancy &times; fraction-of-bins with
conditional IQR &ge; 50% of global. The residual axis is re-fit against each candidate CV1
(<code>resPC1_refit</code>), and deliberately redundant pairs were planted as negative controls.</dd>
<dt>What came out:</dt>
<dd>All negative controls score &asymp; 0 &mdash; the metric fails pairs when it should. The full 60-pair
table lives in genpept_r7_orthogonal_pairs.csv; the leaders:</dd>
</dl>
<table><tr><th>#</th><th>CV1</th><th>CV2</th><th>r_p</th><th>r_s</th><th>occ %</th><th>min N</th><th>cond IQR med</th><th>ent1</th><th>ent2</th><th>score</th></tr>
{top_pairs}
</table>
<h3>Planted negative controls (must score low &mdash; they do)</h3>
<table><tr><th>CV1</th><th>CV2</th><th>r_p</th><th>r_s</th><th>score</th></tr>
{ctrl_rows}
</table>
<h3>All CV2 options vs the champion CV1 (bb r&#8320;10)</h3>
<table><tr><th>CV2</th><th>r_s</th><th>occ %</th><th>min N</th><th>cond IQR med</th><th>score</th></tr>
{bb_rows}
</table>
<div class="ok"><b>Verdict:</b> champion stands (bb r&#8320;10 &times; bank residual torsion PC1), with
<b>plain torsion PC2 as the designated fallback CV2</b> &mdash; orthogonal by PCA construction
(|r<sub>s</sub>| 0.04&ndash;0.07 against every candidate CV1, no residualization needed).
The top zone (0.87&ndash;0.92) is a tie; CV1 still follows the census and dynamics checks, not this table.
Bank-side independence does not imply dynamical independence: bank-orthogonal motions can be kinetically coupled.</div>
{fig("cv_orthogonal_pairs_top.png", "Top alternative pairs: joint hex-count maps with r_s, occupancy and min-bin titles.")}

<h2 id="decision-rigor">1.12 Decision dynamics &amp; rigor pack (11 follow-on checks)</h2>
<h3>12a. Dynamics-side validation (chignolin_6 epoch_000, {dn} frames, 96 replica XTCs)</h3>
<table><tr><th>CV1 def</th><th>p0.5–p99.5 bank</th><th>dynamics</th><th>pooled mean±std</th><th>median ACF t½ (ps)</th><th>RMS |∇CV/∂x| (band)</th></tr>
{dyn_rows}
</table>
<p>Coverage transfers essentially intact (bank 0.93 vs MD 0.92 pooled for bb/ca; 0.84 vs 0.81 for heavy) —
the bank-side winners are real dynamics-side coordinates. Median decorrelation ≈ 0.2–0.3 ns per restrained
replica. The legacy def gauges steerability: RMS gradient 0.0 and <b>{fz}% of its own p05–p95 band has
~zero gradient</b> — the pull-stall mechanism measured on production, reproduced as a geometric property.</p>
<h3>12b. Axis stability: plain PCs are robust, the residual scalar is not</h3>
<p>half-bank |cos| medians: plain PC1 {pc1:.3f}, plain PC2 {pc2:.3f} (min {pc2min:.3f});
residual PC1 {res1:.3f} (min {res1min:.3f}); generator-split residual PC1 vs initial-seed-only 0.752.
Eigenvalues backing: residual PC1/PC2 are nearly degenerate (0.101 vs 0.093) — the residual direction is a
2-D eigenspace, not a reliable scalar. <b>CV2 designation switches to plain torsion PC2</b>: orthogonal by
construction, and the robustest option in both tests.</p>
<h3>12c. Mutual information (Kraskov k=3) &amp; discrete intermediates</h3>
<table><tr><th>CV2</th><th>MI (nats)</th><th>null</th><th>excess</th><th>ρ</th></tr>{mi_rows_html}</table>
<p>Contact-order confirms dependence (excess +1.0 nat) as does turn3537 (+0.33); acceptable CV2 candidates
hold excess &le; +0.12. The curl axis is not a continuum: a fine-bandwidth KDE resolves <b>6 discrete states</b>
(S0 all-extended ψ &rarr; S4/S5 all-curled), representative seeds in genpept_r7_mi_states_energy.json.</p>
<table><tr><th>state</th><th>CV2 range</th><th>n</th><th>frac</th><th>ψ(Y2 / P4 / T8) mean</th></tr>{state_lines}</table>
<h3>12e. The designated CV2 scalar: plain torsion PC2</h3>
<p>Top loadings (cos/sin ψ of D3, P4, W9, Y2, E5, T8; sin φ G7) and decile-matched circular-mean
rotations (ψ D3 −44&rarr;+143&deg;, ψ P4 +152&rarr;&minus;14&deg;, ψ W9/Y2/T8/E5 all flip; φ G7 −57&deg;):
plain PC2 is the same family of register/curl flip as the residual axis but <b>includes the D3 turn
anchor</b> rather than pinning it. Deciles are compaction-matched (CV1 0.520 vs 0.527) with zero
residualization. Diffusion content on the 32 epoch-0 traces: median C₁ 0.052, 0/32 &ge; 0.5 (same
non-diffusive honesty note). Fine-bandwidth KDE resolves 7 discrete modes along it.</p>
<h3>12d. Rigor pack (bootstrap, realism, provenance, bias, omega-cis, tiling)</h3>
<table><tr><th>CV</th><th>entropy</th><th>CI95</th><th>nw@k1200</th><th>CI95</th></tr>{bt_rows}</table>
<ul>
<li>morph path realism: geodesic {geo_len} steps ({geo_ok}% edges within median kNN distance), linear
interpolation stays within populated density for {line_ok}% of points</li>
<li>provenance: PCA-frontier seeds are compaction-shifted (CV1 mean 0.734 vs 0.429 initial) —
bank diversity is not generator-uniform</li>
<li>contact-bias strength 0.15 enriched compact states ×3–5 vs unbiased reconstruction
(contact_count tail cc&ge;15: 12.8% vs 2.6%)</li>
<li>cis-ω(D3–P4) = 0.003 — not a bank coordinate</li>
<li>k-center tiling of (CV1, CV2, Rg, e2e): <code>seed_bank_tiling_k16.csv</code></li>
<li>full-bank GBn2 decomposition: genpept_r7_energy_components.csv (entropy/BC/&rho; per term)</li>
</ul>
{fig("cv_dynamics_cv1_validation.png", "12a: pooled CV coverage on MD vs bank; per-replica decorrelation times.")}
{fig("cv_steerability.png", "12b: per-atom steering strength vs CV value; legacy def dead-zone visible.")}
{fig("cv_mi_states_energy.png", "12c: MI bars, 6-state curl ladder, energy components vs CV2.")}
{fig("cv2_plain_pc2.png", "12e: plain PC2 loadings, joint map vs CV1, diffusion C1 on epoch-0 traces.")}

<div class="chapter"><p class="chn">Chapter 2</p><h2>Scale-up and reproducibility</h2></div>

<h2 id="rep2">2.1 Bank-to-bank reproducibility — an independent rep2 bank, same parameters, different seed</h2>
<dl class="did">
<dt>What we did:</dt>
<dd>reran GENPEPT with r7's exact params (200k proposals, <code>chignolin</code> preset, contact bias 0.15,
full backend, BH 120×(3+1), NMA 2 modes × {{0.75,1.35 Å}} × 2 signs, 5 PCA-frontier rounds, adaptive GBn2
minimization, OpenCL) but with seed 54321 instead of 12345 → second bank <code>chignolin_genpept_rep2/</code>
(1,984 survivors; 2,994/3,000 implicit success, identical to r7's 99.8%; 472 BH minima vs 480 — inferred
parent pool landed 118± ≠). All headline machinery re-measured on the new bank.</dd>
<dt>What came out:</dt>
<dd><b>Statistics and rankings replicate.</b> Historically identical window ceilings (26↔26, 27↔27, 24↔24,
23↔23), entropy within ≤0.012, KS shifts of &asymp;0.08 in medians only (rep2 bank is marginally more
compact). The eigenvalue spectrum transfers ([0.122,0.092,…] vs [0.111,0.096,…]).
<b>A new honest caveat appears:</b> across independent banks, plain PC1 transfers (|cos| 0.965) but the
top-5 torsion subspace rotates substantially (plain PC2 |cos| only 0.523; residual PC1 0.453; top-5 overlap
0.865) — CV2 axes are bank-local even when bank-stable. Fit CV2 where the campaign samples, never inherit
it from another bank. On the COMMON r7 PC2 axis, both banks reproduce the same 6-state curl ladder to
&asymp;15&deg; per torsion and state composition &asymp; indistinguishable (corr(bank indicator, projection)
= +0.037), so the ladder itself is physical; its eigenspace labeling is what drifts.</dd>
</dl>
<table><tr><th>CV</th><th>median r7</th><th>median rep2</th><th>entropy r7</th><th>entropy rep2</th><th>nw r7</th><th>nw rep2</th><th>KS</th></tr>
{rep2_rows}
</table>
<table><tr><th>axis</th><th>|cos| between banks</th></tr>
<tr><td>plain PC1</td><td>{rep2_cos['plain_pc1']:.3f}</td></tr>
<tr><td>plain PC2</td><td>{rep2_cos['plain_pc2']:.3f}</td></tr>
<tr><td>residual PC1</td><td>{rep2_cos['res_pc1']:.3f}</td></tr>
<tr><td>top-5 subspace overlap (mean cos&sup2;)</td><td>{rep2_cos['plain_pc1x2_top5_overlap']:.3f}</td></tr>
</table>
{fig("cv_rep2_comparison.png", "Headline CVs overlaid for r7 vs rep2 banks: same ranking, same ceilings, small median shifts.")}
{fig("cv_rep2_axis_agreement.png", "PCA axis agreement across independent banks: PC1 robustly transfers; PC2 and residual do not.")}
{fig("cv_rep2_common_axis.png", "Common-axis projections (top row) and the 6-state ladder texture per bank on the common r7 PC2 axis (bottom): states reproduce to ~15 deg per torsion with composition nearly identical.")}
{fig("cv_rep2_stage_counts.png", "Pipeline stage counts: 3000/2994/472-480/1984-1970 — near-identical funnel.")}

<h2 id="folded">2.2 Folded-state proximity (experimental reference) + rep3 pseudo-FES</h2>
<dl class="did">
<dt>What we did:</dt>
<dd>fetched the chignolin NMR ensemble (RCSB 1UAO, 18 models, identical sequence GYDPETGTWG) and measured
every bank seed: min C&alpha;-RMSD across the ensemble, the two literature-native pair distances
d(D3&nbsp;N,&nbsp;G7&nbsp;O) and d(D3&nbsp;N,&nbsp;T8&nbsp;O), DSSP content, and the rep3 pseudo-FES maps
(5&times; denser than r7). Also re-measured the "folded corner" honestly: the earlier both-&lt;4&#197;
criterion excludes the native ensemble itself (native d37 = 6.2&ndash;7.6 &#197;), so occupancy is reported
against the native-derived envelope instead.</dd>
<dt>What came out:</dt>
<dd>Near-native structure exists as a <b>tail</b>: best RMSD 1.01 &#197; (rep3), 1.61 &#197; (r7);
~11% of seeds within 2.5 &#197;; ~32% within 3 &#197;. The native-pair envelope holds 2 seeds (0.10%) in r7
and 31 seeds (0.31%) in rep3 — growing with bank size. The folded register remains a tail, not a mode, of
the library; near-native seeds come disproportionately from exploration stages. rep3 pseudo-FES maps
carry 19.4% bin occupancy on the CV1&times;PC1 map.</dd>
</dl>
{fig("cv_rep3_massive_folded_proximity.png", "rep3: RMSD-to-1UAO distribution, native-pair pseudo-FES with the ensemble marked, working-CV-space location of near-native seeds, DSSP split.")}
{fig("cv_r7_folded_proximity.png", "r7 (smaller bank, same panels) — proximity gradient scales with bank size.")}
{fig("cv_rep3massive_pseudofes.png", "rep3 pseudo-FES maps: CV1xPC2, Rg x E2E, CV1xPC1 (search density, not equilibrium FES).")}

<h2 id="rep3full">2.3 The complete campaign battery, re-run on the 10,019-seed bank</h2>
<dl class="did">
<dt>What we did:</dt>
<dd>reran every bank-side analysis of this report on rep3_massive with identical machinery: the 298-CV census
(after fixing a genuine rational-switch midpoint singularity found on the big bank: one pair at exactly
r&#8320; gives 0/0; the l'H&ocirc;pital limit n/m is the correct value), statistics and rankings, contact grid,
5&times;15 orthogonality matrix, joint maps, diffusion maps on a 3k subset, TwoNN, Fourier (harmonics,
winding, Fisher-Lee), MI, half-bank and generator-split axis stability, and the full-bank GBn2 decomposition
(with a group-assignment ordering fix: contexts snapshot forces at construction).</dd>
<dt>What came out:</dt>
<dd>Every headline conclusion survives on 5&times; density — and two sharpen. Rankings: CA-only r&#8320;10
&#946;6 marginally edges backbone-heavy (entropy 0.982 vs 0.979, window ceilings 28/27) — a statistical tie at
the top, legacy still degenerate. <b>Axis stability markedly improves with bank size:</b> half-bank |cos|
median(min) = 0.998(0.994) plain PC1, 0.994(0.975) plain PC2, 0.991(0.943) residual PC1 — even the residual
scalar becomes bank-robust at 10k seeds, and generator splits reach 0.98. TwoNN: d 3.5 torsion / 4.7 CA-dRMSD
(same ~4-D manifold). KDE resolves 3 macro-states on this bank. Energy decomposition on all 10,019:
bond/angle/torsion components drift slightly with CV2 (|&rho;| &le; 0.34), nonbonded and GB-solvation carry
large canceling variance (&sigma; 135 / 124 kJ/mol) while total stays tight (&sigma; 26.7 kJ/mol) — the
compensation is structural, not noise.</dd>
</dl>
<h3>Top 12 by entropy on rep3</h3>
<table><tr><th>#</th><th>CV</th><th>family</th><th>median</th><th>IQR</th><th>entropy</th><th>BC</th><th>nw@k1200</th></tr>
{rep3_top_rows}</table>
<h3>Intrinsic dimension and manifold (3k subset)</h3>
<p>TwoNN: torsion d = {rep3_tw['torsion_3k']:.2f}, CA-dRMSD d = {rep3_tw['ca_drmsd_3k']:.2f}. &nbsp;
dmap torsion &lambda;&#8321;..&#8323; = {[round(l, 4) for l in rep3_dm['torsion']['lams'][1:4]]},
ca_drmsd &lambda;&#8321;..&#8323; = {[round(l, 4) for l in rep3_dm['ca_drmsd']['lams'][1:4]]}.</p>
<h3>Mutual information vs CV1 (bb r&#8320;10 &#946;3)</h3>
<table><tr><th>CV2</th><th>MI (nats)</th><th>excess</th></tr>{rep3_mi_rows}</table>
<h3>Axis stability on the 10k bank</h3>
<table><tr><th>axis</th><th>median |cos|</th><th>min / pc2</th></tr>{rep3_stab_rows}</table>
<h3>GBn2 energy decomposition (10,019 single-points, shared context)</h3>
<table><tr><th>term</th><th>median (kJ/mol)</th><th>&rho;(CV1)</th><th>&rho;(CV2)</th></tr>{rep3_energy_rows}</table>
<h3>Folded state on rep3 (from section 2.2)</h3>
<p>{rep3_folded['native_envelope_seeds']} seeds (0.31%) inside the native envelope; best RMSD
{rep3_folded['best_rmsd_A']:.2f} &#197;; KDE on plain PC2 gives {len(rep3_kde['states'])} macro-states at
modes {[round(m, 2) for m in rep3_kde['modes']]}.</p>
{fig("cv_rep3_histograms.png", "rep3: 30 head CV distributions.")}
{fig("cv_rep3_joint_maps.png", "rep3: CV1xCV2 joint maps (windows-feasibility); conditional spread intact.")}
{fig("cv_rep3_advanced.png", "rep3 advanced battery: KDE states, circular harmonics, dmap spectra, MI, stability, decomposition.")}
{fig("cv_rep3_energy_vs_cv2.png", "rep3: group-corrected decomposition vs plain PC2.")}

<h2 id="energy-terms">2.4 GBn2 energy-term behavior across the FES maps (rep3)</h2>
<dl class="did">
<dt>What we did:</dt>
<dd>for every seed the full GBn2 decomposition (bond / angle / torsion strain, direct nonbonded,
GB solvation, total) was overlaid onto the landscape maps as per-bin mean-value maps (bins &ge; 20);
trend curves with SEM and bootstrap slope CIs along CV1 / CV2 / Rg / E2E; a compensation analysis
(cross-term correlation; variance compensation coefficient 1 &minus; Var(total)/&Sigma;Var(parts));
the 3 macro-states' composition; and the 31-seed native-envelope subset vs the rest with
permutation-tested mean differences.</dd>
<dt>What came out:</dt>
<dd>Compaction is a packing-vs-solvation trade (nonbonded &minus;140, GB solvation +95 kJ/mol per
contact-CV unit, total only &minus;20). CV2's register flip is energetically flat on every term
(|slope| &le; 3 kJ/mol/unit) — the orthogonal motion costs nothing. Near-native seeds carry mostly
<b>strain</b> costs (+1 bond, +6 angle, +6 torsion kJ/mol; permutation p &le; 0.036) partially
canceled by nonbonded packing (&minus;45, p = 0.06), not significant in total &mdash; on this bank
the folded register's signature is architectural rigidity, not an energy well.</dd>
</dl>
<h3>Trend slopes per term (kJ/mol per coordinate unit)</h3>
<table><tr><th></th><th>bond</th><th>angle</th><th>torsion</th><th>nonbonded</th><th>gb_solvation</th><th>total</th></tr>
{trend_rows}</table>
<h3>Near-native ({nnv['n']} seeds, native envelope) vs rest</h3>
<table><tr><th>term</th><th>mean difference (kJ/mol)</th><th>permutation p</th></tr>{nn_rows}</table>
<p>Partial-term correlation (nonbonded &times; gb_solvation): <b>{ent['corr_partial_terms']['nonbonded']['gb_solvation']:+.3f}</b>;
variance compensation coefficient: <b>{ent['compensation_coeff']:.3f}</b>.</p>
{fig("cv_rep3_energy_term_maps.png", "Per-term mean-value maps over three coordinate pairs with bank pseudo-FES contours overlaid: where strain, packing, and solvation each pay.")}
{fig("cv_rep3_energy_term_trends.png", "Per-term trend curves along CV1 / CV2 / Rg / E2E with SEM bands.")}
{fig("cv_rep3_energy_compensation.png", "Compensation analysis: partial-term correlation matrix, nonbonded vs GB scatter, per-state composition, and the near-native energy signature with permutation stars.")}

<div class="chapter"><p class="chn">Chapter 3</p><h2>The universal tuning bundle (rep4)</h2></div>

<h2 id="rep4">3.1 rep4: the six universal tuning mechanisms — implemented, run, compared</h2>
<dl class="did">
<dt>What we did:</dt>
<dd>implemented all six universal, ab-initio tunings from the design spec
(<code>docs/superpowers/specs/2026-09-09-genpept-universal-coverage-tunings-design.md</code>)
in GENPEPT.py, validated the parts at unit level, tiled the problem-size bank with the same
scaled parameters as rep3 (800k proposals, seed 81806, bundle fully on), and compared against rep3
(control = legacy contact bias 0.15) via the campaign's folded-state and CV battery.
Three production lessons worth recording: <b>(i)</b> the pipeline hung twice at stage boundaries
(pool deadlock, 0 % CPU; <code>--resume</code> unsticks deterministically both times, non-deterministic
rather than input-tied); <b>(ii)</b> one schema-rebuild in candidate materialization dropped the new
tt_source column until fixed — the mechanism physically applied, only its provenance trace went silent;
<b>(iii)</b> T4's adaptive N–O threshold detected no valley on this bank and fell back to 3.8 Å,
honest rather than fabricated.</dd>
<dt>What came out:</dt>
<dd>See tables. In one line: the bundle fixes the process pathologies it targeted
(acceptance, mirror, SASA tail, dense steerability) but <b>does not move the folded-state yield</b> —
the register-envelope occupancy is 0.31% both ways, and the best RMSD unchanged. That localizes the
folded-basin deficit to the <em>minimizer's reach</em>, not to proposal diversity.</dd>
</dl>
<h3>Mechanism-by-mechanism outcome on rep4 (vs rep3 control)</h3>
<table><tr><th>mechanism</th><th>rep4 outcome</th><th>rep3 control</th><th>verdict</th></tr>
<tr><td>T1 coverage uniformization</td><td>acceptance uplift recorded at smoke scale (59.9% vs 6.9% contact-bias baseline); full provenance in coverage_uniformization.json</td><td>—</td><td>works; physically lifts proposal diversity</td></tr>
<tr><td>T2 basin-driven parents</td><td>295 parents across 110 geometric basins via n^0.25 medoid allocation (cap 6/basin)</td><td>240 raw-diversity parents (r7 default)</td><td>works; rare basins funded</td></tr>
<tr><td>T3 turn-type tiling</td><td>94.5% of candidates carry ≥1 universal-type overlay (all 5 type windows active; physical incorporation verified through column bugfix)</td><td>stratified pool, no type control</td><td>works; visibly shifts turn mixture</td></tr>
<tr><td>T4 adaptive register threshold</td><td>no valley found in 86,400 backbone N–O distances &rarr; honest fallback 3.8 &#197;; register_fraction recorded per survivor (mean 0.095)</td><td>fixed 3.5 &#197; counter (degenerate median 0)</td><td>works as designed; no fabrication, no false valley</td></tr>
<tr><td>T5 mirror balance</td><td>frac(+) 0.502 in final survivors (tolerance 0.2; 0 swaps needed)</td><td>0.540 (rep3)</td><td>works; mirror fold is now a governed channel</td></tr>
<tr><td>T6 SASA lower tail</td><td>final q05 SASA 12.30 &le; pool 12.45 &times; 1.1 (0 swaps needed)</td><td>12.29 (rep3)</td><td>works; tail preserved by constraint</td></tr></table>
<h3>Headline metrics (10k scale, all-bundle vs legacy contact bias)</h3>
<table><tr><th>metric</th><th>rep3 (legacy)</th><th>rep4 (bundle)</th></tr>
<tr><td>native-envelope seeds</td><td>31 (0.31%)</td><td>31 (0.32%)</td></tr>
<tr><td>best CA-RMSD to 1UAO</td><td>1.01 &#197;</td><td>0.79 &#197;</td></tr>
<tr><td>champion CV entropy bb/ca/heavy</td><td>0.978 / 0.979 / 0.939</td><td>0.972 / 0.973 / <b>0.953</b></td></tr>
<tr><td>windows @ k=1200 (bb/ca/heavy)</td><td>26 / 27 / 23</td><td>26 / 27 / 23</td></tr>
<tr><td>median SASA (bank composition)</td><td>13.65 nm&#178;</td><td>13.80 nm&#178;</td></tr>
<tr><td>funnel (candidates &rarr; implicit &rarr; BH &rarr; final)</td><td>12000 &rarr; 11981 &rarr; 1192 &rarr; 10019</td><td>12000 &rarr; 11935 &rarr; 1132 &rarr; 9745</td></tr></table>
<p><b>Interpretation:</b> CT over hypothesis: the folded-basin deficit is not in proposal-space coverage —
it is in what the implicit minimizer does or does not stabilize (native-tight packing is under-reachable
by GBn2-min basin). The correct next lever is on the expansion/relaxation side (e.g. guided BH/NMA around
near-native minima per spec T2's proven behavior), not a bigger search pool.</p>
{fig("cv_rep4_vs_rep3.png", "rep4 (full bundle) vs rep3 (legacy contact bias) at matched scale: CVs, SASA, mirror, RMSD histograms.")}

<h2 id="genpept-now">3.2 What GENPEPT is doing now — stage by stage</h2>
<p>The one-glance answer to "what does the generator actually do today", with the production
numbers of <code>chignolin_genpept_rep4_tuned</code> (seed 81806, full T1&ndash;T6 bundle, ~3 h on an
RTX 3060 Ti). Every number below was re-derived from raw bank artifacts during the audit
(<code>ADVERSARIAL_VERIFICATION.md</code>).</p>

{fig("genpept_pipeline_now.png", "The current pipeline. Blue: stages. Orange: the six universal, sequence-agnostic tunings added this campaign, each wired at the point shown. Green: production funnel counts (800k proposals &rarr; 12,000 candidates &rarr; 11,935 implicit minima &rarr; 1,132 BH minima (+9,536 NMA minima) &rarr; 9,745 final survivors).")}

<h3>Stage by stage</h3>
<table>
<tr><th>#</th><th>stage</th><th>what it does now</th><th>measured outcome (rep4)</th></tr>
<tr><td>1</td><td>stratified Ramachandran proposals</td><td>low-discrepancy (φ,ψ) state and angle draws; <b>T3</b>: 35% of consecutive residue pairs redrawn from universal β-turn windows (5 types, per-position compatibility, per-row provenance)</td><td>94.5% of candidates carry ≥1 turn-type overlay (measured on post-fix smoke-2)</td></tr>
<tr><td>2</td><td>acceptance</td><td><b>T1</b>: Metropolis weight toward sparse (Rg, e2e, contact-count) bins of a startup-calibrated density table, frozen after calibration (deterministic; mutually exclusive with the legacy contact bias)</td><td>60.0% acceptance (480,112/800,000) vs 7.0% under legacy bias</td></tr>
<tr><td>3</td><td>preselection</td><td>clash filter + per-bin quota (Rg/E2E/contacts) keeps shape-unique conformers</td><td>12,000 candidate seeds</td></tr>
<tr><td>4</td><td>implicit minimization</td><td>GBn2 (amber14), adaptive chunked rounds with force checks</td><td>11,935 minima (99.6%)</td></tr>
<tr><td>5</td><td>basin clustering</td><td>geometric basins over the viable-minimum table</td><td>110 BH-parent basins</td></tr>
<tr><td>6</td><td>parent allocation</td><td><b>T2</b>: parents ∝ n_basin^0.25, medoid-per-basin, cap 6 &mdash; rare basins get expansion budget instead of being starved</td><td>295 parents / 110 basins (2–6 each)</td></tr>
<tr><td>7</td><td>basin hopping</td><td>675 K MD kicks + re-minimization from parents</td><td>1,132 BH minima</td></tr>
<tr><td>8</td><td>NMA expansion</td><td>2 modes × {{0.75, 1.35 Å}} × both signs, then re-minimize</td><td>9,536 NMA minima</td></tr>
<tr><td>9</td><td>adaptive exploration</td><td>PCA-frontier loop: 5 rounds × 200k proposals, keep 800</td><td>expansion pool feeding final picks</td></tr>
<tr><td>10</td><td>final selection</td><td><b>T4</b>: register threshold from the bank's own N–O histogram (honest fallback 3.8 Å); <b>T5</b>: mirror-balance stratification (tolerance 0.2); <b>T6</b>: SASA lower-tail preservation (q05 ≤ pool × 1.1)</td><td>9,745 survivors; frac(+) 0.502 with 0 swaps; q05 12.30 ≤ 12.45×1.1; register_fraction mean 0.095</td></tr>
</table>

<h3>What the bundle changed (and did not change)</h3>
<ul>
<li>Fixed as designed: proposal coverage (T1), rare-basin funding (T2), turn-type mixture (T3),
register definition honesty (T4), handedness governance (T5: 0.540 &rarr; 0.502), tail preservation (T6),
and heavy-atom CV entropy (0.939 &rarr; 0.953).</li>
<li>Unchanged: folded-state yield &mdash; native-envelope 31 seeds (0.31% &rarr; 0.32%), best RMSD 1.01 &rarr; 0.79 Å.
The fold-basin deficit is in what the GBn2 minimizer stabilizes, not in what the generator proposes
(see section 4.3 for the direct test of that diagnosis).</li>
</ul>

<h3>Standing caveats</h3>
<ul>
<li>The pipeline deadlocked twice at stage boundaries in production; <code>--resume</code> unsticks it
deterministically. A real scheduling bug, input-independent.</li>
<li>Energies are GBn2 minimized values &mdash; solvent-model-dependent at the ~1 kcal level
(section 4.2); multiplicity g counts viable minimization outcomes, a proxy for basin volume.</li>
</ul>

<h2 id="rep5">3.3 rep5: T7 restraint-forming kicks — first movement of the folded-state metric</h2>
<dl class="did">
<dt>What we did:</dt>
<dd>implemented the T7 candidate the guided-expansion negative result pointed at (section 4.3):
flat-bottom N&ndash;O walls (3.8 &Aring;, 1000 kJ/mol/nm&sup2;, capture 6 &Aring;, any |i&minus;j|&ge;3 pair,
sequence-agnostic) active <b>only during the 675 K BH kick</b> via a global force parameter, zeroed
before re-minimization. Clean A/B: seed 81806, same T1&ndash;T6 bundle, only the flag differs
(<code>chignolin_genpept_rep5_t7.yaml</code>; full detail in <code>REP5_T7.md</code>).</dd>
<dt>What came out:</dt>
<dd>T7 does what it was designed for &mdash; and it is the <b>first mechanism in the campaign that
moves the folded-state metric</b>: native-envelope 31 &rarr; 39 seeds (0.32% &rarr; 0.39%),
registers &ge;3 28.1% &rarr; 35.2%, registers &ge;5 5.8% &rarr; 9.5% (n &asymp; 9.8k). The near-native
RMSD tail is flat (best 1.15 &Aring;, frac &lt;2.5 &Aring; 5.45% &rarr; 5.11%): the new registers include
non-native pairings &mdash; T7 builds the H-bond skeleton, the minimizer still picks the topology.</dd>
<dt>Consequence:</dt>
<dd>the minimizer-reach diagnosis is confirmed and partially corrected. The remaining deficit sits
one level deeper: forming non-native registers is as easy as forming native ones. Next levers:
native-register selection at the final stage (envelope-consistent rewards, not BH), or the solvent
model itself (GBn2's native/misfolded near-degeneracy, section 4.2).</dd>
</dl>
<h3>rep4 vs rep5 headline battery (identical seed, only T7 differs)</h3>
<table><tr><th>metric</th><th>rep4 (T1&ndash;T6)</th><th>rep5 (+T7)</th></tr>
<tr><td>survivors</td><td>9,745</td><td>9,886</td></tr>
<tr><td>native-envelope seeds</td><td>31 (0.32%)</td><td><b>39 (0.39%)</b></td></tr>
<tr><td>registers &ge; 3 / &ge; 5</td><td>28.1% / 5.8%</td><td><b>35.2% / 9.5%</b></td></tr>
<tr><td>best CA-RMSD to 1UAO</td><td>0.79 &#197;</td><td>1.15 &#197;</td></tr>
<tr><td>frac RMSD &lt; 2.5 &#197;</td><td>5.45%</td><td>5.11%</td></tr>
<tr><td>parents capturing restraint pairs</td><td>&mdash;</td><td>97.0% (mean 7.2, max 22)</td></tr>
<tr><td>wall time (RTX 3060 Ti, 16 jobs)</td><td>~3 h</td><td>2 h 38 min</td></tr></table>
<p><b>Honest statistics:</b> the register enrichment is decisive at this n; the envelope gain
(+26% relative) is ~1.4&sigma; on Poisson noise alone &mdash; supportive, not standalone-proof.
Cost: none measurable. Production gotcha recorded: GENPEPT resolves a relative <code>out:</code>
against the config file's directory, not the cwd.</p>

<div class="chapter"><p class="chn">Chapter 4</p><h2>The thermodynamic layer on the bank</h2></div>

<p>Everything in this chapter is <b>post-processing on the frozen rep4 bank</b> — no pipeline change.
The bank is a search-density instrument; these analyses turn it into a thermodynamic starting point
for production MD (window initialization, bias sanity), with the systematics bounded rather than
assumed away.</p>

{fig("genpept_thermo_layer.png", "The thermodynamic measurement layer: the recorded T1 weights de-bias the search density, Boltzmann weights and basin multiplicities (&minus;kT·ln g) turn it into a starting FES, and a consensus second solvent model bounds the systematic.")}

<h2 id="thermo-fes">4.1 From search density to a starting FES (basin level)</h2>
<dl class="did">
<dt>What we did:</dt>
<dd>three weightings of the same 9,745 survivors (raw / de-biased by the recorded T1 acceptance
table / thermodynamic with Boltzmann + basin multiplicity), all at T = 300 K
(<code>genpept_rep4_thermo.py</code> &rarr; <code>THERMO_STARTING_FES.md</code>).</dd>
<dt>What came out:</dt>
<dd>the Boltzmann step, not the de-biasing, reshapes the FES (raw vs thermo Spearman
<b>0.675</b>; raw vs de-biased 0.893). F(Rg) shifts: compact bins +14.4, mid +41.5, extended
+69.4 kcal/mol. Basin multiplicity matters: 9,745 basins over 22,378 viable minima, kT&middot;ln g
median 1.7 / p95 4.5 kcal/mol &mdash; comparable to the whole-bank energy spread; top-20 basins by
mean-E vs by F overlap only 11/20.</dd>
</dl>
{fig("cv_rep4_thermo_fes.png", "Basin-level FES under the three weightings; the thermodynamic map is the one to initialize windows from.")}
{fig("cv_rep4_thermo_basins.png", "F(Rg) shift per bin and the basin-multiplicity term kT·ln g.")}

<h2 id="consensus">4.2 Consensus solvent-model check (GBn2 vs OBC2)</h2>
<dl class="did">
<dt>What we did:</dt>
<dd>all survivors re-scored under a second implicit model (OBC2) single-point
(<code>genpept_rep4_consensus.py</code> &rarr; <code>THERMO_CONSENSUS_MODEL.md</code>).</dd>
<dt>What came out:</dt>
<dd>cross-model Spearman <b>0.986</b>, model gap std <b>1.0 kcal/mol</b> &mdash; the size of the
systematic on every single-model energy claim in this report. Directional preferences agree
(curl: +0.121 vs +0.143; compaction: &minus;0.182 vs &minus;0.153); 18.3% of survivors are
stable-quintile in both models, 1.7% only in each &mdash; treat the union-stable set as
solvent-robust, the rest as model-sensitive.</dd>
</dl>
{fig("cv_rep4_consensus_model.png", "GBn2 vs OBC2: energy scatter, preference stability, and the consensus-stable fraction.")}

<h2 id="guided">4.3 Guided register-rewarded expansion — an honest negative</h2>
<dl class="did">
<dt>What we did:</dt>
<dd>the expansion-side lever the rep4 verdict pointed at, tested without touching GENPEPT.py:
30 best-RMSD survivors, 8 BH steps each (675 K kick + GBn2 re-minimization), Metropolis on
F = E &minus; &lambda;&middot;n_register with &lambda; = 1.5 kcal/register vs &lambda; = 0 control
(<code>genpept_rep4_guided_expansion.py</code> &rarr; <code>GUIDED_EXPANSION.md</code>).</dd>
<dt>What came out:</dt>
<dd><b>nothing moved</b> &mdash; both arms degrade mean RMSD 1.58 &rarr; 1.99 Å, register counts
stay 1.8 &rarr; 1.8, envelope fraction 2/30 in both. Mechanism: selection bias cannot create what
the move set never visits; the 675 K kick + minimization almost never forms new backbone N&ndash;O
registers (the minimizer relaxes away from them), and near-native seeds are not register-rich (mean 1.8).</dd>
<dt>Consequence:</dt>
<dd>the correct lever is restraint-<i>forming</i> dynamics, not selection: flat-bottom N&ndash;O
restraints during the kick itself, then release + minimize. That is a GENPEPT.py change &mdash; a
<b>T7 candidate</b> for the design-spec process.</dd>
</dl>
{fig("cv_rep4_guided_expansion.png", "Control vs register-rewarded arms: RMSD, register count, envelope fraction — indistinguishable.")}

<h2 id="fes-starting">4.4 Starting FES on the working CV axes (CV1 × PC1)</h2>
<dl class="did">
<dt>What we did:</dt>
<dd>the production-usable map: w = w_CU<sup>&minus;1</sup> &times; exp(&minus;(E&minus;E_min)/kT),
per-seed GBn2 energies matched by basename, on CV1 (bb r₀=10 β3) &times; torsion PC1
(<code>genpept_rep4_fes_starting.py</code> &rarr; <code>FES_STARTING.md</code>).</dd>
<dt>What came out:</dt>
<dd>thermo depth <b>21.0 kcal/mol</b> (raw 1.7); raw-vs-thermo Spearman 0.526; 1-D shifts along
CV1 &minus;0.3 &rarr; +19.6 kcal/mol. The 31 native-envelope seeds sit at <b>~7.6 kcal</b> on the
thermo F(CV1) &mdash; the folded register is not a low-free-energy basin under GBn2, the
landscape-level form of the minimizer-reach diagnosis. Sanity: top-10% thermo weight vs
top-energy-quintile agreement 0.67.</dd>
</dl>
{fig("cv_rep4_fes_starting.png", "CV1 × PC1 under the three weightings; red stars = native-envelope seeds.")}
{fig("cv_rep4_fes_starting_1d.png", "1-D F(CV1) raw vs thermo, and the per-bin shift.")}

<h2 id="fes-heavy">4.5 Starting FES on heavy contacts × Rg</h2>
<dl class="did">
<dt>What we did:</dt>
<dd>the same weighting on the swarm-side contact parameterization (all-heavy, r₀=12 Å, β=3) against
heavy-atom Rg (<code>genpept_rep4_fes_heavy_rg.py</code> &rarr; <code>FES_HEAVY_RG.md</code>).</dd>
<dt>What came out:</dt>
<dd>thermo minimum at heavy CV1 <b>0.78</b>, Rg <b>5.78 Å</b> &mdash; compact, contact-rich
conformers; depths 22.0 / 20.4 kcal along the two marginals (raw 1.2); raw-vs-thermo Spearman 0.691.
The native-envelope seeds (median Rg 5.68 Å, median heavy CV1 0.86) sit <b>in the compact,
contact-rich corner</b> near the thermo minimum &mdash; these axes place the fold better than
CV1 &times;PC1 does. Use this map to seed GaREUS windows on the contact CV.</dd>
</dl>
{fig("cv_rep4_fes_heavy_rg.png", "Heavy contacts × Rg under the three weightings; red stars = native-envelope seeds.")}
{fig("cv_rep4_fes_heavy_rg_1d.png", "1-D profiles along heavy CV1 and Rg, raw vs thermo.")}

<h2 id="closest">4.6 The closest bank seeds vs folded chignolin</h2>
<dl class="did">
<dt>What we did:</dt>
<dd>top 6 survivors by min CA-RMSD to the 18-model 1UAO ensemble, each CA-aligned onto its closest
native model and overlaid fully atomistically (native heavy atoms red, seed blue at 45% transparency)
(<code>genpept_rep4_closest_native_overlay.py</code> &rarr; <code>CLOSEST_SEEDS.md</code>).</dd>
<dt>What came out:</dt>
<dd>best <b>0.79 Å</b> (model 9), then 1.14 / 1.18 / 1.24 / 1.29 / 1.36 Å &mdash; near-native
hairpin geometry is <i>in</i> the bank; what the envelope census showed is that it is a thin tail
(0.32%), not a stabilized basin.</dd>
</dl>
{fig("cv_rep4_closest_to_native.png", "Top-6 closest seeds (blue, transparent) overlaid on their closest native 1UAO model (red). Panel captions give the CA-RMSD and model.")}

<h2 id="fes-rep5">4.7 Starting FES on the rep5 (T7) bank</h2>
<dl class="did">
<dt>What we did:</dt>
<dd>the section 4.5 recipe on the T7 bank: each bank's own recorded T1 weights and GBn2
energies, heavy contacts (r₀=12, β3) × heavy Rg, plus a direct rep4-vs-rep5 comparison
(<code>genpept_rep5_fes_starting.py</code> &rarr; <code>FES_STARTING_REP5.md</code>).</dd>
<dt>What came out:</dt>
<dd>the T7 bank's thermodynamic landscape is <b>more compact and envelope-friendly</b>:
thermo minimum at heavy CV1 0.77 / Rg 5.69 &Aring; (rep4: 0.78 / 5.78), and the 39
native-envelope seeds now reach <b>4.1 kcal</b> on the thermo map — materially closer to
the minimum than the rep4 envelope position (median Rg 5.76 &Aring;, median heavy CV1 0.84,
inside the compact/contact-rich corner). Map agreement raw vs thermo Spearman 0.717
(rep4: 0.691). The bin-wise rep5&minus;rep4 difference spans &plusmn;15 kcal in the sparse
tails (sampling noise there) but is small in the densely covered region.</dd>
</dl>
{fig("cv_rep5_fes_heavy_rg.png", "rep5 (T7) starting FES on heavy contacts × Rg under the three weightings; red stars = the 39 native-envelope seeds.")}
{fig("cv_rep5_fes_vs_rep4.png", "Left/middle: 1-D thermo profiles, rep4 vs rep5. Right: bin-wise thermo-FES difference (rep5 − rep4); red stars = envelope seeds. Trust only the densely covered region.")}

<div class="chapter"><p class="chn">Chapter 5</p><h2>Synthesis</h2></div>

<h2 id="residual-context">5.1 The residual-to-contacts motion in context — mechanism &amp; thermodynamics</h2>
<dl class="did">
<dt>What we did:</dt>
<dd>For the fixed-compaction band (|CV1 &minus; 0.481| &le; 0.02; 117 seeds, 24 sampled across CV2):
GBn2 single-point energy decomposition with OpenMM (bond / angle / torsion / direct nonbonded /
GB-solvation; anchor to generator energies within 0.05 kJ/mol); DSSP secondary structure along the
axis; solvation descriptor trends; and a literature check (NCBI E-utilities, abstracts quoted from
PubMed).</dd>
<dt>What came out:</dt>
<dd>The total implicit energy is <b>flat</b> along the axis (&rho; = &minus;0.05) while components
trade strongly &mdash; bond strain worsens (&rho; = +0.38) as GB solvation improves (&rho; = &minus;0.37)
&mdash; the same energy near-degeneracy the literature reports between native and misfolded chignolin
states. Structure stays all-coil elsewhere, but the curled endpoint nucleates helix-like turns in
residues 3&ndash;6; no DSSP strand anywhere on the bank. The curl mildly <i>unpacks</i> the Tyr2&ndash;Trp9
aromatics and buries Asp3.</dd>
</dl>
<h3>Thermodynamic reading</h3>
<p>The curl is a <b>compensated motion</b>: enthalpy flows between intramolecular strain and solvation
at ~constant total, so it is invisible to total-energy ranking &mdash; only a dedicated orthogonal CV
shows it. Its bank-side preference (&rho; = &minus;0.31 on minimized GBn2 energies) is partly a
generator filter (GB solvation favors compact/ionic arrangements), not equilibrium physics.</p>
<h3>Literature context (fetched and quoted from PubMed abstracts)</h3>
<table><tr><th>source</th><th>finding</th><th>bearing on this CV</th></tr>
<tr><td>Enemark &amp; Rajagopalan, PCCP 2012 (doi:10.1039/c2cp40285h)</td>
<td>chignolin folds turn-first ("broken zipper"): turn nucleation &rarr; turn growth &rarr; H-bonds &rarr; hydrophobic packing last; cross-strand packing can be rate-limiting and creates misfolded states</td>
<td>CV2 covers the coordinating DOF the packing-dependent CV1 cannot</td></tr>
<tr><td>Enemark, Kurniawan &amp; Rajagopalan, Sci Rep 2012 (doi:10.1038/srep00649)</td>
<td>staged C-terminal roll-up in pre-folding</td>
<td>curled flank endpoints resemble early roll-up intermediates</td></tr>
<tr><td>Sobieraj &amp; Setny, JCTC 2022 (doi:10.1021/acs.jctc.1c00945)</td>
<td>on CLN025 (identical sequence GYDPETGTWG): Granger causality shows turn-region rearrangements drive folding/unfolding; arm&ndash;arm interactions score low</td>
<td>the causal folding DOF lives in CV2; CV1's arm contacts are the secondary axis &mdash; exactly our split</td></tr>
<tr><td>Maruyama &amp; Mitsutake, JPCB 2018 (doi:10.1021/acs.jpcb.8b00288)</td>
<td>native &asymp; misfolded total energies with different components; Thr6&ndash;Thr8 interaction important for the &pi;-turn</td>
<td>matches the flat-total/compensated-components decomposition along CV2, and the T6/T8 &psi;-flips in its loadings</td></tr>
<tr><td>Amado et al., JPCB 2024 (doi:10.1021/acs.jpcb.3c08271)</td>
<td>Y2&ndash;W9 aromatics and Y2&ndash;P4 CH&ndash;&pi; stabilize the hairpin; refolding 1.15 &mu;s</td>
<td>the curl's mild aromatic unpacking exchanges this organizing interaction for turn pre-organization</td></tr>
</table>
{fig("cv2_thermo_dssp_energy.png", "4-panel: DSSP fractions vs CV2 bin; per-residue strand content at deciles; GBn2 component decomposition at fixed CV1; solvation descriptor trends.")}

<h2 id="rec">5.2 Final recommendation</h2>
<pre>contact_atom_selection: backbone-heavy   # GAREUS mode: N, CA, C, O only
contact_scheme: atom-pairs                 # ≡ residue-balanced for bb; pick one
contact_min_sequence_separation: 4
contact_normalize: true
contact_r0_a: 10.0                         # plateau center; 8–12 all fine
contact_beta_a_inv: 3.0                    # insensitive 1.5–6
secondary_cv: torsion-PCA (plain, component 2 of the bank PCA), eigenvector PERSISTED
             with the run (bank-torsion_pca_cv2.json). Diversity coordinate; for a
             slow-motion CV2 use frozen within-segment tICA. Do NOT re-fit from a moved bank.
# former candidate: residual-vs-CV1 torsion PC1 — demoted: eigenvalue-degenerate scalar
# (resamples collapse |cos| to 0.02; generator-split 0.75 vs plain PC2's 0.965).</pre>
<div class="bad">Do not use without re-checking: legacy (r&#8320; 4.5, β 6) — degenerate (≤ 2 windows);
residue-balanced as a "different" CV — identical here; PLUMED rational switches — worse than logistic;
H-bond counters, strand min-distance, cis-ω, left-helical counts — degenerate on an implicit bank;
Rg + a contact CV1 as a 2-D pair — the same coordinate (|r| ≥ 0.94).</div>
<p>Caveats: bank-side measurements; the dynamics-side equivalents must be re-measured on the swarm
(as for heavy r&#8320;12 in <code>HEAVY_CONTACTS_PARAM_TEST_CHIGNOLIN_GENPEPT.txt</code>). nw at k = 1200 uses
normalized CV units; Å-unit values in the tables are flagged in the text report.</p>

<h2 id="files">5.3 Files, reproduction, limitations</h2>
<table><tr><th>path</th><th>content</th></tr>
<tr><td>GENPEPT_R7_CV_CENSUS_REPORT/genpept_r7_cv_census_values.csv</td><td>1,970 × 298 per-seed CV values</td></tr>
<tr><td>.../genpept_r7_cv_census_stats.csv</td><td>298-row per-CV metric table</td></tr>
<tr><td>.../genpept_r7_cv_census_corr_pearson.csv</td><td>298 × 298 correlation matrix</td></tr>
<tr><td>.../report_tables.txt</td><td>exhaustive tables</td></tr>
<tr><td>.../README.md</td><td>previous summary report</td></tr>
<tr><td>.../CV2_DEEPDIVE.md, DIFFMAP.md, FOURIER.md, TURN_AXIS.md, ORTHOGONAL_PAIRS.md, RESIDUAL_THERM.md, DYNAMICS_VALIDATION.md, STEERABILITY_STABILITY.md, MI_STATES_ENERGY.md, RIGOR_HANDOFF.md, PLAIN_PC2_CV2.md, REP2_COMPARISON.md</td><td>per-analysis notes</td></tr>
<tr><td>chignolin_genpept_rep4_tuned/, REP4_TUNED_COMPARISON.md, genpept_rep4_vs_rep3.json</td><td>bundle-implemented bank + comparison</td></tr>
<tr><td>chignolin_genpept_rep5_t7/, REP5_T7.md</td><td>T7 restraint-forming-kick bank (clean A/B vs rep4, same seed) + verdict</td></tr>
<tr><td>ADVERSARIAL_VERIFICATION.md</td><td>bitwise re-derivation audit of the rep4 comparison battery, GAREUS oracle, determinism sweep</td></tr>
<tr><td>THERMO_STARTING_FES.md, genpept_rep4_thermo.json</td><td>basin-level thermodynamic starting FES (three weightings, multiplicity term)</td></tr>
<tr><td>THERMO_CONSENSUS_MODEL.md, genpept_rep4_consensus.json</td><td>GBn2 vs OBC2 consensus check on all survivors</td></tr>
<tr><td>GUIDED_EXPANSION.md, genpept_rep4_guided_expansion.json</td><td>register-rewarded BH expansion demo (negative result, T7 candidate)</td></tr>
<tr><td>FES_STARTING.md, genpept_rep4_fes_starting.json</td><td>starting FES on CV1 × PC1 (window-init map)</td></tr>
<tr><td>FES_HEAVY_RG.md, genpept_rep4_fes_heavy_rg.json</td><td>starting FES on heavy contacts × Rg</td></tr>
<tr><td>FES_STARTING_REP5.md, genpept_rep5_fes_starting.json</td><td>starting FES on the rep5 (T7) bank + rep4-vs-rep5 comparison</td></tr>
<tr><td>CLOSEST_SEEDS.md, genpept_rep4_closest_to_native.json</td><td>top-6 near-native seed overlays vs 1UAO</td></tr>
<tr><td>chignolin_genpept_rep2/, genpept_r7_vs_rep2.json</td><td>independent second bank + reproducibility numbers</td></tr>
<tr><td>chignolin_genpept_rep3_massive/, REP3_FES_FOLDED.md, REP3_MASSIVE.md, genpept_rep3_fes_folded.json</td><td>10k seed bank + folded-proximity audit + pseudo-FES</td></tr>
<tr><td>REP3_STATS.md, REP3_ADVANCED.md, genpept_rep3_cv_census_{{values,stats}}.csv, genpept_rep3_orthogonal_pairs.csv, genpept_rep3_energy_components.csv, genpept_rep3_advanced.json</td><td>full re-analysis on the 10k bank</td></tr>
<tr><td>REP3_ENERGY_TERMS.md, genpept_rep3_energy_terms.json</td><td>energy-term-by-term landscape analysis</td></tr>
<tr><td>.../genpept_r7_energy_components.csv, seed_bank_tiling_k16.csv</td><td>full-bank decomposition; k-center seed tiling</td></tr>
<tr><td>.../genpept_r7_residual_therm.json</td><td>GBn2 decomposition, DSSP and descriptor correlations</td></tr>
<tr><td>.../genpept_r7_orthogonal_pairs.{{csv,json}}</td><td>per-pair scores for all 60 combinations</td></tr>
<tr><td>.../genpept_r7_diffmap_{{json,psi.csv}}, genpept_r7_fourier.json, bank_torsion_pca_cv2.json</td><td>machine-readable outputs</td></tr>
<tr><td>.../figures/* (& index.html)</td><td>all figures + one-page gallery</td></tr>
<tr><td>.../scripts/genpept_r7_*.py</td><td>the six deterministic generator scripts</td></tr>
</table>
<pre># reproduction (from repo root)
python GENPEPT_R7_CV_CENSUS_REPORT/scripts/genpept_r7_cv_census.py      # 298 CVs, ~17 s
python GENPEPT_R7_CV_CENSUS_REPORT/scripts/genpept_r7_cv_analyze.py     # stats/tables/figs, ~6 s
python GENPEPT_R7_CV_CENSUS_REPORT/scripts/genpept_r7_cv2_deepdive.py   # CV2 + dynamics, ~5 s
python GENPEPT_R7_CV_CENSUS_REPORT/scripts/genpept_r7v2_figures.py      # v2 map family, ~21 s
python GENPEPT_R7_CV_CENSUS_REPORT/scripts/genpept_r7v2_morph.py        # morph mp4/pdb/json, ~22 s
python GENPEPT_R7_CV_CENSUS_REPORT/scripts/genpept_r7_diffmap.py        # dmaps + TwoNN + LASSO, ~11 min
python GENPEPT_R7_CV_CENSUS_REPORT/scripts/genpept_r7_fourier.py        # Fourier triple, ~9 s
python GENPEPT_R7_CV_CENSUS_REPORT/scripts/genpept_r7_turn_axis.py      # turn axis, ~4 s
python GENPEPT_R7_CV_CENSUS_REPORT/scripts/genpept_r7_pairs.py          # orthogonal pairs, ~4 s
python GENPEPT_R7_CV_CENSUS_REPORT/scripts/genpept_r7_residual_thermo.py # DSSP + energies + context, ~11 s
python GENPEPT_R7_CV_CENSUS_REPORT/scripts/genpept_r7_dynamics_cv.py     # dynamics validation, ~3 min (reads replica XTCs)
python GENPEPT_R7_CV_CENSUS_REPORT/scripts/genpept_r7_steer_stability.py # steerability + stability, ~6 s
python GENPEPT_R7_CV_CENSUS_REPORT/scripts/genpept_r7_mi_states_energy.py # MI + states (+10 min first time: full-bank energies), ~9 s cached
python GENPEPT_R7_CV_CENSUS_REPORT/scripts/genpept_r7_rigor.py           # rigor pack, ~5 s
python GENPEPT_R7_CV_CENSUS_REPORT/scripts/genpept_r7_plain_pc2.py       # plain-PC2 workup, ~4 s
# second bank + comparison
python GENPEPT.py --config GENPEPT_R7_CV_CENSUS_REPORT/scripts/chignolin_genpept_rep2.yaml --strict-config --jobs 16 --min-jobs 1 --platform OpenCL  # ~50 min on this box
python GENPEPT_R7_CV_CENSUS_REPORT/scripts/genpept_rep2_compare.py
python GENPEPT_R7_CV_CENSUS_REPORT/scripts/genpept_rep3_fes_folded.py    # 1UAO proximity + FES, ~12 s
python GENPEPT_R7_CV_CENSUS_REPORT/scripts/genpept_rep3_census.py     # 298 CVs on 10k, ~2 min
python GENPEPT_R7_CV_CENSUS_REPORT/scripts/genpept_rep3_stats.py      # rankings/orthogonality/figs, ~10 s
python GENPEPT_R7_CV_CENSUS_REPORT/scripts/genpept_rep3_advanced.py   # dmap/TwoNN/Fourier/MI/energies, ~3 min
python GENPEPT_R7_CV_CENSUS_REPORT/scripts/genpept_rep3_energy_terms.py  # per-term maps/trends/compensation, ~15 s
python GENPEPT.py --config GENPEPT_R7_CV_CENSUS_REPORT/scripts/chignolin_genpept_rep4.yaml --strict-config --resume --jobs 16 --min-jobs 1 --platform OpenCL   # tuned bank
python GENPEPT_R7_CV_CENSUS_REPORT/scripts/genpept_rep4_compare.py    # rep4 vs rep3 battery, ~25 s
# thermodynamic layer (chapter 4), all on the frozen rep4 bank
python GENPEPT_R7_CV_CENSUS_REPORT/scripts/genpept_rep4_thermo.py            # basin FES, ~4 min
python GENPEPT_R7_CV_CENSUS_REPORT/scripts/genpept_rep4_consensus.py         # GBn2 vs OBC2, ~5 min
python GENPEPT_R7_CV_CENSUS_REPORT/scripts/genpept_rep4_guided_expansion.py   # negative-result demo, ~8 min
python GENPEPT_R7_CV_CENSUS_REPORT/scripts/genpept_rep4_fes_starting.py       # CV1 x PC1 FES, ~3 min
python GENPEPT_R7_CV_CENSUS_REPORT/scripts/genpept_rep4_fes_heavy_rg.py       # heavy x Rg FES, ~3 min
python GENPEPT_R7_CV_CENSUS_REPORT/scripts/genpept_rep4_closest_native_overlay.py  # PyMOL overlays, ~4 min
python GENPEPT_R7_CV_CENSUS_REPORT/scripts/genpept_rep5_fes_starting.py       # rep5 FES + rep4 comparison, ~5 min
python GENPEPT_R7_CV_CENSUS_REPORT/scripts/build_cv_report.py                # this report (chapter edition)
# T7 (rep5): restraint-forming kicks, clean A/B vs rep4 at identical seed
python GENPEPT.py --config GENPEPT_R7_CV_CENSUS_REPORT/scripts/chignolin_genpept_rep5_t7.yaml --strict-config --resume --jobs 16 --min-jobs 1 --platform OpenCL   # ~2.6 h
python GENPEPT.py --config GENPEPT_R7_CV_CENSUS_REPORT/scripts/genpept_t7_smoke.yaml --strict-config --resume --jobs 16 --min-jobs 1 --platform OpenCL   # T7 smoke, ~15 min</pre>
<p><b>Limitations:</b> bank-side ensemble (search density, not equilibrium); torsion PCA is bank-specific
(no cross-bank transfer); the residual CV2 is non-diffusive on dynamics; nw values assume normalized CV
units and the k = 1200 ceiling; TwoNN d depends on the chosen metric by construction — quote per-metric.</p>
<p><b>Correction (2026-09-11, adversarial round 3):</b> all CA-RMSD-to-1UAO numbers were previously
computed with a scrambled atom correspondence (seed CA indices applied to the differently-ordered native
topology; <code>md.rmsd</code> without <code>ref_atom_indices</code>). Corrected values are used
throughout this edition: rep3 best 1.01 Å (was 1.16), rep4 best 0.79 Å (was 1.18), rep5 best 1.15 Å
(was 1.30). Comparative verdicts are unchanged (both sides of every A/B were measured the same way);
envelope and register metrics never used RMSD. See <code>ADVERSARIAL_VERIFICATION.md</code> round 3.</p>

</main></body></html>"""

OUT.write_text(html)
print(f"wrote {OUT} ({OUT.stat().st_size/1e6:.1f} MB)")
