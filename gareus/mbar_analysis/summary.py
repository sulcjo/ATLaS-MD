"""Markdown rendering of the already-fully-assembled `s` summary dict
(pmf_summary.md). Relocated verbatim from analyze_gareus_mbar.py (Plan
A6a) -- no logic changes, only module location.
"""
from __future__ import annotations


def _render_health_section_md(s):
    """Markdown lines for the result-health verdict (top of pmf_summary.md)."""
    try:
        from gareus_report import render_verdict_md
        return render_verdict_md(s.get('health'), s.get('warnings_grouped'))
    except Exception:
        return []


def _key_diagnostics_md(s):
    """Surface diagnostics that are otherwise buried in sub-directory files:
    conformational basins, Poincaré recurrence routes. Defensive throughout."""
    lines=[]
    try:
        bt=((s.get('convergence') or {}).get('basin_tracking')) or {}
        if bt.get('enabled') and bt.get('n_basins'):
            _cu=s.get('primary_cv_units','A')
            parts=[]
            for b in (bt.get('basins') or [])[:6]:
                lo=b.get('left_cv_A'); hi=b.get('right_cv_A'); ctr=b.get('center_cv_A')
                if lo is not None and hi is not None:
                    parts.append(f"[{float(lo):.2f}–{float(hi):.2f} {_cu}]")
                elif ctr is not None:
                    parts.append(f"~{float(ctr):.2f} {_cu}")
            det=(': '+', '.join(parts)) if parts else ''
            lines.append(f"- **Basins (CV1):** {int(bt['n_basins'])}{det}")
    except Exception:
        pass
    try:
        pm=s.get('poincare_map') or {}
        if pm.get('available'):
            fr=pm.get('fold_routes_summary'); ur=pm.get('unfold_routes_summary')
            frn=pm.get('fold_recurrence_median_ns'); urn=pm.get('unfold_recurrence_median_ns')
            if fr and str(fr)!='unknown':
                extra=f" (median recurrence {float(frn):.1f} ns)" if frn is not None else ""
                lines.append(f"- **Poincaré fold routes:** {fr}{extra}")
            if ur and str(ur)!='unknown':
                extra=f" (median recurrence {float(urn):.1f} ns)" if urn is not None else ""
                lines.append(f"- **Poincaré unfold routes:** {ur}{extra}")
    except Exception:
        pass
    return (['### Key diagnostics','']+lines+['']) if lines else []


def summary_md(path,s):
    _cu=s.get('primary_cv_units','A')
    lines=['# GaREUS PMF analysis summary','',f"Input: `{s['production_dir']}`",f"Samples/windows: **{s['n_samples']} / {s['n_windows']}**",f"Temperature: **{s['temperature_K']:.2f} K**",f"CV range: **{s['cv_min_A']:.3f} - {s['cv_max_A']:.3f} {_cu}**",f"Selected unbiased PMF: **{s['selected_unbiased_method']}**",f"PMF minimum: **{s['pmf_minimum_cv_A']} {_cu}**",f"PMF span: **{s['pmf_span_kcal_mol']:.3f} kcal/mol**",''] + _render_health_section_md(s) + _key_diagnostics_md(s) + ['## MBAR / umbrella diagnostics','',f"Converged: **{s['mbar']['converged']}** after {s['mbar']['iterations']} iterations",f"Backend: **{s['mbar'].get('backend','unknown')}**" + (f" / threads: **{s['mbar'].get('threads')}**" if s['mbar'].get('threads') else ""),f"Base ESS: **{s['mbar']['base_ess']:.1f}** / {s['n_samples']}", '', '## GaMD boost diagnostics','']
    b=s['boost']
    if b.get('available'):
        lines += [f"Boost mean/std: **{b['mean_kcal_mol']:.3f} / {b['std_kcal_mol']:.3f} kcal/mol**",f"Boost range: **{b['min_kcal_mol']:.3f} - {b['max_kcal_mol']:.3f} kcal/mol**",f"Anharmonicity score: **{b.get('anharmonicity_score')}**",f"Boost exponential ESS fraction: **{b.get('boost_reweight_ess_fraction',0):.3f}**"]
    else: lines.append('No finite variable GaMD boosts found; PMF is umbrella-only unbiased.')
    by_rung=s.get('gamd_boost_by_rung')
    if by_rung:
        # Pooling across rungs (the block above) hides exactly the number
        # that decides whether a boost can be reweighted at all: Miao's
        # cumulant-reweighting criterion (anharmonicity < 0.01) applies PER
        # STATE, and lambda=0 (plain umbrella) vs lambda=1 (full boost) have
        # very different mean/sd by construction.
        lines += ['', '### Per-rung boost/reweighting diagnostics', '',
                  '| λ | n | ⟨ΔV⟩ kcal/mol | σ_ΔV kcal/mol | ⟨ΔV⟩ kT | anharmonicity | skew |',
                  '|---:|---:|---:|---:|---:|---:|---:|']
        for r in by_rung:
            lines.append(
                f"| {r['lambda']:.3f} | {int(r['n'])} | {r['mean_dv_kcal']:.3f} | "
                f"{r['sd_dv_kcal']:.3f} | {r['mean_dv_kt']:.3f} | {r['anharm_nats']:.3f} | {r['skew']:.3f} |"
            )
    rg=s.get('rg',{}) or {}
    lines += ['', '## Radius of gyration diagnostics', '']
    if rg.get('available'):
        lines += [f"Selected Rg PMF method: **{rg.get('selected_unbiased_method')}**",f"Mean ± std Rg: **{float(rg.get('mean_A',float('nan'))):.3f} ± {float(rg.get('std_A',float('nan'))):.3f} Å**",f"Rg PMF minimum: **{rg.get('pmf_minimum_rg_A')} Å**",f"Rg PMF span: **{float(rg.get('pmf_span_kcal_mol',float('nan'))):.3f} kcal/mol**"]
    else:
        lines.append(f"Rg unavailable: {rg.get('reason','not computed')}")
    fes2d=s.get('distance_rg_2d_fes',{}) or {}
    lines += ['', '## Distance vs Rg 2D FES', '']
    if fes2d.get('available'):
        lines += [f"Selected 2D FES method: **{fes2d.get('selected_unbiased_method')}**",f"2D FES minimum: **distance {fes2d.get('pmf_minimum_cv_A')} Å, Rg {fes2d.get('pmf_minimum_rg_A')} Å**",f"2D FES span: **{float(fes2d.get('pmf_span_kcal_mol',float('nan'))):.3f} kcal/mol**",f"Grid: **{fes2d.get('cv_bins')} × {fes2d.get('rg_bins')}** bins",f"Normalization: **{fes2d.get('normalization','minimum shifted to 0')}**"]
    else:
        lines.append(f"2D FES unavailable: {fes2d.get('reason','not computed')}")
    pca=s.get('pca_2d_fes',{}) or {}
    lines += ['', '## PCA1 vs PCA2 2D FES', '']
    if pca.get('available'):
        lines += [f"Selected PCA 2D FES method: **{pca.get('selected_unbiased_method')}**",f"PCA 2D FES minimum: **PCA1 {pca.get('pmf_minimum_pca1_A')} Å, PCA2 {pca.get('pmf_minimum_pca2_A')} Å**",f"PCA 2D FES span: **{float(pca.get('pmf_span_kcal_mol',float('nan'))):.3f} kcal/mol**",f"Grid: **{pca.get('pca_bins')} × {pca.get('pca_bins')}** bins",f"Explained variance: **PC1 {float(pca.get('explained_variance_ratio_pc1',float('nan'))):.3f}, PC2 {float(pca.get('explained_variance_ratio_pc2',float('nan'))):.3f}**",f"Selection: **{pca.get('selection')}** ({pca.get('n_atoms')} atoms)",f"Normalization: **{pca.get('normalization','minimum shifted to 0')}**"]
    else:
        lines.append(f"PCA 2D FES unavailable: {pca.get('reason','not computed')}")
    extra=s.get('extra_observable_pmfs',{}) or {}
    lines += ['', '## Extra trajectory observable PMFs', '']
    if extra.get('available'):
        scalar=extra.get('scalar_pmfs',{}) or {}
        tors=extra.get('torsions',{}) or {}
        c2d=extra.get('contact_2d_fes',{}) or {}
        lines += [f"Selected method: **{extra.get('selected_unbiased_method')}**",f"Frames accumulated: **{extra.get('n_samples')}**",f"Scalar PMFs: **{', '.join(sorted(scalar.keys())) if scalar else 'none'}**",f"Contact 2D FES: **{', '.join(sorted(c2d.keys())) if c2d else 'none'}**",f"Torsion PMFs: **phi residues {tors.get('phi_residues',0)}, psi residues {tors.get('psi_residues',0)}, Ramachandran 2D residues {tors.get('ramachandran_residues',0)}**",f"Secondary-structure residue probabilities: **{extra.get('secondary_structure_residue_count',0)} residues**"]
    else:
        lines.append(f"Extra observable PMFs unavailable: {extra.get('reason','not computed')}")
    lines += ['', '## Outputs', ''] + [f"- `{k}`: `{v}`" for k,v in s['files'].items()] + ['', '## Warnings', '']
    lines += [f'- {w}' for w in s['warnings']] if s['warnings'] else ['- No major automatic warnings.']
    path.write_text('\n'.join(lines)+'\n')
