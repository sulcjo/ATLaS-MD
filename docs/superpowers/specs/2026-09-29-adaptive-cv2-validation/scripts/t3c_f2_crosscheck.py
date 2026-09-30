"""Condition 5: the shape layout's design-measure F''_est (swarm) vs production-sampled curvature.

Read-only (nothing under RUNS/ is written). Three production estimates, all lambda = 0:
  P1 window-local: every CV2-restrained 2-D window, F''_loc = RT / var(cv2 | cv1) - k2
     (conditional variance var2 - cov^2 / var1; Gaussian identity: the conditional precision
     under a 2-D harmonic restraint is F''_22 + k2), plus the marginal RT / var2 - k2;
  P2 CV1-only windows (k2 = 0): the CV2 distribution in a thin CV1 slab sampled without a CV2
     spring -- the closest analogue of the swarm column; pooled RT / var2 and a mixture fit
     (32 time blocks = members, as R3 does) with per-mode RT / variance_curvature;
  P3 union MBAR: lambda = 0 rows of adaptive_union_mbar.npz, own MBAR over the lambda = 0
     states (umbrella energies recomputed exactly), per column the swarm's own CV1 kernel
     (Gaussian of sigma_w1, cut at 3 sigma) times the unbiased weights; pooled RT / var and
     a weighted mixture fit (members = 8 contiguous row blocks per state).
Uncertainty: P1/P2 block bootstrap over the P4 subsample's 32 time blocks per state (200
reps); P3 block bootstrap with f fixed (20 row blocks per state, 100 reps) for the pooled
value and 20 refits for the mixture modes. No replica-level bootstrap is possible: neither
the P4 NPZ nor the union NPZ carries a replica column.

python t3c_f2_crosscheck.py OUT.json
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path('/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler')
sys.path.insert(0, str(REPO))
from gareus.adaptive import cv2_coverage as cov  # noqa: E402
from gareus.adaptive import cv2_resolution as cr  # noqa: E402
from gareus.adaptive.cv2_shape import estimate_f2, fit_cv2_mixture, _log_components  # noqa: E402
from gareus.swarm.cv2_shape_layout import column_kernel  # noqa: E402
from gareus.swarm.ladder_design import R_KCAL_MOL_K  # noqa: E402

T = 300.0
RT = R_KCAL_MOL_K * T
JOB = Path('/home/sulcjo/.claude/jobs/45d68034/tmp')
SHAPE = JOB / 'c9fix/after/shape/c9_shape_r0.0_m8.json'
PAYLOAD = JOB / 'c9dry/replay/adaptive_final_combined_diagnostics.json'
P4NPZ = JOB / 'c9dry/replay/adaptive_final_combined_diagnostics_paired_cv.npz'
UNION = REPO / 'RUNS/chignolin_9/adaptive_production/adaptive_union_mbar.npz'
RNG = np.random.default_rng(20260930)


def q(a, p):
    a = np.asarray([x for x in a if x is not None and math.isfinite(x)])
    return None if not a.size else [float(np.quantile(a, x)) for x in p]


# ---- swarm model --------------------------------------------------------------------------

class SwarmColumn:
    def __init__(self, col):
        self.c1 = float(col['centre1'])
        self.sigma_w1 = float(col['sigma_w1'])
        fit = col['fit']
        self.pooled_var = float(fit['pooled_variance'])
        self.n_members = int(fit['n_members'])
        self.comps = [c for c in fit['components'] if c['accepted']]
        self.all_comps = fit['components']
        self.f2_pool_raw = RT / self.pooled_var
        self.f2_pool = estimate_f2(self.pooled_var, self.pooled_var, self.n_members, T)
        self.f2_comp = [estimate_f2(c['variance_curvature'], self.pooled_var, c['n_members'], T) for c in self.comps]
        self.placement = col['placement']

    def f2(self, z):
        """_ShapeModel.f2: F'' of the dominant accepted component at z (regularised density)."""
        if not self.comps:
            return self.f2_pool, None
        lp = _log_components(np.atleast_1d(float(z)), [c['mean'] for c in self.comps],
                             [c['variance'] for c in self.comps], [c['weight'] for c in self.comps])
        i = int(np.argmax(lp[0]))
        return float(self.f2_comp[i]), i


# ---- P4 subsample helpers ------------------------------------------------------------------

def blocks(src, n_blocks=cr.MEMBER_BLOCKS):
    return cr.member_blocks(src, src.size, n_blocks)


def boot_var(c1, c2, blk, k2, n_rep=200):
    """Bootstrap (over time blocks) of the conditional / marginal F''_loc."""
    ub, inv = np.unique(blk, return_inverse=True)
    out_c, out_m = [], []
    for _ in range(n_rep):
        pick = RNG.integers(0, ub.size, ub.size)
        idx = np.concatenate([np.flatnonzero(inv == j) for j in pick])
        a, b = c1[idx], c2[idx]
        v2, v1 = b.var(), a.var()
        cv = np.mean((a - a.mean()) * (b - b.mean()))
        vc = v2 - cv * cv / v1 if v1 > 0 else v2
        out_c.append(RT / vc - k2)
        out_m.append(RT / v2 - k2)
    return out_c, out_m


def comp_record(c, k2=0.0, pooled=None):
    vc = c.get('variance_curvature') or c['variance']
    return {'mean': c['mean'], 'sd': c['sd'], 'weight': c['weight'], 'n_members': c['n_members'],
            'accepted': c['accepted'], 'variance_curvature': vc,
            'f2_raw': RT / vc - k2,
            'f2_shrunk': (estimate_f2(vc, pooled, c['n_members'], T) - k2) if pooled else None}


def main(out_path):
    t0 = time.time()
    shape = json.loads(SHAPE.read_text())
    cols = [SwarmColumn(c) for c in shape['record']['columns']]
    col_of = {round(c.c1, 4): c for c in cols}
    payload = json.loads(PAYLOAD.read_text())
    p4 = np.load(P4NPZ)
    p4_sid, p4_c1, p4_c2, p4_src = (np.asarray(p4[k]) for k in ('state_id', 'cv1', 'cv2', 'source_index'))

    # ---- P1 / P2 -------------------------------------------------------------------------
    p1, p2 = [], []
    for s in payload['states']:
        pc = s['paired_cv']
        r = pc['restraint']
        if r['gamd_lambda'] != 0.0 or not r['cv1_restrained']:
            continue
        col = col_of.get(round(r['primary_center'], 4))
        if col is None:
            continue
        sid = int(s['state_id'])
        m = p4_sid == sid
        a, b, src = p4_c1[m], p4_c2[m], p4_src[m]
        blk = blocks(src)
        v2, v1, cv = pc['cv2']['var'], pc['cv1']['var'], pc['cov_cv1_cv2']
        vcond = v2 - cv * cv / v1
        if r['cv2_restrained']:
            k2 = float(r['secondary_k'])
            bc, bm = boot_var(a, b, blk, k2)
            f2c, f2m = RT / vcond - k2, RT / v2 - k2
            sw, comp_i = col.f2(pc['cv2']['mean'])
            p1.append({'state': sid, 'c1': r['primary_center'], 'c2': r['secondary_center'], 'k2': k2,
                       'cv2_mean': pc['cv2']['mean'], 'cv2_sd': math.sqrt(v2), 'corr': pc['corr_cv1_cv2'],
                       'f2_prod_cond': f2c, 'f2_prod_cond_ci90': q(bc, (0.05, 0.95)),
                       'f2_prod_marg': f2m, 'f2_prod_marg_ci90': q(bm, (0.05, 0.95)),
                       'f2_swarm_at_mean': sw, 'swarm_component': comp_i,
                       'swarm_fallback_pooled': comp_i is None,
                       'ratio_cond': f2c / sw if sw > 0 else None,
                       'n_pairs': pc['n_pairs'], 'n_sub': int(m.sum())})
        else:
            bs = []
            ub, inv = np.unique(blk, return_inverse=True)
            for _ in range(200):
                pick = RNG.integers(0, ub.size, ub.size)
                idx = np.concatenate([np.flatnonzero(inv == j) for j in pick])
                bs.append(RT / b[idx].var())
            fit = fit_cv2_mixture(b, blk, max_components=3, min_mode_members=8, seed=sid)
            rec = fit.as_record()
            p2.append({'state': sid, 'c1': r['primary_center'], 'cv2_mean': pc['cv2']['mean'],
                       'cv2_sd': math.sqrt(v2), 'f2_pool_prod': RT / v2, 'f2_pool_prod_ci90': q(bs, (0.05, 0.95)),
                       'f2_pool_swarm_raw': col.f2_pool_raw, 'f2_pool_swarm_shrunk': col.f2_pool,
                       'ratio_pool': (RT / v2) / col.f2_pool_raw,
                       'prod_components': [comp_record(c, 0.0, rec['pooled_variance']) for c in rec['components']],
                       'swarm_components': [comp_record(c, 0.0, col.pooled_var) for c in col.all_comps]})
    print(f'P1 {len(p1)} windows, P2 {len(p2)} CV1-only windows ({time.time() - t0:.0f} s)', flush=True)

    # ---- P3 union MBAR ---------------------------------------------------------------------
    with np.load(UNION, allow_pickle=False) as z:
        sid_all = np.asarray(z['state_ids']); lam = np.asarray(z['state_lambdas'])
        pcen, pk = np.asarray(z['primary_centers']), np.asarray(z['primary_k'])
        scen, sk = np.asarray(z['secondary_centers']), np.asarray(z['secondary_k'])
        sampled = np.asarray(z['sampled_state_ids'])
        cv1_all, cv2_all = np.asarray(z['cv_A'], float), np.asarray(z['secondary_cv'], float)
    rep = [i for i in range(sid_all.size) if abs(lam[i]) <= 1e-9]
    pos = {int(sid_all[i]): j for j, i in enumerate(rep)}
    idx = np.asarray([pos.get(int(s), -1) for s in sampled])
    keep = (idx >= 0) & np.isfinite(cv1_all) & np.isfinite(cv2_all)
    cv1, cv2, idx = cv1_all[keep], cv2_all[keep], idx[keep]
    views = [cr.StateView(int(sid_all[i]), float(pcen[i]), float(pk[i]), float(scen[i]), float(sk[i]), 0.0)
             for i in rep]
    u = cov.reduced_umbrella(cv1, cv2, views, 1.0 / RT)
    n_k = np.bincount(idx, minlength=len(rep)).astype(float)
    ok_k = np.flatnonzero(n_k > 0)
    remap = -np.ones(len(rep), dtype=np.int64); remap[ok_k] = np.arange(ok_k.size)
    info = {}
    f = cov.solve_mbar(u[ok_k], n_k[ok_k], info=info)
    lw = cov.log_weights(u[ok_k], n_k[ok_k], f)
    w = np.exp(lw - lw.max()); w /= w.sum()
    sidx = remap[idx]
    blk20 = cov._block_ids(sidx, 20)
    blk8 = cov._block_ids(sidx, 8)
    print(f'union: {cv1.size} lambda=0 rows, {ok_k.size} states, MBAR {info} ({time.time() - t0:.0f} s)', flush=True)
    del u
    p3 = []
    for col in cols:
        kern = column_kernel(cv1, col.c1, col.sigma_w1)
        wt = w * kern
        sel = wt > 0
        zc, wc, bc, b8 = cv2[sel], wt[sel], blk20[sel], blk8[sel]
        tot = wc.sum()
        mu = np.dot(wc, zc) / tot
        var = np.dot(wc, (zc - mu) ** 2) / tot
        n_eff_kish = tot ** 2 / np.sum(wc ** 2)
        ub, inv = np.unique(bc, return_inverse=True)
        sw_b = np.bincount(inv, weights=wc); swz = np.bincount(inv, weights=wc * zc); swzz = np.bincount(inv, weights=wc * zc * zc)
        stratum = ub // 10 ** 7
        strata = [np.flatnonzero(stratum == s) for s in np.unique(stratum)]
        bs = []
        for _ in range(100):
            pick = np.concatenate([RNG.choice(ix, ix.size) for ix in strata])
            W, Z, ZZ = sw_b[pick].sum(), swz[pick].sum(), swzz[pick].sum()
            m_ = Z / W
            bs.append(RT / (ZZ / W - m_ * m_))
        wfit = wc / tot * n_eff_kish     # weights summing to the Kish effective sample size (BIC n)
        fit = fit_cv2_mixture(zc, b8, wfit, max_components=3, min_mode_members=8, seed=int(col.c1 * 1e4))
        rec = fit.as_record()
        # bootstrap refits of the mixture: per accepted component, RT / variance_curvature of the
        # nearest bootstrap component (by mean)
        comp_bs = [[] for _ in rec['components']]
        for rrep in range(20):
            pick = np.concatenate([RNG.choice(ix, ix.size) for ix in strata])
            rows = np.concatenate([np.flatnonzero(inv == j) for j in pick])
            fb = fit_cv2_mixture(zc[rows], b8[rows], wfit[rows], max_components=3,
                                 min_mode_members=8, seed=rrep).as_record()
            for ci, c in enumerate(rec['components']):
                if not fb['components']:
                    continue
                near = min(fb['components'], key=lambda d: abs(d['mean'] - c['mean']))
                if abs(near['mean'] - c['mean']) < 0.5 * c['sd'] + 0.05:
                    comp_bs[ci].append(RT / (near.get('variance_curvature') or near['variance']))
        comps = []
        for ci, c in enumerate(rec['components']):
            cr_ = comp_record(c, 0.0, rec['pooled_variance'])
            cr_['f2_raw_ci90'] = q(comp_bs[ci], (0.05, 0.95))
            cr_['n_boot_matched'] = len(comp_bs[ci])
            comps.append(cr_)
        # swarm component matches
        sw_match = []
        for sc_ in col.comps:
            if not comps:
                break
            near = min(comps, key=lambda d: abs(d['mean'] - sc_['mean']))
            f2_sw_raw = RT / sc_['variance_curvature']
            f2_sw = estimate_f2(sc_['variance_curvature'], col.pooled_var, sc_['n_members'], T)
            sw_match.append({'swarm_mean': sc_['mean'], 'swarm_sd': sc_['sd'], 'swarm_weight': sc_['weight'],
                             'swarm_members': sc_['n_members'], 'f2_swarm_raw': f2_sw_raw, 'f2_swarm_est': f2_sw,
                             'union_mean': near['mean'], 'union_sd': near['sd'], 'union_weight': near['weight'],
                             'union_accepted': near['accepted'], 'f2_union_raw': near['f2_raw'],
                             'f2_union_raw_ci90': near['f2_raw_ci90'], 'mean_distance': abs(near['mean'] - sc_['mean']),
                             'ratio_raw': near['f2_raw'] / f2_sw_raw, 'ratio_vs_est': near['f2_raw'] / f2_sw})
        # implication at the shape layout's centres
        cen = []
        pl = col.placement
        dens_z = np.linspace(zc.min(), zc.max(), 400)
        for zc0, k2s, f2s in zip(pl['centres'], pl['k2'], pl['f2']):
            # union-model F'' at the centre (dominant component of the union fit)
            accs = [c for c in rec['components']] or []
            if accs:
                lp = _log_components(np.atleast_1d(zc0), [c['mean'] for c in accs], [c['variance'] for c in accs],
                                     [c['weight'] for c in accs])
                cc = accs[int(np.argmax(lp[0]))]
                f2u = RT / (cc.get('variance_curvature') or cc['variance'])
            else:
                f2u = RT / var
            sig_s = math.sqrt(RT / (k2s + f2s))
            near_w = float(wc[np.abs(zc - zc0) <= sig_s].sum() / tot)
            k2_prod = max(RT / pl['sigma_w_target'] ** 2 - f2u, f2u)
            cen.append({'centre': zc0, 'k2_shape': k2s, 'f2_swarm': f2s, 'f2_union': f2u,
                        'compression_realised': k2s / (k2s + f2u), 'k2_if_union_f2': k2_prod,
                        'sampled_sigma_swarm_pred': sig_s, 'sampled_sigma_union_pred': math.sqrt(RT / (k2s + f2u)),
                        'sigma_ratio_if_union_f2': math.sqrt((k2s + f2s) / (k2_prod + f2u)),
                        'union_weight_within_sigma': near_w, 'extrapolated': near_w < 0.01})
        p3.append({'c1': col.c1, 'n_rows': int(sel.sum()), 'kish_n_eff': float(n_eff_kish),
                   'f2_pool_union': RT / var, 'f2_pool_union_ci90': q(bs, (0.05, 0.95)),
                   'f2_pool_swarm_raw': col.f2_pool_raw, 'ratio_pool': (RT / var) / col.f2_pool_raw,
                   'union_components': comps, 'swarm_matches': sw_match, 'centres': cen,
                   'swarm_n_accepted': len(col.comps)})
        print(f'  column {col.c1:.4f}: pooled union {RT / var:.2f} vs swarm {col.f2_pool_raw:.2f}; '
              f'{len(comps)} union comps ({time.time() - t0:.0f} s)', flush=True)
    out = {'provenance': {'shape_replay': str(SHAPE), 'payload': str(PAYLOAD), 'p4_npz': str(P4NPZ),
                          'union_npz': str(UNION), 'temperature_k': T, 'rt_kcal_mol': RT,
                          'union_rows_lambda0': int(cv1.size), 'union_mbar': info},
           'p1_window_local': p1, 'p2_cv1_only': p2, 'p3_union': p3, 'wall_s': time.time() - t0}
    Path(out_path).write_text(json.dumps(out, indent=1, default=float))
    print('wrote', out_path)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '/tmp/t3c/f2_crosscheck.json')
