"""Two read-only checks on chignolin_9's lambda = 0 union (same MBAR as t3c_f2_crosscheck.py):

1. which states carry the weight of the narrow union CV2 components at high CV1 (is a band
   real sampling by many states, or one or two states' reweighted samples?);
2. R2's own interval statistics on c9's 2-D columns (``cov.columns`` + ``_interval_stats``):
   contributing centres per heavy interval, split into same-column vs other-column centres, and
   the bootstrap sigma -- re-measuring the 10.2 R2 cell rather than quoting it.

python t3c_union_checks.py F2_CROSSCHECK.json OUT.json
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path('/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler')
sys.path.insert(0, str(REPO))
from gareus.adaptive import cv2_coverage as cov  # noqa: E402
from gareus.adaptive import cv2_resolution as cr  # noqa: E402
from gareus.swarm.cv2_shape_layout import column_kernel  # noqa: E402
from gareus.swarm.ladder_design import R_KCAL_MOL_K  # noqa: E402

T = 300.0
RT = R_KCAL_MOL_K * T
UNION = REPO / 'RUNS/chignolin_9/adaptive_production/adaptive_union_mbar.npz'
PAYLOAD = Path('/home/sulcjo/.claude/jobs/45d68034/tmp/c9dry/replay/adaptive_final_combined_diagnostics.json')


def main(f2_path, out_path):
    f2 = json.load(open(f2_path))
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
    views = [cr.StateView(int(sid_all[i]), float(pcen[i]), float(pk[i]), float(scen[i]), float(sk[i]), 0.0) for i in rep]
    u = cov.reduced_umbrella(cv1, cv2, views, 1.0 / RT)
    n_k = np.bincount(idx, minlength=len(rep)).astype(float)
    ok_k = np.flatnonzero(n_k > 0)
    remap = -np.ones(len(rep), dtype=np.int64); remap[ok_k] = np.arange(ok_k.size)
    f = cov.solve_mbar(u[ok_k], n_k[ok_k])
    lw = cov.log_weights(u[ok_k], n_k[ok_k], f)
    w = np.exp(lw - lw.max()); w /= w.sum()
    sidx = remap[idx]
    sviews = [views[k] for k in ok_k]
    del u
    out = {"narrow_band_attribution": [], "r2_intervals": []}
    # 1. narrow-band attribution
    sw_cols = {round(c['c1'], 4): c for c in f2['p3_union']}
    shape = json.load(open(f2['provenance']['shape_replay']))
    sig1 = {round(c['centre1'], 4): c['sigma_w1'] for c in shape['record']['columns']}
    for c in f2['p3_union']:
        for comp in c['union_components']:
            if comp['sd'] >= 0.13:
                continue
            c1 = round(c['c1'], 4)
            kern = column_kernel(cv1, c['c1'], sig1[c1])
            wt = w * kern
            band = np.abs(cv2 - comp['mean']) <= 1.5 * comp['sd']
            share = np.bincount(sidx[band], weights=wt[band], minlength=ok_k.size)
            tot = share.sum()
            order = np.argsort(share)[::-1]
            top = [{"state": sviews[k].state_id, "c1": round(sviews[k].c1, 4), "k1": round(sviews[k].k1, 1),
                    "c2": round(sviews[k].c2, 3), "k2": round(sviews[k].k2, 3), "weight_share": round(float(share[k] / tot), 3),
                    "n_rows_in_band": int(np.sum(band & (sidx == k)))} for k in order[:4] if share[k] > 0]
            ieff = float(tot ** 2 / np.sum(share ** 2)) if tot > 0 else 0.0
            col_rows = band & (kern > 0)
            out["narrow_band_attribution"].append({
                "c1": c1, "comp_mean": round(comp['mean'], 3), "comp_sd": round(comp['sd'], 3),
                "comp_weight": round(comp['weight'], 3), "rows_in_band": int(col_rows.sum()),
                "slab_weight_fraction_in_band": round(float(wt[band].sum() / wt.sum()), 3),
                "n_states_eff": round(ieff, 2), "top_states": top})
    # 2. R2 intervals on c9's own eligible columns
    diag = json.load(open(PAYLOAD))
    pc = {int(s['state_id']): s['paired_cv'] for s in diag['states']}
    settings = cr.ResolutionSettings(temperature_k=T)
    cviews = {}
    for v in sviews:
        p = pc.get(v.state_id) or {}
        m2 = p.get('cv2') or {}
        cviews[v.state_id] = cr.StateView(v.state_id, v.c1, v.k1, v.c2, v.k2, 0.0, 0, m2.get('mean'), m2.get('var'),
                                          int(p.get('n_pairs', 0) or 0), dict(m2))
    keys = [(round(v.c1, 6), round(v.c2, 6)) for v in sviews]
    uniq = {k: i for i, k in enumerate(dict.fromkeys(keys))}
    centre_of_state = np.asarray([uniq[k] for k in keys], dtype=np.int64)
    centre_c1 = {i: k[0] for k, i in uniq.items()}
    rng = np.random.default_rng([3303, 2])
    for col in cov.columns(cviews, sorted(cviews), settings):
        col = cov._extend_to_weight(col, cv1, cv2, w)
        # pre-v3 fixed 20 blocks per state, so t2_data/t3c_union_checks.json reproduces (v3's default
        # is autocorrelation blocks; the 10.6 re-measure is t3b_cv2_resolution_dryrun.py)
        frac, n_eff, n_contrib, sigma = cov._interval_stats(col, cv1, cv2, sidx, w, centre_of_state, rng,
                                                            block_ids=cov._block_ids(sidx, cov.BOOT_BLOCKS))
        edges = col['edges']
        slab = np.abs(cv1 - col['c1']) <= col['slab_half_width']
        bins = np.clip(np.searchsorted(edges, cv2, side='right') - 1, 0, edges.size - 2)
        rows = []
        for i in range(edges.size - 1):
            if frac[i] < cov.MIN_INTERVAL_WEIGHT:
                continue
            m = slab & (bins == i) & (cv2 >= edges[0]) & (cv2 <= edges[-1])
            sh = np.bincount(centre_of_state[sidx[m]], weights=w[m], minlength=len(uniq))
            sh = sh / sh.sum() if sh.sum() > 0 else sh
            contrib = np.flatnonzero(sh >= cov.CONTRIBUTOR_SHARE)
            same = sum(1 for c_ in contrib if abs(centre_c1[c_] - round(col['c1'], 6)) < 1e-6)
            rows.append({"interval": [round(float(edges[i]), 3), round(float(edges[i + 1]), 3)],
                         "frac": round(float(frac[i]), 3), "n_contrib": int(n_contrib[i]),
                         "n_contrib_same_column": int(same), "n_contrib_other_column": int(len(contrib) - same),
                         "sigma_kT": None if not np.isfinite(sigma[i]) else round(float(sigma[i]), 3)})
        out["r2_intervals"].append({"c1": round(col['c1'], 4), "n_members": len(col['members']),
                                    "slab_half_width": round(col['slab_half_width'], 4), "rows_in_slab": int(slab.sum()),
                                    "heavy_intervals": rows})
    allr = [r for c in out['r2_intervals'] for r in c['heavy_intervals']]
    out["r2_summary"] = {"n_heavy_intervals": len(allr),
                         "n_contrib_min_max": [min(r['n_contrib'] for r in allr), max(r['n_contrib'] for r in allr)],
                         "n_contrib_same_column_min_max": [min(r['n_contrib_same_column'] for r in allr),
                                                           max(r['n_contrib_same_column'] for r in allr)],
                         "n_intervals_same_column_lt2": sum(1 for r in allr if r['n_contrib_same_column'] < 2),
                         "n_intervals_with_other_column_contributors": sum(1 for r in allr if r['n_contrib_other_column'] > 0),
                         "sigma_max_kT": max(r['sigma_kT'] for r in allr if r['sigma_kT'] is not None),
                         "n_sigma_gt_0p25": sum(1 for r in allr if r['sigma_kT'] is not None and r['sigma_kT'] > 0.25),
                         "n_columns": len(out['r2_intervals'])}
    json.dump(out, open(out_path, 'w'), indent=1)
    print(json.dumps(out['r2_summary']))
    for b in out['narrow_band_attribution']:
        print(b['c1'], b['comp_mean'], b['comp_sd'], 'n_states_eff', b['n_states_eff'], [(t['state'], t['c2'], t['k2'], t['weight_share']) for t in b['top_states']])


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
