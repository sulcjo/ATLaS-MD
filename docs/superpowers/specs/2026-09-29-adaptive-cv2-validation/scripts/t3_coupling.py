"""T3 task 3: CV2 coupling-gate predictions vs realised CV1 offsets on chignolin_9, read-only.

Usage: python t3_coupling.py <campaign_dir> <final_combined payload.json> <out.json>

For every lambda = 0 window: ``cv2_coupling_fraction`` at the deployed k2 and on a k2 grid, the
linear-response CV1 shift the gate's model predicts from the REALISED CV2 offset <z - z0>, and
the realised CV1 offset against a reference that lacks the CV2 umbrella (CV2-only: the anchor
state, no restraint; 2D: the CV1-only window at the same CV1 centre). For the CV2 chains the
predicted adjacent-window overlap vs k2 from the local curvature F''_2 = RT / var(z) - k2
estimated at the deployed k2 (harmonic model, equal-width Gaussians).
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from t3_gaussian_overlap import gaussian_pair_overlap  # noqa: E402

from gareus.cv_selection.coupling import R_KCAL_MOL_K, cv2_coupling_fraction, largest_passing_k2  # noqa: E402
from gareus.cv_selection.models import PairModelRuntime  # noqa: E402
from gareus.cv_selection.residual_runtime import compile_component  # noqa: E402

camp = Path(sys.argv[1])
payload = json.loads(Path(sys.argv[2]).read_text())
out = Path(sys.argv[3])
an = camp / "swarm" / "analysis"
rt_model = PairModelRuntime.load(an / "cv_pair_model.json", an / "cv_candidate_set.json", an / "cv_feature_schema.json")
fit, j = rt_model.fit, rt_model.j
comp = compile_component(fit, int(j), norm=1.0)
T = float(payload["edge_metric"]["temperature_k"])
RT = R_KCAL_MOL_K * T
SWARM_POOLED_SD = 0.195   # CLAUDE.md P5 entry: swarm round_000 pooled CV1 sd after discard (k1zero_c9.py)

st = {}
for s in payload["states"]:
    pc = s.get("paired_cv") or {}
    r = pc.get("restraint") or {}
    if abs(float(r.get("gamd_lambda") or 0.0)) > 1e-9:
        continue
    st[int(s["state_id"])] = dict(
        c1=r["primary_center"], k1=float(r.get("primary_k") or 0.0), c2=r.get("secondary_center"),
        k2=float(r.get("secondary_k") or 0.0), m1=pc["cv1"]["mean"], v1=pc["cv1"]["var"],
        m2=(pc.get("cv2") or {}).get("mean"), v2=(pc.get("cv2") or {}).get("var"), n=pc.get("n_pairs"))


def dzdc(c):
    t = (c - comp.anchor_mean) / comp.anchor_std
    t = min(max(t, comp.clamp_lo), comp.clamp_hi)
    return -(comp.k1 + 2.0 * comp.k2 * t) / (comp.sigma_j * comp.anchor_std)


anchor = [sid for sid, s in st.items() if s["k1"] <= 0 and s["k2"] <= 0]
a_m1 = st[anchor[0]]["m1"] if anchor else None
a_sd1 = math.sqrt(st[anchor[0]]["v1"]) if anchor else None
a_m2 = st[anchor[0]]["m2"] if anchor else None
a_slope = None
for s_ in payload["states"]:
    if anchor and int(s_["state_id"]) == anchor[0]:
        pc_ = s_["paired_cv"]
        if pc_.get("cov_cv1_cv2") is not None and (pc_.get("cv2") or {}).get("var"):
            a_slope = float(pc_["cov_cv1_cv2"]) / float(pc_["cv2"]["var"])
cv1_only = {round(s["c1"], 4): sid for sid, s in st.items() if s["k1"] > 0 and s["k2"] <= 0}
K2_GRID = [0.1, 0.2, 0.3, 0.42, 0.6, 0.8, 1.0, 1.1795, 1.5, 2.0, 2.5, 3.0, 3.54]

rows = []
for sid, s in sorted(st.items()):
    if s["k2"] <= 0:
        continue
    k1zero = s["k1"] <= 0
    z_off = None if s["m2"] is None else s["m2"] - s["c2"]
    d = dzdc(s["m1"])
    rec = dict(state_id=sid, c1=s["c1"], k1=s["k1"], c2=s["c2"], k2=s["k2"], m1=s["m1"], sd1=math.sqrt(s["v1"]),
               m2=s["m2"], sd2=None if s["v2"] is None else math.sqrt(s["v2"]), z_offset=z_off, dz_dc=d,
               mode="k1_zero" if k1zero else "k1")
    refs = {"swarm_pooled": SWARM_POOLED_SD, "anchor_production": a_sd1} if k1zero else {"designed": None}
    for name, sref in refs.items():
        w = dict(primary_center=s["c1"], primary_k=s["k1"], secondary_center=s["c2"], temperature_k=T,
                 sigma_w1=sref)
        g = cv2_coupling_fraction(fit, j, s["k2"], w)
        lp = largest_passing_k2(fit, j, s["k2"], w)
        rec[f"frac_{name}"] = g.get("fraction")
        rec[f"curv_ratio_{name}"] = g.get("curvature_ratio")
        rec[f"shift_sigma_bound_{name}"] = g.get("mean_shift_sigma")
        rec[f"passing_k2_{name}"] = lp["k2"]
        rec[f"grid_{name}"] = [(k, cv2_coupling_fraction(fit, j, k, w).get("fraction")) for k in K2_GRID]
    # Linear-response prediction from the realised CV2 offset (force on c = -k2 <z - z0> dz/dc).
    if z_off is not None:
        force = -s["k2"] * z_off * d
        if k1zero:
            rec["pred_shift_lr_swarm_sd"] = force * SWARM_POOLED_SD ** 2 / RT
            rec["pred_shift_lr_anchor_sd"] = None if a_sd1 is None else force * a_sd1 ** 2 / RT
        else:
            rec["pred_shift_lr_designed"] = force / s["k1"]
            rec["pred_shift_lr_realised_curv"] = force * s["v1"] / RT
    if k1zero and a_m1 is not None and a_slope is not None and s["m2"] is not None:
        # Equilibrium-correlation predictor: the anchor's own regression of c on z.
        rec["pred_shift_regression_anchor"] = a_slope * (s["m2"] - a_m2)
    if k1zero and a_m1 is not None:
        rec["realised_shift"] = s["m1"] - a_m1
        rec["realised_shift_in_anchor_sd"] = (s["m1"] - a_m1) / a_sd1
        rec["reference"] = f"anchor state {anchor[0]}"
    elif not k1zero:
        ref = cv1_only.get(round(s["c1"], 4))
        if ref is not None:
            rec["realised_shift"] = s["m1"] - st[ref]["m1"]
            rec["realised_shift_in_sigma_w1"] = rec["realised_shift"] / math.sqrt(RT / s["k1"])
            rec["realised_shift_in_sampled_sd"] = rec["realised_shift"] / math.sqrt(st[ref]["v1"])
            rec["reference"] = f"CV1-only state {ref}"
    rows.append(rec)

# CV2 chain overlap vs k2: adjacent windows along CV2 with identical CV1 restraint.
chains = []
groups = {}
for sid, s in st.items():
    if s["k2"] > 0:
        groups.setdefault((round(s["c1"], 4), s["k1"] > 0), []).append(sid)
for (c1, restrained), ids in sorted(groups.items()):
    ids = sorted(ids, key=lambda x: st[x]["c2"])
    for a, b in zip(ids[:-1], ids[1:]):
        sa, sb = st[a], st[b]
        if sa["v2"] is None or sb["v2"] is None:
            continue
        fpp = [RT / sa["v2"] - sa["k2"], RT / sb["v2"] - sb["k2"]]
        f_edge = max(0.0, float(np.mean(fpp)))
        dz = sb["c2"] - sa["c2"]
        realised_d = abs(sb["m2"] - sa["m2"]) / math.sqrt(0.5 * (sa["v2"] + sb["v2"]))
        grid = []
        for k in K2_GRID:
            dd = k * dz / math.sqrt(RT * (k + f_edge))
            grid.append((k, gaussian_pair_overlap(dd)))
        chains.append(dict(i=a, j=b, cv1_center=c1, cv1_restrained=restrained, dz=dz, fpp=fpp, fpp_used=f_edge,
                           realised_mean_sep_in_sd=realised_d, gaussian_overlap_at_realised=gaussian_pair_overlap(realised_d),
                           pred_overlap_grid=grid))
out.write_text(json.dumps(dict(windows=rows, cv2_chains=chains, anchor=anchor, anchor_cv1_sd=a_sd1,
                               swarm_pooled_sd=SWARM_POOLED_SD, rt=RT, k2_grid=K2_GRID,
                               comp=dict(k1=comp.k1, k2=comp.k2, sigma_j=comp.sigma_j, anchor_mean=comp.anchor_mean,
                                         anchor_std=comp.anchor_std, clamp=[comp.clamp_lo, comp.clamp_hi])), default=float))
print("wrote", out, len(rows), "windows", len(chains), "cv2 chain edges")
