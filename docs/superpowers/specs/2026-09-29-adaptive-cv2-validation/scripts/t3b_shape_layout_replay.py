"""Read-only replay of the 3.2 shape layout on chignolin_9's swarm (nothing written under RUNS/)."""
import json
import sys
import time
import types
from pathlib import Path

import numpy as np

REPO = Path('/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler')
sys.path.insert(0, str(REPO))
from gareus.cv_selection import contracts as C
from gareus.cv_selection.models import evaluate_component, from_candidate_set
from gareus.pep_gamd import PepGamdEnvelope
from gareus.swarm.analyze import _build_swarm_dataset, _load_members
from gareus.swarm.cv2_shape_layout import build_shape_pair_layout
from gareus.swarm.driver import _load_plan
from gareus.swarm.ladder_design import deltav_max_kj

C9 = REPO / 'RUNS/chignolin_9/swarm'
OUT = Path(sys.argv[3]) if len(sys.argv) > 3 else Path('/tmp/t3b/c9_shape')

if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    an = C9 / 'analysis'
    rows, _meta = _load_plan(C9 / 'round_000')
    done, ok_traces, _frames, _missing, ok_features = _load_members(C9 / 'round_000', rows)
    discard = json.loads((an / 'envelope_discard.json').read_text())['pooled_discard_frames']
    args = types.SimpleNamespace(swarm_output_interval_ps=2.0)
    dataset, aux = _build_swarm_dataset(rows, ok_traces, ok_features, discard, args, None, '0' * 64)
    cs = C.CandidateSet.from_json_bytes((an / 'cv_candidate_set.json').read_bytes())
    sel = json.loads((an / 'cv_selection_report.json').read_text())
    j = int(sel['selected_component_index'])
    z2 = evaluate_component(from_candidate_set(cs), j, dataset.features, dataset.anchor.values)
    env = PepGamdEnvelope.from_json(an / 'shared_gamd_setup' / 'shared_gamd_setup_globals.json')
    dv = deltav_max_kj(aux['v_pep'], aux['v_dih'], env)
    lad = json.loads((an / 'ladder_design.json').read_text())
    plan = json.loads((an / 'layout_plan.json').read_text())
    reps = sel['layout']['region_representative_centre_indices']
    reserve = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
    min_members = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    t0 = time.time()
    out = build_shape_pair_layout(
        cv1=dataset.anchor.values, z2=z2, member_ids=dataset.member_ids, deltav_kj=dv, centers1=lad['centers'],
        ks1=lad['k_kcal'], lambdas=lad['lambdas'], temperature_k=300.0, uniform_centres2=sel['cv2_centers'],
        uniform_ks2=sel['cv2_k_kcal'], overlap_sigma=1.5, k_min=1e-3, k_max=1000.0, max_replicas=236,
        region_centre_indices=reps, reserve_fraction=reserve, min_mode_members=min_members)
    wall = time.time() - t0
    lay, rec = out['layout'], out['record']
    print(f'frames {z2.size}, members {np.unique(dataset.member_ids).size}, wall {wall:.1f}s, '
          f'reserve {reserve}, min_mode_members {min_members}')
    print('uniform: kind', plan['kind'], 'spatial', plan['spatial_states'], 'k2', sel['cv2_k_kcal'][0],
          'centres2', [round(c, 3) for c in sel['cv2_centers']])
    print('shape: kind', lay['kind'], 'spatial', lay['spatial_states'], 'sigma_w_target', round(rec['sigma_w_target'], 3),
          'granted', rec['n_granted'], 'dropped', rec['n_dropped'], 'X7', len(rec['mode_axis_windows']))
    for col in rec['columns']:
        fit, pl = col['fit'], col['placement']
        modes = [(round(c['mean'], 2), round(c['sd'], 2), round(c['weight'], 2), c['n_members'], c['accepted'])
                 for c in fit['components']]
        print(f"col {col['index']:2d} c1={col['centre1']:.3f} share={col['weight_share']:.3f} K={fit['n_components']} "
              f"reg={fit['regularisation']:g} modes={modes}")
        print(f"      centres={[round(c, 2) for c in pl['centres']]} kinds={''.join(k[0] for k in pl['kinds'])} "
              f"k2=[{min(pl['k2']):.3f}..{max(pl['k2']):.3f}] floor={pl['n_at_k_floor']} "
              f"compress=[{min(pl['mean_compression']):.2f}..{max(pl['mean_compression']):.2f}]")
    kinds = {}
    for q in rec['requests']:
        kinds.setdefault(q['kind'], [0, 0])[0 if q['granted'] else 1] += 1
    print('requests granted/dropped by kind', kinds)
    if 'adaptive_reserve' in lay:
        print('reserve', lay['adaptive_reserve'])
    (OUT / f'c9_shape_r{reserve}_m{min_members}.json').write_text(json.dumps(
        {'summary': out['summary'], 'record': rec, 'kind': lay['kind'], 'spatial_states': lay['spatial_states'],
         'adaptive_reserve': lay.get('adaptive_reserve')}, indent=1, default=float))
