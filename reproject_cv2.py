#!/usr/bin/env python3
"""Write every secondary-CV definition's value for a run's stored samples.

A run that switches its CV2 definition mid-campaign (``tica_switch_cv2``)
leaves the pooled MBAR problem spliced: each state column's secondary params
are expressed in one CV2 definition, but half the rows carry a cv2 computed
in the other. See ``gareus/mbar_analysis/cv2_reprojection.py`` for the full
argument; the short version is that a pooled solve needs every column to be
evaluable, which needs every regime's cv2 for every row.

This tool produces exactly that table. It does NOT change any analysis by
itself: nothing reads the table unless a caller asks for it, and
``analyze_gareus_mbar`` never triggers this pass on its own -- a 16 GB
trajectory read is not something an analysis run should start implicitly.

Usage
-----
Free part (stored features only; no trajectory access)::

    python reproject_cv2.py RUNS/<run>/adaptive_production

Validation only, no output written::

    python reproject_cv2.py RUNS/<run>/adaptive_production --validate-only

Full coverage, reading trajectories for epochs that stored no features::

    python reproject_cv2.py RUNS/<run>/adaptive_production \
        --trajectories --topology-atom-count <N>

The gate runs first and unconditionally. If the regime that was in effect
while an epoch's observations were recorded does not reproduce that epoch's
own recorded cv2 to ``STORED_OBS_AGREEMENT_ATOL``, the model, the projection
convention or the atom indexing is wrong, and the expensive pass is refused
rather than producing a table nobody should trust.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from gareus.mbar_analysis.cv2_reprojection import (
    CV2_REPROJECTION_FILENAME,
    Cv2Model,
    features_from_xtc,
    load_cv2_model,
    load_stored_obs,
    project_cv2,
    reproject_stored_obs,
    validate_model_against_stored_obs,
)


def discover_models(prod_dir: Path) -> list:
    """Find every CV2 model a run persisted.

    ``tica_state.json`` is written once at the top level when the tICA switch
    fires; each epoch's ``tica/bootstrap_torsion_cv.json`` holds the
    torsion-PCA basis that was in effect before it. Duplicates across epochs
    are kept only once per regime -- the basis is not refitted per epoch.
    """
    models: list = []
    seen: set = set()
    top = prod_dir / 'tica_state.json'
    if top.is_file():
        m = load_cv2_model(top)
        models.append(m)
        seen.add(m.regime)
    for p in sorted(prod_dir.glob('epoch_*/tica/bootstrap_torsion_cv.json')):
        m = load_cv2_model(p)
        if m.regime not in seen:
            models.append(m)
            seen.add(m.regime)
    return models


def epoch_obs_paths(prod_dir: Path) -> dict:
    """``{epoch_dir: [npz, ...]}`` for epochs that recorded dihedral features."""
    out: dict = {}
    for edir in sorted(prod_dir.glob('epoch_*')):
        paths = sorted((edir / 'tica_obs').glob('dihedral_obs_*.npz'))
        if paths:
            out[edir] = paths
    return out


def epoch_regime(edir: Path) -> str:
    """The secondary_cv mode recorded in this epoch's own run manifest."""
    import json
    mf = edir / 'run_manifest.json'
    if not mf.is_file():
        return ''
    try:
        d = json.loads(mf.read_text())
    except (OSError, ValueError):
        return ''
    return str((d.get('resolved_args') or {}).get('secondary_cv', '') or '')


def run_gate(models: list, obs_by_epoch: dict, verbose: bool = True) -> bool:
    """Validate each epoch's own regime against its recorded cv2.

    Returns True only if every epoch with stored observations was checked and
    agreed. An epoch whose regime has no matching model is a failure, not a
    skip: it means the run persisted observations for a definition this tool
    cannot reconstruct.
    """
    by_regime = {m.regime: m for m in models}
    all_ok = True
    for edir, paths in obs_by_epoch.items():
        regime = epoch_regime(edir)
        model = by_regime.get(regime)
        if model is None:
            print(f'  {edir.name}: FAIL no model for recorded regime {regime!r} '
                  f'(have {sorted(by_regime)})')
            all_ok = False
            continue
        res = validate_model_against_stored_obs(model, load_stored_obs(paths))
        ok = bool(res.get('agrees'))
        all_ok &= ok
        if verbose:
            print(f'  {edir.name}: {"ok  " if ok else "FAIL"} regime={regime} '
                  f'n={res.get("n")} max_dev={res.get("max_abs_deviation", float("nan")):.3e} '
                  f'corr={res.get("correlation", float("nan")):+.6f}')
    return all_ok


def reproject_epoch_from_trajectories(edir: Path, models: list, topology: Path,
                                       stride: int) -> dict:
    """Recompute features from an epoch's replica trajectories.

    The expensive path, for epochs that stored no observations. Uses the same
    ``backbone_dihedral_features`` the run itself used, via
    ``features_from_xtc``, so the column order cannot drift.
    """
    phi, psi = models[0].torsion_indices
    xtcs = sorted((edir / 'baseline' / 'replica_trajectories').glob('*.xtc'))
    xtcs += sorted(edir.glob('topup_*/replica_trajectories/*.xtc'))
    if not xtcs:
        return {}
    steps, feats = [], []
    for xtc in xtcs:
        got = features_from_xtc(xtc, phi, psi, stride=stride)
        if got['steps'].size:
            steps.append(got['steps'])
            feats.append(got['features'])
    if not feats:
        return {}
    F = np.vstack(feats)
    out = {'steps': np.concatenate(steps)}
    for m in models:
        out[f'cv2_{m.regime}'] = project_cv2(m, F)
    return out


def write_table(path: Path, epochs: dict, models: list) -> None:
    """Write one parquet with a cv2 column per regime, plus its provenance."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    cols: dict = {'epoch': [], 'steps': []}
    for m in models:
        cols[f'cv2_{m.regime}'] = []
    for epoch_name, tbl in epochs.items():
        n = int(np.asarray(tbl['steps']).size)
        cols['epoch'].append(np.full(n, epoch_name, dtype=object))
        cols['steps'].append(np.asarray(tbl['steps'], dtype=np.int64))
        for m in models:
            key = f'cv2_{m.regime}'
            cols[key].append(np.asarray(tbl.get(key, np.full(n, np.nan)), dtype=np.float64))
    arrays = {k: np.concatenate(v) if v else np.asarray([]) for k, v in cols.items()}
    table = pa.table({k: pa.array(v) for k, v in arrays.items()})
    pq.write_table(table, str(path))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('prod_dir', help="A run's adaptive_production directory")
    p.add_argument('--out', default=None,
                   help=f'Output parquet; default <prod_dir>/{CV2_REPROJECTION_FILENAME}')
    p.add_argument('--validate-only', action='store_true',
                   help='Run the gate and stop; write nothing.')
    p.add_argument('--trajectories', action='store_true',
                   help='Also recompute features from trajectories for epochs that stored '
                        'none. This reads every replica XTC and is the expensive path; '
                        'without it the table covers only epochs with stored observations.')
    p.add_argument('--stride', type=int, default=1,
                   help='Frame stride for the trajectory pass.')
    args = p.parse_args(argv)

    prod_dir = Path(args.prod_dir)
    if not prod_dir.is_dir():
        print(f'not a directory: {prod_dir}', file=sys.stderr)
        return 2

    models = discover_models(prod_dir)
    if len(models) < 2:
        print(f'Found {len(models)} CV2 model(s) '
              f'({", ".join(m.regime for m in models) or "none"}). Reprojection is only '
              f'meaningful when a run changed its CV2 definition; nothing to do.')
        return 0
    print(f'CV2 models: {", ".join(f"{m.regime} <- {m.source.name}" for m in models)}')

    obs_by_epoch = epoch_obs_paths(prod_dir)
    if not obs_by_epoch:
        print('No epoch stored dihedral observations (tica_obs/), so the projection, '
              'the offset convention and the atom indexing cannot be validated. '
              'Refusing to reproject unvalidated.', file=sys.stderr)
        return 3

    print('Validation gate (each epoch against its own recorded regime):')
    if not run_gate(models, obs_by_epoch):
        print('\nGate FAILED. The stored model, the projection convention or the atom '
              'indexing disagrees with what the run recorded. Not writing a table.',
              file=sys.stderr)
        return 4
    print('Gate passed.')
    if args.validate_only:
        return 0

    epochs: dict = {}
    for edir, paths in obs_by_epoch.items():
        epochs[edir.name] = reproject_stored_obs(models, load_stored_obs(paths))
        print(f'  {edir.name}: {epochs[edir.name]["steps"].size} rows from stored features')

    if args.trajectories:
        for edir in sorted(prod_dir.glob('epoch_*')):
            if edir in obs_by_epoch:
                continue
            got = reproject_epoch_from_trajectories(edir, models, prod_dir, args.stride)
            if got:
                epochs[edir.name] = got
                print(f'  {edir.name}: {got["steps"].size} rows from trajectories')
            else:
                print(f'  {edir.name}: no trajectories found; no rows')
    else:
        skipped = [e.name for e in sorted(prod_dir.glob('epoch_*')) if e not in obs_by_epoch]
        if skipped:
            print(f'\nNot covered (no stored features, --trajectories not given): '
                  f'{", ".join(skipped)}')

    out = Path(args.out) if args.out else prod_dir / CV2_REPROJECTION_FILENAME
    write_table(out, epochs, models)
    total = sum(int(np.asarray(t['steps']).size) for t in epochs.values())
    print(f'\nWrote {out} ({total} rows across {len(epochs)} epoch(s)).')
    print('Rows a pooled solve can use are those whose step appears here for EVERY '
          'regime; joins are exact-step and nothing is interpolated.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
