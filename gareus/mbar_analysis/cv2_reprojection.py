"""Evaluate a run's OTHER secondary-CV definition on its stored frames.

Why this exists
---------------
When a run switches its CV2 definition mid-campaign (the tICA auto-switch:
``tica_switch_cv2`` with ``tica_update_after_epochs``), the pooled MBAR
problem stops being a single Hamiltonian. The loader correctly evaluates each
epoch's rows against the bias params that were actually in effect during that
epoch, so for a state column ``k``:

* rows from the regime ``k``'s params belong to are evaluated in the CV2
  definition those params were written for -- correct;
* rows from the OTHER regime are evaluated against the same params using a
  cv2 value computed in a DIFFERENT definition -- an unevaluated Hamiltonian;
* states created after the switch have, on pre-switch rows, no meaningful
  value at all.

Splitting the analysis per regime fixes the binning axis but not the
estimator (see ``pmf._regime_independent_logw``). The only way to recover a
genuinely pooled solve is to make every column evaluable: compute, for every
sample, the cv2 value of EVERY regime's definition. Both definitions are
affine projections of the same backbone-torsion feature vector, so this is a
re-projection, not a re-derivation -- no model is refitted here.

What is cheap and what is not
-----------------------------
The 36-dim sin/cos features are persisted only for the epochs where they were
collected to FIT the tICA model (``tica_obs/dihedral_obs_*.npz``, written when
``tica_obs_interval > 0``). For every other epoch the features must be
recomputed from the trajectories. So:

* ``validate_model_against_stored_obs`` is free -- it needs no trajectory
  access, and it is the gate that proves the model, the projection AND the
  atom indexing are right before any expensive pass is attempted.
* ``features_from_xtc`` is the expensive path, used by ``reproject_cv2.py``.

Joining grids
-------------
Three grids are in play: stored feature rows, trajectory frames, and sample
rows. On the motivating run the stored feature steps are a strict SUBSET of
the sample steps -- every feature row matches a sample step exactly, while
only ~10% of sample rows have a feature row (``tica_obs_interval = 5``).

Joins here are therefore always on an EXACT ``step`` match, and unmatched
rows are dropped and counted. Nothing is interpolated: the two CV2
definitions' weight vectors are 81.9 degrees apart (cosine similarity
0.141) in the shared 36-dim torsion feature space, so interpolating one from
neighbouring frames of the other has no justification. Coverage is
reported so a caller can see what fraction of rows a pooled solve could
actually use.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from gareus.tica import TICAResult, backbone_dihedral_features, project_tica1

__all__ = [
    "CV2_REPROJECTION_FILENAME",
    "Cv2Model",
    "load_cv2_model",
    "project_cv2",
    "validate_model_against_stored_obs",
    "load_stored_obs",
    "reproject_stored_obs",
    "features_from_xtc",
    "join_on_step",
]

# Filename the reprojection CLI writes under a run's adaptive_production/.
# Consumed by the loader when building a regime-complete u_nk.
CV2_REPROJECTION_FILENAME = "cv2_reprojected.parquet"

# Minimum fraction of a block's rows for which a FOREIGN regime's cv2 must be
# recoverable before the reprojection may be used for that block.
#
# A row whose foreign cv2 is missing gets NaN in those u_nk columns, and clean()
# drops any row with a non-finite u_nk. So using a sparse table does not degrade
# the pooled solve gracefully -- it DELETES the uncovered samples. Observed on
# the motivating run: a table built from stored features only covered 9.8% of
# the pre-switch epoch, and using it silently discarded 90.2% of that regime
# (1,562,528 rows -> 152,576).
#
# Below this fraction the reprojection is declined and the caller keeps every
# sample under the documented (spliced) per-epoch-native cv2 instead. A splice
# that is reported is better than a silent 90% data loss.
CV2_REPROJECTION_MIN_COVERAGE = 0.99

# A projection that reproduces a stored CV to worse than this is not the model
# that produced it. Machine-precision agreement is what a correct model, a
# correct projection and correct atom indexing produce together (measured
# 8.9e-16 on the motivating run); anything near this bound means one of the
# three is wrong, so the gate is deliberately tight rather than forgiving.
STORED_OBS_AGREEMENT_ATOL = 1e-8


RESIDUAL_REGIME = 'residual-torsion-pc'


@dataclass(frozen=True)
class Cv2Model:
    """One regime's CV2 definition.

    Either an affine map of torsion features (``TICAResult``: tica-linear,
    torsion-pca) or a residual component against the contact anchor
    (``PairModelRuntime``), whose projection also needs the primary CV.
    """

    regime: str
    result: Any
    source: Optional[Path] = None
    n_phi: Optional[int] = None      # residual regime only: size of the phi block

    @property
    def is_residual(self) -> bool:
        return self.regime == RESIDUAL_REGIME

    @property
    def n_features(self) -> int:
        if self.is_residual:
            return int(self.result.fit.width)
        return int(np.asarray(self.result.weights).size)

    @property
    def torsion_indices(self) -> tuple:
        if self.is_residual:
            if self.n_phi is None:
                raise ValueError(f'{self.regime}: phi/psi split unknown; load through load_cv2_model')
            atoms = [list(q) for q in self.result.feature_atoms]
            return (atoms[:self.n_phi], atoms[self.n_phi:])
        return (list(self.result.phi_torsion_indices),
                list(self.result.psi_torsion_indices))


def load_cv2_model(path, regime: Optional[str] = None) -> Cv2Model:
    """Load a CV2 model from a ``tica_state.json`` / ``bootstrap_torsion_cv.json``.

    Both files share ``TICAResult``'s schema; the only difference is
    ``method`` ('tica' vs 'pca'). The regime name defaults to the run's own
    naming for that method ('tica-linear' / 'torsion-pca').
    """
    path = Path(path)
    d = json.loads(path.read_text())
    if str(d.get('schema', '')).startswith('atlas-cv-selection-pair-model-'):     # v1 (legacy) and v2
        return _load_residual_model(path, d)
    result = TICAResult.from_dict(d)
    if regime is None:
        regime = 'torsion-pca' if str(d.get('method', 'tica')) == 'pca' else 'tica-linear'
    n_expected = 2 * (len(result.phi_torsion_indices) + len(result.psi_torsion_indices))
    n_weights = int(np.asarray(result.weights).size)
    if n_expected and n_weights != n_expected:
        raise ValueError(
            f'{path}: {n_weights} weights but {n_expected} features implied by '
            f'{len(result.phi_torsion_indices)} phi + {len(result.psi_torsion_indices)} psi '
            f'torsions (2 sin/cos each). The model and its torsion indices disagree.')
    if not result.phi_torsion_indices and not result.psi_torsion_indices:
        raise ValueError(f'{path}: no torsion indices stored; cannot recompute features.')
    return Cv2Model(regime=str(regime), result=result, source=path)


def _load_residual_model(path: Path, d: dict) -> Cv2Model:
    """A pair model needs its candidate set and feature schema; they live beside it."""
    from gareus.cv_selection.models import PairModelRuntime

    cs_path = path.parent / 'cv_candidate_set.json'
    fs_path = path.parent / 'cv_feature_schema.json'
    for p in (cs_path, fs_path):
        if not p.exists():
            raise FileNotFoundError(f'{path}: residual pair model needs {p.name} beside it')
    # Analysis reads historical artifacts of either schema; deployability is production's gate.
    runtime = PairModelRuntime.load(path, cs_path, fs_path, require_deployable=False, allow_legacy_v1=True)
    # The pair model does not store the phi/psi split; the schema's torsion names do.
    fs = json.loads(fs_path.read_text())
    n_phi = sum(1 for f in fs['features'] if f['trig'] == 'sin' and f['torsion_name'].startswith('phi'))
    return Cv2Model(regime=RESIDUAL_REGIME, result=runtime, source=path, n_phi=n_phi)


def project_cv2(model: Cv2Model, features: np.ndarray,
                primary_cv: Optional[np.ndarray] = None) -> np.ndarray:
    """Project a (n, n_features) feature matrix onto this regime's CV2.

    A residual model also needs ``primary_cv`` (the recorded anchor value per
    row); refusing without it is the point -- a torsion-only projection would
    be a different coordinate.
    """
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2:
        raise ValueError(f'features must be 2-D (n, n_features), got shape {features.shape}')
    if features.shape[1] != model.n_features:
        raise ValueError(
            f'{model.regime}: features have {features.shape[1]} columns but the model '
            f'has {model.n_features} weights')
    if model.is_residual:
        if primary_cv is None:
            raise ValueError(f'{model.regime}: projection needs the recorded primary CV per row')
        from gareus.cv_selection.models import evaluate_component
        a = np.asarray(primary_cv, dtype=np.float64)
        if a.shape != (features.shape[0],):
            raise ValueError(f'{model.regime}: primary_cv must be (n,), got {a.shape}')
        return evaluate_component(model.result.fit, model.result.j, features, a)
    return project_tica1(features, model.result)


def load_stored_obs(obs_paths: Sequence) -> dict:
    """Concatenate ``tica_obs/dihedral_obs_*.npz`` observation files.

    Returns the recorded features alongside the bookkeeping needed to join
    them to sample rows. ``secondary_cv`` is the cv2 value that was actually
    in effect when the frame was recorded, which is what the validation gate
    compares against.
    """
    feats, steps, windows, primary, secondary = [], [], [], [], []
    for p in obs_paths:
        z = np.load(str(p))
        feats.append(np.asarray(z['features'], dtype=np.float64))
        steps.append(np.asarray(z['steps'], dtype=np.int64))
        windows.append(np.asarray(z['window'], dtype=np.int64))
        primary.append(np.asarray(z['primary_cv'], dtype=np.float64))
        secondary.append(np.asarray(z['secondary_cv'], dtype=np.float64))
    if not feats:
        raise ValueError('no observation files given')
    return {
        'features': np.concatenate(feats, axis=0),
        'steps': np.concatenate(steps),
        'window': np.concatenate(windows),
        'primary_cv': np.concatenate(primary),
        'secondary_cv': np.concatenate(secondary),
    }


def validate_model_against_stored_obs(model: Cv2Model, obs: dict,
                                       atol: float = STORED_OBS_AGREEMENT_ATOL) -> dict:
    """Gate: does this model reproduce the cv2 that was recorded at the time?

    Run this before any trajectory pass. It validates three things at once
    that are otherwise only testable by their joint effect on the final PMF:
    the stored model coefficients, the projection convention (whether the
    offset already absorbs the mean), and the feature COLUMN ORDER implied by
    the torsion index lists.

    Only meaningful for the regime whose definition was in effect while these
    observations were recorded; applying the other regime's model here is how
    you MEASURE the regime change, not how you validate it.
    """
    got = project_cv2(model, obs['features'], obs.get('primary_cv'))
    ref = np.asarray(obs['secondary_cv'], dtype=np.float64)
    finite = np.isfinite(got) & np.isfinite(ref)
    if not np.any(finite):
        return {'regime': model.regime, 'n': 0, 'agrees': False,
                'reason': 'no finite pairs to compare'}
    dev = np.abs(got[finite] - ref[finite])
    max_dev = float(dev.max())
    corr = (float(np.corrcoef(got[finite], ref[finite])[0, 1])
            if finite.sum() > 1 and np.std(got[finite]) > 0 and np.std(ref[finite]) > 0
            else float('nan'))
    return {
        'regime': model.regime,
        'n': int(finite.sum()),
        'max_abs_deviation': max_dev,
        'mean_abs_deviation': float(dev.mean()),
        'correlation': corr,
        'agrees': bool(max_dev <= atol),
        'atol': float(atol),
    }


def reproject_stored_obs(models: Iterable[Cv2Model], obs: dict) -> dict:
    """Every regime's CV2 for the rows whose features are already stored.

    The free part of a reprojection: no trajectory access. Coverage is
    limited to the epochs that recorded observations at all.
    """
    out = {'steps': np.asarray(obs['steps'], dtype=np.int64),
           'window': np.asarray(obs['window'], dtype=np.int64)}
    for model in models:
        out[f'cv2_{model.regime}'] = project_cv2(model, obs['features'], obs.get('primary_cv'))
    return out


def features_from_xtc(xtc_path, phi_torsions: Sequence, psi_torsions: Sequence,
                       stride: int = 1, atom_indices: Optional[Sequence[int]] = None) -> dict:
    """Backbone-torsion features for every frame of one XTC, with its steps.

    Reads the XTC's own integrator step numbers rather than times, so the
    result can be joined to sample rows on an exact ``step`` match. Positions
    come back in nm, which is what ``backbone_dihedral_features`` expects.

    ``atom_indices`` selects a subset when the torsion indices are numbered
    against a different atom ordering than the trajectory's; leave it None
    when they already agree. Whether they agree is exactly what
    ``validate_model_against_stored_obs`` establishes -- run that first.
    """
    try:
        from mdtraj.formats import XTCTrajectoryFile
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError('reading trajectories needs mdtraj') from exc

    phi = [tuple(int(i) for i in q) for q in phi_torsions]
    psi = [tuple(int(i) for i in q) for q in psi_torsions]
    rows, steps = [], []
    with XTCTrajectoryFile(str(xtc_path), 'r') as fh:
        xyz, _time, step, _box = fh.read(stride=stride)
    xyz = np.asarray(xyz, dtype=np.float64)
    if atom_indices is not None:
        xyz = xyz[:, np.asarray(atom_indices, dtype=np.int64), :]
    for frame in range(xyz.shape[0]):
        rows.append(backbone_dihedral_features(xyz[frame], phi, psi))
        steps.append(int(step[frame]))
    return {'features': (np.vstack(rows) if rows else np.empty((0, 2 * (len(phi) + len(psi))))),
            'steps': np.asarray(steps, dtype=np.int64)}


def join_on_step(target_steps: np.ndarray, source_steps: np.ndarray,
                  source_values: np.ndarray) -> tuple:
    """Map ``source_values`` onto ``target_steps`` by EXACT step equality.

    Returns ``(values, matched)``: ``values`` is NaN wherever no source row
    has that step, and ``matched`` is the boolean coverage mask. Nothing is
    interpolated -- see this module's docstring for why that is not an option
    for a CV that is near-orthogonal to the other regime's.

    Duplicate source steps keep their first occurrence.
    """
    target_steps = np.asarray(target_steps, dtype=np.int64)
    source_steps = np.asarray(source_steps, dtype=np.int64)
    source_values = np.asarray(source_values, dtype=np.float64)
    if source_steps.size != source_values.size:
        raise ValueError(f'source steps ({source_steps.size}) and values '
                         f'({source_values.size}) differ in length')
    values = np.full(target_steps.size, np.nan, dtype=np.float64)
    if source_steps.size == 0:
        # Early return, not a guard inside the mask expression: `&` does not
        # short-circuit, so indexing an empty `uniq` would still be evaluated.
        return values, np.zeros(target_steps.size, dtype=bool)
    uniq, first = np.unique(source_steps, return_index=True)
    pos = np.searchsorted(uniq, target_steps)
    pos_clipped = np.clip(pos, 0, uniq.size - 1)
    matched = (pos < uniq.size) & (uniq[pos_clipped] == target_steps)
    values[matched] = source_values[first[pos_clipped[matched]]]
    return values, matched
