"""Shared lambda-ladder boost application, used by every MBAR loader/builder.

Fix for a 2026-09-07 review of Task 6 ("MBAR reduced potentials with the
ladder term"): three call sites (``load_csv``, ``load_parquet``,
``build_union_state_mbar_inputs``) each hand-rolled their own copy of
"validate v_pep/v_dih/envelope, then add beta * pep_gamd_boost_matrix_kj(...)
onto the bias matrix", and two more real loaders
(``load_parquet_adaptive_union``, ``load_epoch_csv_adaptive``) had no copy at
all. This module is the ONE place the term is ever added, so every caller
shares the same missing-energy guard, the same NaN-propagation, and the same
envelope-file resolution.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np


def load_pep_gamd_envelope(run_dir):
    """Locate and load the frozen ``PepGamdEnvelope`` for a run/campaign directory.

    Tries, in order:
    - ``<run_dir>/shared_gamd_setup_globals.json`` (single production-run
      convention; see every ``write_json(out_dir / "shared_gamd_setup_globals.json",
      ...)`` call in ``gareus/production.py``)
    - ``<run_dir>/global_shared_gamd_setup/shared_gamd_setup_globals.json``
      (adaptive-production campaign-wide export convention, where
      ``run_dir`` is the ``adaptive_production`` directory; see
      ``gareus/adaptive_production.py:6035`` and ``gareus/checkpoints.py:45``,
      both of which resolve the export to exactly this path).

    Returns ``None`` if neither file exists. A caller that needs one (some
    state carries ``gamd_lambda > 0``) gets a loud ``ValueError`` out of
    ``apply_ladder_boost_to_u`` instead; a caller that doesn't need one (no
    active rung) never has to care that it's missing.
    """
    from gareus.pep_gamd import PepGamdEnvelope
    run_dir = Path(run_dir)
    for candidate in (
        run_dir / "shared_gamd_setup_globals.json",
        run_dir / "global_shared_gamd_setup" / "shared_gamd_setup_globals.json",
    ):
        if candidate.exists():
            return PepGamdEnvelope.from_json(candidate)
    return None


def apply_ladder_boost_to_u(
    u_nk: np.ndarray,
    v_pep: np.ndarray,
    v_dih: np.ndarray,
    state_lambdas: np.ndarray,
    envelope,
    beta: float,
    meta: dict,
) -> np.ndarray:
    """Add each state's own closed-form Pep-GaMD boost onto ``u_nk``.

    ``u_nk`` is ``(n_samples, n_states)``; ``v_pep``/``v_dih`` are
    ``(n_samples,)``; ``state_lambdas`` is ``(n_states,)``. ``beta`` is the
    scalar the raw kJ/mol boost matrix is multiplied by before being added
    onto ``u_nk`` -- pass the real ``1/(kB*T)`` when ``u_nk`` is already
    reduced/dimensionless (the common case), or ``1.0`` when ``u_nk`` is
    still a plain kJ/mol matrix and the caller reduces by the real beta
    itself afterwards (``build_union_state_mbar_inputs`` needs the
    un-reduced total to keep its own ``gamd_boost_kj_nk`` audit array cheap:
    it recovers that array as a plain subtraction, ``returned - u_nk``,
    rather than recomputing the boost a second time).

    Returns ``u_nk`` unchanged when no state carries ``gamd_lambda > 0``
    (the common case: plain umbrella/REUS, plain GaMD) -- this is the only
    branch that is a true no-op, including on ``meta``, which still gets
    ``meta["gamd_ladder"] = False`` and ``meta["gamd_ladder_samples_without_raw_energies"]
    = 0`` set for a uniform contract across every caller.

    When some state DOES carry ``gamd_lambda > 0``:

    - raises ``ValueError`` naming ``v_pep`` when EVERY sample lacks a
      finite ``(v_pep, v_dih)`` pair -- there is then no data anywhere to
      reweight with, and returning zero boost for every lambda>0 state
      would be scientifically wrong, not merely incomplete;
    - raises ``ValueError`` naming ``v_pep`` when ``envelope`` is ``None``
      (no frozen envelope was found/supplied) -- a ladder rung cannot be
      reweighted without the raw energies it was measured with AND the
      envelope that turns them into a boost;
    - otherwise adds ``beta * pep_gamd_boost_matrix_kj(v_pep, v_dih,
      state_lambdas, envelope).T`` onto ``u_nk``, except that any
      individual sample whose own ``v_pep`` or ``v_dih`` is non-finite gets
      NaN -- not a fabricated 0.0 boost, which is what
      ``pep_gamd_boost_matrix_kj``'s ``_channel_boost`` silently produces
      for a NaN input (``NaN < threshold`` is ``False``, so the ``np.where``
      guard falls through to 0.0) -- in every state column with
      ``gamd_lambda > 0``. This is exclusion-by-propagation, the same
      convention ``gareus.query.reconstruct_bias_matrix``'s docstring
      describes for a missing ``cv2`` under a secondary-restrained window;
      ``clean()`` (``gareus/mbar_analysis/data.py``) then drops those
      (sample, state) rows for the affected states downstream.
      ``meta["gamd_ladder_samples_without_raw_energies"]`` records how many
      samples were affected.
    """
    state_lambdas = np.asarray(state_lambdas, dtype=np.float64)
    lam_active = state_lambdas > 0.0
    if not np.any(lam_active):
        meta["gamd_ladder"] = False
        meta["gamd_ladder_samples_without_raw_energies"] = 0
        return u_nk

    v_pep = np.asarray(v_pep, dtype=np.float64)
    v_dih = np.asarray(v_dih, dtype=np.float64)
    missing = ~np.isfinite(v_pep) | ~np.isfinite(v_dih)
    if missing.all():
        raise ValueError(
            "some state carries gamd_lambda > 0 but no sample has a finite v_pep/v_dih "
            "pair; the λ-ladder cannot be reweighted without the raw channel energies"
        )
    if envelope is None:
        raise ValueError(
            "some state carries gamd_lambda > 0 but no frozen GaMD envelope was "
            "found/supplied; v_pep/v_dih cannot be reweighted under the ladder without it"
        )

    from gareus.pep_gamd import pep_gamd_boost_matrix_kj
    # (n_states, n_samples) -> (n_samples, n_states), matching u_nk's convention.
    boost_kj_nk = pep_gamd_boost_matrix_kj(v_pep, v_dih, state_lambdas, envelope).T.copy()
    n_missing = int(np.count_nonzero(missing))
    if n_missing:
        boost_kj_nk[np.ix_(missing, lam_active)] = np.nan

    meta["gamd_ladder"] = True
    meta["gamd_ladder_samples_without_raw_energies"] = n_missing
    return np.asarray(u_nk, dtype=np.float64) + float(beta) * boost_kj_nk
