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

import math
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
      describes for a missing ``cv2`` under a secondary-restrained window.

      Because ``u_nk`` is a dense matrix (no ragged per-state exclusion is
      representable), ``clean()`` (``gareus/mbar_analysis/data.py``, which
      masks on ``np.all(np.isfinite(d.u_nk), axis=1)``) does not merely drop
      the affected states for that sample -- it drops the WHOLE sample from
      the MBAR solve, from every state including any ``gamd_lambda == 0``
      one. This is weight-correct (``window``/``u_nk`` are sliced together,
      and every solver recomputes ``n_k`` from the post-``clean`` ``window``
      array via ``np.bincount``), just coarser than per-state exclusion:
      a sample that lacked raw energies contributes to the solve nowhere at
      all, not only at the rungs it couldn't be reweighted to.
      ``meta["gamd_ladder_samples_without_raw_energies"]`` records how many
      samples were affected -- equivalently, how many samples this eventually
      costs the solve once ``clean()`` runs.
    """
    state_lambdas = np.asarray(state_lambdas, dtype=np.float64)
    lam_active = state_lambdas > 0.0
    if not np.any(lam_active):
        meta["gamd_ladder"] = False
        meta["gamd_ladder_samples_without_raw_energies"] = 0
        return u_nk

    if envelope is None:
        raise ValueError(
            "some state carries gamd_lambda > 0 but no frozen GaMD envelope was "
            "found/supplied; v_pep/v_dih cannot be reweighted under the ladder without it"
        )
    v_pep = np.asarray(v_pep, dtype=np.float64)
    v_dih = np.asarray(v_dih, dtype=np.float64)
    # A single dihedral boost (envelope.has_total False) has no Total channel: v_pep is
    # NaN by construction there and is not "missing data" -- only v_dih must be finite.
    needs_pep = bool(getattr(envelope, "has_total", True))
    missing = ~np.isfinite(v_dih) | (~np.isfinite(v_pep) if needs_pep else np.zeros(v_dih.shape, dtype=bool))
    if missing.all():
        raise ValueError(
            "some state carries gamd_lambda > 0 but no sample has a finite "
            + ("v_pep/v_dih pair" if needs_pep else "v_dih value")
            + "; the λ-ladder cannot be reweighted without the raw channel energies"
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


def assert_lambda_sources_agree(state_lambdas, per_sample_lambda) -> None:
    """Raise when the registry says "no ladder" but the samples disagree.

    state_registry.csv is the only λ source the union loader reads. When it is
    all-zero the ladder boost is never folded into u_nk and MBAR is told 112
    states differ only by umbrella bias -- which is wrong, not merely noisy, and
    today produces a confident PASS. The per-sample gamd_lambda column is an
    independent witness; if it shows more than one rung, the registry is stale.
    """
    state_lambdas = np.asarray(state_lambdas, dtype=np.float64)
    if np.any(state_lambdas > 0.0) or per_sample_lambda is None:
        return
    per_sample = np.asarray(per_sample_lambda, dtype=np.float64)
    finite = per_sample[np.isfinite(per_sample)]
    if finite.size == 0:
        return
    distinct = np.unique(np.round(finite, 6))
    if distinct.size > 1 or float(distinct.max()) > 0.0:
        raise ValueError(
            "λ-ladder inconsistency: every state in state_registry.csv reads "
            f"gamd_lambda=0, but the samples carry {distinct.size} distinct rung(s) "
            f"up to λ={float(distinct.max()):.4f}. Analysing this run would silently "
            "treat a boosted ensemble as unboosted. Re-write the registry from a "
            "window table that carries gamd_lambda (see Task 1)."
        )


def mbar_state_overlap(u_nk: np.ndarray, f_k: np.ndarray, n_k: np.ndarray) -> np.ndarray:
    """MBAR state-overlap matrix ``O_ij`` for a solved set of states.

    ``W_nk = exp(f_k - u_nk) / sum_l N_l exp(f_l - u_nl)`` (evaluated by
    log-sum-exp, so a large ``u_nk`` cannot overflow), then

        ``O_ij = sum_n N_i W_ni W_nj``.

    This is the diagnostic the λ-ladder needs and a CV histogram cannot give:
    two rungs at one umbrella centre have CV overlap ~1 *by construction*
    whatever their boost spacing, so only the reduced potentials can say
    whether the rungs actually share phase space.

    Conventions, deliberately:

    - ``O = diag(N) @ S`` with ``S`` symmetric, so ``O_ij * N_j == O_ji * N_i``
      exactly, identical states give ``O_ij = N_i / (N_i + N_j)``, and the
      COLUMNS sum to 1 exactly (``sum_i N_i W_ni W_nj = sum_n W_nj = 1``).
      Rows sum to 1 only when every ``N_k`` is equal -- which is the usual
      ladder case and the case the S3 pilot calibration was measured in.
    - This differs from ``pymbar.MBAR.compute_overlap``, whose code
      (``self.N_k * (W.T @ W)``) broadcasts ``N_k`` over the COLUMNS and so
      returns ``N_j * S_ij`` despite its own docstring describing ``N_i``.
      Do not "fix" this to match pymbar: the properties above are what the
      rung diagnostics and their tests are written against.
    - ``O`` is invariant under a constant shift of ``f_k`` (the shift cancels
      between numerator and denominator), so any solver's gauge -- including
      ``solve_mbar``'s ``f_k[0] = 0`` -- is fine.

    ``u_nk`` is ``(n_samples, n_states)``, ``f_k`` and ``n_k`` are
    ``(n_states,)``.  States with ``N_k == 0`` contribute nothing to the
    denominator and get an all-zero row.
    """
    u_nk = np.asarray(u_nk, dtype=np.float64)
    f_k = np.asarray(f_k, dtype=np.float64).reshape(-1)
    n_k = np.asarray(n_k, dtype=np.float64).reshape(-1)
    if u_nk.ndim != 2 or u_nk.shape[1] != f_k.size or f_k.size != n_k.size:
        raise ValueError(
            f"mbar_state_overlap shape mismatch: u_nk {u_nk.shape}, f_k {f_k.shape}, n_k {n_k.shape}"
        )
    with np.errstate(divide="ignore"):
        log_n_k = np.where(n_k > 0.0, np.log(np.maximum(n_k, 1.0e-300)), -np.inf)
    # log W_nk = (f_k - u_nk) - logsumexp_l(log N_l + f_l - u_nl)
    log_num = f_k[None, :] - u_nk
    shifted = log_n_k[None, :] + log_num
    max_l = np.max(np.where(np.isfinite(shifted), shifted, -np.inf), axis=1, keepdims=True)
    max_l = np.where(np.isfinite(max_l), max_l, 0.0)
    log_denom = max_l + np.log(np.sum(np.exp(shifted - max_l), axis=1, keepdims=True))
    w_nk = np.exp(log_num - log_denom)
    return n_k[:, None] * (w_nk.T @ w_nk)


def symmetric_state_overlap(overlap: np.ndarray, i: int, j: int) -> Optional[float]:
    """The per-edge overlap metric ``sqrt(O_ij * O_ji)``.

    ``mbar_state_overlap`` above returns ``O = diag(N) @ S`` with ``S``
    symmetric, so ``O_ij != O_ji`` whenever ``N_i != N_j`` -- and unequal
    per-state sample counts are the normal case under adaptive extension, not
    the exception. The raw ``O[i, j]`` would make an edge's overlap depend on
    which of its two states happens to come first in the pair, which is
    unrelated to anything physical. The geometric mean is the symmetric
    combination, ``S_ij * sqrt(N_i * N_j)``, and it equals ``O_ij`` exactly
    when ``N_i == N_j``.

    Shared home for the convention: ``gareus.adaptive_production`` has its own
    ``_symmetric_state_overlap`` with the same formula and the same reasoning
    (see ``gareus/adaptive_production.py:3697-3725``), predating this one and
    not refactored here to avoid an import-cycle risk (``adaptive_production``
    already imports this module). Any *new* caller of ``mbar_state_overlap``
    -- e.g. ``gareus.mbar_analysis.ladder_overlap`` -- should call this
    function rather than hand-roll a third copy.
    """
    a = float(overlap[i, j])
    b = float(overlap[j, i])
    if not (math.isfinite(a) and math.isfinite(b)) or a < 0.0 or b < 0.0:
        return None
    return math.sqrt(a * b)
