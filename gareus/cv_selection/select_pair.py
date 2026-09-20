"""Choose CV2 for a fixed anchor CV1 from the swarm's unbiased frames, deterministically.

The rule (plan Task 5, after adversarial review):

1. The configured anchor must be deployable (:func:`anchor.score_anchor`).
2. Build a frame-level geometric partition of torsion + shape features with the
   anchor *excluded*, and balance the design measure over its coarse cells.
3. Fit the residual components under that measure.
4. Per component: held-out nonlinear R²(z2 | z1) gated on ``mean + 2·SE`` over
   folds; incremental information about the fine partition; the curvature the
   CV2 umbrella induces along the anchor as a fraction of the CV1 stiffness.
5. Winner: the deployable component with the largest gain, provided that gain
   clears ``min_gain_nats``. Below the floor every candidate is noise and the
   tie-break would crown PC1 while the report read as principled; instead the
   answer is ``cv1_only`` with the reason spelled out.
6. Stability: re-run on two disjoint halves of the seed families and record
   whether the same component wins. Recorded, not gating.

Everything measured is about the swarm's balanced design measure. Nothing is
a statement about the 300 K equilibrium ensemble.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import numpy as np

from . import contracts as C
from .anchor import AnchorCandidate, AnchorScore, score_anchor
from .independence import (DEFAULT_FOLD_SEED, frame_partition, heldout_nonlinear_r2,
                           incremental_cell_information)
from .models import (ResidualFit, coupling_curvature_kcal, evaluate_component,
                     fit_residual_components, to_candidate_set)
from .pair_model import PAIR_MODEL_VERSION, PairModel
from .protocol import NATIVE_BLIND_GENERATOR_PRESETS

DESIGN_MEASURE = "balanced_frame_cells"


@dataclass(frozen=True)
class SwarmDataset:
    features: np.ndarray            # (n, d) canonical torsion features
    feature_schema: C.FeatureSchema
    anchor: AnchorCandidate         # the configured contact CV, values aligned to rows
    shape_features: np.ndarray      # (n, m): rg_nm, e2e_nm -- for the partition only
    groups: np.ndarray              # seed_id per row (fold grouping)
    weights: Optional[np.ndarray] = None   # ignored: the selector balances over its own cells


@dataclass(frozen=True)
class SelectionConfig:
    residual_degree: int = 1
    components: tuple = (1, 2, 3, 4, 5, 6)
    max_nonlinear_r2: float = 0.20          # gate on mean + 2*SE over folds
    n_cells_coarse: int = 8
    n_cells_fine: int = 24
    k1_kcal_reference: float = 1000.0        # contact_adaptive_max_k_kcal, the run's CV1 ceiling
    k2_kcal_reference: float = 1.0           # ~RT/(spacing/1.5)^2 for a standardised z2 with 4 windows
    max_coupling_fraction: float = 0.25
    min_gain_nats: float = 0.02              # an order of magnitude above the ~0.003 smoothing floor
    min_windows_cv1: int = 4
    temperature_k: float = 300.0
    n_folds: int = 4


@dataclass(frozen=True)
class PairSelection:
    status: str                              # "pair" | "cv1_only" | "no_deployable_anchor"
    pair_model: Optional[PairModel]
    candidate_set: Optional[C.CandidateSet]
    anchor: AnchorScore
    report: dict = field(default_factory=dict)


def balanced_weights(labels) -> np.ndarray:
    """Equal total mass per label, then equal mass per row within a label."""
    labels = np.asarray(labels)
    unique, inverse = np.unique(labels, return_inverse=True)
    counts = np.bincount(inverse, minlength=unique.size)
    return 1.0 / (unique.size * counts[inverse])


def _units(kind: str) -> str:
    return "dimensionless" if kind == "nonlocal-contact-fraction" else "nanometer"


def _score_components(fit: ResidualFit, X, a, z1, cells_fine, groups, cfg: SelectionConfig) -> dict[int, dict]:
    scores: dict[int, dict] = {}
    for j in cfg.components:
        if not 1 <= int(j) <= fit.n_components:
            continue
        z2 = evaluate_component(fit, int(j), X, a)
        r2, folds = heldout_nonlinear_r2(z2, z1, groups, n_folds=cfg.n_folds)
        r2_se = float(folds.std(ddof=1) / np.sqrt(folds.size)) if folds.size > 1 else 0.0
        r2_gate = float(folds.mean() + 2.0 * r2_se)
        info = incremental_cell_information(cells_fine, z1, z2, groups, n_folds=cfg.n_folds)
        coupling = coupling_curvature_kcal(fit, int(j), cfg.k2_kcal_reference)
        fraction = coupling / cfg.k1_kcal_reference if cfg.k1_kcal_reference > 0 else float("inf")
        reasons = []
        if r2_gate > cfg.max_nonlinear_r2:
            reasons.append(f"nonlinear R2(z2|z1) gate {r2_gate:.3f} > {cfg.max_nonlinear_r2}")
        if fraction > cfg.max_coupling_fraction:
            reasons.append(f"coupling into CV1 {fraction:.3f} of k1 > {cfg.max_coupling_fraction}")
        scores[int(j)] = {
            "r2_pooled": float(r2), "r2_mean": float(folds.mean()), "r2_se": r2_se, "r2_gate": r2_gate,
            "gain_nats": info["gain"], "l1": info["l1"], "l2": info["l2"], "l12": info["l12"],
            "coupling_curvature_kcal": float(coupling), "coupling_fraction_of_k1": float(fraction),
            "deployable": not reasons, "reasons": reasons,
        }
    return scores


def _pick(scores: Mapping[int, dict], cfg: SelectionConfig) -> tuple[Optional[int], str]:
    deployable = {j: s for j, s in scores.items() if s["deployable"]}
    if not deployable:
        return None, "no component passed the nonlinear-R2 and coupling gates"
    best = max(deployable, key=lambda j: (deployable[j]["gain_nats"], -j))
    if deployable[best]["gain_nats"] < cfg.min_gain_nats:
        return None, (f"no component adds information about the discovery partition "
                      f"(max gain {deployable[best]['gain_nats']:.4f} < {cfg.min_gain_nats:.3f})")
    return best, "max held-out incremental information among deployable components"


def _half_split_winners(X, a, shape, groups, cfg: SelectionConfig) -> tuple[Optional[int], Optional[int]]:
    unique = np.unique(groups)
    unique = unique[np.random.default_rng(DEFAULT_FOLD_SEED).permutation(unique.size)]
    halves = (unique[: unique.size // 2], unique[unique.size // 2:])
    winners = []
    for members in halves:
        rows = np.isin(groups, members)
        if np.unique(groups[rows]).size < cfg.n_folds:
            winners.append(None)
            continue
        try:
            cells_c, _ = frame_partition(np.column_stack([X[rows], shape[rows]]), n_cells=cfg.n_cells_coarse)
            cells_f, _ = frame_partition(np.column_stack([X[rows], shape[rows]]), n_cells=cfg.n_cells_fine)
            fit = fit_residual_components(X[rows], a[rows], degree=cfg.residual_degree,
                                          weights=balanced_weights(cells_c))
            z1 = (a[rows] - fit.anchor_mean) / fit.anchor_std
            winner, _ = _pick(_score_components(fit, X[rows], a[rows], z1, cells_f, groups[rows], cfg), cfg)
            winners.append(winner)
        except ValueError:
            winners.append(None)
    return winners[0], winners[1]


def select_cv_pair(data: SwarmDataset, config: SelectionConfig, *, physical_system_sha256: str,
                   training_rows_sha256: str, library_versions: Mapping[str, str],
                   genpept_preset: str) -> PairSelection:
    # A fold-biased GENPEPT preset (e.g. "chignolin") is accepted and RECORDED, not refused:
    # the selection then rests on a library that knows the fold, and the pair model says so
    # (``genpept_preset``) so no downstream reader can mistake it for an ab initio run.
    # ``NATIVE_BLIND_GENERATOR_PRESETS`` still names the presets that ARE native-blind.
    native_blind_library = genpept_preset in NATIVE_BLIND_GENERATOR_PRESETS
    X = np.asarray(data.features, dtype=np.float64)
    a = np.asarray(data.anchor.values, dtype=np.float64)
    shape = np.asarray(data.shape_features, dtype=np.float64)
    groups = np.asarray(data.groups)
    if not (X.shape[0] == a.shape[0] == shape.shape[0] == groups.shape[0]):
        raise ValueError("features, anchor values, shape features and groups must have equal length")
    if X.shape[1] != data.feature_schema.width:
        raise ValueError(f"feature width {X.shape[1]} != schema width {data.feature_schema.width}")

    anchor = score_anchor(data.anchor, temperature_k=config.temperature_k,
                          k_max_kcal=config.k1_kcal_reference, min_windows=config.min_windows_cv1)
    report: dict[str, Any] = {"genpept_preset": genpept_preset, "native_blind_library": bool(native_blind_library),
                              "anchor": {"kind": anchor.kind, "n_resolvable": anchor.n_resolvable,
                                         "dynamic_range": anchor.dynamic_range,
                                         "deployable": anchor.deployable, "reasons": list(anchor.reasons)},
                              "config": {k: (list(v) if isinstance(v, tuple) else v)
                                         for k, v in vars(config).items()},
                              "design_measure": DESIGN_MEASURE, "n_frames": int(X.shape[0]),
                              "n_seed_families": int(np.unique(groups).size)}
    if not anchor.deployable:
        report["status"] = "no_deployable_anchor"
        return PairSelection("no_deployable_anchor", None, None, anchor, report)

    partition_features = np.column_stack([X, shape])          # the anchor is deliberately absent
    cells_coarse, _ = frame_partition(partition_features, n_cells=config.n_cells_coarse)
    cells_fine, _ = frame_partition(partition_features, n_cells=config.n_cells_fine)
    weights = balanced_weights(cells_coarse)
    fit = fit_residual_components(X, a, degree=config.residual_degree, weights=weights)
    z1 = (a - fit.anchor_mean) / fit.anchor_std

    primary_definition = {"kind": anchor.kind, "units": _units(anchor.kind),
                          "definition": dict(anchor.definition)}
    candidate_set = to_candidate_set(fit, data.feature_schema, primary_definition,
                                     physical_system_sha256, training_rows_sha256, library_versions)

    scores = _score_components(fit, X, a, z1, cells_fine, groups, config)
    winner, why = _pick(scores, config)
    report["components"] = {str(j): s for j, s in scores.items()}
    report["selection_reason"] = why
    runner_ups = [{"component_index": j,
                   "reason": "; ".join(s["reasons"]) if s["reasons"]
                   else f"lower information gain than component {winner}",
                   "scores": {k: v for k, v in s.items() if k not in ("reasons", "deployable")}}
                  for j, s in sorted(scores.items()) if j != winner]
    report["runner_ups"] = runner_ups
    if winner is None:
        report["status"] = "cv1_only"
        return PairSelection("cv1_only", None, candidate_set, anchor, report)

    z2 = evaluate_component(fit, winner, X, a)
    _, folds_rev = heldout_nonlinear_r2(z1, z2, groups, n_folds=config.n_folds)
    half_a, half_b = _half_split_winners(X, a, shape, groups, config)
    best = scores[winner]
    certificate = {
        "cov_q_weighted": float(np.sum(weights * z1 * z2)),
        "r2_z2_given_z1_mean": best["r2_mean"], "r2_z2_given_z1_se": best["r2_se"],
        "r2_z1_given_z2_mean": float(folds_rev.mean()),
        "coupling_curvature_kcal": best["coupling_curvature_kcal"],
        "coupling_fraction_of_k1": best["coupling_fraction_of_k1"],
        "std_unweighted_z2": float(z2.std()),
        "design_measure": f"{DESIGN_MEASURE}_{config.n_cells_coarse}",
        "n_frames": int(X.shape[0]), "n_seed_families": int(np.unique(groups).size),
        "half_split_agrees": bool(half_a == half_b == winner),
        "selected_gain_nats": best["gain_nats"],
        "max_gain_nats": max(s["gain_nats"] for s in scores.values()),
    }
    report["half_split_winners"] = [half_a, half_b]
    pair_model = PairModel.from_mapping({
        "schema": PAIR_MODEL_VERSION,
        "candidate_set_sha256": candidate_set.sha256,
        "feature_schema_sha256": data.feature_schema.sha256,
        "anchor": primary_definition,
        "selected_component_index": int(winner),
        "degree": int(config.residual_degree),
        "certificate": certificate,
        "runner_ups": runner_ups,
        "genpept_preset": genpept_preset,
    })
    report["status"] = "pair"
    report["pair_model_sha256"] = pair_model.sha256
    return PairSelection("pair", pair_model, candidate_set, anchor, report)
