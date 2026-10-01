"""Choose CV2 for a fixed anchor CV1 from the swarm's unbiased frames, deterministically.

The rule (plan Task 5, after adversarial review):

1. The configured anchor must be deployable (:func:`anchor.score_anchor`).
2. Build a frame-level geometric partition of torsion + shape features with the
   anchor *excluded*, and balance the design measure over its coarse cells.
3. Fit the residual components under that measure.
4. Per component: held-out nonlinear R²(z2 | z1) gated on ``mean + 2·SE`` over
   folds; incremental information about the fine partition; the curvature the
   CV2 umbrella induces along the anchor as a fraction of the CV1 stiffness.
5. Winner, ``ranking="gain"`` (the legacy rule, PCA components only): the deployable
   component with the largest gain, provided that gain clears ``min_gain_nats``. Below
   the floor every candidate is noise and the tie-break would crown PC1 while the report
   read as principled; instead the answer is ``cv1_only`` with the reason spelled out.
6. Stability (gain ranking): re-run on two disjoint halves of the seed families and
   record whether the same component wins. Recorded, not gating.

``ranking="slowness"`` (the default) adds the conditional tICA modes of the same residual
(:mod:`.slowness`) as components 7-9 and replaces steps 5-6: a candidate must also clear
the gain floor, a minimum lag autocorrelation at fixed CV1 and a bimodality coefficient at
fixed CV1, and must be reproducible: both seed-family halves, refitted from scratch, must
each contain a candidate that correlates with it (|r| >= ``half_split_min_corr`` on all
rows). The slowest reproducible candidate wins; none gives ``cv1_only``. Reproducibility is
checked per candidate, not on the argmax: near-equal slow modes swap rank between halves
(chignolin_9: psi(P4)+psi(D3) reproduced at |r| 0.93/0.85 but ranked second by 0.02 in one
half), and a gate on the argmax would refuse a stable coordinate over that tie. It needs trajectory order (``member_ids``, ``frame_index``,
``frame_dt_ps``); without it the gain rule runs and the report says why.

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
                     fit_residual_components, to_candidate_set, standardised_anchor)
from .slowness import (anchor_cells, conditional_autocorrelation, fit_conditional_tica,
                       max_bimodality_at_fixed_anchor)
from .pair_model import CERTIFICATE_VERSION_V2, PAIR_MODEL_VERSION_V2, PairModel
from .protocol import NATIVE_BLIND_GENERATOR_PRESETS

DESIGN_MEASURE = "balanced_frame_cells"
PICK_RULES = ("breadth-tie-slowest", "slowest")


@dataclass(frozen=True)
class SwarmDataset:
    features: np.ndarray            # (n, d) canonical torsion features
    feature_schema: C.FeatureSchema
    anchor: AnchorCandidate         # the configured contact CV, values aligned to rows
    shape_features: np.ndarray      # (n, m): rg_nm, e2e_nm -- for the partition only
    groups: np.ndarray              # seed_id per row (fold grouping)
    weights: Optional[np.ndarray] = None   # ignored: the selector balances over its own cells
    member_ids: Optional[np.ndarray] = None    # swarm member per row (trajectory identity)
    frame_index: Optional[np.ndarray] = None   # frame number within that member's trace
    frame_dt_ps: Optional[float] = None        # time between consecutive frames


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
    ranking: str = "slowness"                # "slowness" | "gain" (legacy, PCA only)
    tica_lag_ps: float = 50.0                # conditional tICA lag
    slowness_lag_ps: float = 200.0           # lag of the ranking autocorrelation
    n_tica_components: int = C.MAX_TICA_COMPONENTS
    min_slowness_rho: float = 0.72           # ~exp(-1/3): implied timescale >= 3 lags
    # Slowness ranking: the gain floor is tested on the gain averaged over this many
    # seed-family-to-fold assignments, failing only when mean + 2 sd < min_gain_nats
    # (demonstrably uninformative). 0 = the single historical assignment (rule before
    # 2026-10-01). The single draw moves by +-0.02-0.05 nats on chignolin's swarm.
    gain_resamples: int = 8
    # Slowness ranking, among deployable candidates: "breadth-tie-slowest" (default since
    # 2026-10-01) takes every candidate whose fold-averaged gain is within breadth_tie_sd
    # combined sd of the broadest one, then the slowest of that tie-set; "slowest" ignores
    # breadth (the rule before). Breadth separates the top candidates by less than its own
    # noise on chignolin's swarm (0.07 nats vs sd 0.14 at 112 seed families) while the
    # slowness order is stable, so breadth defines the set and slowness decides inside it.
    pick_rule: str = "breadth-tie-slowest"
    breadth_tie_sd: float = 1.0
    min_bimodality: float = 5.0 / 9.0        # Sarle's coefficient of a uniform distribution
    half_split_min_corr: float = 0.8
    n_cells_slowness: int = 8

    def __post_init__(self):
        if int(self.gain_resamples) < 0:
            raise ValueError("gain_resamples must be >= 0")
        if self.pick_rule not in PICK_RULES:
            raise ValueError(f"pick_rule must be one of {PICK_RULES}, got {self.pick_rule!r}")
        if not float(self.breadth_tie_sd) >= 0.0:
            raise ValueError("breadth_tie_sd must be >= 0")


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
    from .anchor_spec import ANCHOR_KINDS, units
    if kind in ANCHOR_KINDS:            # the units the runtime force evaluates the anchor in
        return units(kind)
    return "dimensionless" if kind == "nonlocal-contact-fraction" else "nanometer"


@dataclass(frozen=True)
class _TimeContext:
    """Trajectory order for the rows being scored (a subset keeps its own rows)."""
    members: np.ndarray
    frames: np.ndarray
    tica_lag: int
    slowness_lag: int

    def subset(self, rows) -> "_TimeContext":
        return _TimeContext(self.members[rows], self.frames[rows], self.tica_lag, self.slowness_lag)


def _time_context(data: SwarmDataset, cfg: SelectionConfig) -> tuple[Optional[_TimeContext], str]:
    if data.member_ids is None or data.frame_index is None or not data.frame_dt_ps or data.frame_dt_ps <= 0:
        return None, "no trajectory order (member_ids, frame_index, frame_dt_ps) in the swarm dataset"
    members, frames = np.asarray(data.member_ids), np.asarray(data.frame_index)
    if members.shape[0] != np.asarray(data.features).shape[0] or frames.shape != members.shape:
        raise ValueError("member_ids and frame_index must have one entry per row")
    dt = float(data.frame_dt_ps)
    return _TimeContext(members, frames.astype(np.int64), max(1, int(round(cfg.tica_lag_ps / dt))),
                        max(1, int(round(cfg.slowness_lag_ps / dt)))), ""


def _fit(X, a, cfg: SelectionConfig, weights, time: Optional[_TimeContext]) -> tuple[ResidualFit, list[str]]:
    """Residual PCA, plus conditional tICA modes when ranking by slowness."""
    fit = fit_residual_components(X, a, degree=cfg.residual_degree, weights=weights)
    notes = []
    if time is not None and cfg.n_tica_components > 0:
        try:
            fit = fit_conditional_tica(fit, X, a, time.members, time.frames, lag=time.tica_lag,
                                       n_modes=cfg.n_tica_components, weights=weights)
        except ValueError as exc:
            notes.append(f"conditional tICA candidates skipped: {exc}")
    return fit, notes


def _score_components(fit: ResidualFit, X, a, z1, cells_fine, groups, cfg: SelectionConfig,
                      time: Optional[_TimeContext] = None, weights=None) -> dict[int, dict]:
    scores: dict[int, dict] = {}
    slow_cells = anchor_cells(a, cfg.n_cells_slowness) if time is not None else None
    for j in range(1, fit.n_components + 1):
        family = fit.families[j - 1]
        if family == C.COMPONENT_FAMILY_PCA and j not in cfg.components:
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
        row = {
            "r2_pooled": float(r2), "r2_mean": float(folds.mean()), "r2_se": r2_se, "r2_gate": r2_gate,
            "gain_nats": info["gain"], "l1": info["l1"], "l2": info["l2"], "l12": info["l12"],
            "coupling_curvature_kcal": float(coupling), "coupling_fraction_of_k1": float(fraction),
        }
        if time is not None or family != C.COMPONENT_FAMILY_PCA:   # gain-mode output stays as before
            row["family"] = family
        if family == C.COMPONENT_FAMILY_TICA:
            row["tica_eigenvalue"] = float(fit.tica_eigenvalues[j - 1])
            row["tica_lag_frames"] = int(fit.tica_lag_frames[j - 1])
        if time is not None:
            rho = conditional_autocorrelation(z2, slow_cells, time.members, time.frames,
                                              time.slowness_lag, weights=weights)
            bim = max_bimodality_at_fixed_anchor(z2, slow_cells)
            row["slowness_rho"] = float(rho)
            row["slowness_lag_frames"] = int(time.slowness_lag)
            row["bimodality_max"] = float(bim)
            if cfg.gain_resamples > 0:
                draws = [info["gain"]] + [
                    incremental_cell_information(cells_fine, z1, z2, groups, n_folds=cfg.n_folds,
                                                 fold_seed=DEFAULT_FOLD_SEED + k)["gain"]
                    for k in range(1, int(cfg.gain_resamples))]
                g_mean, g_sd = float(np.mean(draws)), float(np.std(draws))
                row["gain_nats_mean"], row["gain_nats_sd"] = g_mean, g_sd
                row["gain_resamples"] = len(draws)
                row["gain_pass_fraction"] = float(np.mean(np.asarray(draws) >= cfg.min_gain_nats))
                if g_mean + 2.0 * g_sd < cfg.min_gain_nats:
                    reasons.append(f"information gain {g_mean:.4f} +- {g_sd:.4f} over {len(draws)} fold "
                                   f"assignments: mean + 2 sd < {cfg.min_gain_nats:.3f}")
            elif info["gain"] < cfg.min_gain_nats:
                reasons.append(f"information gain {info['gain']:.4f} < {cfg.min_gain_nats:.3f}")
            if not np.isfinite(rho) or rho < cfg.min_slowness_rho:
                reasons.append(f"slowness rho {rho:.3f} < {cfg.min_slowness_rho} at lag {time.slowness_lag} frames")
            if not np.isfinite(bim) or bim < cfg.min_bimodality:
                reasons.append(f"bimodality at fixed CV1 {bim:.3f} < {cfg.min_bimodality:.3f}")
        row.update({"deployable": not reasons, "reasons": reasons})
        scores[int(j)] = row
    return scores


def breadth_tie_set(deployable: Mapping[int, dict], cfg: SelectionConfig) -> set:
    """Deployable candidates whose breadth is indistinguishable from the broadest one.

    Breadth = the fold-averaged information gain (``gain_nats_mean`` +- ``gain_nats_sd``).
    j joins when mean_j >= mean_best - breadth_tie_sd * sqrt(sd_best^2 + sd_j^2). Without
    resampled gains (gain_resamples 0) or with pick_rule "slowest" every candidate is in the set.
    """
    if cfg.pick_rule != "breadth-tie-slowest" or not deployable or not all(
            "gain_nats_mean" in s for s in deployable.values()):
        return set(deployable)
    mean = {j: float(s["gain_nats_mean"]) for j, s in deployable.items()}
    sd = {j: float(s["gain_nats_sd"]) for j, s in deployable.items()}
    broad = max(deployable, key=lambda j: (mean[j], -j))
    k = float(cfg.breadth_tie_sd)
    return {j for j in deployable if mean[j] >= mean[broad] - k * float(np.hypot(sd[broad], sd[j]))}


def _pick(scores: Mapping[int, dict], cfg: SelectionConfig, ranking: str = "gain") -> tuple[Optional[int], str]:
    deployable = {j: s for j, s in scores.items() if s["deployable"]}
    if ranking == "slowness":
        if not deployable:
            return None, ("no component passed the R2, coupling, gain, slowness, bimodality and "
                          "seed-half reproducibility gates at fixed CV1")
        tie = breadth_tie_set(deployable, cfg)
        best = max(tie, key=lambda j: (deployable[j]["slowness_rho"], -j))
        if len(tie) < len(deployable):
            return best, (f"slowest of the breadth tie-set {sorted(tie)} (fold-averaged gain within "
                          f"{cfg.breadth_tie_sd:g} sd of the broadest) among deployable {sorted(deployable)}")
        return best, "slowest reproducible deployable component at fixed CV1 (lag autocorrelation)"
    if not deployable:
        return None, "no component passed the nonlinear-R2 and coupling gates"
    best = max(deployable, key=lambda j: (deployable[j]["gain_nats"], -j))
    if deployable[best]["gain_nats"] < cfg.min_gain_nats:
        return None, (f"no component adds information about the discovery partition "
                      f"(max gain {deployable[best]['gain_nats']:.4f} < {cfg.min_gain_nats:.3f})")
    return best, "max held-out incremental information among deployable components"


def _half_split_winners(X, a, shape, groups, cfg: SelectionConfig) -> tuple[Optional[int], Optional[int]]:
    """Legacy (gain ranking): refit and re-pick on each half; recorded, not gating."""
    winners = []
    for members in _seed_halves(groups):
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


def _seed_halves(groups) -> tuple[np.ndarray, np.ndarray]:
    unique = np.unique(groups)
    unique = unique[np.random.default_rng(DEFAULT_FOLD_SEED).permutation(unique.size)]
    return unique[: unique.size // 2], unique[unique.size // 2:]


def _half_fits(X, a, shape, groups, cfg: SelectionConfig,
               time: Optional[_TimeContext]) -> list[Optional[ResidualFit]]:
    """Refit (PCA + conditional tICA) from scratch on each half of the seed families."""
    out: list[Optional[ResidualFit]] = []
    for members in _seed_halves(groups):
        rows = np.isin(groups, members)
        try:
            cells_c, _ = frame_partition(np.column_stack([X[rows], shape[rows]]), n_cells=cfg.n_cells_coarse)
            fit, _ = _fit(X[rows], a[rows], cfg, balanced_weights(cells_c),
                          time.subset(rows) if time is not None else None)
            out.append(fit)
        except ValueError:
            out.append(None)
    return out


def _reproducibility(fit: ResidualFit, halves: list, X, a, scores: dict, cfg: SelectionConfig) -> None:
    """Gate every candidate on being reproduced by some candidate of each half (in place)."""
    half_z = [None if h is None else [evaluate_component(h, k, X, a) for k in range(1, h.n_components + 1)]
              for h in halves]
    for j, row in scores.items():
        z = evaluate_component(fit, j, X, a)
        corrs = [None if hz is None else float(max(abs(np.corrcoef(zk, z)[0, 1]) for zk in hz)) for hz in half_z]
        row["half_split_abs_corr"] = corrs
        if any(c is None or not c >= cfg.half_split_min_corr for c in corrs):
            row["reasons"].append(f"not reproduced in both seed-family halves (|r| {corrs}, need >= "
                                  f"{cfg.half_split_min_corr})")
            row["deployable"] = False


def select_cv_pair(data: SwarmDataset, config: SelectionConfig, *, physical_system_sha256: str,
                   training_rows_sha256: str, library_versions: Mapping[str, str],
                   genpept_preset: str, deployment: Optional[Mapping[str, Any]] = None) -> PairSelection:
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
    if config.ranking not in ("slowness", "gain"):
        raise ValueError(f"ranking must be 'slowness' or 'gain', got {config.ranking!r}")
    time, no_time_reason = _time_context(data, config) if config.ranking == "slowness" else (None, "")
    ranking = "slowness" if time is not None else "gain"
    report["ranking"] = ranking
    if config.ranking == "slowness" and time is None:
        report["ranking_fallback_reason"] = no_time_reason
    fit, fit_notes = _fit(X, a, config, weights, time)
    if fit_notes:
        report["notes"] = fit_notes
    z1 = (a - fit.anchor_mean) / fit.anchor_std

    primary_definition = {"kind": anchor.kind, "units": _units(anchor.kind),
                          "definition": dict(anchor.definition)}
    design_measure_name = f"{DESIGN_MEASURE}_{config.n_cells_coarse}"
    candidate_set = to_candidate_set(fit, data.feature_schema, primary_definition,
                                     physical_system_sha256, training_rows_sha256, library_versions, design_measure=design_measure_name)

    scores = _score_components(fit, X, a, z1, cells_fine, groups, config, time, weights)
    if ranking == "slowness":
        _reproducibility(fit, _half_fits(X, a, shape, groups, config, time), X, a, scores, config)
    winner, why = _pick(scores, config, ranking)
    report["components"] = {str(j): s for j, s in scores.items()}
    report["selection_reason"] = why
    lower = "slowness" if ranking == "slowness" else "information gain"
    runner_ups = [{"component_index": j,
                   "reason": "; ".join(s["reasons"]) if s["reasons"]
                   else f"lower {lower} than component {winner}",
                   "scores": {k: v for k, v in s.items() if k not in ("reasons", "deployable")}}
                  for j, s in sorted(scores.items()) if j != winner]
    report["runner_ups"] = runner_ups
    if winner is None:
        report["status"] = "cv1_only"
        return PairSelection("cv1_only", None, candidate_set, anchor, report)

    z2 = evaluate_component(fit, winner, X, a)
    _, folds_rev = heldout_nonlinear_r2(z1, z2, groups, n_folds=config.n_folds)
    if ranking == "slowness":
        half_a = half_b = None                     # per-candidate reproducibility gated above
        agrees = True
        report["half_split"] = {"winner_abs_corr": scores[winner]["half_split_abs_corr"],
                                "min_corr": config.half_split_min_corr}
    else:
        half_a, half_b = _half_split_winners(X, a, shape, groups, config)
        agrees = bool(half_a == half_b == winner)
    best = scores[winner]
    t1 = standardised_anchor(fit, a)                                   # T(a): the fitted regressor
    certificate = {
        "certificate_version": CERTIFICATE_VERSION_V2,
        # exact by construction: the residual is orthogonal to the fitted regressors under the
        # training weights; the raw-anchor covariance is generally nonzero under a clip and is
        # reported, not certified (spec F02)
        "cov_transformed_anchor_weighted": float(np.sum(weights * z2 * (t1 - np.sum(weights * t1)))),
        "cov_raw_anchor_weighted": float(np.sum(weights * z1 * z2)),
        "r2_z2_given_z1_mean": best["r2_mean"], "r2_z2_given_z1_se": best["r2_se"],
        "r2_z1_given_z2_mean": float(folds_rev.mean()),
        "coupling_curvature_kcal": best["coupling_curvature_kcal"],
        "coupling_fraction_of_k1": best["coupling_fraction_of_k1"],
        "std_unweighted_z2": float(z2.std()),
        "design_measure": design_measure_name,
        "n_frames": int(X.shape[0]), "n_seed_families": int(np.unique(groups).size),
        "half_split_agrees": agrees,
        "selected_gain_nats": best["gain_nats"],
        "max_gain_nats": max(s["gain_nats"] for s in scores.values()),
    }
    report["half_split_winners"] = [half_a, half_b]
    binding = {"topology_sha256": None, "physical_system_sha256": None, "contact_pair_list_sha256": None,
               "deployable": False}
    if deployment:
        binding.update({k: deployment.get(k) for k in binding if k in deployment})
        if deployment.get("anchor_binding_sha256"):     # non-contact anchors (anchor_spec)
            binding["anchor_binding_sha256"] = deployment["anchor_binding_sha256"]
        anchor_key = "anchor_binding_sha256" if "anchor_binding_sha256" in binding else "contact_pair_list_sha256"
        binding["deployable"] = bool(deployment.get("deployable", all(
            binding[k] for k in ("topology_sha256", "physical_system_sha256", anchor_key))))
    pair_model = PairModel.from_mapping({
        "schema": PAIR_MODEL_VERSION_V2,
        "candidate_set_sha256": candidate_set.sha256,
        "feature_schema_sha256": data.feature_schema.sha256,
        "anchor": primary_definition,
        "selected_component_index": int(winner),
        "degree": int(config.residual_degree),
        "certificate": certificate,
        "runner_ups": runner_ups,
        "genpept_preset": genpept_preset,
        "basis_transform": dict(candidate_set.basis_transform),
        "deployment": binding,
    })
    report["deployable"] = bool(binding["deployable"])
    report["status"] = "pair"
    report["pair_model_sha256"] = pair_model.sha256
    return PairSelection("pair", pair_model, candidate_set, anchor, report)
