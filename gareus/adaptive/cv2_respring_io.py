"""Respring (``--ap-cv2-respring``): orchestration, disk I/O and the epoch-loop entry point.

``run_epoch_cv2_respring`` is what the adaptive driver calls, in numbered epochs only, right
after the 3.3 hook and only for an epoch that is not being recovered from the P3 applied-actions
ledger (a recovered epoch re-uses the ledger's actions and never proposes again).
``propose_respring`` is the pure orchestrator (tests and the replay script call it directly).
The decision rules are in ``cv2_respring``.

Report ``epoch_NNN/cv2_respring_report.json`` (schema ``cv2_respring_report_v1``)::

    schema_version, epoch, status ("ok" | "error"), error, stage ("numbered_epoch" | "replay"),
    settings   -- RespringSettings.as_record() (knobs, constants, sigma_target and its source),
    cap        -- {max_fraction, n_centres, limit, n_over_cap},
    candidates -- one per CV2-restrained representative-rung state:
        state_id, centre_state_ids (every rung of the centre), c1, k1, c2, k2, lambda,
        decision (proposed | no_action | skipped | refused | deferred), reason, refusal,
        triggered (whole c_real interval below the trigger), metrics {n, var, sd, sigma_w2,
        f2_prod, f2_prod_conditional, c_real, g, g_source (x5 | subsample), n_eff,
        subsample_stride, bootstrap {block_len, n_blocks, guard, block_len_requested},
        var_interval, c_interval, f2_interval, over_compressed},
        proposal {k2_old, k2_new, k2_ratio, predicted_c, predicted_c_interval,
                  predicted_sampled_sigma, min_mean_compression, sigma_target,
                  sigma_target_source, k2_cap, compression_floor_k2, refusal, gate_note?} | null,
    summary    -- cv2_respring.summarise (n_candidates, n_measured, n_triggered, n_proposed,
                  n_unresolved, n_over_compressed, decisions, reasons, c_real_quantiles),
    actions_added -- [{action "respring", state_id (old), k2_old, k2_new}],
    apply      -- (after the applier) {"refused": [...]}: respring actions the applier refused
                  (duplicate, mandatory, no_change, below_k_min, max_replicas_budget, ...);
                  each also counts into summary.n_unresolved.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from gareus.adaptive import cv2_respring as rs

def _atomic_json(path: Path, obj: Any) -> Path:
    from gareus.adaptive.cv2_resolution_io import _atomic_json as _aj  # noqa: PLC0415
    return _aj(path, obj)


def _clean(obj: Any) -> Any:
    from gareus.adaptive.cv2_resolution_io import _clean as _c  # noqa: PLC0415
    return _c(obj)


# ---- settings ------------------------------------------------------------------------------

def find_layout_plan(out_dir: Optional[Path], registry: Any) -> Tuple[Optional[dict], Optional[str]]:
    """The campaign's ``layout_plan.json`` (swarm analysis dir, else beside a state's source
    window table), as ``cv2_resolution_io.find_layout_reserve`` finds it."""
    cands: List[Path] = []
    if out_dir is not None:
        cands.append(Path(out_dir) / "swarm" / "analysis" / "layout_plan.json")
    for s in registry.all_states():
        src = (s.metadata or {}).get("source_csv")
        if src:
            cands.append(Path(src).parent / "layout_plan.json")
    for path in dict.fromkeys(cands):
        if path.exists():
            try:
                return json.loads(path.read_text()), str(path)
            except (OSError, ValueError) as exc:
                print(f"WARNING: {path} unreadable for the respring sigma target ({exc})")
    return None, None


def shape_targets(plan: Optional[Mapping[str, Any]]) -> Tuple[Optional[float], Optional[float]]:
    """(sigma_w_target, min_mean_compression) the 3.2 shape layout recorded, else (None, None)."""
    rec = (plan or {}).get("cv2_shape") or {}
    sigma = rs._finite(rec.get("sigma_w_target"))
    comp = None
    for col in rec.get("columns") or []:
        comp = rs._finite(((col or {}).get("placement") or {}).get("min_mean_compression"))
        if comp is not None:
            break
    return (sigma if sigma is not None and sigma > 0 else None), comp


def settings_from(policy: Any, args: Any, plan: Optional[Mapping[str, Any]] = None,
                  plan_source: Optional[str] = None) -> rs.RespringSettings:
    """Knobs from the (frozen) policy; floors, ceiling, temperature live from args; the sigma
    target and compression from the layout plan's ``cv2_shape`` record when present."""
    from gareus.adaptive_production import _args_temperature_k, _resolve_secondary_k_max  # noqa: PLC0415
    try:
        k2_min = float(getattr(args, "cv2_k_min", 0.0) or 0.0)
    except (TypeError, ValueError):
        k2_min = 0.0
    sigma, comp = shape_targets(plan)
    extra: Dict[str, Any] = {"temperature_k": _args_temperature_k(args),
                             "k2_min": k2_min if math.isfinite(k2_min) else 0.0,
                             "k2_max": _resolve_secondary_k_max(args)}
    if sigma is not None:
        extra.update(sigma_target=sigma, sigma_target_source=f"layout_plan cv2_shape.sigma_w_target ({plan_source})")
    if comp is not None and 0.0 < comp < 1.0:
        extra["min_mean_compression"] = comp
    return rs.RespringSettings.from_policy(policy, **extra)


# ---- views and measurements ----------------------------------------------------------------

def state_rows(registry: Any, diagnostics: Mapping[str, Any]) -> Tuple[List[Dict[str, Any]], int]:
    """(representative-rung views of every active state, number of centres)."""
    from gareus.adaptive_production import AdaptiveDecisionPolicy, _centre_group_key  # noqa: PLC0415
    from gareus.adaptive.cv2_resolution import representative_ids, state_views  # noqa: PLC0415
    views = state_views(registry, diagnostics)
    rows = {int(r.get("state_id")): r for r in diagnostics.get("states", []) or []}
    policy = AdaptiveDecisionPolicy()
    groups: Dict[Tuple, List[int]] = {}
    for s in registry.active_states():
        groups.setdefault(_centre_group_key(s, policy), []).append(int(s.state_id))
    out = []
    for sid in representative_ids(views):
        st, v = registry.get_state(sid), views[sid]
        members = sorted(groups.get(_centre_group_key(st, policy), [sid]))
        mand = any(bool((registry.get_state(m).metadata or {}).get("mandatory")) for m in members)
        pc = (rows.get(sid) or {}).get("paired_cv") or {}
        out.append({"state_id": sid, "c1": v.c1, "k1": v.k1, "c2": v.c2, "k2": v.k2, "lam": v.lam,
                    "created_epoch": v.created_epoch, "metadata": dict(v.metadata), "members": members,
                    "mandatory": mand, "paired_cv": pc})
    return out, len(representative_ids(views))


def _g_of(est: Any) -> Tuple[Optional[float], str]:
    if est is None:
        return None, "none"
    get = est.get if isinstance(est, Mapping) else (lambda k, d=None: getattr(est, k, d))
    if get("status") not in (None, "ok"):
        return None, f"x5_{get('status')}"
    g = rs._finite(get("g_cv2")) or rs._finite(get("g"))
    return (g, "x5") if g is not None and g > 0 else (None, "none")


def measure(view: Mapping[str, Any], sub: Optional[Mapping[str, np.ndarray]], est: Any,
            settings: rs.RespringSettings) -> Optional[Dict[str, Any]]:
    """The window's measurement from its P4 moments + subsample (None: no CV2 moments)."""
    pc = view.get("paired_cv") or {}
    m2 = pc.get("cv2") or {}
    var, n = rs._finite(m2.get("var")), int(m2.get("n") or 0)
    k2 = rs._finite(view.get("k2"))
    if var is None or var <= 0 or n < 2 or k2 is None or k2 <= 0:
        return None
    cond = None
    v1, cov = rs._finite((pc.get("cv1") or {}).get("var")), rs._finite(pc.get("cov_cv1_cv2"))
    if v1 and v1 > 0 and cov is not None and rs._finite(view.get("k1")):
        cond = var - cov * cov / v1
    z = src = step = None
    if sub is not None:
        z = np.asarray(sub.get("cv2"), dtype=float)
        keep = np.isfinite(z)
        z = z[keep]
        src = None if sub.get("source_index") is None else np.asarray(sub["source_index"])[keep]
        step = None if sub.get("step") is None else np.asarray(sub["step"])[keep]
    stride = int(((pc.get("subsample") or {}).get("stride")) or 1)
    g, g_src = _g_of(est)
    return rs.measure_window(var=var, n=n, k2=k2, rt=settings.rt, z=z, source_index=src, step=step,
                             stride=stride, g=g, g_source=g_src, seed=int(view["state_id"]), cond_var=cond)


def touched_state_ids(actions: Sequence[Tuple]) -> set:
    """State ids other actions of this epoch retire, split, insert at or respring."""
    return {int(a[1]) for a in actions if str(a[0]) in ("retire", "split", "insert", rs.ACTION)}


def propose_respring(registry: Any, diagnostics: Mapping[str, Any], actions: Sequence[Tuple],
                     settings: rs.RespringSettings, *, epoch: int, effective: Mapping[int, Any],
                     subsamples: Mapping[int, Mapping[str, np.ndarray]], gate: Any = None
                     ) -> Tuple[List[Tuple], Dict[str, Any]]:
    """(actions + this epoch's respring actions, report). Pure."""
    views, n_centres = state_rows(registry, diagnostics)
    touched = touched_state_ids(actions)
    cands = []
    for v in views:
        if v.get("c2") is None or not (rs._finite(v.get("k2")) or 0.0) > 0.0:
            continue                                # CV1-only / anchors: not a CV2 spring
        skip = rs.structural_skip(v, settings, epoch, touched)
        m = None if skip else measure(v, subsamples.get(int(v["state_id"])), effective.get(int(v["state_id"])),
                                      settings)
        cands.append(rs.evaluate(v, m, settings, epoch=epoch, touched=touched, gate=gate))
    cands, cap = rs.apply_cap(cands, n_centres, settings)
    new = [rs.action_of(c, epoch) for c in cands if c["decision"] == "proposed"]
    report = {"schema_version": rs.SCHEMA_VERSION, "epoch": int(epoch), "status": "ok", "error": None,
              "settings": settings.as_record(), "cap": cap, "candidates": cands,
              "summary": rs.summarise(cands),
              "actions_added": [{"action": rs.ACTION, "state_id": int(a[1]), "k2_old":
                                 a[4][rs.METADATA_KEY]["respring"]["k2_old"], "k2_new": a[2][3]} for a in new]}
    return [*actions, *new], _clean(report)


def _subsamples(diagnostics: Mapping[str, Any]) -> Dict[int, Dict[str, np.ndarray]]:
    from gareus.adaptive.cv2_resolution_io import _subsamples as _s  # noqa: PLC0415
    return _s(diagnostics)


def effective_for(registry: Any, sources: Sequence[Tuple[str, Path]], state_ids: Sequence[int]) -> Dict[int, Any]:
    """X5 g of each state's CV2 series over the phase's sample sources (CV2 axis only)."""
    from gareus.adaptive.effective_samples import state_effective_samples  # noqa: PLC0415
    from gareus.adaptive_production import _load_epoch_window_map  # noqa: PLC0415
    return state_effective_samples(sources, window_map_for=lambda d: _load_epoch_window_map(d, registry),
                                   restrained_axes={int(s): (False, True) for s in state_ids})


def run_epoch_cv2_respring(*, adaptive_dir: Path, epoch_dir: Path, epoch: int, registry: Any,
                           diagnostics: Mapping[str, Any], actions: Sequence[Tuple], policy: Any, args: Any,
                           out_dir: Optional[Path], phase_dirs: Sequence[Path], gate: Any = None) -> List[Tuple]:
    """The driver's hook. Never raises: on failure the report says ``error`` and the epoch's
    actions are returned unchanged."""
    path = Path(epoch_dir) / rs.REPORT_NAME
    try:
        from gareus.adaptive_production import _sample_sources_from_run_root  # noqa: PLC0415
        plan, plan_src = find_layout_plan(out_dir, registry)
        settings = settings_from(policy, args, plan, plan_src)
        views, _n = state_rows(registry, diagnostics)
        ids = [v["state_id"] for v in views if v.get("c2") is not None and (rs._finite(v.get("k2")) or 0) > 0]
        sources = [src for d in phase_dirs for src in _sample_sources_from_run_root(Path(d).name, Path(d))]
        effective = effective_for(registry, sources, ids) if ids else {}
        new_actions, report = propose_respring(registry, diagnostics, actions, settings, epoch=epoch,
                                               effective=effective, subsamples=_subsamples(diagnostics), gate=gate)
        report["stage"] = "numbered_epoch"
        _atomic_json(path, report)
        s = report["summary"]
        print(f"    CV2 respring: {s['n_proposed']} proposed of {s['n_triggered']} under-compressed "
              f"({s['n_measured']} measured, {s['n_unresolved']} unresolved) ({path.name})")
        return list(new_actions)
    except Exception as exc:          # never kill a completed epoch
        print(f"WARNING: CV2 respring failed for epoch {epoch} ({type(exc).__name__}: {exc}); no respring actions")
        try:
            _atomic_json(path, {"schema_version": rs.SCHEMA_VERSION, "epoch": int(epoch), "status": "error",
                                "error": f"{type(exc).__name__}: {exc}", "candidates": [],
                                "summary": rs.summarise([])})
        except OSError:
            pass
        return list(actions)


def annotate_report_with_refusals(epoch_dir: Path, actions: Sequence[Tuple],
                                  refused: Sequence[Mapping[str, Any]]) -> None:
    """Record the applier's refusals of this epoch's respring actions (``apply.refused``) and
    count them into ``summary.n_unresolved``."""
    path = Path(epoch_dir) / rs.REPORT_NAME
    if not path.exists():
        return
    try:
        report = json.loads(path.read_text())
        mine = []
        for r in refused:
            i = r.get("index")
            a = actions[int(i)] if i is not None and int(i) < len(actions) else None
            if a is not None and str(a[0]) == rs.ACTION:
                mine.append({**dict(r), "proposal": [a[0], int(a[1])]})
        report["apply"] = {"refused": mine}
        summ = report.setdefault("summary", {})
        summ["n_refused_by_applier"] = len(mine)
        base = sum(1 for c in report.get("candidates", []) or [] if c.get("triggered") and c.get("decision") != "proposed")
        summ["n_unresolved"] = base + len(mine)          # recomputed: annotating twice is idempotent
        _atomic_json(path, _clean(report))
    except (OSError, ValueError) as exc:
        print(f"WARNING: could not annotate {path} with the applier's refusals ({exc})")


__all__ = ["annotate_report_with_refusals", "effective_for", "find_layout_plan", "measure", "propose_respring",
           "run_epoch_cv2_respring", "settings_from", "shape_targets", "state_rows", "touched_state_ids"]
