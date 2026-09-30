"""Spec 3.3 R1-R3: orchestration, budget, disk I/O and the epoch-loop entry point.

``run_epoch_cv2_resolution`` is what the adaptive driver calls (only with
``--ap-cv2-resolution``) right after the existing proposers; ``propose_cv2_resolution`` is
the pure orchestrator it wraps (tests and the replay CLI call it directly).

Report ``epoch_NNN/cv2_resolution_report.json`` (schema ``cv2_resolution_report_v3``)::

    schema_version, epoch, status ("ok" | "error"), error, stage ("numbered_epoch" | "replay"),
    settings      -- every knob and constant used (ResolutionSettings.as_record),
    rules         -- {"R1"|"R2"|"R3": {"status": "ok"|"unavailable"|"error", "reason", ...,
                      "requested": bool (additive, 2026-09-30; see ``requested_rules``; a report
                      without it is graded as legacy/incomplete metadata)}},
    candidates    -- one record per edge (R1), interval (R2) or state (R3) considered:
        rule, kind ("edge"|"interval"|"state"), state_ids, class (R1: structural|weak|unmeasured),
        decision ("proposed"|"extend"|"flagged"|"refused"|"no_action"|"skipped"),
        reason, refusal (code or null), cost_states, metrics {...},
        proposal {parent_state_id, children: [{primary_center, primary_k, secondary_center, k2,
                  sigma_target, sigma_used, f2_est, k2_cap, predicted_sampled_sigma,
                  mean_compression, min_mean_compression, compression_floor_k2,
                  at_compression_floor, refusal, centred_on, gate_note?}]} | null,
    budget        -- {mode ("reserve"|"no_reserve"|"ignored"), n_rungs, resolution_slots,
                      spent_states, allowance {...}, add_rung {...}, reserve_source},
    actions_added -- [{action ("extend"|"add"|"insert"), state_id (parent or extended state),
                     rule}], actions_removed -- [{action [kind, id], reason}]: what this step did
                     to the epoch's action list,
    summary       -- {counts {rule: {decision: n}}, n_proposed, n_windows_at_compression_floor,
                      n_blocking (proposed only), n_refused_budget (no_reserve/resolution_budget),
                      n_refused_spring_cap (k2_capped_below_compression), n_trapped_or_orthogonal,
                      n_r3_flag_only (v3)},
    apply         -- (after the applier) {"refused": [...]} for this step's actions.

Refusal codes: no_reserve, resolution_budget, k2_at_floor, k2_capped_below_compression (the
4 x parent / cv2_k_max cap or the coupling gate holds k2 below the shape rule's mean-compression
floor), below_k_min (coupling gate), and the applier's own (duplicate, max_replicas_budget, ...)
under ``apply``. R3 metrics carry ``transitions`` (the count R3 decides on, estimator
``transitions_estimator``: replica-path by default since v3), ``transitions_replica``,
``transitions_replica_path`` and ``transitions_state_series`` (replica <= both others; those two
are not ordered, see cv2_resolution_rules._transitions), or, when no replica-resolved series
exists, ``transitions`` None and ``transitions_state_series_lower_bound`` (then R3 never inserts). Every R3 candidate carries ``r3_gate`` (``cv2_resolution.R3_GATES``:
the test that decided it) and ``r3_gate_values`` (the measured numbers behind it).

v2 (2026-09-30) vs v1: ``transitions_lower_bound`` renamed ``transitions_state_series_lower_bound``
(it bounds the state-series count from below and the replica count from above); summary
``n_blocking`` counts proposed candidates only (v1 added budget refusals) and gains
``n_refused_budget`` / ``n_refused_spring_cap``; R3 metrics gain ``r3_gate`` /
``r3_gate_values``; mixture components gain ``reg_variance`` / ``variance_curvature``. 
v3 (2026-09-30, follow-ups (h)) vs v2: R3 metrics gain ``transitions_replica``,
``transitions_replica_path`` and ``would_be`` {decision, refusal, r3_mode}; a passing R3 window
in ``--ap-refine-r3-mode flag`` (the default) is ``flagged`` with reason ``r3_flag_only`` (its
would-be children stay in ``proposal``) instead of ``proposed``; summary gains ``n_r3_flag_only``;
R2 metrics gain ``coverage_count``, ``n_contributing_windows_any`` / ``_same_column``
(``n_contributing_windows`` is the count the rule used) and ``rules.R2`` gains ``row_sources``,
``bootstrap`` (block rule record) and ``coverage_count``; settings gain ``refine_r3_mode``,
``coverage_count``, ``boot_block_g_multiple``, ``boot_min_blocks``. Readers
(``cv2_resolution_summary``, ``is_blocking``) accept v1-v3.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from gareus.adaptive import cv2_coverage as cov
from gareus.adaptive import cv2_resolution as cr
from gareus.adaptive import cv2_resolution_rules as rules
from gareus.adaptive.effective_samples import contiguous_runs
from gareus.adaptive.reserve_budget import reserve_allowances

UNION_NPZ_NAME = "topup_union_mbar.npz"
UNION_OVERLAP_NAME = "topup_union_overlap.json"
UNION_BYTES_PER_CELL = 8.0 * 6.0
_NO_RESERVE_WARNED: set = set()      # adaptive dirs already warned in this process (one per campaign job)


def reset_no_reserve_warnings() -> None:
    """Forget which campaigns were warned (tests)."""
    _NO_RESERVE_WARNED.clear()


def warn_no_reserve_once(adaptive_dir: Path, budget: Mapping[str, Any]) -> bool:
    """One WARNING per campaign job when no P1 reserve governs the resolution budget (every
    R1-R3 proposal is then refused ``no_reserve``). Returns whether it printed."""
    if (budget or {}).get("mode") != "no_reserve":
        return False
    key = str(Path(adaptive_dir).resolve())
    if key in _NO_RESERVE_WARNED:
        return False
    _NO_RESERVE_WARNED.add(key)
    reason = ((budget.get("allowance") or {}).get("reason")) or "no_reserve"
    print(f"WARNING: --ap-cv2-resolution is on but no P1 layout reserve governs this campaign ({reason}): "
          "every CV2-resolution action will be refused (no_reserve; recorded and CAUTION-graded, never "
          "convergence-blocking). Set --swarm-adaptive-reserve-fraction > 0 at swarm design (and "
          "--max-replicas > 0) to fund R1-R3.")
    return True


def _atomic_json(path: Path, obj: Any) -> Path:
    path = Path(path)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=_json_default))
    os.replace(tmp, path)
    return path


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _clean(obj: Any) -> Any:
    """JSON-safe copy: non-finite floats -> None."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return float(obj) if math.isfinite(float(obj)) else None
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def load_history(adaptive_dir: Path) -> Dict[str, Any]:
    path = Path(adaptive_dir) / cr.HISTORY_NAME
    if not path.exists():
        return {"schema_version": cr.HISTORY_SCHEMA, "edges": {}}
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        print(f"WARNING: unreadable {path} ({exc}); CV2-resolution edge history starts empty")
        return {"schema_version": cr.HISTORY_SCHEMA, "edges": {}}


def save_history(adaptive_dir: Path, history: Mapping[str, Any]) -> Path:
    return _atomic_json(Path(adaptive_dir) / cr.HISTORY_NAME, history)


# ---- settings and reserve ------------------------------------------------------------------

def settings_from(policy: Any, args: Any) -> cr.ResolutionSettings:
    """Knobs from the (frozen) policy; floors, ceiling and temperature read live from args."""
    from gareus.adaptive_production import _args_temperature_k, _resolve_secondary_k_max  # noqa: PLC0415

    def _f(name: str) -> float:
        try:
            v = float(getattr(args, name, 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0
        return v if math.isfinite(v) else 0.0
    return cr.ResolutionSettings.from_policy(policy, temperature_k=_args_temperature_k(args),
                                             k1_min=_f("cv1_k_min"), k2_min=_f("cv2_k_min"),
                                             k2_max=_resolve_secondary_k_max(args))


def find_layout_reserve(out_dir: Optional[Path], registry: Any) -> Tuple[Optional[dict], Optional[str]]:
    """The P1 reserve record of the campaign's layout plan (swarm analysis dir, else beside a
    state's source window table), or (None, None)."""
    from gareus.layout_plan import adaptive_reserve  # noqa: PLC0415
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
                return adaptive_reserve(json.loads(path.read_text())), str(path)
            except (OSError, ValueError) as exc:
                print(f"WARNING: {path} unreadable for the CV2-resolution reserve ({exc})")
    return None, None


def _main_action_states(controller: Any, actions: Sequence[Tuple]) -> Tuple[int, int]:
    """(net states the non-rung main actions add, states the add_rung actions add)."""
    net, rung = 0, 0
    for a in actions:
        kind = str(a[0])
        if kind == "add_rung":
            rung += controller._states_added_by(a)
        elif kind in ("add", "tica_coverage_add", "split", "insert"):
            net += controller._states_added_by(a)
        elif kind == "retire":
            net -= 1
    return net, rung


def budget_for_epoch(registry: Any, policy: Any, actions: Sequence[Tuple], reserve: Optional[dict],
                     settings: cr.ResolutionSettings) -> Tuple[Any, List[Tuple], Dict[str, Any], int]:
    """Allowance after the main proposals; add_rung capped at 1/3 of the free slots when a
    reserve governs. Returns (allowance, actions, add_rung record, n_rungs)."""
    from gareus.adaptive_production import AdaptiveProductionController  # noqa: PLC0415
    ctl = AdaptiveProductionController(registry, policy=policy)
    n_rungs = len(ctl._centre_rungs())
    net, rung = _main_action_states(ctl, actions)
    cap = int(getattr(policy, "max_replicas_budget", 0) or 0)
    n_active = len(registry.active_states()) + net
    first = reserve_allowances(cap, n_active, reserve, n_rungs=n_rungs,
                               resolution_share=float(settings.refine_budget_fraction))
    record = {"states_requested": rung, "limit": first.add_rung_slots, "refused": False}
    if first.governed and rung > int(first.add_rung_slots or 0):
        actions = [a for a in actions if str(a[0]) != "add_rung"]
        record.update(refused=True, reason="add_rung_reserve_share",
                      detail=f"add_rung needs {rung} states > 1/3 of {first.free_slots} free slots")
        rung = 0
    allowance = reserve_allowances(cap, n_active, reserve, n_rungs=n_rungs, add_rung_taken=rung,
                                   resolution_share=float(settings.refine_budget_fraction))
    return allowance, list(actions), record, n_rungs


# ---- transitions (R3) ---------------------------------------------------------------------

def _load_samples(sample_dir: Path) -> dict:
    from gareus.query import load_samples  # noqa: PLC0415
    return load_samples(Path(sample_dir))


def _runs_one_source(data: Mapping[str, Any], wmap: Mapping[int, int], wanted: set) -> Dict[int, Dict[str, list]]:
    """Per wanted state: state-indexed runs (segment, step-contiguous), replica-resolved runs
    (additionally broken wherever the replica at the state changes) and replica paths (each
    replica's own visits in step order, one per segment and replica, NaN dropped, joined across
    its absences: ``cv2_resolution_rules._transitions`` "replica-path")."""
    n = int(len(data.get("step", [])))
    steps = np.asarray(data["step"], dtype=np.int64)
    win = np.asarray(data["window_id"], dtype=np.int64)
    seg = np.asarray(data.get("segment_id") if data.get("segment_id") is not None else np.zeros(n)).astype(str)
    _codes, seg_idx = np.unique(seg, return_inverse=True)
    rep_raw = data.get("replica")
    rep = np.asarray(rep_raw, dtype=np.int64) if rep_raw is not None else None
    cv2 = np.asarray(np.ma.asarray(data["cv2"]).astype(float).filled(np.nan), dtype=float)
    state = np.asarray([int(wmap.get(int(w), -1)) for w in win], dtype=np.int64)
    out: Dict[int, Dict[str, list]] = {}
    for sid in wanted:
        for g in np.unique(seg_idx[state == sid]):
            idx = np.flatnonzero((state == sid) & (seg_idx == g))
            idx = idx[np.argsort(steps[idx], kind="stable")]
            rec = out.setdefault(int(sid), {"state_runs": [], "replica_runs": [], "replica_paths": []})
            rec["state_runs"].extend(contiguous_runs(steps[idx], cv2[idx]))
            if rep is None:
                continue
            stride = _stride(steps[idx])
            for r in np.unique(rep[idx]):
                ridx = idx[rep[idx] == r]
                rec["replica_runs"].extend(contiguous_runs(steps[ridx], cv2[ridx], stride=stride))
                z = cv2[ridx]
                rec["replica_paths"].append(z[np.isfinite(z)])
    return out


def _stride(steps: np.ndarray) -> int:
    d = np.diff(np.asarray(steps, dtype=np.int64))
    d = d[d > 0]
    if not d.size:
        return 1
    vals, cnt = np.unique(d, return_counts=True)
    return int(vals[int(np.argmax(cnt))])


def parquet_runs_provider(sources: Sequence[Tuple[str, Path]], window_map_for: Callable[[Path], Mapping[int, int]],
                          *, load: Optional[Callable[[Path], dict]] = None) -> rules.RunsProvider:
    """Runs provider over the phase's sample sources (X5 grouping + replica breaks)."""
    load = load or _load_samples

    def _provide(state_ids: Sequence[int]) -> Dict[int, Dict[str, Any]]:
        wanted = {int(s) for s in state_ids}
        merged: Dict[int, Dict[str, Any]] = {s: {"state_runs": [], "replica_runs": [], "replica_paths": [],
                                                 "source": "parquet"} for s in wanted}
        have_replica = True
        for label, sample_dir in sources:
            try:
                data = load(Path(sample_dir)) or {}
                if not len(data.get("step", [])):
                    continue
                have_replica = have_replica and data.get("replica") is not None
                for sid, rec in _runs_one_source(data, dict(window_map_for(Path(sample_dir)) or {}), wanted).items():
                    merged[sid]["state_runs"].extend(rec["state_runs"])
                    merged[sid]["replica_runs"].extend(rec["replica_runs"])
                    merged[sid]["replica_paths"].extend(rec["replica_paths"])
            except Exception as exc:
                print(f"      cv2 resolution: skipping {label} for transitions ({type(exc).__name__}: {exc})")
        if not have_replica:
            for rec in merged.values():
                rec["source"] = "no_replica_column"
        return merged
    return _provide


# ---- R2 union ------------------------------------------------------------------------------

def _npz_views(z: Mapping[str, np.ndarray]) -> Tuple[List[cr.StateView], np.ndarray]:
    ids = np.asarray(z["state_ids"], dtype=np.int64)
    lam = np.asarray(z["state_lambdas"], dtype=float) if "state_lambdas" in z else np.zeros(ids.size)
    views = []
    for i, sid in enumerate(ids):
        c2 = cr._finite(z["secondary_centers"][i])
        views.append(cr.StateView(int(sid), float(z["primary_centers"][i]), cr._finite(z["primary_k"][i]), c2,
                                  cr._finite(z["secondary_k"][i]) if c2 is not None else None, float(lam[i])))
    return views, lam


def union_row_sources(npz_path: Path, n_rows: int) -> Tuple[Optional[np.ndarray], str]:
    """Per union row, its sample source (index), from the builder's sidecar
    ``<stem>.samples.csv`` (same kept rows, same order, column ``source_dir`` or ``source``).
    (None, reason) when the file is missing, unreadable or has a different row count; R2 then
    treats each state's rows as one source."""
    path = Path(npz_path).with_suffix(".samples.csv")
    if not path.exists():
        return None, f"no {path.name}"
    import csv  # noqa: PLC0415
    try:
        with path.open(newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            col = header.index("source_dir") if "source_dir" in header else header.index("source")
            labels = [row[col] for row in reader]
    except (OSError, ValueError, StopIteration, IndexError) as exc:
        return None, f"{path.name} unreadable ({type(exc).__name__}: {exc})"
    if len(labels) != int(n_rows):
        return None, f"{path.name} has {len(labels)} rows, the NPZ {int(n_rows)}"
    _u, idx = np.unique(np.asarray(labels, dtype=object).astype(str), return_inverse=True)
    return idx.astype(np.int64), path.name


def union_coverage(npz_path: Path, column_views: Mapping[int, cr.StateView], settings: cr.ResolutionSettings,
                   *, epoch: int, max_gb: float = 8.0) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """R2 on one union NPZ: lambda = 0 rows and states only, own MBAR, then ``coverage_holes``."""
    with np.load(Path(npz_path), allow_pickle=False) as z:
        views, lam = _npz_views({k: z[k] for k in ("state_ids", "state_lambdas", "primary_centers", "primary_k",
                                                   "secondary_centers", "secondary_k") if k in z.files})
        sampled = np.asarray(z["sampled_state_ids"], dtype=np.int64)
        cv1, cv2 = np.asarray(z["cv_A"], dtype=float), np.asarray(z["secondary_cv"], dtype=float)
    row_src, src_note = union_row_sources(npz_path, sampled.size)
    rep = [i for i, v in enumerate(views) if abs(v.lam) <= 1e-9]
    pos = {views[i].state_id: j for j, i in enumerate(rep)}
    idx = np.asarray([pos.get(int(s), -1) for s in sampled], dtype=np.int64)
    keep = (idx >= 0) & np.isfinite(cv1) & np.isfinite(cv2)
    need_gb = keep.sum() * len(rep) * UNION_BYTES_PER_CELL / 1e9
    status = {"status": "ok", "reason": None, "npz": str(npz_path), "n_rows": int(keep.sum()),
              "n_states": len(rep), "n_dropped_rows": int((idx >= 0).sum() - keep.sum()), "estimated_gb": need_gb}
    if need_gb > float(max_gb):
        return [], {**status, "status": "unavailable", "reason": f"memory guard: {need_gb:.1f} GB > {max_gb} GB"}
    sub_views = [views[i] for i in rep]
    cv1, cv2, idx = cv1[keep], cv2[keep], idx[keep]
    row_src = None if row_src is None else row_src[keep]
    status["row_sources"] = src_note if row_src is not None else f"one source per state ({src_note})"
    n_k = np.bincount(idx, minlength=len(rep)).astype(float)
    sampled_k = np.flatnonzero(n_k > 0)
    remap = -np.ones(len(rep), dtype=np.int64)
    remap[sampled_k] = np.arange(sampled_k.size)
    sample_views = [sub_views[k] for k in sampled_k]
    # (N, K) layout for gareus-analyze's solver; the resolve-f bootstrap reuses it
    u_nk = np.ascontiguousarray(cov.reduced_umbrella(cv1, cv2, sample_views, 1.0 / settings.rt).T)
    mbar_info: Dict[str, Any] = {}
    t_mbar = time.time()
    f, logw = cov.solve_rows(u_nk, remap[idx], info=mbar_info)
    mbar_info["wall_s"] = time.time() - t_mbar
    status["mbar"] = mbar_info
    if not mbar_info.get("converged"):
        return [], {**status, "status": "unavailable", "reason": "lambda = 0 MBAR did not converge"}
    w = np.exp(logw - logw.max())
    w /= w.sum()
    boot: Dict[str, Any] = {}
    cands = cov.coverage_holes(cv1, cv2, remap[idx], w, sample_views, column_views,
                               settings, epoch=epoch, source_idx=row_src, info=boot, u_nk=u_nk, f_point=f)
    status["bootstrap"] = boot.get("bootstrap")
    status["coverage_count"] = str(settings.coverage_count)
    return cands, status


def phase_union_npz(adaptive_dir: Path, epoch_dir: Path, policy: Any) -> Tuple[Optional[Path], str]:
    """The union NPZ built THIS phase (top-ups on and this phase's union overlap written)."""
    if not bool(getattr(policy, "topups_enabled", False)):
        return None, "no union diagnostics this epoch (top-ups off); R2 is post-union only"
    npz = Path(adaptive_dir) / UNION_NPZ_NAME
    marks = [p for p in [Path(epoch_dir) / UNION_OVERLAP_NAME, *Path(epoch_dir).glob(f"*/{UNION_OVERLAP_NAME}")]
             if p.exists()]
    if not npz.exists() or not marks:
        return None, "no union diagnostics were built for this phase"
    return npz, "this phase's top-up union"


# ---- orchestration -------------------------------------------------------------------------

def _bridge_edge_of(action: Tuple) -> Optional[Tuple[int, int]]:
    """The edge a weak-edge bridger ``add`` names in its reason ("weak edge i-j: ...")."""
    if str(action[0]) != "add" or len(action) < 4 or not str(action[3]).startswith("weak edge "):
        return None
    try:
        head = str(action[3])[len("weak edge "):].split(":", 1)[0]
        a, b = (int(x) for x in head.split("-"))
    except ValueError:
        return None
    return (min(a, b), max(a, b))


def _filter_main_actions(actions: Sequence[Tuple], funded_edges: set, keep_ids: set
                         ) -> Tuple[List[Tuple], List[Dict[str, Any]]]:
    """Drop today's bridges on edges R1 funds, and retirements of protected / R3-parent states."""
    kept, removed = [], []
    for a in actions:
        edge = _bridge_edge_of(a)
        if edge is not None and edge in funded_edges:
            removed.append({"action": list(a[:2]), "reason": "edge bridged by cv2_resolution R1 instead"})
        elif str(a[0]) == "retire" and int(a[1]) in keep_ids:
            removed.append({"action": list(a[:2]), "reason": "protected by cv2_resolution"})
        else:
            kept.append(a)
    return kept, removed


def _extend_actions(cands: Sequence[Mapping[str, Any]], actions: Sequence[Tuple]) -> List[Tuple]:
    taken = {(str(a[0]), int(a[1])) for a in actions if str(a[0]) in ("extend", "retire")}
    out = []
    for c in cands:
        if c["decision"] != "extend":
            continue
        for sid in c["state_ids"]:
            if ("extend", sid) not in taken and ("retire", sid) not in taken:
                taken.add(("extend", sid))
                out.append(("extend", int(sid), f"cv2_resolution R1: {c['reason']}"))
    return out


def _protected_ids(views: Mapping[int, cr.StateView], epoch: int, settings: cr.ResolutionSettings) -> set:
    return {s for s, v in views.items() if cr.is_protected(v, epoch, settings)}


def requested_rules(policy: Any, rules: Mapping[str, Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Each rule record plus ``requested`` (the grader's completeness set, from the live policy).

    R1 and R3 are always requested with --ap-cv2-resolution on: R1 ``unavailable`` under the
    marginal edge metric is an incomplete evaluation (graded CAUTION), not an opt-out.
    R2 is requested only with top-ups on (``policy.topups_enabled``): it reads the top-up union.
    """
    want = {"R1": True, "R2": bool(getattr(policy, "topups_enabled", False)), "R3": True}
    return {k: {**dict(v or {}), "requested": bool(want.get(k, True))} for k, v in rules.items()}


def propose_cv2_resolution(registry: Any, diagnostics: Mapping[str, Any], actions: Sequence[Tuple],
                           settings: cr.ResolutionSettings, policy: Any, *, epoch: int,
                           history: Mapping[str, Any], subsamples: Mapping[int, Mapping[str, np.ndarray]],
                           runs_for: Optional[rules.RunsProvider] = None,
                           union: Optional[Tuple[List[Dict[str, Any]], Dict[str, Any]]] = None,
                           union_reason: str = "no union diagnostics", reserve: Optional[dict] = None,
                           reserve_source: Optional[str] = None, gate: Any = None,
                           ignore_budget: bool = False) -> Tuple[List[Tuple], Dict[str, Any], Dict[str, Any]]:
    """(new action list, report, new edge history). Pure apart from what ``runs_for`` reads."""
    views = cr.state_views(registry, diagnostics)
    rep = cr.representative_ids(views)
    r1, new_history, r1_status = rules.propose_r1(diagnostics, views, rep, settings, history, epoch, gate)
    r3 = rules.propose_r3(views, rep, subsamples, runs_for, settings, epoch, gate)
    r2, r2_status = union if union is not None else ([], {"status": "unavailable", "reason": union_reason})
    allowance, main, rung_rec, n_rungs = budget_for_epoch(registry, policy, actions, reserve, settings)
    cands, budget = cr.allocate([*r1, *r2, *r3], allowance, n_rungs, ignore_budget=ignore_budget)
    budget.update(add_rung=rung_rec, reserve_source=reserve_source)
    funded = sorted([c for c in cands if c["decision"] == "proposed"], key=cr.priority_of)
    funded_edges = {tuple(sorted(c["state_ids"])) for c in funded if c["rule"] == "R1"}
    keep = _protected_ids(views, epoch, settings) | {c["state_ids"][0] for c in funded if c["rule"] == "R3"}
    main, removed = _filter_main_actions(main, funded_edges, keep)
    new_actions = [a for a in (cr.candidate_action(c, epoch) for c in funded) if a is not None]
    extends = _extend_actions(cands, main)
    report = {"schema_version": cr.SCHEMA_VERSION, "epoch": int(epoch), "status": "ok", "error": None,
              "settings": settings.as_record(),
              "edge_metric": {k: (diagnostics.get("edge_metric") or {}).get(k) for k in ("metric", "status", "stage")},
              "rules": requested_rules(policy, {
                  "R1": r1_status, "R2": r2_status,
                  "R3": {"status": "ok" if subsamples else "unavailable",
                         "reason": None if subsamples else "no P4 paired-CV subsample"}}),
              "candidates": cands, "budget": budget, "actions_removed": removed,
              "actions_added": [{"action": str(a[0]), "state_id": int(a[1]),
                                 "rule": ((a[4] if len(a) > 4 else {}).get(cr.METADATA_KEY) or {}).get("rule", "R1")}
                                for a in extends + new_actions],
              "summary": cr.summarise(cands)}
    return [*main, *extends, *new_actions], _clean(report), new_history


def _subsamples(diagnostics: Mapping[str, Any]) -> Dict[int, Dict[str, np.ndarray]]:
    pc = diagnostics.get("paired_cv") or {}
    if pc.get("status") != "ok" or not pc.get("npz") or not Path(pc["npz"]).exists():
        return {}
    from gareus.adaptive.paired_cv import load_paired_subsamples  # noqa: PLC0415
    return load_paired_subsamples(Path(pc["npz"]))


def _column_views(registry: Any, diagnostics: Mapping[str, Any]) -> Dict[int, cr.StateView]:
    views = cr.state_views(registry, diagnostics)
    return {s: views[s] for s in cr.representative_ids(views)}


def run_epoch_cv2_resolution(*, adaptive_dir: Path, epoch_dir: Path, epoch: int, registry: Any,
                             diagnostics: Mapping[str, Any], actions: Sequence[Tuple], policy: Any, args: Any,
                             out_dir: Optional[Path], phase_dirs: Sequence[Path], gate: Any = None) -> List[Tuple]:
    """The driver's hook: propose, persist the edge history, write the report. Never raises:
    on any failure the report says ``error`` and the epoch's actions are returned unchanged."""
    from gareus.adaptive_production import _load_epoch_window_map, _sample_sources_from_run_root  # noqa: PLC0415
    path = Path(epoch_dir) / cr.REPORT_NAME
    try:
        settings = settings_from(policy, args)
        sources = [src for d in phase_dirs for src in _sample_sources_from_run_root(Path(d).name, Path(d))]
        runs_for = parquet_runs_provider(sources, lambda d: _load_epoch_window_map(d, registry))
        npz, why = phase_union_npz(adaptive_dir, epoch_dir, policy)
        union = None
        if npz is not None:
            union = union_coverage(npz, _column_views(registry, diagnostics), settings, epoch=epoch,
                                   max_gb=float(getattr(policy, "topup_diagnostics_max_gb", 8.0)))
        reserve, source = find_layout_reserve(out_dir, registry)
        new_actions, report, history = propose_cv2_resolution(
            registry, diagnostics, actions, settings, policy, epoch=epoch, history=load_history(adaptive_dir),
            subsamples=_subsamples(diagnostics), runs_for=runs_for, union=union, union_reason=why,
            reserve=reserve, reserve_source=source, gate=gate)
        save_history(adaptive_dir, history)
        warn_no_reserve_once(adaptive_dir, report.get("budget") or {})
        report["stage"] = "numbered_epoch"
        _atomic_json(path, report)
        s = report["summary"]
        print(f"    CV2 resolution: {s['n_proposed']} action(s) proposed, {s['n_blocking']} blocking, "
              f"{s['n_refused_budget']} refused for budget, {s['n_refused_spring_cap']} at the spring cap, "
              f"{s['n_trapped_or_orthogonal']} trapped_or_orthogonal ({path.name})")
        return new_actions
    except Exception as exc:          # never kill a completed epoch
        print(f"WARNING: CV2 resolution failed for epoch {epoch} ({type(exc).__name__}: {exc}); no R1-R3 actions")
        try:
            _atomic_json(path, {"schema_version": cr.SCHEMA_VERSION, "epoch": int(epoch), "status": "error",
                                "error": f"{type(exc).__name__}: {exc}", "candidates": [],
                                "summary": {"counts": {}, "n_proposed": 0, "n_blocking": 0, "n_refused_budget": 0,
                                            "n_refused_spring_cap": 0, "n_trapped_or_orthogonal": 0}})
        except OSError:
            pass
        return list(actions)


def annotate_report_with_refusals(epoch_dir: Path, actions: Sequence[Tuple], refused: Sequence[Mapping[str, Any]]
                                  ) -> None:
    """Record which of this step's actions the applier refused (``apply.refused``)."""
    path = Path(epoch_dir) / cr.REPORT_NAME
    if not path.exists():
        return
    try:
        report = json.loads(path.read_text())
        mine = []
        for r in refused:
            i = r.get("index")
            a = actions[int(i)] if i is not None and int(i) < len(actions) else None
            if a is not None and str(a[3] if len(a) > 3 else a[-1]).startswith("cv2_resolution"):
                mine.append({**{k: v for k, v in r.items()}, "proposal": list(a[:2])})
        report["apply"] = {"refused": mine}
        _atomic_json(path, _clean(report))
    except (OSError, ValueError) as exc:
        print(f"WARNING: could not annotate {path} with the applier's refusals ({exc})")


def is_blocking(epoch_dir: Path) -> int:
    """Funded R1-R3 work recorded for this epoch (``proposed`` candidates), 0 if none.

    Recounted from ``candidates`` so a v1 report (whose ``n_blocking`` also counted budget
    refusals) reads the same way; budget refusals never block. A report without candidates
    falls back to its summary."""
    path = Path(epoch_dir) / cr.REPORT_NAME
    if not path.exists():
        return 0
    try:
        rep = json.loads(path.read_text())
        cands = rep.get("candidates")
        if isinstance(cands, list):
            return sum(1 for c in cands if isinstance(c, dict) and c.get("decision") == "proposed")
        return int((rep.get("summary") or {}).get("n_blocking", 0) or 0)
    except (OSError, ValueError, TypeError):
        return 0


__all__ = ["annotate_report_with_refusals", "budget_for_epoch", "find_layout_reserve", "is_blocking",
           "load_history", "parquet_runs_provider", "phase_union_npz", "propose_cv2_resolution",
           "requested_rules", "reset_no_reserve_warnings", "run_epoch_cv2_resolution", "save_history",
           "settings_from",
           "union_coverage", "union_row_sources", "warn_no_reserve_once"]
