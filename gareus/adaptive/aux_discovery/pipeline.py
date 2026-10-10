"""Steps 2-5 of the spec (Section 4, U13): one contacts+H-bond partition, z3 search, placement."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .partitions import FrozenPartition, fit_partition
from .placement import place_workers
from .settings import AuxDiscoverySettings
from .z3_search import NULL_UNINFORMATIVE_STATUS, emit_model, search_z3_sources, z3_null_gate, z3_values, _cooccurring_pairs


@dataclass
class DiscoveryResult:
    status: str
    report: dict = field(default_factory=dict)
    model: object = None
    eval_partition: Optional[FrozenPartition] = None
    placement: Optional[dict] = None


def run_discovery(ft, *, train, holdout, settings: AuxDiscoverySettings, full_topology, k3_max, epoch,
                  eligible_parents=None) -> DiscoveryResult:
    """U13: ONE partition, fitted on CV-residualised contacts + H-bonds only (never torsions), so the torsion-only z3
    must predict structure defined independently of its inputs. z3 candidates come from every passing AND triggered k;
    placement and the frozen evaluation partition use the same fit (largest passing k)."""
    rep = {"n_frames": int(ft.n), "n_train": int(np.sum(train)), "n_holdout": int(np.sum(holdout))}
    lineage = ft.lineage
    X = np.hstack([ft.hc, ft.hb]); fam = ["hc"] * ft.hc.shape[1] + ["hb"] * ft.hb.shape[1]
    # declared conditioning coordinates: CV1 always, CV2 only when the campaign has one (a CV1-only campaign is the
    # supported 1-D case; a NaN inside a declared coordinate is invalid input, never read as "absent")
    cv_names = ["cv1"] + (["cv2"] if getattr(ft, "cv2", None) is not None else [])
    cv = np.column_stack([np.asarray(getattr(ft, nm)) for nm in cv_names])
    part = fit_partition(X, fam, cv, train, holdout, lineage, ft.step, settings, seed=settings.partition_seed,
                         feature_names=list(ft.definition.hc_labels) + list(ft.definition.hb_labels), multi_k=True,
                         cv_names=cv_names)
    block = {"status": part.status, "k": part.choice.k, "table": part.choice.table,
             "hidden_fraction": part.hidden_fraction, "co_occurrence": part.co_occurrence,
             "lineage_info": part.lineage_info,
             "passing_k": [{"k": r["k"], "hidden_fraction": r["hidden_fraction"],
                            "co_occurrence": r["co_occurrence"], "lineage_info": r["lineage_info"],
                            "triggered": r["triggered"]} for r in (part.per_k or [])]}
    cond = getattr(part, "conditioning", None)
    if cond is not None:
        block["conditioning"] = cond
    block["per_k_trigger"] = {str(r["k"]): r["triggered"] for r in (part.per_k or [])}
    rep["partition"] = block
    rep["discovery"] = block  # alias for readers of the pre-U13 report
    if part.status not in ("ok", "keep"):
        return DiscoveryResult(part.status, rep)
    if part.status == "keep":
        return DiscoveryResult("keep", rep, eval_partition=part.frozen)
    sel = list((cond or {}).get("selected") or cv_names)   # z3 guards use the selected dimensions only
    nbins = getattr(part, "n_cond_bins", None) or None
    sources = [(r["k"], r["labels"], _cooccurring_pairs(r["labels"], part.bins, holdout, settings))
               for r in part.per_k if r["triggered"]]
    best, allc = search_z3_sources(ft, sources, part.bins, train, holdout, settings, nbins=nbins,
                                  cond_dims=sel)
    rep["z3_search"] = {"candidates": [c.summary() for c in allc], "chosen": best.summary() if best else None,
                        "sources_k": [k for k, _, _ in sources]}
    if best is not None:
        gate = z3_null_gate(ft, sources, best, part.bins, train, holdout, settings, nbins=nbins,
                             cond_dims=sel)
        rep["z3_search"]["null_gate"] = gate
        if not gate["passed"]:
            best = None
            if gate.get("status") == NULL_UNINFORMATIVE_STATUS:
                return DiscoveryResult(NULL_UNINFORMATIVE_STATUS, rep, eval_partition=part.frozen)
    if best is None:
        return DiscoveryResult("broaden", rep, eval_partition=part.frozen)
    model = emit_model(best, ft.definition, full_topology, label=f"z3-epoch{epoch:03d}",
                       provenance={"epoch": int(epoch), "groups": list(best.groups), "C": best.C,
                                   "descriptor_schema_sha256": ft.definition.schema_sha256})
    z = z3_values(ft.tors, best)
    pl = place_workers(z, part.choice.labels, ft.state_id, lineage, ft.step, train, holdout, settings,
                       k_labels=int(part.choice.k), k3_max=k3_max, eligible_parents=eligible_parents)
    rep["placement"] = {k: pl[k] for k in ("n_states", "n_candidates", "n_eligible", "skipped_states",
                                           "selection_log", "chosen")}
    rep["model_sha256"] = model.model_sha256
    if not pl["chosen"]:
        return DiscoveryResult("no_worker", rep, model, part.frozen, pl)
    return DiscoveryResult("ok", rep, model, part.frozen, pl)
