"""Steps 2-5 of the spec (Section 4): discovery partition, evaluation partition, z3 search, placement."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .partitions import FrozenPartition, fit_partition
from .placement import place_workers
from .settings import AuxDiscoverySettings
from .z3_search import emit_model, search_z3, z3_values


@dataclass
class DiscoveryResult:
    status: str
    report: dict = field(default_factory=dict)
    model: object = None
    eval_partition: Optional[FrozenPartition] = None
    placement: Optional[dict] = None


def run_discovery(ft, *, train, holdout, settings: AuxDiscoverySettings, full_topology, k3_max, epoch) -> DiscoveryResult:
    rep = {"n_frames": int(ft.n), "n_train": int(np.sum(train)), "n_holdout": int(np.sum(holdout))}
    lineage = ft.lineage
    disc_X = np.hstack([ft.tors, ft.hc]); disc_fam = ["tors"] * ft.tors.shape[1] + ["hc"] * ft.hc.shape[1]
    cv = np.c_[ft.cv1, ft.cv2]
    disc = fit_partition(disc_X, disc_fam, cv, train, holdout, lineage, ft.step, settings, seed=settings.partition_seed)
    rep["discovery"] = {"status": disc.status, "k": disc.choice.k, "table": disc.choice.table,
                        "hidden_fraction": disc.hidden_fraction, "co_occurrence": disc.co_occurrence,
                        "lineage_info": disc.lineage_info}
    if disc.status != "ok":
        return DiscoveryResult(disc.status, rep)
    ev_X = np.hstack([ft.hc, ft.hb]); ev_fam = ["hc"] * ft.hc.shape[1] + ["hb"] * ft.hb.shape[1]
    ev = fit_partition(ev_X, ev_fam, cv, train, holdout, lineage, ft.step, settings, seed=settings.partition_seed,
                       feature_names=list(ft.definition.hc_labels) + list(ft.definition.hb_labels))
    rep["evaluation"] = {"status": ev.status, "k": ev.choice.k, "table": ev.choice.table}
    if ev.choice.k is None:
        return DiscoveryResult("no_evaluation_partition", rep)
    best, allc = search_z3(ft, disc.choice.labels, disc.bins, train, holdout, settings)
    rep["z3_search"] = {"candidates": [c.summary() for c in allc], "chosen": best.summary() if best else None}
    if best is None:
        return DiscoveryResult("broaden", rep, eval_partition=ev.frozen)
    model = emit_model(best, ft.definition, full_topology, label=f"z3-epoch{epoch:03d}",
                       provenance={"epoch": int(epoch), "groups": list(best.groups), "C": best.C,
                                   "descriptor_schema_sha256": ft.definition.schema_sha256})
    z = z3_values(ft.tors, best)
    pl = place_workers(z, ev.choice.labels, ft.state_id, lineage, ft.step, train, holdout, settings,
                       k_labels=int(ev.choice.k), k3_max=k3_max)
    rep["placement"] = {k: pl[k] for k in ("n_states", "n_candidates", "n_eligible", "skipped_states",
                                           "selection_log", "chosen")}
    rep["model_sha256"] = model.model_sha256
    if not pl["chosen"]:
        return DiscoveryResult("no_worker", rep, model, ev.frozen, pl)
    return DiscoveryResult("ok", rep, model, ev.frozen, pl)
