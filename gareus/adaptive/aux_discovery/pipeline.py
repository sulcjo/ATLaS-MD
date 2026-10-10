"""Fit the independent contact/H-bond partition and discover backbone or local torsion candidates."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .partitions import FrozenPartition, fit_partition
from .placement import place_workers
from .settings import AuxDiscoverySettings
from .z3_search import (NULL_UNINFORMATIVE_STATUS, emit_model, local_candidate_summary,
                        rank_local_candidates, search_z3_sources, z3_null_gate, z3_values,
                        _cooccurring_pairs)


@dataclass
class DiscoveryResult:
    status: str
    report: dict = field(default_factory=dict)
    model: object = None
    eval_partition: Optional[FrozenPartition] = None
    placement: Optional[dict] = None
    local_candidate: object = None
    source_partition_labels: Optional[np.ndarray] = None
    source_partition_k: Optional[int] = None


def local_split_ids(ft, train, holdout, seed: int) -> np.ndarray:
    """Assign whole training phase/replica carriers to fit or tune; boundary epoch is held out."""
    train = np.asarray(train, dtype=bool)
    holdout = np.asarray(holdout, dtype=bool)
    if train.shape != (ft.n,) or holdout.shape != (ft.n,) or np.any(train & holdout) or not np.all(train | holdout):
        raise ValueError("local split needs disjoint training and holdout masks covering every frame")
    lineages = np.asarray(ft.lineage).astype(str)
    out = np.full(ft.n, 2, dtype=np.int8)
    carrier_ids = sorted(np.unique(lineages[train]))
    if len(carrier_ids) < 2:
        raise ValueError("local search needs at least two training phase/replica carriers")
    carrier_ids.sort(key=lambda carrier: hashlib.sha256(f"{int(seed)}:{carrier}".encode()).digest())
    for position, carrier in enumerate(carrier_ids):
        bucket = position % 2
        out[train & (lineages == carrier)] = bucket
    return out


def _partition_identity(k, labels, bins, train, conditioning) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps({"k": int(k), "conditioning": conditioning}, sort_keys=True,
                             separators=(",", ":"), default=str).encode())
    mask = np.asarray(train, dtype=bool)
    digest.update(np.asarray(mask.shape, dtype=np.int64).tobytes())
    digest.update(mask.astype(np.uint8).tobytes())
    for values in (np.asarray(labels, dtype=np.int64)[mask], np.asarray(bins, dtype=np.int64)[mask]):
        digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
        digest.update(values.tobytes())
    return digest.hexdigest()


def search_local_candidates(ft, sources, regions, train, holdout, settings: AuxDiscoverySettings, *, split_ids=None,
                            active_cv_names=None):
    """Fit all enabled local families and repeat the complete search inside one joint shifted-label null."""
    from .local_search_core import compare_null, fit_local_candidates

    Xb = np.asarray(ft.tors, dtype=float)
    chi = getattr(ft, "chi", None)
    Xc = np.empty((ft.n, 0), dtype=float) if chi is None else np.asarray(chi, dtype=float)
    train, holdout = np.asarray(train, bool), np.asarray(holdout, bool)
    if (Xb.ndim != 2 or Xb.shape[0] != ft.n or Xc.ndim != 2 or Xc.shape[0] != ft.n
            or not np.isfinite(Xb).all() or not np.isfinite(Xc).all()):
        return {"status": "invalid_local_features", "candidates": [], "chosen": None,
                "null_gate": {"status": "invalid_local_features", "passed": False}}
    if not Xc.shape[1] and settings.feature_space == "sidechain":
        return {"status": "no_sidechain_features", "candidates": [], "chosen": None,
                "null_gate": {"status": "no_sidechain_features", "passed": False}}
    X = np.column_stack((Xb, Xc))
    widths = Xb.shape[1], Xc.shape[1]
    if settings.feature_space == "sidechain":
        families = {"sidechain": tuple(range(widths[0], sum(widths)))}
    elif settings.feature_space == "mixed" and not Xc.shape[1]:
        families = {"backbone": tuple(range(widths[0]))}
    elif settings.feature_space in ("mixed", "auto"):
        families = {"backbone": tuple(range(widths[0])),
                    "sidechain": tuple(range(widths[0], sum(widths))),
                    "mixed": tuple(range(sum(widths)))}
    else:
        raise ValueError("local candidate search requires sidechain, mixed or auto feature space")
    split = local_split_ids(ft, train, holdout, settings.partition_seed) if split_ids is None else np.asarray(split_ids)
    if split.shape != (ft.n,) or set(np.unique(split)) != {0, 1, 2} or np.any((split == 2) != holdout):
        raise ValueError("local split IDs must encode fit=0, tune=1 and exactly the epoch holdout=2")
    regions = np.asarray(regions)
    if regions.shape != (ft.n,) or regions.dtype.kind not in "iu" or (regions < 0).any():
        raise ValueError("frozen conditioning bins must be nonnegative integer region IDs")
    cv_names = ["cv1"] + (["cv2"] if getattr(ft, "cv2", None) is not None else [])
    cvs = {name: np.asarray(getattr(ft, name), dtype=float) for name in cv_names}
    if any(values.shape != (ft.n,) or not np.isfinite(values).all() for values in cvs.values()):
        return {"status": "invalid_local_conditioning", "candidates": [], "chosen": None,
                "null_gate": {"status": "invalid_local_conditioning", "passed": False}}

    if not sources:
        return {"status": "no_source_partition", "candidates": [], "chosen": None,
                "null_gate": {"status": "no_source_partition", "passed": False}}
    label_columns = []
    for partition_id, _k, labels in sources:
        values = np.asarray(labels)
        if (not isinstance(partition_id, str) or not partition_id or values.shape != (ft.n,)
                or values.dtype.kind not in "iu" or (values < 0).any()):
            raise ValueError("source partition requires an identity and nonnegative integer labels for every frame")
        label_columns.append(values.astype(np.int64, copy=False))
    if len({x[0] for x in sources}) != len(sources):
        raise ValueError("source partition identities must be unique")
    matrix = np.column_stack(label_columns)
    if matrix.shape != (ft.n, len(sources)):
        raise ValueError("source partition labels must cover every frame")
    lineage = np.asarray(ft.lineage).astype(str)
    null_lineage = np.char.add(np.char.add(lineage, ":local-split-"), split.astype(str))
    active_cv = list(active_cv_names or cv_names)
    if not active_cv or any(name not in cvs for name in active_cv):
        raise ValueError("active CV redundancy guard names differ from the frozen conditioning schema")
    real_rejections = {}
    real_candidates = [None]

    def complete_search(label_matrix):
        all_candidates = []
        rejections = {}
        for column, (partition_id, _k, _labels) in enumerate(sources):
            candidates = fit_local_candidates(X, label_matrix[:, column], regions, split,
                                              partition_id=partition_id, families=families,
                                              c_grid=settings.l1_c_grid)
            for candidate in candidates:
                z = candidate.values(X)
                reason = None
                for name in active_cv:
                    cv = cvs.get(name)
                    if cv is None:
                        continue
                    ok = holdout & np.isfinite(cv) & np.isfinite(z)
                    if ok.sum() < 3 or np.unique(cv[ok]).size < 2:
                        continue
                    corr = float(np.corrcoef(z[ok], cv[ok])[0, 1])
                    if not np.isfinite(corr):
                        reason = f"nonfinite_corr_{name}"
                        break
                    if abs(corr) > settings.max_cv_corr:
                        reason = f"max_cv_corr_{name}"
                        break
                if reason is None:
                    all_candidates.append(candidate)
                else:
                    rejections[reason] = rejections.get(reason, 0) + 1
        all_candidates = rank_local_candidates(all_candidates)
        if real_candidates[0] is None:
            real_rejections.update(rejections)
            real_candidates[0] = all_candidates
        return all_candidates

    gate = compare_null(matrix, null_lineage, ft.step, complete_search, n_null=settings.n_null_z3,
                        seed=int(settings.partition_seed) + 7919)
    candidates = real_candidates[0] or []
    chosen = candidates[0] if candidates else None
    return {"status": "ok" if gate["passed"] and chosen is not None else gate.get("status", "no_real_candidate"),
            "candidates": candidates, "chosen": chosen, "null_gate": gate,
            "candidate_summaries": [local_candidate_summary(c) for c in candidates],
            "rejections": dict(sorted(real_rejections.items())), "families": list(families),
            "source_partition_ids": [str(x[0]) for x in sources]}


def emit_local_model(candidate, ft, topology, *, epoch):
    """Freeze one local candidate as validated v2 model; feature order stays backbone then dictionary chi."""
    from gareus.auxiliary_cv.atom_mapping import topology_metadata
    from gareus.auxiliary_cv.sidechain_core import Primitive, Projection
    from gareus.auxiliary_cv.sidechain_model import SidechainModel

    dictionary = ft.sidechain_dictionary
    if dictionary is None:
        raise ValueError("local sidechain model emission requires frozen sidechain dictionary")
    import mdtraj as md
    from .descriptors import descriptor_definition
    production_definition = descriptor_definition(md.Topology.from_openmm(topology))
    if production_definition.tors_labels != ft.definition.tors_labels:
        raise ValueError("production and discovery backbone feature order differs")
    quads = list(production_definition.phi_quads) + list(production_definition.psi_quads)
    theta = np.asarray(ft.tors_theta_iupac)
    if theta.ndim != 2 or theta.shape[1] != len(quads):
        raise ValueError("raw backbone angles do not match frozen descriptor quadrupoles")
    atoms = list(topology.atoms())
    primitives, features = [], []
    for kind, names, torsion_quads, theta_col in (
            ("phi", production_definition.phi_labels, production_definition.phi_quads,
             range(len(production_definition.phi_quads))),
            ("psi", production_definition.psi_labels, production_definition.psi_quads,
             range(len(production_definition.phi_quads), len(quads)))):
        for name, quad, col in zip(names, torsion_quads, theta_col):
            quad = tuple(int(i) for i in quad)
            central = atoms[quad[2 if kind == "phi" else 1]].residue
            row = {"name": name, "orbit": [list(quad)], "harmonic": 1, "sign": 1,
                   "family": "backbone", "chain_id": str(central.chain.id or ""),
                   "residue_id": str(central.id),
                   "insertion_code": str(getattr(central, "insertionCode", "") or ""),
                   "residue_index": int(central.index), "residue_name": str(central.name),
                   "template": str(central.name), "chi_index": 0}
            for trig in ("sin", "cos"):
                primitives.append(Primitive((quad,), trig))
                features.append({**row, "name": f"{name}_{trig}", "trig": trig})
    for torsion in dictionary.torsions:
        for primitive, trig in zip(torsion.primitives, ("sin", "cos")):
            primitives.append(primitive)
            features.append({"name": f"chi{torsion.chi_index}_{torsion.chain_id}:{torsion.residue_id}:"
                             f"{torsion.insertion_code}_{torsion.template}_{trig}",
                             "orbit": [list(q) for q in primitive.orbit], "trig": trig,
                             "harmonic": primitive.harmonic, "sign": primitive.sign,
                             "family": "sidechain", "chain_id": torsion.chain_id,
                             "residue_id": torsion.residue_id, "insertion_code": torsion.insertion_code,
                             "residue_index": torsion.residue_index, "residue_name": torsion.residue_name,
                             "template": torsion.template, "chi_index": torsion.chi_index})
    X = np.column_stack((np.asarray(ft.tors, dtype=float), np.asarray(ft.chi, dtype=float)))
    if X.shape[1] != len(primitives) or X.shape[1] != len(candidate.coefficients):
        raise ValueError("candidate feature width differs from emitted projection basis")
    projection = Projection(tuple(primitives), tuple(candidate.coefficients),
                            offset=candidate.offset, scale=candidate.scale)
    fitted = np.asarray(candidate.values(X), dtype=float)
    emitted = (X @ np.asarray(projection.coefficients) + projection.offset) / projection.scale
    if fitted.shape != emitted.shape or not np.allclose(fitted, emitted, rtol=1e-12, atol=1e-12):
        raise ValueError("fitted candidate z differs from emitted Projection z")
    atom_keys, bonds = topology_metadata(topology)
    raw = {"schema": "atlas-aux-cv-model-v2", "dictionary_version": dictionary.version,
           "topology_sha256": dictionary.topology_digest, "system_sha256": dictionary.system_digest,
           "atom_keys": [list(k) for k in atom_keys], "bonds": [list(edge) for edge in bonds],
           "features": features, "coefficients": [float(c) for c in projection.coefficients],
           "offset": float(projection.offset), "scale": float(projection.scale),
           "periodic_imaging": "none", "units": "dimensionless",
           "dictionary_exclusions": [vars(x) for x in dictionary.exclusions],
           "label": f"local-epoch{int(epoch):03d}",
           "provenance": {"epoch": int(epoch), "partition_id": candidate.partition_id,
                          "region": int(candidate.region), "pair": list(candidate.pair),
                          "family": candidate.family}}
    model = SidechainModel.from_mapping(raw)
    if model.projection.coefficients != projection.coefficients:
        raise ValueError("validated model changed emitted Projection coefficients")
    return model


def run_discovery(ft, *, train, holdout, settings: AuxDiscoverySettings, full_topology, k3_max, epoch,
                  eligible_parents=None) -> DiscoveryResult:
    """Use contact/H-bond partitions independent of predictors; legacy candidates place here, local candidates retain
    their source partition identity for candidate-consistent placement by the admission adapter."""
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
    sel = list((cond or {}).get("selected") or cv_names)
    if settings.feature_space == "sidechain" and getattr(ft, "chi", None) is None:
        rep["local_search"] = {"status": "no_sidechain_features"}
        return DiscoveryResult("no_sidechain_features", rep, eval_partition=part.frozen)
    has_chi = getattr(ft, "chi", None) is not None and np.asarray(ft.chi).ndim == 2 and ft.chi.shape[1] > 0
    if settings.feature_space in ("sidechain", "mixed", "auto") and not (
            settings.feature_space == "auto" and not has_chi):
        sources = [(_partition_identity(r["k"], r["labels"], part.bins, train, cond),
                    r["k"], r["labels"])
                   for r in part.per_k if r["triggered"]]
        local = search_local_candidates(ft, sources, part.bins, train, holdout, settings,
                                        active_cv_names=sel)
        local_report = {k: v for k, v in local.items() if k not in ("candidates", "chosen")}
        local_report["candidates"] = local["candidate_summaries"]
        local_report["chosen"] = None if local["chosen"] is None else local_candidate_summary(local["chosen"])
        rep["z3_search"] = local_report
        if local["status"] != "ok" or local["chosen"] is None:
            status = local["status"] if local["status"] not in ("heuristic", "no_real_candidate") else "broaden"
            return DiscoveryResult(status, rep, eval_partition=part.frozen)
        chosen = local["chosen"]
        source = next((r for r in part.per_k if _partition_identity(
            r["k"], r["labels"], part.bins, train, cond) == chosen.partition_id), None)
        if source is None:
            raise RuntimeError("local candidate refers to an unknown frozen source partition")
        rep["local_candidate"] = {"partition_id": chosen.partition_id, "k": int(source["k"]),
                                  "region": chosen.region, "pair": list(chosen.pair),
                                  "family": chosen.family}
        model = emit_local_model(chosen, ft, full_topology, epoch=epoch)
        X = np.column_stack((np.asarray(ft.tors, dtype=float), np.asarray(ft.chi, dtype=float)))
        z = chosen.values(X)
        pl = place_workers(z, source["labels"], ft.state_id, lineage, ft.step, train, holdout, settings,
                           k_labels=int(source["k"]), k3_max=k3_max, eligible_parents=eligible_parents,
                           local_support={"labels": source["labels"], "regions": part.bins,
                                          "pair": chosen.pair, "region": chosen.region})
        rep["placement"] = {k: pl[k] for k in ("n_states", "n_candidates", "n_eligible", "skipped_states",
                                               "selection_log", "chosen")}
        rep["model_sha256"] = model.model_sha256
        if not pl["chosen"]:
            return DiscoveryResult("no_worker", rep, model, part.frozen, pl, chosen,
                                   source["labels"], int(source["k"]))
        return DiscoveryResult("ok", rep, model, part.frozen, pl, chosen,
                               source["labels"], int(source["k"]))
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
