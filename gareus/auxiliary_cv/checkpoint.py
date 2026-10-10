# gareus/auxiliary_cv/checkpoint.py
"""Bind auxiliary state to production checkpoints and verify it on resume (spec Section 9).

A resume never guesses a state from CV centres. It checks the frozen state table hash, the force
identity, the carrier count, the assignment checksum, the topology identity, the kernel digest,
every Context's auxiliary parameters as restored by loadCheckpoint (before anything re-applies
them), and that the exchange ledger replays to the checkpoint's assignments.
"""
from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any, Mapping, Sequence

from ..correctness._io import IntegrityError, digest, json_bytes, json_loads
from ..correctness.bias import KJ_PER_KCAL
from ..correctness.state_identity import (canonical_state_definition, compare_state_definitions,
                                          state_definition_hash)
from .ledger import assignment_sha256, replay_assignments

AUX_CHECKPOINT_SCHEMA = "atlas-aux-checkpoint-v1"
_REL = 1e-12


def expected_aux_parameters(state_definition: Mapping[str, Any], window_id: int) -> tuple[float, float]:
    found = [r for r in state_definition["windows"] if r["window_id"] == int(window_id)]
    if len(found) != 1:
        raise IntegrityError(f"no window {window_id} in the auxiliary state table")
    k = float(found[0].get("aux_k", 0.0))
    if k == 0.0:
        return 0.0, 0.0
    if found[0].get("aux_center") is None:
        raise IntegrityError(f"window {window_id} is an active auxiliary state (aux_k {k}) without aux_center")
    return k * KJ_PER_KCAL, float(found[0]["aux_center"])


def read_aux_parameters(context, info) -> tuple[float, float]:
    return float(context.getParameter(info.global_k)), float(context.getParameter(info.global_c))


def topology_identity_sha256(topology) -> str:
    atoms = [[a.index, a.name, a.residue.name, a.residue.index, a.residue.chain.index] for a in topology.atoms()]
    bonds = sorted(sorted([b[0].index, b[1].index]) for b in topology.bonds())
    return digest(json_bytes({"atoms": atoms, "bonds": bonds}))


def _force_record(info) -> dict[str, Any]:
    record = asdict(info)
    record["sub_cv_names"] = list(record["sub_cv_names"])
    return record


# Board condition 8: strict on purpose. Measured 2026-10-08 (OpenMM 8.5.1, Reference and CPU):
# Context.createCheckpoint/loadCheckpoint restores CustomCVForce globals bit-exactly, both into the
# same Context and into a fresh one (scratch verify_C/board_checks.py). CUDA/OpenCL round-trip
# precision is measured in Task 15 Step 4; if a GPU platform is not bit-exact, record the measured
# deviation there and stop (spec Section 17) -- do not loosen _REL here.
def _check_params(state, assignments, observed, label) -> None:
    for r, (window, got) in enumerate(zip(assignments, observed)):
        expect = expected_aux_parameters(state, window)
        if not all(math.isclose(float(o), e, rel_tol=_REL, abs_tol=0.0) for o, e in zip(got, expect)):
            raise IntegrityError(f"{label}: replica {r} (window {window}) Context parameters {tuple(got)} != "
                                 f"state table {expect}")


def aux_checkpoint_block(*, state_definition, force_info, assignments: Sequence[int],
                         observed_params: Sequence[tuple[float, float]], topology_sha256: str,
                         kernel_identity_digest: str, segment_id: str, ledger_anchor: Mapping[str, Any],
                         data_boundary: Mapping[str, Any], sample_basis_sha256: str | None = None) -> dict[str, Any]:
    state = canonical_state_definition(state_definition)
    if len(observed_params) != len(assignments):
        raise IntegrityError("save-time check: observed parameters do not cover every replica")
    _check_params(state, assignments, observed_params, "save-time check")
    block = {
        "schema": AUX_CHECKPOINT_SCHEMA,
        "state_definition": state,
        "state_definition_sha256": state_definition_hash(state),
        "force_info": _force_record(force_info),
        "n_replicas": len(assignments),
        "assignment_sha256": assignment_sha256(assignments),
        "observed_params": [[float(k), float(c)] for k, c in observed_params],
        "topology_sha256": str(topology_sha256),
        "kernel_identity_digest": str(kernel_identity_digest),
        "segment_id": str(segment_id),
        "ledger_anchor": {"segment_id": str(ledger_anchor["segment_id"]),
                          "start_step": int(ledger_anchor["start_step"]),
                          "start_assignments": [int(x) for x in ledger_anchor["start_assignments"]]},
        "data_boundary": json_loads(json_bytes(dict(data_boundary))),
    }
    if sample_basis_sha256 is not None:
        if (not isinstance(sample_basis_sha256, str) or len(sample_basis_sha256) != 64
                or any(c not in "0123456789abcdef" for c in sample_basis_sha256)):
            raise IntegrityError("sample_basis_sha256 must be a lowercase SHA-256 digest")
        block["sample_basis_sha256"] = sample_basis_sha256
    return block


def verify_aux_resume_static(manifest: Mapping[str, Any], *, aux_enabled: bool, state_definition, force_info,
                             topology_sha256: str, kernel_identity_digest: str,
                             sample_basis_sha256: str | None = None) -> None:
    """The checks that need no Context: capability, schema, state definition, force, topology, kernel.

    Run before the resumed segment is registered (Task 14 F4), and again inside verify_aux_resume.
    """
    block = manifest.get("aux")
    if bool(aux_enabled) != (block is not None):
        raise IntegrityError("auxiliary capability differs between checkpoint and this run; a model or "
                             "population change starts a new phase, it is not a resume")
    if block is None:
        return
    if block.get("schema") != AUX_CHECKPOINT_SCHEMA:
        raise IntegrityError(f"unknown auxiliary checkpoint schema {block.get('schema')!r}")
    current = state_definition_hash(state_definition)
    if block["state_definition_sha256"] != current:
        diff = compare_state_definitions(canonical_state_definition(block["state_definition"]),
                                         canonical_state_definition(state_definition))
        fields = ", ".join(f"{d['field']}: {d['reference']!r} -> {d['candidate']!r}" for d in diff[:3])
        raise IntegrityError(f"state_definition_sha256 changed: checkpoint {block['state_definition_sha256']} vs run "
                             f"{current}; first differing fields: {fields}")
    if json_loads(json_bytes(block["force_info"])) != json_loads(json_bytes(_force_record(force_info))):
        raise IntegrityError(f"force_info changed: checkpoint {block['force_info']} vs run {_force_record(force_info)}")
    if block["topology_sha256"] != str(topology_sha256):
        raise IntegrityError("topology identity changed between checkpoint and this run")
    if block["kernel_identity_digest"] != str(kernel_identity_digest):
        raise IntegrityError("kernel identity digest changed between checkpoint and this run")
    if block.get("sample_basis_sha256") != sample_basis_sha256:
        raise IntegrityError("sample angle basis changed between checkpoint and this run")


def verify_aux_resume(manifest: Mapping[str, Any], *, aux_enabled: bool, state_definition, force_info,
                      assignments: Sequence[int], observed_params: Sequence[tuple[float, float]],
                      topology_sha256: str, kernel_identity_digest: str,
                      sample_basis_sha256: str | None = None) -> None:
    """Call with the Context parameters as restored by loadCheckpoint, BEFORE any re-apply."""
    verify_aux_resume_static(manifest, aux_enabled=aux_enabled, state_definition=state_definition,
                             force_info=force_info, topology_sha256=topology_sha256,
                             kernel_identity_digest=kernel_identity_digest,
                             sample_basis_sha256=sample_basis_sha256)
    block = manifest.get("aux")
    if block is None:
        return
    if int(block["n_replicas"]) != len(assignments) or len(observed_params) != len(assignments):
        raise IntegrityError(f"replica/carrier set incomplete: checkpoint {block['n_replicas']}, run {len(assignments)}")
    if block["assignment_sha256"] != assignment_sha256(assignments):
        raise IntegrityError("resumed assignment does not match the checkpoint's assignment checksum")
    state = canonical_state_definition(state_definition)
    _check_params(state, assignments, observed_params, "before re-apply")
    for r, (got, saved) in enumerate(zip(observed_params, block["observed_params"])):
        if not all(math.isclose(float(o), float(s), rel_tol=_REL, abs_tol=0.0) for o, s in zip(got, saved)):
            raise IntegrityError(f"before re-apply: replica {r} restored parameters {tuple(got)} != saved {tuple(saved)}")


def verify_aux_ledger(manifest: Mapping[str, Any], events: Mapping[str, Any]) -> None:
    block = manifest.get("aux")
    if block is None:
        return
    anchor = block["ledger_anchor"]
    if not events or "step" not in events or len(events["step"]) == 0:
        # No exchange ran between the anchor and the checkpoint (e.g. SIGTERM before the first exchange,
        # or a checkpoint interval shorter than the exchange interval): nothing may have moved.
        start = [int(x) for x in anchor["start_assignments"]]
        if start != [int(x) for x in manifest["assignments"]]:
            raise IntegrityError(f"no exchange event in ledger segment {anchor['segment_id']}, yet the checkpoint "
                                 f"assignment {list(manifest['assignments'])} differs from the anchor's {start}")
        return
    got = replay_assignments(events, anchor["start_assignments"], after_step=int(anchor["start_step"]),
                             up_to_step=int(manifest["absolute_step"]))
    if [int(x) for x in got] != [int(x) for x in manifest["assignments"]]:
        raise IntegrityError(f"exchange ledger replays to {got}, checkpoint holds {list(manifest['assignments'])}")


def aux_table_from_checkpoint(manifest: Mapping[str, Any], *, model):
    """Rebuild Stage B's AuxStateTable from the checkpoint block (the resume path's table source)."""
    from .state_table import AuxStateTable
    block = manifest.get("aux")
    if block is None:
        raise IntegrityError("checkpoint has no auxiliary block to resume an auxiliary run from")
    state = canonical_state_definition(block["state_definition"])
    registry = state.get("aux_models") or {}
    if model.model_sha256 not in registry:
        raise IntegrityError(f"--aux-cv-model {model.model_sha256} is not the checkpoint's model {sorted(registry)}")
    windows = sorted(state["windows"], key=lambda r: r["window_id"])
    missing = [r["window_id"] for r in windows if float(r.get("aux_k", 0.0)) != 0.0 and r.get("aux_center") is None]
    if missing:
        raise IntegrityError(f"checkpoint auxiliary table: active window(s) {missing[:8]} without aux_center")
    return AuxStateTable(model, tuple(float(r.get("aux_center", 0.0)) for r in windows),
                         tuple(float(r["aux_k"]) for r in windows),
                         tuple(dict(r["instance"]) for r in windows))       # v2: an instance on every row


def check_checkpoint_rows_align(manifest: Mapping[str, Any], applied_window_key: Sequence[Sequence]) -> None:
    """The checkpoint's frozen auxiliary table pairs row by row with the windows this resume applies.

    ``applied_window_key`` is ``_aux_window_order_key`` of the resumed centres/k (CV1, and CV2 when
    present): a row whose restraints differ would pair an auxiliary term with another window's
    restraints (Task 13 carry-over 9). Inactive axes compare as (0, 0) (``normalize_windows``).
    """
    from .runtime_definition import _RESTRAINT_FIELDS, _applied_rows
    block = manifest.get("aux")
    if block is None:
        raise IntegrityError("checkpoint has no auxiliary block to align the resumed windows with")
    rows = sorted(canonical_state_definition(block["state_definition"])["windows"], key=lambda r: r["window_id"])
    applied = _applied_rows(applied_window_key)
    if len(rows) != len(applied):
        raise IntegrityError(f"checkpoint auxiliary table has {len(rows)} rows, this resume applies "
                             f"{len(applied)} windows")
    for i, (have, want) in enumerate(zip(rows, applied)):
        diff = [k for k in _RESTRAINT_FIELDS if float(have[k]) != float(want[k])]
        if diff or int(have["window_id"]) != int(want["window_id"]):
            raise IntegrityError(f"checkpoint auxiliary table row {i} (window_id {have['window_id']}) {diff} "
                                 f"{[have[k] for k in diff]} != the resumed window's {[want[k] for k in diff]}; "
                                 "the auxiliary term would pair with another window's restraints")
