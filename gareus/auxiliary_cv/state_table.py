"""Auxiliary-state table of a plain run: one frozen (centre, k, instance) per window.

Parsed from the explicit window CSV (``--windows-2d-csv``) columns ``aux_center``,
``aux_k_kcal_mol`` (explicit on every row; 0 = ordinary or sham), optional ``aux_model_sha256``
(must equal the loaded model) and the optional instance columns. Canonicalisation reuses
the Stage A helpers so the runtime table and the v2 state schema can never disagree.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..correctness._io import IntegrityError
from ..correctness.bias import _aux_term
from ..correctness.state_identity import _canonical_instance
from ..windows import AUX_CSV_COLUMNS, INSTANCE_CSV_COLUMNS  # noqa: F401  (re-exported)
from .model import AuxModel


def _cell(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    return "" if value is None else str(value).strip()


def parse_aux_csv_row(row: Mapping[str, Any], offset: int, model_sha256: str) -> dict:
    label = f"window table row {offset}"
    k_raw = _cell(row, "aux_k_kcal_mol")
    if not k_raw:
        raise IntegrityError(f"{label}: aux_k_kcal_mol is required on every row (0 = ordinary or sham state)")
    raw: dict[str, Any] = {"aux_k": float(k_raw)}
    if _cell(row, "aux_center"):
        raw["aux_center"] = float(_cell(row, "aux_center"))
    sha = _cell(row, "aux_model_sha256")
    if sha and sha != model_sha256:
        raise IntegrityError(f"{label}: aux_model_sha256 {sha} is not the loaded model {model_sha256}")
    raw["aux_model_sha256"] = sha or model_sha256
    model, center, k = _aux_term(raw, label)
    instance = None
    if any(_cell(row, key) for key in INSTANCE_CSV_COLUMNS):
        obs = _cell(row, "spawn_source_observation_json")
        instance = _canonical_instance({
            "state_instance_id": _cell(row, "state_instance_id"),
            "state_role": _cell(row, "state_role"),
            "spawn_parent_state_id": _cell(row, "spawn_parent_state_id") or None,
            "spawn_source_observation": json.loads(obs) if obs else None,
            "matched_additional_slot_id": _cell(row, "matched_additional_slot_id") or None,
        }, offset)
        # D3: auxiliary <=> aux_k > 0; ordinary and sham require aux_k == 0. Stage A enforces the
        # same rule in the v2 canonicalisation; checking here gives a CSV-row error message.
        if (instance["state_role"] == "auxiliary") != (k > 0):
            raise IntegrityError(f"{label}: state_role {instance['state_role']!r} contradicts aux_k {k} "
                                 "(auxiliary iff aux_k > 0; ordinary and sham need aux_k == 0)")
    return {"aux_model_sha256": model, "aux_center": center, "aux_k": k, "instance": instance}


@dataclass(frozen=True)
class AuxStateTable:
    model: AuxModel
    centers: tuple[float, ...]
    k_kcal: tuple[float, ...]
    instances: tuple[dict | None, ...]

    @property
    def n(self) -> int:
        return len(self.k_kcal)

    def window_rows(self) -> list[dict]:
        rows = []
        for w in range(self.n):
            active = self.k_kcal[w] > 0
            row = {"aux_model_sha256": self.model.model_sha256 if active else None,
                   "aux_center": self.centers[w], "aux_k": self.k_kcal[w]}
            if self.instances[w] is not None:
                row["instance"] = dict(self.instances[w])
            rows.append(row)
        return rows


def load_aux_state_table(model_path: Path | str, aux_rows: list[Mapping[str, Any]]) -> AuxStateTable:
    model = AuxModel.load(model_path)
    if not aux_rows:
        raise IntegrityError("--aux-cv-model needs a window table with aux_k_kcal_mol on every row")
    parsed = [parse_aux_csv_row(row, offset, model.model_sha256)
              for offset, row in enumerate(aux_rows, start=2)]
    with_instance = [p for p in parsed if p["instance"] is not None]
    if with_instance and len(with_instance) != len(parsed):
        raise IntegrityError("instance columns must be filled on every row or on none")
    if not with_instance:
        raise IntegrityError("auxiliary state tables need the instance columns (state_instance_id, state_role, "
                             "spawn_parent_state_id, matched_additional_slot_id) on every row: a frozen v2 state "
                             "definition requires instance metadata on every window (Stage C ruling M2)")
    ids = [p["instance"]["state_instance_id"] for p in with_instance]
    if len(set(ids)) != len(ids):
        raise IntegrityError(f"duplicate state_instance_id in the window table: {sorted(ids)}")
    known = set(ids)
    for p in with_instance:
        parent = p["instance"]["spawn_parent_state_id"]
        if parent is not None and parent == p["instance"]["state_instance_id"]:
            raise IntegrityError(f"state {parent!r} is its own parent (spawn_parent_state_id)")
        if parent is not None and parent not in known:
            raise IntegrityError(f"state {p['instance']['state_instance_id']!r}: spawn parent {parent!r} is "
                                 "not a state_instance_id of this (frozen, plain-run) table")
    return AuxStateTable(model, tuple(p["aux_center"] for p in parsed), tuple(p["aux_k"] for p in parsed),
                         tuple(p["instance"] for p in parsed))
