"""What the frozen CV2 is: component family, index and tICA lag (spec 3.6).

The candidate set records each component's family and, for a conditional tICA mode, its lag
in frames; the frame spacing lives only in the swarm run that fitted it, so the lag in ps is
recorded by the swarm analysis (``cv_selection_report.json["cv2_component"]``) and read back
from there next to the pair model when a production manifest is written.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from . import contracts as C

SELECTION_REPORT_NAME = "cv_selection_report.json"


def component_label(candidates: "C.CandidateSet", index: int,
                    frame_dt_ps: Optional[float] = None) -> dict[str, Any]:
    comp = next((c for c in candidates.components if c.component_index == int(index)), None)
    if comp is None:
        raise ValueError(f"candidate set has no component {index}")
    lag_frames = comp.tica_lag_frames if comp.family == C.COMPONENT_FAMILY_TICA else None
    lag_ps = (float(lag_frames) * float(frame_dt_ps)
              if lag_frames is not None and frame_dt_ps is not None and frame_dt_ps > 0 else None)
    return {"cv2_component_family": comp.family, "cv2_component_index": int(index),
            "cv2_tica_lag_frames": lag_frames, "cv2_tica_lag_ps": lag_ps,
            "frame_dt_ps": None if frame_dt_ps is None else float(frame_dt_ps)}


def frozen_pair_label(pair_model_path, candidate_set_path) -> Optional[dict[str, Any]]:
    """The label of a frozen pair from its artifacts, or None when no pair is configured."""
    if not pair_model_path or not candidate_set_path:
        return None
    from .pair_model import PairModel

    pair = PairModel.from_json_bytes(Path(pair_model_path).read_bytes())
    candidates = C.CandidateSet.from_json_bytes(Path(candidate_set_path).read_bytes())
    dt = None
    report = Path(pair_model_path).parent / SELECTION_REPORT_NAME
    if report.exists():
        try:
            dt = (json.loads(report.read_text()).get("cv2_component") or {}).get("frame_dt_ps")
        except (OSError, ValueError, AttributeError):
            dt = None
    return component_label(candidates, int(pair.selected_component_index), dt)
