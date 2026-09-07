"""Per-window seed frames and the GENPEPT-compatible seed bank (spec Sec.2 S1 product (5)).

Selection is ab initio: candidate swarm frames are ranked only by |cv1 - centre| (no
native/folded reference, no RMSD-to-native). Seeds are rung-independent -- the same
frames chosen here seed every lambda at a window; the ladder (Task 3) differs only in k0.

``export_seed_bank`` writes a seed bank directory readable by
``gareus.seeding.load_genpept_conformer_library`` (it reads only
``final_survivor_seeds.csv``, column ``survivor_pdb_path``) via
``gareus.tica._append_seed_bank_row``.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List

from gareus.tica import _append_seed_bank_row


def select_window_seed_frames(
    frames: List[Dict[str, Any]],
    centers,
    *,
    per_window: int = 3,
    discard_frames: int = 0,
) -> Dict[int, List[Dict[str, Any]]]:
    """Pick up to ``per_window`` frames nearest each centre, distinct members first.

    ``frames``: dicts with ``member_id``, ``frame``, ``cv1``, ``pdb_path``. Frames with
    ``frame < discard_frames`` (pre-equilibration) are excluded. For each window centre,
    candidates are sorted by |cv1 - centre|; the walk takes the first frame from each
    not-yet-used member until ``per_window`` seeds are collected (or candidates run out).
    Windows with zero eligible frames are omitted from the result.
    """
    eligible = [fr for fr in frames if fr["frame"] >= discard_frames]
    selection: Dict[int, List[Dict[str, Any]]] = {}
    for window_idx, centre in enumerate(centers):
        ranked = sorted(eligible, key=lambda fr: abs(fr["cv1"] - centre))
        chosen: List[Dict[str, Any]] = []
        used_members = set()
        for fr in ranked:
            if len(chosen) >= per_window:
                break
            if fr["member_id"] in used_members:
                continue
            chosen.append(fr)
            used_members.add(fr["member_id"])
        if chosen:
            selection[window_idx] = chosen
    return selection


def export_seed_bank(
    seed_bank_dir: Path,
    selection: Dict[int, List[Dict[str, Any]]],
    centers,
    *,
    round_index: int,
) -> Path:
    """Copy selected frame PDBs into ``seed_bank_dir`` and append seed-bank rows.

    Returns ``seed_bank_dir``. Also writes ``selection.json`` (window -> centre, and
    the chosen frames' member/frame/cv1/|delta cv1|) alongside the CSV.
    """
    seed_bank_dir = Path(seed_bank_dir).resolve()
    seed_bank_dir.mkdir(parents=True, exist_ok=True)

    selection_summary: Dict[str, Any] = {}
    for window_idx, frames in selection.items():
        centre = centers[window_idx]
        chosen_summary = []
        for fr in frames:
            member_id = fr["member_id"]
            frame_no = fr["frame"]
            cv1 = fr["cv1"]
            seed_name = f"w{window_idx:02d}_m{member_id:04d}_f{frame_no:05d}"
            dest_pdb = seed_bank_dir / f"{seed_name}.pdb"
            shutil.copyfile(fr["pdb_path"], dest_pdb)

            source_pdb_path = Path(fr["pdb_path"])
            row = {
                "seed_name": seed_name,
                "survivor_pdb_path": str(dest_pdb),
                "source_run_dir": str(source_pdb_path.parent),
                "source_pdb_path": str(source_pdb_path),
                "source_label": f"swarm_round_{round_index:03d}",
                "source_state_id": window_idx,
                "source_epoch_window": f"round{round_index:03d}",
                "primary_cv_value": cv1,
                "secondary_cv_value": "",
            }
            _append_seed_bank_row(seed_bank_dir, row)
            chosen_summary.append(
                {
                    "member_id": member_id,
                    "frame": frame_no,
                    "cv1": cv1,
                    "abs_delta_cv1": abs(cv1 - centre),
                    "seed_name": seed_name,
                }
            )
        selection_summary[str(window_idx)] = {"centre": centre, "frames": chosen_summary}

    with (seed_bank_dir / "selection.json").open("w") as handle:
        json.dump(selection_summary, handle, indent=2)

    return seed_bank_dir
