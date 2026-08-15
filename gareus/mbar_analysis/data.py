"""Data-loading domain: the ``Data`` dataclass and its lifecycle helpers.

Relocated verbatim from ``analyze_gareus_mbar.py`` (Plan A2 of the
MBAR-analysis modularization sequence; see
docs/superpowers/specs/2026-08-13-mbar-analysis-modularization-a2-design.md).
No behavior change from the original script versions.
"""
from __future__ import annotations

import csv
import json
import math
import re
import warnings as _warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from gareus.units import KJ_PER_KCAL, K_B_KJ_PER_MOL_K


def rjson(path: Path, default=None):
    try:
        return json.loads(path.read_text()) if path.exists() else ({} if default is None else default)
    except Exception:
        return {} if default is None else default


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        x=float(obj)
        return x if math.isfinite(x) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f'Object of type {type(obj).__name__} is not JSON serializable')


def wjson(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=_json_default))


def read_windows(path: Path):
    centers=[]; ks=[]; rows=[]
    if not path.exists(): return np.array([]), np.array([]), rows
    with path.open(newline='') as f:
        for row in csv.DictReader(f):
            rows.append(dict(row))
            c=row.get('center_A') or row.get('center') or row.get('center_a')
            k=row.get('k_kcal_mol_A2') or row.get('k_kcal_A2') or row.get('k')
            if c not in (None,''): centers.append(float(c))
            if k not in (None,''): ks.append(float(k))
    return np.asarray(centers,float), np.asarray(ks,float), rows


def jvec(txt):
    if txt is None or txt=='': return []
    v=json.loads(txt)
    if isinstance(v,dict): return [float(v[k]) for k in sorted(v, key=lambda x:int(x) if str(x).isdigit() else str(x))]
    return [float(x) for x in v]


@dataclass
class Data:
    prod_dir: Path
    out_dir: Path
    cv: np.ndarray
    cv2: np.ndarray
    rg_A: np.ndarray
    window: np.ndarray
    replica: np.ndarray
    step: np.ndarray
    u_nk: np.ndarray
    centers: np.ndarray
    k_kcal: np.ndarray
    beta: float
    temp: float
    boost_kj: np.ndarray
    potential_kj: Optional[np.ndarray]
    source: str
    meta: dict[str,Any]
    boost_dih_kj: Optional[np.ndarray] = None  # dihedral-only component of GaMD boost
