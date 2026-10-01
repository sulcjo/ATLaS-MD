"""The CV1 a residual CV2 is fitted against ("anchor"), by kind (spec 2026-10-01-generic-cv1-anchor).

Every supported anchor is ``c = sum_p w_p f(r_p) / norm`` over atom pairs, in the CV1's own
units, so the compiled residual coordinate (``residual_runtime``: ``c = anchor_sub_cv / norm``)
is unchanged; only how the anchor sub-CV is built, evaluated, measured in the swarm and bound
into the pair model depends on the kind:

| kind | pairs | f(r) | norm | units |
|---|---|---|---|---|
| ``nonlocal-contact-fraction`` | heavy pairs (contact list) | tanh switch | contact denominator | dimensionless |
| ``end-to-end-distance`` | one atom pair (terminal CA) | r (nm) | 0.1 nm per A | angstrom |

This module is the one place that knows those differences. Contacts keep every byte they
had: their definition, binding (``contact_pair_list_sha256``), force expression and k bounds.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..correctness._io import digest, json_bytes

KIND_CONTACT = "nonlocal-contact-fraction"
KIND_E2E = "end-to-end-distance"
ANCHOR_KINDS = (KIND_CONTACT, KIND_E2E)
E2E_ATOM_RULE = "terminal-ca"
NM_PER_ANGSTROM = 0.1
#: Default CV1 force-constant bounds of the distance anchor, kcal/mol/A^2, when --cv1-k-min /
#: --cv1-k-max are unset (0). sigma_w at 300 K: 3.5 A at 0.05, 0.24 A at 10.
E2E_DEFAULT_K_MIN = 0.05
E2E_DEFAULT_K_MAX = 10.0


def anchor_kind_for_args(args) -> str:
    """The anchor kind the run's CV1 defines; refuses CV1s without a residual-CV2 anchor."""
    from ..cv import primary_cv_mode

    mode = primary_cv_mode(args)
    if mode == "nonlocal-contacts":
        return KIND_CONTACT
    if mode == "distance":
        explicit = bool(getattr(args, "cv_atom1", None)) and bool(getattr(args, "cv_atom2", None))
        rule = str(getattr(args, "cv_mode", E2E_ATOM_RULE) or E2E_ATOM_RULE)
        if explicit or rule != E2E_ATOM_RULE:
            raise ValueError("cv2=auto with --cv1 distance supports the terminal CA--CA distance only "
                             "(the swarm records exactly that as e2e_nm); unset --cv-atom1/--cv-atom2")
        return KIND_E2E
    raise ValueError(f"no residual-CV2 anchor for primary CV mode {mode!r}")


def units(kind: str) -> str:
    _check(kind)
    return "dimensionless" if kind == KIND_CONTACT else "angstrom"


def is_contact(kind: str) -> bool:
    _check(kind)
    return kind == KIND_CONTACT


def trace_values(kind: str, trace: Mapping[str, np.ndarray]) -> np.ndarray:
    """The anchor's per-frame values from one swarm member trace (``cv1`` = heavy contacts)."""
    _check(kind)
    if kind == KIND_CONTACT:
        return np.asarray(trace["cv1"], dtype=np.float64)
    return np.asarray(trace["e2e_nm"], dtype=np.float64) / NM_PER_ANGSTROM


def descriptor_value(kind: str, row: Mapping[str, Any]) -> float:
    """The anchor value of one ``seed_descriptors.csv`` row."""
    _check(kind)
    if kind == KIND_CONTACT:
        return float(row["cv1"])
    return float(row["e2e_nm"]) / NM_PER_ANGSTROM


def k_bounds(kind: str, args) -> Tuple[float, float]:
    """(k_min, k_max) of the CV1 ladder, in kcal/mol per anchor unit^2."""
    _check(kind)
    if kind == KIND_CONTACT:
        return (float(getattr(args, "contact_adaptive_min_k_kcal", 5.0)),
                float(getattr(args, "contact_adaptive_max_k_kcal", 1200.0)))
    k_min = float(getattr(args, "cv1_k_min", 0.0) or 0.0)
    k_max = float(getattr(args, "cv1_k_max", 0.0) or 0.0)
    return (k_min if k_min > 0 else E2E_DEFAULT_K_MIN, k_max if k_max > 0 else E2E_DEFAULT_K_MAX)


def value_bounds(kind: str) -> Tuple[Optional[float], Optional[float]]:
    """Physical clip range of the anchor value (the contact fraction lives in [0, 1])."""
    _check(kind)
    return (0.0, 1.0) if kind == KIND_CONTACT else (0.0, None)


def e2e_definition(atom1: int, atom2: int) -> Dict[str, Any]:
    return {"atom1": int(atom1), "atom2": int(atom2), "atom_rule": E2E_ATOM_RULE, "units": "angstrom"}


def binding_digest(kind: str, definition: Mapping[str, Any]) -> Optional[str]:
    """The kind's deployment binding (``anchor_binding_sha256``); None for contacts, which
    keep ``contact_pair_list_sha256``."""
    _check(kind)
    if kind == KIND_CONTACT:
        return None
    return digest(json_bytes({"kind": kind, "atom1": int(definition["atom1"]), "atom2": int(definition["atom2"]),
                              "atom_rule": str(definition["atom_rule"])}))


def anchor_pairs(kind: str, primary_cv_def: Mapping[str, Any]) -> List[Tuple[int, int, float]]:
    """The atom pairs (i, j, weight) of the anchor sub-CV, from the run's primary CV definition."""
    _check(kind)
    if kind == KIND_CONTACT:
        return [tuple(p) for p in primary_cv_def.get("contact_pairs", [])]
    a1, a2 = primary_cv_def.get("cv_atom1"), primary_cv_def.get("cv_atom2")
    if a1 is None or a2 is None:
        raise RuntimeError("distance anchor needs cv_atom1/cv_atom2 in the primary CV definition")
    return [(int(a1), int(a2), 1.0)]


def anchor_norm(kind: str, definition: Mapping[str, Any], pairs: Sequence, args) -> float:
    """``c = sub_cv / norm``: the contact denominator, or 0.1 (nm per A) for a distance in A."""
    _check(kind)
    if kind == KIND_CONTACT:
        from ..cv import contact_normalization_denominator
        return float(definition.get("norm", contact_normalization_denominator(list(pairs), args)))
    return NM_PER_ANGSTROM


def build_anchor_subcv(openmm, kind: str, pairs: Sequence, args):
    """The anchor sub-CV force (raw sum, before ``/ norm``), non-periodic like the CV1 umbrellas."""
    _check(kind)
    if kind == KIND_CONTACT:
        from ..forces import contact_switch_constants_nm
        r0_nm, beta_nm_inv = contact_switch_constants_nm(args)
        force = openmm.CustomBondForce(
            f"contact_weight*0.5*(1-tanh(0.5*{beta_nm_inv:.17g}*(r-{r0_nm:.17g})))")
        force.addPerBondParameter("contact_weight")
        for pair in pairs:
            force.addBond(int(pair[0]), int(pair[1]), [float(pair[2]) if len(pair) > 2 else 1.0])
        return force
    if len(pairs) != 1:
        raise ValueError("a distance anchor has exactly one atom pair")
    force = openmm.CustomBondForce("r")
    force.addBond(int(pairs[0][0]), int(pairs[0][1]), [])
    return force


def value_from_positions(kind: str, positions_nm, pairs: Sequence, runtime) -> float:
    """The anchor value at positions (nm), as the force computes it (no periodic images)."""
    _check(kind)
    if kind == KIND_CONTACT:
        from ..cv import nonlocal_contact_cv_from_positions_nm
        return float(nonlocal_contact_cv_from_positions_nm(positions_nm, pairs, runtime.contact_args()))
    x = np.asarray(positions_nm, dtype=np.float64)
    i, j = int(pairs[0][0]), int(pairs[0][1])
    return float(np.linalg.norm(x[j] - x[i]) / NM_PER_ANGSTROM)


def check_run_matches(kind: str, definition: Mapping[str, Any], pairs: Sequence) -> None:
    """A distance pair model deploys only on the same atom pair (the contact kind is checked by
    ``PairModelRuntime.check_anchor``'s parameter + pair-list comparison)."""
    _check(kind)
    if kind == KIND_CONTACT:
        return
    if len(pairs) != 1:
        raise RuntimeError("distance anchor: the run's CV1 is not a single atom pair")
    got = (int(pairs[0][0]), int(pairs[0][1]))
    want = (int(definition["atom1"]), int(definition["atom2"]))
    if got != want:
        raise RuntimeError(f"anchor atom pair mismatch: model {want}, run CV1 {got}")


def _check(kind: str) -> None:
    if kind not in ANCHOR_KINDS:
        raise ValueError(f"unknown anchor kind {kind!r}; supported: {ANCHOR_KINDS}")


__all__ = ["ANCHOR_KINDS", "E2E_ATOM_RULE", "E2E_DEFAULT_K_MAX", "E2E_DEFAULT_K_MIN", "KIND_CONTACT", "KIND_E2E",
           "NM_PER_ANGSTROM", "anchor_kind_for_args", "anchor_norm", "anchor_pairs", "binding_digest",
           "build_anchor_subcv", "check_run_matches", "descriptor_value", "e2e_definition", "is_contact",
           "k_bounds", "trace_values", "units", "value_bounds", "value_from_positions"]
