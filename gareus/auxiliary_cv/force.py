"""OpenMM CustomCVForce for one auxiliary-CV restraint: select(aux_k, 0.5 aux_k (z - aux_c)^2, 0).

z = (offset + sum of sub-CVs) / scale is the model's FULL scalar (squared as a whole, keeping cross
terms). Sub-CVs are grouped by (trig, sign convention) so mixed-convention models stay exact, and
never use periodic boundary conditions (``periodic_imaging: "none"``). The ``select`` makes an
inactive state (aux_k = 0) contribute exactly zero energy even when (z - aux_c)^2 overflows; its
forces are exactly zero at non-degenerate geometry (a degenerate torsion gives NaN forces, as
OpenMM's own torsion forces do). Globals are kJ/mol per z^2 and z units.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from ..correctness._io import IntegrityError
from ..correctness.bias import KJ_PER_KCAL, finite_number
from .model import AuxModel

AUX_FORCE_NAME = "ATLaSAuxCVUmbrella"
AUX_GLOBAL_K = "aux_k"
AUX_GLOBAL_C = "aux_c"


@dataclass(frozen=True)
class AuxForceInfo:
    name: str
    force_group: int
    model_sha256: str
    global_k: str
    global_c: str
    sub_cv_names: tuple[str, ...]


def build_aux_force(openmm, model: AuxModel, *, force_group: int):
    if isinstance(force_group, bool) or not isinstance(force_group, int) or not 0 <= force_group <= 31:
        raise IntegrityError(f"aux force group must be an integer in 0..31, got {force_group!r}")
    groups: dict[tuple[str, int], list[tuple[tuple[int, ...], float]]] = defaultdict(list)
    for feature, coeff in zip(model.feature_schema.features, model.coefficients):
        if coeff == 0.0:
            continue
        sign = -1 if feature.dihedral_sign_convention == "negated" else 1
        groups[(feature.trig, sign)].append((tuple(feature.atom_indices), float(coeff)))
    cv = openmm.CustomCVForce("0")
    names = []
    for (trig, sign) in sorted(groups):
        name = f"aux_{trig}_{'neg' if sign < 0 else 'dir'}"
        tf = openmm.CustomTorsionForce(f"w*{trig}({'-' if sign < 0 else ''}theta)")
        tf.addPerTorsionParameter("w")
        tf.setUsesPeriodicBoundaryConditions(False)
        for quad, weight in groups[(trig, sign)]:
            tf.addTorsion(*[int(a) for a in quad], [weight])
        cv.addCollectiveVariable(name, tf)
        names.append(name)
    z_expr = f"(({model.offset:.17g})" + "".join(f" + {n}" for n in names) + f")/({model.scale:.17g})"
    cv.setEnergyFunction(
        f"select({AUX_GLOBAL_K}, 0.5*{AUX_GLOBAL_K}*(auxz-{AUX_GLOBAL_C})^2, 0); auxz = {z_expr}")
    cv.addGlobalParameter(AUX_GLOBAL_K, 0.0)
    cv.addGlobalParameter(AUX_GLOBAL_C, 0.0)
    cv.setForceGroup(int(force_group))
    cv.setName(AUX_FORCE_NAME)
    info = AuxForceInfo(AUX_FORCE_NAME, int(force_group), model.model_sha256,
                        AUX_GLOBAL_K, AUX_GLOBAL_C, tuple(names))
    return cv, info


def aux_parameter_values(*, center, k_kcal) -> tuple[float, float]:
    """Validated (k kJ/mol/z^2, centre) for the aux globals; (0, 0) when inactive.

    Pure: raises IntegrityError before any Context is touched. The centre is not evaluated
    when k == 0.
    """
    k = finite_number(k_kcal, "aux_k", minimum=0)
    if k == 0.0:
        return 0.0, 0.0
    k_kj = finite_number(k * KJ_PER_KCAL, "aux_k (kJ/mol)", minimum=0)
    c = finite_number(center, "aux center")
    return k_kj, c


def set_aux_parameters(context, info: AuxForceInfo, *, center, k_kcal) -> None:
    """Set the complete auxiliary target; an inactive state resets both globals to 0."""
    k_kj, c = aux_parameter_values(center=center, k_kcal=k_kcal)
    context.setParameter(info.global_k, k_kj)
    context.setParameter(info.global_c, c)
