# CVaux Stage A: Immutable Definitions and Exact Evaluators — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the frozen auxiliary-CV model, the versioned state schema that carries auxiliary restraints, and one exact energy/gradient implementation shared by the OpenMM force and offline reconstruction. Stage A does not touch production, exchange or storage.

**Architecture:**
- A new package `gareus/auxiliary_cv/` holds three pieces: an immutable model (`model.py`), dihedral features (`features.py`), and a numpy evaluator with analytic gradients (`evaluate.py`).
- An OpenMM `CustomCVForce` builder (`force.py`) must reproduce the evaluator exactly.
- The state schema gains a v2 (`atlas-fixed-state-v2`) next to the untouched v1. `correctness/bias.py` reconstructs the auxiliary term offline from model-indexed z values.
- Stage B wires the force into production and Stage C handles storage. Both get their own plans after this one lands.

**Tech Stack:** Python 3, NumPy, OpenMM 8.5 (Reference/CPU platform in tests), pytest.

**Spec:** `docs/superpowers/specs/2026-10-07-auxiliary-cv-gibbs-production-spec.md` (main dc30285), Sections 3, 5, 8, 16 Stage A and 17. Read Sections 3 and 5 before starting.

**Test runner (user policy):** tests in this repo are run by the local free runner `opencode`, never directly. Each "Run:" step below names the pytest arguments. Dispatch them as
`opencode run "In /run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler: run python -m pytest -q <arguments> and report pass/fail/error counts and every failing test id. Read-only: do not edit, commit or fix anything."`
and treat its report as the result. A Bash hook in this harness refuses commands that call the test runner directly.

**Targeted tests only** (user preference): run the files a task names, never the whole suite.

## Global Constraints

- Energies are kJ/mol inside OpenMM and in the exchange kernel. State tables use kcal/mol. Convert **exactly once** by 4.184 (`KJ_PER_KCAL` in `gareus/correctness/bias.py`).
- An inactive restraint (k = 0) contributes **exactly zero**. Never evaluate `0 * (NaN - centre)**2`. Its canonical form is model null, centre 0.0, k 0.0.
- A missing, negative or nonfinite force constant is an error, never "no pull". Positive strength requires a known model and a finite centre.
- MVP supports **one** auxiliary model per state definition. More than one distinct model is refused with a message naming Stage F.
- Never zero-fill a missing z. A z needed by an active column propagates NaN offline; any runtime use raises.
- Legacy `atlas-fixed-state-v1` definitions must hash **byte-identically** to before this work.
- Use float64 for every auxiliary value.
- Do not assume a fixed force group. `build_aux_force` takes the group from its caller; Stage B allocates it.
- Identity is content: model `label` and `provenance` never enter `model_sha256`. Hash coefficients and normalization, never a filename.
- Angle convention (verified 2026-10-07 on OpenMM 8.5.1):
  - θ = OpenMM `CustomTorsionForce` `theta` = −(the tica/swarm angle, `gareus.tica._dihedral_rad`).
  - A feature with `dihedral_sign_convention: "negated"` (what swarm writes, `gareus/swarm/analyze.py:424`) is `trig(-θ)`; `"direct"` is `trig(θ)`.
  - The Blondel–Karplus formula in Task 3 is dθ/dx: it matched finite differences to 2e-10 on random geometries.

## Review Focus

1. **Dihedral wrap at ±π.** A torsion crossing ±π must give continuous sin/cos features and a continuous z. The Task 2 test pins this.
2. **Collinear (degenerate) torsion atoms.** z is undefined there. The per-structure evaluator and the gradient must raise `AuxGeometryError`; the vectorised offline evaluator returns NaN for that row so analysis can exclude it with a count. The Task 3 test pins both.
3. **One torsion in several features** (its sin and cos, or a phi listed twice). Weights must be applied per feature, not merged by torsion. The Task 4 force and evaluator parity test uses a model with both sin and cos of the same torsion.
4. **kJ/mol state tables.** `aux_k` must be divided by 4.184 together with k1/k2 when `energy_unit == "kJ/mol"`. The Task 5 test pins this.
5. **Mixed sign conventions in one model** (some features `negated`, some `direct`). The force must group sub-CVs by (trig, sign), not by trig alone. The Task 4 parity test covers a mixed model.

---

## File Structure

| File | Responsibility |
|---|---|
| Create `gareus/auxiliary_cv/__init__.py` | Public names re-exported |
| Create `gareus/auxiliary_cv/model.py` | `AuxModel`: parse, validate, content hash, load/write (`atlas-aux-cv-model-v1`) |
| Create `gareus/auxiliary_cv/features.py` | OpenMM-convention dihedrals, unique torsion list, feature values, topology check |
| Create `gareus/auxiliary_cv/evaluate.py` | z from dihedrals / positions, analytic gradient, exact auxiliary energy and forces |
| Create `gareus/auxiliary_cv/force.py` | OpenMM `CustomCVForce` builder + parameter setter |
| Modify `gareus/correctness/bias.py` | `normalize_windows` aux canonicalisation; `reconstruct_bias_matrix(aux_z=)` |
| Modify `gareus/correctness/state_identity.py` | v2 schema, aux model registry, instance metadata, `hamiltonian_sha256` |
| Create `tests/aux_cv_fixture.py` | Shared GA-dipeptide model builder (reuses `tests/pep_gamd_fixture.py`) |
| Create `tests/test_aux_cv_model.py`, `test_aux_cv_features.py`, `test_aux_cv_evaluate.py`, `test_aux_cv_force.py`, `test_aux_cv_state_schema.py`, `test_aux_cv_bias.py` | One test file per unit |

---

### Task 1: Frozen auxiliary model

**Files:**
- Create: `gareus/auxiliary_cv/__init__.py`
- Create: `gareus/auxiliary_cv/model.py`
- Create: `tests/aux_cv_fixture.py`
- Test: `tests/test_aux_cv_model.py`

**Interfaces:**
- Consumes: `gareus.cv_selection.contracts.FeatureSchema` (`.from_mapping(dict)`, `.to_mapping()`, `.features`, `.width`, `.sha256`, `.topology_sha256`), `FEATURE_SCHEMA_VERSION`; `gareus.correctness._io.IntegrityError, digest, json_bytes, json_loads, atomic_bytes`.
- Produces:
  - `AUX_MODEL_SCHEMA = "atlas-aux-cv-model-v1"`;
  - `class AuxModelError(IntegrityError)`;
  - `@dataclass(frozen=True) class AuxModel` with fields `feature_schema: FeatureSchema`, `coefficients: tuple[float, ...]`, `offset: float`, `units: str`, `label: str`, `provenance_json: str`, `model_sha256: str`;
  - methods `AuxModel.from_mapping(raw: Mapping) -> AuxModel`, `AuxModel.load(path) -> AuxModel`, `.to_mapping() -> dict`, `.write(path) -> None`;
  - `tests/aux_cv_fixture.py`: `feature_rows(quads, conventions=None, blocks=None) -> list[dict]`, `model_payload(quads, coefficients, offset=0.0, conventions=None, topology_sha256="0"*64, label="test", provenance=None, blocks=None) -> dict`, `dipeptide() -> dict(topology, positions_nm, quads, labels)`.

- [ ] **Step 1: Write the shared fixture**

```python
# tests/aux_cv_fixture.py
"""Shared builders for the auxiliary-CV tests: model payloads over real backbone torsions."""
from __future__ import annotations

import numpy as np

from gareus.auxiliary_cv.model import AUX_MODEL_SCHEMA
from gareus.cv_selection.contracts import FEATURE_SCHEMA_VERSION


def feature_rows(quads, conventions=None, blocks=None):
    """sin then cos per torsion, in the swarm's canonical order (gareus/swarm/analyze.py:415).

    ``blocks`` gives each torsion's "phi"/"psi" (from backbone_torsion_quads labels) so
    topology checks see true names; without it the names alternate and only synthetic
    (topology-free) tests may use the payload.
    """
    rows = []
    for k, quad in enumerate(quads):
        conv = "negated" if conventions is None else conventions[k]
        block = blocks[k] if blocks is not None else ("phi" if k % 2 == 0 else "psi")
        for trig in ("sin", "cos"):
            rows.append({"index": len(rows), "name": f"{block}-{k}-{trig}", "torsion_name": f"{block}-{k}",
                         "residue_index": int(k), "atom_indices": [int(x) for x in quad],
                         "trig": trig, "dihedral_sign_convention": conv})
    return rows


def model_payload(quads, coefficients, offset=0.0, conventions=None, topology_sha256="0" * 64,
                  label="test", provenance=None, blocks=None):
    return {
        "schema": AUX_MODEL_SCHEMA,
        "feature_schema": {"schema": FEATURE_SCHEMA_VERSION, "topology_sha256": topology_sha256,
                           "features": feature_rows(quads, conventions, blocks)},
        "coefficients": [float(c) for c in coefficients],
        "offset": float(offset),
        "units": "dimensionless",
        "label": label,
        "provenance": provenance if provenance is not None else {"source": "manual"},
    }


def dipeptide():
    """GA dipeptide (solvated, minimised) from pep_gamd_fixture, with its backbone torsions."""
    from pep_gamd_fixture import solvated_dipeptide
    from gareus.imports import import_openmm
    from gareus.mbar_analysis.thermo_frames import backbone_torsion_quads

    _openmm, _app, unit = import_openmm()
    fx = solvated_dipeptide()
    quads, labels = backbone_torsion_quads(fx["topology"], fx["peptide"])
    pos = np.asarray(fx["positions"].value_in_unit(unit.nanometer), dtype=np.float64)
    return {"topology": fx["topology"], "positions_nm": pos, "quads": quads, "labels": labels}
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_aux_cv_model.py
import json

import pytest

from aux_cv_fixture import model_payload
from gareus.auxiliary_cv.model import AUX_MODEL_SCHEMA, AuxModel, AuxModelError
from gareus.correctness._io import IntegrityError

QUADS = [(0, 1, 2, 3), (1, 2, 3, 4)]


def test_round_trip_keeps_identity(tmp_path):
    m = AuxModel.from_mapping(model_payload(QUADS, [0.5, -0.25, 0.0, 1.0], offset=0.3))
    path = tmp_path / "aux.json"
    m.write(path)
    again = AuxModel.load(path)
    assert again == m
    assert again.model_sha256 == m.model_sha256
    assert json.loads(path.read_text())["model_sha256"] == m.model_sha256


def test_label_and_provenance_are_not_identity():
    a = AuxModel.from_mapping(model_payload(QUADS, [1, 0, 0, 0], label="a", provenance={"x": 1}))
    b = AuxModel.from_mapping(model_payload(QUADS, [1, 0, 0, 0], label="b", provenance={"y": 2}))
    assert a.model_sha256 == b.model_sha256


def test_coefficients_offset_and_features_are_identity():
    base = AuxModel.from_mapping(model_payload(QUADS, [1, 0, 0, 0])).model_sha256
    assert AuxModel.from_mapping(model_payload(QUADS, [1, 0, 0, 1e-9])).model_sha256 != base
    assert AuxModel.from_mapping(model_payload(QUADS, [1, 0, 0, 0], offset=1e-9)).model_sha256 != base
    assert AuxModel.from_mapping(model_payload(QUADS, [1, 0, 0, 0],
                                               conventions=["direct", "negated"])).model_sha256 != base


@pytest.mark.parametrize("mutate, message", [
    (lambda p: p.update(coefficients=[1.0, 0.0, 0.0]), "coefficients"),
    (lambda p: p.update(coefficients=[0.0, 0.0, 0.0, 0.0]), "nonzero"),
    (lambda p: p.update(coefficients=[1.0, True, 0.0, 0.0]), "number"),
    (lambda p: p.update(offset=float("nan")), "offset|Nonfinite"),
    (lambda p: p.update(units="angstrom"), "units"),
    (lambda p: p.update(extra=1), "unknown"),
    (lambda p: p.pop("offset"), "missing"),
    (lambda p: p.update(schema="atlas-aux-cv-model-v0"), "schema"),
])
def test_invalid_payloads_are_refused(mutate, message):
    payload = model_payload(QUADS, [1.0, 0.0, 0.0, 0.0])
    mutate(payload)
    with pytest.raises(IntegrityError, match=message):
        AuxModel.from_mapping(payload)


def test_tampered_claimed_digest_is_refused():
    payload = model_payload(QUADS, [1.0, 0.0, 0.0, 0.0])
    payload["model_sha256"] = "f" * 64
    with pytest.raises(AuxModelError, match="digest"):
        AuxModel.from_mapping(payload)


def test_schema_constant():
    assert AUX_MODEL_SCHEMA == "atlas-aux-cv-model-v1"
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `pytest tests/test_aux_cv_model.py`
Expected: FAIL / collection error `ModuleNotFoundError: No module named 'gareus.auxiliary_cv'`

- [ ] **Step 4: Implement the model**

```python
# gareus/auxiliary_cv/__init__.py
"""Auxiliary collective-variable states (spec 2026-10-07-auxiliary-cv-gibbs-production-spec)."""
from .model import AUX_MODEL_SCHEMA, AuxModel, AuxModelError

__all__ = ["AUX_MODEL_SCHEMA", "AuxModel", "AuxModelError"]
```

```python
# gareus/auxiliary_cv/model.py
"""Frozen auxiliary-CV model, schema ``atlas-aux-cv-model-v1``.

z(x) = offset + sum_j coefficients[j] * f_j(x), with f_j = trig_j(s_j * theta_j), where theta_j is
OpenMM's CustomTorsionForce ``theta`` for feature j's atoms and s_j = -1 for the "negated" sign
convention (the tica / swarm convention) or +1 for "direct". Identity is content: ``label`` and
``provenance`` never enter ``model_sha256``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..correctness._io import IntegrityError, atomic_bytes, digest, json_bytes, json_loads
from ..cv_selection.contracts import FeatureSchema

AUX_MODEL_SCHEMA = "atlas-aux-cv-model-v1"
SUPPORTED_UNITS = frozenset({"dimensionless"})
_REQUIRED = {"schema", "feature_schema", "coefficients", "offset", "units"}
_OPTIONAL = {"label", "provenance", "model_sha256"}


class AuxModelError(IntegrityError):
    """An auxiliary-CV model payload cannot be used as the claimed frozen function."""


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AuxModelError(f"{label} must be a number, got {value!r}")
    out = float(value)
    if not math.isfinite(out):
        raise AuxModelError(f"{label} must be finite, got {value!r}")
    return out


@dataclass(frozen=True)
class AuxModel:
    feature_schema: FeatureSchema
    coefficients: tuple[float, ...]
    offset: float
    units: str
    label: str
    provenance_json: str
    model_sha256: str

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "AuxModel":
        data = json_loads(json_bytes(dict(raw)))
        unknown = sorted(set(data) - _REQUIRED - _OPTIONAL)
        if unknown:
            raise AuxModelError(f"aux model has unknown field(s): {', '.join(unknown)}")
        missing = sorted(_REQUIRED - set(data))
        if missing:
            raise AuxModelError(f"aux model is missing field(s): {', '.join(missing)}")
        if data["schema"] != AUX_MODEL_SCHEMA:
            raise AuxModelError(f"aux model requires schema {AUX_MODEL_SCHEMA!r}, got {data['schema']!r}")
        schema = FeatureSchema.from_mapping(data["feature_schema"])
        coeffs = data["coefficients"]
        if not isinstance(coeffs, list) or len(coeffs) != schema.width:
            raise AuxModelError(f"aux model coefficients must list {schema.width} numbers (one per feature)")
        coefficients = tuple(_finite(c, f"coefficients[{i}]") for i, c in enumerate(coeffs))
        if not any(c != 0.0 for c in coefficients):
            raise AuxModelError("aux model needs at least one nonzero coefficient (else z is constant)")
        offset = _finite(data["offset"], "offset")
        if data["units"] not in SUPPORTED_UNITS:
            raise AuxModelError(f"aux model units {data['units']!r} not supported ({sorted(SUPPORTED_UNITS)})")
        label = data.get("label", "")
        if not isinstance(label, str):
            raise AuxModelError("aux model label must be a string")
        provenance = data.get("provenance", {})
        if not isinstance(provenance, dict):
            raise AuxModelError("aux model provenance must be an object")
        identity = {"schema": AUX_MODEL_SCHEMA, "feature_schema": schema.to_mapping(),
                    "coefficients": list(coefficients), "offset": offset, "units": data["units"]}
        sha = digest(json_bytes(identity))
        claimed = data.get("model_sha256")
        if claimed is not None and claimed != sha:
            raise AuxModelError(f"aux model claims digest {claimed} but its contents hash to {sha}")
        return cls(schema, coefficients, offset, data["units"], label,
                   json_bytes(provenance).decode("utf-8"), sha)

    @classmethod
    def load(cls, path: Path | str) -> "AuxModel":
        return cls.from_mapping(json_loads(Path(path).read_bytes()))

    def to_mapping(self) -> dict[str, Any]:
        return {"schema": AUX_MODEL_SCHEMA, "feature_schema": self.feature_schema.to_mapping(),
                "coefficients": list(self.coefficients), "offset": self.offset, "units": self.units,
                "label": self.label, "provenance": json_loads(self.provenance_json),
                "model_sha256": self.model_sha256}

    def write(self, path: Path | str) -> None:
        atomic_bytes(Path(path), json_bytes(self.to_mapping()))
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_aux_cv_model.py`
Expected: PASS (all). If `test_invalid_payloads_are_refused[...missing...]` fails on its message match, the cause is `FeatureSchema` raising first. Adjust the test's `message` only if the raised error is still an `IntegrityError` naming the missing field.

- [ ] **Step 6: Commit**

```bash
git add gareus/auxiliary_cv/__init__.py gareus/auxiliary_cv/model.py tests/aux_cv_fixture.py tests/test_aux_cv_model.py
git commit -m "feat(cvaux): frozen auxiliary-CV model with content identity"
```

---

### Task 2: Dihedral features and topology check

**Files:**
- Create: `gareus/auxiliary_cv/features.py`
- Test: `tests/test_aux_cv_features.py`

**Interfaces:**
- Consumes: `AuxModel` (Task 1); `tests/aux_cv_fixture.dipeptide()`.
- Produces:
  - `class AuxGeometryError(IntegrityError)`;
  - `openmm_dihedrals(xyz_nm, quads) -> np.ndarray` of shape (n_frames, n_torsions), equal to OpenMM `theta`, NaN where the torsion is degenerate;
  - `unique_torsions(model) -> tuple[tuple[tuple[int,int,int,int], ...], np.ndarray]` (the quads in first-appearance order, and the per-feature torsion index);
  - `feature_signs(model) -> np.ndarray` (s_j = -1 for negated, +1 for direct);
  - `feature_values(theta, model) -> np.ndarray` of shape (n_frames, width);
  - `check_feature_atoms(model, topology, *, topology_sha256: str | None = None) -> None`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_aux_cv_features.py
import numpy as np
import pytest

from aux_cv_fixture import dipeptide, model_payload
from gareus.auxiliary_cv.features import (AuxGeometryError, check_feature_atoms, feature_signs,
                                          feature_values, openmm_dihedrals, unique_torsions)
from gareus.auxiliary_cv.model import AuxModel
from gareus.correctness._io import IntegrityError


def _openmm_theta(xyz_nm, quad):
    import openmm as mm
    from openmm import unit
    s = mm.System()
    for _ in range(len(xyz_nm)):
        s.addParticle(1.0)
    f = mm.CustomTorsionForce("theta")
    f.addTorsion(*[int(i) for i in quad], [])
    s.addForce(f)
    c = mm.Context(s, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference"))
    c.setPositions(xyz_nm)
    return c.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)


def test_dihedrals_equal_openmm_theta_on_real_backbone():
    d = dipeptide()
    theta = openmm_dihedrals(d["positions_nm"], d["quads"])[0]
    for t, quad in zip(theta, d["quads"]):
        assert t == pytest.approx(_openmm_theta(d["positions_nm"], quad), abs=1e-9)


def test_dihedrals_are_minus_the_tica_angle():
    from gareus.tica import _dihedral_rad
    rng = np.random.default_rng(3)
    p = rng.normal(size=(4, 3))
    assert openmm_dihedrals(p, [(0, 1, 2, 3)])[0, 0] == pytest.approx(-_dihedral_rad(*p), abs=1e-12)


def test_degenerate_torsion_gives_nan():
    p = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 1, 0]], dtype=float)  # p0,p1,p2 collinear
    assert np.isnan(openmm_dihedrals(p, [(0, 1, 2, 3)])[0, 0])


def test_features_continuous_across_pi():
    # rotate p3 about the b2 axis through +pi: theta wraps from just below +pi to just above -pi
    angles = np.array([np.pi - 1e-6, -np.pi + 1e-6])
    m = AuxModel.from_mapping(model_payload([(0, 1, 2, 3)], [1.0, 1.0]))
    f = feature_values(angles[:, None], m)
    assert np.abs(f[0] - f[1]).max() < 1e-5


def test_negated_and_direct_signs():
    m = AuxModel.from_mapping(model_payload([(0, 1, 2, 3), (1, 2, 3, 4)], [1, 1, 1, 1],
                                            conventions=["negated", "direct"]))
    np.testing.assert_array_equal(feature_signs(m), [-1, -1, 1, 1])
    theta = np.array([[0.3, 0.3]])
    np.testing.assert_allclose(feature_values(theta, m)[0],
                               [np.sin(-0.3), np.cos(-0.3), np.sin(0.3), np.cos(0.3)], atol=1e-15)


def test_unique_torsions_shares_sin_and_cos():
    m = AuxModel.from_mapping(model_payload([(0, 1, 2, 3), (1, 2, 3, 4)], [1, 0, 0, 1]))
    quads, idx = unique_torsions(m)
    assert quads == ((0, 1, 2, 3), (1, 2, 3, 4))
    np.testing.assert_array_equal(idx, [0, 0, 1, 1])


def _blocks(d):
    return [lab.split("-")[0] for lab in d["labels"]]


def test_topology_check_accepts_real_backbone_and_rejects_shifted_atoms():
    d = dipeptide()
    m = AuxModel.from_mapping(model_payload(d["quads"], [1.0] + [0.0] * (2 * len(d["quads"]) - 1),
                                            blocks=_blocks(d)))
    check_feature_atoms(m, d["topology"])
    bad = [tuple(a + 1 for a in q) for q in d["quads"]]
    m_bad = AuxModel.from_mapping(model_payload(bad, [1.0] + [0.0] * (2 * len(bad) - 1),
                                                blocks=_blocks(d)))
    with pytest.raises(IntegrityError, match="atom names"):
        check_feature_atoms(m_bad, d["topology"])


def test_topology_check_compares_topology_digest():
    d = dipeptide()
    m = AuxModel.from_mapping(model_payload(d["quads"], [1.0] + [0.0] * (2 * len(d["quads"]) - 1),
                                            blocks=_blocks(d)))
    with pytest.raises(IntegrityError, match="topology"):
        check_feature_atoms(m, d["topology"], topology_sha256="a" * 64)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_aux_cv_features.py`
Expected: FAIL `ModuleNotFoundError: No module named 'gareus.auxiliary_cv.features'`

- [ ] **Step 3: Implement the features**

```python
# gareus/auxiliary_cv/features.py
"""Dihedral features of an auxiliary-CV model, in OpenMM's ``theta`` convention.

theta = OpenMM CustomTorsionForce ``theta`` = -(gareus.tica._dihedral_rad). A "negated" feature is
trig(-theta) (the tica / swarm convention), a "direct" feature trig(theta).
"""
from __future__ import annotations

import numpy as np

from ..correctness._io import IntegrityError
from .model import AuxModel

#: |n1| and |n2| below this (nm^2) make the torsion undefined (collinear atoms).
DEGENERATE_NORM2_NM2 = 1e-12


class AuxGeometryError(IntegrityError):
    """A configuration where an auxiliary CV is undefined (degenerate torsion)."""


def openmm_dihedrals(xyz_nm, quads) -> np.ndarray:
    """(n_frames, n_torsions) OpenMM-convention dihedrals in radians; NaN where degenerate."""
    xyz = np.asarray(xyz_nm, dtype=np.float64)
    if xyz.ndim == 2:
        xyz = xyz[None]
    q = np.asarray(quads, dtype=np.int64).reshape(-1, 4)
    p0, p1, p2, p3 = (xyz[:, q[:, k], :] for k in range(4))
    b1, b2, b3 = p1 - p0, p2 - p1, p3 - p2
    n1, n2 = np.cross(b1, b2), np.cross(b2, b3)
    nn1, nn2 = np.sum(n1 * n1, axis=-1), np.sum(n2 * n2, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        b2n = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
        m1 = np.cross(n1, b2n)
        theta = -np.arctan2(np.sum(m1 * n2, axis=-1), np.sum(n1 * n2, axis=-1))
    theta[(nn1 < DEGENERATE_NORM2_NM2) | (nn2 < DEGENERATE_NORM2_NM2)] = np.nan
    return theta


def unique_torsions(model: AuxModel):
    quads: list[tuple[int, int, int, int]] = []
    where: dict[tuple[int, int, int, int], int] = {}
    idx = np.empty(model.feature_schema.width, dtype=np.int64)
    for j, feature in enumerate(model.feature_schema.features):
        quad = tuple(int(a) for a in feature.atom_indices)
        if quad not in where:
            where[quad] = len(quads)
            quads.append(quad)
        idx[j] = where[quad]
    return tuple(quads), idx


def feature_signs(model: AuxModel) -> np.ndarray:
    return np.array([-1.0 if f.dihedral_sign_convention == "negated" else 1.0
                     for f in model.feature_schema.features])


def feature_values(theta, model: AuxModel) -> np.ndarray:
    """(n_frames, width) feature matrix from per-unique-torsion theta (n_frames, n_unique)."""
    theta = np.asarray(theta, dtype=np.float64)
    _quads, idx = unique_torsions(model)
    arg = theta[:, idx] * feature_signs(model)[None, :]
    is_sin = np.array([f.trig == "sin" for f in model.feature_schema.features])
    return np.where(is_sin[None, :], np.sin(arg), np.cos(arg))


_EXPECTED = {"phi": ("C", "N", "CA", "C"), "psi": ("N", "CA", "C", "N")}


def check_feature_atoms(model: AuxModel, topology, *, topology_sha256: str | None = None) -> None:
    """Refuse a model whose torsions are not the backbone phi/psi they claim on this topology."""
    if topology_sha256 is not None and topology_sha256 != model.feature_schema.topology_sha256:
        raise IntegrityError(f"aux model was built for topology {model.feature_schema.topology_sha256}, "
                             f"this run's topology is {topology_sha256}")
    atoms = list(topology.atoms())
    for feature in model.feature_schema.features:
        block = feature.torsion_name.split("-")[0]
        if block not in _EXPECTED:
            raise IntegrityError(f"feature {feature.name}: torsion_name must start with phi or psi")
        quad = feature.atom_indices
        if max(quad) >= len(atoms):
            raise IntegrityError(f"feature {feature.name}: atom index beyond topology ({len(atoms)} atoms)")
        names = tuple(atoms[i].name for i in quad)
        if names != _EXPECTED[block]:
            raise IntegrityError(f"feature {feature.name}: atom names {names} are not a {block} "
                                 f"{_EXPECTED[block]}")
        res = [atoms[i].residue.index for i in quad]
        ok = (res[0] + 1 == res[1] == res[2] == res[3]) if block == "phi" else (res[0] == res[1] == res[2] == res[3] - 1)
        if not ok:
            raise IntegrityError(f"feature {feature.name}: atoms span residues {res}, not one {block}")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_aux_cv_features.py tests/test_aux_cv_model.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add gareus/auxiliary_cv/features.py tests/aux_cv_fixture.py tests/test_aux_cv_features.py
git commit -m "feat(cvaux): OpenMM-convention dihedral features and topology check"
```

---

### Task 3: Exact evaluator with analytic gradient

**Files:**
- Create: `gareus/auxiliary_cv/evaluate.py`
- Modify: `gareus/auxiliary_cv/__init__.py` (re-export)
- Test: `tests/test_aux_cv_evaluate.py`

**Interfaces:**
- Consumes: Task 2 `openmm_dihedrals`, `unique_torsions`, `feature_signs`, `feature_values`, `AuxGeometryError`; `gareus.correctness.bias.KJ_PER_KCAL, finite_number`.
- Produces:
  - `z_from_dihedrals(theta, model) -> np.ndarray` of shape (n_frames,), with NaN where any used torsion is NaN;
  - `z_from_positions(xyz_nm, model) -> np.ndarray` of shape (n_frames,);
  - `z_and_gradient(xyz_nm, model) -> tuple[float, np.ndarray]` (the gradient has shape (n_atoms, 3), in z per nm; raises `AuxGeometryError` on a degenerate torsion);
  - `aux_energy_kj(z, center, k_kcal) -> np.ndarray`;
  - `aux_forces_kj_nm(xyz_nm, model, center, k_kcal) -> np.ndarray` of shape (n_atoms, 3).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_aux_cv_evaluate.py
import numpy as np
import pytest

from aux_cv_fixture import dipeptide, model_payload
from gareus.auxiliary_cv.evaluate import (aux_energy_kj, aux_forces_kj_nm, z_and_gradient,
                                          z_from_dihedrals, z_from_positions)
from gareus.auxiliary_cv.features import AuxGeometryError, openmm_dihedrals, unique_torsions
from gareus.auxiliary_cv.model import AuxModel
from gareus.correctness._io import IntegrityError


def _mixed_model(quads, blocks=None):
    rng = np.random.default_rng(7)
    coeffs = rng.normal(size=2 * len(quads))
    conv = ["negated" if k % 2 == 0 else "direct" for k in range(len(quads))]
    return AuxModel.from_mapping(model_payload(quads, coeffs, offset=0.4, conventions=conv, blocks=blocks))


def test_z_matches_hand_formula():
    m = AuxModel.from_mapping(model_payload([(0, 1, 2, 3)], [2.0, -1.0], offset=0.5))
    theta = np.array([[0.7]])
    expected = 0.5 + 2.0 * np.sin(-0.7) - 1.0 * np.cos(-0.7)
    assert z_from_dihedrals(theta, m)[0] == pytest.approx(expected, abs=1e-15)


def test_positions_and_dihedral_paths_agree_on_real_peptide():
    d = dipeptide()
    m = _mixed_model(d["quads"], blocks=[lab.split("-")[0] for lab in d["labels"]])
    quads, _ = unique_torsions(m)
    a = z_from_positions(d["positions_nm"], m)[0]
    b = z_from_dihedrals(openmm_dihedrals(d["positions_nm"], quads), m)[0]
    c, _g = z_and_gradient(d["positions_nm"], m)
    assert a == pytest.approx(b, abs=1e-12) and a == pytest.approx(c, abs=1e-12)


def test_gradient_matches_finite_differences():
    d = dipeptide()
    m = _mixed_model(d["quads"], blocks=[lab.split("-")[0] for lab in d["labels"]])
    x = d["positions_nm"].copy()
    _z, g = z_and_gradient(x, m)
    atoms = sorted({a for q in d["quads"] for a in q})
    h = 1e-6
    for i in atoms:
        for k in range(3):
            xp, xm = x.copy(), x.copy()
            xp[i, k] += h
            xm[i, k] -= h
            fd = (z_from_positions(xp, m)[0] - z_from_positions(xm, m)[0]) / (2 * h)
            assert g[i, k] == pytest.approx(fd, rel=1e-5, abs=1e-6)
    untouched = np.setdiff1d(np.arange(len(x)), atoms)
    assert not np.any(g[untouched])


def test_forces_are_minus_energy_gradient_in_kj_nm():
    d = dipeptide()
    m = _mixed_model(d["quads"], blocks=[lab.split("-")[0] for lab in d["labels"]])
    x = d["positions_nm"].copy()
    z, g = z_and_gradient(x, m)
    f = aux_forces_kj_nm(x, m, center=z - 0.3, k_kcal=2.0)
    np.testing.assert_allclose(f, -2.0 * 4.184 * 0.3 * g, rtol=1e-12, atol=1e-12)


def test_zero_strength_is_exact_zero_even_for_nan_z():
    e = aux_energy_kj(np.array([np.nan, 1.0, -3.0]), center=0.0, k_kcal=0.0)
    assert e.dtype == np.float64 and np.all(e == 0.0)


def test_energy_units_and_value():
    e = aux_energy_kj(np.array([1.5]), center=0.5, k_kcal=2.0)
    assert e[0] == pytest.approx(0.5 * 2.0 * 4.184 * 1.0, abs=1e-12)


@pytest.mark.parametrize("bad_k", [-1.0, float("nan"), float("inf"), None, True])
def test_invalid_strength_is_an_error(bad_k):
    with pytest.raises(IntegrityError):
        aux_energy_kj(np.array([1.0]), center=0.0, k_kcal=bad_k)


def test_active_strength_needs_finite_center():
    with pytest.raises(IntegrityError, match="center"):
        aux_energy_kj(np.array([1.0]), center=float("nan"), k_kcal=1.0)


def test_degenerate_geometry_raises_in_gradient_and_is_nan_offline():
    p = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 1, 0], [4, 1, 1]], dtype=float)
    m = AuxModel.from_mapping(model_payload([(0, 1, 2, 3)], [1.0, 0.0]))
    assert np.isnan(z_from_positions(p, m)[0])
    with pytest.raises(AuxGeometryError):
        z_and_gradient(p, m)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_aux_cv_evaluate.py`
Expected: FAIL `ModuleNotFoundError: No module named 'gareus.auxiliary_cv.evaluate'`

- [ ] **Step 3: Implement the evaluator**

```python
# gareus/auxiliary_cv/evaluate.py
"""Exact auxiliary-CV evaluator: z, analytic dz/dx and the harmonic restraint 0.5 k (z - c)^2.

Units: positions nm, energies kJ/mol, forces kJ/mol/nm, k given in kcal/mol per z^2 (converted
exactly once by KJ_PER_KCAL). This module is the numerical reference for the OpenMM force.
"""
from __future__ import annotations

import numpy as np

from ..correctness._io import IntegrityError
from ..correctness.bias import KJ_PER_KCAL, finite_number
from .features import (AuxGeometryError, feature_signs, feature_values, openmm_dihedrals,
                       unique_torsions)
from .model import AuxModel


def z_from_dihedrals(theta, model: AuxModel) -> np.ndarray:
    feats = feature_values(theta, model)
    return model.offset + feats @ np.asarray(model.coefficients, dtype=np.float64)


def z_from_positions(xyz_nm, model: AuxModel) -> np.ndarray:
    quads, _idx = unique_torsions(model)
    return z_from_dihedrals(openmm_dihedrals(xyz_nm, quads), model)


def _dtheta_dx(p: np.ndarray) -> np.ndarray:
    """Blondel-Karplus gradient of OpenMM theta w.r.t. the 4 atoms, shape (4, 3).

    Verified against finite differences of openmm_dihedrals (2e-10) on random geometries.
    """
    p0, p1, p2, p3 = p
    F, G, H = p0 - p1, p1 - p2, p3 - p2
    A, B = np.cross(F, G), np.cross(H, G)
    A2, B2, Gn = A @ A, B @ B, np.linalg.norm(G)
    g0 = -Gn / A2 * A
    g3 = Gn / B2 * B
    g1 = Gn / A2 * A + (F @ G) / (A2 * Gn) * A - (H @ G) / (B2 * Gn) * B
    g2 = -Gn / B2 * B - (F @ G) / (A2 * Gn) * A + (H @ G) / (B2 * Gn) * B
    return np.array([g0, g1, g2, g3])


def z_and_gradient(xyz_nm, model: AuxModel) -> tuple[float, np.ndarray]:
    x = np.asarray(xyz_nm, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 3:
        raise IntegrityError("z_and_gradient takes one configuration of shape (n_atoms, 3)")
    quads, idx = unique_torsions(model)
    theta = openmm_dihedrals(x, quads)[0]
    if np.isnan(theta).any():
        bad = [quads[t] for t in np.flatnonzero(np.isnan(theta))]
        raise AuxGeometryError(f"auxiliary CV undefined: degenerate torsion(s) {bad}")
    signs = feature_signs(model)
    coeffs = np.asarray(model.coefficients, dtype=np.float64)
    is_sin = np.array([f.trig == "sin" for f in model.feature_schema.features])
    arg = theta[idx] * signs
    # d trig(s*theta)/d theta: sin -> s cos(s theta); cos -> -s sin(s theta)
    dfeat = np.where(is_sin, signs * np.cos(arg), -signs * np.sin(arg))
    dz_dtheta = np.zeros(len(quads))
    np.add.at(dz_dtheta, idx, coeffs * dfeat)
    grad = np.zeros_like(x)
    for t, quad in enumerate(quads):
        if dz_dtheta[t] != 0.0:
            grad[list(quad)] += dz_dtheta[t] * _dtheta_dx(x[list(quad)])
    z = float(model.offset + feature_values(theta[None, :], model)[0] @ coeffs)
    return z, grad


def _strength(k_kcal) -> float:
    return finite_number(k_kcal, "aux_k", minimum=0)


def aux_energy_kj(z, center, k_kcal) -> np.ndarray:
    k = _strength(k_kcal)
    z = np.asarray(z, dtype=np.float64)
    if k == 0.0:
        return np.zeros_like(z)          # exact zero; never 0 * (NaN - c)^2
    c = finite_number(center, "aux center")
    return 0.5 * k * KJ_PER_KCAL * (z - c) ** 2


def aux_forces_kj_nm(xyz_nm, model: AuxModel, center, k_kcal) -> np.ndarray:
    k = _strength(k_kcal)
    x = np.asarray(xyz_nm, dtype=np.float64)
    if k == 0.0:
        return np.zeros_like(x)
    c = finite_number(center, "aux center")
    z, grad = z_and_gradient(x, model)
    return -k * KJ_PER_KCAL * (z - c) * grad
```

Add to `gareus/auxiliary_cv/__init__.py`:

```python
from .evaluate import aux_energy_kj, aux_forces_kj_nm, z_and_gradient, z_from_dihedrals, z_from_positions
from .features import AuxGeometryError, check_feature_atoms, openmm_dihedrals

__all__ += ["aux_energy_kj", "aux_forces_kj_nm", "z_and_gradient", "z_from_dihedrals",
            "z_from_positions", "AuxGeometryError", "check_feature_atoms", "openmm_dihedrals"]
```

`finite_number(value, label, minimum=0)` raises `IntegrityError` for None, booleans, NaN, inf and negatives, so `test_invalid_strength_is_an_error` needs no extra code. The `center` NaN case goes through `finite_number(center, "aux center")`, whose message contains "center".

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_aux_cv_evaluate.py tests/test_aux_cv_features.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add gareus/auxiliary_cv/evaluate.py gareus/auxiliary_cv/__init__.py tests/test_aux_cv_evaluate.py
git commit -m "feat(cvaux): exact auxiliary energy, analytic gradient and forces"
```

---

### Task 4: OpenMM force with exact parity

**Files:**
- Create: `gareus/auxiliary_cv/force.py`
- Modify: `gareus/auxiliary_cv/__init__.py`
- Test: `tests/test_aux_cv_force.py`

**Interfaces:**
- Consumes: `AuxModel`; Task 3 evaluator (as the reference in tests); `gareus.correctness.bias.KJ_PER_KCAL, finite_number`.
- Produces:
  - `AUX_FORCE_NAME = "ATLaSAuxCVUmbrella"`, `AUX_GLOBAL_K = "aux_k"`, `AUX_GLOBAL_C = "aux_c"`;
  - `@dataclass(frozen=True) class AuxForceInfo(name: str, force_group: int, model_sha256: str, global_k: str, global_c: str, sub_cv_names: tuple[str, ...])`;
  - `build_aux_force(openmm, model, *, force_group: int) -> tuple[object, AuxForceInfo]` (the caller adds the force to the System);
  - `set_aux_parameters(context, info, *, center, k_kcal) -> None`, which sets both globals and resets the centre to 0.0 when k = 0.

Stage B will call `build_aux_force` from production setup and `set_aux_parameters` from the `set_window` path.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_aux_cv_force.py
import numpy as np
import pytest

from aux_cv_fixture import dipeptide, model_payload
from gareus.auxiliary_cv.evaluate import aux_energy_kj, aux_forces_kj_nm, z_from_positions
from gareus.auxiliary_cv.force import (AUX_FORCE_NAME, build_aux_force, set_aux_parameters)
from gareus.auxiliary_cv.model import AuxModel
from gareus.correctness._io import IntegrityError


def _context(model, n_atoms, force_group=7):
    import openmm as mm
    s = mm.System()
    for _ in range(n_atoms):
        s.addParticle(1.0)
    force, info = build_aux_force(mm, model, force_group=force_group)
    s.addForce(force)
    ctx = mm.Context(s, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference"))
    return ctx, info, force


def _energy_forces(ctx):
    from openmm import unit
    st = ctx.getState(getEnergy=True, getForces=True)
    return (st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
            np.asarray(st.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)))


def _model(d, conventions=None, coeffs=None, offset=0.4):
    rng = np.random.default_rng(11)
    width = 2 * len(d["quads"])
    return AuxModel.from_mapping(model_payload(
        d["quads"], rng.normal(size=width) if coeffs is None else coeffs, offset=offset,
        conventions=conventions, blocks=[lab.split("-")[0] for lab in d["labels"]]))


@pytest.mark.parametrize("conventions", [None, "mixed"])
def test_energy_and_forces_match_reference(conventions):
    d = dipeptide()
    conv = None if conventions is None else ["negated" if k % 2 == 0 else "direct" for k in range(len(d["quads"]))]
    m = _model(d, conv)
    ctx, info, _ = _context(m, len(d["positions_nm"]))
    ctx.setPositions(d["positions_nm"])
    z = z_from_positions(d["positions_nm"], m)[0]
    set_aux_parameters(ctx, info, center=z - 0.25, k_kcal=3.0)
    e, f = _energy_forces(ctx)
    assert e == pytest.approx(aux_energy_kj(np.array([z]), z - 0.25, 3.0)[0], rel=1e-10, abs=1e-10)
    np.testing.assert_allclose(f, aux_forces_kj_nm(d["positions_nm"], m, z - 0.25, 3.0), rtol=1e-8, atol=1e-8)


def test_zero_strength_contributes_exactly_nothing():
    d = dipeptide()
    m = _model(d)
    ctx, info, _ = _context(m, len(d["positions_nm"]))
    ctx.setPositions(d["positions_nm"])
    set_aux_parameters(ctx, info, center=5.0, k_kcal=0.0)
    e, f = _energy_forces(ctx)
    assert e == 0.0 and not np.any(f)
    assert ctx.getParameter(info.global_c) == 0.0


def test_full_square_keeps_cross_terms():
    d = dipeptide()
    width = 2 * len(d["quads"])
    ca = np.zeros(width); ca[0] = 1.3
    cb = np.zeros(width); cb[-1] = -0.8
    m_ab = _model(d, coeffs=ca + cb, offset=0.2)
    za = z_from_positions(d["positions_nm"], _model(d, coeffs=ca, offset=0.0))[0]
    zb = z_from_positions(d["positions_nm"], _model(d, coeffs=cb, offset=0.0))[0]
    ctx, info, _ = _context(m_ab, len(d["positions_nm"]))
    ctx.setPositions(d["positions_nm"])
    set_aux_parameters(ctx, info, center=0.0, k_kcal=1.0)
    e, _ = _energy_forces(ctx)
    full = 0.5 * 4.184 * (za + zb + 0.2) ** 2
    separate = 0.5 * 4.184 * ((za + 0.2) ** 2 + zb ** 2)
    assert e == pytest.approx(full, rel=1e-10)
    assert abs(full - separate) > 1e-3


def test_force_metadata_and_group():
    d = dipeptide()
    m = _model(d)
    _ctx, info, force = _context(m, len(d["positions_nm"]), force_group=12)
    assert force.getName() == AUX_FORCE_NAME and force.getForceGroup() == 12
    assert info.force_group == 12 and info.model_sha256 == m.model_sha256


@pytest.mark.parametrize("group", [-1, 32])
def test_force_group_range(group):
    import openmm as mm
    d = dipeptide()
    with pytest.raises(IntegrityError, match="force group"):
        build_aux_force(mm, _model(d), force_group=group)


def test_set_parameters_rejects_invalid_strength():
    d = dipeptide()
    ctx, info, _ = _context(_model(d), len(d["positions_nm"]))
    with pytest.raises(IntegrityError):
        set_aux_parameters(ctx, info, center=0.0, k_kcal=-1.0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_aux_cv_force.py`
Expected: FAIL `ModuleNotFoundError: No module named 'gareus.auxiliary_cv.force'`

- [ ] **Step 3: Implement the force**

```python
# gareus/auxiliary_cv/force.py
"""OpenMM CustomCVForce for one auxiliary-CV restraint: select(aux_k, 0.5 aux_k (z - aux_c)^2, 0).

z is the model's FULL scalar sum (squared as a whole, keeping cross terms). Sub-CVs are grouped
by (trig, sign convention) so mixed-convention models stay exact. The ``select`` makes an
inactive state (aux_k = 0) contribute exactly zero. Globals are kJ/mol per z^2 and z units.
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
        for quad, weight in groups[(trig, sign)]:
            tf.addTorsion(*[int(a) for a in quad], [weight])
        cv.addCollectiveVariable(name, tf)
        names.append(name)
    z_expr = f"({model.offset:.17g})" + "".join(f" + {n}" for n in names)
    cv.setEnergyFunction(
        f"select({AUX_GLOBAL_K}, 0.5*{AUX_GLOBAL_K}*(auxz-{AUX_GLOBAL_C})^2, 0); auxz = {z_expr}")
    cv.addGlobalParameter(AUX_GLOBAL_K, 0.0)
    cv.addGlobalParameter(AUX_GLOBAL_C, 0.0)
    cv.setForceGroup(int(force_group))
    cv.setName(AUX_FORCE_NAME)
    info = AuxForceInfo(AUX_FORCE_NAME, int(force_group), model.model_sha256,
                        AUX_GLOBAL_K, AUX_GLOBAL_C, tuple(names))
    return cv, info


def set_aux_parameters(context, info: AuxForceInfo, *, center, k_kcal) -> None:
    """Set the complete auxiliary target; an inactive state resets both globals to 0."""
    k = finite_number(k_kcal, "aux_k", minimum=0)
    if k == 0.0:
        context.setParameter(info.global_k, 0.0)
        context.setParameter(info.global_c, 0.0)
        return
    c = finite_number(center, "aux center")
    context.setParameter(info.global_k, k * KJ_PER_KCAL)
    context.setParameter(info.global_c, c)
```

Add to `__init__.py`: `from .force import AUX_FORCE_NAME, AuxForceInfo, build_aux_force, set_aux_parameters` and extend `__all__`.

Two things to know when debugging:
- OpenMM's Lepton parser accepts `select(x, y, z)`, which returns y when x ≠ 0, else z. If the parser rejects the `auxz` intermediate, inline `z_expr` in both places instead.
- `CustomTorsionForce` `theta` is the OpenMM angle. That is why a `negated` feature uses `sin(-theta)` (see `gareus/production.py:454` `_add_weighted_trig_torsion_force`).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_aux_cv_force.py tests/test_aux_cv_evaluate.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add gareus/auxiliary_cv/force.py gareus/auxiliary_cv/__init__.py tests/test_aux_cv_force.py
git commit -m "feat(cvaux): exact CustomCVForce auxiliary restraint with zero-strength select"
```

---

### Task 5: State schema v2 with auxiliary restraints and instance metadata

**Files:**
- Modify: `gareus/correctness/bias.py` (`normalize_windows`, new `_aux_term`)
- Modify: `gareus/correctness/state_identity.py`
- Test: `tests/test_aux_cv_state_schema.py`

**Interfaces:**
- Consumes: `AuxModel.from_mapping` (Task 1).
- Produces:
  - `state_identity.STATE_SCHEMA_V2 = "atlas-fixed-state-v2"` and `STATE_ROLES = frozenset({"ordinary", "auxiliary", "sham"})`;
  - `make_state_definition(..., aux_models: Mapping[str, Mapping] | None = None)`: a v2 definition when `aux_models` is not None, else v1 unchanged;
  - `hamiltonian_sha256(definition, window_id: int) -> str`;
  - normalized window rows: v2 rows always carry `aux_model_sha256` (str | None), `aux_center` (float) and `aux_k` (float, kcal/mol per z²), plus an optional `instance` dict with exactly the keys `state_instance_id` (nonempty string, unique across rows, stable across phases, unlike `window_id`, which is the per-phase column index), `state_role`, `spawn_parent_state_id`, `spawn_source_observation` (null, or an object with exactly `run`, `segment`, `carrier`, `state`, `checkpoint`, `step`) and `matched_additional_slot_id` (null or nonempty string). v1 rows are unchanged.

Rules (spec Sections 3.1 and 5):
- In v2, every window must state `aux_k` explicitly. k = 0 canonicalises to model None and centre 0.0. k > 0 needs a model present in `aux_models` and a finite centre.
- `aux_models` maps sha → embedded model payload. The key must equal the parsed `model_sha256`. More than one model is refused ("Stage F").
- `instance` is excluded from `hamiltonian_sha256` but included in the definition hash, since the slot table is part of the phase identity.
- In a v2 definition, either every row carries `instance` or none does. Instance ids must be unique (spec Section 5: `state_instance_id` is mapped explicitly to the runtime window index, which is `window_id` here).
- `hamiltonian_sha256` drops the aux keys of an inactive row, so an inactive v2 row hashes like the same v1 physics.

- [ ] **Step 1: Confirm the legacy v1 pin on unmodified code**

`V1_PINNED` below was computed on main dc30285 with `gareus/correctness/` unchanged. Before editing, run this snippet and confirm it prints `dc0acf77e275ed0d1b357238e16e3f1f8248859b06ab647d7e6578817e4c8725`. If it prints something else, `gareus/correctness/` has changed since this plan was written; stop and report rather than re-pinning:

```bash
python - <<'EOF'
from gareus.correctness.state_identity import make_state_definition, state_definition_hash
d = make_state_definition(
    [{"window_id": 0, "center1": 0.2, "k1": 10.0, "center2": 0.0, "k2": 0.0, "gamd_lambda": 0.0},
     {"window_id": 1, "center1": 0.4, "k1": 10.0, "center2": 1.0, "k2": 2.0, "gamd_lambda": 0.0}],
    physical_system_sha256="a" * 64, ensemble="NVT", temperature_k=300.0,
    fixed_box_vectors_nm=[[3, 0, 0], [0, 3, 0], [0, 0, 3]],
    cv1={"kind": "contacts", "units": "dimensionless", "definition": {"r0": 4.5}},
    cv2={"kind": "residual", "units": "dimensionless", "definition": {"v": [1.0]}})
print(state_definition_hash(d))
EOF
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_aux_cv_state_schema.py
import pytest

from aux_cv_fixture import model_payload
from gareus.auxiliary_cv.model import AuxModel
from gareus.correctness._io import IntegrityError
from gareus.correctness.state_identity import (STATE_SCHEMA_V2, hamiltonian_sha256,
                                               make_state_definition, state_definition_hash)

V1_PINNED = "dc0acf77e275ed0d1b357238e16e3f1f8248859b06ab647d7e6578817e4c8725"
BOX = [[3, 0, 0], [0, 3, 0], [0, 0, 3]]
CV1 = {"kind": "contacts", "units": "dimensionless", "definition": {"r0": 4.5}}
CV2 = {"kind": "residual", "units": "dimensionless", "definition": {"v": [1.0]}}
MODEL = AuxModel.from_mapping(model_payload([(0, 1, 2, 3)], [1.0, 0.5], offset=0.1))
SHA = MODEL.model_sha256


def _legacy_windows():
    return [{"window_id": 0, "center1": 0.2, "k1": 10.0, "center2": 0.0, "k2": 0.0, "gamd_lambda": 0.0},
            {"window_id": 1, "center1": 0.4, "k1": 10.0, "center2": 1.0, "k2": 2.0, "gamd_lambda": 0.0}]


def _defn(windows, aux_models=..., energy_unit="kcal/mol"):
    kw = {} if aux_models is ... else {"aux_models": aux_models}
    return make_state_definition(windows, physical_system_sha256="a" * 64, ensemble="NVT",
                                 temperature_k=300.0, fixed_box_vectors_nm=BOX, cv1=CV1, cv2=CV2,
                                 energy_unit=energy_unit, **kw)


def _v2_windows():
    base = _legacy_windows()
    for k, row in enumerate(base):
        row.update(aux_k=0.0, instance={"state_instance_id": f"ord-{k}", "state_role": "ordinary",
                                        "spawn_parent_state_id": None, "spawn_source_observation": None,
                                        "matched_additional_slot_id": None})
    base.append({"window_id": 2, "center1": 0.4, "k1": 10.0, "center2": 1.0, "k2": 2.0,
                 "gamd_lambda": 0.0, "aux_model_sha256": SHA, "aux_center": 1.5, "aux_k": 1.2,
                 "instance": {"state_instance_id": "aux-0", "state_role": "auxiliary", "spawn_parent_state_id": 1,
                              "spawn_source_observation": {"run": "r", "segment": "s", "carrier": 3,
                                                           "state": 1, "checkpoint": "c", "step": 100},
                              "matched_additional_slot_id": "slot-0"}})
    base.append({"window_id": 3, "center1": 0.4, "k1": 10.0, "center2": 1.0, "k2": 2.0,
                 "gamd_lambda": 0.0, "aux_k": 0.0,
                 "instance": {"state_instance_id": "sham-0", "state_role": "sham", "spawn_parent_state_id": 1,
                              "spawn_source_observation": {"run": "r", "segment": "s", "carrier": 3,
                                                           "state": 1, "checkpoint": "c", "step": 100},
                              "matched_additional_slot_id": "slot-0"}})
    return base


def test_legacy_v1_hash_is_unchanged():
    assert state_definition_hash(_defn(_legacy_windows())) == V1_PINNED


def test_v2_round_trip_and_canonical_inactive():
    d = _defn(_v2_windows(), aux_models={SHA: MODEL.to_mapping()})
    assert d["schema"] == STATE_SCHEMA_V2
    rows = {r["window_id"]: r for r in d["windows"]}
    assert rows[0]["aux_model_sha256"] is None and rows[0]["aux_center"] == 0.0 and rows[0]["aux_k"] == 0.0
    assert rows[2]["aux_model_sha256"] == SHA and rows[2]["aux_k"] == 1.2
    assert state_definition_hash(d) == state_definition_hash(_defn(_v2_windows(), aux_models={SHA: MODEL.to_mapping()}))


def test_sham_and_parent_share_hamiltonian_but_not_auxiliary():
    d = _defn(_v2_windows(), aux_models={SHA: MODEL.to_mapping()})
    assert hamiltonian_sha256(d, 3) == hamiltonian_sha256(d, 1)
    assert hamiltonian_sha256(d, 2) != hamiltonian_sha256(d, 1)


def test_inactive_v2_row_hashes_like_v1_physics():
    v1 = _defn(_legacy_windows())
    v2 = _defn(_v2_windows(), aux_models={SHA: MODEL.to_mapping()})
    assert hamiltonian_sha256(v1, 1) == hamiltonian_sha256(v2, 1)


def test_kj_tables_convert_aux_k():
    w = _v2_windows()
    for row in w:
        row["k1"] *= 4.184
        row["k2"] *= 4.184
        row["aux_k"] *= 4.184
    d = _defn(w, aux_models={SHA: MODEL.to_mapping()}, energy_unit="kJ/mol")
    assert {r["window_id"]: r for r in d["windows"]}[2]["aux_k"] == pytest.approx(1.2, abs=1e-12)


@pytest.mark.parametrize("mutate, message", [
    (lambda w, m: w[0].pop("aux_k"), "aux_k"),
    (lambda w, m: w[2].update(aux_k=-1.0), "aux_k"),
    (lambda w, m: w[2].update(aux_k=float("nan")), "aux_k|Nonfinite"),
    (lambda w, m: w[2].pop("aux_center"), "aux_center"),
    (lambda w, m: w[2].update(aux_model_sha256="b" * 64), "unknown auxiliary model"),
    (lambda w, m: w[2]["instance"].update(state_role="worker"), "state_role"),
    (lambda w, m: w[2]["instance"].pop("matched_additional_slot_id"), "instance"),
    (lambda w, m: w[3]["instance"].update(state_instance_id="aux-0"), "duplicate state_instance_id"),
    (lambda w, m: w[3]["instance"].update(state_instance_id=""), "state_instance_id"),
    (lambda w, m: w[2]["instance"]["spawn_source_observation"].pop("step"), "spawn_source_observation"),
    (lambda w, m: w[0].pop("instance"), "every row"),
    (lambda w, m: m.update({"c" * 64: m.pop(SHA)}), "key"),
])
def test_v2_refusals(mutate, message):
    w = _v2_windows()
    m = {SHA: MODEL.to_mapping()}
    mutate(w, m)
    with pytest.raises(IntegrityError, match=message):
        _defn(w, aux_models=m)


def test_more_than_one_model_is_stage_f():
    other = AuxModel.from_mapping(model_payload([(0, 1, 2, 3)], [0.0, 1.0]))
    with pytest.raises(IntegrityError, match="Stage F"):
        _defn(_v2_windows(), aux_models={SHA: MODEL.to_mapping(), other.model_sha256: other.to_mapping()})


def test_v1_rejects_aux_fields():
    w = _legacy_windows()
    w[0]["aux_k"] = 0.0
    with pytest.raises(IntegrityError, match="unknown physics fields"):
        _defn(w)
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `pytest tests/test_aux_cv_state_schema.py`
Expected: FAIL `ImportError: cannot import name 'STATE_SCHEMA_V2'`. `test_legacy_v1_hash_is_unchanged` would already pass on its own.

- [ ] **Step 4: Implement `_aux_term` in `gareus/correctness/bias.py`**

Add after `_term`:

```python
AUX_WINDOW_FIELDS = ("aux_model_sha256", "aux_center", "aux_k")


def _aux_term(window: Mapping[str, Any], label: str) -> tuple[str | None, float, float]:
    """Canonical (model sha, centre, k) of an auxiliary restraint; k = 0 -> (None, 0.0, 0.0)."""
    if "aux_k" not in window:
        raise IntegrityError(f"{label}: auxiliary fields require an explicit aux_k (inactive is 0)")
    force = finite_number(window["aux_k"], f"{label}.aux_k", minimum=0)
    if force == 0:
        return None, 0.0, 0.0
    sha = window.get("aux_model_sha256")
    if (not isinstance(sha, str) or len(sha) != 64
            or any(c not in "0123456789abcdef" for c in sha)):
        raise IntegrityError(f"{label}: active aux_k requires a full aux_model_sha256")
    if "aux_center" not in window:
        raise IntegrityError(f"{label}: active aux_k requires aux_center")
    return sha, finite_number(window["aux_center"], f"{label}.aux_center"), force
```

In `normalize_windows`, after the `gamd_lambda` line and before `normalized.append(window)`, add:

```python
        if any(key in raw for key in AUX_WINDOW_FIELDS):
            (window["aux_model_sha256"], window["aux_center"],
             window["aux_k"]) = _aux_term(raw, label)
```

Rows without aux keys are untouched, so v1 bytes do not change.

- [ ] **Step 5: Implement v2 in `gareus/correctness/state_identity.py`**

Add the constants under `STATE_SCHEMA`:

```python
STATE_SCHEMA_V2 = "atlas-fixed-state-v2"
STATE_ROLES = frozenset({"ordinary", "auxiliary", "sham"})
_V1_WINDOW_FIELDS = {"window_id", "center1", "k1", "center2", "k2", "gamd_lambda"}
_V2_WINDOW_FIELDS = _V1_WINDOW_FIELDS | {"aux_model_sha256", "aux_center", "aux_k", "instance"}
_INSTANCE_FIELDS = {"state_instance_id", "state_role", "spawn_parent_state_id",
                    "spawn_source_observation", "matched_additional_slot_id"}
_OBSERVATION_FIELDS = {"run", "segment", "carrier", "state", "checkpoint", "step"}
_SHARED_FIELDS = ("physical_system_sha256", "ensemble", "temperature_k", "pressure_bar",
                  "fixed_box_vectors_nm", "cv1", "cv2", "boost", "energy_unit")
```

Add the helpers:

```python
def _canonical_aux_models(raw: Any) -> dict[str, dict]:
    from ..auxiliary_cv.model import AuxModel   # lazy: avoids a cv_selection import cycle
    if not isinstance(raw, dict):
        raise IntegrityError("aux_models must map model_sha256 -> embedded model payload")
    if len(raw) > 1:
        raise IntegrityError("More than one auxiliary model per state definition is Stage F "
                             "(spec Section 16); MVP supports one")
    out = {}
    for key, payload in raw.items():
        model = AuxModel.from_mapping(payload)
        if key != model.model_sha256:
            raise IntegrityError(f"aux_models key {key} does not match the model's digest {model.model_sha256}")
        out[key] = model.to_mapping()
    return out


def _canonical_instance(raw: Any, state_id: int) -> dict:
    if not isinstance(raw, dict) or set(raw) != _INSTANCE_FIELDS:
        raise IntegrityError(f"Window {state_id} instance needs exactly {sorted(_INSTANCE_FIELDS)}")
    if raw["state_role"] not in STATE_ROLES:
        raise IntegrityError(f"Window {state_id} instance.state_role must be one of {sorted(STATE_ROLES)}")
    if not isinstance(raw["state_instance_id"], str) or not raw["state_instance_id"]:
        raise IntegrityError(f"Window {state_id} instance.state_instance_id must be a nonempty string")
    parent = raw["spawn_parent_state_id"]
    if parent is not None and (isinstance(parent, bool) or not isinstance(parent, int) or parent < 0):
        raise IntegrityError(f"Window {state_id} instance.spawn_parent_state_id must be null or an id")
    obs = raw["spawn_source_observation"]
    if obs is not None and (not isinstance(obs, dict) or set(obs) != _OBSERVATION_FIELDS):
        raise IntegrityError(f"Window {state_id} instance.spawn_source_observation must be null or "
                             f"exactly {sorted(_OBSERVATION_FIELDS)}")
    slot = raw["matched_additional_slot_id"]
    if slot is not None and (not isinstance(slot, str) or not slot):
        raise IntegrityError(f"Window {state_id} instance.matched_additional_slot_id must be null or a nonempty string")
    return json_loads(json_bytes(raw))
```

Then change `canonical_state_definition`:
- Replace the `required = {...}` check with:

```python
    schema = data.get("schema")
    base_required = {"schema", "physical_system_sha256", "ensemble", "temperature_k",
                     "pressure_bar", "fixed_box_vectors_nm", "cv1", "cv2",
                     "boost", "energy_unit", "windows"}
    if schema == STATE_SCHEMA:
        required, allowed_window = base_required, _V1_WINDOW_FIELDS
    elif schema == STATE_SCHEMA_V2:
        required, allowed_window = base_required | {"aux_models"}, _V2_WINDOW_FIELDS
    else:
        raise IntegrityError(f"State definition requires schema {STATE_SCHEMA} or {STATE_SCHEMA_V2}")
    if set(data) != required:
        raise IntegrityError(f"State definition requires exact schema {schema}; missing/extra fields")
    aux_models = _canonical_aux_models(data["aux_models"]) if schema == STATE_SCHEMA_V2 else None
```

- In the window loop, replace `if set(row) - {...}:` with `if set(row) - allowed_window:`. Keep the same message ("has unknown physics fields; normalize explicitly"). Then add, inside the loop after the k1/k2 unit conversion:

```python
        if schema == STATE_SCHEMA_V2:
            if "aux_k" not in row:
                raise IntegrityError(f"Window {state_id}: v2 requires explicit aux_k (inactive is 0)")
            if energy_unit == "kJ/mol":
                row["aux_k"] /= 4.184
            if row["aux_k"] > 0 and row["aux_model_sha256"] not in aux_models:
                raise IntegrityError(f"Window {state_id} names an unknown auxiliary model "
                                     f"{row['aux_model_sha256']}")
            if "instance" in row:
                row["instance"] = _canonical_instance(row["instance"], state_id)
```

- After the loop, before `return data`: `if aux_models is not None: data["aux_models"] = aux_models`.
- Also after the loop, for v2 only, enforce instance consistency:

```python
    if schema == STATE_SCHEMA_V2:
        with_instance = [row for row in windows if "instance" in row]
        if with_instance and len(with_instance) != len(windows):
            raise IntegrityError("v2 instance metadata must be given on every row or on none")
        ids = [row["instance"]["state_instance_id"] for row in with_instance]
        if len(set(ids)) != len(ids):
            raise IntegrityError(f"duplicate state_instance_id in state table: {sorted(ids)}")
```
- Change `make_state_definition`: add the keyword `aux_models: Mapping | None = None`. Build the dict as today. When `aux_models is not None`, set `"schema": STATE_SCHEMA_V2` and `"aux_models": dict(aux_models)`.

Add the Hamiltonian hash:

```python
def hamiltonian_sha256(definition: Mapping[str, Any], window_id: int) -> str:
    """Identity of one state's potential and ensemble, excluding slot id and spawn provenance."""
    state = canonical_state_definition(definition)
    rows = [row for row in state["windows"] if row["window_id"] == window_id]
    if len(rows) != 1:
        raise IntegrityError(f"No window {window_id} in this state definition")
    physics = {key: value for key, value in rows[0].items() if key not in ("window_id", "instance")}
    model = None
    if physics.get("aux_k", 0.0) > 0:
        model = state["aux_models"][physics["aux_model_sha256"]]
    else:
        for key in ("aux_model_sha256", "aux_center", "aux_k"):
            physics.pop(key, None)
    shared = {key: state[key] for key in _SHARED_FIELDS}
    return digest(json_bytes({"shared": shared, "window": physics, "aux_model": model}))
```

The kJ conversion order matters. `normalize_windows` canonicalises `aux_k` in the input unit, and `canonical_state_definition` then divides by 4.184 once, as it does for k1 and k2. Do not convert in `_aux_term`.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `pytest tests/test_aux_cv_state_schema.py`
Expected: PASS. No dedicated state-identity test file existed before this work, so the pinned v1 hash in this file is the legacy regression.

- [ ] **Step 7: Commit**

```bash
git add gareus/correctness/bias.py gareus/correctness/state_identity.py tests/test_aux_cv_state_schema.py
git commit -m "feat(cvaux): fixed-state schema v2 with auxiliary restraints and instance metadata"
```

---

### Task 6: Strict offline reconstruction of the auxiliary term

**Files:**
- Modify: `gareus/correctness/bias.py` (`reconstruct_bias_matrix`)
- Test: `tests/test_aux_cv_bias.py`

**Interfaces:**
- Consumes: Task 5 normalized rows (`aux_model_sha256`, `aux_center`, `aux_k`).
- Produces: `reconstruct_bias_matrix(..., aux_z: Mapping[str, np.ndarray] | None = None)`, where `aux_z[model_sha256]` is the (N,) z of every sample under that model (float64, NaN where undefined). Stage C's loaders pass it.

Rules:
- An active aux column (k > 0) without `aux_z[sha]` raises `MissingCoordinateError`.
- NaN in z propagates to that column only.
- An inactive row never reads z.
- The term is added inside the umbrella loop, before the `lambda == 0` early return (spec Section 15 flags this exact trap).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_aux_cv_bias.py
import numpy as np
import pytest

from gareus.correctness.bias import MissingCoordinateError, reconstruct_bias_matrix

SHA = "e" * 64
BETA = 1.0 / (0.0083144626 * 300.0)


def _rows():
    return [{"window_id": 0, "center1": 0.2, "k1": 10.0, "center2": 0.0, "k2": 0.0, "gamd_lambda": 0.0},
            {"window_id": 1, "center1": 0.2, "k1": 10.0, "center2": 0.0, "k2": 0.0, "gamd_lambda": 0.0,
             "aux_model_sha256": SHA, "aux_center": 1.0, "aux_k": 2.0},
            {"window_id": 2, "center1": 0.2, "k1": 10.0, "center2": 0.0, "k2": 0.0, "gamd_lambda": 0.0,
             "aux_k": 0.0}]


def test_aux_term_added_in_lambda_zero_path():
    cv1 = np.array([0.1, 0.3])
    z = np.array([0.5, 2.0])
    u = reconstruct_bias_matrix(cv1, None, _rows(), BETA, aux_z={SHA: z})
    extra = BETA * 4.184 * 0.5 * 2.0 * (z - 1.0) ** 2
    np.testing.assert_allclose(u[:, 1] - u[:, 0], extra, rtol=1e-12)
    np.testing.assert_array_equal(u[:, 2], u[:, 0])


def test_missing_model_column_raises():
    with pytest.raises(MissingCoordinateError, match="aux"):
        reconstruct_bias_matrix(np.array([0.1]), None, _rows(), BETA)


def test_nan_z_only_poisons_active_aux_column():
    u = reconstruct_bias_matrix(np.array([0.1, 0.3]), None, _rows(), BETA,
                                aux_z={SHA: np.array([np.nan, 1.0])})
    assert np.isnan(u[0, 1]) and np.isfinite(u[0, 0]) and np.isfinite(u[0, 2])
    assert np.isfinite(u[1]).all()


def test_zero_strength_rows_equal_legacy_matrix_bitwise():
    legacy = [{k: v for k, v in r.items() if not k.startswith("aux_")} for r in _rows()]
    rows = _rows()
    rows[1]["aux_k"] = 0.0
    cv1 = np.array([0.1, 0.3, 0.7])
    a = reconstruct_bias_matrix(cv1, None, legacy, BETA)
    b = reconstruct_bias_matrix(cv1, None, rows, BETA)
    assert np.array_equal(a, b)


def test_aux_z_shape_is_checked():
    with pytest.raises(Exception, match="shape"):
        reconstruct_bias_matrix(np.array([0.1, 0.3]), None, _rows(), BETA, aux_z={SHA: np.array([1.0])})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_aux_cv_bias.py`
Expected: FAIL `TypeError: reconstruct_bias_matrix() got an unexpected keyword argument 'aux_z'`

- [ ] **Step 3: Implement**

In `reconstruct_bias_matrix`, add the keyword `aux_z: Mapping[str, np.ndarray] | None = None` after `meta`. After `rows = normalize_windows(windows)` and the secondary check, add:

```python
    aux_columns = {}
    for column, row in enumerate(rows):
        if row.get("aux_k", 0.0) > 0:
            sha = row["aux_model_sha256"]
            if aux_z is None or sha not in aux_z:
                raise MissingCoordinateError(
                    f"aux z for model {sha} was not supplied, but state "
                    f"{row.get('window_id', column)} has an active auxiliary restraint")
            if sha not in aux_columns:
                aux_columns[sha] = numeric_vector(aux_z[sha], f"aux_z[{sha[:12]}]", n)
```

In the umbrella loop, after the `k2` block, add:

```python
                if row.get("aux_k", 0.0) > 0:
                    delta = aux_columns[row["aux_model_sha256"]] - row["aux_center"]
                    matrix[:, column] += 0.5 * row["aux_k"] * delta * delta
```

`numeric_vector(..., n)` raises `IntegrityError("... must have shape (n,) ...")`, which satisfies `test_aux_z_shape_is_checked`. Update the docstring: "aux_z maps model_sha256 to that model's z for every sample; required for every model with an active state."

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_aux_cv_bias.py tests/test_aux_cv_state_schema.py tests/test_query_reconstruct_bias_matrix_nan_guard.py tests/test_query.py tests/test_lambda_ladder_mbar.py` (the last three are the existing strict-bias regressions; `gareus.query.reconstruct_bias_matrix` wraps the correctness one).
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add gareus/correctness/bias.py tests/test_aux_cv_bias.py
git commit -m "feat(cvaux): strict offline reconstruction of the auxiliary bias term"
```

---

### Task 7: Stage A exit gate and handoff note

**Files:**
- Modify: `CLAUDE.md` (project handoff; add one section)
- Test: all Stage A files

- [ ] **Step 1: Run the whole Stage A set plus the legacy regressions**

Run: `pytest tests/test_aux_cv_model.py tests/test_aux_cv_features.py tests/test_aux_cv_evaluate.py tests/test_aux_cv_force.py tests/test_aux_cv_state_schema.py tests/test_aux_cv_bias.py` `tests/test_query_reconstruct_bias_matrix_nan_guard.py tests/test_query.py tests/test_lambda_ladder_mbar.py`.
Expected: all PASS. This is spec Section 16's Stage A exit: zero-strength, topology, sign, units and gradient tests.

- [ ] **Step 2: Add the handoff section to `CLAUDE.md`**

Insert after the "Adaptive λ ladder" section:

```markdown
## CVaux Stage A (`gareus/auxiliary_cv/`, spec `docs/superpowers/specs/2026-10-07-auxiliary-cv-gibbs-production-spec.md`)

- Built: frozen model `atlas-aux-cv-model-v1` (`AuxModel`, identity = content sha, label/provenance excluded), OpenMM-convention dihedral features (theta = OpenMM `theta` = -tica angle; `negated` feature = trig(-theta)), exact evaluator with Blondel-Karplus gradient (FD-verified), `CustomCVForce` `ATLaSAuxCVUmbrella` = select(aux_k, 0.5 aux_k (z - aux_c)^2, 0) with exact energy/force parity, state schema `atlas-fixed-state-v2` (`aux_models` registry, per-window aux_model_sha256/aux_center/aux_k, optional `instance` {state_instance_id, state_role, spawn_parent_state_id, spawn_source_observation, matched_additional_slot_id}, `hamiltonian_sha256`), `reconstruct_bias_matrix(aux_z=)`.
- Not wired: nothing in production, exchange, storage or analysis calls it yet (Stages B/C). v1 definitions hash byte-identically (pinned test). MVP: one model; >1 refused ("Stage F").
- Tests: `tests/test_aux_cv_*.py`, fixture `tests/aux_cv_fixture.py`.
```

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs(cvaux): Stage A handoff note"
```

---

## Self-review record

- **Spec coverage, Stage A** (spec Section 16: "model registry, state schema extension, canonical identity and shared auxiliary energy evaluator; analytic gradients and an OpenMM force expression; pure offline reconstruction path and an independently evaluated numerical reference"):

  | Spec requirement | Task |
  |---|---|
  | Model | Task 1 |
  | Features, topology, sign | Task 2 |
  | Evaluator and gradient (the numerical reference) | Task 3 |
  | Force expression | Task 4 |
  | Schema, identity, instance metadata, `hamiltonian_sha256` | Task 5 |
  | Offline reconstruction | Task 6 |

- **Section 17 rows covered here:**

  | Test row | Where |
  |---|---|
  | Scalar sum vs sum of squares | Task 4 |
  | Angle sign and wrapping | Task 2 |
  | Gradient check | Task 3 |
  | Zero strength | Tasks 3, 4, 6 |
  | Lambda-zero early return | Task 6 |
  | Missing feature fails closed | Task 6 |

  The remaining Section 17 rows belong to Stages B/C/D: force-group composition, direct cross-energy, Gibbs permutations, NPT, resume, and the c10 frame parity (it needs the recovered artifact).

- **Deliberately absent:**
  - force-group allocation (Stage B, through a new audit helper; Stage A takes the group as a parameter);
  - production `set_window` wiring (Stage B);
  - Parquet/feature storage (Stage C);
  - c10 model recovery (Stage D).
- **Interface names used later:** `AuxModel.model_sha256`, `build_aux_force(openmm, model, *, force_group)`, `set_aux_parameters(context, info, *, center, k_kcal)`, `z_from_dihedrals(theta, model)`, `openmm_dihedrals(xyz_nm, quads)`, `unique_torsions(model)`, `reconstruct_bias_matrix(..., aux_z=)`, `hamiltonian_sha256(definition, window_id)`.
