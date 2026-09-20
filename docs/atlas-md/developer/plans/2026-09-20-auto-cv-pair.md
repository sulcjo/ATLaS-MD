# In-Run Automatic CV2 Selection for a Fixed Contact CV1 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Version:** 0.2 (2026-09-20). v0.1 was adversarially reviewed by two independent agents and a three-judge panel; every confirmed defect is folded in below and listed in the *Verification round* section at the end. **Status: not executed.**

**Goal:** One `gareus` submission keeps the configured heavy-atom contact CV1, automatically picks CV2 as one residualised backbone-torsion component from the swarm's unbiased epoch-0 data (exact zero covariance with CV1 under the swarm measure), freezes the pair, and runs the 2-D × λ-rung production on it — native-blind, with an honest abstention path.

**Scope change from v0.1.** Automatic CV1 (`cv1: auto`) is **removed from this plan**. Reason, verified in code: `gareus/swarm/analyze.py` builds the CV1 umbrella from the pooled *contact* trace unconditionally (`cv1_all = _pool(ok_traces, "cv1", …)` → `cv1_centers_from_samples` → `cv1_force_constants_from_curvature`), `gareus/production.py::add_primary_umbrella_force` implements only `nonlocal-contacts`/`distance`, and `gareus/cli.py::_resolve_cv_aliases` collapses any unknown `--cv1` value to `"distance"`. An auto-selected Rg anchor would have the residual fitted against Rg while the umbrella restrains contacts — `Cov_q(z2, z1) = 0` asserted about a coordinate nothing restrains, silently. Anchor *ranking* survives as a diagnostic that can refuse the configured anchor (Task 3); choosing a different one is a later plan.

**Architecture:** Members additionally record canonical torsion features at trace cadence. Inside `analyze_swarm_stage`, after the frozen Pep-GaMD envelope and before ladder design, `gareus/cv_selection/{anchor,models,independence,select_pair}.py` fit six residual components against the contact CV, score them on frame-level structural cells with held-out grouped folds, and freeze one as `cv_pair_model.json`. A new secondary mode `residual-torsion-pc` gives it a runtime force whose OpenMM expression carries the full chain rule. The ladder becomes 2-D (CV2 centres placed on λ-reweighted swarm frames), the sidecar hands the pair to production, and resume refuses a changed model. Envelope, λ scaling and exchange code are untouched.

**Tech Stack:** Python ≥ 3.10, NumPy (core), OpenMM ≥ 8 (runtime; Reference platform in tests), pytest. No new dependencies.

**Spec:** `docs/atlas-md/developer/equilibrium-cv-auto-pair-spec.md` (design; §3 anchor auto-choice is deferred), `docs/atlas-md/developer/equilibrium-cv-implementation-plan.md` §5.2 (residual-model contract), `docs/atlas-md/developer/equilibrium-cv-selection-adversarial-review.md` R4/R5/R8.

## Global Constraints

- **Native-blind.** No native RMSD, folded reference, native contact list or fold-specific GENPEPT preset enters fitting, ranking or gating. `primary_definition["kind"]` must be in `contracts.NATIVE_BLIND_CV_KINDS`, **and** the contact pair list must be the complete combinatorial set for its declared rule (Task 5 stores the rule and the list digest; Task 6 re-derives and compares).
- **Canonical feature order** is `backbone_dihedral_features` (`gareus/tica.py:157`): `[sin φ₀, cos φ₀, …, sin φ_{n−1}, cos φ_{n−1}, sin ψ₀, cos ψ₀, …]`. Every stored vector is bound to a `FeatureSchema` digest **and** Task 6 compares the schema's atom quadruplets element-by-element to the production topology's `secondary_structure_torsions`. Width alone is never identity.
- **Torsion sign convention:** OpenMM `theta` is the negative of `tica._dihedral_rad`; every runtime trig term goes through `_add_weighted_trig_torsion_force` (`sin(-theta)`/`cos(-theta)`).
- **Components are individual SVD directions**, 1-based. `compute_bootstrap_torsion_pca(component=N)` is a *count* and is never called here.
- **Exact weighted independence:** `Σ_i w_i z1_i z2_i ≤ 1e-8` on the training rows under the balanced weights `w`. The *unweighted* `np.cov` is not zero under non-uniform weights (≈1e-5) and must not be used for the certificate.
- **Units of `ss_k`: kJ/mol/CV², already converted.** The one kcal→kJ conversion is `gareus/production.py:6129` (`secondary_cv_ks_kj = [kcal_to_kj(k) …]`); `gareus/windows.py:218 set_window` writes `ss_k` verbatim; every shipped `ss_k` expression is bare `0.5*ss_k*(…)^2`. **The CV2 force expression must contain no 4.184.** Putting one there makes every CV2 restraint 4.184× stiffer than the CSV says while `reconstruct_bias_matrix` uses the CSV — a silent constant-factor error in the PMF.
- **Runtime force = numeric evaluator.** Reference-platform energy vs NumPy evaluator to `1e-6` relative; finite-difference force to `1e-4` relative; the chain-rule term is mandatory (a mutation dropping it must fail the test).
- **Context parameter names unchanged:** `k`, `r0`, `contact_norm` for CV1; `ss_k`, `ss0` for CV2.
- **Envelope and λ ladder untouched.** No edit to `gareus/pep_gamd.py`, `gareus/swarm/envelope.py`, `design_lambda_ladder`, `set_replica_lambda_for_window`.
- **`tica_switch_cv2` forced off** under `cv2: auto|residual-torsion-pc`; `secondary_cv == "auto"` must **never reach a production force builder** (today it would fall through to `secondary_cv_target_angles → "disabled"` and build a `phi0=psi0=0` force silently).
- **Files ≤ 800 lines. NumPy-only imports** in `gareus/cv_selection/*`.
- **Tests run through the local runner**, never directly: `opencode run "In <repo> run: python -m pytest -q <files>"`.
- **Commit format:** `<type>(cv-selection): <description>` plus the session's attribution lines.

**Out of scope (separate plans):** automatic CV1; the post-campaign `Decision` record with primary-panel estimates; `confirmation_campaigns`; `baseline_arm`; the gradient-cosine diagnostic (needs gradients on saved frames). A single run ends with the pair frozen and recorded, not with a confirmed verdict.

---

## File Structure

| File | Responsibility |
|---|---|
| `gareus/swarm/members.py` (modify: `measure_frame`, `run_member_loop`, `_run_loop_recording_failure`, `run_member`) | record canonical torsion features per trace row |
| `gareus/cv_selection/anchor.py` (create) | deployability + dimensionless dynamic-range diagnostic of the configured anchor |
| `gareus/cv_selection/models.py` (create) | fit residual components; `ResidualFit`, `PairModelRuntime` |
| `gareus/cv_selection/independence.py` (create) | held-out nonlinear R² with fold spread, incremental cell information, frame-level partition |
| `gareus/cv_selection/select_pair.py` (create) | deterministic rule, coupling criterion, stability check, report |
| `gareus/cv_selection/contracts.py` (modify) | `PairModel` artifact |
| `gareus/cv.py` (modify) | mode alias, range, transition set, numeric-cache exclusion, evaluator hook |
| `gareus/production.py` (modify) | `_add_residual_torsion_cv_force`; `add_secondary_structure_cv_force(..., primary_cv_def=None)` + call sites `:6205`, `:6270`; resume check at `:1088` |
| `gareus/mbar_analysis/cv2_reprojection.py` (modify) | widen `Cv2Model`, regime branch in `project_cv2`/`reproject_stored_obs` |
| `gareus/windows.py` (modify: `load_explicit_2d_window_csv`) | accept `k2 == 0` rows with empty centre; keep arrays one-per-window |
| `gareus/swarm/ladder_design.py` (modify) | 2-D layout, reweighted CV2 centres, per-gap `k2`, 2-D CSV writer |
| `gareus/swarm/gates.py` (modify) | `pair_gate` |
| `gareus/swarm/analyze.py` (modify) | dataset assembly, selection, 2-D ladder, sidecar |
| `gareus/cli.py`, `gareus/config.py` (modify) | `--cv2 auto|residual-torsion-pc`, model paths, `cv_selection:` block, `_validate_cv_selection_args(p, args)` |
| `gareus/provenance.py` (modify) | record `cv_pair_model_sha256` |
| `tests/conftest.py` (create) | shared synthetic-swarm builders (the repo has no conftest today) |

---

### Task 1: Members record canonical torsion features

**Files:**
- Modify: `gareus/swarm/members.py` — `measure_frame` (`:46-64`), `run_member_loop` (`:68-107`), `_run_loop_recording_failure` (`:158-203`, keyword-only signature, **no `**kwargs` forwarding** — both new kwargs must be threaded explicitly), `run_member` (`:206-338`). Add imports `from gareus.cv import secondary_structure_torsions`, `from gareus.tica import backbone_dihedral_features`, `from gareus.io import write_json` (none are imported today, `:30-36`).
- Test: `tests/test_swarm_members.py`

**Interfaces:**
- Produces per member: `torsion_features.npy` (float64, `(n_trace_rows, 2·n_phi + 2·n_psi)`, row `i` ↔ `trace.csv` row `i`) and `torsion_index.json` `{"phi_torsions": [[i,j,k,l],…], "psi_torsions": […]}`. `run_member_loop(..., feature_fn: Callable[[], np.ndarray] | None = None, features_path: Path | None = None)`; `_run_loop_recording_failure` gains the same two keyword-only parameters and passes them through.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_swarm_members.py  (append)
import numpy as np
from gareus.swarm.members import run_member_loop


def test_member_loop_writes_torsion_features_aligned_with_trace(tmp_path):
    calls = {"n": 0}

    def measure_fn():
        calls["n"] += 1
        return {"cv1": 0.1 * calls["n"], "rg_nm": 1.0, "e2e_nm": 2.0,
                "v_pep_kj": -1.0, "v_dih_kj": 1.0, "potential_kj": -5.0}

    def feature_fn():
        return np.array([np.sin(calls["n"]), np.cos(calls["n"]), 0.0, 1.0])

    trace = tmp_path / "trace.csv"
    feats = tmp_path / "torsion_features.npy"
    run_member_loop(n_equil_steps=0, n_prod_steps=6, steps_per_frame=2, seed_frame_every=100,
                    step_fn=lambda n: None, measure_fn=measure_fn, write_frame_fn=lambda i: None,
                    trace_path=trace, timestep_ps=0.002, feature_fn=feature_fn,
                    features_path=feats)
    rows = trace.read_text().strip().splitlines()[1:]
    stored = np.load(feats)
    assert stored.shape == (len(rows), 4)
    assert np.isclose(stored[1, 0], np.sin(2.0))
    assert not list(tmp_path.glob("*.tmp*"))          # atomic write left no temp file


def test_member_loop_without_feature_fn_writes_no_feature_file(tmp_path):
    trace = tmp_path / "trace.csv"
    run_member_loop(n_equil_steps=0, n_prod_steps=2, steps_per_frame=1, seed_frame_every=100,
                    step_fn=lambda n: None,
                    measure_fn=lambda: {"cv1": 0.0, "rg_nm": 1.0, "e2e_nm": 1.0,
                                        "v_pep_kj": 0.0, "v_dih_kj": 0.0, "potential_kj": 0.0},
                    write_frame_fn=lambda i: None, trace_path=trace, timestep_ps=0.002)
    assert not (tmp_path / "torsion_features.npy").exists()
```

- [ ] **Step 2: Run to verify it fails** — `python -m pytest -q tests/test_swarm_members.py -k torsion_features` → `TypeError: unexpected keyword argument 'feature_fn'`.

- [ ] **Step 3: Implement**

In `run_member_loop`, where each production trace row is appended, also `feature_rows.append(np.asarray(feature_fn(), dtype=np.float64))` when `feature_fn` is set. After the loop:

```python
if feature_fn is not None:
    stacked = np.vstack(feature_rows) if feature_rows else np.zeros((0, 0))
    # np.save appends ".npy" to any name not ending in it, so a ".npy.tmp" temp name
    # becomes ".npy.tmp.npy" and the rename fails. Keep the suffix ".npy".
    tmp = features_path.with_name(features_path.stem + ".tmp.npy")
    with tmp.open("wb") as fh:
        np.save(fh, stacked)
    tmp.replace(features_path)
```

Thread `feature_fn=None, features_path=None` through `_run_loop_recording_failure` to `run_member_loop`. In `run_member`, next to the `ca_indices` setup: `phi_torsions, psi_torsions = secondary_structure_torsions(topology)`; `write_json(member_dir / "torsion_index.json", {"phi_torsions": [list(map(int, t)) for t in phi_torsions], "psi_torsions": [...]})`. Share one `getState(getPositions=True)` per frame between `measure_fn` and `feature_fn` via a closure cell (`last_positions = {}`), and pass `feature_fn=feature_fn, features_path=member_dir / "torsion_features.npy"` to `_run_loop_recording_failure`.

- [ ] **Step 4: Run** — `python -m pytest -q tests/test_swarm_members.py tests/test_swarm_member_concurrency.py` → PASS.

- [ ] **Step 5: Commit** — `git add gareus/swarm/members.py tests/test_swarm_members.py && git commit -m "feat(cv-selection): record canonical torsion features per swarm member"`

---

### Task 2: Residual torsion components — `models.py`

**Files:**
- Create: `gareus/cv_selection/models.py`
- Test: `tests/test_cv_selection_models.py`

**Interfaces (produces):**

```python
@dataclass(frozen=True)
class ResidualFit:
    coefficients: np.ndarray        # (3, d): rows B0, B1, B2 (B2 all-zero for degree 1)
    residual_mean: np.ndarray       # (d,)
    singular_values: np.ndarray     # (k,)
    right_vectors: np.ndarray       # (k, d), row j-1 is v_j, sign-fixed (largest |coef| positive)
    anchor_mean: float
    anchor_std: float
    anchor_clamp: tuple[float, float]   # training a_std quantiles 0.005/0.995
    projection_mean: np.ndarray     # (k,)
    projection_std: np.ndarray      # (k,)
    degree: int

def fit_residual_components(X, anchor, *, degree=1, n_components=6, weights=None) -> ResidualFit
def evaluate_component(fit, j, X, anchor, *, clamp=False) -> np.ndarray    # standardised z2^(j)
def coupling_curvature_kcal(fit, j, k2_kcal) -> float   # k2 (v_j·B1)^2 / (σ_j^2 σ_c^2), plus degree-2 term at a=0
def to_candidate_set(fit, feature_schema, primary_definition, physical_system_sha256,
                     training_rows_sha256, library_versions) -> contracts.CandidateSet
def from_candidate_set(cs) -> ResidualFit
```

`to_candidate_set` must emit the **full** `CandidateSet` envelope — `schema: CANDIDATE_SET_VERSION`, `kind: CANDIDATE_KIND_QUADRATIC_RESIDUAL`, the four digests/versions, `primary_definition`, `components` — see `tests/fixtures/cv_selection/candidate_set.json` for the exact shape. `primary_definition["definition"]` must be a **non-empty dict** (`state_identity._canonical_cv` rejects `{}`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_cv_selection_models.py
import numpy as np
import pytest
from gareus.cv_selection import contracts as C
from gareus.cv_selection.models import (coupling_curvature_kcal, evaluate_component,
                                        fit_residual_components, from_candidate_set,
                                        to_candidate_set)


def _synthetic(n=2000, d=8, degree=2, seed=0, quad_scale=3.0):
    rng = np.random.default_rng(seed)
    a = rng.uniform(0.0, 0.2, n)
    a_std = (a - a.mean()) / a.std()
    B0, B1 = rng.normal(size=d), rng.normal(size=d)
    B2 = rng.normal(size=d) * quad_scale * (degree == 2)
    hidden = rng.normal(size=(n, d)) @ np.diag(np.linspace(3.0, 0.2, d))
    return B0 + np.outer(a_std, B1) + np.outer(a_std ** 2, B2) + hidden, a


def _balanced(n, k=5, seed=0):
    cells = np.random.default_rng(seed).integers(0, k, n)
    counts = np.bincount(cells, minlength=k)
    return 1.0 / (k * counts[cells])


def test_residual_is_exactly_uncorrelated_with_the_anchor_unweighted():
    X, a = _synthetic(degree=1)
    fit = fit_residual_components(X, a, degree=1)
    for j in range(1, 7):
        z2 = evaluate_component(fit, j, X, a)
        assert abs(np.cov(z2, a)[0, 1]) < 1e-10
        assert abs(z2.mean()) < 1e-10 and abs(z2.std() - 1.0) < 1e-10


def test_residual_is_exactly_uncorrelated_under_the_balanced_weights_it_was_fit_with():
    """The production path fits with balanced weights; orthogonality holds in THAT inner product."""
    X, a = _synthetic(degree=1)
    w = _balanced(len(a))
    fit = fit_residual_components(X, a, degree=1, weights=w)
    z2 = evaluate_component(fit, 1, X, a)
    z1 = (a - fit.anchor_mean) / fit.anchor_std
    assert abs(np.sum(w * z1 * z2)) < 1e-8          # weighted covariance: exact
    assert abs(np.cov(z1, z2)[0, 1]) > 1e-7           # unweighted: NOT zero -- do not certify with it


def test_degree_two_removes_quadratic_dependence_that_degree_one_leaves_in_some_component():
    X, a = _synthetic(degree=2, seed=3, quad_scale=6.0)
    a_std = (a - a.mean()) / a.std()
    q = a_std ** 2 - (a_std ** 2).mean()
    lin, quad = fit_residual_components(X, a, degree=1), fit_residual_components(X, a, degree=2)
    corr = lambda fit, j: abs(np.corrcoef(evaluate_component(fit, j, X, a), q)[0, 1])
    assert all(corr(quad, j) < 1e-6 for j in range(1, 7))
    assert max(corr(lin, j) for j in range(1, 7)) > 0.05      # somewhere, not necessarily PC1


def test_components_are_individual_orthonormal_directions_with_positive_dominant_coefficient():
    X, a = _synthetic()
    V = fit_residual_components(X, a).right_vectors
    assert np.allclose(V @ V.T, np.eye(V.shape[0]), atol=1e-10)
    assert all(row[np.argmax(np.abs(row))] > 0 for row in V)


def test_candidate_set_round_trip_is_exact(feature_schema_8):
    X, a = _synthetic()
    fit = fit_residual_components(X, a)
    cs = to_candidate_set(fit, feature_schema_8, _contact_definition(), "b" * 64, "c" * 64,
                          {"numpy": np.__version__})
    assert cs.kind == C.CANDIDATE_KIND_QUADRATIC_RESIDUAL
    back = from_candidate_set(C.CandidateSet.from_mapping(cs.to_mapping()))
    assert np.array_equal(evaluate_component(fit, 2, X, a), evaluate_component(back, 2, X, a))


def test_clamped_evaluation_freezes_the_anchor_outside_the_training_range():
    X, a = _synthetic(degree=2)
    fit = fit_residual_components(X, a, degree=2)
    far = np.full(3, a.max() + 10 * a.std())
    edge = np.full(3, fit.anchor_mean + fit.anchor_clamp[1] * fit.anchor_std)
    assert np.allclose(evaluate_component(fit, 1, X[:3], far, clamp=True),
                       evaluate_component(fit, 1, X[:3], edge, clamp=True))


def test_coupling_curvature_matches_its_definition():
    X, a = _synthetic(degree=1)
    fit = fit_residual_components(X, a, degree=1)
    v = fit.right_vectors[0]
    expected = 50.0 * (v @ fit.coefficients[1]) ** 2 / (fit.projection_std[0] ** 2 * fit.anchor_std ** 2)
    assert np.isclose(coupling_curvature_kcal(fit, 1, 50.0), expected)


def test_constant_anchor_and_rank_deficient_design_are_refused():
    X, a = _synthetic()
    with pytest.raises(ValueError, match="anchor"):
        fit_residual_components(X, np.full(len(X), 0.05))
    X2, a2 = _synthetic(n=2)
    with pytest.raises(ValueError, match="rank"):
        fit_residual_components(X2, a2, degree=2)


def _contact_definition():
    return {"kind": "nonlocal-contact-fraction", "units": "dimensionless",
            "definition": {"r0_angstrom": 12.0, "beta_per_angstrom": 3.0,
                           "min_sequence_separation": 4, "atom_selection": "heavy",
                           "pair_rule": "all-pairs-min-sep", "normalize": True}}


@pytest.fixture
def feature_schema_8():
    rows = []
    for t in range(4):
        name = "phi" if t < 2 else "psi"
        for trig in ("sin", "cos"):
            rows.append({"index": len(rows), "name": f"{name}-{t + 1}-{trig}",
                         "torsion_name": f"{name}-{t + 1}", "residue_index": t + 1,
                         "atom_indices": [4 * t, 4 * t + 1, 4 * t + 2, 4 * t + 3],
                         "trig": trig, "dihedral_sign_convention": "negated"})
    return C.FeatureSchema.from_mapping({"schema": C.FEATURE_SCHEMA_VERSION,
                                         "topology_sha256": "a" * 64, "features": rows})
```

- [ ] **Step 2: Run to verify they fail** — `ModuleNotFoundError`.

- [ ] **Step 3: Implement `models.py`**

```python
"""Residual torsion components: phi(x) minus its polynomial dependence on the anchor.

Sum_i w_i z2_i a_i == 0 exactly because the residual of a weighted OLS fit with an
intercept is w-orthogonal to every regressor column, and `a` is a regressor column.
The identity holds in the WEIGHTED inner product only.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from . import contracts as C


@dataclass(frozen=True)
class ResidualFit:
    coefficients: np.ndarray
    residual_mean: np.ndarray
    singular_values: np.ndarray
    right_vectors: np.ndarray
    anchor_mean: float
    anchor_std: float
    anchor_clamp: tuple
    projection_mean: np.ndarray
    projection_std: np.ndarray
    degree: int


def _design(a_std, degree):
    cols = [np.ones_like(a_std), a_std] + ([a_std ** 2] if degree == 2 else [])
    return np.column_stack(cols)


def fit_residual_components(X, anchor, *, degree=1, n_components=C.MAX_COMPONENT_INDEX, weights=None):
    X = np.asarray(X, dtype=np.float64); a = np.asarray(anchor, dtype=np.float64)
    if degree not in (1, 2):
        raise ValueError("degree must be 1 or 2")
    if X.ndim != 2 or a.shape != (X.shape[0],):
        raise ValueError("X must be (n, d) and anchor (n,)")
    if not (np.isfinite(X).all() and np.isfinite(a).all()):
        raise ValueError("non-finite input")
    w = np.ones(len(a)) if weights is None else np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    mu_c = float(np.sum(w * a)); sd_c = float(np.sqrt(np.sum(w * (a - mu_c) ** 2)))
    if sd_c <= 1e-12:
        raise ValueError("anchor has zero variance; a constant anchor defines no coordinate")
    a_std = (a - mu_c) / sd_c
    D = _design(a_std, degree); sw = np.sqrt(w)[:, None]
    B, _, rank, _ = np.linalg.lstsq(D * sw, X * sw, rcond=1e-12)
    if rank < D.shape[1]:
        raise ValueError(f"design matrix rank {rank} < {D.shape[1]}; residualisation is undefined")
    if degree == 1:
        B = np.vstack([B, np.zeros((1, X.shape[1]))])
    R = X - _design(a_std, 2) @ B
    mean_R = np.sum(w[:, None] * R, axis=0)
    _, s, Vt = np.linalg.svd((R - mean_R) * sw, full_matrices=False)
    k = min(int(n_components), Vt.shape[0]); V = Vt[:k].copy()
    for row in V:
        row *= np.sign(row[np.argmax(np.abs(row))]) or 1.0
    scores = (R - mean_R) @ V.T
    mu = np.sum(w[:, None] * scores, axis=0)
    sd = np.sqrt(np.sum(w[:, None] * (scores - mu) ** 2, axis=0))
    if np.any(sd <= 1e-12):
        raise ValueError("a component has zero variance on the training data")
    clamp = (float(np.quantile(a_std, 0.005)), float(np.quantile(a_std, 0.995)))
    return ResidualFit(B, mean_R, s[:k], V, mu_c, sd_c, clamp, mu, sd, degree)


def evaluate_component(fit, j, X, anchor, *, clamp=False):
    if not 1 <= j <= fit.right_vectors.shape[0]:
        raise ValueError(f"component {j} out of range")
    X = np.asarray(X, dtype=np.float64); a = np.asarray(anchor, dtype=np.float64)
    a_std = (a - fit.anchor_mean) / fit.anchor_std
    if clamp:
        a_std = np.clip(a_std, *fit.anchor_clamp)
    R = X - _design(a_std, 2) @ fit.coefficients
    s = (R - fit.residual_mean) @ fit.right_vectors[j - 1]
    return (s - fit.projection_mean[j - 1]) / fit.projection_std[j - 1]


def coupling_curvature_kcal(fit, j, k2_kcal):
    """Curvature the CV2 umbrella induces along the anchor, d²U₂/dc² at a = 0."""
    v = fit.right_vectors[j - 1]
    slope = float(v @ fit.coefficients[1])
    return float(k2_kcal) * slope ** 2 / (float(fit.projection_std[j - 1]) ** 2 * fit.anchor_std ** 2)
```

`to_candidate_set`/`from_candidate_set` as described in the Interfaces block; `eigenvalue_tie_flagged = abs(s_j − s_{j±1}) < 1e-6·s_1`; `from_candidate_set` sets `degree = 2 if np.any(B[2] != 0) else 1` and recomputes `anchor_clamp` from a stored `anchor_clamp` pair added to the component mapping (one extra field on every component; update `contracts._COMPONENT_FIELDS` and the fixture generator accordingly).

- [ ] **Step 4: Run** — `python -m pytest -q tests/test_cv_selection_models.py tests/test_cv_selection_contracts.py` → PASS.

- [ ] **Step 5: Commit** — `git add gareus/cv_selection/models.py gareus/cv_selection/contracts.py tests/ && git commit -m "feat(cv-selection): fit individual residual torsion components"`

---

### Task 3: Anchor deployability diagnostic — `anchor.py`

**Files:**
- Create: `gareus/cv_selection/anchor.py`
- Test: `tests/test_cv_selection_anchor.py`

**Interfaces (produces):**

```python
DEPLOYABLE_ANCHOR_KINDS = frozenset({"nonlocal-contact-fraction"})   # kinds with a landed CV1 force

@dataclass(frozen=True)
class AnchorCandidate:  kind: str; definition: dict; values: np.ndarray
@dataclass(frozen=True)
class AnchorScore:
    kind: str; definition: dict; n_resolvable: int; coverage_range: float
    dynamic_range: float          # range_q / sigma_q : dimensionless, unit-invariant
    forceable: bool; deployable: bool; reasons: tuple[str, ...]

def score_anchor(cand, *, temperature_k, k_max_kcal, min_windows, min_dynamic_range=6.0) -> AnchorScore
def rank_anchors(candidates, **kw) -> list[AnchorScore]    # diagnostic ordering by dynamic_range
```

`n_resolvable` uses the shipped `n_resolvable_windows` and is **unit-dependent** (Rg in Å resolves 10× more windows than Rg in nm). It is kept only because the shipped ladder uses it for the *configured* anchor; `dynamic_range = range/σ_q` is the unit-invariant figure and is what `rank_anchors` sorts by. A kind outside `DEPLOYABLE_ANCHOR_KINDS` is scored, reported as `forceable=False`, and can never be deployable.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_cv_selection_anchor.py
import numpy as np
import pytest
from gareus.cv_selection.anchor import AnchorCandidate, rank_anchors, score_anchor


def _cand(kind, values, **definition):
    definition = definition or {"placeholder": True}
    return AnchorCandidate(kind, definition, np.asarray(values, dtype=float))


def test_r7_like_contact_range_is_not_deployable_and_says_why():
    rng = np.random.default_rng(0)
    s = score_anchor(_cand("nonlocal-contact-fraction", rng.uniform(0.0, 0.069, 5000), r0_angstrom=12.0),
                     temperature_k=300.0, k_max_kcal=1200.0, min_windows=4)
    assert s.n_resolvable == 2 and s.deployable is False and s.forceable is True
    assert any("resolvable" in r for r in s.reasons)


def test_dynamic_range_is_invariant_to_units_while_n_resolvable_is_not():
    rng = np.random.default_rng(1)
    rg_nm = rng.uniform(0.55, 1.10, 5000)
    nm = score_anchor(_cand("radius-of-gyration", rg_nm), temperature_k=300.0, k_max_kcal=1200.0, min_windows=4)
    ang = score_anchor(_cand("radius-of-gyration", rg_nm * 10.0), temperature_k=300.0, k_max_kcal=1200.0, min_windows=4)
    assert np.isclose(nm.dynamic_range, ang.dynamic_range)
    assert ang.n_resolvable > 5 * nm.n_resolvable


def test_an_unforceable_kind_is_scored_but_never_deployable():
    rng = np.random.default_rng(2)
    s = score_anchor(_cand("radius-of-gyration", rng.uniform(0.55, 1.10, 5000)),
                     temperature_k=300.0, k_max_kcal=1200.0, min_windows=4)
    assert s.forceable is False and s.deployable is False
    assert any("no runtime force" in r for r in s.reasons)


def test_native_derived_kind_is_refused_before_scoring():
    with pytest.raises(ValueError, match="native-blind"):
        score_anchor(_cand("rmsd-to-native-pdb", np.linspace(0, 1, 10)),
                     temperature_k=300.0, k_max_kcal=1200.0, min_windows=1)
```

- [ ] **Step 2: Run to verify they fail** — module not found.

- [ ] **Step 3: Implement**

```python
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from . import contracts as C
from ..swarm.ladder_design import n_resolvable_windows

DEPLOYABLE_ANCHOR_KINDS = frozenset({"nonlocal-contact-fraction"})


@dataclass(frozen=True)
class AnchorCandidate:
    kind: str; definition: dict; values: np.ndarray


@dataclass(frozen=True)
class AnchorScore:
    kind: str; definition: dict; n_resolvable: int; coverage_range: float
    dynamic_range: float; forceable: bool; deployable: bool; reasons: tuple


def score_anchor(cand, *, temperature_k, k_max_kcal, min_windows, min_dynamic_range=6.0,
                 overlap_sigma=1.5):
    if cand.kind not in C.NATIVE_BLIND_CV_KINDS:
        raise ValueError(f"{cand.kind!r} is not in the native-blind dictionary")
    v = np.asarray(cand.values, dtype=np.float64); v = v[np.isfinite(v)]
    reasons = []
    forceable = cand.kind in DEPLOYABLE_ANCHOR_KINDS
    if not forceable:
        reasons.append(f"{cand.kind}: no runtime force for this anchor kind yet")
    if v.size < 2:
        return AnchorScore(cand.kind, dict(cand.definition), 0, 0.0, 0.0, forceable, False,
                           tuple(reasons + ["fewer than two finite values"]))
    rng_ = float(v.max() - v.min()); sd = float(v.std())
    dyn = rng_ / sd if sd > 0 else 0.0
    n_res = n_resolvable_windows(rng_, temperature_k, k_max_kcal=k_max_kcal, overlap_sigma=overlap_sigma)
    if n_res < min_windows:
        reasons.append(f"{n_res} resolvable windows < required {min_windows} at k_max={k_max_kcal}")
    if dyn < min_dynamic_range:
        reasons.append(f"dynamic range {dyn:.2f} sigma < {min_dynamic_range}")
    return AnchorScore(cand.kind, dict(cand.definition), n_res, rng_, dyn, forceable,
                       forceable and not reasons, tuple(reasons))


def rank_anchors(candidates, **kw):
    scored = [(-(s.dynamic_range), i, s) for i, s in
              enumerate(score_anchor(c, **kw) for c in candidates)]
    return [s for _, _, s in sorted(scored, key=lambda t: t[:2])]
```

- [ ] **Step 4: Run** → PASS. **Step 5: Commit** — `git commit -m "feat(cv-selection): anchor deployability and dynamic-range diagnostic"`

---

### Task 4: Independence diagnostics — `independence.py`

**Files:**
- Create: `gareus/cv_selection/independence.py`
- Test: `tests/test_cv_selection_independence.py`

**Interfaces (produces):**

```python
def grouped_folds(groups, n_folds, *, seed=20260920) -> list[np.ndarray]   # groups shuffled once, then strided
def heldout_nonlinear_r2(z_target, z_predictor, groups, *, n_bins=10, n_folds=4) -> tuple[float, np.ndarray]
    # (pooled R², per-fold R²)
def frame_partition(features, *, n_cells, seed=20260920) -> tuple[np.ndarray, np.ndarray]
    # farthest-point centres on standardised per-frame features, then nearest-centre labels; (labels, centres)
def incremental_cell_information(cells, z1, z2, groups, *, n_bins=10, n_folds=4, alpha=1.0) -> dict
    # {"l1", "l2", "l12", "gain": l1 - l12}
```

Design decisions fixed by the review:
- `gain = L1 − L12`, i.e. `I(cell; z2 | z1)`. **Not** `min(L1, L2) − L12`: the moment a candidate beats `z1` on its own, the `min` switches to `I(cell; z1 | z2) ≈ 0` and a perfect CV2 scores the same as noise (measured: −0.003 nats for both).
- The joint model uses `⌈√n_bins⌉` bins per axis so its capacity (≈ n_bins cells) matches the marginals'; otherwise the 100-cell joint table is smoothed harder and `L12` is biased upward.
- `grouped_folds` shuffles the unique groups under a fixed seed before striding: `plan.csv` emits seeds in cell order, so an unshuffled stride gives each fold a different stratum mix.
- Cells are **frame-level** (`frame_partition`), never the seed-level `plan.csv` strata (constant within a member ⇒ effective n = number of seeds; and CV1 is a stratification axis ⇒ leakage).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_cv_selection_independence.py
import numpy as np
from gareus.cv_selection.independence import (frame_partition, grouped_folds,
                                              heldout_nonlinear_r2, incremental_cell_information)


def _groups(n, k=60, seed=0):
    return np.random.default_rng(seed).integers(0, k, n)


def test_independent_coordinates_have_near_zero_heldout_r2():
    rng = np.random.default_rng(0)
    z1, z2 = rng.normal(size=20000), rng.normal(size=20000)
    pooled, folds = heldout_nonlinear_r2(z2, z1, _groups(20000))
    assert pooled < 0.02 and folds.shape == (4,)


def test_a_quadratic_dependence_survives_linear_residualisation_and_is_detected():
    rng = np.random.default_rng(1)
    z1 = rng.normal(size=20000)
    z2 = z1 ** 2 - 1.0 + 0.3 * rng.normal(size=20000)
    assert abs(np.cov(z1, z2)[0, 1]) < 0.05
    assert heldout_nonlinear_r2(z2, z1, _groups(20000))[0] > 0.6


def test_folds_never_split_a_group_and_are_shuffled():
    groups = np.repeat(np.arange(40), 25)                 # sorted, like plan.csv order
    folds = grouped_folds(groups, 4)
    for hold in folds:
        assert set(groups[hold]).isdisjoint(set(groups[np.setdiff1d(np.arange(1000), hold)]))
    assert sorted(np.unique(groups[folds[0]])) != list(range(0, 40, 4))   # not a plain stride


def test_a_perfect_new_coordinate_scores_high_not_zero():
    """The v0.1 rule min(L1, L2) - L12 returned -0.003 here. L1 - L12 must return the truth."""
    rng = np.random.default_rng(2)
    z1 = rng.normal(size=20000); z2 = rng.normal(size=20000)
    cells = (z2 > 0).astype(int)
    out = incremental_cell_information(cells, z1, z2, _groups(20000))
    assert out["gain"] > 0.5 and out["l2"] < out["l1"]


def test_a_copy_of_z1_adds_no_information():
    rng = np.random.default_rng(3)
    z1 = rng.normal(size=20000)
    cells = (z1 > 0).astype(int)
    assert abs(incremental_cell_information(cells, z1, 2.0 * z1, _groups(20000))["gain"]) < 0.05


def test_frame_partition_is_deterministic_and_covers_every_frame():
    feats = np.random.default_rng(4).normal(size=(5000, 12))
    labels_a, centres = frame_partition(feats, n_cells=8)
    labels_b, _ = frame_partition(feats, n_cells=8)
    assert np.array_equal(labels_a, labels_b) and centres.shape == (8, 12)
    assert set(np.unique(labels_a)) == set(range(8))
```

- [ ] **Step 2: Run to verify they fail** — module not found.

- [ ] **Step 3: Implement**

```python
from __future__ import annotations
import math
import numpy as np


def grouped_folds(groups, n_folds, *, seed=20260920):
    uniq = np.unique(np.asarray(groups))
    uniq = uniq[np.random.default_rng(seed).permutation(len(uniq))]
    return [np.flatnonzero(np.isin(groups, uniq[i::n_folds])) for i in range(n_folds)]


def _edges(x, n_bins):
    return np.quantile(x, np.linspace(0.0, 1.0, n_bins + 1)[1:-1])


def heldout_nonlinear_r2(z_target, z_predictor, groups, *, n_bins=10, n_folds=4):
    y = np.asarray(z_target, float); x = np.asarray(z_predictor, float)
    per_fold = []; sse = sst = 0.0
    for hold in grouped_folds(groups, n_folds):
        train = np.setdiff1d(np.arange(len(y)), hold)
        edges = _edges(x[train], n_bins)
        b_tr, b_ho = np.digitize(x[train], edges), np.digitize(x[hold], edges)
        means = np.array([y[train][b_tr == b].mean() if np.any(b_tr == b) else y[train].mean()
                          for b in range(n_bins)])
        f_sse = float(np.sum((y[hold] - means[b_ho]) ** 2))
        f_sst = float(np.sum((y[hold] - y[train].mean()) ** 2))
        per_fold.append(1.0 - f_sse / f_sst if f_sst > 0 else 0.0)
        sse += f_sse; sst += f_sst
    return (1.0 - sse / sst if sst > 0 else 0.0), np.asarray(per_fold)


def frame_partition(features, *, n_cells, seed=20260920):
    F = np.asarray(features, float)
    F = (F - F.mean(axis=0)) / np.where(F.std(axis=0) > 0, F.std(axis=0), 1.0)
    centres = [F[0]]                          # first centre: first row (immutable row order)
    d = np.linalg.norm(F - centres[0], axis=1)
    while len(centres) < n_cells:
        centres.append(F[int(np.argmax(d))])
        d = np.minimum(d, np.linalg.norm(F - centres[-1], axis=1))
    centres = np.asarray(centres)
    labels = np.argmin(((F[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2), axis=1)
    return labels, centres


def _heldout_cross_entropy(cells, coords, groups, n_bins, n_folds, alpha):
    cells = np.asarray(cells); coords = np.asarray(coords, float)
    coords = coords[:, None] if coords.ndim == 1 else coords
    k_axes = coords.shape[1]
    bins_per_axis = n_bins if k_axes == 1 else max(2, math.ceil(n_bins ** (1.0 / k_axes)))
    n_cells = int(cells.max()) + 1; total = 0.0; count = 0
    for hold in grouped_folds(groups, n_folds):
        train = np.setdiff1d(np.arange(len(cells)), hold)
        edges = [_edges(coords[train, k], bins_per_axis) for k in range(k_axes)]
        def key(rows):
            idx = np.zeros(len(rows), dtype=np.int64)
            for k, e in enumerate(edges):
                idx = idx * bins_per_axis + np.digitize(coords[rows, k], e)
            return idx
        k_tr, k_ho = key(train), key(hold)
        table = {}
        for kk, c in zip(k_tr, cells[train]):
            table.setdefault(kk, np.zeros(n_cells))[c] += 1
        prior = np.bincount(cells[train], minlength=n_cells) + alpha
        for kk, c in zip(k_ho, cells[hold]):
            counts = table.get(kk, np.zeros(n_cells)) + alpha * prior / prior.sum()
            total += -np.log(counts[c] / counts.sum()); count += 1
    return total / max(count, 1)


def incremental_cell_information(cells, z1, z2, groups, *, n_bins=10, n_folds=4, alpha=1.0):
    l1 = _heldout_cross_entropy(cells, z1, groups, n_bins, n_folds, alpha)
    l2 = _heldout_cross_entropy(cells, z2, groups, n_bins, n_folds, alpha)
    l12 = _heldout_cross_entropy(cells, np.column_stack([z1, z2]), groups, n_bins, n_folds, alpha)
    return {"l1": float(l1), "l2": float(l2), "l12": float(l12), "gain": float(l1 - l12)}
```

- [ ] **Step 4: Run** → PASS. **Step 5: Commit** — `git commit -m "feat(cv-selection): held-out independence and information diagnostics"`

---

### Task 5: Selection rule, coupling criterion, stability — `select_pair.py` + `PairModel`

**Files:**
- Create: `gareus/cv_selection/select_pair.py`
- Modify: `gareus/cv_selection/contracts.py` — `PairModel` (`PAIR_MODEL_VERSION = "atlas-cv-selection-pair-model-v1"`): `candidate_set_sha256`, `feature_schema_sha256`, `anchor` (canonical CV definition incl. `pair_rule`, `pair_list_sha256`, `norm`), `selected_component_index`, `degree`, `certificate` (`cov_q_weighted`, `r2_z2_given_z1_mean`, `r2_z2_given_z1_se`, `r2_z1_given_z2_mean`, `coupling_curvature_kcal`, `coupling_fraction_of_k1`, `std_unweighted_z2`, `design_measure`, `n_frames`, `n_seed_families`, `half_split_agrees`), `runner_ups` (`{component_index, reason, scores}`), `genpept_preset`, `sha256`. `_parse` requires `anchor["kind"] in NATIVE_BLIND_CV_KINDS`, `abs(cov_q_weighted) ≤ 1e-8`, `degree ∈ {1,2}`, `coupling_fraction_of_k1 ≤ 1.0`.
- Test: `tests/test_cv_selection_select_pair.py`

**Interfaces (produces):**

```python
@dataclass(frozen=True)
class SwarmDataset:
    features: np.ndarray; feature_schema: C.FeatureSchema
    anchor: AnchorCandidate                # the configured contact CV, values aligned to rows
    shape_features: np.ndarray             # (n, 2): rg_nm, e2e_nm — for the frame partition only
    groups: np.ndarray                     # seed_id per row
    weights: np.ndarray                    # balanced over frame cells (see below)

@dataclass(frozen=True)
class SelectionConfig:
    residual_degree: int = 1
    components: tuple = (1, 2, 3, 4, 5, 6)
    max_nonlinear_r2: float = 0.20          # gate on mean + 2*SE over folds
    n_cells_coarse: int = 8; n_cells_fine: int = 24
    k1_kcal_reference: float = 1000.0        # contact_adaptive_max_k_kcal default; passed from args
    k2_kcal_reference: float = 50.0          # the k2 at which coupling is evaluated
    max_coupling_fraction: float = 0.25      # coupling curvature / k1 reference
    min_windows_cv1: int = 4
    temperature_k: float = 300.0

@dataclass(frozen=True)
class PairSelection:
    status: str            # "pair" | "cv1_only" | "no_deployable_anchor"
    pair_model: C.PairModel | None; candidate_set: C.CandidateSet | None
    anchor: AnchorScore; report: dict

def balanced_weights(labels) -> np.ndarray
def select_cv_pair(data, config, *, physical_system_sha256, training_rows_sha256,
                   library_versions, genpept_preset) -> PairSelection
```

Rule:
1. `anchor = score_anchor(data.anchor, …)`; not deployable → `no_deployable_anchor`.
2. Frame partition: `cells_coarse, _ = frame_partition(np.column_stack([features, shape_features]), n_cells=8)` — **the anchor CV is excluded** from the partition features; `cells_fine` likewise with 24. `weights = balanced_weights(cells_coarse)`.
3. `fit = fit_residual_components(features, anchor.values, degree, weights=weights)`; `z1 = (a − μ_c)/σ_c`.
4. Per component `j`: `z2`; `(r2, folds) = heldout_nonlinear_r2(z2, z1, groups)`; `r2_gate = folds.mean() + 2·folds.std(ddof=1)/√4`; `gain = incremental_cell_information(cells_fine, z1, z2, groups)["gain"]`; `coupling = coupling_curvature_kcal(fit, j, k2_kcal_reference)`; `frac = coupling / k1_kcal_reference`. Deployable iff `r2_gate ≤ max_nonlinear_r2` and `frac ≤ max_coupling_fraction`. Everything recorded.
5. Winner: deployable component with max `gain`, **provided `max gain ≥ min_gain_nats` (config, default 0.02 — an order of magnitude above the ~0.003-nat smoothing floor measured on synthetic data)**. Below that floor every candidate is indistinguishable from noise and the lowest-index tie-break would install PC1 while the report presented a content-free choice as principled; instead return `cv1_only` with reason `"no component adds information about the discovery partition (max gain %.4f < %.3f)"`. Ties above the floor → lowest `j`. No deployable component → `cv1_only`.
6. Stability: rerun steps 3–5 on two disjoint halves of the seed families (shuffled with the fold seed); `half_split_agrees = (winner_A == winner_B == winner)`. Recorded, not gating.
7. Certificate as listed; `PairModel.from_mapping`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_cv_selection_select_pair.py
import numpy as np
import pytest
from gareus.cv_selection import contracts as C
from gareus.cv_selection.anchor import AnchorCandidate
from gareus.cv_selection.select_pair import (SelectionConfig, SwarmDataset, balanced_weights,
                                             select_cv_pair)

ARGS = dict(physical_system_sha256="b" * 64, training_rows_sha256="c" * 64,
            library_versions={"numpy": np.__version__}, genpept_preset="broad")


def _contact_anchor(values):
    return AnchorCandidate("nonlocal-contact-fraction",
                           {"r0_angstrom": 12.0, "beta_per_angstrom": 3.0, "min_sequence_separation": 4,
                            "atom_selection": "heavy", "pair_rule": "all-pairs-min-sep",
                            "normalize": True, "pair_list_sha256": "d" * 64, "norm": 45.0},
                           values)


def _dataset(n=6000, seed=0, wide=True, quadratic_mode=False):
    rng = np.random.default_rng(seed)
    a = rng.uniform(0.0, 0.5 if wide else 0.069, n)
    a_std = (a - a.mean()) / a.std()
    d = 8
    hidden = rng.normal(size=(n, d)) @ np.diag(np.linspace(3.0, 0.3, d))
    X = np.outer(a_std, rng.normal(size=d)) + hidden
    if quadratic_mode:
        X[:, 0] += 3.0 * (a_std ** 2 - 1.0)
    shape = np.column_stack([0.6 + 0.3 * a + 0.05 * rng.normal(size=n),
                             1.0 + hidden[:, 1] * 0.1])
    groups = rng.integers(0, 60, n)
    return SwarmDataset(X, _schema(d), _contact_anchor(a), shape, groups,
                        balanced_weights(rng.integers(0, 8, n)))


def _schema(d): ...   # identical to Task 2's feature_schema_8 builder, parametrised by d


def test_selects_a_residual_component_with_an_exact_weighted_certificate():
    sel = select_cv_pair(_dataset(), SelectionConfig(), **ARGS)
    assert sel.status == "pair"
    cert = sel.pair_model.certificate
    assert abs(cert["cov_q_weighted"]) < 1e-8
    assert cert["coupling_fraction_of_k1"] <= 0.25
    assert isinstance(cert["half_split_agrees"], bool)


def test_a_quadratic_hidden_mode_is_set_aside_with_the_fold_spread_recorded():
    sel = select_cv_pair(_dataset(quadratic_mode=True), SelectionConfig(), **ARGS)
    skipped = {r["component_index"]: r for r in sel.pair_model.runner_ups}
    assert any("nonlinear" in r["reason"] for r in skipped.values())
    assert all("r2_se" in r["scores"] for r in skipped.values())


def test_a_component_that_couples_too_hard_into_cv1_is_set_aside():
    sel = select_cv_pair(_dataset(), SelectionConfig(max_coupling_fraction=1e-9), **ARGS)
    assert sel.status == "cv1_only"
    assert all("coupling" in r["reason"] for r in sel.report["runner_ups"])


def test_narrow_anchor_is_no_deployable_anchor():
    sel = select_cv_pair(_dataset(wide=False), SelectionConfig(), **ARGS)
    assert sel.status == "no_deployable_anchor" and sel.pair_model is None


def test_selection_is_deterministic():
    a = select_cv_pair(_dataset(), SelectionConfig(), **ARGS)
    b = select_cv_pair(_dataset(), SelectionConfig(), **ARGS)
    assert a.pair_model.sha256 == b.pair_model.sha256


def test_balanced_weights_give_every_cell_equal_mass():
    cells = np.array([0] * 90 + [1] * 10)
    w = balanced_weights(cells)
    assert abs(w[cells == 0].sum() - w[cells == 1].sum()) < 1e-12 and abs(w.sum() - 1.0) < 1e-12


def test_a_fold_specific_genpept_preset_is_refused():
    with pytest.raises(ValueError, match="native-blind"):
        select_cv_pair(_dataset(), SelectionConfig(), **{**ARGS, "genpept_preset": "chignolin"})
```

- [ ] **Step 2–5:** fail → implement per the rule above (`_units("nonlocal-contact-fraction") = "dimensionless"`; `genpept_preset not in {"broad"}` → `ValueError`) → run `tests/test_cv_selection_select_pair.py tests/test_cv_selection_contracts.py` → PASS → `git commit -m "feat(cv-selection): deterministic pair selection with coupling and stability checks"`.

---

### Task 6: Runtime force and numeric evaluator for `residual-torsion-pc`

**Files:**
- Modify: `gareus/cv.py` — `secondary_cv_mode` aliases (`:269`), `secondary_cv_is_transition` (`:312`), `secondary_cv_range` (`:317`), `_ensure_secondary_cv_numeric_cache` exclusion set (`:488`), new branch in `secondary_structure_score_from_positions_nm` **after** the cache call at `:507`, next to the `tica-linear/torsion-pca` branch at `:509`.
- Modify: `gareus/production.py` — `add_secondary_structure_cv_force(openmm, system, topology, args, force_group=29, *, primary_cv_def=None)` (`:1434`) and **both call sites `:6205` and `:6270`** pass `primary_cv_def=`; add `contact_normalization_denominator` to the `from .cv import (…)` block (`:59-82`); new `_add_residual_torsion_cv_force` next to `_add_linear_torsion_cv_force`.
- Modify: `gareus/cv_selection/models.py` — `PairModelRuntime`.
- Modify: `gareus/mbar_analysis/cv2_reprojection.py` — `Cv2Model.result: TICAResult | PairModelRuntime`, `load_cv2_model` branch on `schema == PAIR_MODEL_VERSION`, `project_cv2` regime branch using `obs["primary_cv"]` (the key `load_stored_obs` actually returns; there is no `cv1` key).
- Test: `tests/test_cv_selection_forces.py`

**Interfaces (produces):**

```python
@dataclass(frozen=True)
class PairModelRuntime:
    fit: ResidualFit; j: int; anchor_kind: str; anchor_definition: dict; pair_sha256: str
    feature_atoms: tuple[tuple[int,int,int,int], ...]     # phi then psi quadruplets from the FeatureSchema
    @classmethod
    def load(cls, pair_model_path, candidate_set_path, feature_schema_path) -> "PairModelRuntime"
    def check_topology(self, phi_torsions, psi_torsions) -> None      # element-wise equality or RuntimeError
    def check_anchor(self, args, contact_pairs) -> None               # r0, beta, min_sep, selection, normalize, norm, pair digest

def residual_cv2_from_positions_nm(positions_nm, runtime, phi_torsions, psi_torsions, contact_pairs, args) -> float
def _add_residual_torsion_cv_force(openmm, system, phi_torsions, psi_torsions, contact_pairs,
                                   runtime, args, *, force_group) -> dict
```

The secondary-CV info dict stores the runtime object and `args` under **underscore-prefixed keys** (`_runtime`, `_args`) — `production.py:6209` merges this dict into `secondary_cv_metadata`, `:4743` writes it through `_json_ready`, which strips only `_`-prefixed keys and would otherwise `TypeError` on a dataclass/Namespace.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_cv_selection_forces.py
import numpy as np
import pytest
openmm = pytest.importorskip("openmm")
from openmm import unit

from gareus.cv import residual_cv2_from_positions_nm
from gareus.cv_selection.models import PairModelRuntime, ResidualFit
from gareus.production import _add_residual_torsion_cv_force

KJ_PER_KCAL = 4.184


def _peptide_like(n_atoms=16, seed=0):
    rng = np.random.default_rng(seed)
    pos = np.cumsum(rng.normal(scale=0.12, size=(n_atoms, 3)), axis=0) + 2.0
    phi = [(0, 1, 2, 3), (4, 5, 6, 7)]; psi = [(8, 9, 10, 11), (12, 13, 14, 15)]
    contacts = [(0, 15, 1.0), (1, 14, 1.0), (2, 13, 1.0), (3, 12, 1.0)]
    return pos, phi, psi, contacts


class _Args:
    contact_r0_a = 12.0; contact_beta_a_inv = 3.0; contact_normalize = True
    contact_min_sequence_separation = 4; contact_atom_selection = "heavy"; contact_scheme = "atom-pairs"


def _runtime(phi, psi, contacts, d=8, degree=2, seed=1):
    rng = np.random.default_rng(seed)
    B = rng.normal(size=(3, d)) * 0.3
    if degree == 1:
        B[2] = 0.0
    V = np.linalg.qr(rng.normal(size=(d, d)))[0].T
    fit = ResidualFit(B, rng.normal(size=d) * 0.1, np.linspace(2, 0.5, d), V, 0.05, 0.02,
                      (-3.0, 3.0), np.zeros(d), np.ones(d), degree)
    from gareus.cv import contact_normalization_denominator
    definition = {"r0_angstrom": 12.0, "beta_per_angstrom": 3.0, "min_sequence_separation": 4,
                  "atom_selection": "heavy", "pair_rule": "explicit-test-pairs", "normalize": True,
                  "pair_list_sha256": "d" * 64, "norm": contact_normalization_denominator(contacts, _Args())}
    return PairModelRuntime(fit, 2, "nonlocal-contact-fraction", definition, "f" * 64, tuple(phi + psi))


def _energy_forces(pos, phi, psi, contacts, runtime, ss_k_kj, c2):
    system = openmm.System()
    for _ in range(len(pos)):
        system.addParticle(12.0)
    meta = _add_residual_torsion_cv_force(openmm, system, phi, psi, contacts, runtime, _Args(), force_group=29)
    ctx = openmm.Context(system, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName("Reference"))
    ctx.setPositions(pos * unit.nanometer)
    ctx.setParameter("ss_k", ss_k_kj); ctx.setParameter("ss0", c2)
    st = ctx.getState(getEnergy=True, getForces=True, groups={29})
    return (st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
            np.asarray(st.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)), meta)


@pytest.mark.parametrize("degree", [1, 2])
def test_umbrella_energy_matches_the_numpy_evaluator_with_ss_k_in_kJ(degree):
    """ss_k reaches the Context already in kJ/mol/CV^2 (production.py:6129 converts; set_window does not).
    The force expression must therefore contain NO 4.184."""
    pos, phi, psi, contacts = _peptide_like()
    rt = _runtime(phi, psi, contacts, degree=degree)
    k2_kcal, c2 = 40.0, 0.3
    e, _, _ = _energy_forces(pos, phi, psi, contacts, rt, k2_kcal * KJ_PER_KCAL, c2)
    z2 = residual_cv2_from_positions_nm(pos, rt, phi, psi, contacts, _Args())
    expected = 0.5 * k2_kcal * KJ_PER_KCAL * (z2 - c2) ** 2
    assert abs(e - expected) <= 1e-6 * max(1.0, abs(expected))


def test_forces_match_finite_differences_including_the_chain_rule_term():
    pos, phi, psi, contacts = _peptide_like(seed=5)
    rt = _runtime(phi, psi, contacts, degree=2)
    _, f, _ = _energy_forces(pos, phi, psi, contacts, rt, 40.0 * KJ_PER_KCAL, 0.3)
    h = 1e-6
    for atom in (0, 3, 9, 15):
        for ax in range(3):
            p = pos.copy(); p[atom, ax] += h
            m = pos.copy(); m[atom, ax] -= h
            ep = _energy_forces(p, phi, psi, contacts, rt, 40.0 * KJ_PER_KCAL, 0.3)[0]
            em = _energy_forces(m, phi, psi, contacts, rt, 40.0 * KJ_PER_KCAL, 0.3)[0]
            fd = -(ep - em) / (2 * h)
            assert abs(f[atom, ax] - fd) <= 1e-4 * max(1.0, abs(fd)), (atom, ax, f[atom, ax], fd)


def test_dropping_the_anchor_term_changes_the_energy():
    pos, phi, psi, contacts = _peptide_like(seed=7)
    rt = _runtime(phi, psi, contacts, degree=2)
    e_full = _energy_forces(pos, phi, psi, contacts, rt, 40.0 * KJ_PER_KCAL, 0.3)[0]
    torsion_only = ResidualFit(np.zeros_like(rt.fit.coefficients), rt.fit.residual_mean, rt.fit.singular_values,
                               rt.fit.right_vectors, rt.fit.anchor_mean, rt.fit.anchor_std, rt.fit.anchor_clamp,
                               rt.fit.projection_mean, rt.fit.projection_std, 2)
    mut = PairModelRuntime(torsion_only, 2, rt.anchor_kind, rt.anchor_definition, rt.pair_sha256, rt.feature_atoms)
    assert abs(e_full - _energy_forces(pos, phi, psi, contacts, mut, 40.0 * KJ_PER_KCAL, 0.3)[0]) > 1e-3


def test_degree_two_anchor_is_clamped_to_the_training_range():
    """Beyond the clamp the chain factor stops growing, so the restraint force is bounded."""
    pos, phi, psi, contacts = _peptide_like(seed=8)
    rt = _runtime(phi, psi, contacts, degree=2)
    z_num_far = residual_cv2_from_positions_nm(pos, rt, phi, psi, contacts, _Args())
    e, _, meta = _energy_forces(pos, phi, psi, contacts, rt, 40.0 * KJ_PER_KCAL, 0.3)
    assert meta["anchor_clamp"] == list(rt.fit.anchor_clamp)
    assert np.isfinite(z_num_far) and np.isfinite(e)


def test_a_topology_whose_torsions_differ_from_the_schema_is_refused():
    pos, phi, psi, contacts = _peptide_like(seed=9)
    rt = _runtime(phi, psi, contacts)
    swapped = [phi[1], phi[0]]                     # same width, permuted
    with pytest.raises(RuntimeError, match="feature schema"):
        _energy_forces(pos, swapped, psi, contacts, rt, 1.0, 0.0)


def test_a_production_run_with_a_different_contact_definition_is_refused():
    pos, phi, psi, contacts = _peptide_like(seed=10)
    rt = _runtime(phi, psi, contacts)
    class Other(_Args):
        contact_r0_a = 10.0
    system = openmm.System()
    for _ in range(len(pos)):
        system.addParticle(12.0)
    with pytest.raises(RuntimeError, match="anchor"):
        _add_residual_torsion_cv_force(openmm, system, phi, psi, contacts, rt, Other(), force_group=29)


def test_reconstructed_bias_equals_the_bias_the_context_actually_applied():
    """The load-bearing MBAR property: u_nk rebuilt from recorded (cv1, cv2) and the CSV's kcal
    stiffness must equal the umbrella energy OpenMM evaluated. A unit slip anywhere (a stray 4.184,
    a kJ CSV, a different norm) fails this, whereas the energy-vs-evaluator test above cannot see it."""
    from gareus.cv import nonlocal_contact_cv_from_positions_nm
    from gareus.query import reconstruct_bias_matrix
    pos, phi, psi, contacts = _peptide_like(seed=12)
    rt = _runtime(phi, psi, contacts, degree=1)
    k2_kcal, c2 = 30.0, -0.4
    e_kj, _, _ = _energy_forces(pos, phi, psi, contacts, rt, k2_kcal * KJ_PER_KCAL, c2)
    z1 = nonlocal_contact_cv_from_positions_nm(pos, contacts, _Args())
    z2 = residual_cv2_from_positions_nm(pos, rt, phi, psi, contacts, _Args())
    beta = 1.0 / (0.0083144626 * 300.0)                       # mol/kJ, as reconstruct_bias_matrix expects
    windows = [{"window_id": 0, "center1": z1, "k1": 0.0, "center2": c2, "k2": k2_kcal, "gamd_lambda": 0.0}]
    u = reconstruct_bias_matrix(np.array([z1]), np.array([z2]), windows, beta)
    assert abs(u[0, 0] / beta - e_kj) <= 1e-6 * max(1.0, e_kj)


def test_zero_stiffness_state_still_exposes_the_cv_value():
    pos, phi, psi, contacts = _peptide_like(seed=11)
    rt = _runtime(phi, psi, contacts)
    e, f, meta = _energy_forces(pos, phi, psi, contacts, rt, 0.0, 0.0)
    assert e == 0.0 and np.allclose(f, 0.0)
    assert np.isfinite(residual_cv2_from_positions_nm(pos, rt, phi, psi, contacts, _Args()))
    assert meta["mode"] == "residual-torsion-pc" and meta["pair_model_sha256"] == "f" * 64
    assert "_runtime" in meta and "runtime" not in meta        # JSON-safe by the underscore convention
```

- [ ] **Step 2: Run to verify they fail** — `ImportError`.

- [ ] **Step 3: Implement**

`cv.py`: aliases `"residual-torsion-pc"`, `"residual-pc"` → `"residual-torsion-pc"`; add to `secondary_cv_is_transition`; `secondary_cv_range` → `(-6.0, 6.0)`; add to the exclusion set at `:488`; numeric evaluator:

```python
def residual_cv2_from_positions_nm(positions_nm, runtime, phi_torsions, psi_torsions, contact_pairs, args) -> float:
    from .tica import backbone_dihedral_features
    from .cv_selection.models import evaluate_component
    feats = backbone_dihedral_features(np.asarray(positions_nm), phi_torsions, psi_torsions)
    a = nonlocal_contact_cv_from_positions_nm(positions_nm, contact_pairs, args)
    return float(evaluate_component(runtime.fit, runtime.j, feats[None, :], np.array([a]),
                                    clamp=(runtime.fit.degree == 2))[0])
```

and in `secondary_structure_score_from_positions_nm`, after the cache call: `if mode == "residual-torsion-pc": return residual_cv2_from_positions_nm(positions_nm, ss_info["_runtime"], ss_info["phi_torsions"], ss_info["psi_torsions"], ss_info["contact_pairs"], ss_info["_args"])`.

`production.py::_add_residual_torsion_cv_force`:

```python
def _add_residual_torsion_cv_force(openmm, system, phi_torsions, psi_torsions, contact_pairs,
                                   runtime, args, *, force_group):
    runtime.check_topology(phi_torsions, psi_torsions)      # RuntimeError("... feature schema ...")
    runtime.check_anchor(args, contact_pairs)               # RuntimeError("... anchor ...")
    fit, j = runtime.fit, runtime.j
    v = np.asarray(fit.right_vectors[j - 1], dtype=np.float64)
    n_phi, n_psi = len(phi_torsions), len(psi_torsions)
    cv = openmm.CustomCVForce("0"); names = []
    for fname, tors, w, trig in (("sum_sin_phi", phi_torsions, v[0:2*n_phi:2], "sin"),
                                 ("sum_cos_phi", phi_torsions, v[1:2*n_phi:2], "cos"),
                                 ("sum_sin_psi", psi_torsions, v[2*n_phi::2], "sin"),
                                 ("sum_cos_psi", psi_torsions, v[2*n_phi+1::2], "cos")):
        if len(tors):
            cv.addCollectiveVariable(fname, _add_weighted_trig_torsion_force(openmm, tors, w, trig))
            names.append(fname)
    # Private copy of the contact sum: OpenMM allows one parent per child Force, so the CV1
    # umbrella's CustomBondForce cannot be shared. Same expression as forces.py:81, verbatim.
    r0_nm = float(args.contact_r0_a) * 0.1; beta = float(args.contact_beta_a_inv) * 10.0
    csum = openmm.CustomBondForce(f"contact_weight*0.5*(1-tanh(0.5*{beta:.17g}*(r-{r0_nm:.17g})))")
    csum.addPerBondParameter("contact_weight")
    for pair in contact_pairs:
        csum.addBond(int(pair[0]), int(pair[1]), [float(pair[2]) if len(pair) > 2 else 1.0])
    cv.addCollectiveVariable("res_contacts", csum)
    norm = float(runtime.anchor_definition["norm"])    # check_anchor proved it equals the live denominator
    K0 = float(v @ (fit.coefficients[0] + fit.residual_mean) + fit.projection_mean[j - 1])
    K1 = float(v @ fit.coefficients[1]); K2 = float(v @ fit.coefficients[2])
    a_raw = f"((res_contacts/{norm:.17g}) - {fit.anchor_mean:.17g})/{fit.anchor_std:.17g}"
    lo, hi = fit.anchor_clamp
    a_expr = f"min({hi:.17g}, max({lo:.17g}, {a_raw}))" if fit.degree == 2 else a_raw
    z2 = (f"(({' + '.join(names)}) - {K0:.17g} - {K1:.17g}*({a_expr}) - {K2:.17g}*({a_expr})^2)"
          f"/{fit.projection_std[j - 1]:.17g}")
    cv.addGlobalParameter("ss_k", 0.0); cv.addGlobalParameter("ss0", 0.0)
    cv.setEnergyFunction(f"0.5*ss_k*({z2}-ss0)^2")          # ss_k is kJ/mol/CV^2: NO 4.184 here
    cv.setForceGroup(int(force_group)); system.addForce(cv)
    return {"enabled": True, "mode": "residual-torsion-pc", "pair_model_sha256": runtime.pair_sha256,
            "component_index": int(j), "degree": int(fit.degree), "anchor_clamp": list(fit.anchor_clamp),
            "range_min": -6.0, "range_max": 6.0, "force_group": int(force_group),
            "phi_torsions": [list(map(int, t)) for t in phi_torsions],
            "psi_torsions": [list(map(int, t)) for t in psi_torsions],
            "contact_pairs": [list(p) for p in contact_pairs], "linear_subcv_names": names,
            "_runtime": runtime, "_args": args}
```

Clamping applies to degree 2 only: the chain factor `(K1 + 2K2·a)/(σ_jσ_c)` grows without bound in `a` and the CLAUDE.md record of Pep-GaMD NaN-ing at boost onset makes an unbounded extra restraint on the contact coordinate a plausible repeat. Degree 1 is linear in `a`, no clamp.

Dispatch: in `add_secondary_structure_cv_force`, `if mode == "residual-torsion-pc": runtime = PairModelRuntime.load(args.secondary_cv_model, args.secondary_cv_candidate_set, args.secondary_cv_feature_schema); return _add_residual_torsion_cv_force(openmm, system, phi_torsions, psi_torsions, primary_cv_def["contact_pairs"], runtime, args, force_group=force_group)`; raise `RuntimeError` if `primary_cv_def is None`. `if mode == "auto": raise RuntimeError("cv2=auto must be resolved by the swarm stage before production")`.

- [ ] **Step 4: Run** — `python -m pytest -q tests/test_cv_selection_forces.py tests/test_secondary_cv_restraint_sign_convention.py tests/test_cv2_reprojection.py` → PASS.

- [ ] **Step 5: Commit** — `git commit -m "feat(cv-selection): residual-torsion-pc runtime force with full chain rule"`

---

### Task 7: Two-dimensional ladder design and loader support

**Files:**
- Modify: `gareus/swarm/ladder_design.py` — add `design_2d_layout`, `reweighted_cv2_centers`, `cv2_force_constants_per_gap`, `write_ladder_windows_2d_csv`; `write_csv_atomic` and `Path` are already imported (`:20-22`).
- Modify: `gareus/windows.py::load_explicit_2d_window_csv` (`:1236-1420`): today `_csv_first_present` treats `""` as absent, so a `k2 == 0` row with an empty centre flips `has_secondary=False`, and `:1366` raises `"mixes rows with and without secondary_cv_center"`; worse, `:1345-1347` only append to `secondary_centers`/`secondary_k_list` when `has_secondary`, so a mixed table returns arrays shorter than `centers` that `set_window` indexes by window. **Required change**, not conditional: decide `has_secondary` from the presence of a `secondary_cv_k_kcal_mol` column; when `k2 == 0` and the centre is empty, append `(0.0, 0.0)` so every array stays one-per-window.
- Depends on Task 6's `gareus/cv.py` edit (`secondary_cv_range("residual-torsion-pc") == (-6, 6)`); until then the loader rejects `center2 = -1.0` as out of `(0, 1)`.
- Test: `tests/test_swarm_ladder_design.py` (append), `tests/test_explicit_window_table_lambda.py` (append)

**Interfaces (produces):**

```python
def design_2d_layout(n1, n2, *, n_rungs, max_replicas) -> dict
    # max_replicas <= 0 -> ValueError("max_replicas must be set; the default 0 cannot size a 2-D ladder")
def reweighted_cv2_centers(z2, deltav_kj, temperature_k, lambdas, n_windows, *, lo_q=0.02, hi_q=0.98) -> dict
    # {"centers": ..., "per_rung_quantiles": {lam: (lo, hi)}, "per_rung_ess": {lam: ess}}
    # centres span the UNION of the lambda=0 and top-rung reweighted [lo_q, hi_q] intervals
def cv2_force_constants_per_gap(centers, temperature_k, *, overlap_sigma=1.5, k_min_kcal, k_max_kcal) -> list[float]
    # per centre: spacing = smaller adjacent gap; sigma = spacing/overlap_sigma; k = RT/sigma^2, clamped
def write_ladder_windows_2d_csv(path, rows, lambdas) -> Path
```

`reweighted_cv2_centers` reuses the shipped `_reweighted_sigma_and_ess`'s weights (`w ∝ exp(−βλΔV_max)`, `ladder_design.py:38`): the Pep-GaMD dihedral channel boosts exactly the torsion energies z2 is built from, so the λ>0 rungs sample a broader z2 than the unbiased swarm — centres placed on the λ=0 quantiles alone leave the top rungs outside the grid. When `design_lambda_ladder` marks `extrapolated_from_rung`, the same rungs' z2 quantiles are unsupported; record that flag in the layout.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_swarm_ladder_design.py (append)
import csv
import numpy as np
import pytest
from gareus.swarm.ladder_design import (cv2_force_constants_per_gap, design_2d_layout,
                                        reweighted_cv2_centers, write_ladder_windows_2d_csv)


def test_small_product_uses_a_joint_grid_and_respects_the_cap():
    lay = design_2d_layout(6, 4, n_rungs=4, max_replicas=128)
    assert lay["kind"] == "joint" and lay["spatial_states"] == 24 and 24 * 4 <= 128


def test_large_product_falls_back_to_sparse_axes_plus_bridge_and_patches():
    lay = design_2d_layout(12, 8, n_rungs=4, max_replicas=128)
    cells = lay["cells"]
    assert lay["kind"] == "sparse" and lay["spatial_states"] * 4 <= 128
    assert (None, None) in cells
    assert sum(1 for i1, i2 in cells if i2 is None and i1 is not None) == 12
    assert sum(1 for i1, i2 in cells if i1 is None and i2 is not None) == 8


def test_default_zero_replica_cap_is_refused_not_silently_zero():
    with pytest.raises(ValueError, match="max_replicas"):
        design_2d_layout(6, 4, n_rungs=4, max_replicas=0)


def test_cv2_centres_widen_with_the_boost_rungs():
    rng = np.random.default_rng(0)
    z2 = rng.normal(size=20000)
    dv = 20.0 * (z2 ** 2)                       # boost correlates with |z2|: top rung samples the tails
    out = reweighted_cv2_centers(z2, dv, 300.0, [0.0, 0.5, 1.0], 5)
    lo0, hi0 = out["per_rung_quantiles"][0.0]; lo1, hi1 = out["per_rung_quantiles"][1.0]
    assert lo1 <= lo0 and hi1 >= hi0
    assert out["centers"][0] <= lo0 and out["centers"][-1] >= hi0
    assert all(out["per_rung_ess"][lam] > 0 for lam in (0.0, 0.5, 1.0))


def test_per_gap_force_constants_are_stiffer_where_centres_are_closer():
    ks = cv2_force_constants_per_gap(np.array([-2.0, -1.0, -0.5, 0.5, 2.0]), 300.0, k_min_kcal=1.0, k_max_kcal=1000.0)
    assert ks[2] > ks[0] and len(ks) == 5


def test_2d_csv_round_trips_through_the_production_loader(tmp_path):
    from gareus.windows import load_explicit_2d_window_csv
    rows = [dict(center1=0.02, k1=800.0, center2=-1.0, k2=50.0),
            dict(center1=0.04, k1=800.0, center2=None, k2=0.0)]
    path = write_ladder_windows_2d_csv(tmp_path / "w.csv", rows, [0.0, 0.5, 1.0])
    with path.open() as fh:
        assert len(list(csv.DictReader(fh))) == 6

    class Args:
        secondary_cv = "residual-torsion-pc"; contact_k_kcal = None; secondary_cv_k_kcal = 50.0
    centers, ks, sec_c, sec_k, meta, _ = load_explicit_2d_window_csv(Args(), path)
    assert len(centers) == len(sec_c) == len(sec_k) == 6      # one-per-window, including the k2=0 rows
    assert sec_k[3] == 0.0
```

- [ ] **Step 2–5:** fail → implement (`design_2d_layout` as v0.1 plus the cap guard; `reweighted_cv2_centers` using `np.exp(-beta*lam*dv)` weights and weighted quantiles; per-gap `k2`) → loader change in `windows.py` → run `tests/test_swarm_ladder_design.py tests/test_explicit_window_table_lambda.py` → PASS → `git commit -m "feat(cv-selection): two-dimensional ladder with lambda-reweighted CV2 centres"`.

---

### Task 8: Wire selection into `analyze_swarm_stage` with a pair gate

**Files:**
- Modify: `gareus/swarm/analyze.py` — `_load_members` (`:59-98`; returns a **4-tuple** today, unpacked at `:220` — extend to 5 and update the unpack), `analyze_swarm_stage` (`:212-409`; local variable is `rows`, not `plan_rows`), `_write_sidecar` (`:107-139`); add `from gareus.cv import secondary_cv_mode` (not imported today).
- Modify: `gareus/swarm/gates.py` — `pair_gate`; `evaluate_gates(..., selection=None)`.
- Create: `tests/conftest.py` — the repo has **no conftest today** and `tests/test_swarm_analyze.py` uses plain helpers `_fake_round(out, …)` (`:6`) and `_args()` (`:28`), not fixtures. Move/extend those helpers into `conftest.py` as fixtures `synthetic_swarm(tmp_path, *, with_features, wide_anchor)` and `swarm_args(**overrides)`; note `_fake_round` returns `out/"swarm"/"round_000"` and `analyze_swarm_stage` takes **`out`** (the grandparent), not the round dir's parent.
- Test: `tests/test_swarm_analyze.py`, `tests/test_swarm_gates.py`

**Gate semantics (fixed by review):**

| `selection.status` | `fallback` | gate | ladder written |
|---|---|---|---|
| `pair` | any | ok | 2-D |
| `cv1_only` (anchor deployable, no component passed) | `cv1_only` | ok, `note` recorded | 1-D |
| `cv1_only` | `refuse` | fail | withheld |
| `no_deployable_anchor` | **any** | **fail** | **withheld** |

An undeployable anchor never proceeds: running the CV the selector just certified as resolving two windows is the exact failure the spec calls the binding constraint. There is no `cv1_only_undeployable` escape in this plan.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_swarm_analyze.py (append; fixtures from conftest.py)
import yaml
from gareus.swarm.analyze import analyze_swarm_stage


def test_auto_cv2_writes_a_pair_model_and_a_two_dimensional_ladder(synthetic_swarm, swarm_args):
    out = synthetic_swarm(with_features=True, wide_anchor=True)
    report = analyze_swarm_stage(out, swarm_args(secondary_cv="auto"))
    an = out / "swarm" / "analysis"
    assert (an / "cv_pair_model.json").exists() and (an / "cv_selection_report.json").exists()
    with (an / "windows_lambda_ladder.csv").open() as fh:
        header = fh.readline().strip().split(",")
    assert {"secondary_cv_center", "secondary_cv_k_kcal_mol"} <= set(header)
    assert report["cv_selection"]["status"] == "pair"
    sidecar = yaml.safe_load((an / "ladder_run_args.yaml").read_text())
    assert sidecar["cvs"] == {"cv1": "contacts", "cv2": "residual-torsion-pc"}
    assert sidecar["tica_switch_cv2"] is False
    for key in ("secondary_cv_model", "secondary_cv_candidate_set", "secondary_cv_feature_schema"):
        assert (an / sidecar[key].split("/")[-1]).exists()


def test_narrow_anchor_fails_the_pair_gate_whatever_the_fallback(synthetic_swarm, swarm_args):
    out = synthetic_swarm(with_features=True, wide_anchor=False)
    report = analyze_swarm_stage(out, swarm_args(secondary_cv="auto", cv_selection_fallback="cv1_only"))
    assert report["cv_selection"]["status"] == "no_deployable_anchor"
    assert report["status"] == "fail" and report["gate"]["gates"]["pair"]["ok"] is False
    assert not (out / "swarm" / "analysis" / "windows_lambda_ladder.csv").exists()


def test_a_crashed_member_without_features_does_not_abort_the_analysis(synthetic_swarm, swarm_args):
    out = synthetic_swarm(with_features=True, wide_anchor=True, crash_one_member=True)
    report = analyze_swarm_stage(out, swarm_args(secondary_cv="auto"))
    assert report["cv_selection"]["status"] == "pair"


def test_explicit_cv2_none_leaves_the_existing_path_untouched(synthetic_swarm, swarm_args):
    out = synthetic_swarm(with_features=False, wide_anchor=True)
    report = analyze_swarm_stage(out, swarm_args(secondary_cv="none"))
    assert "cv_selection" not in report


# tests/test_swarm_gates.py (append)
def test_pair_gate_semantics():
    from gareus.swarm.gates import pair_gate
    assert pair_gate({"status": "pair"}, fallback="refuse")["ok"]
    assert pair_gate({"status": "cv1_only"}, fallback="cv1_only")["ok"]
    assert not pair_gate({"status": "cv1_only"}, fallback="refuse")["ok"]
    assert not pair_gate({"status": "no_deployable_anchor"}, fallback="cv1_only")["ok"]
```

- [ ] **Step 2: Run to verify they fail** — fixtures missing / `KeyError: 'cv_selection'`.

- [ ] **Step 3: Implement**

`gates.py`:

```python
def pair_gate(selection, *, fallback):
    status = selection.get("status")
    ok = status == "pair" or (status == "cv1_only" and fallback == "cv1_only")
    reasons = [] if ok else [f"cv pair selection returned {status!r} with fallback={fallback!r}"]
    if status == "no_deployable_anchor":
        reasons = [f"the configured anchor is not deployable ({selection.get('anchor_reasons')}); "
                   "no fallback runs an anchor the selector rejected"]
    return {"ok": ok, "status": status, "fallback": fallback, "reasons": reasons}
```

`analyze.py`: `_load_members` also loads `torsion_features.npy` for members with `status == "ok"` into `ok_features: dict[int, np.ndarray]`; a member that has no feature file **and** `status == "ok"` raises (`RuntimeError("member … completed without torsion_features.npy")`); crashed members are already excluded by the existing `status != "ok"` filter at `:82-84`. In `analyze_swarm_stage`, after the envelope and `cv1_all`, when `secondary_cv_mode(args) == "auto"`:

```python
data = _swarm_dataset(ok_traces, ok_features, rows, discard, args)   # anchor = contacts from "cv1"
cfg = SelectionConfig(residual_degree=int(args.cv_selection_residual_degree),
                      max_nonlinear_r2=float(args.cv_selection_max_nonlinear_r2),
                      max_coupling_fraction=float(args.cv_selection_max_coupling_fraction),
                      k1_kcal_reference=float(args.contact_adaptive_max_k_kcal),     # the run's real CV1 ceiling (default 1000)
                      k2_kcal_reference=float(args.cv_selection_k2_reference_kcal),
                      min_windows_cv1=int(args.cv_selection_min_windows_cv1),
                      temperature_k=float(args.temperature_k))
sel = select_cv_pair(data, cfg, physical_system_sha256=file_digest(system_dir / "base_system.xml"),
                     training_rows_sha256=_features_digest(ok_features), library_versions=_library_versions(),
                     genpept_preset=str(getattr(args, "diversity_bank_preset", "broad")))
```

Write `cv_pair_model.json`, `cv_candidate_set.json`, `cv_feature_schema.json`, `cv_selection_report.json`. If `sel.status == "pair"`: `z2_all`, `dv = deltav_max_kj(v_pep, v_dih, env)`, `centres2 = reweighted_cv2_centers(z2_all, dv, T, ladder["lambdas"], n2)`, `ks2 = cv2_force_constants_per_gap(...)`, `layout = design_2d_layout(n_win, n2, n_rungs=len(ladder["lambdas"]), max_replicas=int(args.max_replicas))`, then **CV1 spacing check**: effective curvature `k1_i + coupling_curvature_kcal(fit, j, k2_i)` must keep `σ_w` within 10% of the design value or the report carries `cv1_width_shrink` and the gate warns. Rows from `layout["cells"]`; `write_ladder_windows_2d_csv`. Else 1-D path unchanged. `evaluate_gates(..., selection=selection_dict_or_None)`; sidecar gains `cvs`, three model paths, `tica_switch_cv2: false`.

- [ ] **Step 4: Run** — `python -m pytest -q tests/test_swarm_analyze.py tests/test_swarm_gates.py tests/test_swarm_epoch0_orchestration.py` → PASS.

- [ ] **Step 5: Commit** — `git commit -m "feat(cv-selection): select and freeze CV2 inside swarm analysis"`

---

### Task 9: CLI and config surface

**Files:**
- Modify: `gareus/cli.py` — `--cv2` choices (`:109`) gain `"auto", "residual-torsion-pc", "residual-pc"`; new flags below; `_resolve_cv_aliases` (`:826-843`) is **left as is** for `--cv1` (no `auto`); new `_validate_cv_selection_args(p, args)` — takes the **parser** so it can `p.error` (the neighbouring validators take `args` only and raise `ValueError`).
- Modify: `gareus/config.py` (`:88-92` routes `cvs.cv1/cv2`): add `cv_selection:` → `cv_selection_*` dests.
- Test: `tests/test_cv_selection_cli.py`

New flags: `--secondary-cv-model`, `--secondary-cv-candidate-set`, `--secondary-cv-feature-schema` (paths), `--cv-selection-residual-degree {1,2}=1`, `--cv-selection-max-nonlinear-r2=0.20`, `--cv-selection-max-coupling-fraction=0.25`, `--cv-selection-k2-reference-kcal=50.0`, `--cv-selection-min-windows-cv1=4`, `--cv-selection-fallback {cv1_only,refuse}=cv1_only`, `--swarm-n-windows-cv2=4`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_cv_selection_cli.py
import pytest
from gareus.cli import parse_args


def test_auto_cv2_parses_with_documented_defaults_and_forces_the_tica_switch_off():
    a = parse_args(["--seq", "GYDPETGTWG", "--cv2", "auto", "--window-mode", "adaptive-production",
                    "--tica-switch-cv2"])
    assert a.secondary_cv == "auto" and a.primary_cv == "nonlocal-contacts"
    assert a.cv_selection_residual_degree == 1 and a.cv_selection_max_coupling_fraction == 0.25
    assert a.tica_switch_cv2 is False


def test_cv1_auto_is_not_accepted_in_this_release():
    with pytest.raises(SystemExit):
        parse_args(["--seq", "GYDPETGTWG", "--cv1", "auto"])


def test_auto_cv2_cannot_reach_a_manual_production_run():
    with pytest.raises(SystemExit):
        parse_args(["--seq", "GYDPETGTWG", "--cv2", "auto", "--window-mode", "manual"])


def test_residual_mode_in_production_requires_all_three_model_paths():
    with pytest.raises(SystemExit):
        parse_args(["--seq", "GYDPETGTWG", "--cv2", "residual-torsion-pc", "--window-mode", "manual",
                    "--secondary-cv-model", "m.json"])
    a = parse_args(["--seq", "GYDPETGTWG", "--cv2", "residual-torsion-pc", "--window-mode", "manual",
                    "--secondary-cv-model", "m.json", "--secondary-cv-candidate-set", "c.json",
                    "--secondary-cv-feature-schema", "f.json", "--windows-2d-csv", "w.csv"])
    assert a.secondary_cv == "residual-torsion-pc"


def test_yaml_cv_selection_block_reaches_args(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("schema_version: '2.0'\nsequence: {seq: GYDPETGTWG}\n"
                   "cvs: {cv1: contacts, cv2: auto}\nwindows: {window_mode: adaptive-production}\n"
                   "cv_selection: {residual_degree: 2, max_nonlinear_r2: 0.1, fallback: refuse}\n")
    a = parse_args(["--config", str(cfg)])
    assert a.cv_selection_residual_degree == 2 and a.cv_selection_fallback == "refuse"
```

- [ ] **Step 2–5:** fail → implement → run `tests/test_cv_selection_cli.py tests/test_config_profiles.py` → PASS → `git commit -m "feat(cv-selection): cv2 auto and residual-torsion-pc CLI surface"`.

---

### Task 10: Provenance and resume safety

**Files:**
- Modify: `gareus/provenance.py::_method_settings(args)` (`:270`): record `cv_pair_model_sha256` (read the `sha256` field of `args.secondary_cv_model` via `gareus.correctness._io.json_loads`).
- Modify: `gareus/production.py::reconcile_resume_secondary_cv_metadata(secondary_cv_metadata, secondary_cv_centers, *, current_pair_sha256=None)` (`:5742`) and its call site (`:1088`).
- Test: `tests/test_cv_selection_resume.py` — as v0.1 (three tests: refuse changed digest, accept same, manifest records digest).

- [ ] Steps 1–5 as v0.1; commit `feat(cv-selection): record and enforce the frozen pair digest across resume`.

---

### Task 11: End-to-end dry run, example config, task log

**Files:**
- Create: `tests/test_cv_selection_e2e_dry.py` — synthetic swarm (Task 8 fixture) → `analyze_swarm_stage(out, …)` → sidecar → `parse_args([...sidecar values...])`; asserts `secondary_cv == "residual-torsion-pc"`, `tica_switch_cv2 is False`, and that `PairModelRuntime.load(...)` on the sidecar's three paths succeeds and `check_topology` passes against the fixture's torsion index.
- Create: `examples/chignolin_auto_cv2.yaml` — copy of `smoke/config/chignolin_8.yaml` with `cvs: {cv1: contacts, cv2: auto}` and a `cv_selection:` block with the defaults above.
- Modify: `docs/atlas-md/developer/equilibrium-cv-task-log.md` (one entry per task), `mkdocs.yml` (nav: this plan).

- [ ] Steps as v0.1; commit `test(cv-selection): swarm-to-production dry run and example config`.

---

## Verification round (v0.1 → v0.2)

Two independent fresh-context agents (code-fact check; scientific/design attack) and a three-judge panel reviewed v0.1. Every item below was reproduced before being acted on; the two decisive ones were re-run by the author independently of the agents.

| # | Finding (v0.1) | Reproduced by | Resolution in v0.2 |
|---|---|---|---|
| 1 | `gain = min(L1,L2) − L12` is noise where it must discriminate; the plan's own Task 4 fixture gave **−0.0027** against an assertion of `> 0.3` (fixed rule: **+0.6856**) | author probe | Task 4: `gain = L1 − L12`; joint-bin capacity matched; folds shuffled |
| 2 | Cells were seed-level `plan.csv` strata (constant per member ⇒ n ≈ 60; CV1 is a stratification axis ⇒ leakage) | design agent | Task 4 `frame_partition` on per-frame torsion+shape features, anchor excluded; Task 5 uses it |
| 3 | `ss_k` claimed kcal with `set_window` converting; in fact `production.py:6129` converts and `set_window` writes kJ verbatim. The test's `×4.184` steered toward a 4.184× over-stiff CV2 with no diagnostic | author probe + both agents | Global constraint; Task 6 test sets `ss_k` in kJ; prose corrected |
| 4 | Auto CV1 could select Rg, which nothing forces and which `_resolve_cv_aliases` collapses to `distance`; the swarm builds the CV1 umbrella from contacts unconditionally | both agents | **Auto CV1 removed from this plan**; Task 3 is a deployability diagnostic; `DEPLOYABLE_ANCHOR_KINDS` |
| 5 | Resolvable-window count is unit-dependent (Rg in Å resolves 10× more windows than in nm; `--contact-normalize false` flips the verdict); vacuous for standardised z2 | design agent | Task 3 adds unit-invariant `dynamic_range`; CV2 window-count criterion dropped |
| 6 | CV2→CV1 coupling curvature `k2(v·B1)²/(σ_j²σ_c²)` varies ~650× across candidates and was never looked at; can reach a quarter of `k1_max` | design agent | Task 2 `coupling_curvature_kcal`; Task 5 deployability criterion + certificate; Task 8 CV1-width check |
| 7 | CV2 centres from unbiased swarm quantiles while λ>0 rungs sample a boosted, broader z2 | design agent | Task 7 `reweighted_cv2_centers` on the λ=0 ∪ top-rung reweighted interval, ESS recorded |
| 8 | Two contradictory Task 8 tests for `no_deployable_anchor`; fallback ran the CV the selector rejected | design agent | Gate table above: undeployable anchor fails unconditionally |
| 9 | `PairModel` bound by width only: permuted torsions or a different contact definition deploy silently | design agent | Task 6 `check_topology`, `check_anchor`; `norm`, `pair_rule`, `pair_list_sha256` in the anchor definition |
| 10 | Blindness enforced by a kind string; contact pair list unconstrained; GENPEPT preset unchecked | design agent | Task 5 refuses non-`broad` presets; pair rule + list digest stored and re-derived |
| 11 | `R² ≤ 0.20` point-estimate gate with swarm-to-swarm sd ≈ 0.064 | design agent | Task 4 returns per-fold R²; Task 5 gates on mean + 2·SE; half-split stability recorded |
| 12 | Joint table smoothed harder than marginals; folds confounded with cell order | design agent | Task 4 |
| 13 | Degree-2 chain factor unbounded in `a` against a documented NaN history | design agent | Task 2 `anchor_clamp`; Task 6 clamps in the expression and the evaluator |
| 14 | `np.save` appends `.npy` to a `.npy.tmp` name → rename fails, feature file never written; `run_member` calls `_run_loop_recording_failure`, not `run_member_loop`; missing imports | code agent | Task 1 |
| 15 | `ss_info["runtime"]`/`["args"]` would `TypeError` in `_json_ready`; `primary_cv_def` not in scope; `contact_normalization_denominator` not imported in production | code agent | Task 6 underscore keys; signature + two call sites; import |
| 16 | 2-D loader rejects an empty centre and returns length-desynced arrays | code agent | Task 7 required loader change |
| 17 | Fixtures named in Task 8 do not exist; no `conftest.py`; wrong path arithmetic; `_load_members` 4-tuple; `rows` not `plan_rows`; `secondary_cv_mode` not imported; "refuse if any member lacks features" would abort on any crashed member | code agent | Task 8 |
| 18 | `max_replicas` default is `0` → `cap = 0`; CV1 ceiling attr is `contact_adaptive_max_k_kcal` (default 1000), not `cv1_k_max`/1200 | code agent | Tasks 7, 8 |
| 19 | `obs["cv1"]` does not exist (`primary_cv`); `Cv2Model` wraps a `TICAResult` | code agent | Task 6 |
| 20 | `secondary_cv == "auto"` reaching production builds a `phi0=psi0=0` force silently | code agent | Global constraint; Task 6 raises; Task 9 `p.error` |
| 21 | Weighted-vs-unweighted covariance conventions and inconsistent thresholds (1e-10 / 1e-8) | code agent | Global constraint 1e-8 weighted; Task 2 weighted test |

**Accepted as sound by both agents and the author:** MBAR bias reconstruction stays exact (bias is an exact function of the recorded `(c, z2)`, both forces implement exactly that function, both CVs recorded in every state); residualising `sin`/`cos` independently does not corrupt the coordinate (the construction is equivariant under per-torsion phase rotations — verified numerically to `|corr| = 1.0`); torsion sign convention via `_add_weighted_trig_torsion_force`; no catastrophic cancellation in the OpenMM expression (`|v| = 1`, `|φ| ≤ 1`, `K0 = O(1)`); freezing the digest and refusing a changed one on resume.

**Small board (kimi, mini, thinker; chair glm), on v0.1:** split **REJECT 2–1** after one debate round (kimi REJECT 80, thinker REJECT 90, mini ACCEPT-WITH-CHANGES 88; aggregate 83). Its five majority defects are items 1, 3, 4, 8 and 2 above and were already reproduced and resolved in v0.2. Of its seven resubmission conditions: (1) single kcal→kJ convention plus a test that reconstructed `u_nk` equals the applied bias — **adopted**, Task 6 `test_reconstructed_bias_equals_the_bias_the_context_actually_applied`; (2) deployable anchor dictionary — **adopted** (auto-CV1 removed; the `"contacts"` literal in the CSV writer is now correct by construction); (3) `L1 − L12` plus a guard for the all-gains-≈-0 regime — **adopted**, Task 5 `min_gain_nats`; (4) gate/fallback contract — **adopted and stricter** (an undeployable anchor fails unconditionally); (5) anchor excluded from the partition features — **adopted**; (7) load-time anchor-definition check — **adopted**, Task 6 `check_anchor`. Condition (6), "replace the exact-equality JSON round-trip assertion, decimal serialization cannot satisfy it", is **refuted by measurement**: the canonical encoder is Python `json` with shortest-repr floats, and 10⁵ random float64 values spanning 30 orders of magnitude round-trip `np.array_equal`-exact through `json_bytes → json_loads`. Task 2 keeps the exact assertion; the dissent's premise was wrong, not the contract. Full judgment: `docs/atlas-md/developer/reviews/auto-cv-pair-plan-small-board.md`.

**Still open, deliberately:** connectivity of a sparse layout is asserted, not predicted (the k2=0 axis windows sample the *unbiased* conditional and may not overlap distant axis-2 windows) — the shipped end-of-campaign connected-components check will fail such a run; predicting overlap from the swarm's joint density before writing the CSV belongs with the joint-pair plan. Codex co-review did not run (workspace out of credits); its leg is absent, not merged.
