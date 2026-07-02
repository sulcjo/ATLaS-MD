# Bootstrap Torsion-Linear CV2 Design

## Goal

Replace `rama-map` as the first-epoch secondary CV for peptide runs.

The replacement must be usable before MD samples exist, preserve per-residue backbone torsion information, remain differentiable in OpenMM, and hand off cleanly to the existing adaptive `tica-linear` CV after epoch 0.

## Problem

`rama-map` compresses many distinct backbone states into one scalar:

- phi and psi are averaged into one basin score.
- all residues are averaged together.
- basin labels are mapped onto an arbitrary scalar order.
- intermediate scalar values are not physical transition states.
- different torsion patterns can produce the same CV2 value.

This makes `rama-map` poor as a generic orthogonal partner to CV1 contacts.

## Recommended Approach

Add a first-epoch bootstrap mode:

```yaml
cvs:
  cv1: contacts
  cv2: torsion-pca

bootstrap_torsion_cv:
  source: seeds
  residualize_against_cv1: true
  component: 1
  min_seed_count: 20
```

`torsion-pca` computes backbone circular features from the GENPEPT seed ensemble:

```text
X = [sin(phi_i), cos(phi_i), sin(psi_i), cos(psi_i)] for each residue torsion
```

It fits PC1 in this feature space, optionally after removing linear dependence on seed CV1/contact fraction. The resulting scalar is:

```text
CV2 = (X - mean) @ weights
```

The fitted model is stored in the same JSON schema used by `TICAResult`, with metadata marking it as PCA/bootstrap rather than kinetic tICA. The existing OpenMM `tica-linear` force-builder path can then apply the linear torsion CV with minimal new force code.

## Why This Approach

This keeps the useful part of the current adaptive design: epoch 0 has a cheap broad backbone CV, and later epochs switch to a kinetic slow-mode CV. It removes the lossy Rama averaging that made epoch 0 weak.

Benefits:

- keeps residue identity;
- keeps phi and psi separate;
- handles angle periodicity with sin/cos features;
- is differentiable through existing torsion sub-CVs;
- can be made less redundant with CV1 by residualizing features against contact fraction;
- works before production MD exists;
- uses the same linear-CV machinery as `tica-linear`;
- supports permanent `tica_switch_cv2` after epoch 0.

## Alternatives Considered

### Motif Torsion-Path CV

Define a path variable between known torsion motifs such as extended, turn, hairpin, and folded-like states. This is interpretable for peptides with known structural motifs, especially beta-hairpins, but it is less generic and needs curated motif definitions.

### Backbone H-Bond Or Turn-Closure CV

Use hydrogen-bond, turn-closure, or beta-hairpin geometry as CV2. This is physically clear for hairpin systems, but not universal for arbitrary short peptides and can become too coupled to sidechain contact formation.

### Keep Rama-Map Until tICA

Least code churn, but keeps the known failure: distinct torsion patterns alias into one scalar before the tICA switch.

## Architecture

### Config

Add canonical secondary CV alias:

```text
torsion-pca
```

Accepted aliases:

```text
bootstrap-torsion
bootstrap-linear
torsion-linear
```

Add optional config group flattened into argparse/config namespace:

```yaml
bootstrap_torsion_cv:
  source: seeds
  residualize_against_cv1: true
  component: 1
  min_seed_count: 20
  state_file: ""
```

Defaults:

- `source: seeds`
- `residualize_against_cv1: true`
- `component: 1`
- `min_seed_count: 20`
- `state_file: <out>/tica/bootstrap_torsion_cv.json`

No existing default changes. Template configs switch from `rama-map` to `torsion-pca` only where the template is explicitly for the new two-stage CV workflow.

### Model Fitting

Create a pure NumPy fitter in `gareus/tica.py` or a small adjacent module:

```text
compute_bootstrap_torsion_pca(X, cv1=None, residualize=True, component=1) -> TICAResult
```

Steps:

1. collect seed conformer positions;
2. build phi/psi torsion feature matrix using `backbone_dihedral_features`;
3. compute seed CV1 contact fraction when `residualize_against_cv1` is true;
4. regress each feature column on `[1, cv1]`;
5. fit PCA/SVD on residualized features;
6. choose requested component;
7. normalize sign deterministically;
8. store result as `TICAResult`-compatible JSON.

Set `eigenvalue` to explained variance ratio. Set `lag` to `0` for bootstrap PCA and add metadata `method: pca`. Any downstream consumer that assumes positive lag must branch on `method: pca`.

### Force Construction

`torsion-pca` must build the same OpenMM force form as `tica-linear`:

```text
0.5 * ss_k * (((sum_j weight_j * feature_j) + offset) - ss0)^2
```

Implementation must avoid duplicating force construction. Required layout:

- refactor current `tica-linear` force construction into helper:
  `add_linear_torsion_cv_force(...)`;
- call helper for both `tica-linear` and `torsion-pca`;
- keep metadata `mode` as selected mode, while storing `linear_cv_kind`.

### Adaptive Handoff

Recommended YAML:

```yaml
cvs:
  cv1: contacts
  cv2: torsion-pca

tica:
  tica_obs_interval: 50
  tica_lag_frames: 50
  tica_update_after_epochs: [0]
  tica_switch_cv2: true
  tica_linear_k_min: 5.0
  tica_linear_k_max: 50.0
```

Epoch behavior:

- epoch 0: CV2 is bootstrap torsion PCA;
- after epoch 0: fit tICA from trajectory observations;
- epoch 1 onward: switch permanently to `tica-linear`;
- resume restores the switch from epoch summary metadata.

### Analysis

Analysis must label bootstrap CV2 separately from kinetic tICA:

- axis label: `Bootstrap torsion PC1`;
- metadata mode: `torsion-pca`;
- state file records `method: pca`;
- MBAR versioning treats `torsion-pca` and `tica-linear` as different CV definitions.

This prevents mixing epoch 0 bootstrap samples with post-switch tICA samples as if they used the same CV.

## Error Handling

Fail closed when:

- seed conformer directory is missing;
- fewer than `min_seed_count` usable conformers are available;
- topology torsion count does not match stored state feature count;
- PCA variance is numerically zero;
- residualization leaves rank-deficient all-zero features;
- `torsion-pca` is selected but no bootstrap state can be built or loaded.

Fallback to `rama-map` must not happen silently. If user wants fallback, it must be explicit in config.

## Testing

Focused tests:

- `secondary_cv_mode("torsion-pca") == "torsion-pca"`;
- bootstrap PCA returns weights with expected feature dimension;
- residualizing against CV1 reduces linear correlation with CV1 on synthetic data;
- state JSON loads through the shared linear-CV path;
- OpenMM-free metadata tests verify mode, range, weights, offset, torsion counts;
- resume/switch test verifies `torsion-pca -> tica-linear` transition;
- analysis guard test verifies mixed CV definitions are not pooled silently.

Verification commands:

```bash
pytest -q tests/test_tica_cv_mode.py
pytest -q tests/test_bootstrap_torsion_cv.py
python -m py_compile gareus/cv.py gareus/tica.py gareus/production.py
git diff --check
```

## Out Of Scope

- changing CV1 contact definition;
- making Gibbs exchange default;
- replacing tICA fitting;
- adding motif-specific hairpin CVs;
- changing GaMD boost scope;
- reprocessing existing runs.
