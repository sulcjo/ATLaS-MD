# Contact-map tICA as CV1 (generic-anchor phase 2) -- design spec

Date: 2026-10-01. Status: proposed, not built. Builds on
`2026-10-01-generic-cv1-anchor.md` (phase 1, merged in PR #127: `cv_selection/anchor_spec.py`,
end-to-end distance anchor) and the CV2 selection rules of PRs #127/#128 (robust gain floor,
breadth tie-set then slowest).

## 1. What and why

**The coordinate.** For every residue pair (i, j) with |i - j| >= 3 (28 pairs for chignolin),
a smooth contact count

    s_ij = sum over heavy atoms a in i, b in j of  0.5 (1 - tanh(0.5 beta (r_ab - r0)))

(the existing contact switch; r0 = 4.5 A, beta = 6 /A by default, see section 7). tICA on the
per-frame vector s (lag within swarm members) gives the slow linear combinations; CV1 is the
selected one, standardised:

    c = (w . s - mu) / sigma = sum over heavy pairs p of  (w_ij(p) / sigma) * switch(r_p)  -  mu / sigma

**Why.** On chignolin's swarm (2026-10-01 pair search, 4,872 PDB frames, `~/.claude/jobs/6bfa4c5c/
tmp/modality/pair_search_cmap.json`) the min-distance version of contact-map tIC1 was the best
single coordinate: held-out CA-cluster information 0.25 +- 0.03 nats (e2e 0.13, contact
fraction 0.01) and slow (rho 0.974 at 200 ps; e2e 0.74, contacts 0.80); paired with torsion
tIC1 it was the only pair top-tier on CA coverage, backbone-basin coverage and slowness. Those
numbers are for a NON-smooth definition and must be re-measured on the definition above
(section 6, gate G0) before any of this ships.

**Key simplification.** `c` is a weighted switch sum over heavy-atom pairs plus a constant --
exactly the form of today's contact CV1 (`forces.add_contact_umbrella_force`: one
`CustomBondForce` with per-bond `contact_weight`, divided by `contact_norm`) with fitted,
signed weights, a free normalisation and an offset. Most of the production contact path is
reused; the new pieces are the feature recording, the fit, the frozen CV1 artifact and the
anchor kind.

## 2. Swarm: record the contact map

- `swarm/members.py` `measure_frame`: additionally compute s (n_pairs) per trace row and write
  `member_NNNN/contact_map_features.npy` (n_rows x n_pairs, float32) + `contact_map_index.json`
  (residue pairs, heavy-atom pair list digest, switch parameters), aligned with `trace.csv` row
  by row exactly like `torsion_features.npy`.
- Cost: heavy pairs between residues >= 3 apart (chignolin: ~1,200), one vectorised distance
  evaluation per row; negligible next to the MD step.
- Old swarms (chignolin_9 round 0, 1 ns): no recorded map. Fallback: compute s from the
  member's `frames/frame_*.pdb` (one per 20 ps, 4,872 rows after discard) and fit/design on
  those rows only, with a report warning (`contact_map_source: pdb_frames`). The 2026-10-01
  5 ns chignolin_10 swarm should record the map natively (section 8).

## 3. Fit and freeze CV1 (swarm analysis)

New module `gareus/cv_selection/contact_map_cv1.py`, called from `swarm/analyze.py` before the
CV1 ladder when the config asks for it (`cv1: contact-map-tica`):

1. Features S (n x 28), design weights = the selection's balanced frame cells (same measure as
   CV2 selection), member ids + frame index for lag pairs (never across members or gaps).
2. tICA at `--cv1-tica-lag-ps` (default 50 ps), symmetrised, regularised (ridge 1e-6 x trace),
   up to `--cv1-n-tica` modes (default 3). Sign: largest |weight| positive (deterministic).
3. Gates per mode (same machinery as CV2): slowness at fixed lag (rho >= `--cv-selection-min-
   slowness`), reproducibility (each seed-family half refitted from scratch contains a mode
   with |r| >= 0.8), anchor resolvability (`anchor.score_anchor`: >= `min_windows_cv1`
   resolvable windows at the CV1 k bounds).
4. Pick: slowest reproducible mode (CV1 has no breadth floor in this spec; see section 7).
   None passes -> fall back to the configured fallback CV1 (`--cv1-fallback contacts|distance`,
   default contacts) and record why.
5. Freeze `swarm/analysis/cv1_model.json` (schema `cv1_contact_map_v1`): residue pairs, the
   full heavy-atom pair list with per-pair weights w_ij(p) / sigma, offset mu / sigma, switch
   r0/beta, lag, eigenvalue, half-split correlations, training-rows digest, topology digest,
   and its own sha256. Weights are float64 verbatim (same "JSON holds float64" rule as the
   candidate set).

## 4. Anchor kind `contact-map-component` (anchor_spec)

| field | value |
|---|---|
| pairs | the model's heavy-atom pairs, weight = w_ij / sigma |
| f(r) | the contact switch (identical expression to the contact kind) |
| norm | 1 (weights already carry 1/sigma); the offset mu/sigma is a constant |
| units | dimensionless (standardised: swarm sd 1) |
| trace values | S @ w, standardised (or from PDB frames on old swarms) |
| k bounds | `--cv1-k-min/max` in kcal/mol per unit^2; unset -> 0.5 / 200 (sigma_w 1.1 .. 0.055 at 300 K) |
| value bounds | none (signed coordinate) |
| binding | `anchor_binding_sha256` = digest of the cv1_model.json sha256 |

The offset: `c = sub_cv - mu/sigma`. Rather than threading an offset through every
consumer, window centres and samples are expressed in sub-CV units (`c_raw = c + mu/sigma`);
the standardised value is only used for reporting. The residual CV2 then has the existing form
`c = sub_cv / norm` with norm = 1 and needs no change.

## 5. Production

- CLI: `--cv1 contact-map` + `--cv1-model PATH` (frozen artifact; `cv1: contact-map-tica` in
  the swarm stage writes the path into the epoch-0 sidecar). `primary_cv_mode` gains
  `contact-map`.
- `cv.prepare_primary_cv_definition`: mode `contact-map` -> load the model, verify topology +
  pair list against the run's topology (refuse on mismatch), return `contact_pairs` = the
  weighted list and `contact_norm` = 1. `forces.add_contact_umbrella_force` is reused as is
  (signed weights are valid per-bond parameters); `contact_normalization_denominator` must
  return the model's norm for this mode (today it derives the denominator from the pair list).
- Everywhere that branches on `primary_cv_is_contacts` (seeding pull ramps, preflight spacing,
  start-quality gate, observe fast path, shared CV force layout, positions evaluator
  `nonlocal_contact_cv_from_positions_nm`, MBAR loaders' CV1 units) must treat `contact-map`
  as contact-like but with its own norm and no [0, 1] range. Each of those sites gets a test.
- The shared contact layout (CV1 umbrella carried on the residual CV2 force) works unchanged:
  the anchor sub-CV is the same weighted sum.
- Resume / manifest: the model path + sha256 in `run_manifest.method_settings` and the
  checkpoint metadata; a resume refuses a different model.

## 6. Validation gates (in order; stop at the first failure)

- **G0 (before building production code).** Re-run the 2026-10-01 pair search with the smooth
  definition of section 1 on the chignolin swarm. Proceed only if contact-map tIC1 keeps
  CA-cluster coverage >= e2e + 0.05 nats and rho(200 ps) >= 0.95, i.e. the smooth version is
  still the best single CV1. Otherwise stop: the min-distance result did not transfer.
- **G1 (fit).** On the chignolin swarm: selected mode reproduced in both seed halves (|r| >=
  0.8), sign deterministic, artifact digest stable across two runs.
- **G2 (force).** Force-vs-evaluator oracle (energy and forces by finite differences, Reference
  platform) for the CV1 umbrella and for the residual CV2 anchored on it; shared-layout
  energies equal split-layout energies (as `tests/test_shared_contact_cv_force.py`).
- **G3 (replay).** Swarm analyze replay with `cv1: contact-map-tica`: CV1 ladder, CV2 pick
  (tie-set rule), shape layout and reserve; report the layout next to the contacts and e2e
  replays.
- **G4 (no regression).** Contacts and e2e paths byte-identical (windows CSV, layout, pair
  model, candidate set) on the same replay.

## 7. Decisions (user, 2026-10-01) and remaining open items

- **CV1 breadth: yes.** CV1 candidates are ranked like CV2: robust breadth tie-set, then the
  slowest. Breadth for CV1 = held-out information about a conformational partition that does not
  use the candidate itself (CA-geometry clusters + per-residue backbone basins, as the 2026-10-01
  pair search), fold-averaged over seed-family assignments.
- **Fallback: contacts** when no CV1 candidate passes.
- **`cv1: auto` across all kinds: yes.** Candidates = contacts, end-to-end distance and the
  contact-map tICA modes; each passes the same CV1 gates (resolvable windows, slowness,
  seed-half reproducibility, breadth floor); the pick is the breadth tie-set then the slowest;
  CV2 is then selected against the chosen CV1 by the existing rules.
- **Contact definition (calibrated 2026-10-01 on the chignolin swarm, 4,872 PDB frames; G0
  PASSED; script `~/.claude/jobs/6bfa4c5c/tmp/modality/cmap_calib2.py`):** per residue pair the
  **soft-min heavy-atom distance** d_ij = -lambda ln sum_ab exp(-r_ab/lambda) (lambda 0.2 A),
  through the rational switch (1 - (d/r0)^6)/(1 - (d/r0)^12), **r0 4.5 A**, residue pairs with
  **|i - j| >= 3** (28 for chignolin). This REPLACES the sum-of-atom-switches definition of
  section 1. Measured tIC1 (mean +- sd over seed-family fold assignments):

  | per-pair contact | sep | CA cov. | basin cov. | rho 200 ps | + torsion tIC1 CA / basin / VAMP-2 |
  |---|---|---|---|---|---|
  | soft-min, rational r0 4.5 A | >= 3 | 0.261 +- 0.024 | 0.31 +- 0.14 | 0.975 | 0.30 / 0.86 / 1.94 |
  | soft-min | >= 2 | 0.248 +- 0.033 | 0.32 +- 0.13 | 0.976 | 0.30 / 0.85 / 1.94 |
  | min distance (not smooth) | >= 3 | 0.248 +- 0.029 | 0.29 +- 0.14 | 0.974 | 0.29 / 0.89 / 1.93 |
  | sum of atom switches r0 4.5 A | >= 3 | 0.102 +- 0.025 | 0.33 +- 0.05 | 0.925 | 0.30 / 0.78 / 1.85 |
  | sum of atom switches r0 12 A | >= 3 | -0.004 +- 0.032 | -0.14 +- 0.08 | 0.915 | 0.13 / 0.53 / 1.81 |
  | e2e (baseline) | -- | 0.134 | -0.16 | 0.744 | 0.36 / 0.38 / 1.55 |
  | contacts (baseline) | -- | 0.009 | -0.19 | 0.802 | 0.22 / 0.30 / 1.61 |

  Separation 2 vs 3 is within noise; 3 uses fewer pairs. tIC1 = turn formation around residues
  2-7 (top correlations D3-T6, Y2-T6, Y2-E5, D3-G7, ~0.75-0.85), not the Y2-W9 register.
  **Implementation risk:** in OpenMM the soft-min needs per residue pair a sub-CV
  sum exp(-r/lambda); at lambda 0.02 nm far pairs underflow single precision (log 0). Use a
  per-pair reference offset exp(-(r - r_ref)/lambda) or a softer lambda (0.5 A, re-score), and
  make G2 cover far-apart pairs. The CV1 force is then a CustomCVForce over 28 sub-CVs (not
  the single weighted CustomBondForce of section 1), and the residual-CV2 anchor sub-CV is the
  same composite: anchor_spec's "c = sub_cv / norm" generalises to "c = f(sub_cvs)" for this kind.

### Original open items (for the record)


1. Contact definition: |i - j| >= 3 (28 pairs) or >= 2; heavy-atom switch r0 4.5 A / beta 6
   (local contacts) vs the CV1 contact scheme's r0 12 A / beta 3 (long-range); per-pair sum vs
   per-pair mean (sum weights large residues more).
2. CV1 pick rule: slowest reproducible mode only, or also a breadth criterion (the CV2 tie-set
   rule needs a breadth measure for CV1 that does not use CV1 itself; the CA-cluster
   information of the pair search is one candidate).
3. Fallback when no mode passes: contacts (default) or e2e.
4. Whether `cv1: auto` (pick among contacts / e2e / contact-map by the same gates) is in scope
   now or later.

## 8. Sequencing and cost

| step | content | depends on |
|---|---|---|
| 1 | G0 smooth-definition pair search (read-only script) | -- |
| 2 | swarm feature recording (`contact_map_features.npy`) | G0 pass |
| 3 | `contact_map_cv1.py` fit + artifact + analyze wiring + G1 | 2 |
| 4 | anchor kind + production mode + all contact-like call sites + G2 | 3 |
| 5 | G3/G4 replays, docs, PR | 4 |

Rough size: comparable to phase 1 plus the fit module (~10-14 functions, 8-10 files, ~2-3 new
test files). Step 2 is small and independent: landing it before chignolin_10's 5 ns swarm runs
means that swarm records the contact map natively, so the fit later uses all 48k+ rows instead
of the 20 ps PDB frames. Recommended order: G0 now, then step 2 before launching chignolin_10.
